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

// The GGUF arm of this seam. `ModelSource` (hipfire-runtime) is matched
// exhaustively by dozens of arch carriers, and no carrier can run GGUF weights
// yet (no ggml kernel is wired), so GGUF is not a `ModelSource` variant:
// `crate::gguf_native::GgufNativeModel::source_meta` (gated by
// `HIPFIRE_GGUF_NATIVE=1`, refusing `load refused:` otherwise) produces the
// same three facts this struct carries. `arch_id` is in the HFQ header
// namespace (`hipfire_runtime::arch_mapping`, from `general.architecture`),
// not the `derive_arch_id` directory namespace.
impl From<crate::gguf_native::GgufSourceMeta> for SourceMeta {
    fn from(m: crate::gguf_native::GgufSourceMeta) -> Self {
        SourceMeta {
            tokenizer: m.tokenizer,
            chat_template: m.chat_template,
            chat_template_source: m.chat_template_source,
            arch_id: m.arch_id,
        }
    }
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

/// Host RAM an `auto` PLE table prewarm leaves free to protect the unified
/// memory the GPU and the request path share. Larger than the hot-row store's
/// reserve: on Halo (MemAvailable ~24 GiB at ready) prewarming the whole
/// 26.8 GiB q4 table pushed reclaim into the request, and Ciru 8K fell to
/// 1,734 tok/s from 1,828 at 15 GiB. `auto` also caps at
/// [`PLE_PREWARM_AUTO_CAP_BYTES`] unless a discrete GPU holds its experts in
/// host RAM.
const PLE_PREWARM_RESERVE_BYTES: u64 = 8 << 30;

/// Most an `auto` PLE table prewarm reads on unified memory and on a discrete
/// GPU whose experts are all resident. An explicit
/// `memory.ple_prewarm_bytes` ignores it.
const PLE_PREWARM_AUTO_CAP_BYTES: u64 = 15 << 30;

/// Bytes of the q4 PLE table to prewarm now that the model is ready: an
/// explicit `memory.ple_prewarm_bytes`, or (`auto`) `MemAvailable - CMA` minus
/// [`PLE_PREWARM_RESERVE_BYTES`] (0 when MemAvailable is unknown), capped at
/// [`PLE_PREWARM_AUTO_CAP_BYTES`] unless the GPU is discrete and holds
/// host-placed experts. The snapshot is taken after the weights, pinned
/// experts and MTP are attached, so it already excludes them and nothing
/// further is subtracted here. When uncapped the budget grows with host RAM
/// and covers the whole table when the table fits above the reserve; the store
/// and [`ple_resident_bytes`] cap it at the table's data.  The flag is whether
/// the budget was set to zero.
fn ple_prewarm_budget(is_uma: bool, has_host_experts: bool) -> (u64, Option<u64>, bool) {
    let available = hipfire_runtime::ple_hot_store::mem_available_bytes();
    let (budget, explicit_zero) = ple_prewarm_budget_for(
        hipfire_config::memory::ple_prewarm_bytes(),
        available,
        is_uma,
        has_host_experts,
    );
    (budget, available, explicit_zero)
}

/// [`ple_prewarm_budget`] for a given request and `MemAvailable - CMA`. Only
/// a discrete GPU (`!is_uma`) with host-placed experts is uncapped.
fn ple_prewarm_budget_for(
    request: hipfire_config::memory::PleHotStoreBudget,
    available: Option<u64>,
    is_uma: bool,
    has_host_experts: bool,
) -> (u64, bool) {
    match request {
        hipfire_config::memory::PleHotStoreBudget::Bytes(bytes) => (bytes, bytes == 0),
        hipfire_config::memory::PleHotStoreBudget::Auto => {
            let budget = available.map_or(0, |available| {
                available.saturating_sub(PLE_PREWARM_RESERVE_BYTES)
            });
            (
                if !is_uma && has_host_experts {
                    budget
                } else {
                    budget.min(PLE_PREWARM_AUTO_CAP_BYTES)
                },
                false,
            )
        }
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

    fn mtp_measured_ar_floor(&self) -> bool {
        hipfire_arch_qwen4::MTP_MEASURED_AR_FLOOR
    }

    fn admit_topology(
        &self,
        _arch_id: u32,
        is_dir: bool,
        pp: usize,
        _kv_backend: KvBackend,
    ) -> Result<(), String> {
        if is_dir {
            return Err(crate::admission::QWEN4_SAFETENSORS_REFUSAL.into());
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
        // DFlash2 runs only when explicitly requested, with its draft; native
        // MTP stays the default speculator.
        let dflash = spec.dflash == Some(true);
        if dflash && draft_path.is_none() {
            return Err(
                "qwen4: DFlash (speculation.dflash = on) needs a DFlash2 draft \
                 (--model-draft / params.draft)"
                    .into(),
            );
        }
        if dflash && spec.mtp == Some(true) {
            return Err(
                "qwen4: DFlash and native MTP are both requested; pick one speculator".into(),
            );
        }
        let unsupported = [
            (
                draft_path.is_some() && !dflash,
                "draft model (draft_path; --spec dflash runs a DFlash2 draft)",
            ),
            (options.eagle_drafter, "EAGLE drafter"),
            (options.kv_adaptive, "adaptive KV"),
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
        // Admission checked the draft is present; DFlash2 replaces native MTP.
        let dflash_draft = ctx.draft_path.filter(|_| ctx.spec.dflash == Some(true));
        if let Some(draft) = dflash_draft {
            hipfire_config::load_summary::note(
                hipfire_config::load_summary::Facet::SpecOn,
                format!(
                    "dflash (draft {})",
                    std::path::Path::new(draft)
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_else(|| draft.to_string())
                ),
            );
            if native_mtp {
                hipfire_config::load_summary::note(
                    hipfire_config::load_summary::Facet::SpecOff,
                    "MTP off: DFlash draft replaces native MTP",
                );
            }
        }
        let native_mtp = native_mtp && dflash_draft.is_none();
        let meta = resolve_source_meta(&src, ctx.path)?;
        let shared = Qwen4RankShared {
            continuous_batch: ctx.batch_request.lanes().is_some(),
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
            let split = qwen4_ep_vram_layers(
                ctx.gpu,
                &shared,
                &plan
                    .seal
                    .expert_spans()
                    .map_err(|error| format!("qwen4: {error}"))?,
                &free,
                &src,
                &ctx.batch_request,
            )?;
            plan.expert_vram_layers = split.layers;
            eprintln!(
                "qwen4 EP plan: {} ranks on devices {:?} (mesh epoch {}, {} sealed expert groups), free VRAM per rank {:?} MiB, expert VRAM split {} MoE layers on every rank",
                plan.seal.ranks(),
                devices,
                plan.seal.mesh().epoch().as_u64(),
                plan.seal.plans().len(),
                free.iter().map(|bytes| bytes >> 20).collect::<Vec<_>>(),
                plan.expert_vram_layers
            );
            // One trade line per load (not per rank), only when this load
            // asked for lanes.
            if let Some(line) = hipfire_arch_qwen4::expert_residency::ep_lane_reservation_line(
                shared.max_seq,
                &split,
            ) {
                eprintln!("{line}");
            }
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
        // Rows a verify forward of the placed load runs row-exact; the n-gram
        // copy lane's take cap reads it. Experts hosted on the CPU verify at most the
        // admitted CPU window (legacy 4 floor, 64 only when actually prewarmed),
        // otherwise the per-GPU row-exact rows apply.
        let verify_rows = qwen4_placed_verify_rows(&loaded.bundle, &ctx.gpu.arch);
        // An explicit K keeps its value up to the physical executor width and
        // the placed load's qualified verify window (the CPU-window hard cap
        // narrows it below); the `auto` cap is resolved here from the placed
        // load's own CPU ownership: one below the smaller of the GPU's
        // row-exact rows and the armed CPU mode's window. A resource-64
        // reserve alone never widens an actual proposer.
        let max_k = if qwen4_explicit_mtp_k(ctx.spec).is_some() {
            max_k
                .min(hipfire_runtime::spec_economics::DERIVED_MAX_DRAFTS)
                .min(verify_rows.saturating_sub(1).max(1))
        } else {
            qwen4_auto_mtp_k(&ctx.gpu.arch, loaded.bundle.native_mtp_cpu_window_rows())
        };
        let cpu_splice = loaded.bundle.has_cpu_deferred_experts();
        let speculator = match dflash_draft {
            Some(draft) => match qwen4_dflash2_speculator(
                ctx.gpu,
                draft,
                &mut loaded.bundle,
                max_seq,
                cpu_splice,
                qwen4_dflash2_block(ctx.spec, verify_rows),
                verify_rows,
            ) {
                Ok(speculator) => Some(speculator),
                Err(error) => {
                    let _ = loaded.bundle.free_gpu(ctx.gpu);
                    return Err(error);
                }
            },
            None => native_mtp.then(|| {
                qwen4_native_mtp_speculator(
                    max_k,
                    qwen4_explicit_mtp_k(ctx.spec),
                    max_seq,
                    &ctx.gpu.arch,
                    meta.tokenizer.special_token_id("<|im_end|>"),
                    meta.tokenizer.special_token_id("</think>"),
                    true,
                    verify_rows,
                    loaded.bundle.state_format().gdn,
                    loaded.measured_load_identity.clone(),
                )
            }),
        };

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

/// Rows one verify forward of a placed Qwen4 load runs row-exact against the
/// single-row decode: the admitted CPU routed-expert window (the loader-proven
/// D1 storage, [`Qwen4Bundle::cpu_d1_storage_capacity_rows`], floored at
/// [`hipfire_arch_qwen4::program::QWEN4_CPU_SPLICE_MAX_ROWS`]) when host-placed
/// experts run on the CPU, else the GPU's `qwen4_row_exact_verify_rows`.
/// The native MTP draft cap is one below it.
fn qwen4_placed_verify_rows(
    bundle: &hipfire_arch_qwen4::bundle::Qwen4Bundle,
    gpu_arch: &str,
) -> usize {
    let gpu_rows = rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(gpu_arch);
    if bundle.has_cpu_deferred_experts() || bundle.native_mtp_cpu_window_rows().is_some() {
        gpu_rows.min(
            bundle
                .cpu_d1_storage_capacity_rows()
                .max(hipfire_arch_qwen4::program::QWEN4_CPU_SPLICE_MAX_ROWS),
        )
    } else {
        gpu_rows
    }
}

/// The explicit native MTP draft window of a Qwen4 load: the load param
/// `speculation.mtp_k`, else the process config (`HIPFIRE_MTP_K` /
/// `speculation.mtp_k`); `None` = `auto`.
fn qwen4_explicit_mtp_k(spec: SpecLoadCfg) -> Option<usize> {
    spec.mtp_k.or(hipfire_runtime::config::get().mtp_k)
}

/// The `auto` native MTP draft cap: one below the smaller of the GPU's
/// row-exact verify rows and `cpu_window_rows`, the window of the armed CPU
/// routed-expert mode (`Qwen4Bundle::native_mtp_cpu_window_rows`, `None` while
/// no layer is CPU-owned). Before placement is known pass `None`. No
/// load-time policy clamp here: the config default already sizes the auto
/// cap from proven rows, and the derived/drafter ceilings bound it by the
/// physical [`hipfire_runtime::spec_economics::DERIVED_MAX_DRAFTS`] width plus
/// the qualified bundle/CPU and explicit-user bounds.
fn qwen4_auto_mtp_k(gpu_arch: &str, cpu_window_rows: Option<usize>) -> usize {
    let gpu_rows = rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(gpu_arch);
    let rows = cpu_window_rows.map_or(gpu_rows, |window| gpu_rows.min(window));
    hipfire_config::mtp_k_default(hipfire_arch_qwen4::ARCH_ID, rows).max(1)
}

/// The native MTP draft window of a Qwen4 load before placement: an explicit
/// K or the GPU's `auto` cap. Physically bounded by the derived executor
/// width ([`hipfire_runtime::spec_economics::DERIVED_MAX_DRAFTS`]) and at
/// least 1; the qualified bundle/CPU and explicit-user bounds apply at the
/// Single/PP resolution below. A resource-64 reserve alone never widens an
/// actual proposer. The one resolution of the Single and the
/// pipeline-parallel load; the Single load re-resolves an `auto` cap after
/// placement (`qwen4_auto_mtp_k`) and narrows a CPU-placed load to the CPU
/// window.
fn qwen4_native_mtp_k(spec: SpecLoadCfg, gpu_arch: &str) -> usize {
    qwen4_explicit_mtp_k(spec)
        .map(|k| k.clamp(1, hipfire_runtime::spec_economics::DERIVED_MAX_DRAFTS))
        .unwrap_or_else(|| qwen4_auto_mtp_k(gpu_arch, None))
}

/// Qwen4 DFlash2 width under `speculation.dflash_block = auto`: B=8 (seven
/// drafts, of which the confidence cut verifies a prefix) is the fastest
/// measured width on gfx1151 (HIPF-135: 52.0 tok/s vs B=4 48.6).
const QWEN4_DFLASH2_AUTO_BLOCK: usize = 8;

/// The proposal width of a Qwen4 DFlash2 load: an explicit
/// `speculation.dflash_block`, else [`QWEN4_DFLASH2_AUTO_BLOCK`], clamped to
/// the rows a verify runs row-exact (`verify_rows`: 8 on gfx1151) so greedy
/// output stays the AR stream.
fn qwen4_dflash2_block(spec: SpecLoadCfg, verify_rows: usize) -> usize {
    match spec.dflash_block {
        Some(block) if !spec.dflash_block_auto => block,
        _ => QWEN4_DFLASH2_AUTO_BLOCK,
    }
    .min(verify_rows)
}

/// The shared DFlash2 speculator (`--spec dflash --model-draft <draft>`) on
/// a single-card Qwen4 bundle at proposal width `block`
/// ([`qwen4_dflash2_block`]). CPU-hosted routed experts (a 4-row verify
/// window) are refused rather than guessed at.
fn qwen4_dflash2_speculator(
    gpu: &mut rdna_compute::Gpu,
    draft: &str,
    bundle: &mut hipfire_arch_qwen4::bundle::Qwen4Bundle,
    max_seq: usize,
    cpu_splice: bool,
    block: usize,
    verify_rows: usize,
) -> Result<Box<dyn hipfire_runtime::spec::Speculator>, String> {
    if cpu_splice {
        return Err(
            "qwen4: DFlash2 is not supported with CPU-hosted routed experts; \
             load the experts on the GPU or use native MTP (--spec mtp)"
                .into(),
        );
    }
    let draft_hfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(draft))
        .map_err(|error| format!("qwen4: DFlash2 draft '{draft}': {error}"))?;
    let config = hipfire_runtime::dflash::DflashConfig::from_hfq(&draft_hfq).ok_or_else(|| {
        format!("qwen4: '{draft}' is not a DFlash draft (missing `dflash` metadata)")
    })?;
    drop(draft_hfq);
    if config.hidden != bundle.config.hidden_size || config.vocab_size != bundle.config.vocab_size {
        return Err(format!(
            "qwen4: DFlash2 draft '{draft}' (hidden {}, vocab {}) was not made for this model \
             (hidden {}, vocab {})",
            config.hidden, config.vocab_size, bundle.config.hidden_size, bundle.config.vocab_size
        ));
    }
    let speculator =
        hipfire_runtime::dflash2_spec::build_dflash2_speculator(gpu, draft, bundle, max_seq, block)
            .map_err(|error| format!("qwen4: DFlash2 speculator: {error}"))?;
    eprintln!(
        "  qwen4 DFlash2 speculator: {draft} (block {block}, target layers {:?}, verify rows {verify_rows})",
        config.target_layer_ids
    );
    Ok(speculator)
}

/// The generic speculator around the native Qwen4 MTP head, with the load log
/// lines. `end_of_turn` is the tokenizer's `<|im_end|>` id and `think_close`
/// its `</think>` id. `ngram_modifier` false (a pipeline-parallel load) builds
/// it without an n-gram pool whatever `speculation.mtp_ngram` resolves to,
/// so no request can arm a takeover.
/// `explicit_mtp_k` preserves user intent separately from the incumbent
/// `max_k`: `mode auto` must not become an economic ceiling for derived
/// selection. `verify_rows` is the verify window the n-gram copy lane's take
/// cap reads: the per-GPU row-exact rows, or the 4-row CPU splice window when
/// host-placed routed experts run on the CPU. `gdn` is the loaded GDN state
/// format; under `q8_ef` sampled requests stay on AR
/// ([`hipfire_arch_qwen4::mtp_spec::qwen4_mtp_sampled_allowed`]).
/// `measured_load_identity` is the settled descriptor composed once per rank
/// at the return site (already `Arc`-shared, never re-hashed).
fn qwen4_native_mtp_speculator(
    max_k: usize,
    explicit_mtp_k: Option<usize>,
    max_seq: usize,
    gpu_arch: &str,
    end_of_turn: Option<u32>,
    think_close: Option<u32>,
    ngram_modifier: bool,
    verify_rows: usize,
    gdn: hipfire_arch_qwen4::GdnStateFormat,
    measured_load_identity: Option<
        std::sync::Arc<hipfire_arch_qwen4::measured_identity::Qwen4MeasuredLoadIdentity>,
    >,
) -> Box<dyn hipfire_runtime::spec::Speculator> {
    eprintln!("  qwen4 native MTP speculator enabled (K={max_k})");
    hipfire_config::load_summary::note(
        hipfire_config::load_summary::Facet::SpecOn,
        format!("qwen4 native mtp K={max_k}"),
    );
    if hipfire_config::mtp_sampled_enabled()
        && !hipfire_arch_qwen4::mtp_spec::qwen4_mtp_sampled_allowed(gdn)
    {
        eprintln!(
            "  qwen4 native MTP sampled: off for GDN q8_ef (sampled requests use AR until the sampled-MTP law is verified)"
        );
    }
    let arch = hipfire_arch_qwen4::ARCH_ID;
    let mut copy_lane = false;
    let ngram = if !ngram_modifier {
        eprintln!("  qwen4 native MTP n-gram: off (topology.pp)");
        hipfire_config::load_summary::note(
            hipfire_config::load_summary::Facet::CopyLane,
            "n-gram copy lane off (topology.pp)",
        );
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
                hipfire_config::load_summary::note(
                    hipfire_config::load_summary::Facet::CopyLane,
                    if ngram_on && copy_lane {
                        "n-gram copy lane on"
                    } else {
                        "n-gram copy lane off"
                    },
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
        explicit_mtp_k,
        max_seq,
        end_of_turn,
        think_close,
        ngram,
        copy_lane,
        gdn,
        measured_load_identity,
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
    /// The load requests continuous batching (the daemon stages lanes after
    /// the load): decode-share is single-stream and refuses (a lane needs a
    /// scratch bank of its own). `false` for every caller that stages no lanes.
    pub continuous_batch: bool,
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
    /// Immutable measured-load descriptor, independent of chooser policy.
    /// Missing verified evidence or unsupported scope leaves a named absence.
    /// Shared once per load; the drafter borrows the same Arc.
    pub measured_load_identity:
        Option<std::sync::Arc<hipfire_arch_qwen4::measured_identity::Qwen4MeasuredLoadIdentity>>,
    /// Frozen conservative allocation-plan union of a supported fresh rank
    /// load, not a measured peak or a qualification of wider execution.
    pub measured_resource_snapshot:
        Option<std::sync::Arc<hipfire_arch_qwen4::measured_resources::Qwen4LoadResourceSnapshot>>,
}

// ─── Measured load identity (FN chooser) ─────────────────────────────
//
// Single immutable descriptor per load, composed once from EXISTING verified
// seals plus the settled placement/scope/config identity and passed as a
// shared Arc into the drafter builder. No per-window hashing or
// serialization; the module caches its canonical JSON at construction.
// Missing verified evidence => None + named eprintln reason; the ordinary
// load stays a real incumbent, never an invented hash or expected pin.
/// Verified SHA-256 of `path` against its live stat, or `None` naming why.
fn qwen4_verified_sha(
    cache: &hipfire_runtime::verified_cache::VerifiedCache,
    path: &std::path::Path,
    what: &str,
) -> Option<String> {
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) => {
            eprintln!("  qwen4 measured load: unavailable ({what} stat failed: {error})");
            return None;
        }
    };
    let stat = hipfire_runtime::verified_cache::StatKey::of(&metadata);
    match cache.verified_digest(path, &stat) {
        Some(digest) => Some(digest),
        None => {
            eprintln!(
                "  qwen4 measured load: unavailable ({what} has no verified cache seal; \
                 hash the artifact through the same cache outside inference)"
            );
            None
        }
    }
}
/// Render one process-config scalar with the existing canonical conventions
/// (bool 1/0, integer/float display, strings verbatim; Null omitted).
fn qwen4_config_scalar(value: &hipfire_config::ConfigValue) -> Option<String> {
    match value {
        hipfire_config::ConfigValue::Bool(value) => {
            Some(if *value { "1".into() } else { "0".into() })
        }
        hipfire_config::ConfigValue::Integer(value) => Some(value.to_string()),
        hipfire_config::ConfigValue::Float(value) => Some(value.to_string()),
        hipfire_config::ConfigValue::String(value) => Some(value.clone()),
        hipfire_config::ConfigValue::Null => None,
    }
}
/// The verified cache of the hipfire home (read-only at load): the same
/// `<home>/cache/verified-artifacts.json` the CLI pull records genuine
/// pins through. A missing home yields a disabled cache (every lookup a
/// named miss), never a fabricated digest.
fn verified_cache_for_load() -> hipfire_runtime::verified_cache::VerifiedCache {
    match hipfire_config::hipfire_home() {
        Ok(home) => hipfire_runtime::verified_cache::VerifiedCache::in_home(&home),
        Err(error) => {
            eprintln!("  qwen4 measured load: unavailable (hipfire home: {error})");
            hipfire_runtime::verified_cache::VerifiedCache::disabled()
        }
    }
}
/// Stable explicit recipe string (erased bool-only classes are not identity).
fn qwen4_recipe_string(recipe: &hipfire_arch_qwen4::weights::Qwen4TierRecipe) -> String {
    match recipe {
        hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextPacked {
            trunk_tier,
            mtp_source,
        } => {
            format!("flash-next-packed/trunk={trunk_tier:?}/mtp_source={mtp_source}")
        }
        hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab {
            trunk_tier,
            mtp_source,
            ..
        } => {
            format!("flash-next-slab/trunk={trunk_tier:?}/mtp_source={mtp_source}")
        }
        hipfire_arch_qwen4::weights::Qwen4TierRecipe::StockFp8 => "stock-fp8".to_string(),
        hipfire_arch_qwen4::weights::Qwen4TierRecipe::SyntheticSourceBf16 => {
            "synthetic-source-bf16".to_string()
        }
        hipfire_arch_qwen4::weights::Qwen4TierRecipe::SyntheticPacked => {
            "synthetic-packed".to_string()
        }
    }
}
/// Compose this rank's immutable measured-load descriptor from settled seals.
///
/// Policy-independent: composition never reads the FN chooser key — the
/// economic choice stays gated at the actual controller/drafter (which keeps
/// the real incumbent whenever the flag is off or no descriptor binds). An
/// eligible profile load and the incumbent bind the SAME once-per-load Arc:
/// chooser 0/1 selects policy, never a different numerical identity.
/// `overlay_path/overlay_bytes` name the APPLIED overlay evidence (or `None`
/// when no overlay was applied). Mere file presence never counts — only
/// applied artifacts hash.
/// `capture_rows` is the settled native ring rows from the loaded bundle
/// (zero where the route captures no rows). `gpu_uuid` is the actual
/// initialized device identity. Returns `None` with a named eprintln reason
/// when any actually-applied artifact lacks a verified seal or
/// placement/scope identity is unavailable; the ordinary load is
/// unaffected. No config/CLI surface is added.
#[allow(clippy::too_many_arguments)]
fn qwen4_measured_load_identity(
    bundle: &hipfire_arch_qwen4::bundle::Qwen4Bundle,
    hfq_path: &std::path::Path,
    model_bytes: u64,
    ple_sha: &str,
    ple_bytes: u64,
    ple_encoding: &str,
    overlay_path: Option<&std::path::Path>,
    overlay_bytes: Option<u64>,
    hotset_path: Option<&std::path::Path>,
    hotset_bytes: Option<u64>,
    max_seq: usize,
    capture_rows: usize,
    gpu_arch: &str,
    gpu_uuid: &str,
    gpu_flags: &rdna_compute::FeatureFlags,
    cache: &hipfire_runtime::verified_cache::VerifiedCache,
) -> Option<std::sync::Arc<hipfire_arch_qwen4::measured_identity::Qwen4MeasuredLoadIdentity>> {
    let mut fields = std::collections::BTreeMap::new();
    fields.insert("schema".to_string(), "qwen4-measured-load.v1".to_string());
    let model_sha = qwen4_verified_sha(cache, hfq_path, "model")?;
    fields.insert("model.sha256".to_string(), model_sha);
    fields.insert("model.bytes".to_string(), model_bytes.to_string());
    fields.insert("ple.sha256".to_string(), ple_sha.to_string());
    fields.insert("ple.bytes".to_string(), ple_bytes.to_string());
    fields.insert("ple.encoding".to_string(), ple_encoding.to_string());
    match (overlay_path, overlay_bytes) {
        (Some(path), Some(bytes)) => {
            let sha = qwen4_verified_sha(cache, path, "applied trunk overlay")?;
            fields.insert("overlay.sha256".to_string(), sha);
            fields.insert("overlay.bytes".to_string(), bytes.to_string());
        }
        (None, None) => {
            fields.insert("overlay".to_string(), "none".to_string());
        }
        _ => {
            eprintln!("  qwen4 measured load: unavailable (applied overlay path/byte evidence incomplete)");
            return None;
        }
    }
    match (hotset_path, hotset_bytes) {
        (Some(path), Some(bytes)) => {
            let sha = qwen4_verified_sha(cache, path, "applied hot set")?;
            fields.insert("hotset.sha256".to_string(), sha);
            fields.insert("hotset.bytes".to_string(), bytes.to_string());
        }
        (None, None) => {
            fields.insert("hotset".to_string(), "none".to_string());
        }
        _ => {
            eprintln!("  qwen4 measured load: unavailable (applied hot-set path/byte evidence incomplete)");
            return None;
        }
    }
    // Settled placement: existing placement_digest over the actually
    // host-placed trunk layers, plus the tiered hot-set logical digest when
    // tiered (both when tiered), plus the host execution/split identity from
    // the existing bundle scope — never a new placement algorithm.
    let trunk_layers = bundle.config.num_hidden_layers;
    let host_layers = hipfire_arch_qwen4::expert_residency::host_placed_trunk_layers(
        &bundle.weights.manifest.weights,
    );
    let mut placement =
        hipfire_arch_qwen4::expert_residency::placement_digest(trunk_layers, &host_layers);
    if let Some(lease) = bundle.weights.expert_tiers() {
        use std::fmt::Write as _;
        let mut hex = String::with_capacity(64);
        for byte in lease.logical_digest() {
            let _ = write!(hex, "{byte:02x}");
        }
        placement.push('/');
        placement.push_str(&hex);
    }
    let host_exec = bundle.host_exec_identity();
    if !host_exec.is_empty() {
        placement.push('/');
        placement.push_str(&host_exec);
    }
    fields.insert("placement.digest".to_string(), placement);
    fields.insert(
        "recipe".to_string(),
        qwen4_recipe_string(&bundle.weights.manifest.identity.recipe),
    );
    fields.insert("state".to_string(), format!("{:?}", bundle.state_format()));
    fields.insert(
        "kv_backend".to_string(),
        bundle.state.qsa_backend().name().to_string(),
    );
    fields.insert("max_seq".to_string(), max_seq.to_string());
    fields.insert("capture_rows".to_string(), capture_rows.to_string());
    fields.insert("gpu.arch".to_string(), gpu_arch.to_string());
    fields.insert("gpu.uuid".to_string(), gpu_uuid.to_string());
    let scope = hipfire_runtime::session_cache::SessionState::snapshot_scope(
        bundle,
        hipfire_runtime::session_cache::SessionRoute::Mtp,
    );
    let Some(scope) = scope else {
        eprintln!("  qwen4 measured load: unavailable (MTP snapshot scope unavailable)");
        return None;
    };
    fields.insert("scope".to_string(), scope);
    fields.insert("flags".to_string(), format!("{gpu_flags:?}"));
    // Immutable process-route identity: kernel.*/diagnostic.kernel.*/memory.*
    // and developer.* route policy; never ambient env, filesystem or context
    // roots. The chooser-only derived_k key selects the FN depth policy, not
    // the numerical route, so it is skipped explicitly before insertion: the
    // module validator rejects it rather than stripping it, and a kept key
    // would force every derived load to `None`.
    let process = hipfire_config::active_or_local_process_config();
    for (key, value) in &process.values.values {
        if hipfire_arch_qwen4::measured_identity::CHOOSER_POLICY_KEY.strip_prefix("config.")
            == Some(key.as_str())
        {
            continue;
        }
        let keep = key.starts_with("kernel.")
            || key.starts_with("diagnostic.kernel.")
            || key.starts_with("memory.")
            || key.starts_with("developer.");
        if !keep {
            continue;
        }
        if let Some(rendered) = qwen4_config_scalar(value) {
            fields.insert(format!("config.{key}"), rendered);
        }
    }
    match hipfire_arch_qwen4::measured_identity::Qwen4MeasuredLoadIdentity::from_fields(fields) {
        Ok(identity) => Some(std::sync::Arc::new(identity)),
        Err(reason) => {
            eprintln!("  qwen4 measured load: unavailable ({reason})");
            None
        }
    }
}

/// One owning placement census plus settled placement-component upper bounds.
/// No second planner: borrowed views/aliases own nothing, the new capture is
/// inside the native-MTP reserve, and VMM census terms charge mapped granules.
/// Original capacity inputs survive fulfillment; parked pools are read again
/// only after all load allocations, including the actual capture, settle.
struct Qwen4SettledResourceEvidence {
    reserve: hipfire_arch_qwen4::expert_residency::AutoReserve,
    non_expert_device_bytes: u64,
    routed_device_bytes: u64,
    host_mapped_bytes: u64,
    tables_bytes: u64,
    stage_bytes: u64,
    margin_bytes: u64,
    host_check_bytes: u64,
    stage_mirror_bytes: u64,
    mem_available: Option<u64>,
    ttm_pool: u64,
    gtt_free_bytes: Option<u64>,
}

/// Compose the frozen resource snapshot beside the measured-load identity.
///
/// The reserve and charges are the retained owning branch's values, captured
/// before manifest mutation. Owner extents correct known padding intersections;
/// stage/mirror/table terms are source-proven conservative component bounds,
/// not claims of measured physical usage. The two parked-pool instants remain
/// explicitly charged where their intersection is unknown. Only a supported
/// single-rank actual-64 load with its original capacity basis can publish.
#[allow(clippy::too_many_arguments)]
fn qwen4_measured_resource_snapshot(
    identity: &std::sync::Arc<hipfire_arch_qwen4::measured_identity::Qwen4MeasuredLoadIdentity>,
    gpu: &rdna_compute::Gpu,
    bundle: &hipfire_arch_qwen4::bundle::Qwen4Bundle,
    preallocation_pool_bytes: u64,
    preallocation_free_device_bytes: Option<u64>,
    mut evidence: Qwen4SettledResourceEvidence,
    admitted_max_seq: usize,
    admitted_chunk_rows: usize,
    capture_rows: usize,
    config: &hipfire_arch_qwen4::Qwen4Config,
    gdn: hipfire_arch_qwen4::GdnStateFormat,
    kv_backend_name: &str,
    scratch_mode: hipfire_arch_qwen4::gpu_forward::Qwen4ForwardScratchMode,
) -> Option<std::sync::Arc<hipfire_arch_qwen4::measured_resources::Qwen4LoadResourceSnapshot>> {
    use hipfire_arch_qwen4::measured_resources::{
        Qwen4AdmittedShape, Qwen4BackingKind, Qwen4CaptureGeometry, Qwen4HostCheckInputs,
        Qwen4LoadBacking, Qwen4LoadResourceSnapshot, Qwen4PlacementBytes, Qwen4RingMode,
    };
    // Row-exact 64 enforced by `Qwen4LoadResourceSnapshot::new`; this early
    // gate only names the reason before the typed constructor does.
    if capture_rows != 64 {
        eprintln!("  qwen4 measured resources: unavailable (settled capture rows {capture_rows}, not the row-exact 64)");
        return None;
    }
    if admitted_chunk_rows < capture_rows {
        eprintln!("  qwen4 measured resources: unavailable (settled forward scratch has {admitted_chunk_rows} rows, below actual capture {capture_rows})");
        return None;
    }
    // Fulfillment's owning handles provide the physical extent authority for
    // padding/cutover growth. Aliases and Borrowed views (including tier
    // projections) are not allocations. Tier owners replace, never supplement,
    // the removed whole-trunk expert owners. This only increases a component
    // bound; it does not re-plan placement or subtract unknown intersections.
    let owner_bytes = || -> Result<(u64, u64), String> {
        use hipfire_arch_qwen4::expert_tier_plan::Placement;
        let tier = bundle.weights.expert_tiers();
        let mut device = tier.map_or(0, |lease| lease.owner_bytes(Placement::Device));
        let mut host = tier.map_or(0, |lease| lease.owner_bytes(Placement::HostMapped));
        for taken in &bundle.weights.taken {
            let hipfire_runtime::weight_store::WeightHandle::Resident(tensor) = &taken.handle
            else {
                continue; // Tied aliases have no physical owner.
            };
            if tensor.buf.is_borrowed() {
                continue;
            }
            if tensor.buf.is_host_mapped() {
                let bytes = (tensor.buf.size() as u64)
                    .checked_add(hipfire_arch_qwen4::expert_residency::HOST_MAPPED_PAD_BYTES)
                    .ok_or("host owner extent overflows")?;
                host = host
                    .checked_add(bytes)
                    .ok_or("host owner union overflows")?;
            } else {
                let bytes = if tensor.buf.is_vmm_owner() {
                    gpu.vmm_mapped_bytes(tensor)
                        .ok_or("owning weight has no VMM mapped-byte authority")?
                } else {
                    tensor.buf.size()
                } as u64;
                device = device
                    .checked_add(bytes)
                    .ok_or("device owner union overflows")?;
            }
        }
        Ok((device, host))
    };
    let (device_owners, host_owners) = match owner_bytes() {
        Ok(bytes) => bytes,
        Err(reason) => {
            eprintln!("  qwen4 measured resources: unavailable ({reason})");
            return None;
        }
    };
    let corrected_non_expert = evidence
        .non_expert_device_bytes
        .checked_add(evidence.routed_device_bytes)
        .and_then(|planned| {
            evidence
                .non_expert_device_bytes
                .checked_add(device_owners.saturating_sub(planned))
        });
    let Some(corrected_non_expert) = corrected_non_expert else {
        eprintln!(
            "  qwen4 measured resources: unavailable (owning weight extent correction overflows)"
        );
        return None;
    };
    evidence.non_expert_device_bytes = corrected_non_expert;
    evidence.host_mapped_bytes = evidence.host_mapped_bytes.max(host_owners);
    evidence.host_check_bytes = evidence.host_check_bytes.max(host_owners);
    // A direct initial capture retires the standalone recurrent plane into
    // the pool and rebinds recurrent tensors as Borrowed ring views. There is
    // no old capture ring to add (old_capture_bytes=0). The original reserve
    // still bounds the pre-capture state; it may intersect the settled pool's
    // retired planes. Both instants remain deliberately charged by the frozen
    // conservative-union contract, never called exact physical usage.
    let owned_gdn_layers = config.n_linear_layers();
    let Some(new_capture_bytes) = hipfire_arch_qwen4::mtp_spec::qwen4_capture_bytes(
        config,
        gdn,
        capture_rows,
        owned_gdn_layers,
    ) else {
        eprintln!("  qwen4 measured resources: unavailable (actual capture geometry has no checked byte authority)");
        return None;
    };
    if new_capture_bytes == 0 {
        eprintln!("  qwen4 measured resources: unavailable (actual capture ring holds no bytes)");
        return None;
    }
    let admitted = Qwen4AdmittedShape {
        max_seq: admitted_max_seq,
        chunk_rows: admitted_chunk_rows,
        capture_rows,
        ranks: 1,
        gdn_state: gdn.name().to_string(),
        kv_backend: kv_backend_name.to_string(),
        scratch_mode: format!("{scratch_mode:?}"),
        geometry: Qwen4CaptureGeometry {
            linear_num_key_heads: config.linear_num_key_heads,
            linear_num_value_heads: config.linear_num_value_heads,
            linear_key_head_dim: config.linear_key_head_dim,
            linear_value_head_dim: config.linear_value_head_dim,
            owned_gdn_layers,
        },
        mode: Qwen4RingMode::DirectInitial64,
        old_capture_bytes: 0,
        new_capture_bytes,
    };
    let placement = Qwen4PlacementBytes {
        non_expert_device_bytes: evidence.non_expert_device_bytes,
        routed_device_bytes: evidence.routed_device_bytes,
        host_mapped_bytes: evidence.host_mapped_bytes,
        tables_bytes: evidence.tables_bytes,
        stage_bytes: evidence.stage_bytes,
        margin_bytes: evidence.margin_bytes,
    };
    let host = Qwen4HostCheckInputs {
        host_check_bytes: evidence.host_check_bytes,
        stage_mirror_bytes: evidence.stage_mirror_bytes,
        host_headroom_bytes: hipfire_arch_qwen4::expert_residency::HOST_RAM_HEADROOM_BYTES,
        ttm_pool_bytes: evidence.ttm_pool,
        mem_available_bytes: evidence.mem_available,
        gtt_budget_bytes: evidence.gtt_free_bytes,
    };
    let backing = Qwen4LoadBacking {
        kind: if gpu.is_uma() {
            Qwen4BackingKind::UmaShared
        } else {
            Qwen4BackingKind::Discrete
        },
        preallocation_pooled_device_bytes: preallocation_pool_bytes,
        settled_pooled_device_bytes: gpu.pool_parked_bytes() as u64,
    };
    match Qwen4LoadResourceSnapshot::new(
        std::sync::Arc::clone(identity),
        admitted,
        evidence.reserve,
        placement,
        host,
        preallocation_free_device_bytes,
        backing,
    ) {
        Ok(snapshot) => {
            // A source-proven upper bound is not a measured peak. Publish it
            // only when its complete union fits the ORIGINAL capacity inputs;
            // a post-load free-memory delta cannot reconstruct that basis.
            let Some(free) = preallocation_free_device_bytes else {
                eprintln!(
                    "  qwen4 measured resources: unavailable (original device capacity unreadable)"
                );
                return None;
            };
            let Some(device_capacity) = free.checked_add(preallocation_pool_bytes) else {
                eprintln!("  qwen4 measured resources: unavailable (original reclaimable device capacity overflows)");
                return None;
            };
            if snapshot.device_plan_bytes() > device_capacity {
                eprintln!("  qwen4 measured resources: unavailable (device union upper bound {} exceeds original reclaimable capacity {device_capacity})",
                    snapshot.device_plan_bytes());
                return None;
            }
            let Some(available) = evidence.mem_available else {
                eprintln!(
                    "  qwen4 measured resources: unavailable (original host capacity unreadable)"
                );
                return None;
            };
            if snapshot.host_plan_bytes() > available.saturating_add(evidence.ttm_pool) {
                eprintln!("  qwen4 measured resources: unavailable (host union upper bound exceeds original MemAvailable plus TTM pool)");
                return None;
            }
            let gtt_plan = evidence
                .host_check_bytes
                .checked_add(evidence.stage_mirror_bytes)
                .and_then(|bytes| {
                    bytes.checked_add(if gpu.is_uma() {
                        snapshot.device_plan_bytes()
                    } else {
                        0
                    })
                });
            let Some(gtt_plan) = gtt_plan else {
                eprintln!(
                    "  qwen4 measured resources: unavailable (GTT component union overflows)"
                );
                return None;
            };
            if evidence.gtt_free_bytes.is_some_and(|free| gtt_plan > free) {
                eprintln!("  qwen4 measured resources: unavailable (GTT union upper bound exceeds original GTT room)");
                return None;
            }
            Some(std::sync::Arc::new(snapshot))
        }
        Err(reason) => {
            eprintln!("  qwen4 measured resources: unavailable ({reason})");
            None
        }
    }
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
        return Err(crate::admission::QWEN4_SAFETENSORS_REFUSAL.into());
    };
    let receipt = hipfire_arch_qwen4::admit_hfqm_artifact_on(hfq, Some(&gpu.arch))
        .map_err(|error| error.to_string())?;
    let identity = hfq
        .source_identity()
        .map_err(|error| format!("qwen4: EP source identity: {error}"))?;
    // Admission attached rank 0's PLE rows: the generation binds them and the
    // trunk-overlay sidecar with the base file.
    let generation = hipfire_arch_qwen4::ep_seal::qwen4_source_generation(
        hfq,
        std::path::Path::new(&shared.path),
    )?;
    let seal = hipfire_arch_qwen4::ep_seal::Qwen4EpSeal::seal(
        &receipt.manifest,
        receipt.config.num_hidden_layers,
        ep,
        &devices,
        &identity,
        &generation,
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

/// One union for the tier lease OR whole-layer tables (aliases), the distinct
/// stage tables, and optional kept-expert masks, each exactly once. Native
/// MTP owns its own table.
fn qwen4_expert_table_bytes(
    stage_possible: bool,
    (layers, experts, expert_keep): (usize, usize, bool),
) -> Result<u64, String> {
    use hipfire_arch_qwen4::expert_residency as residency;
    // The base owns the gate/up and down pointer tables (16 bytes per expert)
    // plus the 1-byte-per-expert codec (`EXPERT_TIER_TABLE_BYTES`). The stage
    // allocates two F32 address tables per staged layer (gate/up and down, 16
    // bytes per expert) and borrows the codec, so it adds 16 — never a second
    // copy of the owned 17.
    let stage = if stage_possible {
        (layers as u64)
            .checked_mul(experts as u64)
            .and_then(|entries| entries.checked_mul(2 * 8))
            .ok_or("stage-table component union overflows")?
    } else {
        0
    };
    let mask = if expert_keep {
        (layers as u64)
            .checked_add(1)
            .and_then(|owners| owners.checked_mul(experts as u64))
            .ok_or("keep-mask component union overflows")?
    } else {
        0
    };
    residency::EXPERT_TIER_TABLE_BYTES
        .checked_add(stage)
        .and_then(|bytes| bytes.checked_add(mask))
        .ok_or_else(|| "table/mask component union overflows".into())
}

/// First 8 bytes of a digest as 16 lowercase hex digits (the load-log form).
fn digest_hex16(digest: &[u8; 32]) -> String {
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Bytes of the BorrowPrefill device pointer tables a decode-share plan owns:
/// per prefill class view, gate/up and down, two `u64` pointers per expert
/// (`24,576 x 2 x 8`); 0 for a plan without donors. The codec table is shared.
fn qwen4_decode_share_table_bytes(
    plan: &hipfire_arch_qwen4::expert_tier_plan::ExpertTierPlan,
    (layers, experts, _): (usize, usize, bool),
) -> Result<u64, String> {
    let Some(tail) = plan
        .decode_share_tail()
        .filter(|tail| !tail.donors.is_empty())
    else {
        return Ok(0);
    };
    let views = tail.class_donors.len().max(1) as u64;
    (layers as u64)
        .checked_mul(experts as u64)
        .and_then(|entries| entries.checked_mul(2 * 8))
        .and_then(|bytes| bytes.checked_mul(views))
        .ok_or_else(|| "decode-share table bytes overflow".to_string())
}

/// Record the borrowed weight-shadow region (`shadow_bytes` at the start of
/// the donor tail; every class span of `plan` already includes it) on a plan
/// that carries a donor tail built for `sharing`. A plan without a donor tail
/// (sharing refused or ineligible) is returned untouched: its shadows stay
/// resident.
fn qwen4_record_shadow_region(
    plan: hipfire_arch_qwen4::expert_tier_plan::ExpertTierPlan,
    sharing: bool,
    shadow_bytes: u64,
) -> Result<hipfire_arch_qwen4::expert_tier_plan::ExpertTierPlan, String> {
    if sharing && plan.decode_share_tail().is_some() {
        plan.with_decode_share_shadows(shadow_bytes)
            .map_err(|error| format!("decode-share shadow region: {error}"))
    } else {
        Ok(plan)
    }
}

/// The production hotset/stage fixed point, without I/O or admission effects.
///
/// `decode_share_need` is `Some(S)` when decode-share `bind` is admitted: the
/// ranked plan is re-laid out with a donor tail of at least `S` bytes
/// ([`ExpertTierPlan::with_decode_share`]). The device bytes that layout adds
/// (`O = shared - plain` owner bytes, the guard and any padding growth) and
/// the second device pointer tables join this fixed point beside the stage
/// bytes, which are sized from the BorrowPrefill placement. The hot budget
/// is not grown: the ranked prefix is the plain plan's up to the experts `O`
/// displaces. `None` keeps every byte of the plain path.
fn qwen4_plan_hotset_placement(
    slabs: &[hipfire_arch_qwen4::expert_slab_table::ExpertSlab],
    rank: &[u32],
    hotset_budget: hipfire_arch_qwen4::expert_hotset::HotsetBudget,
    total: u64,
    stage_enabled: bool,
    table_geometry: (usize, usize, bool),
    decode_share_need: Option<u64>,
    decode_share_class_spans: &[u64],
) -> Result<
    (
        hipfire_arch_qwen4::expert_tier_plan::ExpertTierPlan,
        u64,
        u64,
    ),
    String,
> {
    use hipfire_arch_qwen4::expert_hotset as hotsets;
    use hipfire_arch_qwen4::expert_residency as residency;
    use hipfire_arch_qwen4::expert_tier_plan::Placement;
    const MIB: u64 = 1 << 20;
    let mut stage = 0u64;
    // The base tier table is already charged by expert_hotset_budget. The
    // separate stage tables and optional masks must share its fixed point.
    let mut extra_tables =
        qwen4_expert_table_bytes(false, table_geometry)? - residency::EXPERT_TIER_TABLE_BYTES;
    // Device bytes the donor layout adds over the plain plan (zero until a
    // shared plan has been laid out).
    let mut share_overhead = 0u64;
    let mut share = decode_share_need.filter(|need| *need > 0);
    loop {
        let available = hotsets::require_budget(total.checked_sub(stage)
            .and_then(|bytes| bytes.checked_sub(extra_tables))
            .and_then(|bytes| bytes.checked_sub(share_overhead)).ok_or_else(|| {
            format!(
                "the {} MiB expert budget cannot hold the {} MiB of long-prefill stage buffers; free GPU memory or lower the expert placement",
                total / MIB,
                stage / MIB
            )
        })).map_err(|skip| skip.to_string())?;
        let budget = hotset_budget.select(available)?;
        let plain =
            hipfire_arch_qwen4::expert_tier_plan::plan_expert_tiers_from_slabs(slabs, rank, budget)
                .map_err(|error| error.to_string())?;
        hotsets::require_experts(plain.device_experts()).map_err(|skip| skip.to_string())?;
        let mut overhead = 0u64;
        let plan = match share {
            Some(need) => {
                match plain
                    .clone()
                    .with_decode_share(rank, need)
                    .and_then(|(shared, _)| {
                        if decode_share_class_spans.is_empty() {
                            Ok(shared)
                        } else {
                            shared.with_decode_share_classes(decode_share_class_spans)
                        }
                    }) {
                    Ok(shared) => {
                        overhead = shared
                            .owner_bytes(Placement::Device)
                            .checked_sub(plain.owner_bytes(Placement::Device))
                            .ok_or("decode-share layout shrank the device owner")?;
                        shared
                    }
                    Err(error) => {
                        eprintln!("  qwen4 decode-share: ineligible: {error}; plain placement");
                        share = None;
                        if share_overhead != 0 {
                            // The earlier iterations reserved room for the donor
                            // layout; re-plan without it.
                            share_overhead = 0;
                            continue;
                        }
                        plain
                    }
                }
            }
            None => plain,
        };
        // new_tier allocates no stage without a host owner; the byte helper
        // alone still includes both tail pads for an empty host tier.
        let stage_possible = stage_enabled && plan.owner_bytes(Placement::HostMapped) > 0;
        let need = if stage_possible {
            residency::expert_tier_stage_bytes_prefill(&plan)
        } else {
            0
        };
        let need_tables = qwen4_expert_table_bytes(stage_possible, table_geometry)?
            - residency::EXPERT_TIER_TABLE_BYTES
            + qwen4_decode_share_table_bytes(&plan, table_geometry)?;
        if need <= stage && need_tables <= extra_tables && overhead <= share_overhead {
            return Ok((plan, budget, stage));
        }
        stage = stage.max(need);
        extra_tables = extra_tables.max(need_tables);
        share_overhead = share_overhead.max(overhead);
    }
}
/// What the expert placement is sized from, for one rank of a `ranks`-way
/// load: the replicated non-expert bytes, one MoE layer's routed-expert bytes
/// on this rank, the `auto` reserve ([`residency::AutoReserve`]: the larger
/// of the historical measured formula and a bottom-up census of the first
/// full-context request, itemised; the census alone where it carries the
/// measured HIP runtime overhead, [`residency::runtime_overhead_bytes`]) and
/// its native MTP and QSA gather parts, returned too. The assembly is
/// [`hipfire_arch_qwen4::expert_residency::placement_inputs`], shared with
/// every example/gate that sizes the same placement (one convention); this
/// keeps only the loader-private MTP host-spill policy. The one place this
/// is computed for the load, and for [`qwen4_ep_vram_layers`]. `chunk_rows`
/// is the chunk the reserve is sized from: the load passes the requested
/// prefill chunk
/// ([`hipfire_arch_qwen4::gpu_forward::qwen4_prefill_chunk_requested`])
/// for the placement.
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
    spans: &hipfire_arch_qwen4::expert_residency::EpExpertSpans,
    chunk_rows: usize,
    share: Option<hipfire_arch_qwen4::expert_residency::DecodeShareForward>,
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
    // Every placement host-maps at least the MTP layer's routed experts, so
    // the head is still attached after it only where the host-mapped policy
    // keeps it (exact gfx1100 or gfx1201, or an explicit `--spec mtp`); only then does
    // it need VRAM.
    let mtp_kept = native_mtp
        && crate::admission::qwen4_mtp_with_host_mapped_experts(
            spec,
            &gpu.arch,
            usize::from(routed_experts),
        );
    let inputs = hipfire_arch_qwen4::expert_residency::placement_inputs_with_forward(
        &hipfire_arch_qwen4::expert_residency::PlacementInputsArgs {
            gpu,
            config,
            weights,
            bytes_of,
            backend,
            use_ranges,
            max_seq,
            vmm_granularity,
            mtp_kept,
            max_k,
            routed_experts,
            state_format,
            scratch_mode,
            spans,
            chunk_rows,
        },
        share,
    )?;
    Ok((
        inputs.non_expert,
        inputs.layer_experts,
        inputs.reserve,
        inputs.mtp_bytes,
        inputs.gather_bytes,
    ))
}
/// Shared placement mode the metadata-only preload plan and the production load
/// both resolve through the genuine owners: `FixedLayers(n)` is an explicit
/// fixed-N outcome (N0/mid/all), `Auto` is the whole-layer `auto` outcome,
/// `Unified` is the UMA resident outcome. Hotset lineage is reported beside
/// `mode` by the tier-plan callpath, not by a second arm here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Qwen4SharedPlacement {
    FixedLayers(usize),
    Auto,
    Unified,
}

/// One observation from the production placement census and capacity owner.
type Qwen4DecideAuto = (
    u64,
    u64,
    u64,
    hipfire_arch_qwen4::expert_residency::AutoReserve,
    Option<u64>,
    Option<u64>,
);

/// Policy facts already selected by their I/O/admission owners.
struct Qwen4PlacementPolicy {
    band: bool,
    hotset: bool,
    /// The admitted S1 profile's OWN placement charges
    /// ([`hipfire_arch_qwen4::slowlink_admission::SlowlinkProfile::charges`]):
    /// `Some` exactly when a profile was admitted. The profile already sized its
    /// resident layers against them, so the settled placement is owned by it.
    admitted_s1: Option<hipfire_arch_qwen4::expert_residency::PlacementCharges>,
    ep_agreed: Option<usize>,
    use_ranges: bool,
    explicit: Option<hipfire_arch_qwen4::expert_residency::ExpertVramLayers>,
    slowlink_bytes: u64,
    table_geometry: (usize, usize, bool),
}

struct Qwen4DecidedPlacement {
    policy: Option<hipfire_arch_qwen4::expert_residency::ExpertVramLayers>,
    charges: Option<hipfire_arch_qwen4::expert_residency::PlacementCharges>,
    reserve: Option<u64>,
    measured: Option<Qwen4DecideAuto>,
    uma_log: Option<(u64, u64, String, u64, bool)>,
    unset_log: Option<(u64, u64, u64, String, bool)>,
}

/// The existing production policy decision, shared with the metadata probe.
/// Capacity is queried lazily, in the original branch and epoch. Counting an
/// auto split remains a separate step: production must run its existing
/// expert-keep, EP-fit and dense checks before that step.
fn qwen4_decide_placement(
    facts: Qwen4PlacementPolicy,
    routed: impl Fn() -> Result<u64, String>,
    auto_inputs: impl Fn() -> Result<Qwen4DecideAuto, String>,
    uma_refuse: impl Fn() -> bool,
) -> Result<Qwen4DecidedPlacement, String> {
    use hipfire_arch_qwen4::expert_residency as residency;
    const MIB: u64 = 1 << 20;
    let mut result = Qwen4DecidedPlacement {
        policy: None,
        charges: None,
        reserve: None,
        measured: None,
        uma_log: None,
        unset_log: None,
    };
    result.policy = if facts.band {
        if facts.explicit.is_some() {
            return Err(format!(
                "qwen4: {} cannot be used with pipeline parallelism (topology.pp > 1) because pipeline stages keep every expert on the GPU; unset it or set topology.pp = 1",
                residency::EXPERT_VRAM_LAYERS_ENV
            ));
        }
        None
    } else if facts.hotset {
        None
    } else if let Some(charges) = facts.admitted_s1 {
        // The admitted profile places exactly its own resident layers and
        // never re-queries capacity, so no census arm runs to derive the
        // settled margin/stage: the profile's own quote IS the owner.
        result.charges = Some(charges);
        Some(residency::ExpertVramLayers::Auto)
    } else if let Some(layers) = facts.ep_agreed {
        Some(residency::ExpertVramLayers::Layers(layers))
    } else if facts.use_ranges {
        result.charges = Some(residency::PlacementCharges {
            margin: 0,
            stage: 0,
            slowlink: facts.slowlink_bytes,
        });
        if facts.explicit.is_none() {
            let (free, non_expert, _, reserve, ..) = auto_inputs()?;
            let reserve_note = reserve.describe();
            let reserve = reserve.bytes();
            let experts = routed()?;
            let fits = residency::fits_fully_resident(free, non_expert, experts, reserve);
            if !fits && uma_refuse() {
                return Err(format!(
                    "qwen4: unified memory cannot hold the resident load ({} MiB routed experts + {} MiB non-expert weights + {} MiB reserve ({reserve_note}) {} {} MiB free device memory); free host memory, or set memory.oom_guard = off to load anyway",
                    experts / MIB,
                    non_expert / MIB,
                    reserve / MIB,
                    if fits { "fit in" } else { "exceed" },
                    free / MIB
                ));
            }
            result.reserve = Some(reserve);
            result.uma_log = Some((free, non_expert, reserve_note, experts, fits));
        }
        facts.explicit
    } else {
        residency::resolve_expert_vram_layers(facts.explicit, || {
            let inputs = auto_inputs()?;
            let experts = routed()?;
            let (free, non_expert, layer_experts) = (inputs.0, inputs.1, inputs.2);
            let tables = qwen4_expert_table_bytes(false, facts.table_geometry)?;
            let available = free
                .checked_sub(tables)
                .ok_or("qwen4: no device budget after expert tables")?;
            let fit = residency::resident_fit(
                available,
                non_expert,
                experts,
                layer_experts,
                inputs.3.bytes(),
            )
            .map_err(|error| format!("qwen4: {error}"))?;
            result.charges = Some(fit.charges);
            result.reserve = Some(fit.reserve);
            result.unset_log = Some((free, non_expert, experts, inputs.3.describe(), fit.fits));
            result.measured = Some(inputs);
            Ok(fit.fits)
        })?
    };
    Ok(result)
}

/// Inputs the shared placement preview is sized from: the owning census plus
/// the actual capacity/native/policy context both callpaths already hold.
/// Plain data, never a new planner type.
struct Qwen4SharedPlacementInputs {
    stage_enabled: bool,
    num_hidden_layers: usize,
    num_experts: usize,
    moe_layers: usize,
    recipe_is_slab: bool,
    use_ranges: bool,
    slab_stage: bool,
    borrowed_donors: bool,
    whole_first: hipfire_arch_qwen4::expert_hotset::WholeFirstDecision,
    explicit: Option<hipfire_arch_qwen4::expert_residency::ExpertVramLayers>,
    expert_keep: bool,
    non_expert: u64,
    layer_experts: u64,
    reserve_base: u64,
    free_bytes: u64,
}

/// Shared outcome measured with the canonical owners after the shared mutation:
/// `(mode, whole_layers_or_none, moved, routed_device_bytes,
/// host_mapped_bytes, tables_bytes, stage_mirror_bytes)`. `whole_layers` is
/// the applied whole-layer count (`None` for a settled fully-resident outcome
/// that applies no mutation); `moved` is entries the shared mutation moved to
/// host RAM.
type Qwen4SharedPlacementOutcome = (
    Qwen4SharedPlacement,
    Option<usize>,
    usize,
    u64,
    u64,
    u64,
    u64,
);

/// Apply an already resolved whole-layer decision and measure its metadata.
/// No policy or fit decision is made here; `None` leaves residency untouched.
fn qwen4_shared_actual_placement(
    weights: &mut [hipfire_runtime::weight_manifest::WeightEntry],
    bytes_of: &dyn Fn(&hipfire_runtime::weight_manifest::WeightEntry) -> Option<u64>,
    inputs: &Qwen4SharedPlacementInputs,
    vram_layers: Option<usize>,
) -> Result<Qwen4SharedPlacementOutcome, String> {
    use hipfire_arch_qwen4::expert_residency as residency;
    let moved = vram_layers.map_or(0, |layers| {
        qwen4_shared_apply_placement(weights, layers, None)
    });
    let routed = residency::resident_routed_expert_bytes(weights, bytes_of)?;
    let host = residency::host_mapped_bytes(weights, bytes_of)?;
    // Host-mapped MTP experts do not make a trunk stage possible.
    let host_trunk = vram_layers.is_some_and(|layers| layers < inputs.num_hidden_layers);
    let (tables, mirror) = qwen4_shared_table_mirror(inputs, if host_trunk { host } else { 0 })?;
    let mode = if inputs.use_ranges {
        Qwen4SharedPlacement::Unified
    } else {
        match inputs.explicit {
            Some(residency::ExpertVramLayers::Layers(layers)) => {
                Qwen4SharedPlacement::FixedLayers(layers)
            }
            _ => Qwen4SharedPlacement::Auto,
        }
    };
    Ok((mode, vram_layers, moved, routed, host, tables, mirror))
}

/// Canonical auto count: `WholeFirstPlan::auto` under whole-first, else
/// `auto_placement_layers`. Pure; the one owner both callpaths share.
/// Tables ride the conditional stage-payload policy: solve without EXTRA stage
/// tables first, and only when the actual placement needs a stage re-solve with
/// the real extra owned stage charge (the distinct stage pointer tables).
fn qwen4_shared_auto_whole_layers(
    inputs: &Qwen4SharedPlacementInputs,
    charges: &hipfire_arch_qwen4::expert_residency::PlacementCharges,
) -> Result<
    (
        usize,
        Option<hipfire_arch_qwen4::expert_residency::WholeFirstPlan>,
        hipfire_arch_qwen4::expert_residency::PlacementCharges,
    ),
    String,
> {
    use hipfire_arch_qwen4::expert_hotset as hotsets;
    use hipfire_arch_qwen4::expert_residency as residency;
    let geometry = (
        inputs.num_hidden_layers,
        inputs.num_experts,
        inputs.expert_keep,
    );
    let solve = |free: u64,
                 table_stage: bool|
     -> Result<(usize, Option<residency::WholeFirstPlan>), String> {
        let tables = qwen4_expert_table_bytes(table_stage, geometry)?;
        let available = free
            .checked_sub(tables)
            .ok_or("no device budget after expert tables")?;
        if inputs.whole_first == hotsets::WholeFirstDecision::Active {
            let plan = residency::WholeFirstPlan::auto(
                available,
                inputs.non_expert,
                inputs.layer_experts,
                inputs.num_hidden_layers,
                inputs.moe_layers,
                inputs.reserve_base,
                charges,
                inputs.borrowed_donors,
            )?;
            Ok((plan.whole_layers, Some(plan)))
        } else {
            let layers = residency::auto_placement_layers(
                available,
                inputs.non_expert,
                inputs.layer_experts,
                inputs.num_hidden_layers,
                inputs.reserve_base,
                charges,
            )?;
            Ok((layers, None))
        }
    };
    // First solve carries only the base tables and the masks, with the actual
    // `charges` stage payload; no EXTRA stage tables yet.
    let (base_layers, base_plan) = solve(inputs.free_bytes, false)?;
    // Exactly the distinct stage pointer tables (16/expert): the base solve
    // already counts the base tables and the masks once.
    let stage_extra =
        qwen4_expert_table_bytes(true, geometry)? - qwen4_expert_table_bytes(false, geometry)?;
    // A fully resident hot set builds no stage at all, so it pays no EXTRA
    // stage charge; only an actual host/spill placement re-solves.
    let (layers, plan) =
        if base_layers >= inputs.moe_layers || !inputs.stage_enabled || !inputs.slab_stage {
            (base_layers, base_plan)
        } else {
            let (staged_layers, staged_plan) = solve(
                inputs
                    .free_bytes
                    .checked_sub(stage_extra)
                    .ok_or("no device budget after expert tables")?,
                false,
            )?;
            (staged_layers.min(base_layers), staged_plan)
        };
    Ok((layers, plan, *charges))
}

/// Checked tables/mirrors owner: base pointer/codec tables plus the separate
/// stage pointer tables when `host > 0` and staging is enabled, and the keep
/// mask `(layers + 1) * experts` when `expert_keep` is set. Mirrors: the
/// slab rule (0 under a dedicated slab stage, else the staged
/// `PlacementCharges` upper bound) or the packed two-layer bound; 0 when no
/// stage is possible or a tier plan owns the stage.
fn qwen4_shared_table_mirror(
    inputs: &Qwen4SharedPlacementInputs,
    host_mapped_bytes: u64,
) -> Result<(u64, u64), String> {
    use hipfire_arch_qwen4::expert_residency as residency;
    let stage_possible = host_mapped_bytes > 0 && inputs.stage_enabled;
    let tables = qwen4_expert_table_bytes(
        stage_possible,
        (
            inputs.num_hidden_layers,
            inputs.num_experts,
            inputs.expert_keep,
        ),
    )?;
    let mirror = if stage_possible {
        if inputs.recipe_is_slab {
            if inputs.slab_stage {
                0
            } else {
                qwen4_shared_charges(inputs.layer_experts, true, true, 0)?
                    .stage
                    .checked_add(2 * residency::HOST_MAPPED_PAD_BYTES)
                    .ok_or("slab restore upper bound overflows")?
            }
        } else {
            inputs
                .layer_experts
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(4 * residency::HOST_MAPPED_PAD_BYTES))
                .ok_or("packed restore upper bound overflows")?
        }
    } else {
        0
    };
    Ok((tables, mirror))
}

/// Apply whole-layer placement: trunk layers `>= vram_layers` and the MTP layer
/// become `HostMapped`; an override set keeps exactly the listed residents.
/// Reuses the canonical mutation owners; never clones. Returns moved entries.
fn qwen4_shared_apply_placement(
    weights: &mut [hipfire_runtime::weight_manifest::WeightEntry],
    vram_layers: usize,
    resident_set: Option<&[usize]>,
) -> usize {
    match resident_set {
        Some(set) => hipfire_arch_qwen4::expert_residency::place_routed_experts_set(weights, set),
        None => hipfire_arch_qwen4::expert_residency::place_routed_experts(weights, vram_layers),
    }
}

/// Shared charges: `PlacementCharges::new(layer, staged && slab_stage)` plus the
/// S1 slow-link census. Pure; callers add their own `qwen4: ...` prefix.
fn qwen4_shared_charges(
    layer_experts: u64,
    staged: bool,
    slab_stage: bool,
    slowlink_bytes: u64,
) -> Result<hipfire_arch_qwen4::expert_residency::PlacementCharges, String> {
    hipfire_arch_qwen4::expert_residency::PlacementCharges::new(layer_experts, staged && slab_stage)
        .map(|charges| charges.with_slowlink(slowlink_bytes))
}

/// The owning margin/stage charge of a settled single-rank placement. Every
/// placement path records the charge its own owner derived (the hot set's tier
/// plan, the resident-fit census, the explicit/auto census, or an admitted S1
/// profile's quote); a settled placement no owner charged is a refusal, never a
/// zero charge.
fn qwen4_settled_charges(
    settled: Option<hipfire_arch_qwen4::expert_residency::PlacementCharges>,
) -> Result<hipfire_arch_qwen4::expert_residency::PlacementCharges, String> {
    settled.ok_or_else(|| "qwen4: settled placement has no owning margin/stage charge".into())
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
/// ([`open_qwen4_rank_source`]); nothing is allocated. The plan's `layers` is
/// that split. With `batch` asking for lanes (VMM batch, more than one lane)
/// the AR lane store's per-rank device bytes are reserved first (resident
/// layers can only shrink), see
/// [`hipfire_arch_qwen4::expert_residency::ep_vram_plan`]; a default `batch`
/// returns exactly the baseline split.
pub fn qwen4_ep_vram_layers(
    gpu: &rdna_compute::Gpu,
    shared: &Qwen4RankShared,
    spans: &hipfire_arch_qwen4::expert_residency::EpExpertSpans,
    free_per_rank: &[u64],
    src: &ModelSource,
    batch: &hipfire_runtime::loader_api::BatchLoadRequest,
) -> Result<hipfire_arch_qwen4::expert_residency::EpVramPlan, String> {
    let ModelSource::Hfq(hfq) = src else {
        return Err(crate::admission::QWEN4_SAFETENSORS_REFUSAL.into());
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
        .with_expert_parallel(hipfire_arch_qwen4::weights::Qwen4EpPlacement {
            rank: 0,
            ranks: spans.ranks(),
        })
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
    let chunk_rows = hipfire_arch_qwen4::gpu_forward::qwen4_prefill_chunk_requested(
        gpu,
        shared.max_seq,
        scratch_mode,
    );
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
        spans,
        chunk_rows,
        None,
    )?;
    // Same charges as each rank's load applies to an expert-parallel split
    // (no slab stage: `new_slab` has no EP route), so the agreed split never
    // exceeds what a rank's own check keeps.
    let reserve = qwen4_shared_charges(layer_experts, false, false, 0)
        .and_then(|charges| charges.reserve(reserve.bytes()))
        .map_err(|error| format!("qwen4: {error}"))?;
    // The lane store the stage after this load commits on every rank, from
    // the same bytes helper `stage_rank_lanes` uses (same context commit:
    // backend, `max_seq`, requested chunk and VMM granularity).
    let lane = qwen4_ep_lane_request(batch, shared.backend, |lanes, row_budget| {
        hipfire_arch_qwen4::ep::ep_lane_store_device_bytes(
            &config,
            state_format,
            shared.backend,
            shared.max_seq,
            chunk_rows,
            shared.vmm_granularity,
            lanes,
            row_budget,
            None,
        )
    });
    Ok(hipfire_arch_qwen4::expert_residency::ep_vram_plan(
        &config,
        free_per_rank,
        non_expert,
        layer_experts,
        reserve,
        lane,
    ))
}

/// The AR-lane reservation a load's [`BatchLoadRequest`] asks of an
/// expert-parallel split, resolved to one rank's lane-store bytes
/// (`lane_bytes(lanes, row_budget)`, `None` on overflow). `None` when the
/// request is off (no VMM batch, or fewer than two lanes): the split is then
/// the baseline. A KV backend other than VMM keeps the request but with no
/// bytes, because the lane staging refuses that backend whatever is reserved.
/// The staged row budget is [`hipfire_arch_qwen4::ep::ep_lane_row_budget`],
/// the rule the batch staging applies.
fn qwen4_ep_lane_request(
    batch: &hipfire_runtime::loader_api::BatchLoadRequest,
    backend: hipfire_arch_qwen4::Qwen4KvBackend,
    lane_bytes: impl FnOnce(usize, usize) -> Option<u64>,
) -> Option<hipfire_arch_qwen4::expert_residency::EpLaneRequest> {
    let lanes = batch.lanes()?;
    let row_budget = hipfire_arch_qwen4::ep::ep_lane_row_budget(batch.row_budget, lanes);
    let bytes = if backend == hipfire_arch_qwen4::Qwen4KvBackend::Vmm {
        lane_bytes(lanes, row_budget).ok_or_else(|| "lane store device bytes overflow".to_string())
    } else {
        Err("the QSA context backend is not VMM, so lane staging refuses".to_string())
    };
    Some(hipfire_arch_qwen4::expert_residency::EpLaneRequest {
        lanes,
        row_budget,
        bytes,
    })
}

/// Metadata-only preload resource plan for one Qwen4 rank (`Unit B`).
///
/// Source-planned, unmeasured, non-qualifying: admits the HFQ artifact and
/// joins the existing placement/census/host/PLE/staging/runtime authorities
/// into one finite conservative component union. Performs no weight
/// fulfillment, no source symmetry scan, no sidecar verification or hash, no
/// GPU allocation, no fit declaration and no qualification. Missing metadata
/// or any checked-overflow is a named `Err`, never a fabricated zero.
/// `shared.max_seq` is the already-resolved planning context (never an
/// oracle-forced value); `ranks` is always 1.
pub fn qwen4_preload_resource_plan(
    gpu: &rdna_compute::Gpu,
    shared: &Qwen4RankShared,
    hfq: &hipfire_runtime::hfq::HfqFile,
) -> Result<serde_json::Value, String> {
    use hipfire_arch_qwen4::expert_residency as residency;
    let ck = |a: u64, b: u64, what: &str| {
        a.checked_add(b)
            .ok_or_else(|| format!("qwen4 preload plan: {what} overflows"))
    };
    // Admit-only artifact read: config, manifest and recipe. No allocation.
    let receipt = hipfire_arch_qwen4::admit_hfqm_artifact_on(hfq, Some(&gpu.arch))
        .map_err(|error| format!("qwen4 preload plan: artifact admission: {error}"))?;
    let recipe = receipt.recipe().clone();
    let config = receipt.config;
    // Own the admitted metadata for the placement probe below (no weights
    // clone): the shared in-place mutation borrows this owned manifest.
    let mut manifest = receipt.manifest;
    let use_ranges = gpu.is_uma();
    let routed_experts = !config.ffn.is_dense();
    let max_seq = shared.max_seq;
    if max_seq == 0 || max_seq > hipfire_arch_qwen4::QWEN4_MAX_CONTEXT {
        return Err(format!(
            "qwen4 preload plan: max_seq {max_seq} is outside the admitted context range 1..={}",
            hipfire_arch_qwen4::QWEN4_MAX_CONTEXT
        ));
    }
    // Ordinary resolved config: actual backend/state/max_k, never oracle-forced.
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &shared.kv_mode,
        shared.state_quant.as_deref().unwrap_or(""),
        gpu,
        &config,
    )
    .map_err(|error| format!("qwen4 preload plan: state format: {error}"))?;
    // Metadata-only sizes: index `data_size` through the slab plan, never
    // `tensor_data` (no payload access).
    let bytes_of = |entry: &hipfire_runtime::weight_manifest::WeightEntry| {
        crate::qwen4_slab::qwen4_entry_bytes(&recipe, entry, |entry| {
            hfq.find_tensor_info(&entry.name)
                .map(|info| info.data_size as u64)
        })
    };
    // Conservative supported scratch-mode union without the payload scan:
    // `Full` always; the compact route only where its device/flag/profile
    // terms already allow it. The selected route stays unknown here, so the
    // chunk the census, PLE staging and pool workspace are sized from is the
    // largest default among the possible modes (`chunk_rows`); each mode's
    // own scratch bytes are taken at its own default chunk.
    use hipfire_arch_qwen4::gpu_forward::Qwen4ForwardScratchMode;
    let compact_possible =
        hipfire_arch_qwen4::gpu_forward::qwen4_compact_scratch_possible(gpu, &config);
    let full_chunk = hipfire_arch_qwen4::gpu_forward::qwen4_prefill_chunk_requested(
        gpu,
        max_seq,
        Qwen4ForwardScratchMode::Full,
    );
    let compact_chunk = hipfire_arch_qwen4::gpu_forward::qwen4_prefill_chunk_requested(
        gpu,
        max_seq,
        Qwen4ForwardScratchMode::FnSymGfx1201,
    );
    let chunk_rows = if compact_possible {
        full_chunk.max(compact_chunk)
    } else {
        full_chunk
    };
    if chunk_rows == 0 {
        return Err("qwen4 preload plan: requested prefill chunk is zero".into());
    }
    let full_scratch = hipfire_arch_qwen4::gpu_forward::qwen4_forward_device_bytes(
        &config,
        full_chunk,
        Qwen4ForwardScratchMode::Full,
    )
    .ok_or("qwen4 preload plan: full scratch has no checked byte authority")?;
    let compact_scratch = if compact_possible {
        Some(
            hipfire_arch_qwen4::gpu_forward::qwen4_forward_device_bytes(
                &config,
                compact_chunk,
                Qwen4ForwardScratchMode::FnSymGfx1201,
            )
            .ok_or("qwen4 preload plan: compact scratch has no checked byte authority")?,
        )
    } else {
        None
    };
    let scratch_union_bytes =
        compact_scratch.map_or(full_scratch, |compact| full_scratch.max(compact));
    // Effective ordinary MTP resolution with the load's own private resolvers,
    // from shared.spec and the actual gpu/path (single-rank pp=tp=ep=1) —
    // never oracle-forced. shared.native_mtp/max_k must themselves be
    // ordinary-resolved (C populates spec from the runtime config); the
    // effective values below are what the plan charges and reports.
    let effective_native_mtp =
        crate::admission::qwen4_native_mtp(shared.spec, &gpu.arch, &shared.path, 1, 1, 1)
            .and_then(|native| {
                crate::admission::qwen4_native_mtp_for_head(
                    native,
                    shared.spec.mtp,
                    config.mtp.is_some(),
                    config.profile.id(),
                )
            })
            .map_err(|error| format!("qwen4 preload plan: native MTP: {error}"))?;
    let effective_max_k = qwen4_native_mtp_k(shared.spec, &gpu.arch);
    // Placement census at the requested chunk (same convention as the load).
    let mtp_kept = effective_native_mtp
        && crate::admission::qwen4_mtp_with_host_mapped_experts(
            shared.spec,
            &gpu.arch,
            usize::from(routed_experts),
        );
    let inputs = residency::placement_inputs(&residency::PlacementInputsArgs {
        gpu,
        config: &config,
        weights: &manifest.weights,
        bytes_of: &bytes_of,
        backend: shared.backend,
        use_ranges,
        max_seq,
        vmm_granularity: shared.vmm_granularity,
        mtp_kept,
        max_k: effective_max_k,
        routed_experts,
        state_format,
        scratch_mode: Qwen4ForwardScratchMode::Full,
        spans: &residency::EpExpertSpans::single(),
        chunk_rows,
    })
    .map_err(|error| format!("qwen4 preload plan: placement census: {error}"))?;
    // Whole-layer margin/stage charges; S1 slow-link is NOT selected here
    // (its selector needs the host-link probe and the payload scan, both
    // forbidden in metadata-only planning). Report the incumbent union with
    // an explicit reason, never a forced S1 zero presented as selected.
    let slab_stage = residency::dedicated_slab_stage(
        residency::expert_stage_placement_enabled(&gpu.arch),
        matches!(
            recipe,
            hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. }
        ),
        false,
    );
    // S1 slow-link union under the existing legal conditions: probe-free
    // `preflight` first; where it passes, `single_device_facts` (no payload
    // scan: the scan closure answers `Ok(false)`, so the layout route holds)
    // plus `load_admission_inputs` and `project_at_tile` per admissible route
    // (layout, and `AggregateSymIu4` where the layout is the stacked
    // aggregate one: the conservative sym-mode bound without selecting it).
    // The union charge is the max over projecting routes; the link stays
    // unmeasured in planning, so no profile is admitted and no resident-layer
    // verdict is read. A legal decline carries its reason with no S1 charge —
    // a declined union, never a forced zero presented as selected.
    let (s1_slowlink_charge_bytes, s1_note, s1_value) = preload_s1_union(
        gpu,
        &config,
        &recipe,
        &manifest.weights,
        &bytes_of,
        shared.backend,
        max_seq,
        shared.vmm_granularity,
        effective_native_mtp,
        mtp_kept,
        effective_max_k,
        state_format,
        routed_experts,
        use_ranges,
        &shared.path,
    )?;
    // Shared cohesive metadata: fixed-N / auto / hotset / native intent resolved
    // through the genuine owners before byte accounting, with no fake admission.
    // Fresh observed instants (inputs, never predictions): the auto query epoch
    // needs an observed free-VRAM quote.
    let fresh_pool_bytes = u64::try_from(gpu.pool_parked_bytes()).unwrap_or(u64::MAX);
    let fresh_free_device = gpu
        .device_mem_info()
        .map(|(free, _)| u64::try_from(free).unwrap_or(u64::MAX))
        .ok();
    let mem_available = rdna_compute::kv_slots::mem_available_bytes();
    let ttm_pool = residency::ttm_pool_estimate_on(gpu);
    let preallocation_gtt = residency::gtt_budget();
    let hotset_policy_for_plan = hipfire_arch_qwen4::expert_hotset::HotsetPolicy::from_env()
        .map_err(|error| format!("qwen4 preload plan: hotset policy unreadable: {error}"))?;
    let placement_policy_for_plan = hipfire_arch_qwen4::expert_hotset::PlacementPolicy::from_env()
        .map_err(|error| format!("qwen4 preload plan: placement policy unreadable: {error}"))?;
    let hotset_budget_for_plan = hipfire_arch_qwen4::expert_hotset::HotsetBudget::from_env()
        .map_err(|error| format!("qwen4 preload plan: hotset budget unreadable: {error}"))?;
    let explicit_for_plan = residency::expert_vram_layers_from_env().map_err(|error| {
        format!("qwen4 preload plan: explicit expert layers unreadable: {error}")
    })?;
    let hotset_env_for_plan = |explicit: bool| hipfire_arch_qwen4::expert_hotset::HotsetEnv {
        gpu_arch: &gpu.arch,
        unified_memory: use_ranges,
        expert_parallel: false,
        explicit_vram_layers: explicit,
        source: match &recipe {
            hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextPacked { .. } => {
                hipfire_arch_qwen4::expert_hotset::HotsetSource::Aggregate
            }
            hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. } => {
                hipfire_arch_qwen4::expert_hotset::HotsetSource::SlabArena
            }
            hipfire_arch_qwen4::weights::Qwen4TierRecipe::StockFp8 => {
                hipfire_arch_qwen4::expert_hotset::HotsetSource::StockFp8
            }
            _ => hipfire_arch_qwen4::expert_hotset::HotsetSource::Nonaggregate(
                "not an aggregate MQ4 artifact".into(),
            ),
        },
    };
    let whole_first_for_plan = hipfire_arch_qwen4::expert_hotset::decide_whole_first(
        placement_policy_for_plan,
        &hotset_env_for_plan(explicit_for_plan.is_some()),
        false,
    );
    // Fixed-N / auto / unified resolution inputs: actual context/native/capacity
    // inputs, genuine owner hotset metadata reader below, never fake admission.
    // The auto query epoch needs an observed free-VRAM quote: absent is a named
    // error, never an invented zero (payload-only accounting would understate).
    let moe_layers_for_plan = residency::moe_layer_count(&config);
    let free_for_plan = fresh_free_device.ok_or(
        "qwen4 preload plan: device free memory unreadable (no zero assumed); retry with the device queryable",
    )?;
    // Genuine owner hotset metadata reader: HotsetBase::from_hfq (index-span
    // only, no payload) + HotsetSidecar::load_bound + slabs +
    // plan_expert_tiers_from_slabs with the actual inputs. A fixed budget fails
    // closed; otherwise a refusal keeps whole-layer placement. Whole-first
    // active never consults the sidecar.
    let hotset_tier_for_plan = if whole_first_for_plan
        == hipfire_arch_qwen4::expert_hotset::WholeFirstDecision::Active
    {
        if hotset_budget_for_plan.is_fixed() {
            return Err(format!(
                "qwen4 preload plan: {} requires an applicable ranked hotset, but {}=whole-first keeps whole-layer placement",
                hipfire_arch_qwen4::expert_hotset::HOT_BUDGET_MIB_ENV,
                hipfire_arch_qwen4::expert_hotset::PLACEMENT_POLICY_ENV,
            ));
        }
        None
    } else {
        match hipfire_arch_qwen4::expert_hotset::preflight(
            &hotset_policy_for_plan,
            std::path::Path::new(&shared.path),
            &hotset_env_for_plan(explicit_for_plan.is_some()),
        ) {
            Ok(path) => {
                let prepare = || -> Result<_, String> {
                    let base = hipfire_arch_qwen4::expert_hotset::HotsetBase::from_hfq(hfq)
                        .map_err(|error| format!("qwen4 preload plan: hotset base: {error}"))?;
                    let sidecar = hipfire_arch_qwen4::expert_hotset::HotsetSidecar::load_bound(
                        &path,
                        config.num_hidden_layers,
                        config.num_experts,
                        config.num_experts_per_tok,
                        &base,
                    )
                    .map_err(|error| format!("qwen4 preload plan: hotset sidecar: {error}"))?;
                    manifest
                        .check_expert_tiers()
                        .map_err(|error| format!("qwen4 preload plan: hotset tiers: {error}"))?;
                    let slabs = hipfire_arch_qwen4::artifact::hotset_aggregate_slabs(
                        hfq, &config, &manifest,
                    )
                    .map_err(|error| format!("qwen4 preload plan: hotset slabs: {error}"))?;
                    let total = hipfire_arch_qwen4::expert_hotset::require_budget(
                        residency::expert_hotset_budget(
                            free_for_plan,
                            inputs.non_expert,
                            inputs.reserve.bytes(),
                        ),
                    )
                    .map_err(|skip| format!("qwen4 preload plan: hotset budget: {skip}"))?;
                    let stage_enabled = residency::expert_stage_placement_enabled(&gpu.arch);
                    let (plan, budget, stage) = qwen4_plan_hotset_placement(
                        &slabs,
                        sidecar.rank(),
                        hotset_budget_for_plan,
                        total,
                        stage_enabled,
                        (
                            config.num_hidden_layers,
                            config.num_experts,
                            config.expert_keep.is_some(),
                        ),
                        // The preload plan sizes the forward at `Full` scratch
                        // (its scratch mode is not decided yet), which decode-share
                        // refuses, so it never predicts a donor tail.
                        None,
                        &[],
                    )
                    .map_err(|error| format!("qwen4 preload plan: hotset plan: {error}"))?;
                    plan.check_executable(&gpu.arch)
                        .map_err(|error| format!("qwen4 preload plan: hotset plan: {error}"))?;
                    let host_bytes = plan
                        .owner_bytes(hipfire_arch_qwen4::expert_tier_plan::Placement::HostMapped)
                        .checked_add(residency::hotset_host_mapped_bytes(
                            &manifest.weights,
                            &bytes_of,
                            config.num_hidden_layers,
                        )?)
                        .ok_or("qwen4 preload plan: host byte accounting overflow")?;
                    residency::check_host_ram(host_bytes, mem_available, ttm_pool)?;
                    residency::check_gtt_cap(host_bytes, preallocation_gtt)?;
                    let (latest_free, _) = gpu
                        .device_mem_info()
                        .map_err(|error| format!("memory re-query before materialize: {error}"))?;
                    let latest_budget = hipfire_arch_qwen4::expert_hotset::require_budget(
                        residency::expert_hotset_budget(
                            latest_free as u64,
                            inputs.non_expert,
                            inputs.reserve.bytes(),
                        )
                        .and_then(|bytes| {
                            let extra_tables = qwen4_expert_table_bytes(
                                stage_enabled
                                    && plan.owner_bytes(
                                        hipfire_arch_qwen4::expert_tier_plan::Placement::HostMapped,
                                    ) > 0,
                                (
                                    config.num_hidden_layers,
                                    config.num_experts,
                                    config.expert_keep.is_some(),
                                ),
                            )? - residency::EXPERT_TIER_TABLE_BYTES;
                            bytes
                                .checked_sub(stage)
                                .and_then(|bytes| bytes.checked_sub(extra_tables))
                                .ok_or_else(|| {
                                    "no budget after long-prefill stage/table owners".to_string()
                                })
                        }),
                    )
                    .map_err(|skip| skip.to_string())?;
                    hotset_budget_for_plan.select(latest_budget)?;
                    if plan.owner_bytes(hipfire_arch_qwen4::expert_tier_plan::Placement::Device)
                        > latest_budget
                    {
                        return Err(format!(
                        "free VRAM dropped to {} MiB before the experts were placed; the planned GPU-resident experts need more than the remaining {} MiB. Free GPU memory or place fewer experts on the GPU",
                        latest_free as u64 / (1 << 20), latest_budget / (1 << 20),
                    ));
                    }
                    Ok((plan, budget, stage, host_bytes))
                };
                match prepare() {
                    Ok((plan, budget, stage, host_bytes)) => {
                        Some((path, plan, budget, stage, host_bytes))
                    }
                    Err(reason) => {
                        if hotset_budget_for_plan.is_fixed() {
                            return Err(format!(
                                "qwen4: {} could not be applied: {reason}",
                                hipfire_arch_qwen4::expert_hotset::HOT_BUDGET_MIB_ENV,
                            ));
                        }
                        eprintln!(
                            "  qwen4 expert hot set: {} present, not applied ({reason}); whole-layer placement",
                            path.display(),
                        );
                        None
                    }
                }
            }
            Err(skip) => {
                if hotset_budget_for_plan.is_fixed() {
                    return Err(format!(
                        "qwen4 preload plan: {} could not be applied: {skip}",
                        hipfire_arch_qwen4::expert_hotset::HOT_BUDGET_MIB_ENV,
                    ));
                }
                None
            }
        }
    };
    // The SAME shared decision production resolves: full precedence through
    // the genuine owners (band/hotset/S1-unadmitted/EP-agreed(expert_parallel
    // false here)/UMA/unset-or-explicit), with the observed quotes injected
    // in owner order. Unknown S1 stays named unadmitted (no profile or
    // resident-layer verdict); ordinary auto uses its own lazy capacity epoch.
    if !routed_experts {
        residency::refuse_host_expert_request(explicit_for_plan, config.profile.id())
            .map_err(|error| format!("qwen4 preload plan: {error}"))?;
    }
    let auto_inputs_for_plan = || -> Result<Qwen4DecideAuto, String> {
        let (free, _) = gpu
            .device_mem_info()
            .map_err(|error| format!("qwen4: device memory query: {error}"))?;
        Ok((
            free as u64,
            inputs.non_expert,
            inputs.layer_experts,
            inputs.reserve,
            inputs.mtp_bytes,
            inputs.gather_bytes,
        ))
    };
    let mut decided_for_plan = qwen4_decide_placement(
        Qwen4PlacementPolicy {
            band: false,
            hotset: hotset_tier_for_plan.is_some(),
            admitted_s1: None,
            ep_agreed: None,
            use_ranges,
            explicit: explicit_for_plan,
            slowlink_bytes: 0,
            table_geometry: (
                config.num_hidden_layers,
                config.num_experts,
                config.expert_keep.is_some(),
            ),
        },
        || {
            residency::routed_expert_bytes(&manifest.weights, &bytes_of)
                .map_err(|error| format!("qwen4: {error}"))
        },
        &auto_inputs_for_plan,
        || {
            state_format.qsa_f16_arena
                || state_format.qsa_pooled_bf16
                || hipfire_config::oom_guard_effective(Some(&gpu.arch))
        },
    )
    .map_err(|error| format!("qwen4 preload plan: {error}"))?;
    // Derive the measurement inputs FROM the shared decision (never a
    // parallel computation): policy lineage selects the count path, the
    // settled charges/reserve flow into the union, and the S1-adjusted
    // payload is unavailable outside the auto arm (tiers handle their own
    // lineage below).
    let mut shared_inputs = Qwen4SharedPlacementInputs {
        // Component envelope only: without measuring the ordinary host link,
        // include the possible borrowed-stage owner unless explicitly off.
        // This does not select a stage or enter placement's charged reserve.
        stage_enabled: hipfire_config::developer_var(
            hipfire_arch_qwen4::gpu_forward::QWEN4_EXPERT_STAGE_ENV,
        )
        .ok()
        .as_deref()
            != Some("0"),
        num_hidden_layers: config.num_hidden_layers,
        num_experts: config.num_experts,
        moe_layers: moe_layers_for_plan,
        recipe_is_slab: matches!(
            recipe,
            hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. }
        ),
        use_ranges,
        slab_stage,
        borrowed_donors: residency::expert_stage_placement_enabled(&gpu.arch)
            && matches!(
                recipe,
                hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextPacked { .. }
            ),
        whole_first: whole_first_for_plan,
        explicit: explicit_for_plan,
        expert_keep: config.expert_keep.is_some(),
        non_expert: inputs.non_expert,
        layer_experts: inputs.layer_experts,
        reserve_base: inputs.reserve.bytes(),
        free_bytes: free_for_plan,
    };
    // Owned-metadata probe: the shared in-place mutation borrows the owned
    // admitted manifest, never a weights clone. Order matches production
    // (`preload_component_union` below recomputes the census from these same
    // residency-insensitive owners), so no stale-census fabrication.
    if config.expert_keep.is_some() && decided_for_plan.policy.is_some() {
        return Err("qwen4.expert_keep artifacts require fully resident single-device expert payloads; host-mapped/streamed expert placement is not supported (unset HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS and ensure the kept payload fits)".into());
    }
    if !routed_experts && decided_for_plan.policy.is_some() {
        return Err(format!(
            "qwen4 preload plan: profile {} has a dense FFN (no routed experts to move to host RAM) and its resident weights do not fit device memory",
            config.profile.id()
        ));
    }
    if use_ranges && decided_for_plan.policy.is_some() {
        return Err(format!(
            "qwen4: {} puts experts in host RAM, which only a discrete GPU supports; unset it on this unified-memory device",
            residency::EXPERT_VRAM_LAYERS_ENV
        ));
    }
    let (resolved_layers, settled_charges) = match decided_for_plan.policy {
        Some(residency::ExpertVramLayers::Layers(layers)) => (
            Some(layers),
            Some(qwen4_shared_charges(
                inputs.layer_experts,
                layers < config.num_hidden_layers,
                slab_stage,
                0,
            )?),
        ),
        Some(residency::ExpertVramLayers::Auto) => {
            let quote = match decided_for_plan.measured.take() {
                Some(quote) => quote,
                None => auto_inputs_for_plan()?,
            };
            shared_inputs.free_bytes = quote.0;
            shared_inputs.non_expert = quote.1;
            shared_inputs.layer_experts = quote.2;
            shared_inputs.reserve_base = quote.3.bytes();
            let charges = qwen4_shared_charges(inputs.layer_experts, true, slab_stage, 0)?;
            let (layers, _, charges) = qwen4_shared_auto_whole_layers(&shared_inputs, &charges)?;
            (Some(layers), Some(charges))
        }
        None => (None, decided_for_plan.charges),
    };
    // Ordinary placement never treats an unknown S1 projection as admitted.
    // The component union below separately retains the S1 resource envelope.
    let charges = match &hotset_tier_for_plan {
        Some((_, _, _, stage, _)) => residency::PlacementCharges {
            margin: residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES,
            stage: *stage,
            slowlink: 0,
        },
        None => settled_charges.unwrap_or(qwen4_shared_charges(
            inputs.layer_experts,
            false,
            slab_stage,
            0,
        )?),
    }
    .with_slowlink(s1_slowlink_charge_bytes);
    // Snapshot the pre-mutation census the union re-verifies below: the shared
    // mutation runs before `preload_component_union` recomputes the same
    // owners, and the owners below show the recompute is mutation-insensitive
    // (external/tied/routed-name filtering, never a placed-residency count).
    let census_non_expert = inputs.non_expert;
    let census_reserve = inputs.reserve;
    let (shared_tier_device, shared_tier_host, shared_tier_stage) = match &hotset_tier_for_plan {
        Some((_, plan, _, stage, _)) => {
            use hipfire_arch_qwen4::expert_tier_plan::Placement;
            (
                plan.owner_bytes(Placement::Device),
                plan.owner_bytes(Placement::HostMapped),
                *stage,
            )
        }
        None => (0, 0, 0),
    };
    let (
        _shared_mode,
        _shared_whole,
        _shared_moved,
        routed_device_bytes,
        host_mapped_bytes,
        tables_bytes,
        stage_mirror_bytes,
    ) = if hotset_tier_for_plan.is_some() {
        // Exactly the production metadata cutover: tier owners replace the
        // trunk aggregates, while MTP entries follow ordinary host placement.
        qwen4_shared_apply_placement(&mut manifest.weights, config.num_hidden_layers, None);
        manifest = manifest
            .with_expert_tiers()
            .map_err(|error| error.to_string())?;
        let mut routed = residency::resident_routed_expert_bytes(&manifest.weights, &bytes_of)?;
        let mut host = residency::host_mapped_bytes(&manifest.weights, &bytes_of)?;
        routed = routed
            .checked_add(shared_tier_device)
            .ok_or("qwen4 preload plan: routed owner union overflows")?;
        host = host
            .checked_add(shared_tier_host)
            .ok_or("qwen4 preload plan: host owner union overflows")?;
        // Tier stages never borrow device donors or allocate restores.
        let tables = qwen4_expert_table_bytes(
            shared_tier_host > 0 && shared_inputs.stage_enabled,
            (
                config.num_hidden_layers,
                config.num_experts,
                config.expert_keep.is_some(),
            ),
        )?;
        let _ = shared_tier_stage;
        (
            Qwen4SharedPlacement::Auto,
            None,
            0usize,
            routed,
            host,
            tables,
            0u64,
        )
    } else {
        qwen4_shared_actual_placement(
            &mut manifest.weights,
            &bytes_of,
            &shared_inputs,
            resolved_layers,
        )
        .map_err(|error| format!("qwen4 preload plan: {error}"))?
    };
    // Host cache: 256MiB PLE userspace page cache; RAM-only page cache stays
    // separate from pinned/GTT demand in the whole-load join below.
    let page_cache_bytes = hipfire_runtime::external_rows::PAGE_CACHE_BYTES as u64;
    // PLE rows: the eligible data section, never the file extent. Source
    // classification from the artifact metadata, not from the attached
    // handle alone:
    // - no `config.ple`: legitimately PLE-0;
    // - sidecar attached (`hfq.ple_rows()`): header `data_bytes` envelope;
    // - otherwise (embedded or C-attached-nothing): size from the manifest's
    //   `ExternalRows` shard declarations — `valid_rows * row_bytes` per
    //   shard, checked. A nonzero profile with no shards and no attached
    //   sidecar is a named missing failure, never a silent zero. External
    //   shards stay file-backed: they enter only the PLE terms below, never
    //   the weight owners. Staging uses the same assembly inputs as the load.
    let (ple_source_label, ple_data_bytes) = match (&config.ple, hfq.ple_rows()) {
        (None, _) => ("absent", 0),
        (Some(_), Some(rows)) => {
            let header = rows.header();
            (
                "sidecar-attached",
                preload_ple_envelope(header.data_bytes, header.data_offset)?,
            )
        }
        (Some(ple), None) => {
            let mut total = 0u64;
            let mut shards = 0usize;
            for entry in manifest.weights.iter() {
                let hipfire_runtime::weight_manifest::WeightResidency::ExternalRows {
                    row_bytes,
                    valid_rows,
                } = entry.residency
                else {
                    continue;
                };
                // Reach bound, stated honestly: the eligible data rows times
                // the declared row stride. Physical padding past `valid_rows`
                // is not resident data and is not charged.
                let shard = (valid_rows as u64)
                    .checked_mul(row_bytes as u64)
                    .ok_or("qwen4 preload plan: PLE embedded shard bytes overflow")?;
                total = total
                    .checked_add(shard)
                    .ok_or("qwen4 preload plan: PLE embedded data union overflows")?;
                shards += 1;
            }
            if shards == 0 {
                return Err("qwen4 preload plan: PLE rows required by artifact metadata (nonzero profile) but no embedded shards found and no sidecar attached; C must attach the ordinary selected sidecar header before calling the plan"
                    .into());
            }
            if shards != ple.shard_count {
                return Err(format!(
                    "qwen4 preload plan: PLE embedded shard count {shards} disagrees with profile shard_count {}",
                    ple.shard_count
                ));
            }
            ("embedded-manifest", total)
        }
    };
    let (ple_row_width, ple_head_count) = config
        .ple
        .as_ref()
        .map_or((0usize, 0usize), |ple| (ple.row_width, ple.head_count));
    let decoded_row_bytes = (ple_row_width as u64)
        .checked_mul(2)
        .ok_or("qwen4 preload plan: PLE decoded row bytes overflow")?;
    if ple_data_bytes > 0 && decoded_row_bytes == 0 {
        return Err("qwen4 preload plan: PLE rows sized but row geometry missing".into());
    }
    let ple_staging_upload_bytes = if ple_data_bytes > 0 && decoded_row_bytes > 0 {
        let staging_rows = chunk_rows
            .checked_mul(ple_head_count)
            .ok_or("qwen4 preload plan: PLE staging rows overflow")?;
        let (_, two_buffers) = preload_staging_bytes(staging_rows, decoded_row_bytes)?;
        two_buffers
    } else {
        0
    };
    // Load-upload transient: applicable finite branch only (direct ring vs
    // range fallback), never an extra copy of final host destinations.
    let direct_branch =
        !use_ranges && hipfire_config::memory::qwen4_direct_load() && !hfq.has_overlay();
    let staging_upload_bytes = if direct_branch {
        2 * hipfire_runtime::hfq_parallel::HFQ_READER_LANES as u64 * 16 * 1024 * 1024
    } else {
        hipfire_runtime::weight_store::RANGE_UPLOAD_CHUNK_BYTES as u64
    };
    let gtt_free =
        preallocation_gtt.map(|budget| budget.limit_bytes.saturating_sub(budget.used_bytes));
    let session_cache_configured = hipfire_config::memory::session_cache_bytes();
    // Device allocation granularity: required source input. A failed query
    // is a named error, never an invented zero (payload-only accounting
    // would silently understate every rounded owner).
    let granule_bytes = gpu
        .vmm_recommended_granularity()
        .map(|granule| granule as u64)
        .map_err(|error| format!("qwen4 preload plan: device allocation granularity: {error}"))?;
    // placement-input owners, rounding, native phases, runtime allowance,
    // initial pool/host/TTM terms. B supplies the checked scalars only and
    // embeds the returned union value — no second union here.
    let scratch_label: &str = if compact_possible {
        "Full+FnSymGfx1201-supported-union;selected-unknown"
    } else {
        "Full"
    };
    let s1_note = format!(
        "{s1_note}; residency is the ordinary production placement, not an admitted S1 placement; ordinary host link is unmeasured, so tables/restores include the possible borrowed-stage envelope unless explicitly disabled"
    );
    let component_union = hipfire_arch_qwen4::measured_resources::preload_component_union(
        &residency::PlacementInputsArgs {
            gpu,
            config: &config,
            weights: &manifest.weights,
            bytes_of: &bytes_of,
            backend: shared.backend,
            use_ranges,
            max_seq,
            vmm_granularity: shared.vmm_granularity,
            mtp_kept,
            max_k: effective_max_k,
            routed_experts,
            state_format,
            scratch_mode: Qwen4ForwardScratchMode::Full,
            spans: &residency::EpExpertSpans::single(),
            chunk_rows,
        },
        charges,
        census_reserve,
        census_non_expert,
        routed_device_bytes,
        host_mapped_bytes,
        tables_bytes,
        stage_mirror_bytes,
        fresh_pool_bytes,
        fresh_free_device,
        mem_available,
        ttm_pool,
        gtt_free,
        granule_bytes,
        scratch_label,
        &s1_note,
    )
    .map_err(|error| format!("qwen4 preload plan: {error}"))?;
    // Whole-load checked join, extracted for same-file arithmetic tests: A's
    // host union plus B's explicit PLE/session/upload terms, then the
    // check_host_ram-faithful admission figures. UMA device-backed already
    // counted exactly once inside A, never re-added.
    let (whole_host_union_bytes, whole_load_join) = preload_whole_host_join(
        &component_union,
        use_ranges,
        ple_data_bytes,
        ple_staging_upload_bytes,
        page_cache_bytes,
        staging_upload_bytes,
        session_cache_configured,
        mem_available,
        ttm_pool,
    )?;
    let mut plan = serde_json::json!({
        "kind": "qwen4_preload_resource_plan",
        "mode": "source-planned",
        "unmeasured": true,
        "nonqualification": true,
        "context": { "max_seq": max_seq, "chunk_rows": chunk_rows, "ranks": 1,
            "backend": shared.backend.name(), "state_qsa": state_format.qsa.name(),
            "state_gdn": state_format.gdn.name(),
            "native_mtp": effective_native_mtp, "native_mtp_shared": shared.native_mtp,
            "mtp_kept": mtp_kept, "max_k": effective_max_k, "max_k_shared": shared.max_k,
            "mtp_note": "effective values resolved from shared.spec with the actual gpu/path (single-rank pp=tp=ep=1); shared carries the ordinary-resolved request, never an oracle-forced value",
            "routed_experts": routed_experts, "use_ranges": use_ranges,
            "granule_bytes": granule_bytes, "granule_source": "device-query" },
        "ple": { "source": ple_source_label,
            "data_bytes": ple_data_bytes, "resident_envelope_bytes": ple_data_bytes,
            "row_width": ple_row_width, "head_count": ple_head_count,
            "staging_upload_bytes": ple_staging_upload_bytes,
            "page_cache_bytes": page_cache_bytes,
            "note": "auto-prewarm later sample unknown: envelope is the header data section for sidecars, manifest eligible rows for embedded; never the file extent" },
        "load_upload_transient_bytes": staging_upload_bytes,
        "session_cache_configured_bytes": session_cache_configured,
        "fresh": { "pool_parked_bytes": fresh_pool_bytes,
            "free_device_bytes": fresh_free_device, "mem_available_bytes": mem_available,
            "ttm_pool_bytes": ttm_pool, "gtt_free_bytes": gtt_free },
        "component_union": component_union,
        "whole_load_join": whole_load_join,
    });
    // Surface the one figure operators compare against capacity: the joined
    // whole-load host union beside the original fresh quotes.
    plan["whole_load_host_union_bytes"] = serde_json::json!(whole_host_union_bytes);
    Ok(plan)
}

/// Whole-load host report, all checked. Source classification, from what each
/// term is backed by:
/// - pinned/GTT demand (unmovable, `check_host_ram`-relevant): A's host union
///   (pinned host-mapped experts + stage mirrors + UMA-once device-backed),
///   plus B's pinned userspace staging (PLE row-store buffers, direct upload
///   ring or range chunk). UMA device-backed memory was already counted
///   exactly once inside A and is never re-added here.
/// - RAM-only terms (reclaimable page cache / host capacity, never pinned
///   demand): the PLE eligible data-section resident envelope, the 256MiB
///   page cache, and the session-cache configured cap. Reported alongside,
///   never folded into the pinned demand.
/// Capacity is an observed optional quote: `None` serializes as JSON null
/// with a named unavailable reason (an observed quote may be absent); `Some`
/// joins TTM with a CHECKED add (overflow is `Err`, never saturating).
/// Returns `(whole_host_union_bytes, report_value)` — the union of pinned
/// demand plus the separately-labelled RAM-only terms. Plain tuple, never a
/// new planner type.
#[allow(clippy::too_many_arguments)]
fn preload_whole_host_join(
    component_union: &serde_json::Value,
    use_ranges: bool,
    ple_resident_envelope_bytes: u64,
    ple_staging_upload_bytes: u64,
    page_cache_bytes: u64,
    load_upload_transient_bytes: u64,
    session_cache_configured_bytes: u64,
    mem_available_bytes: Option<u64>,
    ttm_pool_bytes: u64,
) -> Result<(u64, serde_json::Value), String> {
    let num = |field: &str| {
        component_union
            .get(field)
            .and_then(|bytes| bytes.as_u64())
            .ok_or_else(|| format!("qwen4 preload plan: component union has no numeric '{field}'"))
    };
    let a_host_union = num("host_union_upper_bound_bytes")?;
    let a_device_union = num("device_union_upper_bound_bytes")?;
    let a_device_backed = num("device_backed_host_bytes")?;
    let a_uma = component_union
        .get("uma_shared_backing")
        .and_then(|uma| uma.as_bool())
        .ok_or_else(|| {
            "qwen4 preload plan: component union has no bool 'uma_shared_backing'".to_string()
        })?;
    if a_uma != use_ranges {
        return Err("qwen4 preload plan: component union backing disagrees with this rank".into());
    }
    let ck = |a: u64, b: u64, what: &str| {
        a.checked_add(b)
            .ok_or_else(|| format!("qwen4 preload plan: {what} overflows"))
    };
    // Pinned/GTT demand: A's host union plus B's pinned userspace staging
    // (row-store buffers, upload transient). Reclaimable PLE data-section
    // pages, the page cache and the session cap are RAM-only and stay out.
    let mut pinned_gtt_demand_bytes = a_host_union;
    pinned_gtt_demand_bytes = ck(
        pinned_gtt_demand_bytes,
        ple_staging_upload_bytes,
        "pinned demand with PLE staging",
    )?;
    pinned_gtt_demand_bytes = ck(
        pinned_gtt_demand_bytes,
        load_upload_transient_bytes,
        "pinned demand with upload transient",
    )?;
    // RAM-only terms: reported alongside, never folded into pinned demand.
    let mut ram_only_host_bytes = ple_resident_envelope_bytes;
    ram_only_host_bytes = ck(
        ram_only_host_bytes,
        page_cache_bytes,
        "RAM-only host with page cache",
    )?;
    ram_only_host_bytes = ck(
        ram_only_host_bytes,
        session_cache_configured_bytes,
        "RAM-only host with session cache cap",
    )?;
    let whole_host = ck(
        pinned_gtt_demand_bytes,
        ram_only_host_bytes,
        "whole host with RAM-only terms",
    )?;
    // Admission figures faithful to `check_host_ram`: the pinned/GTT demand,
    // the 4GiB headroom as an explicit separate allowance, and the observed
    // capacity quote (MemAvailable may be absent: JSON null + named reason).
    let headroom_bytes = hipfire_arch_qwen4::expert_residency::HOST_RAM_HEADROOM_BYTES;
    let admission_needed_bytes = ck(
        pinned_gtt_demand_bytes,
        headroom_bytes,
        "host admission charge with headroom",
    )?;
    let (admission_capacity_bytes, capacity_note) = match mem_available_bytes {
        Some(available) => (
            serde_json::json!(available
                .checked_add(ttm_pool_bytes)
                .ok_or("qwen4 preload plan: host admission capacity overflows")?),
            "MemAvailable plus TTM pool (observed reclaimable pages, never a planned allocation); headroom stays an explicit separate allowance, never usage; reporting only, not a fit verdict",
        ),
        None => (
            serde_json::Value::Null,
            "MemAvailable unreadable at plan time: capacity unavailable (no zero assumed); TTM pool alone is not a capacity basis; reporting only, not a fit verdict",
        ),
    };
    Ok((
        whole_host,
        serde_json::json!({
            "a_device_union_bytes": a_device_union,
            "a_host_union_bytes": a_host_union,
            "a_device_backed_host_bytes": a_device_backed,
            "pinned_gtt_demand_bytes": pinned_gtt_demand_bytes,
            "explicit_pinned_terms_bytes": {
                "ple_staging_upload_bytes": ple_staging_upload_bytes,
                "load_upload_transient_bytes": load_upload_transient_bytes,
            },
            "ram_only_host_bytes": ram_only_host_bytes,
            "explicit_ram_only_terms_bytes": {
                "ple_resident_envelope_bytes": ple_resident_envelope_bytes,
                "page_cache_bytes": page_cache_bytes,
                "session_cache_configured_bytes": session_cache_configured_bytes,
            },
            "whole_host_union_bytes": whole_host,
            "headroom_bytes": headroom_bytes,
            "admission_needed_bytes": admission_needed_bytes,
            "admission_capacity_bytes_or_null": admission_capacity_bytes,
            "admission_capacity_note": capacity_note,
            "whole_device_union_bytes": a_device_union,
            "note": "checked join: pinned/GTT demand (A host union + B pinned staging/upload, UMA device-backed counted once in A) kept separate from RAM-only terms (reclaimable PLE envelope, page cache, session cap); reporting only, not a fit declaration",
        }),
    ))
}

/// The S1 slow-link union: `(slowlink_charge_bytes, note, report_value)`.
/// Probe-free legal conditions first (`preflight` on the real load shape);
/// where they decline, the declined union carries its reason with no S1
/// charge. Where the conditions pass, the existing `single_device_facts`
/// (scan closure answers `Ok(false)`, so the layout route holds) feeds
/// `load_admission_inputs` and `project_at_tile` per admissible route — the
/// layout route, plus `AggregateSymIu4` where the layout is the stacked
/// aggregate one (the conservative sym-mode bound without selecting it). The
/// charge is the max over projecting routes; the host link stays unmeasured,
/// so no profile is admitted and no resident-layer verdict is read.
/// Required-source failures are named `Err`, never a silent zero. No new
/// planner types: the existing typed S1 authorities feed a plain tuple.
fn preload_s1_union(
    gpu: &rdna_compute::Gpu,
    config: &hipfire_arch_qwen4::Qwen4Config,
    recipe: &hipfire_arch_qwen4::weights::Qwen4TierRecipe,
    weights: &[hipfire_runtime::weight_manifest::WeightEntry],
    bytes_of: &dyn Fn(&hipfire_runtime::weight_manifest::WeightEntry) -> Option<u64>,
    backend: hipfire_arch_qwen4::Qwen4KvBackend,
    max_seq: usize,
    vmm_granularity: usize,
    native_mtp: bool,
    mtp_kept: bool,
    max_k: usize,
    state_format: hipfire_arch_qwen4::Qwen4StateFormat,
    routed_experts: bool,
    use_ranges: bool,
    model_path: &str,
) -> Result<(u64, String, serde_json::Value), String> {
    use hipfire_arch_qwen4::slowlink_admission as s1;
    let declined = |reason: String| {
        (
            0u64,
            reason.clone(),
            serde_json::json!({
                "eligible": false, "admitted": false, "slowlink_charge_bytes": 0u64,
                "link": "unmeasured", "reason": reason,
            }),
        )
    };
    // Ordinary process-snapshot controls, same readings as the load.
    let explicit = hipfire_arch_qwen4::expert_residency::expert_vram_layers_from_env()
        .map(|explicit| explicit.is_some())
        .map_err(|error| {
            format!("qwen4 preload plan: S1 union: explicit expert VRAM layers unreadable: {error}")
        })?;
    let hotset_policy =
        hipfire_arch_qwen4::expert_hotset::HotsetPolicy::from_env().map_err(|error| {
            format!("qwen4 preload plan: S1 union: hotset policy unreadable: {error}")
        })?;
    let placement_policy =
        hipfire_arch_qwen4::expert_hotset::PlacementPolicy::from_env().map_err(|error| {
            format!("qwen4 preload plan: S1 union: placement policy unreadable: {error}")
        })?;
    let hotset_source = match recipe {
        hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextPacked { .. } => {
            hipfire_arch_qwen4::expert_hotset::HotsetSource::Aggregate
        }
        hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. } => {
            hipfire_arch_qwen4::expert_hotset::HotsetSource::SlabArena
        }
        hipfire_arch_qwen4::weights::Qwen4TierRecipe::StockFp8 => {
            hipfire_arch_qwen4::expert_hotset::HotsetSource::StockFp8
        }
        _ => hipfire_arch_qwen4::expert_hotset::HotsetSource::Nonaggregate(
            "not an aggregate MQ4 artifact".into(),
        ),
    };
    let hotset_env = |explicit: bool| hipfire_arch_qwen4::expert_hotset::HotsetEnv {
        gpu_arch: &gpu.arch,
        unified_memory: use_ranges,
        expert_parallel: false,
        explicit_vram_layers: explicit,
        source: hotset_source.clone(),
    };
    let whole_first = hipfire_arch_qwen4::expert_hotset::decide_whole_first(
        placement_policy,
        &hotset_env(explicit),
        false,
    ) == hipfire_arch_qwen4::expert_hotset::WholeFirstDecision::Active;
    let hotset_sidecar = hipfire_arch_qwen4::expert_hotset::preflight(
        &hotset_policy,
        std::path::Path::new(model_path),
        &hotset_env(explicit),
    )
    .is_ok();
    let chunk_rows_env = hipfire_config::developer_var("HIPFIRE_PREFILL_CHUNK_ROWS").ok();
    let npu_trunk = hipfire_arch_qwen4::npu_trunk::NpuTrunkMode::from_env()
        .map(|mode| s1::npu_trunk_selected(&gpu.arch, mode))
        .map_err(|error| {
            format!("qwen4 preload plan: S1 union: NPU trunk mode unreadable: {error}")
        })?;
    let shape = s1::LoadShape {
        arch: &gpu.arch,
        unified_memory: use_ranges,
        expert_parallel: false,
        pipeline_band: false,
        routed_experts,
        expert_keep: config.expert_keep.is_some(),
        explicit_vram_layers: explicit,
        hotset_sidecar,
        whole_first,
        stage_opt_out: hipfire_config::developer_var(
            hipfire_arch_qwen4::gpu_forward::QWEN4_EXPERT_STAGE_ENV,
        )
        .ok()
        .as_deref()
            == Some("0"),
        npu_trunk,
        chunk_rows_env: chunk_rows_env.as_deref(),
        max_seq,
    };
    // A probe-free legal decline is a declined union (reason, no charge), not
    // a failure. Everything past it is required source: facts, inputs and
    // per-route projection arithmetic propagate as named errors.
    if let Err(decline) = s1::preflight(s1::SlowlinkMode::from_env(), &shape) {
        return Ok(declined(format!("probe-free legal decline: {decline}")));
    }
    // No payload scan in planning: the scan closure answers `Ok(false)`, so
    // the layout route holds (the blessed no-artifact-bytes answer).
    let no_scan = || Ok::<bool, String>(false);
    let load = s1::SingleDeviceLoad {
        gpu,
        config,
        weights,
        bytes_of,
        backend,
        max_seq,
        vmm_granularity,
        native_mtp,
        mtp_kept,
        max_k,
        state_format,
        scratch_mode: hipfire_arch_qwen4::gpu_forward::Qwen4ForwardScratchMode::Full,
        purpose: s1::AdmissionPurpose::Product,
        routed_source_symmetric: &no_scan,
    };
    let facts = s1::single_device_facts(&load)
        .map_err(|error| format!("qwen4 preload plan: S1 union: facts unavailable: {error}"))?;
    // The link stays unmeasured in planning. The typed API requires an f64,
    // so the contained `NaN` below is passed ONLY to the sizing query path
    // (`load_admission_inputs` -> `project_at_tile`): `link_gbps` is stored
    // into `AdmissionInputs.link_gbps` and never read there — the census,
    // granule slack, reserve and budget math reads `memory.free_bytes`
    // only (`describe` reads it, and it is never called here). It is never
    // serialized and no profile is admitted from it. A fabricated finite
    // link would claim a measurement that never happened; `NaN` poisons any
    // future reader instead of succeeding with invented data.
    let link_gbps = f64::NAN;
    let layout = s1::window_route_of(weights);
    let mut routes = vec![facts.window_route];
    if layout == s1::WindowRoute::Aggregate
        && facts.window_route != s1::WindowRoute::AggregateSymIu4
    {
        routes.push(s1::WindowRoute::AggregateSymIu4);
    }
    let mut projected = Vec::new();
    for route in routes {
        let route_facts = s1::LoadFacts {
            window_route: route,
            sym_iu4_source_symmetric: None,
            config: facts.config,
            weights: facts.weights,
            bytes_of: facts.bytes_of,
            purpose: facts.purpose,
            memory: facts.memory,
            tile_reserve: facts.tile_reserve,
            incumbent_reserve: facts.incumbent_reserve,
            base_runtime_reserved_bytes: facts.base_runtime_reserved_bytes,
            granule: facts.granule,
            state_format: facts.state_format,
            native_mtp: facts.native_mtp,
            window_rows: facts.window_rows,
            checkpoint_stride_rows: facts.checkpoint_stride_rows,
        };
        let inputs = s1::load_admission_inputs(&route_facts, link_gbps).map_err(|decline| {
            format!(
                "qwen4 preload plan: S1 union: {route:?} admission inputs unavailable: {decline}"
            )
        })?;
        let projection =
            s1::project_at_tile(&inputs, s1::S1_PRODUCT_TILE_ROWS).map_err(|decline| {
                format!("qwen4 preload plan: S1 union: {route:?} tile projection unavailable: {decline}")
            })?;
        // Census and granule totals are required arithmetic: missing sums are
        // named errors, never silent route drops. A route the artifact layout
        // cannot take is the one legal skip: it is recorded, and the union
        // still needs at least one projecting route below.
        let census = projection.census.total().map_err(|error| {
            format!("qwen4 preload plan: S1 union: {route:?} census total unavailable: {error}")
        })?;
        let granule = projection.granule.total().map_err(|error| {
            format!("qwen4 preload plan: S1 union: {route:?} granule total unavailable: {error}")
        })?;
        let charge = projection.charges.slowlink;
        projected.push(serde_json::json!({
            "route": format!("{route:?}"),
            "tile_rows": projection.tile_rows,
            "census_total_bytes": census,
            "granule_total_bytes": granule,
            "slowlink_charge_bytes": charge,
        }));
    }
    if projected.is_empty() {
        return Err(
            "qwen4 preload plan: S1 union: no S1 route projects (all routes errored above)".into(),
        );
    }
    let union_charge = projected
        .iter()
        .filter_map(|route| route.get("slowlink_charge_bytes")?.as_u64())
        .max()
        .unwrap_or(0);
    let note = format!(
        "probe-free eligible; link unmeasured so not admitted; union charge {union_charge} over {} route(s)",
        projected.len()
    );
    Ok((
        union_charge,
        note.clone(),
        serde_json::json!({
            "eligible": true, "admitted": false,
            "slowlink_charge_bytes": union_charge,
            "link": "unmeasured",
            "tile_rows": s1::S1_PRODUCT_TILE_ROWS,
            "reason": note,
            "routes": projected,
        }),
    ))
}

/// `upload_expert_keep_mask` owns one byte per expert slot per trunk layer
/// and, when kept, the MTP head. Bound the optional head without selecting it.
fn preload_keep_mask_bytes(layers: usize, experts: usize) -> Result<u64, String> {
    let bytes = layers
        .checked_add(1)
        .and_then(|owners| owners.checked_mul(experts))
        .ok_or("qwen4 preload plan: keep-mask component union overflows")?;
    u64::try_from(bytes).map_err(|_| "qwen4 preload plan: keep-mask bytes exceed u64".into())
}

/// Charge the sidecar header's physical data section, not its virtual mapping
/// or file extent; still reject an overflowing header data-section endpoint.
fn preload_ple_envelope(data_bytes: u64, data_offset: u64) -> Result<u64, String> {
    data_offset
        .checked_add(data_bytes)
        .ok_or("qwen4 preload plan: PLE header data endpoint overflows")?;
    Ok(data_bytes)
}

/// One staging buffer and the two-buffer PLE userspace charge from the same
/// assembly inputs as the load: `staging_rows * decoded_row_bytes` floored at
/// one 8MiB budget of whole rows. Checked throughout. (The 128MiB bound
/// applies to the ordered device-upload ring only, never to PLE staging.)
fn preload_staging_bytes(
    staging_rows: usize,
    decoded_row_bytes: u64,
) -> Result<(u64, u64), String> {
    if decoded_row_bytes == 0 {
        return Err("qwen4 preload plan: PLE decoded row bytes are zero".into());
    }
    let wanted = (staging_rows as u64)
        .checked_mul(decoded_row_bytes)
        .ok_or("qwen4 preload plan: PLE staging rows overflow")?;
    const BUDGET: u64 = 8 * 1024 * 1024;
    let floor = BUDGET
        .checked_div(decoded_row_bytes)
        .and_then(|rows| rows.checked_mul(decoded_row_bytes))
        .ok_or("qwen4 preload plan: PLE staging floor overflows")?;
    let one = wanted.max(floor);
    let two = one
        .checked_mul(2)
        .ok_or("qwen4 preload plan: PLE staging buffers overflow")?;
    Ok((one, two))
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
    load_qwen4_rank_scoped(
        gpu,
        ep_seal,
        shared,
        expert_vram_layers,
        session,
        src,
        None,
        None,
    )
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
///
/// `pp_chunk_rows` is the canonical prefill chunk the PP load chose once for
/// every band and the mesh ([`load_qwen4_pp_bundle`]); a band requires it
/// and a whole-model load passes `None`.
fn load_qwen4_rank_scoped(
    gpu: &mut rdna_compute::Gpu,
    ep_seal: Option<hipfire_arch_qwen4::ep_seal::Qwen4EpRankSeal>,
    shared: &Qwen4RankShared,
    expert_vram_layers: Option<usize>,
    session: Option<Qwen4SessionSeed<'_>>,
    src: ModelSource,
    band: Option<&hipfire_arch_qwen4::weights::Qwen4BandScope>,
    pp_chunk_rows: Option<usize>,
) -> Result<Qwen4RankLoad, String> {
    let backend = shared.backend;
    let vmm_granularity = shared.vmm_granularity;
    let max_k = shared.max_k;
    let mut native_mtp = shared.native_mtp;
    // Existing owner counters distinguish a fresh Gpu from retained or reused
    // model state without inventing allocation identities. A retained load has
    // no original preload basis here and cannot publish `fresh_rank_load`.
    let fresh_resource_basis = gpu.pool_stats().0 == 0
        && gpu.host_mapped_count() == 0
        && gpu.vmm_allocation_count() == 0
        && gpu.qsa_gather_scratch_bytes() == 0;
    let hotset_policy = hipfire_arch_qwen4::expert_hotset::HotsetPolicy::from_env()?;
    let placement_policy = hipfire_arch_qwen4::expert_hotset::PlacementPolicy::from_env()?;
    let hotset_budget = hipfire_arch_qwen4::expert_hotset::HotsetBudget::from_env()?;
    let ep = match &ep_seal {
        Some(bound) => Some(
            bound
                .seal
                .placement(bound.rank)
                .map_err(|error| format!("qwen4: {error}"))?,
        ),
        None => None,
    };
    // Every rank's routed-expert span, read from the seal (one rank: whole stacks).
    let ep_spans = match &ep_seal {
        Some(bound) => bound
            .seal
            .expert_spans()
            .map_err(|error| format!("qwen4: {error}"))?,
        None => hipfire_arch_qwen4::expert_residency::EpExpertSpans::single(),
    };
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
        return Err(crate::admission::QWEN4_SAFETENSORS_REFUSAL.into());
    };
    // The PLE rows source (`memory.ple_format`).  Admission already
    // resolved and attached a sidecar on this source; resolving again is
    // idempotent and yields the choice for the load log (and covers a
    // source that reaches the carrier without admission).  A developer
    // profile's embedded rows (or a profile without PLE) have no sidecar
    // to resolve: `None`.
    let admission_started = std::time::Instant::now();
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
    rdna_compute::load_phases::add(
        "qwen4_artifact_admission",
        rdna_compute::load_phases::ns_since(admission_started),
        0,
    );
    // A band's forward scratch plan is chosen once, from the full manifest,
    // so every band attaches the same plan (a band-scoped manifest would
    // miss the PLE entries the selector inspects).
    let full_scratch_mode = match band {
        Some(_) => {
            let _scan = rdna_compute::load_phases::span("qwen4_symmetry_scan");
            Some(
                hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
                    ctx.gpu,
                    &hfq,
                    &receipt.config,
                    &receipt.manifest,
                )
                .map_err(|error| error.to_string())?,
            )
        }
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
    // An EP rank must load the sealed source generation: base file, the PLE
    // rows just resolved and the trunk-overlay sidecar just bound. Checked
    // after both are bound, before any allocation, so a sidecar created or
    // replaced in between is refused (fail closed).
    if let Some(bound) = &ep_seal {
        let generation = hipfire_arch_qwen4::ep_seal::qwen4_source_generation(
            &hfq,
            std::path::Path::new(ctx.path),
        )?;
        bound.seal.check_source(&generation)?;
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
        hipfire_config::load_summary::note(
            hipfire_config::load_summary::Facet::SpecOff,
            format!(
                "MTP off: profile {} has no MTP head",
                receipt.config.profile.id()
            ),
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
    let ple_descriptors_started = std::time::Instant::now();
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
    rdna_compute::load_phases::add(
        "ple_setup",
        rdna_compute::load_phases::ns_since(ple_descriptors_started),
        0,
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
    // Applied-PLE evidence for the measured-load descriptor, preserved BEFORE
    // `ple_rows` moves into bundle attachment: whether the APPLIED rows source
    // is a sidecar (with its path/bytes/encoding) or honestly ineligible
    // (embedded/absent under the current FN fixture contract). No device
    // copy, no re-hash; composition reads this settled evidence after load.
    let applied_ple_rows_is_sidecar = matches!(
        receipt.ple_rows,
        hipfire_arch_qwen4::Qwen4PleRows::Sidecar { .. }
    );
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
        None => {
            let _scan = rdna_compute::load_phases::span("qwen4_symmetry_scan");
            hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
                ctx.gpu, &hfq, &config, &manifest,
            )
            .map_err(|error| error.to_string())?
        }
    };
    if scratch_mode != hipfire_arch_qwen4::gpu_forward::Qwen4ForwardScratchMode::Full {
        let reason = if matches!(
            manifest.identity.recipe,
            hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. }
        ) {
            "every routed trunk layer is an expert-slab arena; HIPFIRE_QWEN4_COMPACT_SCRATCH=0 keeps the full plan"
        } else {
            "symmetric routed experts assumed; the GPU symmetry check at attach confirms or re-plans Full"
        };
        eprintln!("  qwen4 forward scratch: {scratch_mode:?} plan ({reason})");
    }
    let use_ranges = ctx.gpu.is_uma();
    if use_ranges {
        hfq.drop_mmap();
    }
    // Direct weight reads (HIPFIRE_QWEN4_DIRECT_LOAD, default on; exactly
    // `0` is the old mmap + single-thread-copy path): discrete GPUs only.
    let direct_source = if use_ranges || !hipfire_config::memory::qwen4_direct_load() {
        None
    } else if hfq.has_overlay() {
        None
    } else {
        Some(
            hipfire_runtime::hfq_parallel::DirectSource::open(&hfq)
                .map_err(|error| format!("qwen4: direct load: {error}"))?,
        )
    };
    match &direct_source {
        Some(direct) => eprintln!(
            "  qwen4 direct load: on ({} lanes, {})",
            hipfire_runtime::hfq_parallel::HFQ_READER_LANES,
            if direct.is_direct_available() {
                "O_DIRECT"
            } else {
                "buffered-only"
            }
        ),
        None => eprintln!("  qwen4 direct load: off"),
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
    // `native_mtp` as the placement sees it: the later host-spill decline
    // must not move the reserve the placement was sized with.
    let placement_native_mtp = native_mtp;
    let placement_started = std::time::Instant::now();
    // The chunk the expert placement reserve, forward census and pool
    // workspace are sized from: the requested chunk (explicit
    // `HIPFIRE_PREFILL_CHUNK_ROWS`, else the arch default). On gfx1100 that
    // default measures the host link once (cached per device; a unified
    // device is classified without a copy) and is 8192 only on a narrow
    // discrete link, so the placement cedes resident layers for the wider
    // chunk there. On gfx1201 the default follows the load's scratch plan
    // (16384 rows under the compact plan, 4096 under `Full`). Under pipeline
    // parallelism (`band.is_some()`) the chunk is the one the PP load chose
    // once (`load_qwen4_pp_bundle`: the requested chunk, stepped down by the
    // PP auto-fit when the cards cannot hold it) and handed to every band;
    // bands refuse a chunk that differs from the mesh chunk. Every other
    // arch and any explicit value never probes.
    let placement_chunk = if band.is_some() {
        pp_chunk_rows.ok_or_else(|| {
            "qwen4: topology.pp band load without the mesh's canonical prefill chunk".to_string()
        })?
    } else {
        hipfire_arch_qwen4::gpu_forward::qwen4_prefill_chunk_requested(
            &*ctx.gpu,
            ctx.max_seq,
            scratch_mode,
        )
    };
    // Freeze the original capacity inputs before placement, materialization or
    // capture growth. Pool readings are separate instants, not a free-memory
    // delta and not evidence of an old model's original preload basis.
    let preallocation_pool_bytes = u64::try_from(ctx.gpu.pool_parked_bytes()).unwrap_or(u64::MAX);
    let preallocation_free_device_bytes = ctx
        .gpu
        .device_mem_info()
        .map(|(free, _)| u64::try_from(free).unwrap_or(u64::MAX))
        .ok();
    let preallocation_mem_available = rdna_compute::kv_slots::mem_available_bytes();
    let preallocation_ttm_pool = residency::ttm_pool_estimate_on(&*ctx.gpu);
    let preallocation_gtt = residency::gtt_budget();
    // Exactly one owner of the planning inputs, including explicit-layers and
    // whole-first resident paths that previously never invoked `auto`.
    let placement_inputs = if band.is_none() {
        Some(qwen4_placement_inputs(
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
            placement_native_mtp,
            routed_experts,
            state_format,
            scratch_mode,
            &ep_spans,
            placement_chunk,
            None,
        )?)
    } else {
        None
    };
    // Decode-share `on` (verdict change 4): an admitted load charges only the
    // dedicated forward (resident bank + persistent token ids + speculative
    // buffers) instead of the whole prefill scratch, because the prefill
    // scratch lives on the donor tail. The reserve is computed here beside
    // the plain one, from the same facts, and only used if the decision
    // below admits sharing and the plan carries a donor tail; every other
    // load never reads it (and the plain reserve is byte-identical).
    let decode_share_mode = hipfire_arch_qwen4::decode_share::DecodeShareMode::from_env();
    // The weight shadows an admitted `on` / `fixed` load keeps in the donor
    // tail instead of resident VRAM (F16 weight shadows + shared-expert INT8
    // shadows, read only by the borrowed >= 512-row prompt tiles; rebuilt per
    // borrow epoch). `None`: they stay resident and charged as today. The
    // region is laid out by the one pure plan the forward re-derives at attach.
    let decode_share_shadow_region: Option<
        hipfire_arch_qwen4::decode_share_shadows::ShadowRegionPlan,
    > = if band.is_none()
        && decode_share_mode.borrows_shadows(
            &ctx.gpu.arch,
            hipfire_arch_qwen4::decode_share::decode_share_shadows_enabled(),
        )
        && placement_chunk > hipfire_arch_qwen4::gpu_forward::QWEN4_DECODE_SHARE_RESIDENT_ROWS
    {
        match hipfire_arch_qwen4::decode_share_shadows::shadow_flags(
            &ctx.gpu.arch,
            ctx.gpu.qwen4_f16_shadow_capable(),
            ctx.gpu.qwen4_shared_i8_enabled(),
            placement_chunk,
        ) {
            Ok((f16, shared_i8)) => hipfire_arch_qwen4::decode_share_shadows::plan_shadow_region(
                &manifest.weights,
                config.num_hidden_layers,
                f16,
                shared_i8,
            )
            .map_err(|error| format!("qwen4 decode-share: {error}"))?
            .into_nonempty(),
            Err(reason) => {
                if hipfire_arch_qwen4::decode_share::DecodeShareMode::explicit() {
                    eprintln!("  qwen4 decode-share: weight shadows stay resident ({reason})");
                }
                None
            }
        }
    } else {
        None
    };
    let decode_share_shadow_bytes: u64 = decode_share_shadow_region
        .as_ref()
        .map_or(0, |region| region.total_bytes);
    let shared_placement_inputs = if band.is_none()
        && decode_share_mode.grows_hot_budget()
        && ctx.gpu.arch == "gfx1201"
        && ep.is_none()
        && scratch_mode == hipfire_arch_qwen4::gpu_forward::Qwen4ForwardScratchMode::FnSymGfx1201
        && placement_chunk > hipfire_arch_qwen4::gpu_forward::QWEN4_DECODE_SHARE_RESIDENT_ROWS
        && !hotset_budget.is_fixed()
    {
        match (
            hipfire_arch_qwen4::gpu_forward::qwen4_decode_share_dedicated_bytes(
                &config,
                placement_chunk,
                scratch_mode,
            ),
            hipfire_arch_qwen4::gpu_forward::qwen4_decode_share_borrow_bytes(
                &config,
                placement_chunk,
                scratch_mode,
            ),
        ) {
            (Some(dedicated), Some(borrowed)) => qwen4_placement_inputs(
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
                placement_native_mtp,
                routed_experts,
                state_format,
                scratch_mode,
                &ep_spans,
                placement_chunk,
                Some(hipfire_arch_qwen4::expert_residency::DecodeShareForward {
                    dedicated,
                    borrowed,
                    shadows_borrowed: decode_share_shadow_bytes,
                }),
            )
            .ok(),
            _ => None,
        }
    } else {
        None
    };
    let capture_reserve_drafts =
        hipfire_arch_qwen4::mtp_spec::native_mtp_proposal_drafts(max_k, &ctx.gpu.arch);
    let planned_capture = placement_native_mtp
        .then(|| {
            hipfire_arch_qwen4::mtp_spec::native_mtp_capture(
                &*ctx.gpu,
                &config,
                state_format.gdn,
                capture_reserve_drafts,
            )
        })
        .flatten();
    // Geometry alone is not admission. In particular the existing head/spill
    // policy can leave a fully resident load's native-MTP census absent; do
    // not materialize a predicted ring that its owning reserve did not admit.
    let capture_plan_admitted = match (placement_inputs, planned_capture) {
        (Some((_, _, reserve, Some(_), _)), Some((gdn, rows))) => {
            hipfire_arch_qwen4::mtp_spec::qwen4_capture_bytes(
                &config,
                gdn,
                rows,
                config.n_linear_layers(),
            )
            .is_some_and(|bytes| bytes > 0 && reserve.census.mtp >= bytes)
        }
        _ => false,
    };
    if band.is_none() && planned_capture.is_some() && !capture_plan_admitted {
        eprintln!("  qwen4 native capture: not preallocated (owning native-MTP census {:?} does not admit the requested capture; the existing pre-load head/spill policy must reserve it)",
            placement_inputs.as_ref().map(|inputs| inputs.2.census.mtp));
    }
    // What `auto` sizes its placement from: free VRAM, the non-expert and
    // per-layer expert bytes, and the reserve (with its native MTP and
    // QSA gather parts) at `placement_chunk`. Unified memory charges context
    // arenas as committed (VMM: the first chunk's pages; legacy: all of
    // `max_seq`), never their virtual extent; a discrete card charges legacy
    // storage of `max_seq` whatever the backend, so VMM growth stays inside
    // the reserve and placement matches legacy.
    let auto_inputs_at = |chunk_rows: usize| -> Result<
        (
            u64,
            u64,
            u64,
            residency::AutoReserve,
            Option<u64>,
            Option<u64>,
        ),
        String,
    > {
        let (free, _) = ctx
            .gpu
            .device_mem_info()
            .map_err(|error| format!("qwen4: device memory query: {error}"))?;
        let (non_expert, layer_experts, reserve, mtp_bytes, gather_bytes) =
            placement_inputs.ok_or("qwen4: pipeline band has no single-rank placement census")?;
        Ok((
            free as u64,
            non_expert,
            layer_experts,
            reserve,
            mtp_bytes,
            gather_bytes,
        ))
    };
    // (`auto_inputs` is defined after the S1 selection below.)
    // What a whole-layer placement keeps free beyond the reserve: the safety
    // margin and, on a slab load that allocates them (`staged`), the two
    // dedicated long-prefill stage buffers sized from the largest layer.
    // The hot set path charges its own margin and tier-plan stage.
    // The stage switch here is the placement switch that ignores the
    // measured link (explicit `HIPFIRE_QWEN4_EXPERT_STAGE`, else gfx1201). A
    // slab stage the link alone would enable is not charged because it
    // allocates nothing: it borrows two resident slab layers as its stage
    // buffers (`Qwen4ExpertStage::new_slab`, `borrowed`), so it can neither
    // take uncharged VRAM nor displace a resident layer.
    let slab_stage = residency::dedicated_slab_stage(
        residency::expert_stage_placement_enabled(&ctx.gpu.arch),
        matches!(
            manifest.identity.recipe,
            hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. }
        ),
        ep.is_some(),
    );
    // (`placement_charges` is defined after the S1 selection below.)
    let placement_charges = |layer_experts: u64, staged: bool| {
        qwen4_shared_charges(layer_experts, staged, slab_stage, 0)
            .map_err(|error| format!("qwen4: {error}"))
    };
    let mut settled_placement_charges = None;
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
    // S1 slow-link layer-major prefill (`HIPFIRE_QWEN4_SLOWLINK_PREFILL`, default
    // off until promoted). Selected here, before any placement is sized and
    // before any allocation: the probe-free checks first, the host link only
    // for a load that passes them, then the joint placement at the purpose's
    // tile (`AdmissionPurpose::Product`: the 768-row tile, prompt arena, expert
    // page store, panel scratch, boundary state), keeping the LARGEST resident
    // layer count that fits the post-initialization free memory (ten is only
    // the preferred count, never a floor; zero is valid when the base fits).
    // Every other load keeps today's placement, reserve and log byte for byte
    // (zero S1 charge).
    let slowlink_mode = hipfire_arch_qwen4::slowlink_admission::SlowlinkMode::from_env();
    // Construct one immutable plan only inside the selector's lazy facts
    // closure. The same plan is charged below and handed to fulfillment.
    let mut slowlink_weight_arenas = None;
    // Device free memory sampled immediately before the host-link
    // initialization. The selector runs the link closure only after its
    // probe-free preconditions pass (default-off, UMA and other ineligible
    // loads never reach it), so those query nothing; an eligible load pays ONE
    // `hipMemGetInfo` here even when the link then measures fast or fails to
    // measure (the class is unknown until the probe runs). The admission's own
    // free sample is taken after the initialization.
    let slowlink_pre_init_free = std::cell::Cell::new(None::<u64>);
    let slowlink_profile = {
        use hipfire_arch_qwen4::slowlink_admission as s1;
        let chunk_rows_env = hipfire_config::developer_var("HIPFIRE_PREFILL_CHUNK_ROWS").ok();
        let stage_env =
            hipfire_config::developer_var(hipfire_arch_qwen4::gpu_forward::QWEN4_EXPERT_STAGE_ENV)
                .ok();
        // Sealed layout/panel/codec/kernel verdicts admit before
        // materialization (never the recipe name or a raw offload veto); an
        // actually CPU-split sealed selection refuses at the dispatch gate.
        let shape = s1::LoadShape {
            arch: &ctx.gpu.arch,
            unified_memory: use_ranges,
            expert_parallel: ep.is_some() || expert_vram_layers.is_some(),
            pipeline_band: band.is_some(),
            routed_experts,
            expert_keep: config.expert_keep.is_some(),
            explicit_vram_layers: explicit.is_some(),
            hotset_sidecar: hotset_path.is_some(),
            whole_first: whole_first == hotsets::WholeFirstDecision::Active,
            stage_opt_out: stage_env.as_deref() == Some("0"),
            npu_trunk: s1::npu_trunk_selected(
                &ctx.gpu.arch,
                hipfire_arch_qwen4::npu_trunk::NpuTrunkMode::from_env()?,
            ),
            chunk_rows_env: chunk_rows_env.as_deref(),
            max_seq: ctx.max_seq,
        };
        // The symmetric IU4 window route's one source-byte scan of every routed trunk layer
        // (run only when its device terms hold, `HIPFIRE_QWEN4_MOE_SYM_IU4=1`).
        let routed_source_symmetric = || {
            hipfire_arch_qwen4::artifact::routed_headers_symmetric_verdict(&hfq, &manifest)
                .map_err(|error| error.to_string())
        };
        s1::select_load_profile(
            slowlink_mode,
            &shape,
            || {
                slowlink_pre_init_free
                    .set(ctx.gpu.device_mem_info().ok().map(|(free, _)| free as u64));
                hipfire_arch_qwen4::host_link::host_link(&*ctx.gpu)
            },
            || {
                let load = s1::SingleDeviceLoad {
                    gpu: &*ctx.gpu,
                    config: &config,
                    weights: &manifest.weights,
                    bytes_of: &bytes_of,
                    backend,
                    max_seq: ctx.max_seq,
                    vmm_granularity,
                    native_mtp: placement_native_mtp,
                    mtp_kept: placement_native_mtp
                        && crate::admission::qwen4_mtp_with_host_mapped_experts(
                            ctx.spec,
                            &ctx.gpu.arch,
                            usize::from(routed_experts),
                        ),
                    max_k,
                    state_format,
                    scratch_mode,
                    purpose: s1::AdmissionPurpose::Product,
                    routed_source_symmetric: &routed_source_symmetric,
                };
                let arenas = s1::single_device_arena_plan(&load)?;
                let facts = s1::single_device_facts_with_arenas(
                    &load,
                    &arenas,
                    slowlink_pre_init_free.get(),
                )?;
                slowlink_weight_arenas = Some(arenas);
                Ok(facts)
            },
        )?
    };
    if slowlink_profile.is_none() {
        slowlink_weight_arenas = None;
    }
    // The S1 profile sizes its placement from its own purpose tile (product:
    // 768 rows), never the incumbent chunk; every other load keeps the
    // incumbent placement chunk.
    let placement_chunk = slowlink_profile
        .as_ref()
        .map_or(placement_chunk, |profile| profile.tile_rows);
    let slowlink_reserve = match &slowlink_profile {
        Some(profile) => profile
            .reserve_bytes()
            .map_err(|error| format!("qwen4: {error}"))?,
        None => 0,
    };
    let auto_inputs = || auto_inputs_at(placement_chunk);
    // What a whole-layer placement keeps free beyond the reserve: the safety
    // margin, on a slab load that allocates them (`staged`) the two dedicated
    // long-prefill stage buffers sized from the largest layer, and (S1 only)
    // the slow-link prefill profile's census. The hot set path charges its own
    // margin and tier-plan stage.
    let placement_charges = |layer_experts: u64, staged: bool| {
        qwen4_shared_charges(layer_experts, staged, slab_stage, slowlink_reserve)
            .map_err(|error| format!("qwen4: {error}"))
    };
    // All optional-sidecar decisions finish against the original manifest:
    // binding, source/manifest eligibility, host limits, plan and a fresh
    // free-VRAM check. A fixed budget fails closed; otherwise a refusal
    // leaves whole-layer placement untouched.
    // Decode-share (`HIPFIRE_QWEN4_DECODE_SHARE`): `bind` plans a donor tail
    // inside the ranked hot-set plan below and builds the mirror and both read
    // views at materialization; the forward does not use them yet. Admission
    // (verdict change 5) is checked here, before any placement is sized, from
    // facts known at this point: the arch, the single-device topology (no band,
    // no expert parallelism, no NPU trunk), the ranked hot-set tier (a sidecar
    // applies, whole-first does not), the load's scratch plan, the S1
    // slow-link verdict, DFlash, the explicit `memory.offload_exec` and a forced
    // `HIPFIRE_QWEN4_D2_DEFAULT=1`, and the placement chunk. The measured
    // (unset) D2 default is only decided inside `attach_forward`; it is
    // re-checked after attach below. A refusal logs its reason and keeps the
    // plain placement.
    // A refusal (and the pin below) logs only when the knob was set
    // explicitly: an unset knob is the product default and a load it does not
    // admit stays silent and exactly as without decode-share.
    let decode_share_explicit = hipfire_arch_qwen4::decode_share::DecodeShareMode::explicit();
    let decode_share_need = if decode_share_mode
        == hipfire_arch_qwen4::decode_share::DecodeShareMode::Off
    {
        None
    } else if !decode_share_mode.plans_donors() {
        if decode_share_explicit {
            eprintln!(
                "  qwen4 decode-share: mode {} plans no donor tail; plain placement",
                decode_share_mode.name()
            );
        }
        None
    } else {
        let facts = hipfire_arch_qwen4::decode_share::DecodeShareFacts {
            arch: &ctx.gpu.arch,
            pipeline_band: band.is_some(),
            expert_parallel: ep.is_some() || expert_vram_layers.is_some(),
            // An unparseable NPU knob is the attach's error to report; here it
            // only means "not selected" for this decision.
            npu_trunk: match hipfire_arch_qwen4::npu_trunk::NpuTrunkMode::from_env() {
                Ok(npu) => {
                    hipfire_arch_qwen4::slowlink_admission::npu_trunk_selected(&ctx.gpu.arch, npu)
                }
                Err(_) => false,
            },
            ranked_hotset_tier: hotset_path.is_some() && band.is_none(),
            scratch_mode,
            slowlink_profile: slowlink_profile.is_some(),
            dflash: ctx.spec.dflash == Some(true),
            offload_exec: hipfire_config::memory::offload_exec_explicit(),
            d2_default_forced: hipfire_config::memory::qwen4_d2_default() == Some(true),
            continuous_batch: shared.continuous_batch,
            prefill_chunk: placement_chunk,
        };
        match hipfire_arch_qwen4::decode_share::decode_share_admission(&facts) {
            Err(reason) => {
                if decode_share_explicit {
                    eprintln!("  qwen4 decode-share: refused: {reason}; plain placement");
                }
                None
            }
            Ok(())
                if hipfire_arch_qwen4::decode_share::pinned_budget_refusal(
                    decode_share_mode,
                    hotset_budget.is_fixed(),
                )
                .is_some() =>
            {
                // A pinned hot budget does not grow, so sharing would reclaim
                // no residency: `on` is `off` for this load (`fixed` / `bind`
                // are controls and still apply).
                if decode_share_explicit {
                    eprintln!(
                        "{}",
                        hipfire_arch_qwen4::decode_share::pinned_budget_refusal(
                            decode_share_mode,
                            true
                        )
                        .unwrap_or_default()
                    );
                }
                None
            }
            Ok(()) => {
                // The donor tail holds the borrowed weight shadows (if any)
                // and then the largest class's scratch: the largest class span.
                let span = hipfire_arch_qwen4::gpu_forward::qwen4_decode_share_class_spans(
                    &config,
                    placement_chunk,
                    scratch_mode,
                    decode_share_shadow_bytes,
                )
                .and_then(|spans| spans.last().copied());
                if span.is_none() && decode_share_explicit {
                    eprintln!(
                        "  qwen4 decode-share: refused: the borrowed scratch span cannot be sized; plain placement"
                    );
                }
                span
            }
        }
    };
    // The donor-tail bytes each prefill class dirties: the plan builds one
    // BorrowPrefill view per class (donors a class can overwrite read their
    // mirror, the rest stay in VRAM). Empty (one all-donors view) only if the
    // spans cannot be sized.
    let decode_share_class_spans: Vec<u64> = if decode_share_need.is_some() {
        hipfire_arch_qwen4::gpu_forward::qwen4_decode_share_class_spans(
            &config,
            placement_chunk,
            scratch_mode,
            decode_share_shadow_bytes,
        )
        .unwrap_or_default()
    } else {
        Vec::new()
    };
    // `on` grows the hot budget from the dedicated reserve; `bind` / `fixed`
    // keep the plain budget (controls).
    let decode_share_grow = decode_share_need.is_some()
        && decode_share_mode.grows_hot_budget()
        && shared_placement_inputs.is_some();
    // The growth of an applied `on` placement: budget bytes gained, device
    // experts before and after, and the reserve actually used.
    let mut decode_share_growth: Option<(u64, usize, usize, residency::AutoReserve)> = None;
    let mut hotset_host_bytes = 0;
    let (hotset, hotset_plan) = if let Some(path) =
        hotset_path.as_deref().filter(|_| band.is_none())
    {
        let prepare = |grow: bool, need: Option<u64>| -> Result<_, String> {
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
            let (free, non_expert, _, plain_reserve, mtp_bytes, gather_bytes) = auto_inputs()?;
            // "Decide share -> compute reserve -> plan": a growing attempt sizes
            // the budget from the dedicated reserve, every other attempt from
            // the plain one (never a plain forward on a budget sized for sharing).
            let reserve_obj = if grow {
                shared_placement_inputs
                    .map(|inputs| inputs.2)
                    .ok_or("the decode-share reserve is unavailable")?
            } else {
                plain_reserve
            };
            let reserve_note = reserve_obj.describe();
            let reserve = reserve_obj.bytes();
            let total =
                hotsets::require_budget(residency::expert_hotset_budget(free, non_expert, reserve))
                    .map_err(|skip| skip.to_string())?;
            // Retain the automatic fixed point for stage buffers. Under a
            // pin, reserve those buffers alongside the unchanged hot budget.
            let stage_enabled = residency::expert_stage_placement_enabled(&ctx.gpu.arch);
            let table_geometry = (
                config.num_hidden_layers,
                config.num_experts,
                config.expert_keep.is_some(),
            );
            let (plan, budget, stage) = qwen4_plan_hotset_placement(
                &slabs,
                sidecar.rank(),
                hotset_budget,
                total,
                stage_enabled,
                table_geometry,
                need,
                &decode_share_class_spans,
            )?;
            // Every class span already includes the borrowed weight-shadow
            // region at the start of the donor tail; record it so the lease
            // (and the forward) know the scratch backing starts after it.
            let plan = qwen4_record_shadow_region(plan, need.is_some(), decode_share_shadow_bytes)?;
            let growth = if grow && plan.decode_share_tail().is_some() {
                // What the plain reserve would have placed, for the load log.
                let plain_total = hotsets::require_budget(residency::expert_hotset_budget(
                    free,
                    non_expert,
                    plain_reserve.bytes(),
                ))
                .map_err(|skip| skip.to_string())?;
                let (plain_plan, _, _) = qwen4_plan_hotset_placement(
                    &slabs,
                    sidecar.rank(),
                    hotset_budget,
                    plain_total,
                    stage_enabled,
                    table_geometry,
                    None,
                    &[],
                )?;
                Some((
                    total.saturating_sub(plain_total),
                    plain_plan.device_experts(),
                    plan.device_experts(),
                    reserve_obj,
                ))
            } else {
                None
            };
            // The base tier table was already subtracted by expert_hotset_budget.
            let extra_tables = qwen4_expert_table_bytes(
                stage_enabled
                    && plan
                        .owner_bytes(hipfire_arch_qwen4::expert_tier_plan::Placement::HostMapped)
                        > 0,
                table_geometry,
            )? - residency::EXPERT_TIER_TABLE_BYTES
                + qwen4_decode_share_table_bytes(&plan, table_geometry)?;
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
                preallocation_mem_available,
                preallocation_ttm_pool,
            )?;
            residency::check_gtt_cap(host_bytes, preallocation_gtt)?;
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
                            .and_then(|bytes| bytes.checked_sub(extra_tables))
                            .ok_or_else(|| {
                                "no budget after long-prefill stage/table owners".to_string()
                            })
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
                "  qwen4 expert hot set budget: {} MiB free - {} MiB non-expert weights - {} MiB reserve ({reserve_note}; {} MiB native MTP, {} MiB QSA gather) - {} bytes base tier tables - {} MiB safety margin - {} MiB long-prefill stage buffers - {} bytes extra stage tables/masks = {} MiB",
                free / MIB, non_expert / MIB, reserve / MIB,
                mtp_bytes.unwrap_or(0) / MIB, gather_bytes.unwrap_or(0) / MIB,
                residency::EXPERT_TIER_TABLE_BYTES,
                residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES / MIB, stage / MIB, extra_tables, (total - stage - extra_tables) / MIB,
            );
            if hotset_budget.is_fixed() {
                eprintln!(
                    "  qwen4 expert hot set fixed budget: {}={} MiB ({} bytes); ranked prefix independent of MTP mode",
                    hotsets::HOT_BUDGET_MIB_ENV, budget / MIB, budget,
                );
            }
            Ok((
                sidecar,
                std::sync::Arc::new(plan),
                budget,
                host_bytes,
                stage,
                growth,
            ))
        };
        // An `on` attempt that lowered the reserve is only kept if the plan
        // really carries the donor tail; a refusal anywhere after the reserve
        // was lowered recomputes the placement with the plain reserve.
        let attempt = match prepare(decode_share_grow, decode_share_need) {
            Ok(done) if !decode_share_grow || done.1.decode_share_tail().is_some() => Ok(done),
            refused if decode_share_grow => {
                match &refused {
                    Err(reason) => eprintln!(
                        "  qwen4 decode-share: on: sharing refused after the reserve was lowered ({reason}); recomputing the placement with the plain reserve"
                    ),
                    Ok(_) => eprintln!(
                        "  qwen4 decode-share: on: no donor tail after the reserve was lowered; recomputing the placement with the plain reserve"
                    ),
                }
                prepare(false, None)
            }
            other => other,
        };
        match attempt {
            Ok((sidecar, plan, budget, host_bytes, stage, growth)) => {
                hotset_host_bytes = host_bytes;
                decode_share_growth = growth;
                settled_placement_charges = Some(residency::PlacementCharges {
                    margin: residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES,
                    stage,
                    slowlink: slowlink_reserve,
                });
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
    let mut unset_note = String::new();
    // The one shared whole-layer placement decision: full precedence through
    // the genuine owners, with every observation injected in owner order.
    // Hotset and admitted-S1 arms never query; admitted S1 never re-queries.
    let decided = qwen4_decide_placement(
        Qwen4PlacementPolicy {
            band: band.is_some(),
            hotset: hotset_plan.is_some(),
            admitted_s1: slowlink_profile.as_ref().map(|profile| profile.charges),
            ep_agreed: expert_vram_layers,
            use_ranges,
            explicit,
            slowlink_bytes: slowlink_reserve,
            table_geometry: (
                config.num_hidden_layers,
                config.num_experts,
                config.expert_keep.is_some(),
            ),
        },
        || {
            residency::routed_expert_bytes_ranks(&manifest.weights, bytes_of, &ep_spans)
                .map_err(|error| format!("qwen4: {error}"))
        },
        &auto_inputs,
        || {
            state_format.qsa_f16_arena
                || state_format.qsa_pooled_bf16
                || hipfire_config::oom_guard_effective(Some(&ctx.gpu.arch))
        },
    )?;
    if decided.charges.is_some() {
        settled_placement_charges = decided.charges;
    }
    let mut measured = decided.measured;
    if let Some((free, non_expert, reserve_note, experts, fits)) = decided.uma_log {
        let detail = format!(
            "{} MiB routed experts + {} MiB non-expert weights + {} MiB reserve ({reserve_note}) {} {} MiB free device memory",
            experts / MIB,
            non_expert / MIB,
            decided.reserve.expect("UMA reserve") / MIB,
            if fits { "fit in" } else { "exceed" },
            free / MIB
        );
        if !fits {
            eprintln!("  qwen4: {detail} (memory.oom_guard off: loading anyway)");
        }
        unset_note = format!(" ({detail})");
    }
    if let Some((free, non_expert, experts, base_note, fits)) = decided.unset_log {
        let charges = settled_placement_charges.expect("resident-fit charges");
        unset_note = format!(
            " ({} MiB routed experts + {} MiB non-expert weights + {} MiB reserve (base {}; plus {}) {} {} MiB free VRAM)",
            experts / MIB,
            non_expert / MIB,
            decided.reserve.expect("resident-fit reserve") / MIB,
            base_note,
            charges.describe(),
            if fits { "fit in" } else { "exceed" },
            free / MIB
        );
    }
    let placement = decided.policy;
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
        let charges = placement_charges(layer_experts, false)?;
        settled_placement_charges = Some(charges);
        let reserve = charges
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
    let stage_placement_enabled = residency::expert_stage_placement_enabled(&ctx.gpu.arch);
    let borrowed_stage_donors = stage_placement_enabled
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
        // An admitted S1 profile already carries the maximal-fitting resident
        // count from the ONE post-initialization quote (`profile.placement`).
        // The placement never re-queries free memory or re-derives the joint
        // arithmetic from another epoch: it places exactly that count.
        let admitted = slowlink_profile
            .as_ref()
            .map(|profile| profile.resident_layers);
        let vram_layers = match (policy, admitted) {
            (residency::ExpertVramLayers::Layers(layers), _) => {
                let (_, layer_experts, ..) = placement_inputs
                    .ok_or("qwen4: explicit expert layers have no owning placement census")?;
                settled_placement_charges = Some(placement_charges(
                    layer_experts,
                    layers < config.num_hidden_layers,
                )?);
                layers
            }
            (residency::ExpertVramLayers::Auto, Some(layers)) => layers,
            (residency::ExpertVramLayers::Auto, None) => {
                let (free, non_expert, layer_experts, auto, mtp_bytes, gather_bytes) =
                    match measured.take() {
                        Some(inputs) => inputs,
                        None => auto_inputs()?,
                    };
                let gather_note = gather_bytes
                    .map(|bytes| format!(", {} MiB QSA gather scratch", bytes / MIB))
                    .unwrap_or_default();
                let placement_layer_bytes = match &slowlink_profile {
                    Some(profile) => profile
                        .placement_layer_bytes(layer_experts)
                        .map_err(|error| format!("qwen4: {error}"))?,
                    None => layer_experts,
                };
                let charges = placement_charges(layer_experts, true)?;
                let count_inputs = Qwen4SharedPlacementInputs {
                    stage_enabled: stage_placement_enabled,
                    num_hidden_layers: config.num_hidden_layers,
                    num_experts: config.num_experts,
                    moe_layers,
                    recipe_is_slab: slab_stage,
                    use_ranges,
                    slab_stage,
                    borrowed_donors: borrowed_stage_donors,
                    whole_first,
                    explicit,
                    expert_keep: config.expert_keep.is_some(),
                    non_expert,
                    layer_experts: placement_layer_bytes,
                    reserve_base: auto.placement_bytes(slowlink_profile.is_some()),
                    free_bytes: free,
                };
                let (layers, plan, charges) =
                    qwen4_shared_auto_whole_layers(&count_inputs, &charges)
                        .map_err(|error| format!("qwen4: {error}"))?;
                settled_placement_charges = Some(charges);
                let charged = charges
                    .reserve(auto.placement_bytes(slowlink_profile.is_some()))
                    .map_err(|error| format!("qwen4: {error}"))?;
                eprintln!(
                    "  qwen4 auto expert placement: {} MiB free, {} MiB non-expert weights, {} MiB reserved (base {}; {} MiB native MTP{gather_note}; plus {}), {} MiB per expert layer",
                    free / MIB,
                    non_expert / MIB,
                    charged / MIB,
                    auto.describe(),
                    mtp_bytes.unwrap_or(0) / MIB,
                    charges.describe(),
                    placement_layer_bytes / MIB
                );
                whole_first_plan = plan;
                layers
            }
        };
        // The default stays on the incumbent prefix path without allocating
        // a set. Only an admitted S1 load may override its layer identities.
        let resident_set = {
            use hipfire_arch_qwen4::slowlink_admission as s1;
            if slowlink_profile.is_some()
                && s1::resident_layer_switch_is_override(
                    hipfire_config::developer_var(s1::S1_RESIDENT_LAYERS_ENV)
                        .ok()
                        .as_deref(),
                )
            {
                Some(s1::s1_resident_layer_set(
                    config.num_hidden_layers,
                    vram_layers,
                )?)
            } else {
                None
            }
        };
        let moved = qwen4_shared_apply_placement(
            &mut manifest.weights,
            vram_layers,
            resident_set.as_deref(),
        );
        applied_layers = resident_set
            .as_ref()
            .map_or(vram_layers.min(config.num_hidden_layers), Vec::len);
        // This rank's pinned bytes; the host check covers every rank that
        // has not pinned yet (this one and the later ranks), each at its own
        // share (an uneven split gives the low ranks one more expert).
        let rank_host_bytes =
            residency::host_mapped_bytes_ranks(&manifest.weights, bytes_of, &ep_spans)
                .map_err(|error| format!("qwen4: {error}"))?;
        let own_rank = ep.map_or(0, |placement| placement.rank);
        let host_bytes = rank_host_bytes[own_rank];
        let host_check_bytes = rank_host_bytes[own_rank..]
            .iter()
            .fold(0u64, |total, bytes| total.saturating_add(*bytes));
        let mem_available = preallocation_mem_available;
        let ttm_pool = preallocation_ttm_pool;
        residency::check_host_ram(host_check_bytes, mem_available, ttm_pool)
            .map_err(|error| format!("qwen4: {error}"))?;
        residency::check_gtt_cap(host_check_bytes, preallocation_gtt)
            .map_err(|error| format!("qwen4: {error}"))?;
        // SKU emulation: the same pinned-host (GTT) bytes against the capacity
        // ledger's shared-host/GTT room, in addition to the real checks above.
        // No-op when HIPFIRE_EMU is off.
        ctx.gpu
            .emu_check_host_allocation(host_check_bytes, true)
            .map_err(|error| format!("qwen4: {error}"))?;
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        if let Some(resident) = &resident_set {
            eprintln!(
                "  qwen4 routed experts: layers {resident:?} in VRAM, {moved} expert tensors ({:.1} GiB) in pinned host RAM \
                 (MemAvailable - CMA {:.1} GiB + {:.1} GiB estimated in TTM's page pool)",
                gib(host_bytes),
                gib(mem_available.unwrap_or(0)),
                gib(ttm_pool)
            );
        } else {
            eprintln!(
                "  qwen4 routed experts: layers 0..{} in VRAM, {moved} expert tensors ({:.1} GiB) in pinned host RAM \
                 (MemAvailable - CMA {:.1} GiB + {:.1} GiB estimated in TTM's page pool)",
                vram_layers.min(config.num_hidden_layers),
                gib(host_bytes),
                gib(mem_available.unwrap_or(0)),
                gib(ttm_pool)
            );
        }
        hipfire_config::load_summary::note(
            hipfire_config::load_summary::Facet::Placement,
            match &resident_set {
                Some(resident) => format!(
                    "experts: {} layers in VRAM, {moved} tensors ({:.1} GiB) host-mapped",
                    resident.len(),
                    gib(host_bytes)
                ),
                None => format!(
                    "experts: layers 0..{} in VRAM, {moved} tensors ({:.1} GiB) host-mapped",
                    vram_layers.min(config.num_hidden_layers),
                    gib(host_bytes)
                ),
            },
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
            hipfire_config::load_summary::note(
                hipfire_config::load_summary::Facet::SpecOff,
                format!(
                    "MTP off: {moved} expert tensors host-mapped on {} ({})",
                    ctx.gpu.arch,
                    hipfire_config::mtp_host_spill_reason()
                ),
            );
        }
    }
    if placement.is_none() && moe_layers > 0 {
        hipfire_config::load_summary::note(
            hipfire_config::load_summary::Facet::Placement,
            "experts: fully resident",
        );
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
            hipfire_config::load_summary::note(
                hipfire_config::load_summary::Facet::SpecOff,
                format!(
                    "MTP off: {host_projections} expert tensors host-mapped on {} ({})",
                    ctx.gpu.arch,
                    hipfire_config::mtp_host_spill_reason()
                ),
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
    rdna_compute::load_phases::add(
        "qwen4_placement",
        rdna_compute::load_phases::ns_since(placement_started),
        0,
    );
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
    // What a discrete card keeps in host RAM: the host-mapped experts plus a
    // hot set's host tier. Only the q4 PLE prewarm paths read it, so it is
    // zero on unified memory and when neither a rank table nor a q4 sidecar
    // exists (no per-rank vector is built for the q8 path).
    let host_expert_bytes: u64 = if use_ranges || (rank_table.is_none() && q4_prewarm.is_none()) {
        0
    } else {
        residency::host_mapped_bytes_ranks(&manifest.weights, bytes_of, &ep_spans)
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
    // Freeze the settled manifest/tier ownership before `hotset_plan`, HFQ and
    // the fulfillment transaction consume their sources. Trunk tiers replace
    // the removed aggregate owners; only unchanged MTP entries remain here.
    // An `on` load settled against the dedicated reserve: the evidence is
    // sized from the reserve the placement actually used.
    let placement_inputs =
        placement_inputs.map(|(non_expert, layer_experts, reserve, mtp, gather)| {
            (
                non_expert,
                layer_experts,
                decode_share_growth.map_or(reserve, |growth| growth.3),
                mtp,
                gather,
            )
        });
    let mut resource_evidence = if let Some((non_expert, layer_experts, reserve, ..)) =
        placement_inputs.filter(|_| {
            ep.is_none()
                && config.profile.is_flash_next()
                && config.num_hidden_layers
                    == hipfire_arch_qwen4::expert_slabs::TRUNK_LAYERS as usize
                && config.num_experts
                    == hipfire_arch_qwen4::expert_slabs::EXPERTS_PER_LAYER as usize
        }) {
        let charges = qwen4_settled_charges(settled_placement_charges)?;
        let (tier_device, tier_host, stage, margin) = if let Some((plan, _)) = &hotset_plan {
            use hipfire_arch_qwen4::expert_tier_plan::Placement;
            (
                plan.owner_bytes(Placement::Device),
                plan.owner_bytes(Placement::HostMapped),
                charges.stage,
                charges.margin,
            )
        } else {
            (0, 0, charges.stage, charges.margin)
        };
        // The frozen FN table bound covers either the tier lease's owned
        // pointer/codec tables OR the whole-layer runtime's owned tables, not
        // both: tier runtime tensors borrow the lease. Stage tables are
        // distinct allocations: two pointer tables per staged trunk layer,
        // with no second codec table (the stage borrows the owner's codec).
        // MTP's own table is already in the native-MTP census.
        let host_trunk =
            hotset_plan
                .as_ref()
                .map_or(applied_layers < config.num_hidden_layers, |(plan, _)| {
                    plan.owner_bytes(hipfire_arch_qwen4::expert_tier_plan::Placement::HostMapped)
                        > 0
                });
        let stage_possible = host_trunk && residency::expert_stage_enabled(&*ctx.gpu);
        let tables = qwen4_expert_table_bytes(
            stage_possible,
            (
                config.num_hidden_layers,
                config.num_experts,
                config.expert_keep.is_some(),
            ),
        )? + hotset_plan.as_ref().map_or(Ok(0), |(plan, _)| {
            qwen4_decode_share_table_bytes(
                plan,
                (
                    config.num_hidden_layers,
                    config.num_experts,
                    config.expert_keep.is_some(),
                ),
            )
        })?;
        // Restore sources are disjoint host-only allocations, never another
        // copy of the borrowed device donors. Use the actual owning planner's
        // largest layer payload and the allocators' tail/pinned padding rules,
        // not host-mapped allocation COUNT as a counterfeit byte census.
        // A refused/disabled stage can allocate less; these remain explicitly
        // conservative component upper bounds, not measured physical bytes.
        let stage_mirror_bytes = if stage_possible && hotset_plan.is_none() {
            if matches!(
                recipe,
                hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. }
            ) {
                if slab_stage {
                    0 // Dedicated VRAM stage buffers have no restore mirrors.
                } else {
                    // Each borrowed slab prefix is <= largest payload + tail.
                    residency::PlacementCharges::new(layer_experts, true)?
                        .stage
                        .checked_add(2 * residency::HOST_MAPPED_PAD_BYTES)
                        .ok_or("qwen4: slab restore upper bound overflows")?
                }
            } else {
                // Packed donors restore gate/up and down separately: four
                // pinned allocations over two planned layer payloads.
                layer_experts
                    .checked_mul(2)
                    .and_then(|bytes| bytes.checked_add(4 * residency::HOST_MAPPED_PAD_BYTES))
                    .ok_or("qwen4: packed restore upper bound overflows")?
            }
        } else {
            0 // Tier stages never borrow device donors or allocate restores.
        };
        let routed_device_bytes =
            residency::resident_routed_expert_bytes(&manifest.weights, bytes_of)?
                .checked_add(tier_device)
                .ok_or("qwen4: routed owner union overflows")?;
        let host_mapped_bytes = residency::host_mapped_bytes(&manifest.weights, bytes_of)?
            .checked_add(tier_host)
            .ok_or("qwen4: host owner union overflows")?;
        let evidence = Qwen4SettledResourceEvidence {
            reserve,
            non_expert_device_bytes: non_expert,
            routed_device_bytes,
            host_mapped_bytes,
            tables_bytes: tables,
            stage_bytes: stage,
            margin_bytes: margin,
            host_check_bytes: host_mapped_bytes,
            stage_mirror_bytes,
            mem_available: preallocation_mem_available,
            ttm_pool: preallocation_ttm_pool,
            gtt_free_bytes: preallocation_gtt
                .map(|budget| budget.limit_bytes.saturating_sub(budget.used_bytes)),
        };
        // UMA's device owners are backed by the same RAM. The historical
        // expert-only host check missed this entire disjoint component. Charge
        // the admitted full-context census (VMM mapped granules, not virtual
        // extent), owners, stage and parked pool ONCE, plus host-only owners.
        let device_backed = non_expert
            .checked_add(routed_device_bytes)
            .and_then(|bytes| bytes.checked_add(reserve.bytes()))
            .and_then(|bytes| bytes.checked_add(tables))
            .and_then(|bytes| bytes.checked_add(stage))
            .and_then(|bytes| bytes.checked_add(margin))
            .and_then(|bytes| bytes.checked_add(preallocation_pool_bytes))
            .ok_or("qwen4: planned device component union overflows")?;
        let host_only = host_mapped_bytes
            .checked_add(stage_mirror_bytes)
            .ok_or("qwen4: planned host-only component union overflows")?;
        let host_union = host_only
            .checked_add(if use_ranges { device_backed } else { 0 })
            .ok_or("qwen4: planned shared RAM union overflows")?;
        let admitted_capture_64 = fresh_resource_basis
            && native_mtp
            && capture_plan_admitted
            && planned_capture.is_some_and(|(_, rows)| rows == 64);
        if admitted_capture_64 {
            // Explicit-layers paths use this same owning census too: reserving
            // 64 rows in a formula is not permission to materialize an
            // unfittable ring. Admission happens before owners are allocated.
            let capacity = preallocation_free_device_bytes
                .and_then(|free| free.checked_add(preallocation_pool_bytes))
                .ok_or("qwen4: original device capacity unavailable for capture admission")?;
            if device_backed > capacity {
                return Err(format!("qwen4: 64-row planned device union upper bound {device_backed} exceeds original capacity {capacity}"));
            }
        }
        if use_ranges || admitted_capture_64 {
            let host_admission = residency::check_host_ram(
                host_union,
                preallocation_mem_available,
                preallocation_ttm_pool,
            )
            .and_then(|()| residency::check_gtt_cap(host_union, preallocation_gtt));
            if let Err(reason) = host_admission {
                if admitted_capture_64
                    || state_format.qsa_f16_arena
                    || state_format.qsa_pooled_bf16
                    || hipfire_config::oom_guard_effective(Some(&ctx.gpu.arch))
                {
                    return Err(format!("qwen4: planned RAM/GTT upper bound: {reason}"));
                }
                eprintln!("  qwen4: planned RAM/GTT upper bound: {reason} (memory.oom_guard off: loading anyway; no fit snapshot)");
            }
        }
        Some(evidence)
    } else {
        None
    };
    // A rank-order PLE table's worker starts now: it reads the table's
    // hottest prefix at a capped rate while the weights load, and
    // prewarms on from there once the model is ready. The PLE hot-row
    // store also opens before the weight sweep (while the source index is
    // in hand); its worker verifies the pack meanwhile and prewarms it
    // once the model is ready. Dropping either on an error path stops and
    // joins its worker.
    let ple_workers_started = std::time::Instant::now();
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
            host_expert_bytes
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
    rdna_compute::load_phases::add(
        "ple_setup",
        rdna_compute::load_phases::ns_since(ple_workers_started),
        0,
    );
    // Every allocation of the weight sweep is immutable model data: railgun
    // check mode verifies it unchanged instead of snapshotting it (§2.4).
    let weights_role =
        hip_bridge::registry::role_scope(hip_bridge::registry::AllocationRole::Weights);
    // HIPFIRE_LOAD_WEIGHT_DIGEST: every rank/band appends its records after
    // a successful fulfillment (and the hot-set materialization below
    // appends too). Only a single-device load truncates the file; an EP/PP
    // run's caller supplies a fresh path.
    let digest_path = hipfire_config::memory::load_weight_digest_path();
    if let Some(path) = digest_path.as_deref() {
        if ep.is_none() && band.is_none() {
            hipfire_runtime::weight_store::truncate_weight_digest(path)
                .map_err(|error| format!("{}: {error}", path.display()))?;
        }
    }
    let expert_tiers = if let Some((plan, budget)) = hotset_plan {
        use hipfire_arch_qwen4::expert_tier_plan::Placement;
        let device_experts = plan.device_experts();
        let share_tail = plan.decode_share_tail().cloned();
        let tiers_io0 = rdna_compute::load_phases::thread_io();
        let tiers_started = std::time::Instant::now();
        let source = |offset, len| {
            let offset = usize::try_from(offset).map_err(|_| "hotset source offset overflow")?;
            let len = usize::try_from(len).map_err(|_| "hotset source length overflow")?;
            hfq.data_range(offset, len)
                .ok_or_else(|| format!("HIPFIRE_QWEN4_EXPERT_HOTSET: source span {offset}+{len} is outside the artifact mmap"))
        };
        let lease = if let Some(direct) = direct_source.as_ref() {
            hipfire_arch_qwen4::expert_tier_owners::materialize_direct(
                ctx.gpu, plan, direct, &source,
            )
        } else {
            hipfire_arch_qwen4::expert_tier_owners::materialize(ctx.gpu, plan, &source)
        }.map_err(|error| format!(
            "HIPFIRE_QWEN4_EXPERT_HOTSET: materialize failed after tier placement was committed (owners may be partly allocated; cannot safely fall back): {error}"
        ))?;
        let tiers_ns = rdna_compute::load_phases::ns_since(tiers_started);
        rdna_compute::load_phases::add("expert_tiers", tiers_ns, 0);
        if let (Some((read0, faults0)), Some((read1, faults1))) =
            (tiers_io0, rdna_compute::load_phases::thread_io())
        {
            rdna_compute::load_phases::add_amortized(
                "expert_tiers.storage_read",
                tiers_ns,
                read1.saturating_sub(read0),
            );
            rdna_compute::load_phases::note(
                "expert_tiers.major_faults",
                faults1.saturating_sub(faults0).to_string(),
            );
        }
        let digest = lease.logical_digest();
        use std::fmt::Write as _;
        let mut digest_hex = String::with_capacity(16);
        for byte in &digest[..8] {
            write!(&mut digest_hex, "{byte:02x}").expect("writing to a String");
        }
        let sidecar = hotset.as_ref().expect("a hotset plan has a sidecar");
        lease
            .bind_hotset_sidecar_sha256(sidecar.sha256())
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
        if let Some(tail) = share_tail.as_ref().filter(|tail| !tail.donors.is_empty()) {
            let share_digest = |view| {
                lease
                    .view_digest(view)
                    .map(|digest| digest_hex16(&digest))
                    .expect("a decode-share lease carries both views")
            };
            let mut donor_layers: Vec<u32> = tail.donors.iter().map(|id| id / 512).collect();
            donor_layers.dedup();
            // `, shadows S MiB in donor tail (F16 F MiB, INT8 I MiB)` when the
            // weight shadows are borrowed inside the tail; empty otherwise.
            let shadows_text = match (tail.shadow_bytes, decode_share_shadow_region.as_ref()) {
                (0, _) => String::new(),
                (bytes, Some(region)) => format!(
                    ", shadows {} MiB in donor tail (F16 {} MiB, INT8 {} MiB)",
                    bytes / MIB,
                    region.f16_bytes / MIB,
                    region.i8_bytes / MIB,
                ),
                (bytes, None) => format!(", shadows {} MiB in donor tail", bytes / MIB),
            };
            eprintln!(
                "  qwen4 decode-share: mode {}, {} donors ({} MiB) from {} layers, device tail [{}, {}), mirror host +{} MiB, guard {} B, borrow digest {}, resident digest {}{shadows_text}",
                decode_share_mode.name(),
                tail.donors.len(),
                tail.donor_bytes / MIB,
                donor_layers.len(),
                tail.donor_offset,
                tail.donor_offset + tail.donor_bytes,
                tail.donor_bytes / MIB,
                tail.donor_offset - tail.guard_offset,
                share_digest(hipfire_arch_qwen4::expert_tier_owners::ExpertReadView::BorrowPrefill(
                    lease.borrow_classes().saturating_sub(1)
                )),
                share_digest(hipfire_arch_qwen4::expert_tier_owners::ExpertReadView::Resident),
            );
            {
                let rows =
                    hipfire_arch_qwen4::gpu_forward::qwen4_decode_share_classes(placement_chunk);
                let donors = lease.borrow_class_donors();
                let classes = if rows.len() == donors.len() {
                    rows.iter()
                        .zip(&donors)
                        .map(|(rows, donors)| format!("{rows}:{donors}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                } else {
                    format!("all:{}", donors.last().copied().unwrap_or(0))
                };
                eprintln!("  qwen4 decode-share: classes [{classes}] donors dirty");
            }
            if let Some((grown_bytes, plain_experts, experts, _)) = decode_share_growth {
                let dedicated =
                    hipfire_arch_qwen4::gpu_forward::qwen4_decode_share_dedicated_bytes(
                        &config,
                        placement_chunk,
                        scratch_mode,
                    )
                    .unwrap_or(0);
                eprintln!(
                    "  qwen4 decode-share: on, hot budget +{} MiB (+{} experts: {} -> {}), {} donors ({} MiB) from {} layers, resident bank {} rows ({} MiB dedicated), prefill classes {:?}, restore <= {} MiB{shadows_text}",
                    grown_bytes / MIB,
                    experts.saturating_sub(plain_experts),
                    plain_experts,
                    experts,
                    tail.donors.len(),
                    tail.donor_bytes / MIB,
                    donor_layers.len(),
                    hipfire_arch_qwen4::gpu_forward::QWEN4_DECODE_SHARE_RESIDENT_ROWS,
                    dedicated / MIB,
                    hipfire_arch_qwen4::gpu_forward::qwen4_decode_share_classes(placement_chunk),
                    tail.donor_bytes / MIB,
                );
            }
            if hipfire_arch_qwen4::decode_share::decode_share_verify_enabled() {
                let prefix = usize::try_from(tail.donor_bytes)
                    .map_err(|_| "qwen4 decode-share: donor extent overflows usize".to_string())?;
                let sha = lease.verify_donor_tail(ctx.gpu, prefix).map_err(|error| {
                    format!("qwen4 decode-share: donor tail verification failed: {error}")
                })?;
                eprintln!(
                    "  qwen4 decode-share: donor tail verified sha256 {}",
                    sha.iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>()
                );
            }
        }
        eprintln!(
            "  qwen4 expert hot set identity: aggregate-MQ4 structure validated (no full artifact digest held by this loader; artifact not rehashed); model_sha256 {}, counts_sha256 {}, format_map {}, format_map_sha256 {}",
            sidecar.model_sha256(), sidecar.counts_sha256(), sidecar.format_map(), sidecar.format_map_sha256(),
        );
        Some(lease)
    } else {
        None
    };
    let decode_share_active = expert_tiers
        .as_ref()
        .is_some_and(|lease| lease.decode_share().is_some());
    let npu_identity = hipfire_arch_qwen4::npu_trunk::model_index_identity(&hfq, &ctx.gpu.arch);
    let mut source = HfqModelSource::from_hfq(hfq);
    // Preserve the incumbent direct-read path outside the admission-owned S1 arenas.
    // A sealed EP rank fulfills through `fulfill_manifest_from_payloads` on the
    // seal's mesh (no `EpFulfill` argument, no direct reader); the arenas path
    // stays single-device and refuses an EP rank before any allocation.
    let direct_weights = direct_source.is_some() && slowlink_profile.is_none() && ep.is_none();
    source.set_prefer_ranges(direct_weights);
    let weights_io0 = rdna_compute::load_phases::thread_io();
    let weights_started = std::time::Instant::now();
    let payload_source = |entry: &hipfire_runtime::weight_manifest::WeightEntry| {
        crate::qwen4_slab::qwen4_fulfillment_payload(
            &source,
            &recipe,
            entry,
            qwen4_use_range_payload(
                use_ranges
                    || (direct_weights
                        && !(ep.is_some()
                            && matches!(
                                entry.policy,
                                hipfire_runtime::weight_manifest::ShardPolicy::ExpertSharded { .. }
                            ))),
                entry.residency,
            ),
        )
    };
    let transaction = if slowlink_profile.is_some() {
        if ep.is_some() {
            return Err("qwen4: an S1 weight arena cannot use expert-parallel fulfillment".into());
        }
        let arenas = slowlink_weight_arenas.take().ok_or_else(|| {
            "qwen4: admitted S1 profile has no charged weight arena plan".to_string()
        })?;
        hipfire_runtime::weight_store::fulfill_manifest_from_payloads_with_arenas(
            &manifest.weights,
            &mesh,
            config.num_hidden_layers,
            ctx.gpu,
            expected,
            arenas,
            payload_source,
        )
    } else if ep.is_some() {
        hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
            &manifest.weights,
            &mesh,
            config.num_hidden_layers,
            ctx.gpu,
            expected,
            payload_source,
        )
    } else {
        hipfire_runtime::weight_store::fulfill_manifest_from_payloads_direct(
            &manifest.weights,
            &mesh,
            config.num_hidden_layers,
            ctx.gpu,
            expected,
            direct_source.as_ref(),
            payload_source,
        )
    }
    .map_err(|error| format!("qwen4: manifest fulfillment failed: {error}"))?;
    if let Some(path) = digest_path.as_deref() {
        match hipfire_runtime::weight_store::write_weight_digest(path, &transaction, &*ctx.gpu) {
            Ok(records) => eprintln!(
                "  qwen4 weight digest: {records} records appended to {}",
                path.display()
            ),
            Err(error) => {
                let _ = transaction.rollback(&mut *ctx.gpu);
                return Err(format!("qwen4: weight digest {}: {error}", path.display()));
            }
        }
    }
    let weights_ns = rdna_compute::load_phases::ns_since(weights_started);
    rdna_compute::load_phases::add("weights", weights_ns, 0);
    if let (Some((read0, faults0)), Some((read1, faults1))) =
        (weights_io0, rdna_compute::load_phases::thread_io())
    {
        rdna_compute::load_phases::add_amortized(
            "weights.storage_read",
            weights_ns,
            read1.saturating_sub(read0),
        );
        rdna_compute::load_phases::note(
            "weights.major_faults",
            faults1.saturating_sub(faults0).to_string(),
        );
    }
    drop(weights_role);
    let scope = band
        .cloned()
        .unwrap_or_else(|| hipfire_arch_qwen4::weights::Qwen4BandScope::single(&config));
    let assemble_started = std::time::Instant::now();
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
    // Preserve the candidate before attachment consumes it, then bind it to
    // the settled application marker after `attach_forward`. Mere presence or
    // successful attachment is not proof that digest/pre-checks applied it.
    let applied_overlay = trunk_overlay.as_ref().map(|overlay| {
        (
            overlay.path().to_path_buf(),
            overlay.bound().payload_bytes(),
        )
    });
    if let Some(overlay) = trunk_overlay {
        if let Err(error) = bundle.attach_trunk_overlay(overlay) {
            let detail = error.to_string();
            let _ = bundle.free_gpu(ctx.gpu);
            return Err(format!("qwen4: trunk overlay setup failed: {detail}"));
        }
    }
    rdna_compute::load_phases::add(
        "qwen4_assemble",
        rdna_compute::load_phases::ns_since(assemble_started),
        0,
    );
    let forward_started = std::time::Instant::now();
    // The immutable S1 profile the placement above was charged for: the forward
    // allocates its prompt arena and expert page store from it.
    if let Some(profile) = slowlink_profile {
        bundle.attach_slowlink_profile(profile);
    }
    let overlay_marker_was_set = bundle.config.trunk_iu4_default.is_some();
    if let Err(error) = bundle.attach_forward(ctx.gpu, placement_chunk, scratch_mode) {
        let detail = error.to_string();
        let _ = bundle.free_gpu(ctx.gpu);
        return Err(format!("qwen4: forward setup failed: {detail}"));
    }
    // `apply_pending_trunk_overlay` writes this marker only on Applied. A
    // pre-existing marker cannot distinguish Applied from Refused, so that
    // source shape stays explicitly ambiguous rather than hashing a candidate.
    let overlay_identity_ambiguous = applied_overlay.is_some() && overlay_marker_was_set;
    let applied_overlay = applied_overlay.filter(|_| bundle.config.trunk_iu4_default.is_some());
    rdna_compute::load_phases::add(
        "qwen4_forward_state",
        rdna_compute::load_phases::ns_since(forward_started),
        0,
    );
    if let Some(scope) = band {
        // `attach_forward` searches smaller chunk rungs when the largest does
        // not fit; a band never shrinks the canonical chunk on its own. A band
        // keeps the placement chunk.
        let canonical = placement_chunk;
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
    // The CPU splice (D1) and the D2 split arm inside `attach_forward` (D1 and
    // the tier split by `memory.offload_exec=cpu`, the split also by the
    // per-load measured D2 default), so the loaded placement is only known
    // here; the native-MTP admission above is unchanged by it.
    let split_kinds = bundle.split_owner_kinds();
    if !split_kinds.is_empty() {
        let mut kinds = std::collections::BTreeMap::<&str, usize>::new();
        for (_, kind) in &split_kinds {
            *kinds.entry(kind).or_default() += 1;
        }
        let by_kind: Vec<String> = kinds
            .iter()
            .map(|(kind, n)| format!("{n} {kind}"))
            .collect();
        let source = if bundle.has_cpu_deferred_experts() {
            "offload_exec=cpu"
        } else {
            "measured D2 default"
        };
        // The cold return, readiness and join are reported from the per-layer
        // effective modes; the request is the arch default on gfx1100 (mapped +
        // cpwait, explicit env wins), so a load that asked for `mapped` /
        // `cpwait` (or has such a layer) says so and names the provenance.
        let transport = hipfire_config::memory::qwen4_d2_transport(&ctx.gpu.arch).ok();
        let provenance = match transport {
            Some(t) if t.arch_default => {
                format!("; transport request: default ({})", ctx.gpu.arch)
            }
            _ => String::new(),
        };
        let return_modes = bundle.split_return_modes();
        let mapped = return_modes
            .iter()
            .filter(|(_, mode)| *mode == "mapped")
            .count();
        let cold_return = if mapped > 0
            || transport.is_some_and(|t| t.ret == hipfire_config::memory::D2Return::Mapped)
        {
            format!(
                "; cold return: {mapped} mapped, {} copy",
                return_modes.len() - mapped
            )
        } else {
            String::new()
        };
        // Likewise the readiness and join mechanisms.
        let ready_modes = bundle.split_ready_modes();
        let event = ready_modes
            .iter()
            .filter(|(_, mode)| *mode == "event")
            .count();
        let ready = if event > 0
            || hipfire_config::memory::qwen4_d2_ready()
                == Ok(hipfire_config::memory::D2Ready::Event)
        {
            format!(
                "; input ready: {event} event, {} sync",
                ready_modes.len() - event
            )
        } else {
            String::new()
        };
        let join_modes = bundle.split_join_modes();
        let cpwait = join_modes
            .iter()
            .filter(|(_, mode)| *mode == "cpwait")
            .count();
        let join = if cpwait > 0
            || transport.is_some_and(|t| t.join == hipfire_config::memory::D2Join::CpWait)
        {
            format!(
                "; join: {cpwait} cpwait, {} host",
                join_modes.len() - cpwait
            )
        } else {
            String::new()
        };
        eprintln!(
            "  qwen4 routed experts: CPU split (D2) armed on {} layers ({}; {source}{cold_return}{ready}{join}{provenance}): decode and <=4-row verify run their host-placed cold experts on the CPU; wider verify needs the qualified lab-wide CPU window (default 64 when admitted+prewarmed)",
            split_kinds.len(),
            by_kind.join(", ")
        );
    }
    let split_layers = split_kinds.len();
    // The measured D2 default of an unset `memory.offload_exec` is decided inside
    // `attach_forward`, after the donor tail was materialized: the one admission
    // condition that cannot be applied at the plan point. A shared plan cannot
    // fall back to the plain placement here, so a CPU split on any layer fails
    // the load with the way out.
    if decode_share_active && (split_layers > 0 || bundle.has_cpu_deferred_experts()) {
        let _ = bundle.free_gpu(ctx.gpu);
        return Err(format!(
            "qwen4 decode-share: the D2 CPU expert split armed on {split_layers} layers after the donor tail was bound; \
             set HIPFIRE_QWEN4_D2_DEFAULT=0 (or {}=off) so decode-share and the CPU split are not combined",
            hipfire_arch_qwen4::decode_share::DECODE_SHARE_ENV
        ));
    }
    if bundle.has_cpu_deferred_experts() && split_layers == 0 {
        eprintln!(
            "  qwen4 routed experts: CPU splice armed (offload_exec=cpu): decode and <=4-row verify run host-placed experts on the CPU; wider verify needs the qualified lab-wide CPU window (default 64 when admitted+prewarmed)"
        );
    }
    let initial_capture_rows = bundle.native_mtp_capture_rows();
    // CPU lab plan (BEFORE any physical-cap read): the lab-wide 64-row need is the
    // planned native-MTP window `capture_reserve_drafts + 1` saturating the physical
    // bound — NOT the settled ring (which the exclusion below never grows on a
    // CPU-owned load) and NOT the stored 0/4 physical cap (which must not clip the
    // request before the prewarm). Scope-gated: eligible single-device non-EP/PP/lane
    // Flash-Next, explicit CPU ownership (D1 splice or D2 split armed), override
    // present, real `max_k` drafts; else 0 (legacy default: nothing reserved).
    let cpu_lab_need_rows = {
        let eligible = band.is_none()
            && ep.is_none()
            && bundle.config.profile.is_flash_next()
            && bundle.lanes().is_none()
            && (bundle.has_cpu_deferred_experts() || bundle.native_mtp_cpu_window_rows().is_some())
            && rdna_compute::prefill_invariant::qwen4_wide_verify_override_present(&ctx.gpu.arch);
        if eligible {
            capture_reserve_drafts
                .saturating_add(1)
                .min(hipfire_config::QWEN4_VERIFY_MAX_ROWS)
        } else {
            0
        }
    };
    if native_mtp {
        let mtp_started = std::time::Instant::now();
        if let Err(error) = bundle.attach_mtp(ctx.gpu, ctx.max_seq) {
            let detail = error.to_string();
            let _ = bundle.free_gpu(ctx.gpu);
            return Err(format!("qwen4: MTP setup failed: {detail}"));
        }
        // Same proposal bound the existing reserve admitted; never promote an
        // 8-row plan or allocate 64 for PP/EP/non-FN/lane execution. A qualified
        // single-FN CPU lab load with a 64-row plan reserves the capture ring even
        // though `native_mtp_cpu_window_rows()` is `Some` (CPU-owned): the previous
        // exclusion starved exactly the loads the prewarm below serves.
        let cpu_lab_capture =
            cpu_lab_need_rows > hipfire_arch_qwen4::program::QWEN4_CPU_SPLICE_MAX_ROWS;
        if band.is_none()
            && ep.is_none()
            && bundle.config.profile.is_flash_next()
            && (bundle.native_mtp_cpu_window_rows().is_none() || cpu_lab_capture)
            && bundle.lanes().is_none()
            && capture_plan_admitted
            && initial_capture_rows == 0
            && fresh_resource_basis
            && resource_evidence.is_some()
        {
            if let Err(error) = bundle.prepare_native_mtp_capture(ctx.gpu, capture_reserve_drafts) {
                let detail = error.to_string();
                let _ = bundle.free_gpu(ctx.gpu);
                return Err(format!(
                    "qwen4: admitted MTP capture setup failed: {detail}"
                ));
            }
        }
        rdna_compute::load_phases::add(
            "qwen4_mtp_attach",
            rdna_compute::load_phases::ns_since(mtp_started),
            0,
        );
    }
    // Actual CPU-owned expert storage for an admitted lab-wide window (D1 splice
    // and/or D2 split): the checked REQUESTED plan is consumed BEFORE any allocation
    // (host/device budget charge), then real reservations land on the execution
    // thread before the first window — never a metadata claim. The request is the
    // admitted plan (`cpu_lab_need_rows`, the admitted capture drafts + 1 saturating
    // at the physical bound) — NOT the settled ring, NOT the stored physical cap, and
    // NEVER the GPU arm width (a wide8 arm with an admitted 64-row window needs the
    // same CPU64 backend; the CPU request is not clipped to 4/8 before allocation).
    // Legacy default loads (no override, small K, unadmitted capture) request nothing
    // and keep on-demand growth with byte-identical placement. A refused plan, charge,
    // prewarm or proof is a load failure, never a silent narrow fallback: anything
    // narrower would strand the admitted capture below a refused window.
    if band.is_none() && ep.is_none() {
        let override_present =
            rdna_compute::prefill_invariant::qwen4_wide_verify_override_present(&ctx.gpu.arch);
        // D1 scope: the single non-EP/PP/lane Flash-Next load with an armed D1 splice
        // whose allocated GPU scratch holds the plan need (the frozen required-rows
        // formula below turns this scope into real rows or 0).
        let d1_scope = bundle.has_cpu_deferred_experts()
            && bundle.config.profile.is_flash_next()
            && bundle.lanes().is_none()
            // Allocated GPU scratch covers the plan need (`spec_chunk_rows` is the
            // attached `max_chunk`; spec logit rows saturate at the physical 64,
            // so `max_chunk >= need` with `need <= 64` covers both).
            && bundle.spec_chunk_rows().unwrap_or(0) >= cpu_lab_need_rows;
        let d1_rows = hipfire_arch_qwen4::expert_residency::cpu_d1_required_storage_rows(
            override_present,
            d1_scope,
            cpu_lab_need_rows,
        );
        // D2 scope: the same single-device Flash-Next shape with at least one armed
        // split executor, the lab-wide override present and a real admitted need.
        // D2-only loads reserve their own executor storage (plus the shared host-slot
        // scratch, needed by D2 as well) and never prewarm the D1 pinned stage.
        let d2_need = if split_layers > 0
            && bundle.config.profile.is_flash_next()
            && bundle.lanes().is_none()
            && override_present
            && cpu_lab_need_rows > 0
        {
            cpu_lab_need_rows
        } else {
            0
        };
        let req = d1_rows.max(d2_need);
        // At or below the legacy narrow cap the default-4 executors and the on-demand
        // D1 splice already cover the window: no plan, no charge, no reserve.
        if req > hipfire_dispatch::cpu_split::CPU_SPLIT_MAX_ROWS {
            // Checked REQUESTED storage plan BEFORE any allocation (the sum of actual
            // armed D2 per-layer requested bytes + the D1 owned stage + host scratch,
            // with the GPU views and capture reconciled but never charged twice — all
            // existing authorities): `None` on overflow or uncovered views is a refusal
            // before anything is allocated.
            let Some(plan) = bundle.cpu_requested_storage_bytes(req, d1_rows) else {
                let _ = bundle.free_gpu(ctx.gpu);
                return Err(format!(
                    "qwen4: admitted CPU window of {req} rows has no checked storage plan"
                ));
            };
            // Charge the plan's host/device footprint BEFORE allocating, with the
            // existing budget patterns on live readings. D2 replacements free the old
            // executor before building the new one (sequential, in the GF body), so no
            // old+new overlap peak is charged — only the new allocations.
            let Some(host_pinned_bytes) = (plan.stage_bytes as u64)
                .checked_add(plan.d2_input_bytes as u64)
                .and_then(|bytes| bytes.checked_add(plan.d2_mask_bytes as u64))
                .and_then(|bytes| bytes.checked_add(plan.d2_output_bytes as u64))
                .and_then(|bytes| bytes.checked_add(plan.d2_mapped_bytes as u64))
            else {
                let _ = bundle.free_gpu(ctx.gpu);
                return Err(format!(
                    "qwen4: admitted CPU window of {req} rows overflows the host charge"
                ));
            };
            let Some(host_inc_bytes) =
                host_pinned_bytes.checked_add(plan.host_scratch_bytes as u64)
            else {
                let _ = bundle.free_gpu(ctx.gpu);
                return Err(format!(
                    "qwen4: admitted CPU window of {req} rows overflows the host charge"
                ));
            };
            let current_mem_available = rdna_compute::kv_slots::mem_available_bytes();
            if let Err(reason) = residency::check_host_ram(
                host_inc_bytes,
                current_mem_available,
                residency::ttm_pool_estimate_on(&*ctx.gpu),
            ) {
                let _ = bundle.free_gpu(ctx.gpu);
                return Err(format!(
                    "qwen4: admitted CPU window of {req} rows does not fit host RAM: {reason}"
                ));
            }
            if let Err(reason) =
                residency::check_gtt_cap(host_pinned_bytes, residency::gtt_budget())
            {
                let _ = bundle.free_gpu(ctx.gpu);
                return Err(format!(
                    "qwen4: admitted CPU window of {req} rows does not fit the GTT cap: {reason}"
                ));
            }
            if let Err(error) = ctx.gpu.emu_check_host_allocation(host_pinned_bytes, true) {
                let _ = bundle.free_gpu(ctx.gpu);
                return Err(format!(
                    "qwen4: admitted CPU window host charge refused: {error}"
                ));
            }
            // Device: one `masked_indices` malloc per D2 layer (the transfer buffers
            // above are host-pinned; the mapped return is host-mapped).
            let device_inc_bytes = plan.d2_mask_bytes as u64;
            if device_inc_bytes > 0 {
                let free_device = match ctx.gpu.device_mem_info() {
                    Ok((free, _)) => u64::try_from(free).unwrap_or(u64::MAX),
                    Err(error) => {
                        let _ = bundle.free_gpu(ctx.gpu);
                        return Err(format!(
                            "qwen4: device memory query for the CPU window: {error}"
                        ));
                    }
                };
                if free_device < device_inc_bytes {
                    let _ = bundle.free_gpu(ctx.gpu);
                    return Err(format!(
                        "qwen4: admitted CPU window of {req} rows needs {device_inc_bytes} device bytes, {free_device} free"
                    ));
                }
            }
            // D1 prewarm: the pinned transfer stage plus the host-slot scratch, then the
            // live capacity proof publishes the stored window. Unchanged reservations,
            // now behind the consumed plan above.
            if d1_rows > 0 {
                let Some((dim, mi, top_k)) = bundle.cpu_d1_geometry() else {
                    let _ = bundle.free_gpu(ctx.gpu);
                    return Err("qwen4: admitted D1 window has no expert geometry".to_string());
                };
                let Some(slots) =
                    hipfire_arch_qwen4::expert_residency::cpu_d1_host_slots(d1_rows, top_k)
                else {
                    let _ = bundle.free_gpu(ctx.gpu);
                    return Err(format!(
                        "qwen4: admitted D1 window of {d1_rows} rows overflows the host-slot extent"
                    ));
                };
                if let Err(error) =
                    hipfire_dispatch::cpu_exec::reserve_moe_cpu_host_slot_scratch(dim, mi, slots)
                {
                    let detail = error.to_string();
                    let _ = bundle.free_gpu(ctx.gpu);
                    return Err(format!(
                        "qwen4: admitted D1 host-slot prewarm failed: {detail}"
                    ));
                }
                let Some(values) =
                    hipfire_arch_qwen4::expert_residency::cpu_d1_stage_values(dim, top_k, d1_rows)
                else {
                    let _ = bundle.free_gpu(ctx.gpu);
                    return Err(format!(
                        "qwen4: admitted D1 window of {d1_rows} rows overflows the transfer extent"
                    ));
                };
                if let Err(error) =
                    hipfire_dispatch::cpu_exec::reserve_cpu_transfer_values(ctx.gpu, values)
                {
                    let detail = error.to_string();
                    let _ = bundle.free_gpu(ctx.gpu);
                    return Err(format!(
                        "qwen4: admitted D1 transfer prewarm failed: {detail}"
                    ));
                }
                if !bundle.note_cpu_d1_wide_storage_prewarmed(d1_rows, ctx.gpu.device_id) {
                    let _ = bundle.free_gpu(ctx.gpu);
                    return Err(format!(
                        "qwen4: admitted D1 window of {d1_rows} rows failed live capacity proof"
                    ));
                }
                eprintln!("  qwen4 D1 CPU experts: prewarmed {d1_rows}-row window ({slots} slots)");
            }
            // Actual D2 executor prewarm: the shared host-slot scratch for the requested
            // window (max-based, a no-op when the D1 reserve above already covers it) plus
            // the real per-layer executor resize. The D1 pinned transfer stage is skipped
            // for D2-only loads.
            if d2_need > 0 {
                let Some((dim, mi, top_k)) = bundle.cpu_d1_geometry() else {
                    let _ = bundle.free_gpu(ctx.gpu);
                    return Err("qwen4: admitted D2 window has no expert geometry".to_string());
                };
                let Some(slots) =
                    hipfire_arch_qwen4::expert_residency::cpu_d1_host_slots(req, top_k)
                else {
                    let _ = bundle.free_gpu(ctx.gpu);
                    return Err(format!(
                        "qwen4: admitted D2 window of {req} rows overflows the host-slot extent"
                    ));
                };
                if let Err(error) =
                    hipfire_dispatch::cpu_exec::reserve_moe_cpu_host_slot_scratch(dim, mi, slots)
                {
                    let detail = error.to_string();
                    let _ = bundle.free_gpu(ctx.gpu);
                    return Err(format!(
                        "qwen4: admitted D2 host-slot prewarm failed: {detail}"
                    ));
                }
                match bundle.reserve_cpu_d2_wide_rows(ctx.gpu, req) {
                    Ok(true) => {
                        eprintln!(
                            "  qwen4 D2 CPU experts: prewarmed {req}-row window ({slots} slots)"
                        );
                    }
                    Ok(false) => {}
                    Err(error) => {
                        let detail = error.to_string();
                        let _ = bundle.free_gpu(ctx.gpu);
                        return Err(format!(
                            "qwen4: admitted D2 window prewarm failed: {detail}"
                        ));
                    }
                }
            }
            // Reconcile the ACTUAL post-reserve census with the plan and the live
            // capacity: the common capacity must cover the request and no settled byte
            // term may exceed its planned upper bound. The consumed census then joins
            // the settled resource evidence so the published snapshot accounts the real
            // CPU window (never part of the pre-allocation placement census).
            let Some(actual) = bundle.cpu_storage_bytes() else {
                let _ = bundle.free_gpu(ctx.gpu);
                return Err(format!(
                    "qwen4: admitted CPU window of {req} rows has no settled storage census"
                ));
            };
            let fits_plan = actual.rows >= req
                && actual.slots <= plan.slots
                && actual.stage_values <= plan.stage_values
                && actual.stage_bytes <= plan.stage_bytes
                && actual.d2_input_bytes <= plan.d2_input_bytes
                && actual.d2_mask_bytes <= plan.d2_mask_bytes
                && actual.d2_output_bytes <= plan.d2_output_bytes
                && actual.d2_transfer_bytes <= plan.d2_transfer_bytes
                && actual.d2_mapped_bytes <= plan.d2_mapped_bytes
                && actual.host_scratch_bytes <= plan.host_scratch_bytes;
            if !fits_plan {
                let _ = bundle.free_gpu(ctx.gpu);
                return Err(format!(
                    "qwen4: admitted CPU window of {req} rows settled outside its storage plan"
                ));
            }
            if let Some(evidence) = resource_evidence.as_mut() {
                let settled_host_bytes = (actual.stage_bytes as u64)
                    .checked_add(actual.d2_input_bytes as u64)
                    .and_then(|bytes| bytes.checked_add(actual.d2_mask_bytes as u64))
                    .and_then(|bytes| bytes.checked_add(actual.d2_output_bytes as u64))
                    .and_then(|bytes| bytes.checked_add(actual.d2_mapped_bytes as u64))
                    .and_then(|bytes| bytes.checked_add(actual.host_scratch_bytes as u64));
                let Some(settled_host_bytes) = settled_host_bytes else {
                    let _ = bundle.free_gpu(ctx.gpu);
                    return Err(format!(
                        "qwen4: admitted CPU window of {req} rows overflows the settled host charge"
                    ));
                };
                let Some(total) = evidence.host_check_bytes.checked_add(settled_host_bytes) else {
                    let _ = bundle.free_gpu(ctx.gpu);
                    return Err(format!(
                        "qwen4: admitted CPU window of {req} rows overflows the settled host census"
                    ));
                };
                evidence.host_check_bytes = total;
            }
        }
    }
    let post_load_started = std::time::Instant::now();
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
    // An uncapped `auto` prewarm is for a discrete GPU whose routed experts
    // actually sit in host RAM (host-mapped bytes, a hot set's host tier, or
    // an armed CPU splice/tier split); unified memory and a fully resident
    // discrete card keep the cap.
    let mut ple_prewarm = ple_prewarm_budget(
        use_ranges,
        host_expert_bytes > 0 || bundle.has_cpu_deferred_experts(),
    );
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
                    )
                    .filter(|(_, rows)| bundle.native_mtp_capture_rows() < *rows),
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
    // worker prewarms the hottest rows into the page cache, only while no
    // request is in flight (each request holds a `ServingGuard` from
    // admission to its terminal event, so the prewarm pauses for the whole
    // request, not per PLE fetch, and resumes once it has been idle).
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
    rdna_compute::load_phases::add(
        "qwen4_post_load",
        rdna_compute::load_phases::ns_since(post_load_started),
        0,
    );
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
    // Measured-load identity: composed ONCE per eligible rank here, after the
    // actual forward/scope/attachment settled (MTP head, session cache, PLE
    // stores, forward program) and before the rank return. Policy-independent:
    // composition never reads the FN chooser key and never enables the
    // economic policy — that gate stays at the actual controller/drafter,
    // which keeps the real incumbent whenever the flag is off or no
    // descriptor binds. An eligible profile load and the incumbent bind the
    // SAME once-per-load Arc. Supported loads with real verified evidence
    // construct `Some`; unsupported/missing seals stay `None` with a named
    // log reason and the ordinary load is unaffected. No full-model or
    // per-window hashing; the verified cache, not an expected pin, is the
    // only source of SHA evidence. The resource snapshot below composes the
    // retained planner/owner union against this same Arc after capture settles.
    let measured_load_identity = {
        // Bands and EP ranks never carry the descriptor: a band's settled
        // scope is the band fragment (band loads attach their own session
        // caches and the MTP scope needs the head), and an EP rank attaches
        // single-scope state on its own device — so both defer to the
        // ordinary incumbent with a named reason.
        if band.is_some() {
            eprintln!("  qwen4 measured load: unavailable (topology.pp band loads carry no measured descriptor)");
            None
        } else if ep.is_some() {
            eprintln!("  qwen4 measured load: unavailable (expert-parallel ranks carry no measured descriptor)");
            None
        } else if bundle.lanes().is_some()
            || !bundle.config.profile.is_flash_next()
            || !native_mtp
            || bundle.native_mtp_cpu_window_rows().is_some()
        {
            eprintln!("  qwen4 measured load: unavailable (requires single FN native MTP without lanes or CPU splice/split)");
            None
        } else if overlay_identity_ambiguous {
            eprintln!("  qwen4 measured load: unavailable (pre-existing trunk overlay marker cannot prove this candidate was applied)");
            None
        } else {
            // PLE evidence uses the APPLIED source only: a bound sidecar rows
            // source hashes the actual sidecar file; an embedded source is
            // honestly ineligible (never a placeholder SHA). The preserved
            // `(path, bytes, encoding)` triple below names the same applied
            // evidence attachments were built from (captured before
            // `ple_rows` moved into attachment) — no device copy, no re-hash.
            enum PleEvidence {
                // Verified sidecar evidence ready to hash-check at composition.
                Ready {
                    sha: String,
                    bytes: u64,
                    encoding: String,
                },
                // Applied source exists but lacks a verified seal: the
                // factory call below seals `None` naming the sidecar.
                UnsealedSidecar,
                // Embedded or absent rows: honestly ineligible under the
                // current FN fixture contract, never a fabricated digest.
                Ineligible,
            }
            let ple_evidence = if applied_ple_rows_is_sidecar {
                let cache = verified_cache_for_load();
                match qwen4_verified_sha(&cache, &ple_rows_path, "applied PLE sidecar") {
                    Some(sha) => PleEvidence::Ready {
                        sha,
                        bytes: ple_rows_bytes,
                        encoding: rows_encoding.clone(),
                    },
                    None => PleEvidence::UnsealedSidecar,
                }
            } else {
                PleEvidence::Ineligible
            };
            match ple_evidence {
                PleEvidence::Ineligible => {
                    eprintln!("  qwen4 measured load: unavailable (applied PLE source is not a verified sidecar; embedded/absent rows are ineligible, never a fabricated digest)");
                    None
                }
                PleEvidence::UnsealedSidecar => None,
                PleEvidence::Ready {
                    sha,
                    bytes,
                    encoding,
                } => {
                    // Model-file bytes from the live file length (the same
                    // file the verified SHA above was sealed against).
                    let model_bytes = std::fs::metadata(ctx.path)
                        .map(|metadata| metadata.len())
                        .unwrap_or(0);
                    // Actual state-ring allocation, never the sizing prediction.
                    let capture_rows = bundle.native_mtp_capture_rows();
                    // Device identity with explicit error handling: a HIP
                    // UUID failure seals `None` instead of an empty-string
                    // identity field.
                    let gpu_uuid = match ctx.gpu.hip.device_uuid(ctx.gpu.device_id) {
                        Ok(uuid) => Some(uuid),
                        Err(error) => {
                            eprintln!("  qwen4 measured load: unavailable (device UUID query failed: {error})");
                            None
                        }
                    };
                    match &gpu_uuid {
                        None => None,
                        Some(gpu_uuid) => {
                            let (overlay_path, overlay_bytes) = match applied_overlay {
                                Some((path, bytes)) => (Some(path), Some(bytes)),
                                None => (None, None),
                            };
                            // Hot-set evidence uses the actual applied sidecar path;
                            // its file length is the byte evidence the applied plan
                            // was built from (the plan, not a filename guess, proves
                            // it was applied).
                            let (hotset_path_opt, hotset_bytes) = match hotset_path
                                .as_deref()
                                .filter(|_| bundle.weights.expert_tiers().is_some())
                            {
                                Some(path) => {
                                    let bytes =
                                        std::fs::metadata(path).map(|metadata| metadata.len()).ok();
                                    (Some(path.to_path_buf()), bytes)
                                }
                                None => (None, None),
                            };
                            let overlay_path_ref = overlay_path.as_deref();
                            let hotset_path_ref = hotset_path_opt.as_deref();
                            let cache = verified_cache_for_load();
                            qwen4_measured_load_identity(
                                &bundle,
                                std::path::Path::new(ctx.path),
                                model_bytes,
                                &sha,
                                bytes,
                                &encoding,
                                overlay_path_ref,
                                overlay_bytes,
                                hotset_path_ref,
                                hotset_bytes,
                                max_seq,
                                capture_rows,
                                &ctx.gpu.arch,
                                gpu_uuid,
                                &ctx.gpu.flags,
                                &cache,
                            )
                        }
                    }
                }
            }
        }
    };
    // Publish the typed snapshot only from the actual settled capture and the
    // single retained planner/owner census, sharing the descriptor's Arc.
    let measured_resource_snapshot = if !fresh_resource_basis || initial_capture_rows != 0 {
        eprintln!("  qwen4 measured resources: unavailable (retained/reused Gpu or prior {initial_capture_rows}-row ring has no original fresh-load preload basis)");
        None
    } else if let (Some(identity), Some(evidence)) =
        (measured_load_identity.as_ref(), resource_evidence.take())
    {
        qwen4_measured_resource_snapshot(
            identity,
            &*ctx.gpu,
            &bundle,
            preallocation_pool_bytes,
            preallocation_free_device_bytes,
            evidence,
            max_seq,
            bundle.spec_chunk_rows().unwrap_or(0),
            bundle.native_mtp_capture_rows(),
            &bundle.config,
            state_format.gdn,
            bundle.state.qsa_backend().name(),
            bundle.forward_scratch_mode().unwrap_or(scratch_mode),
        )
    } else {
        eprintln!("  qwen4 measured resources: unavailable (single-rank settled identity or placement evidence unavailable)");
        None
    };
    let card_cap = ctx.sequence.map(|sequence| sequence.card_cap);
    Ok(Qwen4RankLoad {
        bundle,
        native_mtp,
        max_k,
        max_seq,
        card_cap,
        measured_load_identity,
        measured_resource_snapshot,
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
/// one split-and-chunk decision ([`crate::admission::qwen4_pp_auto_fit`]: the
/// per-rank capacity census against each rank's free VRAM, refusing naming
/// `topology.pp` before any allocation; it keeps the admitted split at the
/// requested chunk whenever that fits and otherwise steps the chunk down and
/// rebalances the bands, never touching an explicit `HIPFIRE_PP_LAYERS` or a
/// pinned `HIPFIRE_PREFILL_CHUNK_ROWS`), edge provisioning sized to the
/// chosen canonical chunk's HC streams, then one band-scoped load per rank
/// (stage 0 on the admitted source, the rest on a fresh source each).
fn load_qwen4_pp_bundle(
    daemon: &mut rdna_compute::Gpu,
    admitted: &hipfire_runtime::mesh_placement::AdmittedMesh,
    placement: &hipfire_runtime::mesh_placement::BandPlacement,
    explicit_layers: bool,
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
    let (chosen, scopes, edge_bytes, chunk_rows) = {
        let ModelSource::Hfq(hfq) = &src else {
            return Err(crate::admission::QWEN4_SAFETENSORS_REFUSAL.into());
        };
        let receipt = q4::admit_hfqm_artifact_on(hfq, Some(&daemon.arch))
            .map_err(|error| error.to_string())?;
        q4::bundle::admit_executable_profile(&receipt.config)
            .map_err(|error| format!("qwen4: {error}"))?;
        let config = receipt.config.clone();
        let state_format = q4::resolve_state_format(
            &shared.kv_mode,
            shared.state_quant.as_deref().unwrap_or(""),
            daemon,
            &config,
        )?;
        let scratch_mode =
            q4::artifact::source_forward_scratch_mode(daemon, hfq, &config, &receipt.manifest)
                .map_err(|error| error.to_string())?;
        let requested_chunk =
            q4::gpu_forward::qwen4_pp_prefill_chunk_requested(daemon, shared.max_seq, scratch_mode);
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
        // The one split-and-chunk decision of this load
        // ([`crate::admission::qwen4_pp_auto_fit`]): the admitted split at the
        // requested chunk whenever it fits (today's load, unchanged), else the
        // largest chunk some split fits and the most balanced such split. An
        // explicit `HIPFIRE_PP_LAYERS` is never searched, and a pinned
        // `HIPFIRE_PREFILL_CHUNK_ROWS` only has its split searched.
        let admitted_counts: Vec<usize> = (0..stages)
            .map(|stage| placement.band(stage).len())
            .collect();
        let n_layers = placement.n_trunk_layers;
        let band_placement = |counts: &[usize]| {
            hipfire_runtime::mesh_placement::BandPlacement::from_counts(
                &admitted.mesh,
                n_layers,
                counts,
            )
            .and_then(|bands| {
                bands.with_first_band_min(crate::admission::QWEN4_PP_FIRST_BAND_MIN_LAYERS)
            })
            .map_err(|error| format!("qwen4: topology.pp: {error}"))
        };
        let fit = crate::admission::qwen4_pp_auto_fit(
            &admitted_counts,
            explicit_layers,
            crate::admission::QWEN4_PP_FIRST_BAND_MIN_LAYERS,
            requested_chunk,
            q4::gpu_forward::qwen4_prefill_chunk_rows_pinned(),
            |counts, chunk_rows| {
                let candidate = band_placement(counts)?;
                crate::admission::qwen4_pp_rank_lines(&crate::admission::Qwen4PpBudgetInputs {
                    config: &config,
                    placement: &candidate,
                    weights: &receipt.manifest.weights,
                    bytes_of: &bytes_of,
                    format: state_format,
                    context: &context,
                    chunk_rows,
                    mode: scratch_mode,
                    free_per_rank: &free_per_rank,
                    reserve_bytes,
                    native_mtp: native_mtp_inputs,
                })
            },
        )?;
        if let Some(line) = fit.log_line() {
            eprintln!("{line}");
        }
        for line in &fit.fits {
            eprintln!(
                "  qwen4 PP rank {}: layers {:?} need {} MiB of {} MiB free{}",
                line.stage,
                line.band,
                line.required >> 20,
                line.free >> 20,
                if native_mtp {
                    format!(" (incl. {} MiB native MTP)", line.native_mtp >> 20)
                } else {
                    String::new()
                }
            );
        }
        // From here the split and the chunk are fixed for every band, the
        // edge buffers and the mesh.
        let chunk_rows = fit.chunk_rows;
        let chosen = band_placement(&fit.counts)?;
        let scopes = (0..stages)
            .map(|stage| q4::weights::Qwen4BandScope::from_placement(&config, &chosen, stage))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("qwen4: topology.pp: {error}"))?;
        let edge_bytes = crate::admission::qwen4_pp_edge_max_bytes(&config, chunk_rows)?;
        (chosen, scopes, edge_bytes, chunk_rows)
    };
    let placement = &chosen;
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
                Some(chunk_rows),
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
        continuous_batch: false,
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
    let (bundle, gpus, native_mtp) = load_qwen4_pp_bundle(
        ctx.gpu,
        admitted,
        placement,
        // `HIPFIRE_PP_LAYERS` (admission's `explicit_counts`): the
        // operator's split is never searched.
        ctx.pp_layer_counts.is_some(),
        &shared,
        session,
        src,
    )?;
    let max_seq = ctx.max_seq;
    let speculator = native_mtp.then(|| {
        // PP mesh: unsupported shape for the measured descriptor; the drafter
        // keeps the real incumbent (None) by construction.
        qwen4_native_mtp_speculator(
            max_k,
            qwen4_explicit_mtp_k(ctx.spec),
            max_seq,
            &ctx.gpu.arch,
            meta.tokenizer.special_token_id("<|im_end|>"),
            meta.tokenizer.special_token_id("</think>"),
            false,
            rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(&ctx.gpu.arch),
            bundle.state_format().gdn,
            None,
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
        KvPair::Native(k) => {
            let name = kv_mode::qwen_k_display_name(k);
            eprintln!(
                "  KV cache: requested mode={mode_raw}, effective KV={name} (site {})",
                policy.site
            );
            hipfire_config::load_summary::note(hipfire_config::load_summary::Facet::Kv, name);
        }
        KvPair::Split(k, v) => {
            let k_name = kv_mode::qwen_k_display_name(k);
            let v_name = kv_mode::qwen_v_display_name(v);
            eprintln!(
                "  KV cache: requested mode={mode_raw}, effective K={k_name} V={v_name} (site {})",
                policy.site
            );
            eprintln!("  K={k_name} V={v_name}");
            hipfire_config::load_summary::note(
                hipfire_config::load_summary::Facet::Kv,
                format!("K={k_name} V={v_name}"),
            );
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
    fn mtp_measured_ar_floor(&self) -> bool {
        hipfire_arch_qwen35::MTP_MEASURED_AR_FLOOR
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
                    use hipfire_arch_qwen35_vl::qwen35_vl::{
                        self, load_vision_weights, TowerResidency,
                    };
                    use hipfire_arch_qwen35_vl::Qwen35Vl;
                    use hipfire_runtime::arch::Architecture;

                    let residency = TowerResidency::from_vision_mode(&ctx.vision_mode);
                    // After a tower loads (either residency) the automatic
                    // max_seq must leave room for its activation scratch.
                    fn reserve_tower_activation(
                        ctx: &mut LoadCtx<'_>,
                        vc: &qwen35_vl::VisionConfig,
                        residency: TowerResidency,
                    ) {
                        let patches = qwen35_vl::max_image_patches(vc);
                        let bytes = qwen35_vl::tower_activation_peak_bytes(vc, patches);
                        if let Some(seq) = ctx.sequence.as_mut() {
                            seq.reserve_bytes = seq.reserve_bytes.saturating_add(bytes);
                        }
                        eprintln!(
                            "  vision tower: {}; activation reserve {} MiB for {patches} patches",
                            match residency {
                                TowerResidency::Resident => "resident",
                                TowerResidency::HostMapped => "host-mapped",
                                TowerResidency::Streamed => "streamed",
                            },
                            bytes >> 20
                        );
                    }

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
                        let vl_hfq = hipfire_runtime::hfq::HfqFile::open(vl)
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
                        let vw = load_vision_weights(&vl_hfq, &vc, ctx.gpu, residency)
                            .map_err(|e| format!("VL weight load from .vl: {e:?}"))?;
                        reserve_tower_activation(ctx, &vc, residency);
                        eprintln!(
                            "  VL model: vision encoder (hidden={}, layers={})",
                            vc.hidden_size, vc.num_layers
                        );
                        (Some(vc), Some(vw))
                    } else if has_inline_vision {
                        let vc = Qwen35Vl::config_from_hfq(&hfq_file).ok();
                        match vc {
                            Some(vc) => {
                                let vw = load_vision_weights(&hfq_file, &vc, ctx.gpu, residency)
                                    .map_err(|e| eprintln!("  VL weight load failed: {e}"))
                                    .ok();
                                if vw.is_some() {
                                    reserve_tower_activation(ctx, &vc, residency);
                                }
                                eprintln!(
                                    "  VL model: vision encoder (hidden={}, layers={})",
                                    vc.hidden_size, vc.num_layers
                                );
                                (Some(vc), vw)
                            }
                            _ => (None, None),
                        }
                    } else if let Some(p) = ctx.vision_path.clone().as_ref() {
                        let sidecar = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(p))
                            .map_err(|e| {
                                format!("vision sidecar '{}': open failed: {e}", p.display())
                            })?;
                        let vc = Qwen35Vl::config_from_hfq(&hfq_file)
                            .ok()
                            .or_else(|| Qwen35Vl::config_from_hfq(&sidecar).ok())
                            .ok_or_else(|| {
                                "qwen35-vl: vision tower requested but no vision_config in \
                                 trunk or sidecar metadata — requantize the trunk or pack \
                                 the sidecar with --include-vision"
                                    .to_string()
                            })?;
                        let vw = load_vision_weights(&sidecar, &vc, ctx.gpu, residency).map_err(
                            |e| format!("vision sidecar '{}': tower load failed: {e}", p.display()),
                        )?;
                        reserve_tower_activation(ctx, &vc, residency);
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
                let auto_native = hipfire_runtime::kv_mode::qwen35_native_auto_eligible(
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
                    auto_native,
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
            swap_disk_bytes: cfg.swap_disk_bytes,
            is_vl: cfg.is_vl,
            vl_path: cfg.vl_path,
            vision_residency: hipfire_arch_qwen35_vl::qwen35_vl::TowerResidency::from_vision_mode(
                &cfg.vision_mode,
            ),
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
    fn mtp_measured_ar_floor(&self) -> bool {
        hipfire_arch_deepseek4::MTP_MEASURED_AR_FLOOR
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
    fn qwen4_dflash2_width_is_eight_under_auto_and_explicit_otherwise_within_exact_rows() {
        let width = |dflash_block, dflash_block_auto, rows| {
            super::qwen4_dflash2_block(
                SpecLoadCfg {
                    dflash_block,
                    dflash_block_auto,
                    ..SpecLoadCfg::default()
                },
                rows,
            )
        };
        // The daemon resolves `auto` to 16 (8 batched); Qwen4 keeps its 8.
        assert_eq!(width(Some(16), true, 8), 8);
        assert_eq!(width(Some(4), true, 8), 8);
        assert_eq!(width(None, false, 8), 8);
        assert_eq!(width(Some(6), false, 8), 6);
        // Never past the row-exact verify rows.
        assert_eq!(width(Some(16), false, 8), 8);
        assert_eq!(width(None, false, 3), 3);
    }

    #[test]
    fn dflash_needs_an_explicit_request_its_draft_and_no_forced_mtp() {
        let dflash = SourceAdmissionOptions {
            spec: SpecLoadCfg {
                dflash: Some(true),
                ..resolved_ar_spec()
            },
            ..Default::default()
        };
        assert!(Qwen4Carrier
            .admit_options(Some("draft.hfq"), dflash)
            .is_ok());
        let error = Qwen4Carrier.admit_options(None, dflash).unwrap_err();
        assert!(error.contains("needs a DFlash2 draft"), "{error}");
        let both = SourceAdmissionOptions {
            spec: SpecLoadCfg {
                mtp: Some(true),
                ..dflash.spec
            },
            ..dflash
        };
        let error = Qwen4Carrier
            .admit_options(Some("draft.hfq"), both)
            .unwrap_err();
        assert!(error.contains("pick one"), "{error}");
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

    /// Uniform metadata slabs (padded payload 1,536 B per expert) and the id
    /// rank, as the placement fixture above.
    fn decode_share_fixture() -> (
        Vec<hipfire_arch_qwen4::expert_slab_table::ExpertSlab>,
        Vec<u32>,
    ) {
        use hipfire_arch_qwen4::{
            expert_slab_table::{ExpertSlab, ExpertSlabSpan},
            expert_slabs::ExpertFormat,
        };
        let slabs = (0..48 * 512)
            .map(|id| ExpertSlab {
                format: ExpertFormat::Mq4Xts,
                gate_up: ExpertSlabSpan {
                    offset: id * 1500,
                    len: 1000,
                },
                down: ExpertSlabSpan {
                    offset: id * 1500 + 1000,
                    len: 500,
                },
            })
            .collect();
        // Round-robin over layers (expert 0 of every layer first): every layer
        // keeps host experts, so every resident expert may donate.
        let rank = (0..512u32)
            .flat_map(|expert| (0..48u32).map(move |layer| layer * 512 + expert))
            .collect();
        (slabs, rank)
    }

    fn device_ids(plan: &hipfire_arch_qwen4::expert_tier_plan::ExpertTierPlan) -> Vec<usize> {
        use hipfire_arch_qwen4::expert_tier_plan::Placement;
        (0..48 * 512)
            .filter(|id| plan.entry(id / 512, id % 512).unwrap().tier == Placement::Device)
            .collect()
    }

    #[test]
    fn decode_share_hotset_off_is_byte_identical_to_the_plain_fixed_point() {
        use hipfire_arch_qwen4::{
            expert_hotset::HotsetBudget,
            expert_residency as residency,
            expert_tier_plan::{plan_expert_tiers_from_slabs, Placement},
        };
        let (slabs, rank) = decode_share_fixture();
        for (stage_enabled, payload_room) in
            [(false, 16u64 << 20), (true, 16 << 20), (true, 64 << 20)]
        {
            let free = residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES + payload_room;
            let total = residency::expert_hotset_budget(free, 0, 0).unwrap();
            let run = |need| {
                super::qwen4_plan_hotset_placement(
                    &slabs,
                    &rank,
                    HotsetBudget::parse(None).unwrap(),
                    total,
                    stage_enabled,
                    (48, 512, false),
                    need,
                    &[],
                )
                .unwrap()
            };
            let (plan, budget, stage) = run(None);
            assert!(plan.decode_share_tail().is_none());
            // The plain solve: today's plan for the same budget, stage from the
            // Resident placement, no second pointer tables.
            let direct = plan_expert_tiers_from_slabs(&slabs, &rank, budget).unwrap();
            assert_eq!(plan.logical_digest(), direct.logical_digest());
            let staged = stage_enabled && plan.owner_bytes(Placement::HostMapped) > 0;
            assert_eq!(
                stage,
                if staged {
                    residency::expert_tier_stage_bytes(&plan)
                } else {
                    0
                }
            );
            let tables = super::qwen4_expert_table_bytes(staged, (48, 512, false)).unwrap();
            assert_eq!(
                budget + stage + tables + residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES,
                free
            );
            // A zero-byte request is "off" too.
            let (zero, zero_budget, zero_stage) = run(Some(0));
            assert_eq!(zero.logical_digest(), plan.logical_digest());
            assert_eq!((zero_budget, zero_stage), (budget, stage));
        }
    }

    #[test]
    fn decode_share_hotset_bind_keeps_the_ranked_prefix_and_charges_its_overhead() {
        use hipfire_arch_qwen4::{
            expert_arena::ARENA_TAIL_PAD_BYTES,
            expert_hotset::HotsetBudget,
            expert_residency as residency,
            expert_tier_plan::{plan_expert_tiers_from_slabs, Placement},
        };
        const PADDED: u64 = 1024 + 512;
        let (slabs, rank) = decode_share_fixture();
        let need = 1u64 << 20;
        for (stage_enabled, payload_room) in
            [(false, 16u64 << 20), (true, 16 << 20), (true, 24 << 20)]
        {
            let free = residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES + payload_room;
            let total = residency::expert_hotset_budget(free, 0, 0).unwrap();
            let run = |need| {
                super::qwen4_plan_hotset_placement(
                    &slabs,
                    &rank,
                    HotsetBudget::parse(None).unwrap(),
                    total,
                    stage_enabled,
                    (48, 512, false),
                    need,
                    &[],
                )
                .unwrap()
            };
            let (plain, plain_budget, plain_stage) = run(None);
            let (shared, budget, stage) = run(Some(need));
            let tail = shared
                .decode_share_tail()
                .expect("bind lays out a donor tail");
            assert!(tail.donor_bytes >= need);
            // The ranked set: the first `n` ranked ids, a prefix of the plain
            // plan's, losing only what the extra bytes displace.
            let ranked = |plan: &hipfire_arch_qwen4::expert_tier_plan::ExpertTierPlan| {
                let mut want: Vec<usize> = rank[..plan.device_experts()]
                    .iter()
                    .map(|id| *id as usize)
                    .collect();
                want.sort_unstable();
                want
            };
            let plain_ids = device_ids(&plain);
            let shared_ids = device_ids(&shared);
            assert_eq!(plain_ids, ranked(&plain));
            assert_eq!(shared_ids, ranked(&shared));
            assert!(shared_ids
                .iter()
                .all(|id| plain_ids.binary_search(id).is_ok()));
            // The extra device bytes are the guard (and nothing else here:
            // every offset is already 256-aligned).
            let same_budget = plan_expert_tiers_from_slabs(&slabs, &rank, budget).unwrap();
            let overhead =
                shared.owner_bytes(Placement::Device) - same_budget.owner_bytes(Placement::Device);
            let tail_pad = ((ARENA_TAIL_PAD_BYTES as u64).max(256) + 255) & !255;
            assert_eq!(
                overhead, tail_pad,
                "stage {stage_enabled} room {payload_room}"
            );
            assert_eq!(tail.donor_offset - tail.guard_offset, tail_pad);
            // Every byte is charged exactly once: budget + overhead + stage +
            // tables (base, stage, second device pointer tables) + margin.
            let staged = stage_enabled && shared.owner_bytes(Placement::HostMapped) > 0;
            let share_tables = (48 * 512 * 2 * 8) as u64;
            assert_eq!(
                super::qwen4_decode_share_table_bytes(&shared, (48, 512, false)).unwrap(),
                share_tables
            );
            let tables =
                super::qwen4_expert_table_bytes(staged, (48, 512, false)).unwrap() + share_tables;
            assert_eq!(
                budget + overhead + stage + tables + residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES,
                free
            );
            assert!(
                shared.owner_bytes(Placement::Device)
                    + stage
                    + tables
                    + residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES
                    <= free
            );
            // Stage bytes come from the BorrowPrefill placement.
            assert_eq!(
                stage,
                if staged {
                    residency::expert_tier_stage_bytes_prefill(&shared)
                } else {
                    0
                }
            );
            assert!(stage >= plain_stage);
            // Displaced experts: at most what the budget difference holds.
            let lost = plain.device_experts() - shared.device_experts();
            assert!(budget <= plain_budget);
            assert!(
                lost as u64 <= (plain_budget - budget).div_ceil(PADDED),
                "lost {lost} experts for {} B",
                plain_budget - budget
            );
            assert!(
                shared.owner_bytes(Placement::HostMapped)
                    > plain.owner_bytes(Placement::HostMapped)
            );
        }
    }

    #[test]
    fn decode_share_hotset_insufficient_donors_keep_the_plain_plan() {
        use hipfire_arch_qwen4::{expert_hotset::HotsetBudget, expert_residency as residency};
        let (slabs, rank) = decode_share_fixture();
        let free = residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES + (16u64 << 20);
        let total = residency::expert_hotset_budget(free, 0, 0).unwrap();
        let run = |need| {
            super::qwen4_plan_hotset_placement(
                &slabs,
                &rank,
                HotsetBudget::parse(None).unwrap(),
                total,
                true,
                (48, 512, false),
                need,
                &[],
            )
            .unwrap()
        };
        let (plain, plain_budget, plain_stage) = run(None);
        // More than every resident expert holds: ineligible, plain plan.
        let (kept, budget, stage) = run(Some(1 << 40));
        assert!(kept.decode_share_tail().is_none());
        assert_eq!(kept.logical_digest(), plain.logical_digest());
        assert_eq!((budget, stage), (plain_budget, plain_stage));
    }

    /// One BorrowPrefill table set per prefill class: the fixed point charges
    /// every set, a class-less plan still owns exactly one.
    #[test]
    fn decode_share_table_charge_counts_every_class_set() {
        use hipfire_arch_qwen4::{
            expert_hotset::HotsetBudget,
            expert_residency as residency,
            expert_tier_plan::{plan_expert_tiers_from_slabs, Placement},
        };
        const PADDED: u64 = 1024 + 512;
        let (slabs, rank) = decode_share_fixture();
        let need = 1u64 << 20;
        let spans = [need / 4, need / 2, need];
        let free = residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES + (24u64 << 20);
        let total = residency::expert_hotset_budget(free, 0, 0).unwrap();
        let run = |spans: &[u64]| {
            super::qwen4_plan_hotset_placement(
                &slabs,
                &rank,
                HotsetBudget::parse(None).unwrap(),
                total,
                true,
                (48, 512, false),
                Some(need),
                spans,
            )
            .unwrap()
        };
        let one_set = (48 * 512 * 2 * 8) as u64;
        let (single, single_budget, _) = run(&[]);
        let (three, budget, stage) = run(&spans);
        let tail = three.decode_share_tail().unwrap();
        assert_eq!(tail.class_spans, spans);
        // Donors start every PADDED bytes: a class dirties ceil(span / PADDED).
        assert_eq!(
            tail.class_donors,
            spans
                .iter()
                .map(|span| span.div_ceil(PADDED) as usize)
                .collect::<Vec<_>>()
        );
        assert_eq!(*tail.class_donors.last().unwrap(), tail.donors.len());
        assert_eq!(single.decode_share_tail().unwrap().class_donors.len(), 1);
        let geometry = (48, 512, false);
        assert_eq!(
            super::qwen4_decode_share_table_bytes(&single, geometry).unwrap(),
            one_set
        );
        assert_eq!(
            super::qwen4_decode_share_table_bytes(&three, geometry).unwrap(),
            3 * one_set
        );
        // The fixed point pays for the two extra sets out of the hot budget.
        assert!(
            budget + 2 * one_set <= single_budget + PADDED,
            "{budget} vs {single_budget}"
        );
        assert!(budget < single_budget);
        let staged = three.owner_bytes(Placement::HostMapped) > 0;
        let same_budget = plan_expert_tiers_from_slabs(&slabs, &rank, budget).unwrap();
        let overhead =
            three.owner_bytes(Placement::Device) - same_budget.owner_bytes(Placement::Device);
        let tables = super::qwen4_expert_table_bytes(staged, geometry).unwrap() + 3 * one_set;
        assert_eq!(
            budget + overhead + stage + tables + residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES,
            free
        );
    }

    /// The borrowed weight-shadow region is part of the class spans (every class
    /// dirties it), recorded on the plan's tail, and absent (plain placement,
    /// shadows resident) when sharing was refused or the region is zero.
    #[test]
    fn decode_share_shadow_region_is_recorded_in_the_tail_and_in_every_class_span() {
        use hipfire_arch_qwen4::{expert_hotset::HotsetBudget, expert_residency as residency};
        const PADDED: u64 = 1024 + 512;
        let (slabs, rank) = decode_share_fixture();
        let shadow = 3 * 256u64;
        let scratch = [PADDED, 2 * PADDED, 4 * PADDED];
        let spans: Vec<u64> = scratch.iter().map(|span| span + shadow).collect();
        let need = *spans.last().unwrap();
        let free = residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES + (24u64 << 20);
        let total = residency::expert_hotset_budget(free, 0, 0).unwrap();
        let (plan, _, _) = super::qwen4_plan_hotset_placement(
            &slabs,
            &rank,
            HotsetBudget::parse(None).unwrap(),
            total,
            true,
            (48, 512, false),
            Some(need),
            &spans,
        )
        .unwrap();
        let shared = super::qwen4_record_shadow_region(plan.clone(), true, shadow).unwrap();
        let tail = shared.decode_share_tail().unwrap();
        assert_eq!(tail.shadow_bytes, shadow);
        assert_eq!(tail.class_spans, spans);
        // Every class dirties at least the donors the shadow region spans.
        let region_donors = shadow.div_ceil(PADDED) as usize;
        assert!(tail
            .class_donors
            .iter()
            .all(|&donors| donors >= region_donors));
        assert_eq!(*tail.class_donors.last().unwrap(), tail.donors.len());
        // The donor tail covers the shadows and the largest scratch class.
        assert!(tail.donor_bytes >= need);
        // Refused sharing (`need` absent) records nothing and changes nothing.
        let (plain, _, _) = super::qwen4_plan_hotset_placement(
            &slabs,
            &rank,
            HotsetBudget::parse(None).unwrap(),
            total,
            true,
            (48, 512, false),
            None,
            &[],
        )
        .unwrap();
        let same = super::qwen4_record_shadow_region(plain.clone(), false, shadow).unwrap();
        assert!(same.decode_share_tail().is_none());
        assert_eq!(same.logical_digest(), plain.logical_digest());
        // A zero region (shadows stay resident) leaves a sharing plan's tail at 0.
        let none = super::qwen4_record_shadow_region(plan, true, 0).unwrap();
        assert_eq!(none.decode_share_tail().unwrap().shadow_bytes, 0);
        // A region the donor tail cannot hold is a load error, never a silent shrink.
        let too_big = *tail.class_spans.first().unwrap() + 256;
        assert!(super::qwen4_record_shadow_region(shared, true, too_big).is_err());
    }

    /// `on` against the real Flash-Next expert size (2,611,200 B padded) and the
    /// R9700 budget: the grown budget (the reserve the dedicated scratch no
    /// longer needs) places at least 0.8 K more experts than `off`, the donor
    /// tail stays within K plus rounding, and a refused tail recomputes the
    /// plain placement, which is exactly the `off` plan.
    #[test]
    fn decode_share_on_grows_the_resident_set_and_a_refusal_recomputes_the_off_plan() {
        use hipfire_arch_qwen4::{
            expert_hotset::HotsetBudget,
            expert_residency as residency,
            expert_slab_table::{ExpertSlab, ExpertSlabSpan},
            expert_slabs::ExpertFormat,
        };
        const GU: usize = 1_740_800;
        const DN: usize = 870_400;
        const PADDED: u64 = (GU + DN) as u64;
        let slabs: Vec<ExpertSlab> = (0..48 * 512)
            .map(|id: usize| ExpertSlab {
                format: ExpertFormat::Mq4Xts,
                gate_up: ExpertSlabSpan {
                    offset: id * (GU + DN),
                    len: GU,
                },
                down: ExpertSlabSpan {
                    offset: id * (GU + DN) + GU,
                    len: DN,
                },
            })
            .collect();
        let rank: Vec<u32> = (0..512u32)
            .flat_map(|expert| (0..48u32).map(move |layer| layer * 512 + expert))
            .collect();
        let free = residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES + (14_525u64 << 20);
        let total = residency::expert_hotset_budget(free, 0, 0).unwrap();
        // F - D and S of the R9700 receipt, rounded down / up.
        let (reclaimed, span) = (4_400_000_000u64, 4_700_000_000u64);
        let run = |total, need| {
            super::qwen4_plan_hotset_placement(
                &slabs,
                &rank,
                HotsetBudget::parse(None).unwrap(),
                total,
                true,
                (48, 512, false),
                need,
                &[],
            )
            .unwrap()
        };
        let (off, off_budget, off_stage) = run(total, None);
        assert!(off.decode_share_tail().is_none());
        let (on, _, _) = run(total + reclaimed, Some(span));
        let tail = on.decode_share_tail().expect("on lays out a donor tail");
        let donors = tail.donors.len() as u64;
        assert!(tail.donor_bytes >= span);
        assert!(
            donors >= span.div_ceil(PADDED) && donors <= span.div_ceil(PADDED) + 2,
            "{donors} donors for {span} B"
        );
        let gain = (on.device_experts() - off.device_experts()) as u64;
        assert!(
            gain * 10 >= donors * 8,
            "grew {gain} experts for {donors} donors (< 0.8 K)"
        );
        // The grown placement is never larger than the grown budget.
        assert!(on.device_experts() > off.device_experts());
        // Insufficient donors at the grown budget: no tail, so the loader
        // recomputes with the plain reserve; that plan is the `off` plan exactly.
        let (refused, _, _) = run(total + reclaimed, Some(1 << 50));
        assert!(refused.decode_share_tail().is_none());
        let (again, again_budget, again_stage) = run(total, None);
        assert_eq!(again.logical_digest(), off.logical_digest());
        assert_eq!((again_budget, again_stage), (off_budget, off_stage));
    }

    #[test]
    fn expert_placement_budgets_disjoint_tables_and_stage() {
        use hipfire_arch_qwen4::{
            expert_hotset::HotsetBudget,
            expert_residency as residency,
            expert_slab_table::{ExpertSlab, ExpertSlabSpan},
            expert_slabs::ExpertFormat,
            expert_tier_plan::{plan_expert_tiers_from_slabs, Placement},
        };
        // Pure metadata, with the nonaligned spans used by the tier planner's
        // CPU fixtures. The actual planner owns payload packing and tail pads.
        let slabs: Vec<_> = (0..48 * 512)
            .map(|id| ExpertSlab {
                format: ExpertFormat::Mq4Xts,
                gate_up: ExpertSlabSpan {
                    offset: id * 1500,
                    len: 1000,
                },
                down: ExpertSlabSpan {
                    offset: id * 1500 + 1000,
                    len: 500,
                },
            })
            .collect();
        let rank: Vec<u32> = (0..48 * 512).collect();
        for (stage_enabled, keep, payload_room) in [
            (false, false, 16u64 << 20),
            (true, false, 16 << 20),
            (true, true, 16 << 20),
            (true, false, 64 << 20),
        ] {
            let free = residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES + payload_room;
            let total = residency::expert_hotset_budget(free, 0, 0).unwrap();
            let (plan, budget, stage) = super::qwen4_plan_hotset_placement(
                &slabs,
                &rank,
                HotsetBudget::parse(None).unwrap(),
                total,
                stage_enabled,
                (48, 512, keep),
                None,
                &[],
            )
            .unwrap();
            let host = plan.owner_bytes(Placement::HostMapped) > 0;
            let staged = stage_enabled && host;
            // Base owns the pointer tables plus the codec (17 per expert);
            // the stage adds only its own pointer tables (16 per expert);
            // optional masks ride once alongside.
            let tables = (48 * 512 * (2 * 8 + 1)) as u64
                + if staged {
                    (48 * 512 * (2 * 8)) as u64
                } else {
                    0
                }
                + if keep { 49 * 512 } else { 0 };
            assert!(
                plan.owner_bytes(Placement::Device)
                    + stage
                    + tables
                    + residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES
                    <= free
            );
            // Equality rejects duplicate table/payload charges as well as
            // omissions: the selected budget contains every remaining byte.
            assert_eq!(
                budget + stage + tables + residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES,
                free
            );
            assert_eq!(
                tables,
                super::qwen4_expert_table_bytes(staged, (48, 512, keep)).unwrap()
            );
            assert_eq!(stage > 0, staged);
            assert_eq!(host, payload_room == 16 << 20);
            if staged && !keep {
                // Counterfactual shipped solve misses the distinct stage
                // tables, and the actual packed owner crosses capacity.
                let old = plan_expert_tiers_from_slabs(&slabs, &rank, total - stage).unwrap();
                assert!(
                    old.owner_bytes(Placement::Device)
                        + stage
                        + tables
                        + residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES
                        > free
                );
            }
        }
        // Whole-layer AUTO cedes a layer only at the genuine byte boundary.
        let layer = 2u64 << 30;
        let base = 1u64 << 30;
        let charges = residency::PlacementCharges::new(layer, true).unwrap();
        let tables = super::qwen4_expert_table_bytes(true, (48, 512, true)).unwrap();
        let free = charges.reserve(base).unwrap() + tables + 3 * layer;
        for (capacity, expected) in [(free, 3), (free - 1, 2)] {
            let available = capacity.checked_sub(tables).unwrap();
            assert_eq!(
                residency::auto_placement_layers(available, 0, layer, 48, base, &charges,).unwrap(),
                expected
            );
        }
        // Preserve the existing reserve, including its historical stage term.
        // At the real full-residency boundary the plan creates no stage, so
        // charging the EXTRA pointer tables must not spill a whole layer.
        let base_tables = super::qwen4_expert_table_bytes(false, (48, 512, true)).unwrap();
        let stage_extra =
            super::qwen4_expert_table_bytes(true, (48, 512, true)).unwrap() - base_tables;
        let full_capacity = charges.reserve(base).unwrap() + base_tables + 48 * layer;
        let solve_whole = |capacity: u64| {
            residency::WholeFirstPlan::auto(
                capacity.checked_sub(base_tables).unwrap(),
                0,
                layer,
                48,
                48,
                base,
                &charges,
                false,
            )
            .unwrap()
        };
        let whole = solve_whole(full_capacity);
        assert_eq!(whole.whole_layers, 48);
        assert!(matches!(whole.stage, residency::WholeFirstStage::None));
        assert!(solve_whole(full_capacity - stage_extra).whole_layers < 48);
    }
}

#[cfg(test)]
mod qwen4_ple_load_fill_tests {
    use super::{
        ple_load_fill_bytes, ple_prewarm_budget_for, ple_resident_bytes, PLE_LOAD_FILL_AUTO_BYTES,
        PLE_PREWARM_AUTO_CAP_BYTES, PLE_PREWARM_RESERVE_BYTES,
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
        // 14,846 MiB. The 14 GiB in-load fill is inside it, so the resident
        // charge is the budget.
        let (budget, zero) = ple_prewarm_budget_for(Auto, Some(23_038 * MIB), true, false);
        assert_eq!((budget, zero), (14_846 * MIB, false));
        assert_eq!(
            ple_resident_bytes(budget, PLE_LOAD_FILL_AUTO_BYTES, 27 * GIB),
            14_846 * MIB
        );
        // The auto cap, an explicit value, zero and unknown MemAvailable.
        assert_eq!(
            ple_prewarm_budget_for(Auto, Some(112 * GIB), true, false),
            (15 * GIB, false)
        );
        assert_eq!(
            ple_prewarm_budget_for(Bytes(2 * GIB), None, true, false),
            (2 * GIB, false)
        );
        assert_eq!(
            ple_prewarm_budget_for(Bytes(0), Some(112 * GIB), true, false),
            (0, true)
        );
        assert_eq!(ple_prewarm_budget_for(Auto, None, true, false), (0, false));
    }

    #[test]
    fn auto_budget_follows_ram_past_15_gib_when_offloaded_and_the_table_caps_residency() {
        const MIB: u64 = 1 << 20;
        // The 26.82 GiB q4 PLE table (data_bytes of the store).
        let table = 27_463 * MIB;
        // Discrete GPU with host-placed experts, 112 GiB available: budget is
        // available - reserve, above the 15 GiB cap, and the resident charge
        // caps at the whole table.
        let (budget, zero) = ple_prewarm_budget_for(Auto, Some(112 * GIB), false, true);
        assert_eq!(
            (budget, zero),
            (112 * GIB - PLE_PREWARM_RESERVE_BYTES, false)
        );
        assert!(budget > 15 * GIB);
        assert_eq!(ple_resident_bytes(budget, 0, table), table);
        assert_eq!(
            ple_resident_bytes(budget, PLE_LOAD_FILL_AUTO_BYTES, table),
            table
        );
        // Just enough RAM for the whole table plus the reserve.
        let (budget, _) =
            ple_prewarm_budget_for(Auto, Some(table + PLE_PREWARM_RESERVE_BYTES), false, true);
        assert_eq!(budget, table);
        assert_eq!(ple_resident_bytes(budget, 0, table), table);
        // One MiB short of that: a partial prewarm that keeps the reserve.
        let (budget, _) = ple_prewarm_budget_for(
            Auto,
            Some(table + PLE_PREWARM_RESERVE_BYTES - MIB),
            false,
            true,
        );
        assert_eq!(budget, table - MIB);
        assert_eq!(ple_resident_bytes(budget, 0, table), table - MIB);
        // 35 GiB available leaves 27 GiB over the reserve.
        let (budget, _) = ple_prewarm_budget_for(Auto, Some(35 * GIB), false, true);
        assert_eq!(budget, 27 * GIB);
    }

    #[test]
    fn auto_budget_keeps_the_15_gib_cap_unless_a_discrete_gpu_has_host_experts() {
        // Unified memory (Halo), with or without a host-expert signal, and a
        // fully-resident discrete card (gfx1201, low and 23 GiB available,
        // large host RAM) share one beta budget: available minus reserve,
        // capped at 15 GiB.
        for (is_uma, has_host_experts) in [(true, false), (true, true), (false, false)] {
            for available in [12 * GIB, 23 * GIB, 112 * GIB] {
                let expected =
                    (available - PLE_PREWARM_RESERVE_BYTES).min(PLE_PREWARM_AUTO_CAP_BYTES);
                assert_eq!(
                    ple_prewarm_budget_for(Auto, Some(available), is_uma, has_host_experts),
                    (expected, false)
                );
            }
            assert_eq!(
                ple_prewarm_budget_for(Auto, Some(112 * GIB), is_uma, has_host_experts),
                (15 * GIB, false)
            );
        }
        // A discrete GPU with host-placed experts (gfx1100 offload) is
        // uncapped: available minus reserve.
        assert_eq!(
            ple_prewarm_budget_for(Auto, Some(112 * GIB), false, true),
            (112 * GIB - PLE_PREWARM_RESERVE_BYTES, false)
        );
        assert_eq!(
            ple_prewarm_budget_for(Auto, Some(23 * GIB), false, true),
            (15 * GIB, false)
        );
    }

    #[test]
    fn auto_budget_is_zero_without_headroom_over_the_reserve() {
        // At, under and far under the reserve, and with no MemAvailable reading.
        assert_eq!(
            ple_prewarm_budget_for(Auto, Some(PLE_PREWARM_RESERVE_BYTES), false, true),
            (0, false)
        );
        assert_eq!(
            ple_prewarm_budget_for(Auto, Some(PLE_PREWARM_RESERVE_BYTES - 1), false, true),
            (0, false)
        );
        assert_eq!(
            ple_prewarm_budget_for(Auto, Some(0), false, true),
            (0, false)
        );
        assert_eq!(ple_prewarm_budget_for(Auto, None, false, true), (0, false));
        assert_eq!(ple_resident_bytes(0, 0, 27 * GIB), 0);
        // One byte over the reserve is a one-byte budget.
        assert_eq!(
            ple_prewarm_budget_for(Auto, Some(PLE_PREWARM_RESERVE_BYTES + 1), false, true),
            (1, false)
        );
        assert_eq!(
            ple_prewarm_budget_for(Auto, Some(PLE_PREWARM_RESERVE_BYTES + 1), true, false),
            (1, false)
        );
    }

    #[test]
    fn explicit_budget_ignores_available_ram_and_the_reserve() {
        // An explicit value is used as given, whatever MemAvailable says and
        // whatever the placement (the cap is Auto-only).
        for (is_uma, has_host_experts) in [(true, false), (false, false), (false, true)] {
            assert_eq!(
                ple_prewarm_budget_for(Bytes(20 * GIB), Some(GIB), is_uma, has_host_experts),
                (20 * GIB, false)
            );
            assert_eq!(
                ple_prewarm_budget_for(Bytes(20 * GIB), Some(112 * GIB), is_uma, has_host_experts),
                (20 * GIB, false)
            );
            // An explicit zero turns the prewarm off and is reported as such.
            assert_eq!(
                ple_prewarm_budget_for(Bytes(0), None, is_uma, has_host_experts),
                (0, true)
            );
            assert_eq!(
                ple_prewarm_budget_for(Bytes(0), Some(0), is_uma, has_host_experts),
                (0, true)
            );
        }
        // An explicit over-the-table value still charges only the table.
        assert_eq!(ple_resident_bytes(40 * GIB, 0, 27 * GIB), 27 * GIB);
    }

    #[test]
    fn saturated_edges_do_not_overflow() {
        assert_eq!(
            ple_prewarm_budget_for(Auto, Some(u64::MAX), false, true),
            (u64::MAX - PLE_PREWARM_RESERVE_BYTES, false)
        );
        assert_eq!(
            ple_prewarm_budget_for(Bytes(u64::MAX), None, false, true),
            (u64::MAX, false)
        );
        let (budget, _) = ple_prewarm_budget_for(Auto, Some(u64::MAX), false, true);
        assert_eq!(ple_resident_bytes(budget, 0, 27 * GIB), 27 * GIB);
        assert_eq!(ple_resident_bytes(u64::MAX, u64::MAX, 27 * GIB), 27 * GIB);
        assert_eq!(ple_resident_bytes(u64::MAX, 0, u64::MAX), u64::MAX);
        // Not offloaded: the 15 GiB cap holds at the same saturated edge.
        assert_eq!(
            ple_prewarm_budget_for(Auto, Some(u64::MAX), false, false),
            (PLE_PREWARM_AUTO_CAP_BYTES, false)
        );
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
mod qwen4_measured_identity_bounds_tests {
    use super::{qwen4_native_mtp_k, qwen4_recipe_string};
    use hipfire_arch_qwen4::weights::Qwen4TierRecipe;
    use rdna_compute::DType;

    #[test]
    fn derived_proposal_cap_is_the_executor_width_not_a_secret_ten_cap() {
        // The actual proposal choice 1..63 rides on
        // `spec_economics::DERIVED_MAX_DRAFTS` through the real
        // `native_mtp_proposal_drafts` helper; the incumbent `auto` path
        // must agree with that width on a no-takeover arch.
        assert_eq!(hipfire_runtime::spec_economics::DERIVED_MAX_DRAFTS, 63);
        for max_k in [1usize, 3, 7, 10, 31, 63] {
            let incumbent =
                hipfire_arch_qwen4::mtp_spec::native_mtp_proposal_drafts(max_k, "gfx1151");
            assert!(
                (1..=63).contains(&incumbent),
                "incumbent drafts {incumbent} must stay in the executor width 1..63"
            );
        }
    }

    #[test]
    fn chooser_policy_key_never_enters_the_numerical_identity() {
        // The factory skips `CHOOSER_POLICY_KEY` explicitly before insertion
        // (the module validator rejects it rather than stripping it): a kept
        // key would force every derived load to `None`.
        assert_eq!(
            hipfire_arch_qwen4::measured_identity::CHOOSER_POLICY_KEY,
            "config.developer.qwen4_mtp_derived_k"
        );
    }

    #[test]
    fn explicit_k_keeps_sixty_three_without_a_ten_cap() {
        // An explicit 63 stays 63 up to the physical executor width; the
        // pipeline-parallel path keeps its own qualified arithmetic elsewhere.
        let spec = hipfire_runtime::loader_api::SpecLoadCfg {
            mtp_k: Some(63),
            ..Default::default()
        };
        assert_eq!(qwen4_native_mtp_k(spec, "gfx1201"), 63);
        assert_eq!(qwen4_native_mtp_k(spec, "gfx1151"), 63);
        // The `qwen4_recipe_string` helper stays total over every recipe
        // variant (no silent empty identity for a new recipe).
        for recipe in [
            Qwen4TierRecipe::FlashNextPacked {
                trunk_tier: DType::Q8_0,
                mtp_source: false,
            },
            Qwen4TierRecipe::StockFp8,
            Qwen4TierRecipe::SyntheticSourceBf16,
            Qwen4TierRecipe::SyntheticPacked,
        ] {
            assert!(!qwen4_recipe_string(&recipe).is_empty());
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

#[cfg(test)]
mod qwen4_slowlink_tests {
    use hipfire_arch_qwen4::host_link::HostLink;
    use hipfire_arch_qwen4::slowlink_admission as s1;
    use std::cell::Cell;

    const MIB: u64 = 1 << 20;
    const LAYER: u64 = 512 * (1_740_800 + 870_400);
    const NON_EXPERT: u64 = 5_364_000_000;
    const BASE_RESERVE: u64 = 5261 * MIB;
    const FIXED_STATE: u64 = 64 * MIB;
    // One F32 wide row (`hc_count * hidden * 4`): the native-MTP entry pending copy.
    const PENDING_SAVE: u64 = 4 * 2560 * 4;
    const MARGIN: u64 = 512 * MIB;

    fn shape(arch: &str) -> s1::LoadShape<'_> {
        s1::LoadShape {
            arch,
            unified_memory: false,
            expert_parallel: false,
            pipeline_band: false,
            routed_experts: true,
            expert_keep: false,
            explicit_vram_layers: false,
            hotset_sidecar: false,
            whole_first: false,
            stage_opt_out: false,
            npu_trunk: false,
            chunk_rows_env: None,
            max_seq: 16640,
        }
    }

    /// The real MQ4 fixture (H0 receipt `source.layer_identity`): gate/up
    /// `[512, 1280, 2560]`, down `[512, 2560, 640]`, so hidden 2560 and expert
    /// intermediate 640 (1,740,800 and 870,400 payload bytes per expert).
    fn geometry() -> s1::ModelGeometry {
        s1::ModelGeometry {
            hidden: 2560,
            hc_count: 4,
            num_experts: 512,
            experts_per_token: 10,
            moe_intermediate: 640,
            shared_intermediate: 640,
            // The real MQ4 artifact is the stacked-expert aggregate layout.
            route: s1::WindowRoute::Aggregate,
        }
    }

    /// The S1 charge exactly as admission computes it ([`s1::census_of`], every
    /// term: ring, lookahead, route storage, A4, group scatter, HC arena and
    /// boundary state), never a second copy of the formula.
    fn census_total() -> u64 {
        let (_, _, census) = s1::census_of(&inputs(0, 7.09)).unwrap();
        census.total().unwrap()
    }

    /// Free VRAM that keeps exactly `layers` layers beside the S1 charge.
    fn free_for(layers: u64) -> u64 {
        NON_EXPERT + BASE_RESERVE + MARGIN + census_total() + layers * LAYER + 1
    }

    fn inputs(free: u64, link_gbps: f64) -> s1::AdmissionInputs {
        inputs_with_memory(
            s1::AdmissionMemory {
                pre_init_free_bytes: None,
                free_bytes: free,
            },
            link_gbps,
        )
    }

    fn inputs_with_memory(memory: s1::AdmissionMemory, link_gbps: f64) -> s1::AdmissionInputs {
        s1::AdmissionInputs {
            geometry: geometry(),
            memory,
            non_expert: NON_EXPERT,
            layer_expert_bytes: LAYER,
            gate_up_layer_bytes: 512 * 1_740_800,
            down_layer_bytes: 512 * 870_400,
            trunk_layers: 48,
            base_reserve: BASE_RESERVE,
            baseline_layers: 10,
            fixed_state_bytes: FIXED_STATE,
            pending_save_bytes: PENDING_SAVE,
            diagnostic_bytes: 0,
            base_runtime_reserved_bytes: 0,
            granule: s1::GranuleInputs::default(),
            checkpoint_stride_rows: 8192,
            window_rows: s1::S1_WINDOW_ROWS,
            link_gbps,
            // The product loader's own purpose (never inferred from the
            // diagnostic bytes): it admits the product tile.
            purpose: s1::AdmissionPurpose::Product,
            sym_iu4_source_symmetric: None,
        }
    }

    /// `(probes, input builds, outcome)` of one selection.
    fn select(
        mode: s1::SlowlinkMode,
        arch: &str,
        link: HostLink,
        free: u64,
    ) -> (u32, u32, Result<Option<s1::SlowlinkProfile>, String>) {
        select_memory(
            mode,
            arch,
            link,
            s1::AdmissionMemory {
                pre_init_free_bytes: None,
                free_bytes: free,
            },
        )
    }

    /// [`select`] from an explicit memory observation (the optional sample
    /// before the link initialization and the actual one after it).
    fn select_memory(
        mode: s1::SlowlinkMode,
        arch: &str,
        link: HostLink,
        memory: s1::AdmissionMemory,
    ) -> (u32, u32, Result<Option<s1::SlowlinkProfile>, String>) {
        let (probes, builds) = (Cell::new(0), Cell::new(0));
        let outcome = s1::select_profile(
            mode,
            &shape(arch),
            || {
                probes.set(probes.get() + 1);
                link
            },
            |gbps| {
                builds.set(builds.get() + 1);
                Ok(inputs_with_memory(memory, gbps))
            },
        );
        (probes.get(), builds.get(), outcome)
    }

    #[test]
    fn slowlink_killed_selector_never_probes_or_builds_inputs() {
        let (probes, builds, outcome) = select(
            s1::SlowlinkMode::Off,
            "gfx1100",
            HostLink::Discrete { gbps: 7.09 },
            free_for(10),
        );
        assert_eq!((probes, builds), (0, 0));
        assert_eq!(outcome, Ok(None));
    }

    #[test]
    fn slowlink_unset_selector_probes_and_admits_like_force() {
        for mode in [s1::SlowlinkMode::Auto, s1::SlowlinkMode::Force] {
            let (probes, builds, outcome) = select(
                mode,
                "gfx1100",
                HostLink::Discrete { gbps: 7.09 },
                free_for(10),
            );
            assert_eq!((probes, builds), (1, 1), "{mode:?}");
            let profile = outcome.unwrap().expect("admitted");
            assert_eq!(profile.resident_layers, 10, "{mode:?}");
            assert_eq!(profile.tile_rows, s1::S1_PRODUCT_TILE_ROWS, "{mode:?}");
        }
    }

    /// 0.4.3 RC blocker: Flash-Next on a gfx1100 24 GB discrete card (the XTX,
    /// 7.1 GB/s H2D link) selects S1, places exactly the profile's resident
    /// layers, and then refused with `settled placement has no owning
    /// margin/stage charge`: the admitted-S1 arm never recorded a charge. The
    /// profile owns it, so the shared decision now returns it and the settled
    /// guard accepts it. D2 (`HIPFIRE_QWEN4_D2_DEFAULT` on or off) is decided
    /// AFTER the settled placement and is no input of this decision, which is why
    /// both XTX lanes failed identically; the same placement serves either.
    #[test]
    fn slowlink_admitted_gfx1100_placement_settles_with_the_profiles_own_charges() {
        use hipfire_arch_qwen4::expert_residency::ExpertVramLayers;
        for layers in [0u64, 8, 10, 11] {
            let (_, _, outcome) = select(
                s1::SlowlinkMode::Auto,
                "gfx1100",
                HostLink::Discrete { gbps: 7.09 },
                free_for(layers),
            );
            let profile = outcome.unwrap().expect("a narrow gfx1100 link admits S1");
            assert_eq!(profile.resident_layers as u64, layers);
            let slowlink = profile.reserve_bytes().unwrap();
            let decided = super::qwen4_decide_placement(
                super::Qwen4PlacementPolicy {
                    band: false,
                    hotset: false,
                    admitted_s1: Some(profile.charges),
                    ep_agreed: None,
                    use_ranges: false,
                    explicit: None,
                    slowlink_bytes: slowlink,
                    table_geometry: (48, 512, false),
                },
                || panic!("an admitted S1 load never censuses routed bytes"),
                || panic!("an admitted S1 load never re-queries capacity"),
                || panic!("a discrete card has no UMA policy"),
            )
            .expect("admitted S1 decision");
            assert_eq!(decided.policy, Some(ExpertVramLayers::Auto), "{layers}");
            assert!(decided.measured.is_none() && decided.reserve.is_none());
            // The settled guard accepts the placement with the owning charge.
            let charges = super::qwen4_settled_charges(decided.charges)
                .expect("an admitted S1 placement carries its owning charges");
            assert_eq!(charges.margin, MARGIN, "{layers}");
            assert_eq!(
                charges.stage, 0,
                "S1 allocates no dedicated stage: {layers}"
            );
            assert_eq!(charges.slowlink, slowlink, "{layers}");
            // Exactly what the loader's own census would derive on gfx1100 (no
            // dedicated slab stage), and the reserve the layers were quoted on.
            assert_eq!(
                charges,
                super::qwen4_shared_charges(LAYER, true, false, slowlink).unwrap()
            );
            assert_eq!(
                charges.reserve(BASE_RESERVE).unwrap(),
                profile.placement.reserve_bytes,
                "{layers}"
            );
        }
        // A settled placement no owner charged is still a refusal, never zero.
        assert!(super::qwen4_settled_charges(None)
            .unwrap_err()
            .contains("no owning margin/stage charge"));
    }

    #[test]
    fn slowlink_arch_is_identity_and_still_probes_for_inputs() {
        // No exact-arch allowlist: every arch probes once and builds inputs on
        // a narrow link; only sealed capability verdicts refuse.
        for arch in [
            "gfx906", "gfx942", "gfx1010", "gfx1030", "gfx1100", "gfx1101", "gfx1151", "gfx1200",
            "gfx1201",
        ] {
            let (probes, builds, outcome) = select(
                s1::SlowlinkMode::Force,
                arch,
                HostLink::Discrete { gbps: 7.09 },
                free_for(10),
            );
            assert_eq!((probes, builds), (1, 1), "{arch}");
            assert!(outcome.unwrap().is_some(), "{arch}");
        }
    }

    #[test]
    fn slowlink_link_truth_table_selects_only_a_measured_narrow_discrete_link() {
        for (link, selected) in [
            (HostLink::Discrete { gbps: 7.09 }, true),
            (HostLink::Discrete { gbps: 15.99 }, true),
            (HostLink::Discrete { gbps: 16.0 }, false),
            (HostLink::Discrete { gbps: 26.0 }, false),
            (HostLink::Discrete { gbps: 50.0 }, false),
            (HostLink::Unified, false),
            (HostLink::Unmeasured, false),
        ] {
            let (probes, builds, outcome) =
                select(s1::SlowlinkMode::Force, "gfx1100", link, free_for(10));
            // The link is measured once, and no S1 input is built (no memory
            // query, manifest scan or allocation) for a link that cannot select.
            assert_eq!(probes, 1, "{link:?}");
            assert_eq!(builds, u32::from(selected), "{link:?}");
            let profile = outcome.unwrap();
            assert_eq!(profile.is_some(), selected, "{link:?}");
            if let Some(profile) = profile {
                // `free_for(10)` keeps exactly ten, so ten is also the maximal fit
                // (the count is chosen by fit; ten is only the preferred count).
                assert_eq!(profile.resident_layers, 10);
                assert_eq!(profile.tile_rows, s1::S1_PRODUCT_TILE_ROWS);
                assert_eq!(profile.purpose, s1::AdmissionPurpose::Product);
            }
        }
    }

    /// The resident count is the LARGEST that fits (never a floor): zero is a
    /// valid fit when the base fits, nine and ten are plain data, eleven is
    /// admitted when the free memory keeps it.
    #[test]
    fn slowlink_forced_admission_keeps_the_maximal_fitting_resident_count() {
        for layers in [0u64, 1, 9, 10, 11] {
            let (_, builds, outcome) = select(
                s1::SlowlinkMode::Force,
                "gfx1100",
                HostLink::Discrete { gbps: 7.09 },
                free_for(layers),
            );
            assert_eq!(builds, 1);
            let profile = outcome.unwrap().expect("a fitting load is admitted");
            assert_eq!(profile.resident_layers as u64, layers, "{layers}");
            // One byte above the layer boundary is the whole discretionary slack.
            assert_eq!(profile.device_snapshot_budget_bytes, 1, "{layers}");
        }
        // One byte less than layer `n` keeps `n - 1` with the rest as slack.
        for layers in [1u64, 10] {
            let (_, _, outcome) = select(
                s1::SlowlinkMode::Force,
                "gfx1100",
                HostLink::Discrete { gbps: 7.09 },
                free_for(layers) - 2,
            );
            let profile = outcome.unwrap().expect("a fitting load is admitted");
            assert_eq!(profile.resident_layers as u64, layers - 1, "{layers}");
            assert_eq!(profile.device_snapshot_budget_bytes, LAYER - 1, "{layers}");
        }
    }

    /// A base that does not fit is a forced capacity refusal naming the
    /// selector, distinct from the valid zero-resident fit above.
    #[test]
    fn slowlink_forced_base_deficit_fails_the_load_not_a_zero_resident_fit() {
        let (_, builds, outcome) = select(
            s1::SlowlinkMode::Force,
            "gfx1100",
            HostLink::Discrete { gbps: 7.09 },
            free_for(0) - 2,
        );
        assert_eq!(builds, 1);
        let error = outcome.unwrap_err();
        assert!(
            error.contains("HIPFIRE_QWEN4_SLOWLINK_PREFILL=1"),
            "{error}"
        );
    }

    /// The measured initialization loss reaches admission only through the
    /// actual free sample taken after it: the same pre-initialization free
    /// keeps ten with no loss and nine after the j1811 measured 153,092,096 B
    /// interval. No tax is assumed and no sample is invented when unknown.
    #[test]
    fn slowlink_admission_charges_the_measured_initialization_loss_once() {
        const MEASURED_LOSS: u64 = 25_713_180_672 - 25_560_088_576;
        assert_eq!(MEASURED_LOSS, 153_092_096);
        let before = free_for(10);
        let link = HostLink::Discrete { gbps: 7.09 };
        let admit = |memory| {
            select_memory(s1::SlowlinkMode::Force, "gfx1100", link, memory)
                .2
                .unwrap()
                .expect("admitted")
        };
        let same = admit(s1::AdmissionMemory {
            pre_init_free_bytes: Some(before),
            free_bytes: before,
        });
        assert_eq!(same.resident_layers, 10);
        assert_eq!(
            same.placement.memory.observed_initialization_bytes(),
            Some(0)
        );
        let lost = admit(s1::AdmissionMemory {
            pre_init_free_bytes: Some(before),
            free_bytes: before - MEASURED_LOSS,
        });
        assert_eq!(lost.resident_layers, 9);
        assert_eq!(
            lost.placement.memory.observed_initialization_bytes(),
            Some(MEASURED_LOSS)
        );
        // Charged once: slack is the actual free minus the nine-layer need.
        assert_eq!(lost.device_snapshot_budget_bytes, LAYER + 1 - MEASURED_LOSS);
        let unknown = admit(s1::AdmissionMemory {
            pre_init_free_bytes: None,
            free_bytes: before - MEASURED_LOSS,
        });
        assert_eq!(unknown.resident_layers, 9);
        assert_eq!(
            unknown.placement.memory.observed_initialization_bytes(),
            None
        );
    }

    #[test]
    fn slowlink_ineligible_forced_loads_are_the_incumbent_not_an_error() {
        let mut load = shape("gfx1100");
        load.explicit_vram_layers = true;
        let outcome = s1::select_profile(
            s1::SlowlinkMode::Force,
            &load,
            || unreachable!("probed"),
            |_| unreachable!("inputs built"),
        );
        assert_eq!(outcome, Ok(None));
        // A fully resident fit has nothing to stream: incumbent, not an error.
        let (_, builds, outcome) = select(
            s1::SlowlinkMode::Force,
            "gfx1100",
            HostLink::Discrete { gbps: 7.09 },
            1 << 40,
        );
        assert_eq!((builds, outcome), (1, Ok(None)));
    }
}

#[cfg(test)]
mod ep_lane_reservation_tests {
    use super::qwen4_ep_lane_request;
    use hipfire_arch_qwen4::Qwen4KvBackend;
    use hipfire_runtime::loader_api::BatchLoadRequest;

    fn request(vmm_batch: bool, lanes: usize, row_budget: usize) -> BatchLoadRequest {
        BatchLoadRequest {
            vmm_batch,
            continuous_batch_size: lanes,
            row_budget,
        }
    }

    #[test]
    fn ep_lane_reservation_off_and_single_lane_request_nothing() {
        let never = |_: usize, _: usize| -> Option<u64> { panic!("no bytes without a request") };
        for batch in [
            BatchLoadRequest::default(),
            request(true, 1, 4096),
            request(false, 8, 4096),
        ] {
            assert_eq!(
                qwen4_ep_lane_request(&batch, Qwen4KvBackend::Vmm, never),
                None
            );
        }
    }

    #[test]
    fn ep_lane_reservation_request_uses_lanes_and_the_staged_row_budget() {
        let seen = std::cell::Cell::new((0, 0));
        let lane = qwen4_ep_lane_request(&request(true, 8, 4), Qwen4KvBackend::Vmm, |l, r| {
            seen.set((l, r));
            Some(123)
        })
        .unwrap();
        // Row budget below the lane count is raised to it, as the staging does.
        assert_eq!(seen.get(), (8, 8));
        assert_eq!((lane.lanes, lane.row_budget, lane.bytes), (8, 8, Ok(123)));
        let lane = qwen4_ep_lane_request(&request(true, 4, 4096), Qwen4KvBackend::Vmm, |l, r| {
            seen.set((l, r));
            Some(7)
        })
        .unwrap();
        assert_eq!(seen.get(), (4, 4096));
        assert_eq!(lane.bytes, Ok(7));
    }

    #[test]
    fn ep_lane_reservation_request_reports_overflow_and_non_vmm_backend() {
        let lane = qwen4_ep_lane_request(&request(true, 4, 4096), Qwen4KvBackend::Vmm, |_, _| None)
            .unwrap();
        assert!(lane.bytes.unwrap_err().contains("overflow"));
        let lane =
            qwen4_ep_lane_request(&request(true, 4, 4096), Qwen4KvBackend::Legacy, |_, _| {
                panic!("legacy QSA never stages lanes")
            })
            .unwrap();
        assert!(lane.bytes.unwrap_err().contains("not VMM"));
    }
}

#[cfg(test)]
mod qwen4_preload_plan_tests {
    use super::{
        preload_keep_mask_bytes, preload_ple_envelope, preload_staging_bytes,
        preload_whole_host_join, qwen4_shared_apply_placement, qwen4_shared_auto_whole_layers,
        qwen4_shared_charges, qwen4_shared_table_mirror, Qwen4SharedPlacement,
        Qwen4SharedPlacementInputs,
    };
    const MIB: u64 = 1 << 20;

    #[test]
    fn preload_staging_floors_one_8mib_budget_of_whole_rows() {
        // A single row wants less than the budget: one buffer is the floored
        // budget, two buffers double it.
        assert_eq!(preload_staging_bytes(1, 1), Ok((8 * MIB, 16 * MIB)));
        assert_eq!(preload_staging_bytes(4096, 2048), Ok((8 * MIB, 16 * MIB)));
    }

    #[test]
    fn preload_staging_grows_past_128mib_without_a_cap() {
        // The 128MiB bound is the ordered device-upload ring's, never PLE
        // staging's: a large prefetch charges its real two buffers.
        let wanted = 100_000u64 * 2048;
        assert_eq!(
            preload_staging_bytes(100_000, 2048),
            Ok((wanted, wanted * 2))
        );
    }

    #[test]
    fn preload_staging_rejects_zero_rows_and_overflow() {
        assert!(preload_staging_bytes(16, 0).is_err());
        assert!(preload_staging_bytes(usize::MAX, 2).is_err());
        assert!(preload_staging_bytes(usize::MAX, usize::MAX as u64).is_err());
    }

    #[test]
    fn preload_keep_mask_counts_trunk_layers_plus_head() {
        assert_eq!(preload_keep_mask_bytes(48, 512), Ok(49 * 512));
        assert_eq!(preload_keep_mask_bytes(0, 8), Ok(8));
    }

    #[test]
    fn preload_keep_mask_rejects_overflow() {
        assert!(preload_keep_mask_bytes(usize::MAX, 2).is_err());
        assert!(preload_keep_mask_bytes(usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn preload_ple_envelope_is_header_data_not_file_extent() {
        // A small header section inside a large file: the envelope is the
        // header's data bytes; the extent size is irrelevant.
        assert_eq!(preload_ple_envelope(100, 50), Ok(100));
        assert_eq!(preload_ple_envelope(0, 0), Ok(0));
    }

    #[test]
    fn preload_ple_envelope_rejects_corrupt_header_arithmetic() {
        assert!(preload_ple_envelope(u64::MAX, 1).is_err());
        assert!(preload_ple_envelope(u64::MAX, u64::MAX).is_err());
    }

    fn join_union(host: u64, device: u64, backed: u64, uma: bool) -> serde_json::Value {
        serde_json::json!({
            "host_union_upper_bound_bytes": host,
            "device_union_upper_bound_bytes": device,
            "device_backed_host_bytes": backed,
            "uma_shared_backing": uma,
        })
    }

    #[test]
    fn preload_whole_host_join_splits_pinned_and_ram_only() {
        let headroom = hipfire_arch_qwen4::expert_residency::HOST_RAM_HEADROOM_BYTES;
        assert_eq!(headroom, 4 << 30);
        let (whole_host, report) = preload_whole_host_join(
            &join_union(100, 1000, 0, false),
            false,
            10, // PLE resident envelope: RAM-only, never pinned demand
            2,  // PLE staging: pinned
            4,  // page cache: RAM-only
            8,  // upload transient: pinned
            16, // session cap: RAM-only
            Some(1 << 40),
            32, // TTM pool: observed reclaimable, never planned
        )
        .expect("join");
        // Pinned/GTT demand: A host union + pinned staging/upload only.
        assert_eq!(
            report["pinned_gtt_demand_bytes"],
            serde_json::json!(100 + 2 + 8)
        );
        // RAM-only terms stay separate, never folded into pinned demand.
        assert_eq!(
            report["ram_only_host_bytes"],
            serde_json::json!(10 + 4 + 16)
        );
        assert_eq!(whole_host, 100 + 2 + 8 + 10 + 4 + 16);
        assert_eq!(
            report["whole_host_union_bytes"],
            serde_json::json!(whole_host)
        );
        // Headroom is an explicit separate allowance on the pinned demand.
        assert_eq!(report["headroom_bytes"], serde_json::json!(headroom));
        assert_eq!(
            report["admission_needed_bytes"],
            serde_json::json!(100 + 2 + 8 + headroom)
        );
        // TTM-adjusted capacity is observed reclaimable pages, never planned.
        assert_eq!(
            report["admission_capacity_bytes_or_null"],
            serde_json::json!((1_u64 << 40) + 32)
        );
        assert_eq!(report["a_device_union_bytes"], serde_json::json!(1000));
        assert_eq!(report["a_host_union_bytes"], serde_json::json!(100));
        assert_eq!(report["a_device_backed_host_bytes"], serde_json::json!(0));
    }

    #[test]
    fn preload_whole_host_join_reports_missing_capacity_as_null() {
        let (_, report) = preload_whole_host_join(
            &join_union(100, 0, 0, false),
            false,
            0,
            0,
            0,
            0,
            0,
            None,
            32,
        )
        .expect("join with unknown capacity");
        // No zero assumed: JSON null plus a named unavailable reason.
        assert_eq!(
            report["admission_capacity_bytes_or_null"],
            serde_json::Value::Null
        );
        assert!(report["admission_capacity_note"]
            .as_str()
            .is_some_and(|note| note.contains("unreadable")));
    }

    #[test]
    fn preload_whole_host_join_rejects_missing_keys_and_backing_mismatch() {
        let good = join_union(1, 2, 0, false);
        assert!(
            preload_whole_host_join(&serde_json::json!({}), false, 0, 0, 0, 0, 0, None, 0).is_err()
        );
        assert!(preload_whole_host_join(&good, true, 0, 0, 0, 0, 0, None, 0).is_err());
    }

    #[test]
    fn preload_whole_host_join_rejects_overflow() {
        // Envelope at u64::MAX overflows the RAM-only checked add, never wraps.
        assert!(preload_whole_host_join(
            &join_union(1, 2, 0, false),
            false,
            u64::MAX,
            0,
            0,
            0,
            0,
            None,
            0
        )
        .is_err());
        // Headroom on top of u64::MAX pinned demand overflows the admission add.
        assert!(preload_whole_host_join(
            &join_union(u64::MAX, 0, 0, false),
            false,
            0,
            1,
            0,
            0,
            0,
            None,
            0
        )
        .is_err());
        // Some(MemAvailable) near u64::MAX plus TTM pool overflows the checked
        // capacity add instead of saturating.
        assert!(preload_whole_host_join(
            &join_union(0, 0, 0, false),
            false,
            0,
            0,
            0,
            0,
            0,
            Some(u64::MAX),
            1
        )
        .is_err());
    }

    fn shared_inputs(
        explicit: Option<hipfire_arch_qwen4::expert_residency::ExpertVramLayers>,
        use_ranges: bool,
        layer_experts: u64,
        reserve_base: u64,
        _slowlink_bytes: u64,
        free_bytes: u64,
    ) -> Qwen4SharedPlacementInputs {
        Qwen4SharedPlacementInputs {
            stage_enabled: true,
            num_hidden_layers: 48,
            num_experts: 512,
            moe_layers: 49,
            recipe_is_slab: false,
            use_ranges,
            slab_stage: false,
            borrowed_donors: false,
            whole_first: hipfire_arch_qwen4::expert_hotset::WholeFirstDecision::Inactive,
            explicit,
            expert_keep: false,
            non_expert: 0,
            layer_experts,
            reserve_base,
            free_bytes,
        }
    }
    #[test]
    fn preload_shared_policy_keeps_admitted_and_fixed_paths_lazy() {
        use hipfire_arch_qwen4::expert_residency::{ExpertVramLayers, PlacementCharges};
        let s1_charges = PlacementCharges::new(1 << 30, false)
            .unwrap()
            .with_slowlink(7 << 20);
        for (hotset, admitted_s1, explicit, expected) in [
            (true, None, None, None),
            (false, Some(s1_charges), None, Some(ExpertVramLayers::Auto)),
            (
                false,
                None,
                Some(ExpertVramLayers::Layers(0)),
                Some(ExpertVramLayers::Layers(0)),
            ),
        ] {
            let decided = super::qwen4_decide_placement(
                super::Qwen4PlacementPolicy {
                    band: false,
                    hotset,
                    admitted_s1,
                    ep_agreed: None,
                    use_ranges: false,
                    explicit,
                    slowlink_bytes: 0,
                    table_geometry: (48, 512, false),
                },
                || panic!("this policy must not census routed bytes"),
                || panic!("this policy must not query capacity"),
                || panic!("this policy must not inspect UMA oom policy"),
            )
            .expect("lazy policy");
            assert_eq!(decided.policy, expected);
            // An admitted S1 profile settles with ITS charges; every other lazy
            // arm defers the charge to the owner that sizes it.
            assert_eq!(decided.charges, admitted_s1);
            assert!(decided.measured.is_none());
        }
    }

    #[test]
    fn preload_shared_native_off_auto_reuses_quote_and_charged_reserve() {
        use hipfire_arch_qwen4::expert_residency as residency;
        let reserve = residency::AutoReserve::new(
            None,
            residency::ReserveCensus {
                qsa_context: 0,
                state: 0,
                forward: 0,
                f16_shadows: 0,
                shared_i8_shadows: 0,
                workspace: std::array::from_fn(|_| ("fixture", 0)),
                qsa_selector: 0,
                mtp: 0,
                gather: 0,
                runtime: 32,
            },
        )
        .expect("native-off reserve");
        let mut inputs = shared_inputs(None, false, 1 << 30, reserve.bytes(), 0, 0);
        inputs.num_hidden_layers = 8;
        inputs.moe_layers = 8;
        inputs.slab_stage = true;
        inputs.recipe_is_slab = true;
        let charges = qwen4_shared_charges(inputs.layer_experts, true, true, 0).unwrap();
        let tables = qwen4_shared_table_mirror(&inputs, 1).unwrap().0;
        inputs.free_bytes =
            charges.reserve(reserve.bytes()).unwrap() + tables + 2 * inputs.layer_experts;
        let queries = std::cell::Cell::new(0);
        let decided = super::qwen4_decide_placement(
            super::Qwen4PlacementPolicy {
                band: false,
                hotset: false,
                admitted_s1: None,
                ep_agreed: None,
                use_ranges: false,
                explicit: None,
                slowlink_bytes: 0,
                table_geometry: (
                    inputs.num_hidden_layers,
                    inputs.num_experts,
                    inputs.expert_keep,
                ),
            },
            || Ok(8 * inputs.layer_experts),
            || {
                queries.set(queries.get() + 1);
                Ok((
                    inputs.free_bytes,
                    0,
                    inputs.layer_experts,
                    reserve,
                    None,
                    None,
                ))
            },
            || false,
        )
        .unwrap();
        assert_eq!(decided.policy, Some(residency::ExpertVramLayers::Auto));
        let measured = decided.measured.expect("unset keeps its deciding quote");
        assert_eq!(queries.get(), 1);
        assert_eq!(measured.0, inputs.free_bytes);
        assert_eq!(measured.3.census.mtp, 0);
        let (layers, _, charges) = qwen4_shared_auto_whole_layers(&inputs, &charges).unwrap();
        assert_eq!(layers, 2);
        assert!(charges.stage > 0);
        assert_eq!(
            charges.reserve(reserve.bytes()).unwrap(),
            reserve.bytes() + charges.margin + charges.stage
        );
        let mut weights: Vec<_> = (0..8)
            .map(|layer| {
                hipfire_runtime::weight_manifest::WeightEntry::layer(
                    &format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"),
                    layer,
                    vec![],
                    rdna_compute::DType::F16,
                    hipfire_runtime::weight_manifest::ShardPolicy::Replicate,
                )
            })
            .collect();
        let (_, _, moved, device, host, tables, mirror) = super::qwen4_shared_actual_placement(
            &mut weights,
            &|_| Some(inputs.layer_experts),
            &inputs,
            Some(layers),
        )
        .unwrap();
        assert_eq!(moved, 6);
        assert_eq!(device, 2 * inputs.layer_experts);
        assert_eq!(
            host,
            6 * inputs.layer_experts + 6 * residency::HOST_MAPPED_PAD_BYTES
        );
        assert_eq!(
            device + host,
            8 * inputs.layer_experts + 6 * residency::HOST_MAPPED_PAD_BYTES
        );
        assert_eq!(
            tables,
            residency::EXPERT_TIER_TABLE_BYTES
                + inputs.num_hidden_layers as u64
                    * 2
                    * (2 * inputs.num_experts * rdna_compute::DType::F32.size()) as u64,
        );
        assert_eq!(mirror, 0);
        assert_eq!(queries.get(), 1);
    }

    #[test]
    fn shared_charges_conserve_margin_stage_slowlink() {
        use hipfire_arch_qwen4::expert_residency::ExpertVramLayers;
        let _ = ExpertVramLayers::Auto;
        let layer = 1u64 << 30;
        let charges = qwen4_shared_charges(layer, true, false, 7).expect("charges");
        let base = 3u64 << 30;
        let charged = charges.reserve(base).expect("charged reserve");
        let expect = base
            .checked_add(charges.margin)
            .and_then(|bytes| bytes.checked_add(charges.stage))
            .and_then(|bytes| bytes.checked_add(7))
            .expect("fixture arithmetic");
        assert_eq!(charged, expect);
        assert!(qwen4_shared_charges(u64::MAX, true, false, 1)
            .expect("charges")
            .reserve(u64::MAX)
            .is_err());
    }

    #[test]
    fn shared_fixed_n0_moves_every_routed_expert() {
        use hipfire_arch_qwen4::expert_residency::ExpertVramLayers;
        let inputs = shared_inputs(Some(ExpertVramLayers::Layers(0)), false, 10, 0, 0, 0);
        let mut weights = vec![hipfire_runtime::weight_manifest::WeightEntry::layer(
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            0,
            vec![],
            rdna_compute::DType::F16,
            hipfire_runtime::weight_manifest::ShardPolicy::Replicate,
        )];
        let bytes_of = |_: &hipfire_runtime::weight_manifest::WeightEntry| Some(8u64);
        let (mode, whole, moved, _routed, host, tables, mirror) =
            super::qwen4_shared_actual_placement(&mut weights, &bytes_of, &inputs, Some(0))
                .expect("fixed-0");
        assert_eq!(mode, Qwen4SharedPlacement::FixedLayers(0));
        assert_eq!(whole, Some(0));
        assert_eq!(moved, 1);
        assert_eq!(
            host,
            8 + hipfire_arch_qwen4::expert_residency::HOST_MAPPED_PAD_BYTES
        );
        assert_eq!(
            tables,
            hipfire_arch_qwen4::expert_residency::EXPERT_TIER_TABLE_BYTES
                + 48 * 2 * (2 * 512 * rdna_compute::DType::F32.size()) as u64
        );
        assert_eq!(
            mirror,
            20 + 4 * hipfire_arch_qwen4::expert_residency::HOST_MAPPED_PAD_BYTES
        );
        let nativeoff = Qwen4SharedPlacementInputs {
            use_ranges: true,
            ..shared_inputs(None, true, 10, 0, 0, 0)
        };
        let (mode, _, _, _, _, _, _) =
            super::qwen4_shared_actual_placement(&mut weights, &bytes_of, &nativeoff, None)
                .expect("nativeoff-uma");
        assert_eq!(mode, Qwen4SharedPlacement::Unified);
    }

    #[test]
    fn shared_fixed_mid_all_conserve_routed_bytes() {
        // Fixed-N mid (24 of 48) and all (48 of 48): the decided count is the
        // stated N, the mutation moves exactly the trunk layers >= N plus the
        // MTP layer, and resident + host payload conserves routed bytes;
        // pinned allocation padding is charged separately by the host owner.
        for layers in [24usize, 48] {
            let inputs = shared_inputs(
                Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Layers(layers)),
                false,
                10,
                0,
                0,
                0,
            );
            let entry = |layer: usize| {
                hipfire_runtime::weight_manifest::WeightEntry::layer(
                    &format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"),
                    layer,
                    vec![],
                    rdna_compute::DType::F16,
                    hipfire_runtime::weight_manifest::ShardPolicy::Replicate,
                )
            };
            let mut weights: Vec<_> = (0..48).map(entry).collect();
            weights.push(hipfire_runtime::weight_manifest::WeightEntry::layer(
                "mtp.layers.0.mlp.experts.gate_up_proj",
                0,
                vec![],
                rdna_compute::DType::F16,
                hipfire_runtime::weight_manifest::ShardPolicy::Replicate,
            ));
            let bytes_of = |_: &hipfire_runtime::weight_manifest::WeightEntry| Some(8u64);
            let total = 49u64 * 8;
            let (mode, whole, moved, routed, host, tables, mirror) =
                super::qwen4_shared_actual_placement(
                    &mut weights,
                    &bytes_of,
                    &inputs,
                    Some(layers),
                )
                .expect("fixed-n");
            assert_eq!(mode, Qwen4SharedPlacement::FixedLayers(layers));
            assert_eq!(whole, Some(layers));
            assert_eq!(moved, 49 - layers.min(48));
            assert_eq!(
                routed + host,
                total + moved as u64 * hipfire_arch_qwen4::expert_residency::HOST_MAPPED_PAD_BYTES,
            );
            if layers == 48 {
                assert_eq!(moved, 1);
                assert_eq!(
                    host,
                    8 + hipfire_arch_qwen4::expert_residency::HOST_MAPPED_PAD_BYTES
                );
                assert_eq!(
                    tables,
                    hipfire_arch_qwen4::expert_residency::EXPERT_TIER_TABLE_BYTES
                );
                assert_eq!(mirror, 0);
            } else {
                assert!(moved > 0 && host > 0 && routed > 0);
            }
        }
    }

    #[test]
    fn shared_auto_counts_mirrors_tables_overflow() {
        let layer = 2u64 << 30;
        let inputs = shared_inputs(None, false, layer, 1u64 << 30, 5u64 << 20, u64::MAX / 4);
        let charges = qwen4_shared_charges(layer, true, false, 5u64 << 20).expect("charges");
        assert_eq!(
            charges
                .reserve(inputs.reserve_base)
                .expect("auto charged reserve"),
            inputs
                .reserve_base
                .checked_add(charges.margin)
                .and_then(|bytes| bytes.checked_add(charges.stage))
                .and_then(|bytes| bytes.checked_add(5u64 << 20))
                .expect("fixture arithmetic")
        );
        let (tables, mirror) = qwen4_shared_table_mirror(&inputs, 1).expect("owners");
        assert_eq!(
            tables,
            hipfire_arch_qwen4::expert_residency::EXPERT_TIER_TABLE_BYTES
                + 48 * 2 * (2 * 512 * rdna_compute::DType::F32.size()) as u64
        );
        assert_eq!(
            mirror,
            layer
                .checked_mul(2)
                .and_then(|bytes| bytes
                    .checked_add(4 * hipfire_arch_qwen4::expert_residency::HOST_MAPPED_PAD_BYTES))
                .expect("fixture arithmetic")
        );
        assert!(qwen4_shared_table_mirror(&inputs, u64::MAX).is_ok());
        let overflow = Qwen4SharedPlacementInputs {
            layer_experts: u64::MAX,
            ..shared_inputs(None, false, u64::MAX, 0, 0, 0)
        };
        assert!(qwen4_shared_table_mirror(&overflow, 1).is_err());

        use hipfire_arch_qwen4::{
            expert_hotset::HotsetBudget,
            expert_residency as residency,
            expert_slab_table::{ExpertSlab, ExpertSlabSpan},
            expert_slabs::ExpertFormat,
            expert_tier_plan::{plan_expert_tiers_from_slabs, Placement},
        };
        // Real FN geometry, metadata only, with the same nonaligned spans as
        // the tier planner's CPU fixtures. Packing/tail costs belong to the
        // owner planner, not this budget oracle.
        let slabs: Vec<_> = (0..48 * 512)
            .map(|id| ExpertSlab {
                format: ExpertFormat::Mq4Xts,
                gate_up: ExpertSlabSpan {
                    offset: id * 1500,
                    len: 1000,
                },
                down: ExpertSlabSpan {
                    offset: id * 1500 + 1000,
                    len: 500,
                },
            })
            .collect();
        let rank: Vec<u32> = (0..48 * 512).collect();
        for (stage_enabled, keep, payload_room) in [
            (false, false, 16u64 << 20),
            (true, false, 16 << 20),
            (true, true, 16 << 20),
            (true, false, 64 << 20),
        ] {
            let free = residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES + payload_room;
            let total = residency::expert_hotset_budget(free, 0, 0).unwrap();
            let (plan, budget, stage) = super::qwen4_plan_hotset_placement(
                &slabs,
                &rank,
                HotsetBudget::parse(None).unwrap(),
                total,
                stage_enabled,
                (48, 512, keep),
                None,
                &[],
            )
            .unwrap();
            // Match actual stage construction: two F32[2 * experts] pointer
            // tensors per staged layer, and NO additional codec tensor.
            let host = plan.owner_bytes(Placement::HostMapped) > 0;
            let staged = stage_enabled && host;
            let stage_tables = if staged {
                48 * 2 * (2 * 512 * rdna_compute::DType::F32.size()) as u64
            } else {
                0
            };
            let tables =
                (48 * 512 * (2 * 8 + 1)) as u64 + stage_tables + if keep { 49 * 512 } else { 0 };
            assert_eq!(
                super::qwen4_expert_table_bytes(staged, (48, 512, keep)).unwrap(),
                tables,
            );
            assert_eq!(host, payload_room == 16 << 20);
            let device = plan.owner_bytes(Placement::Device);
            assert!(device + stage + tables + residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES <= free);
            // Exact available budget also rules out a duplicate table or
            // expert-payload charge hidden inside the stage reserve.
            assert_eq!(
                budget + stage + tables + residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES,
                free
            );
            assert_eq!(stage > 0, staged);
            if staged && !keep {
                // The old solve budgets only the base table, and its actual
                // packed owners exceed the settled two-table bound.
                let old = plan_expert_tiers_from_slabs(&slabs, &rank, total - stage).unwrap();
                assert!(
                    old.owner_bytes(Placement::Device)
                        + stage
                        + tables
                        + residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES
                        > free
                );
            }
        }

        // Whole-layer AUTO must charge those same separate tables without
        // counting the routed payload twice. A one-byte table shortfall
        // cedes one layer through the existing solver, never a new cap.
        let mut tight = shared_inputs(None, false, layer, 1 << 30, 0, 0);
        tight.expert_keep = true;
        // Beta policy: the distinct stage pointer tables are charged only for a
        // dedicated slab stage that a host placement actually builds.
        tight.slab_stage = true;
        let charges = qwen4_shared_charges(layer, true, true, 0).unwrap();
        let tables = qwen4_shared_table_mirror(&tight, 1).unwrap().0;
        tight.free_bytes = charges.reserve(tight.reserve_base).unwrap() + 3 * layer + tables;
        assert_eq!(
            qwen4_shared_auto_whole_layers(&tight, &charges).unwrap().0,
            3
        );
        tight.free_bytes -= 1;
        assert_eq!(
            qwen4_shared_auto_whole_layers(&tight, &charges).unwrap().0,
            2
        );

        // At the exact fully-resident boundary AUTO charges the shipped
        // `charges` payload once (it is unconditional in the landed policy)
        // and the base codec table once; the distinct stage pointer tables
        // are charged only by a stage re-solve, which a fully resident
        // placement never takes. Cover both policy routes, the stage-off
        // path and independent keep masks.
        for stage_enabled in [false, true] {
            for keep in [false, true] {
                for whole_first in [
                    hipfire_arch_qwen4::expert_hotset::WholeFirstDecision::Inactive,
                    hipfire_arch_qwen4::expert_hotset::WholeFirstDecision::Active,
                ] {
                    let mut full = shared_inputs(None, false, layer, 1 << 30, 0, 0);
                    full.stage_enabled = stage_enabled;
                    full.expert_keep = keep;
                    full.whole_first = whole_first;
                    full.borrowed_donors = stage_enabled;
                    let charges = qwen4_shared_charges(layer, true, stage_enabled, 0).unwrap();
                    let tables = 48 * 512 * (2 * 8 + 1) + if keep { 49 * 512 } else { 0 };
                    full.free_bytes = charges.reserve(full.reserve_base).unwrap()
                        + full.num_hidden_layers as u64 * layer
                        + tables;
                    let (layers, _plan, settled) =
                        qwen4_shared_auto_whole_layers(&full, &charges).unwrap();
                    assert_eq!(layers, full.num_hidden_layers);
                    assert_eq!(settled.margin, charges.margin);
                    assert_eq!(settled.stage, charges.stage);
                    assert_eq!(qwen4_shared_table_mirror(&full, 0).unwrap().0, tables);
                    full.free_bytes -= 1;
                    assert!(
                        qwen4_shared_auto_whole_layers(&full, &charges).unwrap().0
                            < full.num_hidden_layers
                    );
                }
            }
        }
    }

    #[test]
    fn shared_mutation_reuses_canonical_owners() {
        let mut inputs = shared_inputs(
            Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Layers(1)),
            false,
            10,
            0,
            0,
            0,
        );
        // This fixture tests canonical mutation, not staging. Budget the
        // stage-off owners plus exactly one layer through the real census;
        // charging stage tables here leaves room for all 48 tiny payloads.
        inputs.stage_enabled = false;
        let charges = qwen4_shared_charges(10, true, false, 0).expect("charges");
        inputs.free_bytes = inputs.non_expert
            + charges.reserve(inputs.reserve_base).expect("reserve")
            + qwen4_shared_table_mirror(&inputs, 0)
                .expect("table owners")
                .0
            + inputs.layer_experts;
        let entry = |layer: usize| {
            hipfire_runtime::weight_manifest::WeightEntry::layer(
                &format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"),
                layer,
                vec![],
                rdna_compute::DType::F16,
                hipfire_runtime::weight_manifest::ShardPolicy::Replicate,
            )
        };
        let mut weights = vec![entry(0), entry(1)];
        let moved = qwen4_shared_apply_placement(&mut weights, 1, None);
        assert_eq!(moved, 1);
        let mut weights = vec![entry(0), entry(1)];
        let moved = qwen4_shared_apply_placement(&mut weights, 0, Some(&[1usize]));
        assert_eq!(moved, 1);
        let (whole, _, _) = qwen4_shared_auto_whole_layers(&inputs, &charges).expect("auto count");
        assert_eq!(whole, 1);
    }
}
