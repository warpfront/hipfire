// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_rowtile_capture — live eager rowtile operand producer.
//!
//! Loads a pinned FlashNext model via the production serve loader
//! (`hipfire_loader::{open_qwen4_rank_source, load_qwen4_rank}` with AR-only
//! spec), runs the canonical prompt prefill, then performs exactly
//! 64 sequential real singleton `forward_token` greedy continuation steps while
//! a live `lab` projection observer (NOT replay-prefix, NOT retained_AR state
//! reconstruction) downloads the actual immutable GPU operands (W including
//! decoded Q8 copies rather than original HFQ bytes, and the consumed-basis
//! input X) synchronously at every eager singleton launch boundary.
//!
//! Observer contract (frozen, owned by RuntimePolicy/RuntimeGemv):
//! `rdna_compute` `lab` `LabProjectionLaunch { symbol, family, form, weights,
//! x, rows, k, scale, hc_act, grid, block, shared_mem }`; the observer is
//! `Fn(&Gpu, &launch) -> HipResult<()>` installed with
//! `Gpu::set_lab_projection_observer`, whose previous observer is always
//! restored (also on error). Only `rows == 1` launches with a canonical form
//! (`single|x2|x4|hc_k4|hc_k8`) are selection candidates; any other event
//! (e.g. batched `narrow`/`narrow_x4` lane launches) is census-counted and
//! skipped, never normalized or silently dropped.
//!
//! CLI: `--pins pins.json --out OUT [--n-ctx 8192]`.
//! Pins schema: `{schema:"qwen4-rowtile-pins-v1",arch:"gfx1151|gfx1201|gfx1100",
//! model:{path,sha256},prompt:{path,sha256},sidecars:[{path,sha256}...]}`;
//! paths are pins-file-relative; model and weight sidecar SHA-256 pins require
//! prior stat-bound full SHA verification in the read-only runtime cache (never
//! recomputed here). Prompt bytes are hashed here. The actual chosen PLE-rows sidecar and the
//! applied trunk-overlay sidecar must both be pinned entries. Output `OUT`
//! must not exist. Genuine eager route is forced: `HIPFIRE_REPLAY_BACKEND=off`
//! is required (explicit), manual capture must be off, `HIPFIRE_GRAPH=1` /
//! `HIPFIRE_AR_GRAPH=1` are refused, and the live `graphs.capture_mode` /
//! `replay.is_recording()` flags are checked inside every callback.
//!
//! Per selected family/form/shape representative (first occurrence at row 0,
//! stable occurrence + full W-byte SHA required at rows 1..=63) this writes
//! `OUT/caseNNN/{w0.bin..w3.bin,x.bin,manifest.json,source-index.json}` plus
//! `meta.json` (MQ4), `hc.json` (BF16 activated `hc_k4` only), root
//! `index.json` (`qwen4-rowtile-index-v1`) and root `qwen4-capture.json`
//! (`q8_rowtile_capture.v1`, Q8 cases only). Missing families are reported,
//! never synthesized. No F32 capture family exists per census.
//!
//! Capture mode is `eager-liveprojection-operands-only` with
//! `retained_identity_claim:false`: operands only, no identity certification.
//! This producer performs no byte/state/ID/cost receipt role.

use hip_bridge::{HipError, HipResult};
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::Qwen4KvBackend;
use hipfire_loader::{load_qwen4_rank, open_qwen4_rank_source, Qwen4RankShared};
use hipfire_runtime::llama::argmax;
use hipfire_runtime::loader_api::{ModelSource, SpecLoadCfg};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::verified_cache::{StatKey, VerifiedCache};
use rdna_compute::gemv::WideTwinFamily;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
// `reference-parity` enables `rdna-compute/lab`, which provides
// `Gpu::set_lab_projection_observer` / `lab_kernel_device_identity`; the closure
// parameter types below are inferred from the setter, so the private
// `rdna_compute::dispatch` module path is never named here.
const ROWS: usize = 64;
const USAGE: &str = "usage: qwen4_rowtile_capture --pins pins.json --out OUT [--n-ctx 8192]";
const CAPTURE_MODE: &str = "eager-liveprojection-operands-only";
/// Actual mixed BF16/Q8 decode-head launch: the runtime emits TWO homogeneous
/// `x2` callback groups (BF16 first, then Q8) with this same symbol and grid
/// for ONE actual kernel launch (`qwen4_decode_head_fuse`).
const MIXED_PAIR_SYMBOL: &str = "gemv_bf16_xf32_x2_q8_0_k2560_staged_pair";

fn es<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

fn hip_err(context: &str) -> HipError {
    HipError::new(0, context)
}

fn sha_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha(sha: &str) -> bool {
    sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit())
}

// ---------------------------------------------------------------------------
// Pins
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Pin {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
struct Pins {
    schema: String,
    arch: String,
    model: Pin,
    prompt: Pin,
    sidecars: Vec<Pin>,
}

#[derive(Clone)]
struct ResolvedPin {
    /// Canonical absolute path used by the loader.
    canonical: PathBuf,
    sha256: String,
}

fn resolve_pin(parent: &Path, pin: &Pin, label: &str) -> Result<(PathBuf, ResolvedPin), String> {
    if pin.path.is_empty() || Path::new(&pin.path).is_absolute() || pin.path.contains("..") {
        return Err(format!("{label}: path must be nonempty and pins-relative"));
    }
    if !valid_sha(&pin.sha256) {
        return Err(format!("{label}: invalid SHA256 {:?}", pin.sha256));
    }
    let joined = parent.join(&pin.path);
    let canonical = joined
        .canonicalize()
        .map_err(|e| format!("{label} {}: {e}", joined.display()))?;
    Ok((
        joined,
        ResolvedPin {
            canonical,
            sha256: pin.sha256.to_ascii_lowercase(),
        },
    ))
}

/// Reuse a prior full SHA only for this exact canonical path and live stat.
/// No payload read, cache write, alias fallback, or streaming fallback.
fn verify_cached(
    cache: &VerifiedCache,
    joined: &Path,
    pin: &mut ResolvedPin,
    label: &str,
) -> Result<(), String> {
    if joined.canonicalize().map_err(es)? != pin.canonical {
        return Err(format!("{label}: canonical path drift"));
    }
    let metadata = std::fs::metadata(&pin.canonical)
        .map_err(|e| format!("{label} {}: {e}", pin.canonical.display()))?;
    let actual = cache
        .verified_digest(&pin.canonical, &StatKey::of(&metadata))
        .ok_or_else(|| format!(
            "{label} {}: missing prior stat-bound full SHA in runtime verified cache; refusing without payload rehash",
            pin.canonical.display()
        ))?;
    if actual != pin.sha256 {
        return Err(format!(
            "{label} {}: SHA256 mismatch actual={actual} pinned={}",
            pin.canonical.display(),
            pin.sha256
        ));
    }
    pin.sha256 = actual;
    Ok(())
}

fn verify_small(
    joined: &Path,
    canonical: &Path,
    sha: &str,
    label: &str,
) -> Result<Vec<u8>, String> {
    let bytes =
        std::fs::read(canonical).map_err(|e| format!("{label} {}: {e}", joined.display()))?;
    let actual = sha_hex(&bytes);
    if actual != sha {
        return Err(format!(
            "{label} {}: SHA256 mismatch actual={actual} pinned={sha}",
            joined.display()
        ));
    }
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// Loader (production serve load via `hipfire_loader` carriers, AR-only: no
// MTP head, no drafters, no expert-layer env override; no measured-identity
// requirement so embedded PLE stays eligible)
// ---------------------------------------------------------------------------

struct Model {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    vocab: usize,
    tokenizer: Tokenizer,
    arch: String,
    recipe: String,
    ple_rows_source: String,
    overlay_applied: bool,
    model_pin: ResolvedPin,
    prompt_pin: ResolvedPin,
    prompt_text: String,
}

fn recipe_name(recipe: &hipfire_arch_qwen4::weights::Qwen4TierRecipe) -> String {
    use hipfire_arch_qwen4::weights::Qwen4TierRecipe as R;
    match recipe {
        R::FlashNextPacked {
            trunk_tier,
            mtp_source,
        } => {
            format!("FlashNextPacked(trunk_tier={trunk_tier:?},mtp_source={mtp_source})")
        }
        R::FlashNextSlab {
            trunk_tier,
            mtp_source,
            ..
        } => {
            format!("FlashNextSlab(trunk_tier={trunk_tier:?},mtp_source={mtp_source})")
        }
        R::StockFp8 => "StockFp8".to_string(),
        R::SyntheticSourceBf16 => "SyntheticSourceBf16".to_string(),
        R::SyntheticPacked => "SyntheticPacked".to_string(),
    }
}

fn is_flash_next(recipe: &hipfire_arch_qwen4::weights::Qwen4TierRecipe) -> bool {
    use hipfire_arch_qwen4::weights::Qwen4TierRecipe as R;
    matches!(recipe, R::FlashNextPacked { .. } | R::FlashNextSlab { .. })
}

impl Model {
    fn load(
        expected_arch: &str,
        model: (PathBuf, ResolvedPin),
        prompt_bytes: Vec<u8>,
        prompt: (PathBuf, ResolvedPin),
        sidecars: &[(PathBuf, ResolvedPin)],
        n_ctx: usize,
    ) -> Result<Self, String> {
        let model_pin = &model.1;
        let _ = (&model.0, &prompt.0);
        let path: &Path = &model_pin.canonical;
        let model_str = path
            .to_str()
            .ok_or_else(|| format!("model path {} is not UTF-8", path.display()))?;
        let model_bytes = std::fs::metadata(path)
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len();
        // Daemon parity: real HFQ arch probe + pre-HIP KFD/ROCR masking before first HIP init.
        hipfire_loader::prepare_host_memory_for(path);
        let mut gpu = Gpu::init().map_err(es)?;
        if gpu.arch != expected_arch {
            return Err(format!(
                "pins arch {expected_arch:?} != actual GPU arch {:?}",
                gpu.arch
            ));
        }
        // Real context, AR-only: explicit spec OFF (no MTP head, no drafters),
        // backend/VMM/KV from the actual device and runtime config. Placement
        // (including any hot set) is owned by the carrier load below.
        let backend = Qwen4KvBackend::automatic(&gpu);
        let vmm_granularity = match backend {
            Qwen4KvBackend::Vmm => gpu
                .vmm_recommended_granularity()
                .map_err(|e| format!("qwen4: VMM granularity: {e}"))?,
            Qwen4KvBackend::Legacy => 1,
        };
        let shared = Qwen4RankShared {
            continuous_batch: false,
            path: model_str.to_string(),
            max_seq: n_ctx,
            sequence: None,
            backend,
            vmm_granularity,
            kv_mode: hipfire_runtime::config::get().kv_mode.clone(),
            state_quant: std::env::var("HIPFIRE_STATE_QUANT")
                .ok()
                .filter(|value| !value.is_empty()),
            spec: SpecLoadCfg {
                ngram_draft: Some(false),
                mtp: Some(false),
                dflash: Some(false),
                dspark: Some(false),
                ..SpecLoadCfg::default()
            },
            native_mtp: false,
            max_k: 0,
        };
        // Single owning producer: the carrier-owned source open already
        // resolves the PLE rows. Tokenizer, admission and the overlay
        // candidate read from THIS source, and the SAME source moves into the
        // production rank load (no second open, no cloned fulfillment).
        let source = open_qwen4_rank_source(&shared, &gpu.arch)?;
        let tokenizer = match &source {
            ModelSource::Hfq(hfq) => Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .map_err(|e| format!("tokenizer: {e}"))?,
            _ => return Err("model is not an HFQ artifact".into()),
        };
        // Pre-load overlay candidate + envelope marker (carrier 5396-5404):
        // the candidate is bound to the settled application marker only after
        // `attach_forward`. A candidate against a pre-existing marker is the
        // ambiguous shape (carrier 5429-5433) and fails before the load so no
        // GPU allocation is spent on an unclaimable overlay.
        let overlay_candidate = {
            let ModelSource::Hfq(hfq) = &source else {
                return Err("model source is not an HFQ artifact".into());
            };
            let admitted = hipfire_arch_qwen4::admit_hfqm_artifact_on(hfq, Some(&gpu.arch))
                .map_err(|e| format!("qwen4 admission: {e}"))?;
            // Refuse any profile that is not FlashNext (no F32 operand family
            // exists per census; synthetic/stock profiles have no capture schema).
            if !is_flash_next(&admitted.manifest.identity.recipe) {
                return Err(format!(
                    "refusing non-FlashNext recipe {}: no rowtile capture schema",
                    recipe_name(&admitted.manifest.identity.recipe)
                ));
            }
            let overlay =
                hipfire_arch_qwen4::resolve_trunk_overlay(hfq, path, &gpu.arch, &admitted.config)
                    .map_err(|e| format!("qwen4 trunk overlay: {e}"))?;
            let marker_was_set = admitted.config.trunk_iu4_default.is_some();
            if overlay.is_some() && marker_was_set {
                return Err("qwen4 trunk overlay: pre-existing trunk marker with an overlay candidate cannot prove this candidate was applied; refusing an ambiguous overlay claim".into());
            }
            overlay.map(|overlay| overlay.path().to_path_buf())
        };
        // The actual public producer: the production rank load consumes the
        // same source (admission, placement, fulfillment, assembly, forward
        // program, NPU identity, overlay attach).
        let loaded = load_qwen4_rank(&mut gpu, None, &shared, None, None, source)?;
        if loaded.max_seq < n_ctx {
            return Err(format!(
                "the loader admitted max_seq {}, below --n-ctx {n_ctx}",
                loaded.max_seq
            ));
        }
        let bundle = loaded.bundle;
        // Settled overlay condition: the pre-load candidate AND the settled
        // post-`attach_forward` marker. A mere candidate/path never claims
        // applied; only the settled applied sidecar must be pinned.
        let overlay_applied =
            overlay_candidate.is_some() && bundle.config.trunk_iu4_default.is_some();
        if overlay_applied {
            let candidate = overlay_candidate.as_ref().ok_or_else(|| {
                "qwen4 trunk overlay: settled marker with no candidate".to_string()
            })?;
            let canonical = candidate
                .canonicalize()
                .map_err(|e| format!("trunk overlay {}: {e}", candidate.display()))?;
            if !sidecars.iter().any(|(_, pin)| pin.canonical == canonical) {
                return Err(format!(
                    "applied trunk overlay {} is not a pinned sidecar",
                    canonical.display()
                ));
            }
        }
        // The actual applied PLE-rows sidecar must be a pinned entry; embedded
        // rows are covered by the model hash.
        let ple_rows_source = match bundle.ple_source() {
            Some(hipfire_arch_qwen4::Qwen4PleRows::Sidecar {
                path: rows_path, ..
            }) => {
                let canonical = rows_path
                    .canonicalize()
                    .map_err(|e| format!("PLE rows sidecar {}: {e}", rows_path.display()))?;
                if !sidecars.iter().any(|(_, pin)| pin.canonical == canonical) {
                    return Err(format!(
                        "PLE rows sidecar {} is not a pinned sidecar",
                        canonical.display()
                    ));
                }
                let bytes = std::fs::metadata(&canonical)
                    .map_err(|e| format!("stat {}: {e}", canonical.display()))?
                    .len();
                format!("{} ({bytes} bytes)", canonical.display())
            }
            Some(_) => "embedded model rows".to_string(),
            None => "absent (profile declares no PLE)".to_string(),
        };
        eprintln!("qwen4_rowtile_capture: ple rows: {ple_rows_source}");
        // The carrier-applied expert hot set, if any, must be a pinned entry.
        // The tier lease (`weights.expert_tiers()`), not a neighboring path,
        // is the applied evidence.
        if let Some(sha) = bundle
            .weights
            .expert_tiers()
            .and_then(|lease| lease.hotset_sidecar_sha256())
        {
            let want = sha.to_ascii_lowercase();
            if !sidecars.iter().any(|(_, pin)| pin.sha256 == want) {
                return Err(format!(
                    "applied expert hotset {sha} is not a pinned sidecar"
                ));
            }
        }
        let vocab = bundle.config.vocab_size;
        let recipe = recipe_name(&bundle.weights.manifest.identity.recipe);
        eprintln!(
            "qwen4_rowtile_capture: loaded {} ({:.2} GB) on {} (vocab {vocab}), trunk overlay {}",
            path.display(),
            model_bytes as f64 / 1e9,
            gpu.arch,
            if overlay_applied { "applied" } else { "absent" }
        );
        let prompt_text = String::from_utf8(prompt_bytes)
            .map_err(|_| "prompt file is not valid UTF-8".to_string())?;
        Ok(Self {
            arch: gpu.arch.clone(),
            gpu,
            bundle,
            vocab,
            tokenizer,
            recipe,
            ple_rows_source,
            overlay_applied,
            model_pin: model.1.clone(),
            prompt_pin: prompt.1.clone(),
            prompt_text,
        })
    }
}

// ---------------------------------------------------------------------------
// Live capture state
// ---------------------------------------------------------------------------

/// Stable representative identity: family + canonical form + full shape +
/// HC scale bits. The weight digest is tracked separately per occurrence and
/// must be byte-identical across all 64 rows (pointer alone is insufficient).
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SelKey {
    family: u8,
    form: String,
    ms: [usize; 4],
    k: usize,
    scale_bits: Option<u32>,
    hc_act: bool,
}

fn family_name(family: u8) -> &'static str {
    match family {
        0 => "mq4",
        1 => "mq6",
        2 => "bf16",
        3 => "q8",
        _ => "unknown",
    }
}

fn family_of(family: WideTwinFamily) -> Option<u8> {
    match family {
        WideTwinFamily::Mq4 => Some(0),
        WideTwinFamily::Mq6 => Some(1),
        WideTwinFamily::Bf16 => Some(2),
        WideTwinFamily::Q8 => Some(3),
        _ => None,
    }
}

fn canonical_form(form: &str) -> bool {
    matches!(form, "single" | "x2" | "x4" | "hc_k4" | "hc_k8")
}

/// Meaningful weight bytes per slot, or an error when K is outside the
/// family domain (never synthesize extents).
fn weight_len(family: u8, m: usize, k: usize) -> Result<usize, String> {
    if m == 0 {
        return Ok(0);
    }
    match family {
        0 => {
            if k == 0 || k % 256 != 0 {
                return Err(format!("mq4 K={k} is outside the K%256 domain"));
            }
            m.checked_mul(k / 256)
                .and_then(|n| n.checked_mul(136))
                .ok_or_else(|| "mq4 weight extent overflow".to_string())
        }
        1 => {
            if k == 0 || k % 256 != 0 {
                return Err(format!("mq6 K={k} is outside the K%256 domain"));
            }
            m.checked_mul(k / 256)
                .and_then(|n| n.checked_mul(200))
                .ok_or_else(|| "mq6 weight extent overflow".to_string())
        }
        2 => m
            .checked_mul(k)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| "bf16 weight extent overflow".to_string()),
        3 => {
            if k == 0 || k % 32 != 0 {
                return Err(format!("q8 K={k} is outside the K%32 domain"));
            }
            m.checked_mul(k / 32)
                .and_then(|n| n.checked_mul(34))
                .ok_or_else(|| "q8 weight extent overflow".to_string())
        }
        _ => Err("unknown family".to_string()),
    }
}

fn check_finite_x(x: &[u8], label: &str) -> Result<(), String> {
    if x.len() % 4 != 0 {
        return Err(format!(
            "{label}: X extent {} is not a multiple of 4",
            x.len()
        ));
    }
    for (i, word) in x.chunks_exact(4).enumerate() {
        let v = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        if !v.is_finite() {
            return Err(format!("{label}: non-finite X value at element {i}"));
        }
    }
    Ok(())
}

fn check_finite_w(family: u8, bytes: &[u8], m: usize, k: usize, label: &str) -> Result<(), String> {
    match family {
        2 => {
            if bytes.len() != m * k * 2 {
                return Err(format!("{label}: bf16 extent mismatch"));
            }
            for (i, h) in bytes.chunks_exact(2).enumerate() {
                if u16::from_le_bytes([h[0], h[1]]) & 0x7c00 == 0x7c00 {
                    return Err(format!("{label}: non-finite bf16 weight at element {i}"));
                }
            }
        }
        3 => {
            for (i, blk) in bytes.chunks_exact(34).enumerate() {
                if u16::from_le_bytes([blk[0], blk[1]]) & 0x7c00 == 0x7c00 {
                    return Err(format!("{label}: non-finite q8 scale in block {i}"));
                }
            }
        }
        0 | 1 => {
            let group = if family == 0 { 136 } else { 200 };
            if bytes.len() % group != 0 {
                return Err(format!(
                    "{label}: packed extent is not a whole number of groups"
                ));
            }
            for (g, grp) in bytes.chunks_exact(group).enumerate() {
                for (i, h) in grp[..8].chunks_exact(2).enumerate() {
                    if u16::from_le_bytes([h[0], h[1]]) & 0x7c00 == 0x7c00 {
                        return Err(format!(
                            "{label}: non-finite packed f16 header group={g} header={i}"
                        ));
                    }
                }
            }
        }
        _ => return Err(format!("{label}: unknown family")),
    }
    Ok(())
}

#[derive(Clone)]
struct RowRec {
    row: usize,
    position: usize,
    token: u32,
    occurrence: usize,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
}

#[derive(Clone)]
struct LiveCase {
    key: SelKey,
    symbol: String,
    occurrence: usize,
    module: String,
    object_sha256: String,
    container_sha256: String,
    w: [Vec<u8>; 4],
    w_sha: [String; 4],
    x_rows: Vec<u8>,
    rows_done: usize,
    captures: Vec<RowRec>,
    component_offset: usize,
    mixed: Option<serde_json::Value>,
}

struct PendingMixed {
    row: usize,
    ms: [usize; 2],
    sha: [String; 2],
}

struct State {
    row: usize,
    position: usize,
    token: u32,
    counters: HashMap<SelKey, usize>,
    order: Vec<SelKey>,
    cases: HashMap<SelKey, LiveCase>,
    pending_mixed_bf16: Option<PendingMixed>,
    skipped_launches: BTreeMap<String, usize>,
    events: usize,
    error: Option<String>,
}

impl State {
    fn begin_row(&mut self, row: usize, position: usize, token: u32) {
        self.row = row;
        self.position = position;
        self.token = token;
        self.counters.clear();
    }

    fn fail(&mut self, message: String) -> HipError {
        if self.error.is_none() {
            self.error = Some(message.clone());
        }
        hip_err(&message)
    }
}

/// Download `len` bytes from the base of `tensor` after a device sync.
/// `memcpy_dtoh` permits `dst.len() <= src.size`; the caller passes the exact
/// meaningful extent, so a larger allocation is truncated without any unsafe
/// buffer ownership or lifetime widening.
fn download_prefix(
    gpu: &Gpu,
    tensor: &GpuTensor,
    len: usize,
    label: &str,
) -> Result<Vec<u8>, HipError> {
    if tensor.buf.size() < len {
        return Err(hip_err(&format!(
            "{label}: device buffer {} bytes is smaller than meaningful extent {len}",
            tensor.buf.size()
        )));
    }
    gpu.hip.device_synchronize()?;
    let mut out = vec![0u8; len];
    gpu.hip.memcpy_dtoh(&mut out, &tensor.buf)?;
    Ok(out)
}

fn observe(
    state: &Arc<Mutex<State>>,
    gpu: &Gpu,
    symbol: &str,
    family: WideTwinFamily,
    form: &'static str,
    weights: [Option<(&GpuTensor, usize)>; 4],
    x: &GpuTensor,
    rows: usize,
    k: usize,
    scale: Option<f32>,
    hc_act: bool,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
) -> HipResult<()> {
    let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(message) = st.error.clone() {
        return Err(hip_err(&message));
    }
    // Genuine eager route only: any active graph capture or replay recording
    // fails the operands-only claim instead of appearing as valid capture.
    if gpu.graphs.capture_mode || gpu.replay.is_recording() {
        return Err(st.fail("capture during graph capture or replay recording".to_string()));
    }
    st.events += 1;
    let Some(family_id) = family_of(family) else {
        *st.skipped_launches
            .entry(format!("unknown-family/{form}/rows={rows}"))
            .or_insert(0) += 1;
        return Ok(());
    };
    if rows != 1 || !canonical_form(form) {
        // Batched lane/narrow launches are a different launch class, not a
        // singleton decode row: census-count, never normalize or drop silently.
        *st.skipped_launches
            .entry(format!("{}/{form}/rows={rows}", family_name(family_id)))
            .or_insert(0) += 1;
        return Ok(());
    }
    if scale.is_some() != hc_act {
        return Err(st.fail(format!(
            "{symbol}: hc_act={hc_act} contradicts scale presence {}",
            scale.is_some()
        )));
    }
    if let Some(s) = scale {
        if !s.is_finite() {
            return Err(st.fail(format!("{symbol}: non-finite HC scale")));
        }
    }
    let mut ms = [0usize; 4];
    for (i, slot) in weights.iter().enumerate() {
        if let Some(pair) = slot {
            ms[i] = pair.1;
        }
    }
    let key = SelKey {
        family: family_id,
        form: form.to_string(),
        ms,
        k,
        scale_bits: scale.map(f32::to_bits),
        hc_act,
    };
    let row = st.row;
    let position = st.position;
    let token = st.token;
    // Actual admitted loaded owner, container and extracted per-architecture ELF.
    // This observer runs before launch; this is not invocation history.
    let (module, container_sha256, object_sha256) = match gpu.lab_kernel_device_identity(symbol) {
        Ok(identity) => identity,
        Err(e) => {
            let message = format!("{symbol}: no loaded kernel identity ({e})");
            return Err(st.fail(message));
        }
    };
    // Per-step occurrence counter for this exact symbol+shape, so the two
    // homogeneous groups of one mixed launch never confuse each other.
    let occurrence = st.counters.get(&key).copied().unwrap_or(0);
    st.counters.insert(key.clone(), occurrence + 1);
    // Download immutable operands synchronously. The observer fires after the
    // kernel is ensured/loaded and before launch, so these bytes are the exact
    // inputs the launch consumes.
    let mut w: [Vec<u8>; 4] = Default::default();
    for (i, slot) in weights.iter().enumerate() {
        let tensor = slot.map(|pair| pair.0);
        let len = match weight_len(family_id, ms[i], k) {
            Ok(len) => len,
            Err(e) => return Err(st.fail(format!("{symbol}: {e}"))),
        };
        if ms[i] == 0 {
            continue;
        }
        let tensor = tensor
            .ok_or_else(|| st.fail(format!("{symbol}: slot {i} has M={} but no tensor", ms[i])))?;
        let bytes = match download_prefix(gpu, tensor, len, symbol) {
            Ok(bytes) => bytes,
            Err(e) => return Err(st.fail(format!("{symbol} slot {i}: {e}"))),
        };
        if let Err(e) = check_finite_w(family_id, &bytes, ms[i], k, symbol) {
            return Err(st.fail(e));
        }
        w[i] = bytes;
    }
    let x_len = k
        .checked_mul(4)
        .ok_or_else(|| st.fail(format!("{symbol}: X extent overflow")))?;
    let x_row = match download_prefix(gpu, x, x_len, symbol) {
        Ok(bytes) => bytes,
        Err(e) => return Err(st.fail(format!("{symbol} X: {e}"))),
    };
    if let Err(e) = check_finite_x(&x_row, symbol) {
        return Err(st.fail(e));
    }
    let mut w_sha: [String; 4] = Default::default();
    for i in 0..4 {
        w_sha[i] = sha_hex(&w[i]);
    }
    let rec = RowRec {
        row,
        position,
        token,
        occurrence,
        grid,
        block,
        shared_mem,
    };
    let is_mixed = symbol == MIXED_PAIR_SYMBOL;
    if row == 0 {
        if st.cases.contains_key(&key) {
            // A second distinct occurrence of an already-selected signature in
            // row 0 (e.g. per-expert launches sharing one symbol+shape): only
            // the first occurrence is the representative; later ones are not
            // consolidated.
            return Ok(());
        }
        let (component_offset, mixed) = if is_mixed {
            if family_id == 2 {
                st.pending_mixed_bf16 = Some(PendingMixed {
                    row,
                    ms: [ms[0], ms[1]],
                    sha: [w_sha[0].clone(), w_sha[1].clone()],
                });
                (0usize, None)
            } else {
                match st.pending_mixed_bf16.take() {
                    Some(pending) if pending.row == row => (
                        2usize,
                        // Canonical frozen `mixed` object: the BF16 subgroup's
                        // own Ms[2] live under `bf16_ms`; the Q8 subgroup keeps
                        // its own actual Ms in ms/weights slots [MQ,MQ,0,0].
                        Some(json!({
                            "symbol": MIXED_PAIR_SYMBOL,
                            "bf16_ms": [pending.ms[0], pending.ms[1]],
                            "bf16_weight_sha256": [pending.sha[0].clone(), pending.sha[1].clone()],
                        })),
                    ),
                    _ => {
                        return Err(st.fail(format!(
                            "{symbol}: Q8 subgroup without the paired adjacent BF16 subgroup at row {row}"
                        )));
                    }
                }
            }
        } else {
            (0usize, None)
        };
        let mut x_rows = Vec::with_capacity(ROWS * x_len);
        x_rows.extend_from_slice(&x_row);
        st.order.push(key.clone());
        st.cases.insert(
            key.clone(),
            LiveCase {
                key: key.clone(),
                symbol: symbol.to_string(),
                occurrence,
                module,
                object_sha256,
                container_sha256,
                w,
                w_sha,
                x_rows,
                rows_done: 1,
                captures: vec![rec],
                component_offset,
                mixed,
            },
        );
        return Ok(());
    }
    // Rows 1..=63: the selected occurrence must recur with identical symbol,
    // module, object SHA and W bytes; a brand-new signature the row-0 sweep
    // missed fails closed instead of silently extending the capture.
    // Expectations are snapshotted (owned) first: `st.fail` needs `&mut st`,
    // so no `&mut` case borrow may stay live across any comparison below.
    let (
        exp_symbol,
        exp_occurrence,
        exp_rows_done,
        exp_module,
        exp_object,
        exp_container,
        exp_w_sha,
        exp_mixed,
    ) = match st.cases.get(&key) {
        Some(case) => (
            case.symbol.clone(),
            case.occurrence,
            case.rows_done,
            case.module.clone(),
            case.object_sha256.clone(),
            case.container_sha256.clone(),
            case.w_sha.clone(),
            case.mixed.clone(),
        ),
        None => {
            return Err(st.fail(format!(
                "{symbol} {} K={k} ms={ms:?}: new signature at row {row} missed at row 0",
                family_name(family_id)
            )));
        }
    };
    if occurrence != exp_occurrence {
        // A non-representative occurrence of a selected signature: not data.
        return Ok(());
    }
    if exp_rows_done != row {
        return Err(st.fail(format!(
            "{exp_symbol}: selected occurrence {exp_occurrence} missing at row {row}"
        )));
    }
    if symbol != exp_symbol
        || module != exp_module
        || object_sha256 != exp_object
        || container_sha256 != exp_container
    {
        return Err(st.fail(format!(
            "{exp_symbol}: singleton identity changed at row {row} (was {exp_module}/{exp_object})"
        )));
    }
    for i in 0..4 {
        if w_sha[i] != exp_w_sha[i] {
            return Err(st.fail(format!(
                "{exp_symbol}: immutable W slot {i} changed at row {row}"
            )));
        }
    }
    if is_mixed {
        if family_id == 2 {
            st.pending_mixed_bf16 = Some(PendingMixed {
                row,
                ms: [ms[0], ms[1]],
                sha: [w_sha[0].clone(), w_sha[1].clone()],
            });
        } else {
            match st.pending_mixed_bf16.take() {
                Some(pending) if pending.row == row => {
                    let expected = exp_mixed.as_ref().ok_or_else(|| {
                        st.fail(format!("{symbol}: Q8 case lost its mixed record"))
                    })?;
                    let want = json!({
                        "symbol": MIXED_PAIR_SYMBOL,
                        "bf16_ms": [pending.ms[0], pending.ms[1]],
                        "bf16_weight_sha256": [pending.sha[0].clone(), pending.sha[1].clone()],
                    });
                    if *expected != want {
                        return Err(st.fail(format!(
                            "{symbol}: mixed BF16 counterpart changed at row {row}"
                        )));
                    }
                }
                _ => {
                    return Err(st.fail(format!(
                        "{symbol}: Q8 subgroup without the paired adjacent BF16 subgroup at row {row}"
                    )));
                }
            }
        }
    }
    if !st.cases.contains_key(&key) {
        return Err(st.fail(format!("{exp_symbol}: selected case vanished at row {row}")));
    }
    let case = st
        .cases
        .get_mut(&key)
        .ok_or_else(|| hip_err("selected case vanished"))?;
    case.x_rows.extend_from_slice(&x_row);
    case.captures.push(rec);
    case.rows_done += 1;
    Ok(())
}

// ---------------------------------------------------------------------------
// Finalization
// ---------------------------------------------------------------------------

struct CaseFiles {
    name: String,
    dir: String,
    manifest_sha: String,
    source_sha: String,
}

fn case_name(family: u8, form: &str, k: usize, ms: [usize; 4]) -> String {
    let mut name = format!("{}_{form}_k{k}", family_name(family));
    for m in ms.iter().filter(|&&m| m > 0) {
        name.push_str(&format!("_m{m}"));
    }
    name
}

fn write_case(
    out: &Path,
    index: usize,
    case: &LiveCase,
    arch: &str,
    model: &ResolvedPin,
    prompt: &ResolvedPin,
    recipe: &str,
    ple_rows: &str,
    overlay_applied: bool,
    n_ctx: usize,
    prompt_len: usize,
) -> Result<CaseFiles, String> {
    let dir = format!("case{index:03}");
    let dir_path = out.join(&dir);
    std::fs::create_dir(&dir_path).map_err(|e| format!("mkdir {}: {e}", dir_path.display()))?;
    let name = case_name(case.key.family, &case.key.form, case.key.k, case.key.ms);
    let mut weights_json = Vec::with_capacity(4);
    for i in 0..4 {
        let file = format!("w{i}.bin");
        std::fs::write(dir_path.join(&file), &case.w[i])
            .map_err(|e| format!("write {dir}/{file}: {e}"))?;
        weights_json.push(json!({"path": file, "sha256": case.w_sha[i]}));
    }
    let x_sha = sha_hex(&case.x_rows);
    std::fs::write(dir_path.join("x.bin"), &case.x_rows)
        .map_err(|e| format!("write {dir}/x.bin: {e}"))?;
    let scale_json = match case.key.scale_bits {
        Some(bits) => json!(f32::from_bits(bits)),
        None => json!(null),
    };
    let bindings: Vec<serde_json::Value> = (0..4)
        .map(|i| json!({"name": "", "layer": null, "byte_offset": 0, "bytes": case.w[i].len()}))
        .collect();
    let captures: Vec<serde_json::Value> = case
        .captures
        .iter()
        .map(|c| {
            json!({
                "row": c.row, "position": c.position, "token": c.token,
                "occurrence": c.occurrence, "grid": c.grid, "block": c.block,
                "shared_mem": c.shared_mem,
            })
        })
        .collect();
    // Per-case source object, direct (NOT nested): the full original mixed
    // component Ms/types/offsets and the actual caller symbol/object SHA are
    // preserved here; homogeneous groups make no DAG equivalence claim.
    let source = json!({
        "schema": "qwen4-rowtile-source-v1",
        "name": name,
        "family": family_name(case.key.family),
        "form": case.key.form,
        "ms": case.key.ms,
        "k": case.key.k,
        "rows": ROWS,
        "weights": weights_json,
        "x": {"path": "x.bin", "sha256": x_sha},
        "singleton": {
            "symbol": case.symbol,
            "module": case.module,
            "object_sha256": case.object_sha256,
            "container_sha256": case.container_sha256,
        },
        "weight_bindings": bindings,
        "captures": captures,
        "model": {"path": model.canonical.display().to_string(), "sha256": model.sha256},
        "prompt": {"path": prompt.canonical.display().to_string(), "sha256": prompt.sha256},
        "capture_mode": CAPTURE_MODE,
        "retained_identity_claim": false,
        "hc_act": case.key.hc_act,
        "scale": scale_json,
        "component_offset": case.component_offset,
        "mixed": match &case.mixed {
            Some(m) => m.clone(),
            None => json!(null),
        },
        "recipe": recipe,
        "ple_rows": ple_rows,
        "overlay_applied": overlay_applied,
        "n_ctx": n_ctx,
        "prompt_len": prompt_len,
    });
    let source_text = serde_json::to_string_pretty(&source).map_err(es)?;
    let source_sha = sha_hex(source_text.as_bytes());
    std::fs::write(dir_path.join("source-index.json"), &source_text)
        .map_err(|e| format!("write {dir}/source-index.json: {e}"))?;
    let manifest = json!({
        "schema": "qwen4-rowtile-operands-v1",
        "family": family_name(case.key.family),
        "name": name,
        "arch": arch,
        "form": case.key.form,
        "ms": case.key.ms,
        "k": case.key.k,
        "rows": ROWS,
        "weights": weights_json,
        "x": {"path": "x.bin", "sha256": x_sha},
        "scale": scale_json,
        "hc_act": case.key.hc_act,
        "provenance": {"source": "source-index.json", "source_sha256": source_sha},
    });
    let manifest_text = serde_json::to_string_pretty(&manifest).map_err(es)?;
    let manifest_sha = sha_hex(manifest_text.as_bytes());
    std::fs::write(dir_path.join("manifest.json"), &manifest_text)
        .map_err(|e| format!("write {dir}/manifest.json: {e}"))?;
    // MQ4 legacy sidecar: unchanged `mq4_rows_oracle.capture.v1` schema for
    // the first nonzero slot sharing the same X.
    if case.key.family == 0 {
        let slot = case
            .key
            .ms
            .iter()
            .position(|&m| m > 0)
            .ok_or_else(|| format!("{name}: mq4 case has no nonzero slot"))?;
        let meta = json!({
            "schema": "mq4_rows_oracle.capture.v1",
            "arch": arch,
            "M": case.key.ms[slot],
            "K": case.key.k,
            "rows": ROWS,
            "weights": {"path": format!("w{slot}.bin"), "sha256": case.w_sha[slot]},
            "activations": {"path": "x.bin", "sha256": x_sha},
            "provenance": {"source": "source-index.json", "source_sha256": source_sha},
        });
        std::fs::write(
            dir_path.join("meta.json"),
            serde_json::to_string_pretty(&meta).map_err(es)?,
        )
        .map_err(|e| format!("write {dir}/meta.json: {e}"))?;
    }
    // BF16 activated legacy sidecar: strict legacy schema, ONLY hc_k4 with a
    // real finite activated scale. Raw HC stays on the common manifest.
    if case.key.family == 2 && case.key.form == "hc_k4" && case.key.hc_act {
        let scale = match case.key.scale_bits {
            Some(bits) => f32::from_bits(bits),
            None => return Err(format!("{name}: activated hc without a scale")),
        };
        if !scale.is_finite() {
            return Err(format!("{name}: non-finite activated hc scale"));
        }
        let hc = json!({
            "arch": arch,
            "source": format!("source-index.json sha256:{source_sha}"),
            "m": case.key.ms[0],
            "k": case.key.k,
            "scale": scale,
            "weights": "w0.bin",
            "weights_sha256": case.w_sha[0],
            "x": "x.bin",
            "x_sha256": x_sha,
        });
        std::fs::write(
            dir_path.join("hc.json"),
            serde_json::to_string_pretty(&hc).map_err(es)?,
        )
        .map_err(|e| format!("write {dir}/hc.json: {e}"))?;
    }
    Ok(CaseFiles {
        name,
        dir,
        manifest_sha,
        source_sha,
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    if let Err(e) = run() {
        eprintln!("qwen4_rowtile_capture: error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut pins_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut n_ctx: usize = 8192;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--pins") => {
                pins_path = Some(PathBuf::from(args.next().ok_or("--pins requires a path")?));
            }
            Some("--out") => {
                out_path = Some(PathBuf::from(args.next().ok_or("--out requires a path")?));
            }
            Some("--n-ctx") => {
                let value = args.next().ok_or("--n-ctx requires a value")?;
                n_ctx = value
                    .to_str()
                    .and_then(|v| v.parse().ok())
                    .filter(|&v| v > 0)
                    .ok_or("--n-ctx requires a positive integer")?;
            }
            _ => return Err(format!("{USAGE}")),
        }
    }
    let pins_path = pins_path.ok_or(USAGE)?;
    let out_path = out_path.ok_or(USAGE)?;

    // Explicit eager HIP diagnostic mode: `HIPFIRE_REPLAY_BACKEND=off` is the
    // source-mapped eager spelling; anything else (including unset) is refused
    // rather than assumed.
    match hipfire_config::process_value("HIPFIRE_REPLAY_BACKEND").as_deref() {
        Some("off") => {}
        other => {
            return Err(format!(
                "refusing capture: HIPFIRE_REPLAY_BACKEND must be explicitly \"off\" for eager operands, got {other:?}"
            ));
        }
    }
    if hipfire_config::process_value("HIPFIRE_REPLAY_MANUAL_CAPTURE")
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "on"))
    {
        return Err("refusing capture: HIPFIRE_REPLAY_MANUAL_CAPTURE is enabled".into());
    }
    for var in ["HIPFIRE_GRAPH", "HIPFIRE_AR_GRAPH"] {
        if hipfire_config::process_value(var).as_deref() == Some("1") {
            return Err(format!("refusing capture: {var}=1 enables graph capture"));
        }
    }

    let pins_bytes =
        std::fs::read(&pins_path).map_err(|e| format!("read {}: {e}", pins_path.display()))?;
    let pins: Pins = serde_json::from_slice(&pins_bytes).map_err(|e| format!("pins: {e}"))?;
    if pins.schema != "qwen4-rowtile-pins-v1" {
        return Err(format!(
            "pins schema {:?} != qwen4-rowtile-pins-v1",
            pins.schema
        ));
    }
    if !matches!(pins.arch.as_str(), "gfx1151" | "gfx1201" | "gfx1100") {
        return Err(format!(
            "pins arch {:?} is outside gfx1151|gfx1201|gfx1100",
            pins.arch
        ));
    }
    let pins_parent = pins_path.parent().unwrap_or_else(|| Path::new("."));
    let mut model = resolve_pin(pins_parent, &pins.model, "model")?;
    let prompt = resolve_pin(pins_parent, &pins.prompt, "prompt")?;
    let mut sidecars = Vec::with_capacity(pins.sidecars.len());
    for (i, pin) in pins.sidecars.iter().enumerate() {
        sidecars.push(resolve_pin(pins_parent, pin, &format!("sidecars[{i}]"))?);
    }
    let home = hipfire_config::hipfire_home()
        .map_err(|e| format!("runtime verified cache home unavailable: {e}"))?;
    let cache = VerifiedCache::in_home(&home);
    verify_cached(&cache, &model.0, &mut model.1, "model")?;
    let prompt_bytes = verify_small(&prompt.0, &prompt.1.canonical, &prompt.1.sha256, "prompt")?;
    for (i, (joined, pin)) in sidecars.iter_mut().enumerate() {
        verify_cached(&cache, joined, pin, &format!("sidecars[{i}]"))?;
    }
    if std::fs::symlink_metadata(&out_path).is_ok() {
        return Err(format!("refusing capture: {} exists", out_path.display()));
    }

    let mut m = Model::load(&pins.arch, model, prompt_bytes, prompt, &sidecars, n_ctx)?;
    let prompt_ids = m.tokenizer.encode(&m.prompt_text);
    if prompt_ids.is_empty() {
        return Err("prompt encodes to zero tokens".into());
    }
    let prompt_len = prompt_ids.len();
    if prompt_len
        .checked_add(ROWS)
        .map(|n| n > n_ctx)
        .unwrap_or(true)
    {
        return Err(format!(
            "prompt len {prompt_len} + {ROWS} exceeds n_ctx {n_ctx}"
        ));
    }
    eprintln!(
        "qwen4_rowtile_capture: arch={} prompt_len={prompt_len} n_ctx={n_ctx}",
        m.arch
    );

    // Logits buffers allocated before the observer is armed.
    let prompt_logits = m.gpu.zeros(&[m.vocab], DType::F32).map_err(es)?;
    let row_logits = m.gpu.zeros(&[m.vocab], DType::F32).map_err(es)?;

    // Canonical prefill; the observer is armed ONLY after it (no synthetic
    // warm tokens pollute the occurrence census).
    m.bundle
        .prefill_final(&mut m.gpu, &prompt_ids, 0, &prompt_logits)
        .map_err(|e| format!("prefill_final: {e}"))?;
    let seed = m.gpu.download_f32(&prompt_logits).map_err(es)?;
    if seed.iter().any(|v| !v.is_finite()) {
        return Err("non-finite prefill logits".into());
    }
    let mut token = argmax(&seed);
    eprintln!("qwen4_rowtile_capture: prefill done, seed token={token}");

    let state = Arc::new(Mutex::new(State {
        row: 0,
        position: prompt_len,
        token,
        counters: HashMap::new(),
        order: Vec::new(),
        cases: HashMap::new(),
        pending_mixed_bf16: None,
        skipped_launches: BTreeMap::new(),
        events: 0,
        error: None,
    }));
    let observer_state = Arc::clone(&state);
    // The closure borrows nothing: all mutation flows through the mutex, so no
    // foreign buffer ownership or lifetime widening is involved. The launch
    // parameter names the neutral root `rdna_compute::LabProjectionLaunch`
    // type directly (re-exported alongside `LabKernelFaultReceipt`).
    let observer = Arc::new(
        move |gpu: &Gpu, launch: &rdna_compute::LabProjectionLaunch<'_>| -> HipResult<()> {
            let (symbol, family, form, w0, w1, w2, w3, x) = (
                launch.symbol,
                launch.family,
                launch.form,
                launch.weights[0],
                launch.weights[1],
                launch.weights[2],
                launch.weights[3],
                launch.x,
            );
            let (rows, k, scale, hc_act, grid, block, shared_mem) = (
                launch.rows,
                launch.k,
                launch.scale,
                launch.hc_act,
                launch.grid,
                launch.block,
                launch.shared_mem,
            );
            observe(
                &observer_state,
                gpu,
                symbol,
                family,
                form,
                [w0, w1, w2, w3],
                x,
                rows,
                k,
                scale,
                hc_act,
                grid,
                block,
                shared_mem,
            )
        },
    );

    // Install the observer, keeping the previous one for unconditional restore
    // (its type is inferred, never named; no other file is touched).
    let old = m.gpu.set_lab_projection_observer(Some(observer));
    // The capture loop runs in a closure so every error path below still
    // restores the previous observer before returning.
    let capture_result = (|| -> Result<(Vec<u32>, Vec<usize>), String> {
        let mut tokens = Vec::with_capacity(ROWS);
        let mut positions = Vec::with_capacity(ROWS);
        for r in 0..ROWS {
            let position = prompt_len + r;
            {
                let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(message) = st.error.clone() {
                    return Err(message);
                }
                st.begin_row(r, position, token);
            }
            m.bundle
                .forward_token(&mut m.gpu, token, &row_logits, None)
                .map_err(|e| {
                    let suffix = state
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .error
                        .clone()
                        .map(|message| format!(" ({message})"))
                        .unwrap_or_default();
                    format!("forward_token row {r}{suffix}: {e}")
                })?;
            let logits = m.gpu.download_f32(&row_logits).map_err(es)?;
            if logits.iter().any(|v| !v.is_finite()) {
                return Err(format!("non-finite logits at row {r}"));
            }
            // Every selected occurrence must have recurred exactly once per
            // row; a missing one fails here, not as a short capture.
            {
                let st = state.lock().unwrap_or_else(|e| e.into_inner());
                for key in &st.order {
                    let case = st
                        .cases
                        .get(key)
                        .ok_or_else(|| "selected case missing".to_string())?;
                    if case.rows_done != r + 1 {
                        return Err(format!(
                            "{}: selected occurrence {} missing at row {r}",
                            case.symbol, case.occurrence
                        ));
                    }
                }
            }
            tokens.push(token);
            positions.push(position);
            token = argmax(&logits);
            if r % 16 == 15 {
                eprintln!(
                    "qwen4_rowtile_capture: row {}/{} token={token}",
                    r + 1,
                    ROWS
                );
            }
        }
        Ok((tokens, positions))
    })();
    m.gpu.set_lab_projection_observer(old);
    let (tokens, positions) = capture_result?;

    let st = state.lock().unwrap_or_else(|e| e.into_inner());
    if st.order.is_empty() {
        return Err("no canonical singleton projection events observed in 64 decode steps".into());
    }
    eprintln!(
        "qwen4_rowtile_capture: {} cases, {} events, skipped={:?}",
        st.order.len(),
        st.events,
        st.skipped_launches
    );
    std::fs::create_dir(&out_path).map_err(|e| format!("mkdir {}: {e}", out_path.display()))?;
    let mut index_cases = Vec::with_capacity(st.order.len());
    let mut q8_cases = Vec::new();
    for (index, key) in st.order.iter().enumerate() {
        // The BF16 subgroup of a mixed launch finalizes before the Q8 one
        // (first-occurrence order): backfill its `mixed` object from the Q8
        // subgroup's frozen record so both directions carry it.
        let mut case = st
            .cases
            .get(key)
            .ok_or_else(|| "selected case missing".to_string())?
            .clone();
        if case.symbol == MIXED_PAIR_SYMBOL && case.key.family == 2 && case.mixed.is_none() {
            let q8 =
                st.order.iter().filter_map(|k| st.cases.get(k)).find(|c| {
                    c.symbol == MIXED_PAIR_SYMBOL && c.key.family == 3 && c.mixed.is_some()
                });
            if let Some(q8) = q8 {
                case.mixed = q8.mixed.clone();
            }
        }
        if case.rows_done != ROWS || case.captures.len() != ROWS {
            return Err(format!(
                "{}: short capture {}/{}",
                case.symbol, case.rows_done, ROWS
            ));
        }
        let files = write_case(
            &out_path,
            index,
            &case,
            &m.arch,
            &m.model_pin,
            &m.prompt_pin,
            &m.recipe,
            &m.ple_rows_source,
            m.overlay_applied,
            n_ctx,
            prompt_len,
        )?;
        eprintln!(
            "qwen4_rowtile_capture: {} family={} form={} k={} ms={:?} {}",
            files.dir,
            family_name(case.key.family),
            case.key.form,
            case.key.k,
            case.key.ms,
            case.symbol
        );
        if case.key.family == 3 {
            let kind = match case.key.form.as_str() {
                "single" => "single",
                "x2" => "x2",
                "hc_k8" => "k8",
                other => {
                    return Err(format!(
                        "{}: Q8 form {other:?} has no q8_rowtile_capture.v1 kind",
                        files.name
                    ));
                }
            };
            let mut ms: Vec<usize> = case.key.ms.iter().copied().filter(|&m| m > 0).collect();
            if kind == "x2" && ms.len() != 2 {
                return Err(format!(
                    "{}: Q8 x2 case needs 2 nonzero Ms, got {ms:?}",
                    files.name
                ));
            }
            if kind != "x2" && ms.len() != 1 {
                ms.truncate(1);
            }
            let mut weights = Vec::new();
            for (i, m) in case.key.ms.iter().enumerate() {
                if *m > 0 {
                    weights.push(json!({
                        "path": format!("{}/w{i}.bin", files.dir),
                        "sha256": case.w_sha[i],
                    }));
                }
            }
            q8_cases.push(json!({
                "name": files.name,
                "kind": kind,
                "k": case.key.k,
                "ms": ms,
                "scale": match case.key.scale_bits {
                    Some(bits) => json!(f32::from_bits(bits)),
                    None => json!(null),
                },
                "weights": weights,
                "x": {
                    "path": format!("{}/x.bin", files.dir),
                    "sha256": sha_hex(&case.x_rows),
                },
                "singleton": {
                    "module": case.module,
                    "symbol": case.symbol,
                    "object_sha256": case.object_sha256,
                    "container_sha256": case.container_sha256,
                },
            }));
        }
        index_cases.push(json!({
            "name": files.name,
            "family": family_name(case.key.family),
            "form": case.key.form,
            "ms": case.key.ms,
            "k": case.key.k,
            "manifest": {"path": format!("{}/manifest.json", files.dir), "sha256": files.manifest_sha},
            "source": {"path": format!("{}/source-index.json", files.dir), "sha256": files.source_sha},
        }));
    }
    let missing_families: Vec<&str> = ["mq4", "mq6", "q8", "bf16"]
        .into_iter()
        .filter(|f| !st.order.iter().any(|k| family_name(k.family) == *f))
        .collect();
    if !missing_families.is_empty() {
        eprintln!("qwen4_rowtile_capture: missing families (reported, not synthesized): {missing_families:?}");
    }
    let index = json!({
        "schema": "qwen4-rowtile-index-v1",
        "arch": m.arch,
        "model": {"path": m.model_pin.canonical.display().to_string(), "sha256": m.model_pin.sha256},
        "payload_verification": {"authority": "runtime verified cache; prior stat-bound full SHA", "recomputed": false},
        "prompt": {"path": m.prompt_pin.canonical.display().to_string(), "sha256": m.prompt_pin.sha256},
        "rows": ROWS,
        "positions": positions,
        "tokens": tokens,
        "cases": index_cases,
        "missing_families": missing_families,
        "skipped_launches": st.skipped_launches,
        "capture_mode": CAPTURE_MODE,
        "retained_identity_claim": false,
    });
    let index_text = serde_json::to_string_pretty(&index).map_err(es)?;
    let index_sha = sha_hex(index_text.as_bytes());
    std::fs::write(out_path.join("index.json"), &index_text)
        .map_err(|e| format!("write index.json: {e}"))?;
    if !q8_cases.is_empty() {
        let q8 = json!({
            "schema": "q8_rowtile_capture.v1",
            "arch": m.arch,
            "source_index": {"path": "index.json", "sha256": index_sha},
            "cases": q8_cases,
        });
        std::fs::write(
            out_path.join("q8-capture.json"),
            serde_json::to_string_pretty(&q8).map_err(es)?,
        )
        .map_err(|e| format!("write q8-capture.json: {e}"))?;
    }
    eprintln!(
        "qwen4_rowtile_capture: wrote {} cases to {}",
        st.order.len(),
        out_path.display()
    );
    Ok(())
}
