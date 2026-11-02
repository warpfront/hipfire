// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_g2_replay_gate — real-model retained-body replay gate for an
//! expert-slab (ctl3-hot) Qwen4 HFQM artifact.
//!
//! One process, one real model, three phases; every check prints a JSON row
//! with an exact mismatch count (no tolerance) and an error string.
//!
//! 1. **Capture.** The production lifecycle (`ReplayController::
//!    configure_model_default(true)`, then the first eligible single-token
//!    forward) records the decode body, closes the capture window, prepares the
//!    retained plan and latches the route identities. The gate asserts the
//!    window opened (`Armed` before), closed (`Ready` after), recorded launches,
//!    latched identities and held no device copy, memset or readback.
//! 2. **Positive.** From the same primed state, ordinary HIP (`ShadowBodyRoute::
//!    Hip`) is the baseline; the production route (`Ready` → replay), the
//!    prepared plan (`ShadowBodyRoute::Plan`) and the recorded-HIP oracle
//!    (`ShadowBodyRoute::HipOracle`) must each reproduce it byte for byte (logits,
//!    GDN recurrent/conv, QSA K/V/index keys/partial/selected, PLE conv and hyper
//!    feedback, lengths, position) at `--iterations` consecutive positions. Every
//!    arm also proves it ran: plan arms by the controller's replay observation
//!    (`count == iterations`, first/last position exact) and by launching fewer
//!    HIP kernels than the recorded body; the oracle by launching at least the
//!    recorded body.
//! 3. **Changed binding.** The bundle (weights, expert tables, forward scratch) is
//!    freed and rebuilt on the same `Gpu` between capture and replay, so every
//!    expert table id and pointer mapping the tape latched is gone. The prepared
//!    plan is still `Ready`. A forward on the rebuilt bundle through the
//!    production route, the plan and the oracle must each be refused by
//!    `ReplayController::validate_route_identities` (error `Qwen4 retained replay
//!    refused: retained route identity ...`) before any retained submit: no
//!    replay observed, no recorded body launched. The refusal poisons the route
//!    (`retained route binding changed`) and the next forward runs on HIP.
//!
//! Route admission. The gate is required only where the product itself takes
//! the retained Redline route by default: after the load, the loaded model's
//! facts (GPU arch, `qwen4`, file extension, single-GPU mesh, no drafter
//! installed, expert parallelism, platform refusal) are passed to
//! `hipfire_runtime::config::retained_redline_default`, the daemon's own
//! admission (`.mq4r` on gfx1100/gfx1151/gfx1201, or Qwen4 on exact gfx1201
//! with no drafter). Where the product does not take that route the gate
//! does not force it: it frees the bundle, runs no capture, positive or
//! negative phase, and writes a successful structured SKIP report
//! (`"status": "skip"`, `"ok": true`, no checks, reason
//! `Redline not on the default route (config.rs:15–22)`). A SKIP is never a
//! PASS: it carries no check rows, and no result on a forced route is evidence
//! about the default one.
//!
//! On a default route, a check that cannot be evaluated is a failing check
//! with its error, never a skipped one. Exit status is 0 only when every check
//! passes (or the route is not the default and the report is a SKIP).
//!
//! Usage (gfx1201, discrete GPU; the host-mapped layer split is the same one
//! `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` selects for the other harnesses):
//!   qwen4_g2_replay_gate --model ctl3-hot.hfq [--output gate.json|-]
//!       [--expert-vram-layers 20|all] [--context 128] [--iterations 5]
//!       [--max-seq 1024]
//!
//! The replay backend must not be pinned by the environment
//! (`HIPFIRE_REPLAY_BACKEND` other than `auto`, `HIPFIRE_REPLAY_MANUAL_CAPTURE`):
//! an explicit setting is a failing `replay_default_configured` check.
//! `HIPFIRE_REPLAY_TRANSPORT=aql` runs the AQL twin of the same gate.

use hipfire_arch_qwen4::admit_hfqm_artifact_on;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource as _, SourcePayload};
use hipfire_runtime::ple_sidecar::attach_validation_ple_rows;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::replay::{ReplayObservation, ReplayState, ShadowBodyRoute};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: qwen4_g2_replay_gate --model CTL3-HOT.hfq [--output gate.json|-] [--expert-vram-layers N|all] [--context 128] [--iterations 5] [--max-seq 1024]";

const DEFAULT_EXPERT_VRAM_LAYERS: usize = 20;
const DEFAULT_CONTEXT: usize = 128;
const DEFAULT_ITERATIONS: usize = 5;
const DEFAULT_MAX_SEQ: usize = 1024;
/// Context slack past the last measured position (the decode forwards also
/// reserve one trailing row).
const POSITION_SLACK: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VramLayers {
    All,
    Layers(usize),
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    model: PathBuf,
    output: Option<PathBuf>,
    vram_layers: VramLayers,
    context: usize,
    iterations: usize,
    max_seq: usize,
}

fn parse_count(flag: &str, value: &str) -> Result<usize, String> {
    value.parse::<usize>().map_err(|e| format!("{flag}: {e}"))
}

fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Args, String> {
    let mut args = args.into_iter();
    let mut model = None;
    let mut output = None;
    let mut vram_layers = VramLayers::Layers(DEFAULT_EXPERT_VRAM_LAYERS);
    let (mut context, mut iterations, mut max_seq) =
        (DEFAULT_CONTEXT, DEFAULT_ITERATIONS, DEFAULT_MAX_SEQ);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--output" => {
                let value = value()?;
                output = (value != "-").then(|| PathBuf::from(value));
            }
            "--expert-vram-layers" => {
                let value = value()?;
                vram_layers = if value == "all" {
                    VramLayers::All
                } else {
                    VramLayers::Layers(parse_count("--expert-vram-layers", &value)?)
                };
            }
            "--context" => context = parse_count("--context", &value()?)?,
            "--iterations" => iterations = parse_count("--iterations", &value()?)?,
            "--max-seq" => max_seq = parse_count("--max-seq", &value()?)?,
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    let model = model.ok_or("--model is required")?;
    if context < 2 {
        return Err("--context must be >= 2 (the prime is a multi-row prefill)".into());
    }
    if iterations == 0 {
        return Err("--iterations must be > 0".into());
    }
    if context
        .checked_add(iterations)
        .and_then(|n| n.checked_add(POSITION_SLACK))
        .is_none_or(|n| n > max_seq)
    {
        return Err(format!(
            "--max-seq {max_seq} must be >= --context {context} + --iterations {iterations} + {POSITION_SLACK}"
        ));
    }
    Ok(Args {
        model,
        output,
        vram_layers,
        context,
        iterations,
        max_seq,
    })
}

/// One row per check: exact mismatch count (`null` when the check could not be
/// evaluated), error text, and the evidence the verdict came from.
#[derive(Default)]
struct Gate {
    checks: Vec<Value>,
}

impl Gate {
    fn push(
        &mut self,
        name: &str,
        pass: bool,
        mismatches: Option<u64>,
        error: Option<String>,
        detail: Value,
    ) {
        eprintln!(
            "qwen4_g2_replay_gate: {} {name}{}",
            if pass { "PASS" } else { "FAIL" },
            error
                .as_deref()
                .map(|e| format!(": {e}"))
                .unwrap_or_default()
        );
        self.checks.push(json!({
            "name": name,
            "pass": pass,
            "mismatch_count": mismatches,
            "error": error,
            "detail": detail,
        }));
    }

    /// A check decided by an exact mismatch count: passes iff it is zero.
    fn count(&mut self, name: &str, mismatches: u64, detail: Value) {
        self.push(name, mismatches == 0, Some(mismatches), None, detail);
    }

    /// A check that could not be evaluated: failing, with the reason.
    fn error(&mut self, name: &str, error: impl Into<String>) {
        self.push(name, false, None, Some(error.into()), Value::Null);
    }

    fn all_pass(&self) -> bool {
        self.checks
            .iter()
            .all(|check| check["pass"].as_bool() == Some(true))
    }

    fn failed(&self) -> Vec<Value> {
        self.checks
            .iter()
            .filter(|check| check["pass"].as_bool() != Some(true))
            .map(|check| check["name"].clone())
            .collect()
    }
}

/// What the loader reports about the model it built.
struct LoadFacts {
    slab_recipe: bool,
    host_mapped_tensors: usize,
    vocab: usize,
    /// Loader arch label (`hipfire_loader::arch_label(ARCH_ID)`; the loader
    /// crate depends on this one, so the label is named here).
    model_arch: &'static str,
    /// Mesh parallelism of the load: one rank.
    pp: usize,
    tp: usize,
    /// Whether a speculative drafter was installed by the load. This loader
    /// builds the bare bundle (no MTP, no DFlash, no n-gram), so never.
    has_drafter: bool,
}

const MODEL_ARCH: &str = "qwen4";

/// Why a model that is not on the product's default Redline route is not run.
const NOT_DEFAULT_REASON: &str = "Redline not on the default route (config.rs:15–22)";

/// The inputs of the daemon's retained-Redline admission for this load.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RouteFacts {
    gpu_arch: String,
    model_arch: String,
    model_path: String,
    pp: usize,
    tp: usize,
    ep: usize,
    has_drafter: bool,
    platform_refusal: Option<String>,
}

impl RouteFacts {
    fn from_load(gpu_arch: &str, model: &Path, facts: &LoadFacts) -> Self {
        Self {
            gpu_arch: gpu_arch.to_string(),
            model_arch: facts.model_arch.to_string(),
            model_path: model.to_string_lossy().into_owned(),
            pp: facts.pp,
            tp: facts.tp,
            // The loader's `effective_ep` for a Qwen4 model.
            ep: hipfire_config::topology::qwen4_ep().unwrap_or(1),
            has_drafter: facts.has_drafter,
            platform_refusal: hipfire_config::devices::redline_platform_refusal(),
        }
    }

    /// The daemon's decision: `retained_redline_default` and no platform
    /// refusal. The same function the product calls, not a copy of it.
    fn default_route(&self) -> bool {
        hipfire_runtime::config::retained_redline_default(
            &self.gpu_arch,
            &self.model_arch,
            &self.model_path,
            self.pp,
            self.tp,
            self.ep,
            self.has_drafter,
        ) && self.platform_refusal.is_none()
    }

    fn to_json(&self) -> Value {
        json!({
            "default_route": self.default_route(),
            "gpu_arch": self.gpu_arch,
            "model_arch": self.model_arch,
            "model_path": self.model_path,
            "pp": self.pp,
            "tp": self.tp,
            "ep": self.ep,
            "has_drafter": self.has_drafter,
            "platform_refusal": self.platform_refusal,
        })
    }
}

/// The slab-capable range-loaded path of `qwen4_greedy_ids` / `qwen4_kld`
/// (arena payloads, fixed layer placement with the host-mapped borrowed
/// fallback), without MTP, the session cache or the PLE hot store.
fn load_bundle(
    gpu: &mut Gpu,
    path: &Path,
    max_seq: usize,
    vram_layers: VramLayers,
) -> Result<(Qwen4Bundle, LoadFacts), String> {
    let mut hfq = HfqFile::open(path).map_err(|e| format!("open HFQM {}: {e}", path.display()))?;
    // Never derives a missing q4 PLE table: derive it with
    // `hipfire ple derive` (or a serve load) first.
    let choice = attach_validation_ple_rows(
        &mut hfq,
        path,
        hipfire_config::memory::ple_format(),
        &gpu.arch,
    )
    .map_err(|e| format!("qwen4 PLE rows: {e}"))?;
    eprintln!("qwen4_g2_replay_gate: ple rows: {}", choice.format.name());
    let receipt = admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
        .map_err(|e| format!("qwen4 admission: {e}"))?;
    let trunk_overlay =
        hipfire_arch_qwen4::resolve_trunk_overlay(&hfq, path, &gpu.arch, &receipt.config)
            .map_err(|e| format!("qwen4 trunk overlay: {e}"))?;
    let scratch_mode = hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
        gpu,
        &hfq,
        &receipt.config,
        &receipt.manifest,
    )
    .map_err(|e| e.to_string())?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().map_err(|e| format!("qwen4 mesh: {e}"))?;
    let expected = WeightOrigin::for_single(&mesh, gpu);
    let npu_identity = hipfire_arch_qwen4::npu_trunk::model_index_identity(&hfq, &gpu.arch);
    let slab_recipe = matches!(
        receipt.manifest.identity.recipe,
        hipfire_arch_qwen4::Qwen4TierRecipe::FlashNextSlab { .. }
    );
    let source = HfqModelSource::from_hfq(hfq);
    let mut weights = receipt.manifest.weights.clone();
    let host_mapped_tensors = match vram_layers {
        VramLayers::All => 0,
        VramLayers::Layers(layers) => {
            if gpu.is_uma() {
                return Err("host-mapped experts are only for discrete GPUs".into());
            }
            let moved =
                hipfire_arch_qwen4::expert_residency::place_routed_experts(&mut weights, layers);
            eprintln!(
                "qwen4_g2_replay_gate: routed experts: layers 0..{layers} in VRAM, {moved} tensors host-mapped"
            );
            moved
        }
    };
    let transaction = fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        receipt.config.num_hidden_layers,
        gpu,
        expected,
        |entry| {
            if let Some(payload) = hipfire_arch_qwen4::expert_arena::recipe_arena_payload(
                &source,
                &receipt.manifest.identity.recipe,
                entry,
            ) {
                return payload;
            }
            // The store rejects a host-mapped entry as a range: borrow it
            // from the resident source bytes, as the product loader does.
            if entry.residency == hipfire_runtime::weight_manifest::WeightResidency::HostMapped {
                let (info, bytes) = source
                    .tensor_data(&entry.name)
                    .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
                return Ok(SourcePayload::Borrowed { info, bytes });
            }
            source
                .tensor_range(&entry.name)
                .map_err(|e| e.to_string())?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .map_err(|e| format!("qwen4 manifest fulfillment: {e}"))?;
    let vocab = receipt.config.vocab_size;
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        // Qwen3.5's DeltaNet `state_quant` knob: auto (q8) or fp32.
        &std::env::var("HIPFIRE_STATE_QUANT").unwrap_or_default(),
        gpu,
        &receipt.config,
    )?;
    let backend = hipfire_arch_qwen4::Qwen4KvBackend::automatic(gpu);
    let mut bundle = Qwen4Bundle::assemble_with_ple_rows(
        receipt.config,
        &receipt.manifest.identity.recipe,
        transaction,
        &receipt.placements,
        gpu,
        max_seq,
        receipt.ple,
        receipt.ple_rows,
        state_format,
        backend,
        None,
    )
    .map_err(|e| format!("qwen4 bundle assembly: {e}"))?;
    bundle.attach_npu_trunk_identity(npu_identity);
    if let Some(overlay) = trunk_overlay {
        bundle
            .attach_trunk_overlay(overlay)
            .map_err(|e| format!("qwen4 trunk overlay: {e}"))?;
    }
    bundle
        .attach_forward(gpu, max_seq, scratch_mode)
        .map_err(|e| format!("qwen4 forward setup: {e}"))?;
    Ok((
        bundle,
        LoadFacts {
            slab_recipe,
            host_mapped_tensors,
            vocab,
            model_arch: MODEL_ARCH,
            // `DeviceMesh::single()`: one rank, no pipeline or tensor split.
            pp: 1,
            tp: 1,
            has_drafter: false,
        },
    ))
}

// ── state snapshot ─────────────────────────────────────────────────────

/// Everything one forward writes that a later forward or a reader observes.
struct Snapshot {
    logits: Vec<u8>,
    argmax: u32,
    position: usize,
    lengths: Vec<(usize, usize, usize, usize, usize)>,
    regions: Vec<(String, Vec<u8>)>,
}

fn append_buffer(
    gpu: &Gpu,
    output: &mut Vec<u8>,
    buffer: &hip_bridge::DeviceBuffer,
) -> Result<(), String> {
    let start = output.len();
    output.resize(start + buffer.size(), 0);
    gpu.hip
        .memcpy_dtoh(&mut output[start..], buffer)
        .map_err(|e| e.to_string())
}

/// `len` elements of `tensor` from `offset`; an empty slice reads nothing.
fn tensor_slice_bytes(
    gpu: &Gpu,
    tensor: &GpuTensor,
    offset: usize,
    len: usize,
) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    if len == 0 {
        return Ok(bytes);
    }
    if offset.saturating_add(len) > tensor.numel() {
        return Err(format!(
            "tensor slice {offset}+{len} exceeds {}",
            tensor.numel()
        ));
    }
    let view = tensor.sub_offset(offset, len);
    append_buffer(gpu, &mut bytes, &view.buf)?;
    Ok(bytes)
}

fn whole_bytes(gpu: &Gpu, tensor: &GpuTensor) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    append_buffer(gpu, &mut bytes, &tensor.buf)?;
    Ok(bytes)
}

/// Capacity-pinned QSA arenas are read up to their active mark only: beyond
/// `full_len`/`raw_len`/`pooled_len` the model never reads them and `reset`
/// zeroes them. The bounded buffers (partial block, selected indices) are read
/// whole.
fn snapshot(gpu: &Gpu, bundle: &Qwen4Bundle, logits: &GpuTensor) -> Result<Snapshot, String> {
    let config = &bundle.config;
    let raw_width = config
        .indexer_kv_heads
        .checked_mul(config.indexer_head_dim)
        .ok_or("raw index width overflows")?;
    let logits_bytes = whole_bytes(gpu, logits)?;
    let argmax = logits_bytes
        .chunks_exact(4)
        .enumerate()
        .map(|(index, chunk)| {
            (
                index as u32,
                f32::from_le_bytes(chunk.try_into().expect("4-byte f32 chunk")),
            )
        })
        .filter(|(_, value)| value.is_finite())
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(index, _)| index);

    let state = &bundle.state;
    let mut regions: Vec<(String, Vec<u8>)> = Vec::new();
    for (layer, gdn) in state.gdn.iter().enumerate() {
        for (name, tensor) in [("recurrent", &gdn.recurrent), ("conv", &gdn.conv)] {
            regions.push((format!("gdn.{layer}.{name}"), whole_bytes(gpu, tensor)?));
        }
    }
    let mut lengths = Vec::with_capacity(state.qsa.len());
    for (layer, qsa) in state.qsa.iter().enumerate() {
        let active = [
            (
                "full_keys",
                &qsa.full_keys,
                qsa.full_len,
                qsa.full_row_units,
            ),
            (
                "full_values",
                &qsa.full_values,
                qsa.full_len,
                qsa.full_row_units,
            ),
            ("raw_keys", &qsa.raw_index_keys, qsa.raw_len, raw_width),
            ("pooled_keys", &qsa.pooled_keys, qsa.pooled_len, raw_width),
        ];
        for (name, tensor, rows, width) in active {
            let elements = rows
                .checked_mul(width)
                .ok_or_else(|| format!("qsa.{layer}.{name} extent overflows"))?;
            regions.push((
                format!("qsa.{layer}.{name}"),
                tensor_slice_bytes(gpu, tensor, 0, elements)?,
            ));
        }
        for (name, tensor) in [
            ("partial_keys", &qsa.partial_keys),
            ("partial_values", &qsa.partial_values),
            ("selected", &qsa.selected_indices),
        ] {
            regions.push((format!("qsa.{layer}.{name}"), whole_bytes(gpu, tensor)?));
        }
        lengths.push((
            qsa.full_len,
            qsa.raw_len,
            qsa.pooled_len,
            qsa.partial_len,
            qsa.selected_len,
        ));
    }
    for (name, tensor) in state
        .ple_conv
        .as_ref()
        .map(|tensor| ("ple.conv", tensor))
        .into_iter()
        .chain(
            state
                .hyper_feedback
                .as_ref()
                .map(|t| ("ple.hyper_feedback", t)),
        )
    {
        regions.push((name.to_string(), whole_bytes(gpu, tensor)?));
    }
    Ok(Snapshot {
        logits: logits_bytes,
        argmax,
        position: state.position,
        lengths,
        regions,
    })
}

/// Differing bytes of two buffers, plus the length difference.
fn diff_bytes(a: &[u8], b: &[u8]) -> u64 {
    let common = a.len().min(b.len());
    let differing = a[..common]
        .iter()
        .zip(&b[..common])
        .filter(|(x, y)| x != y)
        .count();
    (differing + a.len().abs_diff(b.len())) as u64
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Mismatch {
    logits: u64,
    gdn: u64,
    qsa: u64,
    ple: u64,
    lengths: u64,
    position: u64,
    argmax: u64,
}

impl Mismatch {
    fn add(&mut self, other: Mismatch) {
        self.logits += other.logits;
        self.gdn += other.gdn;
        self.qsa += other.qsa;
        self.ple += other.ple;
        self.lengths += other.lengths;
        self.position += other.position;
        self.argmax += other.argmax;
    }

    fn json(&self) -> Value {
        json!({
            "logits": self.logits, "gdn": self.gdn, "qsa": self.qsa, "ple": self.ple,
            "lengths": self.lengths, "position": self.position, "argmax": self.argmax,
        })
    }
}

/// Exact byte comparison of two snapshots, grouped by state family.
fn compare(candidate: &Snapshot, baseline: &Snapshot) -> Mismatch {
    let mut out = Mismatch {
        logits: diff_bytes(&candidate.logits, &baseline.logits),
        lengths: candidate
            .lengths
            .iter()
            .zip(&baseline.lengths)
            .filter(|(a, b)| a != b)
            .count() as u64
            + candidate.lengths.len().abs_diff(baseline.lengths.len()) as u64,
        position: u64::from(candidate.position != baseline.position),
        argmax: u64::from(candidate.argmax != baseline.argmax),
        ..Mismatch::default()
    };
    for index in 0..candidate.regions.len().max(baseline.regions.len()) {
        let (name, mismatched) = match (candidate.regions.get(index), baseline.regions.get(index)) {
            (Some((name, a)), Some((other, b))) if name == other => {
                (name.as_str(), diff_bytes(a, b))
            }
            // A region the other side lacks, or under another name, differs in
            // every byte (and at least once when empty).
            (Some((name, a)), _) => (name.as_str(), a.len().max(1) as u64),
            (None, Some((name, b))) => (name.as_str(), b.len().max(1) as u64),
            (None, None) => continue,
        };
        if name.starts_with("gdn.") {
            out.gdn += mismatched;
        } else if name.starts_with("qsa.") {
            out.qsa += mismatched;
        } else {
            out.ple += mismatched;
        }
    }
    out
}

fn f32_stats(bytes: &[u8]) -> (u64, u64) {
    let mut non_finite = 0u64;
    let mut non_zero = 0u64;
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().expect("4-byte f32 chunk"));
        if !value.is_finite() {
            non_finite += 1;
        }
        if value != 0.0 {
            non_zero += 1;
        }
    }
    (non_finite, non_zero)
}

// ── arms ───────────────────────────────────────────────────────────────

fn prime_tokens(context: usize) -> Vec<u32> {
    (0..context as u32).map(|i| 10 + (i % 1000)).collect()
}

fn decode_token(iteration: usize) -> u32 {
    101 + (iteration as u32 % 1000)
}

fn launched() -> u64 {
    hip_bridge::launch_counters::launch_kernel::count()
}

struct ArmRun {
    snapshots: Vec<Snapshot>,
    /// HIP kernel launches of each decode forward.
    launched: Vec<u64>,
    /// Retained replays the controller counted over the decode forwards.
    observed: ReplayObservation,
    state_after: ReplayState,
}

/// Reset, prime `context` tokens (a multi-row prefill, never a retained body),
/// then `iterations` single-token forwards under `route`, snapshotting each.
/// `route = None` is the production decision (arm / route / HIP by state).
fn run_arm(
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    route: Option<ShadowBodyRoute>,
    context: usize,
    iterations: usize,
) -> Result<ArmRun, String> {
    let vocab = bundle.config.vocab_size;
    bundle.reset(gpu).map_err(|e| format!("reset: {e}"))?;
    // Fresh zeroed logits per arm: a replay that never ran leaves them zero.
    let logits = gpu
        .zeros(&[vocab], DType::F32)
        .map_err(|e| format!("logits alloc: {e}"))?;
    let outcome = (|| -> Result<ArmRun, String> {
        gpu.replay.set_shadow_body_route(route);
        bundle
            .forward_chunk_final(gpu, &prime_tokens(context), &logits, None)
            .map_err(|e| format!("prime: {e}"))?;
        gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
        gpu.replay.begin_replay_observation_window();
        let mut snapshots = Vec::with_capacity(iterations);
        let mut launches = Vec::with_capacity(iterations);
        for i in 0..iterations {
            let before = launched();
            bundle
                .forward_token(gpu, decode_token(i), &logits, None)
                .map_err(|e| format!("forward {i} at position {}: {e}", bundle.state.position))?;
            launches.push(launched().saturating_sub(before));
            gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
            snapshots.push(snapshot(gpu, bundle, &logits)?);
        }
        Ok(ArmRun {
            snapshots,
            launched: launches,
            observed: gpu.replay.replay_observation(),
            state_after: gpu.replay.state(),
        })
    })();
    gpu.replay.set_shadow_body_route(None);
    let _ = gpu.free_tensor(logits);
    outcome
}

fn plan_ready(gpu: &Gpu) -> bool {
    gpu.replay.prepared_pm4_plan_ready() || gpu.replay.prepared_aql_plan_ready()
}

// ── phase 1: capture ───────────────────────────────────────────────────

struct Capture {
    recorded: usize,
    arm: ArmRun,
}

fn capture_phase(
    gate: &mut Gate,
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    context: usize,
    report: &mut serde_json::Map<String, Value>,
) -> Option<Capture> {
    let before = gpu.replay.state();
    gate.push(
        "capture_window_armed_before",
        before == ReplayState::Armed,
        Some(u64::from(before != ReplayState::Armed)),
        (before != ReplayState::Armed).then(|| {
            format!(
                "replay state {before:?} before the capture forward (want Armed); fallback_reason={:?}",
                gpu.replay.fallback_reason()
            )
        }),
        json!({"state": format!("{before:?}"), "transport": gpu.replay.transport_name()}),
    );
    let arm = match run_arm(gpu, bundle, None, context, 1) {
        Ok(arm) => arm,
        Err(error) => {
            gate.error("capture_forward", error);
            return None;
        }
    };
    let state = gpu.replay.state();
    let summary = gpu.replay.capture_summary();
    let effects = gpu.replay.capture_window_effects();
    let identity = gpu.replay.prepared_route_identity();
    let identities = gpu.replay.route_identity_count();
    let recorded = gpu.replay.recorded_launches().len();
    let hip_launches = arm.launched.first().copied().unwrap_or(0);

    let closed = state == ReplayState::Ready && plan_ready(gpu);
    gate.push(
        "capture_executed",
        closed
            && recorded > 0
            && summary.launch_count == recorded
            && hip_launches >= recorded as u64,
        Some(
            u64::from(!closed)
                + u64::from(recorded == 0)
                + u64::from(summary.launch_count != recorded)
                + u64::from(hip_launches < recorded as u64),
        ),
        (!closed).then(|| {
            format!(
                "capture did not close into a prepared plan: state {state:?}, fallback_reason={:?}",
                gpu.replay.fallback_reason()
            )
        }),
        json!({
            "state": format!("{state:?}"),
            "recorded_launches": recorded,
            "summary_launches": summary.launch_count,
            "unique_kernels": summary.unique_kernel_count,
            "sequence_hash": format!("{:016x}", summary.sequence_hash),
            "capture_forward_hip_launches": hip_launches,
            "plan_ready": plan_ready(gpu),
        }),
    );
    gate.push(
        "capture_window_effects_replayable",
        effects.replayable(),
        Some(u64::from(!effects.replayable())),
        None,
        json!({
            "htod": effects.htod, "dtod": effects.dtod,
            "dtoh": effects.dtoh, "memset": effects.memset,
        }),
    );
    gate.push(
        "capture_route_identities_latched",
        identities > 0,
        Some(u64::from(identities == 0)),
        (identities == 0).then(|| {
            "the tape latched no route identity: the sealed MoE route never noted its \
             pointer mapping, so replay could not validate any binding"
                .to_string()
        }),
        json!({"route_identities": identities}),
    );
    let dispatch_matches = identity
        .as_ref()
        .is_some_and(|identity| identity.dispatch_count == recorded);
    gate.push(
        "capture_prepared_dispatches_equal_launches",
        dispatch_matches,
        Some(u64::from(!dispatch_matches)),
        identity
            .is_none()
            .then(|| "no prepared route identity after capture".to_string()),
        json!({
            "prepared_dispatches": identity.as_ref().map(|i| i.dispatch_count),
            "queue_id": identity.as_ref().map(|i| i.queue_id),
            "command_dwords": identity.as_ref().and_then(|i| i.command_dwords),
            "packets": identity.as_ref().and_then(|i| i.packet_count),
        }),
    );
    report.insert(
        "capture".into(),
        json!({
            "transport": gpu.replay.transport_name(),
            "launches": summary.launch_count,
            "unique_kernels": summary.unique_kernel_count,
            "sequence_hash": format!("{:016x}", summary.sequence_hash),
            "route_identities": identities,
            "position": context,
        }),
    );
    if !closed || recorded == 0 {
        return None;
    }
    Some(Capture { recorded, arm })
}

// ── phase 2: positive ──────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Executor {
    /// `shadow_body_route = None`, `Ready`: the production route.
    Production,
    Plan,
    HipOracle,
}

impl Executor {
    fn name(self) -> &'static str {
        match self {
            Self::Production => "production",
            Self::Plan => "plan",
            Self::HipOracle => "hip_oracle",
        }
    }

    fn route(self) -> Option<ShadowBodyRoute> {
        match self {
            Self::Production => None,
            Self::Plan => Some(ShadowBodyRoute::Plan),
            Self::HipOracle => Some(ShadowBodyRoute::HipOracle),
        }
    }
}

fn positive_phase(
    gate: &mut Gate,
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    capture: &Capture,
    context: usize,
    iterations: usize,
) {
    let recorded = capture.recorded as u64;
    let hip = match run_arm(gpu, bundle, Some(ShadowBodyRoute::Hip), context, iterations) {
        Ok(arm) => arm,
        Err(error) => {
            gate.error("hip_baseline", error);
            return;
        }
    };
    let (non_finite, non_zero) = hip
        .snapshots
        .iter()
        .map(|s| f32_stats(&s.logits))
        .fold((0, 0), |(a, b), (c, d)| (a + c, b + d.min(1)));
    let silent = (iterations as u64).saturating_sub(non_zero);
    gate.count(
        "hip_baseline_logits_finite_nonzero",
        non_finite + silent,
        json!({"non_finite_values": non_finite, "all_zero_positions": silent}),
    );
    let short = hip.launched.iter().filter(|&&n| n < recorded).count() as u64;
    gate.count(
        "hip_baseline_ran_recorded_body",
        short,
        json!({"recorded_launches": recorded, "launched_per_forward": hip.launched}),
    );
    // The capture forward is an ordinary HIP forward at the same position
    // from the same primed state: the first baseline row must equal it.
    let capture_vs_hip = match (capture.arm.snapshots.first(), hip.snapshots.first()) {
        (Some(captured), Some(baseline)) => Some(compare(captured, baseline)),
        _ => None,
    };
    match capture_vs_hip {
        Some(m) => gate.count(
            "capture_forward_matches_hip_baseline",
            m.logits + m.gdn + m.qsa + m.ple + m.lengths + m.position + m.argmax,
            m.json(),
        ),
        None => gate.error(
            "capture_forward_matches_hip_baseline",
            "missing capture snapshot",
        ),
    }

    for executor in [Executor::Production, Executor::Plan, Executor::HipOracle] {
        let name = executor.name();
        let ready_before = gpu.replay.state();
        let arm = match run_arm(gpu, bundle, executor.route(), context, iterations) {
            Ok(arm) => arm,
            Err(error) => {
                gate.error(
                    &format!("positive_{name}_forward"),
                    format!(
                        "{error} (state before {ready_before:?}, now {:?}, fallback_reason={:?})",
                        gpu.replay.state(),
                        gpu.replay.fallback_reason()
                    ),
                );
                continue;
            }
        };
        // It ran: the plan executors were counted by the controller and issued
        // fewer HIP kernels than the recorded body; the oracle re-issued it.
        let last = context + iterations - 1;
        let (executed_mismatch, evidence) = match executor {
            Executor::Production | Executor::Plan => {
                let count_off = arm.observed.count.abs_diff(iterations as u64);
                let first_off = u64::from(arm.observed.first_position != Some(context));
                let last_off = u64::from(arm.observed.last_position != Some(last));
                let failed = u64::from(arm.observed.failed);
                let body_ran = arm.launched.iter().filter(|&&n| n >= recorded).count() as u64;
                (
                    count_off + first_off + last_off + failed + body_ran,
                    json!({
                        "observed_replays": arm.observed.count,
                        "expected_replays": iterations,
                        "first_position": arm.observed.first_position,
                        "last_position": arm.observed.last_position,
                        "replay_failed": arm.observed.failed,
                        "hip_launches_per_forward": arm.launched,
                        "recorded_launches": recorded,
                    }),
                )
            }
            Executor::HipOracle => {
                let short = arm.launched.iter().filter(|&&n| n < recorded).count() as u64;
                (
                    short,
                    json!({
                        "hip_launches_per_forward": arm.launched,
                        "recorded_launches": recorded,
                    }),
                )
            }
        };
        gate.count(
            &format!("positive_{name}_executed"),
            executed_mismatch,
            evidence,
        );
        let stays_ready = arm.state_after == ReplayState::Ready;
        gate.push(
            &format!("positive_{name}_route_stays_ready"),
            stays_ready,
            Some(u64::from(!stays_ready)),
            (!stays_ready).then(|| {
                format!(
                    "replay state {:?} after the arm, fallback_reason={:?}",
                    arm.state_after,
                    gpu.replay.fallback_reason()
                )
            }),
            json!({"state": format!("{:?}", arm.state_after)}),
        );

        let mut total = Mismatch::default();
        let mut per_position = Vec::with_capacity(iterations);
        for (candidate, baseline) in arm.snapshots.iter().zip(&hip.snapshots) {
            let m = compare(candidate, baseline);
            per_position.push(json!({"position": baseline.position, "mismatch": m.json()}));
            total.add(m);
        }
        let compared: u64 = hip.snapshots.first().map_or(0, |s| {
            s.regions.iter().map(|(_, b)| b.len() as u64).sum::<u64>() + s.logits.len() as u64
        });
        for (family, count) in [
            ("logits", total.logits),
            ("gdn", total.gdn),
            ("qsa", total.qsa),
            ("ple", total.ple),
            ("lengths", total.lengths),
            ("position", total.position),
            ("argmax", total.argmax),
        ] {
            let positions = if family == "logits" {
                Value::Array(per_position.clone())
            } else {
                Value::Null
            };
            gate.count(
                &format!("positive_{name}_{family}_exact"),
                count,
                json!({
                    "positions": iterations,
                    "bytes_compared_per_position": compared,
                    "per_position": positions,
                }),
            );
        }
    }
}

// ── phase 3: changed binding ───────────────────────────────────────────

/// The refusal `validate_route_identities` produces for a binding the tape
/// never latched, as wrapped by the forward.
fn is_binding_refusal(error: &str) -> bool {
    error.contains("retained replay refused")
        && error.contains("retained route identity for table")
        && (error.contains("is not bound at replay") || error.contains("changed since capture"))
}

struct Attempt {
    error: Option<String>,
    observed_before: ReplayObservation,
    observed_after: ReplayObservation,
    launched: u64,
    state_after: ReplayState,
    reason_after: Option<String>,
}

fn attempt(
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    logits: &GpuTensor,
    route: Option<ShadowBodyRoute>,
) -> Attempt {
    let observed_before = gpu.replay.replay_observation();
    gpu.replay.set_shadow_body_route(route);
    let before = launched();
    let result = bundle.forward_token(gpu, decode_token(0), logits, None);
    let launched = launched().saturating_sub(before);
    gpu.replay.set_shadow_body_route(None);
    let _ = gpu.hip.device_synchronize();
    Attempt {
        error: result.err().map(|e| e.to_string()),
        observed_before,
        observed_after: gpu.replay.replay_observation(),
        launched,
        state_after: gpu.replay.state(),
        reason_after: gpu.replay.fallback_reason().map(str::to_string),
    }
}

/// `bundle` is the rebuilt one; `gpu.replay` still holds the plan captured on
/// the bundle it replaced.
fn negative_phase(
    gate: &mut Gate,
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    capture: &Capture,
    context: usize,
) {
    let recorded = capture.recorded as u64;
    let state = gpu.replay.state();
    let identities = gpu.replay.route_identity_count();
    let survived = state == ReplayState::Ready && plan_ready(gpu) && identities > 0;
    gate.push(
        "rebuild_kept_prepared_plan",
        survived,
        Some(u64::from(!survived)),
        (!survived).then(|| {
            format!(
                "after the bundle rebuild the controller is {state:?} (plan_ready={}, \
                 route_identities={identities}, fallback_reason={:?}); a refusal could not \
                 be attributed to binding validation",
                plan_ready(gpu),
                gpu.replay.fallback_reason()
            )
        }),
        json!({
            "state": format!("{state:?}"),
            "plan_ready": plan_ready(gpu),
            "route_identities": identities,
        }),
    );
    if !survived {
        return;
    }

    let vocab = bundle.config.vocab_size;
    if let Err(error) = bundle.reset(gpu) {
        gate.error("negative_prime", format!("reset: {error}"));
        return;
    }
    let logits = match gpu.zeros(&[vocab], DType::F32) {
        Ok(logits) => logits,
        Err(error) => {
            gate.error("negative_prime", format!("logits alloc: {error}"));
            return;
        }
    };
    if let Err(error) = bundle
        .forward_chunk_final(gpu, &prime_tokens(context), &logits, None)
        .map_err(|e| e.to_string())
        .and_then(|()| gpu.hip.device_synchronize().map_err(|e| e.to_string()))
    {
        gate.error("negative_prime", format!("prime: {error}"));
        let _ = gpu.free_tensor(logits);
        return;
    }

    let mut first_poison: Option<(ReplayState, Option<String>)> = None;
    for executor in [Executor::Production, Executor::Plan, Executor::HipOracle] {
        let name = executor.name();
        let tried = attempt(gpu, bundle, &logits, executor.route());
        if first_poison.is_none() {
            first_poison = Some((tried.state_after, tried.reason_after.clone()));
        }
        let (refused, why) = match tried.error.as_deref() {
            Some(error) if is_binding_refusal(error) => (true, None),
            Some(error) => (
                false,
                Some(format!("forward failed, but not with the binding refusal: {error}")),
            ),
            None => (
                false,
                Some(
                    "forward succeeded against a rebuilt binding: the changed binding was not refused"
                        .to_string(),
                ),
            ),
        };
        gate.push(
            &format!("negative_{name}_refused"),
            refused,
            Some(u64::from(!refused)),
            why,
            json!({"forward_error": tried.error}),
        );
        let replays = tried
            .observed_after
            .count
            .abs_diff(tried.observed_before.count);
        let body = u64::from(tried.launched >= recorded);
        let flagged = u64::from(tried.observed_after.failed != tried.observed_before.failed);
        gate.count(
            &format!("negative_{name}_no_retained_execution"),
            replays + body + flagged,
            json!({
                "replays_observed": replays,
                "hip_launches": tried.launched,
                "recorded_launches": recorded,
                "replay_failed_flag_changed": flagged == 1,
            }),
        );
    }
    if let Some((state, reason)) = first_poison {
        let poisoned = state == ReplayState::Fallback
            && reason
                .as_deref()
                .is_some_and(|reason| reason.contains("retained route binding changed"));
        gate.push(
            "negative_refusal_poisons_route",
            poisoned,
            Some(u64::from(!poisoned)),
            (!poisoned).then(|| {
                format!("after the refusal the controller is {state:?}, fallback_reason={reason:?}")
            }),
            json!({"state": format!("{state:?}"), "fallback_reason": reason}),
        );
    }

    // The poisoned route runs the next forward on HIP, over the same state.
    let recovered = attempt(gpu, bundle, &logits, None);
    let logits_ok = whole_bytes(gpu, &logits)
        .map(|bytes| f32_stats(&bytes))
        .map_err(|e| e.to_string());
    let (bad, detail_error) = match (&recovered.error, &logits_ok) {
        (Some(error), _) => (1, Some(format!("recovery forward failed: {error}"))),
        (None, Err(error)) => (1, Some(format!("recovery logits read failed: {error}"))),
        (None, Ok((non_finite, non_zero))) => {
            let ran_hip = recovered.launched >= recorded;
            let still_poisoned = recovered.state_after == ReplayState::Fallback;
            let clean = *non_finite == 0 && *non_zero > 0;
            (
                u64::from(!ran_hip) + u64::from(!still_poisoned) + u64::from(!clean),
                None,
            )
        }
    };
    gate.push(
        "negative_recovers_on_hip",
        bad == 0,
        Some(bad),
        detail_error,
        json!({
            "state": format!("{:?}", recovered.state_after),
            "hip_launches": recovered.launched,
            "recorded_launches": recorded,
        }),
    );
    let _ = gpu.free_tensor(logits);
}

// ── driver ─────────────────────────────────────────────────────────────

fn run(args: &Args) -> Result<(Value, bool), String> {
    let started = Instant::now();
    let mut gate = Gate::default();
    let mut report = serde_json::Map::new();
    let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
    let arch = gpu.arch.to_string();

    let (mut bundle, facts) =
        match load_bundle(&mut gpu, &args.model, args.max_seq, args.vram_layers) {
            Ok(loaded) => loaded,
            Err(error) => {
                gate.error("load_bundle", error);
                return Ok((finish(&gate, &report, args, &arch, started), false));
            }
        };
    // The gate is required only where the product takes the retained Redline
    // route by default, decided from the loaded model's facts by the daemon's
    // own admission. Elsewhere it does not force the route: no capture,
    // positive or negative phase runs, and the report is a successful SKIP.
    let route = RouteFacts::from_load(&arch, &args.model, &facts);
    if !route.default_route() {
        eprintln!(
            "qwen4_g2_replay_gate: SKIP: {NOT_DEFAULT_REASON} \
             (gpu_arch={}, model_arch={}, model={}, pp={}, tp={}, ep={}, drafter={}, \
             platform_refusal={:?})",
            route.gpu_arch,
            route.model_arch,
            route.model_path,
            route.pp,
            route.tp,
            route.ep,
            route.has_drafter,
            route.platform_refusal,
        );
        if let Err(error) = bundle.free_gpu(&mut gpu) {
            gate.error("skip_free_bundle", error.to_string());
            return Ok((finish(&gate, &report, args, &arch, started), false));
        }
        return Ok((skip_report(&gate, &route, args, &arch, started), true));
    }
    report.insert("route".into(), route.to_json());
    gate.push(
        "artifact_is_expert_slab",
        facts.slab_recipe,
        Some(u64::from(!facts.slab_recipe)),
        (!facts.slab_recipe).then(|| {
            "the artifact's manifest recipe is not FlashNextSlab: this gate is for ctl3-hot \
             slab artifacts"
                .to_string()
        }),
        json!({"host_mapped_tensors": facts.host_mapped_tensors, "vocab": facts.vocab}),
    );

    // The daemon's post-load default on a default route: Auto backend,
    // retained PM4 transport, state Armed, automatic lifecycle (the first
    // eligible forward captures).
    let configured = gpu.replay.configure_model_default(true);
    let armed = gpu.replay.state() == ReplayState::Armed;
    gate.push(
        "replay_default_configured",
        configured && armed,
        Some(u64::from(!configured) + u64::from(!armed)),
        (!(configured && armed)).then(|| {
            format!(
                "configure_model_default(true) returned {configured}, state {:?}: an explicit \
                 HIPFIRE_REPLAY_BACKEND / HIPFIRE_REPLAY_MANUAL_CAPTURE is in effect",
                gpu.replay.state()
            )
        }),
        json!({"transport": gpu.replay.transport_name(), "state": format!("{:?}", gpu.replay.state())}),
    );
    if !(configured && armed) {
        return Ok((finish(&gate, &report, args, &arch, started), false));
    }

    let capture = capture_phase(&mut gate, &mut gpu, &mut bundle, args.context, &mut report);
    let Some(capture) = capture else {
        gate.error(
            "positive_and_negative_phases",
            "not run: the capture phase did not produce a prepared retained plan",
        );
        let _ = bundle.free_gpu(&mut gpu);
        return Ok((finish(&gate, &report, args, &arch, started), false));
    };
    positive_phase(
        &mut gate,
        &mut gpu,
        &mut bundle,
        &capture,
        args.context,
        args.iterations,
    );

    // Rebuild between capture and replay: every table id and pointer mapping the
    // tape latched is gone; the prepared plan stays installed on `gpu.replay`.
    match bundle.free_gpu(&mut gpu) {
        Ok(()) => {}
        Err(error) => {
            gate.error("rebuild_free_old_bundle", error.to_string());
            return Ok((finish(&gate, &report, args, &arch, started), false));
        }
    }
    match load_bundle(&mut gpu, &args.model, args.max_seq, args.vram_layers) {
        Ok((mut rebuilt, _)) => {
            negative_phase(&mut gate, &mut gpu, &mut rebuilt, &capture, args.context);
            if let Err(error) = rebuilt.free_gpu(&mut gpu) {
                gate.error("rebuild_free_new_bundle", error.to_string());
            }
        }
        Err(error) => gate.error("rebuild_load_bundle", error),
    }
    let ok = gate.all_pass();
    Ok((finish(&gate, &report, args, &arch, started), ok))
}

fn finish(
    gate: &Gate,
    report: &serde_json::Map<String, Value>,
    args: &Args,
    arch: &str,
    started: Instant,
) -> Value {
    let mut out = serde_json::Map::new();
    out.insert("gate".into(), json!("qwen4_g2_replay_gate"));
    out.insert("ok".into(), json!(gate.all_pass()));
    out.insert(
        "status".into(),
        json!(if gate.all_pass() { "pass" } else { "fail" }),
    );
    out.insert("model".into(), json!(args.model.display().to_string()));
    out.insert("arch".into(), json!(arch));
    let layers = match args.vram_layers {
        VramLayers::All => json!("all"),
        VramLayers::Layers(n) => json!(n),
    };
    out.insert(
        "params".into(),
        json!({
            "context": args.context,
            "iterations": args.iterations,
            "max_seq": args.max_seq,
            "expert_vram_layers": layers,
        }),
    );
    for (key, value) in report {
        out.insert(key.clone(), value.clone());
    }
    out.insert("failed_checks".into(), Value::Array(gate.failed()));
    out.insert("checks".into(), Value::Array(gate.checks.clone()));
    out.insert("elapsed_s".into(), json!(started.elapsed().as_secs_f64()));
    Value::Object(out)
}

/// The successful report of a model that is not on the product's default
/// Redline route: `ok`, no checks and no failed checks, `"status": "skip"` and
/// the exact reason. Never a PASS (there is no check row to count) and never
/// a failure (nothing was required of the route). `gate` must hold no checks.
fn skip_report(
    gate: &Gate,
    route: &RouteFacts,
    args: &Args,
    arch: &str,
    started: Instant,
) -> Value {
    debug_assert!(gate.checks.is_empty());
    let mut report = serde_json::Map::new();
    report.insert("route".into(), route.to_json());
    let mut out = finish(gate, &report, args, arch, started);
    let fields = out.as_object_mut().expect("report is an object");
    fields.insert("status".into(), json!("skip"));
    fields.insert("skipped".into(), json!(true));
    fields.insert("reason".into(), json!(NOT_DEFAULT_REASON));
    out
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return;
    }
    let args = match parse_args(argv) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("qwen4_g2_replay_gate: {error}\n{USAGE}");
            std::process::exit(2);
        }
    };
    let (report, ok) = match run(&args) {
        Ok(result) => result,
        Err(error) => {
            // Before any check could be recorded (no GPU).
            let report = json!({
                "gate": "qwen4_g2_replay_gate",
                "ok": false,
                "model": args.model.display().to_string(),
                "failed_checks": ["gpu_init"],
                "checks": [{"name": "gpu_init", "pass": false, "mismatch_count": null,
                            "error": error, "detail": null}],
            });
            (report, false)
        }
    };
    let text = serde_json::to_string_pretty(&report).expect("report serializes");
    let written = match &args.output {
        Some(path) => std::fs::write(path, format!("{text}\n"))
            .map_err(|e| format!("write {}: {e}", path.display())),
        None => writeln!(std::io::stdout().lock(), "{text}").map_err(|e| e.to_string()),
    };
    if let Err(error) = written {
        eprintln!("qwen4_g2_replay_gate: {error}");
        std::process::exit(2);
    }
    if !ok {
        eprintln!("qwen4_g2_replay_gate: FAILED");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn args_defaults_and_required() {
        let a = parse_args(argv("--model m.hfq")).unwrap();
        assert_eq!(
            a,
            Args {
                model: "m.hfq".into(),
                output: None,
                vram_layers: VramLayers::Layers(20),
                context: 128,
                iterations: 5,
                max_seq: 1024,
            }
        );
        assert!(parse_args(argv("--context 64")).is_err());
    }

    #[test]
    fn args_full_and_bad() {
        let a = parse_args(argv(
            "--model m --output o.json --expert-vram-layers all --context 64 --iterations 3 --max-seq 128",
        ))
        .unwrap();
        assert_eq!(a.output, Some("o.json".into()));
        assert_eq!(a.vram_layers, VramLayers::All);
        assert_eq!((a.context, a.iterations, a.max_seq), (64, 3, 128));
        for bad in [
            "--model m --context 1",
            "--model m --iterations 0",
            "--model m --max-seq 100",
            "--model m --expert-vram-layers x",
            "--model m --bogus 1",
            "--model m --context",
        ] {
            assert!(parse_args(argv(bad)).is_err(), "{bad}");
        }
        assert_eq!(
            parse_args(argv("--model m --output -")).unwrap().output,
            None
        );
    }

    #[test]
    fn byte_diff_counts_content_and_length() {
        assert_eq!(diff_bytes(&[1, 2, 3], &[1, 2, 3]), 0);
        assert_eq!(diff_bytes(&[1, 2, 3], &[1, 9, 9]), 2);
        assert_eq!(diff_bytes(&[1, 2, 3], &[1, 2]), 1);
        assert_eq!(diff_bytes(&[], &[0, 0]), 2);
    }

    fn snap(logits: &[u8], regions: &[(&str, &[u8])]) -> Snapshot {
        Snapshot {
            logits: logits.to_vec(),
            argmax: 0,
            position: 7,
            lengths: vec![(1, 1, 1, 1, 1)],
            regions: regions
                .iter()
                .map(|(n, b)| (n.to_string(), b.to_vec()))
                .collect(),
        }
    }

    #[test]
    fn compare_groups_by_family() {
        let base = snap(
            &[0; 8],
            &[
                ("gdn.0.conv", &[1, 2]),
                ("qsa.0.selected", &[3]),
                ("ple.conv", &[4]),
            ],
        );
        assert_eq!(compare(&base, &base), Mismatch::default());
        let mut other = snap(
            &[0, 0, 0, 0, 0, 0, 0, 1],
            &[
                ("gdn.0.conv", &[9, 9]),
                ("qsa.0.selected", &[3]),
                ("ple.conv", &[5]),
            ],
        );
        other.position = 8;
        other.lengths = vec![(2, 1, 1, 1, 1)];
        other.argmax = 3;
        assert_eq!(
            compare(&other, &base),
            Mismatch {
                logits: 1,
                gdn: 2,
                qsa: 0,
                ple: 1,
                lengths: 1,
                position: 1,
                argmax: 1
            }
        );
        // A missing or renamed region is never equal.
        let renamed = snap(
            &[0; 8],
            &[
                ("gdn.1.conv", &[1, 2]),
                ("qsa.0.selected", &[3]),
                ("ple.conv", &[4]),
            ],
        );
        assert!(compare(&renamed, &base).gdn > 0);
    }

    #[test]
    fn binding_refusal_is_the_validation_error_only() {
        let refusal = "Qwen4 retained replay refused: retained route identity for table 0x2a \
                       (latched x) is not bound at replay";
        assert!(is_binding_refusal(refusal));
        assert!(is_binding_refusal(
            "Qwen4 retained replay refused: retained route identity for table 0x2a changed since capture: latched a, bound b"
        ));
        assert!(!is_binding_refusal(
            "Qwen4 retained replay failed: position 128 exceeds prepared max_position"
        ));
        assert!(!is_binding_refusal(
            "Qwen4 retained replay refused: staging-stream sync before PM4 replay failed"
        ));
        assert!(!is_binding_refusal("forward 0: some other error"));
    }

    #[test]
    fn f32_stats_counts_non_finite_and_non_zero() {
        let bytes: Vec<u8> = [0.0f32, 1.5, f32::NAN, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(f32_stats(&bytes), (1, 3));
    }

    fn load_facts(has_drafter: bool) -> LoadFacts {
        LoadFacts {
            slab_recipe: true,
            host_mapped_tensors: 0,
            vocab: 1,
            model_arch: MODEL_ARCH,
            pp: 1,
            tp: 1,
            has_drafter,
        }
    }

    /// The route a load of `path` on `gpu_arch` would face: single rank, the
    /// loader's Qwen4 facts, `ep` 1 and no platform refusal.
    fn route(gpu_arch: &str, path: &str, has_drafter: bool) -> RouteFacts {
        RouteFacts {
            ep: 1,
            platform_refusal: None,
            ..RouteFacts::from_load(gpu_arch, Path::new(path), &load_facts(has_drafter))
        }
    }

    #[test]
    fn model_arch_is_the_loader_label_for_qwen4() {
        assert_eq!(
            hipfire_arch_qwen4::ARCH_ID,
            16,
            "arch_label(16) == \"qwen4\""
        );
        assert_eq!(MODEL_ARCH, "qwen4");
    }

    #[test]
    fn halo_mq3_is_not_the_default_route() {
        // Strix Halo (gfx1151) takes Redline by default only for `.mq4r`; the
        // gfx1201 Qwen4 exception does not apply to it.
        for path in [
            "/m/flash-next-halo.mq3",
            "/m/flash-next-ctl3-hot.hfq",
            "/m/x.mq4",
        ] {
            for has_drafter in [false, true] {
                assert!(
                    !route("gfx1151", path, has_drafter).default_route(),
                    "{path} drafter={has_drafter}"
                );
            }
        }
    }

    #[test]
    fn mq4r_default_arches_run_the_gate() {
        for gpu_arch in ["gfx1100", "gfx1151", "gfx1201"] {
            assert!(
                route(gpu_arch, "/m/flash-next.mq4r", false).default_route(),
                "{gpu_arch}"
            );
            assert!(
                route(gpu_arch, "/m/FLASH-NEXT.MQ4R", false).default_route(),
                "{gpu_arch} upper-case extension"
            );
        }
    }

    #[test]
    fn gfx1201_exception_runs_without_a_drafter_and_skips_with_one() {
        assert!(route("gfx1201", "/m/ctl3-hot.hfq", false).default_route());
        assert!(route("gfx1201", "/m/flash-next.mq3", false).default_route());
        assert!(!route("gfx1201", "/m/ctl3-hot.hfq", true).default_route());
        // An `.mq4r` is the extension route: a drafter does not take it off.
        assert!(route("gfx1201", "/m/flash-next.mq4r", true).default_route());
    }

    #[test]
    fn unsupported_arch_skips() {
        for gpu_arch in ["gfx1030", "gfx1100", "gfx1200", "gfx90a", "gfx942"] {
            assert!(
                !route(gpu_arch, "/m/ctl3-hot.hfq", false).default_route(),
                "{gpu_arch} .hfq"
            );
        }
        for gpu_arch in ["gfx1030", "gfx1200", "gfx90a", "gfx942"] {
            assert!(
                !route(gpu_arch, "/m/flash-next.mq4r", false).default_route(),
                "{gpu_arch} .mq4r"
            );
        }
    }

    #[test]
    fn multi_rank_expert_parallel_and_platform_refusal_skip() {
        let mut facts = route("gfx1201", "/m/ctl3-hot.hfq", false);
        assert!(facts.default_route());
        facts.ep = 2;
        assert!(!facts.default_route());
        facts.ep = 1;
        facts.pp = 2;
        assert!(!facts.default_route());
        facts.pp = 1;
        facts.tp = 2;
        assert!(!facts.default_route());
        facts.tp = 1;
        facts.platform_refusal = Some("uncertified platform".into());
        assert!(!facts.default_route());
    }

    #[test]
    fn skip_report_is_ok_checkless_and_names_the_exact_reason() {
        let args = parse_args(argv("--model /m/halo.mq3")).unwrap();
        let facts = route("gfx1151", "/m/halo.mq3", false);
        let report = skip_report(&Gate::default(), &facts, &args, "gfx1151", Instant::now());
        assert_eq!(report["gate"], "qwen4_g2_replay_gate");
        assert_eq!(report["ok"], true);
        assert_eq!(report["status"], "skip");
        assert_eq!(report["skipped"], true);
        assert_eq!(
            report["reason"],
            "Redline not on the default route (config.rs:15–22)"
        );
        assert_eq!(report["failed_checks"], json!([]));
        assert_eq!(report["checks"], json!([]), "no replay phase ran");
        assert_eq!(report["route"]["default_route"], false);
        assert_eq!(report["route"]["gpu_arch"], "gfx1151");
        assert_eq!(report["params"]["expert_vram_layers"], 20);
    }

    #[test]
    fn a_run_report_is_pass_or_fail_never_skip() {
        let args = parse_args(argv("--model /m/ctl3-hot.hfq")).unwrap();
        let mut gate = Gate::default();
        gate.count("replay_default_configured", 0, Value::Null);
        let ok = finish(
            &gate,
            &serde_json::Map::new(),
            &args,
            "gfx1201",
            Instant::now(),
        );
        assert_eq!(
            (ok["ok"].clone(), ok["status"].clone()),
            (json!(true), json!("pass"))
        );
        assert!(ok.get("skipped").is_none() && ok.get("reason").is_none());
        gate.count("positive_production_logits_exact", 3, Value::Null);
        let bad = finish(
            &gate,
            &serde_json::Map::new(),
            &args,
            "gfx1201",
            Instant::now(),
        );
        assert_eq!(
            (bad["ok"].clone(), bad["status"].clone()),
            (json!(false), json!("fail"))
        );
    }
}
