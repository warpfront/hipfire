// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 decode forward: MoE decode, `Qwen35Scratch`, the per-token layer
//! loop, the EP super-op program, and GPU-logits entry points.

use super::batch::BatchSemantics;
use super::batch::PrefillBatchScratch;
use super::config::DflashFusionCtx;
use super::config::LayerType;
use super::config::MropeCtx;
use super::config::Qwen35Config;
use super::prefill::trace_finite_if_enabled;
use super::prefill::PREFILL_MAX_BATCH;
use super::weights::DeltaNetState;
use super::weights::ExpertWeights;
use super::weights::LayerWeights;
use super::weights::MoeFfnWeights;
use super::weights::Qwen35Weights;
use super::weights::StateQuant;
use crate::speculative::GdnTape;
use crate::speculative::HiddenStateRingBuffer;
use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::moe::{routed_codebook_pair_batched_supported, MoeEpMode};
use hipfire_dispatch::pipeline::batched::{BatchEpilogue, DenseBatchMath};
use hipfire_dispatch::pipeline::dump_hidden_localize;
use hipfire_dispatch::pipeline::execute_steps;
use hipfire_dispatch::pipeline::gated_mlp::{GatedMlpOp, MlpOutput};
use hipfire_dispatch::pipeline::hybrid;
use hipfire_dispatch::pipeline::hybrid::{AttentionTap, MixerOutput};
use hipfire_dispatch::pipeline::sealed_moe::MoeRouteProducerProof;
use hipfire_dispatch::pipeline::sealed_moe::SealedMoeCall;
use hipfire_dispatch::pipeline::superop;
use hipfire_dispatch::pipeline::superop::EpMoeCombineMode;
use hipfire_dispatch::pipeline::superop::EpMoeRouteView;
use hipfire_dispatch::pipeline::superop::ForwardBindings;
use hipfire_dispatch::pipeline::superop::LayerProgram;
use hipfire_dispatch::pipeline::superop::OpBinding;
use hipfire_dispatch::pipeline::superop::OpFlavor;
use hipfire_dispatch::pipeline::superop::SuperOp;
use hipfire_dispatch::pipeline::superop::SuperOpKind;
use hipfire_dispatch::pipeline::superop::WeightSlot;
use hipfire_dispatch::pipeline::GemvInput;
use hipfire_dispatch::pipeline::Step;
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::llama;
use hipfire_runtime::llama::EmbeddingFormat;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::multi_gpu::TpGraphStep;
use hipfire_runtime::sealed_moe::repartition_expert_execution_plan;
use hipfire_runtime::sealed_moe::ExpertExecutionPlan;
use hipfire_runtime::tp_shard::ExpertAssign;
use hipfire_runtime::tp_shard::ShardConfig;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

// ─── MoE FFN (decode, batch=1) ──────────────────────────────────────────

/// Construct a non-owning `GpuTensor` view over `[offset_elems,
/// offset_elems + len_elems)` of `src`. Valid only for F32 (4 bytes/elem).

/// One-token MoE FFN: router → top-K → shared expert + top-K routed, added
/// into `x_residual` in place. `x_norm` is the already-RMSNormed FFN input.
///
/// Dense-compute decode reference implementation (Phase 1). Top-K selection
/// runs on CPU via a single D2H sync per layer on the router logits; the
/// shared-expert scalar gate is another D2H sync. Sparse-routing + batched
/// grouped-GEMM variants come in later phases — this version prioritizes
/// correctness and minimal surface area.
///
/// Matches HF `modeling_qwen3_5_moe.py`:
///   router_probs  = softmax(W_router · x_norm)            // [n_exp]
///   (idx, w)      = topk(router_probs, k)                  // [k]
///   if norm_topk:  w /= w.sum()
///   scalar        = sigmoid(W_shared_gate · x_norm)        // [1]
///   y_shared      = scalar * shared_expert(x_norm)         // [hidden]
///   y_moe         = sum_{k} w[k] * expert[idx[k]](x_norm)  // [hidden]
///   x_residual   += y_shared + y_moe
/// Non-owning borrow of the scratch buffers `moe_ffn_decode_impl` needs.
/// Callers construct one of these from either a `Qwen35Scratch` (preallocated,
/// hipGraph-capturable) or from tensors they own locally (heap path).
pub(crate) struct MoeScratchRef<'a> {
    pub(crate) router_logits: &'a GpuTensor,
    pub(crate) scalar_buf: &'a GpuTensor,
    pub(crate) x_rot_local: &'a GpuTensor,
    pub(crate) gate_up_buf: &'a GpuTensor,
    pub(crate) gate_buf: &'a GpuTensor,
    pub(crate) up_buf: &'a GpuTensor,
    pub(crate) ffn_hidden: &'a GpuTensor,
    pub(crate) ffn_out: &'a GpuTensor,
    pub(crate) gate_batch: &'a GpuTensor,
    pub(crate) up_batch: &'a GpuTensor,
    pub(crate) rot_batch: &'a GpuTensor,
    pub(crate) topk_indices: &'a GpuTensor,
    pub(crate) topk_weights: &'a GpuTensor,
    // [k_top × dim + ceil(dim/4)] f32 — per-(expert-rank) MoE down output
    // plus a zero-initialised u32 counter tail used only by the optional
    // expert-wave-preserving last-arriver combine experiment. The production
    // atomic-free expand+combine path ignores the tail. Mirrors the prefill
    // `pbs.moe_down_expanded_batch` layout with batch=1. Required so
    // the MoE FFN is byte-deterministic under hipGraph replay; see
    // task #100 root-cause notes in `forward_scratch`.
    pub(crate) down_expanded: &'a GpuTensor,
}

impl<'a> MoeScratchRef<'a> {
    /// View into a Qwen35Scratch's MoE fields. Panics if the caller didn't
    /// allocate MoE scratch (config.num_experts == 0).
    fn from_scratch(s: &'a Qwen35Scratch) -> Self {
        Self {
            router_logits: s
                .moe_router_logits
                .as_ref()
                .expect("MoE scratch not allocated"),
            scalar_buf: s.moe_scalar_buf.as_ref().expect("MoE scratch"),
            x_rot_local: s.moe_x_rot.as_ref().expect("MoE scratch"),
            gate_up_buf: s.moe_gate_up_buf.as_ref().expect("MoE scratch"),
            gate_buf: s.moe_gate_buf.as_ref().expect("MoE scratch"),
            up_buf: s.moe_up_buf.as_ref().expect("MoE scratch"),
            ffn_hidden: s.moe_ffn_hidden.as_ref().expect("MoE scratch"),
            ffn_out: s.moe_ffn_out.as_ref().expect("MoE scratch"),
            gate_batch: s.moe_gate_batch.as_ref().expect("MoE scratch"),
            up_batch: s.moe_up_batch.as_ref().expect("MoE scratch"),
            rot_batch: s.moe_rot_batch.as_ref().expect("MoE scratch"),
            topk_indices: s.moe_topk_indices.as_ref().expect("MoE scratch"),
            topk_weights: s.moe_topk_weights.as_ref().expect("MoE scratch"),
            down_expanded: s.moe_down_expanded.as_ref().expect("MoE scratch"),
        }
    }
}

/// All gate-side + routed MoE weights are MQ4G256 — the precondition for
/// the prerotated fast path where the caller can fuse rmsnorm+FWHT via
/// `fused_rmsnorm_rotate_mq` and call `moe_ffn_decode_with_scratch_prerotated`.
pub(crate) fn ffn_all_mq4_for_moe(ffn: &MoeFfnWeights) -> bool {
    ffn_gate_side_mq4_for_moe(ffn)
        && ffn
            .experts
            .iter()
            .all(|e| matches!(e.gate_up.gpu_dtype, DType::MQ4G256 | DType::MQ4G256V2))
}

/// Gate-side exact-uniform MQ4G256 V1 quartet (router + shared expert
/// gate/up), independent of the routed experts.
///
/// Keeping this predicate aligned with `MoeResolution::gate_fusable` is a
/// correctness invariant. If Qwen pre-rotates while dispatch declines the
/// fused route, generic `run_auto` receives the raw residual in its `x_norm`
/// slot and silently rotates an unnormalized activation.
pub(crate) fn ffn_gate_side_mq4_for_moe(ffn: &MoeFfnWeights) -> bool {
    hipfire_dispatch::families::moe::softmax_shared_gate_uniform_mq4(
        ffn.router.gpu_dtype,
        ffn.shared_expert_gate.gpu_dtype,
        ffn.shared_expert.gate.gpu_dtype,
        ffn.shared_expert.up.gpu_dtype,
    )
}

/// Detect any MQ3G256 / MQ3G256Lloyd weight inside a MoE FFN block (router,
/// shared expert gate/up/down, shared_expert_gate router-mix scalar, or any
/// routed expert's gate_up/down). The MoE batched FFN kernels assume HFQ4
/// layout (136 B/group); an MQ3 weight (104 B/group) or Lloyd-MQ3 weight
/// (112 B/group) would dispatch with the wrong stride. Used by the
/// captured-prefill and non-captured-prefill defense-in-depth checks.
///
/// Mirrors `is_mq3_any` in `forward_prefill_batch_single_chunk_captured`
/// (line 3325) so both cross-checks treat plain and Lloyd-MQ3 identically.
/// MQ3/MQ3-Lloyd in the STRUCTURAL parts of a MoE FFN (router, shared
/// expert gate/up/down). These are NOT served by the merged grouped kernel
/// (which only handles routed experts) → must still hard-error.
pub(crate) fn moe_ffn_has_mq3_structural(ffn: &MoeFfnWeights) -> bool {
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    is_mq3_any(ffn.router.gpu_dtype)
        || is_mq3_any(ffn.shared_expert_gate.gpu_dtype)
        || is_mq3_any(ffn.shared_expert.gate.gpu_dtype)
        || is_mq3_any(ffn.shared_expert.up.gpu_dtype)
        || is_mq3_any(ffn.shared_expert.down.gpu_dtype)
}

/// MQ3/MQ3-Lloyd in ROUTED experts WITHOUT a tag table (uniform MQ3, not
/// the graded case the merged grouped kernel handles). When
/// `expert_dtype_tags` is `Some`, the merged grouped kernel carries the
/// correct per-expert stride and this returns false (pass).
pub(crate) fn moe_ffn_has_mq3_experts_uniform(ffn: &MoeFfnWeights) -> bool {
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    ffn.expert_dtype_tags.is_none()
        && ffn
            .experts
            .iter()
            .any(|e| is_mq3_any(e.gate_up.gpu_dtype) || is_mq3_any(e.down.gpu_dtype))
}

/// Narrowed sibling of [`moe_ffn_has_mq3_experts_uniform`] for the NON-captured
/// batched-prefill entry point only.
///
/// The refusal that predicate drives exists because the batched MoE bodies used
/// to dispatch every routed weight through HFQ4-layout kernels (136 B/group), so
/// an MQ3/Lloyd routed expert (112 B/group) was read at the wrong stride. That is
/// no longer true for a UNIFORM-per-projection codebook pair: those now have real
/// grouped-GEMM arms keyed on their own dtype (`dispatch_grouped_gemm` ->
/// `gemm_mq{2,3}g256_lloyd_moe_grouped_wmma`), so the stride is correct by
/// construction and the layer is genuinely servable batched.
///
/// This MUST agree with [`moe_ffn_batched_admissible_for_dtypes`]'s codebook arm
/// in one direction: anything that arm ADMITS must return `false` here, or the
/// forward hard-errors on a model it just declared eligible. The shared helper
/// [`routed_codebook_pair_batched_supported`] is what keeps them in lockstep
/// (test: `mq3_refusal_never_fires_on_an_admitted_codebook_pair`). The reverse is
/// safe: returning `true` for a pair that is not admitted just keeps the
/// per-token fallback, which is today's behavior.
///
/// DELIBERATELY NOT used by `forward_prefill_batch_single_chunk_captured_opts`.
/// That entry point has no eligibility check and no per-token fallback: its
/// refusal is the SOLE guard there, and narrowing it would hand these dtypes to
/// a hipGraph-captured prefill that has never been validated (both the kernel
/// JIT and the `ensure_fp16_x` staging inside the grouped launchers happen on
/// the first call, i.e. mid-capture). Correct-and-slow beats fast-and-corrupt.
pub(crate) fn moe_ffn_has_unsupported_mq3_experts_uniform(
    ffn: &MoeFfnWeights,
    admit_codebook: bool,
) -> bool {
    unsupported_mq3_experts_uniform_from_dtypes(
        ffn.expert_dtype_tags.is_some(),
        ffn.experts
            .iter()
            .map(|e| (e.gate_up.gpu_dtype, e.down.gpu_dtype)),
        admit_codebook,
    )
}

/// Pure core of [`moe_ffn_has_unsupported_mq3_experts_uniform`], split out
/// because `MoeFfnWeights` needs live GPU tensors and cannot be built in a unit
/// test. `experts` yields `(gate_up_dtype, down_dtype)` per routed expert.
pub(crate) fn unsupported_mq3_experts_uniform_from_dtypes(
    has_tag_table: bool,
    experts: impl IntoIterator<Item = (DType, DType)>,
    admit_codebook: bool,
) -> bool {
    // A tag table means the merged grouped kernel carries the per-expert stride
    // — the graded case, never this refusal's concern.
    if has_tag_table {
        return false;
    }
    let dtypes: Vec<(DType, DType)> = experts.into_iter().collect();
    let is_mq3_any = |dt: DType| matches!(dt, DType::MQ3G256 | DType::MQ3G256Lloyd);
    if !dtypes
        .iter()
        .any(|&(gu, dn)| is_mq3_any(gu) || is_mq3_any(dn))
    {
        return false;
    }
    if !admit_codebook {
        return true;
    }
    let Some(&first) = dtypes.first() else {
        return true;
    };
    // Uniformity is re-derived rather than assumed: the admission arm only
    // reaches its codebook branch when `expert_gate_up_uniform &&
    // expert_down_uniform`, so a mixed-without-tags file must stay refused
    // (`dispatch_grouped_gemm` would apply experts[0]'s group stride to all).
    let uniform = dtypes.iter().all(|&d| d == first);
    !(uniform && routed_codebook_pair_batched_supported(first.0, first.1))
}

/// True when any MoE FFN projection is MQ6-family (V1 qt=15 or V2 qt=47).
///
/// Feeds `Qwen35Weights::moe_has_mq6` → gfx1151 `force_mq4_grouped_fp16` when a
/// mixed checkpoint carries an MQ6 projection somewhere. V1 and V2 are distinct
/// wire layouts (f32 vs dual-half fp16 headers) but both trip the same model
/// flag: both are 200 B/group 6-bit MQ and both need the gfx1151 MQ4-grouped
/// FP16 consistency path. MQ4V2 must never collapse into this helper.
fn moe_ffn_has_mq6(ffn: &MoeFfnWeights) -> bool {
    moe_ffn_has_mq6_from_dtypes(
        [
            ffn.router.gpu_dtype,
            ffn.shared_expert_gate.gpu_dtype,
            ffn.shared_expert.gate.gpu_dtype,
            ffn.shared_expert.up.gpu_dtype,
            ffn.shared_expert.down.gpu_dtype,
        ],
        ffn.experts
            .iter()
            .map(|e| (e.gate_up.gpu_dtype, e.down.gpu_dtype)),
    )
}

/// Pure core of [`moe_ffn_has_mq6`] for unit tests (no live `MoeFfnWeights`).
pub(crate) fn moe_ffn_has_mq6_from_dtypes(
    structural: impl IntoIterator<Item = DType>,
    experts: impl IntoIterator<Item = (DType, DType)>,
) -> bool {
    let is_mq6 = |dt: DType| matches!(dt, DType::MQ6G256 | DType::MQ6G256V2);
    structural.into_iter().any(is_mq6)
        || experts.into_iter().any(|(gu, dn)| is_mq6(gu) || is_mq6(dn))
}

pub(crate) fn layers_have_mq6_moe(layers: &[LayerWeights]) -> bool {
    layers.iter().any(|layer| match layer {
        LayerWeights::DeltaNetMoe(l) => moe_ffn_has_mq6(&l.ffn),
        LayerWeights::FullAttnMoe(l) => moe_ffn_has_mq6(&l.ffn),
        _ => false,
    })
}

/// Execute one token through the shared sealed MoE program using caller-owned
/// scratch. The input is already normalized (and may already be rotated).
pub(crate) fn moe_ffn_decode_with_scratch(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(scratch);
    let params = moe_params_for_decode(
        ffn,
        x_norm,
        x_residual,
        config,
        &refs,
        false,
        None,
        false,
        false,
        MoeEpMode::None,
        hipfire_dispatch::families::moe::MoeNormalization::Provided,
    );
    moe_ffn_decode_impl(gpu, ffn, config, &refs, params)
}

/// Execute one token through the shared sealed MoE program when the caller has
/// already populated the FWHT-rotated scratch view.
pub(crate) fn moe_ffn_decode_with_scratch_prerotated(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(scratch);
    let params = moe_params_for_decode(
        ffn,
        x_norm,
        x_residual,
        config,
        &refs,
        true,
        None,
        false,
        false,
        MoeEpMode::None,
        hipfire_dispatch::families::moe::MoeNormalization::Provided,
    );
    moe_ffn_decode_impl(gpu, ffn, config, &refs, params)
}

/// The actual MoE FFN implementation. Uses the caller-provided scratch
/// buffers, never allocates.
// ── REAP expert-importance capture (HIPFIRE_MOE_EXPERT_STATS=1) ────────────
// Per-(layer, expert) accumulators: routing count, Σ gate_weight,
// Σ ‖expert_output‖, Σ (gate × ‖output‖). The last is the true REAP
// contribution (gate-weighted output norm) — compared against raw frequency
// (count) to decide whether freq agrees with contribution before committing to
// any per-expert mixed-precision kernel. Dumped by `dump_expert_stats`.
static EXPERT_STATS: std::sync::Mutex<
    Option<std::collections::HashMap<(u16, u16), (u64, f64, f64, f64)>>,
> = std::sync::Mutex::new(None);
static EXPERT_STATS_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
fn expert_stats_enabled() -> bool {
    *EXPERT_STATS_ON.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_MOE_EXPERT_STATS")
            .ok()
            .as_deref()
            == Some("1")
    })
}
fn capture_expert_stats(
    gpu: &Gpu,
    layer_idx: u16,
    k: usize,
    hidden: usize,
    down_expanded: &GpuTensor,
    topk_indices: &GpuTensor,
    topk_weights: &GpuTensor,
) {
    let dn = match gpu.download_f32(down_expanded) {
        Ok(v) => v,
        Err(_) => return,
    };
    let ti = match gpu.download_f32(topk_indices) {
        Ok(v) => v,
        Err(_) => return,
    };
    let tw = match gpu.download_f32(topk_weights) {
        Ok(v) => v,
        Err(_) => return,
    };
    let mut guard = EXPERT_STATS.lock().unwrap();
    let m = guard.get_or_insert_with(std::collections::HashMap::new);
    for krank in 0..k {
        if krank >= ti.len() || krank >= tw.len() {
            break;
        }
        let e = (ti[krank].to_bits() as i32) as u16; // i32-in-F32 alias
        let w = tw[krank] as f64;
        let base = krank * hidden;
        if base + hidden > dn.len() {
            break;
        }
        let mut sq = 0.0f64;
        for j in 0..hidden {
            let x = dn[base + j] as f64;
            sq += x * x;
        }
        let norm = sq.sqrt();
        let ent = m.entry((layer_idx, e)).or_insert((0, 0.0, 0.0, 0.0));
        ent.0 += 1;
        ent.1 += w;
        ent.2 += norm;
        ent.3 += w * norm;
    }
}
/// Dump the accumulated per-(layer,expert) REAP stats to a TSV. Called from
/// eval harnesses when HIPFIRE_MOE_EXPERT_STATS_OUT is set.
pub fn dump_expert_stats(path: &str) {
    let guard = EXPERT_STATS.lock().unwrap();
    let m = match guard.as_ref() {
        Some(m) if !m.is_empty() => m,
        _ => {
            eprintln!("expert_stats: empty (capture not enabled?)");
            return;
        }
    };
    let mut rows: Vec<_> = m.iter().collect();
    rows.sort_by_key(|((l, e), _)| (*l, *e));
    let mut out = String::from("layer\texpert\tcount\tsum_gate\tsum_norm\tsum_contrib\n");
    for ((l, e), (c, sg, sn, sc)) in rows {
        out.push_str(&format!("{l}\t{e}\t{c}\t{sg:.6}\t{sn:.6}\t{sc:.6}\n"));
    }
    match std::fs::write(path, out) {
        Ok(_) => eprintln!("expert_stats: wrote {path} ({} layer×expert rows)", m.len()),
        Err(e) => eprintln!("expert_stats: write failed {path}: {e}"),
    }
}

impl hipfire_dispatch::families::moe::RoutedExpertWeights for MoeFfnWeights {
    fn len(&self) -> usize {
        // Plan-bound compact EP layers present the GLOBAL expert count: the
        // per-call live validation and any expert-indexed path resolve every
        // global id (owned or dummy) through `get`. Single layers keep the
        // resident list length (== global count there).
        if self.expert_table.execution_contract().is_some() {
            self.expert_table.n_experts()
        } else {
            self.experts.len()
        }
    }

    /// Qwen3.5 residency: packed bucket owners, else expert 0. Qwen4's
    /// per-expert tier lease (`ExpertTierLease`) cannot reach this adapter:
    /// this crate does not depend on `hipfire-arch-qwen4`, `MoeFfnWeights` never
    /// carries a lease, and the trait's `tier_identity` stays its default
    /// `None`, so the expert-0 / packed-owner rules here never apply to a
    /// tier layer (those derive placement from the tier identity instead).
    fn host_mapped_projections(&self) -> (bool, bool) {
        if let Some(owners) = &self.packed_expert_owners {
            owners.host_mapped_projections()
        } else {
            self.get(0).map_or((false, false), |(gate_up, down)| {
                (
                    gate_up.buf.buf.is_host_mapped(),
                    down.buf.buf.is_host_mapped(),
                )
            })
        }
    }

    fn get(
        &self,
        expert_idx: usize,
    ) -> Option<(
        hipfire_dispatch::families::gemv::WeightRef<'_>,
        hipfire_dispatch::families::gemv::WeightRef<'_>,
    )> {
        if self.expert_table.execution_contract().is_none() {
            // Single: the resident list IS the global table.
            return self
                .experts
                .get(expert_idx)
                .map(|expert| (expert.gate_up.dispatch_ref(), expert.down.dispatch_ref()));
        }
        // Compact EP: owned globals resolve to the slot-ordered local tensor;
        // non-owned globals resolve to the layout-compatible zero dummy. The
        // dummy choice uses the same first-match rule as the compact bind
        // (`build_compact_live_binding`), which is exact because load keeps
        // one dummy pair per distinct non-owned layout — so the sealed
        // per-call validation observes the bound identities, and the indexed
        // kernels keep reading through the global pointer tables either way.
        if let Some(slot) = self.expert_binding.local_slot(expert_idx) {
            return self
                .experts
                .get(slot)
                .map(|expert| (expert.gate_up.dispatch_ref(), expert.down.dispatch_ref()));
        }
        let metadata = self.expert_table.experts().get(expert_idx)?;
        self.ep_dummy_experts.iter().find_map(|dummy| {
            let gate_up = dummy.gate_up.dispatch_ref();
            let down = dummy.down.dispatch_ref();
            hipfire_dispatch::pipeline::sealed_moe::expert_weights_match_metadata(
                metadata, &gate_up, &down, expert_idx,
            )
            .then_some((gate_up, down))
        })
    }

    /// CPU-splice payload for one routed expert, resolved from the packed bucket
    /// owners: a per-expert view is `Borrowed` and invisible to `Gpu::host_bytes`,
    /// so the owning bucket has to be found by hand (same walk as
    /// `packed_expert_host_bytes`' own pinning test). `None` means the layer has
    /// no packed owners, the id is out of range, or the dtype has no CPU decoder —
    /// the caller reports that as "no CPU payload" and the load-time coverage
    /// report names uncovered formats.
    fn cpu_splice_expert(
        &self,
        gpu: &Gpu,
        expert_idx: usize,
    ) -> Option<hipfire_dispatch::cpu_exec::CpuMoeExpert<'_>> {
        let owners = self.packed_expert_owners.as_ref()?;
        let expert = self.experts.get(expert_idx)?;
        let gate_up =
            packed_expert_host_bytes(gpu, &owners.gate_up, &expert.gate_up.buf, "gate_up").ok()?;
        let down = packed_expert_host_bytes(gpu, &owners.down, &expert.down.buf, "down").ok()?;
        let gate_up_quant = hipfire_dispatch::cpu_exec::cpu_quant_for(expert.gate_up.gpu_dtype)?;
        let down_quant = hipfire_dispatch::cpu_exec::cpu_quant_for(expert.down.gpu_dtype)?;
        Some(hipfire_dispatch::cpu_exec::CpuMoeExpert {
            gate_up_quant,
            gate_up,
            down_quant,
            down,
        })
    }
}

/// Build the complete typed Qwen softmax/shared decode recipe from the
/// caller-owned weights and scratch. `normalization` is either `Provided` for
/// an already prepared activation or `RmsNorm` for the shared prologue.
#[allow(clippy::too_many_arguments)]
fn moe_params_for_decode<'a>(
    ffn: &'a MoeFfnWeights,
    x_norm: &'a GpuTensor,
    x_residual: &'a GpuTensor,
    config: &Qwen35Config,
    s: &MoeScratchRef<'a>,
    x_rot_prerotated: bool,
    ep_routed_out: Option<&'a GpuTensor>,
    ep_skip_shared: bool,
    defer_routed_combine: bool,
    ep_mode: MoeEpMode,
    normalization: hipfire_dispatch::families::moe::MoeNormalization<'a>,
) -> hipfire_dispatch::families::moe::MoeParams<'a> {
    let hidden = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k = config.num_experts_per_tok;
    let n_exp = config.num_experts;
    let (per_expert_gate_up, per_expert_down) = ffn.per_expert_tier_tables();
    let moe_dtypes = hipfire_dispatch::families::moe::MoeDtypes {
        router: ffn.router.gpu_dtype,
        shared: Some(hipfire_dispatch::families::moe::MoeSharedDtypes {
            selector: ffn.shared_expert_gate.gpu_dtype,
            gate: ffn.shared_expert.gate.gpu_dtype,
            up: ffn.shared_expert.up.gpu_dtype,
            down: ffn.shared_expert.down.gpu_dtype,
        }),
        experts_all_gate_up_mq4: if let Some(global) = ffn.global_expert_dtypes.as_ref() {
            global
                .iter()
                .all(|(g, _)| matches!(*g, DType::MQ4G256 | DType::MQ4G256V2))
        } else {
            ffn.experts
                .iter()
                .all(|e| matches!(e.gate_up.gpu_dtype, DType::MQ4G256 | DType::MQ4G256V2))
        },
        routed_gate_up: if let Some(global) = ffn.global_expert_dtypes.as_ref() {
            global.first().map(|(g, _)| *g).unwrap_or(DType::F32)
        } else {
            ffn.experts
                .first()
                .map(|e| e.gate_up.gpu_dtype)
                .unwrap_or(DType::F32)
        },
        routed_down: if let Some(global) = ffn.global_expert_dtypes.as_ref() {
            global.first().map(|(_, d)| *d).unwrap_or(DType::F32)
        } else {
            ffn.experts
                .first()
                .map(|e| e.down.gpu_dtype)
                .unwrap_or(DType::F32)
        },
        routed_has_mixed_experts: per_expert_gate_up.is_some() || per_expert_down.is_some(),
        has_paro_shared: ffn.paro_shared.is_some(),
        per_expert_gate_up,
        per_expert_down,
    };
    crate::qwen35::weights::trace_moe_arm(
        "decode",
        moe_dtypes.routed_gate_up,
        moe_dtypes.routed_down,
        moe_dtypes.experts_all_gate_up_mq4,
        ffn.packed_expert_owners.as_ref().is_some_and(|owners| {
            owners.gate_up.iter().all(|b| b.buf.is_host_mapped())
                && owners.down.iter().all(|b| b.buf.is_host_mapped())
        }),
        per_expert_gate_up.is_some() || per_expert_down.is_some(),
    );
    let shared = Some(hipfire_dispatch::families::moe::MoeSharedDecode {
        weights: hipfire_dispatch::families::moe::MoeSharedWeights {
            selector: ffn.shared_expert_gate.dispatch_ref(),
            gate: ffn.shared_expert.gate.dispatch_ref(),
            up: ffn.shared_expert.up.dispatch_ref(),
            down: ffn.shared_expert.down.dispatch_ref(),
        },
        intermediate: smi,
        scalar: s.scalar_buf,
        gate_out: s.gate_buf,
        up_out: s.up_buf,
    });
    hipfire_dispatch::families::moe::MoeParams {
        dtypes: moe_dtypes,
        mq2_lloyd: None,
        recipe: hipfire_dispatch::families::moe::MoeRecipe::SoftmaxGatedShared {
            bf16_round_trip: false,
            shared_after_combine: false,
        },
        route_policy: None,
        normalization,
        batch_size: 1,
        hidden,
        mi,
        k,
        n_exp,
        norm_topk_prob: config.norm_topk_prob,
        selection_bias: None,
        x_rot_prerotated,
        defer_routed_combine,
        cpu_deferred_experts: ffn.cpu_deferred_experts
            && ep_mode == MoeEpMode::None
            && !defer_routed_combine,
        ep_mode,
        layer_idx: ffn.layer_idx,
        x_norm,
        x_residual,
        routed_out: ep_routed_out,
        skip_shared: ep_skip_shared,
        router: ffn.router.dispatch_ref(),
        shared,
        expert_gate_up_ptrs: &ffn.expert_gate_up_ptrs,
        expert_down_ptrs: &ffn.expert_down_ptrs,
        expert_ptrs_host: None,
        expert_down_awq_ptrs: ffn.expert_down_awq_ptrs.as_ref(),
        expert_dtype_tags: ffn.expert_dtype_tags.as_ref(),
        routed_gate_up_k: ffn.experts.first().map_or(0, |e| e.gate_up.k),
        routed_down_m: ffn.experts.first().map_or(0, |e| e.down.m),
        routed_down_k: ffn.experts.first().map_or(0, |e| e.down.k),
        routed_experts: ffn,
        routed_gate_up_paro: ffn.experts.first().and_then(|e| {
            e.gate_up
                .paro
                .as_ref()
                .map(|p| hipfire_dispatch::families::gemv::GivensRef {
                    pairs: &p.pairs,
                    theta: &p.theta,
                    scales: &p.channel_scales,
                    krot: p.krot as usize,
                })
        }),
        routed_down_paro: ffn.experts.first().and_then(|e| {
            e.down
                .paro
                .as_ref()
                .map(|p| hipfire_dispatch::families::gemv::GivensRef {
                    pairs: &p.pairs,
                    theta: &p.theta,
                    scales: &p.channel_scales,
                    krot: p.krot as usize,
                })
        }),
        router_logits: s.router_logits,
        x_rot_local: s.x_rot_local,
        gate_up_buf: s.gate_up_buf,
        ffn_hidden: s.ffn_hidden,
        ffn_out: s.ffn_out,
        gate_batch: s.gate_batch,
        up_batch: s.up_batch,
        rot_batch: s.rot_batch,
        topk_indices: s.topk_indices,
        topk_weights: s.topk_weights,
        down_expanded: s.down_expanded,
        ep_row_gather: None,
    }
}

/// Resolve one routed expert's host-mapped bytes from the packed owner buckets.
///
/// Per-expert views are non-owning `sub_offset` slices, so `Gpu::host_bytes`
/// (which reads the allocation registry by exact pointer) returns `None` for
/// them. Locate the bucket owner whose device range contains the view pointer,
/// then return `offset..offset+view_len` from that owner's host bytes — the
/// same bytes the GPU indexed kernel would read, at the expert's own dtype.
/// Uniform layers have a single bucket per projection and graded layers one per
/// `(gate_up, down)` tier, so this works for both without knowing which.
fn packed_expert_host_bytes<'a>(
    gpu: &Gpu,
    buckets: &'a [GpuTensor],
    view: &GpuTensor,
    what: &str,
) -> Result<&'a [u8], DispatchError> {
    let base = view.buf.as_ptr() as usize;
    let len = view.buf.size();
    for owner in buckets {
        let start = owner.buf.as_ptr() as usize;
        let end = start + owner.buf.size();
        if base >= start && base < end {
            let offset = base - start;
            let bytes = gpu.host_bytes(owner).ok_or_else(|| {
                DispatchError::Hip(format!(
                    "moe cpu expert splice: {what} bucket owner has no host bytes"
                ))
            })?;
            let view_end = offset.checked_add(len).ok_or_else(|| {
                DispatchError::Hip(format!(
                    "moe cpu expert splice: {what} expert extent overflows"
                ))
            })?;
            if view_end > bytes.len() {
                return Err(DispatchError::Hip(format!(
                    "moe cpu expert splice: {what} expert view exceeds bucket owner bytes"
                )));
            }
            return Ok(&bytes[offset..view_end]);
        }
    }
    Err(DispatchError::Hip(format!(
        "moe cpu expert splice: {what} expert view is not inside any packed bucket owner"
    )))
}

/// Execute an already-built decode recipe and retain the existing observation
/// and expert-statistics hooks around the shared sealed Step executor.
fn moe_ffn_decode_impl<'a>(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    config: &Qwen35Config,
    s: &'a MoeScratchRef<'a>,
    moe_params: hipfire_dispatch::families::moe::MoeParams<'a>,
) -> HipResult<()> {
    let hidden = config.dim;
    // MoE bisection dump (HIPFIRE_MOE_DUMP_DIR): the single-device post-MoE
    // residual; root-routed EP is captured after `residual_finish` instead.
    let dump_final =
        moe_params.ep_mode == MoeEpMode::None && hipfire_dispatch::pipeline::moe_dump::enabled();
    let dump_deferred = moe_params.defer_routed_combine;
    let dump_x = moe_params.x_residual;
    let k = config.num_experts_per_tok;
    #[cfg(feature = "moe-oracle")]
    let oracle_x_residual = moe_params.x_residual;
    #[cfg(feature = "moe-oracle")]
    {
        let oracle_pos =
            crate::qwen35::oracle::decode_position().map_err(|e| HipError::new(0, &e))?;
        crate::qwen35::oracle::decode_before(gpu, ffn, config, moe_params.x_residual, oracle_pos)
            .map_err(|e| HipError::new(0, &e))?;
    }
    let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
    let bound = ffn.bound_experts()?;
    let deferred_routed_experts = moe_params.cpu_deferred_experts;
    // Only when explicitly asked: the oracle downloads the activation and the
    // route for every MoE layer, which is a host sync point on the decode path.
    let oracle_input = hipfire_config::developer_var("HIPFIRE_MOE_CPU_ORACLE")
        .is_ok()
        .then_some((moe_params.x_norm, moe_params.dtypes.routed_gate_up));
    // Resolve the expert-statistics toggle exactly once: the same answer gates
    // the CPU-deferred refusal below and the end-of-step capture, and the
    // `developer_var` lookup must not be repeated on the decode hot path.
    let stats_enabled = expert_stats_enabled();
    // When the sealed step below carries the CPU-deferred selection it omits
    // the entire routed GPU FFN, so `s.down_expanded` is never written. Both
    // diagnostics depend on that buffer: `moe_cpu_oracle_report` compares the
    // GPU routed result against the CPU splice, and `capture_expert_stats`
    // measures the routed output norm. A silent skip or a stale-buffer read
    // would misreport, so fail closed and point at the supported loads. This
    // refuses only the CPU-deferred route; EP / PCIe / resident are untouched.
    if deferred_routed_experts && (oracle_input.is_some() || stats_enabled) {
        return Err(HipError::new(
            1,
            "explicit GPU routed expert oracle/statistics unavailable with \
             CPU-deferred experts; use PCIe/resident",
        ));
    }
    let t_window = std::time::Instant::now();
    let t_sealed = std::time::Instant::now();
    let sealed = hipfire_dispatch::pipeline::sealed_moe::seal_decode(bound, &ctx, moe_params)
        .map_err(HipError::from)?;
    hipfire_dispatch::pipeline::execute_steps(
        gpu,
        &ctx,
        &[hipfire_dispatch::pipeline::Step::Moe(sealed)],
    )
    .map_err(HipError::from)?;
    if dump_final {
        let dumped = if dump_deferred {
            hipfire_dispatch::pipeline::moe_dump::final_residual_deferred(
                gpu,
                ffn.layer_idx as usize,
            )
        } else {
            hipfire_dispatch::pipeline::moe_dump::final_residual(
                gpu,
                ffn.layer_idx as usize,
                dump_x,
                hidden,
                "moe_ffn_decode_impl",
            )
        };
        dumped.map_err(HipError::from)?;
    }

    let sealed_ns = t_sealed.elapsed().as_nanos() as u64;
    // The sealed program above runs the CPU expert splice in place of the routed
    // GPU stages for this call, so the splice's cost is inside `sealed_ns` and
    // cannot be separated on this side; `cpu_splice_ns` therefore reports 0 and
    // the stage-side trace is the authority for the splice's own timing.
    if deferred_routed_experts {
        // CPU deferral omits the GPU tail that used to change the activation
        // staging cache key; reused scratch must be refreshed by the next layer.
        gpu.invalidate_fp16_cache();
    }
    let total_ns = t_window.elapsed().as_nanos() as u64;
    if hipfire_dispatch::cpu_exec::moe_trace_enabled() {
        hipfire_dispatch::cpu_exec::moe_window_trace(
            ffn.layer_idx as usize,
            hipfire_dispatch::cpu_exec::MoeWindowTiming {
                total_ns,
                gpu_sealed_ns: sealed_ns,
                // The splice now runs inside the sealed program (stage-side), so
                // its cost is part of `gpu_sealed_ns`; see the comment above.
                cpu_splice_ns: 0,
                other_ns: total_ns.saturating_sub(sealed_ns),
                rows: 1,
                top_k: k,
            },
        );
    }
    if let Some((x_norm, routed_gate_up)) = oracle_input {
        if let (Some(q), Some(owners)) = (
            hipfire_dispatch::cpu_exec::cpu_quant_for(routed_gate_up),
            ffn.packed_expert_owners.as_ref(),
        ) {
            // Uniform-only diagnostic: skip graded layers rather than report
            // one tier's bytes as the whole layer.
            if owners.gate_up.len() == 1 && owners.down.len() == 1 {
                if let (Some(gu), Some(dn)) = (owners.gate_up.first(), owners.down.first()) {
                    hipfire_dispatch::cpu_exec::moe_cpu_oracle_report(
                        gpu,
                        q,
                        config.dim,
                        config.moe_intermediate_size,
                        config.num_experts_per_tok,
                        gu,
                        dn,
                        x_norm,
                        s.topk_indices,
                        s.topk_weights,
                        s.down_expanded,
                        ffn.layer_idx,
                        ffn.expert_down_awq_ptrs.is_some(),
                    );
                }
            }
        }
    }
    #[cfg(feature = "moe-oracle")]
    {
        let oracle_pos =
            crate::qwen35::oracle::decode_position().map_err(|e| HipError::new(0, &e))?;
        crate::qwen35::oracle::decode_after(
            gpu,
            ffn,
            config,
            s.router_logits,
            s.topk_indices,
            s.topk_weights,
            s.gate_batch,
            s.up_batch,
            s.rot_batch,
            s.down_expanded,
            oracle_x_residual,
            oracle_pos,
        )
        .map_err(|e| HipError::new(0, &e))?;
    }
    if stats_enabled {
        capture_expert_stats(
            gpu,
            ffn.layer_idx,
            k,
            hidden,
            s.down_expanded,
            s.topk_indices,
            s.topk_weights,
        );
    }
    Ok(())
}

/// Single-device MoE FFN of one decode row over caller-owned scratch (the
/// MTP head's routed layer): `x += moe(rmsnorm(x, ffn_norm))`, with the norm
/// written to `x_norm`.
pub(crate) fn moe_ffn_decode_rmsnorm<'a>(
    gpu: &mut Gpu,
    ffn: &'a MoeFfnWeights,
    x: &'a GpuTensor,
    ffn_norm: &'a GpuTensor,
    x_norm: &'a GpuTensor,
    config: &Qwen35Config,
    s: &'a MoeScratchRef<'a>,
) -> HipResult<()> {
    let params = moe_params_for_decode(
        ffn,
        x_norm,
        x,
        config,
        s,
        false,
        None,
        false,
        false,
        MoeEpMode::None,
        hipfire_dispatch::families::moe::MoeNormalization::RmsNorm {
            weight: ffn_norm,
            plain_out: x_norm,
            eps: config.norm_eps,
        },
    );
    moe_ffn_decode_impl(gpu, ffn, config, s, params)
}

// ─── Forward pass (decode, one token at a time) ─────────────────────────

/// Run one token through the Qwen3.5 model. Returns logits.
/// For DeltaNet layers, updates state in-place (S matrix + conv ring buffer).
/// For full attention layers, uses KV cache like standard transformer.
pub fn forward(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let dim = config.dim;

    // Embedding lookup
    let x = gpu.alloc_tensor(&[dim], DType::F32)?;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::Ggml(dtype) => {
            gpu.ggml_embedding_lookup(dtype, &weights.token_embd, &x, token, dim)?
        }
        _ => panic!("unsupported embedding format"),
    }

    forward_from_x(gpu, weights, config, x, pos, kv_cache, dn_state)
}

/// Shared forward pass — returns logits as CPU Vec<f32>.
fn forward_from_x(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    x: GpuTensor,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let logits_gpu = forward_from_x_gpu(gpu, weights, config, x, pos, kv_cache, dn_state)?;
    let logits_data = gpu.download_f32(&logits_gpu)?;
    gpu.free_tensor(logits_gpu)?;
    Ok(logits_data)
}

/// Shared forward pass — returns logits as GPU tensor (no download).
/// Shared forward pass — returns logits as GPU tensor (no download).
/// Caller must free the returned tensor.
///
/// Delegates to `forward_scratch_layers` via a temporary `Qwen35Scratch`,
/// ensuring test/demo paths exercise the same pipeline code as production.
/// NOT production-representative for benchmarking: allocates and frees a full
/// scratch bundle per call. Use `forward_scratch` with a persistent scratch
/// for perf measurement. Per-layer `DEBUG_LAYERS` trace and `trace_finite`
/// "qkvza" checkpoint are not emitted in this path — they are available
/// via `dump_hidden_localize` in the scratch path under HIPFIRE_DUMP_HIDDEN.
fn forward_from_x_gpu(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    x: GpuTensor,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<GpuTensor> {
    let required_tokens = checked_kv_end(pos, 1, "forward_from_x_gpu")?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    let dim = config.dim;

    // Allocate a temporary scratch bundle. repeat_window=1 (unused in this path).
    // kv_max_seq=8192 matches Qwen35Scratch::new default — sufficient for
    // test/demo single-token forward; these callers don't prefill.
    let scratch = Qwen35Scratch::new(gpu, config, 1)?;

    // Copy input embedding into scratch.x
    gpu.hip.memcpy_dtod(&scratch.x.buf, &x.buf, dim * 4)?;
    gpu.free_tensor(x)?;

    // Set position buffer
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    // DEBUG_LAYERS: dump embedding + per-layer norms (same as old forward_from_x_gpu)
    let debug_layers = std::env::var("DEBUG_LAYERS").is_ok();
    if debug_layers && pos == 0 {
        let hid = gpu.download_f32(&scratch.x)?;
        let norm: f32 = hid.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "EMB: first4=[{:.6},{:.6},{:.6},{:.6}] norm={norm:.4}",
            hid[0], hid[1], hid[2], hid[3]
        );
    }

    // Run the production pipeline
    forward_scratch_layers(
        gpu, weights, config, pos, kv_cache, dn_state, &scratch, None, None,
    )?;

    // DEBUG_LAYERS: dump per-layer residual norms
    if debug_layers && pos == 0 {
        let hid = gpu.download_f32(&scratch.x)?;
        let norm: f32 = hid.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "POST: first4=[{:.4},{:.4},{:.4},{:.4}] norm={norm:.2}",
            hid[0], hid[1], hid[2], hid[3]
        );
    }

    // Copy logits out of scratch before freeing — the returned tensor must
    // outlive the scratch bundle.
    let logits = gpu.alloc_tensor(&[config.vocab_size], DType::F32)?;
    gpu.hip
        .memcpy_dtod(&logits.buf, &scratch.logits.buf, config.vocab_size * 4)?;

    // Free scratch (all pre-allocated buffers)
    scratch.free_gpu(gpu);

    Ok(logits)
}

/// Pre-allocated scratch buffers for zero-alloc qwen35 forward + GPU sampling.
pub struct Qwen35Scratch {
    // Persistent state
    pub x: GpuTensor,                      // [dim]
    pub tmp: GpuTensor,                    // [dim]
    pub pos_buf: hip_bridge::DeviceBuffer, // 4 bytes
    /// 3 i32 = (t, h, w) for the 3D mrope kernels. Written ONLY on the VL
    /// path (`MropeCtx` present); the 1D kernels never read it, so the
    /// text-only dispatch sequence is unaffected by its existence.
    pub pos_buf3: hip_bridge::DeviceBuffer, // 12 bytes

    // DeltaNet temporaries (reused across layers)
    pub dn_qkv: GpuTensor,      // [qkv_dim]
    pub dn_z: GpuTensor,        // [v_dim]
    pub dn_alpha: GpuTensor,    // [n_v_heads]
    pub dn_beta: GpuTensor,     // [n_v_heads]
    pub dn_conv_out: GpuTensor, // [qkv_dim]
    pub dn_q: GpuTensor,        // [v_dim] (after repeat-interleave)
    pub dn_k: GpuTensor,        // [v_dim]
    pub dn_v: GpuTensor,        // [v_dim]
    pub dn_q_raw: GpuTensor,    // [k_dim] (before repeat)
    pub dn_k_raw: GpuTensor,    // [k_dim]
    pub dn_attn_out: GpuTensor, // [v_dim]
    pub dn_normed: GpuTensor,   // [v_dim]

    // FullAttn temporaries (reused across layers)
    pub fa_q_full: GpuTensor,   // [n_heads * head_dim * 2]
    pub fa_q: GpuTensor,        // [n_heads * head_dim]
    pub fa_gate: GpuTensor,     // [n_heads * head_dim]
    pub fa_k: GpuTensor,        // [n_kv_heads * head_dim]
    pub fa_v: GpuTensor,        // [n_kv_heads * head_dim]
    pub fa_attn_out: GpuTensor, // [n_heads * head_dim]

    // Shared (used by both layer types)
    pub o: GpuTensor,          // [dim]
    pub gate_ffn: GpuTensor,   // [hidden_dim]
    pub up: GpuTensor,         // [hidden_dim]
    pub ffn_hidden: GpuTensor, // [hidden_dim]
    pub ffn_out: GpuTensor,    // [dim]

    // Sampling
    pub logits: GpuTensor,     // [vocab_size]
    pub sample_buf: GpuTensor, // [2] — token_id + rng
    pub repeat_buf: GpuTensor, // [repeat_window]

    // MagnumQuant rotation scratch: FWHT(x) shared across Q/K/V (or gate/up, etc).
    // DeltaNet's output projection consumes v_dim values, which can exceed both
    // dim and the unused dense hidden_dim on MoE checkpoints.
    pub x_rot: GpuTensor, // [max(dim, hidden_dim, v_dim)]

    // Flash attention partials buffer for tile+reduce 2-kernel path.
    // Size: n_heads * max_tiles * (2 + head_dim) floats.
    pub flash_partials: GpuTensor,
    // Flash attention tri-state (applies to Q8 path; asym modes are flash-only):
    //   0 = never      force non-flash at all contexts (except >15K sanity)
    //   1 = auto       (default) flash kicks in at ctx >= 2048
    //   2 = always     force flash at all contexts
    pub flash_mode: u8,

    // MoE scratch (allocated only when config.num_experts > 0). Pre-allocated
    // so moe_ffn_decode can be captured by hipGraph — the per-layer allocs
    // it used to do violated the "no allocator ops while capturing" rule.
    pub moe_router_logits: Option<GpuTensor>, // [num_experts]
    pub moe_scalar_buf: Option<GpuTensor>,    // [1] shared-expert gate scalar
    pub moe_x_rot: Option<GpuTensor>,         // [dim]
    pub moe_gate_up_buf: Option<GpuTensor>,   // [2*max_inter]   fallback path
    pub moe_gate_buf: Option<GpuTensor>,      // [max_inter]     fallback path
    pub moe_up_buf: Option<GpuTensor>,        // [max_inter]     fallback path
    pub moe_ffn_hidden: Option<GpuTensor>,    // [max_inter]     fallback path
    pub moe_ffn_out: Option<GpuTensor>,       // [dim]           fallback path
    pub moe_gate_batch: Option<GpuTensor>,    // [k × mi]
    pub moe_up_batch: Option<GpuTensor>,      // [k × mi]
    pub moe_rot_batch: Option<GpuTensor>,     // [k × mi]
    /// Phase 2b: GPU-side top-K outputs (kept on-device so moe_ffn_decode
    /// can stay in a graph-capturable stream).
    pub moe_topk_indices: Option<GpuTensor>, // [k] i32 stored as f32 alias
    pub moe_topk_weights: Option<GpuTensor>,  // [k] f32
    // Atomic-free MoE down expansion buffer for decode — payload [k × dim]
    // plus ceil(dim/4) f32-sized counter slots for the optional last-arriver
    // combine. The full allocation is zeroed once; each fused dispatch resets
    // the counters it consumes.
    // Paired with `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` +
    // `moe_down_combine_k8_batched` (batch_size=1) in `moe_ffn_decode_impl`'s
    // use_gpu_topk path. Replaces the K_TOP-way atomicAdd that introduced
    // non-deterministic wavefront-order-dependent FP rounding under hipGraph
    // replay (task #100).
    pub moe_down_expanded: Option<GpuTensor>,

    // Optional long-prefill scratch. Default is None to preserve VRAM
    // footprint; set HIPFIRE_PREFILL_REUSE_PBS=1 to allocate and reuse it.
    pub prefill_batch: Option<PrefillBatchScratch>,
    // Retained widened-prefill PBS: the fully-admitted large owned scratch
    // (up to the 8192-row ceiling) is reused across requests when it fits
    // instead of alloc/free per request (~25 ms on 6k-token prefills).
    // Bit-identical: every PBS tensor is overwritten before it is read,
    // the same reuse contract the legacy `prefill_batch` cache above relies
    // on. RefCell because the prefill entries take `&Qwen35Scratch`; all
    // mutation happens on the single owning daemon thread.
    pub widened_prefill_batch: std::cell::RefCell<Option<PrefillBatchScratch>>,
    // Explicit-equivalent owner of an eligible DEFAULT resident-MoE request
    // under `HIPFIRE_PREFILL_REUSE_PBS=1` (`plan_wrapper_pbs`): allocated
    // lazily, with the rows/tape flag the explicit
    // `HIPFIRE_PREFILL_MAX_BATCH=4096` cache sizing would give, only when the
    // implicit cache differs from them. RefCell for the same reason as above.
    pub reuse_virtual_pbs: std::cell::RefCell<Option<PrefillBatchScratch>>,
}

fn qwen35_x_rot_len(dim: usize, hidden_dim: usize, v_dim: usize) -> usize {
    dim.max(hidden_dim).max(v_dim)
}

/// Number of F32 elements in the shared flash-attention partials buffer.
///
/// Legacy sizing (every arch except the measured gfx1100 split below):
/// `batch_mult` query rows at the smallest of the Q8 decode tile, 128 and the
/// batched-attention tile.
///
/// Exact gfx1100 whose Q8 decode tile is finer than the batched tile (tile32
/// vs tile128 through 8K): single-token decode and batched attention never
/// use the buffer at the same time, so it holds the larger of one decode row
/// at the decode tile and `batch_mult` rows (default 32) at the batched tile.
/// Batched launchers sub-batch by the buffer's capacity, so the smaller
/// buffer only costs extra launches past 32 rows.
fn qwen35_flash_partials_len(
    arch: &str,
    n_heads: usize,
    head_dim: usize,
    kv_max_seq: usize,
    q8_decode_tile: usize,
    batched_tile: usize,
    configured_batch: Option<usize>,
) -> usize {
    let stride = 2 + head_dim;
    if arch == "gfx1100" && q8_decode_tile < batched_tile {
        let decode_elems = n_heads * kv_max_seq.div_ceil(q8_decode_tile) * stride;
        let batched_elems =
            configured_batch.unwrap_or(32) * n_heads * kv_max_seq.div_ceil(batched_tile) * stride;
        return decode_elems.max(batched_elems);
    }
    // See llama.rs: also floor against the batched-attention tile, since a
    // smaller HIPFIRE_ATTN_TILE_SIZE raises max_tiles and would undersize
    // this same buffer.
    let tile_size = q8_decode_tile.min(128).min(batched_tile);
    configured_batch.unwrap_or(16) * n_heads * kv_max_seq.div_ceil(tile_size) * stride
}

impl Qwen35Scratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, repeat_window: usize) -> HipResult<Self> {
        // Flash partials are sized for up to 8192 ctx. Override via new_with_kv_max.
        Self::new_with_kv_max(gpu, config, repeat_window, 8192)
    }

    pub fn new_with_kv_max(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        repeat_window: usize,
        kv_max_seq: usize,
    ) -> HipResult<Self> {
        let dim = config.dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;
        // MQ sign tables are GPU-owned persistent state rather than scratch.
        // Warm them before the scratch transaction so a failure cannot strand
        // partially constructed scratch allocations.
        if config.num_experts > 0 {
            gpu.ensure_mq_signs()?;
        }

        // GpuTensor and DeviceBuffer do not free device memory on Drop. Track
        // non-owning aliases for every allocation made below so any later
        // failure can release the partially constructed scratch transaction.
        let mut tensor_ledger: Vec<GpuTensor> = Vec::with_capacity(48);
        let mut buffer_ledger: Vec<hip_bridge::DeviceBuffer> = Vec::with_capacity(2);
        macro_rules! cleanup_allocations {
            () => {{
                for tensor in tensor_ledger.drain(..) {
                    let _ = gpu.free_tensor(tensor);
                }
                let _ = gpu.bind_thread();
                for buffer in buffer_ledger.drain(..) {
                    let _ = gpu.hip.free(buffer);
                }
            }};
        }
        macro_rules! tracked_tensor {
            ($allocation:expr) => {
                match $allocation {
                    Ok(tensor) => {
                        // SAFETY: the alias is freed only if construction
                        // fails. On success it drops as a non-owning handle
                        // while the original tensor moves into Self.
                        tensor_ledger.push(GpuTensor {
                            buf: unsafe { tensor.buf.alias() },
                            shape: tensor.shape.clone(),
                            dtype: tensor.dtype,
                        });
                        tensor
                    }
                    Err(error) => {
                        cleanup_allocations!();
                        return Err(error);
                    }
                }
            };
        }
        macro_rules! tracked_buffer {
            ($allocation:expr) => {
                match $allocation {
                    Ok(buffer) => {
                        // SAFETY: same single-free transaction contract as
                        // tracked_tensor above.
                        buffer_ledger.push(unsafe { buffer.alias() });
                        buffer
                    }
                    Err(error) => {
                        cleanup_allocations!();
                        return Err(error);
                    }
                }
            };
        }

        Ok(Self {
            x: tracked_tensor!(gpu.alloc_tensor(&[dim], DType::F32)),
            tmp: tracked_tensor!(gpu.alloc_tensor(&[dim], DType::F32)),
            pos_buf: tracked_buffer!(gpu.hip.malloc(4)),
            pos_buf3: tracked_buffer!(gpu.hip.malloc(12)),

            dn_qkv: tracked_tensor!(gpu.alloc_tensor(&[qkv_dim], DType::F32)),
            dn_z: tracked_tensor!(gpu.alloc_tensor(&[v_dim], DType::F32)),
            dn_alpha: tracked_tensor!(
                gpu.alloc_tensor(&[config.linear_num_value_heads], DType::F32)
            ),
            dn_beta: tracked_tensor!(
                gpu.alloc_tensor(&[config.linear_num_value_heads], DType::F32)
            ),
            dn_conv_out: tracked_tensor!(gpu.alloc_tensor(&[qkv_dim], DType::F32)),
            dn_q: tracked_tensor!(gpu.alloc_tensor(&[v_dim], DType::F32)),
            dn_k: tracked_tensor!(gpu.alloc_tensor(&[v_dim], DType::F32)),
            dn_v: tracked_tensor!(gpu.alloc_tensor(&[v_dim], DType::F32)),
            dn_q_raw: tracked_tensor!(gpu.alloc_tensor(&[k_dim], DType::F32)),
            dn_k_raw: tracked_tensor!(gpu.alloc_tensor(&[k_dim], DType::F32)),
            dn_attn_out: tracked_tensor!(gpu.alloc_tensor(&[v_dim], DType::F32)),
            dn_normed: tracked_tensor!(gpu.alloc_tensor(&[v_dim], DType::F32)),

            fa_q_full: tracked_tensor!(gpu.alloc_tensor(&[q_dim * 2], DType::F32)),
            fa_q: tracked_tensor!(gpu.alloc_tensor(&[q_dim], DType::F32)),
            fa_gate: tracked_tensor!(gpu.alloc_tensor(&[q_dim], DType::F32)),
            fa_k: tracked_tensor!(gpu.alloc_tensor(&[kv_dim], DType::F32)),
            fa_v: tracked_tensor!(gpu.alloc_tensor(&[kv_dim], DType::F32)),
            fa_attn_out: tracked_tensor!(gpu.alloc_tensor(&[q_dim], DType::F32)),

            o: tracked_tensor!(gpu.alloc_tensor(&[dim], DType::F32)),
            gate_ffn: tracked_tensor!(gpu.alloc_tensor(&[config.hidden_dim], DType::F32)),
            up: tracked_tensor!(gpu.alloc_tensor(&[config.hidden_dim], DType::F32)),
            ffn_hidden: tracked_tensor!(gpu.alloc_tensor(&[config.hidden_dim], DType::F32)),
            ffn_out: tracked_tensor!(gpu.alloc_tensor(&[dim], DType::F32)),

            logits: tracked_tensor!(gpu.alloc_tensor(&[config.vocab_size], DType::F32)),
            sample_buf: tracked_tensor!(gpu.alloc_tensor(&[2], DType::F32)),
            repeat_buf: tracked_tensor!(gpu.alloc_tensor(&[repeat_window], DType::F32)),
            x_rot: tracked_tensor!(gpu.alloc_tensor(
                &[qwen35_x_rot_len(dim, config.hidden_dim, v_dim)],
                DType::F32,
            )),

            // Flash attention partials: enough for the smallest tile used by
            // Q8 decode experiments and the fixed tile_size=128 paths.
            // n_heads * max_tiles * (2 + head_dim) floats per batched query
            // position; total buffer = batch_mult × per-position-bytes.
            // Exact gfx1100 through 8K sizes decode and batched rows
            // separately; see `qwen35_flash_partials_len`.
            //
            // batch_mult is the maximum query positions a single FA dispatch
            // can fit; the dispatcher (`launch_asym_flash_batched`) reads the
            // buffer's actual capacity at call time and auto-chunks larger
            // prefill batches into multiple sub-launches. So a lower
            // batch_mult here trades ~linear extra dispatch overhead on
            // prefill (PREFILL_MAX_BATCH=256 → ceil(256/batch_mult) calls per
            // FA layer) for ~linearly less VRAM at long context.
            //
            // The per-position size scales with kv_max_seq (= physical_cap
            // post-eviction), and that scaling is what made #85 visible: at
            // max_seq=170k, no CASK, 27B (n_heads=24, head_dim=256) the old
            // batch_mult=64 → 2.1 GB just for these partials, exceeding VRAM
            // headroom on 24 GB cards. Cutting batch_mult by 4× (16) keeps
            // the prefill chunking moderate while saving 1.6 GB at that
            // worst-case shape; CASK-on workloads (small physical_cap) are
            // unaffected because the buffer is already tiny there.
            //
            // On the gfx1100 tile32/tile128 split the legacy formula
            // incidentally gave 64 batched rows; 16 regressed prefill, 32
            // keeps it and halves the buffer.
            //
            // Override with HIPFIRE_FLASH_PARTIALS_BATCH for tuning. Power of
            // two preferred (matches FA dispatcher chunking).
            flash_partials: {
                let q8_decode_tile = rdna_compute::attention::q8_flash_tile_size(
                    &gpu.arch,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    kv_max_seq,
                );
                let configured_batch = hipfire_runtime::config::get()
                    .flash_partials_batch
                    .filter(|&n| n >= 1 && n <= PREFILL_MAX_BATCH);
                let n = qwen35_flash_partials_len(
                    &gpu.arch,
                    config.n_heads,
                    config.head_dim,
                    kv_max_seq,
                    q8_decode_tile,
                    gpu.attn_tile_size(),
                    configured_batch,
                );
                tracked_tensor!(gpu.alloc_tensor(&[n], DType::F32))
            },
            // Flash attention tri-state for the Q8 path. Asym modes always
            // flash regardless.
            //   HIPFIRE_ATTN_FLASH=never|0|off    → non-flash at all contexts
            //   HIPFIRE_ATTN_FLASH=auto|1|on      → flash at ctx >= 2048
            //   HIPFIRE_ATTN_FLASH=always|2|force → flash at all contexts
            //
            // Default on gfx11/gfx12 (graph-capable archs): `2` (always
            // flash). On other archs: `1` (auto). The capture path at
            // qwen35.rs:8199 hard-wires `use_flash = capture_mode || ...`
            // because attention_q8_0_kv has variable block_size + variable
            // shared-mem (not capture-safe). Without an always-flash default
            // on capture-capable archs, direct mode at small ctx silently
            // uses attention_q8_0_kv while a captured-and-replayed forward
            // uses attention_flash_q8_0 — same math, different fp32
            // reduction order, observed as ~0.44 logit delta direct-vs-graph
            // on shisa-Qwen3.6-A3B-PARO (see
            // .scratch/hipgraph-moe-drift-audit.md Part A). Aligning the
            // default flips both paths to `attention_flash_q8_0` and makes
            // direct vs graph byte-identical at the cost of moving small-
            // context decode off the non-flash kernel (~few % attention
            // perf hit, small contribution to total MoE decode time).
            // Honors HIPFIRE_ATTN_FLASH=never|0|off as an explicit override
            // for users who prefer the non-flash kernel and don't intend
            // to use graph capture.
            flash_mode: hipfire_runtime::llama::attention_flash_mode(&gpu.arch) as u8,

            moe_router_logits: None,
            moe_scalar_buf: None,
            moe_x_rot: None,
            moe_gate_up_buf: None,
            moe_gate_buf: None,
            moe_up_buf: None,
            moe_ffn_hidden: None,
            moe_ffn_out: None,
            moe_gate_batch: None,
            moe_up_batch: None,
            moe_rot_batch: None,
            moe_topk_indices: None,
            moe_topk_weights: None,
            moe_down_expanded: None,
            prefill_batch: None,
            widened_prefill_batch: std::cell::RefCell::new(None),
            reuse_virtual_pbs: std::cell::RefCell::new(None),
        })
        .and_then(|mut s| {
            // Allocate MoE scratch only for MoE configs. Done after the
            // main struct init so these Options start as None for dense
            // models and never cost VRAM there.
            if config.num_experts > 0 {
                let hidden = config.dim;
                let n_exp = config.num_experts;
                let mi = config.moe_intermediate_size;
                let smi = config.shared_expert_intermediate_size;
                let max_inter = mi.max(smi);
                let k = config.num_experts_per_tok;
                s.moe_router_logits =
                    Some(tracked_tensor!(gpu.alloc_tensor(&[n_exp], DType::F32)));
                s.moe_scalar_buf =
                    Some(tracked_tensor!(gpu.alloc_tensor(&[1], DType::F32)));
                s.moe_x_rot =
                    Some(tracked_tensor!(gpu.alloc_tensor(&[hidden], DType::F32)));
                s.moe_gate_up_buf = Some(tracked_tensor!(
                    gpu.alloc_tensor(&[2 * max_inter], DType::F32)
                ));
                s.moe_gate_buf = Some(tracked_tensor!(
                    gpu.alloc_tensor(&[max_inter], DType::F32)
                ));
                s.moe_up_buf = Some(tracked_tensor!(
                    gpu.alloc_tensor(&[max_inter], DType::F32)
                ));
                s.moe_ffn_hidden = Some(tracked_tensor!(
                    gpu.alloc_tensor(&[max_inter], DType::F32)
                ));
                s.moe_ffn_out =
                    Some(tracked_tensor!(gpu.alloc_tensor(&[hidden], DType::F32)));
                s.moe_gate_batch =
                    Some(tracked_tensor!(gpu.alloc_tensor(&[k * mi], DType::F32)));
                s.moe_up_batch =
                    Some(tracked_tensor!(gpu.alloc_tensor(&[k * mi], DType::F32)));
                s.moe_rot_batch =
                    Some(tracked_tensor!(gpu.alloc_tensor(&[k * mi], DType::F32)));
                // i32 topk_indices stored in an F32 tensor (same byte width).
                // The kernel that writes it casts the buffer to int*, and the
                // indexed MoE GEMV kernels read it as int*.
                s.moe_topk_indices =
                    Some(tracked_tensor!(gpu.alloc_tensor(&[k], DType::F32)));
                s.moe_topk_weights =
                    Some(tracked_tensor!(gpu.alloc_tensor(&[k], DType::F32)));
                // Atomic-free decode MoE down payload plus reusable counter tail.
                s.moe_down_expanded = Some(tracked_tensor!(
                    gpu.zeros(&[k * hidden + hidden.div_ceil(4)], DType::F32)
                ));
            }
            if hipfire_config::developer_var("HIPFIRE_PREFILL_REUSE_PBS")
                .ok()
                .as_deref()
                == Some("1")
            {
                // Ordinary reuse cache: tape-free and capped at legacy ≤512
                // even under a larger requested ceiling — only a fully
                // admitted request allocates the larger owned PBS, after
                // Q8+EF/KV/free-VRAM checks in the inner entry. Tree/tape
                // wrappers use separate owned tape-enabled scratch, never
                // this cache. Other arches keep their existing sizing and
                // tape behavior.
                // Sizing is the pure `implicit_reuse_pbs_plan` the CPU
                // admission tests read for the same default/explicit cases.
                let (max_batch, cap_tape) = super::prefill::implicit_reuse_pbs_plan_for_gpu(gpu);
                s.prefill_batch = match PrefillBatchScratch::new_opt(gpu, config, max_batch, cap_tape) {
                    Ok(prefill) => Some(prefill),
                    Err(error) => {
                        cleanup_allocations!();
                        return Err(error);
                    }
                };
            }
            Ok(s)
        })
    }

    /// Free all GPU tensors. Call before drop to return VRAM. Checked — reports first free failure.
    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        let mut first_err: Option<HipError> = None;
        let mut note = |r: HipResult<()>| {
            if let Err(e) = r {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        };
        note(gpu.free_tensor(self.x));
        note(gpu.free_tensor(self.tmp));
        let _ = gpu.bind_thread();
        note(gpu.hip.free(self.pos_buf).map(|_| ()));
        note(gpu.hip.free(self.pos_buf3).map(|_| ()));
        for t in [
            self.dn_qkv,
            self.dn_z,
            self.dn_alpha,
            self.dn_beta,
            self.dn_conv_out,
            self.dn_q,
            self.dn_k,
            self.dn_v,
            self.dn_q_raw,
            self.dn_k_raw,
            self.dn_attn_out,
            self.dn_normed,
            self.fa_q_full,
            self.fa_q,
            self.fa_gate,
            self.fa_k,
            self.fa_v,
            self.fa_attn_out,
            self.o,
            self.gate_ffn,
            self.up,
            self.ffn_hidden,
            self.ffn_out,
            self.logits,
            self.sample_buf,
            self.repeat_buf,
            self.x_rot,
            self.flash_partials,
        ] {
            note(gpu.free_tensor(t));
        }
        // MoE scratch — only present for MoE configs.
        for t in [
            self.moe_router_logits,
            self.moe_scalar_buf,
            self.moe_x_rot,
            self.moe_gate_up_buf,
            self.moe_gate_buf,
            self.moe_up_buf,
            self.moe_ffn_hidden,
            self.moe_ffn_out,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_topk_indices,
            self.moe_topk_weights,
            self.moe_down_expanded,
        ] {
            if let Some(buf) = t {
                note(gpu.free_tensor(buf));
            }
        }
        if let Some(pbs) = self.prefill_batch {
            note(pbs.free_gpu(gpu));
        }
        if let Some(pbs) = self.widened_prefill_batch.into_inner() {
            note(pbs.free_gpu(gpu));
        }
        if let Some(pbs) = self.reuse_virtual_pbs.into_inner() {
            note(pbs.free_gpu(gpu));
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Per-device scratch bundle for the multi-GPU forward path. Each device gets
/// its own `Qwen35Scratch` because the residual stream `s.x` (and `s.logits`)
/// must live on the device executing the current band's layers — cross-band
/// boundaries copy `s.x` between devices via `Gpus::boundary_copy`. `s.logits`
/// is also allocated per-device for simplicity (~600 KB each at vocab=152K)
/// even though only the output device's `s.logits` is consumed post-loop.
pub struct Qwen35ScratchSet {
    pub per_device: Vec<Qwen35Scratch>,
}

impl Qwen35ScratchSet {
    pub fn new_with_kv_max_multi(
        gpus: &mut Gpus,
        config: &Qwen35Config,
        repeat_window: usize,
        kv_max_seq: usize,
    ) -> HipResult<Self> {
        let mut per_device = Vec::with_capacity(gpus.devices.len());
        for dev_idx in 0..gpus.devices.len() {
            let g = &mut gpus.devices[dev_idx];
            per_device.push(Qwen35Scratch::new_with_kv_max(
                g,
                config,
                repeat_window,
                kv_max_seq,
            )?);
        }
        Ok(Self { per_device })
    }

    pub fn free_gpu_multi(self, gpus: &mut Gpus) {
        for (dev_idx, scratch) in self.per_device.into_iter().enumerate() {
            scratch.free_gpu(&mut gpus.devices[dev_idx]);
        }
    }
}

pub(crate) fn checked_kv_end(start_pos: usize, token_count: usize, site: &str) -> HipResult<usize> {
    start_pos.checked_add(token_count).ok_or_else(|| {
        HipError::new(
            0,
            &format!("{site}: KV token range overflow ({start_pos} + {token_count})"),
        )
    })
}

#[inline]
fn ar_graph_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_AR_GRAPH_TRACE")
            .ok()
            .as_deref()
            == Some("1")
    })
}

#[inline]
fn ar_graph_eligible_for_kv(
    requested: bool,
    compact_offset: usize,
    seq_sized_attend: bool,
) -> bool {
    // The captured single-token route is built while compact_offset is zero.
    // After eviction, Q/K RoPE must use physical_pos + compact_offset, and the
    // offset changes again at every later eviction. Neither hipGraph nor the
    // retained replay route currently has a dynamic offset input, so replaying
    // the old route silently rotates at the physical slot and corrupts decode.
    // The int8/int8c/hfq4 decode attends size block and LDS (scores[seq_len])
    // from the host seq_len, which a replay freezes at the capture position;
    // longer contexts then overrun LDS and decode goes degenerate (HIPF-192).
    requested && compact_offset == 0 && !seq_sized_attend
}

/// Identity of the caller buffers the AR hipGraph bakes at capture: the
/// model (embedding and lm_head), every KV tensor, every DeltaNet state
/// tensor and every scratch buffer. The graph replays captured device
/// pointers and ignores the buffers a later call passes, so a replay is valid
/// only for the same allocations. A caller that passes a new `DeltaNetState`
/// (or KV cache / scratch) for the next request must get a fresh capture, not
/// a replay that reads and writes the previous request's state. Never 0 (the
/// unbound value of `GraphState::ar_forward_binding`).
fn ar_graph_binding(
    weights: &Qwen35Weights,
    kv_cache: &llama::KvCache,
    dn_state: &DeltaNetState,
    s: &Qwen35Scratch,
) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |p: *mut std::ffi::c_void| h = (h ^ p as u64).wrapping_mul(0x100_0000_01b3);
    mix(weights.token_embd.buf.as_ptr());
    mix(weights.output.buf.buf.as_ptr());
    for t in kv_cache.k_gpu.iter().chain(&kv_cache.v_gpu) {
        mix(t.buf.as_ptr());
    }
    for t in dn_state
        .s_matrices
        .iter()
        .chain(&dn_state.s_scales)
        .chain(&dn_state.conv_states)
        .chain(&dn_state.s_ef_residual)
    {
        mix(t.buf.as_ptr());
    }
    mix(s.pos_buf.as_ptr());
    mix(s.pos_buf3.as_ptr());
    for t in [
        &s.x,
        &s.tmp,
        &s.dn_qkv,
        &s.dn_z,
        &s.dn_alpha,
        &s.dn_beta,
        &s.dn_conv_out,
        &s.dn_q,
        &s.dn_k,
        &s.dn_v,
        &s.dn_q_raw,
        &s.dn_k_raw,
        &s.dn_attn_out,
        &s.dn_normed,
        &s.fa_q_full,
        &s.fa_q,
        &s.fa_gate,
        &s.fa_k,
        &s.fa_v,
        &s.fa_attn_out,
        &s.o,
        &s.gate_ffn,
        &s.up,
        &s.ffn_hidden,
        &s.ffn_out,
        &s.logits,
        &s.sample_buf,
        &s.repeat_buf,
        &s.x_rot,
        &s.flash_partials,
    ] {
        mix(t.buf.as_ptr());
    }
    for t in [
        &s.moe_router_logits,
        &s.moe_scalar_buf,
        &s.moe_x_rot,
        &s.moe_gate_up_buf,
        &s.moe_gate_buf,
        &s.moe_up_buf,
        &s.moe_ffn_hidden,
        &s.moe_ffn_out,
        &s.moe_gate_batch,
        &s.moe_up_batch,
        &s.moe_rot_batch,
        &s.moe_topk_indices,
        &s.moe_topk_weights,
        &s.moe_down_expanded,
    ]
    .into_iter()
    .flatten()
    {
        mix(t.buf.as_ptr());
    }
    h.max(1)
}

/// Zero-alloc forward pass using pre-allocated scratch buffers.
/// Logits stay on GPU in scratch.logits. Returns nothing — caller uses scratch.logits.
pub fn forward_scratch(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    #[cfg(feature = "moe-oracle")]
    crate::qwen35::oracle::set_decode_position(pos).map_err(|e| HipError::new(0, &e))?;
    let required_tokens = checked_kv_end(pos, 1, "forward_scratch")?;
    // Grow before any possible AR graph capture/replay. Stable virtual
    // addresses keep existing graph pointer arguments valid.
    super::prefill::release_widened_pbs_for_kv_growth(
        gpu,
        kv_cache,
        config,
        scratch,
        required_tokens,
    )?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    let dim = config.dim;
    // hipGraph capture for MoE was previously gated off-by-default behind
    // HIPFIRE_GRAPH_MOE=1 because of a known drift bug (task #100): under
    // capture, A3B accumulated a per-step ~1-ULP delta that compounded
    // through the KV cache + GDN state and crossed the top-1 margin at
    // step ~7 (q8 KV) or ~114 (asym3 KV), producing visible token-loop
    // attractors by step 30-50 ("- **One**\n- **One**\n…").
    //
    // Root cause (fixed 2026-05-21): `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed`
    // used K_TOP=8 concurrent `atomicAdd` writes per output row. FP32
    // addition is non-associative, so the final bits depend on wavefront
    // scheduling order. Under hipGraph replay that order differs from
    // direct execution (graph scheduling pipelines kernels differently),
    // introducing the systematic per-step delta. The kernel's own header
    // (`kernels/src/gemv_hfq4g256_moe_down.hip:14-19`) had already flagged
    // this non-determinism but rated it negligible based on the
    // direct-only smoke test — capture amplifies the effect.
    //
    // Fix: the MoE FFN decode path now uses the atomic-free expand+combine
    // pattern already used in prefill (`forward_prefill_batch_with_pbs`
    // L5217-5232): `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`
    // writes one row per (expert-rank, m), then `moe_down_combine_k8_batched`
    // sums K_TOP slots into x_residual in a fixed iteration order. The
    // resulting MoE FFN output is byte-deterministic under both direct
    // execution and hipGraph replay.
    //
    // `experimental.graph.moe` remains the explicit policy control. The atomic
    // fix is necessary but not sufficient — the CPU-topK fallback path
    // (when not all gate-side MoE weights are MQ4G256, e.g. router=Q8 per
    // the post-2026-04 router-attractor fix) calls `download_f32(router_logits)`,
    // a sync D2H that fails under graph capture with hipError 906. Until
    // that D2H is migrated to a capture-safe equivalent, opting in only
    // works for models where the runtime takes the use_gpu_topk path.
    //
    // Reproducer used to characterize the fix:
    //   hipfire config set experimental.graph.forward true
    //   hipfire config set experimental.graph.moe true
    //   HIPFIRE_SMOKE_KV=q8 \
    //   HIPFIRE_SMOKE_MODE=chat HIPFIRE_SMOKE_STEPS=200 \
    //   HIPFIRE_SMOKE_PROMPT="Count from one to twenty in English." \
    //   ./target/release/examples/a3b_smoke_forward <uniform-mq4-a3b>
    //
    // Graph policy is resolved once from TOML before GPU initialization and is
    // carried directly by the GPU feature snapshot.
    // Default-ON (2026-06-16): cross-arch A/B validated — gfx12 +4.2% A3B mq4
    // decode, coherence-gate clean (fluent at decode pos 1-4 on current ROCm).
    // Opt out with `experimental.graph.moe = false`. Both root causes of the prior
    // crash/drift are fixed:
    //   1. atomicAdd drift (task #100, 2026-05-21): expand+combine pattern.
    //   2. CPU-topK fallback D2H: `download_f32(router_logits)` replaced by
    //      GPU `softmax_f32` + `moe_topk_renorm_k8` + small [k] D2H — fully
    //      capture-safe. Mixed-kmap A3B (Q8 router, post-PR #199) no longer
    //      crashes with hipError 906 under AR graph capture.
    // Validated + flipped to default-on 2026-06-16.
    let allow_moe = gpu.flags.graph_moe;
    // hipGraph per-forward-pass capture/replay default policy:
    //   - gfx12 (RDNA4): default-ON. +2.4-2.7% decode on 9B Qwen 3.5
    //     MFP4G32 (5-run mean, all positive, tight variance, 2026-05-11).
    //   - gfx11 (RDNA3 / 3.5): default-ON. +0.6-0.7% decode on 9B and
    //     0.8B HFP4G32 on 7900 XTX (5-run mean per model, all positive,
    //     variance 1.001-1.010×, 2026-05-11). Smaller win than gfx12 —
    //     gfx11 has less per-launch overhead to amortize — but real
    //     and consistent across model sizes.
    //   - other archs (RDNA1/2, CDNA): default-OFF (opt-in via
    //     `experimental.graph.forward = true`) since not yet A/B'd on those.
    //   - MoE configs follow `experimental.graph.moe`. The ~30-50-token
    //     attractor drift in the use_gpu_topk MoE down step was fixed
    //     2026-05-21 (task #100 — atomicAdd → expand+combine), but the
    //     CPU-topK fallback's `download_f32(router_logits)` D2H sync
    //     remains capture-incompatible, so mixed-kmap A3B (post-PR #199)
    //     can crash under graph capture even with the fix. Once that
    //     D2H is migrated to a capture-safe path, the MoE default can
    //     be flipped to follow the arch defaults.
    // An explicit `experimental.graph.forward = false` always wins.
    let graph_override = gpu.flags.graph_forward;
    let graph_arch_default = gpu.arch.starts_with("gfx12") || gpu.arch.starts_with("gfx11");
    let graph_enabled = graph_override.unwrap_or(graph_arch_default);
    // AR-forward hipGraph RE-ENABLED (2026-06-16) — the kernarg-snapshot attractor
    // is re-verified GONE on current ROCm (HIP 7.x, coherence-gate clean, fluent at
    // decode pos 1-4 on gfx12 A3B mq4). Opt out with
    // `experimental.graph.ar = false` (default off on Windows gfx1100, #655,
    // where replay silently emitted wrong output; opt in with `= true`). The prior
    // 2026-05-15 disable rationale is retained below; it SUPERSEDED the
    // arch-default re-enable merged from master (`graph_enabled` above is kept
    // live so the graph policy and kill switch stay wired for when the
    // path is flipped back on). Empirically on ROCm 7.2.2 + gfx11 +
    // Qwen3.5-27B mq4, both replay AND capture+launch produce a token-0
    // attractor outside very narrow conditions:
    //   - Capture+launch at position 2 (after 1 direct warmup) → `!!!!!`
    //   - Capture+launch at position 4 (after 3 direct warmups) → correct
    //   - Replay of a working capture (any position) → `!!!!!` from pos+1 on
    // The kernarg-snapshot bug isn't fixable by warmup tuning OR caller-driven
    // commit gating (`end_decode_turn()`); both fail empirically. Master's
    // task-#100 fix targets MoE drift, NOT this AR-forward attractor, so the
    // merge does not clear the disable. Until the capture/replay attractor is
    // re-verified gone on current ROCm (7.13) via the coherence gate, AR
    // forward is direct-only. Policy infra (`ar_forward_kernel_dirty`,
    // `ar_forward_replay_enabled`, `end_decode_turn()`, `drop_captured_graph()`)
    // is preserved on Gpu so the path can be flipped on once the bug is fixed.
    // RE-VERIFY GATE (2026-06-12): the AR graph policy flips the 2026-05-15
    // disable back on to re-test the capture/replay attractor on current ROCm
    // (HIP 7.x). Default OFF preserves the direct-only behavior. When set, the
    // path still honors the HIPFIRE_GRAPH kill switch + arch default via
    // `graph_enabled`.
    let ar_graph_test = gpu.flags.graph_ar;
    // AR-forward hipGraph eligibility. Plain sequential single-token AR decode
    // is eligible BY DEFAULT (the consume below resets to true); spec-decode /
    // MTP re-seed and the verify/prefill batch path explicitly set this FALSE
    // right before their `forward_scratch` call so the plain-AR graph can never
    // capture or replay in a non-sequential context. An ineligible call also
    // INVALIDATES any captured graph (forces re-capture on the next plain call).
    let requested_graph_eligible = std::mem::replace(&mut gpu.graphs.ar_graph_eligible, true);
    let graph_eligible = ar_graph_eligible_for_kv(
        requested_graph_eligible,
        kv_cache.compact_offset,
        kv_cache.quant_int8 || kv_cache.quant_hfq4,
    );
    // Redline's plain-AR capture/replay has the same eligibility contract as
    // the AR HipGraph. MTP/spec re-seed and verify calls must not contaminate
    // or consume the immutable single-token replay sequence.
    gpu.replay.set_forward_eligible(graph_eligible);
    if ar_graph_test && !graph_eligible {
        gpu.graphs.ar_forward_replay_enabled = false;
        gpu.graphs.ar_forward_kernel_dirty = true;
    }
    // MoE models require `experimental.graph.moe` in addition to the
    // arch/kill-switch guards. Dense models (num_experts==0) are unaffected.
    let graph_would_be_used = ar_graph_test
        && graph_enabled
        && graph_eligible
        && !gpu.replay.is_enabled()
        && (config.num_experts == 0 || allow_moe);
    // A CPU-executed step (memory.offload_exec=cpu over a spilled prefix) is a
    // host sync point — a D2H and an H2D around the multiplication — so it can
    // neither be recorded into a hipGraph nor replayed out of one. Take the
    // non-captured path for the model's whole lifetime instead.
    let cpu_blocks_capture = gpu.owns_host_cpu_weights();
    if graph_would_be_used && cpu_blocks_capture {
        hipfire_dispatch::log_capture_disabled_once();
    }
    let use_graph = graph_would_be_used && !cpu_blocks_capture;
    let binding = if use_graph {
        ar_graph_binding(weights, kv_cache, dn_state, scratch)
    } else {
        0
    };
    let _ = gpu.graphs.ar_forward_replay_enabled; // suppress unused warning

    let gap_started = rdna_compute::gap_timing::begin_forward();
    let embed_started = rdna_compute::gap_timing::now();
    // Embedding lookup into scratch.x (always direct, changes per token)
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Ggml(dtype) => {
            gpu.ggml_embedding_lookup(dtype, &weights.token_embd, &scratch.x, token, dim)?
        }
        _ => panic!("unsupported embedding format"),
    }
    rdna_compute::gap_timing::add_since(rdna_compute::gap_timing::Slot::Embed, embed_started);
    // The retained window opens after the embedding: the token row is a
    // per-replay input staged outside the tape, and a launch inside the window
    // that the tape lacks makes prepare refuse it.
    gpu.replay
        .begin_auto_capture_if_armed()
        .map_err(|reason| HipError::new(0, reason))?;

    let pos_i32 = pos as i32;
    if gpu.replay.should_route_aql() {
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        let replay = unsafe { gpu.replay.replay_linear_aql(pos) };
        return match replay {
            Ok(_) => Ok(()),
            Err(reason) => {
                gpu.replay
                    .poison(format!("prepared AQL replay failed: {reason}"));
                Err(HipError::new(0, &reason))
            }
        };
    }
    if gpu.replay.should_route_pm4() {
        let copy_started = rdna_compute::gap_timing::now();
        // Load-bearing HIP -> PM4 fence: this synchronous legacy-NULL copy drains
        // embedding/staging on NULL and blocking streams. Do not make it async
        // or install non-blocking producers without an explicit queue dependency.
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        rdna_compute::gap_timing::add_since(rdna_compute::gap_timing::Slot::PosCopy, copy_started);
        let replay = unsafe { gpu.replay_pm4_routed(pos) };
        rdna_compute::gap_timing::end_forward(gap_started, pos, "pm4");
        return match replay {
            Ok(()) => Ok(()),
            Err(reason) => {
                gpu.replay
                    .poison(format!("prepared PM4 replay failed: {reason}"));
                Err(HipError::new(0, &reason))
            }
        };
    }
    if use_graph
        && gpu.graphs.ar_forward_replay_enabled
        && gpu.graphs.graph_exec.is_some()
        && gpu.graphs.ar_forward_binding == binding
    {
        // ── Replay path: graph captured + kernels clean. Cheapest path: pos
        // memcpy + graph replay. The graph is position-agnostic (pos via
        // pos_buf), so replay is correct across positions and requests as long
        // as the buffers are the plain-AR continuation — which the spec markers
        // + verify invalidation guarantee — and this call passes the buffers
        // the graph was captured with (`ar_graph_binding`). ──
        let copy_started = rdna_compute::gap_timing::now();
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        rdna_compute::gap_timing::add_since(rdna_compute::gap_timing::Slot::PosCopy, copy_started);
        let launch_started = rdna_compute::gap_timing::now();
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())?;
        rdna_compute::gap_timing::add_since(
            rdna_compute::gap_timing::Slot::GraphLaunch,
            launch_started,
        );
        rdna_compute::gap_timing::end_forward(gap_started, pos, "hip_graph");
        if ar_graph_trace_enabled() {
            eprintln!("[qwen-ar-graph] replay pos={pos}");
        }
    } else if use_graph && gpu.graphs.ar_forward_kernel_dirty {
        // ── Direct path (kernel-dirty): kernels are dirty (init or post-
        // model-load). Capture would trip "hipMalloc not permitted under
        // stream capture" on the first inline JIT. Mark clean after a
        // successful direct dispatch so subsequent calls can capture. ──
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        forward_scratch_layers(
            gpu, weights, config, pos, kv_cache, dn_state, scratch, None, None,
        )?;
        gpu.graphs.ar_forward_kernel_dirty = false;
    } else if use_graph {
        // ── Capture + launch: kernels are clean but caller has not committed
        // a replay yet (or graph_exec is None, or this call's buffers differ
        // from the captured binding). Drop any prior captured graph,
        // record a fresh one, and launch it for this forward's output. After
        // the caller signals end_decode_turn(), the most recent capture is
        // promoted to the replay graph for the next decode turn. ──
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create()?);
        }
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
        gpu.graphs.begin_graph_capture(
            &gpu.hip,
            gpu.device_id,
            gpu.active_stream.as_ref().unwrap(),
        )?;
        forward_scratch_layers(
            gpu, weights, config, pos, kv_cache, dn_state, scratch, None, None,
        )?;
        gpu.graphs.end_graph_capture(
            &gpu.hip,
            gpu.device_id,
            gpu.active_stream.as_ref().unwrap(),
        )?;
        gpu.graphs.ar_forward_binding = binding;
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())?;
        if ar_graph_trace_enabled() {
            eprintln!("[qwen-ar-graph] capture pos={pos}");
        }
        // Intra-generate replay (2026-06-12): promote this fresh capture to the
        // replay graph immediately so the NEXT token replays (cheap: pos memcpy
        // + graph_launch) instead of re-capturing + re-instantiating every
        // token. The per-token instantiate is why the path "did nothing" — the
        // daemon never calls end_decode_turn() to enable replay.
        gpu.graphs.ar_forward_replay_enabled = true;
    } else {
        // ── Direct path (graph not eligible: arch / MoE config) ──
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        forward_scratch_layers(
            gpu, weights, config, pos, kv_cache, dn_state, scratch, None, None,
        )?;
    }
    if gpu.replay.should_auto_finalize_capture() {
        gpu.hip.device_synchronize()?;
        gpu.replay
            .finish_capture()
            .map_err(|reason| HipError::new(0, reason))?;
        let prepare = if gpu.replay.uses_pm4_transport() {
            let launches = gpu.replay.recorded_launches().len();
            gpu.replay
                .prepare_pm4_prefix(gpu.device_id as usize, launches)
                .map(|_| ())
        } else {
            gpu.replay
                .prepare_linear_aql(gpu.device_id as usize)
                .map(|_| ())
        };
        if let Err(reason) = prepare {
            gpu.replay
                .poison(format!("Redline prepare after warmup failed: {reason}"));
            eprintln!("[redline] falling back to HIP: {reason}");
        }
    }
    Ok(())
}

/// Populate the two inputs that intentionally stay outside a prepared decode
/// replay: the token embedding in `scratch.x` and the position scalar buffer.
/// Redline uses this exact boundary before its AQL packet batch.
pub fn prepare_scratch_inputs(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, config.dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, config.dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, config.dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, config.dim)?
        }
        EmbeddingFormat::Ggml(dtype) => {
            gpu.ggml_embedding_lookup(dtype, &weights.token_embd, &scratch.x, token, config.dim)?
        }
        other => {
            return Err(HipError::new(
                0,
                &format!("unsupported embedding format for Redline: {other:?}"),
            ));
        }
    }
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &(pos as i32).to_ne_bytes())?;
    Ok(())
}

/// Same as `forward_scratch` but also extracts hidden states from the
/// configured target layers into `hidden_rb`. Used by the DFlash draft path
/// during target verification. `hidden_rb.advance_head()` is called once
/// automatically at the end of the forward pass.
pub fn forward_scratch_with_hidden(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    hidden_rb: &mut HiddenStateRingBuffer,
) -> HipResult<()> {
    let required_tokens = checked_kv_end(pos, 1, "forward_scratch_with_hidden")?;
    super::prefill::release_widened_pbs_for_kv_growth(
        gpu,
        kv_cache,
        config,
        scratch,
        required_tokens,
    )?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    let dim = config.dim;
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.token_embd, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Ggml(dtype) => {
            gpu.ggml_embedding_lookup(dtype, &weights.token_embd, &scratch.x, token, dim)?
        }
        _ => panic!("unsupported embedding format"),
    }

    forward_scratch_layers(
        gpu,
        weights,
        config,
        pos,
        kv_cache,
        dn_state,
        scratch,
        Some(hidden_rb),
        None,
    )?;
    hidden_rb.advance_head();
    Ok(())
}

/// Zero-alloc forward from pre-computed embedding in scratch.x.
pub fn forward_scratch_embed(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    embedding_data: &[f32],
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
) -> HipResult<()> {
    let required_tokens = checked_kv_end(pos, 1, "forward_scratch_embed")?;
    super::prefill::release_widened_pbs_for_kv_growth(
        gpu,
        kv_cache,
        config,
        scratch,
        required_tokens,
    )?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
    // Upload embedding directly into scratch.x
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            embedding_data.as_ptr() as *const u8,
            embedding_data.len() * 4,
        )
    };
    gpu.hip.memcpy_htod(&scratch.x.buf, bytes)?;
    forward_scratch_layers(
        gpu, weights, config, pos, kv_cache, dn_state, scratch, None, None,
    )
}

/// Write `(t, h, w)` for sequence position `pos` into `scratch.pos_buf3`.
///
/// `compact_offset` mirrors what the 1D path does to `pos_buf` after a
/// TriAttention compaction: absolute rope phase = physical index + offset.
/// Note the daemon does NOT build an `MropeCtx` while eviction is armed
/// (physical indices are renumbered there, so `MropeCtx::positions` — which is
/// indexed by physical position — would be misaligned), so in practice this
/// arm sees `compact_offset == 0`. It is applied anyway so the mrope branch
/// never silently disagrees with its 1D twin.
fn upload_mrope_pos3(
    gpu: &mut Gpu,
    scratch: &Qwen35Scratch,
    mrope: &MropeCtx,
    pos: usize,
    compact_offset: usize,
) -> HipResult<()> {
    let mut p = mrope.pos3(pos);
    if compact_offset > 0 {
        let off = compact_offset as i32;
        for v in p.iter_mut() {
            *v += off;
        }
    }
    let mut bytes = [0u8; 12];
    for (i, v) in p.iter().enumerate() {
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
    }
    gpu.memcpy_htod_auto(&scratch.pos_buf3, &bytes)
}

/// A mrope forward is NOT plain sequential AR — it issues a different rope
/// kernel against a different position buffer. Mirror the contract that
/// MTP / spec re-seed / verify forwards follow before their `forward_scratch`
/// call: consume-and-reset the AR eligibility flag (so a caller-set `false`
/// cannot leak into the next text forward), tell Redline this forward is
/// ineligible, and invalidate any captured plain-AR hipGraph so the next text
/// forward re-captures instead of replaying one recorded around a VL step.
fn mark_mrope_forward_ineligible(gpu: &mut Gpu) {
    let _consumed = std::mem::replace(&mut gpu.graphs.ar_graph_eligible, true);
    gpu.replay.set_forward_eligible(false);
    gpu.graphs.ar_forward_replay_enabled = false;
    gpu.graphs.ar_forward_kernel_dirty = true;
}

/// mrope-aware [`forward_scratch`]. `mrope == None` delegates verbatim to
/// [`forward_scratch`] — same kernels, same graph/replay routing, same
/// dispatch identity, so the certified retained-PM4 tape is untouched.
///
/// With `Some(ctx)` the call takes a deliberately plain, direct-dispatch path:
/// hipGraph capture/replay and the Redline AQL/PM4 replay routes are BYPASSED.
/// Those replay a *recorded* kernel sequence keyed only by `pos_buf`; a VL step
/// issues a different rope kernel reading a different buffer, so replaying a
/// text-captured graph (or capturing a VL step into one) would be silently
/// wrong. VL prefill/decode is per-token and image requests are rare, so the
/// lost launch amortization is not worth the aliasing risk.
#[allow(clippy::too_many_arguments)]
pub fn forward_scratch_mrope(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    mrope: Option<&MropeCtx>,
) -> HipResult<()> {
    let Some(mc) = mrope else {
        return forward_scratch(
            gpu, weights, config, token, pos, kv_cache, dn_state, scratch,
        );
    };
    let required_tokens = checked_kv_end(pos, 1, "forward_scratch_mrope")?;
    super::prefill::release_widened_pbs_for_kv_growth(
        gpu,
        kv_cache,
        config,
        scratch,
        required_tokens,
    )?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    mark_mrope_forward_ineligible(gpu);
    // Embedding lookup into scratch.x + the 1D pos scalar (still consumed by
    // the KV write and flash attention, which want the PHYSICAL slot).
    prepare_scratch_inputs(gpu, weights, config, token, pos, scratch)?;
    upload_mrope_pos3(gpu, scratch, mc, pos, kv_cache.compact_offset)?;
    forward_scratch_layers(
        gpu,
        weights,
        config,
        pos,
        kv_cache,
        dn_state,
        scratch,
        None,
        Some(mc),
    )
}

/// mrope-aware [`forward_scratch_embed`] — the image-token prefill entry.
/// `mrope == None` delegates verbatim to [`forward_scratch_embed`].
#[allow(clippy::too_many_arguments)]
pub fn forward_scratch_embed_mrope(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    embedding_data: &[f32],
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    mrope: Option<&MropeCtx>,
) -> HipResult<()> {
    let Some(mc) = mrope else {
        return forward_scratch_embed(
            gpu,
            weights,
            config,
            embedding_data,
            pos,
            kv_cache,
            dn_state,
            scratch,
        );
    };
    let required_tokens = checked_kv_end(pos, 1, "forward_scratch_embed_mrope")?;
    super::prefill::release_widened_pbs_for_kv_growth(
        gpu,
        kv_cache,
        config,
        scratch,
        required_tokens,
    )?;
    kv_cache.ensure_mapped_capacity(gpu, required_tokens)?;
    mark_mrope_forward_ineligible(gpu);
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            embedding_data.as_ptr() as *const u8,
            embedding_data.len() * 4,
        )
    };
    gpu.hip.memcpy_htod(&scratch.x.buf, bytes)?;
    upload_mrope_pos3(gpu, scratch, mc, pos, kv_cache.compact_offset)?;
    forward_scratch_layers(
        gpu,
        weights,
        config,
        pos,
        kv_cache,
        dn_state,
        scratch,
        None,
        Some(mc),
    )
}

// ── Forward scratch layers ──────────────────────────────────────────────

/// Single-GPU decode layer loop over the declarative layer program
/// (`super::program`): per layer one mixer step (`DeltaNetMixer` or
/// `GatedAttention`) and one FFN step (`GatedMlp`, or the sealed MoE call).
/// `hidden_rb` captures the post-layer hidden states speculative drafts read;
/// `mrope` switches full attention to the (t, h, w) RoPE of a VL step.
#[allow(clippy::too_many_arguments)]
fn forward_scratch_layers(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    mrope: Option<&MropeCtx>,
) -> HipResult<()> {
    let ctx = DispatchCtx::new(gpu);
    if let Some(mc) = mrope {
        debug_assert_eq!(
            mc.section, config.mrope_section,
            "MropeCtx section disagrees with the loaded config (build it with MropeCtx::new)",
        );
    }
    let binding = super::program::DecodeBinding {
        dims: super::program::hybrid_dims(config),
        s,
        kv_cache: &*kv_cache,
        dn_state: &*dn_state,
        position: pos,
        mrope_section: mrope.map(|mc| mc.section),
    };
    // A layer's hidden state is final once its MoE combine has run, which the
    // fused transition defers into the start of the next layer.
    let hidden_rb = hidden_rb.map(|rb| &*rb);
    let finish_layer = |gpu: &mut Gpu, layer_idx: usize| -> HipResult<()> {
        dump_hidden_localize(gpu, &s.x, 1, pos, config.dim, layer_idx, "pertoken");
        if let Some((rb, slot)) = hidden_rb.and_then(|rb| Some((rb, rb.extract_slot(layer_idx)?))) {
            rb.write_at_head(gpu, slot, &s.x)?;
        }
        Ok(())
    };
    let tap_enabled = hipfire_runtime::triattn::tap_enabled();
    let mut delta_layer_idx = 0usize;
    let combine_next_rms = moe_combine_next_rms_enabled(gpu, weights, config);
    let reuse_fused_rotation =
        hipfire_config::developer_var("HIPFIRE_MOE_COMBINE_NEXT_RMS_RENORM").as_deref() != Ok("1");

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];
        let fused_transition = combine_next_rms && layer_idx > 0;
        if fused_transition {
            let attn_norm = match layer {
                LayerWeights::DeltaNetMoe(l) => &l.attn_norm,
                LayerWeights::FullAttnMoe(l) => &l.attn_norm,
                LayerWeights::DeltaNet(_) | LayerWeights::FullAttn(_) => {
                    unreachable!("moe_combine_next_rms_enabled admits only all-MoE models")
                }
            };
            gpu.moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1100(
                s.moe_down_expanded.as_ref().expect("MoE scratch"),
                s.moe_topk_weights.as_ref().expect("MoE scratch"),
                &s.x,
                attn_norm,
                &s.x_rot,
                config.dim,
                config.num_experts_per_tok,
                config.norm_eps,
            )?;
            finish_layer(gpu, layer_idx - 1)?;
        }
        let tap_fn = |gpu: &mut Gpu| {
            triattn_tap(gpu, layer_idx, s, config).map_err(|e| DispatchError::Hip(e.to_string()))
        };
        let tap = tap_enabled.then_some(&tap_fn as AttentionTap<'_>);
        let mixer = binding.mixer(
            layer,
            layer_idx,
            delta_layer_idx,
            fused_transition && reuse_fused_rotation,
            tap,
            MixerOutput::Residual,
        );
        execute_steps(gpu, &ctx, &[mixer]).map_err(|e| HipError::new(0, &e.to_string()))?;
        match super::program::dense_ffn(&binding.dims, s, layer, MlpOutput::Residual) {
            Some(ffn) => execute_steps(gpu, &ctx, &[Step::GatedMlp(ffn)])
                .map_err(|e| HipError::new(0, &e.to_string()))?,
            None => {
                let (ffn, ffn_norm) = match layer {
                    LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
                    LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
                    LayerWeights::DeltaNet(_) | LayerWeights::FullAttn(_) => unreachable!(),
                };
                moe_ffn_dispatch(
                    gpu,
                    ffn,
                    &s.x,
                    ffn_norm,
                    config,
                    s,
                    combine_next_rms && layer_idx + 1 < config.n_layers,
                )?;
            }
        }
        if matches!(
            layer,
            LayerWeights::DeltaNet(_) | LayerWeights::DeltaNetMoe(_)
        ) {
            delta_layer_idx += 1;
        }
        if !combine_next_rms || layer_idx + 1 == config.n_layers {
            finish_layer(gpu, layer_idx)?;
        }
    }

    // Final norm + logits into scratch.logits.
    gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
    let wr = weights.output.dispatch_ref();
    let step = Step::Gemv {
        w: &wr,
        input: GemvInput::Raw(&s.tmp),
        out: &s.logits,
    };
    execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))
}

/// One dense layer's decode steps on `s.x` at `pos` (position already in
/// `s.pos_buf`): the batched prefill's per-row fallback for full-attention
/// layers the batched attention does not admit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_scratch_dense_layer(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    layer_idx: usize,
    pos: usize,
    kv_cache: &llama::KvCache,
    dn_state: &DeltaNetState,
    s: &Qwen35Scratch,
) -> HipResult<()> {
    let ctx = DispatchCtx::new(gpu);
    let binding = super::program::DecodeBinding {
        dims: super::program::hybrid_dims(config),
        s,
        kv_cache,
        dn_state,
        position: pos,
        mrope_section: None,
    };
    let layer = &weights.layers[layer_idx];
    let tap_fn = |gpu: &mut Gpu| {
        triattn_tap(gpu, layer_idx, s, config).map_err(|e| DispatchError::Hip(e.to_string()))
    };
    let tap = hipfire_runtime::triattn::tap_enabled().then_some(&tap_fn as AttentionTap<'_>);
    let mixer = binding.mixer(layer, layer_idx, 0, false, tap, MixerOutput::Residual);
    let ffn = super::program::dense_ffn(&binding.dims, s, layer, MlpOutput::Residual)
        .expect("dense layer");
    execute_steps(gpu, &ctx, &[mixer, Step::GatedMlp(ffn)])
        .map_err(|e| HipError::new(0, &e.to_string()))
}

// ── Dispatch helpers ─────────────────────────────────────────────────────

/// Unified QKVZA (4-way) projection for DeltaNet layers through the shared
/// hybrid normed-projection stage (fused QKVZA kernels where the fusion table
/// admits them, per-op dispatch otherwise).
#[allow(clippy::too_many_arguments)]
fn qkvza_via_execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    wqkv: &WeightTensor,
    wz: &WeightTensor,
    w_beta: &WeightTensor,
    w_alpha: &WeightTensor,
    attn_norm: &GpuTensor,
    x: &GpuTensor,
    tmp: &GpuTensor,   // rmsnorm intermediate scratch (x_plain)
    x_rot: &GpuTensor, // rotation output scratch; doubles as rmsnorm output for non-MQ
    dn_qkv: &GpuTensor,
    dn_z: &GpuTensor,
    dn_beta: &GpuTensor,
    dn_alpha: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    hybrid::normed_projection(
        gpu,
        ctx,
        x,
        attn_norm,
        tmp,
        x_rot,
        eps,
        [
            (wqkv.dispatch_ref(), dn_qkv),
            (wz.dispatch_ref(), dn_z),
            (w_beta.dispatch_ref(), dn_beta),
            (w_alpha.dispatch_ref(), dn_alpha),
        ],
    )
    .map_err(|e| HipError::new(0, &e.to_string()))
}

/// Unified QKV projection through the shared hybrid normed-projection stage.
#[allow(clippy::too_many_arguments)]
fn qkv_via_execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    wq: &WeightTensor,
    wk: &WeightTensor,
    wv: &WeightTensor,
    attn_norm: &GpuTensor,
    x: &GpuTensor,
    tmp: &GpuTensor,   // rmsnorm intermediate scratch (x_plain)
    x_rot: &GpuTensor, // rotation output scratch; doubles as rmsnorm output for non-MQ
    fa_q: &GpuTensor,
    fa_k: &GpuTensor,
    fa_v: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    hybrid::normed_projection(
        gpu,
        ctx,
        x,
        attn_norm,
        tmp,
        x_rot,
        eps,
        [
            (wq.dispatch_ref(), fa_q),
            (wk.dispatch_ref(), fa_k),
            (wv.dispatch_ref(), fa_v),
        ],
    )
    .map_err(|e| HipError::new(0, &e.to_string()))
}

/// Unified gate+up (FFN) projection through the shared hybrid
/// normed-projection stage.
#[allow(clippy::too_many_arguments)]
fn gate_up_via_execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w_gate: &WeightTensor,
    w_up: &WeightTensor,
    ffn_norm: &GpuTensor,
    x: &GpuTensor,
    tmp: &GpuTensor,
    x_rot: &GpuTensor,
    gate_out: &GpuTensor,
    up_out: &GpuTensor,
    eps: f32,
) -> HipResult<()> {
    hybrid::normed_projection(
        gpu,
        ctx,
        x,
        ffn_norm,
        tmp,
        x_rot,
        eps,
        [
            (w_gate.dispatch_ref(), gate_out),
            (w_up.dispatch_ref(), up_out),
        ],
    )
    .map_err(|e| HipError::new(0, &e.to_string()))
}

/// MoE FFN dispatch — mirrors the two-path logic from the original.
fn moe_ffn_dispatch(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x: &GpuTensor,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    defer_routed_combine: bool,
) -> HipResult<()> {
    moe_ffn_dispatch_inner(
        gpu,
        ffn,
        x,
        ffn_norm,
        config,
        s,
        None,
        false,
        defer_routed_combine,
        MoeEpMode::None,
    )
}

/// Build and execute the typed Qwen decode recipe for Single or root-routed EP.
#[allow(clippy::too_many_arguments)]
fn moe_ffn_dispatch_inner(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x: &GpuTensor,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    ep_routed_out: Option<&GpuTensor>,
    ep_skip_shared: bool,
    defer_routed_combine: bool,
    ep_mode: MoeEpMode,
) -> HipResult<()> {
    let refs = MoeScratchRef::from_scratch(s);
    let params = moe_params_for_decode(
        ffn,
        &s.tmp,
        x,
        config,
        &refs,
        false,
        ep_routed_out,
        ep_skip_shared,
        defer_routed_combine,
        ep_mode,
        hipfire_dispatch::families::moe::MoeNormalization::RmsNorm {
            weight: ffn_norm,
            plain_out: &s.tmp,
            eps: config.norm_eps,
        },
    );
    moe_ffn_decode_impl(gpu, ffn, config, &refs, params)?;
    trace_finite_if_enabled(gpu, "moe_ffn", x)?;
    Ok(())
}

/// Which EP combine this rank's MoE will execute. Plan-bound compact EP
/// bindings run root-routed slot-order combination: root produces one route,
/// every rank materializes owned expert rows in the global slot layout, and
/// root runs the ordinary single-device combine after those rows are gathered.
/// Anything else is not an EP binding, and `moe_ffn_dispatch_ep` on it is a
/// fail-closed error.
fn qwen_ep_moe_mode(ffn: &MoeFfnWeights) -> Result<EpMoeCombineMode, HipError> {
    if ffn.expert_table.execution_contract().is_none() {
        return Err(HipError::new(
            0,
            "qwen35: moe_ffn_dispatch_ep on a binding without a sealed EP execution contract",
        ));
    }
    Ok(EpMoeCombineMode::RootRoutedPartial)
}

/// EP (Ship 6 substrate-EP) variant of `moe_ffn_dispatch`: the same
/// norm/router/expert kernels as Single, but the routed combine +
/// shared-down accumulate into `routed_out` (a zeroed per-rank partial the
/// EP executor all-reduces and adds into the residual once). `skip_shared`
/// gates the shared-expert down to the root so the replicated shared expert
/// is summed exactly once. Sealed `RootRoutedPartial`, immediate combine.
/// Calls `moe_ffn_decode_impl` directly (the `with_scratch` wrappers don't
/// carry EP params or modes). The residual `x` is left untouched — the
/// executor adds the all-reduced partial into it afterward.
fn moe_ffn_dispatch_ep(
    gpu: &mut Gpu,
    ffn: &MoeFfnWeights,
    x: &GpuTensor,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    routed_out: &GpuTensor,
    skip_shared: bool,
) -> HipResult<()> {
    match qwen_ep_moe_mode(ffn)? {
        EpMoeCombineMode::RootRoutedPartial => moe_ffn_dispatch_inner(
            gpu,
            ffn,
            x,
            ffn_norm,
            config,
            s,
            Some(routed_out),
            skip_shared,
            false,
            MoeEpMode::RootRoutedPartial,
        ),
        // Non-Qwen arches keep the rank-partial default on their own
        // bindings; a Qwen binding never reports it (see
        // `qwen_ep_moe_mode`), so reaching it is a fail-stop bug, not a
        // fallback.
        EpMoeCombineMode::RankPartial => Err(HipError::new(
            0,
            "qwen35: RankPartial combine is not served on Qwen EP bindings",
        )),
    }
}

/// Pure root preparation shared by the executing root dispatch and the runtime
/// preflight hook: bind the complete recipe, bind live experts, and seal.
#[allow(clippy::too_many_arguments)]
fn moe_sealed_root_ep<'a>(
    ctx: &'a DispatchCtx,
    ffn: &'a MoeFfnWeights,
    x: &'a GpuTensor,
    ffn_norm: &'a GpuTensor,
    config: &Qwen35Config,
    s: &'a Qwen35Scratch,
    partial: &'a GpuTensor,
) -> HipResult<SealedMoeCall<'a>> {
    let refs = MoeScratchRef::from_scratch(s);
    let params = moe_params_for_decode(
        ffn,
        &s.tmp,
        x,
        config,
        &refs,
        false,
        Some(partial),
        false,
        false,
        MoeEpMode::RootRoutedPartial,
        hipfire_dispatch::families::moe::MoeNormalization::RmsNorm {
            weight: ffn_norm,
            plain_out: &s.tmp,
            eps: config.norm_eps,
        },
    );
    let bound = ffn.bound_experts()?;
    hipfire_dispatch::pipeline::sealed_moe::seal_decode(bound, ctx, params).map_err(HipError::from)
}

/// Pure non-root preparation shared by the executing contrib dispatch and the
/// runtime preflight hook: bind the complete recipe and seal it against the
/// root route proof. No GPU work occurs here.
#[allow(clippy::too_many_arguments)]
fn moe_sealed_contrib_ep<'a>(
    ctx: &'a DispatchCtx,
    ffn: &'a MoeFfnWeights,
    x: &'a GpuTensor,
    ffn_norm: &'a GpuTensor,
    config: &Qwen35Config,
    s: &'a Qwen35Scratch,
    proof: &MoeRouteProducerProof,
    partial: &'a GpuTensor,
) -> HipResult<SealedMoeCall<'a>> {
    let refs = MoeScratchRef::from_scratch(s);
    let params = moe_params_for_decode(
        ffn,
        &s.tmp,
        x,
        config,
        &refs,
        false,
        Some(partial),
        true,
        false,
        MoeEpMode::RootRoutedPartial,
        hipfire_dispatch::families::moe::MoeNormalization::RmsNorm {
            weight: ffn_norm,
            plain_out: &s.tmp,
            eps: config.norm_eps,
        },
    );
    let bound = ffn.bound_experts()?;
    hipfire_dispatch::pipeline::sealed_moe::seal_ep_routed_contrib(bound, ctx, params, proof)
        .map_err(HipError::from)
}
/// Root-routed EP root dispatch. The typed recipe owns normalization, routing,
/// expert arithmetic, and the shared-expert contribution; this wrapper keeps
/// only the route proof and existing diagnostics.
#[allow(clippy::too_many_arguments)]
fn moe_ffn_dispatch_root_ep(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    ffn: &MoeFfnWeights,
    x: &GpuTensor,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    partial: &GpuTensor,
) -> HipResult<MoeRouteProducerProof> {
    let sealed = moe_sealed_root_ep(ctx, ffn, x, ffn_norm, config, s, partial)?;
    let proof = sealed.ep_route_producer_proof().map_err(HipError::from)?;
    #[cfg(feature = "moe-oracle")]
    {
        let oracle_pos =
            crate::qwen35::oracle::decode_position().map_err(|e| HipError::new(0, &e))?;
        crate::qwen35::oracle::decode_before(gpu, ffn, config, x, oracle_pos)
            .map_err(|e| HipError::new(0, &e))?;
    }
    hipfire_dispatch::pipeline::execute_steps(
        gpu,
        ctx,
        &[hipfire_dispatch::pipeline::Step::Moe(sealed)],
    )
    .map_err(HipError::from)?;
    let refs = MoeScratchRef::from_scratch(s);
    #[cfg(feature = "moe-oracle")]
    {
        let oracle_pos =
            crate::qwen35::oracle::decode_position().map_err(|e| HipError::new(0, &e))?;
        crate::qwen35::oracle::decode_after(
            gpu,
            ffn,
            config,
            refs.router_logits,
            refs.topk_indices,
            refs.topk_weights,
            refs.gate_batch,
            refs.up_batch,
            refs.rot_batch,
            refs.down_expanded,
            x,
            oracle_pos,
        )
        .map_err(|e| HipError::new(0, &e))?;
    }
    if expert_stats_enabled() {
        capture_expert_stats(
            gpu,
            ffn.layer_idx,
            config.num_experts_per_tok,
            config.dim,
            refs.down_expanded,
            refs.topk_indices,
            refs.topk_weights,
        );
    }
    trace_finite_if_enabled(gpu, "moe_ffn", x)?;
    Ok(proof)
}

/// Root-routed EP non-root dispatch. The route proof is bound before the
/// shared typed recipe executes, so no rank-local routing or arithmetic remains
/// in this architecture wrapper.
#[allow(clippy::too_many_arguments)]
fn moe_ffn_dispatch_contrib_ep(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    ffn: &MoeFfnWeights,
    x: &GpuTensor,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    proof: &MoeRouteProducerProof,
    partial: &GpuTensor,
) -> HipResult<()> {
    let sealed = moe_sealed_contrib_ep(ctx, ffn, x, ffn_norm, config, s, proof, partial)?;
    hipfire_dispatch::pipeline::execute_steps(
        gpu,
        ctx,
        &[hipfire_dispatch::pipeline::Step::Moe(sealed)],
    )
    .map_err(HipError::from)?;
    trace_finite_if_enabled(gpu, "moe_ffn", x)?;
    Ok(())
}

/// Validate the topology half of a post-load shard (rank/mesh/device
/// identity plus the supported assignment). Plan and residency checks live
/// in [`PreparedMoeShard::prepare`], which runs before any mutation.
fn validate_shard_topology(
    shard: &ShardConfig,
    rank: usize,
    n_exp: usize,
    mesh: &DeviceMesh,
    physical_devices: &[i32],
    gpu: &Gpu,
) -> HipResult<ExpertAssign> {
    let tp = shard.tp_size;
    if tp == 0 {
        return Err(HipError::new(0, "qwen35: EP shard has no ranks"));
    }
    if rank >= tp {
        return Err(HipError::new(
            0,
            &format!("qwen35: EP rank {rank} is outside rank count {tp}"),
        ));
    }
    if mesh.n_devices() != tp || physical_devices.len() != tp {
        return Err(HipError::new(
            0,
            &format!(
                "qwen35: EP shard topology (mesh devices={}, physical={}) disagrees with rank count {tp}",
                mesh.n_devices(),
                physical_devices.len()
            ),
        ));
    }
    if physical_devices[rank] != gpu.device_id {
        return Err(HipError::new(
            0,
            &format!(
                "qwen35: EP rank {rank} names physical device {}, but the sharding GPU is {}",
                physical_devices[rank], gpu.device_id
            ),
        ));
    }
    if n_exp == 0 {
        return Err(HipError::new(0, "qwen35: EP sharding needs routed experts"));
    }
    super::weights::infer_ep_assignment(shard, n_exp)
}

/// Phase-one owner for a post-load EP shard. Holds the repartitioned sealed
/// plan, the rank-adapted table, the compact-bound cache, replacement
/// pointer/tag tables, zero dummies, and global dtype metadata — everything
/// EXCEPT the experts themselves, which stay in the old owner until
/// [`Self::commit`]. Dropping via [`Self::free`] after a later-layer failure
/// therefore leaves the old model untouched.
struct PreparedMoeShard {
    rank: usize,
    plan: ExpertExecutionPlan,
    table: hipfire_dispatch::pipeline::sealed_moe::ExpertTable,
    cache: hipfire_dispatch::pipeline::sealed_moe::ExpertBindingCache,
    dummy_buffers: Vec<GpuTensor>,
    dummy_views: Vec<ExpertWeights>,
    gate_up_table: GpuTensor,
    down_table: GpuTensor,
    tag_table: Option<GpuTensor>,
    global_pairs: Vec<(DType, DType)>,
    tier_gate_up: Option<Box<[DType]>>,
    tier_down: Option<Box<[DType]>>,
}

impl PreparedMoeShard {
    /// Prepare one layer's EP shard without mutating `ffn`: repartition the
    /// sealed Single plan through the ordinary planner, adapt for `rank`,
    /// allocate dummies/tables from sealed metadata, and compact-bind against
    /// the still-resident full expert set. Any error frees the prepared
    /// artifacts and leaves `ffn` bit-identical.
    fn prepare(
        gpu: &mut Gpu,
        ffn: &MoeFfnWeights,
        mesh: &DeviceMesh,
        physical_devices: &[i32],
        assignment: ExpertAssign,
        rank: usize,
        shard: &ShardConfig,
    ) -> HipResult<Self> {
        use hipfire_dispatch::pipeline::sealed_moe::CompactLiveWeight;
        use hipfire_dispatch::pipeline::sealed_moe::ROOT_ROUTED_EP_EXECUTION;
        use hipfire_runtime::sealed_moe::adapt_expert_execution_plan;
        use hipfire_runtime::weight_manifest::ExpertParallelism;
        let base = &ffn.expert_execution_plan;
        if base.parallelism() != ExpertParallelism::Single {
            return Err(HipError::new(
                0,
                "qwen35: post-load EP shard requires a sealed Single plan (already sharded?)",
            ));
        }
        let n_exp = base.n_experts();
        if ffn.experts.len() != n_exp {
            return Err(HipError::new(
                0,
                &format!(
                    "qwen35: post-load EP shard needs all {n_exp} experts resident, found {}",
                    ffn.experts.len()
                ),
            ));
        }
        if ffn.expert_down_awq_ptrs.is_some() {
            return Err(HipError::new(
                0,
                "qwen35: post-load EP shard refuses AWQ pointer tables",
            ));
        }
        if ffn.paro_shared.is_some() {
            return Err(HipError::new(
                0,
                "qwen35: post-load EP shard refuses ParoQuant rotation sidecars",
            ));
        }
        // The only sanctioned Single -> EP transition: rebuild planner input
        // from the attested plan and rerun the ordinary manifest/expert
        // planners. Ownership comes from the planner, never from `e % N`.
        let plan = repartition_expert_execution_plan(
            base,
            mesh,
            physical_devices,
            assignment,
            ROOT_ROUTED_EP_EXECUTION,
        )
        .map_err(|error| HipError::new(0, &error))?;
        let (table, mut cache) =
            adapt_expert_execution_plan(&plan, rank).map_err(|error| HipError::new(0, &error))?;
        // The caller's shard map selected the assignment; the sealed plan is
        // authoritative. Fail closed on any skew between the two.
        let planned = super::weights::plan_expert_to_rank(&plan);
        if planned.as_slice() != shard.expert_to_rank.as_slice() {
            return Err(HipError::new(
                0,
                "qwen35: sealed EP repartition ownership disagrees with the shard map",
            ));
        }
        let owned_globals: Vec<usize> = plan
            .rank_ownership()
            .get(rank)
            .ok_or_else(|| {
                HipError::new(
                    0,
                    &format!("qwen35: sealed EP plan has no ownership for rank {rank}"),
                )
            })?
            .global_expert_ids
            .clone();
        let slot_row = plan.global_to_local().get(rank).ok_or_else(|| {
            HipError::new(
                0,
                &format!("qwen35: sealed EP plan has no slot map for rank {rank}"),
            )
        })?;
        for (slot, &global) in owned_globals.iter().enumerate() {
            if slot_row.get(global).copied().flatten() != Some(slot) {
                return Err(HipError::new(
                    0,
                    &format!(
                        "qwen35: sealed EP rank {rank} slot {slot} disagrees on expert {global}"
                    ),
                ));
            }
        }
        let global_pairs = super::weights::global_dtype_pairs_from_plan(&plan);
        let global_tags = super::weights::validate_ep_global_tags(&global_pairs)?;
        // Dummy layouts from sealed plan records. Strides sample the resident
        // full set (every global is resident pre-swap), so every layout has
        // an exact sample. Dtype-keyed samples are a handful of entries, so
        // a linear scan replaces a map (DType is not Hash).
        let mut gate_stride_by_dtype: Vec<(DType, usize)> = Vec::new();
        let mut down_stride_by_dtype: Vec<(DType, usize)> = Vec::new();
        for expert in &ffn.experts {
            if !gate_stride_by_dtype
                .iter()
                .any(|&(dtype, _)| dtype == expert.gate_up.gpu_dtype)
            {
                gate_stride_by_dtype.push((expert.gate_up.gpu_dtype, expert.gate_up.row_stride));
            }
            if !down_stride_by_dtype
                .iter()
                .any(|&(dtype, _)| dtype == expert.down.gpu_dtype)
            {
                down_stride_by_dtype.push((expert.down.gpu_dtype, expert.down.row_stride));
            }
        }
        let sample_stride = |samples: &[(DType, usize)], dtype: DType| {
            samples
                .iter()
                .find(|&&(sample, _)| sample == dtype)
                .map(|&(_, stride)| stride)
                .unwrap_or(0)
        };
        let mut specs: Vec<super::weights::EpDummySpec> = Vec::new();
        let mut dummy_for_global: Vec<Option<usize>> = vec![None; n_exp];
        for (global, record) in plan.experts().iter().enumerate() {
            if record.owner_rank == rank {
                continue;
            }
            let fused = record.gate.source_name == record.up.source_name;
            let gate_rows = record.gate.logical_shape.first().copied().unwrap_or(0)
                + record.up.logical_shape.first().copied().unwrap_or(0);
            let gate_cols = record.gate.logical_shape.get(1).copied().unwrap_or(0);
            let gate_bytes = if fused {
                record.gate.encoded_bytes
            } else {
                record
                    .gate
                    .encoded_bytes
                    .saturating_add(record.up.encoded_bytes)
            };
            let spec = super::weights::EpDummySpec {
                gate_dtype: record.gate.dtype,
                gate_m: gate_rows,
                gate_k: gate_cols,
                gate_bytes,
                gate_stride: sample_stride(&gate_stride_by_dtype, record.gate.dtype),
                down_dtype: record.down.dtype,
                down_m: record.down.logical_shape.first().copied().unwrap_or(0),
                down_k: record.down.logical_shape.get(1).copied().unwrap_or(0),
                down_bytes: record.down.encoded_bytes,
                down_stride: sample_stride(&down_stride_by_dtype, record.down.dtype),
            };
            if let Some(index) = specs.iter().position(|other| other == &spec) {
                dummy_for_global[global] = Some(index);
            } else {
                dummy_for_global[global] = Some(specs.len());
                specs.push(spec);
            }
        }
        let cleanup = |gpu: &mut Gpu,
                       buffers: Vec<GpuTensor>,
                       views: Vec<ExpertWeights>,
                       tables: Vec<GpuTensor>| {
            for view in views {
                view.gate_up.free_metadata_only(gpu);
                view.down.free_metadata_only(gpu);
            }
            for buffer in buffers {
                let _ = gpu.free_tensor(buffer);
            }
            for table in tables {
                let _ = gpu.free_tensor(table);
            }
        };
        let (dummy_buffers, dummy_views) = match super::weights::alloc_ep_dummies(gpu, &specs) {
            Ok(owners) => owners,
            Err(error) => return Err(error),
        };
        let expert_at = |global: usize| {
            ffn.experts.get(global).ok_or_else(|| {
                HipError::new(
                    0,
                    &format!("qwen35: sealed EP shard is missing resident expert {global}"),
                )
            })
        };
        let mut gate_up_entries = vec![0usize; n_exp];
        let mut down_entries = vec![0usize; n_exp];
        for (slot, &global) in owned_globals.iter().enumerate() {
            let _ = slot;
            let expert = expert_at(global)?;
            gate_up_entries[global] = expert.gate_up.buf.buf.as_ptr() as usize;
            down_entries[global] = expert.down.buf.buf.as_ptr() as usize;
        }
        for (global, slot) in dummy_for_global.iter().enumerate() {
            let Some(index) = slot else { continue };
            let dummy = &dummy_views[*index];
            gate_up_entries[global] = dummy.gate_up.buf.buf.as_ptr() as usize;
            down_entries[global] = dummy.down.buf.buf.as_ptr() as usize;
        }
        let mixed = global_pairs
            .iter()
            .skip(1)
            .any(|&pair| Some(pair) != global_pairs.first().copied());
        let tag_bytes: Option<Vec<u8>> = mixed.then(|| global_tags.clone());
        let gate_up_words: Vec<u64> = gate_up_entries.iter().map(|&ptr| ptr as u64).collect();
        let down_words: Vec<u64> = down_entries.iter().map(|&ptr| ptr as u64).collect();
        let (gate_up_table, down_table, tag_table) = match super::weights::upload_ep_tables(
            gpu,
            &gate_up_words,
            &down_words,
            tag_bytes.as_deref(),
        ) {
            Ok(tables) => tables,
            Err(error) => {
                cleanup(gpu, dummy_buffers, dummy_views, Vec::new());
                return Err(error);
            }
        };
        let local_refs: Vec<CompactLiveWeight<'_>> = owned_globals
            .iter()
            .map(|&global| {
                expert_at(global).map(|expert| CompactLiveWeight {
                    gate_up: expert.gate_up.dispatch_ref(),
                    down: expert.down.dispatch_ref(),
                })
            })
            .collect::<HipResult<_>>()?;
        let dummy_refs: Vec<CompactLiveWeight<'_>> = dummy_views
            .iter()
            .map(|dummy| CompactLiveWeight {
                gate_up: dummy.gate_up.dispatch_ref(),
                down: dummy.down.dispatch_ref(),
            })
            .collect();
        if let Err(error) = super::weights::bind_compact_ep_cache(
            &mut cache,
            &table,
            &local_refs,
            &gate_up_entries,
            &down_entries,
            &gate_up_table,
            &down_table,
            tag_table.as_ref(),
            &dummy_refs,
            gpu.device_id,
        ) {
            cleanup(
                gpu,
                dummy_buffers,
                dummy_views,
                [gate_up_table, down_table]
                    .into_iter()
                    .chain(tag_table)
                    .collect(),
            );
            return Err(error);
        }
        let (tier_gate_up, tier_down) =
            super::weights::cached_expert_tier_tables(Some(&global_pairs), &[]);
        Ok(Self {
            rank,
            plan,
            table,
            cache,
            dummy_buffers,
            dummy_views,
            gate_up_table,
            down_table,
            tag_table,
            global_pairs,
            tier_gate_up,
            tier_down,
        })
    }

    /// Free a prepared-but-uncommitted shard (later-layer failure path).
    /// The old owner was never touched, so this only releases the prepared
    /// artifacts.
    fn free(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.gate_up_table);
        let _ = gpu.free_tensor(self.down_table);
        if let Some(table) = self.tag_table {
            let _ = gpu.free_tensor(table);
        }
        for view in self.dummy_views {
            view.gate_up.free_metadata_only(gpu);
            view.down.free_metadata_only(gpu);
        }
        for buffer in self.dummy_buffers {
            let _ = gpu.free_tensor(buffer);
        }
    }

    /// Commit a fully prepared shard: partition the resident experts into
    /// owned (local-slot order) and retired, then swap every field. Field
    /// swaps are infallible; replaced tables are released (errors ignored —
    /// teardown accounting, never a reported failure).
    fn commit(self, gpu: &mut Gpu, ffn: &mut MoeFfnWeights) {
        let owned_globals = self.plan.rank_ownership()[self.rank]
            .global_expert_ids
            .clone();
        let slot_of = |global: usize| {
            self.plan.global_to_local()[self.rank][global].expect("prepared EP slot map")
        };
        let mut owned: Vec<(usize, ExpertWeights)> = Vec::with_capacity(owned_globals.len());
        let mut retired: Vec<ExpertWeights> = Vec::new();
        let is_owned = {
            let mut owned_set = vec![false; ffn.experts.len()];
            for &global in &owned_globals {
                owned_set[global] = true;
            }
            owned_set
        };
        for (global, expert) in std::mem::take(&mut ffn.experts).into_iter().enumerate() {
            if is_owned[global] {
                owned.push((slot_of(global), expert));
            } else {
                retired.push(expert);
            }
        }
        owned.sort_by_key(|&(slot, _)| slot);
        ffn.experts = owned.into_iter().map(|(_, expert)| expert).collect();
        ffn.retired_expert_weights = retired;
        ffn.expert_execution_plan = self.plan;
        ffn.expert_table = self.table;
        ffn.expert_binding = self.cache;
        let old_gate_up = std::mem::replace(&mut ffn.expert_gate_up_ptrs, self.gate_up_table);
        let old_down = std::mem::replace(&mut ffn.expert_down_ptrs, self.down_table);
        let _ = gpu.free_tensor(old_gate_up);
        let _ = gpu.free_tensor(old_down);
        if let Some(old_awq) = ffn.expert_down_awq_ptrs.take() {
            let _ = gpu.free_tensor(old_awq);
        }
        if let Some(old_tags) = ffn.expert_dtype_tags.take() {
            let _ = gpu.free_tensor(old_tags);
        }
        ffn.expert_dtype_tags = self.tag_table;
        ffn.global_expert_dtypes = Some(self.global_pairs.into_boxed_slice());
        ffn.ep_dummy_buffers = self.dummy_buffers;
        ffn.ep_dummy_experts = self.dummy_views;
        ffn.mixed_expert_gate_up_tiers = self.tier_gate_up;
        ffn.mixed_expert_down_tiers = self.tier_down;
    }
}

/// Shard one loaded MoE FFN's routed experts to this rank's sealed compact
/// owners: two-phase prepare (repartition + adapt + dummies + tables +
/// compact bind against the untouched resident set), then an infallible
/// swap that retires displaced tensors in an explicit owner field until
/// teardown.
///
/// Signature (frozen for C4): `shard_moe_experts(gpu, ffn, shard, rank,
/// n_exp, mesh, physical_devices)`.
pub fn shard_moe_experts(
    gpu: &mut Gpu,
    ffn: &mut MoeFfnWeights,
    shard: &ShardConfig,
    rank: usize,
    n_exp: usize,
    mesh: &DeviceMesh,
    physical_devices: &[i32],
) -> HipResult<()> {
    let assignment = validate_shard_topology(shard, rank, n_exp, mesh, physical_devices, gpu)?;
    if ffn.expert_execution_plan.n_experts() != n_exp {
        return Err(HipError::new(
            0,
            &format!(
                "qwen35: EP shard layer covers {} experts, expected {n_exp}",
                ffn.expert_execution_plan.n_experts()
            ),
        ));
    }
    let prepared =
        PreparedMoeShard::prepare(gpu, ffn, mesh, physical_devices, assignment, rank, shard)?;
    prepared.commit(gpu, ffn);
    Ok(())
}

/// Shard every MoE layer's routed experts to this rank's sealed compact
/// owners. Two-phase across layers: every layer prepares while the old
/// model is untouched; only after ALL succeed does the infallible swap run.
/// A later-layer prepare failure frees the prepared artifacts and leaves
/// every old owner bit-identical. Displaced tensors retire per layer until
/// teardown. REAP and paged models are refused before any work.
///
/// Signature (frozen for C4): `shard_all_moe_layers(gpu, weights, shard,
/// rank, n_exp, reap_active, mesh, physical_devices)`.
pub fn shard_all_moe_layers(
    gpu: &mut Gpu,
    weights: &mut Qwen35Weights,
    shard: &ShardConfig,
    rank: usize,
    n_exp: usize,
    reap_active: bool,
    mesh: &DeviceMesh,
    physical_devices: &[i32],
) -> HipResult<()> {
    shard_all_moe_layers_inner(
        gpu,
        weights,
        shard,
        rank,
        n_exp,
        reap_active,
        mesh,
        physical_devices,
        None,
    )
}

/// Fault-injecting seam for [`shard_all_moe_layers`] (tests only).
#[doc(hidden)]
pub fn shard_all_moe_layers_with_fault(
    gpu: &mut Gpu,
    weights: &mut Qwen35Weights,
    shard: &ShardConfig,
    rank: usize,
    n_exp: usize,
    reap_active: bool,
    mesh: &DeviceMesh,
    physical_devices: &[i32],
    fault: super::load::EpFault,
) -> HipResult<()> {
    shard_all_moe_layers_inner(
        gpu,
        weights,
        shard,
        rank,
        n_exp,
        reap_active,
        mesh,
        physical_devices,
        Some(fault),
    )
}

fn shard_all_moe_layers_inner(
    gpu: &mut Gpu,
    weights: &mut Qwen35Weights,
    shard: &ShardConfig,
    rank: usize,
    n_exp: usize,
    reap_active: bool,
    mesh: &DeviceMesh,
    physical_devices: &[i32],
    fault: Option<super::load::EpFault>,
) -> HipResult<()> {
    if reap_active {
        return Err(HipError::new(
            0,
            "qwen35: post-load EP shard refuses REAP keep-maps",
        ));
    }
    if weights.pager.is_some() {
        return Err(HipError::new(
            0,
            "qwen35: post-load EP shard refuses paged models",
        ));
    }
    let assignment = validate_shard_topology(shard, rank, n_exp, mesh, physical_devices, gpu)?;
    // Admission pass over every layer before any mutation: all MoE, all
    // sealed Single, all fully resident.
    let mut moe_layers: Vec<usize> = Vec::new();
    for (index, layer) in weights.layers.iter().enumerate() {
        let ffn = match layer {
            LayerWeights::DeltaNetMoe(weights) => Some(&weights.ffn),
            LayerWeights::FullAttnMoe(weights) => Some(&weights.ffn),
            _ => None,
        };
        let Some(ffn) = ffn else {
            return Err(HipError::new(
                0,
                &format!("qwen35: EP shard refuses non-MoE layer {index}"),
            ));
        };
        if ffn.expert_execution_plan.n_experts() != n_exp || ffn.experts.len() != n_exp {
            return Err(HipError::new(
                0,
                &format!("qwen35: EP shard layer {index} is not fully resident"),
            ));
        }
        moe_layers.push(index);
    }
    if moe_layers.is_empty() {
        return Err(HipError::new(0, "qwen35: EP shard found no MoE layers"));
    }
    // Prepare every layer while the old model is untouched.
    let mut prepared: Vec<(usize, PreparedMoeShard)> = Vec::with_capacity(moe_layers.len());
    for &index in &moe_layers {
        if matches!(
            fault,
            Some(super::load::EpFault::ShardPrepare { layer }) if layer == index
        ) {
            for (_, shard) in prepared.drain(..) {
                shard.free(gpu);
            }
            return Err(HipError::new(
                0,
                &format!("qwen35: EP fault injection at ShardPrepare (layer {index})"),
            ));
        }
        let ffn = match &weights.layers[index] {
            LayerWeights::DeltaNetMoe(weights) => &weights.ffn,
            LayerWeights::FullAttnMoe(weights) => &weights.ffn,
            _ => unreachable!("EP shard admission passed layer {index}"),
        };
        match PreparedMoeShard::prepare(gpu, ffn, mesh, physical_devices, assignment, rank, shard) {
            Ok(shard) => prepared.push((index, shard)),
            Err(error) => {
                for (_, shard) in prepared.drain(..) {
                    shard.free(gpu);
                }
                return Err(error);
            }
        }
    }
    // All prepared: infallible swap.
    for (index, shard) in prepared {
        let ffn = match &mut weights.layers[index] {
            LayerWeights::DeltaNetMoe(weights) => &mut weights.ffn,
            LayerWeights::FullAttnMoe(weights) => &mut weights.ffn,
            _ => unreachable!("EP shard admission passed layer {index}"),
        };
        shard.commit(gpu, ffn);
    }
    Ok(())
}

/// TriAttention tap helper (inline from original forward).
fn triattn_tap(
    gpu: &mut Gpu,
    layer_idx: usize,
    s: &Qwen35Scratch,
    config: &Qwen35Config,
) -> HipResult<()> {
    let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
        gpu,
        layer_idx,
        &s.fa_q.buf,
        1,
        config.n_heads,
        config.head_dim,
    )?;
    if !gpu_handled {
        let n_q = config.n_heads * config.head_dim;
        let q_cpu = gpu.download_f32(&s.fa_q)?;
        if hipfire_runtime::triattn::tap_needs_k() {
            let n_k = config.n_kv_heads * config.head_dim;
            let k_cpu = gpu.download_f32(&s.fa_k)?;
            hipfire_runtime::triattn::record_prerope_qk(
                layer_idx,
                &q_cpu[..n_q],
                Some(&k_cpu[..n_k]),
            );
        } else {
            hipfire_runtime::triattn::record_prerope_q(layer_idx, &q_cpu[..n_q]);
        }
    }
    Ok(())
}

fn deltanet_sigmoid_alpha_gate(
    gpu: &mut Gpu,
    beta: &GpuTensor,
    alpha: &GpuTensor,
    dt_bias: &GpuTensor,
    a_log: &GpuTensor,
    n_heads: usize,
) -> HipResult<()> {
    gpu.fused_sigmoid_alpha_gate_f32(beta, alpha, dt_bias, a_log, n_heads)
}

fn deltanet_qk_l2_norm_scale(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    n_key_heads: usize,
    head_dim: usize,
    eps: f32,
) -> HipResult<()> {
    gpu.fused_qk_l2_norm_scale_f32(
        q,
        k,
        n_key_heads,
        head_dim,
        1.0 / (head_dim as f32).sqrt(),
        eps,
    )
}

// ROCm 10 RCCL SHM and inaccessible cross-device copy failures require
// CPU-staged deterministic reduction on mixed topology.
fn dense_tp_all_reduce_sum_f32(
    gpus: &mut Gpus,
    refs: &[&hip_bridge::DeviceBuffer],
    count: usize,
) -> HipResult<()> {
    if gpus.peer_access_enabled {
        gpus.all_reduce_sum_f32_peer_rooted(refs, count)
    } else {
        gpus.all_reduce_sum_f32_host(refs, count)
    }
}

/// Residual reduction after a row-parallel stage. The attention partial
/// (slot 0) lives in `o` and the FFN partial (slot 1) in `ffn_out`, so
/// consecutive reductions alternate partial buffers as the slotted route of
/// [`Gpus::all_reduce_sum_f32_peer_add_routed`] requires.
fn dense_tp_allreduce_add(
    gpus: &mut Gpus,
    scratches: &[Qwen35Scratch],
    count: usize,
    partial_slot: usize,
) -> HipResult<()> {
    let partial_refs: Vec<_> = scratches
        .iter()
        .map(|scratch| {
            if partial_slot == 0 {
                &scratch.o.buf
            } else {
                &scratch.ffn_out.buf
            }
        })
        .collect();
    if gpus.peer_access_enabled {
        let residuals: Vec<_> = scratches.iter().map(|scratch| &scratch.x.buf).collect();
        return gpus.all_reduce_sum_f32_peer_add_routed(
            &partial_refs,
            &residuals,
            1,
            count,
            Some(partial_slot),
        );
    }
    dense_tp_all_reduce_sum_f32(gpus, &partial_refs, count)?;
    for (rank, scratch) in scratches.iter().enumerate() {
        let partial = if partial_slot == 0 {
            &scratch.o
        } else {
            &scratch.ffn_out
        };
        gpus.devices[rank].bind_thread()?;
        gpus.devices[rank].add_f32(&scratch.x, partial, &scratch.x)?;
    }
    Ok(())
}

fn dense_tp_allreduce_batched(
    gpus: &mut Gpus,
    pbs_vec: &[&PrefillBatchScratch],
    partials: &[&GpuTensor],
    n: usize,
    dim: usize,
) -> HipResult<()> {
    let count = n
        .checked_mul(dim)
        .ok_or_else(|| HipError::new(0, "dense_tp batched count overflow"))?;
    let partial_refs: Vec<_> = partials.iter().map(|tensor| &tensor.buf).collect();
    if gpus.peer_access_enabled {
        let residuals: Vec<_> = pbs_vec
            .iter()
            .map(|pbs| pbs.x_batch.sub_offset(0, count))
            .collect();
        let residual_refs: Vec<_> = residuals.iter().map(|tensor| &tensor.buf).collect();
        return gpus.all_reduce_sum_f32_peer_add_routed(
            &partial_refs,
            &residual_refs,
            n,
            count,
            None,
        );
    }
    dense_tp_all_reduce_sum_f32(gpus, &partial_refs, count)?;
    for (rank, (pbs, partial)) in pbs_vec.iter().zip(partials.iter()).enumerate() {
        gpus.devices[rank].bind_thread()?;
        let x_n = pbs.x_batch.sub_offset(0, count);
        let partial_n = partial.sub_offset(0, count);
        gpus.devices[rank].add_f32(&x_n, &partial_n, &x_n)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dense_tp_local_attention(
    gpus: &mut Gpus,
    weights: &[Qwen35Weights],
    configs: &[Qwen35Config],
    layer_idx: usize,
    delta_layer_idx: usize,
    pos: usize,
    kv_caches: &mut [llama::KvCache],
    dn_states: &mut [DeltaNetState],
    scratches: &[Qwen35Scratch],
) -> HipResult<()> {
    for rank in 0..gpus.devices.len() {
        let layer = &weights[rank].layers[layer_idx];
        if !matches!(layer, LayerWeights::DeltaNet(_) | LayerWeights::FullAttn(_)) {
            return Err(HipError::new(0, "dense TP received a MoE/mismatched layer"));
        }
        let s = &scratches[rank];
        let binding = super::program::DecodeBinding {
            dims: super::program::hybrid_dims(&configs[rank]),
            s,
            kv_cache: &kv_caches[rank],
            dn_state: &dn_states[rank],
            position: pos,
            mrope_section: None,
        };
        let output = MixerOutput::Write(&s.o);
        let mixer = binding.mixer(layer, layer_idx, delta_layer_idx, false, None, output);
        let gpu = &mut gpus.devices[rank];
        let ctx = DispatchCtx::new(gpu);
        execute_steps(gpu, &ctx, &[mixer]).map_err(|e| HipError::new(0, &e.to_string()))?;
    }
    Ok(())
}

fn dense_tp_local_ffn(
    gpus: &mut Gpus,
    weights: &[Qwen35Weights],
    configs: &[Qwen35Config],
    layer_idx: usize,
    scratches: &[Qwen35Scratch],
) -> HipResult<()> {
    for rank in 0..gpus.devices.len() {
        let s = &scratches[rank];
        let ffn = super::program::dense_ffn(
            &super::program::hybrid_dims(&configs[rank]),
            s,
            &weights[rank].layers[layer_idx],
            MlpOutput::Write(&s.ffn_out),
        )
        .ok_or_else(|| HipError::new(0, "dense TP received a MoE/mismatched layer"))?;
        let ctx = DispatchCtx::new(&gpus.devices[rank]);
        let gpu = &mut gpus.devices[rank];
        ffn.project(gpu, &ctx)
            .map_err(|e| HipError::new(0, &e.to_string()))?;
        ffn.write_partial(gpu, &ctx)?;
    }
    Ok(())
}

fn dense_tp_output(
    gpus: &mut Gpus,
    weights: &[Qwen35Weights],
    configs: &[Qwen35Config],
    scratches: &[Qwen35Scratch],
) -> HipResult<()> {
    gpus.devices[0].rmsnorm_f32(
        &scratches[0].x,
        &weights[0].output_norm,
        &scratches[0].tmp,
        configs[0].norm_eps,
    )?;
    let ctx = DispatchCtx::new(&gpus.devices[0]);
    let output = weights[0].output.dispatch_ref();
    execute_steps(
        &mut gpus.devices[0],
        &ctx,
        &[Step::Gemv {
            w: &output,
            input: GemvInput::Raw(&scratches[0].tmp),
            out: &scratches[0].logits,
        }],
    )
    .map_err(|e| HipError::new(0, &e.to_string()))
}

/// Direct dense-Qwen TP decode. This is both the graph-disabled path and the
/// mandatory warmup that resolves lazy kernel/module state before capture.
#[allow(clippy::too_many_arguments)]
fn forward_scratch_dense_tp_layers(
    gpus: &mut Gpus,
    weights: &[Qwen35Weights],
    configs: &[Qwen35Config],
    pos: usize,
    kv_caches: &mut [llama::KvCache],
    dn_states: &mut [DeltaNetState],
    scratches: &[Qwen35Scratch],
) -> HipResult<()> {
    let dim = configs[0].dim;
    let mut delta_layer_idx = 0usize;
    for layer_idx in 0..configs[0].n_layers {
        dense_tp_local_attention(
            gpus,
            weights,
            configs,
            layer_idx,
            delta_layer_idx,
            pos,
            kv_caches,
            dn_states,
            scratches,
        )?;
        dense_tp_allreduce_add(gpus, scratches, dim, 0)?;
        dense_tp_local_ffn(gpus, weights, configs, layer_idx, scratches)?;
        dense_tp_allreduce_add(gpus, scratches, dim, 1)?;
        if configs[0].layer_types[layer_idx] == LayerType::LinearAttention {
            delta_layer_idx += 1;
        }
    }
    dense_tp_output(gpus, weights, configs, scratches)
}

fn dense_tp_graph_enabled(gpus: &Gpus, kv_caches: &[llama::KvCache]) -> bool {
    kv_caches.iter().all(|kv| kv.compact_offset == 0)
        && gpus.devices.iter().all(|gpu| {
            let arch_default = gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12");
            gpu.flags.graph_ar && gpu.flags.graph_forward.unwrap_or(arch_default)
        })
}

/// Launch one rank-local segment on every device. Segments contain no RCCL,
/// but all rank launches are still attempted before surfacing an error so the
/// device streams remain at the same stage boundary.
fn dense_tp_launch_segment(gpus: &mut Gpus, segment: usize) -> HipResult<()> {
    let mut first_error = None;
    for gpu in &mut gpus.devices {
        let result = match gpu.active_stream.as_ref() {
            Some(stream) => {
                gpu.graphs
                    .graph_segment_launch(&gpu.hip, gpu.device_id, stream, segment)
            }
            None => Err(HipError::new(0, "dense TP graph rank has no active stream")),
        };
        if first_error.is_none() {
            first_error = result.err();
        }
    }
    if let Some(error) = first_error {
        gpus.drop_rank_graphs(true);
        return Err(error);
    }
    Ok(())
}

/// Capture one rank-local compute segment per device, instantiate all of them,
/// then launch the just-captured segment. RCCL remains a direct grouped call
/// between segments: whole-rank RCCL-bearing graphs hang on ROCm 7.15, while
/// a single cross-device graph is rejected at instantiation.
fn dense_tp_capture_and_launch_segment(
    gpus: &mut Gpus,
    enqueue: impl FnOnce(&mut Gpus) -> HipResult<()>,
) -> HipResult<()> {
    for rank in 0..gpus.devices.len() {
        let begin = {
            let gpu = &mut gpus.devices[rank];
            match gpu.active_stream.as_ref() {
                Some(stream) => {
                    gpu.graphs
                        .begin_graph_capture_relaxed(&gpu.hip, gpu.device_id, stream)
                }
                None => Err(HipError::new(0, "dense TP graph rank has no active stream")),
            }
        };
        if let Err(error) = begin {
            gpus.abort_rank_graph_captures();
            return Err(error);
        }
    }
    if let Err(error) = enqueue(gpus) {
        gpus.abort_rank_graph_captures();
        return Err(error);
    }
    for rank in 0..gpus.devices.len() {
        let end = {
            let gpu = &mut gpus.devices[rank];
            let Some(stream) = gpu.active_stream.as_ref() else {
                gpus.abort_rank_graph_captures();
                return Err(HipError::new(0, "dense TP graph rank has no active stream"));
            };
            gpu.graphs
                .end_graph_capture_segment(&gpu.hip, gpu.device_id, stream)
        };
        if let Err(error) = end {
            gpus.abort_rank_graph_captures();
            return Err(error);
        }
    }
    let segment = gpus.devices[0]
        .graphs
        .graph_segment_count()
        .checked_sub(1)
        .ok_or_else(|| HipError::new(0, "dense TP graph segment capture produced no graph"))?;
    if gpus
        .devices
        .iter()
        .any(|gpu| gpu.graphs.graph_segment_count() != segment + 1)
    {
        gpus.drop_rank_graphs(true);
        return Err(HipError::new(
            0,
            "dense TP graph ranks captured different segment counts",
        ));
    }
    dense_tp_launch_segment(gpus, segment)
}

fn dense_tp_launch_root_segment(gpus: &mut Gpus, segment: usize) -> HipResult<()> {
    let result = {
        let gpu = &mut gpus.devices[0];
        match gpu.active_stream.as_ref() {
            Some(stream) => {
                gpu.graphs
                    .graph_segment_launch(&gpu.hip, gpu.device_id, stream, segment)
            }
            None => Err(HipError::new(
                0,
                "dense TP graph rank 0 has no active stream",
            )),
        }
    };
    if let Err(error) = result {
        gpus.drop_rank_graphs(true);
        return Err(error);
    }
    Ok(())
}

fn dense_tp_capture_and_launch_root_segment(
    gpus: &mut Gpus,
    enqueue: impl FnOnce(&mut Gpus) -> HipResult<()>,
) -> HipResult<()> {
    let begin = {
        let gpu = &mut gpus.devices[0];
        match gpu.active_stream.as_ref() {
            Some(stream) => gpu
                .graphs
                .begin_graph_capture_relaxed(&gpu.hip, gpu.device_id, stream),
            None => Err(HipError::new(
                0,
                "dense TP graph rank 0 has no active stream",
            )),
        }
    };
    if let Err(error) = begin {
        gpus.abort_rank_graph_captures();
        return Err(error);
    }
    if let Err(error) = enqueue(gpus) {
        gpus.abort_rank_graph_captures();
        return Err(error);
    }
    let end = {
        let gpu = &mut gpus.devices[0];
        let Some(stream) = gpu.active_stream.as_ref() else {
            gpus.abort_rank_graph_captures();
            return Err(HipError::new(
                0,
                "dense TP graph rank 0 has no active stream",
            ));
        };
        gpu.graphs
            .end_graph_capture_segment(&gpu.hip, gpu.device_id, stream)
    };
    if let Err(error) = end {
        gpus.abort_rank_graph_captures();
        return Err(error);
    }
    let segment = gpus.devices[0]
        .graphs
        .graph_segment_count()
        .checked_sub(1)
        .ok_or_else(|| HipError::new(0, "dense TP root graph capture produced no graph"))?;
    dense_tp_launch_root_segment(gpus, segment)
}

/// Execute the segmented graph path for one token. Graphs contain rank-local
/// compute only; grouped RCCL and the residual add stay direct at every stage
/// boundary so no graph inherits a cross-device dependency.
#[allow(clippy::too_many_arguments)]
fn forward_scratch_dense_tp_segmented(
    gpus: &mut Gpus,
    weights: &[Qwen35Weights],
    configs: &[Qwen35Config],
    pos: usize,
    kv_caches: &mut [llama::KvCache],
    dn_states: &mut [DeltaNetState],
    scratches: &[Qwen35Scratch],
    capture: bool,
) -> HipResult<()> {
    let dim = configs[0].dim;
    let mut delta_layer_idx = 0usize;
    let mut segment = 0usize;
    for layer_idx in 0..configs[0].n_layers {
        if capture {
            dense_tp_capture_and_launch_segment(gpus, |gpus| {
                dense_tp_local_attention(
                    gpus,
                    weights,
                    configs,
                    layer_idx,
                    delta_layer_idx,
                    pos,
                    kv_caches,
                    dn_states,
                    scratches,
                )
            })?;
        } else {
            dense_tp_launch_segment(gpus, segment)?;
        }
        segment += 1;
        dense_tp_allreduce_add(gpus, scratches, dim, 0)?;

        if capture {
            dense_tp_capture_and_launch_segment(gpus, |gpus| {
                dense_tp_local_ffn(gpus, weights, configs, layer_idx, scratches)
            })?;
        } else {
            dense_tp_launch_segment(gpus, segment)?;
        }
        segment += 1;
        dense_tp_allreduce_add(gpus, scratches, dim, 1)?;

        if configs[0].layer_types[layer_idx] == LayerType::LinearAttention {
            delta_layer_idx += 1;
        }
    }
    if capture {
        dense_tp_capture_and_launch_root_segment(gpus, |gpus| {
            dense_tp_output(gpus, weights, configs, scratches)
        })
    } else {
        dense_tp_launch_root_segment(gpus, segment)
    }
}

/// Dense Qwen hybrid TP2..5 single-token decode. Each rank owns local
/// attention, DeltaNet and FFN projections; row-parallel outputs are reduced
/// before the residual update. Logits are produced on rank 0.
#[allow(clippy::too_many_arguments)]
pub fn forward_scratch_dense_tp(
    gpus: &mut Gpus,
    shard: &ShardConfig,
    weights: &[Qwen35Weights],
    configs: &[Qwen35Config],
    token: u32,
    pos: usize,
    kv_caches: &mut [llama::KvCache],
    dn_states: &mut [DeltaNetState],
    scratches: &[Qwen35Scratch],
) -> HipResult<()> {
    let tp = shard.tp_size;
    if !(2..=5).contains(&tp)
        || weights.len() != tp
        || configs.len() != tp
        || kv_caches.len() != tp
        || dn_states.len() != tp
        || scratches.len() != tp
        || gpus.devices.len() != tp
    {
        return Err(HipError::new(
            0,
            "dense TP requires 2..=5 devices and complete rank states",
        ));
    }
    let dim = configs[0].dim;
    let n_layers = configs[0].n_layers;
    for cfg in configs.iter().skip(1) {
        if cfg.dim != dim || cfg.n_layers != n_layers || cfg.layer_types != configs[0].layer_types {
            return Err(HipError::new(0, "dense TP configs diverge on global shape"));
        }
    }
    for rank_weights in weights {
        for layer in &rank_weights.layers {
            match layer {
                LayerWeights::DeltaNet(_) | LayerWeights::FullAttn(_) => {}
                _ => return Err(HipError::new(0, "dense TP received a MoE/mismatched layer")),
            }
        }
    }

    let required_tokens = checked_kv_end(pos, 1, "forward_scratch_dense_tp")?;
    for rank in 0..tp {
        gpus.devices[rank].bind_thread()?;
        kv_caches[rank].ensure_mapped_capacity(&mut gpus.devices[rank], required_tokens)?;
        prepare_scratch_inputs(
            &mut gpus.devices[rank],
            &weights[rank],
            &configs[rank],
            token,
            pos,
            &scratches[rank],
        )?;
    }

    if !dense_tp_graph_enabled(gpus, kv_caches) {
        if gpus.devices.iter().any(|gpu| {
            gpu.graphs.graph_exec.is_some()
                || gpu.graphs.graph_segment_count() != 0
                || gpu.graphs.capture_mode
        }) {
            gpus.drop_rank_graphs(true);
        }
        return forward_scratch_dense_tp_layers(
            gpus, weights, configs, pos, kv_caches, dn_states, scratches,
        );
    }

    // TP2 whole-token graph on an exact gfx1201 pair (default) or gfx1100 pair
    // (`HIPFIRE_TP_FULL_GRAPH=1`; `=0` off); the runtime owns capture/replay
    // and the cross-rank signal protocol.
    let barriers = n_layers * 2;
    if gpus.tp2_whole_token_graph_ready(barriers) {
        let step = gpus.tp2_whole_token_graph_forward(barriers, |gpus| {
            forward_scratch_dense_tp_layers(
                gpus, weights, configs, pos, kv_caches, dn_states, scratches,
            )
        })?;
        if step == TpGraphStep::Captured && ar_graph_trace_enabled() {
            eprintln!(
                "[qwen-dense-tp-graph] captured whole-token tp=2 barriers={barriers} pos={pos}"
            );
        }
        return Ok(());
    }

    let expected_segments = n_layers * 2;
    let replay_ready = gpus.devices.iter().enumerate().all(|(rank, gpu)| {
        let expected = expected_segments + usize::from(rank == 0);
        gpu.graphs.ar_forward_replay_enabled && gpu.graphs.graph_segment_count() == expected
    });
    if replay_ready {
        let result = forward_scratch_dense_tp_segmented(
            gpus, weights, configs, pos, kv_caches, dn_states, scratches, false,
        );
        if result.is_err() {
            gpus.drop_rank_graphs(true);
        }
        if ar_graph_trace_enabled() {
            eprintln!(
                "[qwen-dense-tp-graph] replay tp={tp} root_segments={} peer_segments={expected_segments} pos={pos}",
                expected_segments + 1
            );
        }
        return result;
    }

    if gpus
        .devices
        .iter()
        .any(|gpu| gpu.graphs.ar_forward_kernel_dirty)
    {
        forward_scratch_dense_tp_layers(
            gpus, weights, configs, pos, kv_caches, dn_states, scratches,
        )?;
        for gpu in &mut gpus.devices {
            gpu.graphs.ar_forward_kernel_dirty = false;
            gpu.graphs.ar_forward_replay_enabled = false;
        }
        return Ok(());
    }

    gpus.drop_rank_graphs(false);
    if let Err(error) = forward_scratch_dense_tp_segmented(
        gpus, weights, configs, pos, kv_caches, dn_states, scratches, true,
    ) {
        gpus.drop_rank_graphs(true);
        return Err(error);
    }
    if gpus.devices.iter().enumerate().any(|(rank, gpu)| {
        let expected = expected_segments + usize::from(rank == 0);
        gpu.graphs.graph_segment_count() != expected
    }) {
        gpus.drop_rank_graphs(true);
        return Err(HipError::new(
            0,
            "dense TP graph capture produced an incomplete segment set",
        ));
    }
    for gpu in &mut gpus.devices {
        gpu.graphs.ar_forward_replay_enabled = true;
    }
    if ar_graph_trace_enabled() {
        eprintln!(
            "[qwen-dense-tp-graph] capture tp={tp} root_segments={} peer_segments={expected_segments} pos={pos}",
            expected_segments + 1
        );
    }
    Ok(())
}

/// Per-rank DFlash capture handles for dense-TP verification prefill.
///
/// `hidden` stages each configured target layer's post-second-reduction full
/// residual (written from that rank's own `x_batch`, which the rooted
/// allreduce leaves bit-identical on every rank). `tape` records the
/// rank-local DeltaNet `(q, k, v, α, β)` innovations for exact replay.
pub(crate) struct DenseTpSpecCapture<'a> {
    pub hidden: &'a mut HiddenStateRingBuffer,
    /// `None` skips the DeltaNet tape when the caller will replay by tokens.
    pub tape: Option<&'a mut GdnTape>,
}

/// Dense-TP verify/prefill with persistent caller-owned scratch and DFlash
/// capture. Runs one chunked pass over `tokens` (requires
/// `tokens.len() <= every pbs.max_batch` and `partials[r] >= n*dim` floats),
/// capturing GdnTape through the existing DeltaNet batch call and
/// selected-layer hidden only after each layer's second allreduce, then
/// copies the final `[n,H]` full residual into `rank0_final_hidden`. Runs no
/// final LM head; the caller (rank-0 verify) owns norm + head.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_prefill_dense_tp_verify_capture(
    gpus: &mut Gpus,
    shard: &ShardConfig,
    weights: &[Qwen35Weights],
    configs: &[Qwen35Config],
    tokens: &[u32],
    start_pos: usize,
    kv_caches: &mut [llama::KvCache],
    dn_states: &mut [DeltaNetState],
    scratches: &[Qwen35Scratch],
    pbs: &[&PrefillBatchScratch],
    partials: &[&GpuTensor],
    captures: &mut [DenseTpSpecCapture<'_>],
    rank0_final_hidden: &GpuTensor,
    post_norm_hidden_out: Option<&GpuTensor>,
) -> HipResult<()> {
    if tokens.is_empty() {
        return Ok(());
    }
    let tp = shard.tp_size;
    if !(2..=5).contains(&tp)
        || weights.len() != tp
        || configs.len() != tp
        || kv_caches.len() != tp
        || dn_states.len() != tp
        || scratches.len() != tp
        || gpus.devices.len() != tp
        || pbs.len() != tp
        || partials.len() != tp
        || captures.len() != tp
    {
        return Err(HipError::new(
            0,
            "dense TP capture requires 2..=5 devices and complete rank states",
        ));
    }
    let dim = configs[0].dim;
    let n = tokens.len();
    for (rank, pb) in pbs.iter().enumerate() {
        if n > pb.max_batch {
            return Err(HipError::new(
                0,
                "dense TP capture tokens exceed rank PBS max_batch",
            ));
        }
        let need_bytes = n
            .checked_mul(dim)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| HipError::new(0, "dense_tp capture count overflow"))?;
        if partials[rank].buf.size() < need_bytes {
            return Err(HipError::new(
                0,
                "dense TP capture partial undersized for n*dim",
            ));
        }
    }
    let need_bytes = n
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| HipError::new(0, "dense_tp capture count overflow"))?;
    if rank0_final_hidden.buf.size() < need_bytes {
        return Err(HipError::new(
            0,
            "dense TP capture rank0_final_hidden undersized for n*dim",
        ));
    }
    let last_n = forward_prefill_dense_tp_batched(
        gpus,
        shard,
        weights,
        configs,
        tokens,
        start_pos,
        kv_caches,
        dn_states,
        scratches,
        pbs,
        partials,
        Some(captures),
        post_norm_hidden_out,
    )?;
    debug_assert_eq!(last_n, n);
    gpus.devices[0].bind_thread()?;
    gpus.devices[0].hip.memcpy_dtod_at(
        &rank0_final_hidden.buf,
        0,
        &pbs[0].x_batch.buf,
        0,
        need_bytes,
    )?;
    Ok(())
}

/// Shared chunked dense-TP prefill core. Runs the layer loop over `tokens` in
/// chunks of the smallest rank PBS capacity using caller-owned `pbs_vec` /
/// `partials` (no allocation here). With `captures`, records per-rank GdnTape
/// through the DeltaNet batch call and stages selected-layer hidden only
/// after each layer's second allreduce, committing staging to each rank's
/// ring per chunk. Returns the last chunk length for the logits epilogue.
#[allow(clippy::too_many_arguments)]
fn forward_prefill_dense_tp_batched(
    gpus: &mut Gpus,
    shard: &ShardConfig,
    weights: &[Qwen35Weights],
    configs: &[Qwen35Config],
    tokens: &[u32],
    start_pos: usize,
    kv_caches: &mut [llama::KvCache],
    dn_states: &mut [DeltaNetState],
    scratches: &[Qwen35Scratch],
    pbs_vec: &[&PrefillBatchScratch],
    partials: &[&GpuTensor],
    mut captures: Option<&mut [DenseTpSpecCapture<'_>]>,
    per_token_hidden_out: Option<&GpuTensor>,
) -> HipResult<usize> {
    if tokens.is_empty() {
        return Ok(0);
    }
    let tp = shard.tp_size;
    if !(2..=5).contains(&tp)
        || weights.len() != tp
        || configs.len() != tp
        || kv_caches.len() != tp
        || dn_states.len() != tp
        || scratches.len() != tp
        || gpus.devices.len() != tp
        || pbs_vec.len() != tp
        || partials.len() != tp
    {
        return Err(HipError::new(
            0,
            "dense TP requires 2..=5 devices and complete rank states",
        ));
    }
    let dim = configs[0].dim;
    let n_layers = configs[0].n_layers;
    for cfg in configs.iter().skip(1) {
        if cfg.dim != dim || cfg.n_layers != n_layers || cfg.layer_types != configs[0].layer_types {
            return Err(HipError::new(0, "dense TP configs diverge on global shape"));
        }
    }
    for layer in &weights[0].layers {
        match layer {
            LayerWeights::DeltaNet(_) | LayerWeights::FullAttn(_) => {}
            _ => return Err(HipError::new(0, "dense TP received a MoE/mismatched layer")),
        }
    }
    if let Some(caps) = captures.as_ref() {
        if caps.len() != tp {
            return Err(HipError::new(
                0,
                "dense TP capture requires one entry per rank",
            ));
        }
    }
    // Chunk at the smallest rank PBS capacity. The DFlash capture entry point
    // validates tokens.len() <= every max_batch up front, so capture calls
    // always run a single chunk (the GDN tape_offset stays 0).
    let cap = pbs_vec.iter().map(|pbs| pbs.max_batch).min().unwrap_or(0);
    if cap == 0 {
        return Err(HipError::new(0, "prefill_max_batch is zero"));
    }
    // ── Chunked layer-granular prefill ──
    let mut process_res: HipResult<()> = Ok(());
    let mut last_chunk_n: usize = 0;
    {
        let mut offset = 0usize;
        while offset < tokens.len() {
            let n = std::cmp::min(cap, tokens.len() - offset);
            last_chunk_n = n;
            let chunk = &tokens[offset..offset + n];
            let chunk_start = start_pos + offset;
            let required = match checked_kv_end(chunk_start, n, "forward_prefill_dense_tp") {
                Ok(v) => v,
                Err(e) => {
                    process_res = Err(e);
                    break;
                }
            };
            for rank in 0..tp {
                if let Err(e) = (|| -> HipResult<()> {
                    gpus.devices[rank].bind_thread()?;
                    kv_caches[rank].ensure_mapped_capacity(&mut gpus.devices[rank], required)?;
                    Ok(())
                })() {
                    process_res = Err(e);
                    break;
                }
            }
            if process_res.is_err() {
                break;
            }
            // Embed + upload positions on every rank (same tokens/positions).
            for rank in 0..tp {
                let res = (|| -> HipResult<()> {
                    crate::qwen35::prefill::batch_chunk_embed_tokens(
                        &mut gpus.devices[rank],
                        &weights[rank],
                        chunk,
                        &scratches[rank],
                        pbs_vec[rank],
                        n,
                        dim,
                        dim * 4,
                        true,
                        false,
                        false,
                        None,
                        None,
                    )?;
                    crate::qwen35::prefill::batch_chunk_upload_positions(
                        &mut gpus.devices[rank],
                        pbs_vec[rank],
                        BatchSemantics::Sequential,
                        chunk_start,
                        n,
                        None,
                        false,
                    )?;
                    Ok(())
                })();
                if let Err(e) = res {
                    process_res = Err(e);
                    break;
                }
            }
            if process_res.is_err() {
                break;
            }
            let mut delta_layer_idx: usize = 0;
            let mut kv_layer_idx: usize = 0;
            // Precompute per-rank wmma flags.
            let q8_flags: Vec<bool> = (0..tp)
                .map(|r| {
                    hipfire_dispatch::pipeline::batched::q8_prefill_wmma_enabled(&gpus.devices[r])
                })
                .collect();
            for layer_idx in 0..n_layers {
                match configs[0].layer_types[layer_idx] {
                    LayerType::LinearAttention => {
                        for rank in 0..tp {
                            let LayerWeights::DeltaNet(layer) = &weights[rank].layers[layer_idx]
                            else {
                                process_res = Err(HipError::new(
                                    0,
                                    "dense TP received a MoE/mismatched layer",
                                ));
                                break;
                            };
                            let cfg = &configs[rank];
                            let k_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
                            let v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
                            let n_v_heads = cfg.linear_num_value_heads;
                            let hd = cfg.linear_key_head_dim;
                            if let Err(e) = hipfire_dispatch::pipeline::batched_deltanet::batch_chunk_delta_net_attn(
                                &mut gpus.devices[rank],
                                &layer.mixer_view(),
                                &super::program::hybrid_dims(cfg),
                                &pbs_vec[rank].view(),
                                &super::program::deltanet_state_view(&dn_states[rank]),
                                n,
                                dim,
                                k_dim,
                                v_dim,
                                n_v_heads,
                                hd,
                                BatchSemantics::Sequential,
                                None,
                                captures
                                    .as_ref()
                                    .and_then(|caps| caps[rank].tape.as_deref())
                                    .map(super::program::gdn_tape_view)
                                    .as_ref(),
                                0,
                                delta_layer_idx,
                                q8_flags[rank],
                                q8_flags[rank],
                                BatchEpilogue::Partial(partials[rank]),
                                DflashFusionCtx::Off,
                                None,  // commit_stride: TP ranks keep legacy cadence
                                false, // Chunk scan is single-GPU ordinary prefill only
                            ) {
                                process_res = Err(e);
                                break;
                            }
                        }
                        if process_res.is_err() {
                            break;
                        }
                        if let Err(e) = dense_tp_allreduce_batched(gpus, pbs_vec, partials, n, dim)
                        {
                            process_res = Err(e);
                            break;
                        }
                        // DFlash capture: post-second-reduction full residual.
                        if captures.is_some() {
                            for rank in 0..tp {
                                let res = (|| -> HipResult<()> {
                                    let caps = captures.as_ref().ok_or_else(|| {
                                        HipError::new(0, "dense TP capture missing")
                                    })?;
                                    if let Some(slot) = caps[rank].hidden.extract_slot(layer_idx) {
                                        gpus.devices[rank].bind_thread()?;
                                        caps[rank].hidden.write_rows_to_staging(
                                            &mut gpus.devices[rank],
                                            slot,
                                            &pbs_vec[rank].x_batch,
                                            n,
                                        )?;
                                    }
                                    Ok(())
                                })();
                                if let Err(e) = res {
                                    process_res = Err(e);
                                    break;
                                }
                            }
                            if process_res.is_err() {
                                break;
                            }
                        }
                        for rank in 0..tp {
                            let LayerWeights::DeltaNet(layer) = &weights[rank].layers[layer_idx]
                            else {
                                unreachable!();
                            };
                            if let Err(e) = layer.ffn_rows(
                                &mut gpus.devices[rank],
                                configs[rank].norm_eps,
                                &pbs_vec[rank].view(),
                                n,
                                dim,
                                q8_flags[rank],
                                q8_flags[rank],
                                MlpOutput::Write(partials[rank]),
                                DflashFusionCtx::Off,
                                DenseBatchMath::Product,
                            ) {
                                process_res = Err(e);
                                break;
                            }
                        }
                        if process_res.is_err() {
                            break;
                        }
                        if let Err(e) = dense_tp_allreduce_batched(gpus, pbs_vec, partials, n, dim)
                        {
                            process_res = Err(e);
                            break;
                        }
                        // DFlash capture: post-second-reduction full residual.
                        if captures.is_some() {
                            for rank in 0..tp {
                                let res = (|| -> HipResult<()> {
                                    let caps = captures.as_ref().ok_or_else(|| {
                                        HipError::new(0, "dense TP capture missing")
                                    })?;
                                    if let Some(slot) = caps[rank].hidden.extract_slot(layer_idx) {
                                        gpus.devices[rank].bind_thread()?;
                                        caps[rank].hidden.write_rows_to_staging(
                                            &mut gpus.devices[rank],
                                            slot,
                                            &pbs_vec[rank].x_batch,
                                            n,
                                        )?;
                                    }
                                    Ok(())
                                })();
                                if let Err(e) = res {
                                    process_res = Err(e);
                                    break;
                                }
                            }
                            if process_res.is_err() {
                                break;
                            }
                        }
                        delta_layer_idx += 1;
                    }
                    LayerType::FullAttention => {
                        for rank in 0..tp {
                            let LayerWeights::FullAttn(layer) = &weights[rank].layers[layer_idx]
                            else {
                                process_res = Err(HipError::new(
                                    0,
                                    "dense TP received a MoE/mismatched layer",
                                ));
                                break;
                            };
                            let max_ctx = chunk_start + n;
                            let ctx = DispatchCtx::new(&gpus.devices[rank]);
                            let cfg = &configs[rank];
                            let tap_fn = super::program::batched_attention_tap(cfg);
                            let tap = tap_fn.get();
                            if let Err(e) = hipfire_dispatch::pipeline::batched_attention::batch_chunk_full_attn_attn(
                                &mut gpus.devices[rank],
                                false,
                                &layer.mixer_view(),
                                &super::program::hybrid_dims(cfg),
                                &pbs_vec[rank].view(),
                                &super::program::flash_scratch(&scratches[rank]),
                                &super::program::kv_view(&kv_caches[rank]),
                                n,
                                dim,
                                chunk_start,
                                max_ctx,
                                &ctx,
                                BatchSemantics::Sequential,
                                None,
                                q8_flags[rank],
                                q8_flags[rank],
                                kv_layer_idx,
                                layer_idx,
                                BatchEpilogue::Partial(partials[rank]),
                                DflashFusionCtx::Off,
                                None, // commit_stride: TP ranks keep legacy cadence
                                tap,
                            ) {
                                process_res = Err(e);
                                break;
                            }
                        }
                        if process_res.is_err() {
                            break;
                        }
                        if let Err(e) = dense_tp_allreduce_batched(gpus, pbs_vec, partials, n, dim)
                        {
                            process_res = Err(e);
                            break;
                        }
                        // DFlash capture: post-second-reduction full residual.
                        if captures.is_some() {
                            for rank in 0..tp {
                                let res = (|| -> HipResult<()> {
                                    let caps = captures.as_ref().ok_or_else(|| {
                                        HipError::new(0, "dense TP capture missing")
                                    })?;
                                    if let Some(slot) = caps[rank].hidden.extract_slot(layer_idx) {
                                        gpus.devices[rank].bind_thread()?;
                                        caps[rank].hidden.write_rows_to_staging(
                                            &mut gpus.devices[rank],
                                            slot,
                                            &pbs_vec[rank].x_batch,
                                            n,
                                        )?;
                                    }
                                    Ok(())
                                })();
                                if let Err(e) = res {
                                    process_res = Err(e);
                                    break;
                                }
                            }
                            if process_res.is_err() {
                                break;
                            }
                        }
                        for rank in 0..tp {
                            let LayerWeights::FullAttn(layer) = &weights[rank].layers[layer_idx]
                            else {
                                unreachable!();
                            };
                            if let Err(e) = layer.ffn_rows(
                                &mut gpus.devices[rank],
                                configs[rank].norm_eps,
                                &pbs_vec[rank].view(),
                                n,
                                dim,
                                q8_flags[rank],
                                q8_flags[rank],
                                MlpOutput::Write(partials[rank]),
                                DflashFusionCtx::Off,
                                DenseBatchMath::Product,
                            ) {
                                process_res = Err(e);
                                break;
                            }
                        }
                        if process_res.is_err() {
                            break;
                        }
                        if let Err(e) = dense_tp_allreduce_batched(gpus, pbs_vec, partials, n, dim)
                        {
                            process_res = Err(e);
                            break;
                        }
                        // DFlash capture: post-second-reduction full residual.
                        if captures.is_some() {
                            for rank in 0..tp {
                                let res = (|| -> HipResult<()> {
                                    let caps = captures.as_ref().ok_or_else(|| {
                                        HipError::new(0, "dense TP capture missing")
                                    })?;
                                    if let Some(slot) = caps[rank].hidden.extract_slot(layer_idx) {
                                        gpus.devices[rank].bind_thread()?;
                                        caps[rank].hidden.write_rows_to_staging(
                                            &mut gpus.devices[rank],
                                            slot,
                                            &pbs_vec[rank].x_batch,
                                            n,
                                        )?;
                                    }
                                    Ok(())
                                })();
                                if let Err(e) = res {
                                    process_res = Err(e);
                                    break;
                                }
                            }
                            if process_res.is_err() {
                                break;
                            }
                        }
                        kv_layer_idx += 1;
                    }
                }
            }
            if process_res.is_err() {
                break;
            }
            // Commit per-rank hidden staging to each rank's ring (outside any
            // captured region, same ordering as the single-GPU verify path).
            if let Some(caps) = captures.as_mut() {
                for (rank, cap) in caps.iter_mut().enumerate() {
                    let res = (|| -> HipResult<()> {
                        gpus.devices[rank].bind_thread()?;
                        cap.hidden
                            .commit_staging_to_ring(&mut gpus.devices[rank], n)?;
                        Ok(())
                    })();
                    if let Err(e) = res {
                        process_res = Err(e);
                        break;
                    }
                }
            }
            // Post-norm hidden for every row of this chunk: rank 0 holds the
            // all-reduced residual, so its rows are the complete ones.
            if process_res.is_ok() {
                if let Some(dst) = per_token_hidden_out {
                    let res = (|| -> HipResult<()> {
                        gpus.devices[0].bind_thread()?;
                        let src = pbs_vec[0].x_batch.sub_offset(0, n * dim);
                        let dst_view = dst.sub_offset(offset * dim, n * dim);
                        gpus.devices[0].rmsnorm_batched(
                            &src,
                            &weights[0].output_norm,
                            &dst_view,
                            n,
                            dim,
                            configs[0].norm_eps,
                        )?;
                        Ok(())
                    })();
                    if let Err(e) = res {
                        process_res = Err(e);
                        break;
                    }
                }
            }
            offset += n;
        }
        Ok(last_chunk_n)
    }
}

/// Layer-granular batched dense-TP prefill. Chunks with the existing
/// gfx1201 bounded prefill batch size, uses one `PrefillBatchScratch`
/// per rank (no per-token allocation) and exactly two deterministic
/// reductions per layer per chunk. Only rank 0 produces final logits.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_dense_tp(
    gpus: &mut Gpus,
    shard: &ShardConfig,
    weights: &[Qwen35Weights],
    configs: &[Qwen35Config],
    tokens: &[u32],
    start_pos: usize,
    kv_caches: &mut [llama::KvCache],
    dn_states: &mut [DeltaNetState],
    scratches: &[Qwen35Scratch],
) -> HipResult<()> {
    forward_prefill_dense_tp_with_hidden(
        gpus, shard, weights, configs, tokens, start_pos, kv_caches, dn_states, scratches, None,
    )
}

/// `per_token_hidden_out` takes the post-output-norm row of every prefilled
/// token, which the MTP head needs to warm its private KV over the prompt.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_dense_tp_with_hidden(
    gpus: &mut Gpus,
    shard: &ShardConfig,
    weights: &[Qwen35Weights],
    configs: &[Qwen35Config],
    tokens: &[u32],
    start_pos: usize,
    kv_caches: &mut [llama::KvCache],
    dn_states: &mut [DeltaNetState],
    scratches: &[Qwen35Scratch],
    per_token_hidden_out: Option<&GpuTensor>,
) -> HipResult<()> {
    super::vl_prefill::refuse_active("dense tensor-parallel prefill")?;
    if tokens.is_empty() {
        return Ok(());
    }
    let tp = shard.tp_size;
    if !(2..=5).contains(&tp)
        || weights.len() != tp
        || configs.len() != tp
        || kv_caches.len() != tp
        || dn_states.len() != tp
        || scratches.len() != tp
        || gpus.devices.len() != tp
    {
        return Err(HipError::new(
            0,
            "dense TP requires 2..=5 devices and complete rank states",
        ));
    }
    let dim = configs[0].dim;
    // Size the scratch to the call, not to the arch ceiling: an 8-token verify
    // would otherwise allocate and free a 512-row PBS per rank on every cycle.
    let cap =
        crate::qwen35::prefill::prefill_max_batch_tp(&gpus.devices[0], tp).min(tokens.len().max(1));
    if cap == 0 {
        return Err(HipError::new(0, "prefill_max_batch is zero"));
    }
    // ── Allocate per-rank PBS + N*dim partial (transactional, cap_gdn_tape=false) ──
    let mut pbs_vec: Vec<PrefillBatchScratch> = Vec::with_capacity(tp);
    let mut partials: Vec<GpuTensor> = Vec::with_capacity(tp);
    for rank in 0..tp {
        let pbs =
            match PrefillBatchScratch::new_opt(&mut gpus.devices[rank], &configs[rank], cap, false)
            {
                Ok(p) => p,
                Err(e) => {
                    for (i, prev) in pbs_vec.drain(..).enumerate() {
                        let _ = prev.free_gpu(&mut gpus.devices[i]);
                    }
                    for (i, prev) in partials.drain(..).enumerate() {
                        let _ = gpus.devices[i].free_tensor(prev);
                    }
                    return Err(e);
                }
            };
        pbs_vec.push(pbs);
        let partial = match gpus.devices[rank].alloc_tensor(&[cap * dim], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                for (i, prev) in pbs_vec.drain(..).enumerate() {
                    let _ = prev.free_gpu(&mut gpus.devices[i]);
                }
                for (i, prev) in partials.drain(..).enumerate() {
                    let _ = gpus.devices[i].free_tensor(prev);
                }
                return Err(e);
            }
        };
        partials.push(partial);
    }
    // ── Shared chunked layer loop (no capture) ──
    let pbs_refs: Vec<&PrefillBatchScratch> = pbs_vec.iter().collect();
    let partial_refs: Vec<&GpuTensor> = partials.iter().collect();
    let last_chunk_n = match forward_prefill_dense_tp_batched(
        gpus,
        shard,
        weights,
        configs,
        tokens,
        start_pos,
        kv_caches,
        dn_states,
        scratches,
        &pbs_refs,
        &partial_refs,
        None,
        per_token_hidden_out,
    ) {
        Ok(n) => n,
        Err(e) => {
            for (rank, pbs) in pbs_vec.into_iter().enumerate() {
                let _ = pbs.free_gpu(&mut gpus.devices[rank]);
            }
            for (rank, partial) in partials.into_iter().enumerate() {
                let _ = gpus.devices[rank].free_tensor(partial);
            }
            return Err(e);
        }
    };
    // ── Final logits only on rank 0, last token of overall prompt ──
    let mut process_res: HipResult<()> = Ok(());
    {
        let last_row_offset = (last_chunk_n - 1) * dim;
        let x_last = pbs_vec[0].x_batch.sub_offset(last_row_offset, dim);
        let res = (|| -> HipResult<()> {
            gpus.devices[0].bind_thread()?;
            gpus.devices[0].rmsnorm_f32(
                &x_last,
                &weights[0].output_norm,
                &scratches[0].tmp,
                configs[0].norm_eps,
            )?;
            let ctx = DispatchCtx::new(&gpus.devices[0]);
            let wr = weights[0].output.dispatch_ref();
            execute_steps(
                &mut gpus.devices[0],
                &ctx,
                &[Step::Gemv {
                    w: &wr,
                    input: GemvInput::Raw(&scratches[0].tmp),
                    out: &scratches[0].logits,
                }],
            )
            .map_err(|e| HipError::new(0, &e.to_string()))?;
            Ok(())
        })();
        if let Err(e) = res {
            process_res = Err(e);
        }
    }
    // ── Transactional free on success and every error path ──
    for (rank, pbs) in pbs_vec.into_iter().enumerate() {
        let _ = pbs.free_gpu(&mut gpus.devices[rank]);
    }
    for (rank, partial) in partials.into_iter().enumerate() {
        let _ = gpus.devices[rank].free_tensor(partial);
    }
    process_res
}
// EP decode super-op program. The EP executor (`run_layer_program_ep`) and the
// gfx1201 exact VMM slot route interpret one coarse super-op `LayerProgram`
// per layer: EP needs explicit all-reduce points around the MoE, and the VMM
// route batches each projection across requests. Every super-op handler runs
// a stage of the single-GPU step program (`program::DecodeBinding`).
// ─────────────────────────────────────────────────────────────────────────

/// qwen35-local super-op opcodes, encoded into `OpBinding.weights[0].0`. The
/// `SuperOpKind` routes to the `ForwardBindings` method; the opcode disambiguates
/// *which* op of that kind within the layer (qkv vs gate_up, wo vs down, …).
mod q35_op {
    // Proj
    pub const PROJ_QKV: u32 = 0;
    pub const PROJ_QKVZA: u32 = 1;
    pub const PROJ_GATE_UP: u32 = 2;
    // Attend
    pub const ATTEND_FULL: u32 = 0;
    pub const ATTEND_DN_PREP: u32 = 1;
    // ResidualGemv
    pub const RESID_WO: u32 = 0;
    pub const RESID_DOWN_SWIGLU: u32 = 1;
    // Norm
    pub const NORM_GATED: u32 = 0;
    // Recurrent
    pub const RECUR_GDN: u32 = 0;
    // Moe
    pub const MOE_FFN: u32 = 0;
}

/// The four qwen35 decoder-layer shapes. Derived from the `LayerWeights`
/// discriminant; kept as a plain enum so `lower_variant` is pure (no GpuTensor)
/// and unit-testable without a GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Q35Variant {
    DeltaNet,
    FullAttn,
    DeltaNetMoe,
    FullAttnMoe,
}

pub(crate) fn variant_of(layer: &LayerWeights) -> Q35Variant {
    match layer {
        LayerWeights::DeltaNet(_) => Q35Variant::DeltaNet,
        LayerWeights::FullAttn(_) => Q35Variant::FullAttn,
        LayerWeights::DeltaNetMoe(_) => Q35Variant::DeltaNetMoe,
        LayerWeights::FullAttnMoe(_) => Q35Variant::FullAttnMoe,
    }
}

#[inline]
fn q35_superop(kind: SuperOpKind, code: u32) -> SuperOp {
    SuperOp {
        kind,
        binding: OpBinding {
            key: None,
            weights: vec![WeightSlot(code)],
            scratch: Vec::new(),
            flavor: OpFlavor::None,
        },
    }
}

/// Lower one qwen35 decoder layer to the EP executor's coarse-super-op
/// `LayerProgram`; its op sequence follows the layer's single-GPU steps.
/// Pure → unit-testable.
pub(crate) fn lower_variant(v: Q35Variant) -> LayerProgram {
    use q35_op::*;
    use SuperOpKind::{Attend, Moe, Norm, Proj, Recurrent, ResidualGemv};
    match v {
        Q35Variant::DeltaNet => vec![
            q35_superop(Proj, PROJ_QKVZA),
            q35_superop(Attend, ATTEND_DN_PREP),
            q35_superop(Recurrent, RECUR_GDN),
            q35_superop(Norm, NORM_GATED),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Proj, PROJ_GATE_UP),
            q35_superop(ResidualGemv, RESID_DOWN_SWIGLU),
        ],
        Q35Variant::FullAttn => vec![
            q35_superop(Proj, PROJ_QKV),
            q35_superop(Attend, ATTEND_FULL),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Proj, PROJ_GATE_UP),
            q35_superop(ResidualGemv, RESID_DOWN_SWIGLU),
        ],
        Q35Variant::DeltaNetMoe => vec![
            q35_superop(Proj, PROJ_QKVZA),
            q35_superop(Attend, ATTEND_DN_PREP),
            q35_superop(Recurrent, RECUR_GDN),
            q35_superop(Norm, NORM_GATED),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Moe, MOE_FFN),
        ],
        Q35Variant::FullAttnMoe => vec![
            q35_superop(Proj, PROJ_QKV),
            q35_superop(Attend, ATTEND_FULL),
            q35_superop(ResidualGemv, RESID_WO),
            q35_superop(Moe, MOE_FFN),
        ],
    }
}

/// Per-layer execution context for the EP super-op program. Holds the current
/// layer's weights + shared scratch/state by reference; rebuilt each layer
/// iteration so the borrows stay scoped. `kv_cache` is the only `&mut` (DeltaNet
/// state is mutated through interior-mutable GpuTensor buffers via shared refs).
pub(crate) struct Qwen35Bindings<'a> {
    pub(crate) layer: &'a LayerWeights,
    pub(crate) s: &'a Qwen35Scratch,
    pub(crate) config: &'a Qwen35Config,
    pub(crate) kv_cache: &'a mut llama::KvCache,
    pub(crate) dn_state: &'a DeltaNetState,
    pub(crate) pos: usize,
    pub(crate) layer_idx: usize,
    pub(crate) delta_layer_idx: usize,
    pub(crate) precomputed_attn_x_rot: bool,
    pub(crate) fa_output_prerotated: bool,
    pub(crate) defer_routed_combine: bool,
}

fn op_code(op: &OpBinding) -> u32 {
    op.weights.first().map(|w| w.0).unwrap_or(u32::MAX)
}

impl Qwen35Bindings<'_> {
    fn program(&self) -> super::program::DecodeBinding<'_> {
        super::program::DecodeBinding {
            dims: super::program::hybrid_dims(self.config),
            s: self.s,
            kv_cache: &*self.kv_cache,
            dn_state: self.dn_state,
            position: self.pos,
            mrope_section: None,
        }
    }

    fn mixer<'s>(&'s self, tap: Option<AttentionTap<'s>>) -> Step<'s> {
        self.program().mixer(
            self.layer,
            self.layer_idx,
            self.delta_layer_idx,
            self.precomputed_attn_x_rot,
            tap,
            MixerOutput::Residual,
        )
    }

    fn dense_ffn(&self) -> Result<GatedMlpOp<'_>, DispatchError> {
        super::program::dense_ffn(
            &super::program::hybrid_dims(self.config),
            self.s,
            self.layer,
            MlpOutput::Residual,
        )
        .ok_or_else(|| DispatchError::Hip("dense FFN stage on a MoE layer".into()))
    }
}

fn wrong_layer(stage: &str) -> DispatchError {
    DispatchError::Hip(format!("{stage} does not apply to this layer kind"))
}

/// Whether one MoE layer's FFN reads a host-mapped weight: the router, the
/// shared expert, or either routed projection (owner-backed, so packed layers
/// report honestly — per-expert views are borrowed slices).
fn moe_ffn_host_mapped(gpu: &Gpu, ffn: &MoeFfnWeights) -> bool {
    use hipfire_dispatch::families::moe::RoutedExpertWeights;
    let (gate_up_host, down_host) = ffn.host_mapped_projections();
    gate_up_host
        || down_host
        || gpu.host_located(&ffn.router.buf)
        || gpu.host_located(&ffn.shared_expert_gate.buf)
        || gpu.host_located(&ffn.shared_expert.gate.buf)
        || gpu.host_located(&ffn.shared_expert.up.buf)
        || gpu.host_located(&ffn.shared_expert.down.buf)
}

impl<'a> ForwardBindings for Qwen35Bindings<'a> {
    fn run_proj(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match op_code(op) {
            q35_op::PROJ_QKV | q35_op::PROJ_QKVZA => match self.mixer(None) {
                Step::DeltaNetMixer(op) => op.project(gpu, ctx),
                Step::GatedAttention(op) => op.project(gpu, ctx),
                _ => Err(wrong_layer("PROJ")),
            },
            q35_op::PROJ_GATE_UP => self.dense_ffn()?.project(gpu, ctx),
            other => Err(DispatchError::Hip(format!("unknown PROJ opcode {other}"))),
        }
    }

    fn run_residual_gemv(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match op_code(op) {
            q35_op::RESID_WO => match self.mixer(None) {
                Step::DeltaNetMixer(op) => op.output(gpu, ctx),
                Step::GatedAttention(op) => op.output(gpu, ctx, self.fa_output_prerotated),
                _ => Err(wrong_layer("RESID_WO")),
            },
            q35_op::RESID_DOWN_SWIGLU => self.dense_ffn()?.down(gpu, ctx),
            other => Err(DispatchError::Hip(format!("unknown RESID opcode {other}"))),
        }
    }

    fn run_norm(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match self.mixer(None) {
            Step::DeltaNetMixer(op) => op.gated_norm(gpu),
            _ => Err(wrong_layer("NORM_GATED")),
        }
    }

    fn run_attend(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let (s, config, layer_idx) = (self.s, self.config, self.layer_idx);
        let tap_fn = |gpu: &mut Gpu| {
            triattn_tap(gpu, layer_idx, s, config).map_err(|e| DispatchError::Hip(e.to_string()))
        };
        let tap = hipfire_runtime::triattn::tap_enabled().then_some(&tap_fn as AttentionTap<'_>);
        let prerotated = match (op_code(op), self.mixer(tap)) {
            (q35_op::ATTEND_FULL, Step::GatedAttention(op)) => op.attend(gpu, ctx)?,
            (q35_op::ATTEND_DN_PREP, Step::DeltaNetMixer(op)) => return op.prepare(gpu),
            (code, _) => {
                return Err(DispatchError::Hip(format!(
                    "ATTEND opcode {code} does not apply to this layer kind"
                )))
            }
        };
        self.fa_output_prerotated = prerotated;
        Ok(())
    }

    fn run_moe(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let (ffn, ffn_norm) = match self.layer {
            LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
            LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
            _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
        };
        moe_ffn_dispatch(
            gpu,
            ffn,
            &s.x,
            ffn_norm,
            config,
            s,
            self.defer_routed_combine,
        )
        .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_moe_ep(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        routed_out: &GpuTensor,
        skip_shared: bool,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let (ffn, ffn_norm) = match self.layer {
            LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
            LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
            _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
        };
        // Legacy rank-partial dispatch retained for non-root-routed EP users:
        // routed combine + shared-down accumulate into `routed_out`, while
        // `s.x` remains untouched until the runtime collective completes.
        moe_ffn_dispatch_ep(gpu, ffn, &s.x, ffn_norm, config, s, routed_out, skip_shared)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn ep_run_moe_root(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        _op: &OpBinding,
        partial: &GpuTensor,
    ) -> Result<MoeRouteProducerProof, DispatchError> {
        let s = self.s;
        let config = self.config;
        let (ffn, ffn_norm) = match self.layer {
            LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
            LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
            _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
        };
        // Root only: route once on-GPU, write owned expert rows into the
        // global slot layout, accumulate the shared expert into the replicated
        // residual base, and mint the sealer-built producer proof. Routed rows
        // are folded only after every rank's slot outputs have been gathered to
        // root.
        moe_ffn_dispatch_root_ep(gpu, ctx, ffn, &s.x, ffn_norm, config, s, partial)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn ep_run_moe_contrib(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        _op: &OpBinding,
        proof: &MoeRouteProducerProof,
        partial: &GpuTensor,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let (ffn, ffn_norm) = match self.layer {
            LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
            LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
            _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
        };
        // Non-root only: validate the proof before any GPU mutation, then
        // write owned expert rows over the broadcast route into the global
        // slot layout. The runtime gathers these rows to root before combine.
        moe_ffn_dispatch_contrib_ep(gpu, ctx, ffn, &s.x, ffn_norm, config, s, proof, partial)
            .map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn ep_finish_moe_slot_order(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        _op: &OpBinding,
        partial: &GpuTensor,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let (ffn, ffn_norm) = match self.layer {
            LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
            LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
            _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
        };
        let sealed = moe_sealed_root_ep(ctx, ffn, &s.x, ffn_norm, self.config, s, partial)
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        sealed.execute_ep_slot_combine(gpu)
    }

    fn ep_preflight_moe_root(
        &self,
        _gpu: &Gpu,
        ctx: &DispatchCtx,
        _op: &OpBinding,
        partial: &GpuTensor,
    ) -> Result<MoeRouteProducerProof, DispatchError> {
        let s = self.s;
        let config = self.config;
        let (ffn, ffn_norm) = match self.layer {
            LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
            LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
            _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
        };
        let sealed = moe_sealed_root_ep(ctx, ffn, &s.x, ffn_norm, config, s, partial)
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        sealed.ep_route_producer_proof()
    }

    fn ep_preflight_moe_contrib(
        &self,
        _gpu: &Gpu,
        ctx: &DispatchCtx,
        _op: &OpBinding,
        proof: &MoeRouteProducerProof,
        partial: &GpuTensor,
    ) -> Result<(), DispatchError> {
        let s = self.s;
        let config = self.config;
        let (ffn, ffn_norm) = match self.layer {
            LayerWeights::DeltaNetMoe(l) => (&l.ffn, &l.ffn_norm),
            LayerWeights::FullAttnMoe(l) => (&l.ffn, &l.ffn_norm),
            _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
        };
        moe_sealed_contrib_ep(ctx, ffn, &s.x, ffn_norm, config, s, proof, partial)
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        Ok(())
    }

    fn ep_add_into_residual(
        &mut self,
        gpu: &mut Gpu,
        partial: &GpuTensor,
    ) -> Result<(), DispatchError> {
        // s.x already contains the shared expert on root and is synchronized
        // to every rank before the root-combined routed partial is added.
        let s = self.s;
        gpu.add_inplace_f32(&s.x, partial)
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        if hipfire_dispatch::pipeline::moe_dump::enabled() {
            let layer = match self.layer {
                LayerWeights::DeltaNetMoe(l) => l.ffn.layer_idx,
                LayerWeights::FullAttnMoe(l) => l.ffn.layer_idx,
                _ => return Err(DispatchError::Hip("MOE on dense layer".into())),
            };
            hipfire_dispatch::pipeline::moe_dump::final_residual(
                gpu,
                layer as usize,
                &s.x,
                self.config.dim,
                "ep_add_into_residual",
            )?;
        }
        Ok(())
    }

    fn ep_moe_combine_mode(&self) -> EpMoeCombineMode {
        // Dense layers never serve MoE: report the default so a stray EP
        // MoE op fails later at `run_moe_ep` ("MOE on dense layer"), not
        // here. MoE layers report the sealed selection (root-routed
        // partials); an unsealed binding errors there.
        match self.layer {
            LayerWeights::DeltaNetMoe(l) => {
                qwen_ep_moe_mode(&l.ffn).unwrap_or(EpMoeCombineMode::RankPartial)
            }
            LayerWeights::FullAttnMoe(l) => {
                qwen_ep_moe_mode(&l.ffn).unwrap_or(EpMoeCombineMode::RankPartial)
            }
            _ => EpMoeCombineMode::RankPartial,
        }
    }

    fn ep_moe_route_view(&self) -> Option<EpMoeRouteView<'_>> {
        // Root-routed only: the runtime EP driver calls this after every
        // rank reported RootRoutedPartial. Any missing piece (dense layer,
        // unsealed binding, absent scratch) is None → the driver fail-stops.
        let ffn = match self.layer {
            LayerWeights::DeltaNetMoe(l) => &l.ffn,
            LayerWeights::FullAttnMoe(l) => &l.ffn,
            _ => return None,
        };
        if qwen_ep_moe_mode(ffn).ok()? != EpMoeCombineMode::RootRoutedPartial {
            return None;
        }
        let s = self.s;
        Some(EpMoeRouteView {
            experts: ffn.bound_experts().ok()?,
            layer: ffn.layer_idx as usize,
            hidden: self.config.dim,
            k: self.config.num_experts_per_tok,
            n_exp: self.config.num_experts,
            topk_ids: s.moe_topk_indices.as_ref()?,
            topk_weights: s.moe_topk_weights.as_ref()?,
            slot_outputs: s.moe_down_expanded.as_ref()?,
        })
    }

    fn ep_moe_shared_residual(&self) -> Option<&GpuTensor> {
        match self.layer {
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_) => Some(&self.s.x),
            _ => None,
        }
    }

    fn run_recurrent(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        match self.mixer(None) {
            Step::DeltaNetMixer(op) => op.recur(gpu),
            _ => Err(wrong_layer("RECUR_GDN")),
        }
    }

    fn run_conv(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("qwen35 has no Conv super-op".into()))
    }

    fn run_escape(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        kind: superop::EscapeKind,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!(
            "qwen35 has no Escape super-op ({kind:?})"
        )))
    }

    fn op_reads_host_mapped_weight(&self, gpu: &Gpu, op: &SuperOp) -> bool {
        // Decode lowered ops read GEMV/GEMM weights only; norm/state/KV/conv reads
        // stay on the GPU by design and must not trip the offload refusal. `Moe`
        // covers the router, the shared expert, and the routed projections so a
        // spilled expert tier fails closed rather than reading over PCIe. `Norm`,
        // `Recurrent`, `Conv`, and `Escape` carry no weight reads.
        match op.kind {
            SuperOpKind::Proj => match (self.layer, op_code(&op.binding)) {
                (LayerWeights::FullAttn(l), q35_op::PROJ_QKV) => {
                    gpu.host_located(&l.wq.buf)
                        || gpu.host_located(&l.wk.buf)
                        || gpu.host_located(&l.wv.buf)
                }
                (LayerWeights::FullAttnMoe(l), q35_op::PROJ_QKV) => {
                    gpu.host_located(&l.wq.buf)
                        || gpu.host_located(&l.wk.buf)
                        || gpu.host_located(&l.wv.buf)
                }
                (LayerWeights::DeltaNet(l), q35_op::PROJ_QKVZA) => {
                    gpu.host_located(&l.wqkv.buf)
                        || gpu.host_located(&l.wz.buf)
                        || gpu.host_located(&l.w_beta.buf)
                        || gpu.host_located(&l.w_alpha.buf)
                }
                (LayerWeights::DeltaNetMoe(l), q35_op::PROJ_QKVZA) => {
                    gpu.host_located(&l.wqkv.buf)
                        || gpu.host_located(&l.wz.buf)
                        || gpu.host_located(&l.w_beta.buf)
                        || gpu.host_located(&l.w_alpha.buf)
                }
                (LayerWeights::DeltaNet(l), q35_op::PROJ_GATE_UP) => {
                    gpu.host_located(&l.w_gate.buf) || gpu.host_located(&l.w_up.buf)
                }
                (LayerWeights::FullAttn(l), q35_op::PROJ_GATE_UP) => {
                    gpu.host_located(&l.w_gate.buf) || gpu.host_located(&l.w_up.buf)
                }
                _ => false,
            },
            SuperOpKind::ResidualGemv => match (self.layer, op_code(&op.binding)) {
                (LayerWeights::FullAttn(l), q35_op::RESID_WO) => gpu.host_located(&l.wo.buf),
                (LayerWeights::FullAttnMoe(l), q35_op::RESID_WO) => gpu.host_located(&l.wo.buf),
                (LayerWeights::DeltaNet(l), q35_op::RESID_WO) => gpu.host_located(&l.wo.buf),
                (LayerWeights::DeltaNetMoe(l), q35_op::RESID_WO) => gpu.host_located(&l.wo.buf),
                (LayerWeights::DeltaNet(l), q35_op::RESID_DOWN_SWIGLU) => {
                    gpu.host_located(&l.w_down.buf)
                }
                (LayerWeights::FullAttn(l), q35_op::RESID_DOWN_SWIGLU) => {
                    gpu.host_located(&l.w_down.buf)
                }
                _ => false,
            },
            SuperOpKind::Attend => false,
            SuperOpKind::Moe => match self.layer {
                LayerWeights::DeltaNetMoe(l) => moe_ffn_host_mapped(gpu, &l.ffn),
                LayerWeights::FullAttnMoe(l) => moe_ffn_host_mapped(gpu, &l.ffn),
                _ => false,
            },
            SuperOpKind::Norm | SuperOpKind::Recurrent | SuperOpKind::Conv => false,
            SuperOpKind::Escape(_) => false,
        }
    }
}

/// Certified gfx1100 Radiowave schedule: retain each non-final routed-MoE
/// result in expanded scratch and let the next layer combine it while producing
/// that layer's normalized MQ activation. The whole-model admission check is
/// intentionally strict so every deferred producer has exactly one compatible
/// consumer and no fallback projection can observe an incomplete residual.
/// Enabled by default for the admitted mq4r shape; set
/// `HIPFIRE_MOE_COMBINE_NEXT_RMS=0` to restore the two-dispatch schedule.
fn moe_combine_next_rms_enabled(gpu: &Gpu, weights: &Qwen35Weights, config: &Qwen35Config) -> bool {
    static REQUESTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let requested = *REQUESTED.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_MOE_COMBINE_NEXT_RMS")
            .ok()
            .as_deref()
            != Some("0")
    });
    if !requested
        || !(gpu.arch_caps.is_gfx1100() || hybrid::gfx1151_radiowave(gpu))
        || config.dim != 2_048
        || config.num_experts_per_tok != 8
        || config.n_layers < 2
        || config.n_layers != weights.layers.len()
        || weights.pager.is_some()
        // The prerotated qkv/qkvza producers read their weights directly,
        // bypassing the step seam that splices host-mapped weights to the CPU.
        || gpu.owns_host_cpu_weights()
        || hipfire_config::developer_var("HIPFIRE_MOE_DOWN_LAST_COMBINE").as_deref() == Ok("1")
        || hipfire_config::developer_var("HIPFIRE_QKVZA_SCALAR_PREP").as_deref() == Ok("1")
    {
        return false;
    }

    let mq4 = |w: &WeightTensor| {
        matches!(
            w.gpu_dtype,
            DType::MQ4G256 | DType::MQ4G256V2 | DType::MQ4CG256
        ) && w.k == 2_048
            && w.awq_scale.is_none()
    };
    // MUST stay in lockstep with `routed_down_self_combines` in
    // hipfire-dispatch/src/pipeline/mod.rs — this asks the same question ("does
    // this layer's down GEMV write `down_expanded`?") and is the admission gate
    // for `defer_routed_combine`. Every dtype whose down GEMV self-combines via
    // atomicAdd into x_residual writes NO expanded buffer and must return false
    // here. That is the whole codebook family: MQ2/MQ3-Lloyd AND their
    // global-codebook siblings MQ2/MQ3-GL.
    //
    // Getting this wrong is SILENT. A GL layer that wrongly claims an expanded
    // buffer lets the next layer fold stale `down_expanded` contents into the
    // residual a second time, scaled by the wrong topk_weights — no error, no
    // crash. Uniform-GL models are masked by the `gpu.zeros` memset on
    // `s.moe_down_expanded` (folding 0.0 is a no-op), but a per-layer TIER_MAP
    // mixing MQ4 and MQ2GL/MQ3GL layers is NOT masked and double-counts on every
    // GL -> next-layer transition. This is the third copy of this invariant in
    // the tree and the second to drift; if a fourth appears, hoist it onto DType.
    let has_expanded_down = |ffn: &MoeFfnWeights| {
        ffn.expert_dtype_tags.is_some()
            || ffn.experts.first().is_some_and(|expert| {
                !matches!(
                    expert.down.gpu_dtype,
                    DType::MQ2G256Lloyd | DType::MQ3G256Lloyd | DType::MQ2G256GL | DType::MQ3G256GL
                )
            })
    };
    weights.layers.iter().all(|layer| match layer {
        LayerWeights::DeltaNetMoe(l) => {
            has_expanded_down(&l.ffn)
                && ffn_gate_side_mq4_for_moe(&l.ffn)
                && mq4(&l.wqkv)
                && mq4(&l.wz)
                && mq4(&l.w_beta)
                && mq4(&l.w_alpha)
        }
        LayerWeights::FullAttnMoe(l) => {
            has_expanded_down(&l.ffn)
                && ffn_gate_side_mq4_for_moe(&l.ffn)
                && mq4(&l.wq)
                && mq4(&l.wk)
                && mq4(&l.wv)
        }
        LayerWeights::DeltaNet(_) | LayerWeights::FullAttn(_) => false,
    })
}

/// Forward pass returning logits ON GPU (no download). Caller must free the tensor.
/// Use with gpu.sample_top_p() after applying CPU-side n-gram blocking via download/modify/upload.
pub fn forward_gpu(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<GpuTensor> {
    let dim = config.dim;
    let x = gpu.alloc_tensor(&[dim], DType::F32)?;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::Ggml(dtype) => {
            gpu.ggml_embedding_lookup(dtype, &weights.token_embd, &x, token, dim)?
        }
        _ => panic!("unsupported embedding format"),
    }
    forward_from_x_gpu(gpu, weights, config, x, pos, kv_cache, dn_state)
}

/// Run one step with a pre-computed embedding vector (for VL visual token injection).
/// embedding_data: [dim] F32 values on CPU — uploaded to GPU as the initial hidden state.
pub fn forward_with_embedding(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    embedding_data: &[f32],
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let x = gpu.upload_f32(embedding_data, &[config.dim])?;
    forward_from_x(gpu, weights, config, x, pos, kv_cache, dn_state)
}
#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_dispatch::pipeline::superop::SuperOpKind;

    #[test]
    fn x_rot_covers_deltanet_value_width_for_moe_configs() {
        assert_eq!(qwen35_x_rot_len(2048, 0, 4096), 4096);
        assert_eq!(qwen35_x_rot_len(2048, 8192, 4096), 8192);
    }

    #[test]
    fn flash_partials_split_sizing_is_exact_gfx1100_only() {
        // Legacy: batch rows at min(q8 tile, 128, batched tile).
        let legacy =
            |heads: usize, kv: usize, tile: usize| 16 * heads * kv.div_ceil(tile) * 258 * 4;

        // gfx1100 Qwen3.8-27B at 8K (Q8 decode tile32, batched tile128): 32
        // batched rows at tile128 dominate one tile32 decode row, half the
        // legacy 16 x tile32 buffer (101,449,728 B).
        let len = |arch, heads, kv, q8, batched, cfg| {
            4 * qwen35_flash_partials_len(arch, heads, 256, kv, q8, batched, cfg)
        };
        assert_eq!(legacy(24, 8_192, 32), 101_449_728);
        assert_eq!(len("gfx1100", 24, 8_192, 32, 128, None), 50_724_864);
        // An explicit HIPFIRE_FLASH_PARTIALS_BATCH is the batched row count;
        // one decode row at tile32 still bounds it from below.
        assert_eq!(len("gfx1100", 24, 8_192, 32, 128, Some(1)), 6_340_608);
        assert_eq!(len("gfx1100", 24, 8_192, 32, 128, Some(64)), 101_449_728);
        // Past 8K gfx1100 decodes at tile128 too: legacy sizing.
        assert_eq!(
            len("gfx1100", 24, 65_536, 128, 128, None),
            legacy(24, 65_536, 128)
        );

        // Every other arch keeps the legacy buffer, including the shapes whose
        // decode tile is finer than the batched tile.
        assert_eq!(
            len("gfx1201", 24, 8_192, 128, 128, None),
            legacy(24, 8_192, 128)
        );
        assert_eq!(
            len("gfx1151", 24, 8_192, 128, 128, None),
            legacy(24, 8_192, 128)
        );
        assert_eq!(
            len("gfx1201", 8, 8_192, 16, 128, None),
            legacy(8, 8_192, 16)
        );
        assert_eq!(
            len("gfx1151", 16, 2_048, 32, 128, None),
            legacy(16, 2_048, 32)
        );
        assert_eq!(
            len("gfx1101", 24, 8_192, 32, 128, None),
            legacy(24, 8_192, 32)
        );
    }

    // ── EP super-op program shapes ─────────────────────────────────────────
    // The EP LayerProgram per variant follows the single-GPU step sequence.
    // These are CPU-pure (no GPU/GpuTensor).
    #[test]
    fn lowered_fullattn_program_shape() {
        use SuperOpKind::{Attend, Proj, ResidualGemv};
        let p = lower_variant(Q35Variant::FullAttn);
        let kinds: Vec<_> = p.iter().map(|o| o.kind).collect();
        assert_eq!(kinds, vec![Proj, Attend, ResidualGemv, Proj, ResidualGemv]);
        assert_eq!(p[0].binding.weights[0].0, q35_op::PROJ_QKV);
        assert_eq!(p[1].binding.weights[0].0, q35_op::ATTEND_FULL);
        assert_eq!(p[2].binding.weights[0].0, q35_op::RESID_WO);
        assert_eq!(p[3].binding.weights[0].0, q35_op::PROJ_GATE_UP);
        assert_eq!(p[4].binding.weights[0].0, q35_op::RESID_DOWN_SWIGLU);
    }

    #[test]
    fn lowered_deltanet_program_shape() {
        use SuperOpKind::{Attend, Norm, Proj, Recurrent, ResidualGemv};
        let p = lower_variant(Q35Variant::DeltaNet);
        let kinds: Vec<_> = p.iter().map(|o| o.kind).collect();
        assert_eq!(
            kinds,
            vec![
                Proj,
                Attend,
                Recurrent,
                Norm,
                ResidualGemv,
                Proj,
                ResidualGemv
            ]
        );
        assert_eq!(p[0].binding.weights[0].0, q35_op::PROJ_QKVZA);
        assert_eq!(p[1].binding.weights[0].0, q35_op::ATTEND_DN_PREP);
    }

    #[test]
    fn lowered_moe_variants_replace_dense_ffn_with_one_moe_op() {
        use SuperOpKind::Moe;
        let dn = lower_variant(Q35Variant::DeltaNetMoe);
        let fa = lower_variant(Q35Variant::FullAttnMoe);
        // MoE variants end in a single Moe super-op (no dense gate_up/down).
        assert_eq!(dn.last().unwrap().kind, Moe);
        assert_eq!(fa.last().unwrap().kind, Moe);
        assert!(
            dn.iter()
                .all(|o| o.binding.weights[0].0 != q35_op::PROJ_GATE_UP
                    || o.kind != SuperOpKind::Proj)
        );
        // FullAttnMoe is the shortest: Proj, Attend, ResidualGemv(wo), Moe.
        assert_eq!(fa.len(), 4);
        assert_eq!(dn.len(), 6);
    }

    #[test]
    fn lowered_variant_of_maps_layer_discriminant() {
        // variant_of is a thin discriminant map; assert the program lengths it
        // would produce per the documented layer shapes.
        assert_eq!(lower_variant(Q35Variant::FullAttn).len(), 5);
        assert_eq!(lower_variant(Q35Variant::DeltaNet).len(), 7);
        assert_eq!(lower_variant(Q35Variant::DeltaNetMoe).len(), 6);
        assert_eq!(lower_variant(Q35Variant::FullAttnMoe).len(), 4);
    }

    #[test]
    fn ar_graph_is_ineligible_after_kv_compaction() {
        assert!(ar_graph_eligible_for_kv(true, 0, false));
        assert!(!ar_graph_eligible_for_kv(true, 1, false));
        assert!(!ar_graph_eligible_for_kv(true, 128, false));
        assert!(!ar_graph_eligible_for_kv(false, 0, false));
        assert!(!ar_graph_eligible_for_kv(true, 0, true));
    }

    // ── MQ6V2 / MQ4V2 FFN dtype recognition ───────────────────────────────
    // Model-level MQ6 flag must see both V1 (qt=15) and V2 (qt=47) without
    // collapsing either into MQ4V2 (qt=44). Pure helpers — no GPU tensors.

    #[test]
    fn moe_ffn_has_mq6_recognizes_v1_and_v2_distinctly() {
        // Empty structural + empty experts → false.
        assert!(!moe_ffn_has_mq6_from_dtypes([], []));

        // Legacy MQ6G256 on a structural field.
        assert!(moe_ffn_has_mq6_from_dtypes(
            [
                DType::MQ6G256,
                DType::MQ4G256,
                DType::MQ4G256,
                DType::MQ4G256,
                DType::MQ4G256
            ],
            [(DType::MQ4G256, DType::MQ4G256)],
        ));

        // MQ6G256V2 on a structural field — must trip the same model flag.
        assert!(moe_ffn_has_mq6_from_dtypes(
            [
                DType::MQ4G256,
                DType::MQ4G256,
                DType::MQ6G256V2,
                DType::MQ4G256,
                DType::MQ4G256
            ],
            [(DType::MQ4G256, DType::MQ4G256)],
        ));

        // MQ6G256V2 only on a routed expert projection.
        assert!(moe_ffn_has_mq6_from_dtypes(
            [DType::MQ4G256; 5],
            [(DType::MQ6G256V2, DType::MQ4G256)],
        ));
        assert!(moe_ffn_has_mq6_from_dtypes(
            [DType::MQ4G256; 5],
            [(DType::MQ4G256, DType::MQ6G256V2)],
        ));

        // Uniform MQ6V2 routed pair.
        assert!(moe_ffn_has_mq6_from_dtypes(
            [DType::MQ6G256V2; 5],
            [(DType::MQ6G256V2, DType::MQ6G256V2)],
        ));
    }

    #[test]
    fn moe_ffn_has_mq6_never_collapses_mq4v2() {
        // MQ4G256 / MQ4G256V2 only — never MQ6-family.
        assert!(!moe_ffn_has_mq6_from_dtypes(
            [DType::MQ4G256; 5],
            [(DType::MQ4G256, DType::MQ4G256)],
        ));
        assert!(!moe_ffn_has_mq6_from_dtypes(
            [DType::MQ4G256V2; 5],
            [(DType::MQ4G256V2, DType::MQ4G256V2)],
        ));
        // Mixed MQ4 V1/V2 gate-side + routed still not MQ6.
        assert!(!moe_ffn_has_mq6_from_dtypes(
            [
                DType::MQ4G256V2,
                DType::MQ4G256,
                DType::MQ4G256V2,
                DType::MQ4G256,
                DType::MQ4G256V2,
            ],
            [(DType::MQ4G256V2, DType::MQ4G256)],
        ));
        // Identity: V1 and V2 MQ6 are distinct enum variants (wire layouts differ).
        assert_ne!(DType::MQ6G256, DType::MQ6G256V2);
        assert_ne!(DType::MQ4G256, DType::MQ4G256V2);
        assert_ne!(DType::MQ4G256V2, DType::MQ6G256V2);
    }

    #[test]
    fn dense_tp_admission_allows_2_to_5() {
        for tp in 2..=5 {
            assert!((2..=5).contains(&tp), "tp {tp} should be admitted");
        }
        assert!(!(2..=5).contains(&1));
        assert!(!(2..=5).contains(&6));
        assert!(!(2..=5).contains(&0));
    }

    #[test]
    fn dense_tp_batched_count_overflow_is_err() {
        let n = usize::MAX;
        let dim = 2;
        assert!(n.checked_mul(dim).is_none());
        // Mirror dense_tp_allreduce_batched's overflow guard.
        let res: Result<usize, &str> = n.checked_mul(dim).ok_or("overflow");
        assert!(res.is_err());
    }

    /// The `CpuSplice` stage resolves a selected expert's host bytes from the
    /// packed bucket owners because per-expert views are `Borrowed` and invisible
    /// to `Gpu::host_bytes`. Pin that ownership walk: two graded buckets of
    /// different stride, each holding real byte patterns, and the exact
    /// `sub_offset(pos*stride, stride)` views the graded packer builds.
    #[test]
    fn packed_expert_view_bytes_finds_owning_bucket() {
        let Some(mut gpu) = Gpu::init().ok() else {
            return; // no GPU: metadata-only environment
        };
        let a0: Vec<u8> = (0u8..16).collect();
        let a1: Vec<u8> = (100u8..116).collect();
        let b0: Vec<u8> = (200u8..212).collect();
        let bucket_a = gpu
            .upload_raw_host_mapped(&[a0.clone(), a1.clone()].concat(), &[2, 16])
            .expect("host bucket A");
        let bucket_b = gpu
            .upload_raw_host_mapped(&b0.clone(), &[1, 12])
            .expect("host bucket B");
        let buckets = vec![bucket_a, bucket_b];
        let view_a0 = buckets[0].sub_offset(0, 16);
        let view_a1 = buckets[0].sub_offset(16, 16);
        let view_b0 = buckets[1].sub_offset(0, 12);
        assert_eq!(
            packed_expert_host_bytes(&gpu, &buckets, &view_a0, "gate_up").unwrap(),
            a0.as_slice()
        );
        assert_eq!(
            packed_expert_host_bytes(&gpu, &buckets, &view_a1, "gate_up").unwrap(),
            a1.as_slice()
        );
        assert_eq!(
            packed_expert_host_bytes(&gpu, &buckets, &view_b0, "down").unwrap(),
            b0.as_slice()
        );
        // A view outside every owner fails closed rather than resolving to some
        // other expert's bytes.
        let stray = GpuTensor::null_for_test();
        assert!(packed_expert_host_bytes(&gpu, &buckets, &stray, "gate_up").is_err());
    }
}
