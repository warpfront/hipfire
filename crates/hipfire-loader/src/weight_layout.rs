// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Group-major (GM) weight-layout producer entry for MQ4G256V2 IU4 prefill.
//!
//! The IU4 `_gm_col_gfx1151` kernels read group `(row, kb)` at
//! `136*(kb*M+row)` instead of the shipping `136*(row*gpr+kb)`. This module is
//! the load-time producer entry: [`warm_mq4v2_gm`] walks the dense MQ4V2
//! tensors a bundle actually feeds the IU4 selector, builds each duplicate
//! through `Gpu::ensure_mq4v2_gm` (the `mq4v2_relayout_gm` permute kernel),
//! and logs the total extra bytes ONCE at load.
//!
//! Producer substitution for the future on-disk prefill-layout sidecar
//! (precomputed group-major bytes shipped or cached next to the `.hfq` and
//! mmapped at load): add a [`GmProducer::Sidecar`] arm that installs the
//! mmapped bytes and records [`rdna_compute::WeightLayout::GroupMajor`]
//! via `Gpu::set_mq4v2_weight_layout` instead of launching the permute
//! kernel — kernels and selectors stay untouched. No sidecar implementation
//! in this unit.
//!
//! Warming is best-effort and never fails a load: a per-tensor failure is
//! logged and skipped, and the IU4 selector falls back to the row-major
//! tensor (retrying lazily). Warming is also skipped entirely unless the IU4
//! route itself is opted in — the selector is the only GM consumer, so
//! default-off loads stay byte-identical in VRAM.

use hipfire_runtime::llama::WeightTensor;
use rdna_compute::{DType, Gpu};

/// Producer selected for the GM bytes. Only [`GmProducer::Permute`] exists in
/// this unit; the match in [`warm_mq4v2_gm`] grows the sidecar arm without
/// touching any other site.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum GmProducer {
    #[default]
    Permute,
}

/// Whether a weight tensor is a GM duplicate candidate: MQ4G256V2 with whole
/// G256 groups and a non-empty output dim. Mirrors
/// `rdna_compute::dispatch::mq4v2_gm_eligible` plus the dtype gate (the loader
/// sees `WeightTensor`; the selector sees the raw tensor).
pub fn mq4v2_gm_candidate(w: &WeightTensor) -> bool {
    w.gpu_dtype == DType::MQ4G256V2 && w.k % 256 == 0 && w.m > 0
}

/// Build GM duplicates for every candidate in `tensors` through `producer`,
/// logging the total extra bytes once. Best-effort (see module docs).
/// Returns the total duplicate bytes built.
pub fn warm_mq4v2_gm(
    gpu: &mut Gpu,
    tensors: &[(&str, &WeightTensor)],
    producer: GmProducer,
) -> usize {
    if !gpu.prefill_weight_layout_gm_enabled() {
        return 0;
    }
    // The IU4 selector is the only GM consumer; without its opt-in there is
    // nothing to pre-warm.
    if !gpu.flags.gfx11_mmq_iu4_enabled() {
        return 0;
    }
    // Only `Permute` exists; the sidecar future adds its arm here.
    debug_assert_eq!(producer, GmProducer::Permute);
    let mut total = 0usize;
    let mut n = 0usize;
    for (name, w) in tensors {
        if !mq4v2_gm_candidate(w) {
            continue;
        }
        match gpu.ensure_mq4v2_gm(&w.buf, w.m, w.k) {
            Ok(Some(_)) => {
                total += rdna_compute::mq4v2_gm_extra_bytes(w.m, w.k);
                n += 1;
            }
            Ok(None) => {}
            Err(e) => eprintln!("  [gm] warm skipped {name}: {e}"),
        }
    }
    if n > 0 {
        eprintln!("  [gm] group-major prefill duplicates: {n} MQ4G256V2 tensors, +{total} B VRAM");
    }
    total
}

/// Collect `(name, weight)` for every dense IU4-consumed tensor in a Qwen35
/// bundle: per-layer attention + FFN projections (the same dense set
/// `MmqScreenable` walks). Excludes:
/// - `output` (the lm_head prefill uses the batched-lmhead path, never the
///   IU4 selector — warming it would duplicate ~270 MB for nothing);
/// - MoE routed experts (indexed/batched kernels, never the IU4 selector).
pub fn qwen35_mq4v2_tensors(
    weights: &hipfire_arch_qwen35::qwen35::Qwen35Weights,
) -> Vec<(String, &WeightTensor)> {
    use hipfire_arch_qwen35::qwen35::LayerWeights;
    let mut out: Vec<(String, &WeightTensor)> = Vec::new();
    for (li, layer) in weights.layers.iter().enumerate() {
        macro_rules! push {
            ($n:literal, $w:expr) => {
                out.push((format!("layers.{li}.{}", $n), $w));
            };
        }
        match layer {
            LayerWeights::DeltaNet(w) => {
                push!("wqkv", &w.wqkv);
                push!("wz", &w.wz);
                push!("w_alpha", &w.w_alpha);
                push!("w_beta", &w.w_beta);
                push!("wo", &w.wo);
                push!("w_gate", &w.w_gate);
                push!("w_up", &w.w_up);
                push!("w_down", &w.w_down);
            }
            LayerWeights::FullAttn(w) => {
                push!("wq", &w.wq);
                push!("wk", &w.wk);
                push!("wv", &w.wv);
                push!("wo", &w.wo);
                push!("w_gate", &w.w_gate);
                push!("w_up", &w.w_up);
                push!("w_down", &w.w_down);
            }
            LayerWeights::DeltaNetMoe(w) => {
                push!("wqkv", &w.wqkv);
                push!("wz", &w.wz);
                push!("w_alpha", &w.w_alpha);
                push!("w_beta", &w.w_beta);
                push!("wo", &w.wo);
                push!("ffn.router", &w.ffn.router);
            }
            LayerWeights::FullAttnMoe(w) => {
                push!("wq", &w.wq);
                push!("wk", &w.wk);
                push!("wv", &w.wv);
                push!("wo", &w.wo);
                push!("ffn.router", &w.ffn.router);
            }
        }
    }
    out
}
