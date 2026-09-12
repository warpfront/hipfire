// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 carrier bundle loader — HFQ path.
//!
//! Verbatim relocation of the model-loading work from
//! `hipfire-loader/src/carriers.rs::Gemma4Carrier::load`. The loader retains
//! `LoadedModel` assembly, `SourceMeta`/`resolve_source_meta`, chat-template
//! and tokenizer handling, `Gemma4EagleState` side-car load, and
//! `spec_build::build_speculator`. This module owns the GPU bundle construction
//! (lowered vs eager decision, weight/state/KV allocation) with error strings
//! byte-identical to the prior inline block.

use crate::config::Gemma4Config;
use crate::gemma4::{Gemma4State, Gemma4Weights};
use crate::lowered;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use rdna_compute::Gpu;

// ─── Helpers moved verbatim from carriers.rs ─────────────────────────────

/// Route selection shared by the carrier load path and source admission.
pub fn gemma4_use_lowered(
    enable_moe_block: bool,
    want_batched: bool,
    has_drafter: bool,
    is_e_series: bool,
) -> bool {
    enable_moe_block || (want_batched && !has_drafter && !is_e_series)
}

/// Pure context admission for Gemma 4. Refuses the lowered route when
/// `max_seq < 128`. Eager loads are unrestricted (no eager assert exists).
///
/// Callers compute `use_lowered` via [`gemma4_use_lowered`] / source probes so
/// the refusal string is byte-identical at admission and carrier layers.
pub fn gemma4_context_admission(max_seq: usize, use_lowered: bool) -> Result<(), String> {
    if use_lowered && max_seq < 128 {
        return Err(format!(
            "gemma4 lowered path requires max_seq >= 128 (got {max_seq})"
        ));
    }
    Ok(())
}

/// Mirror carrier route selection from an already-open HFQ source + env gates.
/// `has_drafter` is the EAGLE `params.drafter` presence (not DFlash draft).
pub fn gemma4_source_uses_lowered(hfq: &hipfire_runtime::hfq::HfqFile, has_drafter: bool) -> bool {
    let lowered_cfg = lowered::config_from_hfq(hfq);
    let want_batched = lowered::batched_prefill_enabled() || lowered::wmma_prefill_enabled();
    let Some(lcfg) = &lowered_cfg else {
        return false;
    };
    let lowered_is_moe = lcfg.enable_moe_block;
    let is_e_series = if lowered_is_moe {
        false
    } else {
        match Gemma4Config::from_hfq(hfq) {
            Ok(cfg) => cfg.hidden_size_per_layer_input != 0 || cfg.num_kv_shared_layers != 0,
            Err(_) => false,
        }
    };
    gemma4_use_lowered(
        lcfg.enable_moe_block,
        want_batched,
        has_drafter,
        is_e_series,
    )
}

fn gemma4_validate_drafter_route(is_e_series: bool, has_drafter: bool) -> Result<(), String> {
    if is_e_series && has_drafter {
        return Err(
            "gemma4: E2B/E4B EAGLE spec-decode is not yet supported; load the E-series target without params.drafter"
                .into(),
        );
    }
    Ok(())
}

// ─── Bundle types ─────────────────────────────────────────────────────────

pub struct Gemma4EagerBundle {
    pub config: Gemma4Config,
    pub weights: Gemma4Weights,
    pub state: Gemma4State,
}

pub struct Gemma4LoweredBundle {
    pub config: lowered::Gemma4Config,
    pub weights: lowered::Gemma4Weights,
    pub scratch: lowered::Gemma4Scratch,
    pub kv_sliding: KvCache,
    pub kv_full: KvCache,
}

pub enum Gemma4Bundle {
    Eager(Gemma4EagerBundle),
    Lowered(Gemma4LoweredBundle),
}

fn lowered_kv_layer_counts(layer_types: &[lowered::LayerType]) -> (usize, usize) {
    layer_types
        .iter()
        .fold((0, 0), |(sliding, full), layer_type| match layer_type {
            lowered::LayerType::Sliding => (sliding + 1, full),
            lowered::LayerType::Full => (sliding, full + 1),
        })
}

/// Preserve the primary operation error; append cleanup failure context when present.
fn append_cleanup_context(op_err: String, cleanup: Result<(), String>) -> String {
    match cleanup {
        Ok(()) => op_err,
        Err(c) => format!("{op_err}; cleanup also failed: {c}"),
    }
}

fn free_lowered_weights(weights: lowered::Gemma4Weights, gpu: &mut Gpu) {
    weights.free_gpu(gpu);
}

fn free_lowered_scratch_and_weights(
    scratch: lowered::Gemma4Scratch,
    weights: lowered::Gemma4Weights,
    gpu: &mut Gpu,
) {
    scratch.free_gpu(gpu);
    free_lowered_weights(weights, gpu);
}

fn free_lowered_sliding_scratch_weights(
    kv_sliding: KvCache,
    scratch: lowered::Gemma4Scratch,
    weights: lowered::Gemma4Weights,
    gpu: &mut Gpu,
) -> Result<(), String> {
    let mut first: Option<String> = None;
    if let Err(e) = kv_sliding.free_gpu(gpu) {
        first = Some(e.to_string());
    }
    scratch.free_gpu(gpu);
    free_lowered_weights(weights, gpu);
    match first {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

// ─── Bundle load ──────────────────────────────────────────────────────────

/// Build the Gemma 4 GPU bundle from an HFQ source.
///
/// `ModelSource::Dir` returns the same error string the carrier previously
/// emitted inline. HFQ path is verbatim: lowered/eager selection,
/// `want_batched` env gate, E-series validation, weight/state/KV allocation,
/// and the preserved `eprintln!` diagnostics for the chosen path.
pub fn load_gemma4_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Gemma4Bundle, String> {
    let hfq = match src {
        ModelSource::Hfq(h) => h,
        ModelSource::Dir(_) => {
            return Err("gemma4: safetensors Dir load not yet wired — use HFQ (quantize with --arch-id 13) or add config_from_source to hipfire-arch-gemma4".into());
        }
    };

    // ── Lowered vs eager selection (MoE or batched prefill opt-in) ──
    // Arch-13 MoE (26B-A4B `enable_moe_block`) must go through `lowered`, which
    // carries the parallel-MoE branch. We also route DENSE models through
    // `lowered` when the operator opts into batched/WMMA prefill — that path
    // lives only in `lowered::forward_prefill_batch`. E2B/E4B stay on eager
    // because lowered does not implement PLE, KV sharing, or E2B's double-wide
    // shared-layer FFN. EAGLE spec-decode (`params.drafter`) requires the eager
    // `Gemma4State`, so a drafter request always wins and keeps the eager path
    // (batched prefill opt-in is ignored when a drafter is present).
    let lowered_cfg = lowered::config_from_hfq(&hfq);
    let want_batched = lowered::batched_prefill_enabled() || lowered::wmma_prefill_enabled();
    let lowered_is_moe = lowered_cfg
        .as_ref()
        .is_some_and(|lcfg| lcfg.enable_moe_block);
    let eager_config = if lowered_is_moe {
        None
    } else {
        Some(Gemma4Config::from_hfq(&hfq)?)
    };
    let is_e_series = eager_config
        .as_ref()
        .is_some_and(|cfg| cfg.hidden_size_per_layer_input != 0 || cfg.num_kv_shared_layers != 0);
    if is_e_series {
        eager_config.as_ref().unwrap().e_series_variant()?;
    }
    gemma4_validate_drafter_route(is_e_series, ctx.gemma4_drafter_path.is_some())?;
    let use_lowered = if let Some(lcfg) = &lowered_cfg {
        gemma4_use_lowered(
            lcfg.enable_moe_block,
            want_batched,
            ctx.gemma4_drafter_path.is_some(),
            is_e_series,
        )
    } else {
        false
    };
    // Refuse before any device allocation so direct-carrier callers match
    // source-admission refusal (prior model / pool state stay untouched).
    gemma4_context_admission(ctx.max_seq, use_lowered)?;
    // The lowered/MoE path is served: `generate_gemma4_lowered` in
    // hipfire-generate handles `Gemma4Lowered` models end to end, so a
    // `use_lowered` selection proceeds directly to weight/scratch/KV upload.
    if use_lowered {
        let lcfg = lowered_cfg.unwrap();
        let (n_sliding_layers, n_full_layers) = lowered_kv_layer_counts(&lcfg.layer_types);
        let mut hfq2 = hfq;
        // Construction order: weights → scratch → constants → sliding KV → full KV.
        // On any later error free every completed earlier owner in reverse.
        let weights = lowered::load_weights(&mut hfq2, &lcfg, ctx.gpu)
            .map_err(|e| format!("gemma4 (lowered) load_weights: {e:?}"))?;
        let scratch = match lowered::Gemma4Scratch::new(ctx.gpu, &lcfg, ctx.max_seq) {
            Ok(v) => v,
            Err(e) => {
                free_lowered_weights(weights, ctx.gpu);
                return Err(format!("gemma4 (lowered) scratch: {e:?}"));
            }
        };
        if let Err(e) = lowered::init_scratch_constants(ctx.gpu, &scratch, lcfg.full_head_dim) {
            free_lowered_scratch_and_weights(scratch, weights, ctx.gpu);
            return Err(format!("gemma4 (lowered) init_scratch_constants: {e:?}"));
        }
        // Physical ring is min(window, max_seq); logical full/scratch stay max_seq.
        let sliding_cap = lcfg.sliding_window.min(ctx.max_seq);
        let kv_sliding = match KvCache::new_gpu_q8_capped(
            ctx.gpu,
            n_sliding_layers,
            lcfg.sliding_n_kv_heads,
            lcfg.sliding_head_dim,
            ctx.max_seq,
            sliding_cap,
        ) {
            Ok(v) => v,
            Err(e) => {
                free_lowered_scratch_and_weights(scratch, weights, ctx.gpu);
                return Err(format!(
                    "gemma4 (lowered) sliding KV alloc (q8 ring): {e:?}"
                ));
            }
        };
        // Full tier remains asym3 (not Q8); logical limit is max_seq.
        let kv_full = match KvCache::new_gpu_asym3_gemma4(
            ctx.gpu,
            n_full_layers,
            lcfg.full_n_kv_heads,
            lcfg.full_head_dim,
            ctx.max_seq,
        ) {
            Ok(v) => v,
            Err(e) => {
                let cleanup =
                    free_lowered_sliding_scratch_weights(kv_sliding, scratch, weights, ctx.gpu);
                return Err(append_cleanup_context(
                    format!("gemma4 (lowered) full KV alloc: {e:?}"),
                    cleanup,
                ));
            }
        };
        eprintln!(
            "  gemma4 lowered path: moe={} batched_opt_in={} (sliding q8-ring + full asym3 KV)",
            lcfg.enable_moe_block, want_batched,
        );
        return Ok(Gemma4Bundle::Lowered(Gemma4LoweredBundle {
            config: lcfg,
            weights,
            scratch,
            kv_sliding,
            kv_full,
        }));
    }
    // ── Eager dense / E-series path ──
    let config = match eager_config {
        Some(c) => c,
        None => Gemma4Config::from_hfq(&hfq)?,
    };
    if is_e_series {
        eprintln!(
            "  gemma4 E-series eager path: {:?} (PLE + shared KV)",
            config.e_series_variant()?
        );
    }
    let weights = Gemma4Weights::load(&hfq, &config, ctx.gpu)?;
    let state = Gemma4State::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
        .map_err(|e| format!("gemma4: Gemma4State::new_with_max_seq failed: {e}"))?;
    let _ = &weights;
    Ok(Gemma4Bundle::Eager(Gemma4EagerBundle {
        config,
        weights,
        state,
    }))
}

// Alias for task's naming convention if callers use `load_bundle`.
pub use load_gemma4_bundle as load_bundle;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowered_kv_counts_follow_attention_layer_types() {
        let layer_types = (0..48)
            .map(|layer_idx| {
                if (layer_idx + 1) % 6 == 0 {
                    lowered::LayerType::Full
                } else {
                    lowered::LayerType::Sliding
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(lowered_kv_layer_counts(&layer_types), (40, 8));
    }

    #[test]
    fn scratch_geometry_uses_single_max_seq_authority() {
        // No HIPFIRE_KV_SEQ env var should affect geometry; both partials
        // and KV are sized from the same ctx.max_seq. Verify the pure helpers
        // that the loader now uses.
        let max_seq_small = 32768usize;
        let max_seq_large = 131072usize;
        let n_heads = 32usize;
        let full_hd = 512usize;
        let s_small = lowered::gemma4_flash_partials_len(max_seq_small, n_heads, full_hd);
        let s_large = lowered::gemma4_flash_partials_len(max_seq_large, n_heads, full_hd);
        assert_eq!(s_small, 4_210_688);
        assert_eq!(s_large, 16_842_752);
        assert_eq!(s_large, 4 * s_small);
        let pb_small = lowered::gemma4_pb_flash_partials_len(max_seq_small, n_heads, full_hd);
        let pb_large = lowered::gemma4_pb_flash_partials_len(max_seq_large, n_heads, full_hd);
        assert_eq!(pb_small, 128 * s_small);
        assert_eq!(pb_large, 128 * s_large);
        assert_eq!(pb_large, 2_155_872_256);
    }

    #[test]
    fn scratch_geometry_has_no_env_mismatch() {
        // Simulate that an old env var could earlier cause mismatch between
        // KV (ctx.max_seq) and scratch (HIPFIRE_KV_SEQ). After the fix both
        // derive from the same max_seq, so the arithmetic must be identical
        // for any max_seq value, including 131072 without GPU alloc.
        for &max_seq in &[8192usize, 32768, 65536, 131072] {
            let n_heads = 32;
            let hd = 512;
            let tiles = max_seq.div_ceil(lowered::GEMMA4_FLASH_TILE);
            let expected = n_heads * tiles * (2 + hd);
            assert_eq!(
                lowered::gemma4_flash_partials_len(max_seq, n_heads, hd),
                expected
            );
            assert_eq!(
                lowered::gemma4_pb_flash_partials_len(max_seq, n_heads, hd),
                lowered::GEMMA4_MAX_PREFILL_BATCH * expected
            );
        }
    }

    #[test]
    fn context_admission_refuses_lowered_below_floor() {
        assert!(gemma4_context_admission(64, true).is_err());
        assert!(gemma4_context_admission(127, true).is_err());
        assert!(gemma4_context_admission(128, true).is_ok());
        assert!(gemma4_context_admission(512, true).is_ok());
    }

    #[test]
    fn context_admission_eager_exempt_at_any_seq() {
        // Eager has no min-context assert; small contexts stay admitted.
        assert!(gemma4_context_admission(1, false).is_ok());
        assert!(gemma4_context_admission(64, false).is_ok());
        assert!(gemma4_context_admission(127, false).is_ok());
        assert!(gemma4_context_admission(128, false).is_ok());
    }

    #[test]
    fn use_lowered_route_matrix() {
        // MoE always lowered.
        assert!(gemma4_use_lowered(true, false, false, false));
        assert!(gemma4_use_lowered(true, true, true, true));
        // Dense batched opt-in, no drafter, not E-series.
        assert!(gemma4_use_lowered(false, true, false, false));
        // Drafter or E-series keep eager even with batched opt-in.
        assert!(!gemma4_use_lowered(false, true, true, false));
        assert!(!gemma4_use_lowered(false, true, false, true));
        // No batched / no MoE → eager.
        assert!(!gemma4_use_lowered(false, false, false, false));
    }
}
