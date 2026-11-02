// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Synthetic `bench_prefill` / `bench_decode` forwards for every loaded
//! multi-device topology `serve` admits.
//!
//! The routes here call the very entrypoints the serve generate paths call:
//! Qwen3.5 PP `forward_prefill_batch_multi` / `forward_scratch_multi`, Qwen4 PP
//! `pp_prefill_final` / `pp_forward_token_or_argmax`, Qwen3.5 MoE EP
//! [`qwen35_moe_ep_prefill_window`] / [`qwen35_moe_ep_decode_forward`], dense
//! Qwen TP [`dense_tp_prefill_chunk`] / [`dense_tp_decode_forward`], DeepSeek4
//! EP [`ds4_ep_prefill`] / `forward_ep` and MiniMax EP `forward_ep`. The last
//! four on the Qwen side live in `crate::qwen` and are shared with the serve
//! loops themselves. Single-card loads and Qwen4 expert parallelism (whose
//! bundle fans the single-device call out over its ranks) keep the architecture
//! carriers' hooks and never reach this module.
//!
//! The daemon's own `Gpu` is not a model device on a mesh route, so every entry
//! synchronizes the model devices itself before it returns: the caller's timer
//! then spans completed work on every rank.

use crate::qwen::{
    dense_tp_decode_forward, dense_tp_mesh_drain_inner, dense_tp_prefill_chunk,
    dense_tp_prefill_chunk_len, ds4_ep_prefill, qwen35_moe_ep_decode_forward,
    qwen35_moe_ep_prefill_window, qwen4_bundle_and_mesh,
};
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_minimax as minimax;
use hipfire_arch_qwen35::qwen35;
use hipfire_loader::bench_topology::BenchRoute;
use hipfire_loader::{EpArch, EpState, LoadedModel};
use hipfire_runtime::multi_gpu::Gpus;
use std::any::Any;

/// The deterministic decode-token schedule of `bench_decode`: the same
/// `101 + i % 1000` the single-card carriers step with.
fn decode_token(i: usize) -> u32 {
    101 + (i as u32 % 1000)
}

/// Attested all-rank reset of a mesh route's request state (host cursors,
/// recurrent state, KV, graph invalidation, per-device sync) through the shared
/// mesh epilogue `serve` uses. `Single` and Qwen4 EP are reset by the daemon's
/// ordinary single-device sequence.
pub fn reset_mesh_for_bench(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), String> {
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    let epilogue = crate::common::reset_mesh_request_state(m, Some(gpu));
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    if epilogue.rolled_back {
        Ok(())
    } else {
        Err(epilogue
            .context
            .unwrap_or_else(|| "rollback could not be attested".to_string()))
    }
}

/// Drain every model device of `gpus` (each under its own bound thread) and,
/// when `waits` is set, surface the bounded device-wait / router faults the
/// whole-token TP graphs record.
fn sync_gpus(gpus: &mut Gpus, waits: bool) -> Result<(), String> {
    for (rank, device) in gpus.devices.iter_mut().enumerate() {
        device
            .bind_thread()
            .map_err(|e| format!("rank {rank} bind_thread: {e:?}"))?;
        device
            .hip
            .device_synchronize()
            .map_err(|e| format!("rank {rank} device_synchronize: {e:?}"))?;
    }
    if waits {
        gpus.take_device_wait_errors()
            .map_err(|e| format!("device wait: {e}"))?;
    }
    Ok(())
}

/// Quiesce the model devices of `route`. A no-op for routes the daemon `Gpu`
/// itself drains.
pub fn quiesce_mesh(route: BenchRoute, m: &mut LoadedModel) -> Result<(), String> {
    match route {
        BenchRoute::Single | BenchRoute::Qwen4Ep => Ok(()),
        BenchRoute::Qwen35Pp | BenchRoute::Qwen4Pp | BenchRoute::CarrierPp => {
            match m.pp_gpus.as_mut() {
                Some(gpus) => sync_gpus(gpus, false),
                None => Err("pipeline-parallel model has no Gpus".to_string()),
            }
        }
        BenchRoute::Qwen35DenseTp => match m.ep.as_mut() {
            Some(ep) => {
                // The mesh route only enqueues: retire its epochs (a device
                // sync of every rank) so the timer covers completed work,
                // then run the common sync and wait check whatever happened.
                let drained = dense_tp_mesh_drain_inner(&mut ep.gpus, &mut ep.inner)
                    .map_err(|e| format!("mesh epoch drain: {e}"));
                let synced = sync_gpus(&mut ep.gpus, true);
                drained.and(synced)
            }
            None => Err("dense TP model has no EP state".to_string()),
        },
        BenchRoute::Qwen35MoeEp | BenchRoute::Ds4Ep | BenchRoute::MinimaxEp => {
            match m.ep.as_mut() {
                Some(ep) => sync_gpus(&mut ep.gpus, false),
                None => Err("expert-parallel model has no EP state".to_string()),
            }
        }
    }
}

fn not_a_mesh_route(route: BenchRoute) -> String {
    format!("{route:?} runs the architecture carrier's bench hooks, not a mesh route")
}

fn qwen35_pp_bundle(
    m: &mut LoadedModel,
) -> Result<(&mut hipfire_arch_qwen35::Qwen35Bundle, &mut Gpus), String> {
    let bundle = m
        .state
        .as_mut()
        .and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        })
        .ok_or_else(|| "pipeline-parallel model has no Qwen3.5 bundle".to_string())?;
    let gpus = m
        .pp_gpus
        .as_mut()
        .ok_or_else(|| "pipeline-parallel model has no Gpus".to_string())?;
    Ok((bundle, gpus))
}

fn qwen35_pp_prefill(m: &mut LoadedModel, tokens: &[u32]) -> Result<(), String> {
    let (b, gpus) = qwen35_pp_bundle(m)?;
    let scratch_set = b
        .pp_scratch_set
        .as_ref()
        .ok_or_else(|| "pipeline-parallel model has no per-device scratch".to_string())?;
    qwen35::forward_prefill_batch_multi(
        gpus,
        &b.weights,
        &b.config,
        tokens,
        0,
        &mut b.kv_cache,
        &mut b.dn_state,
        scratch_set,
    )
    .map_err(|e| format!("forward_prefill_batch_multi: {e}"))
}

fn qwen35_pp_decode(m: &mut LoadedModel, token: u32, pos: usize) -> Result<(), String> {
    let (b, gpus) = qwen35_pp_bundle(m)?;
    let scratch_set = b
        .pp_scratch_set
        .as_ref()
        .ok_or_else(|| "pipeline-parallel model has no per-device scratch".to_string())?;
    qwen35::forward_scratch_multi(
        gpus,
        &b.weights,
        &b.config,
        token,
        pos,
        &mut b.kv_cache,
        &mut b.dn_state,
        scratch_set,
    )
    .map_err(|e| format!("forward_scratch_multi decode: {e}"))
}

/// Qwen4 mesh PP: the logits row lives on the daemon `Gpu` exactly as in the
/// serve AR producer (`generate_ar_with_forward`).
fn qwen4_pp_prefill(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    tokens: &[u32],
) -> Result<(), String> {
    let vocab = m
        .qwen4()
        .map(|bundle| bundle.config.vocab_size)
        .ok_or_else(|| "qwen4 pipeline-parallel model has no Qwen4 bundle".to_string())?;
    let logits = gpu
        .zeros(&[vocab], rdna_compute::DType::F32)
        .map_err(|e| format!("qwen4 pp bench logits: {e:?}"))?;
    let result = match qwen4_bundle_and_mesh(m) {
        Some((bundle, Some(gpus))) => bundle
            .pp_prefill_final(gpus, tokens, 0, &logits)
            .map_err(|e| e.to_string()),
        _ => Err("qwen4 pipeline-parallel model has no Gpus before prefill".to_string()),
    };
    let _ = gpu.free_tensor(logits);
    result
}

fn qwen4_pp_decode(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    context: usize,
    iterations: usize,
) -> Result<(), String> {
    let vocab = m
        .qwen4()
        .map(|bundle| bundle.config.vocab_size)
        .ok_or_else(|| "qwen4 pipeline-parallel model has no Qwen4 bundle".to_string())?;
    let logits = gpu
        .zeros(&[vocab], rdna_compute::DType::F32)
        .map_err(|e| format!("qwen4 pp bench logits: {e:?}"))?;
    let result = match qwen4_bundle_and_mesh(m) {
        Some((bundle, Some(gpus))) => (0..iterations).try_for_each(|i| {
            bundle
                .pp_forward_token_or_argmax(gpus, Some(decode_token(i)), &logits)
                .map(|_| ())
                .map_err(|e| format!("iter {i} pos {}: {e}", context + i))
        }),
        _ => Err("qwen4 pipeline-parallel model has no Gpus during decode".to_string()),
    };
    let _ = gpu.free_tensor(logits);
    result
}

fn minimax_ep_forward(
    m: &mut LoadedModel,
    token: u32,
    pos: usize,
    what: &str,
) -> Result<(), String> {
    let Some(EpState { gpus, inner }) = m.ep.as_mut() else {
        return Err(format!("minimax EP {what} without EP state"));
    };
    let EpArch::Minimax {
        config,
        weights,
        state,
        partials,
    } = inner
    else {
        return Err("EP arch mismatch (expected minimax)".to_string());
    };
    minimax::forward::forward_ep(gpus, weights, config, state, partials, token, pos as u32)
        .map_err(|e| format!("forward_ep {what}: {}", format!("{e}").replace('"', "'")))
}

fn ds4_ep_decode(m: &mut LoadedModel, token: u32, pos: usize) -> Result<(), String> {
    let Some(EpState { gpus, inner }) = m.ep.as_mut() else {
        return Err("ds4 EP decode without EP state".to_string());
    };
    let EpArch::Ds4 {
        config,
        weights,
        state,
        partials,
        ..
    } = inner
    else {
        return Err("EP arch mismatch (expected ds4)".to_string());
    };
    deepseek4::forward::forward_ep(gpus, weights, config, state, partials, token, pos as u32)
        .map_err(|e| format!("forward_ep decode: {}", format!("{e}").replace('"', "'")))
}

/// Prefill `tokens` from position 0 on a mesh route, then drain every model
/// device. The state was reset by the caller ([`reset_mesh_for_bench`]).
pub fn bench_prefill_mesh(
    route: BenchRoute,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    tokens: &[u32],
) -> Result<(), String> {
    let run = match route {
        BenchRoute::Single | BenchRoute::Qwen4Ep | BenchRoute::CarrierPp => {
            return Err(not_a_mesh_route(route))
        }
        BenchRoute::Qwen35Pp => qwen35_pp_prefill(m, tokens),
        BenchRoute::Qwen4Pp => qwen4_pp_prefill(m, gpu, tokens),
        BenchRoute::Qwen35MoeEp => {
            let mut off = 0usize;
            let mut result = Ok(());
            while off < tokens.len() {
                match qwen35_moe_ep_prefill_window(m, tokens, off) {
                    Ok(end) => off = end,
                    Err(e) => {
                        result = Err(e);
                        break;
                    }
                }
            }
            result
        }
        BenchRoute::Qwen35DenseTp => dense_tp_prefill_chunk_len(m).and_then(|chunk| {
            tokens
                .chunks(chunk)
                .enumerate()
                .try_for_each(|(index, part)| dense_tp_prefill_chunk(m, part, index * chunk))
        }),
        BenchRoute::Ds4Ep => ds4_ep_prefill(m, tokens, || false).map(|_| ()),
        BenchRoute::MinimaxEp => tokens
            .iter()
            .enumerate()
            .try_for_each(|(pos, &token)| minimax_ep_forward(m, token, pos, "prefill")),
    };
    let synced = quiesce_mesh(route, m);
    run.and(synced)
}

/// `iterations` single-token decode forwards at positions `context..`, then
/// drain every model device. The context was primed by
/// [`bench_prefill_mesh`]. Errors name the failing iteration.
pub fn bench_decode_run_mesh(
    route: BenchRoute,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    context: usize,
    iterations: usize,
) -> Result<(), String> {
    let run = match route {
        BenchRoute::Single | BenchRoute::Qwen4Ep | BenchRoute::CarrierPp => {
            return Err(not_a_mesh_route(route))
        }
        BenchRoute::Qwen4Pp => qwen4_pp_decode(m, gpu, context, iterations),
        _ => (0..iterations).try_for_each(|i| {
            let (token, pos) = (decode_token(i), context + i);
            let step = match route {
                BenchRoute::Qwen35Pp => qwen35_pp_decode(m, token, pos),
                BenchRoute::Qwen35MoeEp => qwen35_moe_ep_decode_forward(m, token, pos),
                BenchRoute::Qwen35DenseTp => dense_tp_decode_forward(m, token, pos),
                BenchRoute::Ds4Ep => ds4_ep_decode(m, token, pos),
                BenchRoute::MinimaxEp => minimax_ep_forward(m, token, pos, "decode"),
                BenchRoute::Single
                | BenchRoute::Qwen4Ep
                | BenchRoute::Qwen4Pp
                | BenchRoute::CarrierPp => Err(not_a_mesh_route(route)),
            };
            step.map_err(|e| format!("iter {i} pos {pos}: {e}"))
        }),
    };
    let synced = quiesce_mesh(route, m);
    run.and(synced)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_tokens_follow_the_single_card_schedule() {
        assert_eq!(decode_token(0), 101);
        assert_eq!(decode_token(1), 102);
        assert_eq!(decode_token(999), 1100);
        assert_eq!(decode_token(1000), 101);
    }

    #[test]
    fn carrier_routes_are_not_mesh_entries() {
        for route in [
            BenchRoute::Single,
            BenchRoute::Qwen4Ep,
            BenchRoute::CarrierPp,
        ] {
            assert!(route.uses_carrier_hooks());
            assert!(not_a_mesh_route(route).contains("carrier"));
        }
        // A carrier-hook pp route is still a mesh route: its devices are not
        // the daemon `Gpu`.
        assert!(BenchRoute::CarrierPp.is_mesh());
        assert!(!BenchRoute::Single.is_mesh() && BenchRoute::Qwen4Ep.is_mesh());
        for route in [
            BenchRoute::Qwen35Pp,
            BenchRoute::Qwen4Pp,
            BenchRoute::Qwen35MoeEp,
            BenchRoute::Qwen35DenseTp,
            BenchRoute::Ds4Ep,
            BenchRoute::MinimaxEp,
        ] {
            assert!(!route.uses_carrier_hooks() && route.is_mesh(), "{route:?}");
        }
    }
}
