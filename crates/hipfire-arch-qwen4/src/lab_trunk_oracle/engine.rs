// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU half of the trunk-shard layer oracle: localise a loaded bundle to a trunk-shard rank by
//! hand, then run ONE block of the real forward program on injected HC streams.
//!
//! **How a block is isolated.** The forward program is the product's own
//! (`forward_chunk_inner` builds every layer's steps); `forward_mesh_chunk` executes a
//! segment of its op range (the entry the TP mesh uses between collectives). A case runs
//! three segments of one program, in order:
//!
//! 1. the PREFIX (`Embed`, layers before the tested one) on real tokens,
//! 2. the tested layer's MIXER segment (HC read, GDN/QSA, out/o projection, attention HC
//!    write), and optionally
//! 3. its FFN segment (MLP HC read, MoE, MLP HC write).
//!
//! Between segments the HC streams (`scratch.streams`, the mesh's stream view) are replaced
//! by the injected input, so the block under test sees exactly the same bytes on every arm
//! and rank. The last segment (the head, the state commit) is never run: nothing is
//! committed, and the next case starts from `Qwen4Bundle::reset`.

use super::*;
use crate::gpu_forward::{Qwen4GpuForward, Qwen4MeshHead, Qwen4OutputRows};
use crate::state::Qwen4StateFormat;
use crate::weights::trunk_shard::QsaMode;
use hipfire_runtime::multi_gpu::executor::MeshRoute;
use rdna_compute::{DType, Gpu, GpuTensor};

fn mesh_route(route: RouteSel) -> MeshRoute {
    match route {
        RouteSel::Decode => MeshRoute::Decode,
        RouteSel::Verify => MeshRoute::Verify,
        RouteSel::Prefill => MeshRoute::Prefill,
    }
}

fn es<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

// ============================================================================================
// Localising a bundle
// ============================================================================================

/// Facts of a localised bundle, for the receipt.
#[derive(Clone, Debug)]
pub struct Localized {
    /// `Some((rank, parts))` when the trunk was cut, `None` for the unsharded reference.
    pub shard: Option<(usize, usize)>,
    pub line: String,
}

/// Turn a bundle loaded by the product loader (global config, forward attached) into the
/// arm's bundle: AR-only config, and for `rank = Some(r)` the rank-local config with its half
/// of the trunk cut by [`crate::weights::Qwen4Weights::shard_trunk`] (after the product's
/// overlay/requant, exactly where `apply_trunk_shard` runs), a state allocated at the local
/// sizes and a forward rebuilt on it. The reference (`rank = None`) goes through the same
/// free/re-attach so both arms share the AR-only config and a fresh forward.
///
/// This is the by-hand equivalent of the load path the `HIPFIRE_QWEN4_EP_TRUNK_SHARD` switch
/// will take; it does NOT read the switch (inert until the route is ready) and runs on a
/// single-device load, so it covers the mixer blocks (the FFN partial needs the EP2 expert
/// ownership).
pub fn localize_bundle(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    rank: Option<usize>,
    max_seq: usize,
) -> Result<Localized, String> {
    gpu.bind_thread().map_err(es)?;
    if bundle.execution.is_none() {
        return Err("localize_bundle: the bundle has no forward attached".into());
    }
    if bundle.ranks() != 1 {
        return Err("localize_bundle: needs a single-device bundle".into());
    }
    let state_format: Qwen4StateFormat = bundle.state_format();
    let backend = bundle.state.qsa_backend();
    let ar = bundle.config.clone().into_ar_only()?;
    let (local, shard) = match rank {
        Some(rank) => (ar.trunk_shard_local(rank, 2)?, Some((rank, 2))),
        None => (ar.clone(), None),
    };
    // 1. Drop the global forward (and everything sized from the global config).
    let forward = bundle.execution.take().ok_or("no forward")?;
    let (max_chunk, mode) = (forward.scratch.max_chunk, forward.scratch.mode);
    forward.free_gpu(gpu).map_err(es)?;
    if let Some(mtp) = bundle.mtp.take() {
        mtp.free_gpu(gpu).map_err(es)?;
    }
    for tensor in [
        bundle.spec_logits.take(),
        bundle.spec_top1.take(),
        bundle.spec_hidden.take(),
    ]
    .into_iter()
    .flatten()
    {
        gpu.free_tensor(tensor).map_err(es)?;
    }
    // 2. Cut the weights (rank-local only).
    let mut cut = String::from("not cut (reference)");
    if let Some(rank) = rank {
        let applied = bundle
            .weights
            .shard_trunk(gpu, &ar, QsaMode::GroupInterleaved, rank, 2)?;
        cut = format!(
            "{} tensors, {} -> {} MiB resident",
            applied.tensors,
            applied.before_bytes >> 20,
            applied.after_bytes >> 20
        );
    }
    // 3. State at the local sizes.
    let scope = bundle.weights.band_scope().clone();
    let mut state = crate::state::Qwen4State::new_with_backend_scoped(
        gpu,
        &local,
        max_seq,
        state_format,
        backend,
        &scope,
    )
    .map_err(es)?;
    state.bind_transaction_generation(bundle.weight_store.inventory_len() as u64);
    let old = std::mem::replace(&mut bundle.state, state);
    old.free_gpu(gpu).map_err(es)?;
    bundle.config = local.clone();
    // 4. The forward on the local config.
    bundle.attach_forward(gpu, max_chunk, mode).map_err(es)?;
    Ok(Localized {
        shard,
        line: format!(
            "{}: key heads {} value heads {} q heads {} kv heads {} shared {}; cut: {cut}; forward chunk {} rows",
            match shard {
                Some((rank, parts)) => format!("trunk shard rank {rank}/{parts}"),
                None => "unsharded AR-only".to_string(),
            },
            local.linear_num_key_heads,
            local.linear_num_value_heads,
            local.num_attention_heads,
            local.num_key_value_heads,
            local.shared_expert_intermediate_size,
            max_chunk
        ),
    })
}

/// This thread's trunk-shard route for the NEXT program builds: the mixer HC writes of the
/// forward carry the fused all-reduce through `exchange` with mixer partials of the given wire
/// format (`Some((exchange, rank, mixer))`), or none
/// (`None`). The FFN route stays off: the oracle runs the TP mixer only (see the example's
/// design note). Per thread: call on the rank's own thread.
pub fn install_route(route: Option<(Exchange, usize, ArHcPartial)>) {
    crate::trunk_route::install(route.map(|(exchange, rank, mixer)| {
        crate::trunk_route::TrunkShardRoute::new(exchange, rank, true, false, mixer)
    }));
}

// ============================================================================================
// Running a case
// ============================================================================================

/// Where the block input comes from.
#[derive(Clone, Debug)]
pub enum InputSource {
    /// [`synthetic_stream`] of this seed.
    Synthetic(u64),
    /// The real streams the PREFIX produced (layers before the tested one on the prompt
    /// tokens): the input the product's own forward would hand the layer.
    Real,
    /// These bytes (a reference arm's recorded input).
    Fixed(Vec<u8>),
}

/// The stream geometry of a case on a loaded bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub route_rows: usize,
    pub kind: StreamKind,
    /// Rows the forward's scratch was built for.
    pub max_chunk: usize,
}

/// This rank's side of a TP2 run.
#[derive(Clone)]
pub struct TpHook {
    pub exchange: Exchange,
    pub rank: usize,
    /// Wire format of the mixer partials (`topology.ep_ar_wire`).
    pub mixer: ArHcPartial,
}

/// What [`run_case`] returns.
#[derive(Clone, Debug)]
pub struct RunOutcome {
    pub geometry: Geometry,
    /// The captures in execution order.
    pub blocks: Vec<(Block, BlockCapture)>,
    pub notes: Vec<String>,
}

fn take_forward(bundle: &mut Qwen4Bundle) -> Result<Qwen4GpuForward, String> {
    bundle
        .execution
        .take()
        .ok_or_else(|| "the bundle has no forward attached".to_string())
}

/// Stream kind, routing rows and capacity of `spec` on this bundle.
pub fn geometry(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    spec: &CaseSpec,
) -> Result<Geometry, String> {
    gpu.bind_thread().map_err(es)?;
    let forward = take_forward(bundle)?;
    let result = forward
        .mesh_layout(bundle, gpu, mesh_route(spec.route), spec.rows)
        .map(|layout| Geometry {
            route_rows: layout.route_rows,
            kind: StreamKind::from_state_bf16(layout.bf16_streams),
            max_chunk: forward.scratch.max_chunk,
        })
        .map_err(es);
    bundle.execution = Some(forward);
    result
}

fn upload(gpu: &Gpu, view: &GpuTensor, bytes: &[u8]) -> Result<(), String> {
    gpu.hip.device_synchronize().map_err(es)?;
    gpu.hip.memcpy_htod(&view.buf, bytes).map_err(es)?;
    gpu.hip.device_synchronize().map_err(es)
}

fn download(gpu: &Gpu, view: &GpuTensor, bytes: usize) -> Result<Vec<u8>, String> {
    gpu.hip.device_synchronize().map_err(es)?;
    let mut out = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut out, &view.buf).map_err(es)?;
    Ok(out)
}

fn download_f32(gpu: &Gpu, tensor: &GpuTensor, elements: usize) -> Result<Vec<f32>, String> {
    gpu.hip.device_synchronize().map_err(es)?;
    gpu.download_f32(&tensor.sub_offset(0, elements))
        .map_err(es)
}

/// Run the requested `blocks` of `spec.layer` on the loaded (and, for TP, localised) bundle.
///
/// `blocks` run in layer order; `Ffn` after `Mixer` is CHAINED (the FFN sees the mixer's
/// output), `Ffn` alone sees the injected input (the mixer runs first only to keep the
/// program in order and is discarded). With `tp`, the mixer segment runs under this rank's
/// route (fused all-reduce); every other segment runs with no route. Both TP ranks must call
/// this with identical arguments (the all-reduce rendezvouses inside the mixer segment).
#[allow(clippy::too_many_arguments)]
pub fn run_case(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    spec: &CaseSpec,
    blocks: &[Block],
    tokens: &[u32],
    input: &InputSource,
    tp: Option<&TpHook>,
) -> Result<RunOutcome, String> {
    gpu.bind_thread().map_err(es)?;
    if tokens.len() < spec.rows {
        return Err(format!(
            "{} tokens for {} rows: pass --prompt-ids/--prompt-file with enough tokens",
            tokens.len(),
            spec.rows
        ));
    }
    let plan = layer_plan(&bundle.config, spec.layer)?;
    install_route(None);
    bundle.reset(gpu).map_err(es)?;
    let mut forward = take_forward(bundle)?;
    let result = run_case_inner(
        bundle,
        &mut forward,
        gpu,
        spec,
        &plan,
        blocks,
        &tokens[..spec.rows],
        input,
        tp,
    );
    install_route(None);
    forward.abandon_mesh_program();
    bundle.execution = Some(forward);
    result
}

#[allow(clippy::too_many_arguments)]
fn run_case_inner(
    bundle: &mut Qwen4Bundle,
    forward: &mut Qwen4GpuForward,
    gpu: &mut Gpu,
    spec: &CaseSpec,
    plan: &LayerPlan,
    blocks: &[Block],
    tokens: &[u32],
    input: &InputSource,
    tp: Option<&TpHook>,
) -> Result<RunOutcome, String> {
    let route = mesh_route(spec.route);
    let rows = spec.rows;
    if rows > forward.scratch.max_chunk {
        return Err(format!(
            "{rows} rows exceed the forward's {} row chunk",
            forward.scratch.max_chunk
        ));
    }
    let layout = forward.mesh_layout(bundle, gpu, route, rows).map_err(es)?;
    let kind = StreamKind::from_state_bf16(layout.bf16_streams);
    let geometry = Geometry {
        route_rows: layout.route_rows,
        kind,
        max_chunk: forward.scratch.max_chunk,
    };
    forward
        .prepare_mesh_capacity(bundle, gpu, rows, layout.route_rows)
        .map_err(es)?;
    // Verify on the (whole-model) output band carries a head and the wide capture; the head
    // ops are never reached, the buffers only satisfy the entry's contract.
    let logits = gpu
        .zeros(&[bundle.config.vocab_size], DType::F32)
        .map_err(es)?;
    let capture = gpu
        .zeros(
            &[rows * bundle.config.hc_count * bundle.config.hidden_size],
            DType::F32,
        )
        .map_err(es)?;
    let verify = route == MeshRoute::Verify;
    let head = verify.then_some(Qwen4MeshHead {
        logits: &logits,
        top1: None,
        rows: Qwen4OutputRows::Final,
    });
    let wide = verify.then_some(&capture);
    // The plan numbers ops from the step graph's first op; the mesh numbers them from the
    // band's first op (0 for the whole-model band, but never assume it).
    let base = forward.mesh_op_range(false, false).start;
    let segment = |bundle: &mut Qwen4Bundle,
                   forward: &mut Qwen4GpuForward,
                   gpu: &mut Gpu,
                   ops: std::ops::Range<u32>|
     -> Result<(), String> {
        let ops = ops.start + base..ops.end + base;
        forward
            .forward_mesh_chunk(
                bundle,
                gpu,
                tokens,
                layout.route_rows,
                route,
                head,
                wide,
                ops,
            )
            .map_err(es)?;
        gpu.hip.device_synchronize().map_err(es)?;
        gpu.take_device_wait_error().map_err(es)
    };

    // 1. Prefix.
    segment(bundle, forward, gpu, plan.prefix.clone())?;
    let view = forward.mesh_stream_view(&layout).map_err(es)?;
    let bytes = layout.stream_bytes;
    let real = download(gpu, &view, bytes)?;
    let mut stream: Vec<u8> = match input {
        InputSource::Real => real,
        InputSource::Synthetic(seed) => synthetic_stream(*seed, rows, kind),
        InputSource::Fixed(fixed) => fixed.clone(),
    };
    if stream.len() != bytes {
        return Err(format!(
            "input stream is {} bytes, the {} stream of {rows} rows is {bytes}",
            stream.len(),
            kind.name()
        ));
    }
    let fuse = gpu.flags.qwen4_hc_fuse_level();
    let mut notes = vec![format!(
        "route {} rows {rows} routing rows {} streams {} (HC_FUSE level {fuse})",
        spec.route.name(),
        layout.route_rows,
        kind.name()
    )];
    let mut out: Vec<(Block, BlockCapture)> = Vec::new();
    let mut device_current = false; // does `scratch.streams` already hold `stream`?
    let mut mixer_done = false;
    let mut ordered: Vec<Block> = blocks.to_vec();
    ordered.sort();
    ordered.dedup();
    for block in ordered {
        if block == Block::Ffn && !mixer_done {
            // Keep the program in order; its effect on the streams is discarded below.
            upload(gpu, &view, &stream)?;
            segment(bundle, forward, gpu, plan.mixer.clone())?;
            mixer_done = true;
            device_current = false;
        }
        if !device_current {
            upload(gpu, &view, &stream)?;
        }
        let input_bytes = stream.clone();
        let (ops, tp_here) = match block {
            Block::Mixer => (plan.mixer.clone(), tp),
            Block::Ffn => (plan.ffn.clone(), None),
        };
        if let Some(hook) = tp_here {
            install_route(Some((hook.exchange.clone(), hook.rank, hook.mixer)));
        }
        let ran = segment(bundle, forward, gpu, ops);
        install_route(None);
        ran?;
        if block == Block::Mixer {
            mixer_done = true;
        }
        let output = download(gpu, &view, bytes)?;
        let gates = download_f32(gpu, &forward.scratch.hc_gates, rows * HC_BRANCHES)?;
        // The block output: a TP rank's partial is always materialised (the fused write is
        // declined for an all-reduce write); on the reference the projection may have fused
        // its HC write into the GEMM epilogue (HC_FUSE >= 2), leaving nothing to read.
        let materialised = match (block, tp_here.is_some()) {
            (Block::Mixer, true) => true,
            (Block::Mixer, false) => fuse < 2,
            (Block::Ffn, _) => false,
        };
        let partial_tensor = match block {
            Block::Mixer => &forward.scratch.attention_output,
            Block::Ffn => &forward.scratch.ffn_output,
        };
        let partial = if materialised {
            Some(download_f32(gpu, partial_tensor, rows * AR_HC_HIDDEN)?)
        } else {
            None
        };
        out.push((
            block,
            BlockCapture {
                rows,
                kind,
                input: input_bytes,
                output: output.clone(),
                gates,
                partial,
            },
        ));
        stream = output;
        device_current = true;
    }
    notes.push(format!(
        "layer {} ({})",
        spec.layer,
        layer_type_name(plan.layer_type)
    ));
    Ok(RunOutcome {
        geometry,
        blocks: out,
        notes,
    })
}

// ============================================================================================
// Rank-pair qualification of the exchange
// ============================================================================================

/// Register `exchange`'s mailbox for `rank` on `gpu` (call on the rank's thread, before any
/// rank calls [`enable_exchange`]).
pub fn register_exchange(
    exchange: &ArHcExchange,
    rank: usize,
    gpu: &mut Gpu,
) -> Result<(), String> {
    exchange.register(rank, gpu).map_err(es)
}

/// Enable peer access from `rank`'s device to the peer's (after BOTH ranks registered).
pub fn enable_exchange(exchange: &ArHcExchange, rank: usize, gpu: &mut Gpu) -> Result<(), String> {
    exchange.enable_peers(rank, gpu).map_err(es)
}

/// The exchange's own load-time qualification on both stream kinds (BOTH ranks must call this
/// concurrently, after both enabled peer access: the kernels rendezvous).
pub fn qualify_exchange(exchange: &ArHcExchange, rank: usize, gpu: &mut Gpu) -> Result<(), String> {
    exchange
        .qualify(rank, gpu, 8, 16)
        .map_err(|e| format!("qualification: {e}"))
}

/// Zero the exchange's generation words and mailbox on `rank` (both ranks drained), after a
/// failed case.
pub fn reset_exchange(exchange: &ArHcExchange, rank: usize, gpu: &mut Gpu) -> Result<(), String> {
    exchange.reset(rank, gpu).map_err(es)
}
