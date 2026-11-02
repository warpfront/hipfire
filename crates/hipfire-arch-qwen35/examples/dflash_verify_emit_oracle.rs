// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase oracle for railgun design A slice A5 (byte-identity ladder, step 3:
//! "HIP graph verify versus emitted HIP"; design `docs/design/2026-10-25-dflash-railgun-window.md`
//! §2 "Authoring, not capture", §3, §4, §6, §7).
//!
//! The same preimage (DeltaNet S/scale/conv/EF snapshot, every KV buffer,
//! hidden ring + staging + head/written, GDN tape, verify outputs poisoned,
//! GDN requant-frame counter, pbs token/position inputs) is run twice:
//!
//! * **graph arm** — the current `verify_dflash_block` HIP-graph body. Three
//!   restored windows give warmup (direct), capture(+first launch) and a pure
//!   graph *replay*; the replay is the arm result. The capture window runs
//!   under the hip-bridge launch witness, which records every real blob launch
//!   (symbol, grid, block, LDS, kernarg bytes) — the independent ground truth
//!   for the launch list.
//! * **twin arm** — `plan_dflash_verify_full` (rdna-compute plan sink, nothing
//!   launched while planning) followed by `Gpu::run_planned_ops` (ordinary HIP
//!   launches of the planned kernarg images, stream-ordered copies/fills),
//!   then the host ring delta published once.
//!
//! Gates (all must hold; exit code 1 otherwise):
//!
//! 1. the graph arm really replayed a captured graph (`verify_has_graph`);
//! 2. the plan's kernel list equals the capture-window witness: same symbols
//!    in the same order, same grid/block/LDS, byte-identical kernarg images;
//! 3. the declared patch table is complete: a second plan at another
//!    `start_pos` + ring head differs only at declared `Scalar32` words;
//! 4. every compared surface is byte-identical between arms: logits, final
//!    hidden, argmax (device + host), GDN tape, live DeltaNet
//!    S/scale/conv/EF, every KV buffer (so rows `[p, p+16)`), hidden ring,
//!    hidden staging, ring head/written, GDN frame counter;
//! 5. negative controls: a one-byte flip in each surface class is caught by
//!    the comparator, and a wrong-head plan is caught by gate 2.
//!
//! Run (gfx1201; fp8 KV; HIPFIRE_VERIFY_GRAPH=1 admits the graph body for
//! exact-gfx1201 MQ4G256 heads, which are default-off):
//!
//! ```text
//! HIPFIRE_VERIFY_GRAPH=1 cargo run --release -p hipfire-arch-qwen35 \
//!   --example dflash_verify_emit_oracle -- \
//!   --model ~/.hipfire/models/qwen3.8-27b.mq4-xts \
//!   --draft ~/.hipfire/models/qwen38-27b-dflash-mq4.hfq \
//!   --kv fp8 --ctx 4096 --pos 1024 --ring-slack 64
//! # SWA/ring-wrap commit (head + 16 > ring capacity):
//!   ... --ring-slack 8
//! ```

use hip_bridge::HipResult;
use hipfire_arch_qwen35::dflash_verify_emit::{
    plan_dflash_verify_full, undeclared_varying_words, DflashVerifyPlan,
};
use hipfire_arch_qwen35::qwen35::{self, LayerType};
use hipfire_arch_qwen35::speculative::{
    verify_dflash_block, DeltaNetSnapshot, GdnTape, HiddenStateRingBuffer, KvMode, ModelSlot,
    ModelSlotConfig, VerifyScratch,
};
use hipfire_runtime::dflash::DflashConfig;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use rdna_compute::plan_sink::plan_symbols;
use rdna_compute::{Gpu, GpuTensor, PlannedKernel, PlannedOp};
use std::path::{Path, PathBuf};

const B: usize = 16;

struct Args {
    model: PathBuf,
    draft: PathBuf,
    kv: String,
    ctx: usize,
    pos: usize,
    ring_slack: usize,
}

fn expand_home(path: &str) -> PathBuf {
    match path.strip_prefix("~/") {
        Some(rest) => Path::new(&std::env::var("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(path),
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        model: expand_home("~/.hipfire/models/qwen3.8-27b.mq4-xts"),
        draft: expand_home("~/.hipfire/models/qwen38-27b-dflash-mq4.hfq"),
        kv: "fp8".into(),
        ctx: 4096,
        pos: 1024,
        ring_slack: 64,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => args.model = expand_home(&value()?),
            "--draft" => args.draft = expand_home(&value()?),
            "--kv" => args.kv = value()?,
            "--ctx" => args.ctx = value()?.parse().map_err(|e| format!("--ctx: {e}"))?,
            "--pos" => args.pos = value()?.parse().map_err(|e| format!("--pos: {e}"))?,
            "--ring-slack" => {
                args.ring_slack = value()?.parse().map_err(|e| format!("--ring-slack: {e}"))?
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if args.kv != "fp8" && args.kv != "q8" {
        return Err(format!("--kv must be fp8 or q8 (got {})", args.kv));
    }
    if args.pos + B + 8 > args.ctx {
        return Err(format!(
            "--ctx {} too small for --pos {}",
            args.ctx, args.pos
        ));
    }
    Ok(args)
}

fn he<T>(r: HipResult<T>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("{what}: {e}"))
}

// ── Surfaces: named byte blobs of every compared device/host state ─────────

type Surfaces = Vec<(String, Vec<u8>)>;

fn download(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0u8; t.buf.size()];
    he(gpu.hip.memcpy_dtoh(&mut bytes, &t.buf), "download")?;
    Ok(bytes)
}

fn upload(gpu: &Gpu, t: &GpuTensor, bytes: &[u8]) -> Result<(), String> {
    he(gpu.hip.memcpy_htod(&t.buf, bytes), "upload")
}

fn poison(gpu: &Gpu, t: &GpuTensor) -> Result<(), String> {
    he(gpu.hip.memset(&t.buf, 0xCD, t.buf.size()), "poison")
}

fn push_group(
    out: &mut Surfaces,
    gpu: &Gpu,
    name: &str,
    tensors: &[GpuTensor],
) -> Result<(), String> {
    for (i, t) in tensors.iter().enumerate() {
        out.push((format!("{name}[{i}]"), download(gpu, t)?));
    }
    Ok(())
}

fn host_word(name: &str, value: u64) -> (String, Vec<u8>) {
    (name.to_string(), value.to_le_bytes().to_vec())
}

fn surfaces(
    gpu: &Gpu,
    target: &ModelSlot,
    ring: &HiddenStateRingBuffer,
    tape: &GdnTape,
    vs: &VerifyScratch,
) -> Result<Surfaces, String> {
    he(gpu.hip.device_synchronize(), "device_synchronize")?;
    let mut s = Surfaces::new();
    push_group(&mut s, gpu, "dn.s", &target.dn_state.s_matrices)?;
    push_group(&mut s, gpu, "dn.scale", &target.dn_state.s_scales)?;
    push_group(&mut s, gpu, "dn.conv", &target.dn_state.conv_states)?;
    push_group(&mut s, gpu, "dn.ef", &target.dn_state.s_ef_residual)?;
    push_group(&mut s, gpu, "kv.k", &target.kv_cache.k_gpu)?;
    push_group(&mut s, gpu, "kv.v", &target.kv_cache.v_gpu)?;
    push_group(&mut s, gpu, "kv.k_scales", &target.kv_cache.k_scales)?;
    push_group(&mut s, gpu, "kv.v_scales", &target.kv_cache.v_scales)?;
    push_group(&mut s, gpu, "ring.layer", &ring.layer_bufs)?;
    push_group(&mut s, gpu, "ring.staging", &ring.staging_bufs)?;
    push_group(&mut s, gpu, "tape.qkv", &tape.qkv_bufs)?;
    push_group(&mut s, gpu, "tape.alpha", &tape.alpha_bufs)?;
    push_group(&mut s, gpu, "tape.beta", &tape.beta_bufs)?;
    s.push((
        "verify.final_hidden".into(),
        download(gpu, &vs.final_hidden)?,
    ));
    s.push(("verify.logits".into(), download(gpu, &vs.logits)?));
    s.push(("verify.argmax".into(), download(gpu, &vs.argmax)?));
    s.push(("verify.rot".into(), download(gpu, &vs.rot)?));
    s.push(host_word("host.ring_head", ring.head as u64));
    s.push(host_word("host.ring_written", ring.written as u64));
    s.push(host_word(
        "host.gdn_frame",
        u64::from(rdna_compute::norm::gdn_requant_frame_checkpoint()),
    ));
    Ok(s)
}

/// Mismatch descriptions between two surface sets (empty = byte-identical).
fn compare(a: &Surfaces, b: &Surfaces) -> Vec<String> {
    let mut out = Vec::new();
    if a.len() != b.len() {
        out.push(format!("{} surfaces vs {}", a.len(), b.len()));
        return out;
    }
    for ((na, ba), (nb, bb)) in a.iter().zip(b) {
        if na != nb {
            out.push(format!("surface name {na} vs {nb}"));
            continue;
        }
        if ba.len() != bb.len() {
            out.push(format!("{na}: {} bytes vs {}", ba.len(), bb.len()));
            continue;
        }
        if ba == bb {
            continue;
        }
        let first = ba.iter().zip(bb).position(|(x, y)| x != y).unwrap_or(0);
        let count = ba.iter().zip(bb).filter(|(x, y)| x != y).count();
        out.push(format!(
            "{na}: {count} of {} bytes differ, first at byte {first} ({:#04x} vs {:#04x})",
            ba.len(),
            ba[first],
            bb[first]
        ));
    }
    out
}

fn total_bytes(s: &Surfaces) -> usize {
    s.iter().map(|(_, b)| b.len()).sum()
}

// ── Preimage restore ────────────────────────────────────────────────────────

struct Preimage {
    surfaces: Surfaces,
    ring_head: usize,
    ring_written: usize,
    frame: u32,
}

#[allow(clippy::too_many_arguments)]
fn restore_preimage(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    ring: &mut HiddenStateRingBuffer,
    tape: &GdnTape,
    vs: &VerifyScratch,
    snap: &DeltaNetSnapshot,
    pre: &Preimage,
) -> Result<(), String> {
    he(gpu.hip.device_synchronize(), "device_synchronize")?;
    he(
        snap.restore_to(&mut target.dn_state, gpu),
        "DeltaNet restore",
    )?;
    let by_name = |name: &str| -> &Vec<u8> {
        &pre.surfaces
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("preimage surface {name}"))
            .1
    };
    let group = |gpu: &Gpu, name: &str, tensors: &[GpuTensor]| -> Result<(), String> {
        for (i, t) in tensors.iter().enumerate() {
            upload(gpu, t, by_name(&format!("{name}[{i}]")))?;
        }
        Ok(())
    };
    group(gpu, "kv.k", &target.kv_cache.k_gpu)?;
    group(gpu, "kv.v", &target.kv_cache.v_gpu)?;
    group(gpu, "kv.k_scales", &target.kv_cache.k_scales)?;
    group(gpu, "kv.v_scales", &target.kv_cache.v_scales)?;
    group(gpu, "ring.layer", &ring.layer_bufs)?;
    group(gpu, "ring.staging", &ring.staging_bufs)?;
    ring.head = pre.ring_head;
    ring.written = pre.ring_written;
    for t in tape
        .qkv_bufs
        .iter()
        .chain(&tape.alpha_bufs)
        .chain(&tape.beta_bufs)
    {
        poison(gpu, t)?;
    }
    for t in [&vs.final_hidden, &vs.logits, &vs.argmax, &vs.rot] {
        poison(gpu, t)?;
    }
    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(pre.frame);
    he(gpu.hip.device_synchronize(), "device_synchronize")?;
    Ok(())
}

// ── Pseudo-random inputs ────────────────────────────────────────────────────

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

fn tokens(state: &mut u64, n: usize, vocab: usize) -> Vec<u32> {
    let span = (vocab.min(60_000) - 1000) as u64;
    (0..n).map(|_| 1000 + (lcg(state) % span) as u32).collect()
}

// ── Launch-list comparison ──────────────────────────────────────────────────

fn compare_kernels(plan: &[&PlannedKernel], witness: &[PlannedKernel]) -> Vec<String> {
    let mut out = Vec::new();
    if plan.len() != witness.len() {
        out.push(format!(
            "plan has {} kernels, capture witness {}",
            plan.len(),
            witness.len()
        ));
    }
    for (i, (p, w)) in plan.iter().zip(witness).enumerate() {
        let mut why = Vec::new();
        if p.symbol != w.symbol {
            why.push(format!("symbol {} vs {}", p.symbol, w.symbol));
        }
        if p.code_object_digest != w.code_object_digest {
            why.push("code object digest".to_string());
        }
        if p.grid != w.grid {
            why.push(format!("grid {:?} vs {:?}", p.grid, w.grid));
        }
        if p.block != w.block {
            why.push(format!("block {:?} vs {:?}", p.block, w.block));
        }
        if p.dynamic_lds != w.dynamic_lds {
            why.push(format!("lds {} vs {}", p.dynamic_lds, w.dynamic_lds));
        }
        // A witness kernarg of 0 bytes is a kernelParams-form launch (bytes
        // not recoverable): symbol/geometry/LDS only.
        if !w.kernarg.is_empty() && p.kernarg != w.kernarg {
            let at = p
                .kernarg
                .iter()
                .zip(&w.kernarg)
                .position(|(x, y)| x != y)
                .unwrap_or_else(|| p.kernarg.len().min(w.kernarg.len()));
            why.push(format!(
                "kernarg ({} vs {} bytes, first diff at byte {at})",
                p.kernarg.len(),
                w.kernarg.len()
            ));
        }
        if !why.is_empty() {
            out.push(format!("kernel {i} {}: {}", w.symbol, why.join("; ")));
            if out.len() >= 12 {
                out.push("(further kernel mismatches elided)".into());
                break;
            }
        }
    }
    out
}

fn run() -> Result<bool, String> {
    let args = parse_args()?;
    let mut gate_ok = true;
    let mut gate = |name: &str, problems: &[String]| {
        if problems.is_empty() {
            println!("[PASS] {name}");
        } else {
            gate_ok = false;
            println!("[FAIL] {name}");
            for p in problems {
                println!("         {p}");
            }
        }
    };

    let mut gpu = he(Gpu::init(), "gpu init")?;
    println!("gpu: {}", gpu.arch);
    if gpu.arch != "gfx1201" {
        println!(
            "WARN: the railgun window is exact gfx1201; running on {}",
            gpu.arch
        );
    }

    let draft_hfq = HfqFile::open(&args.draft).map_err(|e| format!("open draft: {e:?}"))?;
    let draft_cfg = DflashConfig::from_hfq(&draft_hfq).ok_or("parse DflashConfig")?;

    // ── Target (fp8 KV replaces the Q8 cache `ModelSlot::load` builds; the
    // slot constructor rejects fp8, the carrier path builds it the same way).
    let mut slot_cfg = ModelSlotConfig::default();
    slot_cfg.max_seq = args.ctx;
    slot_cfg.kv_mode = KvMode::Q8;
    let mut target = he(
        ModelSlot::load(&mut gpu, &args.model, "target", slot_cfg),
        "load target",
    )?;
    if args.kv == "fp8" {
        let is_kv: Vec<bool> = target
            .config
            .layer_types
            .iter()
            .map(|t| *t == LayerType::FullAttention)
            .collect();
        let fp8 = he(
            KvCache::new_gpu_fp8_filtered(
                &mut gpu,
                &is_kv,
                target.config.n_kv_heads,
                target.config.head_dim,
                args.ctx,
            ),
            "fp8 KV",
        )?;
        let old = std::mem::replace(&mut target.kv_cache, fp8);
        he(old.free_gpu(&mut gpu), "free q8 KV")?;
        target.slot_config.kv_mode = KvMode::Fp8;
    }
    println!(
        "target: {} layers dim {} vocab {} kv_fp8={} output dtype {:?}",
        target.config.n_layers,
        target.config.dim,
        target.config.vocab_size,
        target.kv_cache.quant_fp8,
        target.weights.output.gpu_dtype
    );
    if draft_cfg.hidden != target.config.dim {
        return Err(format!(
            "draft hidden {} != target dim {}",
            draft_cfg.hidden, target.config.dim
        ));
    }

    let dim = target.config.dim;
    let vocab = target.config.vocab_size;
    let ring_cap = args.pos + args.ring_slack;
    let mut ring = he(
        HiddenStateRingBuffer::new(
            &mut gpu,
            target.config.n_layers,
            draft_cfg.num_extract(),
            dim,
            ring_cap,
            qwen35::PREFILL_MAX_BATCH.max(B),
        ),
        "hidden ring",
    )?;
    let mut snap = he(
        DeltaNetSnapshot::new_for(&mut gpu, &target.dn_state),
        "snapshot",
    )?;
    let mut tape = he(
        GdnTape::new_for_dflash(&mut gpu, &target.config, B),
        "gdn tape",
    )?;
    let vs = he(
        VerifyScratch::with_prefill(
            &mut gpu,
            B,
            dim,
            vocab,
            target.weights.output.k,
            &target.config,
        ),
        "verify scratch",
    )?;

    // ── Seed: a real prefill, so DeltaNet state, KV rows and the hidden ring
    // hold non-trivial bytes at the verify position.
    let mut rng = 0x5eed_0000_df1a_u64 ^ (args.pos as u64);
    let prompt = tokens(&mut rng, args.pos, vocab);
    he(
        qwen35::forward_prefill_batch(
            &mut gpu,
            &target.weights,
            &target.config,
            &prompt,
            0,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            Some(&mut ring),
            None,
            None,
            None,
        ),
        "seed prefill",
    )?;
    he(gpu.hip.device_synchronize(), "device_synchronize")?;
    let block = tokens(&mut rng, B, vocab);
    let pos = args.pos;
    println!(
        "seeded: pos {pos} ring cap {ring_cap} head {} written {} (commit wraps: {})",
        ring.head,
        ring.written,
        ring.head + B > ring_cap
    );

    he(snap.save_from(&target.dn_state, &mut gpu), "snapshot save")?;
    let frame0 = rdna_compute::norm::gdn_requant_frame_checkpoint();
    let pre = Preimage {
        surfaces: surfaces(&gpu, &target, &ring, &tape, &vs)?,
        ring_head: ring.head,
        ring_written: ring.written,
        frame: frame0,
    };
    println!(
        "preimage: {} surfaces, {:.1} MiB, gdn frame {frame0}",
        pre.surfaces.len(),
        total_bytes(&pre.surfaces) as f64 / (1 << 20) as f64
    );

    // Restore fidelity: restoring the snapshot/KV/ring must reproduce the
    // preimage bytes before either arm runs.
    restore_preimage(&mut gpu, &mut target, &mut ring, &tape, &vs, &snap, &pre)?;
    let mut fidelity: Vec<String> = Vec::new();
    {
        let now = surfaces(&gpu, &target, &ring, &tape, &vs)?;
        for line in compare(&pre.surfaces, &now) {
            // The poisoned tape / verify outputs are expected to differ.
            if !line.starts_with("tape.") && !line.starts_with("verify.") {
                fidelity.push(line);
            }
        }
    }
    gate(
        "restore fidelity (DN snapshot, KV, ring, head/written)",
        &fidelity,
    );

    // ── Graph arm: warmup (direct), capture (+first launch), replay. ─────
    let mut capture_witness: Vec<PlannedKernel> = Vec::new();
    let mut capture_surfaces = Surfaces::new();
    let mut warm_surfaces = Surfaces::new();
    let mut graph_out = None;
    for step in 0..3 {
        restore_preimage(&mut gpu, &mut target, &mut ring, &tape, &vs, &snap, &pre)?;
        if step == 1 {
            he(gpu.launch_witness_begin(), "launch witness")?;
        }
        let out = he(
            verify_dflash_block(
                &mut gpu,
                &mut target,
                &block,
                pos,
                &mut ring,
                Some(&mut tape),
                false,
                &vs,
            ),
            "graph verify",
        );
        if step == 1 {
            capture_witness = gpu.launch_witness_end()?;
        }
        let out = out?;
        match step {
            0 => warm_surfaces = surfaces(&gpu, &target, &ring, &tape, &vs)?,
            1 => capture_surfaces = surfaces(&gpu, &target, &ring, &tape, &vs)?,
            _ => graph_out = Some(out),
        }
    }
    let has_graph = gpu.graphs.verify_has_graph(B);
    let graph_surfaces = surfaces(&gpu, &target, &ring, &tape, &vs)?;
    let graph_out = graph_out.expect("graph replay output");
    println!(
        "graph arm: verify graph cached={has_graph}, capture witness {} kernels, argmax {:?}",
        capture_witness.len(),
        &graph_out.argmax_per_pos
    );
    gate(
        "graph arm replayed a captured graph (set HIPFIRE_VERIFY_GRAPH=1 on gfx1201 MQ4G256)",
        &if has_graph {
            vec![]
        } else {
            vec!["no HIP graph is cached for B=16: the body took the direct route".to_string()]
        },
    );
    // Info only: warmup is the direct (non-capture) route; capture's first
    // launch must equal the replay (graph determinism from one preimage).
    // The capture window reserves GDN frames; a replay reserves none.
    let capture_vs_replay: Vec<String> = compare(&capture_surfaces, &graph_surfaces)
        .into_iter()
        .filter(|line| !line.starts_with("host.gdn_frame"))
        .collect();
    gate(
        "graph determinism: capture+first-launch == replay",
        &capture_vs_replay,
    );
    let warm_vs_replay = compare(&warm_surfaces, &graph_surfaces);
    println!(
        "info: warmup (direct route) vs graph replay: {} differing surfaces{}",
        warm_vs_replay.len(),
        if warm_vs_replay.is_empty() {
            ""
        } else {
            " (direct and capture routes differ by design)"
        }
    );

    // ── Plan at the preimage. ────────────────────────────────────────────
    restore_preimage(&mut gpu, &mut target, &mut ring, &tape, &vs, &snap, &pre)?;
    let plan: DflashVerifyPlan = plan_dflash_verify_full(
        &mut gpu,
        &mut target,
        &block,
        pos,
        &ring,
        Some(&mut tape),
        &vs,
    )?;
    let frame_after_plan = rdna_compute::norm::gdn_requant_frame_checkpoint();
    println!(
        "plan: {} ops ({} forward / {} commit / {} head), kernels {}, frames reserved {} (base {}), host delta {:?}",
        plan.ops.len(),
        plan.forward.len(),
        plan.commit.len(),
        plan.head.len(),
        plan.kernels().count(),
        plan.gdn_frames_reserved,
        plan.gdn_frame_base,
        plan.host_delta
    );
    let symbols = plan_symbols(&plan.ops);
    let copies = plan
        .ops
        .iter()
        .filter(|o| matches!(o, PlannedOp::Copy { .. }))
        .count();
    let fills = plan
        .ops
        .iter()
        .filter(|o| matches!(o, PlannedOp::Fill { .. }))
        .count();
    println!("plan routed through Copy: {copies}, Fill: {fills}");
    gate(
        "planning is pure (frame counter restored, ring head untouched)",
        &if frame_after_plan == frame0
            && ring.head == pre.ring_head
            && ring.written == pre.ring_written
        {
            vec![]
        } else {
            vec![format!(
                "frame {frame_after_plan} (want {frame0}), head {} written {}",
                ring.head, ring.written
            )]
        },
    );

    // Gate 2: plan vs the capture-window witness.
    let plan_kernels: Vec<&PlannedKernel> = plan.kernels().collect();
    let launch_problems = compare_kernels(&plan_kernels, &capture_witness);
    gate(
        "plan launch list == capture witness (symbols, order, grid, block, LDS, kernarg bytes)",
        &launch_problems,
    );
    let witness_symbols: Vec<&str> = capture_witness.iter().map(|k| k.symbol.as_str()).collect();
    let plan_kernel_symbols: Vec<&str> = plan_kernels.iter().map(|k| k.symbol.as_str()).collect();
    gate(
        "symbol lists identical",
        &if witness_symbols == plan_kernel_symbols {
            vec![]
        } else {
            vec!["plan and capture-witness symbol lists differ".to_string()]
        },
    );
    {
        use std::collections::BTreeMap;
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for s in &plan_kernel_symbols {
            *counts.entry(s).or_default() += 1;
        }
        println!("symbols ({} distinct):", counts.len());
        for (s, n) in &counts {
            println!("  {n:>4} x {s}");
        }
    }

    // Negative control for gate 2: a wrong ring head must not equal the witness.
    {
        let mut wrong: Vec<PlannedKernel> = plan_kernels.iter().map(|k| (*k).clone()).collect();
        if let Some(k) = wrong
            .iter_mut()
            .find(|k| k.symbol.contains("hidden_commit5"))
        {
            let head =
                u32::from_le_bytes([k.kernarg[80], k.kernarg[81], k.kernarg[82], k.kernarg[83]]);
            k.kernarg[80..84].copy_from_slice(&(head ^ 1).to_le_bytes());
        }
        let refs: Vec<&PlannedKernel> = wrong.iter().collect();
        let caught = !compare_kernels(&refs, &capture_witness).is_empty();
        gate(
            "negative control: a flipped commit5 head word is caught by the launch comparator",
            &if caught {
                vec![]
            } else {
                vec!["flip not detected".to_string()]
            },
        );
    }

    // Gate 3: patch table completeness against a second shape.
    let table = plan.patch_table();
    println!("declared patch table:");
    for (node, symbol, patches) in &table {
        println!("  kernel #{node} {symbol}:");
        for p in *patches {
            println!("    {p:?}");
        }
    }
    {
        let saved_head = ring.head;
        let alt_pos = pos + 5;
        ring.head = (saved_head + 3) % ring_cap;
        let alt = plan_dflash_verify_full(
            &mut gpu,
            &mut target,
            &block,
            alt_pos,
            &ring,
            Some(&mut tape),
            &vs,
        );
        ring.head = saved_head;
        let problems = match alt {
            Err(e) => vec![format!("second-shape plan failed: {e}")],
            Ok(alt) => match undeclared_varying_words(&plan, &alt) {
                Err(e) => vec![format!("topology differs between shapes: {e}")],
                Ok(undeclared) => {
                    println!(
                        "shape B (pos {alt_pos}, head {}): kernels differing only at declared words: {}",
                        alt.ring_head,
                        alt.kernels().count()
                    );
                    undeclared
                        .iter()
                        .map(|d| {
                            format!(
                                "undeclared varying word: kernel #{} {} offset {} ({:#x} vs {:#x})",
                                d.kernel, d.symbol, d.offset, d.a, d.b
                            )
                        })
                        .collect()
                }
            },
        };
        gate(
            "patch table complete for a second shape (pos+5, head+3)",
            &problems,
        );
    }
    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame0);

    // ── Twin arm: execute the plan through ordinary HIP launches. ────────
    restore_preimage(&mut gpu, &mut target, &mut ring, &tape, &vs, &snap, &pre)?;
    {
        let pbs = vs
            .prefill_batch
            .as_ref()
            .ok_or("verify scratch has no pbs")?;
        he(
            qwen35::upload_prefill_batch_inputs(&mut gpu, pbs, &block, pos),
            "upload pbs inputs",
        )?;
    }
    he(gpu.run_planned_ops(&plan.ops), "run planned ops")?;
    if let Some(stream) = gpu.active_stream.as_ref() {
        he(gpu.hip.stream_synchronize(stream), "stream sync")?;
    }
    he(gpu.hip.device_synchronize(), "device_synchronize")?;
    let delta = ring.commit_delta(B);
    ring.apply_commit_delta(delta);
    let twin_surfaces = surfaces(&gpu, &target, &ring, &tape, &vs)?;
    let twin_argmax: Vec<u32> = {
        let mut bytes = vec![0u8; B * 4];
        he(
            gpu.hip.memcpy_dtoh(&mut bytes, &vs.argmax.buf),
            "argmax download",
        )?;
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    println!("twin arm: argmax {twin_argmax:?}");

    gate(
        "argmax (host) twin == graph",
        &if twin_argmax == graph_out.argmax_per_pos {
            vec![]
        } else {
            vec![format!(
                "graph {:?} vs twin {:?}",
                graph_out.argmax_per_pos, twin_argmax
            )]
        },
    );
    let arm_mismatch = compare(&graph_surfaces, &twin_surfaces);
    println!(
        "compared {} surfaces / {:.1} MiB per arm",
        graph_surfaces.len(),
        total_bytes(&graph_surfaces) as f64 / (1 << 20) as f64
    );
    gate(
        "twin == graph: logits, final hidden, argmax, tape, DeltaNet S/scale/conv/EF, KV, ring, staging, head/written, frame",
        &arm_mismatch,
    );

    // Gate 5: comparator negative controls (one byte flipped per surface class).
    {
        let mut missed = Vec::new();
        for prefix in [
            "verify.logits",
            "verify.final_hidden",
            "verify.argmax",
            "tape.qkv",
            "dn.s[",
            "dn.conv",
            "kv.k[",
            "ring.layer",
            "ring.staging",
            "host.ring_head",
            "host.gdn_frame",
        ] {
            let mut flipped = twin_surfaces.clone();
            let Some(entry) = flipped
                .iter_mut()
                .find(|(n, b)| n.starts_with(prefix) && !b.is_empty())
            else {
                continue;
            };
            let mid = entry.1.len() / 2;
            entry.1[mid] ^= 0x01;
            if compare(&graph_surfaces, &flipped).is_empty() {
                missed.push(format!("flip in {prefix}* not detected"));
            }
        }
        gate(
            "negative control: byte flips per surface class are caught",
            &missed,
        );
    }

    // Round-trip: hand the preimage state back (not part of any gate).
    restore_preimage(&mut gpu, &mut target, &mut ring, &tape, &vs, &snap, &pre)?;

    println!(
        "{}",
        if gate_ok {
            "VERDICT: PASS — emitted HIP twin is byte-identical to the HIP-graph verify"
        } else {
            "VERDICT: FAIL"
        }
    );
    Ok(gate_ok)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("dflash_verify_emit_oracle: {e}");
            std::process::exit(2);
        }
    }
}
