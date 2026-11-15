// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx942 native-MTP window cost by draft depth, on the real model.
//!
//! Loads MODEL plus its `<model>.mtp` head sidecar through the stock-FP8
//! oracle recipe (`tests/common/stock_mtp_load.rs`, a standalone load that
//! also honors a fixed `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`), prefills PROMPT, then
//! runs real MTP windows through
//! `Qwen4MtpDrafter::mtp_step` with the draft budget cycling `0, 1, .., 7`
//! window by window (so every depth sees the same text, context range and
//! acceptance regime). Each window is timed wall-clock with the device drained
//! on both sides, i.e. a window's whole price: `d` head draft steps, the
//! `d + 1` row verify, the replay repair of a partial accept, the head's
//! kept-row rebuild and the commit. Depth `0` is the interleaved window of
//! budget 0 (one single-row forward plus one head step): the unit
//! `MTP_WINDOW_COST_MS_GFX942[0]` prices every other depth against. A plain AR
//! token (`forward_token_or_argmax`, no head work) is timed after the windows
//! on the same state, for the ms/token comparison.
//!
//! The depth is a pure function of the window budget only under the forced
//! gfx942 depth knob, so run it with
//! `HIPFIRE_QWEN4_MTP_G942_DEPTH=7 HIPFIRE_G942_TRUNK_GEMV` unset (both are
//! checked): the planner then drafts `min(7, budget)` on the batched route with
//! no floor, calibration or cost table.
//!
//! The last line prints the measured table as the Rust literal of
//! `MTP_WINDOW_COST_MS_GFX942` (`mtp_spec.rs`): mean window ms per depth. A
//! depth whose windows pruned drafts (`drafts` below `depth`) is flagged: its
//! mean is the price of the capped route the floor would also observe.
//!
//! usage: qwen4_mtp_g942_cost MODEL.hfq PROMPT.txt [WINDOWS_PER_DEPTH=24]
//!        [AR_STEPS=12] [MAX_SEQ=4096]
//!
//! ```text
//! HIPFIRE_QWEN4_MTP_G942_DEPTH=7 cargo run --release -p hipfire-arch-qwen4 \
//!   --features lab --example qwen4_mtp_g942_cost -- MODEL.hfq PROMPT.txt
//! ```

use hipfire_arch_qwen4::mtp_spec::{Qwen4MtpDrafter, MTP_G942_DEPTH_ENV};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::spec::MtpDrafter;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu};
use std::path::Path;
use std::time::Instant;

#[path = "../tests/common/stock_mtp_load.rs"]
mod stock_mtp_load;

/// Deepest native depth (the drafter's few-row cap, `FEW_ROW_ROUTE_ROWS - 1`).
const MAX_DEPTH: usize = 7;
/// Leading rounds (one window per depth each) not recorded: kernel/JIT warm-up.
const WARMUP_ROUNDS: usize = 2;

#[derive(Default)]
struct Cell {
    ms: Vec<f64>,
    drafts: Vec<usize>,
    accepted: Vec<usize>,
    committed: Vec<usize>,
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn mean_usize(values: &[usize]) -> f64 {
    values.iter().sum::<usize>() as f64 / values.len().max(1) as f64
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.get(sorted.len() / 2).copied().unwrap_or(f64::NAN)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: qwen4_mtp_g942_cost MODEL.hfq PROMPT.txt [WINDOWS_PER_DEPTH=24] [AR_STEPS=12] [MAX_SEQ=4096]".into());
    }
    let model = Path::new(&args[1]);
    let prompt = std::fs::read_to_string(&args[2]).map_err(|e| format!("{}: {e}", args[2]))?;
    let parse = |index: usize, default: usize| -> Result<usize, String> {
        args.get(index).map_or(Ok(default), |value| {
            value
                .parse()
                .map_err(|_| format!("argument {index} ({value:?}) is not a count"))
        })
    };
    let per_depth = parse(3, 24)?.max(1);
    let ar_steps = parse(4, 12)?.max(1);
    let max_seq = parse(5, 4096)?;

    // The depth of each window is its draft budget: the forced-depth knob must
    // sit at its maximum so `min(knob, budget)` is the budget.
    let depth_env = std::env::var(MTP_G942_DEPTH_ENV).ok();
    let want_depth = MAX_DEPTH.to_string();
    if depth_env.as_deref().map(str::trim) != Some(want_depth.as_str()) {
        return Err(format!(
            "{MTP_G942_DEPTH_ENV} must be {MAX_DEPTH} (got {depth_env:?}): the example sets each window's depth through its draft budget"
        ));
    }
    if std::env::var_os("HIPFIRE_G942_TRUNK_GEMV").is_some() {
        return Err("HIPFIRE_G942_TRUNK_GEMV must be unset: the planner refuses native MTP under it (its GEMV decode is not bitwise the multi-row verify)".into());
    }

    if !stock_mtp_load::has_stock_mtp_sidecar(model) {
        return Err(format!(
            "{} has no `<model>.mtp` head sidecar next to it",
            model.display()
        ));
    }
    let tokenizer = {
        let hfq = HfqFile::open(model).map_err(|e| e.to_string())?;
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| e.to_string())?
    };
    hipfire_loader::prepare_host_memory_for(model);
    let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
    if gpu.arch != "gfx942" {
        return Err(format!(
            "this tool prices the gfx942 table; the loaded GPU is {}",
            gpu.arch
        ));
    }
    let state_request = std::env::var("HIPFIRE_STATE_QUANT").unwrap_or_default();
    let mut bundle =
        stock_mtp_load::load_stock_mtp_bundle(&mut gpu, model, max_seq, &state_request)?;
    let vocab = bundle.config.vocab_size;
    let eos = bundle.config.eos_token_id;
    let tokens = tokenizer.encode(&prompt);
    // Every round runs one window per depth, committing at most depth + 1.
    let round_rows = (MAX_DEPTH + 1) * (MAX_DEPTH + 2) / 2;
    let worst_rows = tokens.len() + (per_depth + WARMUP_ROUNDS) * round_rows;
    if tokens.is_empty() || worst_rows > max_seq {
        return Err(format!(
            "prompt of {} tokens plus up to {} window rows needs {worst_rows} positions, over MAX_SEQ {max_seq}: raise MAX_SEQ or lower WINDOWS_PER_DEPTH",
            tokens.len(),
            (per_depth + WARMUP_ROUNDS) * round_rows
        ));
    }

    let mut drafter = Qwen4MtpDrafter::new(MAX_DEPTH, max_seq, None);
    let mut seed = drafter
        .mtp_prefill(&mut gpu, &mut bundle, &tokens, &tokens, 0, false, &|| false)?
        .into_first_token()?;
    let mut ids = vec![seed];
    let context_start = bundle.state.position;
    let mut cells: Vec<Cell> = (0..=MAX_DEPTH).map(|_| Cell::default()).collect();
    for round in 0..per_depth + WARMUP_ROUNDS {
        for (depth, cell) in cells.iter_mut().enumerate() {
            let position = bundle.state.position;
            if position + depth + 2 > max_seq {
                return Err(format!(
                    "context {position} reached MAX_SEQ {max_seq} in round {round}: raise MAX_SEQ or lower WINDOWS_PER_DEPTH"
                ));
            }
            gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
            let started = Instant::now();
            let window = drafter.mtp_step(
                &mut gpu,
                &mut bundle,
                position,
                seed,
                &ids,
                depth,
                eos,
                None,
            )?;
            gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
            let ms = started.elapsed().as_secs_f64() * 1e3;
            if round >= WARMUP_ROUNDS {
                cell.ms.push(ms);
                cell.drafts.push(window.drafts_generated);
                cell.accepted.push(window.accepted);
                cell.committed.push(window.committed.len());
            }
            seed = *window.committed.last().ok_or("empty MTP window")?;
            ids.extend_from_slice(&window.committed);
        }
    }
    let context_end = bundle.state.position;
    let stats = drafter.request_stats();
    println!("{stats:?}");

    // Plain AR on the same state: `seed` is the pending token at the live
    // position; each step forwards it and returns the greedy next one.
    let logits = gpu.zeros(&[vocab], DType::F32).map_err(|e| e.to_string())?;
    let mut ar_ms = Vec::with_capacity(ar_steps);
    let mut token = seed;
    for _ in 0..ar_steps {
        gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
        let started = Instant::now();
        token = bundle
            .forward_token_or_argmax(&mut gpu, Some(token), &logits)
            .map_err(|e| e.to_string())?;
        gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
        ar_ms.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let ar = mean(&ar_ms);

    println!(
        "model {} prompt {} tokens, context {context_start}..{context_end}, {per_depth} windows per depth ({WARMUP_ROUNDS} warm-up rounds dropped), {} generated ids",
        model.display(),
        tokens.len(),
        ids.len()
    );
    println!(
        "AR token (forward_token_or_argmax): mean {ar:.2} ms, median {:.2} ms over {ar_steps} steps",
        median(&ar_ms)
    );
    println!(
        "{:>5} {:>8} {:>10} {:>10} {:>7} {:>9} {:>9} {:>10} {:>8}",
        "depth",
        "windows",
        "mean_ms",
        "median_ms",
        "drafts",
        "accepted",
        "committed",
        "ms/token",
        "vs_AR"
    );
    let mut table = Vec::with_capacity(MAX_DEPTH + 1);
    for (depth, cell) in cells.iter().enumerate() {
        let window_ms = mean(&cell.ms);
        let committed = mean_usize(&cell.committed);
        let per_token = window_ms / committed.max(f64::MIN_POSITIVE);
        let drafts = mean_usize(&cell.drafts);
        let flag = if drafts + 1e-9 < depth as f64 {
            "  <- pruned drafts"
        } else {
            ""
        };
        println!(
            "{depth:>5} {:>8} {window_ms:>10.2} {:>10.2} {drafts:>7.2} {:>9.2} {committed:>9.2} {per_token:>10.2} {:>7.2}x{flag}",
            cell.ms.len(),
            median(&cell.ms),
            mean_usize(&cell.accepted),
            per_token / ar,
        );
        table.push(format!("Some({window_ms:.2})"));
    }
    println!(
        "MTP_WINDOW_COST_MS_GFX942 = [{}] // mean window ms, context {context_start}..{context_end}, AR {ar:.2} ms",
        table.join(", ")
    );
    gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    Box::new(drafter).mtp_free(&mut gpu);
    bundle.free_gpu(&mut gpu).map_err(|e| e.to_string())
}
