// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Multi-slot KV-storage oracle: `kv_backend=legacy` (arena) vs
//! `kv_backend=vmm` (per-slot VMM reservations), on one GPU, in one process.
//!
//! Asserts the properties the VMM route is allowed to claim
//! (`docs/plans/2026-10-22-vmm-kv-multislot-engine.md` §5, §11):
//!
//! * **Isolation (I9)** — on the VMM route a request's FIRST generated token
//!   comes from its own prefill, whose arithmetic is composition-independent
//!   (per-slot singleton prefill). A concurrent lane must therefore emit the
//!   same first token as that prompt run alone. Cross-slot KV contamination
//!   (a row addressing the wrong owner) changes it immediately.
//! * **Composition determinism (I9)** — repeating a run with the same batch
//!   composition reproduces every id, on both storage routes.
//! * **Graph coverage** — with `HIPFIRE_SLOTS_DECODE_GRAPH=1`, the VMM route
//!   must have replayed a captured decode graph (`graph_replays > 0`).
//!
//! It deliberately does **not** assert lane-for-lane or cross-route equality:
//! projection/attention arithmetic is row-count dependent on both routes, and
//! a lane's row count changes as other lanes join or leave. The arena route
//! demonstrates this even between two identical concurrent prompts (one may
//! prefill in a mixed step, the other alone through the single-slot path), so
//! the cross-route match is printed for information, not asserted.
//!
//! # Run
//!
//! ```text
//! cargo run --release -p hipfire-runtime --features lab --example test_slots_vmm_oracle \
//!   -- ~/models/hipfire/Qwen/Qwen3.5-4B-MQ4/qwen3.5-4b.mq4
//! # graph arm
//! HIPFIRE_SLOTS_DECODE_GRAPH=1 cargo run --release -p hipfire-runtime --features lab \
//!   --example test_slots_vmm_oracle -- <model>
//! ```

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet,arch-qwen35 (the `lab` feature)");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::serve_engine::{EngineConfig, SlotEngine};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::serve::{Continuation, EngineStats, Event, SubmitRequest};
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc::channel;

    const N_SLOTS: usize = 2;
    const CAP_TOKENS: usize = 1024;
    const MAX_TOKENS: usize = 24;
    const PROMPTS: [&str; 2] = [
        "Write a Python function named add that returns the sum of two integers.",
        "Explain in two sentences how quicksort picks a pivot.",
    ];

    struct Run {
        serial: Vec<Vec<u32>>,
        serial_repeat: Vec<Vec<u32>>,
        burst: Vec<Vec<u32>>,
        burst_repeat: Vec<Vec<u32>>,
        stats: EngineStats,
    }

    /// One engine: serial runs, then two identical bursts (one submission per
    /// slot, all at once), then a consuming shutdown so the next route can load.
    fn run_engine(
        model_path: &str,
        backend: &str,
        prompts: &[Vec<u32>],
        draft: Option<&Path>,
    ) -> Run {
        let engine = SlotEngine::spawn(EngineConfig {
            model_path: PathBuf::from(model_path),
            n_slots: prompts.len().max(N_SLOTS),
            cap_tokens: CAP_TOKENS,
            prefill_chunk: 512,
            host_budget_bytes: 4 * 1024 * 1024 * 1024,
            swap_dir: std::env::temp_dir().join("hipfire-vmm-oracle-swap"),
            is_vl: false,
            vl_path: None,
            mtp_k: 0,
            mtp_mode: hipfire_config::MtpMode::Off,
            kv_mode_raw: "q8".to_string(),
            kv_backend: backend.to_string(),
            prefix_cache: false,
            prefix_cache_max_bytes: 0,
            max_batch_tokens: 4096,
            prefill_min_tokens: 1,
            wait_max_count: 64,
            wait_max_bytes: 256 * 1024 * 1024,
            queue_timeout_ms: 30_000,
            structured_jump_forward: false,
            dflash_draft: draft.map(Path::to_path_buf),
            dflash_required: draft.is_some(),
        })
        .unwrap_or_else(|e| panic!("SlotEngine::spawn({backend}): {e}"));

        let serial: Vec<Vec<u32>> = prompts.iter().map(|p| ask(&engine, p)).collect();
        let serial_repeat: Vec<Vec<u32>> = prompts.iter().map(|p| ask(&engine, p)).collect();
        let burst_ids = burst(&engine, prompts);
        let burst_repeat = burst(&engine, prompts);
        let stats = engine.stats();
        engine
            .shutdown_engine()
            .unwrap_or_else(|e| panic!("shutdown({backend}): {e}"));
        Run {
            serial,
            serial_repeat,
            burst: burst_ids,
            burst_repeat,
            stats,
        }
    }

    /// Submit every prompt at once (one request per slot) and collect ids.
    fn burst(engine: &SlotEngine, prompts: &[Vec<u32>]) -> Vec<Vec<u32>> {
        let mut handles = Vec::new();
        for prompt in prompts {
            let (tx, rx) = channel::<Event>();
            engine
                .submit(SubmitRequest {
                    prompt_tokens: prompt.clone(),
                    convo: Vec::new(),
                    continuation: Continuation::Cold,
                    max_tokens: MAX_TOKENS,
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 0,
                    seed: 0,
                    repeat_window: 0,
                    repeat_penalty: 1.0,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    min_p: 0.0,
                    logprobs_top_k: None,
                    visual_data: None,
                    json_schema: None,
                    started_in_think: false,
                    think_budget: 0,
                    request_tag: 0,
                    queue_bytes: 0,
                    reply: tx,
                })
                .expect("submit");
            handles.push(std::thread::spawn(move || collect(rx)));
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("client thread"))
            .collect()
    }

    fn ask(engine: &SlotEngine, prompt: &[u32]) -> Vec<u32> {
        let (tx, rx) = channel::<Event>();
        engine
            .submit(SubmitRequest {
                prompt_tokens: prompt.to_vec(),
                convo: Vec::new(),
                continuation: Continuation::Cold,
                max_tokens: MAX_TOKENS,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 0,
                repeat_window: 0,
                repeat_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                logprobs_top_k: None,
                visual_data: None,
                json_schema: None,
                started_in_think: false,
                think_budget: 0,
                request_tag: 0,
                queue_bytes: 0,
                reply: tx,
            })
            .expect("submit");
        collect(rx)
    }

    fn collect(rx: std::sync::mpsc::Receiver<Event>) -> Vec<u32> {
        let mut ids = Vec::new();
        while let Ok(ev) = rx.recv() {
            match ev {
                Event::Token { id, .. } => ids.push(id),
                Event::Rejected { reason, .. } => panic!("request rejected: {reason}"),
                Event::Done { .. } => break,
                _ => {}
            }
        }
        assert!(!ids.is_empty(), "request produced no tokens");
        ids
    }

    let model_path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("HIPFIRE_SLOTS_VMM_MODEL").ok());
    let Some(model_path) = model_path else {
        eprintln!(
            "skip: pass a model path or set HIPFIRE_SLOTS_VMM_MODEL \
             (e.g. ~/models/hipfire/Qwen/Qwen3.5-4B-MQ4/qwen3.5-4b.mq4)"
        );
        return;
    };
    let draft = std::env::var_os("HIPFIRE_SLOTS_ORACLE_DRAFT").map(PathBuf::from);
    if let Some(d) = &draft {
        println!("draft: {}", d.display());
    }
    println!("=== test_slots_vmm_oracle ===\nmodel: {model_path}");

    let hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    drop(hfq);
    let prompts: Vec<Vec<u32>> = PROMPTS.iter().map(|p| tokenizer.encode(p)).collect();

    // One engine at a time: each spawn loads its own weights, and the
    // consuming shutdown releases VRAM (and invalidates caches) before the
    // next route loads.
    let arena = run_engine(&model_path, "legacy", &prompts, draft.as_deref());
    let vmm = run_engine(&model_path, "vmm", &prompts, draft.as_deref());

    // Composition determinism: the same serial run and the same burst
    // reproduce every id on both storage routes.
    for (name, run) in [("arena", &arena), ("vmm", &vmm)] {
        assert_eq!(
            run.serial, run.serial_repeat,
            "{name}: repeating a serial run changed its ids"
        );
        assert_eq!(
            run.burst, run.burst_repeat,
            "{name}: repeating the same batch composition changed its ids"
        );
    }
    // Isolation: a VMM lane's first token comes from its own singleton
    // prefill, so concurrency must not change it.
    for (i, _prompt) in prompts.iter().enumerate() {
        assert_eq!(
            vmm.burst[i][0], vmm.serial[i][0],
            "vmm: prompt {i} first token changed under concurrency — a lane's \
             prefill saw another lane's KV"
        );
    }
    // Informational: how often the two storage routes agree lane-for-lane at
    // the same burst. Differing is allowed (row-count-dependent arithmetic).
    let cross: Vec<bool> = (0..prompts.len())
        .map(|i| vmm.burst[i] == arena.burst[i])
        .collect();

    // Graph coverage: the flag is read once per process when the Gpu starts.
    if std::env::var("HIPFIRE_SLOTS_DECODE_GRAPH").as_deref() == Ok("1") {
        assert!(
            vmm.stats.graph_replays > 0,
            "HIPFIRE_SLOTS_DECODE_GRAPH=1 but the VMM route replayed no graph \
             (captures {}, replays {})",
            vmm.stats.graph_captures,
            vmm.stats.graph_replays
        );
        assert!(
            arena.stats.graph_replays > 0,
            "HIPFIRE_SLOTS_DECODE_GRAPH=1 but the arena route replayed no graph"
        );
    }

    // Optional JSON dump so two runs (graph off/on) can be diffed externally.
    if let Ok(path) = std::env::var("HIPFIRE_SLOTS_ORACLE_DUMP") {
        let dump = serde_json::json!({
            "arena": { "serial": arena.serial, "burst": arena.burst },
            "vmm": { "serial": vmm.serial, "burst": vmm.burst },
            "graph": {
                "arena_captures": arena.stats.graph_captures,
                "arena_replays": arena.stats.graph_replays,
                "vmm_captures": vmm.stats.graph_captures,
                "vmm_replays": vmm.stats.graph_replays,
            },
        });
        std::fs::write(&path, serde_json::to_vec(&dump).expect("serialize dump"))
            .unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!("dumped ids to {path}");
    }
    println!(
        "arena: captures={} replays={} | vmm: captures={} replays={}",
        arena.stats.graph_captures,
        arena.stats.graph_replays,
        vmm.stats.graph_captures,
        vmm.stats.graph_replays
    );
    println!("vmm/arena burst lane match (informational): {cross:?}");
    println!("PASS: vmm first-token isolation + composition determinism on both routes");
}
