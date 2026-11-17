// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Multi-slot slab-route oracle: composition determinism on the fixed
//! per-slot slab (`kv_backend=legacy`, no shared page pool), on one GPU.
//!
//! * **Composition determinism (I9)**: repeating a run with the same batch
//!   composition reproduces every id.
//! * **Graph coverage**: with `HIPFIRE_SLOTS_DECODE_GRAPH=1`, the route must
//!   have replayed a captured decode graph (`graph_replays > 0`).
//!
//! It deliberately does **not** assert lane-for-lane equality between serial
//! and concurrent runs: projection/attention arithmetic is row-count
//! dependent, and a lane's row count changes as other lanes join or leave.
//!
//! # Run
//!
//! ```text
//! cargo run --release -p hipfire-runtime --features lab --example test_slots_slab_oracle \
//!   -- ~/models/hipfire/Qwen/Qwen3.5-4B-MQ4/qwen3.5-4b.mq4
//! # graph arm
//! HIPFIRE_SLOTS_DECODE_GRAPH=1 cargo run --release -p hipfire-runtime --features lab \
//!   --example test_slots_slab_oracle -- <model>
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
    /// slot, all at once), then a consuming shutdown.
    fn run_engine(model_path: &str, prompts: &[Vec<u32>], draft: Option<&Path>) -> Run {
        let engine = SlotEngine::spawn(EngineConfig {
            model_path: PathBuf::from(model_path),
            n_slots: prompts.len().max(N_SLOTS),
            cap_tokens: CAP_TOKENS,
            prefill_chunk: 512,
            host_budget_bytes: 4 * 1024 * 1024 * 1024,
            swap_dir: std::env::temp_dir().join("hipfire-slab-oracle-swap"),
            swap_disk_bytes: 1 << 30,
            is_vl: false,
            vl_path: None,
            vision_residency: hipfire_arch_qwen35_vl::qwen35_vl::TowerResidency::Streamed,
            mtp_k: 0,
            mtp_mode: hipfire_config::MtpMode::Off,
            kv_mode_raw: "q8".to_string(),
            kv_backend: "legacy".to_string(),
            // The fixed per-slot slab, not the shared paged pool.
            multi_slot_paged: false,
            multi_slot_paged_pages: 0,
            multi_slot_paged_backend: "arena".to_string(),
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
        .unwrap_or_else(|e| panic!("SlotEngine::spawn: {e}"));

        let serial: Vec<Vec<u32>> = prompts.iter().map(|p| ask(&engine, p)).collect();
        let serial_repeat: Vec<Vec<u32>> = prompts.iter().map(|p| ask(&engine, p)).collect();
        let burst_ids = burst(&engine, prompts);
        let burst_repeat = burst(&engine, prompts);
        let stats = engine.stats();
        engine
            .shutdown_engine()
            .unwrap_or_else(|e| panic!("shutdown: {e}"));
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
        .or_else(|| std::env::var("HIPFIRE_SLOTS_ORACLE_MODEL").ok());
    let Some(model_path) = model_path else {
        eprintln!(
            "skip: pass a model path or set HIPFIRE_SLOTS_ORACLE_MODEL \
             (e.g. ~/models/hipfire/Qwen/Qwen3.5-4B-MQ4/qwen3.5-4b.mq4)"
        );
        return;
    };
    let draft = std::env::var_os("HIPFIRE_SLOTS_ORACLE_DRAFT").map(PathBuf::from);
    if let Some(d) = &draft {
        println!("draft: {}", d.display());
    }
    println!("=== test_slots_slab_oracle ===\nmodel: {model_path}");

    let hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    drop(hfq);
    let prompts: Vec<Vec<u32>> = PROMPTS.iter().map(|p| tokenizer.encode(p)).collect();

    let slab = run_engine(&model_path, &prompts, draft.as_deref());

    // Composition determinism: the same serial run and the same burst
    // reproduce every id.
    assert_eq!(
        slab.serial, slab.serial_repeat,
        "repeating a serial run changed its ids"
    );
    assert_eq!(
        slab.burst, slab.burst_repeat,
        "repeating the same batch composition changed its ids"
    );

    // Graph coverage: the flag is read once per process when the Gpu starts.
    if std::env::var("HIPFIRE_SLOTS_DECODE_GRAPH").as_deref() == Ok("1") {
        assert!(
            slab.stats.graph_replays > 0,
            "HIPFIRE_SLOTS_DECODE_GRAPH=1 but the slab route replayed no graph \
             (captures {}, replays {})",
            slab.stats.graph_captures,
            slab.stats.graph_replays
        );
    }

    // Optional JSON dump so two runs (graph off/on) can be diffed externally.
    if let Ok(path) = std::env::var("HIPFIRE_SLOTS_ORACLE_DUMP") {
        let dump = serde_json::json!({
            "slab": { "serial": slab.serial, "burst": slab.burst },
            "graph": {
                "captures": slab.stats.graph_captures,
                "replays": slab.stats.graph_replays,
            },
        });
        std::fs::write(&path, serde_json::to_vec(&dump).expect("serialize dump"))
            .unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!("dumped ids to {path}");
    }
    println!(
        "slab: captures={} replays={}",
        slab.stats.graph_captures, slab.stats.graph_replays
    );
    println!("PASS: composition determinism on the slab route");
}
