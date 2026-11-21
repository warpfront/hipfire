// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Multi-slot KV-storage oracle: the fixed-slab route
//! (`multi_slot_paged=false`) vs the shared paged pool with the VMM physical
//! backend (`multi_slot_paged=true, multi_slot_paged_backend=vmm`), on one
//! GPU, in one process.
//!
//! `kv_backend=vmm` is no longer a separate per-slot-owner mode on this
//! engine: it means the shared paged pool and is refused unless
//! `multi_slot_paged` is set, so the storage choice is expressed by
//! `multi_slot_paged` + `multi_slot_paged_backend` (see `docs/SERVE.md`).
//!
//! Asserts the properties the paged route is allowed to claim
//! (`docs/plans/2026-10-22-vmm-kv-multislot-engine.md` §5, §11):
//!
//! * **Isolation (I9)** — on the paged route a request's FIRST generated token
//!   comes from its own prefill, whose arithmetic is composition-independent
//!   (per-slot singleton prefill). A concurrent lane must therefore emit the
//!   same first token as that prompt run alone. Cross-slot KV contamination
//!   (a row addressing the wrong owner) changes it immediately.
//! * **Composition determinism (I9)** — repeating a run with the same batch
//!   composition reproduces every id, on both storage routes.
//! * **Graph coverage** — with `HIPFIRE_SLOTS_DECODE_GRAPH=1`, the slab route
//!   must have replayed a captured decode graph (`graph_replays > 0`). The
//!   shared paged pool's decode step is eager on purpose
//!   (`forward_slots.rs` skips capture when `pool.is_paged()`), so its
//!   counters are printed for information and not asserted.
//!
//! It deliberately does **not** assert lane-for-lane or cross-route equality:
//! projection/attention arithmetic is row-count dependent on both routes, and
//! a lane's row count changes as other lanes join or leave. The slab route
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

    /// KV-storage route under test: fixed slabs, or the shared paged pool
    /// with a chosen physical backend.
    #[derive(Clone, Copy)]
    struct Route {
        label: &'static str,
        paged: bool,
        backend: &'static str,
    }

    /// One engine: serial runs, then two identical bursts (one submission per
    /// slot, all at once), then a consuming shutdown so the next route can load.
    fn run_engine(
        model_path: &str,
        route: Route,
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
            swap_disk_bytes: 0,
            is_vl: false,
            vl_path: None,
            vision_residency: hipfire_arch_qwen35_vl::qwen35_vl::TowerResidency::Streamed,
            mtp_k: 0,
            mtp_mode: hipfire_config::MtpMode::Off,
            kv_mode_raw: "q8".to_string(),
            kv_backend: "legacy".to_string(),
            multi_slot_paged: route.paged,
            multi_slot_paged_pages: 0,
            multi_slot_paged_backend: route.backend.to_string(),
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
        .unwrap_or_else(|e| panic!("SlotEngine::spawn({}): {e}", route.label));

        let serial: Vec<Vec<u32>> = prompts.iter().map(|p| ask(&engine, p)).collect();
        let serial_repeat: Vec<Vec<u32>> = prompts.iter().map(|p| ask(&engine, p)).collect();
        let burst_ids = burst(&engine, prompts);
        let burst_repeat = burst(&engine, prompts);
        let stats = engine.stats();
        engine
            .shutdown_engine()
            .unwrap_or_else(|e| panic!("shutdown({}): {e}", route.label));
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
    let slab = run_engine(
        &model_path,
        Route {
            label: "slab",
            paged: false,
            backend: "arena",
        },
        &prompts,
        draft.as_deref(),
    );
    let paged = run_engine(
        &model_path,
        Route {
            label: "paged-vmm",
            paged: true,
            backend: "vmm",
        },
        &prompts,
        draft.as_deref(),
    );

    // Composition determinism: the same serial run and the same burst
    // reproduce every id on both storage routes.
    for (name, run) in [("slab", &slab), ("paged", &paged)] {
        assert_eq!(
            run.serial, run.serial_repeat,
            "{name}: repeating a serial run changed its ids"
        );
        assert_eq!(
            run.burst, run.burst_repeat,
            "{name}: repeating the same batch composition changed its ids"
        );
    }
    // Isolation: a paged lane's first token comes from its own singleton
    // prefill, so concurrency must not change it.
    for (i, _prompt) in prompts.iter().enumerate() {
        assert_eq!(
            paged.burst[i][0], paged.serial[i][0],
            "paged: prompt {i} first token changed under concurrency — a lane's \
             prefill saw another lane's KV"
        );
    }
    // Informational: how often the two storage routes agree lane-for-lane at
    // the same burst. Differing is allowed (row-count-dependent arithmetic).
    let cross: Vec<bool> = (0..prompts.len())
        .map(|i| paged.burst[i] == slab.burst[i])
        .collect();

    // Graph coverage: the flag is read once per process when the Gpu starts.
    // The shared paged pool decodes eagerly by design (`forward_slots.rs`
    // skips capture when `pool.is_paged()`), so only the slab route owes a
    // capture/replay; the paged counters are printed below for information.
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
            "paged": { "serial": paged.serial, "burst": paged.burst },
            "graph": {
                "slab_captures": slab.stats.graph_captures,
                "slab_replays": slab.stats.graph_replays,
                "paged_captures": paged.stats.graph_captures,
                "paged_replays": paged.stats.graph_replays,
            },
        });
        std::fs::write(&path, serde_json::to_vec(&dump).expect("serialize dump"))
            .unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!("dumped ids to {path}");
    }
    println!(
        "slab: captures={} replays={} | paged: captures={} replays={}",
        slab.stats.graph_captures,
        slab.stats.graph_replays,
        paged.stats.graph_captures,
        paged.stats.graph_replays
    );
    println!("paged/slab burst lane match (informational): {cross:?}");
    println!("PASS: paged first-token isolation + composition determinism on both routes");
}
