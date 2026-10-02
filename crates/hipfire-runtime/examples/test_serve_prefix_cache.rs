// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! P5 GPU compose oracle: cross-session radix prefix reuse on the slot engine.
//!
//! Cells (X2):
//!   * greedy AR/MTP: warm identical prompt reuses ≥128 tokens and matches
//!     the cold continuation; a divergent suffix still reuses the prefix
//!   * sampled AR: same seed is deterministic and still reuses
//!   * grammar AR: JSON Schema mask + prefix reuse; output parses as JSON
//!   * A10 MTP cache visibility: long generation crossing page boundaries
//!     under MTP still replays identically from cache (candidate rows never
//!     become cache-visible, spec §4.6.2), plus a forced full-reject cell
//!     (HIPFIRE_FAULT_MTP_FULL_REJECT=1) proving the τ=1 repair path is
//!     exact and rejected rows stay invisible
//!   * A13 mixed load: a long cold prefill and short warm requests all make
//!     progress (spec §5.3 S3), plus a concurrent phase (4 requests, 2
//!     slots) proving WaitQueue losers complete with exact outputs
//!   * A20 lifecycle: repeated warm hits, reset → cold, re-warm after reset,
//!     with free-page plateau + post-reset leak assertions (spec §9.2)
//!   * A19 (--fault-publish): HIPFIRE_FAULT_PREFIX_PUBLISH=1 makes the first
//!     publication fail → honest miss, unchanged output (fail-closed)
//!
//! Build: `cargo run -p hipfire-runtime --example test_serve_prefix_cache --features lab -- <model.hfq>`
//! Optional: `--mtp-k N` (default 4 when a sidecar exists, else 0), `--fault-publish`.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features lab (deltanet is default)");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::mtp_head::find_mtp_sidecar;
    use hipfire_arch_qwen35::serve_engine::{EngineConfig, SlotEngine};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::serve::{Continuation, Event, RejectClass, SubmitRequest};
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc::channel;

    const PAGE: usize = 128;

    // A19 fault mode must be armed before any engine work.
    let fault_publish = std::env::args().any(|a| a == "--fault-publish");
    if fault_publish {
        std::env::set_var("HIPFIRE_FAULT_PREFIX_PUBLISH", "1");
    }

    let mut model_path = None;
    let mut mtp_k_arg: Option<usize> = None;
    let mut fault_hip_class: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--mtp-k" {
            mtp_k_arg = Some(
                args.next()
                    .expect("--mtp-k needs a value")
                    .parse()
                    .expect("mtp_k"),
            );
        } else if a == "--fault-hip" {
            fault_hip_class = Some("launch".to_string());
        } else if let Some(v) = a.strip_prefix("--fault-hip=") {
            fault_hip_class = Some(v.to_string());
        } else if !a.starts_with('-') {
            model_path = Some(a);
        }
    }
    let model_path = model_path.unwrap_or_else(|| {
        let home = std::env::var("HOME").expect("HOME not set");
        format!("{home}/.hipfire/models/qwen3.5-4b.mq4v2.hfq")
    });
    let sidecar = find_mtp_sidecar(Path::new(&model_path));
    let mtp_k = mtp_k_arg.unwrap_or(if sidecar.is_some() { 4 } else { 0 });
    println!("=== test_serve_prefix_cache ===\nmodel: {model_path}");
    println!(
        "mtp_k={mtp_k} sidecar={}",
        sidecar
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".into())
    );
    if mtp_k > 0 && sidecar.is_none() {
        panic!("mtp_k={mtp_k} requested but no .mtp sidecar next to the trunk");
    }

    let hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    drop(hfq);

    let mut prefix = tokenizer.encode("The capital of France is Paris. ");
    let filler = tokenizer.encode("It is known for museums, food, and art. ");
    while prefix.len() < PAGE * 2 {
        prefix.extend_from_slice(&filler);
    }
    prefix.truncate(PAGE * 2);

    let italy: Vec<u32> = {
        let mut v = prefix.clone();
        v.extend(tokenizer.encode(" The capital of Italy is"));
        v
    };
    let germany: Vec<u32> = {
        let mut v = prefix.clone();
        v.extend(tokenizer.encode(" The capital of Germany is"));
        v
    };
    let json_prompt: Vec<u32> = {
        let mut v = prefix.clone();
        v.extend(
            tokenizer
                .encode(" Reply with a JSON object {\"city\": \"<name>\"} naming Italy's capital."),
        );
        v
    };
    assert!(italy.len() > PAGE * 2);
    assert_ne!(italy, germany);

    let engine = SlotEngine::spawn(EngineConfig {
        model_path: PathBuf::from(&model_path),
        n_slots: 2,
        // Pool sizing is slots x cap over 128-token pages. 2048 keeps every
        // published path (italy + germany + generated + atlantis) resident
        // alongside two active slots, so the A20 soak measures steady-state
        // boundedness instead of colliding with legitimate §4.4 eviction —
        // the pinned-path release fix (wave 8) made cached paths evictable
        // again, which a 16-page pool turned into a soak miss.
        cap_tokens: 2048,
        prefill_chunk: 256,
        host_budget_bytes: 4 * 1024 * 1024 * 1024,
        swap_dir: std::env::temp_dir().join("hipfire-prefix-cache-swap"),
        is_vl: false,
        vl_path: None,
        mtp_k,
        kv_mode_raw: String::new(),
        kv_backend: "legacy".to_string(),
        prefix_cache: true,
        prefix_cache_max_bytes: 256 * 1024 * 1024,
        max_batch_tokens: 4096,
        prefill_min_tokens: 1,
        wait_max_count: 64,
        wait_max_bytes: 256 * 1024 * 1024,
        queue_timeout_ms: 30_000,
        structured_jump_forward: false,
        dflash_draft: None,
        dflash_required: false,
    })
    .expect("SlotEngine::spawn");
    println!(
        "engine up: prefix_cache on, {}-token shared prefix",
        prefix.len()
    );

    #[derive(Clone)]
    struct RunSpec {
        temperature: f32,
        seed: u32,
        json_schema: Option<serde_json::Value>,
        max_tokens: usize,
    }
    impl RunSpec {
        fn greedy(max_tokens: usize) -> Self {
            Self {
                temperature: 0.0,
                seed: 0,
                json_schema: None,
                max_tokens,
            }
        }
    }

    fn run(engine: &SlotEngine, prompt: Vec<u32>, spec: &RunSpec) -> (usize, Vec<u32>) {
        let (tx, rx) = channel::<Event>();
        engine
            .submit(SubmitRequest {
                prompt_tokens: prompt,
                convo: Vec::new(),
                continuation: Continuation::Cold,
                max_tokens: spec.max_tokens,
                temperature: spec.temperature,
                top_p: 1.0,
                top_k: 0,
                seed: spec.seed,
                repeat_window: 0,
                repeat_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                visual_data: None,
                json_schema: spec.json_schema.clone(),
                started_in_think: false,
                think_budget: usize::MAX,
                queue_bytes: 0,
                request_tag: 1,
                reply: tx,
            })
            .expect("submit");
        let mut reused = 0usize;
        let mut tokens = Vec::new();
        while let Ok(ev) = rx.recv() {
            match ev {
                Event::Accepted { reused: r, .. } => reused = r,
                Event::Token { id } => tokens.push(id),
                Event::Rejected { reason, .. } => panic!("rejected: {reason}"),
                Event::Done { .. } => break,
            }
        }
        (reused, tokens)
    }

    let greedy = RunSpec::greedy(8);

    // ── A19 fault-publish mode: reduced cell set, fail-closed proof ────
    // The injected fault fails the FIRST publication attempt (prefill
    // chunk boundary). Required observables: the request still succeeds
    // with IDENTICAL output, the warm run is an honest miss (reused=0 —
    // nothing was published), and nothing corrupts (spec §5.4 S4: a cache
    // failure may not undo a successful request nor fake a hit).
    if fault_publish {
        println!("--- A19 fault-publish mode ---");
        let (reused_cold, toks_cold) = run(&engine, italy.clone(), &greedy);
        println!(
            "  fault cold Italy: reused={reused_cold} generated={}",
            toks_cold.len()
        );
        assert_eq!(reused_cold, 0);
        assert!(!toks_cold.is_empty());
        let (reused_warm, toks_warm) = run(&engine, italy.clone(), &greedy);
        println!("  fault warm Italy: reused={reused_warm}");
        assert_eq!(
            reused_warm, 0,
            "publish was injected-failed: the warm run must be an honest miss"
        );
        assert_eq!(
            toks_warm, toks_cold,
            "fail-closed publish must not change generation"
        );
        println!("PASS (A19)");
        return;
    }

    // ── A19 fault-hip mode: device-fault injection below the engine ────
    // HIPFIRE_FAULT_HIP=<class> makes the FIRST bridge call of that class
    // fail. Required observables (spec §5.4 S4): the affected request is
    // REJECTED with the fault reason (never a fake Done or garbage
    // tokens), the engine quarantines the failed step's state, and the
    // next identical request — fault disarmed — completes with output
    // byte-identical to the pre-fault reference (no poisoned-resource
    // reuse, no same-forward fallback execution).
    if let Some(class) = fault_hip_class {
        println!("--- A19 fault-hip mode (class={class}) ---");
        // Reference BEFORE arming, so the fault lands mid-request. Arming
        // writes the bridge's atomics directly — mutating the process
        // environment from a running engine is UB against concurrent readers.
        let (reused_ref, toks_ref) = run(&engine, italy.clone(), &greedy);
        println!(
            "  reference: reused={reused_ref} generated={}",
            toks_ref.len()
        );
        assert!(!toks_ref.is_empty());
        hip_bridge::arm_hip_fault(&class, 1);
        let (tx, rx) = channel::<Event>();
        engine
            .submit(SubmitRequest {
                prompt_tokens: italy.clone(),
                convo: Vec::new(),
                continuation: Continuation::Cold,
                max_tokens: 8,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 0,
                repeat_window: 0,
                repeat_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                visual_data: None,
                json_schema: None,
                started_in_think: false,
                think_budget: usize::MAX,
                queue_bytes: 0,
                request_tag: 2,
                reply: tx,
            })
            .expect("faulted submit");
        let mut saw_rejection = false;
        let mut saw_done = false;
        let mut emitted = 0usize;
        while let Ok(ev) = rx.recv() {
            match ev {
                Event::Rejected { reason, .. } => {
                    println!("  faulted request rejected: {reason}");
                    saw_rejection = true;
                }
                Event::Token { .. } => emitted += 1,
                Event::Done { .. } => saw_done = true,
                Event::Accepted { .. } => {}
            }
        }
        hip_bridge::arm_hip_fault(&class, 0);
        assert!(saw_rejection, "the faulted request must be typed-rejected");
        assert!(
            !saw_done,
            "a device-faulted request must never surface as a successful Done"
        );
        assert!(
            emitted == 0,
            "a device-faulted request must not emit sampled tokens as if real"
        );
        // Recovery: the same request without the fault must replay the
        // reference exactly. launch/upload faults fail inside the step
        // forward, whose handler closes state and keeps the engine alive —
        // recovery runs on the same engine. A sync fault hits the
        // device_synchronize guard, which POISONS the engine (spec §5.4 S4:
        // quarantine what cannot be proven valid) — recovery needs a fresh
        // engine, and must still match byte-for-byte.
        let (reused_rec, toks_rec) = match engine.reset() {
            Ok(()) => run(&engine, italy.clone(), &greedy),
            Err(_) => {
                println!("  engine poisoned by sync fault — recreating for recovery");
                let fresh = SlotEngine::spawn(EngineConfig {
                    model_path: PathBuf::from(&model_path),
                    n_slots: 2,
                    cap_tokens: 2048,
                    prefill_chunk: 256,
                    host_budget_bytes: 4 * 1024 * 1024 * 1024,
                    swap_dir: std::env::temp_dir().join("hipfire-prefix-cache-swap"),
                    is_vl: false,
                    vl_path: None,
                    mtp_k,
                    kv_mode_raw: String::new(),
                    kv_backend: "legacy".to_string(),
                    prefix_cache: true,
                    prefix_cache_max_bytes: 256 * 1024 * 1024,
                    max_batch_tokens: 4096,
                    prefill_min_tokens: 1,
                    wait_max_count: 64,
                    wait_max_bytes: 256 * 1024 * 1024,
                    queue_timeout_ms: 30_000,
                    structured_jump_forward: false,
                    dflash_draft: None,
                    dflash_required: false,
                })
                .expect("fresh engine after poison");
                let out = run(&fresh, italy.clone(), &greedy);
                drop(fresh);
                out
            }
        };
        println!(
            "  recovery: reused={reused_rec} generated={}",
            toks_rec.len()
        );
        assert_eq!(
            toks_rec, toks_ref,
            "post-fault recovery must match the reference exactly"
        );
        println!("PASS (A19 fault-hip {class})");
        return;
    }

    // ── A20 swap/spill mode: model-swap respawn + idle spill/restore ────
    // Spec §5.4 A20: "repeated model swap, cache on/off, idle
    // spill/restore, repeated long soak — bounded plateau after warmup,
    // correct reset, no monotonic resource leak or stale state". The soak
    // and reset halves live in the default flow below; this mode covers
    // the two missing pieces:
    //   * MODEL SWAP: shutdown + respawn must rebuild a working engine
    //     whose cache starts cold and re-warms correctly, repeatedly.
    //   * IDLE SPILL/RESTORE: an idle session evicted under slot pressure
    //     must come back through the swap-restore path with its stored
    //     prefix intact (reused >= prompt), not silently re-prefill.
    if std::env::args().any(|a| a == "--a20") {
        println!("--- A20 swap/spill mode ---");

        // Spill/restore on the live engine. Two slots: S and T fill them,
        // a third cold request forces the LRU idle session (S) out to
        // swap; a named reentry on S must restore it.
        let submit_turn = |continuation: Continuation, convo: Vec<u64>| -> (u64, usize, Vec<u32>) {
            let (tx, rx) = channel::<Event>();
            engine
                .submit(SubmitRequest {
                    prompt_tokens: italy.clone(),
                    convo,
                    continuation,
                    max_tokens: 8,
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 0,
                    seed: 0,
                    repeat_window: 0,
                    repeat_penalty: 1.0,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    min_p: 0.0,
                    visual_data: None,
                    json_schema: None,
                    started_in_think: false,
                    think_budget: usize::MAX,
                    queue_bytes: 0,
                    request_tag: 3,
                    reply: tx,
                })
                .expect("a20 submit");
            let (mut session, mut reused) = (u64::MAX, 0usize);
            let mut tokens = Vec::new();
            while let Ok(ev) = rx.recv() {
                match ev {
                    Event::Accepted {
                        session: s,
                        reused: r,
                        ..
                    } => {
                        session = s;
                        reused = r;
                    }
                    Event::Token { id } => tokens.push(id),
                    Event::Done { .. } => break,
                    Event::Rejected { reason, .. } => panic!("a20 rejected: {reason}"),
                }
            }
            assert_ne!(session, u64::MAX, "engine never accepted");
            (session, reused, tokens)
        };

        let convo_a = vec![0xA20u64];
        let convo_b = vec![0xB20u64];
        let convo_c = vec![0xC20u64];
        let (sess_a, _, toks_a) = submit_turn(Continuation::Cold, convo_a.clone());
        let (_sess_b, _, _) = submit_turn(Continuation::Cold, convo_b.clone());
        // Third cold request: both slots busy-idle -> LRU victim (sess_a)
        // spills to swap.
        let (_sess_c, _, _) = submit_turn(Continuation::Cold, convo_c.clone());
        let evictions = engine.stats().evictions;
        assert!(evictions >= 1, "slot pressure must have spilled a session");
        // Named reentry on the spilled session: restore path must bring it
        // back with its stored prefix (reused >= prompt length).
        let suffix = tokenizer.encode(" And the capital of Italy is");
        let (restored, reused_r, toks_r) = submit_turn(
            Continuation::ToolResults {
                tokens: suffix,
                session: sess_a,
            },
            convo_a.clone(),
        );
        let restores = engine.stats().restores;
        assert_eq!(restored, sess_a, "reentry must land on session A");
        assert!(
            restores >= 1,
            "the spilled session must have been restored, not re-prefilled"
        );
        assert!(
            reused_r >= italy.len(),
            "restored session must reuse its stored prefix ({reused_r} < {})",
            italy.len()
        );
        assert_eq!(
            toks_r.len(),
            toks_a.len(),
            "restored turn must generate the full budget"
        );
        println!(
            "  spill/restore: evictions={evictions} restores={restores} \
             reused={reused_r}/{}",
            italy.len()
        );

        // Model swap: shutdown + respawn, twice. Each fresh engine must
        // cold-start (reused=0) then re-warm (reused=256) — proving cache
        // state does not leak across engine lifetimes and teardown frees
        // the pool completely.
        let cfg = EngineConfig {
            model_path: PathBuf::from(&model_path),
            n_slots: 2,
            cap_tokens: 2048,
            prefill_chunk: 256,
            host_budget_bytes: 4 * 1024 * 1024 * 1024,
            swap_dir: std::env::temp_dir().join("hipfire-prefix-cache-swap"),
            is_vl: false,
            vl_path: None,
            mtp_k,
            kv_mode_raw: String::new(),
            kv_backend: "legacy".to_string(),
            prefix_cache: true,
            prefix_cache_max_bytes: 256 * 1024 * 1024,
            max_batch_tokens: 4096,
            prefill_min_tokens: 1,
            wait_max_count: 64,
            wait_max_bytes: 256 * 1024 * 1024,
            queue_timeout_ms: 30_000,
            structured_jump_forward: false,
            dflash_draft: None,
            dflash_required: false,
        };
        engine.shutdown_engine().expect("a20 shutdown");
        for cycle in 0..2 {
            let fresh = SlotEngine::spawn(cfg.clone()).expect("a20 respawn");
            let (r_cold, t_cold) = run(&fresh, italy.clone(), &greedy);
            assert_eq!(r_cold, 0, "swap cycle {cycle}: fresh engine must be cold");
            let (r_warm, t_warm) = run(&fresh, italy.clone(), &greedy);
            assert_eq!(
                r_warm, 256,
                "swap cycle {cycle}: re-warm must hit the full prefix"
            );
            assert_eq!(t_warm, t_cold, "swap cycle {cycle}: output drift");
            println!(
                "  swap cycle {cycle}: cold reused=0, warm reused={r_warm}, \
                 free_pages={}",
                fresh.stats().pool_free_pages
            );
            fresh.shutdown_engine().expect("a20 cycle shutdown");
        }
        println!("PASS (A20 swap/spill)");
        return;
    }

    let (reused_cold, toks_italy_1) = run(&engine, italy.clone(), &greedy);
    println!(
        "  greedy cold Italy: reused={reused_cold} generated={}",
        toks_italy_1.len()
    );
    assert_eq!(reused_cold, 0, "first request must be a cold miss");
    assert!(!toks_italy_1.is_empty(), "cold request produced no tokens");

    let (reused_warm, toks_italy_2) = run(&engine, italy.clone(), &greedy);
    println!(
        "  greedy warm Italy: reused={reused_warm} generated={}",
        toks_italy_2.len()
    );
    assert!(
        reused_warm >= PAGE,
        "warm identical prompt must reuse at least one full page, got {reused_warm}"
    );
    // The italy prompt is NOT page-aligned (256 + suffix), so the warm run
    // exercises SuffixRecompute: restore the page-aligned checkpoint and
    // recompute the tail. This is the cell that catches a checkpoint whose
    // recurrent state was relabeled from a later boundary (spec §4.5).
    assert_eq!(
        toks_italy_1, toks_italy_2,
        "warm identical prompt must match the cold greedy continuation"
    );

    let (reused_branch, toks_germany) = run(&engine, germany, &greedy);
    println!(
        "  greedy branch Germany: reused={reused_branch} generated={}",
        toks_germany.len()
    );
    assert!(
        reused_branch >= PAGE,
        "divergent suffix must still reuse the shared prefix pages, got {reused_branch}"
    );
    assert_ne!(
        toks_germany, toks_italy_1,
        "divergent suffix must not replay the other branch"
    );

    // Sampled AR (X2): MTP is off for temperature>0; prefix reuse still applies.
    let sampled = RunSpec {
        temperature: 0.8,
        seed: 7,
        json_schema: None,
        max_tokens: 8,
    };
    let (reused_s1, toks_s1) = run(&engine, italy.clone(), &sampled);
    let (reused_s2, toks_s2) = run(&engine, italy.clone(), &sampled);
    println!(
        "  sampled Italy: reused={reused_s1}/{reused_s2} generated={}",
        toks_s1.len()
    );
    assert!(
        reused_s1 >= PAGE && reused_s2 >= PAGE,
        "sampled AR must reuse the already-published prefix, got {reused_s1}/{reused_s2}"
    );
    assert_eq!(
        toks_s1, toks_s2,
        "same seed + same prompt must be deterministic under request-private RNG"
    );

    // Grammar AR (X2): constrained decode must reuse prefix and stay valid JSON.
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"],
        "additionalProperties": false
    });
    let grammar = RunSpec {
        temperature: 0.0,
        seed: 0,
        json_schema: Some(schema),
        max_tokens: 48,
    };
    let (reused_g, toks_g) = run(&engine, json_prompt, &grammar);
    let json_text = String::from_utf8_lossy(&tokenizer.decode_bytes(&toks_g)).into_owned();
    println!(
        "  grammar JSON: reused={reused_g} generated={} text={json_text:?}",
        toks_g.len()
    );
    assert!(
        reused_g >= PAGE,
        "grammar AR must reuse the published prefix, got {reused_g}"
    );
    let parsed: serde_json::Value = serde_json::from_str(json_text.trim())
        .unwrap_or_else(|e| panic!("grammar output is not JSON ({e}): {json_text:?}"));
    assert!(
        parsed.get("city").and_then(|v| v.as_str()).is_some(),
        "grammar JSON missing city string: {parsed}"
    );

    // ── A10: MTP cache visibility across generated-page boundaries ─────
    // A long generation (96 tokens) under MTP crosses at least one page
    // boundary during decode, exercising verify/repair writes and the
    // terminal generated-prefix publication. Spec §4.6.2: speculative
    // candidate rows never become shareable — evidenced by the warm run
    // replaying the cold generation EXACTLY.
    let long_greedy = RunSpec::greedy(96);
    let (reused_l1, toks_l1) = run(&engine, italy.clone(), &long_greedy);
    let (reused_l2, toks_l2) = run(&engine, italy.clone(), &long_greedy);
    println!(
        "  A10 long-generate: reused={reused_l1}/{reused_l2} generated={}/{}",
        toks_l1.len(),
        toks_l2.len()
    );
    assert!(
        !toks_l1.is_empty(),
        "long generation must produce tokens for this cell"
    );
    assert_eq!(
        toks_l1, toks_l2,
        "warm long generation must match cold exactly — no candidate row may be cache-visible"
    );

    // ── A10 forced full-reject (τ=1 repair + cache visibility) ──────────
    // `arm_mtp_full_reject` makes every verify cycle reject all
    // candidates, so the generation advances one trunk-argmax token per
    // cycle — the same greedy sequence the accepting path produces, just
    // slower. Asserting equality against toks_l1 proves the full-reject
    // repair path (zero-length accepted prefix, DN rollback + replay) is
    // exact; the warm replay equality proves rejected candidate rows never
    // became cache-visible (spec §4.6.2, A10 "full reject").
    hipfire_arch_qwen35::mtp_spec::arm_mtp_full_reject(true);
    let (reused_fr, toks_fr) = run(&engine, italy.clone(), &long_greedy);
    hipfire_arch_qwen35::mtp_spec::arm_mtp_full_reject(false);
    println!(
        "  A10 full-reject cold: reused={reused_fr} generated={}",
        toks_fr.len()
    );
    assert_eq!(
        toks_fr, toks_l1,
        "forced full-reject MTP must produce the accepting-MTP greedy sequence"
    );
    let (reused_fw, toks_fw) = run(&engine, italy.clone(), &long_greedy);
    println!("  A10 full-reject warm: reused={reused_fw}");
    assert!(
        reused_fw >= PAGE,
        "warm replay after full-reject cycles must reuse the published prefix"
    );
    assert_eq!(
        toks_fw, toks_fr,
        "warm replay must match the full-reject generation — rejected candidates are not cache-visible"
    );

    // ── A13: mixed load progress bound (spec §5.3 S3) ──────────────────
    // A long COLD prefill (distinct prefix) submitted concurrently with
    // short warm requests: all must complete — the long prefill may not
    // starve behind the warm hits, and the warm hits may not starve behind
    // the long prefill (FairQueue rotation + prefill quantum).
    let mut atlantis: Vec<u32> =
        tokenizer.encode("The lost city of Atlantis was said to lie beyond the pillars. ");
    let atl_filler = tokenizer.encode("Sailors traded pearls and told of shining harbors there. ");
    while atlantis.len() < PAGE * 6 {
        atlantis.extend_from_slice(&atl_filler);
    }
    atlantis.truncate(PAGE * 6);
    let mut atlantis_long = atlantis.clone();
    atlantis_long.extend(tokenizer.encode(" The kings of Atlantis ruled"));
    let atl_short: Vec<u32> = {
        let mut v = atlantis.clone();
        v.extend(tokenizer.encode(" The temples of Atlantis were"));
        v
    };

    // Submit the long cold prefill first (submit is non-blocking — the
    // engine starts chunking it), then land the short warm request while
    // that prefill is still running. Both are drained afterwards.
    let (tx_long, rx_long) = channel::<Event>();
    engine
        .submit(SubmitRequest {
            prompt_tokens: atlantis_long,
            convo: Vec::new(),
            continuation: Continuation::Cold,
            max_tokens: 4,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            repeat_window: 0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.0,
            visual_data: None,
            json_schema: None,
            started_in_think: false,
            think_budget: usize::MAX,
            queue_bytes: 0,
            request_tag: 4,
            reply: tx_long,
        })
        .expect("submit long");
    std::thread::sleep(std::time::Duration::from_millis(300));
    let (reused_short, toks_short) = run(&engine, atl_short.clone(), &greedy);
    let mut reused_long = 0usize;
    let mut toks_long: Vec<u32> = Vec::new();
    while let Ok(ev) = rx_long.recv() {
        match ev {
            Event::Accepted { reused: r, .. } => reused_long = r,
            Event::Token { id } => toks_long.push(id),
            Event::Rejected { reason, .. } => panic!("long request rejected: {reason}"),
            Event::Done { .. } => break,
        }
    }
    println!(
        "  A13 mixed: long reused={reused_long} generated={}; short reused={reused_short} generated={}",
        toks_long.len(),
        toks_short.len()
    );
    assert!(
        !toks_long.is_empty(),
        "the long cold prefill must complete (no starvation)"
    );
    assert!(
        !toks_short.is_empty(),
        "the short warm request must complete (no starvation)"
    );
    assert_eq!(
        reused_long, 0,
        "the distinct long prefix must be a cold miss"
    );

    // ── A13 concurrent adversarial phase ────────────────────────────────
    // Warm hits, a warm miss-then-hit branch, and one small fresh cold
    // request all submitted back-to-back against two slots: losers must
    // park in the WaitQueue and still complete (bounded waiting, spec
    // §5.3 S3), with per-request outputs exact — reorder must not
    // cross-contaminate greedy generations.
    let small_cold: Vec<u32> = {
        let mut v = tokenizer.encode("Tiny fresh cold prompt for the concurrency cell. ");
        v.extend(tokenizer.encode("It only needs a page or two of KV."));
        v
    };
    // Spec §5.4 A13 asks for *mixed samplers* and *per-request wait-bound*
    // assertions on top of the exactness check: a sampled request and a
    // penalized request join the two greedy warm hits, each on its own
    // convo domain, and every request must individually complete inside
    // the wait bound — a parked loser that never drains is a starvation
    // bug the aggregate wall-clock would hide.
    const WAIT_BOUND: std::time::Duration = std::time::Duration::from_secs(120);
    struct Adversarial {
        prompt: Vec<u32>,
        convo: Vec<u64>,
        temperature: f32,
        seed: u32,
        repeat_penalty: f32,
        expect: Option<Vec<u32>>,
    }
    let specs: Vec<Adversarial> = vec![
        Adversarial {
            prompt: italy.clone(),
            convo: vec![0xA13_1],
            temperature: 0.0,
            seed: 0,
            repeat_penalty: 1.0,
            expect: Some(toks_italy_1.clone()),
        },
        Adversarial {
            prompt: italy.clone(),
            convo: vec![0xA13_2],
            temperature: 0.0,
            seed: 0,
            repeat_penalty: 1.0,
            expect: Some(toks_italy_1.clone()),
        },
        Adversarial {
            prompt: atl_short.clone(),
            convo: vec![0xA13_3],
            temperature: 0.8,
            seed: 7,
            repeat_penalty: 1.0,
            expect: None,
        },
        Adversarial {
            prompt: small_cold,
            convo: vec![0xA13_4],
            temperature: 0.0,
            seed: 0,
            repeat_penalty: 1.15,
            expect: None,
        },
    ];
    let t0 = std::time::Instant::now();
    let rxs: Vec<std::sync::mpsc::Receiver<Event>> = specs
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let (tx, rx) = channel::<Event>();
            engine
                .submit(SubmitRequest {
                    prompt_tokens: s.prompt.clone(),
                    convo: s.convo.clone(),
                    continuation: Continuation::Cold,
                    max_tokens: 8,
                    temperature: s.temperature,
                    top_p: 1.0,
                    top_k: 0,
                    seed: s.seed,
                    repeat_window: 0,
                    repeat_penalty: s.repeat_penalty,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    min_p: 0.0,
                    visual_data: None,
                    json_schema: None,
                    started_in_think: false,
                    think_budget: usize::MAX,
                    queue_bytes: 0,
                    request_tag: 0xA13_000 + i as u64,
                    reply: tx,
                })
                .expect("concurrent submit");
            rx
        })
        .collect();
    let handles: Vec<_> = rxs
        .into_iter()
        .zip(specs)
        .map(|(rx, s)| {
            std::thread::spawn(move || {
                let start = std::time::Instant::now();
                let mut reused = 0usize;
                let mut tokens = Vec::new();
                while let Ok(ev) = rx.recv() {
                    match ev {
                        Event::Accepted { reused: r, .. } => reused = r,
                        Event::Token { id } => tokens.push(id),
                        Event::Rejected { reason, .. } => {
                            panic!("concurrent request rejected: {reason}")
                        }
                        Event::Done { .. } => break,
                    }
                }
                (reused, tokens, s.expect, start.elapsed())
            })
        })
        .collect();
    for (i, h) in handles.into_iter().enumerate() {
        let (reused, tokens, expect, elapsed) = h.join().expect("concurrent thread");
        assert!(
            !tokens.is_empty(),
            "concurrent request {i} produced no tokens"
        );
        assert!(
            elapsed < WAIT_BOUND,
            "request {i} exceeded the per-request wait bound ({elapsed:?} >= {WAIT_BOUND:?})"
        );
        if let Some(expect) = expect {
            assert_eq!(
                tokens, expect,
                "concurrent warm request {i} must replay the exact reference tokens"
            );
        }
        let _ = reused;
    }
    println!(
        "  A13 concurrent: 4 requests (2 greedy + sampled + penalized, \
         distinct convos) all exact in {:?}",
        t0.elapsed()
    );

    // A20: repeated warm hits stay bounded, then reset forces a cold miss.
    // Pool telemetry (spec §9.2 / A20): identical request cycles at steady
    // state must not walk free pages downward — the page-level leak
    // signature.
    let mut soak_free: Vec<usize> = Vec::new();
    for i in 0..4 {
        let (r, toks) = run(&engine, italy.clone(), &greedy);
        soak_free.push(engine.stats().pool_free_pages);
        println!(
            "  soak[{i}]: reused={r} generated={} free_pages={}",
            toks.len(),
            soak_free[i]
        );
        assert!(r >= PAGE, "soak request {i} lost prefix reuse ({r})");
        assert_eq!(toks, toks_italy_1, "soak request {i} drifted from greedy");
    }
    assert!(
        soak_free[3] + 1 >= soak_free[0],
        "free pages declined across identical soak cycles: {:?} (leak)",
        soak_free
    );
    let pre_reset_free = soak_free[3];
    engine.reset().expect("reset after idle");
    let (reused_after_reset, toks_after_reset) = run(&engine, italy.clone(), &greedy);
    println!("  after reset: reused={reused_after_reset}");
    assert_eq!(
        reused_after_reset, 0,
        "reset must drop the radix (allocation_epoch bump); got reused={reused_after_reset}"
    );
    assert_eq!(
        toks_after_reset, toks_italy_1,
        "post-reset cold generation must still match the original greedy run"
    );
    // Post-reset the pool must be back at-or-above the steady-state level:
    // the radix cache leases were released (fix 11) and sessions closed.
    let post_reset_free = engine.stats().pool_free_pages;
    println!("  post-reset free_pages={post_reset_free} (pre-reset {pre_reset_free})");
    assert!(
        post_reset_free >= pre_reset_free,
        "reset leaked pages: post-reset free {post_reset_free} < pre-reset {pre_reset_free}"
    );

    // A20 re-warm: publications resume after reset — the fresh cold run
    // above republished, so the next warm run reuses again and replays.
    let (reused_rewarm, toks_rewarm) = run(&engine, italy.clone(), &greedy);
    println!("  re-warm after reset: reused={reused_rewarm}");
    assert!(
        reused_rewarm >= PAGE,
        "publications must resume after reset, got reused={reused_rewarm}"
    );
    assert_eq!(toks_rewarm, toks_italy_1);

    let stats = engine.stats();
    println!(
        "  stats: admitted={} reused_tokens={} prefix_hits={}",
        stats.admitted, stats.reused_tokens, stats.prefix_hits
    );
    assert!(
        stats.reused_tokens >= PAGE * 2,
        "engine must account reused tokens from warm requests"
    );
    println!("PASS");
}
