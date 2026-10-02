// Minimal A13 concurrent-adversarial repro: submit N requests to a 2-slot
// engine, log every event per request, per-request recv timeout so a wedge
// is attributable to the engine (no events) not the test (no timeout).
// Client timeout 90s >> engine queue_timeout_ms 30s so a queue-timeout
// Rejected is distinguishable from a true no-event deadlock.
//
// Usage: a13_repro <model> <n_requests> [--mtp-k K]
#[cfg(not(feature = "lab"))]
fn main() {
    eprintln!("build with --features lab");
}

#[cfg(feature = "lab")]
fn main() {
    use hipfire_arch_qwen35::serve_engine::{EngineConfig, SlotEngine};
    use hipfire_runtime::serve::{Continuation, Event, RejectClass, SubmitRequest};
    use std::path::PathBuf;
    use std::sync::mpsc::channel;
    use std::time::{Duration, Instant};

    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("model path").clone();
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let mtp_k: usize = args
        .iter()
        .position(|a| a == "--mtp-k")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let engine = SlotEngine::spawn(EngineConfig {
        model_path: PathBuf::from(&model_path),
        n_slots: 2,
        cap_tokens: 2048,
        prefill_chunk: 256,
        host_budget_bytes: 4 * 1024 * 1024 * 1024,
        swap_dir: std::env::temp_dir().join("hipfire-a13-repro-swap"),
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
    .expect("spawn");
    println!("engine up: 2 slots, submitting {n} concurrent requests");

    // Distinct prompts so each request is a genuine cold admit.
    let rxs: Vec<_> = (0..n)
        .map(|i| {
            let (tx, rx) = channel::<Event>();
            let prompt: Vec<u32> = (0..64).map(|t| (1000 + i * 100 + t) as u32).collect();
            engine
                .submit(SubmitRequest {
                    prompt_tokens: prompt,
                    convo: vec![0xA13_000 + i as u64],
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
                    think_budget: 0,
                    queue_bytes: 0,
                    request_tag: 0xA13_000 + i as u64,
                    reply: tx,
                })
                .expect("submit");
            rx
        })
        .collect();

    let handles: Vec<_> = rxs
        .into_iter()
        .enumerate()
        .map(|(i, rx)| {
            std::thread::spawn(move || {
                let start = Instant::now();
                let mut log = Vec::new();
                loop {
                    match rx.recv_timeout(Duration::from_secs(90)) {
                        Ok(Event::Accepted { reused, .. }) => {
                            log.push(format!("Accepted(reused={reused})"))
                        }
                        Ok(Event::Token { id }) => log.push(format!("Tok({id})")),
                        Ok(Event::Rejected { reason, .. }) => {
                            log.push(format!("Rejected({reason})"));
                            break;
                        }
                        Ok(Event::Done { .. }) => {
                            log.push("Done".into());
                            break;
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            log.push("TIMEOUT-90s-no-event".into());
                            break;
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            log.push("DISCONNECTED".into());
                            break;
                        }
                    }
                }
                (i, start.elapsed(), log)
            })
        })
        .collect();

    for h in handles {
        let (i, el, log) = h.join().unwrap();
        println!("req{i} ({el:?}): {}", log.join(" "));
    }
    println!("done");
}
