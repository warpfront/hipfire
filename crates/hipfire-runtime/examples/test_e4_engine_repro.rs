// Temporary audit probe: submit the E4 enum/const schema request directly
// to SlotEngine and watch events + per-token mask decisions.
#![cfg(feature = "deltanet")]

fn main() {
    use hipfire_arch_qwen35::serve_engine::{EngineConfig, SlotEngine};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::serve::{Continuation, Event, RejectClass, SubmitRequest};
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc::channel;
    use std::time::Duration;

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| { eprintln!("usage: test_e4_engine_repro <model.hfq>"); std::process::exit(2) });
    println!("=== E4 engine repro === model={model_path}");

    let hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    drop(hfq);

    // --think: leave the think span OPEN (daemon semantics for
    // enable_thinking=true) so the model generates reasoning then </think>.
    let think = std::env::args().any(|a| a == "--think");
    let rendered = if think {
        "<|im_start|>user\nPick a city and its rank as JSON.<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n"
    } else {
        "<|im_start|>user\nPick a city and its rank as JSON.<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    };
    let prompt_tokens = tok.encode(rendered);

    let schema_path = std::env::args()
        .skip(2)
        .find(|a| !a.starts_with('-') && !a.parse::<usize>().is_ok());
    let max_tok = std::env::args()
        .skip(2)
        .find_map(|a| a.parse::<usize>().ok());
    let schema: serde_json::Value = if let Some(p) = schema_path {
        serde_json::from_str(&std::fs::read_to_string(&p).expect("schema file")).expect("schema")
    } else {
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "enum": ["Paris", "London", "Rome"]},
                "rank": {"type": "integer", "enum": [1, 2, 3]},
                "fixed": {"const": "kappa"}
            },
            "required": ["city", "rank", "fixed"],
            "additionalProperties": false
        })
    };

    let engine = SlotEngine::spawn(EngineConfig {
        model_path: PathBuf::from(&model_path),
        n_slots: 1,
        cap_tokens: 8192,
        prefill_chunk: 256,
        host_budget_bytes: 4 * 1024 * 1024 * 1024,
        swap_dir: std::env::temp_dir().join("hipfire-e4-repro-swap"),
        is_vl: false,
        vl_path: None,
        mtp_k: 0, // isolate from MTP entirely
        kv_mode_raw: String::new(),
        kv_backend: "legacy".to_string(),
        prefix_cache: false,
        prefix_cache_max_bytes: 0,
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

    let (reply_tx, reply_rx) = channel();
    let req = SubmitRequest {
        prompt_tokens,
        convo: Vec::new(),
        continuation: Continuation::Cold,
        max_tokens: max_tok.unwrap_or(96),
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
        json_schema: Some(schema),
        started_in_think: think,
        think_budget: usize::MAX,
        queue_bytes: 0,
        request_tag: 1,
        reply: reply_tx,
    };
    engine.submit(req).expect("submit");

    let mut text = String::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(600);
    loop {
        if std::time::Instant::now() > deadline {
            println!("HARD TIMEOUT");
            break;
        }
        match reply_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Event::Accepted {
                reused, prefill, ..
            }) => {
                println!("Accepted reused={reused} prefill={prefill}");
            }
            Ok(Event::Token { id }) => text.push_str(&tok.decode(&[id])),
            Ok(Event::Done { reason, generated }) => {
                println!("Done reason={reason:?} generated={generated}");
                println!("OUTPUT: {:?}", text);
                break;
            }
            Ok(Event::Rejected { reason, .. }) => {
                println!("Rejected: {reason}");
                println!("OUTPUT-SO-FAR: {:?}", text);
                break;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                print!(".");
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                println!("CHANNEL CLOSED — engine thread died");
                break;
            }
        }
    }
    let _ = engine.shutdown_engine();
}
