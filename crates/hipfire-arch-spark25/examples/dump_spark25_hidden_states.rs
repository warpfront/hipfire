// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Dump per-layer hidden states from Spark-X2.5 dense forward for offline
//! comparison against the HF transformers oracle (arch bring-up validation).
//!
//! Reads token ids from JSON array or HFKLDR-like file, runs
//! `decode_step_capture` per position with all layers captured, and writes a
//! single HFHS binary of post-layer (pre-final-norm) hidden states —
//! byte-format-compatible with `scripts/dump_hf_hidden_states.py` +
//! `scripts/compare_hidden_states.py`.
//!
//! Format: magic 8B = b"HFHS\0\0\0\0", n_layers u32, n_pos u32, hidden_dim u32, reserved u32, then f32 body [n_layers, n_pos, hidden] row-major.
//!
//! Usage:
//!   dump_spark25_hidden_states --hfq <model.hfq> --tokens <json|txt> --out <native.hfhs> [--input-ids <path>] [--logits-out <path>]
//!   Tokens arg may be a JSON array like "[1,2,3]" or a path to a file containing JSON array.
//!   Optional --logits-out writes headerless LE f32 [n_pos, vocab] matching oracle body layout.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_spark25::config::Spark25Config;
    use hipfire_arch_spark25::forward::decode_step_capture;
    use hipfire_runtime::hfq::HfqFile;
    use std::io::{BufWriter, Write};
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut hfq_path: Option<PathBuf> = None;
    let mut tokens_arg: Option<String> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut input_ids_path: Option<PathBuf> = None;
    let mut logits_out_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--hfq" | "--model" => {
                hfq_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--tokens" => {
                tokens_arg = Some(argv[i + 1].clone());
                i += 2;
            }
            "--input-ids" => {
                input_ids_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--out" | "--output" => {
                out_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--logits-out" => {
                logits_out_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!("usage: dump_spark25_hidden_states --hfq <model.hfq> --tokens <json|[json_file]> --out <native.hfhs> [--logits-out <path>]");
                eprintln!("   or: dump_spark25_hidden_states --hfq <model.hfq> --input-ids <json_file> --out <native.hfhs> [--logits-out <path>]");
                eprintln!("  --logits-out  headerless LE f32 [n_pos, vocab] next-token logits");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    let hfq_path = hfq_path.expect("--hfq required");
    let out_path = out_path.expect("--out required");

    // Resolve tokens
    let tokens: Vec<u32> = if let Some(p) = input_ids_path {
        let s = std::fs::read_to_string(&p).expect("read input_ids");
        serde_json::from_str(&s).expect("parse input_ids JSON")
    } else if let Some(arg) = tokens_arg {
        // Try as file first, then as inline JSON
        if std::path::Path::new(&arg).exists() {
            let s = std::fs::read_to_string(&arg).expect("read tokens file");
            serde_json::from_str(s.trim()).expect("parse tokens JSON file")
        } else {
            serde_json::from_str(arg.trim()).expect("parse --tokens JSON array")
        }
    } else {
        eprintln!("--tokens or --input-ids required");
        std::process::exit(2);
    };

    if tokens.is_empty() {
        eprintln!("no tokens");
        std::process::exit(2);
    }
    eprintln!("tokens: {} ids", tokens.len());

    // Load model
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(&hfq_path).expect("open hfq");
    eprintln!("hfq arch_id={}", hfq.arch_id);
    if hfq.arch_id != 16 {
        eprintln!(
            "warning: expected arch_id 16 for spark2_5, got {}",
            hfq.arch_id
        );
    }
    let cfg = Spark25Config::from_hfq(&hfq).expect("config");
    eprintln!(
        "config: dim={} hidden={} layers={} heads={} kv={} head_dim={} vocab={} SWA={}",
        cfg.dim,
        cfg.hidden_dim,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.vocab_size,
        cfg.sliding_window
    );
    let weights =
        hipfire_arch_spark25::spark25::Spark25Weights::load(&hfq, &cfg, &mut gpu).expect("weights");
    let mut state =
        hipfire_arch_spark25::spark25::Spark25State::new(&cfg, &mut gpu, tokens.len() + 16)
            .expect("state");

    // Per-token forward with capture
    let n_layers = cfg.n_layers;
    let hidden = cfg.dim;
    let vocab = cfg.vocab_size;
    let n_pos = tokens.len();
    let mut capture: Vec<Vec<f32>> = vec![Vec::with_capacity(n_pos * hidden); n_layers];

    let mut logits_out = logits_out_path.as_ref().map(|p| {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).expect("create logits-out parent dir");
            }
        }
        BufWriter::new(std::fs::File::create(p).expect("create logits-out"))
    });

    let t0 = std::time::Instant::now();
    for (pos, &tok) in tokens.iter().enumerate() {
        if pos % 64 == 0 {
            eprintln!("  pos {}/{} tok={}", pos, n_pos, tok);
        }
        let logits = decode_step_capture(
            &cfg,
            &weights,
            &mut state,
            &mut gpu,
            tok,
            pos as u32,
            &mut capture,
        )
        .expect("decode capture");
        if let Some(w) = logits_out.as_mut() {
            assert_eq!(
                logits.len(),
                vocab,
                "logits len {} != vocab {} at pos {}",
                logits.len(),
                vocab,
                pos
            );
            for &v in &logits {
                w.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }
    eprintln!("forward complete in {:.2}s", t0.elapsed().as_secs_f64());

    if let Some(mut w) = logits_out.take() {
        w.flush().unwrap();
        let p = logits_out_path.as_ref().unwrap();
        eprintln!(
            "wrote logits {} ({} pos, vocab {})",
            p.display(),
            n_pos,
            vocab
        );
    }

    // Write HFHS (hidden states format)
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out = BufWriter::new(std::fs::File::create(&out_path).expect("create out"));
    out.write_all(b"HFHS\0\0\0\0").unwrap();
    out.write_all(&(n_layers as u32).to_le_bytes()).unwrap();
    out.write_all(&(n_pos as u32).to_le_bytes()).unwrap();
    out.write_all(&(hidden as u32).to_le_bytes()).unwrap();
    out.write_all(&0u32.to_le_bytes()).unwrap();
    for l in 0..n_layers {
        assert_eq!(
            capture[l].len(),
            n_pos * hidden,
            "layer {l} capture len mismatch"
        );
        for &v in &capture[l] {
            out.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    out.flush().unwrap();
    eprintln!(
        "wrote {} ({} layers, {} pos, hidden {})",
        out_path.display(),
        n_layers,
        n_pos,
        hidden
    );
    eprintln!("compare with: python scripts/compare_hidden_states.py --hf .scratch/spark25-validation/oracle-fp32.hfhs --hipfire {}", out_path.display());
}
