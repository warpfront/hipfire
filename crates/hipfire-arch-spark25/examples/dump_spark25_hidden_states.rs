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
//!   dump_spark25_hidden_states --hfq <model.hfq> --parity [--parity-tol <f32>]
//!   Tokens arg may be a JSON array like "[1,2,3]" or a path to a file containing JSON array.
//!   Optional --logits-out writes headerless LE f32 [n_pos, vocab] matching oracle body layout.
//!   --parity runs serial-vs-chunked prefill parity (no dump outputs) and prints JSON verdict.

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
    let mut parity = false;
    let mut parity_tol: f32 = 1e-4;
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
            "--parity" => {
                parity = true;
                i += 1;
            }
            "--parity-tol" => {
                parity_tol = argv[i + 1]
                    .parse::<f32>()
                    .expect("--parity-tol requires a float");
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!("usage: dump_spark25_hidden_states --hfq <model.hfq> --tokens <json|[json_file]> --out <native.hfhs> [--logits-out <path>]");
                eprintln!("   or: dump_spark25_hidden_states --hfq <model.hfq> --input-ids <json_file> --out <native.hfhs> [--logits-out <path>]");
                eprintln!("  --logits-out  headerless LE f32 [n_pos, vocab] next-token logits");
                eprintln!("   or: dump_spark25_hidden_states --hfq <model.hfq> --parity [--parity-tol <f32>]");
                eprintln!(
                    "  --parity  serial-vs-chunked prefill parity on the same quantized weights;"
                );
                eprintln!(
                    "            compares last-row per-layer hidden + logits, n_tokens, KV bytes,"
                );
                eprintln!("            prefix extension, decode continuation, and a cancel/recovery probe;");
                eprintln!("            prints JSON to stdout, exits 0 on pass / 1 on failure");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    let hfq_path = hfq_path.expect("--hfq required");
    if parity {
        std::process::exit(run_serial_vs_chunked_parity(&hfq_path, parity_tol));
    }
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

/// Serial-vs-chunked prefill parity for `--parity`.
///
/// Compares the production chunked helpers (`prefill_chunked_cancellable` /
/// `prefill_chunked_capture`) against the serial `decode_step` baseline on the
/// SAME loaded quantized weights — no golden hashes. Probe shapes cover ragged
/// chunk tails (65/70/127 @ chunk 64) and SWA window boundaries (511/512/513),
/// plus a prefix-extension probe, a decode-continuation probe, and a
/// cancel/reset/recovery probe. Prints one JSON object to stdout and returns
/// the process exit code: 0 on pass, 1 on any mismatch or kernel failure.
#[cfg(feature = "deltanet")]
fn run_serial_vs_chunked_parity(hfq_path: &std::path::Path, tol: f32) -> i32 {
    use hipfire_arch_spark25::config::Spark25Config;
    use hipfire_arch_spark25::forward::{
        decode_step, decode_step_capture, prefill_chunked_cancellable, prefill_chunked_capture,
        prefill_from, SparkPrefillScratch, SPARK_PREFILL_CHUNK,
    };
    use hipfire_arch_spark25::spark25::{Spark25State, Spark25Weights};

    fn max_abs(v: &[f32]) -> f32 {
        v.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
    }
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()))
    }
    fn argmax(v: &[f32]) -> usize {
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        best
    }
    fn download_kv_bytes(
        gpu: &mut rdna_compute::Gpu,
        state: &Spark25State,
    ) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        for cache in [&state.kv_sliding, &state.kv_full] {
            for t in cache
                .k_gpu
                .iter()
                .chain(cache.v_gpu.iter())
                .chain(cache.k_scales.iter())
                .chain(cache.v_scales.iter())
            {
                let mut b = vec![0u8; t.buf.size()];
                gpu.hip
                    .memcpy_dtoh(&mut b, &t.buf)
                    .map_err(|e| format!("parity kv dtoh: {e:?}"))?;
                out.extend_from_slice(&b);
            }
        }
        Ok(out)
    }
    /// Serial baseline: `decode_step` per position, capture only at the last
    /// position (capture is download-only, so skipping it earlier is bitwise
    /// identical). Returns (per-layer last rows, last logits).
    fn serial_last_row(
        cfg: &Spark25Config,
        weights: &Spark25Weights,
        state: &mut Spark25State,
        gpu: &mut rdna_compute::Gpu,
        tokens: &[u32],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        state.reset(gpu)?;
        let n = tokens.len();
        let mut logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            if pos + 1 == n {
                let mut cap: Vec<Vec<f32>> = vec![Vec::new(); cfg.n_layers];
                logits = decode_step_capture(cfg, weights, state, gpu, tok, pos as u32, &mut cap)?;
                let rows = cap
                    .iter()
                    .map(|c| {
                        if c.len() >= cfg.dim {
                            c[c.len() - cfg.dim..].to_vec()
                        } else {
                            c.clone()
                        }
                    })
                    .collect();
                return Ok((rows, logits));
            } else {
                logits = decode_step(cfg, weights, state, gpu, tok, pos as u32)?;
            }
        }
        Ok((vec![Vec::new(); cfg.n_layers], logits))
    }
    /// Production chunk path. Does NOT reset: the caller resets for full
    /// probes or leaves state for extension probes. Frees scratch on every
    /// path, including kernel failure.
    fn chunked_last_row(
        cfg: &Spark25Config,
        weights: &Spark25Weights,
        state: &mut Spark25State,
        gpu: &mut rdna_compute::Gpu,
        tokens: &[u32],
        start_pos: u32,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        let scratch = SparkPrefillScratch::new(gpu, cfg, SPARK_PREFILL_CHUNK)?;
        let mut cap: Vec<Vec<f32>> = vec![Vec::new(); cfg.n_layers];
        let r = prefill_chunked_capture(
            cfg, weights, state, gpu, tokens, start_pos, &scratch, &mut cap,
        );
        scratch.free_gpu(gpu);
        let logits = r?;
        let rows = cap
            .iter()
            .map(|c| {
                if c.len() >= cfg.dim {
                    c[c.len() - cfg.dim..].to_vec()
                } else {
                    c.clone()
                }
            })
            .collect();
        Ok((rows, logits))
    }
    #[allow(clippy::too_many_arguments)]
    fn compare_probe(
        s_rows: &[Vec<f32>],
        c_rows: &[Vec<f32>],
        s_logits: &[f32],
        c_logits: &[f32],
        s_n: usize,
        c_n: usize,
        s_kv: &[u8],
        c_kv: &[u8],
        tol: f32,
    ) -> serde_json::Value {
        let mut layers_ok = true;
        let mut worst = 0.0f32;
        let mut worst_layer = 0usize;
        let per_layer: Vec<serde_json::Value> = s_rows
            .iter()
            .zip(c_rows.iter())
            .enumerate()
            .map(|(l, (a, b))| {
                let (d, m) = if a.len() == b.len() && !a.is_empty() {
                    (max_abs_diff(a, b), max_abs(a))
                } else {
                    (f32::MAX, 0.0)
                };
                let scaled = d / m.max(1.0);
                let ok = d <= tol;
                layers_ok &= ok;
                if d < f32::MAX && d > worst {
                    worst = d;
                    worst_layer = l;
                }
                serde_json::json!({"layer": l, "max_abs": d, "scaled": scaled, "pass": ok})
            })
            .collect();
        let logits_abs = if s_logits.len() == c_logits.len() && !s_logits.is_empty() {
            max_abs_diff(s_logits, c_logits)
        } else {
            f32::MAX
        };
        let argmax_equal = !s_logits.is_empty()
            && s_logits.len() == c_logits.len()
            && argmax(s_logits) == argmax(c_logits);
        let ntokens_equal = s_n == c_n;
        let mut kv_diff_bytes = 0usize;
        let mut kv_first_diff: Option<usize> = None;
        let kv_equal = s_kv.len() == c_kv.len() && {
            for (i, (&x, &y)) in s_kv.iter().zip(c_kv.iter()).enumerate() {
                if x != y {
                    kv_diff_bytes += 1;
                    if kv_first_diff.is_none() {
                        kv_first_diff = Some(i);
                    }
                }
            }
            kv_diff_bytes == 0
        };
        let pass = layers_ok && logits_abs <= tol && argmax_equal && ntokens_equal && kv_equal;
        serde_json::json!({
            "hidden_worst_max_abs": worst,
            "hidden_worst_layer": worst_layer,
            "per_layer": per_layer,
            "logits_max_abs": logits_abs,
            "argmax_equal": argmax_equal,
            "ntokens": [s_n, c_n],
            "ntokens_equal": ntokens_equal,
            "kv_bytes": s_kv.len(),
            "kv_diff_bytes": kv_diff_bytes,
            "kv_first_diff": kv_first_diff,
            "kv_equal": kv_equal,
            "pass": pass,
        })
    }

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(hfq_path).expect("open hfq");
    let cfg = Spark25Config::from_hfq(&hfq).expect("config");
    eprintln!(
        "parity: dim={} layers={} vocab={} SWA={} arch={} tol={tol}",
        cfg.dim, cfg.n_layers, cfg.vocab_size, cfg.sliding_window, arch,
    );
    let weights = Spark25Weights::load(&hfq, &cfg, &mut gpu).expect("weights");
    // Largest probe is 513 tokens plus one decode-continuation step.
    let mut state = Spark25State::new(&cfg, &mut gpu, 600).expect("state");
    if state.max_seq < 530 {
        println!(
            "{}",
            serde_json::json!({"pass": false, "error": format!("state max_seq {} < 530; cannot cover SWA 513 probe", state.max_seq)})
        );
        return 1;
    }
    // Deterministic synthetic ids in [1, vocab).
    let synth = |n: usize| -> Vec<u32> {
        (0..n)
            .map(|i| {
                1 + ((i as u64).wrapping_mul(2654435761).wrapping_add(11)
                    % (cfg.vocab_size as u64 - 1)) as u32
            })
            .collect()
    };
    let mut all_pass = true;
    let mut shape_results = Vec::new();
    // Ragged chunk tails (64+1, 64+6, 64+63) and SWA window boundaries.
    for &n in &[65usize, 70, 127, 511, 512, 513] {
        eprintln!("parity: shape {n}");
        let tokens = synth(n);
        let serial: (Vec<Vec<f32>>, Vec<f32>, usize, Vec<u8>, Option<String>) =
            match (|| -> Result<_, String> {
                let (rows, logits) =
                    serial_last_row(&cfg, &weights, &mut state, &mut gpu, &tokens)?;
                let kv = download_kv_bytes(&mut gpu, &state)?;
                Ok((rows, logits, state.n_tokens, kv))
            })() {
                Ok((rows, logits, n, kv)) => (rows, logits, n, kv, None),
                Err(e) => (
                    Vec::new(),
                    Vec::new(),
                    0,
                    Vec::new(),
                    Some(format!("serial: {e}")),
                ),
            };
        // The serial side resets internally; reset explicitly before the
        // chunked side so both see identical initials.
        let chunked: (Vec<Vec<f32>>, Vec<f32>, usize, Vec<u8>, Option<String>) =
            match (|| -> Result<_, String> {
                state.reset(&mut gpu)?;
                let (rows, logits) =
                    chunked_last_row(&cfg, &weights, &mut state, &mut gpu, &tokens, 0)?;
                let kv = download_kv_bytes(&mut gpu, &state)?;
                Ok((rows, logits, state.n_tokens, kv))
            })() {
                Ok((rows, logits, n, kv)) => (rows, logits, n, kv, None),
                Err(e) => (
                    Vec::new(),
                    Vec::new(),
                    0,
                    Vec::new(),
                    Some(format!("chunked: {e}")),
                ),
            };
        let mut entry = serde_json::json!({"n": n});
        match (&serial.4, &chunked.4) {
            (None, None) => {
                let cmp = compare_probe(
                    &serial.0, &chunked.0, &serial.1, &chunked.1, serial.2, chunked.2, &serial.3,
                    &chunked.3, tol,
                );
                if cmp.get("pass") != Some(&serde_json::Value::Bool(true)) {
                    all_pass = false;
                }
                entry["compare"] = cmp;
            }
            _ => {
                all_pass = false;
                entry["serial_error"] = serde_json::Value::from(serial.4.clone());
                entry["chunked_error"] = serde_json::Value::from(chunked.4.clone());
                entry["pass"] = serde_json::Value::Bool(false);
            }
        }
        shape_results.push(entry);
    }
    // Prefix extension: prefill 40, extend 30 more. Serial extends with the
    // per-token loop (capturing the last row); chunked extends with a second
    // production call at start_pos=40. Also checks `prefill_from` agrees.
    eprintln!("parity: prefix extension 40+30");
    let ext_tokens = synth(70);
    let extension = (|| -> Result<serde_json::Value, String> {
        state.reset(&mut gpu).map_err(|e| format!("reset: {e}"))?;
        let mut s_logits = Vec::new();
        let mut s_rows: Vec<Vec<f32>> = Vec::new();
        for (pos, &tok) in ext_tokens.iter().enumerate() {
            if pos + 1 == ext_tokens.len() {
                let mut cap: Vec<Vec<f32>> = vec![Vec::new(); cfg.n_layers];
                s_logits = decode_step_capture(
                    &cfg, &weights, &mut state, &mut gpu, tok, pos as u32, &mut cap,
                )?;
                s_rows = cap
                    .iter()
                    .map(|c| c[c.len().saturating_sub(cfg.dim)..].to_vec())
                    .collect();
            } else {
                s_logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos as u32)?;
            }
        }
        let s_n = state.n_tokens;
        let s_kv = download_kv_bytes(&mut gpu, &state)?;
        // `prefill_from` must agree with the manual loop.
        state.reset(&mut gpu)?;
        for (p, &t) in ext_tokens[..40].iter().enumerate() {
            decode_step(&cfg, &weights, &mut state, &mut gpu, t, p as u32)?;
        }
        let pf_logits = prefill_from(&cfg, &weights, &mut state, &mut gpu, &ext_tokens[40..], 40)?;
        let pf_abs = max_abs_diff(&s_logits, &pf_logits);
        // Chunked extension from a reset state.
        state.reset(&mut gpu)?;
        let scratch = SparkPrefillScratch::new(&mut gpu, &cfg, SPARK_PREFILL_CHUNK)?;
        let r0 = prefill_chunked_cancellable(
            &cfg,
            &weights,
            &mut state,
            &mut gpu,
            &ext_tokens[..40],
            0,
            &scratch,
            &|| false,
        );
        scratch.free_gpu(&mut gpu);
        r0?;
        let (c_rows, c_logits) =
            chunked_last_row(&cfg, &weights, &mut state, &mut gpu, &ext_tokens[40..], 40)?;
        let c_n = state.n_tokens;
        let c_kv = download_kv_bytes(&mut gpu, &state)?;
        let mut v = compare_probe(
            &s_rows, &c_rows, &s_logits, &c_logits, s_n, c_n, &s_kv, &c_kv, tol,
        );
        v["prefill_from_max_abs"] = serde_json::json!(pf_abs);
        v["prefill_from_agrees"] = serde_json::json!(pf_abs <= tol);
        Ok(v)
    })();
    let extension = match extension {
        Ok(v) => {
            if v.get("pass") != Some(&serde_json::Value::Bool(true))
                || v.get("prefill_from_agrees") != Some(&serde_json::Value::Bool(true))
            {
                all_pass = false;
            }
            v
        }
        Err(e) => {
            all_pass = false;
            serde_json::json!({"pass": false, "error": e})
        }
    };
    // Decode continuation: rebuild both full-127 states, step one token at
    // pos 127 on each, compare next-token logits.
    eprintln!("parity: decode continuation");
    let cont_tokens = synth(127);
    let next_tok = synth(514)[513];
    let continuation = (|| -> Result<serde_json::Value, String> {
        let (_, _) = serial_last_row(&cfg, &weights, &mut state, &mut gpu, &cont_tokens)?;
        let s_c = decode_step(&cfg, &weights, &mut state, &mut gpu, next_tok, 127)?;
        state.reset(&mut gpu)?;
        let _ = chunked_last_row(&cfg, &weights, &mut state, &mut gpu, &cont_tokens, 0)?;
        let c_c = decode_step(&cfg, &weights, &mut state, &mut gpu, next_tok, 127)?;
        let d = max_abs_diff(&s_c, &c_c);
        let arg_eq = argmax(&s_c) == argmax(&c_c);
        Ok(serde_json::json!({
            "logits_max_abs": d,
            "argmax_equal": arg_eq,
            "pass": d <= tol && arg_eq,
        }))
    })();
    let continuation = match continuation {
        Ok(v) => {
            if v.get("pass") != Some(&serde_json::Value::Bool(true)) {
                all_pass = false;
            }
            v
        }
        Err(e) => {
            all_pass = false;
            serde_json::json!({"pass": false, "error": e})
        }
    };
    // Cancel/reset/recovery: an immediately-aborting prefill must fail with
    // an abort error (never a silent slow fallback), and a reset must leave
    // a state that serves a fresh serial prefill exactly.
    eprintln!("parity: cancel probe");
    let cancel_probe = (|| -> Result<serde_json::Value, String> {
        state.reset(&mut gpu)?;
        let tokens70 = synth(70);
        let scratch = SparkPrefillScratch::new(&mut gpu, &cfg, SPARK_PREFILL_CHUNK)?;
        let r = prefill_chunked_cancellable(
            &cfg,
            &weights,
            &mut state,
            &mut gpu,
            &tokens70,
            0,
            &scratch,
            &|| true,
        );
        scratch.free_gpu(&mut gpu);
        let aborted = matches!(&r, Err(e) if e.to_lowercase().contains("abort"));
        state.reset(&mut gpu)?;
        let recovered = if aborted {
            let t3 = synth(3);
            let mut ok = true;
            for (pos, &tok) in t3.iter().enumerate() {
                if decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos as u32).is_err() {
                    ok = false;
                    break;
                }
            }
            ok && state.n_tokens == 3
        } else {
            false
        };
        Ok(serde_json::json!({
            "aborted": aborted,
            "abort_error": match &r {
                Err(e) => serde_json::Value::from(e.clone()),
                Ok(_) => serde_json::Value::Null,
            },
            "recovered": recovered,
            "pass": aborted && recovered,
        }))
    })();
    let cancel_probe = match cancel_probe {
        Ok(v) => {
            if v.get("pass") != Some(&serde_json::Value::Bool(true)) {
                all_pass = false;
            }
            v
        }
        Err(e) => {
            all_pass = false;
            serde_json::json!({"pass": false, "error": e})
        }
    };
    println!(
        "{}",
        serde_json::json!({
            "mode": "spark25 serial-vs-chunked prefill parity",
            "arch": arch,
            "chunk": SPARK_PREFILL_CHUNK,
            "tolerance": tol,
            "shapes": shape_results,
            "extension": extension,
            "decode_continuation": continuation,
            "cancel_probe": cancel_probe,
            "pass": all_pass,
        })
    );
    if all_pass {
        0
    } else {
        1
    }
}
