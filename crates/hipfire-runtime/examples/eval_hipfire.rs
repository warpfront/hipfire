// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! eval_hipfire — KLD eval for hipfire quant variants against a BF16 reference.
//!
//! Loads a hipfire model, reads the slice (or pre-tokenized tokens), reads
//! the BF16 reference in hipfire β format (HFKLDR), runs forward inference
//! chunk-by-chunk over the matched eval tokens, computes per-token KLD via
//! a top-K-of-reference approximation, bins per-sequence, emits HFKSEQ
//! output that `kld_reduce.py` aggregates.
//!
//! Usage:
//!   eval_hipfire --model <path-to-hfq-model> \
//!                --ref   <path-to-hipfire-β-ref> \
//!                --output <path-to-output.kldseq> \
//!                [--variant <name>=auto-from-model-path] \
//!                [--arch <name>=auto-from-gpu] \
//!                [--kv-mode <mode>=asym3] \
//!                [--scoring-mode <per-token|prefill|prefill-whole|prefill-prompt>=prefill]
//!                [--scored-max-batch <N>]
//!
//! Scoring modes (per `docs/plans/issue-113-quant-quality-eval.md` §5), with
//! the kernel admission each one runs (qwen35). Hidden rows are captured
//! either through `forward_prefill_batch`'s verify capture
//! (`per_token_hidden_out`), which selects the speculative-verify kernels
//! (`DispatchWorkload::SpeculativeVerify`; the GDN chunk scan is never
//! admitted — the sequential GDN recurrence runs), or through
//! `forward_prefill_batch_capture_hidden`, which is a plain prompt prefill
//! plus output rows (`DispatchWorkload::Standard`, GDN chunk scan admitted
//! wherever a plain prefill admits it):
//!   prefill:   (default, canonical since 2026-05-11) forward_prefill_batch
//!              (transformer stack batched, lm_head fan-out per scored
//!              position). ~7× wall-clock vs per-token on gfx1100/gfx1151
//!              9B Q3/Q4. Requires the model's LA dtype to be in
//!              `is_batchable_la`'s OK set; auto-falls-back to per-token
//!              inside `forward_prefill_batch` otherwise (e.g., MQ4-Lloyd,
//!              HFP4G32, MFP4G32 — no batched kernel yet). Split in two
//!              calls (prefix, then the scored region): the prefix call
//!              captures nothing and has plain prompt-prefill admission
//!              (`DispatchWorkload::Standard`, GDN chunk scan admissible);
//!              the scored-region call captures hidden rows through the
//!              verify capture, so it runs the speculative-verify kernels
//!              (`DispatchWorkload::SpeculativeVerify`, no GDN chunk scan).
//!   per-token: forward_scratch in a per-position loop (single-token decode
//!              kernels; no batched prefill admission, no GDN chunk scan,
//!              no verify workload). Historical baseline,
//!              retained for direct comparison against the 2026-05-08 kldseqs
//!              under `results/2026-05-08/per-seq/*__per-token.kldseq`.
//!   prefill-whole: (opt-in, qwen35 only) the whole window — prefix, scored
//!              region and the final unscored token — as ONE
//!              forward_prefill_batch call, scoring the same reference
//!              positions from the captured rows (`n_ctx/2..`). An n_ctx-512
//!              window becomes one 512-row prefill, so routes that gate on the
//!              chunk's row count see what a real prompt prefill sees; the
//!              split `prefill` mode runs 256 + 255 rows. The whole-window
//!              plan is `eval_common::whole_window_plan` (unit-tested). The
//!              existing modes' forward calls are unchanged (partly
//!              source-pinned) and their scoring math/output is preserved by
//!              extraction into `eval_common::kld_from_block`, `seq_stats`,
//!              `write_hfkseq` (fixed-fixture golden: `cargo test -p
//!              hipfire-runtime --lib eval_common`). Results
//!              are NOT comparable with split-`prefill` kldseqs: KV/state
//!              commit cadence differs, so compare like with like.
//!              Its hidden rows are captured through `forward_prefill_batch`'s
//!              verify capture, which selects the speculative-verify kernels
//!              (`DispatchWorkload::SpeculativeVerify`, e.g. the sequential
//!              GDN recurrence, never the chunk scan).
//!   prefill-prompt: (opt-in, qwen35 only) the same whole-window plan as
//!              `prefill-whole`, but captured through
//!              `forward_prefill_batch_capture_hidden`, whose chunk planning,
//!              admission (GDN chunk scan included) and dispatch workload
//!              (`DispatchWorkload::Standard`) are exactly a plain prompt
//!              prefill's. This is the mode that measures the production
//!              prompt route; not comparable with `prefill-whole` kldseqs
//!              where a route is capture-gated.
//!
//! `--scored-max-batch N` (split `prefill` mode only, qwen35, N >= 2): the
//! scored-region call goes through `forward_prefill_batch_capped` with cap N,
//! so the scored rows run as consecutive N-row verify-capture forwards (the
//! shape of an MTP K = N-1 verify) instead of one 1023-row call. The prefix
//! call is unchanged. Use it to measure routes that gate on a small row count
//! (e.g. the exact-gfx1100 2..=5-row verify GEMV); compare like with like,
//! since the KV/state commit cadence differs from the default split mode.
//! Absent (the default), the scored call is byte-for-byte the historical one.
//!
//! Output: HFKSEQ format (see kldref_format.py) — per-sequence (mean, p99)
//! KLD as fp64 pairs.
//!
//! `--per-token-output <path>` (any scoring mode, default off) additionally
//! writes HFKTOK v1: `HFKTOK\0\0`, u32 version=1, u32 n_chunk, u32
//! scored_per_chunk, u32 reserved=0, then `n_chunk * scored_per_chunk`
//! little-endian f64 per-token KLDs in scoring order — the exact values whose
//! sequential f64 sum / count is each chunk's HFKSEQ `mean_kld` (bit for bit).
//! Without the flag the kldseq bytes and behaviour are unchanged.
//!
//! Plan: docs/plans/issue-113-quant-quality-eval.md (rev-3.2).
//!
//! Multi-arch dispatch (Tier-1): mirrors `calib_sweep.rs` arch detection
//!   llama 0|1, qwen35 5|6, gemma4 13, glimmer 14, lfm2 11.
//!   Per-arch chunking mirrors calib_sweep:
//!     gemma4 uses forward::forward_prefill_batch (step program, 64-row chunks) with q8 KV
//!     glimmer chunks 192 (glimmer_prefill_chunk_size / prefill_with_capture)
//!     lfm2 uses forward_decode_batch_lfm with explicit positions, q8 KV
//!     qwen35 uses forward_prefill_batch with q8 KV at seq_len+16

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{KvCache, VMode};
    use rdna_compute::DType;
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Read};
    use std::path::PathBuf;
    use std::time::Instant;

    // -------- args --------
    struct Args {
        model: PathBuf,
        ref_path: PathBuf,
        output: PathBuf,
        kv_mode: String,
        kv_v: String,
        scoring_mode: String,
        max_chunks: Option<usize>,
        scored_max_batch: Option<usize>,
        tp: Option<usize>,
        per_token_output: Option<PathBuf>,
    }
    // Every mode's kernel admission, shared by the --scoring-mode error and
    // --help text (kept in sync with the module docs above).
    const SCORING_MODE_HELP: &str = "\
  scoring modes (qwen35 admission; prefill-whole / prefill-prompt are qwen35 only):
    prefill         (default) two forward_prefill_batch calls: the prefix call is a plain
                    prompt prefill (Standard workload, GDN chunk scan admissible); the
                    scored-region call captures hidden rows through the verify capture
                    (SpeculativeVerify kernels, no GDN chunk scan)
    per-token       forward_scratch per position: single-token decode kernels, no batched
                    prefill admission or GDN chunk scan
    prefill-whole   whole window in ONE forward_prefill_batch call with verify capture
                    (SpeculativeVerify kernels, no GDN chunk scan)
    prefill-prompt  whole window in ONE forward_prefill_batch_capture_hidden call: plain
                    production prompt-prefill admission (Standard workload, GDN chunk
                    scan admitted where a plain prefill admits it)";
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut ref_path: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut kv_mode = "asym3".to_string();
    let mut kv_v = "q8".to_string();
    let mut scoring_mode = "prefill".to_string();
    let mut max_chunks: Option<usize> = None;
    let mut scored_max_batch: Option<usize> = None;
    let mut tp: Option<usize> = None;
    let mut per_token_output: Option<PathBuf> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--ref" => {
                ref_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--kv-mode" => {
                let v = argv[i + 1].clone();
                if !matches!(
                    v.as_str(),
                    "q8" | "fp8"
                        | "bf16"
                        | "asym2"
                        | "asym3"
                        | "asym4"
                        | "fwht2"
                        | "fwht3"
                        | "fwht4"
                        | "f32"
                        | "f16"
                ) {
                    eprintln!("--kv-mode must be one of: q8 fp8 bf16 asym2 asym3 asym4 fwht2 fwht3 fwht4 f32 f16 (got {v})");
                    std::process::exit(1);
                }
                kv_mode = v;
                i += 2;
            }
            "--kv-v" => {
                let v = argv[i + 1].clone();
                if !matches!(v.as_str(), "q8" | "lloyd2" | "lloyd3" | "lloyd4") {
                    eprintln!("--kv-v must be one of: q8 lloyd2 lloyd3 lloyd4 (got {v})");
                    std::process::exit(1);
                }
                kv_v = v;
                i += 2;
            }
            "--scoring-mode" => {
                let v = argv[i + 1].clone();
                if !matches!(
                    v.as_str(),
                    "per-token" | "prefill" | "prefill-whole" | "prefill-prompt"
                ) {
                    eprintln!(
                        "--scoring-mode must be one of: per-token prefill prefill-whole prefill-prompt (got {v})\n{SCORING_MODE_HELP}"
                    );
                    std::process::exit(1);
                }
                scoring_mode = v;
                i += 2;
            }
            "--max-chunks" => {
                max_chunks = Some(argv[i + 1].parse().expect("--max-chunks must be integer"));
                i += 2;
            }
            "--per-token-output" => {
                per_token_output = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--tp" => {
                tp = Some(argv[i + 1].parse().expect("--tp must be integer"));
                i += 2;
            }
            "--scored-max-batch" => {
                let n: usize = argv[i + 1]
                    .parse()
                    .expect("--scored-max-batch must be integer");
                if n < 2 {
                    eprintln!("--scored-max-batch must be >= 2 (got {n})");
                    std::process::exit(1);
                }
                scored_max_batch = Some(n);
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!("Usage: eval_hipfire --model <path> --ref <path> --output <path> [--kv-mode asym3] [--kv-v q8] [--scoring-mode prefill|prefill-whole|prefill-prompt|per-token] [--max-chunks N] [--scored-max-batch N] [--tp N (qwen35 dense, prefill, q8/q8 KV)]\n{SCORING_MODE_HELP}");
                eprintln!("       [--per-token-output <path.ktok>]  (HFKTOK v1 per-token KLD f64s, any scoring mode; kldseq bytes unchanged)");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let args = Args {
        model: model.expect("--model required"),
        ref_path: ref_path.expect("--ref required"),
        output: output.expect("--output required"),
        kv_mode,
        kv_v,
        scoring_mode,
        max_chunks,
        scored_max_batch,
        tp,
        per_token_output,
    };
    if args.scored_max_batch.is_some() && args.scoring_mode != "prefill" {
        eprintln!("--scored-max-batch applies to --scoring-mode prefill only");
        std::process::exit(1);
    }
    if let Some(tp) = args.tp {
        if !(2..=8).contains(&tp) {
            eprintln!("--tp must be in 2..=8 (got {tp})");
            std::process::exit(1);
        }
        if args.scoring_mode != "prefill" || args.scored_max_batch.is_some() {
            eprintln!("--tp applies to --scoring-mode prefill only (no --scored-max-batch)");
            std::process::exit(1);
        }
        if args.kv_mode != "q8" || args.kv_v != "q8" {
            eprintln!("--tp requires --kv-mode q8 --kv-v q8 (per-rank q8 KV)");
            std::process::exit(1);
        }
        if args.model.is_dir() {
            eprintln!("--tp is not supported for safetensors/paro model directories");
            std::process::exit(1);
        }
    }

    // -------- eval-mode env vars (must precede Gpu::init / forward) --------
    // Per plan §"Eval-mode hipfire flags": force OFF for prompt normalize +
    // graph capture; record kv-mode in env for downstream tooling. Logged so
    // a user reading the run output sees the override explicitly.
    //
    // Note on HIPFIRE_GRAPH=0: byte-equality between graph=0 and graph=1
    // was verified on 2026-05-08 against this binary's forward path
    // (dense Qwen3.5-9B mq4, prefill 64 tokens, kv_mode=asym3) — sha256
    // matched, 0/248320 logits differed. The plan's force-OFF is therefore
    // a determinism *style* choice, not a correctness requirement: a
    // future contributor can safely flip this to opt-out (respect a
    // pre-existing env value) for cards where graph mode would shave
    // kernel-launch overhead. On 2026-05-08's gfx1100 baseline run the
    // card was power-capped at the kernel-throughput ceiling, so graph
    // mode wouldn't have helped — but that's hardware-specific.
    // The MoE-config drift documented in
    // hipfire-arch-qwen35/src/qwen35.rs:2906-2932 still applies and is
    // already gated by `config.num_experts == 0`, so dense models are
    // unaffected.
    // SAFETY: single-threaded init phase; no other threads observing env.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_KV_MODE", &args.kv_mode);
        std::env::set_var("HIPFIRE_KV_V", &args.kv_v);
        // For prefill scoring, pre-allocate the PrefillBatchScratch via
        // Qwen35Scratch's HIPFIRE_PREFILL_REUSE_PBS hook so the 1175 chunk
        // calls don't each pay 25-tensor alloc/free overhead. (Plan §M1.)
        if args.scoring_mode == "prefill" {
            std::env::set_var("HIPFIRE_PREFILL_REUSE_PBS", "1");
        }
        if args.scoring_mode == "prefill-whole" {
            std::env::set_var("HIPFIRE_PREFILL_REUSE_PBS", "1");
        }
        if args.scoring_mode == "prefill-prompt" {
            std::env::set_var("HIPFIRE_PREFILL_REUSE_PBS", "1");
        }
    }
    eprintln!(
        "eval_hipfire: forced HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 \
         HIPFIRE_KV_MODE={} HIPFIRE_KV_V={} scoring_mode={}",
        args.kv_mode, args.kv_v, args.scoring_mode
    );

    // -------- ref sha256 sanity (M1) --------
    hipfire_runtime::eval_common::verify_ref_sha256(&args.ref_path, "eval_hipfire");

    // -------- GPU init --------
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!(
        "eval_hipfire: arch={} model={}",
        gpu.arch,
        args.model.display()
    );

    // -------- arch detection (mirrors calib_sweep.rs:654-658) --------
    // HFQ header is authoritative for file models; for safetensors dirs we
    // probe SafetensorsSource. This is byte-identical to calib_sweep's
    // `arch = arg("--arch").unwrap_or(hfq.arch_id)` dispatch.
    // GGUF-native model (`HIPFIRE_GGUF_NATIVE=1`; refuses with `load refused:` when
    // the switch is off): the file is parsed, admitted per tensor and its bpw
    // census / G3 badge logged before any GPU weight upload.
    let gguf_model = if args.model.extension().is_some_and(|e| e == "gguf") {
        Some(
            hipfire_loader::gguf_native::GgufNativeModel::open(&args.model).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            }),
        )
    } else {
        None
    };
    let arch_id: u32 = if let Some(m) = &gguf_model {
        eprintln!("eval_hipfire: detected arch_id={} (GGUF)", m.arch_id);
        m.arch_id
    } else if args.model.is_dir() {
        let src = hipfire_runtime::safetensors_source::SafetensorsSource::open(&args.model)
            .unwrap_or_else(|e| {
                eprintln!("safetensors open {}: {e}", args.model.display());
                std::process::exit(1)
            });
        let id = src.arch_id();
        eprintln!("eval_hipfire: detected arch_id={id} (safetensors dir)");
        id
    } else {
        let hfq = HfqFile::open(&args.model).unwrap_or_else(|e| {
            eprintln!("open model {}: {e}", args.model.display());
            std::process::exit(1)
        });
        let id = hfq.arch_id;
        eprintln!("eval_hipfire: detected arch_id={id} (HFQ header)");
        id
    };
    let is_llama = matches!(arch_id, 0 | 1);
    let is_qwen35 = matches!(arch_id, 5 | 6);
    let is_gemma = arch_id == 13;
    let is_glimmer = arch_id == 14;
    let is_lfm = arch_id == 11;
    if args.tp.is_some() && !is_qwen35 {
        eprintln!(
            "eval_hipfire: --tp is implemented for qwen35 (arch_id 5|6) only (got {arch_id})"
        );
        std::process::exit(1);
    }
    if !(is_llama || is_qwen35 || is_gemma || is_glimmer || is_lfm) {
        eprintln!(
            "unsupported arch_id {arch_id}: eval_hipfire wires dense 0|1 (llama), 5|6 (qwen35), 13 (gemma4), 14 (glimmer), 11 (lfm2) — matches calib_sweep.rs:658"
        );
        std::process::exit(1);
    }

    // -------- shared KLD helper (arch-independent) --------
    // `eval_common::kld_from_block` (extracted from here; executed by
    // eval_common's legacy-output golden test). Do NOT duplicate this math per
    // arch; a divergent copy would silently make two models' numbers
    // incomparable.
    use hipfire_runtime::eval_common::kld_from_block;

    // -------- helper to read ref header (shared) --------
    // Open once; after header the file cursor sits at the token stream.
    // Each arch branch will validate vocab, read tokens, and drive scoring.
    let ref_file = File::open(&args.ref_path).expect("open ref");
    let mut ref_in = BufReader::with_capacity(8 * 1024 * 1024, ref_file);
    let mut magic = [0u8; 8];
    ref_in.read_exact(&mut magic).expect("read ref magic");
    if &magic != b"HFKLDR\0\0" {
        eprintln!("bad ref magic: {magic:?}");
        std::process::exit(2);
    }
    let mut hdr = [0u8; 24];
    ref_in.read_exact(&mut hdr).expect("read ref header");
    let version = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    let n_ctx = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let ref_n_vocab = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let n_chunk = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let top_k = u16::from_le_bytes(hdr[16..18].try_into().unwrap()) as usize;
    let _flags = u16::from_le_bytes(hdr[18..20].try_into().unwrap());
    if version != 1 {
        eprintln!("unsupported ref version {version}");
        std::process::exit(2);
    }
    let scored_per_chunk = n_ctx - 1 - n_ctx / 2;
    // Quality-only: execute the final, unscored token so the scored-region
    // forward is tile-aligned; score only the original reference positions.
    let pad_scoring_tail = std::env::var("HIPFIRE_FP8_G3_PAD_TAIL").as_deref() == Ok("1");
    // `prefill-whole` (additive; the existing modes above and below are
    // untouched): the whole window (prefix + scored region + the final token)
    // as ONE prefill call, so an n_ctx-512 reference runs 512-row prefills and
    // row-count-gated prefill routes see the full window. It always pads the
    // final token and scores the same reference positions
    // (`eval_common::whole_window_plan`).
    let whole_window =
        args.scoring_mode == "prefill-whole" || args.scoring_mode == "prefill-prompt";
    let prompt_capture = args.scoring_mode == "prefill-prompt";
    let pad_scoring_tail =
        hipfire_runtime::eval_common::scoring_tail_padded(whole_window, pad_scoring_tail);
    if pad_scoring_tail {
        eprintln!("eval_hipfire: quality-only final-token padding; scored positions unchanged");
    }
    if whole_window && !is_qwen35 {
        eprintln!(
            "eval_hipfire: --scoring-mode prefill-whole / prefill-prompt is implemented for qwen35 only"
        );
        std::process::exit(1);
    }
    let effective_n_chunk = match args.max_chunks {
        Some(m) => m.min(n_chunk),
        None => n_chunk,
    };
    if let Some(m) = args.max_chunks {
        eprintln!(
            "eval_hipfire: --max-chunks {m} → effective_n_chunk = {effective_n_chunk}/{n_chunk}"
        );
    }
    let total_scored = scored_per_chunk * effective_n_chunk;
    let per_token_block_bytes = 8 + 8 * top_k;
    eprintln!(
        "eval_hipfire: ref n_ctx={n_ctx} n_vocab={ref_n_vocab} n_chunk={n_chunk} top_k={top_k}"
    );
    eprintln!(
        "  scored/chunk={scored_per_chunk}  total_scored={total_scored}  block={per_token_block_bytes}B  arch_id={arch_id}"
    );

    // Shared output accumulators; arch branches fill them.
    let mut mean_kld_per_seq: Vec<f64> = Vec::with_capacity(effective_n_chunk);
    let mut p99_kld_per_seq: Vec<f64> = Vec::with_capacity(effective_n_chunk);
    let mut mean_nll_per_seq: Vec<f64> = Vec::with_capacity(effective_n_chunk);
    let mut block_buf = vec![0u8; per_token_block_bytes];
    // HFKTOK v1 per-token KLDs (only with --per-token-output): the very f64s
    // whose chunk mean the HFKSEQ `mean_kld` records.
    let mut ktok: Option<hipfire_runtime::eval_common::TokenKldRecorder> =
        args.per_token_output.as_ref().map(|_| {
            hipfire_runtime::eval_common::TokenKldRecorder::new(scored_per_chunk, effective_n_chunk)
        });
    let t0 = Instant::now();
    let mut total_scored_done = 0usize;
    let scoring_start = n_ctx / 2;

    // ===== Arch dispatch (mirrors calib_sweep.rs:684ff) =====
    if let (true, Some(tp)) = (is_qwen35, args.tp) {
        // ----- qwen35 dense tensor-parallel prefill scoring (`--tp N`) -----
        // Hand dense-TP route over `tp` devices: the prefix and the scored
        // region go through `forward_prefill_dense_tp_with_hidden` (the scored
        // call captures the post-output-norm hidden row of every token on
        // rank 0); the per-row lm_head + KLD scoring is the single-card code
        // on rank 0's full output weights. `forward_prefill_dense_tp_with_hidden`
        // chunks internally (`prefill_max_batch_tp`) and allocates its own
        // per-call PrefillBatchScratch, so the rank `Qwen35Scratch` only needs
        // the decode-batch width (128, as the serve loader) plus `kv_max`.
        use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
        use hipfire_runtime::multi_gpu::Gpus;
        use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};

        // A device-0 context from the single-card init above would sit beside
        // rank 0's own; the TP route owns every device.
        drop(gpu);

        let mut hfq = HfqFile::open(&args.model).expect("open model");
        let global = qwen35::config_from_hfq(&hfq).expect("read config");
        if global.num_experts != 0 {
            eprintln!(
                "eval_hipfire: --tp is dense-only (num_experts={})",
                global.num_experts
            );
            std::process::exit(1);
        }
        if ref_n_vocab != global.vocab_size {
            eprintln!(
                "vocab mismatch: ref says {ref_n_vocab}, model says {}",
                global.vocab_size
            );
            std::process::exit(2);
        }
        let shard = ShardConfig::new(tp, false, 0, ExpertAssign::Stride).unwrap_or_else(|e| {
            eprintln!("eval_hipfire: --tp {tp}: {e:?}");
            std::process::exit(1)
        });
        let layouts = qwen35::dense_tp_rank_layouts(&global, &shard).unwrap_or_else(|e| {
            eprintln!("eval_hipfire: --tp {tp}: dense TP layout: {e}");
            std::process::exit(1)
        });
        qwen35::preflight_weights_dense_tp(&hfq, &global, &shard).unwrap_or_else(|e| {
            eprintln!("eval_hipfire: --tp {tp}: dense TP preflight: {e}");
            std::process::exit(1)
        });
        let configs: Vec<qwen35::Qwen35Config> = layouts
            .iter()
            .map(|l| qwen35::local_dense_tp_config(&global, l))
            .collect();

        let n_tokens = n_ctx * n_chunk;
        let mut tokens_raw = vec![0u8; n_tokens * 4];
        ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
        let tokens: Vec<u32> = tokens_raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        let mut gpus = Gpus::init_tp(tp, global.n_layers).expect("init_tp");
        let kv_max = n_ctx + 16;
        let (mut weights, mut scratches, mut kvs, mut dns) = (vec![], vec![], vec![], vec![]);
        for (rank, g) in gpus.devices.iter_mut().enumerate() {
            g.bind_thread().expect("bind rank thread");
            g.active_stream = Some(g.hip.stream_create().expect("rank stream"));
            let local = &configs[rank];
            weights.push(
                qwen35::load_weights_dense_tp_rank(&mut hfq, &global, g, &layouts[rank])
                    .unwrap_or_else(|e| panic!("load TP rank {rank}: {e:?}")),
            );
            scratches
                .push(Qwen35Scratch::new_with_kv_max(g, local, 128, kv_max).expect("rank scratch"));
            kvs.push(
                KvCache::new_gpu_q8(g, local.n_layers, local.n_kv_heads, local.head_dim, kv_max)
                    .expect("rank kv"),
            );
            dns.push(DeltaNetState::new(g, local).expect("rank DeltaNet state"));
        }
        let peer = gpus.can_access_peer_all().expect("can_access_peer_all");
        if peer {
            gpus.prepare_tp2_whole_token_graph(global.n_layers * 2, global.dim * 4)
                .expect("prepare_tp2_whole_token_graph");
            assert!(gpus.enable_peer_all().expect("enable_peer_all"));
        }
        eprintln!("eval_hipfire: dense TP route=hand tp={tp} peer_access={peer}");
        {
            let mut knobs: Vec<(String, String)> = std::env::vars()
                .filter(|(k, _)| k.starts_with("HIPFIRE_TP_"))
                .collect();
            knobs.sort();
            if knobs.is_empty() {
                eprintln!("eval_hipfire: no HIPFIRE_TP_* env set");
            }
            for (k, v) in knobs {
                eprintln!("eval_hipfire: {k}={v}");
            }
        }

        gpus.devices[0].bind_thread().expect("bind rank 0");
        let h_buf = gpus.devices[0]
            .alloc_tensor(
                &[scored_per_chunk + usize::from(pad_scoring_tail), global.dim],
                DType::F32,
            )
            .expect("alloc hidden_buf");
        let vocab = global.vocab_size;
        let f16_lmhead = weights[0].output.gpu_dtype == DType::F16;

        for c in 0..effective_n_chunk {
            for (rank, g) in gpus.devices.iter_mut().enumerate() {
                g.bind_thread().expect("bind rank thread");
                dns[rank].reset(g).expect("reset DeltaNet state");
            }
            let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
            let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
            let mut chunk_nll_sum: f64 = 0.0;
            let mut chunk_nll_count: usize = 0;

            qwen35::forward_prefill_dense_tp_with_hidden(
                &mut gpus,
                &shard,
                &weights,
                &configs,
                &chunk_tokens[0..scoring_start],
                0,
                &mut kvs,
                &mut dns,
                &scratches,
                None,
            )
            .expect("forward_prefill_dense_tp prefix");
            qwen35::forward_prefill_dense_tp_with_hidden(
                &mut gpus,
                &shard,
                &weights,
                &configs,
                &chunk_tokens[scoring_start..(n_ctx - 1 + usize::from(pad_scoring_tail))],
                scoring_start,
                &mut kvs,
                &mut dns,
                &scratches,
                Some(&h_buf),
            )
            .expect("forward_prefill_dense_tp scored");

            gpus.devices[0].bind_thread().expect("bind rank 0");
            let gpu0 = &mut gpus.devices[0];
            let batched_logits: Option<rdna_compute::GpuTensor> = if f16_lmhead {
                let alloc = gpu0
                    .alloc_tensor(&[scored_per_chunk, vocab], DType::F32)
                    .expect("alloc batched lm_head logits");
                gpu0.gemm_f16_batched_lmhead(
                    &weights[0].output.buf,
                    &h_buf,
                    &alloc,
                    vocab,
                    global.dim,
                    scored_per_chunk,
                )
                .expect("gemm_f16_batched_lmhead");
                Some(alloc)
            } else {
                None
            };

            for j in 0..scored_per_chunk {
                let logits_view = if let Some(all_logits) = &batched_logits {
                    all_logits.sub_offset(j * vocab, vocab)
                } else {
                    let row_view = h_buf.sub_offset(j * global.dim, global.dim);
                    let ctx = hipfire_runtime::llama::DispatchCtx::new(gpu0);
                    let _ = hipfire_runtime::llama::gemv_family()
                        .run_auto(
                            &ctx,
                            gpu0,
                            &weights[0].output.dispatch_ref(),
                            &row_view,
                            &scratches[0].logits,
                        )
                        .expect("gemv lm_head");
                    scratches[0].logits.sub_offset(0, vocab)
                };
                let pos = scoring_start + j;
                let actual_next = chunk_tokens[pos + 1] as usize;
                ref_in.read_exact(&mut block_buf).expect("read ref block");
                let cand_logits = gpu0.download_f32(&logits_view).expect("download logits");
                let (kld, nll) = kld_from_block(&cand_logits, &block_buf, top_k, actual_next);
                chunk_klds.push(kld);
                if let Some(n) = nll {
                    chunk_nll_sum += n;
                    chunk_nll_count += 1;
                }
                total_scored_done += 1;
                if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                    let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                    let elapsed = t0.elapsed().as_secs_f64();
                    let rate = total_scored_done as f64 / elapsed.max(1e-9);
                    eprint!(
                        "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                        c + 1,
                        effective_n_chunk,
                        total_scored_done,
                        total_scored,
                        pct,
                        rate
                    );
                }
            }
            if let Some(alloc) = batched_logits {
                let _ = gpus.devices[0].free_tensor(alloc);
            }

            if let Some(r) = ktok.as_mut() {
                r.push_chunk(&chunk_klds).unwrap_or_else(|e| {
                    eprintln!("eval_hipfire: {e}");
                    std::process::exit(1);
                });
            }

            let (mean, p99, mean_nll) = hipfire_runtime::eval_common::seq_stats(
                &chunk_klds,
                chunk_nll_sum,
                chunk_nll_count,
            );
            mean_kld_per_seq.push(mean);
            p99_kld_per_seq.push(p99);
            mean_nll_per_seq.push(mean_nll);
        }
        let _ = gpus.devices[0].free_tensor(h_buf);
        let _ = gpus.free_peer_reduce_scratch();
    } else if is_qwen35 {
        // ----- qwen35 5|6 — WIRED (preserved byte-identical behavior) -----
        // Quant dtypes admitted: MQ4G256 (campaign --format mq4), MQ8G256,
        // HFQ4G256, Q8_0, F32, BF16 (via HIPFIRE_CALIB_BF16 / F32 widen). The
        // loader (`qwen35::load_weights`) matches quant_type 13->MQ4G256 etc and
        // respects calib_force_bf16 for teachers.
        use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};

        // Auto-route safetensors directories (ParoQuant / AWQ / HF native) — mirrors
        // daemon.rs:1500-1504. HFQ files take the canonical HFQ path below.
        let (config, weights) = if let Some(m) = &gguf_model {
            eprintln!("  loading via GGUF-native source arch_id={arch_id}");
            let weights = m.load_qwen35_weights(&mut gpu).expect("load GGUF weights");
            (m.hparams.config.clone(), weights)
        } else if args.model.is_dir() {
            use hipfire_runtime::safetensors_source::SafetensorsSource;
            let source = SafetensorsSource::open(&args.model).expect("safetensors open");
            let config = qwen35::config_from_safetensors(&source).expect("config_from_safetensors");
            eprintln!("  loading via safetensors (ParoQuant path) arch_id={arch_id}");
            let mut paro_source =
                qwen35::ParoSource::new(&source, &config).expect("ParoSource::new");
            let paro_layout = qwen35::Layout::single(config.n_layers);
            let weights = qwen35::load_weights(
                &mut paro_source,
                std::slice::from_mut(&mut gpu),
                &paro_layout,
            )
            .expect("load_weights");
            (config, weights)
        } else {
            let mut hfq = HfqFile::open(&args.model).expect("open model");
            let config = qwen35::config_from_hfq(&hfq).expect("read config");
            let weights = {
                let mut src = qwen35::HfqSource::new(&mut hfq, &config);
                let layout = qwen35::Layout::single(config.n_layers);
                qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
            }
            .expect("load weights");
            (config, weights)
        };
        if ref_n_vocab != config.vocab_size {
            eprintln!(
                "vocab mismatch: ref says {ref_n_vocab}, model says {}",
                config.vocab_size
            );
            std::process::exit(2);
        }
        // Read tokens (n_ctx * n_chunk u32s).
        let n_tokens = n_ctx * n_chunk;
        let mut tokens_raw = vec![0u8; n_tokens * 4];
        ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
        let tokens: Vec<u32> = tokens_raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        // -------- KV cache + DeltaNet state + scratch --------
        // qwen35 uses forward_prefill_batch with a q8 KV at seq_len+16 (calib_sweep qwen35 arm: seq_len+16, KvCache::new_gpu_q8).
        let kv_max = n_ctx + 16;
        let is_kv_layer: Vec<bool> = config
            .layer_types
            .iter()
            .map(|t| *t == qwen35::LayerType::FullAttention)
            .collect();
        let mut kv_cache = match args.kv_mode.as_str() {
            "q8" => KvCache::new_gpu_q8(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "fp8" => {
                // Stage-b Q0: native fp8 KV (E4M3FN + f16 scale per
                // token/head). Admission is the shared mode-specific guard
                // before allocation: neutral V (`--kv-v q8` spelling
                // identifies fp8/fp8) plus qwen35_native_refusal for Fp8
                // (exact gfx1201, single GPU, no adaptive/CASK; H24/KV4/D256
                // dense or H16/KV2/D256 MoE); never falls through to a q8
                // reader.
                if args.kv_v != "q8" {
                    eprintln!(
                        "eval_hipfire: --kv-mode fp8 requires --kv-v q8 (neutral V spelling); got --kv-v {}",
                        args.kv_v
                    );
                    std::process::exit(1);
                }
                if let Some(reason) = hipfire_runtime::kv_mode::qwen35_native_refusal(
                    hipfire_runtime::kv_mode::KvMode::Fp8,
                    gpu.arch.as_str(),
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    1,
                    false,
                    false,
                ) {
                    eprintln!("eval_hipfire: --kv-mode fp8 refused: {reason}");
                    std::process::exit(1);
                }
                eprintln!("eval_hipfire: fp8/fp8 native KV (E4M3FN + f16/token/head)");
                KvCache::new_gpu_fp8_filtered(
                    &mut gpu,
                    &is_kv_layer,
                    config.n_kv_heads,
                    config.head_dim,
                    kv_max,
                )
                .unwrap()
            }
            "bf16" => {
                // Quality control: unscaled bf16 K/V (stride D*2 per head,
                // no scales). Admission is the shared mode-specific guard
                // before allocation: neutral V, dense only, plus
                // qwen35_native_refusal for Bf16 (exact gfx1201, single GPU,
                // no adaptive/CASK, dense H24/Hkv4/D256); never falls through
                // to a q8 reader. Expected to read ≈ the f32 reference.
                if args.kv_v != "q8" {
                    eprintln!(
                        "eval_hipfire: --kv-mode bf16 requires --kv-v q8 (neutral V spelling); got --kv-v {}",
                        args.kv_v
                    );
                    std::process::exit(1);
                }
                if config.num_experts != 0 {
                    eprintln!(
                        "eval_hipfire: --kv-mode bf16 admitted only for dense Qwen (num_experts={})",
                        config.num_experts
                    );
                    std::process::exit(1);
                }
                if let Some(reason) = hipfire_runtime::kv_mode::qwen35_native_refusal(
                    hipfire_runtime::kv_mode::KvMode::Bf16,
                    gpu.arch.as_str(),
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    1,
                    false,
                    false,
                ) {
                    eprintln!("eval_hipfire: --kv-mode bf16 refused: {reason}");
                    std::process::exit(1);
                }
                eprintln!("eval_hipfire: bf16/bf16 unscaled KV (quality control)");
                KvCache::new_gpu_bf16_filtered(
                    &mut gpu,
                    &is_kv_layer,
                    config.n_kv_heads,
                    config.head_dim,
                    kv_max,
                )
                .unwrap()
            }
            "asym4" => KvCache::new_gpu_asym4(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "asym3" => KvCache::new_gpu_asym3(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "asym2" => KvCache::new_gpu_asym2(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "fwht4" => KvCache::new_gpu_fwht4_filtered(
                &mut gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "fwht3" => KvCache::new_gpu_fwht3_filtered(
                &mut gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "fwht2" => KvCache::new_gpu_fwht2_filtered(
                &mut gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "f32" | "f16" => KvCache::new_gpu(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            other => panic!("unknown --kv-mode: {other}"),
        };
        let v_mode = match args.kv_v.as_str() {
            "q8" => VMode::Q8,
            "lloyd2" => VMode::Lloyd2,
            "lloyd3" => VMode::Lloyd3,
            "lloyd4" => VMode::Lloyd4,
            other => panic!("unknown --kv-v: {other}"),
        };
        if v_mode != VMode::Q8 {
            kv_cache.set_v_mode_realloc(&mut gpu, v_mode).unwrap();
        }
        let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();

        // prefill-whole (additive): one captured call over the whole window.
        let whole_plan =
            whole_window.then(|| hipfire_runtime::eval_common::whole_window_plan(n_ctx));
        let hidden_buf = if args.scoring_mode == "prefill" {
            Some(
                gpu.alloc_tensor(
                    &[scored_per_chunk + usize::from(pad_scoring_tail), config.dim],
                    DType::F32,
                )
                .expect("alloc hidden_buf"),
            )
        } else if let Some(plan) = whole_plan.as_ref() {
            Some(
                gpu.alloc_tensor(&[plan.hidden_rows, config.dim], DType::F32)
                    .expect("alloc hidden_buf"),
            )
        } else {
            None
        };

        // -------- per-chunk loop (preserved byte-identical) --------
        for c in 0..effective_n_chunk {
            dn_state.reset(&mut gpu);
            let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
            let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
            let mut chunk_nll_sum: f64 = 0.0;
            let mut chunk_nll_count: usize = 0;

            if args.scoring_mode == "per-token" {
                for pos in 0..(n_ctx - 1) {
                    qwen35::forward_scratch(
                        &mut gpu,
                        &weights,
                        &config,
                        chunk_tokens[pos],
                        pos,
                        &mut kv_cache,
                        &mut dn_state,
                        &scratch,
                    )
                    .expect("forward_scratch");
                    if pos < scoring_start {
                        continue;
                    }
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let cand_logits = gpu.download_f32(&scratch.logits).expect("download logits");
                    let (kld, nll) = kld_from_block(&cand_logits, &block_buf, top_k, actual_next);
                    chunk_klds.push(kld);
                    if let Some(n) = nll {
                        chunk_nll_sum += n;
                        chunk_nll_count += 1;
                    }
                    total_scored_done += 1;
                    if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                        let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = total_scored_done as f64 / elapsed.max(1e-9);
                        eprint!(
                            "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                            c + 1,
                            effective_n_chunk,
                            total_scored_done,
                            total_scored,
                            pct,
                            rate
                        );
                    }
                }
            } else {
                // Prefill mode: batch the transformer stack via two
                // forward_prefill_batch calls (prefix + scored region), then
                // weight_gemv per scored position on the captured hidden states.
                let h_buf = hidden_buf.as_ref().expect("hidden_buf in prefill mode");
                let whole_scored;
                let h_buf = if let Some(plan) = whole_plan.as_ref() {
                    // prefill-whole / prefill-prompt: prefix, scored region and
                    // the final token in one call; score rows
                    // `first_scored_row..` (the same reference positions as the
                    // split mode). prefill-prompt captures through the plain
                    // prompt-prefill entry (production route); prefill-whole
                    // keeps the verify capture.
                    if prompt_capture {
                        qwen35::forward_prefill_batch_capture_hidden(
                            &mut gpu,
                            &weights,
                            &config,
                            &chunk_tokens[plan.rows.clone()],
                            0,
                            &mut kv_cache,
                            &mut dn_state,
                            &scratch,
                            h_buf,
                        )
                        .expect("forward_prefill_batch_capture_hidden whole window");
                    } else {
                        qwen35::forward_prefill_batch(
                            &mut gpu,
                            &weights,
                            &config,
                            &chunk_tokens[plan.rows.clone()],
                            0,
                            &mut kv_cache,
                            &mut dn_state,
                            &scratch,
                            None,
                            Some(h_buf),
                            None,
                            None,
                        )
                        .expect("forward_prefill_batch whole window");
                    }
                    whole_scored = h_buf.sub_offset(
                        plan.first_scored_row * config.dim,
                        plan.scored_rows * config.dim,
                    );
                    &whole_scored
                } else if let Some(cap) = args.scored_max_batch {
                    // `--scored-max-batch`: the same prefix call, then the
                    // scored region as consecutive `cap`-row verify-capture
                    // forwards (the shape of an MTP K = cap-1 verify).
                    qwen35::forward_prefill_batch(
                        &mut gpu,
                        &weights,
                        &config,
                        &chunk_tokens[0..scoring_start],
                        0,
                        &mut kv_cache,
                        &mut dn_state,
                        &scratch,
                        None,
                        None,
                        None,
                        None,
                    )
                    .expect("forward_prefill_batch prefix (scored-max-batch)");
                    let scored_rows = scoring_start..(n_ctx - 1 + usize::from(pad_scoring_tail));
                    qwen35::forward_prefill_batch_capped(
                        &mut gpu,
                        &weights,
                        &config,
                        &chunk_tokens[scored_rows],
                        scoring_start,
                        &mut kv_cache,
                        &mut dn_state,
                        &scratch,
                        None,
                        Some(h_buf),
                        None,
                        None,
                        cap,
                    )
                    .expect("forward_prefill_batch_capped scored");
                    h_buf
                } else {
                    qwen35::forward_prefill_batch(
                        &mut gpu,
                        &weights,
                        &config,
                        &chunk_tokens[0..scoring_start],
                        0,
                        &mut kv_cache,
                        &mut dn_state,
                        &scratch,
                        None,
                        None,
                        None,
                        None,
                    )
                    .expect("forward_prefill_batch prefix");
                    qwen35::forward_prefill_batch(
                        &mut gpu,
                        &weights,
                        &config,
                        &chunk_tokens[scoring_start..(n_ctx - 1 + usize::from(pad_scoring_tail))],
                        scoring_start,
                        &mut kv_cache,
                        &mut dn_state,
                        &scratch,
                        None,
                        Some(h_buf),
                        None,
                        None,
                    )
                    .expect("forward_prefill_batch scored");
                    h_buf
                };

                let f16_lmhead = weights.output.gpu_dtype == DType::F16;
                let batched_logits: Option<rdna_compute::GpuTensor> = if f16_lmhead {
                    let alloc = gpu
                        .alloc_tensor(&[scored_per_chunk, config.vocab_size], DType::F32)
                        .expect("alloc batched lm_head logits");
                    gpu.gemm_f16_batched_lmhead(
                        &weights.output.buf,
                        h_buf,
                        &alloc,
                        config.vocab_size,
                        config.dim,
                        scored_per_chunk,
                    )
                    .expect("gemm_f16_batched_lmhead");
                    Some(alloc)
                } else {
                    None
                };

                for j in 0..scored_per_chunk {
                    let logits_view = if let Some(ref all_logits) = batched_logits {
                        all_logits.sub_offset(j * config.vocab_size, config.vocab_size)
                    } else {
                        let row_view = h_buf.sub_offset(j * config.dim, config.dim);
                        let ctx = hipfire_runtime::llama::DispatchCtx::new(&gpu);
                        let _ = hipfire_runtime::llama::gemv_family()
                            .run_auto(
                                &ctx,
                                &mut gpu,
                                &weights.output.dispatch_ref(),
                                &row_view,
                                &scratch.logits,
                            )
                            .expect("gemv lm_head");
                        scratch.logits.sub_offset(0, config.vocab_size)
                    };
                    let pos = scoring_start + j;
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let cand_logits = gpu.download_f32(&logits_view).expect("download logits");
                    let (kld, nll) = kld_from_block(&cand_logits, &block_buf, top_k, actual_next);
                    chunk_klds.push(kld);
                    if let Some(n) = nll {
                        chunk_nll_sum += n;
                        chunk_nll_count += 1;
                    }
                    total_scored_done += 1;
                    if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                        let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = total_scored_done as f64 / elapsed.max(1e-9);
                        eprint!(
                            "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                            c + 1,
                            effective_n_chunk,
                            total_scored_done,
                            total_scored,
                            pct,
                            rate
                        );
                    }
                }
                if let Some(alloc) = batched_logits {
                    let _ = gpu.free_tensor(alloc);
                }
            }

            if let Some(r) = ktok.as_mut() {
                r.push_chunk(&chunk_klds).unwrap_or_else(|e| {
                    eprintln!("eval_hipfire: {e}");
                    std::process::exit(1);
                });
            }

            // Per-sequence (mean, p99, mean NLL): `eval_common::seq_stats`
            // (extracted from here; executed by the golden test).
            let (mean, p99, mean_nll) = hipfire_runtime::eval_common::seq_stats(
                &chunk_klds,
                chunk_nll_sum,
                chunk_nll_count,
            );
            mean_kld_per_seq.push(mean);
            p99_kld_per_seq.push(p99);
            mean_nll_per_seq.push(mean_nll);
        }
        // qwen35 owns hidden_buf drop; keep eprintln identity
        drop(hidden_buf);
        if let Ok(p) = std::env::var("HIPFIRE_MOE_EXPERT_STATS_OUT") {
            qwen35::dump_expert_stats(&p);
        }
    } else if is_llama {
        // ----- llama 0|1 — WIRED (mirrors eval_hipfire_llama.rs) -----
        // Quant dtypes admitted: Q4K, HFQ4G256, HFQ4G128, HFQ6G256, HFQ2/3,
        // MQ4G256 (campaign mq4), MQ8G256, MQ6G256, etc. plus F32 teacher.
        // The Llama loader (`hipfire_arch_llama::Llama`) accepts all HFQ quant
        // types (see hipfire-runtime/src/hfq.rs quant_type match); no hard filter.
        use hipfire_arch_llama::Llama;
        use hipfire_runtime::arch::Architecture;
        use hipfire_runtime::llama;

        if args.model.is_dir() {
            eprintln!("eval_hipfire: llama dir models not yet wired — use HFQ for arch 0|1");
            std::process::exit(1);
        }
        let mut hfq = HfqFile::open(&args.model).expect("open model");
        let config = <Llama as Architecture>::config_from_hfq(&hfq).expect("read config");
        if ref_n_vocab != config.vocab_size {
            eprintln!(
                "vocab mismatch: ref says {ref_n_vocab}, model says {}",
                config.vocab_size
            );
            std::process::exit(2);
        }
        let n_tokens = n_ctx * n_chunk;
        let mut tokens_raw = vec![0u8; n_tokens * 4];
        ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
        let tokens: Vec<u32> = tokens_raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        let kv_max = n_ctx + 16;
        let mut kv_cache = match args.kv_mode.as_str() {
            "q8" => KvCache::new_gpu_q8(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "asym4" => KvCache::new_gpu_asym4(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "asym3" => KvCache::new_gpu_asym3(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "asym2" => KvCache::new_gpu_asym2(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            "f32" | "f16" => KvCache::new_gpu(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .unwrap(),
            other => {
                eprintln!("llama: --kv-mode {other} not in q8/asym2/3/4/f32");
                std::process::exit(1);
            }
        };
        let scratch = <Llama as Architecture>::new_state(&mut gpu, &config).unwrap();
        let weights = <Llama as Architecture>::load_weights(&mut hfq, &config, &mut gpu)
            .expect("load weights");
        if args.scoring_mode == "prefill" {
            eprintln!("eval_hipfire: llama arch prefill not wired — falling back to per-token (prefill hook absent on llama arch, see eval_hipfire_llama.rs)");
        }
        for c in 0..effective_n_chunk {
            let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
            let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
            let mut chunk_nll_sum: f64 = 0.0;
            let mut chunk_nll_count: usize = 0;
            kv_cache.clear_gpu(&mut gpu).expect("kv clear");
            for pos in 0..(n_ctx - 1) {
                llama::forward_scratch_embed(
                    &mut gpu,
                    &weights,
                    &config,
                    chunk_tokens[pos],
                    pos,
                    &scratch,
                )
                .expect("forward_scratch_embed");
                llama::forward_scratch_compute(
                    &mut gpu,
                    &weights,
                    &config,
                    pos,
                    &mut kv_cache,
                    &scratch,
                )
                .expect("forward_scratch_compute");
                if pos < scoring_start {
                    continue;
                }
                let actual_next = chunk_tokens[pos + 1] as usize;
                ref_in.read_exact(&mut block_buf).expect("read ref block");
                let cand_logits = gpu.download_f32(&scratch.logits).expect("download logits");
                let (kld, nll) = kld_from_block(&cand_logits, &block_buf, top_k, actual_next);
                chunk_klds.push(kld);
                if let Some(n) = nll {
                    chunk_nll_sum += n;
                    chunk_nll_count += 1;
                }
                total_scored_done += 1;
                if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                    let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                    let elapsed = t0.elapsed().as_secs_f64();
                    let rate = total_scored_done as f64 / elapsed.max(1e-9);
                    eprint!(
                        "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                        c + 1,
                        effective_n_chunk,
                        total_scored_done,
                        total_scored,
                        pct,
                        rate
                    );
                }
            }
            if let Some(r) = ktok.as_mut() {
                r.push_chunk(&chunk_klds).unwrap_or_else(|e| {
                    eprintln!("eval_hipfire: {e}");
                    std::process::exit(1);
                });
            }
            if chunk_klds.is_empty() {
                mean_kld_per_seq.push(0.0);
                p99_kld_per_seq.push(0.0);
                mean_nll_per_seq.push(f64::NAN);
                continue;
            }
            let mean: f64 = chunk_klds.iter().copied().sum::<f64>() / chunk_klds.len() as f64;
            let mut sorted = chunk_klds.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
            let p99 = sorted[p99_idx];
            let mean_nll = if chunk_nll_count > 0 {
                chunk_nll_sum / chunk_nll_count as f64
            } else {
                f64::NAN
            };
            mean_kld_per_seq.push(mean);
            p99_kld_per_seq.push(p99);
            mean_nll_per_seq.push(mean_nll);
        }
    } else if is_gemma {
        // ----- gemma4 13 — WIRED -----
        // Quant dtypes admitted: MQ4G256(V2), HFQ4G256, HFQ4G128, HFQ6G256, HFQ2/3,
        // Q4K, MQ8G256, MQ6G256, MQ3G256, MQ2G256, plus F32/BF16 teachers (BF16
        // kept as BF16 when HIPFIRE_CALIB_BF16=1 else widened to F32). See
        // gemma4.rs:load_wt. Batched prefill needs a Q8 KV cache.
        use hipfire_arch_gemma4::{forward, Gemma4Config, Gemma4State, Gemma4Weights};
        if args.model.is_dir() {
            eprintln!("eval_hipfire: gemma4 dir not yet wired — use HFQ for arch 13");
            std::process::exit(1);
        }
        let hfq = HfqFile::open(&args.model).expect("open model");
        let cfg = Gemma4Config::from_hfq(&hfq).expect("gemma4 config");
        if ref_n_vocab != cfg.vocab_size {
            eprintln!(
                "vocab mismatch: ref says {ref_n_vocab}, model says {}",
                cfg.vocab_size
            );
            std::process::exit(2);
        }
        let n_tokens = n_ctx * n_chunk;
        let mut tokens_raw = vec![0u8; n_tokens * 4];
        ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
        let tokens: Vec<u32> = tokens_raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let weights = Gemma4Weights::load(&hfq, &cfg, &mut gpu).expect("gemma4 load weights");
        let kv_max = n_ctx + 16;
        // Q8 KV — mirrors calib_sweep: gemma sliding+full both q8 (lines 853-854)
        if args.kv_mode != "q8" {
            eprintln!("eval_hipfire: gemma4 requires --kv-mode q8 (got {}); F32 cache has no KvWriteF32 kernel — refusing", args.kv_mode);
            std::process::exit(1);
        }
        let mut state =
            Gemma4State::new_with_max_seq(&mut gpu, &cfg, kv_max).expect("gemma4 state");

        for c in 0..effective_n_chunk {
            state
                .kv_sliding
                .clear_gpu(&mut gpu)
                .expect("kv sliding clear");
            state.kv_full.clear_gpu(&mut gpu).expect("kv full clear");
            let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
            let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
            let mut chunk_nll_sum: f64 = 0.0;
            let mut chunk_nll_count: usize = 0;

            if args.scoring_mode == "prefill" {
                // Batched prefill of the prefix, then per-token for the scored window.
                forward::forward_prefill_batch(
                    &cfg,
                    &weights,
                    &mut state,
                    &mut gpu,
                    &chunk_tokens[..scoring_start],
                    0,
                )
                .expect("gemma prefill prefix");
                for pos in scoring_start..(n_ctx - 1) {
                    let cand_logits = forward::decode_step(
                        &cfg,
                        &weights,
                        &mut state,
                        &mut gpu,
                        chunk_tokens[pos],
                        pos as u32,
                    )
                    .expect("gemma decode_step");
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let (kld, nll) = kld_from_block(&cand_logits, &block_buf, top_k, actual_next);
                    chunk_klds.push(kld);
                    if let Some(n) = nll {
                        chunk_nll_sum += n;
                        chunk_nll_count += 1;
                    }
                    total_scored_done += 1;
                    if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                        let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = total_scored_done as f64 / elapsed.max(1e-9);
                        eprint!(
                            "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                            c + 1,
                            effective_n_chunk,
                            total_scored_done,
                            total_scored,
                            pct,
                            rate
                        );
                    }
                }
            } else {
                for pos in 0..(n_ctx - 1) {
                    let cand_logits = forward::decode_step(
                        &cfg,
                        &weights,
                        &mut state,
                        &mut gpu,
                        chunk_tokens[pos],
                        pos as u32,
                    )
                    .expect("gemma decode_step");
                    if pos < scoring_start {
                        continue;
                    }
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let (kld, nll) = kld_from_block(&cand_logits, &block_buf, top_k, actual_next);
                    chunk_klds.push(kld);
                    if let Some(n) = nll {
                        chunk_nll_sum += n;
                        chunk_nll_count += 1;
                    }
                    total_scored_done += 1;
                    if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                        let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = total_scored_done as f64 / elapsed.max(1e-9);
                        eprint!(
                            "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                            c + 1,
                            effective_n_chunk,
                            total_scored_done,
                            total_scored,
                            pct,
                            rate
                        );
                    }
                }
            }
            if let Some(r) = ktok.as_mut() {
                r.push_chunk(&chunk_klds).unwrap_or_else(|e| {
                    eprintln!("eval_hipfire: {e}");
                    std::process::exit(1);
                });
            }
            if chunk_klds.is_empty() {
                mean_kld_per_seq.push(0.0);
                p99_kld_per_seq.push(0.0);
                mean_nll_per_seq.push(f64::NAN);
                continue;
            }
            let mean: f64 = chunk_klds.iter().copied().sum::<f64>() / chunk_klds.len() as f64;
            let mut sorted = chunk_klds.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
            let p99 = sorted[p99_idx];
            let mean_nll = if chunk_nll_count > 0 {
                chunk_nll_sum / chunk_nll_count as f64
            } else {
                f64::NAN
            };
            mean_kld_per_seq.push(mean);
            p99_kld_per_seq.push(p99);
            mean_nll_per_seq.push(mean_nll);
        }
        let _ = &weights;
    } else if is_glimmer {
        // ----- glimmer 14 — WIRED -----
        // Quant dtypes admitted: Q8_0 (embed), HFQ4G256/G128, MQ4G256 etc (same family as gemma4/qwen35).
        // See glimmer.rs load weights quant_type match (3->Q8,6->HFQ4G256 etc).
        // Prefills via prefill_with_capture, chunk 192 (mirrors calib_sweep glimmer arm: prefill_chunk 192).
        use hipfire_arch_muse_glimmer::config::GlimmerConfig;
        use hipfire_arch_muse_glimmer::forward as glimmer_fwd;
        use hipfire_arch_muse_glimmer::glimmer::{GlimmerState, GlimmerWeights};
        if args.model.is_dir() {
            eprintln!("eval_hipfire: glimmer dir not yet wired — use HFQ for arch 14");
            std::process::exit(1);
        }
        let mut hfq = HfqFile::open(&args.model).expect("open model");
        let cfg = GlimmerConfig::from_hfq(&hfq).expect("glimmer cfg");
        if ref_n_vocab != cfg.vocab_size {
            eprintln!(
                "vocab mismatch: ref says {ref_n_vocab}, model says {}",
                cfg.vocab_size
            );
            std::process::exit(2);
        }
        let n_tokens = n_ctx * n_chunk;
        let mut tokens_raw = vec![0u8; n_tokens * 4];
        ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
        let tokens: Vec<u32> = tokens_raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let weights = GlimmerWeights::load(&hfq, &cfg, &mut gpu).expect("glimmer weights");
        let kv_max = n_ctx + 16;
        let mut state =
            GlimmerState::new_with_max_seq(&mut gpu, &cfg, kv_max).expect("glimmer state");
        // chunk size 192 — mirrors calib_sweep glimmer_prefill_chunk_size (192)
        let glimmer_chunk: usize = 192;

        for c in 0..effective_n_chunk {
            state.reset();
            // For scoring_start prefix, use batched prefill chunked to 192 to mirror calib_sweep
            // Note: GlimmerState reset clears n_tokens but KV buffers remain; next prefill overwrites from 0.
            let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
            let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
            let mut chunk_nll_sum: f64 = 0.0;
            let mut chunk_nll_count: usize = 0;

            if args.scoring_mode == "prefill" {
                // Prefill prefix chunked 192 (mirrors calib_sweep glimmer_prefill_chunk_size 192)
                let mut pos = 0usize;
                while pos < scoring_start {
                    let end = (pos + glimmer_chunk).min(scoring_start);
                    let sub = &chunk_tokens[pos..end];
                    glimmer_fwd::prefill_with_capture(
                        &cfg,
                        &weights,
                        &mut state,
                        &mut gpu,
                        sub,
                        pos as u32,
                        &[],
                        &mut Vec::new(),
                    )
                    .expect("glimmer prefill prefix");
                    pos = end;
                }
                // Batched scored window: one 52-layer batched forward capturing hidden states,
                // mirroring qwen35's prefill shape (prefix batched, scored batched capturing, then
                // per-position lm_head on captured states).
                //
                // capture_layers = [cfg.n_layers - 1] i.e. [51] (0-based). Rationale:
                //   - `prefill_chunk_batched` (forward.rs:3738-3755) captures post-layer residuals
                //     `x` after each decoder layer (sandwich post-norm + residual already applied).
                //     See doc at forward.rs:1159-1164 "post-layer residual `x`".
                //   - `decode_step_body` (forward.rs:462-504) runs 52 layers (0..51), then
                //     `rmsnorm_f32(&state.x, &weights.final_norm, &state.tmp, eps=1e-5)` ->
                //     `weight_gemv(&state.tmp, lm_head)` -> scale -> softcap.
                //     The tensor lm_head consumes is the FINAL-NORMED hidden (tmp), NOT the
                //     raw residual. No capture tap can directly yield post-final-norm because
                //     final norm lives outside the layer loop (forward.rs:485-487).
                //   - Therefore we capture the last layer's post-residual (layer 51) which
                //     is `state.x` before final norm, then run `rmsnorm_batched` over the
                //     captured buffer so the lm_head sees exactly the same input as decode_step.
                //     Capturing 51 is correct because cfg.n_layers==52 (config.rs:27).
                //
                // hidden_out layout (forward.rs:3890-3893, 2836-2852, 2647-2659):
                //   Host Host path: `cap_buf` sized `b*cap_cnt*dim` then `hidden_out.extend_from_slice`.
                //   For cap_cnt=1, total `B*dim` floats, position-major row-major `[B, dim]`.
                //   Row j at offset j*dim corresponds to scored position pos=scoring_start+j
                //   predicting token chunk_tokens[pos+1] — identical to per-token `actual_next`
                //   alignment. Verified via prefill_chunk_batched download loops (x is [B*dim]).
                let scored_slice = &chunk_tokens[scoring_start..(n_ctx - 1)];
                let scored_len = scored_slice.len(); // == scored_per_chunk
                assert_eq!(scored_len, scored_per_chunk);
                let cap_layer = cfg.n_layers - 1;
                let mut hidden_host: Vec<f32> = Vec::with_capacity(scored_len * cfg.dim);
                glimmer_fwd::prefill_with_capture(
                    &cfg,
                    &weights,
                    &mut state,
                    &mut gpu,
                    scored_slice,
                    scoring_start as u32,
                    &[cap_layer],
                    &mut hidden_host,
                )
                .expect("glimmer prefill scored capture");
                assert_eq!(
                    hidden_host.len(),
                    scored_len * cfg.dim,
                    "hidden_out layout: B*dim"
                );
                // Move captured pre-norm hidden to GPU for batched final norm.
                // Upload as flat [B*dim] F32, then rmsnorm_batched -> normed.
                let hidden_gpu = gpu
                    .upload_f32(&hidden_host, &[scored_len * cfg.dim])
                    .expect("upload hidden");
                let normed_gpu = gpu
                    .alloc_tensor(&[scored_len * cfg.dim], rdna_compute::DType::F32)
                    .expect("alloc normed");
                gpu.rmsnorm_batched(
                    &hidden_gpu,
                    &weights.final_norm,
                    &normed_gpu,
                    scored_len,
                    cfg.dim,
                    cfg.rms_norm_eps,
                )
                .expect("rmsnorm_batched scored");
                // Per-position lm_head on the final-normed hidden. Glimmer's lm_head is
                // typically HFQ4/MQ4 (not F16), so `gemm_f16_batched_lmhead` is absent
                // (qwen uses it only when weights.output.gpu_dtype==F16). Use the
                // cheap per-row weight_gemv fallback that qwen also uses for non-F16
                // — one GEMV per scored token is negligible vs the 52-layer forward
                // that we already collapsed from 1024 passes to one batched pass.
                let logits_gpu = gpu
                    .alloc_tensor(&[cfg.vocab_size], rdna_compute::DType::F32)
                    .expect("alloc logits");
                for j in 0..scored_len {
                    let pos = scoring_start + j;
                    let row = normed_gpu.sub_offset(j * cfg.dim, cfg.dim);
                    hipfire_runtime::llama::weight_gemv(
                        &mut gpu,
                        &weights.lm_head,
                        &row,
                        &logits_gpu,
                    )
                    .expect("glimmer lm_head");
                    if cfg.output_multiplier != 1.0 {
                        gpu.scale_f32(&logits_gpu, cfg.output_multiplier)
                            .expect("scale");
                    }
                    if cfg.final_logit_softcapping > 0.0 {
                        gpu.logit_softcap_f32(
                            &logits_gpu,
                            cfg.vocab_size,
                            cfg.final_logit_softcapping,
                        )
                        .expect("softcap");
                    }
                    let logits = gpu.download_f32(&logits_gpu).expect("download logits");
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let (kld, nll) = kld_from_block(&logits, &block_buf, top_k, actual_next);
                    chunk_klds.push(kld);
                    if let Some(n) = nll {
                        chunk_nll_sum += n;
                        chunk_nll_count += 1;
                    }
                    total_scored_done += 1;
                    if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                        let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = total_scored_done as f64 / elapsed.max(1e-9);
                        eprint!(
                            "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                            c + 1,
                            effective_n_chunk,
                            total_scored_done,
                            total_scored,
                            pct,
                            rate
                        );
                    }
                }
                let _ = gpu.free_tensor(hidden_gpu);
                let _ = gpu.free_tensor(normed_gpu);
                let _ = gpu.free_tensor(logits_gpu);
            } else {
                for pos in 0..(n_ctx - 1) {
                    let logits = glimmer_fwd::decode_step(
                        &cfg,
                        &weights,
                        &mut state,
                        &mut gpu,
                        chunk_tokens[pos],
                        pos as u32,
                    )
                    .expect("glimmer decode");
                    if pos < scoring_start {
                        continue;
                    }
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let (kld, nll) = kld_from_block(&logits, &block_buf, top_k, actual_next);
                    chunk_klds.push(kld);
                    if let Some(n) = nll {
                        chunk_nll_sum += n;
                        chunk_nll_count += 1;
                    }
                    total_scored_done += 1;
                    if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                        let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = total_scored_done as f64 / elapsed.max(1e-9);
                        eprint!(
                            "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                            c + 1,
                            effective_n_chunk,
                            total_scored_done,
                            total_scored,
                            pct,
                            rate
                        );
                    }
                }
            }
            if let Some(r) = ktok.as_mut() {
                r.push_chunk(&chunk_klds).unwrap_or_else(|e| {
                    eprintln!("eval_hipfire: {e}");
                    std::process::exit(1);
                });
            }
            if chunk_klds.is_empty() {
                mean_kld_per_seq.push(0.0);
                p99_kld_per_seq.push(0.0);
                mean_nll_per_seq.push(f64::NAN);
                continue;
            }
            let mean: f64 = chunk_klds.iter().copied().sum::<f64>() / chunk_klds.len() as f64;
            let mut sorted = chunk_klds.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
            let p99 = sorted[p99_idx];
            let mean_nll = if chunk_nll_count > 0 {
                chunk_nll_sum / chunk_nll_count as f64
            } else {
                f64::NAN
            };
            mean_kld_per_seq.push(mean);
            p99_kld_per_seq.push(p99);
            mean_nll_per_seq.push(mean_nll);
        }
        let _ = &weights;
    } else if is_lfm {
        // ----- lfm2 11 — WIRED (dense 1.2b/350m, num_experts==0) -----
        // Quant dtypes admitted: dense projections Q8_0/HFQ4G256/MQ4G256 (batch_weight_formats_supported);
        // embeddings Q8/HFQ4; lm_head Q8/HFQ4/MQ4/HFQ6/MQ6/MQ3 (+F32/BF16 teachers).
        // See lfm2moe/src/lfm2moe.rs batch_weight_formats_supported and dispatch's ProjectLadder::Gemm.
        // Uses forward_decode_batch_lfm with explicit positions; KV is q8 and only some layers carry KV
        // (KvCache q8 filtered on is_kv_layer, lane_capacity = n_ctx+16).
        use hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState;
        use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
        use hipfire_arch_lfm2moe::forward_batch;
        use hipfire_arch_lfm2moe::lfm2moe::Lfm2MoeWeights;
        if args.model.is_dir() {
            eprintln!("eval_hipfire: lfm2 dir not yet wired — use HFQ for arch 11");
            std::process::exit(1);
        }
        let mut hfq = HfqFile::open(&args.model).expect("open model");
        let cfg = Lfm2MoeConfig::from_hfq(&hfq).unwrap_or_else(|e| panic!("lfm2 cfg: {e}"));
        if cfg.num_experts != 0 {
            eprintln!(
                "lfm2 MoE (num_experts={}) not admitted for eval_hipfire dense lane",
                cfg.num_experts
            );
            std::process::exit(1);
        }
        if ref_n_vocab != cfg.vocab_size {
            eprintln!(
                "vocab mismatch: ref says {ref_n_vocab}, model says {}",
                cfg.vocab_size
            );
            std::process::exit(2);
        }
        if let Err(e) = forward_batch::batch_weight_formats_supported(
            &Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu)
                .as_ref()
                .map(|w| w as &Lfm2MoeWeights)
                .unwrap_or_else(|_| panic!("peek")),
        ) {
            // placeholder: actual check after load below; keep compile shape
            let _ = e;
        }
        // Reload after peek above — we need weights for real
        let mut hfq2 = HfqFile::open(&args.model).expect("reopen model");
        let weights = Lfm2MoeWeights::load(&mut hfq2, &cfg, &mut gpu).expect("lfm2 weights");
        if let Err(e) = forward_batch::batch_weight_formats_supported(&weights) {
            eprintln!("unsupported lfm2 weight formats for batched decode: {e}");
            std::process::exit(2);
        }
        let n_tokens = n_ctx * n_chunk;
        let mut tokens_raw = vec![0u8; n_tokens * 4];
        ref_in.read_exact(&mut tokens_raw).expect("read ref tokens");
        let tokens: Vec<u32> = tokens_raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let kv_max = n_ctx + 16;
        let max_batch = n_ctx;
        let mut state = Lfm2DecodeBatchState::new(&mut gpu, &cfg, max_batch, kv_max, 32)
            .expect("lfm batch state");

        for c in 0..effective_n_chunk {
            state.reset(&mut gpu).expect("lfm reset");
            let chunk_tokens = &tokens[c * n_ctx..(c + 1) * n_ctx];
            let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
            let mut chunk_nll_sum: f64 = 0.0;
            let mut chunk_nll_count: usize = 0;

            // LFM2 scoring: batched decode for the entire chunk (positions 0..n_ctx-1)
            // then score each scored position's logits from state.logits.
            // This matches calib_sweep's lfm arm: forward_decode_batch_lfm with per-chunk positions.
            if args.scoring_mode == "prefill" || args.scoring_mode == "per-token" {
                let positions: Vec<usize> = (0..chunk_tokens.len()).collect();
                forward_batch::forward_decode_batch_lfm(
                    &mut gpu,
                    &weights,
                    &cfg,
                    chunk_tokens,
                    &positions,
                    &mut state,
                )
                .expect("lfm forward batch");
                // state.logits holds [max_batch * vocab] batched; lane i logits at offset i*vocab
                let all_logits = gpu
                    .download_f32(&state.logits)
                    .expect("download lfm logits");
                for pos in scoring_start..(n_ctx - 1) {
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let lane_logits = &all_logits[pos * cfg.vocab_size..(pos + 1) * cfg.vocab_size];
                    let (kld, nll) = kld_from_block(lane_logits, &block_buf, top_k, actual_next);
                    chunk_klds.push(kld);
                    if let Some(n) = nll {
                        chunk_nll_sum += n;
                        chunk_nll_count += 1;
                    }
                    total_scored_done += 1;
                    if total_scored_done % 1024 == 0 || total_scored_done == total_scored {
                        let pct = total_scored_done as f64 * 100.0 / total_scored as f64;
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = total_scored_done as f64 / elapsed.max(1e-9);
                        eprint!(
                            "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                            c + 1,
                            effective_n_chunk,
                            total_scored_done,
                            total_scored,
                            pct,
                            rate
                        );
                    }
                }
            }

            if let Some(r) = ktok.as_mut() {
                r.push_chunk(&chunk_klds).unwrap_or_else(|e| {
                    eprintln!("eval_hipfire: {e}");
                    std::process::exit(1);
                });
            }
            if chunk_klds.is_empty() {
                mean_kld_per_seq.push(0.0);
                p99_kld_per_seq.push(0.0);
                mean_nll_per_seq.push(f64::NAN);
                continue;
            }
            let mean: f64 = chunk_klds.iter().copied().sum::<f64>() / chunk_klds.len() as f64;
            let mut sorted = chunk_klds.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
            let p99 = sorted[p99_idx];
            let mean_nll = if chunk_nll_count > 0 {
                chunk_nll_sum / chunk_nll_count as f64
            } else {
                f64::NAN
            };
            mean_kld_per_seq.push(mean);
            p99_kld_per_seq.push(p99);
            mean_nll_per_seq.push(mean_nll);
        }
        let _ = &weights;
    }

    eprintln!();
    eprintln!(
        "eval_hipfire: scored {total_scored_done} tokens in {:.1}s ({:.0} tok/s) arch_id={arch_id}",
        t0.elapsed().as_secs_f64(),
        total_scored_done as f64 / t0.elapsed().as_secs_f64().max(1e-9),
    );

    // -------- write HFKSEQ output (v2: adds mean_nll per chunk) --------
    // Arch-independent: same HFKSEQ writer for all arches, so numbers are comparable.
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output parent dir");
        }
    }
    let out_file = File::create(&args.output).expect("create output");
    let mut out = BufWriter::new(out_file);
    hipfire_runtime::eval_common::write_hfkseq(
        &mut out,
        effective_n_chunk,
        &mean_kld_per_seq,
        &p99_kld_per_seq,
        &mean_nll_per_seq,
    )
    .unwrap();
    if let Some(path) = &args.per_token_output {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).expect("create per-token output parent dir");
            }
        }
        let mut tok_out = BufWriter::new(File::create(path).expect("create per-token output"));
        ktok.as_ref()
            .expect("per-token recorder")
            .write(&mut tok_out)
            .expect("write per-token output");
        eprintln!("eval_hipfire: wrote {} (HFKTOK v1)", path.display());
    }

    let overall_mean: f64 =
        mean_kld_per_seq.iter().copied().sum::<f64>() / mean_kld_per_seq.len() as f64;
    let nll_finite: Vec<f64> = mean_nll_per_seq
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    let overall_nll: f64 = if nll_finite.is_empty() {
        f64::NAN
    } else {
        nll_finite.iter().copied().sum::<f64>() / nll_finite.len() as f64
    };
    let overall_ppl = overall_nll.exp();
    eprintln!(
        "eval_hipfire: slice-mean KLD = {:.6}  mean NLL = {:.6}  PPL = {:.4} arch_id={arch_id}",
        overall_mean, overall_nll, overall_ppl
    );
    eprintln!("eval_hipfire: wrote {}", args.output.display());
    if is_qwen35 {
        if let Ok(p) = std::env::var("HIPFIRE_MOE_EXPERT_STATS_OUT") {
            hipfire_arch_qwen35::qwen35::dump_expert_stats(&p);
        }
    }
}

// (verify_ref_sha256 now lives in hipfire_runtime::eval_common — see
// crates/hipfire-runtime/src/eval_common.rs)
