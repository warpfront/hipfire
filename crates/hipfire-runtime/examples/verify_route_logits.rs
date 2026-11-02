//! Teacher-forced, DFlash-ChainVerify-shaped logit dump (qwen35 only).
//!
//! Prefills `tokens[0..ctx)` with `forward_prefill_batch`, then feeds
//! `tokens[ctx..ctx+score)` in blocks of `--block` through
//! `forward_prefill_batch_with_pbs_opts(.., DflashFusionCtx::ChainVerify)` at
//! advancing `start_pos` (the shape a DFlash verify block takes), applying the
//! lm_head to each block's hidden rows and appending raw LE f32 logits.
//!
//! The attention route (HIPFIRE_GFX12_FA2_SPLIT_VERIFY_FP8) is read by
//! rdna-compute from the environment; this program never sets it.
//!
//! usage: verify_route_logits --model <path> --tokens <u32 LE file> --ctx <N>
//!        --score <M> [--block 8] [--kv-mode fp8] --max-seq <S> --out <logits.f32>
//!        [--ids-out <argmax ids u32 LE>]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, DflashFusionCtx, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use rdna_compute::DType;
    use std::io::{BufWriter, Write};
    use std::path::PathBuf;
    use std::time::Instant;

    // -------- args --------
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut tokens_path: Option<PathBuf> = None;
    let mut ctx: Option<usize> = None;
    let mut score: Option<usize> = None;
    let mut block: usize = 8;
    let mut kv_mode = "fp8".to_string();
    let mut max_seq: Option<usize> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut ids_out: Option<PathBuf> = None;
    let mut i = 1;
    while i < argv.len() {
        let val = |i: &mut usize| -> String {
            *i += 1;
            argv.get(*i)
                .unwrap_or_else(|| {
                    eprintln!("missing value for {}", argv[*i - 1]);
                    std::process::exit(1)
                })
                .clone()
        };
        match argv[i].as_str() {
            "--model" => model = Some(PathBuf::from(val(&mut i))),
            "--tokens" => tokens_path = Some(PathBuf::from(val(&mut i))),
            "--ctx" => ctx = Some(val(&mut i).parse().expect("--ctx")),
            "--score" => score = Some(val(&mut i).parse().expect("--score")),
            "--block" => block = val(&mut i).parse().expect("--block"),
            "--kv-mode" => kv_mode = val(&mut i),
            "--max-seq" => max_seq = Some(val(&mut i).parse().expect("--max-seq")),
            "--out" => out_path = Some(PathBuf::from(val(&mut i))),
            "--ids-out" => ids_out = Some(PathBuf::from(val(&mut i))),
            "-h" | "--help" => {
                eprintln!("Usage: verify_route_logits --model <path> --tokens <file> --ctx <N> --score <M> [--block 8] [--kv-mode fp8] --max-seq <S> --out <logits.f32> [--ids-out <ids.bin>]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    let model = model.expect("--model required");
    let tokens_path = tokens_path.expect("--tokens required");
    let ctx = ctx.expect("--ctx required");
    let score_req = score.expect("--score required");
    let max_seq = max_seq.expect("--max-seq required");
    let out_path = out_path.expect("--out required");
    if kv_mode != "fp8" {
        eprintln!("verify_route_logits: only --kv-mode fp8 (native fp8 KV) is supported");
        std::process::exit(1);
    }
    if block == 0 {
        eprintln!("--block must be >= 1");
        std::process::exit(1);
    }

    // -------- tokens --------
    let raw = std::fs::read(&tokens_path).expect("read tokens");
    if raw.len() % 4 != 0 {
        eprintln!("tokens file size is not a multiple of 4");
        std::process::exit(1);
    }
    let tokens: Vec<u32> = raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if ctx == 0 || ctx >= tokens.len() {
        eprintln!("--ctx {ctx} must be in 1..{}", tokens.len());
        std::process::exit(1);
    }
    let score_n = score_req.min(tokens.len() - ctx);
    if score_n < score_req {
        eprintln!(
            "verify_route_logits: clamped --score {score_req} -> {score_n} (tokens.len()={})",
            tokens.len()
        );
    }
    if ctx + score_n > max_seq {
        eprintln!("--max-seq {max_seq} < ctx+score {}", ctx + score_n);
        std::process::exit(1);
    }

    // Same eval-mode env as eval_hipfire, EXCEPT the route flag which is
    // left untouched. SAFETY: single-threaded init phase.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_KV_MODE", &kv_mode);
        std::env::set_var("HIPFIRE_KV_V", "q8");
        std::env::set_var("HIPFIRE_PREFILL_REUSE_PBS", "1");
    }

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!(
        "verify_route_logits: arch={} model={}",
        gpu.arch,
        model.display()
    );
    eprintln!(
        "verify_route_logits: HIPFIRE_GFX12_FA2_SPLIT_VERIFY_FP8={:?}",
        std::env::var("HIPFIRE_GFX12_FA2_SPLIT_VERIFY_FP8").ok()
    );

    // -------- arch detection (qwen35 only) --------
    let mut hfq = HfqFile::open(&model).unwrap_or_else(|e| {
        eprintln!("open model {}: {e}", model.display());
        std::process::exit(1)
    });
    let arch_id = hfq.arch_id;
    if !matches!(arch_id, 5 | 6) {
        eprintln!("verify_route_logits: unsupported arch_id {arch_id} (qwen35 5|6 only)");
        std::process::exit(1);
    }
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");
    let vocab = config.vocab_size;
    if tokens.iter().any(|&t| t as usize >= vocab) {
        eprintln!("token id out of vocab range ({vocab})");
        std::process::exit(1);
    }

    // -------- KV cache + DeltaNet state + scratch --------
    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == qwen35::LayerType::FullAttention)
        .collect();
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
        eprintln!("verify_route_logits: native fp8 KV refused: {reason}");
        std::process::exit(1);
    }
    let mut kv_cache = KvCache::new_gpu_fp8_filtered(
        &mut gpu,
        &is_kv_layer,
        config.n_kv_heads,
        config.head_dim,
        max_seq,
    )
    .unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let h_buf = gpu
        .alloc_tensor(&[block, config.dim], DType::F32)
        .expect("alloc h_buf");
    eprintln!(
        "verify_route_logits: fusion=DflashFusionCtx::ChainVerify kv=fp8 (new_gpu_fp8_filtered) ctx={ctx} score={score_n} block={block} max_seq={max_seq}"
    );

    // -------- (1) prefix prefill --------
    let t0 = Instant::now();
    qwen35::forward_prefill_batch(
        &mut gpu,
        &weights,
        &config,
        &tokens[0..ctx],
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
    eprintln!("prefill: {:.2}s", t0.elapsed().as_secs_f64());

    // -------- (2)+(3) teacher-forced verify blocks --------
    let mut out = BufWriter::new(std::fs::File::create(&out_path).expect("create --out"));
    let mut ids_w = ids_out
        .as_ref()
        .map(|p| BufWriter::new(std::fs::File::create(p).expect("create --ids-out")));
    let f16_lmhead = weights.output.gpu_dtype == DType::F16;
    let n_blocks = score_n.div_ceil(block);
    let mut rows = 0usize;
    for j in 0..n_blocks {
        let lo = ctx + j * block;
        let hi = (lo + block).min(ctx + score_n);
        let n = hi - lo;
        let tb = Instant::now();
        qwen35::forward_prefill_batch_with_pbs_opts(
            &mut gpu,
            &weights,
            &config,
            &tokens[lo..hi],
            lo,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            Some(&h_buf),
            None,
            None,
            None,
            None,
            None,
            false,
            DflashFusionCtx::ChainVerify,
        )
        .expect("forward_prefill_batch_with_pbs_opts ChainVerify");

        let batched_logits: Option<rdna_compute::GpuTensor> = if f16_lmhead {
            let alloc = gpu
                .alloc_tensor(&[n, vocab], DType::F32)
                .expect("alloc batched lm_head logits");
            gpu.gemm_f16_batched_lmhead(&weights.output.buf, &h_buf, &alloc, vocab, config.dim, n)
                .expect("gemm_f16_batched_lmhead");
            Some(alloc)
        } else {
            None
        };
        for r in 0..n {
            let logits_view = if let Some(all) = &batched_logits {
                all.sub_offset(r * vocab, vocab)
            } else {
                let row_view = h_buf.sub_offset(r * config.dim, config.dim);
                let dctx = hipfire_runtime::llama::DispatchCtx::new(&gpu);
                let _ = hipfire_runtime::llama::gemv_family()
                    .run_auto(
                        &dctx,
                        &mut gpu,
                        &weights.output.dispatch_ref(),
                        &row_view,
                        &scratch.logits,
                    )
                    .expect("gemv lm_head");
                scratch.logits.sub_offset(0, vocab)
            };
            let logits = gpu.download_f32(&logits_view).expect("download logits");
            let mut best = 0usize;
            for (k, v) in logits.iter().enumerate() {
                if *v > logits[best] {
                    best = k;
                }
            }
            let mut bytes = Vec::with_capacity(logits.len() * 4);
            for v in &logits {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            out.write_all(&bytes).expect("write logits");
            if let Some(w) = ids_w.as_mut() {
                w.write_all(&(best as u32).to_le_bytes())
                    .expect("write ids");
            }
            rows += 1;
        }
        eprintln!(
            "block {}/{} pos={} n={} {:.1} ms",
            j + 1,
            n_blocks,
            lo,
            n,
            tb.elapsed().as_secs_f64() * 1e3
        );
    }
    out.flush().expect("flush out");
    if let Some(w) = ids_w.as_mut() {
        w.flush().expect("flush ids");
    }
    println!("ROWS={rows} VOCAB={vocab}");
}
