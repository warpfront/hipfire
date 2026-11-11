// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GGUF-native greedy generation (`HIPFIRE_GGUF_NATIVE=1`, dense Qwen3.5 / 3.8
//! GGUF on gfx1201): prefill each prompt, then greedy-decode `--max-tokens`
//! tokens on the ggml decode route. Prints the token ids and the decoded text
//! of every prompt and, with `--out`, writes the ids (one prompt per line) so
//! two fresh processes can be compared byte for byte.
//!
//! Usage: gguf_native_gen <model.gguf> [--prompt TEXT]... [--max-tokens N]
//!        [--chat] [--out ids.txt] [--time]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_loader::gguf_native::GgufNativeModel;
    use hipfire_runtime::llama::{self, KvCache};
    use std::io::Write;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: gguf_native_gen <model.gguf> [--prompt TEXT]... [--max-tokens N] [--chat] [--out ids.txt] [--time]");
        std::process::exit(1);
    }
    let model_path = std::path::PathBuf::from(&args[1]);
    let mut prompts: Vec<String> = Vec::new();
    let mut max_tokens = 64usize;
    let mut chat = false;
    let mut out_path: Option<String> = None;
    let mut timing = false;
    let mut layers_keep: Option<usize> = None;
    let mut compare: Vec<usize> = Vec::new();
    let mut dequant_check = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                prompts.push(args[i + 1].clone());
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = args[i + 1].parse().expect("--max-tokens integer");
                i += 2;
            }
            "--chat" => {
                chat = true;
                i += 1;
            }
            "--out" => {
                out_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--time" => {
                timing = true;
                i += 1;
            }
            "--dequant-check" => {
                dequant_check = true;
                i += 1;
            }
            "--layers" => {
                layers_keep = Some(args[i + 1].parse().expect("--layers integer"));
                i += 2;
            }
            "--compare" => {
                compare = args[i + 1]
                    .split(',')
                    .map(|s| s.parse().expect("--compare list of integers"))
                    .collect();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    if prompts.is_empty() {
        prompts = vec![
            "The capital of France is".to_string(),
            "def fibonacci(n):".to_string(),
            "A hash map is a data structure that".to_string(),
        ];
    }

    let model = GgufNativeModel::open(&model_path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_gguf(model.gguf())
        .unwrap_or_else(|e| panic!("tokenizer: {e}"));
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!(
        "gguf_native_gen: arch={} model={}",
        gpu.arch,
        model_path.display()
    );
    let mut config = model.hparams.config.clone();
    let t_load = Instant::now();
    let mut weights = model
        .load_qwen35_weights(&mut gpu)
        .expect("load GGUF weights");
    eprintln!("weights loaded in {:.2}s", t_load.elapsed().as_secs_f64());
    if let Some(l) = layers_keep {
        // Debug truncation: the first `l` layers only (logits are not meaningful,
        // the batched and decode paths are compared against each other).
        config.n_layers = l;
        config.layer_types.truncate(l);
        weights.layers.truncate(l);
    }

    let kv_seq = 2048usize;
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_seq,
    )
    .unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    if dequant_check {
        // Device self-test of the prefill dequantisation: for a weight W,
        // `y1 = mmvq(W, x)` (the decode route, A8 activations) against
        // `y2 = (dequantised W) . x` accumulated in f64 on the host. The two
        // must agree to the q8_1 activation rounding (cosine ~ 1 - 1e-5).
        use hipfire_runtime::llama::WeightTensor;
        use qwen35::LayerWeights;
        let mut items: Vec<(String, &WeightTensor)> = Vec::new();
        for li in [0usize, 3] {
            match &weights.layers[li] {
                LayerWeights::DeltaNet(l) => {
                    for (n, w) in [
                        ("wqkv", &l.wqkv),
                        ("wz", &l.wz),
                        ("wo", &l.wo),
                        ("gate", &l.w_gate),
                        ("up", &l.w_up),
                        ("down", &l.w_down),
                    ] {
                        items.push((format!("L{li}.{n}"), w));
                    }
                }
                LayerWeights::FullAttn(l) => {
                    for (n, w) in [
                        ("wq", &l.wq),
                        ("wk", &l.wk),
                        ("wv", &l.wv),
                        ("wo", &l.wo),
                        ("gate", &l.w_gate),
                        ("up", &l.w_up),
                        ("down", &l.w_down),
                    ] {
                        items.push((format!("L{li}.{n}"), w));
                    }
                }
                _ => {}
            }
        }
        items.push(("output".to_string(), &weights.output));
        for (name, w) in items {
            if !w.gpu_dtype.is_ggml() {
                continue;
            }
            let (m, k) = (w.m, w.k);
            let mut seed = 0x9E3779B97F4A7C15u64 ^ (m as u64 * 31 + k as u64);
            let x: Vec<f32> = (0..k)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
                })
                .collect();
            let xg = gpu.upload_f32(&x, &[k]).unwrap();
            let yg = gpu.alloc_tensor(&[m], rdna_compute::DType::F32).unwrap();
            gpu.ggml_gemv(w.gpu_dtype, &w.buf, m, k, &xg, &yg, false)
                .unwrap();
            let y1 = gpu.download_f32(&yg).unwrap();
            let wf = gpu
                .alloc_tensor(&[m * k], rdna_compute::DType::F32)
                .unwrap();
            gpu.ggml_dequant_weight(w.gpu_dtype, &w.buf, m, k, &wf, Some(false))
                .unwrap();
            let wh = gpu.download_f32(&wf).unwrap();
            let (mut dot, mut na, mut nb, mut maxd) = (0f64, 0f64, 0f64, 0f64);
            for r in 0..m {
                let row = &wh[r * k..(r + 1) * k];
                let y2: f64 = row.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
                let y1r = y1[r] as f64;
                dot += y1r * y2;
                na += y1r * y1r;
                nb += y2 * y2;
                maxd = maxd.max((y1r - y2).abs());
            }
            println!(
                "DEQUANT {name} {:?} m={m} k={k} cos={:.7} max_abs_diff={maxd:.5} rms_y={:.4}",
                w.gpu_dtype,
                dot / (na.sqrt() * nb.sqrt()).max(1e-30),
                (nb / m as f64).sqrt()
            );
            if name != "output" {
                // Batched prefill GEMM (A16 route) on 3 rows against the same host reference.
                let nb_rows = 3usize;
                let mut x3 = Vec::with_capacity(nb_rows * k);
                for t in 0..nb_rows {
                    x3.extend(
                        x.iter()
                            .map(|v| v * (1.0 + t as f32 * 0.37) - t as f32 * 0.05),
                    );
                }
                let x3g = gpu.upload_f32(&x3, &[nb_rows * k]).unwrap();
                let y3g = gpu
                    .alloc_tensor(&[nb_rows * m], rdna_compute::DType::F32)
                    .unwrap();
                gpu.ggml_prefill_gemm(w.gpu_dtype, &w.buf, m, k, &x3g, &y3g, nb_rows)
                    .unwrap();
                let y3 = gpu.download_f32(&y3g).unwrap();
                let (mut dot3, mut na3, mut nb3, mut maxd3) = (0f64, 0f64, 0f64, 0f64);
                for t in 0..nb_rows {
                    for r in 0..m {
                        let row = &wh[r * k..(r + 1) * k];
                        let xr = &x3[t * k..(t + 1) * k];
                        let y2: f64 = row.iter().zip(xr).map(|(a, b)| *a as f64 * *b as f64).sum();
                        let y1r = y3[t * m + r] as f64;
                        dot3 += y1r * y2;
                        na3 += y1r * y1r;
                        nb3 += y2 * y2;
                        maxd3 = maxd3.max((y1r - y2).abs());
                    }
                }
                println!(
                    "PREFILL_GEMM {name} n={nb_rows} cos={:.7} max_abs_diff={maxd3:.5}",
                    dot3 / (na3.sqrt() * nb3.sqrt()).max(1e-30)
                );
                gpu.free_tensor(y3g).unwrap();
                gpu.free_tensor(x3g).unwrap();
            }
            gpu.free_tensor(wf).unwrap();
            gpu.free_tensor(yg).unwrap();
            gpu.free_tensor(xg).unwrap();
        }
        if let LayerWeights::DeltaNet(l) = &weights.layers[0] {
            for (name, w) in [("L0.w_beta", &l.w_beta), ("L0.w_alpha", &l.w_alpha)] {
                let (m, k) = (w.m, w.k);
                let nb_rows = 3usize;
                let x3: Vec<f32> = (0..nb_rows * k)
                    .map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0)
                    .collect();
                let x3g = gpu.upload_f32(&x3, &[nb_rows * k]).unwrap();
                let y3g = gpu
                    .alloc_tensor(&[nb_rows * m], rdna_compute::DType::F32)
                    .unwrap();
                gpu.gemm_f32_register_tiled(&w.buf, &x3g, &y3g, m, k, nb_rows)
                    .unwrap();
                let y3 = gpu.download_f32(&y3g).unwrap();
                let wh = gpu.download_f32(&w.buf).unwrap();
                let mut maxd = 0f64;
                for t in 0..nb_rows {
                    for r in 0..m {
                        let y2: f64 = wh[r * k..(r + 1) * k]
                            .iter()
                            .zip(&x3[t * k..(t + 1) * k])
                            .map(|(a, b)| *a as f64 * *b as f64)
                            .sum();
                        maxd = maxd.max((y3[t * m + r] as f64 - y2).abs());
                    }
                }
                println!(
                    "F32_GEMM {name} {:?} m={m} k={k} max_abs_diff={maxd:.6}",
                    w.gpu_dtype
                );
            }
        }
        return;
    }
    if !compare.is_empty() {
        // Batched prefill vs token-by-token decode on the same tokens: the
        // last-token logits must agree to rounding. Prints cosine, max abs
        // difference and the argmax of each path.
        let text = "The history of the Roman Empire spans many centuries, beginning with the rise of Augustus \
            and continuing through the crisis of the third century, the reforms of Diocletian and Constantine, \
            and the gradual decline of the western provinces. Historians debate the causes of the decline, \
            pointing to economic stress, military overreach, political instability and external pressure. \
            In the east the empire endured for another thousand years, preserving law, learning and art. ";
        let mut base: Vec<u32> = Vec::new();
        while base.len() < 2048 {
            base.extend(tokenizer.encode(text));
        }
        for &n in &compare {
            let toks = &base[..n];
            dn_state.reset(&mut gpu).expect("reset state");
            kv_cache.clear_gpu(&mut gpu).expect("clear kv");
            qwen35::forward_prefill_batch(
                &mut gpu,
                &weights,
                &config,
                toks,
                0,
                &mut kv_cache,
                &mut dn_state,
                &scratch,
                None,
                None,
                None,
                None,
            )
            .expect("prefill forward failed");
            let la = gpu.download_f32(&scratch.logits).unwrap();
            let conv_a = gpu.download_f32(&dn_state.conv_states[0]).unwrap();
            let sc_a = if dn_state.s_scales.is_empty() {
                Vec::new()
            } else {
                gpu.download_f32(&dn_state.s_scales[0]).unwrap()
            };
            dn_state.reset(&mut gpu).expect("reset state");
            kv_cache.clear_gpu(&mut gpu).expect("clear kv");
            for (p, &t) in toks.iter().enumerate() {
                qwen35::forward_scratch(
                    &mut gpu,
                    &weights,
                    &config,
                    t,
                    p,
                    &mut kv_cache,
                    &mut dn_state,
                    &scratch,
                )
                .expect("decode forward failed");
            }
            let lb = gpu.download_f32(&scratch.logits).unwrap();
            let conv_b = gpu.download_f32(&dn_state.conv_states[0]).unwrap();
            let sc_b = if dn_state.s_scales.is_empty() {
                Vec::new()
            } else {
                gpu.download_f32(&dn_state.s_scales[0]).unwrap()
            };
            let rel = |a: &[f32], b: &[f32]| -> (f32, f32) {
                let md = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                let mb = b.iter().map(|y| y.abs()).fold(0f32, f32::max);
                (md, md / mb.max(1e-30))
            };
            let (cd, cr) = rel(&conv_a, &conv_b);
            let (sd, sr) = rel(&sc_a, &sc_b);
            println!("STATE0 n={n} conv_max_abs={cd:.5} conv_rel={cr:.5} s_scales_max_abs={sd:.6} s_scales_rel={sr:.5}");
            if let Some(pbs) = scratch.prefill_batch.as_ref() {
                let last = |gpu: &mut rdna_compute::Gpu,
                            t: &rdna_compute::GpuTensor,
                            w: usize|
                 -> Vec<f32> {
                    let all = gpu.download_f32(t).unwrap();
                    all[(n - 1) * w..n * w].to_vec()
                };
                let nvh = config.linear_num_value_heads;
                let vd = nvh * config.linear_value_head_dim;
                let kd = config.linear_num_key_heads * config.linear_key_head_dim;
                let qkvd = 2 * kd + vd;
                let pairs: Vec<(&str, Vec<f32>, Vec<f32>)> = vec![
                    (
                        "beta",
                        last(&mut gpu, &pbs.dn_beta_batch, nvh),
                        gpu.download_f32(&scratch.dn_beta).unwrap(),
                    ),
                    (
                        "alpha",
                        last(&mut gpu, &pbs.dn_alpha_batch, nvh),
                        gpu.download_f32(&scratch.dn_alpha).unwrap(),
                    ),
                    (
                        "qkv_pre_conv",
                        last(&mut gpu, &pbs.dn_qkv_batch, qkvd),
                        gpu.download_f32(&scratch.dn_qkv).unwrap(),
                    ),
                    (
                        "z",
                        last(&mut gpu, &pbs.dn_z_batch, vd),
                        gpu.download_f32(&scratch.dn_z).unwrap(),
                    ),
                    (
                        "q",
                        last(&mut gpu, &pbs.dn_q_batch, vd),
                        gpu.download_f32(&scratch.dn_q).unwrap(),
                    ),
                    (
                        "k",
                        last(&mut gpu, &pbs.dn_k_batch, vd),
                        gpu.download_f32(&scratch.dn_k).unwrap(),
                    ),
                    (
                        "v",
                        last(&mut gpu, &pbs.dn_v_batch, vd),
                        gpu.download_f32(&scratch.dn_v).unwrap(),
                    ),
                    (
                        "attn_out",
                        last(&mut gpu, &pbs.dn_attn_out_batch, vd),
                        gpu.download_f32(&scratch.dn_attn_out).unwrap(),
                    ),
                    (
                        "normed",
                        last(&mut gpu, &pbs.dn_normed_batch, vd),
                        gpu.download_f32(&scratch.dn_normed).unwrap(),
                    ),
                ];
                for (name, a, b) in pairs {
                    let len = a.len().min(b.len());
                    let (md, mr) = rel(&a[..len], &b[..len]);
                    let mb = b.iter().take(len).map(|y| y.abs()).fold(0f32, f32::max);
                    println!("  INTERMEDIATE {name} len={len} max_abs={md:.5} rel_to_max={mr:.5} max_ref={mb:.4}");
                }
            }
            let (mut dot, mut na, mut nb, mut maxd) = (0f64, 0f64, 0f64, 0f32);
            for (a, b) in la.iter().zip(&lb) {
                dot += *a as f64 * *b as f64;
                na += (*a as f64).powi(2);
                nb += (*b as f64).powi(2);
                maxd = maxd.max((a - b).abs());
            }
            println!(
                "COMPARE n={n} layers={} cos={:.6} max_abs_diff={maxd:.4} argmax_prefill={} argmax_decode={}",
                config.n_layers,
                dot / (na.sqrt() * nb.sqrt()).max(1e-30),
                llama::argmax(&la),
                llama::argmax(&lb)
            );
        }
        return;
    }

    let mut out: Option<std::fs::File> = out_path
        .as_ref()
        .map(|p| std::fs::File::create(p).expect("create --out"));
    for (pi, prompt) in prompts.iter().enumerate() {
        let prompt_tokens: Vec<u32> = if chat {
            let mut t = Vec::new();
            for piece in ["<|im_start|>", "user", "\n"] {
                t.extend(tokenizer.encode(piece));
            }
            t.extend(tokenizer.encode(prompt));
            for piece in ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"] {
                t.extend(tokenizer.encode(piece));
            }
            t
        } else {
            tokenizer.encode(prompt)
        };
        dn_state.reset(&mut gpu).expect("reset state");
        kv_cache.clear_gpu(&mut gpu).expect("clear kv");
        let t_pre = Instant::now();
        qwen35::forward_prefill_batch(
            &mut gpu,
            &weights,
            &config,
            &prompt_tokens,
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            None,
            None,
            None,
        )
        .expect("prefill forward failed");
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        let pre_ms = t_pre.elapsed().as_secs_f64() * 1e3;
        let mut next = llama::argmax(&logits);
        let mut ids: Vec<u32> = vec![next];
        let mut pos = prompt_tokens.len();
        let t_dec = Instant::now();
        while ids.len() < max_tokens && next != config.eos_token && pos < kv_seq {
            qwen35::forward_scratch(
                &mut gpu,
                &weights,
                &config,
                next,
                pos,
                &mut kv_cache,
                &mut dn_state,
                &scratch,
            )
            .expect("decode forward failed");
            let logits = gpu.download_f32(&scratch.logits).unwrap();
            next = llama::argmax(&logits);
            ids.push(next);
            pos += 1;
        }
        let dec_s = t_dec.elapsed().as_secs_f64();
        let line = ids
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        println!("PROMPT {pi}: {prompt:?} ({} tokens)", prompt_tokens.len());
        println!("IDS {pi}: {line}");
        println!("TEXT {pi}: {:?}", tokenizer.decode(&ids));
        if timing {
            println!(
                "TIME {pi}: prefill {:.1} ms ({} tok), decode {} tok in {:.2} s = {:.2} tok/s",
                pre_ms,
                prompt_tokens.len(),
                ids.len() - 1,
                dec_s,
                (ids.len() - 1) as f64 / dec_s.max(1e-9)
            );
        }
        if let Some(f) = out.as_mut() {
            writeln!(f, "{line}").expect("write ids");
        }
    }
}
