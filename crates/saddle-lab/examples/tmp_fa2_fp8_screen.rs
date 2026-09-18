// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Stage-b real-data screen: fp8 (E4M3FN) QK + PV on raw pre-quant taps.
//!
//! Plan of record: `docs/plans/2026-09-18-gfx1201-fa2-stage-b.md` (§8, §10/S7).
//! This REPLACES the old screen staged math, which normalized P over the whole
//! sequence and used q8-dequantized V — its 31.8 % underflow is not a stage-b
//! predictor. Do not cite the old screen.
//!
//! # Data provenance
//!
//! Real Q, K, V from `qwen3.8-27b.mq4-xt` on gfx1201, driven through the
//! production prefill entry (`forward_prefill_batch_with_pbs_opts`,
//! truncated at L+1) in arch-native 384-row chunks. EVERY chunk's K/V is
//! tapped before the writer: `fa_k_batch`/`fa_v_batch` are the raw pre-quant
//! f32 rows the writer consumes, so concatenating all chunks gives raw
//! pre-quant K/V for every attended prefix row — never reconstructed by
//! dequantizing q8. The stored q8 cache is downloaded too, but only to
//! prove the taps (self-check A) and to feed the route-Q requant arm —
//! never as stage-b operand input.
//!
//! Per layer: raw Q/K/V (f32), positions, gate logits, engine O, and a
//! kernel differential (the CURRENT q8/f16 FA2 kernel re-run on the tapped
//! operands; must match the engine O bit-for-bit — same kernel, same
//! inputs — which proves the taps are the true kernel inputs).
//!
//! # Arms (all online KT16 order, f32 state/round points unless noted)
//!
//! - arm1 (kernel q8/f16): the production slice-A kernel re-run on the taps.
//!   Bit-identity gate: `--kernel-out DIR` writes its raw O + gate f32 files
//!   for the md5 check against the retained reference (raw-O
//!   `1f7a1183a751de7555066bcf5a1ca11c`, gate
//!   `8c0287afddd1a3aef7f43496c3127c23` on layer 35 / 4224 tokens /
//!   qstride 4). Arm 1 is the reference every other arm reports against.
//! - arm2 (cpu-q8ref): production q8/f16 route — stored-q8 blocks decoded to
//!   f16 round points, f16 Q pre-convert, f16 P. Must track arm 1 to
//!   WMMA-tolerance (self-check B).
//! - arm3 (q0): native fp8 KV + f16 arithmetic — K/V decoded as
//!   `f16(f32(s)*e4m3(code))`, otherwise arm2's round points. Format-only
//!   error; the Q0 predictor.
//! - arm4 (qkdiag): stage-b fp8 QK legs (`dot(Q8,K8)*sk*(sq*attn)`) with
//!   exact-f32 weighted PV (no P quant, no changing unit). Isolates QK
//!   error from PV error.
//! - arm5 (stageb-N): both fp8 legs with the bounded changing-unit O
//!   recurrence (`w=e*sv`, `bnew=max(max(w)/448,2^-64)`, `p8=RNE(w/bnew)`,
//!   `rho=(alpha*bprev)/bnew`, `Ofr*=rho; Ofr+=P8*V8`), route-N operands
//!   (native codes + row scales, plan §2.1 rule).
//! - arm6 (stageb-Q): same recurrence on route-Q operands: stored q8 blocks
//!   requantized per §2.3 (`s_row` = smallest f16 >= 127*max_b sf_b/448,
//!   floor 2^-24, all-zero row -> 1; code' = E4M3_RNE(c*(sf_b/s_row))),
//!   with the `448*f32(s_row) >= amax_row` invariant asserted per row.
//!   Q->e4m3 with f32 `sq` is shared by arms 4-6.
//!
//! Gate: the engine buffer may be pre- or post-gate at this HEAD, so a
//! probe compares the kernel differential against the engine O with and
//! without one sigmoid, and every CPU O arm is compared pre-gate (against
//! arm 1) plus gated-iff-engine-is-gated.
//!
//! # Tap census (plan §8.1(B) attribution tool)
//!
//! Per FA layer over the raw pre-quant taps: K/V `amax`, p99.9 of |x|, and
//! the fraction of dims clipped below the e4m3 subnormal floor under the
//! per-256 row scale (lost-to-zero and subnormal fractions separately).
//! `--census-only` runs taps + census without the CPU arms (also usable on
//! non-FA layers, where it reports taps as unpopulated).
//!
//! # Offline dump / replay
//!
//! `--dump DIR` writes per-layer operand bundles (raw Q/K/V, gate,
//! engine O, kernel O, stored q8 bytes, positions, manifest). `--replay DIR`
//! re-runs the codec self-test, edge cases, all CPU arms and metrics with
//! NO GPU and NO model. Replay a single layer dir or a parent of several.
//!
//! Usage (ordinal-1 convention for this unit):
//!   HOME=/home/kaden/.hipfire-homes/ab1 ROCR_VISIBLE_DEVICES=1 \
//!   HIPFIRE_KERNEL_CACHE=$HOME/.hipfire_kernels \
//!   cargo run -p saddle-lab --example tmp_fa2_fp8_screen -- \
//!     --model /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt \
//!     [--tokens 4224] [--layer 35] [--qstride 4] [--dump DIR]
//!     [--kernel-out DIR] [--census-only]
//!     [--prompt-ids FILE | --prompt-range START]
//!   cargo run -p saddle-lab --example tmp_fa2_fp8_screen -- --replay DIR \
//!     [--qstride 4]
//!
//! Prompt: default token ids are 0..N (the WT2-style screen prompt). The
//! screen accepts no text prompt (no tokenizer on this path); `--prompt-ids`
//! loads whitespace-separated u32 ids from a file, `--prompt-range START`
//! uses START..START+N. There is no ag-corpus token file in the receipts,
//! so an ag-corpus-faithful prompt cannot be run from here — reported as-is.
//!
//! Stop gate (plan §8.1/§11): stage-b P underflow <= 0.1 % of positive valid
//! weighted entries (count AND lost mass reported, per route). Informational;
//! KLD admits.

#[cfg(not(all(feature = "deltanet", feature = "arch-qwen35")))]
fn main() {
    eprintln!("build with --features deltanet,arch-qwen35");
}

#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn main() {
    use hipfire_arch_qwen35::qwen35::{
        self, DflashFusionCtx, DeltaNetState, LayerType, LayerWeights, PrefillBatchScratch,
        Qwen35Scratch,
    };
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use std::path::PathBuf;

    // ---------------- args ----------------
    let argv: Vec<String> = std::env::args().collect();
    let mut model = PathBuf::from("/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt");
    let mut n_tokens: usize = 4224; // multiple of CHUNK; 4224 = 11 full 384-row chunks
    let mut layer_arg = String::from("35");
    let mut qstride: usize = 4; // sample every 4th final-chunk query (96 queries)
    let mut dump_dir: Option<PathBuf> = None;
    let mut replay_dir: Option<PathBuf> = None;
    let mut kernel_out: Option<PathBuf> = None;
    let mut prompt_ids: Option<PathBuf> = None;
    let mut prompt_range: Option<u32> = None;
    let mut census_only = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--tokens" => {
                n_tokens = argv[i + 1].parse().expect("--tokens");
                i += 2;
            }
            "--layer" => {
                layer_arg = argv[i + 1].clone();
                i += 2;
            }
            "--qstride" => {
                qstride = argv[i + 1].parse().expect("--qstride");
                i += 2;
            }
            "--dump" => {
                dump_dir = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--replay" => {
                replay_dir = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--kernel-out" => {
                kernel_out = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt-ids" => {
                prompt_ids = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt-range" => {
                prompt_range = Some(argv[i + 1].parse().expect("--prompt-range"));
                i += 2;
            }
            "--census-only" => {
                census_only = true;
                i += 1;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: tmp_fa2_fp8_screen [--model path] [--tokens N] [--layer 35|early|mid|late|CSV] [--qstride S] [--dump DIR] [--kernel-out DIR] [--census-only] [--prompt-ids FILE | --prompt-range START] | --replay DIR [--qstride S]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    // SAFETY: single-threaded init phase; no other threads observing env.
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_KV_MODE", "q8");
    }

    // Codec self-test runs in BOTH capture and replay (no GPU needed).
    codec_self_test();
    // Pure-CPU recurrence edge cases (no GPU needed).
    edge_case_self_test();

    if let Some(rdir) = &replay_dir {
        run_replay(rdir, qstride);
        return;
    }

    const CHUNK: usize = 384; // gfx1201 arch chunk
    assert_eq!(n_tokens % CHUNK, 0, "N={n_tokens} must be a multiple of {CHUNK}");
    assert!(qstride >= 1);

    // ---------------- load model ----------------
    let mut hfq = HfqFile::open(&model).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    eprintln!(
        "model={} dim={} layers={} heads={} kv_heads={} head_dim={} norm_eps={}",
        model.display(),
        config.dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        config.norm_eps
    );
    let fa_layers: Vec<usize> = config
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == LayerType::FullAttention)
        .map(|(l, _)| l)
        .collect();
    eprintln!("full-attention layers: {fa_layers:?}");
    assert!(!fa_layers.is_empty(), "no FullAttention layers");
    let layers = resolve_layers(&layer_arg, &fa_layers, census_only);
    if census_only {
        for &l in &layers {
            assert!(
                l < config.n_layers,
                "layer {l} out of range (0..{})",
                config.n_layers
            );
        }
    }

    let (nh, nkv, hd) = (config.n_heads, config.n_kv_heads, config.head_dim);
    assert_eq!((nh, nkv, hd), (24, 4, 256), "FA2 gfx1201 needs H24/KV4/D256");
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    let row_stride = nkv * (hd / 32) * 34; // stored-q8 row stride (self-check A only)
    let gqa_group = nh / nkv;
    let scale_attn = 1.0f64 / (hd as f64).sqrt();

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("weights");

    // Prompt token ids: default 0..N (WT2-style); --prompt-ids loads u32 ids
    // from a file; --prompt-range START uses START..START+N. No text prompt:
    // no tokenizer exists on this path.
    let prompt_desc: String = if let Some(p) = &prompt_ids {
        assert!(prompt_range.is_none(), "--prompt-ids with --prompt-range");
        let txt = std::fs::read_to_string(p).expect("read prompt-ids");
        let ids: Vec<u32> = txt
            .split_whitespace()
            .map(|s| s.parse().unwrap_or_else(|_| panic!("bad prompt id: {s}")))
            .collect();
        assert!(ids.len() >= n_tokens, "prompt file has {} ids, need {n_tokens}", ids.len());
        eprintln!("prompt: {} ids from {}", ids.len(), p.display());
        format!("file:{}[..{n_tokens}]", p.display())
    } else if let Some(s) = prompt_range {
        eprintln!("prompt: range {s}..{}", s as usize + n_tokens);
        format!("range:{s}..{}", s as usize + n_tokens)
    } else {
        eprintln!("prompt: default 0..{n_tokens}");
        format!("range:0..{n_tokens}")
    };
    let tokens: Vec<u32> = if let Some(p) = &prompt_ids {
        let txt = std::fs::read_to_string(p).expect("read prompt-ids");
        txt.split_whitespace()
            .map(|s| s.parse().expect("prompt id"))
            .take(n_tokens)
            .collect()
    } else if let Some(s) = prompt_range {
        (s..s + n_tokens as u32).collect()
    } else {
        (0..n_tokens as u32).collect()
    };
    let kv_max = n_tokens + 16;
    let n_chunk = n_tokens / CHUNK;

    for &want in &layers {
        // FA membership: arms need dense FullAttn; census-only tolerates other
        // layer types (taps then report as unpopulated — see below).
        let is_fa = matches!(&weights.layers[want], LayerWeights::FullAttn(_));
        if !is_fa && !census_only {
            eprintln!("layer {want} is not dense FullAttn; skipping (use --census-only to tap it)");
            continue;
        }
        eprintln!(
            "=== screen layer L={want} ({}; prompt {prompt_desc}) ===",
            if is_fa { "dense FullAttn" } else { "non-FA, census-only" }
        );

        // Fresh cache + DN state per layer so layers never see each other.
        let mut kv_cache =
            KvCache::new_gpu_q8(&mut gpu, config.n_layers, nkv, hd, kv_max).expect("kv cache q8");
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn state");
        dn_state.reset(&mut gpu).expect("dn reset");
        let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");
        let pbs = PrefillBatchScratch::new(&mut gpu, &config, CHUNK).expect("pbs");

        // Per-chunk prefill; tap EVERY chunk's raw K/V (pre-writer f32).
        let mut q_all = vec![0.0f32; n_tokens * q_dim];
        let mut k_all = vec![0.0f32; n_tokens * kv_dim];
        let mut v_all = vec![0.0f32; n_tokens * kv_dim];
        let mut g_all = vec![0.0f32; n_tokens * q_dim];
        let mut o_all = vec![0.0f32; n_tokens * q_dim];
        let t0 = std::time::Instant::now();
        for c in 0..n_chunk {
            let start = c * CHUNK;
            qwen35::forward_prefill_batch_with_pbs_opts(
                &mut gpu,
                &weights,
                &config,
                &tokens[start..start + CHUNK],
                start,
                &mut kv_cache,
                &mut dn_state,
                &scratch,
                None,
                None,
                None,
                None,
                Some(&pbs),
                None,
                Some(want + 1),
                false,
                DflashFusionCtx::Off,
            )
            .unwrap_or_else(|e| panic!("chunk {c} prefill: {e:?}"));
            // Position check: chunk c must own absolute positions start..start+CHUNK.
            let pos_bytes = dtoh_bytes(&gpu, &pbs.positions.buf, CHUNK * 4, "positions");
            for r in 0..CHUNK {
                let p = i32::from_le_bytes(pos_bytes[r * 4..r * 4 + 4].try_into().unwrap());
                assert_eq!(p as usize, start + r, "chunk {c} position drift at row {r}");
            }
            let q = gpu.download_f32(&pbs.fa_q_batch).expect("fa_q");
            let k = gpu.download_f32(&pbs.fa_k_batch).expect("fa_k");
            let v = gpu.download_f32(&pbs.fa_v_batch).expect("fa_v");
            let g = gpu.download_f32(&pbs.fa_gate_batch).expect("fa_gate");
            let o = gpu.download_f32(&pbs.fa_attn_out_batch).expect("fa_attn_out");
            assert_eq!(q.len(), CHUNK * q_dim, "fa_q shape c={c}");
            assert_eq!(k.len(), CHUNK * kv_dim, "fa_k shape c={c}");
            assert_eq!(v.len(), CHUNK * kv_dim, "fa_v shape c={c}");
            q_all[start * q_dim..(start + CHUNK) * q_dim].copy_from_slice(&q);
            k_all[start * kv_dim..(start + CHUNK) * kv_dim].copy_from_slice(&k);
            v_all[start * kv_dim..(start + CHUNK) * kv_dim].copy_from_slice(&v);
            g_all[start * q_dim..(start + CHUNK) * q_dim].copy_from_slice(&g);
            o_all[start * q_dim..(start + CHUNK) * q_dim].copy_from_slice(&o);
        }
        eprintln!(
            "prefill tapped (layers 0..={want}, {n_chunk} chunks) in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        if !is_fa {
            // Non-FA layer under --census-only: this layer writes no FA K/V
            // taps (linear layers project into dn_qkv_batch instead). The
            // buffers below hold stale rows from the last FA layer <= want
            // in the truncated prefix — NOT this layer's K/V.
            let knz = k_all.iter().filter(|x| **x != 0.0).count();
            let vnz = v_all.iter().filter(|x| **x != 0.0).count();
            let kamax = k_all.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let vamax = v_all.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            eprintln!(
                "layer {want} is not FullAttention: no own FA K/V taps (buffers hold stale FA rows: K nonzero {knz}/{} amax={kamax:.3e}; V nonzero {vnz}/{} amax={vamax:.3e}); census N/A — use an FA layer for the §8.1(B) census",
                k_all.len(),
                v_all.len()
            );
            continue;
        }
        for (name, v) in [
            ("Q", &q_all),
            ("K", &k_all),
            ("V", &v_all),
            ("Oeng", &o_all),
            ("gate", &g_all),
        ] {
            assert!(v.iter().all(|x| x.is_finite()), "non-finite in tapped {name}");
        }
        // §8.1(B) attribution census on the raw pre-quant taps (K and V).
        tap_census("K", &k_all, n_tokens, nkv, hd, kv_dim);
        tap_census("V", &v_all, n_tokens, nkv, hd, kv_dim);
        if census_only {
            continue;
        }
        let k_stored = dtoh_bytes(
            &gpu,
            &kv_cache.k_gpu[want].buf,
            n_tokens * row_stride,
            "k_cache",
        );
        let v_stored = dtoh_bytes(
            &gpu,
            &kv_cache.v_gpu[want].buf,
            n_tokens * row_stride,
            "v_cache",
        );
        // Self-check A: stored-q8 blocks vs raw taps (proves block parse AND
        // that fa_k/fa_v are the true pre-writer rows).
        self_check_a(&k_all, &v_all, &k_stored, &v_stored, n_tokens, nkv, hd, kv_dim, row_stride);

        // Kernel differential on the FINAL chunk (Q from fa taps = still
        // resident in pbs after the last chunk; K/V = full prefix cache).
        let d_out = gpu
            .zeros(&[CHUNK * q_dim], rdna_compute::DType::F32)
            .expect("fa2 diff out");
        gpu.attention_q8_0_fa2_gqa_gfx1201(
            &pbs.fa_q_batch,
            &kv_cache.k_gpu[want],
            &kv_cache.v_gpu[want],
            &d_out,
            &pbs.positions,
            nh,
            nkv,
            hd,
            n_tokens,
            CHUNK,
        )
        .unwrap_or_else(|e| panic!("fa2 diff launch: {e:?}"));
        let o_kern = gpu.download_f32(&d_out).expect("fa2 diff O");
        assert_eq!(o_kern.len(), CHUNK * q_dim, "fa2 diff O shape");
        assert!(o_kern.iter().all(|x| x.is_finite()), "non-finite in fa2 diff O");

        // Gate probe: engine buffer pre- or post-gate?
        let fin0 = (n_tokens - CHUNK) * q_dim;
        let gated: Vec<f32> = o_kern
            .iter()
            .zip(g_all[fin0..fin0 + CHUNK * q_dim].iter())
            .map(|(o, g)| o * sigmoid(*g))
            .collect();
        let eng: &[f32] = &o_all[fin0..fin0 + CHUNK * q_dim];
        let d_raw = max_abs_f32(&o_kern, eng);
        let d_gated = max_abs_f32(&gated, eng);
        // Bit-exact tap proof needs the right gate side; report both.
        eprintln!(
            "gate probe L={want}: max|kern-eng| raw={d_raw:.3e} gated-once={d_gated:.3e}"
        );
        let engine_post_gate = d_gated < d_raw;
        let tap_gap = d_gated.min(d_raw);
        eprintln!(
            "engine buffer is {} (tap gap {tap_gap:.3e}; bit-exact=0 required for tap proof)",
            if engine_post_gate { "POST-gate" } else { "PRE-gate" }
        );
        // Arm-1 bit-identity bundle: raw kernel O + gate for the md5 gate.
        if let Some(kdir) = &kernel_out {
            let ldir = kdir.join(format!("layer{want}"));
            std::fs::create_dir_all(&ldir).expect("kernel-out layer dir");
            write_f32(&ldir.join("kern.O.f32"), &o_kern);
            write_f32(&ldir.join("kern.gate.f32"), &g_all[fin0..fin0 + CHUNK * q_dim]);
            std::fs::write(
                ldir.join("kern.meta.txt"),
                format!(
                    "layer={want} n_tokens={n_tokens} chunk={CHUNK} nh={nh} nkv={nkv} hd={hd} prompt={prompt_desc} engine_post_gate={engine_post_gate} tap_gap={tap_gap:.6e}\n"
                ),
            )
            .expect("kernel-out meta");
            eprintln!("arm-1 bundle (layer {want}) in {} — md5 against 1f7a1183a751de7555066bcf5a1ca11c / 8c0287afddd1a3aef7f43496c3127c23", ldir.display());
        }

        // Optional offline bundle.
        if let Some(ddir) = &dump_dir {
            let ldir = ddir.join(format!("layer{want}"));
            std::fs::create_dir_all(&ldir).expect("dump layer dir");
            write_f32(&ldir.join("q_final.f32"), &q_all[(n_tokens - CHUNK) * q_dim..]);
            write_f32(&ldir.join("k_raw.f32"), &k_all);
            write_f32(&ldir.join("v_raw.f32"), &v_all);
            write_f32(
                &ldir.join("gate_final.f32"),
                &g_all[(n_tokens - CHUNK) * q_dim..],
            );
            write_f32(
                &ldir.join("o_eng_final.f32"),
                &o_all[(n_tokens - CHUNK) * q_dim..],
            );
            write_f32(&ldir.join("o_kern_final.f32"), &o_kern);
            std::fs::write(ldir.join("k_q8.bin"), &k_stored).expect("dump k_q8");
            std::fs::write(ldir.join("v_q8.bin"), &v_stored).expect("dump v_q8");
            let pos_bytes = dtoh_bytes(&gpu, &pbs.positions.buf, CHUNK * 4, "positions");
            std::fs::write(ldir.join("positions.i32"), &pos_bytes).expect("dump positions");
            std::fs::write(
                ldir.join("manifest.json"),
                serde_json::json!({
                    "layer": want,
                    "n_tokens": n_tokens,
                    "chunk": CHUNK,
                    "nh": nh,
                    "nkv": nkv,
                    "hd": hd,
                    "qstride": qstride,
                    "prompt": prompt_desc,
                    "engine_post_gate": engine_post_gate,
                    "tap_gap": tap_gap,
                })
                .to_string(),
            )
            .expect("dump manifest");
            eprintln!("dumped layer {want} bundle to {}", ldir.display());
        }

        // CPU screen on final-chunk sampled queries (causal over full prefix).
        let bundle = CpuBundle {
            want,
            n_tokens,
            chunk: CHUNK,
            nh,
            nkv,
            hd,
            kv_dim,
            q_dim,
            gqa_group,
            scale_attn,
            row_stride,
            qstride,
            engine_post_gate,
            q_final: q_all[(n_tokens - CHUNK) * q_dim..].to_vec(),
            k_raw: k_all.clone(),
            v_raw: v_all.clone(),
            gate_final: g_all[(n_tokens - CHUNK) * q_dim..].to_vec(),
            o_kern: o_kern.clone(),
            k_stored: k_stored.clone(),
            v_stored: v_stored.clone(),
        };
        run_cpu_screen(&bundle);
    }

    // ---------------- helpers (inner fns: no env capture) ----------------
    fn dtoh_bytes(
        gpu: &rdna_compute::Gpu,
        buf: &hip_bridge::DeviceBuffer,
        n: usize,
        tag: &str,
    ) -> Vec<u8> {
        let mut out = vec![0u8; n];
        gpu.hip
            .memcpy_dtoh(&mut out, buf)
            .unwrap_or_else(|e| panic!("dtoh {tag}: {e:?}"));
        out
    }
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    fn max_abs_f32(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| ((*x - *y).abs()) as f64)
            .fold(0.0, f64::max)
    }
    fn write_f32(path: &std::path::Path, v: &[f32]) {
        let b: &[u8] = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
        };
        std::fs::write(path, b).expect("write f32 blob");
    }
    fn resolve_layers(arg: &str, fa_layers: &[usize], allow_any: bool) -> Vec<usize> {
        let mut out = Vec::new();
        for item in arg.split(',') {
            let item = item.trim();
            let l = match item {
                "early" => fa_layers[0],
                "mid" => fa_layers[fa_layers.len() / 2],
                "late" => fa_layers[fa_layers.len() - 1],
                _ => item.parse().unwrap_or_else(|_| panic!("bad --layer item: {item}")),
            };
            if !allow_any {
                assert!(
                    fa_layers.contains(&l),
                    "layer {l} is not FullAttention per config"
                );
            }
            if !out.contains(&l) {
                out.push(l);
            }
        }
        assert!(!out.is_empty(), "--layer selected nothing");
        out
    }
}

#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
    /// Self-check A: stored q8_0 blocks vs raw pre-writer taps, over every
    /// prefix row. Tolerance accounts for the write kernel's f32
    /// `val*(127/amax)` vs stored f16(amax/127) tie ambiguity (half-ULP of
    /// the block scale plus f16 rounding of the tap).
    fn self_check_a(
        k_all: &[f32],
        v_all: &[f32],
        k_stored: &[u8],
        v_stored: &[u8],
        n_tokens: usize,
        nkv: usize,
        hd: usize,
        kv_dim: usize,
        row_stride: usize,
    ) {
        let blks = hd / 32;
        let mut worst = 0.0f64;
        let mut tie_n = 0u64;
        for (all, stored, tag) in [(k_all, k_stored, "K"), (v_all, v_stored, "V")] {
            for p in 0..n_tokens {
                for h in 0..nkv {
                    for b in 0..blks {
                        let ko = p * row_stride + (h * blks + b) * 34;
                        let sf = f16_to_f32(u16::from_le_bytes([stored[ko], stored[ko + 1]]));
                        assert!(sf.is_finite(), "non-finite stored scale");
                        for j in 0..32 {
                            let d = h * hd + b * 32 + j;
                            let wv = (sf as f64) * (stored[ko + 2 + j] as i8 as f64);
                            let got = all[p * kv_dim + d] as f64;
                            let diff = (got - wv).abs();
                            worst = worst.max(diff);
                            let tol = 0.5 * (sf as f64) * (1.0 + 1e-6)
                                + 2.0f64.powi(-9) * wv.abs().max(1e-6);
                            if diff > tol {
                                // Tie-adjacent write rounding: allow 0.75*sf.
                                let qm = got / (sf as f64);
                                let tie = ((qm - qm.floor()).abs() - 0.5).abs();
                                assert!(
                                    tie <= qm.abs() * 2.0f64.powi(-9) + 1e-6
                                        && diff <= 0.75 * (sf as f64) + 2.0f64.powi(-9) * wv.abs().max(1e-6),
                                    "{tag} unexplained p={p} d={d}: cache={wv} tap={got} sf={sf}"
                                );
                                tie_n += 1;
                            }
                        }
                    }
                }
            }
        }
        eprintln!("self-check A (stored-q8 vs raw taps, all {n_tokens} rows): worst |diff|={worst:.3e} tie-adjacent={tie_n} — taps OK");
    }
// ================= replay (GPU-free) =================
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn run_replay(dir: &std::path::Path, qstride: usize) {
    // Single layer dir (has manifest.json) or a parent of several.
    let mut layers = Vec::new();
    if dir.join("manifest.json").exists() {
        layers.push(dir.to_path_buf());
    } else {
        let mut subs: Vec<_> = std::fs::read_dir(dir)
            .expect("read replay dir")
            .map(|e| e.expect("replay entry").path())
            .filter(|p| p.join("manifest.json").exists())
            .collect();
        subs.sort();
        layers = subs;
    }
    assert!(!layers.is_empty(), "no layer bundles under {}", dir.display());
    for ldir in layers {
        let manifest =
            std::fs::read_to_string(ldir.join("manifest.json")).expect("read manifest");
        let m: serde_json::Value = serde_json::from_str(&manifest).expect("parse manifest");
        let get = |k: &str| m[k].as_u64().unwrap_or_else(|| panic!("manifest lacks {k}")) as usize;
        let want = get("layer");
        let n_tokens = get("n_tokens");
        let chunk = get("chunk");
        let nh = get("nh");
        let nkv = get("nkv");
        let hd = get("hd");
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let read_f32 = |name: &str| -> Vec<f32> {
            let b = std::fs::read(ldir.join(name)).expect("read blob");
            assert_eq!(b.len() % 4, 0, "{name} size");
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let bundle = CpuBundle {
            want,
            n_tokens,
            chunk,
            nh,
            nkv,
            hd,
            kv_dim,
            q_dim,
            gqa_group: nh / nkv,
            scale_attn: 1.0 / (hd as f64).sqrt(),
            row_stride: nkv * (hd / 32) * 34,
            qstride,
            engine_post_gate: m["engine_post_gate"].as_bool().unwrap_or(false),
            q_final: read_f32("q_final.f32"),
            k_raw: read_f32("k_raw.f32"),
            v_raw: read_f32("v_raw.f32"),
            gate_final: read_f32("gate_final.f32"),
            o_kern: read_f32("o_kern_final.f32"),
            k_stored: std::fs::read(ldir.join("k_q8.bin")).expect("read k_q8"),
            v_stored: std::fs::read(ldir.join("v_q8.bin")).expect("read v_q8"),
        };
        self_check_a(&bundle.k_raw, &bundle.v_raw, &bundle.k_stored, &bundle.v_stored, n_tokens, nkv, hd, kv_dim, bundle.row_stride);
        assert_eq!(bundle.q_final.len(), chunk * q_dim, "q_final shape");
        eprintln!("--- replay {} (layer {want}) ---", ldir.display());
        tap_census("K", &bundle.k_raw, n_tokens, nkv, hd, kv_dim);
        tap_census("V", &bundle.v_raw, n_tokens, nkv, hd, kv_dim);
        run_cpu_screen(&bundle);
    }
}

// ================= CPU screen =================
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
struct CpuBundle {
    want: usize,
    n_tokens: usize,
    chunk: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    kv_dim: usize,
    q_dim: usize,
    gqa_group: usize,
    scale_attn: f64,
    row_stride: usize,
    qstride: usize,
    engine_post_gate: bool,
    q_final: Vec<f32>,
    k_raw: Vec<f32>,
    v_raw: Vec<f32>,
    gate_final: Vec<f32>,
    o_kern: Vec<f32>,
    k_stored: Vec<u8>,
    v_stored: Vec<u8>,
}

/// Full CPU screen: planes, six arms, metrics, underflow censuses.
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn run_cpu_screen(b: &CpuBundle) {
    let t1 = std::time::Instant::now();
    // Sampled queries from the final chunk (causal over all N keys).
    let qrows: Vec<usize> = (0..b.chunk).step_by(b.qstride).collect();
    let nq = qrows.len();
    eprintln!(
        "CPU screen L={} N={} queries={nq} (stride {}) x {} heads x up to {} keys x {} dims",
        b.want, b.n_tokens, b.qstride, b.nh, b.n_tokens, b.hd
    );

    // ---- planes ----
    // arm2: stored-q8 blocks -> f16 round points (production operands).
    let mut k1 = vec![0.0f64; b.n_tokens * b.kv_dim];
    let mut v1 = vec![0.0f64; b.n_tokens * b.kv_dim];
    parse_q8_plane(&b.k_stored, &mut k1, b.n_tokens, b.nkv, b.hd, b.kv_dim, b.row_stride);
    parse_q8_plane(&b.v_stored, &mut v1, b.n_tokens, b.nkv, b.hd, b.kv_dim, b.row_stride);
    // arms 3-5: native fp8 from RAW taps (plan §2 smallest-f16->= rule, f32-faithful).
    let mut k_codes = vec![0u8; b.n_tokens * b.kv_dim];
    let mut k_sc = vec![0.0f32; b.n_tokens * b.nkv];
    let mut v_codes = vec![0u8; b.n_tokens * b.kv_dim];
    let mut v_sc = vec![0.0f32; b.n_tokens * b.nkv];
    for p in 0..b.n_tokens {
        for h in 0..b.nkv {
            let kr = &b.k_raw[p * b.kv_dim + h * b.hd..p * b.kv_dim + (h + 1) * b.hd];
            let (s, c) = fp8_encode_row(kr).unwrap_or_else(|e| panic!("K encode p={p} h={h}: {e}"));
            k_sc[p * b.nkv + h] = s;
            k_codes[p * b.kv_dim + h * b.hd..p * b.kv_dim + (h + 1) * b.hd].copy_from_slice(&c);
            let vr = &b.v_raw[p * b.kv_dim + h * b.hd..p * b.kv_dim + (h + 1) * b.hd];
            let (s, c) = fp8_encode_row(vr).unwrap_or_else(|e| panic!("V encode p={p} h={h}: {e}"));
            v_sc[p * b.nkv + h] = s;
            v_codes[p * b.kv_dim + h * b.hd..p * b.kv_dim + (h + 1) * b.hd].copy_from_slice(&c);
        }
    }
    // arm6 (route Q): stored q8 blocks -> e4m3 requant per §2.3 (header-only
    // s_row; 448*f32(s_row) >= amax_row asserted per row inside).
    let mut qk_codes = vec![0u8; b.n_tokens * b.kv_dim];
    let mut qk_sc = vec![0.0f32; b.n_tokens * b.nkv];
    let mut qv_codes = vec![0u8; b.n_tokens * b.kv_dim];
    let mut qv_sc = vec![0.0f32; b.n_tokens * b.nkv];
    {
        let blks = b.hd / 32;
        let mut smin = f32::INFINITY;
        let mut smax = 0.0f32;
        for (stored, codes, sc, tag) in [
            (&b.k_stored, &mut qk_codes, &mut qk_sc, "K"),
            (&b.v_stored, &mut qv_codes, &mut qv_sc, "V"),
        ] {
            for p in 0..b.n_tokens {
                for h in 0..b.nkv {
                    let base = p * b.row_stride + h * blks * 34;
                    let mut scales = [0.0f32; 8];
                    let mut icodes = [0i8; 256];
                    for blk in 0..blks {
                        let ko = base + blk * 34;
                        scales[blk] =
                            f16_to_f32(u16::from_le_bytes([stored[ko], stored[ko + 1]]));
                        for j in 0..32 {
                            icodes[blk * 32 + j] = stored[ko + 2 + j] as i8;
                        }
                    }
                    let (s, c) = requant_q8_row(&scales, &icodes)
                        .unwrap_or_else(|e| panic!("{tag} requant p={p} h={h}: {e}"));
                    smin = smin.min(s);
                    smax = smax.max(s);
                    sc[p * b.nkv + h] = s;
                    codes[p * b.kv_dim + h * b.hd..p * b.kv_dim + (h + 1) * b.hd]
                        .copy_from_slice(&c);
                }
            }
        }
        eprintln!("route-Q s_row range over {n} rows x {g} heads: [{smin:.3e}, {smax:.3e}] (f16 grid, 448*s covers amax_row per-row asserted)", n = b.n_tokens, g = b.nkv);
    }
    // Q planes per sampled query/head: f16 (paths 1-2) and fp8+sq (paths 3-4).
    let mut q1 = vec![0.0f64; nq * b.nh * b.hd];
    let mut q_codes = vec![0u8; nq * b.nh * b.hd];
    let mut q_sq = vec![0.0f32; nq * b.nh];
    for (qi, &r) in qrows.iter().enumerate() {
        for h in 0..b.nh {
            let mut am = 0.0f32;
            for d in 0..b.hd {
                let qv = b.q_final[r * b.q_dim + h * b.hd + d];
                q1[(qi * b.nh + h) * b.hd + d] =
                    f16_to_f64(f32_to_f16_bits(qv));
                am = am.max(qv.abs());
            }
            // §3: sq = amax/448 in f32, zero row sq=1 (NOT f16-snapped).
            let sq = if am == 0.0 { 1.0 } else { am / 448.0 };
            q_sq[qi * b.nh + h] = sq;
            for d in 0..b.hd {
                let qv = b.q_final[r * b.q_dim + h * b.hd + d];
                q_codes[(qi * b.nh + h) * b.hd + d] = e4m3_encode((qv / sq) as f64);
            }
        }
    }

    // ---- per-(query,head) online arms ----
    let mut o_ref = vec![0.0f64; nq * b.nh * b.hd];
    let mut o_q8 = vec![0.0f64; nq * b.nh * b.hd];
    let mut o_q0 = vec![0.0f64; nq * b.nh * b.hd];
    let mut o_qk = vec![0.0f64; nq * b.nh * b.hd];
    let mut o_sb = vec![0.0f64; nq * b.nh * b.hd];
    let mut o_rq = vec![0.0f64; nq * b.nh * b.hd];
    let mut s_err_q0 = 0.0f64;
    let mut s_err_qk = 0.0f64;
    let mut s_err_sb = 0.0f64;
    let mut s_err_rq = 0.0f64;
    // arm5 (route N) P-underflow census over valid keys.
    let mut w_pos: u64 = 0;
    let mut w_zero: u64 = 0;
    let mut w_mass: f64 = 0.0;
    let mut w_lost: f64 = 0.0;
    // arm6 (route Q) P-underflow census over valid keys.
    let mut q_pos: u64 = 0;
    let mut q_zero: u64 = 0;
    let mut q_mass: f64 = 0.0;
    let mut q_lost: f64 = 0.0;
    let mut exp_zero: u64 = 0; // true f32 exp zeros (e underflows f32)
    let mut zero_v_heads: u64 = 0; // (q,h) with all-zero V codes on valid keys
    let mut per_q_worst = vec![0.0f64; nq];

    for (qi, &r) in qrows.iter().enumerate() {
        let pos = b.n_tokens - b.chunk + r;
        let nk = pos + 1;
        for h in 0..b.nh {
            let kh = h / b.gqa_group;
            let qo = (qi * b.nh + h) * b.hd;
            // V-head zero census input.
            let mut v_head_nz = false;
            for k in 0..nk {
                let ko = k * b.kv_dim + kh * b.hd;
                for d in 0..b.hd {
                    if v_codes[ko + d] != 0 {
                        v_head_nz = true;
                        break;
                    }
                }
                if v_head_nz {
                    break;
                }
            }
            if !v_head_nz {
                zero_v_heads += 1;
            }
            // Online KT16 blocks over valid keys.
            let nblk = nk.div_ceil(16);
            // path state: (m, l) + O accumulators.
            let mut m0 = f64::NEG_INFINITY;
            let mut l0 = 0.0;
            let mut m1 = f32::NEG_INFINITY as f64;
            let mut l1 = 0.0f64;
            let mut m2 = f32::NEG_INFINITY as f64;
            let mut l2 = 0.0f64;
            let mut m3 = f32::NEG_INFINITY as f64;
            let mut l3 = 0.0f64;
            let mut m4 = f32::NEG_INFINITY as f64;
            let mut l4 = 0.0f64;
            let mut o0 = vec![0.0f64; b.hd];
            let mut o1 = vec![0.0f64; b.hd];
            let mut o2 = vec![0.0f64; b.hd];
            let mut o3 = vec![0.0f64; b.hd];
            // arm5 changing-unit state.
            let mut ofr = vec![0.0f32; b.hd];
            let mut bprev = 1.0f32;
            // arm6 (route Q) state: same recurrence on requantized operands.
            let mut m5 = f32::NEG_INFINITY as f64;
            let mut l5 = 0.0f64;
            let mut o5 = vec![0.0f64; b.hd];
            let mut ofrq = vec![0.0f32; b.hd];
            let mut bprevq = 1.0f32;
            let sq = q_sq[qi * b.nh + h];
            let t_q = (sq * b.scale_attn as f32) as f64; // sq*attn in f32 (§3)
            for blk in 0..nblk {
                let ks = blk * 16;
                let ke = (ks + 16).min(nk);
                // scores for this block.
                let mut s0 = vec![0.0f64; ke - ks];
                let mut s1 = vec![0.0f64; ke - ks];
                let mut s2 = vec![0.0f64; ke - ks];
                let mut s3 = vec![0.0f64; ke - ks];
                let mut s4 = vec![0.0f64; ke - ks];
                let mut s5 = vec![0.0f64; ke - ks];
                for (j, k) in (ks..ke).enumerate() {
                    let ko = k * b.kv_dim + kh * b.hd;
                    // arm1: exact f64 from raw.
                    let mut d0 = 0.0;
                    // arm2: f16 Q/K.
                    let mut d1 = 0.0;
                    // arm3/5 K: f16(s*code).
                    let sk = k_sc[k * b.nkv + kh];
                    let mut d2 = 0.0;
                    // path3/4 QK leg: integer-grid dot.
                    let mut acc: f64 = 0.0;
                    for d in 0..b.hd {
                        let qf = b.q_final[r * b.q_dim + h * b.hd + d] as f64;
                        let kf = b.k_raw[ko + d] as f64;
                        d0 += qf * kf;
                        d1 += q1[qo + d] * k1[ko + d];
                        d2 += q1[qo + d] * fp8_decode_q0(sk, k_codes[ko + d]);
                        acc += e4m3_decode(q_codes[qo + d])
                            * e4m3_decode(k_codes[ko + d]);
                    }
                    s0[j] = d0 * b.scale_attn;
                    s1[j] = f32r(d1 * b.scale_attn);
                    s2[j] = f32r(d2 * b.scale_attn);
                    // §3: score = acc * sk * (sq * attn), left-assoc f32.
                    let s3v = f32r(f32r(acc * (sk as f64)) * t_q);
                    s3[j] = s3v;
                    s4[j] = s3v;
                    // arm6 (route Q): same QK leg on requantized operands.
                    let skq = qk_sc[k * b.nkv + kh];
                    let mut accq: f64 = 0.0;
                    for d in 0..b.hd {
                        accq += e4m3_decode(q_codes[qo + d])
                            * e4m3_decode(qk_codes[ko + d]);
                    }
                    let s5v = f32r(f32r(accq * (skq as f64)) * t_q);
                    s5[j] = s5v;
                    s_err_rq = s_err_rq.max((s5[j] - s0[j]).abs());
                    s_err_q0 = s_err_q0.max((s2[j] - s0[j]).abs());
                    s_err_qk = s_err_qk.max((s3[j] - s0[j]).abs());
                    s_err_sb = s_err_sb.max((s4[j] - s0[j]).abs());
                }
                // block max + online updates per path.
                let bmax0 = s0.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let bmax1 = s1.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let bmax2 = s2.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let bmax3 = s3.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let bmax4 = s4.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let bmax5 = s5.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                // arm1 exact.
                let (nm0, a0) = online_step(m0, bmax0);
                let mut nl0 = a0 * l0;
                // f32-rounded arms.
                let (nm1, a1) = online_step_f32(m1, bmax1);
                let mut nl1 = f32r(a1 * l1);
                let (nm2, a2) = online_step_f32(m2, bmax2);
                let mut nl2 = f32r(a2 * l2);
                let (nm3, a3) = online_step_f32(m3, bmax3);
                let mut nl3 = f32r(a3 * l3);
                let (nm4, a4) = online_step_f32(m4, bmax4);
                let (nm5, a5) = online_step_f32(m5, bmax5);
                // arm5: w = e*sv per key, block bnew, rho, Ofr update.
                // l's denominator sums UNWEIGHTED e (§5); accumulate it here.
                let mut w4 = vec![0.0f32; ke - ks];
                let mut emax = 0.0f32;
                let mut l4e = 0.0f64;
                for (j, k) in (ks..ke).enumerate() {
                    let e4 = f32r((s4[j] - nm4).exp()) as f32;
                    if e4 == 0.0 && s4[j].is_finite() {
                        exp_zero += 1;
                    }
                    l4e = f32r(l4e + e4 as f64);
                    let sv = v_sc[k * b.nkv + kh];
                    let w = e4 * sv;
                    w4[j] = w;
                    emax = emax.max(w);
                }
                // bnew = max(max(w)/448, 2^-64), all f32.
                let bnew = (emax / 448.0f32).max(2.0f32.powi(-64));
                let rho = f32r(((a4 as f32 * bprev) / bnew) as f64);
                let rho = f32r(rho);
                // arm6: wq = eq*svq per key, block bnewq, rhoq.
                let mut w5 = vec![0.0f32; ke - ks];
                let mut emaxq = 0.0f32;
                let mut l5e = 0.0f64;
                for (j, k) in (ks..ke).enumerate() {
                    let e5 = f32r((s5[j] - nm5).exp()) as f32;
                    l5e = f32r(l5e + e5 as f64);
                    let svq = qv_sc[k * b.nkv + kh];
                    let wq = e5 * svq;
                    w5[j] = wq;
                    emaxq = emaxq.max(wq);
                }
                let bnewq = (emaxq / 448.0f32).max(2.0f32.powi(-64));
                let rhoq = f32r(((a5 as f32 * bprevq) / bnewq) as f64);
                let rhoq = f32r(rhoq);
                // rescale + PV per path.
                for d in 0..b.hd {
                    o0[d] = o0[d] * a0;
                    o1[d] = f32r(o1[d] * a1);
                    o2[d] = f32r(o2[d] * a2);
                    o3[d] = f32r(o3[d] * a3);
                }
                for d in 0..b.hd {
                    ofr[d] *= rho as f32;
                }
                for d in 0..b.hd {
                    o5[d] = f32r(o5[d] * a5);
                }
                for d in 0..b.hd {
                    ofrq[d] *= rhoq as f32;
                }
                for (j, k) in (ks..ke).enumerate() {
                    let ko = k * b.kv_dim + kh * b.hd;
                    let e0 = (s0[j] - nm0).exp();
                    let e1 = f32r((s1[j] - nm1).exp());
                    let e2 = f32r((s2[j] - nm2).exp());
                    let e3 = f32r((s3[j] - nm3).exp());
                    nl0 += e0;
                    nl1 = f32r(nl1 + e1);
                    nl2 = f32r(nl2 + e2);
                    nl3 = f32r(nl3 + e3);
                    let p0 = e0;
                    let p1 = rne_f16_f64(e1);
                    let p2 = rne_f16_f64(e2);
                    // arm4: exact-f32 weighted PV (no P quant).
                    let sv3 = v_sc[k * b.nkv + kh] as f64;
                    // arm5: p8 grid.
                    let p8 = e4m3_decode(e4m3_encode((w4[j] / bnew) as f64));
                    // underflow census: positive finite w -> zero p8.
                    if w4[j] > 0.0 && w4[j].is_finite() {
                        w_pos += 1;
                        w_mass += w4[j] as f64;
                        if p8 == 0.0 {
                            w_zero += 1;
                            w_lost += w4[j] as f64;
                        }
                    }
                    // arm6: e5/p8q on the route-Q recurrence.
                    let e5 = f32r((s5[j] - nm5).exp());
                    let svq5 = qv_sc[k * b.nkv + kh] as f64;
                    let p8q = e4m3_decode(e4m3_encode((w5[j] / bnewq) as f64));
                    // arm6 underflow census: positive finite wq -> zero p8q.
                    if w5[j] > 0.0 && w5[j].is_finite() {
                        q_pos += 1;
                        q_mass += w5[j] as f64;
                        if p8q == 0.0 {
                            q_zero += 1;
                            q_lost += w5[j] as f64;
                        }
                    }
                    for d in 0..b.hd {
                        o0[d] += p0 * (b.v_raw[ko + d] as f64);
                        o1[d] = f32r(o1[d] + p1 * v1[ko + d]);
                        o2[d] = f32r(o2[d] + p2 * fp8_decode_q0(v_sc[k * b.nkv + kh], v_codes[ko + d]));
                        let vdiag = sv3 * e4m3_decode(v_codes[ko + d]);
                        o3[d] = f32r(o3[d] + e3 * vdiag);
                        let vdiagq = svq5 * e4m3_decode(qv_codes[ko + d]);
                        o5[d] = f32r(o5[d] + e5 * vdiagq);
                        ofrq[d] += (p8q * e4m3_decode(qv_codes[ko + d])) as f32;
                        ofr[d] += (p8 * e4m3_decode(v_codes[ko + d])) as f32;
                    }
                }
                // round Ofr block accumulation to f32 (documented point).
                for d in 0..b.hd {
                    ofr[d] = f32r(ofr[d] as f64) as f32;
                }
                for d in 0..b.hd {
                    ofrq[d] = f32r(ofrq[d] as f64) as f32;
                }
                m0 = nm0;
                l0 = nl0;
                m1 = nm1;
                l1 = nl1;
                m2 = nm2;
                l2 = nl2;
                m3 = nm3;
                l3 = nl3;
                m4 = nm4;
                l4 = f32r(a4 * l4 + l4e);
                bprev = bnew;
                m5 = nm5;
                l5 = f32r(a5 * l5 + l5e);
                bprevq = bnewq;
            }
            // completion: divide by l (arms 1-4); arms 5-6: Ofr*(bprev/l).
            for d in 0..b.hd {
                o_ref[qo + d] = if l0 == 0.0 { 0.0 } else { o0[d] / l0 };
                o_q8[qo + d] = if l1 == 0.0 { 0.0 } else { f32r(o1[d] / l1) };
                o_q0[qo + d] = if l2 == 0.0 { 0.0 } else { f32r(o2[d] / l2) };
                o_qk[qo + d] = if l3 == 0.0 { 0.0 } else { f32r(o3[d] / l3) };
                let osb = if l4 == 0.0 {
                    0.0
                } else {
                    f32r((ofr[d] * (bprev / l4 as f32)) as f64)
                };
                o_sb[qo + d] = osb;
                let orq = if l5 == 0.0 {
                    0.0
                } else {
                    f32r((ofrq[d] * (bprevq / l5 as f32)) as f64)
                };
                o_rq[qo + d] = orq;
            }
        }
    }
    eprintln!("CPU math done in {:.1}s", t1.elapsed().as_secs_f64());

    // ---- self-check B: arm2 (q8/f16 online) vs arm 1 (kernel differential) ----
    // Both pre-gate. Tolerance covers WMMA-f32 vs host-f64 reduction order.
    let mut bmax = 0.0f64;
    for (qi, &r) in qrows.iter().enumerate() {
        for h in 0..b.nh {
            for d in 0..b.hd {
                let a = o_q8[(qi * b.nh + h) * b.hd + d];
                let k = b.o_kern[r * b.q_dim + h * b.hd + d] as f64;
                bmax = bmax.max((a - k).abs());
            }
        }
    }
    eprintln!("self-check B (CPU-q8ref O vs kernel differential, pre-gate): max-abs={bmax:.3e}");

    // ---- metrics vs arm 1 (kernel differential, pre-gate) and vs exact ----
    let mut mags: Vec<f64> = o_ref.iter().map(|x| x.abs()).collect();
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t99 = mags[(mags.len() as f64 * 0.99) as usize];
    eprintln!("|O_ref|: p99 threshold T={t99:.6e}, max={:.6e}", mags[mags.len() - 1]);
    // per-query worst (arm5 vs ref) for outlier triage.
    for qi in 0..nq {
        let mut w = 0.0f64;
        for h in 0..b.nh {
            for d in 0..b.hd {
                w = w.max((o_sb[(qi * b.nh + h) * b.hd + d] - o_ref[(qi * b.nh + h) * b.hd + d]).abs());
            }
        }
        per_q_worst[qi] = w;
    }
    // kernel O covers the final chunk densely; compare on sampled rows only.
    let kern_f64: Vec<f64> = b.o_kern.iter().map(|x| *x as f64).collect();
    let mut kern_sampled = vec![0.0f64; nq * b.nh * b.hd];
    for (qi, &r) in qrows.iter().enumerate() {
        for h in 0..b.nh {
            for d in 0..b.hd {
                kern_sampled[(qi * b.nh + h) * b.hd + d] =
                    kern_f64[r * b.q_dim + h * b.hd + d];
            }
        }
    }
    // All CPU arms are sampled-only already; tail-1% keys off |O_ref| >= t99.
    let metric_vs = |o: &[f64], rf: &[f64]| -> (f64, f64) {
        let mut max_abs: f64 = 0.0;
        let mut tail_sum = 0.0;
        let mut tail_n = 0u64;
        for (idx, &x) in o.iter().enumerate() {
            let e = (x - rf[idx]).abs();
            max_abs = max_abs.max(e);
            if o_ref[idx].abs() >= t99 {
                tail_sum += e;
                tail_n += 1;
            }
        }
        (max_abs, tail_sum / (tail_n.max(1) as f64))
    };
    let (q8_a1, q8_a1t) = metric_vs(&o_q8, &kern_sampled);
    let (q0_a1, q0_a1t) = metric_vs(&o_q0, &kern_sampled);
    let (qk_a1, qk_a1t) = metric_vs(&o_qk, &kern_sampled);
    let (sb_a1, sb_a1t) = metric_vs(&o_sb, &kern_sampled);
    let (rq_a1, rq_a1t) = metric_vs(&o_rq, &kern_sampled);
    let (kk_ex, kk_ext) = metric_vs(&kern_sampled, &o_ref);
    let (q8_ex, q8_ext) = metric_vs(&o_q8, &o_ref);
    let (q0_ex, q0_ext) = metric_vs(&o_q0, &o_ref);
    let (qk_ex, qk_ext) = metric_vs(&o_qk, &o_ref);
    let (sb_ex, sb_ext) = metric_vs(&o_sb, &o_ref);
    let (rq_ex, rq_ext) = metric_vs(&o_rq, &o_ref);
    let u_pct = 100.0 * w_zero as f64 / (w_pos.max(1) as f64);
    let lost_pct = 100.0 * w_lost / w_mass.max(1e-30);
    let uq_pct = 100.0 * q_zero as f64 / (q_pos.max(1) as f64);
    let lostq_pct = 100.0 * q_lost / q_mass.max(1e-30);
    println!();
    println!(
        "=== stage-b screen: L={} N={} queries={nq} heads={} hd={} p99(|O|)={t99:.4e} ===",
        b.want, b.n_tokens, b.nh, b.hd
    );
    println!("--- per-arm vs ARM 1 (kernel q8/f16 differential, pre-gate) ---");
    println!("{:<10} {:>12} {:>12}  note", "arm", "max-abs", "tail-1%");
    println!("{:<10} {:>12.3e} {:>12.3e}  cpu-q8ref vs arm1 (self-check-B mate)", "arm2", q8_a1, q8_a1t);
    println!("{:<10} {:>12.3e} {:>12.3e}  Q0 format-only (the Q0 predictor)", "arm3", q0_a1, q0_a1t);
    println!("{:<10} {:>12.3e} {:>12.3e}  fp8-QK + exact weighted PV", "arm4", qk_a1, qk_a1t);
    println!("{:<10} {:>12.3e} {:>12.3e}  full stage-b route N (changing-unit O)", "arm5", sb_a1, sb_a1t);
    println!("{:<10} {:>12.3e} {:>12.3e}  full stage-b route Q (changing-unit O)", "arm6", rq_a1, rq_a1t);
    println!("--- per-arm vs EXACT (f64 reference) ---");
    println!("{:<10} {:>12} {:>12}  note", "arm", "max-abs", "tail-1%");
    println!("{:<10} {:>12.3e} {:>12.3e}  arm1 production gap (kernel vs exact)", "arm1", kk_ex, kk_ext);
    println!("{:<10} {:>12.3e} {:>12.3e}  cpu-q8ref", "arm2", q8_ex, q8_ext);
    println!("{:<10} {:>12.3e} {:>12.3e}  Q0", "arm3", q0_ex, q0_ext);
    println!("{:<10} {:>12.3e} {:>12.3e}  qkdiag", "arm4", qk_ex, qk_ext);
    println!("{:<10} {:>12.3e} {:>12.3e}  stageb-N", "arm5", sb_ex, sb_ext);
    println!("{:<10} {:>12.3e} {:>12.3e}  stageb-Q", "arm6", rq_ex, rq_ext);
    println!();
    println!("score max|Δ| vs exact: q0={s_err_q0:.3e} qk={s_err_qk:.3e} sb={s_err_sb:.3e} rq={s_err_rq:.3e}");
    println!(
        "P underflow route N (arm5, valid keys): {w_zero}/{w_pos} = {u_pct:.4}% entries, lost mass {lost_pct:.4}%"
    );
    println!(
        "P underflow route Q (arm6, valid keys): {q_zero}/{q_pos} = {uq_pct:.4}% entries, lost mass {lostq_pct:.4}%"
    );
    println!("true-f32-exp zeros: {exp_zero}; (q,h) with all-zero V codes: {zero_v_heads}");
    println!(
        "STOP GATE (plan §8.1/§11: underflow <= 0.1% of positive valid weighted entries, per route): N={} Q={}",
        if u_pct <= 0.1 { "PASS" } else { "FAIL — stop the two-fp8-leg plan" },
        if uq_pct <= 0.1 { "PASS" } else { "FAIL — stop the two-fp8-leg plan" }
    );
    println!("engine buffer: {}", if b.engine_post_gate { "POST-gate (CPU O gated-iff compared)" } else { "PRE-gate" });
    // worst-3 sampled queries (absolute row in final chunk) for triage.
    let mut order: Vec<usize> = (0..nq).collect();
    order.sort_by(|&a, &c| per_q_worst[c].partial_cmp(&per_q_worst[a]).unwrap());
    print!("worst sampled queries (final-chunk row:max-abs):");
    for &qi in order.iter().take(3) {
        print!(" {}:{:.3e}", qrows[qi], per_q_worst[qi]);
    }
    println!();
}

// Online softmax step, exact (arm1).
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn online_step(mold: f64, bmax: f64) -> (f64, f64) {
    let nm = mold.max(bmax);
    let a = if mold == f64::NEG_INFINITY { 0.0 } else { (mold - nm).exp() };
    (nm, a)
}

// Online softmax step, f32 state (arms 2-6): max/exp/alpha rounded to f32.
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn online_step_f32(mold: f64, bmax: f64) -> (f64, f64) {
    let nm = f32r(mold.max(bmax));
    let a = if mold == f64::NEG_INFINITY {
        0.0
    } else {
        f32r((mold - nm).exp())
    };
    (nm, a)
}

// f64 value rounded to f32 (host model of an f32 register/round point).
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn f32r(x: f64) -> f64 {
    (x as f32) as f64
}

// Parse stored q8_0 blocks into f16 round-point values (arm2 operands).
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn parse_q8_plane(
    stored: &[u8],
    out: &mut [f64],
    n_tokens: usize,
    nkv: usize,
    hd: usize,
    kv_dim: usize,
    row_stride: usize,
) {
    let blks = hd / 32;
    for p in 0..n_tokens {
        for h in 0..nkv {
            for b in 0..blks {
                let ko = p * row_stride + (h * blks + b) * 34;
                let sf = f16_to_f32(u16::from_le_bytes([stored[ko], stored[ko + 1]]));
                for j in 0..32 {
                    let d = h * hd + b * 32 + j;
                    out[p * kv_dim + d] = rne_f16_f64((sf as f64) * (stored[ko + 2 + j] as i8 as f64));
                }
            }
        }
    }
}

// Q0 decode: f16(f32(scale) * e4m3(code)), as f64.
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn fp8_decode_q0(scale: f32, code: u8) -> f64 {
    let v = scale * (e4m3_decode(code) as f32);
    f16_to_f64(f32_to_f16_bits(v))
}

// Native fp8 row encode (route-N operand rule, plan §2): smallest f16
// s >= a/448 (min 2^-24); code = E4M3_RNE(x/f32(s)) with an upward-half
// increment when 448*f32(s) < a; all-zero row writes s=1 and zero codes.
// Nonfinite input is rejected (oracle must not admit it).
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn fp8_encode_row(x: &[f32]) -> Result<(f32, Vec<u8>), String> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite row input".to_string());
    }
    let a = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if a == 0.0 {
        return Ok((1.0, vec![0u8; x.len()]));
    }
    let need = a / 448.0f32;
    // Round UP to the f16 grid.
    let mut bits = f32_to_f16_bits(need);
    if bits == 0x7C00 {
        return Err("unrepresentable scale (overflow)".to_string());
    }
    if f16_to_f32(bits) < need {
        bits += 1;
        if bits == 0x7C00 {
            return Err("unrepresentable scale (overflow)".to_string());
        }
    }
    let mut s = f16_to_f32(bits);
    if s < 5.9604645e-8 {
        // Minimum scale floor 2^-24 (smallest positive f16 subnormal).
        s = 5.9604645e-8;
    }
    // Upward-half increment: the stored scale must satisfy 448*s >= a.
    if 448.0f32 * s < a {
        let b2 = f32_to_f16_bits(s) + 1;
        if b2 == 0x7C00 {
            return Err("unrepresentable scale (half bump overflow)".to_string());
        }
        s = f16_to_f32(b2);
    }
    let codes = x.iter().map(|v| e4m3_encode((*v / s) as f64)).collect();
    Ok((s, codes))
}
// Smallest f16 grid value >= need (need finite, >= 0). need == 0 maps to
// +0.0; the zero-row s=1 rule lives in the callers. Rounding up off the grid
// can only land on a subnormal-or-larger f16, so the 2^-24 floor is automatic.
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn f16_ceil_pos(need: f32) -> f32 {
    assert!(need.is_finite() && need >= 0.0, "bad ceil need {need}");
    if need == 0.0 {
        return 0.0;
    }
    let mut bits = f32_to_f16_bits(need);
    if bits == 0x7C00 {
        panic!("unrepresentable ceil need {need:.6e}");
    }
    if f16_to_f32(bits) < need {
        bits += 1;
        if bits == 0x7C00 {
            panic!("unrepresentable ceil need (bump) {need:.6e}");
        }
    }
    f16_to_f32(bits)
}

// Route-Q requantize, plan §2.3: eight q8_0 block scales + 256 int8 codes ->
// one f16 row scale s_row + 256 e4m3 codes. s_row = smallest f16 >=
// (127*max_b sf_b)/448 (floor 2^-24; all-zero row -> s_row = 1);
// code' = E4M3_RNE(c * (sf_b / f32(s_row))) with the f32 ratio the kernel
// forms. Asserts the 448*f32(s_row) >= amax_row invariant per row, where
// amax_row is measured from the actual stored codes.
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn requant_q8_row(scales: &[f32; 8], codes: &[i8; 256]) -> Result<(f32, Vec<u8>), String> {
    let mut maxsf = 0.0f32;
    for &s in scales.iter() {
        if !s.is_finite() || s < 0.0 {
            return Err(format!("non-finite/negative block scale {s}"));
        }
        maxsf = maxsf.max(s);
    }
    if maxsf == 0.0 {
        // All-zero row (all scales zero => all codes must be zero).
        if codes.iter().any(|&c| c != 0) {
            return Err("zero scales with nonzero codes".to_string());
        }
        return Ok((1.0, vec![0u8; 256]));
    }
    let need = 127.0f32 * maxsf / 448.0f32;
    let mut s = f16_ceil_pos(need).max(5.9604645e-8);
    // Upward-half increment: f32 rounding of the ceil can sit below need.
    if 448.0f32 * s < 127.0f32 * maxsf {
        let b2 = f32_to_f16_bits(s) + 1;
        if b2 == 0x7C00 {
            return Err("unrepresentable s_row (half bump overflow)".to_string());
        }
        s = f16_to_f32(b2);
    }
    let mut amax = 0.0f32;
    for b in 0..8 {
        for j in 0..32 {
            amax = amax.max((scales[b] * codes[b * 32 + j] as f32).abs());
        }
    }
    if !(448.0f32 * s >= amax) {
        return Err(format!("s_row invariant fails: 448*{s:.6e} < amax {amax:.6e}"));
    }
    let mut out = Vec::with_capacity(256);
    for b in 0..8 {
        let ratio = scales[b] / s;
        for j in 0..32 {
            out.push(e4m3_encode((codes[b * 32 + j] as f32 * ratio) as f64));
        }
    }
    Ok((s, out))
}

// §8.1(B) attribution census over raw pre-quant taps: amax, p99.9 of |x|,
// and the fraction of dims clipped below the e4m3 subnormal floor under the
// per-256 row scale (lost-to-zero and subnormal fractions separately).
// The row scale follows the native rule (smallest f16 >= amax/448, floor
// 2^-24, zero row -> 1); a dim "clips" when its scaled value encodes to
// code 0 while |x| > 0.
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn tap_census(tag: &str, raw: &[f32], n_tokens: usize, nkv: usize, hd: usize, kv_dim: usize) {
    assert_eq!(raw.len(), n_tokens * kv_dim, "{tag} tap shape");
    assert!(raw.iter().all(|x| x.is_finite()), "{tag} taps non-finite");
    let n = raw.len();
    let mut mags: Vec<f32> = raw.iter().map(|x| x.abs()).collect();
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let amax = mags[n - 1];
    let p999 = mags[(n * 999 / 1000).min(n - 1)];
    let p99 = mags[(n * 99 / 100).min(n - 1)];
    let mut denom = 0u64;
    let mut lost_zero = 0u64;
    let mut subnormal = 0u64;
    for p in 0..n_tokens {
        for h in 0..nkv {
            let row = &raw[p * kv_dim + h * hd..p * kv_dim + (h + 1) * hd];
            let a = row.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let s = if a == 0.0 { 1.0 } else { f16_ceil_pos(a / 448.0).max(5.9604645e-8) };
            for &x in row {
                denom += 1;
                if x != 0.0 {
                    let c = e4m3_encode((x / s) as f64) & 0x7F;
                    if c == 0 {
                        lost_zero += 1;
                    } else if c <= 0x07 {
                        subnormal += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "tap census {tag}: n={n} amax={amax:.6e} p99.9={p999:.6e} p99={p99:.6e} | clipped-to-zero {lost_zero}/{denom}={:.4}% subnormal {subnormal}/{denom}={:.4}%",
        100.0 * lost_zero as f64 / denom as f64,
        100.0 * subnormal as f64 / denom as f64
    );
}
// Codec + scale-selection self-test (GPU-free). Vectors pin the native §2
// rule and the route-Q §2.3 rule, half-subnormal scales, upward-half bumps.
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn codec_self_test() {
    for i in -127..=127i32 {
        let v = i as f64;
        let err = (e4m3_decode(e4m3_encode(v)) - v).abs();
        let bound = if v.abs() <= 14.0 {
            0.0
        } else if v.abs() < 32.0 {
            1.0
        } else if v.abs() < 64.0 {
            2.0
        } else {
            4.0
        };
        assert!(err <= bound, "int8 {i} e4m3 err {err}");
    }
    assert_eq!(e4m3_encode(1.0), 0x38);
    assert_eq!(e4m3_decode(0x38), 1.0);
    assert_eq!(e4m3_encode(448.0), 0x7E);
    assert_eq!(e4m3_decode(0x7E), 448.0);
    assert_eq!(e4m3_encode(0.015625), 0x08);
    assert_eq!(e4m3_encode(0.0), 0x00);
    assert_eq!(e4m3_encode(-0.0), 0x80);
    assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
    assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
    // §2.1 scale selection.
    let (s, c) = fp8_encode_row(&[0.0; 256]).expect("zero row");
    assert_eq!(s, 1.0);
    assert!(c.iter().all(|&x| x == 0));
    // amax = 448 -> s = 1.
    let mut row = [0.0f32; 256];
    row[0] = 448.0;
    row[1] = -448.0;
    let (s, c) = fp8_encode_row(&row).expect("full-scale row");
    assert_eq!(s, 1.0);
    assert_eq!(c[0], 0x7E);
    assert_eq!(c[1], 0xFE);
    // tiny amax pins to the 2^-24 floor.
    let mut row = [0.0f32; 256];
    row[0] = 1.0e-20;
    let (s, _) = fp8_encode_row(&row).expect("tiny row");
    assert_eq!(s, 5.9604645e-8);
    // half-subnormal scale grid: need just above a subnormal step forces bump.
    for &need_exp in &[-20i32, -15, -10] {
        let need = 2.0f32.powi(need_exp);
        let mut row = [0.0f32; 256];
        row[0] = need * 448.0;
        let (s, _) = fp8_encode_row(&row).expect("subnormal scale row");
        assert!(448.0 * s >= row[0], "scale covers amax at 2^{need_exp}");
        assert!(s <= 2.0 * need + 1e-30, "scale is minimal at 2^{need_exp}");
    }
    // upward-half bump: amax strictly inside (448*s_prev, 448*s_next].
    let s_prev = f16_to_f32(0x3C00); // 1.0
    let a = 448.0 * s_prev + 1.0;
    let mut row = [0.0f32; 256];
    row[0] = a;
    let (s, c) = fp8_encode_row(&row).expect("half-bump row");
    assert!(448.0 * s >= a, "bumped scale covers amax");
    assert_eq!(c[0], e4m3_encode((a / s) as f64));
    // nonfinite rows are rejected, never silently substituted.
    let mut row = [0.0f32; 256];
    row[0] = f32::INFINITY;
    assert!(fp8_encode_row(&row).is_err());
    row[0] = f32::NAN;
    assert!(fp8_encode_row(&row).is_err());
    // Cross-pins with F's `fp8_bf16_format_tests` (kv.rs): RNE tie and the
    // upward-half scale bump on real magnitudes.
    assert_eq!(e4m3_encode(1.0625), 0x38); // tie 1.0|1.125 -> even (1.0)
    let mut row = [0.0f32; 256];
    row[0] = 448.01;
    let (s, _) = fp8_encode_row(&row).expect("bump row");
    assert_eq!(f32_to_f16_bits(s), 0x3C01); // 1.0 fails 448*1<448.01 -> next-up
    // Route-Q §2.3 vectors: s_row = smallest f16 >= 127*maxsf/448.
    {
        // All-zero row -> s_row = 1, zero codes.
        let (s, c) = requant_q8_row(&[0.0f32; 8], &[0i8; 256]).expect("zero requant");
        assert_eq!(s, 1.0);
        assert!(c.iter().all(|&x| x == 0));
        // Uniform scales sf, codes ±127 -> need = 127*sf/448 exactly.
        let mut codes = [0i8; 256];
        for (i, c) in codes.iter_mut().enumerate() {
            *c = if i % 2 == 0 { 127 } else { -127 };
        }
        let (s, c) = requant_q8_row(&[0.5f32; 8], &codes).expect("uniform requant");
        let need = 127.0f32 * 0.5 / 448.0;
        assert_eq!(s, f16_ceil_pos(need).max(5.9604645e-8));
        assert!(448.0 * s >= 127.0 * 0.5, "s_row covers 127*maxsf");
        assert_eq!(c[0], e4m3_encode((127.0f32 * (0.5 / s)) as f64));
        assert_eq!(c[1], e4m3_encode((-127.0f32 * (0.5 / s)) as f64));
        // Skewed scales: argmax block dominates; maxsf = 2.0.
        let mut scales = [0.25f32; 8];
        scales[3] = 2.0;
        let (s, _) = requant_q8_row(&scales, &[1i8; 256]).expect("skewed requant");
        assert_eq!(s, f16_ceil_pos(127.0 * 2.0 / 448.0).max(5.9604645e-8));
        // Zero scales with nonzero codes is rejected, never silently mapped.
        assert!(requant_q8_row(&[0.0f32; 8], &[1i8; 256]).is_err());
    }
    eprintln!("codec self-test OK (e4m3 grid, native §2 + route-Q §2.3 scale rules, subnormal floors, nonfinite rejection)");
}

// Pure-CPU recurrence edge cases: zero/one/masked rows, tiny and strongly
// changing V scales, rho finiteness. No GPU needed; runs in both modes.
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn edge_case_self_test() {
    // All-masked row: m=-inf, l=0 -> O=0, no NaN.
    let (m, l) = (f64::NEG_INFINITY, 0.0f64);
    let o = if l == 0.0 { 0.0 } else { 1.0 / l };
    assert_eq!(o, 0.0);
    let _ = m;
    // Single valid key: softmax mass 1, O = V (changing-unit: bnew, rho=1 path).
    let e = f32r((0.0f64).exp());
    assert_eq!(e, 1.0);
    // rho sweep: alpha in (0,1], bprev/bnew spanning 2^-24..65504 scales.
    for &sv_prev in &[5.9604645e-8f32, 1.0, 65504.0] {
        for &sv_new in &[5.9604645e-8f32, 1.0, 65504.0] {
            let bprev = (10.0f32 / 448.0f32).max(2.0f32.powi(-64)) * sv_prev.max(1e-30);
            let bnew = (10.0f32 / 448.0f32).max(2.0f32.powi(-64)) * sv_new.max(1e-30);
            let rho = (0.5f32 * bprev) / bnew;
            assert!(rho.is_finite() && rho > 0.0, "rho finite for sv {sv_prev}->{sv_new}");
        }
    }
    // 2^-64 floor contribution bound (§5): 32768*448*16*2^-64 ≈ 1.3e-11.
    let bound = 32768.0 * 448.0 * 16.0 * 2.0f64.powi(-64);
    assert!(bound < 1.3e-11, "floor bound {bound:.3e}");
    // f16 subnormal scales decode exactly on host (device FTZ counterpart is
    // a kernel-bringup check; host pins the exact values here).
    for bits in [0x0001u16, 0x0008, 0x0200, 0x03FF] {
        let v = f16_to_f32(bits);
        assert!(v > 0.0 && v < 6.1035156e-5, "subnormal {bits:#06x} -> {v:.3e}");
        assert_eq!(f32_to_f16_bits(v), bits, "subnormal round-trip {bits:#06x}");
    }
    eprintln!("edge-case self-test OK (masked/single-key/rho-sweep/floor-bound/subnormal round-trip)");
}

// f32 → f16 bits, round-to-nearest-even (matches the kernel's (_Float16)
// cast which is RN; subnormals handled, overflow → inf).
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn f32_to_f16_bits(x: f32) -> u16 {
    assert!(x.is_finite(), "non-finite f32→f16");
    if x == 0.0 {
        return if x.is_sign_negative() { 0x8000 } else { 0 };
    }
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x007F_FFFF;
    if exp == 0 {
        return sign; // f32 subnormal → f16 zero
    }
    let e16 = exp - 112;
    if e16 >= 31 {
        return sign | 0x7C00; // overflow → inf (data never hits this)
    }
    if e16 <= 0 {
        let m = mant | 0x0080_0000;
        let shift = 14 - e16;
        if shift >= 25 {
            return sign;
        }
        let half = 1u32 << (shift - 1);
        let mask = (half << 1).wrapping_sub(1);
        let rem = m & mask;
        let mut q = m >> shift;
        if rem > half || (rem == half && (q & 1) != 0) {
            q += 1;
        }
        if q >= 1024 {
            return sign | 0x0400;
        }
        return sign | (q as u16);
    }
    let keep = mant & !0x1FFF;
    let rem = mant & 0x1FFF;
    let mut om = keep;
    let mut oe = e16;
    if rem > 0x1000 || (rem == 0x1000 && (keep & 0x2000) != 0) {
        om = keep + 0x2000;
        if om > 0x007F_FFFF {
            om = 0;
            oe += 1;
        }
    }
    if oe >= 31 {
        return sign | 0x7C00;
    }
    sign | ((oe as u16) << 10) | ((om >> 13) as u16)
}

#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn f16_to_f32(b: u16) -> f32 {
    let sign = ((b as u32) & 0x8000) << 16;
    let exp = ((b >> 10) & 0x1F) as u32;
    let mant = (b & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e = 127 - 14;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | (0xFF << 23) | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn f16_to_f64(b: u16) -> f64 {
    f16_to_f32(b) as f64
}

// rne-to-f16 of an f64 value, returned as f64.
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn rne_f16_f64(x: f64) -> f64 {
    f16_to_f64(f32_to_f16_bits(x as f32))
}

// Round-half-even on f64 (exact-half detection is exact in binary).
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn rne(v: f64) -> f64 {
    let lo = v.floor();
    let f = v - lo;
    if f < 0.5 {
        lo
    } else if f > 0.5 {
        lo + 1.0
    } else if (lo as i64) % 2 == 0 {
        lo
    } else {
        lo + 1.0
    }
}

// OCP E4M3FN: bias 7, max 448 (0x7E), min-normal 2^-6, submin 2^-9,
// NaN = e==15 && m==7. RNE, finite saturation.
#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn e4m3_encode(x: f64) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign = if x < 0.0 || (x == 0.0 && x.is_sign_negative()) {
        0x80
    } else {
        0
    };
    let ax = x.abs();
    if ax == 0.0 {
        return sign;
    }
    if ax < 0.0009765625 {
        return sign;
    }
    if ax < 0.015625 {
        let k = rne(ax * 512.0) as u32;
        if k >= 8 {
            return sign | 0x08;
        }
        return sign | (k as u8);
    }
    let mut e = ax.log2().floor() as i32;
    if e > 8 {
        return sign | 0x7E;
    }
    if e < -6 {
        e = -6;
    }
    let step = 2f64.powi(e - 3);
    let k = rne(ax / step) as i32;
    let mut ee = e;
    let mut kk = k;
    if kk >= 16 {
        kk = 8;
        ee += 1;
    }
    if ee > 8 {
        return sign | 0x7E;
    }
    if kk > 14 && ee == 8 {
        return sign | 0x7E;
    }
    sign | (((ee + 7) as u8) << 3) | ((kk - 8) as u8)
}

#[cfg(all(feature = "deltanet", feature = "arch-qwen35"))]
fn e4m3_decode(b: u8) -> f64 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as i32;
    if e == 0 {
        s * (m as f64) * 2f64.powi(-9)
    } else {
        s * ((8 + m) as f64) * 2f64.powi(e - 7 - 3)
    }
}
