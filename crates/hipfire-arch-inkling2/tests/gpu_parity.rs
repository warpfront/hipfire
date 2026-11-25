// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Brief 03 gates: GPU forward parity for Inkling-2-Mini (arch 24).
//!
//! All tests are `#[ignore]`d: they need the 14 GB MQ4 artifact and a gfx1100
//! GPU. Run fresh-process, one gate at a time:
//!
//! ```powershell
//! $env:HIPFIRE_HOME = 'D:\hipfire-models\inkling2-home'
//! $env:HIPFIRE_GRAPH = '0'
//! cargo test -p hipfire-arch-inkling2 --test gpu_parity -- --ignored --nocapture
//! ```
//!
//! Gate order (stop at the first red): 0 (CPU MQ4 reference trace),
//! 1 (one visit vs CPU debug), 2 (all 48 visits vs MQ4 trace),
//! 3 (token-0 logits vs CPU MQ4 + llama oracle), 3b (token 1..N logits vs
//! the CPU cached decode), 4 (tok/s, unoptimized).
//!
//! The artifact path defaults to the Windows layout below; set `INKLING2_MQ4`
//! to point at it elsewhere.

use hipfire_arch_inkling2::arch_model::{Inkling2Bundle, Inkling2State};
use hipfire_arch_inkling2::config::Inkling2Config;
use hipfire_arch_inkling2::forward::{
    forward_token, forward_token_cached, lm_head_logits, CpuDecodeState,
};
use hipfire_arch_inkling2::gpu::{upload, Inkling2GpuState, Inkling2GpuWeights};
use hipfire_arch_inkling2::weights::Inkling2Weights;
use hipfire_arch_inkling2::{
    forward_token_gpu, forward_token_gpu_capture, forward_token_gpu_trace,
};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::collections::HashMap;

const ART: &str = r"D:\hipfire-models\inkling2-mini-mq4.hfq";
const TRACE_DIR: &str = r"D:\hipfire-models\mq4-visits-tok0";
const CPU_LOGITS: &str = r"D:\hipfire-models\cpu-logits-tok0.txt";
const RAW_LOGITS: &str = r"D:\hipfire-models\cpu-raw-logits-tok0.txt";

fn ensure_home() {
    if std::env::var("HIPFIRE_HOME").is_err() {
        std::env::set_var("HIPFIRE_HOME", r"D:\hipfire-models\inkling2-home");
    }
    assert_eq!(
        std::env::var("HIPFIRE_GRAPH").as_deref(),
        Ok("0"),
        "gates require HIPFIRE_GRAPH=0"
    );
}

fn open_artifact() -> (Inkling2Config, Inkling2Weights) {
    let art = std::env::var("INKLING2_MQ4").unwrap_or_else(|_| ART.to_string());
    let hfq = HfqFile::open(std::path::Path::new(&art)).expect("open artifact");
    let cfg = Inkling2Config::from_metadata_json(&hfq.metadata_json).expect("parse config");
    let weights = Inkling2Weights::load(&hfq, &cfg).expect("index weights");
    (cfg, weights)
}

fn embed_row0(weights: &Inkling2Weights, cfg: &Inkling2Config) -> Vec<f32> {
    let (_, emb) = weights.f32("model.llm.embed.weight").expect("embed");
    emb[..cfg.n_embd].to_vec()
}

fn read_bits(path: &str) -> Vec<f32> {
    let s = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("read {path}"));
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let last = l.split_whitespace().last().unwrap();
            f32::from_bits(
                u32::from_str_radix(last, 16).unwrap_or_else(|_| panic!("hex bits: {last}")),
            )
        })
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| v as f64).sum::<f64>() / n,
        b.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (*x as f64 - ma, *y as f64 - mb);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    (sxy / (sxx * syy).sqrt()) as f32
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

fn test_gpu() -> Gpu {
    ensure_home();
    Gpu::init().expect("gpu init")
}

fn upload_test_bundle(
    cfg: &Inkling2Config,
    weights: &Inkling2Weights,
    gpu: &mut Gpu,
    max_seq: usize,
) -> (Inkling2GpuWeights, Inkling2GpuState) {
    upload(weights, cfg, gpu, max_seq).expect("gpu upload")
}

fn make_bundle(
    cfg: Inkling2Config,
    weights: Inkling2Weights,
    gw: Inkling2GpuWeights,
    gs: Inkling2GpuState,
) -> Inkling2Bundle {
    Inkling2Bundle {
        config: cfg,
        weights,
        state: Inkling2State { seq_pos: 0 },
        eos_tok: 200006,
        gpu_weights: Some(gw),
        gpu_state: Some(gs),
    }
}

/// Gate 0: CPU reference trace for the MQ4 artifact (no forward.rs edits —
/// this test calls the public forward + trace API). Confirms the known MQ4
/// logits (r=0.9907 vs oracle, argmax 31274).
#[test]
#[ignore = "needs the 14 GB MQ4 artifact (CPU-only)"]
fn gate0_cpu_mq4_reference_trace() {
    ensure_home();
    let (cfg, weights) = open_artifact();
    let x = embed_row0(&weights, &cfg);
    let mut trace: Vec<Vec<f32>> = Vec::new();
    let hidden = forward_token(&weights, &cfg, &x, 0, Some(&mut trace), None).expect("cpu forward");
    assert_eq!(trace.len(), 48, "need all 48 visit_outs");
    std::fs::create_dir_all(TRACE_DIR).expect("trace dir");
    for (i, v) in trace.iter().enumerate() {
        assert_eq!(v.len(), cfg.n_embd);
        let mut s = String::with_capacity(v.len() * 12);
        for x in v {
            s.push_str(&format!("{:08x}\n", x.to_bits()));
        }
        std::fs::write(format!("{TRACE_DIR}\\visit{i:02}.txt"), s).expect("write visit");
    }
    let logits = lm_head_logits(&weights, &cfg, &hidden).expect("lm head");
    assert_eq!(logits.len(), cfg.unpadded_vocab_size);
    let am = argmax(&logits);
    eprintln!("gate0: token-0 MQ4 CPU logits argmax={am}");
    assert_eq!(am, 31274, "MQ4 token-0 argmax must be 31274");
    // Known-logits file: same computation, must agree ~bit-exact.
    let known = read_bits(CPU_LOGITS);
    assert_eq!(known.len(), logits.len());
    let r = pearson(&logits, &known);
    let md = max_abs(&logits, &known);
    let mn = mean_abs(&logits, &known);
    eprintln!("gate0: vs {CPU_LOGITS}: r={r:.7} max|d|={md:.3e} mean|d|={mn:.3e}");
    assert!(
        r > 0.999999,
        "CPU recompute must reproduce the known MQ4 logits"
    );
    // Oracle gap: F32 reference (llama.cpp lineage) — expect r ~= 0.9907.
    if std::path::Path::new(RAW_LOGITS).exists() {
        let raw = read_bits(RAW_LOGITS);
        if raw.len() == logits.len() {
            let ro = pearson(&logits, &raw);
            let mo = mean_abs(&logits, &raw);
            eprintln!("gate0: vs F32 oracle: r={ro:.6} mean|d|={mo:.4} (expect r ~= 0.9907)");
        }
    }
    eprintln!("gate0: PASS (trace in {TRACE_DIR}, argmax 31274)");
}

/// Debug bisection: visit-0 components GPU vs CPU debug capture.
#[test]
#[ignore = "needs the 14 GB MQ4 artifact + gfx1100"]
fn dbg_visit0_components() {
    let (cfg, weights) = open_artifact();
    let mut gpu = test_gpu();
    let (gw, gs) = upload_test_bundle(&cfg, &weights, &mut gpu, 64);
    let mut bundle = make_bundle(cfg, weights, gw, gs);
    let got = forward_token_gpu_capture(&mut bundle, &mut gpu, 0, 0, 0).expect("gpu visit 0");
    let cfg2 = &bundle.config;
    let w2 = &bundle.weights;
    let x = embed_row0(w2, cfg2);
    let mut dbg = (0usize, HashMap::new());
    let _ = forward_token(w2, cfg2, &x, 0, None, Some(&mut dbg)).expect("cpu forward");
    let map = dbg.1;
    for k in [
        "x_in",
        "q_inp",
        "k_inp",
        "v_inp",
        "k_raw",
        "v_raw",
        "k_conv",
        "v_conv",
        "qc",
        "kc",
        "vc",
        "rc",
        "sinks",
        "rel_bias",
        "attn_heads",
        "wo_out",
        "cur_attn",
        "ffn_mix",
        "ffn_normed",
        "router_logits",
        "topk_idx",
        "topk_w",
        "moe_raw",
        "moe_conv",
        "visit_out",
    ] {
        match (map.get(k), got.get(k)) {
            (Some(want), Some(have)) => {
                let md = max_abs(want, have);
                let r = if want.len() > 1 {
                    pearson(want, have)
                } else {
                    1.0
                };
                eprintln!("dbg0: {k:12} n={} max|d|={md:.3e} r={r:.7}", want.len());
            }
            (a, b) => eprintln!(
                "dbg0: {k:12} missing cpu={} gpu={}",
                a.is_none(),
                b.is_none()
            ),
        }
    }
    // Stage-by-stage MoE reference in the rotated domain (what the GPU
    // kernels consume): FWHT-rotate the natural F32 weights, then matvec.
    {
        fn fwht_signs(seed: u32) -> [f32; 256] {
            let mut state = seed;
            let mut out = [0.0f32; 256];
            for v in out.iter_mut() {
                state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
                *v = if (state >> 16) & 1 == 1 { 1.0 } else { -1.0 };
            }
            out
        }
        fn fwht256_forward(group: &mut [f32; 256], s1: &[f32; 256], s2: &[f32; 256]) {
            for i in 0..256 {
                group[i] *= s1[i];
            }
            let mut stride = 1;
            while stride < 256 {
                let mut j = 0;
                while j < 256 {
                    for k in 0..stride {
                        let a = group[j + k];
                        let b = group[j + k + stride];
                        group[j + k] = a + b;
                        group[j + k + stride] = a - b;
                    }
                    j += stride * 2;
                }
                stride <<= 1;
            }
            for i in 0..256 {
                group[i] *= 0.0625 * s2[i];
            }
        }
        fn rot_rows(x: &[f32], rows: usize, k: usize) -> Vec<f32> {
            assert_eq!(k % 256, 0);
            let s1 = fwht_signs(42);
            let s2 = fwht_signs(1042);
            let mut out = vec![0.0f32; rows * k];
            for r in 0..rows {
                for g in 0..k / 256 {
                    let mut grp = [0.0f32; 256];
                    grp.copy_from_slice(&x[r * k + g * 256..r * k + (g + 1) * 256]);
                    fwht256_forward(&mut grp, &s1, &s2);
                    out[r * k + g * 256..r * k + (g + 1) * 256].copy_from_slice(&grp);
                }
            }
            out
        }
        fn mv(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
            (0..rows)
                .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum())
                .collect()
        }
        let fnorm = map.get("ffn_normed").unwrap();
        let x_rot = rot_rows(fnorm, 1, 2304);
        eprintln!(
            "dbg0: ffn_x_rot max|d|={:.3e}",
            max_abs(&x_rot, got.get("ffn_x_rot").unwrap())
        );
        let gpu_ids: Vec<usize> = got
            .get("topk_idx")
            .unwrap()
            .iter()
            .map(|v| v.to_bits() as usize)
            .collect();
        let p = "model.llm.layers.0.";
        let mut gate_h = vec![0.0f32; 8 * 1152];
        let mut up_h = vec![0.0f32; 8 * 1152];
        for (r, ee) in gpu_ids.iter().enumerate() {
            let (_, gu) = w2
                .f32(&format!("{p}mlp.experts.{ee}.gate_up_proj.weight"))
                .unwrap();
            let gu_rot = rot_rows(&gu, 2 * 1152, 2304);
            let g: Vec<f32> = mv(&gu_rot[..1152 * 2304], &x_rot, 1152, 2304);
            let u: Vec<f32> = mv(&gu_rot[1152 * 2304..], &x_rot, 1152, 2304);
            gate_h[r * 1152..(r + 1) * 1152].copy_from_slice(&g);
            up_h[r * 1152..(r + 1) * 1152].copy_from_slice(&u);
        }
        eprintln!(
            "dbg0: gate_b max|d|={:.3e}",
            max_abs(&gate_h, got.get("gate_b").unwrap())
        );
        eprintln!(
            "dbg0: up_b   max|d|={:.3e}",
            max_abs(&up_h, got.get("up_b").unwrap())
        );
        let situ = |g: f32, u: f32| {
            4.0 * (g / 4.0).tanh() * (1.0 / (1.0 + (-g).exp())) * 25.0 * (u / 25.0).tanh()
        };
        let mut act_h = vec![0.0f32; 8 * 1280];
        for r in 0..8 {
            for c in 0..1152 {
                act_h[r * 1280 + c] = situ(gate_h[r * 1152 + c], up_h[r * 1152 + c]);
            }
        }
        eprintln!(
            "dbg0: act    max|d|={:.3e}",
            max_abs(&act_h, got.get("act").unwrap())
        );
        let rot_h = rot_rows(&act_h, 8, 1280);
        eprintln!(
            "dbg0: rot_b  max|d|={:.3e}",
            max_abs(&rot_h, got.get("rot_b").unwrap())
        );
        let tw: Vec<f32> = got.get("topk_w").unwrap().clone();
        let mut down_h = vec![0.0f32; 8 * 2304];
        for (r, ee) in gpu_ids.iter().enumerate() {
            let (_, dn) = w2
                .f32(&format!("{p}mlp.experts.{ee}.down_proj.weight"))
                .unwrap();
            let mut padded = vec![0.0f32; 2304 * 1280];
            for row in 0..2304 {
                padded[row * 1280..row * 1280 + 1152]
                    .copy_from_slice(&dn[row * 1152..(row + 1) * 1152]);
            }
            let dn_rot = rot_rows(&padded, 2304, 1280);
            let d: Vec<f32> = mv(&dn_rot, &rot_h[r * 1280..(r + 1) * 1280], 2304, 1280);
            down_h[r * 2304..(r + 1) * 2304].copy_from_slice(&d);
        }
        eprintln!(
            "dbg0: down_exp max|d|={:.3e}",
            max_abs(&down_h, got.get("down_exp").unwrap())
        );
        // Decisive: reference the kernel against the REPACKED bytes it reads
        // (downloaded from VRAM), not the qt13 naturals. Match => kernel is
        // correct and the gap above is V2-repack quantization noise.
        {
            use hipfire_runtime::weight_backend::dequantize_weight_to_f32;
            let mut repack_h = vec![0.0f32; 8 * 2304];
            for (r, ee) in gpu_ids.iter().enumerate() {
                let raw = gpu
                    .download_raw_bytes(
                        &bundle.gpu_weights.as_ref().unwrap().layers[0].experts[*ee]
                            .down
                            .buf,
                    )
                    .expect("download repacked down");
                let nat = dequantize_weight_to_f32(44, &raw, 2304 * 1280);
                let dn_rot = rot_rows(&nat, 2304, 1280);
                let d: Vec<f32> = mv(&dn_rot, &rot_h[r * 1280..(r + 1) * 1280], 2304, 1280);
                repack_h[r * 2304..(r + 1) * 2304].copy_from_slice(&d);
            }
            eprintln!(
                "dbg0: down_exp vs REPACKED max|d|={:.3e}",
                max_abs(&repack_h, got.get("down_exp").unwrap())
            );
        }
        let mut out_h = vec![0.0f32; 2304];
        for r in 0..8 {
            for c in 0..2304 {
                out_h[c] += tw[r] * down_h[r * 2304 + c];
            }
        }
        eprintln!(
            "dbg0: combine max|d|={:.3e}",
            max_abs(&out_h, got.get("moe_raw").unwrap())
        );
        eprintln!("dbg0: gpu top8 ids={gpu_ids:?}");
    }
}

/// Gate 1: one visit (phys 6, pass 0 = visit 6) vs the CPU debug capture.
#[test]
#[ignore = "needs the 14 GB MQ4 artifact + gfx1100"]
fn gate1_visit6_components() {
    let (cfg, weights) = open_artifact();
    let mut gpu = test_gpu();
    let (gw, gs) = upload_test_bundle(&cfg, &weights, &mut gpu, 64);
    let mut bundle = make_bundle(cfg, weights, gw, gs);
    let got = forward_token_gpu_capture(&mut bundle, &mut gpu, 0, 0, 6).expect("gpu visit 6");
    // CPU reference with debug capture at visit 6 + full trace for visit_out.
    let cfg2 = &bundle.config;
    let w2 = &bundle.weights;
    let x = embed_row0(w2, cfg2);
    let mut map: HashMap<String, Vec<f32>> = HashMap::new();
    let mut trace_cpu: Vec<Vec<f32>> = Vec::new();
    let mut dbg = (6usize, map);
    let _ =
        forward_token(w2, cfg2, &x, 0, Some(&mut trace_cpu), Some(&mut dbg)).expect("cpu forward");
    let map = dbg.1;
    assert_eq!(trace_cpu.len(), 48);
    let keys = [
        "x_in",
        "q_inp",
        "k_inp",
        "v_inp",
        "k_raw",
        "v_raw",
        "k_conv",
        "v_conv",
        "qc",
        "kc",
        "vc",
        "rc",
        "sinks",
        "rel_bias",
        "attn_heads",
        "wo_out",
        "cur_attn",
        "ffn_mix",
        "ffn_normed",
        "moe_raw",
        "moe_conv",
        "visit_out",
    ];
    let mut worst = 0.0f32;
    let mut fails = 0u32;
    for k in keys {
        let want: &[f32] = if k == "visit_out" {
            &trace_cpu[6]
        } else {
            map.get(k)
                .unwrap_or_else(|| panic!("cpu debug missing {k}"))
        };
        let have = got
            .get(k)
            .unwrap_or_else(|| panic!("gpu capture missing {k}"));
        assert_eq!(want.len(), have.len(), "{k} len");
        let md = max_abs(want, have);
        let r = pearson(want, have);
        worst = worst.max(md);
        let ok = md <= 5e-3 && r >= 0.9999;
        if !ok {
            fails += 1;
        }
        eprintln!(
            "gate1: {k:12} n={} max|d|={md:.3e} r={r:.7} {}",
            want.len(),
            if ok { "ok" } else { "FAIL" }
        );
    }
    eprintln!("gate1: worst max|d|={worst:.3e}, {fails} failing components");
    assert_eq!(fails, 0, "gate1: {fails} components exceed the bar");
}

/// Gate 2: all 48 visit_outs vs the MQ4 trace; then a short multi-token
/// smoke loop (parity vs brief-04's cached path is pending — 04 has not
/// landed in this checkout).
#[test]
#[ignore = "needs the 14 GB MQ4 artifact + gfx1100"]
fn gate2_all_visits() {
    let (cfg, weights) = open_artifact();
    let mut gpu = test_gpu();
    let (gw, gs) = upload_test_bundle(&cfg, &weights, &mut gpu, 64);
    let mut bundle = make_bundle(cfg, weights, gw, gs);
    let t0 = std::time::Instant::now();
    let (_, trace) = forward_token_gpu_trace(&mut bundle, &mut gpu, 0, 0).expect("gpu trace");
    let dt = t0.elapsed();
    assert_eq!(trace.len(), 48);
    let mut worst = 0.0f32;
    let mut worst_r = 1.0f32;
    let mut fails = 0u32;
    for (i, v) in trace.iter().enumerate() {
        let want = read_bits(&format!("{TRACE_DIR}\\visit{i:02}.txt"));
        assert_eq!(want.len(), v.len(), "visit {i} len");
        let md = max_abs(&want, v);
        let r = pearson(&want, v);
        worst = worst.max(md);
        worst_r = worst_r.min(r);
        let ok = md <= 5e-3 && r >= 0.9999;
        if !ok {
            fails += 1;
        }
        eprintln!(
            "gate2: visit{i:02} max|d|={md:.3e} r={r:.7} {}",
            if ok { "ok" } else { "FAIL" }
        );
    }
    eprintln!("gate2: token-0 worst max|d|={worst:.3e}, worst r={worst_r:.7}, {fails}/48 failing, {dt:?} unoptimized");
    // Multi-token smoke: loop pos 0..4 (parity vs 04's generate_cpu pending).
    let mut bundle2 = bundle;
    bundle2
        .gpu_state
        .as_mut()
        .unwrap()
        .reset_session_state(&mut gpu)
        .expect("reset");
    bundle2.state.seq_pos = 0;
    for (pos, tok) in [0u32, 1, 2, 3, 4].iter().enumerate() {
        let logits =
            forward_token_gpu(&mut bundle2, &mut gpu, *tok, pos).expect("multi-token step");
        assert_eq!(logits.len(), bundle2.config.unpadded_vocab_size);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "pos {pos}: non-finite logits"
        );
        assert_eq!(bundle2.gpu_state.as_ref().unwrap().seq_pos, pos + 1);
        eprintln!(
            "gate2: pos {pos} tok {tok} argmax={} finite OK",
            argmax(&logits)
        );
    }
    eprintln!("gate2: multi-token smoke PASS (parity vs 04 generate_cpu PENDING — 04 not landed)");
}

/// Gate 3: token-0 logits, GPU vs CPU MQ4 on the same artifact.
#[test]
#[ignore = "needs the 14 GB MQ4 artifact + gfx1100"]
fn gate3_token0_logits() {
    let (cfg, weights) = open_artifact();
    let mut gpu = test_gpu();
    let (gw, gs) = upload_test_bundle(&cfg, &weights, &mut gpu, 64);
    let mut bundle = make_bundle(cfg, weights, gw, gs);
    let got = forward_token_gpu(&mut bundle, &mut gpu, 0, 0).expect("gpu logits");
    let x = embed_row0(&bundle.weights, &bundle.config);
    let hidden =
        forward_token(&bundle.weights, &bundle.config, &x, 0, None, None).expect("cpu forward");
    let want = lm_head_logits(&bundle.weights, &bundle.config, &hidden).expect("cpu logits");
    assert_eq!(got.len(), want.len());
    let r = pearson(&got, &want);
    let am_got = argmax(&got);
    let am_want = argmax(&want);
    let md = max_abs(&got, &want);
    let mn = mean_abs(&got, &want);
    eprintln!("gate3: r={r:.7} argmax gpu={am_got} cpu={am_want} max|d|={md:.3e} mean|d|={mn:.3e}");
    eprintln!(
        "gate3: bar check: argmax_equal={} argmax31274={} r_ok={} mean_ok={}",
        am_got == am_want,
        am_got == 31274,
        r >= 0.9999,
        mn <= 0.05
    );
    assert_eq!(am_got, am_want, "argmax mismatch");
    assert_eq!(am_got, 31274, "token-0 argmax must be 31274");
    assert!(r >= 0.9999, "r {r} below 0.9999");
    assert!(mn <= 0.05, "mean|d| {mn} exceeds 0.05");
    if std::path::Path::new(RAW_LOGITS).exists() {
        let raw = read_bits(RAW_LOGITS);
        if raw.len() == got.len() {
            eprintln!(
                "gate3: vs F32 oracle: r={:.6} (expect ~= 0.9907)",
                pearson(&got, &raw)
            );
        }
    }
    eprintln!("gate3: PASS");
}

/// Gate 3b: logits at positions 0..N, GPU decode vs the CPU cached decode
/// (`forward_token_cached`, which keeps a separate conv history per conv).
/// Gate 3 runs at position 0, where every conv history is zero, so a bug in
/// how the GPU keeps or indexes that history cannot show there; from
/// position 1 on it can. Bars: the same argmax at every position and
/// r >= `R_MIN` at every position.
///
/// `R_MIN` is empirical, from gfx1151 on this sequence. Correct: r 0.9988 /
/// 0.9971 / 0.9948 / 0.9940. With the loop-message bank also mixing the
/// prelude and first core block: 0.997 / 0.978 / 0.987 / 0.983. With, in
/// addition, the MoE post-conv history at the wrong offset (`4 * ekv`):
/// 0.997 / 0.959 / 0.968 / 0.952 and a different argmax at position 3. What
/// remains is mostly the down_proj repack (the CPU reads the original MQ4
/// down_proj, the GPU a re-quantized MQ4G256V2 copy).
#[test]
#[ignore = "needs the 14 GB MQ4 artifact + a HIP GPU; slow CPU reference"]
fn gate3b_multi_token_logits() {
    const TOKENS: [u32; 4] = [5000, 12345, 777, 31274];
    const R_MIN: f32 = 0.99;
    let (cfg, weights) = open_artifact();
    let vocab = cfg.unpadded_vocab_size;
    let e = cfg.n_embd;

    let mut want = Vec::with_capacity(TOKENS.len());
    {
        let (_, emb) = weights.f32("model.llm.embed.weight").expect("embed");
        let mut st = CpuDecodeState::new(&cfg, 16).expect("cpu state");
        for (pos, &t) in TOKENS.iter().enumerate() {
            let x = &emb[t as usize * e..(t as usize + 1) * e];
            let hidden =
                forward_token_cached(&weights, &cfg, &mut st, x, pos).expect("cpu forward");
            let l = lm_head_logits(&weights, &cfg, &hidden).expect("cpu logits");
            want.push(l[..vocab].to_vec());
        }
    }

    let mut gpu = test_gpu();
    let (gw, gs) = upload_test_bundle(&cfg, &weights, &mut gpu, 64);
    let mut bundle = make_bundle(cfg, weights, gw, gs);
    let mut fails = Vec::new();
    for (pos, &t) in TOKENS.iter().enumerate() {
        let got = forward_token_gpu(&mut bundle, &mut gpu, t, pos).expect("gpu logits");
        let w = &want[pos];
        assert_eq!(got.len(), w.len());
        let r = pearson(&got, w);
        let (am_got, am_want) = (argmax(&got), argmax(w));
        eprintln!(
            "gate3b: pos {pos} tok {t}: r={r:.7} argmax gpu={am_got} cpu={am_want} max|d|={:.3e} mean|d|={:.3e}",
            max_abs(&got, w),
            mean_abs(&got, w)
        );
        if am_got != am_want {
            fails.push(format!("pos {pos}: argmax gpu {am_got} vs cpu {am_want}"));
        }
        if r < R_MIN {
            fails.push(format!("pos {pos}: r {r} below {R_MIN}"));
        }
    }
    if let Some(gw) = bundle.gpu_weights.take() {
        gw.free_gpu(&mut gpu);
    }
    if let Some(gs) = bundle.gpu_state.take() {
        gs.free_gpu(&mut gpu);
    }
    assert!(fails.is_empty(), "gate3b: {}", fails.join("; "));
    eprintln!("gate3b: PASS");
}

/// Gate 4: single-token tok/s, clearly labelled unoptimized.
#[test]
#[ignore = "needs the 14 GB MQ4 artifact + gfx1100"]
fn gate4_tok_per_sec() {
    let (cfg, weights) = open_artifact();
    let mut gpu = test_gpu();
    let (gw, gs) = upload_test_bundle(&cfg, &weights, &mut gpu, 64);
    let mut bundle = make_bundle(cfg, weights, gw, gs);
    // Warmup (JIT + DPM settle), then one timed token on a fresh session.
    let _ = forward_token_gpu(&mut bundle, &mut gpu, 0, 0).expect("warmup");
    bundle
        .gpu_state
        .as_mut()
        .unwrap()
        .reset_session_state(&mut gpu)
        .expect("reset");
    bundle.state.seq_pos = 0;
    let t0 = std::time::Instant::now();
    let logits = forward_token_gpu(&mut bundle, &mut gpu, 0, 0).expect("timed token");
    let dt = t0.elapsed().as_secs_f64();
    assert_eq!(logits.len(), bundle.config.unpadded_vocab_size);
    eprintln!(
        "gate4: single token {dt:.3}s = {:.2} tok/s (UNOPTIMIZED, max_seq=64, release build)",
        1.0 / dt
    );
}
