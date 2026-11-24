// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-vs-CPU anchor row (I10) for the dense Q8 attention fixture (B2.x/B2.r).
//!
//! Writes the fixture, loads it through the production carrier
//! (`load_llama_bundle`, route `ManifestPlainLlama`), and compares the engine's
//! logits with the independent CPU reference on the three fixed prompts:
//!
//! * arm `decode`: teacher-forced, one `forward_scratch_*` step per token, the
//!   logits row of every position;
//! * arm `prefill`: `forward_prefill_batch` (the batched Q8 prefill route the
//!   daemon takes for prompts of 4+ tokens), the last position's row;
//! * the embedding lookup of every prompt token is compared bit for bit;
//! * greedy generation: the engine's tokens equal the reference's wherever the
//!   reference pick is well separated;
//! * a planted wrong dequant scale on the CPU side must fail the same budget.
//!
//! Needs a GPU. Without one the test prints `skip` (not a pass) unless
//! `B2R_REQUIRE_GPU` is set, which turns the skip into a failure. Run it inside
//! a fabric job. `B2R_PROMPT_SET=pilot` measures a disjoint prompt set and
//! asserts nothing about the budget (how [`BUDGET`] was frozen);
//! `B2R_RECEIPT=<path>` writes the row as JSON.

use hipfire_arch_llama::cpu_reference::{
    argmax, compare_logits, top2_margin, CpuReference, DequantFault, LogitComparison, BUDGET,
    SEPARATED_MARGIN,
};
use hipfire_arch_llama::fixture::{
    dense_q8_fixture, write_dense_q8_fixture, DIM, HEAD_DIM, INTERMEDIATE, MAX_POSITIONS, N_HEADS,
    N_KV_HEADS, N_LAYERS, PROMPTS, VOCAB,
};
use hipfire_arch_llama::{load_llama_bundle, LlamaBundle};
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::llama::{
    forward_prefill_batch, forward_scratch_compute, forward_scratch_embed, is_batchable_la,
};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource, SpecLoadCfg};
use rdna_compute::{DType, Gpu};
use std::path::Path;

/// Prompts disjoint from [`PROMPTS`], used only to freeze the budget.
const PILOT_PROMPTS: [&[u32]; 4] = [
    &[1, 6, 17, 29, 40, 52],
    &[1, 11, 24, 37, 2, 60, 21, 9, 45],
    &[1, 34, 16, 57, 28, 3, 49, 20, 62, 35, 8, 26, 43, 15, 54, 38],
    &[
        1, 5, 59, 32, 46, 19, 51, 7, 41, 24, 63, 13, 36, 48, 22, 55, 10, 30, 44, 17, 61, 27, 4,
    ],
];

const GENERATED: usize = 8;

fn load_ctx<'a>(path: &'a Path, gpu: &'a mut Gpu) -> LoadCtx<'a> {
    LoadCtx {
        path: path.to_str().expect("fixture path is UTF-8"),
        max_seq: MAX_POSITIONS,
        sequence: None,
        deepseek4_compute_placement: Default::default(),
        deepseek4_experts_per_token: None,
        draft_path: None,
        kv_mode_override: Some("q8"),
        kv_k_override: None,
        kv_v_override: None,
        kv_backend: KvBackend::Legacy,
        kv_adaptive_override: None,
        state_quant_override: None,
        qwen_default_q8: true,
        vision_path: None,
        mtp_path: None,
        vision_mode: "auto".to_string(),
        pp: 1,
        pp_layer_counts: None,
        admitted_mesh: None,
        band_placement: None,
        spec: SpecLoadCfg::default(),
        gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
        xdna: None,
        batch_request: Default::default(),
        ep_ar_wire: Default::default(),
    }
}

fn md5_hex(path: &Path) -> String {
    format!(
        "{:x}",
        md5::compute(std::fs::read(path).expect("read fixture"))
    )
}

/// One decode step: embed, forward, download the logits row.
fn decode_step(gpu: &mut Gpu, bundle: &mut LlamaBundle, token: u32, pos: usize) -> Vec<f32> {
    forward_scratch_embed(
        gpu,
        &bundle.weights,
        &bundle.config,
        token,
        pos,
        &bundle.scratch,
    )
    .expect("embed");
    forward_scratch_compute(
        gpu,
        &bundle.weights,
        &bundle.config,
        pos,
        &mut bundle.kv,
        &bundle.scratch,
    )
    .expect("forward");
    gpu.download_f32(&bundle.scratch.logits).expect("logits")
}

fn engine_argmax(row: &[f32]) -> u32 {
    argmax(row) as u32
}

struct PromptRow {
    len: usize,
    decode: LogitComparison,
    prefill: LogitComparison,
    embed_mismatches: usize,
    generated_engine: Vec<u32>,
    generated_cpu: Vec<u32>,
    generation_separated_agree: bool,
}

fn json_comparison(c: &LogitComparison) -> String {
    format!(
        "{{\"positions\":{},\"max_abs_logit_diff\":{:e},\"mean_abs_logit_diff\":{:e},\
         \"max_kld\":{:e},\"mean_kld\":{:e},\"argmax_flips\":{},\"separated_flips\":{},\
         \"min_margin\":{:.4},\"max_margin\":{:.4}}}",
        c.positions,
        c.max_abs_logit_diff,
        c.mean_abs_logit_diff,
        c.max_kld,
        c.mean_kld,
        c.argmax_flips,
        c.separated_flips,
        c.min_margin,
        c.max_margin
    )
}

#[test]
fn dense_q8_fixture_loads_and_matches_cpu_reference() {
    let pilot = std::env::var("B2R_PROMPT_SET").as_deref() == Ok("pilot");
    let prompts: Vec<&[u32]> = if pilot {
        PILOT_PROMPTS.to_vec()
    } else {
        PROMPTS.to_vec()
    };
    let require_gpu = std::env::var_os("B2R_REQUIRE_GPU").is_some();
    let mut gpu = match Gpu::init() {
        Ok(gpu) => gpu,
        Err(e) if require_gpu => panic!("B2R_REQUIRE_GPU set and no GPU: {e:?}"),
        Err(e) => {
            eprintln!("b2r: skip (no GPU: {e:?}); a skip is not a pass");
            return;
        }
    };
    let arch = gpu.arch.as_str().to_string();

    // Deterministic build: two writes, same bytes.
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("dense-q8-llama.hfq");
    let path_again = dir.path().join("dense-q8-llama-again.hfq");
    write_dense_q8_fixture(&path).expect("write fixture");
    write_dense_q8_fixture(&path_again).expect("write fixture again");
    let digest = md5_hex(&path);
    assert_eq!(
        digest,
        md5_hex(&path_again),
        "fixture build is not deterministic"
    );
    let size = std::fs::metadata(&path).unwrap().len();
    eprintln!("b2r: fixture {size} bytes md5 {digest} gpu {arch}");
    if let Ok(keep) = std::env::var("B2R_KEEP_FIXTURE") {
        std::fs::copy(&path, &keep).expect("export fixture");
    }

    // Production load.
    let fixture = dense_q8_fixture();
    let hfq = HfqFile::open(&path).expect("open fixture");
    assert!(
        !hfq.has_awq_sidecars(),
        "plain file: the ManifestPlainLlama route"
    );
    let started = std::time::Instant::now();
    let mut ctx = load_ctx(&path, &mut gpu);
    let mut bundle = load_llama_bundle(ModelSource::Hfq(hfq), &mut ctx).expect("production load");
    drop(ctx);
    let load_ms = started.elapsed().as_secs_f64() * 1e3;

    // The Q8 path is the one taken: every projection, the embedding and the
    // lm_head are Q8_0 on the GPU, and the geometry is the fixture's.
    let cfg = &bundle.config;
    assert_eq!(
        (
            cfg.dim,
            cfg.hidden_dim,
            cfg.n_layers,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
            cfg.vocab_size
        ),
        (
            DIM,
            INTERMEDIATE,
            N_LAYERS,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            VOCAB
        )
    );
    assert_eq!(bundle.weights.token_embd.dtype, DType::Q8_0, "embedding");
    assert_eq!(bundle.weights.output.gpu_dtype, DType::Q8_0, "lm_head");
    for (i, l) in bundle.weights.layers.iter().enumerate() {
        for (n, w) in [
            ("wq", &l.wq),
            ("wk", &l.wk),
            ("wv", &l.wv),
            ("wo", &l.wo),
            ("gate", &l.w_gate),
            ("up", &l.w_up),
            ("down", &l.w_down),
        ] {
            assert_eq!(w.gpu_dtype, DType::Q8_0, "layer {i} {n}");
        }
    }
    let batched = bundle.weights.layers.iter().all(|l| {
        [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]
            .iter()
            .all(|w| is_batchable_la(w.gpu_dtype, &arch))
    });
    eprintln!("b2r: batched prefill eligible on {arch}: {batched}");

    let cpu = CpuReference::from_tensors(&fixture.tensors).expect("decode fixture");
    let mut rows = Vec::new();
    let mut all_cpu: Vec<Vec<f64>> = Vec::new();
    let mut all_decode: Vec<Vec<f32>> = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        let want = cpu.forward(prompt);

        // Arm decode (also the embedding bit check).
        let mut got = Vec::new();
        let mut embed_mismatches = 0usize;
        for (pos, &token) in prompt.iter().enumerate() {
            let row = decode_step(&mut gpu, &mut bundle, token, pos);
            // `scratch.x` holds the post-layer residual now; the embedding
            // check re-runs the lookup alone.
            forward_scratch_embed(
                &mut gpu,
                &bundle.weights,
                &bundle.config,
                token,
                pos,
                &bundle.scratch,
            )
            .expect("embed");
            let x = gpu.download_f32(&bundle.scratch.x).expect("embedding row");
            embed_mismatches += x
                .iter()
                .zip(cpu.embed_row(token))
                .filter(|(g, c)| g.to_bits() != (**c as f32).to_bits())
                .count();
            got.push(row);
        }
        let decode = compare_logits(&want, &got);
        all_cpu.extend(want.iter().cloned());
        all_decode.extend(got.iter().cloned());

        // Arm prefill: last row of the batched route (KV positions overwritten).
        forward_prefill_batch(
            &mut gpu,
            &bundle.weights,
            &bundle.config,
            prompt,
            0,
            &mut bundle.kv,
            &bundle.scratch,
            None,
        )
        .expect("prefill");
        let last = gpu
            .download_f32(&bundle.scratch.logits)
            .expect("prefill logits");
        let prefill = compare_logits(&want[want.len() - 1..], &[last]);

        // Greedy generation on both sides from the same prompt.
        let mut engine_tokens = Vec::new();
        let mut next = engine_argmax(got.last().unwrap());
        let mut separated_ok = true;
        let mut cpu_tokens = prompt.to_vec();
        for step in 0..GENERATED {
            let cpu_rows = cpu.forward(&cpu_tokens);
            let cpu_row = cpu_rows.last().unwrap();
            let cpu_next = argmax(cpu_row) as u32;
            let agree = cpu_next == next;
            if !agree && top2_margin(cpu_row) >= SEPARATED_MARGIN {
                separated_ok = false;
            }
            engine_tokens.push(next);
            // Teacher-force the reference's own pick so one near-tie flip does
            // not make the two continuations diverge for the rest of the run.
            cpu_tokens.push(cpu_next);
            let row = decode_step(&mut gpu, &mut bundle, cpu_next, prompt.len() + step);
            next = engine_argmax(&row);
        }
        let generated_cpu = cpu_tokens[prompt.len()..].to_vec();
        eprintln!(
            "b2r: prompt {index} len {} decode {decode:?}\n      prefill {prefill:?}\n      \
             embed-bit-mismatches {embed_mismatches} engine {engine_tokens:?} cpu {generated_cpu:?}",
            prompt.len()
        );
        rows.push(PromptRow {
            len: prompt.len(),
            decode,
            prefill,
            embed_mismatches,
            generated_engine: engine_tokens,
            generated_cpu,
            generation_separated_agree: separated_ok,
        });
    }

    // Aggregate over every compared position of every prompt.
    let aggregate = compare_logits(&all_cpu, &all_decode);
    eprintln!("b2r: aggregate decode {aggregate:?}");

    // Coverage: nonconstant logits, live KV state.
    for row in &all_decode {
        assert!(
            row.iter().all(|v| v.is_finite()),
            "non-finite engine logits"
        );
        let (lo, hi) = row
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
        assert!(hi - lo > 1.0, "engine logits are nearly constant");
    }
    assert!(bundle.kv.k_gpu.len() == N_LAYERS && bundle.kv.max_seq >= MAX_POSITIONS);

    // Planted fault: a wrong dequant scale must not fit the same budget. The
    // first four rows are asserted; the last two only record how small a scale
    // error the budget still sees.
    let mut fault_rows = Vec::new();
    for (tensor, mul, blocks, asserted) in [
        (
            "model.layers.0.self_attn.q_proj.weight",
            2.0,
            usize::MAX,
            true,
        ),
        (
            "model.layers.1.mlp.down_proj.weight",
            1.05,
            usize::MAX,
            true,
        ),
        ("model.layers.0.mlp.gate_proj.weight", 0.5, 64, true),
        ("lm_head.weight", 1.02, usize::MAX, true),
        (
            "model.layers.1.mlp.down_proj.weight",
            1.02,
            usize::MAX,
            false,
        ),
        ("lm_head.weight", 1.01, usize::MAX, false),
    ] {
        let bad = CpuReference::from_tensors_with_fault(
            &fixture.tensors,
            Some(&DequantFault {
                tensor: tensor.into(),
                scale_mul: mul,
                blocks,
            }),
        )
        .expect("decode planted fixture");
        let bad_rows: Vec<Vec<f64>> = prompts.iter().flat_map(|p| bad.forward(p)).collect();
        let c = compare_logits(&bad_rows, &all_decode);
        let verdict = BUDGET.check(&c);
        eprintln!("b2r: planted {tensor} x{mul} (first {blocks} blocks): {verdict:?} {c:?}");
        fault_rows.push((tensor, mul, blocks, asserted, c, verdict));
    }

    if let Ok(out) = std::env::var("B2R_RECEIPT") {
        let prompts_json: Vec<String> = rows
            .iter()
            .map(|r| {
                format!(
                    "{{\"len\":{},\"decode\":{},\"prefill_last\":{},\"embed_bit_mismatches\":{},\
                     \"generated_engine\":{:?},\"generated_cpu\":{:?},\
                     \"generation_separated_agree\":{}}}",
                    r.len,
                    json_comparison(&r.decode),
                    json_comparison(&r.prefill),
                    r.embed_mismatches,
                    r.generated_engine,
                    r.generated_cpu,
                    r.generation_separated_agree
                )
            })
            .collect();
        let faults_json: Vec<String> = fault_rows
            .iter()
            .map(|(t, m, b, asserted, c, v)| {
                format!(
                    "{{\"tensor\":\"{t}\",\"scale_mul\":{m},\"blocks\":{},\"asserted\":{asserted},\
                     \"comparison\":{},\"fails_budget\":{}}}",
                    if *b == usize::MAX { -1i64 } else { *b as i64 },
                    json_comparison(c),
                    v.is_err()
                )
            })
            .collect();
        let json = format!(
            "{{\"row\":\"I10\",\"fixture\":\"dense-q8-llama\",\"prompt_set\":\"{}\",\"gpu\":\"{arch}\",\
             \"fixture_bytes\":{size},\"fixture_md5\":\"{digest}\",\"load_ms\":{load_ms:.1},\
             \"batched_prefill\":{batched},\"budget\":{{\"max_abs_logit_diff\":{:e},\"max_kld\":{:e},\
             \"mean_kld\":{:e},\"separated_margin\":{SEPARATED_MARGIN}}},\"aggregate_decode\":{},\
             \"prompts\":[{}],\"planted_faults\":[{}]}}\n",
            if pilot { "pilot" } else { "gate" },
            BUDGET.max_abs_logit_diff,
            BUDGET.max_kld,
            BUDGET.mean_kld,
            json_comparison(&aggregate),
            prompts_json.join(","),
            faults_json.join(",")
        );
        std::fs::write(&out, json).expect("write receipt");
    }

    ArchModel::free_gpu(Box::new(bundle), &mut gpu);

    if pilot {
        eprintln!("b2r: pilot set: measurements only, budget not asserted");
        return;
    }

    // Verdicts.
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(
            r.embed_mismatches, 0,
            "prompt {i}: embedding rows differ from the CPU decode"
        );
        BUDGET
            .check(&r.decode)
            .unwrap_or_else(|e| panic!("prompt {i} decode outside the budget: {e}"));
        BUDGET
            .check(&r.prefill)
            .unwrap_or_else(|e| panic!("prompt {i} prefill outside the budget: {e}"));
        assert!(
            r.generation_separated_agree,
            "prompt {i}: greedy pick flipped at a separated position"
        );
    }
    BUDGET
        .check(&aggregate)
        .expect("aggregate inside the budget");
    for (tensor, mul, _, asserted, c, verdict) in &fault_rows {
        if *asserted {
            assert!(
                verdict.is_err(),
                "planted {tensor} x{mul} passed the budget: {c:?}"
            );
        }
    }
}
