// SPDX-License-Identifier: Apache-2.0

use hipfire_arch_qwen35::qwen35::{
    self, DeltaNetState, HfqSource, Layout, Qwen35Scratch, StateQuant,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_mode::KvMode;
use hipfire_runtime::llama::{KvCache, KvCacheExt, KvDims, KvLayers, KvTarget};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
use rdna_compute::Gpu;
use std::path::Path;

const KV_MAX: usize = 4096;
fn decode_steps() -> usize {
    std::env::var("HIPFIRE_TP_PARITY_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16)
}

fn tp_count() -> usize {
    if let Some(arg) = std::env::args().nth(2) {
        if let Ok(v) = arg.parse::<usize>() {
            if (2..=5).contains(&v) {
                return v;
            }
        }
    }
    for key in ["HIPFIRE_DENSE_TP", "HIPFIRE_TP", "HIPFIRE_TP_PARITY_TP"] {
        if let Ok(val) = std::env::var(key) {
            if let Ok(v) = val.parse::<usize>() {
                if (2..=5).contains(&v) {
                    return v;
                }
            }
        }
    }
    2
}

fn argmax(xs: &[f32]) -> u32 {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn prompt(tokenizer: &Tokenizer) -> Vec<u32> {
    let text = if let Ok(path) = std::env::var("HIPFIRE_TP_PARITY_PROMPT_FILE") {
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read parity prompt {path}: {error}"));
        format!("<|im_start|>user\n{raw}<|im_end|>\n<|im_start|>assistant\n")
    } else {
        "<|im_start|>user\nExplain KV cache briefly.<|im_end|>\n<|im_start|>assistant\n".to_string()
    };
    tokenizer.encode(&text)
}

fn load_single(
    hfq: &mut HfqFile,
    config: &qwen35::Qwen35Config,
    gpu: &mut Gpu,
) -> qwen35::Qwen35Weights {
    let mut source = HfqSource::new(hfq, config);
    qwen35::load_weights(
        &mut source,
        std::slice::from_mut(gpu),
        &Layout::single(config.n_layers),
    )
    .expect("load single weights")
}

fn make_kv(gpu: &mut Gpu, config: &qwen35::Qwen35Config) -> KvCache {
    let is_kv_layer = config
        .layer_types
        .iter()
        .map(|layer| *layer == qwen35::LayerType::FullAttention)
        .collect();
    <KvCache as KvCacheExt>::from_mode(
        KvMode::Q8,
        KvTarget::Single(gpu),
        &KvDims {
            layers: KvLayers::Mask(is_kv_layer),
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            max_seq: KV_MAX,
            physical_cap: Some(KV_MAX),
        },
    )
    .expect("filtered q8 kv")
}

fn run_reference(path: &str, seed: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>) {
    let decode = decode_steps();
    let mut hfq = HfqFile::open(Path::new(path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = Gpu::init().expect("gpu0");
    let weights = load_single(&mut hfq, &config, &mut gpu);
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");
    let mut kv = make_kv(&mut gpu, &config);
    let mut dn = DeltaNetState::new_with_quant(&mut gpu, &config, StateQuant::FP32).expect("dn");
    qwen35::forward_prefill_batch(
        &mut gpu, &weights, &config, seed, 0, &mut kv, &mut dn, &scratch, None, None, None, None,
    )
    .expect("reference batched prefill");
    let mut tokens = Vec::with_capacity(decode);
    let mut logits = Vec::with_capacity(decode);
    let mut next = argmax(&gpu.download_f32(&scratch.logits).unwrap());
    for step in 0..decode {
        tokens.push(next);
        logits.push(gpu.download_f32(&scratch.logits).unwrap());
        if step + 1 < decode {
            qwen35::forward_scratch(
                &mut gpu,
                &weights,
                &config,
                next,
                seed.len() + step,
                &mut kv,
                &mut dn,
                &scratch,
            )
            .expect("reference decode");
            next = argmax(&gpu.download_f32(&scratch.logits).unwrap());
        }
    }
    let _ = scratch.free_gpu(&mut gpu);
    dn.free_gpu(&mut gpu);
    let _ = kv.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.drain_pool();
    (tokens, logits)
}

fn run_tp(path: &str, seed: &[u32], forced: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>) {
    let decode = decode_steps();
    let tp = tp_count();
    let mut hfq = HfqFile::open(Path::new(path)).expect("open model");
    let global = qwen35::config_from_hfq(&hfq).expect("config");
    let shard = ShardConfig::new(tp, false, 0, ExpertAssign::Stride).unwrap();
    let layouts = qwen35::dense_tp_rank_layouts(&global, &shard).expect("dense TP layout");
    qwen35::preflight_weights_dense_tp(&hfq, &global, &shard).expect("dense TP preflight");
    let configs: Vec<_> = layouts
        .iter()
        .map(|l| qwen35::local_dense_tp_config(&global, l))
        .collect();
    let mut gpus =
        Gpus::init_tp(tp, global.n_layers).unwrap_or_else(|e| panic!("init tp{tp}: {e:?}"));
    for gpu in &mut gpus.devices {
        gpu.bind_thread().unwrap();
        gpu.active_stream = Some(gpu.hip.stream_create().unwrap());
    }
    let mut weights = Vec::new();
    let mut scratches = Vec::new();
    let mut kvs = Vec::new();
    let mut dns = Vec::new();
    for rank in 0..tp {
        weights.push(
            qwen35::load_weights_dense_tp_rank(
                &mut hfq,
                &global,
                &mut gpus.devices[rank],
                &layouts[rank],
            )
            .unwrap_or_else(|e| panic!("load TP rank {rank}: {e:?}")),
        );
        scratches.push(Qwen35Scratch::new(&mut gpus.devices[rank], &configs[rank], 128).unwrap());
        kvs.push(make_kv(&mut gpus.devices[rank], &configs[rank]));
        dns.push(
            DeltaNetState::new_with_quant(
                &mut gpus.devices[rank],
                &configs[rank],
                StateQuant::FP32,
            )
            .unwrap(),
        );
    }
    let head_oracle = if std::env::var("HIPFIRE_TP_HEAD_ORACLE").as_deref() == Ok("1") {
        Some(
            gpus.devices[0]
                .alloc_tensor(&[global.vocab_size], rdna_compute::DType::F32)
                .unwrap(),
        )
    } else {
        None
    };
    // Two-phase peer topology: probe without mutation. Complete P2P enables
    // RCCL path; mixed topology uses host-staged allreduce and never enables
    // peers (ROCm does not map later allocations after enable).
    let peer = gpus
        .can_access_peer_all()
        .expect("parity TP can_access_peer_all");
    hipfire_runtime::ep::ensure_rank_streams(&mut gpus).expect("parity TP rank streams");
    if peer {
        if std::env::var("HIPFIRE_TP_PARITY_GRAPHS").as_deref() == Ok("1") {
            let barriers = global.n_layers * 2;
            gpus.prepare_tp2_whole_token_graph(barriers, global.dim * 4)
                .expect("parity TP graph signal tape");
        }
        let enabled = gpus
            .enable_peer_all()
            .expect("parity TP enable_peer_all after rank allocations");
        assert!(
            enabled,
            "parity TP complete topology must enable_peer_all true"
        );
    }
    eprintln!(
        "TP{tp} peer_access={peer} allreduce={}",
        if peer { "rccl" } else { "host-staged" }
    );

    qwen35::forward_prefill_dense_tp(
        &mut gpus, &shard, &weights, &configs, seed, 0, &mut kvs, &mut dns, &scratches,
    )
    .expect("TP prefill");
    if std::env::var("HIPFIRE_TP_PARITY_REPEAT_PREFILL").as_deref() == Ok("1") {
        gpus.devices[0].bind_thread().unwrap();
        let expected = gpus.devices[0].download_f32(&scratches[0].logits).unwrap();
        // Grow beyond the short seed, then reuse the larger PBS for a short
        // request. Padding/strides must depend on live rows, not stale rows.
        let prime = seed.repeat(8);
        for rank in 0..tp {
            dns[rank].reset(&mut gpus.devices[rank]).unwrap();
            kvs[rank].compact_offset = 0;
        }
        qwen35::forward_prefill_dense_tp(
            &mut gpus, &shard, &weights, &configs, &prime, 0, &mut kvs, &mut dns, &scratches,
        )
        .expect("grow TP prefill scratch before shrink test");
        for repetition in 0..3 {
            for rank in 0..tp {
                gpus.devices[rank].bind_thread().unwrap();
                dns[rank].reset(&mut gpus.devices[rank]).unwrap();
                kvs[rank].compact_offset = 0;
            }
            qwen35::forward_prefill_dense_tp(
                &mut gpus, &shard, &weights, &configs, seed, 0, &mut kvs, &mut dns, &scratches,
            )
            .expect("repeated TP prefill");
            gpus.devices[0].bind_thread().unwrap();
            let actual = gpus.devices[0].download_f32(&scratches[0].logits).unwrap();
            assert!(
                expected
                    .iter()
                    .zip(&actual)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "repeated prefill logit divergence at repetition {repetition}"
            );
        }
        println!("PASS repeated prefill 3/3 bit-exact");
    }
    let mut tokens = Vec::new();
    let mut logits = Vec::new();
    for step in 0..decode {
        gpus.devices[0].bind_thread().unwrap();
        let row = gpus.devices[0].download_f32(&scratches[0].logits).unwrap();
        if let Some(out) = &head_oracle {
            // Independent reference: the original unsliced WeightRef, not
            // project_rows with start=0 (which could share an adapter bug).
            let ctx = hipfire_dispatch::context::DispatchCtx::new(&gpus.devices[0]);
            let wr = weights[0].output.dispatch_ref();
            hipfire_dispatch::pipeline::execute_steps(
                &mut gpus.devices[0],
                &ctx,
                &[hipfire_dispatch::pipeline::Step::Gemv {
                    w: &wr,
                    input: hipfire_dispatch::pipeline::GemvInput::Raw(&scratches[0].tmp),
                    out,
                }],
            )
            .unwrap();
            let reference = gpus.devices[0].download_f32(out).unwrap();
            assert!(
                reference.iter().chain(&row).all(|v| v.is_finite()),
                "non-finite head logits at step {step}"
            );
            let unequal = row
                .iter()
                .zip(&reference)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                unequal, 0,
                "full-vs-sharded head logits differ at actual TP state step {step}"
            );
        }
        tokens.push(argmax(&row));
        logits.push(row);
        if step + 1 < decode {
            qwen35::forward_scratch_dense_tp(
                &mut gpus,
                &shard,
                &weights,
                &configs,
                forced[step],
                seed.len() + step,
                &mut kvs,
                &mut dns,
                &scratches,
            )
            .expect("TP decode");
        }
    }
    gpus.drop_rank_graphs(false);
    if let Some(out) = head_oracle {
        gpus.devices[0].free_tensor(out).unwrap();
    }
    gpus.free_peer_reduce_scratch()
        .expect("parity TP free peer scratch");
    (tokens, logits)
}

fn main() {
    std::env::set_var("HIPFIRE_DETERMINISTIC", "1");
    let tp = tp_count();
    let path = std::env::args().nth(1).expect("model path");
    let hfq = HfqFile::open(Path::new(&path)).expect("open tokenizer metadata");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let seed = prompt(&tokenizer);
    drop(hfq);
    let (reference_tokens, reference_logits) = run_reference(&path, &seed);
    let (tp_tokens, tp_logits) = run_tp(&path, &seed, &reference_tokens);
    if let Ok(path) = std::env::var("HIPFIRE_TP_PARITY_DUMP") {
        use std::io::Write;
        let mut out = std::fs::File::create(path).expect("create TP logit dump");
        for row in &tp_logits {
            for value in row {
                out.write_all(&value.to_le_bytes()).unwrap();
            }
        }
    }
    let mut worst = 0.0f32;
    for (step, (reference, tp_row)) in reference_logits.iter().zip(&tp_logits).enumerate() {
        assert!(
            reference.iter().chain(tp_row).all(|v| v.is_finite()),
            "non-finite logits at step {step}"
        );
        let scale = reference
            .iter()
            .fold(0.0f32, |m, x| m.max(x.abs()))
            .max(1e-12);
        let delta = reference
            .iter()
            .zip(tp_row)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        worst = worst.max(delta / scale);
        println!(
            "step={step:02} ref={} tp={} rel={:.3e}",
            reference_tokens[step],
            tp_tokens[step],
            delta / scale
        );
    }
    assert_eq!(reference_tokens, tp_tokens, "TP{tp} argmax divergence");
    // Rank-local QKV reaches Q8 cache quantization with rank-count-dependent
    // FP reduction order. Argmax must stay exact while 32-step TP2..5 fixture
    // drift remains below the measured 0.3% envelope.
    assert!(worst < 3e-3, "TP{tp} relative logit error {worst:.3e}");
    println!(
        "PASS tp={tp} tokens={} worst_rel={worst:.3e}",
        reference_tokens.len()
    );
}
