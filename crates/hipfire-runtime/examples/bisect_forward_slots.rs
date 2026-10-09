// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Diagnostic-only harness for the `forward_batch_slots` golden-test failure
// (SP3 Task 3's `test_forward_slots_golden`, n_slots=1 slot=0 step=0).
// Not part of any gate. For each `max_layer` in 1..=n_layers, runs a FRESH
// reference `forward_prefill_batch_with_pbs_opts` call and a FRESH candidate
// `forward_batch_slots_with_max_layer` call — both bounded to that same
// `max_layer`, both replaying the identical 5-token stream from
// start_pos=0 on brand-new KV cache / DeltaNet state (fresh state is
// required: reusing state across iterations would double-apply the
// DeltaNet recurrence for already-processed layers) — and diffs
// `pbs.x_batch[0..n*dim]` to find the first layer where the two diverge.
//
// Usage:
//   cargo run --release -p hipfire-runtime --features deltanet,arch-qwen35 \
//     --example bisect_forward_slots -- <model.hf4>

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet,arch-qwen35");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::forward_slots::{
        forward_batch_slots_with_max_layer, SlotDescStaging, SlotKvTier,
    };
    use hipfire_arch_qwen35::qwen35::{
        self, DeltaNetState, LayerType, PrefillBatchScratch, Qwen35Scratch,
    };
    use hipfire_runtime::kv_backend::KvBackend;
    use hipfire_runtime::kv_mode::{self, KvMode, SlotKvTierPlan};
    use hipfire_runtime::llama::{KvCache, KvCacheExt, KvDims, KvLayers, KvTarget};
    use hipfire_runtime::slot_batch::SlotBatch;
    use hipfire_runtime::hfq::HfqFile;
    use rdna_compute::kv_slots::{preflight_alloc, R9700_VRAM_BYTES};
    use rdna_compute::slot_pool::{SlotId, SlotPool};
    use rdna_compute::{DType, Gpu, GpuTensor};
    use std::path::Path;

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!(
            "Usage: bisect_forward_slots <model.hf4> [kv-mode]  \
             (kv-mode: q8|fwht2|fwht3|fwht4|bf16; default q8)"
        );
        std::process::exit(1);
    });
    let mode_name = std::env::args().nth(2).unwrap_or_else(|| "q8".to_string());
    let kv_mode::ResolveResult { mode, warning } =
        kv_mode::resolve(&mode_name, &kv_mode::QWEN35_SLOTS_POLICY);
    assert!(
        warning.is_none(),
        "kv mode {mode_name:?} did not cleanly resolve: {warning:?}"
    );
    assert!(mode != KvMode::F16, "f16 has no sequential reference");

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("parse Qwen3.5 config");
    let n_fa_layers = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::FullAttention)
        .count();
    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();
    let plan = SlotKvTierPlan::resolve(mode, config.n_kv_heads, config.head_dim)
        .unwrap_or_else(|e| panic!("kv mode {mode_name}: {e}"));
    let dim = config.dim;

    const PROMPT_LEN: usize = 5;
    const CAP_TOKENS: usize = 64;
    let tokens: Vec<u32> = (0..PROMPT_LEN)
        .map(|i| ((i as u32) * 131 % 900) + 1)
        .collect();

    let weight_bytes = std::fs::metadata(&model_path)
        .expect("stat model file")
        .len();
    let planned = weight_bytes + 2 * 1024 * 1024 * 1024u64; // weights + generous flat slop
    preflight_alloc(planned, R9700_VRAM_BYTES, "bisect_forward_slots").expect("preflight refused");

    let mut gpu = Gpu::init().expect("gpu init");
    let weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");

    println!(
        "model: {} layers, dim={dim}, prompt_len={PROMPT_LEN}",
        config.n_layers
    );
    println!(
        "{:>3} {:>16} {:>14} {:>14} {:>10}",
        "L", "type", "max|ref|", "max|ref-cand|", "rel"
    );

    let kv_seq = (PROMPT_LEN + 16).max(CAP_TOKENS).max(512);

    for max_layer in 1..=config.n_layers {
        // ---- reference: fresh KvCache + DeltaNetState + scratch every time ----
        let mut ref_kv = if mode == KvMode::Q8 {
            KvCache::new_gpu_q8(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            )
            .expect("ref KvCache")
        } else {
            // The SAME mode-generic constructor the sequential carrier uses
            // (golden harness `run_reference_for_slot`), so the bf16/fwht
            // reference is production-true; the legacy q8 constructor stays
            // for the q8 baseline this harness was validated against.
            <KvCache as KvCacheExt>::from_mode_with_backend(
                mode,
                KvBackend::Legacy,
                KvTarget::Single(&mut gpu),
                &KvDims {
                    layers: KvLayers::Mask(is_kv_layer.clone()),
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim,
                    max_seq: kv_seq,
                    physical_cap: Some(kv_seq),
                },
            )
            .expect("ref KvCache (mode-generic)")
        };
        let mut ref_dn = DeltaNetState::new(&mut gpu, &config).expect("ref DeltaNetState");
        let ref_scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_seq)
            .expect("ref Qwen35Scratch");
        let ref_pbs = PrefillBatchScratch::new(&mut gpu, &config, PROMPT_LEN)
            .expect("ref PrefillBatchScratch");

        qwen35::forward_prefill_batch_with_pbs_opts(
            &mut gpu,
            &weights,
            &config,
            &tokens,
            0,
            &mut ref_kv,
            &mut ref_dn,
            &ref_scratch,
            None,
            None,
            None,
            None,
            Some(&ref_pbs),
            None,
            Some(max_layer),
            false,
            qwen35::DflashFusionCtx::Off,
        )
        .expect("reference forward (bounded)");
        gpu.hip.device_synchronize().expect("sync ref");
        let ref_x = gpu
            .download_f32(&ref_pbs.x_batch.sub_offset(0, PROMPT_LEN * dim))
            .expect("dl ref x");

        ref_kv.free_gpu(&mut gpu).expect("free ref_kv");
        ref_dn.free_gpu(&mut gpu);
        ref_scratch.free_gpu(&mut gpu);
        ref_pbs.free_gpu(&mut gpu);

        // ---- candidate: fresh SlotPool/arenas/DeltaNetState/scratch every time ----
        let mut pool = SlotPool::new_with_strides(
            1,
            CAP_TOKENS,
            plan.k_bytes_per_pos,
            plan.v_bytes_per_pos,
        )
        .expect("SlotPool::new_with_strides");
        let slot0 = pool.acquire().expect("acquire slot 0");
        assert_eq!(slot0.0, 0);
        let k_arena_bytes = pool.k_arena_bytes();
        let v_arena_bytes = pool.v_arena_bytes();
        let k_arenas: Vec<GpuTensor> = (0..n_fa_layers)
            .map(|_| gpu.zeros(&[k_arena_bytes], DType::Raw).expect("k_arena"))
            .collect();
        let v_arenas: Vec<GpuTensor> = (0..n_fa_layers)
            .map(|_| gpu.zeros(&[v_arena_bytes], DType::Raw).expect("v_arena"))
            .collect();
        let mut dn_states =
            vec![DeltaNetState::new(&mut gpu, &config).expect("cand DeltaNetState")];
        let mut desc_staging =
            SlotDescStaging::new(&mut gpu, 1, PROMPT_LEN, 0).expect("SlotDescStaging");
        let upload_f32 = |gpu: &mut Gpu, vals: &[f32]| -> GpuTensor {
            let t = gpu
                .alloc_tensor(&[vals.len()], DType::F32)
                .expect("tier table alloc");
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
            gpu.hip
                .memcpy_htod(&t.buf, &bytes)
                .expect("tier table upload");
            t
        };
        let (cos, sin, s1, s2) = if let Some(len) = plan.givens_len {
            let (c, si) = KvCache::gen_givens_angles(42, len);
            (
                Some(upload_f32(&mut gpu, &c)),
                Some(upload_f32(&mut gpu, &si)),
                None,
                None,
            )
        } else if let Some(len) = plan.fwht_len {
            (
                None,
                None,
                Some(upload_f32(&mut gpu, &KvCache::gen_fwht_signs(42, len))),
                Some(upload_f32(&mut gpu, &KvCache::gen_fwht_signs(1042, len))),
            )
        } else {
            (None, None, None, None)
        };
        let kv_tier = SlotKvTier {
            mode,
            givens_cos: cos,
            givens_sin: sin,
            fwht_signs1: s1,
            fwht_signs2: s2,
        };
        let cand_pbs = PrefillBatchScratch::new(&mut gpu, &config, PROMPT_LEN)
            .expect("cand PrefillBatchScratch");
        let cand_scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, CAP_TOKENS)
            .expect("cand Qwen35Scratch");
        let logits_out = gpu
            .zeros(&[config.vocab_size], DType::F32)
            .expect("logits_out");
        let batch = SlotBatch::build(&[(SlotId(0), &tokens[..], 0usize)]);

        forward_batch_slots_with_max_layer(
            &mut gpu,
            &weights,
            &config,
            &batch,
            &mut pool,
            &mut dn_states,
            &k_arenas,
            &v_arenas,
            &mut desc_staging,
            &kv_tier,
            &cand_pbs,
            &cand_scratch,
            &logits_out,
            Some(max_layer),
        )
        .expect("candidate forward (bounded)");
        gpu.hip.device_synchronize().expect("sync cand");
        let cand_x = gpu
            .download_f32(&cand_pbs.x_batch.sub_offset(0, PROMPT_LEN * dim))
            .expect("dl cand x");

        for t in k_arenas {
            gpu.free_tensor(t).expect("free k_arena");
        }
        for t in v_arenas {
            gpu.free_tensor(t).expect("free v_arena");
        }
        for dn in dn_states {
            dn.free_gpu(&mut gpu);
        }
        desc_staging.free_gpu(&mut gpu);
        kv_tier.free_gpu(&mut gpu);
        cand_pbs.free_gpu(&mut gpu);
        cand_scratch.free_gpu(&mut gpu);
        gpu.free_tensor(logits_out).expect("free logits_out");

        let max_ref = ref_x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let max_diff = ref_x
            .iter()
            .zip(cand_x.iter())
            .fold(0.0f32, |a, (&r, &c)| a.max((r - c).abs()));
        let rel = max_diff / max_ref.max(1e-6);
        let lt = config.layer_types[max_layer - 1];
        println!("{max_layer:>3} {lt:>16?} {max_ref:>14.6} {max_diff:>14.6} {rel:>10.4}");
        if rel > 0.02 {
            println!("  >>> first material divergence at layer {max_layer} ({lt:?})");
            break;
        }
    }

    println!("done");
}
