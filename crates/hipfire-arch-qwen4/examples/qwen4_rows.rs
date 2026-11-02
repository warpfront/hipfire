//! Row-amortization probe: GPU wall time of one `forward_chunk` of N rows
//! (the MTP verify shape) against N single-row `forward_token`s, at a decode
//! position after a real prompt prefill, plus a bitwise comparison of the two
//! routes' logits.
//!
//! Model load follows `qwen4_kld`: admission on the GPU architecture
//! (`admit_hfqm_artifact_on`), slab layer arenas served by
//! `expert_arena::recipe_arena_payload`, and `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N`
//! (a fixed count; `auto` is refused) keeping layers 0..N resident with the
//! rest host-mapped and borrowed from the resident source bytes.
//!
//! usage: qwen4_rows MODEL.hfq PROMPT.txt [ROWS=1,2,3,4] [ITERS=6]

use hipfire_arch_qwen4::admit_hfqm_artifact_on;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource as _, SourcePayload};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let path = std::path::Path::new(&args[1]);
    let prompt = std::fs::read_to_string(&args[2]).map_err(|e| e.to_string())?;
    let row_counts: Vec<usize> = args
        .get(3)
        .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![1, 2, 3, 4]);
    let iters: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(6);
    let n_ctx = 2048;

    let mut hfq = HfqFile::open(path).map_err(|e| e.to_string())?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| e.to_string())?;
    let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
    // Admission is GPU-arch aware: an expert-slab artifact admits only on a
    // supported arch. Initialising the GPU allocates no model memory.
    let receipt = admit_hfqm_artifact_on(&hfq, Some(&gpu.arch)).map_err(|e| e.to_string())?;
    // The loader's forward scratch plan, decided from the actual payload
    // while the mapping is readable.
    let scratch_mode = hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
        &gpu,
        &hfq,
        &receipt.config,
        &receipt.manifest,
    )
    .map_err(|e| e.to_string())?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().map_err(|e| e.to_string())?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    // `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` keeps trunk layers 0..N resident
    // and host-maps the rest (a slab artifact larger than one card).
    let weights = {
        let mut weights = receipt.manifest.weights.clone();
        match hipfire_arch_qwen4::expert_residency::expert_vram_layers_from_env()? {
            None => {}
            Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Layers(layers)) => {
                if gpu.is_uma() {
                    return Err("host-mapped experts are only for discrete GPUs".into());
                }
                let moved = hipfire_arch_qwen4::expert_residency::place_routed_experts(
                    &mut weights,
                    layers,
                );
                eprintln!(
                    "qwen4_rows: routed experts: layers 0..{layers} in VRAM, {moved} tensors host-mapped"
                );
            }
            Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Auto) => {
                return Err("qwen4_rows takes a fixed expert layer count, not auto".into());
            }
        }
        weights
    };
    let transaction = fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            if let Some(payload) = hipfire_arch_qwen4::expert_arena::recipe_arena_payload(
                &source,
                &receipt.manifest.identity.recipe,
                entry,
            ) {
                return payload;
            }
            // The store rejects a host-mapped entry as a range: borrow it
            // from the resident source bytes, as the product loader does.
            if entry.residency == hipfire_runtime::weight_manifest::WeightResidency::HostMapped {
                let (info, bytes) = source
                    .tensor_data(&entry.name)
                    .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
                return Ok(SourcePayload::Borrowed { info, bytes });
            }
            source
                .tensor_range(&entry.name)
                .map_err(|e| e.to_string())?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .map_err(|e| e.to_string())?;
    let vocab = receipt.config.vocab_size;
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        // Qwen3.5's DeltaNet `state_quant` knob: auto (q8) or fp32.
        &std::env::var("HIPFIRE_STATE_QUANT").unwrap_or_default(),
        &gpu,
        &receipt.config,
    )?;
    let backend = hipfire_arch_qwen4::Qwen4KvBackend::automatic(&gpu);
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        &receipt.manifest.identity.recipe,
        transaction,
        &receipt.placements,
        &mut gpu,
        n_ctx,
        receipt.ple,
        state_format,
        backend,
    )
    .map_err(|e| e.to_string())?;
    bundle
        .attach_forward(&mut gpu, n_ctx, scratch_mode)
        .map_err(|e| e.to_string())?;
    let logits = gpu
        .zeros(&[8 * vocab], DType::F32)
        .map_err(|e| e.to_string())?;
    let row = logits.sub_offset(0, vocab);
    let tokens = tokenizer.encode(&prompt);
    let filler: Vec<u32> = tokens.iter().copied().cycle().skip(7).take(64).collect();
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b })
            .0
    };

    for &rows in &row_counts {
        let mut chunk_ms = Vec::new();
        let mut single_ms = Vec::new();
        for it in 0..iters {
            bundle.reset(&mut gpu).map_err(|e| e.to_string())?;
            bundle
                .forward_chunk_final(&mut gpu, &tokens, &row, None)
                .map_err(|e| e.to_string())?;
            gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
            let block = &filler[it..it + rows];
            let t = Instant::now();
            bundle
                .forward_chunk(&mut gpu, block, &logits.sub_offset(0, rows * vocab), None)
                .map_err(|e| e.to_string())?;
            gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
            chunk_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let chunk_logits = gpu
                .download_f32(&logits.sub_offset(0, rows * vocab))
                .map_err(|e| e.to_string())?;

            bundle.reset(&mut gpu).map_err(|e| e.to_string())?;
            bundle
                .forward_chunk_final(&mut gpu, &tokens, &row, None)
                .map_err(|e| e.to_string())?;
            gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
            let t = Instant::now();
            let mut single_logits = Vec::with_capacity(rows * vocab);
            for &token in block {
                bundle
                    .forward_token(&mut gpu, token, &row, None)
                    .map_err(|e| e.to_string())?;
                if it == 0 {
                    single_logits.extend(gpu.download_f32(&row).map_err(|e| e.to_string())?);
                }
            }
            gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
            single_ms.push(t.elapsed().as_secs_f64() * 1e3);
            if it == 0 {
                let diff = chunk_logits
                    .iter()
                    .zip(&single_logits)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let agree = (0..rows)
                    .filter(|&r| {
                        argmax(&chunk_logits[r * vocab..(r + 1) * vocab])
                            == argmax(&single_logits[r * vocab..(r + 1) * vocab])
                    })
                    .count();
                let exact = chunk_logits
                    .iter()
                    .zip(&single_logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                println!(
                    "rows {rows}: chunk vs singles max|diff| {diff:.3e} argmax agree {agree}/{rows} bitwise {exact}"
                );
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let chunk_min = chunk_ms.iter().copied().fold(f64::MAX, f64::min);
        println!(
            "rows {rows}: chunk {:.2} ms (min {chunk_min:.2})  singles {:.2} ms  (prompt {} tokens)",
            med(&mut chunk_ms),
            med(&mut single_ms),
            tokens.len()
        );
    }
    Ok(())
}
