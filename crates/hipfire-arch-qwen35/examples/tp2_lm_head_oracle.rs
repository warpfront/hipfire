// SPDX-License-Identifier: Apache-2.0
//! Head-only row-sharding oracle and attribution probe; not product throughput.
use hipfire_arch_qwen35::qwen35::{
    config_from_hfq,
    load::{load_weight_tensor, qwen35_tensor_name_candidates},
    tp_output,
};
use hipfire_runtime::{hfq::HfqFile, multi_gpu::Gpus, weight_backend::MemoryTarget};
use rdna_compute::DType;
use std::{path::Path, time::Instant};

fn main() {
    let path = std::env::args().nth(1).expect("HFQ model path");
    let hfq = HfqFile::open(Path::new(&path)).unwrap();
    let cfg = config_from_hfq(&hfq).unwrap();
    let mut gpus = Gpus::init_tp(2, cfg.n_layers).unwrap();
    let name = if qwen35_tensor_name_candidates("lm_head.weight")
        .iter()
        .any(|name| hfq.find_tensor_info(name).is_some())
    {
        "lm_head.weight"
    } else {
        "embed_tokens.weight"
    };
    let mut heads = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for gpu in &mut gpus.devices {
        gpu.bind_thread().unwrap();
        heads.push(
            load_weight_tensor(
                &hfq,
                gpu,
                name,
                cfg.vocab_size,
                cfg.dim,
                qwen35_tensor_name_candidates,
                MemoryTarget::Device,
            )
            .unwrap(),
        );
        xs.push(gpu.alloc_tensor(&[cfg.dim], DType::F32).unwrap());
        ys.push(gpu.alloc_tensor(&[cfg.vocab_size], DType::F32).unwrap());
    }
    hipfire_runtime::ep::ensure_rank_streams(&mut gpus).unwrap();
    assert!(
        gpus.enable_peer_all().unwrap(),
        "complete peer access required"
    );
    eprintln!(
        "HEAD dtype={:?} rows={} cols={} weight_bytes_per_rank={}",
        heads[0].gpu_dtype,
        cfg.vocab_size,
        cfg.dim,
        heads[0].buf.buf.size()
    );
    let rows = cfg.vocab_size / 2;
    let mut full_ms = Vec::new();
    let mut shard_ms = Vec::new();
    for iteration in 0..16 {
        let x: Vec<f32> = (0..cfg.dim)
            .map(|i| {
                ((i as f32 * 0.731 + iteration as f32 * 0.173).sin() + (i as f32 * 0.019).cos())
                    * 0.7
            })
            .collect();
        let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_ne_bytes()).collect();
        for (rank, gpu) in gpus.devices.iter().enumerate() {
            gpu.bind_thread().unwrap();
            gpu.memcpy_htod_auto(&xs[rank].buf, &bytes).unwrap();
            gpu.hip.device_synchronize().unwrap();
        }
        let t = Instant::now();
        let ctx = hipfire_dispatch::context::DispatchCtx::new(&gpus.devices[0]);
        let wr = heads[0].dispatch_ref();
        hipfire_dispatch::pipeline::execute_steps(
            &mut gpus.devices[0],
            &ctx,
            &[hipfire_dispatch::pipeline::Step::Gemv {
                w: &wr,
                input: hipfire_dispatch::pipeline::GemvInput::Raw(&xs[0]),
                out: &ys[0],
            }],
        )
        .unwrap();
        gpus.devices[0].hip.device_synchronize().unwrap();
        let full = t.elapsed().as_secs_f64() * 1000.;
        let reference = gpus.devices[0].download_f32(&ys[0]).unwrap();
        let t = Instant::now();
        for rank in 0..2 {
            let out = ys[rank].sub_offset(rank * rows, rows);
            tp_output::project_rows(
                &mut gpus.devices[rank],
                &heads[rank],
                &xs[rank],
                &out,
                rank * rows,
                rows,
            )
            .unwrap();
        }
        gpus.barrier_rank_streams_reuse().unwrap();
        let dst = ys[0].sub_offset(rows, rows);
        let src = ys[1].sub_offset(rows, rows);
        gpus.devices[0].bind_thread().unwrap();
        gpus.devices[0]
            .hip
            .memcpy_peer_async(
                &dst.buf,
                gpus.devices[0].device_id,
                &src.buf,
                gpus.devices[1].device_id,
                rows * 4,
                gpus.devices[0].active_stream.as_ref().unwrap(),
            )
            .unwrap();
        gpus.devices[0].hip.device_synchronize().unwrap();
        let shard = t.elapsed().as_secs_f64() * 1000.;
        let actual = gpus.devices[0].download_f32(&ys[0]).unwrap();
        assert!(
            reference.iter().chain(&actual).all(|v| v.is_finite()),
            "non-finite head logits"
        );
        let unequal = reference
            .iter()
            .zip(&actual)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max_abs = reference
            .iter()
            .zip(&actual)
            .fold(0f32, |e, (a, b)| e.max((a - b).abs()));
        println!("iteration={iteration} full_ms={full:.4} shard_gather_ms={shard:.4} unequal={unequal} max_abs={max_abs:.6e}");
        assert_eq!(
            unequal, 0,
            "vocabulary row-sharding must retain every logit bit"
        );
        if iteration >= 4 {
            full_ms.push(full);
            shard_ms.push(shard);
        }
    }
    full_ms.sort_by(f64::total_cmp);
    shard_ms.sort_by(f64::total_cmp);
    println!("PASS 16/16 heads bit-exact; synthetic median full_ms={:.4} shard_gather_ms={:.4} ratio={:.3}",
        full_ms[6], shard_ms[6], full_ms[6] / shard_ms[6]);
    for (rank, head) in heads.into_iter().enumerate() {
        head.free_all(&mut gpus.devices[rank]);
    }
    for (rank, x) in xs.into_iter().enumerate() {
        gpus.devices[rank].free_tensor(x).unwrap();
    }
    for (rank, y) in ys.into_iter().enumerate() {
        gpus.devices[rank].free_tensor(y).unwrap();
    }
    gpus.free_peer_reduce_scratch().unwrap();
}
