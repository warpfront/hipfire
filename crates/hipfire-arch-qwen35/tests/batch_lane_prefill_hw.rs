// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! HIPF-497: a fixed-lane batch prompt seed (`Qwen35DecodeBatchState::prefill_lane`)
//! runs on the batch owner's canonical GDN stream and yields the singleton's prompt
//! logits bit for bit, including on a lane reused after `reset_lane`. Before the fix
//! every lane seed failed `GDN stream adopt_lane_meta: lane/layer/recipe mismatch`.
//!
//! `#[ignore]`d: needs a real HIP GPU and a Qwen3.5 trunk. Run:
//!
//!   HIPFIRE_BATCH_LANE_MODEL=~/.hipfire/models/qwen3.8-27b.mq4l \
//!     cargo test -p hipfire-arch-qwen35 --release --test batch_lane_prefill_hw -- --ignored

use hipfire_arch_qwen35::qwen35::{
    self, DeltaNetState, LayerType, Qwen35DecodeBatchState, Qwen35Scratch, StateQuant,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;
use std::path::Path;

const LANE_CAP: usize = 512;

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_BATCH_LANE_MODEL (Qwen3.5 trunk)"]
fn lane_prefill_matches_singleton_logits() {
    let Some(path) = std::env::var_os("HIPFIRE_BATCH_LANE_MODEL") else {
        eprintln!("skipping: HIPFIRE_BATCH_LANE_MODEL not set");
        return;
    };
    let mut hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = Gpu::init().expect("Gpu::init");
    let weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
            .expect("load_weights")
    };
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 256).expect("scratch");
    // 70 rows: one full 64-row tile plus a pending remainder in the carry.
    let tokens: Vec<u32> = (0..70u32).map(|i| 1000 + 37 * i).collect();

    // Singleton reference on the same Q8 KV geometry as one batch lane.
    let is_kv: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();
    let mut kv = KvCache::new_gpu_q8_filtered(
        &mut gpu,
        &is_kv,
        config.n_kv_heads,
        config.head_dim,
        LANE_CAP,
    )
    .expect("kv");
    let mut dn = DeltaNetState::new_with_quant(&mut gpu, &config, StateQuant::Q8).expect("dn");
    qwen35::forward_prefill_batch(
        &mut gpu, &weights, &config, &tokens, 0, &mut kv, &mut dn, &scratch, None, None, None,
    )
    .expect("singleton prefill");
    let want = gpu.download_f32(&scratch.logits).expect("singleton logits");
    let v = config.vocab_size;

    let mut batch =
        Qwen35DecodeBatchState::new(&mut gpu, &config, 2, LANE_CAP, 32).expect("batch state");
    for lane in [0, 1] {
        batch
            .prefill_lane(&mut gpu, &weights, &config, &scratch, lane, &tokens)
            .expect("prefill_lane");
    }
    batch.reset_lane(&mut gpu, &config, 1).expect("reset_lane");
    batch
        .prefill_lane(&mut gpu, &weights, &config, &scratch, 1, &tokens)
        .expect("prefill_lane after reset");
    let got = gpu.download_f32(&batch.logits).expect("batch logits");
    for lane in 0..2 {
        let row = &got[lane * v..(lane + 1) * v];
        let diff = row
            .iter()
            .zip(&want[..v])
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diff, 0,
            "lane {lane}: {diff}/{v} logits differ from the singleton"
        );
    }
}
