// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only A/B of the EP unit-arena gather on a real Flash-Next artifact: the
//! serial walk over the mapped stacks (`gather_arena_mq4_planar`, the
//! `HIPFIRE_EP_GATHER_DIRECT=0` path) against the parallel `O_DIRECT` fetch +
//! parallel repack (`gather_arena_mq4_planar_fetched`). For each requested layer
//! and both tiers of rank 0 and rank 1 it drops the layer's spans from the page
//! cache, times each way and compares the two arenas byte for byte (SHA-256).
//!
//! ```text
//! cargo run --release -p hipfire-arch-qwen4 --features lab --example bench_ep_gather -- \
//!     --file MODEL.mq4 [--layers 0,10,20] [--budget-pct 45]
//! ```

use hipfire_arch_qwen4::expert_ep_units::{
    place_routed_experts_units, EpRanking, EpUnitGeometry, EpUnitSource, ShippedStack, UnitTier,
    VRAM_ARENA_TAIL_PAD_BYTES,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::hfq_parallel::{
    read_ranges_into_host, reader_tuning, DirectSource, HostRangeCopy,
};
use hipfire_runtime::model_source::ModelSource as _;
use hipfire_runtime::mq4_planar::{
    planar_bytes, shipped_bytes, Mq4PlanarKind, FLASH_NEXT_DOWN, FLASH_NEXT_GATE_UP,
};
use hipfire_runtime::weight_store::sha256_hex;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut file = PathBuf::new();
    let mut layers = vec![0usize, 10, 20];
    let mut budget_pct = 45u64;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--file" => file = PathBuf::from(value()),
            "--layers" => {
                layers = value()
                    .split(',')
                    .map(|part| part.trim().parse().expect("layer list"))
                    .collect()
            }
            "--budget-pct" => budget_pct = value().parse().expect("--budget-pct"),
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(file.is_file(), "--file MODEL is required");

    let (gr, gk) = FLASH_NEXT_GATE_UP;
    let (dr, dk) = FLASH_NEXT_DOWN;
    let gu_src = shipped_bytes(Mq4PlanarKind::GateUp, gr, gk).unwrap();
    let dn_src = shipped_bytes(Mq4PlanarKind::Down, dr, dk).unwrap();
    let geometry = EpUnitGeometry {
        layers: 1,
        experts_per_layer: 512,
        ranks: 2,
        gate_up_bytes: planar_bytes(Mq4PlanarKind::GateUp, gr, gk).unwrap() as u64,
        down_bytes: planar_bytes(Mq4PlanarKind::Down, dr, dk).unwrap() as u64,
    };
    // A fixed pseudo-random hottest-first order, a budget keeping ~budget_pct % of the rank's
    // experts in VRAM: both tiers are populated and scattered like the shipped ranking.
    let mut ids: Vec<u32> = (0..512).collect();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for i in (1..ids.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ids.swap(i, state as usize % (i + 1));
    }
    let budget = geometry.unit_bytes() * 256 * budget_pct / 100 + 2 * VRAM_ARENA_TAIL_PAD_BYTES;
    let plan = place_routed_experts_units(
        &geometry,
        &EpUnitSource::Ranking {
            ranking: EpRanking {
                rank: &ids,
                counts: None,
                sha256: None,
            },
            rank_budget_bytes: vec![budget; 2],
        },
    )
    .expect("plan");

    let hfq = HfqFile::open(&file).expect("open HFQ");
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(HfqFile::open(&file).unwrap());
    let direct = DirectSource::open_path(&file).expect("direct source");
    let drop_cache = |at: usize, len: usize| direct.drop_cached_range(at, len);
    println!("layer,rank,tier,units,mmap_s,direct_s,speedup,identical");
    let (mut total_old, mut total_new, mut mismatches) = (0.0f64, 0.0f64, 0);
    for &layer in &layers {
        let prefix = format!("model.language_model.layers.{layer}.mlp.experts");
        let gate_up_name = format!("{prefix}.gate_up_proj");
        let down_name = format!("{prefix}.down_proj");
        let gu = hfq.find_tensor_info(&gate_up_name).expect("gate_up stack");
        let dn = hfq.find_tensor_info(&down_name).expect("down stack");
        assert_eq!(
            gu.data_size,
            512 * gu_src,
            "{gate_up_name} is not shipped MQ4"
        );
        assert_eq!(dn.data_size, 512 * dn_src, "{down_name} is not shipped MQ4");
        let (gu_at, dn_at) = (gu.data_offset, dn.data_offset);
        for rank in 0..2 {
            let map = plan.layer_map(rank, 0);
            for tier in [UnitTier::Vram, UnitTier::Host] {
                // Cold for both arms.
                drop_cache(gu_at, gu.data_size);
                drop_cache(dn_at, dn.data_size);
                let started = Instant::now();
                let (_, gate_up_all) = source.tensor_data(&gate_up_name).unwrap();
                let (_, down_all) = source.tensor_data(&down_name).unwrap();
                let old = map
                    .gather_arena_mq4_planar(tier, gate_up_all, down_all)
                    .expect("mapped gather");
                let old_s = started.elapsed().as_secs_f64();

                drop_cache(gu_at, gu.data_size);
                drop_cache(dn_at, dn.data_size);
                let started = Instant::now();
                let new = map
                    .gather_arena_mq4_planar_fetched(tier, |spans, scratch| {
                        let mut at = 0usize;
                        let copies: Vec<HostRangeCopy> = spans
                            .iter()
                            .map(|&(stack, offset, len)| {
                                let base = match stack {
                                    ShippedStack::GateUp => gu_at,
                                    ShippedStack::Down => dn_at,
                                };
                                let copy = HostRangeCopy {
                                    file_offset: (base + offset) as u64,
                                    len,
                                    dst: unsafe { scratch.as_mut_ptr().add(at) },
                                };
                                at += len;
                                copy
                            })
                            .collect();
                        unsafe { read_ranges_into_host(&direct, &copies, reader_tuning()) }
                            .map(|_| ())
                            .map_err(|error| error.to_string())
                    })
                    .expect("fetched gather");
                let new_s = started.elapsed().as_secs_f64();
                let same = old == new && sha256_hex(&old) == sha256_hex(&new);
                if !same {
                    mismatches += 1;
                }
                total_old += old_s;
                total_new += new_s;
                println!(
                    "{layer},{rank},{tier:?},{},{old_s:.3},{new_s:.3},{:.1}x,{same}",
                    map.count(tier),
                    old_s / new_s
                );
            }
        }
    }
    println!(
        "total mmap {total_old:.2}s direct {total_new:.2}s ({:.1}x), {mismatches} mismatches",
        total_old / total_new
    );
    assert_eq!(mismatches, 0);
}
