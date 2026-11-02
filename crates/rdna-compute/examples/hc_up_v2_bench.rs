// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipEvent microbench: the persistent gfx1201 / gfx1151 Qwen4 HC up+mix read tail
//! `hyper_read_up_wmma_bf16_gfx1201_v2` / `_gfx1151_v2` against the retiled `_t128` / `_swap` entry it replaces, both through
//! `tensor_ops::hyper_read_up_wmma_route`. hidden 2560, low_rank 320, `up_weight` BF16 [4 * 2560, 320],
//! `low` BF16 [rows, 320], `normalized` F16 [rows, 4 * 2560]; random inputs (the oracle's class a).
//!
//! Per row count and route: 9 batches of 40 back-to-back calls, one event pair per batch -> per-call
//! min and median us over the batches.
//!
//! Bytes per row. DRAM is identical before and after:
//!   normalized read 4 * 2560 * 2 = 20480 B + `mixed` write 2560 * 4 = 10240 B + `low` read 320 * 2
//!   = 640 B = 31360 B.
//! L2-to-LDS weight staging per row:
//!   before (`_t128`): 160 column tiles * 40960 B (64 weight rows x 320 x 2 B) per 128 tokens = 51200 B.
//!   after (`_v2`): the same 160 tiles * 40960 B once per block, a block covering ceil(rows / G) tokens
//!   (G = `hc_up_v2_groups(rows)`), i.e. 160 * 40960 / ceil(rows / G) B per row; plus the 16 KiB LDS table
//!   (2 x 2048 entries), which is built in LDS and generates no global traffic, and which replaces the
//!   baseline's exp / divide chain (about 60 VALU instructions per gate).
//!
//! `ROWS=a,b,..` overrides the row list. Run:
//! cargo run --release --features lab --example hc_up_v2_bench -p rdna-compute
//! (the repo rule is one fresh process per measurement; loop in the shell).

use half::f16;
use rdna_compute::tensor_ops::{
    hc_up_v2_1151_groups, hc_up_v2_groups, hyper_read_up_wmma_route, HcUpRoute, HyperReadUpFused,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const HIDDEN: usize = 2560;
const LOW_RANK: usize = 320;
const WIDE: usize = 4 * HIDDEN;
const CALLS: usize = 40;
const BATCHES: usize = 9;

/// normalized read + mixed write + low read.
const DRAM_BYTES_PER_ROW: usize = WIDE * 2 + HIDDEN * 4 + LOW_RANK * 2;
/// 160 column tiles of 64 weight rows x 320 x 2 B.
const TILE_BYTES_TOTAL: usize = (HIDDEN / 16) * 64 * LOW_RANK * 2;

fn lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0) * scale
        })
        .collect()
}

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn upload_u16(gpu: &mut Gpu, v: &[u16], dtype: DType) -> GpuTensor {
    let bytes: Vec<u8> = v.iter().flat_map(|h| h.to_le_bytes()).collect();
    let mut t = gpu.upload_raw(&bytes, &[v.len()]).expect("upload");
    t.dtype = dtype;
    t.shape = vec![v.len()];
    t
}

/// Per-call microseconds of `CALLS` back-to-back calls, one event pair around the batch.
fn batch_us(gpu: &mut Gpu, f: &mut impl FnMut(&mut Gpu)) -> f64 {
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..CALLS {
        f(gpu);
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
    gpu.hip.event_destroy(start).unwrap();
    gpu.hip.event_destroy(stop).unwrap();
    ms * 1000.0 / CALLS as f64
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// (min, median) per-call us over `BATCHES` batches, after a warm-up.
fn batched(gpu: &mut Gpu, mut f: impl FnMut(&mut Gpu)) -> (f64, f64) {
    for _ in 0..CALLS {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let mut v: Vec<f64> = (0..BATCHES).map(|_| batch_us(gpu, &mut f)).collect();
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    (min, median(&mut v))
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let available =
        (gpu.arch == "gfx1201" && gpu.qwen4_f16_wmma_gfx1201()) || gpu.arch == "gfx1151";
    println!("arch {} v2_available={available}", gpu.arch);
    if !available {
        eprintln!("V2 is not available: it needs exact gfx1201 (F16 WMMA route) or exact gfx1151");
        std::process::exit(2);
    }
    println!(
        "DRAM bytes/row, identical before and after: normalized read {} + mixed write {} + low read {} = {DRAM_BYTES_PER_ROW} B",
        WIDE * 2,
        HIDDEN * 4,
        LOW_RANK * 2
    );
    println!(
        "L2->LDS weight staging per row: before (t128) = {} tiles * {} B per 128 tokens = {} B; \
         after (v2) = {} tiles * {} B per ceil(rows/G) tokens per block (see table), plus a 16 KiB LDS-built \
         gate table that generates no global traffic and replaces the baseline's exp/divide chain",
        HIDDEN / 16,
        TILE_BYTES_TOTAL / (HIDDEN / 16),
        TILE_BYTES_TOTAL / 128,
        HIDDEN / 16,
        TILE_BYTES_TOTAL / (HIDDEN / 16)
    );
    println!("timing: {CALLS} calls/batch x {BATCHES} batches (min/med per call, us)");

    let rows_list: Vec<usize> = std::env::var("ROWS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter(|t| !t.is_empty())
                .map(|t| t.parse().unwrap())
                .collect()
        })
        .unwrap_or_else(|| vec![512, 1024, 2048, 4096, 8192, 16384]);
    let max_rows = *rows_list.iter().max().unwrap();

    let wb: Vec<u16> = lcg(WIDE * LOW_RANK, 11, 0.15 * 1.7)
        .iter()
        .map(|&v| bf16_bits(v))
        .collect();
    let weight = upload_u16(&mut gpu, &wb, DType::BF16);
    drop(wb);
    let lb: Vec<u16> = lcg(max_rows * LOW_RANK, 17, 1.7)
        .iter()
        .map(|&v| bf16_bits(v))
        .collect();
    let nb: Vec<u16> = lcg(max_rows * WIDE, 23, 3.4)
        .iter()
        .map(|&v| f16::from_f32(v).to_bits())
        .collect();

    println!(
        "{:>6} {:>2} {:>9} {:>9} {:>9} {:>9} {:>8} {:>10} {:>10} {:>10}",
        "rows",
        "G",
        "t128_min",
        "t128_med",
        "v2_min",
        "v2_med",
        "speedup",
        "DRAM_B/row",
        "stage_bef",
        "stage_aft"
    );
    for &rows in &rows_list {
        let low = upload_u16(&mut gpu, &lb[..rows * LOW_RANK], DType::BF16);
        let norm = upload_u16(&mut gpu, &nb[..rows * WIDE], DType::F16);
        let mixed = gpu.zeros(&[rows * HIDDEN], DType::F32).unwrap();
        let groups = if gpu.arch == "gfx1151" {
            hc_up_v2_1151_groups(rows)
        } else {
            hc_up_v2_groups(rows)
        };
        let mut time = |route: HcUpRoute| {
            batched(&mut gpu, |g| {
                hyper_read_up_wmma_route(
                    g,
                    &HyperReadUpFused {
                        up_weight: &weight,
                        low: &low,
                        normalized: &norm,
                        mixed: &mixed,
                        rows,
                        hidden: HIDDEN,
                        low_rank: LOW_RANK,
                        normalized_bf16: true,
                    },
                    route,
                )
                .unwrap();
            })
        };
        let (tmin, tmed) = time(HcUpRoute::T128);
        let (vmin, vmed) = time(HcUpRoute::V2);
        let stage_before = TILE_BYTES_TOTAL as f64 / 128.0;
        let stage_after = TILE_BYTES_TOTAL as f64 / rows.div_ceil(groups) as f64;
        println!(
            "{rows:>6} {groups:>2} {tmin:>9.1} {tmed:>9.1} {vmin:>9.1} {vmed:>9.1} {:>7.3}x {DRAM_BYTES_PER_ROW:>10} {stage_before:>10.0} {stage_after:>10.0}",
            tmin / vmin
        );
        for t in [low, norm, mixed] {
            gpu.free_tensor(t).unwrap();
        }
    }
    println!(
        "units: us per call; speedup = t128_min / v2_min; stage_* = L2->LDS weight bytes per row"
    );
}
