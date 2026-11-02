// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1201 HC row-fold microbench (`HIPFIRE_QWEN4_HC_ROW_FOLD`).
//
// hipEvent-timed: the unfused gfx1201 default tail launches vs the row-fold launches
// at the production geometry (four branches, hidden 2560).
//   MoE tail   : combine (order + zinit combine) + bf16_scaled_add_batched + hyper_write +
//                hyper_norm_gate_outputs   vs   moe_combine_order_top10 + hc_row_fold_moe_gfx1201
//   mixer tail : hyper_write + hyper_norm_gate_outputs   vs   hc_row_write_norm_gate_gfx1201
// Usage: bench_hc_row_fold_gfx1201 [rows ...]   (default 4096 16384)
// Output is one `RESULT` line per (tail, rows): median microseconds per launch group over
// BLOCKS interleaved blocks of REPEATS back-to-back iterations.

use rdna_compute::hc_row_fold::{HcRowFoldMoeGfx1201, HcRowWriteNormGateGfx1201};
use rdna_compute::tensor_ops::{
    bf16_scaled_add_batched, hyper_norm_gate_outputs, hyper_write, Bf16ScaledAddBatched,
    HyperNormGateOutputs, HyperWrite,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const M: usize = 2560;
const WIDE: usize = 4 * M;
const REPEATS: usize = 20;
const BLOCKS: usize = 7;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 8_388_608.0 - 1.0
    }
}

fn bf16_bits(rng: &mut Rng, n: usize, scale: f32) -> Vec<u16> {
    (0..n)
        .map(|_| {
            let u = (rng.unit() * scale).to_bits();
            ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
        })
        .collect()
}

fn bf16_tensor(gpu: &mut Gpu, bits: &[u16]) -> GpuTensor {
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let mut t = gpu.upload_raw(&bytes, &[bytes.len()]).expect("bf16 upload");
    t.dtype = DType::BF16;
    t.shape = vec![bits.len()];
    t
}

fn streams_tensor(gpu: &mut Gpu, bits: &[u16]) -> GpuTensor {
    let mut bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    bytes.resize(bits.len() * 4, 0);
    let mut t = gpu
        .upload_raw(&bytes, &[bytes.len()])
        .expect("streams upload");
    t.dtype = DType::F32;
    t.shape = vec![bits.len()];
    t
}

fn f32_tensor(gpu: &mut Gpu, rng: &mut Rng, n: usize, scale: f32) -> GpuTensor {
    let data: Vec<f32> = (0..n).map(|_| rng.unit() * scale).collect();
    gpu.upload_f32(&data, &[n]).expect("f32 upload")
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Microseconds per call of `body`, hipEvent timed over `REPEATS` back-to-back calls.
fn time_us(gpu: &mut Gpu, mut body: impl FnMut(&mut Gpu)) -> f64 {
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..REPEATS {
        body(gpu);
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("sync");
    gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64 * 1_000.0 / REPEATS as f64
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "exact gfx1201 required");
    let mut sizes: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    if sizes.is_empty() {
        sizes = vec![4096, 16384];
    }
    for rows in sizes {
        let mut rng = Rng(0x5eed_0000_0000_0001 ^ rows as u64);
        let grouped_rows = rows * 10 + 16;
        let norm_weight = bf16_tensor(&mut gpu, &bf16_bits(&mut rng, WIDE, 0.3));
        let gate_weight = bf16_tensor(&mut gpu, &bf16_bits(&mut rng, 4 * WIDE, 0.05));
        let grouped = bf16_tensor(&mut gpu, &bf16_bits(&mut rng, grouped_rows * M, 2.0));
        // A permutation of the grouped rows (every row read once), like a real route.
        let inverse: Vec<i32> = (0..rows * 10)
            .map(|i| ((i * 7919 + 13) % (rows * 10)) as i32)
            .collect();
        let experts: Vec<i32> = (0..rows * 10)
            .map(|i| ((i * 37 + i / 10) % 64) as i32)
            .collect();
        let ints = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let inverse = gpu
            .upload_raw(&ints(&inverse), &[rows * 40])
            .expect("inverse");
        let experts = gpu
            .upload_raw(&ints(&experts), &[rows * 40])
            .expect("experts");
        let weights = {
            let data: Vec<f32> = (0..rows * 10).map(|_| rng.unit().abs() / 3.0).collect();
            gpu.upload_f32(&data, &[rows * 10]).expect("weights")
        };
        let shared = f32_tensor(&mut gpu, &mut rng, rows * M, 2.0);
        let selector = f32_tensor(&mut gpu, &mut rng, rows, 1.0);
        let gates = f32_tensor(&mut gpu, &mut rng, rows * 4, 3.0);
        let mixed = f32_tensor(&mut gpu, &mut rng, rows * M, 2.0);
        let order = gpu.zeros(&[rows * 20], DType::F32).expect("order");
        let routed = gpu.zeros(&[rows * M], DType::F32).expect("routed");
        let normalized = gpu.zeros(&[rows * WIDE], DType::F32).expect("normalized");
        let stream_bits = bf16_bits(&mut rng, rows * WIDE, 3.0);
        let streams = streams_tensor(&mut gpu, &stream_bits);
        drop(stream_bits);
        let next_gates = gpu.zeros(&[rows * 4], DType::F32).expect("next gates");
        let row16 = gpu.zeros(&[rows * WIDE], DType::F16).expect("f16 row");

        macro_rules! combine {
            ($gpu:expr) => {
                $gpu.moe_down_combine_grouped_top10_bf16in(
                    &grouped,
                    &inverse,
                    &experts,
                    &weights,
                    &routed,
                    &order,
                    M,
                    grouped_rows,
                    rows,
                    true,
                    64,
                )
                .expect("combine")
            };
        }
        macro_rules! scaled_add {
            ($gpu:expr) => {
                bf16_scaled_add_batched(
                    $gpu,
                    &Bf16ScaledAddBatched {
                        residual: &routed,
                        value: &shared,
                        scalar: &selector,
                        rows,
                        elements: M,
                    },
                )
                .expect("scaled add")
            };
        }
        macro_rules! write {
            ($gpu:expr, $mixed:expr) => {
                hyper_write(
                    $gpu,
                    &HyperWrite {
                        input: &streams,
                        normalized: &normalized,
                        mixed: $mixed,
                        gates: &gates,
                        output: &streams,
                        branches: 4,
                        hidden: M,
                        state_bf16: true,
                    },
                )
                .expect("hyper_write")
            };
        }
        macro_rules! norm_gate {
            ($gpu:expr) => {
                hyper_norm_gate_outputs(
                    $gpu,
                    &HyperNormGateOutputs {
                        input: &streams,
                        norm_weight: &norm_weight,
                        gate_weight: &gate_weight,
                        gates: &next_gates,
                        normalized_f16: &row16,
                        rows,
                        branches: 4,
                        hidden: M,
                        state_bf16: true,
                    },
                )
                .expect("norm gate")
            };
        }
        macro_rules! moe_fold {
            ($gpu:expr) => {{
                $gpu.moe_combine_order_top10(
                    &inverse,
                    &experts,
                    &weights,
                    &order,
                    grouped_rows,
                    rows,
                    64,
                )
                .expect("order");
                $gpu.hc_row_fold_moe_gfx1201(&HcRowFoldMoeGfx1201 {
                    grouped_down: &grouped,
                    order: &order,
                    shared_f32: &shared,
                    selector: &selector,
                    streams: &streams,
                    gates: &gates,
                    norm_weight: &norm_weight,
                    gate_weight: &gate_weight,
                    next_gates: &next_gates,
                    normalized_f16: &row16,
                    rows,
                    hidden: M,
                })
                .expect("moe fold")
            }};
        }
        macro_rules! order_only {
            ($gpu:expr) => {
                $gpu.moe_combine_order_top10(
                    &inverse,
                    &experts,
                    &weights,
                    &order,
                    grouped_rows,
                    rows,
                    64,
                )
                .expect("order")
            };
        }
        macro_rules! mixer_fold {
            ($gpu:expr) => {
                $gpu.hc_row_write_norm_gate_gfx1201(&HcRowWriteNormGateGfx1201 {
                    mixed: &mixed,
                    streams: &streams,
                    gates: &gates,
                    norm_weight: &norm_weight,
                    gate_weight: &gate_weight,
                    next_gates: &next_gates,
                    normalized_f16: &row16,
                    rows,
                    hidden: M,
                })
                .expect("mixer fold")
            };
        }

        // Warm every kernel (JIT) before timing.
        combine!(&mut gpu);
        scaled_add!(&mut gpu);
        write!(&mut gpu, &routed);
        norm_gate!(&mut gpu);
        moe_fold!(&mut gpu);
        write!(&mut gpu, &mixed);
        mixer_fold!(&mut gpu);
        gpu.hip.device_synchronize().expect("warm sync");

        let (mut combine_us, mut add_us, mut write_us, mut norm_us) =
            (vec![], vec![], vec![], vec![]);
        let (mut moe_unfused, mut moe_fused, mut order_us) = (vec![], vec![], vec![]);
        let (mut mixer_write_us, mut mixer_unfused, mut mixer_fused) = (vec![], vec![], vec![]);
        for block in 0..BLOCKS {
            // Alternate the order so neither side owns the cold/hot half.
            let order_fused_first = block & 1 == 1;
            let fused = |gpu: &mut Gpu| time_us(gpu, |gpu| moe_fold!(gpu));
            let mixer_fused_us = |gpu: &mut Gpu| time_us(gpu, |gpu| mixer_fold!(gpu));
            if order_fused_first {
                moe_fused.push(fused(&mut gpu));
                mixer_fused.push(mixer_fused_us(&mut gpu));
            }
            let c = time_us(&mut gpu, |gpu| combine!(gpu));
            order_us.push(time_us(&mut gpu, |gpu| order_only!(gpu)));
            let a = time_us(&mut gpu, |gpu| scaled_add!(gpu));
            let w = time_us(&mut gpu, |gpu| write!(gpu, &routed));
            let n = time_us(&mut gpu, |gpu| norm_gate!(gpu));
            let mw = time_us(&mut gpu, |gpu| write!(gpu, &mixed));
            combine_us.push(c);
            add_us.push(a);
            write_us.push(w);
            norm_us.push(n);
            moe_unfused.push(c + a + w + n);
            mixer_write_us.push(mw);
            mixer_unfused.push(mw + n);
            if !order_fused_first {
                moe_fused.push(fused(&mut gpu));
                mixer_fused.push(mixer_fused_us(&mut gpu));
            }
        }
        let (c, a, w, n) = (
            median(&mut combine_us),
            median(&mut add_us),
            median(&mut write_us),
            median(&mut norm_us),
        );
        let (mw, mu) = (median(&mut mixer_write_us), median(&mut mixer_unfused));
        let (mf, uf) = (median(&mut moe_fused), median(&mut moe_unfused));
        let ord = median(&mut order_us);
        let xf = median(&mut mixer_fused);
        println!(
            "RESULT moe rows={rows} combine={c:.1}us scaled_add={a:.1}us hyper_write={w:.1}us norm_gate={n:.1}us \
             unfused_sum={uf:.1}us fused={mf:.1}us (order={ord:.1}us fold_only={:.1}us) saved={:.1}us ({:.1}%)",
            mf - ord,
            uf - mf,
            100.0 * (uf - mf) / uf
        );
        println!(
            "RESULT mixer rows={rows} hyper_write={mw:.1}us norm_gate={n:.1}us unfused_sum={mu:.1}us \
             fused={xf:.1}us saved={:.1}us ({:.1}%)",
            mu - xf,
            100.0 * (mu - xf) / mu
        );
        for t in [
            norm_weight,
            gate_weight,
            grouped,
            inverse,
            experts,
            weights,
            shared,
            selector,
            gates,
            mixed,
            order,
            routed,
            normalized,
            streams,
            next_gates,
            row16,
        ] {
            gpu.free_tensor(t).expect("free");
        }
    }
}
