// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle and hipEvent microbench of the Qwen4 MoE front producer fusion (exact gfx1201 or
//! exact gfx1151, `HIPFIRE_QWEN4_MOE_FRONT_FUSE`).
//!
//! The sealed grouped prefill's symmetric-IU4 route reads the HC `mixed` rows `[rows, 2560]` F32
//! in three producers on gfx1201 (two on gfx1151, whose F16 WMMA gate/up rotates itself and so
//! never launches the F32 one): `mq_rotate_x` (F32 rotated basis `x_rot_batch`, which no reader of
//! the route consumes), `convert_f32_to_f16` (the F16 X of the router / shared-expert WMMA group
//! launch, or on gfx1151 of the fused shared gate/up PeaceMaker launch's own router + selector
//! groups) and `mq_rotate_x_i4` (the routed gate/up `block_i4_128` sidecar).
//! `Gpu::qwen4_moe_front_producer` (`mq_rotate_x_i4_f16out`) emits the sidecar and the F16 X from
//! one read of `mixed`.
//!
//! ## Phase A, byte oracle (default)
//!
//! For every `rows` in {1, 16, 17, 511, 2047, 4099, 8188 (ragged tails), 8192, 16384} and every data set below,
//! the unfused arm (`Gpu::qwen4_convert_f16_x`, then `Gpu::qwen4_moe_rotate256_i4`) and the fused
//! arm (`Gpu::qwen4_moe_front_producer`) run from a `0xA5` sentinel in the F16 scratch and the IU4
//! sidecar region; EVERY byte of the F16 X (`rows * 2560 * 2`) and of the sidecar
//! (`2560/128 * rows * 72`) is compared. Data sets (per-row modes cycle inside `mixed`):
//! `random` (gaussian, per-row scale), `wide` (±2^e, e in [-30, 30]: F16 overflow, underflow and
//! subnormal range), `ties` (exact RNE ties between adjacent F16 values, normal and subnormal),
//! `special` (±0, ±inf, quiet / signalling NaNs with payloads, F32 subnormals, F32 max, F16 max
//! and the F16 overflow tie, F16 min subnormal and its underflow tie), `mixed` (all of those by
//! row plus all-zero rows and single-spike rows).
//!
//! ## Phase B, plumbing (rows >= 512)
//!
//! Random BF16 router / selector / gate / up weights; `Gpu::gemm_bf16_xf32_f16_wmma_qwen4_groups`
//! (converts X itself) against `Gpu::qwen4_moe_front_producer` + `..._groups_x16(Some(x16))`:
//! router logits, selector scalar, gate and up outputs compared bytewise. On gfx1151 also
//! `Gpu::gemm_bf16_xf32_f16_wmma_qwen4_shared_gu_pm(.., None)` against
//! `..._shared_gu_pm(.., Some(x16))`: router logits, post-sigmoid selector and the F16 hidden.
//!
//! ## Phase C, microbench (`--bench`)
//!
//! hipEvent time per launch (median of 3 passes of `--reps`, inputs cycled over copies totalling
//! more than the 64 MiB MALL so no arm reads a cache-warm `mixed`) at each `--rows` >= 1024:
//! the replaced chain (`mq_rotate_x` + `convert_f32_to_f16` + `mq_rotate_x_i4` on gfx1201,
//! `convert_f32_to_f16` + `mq_rotate_x_i4` on gfx1151) against the fused launch, with the
//! per-row bytes each moves.
//!
//! ```text
//! cargo run --release -p rdna-compute --features lab --example qwen4_moe_front_oracle -- [--bench] [--rows 8192,16384]
//! ```
//!
//! Exit 0 when every comparison is byte-identical, 1 on any mismatch, 2 on an unsupported device.

use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;
use std::time::Instant;

const K: usize = 2560;
const SENTINEL: u8 = 0xA5;
/// `block_i4_128` record bytes; the sidecar is `[K/128, rows]` records.
const RECORD: usize = 72;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / 16_777_216.0
    }
    fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Data {
    Random,
    Wide,
    Ties,
    Special,
    Mixed,
}

impl Data {
    const ALL: [Data; 5] = [
        Data::Random,
        Data::Wide,
        Data::Ties,
        Data::Special,
        Data::Mixed,
    ];
    fn name(self) -> &'static str {
        match self {
            Data::Random => "random",
            Data::Wide => "wide",
            Data::Ties => "ties",
            Data::Special => "special",
            Data::Mixed => "mixed",
        }
    }
}

fn wide(rng: &mut Rng) -> f32 {
    let e = (rng.next() % 61) as i32 - 30;
    let m = 1.0 + rng.unit();
    let s = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
    s * m * (2.0f32).powi(e)
}

/// An exact RNE tie: halfway between two adjacent F16 values, normal or subnormal.
fn tie(rng: &mut Rng) -> f32 {
    let negative = rng.next() & 1 == 1;
    let magnitude = if rng.next() & 3 == 0 {
        // F16 subnormal neighbourhood: 2^-24 * (m + 0.5), m in [0, 1023] (exact in F32).
        let m = (rng.next() % 1024) as f32;
        (m + 0.5) * (2.0f32).powi(-24)
    } else {
        let exp = (rng.next() % 30) as u32 + 1; // F16 biased exponent 1..=30
        let mant = (rng.next() % 1024) as u32;
        f32::from_bits(((exp + 112) << 23) | (mant << 13) | 0x1000)
    };
    if negative {
        -magnitude
    } else {
        magnitude
    }
}

fn special(rng: &mut Rng) -> f32 {
    match rng.next() % 24 {
        0 => 0.0,
        1 => -0.0,
        2 => f32::INFINITY,
        3 => f32::NEG_INFINITY,
        4 => f32::NAN,
        5 => f32::from_bits(0x7FC0_0001 | (((rng.next() & 0x3FFF) as u32) << 9)),
        6 => f32::from_bits(0x7F80_0001 | (((rng.next() & 0x3FFF) as u32) << 9)), // signalling NaN
        7 => f32::from_bits(0xFFC0_0000 | ((rng.next() & 0x7F_FFFF) as u32)),
        8 => f32::from_bits(0xFF80_0001 | ((rng.next() & 0x3F_FFFF) as u32)),
        9 => f32::from_bits((rng.next() & 0x7F_FFFF) as u32), // +F32 subnormal
        10 => f32::from_bits(0x8000_0000 | (rng.next() & 0x7F_FFFF) as u32), // -F32 subnormal
        11 => f32::MAX,
        12 => -f32::MAX,
        13 => 65504.0,
        14 => -65504.0,
        15 => 65520.0, // F16 overflow tie
        16 => 65519.996,
        17 => (2.0f32).powi(-24), // F16 min subnormal
        18 => (2.0f32).powi(-25), // underflow tie to +0
        19 => 1.5 * (2.0f32).powi(-25),
        20 => (2.0f32).powi(-14), // F16 min normal
        21 => f32::MIN_POSITIVE,
        22 => f32::EPSILON,
        _ => rng.normal(),
    }
}

fn fill_row(rng: &mut Rng, data: Data, row: usize, out: &mut [f32]) {
    let scale = 0.25 * (1 + row % 5) as f32;
    match data {
        Data::Random => out.iter_mut().for_each(|v| *v = rng.normal() * scale),
        Data::Wide => out.iter_mut().for_each(|v| *v = wide(rng)),
        Data::Ties => out.iter_mut().for_each(|v| *v = tie(rng)),
        Data::Special => out.iter_mut().for_each(|v| {
            *v = if rng.next() % 10 < 7 {
                rng.normal() * scale
            } else {
                special(rng)
            };
        }),
        Data::Mixed => match row % 8 {
            0 => out.iter_mut().for_each(|v| *v = rng.normal() * scale),
            1 => out.iter_mut().for_each(|v| *v = wide(rng)),
            2 => out.iter_mut().for_each(|v| *v = tie(rng)),
            3 => out.iter_mut().for_each(|v| *v = 0.0),
            4 => out.iter_mut().for_each(|v| {
                *v = if rng.next() % 10 < 7 {
                    rng.normal() * scale
                } else {
                    special(rng)
                };
            }),
            5 => {
                out.iter_mut().for_each(|v| *v = 0.0);
                let i = (rng.next() as usize) % out.len();
                out[i] = if row % 16 == 5 { 3.0e4 } else { -7.5 };
            }
            6 => out.iter_mut().for_each(|v| {
                *v = if rng.next() & 1 == 0 { -0.0 } else { 0.0 };
            }),
            _ => out
                .iter_mut()
                .for_each(|v| *v = rng.normal() * 0.01 * scale),
        },
    }
}

fn host_data(rows: usize, data: Data, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    let mut x = vec![0f32; rows * K];
    for (row, chunk) in x.chunks_mut(K).enumerate() {
        fill_row(&mut rng, data, row, chunk);
    }
    x
}

fn sync(gpu: &Gpu) {
    gpu.hip.device_synchronize().expect("device synchronize");
}

fn fill_device(gpu: &Gpu, ptr: *mut std::ffi::c_void, bytes: usize, val: u8) {
    let view = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) };
    gpu.hip
        .memset(&view, val as i32, bytes)
        .expect("sentinel memset");
    std::mem::forget(view);
}

fn download(gpu: &Gpu, ptr: *mut std::ffi::c_void, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    let view = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) };
    gpu.hip.memcpy_dtoh(&mut out, &view).expect("download");
    std::mem::forget(view);
    out
}

fn sidecar_snapshot(gpu: &Gpu, rows: usize) -> Vec<u8> {
    let (_, ptr) = gpu.scratch.int4_mmq_live();
    assert!(!ptr.is_null(), "no live IU4 sidecar scratch");
    download(gpu, ptr, (K / 128) * rows * RECORD)
}

/// Poison the sidecar region (the scratch is the one both producers reserve).
fn poison_sidecar(gpu: &mut Gpu, rows: usize) {
    let _ = gpu.reserve_int4_mmq(K, rows).expect("reserve sidecar");
    let (_, ptr) = gpu.scratch.int4_mmq_live();
    fill_device(gpu, ptr, (K / 128) * rows * RECORD, SENTINEL);
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or(if a.len() != b.len() {
            Some(a.len().min(b.len()))
        } else {
            None
        })
}

struct Tally {
    cases: usize,
    mismatches: usize,
}

fn phase_a(gpu: &mut Gpu, rows_list: &[usize], t: &mut Tally) {
    for &rows in rows_list {
        for (di, data) in Data::ALL.into_iter().enumerate() {
            let host = host_data(
                rows,
                data,
                0x9E37_79B9_7F4A_7C15 ^ ((rows as u64) << 8) ^ di as u64,
            );
            let x = gpu.upload_f32(&host, &[rows, K]).expect("upload x");
            let n = rows * K;
            // Unfused arm.
            let x16_ref = gpu.qwen4_convert_f16_x(&x, n).expect("convert");
            let x16_ref_bytes = download(gpu, x16_ref.buf.as_ptr(), n * 2);
            poison_sidecar(gpu, rows);
            sync(gpu);
            let _tok = gpu
                .qwen4_moe_rotate256_i4(&x, K, rows)
                .expect("rotate256_i4");
            sync(gpu);
            let side_ref = sidecar_snapshot(gpu, rows);
            // Fused arm, from sentinels.
            let scratch = gpu.qwen4_f16_x_scratch(n).expect("f16 scratch");
            fill_device(gpu, scratch.buf.as_ptr(), n * 2, SENTINEL);
            poison_sidecar(gpu, rows);
            sync(gpu);
            let (tok, x16) = gpu
                .qwen4_moe_front_producer(&x, K, rows)
                .expect("front producer");
            sync(gpu);
            gpu.int4_mmq_prepared_ptr(&tok, K, rows)
                .expect("fused sidecar token is live");
            let x16_bytes = download(gpu, x16.buf.as_ptr(), n * 2);
            let side = sidecar_snapshot(gpu, rows);
            let f16_diff = first_diff(&x16_ref_bytes, &x16_bytes);
            let side_diff = first_diff(&side_ref, &side);
            t.cases += 1;
            let ok = f16_diff.is_none() && side_diff.is_none();
            if !ok {
                t.mismatches += 1;
            }
            println!(
                "{}",
                json!({
                    "phase": "A", "rows": rows, "data": data.name(),
                    "f16_bytes": n * 2, "sidecar_bytes": side.len(),
                    "f16_identical": f16_diff.is_none(), "sidecar_identical": side_diff.is_none(),
                    "first_f16_diff_elem": f16_diff.map(|b| b / 2),
                    "first_sidecar_diff_byte": side_diff,
                })
            );
            if let Some(b) = f16_diff {
                let e = b / 2;
                eprintln!(
                    "F16 mismatch rows={rows} {} elem {e} (row {}, col {}): x={:#010x} ref={:#06x} got={:#06x}",
                    data.name(),
                    e / K,
                    e % K,
                    host[e].to_bits(),
                    u16::from_le_bytes([x16_ref_bytes[e * 2], x16_ref_bytes[e * 2 + 1]]),
                    u16::from_le_bytes([x16_bytes[e * 2], x16_bytes[e * 2 + 1]]),
                );
            }
            if let Some(b) = side_diff {
                let rec = b / RECORD;
                eprintln!(
                    "sidecar mismatch rows={rows} {} byte {b} (record {rec}: kblock {}, token {}, field byte {})",
                    data.name(),
                    rec / rows,
                    rec % rows,
                    b % RECORD
                );
            }
            for t in [x, x16_ref] {
                let _ = gpu.free_tensor(t);
            }
        }
    }
}

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
    (r >> 16) as u16
}

fn upload_bf16(gpu: &mut Gpu, rng: &mut Rng, m: usize, scale: f32) -> GpuTensor {
    let bytes: Vec<u8> = (0..m * K)
        .flat_map(|_| bf16_bits(rng.normal() * scale).to_le_bytes())
        .collect();
    let mut t = gpu.upload_raw(&bytes, &[m, K]).expect("upload bf16 weight");
    t.dtype = DType::BF16;
    t
}

fn phase_b(gpu: &mut Gpu, rows_list: &[usize], t: &mut Tally) {
    let mut rng = Rng(0xB16B_00B5_1201_0001);
    let router = upload_bf16(gpu, &mut rng, 512, 0.02);
    let selector = upload_bf16(gpu, &mut rng, 1, 0.02);
    let gate = upload_bf16(gpu, &mut rng, 640, 0.02);
    let up = upload_bf16(gpu, &mut rng, 640, 0.02);
    for &rows in rows_list.iter().filter(|&&r| r >= 512) {
        let host = host_data(rows, Data::Mixed, 0xDA7A ^ rows as u64);
        // Keep the projections finite-comparable: clamp the special rows' inf / NaN out. The
        // bytes of a NaN output are compared like any other, but F16 overflow rows are kept.
        let x = gpu.upload_f32(&host, &[rows, K]).expect("upload x");
        let mut outs = Vec::new();
        for _ in 0..2 {
            outs.push([
                gpu.zeros(&[rows, 512], DType::F32).unwrap(),
                gpu.zeros(&[rows], DType::F32).unwrap(),
                gpu.zeros(&[rows, 640], DType::F32).unwrap(),
                gpu.zeros(&[rows, 640], DType::F32).unwrap(),
            ]);
        }
        let run = |gpu: &mut Gpu, o: &[GpuTensor; 4], x16: Option<&GpuTensor>| {
            gpu.gemm_bf16_xf32_f16_wmma_qwen4_groups_x16(
                &[
                    &[(&router, &o[0], 512)],
                    &[
                        (&selector, &o[1], 1),
                        (&gate, &o[2], 640),
                        (&up, &o[3], 640),
                    ],
                ],
                &x,
                x16,
                K,
                rows,
            )
            .expect("groups")
        };
        assert!(run(gpu, &outs[0], None), "chain arm must launch");
        let (_tok, x16) = gpu
            .qwen4_moe_front_producer(&x, K, rows)
            .expect("front producer");
        assert!(run(gpu, &outs[1], Some(&x16)), "fused arm must launch");
        sync(gpu);
        let names = ["router", "selector", "gate", "up"];
        let mut all = true;
        for i in 0..4 {
            let a = gpu.download_raw_bytes(&outs[0][i]).unwrap();
            let b = gpu.download_raw_bytes(&outs[1][i]).unwrap();
            let d = first_diff(&a, &b);
            all &= d.is_none();
            println!(
                "{}",
                json!({"phase": "B", "rows": rows, "output": names[i], "bytes": a.len(),
                       "identical": d.is_none(), "first_diff_byte": d})
            );
        }
        if gpu.arch == "gfx1151" {
            // The Flash-Next shared gate/up PeaceMaker launch: its router + selector groups and
            // the gate/up/activation/hidden-conversion kernel, from its own conversion (chain)
            // against the front's F16 X (fused). Outputs: router logits, post-sigmoid selector,
            // F16 hidden.
            let mut pm = Vec::new();
            for _ in 0..2 {
                pm.push([
                    gpu.zeros(&[rows, 512], DType::F32).unwrap(),
                    gpu.zeros(&[rows], DType::F32).unwrap(),
                    gpu.zeros(&[rows, 640], DType::F16).unwrap(),
                ]);
            }
            let run_pm = |gpu: &mut Gpu, o: &[GpuTensor; 3], x16: Option<&GpuTensor>| {
                gpu.gemm_bf16_xf32_f16_wmma_qwen4_shared_gu_pm(
                    (&router, &o[0], 512),
                    (&selector, &o[1], 1),
                    &gate,
                    &up,
                    &o[2],
                    &x,
                    x16,
                    640,
                    K,
                    rows,
                )
                .expect("shared gu pm")
            };
            assert!(run_pm(gpu, &pm[0], None), "pm chain arm must launch");
            let (_tok, x16) = gpu
                .qwen4_moe_front_producer(&x, K, rows)
                .expect("front producer");
            assert!(run_pm(gpu, &pm[1], Some(&x16)), "pm fused arm must launch");
            sync(gpu);
            let names = ["pm_router", "pm_selector", "pm_hidden_f16"];
            let mut pm_all = true;
            for i in 0..3 {
                let a = gpu.download_raw_bytes(&pm[0][i]).unwrap();
                let b = gpu.download_raw_bytes(&pm[1][i]).unwrap();
                let d = first_diff(&a, &b);
                pm_all &= d.is_none();
                println!(
                    "{}",
                    json!({"phase": "B", "rows": rows, "output": names[i], "bytes": a.len(),
                           "identical": d.is_none(), "first_diff_byte": d})
                );
            }
            t.cases += 1;
            if !pm_all {
                t.mismatches += 1;
            }
        }
        t.cases += 1;
        if !all {
            t.mismatches += 1;
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn elapsed_us(gpu: &mut Gpu, reps: usize, mut launch: impl FnMut(&mut Gpu, usize)) -> f64 {
    let start = gpu.hip.event_create().expect("event");
    let stop = gpu.hip.event_create().expect("event");
    for i in 0..3 {
        launch(gpu, i);
    }
    sync(gpu);
    let mut passes = Vec::new();
    for _ in 0..3 {
        gpu.hip.event_record(&start, None).expect("record");
        for i in 0..reps {
            launch(gpu, i);
        }
        gpu.hip.event_record(&stop, None).expect("record");
        gpu.hip.event_synchronize(&stop).expect("sync event");
        passes.push(
            gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64 * 1000.0 / reps as f64,
        );
    }
    median(passes)
}

fn phase_c(gpu: &mut Gpu, rows_list: &[usize], reps: usize) {
    for &rows in rows_list.iter().filter(|&&r| r >= 1024) {
        let n = rows * K;
        let host = host_data(rows, Data::Random, 0xBE7C ^ rows as u64);
        // Cycle inputs over >= 256 MiB so no launch reads a MALL-warm `mixed`.
        let copies = ((256usize << 20) / (n * 4)).max(2);
        let xs: Vec<GpuTensor> = (0..copies)
            .map(|_| gpu.upload_f32(&host, &[rows, K]).expect("upload x"))
            .collect();
        let gfx1201 = gpu.arch == "gfx1201";
        // The F32 rotation only runs in the gfx1201 chain (gfx1151's F16 WMMA gate/up rotates
        // itself and leaves no F32 basis launch).
        let x_rot = gpu.alloc_tensor(&[rows, K], DType::F32).expect("x_rot");
        let t_rot = if gfx1201 {
            elapsed_us(gpu, reps, |g, i| {
                g.rotate_x_mq_batched(&xs[i % copies], &x_rot, K, rows)
                    .expect("mq_rotate_x");
            })
        } else {
            0.0
        };
        let t_cvt = elapsed_us(gpu, reps, |g, i| {
            g.qwen4_convert_f16_x(&xs[i % copies], n).expect("convert");
        });
        let t_i4 = elapsed_us(gpu, reps, |g, i| {
            g.qwen4_moe_rotate256_i4(&xs[i % copies], K, rows)
                .expect("rotate256_i4");
        });
        let t_fused = elapsed_us(gpu, reps, |g, i| {
            g.qwen4_moe_front_producer(&xs[i % copies], K, rows)
                .expect("front");
        });
        // The whole replaced chain launched back to back, as the route issues it.
        let t_chain = elapsed_us(gpu, reps, |g, i| {
            let x = &xs[i % copies];
            if gfx1201 {
                g.rotate_x_mq_batched(x, &x_rot, K, rows)
                    .expect("mq_rotate_x");
            }
            g.qwen4_convert_f16_x(x, n).expect("convert");
            g.qwen4_moe_rotate256_i4(x, K, rows).expect("rotate256_i4");
        });
        let before_row = if gfx1201 { K * 4 + K * 4 } else { 0 } // mq_rotate_x: read + write F32
            + K * 4 + K * 2 // convert: read F32, write F16
            + K * 4 + (K / 128) * RECORD; // mq_rotate_x_i4: read F32, write sidecar
        let after_row = K * 4 + K * 2 + (K / 128) * RECORD;
        println!(
            "{}",
            json!({
                "phase": "C", "rows": rows, "reps": reps, "inputs_cycled": copies,
                "arch": gpu.arch, "us_mq_rotate_x": t_rot, "us_convert_f32_to_f16": t_cvt, "us_mq_rotate_x_i4": t_i4,
                "us_sum_of_replaced": t_rot + t_cvt + t_i4, "us_chain_back_to_back": t_chain,
                "us_saved_chain_minus_fused": t_chain - t_fused,
                "us_fused": t_fused,
                "bytes_per_row_before": before_row, "bytes_per_row_after": after_row,
                "mb_moved_before": before_row * rows / 1_000_000, "mb_moved_after": after_row * rows / 1_000_000,
            })
        );
        for t in xs {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(x_rot);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let rows_list: Vec<usize> = value("--rows")
        .map(|s| s.split(',').map(|v| v.parse().expect("rows")).collect())
        .unwrap_or_else(|| vec![1, 16, 17, 511, 2047, 4099, 8188, 8192, 16384]);
    let reps: usize = value("--reps")
        .map(|s| s.parse().expect("reps"))
        .unwrap_or(50);
    let started = Instant::now();
    let mut gpu = Gpu::init().expect("gpu init");
    if !matches!(gpu.arch.as_str(), "gfx1201" | "gfx1151") || !gpu.qwen4_moe_front_fuse_applies() {
        eprintln!(
            "qwen4_moe_front_oracle: needs exact gfx1201 or gfx1151 with HIPFIRE_QWEN4_MOE_FRONT_FUSE on \
             (arch {}, applies {})",
            gpu.arch,
            gpu.qwen4_moe_front_fuse_applies()
        );
        std::process::exit(2);
    }
    let mut tally = Tally {
        cases: 0,
        mismatches: 0,
    };
    if !flag("--bench-only") {
        phase_a(&mut gpu, &rows_list, &mut tally);
        phase_b(&mut gpu, &rows_list, &mut tally);
    }
    if flag("--bench") || flag("--bench-only") {
        phase_c(&mut gpu, &rows_list, reps);
    }
    println!(
        "{}",
        json!({
            "summary": true, "arch": gpu.arch, "cases": tally.cases, "mismatches": tally.mismatches,
            "all_byte_identical": tally.mismatches == 0, "secs": started.elapsed().as_secs_f64(),
        })
    );
    std::process::exit(if tally.mismatches == 0 { 0 } else { 1 });
}
