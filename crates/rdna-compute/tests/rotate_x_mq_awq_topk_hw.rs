// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware gate for `Gpu::rotate_x_mq_awq_topk` (module `rotate_x_mq_awq_topk`): the per-routed-expert
//! AWQ + FWHT-256 producer of the Inkling-2 `qt44-padded-awq-v1` gate_up input.
//!
//! * every rank row equals `rotate_x_mq_awq(x, S[expert_ids[r]])` bit for bit (same arithmetic) and a
//!   CPU f32 FWHT of `x / S` within rounding;
//! * a planted wrong expert id changes exactly that row, and each row follows ITS id (not the rank);
//! * with `S == 1` every row is bitwise `rotate_x_mq(x)`;
//! * an out-of-range id (`-1`, `n_experts`, huge) yields a zero row over a NaN-poisoned output.
//!
//! `#[ignore]`d: needs a GPU (any arch with the wave32 rotate kernels), no model files.
//!
//!   cargo test -p rdna-compute --release --test rotate_x_mq_awq_topk_hw -- --ignored --nocapture

use rdna_compute::{DType, Gpu, GpuTensor};

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f32 {
        (self.next_u32() & 0xFF_FFFF) as f32 / (1u32 << 23) as f32 - 1.0
    }
    /// Positive scale in [0.25, 4.25).
    fn scale(&mut self) -> f32 {
        0.25 + (self.next_u32() & 0xFFFF) as f32 / 16384.0
    }
}

fn fwht_signs(seed: u32) -> [f32; 256] {
    let mut state = seed;
    let mut out = [0.0f32; 256];
    for v in out.iter_mut() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
        *v = if (state >> 16) & 1 == 1 { 1.0 } else { -1.0 };
    }
    out
}

/// Straight f32 FWHT-256 per group (signs1, butterfly, 1/16, signs2).
fn cpu_rotate(x: &[f32], s: &[f32]) -> Vec<f32> {
    let s1 = fwht_signs(42);
    let s2 = fwht_signs(1042);
    let mut out = vec![0.0f32; x.len()];
    for (g, (xg, sg)) in x.chunks_exact(256).zip(s.chunks_exact(256)).enumerate() {
        let mut v = [0.0f32; 256];
        for i in 0..256 {
            v[i] = (xg[i] / sg[i]) * s1[i];
        }
        let mut stride = 1;
        while stride < 256 {
            let mut j = 0;
            while j < 256 {
                for k in 0..stride {
                    let a = v[j + k];
                    let b = v[j + k + stride];
                    v[j + k] = a + b;
                    v[j + k + stride] = a - b;
                }
                j += 2 * stride;
            }
            stride <<= 1;
        }
        for i in 0..256 {
            out[g * 256 + i] = v[i] * 0.0625 * s2[i];
        }
    }
    out
}

fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    gpu.upload_f32(data, &[data.len()]).expect("upload f32")
}

fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
    gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload i32")
}

fn download_f32(gpu: &mut Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
    let mut v = gpu.download_f32(t).expect("download");
    v.truncate(n);
    v
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

struct Table {
    scales: Vec<Vec<f32>>,
    tensors: Vec<GpuTensor>,
    ptrs: GpuTensor,
}

fn build_table(gpu: &mut Gpu, rng: &mut Lcg, n_experts: usize, k: usize, ones: bool) -> Table {
    let mut scales = Vec::new();
    let mut tensors = Vec::new();
    for _ in 0..n_experts {
        let s: Vec<f32> = (0..k)
            .map(|_| if ones { 1.0 } else { rng.scale() })
            .collect();
        tensors.push(upload_f32(gpu, &s));
        scales.push(s);
    }
    let bytes: Vec<u8> = tensors
        .iter()
        .flat_map(|t| (t.buf.as_ptr() as u64).to_ne_bytes())
        .collect();
    let ptrs = gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload ptrs");
    Table {
        scales,
        tensors,
        ptrs,
    }
}

fn run_case(gpu: &mut Gpu, k: usize, ids: &[i32], n_experts: usize, ones: bool, seed: u64) {
    let top_k = ids.len();
    let mut rng = Lcg(seed);
    let x: Vec<f32> = (0..k).map(|_| rng.signed() * 3.0).collect();
    let table = build_table(gpu, &mut rng, n_experts, k, ones);
    let xt = upload_f32(gpu, &x);
    let idt = upload_i32(gpu, ids);
    let poison = vec![f32::NAN; top_k * k];
    let out = upload_f32(gpu, &poison);
    gpu.rotate_x_mq_awq_topk(&xt, &table.ptrs, &idt, &out, k, top_k, n_experts)
        .expect("rotate_x_mq_awq_topk");
    let got = download_f32(gpu, &out, top_k * k);

    for (r, &id) in ids.iter().enumerate() {
        let row = &got[r * k..(r + 1) * k];
        if id < 0 || id as usize >= n_experts {
            assert!(
                row.iter().all(|&v| v.to_bits() == 0),
                "k={k} rank {r} id {id}: out-of-range id must give an all-zero row"
            );
            continue;
        }
        let e = id as usize;
        // Same-arithmetic oracle: the dense AWQ rotate with this expert's own vector.
        let one = gpu.alloc_tensor(&[k], DType::F32).expect("alloc");
        gpu.rotate_x_mq_awq(&xt, &table.tensors[e], &one, k)
            .expect("rotate_x_mq_awq");
        let want = download_f32(gpu, &one, k);
        assert_eq!(
            bits(row),
            bits(&want),
            "k={k} rank {r} expert {e}: not bitwise rotate_x_mq_awq"
        );
        // Independent CPU reference (association order differs: tolerance).
        let cpu = cpu_rotate(&x, &table.scales[e]);
        let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = row
            .iter()
            .zip(&cpu)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst <= 1e-5 * scale,
            "k={k} rank {r} expert {e}: max |gpu-cpu| {worst} vs scale {scale}"
        );
        if ones {
            let plain = gpu.alloc_tensor(&[k], DType::F32).expect("alloc");
            gpu.rotate_x_mq(&xt, &plain, k).expect("rotate_x_mq");
            let want = download_f32(gpu, &plain, k);
            assert_eq!(
                bits(row),
                bits(&want),
                "k={k} rank {r}: S=1 must be bitwise rotate_x_mq"
            );
            gpu.free_tensor(plain).ok();
        }
        gpu.free_tensor(one).ok();
    }
    for t in table.tensors {
        gpu.free_tensor(t).ok();
    }
    gpu.free_tensor(table.ptrs).ok();
    for t in [xt, idt, out] {
        gpu.free_tensor(t).ok();
    }
}

#[test]
#[ignore = "needs a GPU"]
fn rotate_x_mq_awq_topk_matches_per_expert_awq_rotate_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    // Inkling gate_up (K=2304, 8 ranks, 128 experts), down-width (1280), single group.
    run_case(
        &mut gpu,
        2304,
        &[5, 127, 0, 64, 33, 90, 11, 77],
        128,
        false,
        1,
    );
    run_case(&mut gpu, 1280, &[3, 1, 2, 0, 3, 1, 2, 0], 4, false, 2);
    run_case(&mut gpu, 256, &[0, 1], 2, false, 3);
}

#[test]
#[ignore = "needs a GPU"]
fn rotate_x_mq_awq_topk_follows_ids_not_ranks_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    let k = 2304;
    let n_experts = 16;
    let mut rng = Lcg(77);
    let x: Vec<f32> = (0..k).map(|_| rng.signed()).collect();
    let table = build_table(&mut gpu, &mut rng, n_experts, k, false);
    let xt = upload_f32(&mut gpu, &x);
    let run = |gpu: &mut Gpu, ids: &[i32]| -> Vec<f32> {
        let idt = upload_i32(gpu, ids);
        let out = upload_f32(gpu, &vec![f32::NAN; ids.len() * k]);
        gpu.rotate_x_mq_awq_topk(&xt, &table.ptrs, &idt, &out, k, ids.len(), n_experts)
            .unwrap();
        let v = download_f32(gpu, &out, ids.len() * k);
        gpu.free_tensor(idt).ok();
        gpu.free_tensor(out).ok();
        v
    };
    let base = run(&mut gpu, &[1, 2, 3, 4, 5, 6, 7, 8]);
    // Plant a wrong expert id at rank 3 only.
    let planted = run(&mut gpu, &[1, 2, 3, 9, 5, 6, 7, 8]);
    for r in 0..8 {
        let same = bits(&base[r * k..(r + 1) * k]) == bits(&planted[r * k..(r + 1) * k]);
        assert_eq!(same, r != 3, "rank {r}: only the planted rank may change");
    }
    // Rank 3 of the planted run is exactly rank 0 of a run that selects expert 9 first.
    let nine = run(&mut gpu, &[9]);
    assert_eq!(bits(&planted[3 * k..4 * k]), bits(&nine[..k]));
    // Permuting ids permutes rows.
    let rev = run(&mut gpu, &[8, 7, 6, 5, 4, 3, 2, 1]);
    for r in 0..8 {
        assert_eq!(
            bits(&base[r * k..(r + 1) * k]),
            bits(&rev[(7 - r) * k..(8 - r) * k])
        );
    }
    // Out-of-range ids: zero rows, in-range neighbours untouched.
    let oob = run(
        &mut gpu,
        &[1, -1, 3, n_experts as i32, 5, i32::MAX, 7, i32::MIN],
    );
    for r in 0..8 {
        let row = &oob[r * k..(r + 1) * k];
        if matches!(r, 1 | 3 | 5 | 7) {
            assert!(row.iter().all(|&v| v.to_bits() == 0), "rank {r} not zero");
        } else {
            assert_eq!(bits(row), bits(&base[r * k..(r + 1) * k]), "rank {r}");
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn rotate_x_mq_awq_topk_unit_scale_is_bitwise_rotate_x_mq_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    run_case(&mut gpu, 2304, &[0, 1, 2, 3, 3, 2, 1, 0], 4, true, 9);
    run_case(&mut gpu, 3072, &[2, 2, 1], 3, true, 10);
}

#[test]
#[ignore = "needs a GPU"]
fn rotate_x_mq_awq_topk_rejects_bad_extents_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    let k = 2304;
    let mut rng = Lcg(5);
    let table = build_table(&mut gpu, &mut rng, 2, k, true);
    let xt = upload_f32(&mut gpu, &vec![1.0; k]);
    let idt = upload_i32(&mut gpu, &[0, 1]);
    let out = upload_f32(&mut gpu, &vec![0.0; 2 * k]);
    // k not a multiple of 256, out too small, table too small.
    assert!(gpu
        .rotate_x_mq_awq_topk(&xt, &table.ptrs, &idt, &out, 2000, 2, 2)
        .is_err());
    assert!(gpu
        .rotate_x_mq_awq_topk(&xt, &table.ptrs, &idt, &out, k, 3, 2)
        .is_err());
    assert!(gpu
        .rotate_x_mq_awq_topk(&xt, &table.ptrs, &idt, &out, k, 2, 3)
        .is_err());
}
