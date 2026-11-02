// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness oracle of the lane-batched FullAttention verify chain of the
//! DFlash continuous-batching trunk (exact gfx1201, native fp8-E4M3 KV)
//! against today's per-lane launch sequence.
//!
//! 27B FullAttention shapes (n_heads 24, n_kv_heads 4, head_dim 256, n_rot
//! 64, rope theta 1e7); lanes in {1, 3, 8} x n in {2, 8, 16} (every lane the
//! same n) x two context patterns drawn from {100, 1000, 5000, 20000}
//! tokens, so lanes of one chunk have DIFFERENT context lengths (different
//! `max_tiles`, split counts and tile counts). All lanes live in ONE shared
//! row scratch (lane i owns rows `[i * n, (i + 1) * n)`) with random
//! pre-prep Q+gate / K / V rows and per-row positions `start_i + j`; each
//! lane has its own fp8 KV caches pre-filled (random valid E4M3 codes and
//! f16 scales, no NaN codes) up to `start_i`.
//!
//! - arm A (today): per lane, on that lane's row views: the unfused prep
//!   (`deinterleave_q_rmsnorm_f32_batched`, in-place K `rmsnorm_batched`,
//!   `rope_partial_interleaved_f32_batched`), `kv_cache_write_fp8_e4m3_batched`
//!   for K then V, and `attention_flash_fp8_e4m3_tile_batched` (the VerifyAttn
//!   tile + reduce, or the native-fp8 split-KV verifier for a lane at or past
//!   its 8,192-token default floor) with that lane's own partials buffer;
//! - arm B (lane-batched): the three prep launches once over all rows, two
//!   `Gpu::kv_cache_write_fp8_e4m3_lanes`, then
//!   `Gpu::attention_verify_fp8_lanes_routed`: one
//!   `Gpu::attention_verify_gqa_fp8_lanes` + one
//!   `Gpu::attention_verify_reduce_lanes` over one shared partials buffer
//!   holding the disjoint region of every lane the singleton serves with
//!   VerifyAttn, and that lane's own singleton launch for every lane the
//!   singleton serves with the split-KV verifier (the lane-batched CB verify
//!   takes the same route as the singleton, so batched == singleton above
//!   the floor too; run it with the default route, unset
//!   `HIPFIRE_GFX12_FA2_SPLIT_VERIFY_FP8`, and with `=0`).
//!
//! Compared as bytes: the prep outputs (Q, gate, K, V rows), the attention
//! output rows (pre-filled with a sentinel so a missed store shows), every
//! lane's full K and V cache (so the written rows AND the untouched prefix),
//! and every lane's VerifyAttn partials (zero-initialised in both arms).
//! Prints PASS/FAIL per case; exits nonzero on any mismatch.
//!
//! `cargo run --release -p rdna-compute --features lab,deltanet --example lane_fa_oracle`

use rdna_compute::lane_fa::{LaneFaAttend, LaneFaWrite, LANE_FA_PARTIALS_STRIDE};
use rdna_compute::{DType, Gpu, GpuTensor};

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const Q_DIM: usize = NH * HD;
const KV_DIM: usize = NKV * HD;
const N_ROT: usize = 64;
const THETA: f32 = 10_000_000.0;
const EPS: f32 = 1e-6;
/// Bytes of one fp8 KV row: `NKV * HD` codes + `NKV` f16 scales.
const ROW_BYTES: usize = NKV * (HD + 2);
const CTXS: [usize; 4] = [100, 1000, 5000, 20000];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Uniform in [-1, 1).
    fn sym(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

fn f32s(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    (0..n).map(|_| rng.sym() * amp).collect()
}

/// A random fp8 cache of `cap` rows, the first `filled` rows valid (random
/// non-NaN E4M3 codes, f16 scales in [2^-7, 2^-3)), the rest zero.
fn make_cache(rng: &mut Rng, filled: usize, cap: usize) -> Vec<u8> {
    let mut c = vec![0u8; cap * ROW_BYTES];
    for pos in 0..filled {
        let row = &mut c[pos * ROW_BYTES..(pos + 1) * ROW_BYTES];
        let (codes, scales) = row.split_at_mut(NKV * HD);
        for chunk in codes.chunks_mut(8) {
            let w = rng.next().to_le_bytes();
            for (b, r) in chunk.iter_mut().zip(w) {
                // 0x7f / 0xff are the E4M3FN NaN codes.
                *b = if r & 0x7f == 0x7f { r & 0xbf } else { r };
            }
        }
        for h in 0..NKV {
            let exp = 8 + (rng.next() % 5) as u16; // 2^-7 .. 2^-3
            let mant = (rng.next() & 0x3ff) as u16;
            let bits = (exp << 10) | mant;
            scales[h * 2..h * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    c
}

/// Host-side initial contents shared by both arms.
struct Host {
    q_full: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    pos: Vec<f32>,
    k_cache: Vec<Vec<u8>>,
    v_cache: Vec<Vec<u8>>,
}

fn make_host(rng: &mut Rng, starts: &[usize], n: usize) -> Host {
    let total = starts.len() * n;
    Host {
        q_full: f32s(rng, total * 2 * Q_DIM, 1.0),
        k: f32s(rng, total * KV_DIM, 1.0),
        v: f32s(rng, total * KV_DIM, 1.0),
        // Positions are i32 bits in an F32 tensor, like the scratch's.
        pos: starts
            .iter()
            .flat_map(|&s| (0..n).map(move |j| f32::from_bits((s + j) as u32)))
            .collect(),
        k_cache: starts.iter().map(|&s| make_cache(rng, s, s + n)).collect(),
        v_cache: starts.iter().map(|&s| make_cache(rng, s, s + n)).collect(),
    }
}

/// Device buffers of one arm.
struct Arm {
    q_full: GpuTensor,
    q: GpuTensor,
    gate: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    out: GpuTensor,
    pos: GpuTensor,
    k_cache: Vec<GpuTensor>,
    v_cache: Vec<GpuTensor>,
    /// Arm A: one partials buffer per lane; arm B: a single shared one.
    partials: Vec<GpuTensor>,
}

fn make_arm(gpu: &mut Gpu, h: &Host, total: usize, partials_floats: &[usize]) -> Arm {
    let sentinel = |gpu: &mut Gpu, w: usize| {
        gpu.upload_f32(&vec![f32::from_bits(0x7fc0_dead); total * w], &[total * w])
            .expect("alloc sentinel")
    };
    Arm {
        q_full: gpu
            .upload_f32(&h.q_full, &[h.q_full.len()])
            .expect("q_full"),
        q: sentinel(gpu, Q_DIM),
        gate: sentinel(gpu, Q_DIM),
        k: gpu.upload_f32(&h.k, &[h.k.len()]).expect("k"),
        v: gpu.upload_f32(&h.v, &[h.v.len()]).expect("v"),
        out: sentinel(gpu, Q_DIM),
        pos: gpu.upload_f32(&h.pos, &[h.pos.len()]).expect("pos"),
        k_cache: h
            .k_cache
            .iter()
            .map(|c| gpu.upload_raw(c, &[c.len()]).expect("k_cache"))
            .collect(),
        v_cache: h
            .v_cache
            .iter()
            .map(|c| gpu.upload_raw(c, &[c.len()]).expect("v_cache"))
            .collect(),
        partials: partials_floats
            .iter()
            .map(|&f| gpu.zeros(&[f], DType::F32).expect("partials"))
            .collect(),
    }
}

fn bytes(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    gpu.download_raw_bytes(t).expect("download")
}

fn mismatches(x: &[u8], y: &[u8]) -> usize {
    x.iter().zip(y).filter(|(p, q)| p != q).count() + x.len().abs_diff(y.len())
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    if gpu.arch != "gfx1201" {
        eprintln!("lane_fa_oracle: needs exact gfx1201, got {}", gpu.arch);
        std::process::exit(2);
    }
    if !gpu.lane_fa_batch_admitted(NH, NKV, HD, &[16]) {
        eprintln!("lane_fa_oracle: lane_fa_batch_admitted is false (kernel.verify_attn off / tile != 128?)");
        std::process::exit(2);
    }
    println!("arch={} lane_fa_batch_admitted=true", gpu.arch);

    let mut rng = Rng(0x4C41_4E45);
    let q_norm: Vec<f32> = (0..HD).map(|_| 0.5 + rng.unit()).collect();
    let k_norm: Vec<f32> = (0..HD).map(|_| 0.5 + rng.unit()).collect();
    let d_q_norm = gpu.upload_f32(&q_norm, &[HD]).expect("q_norm");
    let d_k_norm = gpu.upload_f32(&k_norm, &[HD]).expect("k_norm");

    let mut failures = 0usize;
    let mut cases = 0usize;
    let mut split_cases = 0usize;
    for &lanes in &[1usize, 3, 8] {
        for &n in &[2usize, 8, 16] {
            for shift in 0..2usize {
                cases += 1;
                let total = lanes * n;
                let starts: Vec<usize> = (0..lanes)
                    .map(|i| CTXS[(i + 2 * shift) % CTXS.len()])
                    .collect();
                let host = make_host(&mut rng, &starts, n);
                let lane_shape: Vec<(usize, usize)> = starts.iter().map(|&s| (n, s + n)).collect();
                let lane_floats: Vec<usize> = lane_shape
                    .iter()
                    .map(|&l| Gpu::lane_fa_partials_len(NH, &[l]))
                    .collect();
                // Arm B: the lane-table partials (lane layout over the lanes
                // that stay on the VerifyAttn lane kernels) and the model
                // scratch's own singleton partials (`flash_partials`).
                let b_floats = Gpu::lane_fa_partials_len(NH, &lane_shape);
                let single_floats = lane_floats.iter().copied().max().unwrap_or(0);
                let a = make_arm(&mut gpu, &host, total, &lane_floats);
                let b = make_arm(&mut gpu, &host, total, &[b_floats, single_floats]);
                // The singleton route class of each lane in arm A (the
                // lane's own partials buffer is its singleton's scratch).
                let split_a: Vec<bool> = (0..lanes)
                    .map(|i| {
                        gpu.fp8_fa2_split_verify_admitted(
                            starts[i] + n,
                            NH,
                            NKV,
                            HD,
                            n,
                            lane_floats[i],
                        )
                        .is_some()
                    })
                    .collect();
                gpu.hip.device_synchronize().expect("sync");

                // Arm A: today's per-lane launches on that lane's row views.
                for i in 0..lanes {
                    let r0 = i * n;
                    let vw = |t: &GpuTensor, w: usize| t.sub_offset(r0 * w, n * w);
                    gpu.deinterleave_q_rmsnorm_f32_batched(
                        &vw(&a.q_full, 2 * Q_DIM),
                        &vw(&a.q, Q_DIM),
                        &vw(&a.gate, Q_DIM),
                        &d_q_norm,
                        NH,
                        HD,
                        n,
                        EPS,
                    )
                    .expect("A deinterleave");
                    gpu.rmsnorm_batched(
                        &vw(&a.k, KV_DIM),
                        &d_k_norm,
                        &vw(&a.k, KV_DIM),
                        n * NKV,
                        HD,
                        EPS,
                    )
                    .expect("A k rmsnorm");
                    gpu.rope_partial_interleaved_f32_batched(
                        &vw(&a.q, Q_DIM),
                        &vw(&a.k, KV_DIM),
                        &vw(&a.pos, 1),
                        NH,
                        NKV,
                        HD,
                        N_ROT,
                        THETA,
                        n,
                        0,
                    )
                    .expect("A rope");
                    gpu.kv_cache_write_fp8_e4m3_batched(
                        &a.k_cache[i],
                        &vw(&a.k, KV_DIM),
                        &vw(&a.pos, 1),
                        NKV,
                        HD,
                        n,
                    )
                    .expect("A k write");
                    gpu.kv_cache_write_fp8_e4m3_batched(
                        &a.v_cache[i],
                        &vw(&a.v, KV_DIM),
                        &vw(&a.pos, 1),
                        NKV,
                        HD,
                        n,
                    )
                    .expect("A v write");
                    gpu.attention_flash_fp8_e4m3_tile_batched(
                        &vw(&a.q, Q_DIM),
                        &a.k_cache[i],
                        &a.v_cache[i],
                        &vw(&a.out, Q_DIM),
                        &vw(&a.pos, 1),
                        NH,
                        NKV,
                        HD,
                        starts[i] + n,
                        starts[i] + n,
                        n,
                        &a.partials[i],
                        None,
                        0,
                        0,
                        None,
                    )
                    .expect("A attend");
                }
                gpu.hip.device_synchronize().expect("sync A");

                // Arm B: the lane route.
                gpu.deinterleave_q_rmsnorm_f32_batched(
                    &b.q_full, &b.q, &b.gate, &d_q_norm, NH, HD, total, EPS,
                )
                .expect("B deinterleave");
                gpu.rmsnorm_batched(&b.k, &d_k_norm, &b.k, total * NKV, HD, EPS)
                    .expect("B k rmsnorm");
                gpu.rope_partial_interleaved_f32_batched(
                    &b.q, &b.k, &b.pos, NH, NKV, HD, N_ROT, THETA, total, 0,
                )
                .expect("B rope");
                let views = |t: &GpuTensor, w: usize| -> Vec<GpuTensor> {
                    (0..lanes).map(|i| t.sub_offset(i * n * w, n * w)).collect()
                };
                let (k_rows, v_rows, q_rows, out_rows, pos_rows) = (
                    views(&b.k, KV_DIM),
                    views(&b.v, KV_DIM),
                    views(&b.q, Q_DIM),
                    views(&b.out, Q_DIM),
                    views(&b.pos, 1),
                );
                let k_writes: Vec<LaneFaWrite<'_>> = (0..lanes)
                    .map(|i| LaneFaWrite {
                        cache: &b.k_cache[i],
                        src: &k_rows[i],
                        positions: &pos_rows[i],
                        rows: n,
                    })
                    .collect();
                gpu.kv_cache_write_fp8_e4m3_lanes(&k_writes, NKV, HD)
                    .expect("B k write");
                let v_writes: Vec<LaneFaWrite<'_>> = (0..lanes)
                    .map(|i| LaneFaWrite {
                        cache: &b.v_cache[i],
                        src: &v_rows[i],
                        positions: &pos_rows[i],
                        rows: n,
                    })
                    .collect();
                gpu.kv_cache_write_fp8_e4m3_lanes(&v_writes, NKV, HD)
                    .expect("B v write");
                let attends: Vec<LaneFaAttend<'_>> = (0..lanes)
                    .map(|i| LaneFaAttend {
                        q: &q_rows[i],
                        k_cache: &b.k_cache[i],
                        v_cache: &b.v_cache[i],
                        out: &out_rows[i],
                        positions: &pos_rows[i],
                        rows: n,
                        max_ctx_len: starts[i] + n,
                        physical_cap: starts[i] + n,
                    })
                    .collect();
                let split_b = gpu.lane_fa_split_lanes(&attends, &b.partials[1], NH, NKV, HD);
                gpu.attention_verify_fp8_lanes_routed(
                    &attends,
                    &b.partials[0],
                    &b.partials[1],
                    NH,
                    NKV,
                    HD,
                )
                .expect("B attend");
                gpu.hip.device_synchronize().expect("sync B");

                // Compare.
                let mut bad = Vec::new();
                let mut cmp = |name: String, x: Vec<u8>, y: Vec<u8>| {
                    let m = mismatches(&x, &y);
                    if m != 0 {
                        bad.push(format!("{name}:{m}"));
                    }
                };
                for (name, x, y) in [
                    ("q", &a.q, &b.q),
                    ("gate", &a.gate, &b.gate),
                    ("k", &a.k, &b.k),
                    ("v", &a.v, &b.v),
                    ("attn_out", &a.out, &b.out),
                ] {
                    cmp(name.to_string(), bytes(&gpu, x), bytes(&gpu, y));
                }
                for i in 0..lanes {
                    cmp(
                        format!("k_cache[{i}]"),
                        bytes(&gpu, &a.k_cache[i]),
                        bytes(&gpu, &b.k_cache[i]),
                    );
                    cmp(
                        format!("v_cache[{i}]"),
                        bytes(&gpu, &a.v_cache[i]),
                        bytes(&gpu, &b.v_cache[i]),
                    );
                }
                // Lanes the singleton serves with the native-fp8 split-KV
                // verifier run that singleton launch in arm B too, so their
                // partials are the split scratch (not comparable with the
                // VerifyAttn layout; the attention output rows are). Every
                // other lane's partials sit in lane order over the non-split
                // lanes only.
                let b_partials = bytes(&gpu, &b.partials[0]);
                let mut base = 0usize;
                for i in 0..lanes {
                    if split_b[i] {
                        continue;
                    }
                    let len = lane_floats[i] * 4;
                    cmp(
                        format!("partials[{i}]"),
                        bytes(&gpu, &a.partials[i]),
                        b_partials[base..base + len].to_vec(),
                    );
                    base += len;
                }
                if split_a != split_b {
                    bad.push(format!("route_class a={split_a:?} b={split_b:?}"));
                }
                let ok = bad.is_empty();
                if !ok {
                    failures += 1;
                }
                let n_split = split_b.iter().filter(|&&s| s).count();
                split_cases += usize::from(n_split > 0);
                println!(
                    "{} lanes={lanes} n={n:<2} ctx={starts:?} split_lanes={n_split} partials_stride={LANE_FA_PARTIALS_STRIDE} mismatching_buffers=[{}]",
                    if ok { "PASS" } else { "FAIL" },
                    bad.join(", "),
                );

                for t in [
                    a.q_full, a.q, a.gate, a.k, a.v, a.out, a.pos, b.q_full, b.q, b.gate, b.k, b.v,
                    b.out, b.pos,
                ] {
                    gpu.free_tensor(t).expect("free");
                }
                for t in a
                    .k_cache
                    .into_iter()
                    .chain(a.v_cache)
                    .chain(a.partials)
                    .chain(b.k_cache)
                    .chain(b.v_cache)
                    .chain(b.partials)
                {
                    gpu.free_tensor(t).expect("free");
                }
            }
        }
    }

    println!("cases={cases} failures={failures} cases_with_split_lanes={split_cases}");
    if failures != 0 {
        std::process::exit(1);
    }
}
