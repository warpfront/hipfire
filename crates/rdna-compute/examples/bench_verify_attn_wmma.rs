// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Ben1078
// hipfire — see LICENSE and NOTICE in the project root.
//
// Oracle + bench of the exact-gfx1100 WMMA split-K verify attention
// (`Gpu::attention_verify_wmma_q8_gfx1100`, `attention_wmma_verify.rs`)
// against a CPU f64 reference and the incumbent eager verify routes it
// replaces: the Q8 FA2 split-KV verifier (`kernel.gfx1100_fa2_split_verify`,
// above 4096 context) and the VerifyAttn rows twin, both through
// `attention_flash_q8_0_rows_masked_logical` (4..=8 rows). Qwen3.8-27B verify
// shape: 24 q heads / 4 kv heads / head_dim 256, plain Q8_0 KV written by the
// production `kv_cache_write_q8_0_batched`.
//
//   cargo run --release -p rdna-compute --example bench_verify_attn_wmma -- \
//       --seq 40000 --rows 4 [--check] [--hot] [--iters 20] [--reps 7]
//
// --check: every route's output vs the f64 reference (softmax over the
// dequantized K/V, all 24 heads); fails (exit 1) on a non-finite value or a
// WMMA-route relative L2 error >= 2e-3. Timing is cold by default: the KV
// cache is replicated into enough sets (> 200 MB in total) that the timed
// calls stream their KV from DRAM like a decode step does; --hot uses one
// set. GB/s counts the Q8 K+V bytes of `seq` positions.
use rdna_compute::attention_wmma_verify::wmma_verify_attn_partials_len;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn lcg(seed: u32, n: usize, amp: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5) * amp
        })
        .collect()
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let v = if e == 0 {
        (m as f32) * (2.0f32).powi(-24)
    } else if e == 31 {
        if m == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        f32::from_bits(((e + 112) as u32) << 23 | (m << 13))
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

struct Stats {
    rel_max: f64,
    rel_l2: f64,
    nonfinite: usize,
}

fn compare(out: &[f32], reference: &[f64]) -> Stats {
    let (mut max_abs, mut max_ref, mut sd, mut sr, mut nonfinite) = (0.0f64, 0.0f64, 0.0, 0.0, 0);
    for (o, r) in out.iter().zip(reference) {
        let o = *o as f64;
        if !o.is_finite() {
            nonfinite += 1;
        }
        let e = (o - r).abs();
        max_abs = max_abs.max(e);
        max_ref = max_ref.max(r.abs());
        sd += e * e;
        sr += r * r;
    }
    Stats {
        rel_max: max_abs / max_ref.max(1e-30),
        rel_l2: (sd / sr.max(1e-300)).sqrt(),
        nonfinite,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let argval = |k: &str, d: f64| -> f64 {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].parse().unwrap())
            .unwrap_or(d)
    };
    let seq_len = argval("--seq", 40000.0) as usize;
    let rows = argval("--rows", 4.0) as usize;
    let iters = argval("--iters", 20.0) as usize;
    let reps = argval("--reps", 7.0) as usize;
    let qamp = argval("--qamp", 24.0) as f32;
    let check = args.iter().any(|a| a == "--check");
    let hot = args.iter().any(|a| a == "--hot");
    assert!((1..=8).contains(&rows) && seq_len >= rows);
    let (n_heads, n_kv_heads, head_dim) = (24usize, 4usize, 256usize);
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let per_pos = n_kv_heads * (head_dim / 32) * 34;
    let chunk = 8192usize;
    let max_seq = seq_len.div_ceil(chunk) * chunk;
    let set_bytes = 2 * max_seq * per_pos;

    // One partials buffer large enough for every route: the WMMA spans, the
    // FA2 split verifier, and the rows tile (`rows x heads x tiles x (2+hd)`).
    let split_part =
        rdna_compute::attention::gfx1100_q8_fa2_split_partials_len(rows, n_heads, head_dim)
            .unwrap_or(0);
    let rows_part = rows * n_heads * max_seq.div_ceil(64) * (2 + head_dim);
    let part_floats = wmma_verify_attn_partials_len(rows, n_heads, seq_len)
        .max(split_part)
        .max(rows_part);
    let sets = if hot {
        1
    } else {
        (200e6 / set_bytes as f64).ceil() as usize
    }
    .clamp(1, 8);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!(
        "GPU {}  seq={seq_len} rows={rows} sets={sets} ({:.0} MB KV{})",
        gpu.arch,
        (sets * set_bytes) as f64 / 1e6,
        if hot { ", hot" } else { ", cold rotation" }
    );

    // ---- KV set 0 via the production Q8 writer, replicated into the others ----
    let mut ks: Vec<GpuTensor> = Vec::new();
    let mut vs: Vec<GpuTensor> = Vec::new();
    for _ in 0..sets {
        ks.push(gpu.alloc_tensor(&[max_seq * per_pos], DType::Q8_0).unwrap());
        vs.push(gpu.alloc_tensor(&[max_seq * per_pos], DType::Q8_0).unwrap());
    }
    let d_all_pos = gpu.alloc_tensor(&[max_seq], DType::F32).unwrap();
    let all_pos: Vec<u8> = (0..max_seq as i32).flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&d_all_pos.buf, &all_pos).unwrap();
    let mut written = 0usize;
    let mut seed = 0xc3c3u32;
    while written < max_seq {
        let d_kf = gpu
            .upload_f32(&lcg(seed, chunk * kv_dim, 1.0), &[chunk * kv_dim])
            .unwrap();
        let d_vf = gpu
            .upload_f32(&lcg(seed ^ 0x9696, chunk * kv_dim, 1.0), &[chunk * kv_dim])
            .unwrap();
        seed = seed.wrapping_add(7919);
        let pos_view = d_all_pos.sub_offset(written, chunk);
        gpu.kv_cache_write_q8_0_batched(&ks[0], &d_kf, &pos_view, n_kv_heads, head_dim, chunk)
            .unwrap();
        gpu.kv_cache_write_q8_0_batched(&vs[0], &d_vf, &pos_view, n_kv_heads, head_dim, chunk)
            .unwrap();
        gpu.hip.device_synchronize().unwrap();
        gpu.free_tensor(d_kf).unwrap();
        gpu.free_tensor(d_vf).unwrap();
        written += chunk;
    }
    let mut hk = vec![0u8; max_seq * per_pos];
    let mut hv = vec![0u8; max_seq * per_pos];
    gpu.hip.memcpy_dtoh(&mut hk, &ks[0].buf).unwrap();
    gpu.hip.memcpy_dtoh(&mut hv, &vs[0].buf).unwrap();
    for s in 1..sets {
        gpu.hip.memcpy_htod(&ks[s].buf, &hk).unwrap();
        gpu.hip.memcpy_htod(&vs[s].buf, &hv).unwrap();
    }

    let hq = lcg(0xa5a5, rows * q_dim, qamp);
    let d_q = gpu.upload_f32(&hq, &[rows * q_dim]).unwrap();
    let positions: Vec<i32> = (0..rows).map(|i| (seq_len - rows + i) as i32).collect();
    let d_pos = gpu.alloc_tensor(&[rows], DType::F32).unwrap();
    let pos_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&d_pos.buf, &pos_bytes).unwrap();
    let d_part = gpu.zeros(&[part_floats], DType::F32).unwrap();
    let d_out = gpu.zeros(&[rows * q_dim], DType::F32).unwrap();
    let kv_bytes = (seq_len * per_pos * 2) as f64;

    // ---- f64 reference over the dequantized cache (code * f16 scale) ----
    let mut reference = vec![0.0f64; rows * q_dim];
    if check {
        let t0 = Instant::now();
        let dequant = |raw: &[u8]| -> Vec<f32> {
            let mut out = vec![0.0f32; seq_len * kv_dim];
            for t in 0..seq_len {
                for kvh in 0..n_kv_heads {
                    for b in 0..head_dim / 32 {
                        let off = t * per_pos + kvh * 272 + b * 34;
                        let sc = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
                        let o = (t * n_kv_heads + kvh) * head_dim + b * 32;
                        for i in 0..32 {
                            out[o + i] = (raw[off + 2 + i] as i8) as f32 * sc;
                        }
                    }
                }
            }
            out
        };
        let fk = dequant(&hk);
        let fv = dequant(&hv);
        let jobs: Vec<(usize, usize)> = (0..rows)
            .flat_map(|r| (0..n_heads).map(move |h| (r, h)))
            .collect();
        let nth = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .min(16);
        let results: Vec<Vec<(usize, Vec<f64>)>> = std::thread::scope(|sc| {
            let handles: Vec<_> = (0..nth)
                .map(|ti| {
                    let (jobs, fk, fv, hq, positions) = (&jobs, &fk, &fv, &hq, &positions);
                    sc.spawn(move || {
                        let mut outv = Vec::new();
                        for (j, &(r, h)) in jobs.iter().enumerate() {
                            if j % nth != ti {
                                continue;
                            }
                            let kvh = h / (n_heads / n_kv_heads);
                            let n = positions[r] as usize + 1;
                            let q: Vec<f64> = (0..head_dim)
                                .map(|d| hq[r * q_dim + h * head_dim + d] as f64)
                                .collect();
                            let mut s = vec![0.0f64; n];
                            let mut mx = f64::NEG_INFINITY;
                            for t in 0..n {
                                let kr = &fk[(t * n_kv_heads + kvh) * head_dim..][..head_dim];
                                let acc: f64 = (0..head_dim).map(|d| q[d] * kr[d] as f64).sum();
                                s[t] = acc / (head_dim as f64).sqrt();
                                mx = mx.max(s[t]);
                            }
                            let mut den = 0.0f64;
                            let mut o = vec![0.0f64; head_dim];
                            for t in 0..n {
                                let p = (s[t] - mx).exp();
                                den += p;
                                let vr = &fv[(t * n_kv_heads + kvh) * head_dim..][..head_dim];
                                for d in 0..head_dim {
                                    o[d] += p * vr[d] as f64;
                                }
                            }
                            o.iter_mut().for_each(|v| *v /= den);
                            outv.push((r * q_dim + h * head_dim, o));
                        }
                        outv
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for part in results {
            for (base, o) in part {
                reference[base..base + head_dim].copy_from_slice(&o);
            }
        }
        eprintln!(
            "f64 reference: {} (row, head) pairs in {:.1} s",
            jobs.len(),
            t0.elapsed().as_secs_f64()
        );
    }

    let mut cur = 0usize;
    let time =
        |gpu: &mut Gpu, cur: &mut usize, f: &mut dyn FnMut(&mut Gpu, usize) -> bool| -> f64 {
            let t = Instant::now();
            while t.elapsed().as_secs_f64() < 1.0 {
                for _ in 0..20 {
                    *cur = (*cur + 1) % sets;
                    f(gpu, *cur);
                }
                gpu.hip.device_synchronize().unwrap();
            }
            let mut v = Vec::new();
            for _ in 0..reps {
                gpu.hip.device_synchronize().unwrap();
                let t = Instant::now();
                for _ in 0..iters {
                    *cur = (*cur + 1) % sets;
                    f(gpu, *cur);
                }
                gpu.hip.device_synchronize().unwrap();
                v.push(t.elapsed().as_secs_f64() * 1e6 / iters as f64);
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };

    // (name, route): 0 = WMMA, 1 = incumbent with the FA2 split verifier
    // admitted, 2 = incumbent rows twin (split not admitted).
    let routes: [(&str, u8); 3] = [("fa2-split", 1), ("rows-twin", 2), ("wmma", 0)];
    let mut ok = true;
    let mut base_us: Option<(f64, &str)> = None;
    for (name, route) in routes {
        let mut f = |gpu: &mut Gpu, s: usize| -> bool {
            if route == 0 {
                gpu.attention_verify_wmma_q8_gfx1100(
                    &d_q, &ks[s], &vs[s], &d_out, &d_pos, n_heads, n_kv_heads, head_dim, seq_len,
                    rows, &d_part,
                )
                .unwrap()
            } else {
                gpu.attention_flash_q8_0_rows_masked_logical(
                    &d_q,
                    &ks[s],
                    &vs[s],
                    &d_out,
                    &d_pos,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_seq,
                    seq_len,
                    rows,
                    &d_part,
                    route == 1,
                )
                .unwrap()
            }
        };
        if route == 1 && (seq_len <= 4096 || rows < 4) {
            eprintln!("{name:<10} n/a (split verifier needs ctx > 4096 and >= 4 rows)");
            continue;
        }
        gpu.hip.memset(&d_out.buf, 0xff, rows * q_dim * 4).unwrap();
        if !f(&mut gpu, 0) {
            eprintln!("{name:<10} declined this shape");
            if route == 0 {
                ok = false;
            }
            continue;
        }
        gpu.hip.device_synchronize().unwrap();
        let mut line = String::new();
        if check {
            let st = compare(&gpu.download_f32(&d_out).unwrap(), &reference);
            line = format!(
                "  rel_max {:.2e} rel_l2 {:.2e} nonfinite {}",
                st.rel_max, st.rel_l2, st.nonfinite
            );
            ok &= st.nonfinite == 0 && (route != 0 || st.rel_l2 < 2e-3);
        }
        let us = time(&mut gpu, &mut cur, &mut f);
        let speed = base_us.map_or(String::new(), |(b, bn)| format!("  {:.2}x vs {bn}", b / us));
        eprintln!(
            "{name:<10} {us:8.1} us  {:5.0} GB/s{speed}{line}",
            kv_bytes / (us * 1e3)
        );
        if base_us.is_none() {
            base_us = Some((us, name));
        }
    }
    if !ok {
        eprintln!("FAIL");
        std::process::exit(1);
    }
    eprintln!("OK");
}
