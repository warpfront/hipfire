// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LAB ONLY: call-shape invariance oracle of the gfx11 Q8/Q8 ordinary-prefill attention
//! (CHUNK-CANONICAL-DESIGN §1.3, G3). Every row of one fixed prompt must produce the same bytes
//! whatever the prefill was split into: the launcher `attention_q8_0_flash_prefill_wmma` (plus the
//! gfx1151 VerifyAttn twin that the dispatch arm tries first for 1..=32 rows) is run over the same
//! H24/KV4/D256 Q8 cache with the rows partitioned into windows of `n` rows at every start phase,
//! and each row's f32 output is compared byte-for-byte with the aligned 512-row whole-call
//! reference (the bulk route of an aligned chunked prefill).
//!
//! Run: attn_call_shape_oracle_gfx11 [--sizes 1,2,3,...] [--quiet]
//! Exit status 0 only if every compared row is byte-identical.

use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{DType, Gpu, GpuTensor};

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
/// Rows of the compared range (positions BASE .. BASE+RANGE).
const BASE: usize = 512;
const RANGE: usize = 1024;
const CTX: usize = BASE + RANGE;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn pos_tensor(gpu: &mut Gpu, start: usize, n: usize) -> GpuTensor {
    let v: Vec<f32> = (0..n).map(|i| f32::from_bits((start + i) as u32)).collect();
    gpu.upload_f32(&v, &[n]).expect("positions upload")
}

#[allow(clippy::too_many_arguments)]
fn run_window(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    out: &GpuTensor,
    scratch: &GpuTensor,
    row0: usize,
    n: usize,
) {
    let qd = NH * HD;
    let qv = q.sub_offset((row0 - BASE) * qd, n * qd);
    let ov = out.sub_offset((row0 - BASE) * qd, n * qd);
    let pos = pos_tensor(gpu, row0, n);
    let ctx = row0 + n;
    let route = std::env::var("ORC_ROUTE").unwrap_or_else(|_| "dispatch".into());
    match route.as_str() {
        // Direct single launches: the gfx11 FA2 body / the query-tiled WMMA body.
        "fa2" => gpu
            .attention_q8_0_fa2_gqa_gfx11(&qv, k, v, &ov, &pos, NH, NKV, HD, ctx, n)
            .expect("fa2"),
        "slots" => gpu
            .attention_q8_0_flash_prefill_wmma_slots(
                &qv, k, v, &ov, &pos, NH, NKV, HD, ctx, n, None, None, None, None,
            )
            .expect("wmma slots"),
        _ => {
            // Dispatch bypasses the VerifyAttn twin under the ordinary-prefill intent.
            let _ = scratch;
            gpu.attention_q8_0_flash_prefill_wmma(&qv, k, v, &ov, &pos, NH, NKV, HD, ctx, n)
                .expect("flash prefill wmma");
        }
    }
    gpu.free_tensor(pos).ok();
}

/// `--probe P`: bytes of the row at absolute position P as computed in windows whose tile max
/// (hence `seq_len`) differs: [P-k, P+m) for several (k, m).
fn probe(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    scratch: &GpuTensor,
    p: usize,
) {
    let qd = NH * HD;
    let out = gpu.zeros(&[RANGE * qd], DType::F32).expect("probe out");
    let mut classes: Vec<(Vec<u32>, Vec<String>)> = Vec::new();
    for back in [0usize, 1, 2, 3, 5, 8, 15, 16, 17, 31, 40] {
        for fwd in [1usize, 2, 3, 4, 7, 8, 9, 15, 16, 17, 24, 32] {
            let r0 = p - back;
            if r0 < BASE {
                continue;
            }
            run_window(gpu, q, k, v, &out, scratch, r0, back + fwd);
            let got = gpu.download_f32(&out).expect("dl");
            let r = p - BASE;
            let bits: Vec<u32> = got[r * qd..(r + 1) * qd]
                .iter()
                .map(|x| x.to_bits())
                .collect();
            let tag = format!("[{r0},{}) tilemax_pos={}", p + fwd, p + fwd - 1);
            match classes.iter_mut().find(|(b, _)| *b == bits) {
                Some((_, t)) => t.push(tag),
                None => classes.push((bits, vec![tag])),
            }
        }
    }
    println!("probe p={p}: {} distinct byte classes", classes.len());
    for (i, (_, t)) in classes.iter().enumerate() {
        println!(
            "  class {i}: {} windows, e.g. {:?}",
            t.len(),
            &t[..t.len().min(8)]
        );
    }
}

fn main() {
    let mut sizes: Vec<usize> = (1..=130).collect();
    sizes.extend([192, 255, 256, 257, 300, 511, 512, 513, 521, 777, 1000, 1024]);
    let mut quiet = false;
    let mut detail: Option<(usize, usize)> = None;
    let mut probe_at: Option<usize> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--sizes" => {
                sizes = args
                    .next()
                    .expect("--sizes list")
                    .split(',')
                    .map(|s| s.parse().expect("size"))
                    .collect()
            }
            "--quiet" => quiet = true,
            "--probe" => probe_at = Some(args.next().expect("--probe P").parse().expect("P")),
            "--detail" => {
                let v = args.next().expect("--detail n,phase");
                let (a, b) = v.split_once(',').expect("n,phase");
                detail = Some((a.parse().expect("n"), b.parse().expect("phase")));
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let mut gpu = Gpu::init().expect("gpu init");
    let prev = gpu.ordinary_prefill_enter();

    // Q8 K/V cache of CTX positions: random scales and codes.
    let bph = HD / 32;
    let bytes_per_pos = NKV * bph * 34;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut mk = |salt: u8| {
        let mut buf = vec![0u8; CTX * bytes_per_pos];
        for blk in buf.chunks_mut(34) {
            let scale = 0.01 + 0.03 * (rng.unit() * 0.5 + 0.5);
            let h = half_from_f32(scale);
            blk[0] = (h & 0xFF) as u8;
            blk[1] = (h >> 8) as u8;
            for b in blk[2..].iter_mut() {
                *b = ((rng.next() >> 33) as u8).wrapping_add(salt) as i8 as u8;
            }
        }
        buf
    };
    let mut kb = mk(0);
    let mut vb = mk(3);
    // ORC_ZERO_FROM=P zeroes the cache rows at positions >= P (scale 0, codes 0): shows whether
    // data beyond a tile's last key reaches its outputs.
    if let Some(z) = std::env::var("ORC_ZERO_FROM")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        for b in [&mut kb, &mut vb] {
            for x in b[z * bytes_per_pos..].iter_mut() {
                *x = 0;
            }
        }
    }
    let k = gpu.upload_raw(&kb, &[kb.len()]).expect("k");
    let v = gpu.upload_raw(&vb, &[vb.len()]).expect("v");
    let qd = NH * HD;
    let q_data: Vec<f32> = (0..RANGE * qd).map(|_| rng.unit() * 1.5).collect();
    let q = gpu.upload_f32(&q_data, &[RANGE * qd]).expect("q");
    let scratch = gpu.zeros(&[16 << 20], DType::F32).expect("scratch");

    // Reference: aligned 512-row whole calls.
    let out_ref = gpu.zeros(&[RANGE * qd], DType::F32).expect("ref");
    for s in (BASE..BASE + RANGE).step_by(512) {
        run_window(&mut gpu, &q, &k, &v, &out_ref, &scratch, s, 512);
    }
    let reference = gpu.download_f32(&out_ref).expect("dl ref");
    assert!(reference.iter().all(|x| x.is_finite()) && reference.iter().any(|x| *x != 0.0));
    if let Some(p) = probe_at {
        probe(&mut gpu, &q, &k, &v, &scratch, p);
        gpu.ordinary_prefill_exit(prev);
        return;
    }

    let mut failures = 0usize;
    let mut compared = 0usize;
    let out = gpu.zeros(&[RANGE * qd], DType::F32).expect("out");
    println!(
        "arch={} rows={} base={} sizes={}",
        gpu.arch,
        RANGE,
        BASE,
        sizes.len()
    );
    for &n in &sizes {
        let mut bad_phases: Vec<(usize, usize, usize)> = Vec::new();
        for phase in 0..16usize.min(n.max(1)) {
            // Windows of n rows from BASE+phase; the first `phase` rows are covered by one
            // leading window (phase rows), the last window is the remainder.
            let mut windows = Vec::new();
            let mut row = BASE;
            if phase > 0 {
                windows.push((row, phase));
                row += phase;
            }
            while row < BASE + RANGE {
                let len = n.min(BASE + RANGE - row);
                windows.push((row, len));
                row += len;
            }
            for &(r0, len) in &windows {
                run_window(&mut gpu, &q, &k, &v, &out, &scratch, r0, len);
            }
            let got = gpu.download_f32(&out).expect("dl");
            if detail == Some((n, phase)) {
                for &(r0, len) in &windows {
                    let offs: Vec<usize> = (0..len)
                        .filter(|o| {
                            let r = r0 - BASE + o;
                            reference[r * qd..(r + 1) * qd]
                                .iter()
                                .zip(&got[r * qd..(r + 1) * qd])
                                .any(|(x, y)| x.to_bits() != y.to_bits())
                        })
                        .collect();
                    println!(
                        "  window start {r0} (mod16 {}) len {len}: differing offsets {offs:?}",
                        r0 % 16
                    );
                }
            }
            let mut nbad = 0usize;
            let mut first = usize::MAX;
            for r in 0..RANGE {
                compared += 1;
                let (a, b) = (&reference[r * qd..(r + 1) * qd], &got[r * qd..(r + 1) * qd]);
                if a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits()) {
                    nbad += 1;
                    first = first.min(r);
                }
            }
            if nbad > 0 {
                failures += nbad;
                bad_phases.push((phase, nbad, first));
            }
        }
        if !quiet || !bad_phases.is_empty() {
            println!(
                "n={n:5} {} {}",
                if bad_phases.is_empty() {
                    "OK  "
                } else {
                    "DIFF"
                },
                bad_phases
                    .iter()
                    .take(4)
                    .map(|(p, c, f)| format!("[phase {p}: {c} rows differ, first row +{f}]"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    gpu.ordinary_prefill_exit(prev);
    println!("compared_rows={compared} differing_rows={failures}");
    std::process::exit(if failures == 0 { 0 } else { 1 });
}
