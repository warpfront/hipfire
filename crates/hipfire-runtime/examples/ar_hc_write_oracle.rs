// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exactness oracle of the product fused TP2 all-reduce + HC write
//! (`ar_hc_write_gfx1201`, `rdna_compute::ar_hc_write`) on two gfx1201 cards.
//!
//! ```text
//! ar_hc_write_oracle [--rows 1,8,32,100,256] [--iters 2000] [--devices 0,1] [--out DIR]
//! ```
//!
//! Per iteration (one "layer" of chained BF16 streams): fresh random partials (distinct per
//! rank), fresh gate logits. The fused launch is checked against REF = the host fold
//! (`ar_hc_fold`, exact IEEE adds of the two bf16-rounded partials) uploaded as `mixed` and
//! applied by the PRODUCTION `hyper_write` kernel to an identical copy of the streams:
//!  * fused streams byte-equal to REF streams, on both ranks;
//!  * rank 0 streams byte-equal to rank 1 streams;
//!  * the kernel's folded `m` (debug output) byte-equal to the host fold, on both ranks;
//!  * `|m - (p0 + p1)| <= (2^-8 + 2^-23)(|p0| + |p1|)` against the exact sum.
//! Also a bounded-wait fault check at the end (rank 1 skips a launch; rank 0's wait must
//! record `AR_HC_WRITE_GFX1201` in the wait block and the step must fail) unless `--no-fault`.
//! Exit code 3 on any failure; a JSON receipt lands in `--out` (default `.`).

use hipfire_runtime::ar_hc_exchange::ArHcExchange;
use rdna_compute::ar_hc_write::{ar_hc_fold, ar_hc_fold_bound, bf16_bits, AR_HC_HIDDEN};
use rdna_compute::tensor_ops::{hyper_write, HyperWrite};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;
use std::sync::{Arc, Barrier, Mutex};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[derive(Default, Clone, Debug)]
struct Tally {
    iters: usize,
    fused_ne_ref: usize,
    m_ne_ref: usize,
    bound: usize,
    max_bound_ratio: f64,
    first: Option<String>,
}

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn alloc_f32(gpu: &mut Gpu, n: usize) -> GpuTensor {
    gpu.alloc_tensor(&[n], DType::F32).expect("alloc")
}

struct RankResult {
    tallies: Vec<(usize, Tally)>,
    streams: Vec<(usize, Vec<u8>)>,
    fault: Option<String>,
}

#[allow(clippy::too_many_arguments)]
fn run_rank(
    rank: usize,
    device: i32,
    rows_list: Vec<usize>,
    iters: usize,
    fault: bool,
    x: Arc<ArHcExchange>,
    barrier: Arc<Barrier>,
    shared_ref: Arc<Mutex<Vec<Vec<u8>>>>,
) -> Result<RankResult, String> {
    let mut gpu = Gpu::init_with_device(device).map_err(|e| e.to_string())?;
    gpu.ensure_device_wait_block().map_err(|e| e.to_string())?;
    x.register(rank, &mut gpu).map_err(|e| e.to_string())?;
    barrier.wait();
    x.enable_peers(rank, &mut gpu).map_err(|e| e.to_string())?;
    barrier.wait();
    x.qualify(rank, &mut gpu, 8, 16)
        .map_err(|e| format!("qualification: {e}"))?;
    barrier.wait();
    let mut tallies = Vec::new();
    let mut last_streams = Vec::new();
    for &rows in &rows_list {
        let n = rows * AR_HC_HIDDEN;
        let wide = rows * 4 * AR_HC_HIDDEN;
        // streams hold BF16 bits: allocate F32-typed tensors of `wide` elements (2x oversize,
        // the production hyper_write view of bf16 state)
        let streams = alloc_f32(&mut gpu, wide);
        let ref_streams = alloc_f32(&mut gpu, wide);
        let partial = alloc_f32(&mut gpu, n);
        let mixed_ref = alloc_f32(&mut gpu, n);
        let dbg = alloc_f32(&mut gpu, n);
        let gates = alloc_f32(&mut gpu, rows * 4);
        let normalized = alloc_f32(&mut gpu, wide);
        let mut tally = Tally::default();
        let mut rng = Rng(0x0b0b_0b0b ^ (rows as u64) * 0x9E37_79B9);
        for it in 0..iters {
            // identical stream-state stream on both ranks (same seed), distinct partials
            let reset = it % 8 == 0;
            let mut host_p = Vec::with_capacity(2);
            for _ in 0..2 {
                host_p.push(
                    (0..n)
                        .map(|_| {
                            let mag = 2f32.powi(-6 + rng.below(10) as i32);
                            (rng.unit() - 0.5) * 2.0 * mag
                        })
                        .collect::<Vec<f32>>(),
                );
            }
            let host_gates: Vec<f32> = (0..rows * 4).map(|_| (rng.unit() - 0.5) * 8.0).collect();
            let init: Option<Vec<u16>> = reset.then(|| {
                (0..wide)
                    .map(|_| bf16_bits((rng.unit() - 0.5) * 4.0))
                    .collect()
            });
            let m_host: Vec<f32> = (0..n)
                .map(|i| ar_hc_fold(host_p[0][i], host_p[1][i]))
                .collect();
            gpu.hip
                .memcpy_htod(&partial.buf, &bytes_f32(&host_p[rank]))
                .map_err(|e| e.to_string())?;
            gpu.hip
                .memcpy_htod(&gates.buf, &bytes_f32(&host_gates))
                .map_err(|e| e.to_string())?;
            gpu.hip
                .memcpy_htod(&mixed_ref.buf, &bytes_f32(&m_host))
                .map_err(|e| e.to_string())?;
            if let Some(init) = &init {
                let b = bytes_u16(init);
                gpu.hip
                    .memcpy_htod(&streams.buf, &b)
                    .map_err(|e| e.to_string())?;
                gpu.hip
                    .memcpy_htod(&ref_streams.buf, &b)
                    .map_err(|e| e.to_string())?;
            }
            // fused
            // SAFETY: live buffers of the stated extents on this rank's device.
            unsafe {
                x.write(
                    rank,
                    &mut gpu,
                    partial.buf.as_ptr(),
                    streams.buf.as_ptr(),
                    gates.buf.as_ptr(),
                    rows,
                    dbg.buf.as_ptr(),
                )
            }
            .map_err(|e| e.to_string())?;
            // REF: production hyper_write on the host fold
            hyper_write(
                &mut gpu,
                &HyperWrite {
                    input: &ref_streams,
                    normalized: &normalized,
                    mixed: &mixed_ref,
                    gates: &gates,
                    output: &ref_streams,
                    branches: 4,
                    hidden: AR_HC_HIDDEN,
                    state_bf16: true,
                },
            )
            .map_err(|e| e.to_string())?;
            gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
            gpu.take_device_wait_error()
                .map_err(|e| format!("iter {it}: {e}"))?;
            let got = gpu
                .download_raw_bytes(&streams)
                .map_err(|e| e.to_string())?[..wide * 2]
                .to_vec();
            let want = gpu
                .download_raw_bytes(&ref_streams)
                .map_err(|e| e.to_string())?[..wide * 2]
                .to_vec();
            let got_m = f32s(&gpu.download_raw_bytes(&dbg).map_err(|e| e.to_string())?);
            tally.iters += 1;
            let note = |t: &mut Tally, what: String| {
                if t.first.is_none() {
                    t.first = Some(format!("rows {rows} iter {it} rank {rank}: {what}"));
                }
            };
            if got != want {
                tally.fused_ne_ref += 1;
                let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
                note(
                    &mut tally,
                    format!(
                        "fused != REF streams at byte {at}: {:#04x} vs {:#04x}",
                        got[at], want[at]
                    ),
                );
            }
            if got_m
                .iter()
                .zip(&m_host)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                tally.m_ne_ref += 1;
                note(&mut tally, "folded m differs from the host fold".into());
            }
            for i in 0..n {
                let (a, b) = (host_p[0][i], host_p[1][i]);
                let allowed = ar_hc_fold_bound(a, b);
                let err = (f64::from(got_m[i]) - (f64::from(a) + f64::from(b))).abs();
                if allowed > 0.0 {
                    tally.max_bound_ratio = tally.max_bound_ratio.max(err / allowed);
                }
                if err > allowed {
                    tally.bound += 1;
                    note(
                        &mut tally,
                        format!("bound at {i}: m {} p0 {a} p1 {b}", got_m[i]),
                    );
                    break;
                }
            }
            last_streams = got;
        }
        // rank equality: ranks publish their final streams for this row count
        {
            let mut g = shared_ref.lock().unwrap();
            if g.len() < 2 {
                g.resize(2, Vec::new());
            }
            g[rank] = last_streams.clone();
        }
        barrier.wait();
        if rank == 0 {
            let g = shared_ref.lock().unwrap();
            if g[0] != g[1] {
                tally.fused_ne_ref += 1;
                tally.first.get_or_insert_with(|| {
                    format!("rows {rows}: rank 0 and rank 1 final streams differ")
                });
            }
        }
        barrier.wait();
        tallies.push((rows, tally));
        for t in [
            streams,
            ref_streams,
            partial,
            mixed_ref,
            dbg,
            gates,
            normalized,
        ] {
            gpu.free_tensor(t).map_err(|e| e.to_string())?;
        }
    }
    // bounded-wait fault: rank 1 skips one launch, rank 0's wait must time out and say so
    let mut fault_note = None;
    if fault {
        let rows = 1usize;
        let partial = alloc_f32(&mut gpu, AR_HC_HIDDEN);
        let streams = alloc_f32(&mut gpu, 4 * AR_HC_HIDDEN);
        let gates = alloc_f32(&mut gpu, 4);
        if rank == 0 {
            // SAFETY: as above.
            unsafe {
                x.write(
                    0,
                    &mut gpu,
                    partial.buf.as_ptr(),
                    streams.buf.as_ptr(),
                    gates.buf.as_ptr(),
                    rows,
                    std::ptr::null_mut::<c_void>(),
                )
            }
            .map_err(|e| e.to_string())?;
            gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
            fault_note = Some(match gpu.take_device_wait_error() {
                Ok(()) => "FAIL: the unmatched launch did not time out".to_string(),
                Err(e) => {
                    let s = e.to_string();
                    if s.contains("ar_hc_write_gfx1201") {
                        format!("ok: {s}")
                    } else {
                        format!("FAIL: wrong site: {s}")
                    }
                }
            });
        }
        barrier.wait();
        // both ranks must reset after a wait error before reuse
        x.reset(rank, &mut gpu).map_err(|e| e.to_string())?;
        for t in [partial, streams, gates] {
            gpu.free_tensor(t).map_err(|e| e.to_string())?;
        }
    }
    barrier.wait();
    x.release(rank, &mut gpu).map_err(|e| e.to_string())?;
    Ok(RankResult {
        tallies,
        streams: Vec::new(),
        fault: fault_note,
    })
}

fn main() {
    let mut rows_list = vec![1usize, 2, 3, 8, 16, 32, 100, 256];
    let mut iters = 2000usize;
    let mut devices = [0i32, 1i32];
    let mut out = ".".to_string();
    let mut fault = true;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--rows" => {
                i += 1;
                rows_list = args[i]
                    .split(',')
                    .map(|s| s.parse().expect("rows"))
                    .collect();
            }
            "--iters" => {
                i += 1;
                iters = args[i].parse().expect("iters");
            }
            "--devices" => {
                i += 1;
                let d: Vec<i32> = args[i]
                    .split(',')
                    .map(|s| s.parse().expect("device"))
                    .collect();
                devices = [d[0], d[1]];
            }
            "--out" => {
                i += 1;
                out = args[i].clone();
            }
            "--no-fault" => fault = false,
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    let max_rows = *rows_list.iter().max().unwrap();
    let x = Arc::new(
        ArHcExchange::new(max_rows.min(rdna_compute::ar_hc_write::AR_HC_ROW_CAP))
            .expect("exchange"),
    );
    let barrier = Arc::new(Barrier::new(2));
    let shared = Arc::new(Mutex::new(Vec::new()));
    let handles: Vec<_> = (0..2)
        .map(|rank| {
            let (x, b, s, rl) = (
                x.clone(),
                barrier.clone(),
                shared.clone(),
                rows_list.clone(),
            );
            let dev = devices[rank];
            std::thread::spawn(move || run_rank(rank, dev, rl, iters, fault, x, b, s))
        })
        .collect();
    let results: Vec<Result<RankResult, String>> = handles
        .into_iter()
        .map(|h| h.join().expect("thread"))
        .collect();
    let mut failed = false;
    let mut json_rows = Vec::new();
    for (rank, r) in results.iter().enumerate() {
        match r {
            Err(e) => {
                eprintln!("rank {rank}: ERROR {e}");
                failed = true;
            }
            Ok(res) => {
                for (rows, t) in &res.tallies {
                    let bad = t.fused_ne_ref + t.m_ne_ref + t.bound;
                    eprintln!(
                        "rank {rank} rows {rows}: {} iters fused!=REF {} m!=host {} bound {} (max ratio {:.3}){}",
                        t.iters, t.fused_ne_ref, t.m_ne_ref, t.bound, t.max_bound_ratio,
                        t.first.as_deref().map_or(String::new(), |f| format!(" FIRST: {f}"))
                    );
                    failed |= bad != 0;
                    json_rows.push(format!(
                        "{{\"rank\":{rank},\"rows\":{rows},\"iters\":{},\"fused_ne_ref\":{},\"m_ne_host\":{},\"bound_violations\":{},\"max_bound_ratio\":{:.4},\"pass\":{}}}",
                        t.iters, t.fused_ne_ref, t.m_ne_ref, t.bound, t.max_bound_ratio, bad == 0
                    ));
                }
                if let Some(f) = &res.fault {
                    eprintln!("rank {rank} fault check: {f}");
                    failed |= f.starts_with("FAIL");
                }
            }
        }
    }
    let receipt = format!(
        "{{\"oracle\":\"ar_hc_write_oracle\",\"pass\":{},\"iters\":{iters},\"devices\":[{},{}],\"results\":[{}]}}",
        !failed, devices[0], devices[1], json_rows.join(",")
    );
    let _ = std::fs::create_dir_all(&out);
    let path = format!("{out}/ar_hc_write_oracle.json");
    let _ = std::fs::write(&path, &receipt);
    println!("{receipt}");
    eprintln!("receipt: {path}");
    std::process::exit(if failed { 3 } else { 0 });
}
