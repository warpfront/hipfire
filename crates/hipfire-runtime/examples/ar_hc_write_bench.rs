// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Timing bench of the product fused TP2 all-reduce + HC write (`ar_hc_write_gfx1201`) on two
//! gfx1201 cards: splits fixed per-launch latency from bytes/s.
//!
//! ```text
//! ar_hc_write_bench [--rows 1,2,4,8,16,32,64,128,256] [--iters 300] [--iso 100]
//!                   [--devices 0,1] [--stream bf16|f32|both] [--out DIR] [--no-probe]
//! ```
//!
//! Per (stream kind, rows) case, on both ranks concurrently:
//!  * `pipe`: `iters` back-to-back launches between two events (the product shape: launches
//!    are queued, the peer skew is absorbed), mean us per launch;
//!  * `iso`: `iso` launches each started after a host barrier and drained, median us (includes
//!    the host skew of the two rank threads as peer wait);
//!  * `probe` (unless `--no-probe`): the same launch geometry pushing the SAME wire bytes
//!    (`rows * 2560 / 4` granules x 16 B) as pure system-scope b128 stores into the peer's
//!    fine-grained buffer, no tag, no poll, no HC math, and the same with HALF the granules
//!    (the wire of a 100% dense scheme), both ranks pushing at once. The probe is the raw
//!    posted-store ceiling the current kernel is compared against.
//! The per-rank fit `t(rows) = a + b * rows` (least squares over the rows list) is printed
//! for every series. Exit code 3 on any device wait error.
//!
//! `--breakdown`: instead of the series above, the phase breakdown of `arhc_lab/breakdown.rs`
//! (launch floor / push / rendezvous poll / fold+apply / ticket tail, spin-gated pipelines,
//! realtime stamps) for both wires per (stream kind, rows) case.

#[path = "arhc_lab/breakdown.rs"]
mod breakdown;
#[path = "arhc_lab/pushlab.rs"]
mod pushlab;

use hip_bridge::KernargBlob;
use hipfire_runtime::ar_hc_exchange::ArHcExchange;
use hipfire_runtime::ar_hc_exchange::ArHcMode;
use rdna_compute::ar_hc_write::{ar_hc_geometry, StreamKind, AR_HC_HIDDEN, AR_HC_ROW_CAP};
use rdna_compute::Gpu;
use std::ffi::c_void;
use std::sync::{Arc, Barrier, Mutex};

const PROBE_SRC: &str = r#"
#include <hip/hip_runtime.h>
typedef unsigned int arw_u4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ void arw_st128_sys(arw_u4 *p, arw_u4 v) {
    asm volatile("global_store_b128 %0, %1, off scope:SCOPE_SYS" ::"v"(p), "v"(v) : "memory");
}
// `n16` granules of 16 B, lane-strided like the AR kernel's slices (first granule of lane
// `t` of block `b` is b * per + t; per = ceil(n16 / grid)), stores only.
extern "C" __global__ __launch_bounds__(512) void arwire_probe_st16(char *dst, unsigned n16, unsigned tag) {
    const unsigned per = (n16 + gridDim.x - 1) / gridDim.x;
    const unsigned g0 = blockIdx.x * per;
    const unsigned g1 = min(n16, g0 + per);
    arw_u4 *p = reinterpret_cast<arw_u4 *>(dst);
    for (unsigned j = g0 + threadIdx.x; j < g1; j += blockDim.x) {
        arw_st128_sys(p + j, arw_u4{tag, tag, tag, tag});
    }
}
// Same slicing with 8-byte lane stores (the dense scheme: 4 bf16 values per lane, no tag).
extern "C" __global__ __launch_bounds__(512) void arwire_probe_st8(char *dst, unsigned n8, unsigned tag) {
    const unsigned per = (n8 + gridDim.x - 1) / gridDim.x;
    const unsigned g0 = blockIdx.x * per;
    const unsigned g1 = min(n8, g0 + per);
    unsigned long long *p = reinterpret_cast<unsigned long long *>(dst);
    const unsigned long long v = ((unsigned long long)tag << 32) | tag;
    for (unsigned j = g0 + threadIdx.x; j < g1; j += blockDim.x) {
        asm volatile("global_store_b64 %0, %1, off scope:SCOPE_SYS" ::"v"(p + j), "v"(v) : "memory");
    }
}
"#;

fn fit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let (sx, sy) = (xs.iter().sum::<f64>(), ys.iter().sum::<f64>());
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    ((sy - b * sx) / n, b)
}

#[derive(Clone, Debug, Default)]
struct Row {
    kind: &'static str,
    rows: usize,
    pipe_us: f64,
    iso_med_us: f64,
    iso_min_us: f64,
    probe_us: f64,
    probe_half_us: f64,
    probe8_us: f64,
}

struct Rank {
    rows: Vec<Row>,
    bd: Vec<breakdown::BdRow>,
    pl: Vec<(String, f64)>,
    err: Option<String>,
}

fn event_ms(gpu: &Gpu, a: &hip_bridge::Event, b: &hip_bridge::Event) -> Result<f64, String> {
    gpu.hip
        .event_elapsed_ms(a, b)
        .map(f64::from)
        .map_err(|e| e.to_string())
}

#[allow(clippy::too_many_arguments)]
fn run_rank(
    rank: usize,
    device: i32,
    rows_list: Vec<usize>,
    kinds: Vec<StreamKind>,
    iters: usize,
    iso: usize,
    probe: bool,
    x: Arc<ArHcExchange>,
    barrier: Arc<Barrier>,
    probe_ptrs: Arc<Mutex<[usize; 2]>>,
    bd: Option<(bool, Arc<Mutex<[usize; 2]>>)>,
) -> Result<Rank, String> {
    let max_rows = *rows_list.iter().max().unwrap();
    let mut gpu = Gpu::init_with_device(device).map_err(|e| e.to_string())?;
    gpu.ensure_device_wait_block().map_err(|e| e.to_string())?;
    x.register(rank, &mut gpu).map_err(|e| e.to_string())?;
    // probe sink: this rank's fine-grained buffer the PEER writes (allocated before peers are
    // enabled, like the mailbox)
    let probe_bytes = max_rows * AR_HC_HIDDEN / 4 * 16;
    let sink = gpu
        .hip
        .malloc_peer_finegrained(probe_bytes)
        .map_err(|e| e.to_string())?;
    gpu.hip
        .memset(&sink, 0, probe_bytes)
        .map_err(|e| e.to_string())?;
    probe_ptrs.lock().unwrap()[rank] = sink.as_ptr() as usize;
    let lab = match &bd {
        Some((false, p)) => {
            let l = breakdown::alloc(&mut gpu, max_rows)?;
            p.lock().unwrap()[rank] = l.mbox.as_ptr() as usize;
            Some(l)
        }
        _ => None,
    };
    let plab = match &bd {
        Some((true, p)) => {
            let l = pushlab::alloc(&mut gpu)?;
            p.lock().unwrap()[rank] = l.sink.as_ptr() as usize;
            Some(l)
        }
        _ => None,
    };
    barrier.wait();
    x.enable_peers(rank, &mut gpu).map_err(|e| e.to_string())?;
    for sym in ["arwire_probe_st16", "arwire_probe_st8"] {
        gpu.ensure_kernel_public("arwire_probe", PROBE_SRC, sym)
            .map_err(|e| e.to_string())?;
    }
    barrier.wait();
    x.qualify(rank, &mut gpu, 8, 16)
        .map_err(|e| format!("qualification: {e}"))?;
    barrier.wait();
    let peer_sink = probe_ptrs.lock().unwrap()[1 - rank] as *mut c_void;
    if let (Some(l), Some((_, p))) = (&lab, &bd) {
        let peer_lab = p.lock().unwrap()[1 - rank];
        let res = breakdown::run(
            rank, &mut gpu, l, peer_lab, &x, &barrier, &rows_list, &kinds, iters, iso,
        );
        barrier.wait();
        x.release(rank, &mut gpu).map_err(|e| e.to_string())?;
        return Ok(Rank {
            rows: Vec::new(),
            bd: res?,
            pl: Vec::new(),
            err: None,
        });
    }
    if let (Some(_), Some((_, p))) = (&plab, &bd) {
        let peer = p.lock().unwrap()[1 - rank];
        let res = pushlab::run(&mut gpu, peer, &barrier, iters);
        barrier.wait();
        x.release(rank, &mut gpu).map_err(|e| e.to_string())?;
        return Ok(Rank {
            rows: Vec::new(),
            bd: Vec::new(),
            pl: res?,
            err: None,
        });
    }
    let start = gpu.hip.event_create().map_err(|e| e.to_string())?;
    let stop = gpu.hip.event_create().map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    let mut tag = 1u32;
    for &kind in &kinds {
        for &rows in &rows_list {
            let n = rows * AR_HC_HIDDEN;
            let wide = rows * 4 * AR_HC_HIDDEN;
            let partial = gpu.hip.malloc(n * 4).map_err(|e| e.to_string())?;
            let streams = gpu
                .hip
                .malloc(wide * kind.element_bytes())
                .map_err(|e| e.to_string())?;
            let gates = gpu.hip.malloc(rows * 16).map_err(|e| e.to_string())?;
            // small deterministic data; gates 0 keep every row finite whatever libm does
            let hp: Vec<u8> = (0..n)
                .flat_map(|i| (((i % 97) as f32 - 48.0) / 64.0).to_le_bytes())
                .collect();
            gpu.hip
                .memcpy_htod(&partial, &hp)
                .map_err(|e| e.to_string())?;
            gpu.hip
                .memset(&streams, 0, wide * kind.element_bytes())
                .map_err(|e| e.to_string())?;
            gpu.hip
                .memset(&gates, 0, rows * 16)
                .map_err(|e| e.to_string())?;
            let launch = |gpu: &mut Gpu| -> Result<(), String> {
                // SAFETY: live buffers of the extents `kind` / `rows` imply on this device.
                unsafe {
                    x.write(
                        rank,
                        gpu,
                        partial.as_ptr(),
                        streams.as_ptr(),
                        kind,
                        gates.as_ptr(),
                        rows,
                        std::ptr::null_mut::<c_void>(),
                    )
                }
                .map_err(|e| e.to_string())
            };
            let drain = |gpu: &mut Gpu| -> Result<(), String> {
                gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
                gpu.take_device_wait_error().map_err(|e| e.to_string())
            };
            for _ in 0..30 {
                launch(&mut gpu)?;
            }
            drain(&mut gpu)?;
            // pipelined
            barrier.wait();
            gpu.hip
                .event_record(&start, None)
                .map_err(|e| e.to_string())?;
            for _ in 0..iters {
                launch(&mut gpu)?;
            }
            gpu.hip
                .event_record(&stop, None)
                .map_err(|e| e.to_string())?;
            drain(&mut gpu)?;
            let pipe_us = event_ms(&gpu, &start, &stop)? * 1000.0 / iters as f64;
            // isolated
            let mut isos = Vec::with_capacity(iso);
            for _ in 0..iso {
                barrier.wait();
                gpu.hip
                    .event_record(&start, None)
                    .map_err(|e| e.to_string())?;
                launch(&mut gpu)?;
                gpu.hip
                    .event_record(&stop, None)
                    .map_err(|e| e.to_string())?;
                drain(&mut gpu)?;
                isos.push(event_ms(&gpu, &start, &stop)? * 1000.0);
            }
            isos.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let iso_med_us = isos.get(isos.len() / 2).copied().unwrap_or(0.0);
            let iso_min_us = isos.first().copied().unwrap_or(0.0);
            // probes (both ranks push at once, same geometry as the AR launch)
            let mut probe_us = 0.0;
            let mut probe_half_us = 0.0;
            let mut probe8_us = 0.0;
            if probe {
                let geom = ar_hc_geometry(rows).ok_or("no geometry")?;
                for (mode, slot) in [
                    (0, &mut probe_us),
                    (1, &mut probe_half_us),
                    (2, &mut probe8_us),
                ] {
                    let granules = (n / 4) as u32;
                    // mode 0: full wire (n/4 x 16 B); 1: half the granules x 16 B; 2: n/4 x 8 B
                    let granules = if mode == 1 {
                        granules.div_ceil(2)
                    } else {
                        granules
                    };
                    let sym = if mode == 2 {
                        "arwire_probe_st8"
                    } else {
                        "arwire_probe_st16"
                    };
                    let run = |gpu: &Gpu, tag: u32| -> Result<(), String> {
                        let mut b = KernargBlob::new();
                        b.push_ptr(peer_sink);
                        b.push_u32(granules);
                        b.push_u32(tag);
                        gpu.launch_kernel_blob(
                            sym,
                            [geom.0, 1, 1],
                            [geom.1, 1, 1],
                            0,
                            b.as_mut_slice(),
                        )
                        .map_err(|e| e.to_string())
                    };
                    for _ in 0..30 {
                        tag += 1;
                        run(&gpu, tag)?;
                    }
                    drain(&mut gpu)?;
                    barrier.wait();
                    gpu.hip
                        .event_record(&start, None)
                        .map_err(|e| e.to_string())?;
                    for _ in 0..iters {
                        tag += 1;
                        run(&gpu, tag)?;
                    }
                    gpu.hip
                        .event_record(&stop, None)
                        .map_err(|e| e.to_string())?;
                    drain(&mut gpu)?;
                    *slot = event_ms(&gpu, &start, &stop)? * 1000.0 / iters as f64;
                }
            }
            barrier.wait();
            for b in [partial, streams, gates] {
                gpu.hip.free(b).map_err(|e| e.to_string())?;
            }
            out.push(Row {
                kind: kind.name(),
                rows,
                pipe_us,
                iso_med_us,
                iso_min_us,
                probe_us,
                probe_half_us,
                probe8_us,
            });
        }
    }
    barrier.wait();
    x.release(rank, &mut gpu).map_err(|e| e.to_string())?;
    gpu.hip.free(sink).map_err(|e| e.to_string())?;
    Ok(Rank {
        rows: out,
        bd: Vec::new(),
        pl: Vec::new(),
        err: None,
    })
}

fn main() {
    let mut rows_list = vec![1usize, 2, 4, 8, 16, 32, 64, 128, 256];
    let mut iters = 300usize;
    let mut iso = 100usize;
    let mut devices = [0i32, 1i32];
    let mut out = ".".to_string();
    let mut kinds = StreamKind::ALL.to_vec();
    let mut probe = true;
    let mut breakdown_mode = false;
    let mut push_lab = false;
    let mut wire = hipfire_runtime::ar_hc_exchange::ar_hc_mode_from_env();
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
            "--iso" => {
                i += 1;
                iso = args[i].parse().expect("iso");
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
            "--stream" => {
                i += 1;
                kinds = match args[i].as_str() {
                    "bf16" => vec![StreamKind::Bf16],
                    "f32" => vec![StreamKind::F32],
                    "both" => StreamKind::ALL.to_vec(),
                    other => panic!("--stream takes f32|bf16|both, got {other}"),
                };
            }
            "--wire" => {
                i += 1;
                wire = match args[i].as_str() {
                    "tagged" => ArHcMode::Tagged,
                    "dense" => ArHcMode::Dense,
                    "hybrid" => ArHcMode::Hybrid,
                    other => panic!("--wire takes tagged|dense|hybrid, got {other}"),
                };
            }
            "--no-probe" => probe = false,
            "--breakdown" => breakdown_mode = true,
            "--push-lab" => push_lab = true,
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    let max_rows = *rows_list.iter().max().unwrap();
    assert!(max_rows <= AR_HC_ROW_CAP, "rows above {AR_HC_ROW_CAP}");
    let x = Arc::new(ArHcExchange::with_mode(max_rows, wire).expect("exchange"));
    let barrier = Arc::new(Barrier::new(2));
    let ptrs = Arc::new(Mutex::new([0usize; 2]));
    let bd_ptrs =
        (breakdown_mode || push_lab).then(|| (push_lab, Arc::new(Mutex::new([0usize; 2]))));
    let handles: Vec<_> = (0..2)
        .map(|rank| {
            let (x, b, p, rl, kinds, bdp) = (
                x.clone(),
                barrier.clone(),
                ptrs.clone(),
                rows_list.clone(),
                kinds.clone(),
                bd_ptrs.clone(),
            );
            let dev = devices[rank];
            std::thread::spawn(move || {
                run_rank(rank, dev, rl, kinds, iters, iso, probe, x, b, p, bdp)
            })
        })
        .collect();
    let results: Vec<Result<Rank, String>> = handles
        .into_iter()
        .map(|h| h.join().expect("thread"))
        .collect();
    let mut failed = false;
    let mut json = Vec::new();
    for (rank, r) in results.iter().enumerate() {
        match r {
            Err(e) => {
                eprintln!("rank {rank}: ERROR {e}");
                failed = true;
            }
            Ok(res) => {
                let _ = &res.err;
                for (label, us) in &res.pl {
                    // ovl/push rows push 163840 granules x 8 B (4x for "4x bytes") per direction
                    let bytes = if label.contains("4x bytes") { 4.0 } else { 1.0 } * 163840.0 * 8.0;
                    let gbs = if label.contains("apply") {
                        0.0
                    } else {
                        bytes / us / 1e3
                    };
                    eprintln!(
                        "rank {rank} PUSHLAB {label:36} {us:8.2} us  {gbs:6.1} GB/s per direction"
                    );
                    json.push(format!(
                        "{{\"rank\":{rank},\"push_lab\":\"{label}\",\"us\":{us:.3},\"gbs\":{gbs:.2}}}"
                    ));
                }
                for r in &res.bd {
                    eprintln!(
                        "rank {rank} BREAKDOWN {} {} rows={} geometry={}x{}",
                        r.kind,
                        if r.dense { "dense" } else { "tagged" },
                        r.rows,
                        r.geometry.0,
                        r.geometry.1
                    );
                    eprintln!(
                        "  {:24} {:>8} {:>8} {:>8}",
                        "series", "pipe_us", "iso_med", "iso_min"
                    );
                    let mut js = Vec::new();
                    for (label, p, im, imin) in &r.series {
                        eprintln!("  {label:24} {p:8.2} {im:8.2} {imin:8.2}");
                        js.push(format!(
                            "{{\"series\":\"{label}\",\"pipe_us\":{p:.3},\"iso_med_us\":{im:.3},\"iso_min_us\":{imin:.3}}}"
                        ));
                    }
                    let st = r.stamps.unwrap_or_default();
                    eprintln!(
                        "  stamps (m31, us since block entry, mean): push_issued {:.2} poll_done {:.2} apply_issued {:.2} ticket_done {:.2}; body span med {:.2}; gap med {:.2}; stamp/event span {:.3}",
                        st[0], st[1], st[2], st[3], st[4], st[5], st[6]
                    );
                    json.push(format!(
                        "{{\"rank\":{rank},\"breakdown\":true,\"stream\":\"{}\",\"wire\":\"{}\",\"rows\":{},\"grid\":{},\"lanes\":{},\"series\":[{}],\"stamps_us\":[{}]}}",
                        r.kind,
                        if r.dense { "dense" } else { "tagged" },
                        r.rows,
                        r.geometry.0,
                        r.geometry.1,
                        js.join(","),
                        st.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(",")
                    ));
                }
                for kind in &kinds {
                    let rows: Vec<&Row> =
                        res.rows.iter().filter(|r| r.kind == kind.name()).collect();
                    eprintln!("rank {rank} {} mode={} (payload {} B/row; wire 10240 tagged / 5120 dense; probes push 10240 / 5120 / 5120 B/row)", kind.name(), wire.name(), AR_HC_HIDDEN * 2);
                    eprintln!("  rows  pipe_us  iso_med  iso_min  probe_us  probe_half  probe_b64  pipe_wire_GB/s  probe_wire_GB/s");
                    for r in &rows {
                        let nominal = (r.rows * AR_HC_HIDDEN * 4) as f64;
                        let actual =
                            (r.rows * AR_HC_HIDDEN / 4 * x.wire_for(r.rows).granule_bytes()) as f64;
                        eprintln!(
                            "  {:4}  {:7.1}  {:7.1}  {:7.1}  {:8.1}  {:10.1}  {:9.1}  {:14.1}  {:15.1}",
                            r.rows, r.pipe_us, r.iso_med_us, r.iso_min_us, r.probe_us, r.probe_half_us, r.probe8_us,
                            actual / r.pipe_us / 1e3,
                            if r.probe_us > 0.0 { nominal / r.probe_us / 1e3 } else { 0.0 },
                        );
                        json.push(format!(
                            "{{\"rank\":{rank},\"stream\":\"{}\",\"rows\":{},\"pipe_us\":{:.2},\"iso_med_us\":{:.2},\"iso_min_us\":{:.2},\"probe_us\":{:.2},\"probe_half_us\":{:.2},\"probe8_us\":{:.2}}}",
                            r.kind, r.rows, r.pipe_us, r.iso_med_us, r.iso_min_us, r.probe_us, r.probe_half_us, r.probe8_us
                        ));
                    }
                    let xs: Vec<f64> = rows.iter().map(|r| r.rows as f64).collect();
                    for (name, ys) in [
                        ("pipe", rows.iter().map(|r| r.pipe_us).collect::<Vec<_>>()),
                        ("iso_med", rows.iter().map(|r| r.iso_med_us).collect()),
                        ("probe", rows.iter().map(|r| r.probe_us).collect()),
                        ("probe_half", rows.iter().map(|r| r.probe_half_us).collect()),
                        ("probe_b64", rows.iter().map(|r| r.probe8_us).collect()),
                    ] {
                        if xs.len() >= 2 && ys.iter().any(|y| *y > 0.0) {
                            let (a, b) = fit(&xs, &ys);
                            eprintln!(
                                "  fit {name}: t = {a:.1} us + {b:.3} us/row  ({:.1} GB/s per 10240 B/row nominal)",
                                (AR_HC_HIDDEN * 4) as f64 / b / 1e3
                            );
                        }
                    }
                }
            }
        }
    }
    let receipt = format!(
        "{{\"bench\":\"ar_hc_write_bench\",\"wire\":\"{}\",\"pass\":{},\"iters\":{iters},\"iso\":{iso},\"devices\":[{},{}],\"results\":[{}]}}",
        wire.name(), !failed, devices[0], devices[1], json.join(",")
    );
    let _ = std::fs::create_dir_all(&out);
    let path = format!("{out}/ar_hc_write_bench.json");
    let _ = std::fs::write(&path, &receipt);
    println!("{receipt}");
    eprintln!("receipt: {path}");
    std::process::exit(if failed { 3 } else { 0 });
}
