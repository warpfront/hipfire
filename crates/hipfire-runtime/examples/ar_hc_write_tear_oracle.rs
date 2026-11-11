// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Negative (torn / stale / sentinel) tests of the DENSE all-reduce wire
//! (`ar_hc_write_d8_gfx1201`) on two gfx1201 cards.
//!
//! ```text
//! ar_hc_write_tear_oracle [--devices 0,1] [--out DIR] [--hold-ms 40]
//! ```
//!
//! A hostile peer: rank 1's GPU writes the payload dwords of the launch's bank in rank 0's
//! mailbox one dword store at a time in a chosen order, with a host-timed gap, while rank 0
//! runs the REAL kernel. Rank 0's streams are compared with the host model.
//!  * `split_d0_first` / `split_d1_first`: one dword of every granule is visible for
//!    `--hold-ms`, then the other. The kernel must keep waiting (elapsed >= hold) and fold the
//!    complete granules (streams == model). A reader that accepted a half-written granule
//!    would fold the wrong payload or finish early.
//!  * `last_granule_half`, `d0_only`: a granule never completes. The bounded wait must expire
//!    at site `ar_hc_write_gfx1201`, the streams must be untouched.
//!  * `nan_pair`: both ranks push a partial whose bf16 payload dword is all ones (NaN pair);
//!    the sender remaps it so the receiver never mistakes it for EMPTY: no hang, the NaN
//!    columns stay NaN, every other element equals the model byte for byte.
//!  * `rearm`: after a normal paired exchange the used bank of each mailbox is all 0xFF.
//! Exit code 3 on any failure; a JSON receipt lands in `--out`.

use hip_bridge::KernargBlob;
use hipfire_runtime::ar_hc_exchange::{ArHcExchange, ArHcMode};
use rdna_compute::ar_hc_write::{ar_hc_write_model, bf16_bits, ArHcWire, StreamKind, AR_HC_HIDDEN};
use rdna_compute::Gpu;
use std::ffi::c_void;
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

const HOSTILE_SRC: &str = r#"
#include <hip/hip_runtime.h>
// sel 0: dword 0 of every granule; 1: dword 1; 2: both; 3: both except the last granule's
// dword 1. One relaxed system-scope dword store each (never a wider store).
extern "C" __global__ __launch_bounds__(256) void arwire_hostile(char *dst, const unsigned *words,
                                                                 unsigned n, unsigned sel) {
    unsigned *d = reinterpret_cast<unsigned *>(dst);
    for (unsigned j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        const bool d0 = sel == 0 || sel == 2 || sel == 3;
        const bool d1 = sel == 1 || sel == 2 || (sel == 3 && j + 1 != n);
        if (d0) __hip_atomic_store(d + 2 * j, words[2 * j], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        if (d1) __hip_atomic_store(d + 2 * j + 1, words[2 * j + 1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}
"#;

const ROWS: usize = 8;

// (device -> host copy of `len` bytes)
fn dl(gpu: &Gpu, buf: &hip_bridge::DeviceBuffer, len: usize) -> Result<Vec<u8>, String> {
    let mut v = vec![0u8; len];
    gpu.hip
        .memcpy_dtoh(&mut v, buf)
        .map_err(|e| e.to_string())?;
    Ok(v)
}

fn val(rank: usize, i: usize) -> f32 {
    ((i * 37 + rank * 11 + 5) % 201) as f32 / 64.0 - 1.5
}

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn pack(p: &[f32]) -> Vec<u32> {
    p.chunks_exact(4)
        .flat_map(|g| {
            [
                u32::from(bf16_bits(g[0])) | (u32::from(bf16_bits(g[1])) << 16),
                u32::from(bf16_bits(g[2])) | (u32::from(bf16_bits(g[3])) << 16),
            ]
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Expect {
    Folds,
    TimesOut,
}

struct Case {
    name: &'static str,
    first: u32,
    second: Option<u32>,
    expect: Expect,
}

#[derive(Default)]
struct Outcome {
    rows: Vec<(String, bool, String)>,
}

struct Shared {
    gen: u32,
}

#[allow(clippy::too_many_arguments)]
fn run_rank(
    rank: usize,
    device: i32,
    hold_ms: u64,
    x: Arc<ArHcExchange>,
    barrier: Arc<Barrier>,
    shared: Arc<Mutex<Shared>>,
) -> Result<Outcome, String> {
    let e = |e: hip_bridge::HipError| e.to_string();
    let mut gpu = Gpu::init_with_device(device).map_err(e)?;
    gpu.ensure_device_wait_block().map_err(e)?;
    x.register(rank, &mut gpu).map_err(e)?;
    barrier.wait();
    x.enable_peers(rank, &mut gpu).map_err(e)?;
    gpu.ensure_kernel_public("arwire_hostile", HOSTILE_SRC, "arwire_hostile")
        .map_err(e)?;
    barrier.wait();
    x.qualify(rank, &mut gpu, 8, 8)
        .map_err(|er| format!("qualification: {er}"))?;
    barrier.wait();
    let n = ROWS * AR_HC_HIDDEN;
    let p: [Vec<f32>; 2] = [
        (0..n).map(|i| val(0, i)).collect(),
        (0..n).map(|i| val(1, i)).collect(),
    ];
    let streams_host: Vec<u16> = (0..ROWS * 4 * AR_HC_HIDDEN)
        .map(|i| bf16_bits(((i * 13 + 3) % 97) as f32 / 32.0 - 1.5))
        .collect();
    let gates_host = vec![0f32; ROWS * 4];
    let mut want = streams_host.clone();
    ar_hc_write_model(&p[0], &p[1], &mut want, &gates_host, ROWS);

    let partial = gpu.hip.malloc(n * 4).map_err(e)?;
    let streams = gpu.hip.malloc(streams_host.len() * 2).map_err(e)?;
    let gates = gpu.hip.malloc(ROWS * 16).map_err(e)?;
    let words = gpu.hip.malloc(n / 2 * 4).map_err(e)?;
    gpu.hip.memset(&gates, 0, ROWS * 16).map_err(e)?;
    gpu.hip
        .memcpy_htod(&words, &bytes_u32(&pack(&p[1])))
        .map_err(e)?;
    let mut out = Outcome::default();

    let cases = [
        Case {
            name: "split_d0_first",
            first: 0,
            second: Some(1),
            expect: Expect::Folds,
        },
        Case {
            name: "split_d1_first",
            first: 1,
            second: Some(0),
            expect: Expect::Folds,
        },
        Case {
            name: "last_granule_half",
            first: 3,
            second: None,
            expect: Expect::TimesOut,
        },
        Case {
            name: "d0_only",
            first: 0,
            second: None,
            expect: Expect::TimesOut,
        },
    ];
    for case in &cases {
        // rank 0 publishes the bank its launch will use
        if rank == 0 {
            gpu.hip
                .memcpy_htod(&partial, &bytes_f32(&p[0]))
                .map_err(e)?;
            gpu.hip
                .memcpy_htod(&streams, &bytes_u16(&streams_host))
                .map_err(e)?;
            shared.lock().unwrap().gen = x.generation(0, &mut gpu).map_err(e)?;
        }
        barrier.wait();
        let gen = shared.lock().unwrap().gen;
        let mut tag = gen.wrapping_add(1);
        if tag == 0 {
            tag = 2;
        }
        barrier.wait();
        let mut elapsed = Duration::ZERO;
        let mut wait_err = None;
        if rank == 0 {
            let t0 = Instant::now();
            // SAFETY: live buffers of the stated extents on this device.
            unsafe {
                x.write(
                    0,
                    &mut gpu,
                    partial.as_ptr(),
                    streams.as_ptr(),
                    StreamKind::Bf16,
                    gates.as_ptr(),
                    ROWS,
                    std::ptr::null_mut::<c_void>(),
                )
            }
            .map_err(e)?;
            gpu.hip.device_synchronize().map_err(e)?;
            elapsed = t0.elapsed();
            wait_err = gpu.take_device_wait_error().err().map(|er| er.to_string());
        } else {
            // rank 1 never launches: its host arrives so rank 0's launch passes the host gate
            x.skip_launch(1);
            std::thread::sleep(Duration::from_millis(5));
            let dst = x.mailbox_addr(0)
                + x.region_offset(ArHcWire::Dense)
                + (tag & 1) as usize * x.bank_stride(ArHcWire::Dense);
            let hostile = |gpu: &Gpu, sel: u32| -> Result<(), String> {
                let mut b = KernargBlob::new();
                b.push_ptr(dst as *const c_void);
                b.push_ptr(words.as_ptr());
                b.push_u32((n / 4) as u32);
                b.push_u32(sel);
                gpu.launch_kernel_blob(
                    "arwire_hostile",
                    [16, 1, 1],
                    [256, 1, 1],
                    0,
                    b.as_mut_slice(),
                )
                .map_err(|er| er.to_string())?;
                gpu.hip.device_synchronize().map_err(|er| er.to_string())
            };
            hostile(&gpu, case.first)?;
            if let Some(second) = case.second {
                std::thread::sleep(Duration::from_millis(hold_ms));
                hostile(&gpu, second)?;
            }
        }
        barrier.wait();
        if rank == 0 {
            let got = u16s(&dl(&gpu, &streams, streams_host.len() * 2)?);
            let (ok, note) = match case.expect {
                Expect::Folds => {
                    let long_enough = elapsed >= Duration::from_millis(hold_ms.saturating_sub(10));
                    let ok = wait_err.is_none() && got == want && long_enough;
                    (
                        ok,
                        format!(
                            "elapsed {:?} (hold {hold_ms} ms), streams==model {}, wait_err {:?}",
                            elapsed,
                            got == want,
                            wait_err
                        ),
                    )
                }
                Expect::TimesOut => {
                    let site = wait_err
                        .as_deref()
                        .is_some_and(|s| s.contains("ar_hc_write_gfx1201"));
                    // `first == 3`: only the last granule is incomplete, so every block but its own
                    // completes; every other case leaves all granules incomplete. An element is
                    // either original (its block timed out) or the model value (its block folded
                    // complete granules); the incomplete granule's elements must be original.
                    let last_only = case.first == 3;
                    let mut garbage = 0usize;
                    let mut folded_incomplete = 0usize;
                    for (at, &g) in got.iter().enumerate() {
                        if g != streams_host[at] && g != want[at] {
                            garbage += 1;
                        }
                        let col = at % AR_HC_HIDDEN;
                        let row = at / (4 * AR_HC_HIDDEN);
                        let in_last = row == ROWS - 1 && col >= AR_HC_HIDDEN - 4;
                        if (!last_only || in_last) && g != streams_host[at] {
                            folded_incomplete += 1;
                        }
                    }
                    (
                        site && garbage == 0 && folded_incomplete == 0,
                        format!(
                            "elapsed {:?}, site recorded {site}, garbage elements {garbage}, incomplete granule elements folded {folded_incomplete}",
                            elapsed
                        ),
                    )
                }
            };
            out.rows.push((case.name.to_string(), ok, note));
        }
        // the generations are unequal after a skipped / aborted launch: both ranks reset
        x.reset(rank, &mut gpu).map_err(e)?;
        barrier.wait();
    }

    // nan_pair: a real paired exchange whose first granule is a NaN pair on both ranks
    {
        let rows = 1usize;
        let n1 = rows * AR_HC_HIDDEN;
        let mut mine: Vec<f32> = (0..n1).map(|i| val(rank, i)).collect();
        for v in mine.iter_mut().take(4) {
            *v = f32::from_bits(0xFFFF_7000);
        }
        assert_eq!(
            bf16_bits(mine[0]),
            0xFFFF,
            "test premise: the payload half is all ones"
        );
        let p0: Vec<f32> = (0..n1)
            .map(|i| {
                if i < 4 {
                    f32::from_bits(0xFFFF_7000)
                } else {
                    val(0, i)
                }
            })
            .collect();
        let p1: Vec<f32> = (0..n1)
            .map(|i| {
                if i < 4 {
                    f32::from_bits(0xFFFF_7000)
                } else {
                    val(1, i)
                }
            })
            .collect();
        let st: Vec<u16> = streams_host[..4 * AR_HC_HIDDEN].to_vec();
        let g1 = vec![0f32; 4];
        let mut wantn = st.clone();
        ar_hc_write_model(&p0, &p1, &mut wantn, &g1, rows);
        gpu.hip
            .memcpy_htod(&partial, &bytes_f32(&mine))
            .map_err(e)?;
        gpu.hip.memcpy_htod(&streams, &bytes_u16(&st)).map_err(e)?;
        barrier.wait();
        // SAFETY: as above.
        unsafe {
            x.write(
                rank,
                &mut gpu,
                partial.as_ptr(),
                streams.as_ptr(),
                StreamKind::Bf16,
                gates.as_ptr(),
                rows,
                std::ptr::null_mut::<c_void>(),
            )
        }
        .map_err(e)?;
        gpu.hip.device_synchronize().map_err(e)?;
        let werr = gpu.take_device_wait_error().err().map(|er| er.to_string());
        let got = u16s(&dl(&gpu, &streams, 4 * AR_HC_HIDDEN * 2)?);
        let mut bad = 0usize;
        for br in 0..4 {
            for c in 0..AR_HC_HIDDEN {
                let at = br * AR_HC_HIDDEN + c;
                if c < 4 {
                    let b = got[at];
                    if (b & 0x7F80) != 0x7F80 || (b & 0x007F) == 0 {
                        bad += 1;
                    }
                } else if got[at] != wantn[at] {
                    bad += 1;
                }
            }
        }
        // the used bank re-armed
        let genw = x.generation(rank, &mut gpu).map_err(e)?;
        let mb = x
            .download_mailbox(rank, &mut gpu, ArHcWire::Dense)
            .map_err(e)?;
        let (stride, bank) = (x.bank_stride(ArHcWire::Dense), (genw & 1) as usize);
        let rearmed = mb[bank * stride..(bank + 1) * stride]
            .iter()
            .all(|&b| b == 0xFF);
        barrier.wait();
        out.rows.push((
            format!("nan_pair+rearm rank {rank}"),
            werr.is_none() && bad == 0 && rearmed,
            format!("wait_err {werr:?}, bad elements {bad}, bank {bank} re-armed {rearmed}"),
        ));
    }
    barrier.wait();
    x.release(rank, &mut gpu).map_err(e)?;
    Ok(out)
}

fn main() {
    let mut devices = [0i32, 1i32];
    let mut out = ".".to_string();
    let mut hold_ms = 40u64;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
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
            "--hold-ms" => {
                i += 1;
                hold_ms = args[i].parse().expect("hold-ms");
            }
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    let x = Arc::new(ArHcExchange::with_mode(ROWS, ArHcMode::Dense).expect("exchange"));
    let barrier = Arc::new(Barrier::new(2));
    let shared = Arc::new(Mutex::new(Shared { gen: 0 }));
    let handles: Vec<_> = (0..2)
        .map(|rank| {
            let (x, b, s) = (x.clone(), barrier.clone(), shared.clone());
            let dev = devices[rank];
            std::thread::spawn(move || run_rank(rank, dev, hold_ms, x, b, s))
        })
        .collect();
    let results: Vec<Result<Outcome, String>> = handles
        .into_iter()
        .map(|h| h.join().expect("thread"))
        .collect();
    let mut failed = false;
    let mut json = Vec::new();
    for (rank, r) in results.iter().enumerate() {
        match r {
            Err(er) => {
                eprintln!("rank {rank}: ERROR {er}");
                failed = true;
            }
            Ok(o) => {
                for (name, ok, note) in &o.rows {
                    eprintln!(
                        "rank {rank} {name}: {} ({note})",
                        if *ok { "PASS" } else { "FAIL" }
                    );
                    failed |= !ok;
                    json.push(format!(
                        "{{\"rank\":{rank},\"case\":\"{name}\",\"pass\":{ok},\"note\":\"{}\"}}",
                        note.replace('"', "'")
                    ));
                }
            }
        }
    }
    let receipt = format!(
        "{{\"oracle\":\"ar_hc_write_tear_oracle\",\"wire\":\"dense\",\"pass\":{},\"hold_ms\":{hold_ms},\"devices\":[{},{}],\"results\":[{}]}}",
        !failed, devices[0], devices[1], json.join(",")
    );
    let _ = std::fs::create_dir_all(&out);
    let path = format!("{out}/ar_hc_write_tear_oracle.json");
    let _ = std::fs::write(&path, &receipt);
    println!("{receipt}");
    eprintln!("receipt: {path}");
    std::process::exit(if failed { 3 } else { 0 });
}
