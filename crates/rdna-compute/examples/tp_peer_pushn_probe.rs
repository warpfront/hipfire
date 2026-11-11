// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab microbench of the gfx1201 TP4 bf16 peer-push collective
//! (`tp_peer_pushn_reduce_bf16_gfx1201`, kernels/src/tp_peer_pushn.gfx1201.hip)
//! on four exact gfx1201 cards: a graph-captured dependent chain of 128
//! [producer -> collective] pairs per rank, as the decode token runs them.
//!
//! The producer reads the previous residual (`p = h(salt, rank, k, i) - x *
//! 2^-9`, explicit round-to-nearest ops), so every collective depends on the
//! previous one and a stale mailbox read perturbs everything after it. The host
//! replays the exact arithmetic of the shipped route, `x + (r(p0) + r(p1) +
//! r(p2) + r(p3))` with `r` the bf16 round trip, and compares every rank bit
//! for bit.
//!
//! Arms (comma-separated `PROBE_ARMS`): `base` (producers only: the boundary
//! floor), `prodn` (the shipped kernel from `kernels::TP_PEER_PUSHN_GFX1201_SRC`),
//! `file` (the kernel source at `PROBE_KERNEL_FILE`, same entry point and
//! signature, for iteration without rebuilding).
//!
//! Env: PROBE_CORRECT_REPLAYS (2000), PROBE_PIPELINE (queued replays per
//! verified batch, 1), PROBE_TIMING_N (200), PROBE_TIMING_TRIALS (15),
//! PROBE_DELAYS (1: random per-replay rank delay), PROBE_SEED,
//! PROBE_TIMEOUT_TEST (1: missing-peer injection).

use hip_bridge::{DeviceBuffer, Event, Function, Graph, GraphExec, KernargBlob, Module};
use rdna_compute::device_wait::{timeout_ticks, DeviceWaitBlock};
use rdna_compute::{Gpu, KernelCompiler};
use std::time::Instant;

const N: usize = 4;
const COUNT: usize = 5_120;
const BYTES: usize = COUNT * 4;
const BARRIERS: usize = 128;
const CAPTURE_MODE_RELAXED: u32 = 2;
const GRANULE: usize = 16;
const SLOT_STRIDE: usize = (COUNT / 4 * GRANULE).next_multiple_of(256);
const BANK_STRIDE: usize = N * SLOT_STRIDE;
const MBOX_BYTES: usize = 2 * BANK_STRIDE;

const SOURCE: &str = concat!(
    include_str!("../../../kernels/src/device_wait_deadline.h"),
    r#"
__device__ __forceinline__ unsigned probe_mix(unsigned h) {
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    return h;
}
__device__ __forceinline__ float probe_hashf(unsigned salt, unsigned rank, unsigned k, unsigned i) {
    unsigned h = probe_mix(salt * 0x9E3779B1u ^ probe_mix(rank * 0x632BE5ABu + k * 0x1B873593u + i * 0x27D4EB2Fu + 12345u));
    return __fadd_rn(__fmul_rn(__uint2float_rn(h >> 8), 0x1p-23f), -1.0f);
}
extern "C" __global__ void probe_producer(float *__restrict__ part, const float *__restrict__ x,
                                          const unsigned *__restrict__ salt, int rank, int k, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    part[i] = __fsub_rn(probe_hashf(salt[0], (unsigned)rank, (unsigned)k, (unsigned)i),
                        __fmul_rn(x[i], 0x1p-9f));
}
extern "C" __global__ void probe_spin(unsigned ticks) {
    if (threadIdx.x != 0) return;
    unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < (unsigned long long)ticks) { __builtin_amdgcn_s_sleep(1); }
}
"#
);

fn mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

fn hashf(salt: u32, rank: u32, k: u32, i: u32) -> f32 {
    let inner = mix(rank
        .wrapping_mul(0x632B_E5AB)
        .wrapping_add(k.wrapping_mul(0x1B87_3593))
        .wrapping_add(i.wrapping_mul(0x27D4_EB2F))
        .wrapping_add(12345));
    let h = mix(salt.wrapping_mul(0x9E37_79B1) ^ inner);
    ((h >> 8) as f32) * f32::from_bits(0x3400_0000) - 1.0
}

fn x0(i: usize) -> f32 {
    (mix(i as u32 ^ 0xABCD_1234) >> 8) as f32 * f32::from_bits(0x3400_0000) * 8.0 - 8.0
}

/// bf16 round trip of kernels/src/tp_exchange16.h (RNE, NaN canonicalised).
fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    let half = if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        ((bits >> 16) & 0x8000) | 0x7fc0
    } else {
        bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16
    };
    f32::from_bits(half << 16)
}

fn emulate_replay(x: &mut [f32], salt: u32) {
    let c = f32::from_bits(0x3b00_0000);
    let threads = 8;
    let chunk = x.len().div_ceil(threads);
    std::thread::scope(|s| {
        for (ci, part) in x.chunks_mut(chunk).enumerate() {
            s.spawn(move || {
                for (j, xi) in part.iter_mut().enumerate() {
                    let i = (ci * chunk + j) as u32;
                    let mut v = *xi;
                    for k in 0..BARRIERS as u32 {
                        let mut sum = bf16_round(hashf(salt, 0, k, i) - v * c);
                        for r in 1..N as u32 {
                            sum += bf16_round(hashf(salt, r, k, i) - v * c);
                        }
                        v += sum;
                    }
                    *xi = v;
                }
            });
        }
    });
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Base,
    Prodn,
    File,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Base => "base",
            Arm::Prodn => "prodn",
            Arm::File => "file",
        }
    }
}

struct Rank {
    gpu: Gpu,
    _modules: Vec<Module>,
    f_prod: Function,
    f_spin: Function,
    f_n: Function,
    f_file: Option<Function>,
    x: DeviceBuffer,
    part: DeviceBuffer,
    salt: DeviceBuffer,
    gen: DeviceBuffer,
    mbox: DeviceBuffer,
    wb_host: *mut DeviceWaitBlock,
    wb_dev: *mut std::ffi::c_void,
    capturing: std::cell::Cell<bool>,
    blobs: std::cell::RefCell<Vec<Vec<u8>>>,
}

struct Captured {
    graph: Graph,
    exec: GraphExec,
    _blobs: Vec<Vec<u8>>,
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 4) }
}

impl Rank {
    fn bind(&self) {
        self.gpu.bind_thread().expect("bind");
    }
    fn stream(&self) -> &hip_bridge::Stream {
        self.gpu.active_stream.as_ref().expect("stream")
    }
    fn sync(&self) {
        self.bind();
        self.gpu
            .hip
            .stream_synchronize(self.stream())
            .expect("sync");
    }
    fn reset_state(&self, x_init: &[f32]) {
        self.bind();
        let hip = &self.gpu.hip;
        hip.device_synchronize().expect("sync");
        hip.memcpy_htod(&self.x, bytes_of(x_init)).expect("x0");
        hip.memset(&self.gen, 0, 256).expect("gen");
        hip.memset(&self.salt, 0, 256).expect("salt");
        hip.memset(&self.mbox, 0, MBOX_BYTES).expect("mbox");
        unsafe { std::ptr::write_volatile(&mut (*self.wb_host).code, 0) };
        hip.device_synchronize().expect("sync");
    }
    fn download_x(&self) -> Vec<f32> {
        self.bind();
        let mut raw = vec![0u8; BYTES];
        self.gpu.hip.memcpy_dtoh(&mut raw, &self.x).expect("dtoh");
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn wait_code(&self) -> u32 {
        unsafe { std::ptr::read_volatile(&(*self.wb_host).code) }
    }
}

fn launch(rank: &Rank, f: &Function, grid: u32, block: u32, mut blob: KernargBlob) {
    rank.bind();
    blob.pad_to(16);
    let mut v = blob.into_vec();
    let r = if rank.capturing.get() {
        let mut blobs = rank.blobs.borrow_mut();
        blobs.push(v);
        let buf = blobs.last_mut().unwrap();
        // SAFETY: blob bytes match the kernel signature and outlive the graph.
        unsafe {
            rank.gpu.hip.launch_kernel_blob(
                f,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                Some(rank.stream()),
                buf.as_mut_slice(),
            )
        }
    } else {
        // SAFETY: one-shot launch, bytes live across the call.
        unsafe {
            rank.gpu.hip.launch_kernel_blob(
                f,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                Some(rank.stream()),
                v.as_mut_slice(),
            )
        }
    };
    r.expect("launch");
}

fn enqueue_producer(r: &Rank, rk: usize, k: usize) {
    let mut b = KernargBlob::new();
    b.push_ptr(r.part.as_ptr());
    b.push_ptr(r.x.as_ptr());
    b.push_ptr(r.salt.as_ptr());
    b.push_i32(rk as i32);
    b.push_i32(k as i32);
    b.push_i32(COUNT as i32);
    launch(r, &r.f_prod, (COUNT as u32).div_ceil(256), 256, b);
}

fn enqueue_collective(ranks: &[Rank; N], rk: usize, arm: Arm) {
    let r = &ranks[rk];
    let mut b = KernargBlob::new();
    b.push_ptr(r.x.as_ptr());
    b.push_ptr(r.part.as_ptr());
    for q in 0..4 {
        b.push_ptr(ranks[q].mbox.as_ptr());
    }
    b.push_u32(SLOT_STRIDE as u32);
    b.push_u32(BANK_STRIDE as u32);
    b.push_ptr(r.gen.as_ptr());
    b.push_ptr(r.wb_dev);
    b.push_i32(rk as i32);
    b.push_i32(N as i32);
    b.push_i32(COUNT as i32);
    let f = match arm {
        Arm::File => r.f_file.as_ref().expect("PROBE_KERNEL_FILE"),
        _ => &r.f_n,
    };
    launch(r, f, 1, 512, b);
}

fn capture(ranks: &mut [Rank; N], arm: Arm) -> Vec<Captured> {
    for r in ranks.iter_mut() {
        r.bind();
        r.blobs.borrow_mut().clear();
        r.capturing.set(true);
        let s = r.gpu.active_stream.as_ref().expect("stream");
        r.gpu
            .hip
            .stream_begin_capture(s, CAPTURE_MODE_RELAXED)
            .expect("begin capture");
    }
    for k in 0..BARRIERS {
        for rk in 0..N {
            enqueue_producer(&ranks[rk], rk, k);
        }
        if arm != Arm::Base {
            for rk in 0..N {
                enqueue_collective(ranks, rk, arm);
            }
        }
    }
    let mut out = Vec::new();
    for r in ranks.iter_mut() {
        r.bind();
        let s = r.gpu.active_stream.as_ref().expect("stream");
        let graph = r.gpu.hip.stream_end_capture(s).expect("end capture");
        r.capturing.set(false);
        let exec = r.gpu.hip.graph_instantiate(&graph).expect("instantiate");
        out.push(Captured {
            graph,
            exec,
            _blobs: r.blobs.take(),
        });
    }
    out
}

fn launch_graph(ranks: &[Rank; N], caps: &[Captured], rk: usize) {
    let r = &ranks[rk];
    r.bind();
    r.gpu
        .hip
        .graph_launch(&caps[rk].exec, r.stream())
        .expect("graph launch");
}

fn destroy(ranks: &[Rank; N], caps: Vec<Captured>) {
    for (r, c) in ranks.iter().zip(caps) {
        r.bind();
        r.gpu.hip.device_synchronize().expect("sync");
        r.gpu.hip.graph_exec_destroy(c.exec).expect("exec");
        r.gpu.hip.graph_destroy(c.graph).expect("graph");
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn correctness(
    ranks: &mut [Rank; N],
    arm: Arm,
    replays: usize,
    delays: bool,
    seed: u64,
    pipeline: usize,
) -> bool {
    let x_init: Vec<f32> = (0..COUNT).map(x0).collect();
    for r in ranks.iter() {
        r.reset_state(&x_init);
    }
    let caps = capture(ranks, arm);
    let mut host = x_init.clone();
    let mut rng = Rng(seed | 1);
    let mut bad = 0usize;
    let t0 = Instant::now();
    let batch = pipeline.max(1);
    let mut replay = 0usize;
    while replay < replays && bad <= 5 {
        let nb = batch.min(replays - replay);
        let salts: Vec<[u8; 4]> = (0..nb)
            .map(|j| (replay as u32 + j as u32 + 1).to_le_bytes())
            .collect();
        for salt in &salts {
            for r in ranks.iter() {
                r.bind();
                r.gpu
                    .hip
                    .memcpy_htod_async(&r.salt, salt, r.stream())
                    .expect("salt");
            }
            if delays {
                let victim = (rng.next() % N as u64) as usize;
                let us = (rng.next() % 60) as u32;
                let mut b = KernargBlob::new();
                b.push_u32(us * 100);
                ranks[victim].bind();
                launch(&ranks[victim], &ranks[victim].f_spin, 1, 64, b);
            }
            let mut order: Vec<usize> = (0..N).collect();
            if delays {
                for i in (1..N).rev() {
                    order.swap(i, (rng.next() % (i as u64 + 1)) as usize);
                }
            }
            for &rk in &order {
                launch_graph(ranks, &caps, rk);
            }
        }
        for r in ranks.iter() {
            r.sync();
        }
        for salt in &salts {
            emulate_replay(&mut host, u32::from_le_bytes(*salt));
        }
        for (rk, r) in ranks.iter().enumerate() {
            let xr = r.download_x();
            let mismatches = xr
                .iter()
                .zip(&host)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            if mismatches != 0 {
                bad += 1;
                if bad <= 5 {
                    let i = xr
                        .iter()
                        .zip(&host)
                        .position(|(a, b)| a.to_bits() != b.to_bits())
                        .unwrap();
                    println!(
                        "FAIL {} replays {replay}..{} rank {rk}: {mismatches}/{COUNT} words differ, first i={i} got {:e} want {:e}",
                        arm.name(),
                        replay + nb,
                        xr[i],
                        host[i]
                    );
                }
            }
        }
        replay += nb;
    }
    let codes: Vec<u32> = ranks.iter().map(|r| r.wait_code()).collect();
    let ok = bad == 0 && codes.iter().all(|&c| c == 0);
    println!(
        "{} {}: {replays} replays (pipeline {batch}) x {BARRIERS} collectives x {N} ranks ({} generations), {bad} bad, wait codes {codes:?}, {:.1}s",
        if ok { "PASS" } else { "FAIL" },
        arm.name(),
        replays * BARRIERS,
        t0.elapsed().as_secs_f64()
    );
    destroy(ranks, caps);
    ok
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn timing(ranks: &mut [Rank; N], arms: &[Arm], n: usize, trials: usize) {
    let x_init: Vec<f32> = (0..COUNT).map(x0).collect();
    for r in ranks.iter() {
        r.reset_state(&x_init);
    }
    let caps: Vec<Vec<Captured>> = arms.iter().map(|&a| capture(ranks, a)).collect();
    let events: Vec<Vec<Event>> = ranks
        .iter()
        .map(|r| {
            r.bind();
            (0..2 * n)
                .map(|_| r.gpu.hip.event_create().expect("event"))
                .collect()
        })
        .collect();
    let mut gpu_med: Vec<Vec<f64>> = vec![vec![]; arms.len()];
    for trial in 0..trials + 1 {
        for ai in 0..arms.len() {
            for rep in 0..n {
                for rk in 0..N {
                    let r = &ranks[rk];
                    r.bind();
                    r.gpu
                        .hip
                        .event_record(&events[rk][2 * rep], Some(r.stream()))
                        .expect("ev");
                    launch_graph(ranks, &caps[ai], rk);
                    r.gpu
                        .hip
                        .event_record(&events[rk][2 * rep + 1], Some(r.stream()))
                        .expect("ev");
                }
            }
            for r in ranks.iter() {
                r.sync();
            }
            if trial == 0 {
                continue;
            }
            let mut per: Vec<f64> = (0..n)
                .map(|rep| {
                    (0..N)
                        .map(|rk| {
                            let r = &ranks[rk];
                            r.bind();
                            r.gpu
                                .hip
                                .event_elapsed_ms(&events[rk][2 * rep], &events[rk][2 * rep + 1])
                                .expect("elapsed") as f64
                                * 1e3
                        })
                        .fold(0f64, f64::max)
                })
                .collect();
            gpu_med[ai].push(median(&mut per));
        }
    }
    let codes: Vec<u32> = ranks.iter().map(|r| r.wait_code()).collect();
    println!("timing: n={n} replays/trial, {trials} trials/arm, wait codes {codes:?}");
    let mut base_med = 0.0;
    for (ai, a) in arms.iter().enumerate() {
        let mut g = gpu_med[ai].clone();
        let med = median(&mut g);
        if *a == Arm::Base {
            base_med = med;
        }
        println!(
            "RAW arm={} gpu_us_per_replay_median_per_trial={:?}",
            a.name(),
            gpu_med[ai]
                .iter()
                .map(|v| (v * 100.0).round() / 100.0)
                .collect::<Vec<_>>()
        );
        println!(
            "SUMMARY arm={:<6} chain {:8.2} us/replay = {:6.3} us per [producer+collective]  marginal collective vs base {:6.3} us",
            a.name(),
            med,
            med / BARRIERS as f64,
            (med - base_med) / BARRIERS as f64
        );
    }
    for (c, _) in caps.into_iter().zip(arms) {
        destroy(ranks, c);
    }
}

fn main() {
    let arms: Vec<Arm> = std::env::var("PROBE_ARMS")
        .unwrap_or_else(|_| "base,prodn".into())
        .split(',')
        .map(|s| match s.trim() {
            "base" => Arm::Base,
            "prodn" => Arm::Prodn,
            "file" => Arm::File,
            other => panic!("unknown arm {other}"),
        })
        .collect();
    let replays = env_usize("PROBE_CORRECT_REPLAYS", 2000);
    let pipeline = env_usize("PROBE_PIPELINE", 1);
    let n = env_usize("PROBE_TIMING_N", 200);
    let trials = env_usize("PROBE_TIMING_TRIALS", 15);
    let delays = env_usize("PROBE_DELAYS", 1) != 0;
    let seed = env_usize("PROBE_SEED", 0x5EED_1234) as u64;

    let mut gpus: Vec<Gpu> = (0..N as i32)
        .map(|i| Gpu::init_with_device(i).expect("init gpu"))
        .collect();
    for g in &gpus {
        assert_eq!(g.arch, "gfx1201", "probe needs gfx1201, got {}", g.arch);
    }
    let arch = gpus[0].arch.clone();
    let mut compiler = KernelCompiler::new(&arch, String::new()).expect("compiler");
    let probe_image = std::fs::read(
        compiler
            .compile("tp_peer_pushn_probe", SOURCE)
            .expect("compile probe"),
    )
    .unwrap();
    let shipped_image = std::fs::read(
        compiler
            .compile(
                "tp_peer_pushn_shipped",
                rdna_compute::kernels::TP_PEER_PUSHN_GFX1201_SRC,
            )
            .expect("compile shipped"),
    )
    .unwrap();
    let file_image = std::env::var("PROBE_KERNEL_FILE").ok().map(|path| {
        let body = std::fs::read_to_string(&path).expect("read PROBE_KERNEL_FILE");
        let src = format!(
            "{}{}{}",
            include_str!("../../../kernels/src/device_wait_deadline.h"),
            include_str!("../../../kernels/src/tp_exchange16.h"),
            body
        );
        std::fs::read(
            compiler
                .compile("tp_peer_pushn_file", &src)
                .expect("compile file"),
        )
        .unwrap()
    });

    let mut ranks: Vec<Rank> = Vec::new();
    for (rk, mut gpu) in gpus.drain(..).enumerate() {
        gpu.bind_thread().unwrap();
        gpu.active_stream = Some(gpu.hip.stream_create().expect("stream"));
        gpu.ensure_device_wait_block().expect("wait block");
        let hip = &gpu.hip;
        let m_probe = hip.module_load_data(&probe_image).expect("module");
        let m_ship = hip.module_load_data(&shipped_image).expect("module");
        let f_prod = hip.module_get_function(&m_probe, "probe_producer").unwrap();
        let f_spin = hip.module_get_function(&m_probe, "probe_spin").unwrap();
        let f_n = hip
            .module_get_function(&m_ship, "tp_peer_pushn_reduce_bf16_gfx1201")
            .unwrap();
        let mut modules = vec![m_probe, m_ship];
        let f_file = file_image.as_ref().map(|img| {
            let m = hip.module_load_data(img).expect("file module");
            let f = hip
                .module_get_function(&m, "tp_peer_pushn_reduce_bf16_gfx1201")
                .unwrap();
            modules.push(m);
            f
        });
        // allocated BEFORE peer access is enabled
        let mbox = hip.malloc_peer_finegrained(MBOX_BYTES).expect("mailbox");
        let x = hip.malloc(BYTES).unwrap();
        let part = hip.malloc(BYTES).unwrap();
        let salt = hip.malloc(256).unwrap();
        let gen = hip.malloc(256).unwrap();
        let wb_host = hip
            .host_malloc(
                16,
                hip_bridge::HIP_HOST_MALLOC_MAPPED | hip_bridge::HIP_HOST_MALLOC_COHERENT,
            )
            .expect("wb") as *mut DeviceWaitBlock;
        let khz = hip
            .get_device_attribute(10_017, gpu.device_id)
            .expect("clock") as u64;
        unsafe {
            std::ptr::write_volatile(
                wb_host,
                DeviceWaitBlock {
                    code: 0,
                    detail: 0,
                    timeout_ticks: timeout_ticks(khz),
                },
            )
        };
        let wb_dev = hip
            .host_get_device_pointer(wb_host.cast(), 0)
            .expect("wb alias");
        println!(
            "rank {rk}: device {} mailbox {:p} ({MBOX_BYTES} B fine-grained, slot {SLOT_STRIDE} bank {BANK_STRIDE})",
            gpu.device_id,
            mbox.as_ptr()
        );
        ranks.push(Rank {
            gpu,
            _modules: modules,
            f_prod,
            f_spin,
            f_n,
            f_file,
            x,
            part,
            salt,
            gen,
            mbox,
            wb_host,
            wb_dev,
            capturing: std::cell::Cell::new(false),
            blobs: std::cell::RefCell::new(Vec::new()),
        });
    }
    for i in 0..N {
        ranks[i].bind();
        for j in 0..N {
            if i == j {
                continue;
            }
            let (a, b) = (ranks[i].gpu.device_id, ranks[j].gpu.device_id);
            assert!(
                ranks[i].gpu.hip.can_access_peer(a, b).unwrap(),
                "no peer {a}->{b}"
            );
            ranks[i].gpu.hip.enable_peer_access(b).expect("enable peer");
        }
    }
    let [r0, r1, r2, r3]: [Rank; N] = ranks.try_into().ok().expect("four ranks");
    let mut ranks = [r0, r1, r2, r3];

    let mut all_ok = true;
    let skip_correct = env_usize("PROBE_SKIP_CORRECT", 0) != 0;
    for &arm in &arms {
        if arm == Arm::Base || skip_correct {
            continue;
        }
        all_ok &= correctness(&mut ranks, arm, replays, delays, seed, pipeline);
    }
    if all_ok {
        timing(&mut ranks, &arms, n, trials);
    } else {
        println!("correctness failed; timing skipped");
    }
    if env_usize("PROBE_TIMEOUT_TEST", 1) != 0 && all_ok && !skip_correct {
        for &arm in &arms {
            if arm == Arm::Base {
                continue;
            }
            let x_init: Vec<f32> = (0..COUNT).map(x0).collect();
            for r in ranks.iter() {
                r.reset_state(&x_init);
            }
            let t = Instant::now();
            // only rank 0 enters the collective: its bounded wait must stop
            enqueue_collective(&ranks, 0, arm);
            ranks[0].sync();
            let code = ranks[0].wait_code();
            println!(
                "{} timeout injection {}: bounded wait exited after {:.0} ms, wait code {code} (expect 10)",
                if code == 10 { "PASS" } else { "FAIL" },
                arm.name(),
                t.elapsed().as_secs_f64() * 1e3
            );
            all_ok &= code == 10;
        }
    }
    println!("{}", if all_ok { "ALL PASS" } else { "SOME FAIL" });
    std::process::exit(if all_ok { 0 } else { 1 });
}
