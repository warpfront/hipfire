// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! A4-fusion stage-2 oracle + timing on exact gfx1151.
//!
//! The certified V2B gate/up+SiLU (`gemm_gate_up_silu_mq4g256v2_iu4_prepared`,
//! writes f32 `h` [N][M]) followed by `fused_silu_hin_rotate_mq_i4_batched`
//! (writes the 72-byte `block_i4_128` AoS sidecar) is the reference. The
//! runtime reads `HIPFIRE_V2B_A4_EPI` from the process configuration
//! snapshot, so each process runs one arm and the caller compares md5s
//! across processes on the same deterministic inputs:
//!   * unset: the incumbent h and its A4;
//!   * `retile`: the M512xN128 twin in place of the incumbent gate_up (its h
//!     and A4 md5 must equal the unset run's);
//!   * `1`: the incumbent and, in the same process, the fused A4 kernel
//!     (`gemm_gate_up_silu_a4_mq4g256v2_iu4_prepared`), whose records in the
//!     poisoned down slot are compared byte for byte with the hin producer's.
//!
//! Build: `cargo build --release -p rdna-compute --features lab --example
//! test_mq4v2_gate_up_a4_gfx1151` (`--release`: the md5 is plain Rust).
//! usage:
//!   test_mq4v2_gate_up_a4_gfx1151 check M K N CORPUS SCALE SEED
//!       CORPUS = random | edge; SEED decimal or 0x-hex.
//!       Prints input md5s, per-differing-record dumps and one line
//!       `A4CHECK M K N corpus scale seed mode=… h_md5 … a4_md5_ref …
//!        a4_md5_fused …|- a4_diff_records … nan_h … zero_h … poison_h …
//!        unit_records … PASS|FAIL`; exits nonzero on FAIL.
//!   HIPFIRE_V2B_A4_EPI=1 test_mq4v2_gate_up_a4_gfx1151 time M K N ITERS
//!       (production shape: 17408 5120 8192) 1 s DPM warm-up then blocks
//!       G, H, F, F, G, H (G = incumbent gate_up, H = hin producer on G's h,
//!       F = fused), each 3 untimed + ITERS event-timed launches; prints
//!       per-block medians, T_G, T_H, T_F, delta and the 0.781 ms gate.
//!
//! Prints `SKIP` and exits 0 when the device is not exact gfx1151.
//! M = gate_m = up_m = down K, K = hidden, N = tokens (the generators and md5
//! are verbatim from the stage-1 standalone probe `a4fuse`, so the corpora
//! are the same).

use hip_bridge::DeviceBuffer;
use rdna_compute::scratch::Int4MmqPrepared;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::{Duration, Instant};

const POISON: i32 = 0x5a;
const EPI_VAR: &str = "HIPFIRE_V2B_A4_EPI";
const GATE_MS: f64 = -0.781;

// ---------------------------------------------------------------- md5 (RFC 1321)
fn md5(data: &[u8]) -> String {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64)
        .map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32)
        .collect();
    let mut st = [0x67452301u32, 0xefcdab89, 0x98badcfe, 0x10325476];
    let block = |st: &mut [u32; 4], c: &[u8]| {
        let m: Vec<u32> = c
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        let (mut a, mut b, mut cc, mut d) = (st[0], st[1], st[2], st[3]);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & cc) | (!b & d), i),
                1 => ((d & b) | (!d & cc), (5 * i + 1) % 16),
                2 => (b ^ cc ^ d, (3 * i + 5) % 16),
                _ => (cc ^ (b | !d), (7 * i) % 16),
            };
            let t = d;
            d = cc;
            cc = b;
            b = b.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(k[i])
                    .wrapping_add(m[g])
                    .rotate_left(S[i]),
            );
            a = t;
        }
        st[0] = st[0].wrapping_add(a);
        st[1] = st[1].wrapping_add(b);
        st[2] = st[2].wrapping_add(cc);
        st[3] = st[3].wrapping_add(d);
    };
    let full = data.len() / 64 * 64;
    for c in data[..full].chunks_exact(64) {
        block(&mut st, c);
    }
    let mut tail = data[full..].to_vec();
    tail.push(0x80);
    while tail.len() % 64 != 56 {
        tail.push(0);
    }
    tail.extend_from_slice(&((data.len() as u64) * 8).to_le_bytes());
    for c in tail.chunks_exact(64) {
        block(&mut st, c);
    }
    st.iter()
        .flat_map(|w| w.to_le_bytes())
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---------------------------------------------------------------- inputs
/// xorshift64* stream.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn u(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / 16777216.0
    }
}
fn mix(a: u64, b: u64, c: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9E3779B97F4A7C15)
        ^ b.wrapping_mul(0xC2B2AE3D27D4EB4F)
        ^ c.wrapping_mul(0x165667B19E3779F9);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^ (x >> 29)
}

/// MQ4G256V2 rows `[M][K/256][f16 sc0, zp0, sc1, zp1][128 nibble bytes]`.
/// Random: scales 2^-9..2^-5 (f16 exponent 6..10). Edge: one pattern per
/// row, a quarter of the rows special: inf / zero / denormal / max-finite
/// half scales in every group, a NaN scale or a -inf scale over zero codes
/// (inf*0 = NaN) in one group, all-zero codes (codes 0x88 rebias to 0: the
/// row's sums are exactly 0), or denormal scales everywhere (tiny sums);
/// ordinary rows get sparse finite group anomalies (zero codes, zero or
/// denormal scales) at 1/64.
fn weights(m: usize, k: usize, seed: u64, edge: bool) -> Vec<u8> {
    let groups = k / 256;
    let mut v = vec![0u8; m * groups * 136];
    let mut r = Rng(seed | 1);
    for row in 0..m {
        let rh = mix(seed, row as u64, 0);
        let (pat, one) = (rh % 32, (rh >> 8) as usize % groups);
        for g in 0..groups {
            let o = (row * groups + g) * 136;
            let mut sc = [0u16; 2];
            for h in 0..2 {
                sc[h] = (((6 + r.next() % 5) as u16) << 10) | (r.next() as u16 & 0x3ff);
            }
            let zp = [r.next() as u16, r.next() as u16];
            for b in 0..128 {
                v[o + 8 + b] = r.next() as u8;
            }
            if edge {
                match pat {
                    24 => sc = [0x7c00, 0x7c00],
                    25 => sc = [0x0000, 0x8000],
                    26 => sc = [0x0001, 0x8001],
                    27 if g == one => sc = [0x7e00, sc[1]],
                    28 => sc = [0x7bff, 0xfbff],
                    29 => v[o + 8..o + 136].fill(0x88),
                    30 if g == one => {
                        sc = [0xfc00, 0x7c00];
                        v[o + 8..o + 136].fill(0x88);
                    }
                    31 => {
                        sc = [
                            (r.next() as u16) & 0x3ff,
                            0x8000 | ((r.next() as u16) & 0x3ff),
                        ]
                    }
                    0..=23 => match mix(seed ^ 0x9E37, row as u64, g as u64) % 64 {
                        0 => v[o + 8..o + 136].fill(0x88),
                        1 => sc = [0x0000, sc[1]],
                        2 => sc = [sc[0], 0x0002],
                        _ => {}
                    },
                    _ => {}
                }
            }
            for (i, x) in [sc[0], zp[0], sc[1], zp[1]].into_iter().enumerate() {
                v[o + 2 * i..o + 2 * i + 2].copy_from_slice(&x.to_le_bytes());
            }
        }
    }
    v
}

/// `block_i4_128` AoS `[K/128][N]{f32 d, i32 s, u8 qs[64]}`. Random: d =
/// scale * 2^-4 * U[0.5, 1.5). Edge: one pattern per token, a quarter of
/// the tokens special: d = 0, -0, denormal, 1e30 or 2^-110 in every block,
/// NaN or inf in one block, or all-zero codes; ordinary tokens get sparse
/// finite block anomalies (d = 0, -0, denormal, 2^-110, -d, zero codes)
/// at 6/32.
fn xq(n: usize, k: usize, seed: u64, scale: f32, edge: bool) -> Vec<u8> {
    let blocks = k / 128;
    let mut v = vec![0u8; blocks * n * 72];
    let mut r = Rng(seed | 1);
    for kb in 0..blocks {
        for t in 0..n {
            let o = (kb * n + t) * 72;
            let mut d = scale * 0.0625 * (0.5 + r.u());
            let s = r.next() as u32;
            for b in 0..64 {
                v[o + 8 + b] = r.next() as u8;
            }
            if edge {
                let th = mix(seed ^ 0x5851F42D4C957F2D, t as u64, 0);
                let (pat, one) = (th % 32, (th >> 8) as usize % blocks);
                match pat {
                    24 => d = 0.0,
                    25 => d = -0.0,
                    26 => d = f32::from_bits(0x0000_0003),
                    27 if kb == one => d = f32::NAN,
                    28 if kb == one => d = f32::INFINITY,
                    29 => d = 1e30,
                    30 => d = (2.0f32).powi(-110),
                    31 => v[o + 8..o + 72].fill(0),
                    0..=23 => match mix(seed ^ 0xC2B2, kb as u64, t as u64) % 32 {
                        0 => d = 0.0,
                        1 => d = -0.0,
                        2 => d = f32::from_bits(0x0000_0007),
                        3 => d = (2.0f32).powi(-110),
                        4 => d = -d,
                        5 => v[o + 8..o + 72].fill(0),
                        _ => {}
                    },
                    _ => {}
                }
            }
            v[o..o + 4].copy_from_slice(&d.to_le_bytes());
            v[o + 4..o + 8].copy_from_slice(&s.to_le_bytes());
        }
    }
    v
}

/// AWQ scale f32[M]: random positive in [0.25, 4); the edge corpus overwrites
/// a quarter of the rows with exact powers of two (including 2^-20 / 2^20).
fn awq_scales(m: usize, seed: u64, edge: bool) -> Vec<f32> {
    let mut r = Rng((seed ^ 0xA3C1_5EED) | 1);
    let mut v: Vec<f32> = (0..m).map(|_| 0.25 + 3.75 * r.u()).collect();
    if edge {
        const SPECIAL: [f32; 8] = [1.0, 2.0, 0.5, 0.25, 4.0, 1.0 / 1048576.0, 1048576.0, 1.0];
        for (i, x) in v.iter_mut().enumerate() {
            let h = mix(seed ^ 0xA3C1, i as u64, 7);
            if h % 4 == 0 {
                *x = SPECIAL[(h >> 8) as usize % SPECIAL.len()];
            }
        }
    }
    v
}

// ---------------------------------------------------------------- harness
fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}

fn hip<T>(r: Result<T, hip_bridge::HipError>, what: &str) -> T {
    r.unwrap_or_else(|e| die(&format!("{what}: {e}")))
}

fn as_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding/invalid byte patterns; len = 4 * v.len().
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn parse_u64(s: &str) -> u64 {
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(h, 16),
        None => s.parse(),
    }
    .unwrap_or_else(|_| die(&format!("bad integer {s:?}")))
}

/// The process's `HIPFIRE_V2B_A4_EPI` arm. The runtime reads it from the
/// process configuration snapshot (fixed at the first read), so one process
/// runs one arm: unset/`0` the incumbent, `retile` the twin in place of the
/// incumbent gate_up, `1` the incumbent plus the fused entry.
fn mode() -> String {
    std::env::var(EPI_VAR).unwrap_or_default()
}

struct Inputs {
    m: usize,
    k: usize,
    n: usize,
    gate: GpuTensor,
    up: GpuTensor,
    awq: GpuTensor,
    xq: Vec<u8>,
}

fn build(gpu: &mut Gpu, m: usize, k: usize, n: usize, edge: bool, scale: f32, seed: u64) -> Inputs {
    let gate_h = weights(m, k, seed ^ 0x6A7E, edge);
    let up_h = weights(m, k, seed ^ 0x0B0B_5EED, edge);
    let xq_h = xq(n, k, seed, scale, edge);
    let awq_h = awq_scales(m, seed, edge);
    println!("input gate md5 {}", md5(&gate_h));
    println!("input up   md5 {}", md5(&up_h));
    println!("input xq   md5 {}", md5(&xq_h));
    println!("input awq  md5 {}", md5(as_bytes(&awq_h)));
    let gate = hip(gpu.upload_raw(&gate_h, &[gate_h.len()]), "upload gate");
    let up = hip(gpu.upload_raw(&up_h, &[up_h.len()]), "upload up");
    let awq = hip(gpu.upload_f32(&awq_h, &[m]), "upload awq");
    Inputs {
        m,
        k,
        n,
        gate,
        up,
        awq,
        xq: xq_h,
    }
}

/// Fresh x-scratch reservation filled with the Xq bytes (new generation).
fn upload_xq(gpu: &mut Gpu, inp: &Inputs) -> Int4MmqPrepared {
    let res = hip(gpu.reserve_int4_mmq(inp.k, inp.n), "reserve x");
    // SAFETY: non-owning view of the live reservation (k/128*n*72 bytes).
    let view = unsafe { DeviceBuffer::from_raw(res.ptr(), inp.xq.len()) };
    hip(gpu.hip.memcpy_htod(&view, &inp.xq), "htod xq");
    hip(gpu.hip.device_synchronize(), "sync xq");
    Int4MmqPrepared::from_reservation(res)
}

fn a4_bytes(inp: &Inputs) -> usize {
    inp.m / 128 * inp.n * 72
}

fn download(gpu: &Gpu, ptr: *mut std::ffi::c_void, len: usize) -> Vec<u8> {
    hip(gpu.hip.device_synchronize(), "sync before download");
    // SAFETY: non-owning view of a live scratch region of `len` bytes.
    let view = unsafe { DeviceBuffer::from_raw(ptr, len) };
    let mut out = vec![0u8; len];
    hip(gpu.hip.memcpy_dtoh(&mut out, &view), "dtoh");
    out
}

fn gate_up(gpu: &mut Gpu, inp: &Inputs, prep: &Int4MmqPrepared, h: &GpuTensor) -> bool {
    hip(
        gpu.gemm_gate_up_silu_mq4g256v2_iu4_prepared(
            &inp.gate, &inp.up, prep, h, inp.m, inp.m, inp.k, inp.n,
        ),
        "gate_up",
    )
}

fn hin(gpu: &mut Gpu, inp: &Inputs, h: &GpuTensor) -> Int4MmqPrepared {
    let res = hip(gpu.reserve_int4_mmq(inp.m, inp.n), "reserve hin");
    hip(
        gpu.fused_silu_hin_rotate_mq_i4_batched(h, &inp.awq, res, inp.m, inp.n),
        "hin",
    )
}

fn fused(
    gpu: &mut Gpu,
    inp: &Inputs,
    prep: &Int4MmqPrepared,
) -> Option<rdna_compute::scratch::Int4MmqDownPrepared> {
    hip(
        gpu.gemm_gate_up_silu_a4_mq4g256v2_iu4_prepared(
            &inp.gate, &inp.up, prep, &inp.awq, inp.m, inp.m, inp.k, inp.n,
        ),
        "gate_up_a4",
    )
}

fn alloc_h(gpu: &mut Gpu, inp: &Inputs) -> GpuTensor {
    hip(gpu.alloc_tensor(&[inp.n * inp.m], DType::F32), "alloc h")
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn dump_record(label: &str, r: &[u8]) {
    let d = f32::from_le_bytes(r[0..4].try_into().unwrap());
    let s = i32::from_le_bytes(r[4..8].try_into().unwrap());
    println!(
        "    {label}: d={d:e} (0x{:08x}) s={s} qs[..16]={} qs_md5={}",
        d.to_bits(),
        hex(&r[8..24]),
        md5(&r[8..72])
    );
}

// ---------------------------------------------------------------- check
fn check(gpu: &mut Gpu, m: usize, k: usize, n: usize, corpus: &str, scale: f32, seed: u64) -> bool {
    let edge = match corpus {
        "random" => false,
        "edge" => true,
        _ => die("CORPUS must be random|edge"),
    };
    let mode = mode();
    let inp = build(gpu, m, k, n, edge, scale, seed);
    let a4_len = a4_bytes(&inp);
    // Size the x scratch for the larger of the two producers once so no
    // reservation below grows it (growth would invalidate the live sidecar).
    drop(hip(gpu.reserve_int4_mmq(m.max(k), n), "pre-reserve x"));
    drop(hip(gpu.reserve_int4_mmq_down(m, n), "pre-reserve down"));

    // 1. gate_up (the incumbent, or the twin under `retile`) into a
    //    poisoned h, then the hin producer.
    let h = alloc_h(gpu, &inp);
    hip(gpu.hip.memset(&h.buf, POISON, h.buf.size()), "poison h");
    let prep = upload_xq(gpu, &inp);
    if !gate_up(gpu, &inp, &prep, &h) {
        die("A4CHECK: gate_up_silu returned false (not admitted)");
    }
    let h_host = download_f32_bytes(gpu, &h);
    let prep_ref = hin(gpu, &inp, &h);
    let a4_ref_ptr = hip(gpu.int4_mmq_prepared_ptr(&prep_ref, m, n), "ref a4 ptr");
    let a4_ref = download(gpu, a4_ref_ptr, a4_len);

    // 2. `1`: the fused entry's A4 records, straight into the poisoned down slot.
    let a4_fused = (mode == "1").then(|| {
        let res = hip(gpu.reserve_int4_mmq_down(m, n), "reserve down");
        // SAFETY: non-owning view of the live down reservation (a4_len bytes).
        let down_view = unsafe { DeviceBuffer::from_raw(res.ptr(), a4_len) };
        hip(gpu.hip.memset(&down_view, POISON, a4_len), "poison down");
        drop(res);
        let prep = upload_xq(gpu, &inp);
        let Some(p) = fused(gpu, &inp, &prep) else {
            die("A4CHECK: fused gate_up_a4 returned None (not admitted)");
        };
        let a4_fused_ptr = hip(gpu.int4_mmq_down_prepared_ptr(&p, m, n), "fused a4 ptr");
        download(gpu, a4_fused_ptr, a4_len)
    });

    let rec_diff: Vec<usize> = match &a4_fused {
        Some(f) => a4_ref
            .chunks_exact(72)
            .zip(f.chunks_exact(72))
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect(),
        None => Vec::new(),
    };
    if let Some(f) = &a4_fused {
        for &i in rec_diff.iter().take(8) {
            println!("  differing record {i}: token {} block {}", i % n, i / n);
            dump_record("ref  ", &a4_ref[i * 72..i * 72 + 72]);
            dump_record("fused", &f[i * 72..i * 72 + 72]);
        }
    }
    let words = h_host
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()));
    let (mut nan_h, mut zero_h, mut poison_h) = (0usize, 0usize, 0usize);
    for w in words {
        if f32::from_bits(w).is_nan() {
            nan_h += 1;
        }
        if w << 1 == 0 {
            zero_h += 1;
        }
        if w == 0x5a5a_5a5a {
            poison_h += 1;
        }
    }
    let unit_records = a4_ref
        .chunks_exact(72)
        .filter(|r| u32::from_le_bytes(r[0..4].try_into().unwrap()) == 1.0f32.to_bits())
        .count();
    let pass = poison_h == 0 && rec_diff.is_empty();
    println!(
        "A4CHECK {m} {k} {n} {corpus} {scale} {seed:#x} mode={} h_md5 {} a4_md5_ref {} a4_md5_fused {} a4_diff_records {} nan_h {nan_h} zero_h {zero_h} poison_h {poison_h} unit_records {unit_records} {}",
        if mode.is_empty() { "off" } else { &mode },
        md5(&h_host),
        md5(&a4_ref),
        a4_fused.as_deref().map_or_else(|| "-".to_owned(), md5),
        rec_diff.len(),
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn download_f32_bytes(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    hip(gpu.hip.device_synchronize(), "sync before h download");
    let mut out = vec![0u8; t.buf.size()];
    hip(gpu.hip.memcpy_dtoh(&mut out, &t.buf), "dtoh h");
    out
}

// ---------------------------------------------------------------- time
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let l = v.len();
    if l % 2 == 1 {
        v[l / 2]
    } else {
        0.5 * (v[l / 2 - 1] + v[l / 2])
    }
}

struct Timer {
    start: hip_bridge::Event,
    stop: hip_bridge::Event,
}

impl Timer {
    /// GPU time (ms) of the work `f` enqueues on the Gpu's stream.
    fn ms(&self, gpu: &mut Gpu, f: &mut dyn FnMut(&mut Gpu)) -> f64 {
        hip(
            gpu.hip
                .event_record(&self.start, gpu.active_stream.as_ref()),
            "record start",
        );
        f(gpu);
        hip(
            gpu.hip.event_record(&self.stop, gpu.active_stream.as_ref()),
            "record stop",
        );
        hip(gpu.hip.event_synchronize(&self.stop), "sync stop");
        hip(gpu.hip.event_elapsed_ms(&self.start, &self.stop), "elapsed") as f64
    }
}

/// 3 untimed warm-ups, then `iters` individually event-timed calls of `f`.
/// `prep` runs before every launch, outside the timed region, and hands its
/// result to `f`.
fn block<S>(
    gpu: &mut Gpu,
    t: &Timer,
    iters: usize,
    prep: &mut dyn FnMut(&mut Gpu) -> S,
    f: &mut dyn FnMut(&mut Gpu, S),
) -> Vec<f64> {
    for _ in 0..3 {
        let s = prep(gpu);
        f(gpu, s);
    }
    hip(gpu.hip.device_synchronize(), "sync warm-up");
    let mut v = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut s = Some(prep(gpu));
        v.push(t.ms(gpu, &mut |g: &mut Gpu| f(g, s.take().unwrap())));
    }
    v
}

fn time(gpu: &mut Gpu, m: usize, k: usize, n: usize, iters: usize) {
    if iters == 0 {
        die("ITERS must be > 0");
    }
    // G, H and F share one process: `1` keeps gate_up on the incumbent and
    // enables the fused entry.
    if mode() != "1" {
        die("time: run with HIPFIRE_V2B_A4_EPI=1");
    }
    let inp = build(gpu, m, k, n, false, 1.0, 0xA4F1);
    let a4_len = a4_bytes(&inp);
    drop(hip(gpu.reserve_int4_mmq(m.max(k), n), "pre-reserve x"));
    drop(hip(gpu.reserve_int4_mmq_down(m, n), "pre-reserve down"));
    let h = alloc_h(gpu, &inp);
    let timer = Timer {
        start: hip(gpu.hip.event_create(), "event"),
        stop: hip(gpu.hip.event_create(), "event"),
    };

    // 1 s DPM warm-up on the incumbent GEMM.
    let prep = upload_xq(gpu, &inp);
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(1) {
        for _ in 0..4 {
            if !gate_up(gpu, &inp, &prep, &h) {
                die("time: ref gate_up_silu returned false (not admitted)");
            }
        }
        hip(gpu.hip.device_synchronize(), "sync warm-up");
    }

    let mut last_fused = None;
    let mut last_ref = None;
    let mut meds: Vec<(&str, f64)> = Vec::new();
    let order = ["G", "H", "F", "F", "G", "H"];
    for (bi, arm) in order.iter().enumerate() {
        let mut v = match *arm {
            "G" | "F" => {
                // Xq is clobbered by H: fresh reservation + upload per G/F block;
                // G/F never write it, so no re-upload inside the loop.
                let prep = upload_xq(gpu, &inp);
                if *arm == "G" {
                    block(
                        gpu,
                        &timer,
                        iters,
                        &mut |_: &mut Gpu| (),
                        &mut |g: &mut Gpu, ()| {
                            if !gate_up(g, &inp, &prep, &h) {
                                die("time: ref gate_up_silu returned false (not admitted)");
                            }
                        },
                    )
                } else {
                    block(
                        gpu,
                        &timer,
                        iters,
                        &mut |_: &mut Gpu| (),
                        &mut |g: &mut Gpu, ()| match fused(g, &inp, &prep) {
                            Some(p) => last_fused = Some(p),
                            None => die("time: fused returned None (not admitted / flag off)"),
                        },
                    )
                }
            }
            _ => {
                // H reads G's h (written by the preceding G block) and writes
                // the x scratch; the per-launch reservation is made outside
                // the timed region.
                block(
                    gpu,
                    &timer,
                    iters,
                    &mut |g: &mut Gpu| hip(g.reserve_int4_mmq(inp.m, inp.n), "reserve hin"),
                    &mut |g: &mut Gpu, res| {
                        last_ref = Some(hip(
                            g.fused_silu_hin_rotate_mq_i4_batched(&h, &inp.awq, res, inp.m, inp.n),
                            "hin",
                        ));
                    },
                )
            }
        };
        let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let med = median(&mut v);
        println!(
            "BLOCK {} {arm} median_ms {med:.4} min_ms {mn:.4} iters {iters}",
            bi + 1
        );
        meds.push((arm, med));
        hip(gpu.hip.device_synchronize(), "sync block");
    }

    let mean = |tag: &str| {
        let xs: Vec<f64> = meds
            .iter()
            .filter(|(a, _)| *a == tag)
            .map(|(_, x)| *x)
            .collect();
        xs.iter().sum::<f64>() / xs.len() as f64
    };
    let (t_g, t_h, t_f) = (mean("G"), mean("H"), mean("F"));
    let delta = t_f - (t_g + t_h);
    println!("T_G {t_g:.4} ms  T_H {t_h:.4} ms  T_F {t_f:.4} ms");
    println!(
        "delta = T_F - (T_G + T_H) = {delta:.4} ms  gate(delta <= {GATE_MS}) = {}",
        if delta <= GATE_MS { "PASS" } else { "FAIL" }
    );

    let p = last_fused.expect("fused block ran");
    let a4_fused_ptr = hip(gpu.int4_mmq_down_prepared_ptr(&p, m, n), "fused a4 ptr");
    let a4_fused = download(gpu, a4_fused_ptr, a4_len);
    let pr = last_ref.expect("H block ran");
    let a4_ref_ptr = hip(gpu.int4_mmq_prepared_ptr(&pr, m, n), "ref a4 ptr");
    let a4_ref = download(gpu, a4_ref_ptr, a4_len);
    let (mf, mr) = (md5(&a4_fused), md5(&a4_ref));
    println!("a4_md5_fused {mf}");
    println!("a4_md5_ref   {mr}");
    println!("a4 md5 {}", if mf == mr { "MATCH" } else { "DIFF" });
    if mf != mr {
        std::process::exit(1);
    }
}

fn usage() -> ! {
    die("usage: test_mq4v2_gate_up_a4_gfx1151 check M K N CORPUS SCALE SEED | time M K N ITERS")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("SKIP: no GPU ({e})");
            return;
        }
    };
    if gpu.arch != "gfx1151" {
        println!("SKIP: arch {} is not gfx1151", gpu.arch);
        return;
    }
    // The generated MQ4V2 weights are the symmetric layout the V2B tile folds
    // (the loader sets this for symmetric models such as qwen3.8-27b.mq4-xts).
    gpu.mq4v2_symmetric = true;
    let num = |i: usize| -> usize {
        args.get(i)
            .map(|s| parse_u64(s) as usize)
            .unwrap_or_else(|| usage())
    };
    match args.first().map(String::as_str) {
        Some("check") if args.len() == 7 => {
            let scale: f32 = args[5].parse().unwrap_or_else(|_| die("bad SCALE"));
            if !check(
                &mut gpu,
                num(1),
                num(2),
                num(3),
                &args[4],
                scale,
                parse_u64(&args[6]),
            ) {
                std::process::exit(1);
            }
        }
        Some("time") if args.len() == 5 => time(&mut gpu, num(1), num(2), num(3), num(4)),
        _ => usage(),
    }
}
