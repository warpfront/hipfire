// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! T0 (KERNEL-PLAN §1.4 / §5): MQ6SymG256 x A8 decode GEMV rows 1..8 on exact gfx1201 —
//! CPU byte oracle + timing arms.
//!
//! The candidate kernel is loaded straight from `kernels/src/gemv_mq6sym_i8.gfx1201.hip` (JIT
//! with the `block_i8_128` prelude). Activations come from the REAL fused FWHT+A8 producer
//! (`Gpu::rotate_x_mq_i8_batched`, entry `mq_rotate_x_i8_gfx12`).
//!
//! Correctness test (`mq6sym_i8_decode_matches_cpu_bytes_hw`): an independent scalar Rust
//! reference (bit-level 6-bit decode, natural K order recovered from the documented nibble-pair
//! qs order, exact i32 C per 128 block, `sd = RN(f32(s_h) * d)`, `acc = fma(sd, C as f32, acc)`
//! in increasing block order from +0.0) is compared byte-for-byte on
//!   * the producer's `block_i8_128` bytes (vs the CPU quantizer applied to the producer's own
//!     rotated f32 output: d bits, s, every qs byte),
//!   * the per-block i32 C (test-only `-DS6_DEBUG_C` build of the same kernel source),
//!   * the final F32 words and the BF16 entries' u16 words, plus guard words around Y,
//! for exhaustive 6-bit codes, scale classes (0 / subnormal / large / mixed halves), activation
//! classes (zero / ties-to-even / +-127 spikes / tiny / sd-subnormal), N 1..8, K 256/512/2560/6144,
//! M 1/17/48/512/640/2560/6144/10240/12288 and the lm_head M = 248320 (skip with
//! `MQ6SYM_SKIP_LM=1`). A scalar-F32 dequant x activation "quality diagnostic" (f64 accumulate over
//! the producer's rotated activations) reports max relative error; it is not a byte gate.
//!
//! Timing test (`mq6sym_i8_decode_timing_hw`): cold cyclic (>= 256 MiB of rotated weight sets),
//! rows 1/2/4/8: the symmetric kernel alone, producer alone, producer + kernel, a same-job b128
//! streaming roofline over the same bytes, and the OLD product route (affine MQ6G256V2 F32 rotate
//! + `gemv_mq6g256v2` / `gemm_mq6g256v2_f32_rows`) as separate arms. Weights-only GB/s counts the
//! 200-B (qt47 MQ6G256V2, both routes) bytes actually read.
//!
//! `#[ignore]`d: needs an exact gfx1201 device.
//!   cargo test -p rdna-compute --release --features lab --test mq6sym_i8_decode_hw -- --ignored --nocapture --test-threads=1 <name>
//! Env: `MQ6SYM_OUT` = directory for the JSON summary; `MQ6SYM_SKIP_LM=1` skips lm_head cells.

use hip_bridge::{DeviceBuffer, KernargBlob};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;
use std::fmt::Write as _;

const KERNEL_SRC: &str = concat!(
    "#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n",
    include_str!("../../../kernels/src/block_i8_128_quant.hip"),
    include_str!("../../../kernels/src/gemv_mq6sym_i8.gfx1201.hip")
);
const MODULE: &str = "mq6sym_i8_t0";
const MODULE_DBG: &str = "mq6sym_i8_t0_dbg";
const GROUP_BYTES: usize = 200;
const BLOCK_BYTES: usize = 136;
const CHUNK_GROUPS: usize = 12;

fn entry(n: usize, bf16: bool) -> String {
    if bf16 {
        format!("gemv_mq6sym_i8_rows_r{n}_bf16_gfx1201")
    } else {
        format!("gemv_mq6sym_i8_rows_r{n}_gfx1201")
    }
}

fn entry_dbg(n: usize) -> String {
    format!("gemv_mq6sym_i8_rows_r{n}_dbgc_gfx1201")
}

/// `MQ6SYM_DEFS="S6_PF=1,S6_CHUNK_GROUPS=24"` prepends `#define` lines to the kernel source
/// (A/B sweeps without rebuilding the test); unset = the shipped defaults.
fn kernel_defs() -> Vec<(String, String)> {
    std::env::var("MQ6SYM_DEFS")
        .ok()
        .map(|v| {
            v.split(',')
                .filter(|s| !s.is_empty())
                .map(|kv| {
                    let (k, val) = kv.split_once('=').unwrap_or((kv, "1"));
                    (k.to_string(), val.to_string())
                })
                .collect()
        })
        .unwrap_or_default()
}

fn kernel_src() -> String {
    let mut s = String::new();
    for (k, v) in kernel_defs() {
        writeln!(s, "#define {k} {v}").ok();
    }
    s.push_str(KERNEL_SRC);
    s
}

fn chunk_groups() -> usize {
    kernel_defs()
        .iter()
        .find(|(k, _)| k == "S6_CHUNK_GROUPS")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(CHUNK_GROUPS)
}

/// Dynamic LDS of the two-slot A8 ring: the token-split layout (S6Ts<R>::SLOT = 2 (R + B) 144 B
/// cells + 8 R B of d, B = 2 at R = 7 else 1) is the larger of the two layouts at every R.
fn lds_bytes(_k: usize, n: usize) -> u32 {
    let b = if n == 7 { 2 } else { 1 };
    let slot = 2 * (n + b) * 144 + 8 * n;
    (2 * slot).max(528 * n) as u32
}

// ---------------------------------------------------------------------------------------------
// RNG + small helpers
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn gauss(&mut self) -> f32 {
        let r = self.next();
        let mut s = 0f32;
        for i in 0..4 {
            s += ((r >> (16 * i)) & 0xffff) as f32 / 32768.0 - 1.0;
        }
        s * 0.866_025_4
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(4)
}

/// Run `f(chunk_index, chunk)` over fixed-size chunks of `data` on up to `threads()` threads.
fn par_chunks_mut<T: Send, F: Fn(usize, &mut [T]) + Sync>(data: &mut [T], chunk: usize, f: F) {
    let nt = threads();
    let chunks: Vec<(usize, &mut [T])> = data.chunks_mut(chunk).enumerate().collect();
    let mut buckets: Vec<Vec<(usize, &mut [T])>> = (0..nt).map(|_| Vec::new()).collect();
    for (i, c) in chunks {
        buckets[i % nt].push((i, c));
    }
    std::thread::scope(|s| {
        for b in buckets {
            let f = &f;
            s.spawn(move || {
                for (i, c) in b {
                    f(i, c);
                }
            });
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Weight generation (host bytes in the MQ6SymG256 layout)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum CMode {
    Random,
    /// code(row, pos, g) = (row + pos + 7 g) & 63: every value at every position over 64 rows.
    Exhaustive,
    AllLow,
    AllHigh,
    Zero32,
    Alternating,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SMode {
    /// Positive normal scales 2^-7 .. 2^3.
    Normal,
    Zero,
    Subnormal,
    Large,
    /// Per-half random class (zero / subnormal / normal +- / large / one).
    Mixed,
}

fn scale_bits(mode: SMode, rng: &mut Rng) -> u16 {
    match mode {
        SMode::Normal => {
            let exp = 8 + rng.below(11) as u16; // 2^-7 .. 2^3
            (exp << 10) | (rng.below(1024) as u16)
        }
        SMode::Zero => 0,
        SMode::Subnormal => {
            [0x0001u16, 0x0003, 0x01FF, 0x03FF, 0x8001, 0x83FF][rng.below(6) as usize]
        }
        SMode::Large => [0x6FFFu16, 0xEFFF, 0x6C00, 0x6E00][rng.below(4) as usize],
        SMode::Mixed => match rng.below(6) {
            0 => 0,
            1 => [0x0001u16, 0x03FF, 0x8001][rng.below(3) as usize],
            2 => scale_bits(SMode::Normal, rng),
            3 => scale_bits(SMode::Normal, rng) | 0x8000,
            4 => [0x6FFFu16, 0xEFFF][rng.below(2) as usize],
            _ => 0x3C00,
        },
    }
}

/// Triple-pack encoder: 4 codes per 3 bytes `q0 | q1 << 6 | q2 << 12 | q3 << 18`, LE.
fn encode_group(hdr: [u16; 2], codes: &[u8; 256], out: &mut [u8]) {
    // qt47 header: {s0 | z0 << 16}, {s1 | z1 << 16}, z_h = f16(-32 s_h) (exact; the kernels ignore z).
    let z = |s: u16| half::f16::from_f32(-32.0 * f16_to_f32(s)).to_bits();
    out[0..2].copy_from_slice(&hdr[0].to_le_bytes());
    out[2..4].copy_from_slice(&z(hdr[0]).to_le_bytes());
    out[4..6].copy_from_slice(&hdr[1].to_le_bytes());
    out[6..8].copy_from_slice(&z(hdr[1]).to_le_bytes());
    for t in 0..64 {
        let v = (codes[4 * t] as u32)
            | ((codes[4 * t + 1] as u32) << 6)
            | ((codes[4 * t + 2] as u32) << 12)
            | ((codes[4 * t + 3] as u32) << 18);
        out[8 + 3 * t] = v as u8;
        out[8 + 3 * t + 1] = (v >> 8) as u8;
        out[8 + 3 * t + 2] = (v >> 16) as u8;
    }
}

/// Independent bit-level decoder: code c = bits [6c, 6c + 6) of the 192-byte LE stream.
fn decode_codes(group: &[u8]) -> [u8; 256] {
    let payload = &group[8..200];
    let mut out = [0u8; 256];
    for (c, o) in out.iter_mut().enumerate() {
        let bo = 6 * c;
        let lo = payload[bo / 8] as u32;
        let hi = payload.get(bo / 8 + 1).copied().unwrap_or(0) as u32;
        *o = (((lo | (hi << 8)) >> (bo % 8)) & 63) as u8;
    }
    out
}

fn gen_weights(m: usize, k: usize, cm: CMode, sm: SMode, seed: u64) -> Vec<u8> {
    let g_per_row = k / 256;
    let row_bytes = g_per_row * GROUP_BYTES;
    let mut w = vec![0u8; m * row_bytes];
    par_chunks_mut(&mut w, 256 * row_bytes, |ci, chunk| {
        let mut rng = Rng(seed ^ (ci as u64).wrapping_mul(0xA24B_AED4_963E_E407));
        let rows = chunk.len() / row_bytes;
        for r in 0..rows {
            let grow = ci * 256 + r;
            for g in 0..g_per_row {
                let mut codes = [0u8; 256];
                for (p, c) in codes.iter_mut().enumerate() {
                    *c = match cm {
                        CMode::Random => rng.below(64) as u8,
                        CMode::Exhaustive => ((grow + p + 7 * g) & 63) as u8,
                        CMode::AllLow => 0,
                        CMode::AllHigh => 63,
                        CMode::Zero32 => 32,
                        CMode::Alternating => {
                            if (p + g) & 1 == 0 {
                                0
                            } else {
                                63
                            }
                        }
                    };
                }
                let hdr = [scale_bits(sm, &mut rng), scale_bits(sm, &mut rng)];
                let off = r * row_bytes + g * GROUP_BYTES;
                encode_group(hdr, &codes, &mut chunk[off..off + GROUP_BYTES]);
            }
        }
    });
    w
}

// ---------------------------------------------------------------------------------------------
// Activations (F32 inputs of the real producer)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum AMode {
    Gauss,
    Outliers,
    Zero,
    /// Token 0 zero, others gauss.
    HalfZero,
    /// Designed so rotated values sit exactly on k + 0.5 of the quantizer grid (ties-to-even).
    Ties,
    /// One nonzero per 256 group: every rotated value is +-amax -> q = +-127.
    Spike,
    Tiny,
    /// Scale so sd = RN(s_h * d) is f32-subnormal for tiny f16 scales.
    DenormSd,
}

fn gen_act(k: usize, n: usize, mode: AMode, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let mut x = vec![0f32; n * k];
    for t in 0..n {
        for g in 0..k / 256 {
            let row = &mut x[t * k + g * 256..t * k + (g + 1) * 256];
            match mode {
                AMode::Gauss => row.iter_mut().for_each(|v| *v = rng.gauss()),
                AMode::Outliers => {
                    row.iter_mut().for_each(|v| *v = rng.gauss() * 0.5);
                    for _ in 0..3 {
                        row[rng.below(256) as usize] = rng.gauss() * 40.0;
                    }
                }
                AMode::Zero => {}
                AMode::HalfZero => {
                    if t != 0 {
                        row.iter_mut().for_each(|v| *v = rng.gauss());
                    }
                }
                AMode::Ties => {
                    // a = 255 + 2m, b = 253 - 2m at positions 0/1: rotated values are
                    // +-(a + b)/16 = +-31.75 and +-(a - b)/16 = +-(1 + 2m)/8, d = 0.25, so the
                    // small ones quantize at exactly m + 0.5.
                    let m = ((t * 7 + g * 5 + (rng.below(8) as usize)) % 63) as f32;
                    row[0] = 255.0 + 2.0 * m;
                    row[1] = 253.0 - 2.0 * m;
                    if rng.below(2) == 1 {
                        row[1] = -row[1];
                    }
                    if rng.below(2) == 1 {
                        row[0] = -row[0];
                    }
                }
                AMode::Spike => {
                    let v = 0.25 + rng.gauss().abs();
                    row[0] = if rng.below(2) == 1 { v } else { -v };
                }
                AMode::Tiny => row.iter_mut().for_each(|v| *v = rng.gauss() * 1e-30),
                AMode::DenormSd => row.iter_mut().for_each(|v| *v = rng.gauss() * 1e-35),
            }
        }
    }
    x
}

// ---------------------------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------------------------

struct QBlock {
    d: f32,
    s: i32,
    q: [i32; 128],
    ties: usize,
}

/// The shared quantizer contract (block_i8_128_quant.hip header) on 128 contiguous values.
fn quant_block(x: &[f32]) -> QBlock {
    let mut amax = 0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let mut b = QBlock {
        d: 0.0,
        s: 0,
        q: [0; 128],
        ties: 0,
    };
    if amax == 0.0 {
        return b;
    }
    b.d = amax / 127.0;
    for (i, &v) in x.iter().enumerate() {
        let r = v / b.d;
        if (r - r.trunc()).abs() == 0.5 {
            b.ties += 1;
        }
        let q = r.round_ties_even().clamp(-127.0, 127.0) as i32;
        b.q[i] = q;
        b.s += q;
    }
    b
}

/// Documented qs byte order: qs[8g + i] = q[8g + 2i], qs[8g + 4 + i] = q[8g + 2i + 1].
fn qs_from_natural(q: &[i32; 128]) -> [u8; 128] {
    let mut qs = [0u8; 128];
    for g in 0..16 {
        for i in 0..4 {
            qs[8 * g + i] = q[8 * g + 2 * i] as i8 as u8;
            qs[8 * g + 4 + i] = q[8 * g + 2 * i + 1] as i8 as u8;
        }
    }
    qs
}

fn natural_from_qs(qs: &[u8]) -> [i8; 128] {
    let mut q = [0i8; 128];
    for g in 0..16 {
        for i in 0..4 {
            q[8 * g + 2 * i] = qs[8 * g + i] as i8;
            q[8 * g + 2 * i + 1] = qs[8 * g + 4 + i] as i8;
        }
    }
    q
}

fn bf16_bits(v: f32) -> u16 {
    let u = (0.0f32 + v).to_bits();
    let r = if v.is_finite() {
        (u.wrapping_add(0x7FFF).wrapping_add((u >> 16) & 1)) >> 16
    } else {
        u >> 16
    };
    r as u16
}

struct Ref {
    /// [n][m]
    y: Vec<f32>,
    /// [K/128][n][m]
    c: Vec<i32>,
}

/// `blocks` = the producer's bytes, `[K/128][n]` blocks of 136 B.
fn cpu_ref(w: &[u8], m: usize, k: usize, n: usize, blocks: &[u8]) -> Ref {
    let nb = k / 128;
    let g_per_row = k / 256;
    let row_bytes = g_per_row * GROUP_BYTES;
    let mut dd = vec![0f32; nb * n];
    let mut qx = vec![0i8; n * k];
    for b in 0..nb {
        for t in 0..n {
            let blk = &blocks[(b * n + t) * BLOCK_BYTES..(b * n + t + 1) * BLOCK_BYTES];
            dd[b * n + t] = f32::from_le_bytes(blk[0..4].try_into().unwrap());
            qx[t * k + b * 128..t * k + (b + 1) * 128]
                .copy_from_slice(&natural_from_qs(&blk[8..136]));
        }
    }
    // per row: y[n], c[nb][n]; rows are independent -> chunk by 64 rows.
    let per = n + nb * n;
    let mut flat = vec![0u32; m * per];
    par_chunks_mut(&mut flat, 64 * per, |ci, chunk| {
        let rows = chunk.len() / per;
        let mut wq = vec![0i32; k];
        for r in 0..rows {
            let row = ci * 64 + r;
            let rb = &w[row * row_bytes..(row + 1) * row_bytes];
            let mut hs = vec![0f32; 2 * g_per_row];
            for g in 0..g_per_row {
                let grp = &rb[g * GROUP_BYTES..(g + 1) * GROUP_BYTES];
                let codes = decode_codes(grp);
                for p in 0..256 {
                    wq[g * 256 + p] = codes[p] as i32 - 32;
                }
                hs[2 * g] = f16_to_f32(u16::from_le_bytes([grp[0], grp[1]]));
                hs[2 * g + 1] = f16_to_f32(u16::from_le_bytes([grp[4], grp[5]]));
            }
            let out = &mut chunk[r * per..(r + 1) * per];
            for t in 0..n {
                let mut acc = 0.0f32;
                for b in 0..nb {
                    let mut c = 0i32;
                    let wv = &wq[b * 128..(b + 1) * 128];
                    let xv = &qx[t * k + b * 128..t * k + (b + 1) * 128];
                    for i in 0..128 {
                        c += wv[i] * xv[i] as i32;
                    }
                    let sd = hs[b] * dd[b * n + t];
                    acc = sd.mul_add(c as f32, acc);
                    out[n + b * n + t] = c as u32;
                }
                out[t] = acc.to_bits();
            }
        }
    });
    let mut y = vec![0f32; n * m];
    let mut c = vec![0i32; nb * n * m];
    for row in 0..m {
        let o = &flat[row * per..(row + 1) * per];
        for t in 0..n {
            y[t * m + row] = f32::from_bits(o[t]);
            for b in 0..nb {
                c[(b * n + t) * m + row] = o[n + b * n + t] as i32;
            }
        }
    }
    Ref { y, c }
}

/// Quality diagnostic: f64 dequant (`s_h * (q - 32)`) x the producer's rotated F32 activation on
/// a sample of rows. Returns (max |y - ref| / rms(ref), max rel on |ref| > 1e-2 rms, rms(ref)).
fn quality_diag(
    w: &[u8],
    m: usize,
    k: usize,
    n: usize,
    xrot: &[f32],
    y: &[f32],
) -> (f64, f64, f64) {
    let g_per_row = k / 256;
    let row_bytes = g_per_row * GROUP_BYTES;
    let stride = (m / 128).max(1);
    let mut refs = Vec::new();
    let mut got = Vec::new();
    let mut row = 0;
    while row < m {
        let rb = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut wf = vec![0f64; k];
        for g in 0..g_per_row {
            let grp = &rb[g * GROUP_BYTES..(g + 1) * GROUP_BYTES];
            let codes = decode_codes(grp);
            let s0 = f16_to_f32(u16::from_le_bytes([grp[0], grp[1]])) as f64;
            let s1 = f16_to_f32(u16::from_le_bytes([grp[4], grp[5]])) as f64;
            for p in 0..256 {
                wf[g * 256 + p] = if p < 128 { s0 } else { s1 } * (codes[p] as f64 - 32.0);
            }
        }
        for t in 0..n {
            let mut a = 0f64;
            for i in 0..k {
                a += wf[i] * xrot[t * k + i] as f64;
            }
            refs.push(a);
            got.push(y[t * m + row] as f64);
        }
        row += stride;
    }
    let rms = (refs.iter().map(|v| v * v).sum::<f64>() / refs.len().max(1) as f64).sqrt();
    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    for (r, g) in refs.iter().zip(&got) {
        let e = (r - g).abs();
        max_abs = max_abs.max(e);
        if r.abs() > 1e-2 * rms {
            max_rel = max_rel.max(e / r.abs());
        }
    }
    (if rms > 0.0 { max_abs / rms } else { 0.0 }, max_rel, rms)
}

// ---------------------------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------------------------

fn gpu_gate() -> Option<Gpu> {
    let gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: requires exact gfx1201.");
        return None;
    }
    Some(gpu)
}

fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc raw");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod");
    t
}

fn download(gpu: &Gpu, ptr: *mut c_void, bytes: usize) -> Vec<u8> {
    gpu.hip.device_synchronize().expect("sync");
    let view = unsafe { DeviceBuffer::from_raw(ptr, bytes) };
    let mut out = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut out, &view).expect("dtoh");
    out
}

fn blob(
    w: *mut c_void,
    xq: *mut c_void,
    y: *mut c_void,
    m: usize,
    k: usize,
    n: usize,
) -> KernargBlob {
    let mut b = KernargBlob::new();
    b.push_ptr(w);
    b.push_ptr(xq);
    b.push_ptr(y);
    b.push_i32(m as i32);
    b.push_i32(k as i32);
    b.push_i32(n as i32);
    assert_eq!(b.len(), 36, "frozen S6I8 ABI: 36 explicit kernarg bytes");
    b.pad_to(16);
    assert_eq!(
        b.len(),
        48,
        "36 explicit bytes padded to a 16-byte boundary"
    );
    b
}

fn launch_sym(gpu: &Gpu, name: &str, b: &mut KernargBlob, m: usize, k: usize, n: usize) {
    gpu.launch_kernel_blob(
        name,
        [m.div_ceil(32) as u32, 1, 1],
        [128, 1, 1],
        lds_bytes(k, n),
        b.as_mut_slice(),
    )
    .unwrap_or_else(|e| panic!("launch {name}: {e}"));
}

/// Real producer: FWHT256 rotate + A8 blocks (`[K/128][n]`). Returns (xq device ptr, blocks bytes,
/// rotated f32 as the producer stored it).
fn produce(gpu: &mut Gpu, x: &[f32], k: usize, n: usize) -> (*mut c_void, Vec<u8>, Vec<f32>) {
    let xt = gpu.upload_f32(x, &[n * k]).expect("x");
    let xrot = gpu.zeros(&[n * k], DType::F32).expect("xrot");
    let res = gpu.reserve_int8_mmq(k, n).expect("reserve");
    let prepared = gpu
        .rotate_x_mq_i8_batched(&xt, Some(&xrot), res, k, n)
        .expect("producer");
    let ptr = gpu
        .int8_mmq_prepared_ptr(&prepared, k, n)
        .expect("prepared ptr");
    let blocks = download(gpu, ptr, (k / 128) * n * BLOCK_BYTES);
    let rot = gpu.download_f32(&xrot).expect("xrot dl");
    (ptr, blocks, rot)
}

// ---------------------------------------------------------------------------------------------
// Correctness
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct Tally {
    cases: usize,
    pass: usize,
    fail: usize,
    words: u64,
    quant_blocks: u64,
    ties_seen: u64,
    c_words: u64,
    bf16_words: u64,
}

struct WSpec {
    name: String,
    m: usize,
    k: usize,
    cm: CMode,
    sm: SMode,
    seed: u64,
}

const GUARD_WORDS: usize = 64;
const POISON: u32 = 0xDEAD_BEEF;

#[allow(clippy::too_many_arguments)]
fn run_acts(
    gpu: &mut Gpu,
    ws: &WSpec,
    w_host: &[u8],
    w_dev: &GpuTensor,
    acts: &[(usize, AMode)],
    quality: bool,
    tally: &mut Tally,
    report: &mut String,
) {
    let (m, k) = (ws.m, ws.k);
    for &(n, am) in acts {
        let name = format!("{} {}x{} N{} {:?}", ws.name, m, k, n, am);
        let x = gen_act(k, n, am, 0xAC70 ^ (n as u64) ^ ((m as u64) << 8));
        let (xq, blocks, xrot) = produce(gpu, &x, k, n);

        // 1. quant blocks vs the CPU quantizer on the producer's own rotated output.
        let mut quant_bad = 0u64;
        let mut ties = 0u64;
        let nb = k / 128;
        for b in 0..nb {
            for t in 0..n {
                let qb = quant_block(&xrot[t * k + b * 128..t * k + (b + 1) * 128]);
                ties += qb.ties as u64;
                let blk = &blocks[(b * n + t) * BLOCK_BYTES..(b * n + t + 1) * BLOCK_BYTES];
                let ok = blk[0..4] == qb.d.to_bits().to_le_bytes()
                    && blk[4..8] == qb.s.to_le_bytes()
                    && blk[8..136] == qs_from_natural(&qb.q);
                if !ok {
                    quant_bad += 1;
                }
            }
        }
        tally.quant_blocks += (nb * n) as u64;
        tally.ties_seen += ties;

        // 2. reference + kernel words.
        let rf = cpu_ref(w_host, m, k, n, &blocks);
        let nonzero = rf.y.iter().filter(|v| v.to_bits() != 0).count();

        // F32 production entry with guards + poison.
        let y_words = n * m;
        let total = GUARD_WORDS + y_words + GUARD_WORDS;
        let ybuf = gpu.alloc_tensor(&[total], DType::F32).expect("y");
        let poison_bytes: Vec<u8> = (0..total).flat_map(|_| POISON.to_le_bytes()).collect();
        gpu.hip
            .memcpy_htod(&ybuf.buf, &poison_bytes)
            .expect("poison");
        let yp = unsafe { (ybuf.buf.as_ptr() as *mut u8).add(GUARD_WORDS * 4) as *mut c_void };
        let mut b = blob(w_dev.buf.as_ptr(), xq, yp, m, k, n);
        launch_sym(gpu, &entry(n, false), &mut b, m, k, n);
        let yh = download(gpu, ybuf.buf.as_ptr(), total * 4);
        let words: Vec<u32> = yh
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let guards_ok = words[..GUARD_WORDS].iter().all(|&v| v == POISON)
            && words[GUARD_WORDS + y_words..].iter().all(|&v| v == POISON);
        let mut y_bad = 0u64;
        let mut first = None;
        let mut ygpu = vec![0f32; y_words];
        for i in 0..y_words {
            let g = words[GUARD_WORDS + i];
            ygpu[i] = f32::from_bits(g);
            if g != rf.y[i].to_bits() {
                y_bad += 1;
                first.get_or_insert((i / m, i % m, g, rf.y[i].to_bits()));
            }
        }
        tally.words += y_words as u64;

        // BF16 production entry.
        let bt = gpu
            .alloc_tensor(&[(GUARD_WORDS * 2 + y_words) / 2 + 1], DType::F32)
            .expect("ybf");
        let bwords = GUARD_WORDS * 2 + y_words;
        let bpoison: Vec<u8> = (0..bt.byte_size() / 2)
            .flat_map(|_| 0xDEADu16.to_le_bytes())
            .collect();
        gpu.hip.memcpy_htod(&bt.buf, &bpoison).expect("poison bf16");
        let bp = unsafe { (bt.buf.as_ptr() as *mut u8).add(GUARD_WORDS * 2) as *mut c_void };
        let mut b2 = blob(w_dev.buf.as_ptr(), xq, bp, m, k, n);
        launch_sym(gpu, &entry(n, true), &mut b2, m, k, n);
        let bh = download(gpu, bt.buf.as_ptr(), bt.byte_size());
        let bw: Vec<u16> = bh
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let bguards_ok = bw[..GUARD_WORDS].iter().all(|&v| v == 0xDEAD)
            && bw[GUARD_WORDS + y_words..bwords]
                .iter()
                .all(|&v| v == 0xDEAD);
        let mut bf_bad = 0u64;
        for i in 0..y_words {
            if bw[GUARD_WORDS + i] != bf16_bits(rf.y[i]) {
                bf_bad += 1;
            }
        }
        tally.bf16_words += y_words as u64;

        // Debug-C build: per-block integer C.
        let c_words = nb * n * m;
        let dbg_total = GUARD_WORDS + y_words + c_words + GUARD_WORDS;
        let dbuf = gpu.alloc_tensor(&[dbg_total], DType::F32).expect("dbg");
        let dpoison: Vec<u8> = (0..dbg_total).flat_map(|_| POISON.to_le_bytes()).collect();
        gpu.hip
            .memcpy_htod(&dbuf.buf, &dpoison)
            .expect("poison dbg");
        let dp = unsafe { (dbuf.buf.as_ptr() as *mut u8).add(GUARD_WORDS * 4) as *mut c_void };
        let mut b3 = blob(w_dev.buf.as_ptr(), xq, dp, m, k, n);
        launch_sym(gpu, &entry_dbg(n), &mut b3, m, k, n);
        let dh = download(gpu, dbuf.buf.as_ptr(), dbg_total * 4);
        let dw: Vec<u32> = dh
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let dguards_ok = dw[..GUARD_WORDS].iter().all(|&v| v == POISON)
            && dw[GUARD_WORDS + y_words + c_words..]
                .iter()
                .all(|&v| v == POISON);
        let mut c_bad = 0u64;
        let mut dy_bad = 0u64;
        for i in 0..y_words {
            if dw[GUARD_WORDS + i] != rf.y[i].to_bits() {
                dy_bad += 1;
            }
        }
        for i in 0..c_words {
            if dw[GUARD_WORDS + y_words + i] as i32 != rf.c[i] {
                c_bad += 1;
            }
        }
        tally.c_words += c_words as u64;

        let degenerate = matches!(am, AMode::Zero)
            || (am == AMode::HalfZero && n == 1)
            || ws.sm == SMode::Zero
            || ws.cm == CMode::Zero32;
        let alive = degenerate || nonzero > 0;
        let ok = quant_bad == 0
            && y_bad == 0
            && bf_bad == 0
            && c_bad == 0
            && dy_bad == 0
            && guards_ok
            && bguards_ok
            && dguards_ok
            && alive;
        tally.cases += 1;
        if ok {
            tally.pass += 1;
        } else {
            tally.fail += 1;
        }
        let q = if quality && am == AMode::Gauss {
            let (a, r, rms) = quality_diag(w_host, m, k, n, &xrot, &ygpu);
            format!(" quality(max|err|/rms={a:.3e} max_rel={r:.3e} rms={rms:.3e})")
        } else {
            String::new()
        };
        let line = format!(
            "{} {name}: quant_blocks_bad={quant_bad}/{} ties={ties} y_bad={y_bad}/{y_words} bf16_bad={bf_bad} c_bad={c_bad}/{c_words} dbgbuild_y_bad={dy_bad} guards={}{}{} nonzero_ref={nonzero}{q}{}",
            if ok { "PASS" } else { "FAIL" },
            nb * n,
            guards_ok,
            bguards_ok,
            dguards_ok,
            first.map(|f| format!(" first_bad(token={},row={},gpu={:#010x},ref={:#010x})", f.0, f.1, f.2, f.3)).unwrap_or_default(),
        );
        println!("{line}");
        writeln!(report, "{line}").ok();
        drop(ybuf);
        drop(bt);
        drop(dbuf);
    }
}

fn run_weights(
    gpu: &mut Gpu,
    ws: WSpec,
    acts: &[(usize, AMode)],
    quality: bool,
    tally: &mut Tally,
    report: &mut String,
) {
    let w = gen_weights(ws.m, ws.k, ws.cm, ws.sm, ws.seed);
    let wd = upload_u8(gpu, &w);
    run_acts(gpu, &ws, &w, &wd, acts, quality, tally, report);
}

#[test]
fn bit_level_decoder_matches_triple_pack() {
    // Pure CPU: the independent bit-level decoder inverts the 4-codes-per-3-bytes encoder.
    let mut rng = Rng(7);
    let mut g = [0u8; 200];
    for _ in 0..64 {
        let mut codes = [0u8; 256];
        codes.iter_mut().for_each(|c| *c = rng.below(64) as u8);
        encode_group([0x3C00, 0x0001], &codes, &mut g);
        assert_eq!(decode_codes(&g), codes);
    }
    // nibble-pair order round trip
    let mut q = [0i32; 128];
    q.iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = (i as i32) - 64);
    let qs = qs_from_natural(&q);
    let back = natural_from_qs(&qs);
    for i in 0..128 {
        assert_eq!(back[i] as i32, q[i]);
    }
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn mq6sym_i8_decode_ignores_z_hw() {
    // Robustness: random garbage in the qt47 z halves must give byte-identical outputs to z = -32 s.
    let Some(mut gpu) = gpu_gate() else { return };
    let ksrc = kernel_src();
    let mut cases = 0usize;
    let mut bad_words = 0u64;
    let mut words = 0u64;
    for n in 1..=8 {
        for bf in [false, true] {
            gpu.ensure_kernel_public(MODULE, &ksrc, &entry(n, bf))
                .expect("load kernel");
        }
    }
    for &(m, k) in &[(48usize, 512usize), (640, 2560), (2560, 6144)] {
        let w = gen_weights(m, k, CMode::Random, SMode::Normal, 0x2_0000 ^ m as u64);
        let mut wg = w.clone();
        let mut rng = Rng(0x6A2B_A6E0 ^ k as u64);
        for g in wg.chunks_mut(GROUP_BYTES) {
            for i in [2usize, 3, 6, 7] {
                g[i] = rng.next() as u8;
            }
        }
        assert_ne!(w, wg);
        let (wd, wgd) = (upload_u8(&mut gpu, &w), upload_u8(&mut gpu, &wg));
        for n in 1..=8usize {
            let x = gen_act(k, n, AMode::Gauss, 0xAC70 ^ n as u64);
            let (xq, _, _) = produce(&mut gpu, &x, k, n);
            for bf in [false, true] {
                let ybytes = n * m * if bf { 2 } else { 4 };
                let mut outs: Vec<Vec<u8>> = Vec::new();
                for wt in [&wd, &wgd] {
                    let yb = gpu
                        .alloc_tensor(&[ybytes.div_ceil(4)], DType::F32)
                        .expect("y");
                    gpu.hip
                        .memcpy_htod(&yb.buf, &vec![0xA5u8; yb.byte_size()])
                        .expect("poison");
                    let mut b = blob(wt.buf.as_ptr(), xq, yb.buf.as_ptr(), m, k, n);
                    launch_sym(&gpu, &entry(n, bf), &mut b, m, k, n);
                    outs.push(download(&gpu, yb.buf.as_ptr(), ybytes));
                }
                let nz = outs[0].iter().filter(|&&v| v != 0 && v != 0xA5).count();
                assert!(nz > 0, "all-zero/poison output m={m} k={k} n={n}");
                bad_words += outs[0].iter().zip(&outs[1]).filter(|(a, b)| a != b).count() as u64;
                words += ybytes as u64;
                cases += 1;
            }
        }
    }
    println!("Z_GARBAGE cases={cases} bytes={words} mismatching_bytes={bad_words}");
    assert_eq!(bad_words, 0, "kernel output depends on z bytes");
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn mq6sym_i8_decode_matches_cpu_bytes_hw() {
    let Some(mut gpu) = gpu_gate() else { return };
    let ksrc = kernel_src();
    let dbg_src = format!("#define S6_DEBUG_C 1\n{ksrc}");
    for n in 1..=8 {
        for bf in [false, true] {
            gpu.ensure_kernel_public(MODULE, &ksrc, &entry(n, bf))
                .expect("load kernel");
        }
        gpu.ensure_kernel_public(MODULE_DBG, &dbg_src, &entry_dbg(n))
            .expect("load debug-C kernel");
    }

    let mut tally = Tally::default();
    let mut report = String::new();
    let all = std::env::var("MQ6SYM_SKIP_LM").is_err();
    let ns_all: Vec<(usize, AMode)> = (1..=8).map(|n| (n, AMode::Gauss)).collect();

    // A. grid: every K x every M at N=8 (+N=1 on the first K).
    let ks = [256usize, 512, 2560, 6144];
    let ms = [48usize, 512, 640, 2560, 6144, 10240, 12288];
    for &k in &ks {
        for &m in &ms {
            run_weights(
                &mut gpu,
                WSpec {
                    name: "grid".into(),
                    m,
                    k,
                    cm: CMode::Random,
                    sm: SMode::Mixed,
                    seed: 0x6000 ^ (m * 7 + k) as u64,
                },
                &[(8, AMode::Outliers)],
                k == 2560,
                &mut tally,
                &mut report,
            );
        }
    }
    // B. N sweep 1..8 on small and mid matrices (normal scales: quality diagnostic on).
    for (m, k) in [(640usize, 2560usize), (48, 256), (2560, 6144)] {
        run_weights(
            &mut gpu,
            WSpec {
                name: "nsweep".into(),
                m,
                k,
                cm: CMode::Random,
                sm: SMode::Normal,
                seed: 0x7000 ^ m as u64,
            },
            &ns_all,
            true,
            &mut tally,
            &mut report,
        );
    }
    // C. tails / tiny M.
    for m in [1usize, 17, 31, 33, 100, 2563] {
        run_weights(
            &mut gpu,
            WSpec {
                name: "tail".into(),
                m,
                k: 512,
                cm: CMode::Random,
                sm: SMode::Mixed,
                seed: 0x8000 ^ m as u64,
            },
            &[(1, AMode::Gauss), (3, AMode::Gauss), (8, AMode::Gauss)],
            false,
            &mut tally,
            &mut report,
        );
    }
    // D. weight classes (N 1/2/4/8).
    let n4: Vec<(usize, AMode)> = [1usize, 2, 4, 8]
        .iter()
        .map(|&n| (n, AMode::Gauss))
        .collect();
    for (name, cm, sm) in [
        ("exhaustive", CMode::Exhaustive, SMode::Normal),
        ("exhaustive_mixed", CMode::Exhaustive, SMode::Mixed),
        ("all_low", CMode::AllLow, SMode::Normal),
        ("all_high", CMode::AllHigh, SMode::Normal),
        ("alternating", CMode::Alternating, SMode::Normal),
        ("zero_codes32", CMode::Zero32, SMode::Normal),
        ("scale_zero", CMode::Random, SMode::Zero),
        ("scale_subnormal", CMode::Random, SMode::Subnormal),
        ("scale_large", CMode::Random, SMode::Large),
    ] {
        for (m, k) in [(256usize, 512usize), (640, 2560)] {
            run_weights(
                &mut gpu,
                WSpec {
                    name: name.into(),
                    m,
                    k,
                    cm,
                    sm,
                    seed: 0x9000 ^ (m + k) as u64,
                },
                &n4,
                false,
                &mut tally,
                &mut report,
            );
        }
    }
    // E. activation classes through the real producer (normal scales / mixed scales).
    for am in [
        AMode::Zero,
        AMode::HalfZero,
        AMode::Ties,
        AMode::Spike,
        AMode::Tiny,
        AMode::DenormSd,
        AMode::Outliers,
    ] {
        let acts: Vec<(usize, AMode)> = [1usize, 2, 4, 8].iter().map(|&n| (n, am)).collect();
        for sm in [SMode::Normal, SMode::Subnormal] {
            run_weights(
                &mut gpu,
                WSpec {
                    name: format!("act_{sm:?}"),
                    m: 512,
                    k: 2560,
                    cm: CMode::Random,
                    sm,
                    seed: 0xA000,
                },
                &acts,
                false,
                &mut tally,
                &mut report,
            );
        }
    }
    // F. lm_head.
    if all {
        run_weights(
            &mut gpu,
            WSpec {
                name: "lm_head".into(),
                m: 248320,
                k: 2560,
                cm: CMode::Random,
                sm: SMode::Normal,
                seed: 0xB000,
            },
            &[(8, AMode::Outliers), (1, AMode::Gauss)],
            false,
            &mut tally,
            &mut report,
        );
    }

    let summary = format!(
        "SUMMARY cases={} pass={} fail={} y_words={} bf16_words={} c_words={} quant_blocks={} ties_seen={}",
        tally.cases, tally.pass, tally.fail, tally.words, tally.bf16_words, tally.c_words, tally.quant_blocks, tally.ties_seen
    );
    println!("{summary}");
    if let Ok(dir) = std::env::var("MQ6SYM_OUT") {
        std::fs::write(
            format!("{dir}/correctness.txt"),
            format!("{report}{summary}\n"),
        )
        .ok();
    }
    assert!(
        tally.ties_seen > 0,
        "tie-designed activations never produced an exact .5 quantizer tie"
    );
    assert_eq!(tally.fail, 0, "{} case(s) failed", tally.fail);
}

// ---------------------------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------------------------

/// Same-job streaming roofline: pure b128 read of the same bytes (4 independent loads / thread).
const ROOF_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ __launch_bounds__(256) void s6_stream_b128(
    const uint4* __restrict__ p, unsigned long long n16, unsigned* __restrict__ sink) {
    unsigned long long i = (unsigned long long)blockIdx.x * 1024ull + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * 1024ull;
    unsigned a = 0;
    for (; i < n16; i += stride) {
        const unsigned long long i1 = i + 256ull, i2 = i + 512ull, i3 = i + 768ull;
        const uint4 v0 = p[i];
        const uint4 v1 = i1 < n16 ? p[i1] : make_uint4(0, 0, 0, 0);
        const uint4 v2 = i2 < n16 ? p[i2] : make_uint4(0, 0, 0, 0);
        const uint4 v3 = i3 < n16 ? p[i3] : make_uint4(0, 0, 0, 0);
        a ^= v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w ^ v2.x ^ v2.y ^ v2.z ^ v2.w ^ v3.x ^ v3.y ^ v3.z ^ v3.w;
    }
    if (a == 0x9E3779B9u) sink[0] = a;
}
"#;
const ROOF_MODULE: &str = "mq6sym_i8_t0_roof";
const ROOF: &str = "s6_stream_b128";
const MIB: usize = 1 << 20;
const COLD_TARGET: usize = 256 * MIB;

fn timed<F: FnOnce(&mut Gpu)>(gpu: &mut Gpu, f: F) -> f64 {
    let a = gpu.hip.event_create().expect("ev");
    let b = gpu.hip.event_create().expect("ev");
    gpu.hip.event_record(&a, None).expect("rec");
    f(gpu);
    gpu.hip.event_record(&b, None).expect("rec");
    gpu.hip.event_synchronize(&b).expect("sync");
    let ms = gpu.hip.event_elapsed_ms(&a, &b).expect("elapsed") as f64;
    gpu.hip.event_destroy(a).ok();
    gpu.hip.event_destroy(b).ok();
    ms
}

/// Mean us per call over a batch of back-to-back launches (cold sets: >= `min_iters` calls).
fn bench<F: FnMut(&mut Gpu, usize)>(gpu: &mut Gpu, min_iters: usize, mut f: F) -> f64 {
    for i in 0..8 {
        f(gpu, i);
    }
    gpu.hip.device_synchronize().expect("sync");
    let est_ms = timed(gpu, |g| {
        for i in 0..32 {
            f(g, i);
        }
    });
    let est_us = (est_ms * 1000.0 / 32.0).max(0.5);
    let iters = ((60_000.0 / est_us) as usize)
        .clamp(64, 8000)
        .max(min_iters)
        .min(10_000);
    let ms = timed(gpu, |g| {
        for i in 0..iters {
            f(g, i);
        }
    });
    ms * 1000.0 / iters as f64
}

/// Median single-launch event time in us (launch + event overhead included).
fn isolated<F: FnMut(&mut Gpu, usize)>(gpu: &mut Gpu, samples: usize, mut f: F) -> f64 {
    let mut v = Vec::with_capacity(samples);
    for i in 0..samples {
        let ms = timed(gpu, |g| f(g, i));
        v.push(ms * 1000.0);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn fill_random(gpu: &Gpu, t: &GpuTensor, host: &[u8]) {
    let total = t.byte_size();
    let base = t.buf.as_ptr() as *mut u8;
    let mut off = 0;
    while off < total {
        let len = host.len().min(total - off);
        let view = unsafe { DeviceBuffer::from_raw(base.add(off) as *mut c_void, len) };
        gpu.hip.memcpy_htod(&view, &host[..len]).expect("fill htod");
        off += len;
    }
}

struct Row {
    m: usize,
    k: usize,
    r: usize,
    sets: usize,
    sym_bytes: usize,
    old_bytes: usize,
    sym_us: Vec<f64>,
    roof_us: Vec<f64>,
    prod_us: Vec<f64>,
    symprod_us: Vec<f64>,
    old_rot_us: Vec<f64>,
    old_k_us: Vec<f64>,
    old_tot_us: Vec<f64>,
    iso_sym_us: Option<f64>,
    iso_roof_us: Option<f64>,
}

fn gbps(bytes: usize, us: f64) -> f64 {
    bytes as f64 / (us * 1e-6) / 1e9
}

fn rows_json(rows: &[Row]) -> String {
    let arr = |v: &[f64]| {
        format!(
            "[{}]",
            v.iter()
                .map(|x| format!("{x:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    let mut s = String::from("[\n");
    for (i, r) in rows.iter().enumerate() {
        write!(
            s,
            "  {{\"m\":{},\"k\":{},\"rows\":{},\"sets\":{},\"sym_bytes\":{},\"old_bytes\":{},\"sym_us\":{},\"roof_us\":{},\"producer_us\":{},\"sym_plus_producer_us\":{},\"old_rotate_us\":{},\"old_kernel_us\":{},\"old_total_us\":{},\"iso_sym_us\":{},\"iso_roof_us\":{}}}{}\n",
            r.m, r.k, r.r, r.sets, r.sym_bytes, r.old_bytes,
            arr(&r.sym_us), arr(&r.roof_us), arr(&r.prod_us), arr(&r.symprod_us),
            arr(&r.old_rot_us), arr(&r.old_k_us), arr(&r.old_tot_us),
            r.iso_sym_us.map(|v| format!("{v:.3}")).unwrap_or("null".into()),
            r.iso_roof_us.map(|v| format!("{v:.3}")).unwrap_or("null".into()),
            if i + 1 < rows.len() { "," } else { "" }
        )
        .ok();
    }
    s.push(']');
    s
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn mq6sym_i8_decode_timing_hw() {
    let Some(mut gpu) = gpu_gate() else { return };
    for n in 1..=8 {
        gpu.ensure_kernel_public(MODULE, &kernel_src(), &entry(n, false))
            .expect("load kernel");
    }
    // A/B variants: MQ6SYM_VARIANTS="name=KEY:VAL+KEY:VAL;name2=..." (extra arms on the same
    // buffers, alternated with the baseline inside every rep).
    let variants: Vec<(String, usize)> = std::env::var("MQ6SYM_VARIANTS")
        .ok()
        .map(|v| {
            v.split(';')
                .filter(|s| !s.is_empty())
                .map(|spec| {
                    let (name, defs) = spec.split_once('=').unwrap_or((spec, ""));
                    let mut src = String::new();
                    let mut chunk = chunk_groups();
                    for (k, val) in kernel_defs() {
                        writeln!(src, "#define {k} {val}").ok();
                    }
                    for kv in defs.split('+').filter(|s| !s.is_empty()) {
                        let (k, val) = kv.split_once(':').unwrap_or((kv, "1"));
                        writeln!(src, "#define {k} {val}").ok();
                        if k == "S6_CHUNK_GROUPS" {
                            chunk = val.parse().unwrap();
                        }
                    }
                    writeln!(src, "#define S6_SUF _{name}").ok();
                    src.push_str(KERNEL_SRC);
                    for n in 1..=8 {
                        gpu.ensure_kernel_public(
                            &format!("{MODULE}_{name}"),
                            &src,
                            &format!("gemv_mq6sym_i8_rows_r{n}_{name}_gfx1201"),
                        )
                        .expect("load variant");
                    }
                    (name.to_string(), chunk)
                })
                .collect()
        })
        .unwrap_or_default();
    gpu.ensure_kernel_public(ROOF_MODULE, ROOF_SRC, ROOF)
        .expect("load roof");
    let reps: usize = std::env::var("MQ6SYM_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let skip_lm = std::env::var("MQ6SYM_SKIP_LM").is_ok();
    let only: Option<Vec<usize>> = std::env::var("MQ6SYM_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect());
    let mut rng = Rng(0x7131_0000);
    let host: Vec<u8> = (0..16 * MIB).map(|_| rng.next() as u8).collect();

    // (m, k, large streaming cell?)
    let mut cells: Vec<(usize, usize, bool)> = vec![
        (10240, 2560, true),
        (6144, 2560, true),
        (2560, 6144, true),
        (12288, 2560, true),
    ];
    if !skip_lm {
        cells.push((248320, 2560, true));
    }
    cells.extend([
        (48usize, 2560usize, false),
        (512, 2560, false),
        (640, 2560, false),
    ]);
    if let Ok(filter) = std::env::var("MQ6SYM_CELLS") {
        let keep: Vec<String> = filter.split(',').map(|s| s.to_string()).collect();
        cells.retain(|&(m, k, _)| keep.contains(&format!("{m}x{k}")));
    }
    let sym_only = std::env::var("MQ6SYM_SYMONLY").is_ok();

    let mut out_rows: Vec<Row> = Vec::new();
    println!("arch={} device={}", gpu.arch, gpu.device_id);
    for &(m, k, _large) in &cells {
        let g_per_row = k / 256;
        let sym_bytes = m * g_per_row * GROUP_BYTES;
        let old_bytes = m * g_per_row * 200;
        let sym_stride = sym_bytes.div_ceil(256) * 256;
        let old_stride = old_bytes.div_ceil(256) * 256;
        let sets = COLD_TARGET.div_ceil(sym_stride).clamp(6, 4096);
        let sym_buf = gpu
            .alloc_tensor(&[sym_stride * sets], DType::Raw)
            .expect("sym buf");
        let old_buf = gpu
            .alloc_tensor(&[old_stride * sets], DType::Raw)
            .expect("old buf");
        fill_random(&gpu, &sym_buf, &host);
        fill_random(&gpu, &old_buf, &host);
        let old_sets: Vec<GpuTensor> = (0..sets)
            .map(|i| old_buf.sub_offset(i * old_stride, old_bytes))
            .collect();
        let sink = gpu.zeros(&[64], DType::F32).expect("sink");
        let n16 = (sym_bytes / 16) as u64;
        let roof_grid = (n16.div_ceil(1024)).clamp(1, 8192) as u32;
        let tiny = m <= 640 && k == 2560 && sym_bytes < 4 * MIB;

        for &r in &[1usize, 2, 4, 8] {
            if only.as_ref().is_some_and(|o| !o.contains(&r)) {
                continue;
            }
            let mut xr = Rng(0x5000 + r as u64);
            let xh: Vec<f32> = (0..r * k).map(|_| xr.gauss()).collect();
            let x = gpu.upload_f32(&xh, &[r * k]).expect("x");
            let xrot = gpu.zeros(&[r * k], DType::F32).expect("xrot");
            let yold = gpu.zeros(&[r * m], DType::F32).expect("yold");
            let ysym = gpu.zeros(&[r * m], DType::F32).expect("ysym");
            // Prepared sidecar for the kernel-only arms.
            let res = gpu.reserve_int8_mmq(k, r).expect("reserve");
            let prepared = gpu
                .rotate_x_mq_i8_batched(&x, None, res, k, r)
                .expect("producer");
            let xq = gpu.int8_mmq_prepared_ptr(&prepared, k, r).expect("xq ptr");
            let name = entry(r, false);
            let mut var_blobs: Vec<Vec<KernargBlob>> = variants
                .iter()
                .map(|_| {
                    (0..sets)
                        .map(|s| {
                            let w = unsafe {
                                (sym_buf.buf.as_ptr() as *mut u8).add(s * sym_stride) as *mut c_void
                            };
                            blob(w, xq, ysym.buf.as_ptr(), m, k, r)
                        })
                        .collect()
                })
                .collect();
            let mut var_us: Vec<Vec<f64>> = variants.iter().map(|_| Vec::new()).collect();
            let mut sym_blobs: Vec<KernargBlob> = (0..sets)
                .map(|s| {
                    let w = unsafe {
                        (sym_buf.buf.as_ptr() as *mut u8).add(s * sym_stride) as *mut c_void
                    };
                    blob(w, xq, ysym.buf.as_ptr(), m, k, r)
                })
                .collect();
            let mut roof_blobs: Vec<KernargBlob> = (0..sets)
                .map(|s| {
                    let mut b = KernargBlob::new();
                    b.push_ptr(unsafe {
                        (sym_buf.buf.as_ptr() as *mut u8).add(s * sym_stride) as *mut c_void
                    });
                    b.push_u64(n16);
                    b.push_ptr(sink.buf.as_ptr());
                    b.pad_to(16);
                    b
                })
                .collect();
            let mut row = Row {
                m,
                k,
                r,
                sets,
                sym_bytes,
                old_bytes,
                sym_us: vec![],
                roof_us: vec![],
                prod_us: vec![],
                symprod_us: vec![],
                old_rot_us: vec![],
                old_k_us: vec![],
                old_tot_us: vec![],
                iso_sym_us: None,
                iso_roof_us: None,
            };
            let old_kernel = |g: &mut Gpu, i: usize, xrot: &GpuTensor, yold: &GpuTensor| {
                let a = &old_sets[i % sets];
                if r == 1 {
                    g.gemv_mq6g256v2(a, xrot, yold, m, k).expect("old gemv");
                } else {
                    g.gemm_mq6g256v2_f32_rows(a, xrot, yold, m, k, r)
                        .expect("old rows");
                }
            };
            for _rep in 0..reps {
                // Alternate candidate / baseline / roofline inside each rep.
                let us = bench(&mut gpu, sets, |g, i| {
                    launch_sym(g, &name, &mut sym_blobs[i % sets], m, k, r)
                });
                row.sym_us.push(us);
                for (vi, (vname, _vchunk)) in variants.iter().enumerate() {
                    let vn = format!("gemv_mq6sym_i8_rows_r{r}_{vname}_gfx1201");
                    let lds = lds_bytes(k, r);
                    let us = bench(&mut gpu, sets, |g, i| {
                        g.launch_kernel_blob(
                            &vn,
                            [m.div_ceil(32) as u32, 1, 1],
                            [128, 1, 1],
                            lds,
                            var_blobs[vi][i % sets].as_mut_slice(),
                        )
                        .expect("variant launch")
                    });
                    var_us[vi].push(us);
                }
                if !sym_only {
                    let us = bench(&mut gpu, sets, |g, i| old_kernel(g, i, &xrot, &yold));
                    row.old_k_us.push(us);
                }
                let us = bench(&mut gpu, sets, |g, i| {
                    g.launch_kernel_blob(
                        ROOF,
                        [roof_grid, 1, 1],
                        [256, 1, 1],
                        0,
                        roof_blobs[i % sets].as_mut_slice(),
                    )
                    .expect("roof")
                });
                row.roof_us.push(us);
                if !sym_only {
                    let us = bench(&mut gpu, 64, |g, _| {
                        let res = g.reserve_int8_mmq(k, r).expect("reserve");
                        g.rotate_x_mq_i8_batched(&x, None, res, k, r)
                            .expect("producer");
                    });
                    row.prod_us.push(us);
                    let us = bench(&mut gpu, sets, |g, i| {
                        let res = g.reserve_int8_mmq(k, r).expect("reserve");
                        g.rotate_x_mq_i8_batched(&x, None, res, k, r)
                            .expect("producer");
                        launch_sym(g, &name, &mut sym_blobs[i % sets], m, k, r);
                    });
                    row.symprod_us.push(us);
                    let us = bench(&mut gpu, 64, |g, _| {
                        g.rotate_x_mq_batched(&x, &xrot, k, r).expect("old rotate")
                    });
                    row.old_rot_us.push(us);
                    let us = bench(&mut gpu, sets, |g, i| {
                        g.rotate_x_mq_batched(&x, &xrot, k, r).expect("old rotate");
                        old_kernel(g, i, &xrot, &yold);
                    });
                    row.old_tot_us.push(us);
                }
            }
            if tiny {
                row.iso_sym_us = Some(isolated(&mut gpu, 200, |g, i| {
                    launch_sym(g, &name, &mut sym_blobs[i % sets], m, k, r)
                }));
                row.iso_roof_us = Some(isolated(&mut gpu, 200, |g, i| {
                    g.launch_kernel_blob(
                        ROOF,
                        [roof_grid, 1, 1],
                        [256, 1, 1],
                        0,
                        roof_blobs[i % sets].as_mut_slice(),
                    )
                    .expect("roof")
                }));
            }
            let (s, rf, p, sp) = (
                median(&row.sym_us),
                median(&row.roof_us),
                median(&row.prod_us),
                median(&row.symprod_us),
            );
            let (ok_, ort, ot) = (
                median(&row.old_k_us),
                median(&row.old_rot_us),
                median(&row.old_tot_us),
            );
            println!(
                "CELL {m}x{k} rows={r} sets={sets} | sym_kernel {s:8.2} us {:6.1} GB/s | roofline {rf:8.2} us {:6.1} GB/s | producer {p:7.2} us | sym+prod {sp:8.2} us {:6.1} GB/s | old_kernel {ok_:8.2} us ({:6.1} GB/s on 200B) | old_rotate {ort:7.2} us | old_total {ot:8.2} us | speedup kernel {:.2}x total {:.2}x{}",
                gbps(sym_bytes, s), gbps(sym_bytes, rf), gbps(sym_bytes, sp), gbps(old_bytes, ok_), ok_ / s, ot / sp,
                match (row.iso_sym_us, row.iso_roof_us) {
                    (Some(a), Some(b)) => format!(" | isolated(event-incl) sym {a:.2} us roof {b:.2} us"),
                    _ => String::new(),
                }
            );
            println!(
                "  reps sym_us={:?} roof_us={:?} prod_us={:?} symprod_us={:?} old_k_us={:?} old_rot_us={:?} old_tot_us={:?}",
                row.sym_us, row.roof_us, row.prod_us, row.symprod_us, row.old_k_us, row.old_rot_us, row.old_tot_us
            );
            for (vi, (vname, _)) in variants.iter().enumerate() {
                let v = median(&var_us[vi]);
                println!(
                    "  VARIANT {vname} {m}x{k} rows={r}: {v:8.2} us {:6.1} GB/s (reps {:?})",
                    gbps(sym_bytes, v),
                    var_us[vi]
                );
            }
            out_rows.push(row);
            drop(ysym);
        }
    }

    // Kill-line summary: large streaming cells (M >= 2560, K >= 2560).
    let mut min_k = f64::MAX;
    let mut min_kp = f64::MAX;
    for r in out_rows.iter().filter(|r| r.m >= 2560 && r.k >= 2560) {
        min_k = min_k.min(gbps(r.sym_bytes, median(&r.sym_us)));
        min_kp = min_kp.min(gbps(r.sym_bytes, median(&r.symprod_us)));
    }
    let verdict = format!(
        "KILLLINE large cells (M>=2560,K>=2560, rows {:?}): min kernel-only {min_k:.1} GB/s, min producer-inclusive {min_kp:.1} GB/s, threshold 544 GB/s (kernel-only gate) -> {}",
        only.clone().unwrap_or(vec![1, 2, 4, 8]),
        if min_k >= 544.0 { "KEPT" } else { "KILLED" }
    );
    println!("{verdict}");
    if let Ok(dir) = std::env::var("MQ6SYM_OUT") {
        std::fs::write(format!("{dir}/timing.json"), rows_json(&out_rows)).ok();
        std::fs::write(format!("{dir}/timing_verdict.txt"), format!("{verdict}\n")).ok();
    }
}
