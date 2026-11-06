//! MQ8G256 prefill lab (M1 packet of the Raon MQ8G256 project): bit-exact guards and the chain timing
//! of the batched X producer + prefill GEMM (`kernels/src/mq8_rotate_quantize_x_batched.hip`,
//! `gemm_mq8g256_mmq_i8.gfx12.hip`, `gemm_mq8g256_mmq_i8.gfx11.hip`) against today's Q8_0 production chain.
//!
//! ```text
//! mq8_prefill_lab guard|time|all [--layouts a,b] [--shapes M,K,N;...] [--budget-secs S] [--rounds R]
//!                 [--csv-dir D] [--launches L] [--min-rounds R] [--deadline-secs S] [--ring-max N]
//!                 [--sysfs DIR] [--expect-bdf B] [--no-secondary]
//! ```
//! Exit codes: 0 ok, 2 usage / fatal HIP error / unsupported arch, 3 guard failed (no timing runs after a failed guard).
//!
//! # CPU references (EXACT association of `SPEC.md`)
//! * Producer: `mq8_rotate_quantize_x_batched.hip` operation by operation in f32 (Rust f32 never contracts
//!   to fma; the kernel's products with the +-1 / +-0.0625 sign factors are exact, so contraction cannot
//!   change them): `v = x * s1`, FWHT-256 butterflies (strides 1..128 ascending: the kernel's lane-local
//!   strides 1,2,4 then the swizzle strides 8..128 produce the same per-stage values), `v *= 0.0625 * s2`,
//!   `amax = max |v|`, `scale = amax > 0 ? amax / 127 : 1`, `inv = amax > 0 ? 127 / amax : 0`,
//!   `q = clamp(rintf(v * inv), -128, 127)` with rintf = round-half-even. Sign tables: the LCG of
//!   `scratch.rs::gen_fwht_signs` (seeds 42 / 1042), ported below because it is private there.
//! * GEMM: per output `(n, m)`: `acc = 0; for each K128 epoch e ascending: dot = sum_{i<128} W8 * X8 (i32);
//!   scale = RN(sw[m, e/2] * dx[n, e/2]); acc = fmaf(scale, (f32)dot, acc)`; BF16 variants round `acc` RN-even
//!   to BF16 with non-finite passthrough. `sw` is the exact f16 -> f32 decode of the group header.
//! * Weight quantizer: a PORT of `hipfire-quantize::quant_fwht::quantize_mq8g256` (crates/hipfire-quantize/
//!   src/quant_fwht.rs:949, called from pipeline.rs:7401 with `gen_fwht_signs(42/1042, 256)`, main.rs:1083 is the
//!   `use_mq8g256` tensor-class selection) and of `float16.rs::f32_to_f16` (the TRUNCATING f32 -> f16 scale
//!   store). hipfire-quantize is a binary crate whose quantizer functions are `pub(crate)`, so it cannot be
//!   linked from an rdna-compute example; the port is pinned by hand-computed tiny groups in the unit tests.
//!   Weight rounding is `f32::round` (half away from zero, as the quantizer's `.round()`), NOT rintf.
//!
//! # Guard (bit-exact, rc 3 on any mismatch, no timing afterwards)
//! (i) producer GPU vs CPU (codes + scale bits) on small shapes and real shapes (zero groups, zero rows,
//! large dynamic range, rintf tie rows, impulse rows); (ii) GEMM GPU vs CPU for both layouts, F32 and BF16
//! outputs and every tile valid on the arch, tail shapes / odd K/256 / layout A at a +2 byte base, Y
//! NaN-poisoned and followed by a guard band; (iii) real Raon shapes at N = 512 (full compare) and N = 4096
//! (layout A vs B full compare, picker tile vs CPU on every 16th token + the last); (iv) layout A == B.
//!
//! # Timing (after a clean guard)
//! Chains per Raon attention (M, K) at N = 4096 and 512: base (today's production chain on Q8_0 weights of
//! the same logical weights) vs mq8 layout A vs mq8 layout B, ABBA-bracketed batches of `--launches`
//! chain iterations (device_synchronize + Instant per batch), weights on a >= 256 MiB ring (max
//! `--ring-max`, default 16), per-batch load / GPU clock (sysfs, read only) in the CSV, a wall-clock budget
//! per shape. GPU_MAX_HW_QUEUES=1 is set by the card scripts, not here.
use hip_bridge::DeviceBuffer;
use rayon::prelude::*;
use rdna_compute::{
    mq8_admits, mq8_pick_tile, mq8g256_a_bytes, mq8g256_b_bytes, mq8g256_b_code_offset,
    mq8g256_repack_b, DType, Gpu, GpuTensor, HipResult, Mq8Activations, Mq8Tile, Mq8WLayout,
    Q8A8Activations, Q8MmqActivations,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::io::Write;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// numerics: signs, f16 / bf16, FWHT, producer / GEMM references, quantizer ports
// ---------------------------------------------------------------------------------------------

/// Port of `scratch.rs::gen_fwht_signs` (== `hipfire-quantize::quant_fwht::gen_fwht_signs`).
fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

/// Port of `hipfire-quantize/src/float16.rs::f32_to_f16` (truncating mantissa, flush below 2^-24).
fn f32_to_f16_trunc(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0xFF {
        let f16_frac = if frac == 0 { 0 } else { (frac >> 13) | 1 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let f = frac | 0x800000;
        let shift = (1 - new_exp + 13) as u32;
        return ((sign << 15) | (f >> shift)) as u16;
    }
    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

/// Exact f16 -> f32 decode.
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// RN-even f32 -> bf16 -> f32, non-finite passthrough (the kernels' `mq8a8_bf16_rn`).
fn bf16_rn(v: f32) -> f32 {
    if !v.is_finite() {
        return v;
    }
    let u = v.to_bits();
    let lsb = (u >> 16) & 1;
    f32::from_bits(u.wrapping_add(0x7FFF).wrapping_add(lsb) & 0xFFFF_0000)
}

/// `hipfire-quantize::quant_fwht::cpu_fwht_256` (sign, butterflies stride 1..128, scale * sign).
fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 {
        x[i] *= signs1[i];
    }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625f32;
    for i in 0..256 {
        x[i] *= scale * signs2[i];
    }
}

/// One 256-group of the producer (`mq8_rotate_quantize_x_batched`): codes into `q`, returns the F32 scale.
fn producer_group(x: &[f32], s1: &[f32], s2: &[f32], q: &mut [i8]) -> f32 {
    let mut v = [0f32; 256];
    v.copy_from_slice(x);
    cpu_fwht_256(&mut v, s1, s2);
    let amax = v.iter().fold(0.0f32, |m, &t| m.max(t.abs()));
    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
    for i in 0..256 {
        q[i] = (v[i] * inv).round_ties_even().max(-128.0).min(127.0) as i8;
    }
    scale
}

/// CPU producer over `n` tokens of `k` floats: codes `[n, k]` and scales `[k/256, n]`.
fn cpu_producer(x: &[f32], n: usize, k: usize, s1: &[f32], s2: &[f32]) -> (Vec<i8>, Vec<f32>) {
    assert!(k % 256 == 0 && x.len() >= n * k);
    let gpr = k / 256;
    let mut xq = vec![0i8; n * k];
    let per_tok: Vec<Vec<f32>> = xq
        .par_chunks_mut(k)
        .zip(x.par_chunks(k))
        .map(|(q, xr)| {
            (0..gpr)
                .map(|g| {
                    producer_group(
                        &xr[g * 256..g * 256 + 256],
                        s1,
                        s2,
                        &mut q[g * 256..g * 256 + 256],
                    )
                })
                .collect()
        })
        .collect();
    let mut xs = vec![0f32; gpr * n];
    for (t, sc) in per_tok.iter().enumerate() {
        for g in 0..gpr {
            xs[g * n + t] = sc[g];
        }
    }
    (xq, xs)
}

/// One 256-group of `quantize_mq8g256` (port): 258 bytes = f16 scale (truncating) + 256 int8 codes.
fn quantize_mq8g256_group(w: &[f32], s1: &[f32], s2: &[f32], out: &mut [u8]) {
    let mut group = [0.0f32; 256];
    group[..w.len()].copy_from_slice(w);
    cpu_fwht_256(&mut group, s1, s2);
    let amax = group.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
    let sf = f32_to_f16_trunc(scale);
    out[0] = (sf & 0xFF) as u8;
    out[1] = (sf >> 8) as u8;
    for i in 0..256 {
        let q = (group[i] * inv).round().clamp(-128.0, 127.0) as i8;
        out[2 + i] = q as u8;
    }
}

/// Port of `quantize_mq8g256` (layout A bytes: 258 B per group, row-major groups), parallel over groups.
fn quantize_mq8g256_port(w: &[f32], s1: &[f32], s2: &[f32]) -> Vec<u8> {
    let n_blocks = w.len().div_ceil(256);
    let mut out = vec![0u8; n_blocks * 258];
    out.par_chunks_mut(258)
        .zip(w.par_chunks(256))
        .for_each(|(o, g)| quantize_mq8g256_group(g, s1, s2, o));
    out
}

/// Q8_0 quantizer of `bench_q8_raon_shapes.rs::quantize_q8` (34 B blocks of 32, truncating f16 scale).
fn quantize_q8_0(w: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; w.len() / 32 * 34];
    out.par_chunks_mut(34)
        .zip(w.par_chunks(32))
        .for_each(|(o, blk)| {
            let amax = blk.iter().fold(0f32, |a, &b| a.max(b.abs()));
            let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            o[..2].copy_from_slice(&f32_to_f16_trunc(sc).to_le_bytes());
            for (i, &v) in blk.iter().enumerate() {
                o[2 + i] = ((v / sc).round().clamp(-127.0, 127.0) as i8) as u8;
            }
        });
    out
}

/// Decoded MQ8G256 weights for the CPU reference: codes `[m, k]`, f32 scales `[m, k/256]`.
struct WDec {
    m: usize,
    k: usize,
    codes: Vec<i8>,
    scales: Vec<f32>,
}

/// Decode layout A bytes (`m * (k/256) * 258`).
fn decode_a(a: &[u8], m: usize, k: usize) -> WDec {
    let gpr = k / 256;
    assert_eq!(a.len(), m * gpr * 258);
    let mut codes = vec![0i8; m * k];
    let mut scales = vec![0f32; m * gpr];
    for r in 0..m * gpr {
        let g = &a[r * 258..(r + 1) * 258];
        scales[r] = f16_to_f32(u16::from_le_bytes([g[0], g[1]]));
        for i in 0..256 {
            codes[r * 256 + i] = g[2 + i] as i8;
        }
    }
    WDec {
        m,
        k,
        codes,
        scales,
    }
}

/// Independent layout B builder (scale plane, pad to 256 B, code plane); the launcher's
/// `mq8g256_repack_b` is checked against it in the guard.
fn repack_b_ref(a: &[u8], m: usize, k: usize) -> Vec<u8> {
    let gpr = k / 256;
    let code_off = (m * gpr * 2).next_multiple_of(256);
    let mut out = vec![0u8; code_off + m * k];
    for r in 0..m * gpr {
        out[r * 2] = a[r * 258];
        out[r * 2 + 1] = a[r * 258 + 1];
        out[code_off + r * 256..code_off + (r + 1) * 256]
            .copy_from_slice(&a[r * 258 + 2..(r + 1) * 258]);
    }
    out
}

/// GEMM reference of SPEC.md. Returns F32 `[tokens.len(), m]` (token-major rows of the `tokens` list);
/// `xq` is `[n_total, k]`, `xs` is `[k/256, n_total]`.
fn cpu_gemm(w: &WDec, xq: &[i8], xs: &[f32], n_total: usize, tokens: &[usize]) -> Vec<f32> {
    let (m, k) = (w.m, w.k);
    assert!(k % 256 == 0);
    let gpr = k / 256;
    let mut out = vec![0f32; tokens.len() * m];
    out.par_iter_mut()
        .enumerate()
        .with_min_len(16)
        .for_each(|(i, o)| {
            let (ti, mi) = (i / m, i % m);
            let t = tokens[ti];
            let wrow = &w.codes[mi * k..(mi + 1) * k];
            let xrow = &xq[t * k..(t + 1) * k];
            let mut acc = 0.0f32;
            for e in 0..k / 128 {
                let (a, b) = (&wrow[e * 128..e * 128 + 128], &xrow[e * 128..e * 128 + 128]);
                let dot: i32 = a.iter().zip(b).map(|(&p, &q)| p as i32 * q as i32).sum();
                let g = e / 2;
                let scale = w.scales[mi * gpr + g] * xs[g * n_total + t];
                acc = scale.mul_add(dot as f32, acc);
            }
            *o = acc;
        });
    out
}

// ---------------------------------------------------------------------------------------------
// data generation
// ---------------------------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in [-1, 1) (the formula of `bench_q8_raon_shapes.rs::lcg`).
    fn unit(&mut self) -> f32 {
        (self.next_u32() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum XMode {
    Normal,
    /// per-row 10^[-2,3] magnitude, 1/16 zero groups, 1/64 x64 outliers, zero rows
    Wide,
    /// rows of {+-1036, +-996} * 2^j: after the transform |v| in {127, 2.5} * 2^j: rintf ties
    Tie,
    /// one non-zero element per row (codes +-127 everywhere in group 0, every other group is zero)
    Impulse,
}

fn gen_x(n: usize, k: usize, seed: u64, mode: XMode) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut x = vec![0f32; n * k];
    for r in 0..n {
        let row = &mut x[r * k..(r + 1) * k];
        match mode {
            XMode::Normal => row.iter_mut().for_each(|v| *v = rng.unit()),
            XMode::Wide => {
                let rf = 10f32.powf(rng.unit() * 2.5 + 0.5);
                let zero_row = r % 61 == 7;
                for g in 0..k / 256 {
                    let zero = zero_row || rng.below(16) == 0;
                    for v in &mut row[g * 256..g * 256 + 256] {
                        let mut t = rng.unit() * rf;
                        if rng.below(64) == 0 {
                            t *= 64.0;
                        }
                        *v = if zero { 0.0 } else { t };
                    }
                }
            }
            XMode::Tie => {
                let j = rng.below(21) as i32 - 10;
                let p = 2f32.powi(j);
                let (s0, s1) = (
                    if rng.below(2) == 0 { 1.0 } else { -1.0 },
                    if rng.below(2) == 0 { 1.0 } else { -1.0 },
                );
                for g in 0..k / 256 {
                    if g % 5 == 4 {
                        continue;
                    }
                    row[g * 256] = s0 * 1036.0 * p;
                    row[g * 256 + 1] = s1 * 996.0 * p;
                }
            }
            XMode::Impulse => {
                row[0] = if r % 2 == 0 { 1.0 } else { -1.0 } * 1000.0 * (1 + r % 3) as f32;
            }
        }
    }
    x
}

/// BF16-valued f32 weights `[m, k]`. `wide`: per-group 2^[-6,6] magnitude, 1/64 zero groups, rare outliers.
fn gen_w_bf16(m: usize, k: usize, seed: u64, wide: bool) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut w = vec![0f32; m * k];
    for g in 0..m * k / 256 {
        let gs = if wide {
            2f32.powi(rng.below(13) as i32 - 6)
        } else {
            1.0
        };
        let zero = wide && rng.below(64) == 0;
        for v in &mut w[g * 256..g * 256 + 256] {
            let mut t = rng.unit() * 0.05 * gs;
            if wide && rng.below(256) == 0 {
                t *= 16.0;
            }
            *v = if zero { 0.0 } else { bf16_rn(t) };
        }
    }
    w
}

#[derive(Clone, Copy)]
enum RawMode {
    /// random codes over [-128, 127] (1/8 of them extreme), random f16 scales (2^-20 .. 2^4)
    Random,
    /// codes `(-128 | 127) * s2[i]` (even / odd row): with `Impulse` X every product of group 0 has one sign
    Aligned,
}

/// Layout A bytes built directly from random codes (reaches code -128 and extreme dots).
fn gen_w_raw(m: usize, k: usize, seed: u64, mode: RawMode, s2: &[f32]) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let gpr = k / 256;
    let mut a = vec![0u8; m * gpr * 258];
    for r in 0..m * gpr {
        let g = &mut a[r * 258..(r + 1) * 258];
        let e = (rng.below(25) as i32) - 20; // 2^-20 .. 2^4
        let sc = f32_to_f16_trunc(2f32.powi(e) * (1.0 + rng.below(1024) as f32 / 1024.0));
        g[0] = (sc & 0xFF) as u8;
        g[1] = (sc >> 8) as u8;
        let row = r / gpr;
        for i in 0..256 {
            let code: i8 = match mode {
                RawMode::Random => match rng.below(8) {
                    0 => -128,
                    1 => 127,
                    _ => (rng.below(256) as i32 - 128) as i8,
                },
                RawMode::Aligned => {
                    let s = if s2[i] > 0.0 { 1i32 } else { -1 };
                    (if row % 2 == 0 { -128 * s } else { 127 * s }) as i8
                }
            };
            g[2 + i] = code as u8;
        }
    }
    a
}

// ---------------------------------------------------------------------------------------------
// GPU plumbing
// ---------------------------------------------------------------------------------------------

fn ck<T>(r: HipResult<T>, what: &str) -> T {
    match r {
        Ok(v) => v,
        Err(e) => {
            eprintln!("FATAL {what}: {e:?}");
            std::process::exit(2);
        }
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding / invalid bit patterns as bytes; length is exact.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn f32_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    // SAFETY: as above; every byte pattern is a valid f32.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
}
fn i8_bytes_mut(v: &mut [i8]) -> &mut [u8] {
    // SAFETY: as above.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len()) }
}

/// Non-owning (ptr, bytes) handle of a device allocation owned by a [`Scope`].
#[derive(Clone, Copy)]
struct Dbuf {
    ptr: *mut c_void,
    bytes: usize,
}

impl Dbuf {
    fn view(&self) -> DeviceBuffer {
        // SAFETY: the allocation outlives every use (owned by a Scope that frees at the end of the case).
        unsafe { DeviceBuffer::from_raw(self.ptr, self.bytes) }
    }
    fn sub(&self, off: usize, bytes: usize) -> Dbuf {
        assert!(off + bytes <= self.bytes);
        Dbuf {
            ptr: unsafe { (self.ptr as *mut u8).add(off) as *mut c_void },
            bytes,
        }
    }
    fn tensor(&self, shape: Vec<usize>, dtype: DType) -> GpuTensor {
        GpuTensor {
            buf: self.view(),
            shape,
            dtype,
        }
    }
    fn h2d(&self, gpu: &Gpu, src: &[u8]) {
        ck(gpu.hip.memcpy_htod(&self.view(), src), "memcpy_htod");
    }
    fn d2h(&self, gpu: &Gpu, dst: &mut [u8]) {
        ck(gpu.hip.device_synchronize(), "device_synchronize");
        ck(gpu.hip.memcpy_dtoh(dst, &self.view()), "memcpy_dtoh");
    }
    fn memset(&self, gpu: &Gpu, value: i32) {
        ck(gpu.hip.memset(&self.view(), value, self.bytes), "memset");
    }
}

#[derive(Default)]
struct Scope {
    bufs: Vec<DeviceBuffer>,
}

impl Scope {
    fn alloc(&mut self, gpu: &Gpu, bytes: usize) -> Dbuf {
        let b = ck(gpu.hip.malloc(bytes.max(16)), "malloc");
        let d = Dbuf {
            ptr: b.as_ptr(),
            bytes,
        };
        self.bufs.push(b);
        d
    }
    fn upload(&mut self, gpu: &Gpu, src: &[u8]) -> Dbuf {
        let d = self.alloc(gpu, src.len());
        d.h2d(gpu, src);
        d
    }
    fn free(self, gpu: &Gpu) {
        ck(gpu.hip.device_synchronize(), "device_synchronize");
        for b in self.bufs {
            ck(gpu.hip.free(b), "free");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// stats helpers
// ---------------------------------------------------------------------------------------------

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        return f64::NAN;
    }
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn quantile(v: &[f64], q: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        return f64::NAN;
    }
    let p = q * (v.len() - 1) as f64;
    let (lo, hi) = (p.floor() as usize, p.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * (p - lo as f64)
}

// ---------------------------------------------------------------------------------------------
// guard
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct Tally {
    pass: usize,
    fail: usize,
}

impl Tally {
    fn report(&mut self, ok: bool, name: &str, detail: &str) {
        if ok {
            self.pass += 1;
            println!("GUARD PASS {name} {detail}");
        } else {
            self.fail += 1;
            println!("GUARD FAIL {name} {detail}");
        }
        let _ = std::io::stdout().flush();
    }
}

const GUARD_BAND_BYTES: usize = 1 << 16;
const POISON_BYTE: i32 = 0xFF; // f32 0xFFFFFFFF = NaN: a never-written element stays NaN
const BAND_BYTE: i32 = 0xA5;

/// Compare `got` (rows of `m`) with `want` (rows of the listed tokens) bit for bit; Err(detail) on mismatch.
fn cmp_rows(got: &[f32], m: usize, want: &[f32], tokens: Option<&[usize]>) -> Result<(), String> {
    let rows = want.len() / m;
    let mut bad = 0usize;
    let mut first: Option<(usize, usize, u32, u32)> = None;
    for ti in 0..rows {
        let t = tokens.map_or(ti, |l| l[ti]);
        let (g, w) = (&got[t * m..(t + 1) * m], &want[ti * m..(ti + 1) * m]);
        for j in 0..m {
            if g[j].to_bits() != w[j].to_bits() {
                bad += 1;
                first.get_or_insert((t, j, g[j].to_bits(), w[j].to_bits()));
            }
        }
    }
    match first {
        None => Ok(()),
        Some((t, j, g, w)) => Err(format!(
            "{bad} of {} elements differ, first at token {t} row {j}: gpu 0x{g:08x} ({}) cpu 0x{w:08x} ({})",
            rows * m,
            f32::from_bits(g),
            f32::from_bits(w)
        )),
    }
}

fn cmp_bits_full(a: &[f32], b: &[f32]) -> Result<(), String> {
    let mut bad = 0usize;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x.to_bits() != y.to_bits() {
            bad += 1;
            first.get_or_insert((i, x.to_bits(), y.to_bits()));
        }
    }
    match first {
        None => Ok(()),
        Some((i, x, y)) => Err(format!(
            "{bad} of {} elements differ, first at flat index {i}: 0x{x:08x} vs 0x{y:08x}",
            a.len()
        )),
    }
}

struct Signs {
    s1: Vec<f32>,
    s2: Vec<f32>,
}

/// Producer guard: GPU vs CPU codes and scale bits for `x` `[n, k]`. Returns the CPU planes.
fn guard_producer(
    gpu: &mut Gpu,
    tally: &mut Tally,
    name: &str,
    x: &[f32],
    n: usize,
    k: usize,
    signs: &Signs,
    scope: &mut Scope,
) -> (Vec<i8>, Vec<f32>) {
    let (cq, cs) = cpu_producer(x, n, k, &signs.s1, &signs.s2);
    let xd = scope.upload(gpu, f32_bytes(x));
    let xt = xd.tensor(vec![n, k], DType::F32);
    let acts = ck(gpu.mq8_quantize_x(&xt, n, k), "mq8_quantize_x");
    ck(gpu.hip.device_synchronize(), "device_synchronize");
    let mut gq = vec![0i8; n * k];
    let mut gs = vec![0f32; n * (k / 256)];
    Dbuf {
        ptr: acts.xq_ptr(),
        bytes: n * k,
    }
    .d2h(gpu, i8_bytes_mut(&mut gq));
    Dbuf {
        ptr: acts.xs_ptr(),
        bytes: gs.len() * 4,
    }
    .d2h(gpu, f32_bytes_mut(&mut gs));
    let code_bad = gq.iter().zip(&cq).filter(|(a, b)| a != b).count();
    let scale_bad = gs
        .iter()
        .zip(&cs)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let first_code = gq.iter().zip(&cq).position(|(a, b)| a != b);
    let detail = if code_bad + scale_bad == 0 {
        format!(
            "n={n} k={k} ({} codes, {} scales bit-exact)",
            gq.len(),
            gs.len()
        )
    } else {
        format!(
            "n={n} k={k}: {code_bad} code and {scale_bad} scale mismatches{}",
            first_code.map_or(String::new(), |i| format!(
                ", first code at token {} elem {}: gpu {} cpu {}",
                i / k,
                i % k,
                gq[i],
                cq[i]
            ))
        )
    };
    tally.report(
        code_bad + scale_bad == 0,
        &format!("producer {name}"),
        &detail,
    );
    (cq, cs)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Sel {
    Pick,
    Tile(Mq8Tile),
}

fn sel_label(sel: Sel, arch: &str, m: usize, n: usize) -> String {
    match sel {
        Sel::Pick => format!("pick({})", mq8_pick_tile(arch, m, n).label()),
        Sel::Tile(t) => t.label().to_string(),
    }
}

fn launch_gemm(
    gpu: &mut Gpu,
    w: &GpuTensor,
    layout: Mq8WLayout,
    acts: &Mq8Activations,
    y: &GpuTensor,
    m: usize,
    sel: Sel,
    bf16: bool,
) {
    let r = match (sel, bf16) {
        (Sel::Pick, false) => gpu.gemm_mq8g256_gemm(w, layout, acts, y, m),
        (Sel::Pick, true) => gpu.gemm_mq8g256_gemm_bf16(w, layout, acts, y, m),
        (Sel::Tile(t), b) => gpu.gemm_mq8g256_gemm_tile_ex(w, layout, acts, y, m, t, b),
    };
    ck(r, "gemm_mq8g256");
}

struct GemmCase {
    name: String,
    m: usize,
    k: usize,
    n: usize,
    /// layout A weights start this many bytes into their allocation
    a_off: usize,
    /// None: quantizer-port weights from `gen_w_bf16(wide)`; Some: raw codes
    raw: Option<RawMode>,
    wide_w: bool,
    xmode: XMode,
    /// compare only these tokens against the CPU (None: all). Poison and A-vs-B stay full.
    sample: bool,
    /// every valid tile in addition to the picker
    all_tiles: bool,
}

fn layouts_label(l: Mq8WLayout) -> &'static str {
    match l {
        Mq8WLayout::A => "A",
        Mq8WLayout::B => "B",
    }
}

fn guard_gemm_case(
    gpu: &mut Gpu,
    tally: &mut Tally,
    signs: &Signs,
    layouts: &[Mq8WLayout],
    c: &GemmCase,
    seed: u64,
) {
    let arch = gpu.arch.clone();
    let (m, k, n) = (c.m, c.k, c.n);
    let t0 = Instant::now();
    let a_bytes = match c.raw {
        None => {
            let w = gen_w_bf16(m, k, seed, c.wide_w);
            quantize_mq8g256_port(&w, &signs.s1, &signs.s2)
        }
        Some(mode) => gen_w_raw(m, k, seed, mode, &signs.s2),
    };
    assert_eq!(a_bytes.len(), mq8g256_a_bytes(m, k));
    let b_bytes = mq8g256_repack_b(&a_bytes, m, k);
    let wdec = decode_a(&a_bytes, m, k);
    let x = gen_x(n, k, seed ^ 0xABCD, c.xmode);
    let mut scope = Scope::default();
    // producer GPU vs CPU on this case's X
    let (xq, xs) = guard_producer(gpu, tally, &c.name, &x, n, k, signs, &mut scope);
    // the producer guard leaves its activations in the scratch plane: re-run to hold a live handle
    let xd = scope.upload(gpu, f32_bytes(&x));
    let xt = xd.tensor(vec![n, k], DType::F32);
    let acts = ck(gpu.mq8_quantize_x(&xt, n, k), "mq8_quantize_x");
    // CPU reference (full or sampled tokens)
    let tokens: Vec<usize> = if c.sample {
        let mut t: Vec<usize> = (0..n).step_by(16).collect();
        if *t.last().unwrap() != n - 1 {
            t.push(n - 1);
        }
        t
    } else {
        (0..n).collect()
    };
    let tok_opt: Option<&[usize]> = if c.sample { Some(&tokens) } else { None };
    let ref_f32 = cpu_gemm(&wdec, &xq, &xs, n, &tokens);
    let ref_bf16: Vec<f32> = ref_f32.iter().map(|&v| bf16_rn(v)).collect();
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
    // device weights
    let mut wa: Option<(Dbuf, GpuTensor)> = None;
    let mut wb: Option<(Dbuf, GpuTensor)> = None;
    if layouts.contains(&Mq8WLayout::A) {
        let base = scope.alloc(gpu, a_bytes.len() + 16);
        let v = base.sub(c.a_off, a_bytes.len());
        v.h2d(gpu, &a_bytes);
        wa = Some((v, v.tensor(vec![m, k], DType::MQ8G256)));
    }
    if layouts.contains(&Mq8WLayout::B) {
        let d = scope.upload(gpu, &b_bytes);
        wb = Some((d, d.tensor(vec![m, k], DType::MQ8G256)));
    }
    let ybytes = n * m * 4;
    let ybase = scope.alloc(gpu, ybytes + GUARD_BAND_BYTES);
    let yv = ybase.sub(0, ybytes);
    let yt = yv.tensor(vec![n, m], DType::F32);
    let band = ybase.sub(ybytes, GUARD_BAND_BYTES);

    let mut sels = vec![Sel::Pick];
    if c.all_tiles {
        for t in [
            Mq8Tile::T128x128,
            Mq8Tile::T128x64,
            Mq8Tile::T64x128,
            Mq8Tile::T64x64,
        ] {
            if t.valid_on(&arch) {
                sels.push(Sel::Tile(t));
            }
        }
    }
    let mut y_host = vec![0f32; n * m];
    let mut band_host = vec![0u8; GUARD_BAND_BYTES];
    for &sel in &sels {
        for bf16 in [false, true] {
            let want = if bf16 { &ref_bf16 } else { &ref_f32 };
            let mut outs: Vec<(Mq8WLayout, Vec<f32>)> = Vec::new();
            for &layout in layouts {
                let w = match layout {
                    Mq8WLayout::A => &wa.as_ref().unwrap().1,
                    Mq8WLayout::B => &wb.as_ref().unwrap().1,
                };
                yv.memset(gpu, POISON_BYTE);
                band.memset(gpu, BAND_BYTE);
                launch_gemm(gpu, w, layout, &acts, &yt, m, sel, bf16);
                yv.d2h(gpu, f32_bytes_mut(&mut y_host));
                band.d2h(gpu, &mut band_host);
                let poison_left = y_host.iter().filter(|v| v.to_bits() == 0xFFFF_FFFF).count();
                let band_ok = band_host.iter().all(|&b| b == BAND_BYTE as u8);
                let cmp = cmp_rows(&y_host, m, want, tok_opt);
                let name = format!(
                    "gemm {} layout={} tile={} out={} a_off={}",
                    c.name,
                    layouts_label(layout),
                    sel_label(sel, &arch, m, n),
                    if bf16 { "bf16" } else { "f32" },
                    if layout == Mq8WLayout::A { c.a_off } else { 0 }
                );
                let ok = cmp.is_ok() && poison_left == 0 && band_ok;
                let detail = if ok {
                    format!(
                        "M={m} K={k} N={n} ({} elements{} bit-exact, poison clear, guard band intact)",
                        want.len(),
                        if c.sample { " sampled" } else { "" }
                    )
                } else {
                    format!(
                        "M={m} K={k} N={n}: {}{}{}",
                        cmp.err().unwrap_or_default(),
                        if poison_left > 0 {
                            format!("; {poison_left} elements never written (poison left)")
                        } else {
                            String::new()
                        },
                        if band_ok {
                            ""
                        } else {
                            "; GUARD BAND OVERWRITTEN"
                        }
                    )
                };
                tally.report(ok, &name, &detail);
                outs.push((layout, y_host.clone()));
            }
            if outs.len() == 2 {
                let r = cmp_bits_full(&outs[0].1, &outs[1].1);
                tally.report(
                    r.is_ok(),
                    &format!(
                        "gemm {} layoutA==layoutB tile={} out={}",
                        c.name,
                        sel_label(sel, &arch, m, n),
                        if bf16 { "bf16" } else { "f32" }
                    ),
                    &r.err()
                        .unwrap_or_else(|| format!("M={m} K={k} N={n} (all {} elements)", n * m)),
                );
            }
        }
    }
    println!(
        "  [{} done: cpu prep {cpu_ms:.0} ms, total {:.1} s]",
        c.name,
        t0.elapsed().as_secs_f64()
    );
    scope.free(gpu);
}

fn guard_repack(tally: &mut Tally, signs: &Signs) {
    for (m, k) in [(3usize, 256usize), (5, 768), (130, 512), (576, 2560)] {
        let w = gen_w_bf16(m, k, 99, true);
        let a = quantize_mq8g256_port(&w, &signs.s1, &signs.s2);
        let got = mq8g256_repack_b(&a, m, k);
        let want = repack_b_ref(&a, m, k);
        let gpr = k / 256;
        let code_off = mq8g256_b_code_offset(m, k);
        let mut ok = got.len() == want.len()
            && got.len() == mq8g256_b_bytes(m, k)
            && code_off == (m * gpr * 2).next_multiple_of(256)
            && mq8g256_a_bytes(m, k) == a.len();
        if ok {
            ok = got[..m * gpr * 2] == want[..m * gpr * 2] && got[code_off..] == want[code_off..];
        }
        tally.report(
            ok,
            &format!("repack_b M{m}_K{k}"),
            &format!(
                "launcher mq8g256_repack_b vs independent builder ({} B)",
                want.len()
            ),
        );
    }
}

/// Guard (batched entry points): `gemm_mq8g256_batched[_bf16]` == producer + GEMM chain.
fn guard_batched(gpu: &mut Gpu, tally: &mut Tally, signs: &Signs, layouts: &[Mq8WLayout]) {
    let (m, k, n) = (130usize, 512usize, 70usize);
    let w = gen_w_bf16(m, k, 5, true);
    let a = quantize_mq8g256_port(&w, &signs.s1, &signs.s2);
    let b = mq8g256_repack_b(&a, m, k);
    let x = gen_x(n, k, 6, XMode::Wide);
    let mut scope = Scope::default();
    let xd = scope.upload(gpu, f32_bytes(&x));
    let xt = xd.tensor(vec![n, k], DType::F32);
    let y1 = scope.alloc(gpu, n * m * 4);
    let y2 = scope.alloc(gpu, n * m * 4);
    let (t1, t2) = (
        y1.tensor(vec![n, m], DType::F32),
        y2.tensor(vec![n, m], DType::F32),
    );
    for &layout in layouts {
        let wd = match layout {
            Mq8WLayout::A => scope.upload(gpu, &a),
            Mq8WLayout::B => scope.upload(gpu, &b),
        };
        let wt = wd.tensor(vec![m, k], DType::MQ8G256);
        for bf16 in [false, true] {
            y1.memset(gpu, POISON_BYTE);
            y2.memset(gpu, POISON_BYTE);
            let acts = ck(gpu.mq8_quantize_x(&xt, n, k), "mq8_quantize_x");
            launch_gemm(gpu, &wt, layout, &acts, &t1, m, Sel::Pick, bf16);
            ck(
                if bf16 {
                    gpu.gemm_mq8g256_batched_bf16(&wt, layout, &xt, &t2, m, k, n)
                } else {
                    gpu.gemm_mq8g256_batched(&wt, layout, &xt, &t2, m, k, n)
                },
                "gemm_mq8g256_batched",
            );
            let (mut h1, mut h2) = (vec![0f32; n * m], vec![0f32; n * m]);
            y1.d2h(gpu, f32_bytes_mut(&mut h1));
            y2.d2h(gpu, f32_bytes_mut(&mut h2));
            let r = cmp_bits_full(&h1, &h2);
            tally.report(
                r.is_ok(),
                &format!(
                    "batched layout={} out={} == producer+gemm",
                    layouts_label(layout),
                    if bf16 { "bf16" } else { "f32" }
                ),
                &r.err().unwrap_or_else(|| format!("M={m} K={k} N={n}")),
            );
        }
    }
    scope.free(gpu);
}

const RAON_MK: [(usize, usize); 4] = [(5120, 2560), (2560, 5120), (576, 2560), (10240, 512)];

fn run_guard(
    gpu: &mut Gpu,
    layouts: &[Mq8WLayout],
    shapes: Option<&[(usize, usize, usize)]>,
) -> bool {
    let t0 = Instant::now();
    let signs = Signs {
        s1: gen_fwht_signs(42, 256),
        s2: gen_fwht_signs(1042, 256),
    };
    let mut tally = Tally::default();
    println!(
        "== GUARD arch {} layouts {:?}",
        gpu.arch,
        layouts
            .iter()
            .map(|l| layouts_label(*l))
            .collect::<Vec<_>>()
    );

    // (0) layout B repack of the launcher
    guard_repack(&mut tally, &signs);

    // (i) producer, small + real shapes
    println!("== GUARD (i) producer");
    let mut small: Vec<(String, usize, usize, XMode)> = Vec::new();
    for (n, k) in [
        (1usize, 256usize),
        (5, 256),
        (37, 512),
        (130, 768),
        (9, 2560),
        (33, 1024),
    ] {
        for mode in [XMode::Wide, XMode::Normal, XMode::Tie, XMode::Impulse] {
            small.push((format!("{mode:?}_n{n}_k{k}"), n, k, mode));
        }
    }
    for (name, n, k, mode) in small {
        let x = gen_x(n, k, 1000 + (n * 31 + k) as u64, mode);
        let mut scope = Scope::default();
        guard_producer(gpu, &mut tally, &name, &x, n, k, &signs, &mut scope);
        scope.free(gpu);
    }
    for (n, k) in [
        (4096usize, 2560usize),
        (4096, 5120),
        (4096, 512),
        (512, 2560),
        (512, 5120),
        (512, 512),
    ] {
        for mode in [XMode::Wide, XMode::Normal] {
            let x = gen_x(n, k, 2000 + (n + k) as u64, mode);
            let mut scope = Scope::default();
            guard_producer(
                gpu,
                &mut tally,
                &format!("real_{mode:?}_n{n}_k{k}"),
                &x,
                n,
                k,
                &signs,
                &mut scope,
            );
            scope.free(gpu);
        }
    }

    // (ii) GEMM tails / alignment / extremes, every valid tile
    println!("== GUARD (ii) gemm tails, all tiles, both outputs");
    let mk = |name: &str, m, k, n, a_off, raw, wide_w, xmode| GemmCase {
        name: name.to_string(),
        m,
        k,
        n,
        a_off,
        raw,
        wide_w,
        xmode,
        sample: false,
        all_tiles: true,
    };
    let cases = vec![
        mk("tiny_M1_K256_N1", 1, 256, 1, 0, None, false, XMode::Normal),
        mk(
            "onetile_M17_K256_N5",
            17,
            256,
            5,
            0,
            None,
            true,
            XMode::Wide,
        ),
        mk(
            "tail_M130_K512_N130",
            130,
            512,
            130,
            0,
            None,
            true,
            XMode::Wide,
        ),
        mk(
            "tail_M130_K512_N130_off2",
            130,
            512,
            130,
            2,
            None,
            true,
            XMode::Wide,
        ),
        mk(
            "tail_M576_K2560_N37",
            576,
            2560,
            37,
            0,
            None,
            true,
            XMode::Wide,
        ),
        mk("oddK768_M200_N70", 200, 768, 70, 0, None, true, XMode::Wide),
        mk(
            "oddK768_M200_N70_off2",
            200,
            768,
            70,
            2,
            None,
            true,
            XMode::Wide,
        ),
        mk(
            "multitile_M300_K1024_N300",
            300,
            1024,
            300,
            0,
            None,
            false,
            XMode::Normal,
        ),
        mk(
            "raw_random_M97_K768_N65",
            97,
            768,
            65,
            0,
            Some(RawMode::Random),
            false,
            XMode::Wide,
        ),
        mk(
            "raw_random_M97_K768_N65_off2",
            97,
            768,
            65,
            2,
            Some(RawMode::Random),
            false,
            XMode::Tie,
        ),
        mk(
            "raw_extreme_M64_K512_N64",
            64,
            512,
            64,
            0,
            Some(RawMode::Aligned),
            false,
            XMode::Impulse,
        ),
    ];
    for (i, c) in cases.iter().enumerate() {
        guard_gemm_case(gpu, &mut tally, &signs, layouts, c, 7000 + i as u64);
    }
    guard_batched(gpu, &mut tally, &signs, layouts);

    // (iii) real Raon shapes at the picker tile
    println!("== GUARD (iii) real shapes, picker tile");
    let real: Vec<(usize, usize, usize)> = match shapes {
        Some(s) => s.to_vec(),
        None => [512usize, 4096]
            .iter()
            .flat_map(|&n| RAON_MK.iter().map(move |&(m, k)| (m, k, n)))
            .collect(),
    };
    for (i, &(m, k, n)) in real.iter().enumerate() {
        let c = GemmCase {
            name: format!("real_M{m}_K{k}_N{n}"),
            m,
            k,
            n,
            a_off: 0,
            raw: None,
            wide_w: false,
            xmode: XMode::Normal,
            sample: n > 512,
            all_tiles: false,
        };
        guard_gemm_case(gpu, &mut tally, &signs, layouts, &c, 9000 + i as u64);
    }
    println!(
        "GUARD TALLY pass={} fail={} elapsed={:.1}s",
        tally.pass,
        tally.fail,
        t0.elapsed().as_secs_f64()
    );
    println!(
        "GUARD {}",
        if tally.fail == 0 { "PASS ALL" } else { "FAIL" }
    );
    tally.fail == 0
}

// ---------------------------------------------------------------------------------------------
// timing
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Arm {
    Base,
    A,
    B,
    Base32,
    A32,
    B32,
    ProdBase,
    ProdMq,
    GemmBase,
    GemmA,
    GemmB,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Base => "base",
            Arm::A => "mq8A",
            Arm::B => "mq8B",
            Arm::Base32 => "base_f32",
            Arm::A32 => "mq8A_f32",
            Arm::B32 => "mq8B_f32",
            Arm::ProdBase => "prod_base",
            Arm::ProdMq => "prod_mq8",
            Arm::GemmBase => "gemm_base",
            Arm::GemmA => "gemm_mq8A",
            Arm::GemmB => "gemm_mq8B",
        }
    }
}

enum BaseActs {
    A8(Q8A8Activations),
    Mmq(Q8MmqActivations),
}

struct St {
    gfx12: bool,
    m: usize,
    k: usize,
    n: usize,
    x: GpuTensor,
    y: GpuTensor,
    ring_q8: Vec<GpuTensor>,
    ring_a: Vec<GpuTensor>,
    ring_b: Vec<GpuTensor>,
    idx: [usize; 3],
    base_acts: Option<BaseActs>,
    mq_acts: Option<Mq8Activations>,
}

impl St {
    fn next(&mut self, kind: usize) -> usize {
        let len = [self.ring_q8.len(), self.ring_a.len(), self.ring_b.len()][kind].max(1);
        let i = self.idx[kind] % len;
        self.idx[kind] = i + 1;
        i
    }
}

fn base_quantize(gpu: &mut Gpu, st: &St) -> BaseActs {
    if st.gfx12 {
        BaseActs::A8(ck(
            gpu.q8_a8_quantize_x(&st.x, st.n, st.k),
            "q8_a8_quantize_x",
        ))
    } else {
        BaseActs::Mmq(ck(
            gpu.q8_mmq_quantize_x(&st.x, st.n, st.k),
            "q8_mmq_quantize_x",
        ))
    }
}

fn base_gemm(gpu: &mut Gpu, w: &GpuTensor, acts: &BaseActs, y: &GpuTensor, m: usize) {
    match acts {
        BaseActs::A8(a) => ck(
            gpu.gemm_q8_0_a8_gemm_bf16(w, a, y, m),
            "gemm_q8_0_a8_gemm_bf16",
        ),
        BaseActs::Mmq(a) => ck(
            gpu.gemm_q8_0_mmq_gfx11_gemm_bf16(w, a, y, m),
            "gemm_q8_0_mmq_gfx11_gemm_bf16",
        ),
    }
}

fn run_iter(gpu: &mut Gpu, st: &mut St, arm: Arm) {
    let (m, k, n) = (st.m, st.k, st.n);
    match arm {
        Arm::Base => {
            let acts = base_quantize(gpu, st);
            let i = st.next(0);
            base_gemm(gpu, &st.ring_q8[i], &acts, &st.y, m);
        }
        Arm::A | Arm::B => {
            let acts = ck(gpu.mq8_quantize_x(&st.x, n, k), "mq8_quantize_x");
            let (kind, layout) = if arm == Arm::A {
                (1, Mq8WLayout::A)
            } else {
                (2, Mq8WLayout::B)
            };
            let i = st.next(kind);
            let w = if kind == 1 {
                &st.ring_a[i]
            } else {
                &st.ring_b[i]
            };
            ck(
                gpu.gemm_mq8g256_gemm_bf16(w, layout, &acts, &st.y, m),
                "gemm_mq8g256_gemm_bf16",
            );
        }
        Arm::Base32 => {
            let i = st.next(0);
            let w = &st.ring_q8[i];
            if st.gfx12 {
                ck(
                    gpu.gemm_q8_0_a8_batched(w, &st.x, &st.y, m, k, n),
                    "gemm_q8_0_a8_batched",
                );
            } else {
                ck(
                    gpu.gemm_q8_0_mmq_gfx11_batched(w, &st.x, &st.y, m, k, n),
                    "gemm_q8_0_mmq_gfx11_batched",
                );
            }
        }
        Arm::A32 | Arm::B32 => {
            let (kind, layout) = if arm == Arm::A32 {
                (1, Mq8WLayout::A)
            } else {
                (2, Mq8WLayout::B)
            };
            let i = st.next(kind);
            let w = if kind == 1 {
                &st.ring_a[i]
            } else {
                &st.ring_b[i]
            };
            ck(
                gpu.gemm_mq8g256_batched(w, layout, &st.x, &st.y, m, k, n),
                "gemm_mq8g256_batched",
            );
        }
        Arm::ProdBase => {
            let _ = base_quantize(gpu, st);
        }
        Arm::ProdMq => {
            ck(gpu.mq8_quantize_x(&st.x, n, k), "mq8_quantize_x");
        }
        Arm::GemmBase => {
            let i = st.next(0);
            let acts = st.base_acts.as_ref().expect("base acts prepared");
            base_gemm(gpu, &st.ring_q8[i], acts, &st.y, m);
        }
        Arm::GemmA | Arm::GemmB => {
            let (kind, layout) = if arm == Arm::GemmA {
                (1, Mq8WLayout::A)
            } else {
                (2, Mq8WLayout::B)
            };
            let i = st.next(kind);
            let w = if kind == 1 {
                &st.ring_a[i]
            } else {
                &st.ring_b[i]
            };
            let acts = st.mq_acts.as_ref().expect("mq8 acts prepared");
            ck(
                gpu.gemm_mq8g256_gemm_bf16(w, layout, acts, &st.y, m),
                "gemm_mq8g256_gemm_bf16",
            );
        }
    }
}

#[derive(Clone, Copy)]
struct Clk {
    sclk: i64,
    hw: i64,
    pw: i64,
}

fn slurp(p: &str) -> String {
    std::fs::read_to_string(p).unwrap_or_default()
}

/// pp_dpm_sclk active line, hwmon freq1 (MHz) and power (W): sysfs reads only.
fn read_clk(dir: &str) -> Clk {
    let mut c = Clk {
        sclk: -1,
        hw: -1,
        pw: -1,
    };
    for line in slurp(&format!("{dir}/pp_dpm_sclk")).lines() {
        if line.contains('*') {
            if let Some(col) = line.find(':') {
                let digits: String = line[col + 1..]
                    .trim_start()
                    .chars()
                    .take_while(|ch| ch.is_ascii_digit())
                    .collect();
                c.sclk = digits.parse().unwrap_or(-1);
            }
        }
    }
    if let Ok(rd) = std::fs::read_dir(format!("{dir}/hwmon")) {
        for e in rd.flatten() {
            let h = e.path().to_string_lossy().to_string();
            if let Ok(v) = slurp(&format!("{h}/freq1_input")).trim().parse::<i64>() {
                c.hw = v / 1_000_000;
            }
            let mut p = slurp(&format!("{h}/power1_average"));
            if p.trim().is_empty() {
                p = slurp(&format!("{h}/power1_input"));
            }
            if let Ok(v) = p.trim().parse::<i64>() {
                c.pw = v / 1_000_000;
            }
        }
    }
    c
}

fn read_load() -> (f64, f64) {
    let t = slurp("/proc/loadavg");
    let mut it = t.split_whitespace();
    let a = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1.0);
    let b = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1.0);
    (a, b)
}

struct Sample {
    arm: Arm,
    ms: f64,
}

/// One timed unit: `l` iterations of `arm` bracketed by device_synchronize and an Instant.
fn batch(
    gpu: &mut Gpu,
    st: &mut St,
    arm: Arm,
    l: usize,
    sysfs: &str,
) -> (f64, Clk, Clk, (f64, f64)) {
    let load = read_load();
    let pre = read_clk(sysfs);
    ck(gpu.hip.device_synchronize(), "device_synchronize");
    let t0 = Instant::now();
    for _ in 0..l {
        run_iter(gpu, st, arm);
    }
    ck(gpu.hip.device_synchronize(), "device_synchronize");
    let dt = t0.elapsed().as_secs_f64() * 1e3 / l as f64;
    let post = read_clk(sysfs);
    (dt, pre, post, load)
}

#[derive(Clone)]
enum Group {
    /// X, Y, Y, X (ratio X/Y per round)
    Abba(Arm, Arm),
    /// single batches in the given order (palindrome), after `prepare`
    Seq(Vec<Arm>),
}

#[derive(Clone, Default)]
struct Pair {
    med: f64,
    p5: f64,
    p95: f64,
}

fn pair_of(v: &[f64]) -> Pair {
    Pair {
        med: median(v),
        p5: quantile(v, 0.05),
        p95: quantile(v, 0.95),
    }
}

#[derive(Clone)]
struct ShapeResult {
    m: usize,
    k: usize,
    n: usize,
    rounds: usize,
    ms: HashMap<Arm, f64>,
    sclk: HashMap<Arm, f64>,
    /// X/Y ratio per Abba pair
    ratio: HashMap<(Arm, Arm), Pair>,
}

impl ShapeResult {
    fn ms(&self, a: Arm) -> f64 {
        self.ms.get(&a).copied().unwrap_or(f64::NAN)
    }
    fn speedup(&self, arm: Arm) -> Option<&Pair> {
        self.ratio.get(&(Arm::Base, arm))
    }
    fn a_cost_pct(&self) -> Option<f64> {
        self.ratio
            .get(&(Arm::A, Arm::B))
            .map(|p| (p.med - 1.0) * 100.0)
    }
}

fn calls_for(m: usize, k: usize) -> usize {
    match (m, k) {
        (5120, 2560) => 79,
        (2560, 5120) => 25,
        (10240, 512) => 7,
        (576, 2560) => 14,
        _ => 0,
    }
}

struct TimeOpts {
    layouts: Vec<Mq8WLayout>,
    budget: Duration,
    rounds: usize,
    min_rounds: usize,
    launches: usize,
    csv_dir: Option<String>,
    ring_max: usize,
    sysfs: String,
    secondary: bool,
    deadline: Option<Instant>,
}

fn time_shape(
    gpu: &mut Gpu,
    signs: &Signs,
    m: usize,
    k: usize,
    n: usize,
    o: &TimeOpts,
) -> Option<ShapeResult> {
    let arch = gpu.arch.clone();
    let tag = format!("M{m}_K{k}_N{n}");
    if let Some(d) = o.deadline {
        if Instant::now() + Duration::from_secs(10) >= d {
            println!("SKIPPED {arch} {tag}: global deadline reached before setup");
            return None;
        }
    }
    let gfx12 = gpu.arch_caps.is_rdna4();
    let has_a = o.layouts.contains(&Mq8WLayout::A);
    let has_b = o.layouts.contains(&Mq8WLayout::B);
    // weights: the same logical BF16-valued weights in Q8_0 (base) and MQ8G256 (A / B)
    let w = gen_w_bf16(m, k, 0x5EED ^ (m as u64 * 31 + k as u64), false);
    let q8 = quantize_q8_0(&w);
    let a_bytes = quantize_mq8g256_port(&w, &signs.s1, &signs.s2);
    drop(w);
    let b_bytes = mq8g256_repack_b(&a_bytes, m, k);
    let x = gen_x(n, k, 0xFEED ^ (n as u64 * 7 + k as u64), XMode::Normal);
    let mut scope = Scope::default();
    let ring_of = |len: usize| (((256usize << 20) / len.max(1)).max(1)).min(o.ring_max.max(1));
    let mk_ring = |scope: &mut Scope, gpu: &Gpu, bytes: &[u8], dtype: DType| -> Vec<GpuTensor> {
        (0..ring_of(bytes.len()))
            .map(|_| scope.upload(gpu, bytes).tensor(vec![m, k], dtype))
            .collect()
    };
    let ring_q8 = mk_ring(&mut scope, gpu, &q8, DType::Q8_0);
    let ring_a = if has_a {
        mk_ring(&mut scope, gpu, &a_bytes, DType::MQ8G256)
    } else {
        Vec::new()
    };
    let ring_b = if has_b {
        mk_ring(&mut scope, gpu, &b_bytes, DType::MQ8G256)
    } else {
        Vec::new()
    };
    let xd = scope.upload(gpu, f32_bytes(&x));
    let yd = scope.alloc(gpu, n * m * 4);
    let mut st = St {
        gfx12,
        m,
        k,
        n,
        x: xd.tensor(vec![n, k], DType::F32),
        y: yd.tensor(vec![n, m], DType::F32),
        ring_q8,
        ring_a,
        ring_b,
        idx: [0; 3],
        base_acts: None,
        mq_acts: None,
    };
    let pick = mq8_pick_tile(&arch, m, n).label();
    println!(
        "== time {arch} {tag}: tile {pick}, rings q8={} A={} B={} (q8 {:.1} MiB/copy), budget {:.0}s, launches/batch {}",
        st.ring_q8.len(),
        st.ring_a.len(),
        st.ring_b.len(),
        q8.len() as f64 / 1048576.0,
        o.budget.as_secs_f64(),
        o.launches
    );

    // groups of one round
    let mut groups: Vec<Group> = Vec::new();
    if has_a {
        groups.push(Group::Abba(Arm::Base, Arm::A));
    }
    if has_b {
        groups.push(Group::Abba(Arm::Base, Arm::B));
    }
    if has_a && has_b {
        groups.push(Group::Abba(Arm::A, Arm::B));
    }
    groups.push(Group::Abba(Arm::ProdBase, Arm::ProdMq));
    let mut gemm_seq = vec![Arm::GemmBase];
    if has_a {
        gemm_seq.push(Arm::GemmA);
    }
    if has_b {
        gemm_seq.push(Arm::GemmB);
    }
    let mut full = gemm_seq.clone();
    full.extend(gemm_seq.iter().rev());
    groups.push(Group::Seq(full));
    if o.secondary {
        if has_a {
            groups.push(Group::Abba(Arm::Base32, Arm::A32));
        }
        if has_b {
            groups.push(Group::Abba(Arm::Base32, Arm::B32));
        }
    }

    // warm-up of every arm that will run
    let mut warm = vec![Arm::Base, Arm::ProdBase, Arm::ProdMq];
    if has_a {
        warm.push(Arm::A);
    }
    if has_b {
        warm.push(Arm::B);
    }
    if o.secondary {
        warm.push(Arm::Base32);
        if has_a {
            warm.push(Arm::A32);
        }
        if has_b {
            warm.push(Arm::B32);
        }
    }
    for arm in warm {
        for _ in 0..12 {
            run_iter(gpu, &mut st, arm);
        }
        ck(gpu.hip.device_synchronize(), "device_synchronize");
    }
    st.base_acts = Some(base_quantize(gpu, &st));
    st.mq_acts = Some(ck(gpu.mq8_quantize_x(&st.x, n, k), "mq8_quantize_x"));
    for arm in &gemm_seq {
        for _ in 0..12 {
            run_iter(gpu, &mut st, *arm);
        }
    }
    ck(gpu.hip.device_synchronize(), "device_synchronize");

    let mut csv = o.csv_dir.as_ref().map(|d| {
        std::fs::create_dir_all(d).ok();
        let path = format!("{d}/{tag}.csv");
        let fresh = std::fs::metadata(&path).map(|m| m.len() == 0).unwrap_or(true);
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).expect("csv");
        if fresh {
            writeln!(
                f,
                "tag,round,group,pos,arm,what,ms_per_iter,load1,load5,sclk_pre,hw_pre,pw_pre,sclk_post,hw_post,pw_post"
            )
            .ok();
        }
        f
    });

    let start = Instant::now();
    let mut deadline = start + o.budget;
    if let Some(d) = o.deadline {
        deadline = deadline.min(d);
    }
    let mut samples: HashMap<Arm, Vec<f64>> = HashMap::new();
    let mut sclks: HashMap<Arm, Vec<f64>> = HashMap::new();
    let mut ratios: HashMap<(Arm, Arm), Vec<f64>> = HashMap::new();
    let mut rounds_done = 0usize;
    'rounds: for r in 0..o.rounds {
        for gi in 0..groups.len() {
            if Instant::now() >= deadline {
                break 'rounds;
            }
            let g = &groups[(gi + r) % groups.len()];
            let gidx = (gi + r) % groups.len();
            let seq: Vec<Arm> = match g {
                Group::Abba(x, y) => vec![*x, *y, *y, *x],
                Group::Seq(s) => {
                    // prepared activations for the GEMM-alone batches (contiguous with the sequence)
                    st.base_acts = Some(base_quantize(gpu, &st));
                    st.mq_acts = Some(ck(gpu.mq8_quantize_x(&st.x, n, k), "mq8_quantize_x"));
                    s.clone()
                }
            };
            let mut got: Vec<Sample> = Vec::new();
            for (pos, &arm) in seq.iter().enumerate() {
                if Instant::now() >= deadline {
                    break 'rounds; // never start a batch past the deadline; the partial group is dropped
                }
                let (ms, pre, post, load) = batch(gpu, &mut st, arm, o.launches, &o.sysfs);
                if let Some(f) = csv.as_mut() {
                    writeln!(
                        f,
                        "{tag},{r},{gidx},{pos},{},{},{ms:.6},{:.2},{:.2},{},{},{},{},{},{}",
                        arm.name(),
                        match arm {
                            Arm::ProdBase | Arm::ProdMq => "producer",
                            Arm::GemmBase | Arm::GemmA | Arm::GemmB => "gemm",
                            _ => "chain",
                        },
                        load.0,
                        load.1,
                        pre.sclk,
                        pre.hw,
                        pre.pw,
                        post.sclk,
                        post.hw,
                        post.pw
                    )
                    .ok();
                }
                if post.sclk > 0 {
                    sclks.entry(arm).or_default().push(post.sclk as f64);
                }
                got.push(Sample { arm, ms });
            }
            for s in &got {
                samples.entry(s.arm).or_default().push(s.ms);
            }
            if let Group::Abba(x, y) = g {
                let t = |i: usize| got[i].ms;
                ratios
                    .entry((*x, *y))
                    .or_default()
                    .push((t(0) + t(3)) / (t(1) + t(2)));
            }
        }
        rounds_done += 1;
    }
    let res = ShapeResult {
        m,
        k,
        n,
        rounds: rounds_done,
        ms: samples.iter().map(|(a, v)| (*a, median(v))).collect(),
        sclk: sclks.iter().map(|(a, v)| (*a, median(v))).collect(),
        ratio: ratios.iter().map(|(p, v)| (*p, pair_of(v))).collect(),
    };
    scope.free(gpu);
    std::mem::forget(st);
    if rounds_done < o.min_rounds {
        println!(
            "WARN {arch} {tag}: only {rounds_done} complete rounds (< min {}): result is thin",
            o.min_rounds
        );
    }
    print_shape(&arch, &res);
    Some(res)
}

fn fmt_pair(p: Option<&Pair>) -> String {
    match p {
        Some(p) => format!("{:.3} ({:.3} {:.3})", p.med, p.p5, p.p95),
        None => "n/a".to_string(),
    }
}

fn print_shape(arch: &str, r: &ShapeResult) {
    let tag = format!("M{}_K{}_N{}", r.m, r.k, r.n);
    let a_vs_b = r
        .a_cost_pct()
        .map_or("n/a".to_string(), |v| format!("{v:+.2}%"));
    println!(
        "RESULT {arch} {tag} chain base={:.4} mq8A={:.4} mq8B={:.4} speedupA={} speedupB={} A_vs_B={a_vs_b}",
        r.ms(Arm::Base),
        r.ms(Arm::A),
        r.ms(Arm::B),
        fmt_pair(r.speedup(Arm::A)),
        fmt_pair(r.speedup(Arm::B))
    );
    println!(
        "RESULT_PARTS {arch} {tag} producer base={:.4} mq8={:.4} (mq8/base {:.3}x) | gemm-alone base={:.4} mq8A={:.4} mq8B={:.4} | rounds={} sclk_med base={:.0} mq8A={:.0} mq8B={:.0}",
        r.ms(Arm::ProdBase),
        r.ms(Arm::ProdMq),
        r.ms(Arm::ProdMq) / r.ms(Arm::ProdBase),
        r.ms(Arm::GemmBase),
        r.ms(Arm::GemmA),
        r.ms(Arm::GemmB),
        r.rounds,
        r.sclk.get(&Arm::Base).copied().unwrap_or(f64::NAN),
        r.sclk.get(&Arm::A).copied().unwrap_or(f64::NAN),
        r.sclk.get(&Arm::B).copied().unwrap_or(f64::NAN)
    );
    if r.ms.contains_key(&Arm::Base32) {
        println!(
            "RESULT_F32 {arch} {tag} pair(producer+gemm, F32 out) base_f32={:.4} mq8A_f32={:.4} mq8B_f32={:.4} speedupA_f32={} speedupB_f32={}",
            r.ms(Arm::Base32),
            r.ms(Arm::A32),
            r.ms(Arm::B32),
            fmt_pair(r.ratio.get(&(Arm::Base32, Arm::A32))),
            fmt_pair(r.ratio.get(&(Arm::Base32, Arm::B32)))
        );
    }
    let _ = std::io::stdout().flush();
}

/// Final call-weighted prefill-chunk totals and the DECISION block.
fn decide(arch: &str, res: &[ShapeResult], layouts: &[Mq8WLayout]) {
    let has_a = layouts.contains(&Mq8WLayout::A);
    let has_b = layouts.contains(&Mq8WLayout::B);
    let mut ns: Vec<usize> = res.iter().map(|r| r.n).collect();
    ns.sort();
    ns.dedup();
    for n in ns {
        let sel: Vec<&ShapeResult> = res
            .iter()
            .filter(|r| r.n == n && calls_for(r.m, r.k) > 0)
            .collect();
        if sel.is_empty() {
            continue;
        }
        let tot = |arm: Arm| -> f64 {
            sel.iter()
                .map(|r| r.ms(arm) * calls_for(r.m, r.k) as f64)
                .sum::<f64>()
        };
        let (b, a, bb) = (tot(Arm::Base), tot(Arm::A), tot(Arm::B));
        println!(
            "WEIGHTED {arch} N={n} prefill-chunk chain total over {}/4 attention shapes (calls-weighted ms): base={b:.3} mq8A={a:.3} mq8B={bb:.3} speedupA={:.3} speedupB={:.3}",
            sel.len(),
            b / a,
            b / bb
        );
    }
    println!("== DECISION");
    let rule = |layout: &str, arm: Arm| {
        let is12 = arch == "gfx1201" || arch == "gfx1200";
        let (what, thr, subset): (&str, f64, Vec<&ShapeResult>) = if is12 {
            (
                "chain speedup >= 1.30x at every Raon N=4096 shape",
                1.30,
                res.iter()
                    .filter(|r| r.n == 4096 && calls_for(r.m, r.k) > 0)
                    .collect(),
            )
        } else {
            (
                "chain speedup >= 1.00x at every shape (no regression)",
                1.00,
                res.iter().collect(),
            )
        };
        let verdict = if subset.is_empty() {
            "NOT EVALUATED (no shapes timed)".to_string()
        } else if subset
            .iter()
            .all(|r| r.speedup(arm).map_or(false, |p| p.med >= thr))
        {
            "MET".to_string()
        } else {
            "NOT MET".to_string()
        };
        let detail: Vec<String> = subset
            .iter()
            .map(|r| {
                format!(
                    "M{}_K{}_N{}={}",
                    r.m,
                    r.k,
                    r.n,
                    r.speedup(arm)
                        .map_or("n/a".to_string(), |p| format!("{:.3}", p.med))
                )
            })
            .collect();
        println!(
            "DECISION {arch} layout {layout}: {what}: {verdict} [{}]",
            detail.join(" ")
        );
    };
    if has_a {
        rule("A", Arm::A);
    }
    if has_b {
        rule("B", Arm::B);
    }
    if has_a && has_b {
        let mut per: Vec<(String, f64)> = res
            .iter()
            .filter_map(|r| {
                r.a_cost_pct()
                    .map(|p| (format!("M{}_K{}_N{}", r.m, r.k, r.n), p))
            })
            .collect();
        per.sort_by(|x, y| x.0.cmp(&y.0));
        let costs: Vec<f64> = per.iter().map(|p| p.1).collect();
        for (t, p) in &per {
            println!("LAYOUT_COST {arch} {t} A_vs_B={p:+.2}%");
        }
        let med = median(&costs);
        let max = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let b_med = med > 3.0;
        let b_any = max > 3.0;
        let verdict = if b_med || b_any {
            "LAYOUT B"
        } else {
            "LAYOUT A"
        };
        println!(
            "DECISION {arch} layout rule (choose B only if A costs > 3% vs B, median over shapes and per shape): median {med:+.2}% max {max:+.2}% -> {}{}{}",
            verdict,
            if b_med { " [median > 3%]" } else { "" },
            if b_any { " [some shape > 3%]" } else { "" }
        );
        println!("{verdict}");
    } else {
        println!("DECISION {arch} layout rule: not evaluated (need both layouts)");
    }
}

// ---------------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------------

fn usage() -> ! {
    eprintln!(
        "usage: mq8_prefill_lab guard|time|all [--layouts a,b] [--shapes M,K,N;...] [--budget-secs S] [--rounds R]\n\
         \x20      [--csv-dir D] [--launches L] [--min-rounds R] [--deadline-secs S] [--ring-max N] [--sysfs DIR] [--expect-bdf B] [--no-secondary]"
    );
    std::process::exit(2);
}

fn parse_shapes(s: &str) -> Vec<(usize, usize, usize)> {
    s.split(|c| c == ';' || c == ' ')
        .filter(|t| !t.trim().is_empty())
        .map(|t| {
            let v: Vec<usize> = t
                .split(',')
                .map(|x| x.trim().parse().unwrap_or_else(|_| usage()))
                .collect();
            if v.len() != 3 || v[1] % 256 != 0 || v.iter().any(|&x| x == 0) {
                eprintln!("bad shape '{t}': need M,K,N with K % 256 == 0");
                usage()
            }
            (v[0], v[1], v[2])
        })
        .collect()
}

fn main() {
    let t_start = Instant::now();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = match args.first().map(|s| s.as_str()) {
        Some(c @ ("guard" | "time" | "all")) => c.to_string(),
        _ => usage(),
    };
    let mut layouts = vec![Mq8WLayout::A, Mq8WLayout::B];
    let mut shapes: Option<Vec<(usize, usize, usize)>> = None;
    let (mut budget, mut rounds, mut min_rounds, mut launches, mut ring_max) =
        (60.0f64, 60usize, 5usize, 20usize, 16usize);
    let mut csv_dir: Option<String> = None;
    let mut deadline_secs: Option<f64> = None;
    let mut sysfs_arg: Option<String> = None;
    let mut expect_bdf: Option<String> = None;
    let mut secondary = true;
    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        let mut val = || -> String {
            i += 1;
            args.get(i).cloned().unwrap_or_else(|| usage())
        };
        match a {
            "--layouts" => {
                layouts = val()
                    .split(',')
                    .map(|s| match s.trim().to_ascii_lowercase().as_str() {
                        "a" => Mq8WLayout::A,
                        "b" => Mq8WLayout::B,
                        _ => usage(),
                    })
                    .collect();
                layouts.dedup();
            }
            "--shapes" => shapes = Some(parse_shapes(&val())),
            "--budget-secs" => budget = val().parse().unwrap_or_else(|_| usage()),
            "--rounds" => rounds = val().parse().unwrap_or_else(|_| usage()),
            "--min-rounds" => min_rounds = val().parse().unwrap_or_else(|_| usage()),
            "--launches" => launches = val().parse().unwrap_or_else(|_| usage()),
            "--ring-max" => ring_max = val().parse().unwrap_or_else(|_| usage()),
            "--csv-dir" => csv_dir = Some(val()),
            "--deadline-secs" => deadline_secs = Some(val().parse().unwrap_or_else(|_| usage())),
            "--sysfs" => sysfs_arg = Some(val()),
            "--expect-bdf" => expect_bdf = Some(val()),
            "--no-secondary" => secondary = false,
            _ => usage(),
        }
        i += 1;
    }
    if layouts.is_empty() || launches == 0 {
        usage();
    }
    let mut gpu = ck(Gpu::init(), "Gpu::init");
    let arch = gpu.arch.clone();
    println!("Arch: {arch}");
    if !mq8_admits(&arch, 256) {
        eprintln!("arch {arch} is not admitted by the MQ8G256 prefill kernels (gfx12 RDNA4 / gfx1100 / gfx1151)");
        std::process::exit(2);
    }
    let bdf = gpu
        .hip
        .device_pci_bus_id(gpu.device_id)
        .unwrap_or_default()
        .to_ascii_lowercase();
    println!("device {}: pci {bdf}", gpu.device_id);
    if let Some(want) = &expect_bdf {
        let want = want.to_ascii_lowercase();
        if !bdf.ends_with(&want) {
            eprintln!("REFUSED: device pci {bdf} is not the expected {want}");
            std::process::exit(2);
        }
    }
    let sysfs = sysfs_arg.unwrap_or_else(|| format!("/sys/bus/pci/devices/{bdf}"));
    println!("sysfs: {sysfs} (read only)");

    if cmd == "guard" || cmd == "all" {
        if !run_guard(&mut gpu, &layouts, shapes.as_deref()) {
            println!("GUARD FAILED: no timing");
            std::process::exit(3);
        }
    }
    if cmd == "time" || cmd == "all" {
        let production = gpu.arch_caps.is_rdna4() || arch == "gfx1100";
        if !production {
            eprintln!("time: no production Q8 chain is defined for {arch} (gfx12 / gfx1100 only)");
            std::process::exit(2);
        }
        let signs = Signs {
            s1: gen_fwht_signs(42, 256),
            s2: gen_fwht_signs(1042, 256),
        };
        let shapes: Vec<(usize, usize, usize)> = shapes.unwrap_or_else(|| {
            [4096usize, 512]
                .iter()
                .flat_map(|&n| RAON_MK.iter().map(move |&(m, k)| (m, k, n)))
                .collect()
        });
        let opts = TimeOpts {
            layouts: layouts.clone(),
            budget: Duration::from_secs_f64(budget),
            rounds,
            min_rounds,
            launches,
            csv_dir,
            ring_max,
            sysfs,
            secondary,
            deadline: deadline_secs.map(|s| t_start + Duration::from_secs_f64(s)),
        };
        let mut results = Vec::new();
        for (m, k, n) in shapes {
            if let Some(r) = time_shape(&mut gpu, &signs, m, k, n, &opts) {
                results.push(r);
            }
        }
        decide(&arch, &results, &layouts);
        println!("TIME DONE elapsed={:.1}s", t_start.elapsed().as_secs_f64());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signs() -> Signs {
        Signs {
            s1: gen_fwht_signs(42, 256),
            s2: gen_fwht_signs(1042, 256),
        }
    }

    #[test]
    fn sign_tables_are_the_lcg_of_scratch_rs() {
        let s1: Vec<i32> = gen_fwht_signs(42, 16).iter().map(|&v| v as i32).collect();
        let s2: Vec<i32> = gen_fwht_signs(1042, 16).iter().map(|&v| v as i32).collect();
        // hand-run of state = state * 1103515245 + 12345 & 0x7fffffff, sign = bit 16
        assert_eq!(s1, [1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1]);
        assert_eq!(s2, [1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]);
    }

    #[test]
    fn f16_store_truncates_and_decode_is_exact() {
        let halfway_plus = f32::from_bits(1.0f32.to_bits() + 0x1fff);
        assert_eq!(f32_to_f16_trunc(halfway_plus), 0x3c00);
        assert_eq!(f32_to_f16_trunc(1.0), 0x3c00);
        assert_eq!(
            f32_to_f16_trunc(f32::from_bits(1.0f32.to_bits() + 0x2000)),
            0x3c01
        );
        assert_eq!(f32_to_f16_trunc(0.0625 / 127.0), 0x1008); // 4.9212598e-4 = 2^-11 * (1 + 8.06/1024)
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x1008), 2f32.powi(-11) * (1.0 + 8.0 / 1024.0));
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // subnormal decode
    }

    #[test]
    fn bf16_rn_is_ties_to_even_with_passthrough() {
        assert_eq!(bf16_rn(1.0), 1.0);
        assert_eq!(bf16_rn(f32::from_bits(0x3F80_8000)), 1.0); // tie, even stays
        assert_eq!(bf16_rn(f32::from_bits(0x3F81_8000)).to_bits(), 0x3F82_0000); // tie rounds to even (up)
        assert_eq!(bf16_rn(f32::from_bits(0x3F80_8001)).to_bits(), 0x3F81_0000);
        assert!(bf16_rn(f32::NAN).is_nan());
        assert_eq!(bf16_rn(f32::INFINITY), f32::INFINITY);
        assert_eq!(bf16_rn(f32::from_bits(0x7F7F_FFFF)), f32::INFINITY); // RN overflow
    }

    /// Group with w[0] = 1036, w[1] = 996 (s1[0] = s1[1] = +1): the transform gives v[i] = 0.0625 * s2[i] *
    /// (1036 + 996) = 127 * s2[i] for even i and 0.0625 * s2[i] * (1036 - 996) = 2.5 * s2[i] for odd i, so
    /// amax = 127, scale = 1.0 (f16 0x3c00), inv = 1.
    fn tie_group() -> Vec<f32> {
        let mut w = vec![0f32; 256];
        w[0] = 1036.0;
        w[1] = 996.0;
        w
    }

    #[test]
    fn quantizer_port_matches_hand_computed_groups() {
        let s = signs();
        // all-zero group: scale 1.0, zero codes
        let mut out = [0u8; 258];
        quantize_mq8g256_group(&[0.0; 256], &s.s1, &s.s2, &mut out);
        assert_eq!(&out[..2], &[0x00, 0x3c]);
        assert!(out[2..].iter().all(|&b| b == 0));
        // single element 2032 at index 0: v[i] = 2032 * 0.0625 * s2[i] = 127 * s2[i]
        let mut w = vec![0f32; 256];
        w[0] = 2032.0;
        quantize_mq8g256_group(&w, &s.s1, &s.s2, &mut out);
        assert_eq!(&out[..2], &[0x00, 0x3c]);
        for i in 0..256 {
            assert_eq!(out[2 + i] as i8, (127.0 * s.s2[i]) as i8, "elem {i}");
        }
        // tie group: round() (half away from zero) gives |code| 3 on odd elements
        quantize_mq8g256_group(&tie_group(), &s.s1, &s.s2, &mut out);
        assert_eq!(&out[..2], &[0x00, 0x3c]);
        let first8: Vec<i8> = out[2..10].iter().map(|&b| b as i8).collect();
        assert_eq!(first8, [127, -3, -127, -3, -127, 3, 127, -3]);
        for i in 0..256 {
            let mag = if i % 2 == 0 { 127 } else { 3 };
            assert_eq!(out[2 + i] as i8, (mag as f32 * s.s2[i]) as i8, "elem {i}");
        }
        // scale truncation: amax * 2032 / 127 / 16 gives a scale of exactly 0.0625 * 3 / 127 -> f16 truncated
        let mut w = vec![0f32; 256];
        w[0] = 3.0; // v = +-0.1875 everywhere, scale = 0.1875/127 = 1.4763779e-3 = 2^-10 * (1 + 524.09/1024)
        quantize_mq8g256_group(&w, &s.s1, &s.s2, &mut out);
        assert_eq!(u16::from_le_bytes([out[0], out[1]]), (5 << 10) | 524);
        assert!(out[2..].iter().all(|&b| (b as i8).unsigned_abs() == 127));
    }

    #[test]
    fn producer_reference_uses_rintf_and_matches_hand_groups() {
        let s = signs();
        let mut q = [0i8; 256];
        let scale = producer_group(&tie_group(), &s.s1, &s.s2, &mut q);
        assert_eq!(scale, 1.0);
        // rintf(2.5) = 2 (ties to even), the weight quantizer's round() gives 3
        assert_eq!(&q[..8], &[127, -2, -127, -2, -127, 2, 127, -2]);
        // zero group: scale 1.0, zero codes
        let scale = producer_group(&[0.0; 256], &s.s1, &s.s2, &mut q);
        assert_eq!(scale, 1.0);
        assert!(q.iter().all(|&c| c == 0));
        // power-of-two scaled ties stay ties
        let g: Vec<f32> = tie_group().iter().map(|v| v * 0.25).collect();
        let scale = producer_group(&g, &s.s1, &s.s2, &mut q);
        assert_eq!(scale, 0.25);
        assert_eq!(&q[..8], &[127, -2, -127, -2, -127, 2, 127, -2]);
    }

    #[test]
    fn producer_and_weight_quantizer_agree_off_ties() {
        let s = signs();
        let w = gen_w_bf16(1, 512, 3, true);
        let a = quantize_mq8g256_port(&w, &s.s1, &s.s2);
        for g in 0..2 {
            let mut q = [0i8; 256];
            let sc = producer_group(&w[g * 256..g * 256 + 256], &s.s1, &s.s2, &mut q);
            let codes: Vec<i8> = a[g * 258 + 2..(g + 1) * 258]
                .iter()
                .map(|&b| b as i8)
                .collect();
            assert_eq!(codes, q.to_vec(), "group {g}");
            let stored = f16_to_f32(u16::from_le_bytes([a[g * 258], a[g * 258 + 1]]));
            assert!(
                stored <= sc && stored >= sc * (1.0 - 1.0 / 1024.0),
                "truncated f16 scale {stored} vs {sc}"
            );
        }
    }

    #[test]
    fn cpu_producer_scatters_scales_group_major() {
        let s = signs();
        let (n, k) = (3usize, 512usize);
        let x = gen_x(n, k, 1, XMode::Wide);
        let (xq, xs) = cpu_producer(&x, n, k, &s.s1, &s.s2);
        for t in 0..n {
            for g in 0..2 {
                let mut q = [0i8; 256];
                let sc = producer_group(
                    &x[t * k + g * 256..t * k + g * 256 + 256],
                    &s.s1,
                    &s.s2,
                    &mut q,
                );
                assert_eq!(xs[g * n + t].to_bits(), sc.to_bits());
                assert_eq!(&xq[t * k + g * 256..t * k + g * 256 + 256], &q[..]);
            }
        }
    }

    #[test]
    fn gemm_reference_hand_case_and_fma_association() {
        // M = 1, K = 256, N = 1: W codes 1 (scale 2.0), X codes 3 (scale 0.5): dot128 = 384, scale 1.0
        let mut a = vec![0u8; 258];
        a[..2].copy_from_slice(&0x4000u16.to_le_bytes()); // f16 2.0
        a[2..].iter_mut().for_each(|b| *b = 1);
        let w = decode_a(&a, 1, 256);
        assert_eq!(w.scales, [2.0]);
        let xq = vec![3i8; 256];
        let out = cpu_gemm(&w, &xq, &[0.5], 1, &[0]);
        assert_eq!(out, [768.0]); // 384 + 384
                                  // epoch 0: dot = -3, epoch 1: dot = +3 with scale s = 1 + 2^-23; fused fold
                                  // acc0 = RN(-3 s) = -(3 + 2^-22) (tie to even), acc1 = RN(3 s + acc0) = exact 3 s - 3 - 2^-22 (one rounding)
        let s = 1.0f32 + 2f32.powi(-23);
        let mut a = vec![0u8; 258];
        a[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // f16 1.0
        for i in 0..128 {
            a[2 + i] = 0u8; // epoch 0 codes
        }
        // dot = -3 in epoch 0: one weight code -3 against X code 1; epoch 1: +3
        a[2] = (-3i8) as u8;
        a[2 + 128] = 3;
        let w = decode_a(&a, 1, 256);
        let mut xq = vec![0i8; 256];
        xq[0] = 1;
        xq[128] = 1;
        let out = cpu_gemm(&w, &xq, &[s], 1, &[0]);
        let acc0 = (-3.0f64 * s as f64) as f32;
        let want = (3.0f64 * s as f64 + acc0 as f64) as f32;
        assert_eq!(out[0].to_bits(), want.to_bits());
        let unfused = (3.0f32 * s) + acc0;
        assert_ne!(
            unfused.to_bits(),
            want.to_bits(),
            "test must distinguish fma from mul+add"
        );
    }

    #[test]
    fn gemm_reference_samples_tokens_and_group_scale_indexing() {
        // K = 512: groups 0 and 1 carry different scales on both operands
        let s = signs();
        let (m, k, n) = (3usize, 512usize, 4usize);
        let w = gen_w_bf16(m, k, 8, false);
        let a = quantize_mq8g256_port(&w, &s.s1, &s.s2);
        let wd = decode_a(&a, m, k);
        let x = gen_x(n, k, 9, XMode::Wide);
        let (xq, xs) = cpu_producer(&x, n, k, &s.s1, &s.s2);
        let full = cpu_gemm(&wd, &xq, &xs, n, &(0..n).collect::<Vec<_>>());
        let sub = cpu_gemm(&wd, &xq, &xs, n, &[3, 1]);
        assert_eq!(&sub[..m], &full[3 * m..4 * m]);
        assert_eq!(&sub[m..], &full[m..2 * m]);
        // explicit scalar re-derivation of one element (token 2, row 1)
        let (t, r) = (2usize, 1usize);
        let mut acc = 0f32;
        for e in 0..4 {
            let mut dot = 0i32;
            for i in 0..128 {
                dot += wd.codes[r * k + e * 128 + i] as i32 * xq[t * k + e * 128 + i] as i32;
            }
            let g = e / 2;
            acc = f32::mul_add(wd.scales[r * 2 + g] * xs[g * n + t], dot as f32, acc);
        }
        assert_eq!(full[t * m + r].to_bits(), acc.to_bits());
    }

    #[test]
    fn layout_b_builder_places_planes() {
        let s = signs();
        let (m, k) = (5usize, 768usize);
        let a = quantize_mq8g256_port(&gen_w_bf16(m, k, 2, false), &s.s1, &s.s2);
        let b = repack_b_ref(&a, m, k);
        let code_off = (m * 3 * 2).next_multiple_of(256);
        assert_eq!(code_off, 256);
        assert_eq!(b.len(), code_off + m * k);
        assert_eq!(&b[2 * 7..2 * 7 + 2], &a[7 * 258..7 * 258 + 2]);
        assert_eq!(
            &b[code_off + 7 * 256..code_off + 8 * 256],
            &a[7 * 258 + 2..8 * 258]
        );
    }

    #[test]
    fn q8_0_port_blocks() {
        let mut w = vec![0f32; 32];
        w[0] = 127.0;
        w[1] = -63.5;
        let q = quantize_q8_0(&w);
        assert_eq!(q.len(), 34);
        assert_eq!(&q[..2], &0x3c00u16.to_le_bytes()); // scale 1.0
        assert_eq!(q[2] as i8, 127);
        assert_eq!(q[3] as i8, -64); // round half away from zero
        assert!(quantize_q8_0(&[0.0; 32])[2..].iter().all(|&b| b == 0));
    }

    #[test]
    fn statistics_helpers() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!((quantile(&[0.0, 10.0], 0.05) - 0.5).abs() < 1e-12);
        assert!((quantile(&[1.0, 2.0, 3.0], 0.95) - 2.9).abs() < 1e-12);
    }

    #[test]
    fn sampled_rows_compare_by_token() {
        let m = 2;
        let got = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let want = vec![4.0, 5.0, 0.0, 1.0];
        assert!(cmp_rows(&got, m, &want, Some(&[2, 0])).is_ok());
        let bad = vec![4.0, 6.0, 0.0, 1.0];
        let e = cmp_rows(&got, m, &bad, Some(&[2, 0])).unwrap_err();
        assert!(e.contains("token 2 row 1"), "{e}");
    }

    #[test]
    fn shape_parser() {
        assert_eq!(
            parse_shapes("5120,2560,4096;576,2560,512"),
            vec![(5120, 2560, 4096), (576, 2560, 512)]
        );
    }

    #[test]
    fn data_generators_cover_the_edge_classes() {
        let x = gen_x(70, 512, 4, XMode::Wide);
        let zero_rows = (0..70)
            .filter(|&r| x[r * 512..(r + 1) * 512].iter().all(|&v| v == 0.0))
            .count();
        assert!(zero_rows >= 1);
        let imp = gen_x(4, 256, 0, XMode::Impulse);
        assert_eq!(imp.iter().filter(|&&v| v != 0.0).count(), 4);
        let s = signs();
        let raw = gen_w_raw(8, 512, 1, RawMode::Random, &s.s2);
        assert!(raw
            .chunks(258)
            .any(|g| g[2..].iter().any(|&b| b as i8 == -128)));
        let w = gen_w_bf16(4, 256, 1, true);
        assert!(w.iter().all(|v| v.to_bits() & 0xFFFF == 0));
    }
}
