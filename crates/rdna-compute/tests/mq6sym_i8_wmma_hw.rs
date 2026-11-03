// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KERNEL-PLAN T1: byte gate + timing for the symmetric MQ6 (MQ6SymG256) signed-IU8 WMMA
//! GEMM `kernels/src/gemm_mq6sym_i8.gfx1201.hip` (verify tiles N <= 48 and the 128x128
//! prefill tile).
//!
//! Integer lane: the oracle is an independent scalar CPU model of KERNEL-PLAN §1.4: per
//! 128-K block the exact i32 `C = sum (q-32) * qx`, `sd = RN(f32(s_h) * d)`,
//! `acc = fma(sd, C as f32, acc)` in increasing block order from +0.0, F32 store or the
//! incumbent RNE BF16 conversion. It never reads the kernel's fold; it consumes the real
//! `block_i8_128` bytes (copied back from the real A8 producer scratch or crafted).
//!
//! Tests (`#[ignore]`, exact gfx1201; `--test-threads=1 --nocapture`):
//!   mq6sym_i8_wmma_bytes_hw        correctness: every tile entry, F32 + BF16, tails, guards
//!   mq6sym_i8_wmma_timing_hw       cold cyclic timing of tiles vs affine / dot4-style arms
//! Env: `T1_OUT=<dir>` also writes the JSON lines to `<dir>/<test>.jsonl`;
//! `T1_CELLS` (timing) = comma list of `name` filters; `T1_BIG=1` adds N=16384 / lm_head.
//! Competitor sources (read at run time, arm skipped when absent):
//!   `T1_SKINNY_SRC` parked DenseR32 skinny MQ6 F16 WMMA (commit cce88f1245),
//!   `T1_DOT4_SRC` + `T1_DOT4_ENTRY` same-format dot4-style kernel (T0 slice).

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;
use std::io::Write;

const MODULE: &str = "gemm_mq6sym_i8_gfx1201_t1";
const SYM_SRC: &str = concat!(
    "#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n",
    include_str!("../../../kernels/src/block_i8_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq6sym_i8.gfx1201.hip")
);
/// The standalone reference quantizer (`quantize_int8_mmq_ds128`) of the same header.
const QUANT_MODULE: &str = "block_i8_128_quant_t1";
const QUANT_SRC: &str = include_str!("../../../kernels/src/block_i8_128_quant.hip");
const AFFINE_MODULE: &str = "gemm_mq6g256v2_mmq_i8_gfx12_t1";
const AFFINE_ENTRY: &str = "gemm_mq6g256v2_mmq_i8_set_gfx12";
const AFFINE_BF16_ENTRY: &str = "gemm_mq6g256v2_mmq_i8_set_bf16_gfx12";
/// Same-job streaming roofline: a plain grid-stride 16-byte read of the same weight bytes.
const STREAM_MODULE: &str = "t1_stream_read";
const STREAM_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void t1_stream_read(const uint4* __restrict__ p, unsigned long long n16, unsigned* out) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    uint4 a = make_uint4(0, 0, 0, 0);
    for (; i < n16; i += stride) {
        const uint4 v = p[i];
        a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
    }
    if ((a.x ^ a.y ^ a.z ^ a.w) == 0x12345678u) out[0] = 1;
}
"#;

const GROUP: usize = 196;
const AFFINE_GROUP: usize = 200;
const BLK: usize = 136;
const KERNARG_BYTES: usize = 36;
const P128_LDS: u32 = 36864;
const GUARD_LEAD: usize = 64;
const GUARD_TAIL: usize = 256;

// ---------------------------------------------------------------- RNG / numerics

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) ^ self.next_u32() as u64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }
    fn f32_signed(&mut self) -> f32 {
        let a = self.next_u32() & 0x7F_FFFF;
        let v = f32::from_bits(0x3F80_0000 | a) - 1.0;
        if self.next_u32() & 1 == 1 {
            -v
        } else {
            v
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 31) as u32;
    let m = (h & 0x3FF) as u32;
    if e == 0 {
        let v = (m as f32) * (1.0 / 16_777_216.0);
        return if s != 0 { -v } else { v };
    }
    if e == 31 {
        return f32::from_bits(s | 0x7F80_0000 | (m << 13));
    }
    f32::from_bits(s | ((e + 112) << 23) | (m << 13))
}

fn bf16_bits(y: f32) -> u16 {
    let v = 0.0f32 + y;
    let u = v.to_bits();
    if v.is_finite() {
        (u.wrapping_add(0x7FFF).wrapping_add((u >> 16) & 1) >> 16) as u16
    } else {
        (u >> 16) as u16
    }
}

// ---------------------------------------------------------------- format

fn pack_group(s0: u16, s1: u16, codes: &[u8; 256], out: &mut [u8]) {
    out[0..2].copy_from_slice(&s0.to_le_bytes());
    out[2..4].copy_from_slice(&s1.to_le_bytes());
    for i in 0..64 {
        let t = (codes[4 * i] as u32 & 63)
            | ((codes[4 * i + 1] as u32 & 63) << 6)
            | ((codes[4 * i + 2] as u32 & 63) << 12)
            | ((codes[4 * i + 3] as u32 & 63) << 18);
        out[4 + 3 * i] = (t & 255) as u8;
        out[4 + 3 * i + 1] = ((t >> 8) & 255) as u8;
        out[4 + 3 * i + 2] = ((t >> 16) & 255) as u8;
    }
}

/// One row of `k` weights -> (signed codes q-32, f32 scale per 128-K block).
fn decode_row(row: &[u8], k: usize, codes: &mut [i8], scales: &mut [f32]) {
    for g in 0..k / 256 {
        let gb = &row[g * GROUP..(g + 1) * GROUP];
        scales[2 * g] = f16_to_f32(u16::from_le_bytes([gb[0], gb[1]]));
        scales[2 * g + 1] = f16_to_f32(u16::from_le_bytes([gb[2], gb[3]]));
        for i in 0..64 {
            let t = gb[4 + 3 * i] as u32
                | ((gb[4 + 3 * i + 1] as u32) << 8)
                | ((gb[4 + 3 * i + 2] as u32) << 16);
            for c in 0..4 {
                codes[g * 256 + 4 * i + c] = ((t >> (6 * c)) & 63) as i32 as i8 - 32;
            }
        }
    }
}

/// Natural-order int8 of one block_i8_128 (the producer's nibble-pair byte order undone).
fn block_qx(block: &[u8]) -> [i8; 128] {
    let qs = &block[8..136];
    let mut q = [0i8; 128];
    for g in 0..16 {
        for i in 0..4 {
            q[8 * g + 2 * i] = qs[8 * g + i] as i8;
            q[8 * g + 2 * i + 1] = qs[8 * g + 4 + i] as i8;
        }
    }
    q
}

fn make_block(d: f32, q: &[i8; 128]) -> [u8; BLK] {
    let mut b = [0u8; BLK];
    b[0..4].copy_from_slice(&d.to_le_bytes());
    let s: i32 = q.iter().map(|&v| v as i32).sum();
    b[4..8].copy_from_slice(&s.to_le_bytes());
    for g in 0..16 {
        for i in 0..4 {
            b[8 + 8 * g + i] = q[8 * g + 2 * i] as u8;
            b[8 + 8 * g + 4 + i] = q[8 * g + 2 * i + 1] as u8;
        }
    }
    b
}

/// CPU model of `quantize_block_i8_128_wave` (block_i8_128_quant.hip contract).
fn quantize_block_cpu(x: &[f32]) -> [u8; BLK] {
    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let mut q = [0i8; 128];
    let d;
    if amax == 0.0 {
        d = 0.0f32;
    } else {
        d = amax / 127.0f32;
        for i in 0..128 {
            let r = (x[i] / d).round_ties_even();
            q[i] = r.clamp(-127.0, 127.0) as i32 as i8;
        }
    }
    make_block(d, &q)
}

/// `[K/128][N]` blocks of `x` (`[N][K]`, already in the rotated basis).
fn quantize_cpu(x: &[f32], k: usize, n: usize) -> Vec<u8> {
    let kb = k / 128;
    let mut out = vec![0u8; kb * n * BLK];
    for t in 0..n {
        for b in 0..kb {
            let blk = quantize_block_cpu(&x[t * k + b * 128..t * k + (b + 1) * 128]);
            out[(b * n + t) * BLK..(b * n + t + 1) * BLK].copy_from_slice(&blk);
        }
    }
    out
}

// ---------------------------------------------------------------- weights

#[derive(Clone, Copy, Debug)]
enum WKind {
    /// uniform random codes, normal f16 scales (some zero)
    Random,
    /// edge scales: 0, +-subnormals, min normal, 1, max, negative; random codes
    EdgeScales,
    /// rows 0..63 hold code[p] = (p + r) & 63 (every code at every position), rest random
    Exhaustive,
    /// all codes 0 (-32) with scale 1 (extreme C against +-127)
    AllMin,
    /// all codes 63 (+31)
    AllMax,
    /// scale exactly 1.0 for both halves, random codes (C-probe companion)
    UnitScale,
}

const EDGE_SCALES: [u16; 14] = [
    0x0000, 0x8000, 0x0001, 0x0002, 0x03FF, 0x8155, 0x0400, 0x3C00, 0xBC00, 0x3800, 0x7BFF,
    0x2E66, 0x5640, 0x0155,
];

fn gen_weights(m: usize, k: usize, kind: WKind, seed: u64) -> Vec<u8> {
    let gpr = k / 256;
    let mut rng = Lcg(seed);
    let mut w = vec![0u8; m * gpr * GROUP];
    for r in 0..m {
        for g in 0..gpr {
            let mut codes = [32u8; 256];
            let (s0, s1);
            match kind {
                WKind::Random | WKind::EdgeScales | WKind::UnitScale | WKind::Exhaustive => {
                    for c in codes.iter_mut() {
                        *c = (rng.next_u32() & 63) as u8;
                    }
                    if let WKind::Exhaustive = kind {
                        if r < 64 {
                            for (p, c) in codes.iter_mut().enumerate() {
                                *c = ((p + r + g * 7) & 63) as u8;
                            }
                        }
                    }
                    let sc = |rng: &mut Lcg| -> u16 {
                        match kind {
                            WKind::EdgeScales => EDGE_SCALES[rng.below(EDGE_SCALES.len())],
                            WKind::UnitScale => 0x3C00,
                            _ => {
                                if rng.below(40) == 0 {
                                    0
                                } else {
                                    let e = 6 + rng.below(14) as u16;
                                    ((rng.next_u32() & 1) as u16) << 15
                                        | (e << 10)
                                        | (rng.next_u32() & 0x3FF) as u16
                                }
                            }
                        }
                    };
                    s0 = sc(&mut rng);
                    s1 = sc(&mut rng);
                }
                WKind::AllMin => {
                    codes = [0u8; 256];
                    s0 = 0x3C00;
                    s1 = 0x3C00;
                }
                WKind::AllMax => {
                    codes = [63u8; 256];
                    s0 = 0x3C00;
                    s1 = 0x3C00;
                }
            }
            let off = (r * gpr + g) * GROUP;
            pack_group(s0, s1, &codes, &mut w[off..off + GROUP]);
        }
    }
    w
}

// ---------------------------------------------------------------- CPU reference

struct RefTokens {
    toks: Vec<usize>,
    qx: Vec<i8>, // [toks][k]
    d: Vec<f32>, // [toks][kb]
}

fn ref_tokens(xq: &[u8], k: usize, n: usize, toks: &[usize]) -> RefTokens {
    let kb = k / 128;
    let mut qx = vec![0i8; toks.len() * k];
    let mut d = vec![0f32; toks.len() * kb];
    for (ti, &t) in toks.iter().enumerate() {
        for b in 0..kb {
            let blk = &xq[(b * n + t) * BLK..(b * n + t + 1) * BLK];
            d[ti * kb + b] = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
            qx[ti * k + b * 128..ti * k + (b + 1) * 128].copy_from_slice(&block_qx(blk));
        }
    }
    RefTokens {
        toks: toks.to_vec(),
        qx,
        d,
    }
}

/// `Y_ref[row_i][tok_i]` (F32) for the listed rows / tokens.
fn cpu_ref(w: &[u8], k: usize, rows: &[usize], rt: &RefTokens) -> Vec<f32> {
    let gpr = k / 256;
    let kb = k / 128;
    let nt = rt.toks.len();
    let mut out = vec![0f32; rows.len() * nt];
    let nthreads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(4)
        .min(16);
    let chunk = rows.len().div_ceil(nthreads).max(1);
    std::thread::scope(|sc| {
        for (ci, (rows_c, out_c)) in rows.chunks(chunk).zip(out.chunks_mut(chunk * nt)).enumerate() {
            let _ = ci;
            sc.spawn(move || {
                let mut codes = vec![0i8; k];
                let mut scales = vec![0f32; kb];
                for (ri, &r) in rows_c.iter().enumerate() {
                    decode_row(&w[r * gpr * GROUP..(r + 1) * gpr * GROUP], k, &mut codes, &mut scales);
                    for ti in 0..nt {
                        let mut acc = 0.0f32;
                        for b in 0..kb {
                            let a = &codes[b * 128..(b + 1) * 128];
                            let x = &rt.qx[ti * k + b * 128..ti * k + (b + 1) * 128];
                            let c: i32 = a.iter().zip(x).map(|(&p, &q)| p as i32 * q as i32).sum();
                            let sd = scales[b] * rt.d[ti * kb + b];
                            acc = sd.mul_add(c as f32, acc);
                        }
                        out_c[ri * nt + ti] = acc;
                    }
                }
            });
        }
    });
    out
}

/// Sum over blocks of the exact i32 C (the C-probe companion: scales 1.0, d = 1.0).
fn cpu_c_sum(w: &[u8], k: usize, rows: &[usize], rt: &RefTokens) -> Vec<i64> {
    let gpr = k / 256;
    let kb = k / 128;
    let nt = rt.toks.len();
    let mut out = vec![0i64; rows.len() * nt];
    let mut codes = vec![0i8; k];
    let mut scales = vec![0f32; kb];
    for (ri, &r) in rows.iter().enumerate() {
        decode_row(&w[r * gpr * GROUP..(r + 1) * gpr * GROUP], k, &mut codes, &mut scales);
        for ti in 0..nt {
            let mut tot = 0i64;
            for b in 0..kb {
                for i in 0..128 {
                    tot += codes[b * 128 + i] as i64 * rt.qx[ti * k + b * 128 + i] as i64;
                }
            }
            out[ri * nt + ti] = tot;
        }
    }
    out
}

fn pick_subset(total: usize, full_limit: usize, extra: &[usize], rng: &mut Lcg, rand_n: usize) -> Vec<usize> {
    if total <= full_limit {
        return (0..total).collect();
    }
    let mut v: Vec<usize> = Vec::new();
    for i in 0..8.min(total) {
        v.push(i);
    }
    for i in total.saturating_sub(24)..total {
        v.push(i);
    }
    for &e in extra {
        if e < total {
            v.push(e);
        }
    }
    for _ in 0..rand_n {
        v.push(rng.below(total));
    }
    v.sort_unstable();
    v.dedup();
    v
}

// ---------------------------------------------------------------- GPU helpers

fn upload_bytes(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[data.len()], DType::Raw).expect("alloc raw");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod");
    t
}

fn download_bytes(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("dtoh");
    out
}

fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu.alloc_tensor(&[data.len()], DType::F32).expect("alloc f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod f32");
    t
}

#[derive(Clone, Copy, Debug)]
struct Tile {
    /// weight rows per CTA (narrow) / 128 (prefill)
    rows: usize,
    /// token capacity of the tile (16/32/48; 0 = prefill, any N)
    ntok: usize,
    /// token-split grid ([ceil(M/rows), ceil(N/ntok)]): any N, the slices re-stream the weights
    ys: bool,
    /// prefill only: LDS-only main-loop barrier (`_fb` entry)
    fb: bool,
}

impl Tile {
    fn narrow(rows: usize, ntok: usize) -> Tile {
        Tile { rows, ntok, ys: false, fb: false }
    }
    fn split(rows: usize, ntok: usize) -> Tile {
        Tile { rows, ntok, ys: true, fb: false }
    }
    fn p128() -> Tile {
        Tile { rows: 128, ntok: 0, ys: false, fb: false }
    }
    fn p128fb() -> Tile {
        Tile { rows: 128, ntok: 0, ys: false, fb: true }
    }
    fn is_prefill(&self) -> bool {
        self.ntok == 0
    }
    fn entry(&self, bf16: bool) -> String {
        let b = if bf16 { "_bf16" } else { "" };
        if self.is_prefill() {
            format!("gemm_mq6sym_i8_set{b}{}_gfx1201", if self.fb { "_fb" } else { "" })
        } else {
            let y = if self.ys { "y" } else { "" };
            format!("gemm_mq6sym_i8_set{b}_m{}n{}{y}_gfx1201", self.rows, self.ntok)
        }
    }
    fn label(&self) -> String {
        if self.is_prefill() {
            (if self.fb { "p128fb" } else { "p128" }).to_string()
        } else {
            format!("m{}n{}{}", self.rows, self.ntok, if self.ys { "y" } else { "" })
        }
    }
    fn grid(&self, m: usize, n: usize) -> [u32; 3] {
        if self.is_prefill() {
            [m.div_ceil(128) as u32, n.div_ceil(128) as u32, 1]
        } else if self.ys {
            [m.div_ceil(self.rows) as u32, n.div_ceil(self.ntok) as u32, 1]
        } else {
            [m.div_ceil(self.rows) as u32, 1, 1]
        }
    }
    fn block(&self) -> [u32; 3] {
        if self.is_prefill() {
            [256, 1, 1]
        } else {
            [(self.rows / 16 * 32) as u32, 1, 1]
        }
    }
    fn lds(&self) -> u32 {
        if self.is_prefill() {
            P128_LDS
        } else {
            0
        }
    }
    fn admits(&self, n: usize) -> bool {
        self.is_prefill() || self.ys || n <= self.ntok
    }
}

fn narrow_tiles() -> Vec<Tile> {
    let mut v = Vec::new();
    for rows in [32usize, 64] {
        for ntok in [16usize, 32, 48] {
            v.push(Tile::narrow(rows, ntok));
        }
        for ntok in [16usize, 32] {
            v.push(Tile::split(rows, ntok));
        }
    }
    v
}

fn load_entries(gpu: &mut Gpu) {
    let mut tiles = narrow_tiles();
    tiles.push(Tile::p128());
    tiles.push(Tile::p128fb());
    for t in tiles {
        for bf16 in [false, true] {
            let e = t.entry(bf16);
            gpu.ensure_kernel_public(MODULE, SYM_SRC, &e)
                .unwrap_or_else(|err| panic!("load {e}: {err}"));
        }
    }
}

fn launch_sym(gpu: &Gpu, tile: Tile, bf16: bool, w: *mut c_void, xq: *mut c_void, y: *mut c_void, m: usize, k: usize, n: usize) {
    let mut a = KernargBlob::new();
    a.push_ptr(w);
    a.push_ptr(xq);
    a.push_ptr(y);
    a.push_i32(m as i32);
    a.push_i32(k as i32);
    a.push_i32(n as i32);
    assert_eq!(a.len(), KERNARG_BYTES, "frozen ABI: 36 explicit kernarg bytes");
    a.pad_to(16);
    let e = tile.entry(bf16);
    gpu.launch_kernel_blob(&e, tile.grid(m, n), tile.block(), tile.lds(), a.as_mut_slice())
        .unwrap_or_else(|err| panic!("launch {e}: {err}"));
}

fn launch_affine_src(gpu: &Gpu, bf16: bool, w: *mut c_void, xq: *mut c_void, y: *mut c_void, m: usize, k: usize, n: usize) {
    let mut a = KernargBlob::new();
    a.push_ptr(w);
    a.push_ptr(xq);
    a.push_ptr(y);
    a.push_i32(m as i32);
    a.push_i32(k as i32);
    a.push_i32(n as i32);
    a.pad_to(16);
    let e = if bf16 { AFFINE_BF16_ENTRY } else { AFFINE_ENTRY };
    gpu.launch_kernel_blob(
        e,
        [m.div_ceil(128) as u32, n.div_ceil(128) as u32, 1],
        [256, 1, 1],
        P128_LDS,
        a.as_mut_slice(),
    )
    .unwrap_or_else(|err| panic!("launch {e}: {err}"));
}

struct Jsonl {
    file: Option<std::fs::File>,
}
impl Jsonl {
    fn new(name: &str) -> Jsonl {
        let file = std::env::var("T1_OUT").ok().map(|d| {
            std::fs::create_dir_all(&d).ok();
            std::fs::File::create(format!("{d}/{name}.jsonl")).expect("T1_OUT file")
        });
        Jsonl { file }
    }
    fn line(&mut self, s: &str) {
        println!("T1JSON {s}");
        if let Some(f) = self.file.as_mut() {
            writeln!(f, "{s}").ok();
        }
    }
}

fn gate(gpu: &Gpu) -> bool {
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: requires exact gfx1201.");
        return false;
    }
    true
}

/// Real A8 producer on `x` (`[n][k]`, F32): returns (device scratch ptr, rotated f32 rows,
/// host copy of the blocks).
fn produce_real(gpu: &mut Gpu, x: &[f32], k: usize, n: usize) -> (*mut c_void, Vec<f32>, Vec<u8>) {
    let xt = upload_f32(gpu, x);
    let xrot = gpu.alloc_tensor(&[k * n], DType::F32).expect("alloc xrot");
    let res = gpu.reserve_int8_mmq(k, n).expect("reserve int8 mmq");
    let prep = gpu
        .rotate_x_mq_i8_batched(&xt, Some(&xrot), res, k, n)
        .expect("real producer");
    let ptr = gpu.int8_mmq_prepared_ptr(&prep, k, n).expect("prepared ptr");
    gpu.hip.device_synchronize().expect("sync");
    let nbytes = (k / 128) * n * BLK;
    let mut host = vec![0u8; nbytes];
    let copy = gpu.alloc_tensor(&[nbytes], DType::Raw).expect("alloc copy");
    unsafe {
        gpu.hip
            .memcpy_dtod_raw(copy.buf.as_ptr(), ptr as *const c_void, nbytes)
            .expect("dtod");
    }
    gpu.hip.memcpy_dtoh(&mut host, &copy.buf).expect("dtoh xq");
    let rot_bytes = download_bytes(gpu, &xrot, k * n * 4);
    let rot: Vec<f32> = rot_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (ptr, rot, host)
}

fn rand_activations(rng: &mut Lcg, k: usize, n: usize) -> Vec<f32> {
    let mut x = vec![0f32; k * n];
    for t in 0..n {
        let mag = match t % 7 {
            0 => 1.0f32,
            1 => 0.01,
            2 => 30.0,
            3 => 1e-4,
            4 => 300.0,
            5 => 3.0,
            _ => 0.3,
        };
        for i in 0..k {
            x[t * k + i] = rng.f32_signed() * mag;
        }
        if t % 11 == 5 {
            // one outlier per row (saturates the +-127 end of a block)
            x[t * k + rng.below(k)] = 400.0 * mag;
        }
    }
    if n > 3 {
        for i in 0..k {
            x[2 * k + i] = 0.0; // an exactly zero token (d = 0 blocks)
        }
    }
    x
}

struct CaseStats {
    words: u64,
    mismatches: u64,
    cases: u32,
    fails: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &mut Gpu,
    st: &mut CaseStats,
    js: &mut Jsonl,
    label: &str,
    tile: Tile,
    bf16: bool,
    w_host: &[u8],
    w_dev: &GpuTensor,
    xq_ptr: *mut c_void,
    xq_host: &[u8],
    m: usize,
    k: usize,
    n: usize,
    rng: &mut Lcg,
    unit_probe: bool,
    neg: bool,
) -> bool {
    let elt = if bf16 { 2 } else { 4 };
    let ybytes = GUARD_LEAD + m * n * elt + GUARD_TAIL;
    let ya = gpu.alloc_tensor(&[ybytes], DType::Raw).expect("alloc y");
    let yb = gpu.alloc_tensor(&[ybytes], DType::Raw).expect("alloc y2");
    gpu.hip.memset(&ya.buf, 0xA5, ybytes).expect("memset a");
    gpu.hip.memset(&yb.buf, 0x5A, ybytes).expect("memset b");
    let ya_ptr = unsafe { (ya.buf.as_ptr() as *mut u8).add(GUARD_LEAD) as *mut c_void };
    let yb_ptr = unsafe { (yb.buf.as_ptr() as *mut u8).add(GUARD_LEAD) as *mut c_void };
    launch_sym(gpu, tile, bf16, w_dev.buf.as_ptr(), xq_ptr, ya_ptr, m, k, n);
    launch_sym(gpu, tile, bf16, w_dev.buf.as_ptr(), xq_ptr, yb_ptr, m, k, n);
    gpu.hip.device_synchronize().expect("sync");
    let ha = download_bytes(gpu, &ya, ybytes);
    let hb = download_bytes(gpu, &yb, ybytes);
    let vis = m * n * elt;
    let mut bad = 0u64;
    let mut first = String::new();
    // guards
    for i in 0..GUARD_LEAD {
        if ha[i] != 0xA5 || hb[i] != 0x5A {
            bad += 1;
        }
    }
    for i in GUARD_LEAD + vis..ybytes {
        if ha[i] != 0xA5 || hb[i] != 0x5A {
            bad += 1;
        }
    }
    if bad != 0 {
        first = format!("guard bytes clobbered: {bad}");
    }
    // every visible word written (poison A run == poison B run)
    let mut unwritten = 0u64;
    for i in 0..vis {
        if ha[GUARD_LEAD + i] != hb[GUARD_LEAD + i] {
            unwritten += 1;
        }
    }
    if unwritten != 0 && first.is_empty() {
        first = format!("{unwritten} visible bytes differ between poison A/B runs");
    }
    bad += unwritten;
    // CPU reference on (sub)sets
    let rows = pick_subset(m, 600, &[127, 128, 129, 255, 256, 1023, 1024], rng, 96);
    let mut tok_extra: Vec<usize> = Vec::new();
    for b in [16usize, 32, 48, 128, 256, 512, 1024, 2048, 4096, 8192] {
        for d in [-1i64, 0, 1] {
            let v = b as i64 + d;
            if v >= 0 {
                tok_extra.push(v as usize);
            }
        }
    }
    let toks = pick_subset(n, 64, &tok_extra, rng, 40);
    let rt = ref_tokens(xq_host, k, n, &toks);
    let yref = cpu_ref(w_host, k, &rows, &rt);
    let mut cmp = 0u64;
    let mut mism = 0u64;
    let mut nonzero_ref = 0u64;
    for (ri, &r) in rows.iter().enumerate() {
        for (ti, &t) in toks.iter().enumerate() {
            let want = yref[ri * toks.len() + ti];
            let off = GUARD_LEAD + (t * m + r) * elt;
            let got_eq = if bf16 {
                let g = u16::from_le_bytes([ha[off], ha[off + 1]]);
                g == bf16_bits(want)
            } else {
                let g = u32::from_le_bytes([ha[off], ha[off + 1], ha[off + 2], ha[off + 3]]);
                g == want.to_bits()
            };
            cmp += 1;
            if want != 0.0 {
                nonzero_ref += 1;
            }
            if !got_eq {
                mism += 1;
                if first.is_empty() {
                    first = format!("row {r} tok {t} want {want:e} ({:#010x}) got bytes {:02x?}", want.to_bits(), &ha[off..off + elt]);
                }
            }
        }
    }
    if unit_probe {
        // C-probe: scales 1.0 and d = 1.0 -> the output is the exact integer sum of the
        // per-block C values; compared against an independent i64 sum, not the fold.
        let sums = cpu_c_sum(w_host, k, &rows, &rt);
        for (ri, &r) in rows.iter().enumerate() {
            for (ti, &t) in toks.iter().enumerate() {
                let want = sums[ri * toks.len() + ti] as f32;
                let off = GUARD_LEAD + (t * m + r) * elt;
                let ok = if bf16 {
                    u16::from_le_bytes([ha[off], ha[off + 1]]) == bf16_bits(want)
                } else {
                    u32::from_le_bytes([ha[off], ha[off + 1], ha[off + 2], ha[off + 3]])
                        == want.to_bits()
                };
                cmp += 1;
                if !ok {
                    mism += 1;
                    if first.is_empty() {
                        first = format!("C-probe row {r} tok {t} want {} got {:02x?}", sums[ri * toks.len() + ti], &ha[off..off + elt]);
                    }
                }
            }
        }
    }
    let pass = mism == 0 && bad == 0;
    if !neg {
        st.cases += 1;
        st.words += cmp + (vis as u64) / (elt as u64);
        st.mismatches += mism + bad;
        if !pass {
            st.fails.push(format!("{label} {} {} m{m} k{k} n{n}: {first}", tile.label(), if bf16 { "bf16" } else { "f32" }));
        }
        if nonzero_ref * 4 < cmp && !matches!(label, "crafted/AllMin" | "crafted/AllMax") {
            // a mostly-zero reference would make the byte compare vacuous
            let zero_scale_heavy = label.contains("EdgeScales") || label.contains("Exhaustive");
            if !zero_scale_heavy {
                st.fails.push(format!("{label} {} m{m} k{k} n{n}: reference mostly zero ({nonzero_ref}/{cmp})", tile.label()));
            }
        }
    }
    js.line(&format!(
        "{{\"kind\":\"{}\",\"label\":\"{label}\",\"tile\":\"{}\",\"out\":\"{}\",\"m\":{m},\"k\":{k},\"n\":{n},\"cpu_words\":{cmp},\"nonzero_ref\":{nonzero_ref},\"mismatch\":{mism},\"guard_or_unwritten\":{bad},\"pass\":{pass}}}",
        if neg { "negative_control" } else { "case" },
        tile.label(),
        if bf16 { "bf16" } else { "f32" }
    ));
    pass
}

fn tiles_for(n: usize, prefill_ok: bool) -> Vec<Tile> {
    let mut v: Vec<Tile> = narrow_tiles().into_iter().filter(|t| t.admits(n)).collect();
    if prefill_ok {
        v.push(Tile::p128());
        v.push(Tile::p128fb());
    }
    v
}

// ---------------------------------------------------------------- correctness test

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn mq6sym_i8_wmma_bytes_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    if !gate(&gpu) {
        return;
    }
    load_entries(&mut gpu);
    gpu.ensure_kernel_public(QUANT_MODULE, QUANT_SRC, "quantize_int8_mmq_ds128")
        .expect("load standalone quantizer");
    let mut js = Jsonl::new("bytes");
    let mut st = CaseStats {
        words: 0,
        mismatches: 0,
        cases: 0,
        fails: Vec::new(),
    };
    let mut rng = Lcg(0x71_0000_C0DE);
    let big = std::env::var("T1_BIG").is_ok();
    let t0 = std::time::Instant::now();

    // ---- A. producer blocks vs the CPU quantizer (real producer, with edge tokens) ----
    for (k, n) in [(256usize, 16usize), (2560, 33), (6144, 9)] {
        let x = rand_activations(&mut rng, k, n);
        let (_ptr, rot, host) = produce_real(&mut gpu, &x, k, n);
        let want = quantize_cpu(&rot, k, n);
        let diff = host.iter().zip(&want).filter(|(a, b)| a != b).count();
        let pass = diff == 0;
        js.line(&format!("{{\"kind\":\"producer_vs_cpu_quantizer\",\"k\":{k},\"n\":{n},\"bytes\":{},\"diff\":{diff},\"pass\":{pass}}}", host.len()));
        if !pass {
            st.fails.push(format!("producer blocks differ from CPU quantizer k{k} n{n}: {diff} bytes"));
            st.mismatches += diff as u64;
        }
    }

    // ---- B. crafted rotated-basis rows: standalone device quantizer == CPU quantizer ----
    {
        let (k, n) = (512usize, 8usize);
        let mut x = vec![0f32; k * n];
        // 0: zero; 1: spike (amax 127, rest 1e-3); 2: ties at +-(j+0.5) with d == 1; 3: saturation
        for i in 0..128 {
            x[k + i] = 1e-3;
            x[2 * k + i] = if i % 2 == 0 { (i as f32 / 2.0) + 0.5 } else { -((i as f32 / 2.0).floor() + 0.5) };
            x[3 * k + i] = if i % 3 == 0 { 127.0 } else { -127.0 };
            x[4 * k + i] = f32::from_bits(0x0000_0400 + i as u32); // subnormal inputs
            x[5 * k + i] = (i as f32 - 63.5) * 1.0e-30;
            x[6 * k + i] = 1.0e30 * ((i % 5) as f32 - 2.0);
        }
        x[k + 77] = 127.0;
        x[2 * k + 64] = 127.0; // amax exactly 127 -> d == 1
        for i in 128..512 {
            x[3 * k + i] = (i as f32) * 0.37 - 80.0;
            x[7 * k + i] = ((i * 31 % 255) as f32) - 127.0;
        }
        let xt = upload_f32(&mut gpu, &x);
        let nbytes = (k / 128) * n * BLK;
        let dev = gpu.alloc_tensor(&[nbytes], DType::Raw).expect("alloc q");
        let mut a = KernargBlob::new();
        a.push_ptr(xt.buf.as_ptr());
        a.push_ptr(dev.buf.as_ptr());
        a.push_i32(k as i32);
        a.push_i32(n as i32);
        a.pad_to(16);
        gpu.launch_kernel_blob(
            "quantize_int8_mmq_ds128",
            [(k.div_ceil(1024) * n) as u32, 1, 1],
            [256, 1, 1],
            0,
            a.as_mut_slice(),
        )
        .expect("launch standalone quantizer");
        gpu.hip.device_synchronize().expect("sync");
        let got = download_bytes(&gpu, &dev, nbytes);
        let want = quantize_cpu(&x, k, n);
        let diff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        js.line(&format!("{{\"kind\":\"standalone_quantizer_vs_cpu\",\"k\":{k},\"n\":{n},\"diff\":{diff},\"pass\":{}}}", diff == 0));
        if diff != 0 {
            st.fails.push(format!("standalone quantizer differs from CPU quantizer: {diff} bytes"));
            st.mismatches += diff as u64;
        }
    }

    // ---- C. GEMM cases ----
    // (label, wkind, m, k, n list)
    let m_list: &[(usize, usize)] = &[
        (130, 256),   // tail M (130 = 128 + 2), one group
        (48, 2560),   // tiny M
        (1000, 2560), // tail M for 32/64/128
        (2560, 6144), // out_proj shape
        (512, 2560),
        (1, 256),     // single row (clamped tail everywhere)
        (17, 512),
    ];
    let narrow_n = [9usize, 16, 24, 32, 48];
    let mut case_no = 0u64;
    for &(m, k) in m_list {
        for &n in &narrow_n {
            // one activation set per (k, n): real producer (random + zero token + outliers)
            let x = rand_activations(&mut rng, k, n);
            let (xq_ptr, _rot, xq_host) = produce_real(&mut gpu, &x, k, n);
            for (wi, wk) in [WKind::Random, WKind::EdgeScales, WKind::Exhaustive].into_iter().enumerate() {
                let w_host = gen_weights(m, k, wk, 0x5EED_0000 + (case_no << 8) + wi as u64);
                let w_dev = upload_bytes(&mut gpu, &w_host);
                for tile in tiles_for(n, true) {
                    for bf16 in [false, true] {
                        run_case(
                            &mut gpu, &mut st, &mut js, &format!("real-act/{:?}", wk), tile, bf16,
                            &w_host, &w_dev, xq_ptr, &xq_host, m, k, n, &mut rng, false, false,
                        );
                    }
                }
                case_no += 1;
            }
        }
    }

    // ---- D. crafted activation blocks: ties / +-127 / zero / extreme C / C-probe ----
    for (m, k, n) in [(130usize, 512usize, 9usize), (256, 2560, 32), (96, 256, 48)] {
        // crafted blocks: d = 1.0 everywhere (probe) with structured q
        let kb = k / 128;
        let mut xq = vec![0u8; kb * n * BLK];
        for b in 0..kb {
            for t in 0..n {
                let mut q = [0i8; 128];
                for i in 0..128 {
                    q[i] = match t % 6 {
                        0 => 127,
                        1 => -127,
                        2 => 0,
                        3 => ((i as i32 * 5 + b as i32) % 255 - 127) as i8,
                        4 => if i % 2 == 0 { 127 } else { -127 },
                        _ => (rng.next_u32() % 255) as i32 as i8 - 127,
                    };
                }
                let blk = make_block(if t % 6 == 2 { 0.0 } else { 1.0 }, &q);
                xq[(b * n + t) * BLK..(b * n + t + 1) * BLK].copy_from_slice(&blk);
            }
        }
        let xq_dev = upload_bytes(&mut gpu, &xq);
        for (wk, probe) in [
            (WKind::UnitScale, true),
            (WKind::AllMin, true),
            (WKind::AllMax, true),
            (WKind::Random, false),
            (WKind::EdgeScales, false),
        ] {
            let w_host = gen_weights(m, k, wk, 0xC0FFEE + case_no);
            case_no += 1;
            let w_dev = upload_bytes(&mut gpu, &w_host);
            for tile in tiles_for(n, true) {
                for bf16 in [false, true] {
                    run_case(
                        &mut gpu, &mut st, &mut js, &format!("crafted/{:?}", wk), tile, bf16,
                        &w_host, &w_dev, xq_dev.buf.as_ptr(), &xq, m, k, n, &mut rng, probe, false,
                    );
                }
            }
        }
    }

    // ---- E. prefill widths (p128 only): tails, 512/2048/4096 (16384 with T1_BIG) ----
    let mut pre_n = vec![129usize, 512, 2048, 4096];
    if big {
        pre_n.push(16384);
    }
    for &n in &pre_n {
        for &(m, k) in &[(1000usize, 2560usize), (2560, 6144)] {
            let x = rand_activations(&mut rng, k, n);
            let (xq_ptr, _rot, xq_host) = produce_real(&mut gpu, &x, k, n);
            let w_host = gen_weights(m, k, WKind::Random, 0xF00D + n as u64 + m as u64);
            let w_dev = upload_bytes(&mut gpu, &w_host);
            for bf16 in [false, true] {
                for tile in [Tile::p128(), Tile::p128fb()] {
                    run_case(
                        &mut gpu, &mut st, &mut js, "prefill/real-act/Random", tile, bf16,
                        &w_host, &w_dev, xq_ptr, &xq_host, m, k, n, &mut rng, false, false,
                    );
                }
            }
        }
    }

    // ---- F. negative controls: a deliberately corrupted operand must be DETECTED ----
    {
        let (m, k, n) = (130usize, 2560usize, 24usize);
        let x = rand_activations(&mut rng, k, n);
        let (xq_ptr, _rot, xq_host) = produce_real(&mut gpu, &x, k, n);
        let w_host = gen_weights(m, k, WKind::Random, 0xBAD_C0DE);
        let mut undetected = 0usize;
        for (what, tile) in [("code", Tile::narrow(64, 32)), ("scale", Tile::narrow(32, 32)), ("act", Tile::narrow(64, 48)), ("code", Tile::p128()), ("scale", Tile::p128()), ("act", Tile::p128())] {
            let mut w_bad = w_host.clone();
            let mut xq_bad = xq_host.clone();
            match what {
                // one 6-bit code of row 0, group 1 (block 2): flip a payload bit
                "code" => w_bad[GROUP + 4 + 50] ^= 0x04,
                // the s1 scale of row 0 group 0 (block 1)
                "scale" => w_bad[2] ^= 0x08,
                // one activation byte of (block 1, token 0)
                _ => xq_bad[(n) * BLK + 8 + 10] ^= 0x10,
            }
            let w_dev = upload_bytes(&mut gpu, &w_bad);
            let xq_dev = upload_bytes(&mut gpu, &xq_bad);
            let _ = xq_ptr;
            let pass = run_case(
                &mut gpu, &mut st, &mut js, &format!("negative/{what}"), tile, false,
                &w_host, &w_dev, xq_dev.buf.as_ptr(), &xq_host, m, k, n, &mut rng, false, true,
            );
            if pass {
                undetected += 1;
                st.fails.push(format!("negative control {what} on {} was NOT detected", tile.label()));
            }
        }
        js.line(&format!("{{\"kind\":\"negative_summary\",\"controls\":6,\"undetected\":{undetected}}}"));
    }

    let verdict = st.mismatches == 0 && st.fails.is_empty();
    js.line(&format!(
        "{{\"kind\":\"summary\",\"cases\":{},\"words_checked\":{},\"mismatches\":{},\"fails\":{},\"secs\":{:.1},\"pass\":{verdict}}}",
        st.cases, st.words, st.mismatches, st.fails.len(), t0.elapsed().as_secs_f64()
    ));
    for f in st.fails.iter().take(40) {
        eprintln!("FAIL {f}");
    }
    assert!(st.cases > 0, "vacuous run");
    assert!(verdict, "mq6sym wmma byte oracle: {} mismatching cases", st.fails.len());
}

// ---------------------------------------------------------------- timing

fn rand_bytes(rng: &mut Lcg, len: usize) -> Vec<u8> {
    let mut v = vec![0u8; len];
    for c in v.chunks_mut(8) {
        let r = rng.next_u64().to_le_bytes();
        let l = c.len();
        c.copy_from_slice(&r[..l]);
    }
    v
}

/// Random sym weights with valid normal f16 scales (timing content).
fn timing_sym_weights(rng: &mut Lcg, m: usize, k: usize) -> Vec<u8> {
    let mut w = rand_bytes(rng, m * (k / 256) * GROUP);
    for g in w.chunks_mut(GROUP) {
        for h in 0..2 {
            let e = 8 + (g[1 + 2 * h] % 10) as u16;
            let v = (e << 10) | (g[2 * h] as u16) | ((g[2 * h + 1] as u16 & 3) << 8);
            g[2 * h..2 * h + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    w
}

/// Random affine MQ6G256V2 weights (200-B groups: s0 z0 s1 z1 f16 + 192 B).
fn timing_affine_weights(rng: &mut Lcg, m: usize, k: usize) -> Vec<u8> {
    let mut w = rand_bytes(rng, m * (k / 256) * AFFINE_GROUP);
    for g in w.chunks_mut(AFFINE_GROUP) {
        for h in 0..2 {
            let e = 8 + (g[1 + 4 * h] % 10) as u16;
            let s = (e << 10) | (g[4 * h] as u16) | ((g[4 * h + 1] as u16 & 3) << 8);
            g[4 * h..4 * h + 2].copy_from_slice(&s.to_le_bytes());
            let z = ((e + 4) << 10) | (g[4 * h + 2] as u16) | ((g[4 * h + 3] as u16 & 3) << 8);
            g[4 * h + 2..4 * h + 4].copy_from_slice(&z.to_le_bytes());
        }
    }
    w
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Time `iters` launches of `f(set_index)` cycling over `sets` weight sets; event-bracketed,
/// returns microseconds per launch.
fn time_batch(gpu: &mut Gpu, iters: usize, sets: usize, f: &mut dyn FnMut(&Gpu, usize)) -> f64 {
    let s = gpu.active_stream.as_ref();
    let e0 = gpu.hip.event_create().expect("event");
    let e1 = gpu.hip.event_create().expect("event");
    gpu.hip.event_record(&e0, s).expect("record");
    for i in 0..iters {
        f(gpu, i % sets);
    }
    gpu.hip.event_record(&e1, gpu.active_stream.as_ref()).expect("record");
    gpu.hip.event_synchronize(&e1).expect("sync");
    let ms = gpu.hip.event_elapsed_ms(&e0, &e1).expect("elapsed");
    let _ = gpu.hip.event_destroy(e0);
    let _ = gpu.hip.event_destroy(e1);
    ms as f64 * 1000.0 / iters as f64
}

struct Arm<'a> {
    name: String,
    /// bytes of distinct weight data one launch reads
    wbytes: usize,
    run: Box<dyn FnMut(&Gpu, usize) + 'a>,
}

#[test]
#[ignore = "needs an exact gfx1201 GPU; timing, run on an idle card in a fabric timing job"]
fn mq6sym_i8_wmma_timing_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    if !gate(&gpu) {
        return;
    }
    load_entries(&mut gpu);
    gpu.ensure_kernel_public(AFFINE_MODULE, rdna_compute::kernels::GEMM_MQ6G256V2_MMQ_I8_GFX12_SRC, AFFINE_ENTRY)
        .expect("load affine i8 (hipcc source)");
    gpu.ensure_kernel_public(AFFINE_MODULE, rdna_compute::kernels::GEMM_MQ6G256V2_MMQ_I8_GFX12_SRC, AFFINE_BF16_ENTRY)
        .expect("load affine i8 bf16 (hipcc source)");
    let skinny_src = std::env::var("T1_SKINNY_SRC").ok().and_then(|p| std::fs::read_to_string(p).ok());
    // Same-format dot4 loop (T0 slice, rows r1..r8): the whole source behind the A8 prelude.
    let dot4_src: Option<String> = std::env::var("T1_DOT4_SRC").ok().and_then(|p| {
        let src = std::fs::read_to_string(p).ok()?;
        Some(format!("#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n{QUANT_SRC}\n{src}"))
    });
    gpu.ensure_kernel_public(STREAM_MODULE, STREAM_SRC, "t1_stream_read").expect("load stream roofline");
    let stream_out = gpu.alloc_tensor(&[16], DType::Raw).expect("alloc stream out");
    let stream_out_p = stream_out.buf.as_ptr();
    let mut js = Jsonl::new("timing");
    let filter: Option<Vec<String>> = std::env::var("T1_CELLS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let big = std::env::var("T1_BIG").is_ok();
    let mut rng = Lcg(0x71_0000_71E5);
    let rounds: usize = std::env::var("T1_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);

    // (name, M, K)
    let mut shapes: Vec<(&str, usize, usize)> = vec![
        ("qkv_10240x2560", 10240, 2560),
        ("z_6144x2560", 6144, 2560),
        ("out_2560x6144", 2560, 6144),
        ("qproj_12288x2560", 12288, 2560),
    ];
    if big {
        shapes.push(("lm_head_248320x2560", 248320, 2560));
    }
    let narrow_ns = [9usize, 16, 24, 32, 48];
    let dev_pre_n = [512usize, 2048, 4096];

    for &(name, m, k) in &shapes {
        if let Some(f) = &filter {
            if !f.iter().any(|x| name.contains(x.as_str())) {
                continue;
            }
        }
        let sym_bytes = m * (k / 256) * GROUP;
        let aff_bytes = m * (k / 256) * AFFINE_GROUP;
        // cold cyclic sets: >= 6 and >= ~384 MiB of weights (far beyond the 64 MiB MALL)
        let sets = (384usize << 20).div_ceil(sym_bytes).clamp(2, 16).max(if sym_bytes < (80 << 20) { 6 } else { 2 });
        let mut sym_w: Vec<GpuTensor> = Vec::new();
        let mut aff_w: Vec<GpuTensor> = Vec::new();
        let proto_sym = timing_sym_weights(&mut rng, m, k);
        let proto_aff = timing_affine_weights(&mut rng, m, k);
        for s in 0..sets {
            let mut w = proto_sym.clone();
            // distinct contents per set (first bytes) without regenerating GBs
            let w0 = w.len().min(64);
            for (i, b) in w[..w0].iter_mut().enumerate() {
                *b ^= (s as u8).wrapping_mul(37).wrapping_add(i as u8);
            }
            sym_w.push(upload_bytes(&mut gpu, &w));
            let mut a = proto_aff.clone();
            let a0 = a.len().min(64);
            for (i, b) in a[..a0].iter_mut().enumerate() {
                *b ^= (s as u8).wrapping_mul(53).wrapping_add(i as u8);
            }
            aff_w.push(upload_bytes(&mut gpu, &a));
        }
        drop(proto_sym);
        drop(proto_aff);

        // ------- skinny verify cells -------
        for &n in &narrow_ns {
            let cell = format!("{name}/N{n}");
            let mut keep_alive: Vec<GpuTensor> = Vec::new();
            let x = rand_activations(&mut rng, k, n);
            let xt = upload_f32(&mut gpu, &x);
            // 48 token rows even for N < 48: the parked skinny kernel writes whole 16/32-token tiles
            let ybytes = n.max(48) * m * 4;
            let y = gpu.alloc_tensor(&[ybytes], DType::Raw).expect("alloc y");
            let ybf = gpu.alloc_tensor(&[ybytes / 2], DType::Raw).expect("alloc ybf");
            // producer time (alone) with fresh reservation each batch (steady state)
            let prod_us = {
                let mut v = Vec::new();
                for _ in 0..3 {
                    let res0 = gpu.reserve_int8_mmq(k, n).expect("reserve");
                    drop(res0);
                    let t = {
                        let s = gpu.active_stream.as_ref();
                        let e0 = gpu.hip.event_create().unwrap();
                        let e1 = gpu.hip.event_create().unwrap();
                        gpu.hip.event_record(&e0, s).unwrap();
                        for _ in 0..50 {
                            let res = gpu.reserve_int8_mmq(k, n).expect("reserve");
                            let _ = gpu.rotate_x_mq_i8_batched(&xt, None, res, k, n).expect("producer");
                        }
                        gpu.hip.event_record(&e1, gpu.active_stream.as_ref()).unwrap();
                        gpu.hip.event_synchronize(&e1).unwrap();
                        let ms = gpu.hip.event_elapsed_ms(&e0, &e1).unwrap();
                        let _ = gpu.hip.event_destroy(e0);
                        let _ = gpu.hip.event_destroy(e1);
                        ms as f64 * 1000.0 / 50.0
                    };
                    v.push(t);
                }
                median(&mut v)
            };
            // activation for the GEMM arms: the real producer's scratch
            let res = gpu.reserve_int8_mmq(k, n).expect("reserve");
            let prep = gpu.rotate_x_mq_i8_batched(&xt, None, res, k, n).expect("producer");
            let xq = gpu.int8_mmq_prepared_ptr(&prep, k, n).expect("xq ptr");
            gpu.hip.device_synchronize().unwrap();
            let yp = y.buf.as_ptr();
            let ybp = ybf.buf.as_ptr();
            let wp: Vec<*mut c_void> = sym_w.iter().map(|t| t.buf.as_ptr()).collect();
            let ap: Vec<*mut c_void> = aff_w.iter().map(|t| t.buf.as_ptr()).collect();

            let mut arms: Vec<Arm> = Vec::new();
            for tile in tiles_for(n, false) {
                if tile.ys && n <= tile.ntok {
                    continue; // identical to the plain tile
                }
                let outs: Vec<(bool, *mut c_void)> = if std::env::var("T1_BF16").is_ok() {
                    vec![(false, yp), (true, ybp)]
                } else {
                    vec![(false, yp)]
                };
                for (bf16, yb) in outs {
                    let wp = wp.clone();
                    arms.push(Arm {
                        name: format!("sym_{}_{}", tile.label(), if bf16 { "bf16" } else { "f32" }),
                        wbytes: sym_bytes,
                        run: Box::new(move |g, i| launch_sym(g, tile, bf16, wp[i], xq, yb, m, k, n)),
                    });
                }
            }
            {
                let wp = wp.clone();
                arms.push(Arm {
                    name: "sym_p128_f32".into(),
                    wbytes: sym_bytes,
                    run: Box::new(move |g, i| launch_sym(g, Tile::p128(), false, wp[i], xq, yp, m, k, n)),
                });
            }
            {
                let ap = ap.clone();
                arms.push(Arm {
                    name: "affine_i8_hipcc_f32".into(),
                    wbytes: aff_bytes,
                    run: Box::new(move |g, i| launch_affine_src(g, false, ap[i], xq, yp, m, k, n)),
                });
            }
            {
                // roofline arm: read exactly the symmetric weight bytes of the set, 16 B per lane
                let wp = wp.clone();
                let n16 = (sym_bytes / 16) as u64;
                arms.push(Arm {
                    name: "stream_read_roofline".into(),
                    wbytes: sym_bytes,
                    run: Box::new(move |g, i| {
                        let mut a = KernargBlob::new();
                        a.push_ptr(wp[i]);
                        a.push_u64(n16);
                        a.push_ptr(stream_out_p);
                        a.pad_to(16);
                        g.launch_kernel_blob("t1_stream_read", [8192, 1, 1], [256, 1, 1], 0, a.as_mut_slice())
                            .expect("stream launch");
                    }),
                });
            }
            if let Some(src) = &dot4_src {
                // Eight-token activation pass(es): the dot4 kernel takes N <= 8 tokens per
                // launch, so N = 9..48 is ceil(N/8) back-to-back launches that each re-stream the
                // weights (the honest same-format dot4 alternative). Own persistent 8-token
                // sidecar (same producer), then the N-token scratch is rebuilt.
                let mut ok = true;
                for r in 1..=8 {
                    ok &= gpu
                        .ensure_kernel_public("t1_dot4_competitor", src, &format!("gemv_mq6sym_i8_rows_r{r}_gfx1201"))
                        .is_ok();
                }
                if ok {
                    let x8 = rand_activations(&mut rng, k, 8);
                    let x8t = upload_f32(&mut gpu, &x8);
                    let res8 = gpu.reserve_int8_mmq(k, 8).expect("reserve8");
                    let prep8 = gpu.rotate_x_mq_i8_batched(&x8t, None, res8, k, 8).expect("producer8");
                    let p8 = gpu.int8_mmq_prepared_ptr(&prep8, k, 8).expect("ptr8");
                    let nb8 = (k / 128) * 8 * BLK;
                    let xq8 = gpu.alloc_tensor(&[nb8], DType::Raw).expect("alloc xq8");
                    unsafe {
                        gpu.hip.memcpy_dtod_raw(xq8.buf.as_ptr(), p8 as *const c_void, nb8).expect("dtod xq8");
                    }
                    let res = gpu.reserve_int8_mmq(k, n).expect("reserve");
                    let prep = gpu.rotate_x_mq_i8_batched(&xt, None, res, k, n).expect("producer");
                    let again = gpu.int8_mmq_prepared_ptr(&prep, k, n).expect("xq ptr");
                    assert_eq!(again, xq, "scratch pointer moved");
                    gpu.hip.device_synchronize().unwrap();
                    let wp = wp.clone();
                    let xq8p = xq8.buf.as_ptr();
                    keep_alive.push(xq8);
                    let passes = n.div_ceil(8);
                    arms.push(Arm {
                        name: format!("dot4_same_format_x{passes}pass_f32"),
                        wbytes: sym_bytes,
                        run: Box::new(move |g, i| {
                            for p in 0..passes {
                                let r = if (p + 1) * 8 <= n { 8 } else { n - p * 8 };
                                let mut a = KernargBlob::new();
                                a.push_ptr(wp[i]);
                                a.push_ptr(xq8p);
                                a.push_ptr(yp);
                                a.push_i32(m as i32);
                                a.push_i32(k as i32);
                                a.push_i32(r as i32);
                                a.pad_to(16);
                                g.launch_kernel_blob(
                                    &format!("gemv_mq6sym_i8_rows_r{r}_gfx1201"),
                                    [m.div_ceil(32) as u32, 1, 1],
                                    [128, 1, 1],
                                    0,
                                    a.as_mut_slice(),
                                )
                                .expect("dot4 launch");
                            }
                        }),
                    });
                }
            }
            // Parked DenseR32 skinny MQ6 F16 WMMA kernels (commit cce88f1245, affine 200-B groups,
            // F16 activations, different arithmetic): speed reference for N <= 32 only.
            if let Some(src) = &skinny_src {
                let (nt, cfgs): (usize, &[(usize, usize)]) = if n <= 16 {
                    (1, &[(1, 4), (2, 4), (2, 8)])
                } else if n <= 32 {
                    (2, &[(1, 4), (2, 4), (2, 8), (1, 8)])
                } else {
                    (0, &[])
                };
                if nt > 0 {
                    let r = 16 * nt;
                    let x16 = {
                        let v: Vec<u8> = (0..r * k * 2).map(|i| if i % 2 == 0 { 0x00 } else { 0x3C }).collect();
                        upload_bytes(&mut gpu, &v)
                    };
                    let x16p = x16.buf.as_ptr();
                    keep_alive.push(x16);
                    for &(mt, sw) in cfgs {
                        let entry = format!("gemm_mq6g256v2_xf16_skinny_x4_n{nt}_m{mt}_s{sw}");
                        if gpu.ensure_kernel_public("t1_skinny_competitor", src, &entry).is_err() {
                            continue;
                        }
                        let ap = ap.clone();
                        let en = entry.clone();
                        arms.push(Arm {
                            name: format!("parked_{}_f16", entry.trim_start_matches("gemm_mq6g256v2_xf16_skinny_x4_")),
                            wbytes: aff_bytes,
                            run: Box::new(move |g, i| {
                                let mut a = KernargBlob::new();
                                for _ in 0..4 {
                                    a.push_ptr(ap[i]);
                                }
                                a.push_ptr(x16p);
                                for _ in 0..4 {
                                    a.push_ptr(yp);
                                }
                                a.push_i32(m as i32);
                                a.push_i32(0);
                                a.push_i32(0);
                                a.push_i32(0);
                                a.push_i32(k as i32);
                                a.pad_to(16);
                                g.launch_kernel_blob(
                                    &en,
                                    [m.div_ceil(16 * mt) as u32, 1, 1],
                                    [(32 * sw) as u32, 1, 1],
                                    0,
                                    a.as_mut_slice(),
                                )
                                .expect("skinny launch");
                            }),
                        });
                    }
                }
            }

            // Alternate arms inside each round; per arm collect per-round us, report median/min.
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
            let iters = if sym_bytes > (100 << 20) { 12 } else { 60 };
            for r in 0..rounds {
                let order: Vec<usize> = if r % 2 == 0 {
                    (0..arms.len()).collect()
                } else {
                    (0..arms.len()).rev().collect()
                };
                for &ai in &order {
                    // warm
                    {
                        let arm = &mut arms[ai];
                        (arm.run)(&gpu, 0);
                    }
                    gpu.hip
                        .device_synchronize()
                        .unwrap_or_else(|e| panic!("arm {} ({cell}): {e}", arms[ai].name));
                    let us = {
                        let arm = &mut arms[ai];
                        time_batch(&mut gpu, iters, sets, &mut |g, i| (arm.run)(g, i))
                    };
                    samples[ai].push(us);
                }
            }
            for (ai, arm) in arms.iter().enumerate() {
                let mut v = samples[ai].clone();
                let med = median(&mut v);
                let mn = samples[ai].iter().cloned().fold(f64::MAX, f64::min);
                let wgbs = arm.wbytes as f64 / med / 1e3;
                let traffic = (arm.wbytes + (k / 128) * n * BLK + n * m * if arm.name.contains("bf16") { 2 } else { 4 }) as f64;
                js.line(&format!(
                    "{{\"kind\":\"skinny\",\"cell\":\"{cell}\",\"arm\":\"{}\",\"m\":{m},\"k\":{k},\"n\":{n},\"sets\":{sets},\"us_median\":{med:.2},\"us_min\":{mn:.2},\"weights_GBps\":{wgbs:.1},\"total_traffic_GBps\":{:.1},\"weight_bytes\":{},\"producer_us\":{prod_us:.2}}}",
                    arm.name, traffic / med / 1e3, arm.wbytes
                ));
            }
        }

        // ------- prefill cells (compute-bound; two weight sets) -------
        // the lm_head is only ever evaluated for a handful of sampled rows: no prefill cells for it
        let mut pre: Vec<usize> = if m > 100_000 { Vec::new() } else { dev_pre_n.to_vec() };
        if big && m <= 100_000 {
            pre.push(16384);
        }
        for &n in &pre {
            let cell = format!("{name}/N{n}");
            if k == 6144 && n > 4096 {
                continue;
            }
            let x = rand_activations(&mut rng, k, n.min(4096));
            let nn = n.min(4096);
            let _ = nn;
            let xt = upload_f32(&mut gpu, &if n > 4096 { rand_activations(&mut rng, k, n) } else { x });
            let res = gpu.reserve_int8_mmq(k, n).expect("reserve");
            let prep = gpu.rotate_x_mq_i8_batched(&xt, None, res, k, n).expect("producer");
            let xq = gpu.int8_mmq_prepared_ptr(&prep, k, n).expect("xq ptr");
            let y = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y");
            let yp = y.buf.as_ptr();
            let wp: Vec<*mut c_void> = sym_w.iter().map(|t| t.buf.as_ptr()).collect();
            let ap: Vec<*mut c_void> = aff_w.iter().map(|t| t.buf.as_ptr()).collect();
            let mut arms: Vec<Arm> = Vec::new();
            {
                let wp = wp.clone();
                arms.push(Arm {
                    name: "sym_p128_f32".into(),
                    wbytes: sym_bytes,
                    run: Box::new(move |g, i| launch_sym(g, Tile::p128(), false, wp[i], xq, yp, m, k, n)),
                });
            }
            {
                let wp = wp.clone();
                arms.push(Arm {
                    name: "sym_p128fb_f32".into(),
                    wbytes: sym_bytes,
                    run: Box::new(move |g, i| launch_sym(g, Tile::p128fb(), false, wp[i], xq, yp, m, k, n)),
                });
            }
            {
                let ap = ap.clone();
                arms.push(Arm {
                    name: "affine_i8_hipcc_f32".into(),
                    wbytes: aff_bytes,
                    run: Box::new(move |g, i| launch_affine_src(g, false, ap[i], xq, yp, m, k, n)),
                });
            }
            // the production route (PeaceMaker twin where admitted) through the public launcher
            let aff_t: Vec<&GpuTensor> = aff_w.iter().collect();
            let prod_name = gpu
                .gemm_mq6g256v2_i8_kernel_name(m, k, n, &y)
                .unwrap_or("none");
            let ytensor = &y;
            let gpu_ptr: *mut Gpu = &mut gpu;
            let _ = gpu_ptr;
            let prod_entry = prod_name.to_string();
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len() + 1];
            let iters = if n >= 4096 { 6 } else { 12 };
            for r in 0..rounds {
                let order: Vec<usize> = if r % 2 == 0 { vec![0, 1, 2, 3] } else { vec![3, 2, 1, 0] };
                for &ai in &order {
                    let us = if ai < 3 {
                        let arm = &mut arms[ai];
                        (arm.run)(&gpu, 0);
                        gpu.hip.device_synchronize().unwrap();
                        time_batch(&mut gpu, iters, 2, &mut |g, i| (arm.run)(g, i))
                    } else {
                        // public launcher needs &mut Gpu: bracket manually
                        let e0 = gpu.hip.event_create().unwrap();
                        let e1 = gpu.hip.event_create().unwrap();
                        gpu.gemm_mq6g256v2_i8_prequant(aff_t[0], xq, ytensor, m, k, n).expect("prod warm");
                        gpu.hip.device_synchronize().unwrap();
                        gpu.hip.event_record(&e0, gpu.active_stream.as_ref()).unwrap();
                        for i in 0..iters {
                            gpu.gemm_mq6g256v2_i8_prequant(aff_t[i % 2], xq, ytensor, m, k, n).expect("prod");
                        }
                        gpu.hip.event_record(&e1, gpu.active_stream.as_ref()).unwrap();
                        gpu.hip.event_synchronize(&e1).unwrap();
                        let ms = gpu.hip.event_elapsed_ms(&e0, &e1).unwrap();
                        let _ = gpu.hip.event_destroy(e0);
                        let _ = gpu.hip.event_destroy(e1);
                        ms as f64 * 1000.0 / iters as f64
                    };
                    samples[ai].push(us);
                }
            }
            let names = ["sym_p128_f32".to_string(), "sym_p128fb_f32".to_string(), "affine_i8_hipcc_f32".to_string(), format!("affine_i8_production({prod_entry})")];
            for (ai, nm) in names.iter().enumerate() {
                let mut v = samples[ai].clone();
                let med = median(&mut v);
                let mn = samples[ai].iter().cloned().fold(f64::MAX, f64::min);
                let tops = 2.0 * m as f64 * k as f64 * n as f64 / med / 1e6;
                js.line(&format!(
                    "{{\"kind\":\"prefill\",\"cell\":\"{cell}\",\"arm\":\"{nm}\",\"m\":{m},\"k\":{k},\"n\":{n},\"us_median\":{med:.2},\"us_min\":{mn:.2},\"TFLOP_equiv\":{tops:.2}}}"
                ));
            }
        }
    }
}
