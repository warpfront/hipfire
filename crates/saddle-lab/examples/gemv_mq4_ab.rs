// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `gemv_mq4_ab`: A/B harness for the hot decode GEMV (`gemv_mq4g256v2`).
//!
//! Ports nothing; it measures. One arm is always hipcc (compiles
//! `kernels/src/gemv_mq4g256v2.hip` with the production flags), plus one arm
//! per `--elf=PATH` (e.g. a rustc `amdgcn-amd-amdhsa` ELF) loaded through the
//! same HIP dispatcher. Every arm runs the same seeded inputs, y is
//! bit-compared against the hipcc arm, and the hipcc arm itself is sanity
//! checked against a CPU reference using the same 4-accumulator order.
//!
//! Lab-only GLOBAL SoA layout experiment (additive flags; defaults unchanged):
//! ```sh
//! # AoS hipcc vs SoA hipcc, cache-honest rotating working set
//! ./target/release/examples/gemv_mq4_ab --soa --working-set-mib=256 --iters=100
//! # Counterfactual read-only probe (AoS RO vs SoA RO from soa_test.hip)
//! ./target/release/examples/gemv_mq4_ab --soa --read-only --working-set-mib=256
//! # Reverse timed arm order (parity baseline still first arm)
//! ./target/release/examples/gemv_mq4_ab --soa --reverse
//! ```
//!
//! No feature gate: `hip-bridge` is an unconditional `saddle-lab` dependency.

use hip_bridge::{DeviceBuffer, Function, HipRuntime, Module, Stream};
use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

/// Default shapes (Qwen3.8-27B: hidden 5120, intermediate 17408, vocab 248320)
/// plus the two tail-path shapes (5 groups: quads=1 tail=1; 7 groups: tail=3).
const DEFAULT_SHAPES: [(usize, usize); 6] = [
    (5120, 5120),
    (17408, 5120),
    (5120, 17408),
    (248320, 5120),
    (4096, 1280),
    (4096, 1792),
];

/// Deterministic xorshift64* RNG; the seed comes from `--seed`.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_byte(&mut self) -> u8 {
        (self.next_u64() >> 32) as u8
    }
    /// Uniform f32 in [0, 1).
    fn next_unit(&mut self) -> f32 {
        const INV: f32 = 1.0 / 9_007_199_254_740_992.0; // 2^-53
        ((self.next_u64() >> 11) as f64 * INV as f64) as f32
    }
}

/// f32 → f16 bits, round-to-nearest-even (no `half` dep so this example needs
/// no manifest change; the generated scale/zp values are all normal-range).
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 255 {
        // Inf/NaN.
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00; // Overflow → Inf.
    }
    if e <= 0 {
        // Subnormal or underflow to zero.
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let low = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let mut q = (m >> shift) as u16;
        if low > half || (low == half && (q & 1) == 1) {
            q += 1; // May become 0x0400 = smallest normal; encoding is exact.
        }
        return sign | q;
    }
    let low = mant & 0x1FFF;
    let mut q = (mant >> 13) as u16;
    if low > 0x1000 || (low == 0x1000 && (q & 1) == 1) {
        q += 1;
        if q == 0x0400 {
            let e2 = e + 1;
            if e2 >= 31 {
                return sign | 0x7C00;
            }
            return sign | ((e2 as u16) << 10);
        }
    }
    sign | ((e as u16) << 10) | (q & 0x03FF)
}

/// f16 bits → f32, exact (mirrors `__half2float` for the CPU reference).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;
    let b = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e: i32 = 127 - 14;
            let mut m = mant;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x03FF) << 13)
        }
    } else if exp == 31 {
        sign | (0xFF << 23) | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(b)
}

// ——— tiny MD5 + hex (same no-new-deps pattern as vl_vcn_parity) ———
struct Md5 {
    state: [u32; 4],
    count: u64,
    buf: [u8; 64],
    buf_len: usize,
}

impl Md5 {
    fn digest(data: &[u8]) -> [u8; 16] {
        let mut m = Md5 {
            state: [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476],
            count: 0,
            buf: [0; 64],
            buf_len: 0,
        };
        m.update(data);
        m.finish()
    }
}

#[allow(clippy::many_single_char_names)]
impl Md5 {
    fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let b = self.buf;
                Self::block(&mut self.state, &b);
                self.count += 64;
                self.buf_len = 0;
            }
        }
    }
    fn finish(mut self) -> [u8; 16] {
        let bit_len = (self.count + self.buf_len as u64) * 8;
        self.update(&[0x80]);
        while self.buf_len != 56 {
            self.update(&[0x00]);
        }
        self.update(&bit_len.to_le_bytes());
        let mut out = [0u8; 16];
        for (i, s) in self.state.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&s.to_le_bytes());
        }
        out
    }
    fn block(s: &mut [u32; 4], b: &[u8; 64]) {
        const S: [u32; 64] = [
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20,
            5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        ];
        const K: [u32; 64] = [
            0xd76a_a478,
            0xe8c7_b756,
            0x2420_70db,
            0xc1bd_ceee,
            0xf57c_0faf,
            0x4787_c62a,
            0xa830_4613,
            0xfd46_9501,
            0x6980_98d8,
            0x8b44_f7af,
            0xffff_5bb1,
            0x895c_d7be,
            0x6b90_1122,
            0xfd98_7193,
            0xa679_438e,
            0x49b4_0821,
            0xf61e_2562,
            0xc040_b340,
            0x265e_5a51,
            0xe9b6_c7aa,
            0xd62f_105d,
            0x0244_1453,
            0xd8a1_e681,
            0xe7d3_fbc8,
            0x21e1_cde6,
            0xc337_07d6,
            0xf4d5_0d87,
            0x455a_14ed,
            0xa9e3_e905,
            0xfcef_a3f8,
            0x676f_02d9,
            0x8d2a_4c8a,
            0xfffa_3942,
            0x8771_f681,
            0x6d9d_6122,
            0xfde5_380c,
            0xa4be_ea44,
            0x4bde_cfa9,
            0xf6bb_4b60,
            0xbebf_bc70,
            0x289b_7ec6,
            0xeaa1_27fa,
            0xd4ef_3085,
            0x0488_1d05,
            0xd9d4_d039,
            0xe6db_99e5,
            0x1fa2_7cf8,
            0xc4ac_5665,
            0xf429_2244,
            0x432a_ff97,
            0xab94_23a7,
            0xfc93_a039,
            0x655b_59c3,
            0x8f0c_cc92,
            0xffef_f47d,
            0x8584_5dd1,
            0x6fa8_7e4f,
            0xfe2c_e6e0,
            0xa301_4314,
            0x4e08_11a1,
            0xf753_7e82,
            0xbd3a_f235,
            0x2ad7_d2bb,
            0xeb86_d391,
        ];
        let mut m = [0u32; 16];
        for (i, v) in m.iter_mut().enumerate() {
            *v = u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        }
        let (mut a, mut b_, mut c, mut d) = (s[0], s[1], s[2], s[3]);
        for i in 0..64 {
            let (f, g) = match i {
                0..16 => ((b_ & c) | (!b_ & d), i),
                16..32 => ((d & b_) | (!d & c), (5 * i + 1) % 16),
                32..48 => (b_ ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b_ | !d), (7 * i) % 16),
            };
            let tmp = d;
            d = c;
            c = b_;
            b_ = b_.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(K[i])
                    .wrapping_add(m[g])
                    .rotate_left(S[i]),
            );
            a = tmp;
        }
        s[0] = s[0].wrapping_add(a);
        s[1] = s[1].wrapping_add(b_);
        s[2] = s[2].wrapping_add(c);
        s[3] = s[3].wrapping_add(d);
    }
}

fn md5hex(data: &[u8]) -> String {
    Md5::digest(data)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// Explicit args of `gemv_mq4g256v2`, in kernel signature order
/// (A@0, x@8, y@16, M@24, K@28). Passed as a HIP pointer vec.
struct GemvArgs {
    a: u64,
    x: u64,
    y: u64,
    m: i32,
    k: i32,
}

impl GemvArgs {
    fn ptrs(&mut self) -> Vec<*mut c_void> {
        vec![
            (&mut self.a as *mut u64).cast(),
            (&mut self.x as *mut u64).cast(),
            (&mut self.y as *mut u64).cast(),
            (&mut self.m as *mut i32).cast(),
            (&mut self.k as *mut i32).cast(),
        ]
    }
}

/// One benchmark arm: the loaded bytes plus its resolved entry point.
/// `module` is never read after load — it is kept alive because unloading
/// would invalidate `func`.
struct Arm {
    name: String,
    md5: String,
    #[allow(dead_code)]
    module: Module,
    func: Function,
    /// true → weight buffer is GLOBAL SoA; false → AoS (production layout).
    layout_soa: bool,
    /// true → y holds u32 checksum bits (RO probe); false → f32 GEMV.
    read_only: bool,
}

/// Build seeded inputs: weights (`M * (K/256) * 136` bytes) and `x` (`K`
/// f32 in [-1, 1]). Each weight group is an 8 B header (two u32, each packing
/// scale f16 in [0.001, 0.05] / zp f16 in [-0.4, 0.0]) plus 128 random bytes.
fn build_inputs(m: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    assert!(k % 256 == 0, "K={k} must be a multiple of 256");
    let groups = k / 256;
    let mut rng = Rng::new(seed ^ ((m as u64) << 32) ^ (k as u64));
    let mut w = vec![0u8; m * groups * 136];
    for row in 0..m {
        for g in 0..groups {
            let gp = (row * groups + g) * 136;
            for h in 0..2 {
                let sc = 0.001 + rng.next_unit() * 0.049;
                let zp = -0.4 + rng.next_unit() * 0.4;
                let packed = f32_to_f16_bits(sc) as u32 | ((f32_to_f16_bits(zp) as u32) << 16);
                w[gp + 4 * h..gp + 4 * h + 4].copy_from_slice(&packed.to_le_bytes());
            }
            for i in 0..128 {
                w[gp + 8 + i] = rng.next_byte();
            }
        }
    }
    let mut x = vec![0f32; k];
    for v in x.iter_mut() {
        *v = rng.next_unit() * 2.0 - 1.0;
    }
    (w, x)
}

/// CPU reference for one row, replicating the kernel's exact accumulation
/// order: per-lane 4-accumulator interleave (`acc[g % 4]`), per-group DOG
/// terms summed left-to-right, per-lane combine `(acc0+acc1)+(acc2+acc3)`,
/// then the `__shfl_down` tree (offsets 16, 8, 4, 2, 1).
fn cpu_row(w: &[u8], x: &[f32], row: usize, k: usize) -> f32 {
    let groups = k / 256;
    let quads = groups >> 2;
    let tail = groups & 3;
    let row_base = row * groups * 136;
    let mut acc = [[0f32; 4]; 32];
    // One group's 8-nibble dot for lane `tid`, folded exactly like the DOG
    // macro: acc_lane = acc_lane + ((((((t0+t1)+t2)+t3)+t4)+t5)+t6)+t7.
    let dot_group = |gp: usize, g: usize, slot: usize, acc: &mut [[f32; 4]; 32]| {
        let ha = u32::from_le_bytes([w[gp], w[gp + 1], w[gp + 2], w[gp + 3]]);
        let hb = u32::from_le_bytes([w[gp + 4], w[gp + 5], w[gp + 6], w[gp + 7]]);
        for (tid, lane) in acc.iter_mut().enumerate() {
            let hs = if tid < 16 { ha } else { hb };
            let sc = f16_to_f32((hs & 0xFFFF) as u16);
            let zp = f16_to_f32((hs >> 16) as u16);
            let pk = u32::from_le_bytes([
                w[gp + 8 + tid * 4],
                w[gp + 8 + tid * 4 + 1],
                w[gp + 8 + tid * 4 + 2],
                w[gp + 8 + tid * 4 + 3],
            ]);
            let base = g * 256 + tid * 8;
            let mut sum = (sc * ((pk & 0xF) as f32) + zp) * x[base];
            for i in 1..8 {
                let term = (sc * (((pk >> (4 * i)) & 0xF) as f32) + zp) * x[base + i];
                sum += term;
            }
            lane[slot] += sum;
        }
    };
    for q in 0..quads {
        let g = q << 2;
        for s in 0..4 {
            dot_group(row_base + (g + s) * 136, g + s, s, &mut acc);
        }
    }
    for t in 0..tail {
        let g = (quads << 2) + t;
        dot_group(row_base + g * 136, g, t, &mut acc);
    }
    let mut lane = [0f32; 32];
    for (tid, lane_acc) in acc.iter().enumerate() {
        lane[tid] = (lane_acc[0] + lane_acc[1]) + (lane_acc[2] + lane_acc[3]);
    }
    for offset in [16, 8, 4, 2, 1] {
        let prev = lane;
        for tid in 0..32 {
            lane[tid] = if tid + offset < 32 {
                prev[tid] + prev[tid + offset]
            } else {
                prev[tid]
            };
        }
    }
    lane[0]
}

/// GLOBAL SoA: payload plane first (`N*128`), then header plane (`N*8`).
/// `n = row*gpr + g`; payload at `n*128`; header at `N*128 + n*8`.
fn repack_aos_to_soa(aos: &[u8], m: usize, k: usize) -> Vec<u8> {
    let gpr = k / 256;
    let n_total = m * gpr;
    assert_eq!(aos.len(), n_total * 136);
    let mut soa = vec![0u8; n_total * 136];
    for n in 0..n_total {
        let src = n * 136;
        let pdst = n * 128;
        let hdst = n_total * 128 + n * 8;
        soa[pdst..pdst + 128].copy_from_slice(&aos[src + 8..src + 136]);
        soa[hdst..hdst + 8].copy_from_slice(&aos[src..src + 8]);
    }
    soa
}

/// Inverse of [`repack_aos_to_soa`]: GLOBAL SoA → AoS group-major.
fn repack_soa_to_aos(soa: &[u8], m: usize, k: usize) -> Vec<u8> {
    let gpr = k / 256;
    let n_total = m * gpr;
    assert_eq!(soa.len(), n_total * 136);
    let mut aos = vec![0u8; n_total * 136];
    for n in 0..n_total {
        let psrc = n * 128;
        let hsrc = n_total * 128 + n * 8;
        let dst = n * 136;
        aos[dst..dst + 8].copy_from_slice(&soa[hsrc..hsrc + 8]);
        aos[dst + 8..dst + 136].copy_from_slice(&soa[psrc..psrc + 128]);
    }
    aos
}

fn rotl32(x: u32, n: u32) -> u32 {
    x.rotate_left(n)
}

/// CPU RO checksum oracle. Mirrors `gemv_mq4g256v2_ro` / `_soa_ro`:
/// per-lane wrapping pk sum over groups, XOR-shuffle reduce 16..1, then
/// lane0 adds `sum_g (hA + rotl32(hB, 16))`.
fn cpu_row_ro(w: &[u8], row: usize, k: usize, soa: bool) -> u32 {
    let gpr = k / 256;
    let n_plane = w.len() / 136;
    assert_eq!(w.len(), n_plane * 136);
    assert_eq!(n_plane % gpr, 0, "weight bytes must be M*gpr*136");
    let mut lane_acc = [0u32; 32];
    let mut header_acc = 0u32;
    let row_base_n = row * gpr;
    for g in 0..gpr {
        let n = row_base_n + g;
        let (ha, hb, pks) = if soa {
            let hbase = n_plane * 128 + n * 8;
            let pbase = n * 128;
            let ha = u32::from_le_bytes([w[hbase], w[hbase + 1], w[hbase + 2], w[hbase + 3]]);
            let hb = u32::from_le_bytes([w[hbase + 4], w[hbase + 5], w[hbase + 6], w[hbase + 7]]);
            let mut pks = [0u32; 32];
            for tid in 0..32 {
                let o = pbase + tid * 4;
                pks[tid] = u32::from_le_bytes([w[o], w[o + 1], w[o + 2], w[o + 3]]);
            }
            (ha, hb, pks)
        } else {
            let gp = n * 136;
            let ha = u32::from_le_bytes([w[gp], w[gp + 1], w[gp + 2], w[gp + 3]]);
            let hb = u32::from_le_bytes([w[gp + 4], w[gp + 5], w[gp + 6], w[gp + 7]]);
            let mut pks = [0u32; 32];
            for tid in 0..32 {
                let o = gp + 8 + tid * 4;
                pks[tid] = u32::from_le_bytes([w[o], w[o + 1], w[o + 2], w[o + 3]]);
            }
            (ha, hb, pks)
        };
        header_acc = header_acc.wrapping_add(ha.wrapping_add(rotl32(hb, 16)));
        for tid in 0..32 {
            lane_acc[tid] = lane_acc[tid].wrapping_add(pks[tid]);
        }
    }
    // XOR-shuffle reduce (butterfly) offsets 16,8,4,2,1 — matches __shfl_xor.
    for offset in [16usize, 8, 4, 2, 1] {
        let prev = lane_acc;
        for tid in 0..32 {
            lane_acc[tid] = prev[tid] ^ prev[tid ^ offset];
        }
    }
    lane_acc[0].wrapping_add(header_acc)
}

struct Cli {
    elfs: Vec<String>,
    hipcc_extra: Vec<String>,
    iters: usize,
    seed: u64,
    shapes: Vec<(usize, usize)>,
    soa: bool,
    read_only: bool,
    working_set_mib: u64,
    reverse: bool,
}

fn parse_args() -> Cli {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `--elf=PATH` is repeatable.
    let elfs: Vec<String> = args
        .iter()
        .filter_map(|a| a.strip_prefix("--elf="))
        .map(|s| s.to_string())
        .collect();
    let hipcc_extra: Vec<String> = args
        .iter()
        .filter_map(|a| a.strip_prefix("--hipcc-flags="))
        .flat_map(|s| s.split_whitespace().map(|f| f.to_string()))
        .collect();
    let iters: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("--iters=").and_then(|v| v.parse().ok()))
        .unwrap_or(100);
    assert!(iters >= 1, "--iters must be >= 1");
    let seed: u64 = args
        .iter()
        .find_map(|a| a.strip_prefix("--seed=").and_then(|v| v.parse().ok()))
        .unwrap_or(1);
    let shapes: Vec<(usize, usize)> = match args.iter().find_map(|a| a.strip_prefix("--shapes=")) {
        Some(s) => s
            .split(',')
            .map(|pair| {
                let (ms, ks) = pair
                    .split_once('x')
                    .unwrap_or_else(|| panic!("--shapes entry {pair:?} must look like MxK"));
                let m: usize = ms.parse().unwrap_or_else(|_| panic!("bad M in {pair:?}"));
                let k: usize = ks.parse().unwrap_or_else(|_| panic!("bad K in {pair:?}"));
                assert!(k % 256 == 0, "K={k} must be a multiple of 256");
                (m, k)
            })
            .collect(),
        None => DEFAULT_SHAPES.to_vec(),
    };
    let soa = args.iter().any(|a| a == "--soa");
    let read_only = args.iter().any(|a| a == "--read-only");
    let reverse = args.iter().any(|a| a == "--reverse");
    let working_set_mib: u64 = args
        .iter()
        .find_map(|a| {
            a.strip_prefix("--working-set-mib=")
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0);
    if read_only && !soa {
        panic!("--read-only requires --soa");
    }
    if read_only && !elfs.is_empty() {
        panic!("--read-only rejects --elf (RO arms come from gemv_mq4g256v2_soa_test.hip only)");
    }
    Cli {
        elfs,
        hipcc_extra,
        iters,
        seed,
        shapes,
        soa,
        read_only,
        working_set_mib,
        reverse,
    }
}

fn hipcc_compile(
    src: &std::path::Path,
    out: &std::path::Path,
    arch: &str,
    extra: &[String],
) -> Vec<u8> {
    println!("[gemv-ab] hipcc {} -> {} ...", src.display(), out.display());
    let t_cc = Instant::now();
    let st = Command::new("hipcc")
        .args([
            "--genco",
            &format!("--offload-arch={arch}"),
            "-O3",
            "--no-offload-compress",
        ])
        .args(extra)
        .arg("-o")
        .arg(out.to_str().unwrap())
        .arg(src.to_str().unwrap())
        .status()
        .expect("hipcc spawn");
    assert!(st.success(), "hipcc failed for {}", src.display());
    println!("[gemv-ab] hipcc {:.2}s", t_cc.elapsed().as_secs_f32());
    let bytes = std::fs::read(out).expect("read hsaco");
    println!(
        "[gemv-ab] hsaco md5={} ({}B) src={}",
        md5hex(&bytes),
        bytes.len(),
        src.display()
    );
    bytes
}

fn load_arm(
    hip: &HipRuntime,
    used_names: &mut Vec<String>,
    stem: &str,
    bytes: &[u8],
    src: &str,
    symbol: &str,
    layout_soa: bool,
    read_only: bool,
) -> Arm {
    let mut name = stem.to_string();
    let mut n = 2;
    while used_names.iter().any(|u| *u == name) {
        name = format!("{stem}-{n}");
        n += 1;
    }
    used_names.push(name.clone());
    let md5 = md5hex(bytes);
    println!(
        "[gemv-ab] arm {name}: md5={md5} ({}B) src={src} symbol={symbol} soa={layout_soa} ro={read_only}",
        bytes.len()
    );
    let module = hip.module_load_data(bytes).expect("module_load_data");
    let func = hip
        .module_get_function(&module, symbol)
        .unwrap_or_else(|e| panic!("symbol {symbol} in arm {name}: {e}"));
    Arm {
        name,
        md5,
        module,
        func,
        layout_soa,
        read_only,
    }
}

fn launch_once(
    hip: &HipRuntime,
    stream: &Stream,
    arm: &Arm,
    d_a: &DeviceBuffer,
    d_x: &DeviceBuffer,
    d_y: &DeviceBuffer,
    m: usize,
    k: usize,
) {
    let mut gargs = GemvArgs {
        a: d_a.as_ptr() as u64,
        x: d_x.as_ptr() as u64,
        y: d_y.as_ptr() as u64,
        m: m as i32,
        k: k as i32,
    };
    let mut p = gargs.ptrs();
    // SAFETY: module loaded, args are live device pointers/values.
    unsafe {
        hip.launch_kernel(
            &arm.func,
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            Some(stream),
            &mut p,
        )
        .expect("launch");
    }
}

fn dtoh_y(hip: &HipRuntime, d_y: &DeviceBuffer, m: usize) -> Vec<u32> {
    let mut raw = vec![0u8; m * 4];
    hip.memcpy_dtoh(&mut raw, d_y).expect("dtoh y");
    // Decode LE u32 words explicitly; Vec<u8> is not guaranteed u32-aligned.
    raw.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bits_equal(a: &[u32], b: &[u32]) -> (usize, Option<(usize, u32, u32)>) {
    let mut nmismatch = 0usize;
    let mut first = None;
    for (r, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            nmismatch += 1;
            if first.is_none() {
                first = Some((r, x, y));
            }
        }
    }
    (nmismatch, first)
}

fn main() {
    let cli = parse_args();
    let mut any_fail = false;

    let hip = HipRuntime::load().expect("HipRuntime::load");
    hip.set_device(0).expect("set_device(0)");
    let arch = hip.get_arch(0).unwrap_or_else(|_| "gfx1201".to_string());
    println!(
        "[gemv-ab] hip arch={arch} iters={} seed={} soa={} read_only={} working_set_mib={} reverse={}",
        cli.iters, cli.seed, cli.soa, cli.read_only, cli.working_set_mib, cli.reverse
    );
    let stream = hip.stream_create().expect("stream_create");

    let kroot = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels/src");
    let tmp = std::env::temp_dir().join("gemv_mq4_ab");
    std::fs::create_dir_all(&tmp).unwrap();

    let mut used_names: Vec<String> = Vec::new();
    let mut arms: Vec<Arm> = Vec::new();

    if cli.read_only {
        // RO AoS + RO SoA from the lab-only soa_test file.
        let ksrc = kroot.join("gemv_mq4g256v2_soa_test.hip");
        let hsaco = tmp.join("gemv_mq4g256v2_soa_test.hsaco");
        let bytes = hipcc_compile(&ksrc, &hsaco, &arch, &cli.hipcc_extra);
        arms.push(load_arm(
            &hip,
            &mut used_names,
            "ro-aos",
            &bytes,
            &format!("hipcc {}", hsaco.display()),
            "gemv_mq4g256v2_ro",
            false,
            true,
        ));
        arms.push(load_arm(
            &hip,
            &mut used_names,
            "ro-soa",
            &bytes,
            &format!("hipcc {}", hsaco.display()),
            "gemv_mq4g256v2_soa_ro",
            true,
            true,
        ));
    } else {
        // Production AoS hipcc baseline.
        let ksrc = kroot.join("gemv_mq4g256v2.hip");
        let hsaco = tmp.join("gemv_mq4g256v2.hsaco");
        let hipcc_bytes = hipcc_compile(&ksrc, &hsaco, &arch, &cli.hipcc_extra);
        arms.push(load_arm(
            &hip,
            &mut used_names,
            "hipcc",
            &hipcc_bytes,
            &format!("hipcc {}", hsaco.display()),
            "gemv_mq4g256v2",
            false,
            false,
        ));
        if cli.soa {
            let ksrc = kroot.join("gemv_mq4g256v2_soa_test.hip");
            let hsaco = tmp.join("gemv_mq4g256v2_soa.hsaco");
            let soa_bytes = hipcc_compile(&ksrc, &hsaco, &arch, &cli.hipcc_extra);
            arms.push(load_arm(
                &hip,
                &mut used_names,
                "soa",
                &soa_bytes,
                &format!("hipcc {}", hsaco.display()),
                "gemv_mq4g256v2_soa",
                true,
                false,
            ));
        }
        for path in &cli.elfs {
            let bytes = std::fs::read(path)
                .unwrap_or_else(|e| panic!("--elf {path}: failed to read ELF bytes: {e}"));
            let stem = PathBuf::from(path)
                .file_stem()
                .unwrap_or_else(|| panic!("--elf {path}: no file stem"))
                .to_string_lossy()
                .to_string();
            arms.push(load_arm(
                &hip,
                &mut used_names,
                &stem,
                &bytes,
                &format!("elf {path}"),
                "gemv_mq4g256v2",
                false,
                false,
            ));
        }
    }

    // Need SoA weight buffers if any arm is SoA (soa mode or read-only).
    let need_soa_buf = arms.iter().any(|a| a.layout_soa);
    let need_aos_buf = arms.iter().any(|a| !a.layout_soa);

    // Geometric-mean time ratio vs baseline arm, accumulated over shapes.
    let mut log_ratio_sum = vec![0f64; arms.len()];
    let mut shape_count = 0usize;

    for (m, k) in &cli.shapes {
        let (m, k) = (*m, *k);
        let groups = k / 256;
        let weight_bytes = m * groups * 136;
        let slots = if cli.working_set_mib == 0 {
            1usize
        } else {
            let want = (cli.working_set_mib as u128) << 20;
            let s = ((want + weight_bytes as u128 - 1) / weight_bytes as u128) as usize;
            s.max(1)
        };
        let ws_bytes = weight_bytes.saturating_mul(slots);
        println!(
            "== shape M={m} K={k} (groups={groups}) weight_bytes={weight_bytes} slots={slots} working_set_bytes={ws_bytes} =="
        );
        let traffic = m as f64 * groups as f64 * 136.0 + k as f64 * 4.0 + m as f64 * 4.0;
        println!("[gemv-ab] logical_bytes/launch={traffic:.0} (analytical; not DRAM)");

        // Build host slots: same logical weights across layouts, seed.wrapping_add(slot).
        let mut host_aos: Vec<Vec<u8>> = Vec::with_capacity(slots);
        let mut host_soa: Vec<Vec<u8>> = Vec::with_capacity(slots);
        let mut host_x: Vec<Vec<f32>> = Vec::with_capacity(slots);
        let mut slot_seed = Vec::with_capacity(slots);
        for slot in 0..slots {
            let s = cli.seed.wrapping_add(slot as u64);
            slot_seed.push(s);
            let (w, x) = build_inputs(m, k, s);
            println!(
                "[gemv-ab] slot={slot} seed={s} aos_md5={} x_md5={}",
                md5hex(&w),
                md5hex(unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) })
            );
            if need_soa_buf {
                let soa = repack_aos_to_soa(&w, m, k);
                let back = repack_soa_to_aos(&soa, m, k);
                if back != w {
                    eprintln!(
                        "[gemv-ab] FAIL inverse layout slot={slot} M={m} K={k}: SoA round-trip != AoS"
                    );
                    any_fail = true;
                } else {
                    println!("[gemv-ab] slot={slot} soa_md5={} inverse=OK", md5hex(&soa));
                }
                host_soa.push(soa);
            }
            host_aos.push(w);
            host_x.push(x);
        }
        // Shared x across layouts (same per slot); upload one x set and cycle if multi-slot.
        // For multi-slot we also rotate x with the weight slot for determinism.

        // Device weight slots (distinct allocations per arm layout).
        let mut d_aos: Vec<DeviceBuffer> = Vec::new();
        let mut d_soa: Vec<DeviceBuffer> = Vec::new();
        if need_aos_buf {
            for w in &host_aos {
                let d = hip.malloc(w.len()).expect("malloc A aos");
                hip.memcpy_htod(&d, w).expect("htod A aos");
                d_aos.push(d);
            }
        }
        if need_soa_buf {
            for w in &host_soa {
                let d = hip.malloc(w.len()).expect("malloc A soa");
                hip.memcpy_htod(&d, w).expect("htod A soa");
                d_soa.push(d);
            }
        }
        let mut d_xs: Vec<DeviceBuffer> = Vec::new();
        for x in &host_x {
            let x_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
            let d = hip.malloc(x_bytes.len()).expect("malloc x");
            hip.memcpy_htod(&d, x_bytes).expect("htod x");
            d_xs.push(d);
        }
        // One y buffer per arm (reused across slots during verify/timing).
        let mut d_ys: Vec<DeviceBuffer> = Vec::new();
        for _ in &arms {
            d_ys.push(hip.malloc(m * 4).expect("malloc y"));
        }

        // ——— verify ALL slots before any timing ———
        // Per slot: run every arm, bit-compare all rows to baseline; CPU oracle on rows 0,M-1.
        // ok-only: detail is printed immediately, not retained.
        let mut slot_verifies: Vec<bool> = Vec::new();
        for slot in 0..slots {
            let mut ref_bits: Option<Vec<u32>> = None;
            let mut slot_ok = true;
            let mut details = Vec::new();
            for (ai, arm) in arms.iter().enumerate() {
                let d_a = if arm.layout_soa {
                    &d_soa[slot]
                } else {
                    &d_aos[slot]
                };
                launch_once(&hip, &stream, arm, d_a, &d_xs[slot], &d_ys[ai], m, k);
                hip.stream_synchronize(&stream).expect("verify sync");
                let bits = dtoh_y(&hip, &d_ys[ai], m);
                if ai == 0 {
                    // CPU oracle on baseline arm, rows 0 and M-1.
                    let host_w = if arm.layout_soa {
                        &host_soa[slot]
                    } else {
                        &host_aos[slot]
                    };
                    for &row in &[0usize, m - 1] {
                        if arm.read_only {
                            let expect = cpu_row_ro(host_w, row, k, arm.layout_soa);
                            let got = bits[row];
                            if got != expect {
                                slot_ok = false;
                                any_fail = true;
                                details.push(format!(
                                    "CPU-RO FAIL arm={} slot={slot} row={row} expect={expect:#010x} got={got:#010x}",
                                    arm.name
                                ));
                            }
                        } else {
                            let expect = cpu_row(host_w, &host_x[slot], row, k);
                            let got = f32::from_bits(bits[row]);
                            let abs_err = (got - expect).abs();
                            let rel = (abs_err as f64) / (expect.abs() as f64).max(1e-30);
                            // CPU unfused vs GPU contracted FMA: near-zero
                            // cancellation can inflate relative error while
                            // abs stays tiny. Fail only when both abs>1e-5
                            // AND rel>1e-4. Finite gates keep NaN open-fail.
                            let ok = expect.is_finite()
                                && got.is_finite()
                                && abs_err.is_finite()
                                && rel.is_finite()
                                && !((abs_err as f64) > 1e-5 && rel > 1e-4);
                            if !ok {
                                slot_ok = false;
                                any_fail = true;
                                details.push(format!(
                                    "CPU FAIL arm={} slot={slot} row={row} expect={expect} got={got} abs={abs_err} rel={rel:.3e}",
                                    arm.name
                                ));
                            }
                        }
                    }
                    ref_bits = Some(bits);
                    details.push(format!("arm={} ref", arm.name));
                } else {
                    let r = ref_bits.as_ref().unwrap();
                    let (nmis, first) = bits_equal(r, &bits);
                    if nmis == 0 {
                        details.push(format!("arm={} bit-exact", arm.name));
                    } else {
                        slot_ok = false;
                        any_fail = true;
                        let (row, a, b) = first.unwrap();
                        details.push(format!(
                            "arm={} DIFF n={nmis} first row {row} (ref={a:#010x} arm={b:#010x})",
                            arm.name
                        ));
                    }
                }
            }
            let detail = details.join("; ");
            println!(
                "[gemv-ab] verify slot={slot} seed={} {} {}",
                slot_seed[slot],
                if slot_ok { "OK" } else { "FAIL" },
                detail
            );
            slot_verifies.push(slot_ok);
        }
        if slot_verifies.iter().any(|&ok| !ok) {
            eprintln!("[gemv-ab] FAIL parity/CPU on M={m} K={k}: skipping timing for this shape");
            // Free device buffers before next shape.
            for d in d_aos {
                let _ = hip.free(d);
            }
            for d in d_soa {
                let _ = hip.free(d);
            }
            for d in d_xs {
                let _ = hip.free(d);
            }
            for d in d_ys {
                let _ = hip.free(d);
            }
            continue;
        }

        // Timed iters rounded up to full rotations over slots.
        let rotations = (cli.iters + slots - 1) / slots;
        let timed_iters = rotations * slots;
        println!(
            "[gemv-ab] timing rotations={rotations} timed_iters={timed_iters} (requested iters={})",
            cli.iters
        );

        // Arm order for timing follows reverse flag; baseline ratio still arms[0].
        let mut order: Vec<usize> = (0..arms.len()).collect();
        if cli.reverse {
            order.reverse();
        }
        // Fixed wall-clock warm per arm immediately before its timed windows.
        const WARMUP_MS: u64 = 250;
        println!("[gemv-ab] per-arm wall warmup_ms={WARMUP_MS}");

        struct Row {
            name: String,
            us: f64,
            logical_gbs: f64,
            ratio: f64,
            exact: String,
            samples_us: Vec<f64>,
            host_submit_us: f64,
        }
        let mut rows: Vec<Option<Row>> = (0..arms.len()).map(|_| None).collect();
        let mut baseline_us = 0f64;

        for &ai in &order {
            let arm = &arms[ai];
            // Warm this arm alone ≥WARMUP_MS wall time: full timed_iters
            // rotations, one stream_synchronize per warm batch. No events.
            let warm_for = Duration::from_millis(WARMUP_MS);
            let t_warm = Instant::now();
            while t_warm.elapsed() < warm_for {
                for it in 0..timed_iters {
                    let slot = it % slots;
                    let d_a = if arm.layout_soa {
                        &d_soa[slot]
                    } else {
                        &d_aos[slot]
                    };
                    launch_once(&hip, &stream, arm, d_a, &d_xs[slot], &d_ys[ai], m, k);
                }
                hip.stream_synchronize(&stream)
                    .expect("per-arm warmup sync");
            }
            // No host uploads/mallocs inside the event window: all d_* live.
            // Three raw whole-batch GPU windows (timed_iters launches each,
            // round-robin slots). No per-rotation/per-kernel sync. Headline =
            // median us/launch. Host submit metrics stay separate.
            const N_BATCHES: usize = 3;
            let start = hip.event_create().expect("event_create");
            let stop = hip.event_create().expect("event_create");
            let mut gpu_samples_us = Vec::with_capacity(N_BATCHES);
            let mut host_samples_us = Vec::with_capacity(N_BATCHES * timed_iters);
            let mut host_submit_total = 0f64;
            let t_host_all = Instant::now();
            for _batch in 0..N_BATCHES {
                hip.event_record(&start, Some(&stream))
                    .expect("event_record");
                for it in 0..timed_iters {
                    let slot = it % slots;
                    let d_a = if arm.layout_soa {
                        &d_soa[slot]
                    } else {
                        &d_aos[slot]
                    };
                    let t_sub = Instant::now();
                    launch_once(&hip, &stream, arm, d_a, &d_xs[slot], &d_ys[ai], m, k);
                    let sub_us = t_sub.elapsed().as_secs_f64() * 1e6;
                    host_submit_total += sub_us;
                    host_samples_us.push(sub_us);
                }
                hip.event_record(&stop, Some(&stream))
                    .expect("event_record");
                // One sync per whole batch — not per rotation/launch.
                hip.event_synchronize(&stop).expect("event_sync");
                let ms = hip
                    .event_elapsed_ms(&start, &stop)
                    .expect("event_elapsed_ms") as f64;
                // us per launch from the whole-batch window.
                let batch_us = ms * 1e3 / timed_iters as f64;
                gpu_samples_us.push(batch_us);
            }
            hip.event_destroy(start).unwrap();
            hip.event_destroy(stop).unwrap();
            let host_window_us = t_host_all.elapsed().as_secs_f64() * 1e6;
            // Median of the 3 raw batch us/launch samples (not mean/min).
            let mut sorted = gpu_samples_us.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let us = sorted[sorted.len() / 2];
            let logical_gbs = traffic / us / 1e3;
            // Mean host submit across all launches in all batches (separate metric).
            let host_submit_us = host_submit_total / (N_BATCHES * timed_iters) as f64;

            // Log sample identity and host submission times.
            eprintln!(
                "[gemv-ab] arm={} host_submit_mean={host_submit_us:.3}us host_window={host_window_us:.1}us",
                arm.name
            );
            eprint!("[gemv-ab] arm={} raw_host_submit_us=", arm.name);
            for (i, s) in host_samples_us.iter().enumerate() {
                if i > 0 {
                    eprint!(",");
                }
                eprint!("{s:.4}");
            }
            eprintln!();
            eprint!("[gemv-ab] arm={} raw_gpu_us_per_launch=", arm.name);
            for (i, s) in gpu_samples_us.iter().enumerate() {
                if i > 0 {
                    eprint!(",");
                }
                eprint!("{s:.4}");
            }
            eprintln!();
            eprint!("[gemv-ab] arm={} sample_slot_seq=", arm.name);
            for it in 0..timed_iters {
                if it > 0 {
                    eprint!(",");
                }
                eprint!("{}", it % slots);
            }
            eprintln!();

            let exact = if ai == 0 {
                baseline_us = us;
                "ref".to_string()
            } else {
                "pre-verified".to_string()
            };
            let ratio = if ai == 0 {
                1.0
            } else if baseline_us > 0.0 {
                us / baseline_us
            } else {
                // Baseline timed after this arm under --reverse: fill later.
                f64::NAN
            };
            if ai > 0 && ratio.is_finite() {
                log_ratio_sum[ai] += ratio.ln();
            }
            rows[ai] = Some(Row {
                name: arm.name.clone(),
                us,
                logical_gbs,
                ratio,
                exact,
                samples_us: gpu_samples_us,
                host_submit_us,
            });
        }

        // Fix ratios if --reverse timed baseline after others.
        if let Some(b) = &rows[0] {
            baseline_us = b.us;
        }
        for (ai, slot) in rows.iter_mut().enumerate() {
            if ai == 0 {
                continue;
            }
            if let Some(r) = slot.as_mut() {
                if !r.ratio.is_finite() && baseline_us > 0.0 {
                    r.ratio = r.us / baseline_us;
                    log_ratio_sum[ai] += r.ratio.ln();
                }
            }
        }

        shape_count += 1;
        println!(
            "{:<24} {:>10} {:>12} {:>8} {:>10}  {}",
            "arm", "us/launch", "logicalGB/s", "vs-base", "host_sub_us", "exactness"
        );
        // Print in original arm order for stable tables.
        for ai in 0..arms.len() {
            let r = rows[ai].as_ref().unwrap();
            println!(
                "{:<24} {:>10.2} {:>12.1} {:>8.3} {:>10.3}  {}",
                r.name, r.us, r.logical_gbs, r.ratio, r.host_submit_us, r.exact
            );
            // Raw GPU us/launch samples already printed during timing; restate median.
            println!(
                "[gemv-ab] arm={} gpu_us_per_launch_median={:.4} n_samples={} (3 whole-batch windows; no min/best-of)",
                r.name,
                r.us,
                r.samples_us.len()
            );
        }

        // Free device buffers for this shape.
        for d in d_aos {
            let _ = hip.free(d);
        }
        for d in d_soa {
            let _ = hip.free(d);
        }
        for d in d_xs {
            let _ = hip.free(d);
        }
        for d in d_ys {
            let _ = hip.free(d);
        }
    }

    println!("[summary] geometric-mean time ratio vs baseline over {shape_count} shapes:");
    for (ai, arm) in arms.iter().enumerate() {
        if ai == 0 {
            println!(
                "[summary] arm={:<20} md5={} ratio=1.000 (reference) route={}",
                arm.name,
                arm.md5,
                if arm.read_only {
                    "ro"
                } else if arm.layout_soa {
                    "soa"
                } else {
                    "aos"
                }
            );
        } else if shape_count > 0 {
            let geomean = (log_ratio_sum[ai] / shape_count as f64).exp();
            println!(
                "[summary] arm={:<20} md5={} ratio={geomean:.4} route={}",
                arm.name,
                arm.md5,
                if arm.read_only {
                    "ro-soa"
                } else if arm.layout_soa {
                    "soa"
                } else {
                    "aos-elf"
                }
            );
        }
    }

    if any_fail {
        eprintln!("[gemv-ab] FAIL: parity or CPU oracle failure(s); exiting nonzero");
        std::process::exit(1);
    }
}
