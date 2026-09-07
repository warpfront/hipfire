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
//! Run (device 0, GPU lock held by the caller):
//! ```sh
//! flock -w 300 /tmp/hipfire-gpu.lock env HIP_VISIBLE_DEVICES=0 \
//!   ./target/release/examples/gemv_mq4_ab --iters=100
//! flock -w 300 /tmp/hipfire-gpu.lock env HIP_VISIBLE_DEVICES=0 \
//!   ./target/release/examples/gemv_mq4_ab --iters=100 \
//!   --elf=kernels/rust/target/gfx1201/gemv_mq4g256v2-fpc.elf
//! ```
//!
//! No feature gate: `hip-bridge` is an unconditional `saddle-lab` dependency.

use hip_bridge::{Function, HipRuntime, Module};
use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

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
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
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
            0xd76a_a478, 0xe8c7_b756, 0x2420_70db, 0xc1bd_ceee, 0xf57c_0faf, 0x4787_c62a,
            0xa830_4613, 0xfd46_9501, 0x6980_98d8, 0x8b44_f7af, 0xffff_5bb1, 0x895c_d7be,
            0x6b90_1122, 0xfd98_7193, 0xa679_438e, 0x49b4_0821, 0xf61e_2562, 0xc040_b340,
            0x265e_5a51, 0xe9b6_c7aa, 0xd62f_105d, 0x0244_1453, 0xd8a1_e681, 0xe7d3_fbc8,
            0x21e1_cde6, 0xc337_07d6, 0xf4d5_0d87, 0x455a_14ed, 0xa9e3_e905, 0xfcef_a3f8,
            0x676f_02d9, 0x8d2a_4c8a, 0xfffa_3942, 0x8771_f681, 0x6d9d_6122, 0xfde5_380c,
            0xa4be_ea44, 0x4bde_cfa9, 0xf6bb_4b60, 0xbebf_bc70, 0x289b_7ec6, 0xeaa1_27fa,
            0xd4ef_3085, 0x0488_1d05, 0xd9d4_d039, 0xe6db_99e5, 0x1fa2_7cf8, 0xc4ac_5665,
            0xf429_2244, 0x432a_ff97, 0xab94_23a7, 0xfc93_a039, 0x655b_59c3, 0x8f0c_cc92,
            0xffef_f47d, 0x8584_5dd1, 0x6fa8_7e4f, 0xfe2c_e6e0, 0xa301_4314, 0x4e08_11a1,
            0xf753_7e82, 0xbd3a_f235, 0x2ad7_d2bb, 0xeb86_d391,
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
    Md5::digest(data).iter().map(|x| format!("{x:02x}")).collect()
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
                let packed =
                    f32_to_f16_bits(sc) as u32 | ((f32_to_f16_bits(zp) as u32) << 16);
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

fn parse_args() -> (Vec<String>, Vec<String>, usize, u64, Vec<(usize, usize)>) {
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
    let shapes: Vec<(usize, usize)> = match args
        .iter()
        .find_map(|a| a.strip_prefix("--shapes="))
    {
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
    (elfs, hipcc_extra, iters, seed, shapes)
}

fn main() {
    let (elfs, hipcc_extra, iters, seed, shapes) = parse_args();

    let hip = HipRuntime::load().expect("HipRuntime::load");
    hip.set_device(0).expect("set_device(0)");
    let arch = hip.get_arch(0).unwrap_or_else(|_| "gfx1201".to_string());
    println!("[gemv-ab] hip arch={arch} iters={iters} seed={seed}");
    let stream = hip.stream_create().expect("stream_create");

    // ——— hipcc arm (production flags) ———
    let ksrc =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels/src/gemv_mq4g256v2.hip");
    let tmp = std::env::temp_dir().join("gemv_mq4_ab");
    std::fs::create_dir_all(&tmp).unwrap();
    let hsaco_path = tmp.join("gemv_mq4g256v2.hsaco");
    println!("[gemv-ab] hipcc {} ...", ksrc.display());
    let t_cc = Instant::now();
    let st = Command::new("hipcc")
        .args([
            "--genco",
            &format!("--offload-arch={arch}"),
            "-O3",
            "--no-offload-compress",
        ])
        .args(&hipcc_extra)
        .arg("-o")
        .arg(hsaco_path.to_str().unwrap())
        .arg(ksrc.to_str().unwrap())
        .status()
        .expect("hipcc spawn");
    assert!(st.success(), "hipcc failed");
    println!("[gemv-ab] hipcc {:.2}s", t_cc.elapsed().as_secs_f32());
    let hipcc_bytes = std::fs::read(&hsaco_path).expect("read hipcc hsaco");
    println!(
        "[gemv-ab] hipcc hsaco md5={} ({}B)",
        md5hex(&hipcc_bytes),
        hipcc_bytes.len()
    );

    // ——— load every arm through the same HIP dispatcher ———
    let mut used_names: Vec<String> = Vec::new();
    let mut arm_sources: Vec<(String, Vec<u8>, String)> = vec![(
        "hipcc".to_string(),
        hipcc_bytes,
        format!("hipcc {}", hsaco_path.display()),
    )];
    for path in &elfs {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("--elf {path}: failed to read ELF bytes: {e}"));
        let stem = PathBuf::from(path)
            .file_stem()
            .unwrap_or_else(|| panic!("--elf {path}: no file stem"))
            .to_string_lossy()
            .to_string();
        arm_sources.push((stem, bytes, format!("elf {path}")));
    }
    let mut arms: Vec<Arm> = Vec::new();
    for (stem, bytes, src) in arm_sources {
        let mut name = stem.clone();
        let mut n = 2;
        while used_names.iter().any(|u| *u == name) {
            name = format!("{stem}-{n}");
            n += 1;
        }
        used_names.push(name.clone());
        let md5 = md5hex(&bytes);
        println!("[gemv-ab] arm {name}: md5={md5} ({}B) src={src}", bytes.len());
        let module = hip.module_load_data(&bytes).expect("module_load_data");
        let func = hip
            .module_get_function(&module, "gemv_mq4g256v2")
            .expect("gemv_mq4g256v2");
        arms.push(Arm {
            name,
            md5,
            module,
            func,
        });
    }

    // Geometric-mean time ratio vs hipcc, accumulated over shapes.
    let mut log_ratio_sum = vec![0f64; arms.len()];
    let mut shape_count = 0usize;

    for (m, k) in &shapes {
        let (m, k) = (*m, *k);
        let groups = k / 256;
        println!("== shape M={m} K={k} (groups={groups}) ==");
        let traffic =
            m as f64 * groups as f64 * 136.0 + k as f64 * 4.0 + m as f64 * 4.0;

        // Build + upload inputs once per shape.
        let (w, x) = build_inputs(m, k, seed);
        let d_a = hip.malloc(w.len()).expect("malloc A");
        hip.memcpy_htod(&d_a, &w).expect("htod A");
        let x_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4)
        };
        let d_x = hip.malloc(x_bytes.len()).expect("malloc x");
        hip.memcpy_htod(&d_x, x_bytes).expect("htod x");

        // Per arm: 3 warmups, then event-timed `iters` launches.
        struct Row {
            name: String,
            us: f64,
            gbs: f64,
            ratio: f64,
            exact: String,
        }
        let mut rows: Vec<Row> = Vec::new();
        let mut ref_y: Vec<f32> = Vec::new();
        for (ai, arm) in arms.iter().enumerate() {
            let d_y = hip.malloc(m * 4).expect("malloc y");
            let mut gargs = GemvArgs {
                a: d_a.as_ptr() as u64,
                x: d_x.as_ptr() as u64,
                y: d_y.as_ptr() as u64,
                m: m as i32,
                k: k as i32,
            };
            for _ in 0..3 {
                let mut p = gargs.ptrs();
                // SAFETY: module loaded, args are live device pointers/values.
                unsafe {
                    hip.launch_kernel(&arm.func, [m as u32, 1, 1], [32, 1, 1], 0, Some(&stream), &mut p)
                        .expect("warmup launch");
                }
            }
            hip.stream_synchronize(&stream).expect("warmup sync");
            let start = hip.event_create().expect("event_create");
            let stop = hip.event_create().expect("event_create");
            hip.event_record(&start, Some(&stream))
                .expect("event_record");
            let t_host = std::time::Instant::now();
            for _ in 0..iters {
                let mut p = gargs.ptrs();
                // SAFETY: same as above; buffers still live.
                unsafe {
                    hip.launch_kernel(&arm.func, [m as u32, 1, 1], [32, 1, 1], 0, Some(&stream), &mut p)
                        .expect("timed launch");
                }
            }
            let host_us = t_host.elapsed().as_secs_f64() * 1e6 / iters as f64;
            hip.event_record(&stop, Some(&stream))
                .expect("event_record");
            hip.event_synchronize(&stop).expect("event_sync");
            eprintln!("[gemv-ab] host launch {host_us:.2} us/launch");
            let ms = hip
                .event_elapsed_ms(&start, &stop)
                .expect("event_elapsed_ms") as f64;
            hip.event_destroy(start).unwrap();
            hip.event_destroy(stop).unwrap();
            let us = ms * 1e3 / iters as f64;
            let gbs = traffic / us / 1e3;

            let mut raw = vec![0u8; m * 4];
            hip.memcpy_dtoh(&mut raw, &d_y).expect("dtoh y");
            // SAFETY: kernel wrote M f32 elements; length checked above.
            let y: Vec<f32> = unsafe {
                std::slice::from_raw_parts(raw.as_ptr() as *const f32, m).to_vec()
            };

            let (ratio, exact) = if ai == 0 {
                ref_y = y;
                (1.0, "ref".to_string())
            } else {
                let mut nmismatch = 0usize;
                let mut first: Option<(usize, f32, f32)> = None;
                let mut max_rel = 0f64;
                for (r, (&a, &b)) in ref_y.iter().zip(y.iter()).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        nmismatch += 1;
                        if first.is_none() {
                            first = Some((r, a, b));
                        }
                        let rel = ((a - b).abs() as f64) / (a.abs() as f64).max(1e-30);
                        if rel > max_rel {
                            max_rel = rel;
                        }
                    }
                }
                let s = if nmismatch == 0 {
                    "bit-exact".to_string()
                } else {
                    let (r, a, b) = first.unwrap();
                    format!(
                        "DIFF n={nmismatch} first row {r} (hipcc={a:.6} arm={b:.6}) max|rel|={max_rel:.2e}"
                    )
                };
                (us / rows[0].us, s)
            };
            if ai > 0 {
                log_ratio_sum[ai] += (ratio as f64).ln();
            }
            rows.push(Row {
                name: arm.name.clone(),
                us,
                gbs,
                ratio,
                exact,
            });
        }
        shape_count += 1;

        println!(
            "{:<24} {:>10} {:>8} {:>8}  {}",
            "arm", "us/launch", "GB/s", "vs-hipcc", "exactness"
        );
        for r in &rows {
            println!(
                "{:<24} {:>10.2} {:>8.1} {:>8.3}  {}",
                r.name, r.us, r.gbs, r.ratio, r.exact
            );
        }
        // CPU sanity check on the hipcc arm (rows 0 and M-1).
        let mut worst_abs = 0f32;
        let mut worst_rel = 0f64;
        for &row in &[0, m - 1] {
            let expect = cpu_row(&w, &x, row, k);
            let got = ref_y[row];
            worst_abs = worst_abs.max((got - expect).abs());
            worst_rel = worst_rel.max(((got - expect).abs() as f64) / (expect.abs() as f64).max(1e-30));
        }
        let verdict = if worst_rel <= 1e-4 { "OK" } else { "FAIL" };
        println!(
            "[cpu-check] M={m} K={k} rows 0,M-1 maxabserr={worst_abs:.3e} maxrelerr={worst_rel:.3e} {verdict}"
        );
    }

    println!("[summary] geometric-mean time ratio vs hipcc over {shape_count} shapes:");
    for (ai, arm) in arms.iter().enumerate() {
        if ai == 0 {
            println!(
                "[summary] arm={:<20} md5={} ratio=1.000 (reference)",
                arm.name, arm.md5
            );
        } else {
            let geomean = (log_ratio_sum[ai] / shape_count as f64).exp();
            println!(
                "[summary] arm={:<20} md5={} ratio={geomean:.4}",
                arm.name, arm.md5
            );
        }
    }
}
