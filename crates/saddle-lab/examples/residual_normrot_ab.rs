// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `residual_normrot_ab`: A/B harness for the folded residual-GEMV +
//! RMSNorm/FWHT-rotate kernel.
//!
//! Arm `baseline` compiles `kernels/src/gemv_mq4g256v2_residual.hip`
//! (symbol `gemv_mq4g256v2_residual`, grid [M,1,1], block [32,1,1]) and
//! `kernels/src/fused_rmsnorm_mq_rotate.hip` (symbol
//! `fused_rmsnorm_mq_rotate`, grid [1,1,1], block [256,1,1], shared
//! (M+256)*4 as production) and runs them as two launches. Arm `fold`
//! compiles `kernels/src/gemv_mq4g256v2_residual_normrot.hip` (symbol
//! `gemv_mq4g256v2_residual_normrot`, grid [(M+1)/2,1,1], block [32,1,1])
//! as one launch (ctr zeroed once; the kernel resets it).
//!
//! Inputs seeded as in gemv_residual_ab.rs (weights, x, y0 random) plus
//! nw in [0.5,1.5], s1/s2 in {-1,1}, eps=1e-6. x_rot AND y are compared
//! bit-exact (to_bits). Timing runs N=200 iterations WITHOUT re-upload
//! for both arms (y accumulates identically in both arms so results stay
//! comparable; exactness is checked after 1 iteration, then timed).
//!
//! Run (device 0, GPU lock held by the caller):
//! ```sh
//! flock -w 300 /tmp/hipfire-gpu.lock env HIP_VISIBLE_DEVICES=0 \
//!   ./target/release/examples/residual_normrot_ab
//! ```
//!
//! No feature gate: `hip-bridge` is an unconditional `saddle-lab` dependency.

use hip_bridge::{Function, HipRuntime, Module};
use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

/// GEMV shapes (M rows, K cols); the norm length is M in both.
const DEFAULT_SHAPES: [(usize, usize); 2] = [(5120, 5120), (5120, 17408)];

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

/// Explicit args of the residual kernel, in kernel signature order
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

/// Explicit args of `fused_rmsnorm_mq_rotate`, in kernel signature order
/// (x@0, weight@8, signs1@16, signs2@24, x_rot@32, K@40, eps@44).
struct NormArgs {
    x: u64,
    w: u64,
    s1: u64,
    s2: u64,
    xr: u64,
    k: i32,
    eps: f32,
}

impl NormArgs {
    fn ptrs(&mut self) -> Vec<*mut c_void> {
        vec![
            (&mut self.x as *mut u64).cast(),
            (&mut self.w as *mut u64).cast(),
            (&mut self.s1 as *mut u64).cast(),
            (&mut self.s2 as *mut u64).cast(),
            (&mut self.xr as *mut u64).cast(),
            (&mut self.k as *mut i32).cast(),
            (&mut self.eps as *mut f32).cast(),
        ]
    }
}

/// Explicit args of `gemv_mq4g256v2_residual_normrot`, in kernel signature
/// order (A@0, x@8, y@16, M@24, K@28, nw@32, s1@40, s2@48, x_rot@56,
/// eps@60, ctr@64).
struct FoldArgs {
    a: u64,
    x: u64,
    y: u64,
    m: i32,
    k: i32,
    nw: u64,
    s1: u64,
    s2: u64,
    xr: u64,
    eps: f32,
    ctr: u64,
}

impl FoldArgs {
    fn ptrs(&mut self) -> Vec<*mut c_void> {
        vec![
            (&mut self.a as *mut u64).cast(),
            (&mut self.x as *mut u64).cast(),
            (&mut self.y as *mut u64).cast(),
            (&mut self.m as *mut i32).cast(),
            (&mut self.k as *mut i32).cast(),
            (&mut self.nw as *mut u64).cast(),
            (&mut self.s1 as *mut u64).cast(),
            (&mut self.s2 as *mut u64).cast(),
            (&mut self.xr as *mut u64).cast(),
            (&mut self.eps as *mut f32).cast(),
            (&mut self.ctr as *mut u64).cast(),
        ]
    }
}

/// One loaded kernel: the bytes plus its resolved entry point. `module` is
/// never read after load — it is kept alive because unloading would
/// invalidate `func`.
struct Kern {
    #[allow(dead_code)]
    md5: String,
    #[allow(dead_code)]
    module: Module,
    func: Function,
}

fn f32_slice_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// Build seeded inputs: MQ4 weights (`M * (K/256) * 136` bytes), GEMV `x`
/// (`K` f32 in [-1, 1]), initial `y` (`M` f32 in [-1, 1]), norm weight `nw`
/// (`M` f32 in [0.5, 1.5]), and `s1`/`s2` (256 f32 each in {-1, +1}).
fn build_inputs(
    m: usize,
    k: usize,
    seed: u64,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(k % 256 == 0, "K={k} must be a multiple of 256");
    assert!(m % 256 == 0, "M={m} must be a multiple of 256");
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
    let mut y0 = vec![0f32; m];
    for v in y0.iter_mut() {
        *v = rng.next_unit() * 2.0 - 1.0;
    }
    let mut nw = vec![0f32; m];
    for v in nw.iter_mut() {
        *v = 0.5 + rng.next_unit();
    }
    let mut s1 = vec![0f32; 256];
    let mut s2 = vec![0f32; 256];
    for v in s1.iter_mut().chain(s2.iter_mut()) {
        *v = if rng.next_byte() & 1 == 0 { -1.0 } else { 1.0 };
    }
    (w, x, y0, nw, s1, s2)
}

fn parse_args() -> (usize, u64, Vec<(usize, usize)>) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let iters: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("--iters=").and_then(|v| v.parse().ok()))
        .unwrap_or(200);
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
                assert!(m % 256 == 0, "M={m} must be a multiple of 256");
                (m, k)
            })
            .collect(),
        None => DEFAULT_SHAPES.to_vec(),
    };
    (iters, seed, shapes)
}

fn hipcc_build(arch: &str, src_name: &str, out_name: &str) -> Vec<u8> {
    let ksrc =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../kernels/src/{src_name}"));
    let tmp = std::env::temp_dir().join("residual_normrot_ab");
    std::fs::create_dir_all(&tmp).unwrap();
    let hsaco_path = tmp.join(out_name);
    println!("[rn-ab] hipcc {} ...", ksrc.display());
    let t_cc = Instant::now();
    let st = Command::new("hipcc")
        .args([
            "--genco",
            &format!("--offload-arch={arch}"),
            "-O3",
            "--no-offload-compress",
        ])
        .arg("-o")
        .arg(hsaco_path.to_str().unwrap())
        .arg(ksrc.to_str().unwrap())
        .status()
        .expect("hipcc spawn");
    assert!(st.success(), "hipcc failed for {src_name}");
    println!("[rn-ab] hipcc {src_name} {:.2}s", t_cc.elapsed().as_secs_f32());
    std::fs::read(&hsaco_path).expect("read hsaco")
}

fn dtoh_f32(hip: &HipRuntime, stream: &hip_bridge::Stream, d: &hip_bridge::DeviceBuffer, n: usize) -> Vec<f32> {
    let _ = stream;
    let mut raw = vec![0u8; n * 4];
    hip.memcpy_dtoh(&mut raw, d).expect("dtoh");
    // SAFETY: kernel wrote n f32 elements; length checked above.
    unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, n).to_vec() }
}

fn main() {
    let (iters, seed, shapes) = parse_args();

    let hip = HipRuntime::load().expect("HipRuntime::load");
    hip.set_device(0).expect("set_device(0)");
    let arch = hip.get_arch(0).unwrap_or_else(|_| "gfx1201".to_string());
    println!("[rn-ab] hip arch={arch} iters={iters} seed={seed}");
    let stream = hip.stream_create().expect("stream_create");

    // ——— hipcc all three kernels with the production flags ———
    let res_bytes = hipcc_build(&arch, "gemv_mq4g256v2_residual.hip", "residual.hsaco");
    let nrm_bytes = hipcc_build(&arch, "fused_rmsnorm_mq_rotate.hip", "normrot.hsaco");
    let fold_bytes = hipcc_build(
        &arch,
        "gemv_mq4g256v2_residual_normrot.hip",
        "residual_normrot.hsaco",
    );
    println!(
        "[rn-ab] residual hsaco md5={} ({}B)",
        md5hex(&res_bytes),
        res_bytes.len()
    );
    println!(
        "[rn-ab] normrot  hsaco md5={} ({}B)",
        md5hex(&nrm_bytes),
        nrm_bytes.len()
    );
    println!(
        "[rn-ab] fold     hsaco md5={} ({}B)",
        md5hex(&fold_bytes),
        fold_bytes.len()
    );

    // ——— load through the same HIP dispatcher ———
    let load = |bytes: &[u8], sym: &str| -> Kern {
        let md5 = md5hex(bytes);
        let module = hip.module_load_data(bytes).expect("module_load_data");
        let func = hip
            .module_get_function(&module, sym)
            .unwrap_or_else(|e| panic!("module_get_function {sym}: {e:?}"));
        Kern { md5, module, func }
    };
    let k_res = load(&res_bytes, "gemv_mq4g256v2_residual");
    let k_nrm = load(&nrm_bytes, "fused_rmsnorm_mq_rotate");
    let k_fold = load(&fold_bytes, "gemv_mq4g256v2_residual_normrot");
    let eps: f32 = 1e-6;

    for (m, k) in &shapes {
        let (m, k) = (*m, *k);
        println!("== shape M={m} K={k} (norm length M={m}) ==");
        let shared_nrm = ((m + 256) * 4) as u32; // production shared for grid-1 norm
        assert!(shared_nrm <= 65536, "norm shared {shared_nrm}B exceeds 64K LDS");

        let (w, x, y0, nw, s1, s2) = build_inputs(m, k, seed);
        let d_a = hip.malloc(w.len()).expect("malloc A");
        hip.memcpy_htod(&d_a, &w).expect("htod A");
        let d_x = hip.malloc(x.len() * 4).expect("malloc x");
        hip.memcpy_htod(&d_x, f32_slice_bytes(&x)).expect("htod x");
        let y0_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(y0.as_ptr() as *const u8, y0.len() * 4)
        };
        let d_y = hip.malloc(y0_bytes.len()).expect("malloc y");
        let d_nw = hip.malloc(nw.len() * 4).expect("malloc nw");
        hip.memcpy_htod(&d_nw, f32_slice_bytes(&nw)).expect("htod nw");
        let d_s1 = hip.malloc(256 * 4).expect("malloc s1");
        hip.memcpy_htod(&d_s1, f32_slice_bytes(&s1)).expect("htod s1");
        let d_s2 = hip.malloc(256 * 4).expect("malloc s2");
        hip.memcpy_htod(&d_s2, f32_slice_bytes(&s2)).expect("htod s2");
        let d_xr_base = hip.malloc(m * 4).expect("malloc x_rot baseline");
        let d_xr_fold = hip.malloc(m * 4).expect("malloc x_rot fold");
        let d_ctr = hip.malloc(4).expect("malloc ctr");

        let grid_res: [u32; 3] = [m as u32, 1, 1];
        let grid_fold: [u32; 3] = [(m as u32 + 1) / 2, 1, 1];

        // ——— bit-exactness after ONE iteration each ———
        // Baseline pair.
        hip.memcpy_htod(&d_y, y0_bytes).expect("htod y0");
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
            hip.launch_kernel(&k_res.func, grid_res, [32, 1, 1], 0, Some(&stream), &mut p)
                .expect("baseline res launch");
        }
        let mut nargs = NormArgs {
            x: d_y.as_ptr() as u64,
            w: d_nw.as_ptr() as u64,
            s1: d_s1.as_ptr() as u64,
            s2: d_s2.as_ptr() as u64,
            xr: d_xr_base.as_ptr() as u64,
            k: m as i32,
            eps,
        };
        let mut p = nargs.ptrs();
        // SAFETY: same as above.
        unsafe {
            hip.launch_kernel(
                &k_nrm.func,
                [1, 1, 1],
                [256, 1, 1],
                shared_nrm,
                Some(&stream),
                &mut p,
            )
            .expect("baseline nrm launch");
        }
        hip.stream_synchronize(&stream).expect("baseline sync");
        let y_base = dtoh_f32(&hip, &stream, &d_y, m);
        let xr_base = dtoh_f32(&hip, &stream, &d_xr_base, m);

        // Fold: single launch (ctr zeroed once; the kernel resets it).
        hip.memcpy_htod(&d_y, y0_bytes).expect("htod y0");
        hip.memcpy_htod(&d_ctr, &0u32.to_le_bytes()).expect("htod ctr");
        let mut fargs = FoldArgs {
            a: d_a.as_ptr() as u64,
            x: d_x.as_ptr() as u64,
            y: d_y.as_ptr() as u64,
            m: m as i32,
            k: k as i32,
            nw: d_nw.as_ptr() as u64,
            s1: d_s1.as_ptr() as u64,
            s2: d_s2.as_ptr() as u64,
            xr: d_xr_fold.as_ptr() as u64,
            eps,
            ctr: d_ctr.as_ptr() as u64,
        };
        let mut p = fargs.ptrs();
        // SAFETY: same as above.
        unsafe {
            hip.launch_kernel(&k_fold.func, grid_fold, [32, 1, 1], 0, Some(&stream), &mut p)
                .expect("fold launch");
        }
        hip.stream_synchronize(&stream).expect("fold sync");
        let y_fold = dtoh_f32(&hip, &stream, &d_y, m);
        let xr_fold = dtoh_f32(&hip, &stream, &d_xr_fold, m);
        let mut ctr_raw = [0u8; 4];
        hip.memcpy_dtoh(&mut ctr_raw, &d_ctr).expect("dtoh ctr");
        let ctr_val = u32::from_le_bytes(ctr_raw);

        let report = |tag: &str, a: &[f32], b: &[f32]| {
            let mut n = 0usize;
            let mut first: Option<(usize, f32, f32)> = None;
            let mut max_rel = 0f64;
            for (r, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
                if va.to_bits() != vb.to_bits() {
                    n += 1;
                    if first.is_none() {
                        first = Some((r, va, vb));
                    }
                    let rel = ((va - vb).abs() as f64) / (va.abs() as f64).max(1e-30);
                    if rel > max_rel {
                        max_rel = rel;
                    }
                }
            }
            if n == 0 {
                println!("[rn-ab] M={m} K={k}: {tag} bit-exact");
            } else {
                let (r, va, vb) = first.unwrap();
                println!(
                    "[rn-ab] M={m} K={k}: {tag} DIFF n={n} first idx {r} (baseline={va:.6} fold={vb:.6}) max|rel|={max_rel:.2e}"
                );
            }
        };
        report("y", &y_base, &y_fold);
        report("x_rot", &xr_base, &xr_fold);
        println!("[rn-ab] M={m} K={k}: ctr after fold = {ctr_val} (expect 0)");

        // ——— timing: NO re-upload; y accumulates identically in both arms ———
        // Baseline pair: 2 launches per iter.
        for _ in 0..3 {
            let mut p = gargs.ptrs();
            // SAFETY: buffers still live.
            unsafe {
                hip.launch_kernel(&k_res.func, grid_res, [32, 1, 1], 0, Some(&stream), &mut p)
                    .expect("warmup res");
            }
            let mut p = nargs.ptrs();
            // SAFETY: buffers still live.
            unsafe {
                hip.launch_kernel(
                    &k_nrm.func,
                    [1, 1, 1],
                    [256, 1, 1],
                    shared_nrm,
                    Some(&stream),
                    &mut p,
                )
                .expect("warmup nrm");
            }
        }
        hip.stream_synchronize(&stream).expect("warmup sync");
        let start = hip.event_create().expect("event_create");
        let stop = hip.event_create().expect("event_create");
        hip.event_record(&start, Some(&stream)).expect("event_record");
        for _ in 0..iters {
            let mut p = gargs.ptrs();
            // SAFETY: buffers still live.
            unsafe {
                hip.launch_kernel(&k_res.func, grid_res, [32, 1, 1], 0, Some(&stream), &mut p)
                    .expect("timed res");
            }
            let mut p = nargs.ptrs();
            // SAFETY: buffers still live.
            unsafe {
                hip.launch_kernel(
                    &k_nrm.func,
                    [1, 1, 1],
                    [256, 1, 1],
                    shared_nrm,
                    Some(&stream),
                    &mut p,
                )
                .expect("timed nrm");
            }
        }
        hip.event_record(&stop, Some(&stream)).expect("event_record");
        hip.event_synchronize(&stop).expect("event_sync");
        let ms_base = hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64;
        hip.event_destroy(start).unwrap();
        hip.event_destroy(stop).unwrap();
        let us_base = ms_base * 1e3 / iters as f64;

        // Fold: 1 launch per iter (ctr self-resets; no re-upload).
        for _ in 0..3 {
            let mut p = fargs.ptrs();
            // SAFETY: buffers still live.
            unsafe {
                hip.launch_kernel(&k_fold.func, grid_fold, [32, 1, 1], 0, Some(&stream), &mut p)
                    .expect("warmup fold");
            }
        }
        hip.stream_synchronize(&stream).expect("warmup sync");
        let start = hip.event_create().expect("event_create");
        let stop = hip.event_create().expect("event_create");
        hip.event_record(&start, Some(&stream)).expect("event_record");
        for _ in 0..iters {
            let mut p = fargs.ptrs();
            // SAFETY: buffers still live.
            unsafe {
                hip.launch_kernel(&k_fold.func, grid_fold, [32, 1, 1], 0, Some(&stream), &mut p)
                    .expect("timed fold");
            }
        }
        hip.event_record(&stop, Some(&stream)).expect("event_record");
        hip.event_synchronize(&stop).expect("event_sync");
        let ms_fold = hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64;
        hip.event_destroy(start).unwrap();
        hip.event_destroy(stop).unwrap();
        let us_fold = ms_fold * 1e3 / iters as f64;

        // ——— component timings (diag): single launches, same no-re-upload deal ———
        let time_emit = |emit: &mut dyn FnMut()| -> f64 {
            for _ in 0..3 {
                emit();
            }
            hip.stream_synchronize(&stream).expect("warmup sync");
            let start = hip.event_create().expect("event_create");
            let stop = hip.event_create().expect("event_create");
            hip.event_record(&start, Some(&stream)).expect("event_record");
            for _ in 0..iters {
                emit();
            }
            hip.event_record(&stop, Some(&stream)).expect("event_record");
            hip.event_synchronize(&stop).expect("event_sync");
            let ms = hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64;
            hip.event_destroy(start).unwrap();
            hip.event_destroy(stop).unwrap();
            ms * 1e3 / iters as f64
        };
        let us_res_only = time_emit(&mut || {
            let mut p = gargs.ptrs();
            // SAFETY: buffers still live.
            unsafe {
                hip.launch_kernel(&k_res.func, grid_res, [32, 1, 1], 0, Some(&stream), &mut p)
                    .expect("timed res-only");
            }
        });
        let us_nrm_only = time_emit(&mut || {
            let mut p = nargs.ptrs();
            // SAFETY: buffers still live.
            unsafe {
                hip.launch_kernel(
                    &k_nrm.func,
                    [1, 1, 1],
                    [256, 1, 1],
                    shared_nrm,
                    Some(&stream),
                    &mut p,
                )
                .expect("timed nrm-only");
            }
        });

        println!("{:<16} {:>10} {:>10}", "arm", "us/iter", "delta");
        println!("{:<16} {:>10.2} {:>10}", "res-only", us_res_only, "-");
        println!("{:<16} {:>10.2} {:>10}", "nrm-only", us_nrm_only, "-");
        println!("{:<16} {:>10.2} {:>10}", "baseline-pair", us_base, "-");
        println!(
            "{:<16} {:>10.2} {:>10.2}",
            "fold",
            us_fold,
            us_fold - us_base
        );
    }
}
