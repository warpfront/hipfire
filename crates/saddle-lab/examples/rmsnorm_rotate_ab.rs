// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `rmsnorm_rotate_ab`: A/B harness for the fused RMSNorm + FWHT rotation.
//!
//! Two arms, both hipcc with the production flags (no `-D` defines — the
//! default route in `rdna-compute/src/gemv.rs` compiles
//! `kernels/src/fused_rmsnorm_mq_rotate.hip` plain):
//!   - `baseline`: symbol `fused_rmsnorm_mq_rotate`, grid [1,1,1].
//!   - `wg`: symbol `fused_rmsnorm_mq_rotate_wg`, grid [K/256,1,1].
//! Both launch block [256,1,1] with the default-route shared memory
//! ((K+256)*4 bytes). Every shape runs the same seeded inputs, x_rot is
//! bit-compared (`to_bits` equality, all K elements), then both arms are
//! event-timed.
//!
//! Run (device 0, GPU lock held by the caller):
//! ```sh
//! flock -w 300 /tmp/hipfire-gpu.lock env HIP_VISIBLE_DEVICES=0 \
//!   ./target/release/examples/rmsnorm_rotate_ab
//! flock -w 300 /tmp/hipfire-gpu.lock env HIP_VISIBLE_DEVICES=0 \
//!   ./target/release/examples/rmsnorm_rotate_ab --shapes=5120 --flush=0,16,64,160
//! ```
//!
//! `--flush=MB[,MB...]` (default 0): before each timed rmsnorm launch,
//! stream a `cache_flush_stream` kernel over an MB-sized device buffer to
//! evict L2/MALL; only the rmsnorm launch itself is event-timed. This
//! tests cold-cache behavior (the tape shows this kernel at 8.5 us after
//! a 59 MB stream but 28 us after a 142 MB stream).
//!
use hip_bridge::{DeviceBuffer, Function, HipRuntime, Module};

use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

/// Default K shapes (Qwen3.8-27B: hidden 5120, attn 2048, intermediate 17408).
const DEFAULT_SHAPES: [usize; 3] = [5120, 2048, 17408];

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
    /// Uniform f32 in [0, 1).
    fn next_unit(&mut self) -> f32 {
        const INV: f32 = 1.0 / 9_007_199_254_740_992.0; // 2^-53
        ((self.next_u64() >> 11) as f64 * INV as f64) as f32
    }
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

/// Explicit args of `fused_rmsnorm_mq_rotate` / `fused_rmsnorm_mq_rotate_wg`,
/// in kernel signature order
/// (x@0, weight@8, signs1@16, signs2@24, x_rot@32, K@40, eps@44).
/// Passed as a HIP pointer vec.
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

/// Explicit args of `cache_flush_stream`, in kernel signature order
/// (buf@0, n_vec@8, sink@16). Passed as a HIP pointer vec.
struct FlushArgs {
    buf: u64,
    n_vec: u32,
    sink: u64,
}

impl FlushArgs {
    fn ptrs(&mut self) -> Vec<*mut c_void> {
        vec![
            (&mut self.buf as *mut u64).cast(),
            (&mut self.n_vec as *mut u32).cast(),
            (&mut self.sink as *mut u64).cast(),
        ]
    }
}

/// One benchmark arm: the loaded bytes plus its resolved entry point.
/// `module` is never read after load — it is kept alive because unloading
/// would invalidate `func`.
struct Arm {
    name: String,
    #[allow(dead_code)]
    md5: String,
    #[allow(dead_code)]
    module: Module,
    func: Function,
    grid_x: u32,
}

/// Build seeded inputs: `x` (K f32 in [-2, 2]), `weight` (K f32 in
/// [0.5, 1.5]), `signs1`/`signs2` (256 f32 each in {-1, +1}).
fn build_inputs(k: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let x: Vec<f32> = (0..k).map(|_| rng.next_unit() * 4.0 - 2.0).collect();
    let w: Vec<f32> = (0..k).map(|_| 0.5 + rng.next_unit()).collect();
    let s1: Vec<f32> = (0..256)
        .map(|_| if rng.next_unit() < 0.5 { -1.0 } else { 1.0 })
        .collect();
    let s2: Vec<f32> = (0..256)
        .map(|_| if rng.next_unit() < 0.5 { -1.0 } else { 1.0 })
        .collect();
    (x, w, s1, s2)
}

fn f32_slice_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn parse_args() -> (Vec<String>, usize, u64, Vec<usize>, Vec<usize>) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let hipcc_extra: Vec<String> = args
        .iter()
        .filter_map(|a| a.strip_prefix("--hipcc-flags="))
        .flat_map(|s| s.split_whitespace().map(|f| f.to_string()))
        .collect();
    let iters: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("--iters=").and_then(|v| v.parse().ok()))
        .unwrap_or(200);
    assert!(iters >= 1, "--iters must be >= 1");
    let seed: u64 = args
        .iter()
        .find_map(|a| a.strip_prefix("--seed=").and_then(|v| v.parse().ok()))
        .unwrap_or(1);
    let shapes: Vec<usize> = match args.iter().find_map(|a| a.strip_prefix("--shapes=")) {
        Some(s) => s
            .split(',')
            .map(|piece| {
                let k: usize = piece
                    .trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("--shapes entry {piece:?} must be K (e.g. 2048)"));
                assert!(k % 256 == 0, "K={k} must be a multiple of 256");
                k
            })
            .collect(),
        None => DEFAULT_SHAPES.to_vec(),
    };
    let flushes: Vec<usize> = match args.iter().find_map(|a| a.strip_prefix("--flush=")) {
        Some(s) => s
            .split(',')
            .map(|piece| {
                piece
                    .trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("--flush entry {piece:?} must be MB (e.g. 16)"))
            })
            .collect(),
        None => vec![0],
    };
    (hipcc_extra, iters, seed, shapes, flushes)
}

/// hipcc one kernel file with the production flags; return the hsaco bytes.
fn hipcc_build(hipcc_extra: &[String], arch: &str, src_name: &str) -> Vec<u8> {
    let ksrc =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../kernels/src/{src_name}"));
    let tmp = std::env::temp_dir().join("rmsnorm_rotate_ab");
    std::fs::create_dir_all(&tmp).unwrap();
    let stem = src_name.strip_suffix(".hip").unwrap_or(src_name);
    let hsaco_path = tmp.join(format!("{stem}.hsaco"));
    println!("[rmsnorm-ab] hipcc {} ...", ksrc.display());
    let t_cc = Instant::now();
    let st = Command::new("hipcc")
        .args([
            "--genco",
            &format!("--offload-arch={arch}"),
            "-O3",
            "--no-offload-compress",
        ])
        .args(hipcc_extra)
        .arg("-o")
        .arg(hsaco_path.to_str().unwrap())
        .arg(ksrc.to_str().unwrap())
        .status()
        .expect("hipcc spawn");
    assert!(st.success(), "hipcc failed for {src_name}");
    println!("[rmsnorm-ab] hipcc {src_name} {:.2}s", t_cc.elapsed().as_secs_f32());
    std::fs::read(&hsaco_path).expect("read hsaco")
}

fn main() {
    let (hipcc_extra, iters, seed, shapes, flushes) = parse_args();

    let hip = HipRuntime::load().expect("HipRuntime::load");
    hip.set_device(0).expect("set_device(0)");
    let arch = hip.get_arch(0).unwrap_or_else(|_| "gfx1201".to_string());
    println!("[rmsnorm-ab] hip arch={arch} iters={iters} seed={seed} flushes={flushes:?}");
    let stream = hip.stream_create().expect("stream_create");

    // ——— hipcc both arms with the production flags (no extra defines) ———
    let baseline_bytes = hipcc_build(&hipcc_extra, &arch, "fused_rmsnorm_mq_rotate.hip");
    let wg_bytes = hipcc_build(&hipcc_extra, &arch, "fused_rmsnorm_mq_rotate_wg.hip");
    println!(
        "[rmsnorm-ab] baseline hsaco md5={} ({}B)",
        md5hex(&baseline_bytes),
        baseline_bytes.len()
    );
    println!(
        "[rmsnorm-ab] wg hsaco md5={} ({}B)",
        md5hex(&wg_bytes),
        wg_bytes.len()
    );
    let flush_bytes = hipcc_build(&hipcc_extra, &arch, "cache_flush_stream.hip");
    println!(
        "[rmsnorm-ab] flush hsaco md5={} ({}B)",
        md5hex(&flush_bytes),
        flush_bytes.len()
    );

    // ——— load both arms through the same HIP dispatcher ———
    let arm_sources: Vec<(String, Vec<u8>, String, u32)> = vec![
        (
            "baseline".to_string(),
            baseline_bytes,
            "fused_rmsnorm_mq_rotate".to_string(),
            1,
        ),
        (
            "wg".to_string(),
            wg_bytes,
            "fused_rmsnorm_mq_rotate_wg".to_string(),
            0, // grid_x filled per shape (K/256)
        ),
    ];
    let mut arms: Vec<Arm> = Vec::new();
    for (name, bytes, symbol, grid_x) in &arm_sources {
        let md5 = md5hex(bytes);
        println!("[rmsnorm-ab] arm {name}: md5={md5} ({}B) symbol={symbol}", bytes.len());
        let module = hip.module_load_data(bytes).expect("module_load_data");
        let func = hip
            .module_get_function(&module, symbol)
            .unwrap_or_else(|e| panic!("module_get_function {symbol}: {e:?}"));
        arms.push(Arm {
            name: name.clone(),
            md5,
            module,
            func,
            grid_x: *grid_x,
        });
    }
    // ——— load the cache-flush kernel through the same HIP dispatcher ———
    println!(
        "[rmsnorm-ab] flush kernel: md5={} ({}B) symbol=cache_flush_stream",
        md5hex(&flush_bytes),
        flush_bytes.len()
    );
    let _flush_module = hip.module_load_data(&flush_bytes).expect("flush module_load_data");
    let flush_func = hip
        .module_get_function(&_flush_module, "cache_flush_stream")
        .expect("cache_flush_stream");

    let eps: f32 = 1e-6;

    for k in &shapes {
        let k = *k;
        let groups = k / 256;
        // Default-route shared memory: ((K+256)*4) bytes. The kernel only
        // uses reduce[256] (1024 B); the rest is a historical reservation.
        // At K=17408 the default (17408+256)*4 = 70656 B exceeds the 64 KiB
        // per-WG LDS limit, so the production default route is unlaunchable
        // there (the baseline arm fails identically). Clamp to the tight
        // 1024 B both arms equally and say so.
        let default_shared = ((k + 256) * 4) as u32;
        let (shared_mem, shared_note) = if default_shared > 65536 {
            (256 * 4u32, " [note: default (K+256)*4 = 70656B > 64K LDS; both arms use tight 1024B]")
        } else {
            (default_shared, "")
        };
        println!("-- shared_mem={shared_mem}B{shared_note}");

        // Build + upload inputs once per shape.
        let (x, w, s1, s2) = build_inputs(k, seed);
        let d_x = hip.malloc(k * 4).expect("malloc x");
        hip.memcpy_htod(&d_x, f32_slice_bytes(&x)).expect("htod x");
        let d_w = hip.malloc(k * 4).expect("malloc weight");
        hip.memcpy_htod(&d_w, f32_slice_bytes(&w)).expect("htod weight");
        let d_s1 = hip.malloc(256 * 4).expect("malloc signs1");
        hip.memcpy_htod(&d_s1, f32_slice_bytes(&s1)).expect("htod signs1");
        let d_s2 = hip.malloc(256 * 4).expect("malloc signs2");
        hip.memcpy_htod(&d_s2, f32_slice_bytes(&s2)).expect("htod signs2");

        struct Row {
            name: String,
            flush_mb: usize,
            us: f64,
            exact: String,
        }
        let mut rows: Vec<Row> = Vec::new();

        // Flush buffers: one device streaming buffer per nonzero --flush
        // size, filled once per shape with xorshift bytes. The flush kernel
        // streams over the whole buffer before each timed rmsnorm launch to
        // evict L2/MALL; only the rmsnorm launch itself is event-timed.
        struct FlushBuf {
            mb: usize,
            n_vec: u32,
            n_blocks: u32,
            d_buf: DeviceBuffer,
            d_sink: DeviceBuffer,
        }
        let mut flush_bufs: Vec<FlushBuf> = Vec::new();
        {
            let mut frng = Rng::new(seed ^ 0x51ab_3f7d_9e2c_4081);
            for &mb in &flushes {
                if mb == 0 {
                    continue;
                }
                let bytes = mb * 1024 * 1024;
                assert!(bytes % 4096 == 0, "flush {mb}MB not a multiple of 4KiB");
                let n_vec = (bytes / 16) as u32;
                let n_blocks = n_vec / 256;
                let mut host = vec![0u8; bytes];
                for chunk in host.chunks_mut(8) {
                    let v = frng.next_u64().to_le_bytes();
                    let n = chunk.len().min(8);
                    chunk[..n].copy_from_slice(&v[..n]);
                }
                let d_buf = hip.malloc(bytes).expect("malloc flush");
                hip.memcpy_htod(&d_buf, &host).expect("htod flush");
                // One float per block (thread 0 of each block writes its slot).
                let d_sink = hip.malloc(n_blocks as usize * 4).expect("malloc sink");
                flush_bufs.push(FlushBuf {
                    mb,
                    n_vec,
                    n_blocks,
                    d_buf,
                    d_sink,
                });
            }
        }

        let mut ref_y: Vec<f32> = Vec::new();
        for (ai, arm) in arms.iter().enumerate() {
            let grid_x = if arm.grid_x == 0 { groups as u32 } else { arm.grid_x };
            let d_xr = hip.malloc(k * 4).expect("malloc x_rot");
            let mut nargs = NormArgs {
                x: d_x.as_ptr() as u64,
                w: d_w.as_ptr() as u64,
                s1: d_s1.as_ptr() as u64,
                s2: d_s2.as_ptr() as u64,
                xr: d_xr.as_ptr() as u64,
                k: k as i32,
                eps,
            };
            for _ in 0..3 {
                let mut p = nargs.ptrs();
                // SAFETY: module loaded, args are live device pointers/values.
                unsafe {
                    hip.launch_kernel(
                        &arm.func,
                        [grid_x, 1, 1],
                        [256, 1, 1],
                        shared_mem,
                        Some(&stream),
                        &mut p,
                    )
                    .expect("warmup launch");
                }
            }
            hip.stream_synchronize(&stream).expect("warmup sync");

            // Exactness snapshot (unflushed, as before): the flush kernel
            // touches only its own buffer+sink, so flushed runs are equally
            // comparable, but one snapshot per arm keeps the table readable.
            let mut raw = vec![0u8; k * 4];
            hip.memcpy_dtoh(&mut raw, &d_xr).expect("dtoh x_rot");
            // SAFETY: kernel wrote K f32 elements; length checked above.
            let y: Vec<f32> = unsafe {
                std::slice::from_raw_parts(raw.as_ptr() as *const f32, k).to_vec()
            };
            let exact = if ai == 0 {
                ref_y = y;
                "ref".to_string()
            } else {
                let mut nmismatch = 0usize;
                let mut first: Option<(usize, f32, f32)> = None;
                for (idx, (&a, &b)) in ref_y.iter().zip(y.iter()).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        nmismatch += 1;
                        if first.is_none() {
                            first = Some((idx, a, b));
                        }
                    }
                }
                if nmismatch == 0 {
                    "bit-exact".to_string()
                } else {
                    let (idx, a, b) = first.unwrap();
                    format!(
                        "DIFF n={nmismatch} first idx {idx} (baseline={a:.6} wg={b:.6} bits {:08x}/{:08x})",
                        a.to_bits(),
                        b.to_bits()
                    )
                }
            };

            for &mb in &flushes {
                let fb = flush_bufs.iter().find(|f| f.mb == mb);
                // One flush-kernel warmup so its code is hot and only the
                // data it streams is cold.
                if let Some(f) = fb {
                    let mut fargs = FlushArgs {
                        buf: f.d_buf.as_ptr() as u64,
                        n_vec: f.n_vec,
                        sink: f.d_sink.as_ptr() as u64,
                    };
                    let mut fp = fargs.ptrs();
                    // SAFETY: module loaded, args are live device pointers/values.
                    unsafe {
                        hip.launch_kernel(
                            &flush_func,
                            [f.n_blocks, 1, 1],
                            [256, 1, 1],
                            64,
                            Some(&stream),
                            &mut fp,
                        )
                        .expect("flush warmup launch");
                    }
                    hip.stream_synchronize(&stream).expect("flush warmup sync");
                }
                let start = hip.event_create().expect("event_create");
                let stop = hip.event_create().expect("event_create");
                // Per iteration: flush (evict L2/MALL), then time ONLY the
                // rmsnorm launch.
                let mut ms_sum = 0f64;
                for _ in 0..iters {
                    if let Some(f) = fb {
                        let mut fargs = FlushArgs {
                            buf: f.d_buf.as_ptr() as u64,
                            n_vec: f.n_vec,
                            sink: f.d_sink.as_ptr() as u64,
                        };
                        let mut fp = fargs.ptrs();
                        // SAFETY: same as above; buffers still live.
                        unsafe {
                            hip.launch_kernel(
                                &flush_func,
                                [f.n_blocks, 1, 1],
                                [256, 1, 1],
                                64,
                                Some(&stream),
                                &mut fp,
                            )
                            .expect("flush launch");
                        }
                    }
                    hip.event_record(&start, Some(&stream))
                        .expect("event_record");
                    {
                        let mut p = nargs.ptrs();
                        // SAFETY: same as above; buffers still live.
                        unsafe {
                            hip.launch_kernel(
                                &arm.func,
                                [grid_x, 1, 1],
                                [256, 1, 1],
                                shared_mem,
                                Some(&stream),
                                &mut p,
                            )
                            .expect("timed launch");
                        }
                    }
                    hip.event_record(&stop, Some(&stream))
                        .expect("event_record");
                    hip.event_synchronize(&stop).expect("event_sync");
                    ms_sum += hip
                        .event_elapsed_ms(&start, &stop)
                        .expect("event_elapsed_ms") as f64;
                }
                hip.event_destroy(start).unwrap();
                hip.event_destroy(stop).unwrap();
                let us = ms_sum * 1e3 / iters as f64;
                rows.push(Row {
                    name: arm.name.clone(),
                    flush_mb: mb,
                    us,
                    exact: exact.clone(),
                });
            }
        }

        println!(
            "{:<10} {:>8} {:<24} {:>10} {:>8}  {}",
            "K", "flushMB", "arm", "us/launch", "speedup", "exactness"
        );
        for r in &rows {
            let base_us = rows
                .iter()
                .find(|b| b.flush_mb == r.flush_mb && b.name == "baseline")
                .map(|b| b.us)
                .unwrap_or(f64::NAN);
            println!(
                "{:<10} {:>8} {:<24} {:>10.3} {:>8.3}  {}",
                k,
                r.flush_mb,
                r.name,
                r.us,
                base_us / r.us,
                r.exact
            );
        }
    }
}
