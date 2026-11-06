//! MQ8G256 decode / verify lab (M2 packet, Raon attention-class projections, weight layout B).
//!
//! usage: mq8_decode_lab [identity|timing|all] [--rounds N] [--launches L] [--reps R] [--groups a,b]
//!                       [--ns 1,2,4,8,16] [--expect-bdf B] [--sysfs DIR]
//!
//! Lab feature, GPU. `GPU_MAX_HW_QUEUES=1` is expected for timing (the card scripts set it).
//!
//! # identity (exit 2 on any failure; timing still runs in `all`)
//! For every Raon attention (M, K) pair with K % 256 == 0 (RDA q/k/v/g `[5120, 2560]`, RDA o / MLA g / o,
//! `[2560, 5120]`, MLA q_a / kv_a `[576, 2560]`, MLA kv_b `[10240, 512]`; q_b K = 576 stays Q8F16 and is
//! skipped) plus tail / odd-group-count / one-group shapes, weights through the REAL `mq8g256_repack_b`
//! (realistic quantizer-style and adversarial: -128 / 127 codes, zero / denormal / max-finite / negative f16
//! scales), X planes of two kinds (realistic: random rows through the GPU batched producer
//! `mq8_quantize_x_into`; adversarial: handcrafted codes and wide-ranging scales through
//! `mq8_acts_from_planes`):
//! * `gemv_mq8g256_b` (n = 1, one launch per X row) must equal the CPU reference of its documented
//!   association (`cpu_row_b`) bit for bit, on every row for small M and on 256+ sampled rows otherwise;
//! * `gemv_mq8g256_rows_exact_b` for n = 2..=16, 17 and 32 (groups of 16) must equal the per-row singleton
//!   launches bit for bit for every weight row, and leave the guard words after `y` untouched.
//!
//! # timing
//! Production Q8_0 path (`Gpu::gemv_q8_0` at n = 1, `Gpu::gemv_q8_0_rows_exact` at n > 1, F32 X, no x-prep) vs
//! the MQ8G256 layout-B path, per GROUP of projections that share one X (one producer launch feeds every
//! member): kernel-only (X planes ready) and chain (batched producer + the group's GEMVs). Each arm is
//! captured into a HIP graph of `--launches` iterations (the weight ring rotates over >= 256 MiB of copies
//! per shape), replayed `--reps` times per timed batch; arms are interleaved Q8 K C P P C K Q per round and
//! the per-arm medians reported. Graph replay removes the host launch cost (production decodes in graphs).
//!
//! Decision lines (grep `DECISION`): identity, and worst MQ8 / Q8_0 ratio of the kernel-only and chain arms
//! over all groups and n; `ROW` lines carry the numbers per group and n, with the sclk / load snapshot.
use half::f16;
use rdna_compute::gemv_mq8_rows::mq8g256_b_fused_route;
use rdna_compute::{mq8g256_b_bytes, mq8g256_repack_b, Mq8Activations};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const SENTINEL: i32 = 0xff;
/// The projection groups the model really runs: RDA q/k/v/g (one X), MLA q_a/kv_a/g (one X), o (own X),
/// kv_b (own X). The other groups are single-projection views for the table.
const REAL_GROUPS: &[&str] = &[
    "rda_qkvg_x4",
    "mla_qakvag_x3",
    "single_2560x5120",
    "single_10240x512",
];
const GUARD_WORDS: usize = 64;
const RING_MIN_BYTES: usize = 256 << 20;
const RING_MAX: usize = 256;

// ---------------------------------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }
    fn unit(&mut self) -> f32 {
        ((self.next() & 0xff_ffff) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

fn wrap(ptr: *mut std::ffi::c_void, bytes: usize, shape: Vec<usize>, dtype: DType) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) },
        shape,
        dtype,
    }
}

/// F32-typed view over `bytes` of device memory (byte planes live in F32-typed tensors).
fn bytes_tensor(buf: &hip_bridge::DeviceBuffer, bytes: usize) -> GpuTensor {
    wrap(buf.as_ptr(), bytes, vec![bytes / 4], DType::F32)
}

fn f32_tensor(buf: &hip_bridge::DeviceBuffer, elems: usize) -> GpuTensor {
    wrap(buf.as_ptr(), elems * 4, vec![elems], DType::F32)
}

/// Quantizer-style MQ8G256 weights (layout A bytes): f16 scale (truncating) + codes in [-128, 127].
fn gen_weights_realistic(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(m * (k / 256) * 258);
    let mut grp = vec![0f32; 256];
    for _ in 0..m * (k / 256) {
        let mut amax = 0f32;
        for v in grp.iter_mut() {
            *v = rng.unit() * 0.05;
            amax = amax.max(v.abs());
        }
        let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        // truncating f32 -> f16: round the f16 value toward zero.
        let mut h = f16::from_f32(sc);
        if h.to_f32() > sc {
            h = f16::from_bits(h.to_bits() - 1);
        }
        let scf = h.to_f32().max(f32::MIN_POSITIVE);
        out.extend_from_slice(&h.to_bits().to_le_bytes());
        for &v in grp.iter() {
            out.push(((v / scf).round_ties_even().clamp(-128.0, 127.0) as i8) as u8);
        }
    }
    out
}

/// Adversarial weights (layout A bytes): codes biased to the extremes (-128, 127, 0), f16 scales over the
/// whole finite range including denormals, zero and negative values.
fn gen_weights_adversarial(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(m * (k / 256) * 258);
    for gi in 0..m * (k / 256) {
        let bits: u16 = match rng.next() % 8 {
            0 => (rng.next() & 0x03ff) as u16,          // +denormal / zero
            1 => 0x8000 | (rng.next() & 0x03ff) as u16, // -denormal / -zero
            2 => 0x3c00,                                // 1.0
            3 => 0x7bff,                                // max finite
            _ => {
                let exp = 1 + rng.next() % 30; // finite normal
                (((rng.next() & 1) << 15) | (exp << 10) | (rng.next() & 0x03ff)) as u16
            }
        };
        out.extend_from_slice(&bits.to_le_bytes());
        let style = gi % 5;
        for j in 0..256usize {
            let c: i8 = match style {
                0 => -128,
                1 => {
                    if j % 2 == 0 {
                        -128
                    } else {
                        127
                    }
                }
                2 => (rng.next() & 0xff) as u8 as i8,
                3 => match rng.next() % 4 {
                    0 => -128,
                    1 => 127,
                    2 => 0,
                    _ => (rng.next() & 0xff) as u8 as i8,
                },
                _ => 127,
            };
            out.push(c as u8);
        }
    }
    out
}

/// Adversarial X planes `[rows][k]` codes + `[rows][k/256]` scales (row-major host form).
fn gen_planes_adversarial(rows: usize, k: usize, rng: &mut Rng) -> (Vec<u8>, Vec<f32>) {
    let gpr = k / 256;
    let mut xq = Vec::with_capacity(rows * k);
    let mut xs = Vec::with_capacity(rows * gpr);
    for r in 0..rows {
        for g in 0..gpr {
            let scale = match (r + g) % 4 {
                0 => 1.0f32,
                1 => 0.003 + (rng.unit().abs() * 0.01),
                2 => 0.5 + rng.unit().abs() * 8.0,
                _ => 1e-3,
            };
            xs.push(scale);
            for j in 0..256usize {
                let c: i8 = match (r + g) % 5 {
                    0 => -128,
                    1 => {
                        if j % 2 == 0 {
                            127
                        } else {
                            -128
                        }
                    }
                    2 => (rng.next() & 0xff) as u8 as i8,
                    3 => 0,
                    _ => match rng.next() % 3 {
                        0 => -128,
                        1 => 127,
                        _ => (rng.next() & 0xff) as u8 as i8,
                    },
                };
                xq.push(c as u8);
            }
        }
    }
    (xq, xs)
}

/// CPU reference of `gemv_mq8g256_b_r<R>_u<U>` for one weight row (layout A bytes of the row: `k/256`
/// groups of f16 scale + int8[256]) against one X row (`xq` codes, `xs[g]` group scales): the documented
/// association. Lane `l = 16 h + s` owns bytes `[16 s, 16 s + 16)` of group `2 p + h` of pair `p`; one fma
/// chain per lane over the pairs (a lane past the last group contributes nothing); 16-lane butterfly per
/// half; `T_0 + T_1`.
fn cpu_row_b(w_row: &[u8], k: usize, xq: &[u8], xs: &[f32]) -> f32 {
    let gpr = k / 256;
    let npairs = gpr.div_ceil(2);
    let mut lane = [0f32; 32];
    for (l, out) in lane.iter_mut().enumerate() {
        let (h, s) = (l / 16, l % 16);
        let mut acc = 0f32;
        for p in 0..npairs {
            let g = 2 * p + h;
            if g >= gpr {
                continue;
            }
            let gp = &w_row[g * 258..g * 258 + 258];
            let ws = f16::from_bits(u16::from_le_bytes([gp[0], gp[1]])).to_f32();
            let mut dot = 0i32;
            for j in 0..16 {
                dot += (gp[2 + 16 * s + j] as i8 as i32) * (xq[g * 256 + 16 * s + j] as i8 as i32);
            }
            acc = (ws * xs[g]).mul_add(dot as f32, acc);
        }
        *out = acc;
    }
    for off in [1usize, 2, 4, 8] {
        let prev = lane;
        for h in 0..2 {
            for s in 0..16 {
                lane[16 * h + s] = prev[16 * h + s] + prev[16 * h + (s ^ off)];
            }
        }
    }
    lane[0] + lane[16]
}

fn quantize_q8_0(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(w.len() / 32 * 34);
    for blk in w.chunks_exact(32) {
        let amax = blk.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        out.extend_from_slice(&f16::from_f32(sc).to_bits().to_le_bytes());
        for &v in blk {
            out.push(((v / sc).round().clamp(-127.0, 127.0) as i8) as u8);
        }
    }
    out
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn upload(gpu: &mut Gpu, bytes: &[u8]) -> hip_bridge::DeviceBuffer {
    let b = gpu.hip.malloc(bytes.len().max(4)).unwrap();
    gpu.hip.memcpy_htod(&b, bytes).unwrap();
    b
}

fn download(gpu: &mut Gpu, buf: &hip_bridge::DeviceBuffer, bytes: usize) -> Vec<u8> {
    gpu.hip.device_synchronize().unwrap();
    let mut v = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut v, buf).unwrap();
    v
}

fn bits(v: &[u8]) -> Vec<u32> {
    v.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Layout-B bytes of layout-A weights, padded to the allocation the launcher validates.
fn repack(a: &[u8], m: usize, k: usize) -> Vec<u8> {
    let b = mq8g256_repack_b(a, m, k);
    assert_eq!(b.len(), mq8g256_b_bytes(m, k));
    b
}

// ---------------------------------------------------------------------------------------------------
// identity
// ---------------------------------------------------------------------------------------------------

const ROW_COUNTS: [usize; 17] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32];
const PLANE_ROWS: usize = 32;

/// (label, M, K): Raon attention-class (K % 256 == 0) pairs, then tail / odd-group stress shapes.
const RAON_SHAPES: &[(&str, usize, usize)] = &[
    ("rda_qkvg/mla_g", 5120, 2560),
    ("rda_o/mla_o", 2560, 5120),
    ("mla_q_a/kv_a", 576, 2560),
    ("mla_kv_b", 10240, 512),
];
const EXTRA_SHAPES: &[(&str, usize, usize)] = &[
    ("one_group_k256", 33, 256),
    ("tail_k768", 130, 768),
    ("k1280_m257", 257, 1280),
    ("k1792_m64", 64, 1792),
    ("k4864_m31", 31, 4864),
];

/// X planes of PLANE_ROWS rows, host form.
struct Planes {
    xq: Vec<u8>,
    xs: Vec<f32>,
}

/// Device activations for the first `n` rows of `planes` (scales transposed to `[g][n]`).
fn upload_acts(
    gpu: &mut Gpu,
    planes: &Planes,
    first: usize,
    n: usize,
    k: usize,
    keep: &mut Vec<hip_bridge::DeviceBuffer>,
) -> Mq8Activations {
    let gpr = k / 256;
    let xq = &planes.xq[first * k..(first + n) * k];
    let mut xs_t = vec![0f32; gpr * n];
    for r in 0..n {
        for g in 0..gpr {
            xs_t[g * n + r] = planes.xs[(first + r) * gpr + g];
        }
    }
    let xs_bytes: Vec<u8> = xs_t.iter().flat_map(|v| v.to_le_bytes()).collect();
    let q_buf = upload(gpu, xq);
    let s_buf = upload(gpu, &xs_bytes);
    let acts = gpu
        .mq8_acts_from_planes(
            &bytes_tensor(&q_buf, n * k),
            &f32_tensor(&s_buf, gpr * n),
            n,
            k,
        )
        .unwrap();
    keep.push(q_buf);
    keep.push(s_buf);
    acts
}

fn identity(gpu: &mut Gpu) -> bool {
    let mut failures = 0usize;
    let mut cpu_checks = 0usize;
    let mut cross_checks = 0usize;
    let mut guard_checks = 0usize;
    let mut fused_checks = 0usize;
    let mut rng = Rng(0x4d51_3847_0000_0001);
    for &(label, m, k) in RAON_SHAPES.iter().chain(EXTRA_SHAPES) {
        let gpr = k / 256;
        for adversarial in [false, true] {
            let a = if adversarial {
                gen_weights_adversarial(m, k, &mut rng)
            } else {
                gen_weights_realistic(m, k, &mut rng)
            };
            let b = repack(&a, m, k);
            let w_buf = upload(gpu, &b);
            let w_t = bytes_tensor(&w_buf, b.len());
            let mut keep: Vec<hip_bridge::DeviceBuffer> = Vec::new();
            // X planes: realistic rows through the GPU producer, adversarial handcrafted.
            let mut x_dev: Option<hip_bridge::DeviceBuffer> = None;
            let planes = if adversarial {
                let (xq, xs) = gen_planes_adversarial(PLANE_ROWS, k, &mut rng);
                Planes { xq, xs }
            } else {
                let mut x: Vec<f32> = (0..PLANE_ROWS * k)
                    .map(|i| {
                        let r = i / k;
                        let amp = [0.02f32, 1.0, 40.0, 0.3][r % 4];
                        let mut v = rng.unit() * amp;
                        if rng.next() % 61 == 0 {
                            v *= 25.0; // outliers
                        }
                        v
                    })
                    .collect();
                // special rows: all zero (scale 1.0, codes 0) and a single spike in an otherwise zero row
                for i in 0..k {
                    x[5 * k + i] = 0.0;
                    x[6 * k + i] = 0.0;
                }
                x[6 * k + 7] = 3.5;
                let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
                let x_buf = upload(gpu, &xb);
                let xq_buf = gpu.hip.malloc(PLANE_ROWS * k).unwrap();
                let xs_buf = gpu.hip.malloc(PLANE_ROWS * gpr * 4).unwrap();
                let acts = gpu
                    .mq8_quantize_x_into(
                        &f32_tensor(&x_buf, PLANE_ROWS * k),
                        &bytes_tensor(&xq_buf, PLANE_ROWS * k),
                        &f32_tensor(&xs_buf, PLANE_ROWS * gpr),
                        PLANE_ROWS,
                        k,
                    )
                    .unwrap();
                assert_eq!(acts.rows(), PLANE_ROWS);
                let xq = download(gpu, &xq_buf, PLANE_ROWS * k);
                let xs_t = bits(&download(gpu, &xs_buf, PLANE_ROWS * gpr * 4));
                // producer scales are [g][n]; host form is [r][g]
                let mut xs = vec![0f32; PLANE_ROWS * gpr];
                for r in 0..PLANE_ROWS {
                    for g in 0..gpr {
                        xs[r * gpr + g] = f32::from_bits(xs_t[g * PLANE_ROWS + r]);
                    }
                }
                x_dev = Some(x_buf);
                Planes { xq, xs }
            };
            // Singleton launches: one 1-row activation set per X row.
            let y1_buf = gpu.hip.malloc((m + GUARD_WORDS) * 4).unwrap();
            let mut single: Vec<Vec<u32>> = Vec::with_capacity(PLANE_ROWS);
            for r in 0..PLANE_ROWS {
                let acts = upload_acts(gpu, &planes, r, 1, k, &mut keep);
                gpu.hip
                    .memset(&y1_buf, SENTINEL, (m + GUARD_WORDS) * 4)
                    .unwrap();
                let y_t = f32_tensor(&y1_buf, m + GUARD_WORDS);
                gpu.gemv_mq8g256_b(&w_t, &acts, &y_t, m).unwrap();
                let y = bits(&download(gpu, &y1_buf, (m + GUARD_WORDS) * 4));
                guard_checks += 1;
                if y[m..].iter().any(|&w| w != 0xffff_ffff) {
                    println!("IDENTITY FAIL {label} adv={adversarial} n=1 row {r}: guard words overwritten");
                    failures += 1;
                }
                single.push(y[..m].to_vec());
            }
            // The fused-producer entry (F32 X straight into the GEMV, n = 1 and 2; other shapes fall back to
            // producer + GEMV) must equal the planes path bit for bit.
            if let Some(xb) = &x_dev {
                for n in [1usize, 2] {
                    for first in (0..PLANE_ROWS).step_by(n) {
                        let x_view = wrap(
                            unsafe { (xb.as_ptr() as *mut u8).add(first * k * 4) }
                                as *mut std::ffi::c_void,
                            n * k * 4,
                            vec![n * k],
                            DType::F32,
                        );
                        gpu.hip
                            .memset(&y1_buf, SENTINEL, (m + GUARD_WORDS) * 4)
                            .unwrap();
                        let yb = gpu.hip.malloc((n * m + GUARD_WORDS) * 4).unwrap();
                        gpu.hip
                            .memset(&yb, SENTINEL, (n * m + GUARD_WORDS) * 4)
                            .unwrap();
                        let y_t = f32_tensor(&yb, n * m + GUARD_WORDS);
                        gpu.gemv_mq8g256_b_x(&w_t, &x_view, n, &y_t, m, k).unwrap();
                        let y = bits(&download(gpu, &yb, (n * m + GUARD_WORDS) * 4));
                        guard_checks += 1;
                        if y[n * m..].iter().any(|&w| w != 0xffff_ffff) {
                            println!("IDENTITY FAIL {label} fused n={n} first={first}: guard words overwritten");
                            failures += 1;
                        }
                        for r in 0..n {
                            for row in 0..m {
                                fused_checks += 1;
                                if y[r * m + row] != single[first + r][row] {
                                    failures += 1;
                                    if failures <= 20 {
                                        println!(
                                            "IDENTITY FAIL {label} fused n={n} xrow {} wrow {row}: fused {:08x} planes {:08x}",
                                            first + r,
                                            y[r * m + row],
                                            single[first + r][row]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // CPU reference of the association: every weight row when small, else a sample.
            let rows: Vec<usize> = if m <= 600 {
                (0..m).collect()
            } else {
                let mut v = vec![0, 1, 2, 3, m - 4, m - 3, m - 2, m - 1];
                let mut rr = Rng(0x5eed ^ m as u64);
                while v.len() < 264 {
                    v.push((rr.next() as usize) % m);
                }
                v
            };
            for r in 0..PLANE_ROWS {
                for &row in &rows {
                    let want = cpu_row_b(
                        &a[row * gpr * 258..(row + 1) * gpr * 258],
                        k,
                        &planes.xq[r * k..(r + 1) * k],
                        &planes.xs[r * gpr..(r + 1) * gpr],
                    );
                    cpu_checks += 1;
                    if single[r][row] != want.to_bits() {
                        failures += 1;
                        if failures <= 20 {
                            println!(
                                "IDENTITY FAIL {label} adv={adversarial} n=1 xrow {r} wrow {row}: gpu {:08x} cpu {:08x}",
                                single[r][row],
                                want.to_bits()
                            );
                        }
                    }
                }
            }
            // rows-exact vs the singletons.
            for &n in &ROW_COUNTS {
                let acts = upload_acts(gpu, &planes, 0, n, k, &mut keep);
                let y_buf = gpu.hip.malloc((n * m + GUARD_WORDS) * 4).unwrap();
                gpu.hip
                    .memset(&y_buf, SENTINEL, (n * m + GUARD_WORDS) * 4)
                    .unwrap();
                let y_t = f32_tensor(&y_buf, n * m + GUARD_WORDS);
                gpu.gemv_mq8g256_rows_exact_b(&w_t, &acts, &y_t, m).unwrap();
                let y = bits(&download(gpu, &y_buf, (n * m + GUARD_WORDS) * 4));
                guard_checks += 1;
                if y[n * m..].iter().any(|&w| w != 0xffff_ffff) {
                    println!(
                        "IDENTITY FAIL {label} adv={adversarial} n={n}: guard words overwritten"
                    );
                    failures += 1;
                }
                for r in 0..n {
                    for row in 0..m {
                        cross_checks += 1;
                        if y[r * m + row] != single[r][row] {
                            failures += 1;
                            if failures <= 20 {
                                println!(
                                    "IDENTITY FAIL {label} adv={adversarial} n={n} xrow {r} wrow {row}: rows_exact {:08x} singleton {:08x}",
                                    y[r * m + row],
                                    single[r][row]
                                );
                            }
                        }
                    }
                }
            }
            // The gather helper `gemv_mq8g256_b_rows` is the same launches.
            gpu.hip.device_synchronize().unwrap();
            drop(keep);
        }
        println!("identity {label:16} M={m:5} K={k:5} done");
    }
    println!(
        "DECISION identity_rows_exact_bit_identical {} (singleton vs CPU association: {cpu_checks} checks; rows-exact vs singleton: {cross_checks} checks; fused-producer entry vs planes path (n = 1, 2): {fused_checks} checks; guard sentinels: {guard_checks}; failures {failures})",
        if failures == 0 { "PASS" } else { "FAIL" }
    );
    failures == 0
}

// ---------------------------------------------------------------------------------------------------
// timing
// ---------------------------------------------------------------------------------------------------

struct Cfg {
    rounds: usize,
    launches: usize,
    reps: usize,
    groups: Option<Vec<String>>,
    ns: Vec<usize>,
    sysfs: String,
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

struct Group {
    name: &'static str,
    /// (M, K) of every projection that consumes the group's one X (all members share K).
    members: &'static [(usize, usize)],
}

const GROUPS: &[Group] = &[
    Group {
        name: "rda_qkvg_x4",
        members: &[(5120, 2560), (5120, 2560), (5120, 2560), (5120, 2560)],
    },
    Group {
        name: "mla_qakvag_x3",
        members: &[(576, 2560), (576, 2560), (5120, 2560)],
    },
    Group {
        name: "single_5120x2560",
        members: &[(5120, 2560)],
    },
    Group {
        name: "single_2560x5120",
        members: &[(2560, 5120)],
    },
    Group {
        name: "single_576x2560",
        members: &[(576, 2560)],
    },
    Group {
        name: "single_10240x512",
        members: &[(10240, 512)],
    },
];

/// Weights of one (M, K) shape: a ring of Q8_0 copies and a ring of layout-B MQ8G256 copies.
struct ShapeRing {
    m: usize,
    k: usize,
    q8: Vec<GpuTensor>,
    mq: Vec<GpuTensor>,
    _keep: Vec<hip_bridge::DeviceBuffer>,
}

fn build_ring(gpu: &mut Gpu, m: usize, k: usize, rng: &mut Rng) -> ShapeRing {
    let a = gen_weights_realistic(m, k, rng);
    let mq_bytes = repack(&a, m, k);
    let q8_bytes = {
        let w: Vec<f32> = (0..m * k).map(|_| rng.unit() * 0.05).collect();
        quantize_q8_0(&w)
    };
    let ring_of = |len: usize| (RING_MIN_BYTES.div_ceil(len)).clamp(4, RING_MAX);
    let (rq, rm) = (ring_of(q8_bytes.len()), ring_of(mq_bytes.len()));
    let mut keep = Vec::new();
    let mut q8 = Vec::new();
    for _ in 0..rq {
        let b = upload(gpu, &q8_bytes);
        q8.push(wrap(b.as_ptr(), q8_bytes.len(), vec![m, k], DType::Q8_0));
        keep.push(b);
    }
    let mut mq = Vec::new();
    for _ in 0..rm {
        let b = upload(gpu, &mq_bytes);
        mq.push(bytes_tensor(&b, mq_bytes.len()));
        keep.push(b);
    }
    ShapeRing {
        m,
        k,
        q8,
        mq,
        _keep: keep,
    }
}

type Arm<'a> = Box<dyn FnMut(&mut Gpu, usize) + 'a>;

struct Captured {
    graph: hip_bridge::Graph,
    exec: hip_bridge::GraphExec,
}

/// Capture `launches` iterations of `arm` (warmed eagerly first) into an instantiated graph.
fn capture(gpu: &mut Gpu, launches: usize, arm: &mut Arm) -> Captured {
    for i in 0..3 {
        arm(gpu, i);
    }
    gpu.hip.device_synchronize().unwrap();
    gpu.begin_stream_capture().expect("begin capture");
    for i in 0..launches {
        arm(gpu, i);
    }
    let graph = gpu.end_stream_capture().expect("end capture");
    let exec = gpu.hip.graph_instantiate(&graph).expect("instantiate");
    Captured { graph, exec }
}

fn release(gpu: &mut Gpu, c: Captured) {
    gpu.hip.device_synchronize().unwrap();
    gpu.hip.graph_exec_destroy(c.exec).unwrap();
    gpu.hip.graph_destroy(c.graph).unwrap();
}

/// us per captured iteration of one timed batch (`reps` replays between two device syncs).
fn replay_us(gpu: &mut Gpu, c: &Captured, cfg: &Cfg) -> f64 {
    gpu.hip.device_synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..cfg.reps {
        gpu.launch_graph(&c.exec).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / (cfg.reps * cfg.launches) as f64
}

struct Ratios {
    kernel: f64,
    chain: f64,
}

fn timing(gpu: &mut Gpu, cfg: &Cfg) {
    let mut rng = Rng(0x7e57_0000_1234_5678);
    // Shapes used by the selected groups.
    let groups: Vec<&Group> = GROUPS
        .iter()
        .filter(|g| {
            cfg.groups
                .as_ref()
                .is_none_or(|l| l.iter().any(|n| n == g.name))
        })
        .collect();
    let mut shapes: Vec<(usize, usize)> = Vec::new();
    for g in &groups {
        for &s in g.members {
            if !shapes.contains(&s) {
                shapes.push(s);
            }
        }
    }
    let rings: Vec<ShapeRing> = shapes
        .iter()
        .map(|&(m, k)| build_ring(gpu, m, k, &mut rng))
        .collect();
    let ring_of = |m: usize, k: usize| rings.iter().find(|r| r.m == m && r.k == k).unwrap();
    println!(
        "\n# timing: graph replay, {} launches per graph x {} replays per batch, {} ABBA rounds (Q K C F P P F C K Q), ring >= {} MiB",
        cfg.launches,
        cfg.reps,
        cfg.rounds,
        RING_MIN_BYTES >> 20
    );
    let mut worst_policy_real = (0.0f64, String::new());
    let mut worst_policy_other = (0.0f64, String::new());
    let mut worst_kernel_single = (0.0f64, String::new());
    let mut worst_chain_single = (0.0f64, String::new());
    let mut worst_kernel_shared = (0.0f64, String::new());
    let mut worst_chain_shared = (0.0f64, String::new());
    let mut all_ratios: Vec<(String, usize, Ratios)> = Vec::new();
    let max_n = *cfg.ns.iter().max().unwrap();
    for g in &groups {
        let k = g.members[0].1;
        let gpr = k / 256;
        let s = g.members.len();
        // X input (F32 [max_n, k]) and the output buffer shared by every arm.
        let x: Vec<f32> = (0..max_n * k).map(|_| rng.unit()).collect();
        let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x_buf = upload(gpu, &xb);
        let max_m = g.members.iter().map(|&(m, _)| m).max().unwrap();
        let y_buf = gpu.hip.malloc(max_n * max_m * 4).unwrap();
        let xq_buf = gpu.hip.malloc(max_n * k).unwrap();
        let xs_buf = gpu.hip.malloc(max_n * gpr * 4).unwrap();
        for &n in &cfg.ns {
            let x_n = f32_tensor(&x_buf, n * k);
            // Kernel-only activations (producer launched once, planes then read only).
            let acts_ready = gpu
                .mq8_quantize_x_into(
                    &x_n,
                    &bytes_tensor(&xq_buf, n * k),
                    &f32_tensor(&xs_buf, n * gpr),
                    n,
                    k,
                )
                .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let members: Vec<&ShapeRing> = g.members.iter().map(|&(m, k)| ring_of(m, k)).collect();
            let mut q8_arm: Arm = Box::new(|gp: &mut Gpu, i: usize| {
                for (j, r) in members.iter().enumerate() {
                    let w = &r.q8[(i * s + j) % r.q8.len()];
                    let y = f32_tensor(&y_buf, n * r.m);
                    if n == 1 {
                        gp.gemv_q8_0(w, &x_n, &y, r.m, r.k).unwrap();
                    } else {
                        gp.gemv_q8_0_rows_exact(w, &x_n, &y, r.m, r.k, n).unwrap();
                    }
                }
            });
            let mut k_arm: Arm = Box::new(|gp: &mut Gpu, i: usize| {
                for (j, r) in members.iter().enumerate() {
                    let w = &r.mq[(i * s + j) % r.mq.len()];
                    let y = f32_tensor(&y_buf, n * r.m);
                    gp.gemv_mq8g256_b_rows(w, &acts_ready, &y, r.m).unwrap();
                }
            });
            let mut c_arm: Arm = Box::new(|gp: &mut Gpu, i: usize| {
                let acts = gp.mq8_quantize_x(&x_n, n, k).unwrap();
                for (j, r) in members.iter().enumerate() {
                    let w = &r.mq[(i * s + j) % r.mq.len()];
                    let y = f32_tensor(&y_buf, n * r.m);
                    gp.gemv_mq8g256_b_rows(w, &acts, &y, r.m).unwrap();
                }
            });
            let mut p_arm: Arm = Box::new(|gp: &mut Gpu, _i: usize| {
                gp.mq8_quantize_x(&x_n, n, k).unwrap();
            });
            // production fused entry: F32 X straight into the GEMV (n <= 2; falls back to producer + GEMV otherwise)
            let mut f_arm: Arm = Box::new(|gp: &mut Gpu, i: usize| {
                for (j, r) in members.iter().enumerate() {
                    let w = &r.mq[(i * s + j) % r.mq.len()];
                    let y = f32_tensor(&y_buf, n * r.m);
                    gp.gemv_mq8g256_b_x(w, &x_n, n, &y, r.m, r.k).unwrap();
                }
            });
            let cq = capture(gpu, cfg.launches, &mut q8_arm);
            let ck = capture(gpu, cfg.launches, &mut k_arm);
            let cc = capture(gpu, cfg.launches, &mut c_arm);
            let cp = capture(gpu, cfg.launches, &mut p_arm);
            let cf = capture(gpu, cfg.launches, &mut f_arm);
            let (mut tq, mut tk, mut tc, mut tp, mut tf) = (vec![], vec![], vec![], vec![], vec![]);
            let mut clk_pre = read_clk(&cfg.sysfs);
            let load0 = read_load();
            let mut sclks: Vec<f64> = vec![];
            for _ in 0..cfg.rounds {
                tq.push(replay_us(gpu, &cq, cfg));
                tk.push(replay_us(gpu, &ck, cfg));
                tc.push(replay_us(gpu, &cc, cfg));
                tf.push(replay_us(gpu, &cf, cfg));
                tp.push(replay_us(gpu, &cp, cfg));
                tp.push(replay_us(gpu, &cp, cfg));
                tf.push(replay_us(gpu, &cf, cfg));
                tc.push(replay_us(gpu, &cc, cfg));
                tk.push(replay_us(gpu, &ck, cfg));
                tq.push(replay_us(gpu, &cq, cfg));
                let c = read_clk(&cfg.sysfs);
                if c.sclk > 0 {
                    sclks.push(c.sclk as f64);
                }
                clk_pre = c;
            }
            let load1 = read_load();
            let (mq, mk, mc, mp, mf) = (
                median(&mut tq),
                median(&mut tk),
                median(&mut tc),
                median(&mut tp),
                median(&mut tf),
            );
            let (rk, rc, rf) = (mk / mq, mc / mq, mf / mq);
            // production policy: fused entry at n <= 2 (decode / short verify), shared producer + rows-exact above
            let fused_used = mq8g256_b_fused_route(k, n).is_some();
            let (rp, route) = if n <= 2 && fused_used {
                (rf, "fused")
            } else {
                (rc, "chain")
            };
            let sclk_med = if sclks.is_empty() {
                -1.0
            } else {
                median(&mut sclks)
            };
            println!(
                "ROW arch={} group={} n={n} | q8 {mq:8.2} us | mq8_kernel {mk:8.2} ({rk:.4}x) | mq8_chain {mc:8.2} ({rc:.4}x) | mq8_fused {mf:8.2} ({rf:.4}x) | POLICY {route} {rp:.4}x | producer {mp:6.2} | sclk_med {sclk_med:.0} hw {} MHz {} W load {:.2}/{:.2} -> {:.2}/{:.2}",
                gpu.arch, g.name, clk_pre.hw, clk_pre.pw, load0.0, load0.1, load1.0, load1.1
            );
            if REAL_GROUPS.contains(&g.name) {
                if rp > worst_policy_real.0 {
                    worst_policy_real = (rp, format!("{} n={n} ({route})", g.name));
                }
            } else if rp > worst_policy_other.0 {
                worst_policy_other = (rp, format!("{} n={n} ({route})", g.name));
            }
            let at = format!("{} n={n}", g.name);
            let single = s == 1;
            let (wk, wc) = if single {
                (&mut worst_kernel_single, &mut worst_chain_single)
            } else {
                (&mut worst_kernel_shared, &mut worst_chain_shared)
            };
            if rk > wk.0 {
                *wk = (rk, at.clone());
            }
            if rc > wc.0 {
                *wc = (rc, at.clone());
            }
            all_ratios.push((
                g.name.to_string(),
                n,
                Ratios {
                    kernel: rk,
                    chain: rc,
                },
            ));
            release(gpu, cq);
            release(gpu, ck);
            release(gpu, cc);
            release(gpu, cp);
            release(gpu, cf);
        }
    }
    println!();
    let verdict = |name: &str, w: &(f64, String)| {
        println!(
            "DECISION {name} {} (worst MQ8/Q8_0 = {:.4} at {})",
            if w.0 <= 1.0 { "PASS" } else { "FAIL" },
            w.0,
            w.1
        );
    };
    verdict("kernel_single_projection_vs_q8_0", &worst_kernel_single);
    verdict("kernel_shared_group_vs_q8_0", &worst_kernel_shared);
    verdict("chain_single_projection_vs_q8_0", &worst_chain_single);
    verdict("chain_shared_group_vs_q8_0", &worst_chain_shared);
    verdict("production_policy_real_groups_vs_q8_0", &worst_policy_real);
    verdict(
        "production_policy_other_projections_vs_q8_0",
        &worst_policy_other,
    );
    let _ = all_ratios;
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = match args.first() {
        Some(a) if !a.starts_with("--") => a.clone(),
        _ => "all".to_string(),
    };
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let num = |name: &str, default: usize| -> usize {
        flag(name).and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    let mut gpu = Gpu::init().expect("gpu init");
    let bdf = gpu
        .hip
        .device_pci_bus_id(gpu.device_id)
        .unwrap_or_default()
        .to_ascii_lowercase();
    if let Some(want) = flag("--expect-bdf") {
        if !bdf.ends_with(&want.to_ascii_lowercase()) {
            eprintln!("REFUSED: device pci {bdf} is not the expected {want}");
            std::process::exit(2);
        }
    }
    let cfg = Cfg {
        rounds: num("--rounds", 9),
        launches: num("--launches", 40),
        reps: num("--reps", 4),
        groups: flag("--groups").map(|s| s.split(',').map(str::to_string).collect()),
        ns: flag("--ns")
            .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
            .unwrap_or_else(|| vec![1, 2, 4, 8, 16]),
        sysfs: flag("--sysfs").unwrap_or_else(|| format!("/sys/bus/pci/devices/{bdf}")),
    };
    println!(
        "mq8_decode_lab arch={} pci={bdf} mode={mode} GPU_MAX_HW_QUEUES={} sysfs={}",
        gpu.arch,
        std::env::var("GPU_MAX_HW_QUEUES").unwrap_or_else(|_| "(unset)".into()),
        cfg.sysfs
    );
    gpu.ensure_capture_stream().expect("capture stream");
    let mut ok = true;
    if matches!(mode.as_str(), "identity" | "all") {
        ok &= identity(&mut gpu);
    }
    if matches!(mode.as_str(), "timing" | "all") {
        timing(&mut gpu, &cfg);
    }
    if !ok {
        std::process::exit(2);
    }
}
