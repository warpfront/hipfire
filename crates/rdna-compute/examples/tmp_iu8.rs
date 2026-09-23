// THROWAWAY iu8 bring-up harness — DELETE BEFORE COMMIT.
// Parity of the gfx11 int8-WMMA MQ4v2 gate_up candidate vs the MMQ incumbent
// and vs a CPU f64 reference, plus kill-gate timing.
// Run: cargo run --release -p rdna-compute --example tmp_iu8 -- parity
//      cargo run --release -p rdna-compute --example tmp_iu8 -- time <iu8bt8|iu8bt6|iu8bt4|mmq> <n>
use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};
use std::ffi::c_void;
use std::time::Instant;

const PACK_SRC: &str =
    include_str!("../../../kernels/src/pack_f32_to_i8_mq4v2.gfx11.hip");
const IU8_BT8_SRC: &str = concat!(
    "#define HIPFIRE_IU8_BV 8\n#define HIPFIRE_IU8_GATEUP_KERNEL gemm_gate_up_mq4g256v2_wmma_iu8_gfx11_bt8\n",
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256v2_wmma_iu8.gfx11.hip")
);
const IU8_BT6_SRC: &str = concat!(
    "#define HIPFIRE_IU8_BV 6\n#define HIPFIRE_IU8_GATEUP_KERNEL gemm_gate_up_mq4g256v2_wmma_iu8_gfx11_bt6\n",
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256v2_wmma_iu8.gfx11.hip")
);
const IU8_BT4_SRC: &str = concat!(
    "#define HIPFIRE_IU8_BV 4\n#define HIPFIRE_IU8_GATEUP_KERNEL gemm_gate_up_mq4g256v2_wmma_iu8_gfx11_bt4\n",
    include_str!("../../../kernels/src/gemm_gate_up_mq4g256v2_wmma_iu8.gfx11.hip")
);

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    // Standard normal via Box-Muller.
    fn gauss(&mut self) -> f32 {
        let u1 = (self.next() as f64 / u64::MAX as f64).max(1e-12);
        let u2 = self.next() as f64 / u64::MAX as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let s = ((bits >> 15) & 1) as i32;
    let e = ((bits >> 10) & 0x1f) as i32;
    let m = (bits & 0x3ff) as i32;
    let v = if e == 0 {
        m as f32 * (1.0 / 16777216.0)
    } else if e == 31 {
        if m == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (m as f32 / 1024.0 + 1.0) * 2.0f32.powi(e - 15)
    };
    if s == 1 { -v } else { v }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let s = (b >> 16) & 0x8000;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7fffff;
    if e == 255 {
        return (s | 0x7c00 | if m != 0 { 0x0200 } else { 0 }) as u16;
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return (s | 0x7c00) as u16;
    }
    if e16 <= 0 {
        if e16 < -10 {
            return s as u16;
        }
        let m16 = ((m | 0x800000) >> (14 - e16)) as u32;
        // round to nearest even on the dropped bits
        let shift = (14 - e16) as u32;
        let rem = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let inc = if rem > half || (rem == half && (m16 & 1) == 1) { 1 } else { 0 };
        return (s | (m16 + inc)) as u16;
    }
    let m16 = m >> 13;
    let rem = m & 0x1fff;
    let inc = if rem > 0x1000 || (rem == 0x1000 && (m16 & 1) == 1) { 1 } else { 0 };
    let m16 = m16 + inc;
    if m16 == 0x400 {
        return (s | (((e16 + 1) as u32) << 10)) as u16;
    }
    (s | ((e16 as u32) << 10) | m16) as u16
}

// Synthetic MQ4v2 weights with valid v2 headers: per-128-half fp16
// (scale, zero), 128 B nibble payload. Every 7th group kills half1's scale,
// every 11th group is fully dead (both scales 0, random payload).
fn build_mq4v2(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let gpr = k / 256;
    let bpr = gpr * 136;
    let mut rng = Rng(seed);
    let mut out = vec![0u8; m * bpr];
    for row in 0..m {
        for g in 0..gpr {
            let off = row * bpr + g * 136;
            let mut s0 = 0.002 + (rng.below(4800) as f32) * 1e-5;
            let mut z0 = (rng.below(10001) as f32) * 1e-4 - 0.5;
            let mut s1 = 0.002 + (rng.below(4800) as f32) * 1e-5;
            let mut z1 = (rng.below(10001) as f32) * 1e-4 - 0.5;
            if g % 7 == 6 {
                s1 = 0.0;
                z1 = 0.0;
            }
            if g % 11 == 10 {
                s0 = 0.0;
                z0 = 0.0;
                s1 = 0.0;
                z1 = 0.0;
            }
            out[off..off + 2].copy_from_slice(&f32_to_f16_bits(s0).to_le_bytes());
            out[off + 2..off + 4].copy_from_slice(&f32_to_f16_bits(z0).to_le_bytes());
            out[off + 4..off + 6].copy_from_slice(&f32_to_f16_bits(s1).to_le_bytes());
            out[off + 6..off + 8].copy_from_slice(&f32_to_f16_bits(z1).to_le_bytes());
            for i in 0..128 {
                out[off + 8 + i] = rng.below(256) as u8;
            }
        }
    }
    out
}

// CPU f64 reference: Y[oc*M+pr] = sum over K of (s*q+z)*x. One matrix.
fn cpu_ref(w: &[u8], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let gpr = k / 256;
    let bpr = gpr * 136;
    let mut y = vec![0f64; n * m];
    for oc in 0..n {
        for pr in 0..m {
            let mut acc = 0f64;
            for g in 0..gpr {
                let off = pr * bpr + g * 136;
                let s0 = f16_to_f32(u16::from_le_bytes([w[off], w[off + 1]])) as f64;
                let z0 = f16_to_f32(u16::from_le_bytes([w[off + 2], w[off + 3]])) as f64;
                let s1 = f16_to_f32(u16::from_le_bytes([w[off + 4], w[off + 5]])) as f64;
                let z1 = f16_to_f32(u16::from_le_bytes([w[off + 6], w[off + 7]])) as f64;
                for kk in 0..256 {
                    let byte = w[off + 8 + kk / 2];
                    let q = if kk % 2 == 0 { byte & 0xf } else { byte >> 4 } as f64;
                    let h = kk / 128;
                    let wv = if h == 0 { s0 * q + z0 } else { s1 * q + z1 };
                    acc += wv * x[oc * k + g * 256 + kk] as f64;
                }
            }
            y[oc * m + pr] = acc;
        }
    }
    y.into_iter().map(|v| v as f32).collect()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        num += (*x as f64 - *y as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num.sqrt() / den.sqrt().max(1e-30)) as f32
}

fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

struct Iu8Ctx {
    bv: usize,
    sym: &'static str,
    src: &'static str,
}

fn run_iu8(
    gpu: &mut Gpu,
    ctx: &Iu8Ctx,
    a_gate: &rdna_compute::GpuTensor,
    a_up: &rdna_compute::GpuTensor,
    x: &rdna_compute::GpuTensor,
    y_g: &rdna_compute::GpuTensor,
    y_u: &rdna_compute::GpuTensor,
    gate_m: usize,
    up_m: usize,
    k: usize,
    n: usize,
) {
    let gpr = k / 256;
    let x_i8 = gpu.alloc_tensor(&[ceil_div(n * k, 4)], DType::F32).unwrap();
    let hs = gpu
        .alloc_tensor(&[(n * gpr * 2)], DType::F32)
        .unwrap();
    let rs = gpu.alloc_tensor(&[n], DType::F32).unwrap();
    gpu.ensure_kernel_public("tmp_iu8_pack", PACK_SRC, "pack_f32_to_i8_mq4v2_gfx11")
        .expect("JIT pack");
    let modname = format!("tmp_iu8_{}", ctx.sym);
    gpu.ensure_kernel_public(&modname, ctx.src, ctx.sym)
        .expect("JIT iu8");
    // pack
    if std::env::var("TMP_IU8_DEBUG").as_deref() == Ok("1") {
        gpu.fill_f32(&rs, 7.0).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let rs_fill = gpu.download_f32(&rs).unwrap();
        eprintln!("rs-after-fill[0..4]={:?}", &rs_fill[..4]);
        let x_back = gpu.download_f32(x).unwrap();
        eprintln!("x[0..4]={:?} x_has_nan={}", &x_back[..4], x_back.iter().any(|v| !v.is_finite()));
    }
    let mut kb = KernargBlob::new();
    kb.push_ptr(x.buf.as_ptr() as *const c_void);
    kb.push_ptr(x_i8.buf.as_ptr() as *const c_void);
    kb.push_ptr(hs.buf.as_ptr() as *const c_void);
    kb.push_ptr(rs.buf.as_ptr() as *const c_void);
    kb.push_i32(k as i32);
    kb.push_i32(n as i32);
    gpu.launch_kernel_blob(
        "pack_f32_to_i8_mq4v2_gfx11",
        [n as u32, 1, 1],
        [256, 1, 1],
        0,
        kb.as_mut_slice(),
    )
    .expect("launch pack");
    if std::env::var("TMP_IU8_PACKONLY").as_deref() == Ok("1") {
        gpu.free_tensor(x_i8).ok();
        gpu.free_tensor(hs).ok();
        gpu.free_tensor(rs).ok();
        return;
    }
    if std::env::var("TMP_IU8_DEBUG").as_deref() == Ok("1") {
        gpu.hip.device_synchronize().unwrap();
        let rs_post = gpu.download_f32(&rs).unwrap();
        eprintln!("rs-post-pack[0..4]={:?}", &rs_post[..4]);
        let hs_post = gpu.download_f32(&hs).unwrap();
        eprintln!("hs-post-pack[0..4]={:?}", &hs_post[..4]);
    }
    // gemm
    let tm = gate_m + up_m;
    let mut kb = KernargBlob::new();
    kb.push_ptr(a_gate.buf.as_ptr() as *const c_void);
    kb.push_ptr(a_up.buf.as_ptr() as *const c_void);
    kb.push_ptr(x_i8.buf.as_ptr() as *const c_void);
    kb.push_ptr(hs.buf.as_ptr() as *const c_void);
    kb.push_ptr(rs.buf.as_ptr() as *const c_void);
    kb.push_ptr(y_g.buf.as_ptr() as *const c_void);
    kb.push_ptr(y_u.buf.as_ptr() as *const c_void);
    kb.push_i32(gate_m as i32);
    kb.push_i32(up_m as i32);
    kb.push_i32(k as i32);
    kb.push_i32(n as i32);
    gpu.launch_kernel_blob(
        ctx.sym,
        [ceil_div(tm, 16) as u32, ceil_div(n, 16 * ctx.bv) as u32, 1],
        [32, 1, 1],
        0,
        kb.as_mut_slice(),
    )
    .expect("launch iu8");
    if std::env::var("TMP_IU8_DEBUG").as_deref() == Ok("1") {
        let rs_h = gpu.download_f32(&rs).unwrap();
        eprintln!("ax[0..8]={:?}", &rs_h[..8.min(rs_h.len())]);
        let hs_h = gpu.download_f32(&hs).unwrap();
        eprintln!("hs[0..8]={:?}", &hs_h[..8.min(hs_h.len())]);
        let yg_h = gpu.download_f32(y_g).unwrap();
        eprintln!("y_g[0..8]={:?}", &yg_h[..8.min(yg_h.len())]);
    }
    gpu.free_tensor(x_i8).ok();
    gpu.free_tensor(hs).ok();
    gpu.free_tensor(rs).ok();
}

fn parity_ones(gpu: &mut Gpu, ctx: &Iu8Ctx, gate_m: usize, up_m: usize, k: usize, n: usize) {
    let wg = build_mq4v2(gate_m, k, 0x1111);
    let wu = build_mq4v2(up_m, k, 0x2222);
    let x_f32: Vec<f32> = vec![1.0; n * k];
    let a_gate = gpu.upload_raw(&wg, &[gate_m, k]).unwrap();
    let a_up = gpu.upload_raw(&wu, &[up_m, k]).unwrap();
    let x = gpu.upload_f32(&x_f32, &[n, k]).unwrap();
    let y_g = gpu.alloc_tensor(&[n, gate_m], DType::F32).unwrap();
    let y_u = gpu.alloc_tensor(&[n, up_m], DType::F32).unwrap();
    run_iu8(gpu, ctx, &a_gate, &a_up, &x, &y_g, &y_u, gate_m, up_m, k, n);
    gpu.hip.device_synchronize().unwrap();
    let iu8_g = gpu.download_f32(&y_g).unwrap();
    let iu8_u = gpu.download_f32(&y_u).unwrap();
    let ref_g = cpu_ref(&wg, &x_f32, gate_m, k, n);
    let ref_u = cpu_ref(&wu, &x_f32, up_m, k, n);
    let r_g = rel_l2(&iu8_g, &ref_g);
    let r_u = rel_l2(&iu8_u, &ref_u);
    eprintln!("parity-ones gate_m={gate_m} up_m={up_m} K={k} N={n}: relL2 g={r_g:.2e} u={r_u:.2e} => {}",
        if r_g <= 1e-3 && r_u <= 1e-3 { "PASS" } else { "FAIL" });
    gpu.free_tensor(a_gate).ok();
    gpu.free_tensor(a_up).ok();
    gpu.free_tensor(x).ok();
    gpu.free_tensor(y_g).ok();
    gpu.free_tensor(y_u).ok();
    if !(r_g <= 1e-3 && r_u <= 1e-3) {
        std::process::exit(1);
    }
}

fn parity_one(
    gpu: &mut Gpu,
    ctx: &Iu8Ctx,
    gate_m: usize,
    up_m: usize,
    k: usize,
    n: usize,
    seed: u64,
) {
    let mut rng = Rng(seed);
    let wg = build_mq4v2(gate_m, k, seed ^ 0x1111);
    let wu = build_mq4v2(up_m, k, seed ^ 0x2222);
    let x_f32: Vec<f32> = (0..n * k).map(|_| rng.gauss() * 2.0).collect();
    let a_gate = gpu.upload_raw(&wg, &[gate_m, k]).unwrap();
    let a_up = gpu.upload_raw(&wu, &[up_m, k]).unwrap();
    let x = gpu.upload_f32(&x_f32, &[n, k]).unwrap();
    let y_g = gpu.alloc_tensor(&[n, gate_m], DType::F32).unwrap();
    let y_u = gpu.alloc_tensor(&[n, up_m], DType::F32).unwrap();
    let y_g2 = gpu.alloc_tensor(&[n, gate_m], DType::F32).unwrap();
    let y_u2 = gpu.alloc_tensor(&[n, up_m], DType::F32).unwrap();

    run_iu8(gpu, ctx, &a_gate, &a_up, &x, &y_g, &y_u, gate_m, up_m, k, n);
    gpu.gemm_gate_up_mq4g256v2_wmma(
        &a_gate, &a_up, &x, &y_g2, &y_u2, gate_m, up_m, k, n,
    )
    .expect("mmq ref");
    gpu.hip.device_synchronize().unwrap();
    let iu8_g = gpu.download_f32(&y_g).unwrap();
    let iu8_u = gpu.download_f32(&y_u).unwrap();
    let mmq_g = gpu.download_f32(&y_g2).unwrap();
    let mmq_u = gpu.download_f32(&y_u2).unwrap();

    let ref_g = cpu_ref(&wg, &x_f32, gate_m, k, n);
    let ref_u = cpu_ref(&wu, &x_f32, up_m, k, n);
    if std::env::var("TMP_IU8_DEBUG").as_deref() == Ok("1") {
        eprintln!("iu8_g[0..6]={:?}", &iu8_g[..6.min(iu8_g.len())]);
        eprintln!("ref_g[0..6]={:?}", &ref_g[..6.min(ref_g.len())]);
        eprintln!("mmq_g[0..6]={:?}", &mmq_g[..6.min(mmq_g.len())]);
    }
    let r_iu8_g = rel_l2(&iu8_g, &ref_g);
    let r_iu8_u = rel_l2(&iu8_u, &ref_u);
    let r_mmq_g = rel_l2(&mmq_g, &ref_g);
    let r_mmq_u = rel_l2(&mmq_u, &ref_u);
    let r_x_g = rel_l2(&iu8_g, &mmq_g);
    let r_x_u = rel_l2(&iu8_u, &mmq_u);
    // Breakage bound: theoretical per-row int8 floor is ~9e-3 (see yield
    // notes); >5e-2 means systematic error, not quant noise.
    let pass = r_iu8_g <= 5e-2 && r_iu8_u <= 5e-2;
    eprintln!(
        "parity gate_m={gate_m} up_m={up_m} K={k} N={n}: iu8-vs-CPU relL2 g={r_iu8_g:.2e} u={r_iu8_u:.2e} | mmq-vs-CPU g={r_mmq_g:.2e} u={r_mmq_u:.2e} | iu8-vs-mmq g={r_x_g:.2e} u={r_x_u:.2e} => {}",
        if pass { "PASS" } else { "FAIL" }
    );
    gpu.free_tensor(a_gate).ok();
    gpu.free_tensor(a_up).ok();
    gpu.free_tensor(x).ok();
    gpu.free_tensor(y_g).ok();
    gpu.free_tensor(y_u).ok();
    gpu.free_tensor(y_g2).ok();
    gpu.free_tensor(y_u2).ok();
    if !pass {
        std::process::exit(1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("arch={}", gpu.arch);
    let ctx8 = Iu8Ctx {
        bv: 8,
        sym: "gemm_gate_up_mq4g256v2_wmma_iu8_gfx11_bt8",
        src: IU8_BT8_SRC,
    };
    if args.get(1).map(|s| s.as_str()) == Some("parity") {
        // Exact-X math gate: X uniform -> x_i8 exact (no quant noise), so
        // iu8-vs-CPU must be exact to fp32 rounding. Catches math bugs
        // (mapping/fold/sign); Gaussian-X cases below characterize the
        // designed per-row int8 quant-noise floor (~9e-3 theoretical).
        parity_ones(&mut gpu, &ctx8, 512, 512, 5120, 128);
        for &n in &[128usize, 256, 384, 512] {
            parity_one(&mut gpu, &ctx8, 512, 512, 5120, n, 0xC0FFEE);
        }
        // M tail (%16 != 0 exercises the sr clamp + orow guard).
        parity_one(&mut gpu, &ctx8, 500, 500, 5120, 128, 0x5EED);
        return;
    }
    // time <arm> <n>: production gate/up dims.
    let arm = args.get(2).map(|s| s.as_str()).unwrap_or("iu8bt8");
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
    let (gate_m, up_m, k) = (17408usize, 17408usize, 5120usize);
    let flop = 2.0 * n as f64 * k as f64 * (gate_m + up_m) as f64;
    let mut rng = Rng(0xBEEF);
    let wg = build_mq4v2(gate_m, k, 0x1111);
    let wu = build_mq4v2(up_m, k, 0x2222);
    let x_f32: Vec<f32> = (0..n * k).map(|_| rng.gauss()).collect();
    let a_gate = gpu.upload_raw(&wg, &[gate_m, k]).unwrap();
    let a_up = gpu.upload_raw(&wu, &[up_m, k]).unwrap();
    let x = gpu.upload_f32(&x_f32, &[n, k]).unwrap();
    let y_g = gpu.alloc_tensor(&[n, gate_m], DType::F32).unwrap();
    let y_u = gpu.alloc_tensor(&[n, up_m], DType::F32).unwrap();
    let ctx = match arm {
        "iu8bt8" => Iu8Ctx {
            bv: 8,
            sym: "gemm_gate_up_mq4g256v2_wmma_iu8_gfx11_bt8",
            src: IU8_BT8_SRC,
        },
        "iu8bt6" => Iu8Ctx {
            bv: 6,
            sym: "gemm_gate_up_mq4g256v2_wmma_iu8_gfx11_bt6",
            src: IU8_BT6_SRC,
        },
        "iu8bt4" => Iu8Ctx {
            bv: 4,
            sym: "gemm_gate_up_mq4g256v2_wmma_iu8_gfx11_bt4",
            src: IU8_BT4_SRC,
        },
        _ => {
            for _ in 0..3 {
                gpu.gemm_gate_up_mq4g256v2_wmma(
                    &a_gate, &a_up, &x, &y_g, &y_u, gate_m, up_m, k, n,
                )
                .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..30 {
                gpu.gemm_gate_up_mq4g256v2_wmma(
                    &a_gate, &a_up, &x, &y_g, &y_u, gate_m, up_m, k, n,
                )
                .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / 30.0;
            eprintln!("arm=mmq N={n} us/call={us:.1} TFLOPS={:.2}", flop / (us * 1e-6) / 1e12);
            return;
        }
    };
    for _ in 0..3 {
        run_iu8(&mut gpu, &ctx, &a_gate, &a_up, &x, &y_g, &y_u, gate_m, up_m, k, n);
    }
    gpu.hip.device_synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..30 {
        run_iu8(&mut gpu, &ctx, &a_gate, &a_up, &x, &y_g, &y_u, gate_m, up_m, k, n);
    }
    gpu.hip.device_synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / 30.0;
    eprintln!("arm={arm} N={n} us/call={us:.1} TFLOPS={:.2}", flop / (us * 1e-6) / 1e12);
}
