// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201: the dense-GDN prefill chain (`fn_gdn_dense::run`: producer, KKT
//! solve, chunk scan) fed the qkv projection as BF16 bits instead of F32 must
//! produce exactly the bytes of the chain fed the F32 projection: every
//! changed intermediate (the packed F16 q/k/v planes, g, beta, the conv
//! history ring, the Q8 state codes/scales, the scan's BF16 output plane) is
//! compared bytewise. The producer only ever reads the projection through its
//! BF16 rounding, so the two inputs are the same operand. The BF16 operand is
//! built with the host twin of `hip_bfloat16`'s float -> bf16 (RNE, NaN/inf
//! passthrough), the rounding the F32 path applies in-kernel.
//!
//! The qkv GEMM that supplies the BF16 operand (the exact-INT8 MQ6 route's BF16
//! entry) must in turn store `RNE(F32 accumulator)`: `i8_gemm_bf16_entry_is_rne_of_f32_entry`
//! compares its output bits with the host rounding of the F32 entry's bits.
//!
//! `#[ignore]`d: needs an exact gfx1201 with a working HIP toolchain.
//!
//!   cargo test -p rdna-compute --release --features deltanet \
//!       --test fn_gdn_chain_oracle -- --ignored --test-threads=1 --nocapture
//!
//! `microbench` prints hipEvent medians of the dense chain (producer + KKT +
//! scan) for the F32 and the BF16 projection.

#![cfg(feature = "deltanet")]

use half::f16;
use rdna_compute::fn_gdn_dense;
use rdna_compute::tensor_ops::{
    GatedDeltaConvBatched, GatedDeltaParamsBatched, GatedDeltaStepBatched, GdnStateFormat,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const CH: usize = 10240;
const HV: usize = 48;
const STATE_BYTES: usize = HV * 128 * (128 + 4);

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Host twin of `hip_bfloat16`'s float -> bf16 (`float_to_bfloat16`).
fn bf16_hip(f: f32) -> u16 {
    let mut u = f.to_bits();
    if (!u) & 0x7f80_0000 != 0 {
        u = u.wrapping_add(0x7fff + ((u >> 16) & 1));
    } else if u & 0xffff != 0 {
        u |= 0x1_0000;
    }
    (u >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn bytes_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Which special values the generator injects into the projection.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// Ties, subnormals, +-0, large magnitudes; every value finite.
    Finite,
    /// Also NaN (two payloads) and +-inf in a few rows.
    NonFinite,
}

struct Case {
    rows: usize,
    input: Vec<f32>,
    kernel: Vec<u16>,
    history: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    a_log: Vec<u16>,
    dt_bias: Vec<u16>,
    state: Vec<u8>,
    start_cursor: usize,
}

const FINITE_SPECIAL: [u32; 12] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x807f_ffff,
    0x3f80_8000,
    0x3f80_7fff,
    0x3f80_8001,
    0xbf81_8000,
    0x7f7f_ffff,
    0xff7f_ffff,
];
const NONFINITE_SPECIAL: [u32; 6] = [
    0x7fc0_0000,
    0xffc0_0001,
    0x7f80_0000,
    0xff80_0000,
    0x7f80_0001,
    0x7fff_ffff,
];

fn gen_case(rows: usize, seed: u64, mode: Mode) -> Case {
    let mut r = Rng(seed | 1);
    let mut input = vec![0f32; rows * CH];
    for (row, line) in input.chunks_mut(CH).enumerate() {
        for (c, x) in line.iter_mut().enumerate() {
            let scale = match c / 128 {
                0..=31 => 2.0 + (c / 128) as f32 * 0.25,
                _ => 1.0 + ((c / 128) % 5) as f32,
            };
            *x = r.unit() * scale;
            if r.below(150) == 0 {
                *x = f32::from_bits(FINITE_SPECIAL[r.below(FINITE_SPECIAL.len() as u64) as usize]);
            }
            if mode == Mode::NonFinite && row % 97 == 5 && r.below(60) == 0 {
                *x = f32::from_bits(
                    NONFINITE_SPECIAL[r.below(NONFINITE_SPECIAL.len() as u64) as usize],
                );
            }
        }
        // All-zero head rows keep the zero-norm path alive.
        if row % 41 == 3 {
            let head = (row / 41) % 32;
            line[head * 128..head * 128 + 128].fill(0.0);
        }
    }
    let kernel: Vec<u16> = (0..CH * 4).map(|_| bf16_hip(r.unit() * 0.6)).collect();
    let history: Vec<f32> = (0..3 * CH)
        .map(|_| bf16_to_f32(bf16_hip(r.unit() * 2.0)))
        .collect();
    let a: Vec<f32> = (0..rows * HV).map(|_| r.unit() * 8.0).collect();
    let b: Vec<f32> = (0..rows * HV).map(|_| r.unit() * 6.0).collect();
    let a_log: Vec<u16> = (0..HV).map(|_| bf16_hip(r.unit() * 1.5)).collect();
    let dt_bias: Vec<u16> = (0..HV).map(|_| bf16_hip(r.unit() * 2.0)).collect();
    let mut state = vec![0u8; STATE_BYTES];
    for byte in state[..HV * 128 * 128].iter_mut() {
        *byte = (r.next() >> 56) as u8;
    }
    for s in 0..HV * 128 {
        let off = HV * 128 * 128 + s * 4;
        state[off..off + 4].copy_from_slice(&(0.002 + r.unit().abs() * 0.01f32).to_le_bytes());
    }
    Case {
        rows,
        input,
        kernel,
        history,
        a,
        b,
        a_log,
        dt_bias,
        state,
        start_cursor: (seed % 3) as usize,
    }
}

#[derive(PartialEq, Eq)]
struct Outputs {
    out: Vec<u8>,
    state: Vec<u8>,
    proj: Vec<u8>,
    gate: Vec<u8>,
    beta: Vec<u8>,
    history: Vec<u8>,
}

struct Operands {
    input: GpuTensor,
    kernel: GpuTensor,
    history: GpuTensor,
    a: GpuTensor,
    b: GpuTensor,
    a_log: GpuTensor,
    dt_bias: GpuTensor,
    gate: GpuTensor,
    beta: GpuTensor,
    proj: GpuTensor,
    state: GpuTensor,
    out: GpuTensor,
}

fn typed(t: GpuTensor, dtype: DType, shape: Vec<usize>) -> GpuTensor {
    GpuTensor { dtype, shape, ..t }
}

fn operands(gpu: &mut Gpu, c: &Case, bf16_input: bool) -> Operands {
    let n = c.rows;
    let input = if bf16_input {
        let bits: Vec<u16> = c.input.iter().map(|&x| bf16_hip(x)).collect();
        let t = gpu
            .upload_raw(&bytes_u16(&bits), &[n * CH * 2])
            .expect("input bf16");
        typed(t, DType::BF16, vec![n * CH])
    } else {
        gpu.upload_f32(&c.input, &[n * CH]).expect("input f32")
    };
    let kernel = {
        let t = gpu
            .upload_raw(&bytes_u16(&c.kernel), &[CH * 4 * 2])
            .expect("kernel");
        typed(t, DType::BF16, vec![CH * 4])
    };
    let history = gpu.upload_f32(&c.history, &[3 * CH]).expect("history");
    let a = gpu.upload_f32(&c.a, &[n * HV]).expect("a");
    let b = gpu.upload_f32(&c.b, &[n * HV]).expect("b");
    let a_log = typed(
        gpu.upload_raw(&bytes_u16(&c.a_log), &[HV * 2])
            .expect("a_log"),
        DType::BF16,
        vec![HV],
    );
    let dt_bias = typed(
        gpu.upload_raw(&bytes_u16(&c.dt_bias), &[HV * 2])
            .expect("dt"),
        DType::BF16,
        vec![HV],
    );
    let gate = gpu.zeros(&[n * HV], DType::F32).expect("gate");
    let beta = gpu.zeros(&[n * HV], DType::F32).expect("beta");
    let proj = gpu.zeros(&[n * CH], DType::F32).expect("proj");
    let state = typed(
        gpu.upload_raw(&c.state, &[STATE_BYTES]).expect("state"),
        DType::Raw,
        vec![STATE_BYTES],
    );
    let out = gpu.zeros(&[n * 6144], DType::F32).expect("out");
    Operands {
        input,
        kernel,
        history,
        a,
        b,
        a_log,
        dt_bias,
        gate,
        beta,
        proj,
        state,
        out,
    }
}

fn run_chain(gpu: &mut Gpu, o: &Operands, c: &Case) {
    let n = c.rows;
    let conv = GatedDeltaConvBatched {
        input: &o.input,
        kernel: &o.kernel,
        history: &o.history,
        output: &o.proj,
        next_history: &o.history,
        rows: n,
        channels: CH,
        history_rows: 3,
        kernel_size: 4,
        start_cursor: c.start_cursor,
    };
    let params = GatedDeltaParamsBatched {
        a: &o.a,
        b: &o.b,
        a_log: &o.a_log,
        dt_bias: &o.dt_bias,
        gate: &o.gate,
        beta: &o.beta,
        rows: n,
        heads: HV,
    };
    let step = GatedDeltaStepBatched {
        projection: &o.proj,
        gate: &o.gate,
        beta: &o.beta,
        state: &o.state,
        format: GdnStateFormat::Q8,
        residual: None,
        output: &o.out,
        row_states: None,
        rows: n,
        qkv_width: CH,
        key_heads: 16,
        value_heads: HV,
        key_dim: 128,
        value_dim: 128,
        position: 0,
    };
    fn_gdn_dense::run(gpu, &conv, &params, &step).expect("fn_gdn_dense::run");
}

fn download(gpu: &mut Gpu, o: &Operands, rows: usize) -> Outputs {
    gpu.hip.device_synchronize().expect("sync");
    let mut out = gpu.download_raw_bytes(&o.out).expect("out");
    out.truncate(rows * 6144 * 2);
    Outputs {
        out,
        state: gpu.download_raw_bytes(&o.state).expect("state"),
        proj: gpu.download_raw_bytes(&o.proj).expect("proj"),
        gate: gpu.download_raw_bytes(&o.gate).expect("gate"),
        beta: gpu.download_raw_bytes(&o.beta).expect("beta"),
        history: gpu.download_raw_bytes(&o.history).expect("history"),
    }
}

fn free(gpu: &mut Gpu, o: Operands) {
    for t in [
        o.input, o.kernel, o.history, o.a, o.b, o.a_log, o.dt_bias, o.gate, o.beta, o.proj,
        o.state, o.out,
    ] {
        gpu.free_tensor(t).expect("free");
    }
}

fn require_gfx1201() -> Option<Gpu> {
    let gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("skip: needs gfx1201, found {}", gpu.arch);
        return None;
    }
    Some(gpu)
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

fn compare(what: &str, rows: usize, a: &Outputs, b: &Outputs) -> usize {
    let mut total = 0;
    for (name, x, y) in [
        ("scan out", &a.out, &b.out),
        ("state", &a.state, &b.state),
        ("q/k/v planes", &a.proj, &b.proj),
        ("g", &a.gate, &b.gate),
        ("beta", &a.beta, &b.beta),
        ("conv history", &a.history, &b.history),
    ] {
        assert_eq!(x.len(), y.len(), "{what} rows {rows} {name}: length");
        let bad = x.iter().zip(y).filter(|(p, q)| p != q).count();
        if bad != 0 && name == "q/k/v planes" {
            // Locate the first differing F16 elements: plane, row, column.
            let n64 = rows.div_ceil(64) * 64;
            let widths = [("q", 2048usize), ("k", 2048), ("v", 6144)];
            let mut shown = 0;
            for (i, (p, q)) in x.chunks_exact(2).zip(y.chunks_exact(2)).enumerate() {
                if p != q && shown < 12 {
                    let mut e = i;
                    for (plane, w) in widths {
                        if e < n64 * w {
                            eprintln!(
                                "  {plane} row {} col {}: ref {:02x}{:02x} got {:02x}{:02x}",
                                e / w,
                                e % w,
                                p[1],
                                p[0],
                                q[1],
                                q[0]
                            );
                            break;
                        }
                        e -= n64 * w;
                    }
                    shown += 1;
                }
            }
        }
        assert_eq!(
            bad,
            0,
            "{what} rows {rows} {name}: {bad} of {} bytes differ (first at {:?})",
            x.len(),
            first_diff(x, y)
        );
        total += x.len();
    }
    total
}

fn set_chain_fuse(gpu: &mut Gpu, on: bool) {
    std::sync::Arc::make_mut(&mut gpu.flags).qwen4_gdn_chain_fuse = on;
    assert_eq!(gpu.flags.qwen4_gdn_chain_fuse_enabled(), on);
}

fn run_variant(gpu: &mut Gpu, c: &Case, fuse: bool, bf16_input: bool) -> Outputs {
    set_chain_fuse(gpu, fuse);
    let o = operands(gpu, c, bf16_input);
    run_chain(gpu, &o, c);
    let r = download(gpu, &o, c.rows);
    free(gpu, o);
    r
}

#[test]
#[ignore = "needs an exact gfx1201"]
fn chain_fuse_is_bytewise_todays_chain() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    let rows_list = [64usize, 65, 130, 777, 1024, 4096, 4160, 16384];
    for (n, rows) in rows_list.into_iter().enumerate() {
        for mode in [Mode::Finite, Mode::NonFinite] {
            let c = gen_case(rows, 0x5eed_0000 + n as u64 * 17 + mode as u64, mode);
            // Today's chain: the v1 producer on the F32 projection (kill switch).
            let reference = run_variant(&mut gpu, &c, false, false);
            let mut bytes = 0;
            // The v1 producer also reads BF16 bits; the load-batched entries read either.
            for (fuse, bf16, what) in [
                (false, true, "v1 producer, bf16 qkv"),
                (true, false, "v2 producer, f32 qkv"),
                (true, true, "v2 producer, bf16 qkv"),
            ] {
                let got = run_variant(&mut gpu, &c, fuse, bf16);
                bytes = compare(what, rows, &reference, &got);
            }
            // Not vacuous: the scan produced non-zero output (Finite mode).
            if mode == Mode::Finite {
                let nz = reference
                    .out
                    .chunks_exact(2)
                    .filter(|p| p[0] != 0 || (p[1] & 0x7f) != 0)
                    .count();
                assert!(nz > rows * 6144 / 2, "rows {rows}: scan output mostly zero");
            }
            eprintln!("rows {rows:>6} {mode:?}: 3 variants x {bytes} intermediate/output bytes identical to today's chain");
        }
    }
}

// ---- exhaustive check of the producer's SiLU fast path ----------------------------------

/// Test kernel appended to the production module source, so it calls the very device
/// functions the producer uses. One thread per F32 bit pattern.
const SILU_EXHAUSTIVE: &str = r#"
extern "C" __global__ __launch_bounds__(256) void fngd_silu_exhaustive(unsigned* stats, unsigned base) {
    const unsigned bits = base + blockIdx.x * 256u + threadIdx.x;
    const float t = __uint_as_float(bits);
    const __hip_bfloat16 ref = fn_gdn_dense_silu_bf16_exact(t);
    __hip_bfloat16 fast;
    const bool ok = fn_gdn_dense_silu_bf16_fast(t, fast);
    const __hip_bfloat16 prod = fn_gdn_dense_silu_bf16(t);
    const bool bad = prod.data != ref.data;
    // The fast path's own answer must equal the exact one wherever it is accepted.
    const bool bad_fast = ok && fast.data != ref.data;
    const float a = fabsf(t);
    const bool in_domain = a >= 9.5367431640625e-07f && a <= 16.0f;
    unsigned dd = 0;
    if (in_domain) {
        const float u = __builtin_amdgcn_exp2f(t * -1.4426950408889634f);
        const float y = t * __builtin_amdgcn_rcpf(1.0f + u);
        const float ry = t * (1.0f / (1.0f + expf(-t)));
        const int d = (int)__float_as_uint(y) - (int)__float_as_uint(ry);
        dd = (unsigned)(d < 0 ? -d : d);
    }
    const int nbad = __syncthreads_count(bad);
    const int nbadf = __syncthreads_count(bad_fast);
    const int nok = __syncthreads_count(ok);
    const int ndom = __syncthreads_count(in_domain);
    if (threadIdx.x == 0) {
        if (nbad) atomicAdd(&stats[0], (unsigned)nbad);
        if (nbadf) atomicAdd(&stats[1], (unsigned)nbadf);
        atomicAdd(&stats[2], (unsigned)nok);
        atomicAdd(&stats[3], (unsigned)ndom);
    }
    if (bad) atomicMin(&stats[4], bits);
    atomicMax(&stats[5], dd);
}
"#;

#[test]
#[ignore = "needs an exact gfx1201"]
fn silu_fast_path_is_exact_over_all_f32() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    let dir = std::env::temp_dir().join(format!("fngd_silu_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("silu.hip");
    let obj = dir.join("silu.hsaco");
    std::fs::write(
        &src,
        format!(
            "{}\n{}",
            rdna_compute::kernels::FN_GDN_DENSE_SRC,
            SILU_EXHAUSTIVE
        ),
    )
    .unwrap();
    // The recipe of `core_hipcc_flags` (the module's own compile flags).
    let status = std::process::Command::new("hipcc")
        .args([
            "--genco",
            "--offload-arch=gfx1201",
            "-O3",
            "--no-offload-compress",
            "-fuse-cuid=none",
            "-mcode-object-version=6",
            "-o",
        ])
        .arg(&obj)
        .arg(&src)
        .status()
        .expect("hipcc");
    assert!(status.success(), "hipcc failed");
    let module = gpu.hip.module_load(obj.to_str().unwrap()).expect("module");
    let func = gpu
        .hip
        .module_get_function(&module, "fngd_silu_exhaustive")
        .expect("function");
    let mut init = vec![0u8; 6 * 4];
    init[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    let stats = gpu.upload_raw(&init, &[24]).expect("stats");
    const CHUNK_BLOCKS: u32 = 1 << 20; // 2^28 values per launch
    for chunk in 0u32..16 {
        let base: u32 = chunk << 28;
        let mut p = stats.buf.as_ptr();
        let mut b = base;
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &mut p as *mut _ as *mut std::ffi::c_void,
            &mut b as *mut _ as *mut std::ffi::c_void,
        ];
        // SAFETY: both kernel arguments are live for the launch; the kernel signature is
        // (unsigned*, unsigned).
        unsafe {
            gpu.hip
                .launch_kernel(
                    &func,
                    [CHUNK_BLOCKS, 1, 1],
                    [256, 1, 1],
                    0,
                    None,
                    &mut params,
                )
                .expect("launch");
        }
        gpu.hip.device_synchronize().unwrap();
    }
    let raw = gpu.download_raw_bytes(&stats).unwrap();
    let w = |i: usize| u32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
    eprintln!(
        "silu exhaustive over 2^32 F32 inputs: mismatches(production vs exact)={} fast-accepted-wrong={} accepted={} in-domain={} first-bad-bits={:#010x} max|y_fast-y_exact|={} ulp",
        w(0), w(1), w(2), w(3), w(4), w(5)
    );
    assert_eq!(
        w(0),
        0,
        "production SiLU differs from the exact chain (first at {:#010x})",
        w(4)
    );
    assert_eq!(w(1), 0);
    assert!(
        w(2) as f64 > 0.99 * w(3) as f64 * 0.0 + 1.0,
        "fast path never accepted"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ---- the qkv GEMM's BF16 entry ----------------------------------------------------------

const GROUP_BYTES: usize = 200;

fn gen_weights(m: usize, k: usize, seed: u64, big: bool) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    let mut w = vec![0u8; m * (k / 256) * GROUP_BYTES];
    for g in w.as_chunks_mut::<GROUP_BYTES>().0 {
        for h in 0..2 {
            let mag = if big { 30.0 } else { 1.0 };
            let s = f16::from_f32((1e-4 + r.unit().abs() * 1e-3) * mag);
            let z = f16::from_f32(-32.0 * s.to_f32() + r.unit() * s.to_f32());
            g[4 * h..4 * h + 2].copy_from_slice(&s.to_bits().to_le_bytes());
            g[4 * h + 2..4 * h + 4].copy_from_slice(&z.to_bits().to_le_bytes());
        }
        for c in g[8..].chunks_mut(8) {
            let v = r.next().to_le_bytes();
            c.copy_from_slice(&v[..c.len()]);
        }
    }
    w
}

#[test]
#[ignore = "needs an exact gfx1201"]
fn i8_gemm_bf16_entry_is_rne_of_f32_entry() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    let (m, k) = (CH, 2048);
    for (n, big) in [false, true].into_iter().enumerate() {
        let w = gpu
            .upload_raw(
                &gen_weights(m, k, 0x77 + n as u64, big),
                &[m * (k / 256) * GROUP_BYTES],
            )
            .expect("w");
        for rows in [1usize, 16, 777, 4096, 16384] {
            let mut r = Rng(0xabcd + rows as u64);
            let x: Vec<f32> = (0..rows * k)
                .map(|i| {
                    let v = r.unit() * 3.0;
                    // A few outlier rows keep every A8 scale regime alive.
                    if (i / k) % 53 == 7 {
                        v * 40.0
                    } else {
                        v
                    }
                })
                .collect();
            let xt = gpu.upload_f32(&x, &[rows * k]).expect("x");
            let y32 = gpu.zeros(&[rows * m], DType::F32).expect("y32");
            let y16 = gpu.zeros(&[rows * m], DType::BF16).expect("y16");
            let xq = gpu.mq6_i8_prepare_x(&xt, k, rows).expect("prepare");
            gpu.gemm_mq6g256v2_i8_prequant(&w, xq, &y32, m, k, rows)
                .expect("f32 entry");
            gpu.gemm_mq6g256v2_i8_prequant(&w, xq, &y16, m, k, rows)
                .expect("bf16 entry");
            let f = gpu.download_raw_bytes(&y32).unwrap();
            let h = gpu.download_raw_bytes(&y16).unwrap();
            let mut bad = 0usize;
            let mut nonzero = 0usize;
            for (i, word) in f.chunks_exact(4).enumerate() {
                let v = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                let want = bf16_hip(v);
                let got = u16::from_le_bytes([h[2 * i], h[2 * i + 1]]);
                nonzero += (v != 0.0) as usize;
                bad += (want != got) as usize;
            }
            assert_eq!(
                bad, 0,
                "rows {rows} big={big}: {bad} BF16 words differ from RNE(F32)"
            );
            assert!(
                nonzero > rows * m / 2,
                "rows {rows}: GEMM output mostly zero"
            );
            eprintln!(
                "rows {rows:>5} M={m} K={k} big={big}: {} BF16 words == RNE(F32 entry)",
                rows * m
            );
            for t in [xt, y32, y16] {
                gpu.free_tensor(t).unwrap();
            }
        }
        gpu.free_tensor(w).unwrap();
    }
}

// ---- microbench ----------------------------------------------------------------------

fn median_us(gpu: &mut Gpu, iters: usize, mut f: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..3 {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    let mut s = Vec::with_capacity(iters);
    for _ in 0..iters {
        gpu.hip.event_record(&start, None).unwrap();
        f(gpu);
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        s.push(gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1000.0);
    }
    gpu.hip.event_destroy(start).unwrap();
    gpu.hip.event_destroy(stop).unwrap();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

#[test]
#[ignore = "needs an exact gfx1201; timing"]
fn microbench() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    for rows in [4096usize, 16384] {
        let c = gen_case(rows, 0x5eed, Mode::Finite);
        for (fuse, bf16, what) in [
            (false, false, "today(v1,f32)"),
            (true, false, "v2,f32"),
            (true, true, "v2,bf16"),
        ] {
            set_chain_fuse(&mut gpu, fuse);
            let o = operands(&mut gpu, &c, bf16);
            let t = median_us(&mut gpu, 20, |g| run_chain(g, &o, &c));
            free(&mut gpu, o);
            println!("MICROBENCH rows={rows} chain {what}={t:.1}us");
        }
    }
}

/// The qkv GEMM (M = 10240, K = 2048): F32 entry vs BF16 entry, same prequantized X.
#[test]
#[ignore = "needs an exact gfx1201; timing"]
fn microbench_qkv_gemm_entries() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    let (m, k) = (CH, 2048);
    let w = gpu
        .upload_raw(
            &gen_weights(m, k, 0x77, false),
            &[m * (k / 256) * GROUP_BYTES],
        )
        .expect("w");
    for rows in [4096usize, 16384] {
        let mut r = Rng(0xabcd + rows as u64);
        let x: Vec<f32> = (0..rows * k).map(|_| r.unit() * 3.0).collect();
        let xt = gpu.upload_f32(&x, &[rows * k]).expect("x");
        let y32 = gpu.zeros(&[rows * m], DType::F32).expect("y32");
        let y16 = gpu.zeros(&[rows * m], DType::BF16).expect("y16");
        let xq = gpu.mq6_i8_prepare_x(&xt, k, rows).expect("prepare");
        let t32 = median_us(&mut gpu, 20, |g| {
            g.gemm_mq6g256v2_i8_prequant(&w, xq, &y32, m, k, rows)
                .unwrap();
        });
        let t16 = median_us(&mut gpu, 20, |g| {
            g.gemm_mq6g256v2_i8_prequant(&w, xq, &y16, m, k, rows)
                .unwrap();
        });
        println!("MICROBENCH rows={rows} qkv GEMM f32 out={t32:.1}us bf16 out={t16:.1}us");
        for t in [xt, y32, y16] {
            gpu.free_tensor(t).unwrap();
        }
    }
}
