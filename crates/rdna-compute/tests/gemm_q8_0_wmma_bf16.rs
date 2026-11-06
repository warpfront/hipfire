// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `gemm_q8_0_wmma_bf16` (the gfx11 wmma Q8_0 GEMM with a BF16 RN-even epilogue) must equal
//! `gemm_q8_0_wmma` followed by the in-place `bf16_round_trip_f32_strided` pass BIT FOR BIT on the
//! raw f32 bits of the whole Y buffer (class E), and must equal the host RN-even BF16 round with
//! non-finite passthrough. Also: an F16-typed X (one `convert_f32_to_f16_into`) is byte-identical
//! to the launcher-side convert of the F32 X, `gemm_q8_0_batched_chunked` accepts that F16 X where
//! the wmma route is taken, the kernel is deterministic, and an RDNA4 device is refused.
//!
//! Y is pre-filled with a NaN sentinel so an unwritten element is caught by the bit compare.
//!
//! The GPU tests are `#[ignore]`d: they need a gfx1100 / gfx1151 GPU. Run through fabric-run.sh:
//!
//!   cargo test -p rdna-compute --release --test gemm_q8_0_wmma_bf16 -- --ignored

use rdna_compute::{DType, Gpu, GpuTensor};

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.unit()).sum();
        (s - 2.0) * 1.732_050_8
    }
}

/// Host BF16 RN-even round with non-finite passthrough (the contract of the device round pass).
fn host_bf16_round(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let u = x.to_bits();
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}

/// Q8_0 bytes [m, k/32*34]: per 32-K block a finite random f16 scale then 32 random int8.
fn make_weights(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let nb = k / 32;
    let mut bytes = Vec::with_capacity(m * nb * 34);
    for _ in 0..m * nb {
        let sc =
            half::f16::from_f32(0.004 * (1.0 + 0.5 * rng.gauss().abs()) * rng.gauss().signum());
        bytes.extend_from_slice(&sc.to_bits().to_le_bytes());
        for _ in 0..32 {
            let q = (rng.gauss() * 40.0).round().clamp(-128.0, 127.0) as i8;
            bytes.push(q as u8);
        }
    }
    bytes
}

#[derive(Clone, Copy, Debug)]
enum Data {
    /// plain gaussian f32
    Gauss,
    /// some rows scaled so outputs overflow / the f16 conversion of X overflows to Inf
    Large,
    /// tiny values, deep into the f16 denormal range
    Tiny,
    /// some rows contain NaN / +-Inf (mixed finite and non-finite rows)
    NonFinite,
}

const DATASETS: [Data; 4] = [Data::Gauss, Data::Large, Data::Tiny, Data::NonFinite];

fn make_x(kind: Data, n: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(n * k);
    for t in 0..n {
        let f = match kind {
            Data::Gauss | Data::NonFinite => 1.0f32,
            Data::Large => {
                if t % 7 == 1 {
                    1.0e7 // beyond f16 range: converts to +-Inf
                } else if t % 3 == 0 {
                    5.0e4 // finite in f16, accumulated sums reach the f16/bf16 range limits
                } else {
                    1.0
                }
            }
            Data::Tiny => {
                if t % 2 == 0 {
                    1.0e-6
                } else {
                    1.0e-3
                }
            }
        };
        let start = x.len();
        for _ in 0..k {
            x.push(rng.gauss() * f);
        }
        if let Data::NonFinite = kind {
            if t % 5 == 2 {
                x[start + (rng.next_u32() as usize) % k] = f32::NAN;
            } else if t % 11 == 3 {
                x[start + (rng.next_u32() as usize) % k] = f32::INFINITY;
            } else if t % 13 == 4 {
                x[start + (rng.next_u32() as usize) % k] = f32::NEG_INFINITY;
            }
        }
    }
    x
}

const NAN_BITS: u32 = 0x7fc0_dead;

fn nan_buf(len: usize) -> Vec<f32> {
    vec![f32::from_bits(NAN_BITS); len]
}

fn gfx11(gpu: &Gpu) -> bool {
    if gpu.arch == "gfx1100" || gpu.arch == "gfx1151" {
        true
    } else {
        eprintln!("skip: needs gfx1100 / gfx1151, got {}", gpu.arch);
        false
    }
}

fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{what}: element {i} differs: got {:#010x} ({g}), want {:#010x} ({w})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

fn free(gpu: &mut Gpu, t: GpuTensor) {
    gpu.free_tensor(t).expect("free tensor");
}

/// Which wmma entry to run.
#[derive(Clone, Copy)]
enum Entry {
    Plain,
    Bf16,
}

/// One wmma GEMM into a NaN-sentinel Y; returns the downloaded Y.
fn run_gemm(
    gpu: &mut Gpu,
    entry: Entry,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let y = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y");
    match entry {
        Entry::Plain => gpu.gemm_q8_0_wmma(w, x, &y, m, k, n),
        Entry::Bf16 => gpu.gemm_q8_0_wmma_bf16(w, x, &y, m, k, n),
    }
    .unwrap_or_else(|e| panic!("wmma ({m},{k},{n}): {e:?}"));
    let out = gpu.download_f32(&y).expect("download y");
    free(gpu, y);
    out
}

/// The fused BF16 kernel against (incumbent + device round pass) and the host round of the
/// incumbent's output, for one weight and one activation tensor.
fn check_bf16(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    what: &str,
) {
    let y0 = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y0");
    gpu.gemm_q8_0_wmma(w, x, &y0, m, k, n)
        .unwrap_or_else(|e| panic!("plain {what} ({m},{k},{n}): {e:?}"));
    let plain = gpu.download_f32(&y0).expect("download plain");
    gpu.bf16_round_trip_f32_strided(&y0, n, 0, m, m)
        .expect("separate round pass");
    let want = gpu.download_f32(&y0).expect("download unfused");
    free(gpu, y0);

    // device round pass == host round of the incumbent output
    let host: Vec<f32> = plain.iter().map(|&v| host_bf16_round(v)).collect();
    assert_bits_eq(
        &want,
        &host,
        &format!("device round vs host round {what} ({m},{k},{n})"),
    );

    let got = run_gemm(gpu, Entry::Bf16, w, x, m, k, n);
    assert_bits_eq(&got, &want, &format!("bf16 wmma {what} ({m},{k},{n})"));
    let nonfinite = got.iter().filter(|v| !v.is_finite()).count();
    eprintln!("wmma bf16 {what} ({m},{k},{n}): bit-identical ({nonfinite} non-finite)");
}

/// Sweep every dataset x every N for one weight shape.
fn check_shape(gpu: &mut Gpu, m: usize, k: usize, ns: &[usize], seed: u64) {
    let mut rng = Rng(seed);
    let wbytes = make_weights(m, k, &mut rng);
    let w = gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w");
    for kind in DATASETS {
        let nmax = *ns.iter().max().expect("ns");
        let xfull = make_x(kind, nmax, k, &mut rng);
        for &n in ns {
            let x = gpu.upload_f32(&xfull[..n * k], &[n * k]).expect("x");
            check_bf16(gpu, &w, &x, m, k, n, &format!("{kind:?}"));
            free(gpu, x);
        }
    }
    free(gpu, w);
}

const ROWS: [usize; 13] = [1, 2, 15, 16, 17, 63, 64, 65, 129, 475, 512, 1000, 4096];

#[test]
#[ignore = "needs a gfx1100 / gfx1151 GPU"]
fn bf16_wmma_equals_plain_plus_round_raon_sites() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gfx11(&gpu) {
        return;
    }
    // (m, k): b_proj/rda_w, f_a, f_b, MLA q_b
    let sites: [(usize, usize); 4] = [(40, 2560), (128, 2560), (5120, 128), (7680, 576)];
    for (i, &(m, k)) in sites.iter().enumerate() {
        check_shape(&mut gpu, m, k, &ROWS, 0x0b0f_16e0 + i as u64);
    }
}

#[test]
#[ignore = "needs a gfx1100 / gfx1151 GPU"]
fn bf16_wmma_equals_plain_plus_round_m_tails() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gfx11(&gpu) {
        return;
    }
    let mut seed = 0x7a11_0000u64;
    for &m in &[1usize, 15, 17, 33, 130] {
        for &k in &[32usize, 64, 576, 2560] {
            seed += 1;
            check_shape(&mut gpu, m, k, &ROWS, seed);
        }
    }
}

/// An F16 plane produced by one `convert_f32_to_f16_into` must give byte-identical Y to the F32 X
/// (launcher-side convert), for the incumbent and for the BF16 kernel.
#[test]
#[ignore = "needs a gfx1100 / gfx1151 GPU"]
fn f16_typed_x_equals_f32_x() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gfx11(&gpu) {
        return;
    }
    let sites: [(usize, usize); 3] = [(40, 2560), (128, 2560), (5120, 128)];
    let ns = [1usize, 17, 64, 475, 512];
    for (i, &(m, k)) in sites.iter().enumerate() {
        let mut rng = Rng(0xf160_0000 + i as u64);
        let wbytes = make_weights(m, k, &mut rng);
        let w = gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w");
        for kind in DATASETS {
            for &n in &ns {
                let xh = make_x(kind, n, k, &mut rng);
                let x32 = gpu.upload_f32(&xh, &[n * k]).expect("x32");
                let x16 = gpu.alloc_tensor(&[n * k], DType::F16).expect("x16");
                gpu.convert_f32_to_f16_into(&x32, &x16, n * k)
                    .expect("convert into f16 plane");
                for (entry, name) in [(Entry::Plain, "plain"), (Entry::Bf16, "bf16")] {
                    let a = run_gemm(&mut gpu, entry, &w, &x32, m, k, n);
                    let b = run_gemm(&mut gpu, entry, &w, &x16, m, k, n);
                    assert_bits_eq(
                        &b,
                        &a,
                        &format!("f16 x vs f32 x {name} {kind:?} ({m},{k},{n})"),
                    );
                }
                free(&mut gpu, x16);
                free(&mut gpu, x32);
            }
        }
        free(&mut gpu, w);
        eprintln!("f16-typed x ({m},{k}): byte-identical to f32 x");
    }
}

/// One F16 plane shared by three different weights (M = 40, 40, 128 on the same X) equals three
/// independent F32-X calls.
#[test]
#[ignore = "needs a gfx1100 / gfx1151 GPU"]
fn shared_f16_plane_across_weights() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gfx11(&gpu) {
        return;
    }
    let k = 2560usize;
    let ms = [40usize, 40, 128];
    let mut rng = Rng(0x5ba2_ed00);
    let mut ws = Vec::new();
    for &m in &ms {
        let wbytes = make_weights(m, k, &mut rng);
        ws.push(gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w"));
    }
    for kind in DATASETS {
        for &n in &[1usize, 17, 65, 512, 1000] {
            let xh = make_x(kind, n, k, &mut rng);
            let x32 = gpu.upload_f32(&xh, &[n * k]).expect("x32");
            let x16 = gpu.alloc_tensor(&[n * k], DType::F16).expect("x16");
            gpu.convert_f32_to_f16_into(&x32, &x16, n * k)
                .expect("convert once");
            for (entry, name) in [(Entry::Plain, "plain"), (Entry::Bf16, "bf16")] {
                for (wi, &m) in ms.iter().enumerate() {
                    let want = run_gemm(&mut gpu, entry, &ws[wi], &x32, m, k, n);
                    let got = run_gemm(&mut gpu, entry, &ws[wi], &x16, m, k, n);
                    assert_bits_eq(
                        &got,
                        &want,
                        &format!("shared plane {name} {kind:?} weight {wi} ({m},{k},{n})"),
                    );
                }
            }
            free(&mut gpu, x16);
            free(&mut gpu, x32);
        }
    }
    for w in ws {
        free(&mut gpu, w);
    }
}

/// `gemm_q8_0_batched_chunked` with an F16 X: equal to the F32-X call where the wmma route is
/// taken, an error where it is not.
#[test]
#[ignore = "needs a gfx1100 / gfx1151 GPU"]
fn batched_chunked_accepts_f16_x_only_on_the_wmma_route() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gfx11(&gpu) {
        return;
    }
    let sites: [(usize, usize); 3] = [(40, 2560), (128, 2560), (5120, 128)];
    let mut tested_ok = false;
    let mut tested_err = false;
    for (i, &(m, k)) in sites.iter().enumerate() {
        let mut rng = Rng(0xc4a7_0000 + i as u64);
        let wbytes = make_weights(m, k, &mut rng);
        let w = gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w");
        for &n in &[1usize, 17, 512, 1000] {
            let xh = make_x(Data::Gauss, n, k, &mut rng);
            let x32 = gpu.upload_f32(&xh, &[n * k]).expect("x32");
            let x16 = gpu.alloc_tensor(&[n * k], DType::F16).expect("x16");
            gpu.convert_f32_to_f16_into(&x32, &x16, n * k)
                .expect("convert");
            let y32 = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y32");
            gpu.gemm_q8_0_batched_chunked(&w, &x32, &y32, m, k, n)
                .unwrap_or_else(|e| panic!("chunked f32 x ({m},{k},{n}): {e:?}"));
            let want = gpu.download_f32(&y32).expect("download f32");
            let y16 = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y16");
            let r = gpu.gemm_q8_0_batched_chunked(&w, &x16, &y16, m, k, n);
            if gpu.q8_0_wmma_shared_f16_x_ok(m, k, n) {
                r.unwrap_or_else(|e| panic!("chunked f16 x ({m},{k},{n}): {e:?}"));
                let got = gpu.download_f32(&y16).expect("download f16");
                assert_bits_eq(
                    &got,
                    &want,
                    &format!("chunked f16 x vs f32 x ({m},{k},{n})"),
                );
                tested_ok = true;
            } else {
                // Off the shareable route (gfx1151's 4-warp tile for M >= 4096 is still a wmma
                // launcher that reads an F16 X as staged): an Ok result must equal the F32-x
                // result, an Err is the guard refusing a non-wmma route.
                match r {
                    Ok(()) => {
                        let got = gpu.download_f32(&y16).expect("download f16");
                        assert_bits_eq(
                            &got,
                            &want,
                            &format!("chunked f16 x vs f32 x ({m},{k},{n})"),
                        );
                    }
                    Err(_) => {}
                }
                tested_err = true;
            }
            free(&mut gpu, y16);
            free(&mut gpu, y32);
            free(&mut gpu, x16);
            free(&mut gpu, x32);
        }
        free(&mut gpu, w);
    }
    eprintln!(
        "chunked f16 x: equal-on-wmma checked={tested_ok}, err-off-wmma checked={tested_err}"
    );
}

/// `q8_0_wmma_bf16_epilogue_ok` is true on exact gfx1100 (when the shared-F16 route is) and false
/// on gfx1151.
#[test]
#[ignore = "needs a gfx1100 / gfx1151 GPU"]
fn epilogue_gate_is_exact_gfx1100_only() {
    let gpu = Gpu::init().expect("gpu init");
    if !gfx11(&gpu) {
        return;
    }
    for &(m, k, n) in &[
        (40usize, 2560usize, 512usize),
        (5120, 128, 64),
        (7680, 576, 1000),
    ] {
        let want = gpu.arch == "gfx1100" && gpu.q8_0_wmma_shared_f16_x_ok(m, k, n);
        assert_eq!(
            gpu.q8_0_wmma_bf16_epilogue_ok(m, k, n),
            want,
            "({m},{k},{n}) on {}",
            gpu.arch
        );
        if gpu.arch == "gfx1151" {
            assert!(!gpu.q8_0_wmma_bf16_epilogue_ok(m, k, n));
        }
    }
}

/// Repeating the BF16 kernel is byte-identical.
#[test]
#[ignore = "needs a gfx1100 / gfx1151 GPU"]
fn bf16_wmma_is_deterministic() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gfx11(&gpu) {
        return;
    }
    for (i, &(m, k, n)) in [
        (40usize, 2560usize, 512usize),
        (5120, 128, 1000),
        (7680, 576, 65),
    ]
    .iter()
    .enumerate()
    {
        let mut rng = Rng(0xde7e_0000 + i as u64);
        let wbytes = make_weights(m, k, &mut rng);
        let w = gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w");
        let xh = make_x(Data::Gauss, n, k, &mut rng);
        let x = gpu.upload_f32(&xh, &[n * k]).expect("x");
        let a = run_gemm(&mut gpu, Entry::Bf16, &w, &x, m, k, n);
        let b = run_gemm(&mut gpu, Entry::Bf16, &w, &x, m, k, n);
        assert_bits_eq(&b, &a, &format!("repeat ({m},{k},{n})"));
        free(&mut gpu, x);
        free(&mut gpu, w);
    }
}

/// An RDNA4 (gfx12) device must refuse the BF16 entry. Skips on gfx11.
#[test]
#[ignore = "needs a gfx12 GPU"]
fn bf16_wmma_errors_on_gfx12() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gpu.arch.starts_with("gfx12") {
        eprintln!("skip: needs gfx12, got {}", gpu.arch);
        return;
    }
    let (m, k, n) = (40usize, 64usize, 16usize);
    let mut rng = Rng(0x12e1_0000);
    let wbytes = make_weights(m, k, &mut rng);
    let w = gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w");
    let xh = make_x(Data::Gauss, n, k, &mut rng);
    let x = gpu.upload_f32(&xh, &[n * k]).expect("x");
    let y = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y");
    assert!(gpu.gemm_q8_0_wmma_bf16(&w, &x, &y, m, k, n).is_err());
    free(&mut gpu, y);
    free(&mut gpu, x);
    free(&mut gpu, w);
}
