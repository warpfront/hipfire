// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! ParoQ4G64 (#651) kernel parity against CPU oracles: `gemv_hfq4g64`,
//! `gemm_hfq4g64` (batch tails 1/5/8/13 around the 8-row tile) and the
//! 64-channel `givens_rotate_to` (seq 1 and 6, around the 4-row CTA). Shapes:
//! K=960 (SmolLM2-360M hidden, 15 groups: odd, so the second half-warp idles
//! on the last group) and K=2560 (its FFN width, 40 groups).
//!
//! Negative controls: the same G64 bytes through `gemv_hfq4g128` and the same
//! sidecar through the 128-channel Givens kernel must miss the oracle.
//!
//! `#[ignore]`d: needs a wave32 GPU and a working HIP toolchain.
//!
//!   cargo test -p rdna-compute --release --test paro_g64_parity -- --ignored

use half::f16;
use rdna_compute::{DType, Gpu, GpuTensor};

/// f32 accumulation-order noise is ~1e-6 relative on these sizes; layout or
/// rotation-block mistakes are O(1).
const TOL: f32 = 1e-4;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFFFF) as f32 / 32768.0 - 1.0
    }
}

fn upload_bytes(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[data.len()], DType::Raw).expect("alloc");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = gpu.alloc_tensor(&[data.len()], DType::F32).expect("alloc");
    gpu.hip.memcpy_htod(&t.buf, &bytes).expect("htod");
    t
}
fn download_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    gpu.hip.memcpy_dtoh(&mut bytes, &t.buf).expect("dtoh");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// HFQ4-G64 rows: per group `[f32 scale][f32 zero][32 B nibbles]`, low nibble
/// = even element. Returns the packed bytes and the `[m][k]` dequantized oracle.
fn synth_g64(m: usize, k: usize, rng: &mut Rng) -> (Vec<u8>, Vec<f32>) {
    let groups = k / 64;
    let mut packed = Vec::with_capacity(m * groups * 40);
    let mut dense = vec![0f32; m * k];
    for row in 0..m {
        for g in 0..groups {
            let scale = 0.01 + (rng.next() & 0x3F) as f32 * 1e-3;
            let zero = -8.0 * scale + rng.unit() * 0.01;
            packed.extend(scale.to_le_bytes());
            packed.extend(zero.to_le_bytes());
            for b in 0..32 {
                let byte = rng.next() as u8;
                packed.push(byte);
                let col = row * k + g * 64 + 2 * b;
                dense[col] = scale * (byte & 0xF) as f32 + zero;
                dense[col + 1] = scale * (byte >> 4) as f32 + zero;
            }
        }
    }
    (packed, dense)
}

fn matvec(dense: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    (0..m)
        .map(|r| {
            (0..k)
                .map(|c| dense[r * k + c] as f64 * x[c] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
    got.iter()
        .zip(want)
        .fold(0f32, |a, (g, w)| a.max((g - w).abs()))
        / scale
}

/// ParoQuant sidecar for `k` channels in `group`-wide blocks: per rotation a
/// random perfect matching inside each block (`pairs`, block-local i16),
/// angles `theta[krot][k/2]` and `channel_scales[k]`, both f16.
struct Sidecar {
    pairs: Vec<i16>,
    theta: Vec<f16>,
    scales: Vec<f16>,
}

fn synth_sidecar(k: usize, group: usize, krot: usize, rng: &mut Rng) -> Sidecar {
    let mut pairs = Vec::with_capacity(krot * k);
    for _ in 0..krot {
        for _ in 0..k / group {
            let mut perm: Vec<i16> = (0..group as i16).collect();
            for i in (1..group).rev() {
                perm.swap(i, rng.next() as usize % (i + 1));
            }
            pairs.extend(perm);
        }
    }
    Sidecar {
        pairs,
        theta: (0..krot * k / 2)
            .map(|_| f16::from_f32(rng.unit() * 3.0))
            .collect(),
        scales: (0..k)
            .map(|_| f16::from_f32(0.5 + rng.unit().abs()))
            .collect(),
    }
}

/// CPU Givens oracle: scale channels, then per rotation apply each block's
/// disjoint `(pairs[2t], pairs[2t+1])` rotation by `theta[g*group/2 + t]`.
fn givens_ref(x: &[f32], sc: &Sidecar, k: usize, group: usize, krot: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(k) {
        let mut v: Vec<f32> = row
            .iter()
            .zip(&sc.scales)
            .map(|(a, s)| a * s.to_f32())
            .collect();
        for rot in 0..krot {
            for g in 0..k / group {
                let base = g * group;
                for t in 0..group / 2 {
                    let i = base + sc.pairs[rot * k + base + 2 * t] as usize;
                    let j = base + sc.pairs[rot * k + base + 2 * t + 1] as usize;
                    let a = sc.theta[rot * (k / 2) + g * group / 2 + t].to_f32();
                    let (s, c) = a.sin_cos();
                    let (xi, xj) = (v[i], v[j]);
                    v[i] = xi * c + xj * s;
                    v[j] = xj * c - xi * s;
                }
            }
        }
        out.extend(v);
    }
    out
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn gpu_or_skip() -> Option<Gpu> {
    let gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return None;
        }
    };
    if !gpu.arch_caps.is_wave32() {
        eprintln!("SKIP — requires an RDNA wave32 device.");
        return None;
    }
    Some(gpu)
}

#[test]
#[ignore]
fn hfq4g64_gemv_and_gemm_match_oracle() {
    let Some(mut gpu) = gpu_or_skip() else { return };
    let mut rng = Rng(0x651);
    for (m, k) in [(40usize, 960usize), (24, 2560)] {
        let (packed, dense) = synth_g64(m, k, &mut rng);
        assert_eq!(packed.len(), m * DType::ParoQ4G64.row_bytes(k).unwrap());
        let w = upload_bytes(&mut gpu, &packed);

        let x: Vec<f32> = (0..k).map(|_| rng.unit()).collect();
        let want = matvec(&dense, &x, m, k);
        let xd = upload_f32(&mut gpu, &x);
        let y = upload_f32(&mut gpu, &vec![0f32; m]);
        gpu.gemv_hfq4g64(&w, &xd, &y, m, k).expect("gemv_hfq4g64");
        let err = max_rel_err(&download_f32(&gpu, &y, m), &want);
        eprintln!("gemv_hfq4g64 m={m} k={k} max_rel_err={err:.2e}");
        assert!(err < TOL, "gemv_hfq4g64 k={k}: {err}");

        // Negative control: the 72-byte G128 reader over G64 bytes.
        gpu.gemv_hfq4g128(&w, &xd, &y, m, k).expect("gemv_hfq4g128");
        let wrong = max_rel_err(&download_f32(&gpu, &y, m), &want);
        eprintln!("negative gemv_hfq4g128 on G64 bytes k={k} max_rel_err={wrong:.2e}");
        assert!(
            wrong > 0.1,
            "G128 kernel must not reproduce G64 output: {wrong}"
        );

        for n in [1usize, 5, 8, 13] {
            let xb: Vec<f32> = (0..n * k).map(|_| rng.unit()).collect();
            let want: Vec<f32> = xb
                .chunks_exact(k)
                .flat_map(|row| matvec(&dense, row, m, k))
                .collect();
            let xbd = upload_f32(&mut gpu, &xb);
            let yb = upload_f32(&mut gpu, &vec![0f32; n * m]);
            gpu.gemm_hfq4g64(&w, &xbd, &yb, m, k, n)
                .expect("gemm_hfq4g64");
            let err = max_rel_err(&download_f32(&gpu, &yb, n * m), &want);
            eprintln!("gemm_hfq4g64 m={m} k={k} n={n} max_rel_err={err:.2e}");
            assert!(err < TOL, "gemm_hfq4g64 k={k} n={n}: {err}");
        }
    }
}

#[test]
#[ignore]
fn givens_g64_matches_oracle() {
    let Some(mut gpu) = gpu_or_skip() else { return };
    let mut rng = Rng(0x64);
    let krot = 8;
    for k in [960usize, 2560] {
        let sc = synth_sidecar(k, 64, krot, &mut rng);
        let pairs = upload_bytes(&mut gpu, as_bytes(&sc.pairs));
        let theta = upload_bytes(&mut gpu, as_bytes(&sc.theta));
        let scales = upload_bytes(&mut gpu, as_bytes(&sc.scales));
        for seq in [1usize, 6] {
            let x: Vec<f32> = (0..seq * k).map(|_| rng.unit()).collect();
            let want = givens_ref(&x, &sc, k, 64, krot);
            let xd = upload_f32(&mut gpu, &x);
            let out = upload_f32(&mut gpu, &vec![0f32; seq * k]);
            gpu.givens_rotate_to(&xd, &out, &pairs, &theta, &scales, seq, k, krot, 64)
                .expect("givens g64");
            let err = max_rel_err(&download_f32(&gpu, &out, seq * k), &want);
            eprintln!("givens_rotate_to g64 k={k} seq={seq} max_rel_err={err:.2e}");
            assert!(err < TOL, "givens g64 k={k} seq={seq}: {err}");
        }
        if k % 128 == 0 {
            // Negative control: the 128-channel kernel over a 64-block sidecar.
            let x: Vec<f32> = (0..k).map(|_| rng.unit()).collect();
            let want = givens_ref(&x, &sc, k, 64, krot);
            let xd = upload_f32(&mut gpu, &x);
            let out = upload_f32(&mut gpu, &vec![0f32; k]);
            gpu.givens_rotate_to(&xd, &out, &pairs, &theta, &scales, 1, k, krot, 128)
                .expect("givens g128");
            let wrong = max_rel_err(&download_f32(&gpu, &out, k), &want);
            eprintln!("negative givens g128 on G64 sidecar k={k} max_rel_err={wrong:.2e}");
            assert!(wrong > 0.1, "G128 rotation must not reproduce G64: {wrong}");
        }
    }
    let x = upload_f32(&mut gpu, &[0f32; 96]);
    assert!(
        gpu.givens_rotate_to(&x, &x, &x, &x, &x, 1, 96, 1, 96)
            .is_err(),
        "an unsupported rotation width must be refused"
    );
}
