// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity for `attention_flux_vt_wmma_f16kv_f32` (the V-transposed WMMA flash
//! kernel written for the FLUX.1-dev MMDiT shape).
//!
//! Two independent references, because neither alone is sufficient:
//!
//! 1. **f64 CPU reference** on small shapes. This is the only ground truth —
//!    it validates the online-softmax rescale, the WMMA fragment mappings, the
//!    transposed V staging, and both the `B % 64` and `L % 64` tail paths.
//!    Tolerance is set by the f16 K/V operands: the kernel multiplies f16 K and
//!    f16 V, so ~1e-3 relative on a well-conditioned softmax output is the
//!    floor, not a fudge.
//! All three of `attention_flux_vt_wmma_f16kv_f32`,
//! `attention_flux_vtk_wmma_f16kv_f32` and `attention_flux_v2_wmma_f16kv` go
//! through every check below; `v2` differs in tile geometry (128 query rows per
//! workgroup, 8 waves, one barrier pair per key tile, a `v_perm_b32` V
//! transpose), which is why the case list carries `B % 128` tails the other two
//! do not need.
//!
//! 2. **Cross-check against `attention_dflash_wmma_m64_n32_f16kv_v5_f32`** at
//!    the real FLUX shape (n = 4608, 24 heads, hd 128), which the CPU reference
//!    cannot reach in reasonable time. Both kernels are f16-operand flash
//!    attention over identical inputs, so they must agree to f16 noise.
//!
//! Both kernels are instantiated for all four `{q dtype} x {out dtype}`
//! combinations over {f32, f16}, so the dtype surface gets two more checks
//! that are *exact*, not tolerance-based, and would catch a wrong entry, a
//! wrong stride, or a truncating (rather than RNE) store:
//!
//! 3. **f16 out == RNE(f32 out), bit for bit.** Only the store differs
//!    between the two entries; the arithmetic is the same instruction
//!    sequence, so any mismatch is a real defect, not accumulated noise.
//! 4. **f16 Q == f32 Q**, when the f16 Q buffer holds exactly `RNE(q)`. The
//!    f32 entry already rounds Q to f16 for the WMMA A-fragment, so the two
//!    entries consume identical operands. The `2e-3` bound the plan asks for
//!    is therefore checked as a bound, and reported alongside the (expected
//!    zero) exact-mismatch count.
//!
//! Run (GPU required, gpu-lock it):
//! ```
//! cargo run --release --features lab --example test_attention_flux_vt_parity \
//!   -p rdna-compute
//! ```

use rdna_compute::{DType, Gpu, GpuTensor};

/// f32 -> f16, round-to-nearest-even, **including the subnormal range**.
///
/// The subnormal arm is not pedantry: `v_cvt_f16_f32` produces f16 subnormals,
/// so a host helper that flushed `|x| < 2^-14` to zero would disagree with the
/// GPU on roughly one value in 16k of a uniform [-1, 1) draw — enough to make
/// the exact f16-Q and f16-out checks below fail on a kernel that is correct.
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let biased = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x007f_ffff;
    if biased == 0xff {
        // Inf stays Inf; NaN stays NaN (quiet, payload not preserved).
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let unbiased = biased - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased >= -14 {
        // Normal f16: drop 13 mantissa bits, RNE.
        let mut exp = unbiased + 15;
        let mut m = mant >> 13;
        let rem = mant & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
            m += 1;
            if m == 0x400 {
                m = 0;
                exp += 1;
                if exp >= 0x1f {
                    return sign | 0x7c00;
                }
            }
        }
        return sign | ((exp as u16) << 10) | (m as u16);
    }
    // Subnormal f16: the value is m * 2^-24 for integer m, so shift the
    // 24-bit significand (implicit 1 restored) down to that grid and RNE.
    // An f32 subnormal (`biased == 0`, hence `unbiased == -127`) gives
    // `drop == 126` and returns zero at the guard below — correct, it is far
    // under half an f16 subnormal ulp — so by the time the significand is
    // assembled the implicit 1 is always present.
    let drop = 13 + (-unbiased - 14) as u32;
    if drop > 24 {
        return sign;
    }
    let full = mant | 0x0080_0000;
    let m = full >> drop;
    let rem = full & ((1u32 << drop) - 1);
    let half = 1u32 << (drop - 1);
    let mut r = m;
    if rem > half || (rem == half && (m & 1) == 1) {
        r += 1;
    }
    // r == 0x400 rounds up out of the subnormals into the smallest normal,
    // and `sign | 0x400` is exactly that encoding.
    sign | (r as u16)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut e = -1i32;
        let mut m = mant;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        return f32::from_bits(sign | (((e + 127 - 14) as u32) << 23) | (m << 13));
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

/// Deterministic pseudo-random in [-1, 1).
fn prand(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    ((x >> 40) as f32 / 8_388_608.0) - 1.0
}

fn upload_f32(gpu: &mut Gpu, host: &[f32], shape: Vec<usize>) -> GpuTensor {
    let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = gpu.hip.malloc(bytes.len()).expect("malloc f32");
    gpu.hip.memcpy_htod(&buf, &bytes).expect("htod f32");
    let t = GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(buf.as_ptr(), bytes.len()) },
        shape,
        dtype: DType::F32,
    };
    std::mem::forget(buf);
    t
}

fn upload_f16(gpu: &mut Gpu, host_bits: &[u16], shape: Vec<usize>) -> GpuTensor {
    let bytes: Vec<u8> = host_bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = gpu.hip.malloc(bytes.len()).expect("malloc f16");
    gpu.hip.memcpy_htod(&buf, &bytes).expect("htod f16");
    let t = GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(buf.as_ptr(), bytes.len()) },
        shape,
        dtype: DType::F16,
    };
    std::mem::forget(buf);
    t
}

fn zeros_f32(gpu: &mut Gpu, n: usize, shape: Vec<usize>) -> GpuTensor {
    upload_f32(gpu, &vec![0.0f32; n], shape)
}

fn zeros_f16(gpu: &mut Gpu, n: usize, shape: Vec<usize>) -> GpuTensor {
    upload_f16(gpu, &vec![0u16; n], shape)
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    gpu.hip.memcpy_dtoh(&mut bytes, &t.buf).expect("dtoh");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn download_f16_bits(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u16> {
    let mut bytes = vec![0u8; n * 2];
    gpu.hip.memcpy_dtoh(&mut bytes, &t.buf).expect("dtoh f16");
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Which of the three new kernels to launch, or the arch router over them.
#[derive(Clone, Copy)]
enum Kern {
    Vt,
    Vtk,
    /// The barrier-/gather-reworked third generation: 128 query rows per
    /// workgroup, so it exercises tile geometry the other two never do — its
    /// `B` tail starts at `B % 128`, not `B % 64`, and cases like `B = 100`
    /// leave four of its eight waves entirely past the end of the tensor.
    V2,
    Best,
}

/// Launch one kernel with whatever `{q, out}` dtypes the tensors carry.
#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &mut Gpu,
    kern: Kern,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    out: &GpuTensor,
    b: usize,
    l: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
) {
    let r = match kern {
        Kern::Vt => gpu.attention_flux_vt_wmma_f16kv_f32(q, k, v, out, b, l, nh, nkv, hd),
        Kern::Vtk => gpu.attention_flux_vtk_wmma_f16kv_f32(q, k, v, out, b, l, nh, nkv, hd),
        Kern::V2 => gpu.attention_flux_v2_wmma_f16kv(q, k, v, out, b, l, nh, nkv, hd),
        Kern::Best => gpu.attention_flux_best_f16kv_f32(q, k, v, out, b, l, nh, nkv, hd),
    };
    r.expect("launch");
    gpu.hip.device_synchronize().expect("sync");
}

/// Elements where the f16 store is not the round-to-nearest-even rounding of
/// the value the f32 entry stored. Expected to be exactly 0: the two entries
/// run the same arithmetic and differ only in the epilogue store.
fn rne_mismatches(f16_bits: &[u16], f32_vals: &[f32]) -> usize {
    f16_bits
        .iter()
        .zip(f32_vals.iter())
        .filter(|(h, f)| **h != f32_to_f16_bits(**f))
        .count()
}

/// Elements that differ bit-for-bit between two f32 results.
fn exact_mismatches(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// f64 reference. Q is f32 (as the kernel sees it); K/V are read back through
/// f16 so the reference consumes the exact same operand values.
fn cpu_reference(
    q: &[f32],
    k_bits: &[u16],
    v_bits: &[u16],
    b: usize,
    l: usize,
    n_heads: usize,
    n_kv_heads: usize,
    hd: usize,
) -> Vec<f32> {
    let rep = n_heads / n_kv_heads;
    let q_stride = n_heads * hd;
    let kv_stride = n_kv_heads * hd;
    let scale = 1.0f64 / (hd as f64).sqrt();
    let mut out = vec![0.0f32; b * q_stride];
    for head in 0..n_heads {
        let kvh = head / rep;
        for i in 0..b {
            let qrow = &q[i * q_stride + head * hd..][..hd];
            let mut s = vec![0.0f64; l];
            let mut mx = f64::NEG_INFINITY;
            for (j, sj) in s.iter_mut().enumerate() {
                let kb = j * kv_stride + kvh * hd;
                let mut acc = 0.0f64;
                for d in 0..hd {
                    acc += qrow[d] as f64 * f16_bits_to_f32(k_bits[kb + d]) as f64;
                }
                *sj = acc * scale;
                if *sj > mx {
                    mx = *sj;
                }
            }
            let mut denom = 0.0f64;
            for sj in s.iter_mut() {
                *sj = (*sj - mx).exp();
                denom += *sj;
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            for d in 0..hd {
                let mut acc = 0.0f64;
                for (j, sj) in s.iter().enumerate() {
                    acc += *sj * f16_bits_to_f32(v_bits[j * kv_stride + kvh * hd + d]) as f64;
                }
                out[i * q_stride + head * hd + d] = (acc * inv) as f32;
            }
        }
    }
    out
}

struct Stats {
    max_abs: f32,
    /// `max|a-b| / max|ref|`. Attention outputs are convex combinations of V,
    /// so individual elements cross zero and a per-element relative error is
    /// unbounded there and says nothing; scaling by the tensor's own dynamic
    /// range is the meaningful measure.
    rel_inf: f32,
    /// `||a-b||_2 / ||ref||_2`.
    rel_l2: f32,
}

fn compare(a: &[f32], reference: &[f32]) -> Stats {
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    let mut se = 0.0f64;
    let mut sr = 0.0f64;
    for (x, y) in a.iter().zip(reference.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        if y.abs() > max_ref {
            max_ref = y.abs();
        }
        se += (d as f64) * (d as f64);
        sr += (*y as f64) * (*y as f64);
    }
    Stats {
        max_abs,
        rel_inf: max_abs / max_ref.max(1e-30),
        rel_l2: (se.sqrt() / sr.sqrt().max(1e-30)) as f32,
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch: {}", gpu.arch);
    println!();

    let hd = 128usize;
    // (b, l, n_heads, n_kv_heads, label)
    let cases: &[(usize, usize, usize, usize, &str)] = &[
        (64, 64, 1, 1, "exact single tile"),
        (64, 128, 1, 1, "single q tile, 2 k tiles"),
        (128, 256, 2, 2, "2 q tiles, 4 k tiles, 2 heads"),
        (192, 320, 3, 3, "3 heads, 5 k tiles"),
        (100, 256, 2, 2, "B tail (100 % 64 = 36)"),
        (128, 200, 2, 2, "L tail (200 % 64 = 8)"),
        (37, 91, 2, 2, "both tails, tiny"),
        (256, 256, 4, 2, "GQA rep=2"),
        (64, 4608, 1, 1, "FLUX L, one q tile"),
        // `v2` stages 128 query rows per workgroup, so its tail cases are not
        // the ones above: B = 100 leaves four of its eight waves entirely past
        // the end of the tensor, B = 200 is a 72-row second tile, and B = 129
        // is a second tile with a single live row. All three are exact-tile
        // cases for `vt`/`vtk`, so without them nothing tests v2's tail.
        (200, 256, 2, 2, "B tail vs 128-row tile (200 % 128 = 72)"),
        (129, 192, 2, 2, "128-row tile + 1 live row"),
    ];

    // `vtk` and `v2` are gfx11-wave32-only (the gfx11 WMMA intrinsic hipcc
    // rejects on gfx12), so on a non-gfx11 arch those arms skip — announced
    // below — while the arms that do run (`vt`, the router, the v5
    // cross-check) still execute. `SKIP_VTK` (name kept for the existing
    // invocations) is the manual opt-out of the same arms.
    let gfx11_wmma_w32 = gpu.arch_caps.has_wmma_w32();
    let skip_gfx11_only = std::env::var("SKIP_VTK").is_ok() || !gfx11_wmma_w32;
    if !gfx11_wmma_w32 {
        for k in ["attention_flux_vtk_wmma", "attention_flux_v2_wmma"] {
            println!("skip: {k} is gfx11 wave32 WMMA only (arch={})", gpu.arch);
        }
    }
    let mut failures = 0usize;
    // Every row printed below is one check. Reported at the end so a silently
    // shrinking suite (a `continue` that skips a whole kernel, say) is visible
    // rather than reading as a clean pass.
    let mut checks = 0usize;
    // f16 K and f16 V operands put the achievable floor at ~1e-3 of the
    // tensor's dynamic range; 5e-3 leaves headroom for the f32 accumulation
    // order differing from the reference without hiding a real bug (a wrong
    // fragment mapping or a broken rescale shows up at O(1), not O(1e-3)).
    let tol_rel = 5.0e-3f32;
    // The plan's bound on the f16-Q path against the f32-Q path. Both entries
    // consume the same f16 A-fragment, so the measured figure is 0 and this is
    // a ceiling on a defect, not a noise allowance.
    let tol_qf16 = 2.0e-3f32;

    println!(
        "{:<34} {:>6} {:>6} {:>4} {:>4} {:>11} {:>11} {:>11}  {}",
        "kern case", "B", "L", "H", "KVH", "max_abs", "rel_inf", "rel_l2", "verdict"
    );
    println!("{}", "-".repeat(112));

    for &(b, l, nh, nkv, label) in cases {
        let q_len = b * nh * hd;
        let kv_len = l * nkv * hd;
        let q: Vec<f32> = (0..q_len).map(|i| prand(0x1234, i)).collect();
        let k_bits: Vec<u16> = (0..kv_len)
            .map(|i| f32_to_f16_bits(prand(0x5678, i)))
            .collect();
        let v_bits: Vec<u16> = (0..kv_len)
            .map(|i| f32_to_f16_bits(prand(0x9abc, i)))
            .collect();

        // f16 Q holds exactly RNE(q), which is what the f32 entry's Q staging
        // computes on the fly — so the two entries must agree bit for bit.
        let q16_bits: Vec<u16> = q.iter().map(|x| f32_to_f16_bits(*x)).collect();

        let d_q = upload_f32(&mut gpu, &q, vec![b, nh * hd]);
        let d_q16 = upload_f16(&mut gpu, &q16_bits, vec![b, nh * hd]);
        let d_k = upload_f16(&mut gpu, &k_bits, vec![l, nkv * hd]);
        let d_v = upload_f16(&mut gpu, &v_bits, vec![l, nkv * hd]);
        let d_out = zeros_f32(&mut gpu, q_len, vec![b, nh * hd]);
        let d_out16 = zeros_f16(&mut gpu, q_len, vec![b, nh * hd]);

        let want = cpu_reference(&q, &k_bits, &v_bits, b, l, nh, nkv, hd);

        for (tag, kern) in [("vt ", Kern::Vt), ("vtk", Kern::Vtk), ("v2 ", Kern::V2)] {
            if matches!(kern, Kern::Vtk | Kern::V2) && skip_gfx11_only {
                continue;
            }
            let mut row = |name: String, st: &Stats, ok: bool| {
                checks += 1;
                if !ok {
                    failures += 1;
                }
                println!(
                    "{:<34} {:>6} {:>6} {:>4} {:>4} {:>11.3e} {:>11.3e} {:>11.3e}  {}",
                    name,
                    b,
                    l,
                    nh,
                    nkv,
                    st.max_abs,
                    st.rel_inf,
                    st.rel_l2,
                    if ok { "PASS" } else { "FAIL" }
                );
            };

            // (1) q f32 / out f32 — against the f64 CPU reference.
            launch(&mut gpu, kern, &d_q, &d_k, &d_v, &d_out, b, l, nh, nkv, hd);
            let got = download_f32(&gpu, &d_out, q_len);
            let st = compare(&got, &want);
            let ok = st.rel_inf <= tol_rel && st.rel_l2 <= tol_rel;
            row(format!("{tag} {label}"), &st, ok);

            // (2) q f32 / out f16 — must be RNE of (1), exactly.
            launch(
                &mut gpu, kern, &d_q, &d_k, &d_v, &d_out16, b, l, nh, nkv, hd,
            );
            let got16 = download_f16_bits(&gpu, &d_out16, q_len);
            let got16_f32: Vec<f32> = got16.iter().map(|h| f16_bits_to_f32(*h)).collect();
            let bad = rne_mismatches(&got16, &got);
            let st16 = compare(&got16_f32, &got);
            row(format!("{tag} of16 vs RNE(f32) {label}"), &st16, bad == 0);

            // (3) q f16 / out f32 — must equal (1), exactly.
            launch(
                &mut gpu, kern, &d_q16, &d_k, &d_v, &d_out, b, l, nh, nkv, hd,
            );
            let got_q16 = download_f32(&gpu, &d_out, q_len);
            let st_q16 = compare(&got_q16, &got);
            let bad_q16 = exact_mismatches(&got_q16, &got);
            row(
                format!("{tag} qf16 vs qf32 {label}"),
                &st_q16,
                st_q16.rel_inf <= tol_qf16 && st_q16.rel_l2 <= tol_qf16 && bad_q16 == 0,
            );
            // ...and still inside the CPU-reference tolerance on its own.
            let st_q16_ref = compare(&got_q16, &want);
            row(
                format!("{tag} qf16 vs f64 ref {label}"),
                &st_q16_ref,
                st_q16_ref.rel_inf <= tol_rel && st_q16_ref.rel_l2 <= tol_rel,
            );

            // (4) q f16 / out f16 — must be RNE of (3), exactly.
            launch(
                &mut gpu, kern, &d_q16, &d_k, &d_v, &d_out16, b, l, nh, nkv, hd,
            );
            let got_q16_o16 = download_f16_bits(&gpu, &d_out16, q_len);
            let both16_f32: Vec<f32> = got_q16_o16.iter().map(|h| f16_bits_to_f32(*h)).collect();
            let bad_both = rne_mismatches(&got_q16_o16, &got_q16);
            let st_both = compare(&both16_f32, &got_q16);
            row(
                format!("{tag} qf16+of16 vs RNE {label}"),
                &st_both,
                bad_both == 0,
            );
        }
    }

    // ---- Real FLUX shape: cross-check against the incumbent v5 kernel ----
    println!();
    let (b, l, nh, nkv) = (4608usize, 4608usize, 24usize, 24usize);
    let q_len = b * nh * hd;
    let kv_len = l * nkv * hd;
    let q: Vec<f32> = (0..q_len).map(|i| prand(0xfeed, i)).collect();
    let k_bits: Vec<u16> = (0..kv_len)
        .map(|i| f32_to_f16_bits(prand(0xbeef, i)))
        .collect();
    let v_bits: Vec<u16> = (0..kv_len)
        .map(|i| f32_to_f16_bits(prand(0xcafe, i)))
        .collect();
    let q16_bits: Vec<u16> = q.iter().map(|x| f32_to_f16_bits(*x)).collect();
    let d_q = upload_f32(&mut gpu, &q, vec![b, nh * hd]);
    let d_q16 = upload_f16(&mut gpu, &q16_bits, vec![b, nh * hd]);
    let d_k = upload_f16(&mut gpu, &k_bits, vec![l, nkv * hd]);
    let d_v = upload_f16(&mut gpu, &v_bits, vec![l, nkv * hd]);
    let d_new = zeros_f32(&mut gpu, q_len, vec![b, nh * hd]);
    let d_old = zeros_f32(&mut gpu, q_len, vec![b, nh * hd]);
    let d_new16 = zeros_f16(&mut gpu, q_len, vec![b, nh * hd]);

    gpu.attention_flux_vt_wmma_f16kv_f32(&d_q, &d_k, &d_v, &d_new, b, l, nh, nkv, hd)
        .expect("launch new");
    gpu.attention_dflash_wmma_m64_n32_f16kv_v5_f32(&d_q, &d_k, &d_v, &d_old, b, l, nh, nkv, hd)
        .expect("launch v5");
    gpu.hip.device_synchronize().expect("sync");
    let new = download_f32(&gpu, &d_new, q_len);
    let old = download_f32(&gpu, &d_old, q_len);
    let st = compare(&new, &old);
    // Both are f16-operand flash kernels over identical data with different
    // tile orders, so they agree only to f16 accumulation noise.
    let ok = st.rel_inf <= 1.0e-2 && st.rel_l2 <= 1.0e-2;
    checks += 1;
    if !ok {
        failures += 1;
    }
    println!(
        "{:<34} {:>6} {:>6} {:>4} {:>4} {:>11.3e} {:>11.3e} {:>11.3e}  {}",
        "FLUX shape vs v5 (cross-kernel)",
        b,
        l,
        nh,
        nkv,
        st.max_abs,
        st.rel_inf,
        st.rel_l2,
        if ok { "PASS" } else { "FAIL" }
    );

    // Same check for `v2`, which the CPU reference cannot reach at this shape
    // either. It runs a different tile geometry (128 query rows, 8 waves, one
    // barrier pair per key tile) over the same data, so it is an independent
    // witness rather than a restatement of the row above. Skipped with the
    // other gfx11-only kernels: `v2` would not launch on gfx12.
    if !skip_gfx11_only {
        gpu.attention_flux_v2_wmma_f16kv(&d_q, &d_k, &d_v, &d_new, b, l, nh, nkv, hd)
            .expect("launch v2");
        gpu.hip.device_synchronize().expect("sync");
        let v2 = download_f32(&gpu, &d_new, q_len);
        let st_v2 = compare(&v2, &old);
        let ok_v2 = st_v2.rel_inf <= 1.0e-2 && st_v2.rel_l2 <= 1.0e-2;
        checks += 1;
        if !ok_v2 {
            failures += 1;
        }
        println!(
            "{:<34} {:>6} {:>6} {:>4} {:>4} {:>11.3e} {:>11.3e} {:>11.3e}  {}",
            "FLUX shape v2 vs v5 (cross-kernel)",
            b,
            l,
            nh,
            nkv,
            st_v2.max_abs,
            st_v2.rel_inf,
            st_v2.rel_l2,
            if ok_v2 { "PASS" } else { "FAIL" }
        );
    }

    // ---- Real FLUX shape: the dtype surface, on every route ----
    // This is the shape the plan names (4608 x 24 x 128) and the one the
    // forward pass runs, so the exact checks are made here as well as on the
    // small cases. `Kern::Best` is the router: it must pick the right entry
    // from the tensor dtypes on whatever arch this is.
    let mut dtype_row = |name: &str, st: &Stats, ok: bool| {
        checks += 1;
        if !ok {
            failures += 1;
        }
        println!(
            "{:<34} {:>6} {:>6} {:>4} {:>4} {:>11.3e} {:>11.3e} {:>11.3e}  {}",
            name,
            b,
            l,
            nh,
            nkv,
            st.max_abs,
            st.rel_inf,
            st.rel_l2,
            if ok { "PASS" } else { "FAIL" }
        );
    };
    for (tag, kern) in [
        ("vt ", Kern::Vt),
        ("vtk", Kern::Vtk),
        ("v2 ", Kern::V2),
        ("rtd", Kern::Best),
    ] {
        if matches!(kern, Kern::Vtk | Kern::V2) && skip_gfx11_only {
            continue;
        }
        launch(&mut gpu, kern, &d_q, &d_k, &d_v, &d_new, b, l, nh, nkv, hd);
        let base = download_f32(&gpu, &d_new, q_len);

        launch(
            &mut gpu, kern, &d_q, &d_k, &d_v, &d_new16, b, l, nh, nkv, hd,
        );
        let o16 = download_f16_bits(&gpu, &d_new16, q_len);
        let o16_f32: Vec<f32> = o16.iter().map(|h| f16_bits_to_f32(*h)).collect();
        dtype_row(
            &format!("{tag} FLUX of16 vs RNE(f32)"),
            &compare(&o16_f32, &base),
            rne_mismatches(&o16, &base) == 0,
        );

        launch(
            &mut gpu, kern, &d_q16, &d_k, &d_v, &d_new, b, l, nh, nkv, hd,
        );
        let q16_out = download_f32(&gpu, &d_new, q_len);
        let st_q16 = compare(&q16_out, &base);
        dtype_row(
            &format!("{tag} FLUX qf16 vs qf32"),
            &st_q16,
            st_q16.rel_inf <= tol_qf16
                && st_q16.rel_l2 <= tol_qf16
                && exact_mismatches(&q16_out, &base) == 0,
        );

        launch(
            &mut gpu, kern, &d_q16, &d_k, &d_v, &d_new16, b, l, nh, nkv, hd,
        );
        let both = download_f16_bits(&gpu, &d_new16, q_len);
        let both_f32: Vec<f32> = both.iter().map(|h| f16_bits_to_f32(*h)).collect();
        dtype_row(
            &format!("{tag} FLUX qf16+of16 vs RNE"),
            &compare(&both_f32, &q16_out),
            rne_mismatches(&both, &q16_out) == 0,
        );
    }

    // A dtype pair no entry covers must be a clear error, not a launch that
    // reinterprets the buffer. BF16 is 2 bytes like F16, so a missing check
    // here would silently produce garbage rather than fail.
    {
        let d_bad = GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(d_new16.buf.as_ptr(), q_len * 2) },
            shape: vec![b, nh * hd],
            dtype: DType::BF16,
        };
        let e = gpu.attention_flux_best_f16kv_f32(&d_q, &d_k, &d_v, &d_bad, b, l, nh, nkv, hd);
        let ok = e.is_err();
        checks += 1;
        if !ok {
            failures += 1;
        }
        println!();
        println!(
            "router rejects out=BF16: {}  ({})",
            if ok { "PASS" } else { "FAIL" },
            match &e {
                Err(err) => err.to_string(),
                Ok(()) => "launched anyway".to_string(),
            }
        );
        std::mem::forget(d_bad);
    }

    println!();
    println!("rel_inf = max|new-ref| / max|ref|;  rel_l2 = ||new-ref||_2 / ||ref||_2");
    println!(
        "tolerance: CPU-reference cases rel_inf and rel_l2 <= {tol_rel:.0e}; \
         cross-kernel (vs v5) <= 1e-2; \
         f16-Q vs f32-Q <= {tol_qf16:.0e} AND bit-exact; f16-out RNE-exact"
    );
    if failures == 0 {
        println!("ALL PASS ({checks} checks over {} cases)", cases.len() + 1);
    } else {
        println!("{failures} FAILURE(S)");
        std::process::exit(1);
    }
}
