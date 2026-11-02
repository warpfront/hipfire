// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle of the gfx1151 MQ4G256V2 prefill route (`Gpu::gemm_mq4g256v2_xf16`,
//! `_xf16_regions`, `Gpu::gemm_mq4g256v2_hcw`): the certified PM module
//! `qwen4_mq4_x4_pm_gfx1151` (or, with the PM flag off, its exact hipcc bt4 fallback)
//! against the route it replaced.
//!
//! Weights are random valid MQ4G256V2 (finite normal f16 scale/zero per half group,
//! random nibbles) at the Qwen3.8-Flash-Next trunk shapes (M, K): GDN qkv (10240, 2560),
//! z (6144, 2560), a (48, 2560), b (48, 2560), out (2560, 6144); QSA q (12288, 2560),
//! k (512, 2560), v (512, 2560), indexer (640, 2560), o (2560, 6144); plus ragged-row
//! extras (M = 64, 65, 200). For N in {96, 256, 1000, 2048, 4096} activation rows:
//!
//!  (a) PM plain F32 on the shared F16 rotation vs the replaced production route:
//!      `rotate_x_mq_batched` (F32), zeroed Y, hipcc `gemm_mq4g256v2_residual_wmma_gfx11_bt4`
//!      (what `gemm_qwen4_trunk_mq4_xf16` ran) — bitwise;
//!  (b) BF16 output vs the RNE-BF16 of (a)'s reference F32 — bitwise (PM only);
//!  (c) the a/b/z regions launch vs the three separate projections (PM only);
//!  (d) HC-write fusion (out, o at hidden 2560) vs the reference projection followed by
//!      `tensor_ops::hyper_write` (the unfused write the dispatch op runs), for F32 and
//!      BF16 HC streams, and the F32 mirror of the projection (PM only).
//!
//! Every output is pre-filled with NaN and carries a NaN guard on both sides. A
//! negative control (perturbed weights) must disagree with the reference.
//!
//! `cargo run --release -p rdna-compute --features lab --example qwen4_mq4_x4_pm_oracle`
//! Needs exact gfx1151; exits nonzero on any mismatch (2 on another arch).

use rdna_compute::tensor_ops::{hyper_write, HyperWrite};
use rdna_compute::{DType, Gpu, GpuTensor};

const GUARD: usize = 64;
const NAN_F32: u32 = 0x7fc0_0000;
const NAN_BF16: u16 = 0x7fc0;
const NS: [usize; 5] = [96, 256, 1000, 2048, 4096];
const HIDDEN: usize = 2560;
const BRANCHES: usize = 4;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Round-to-nearest-even f16 bits of a finite value in the normal f16 range.
fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    let mut m10 = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m10 & 1 == 1) {
        m10 += 1;
    }
    if m10 == 0x400 {
        m10 = 0;
        exp += 1;
    }
    assert!((1..=30).contains(&exp), "{v} is not a normal f16");
    sign | ((exp as u16) << 10) | m10 as u16
}

fn random_scale(rng: &mut Rng) -> u16 {
    let mag = 2f32.powi(-9) + rng.unit() * 0.2;
    f16_bits(if rng.next() & 1 == 0 { mag } else { -mag })
}

fn random_zero(rng: &mut Rng) -> u16 {
    let mut v = (rng.unit() - 0.5) * 4.0;
    if v.abs() < 2f32.powi(-12) {
        v = 0.125;
    }
    f16_bits(v)
}

/// Host packer: `[m][k / 256]` groups of 136 bytes: hA (scale s0, zero z0), hB (s1, z1),
/// then 128 bytes of nibbles (element e at byte e / 2, low nibble = even e).
fn pack_random_mq4g256v2(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(m * (k / 256) * 136);
    for _ in 0..m * (k / 256) {
        for _half in 0..2 {
            out.extend_from_slice(&random_scale(rng).to_le_bytes());
            out.extend_from_slice(&random_zero(rng).to_le_bytes());
        }
        for _ in 0..128 {
            out.push((rng.next() & 0xff) as u8);
        }
    }
    out
}

/// RNE BF16 bits of a finite F32 given as bits.
fn bf16_rne(bits: u32) -> u16 {
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// (name, M, K).
const SHAPES: [(&str, usize, usize); 13] = [
    ("gdn.qkv", 10240, 2560),
    ("gdn.z", 6144, 2560),
    ("gdn.a", 48, 2560),
    ("gdn.b", 48, 2560),
    ("gdn.out", 2560, 6144),
    ("qsa.q", 12288, 2560),
    ("qsa.k", 512, 2560),
    ("qsa.v", 512, 2560),
    ("qsa.indexer", 640, 2560),
    ("qsa.o", 2560, 6144),
    ("extra.m64", 64, 2560),
    ("extra.m65", 65, 2560),
    ("extra.m200", 200, 2560),
];

struct Mat {
    name: &'static str,
    m: usize,
    k: usize,
    w: GpuTensor,
}

/// A tensor of `elems` elements of `dtype` holding `bytes`.
fn tensor(gpu: &Gpu, bytes: &[u8], dtype: DType, elems: usize) -> GpuTensor {
    assert_eq!(bytes.len(), elems * dtype.size());
    let mut t = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload");
    t.dtype = dtype;
    t.shape = vec![elems];
    t
}

/// `[GUARD NaN][payload NaN or `init`][GUARD NaN]` F32-typed tensor.
fn guarded_f32(gpu: &Gpu, init: Option<&[u32]>, payload: usize) -> GpuTensor {
    let mut words = vec![NAN_F32; GUARD];
    match init {
        Some(init) => words.extend_from_slice(init),
        None => words.extend(std::iter::repeat_n(NAN_F32, payload)),
    }
    words.extend(std::iter::repeat_n(NAN_F32, GUARD));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    tensor(gpu, &bytes, DType::F32, words.len())
}

fn guarded_bf16(gpu: &Gpu, payload: usize) -> GpuTensor {
    let n = GUARD + payload + GUARD;
    let bytes: Vec<u8> = NAN_BF16.to_le_bytes().repeat(n);
    tensor(gpu, &bytes, DType::BF16, n)
}

fn fetch_u32(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.hip.device_synchronize().expect("sync before download");
    gpu.download_raw_bytes(t)
        .expect("download")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn fetch_u16(gpu: &Gpu, t: &GpuTensor) -> Vec<u16> {
    gpu.hip.device_synchronize().expect("sync before download");
    gpu.download_raw_bytes(t)
        .expect("download")
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn diff<T: PartialEq>(got: &[T], want: &[T]) -> usize {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).filter(|(a, b)| a != b).count()
}

/// `got` (a full guarded buffer) against `[GUARD NaN][expected][GUARD NaN]`.
fn diff_guarded<T: PartialEq + Copy>(got: &[T], expected: &[T], nan: T) -> usize {
    let mut want = vec![nan; GUARD];
    want.extend_from_slice(expected);
    want.extend(std::iter::repeat_n(nan, GUARD));
    diff(got, &want)
}

/// The route the PM entries replace: F32 rotation, zeroed Y, hipcc bt4 on the F16
/// conversion of that rotation (`gemm_qwen4_trunk_mq4_xf16`).
fn reference(
    gpu: &mut Gpu,
    w: &GpuTensor,
    xrot: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<u32> {
    let zeros = vec![0u8; n * m * 4];
    let y = tensor(gpu, &zeros, DType::F32, n * m);
    gpu.gemm_mq4g256v2_residual_wmma_gfx1151_bt(w, xrot, &y, m, k, n, 4)
        .expect("bt4 reference");
    let bits = fetch_u32(gpu, &y);
    gpu.free_tensor(y).expect("free ref");
    bits
}

struct Report {
    rows: Vec<(String, usize, usize)>,
}

impl Report {
    fn add(&mut self, label: &str, bad: usize) {
        if bad != 0 {
            eprintln!("MISMATCH {label}: {bad} words differ");
        }
        match self.rows.iter_mut().find(|(l, _, _)| l == label) {
            Some(r) => {
                r.1 += 1;
                r.2 += bad;
            }
            None => self.rows.push((label.to_string(), 1, bad)),
        }
    }
}

fn random_x(gpu: &mut Gpu, n: usize, k: usize, rng: &mut Rng) -> GpuTensor {
    let x: Vec<f32> = (0..n * k)
        .map(|_| (rng.unit() + rng.unit() - 1.0) * 1.5)
        .collect();
    gpu.upload_f32(&x, &[x.len()]).expect("upload x")
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    println!("qwen4_mq4_x4_pm_oracle: arch={arch}");
    if arch != "gfx1151" {
        eprintln!("qwen4_mq4_x4_pm_oracle: needs exact gfx1151 (got {arch}); nothing was checked");
        std::process::exit(2);
    }
    let pm = gpu.gemm_mq4g256v2_pm_applies(2560, 96);
    println!(
        "qwen4_mq4_x4_pm_oracle: route={}",
        if pm {
            "embedded PM image qwen4_mq4_x4_pm_gfx1151 (bf16out, regions and hcw checked)"
        } else {
            "PM off: hipcc bt4 on the shared F16 x (bf16out, regions and hcw not available)"
        }
    );

    let mut rng = Rng(0x5EED_4A11_u64.wrapping_mul(0xA24B_AED4_963E_E407));
    let mut mats: Vec<Mat> = Vec::new();
    for &(name, m, k) in &SHAPES {
        let bytes = pack_random_mq4g256v2(m, k, &mut rng);
        let w = gpu
            .upload_raw(&bytes, &[bytes.len()])
            .expect("upload weights");
        mats.push(Mat { name, m, k, w });
    }
    let find = |name: &str| mats.iter().position(|m| m.name == name).expect("shape");
    let (zi, ai, bi) = (find("gdn.z"), find("gdn.a"), find("gdn.b"));
    let (outi, oi) = (find("gdn.out"), find("qsa.o"));

    let mut report = Report { rows: Vec::new() };
    let mut cases = 0usize;

    for &n in &NS {
        for k in [2560usize, 6144] {
            let x32 = random_x(&mut gpu, n, k, &mut rng);
            let xrot = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc xrot");
            gpu.rotate_x_mq_batched(&x32, &xrot, k, n)
                .expect("rotate f32");
            let mut refs: Vec<Option<Vec<u32>>> = (0..mats.len()).map(|_| None).collect();

            for (i, mat) in mats.iter().enumerate().filter(|(_, mt)| mt.k == k) {
                let (m, label) = (mat.m, format!("{} M={} K={k}", mat.name, mat.m));
                let want = reference(&mut gpu, &mat.w, &xrot, m, k, n);
                // (a) plain F32 on the shared F16 rotation.
                let xf16 = gpu.rotate_x_mq_batched_f16(&x32, k, n).expect("rotate f16");
                let y = guarded_f32(&gpu, None, n * m);
                let view = y.sub_offset(GUARD, n * m);
                gpu.gemm_mq4g256v2_xf16(&mat.w, &xf16, &view, m, k, n)
                    .expect("xf16 launch");
                report.add(
                    &format!("(a) plain {label}"),
                    diff_guarded(&fetch_u32(&gpu, &y), &want, NAN_F32),
                );
                gpu.free_tensor(y).expect("free y");
                cases += 1;
                // (b) BF16 output.
                if pm {
                    let yb = guarded_bf16(&gpu, n * m);
                    let view = yb.sub_offset(GUARD, n * m);
                    let xf16 = gpu.rotate_x_mq_batched_f16(&x32, k, n).expect("rotate f16");
                    gpu.gemm_mq4g256v2_xf16(&mat.w, &xf16, &view, m, k, n)
                        .expect("bf16out launch");
                    let want_bf16: Vec<u16> = want.iter().map(|&b| bf16_rne(b)).collect();
                    report.add(
                        &format!("(b) bf16out {label}"),
                        diff_guarded(&fetch_u16(&gpu, &yb), &want_bf16, NAN_BF16),
                    );
                    gpu.free_tensor(yb).expect("free yb");
                    cases += 1;
                }
                if [zi, ai, bi, outi, oi].contains(&i) {
                    refs[i] = Some(want);
                }
            }

            // (c) a/b/z regions vs the three projections.
            if k == 2560 && pm {
                let regions_idx = [zi, ai, bi];
                let outs: Vec<GpuTensor> = regions_idx
                    .iter()
                    .map(|&i| guarded_f32(&gpu, None, n * mats[i].m))
                    .collect();
                let views: Vec<GpuTensor> = regions_idx
                    .iter()
                    .zip(&outs)
                    .map(|(&i, y)| y.sub_offset(GUARD, n * mats[i].m))
                    .collect();
                let regions = [
                    (&mats[zi].w, &views[0], mats[zi].m),
                    (&mats[ai].w, &views[1], mats[ai].m),
                    (&mats[bi].w, &views[2], mats[bi].m),
                ];
                let xf16 = gpu.rotate_x_mq_batched_f16(&x32, k, n).expect("rotate f16");
                let took = gpu
                    .gemm_mq4g256v2_xf16_regions(&regions, &xf16, k, n)
                    .expect("regions launch");
                let mut bad = if took { 0 } else { 1 };
                if !took {
                    eprintln!("regions launch declined though the PM module applies");
                }
                for (slot, &i) in regions_idx.iter().enumerate() {
                    let want = refs[i].as_ref().expect("reference");
                    bad += diff_guarded(&fetch_u32(&gpu, &outs[slot]), want, NAN_F32);
                }
                report.add("(c) regions z|a|b K=2560", bad);
                cases += 1;
                for y in outs {
                    gpu.free_tensor(y).expect("free region out");
                }
            }

            // (d) HC-write fusion on the output projections.
            if k == 6144 && pm {
                for &wi in &[outi, oi] {
                    let mat = &mats[wi];
                    assert_eq!(mat.m, HIDDEN);
                    assert!(gpu.gemm_mq4g256v2_hcw_applies(mat.m, k, n));
                    let proj = refs[wi].as_ref().expect("reference");
                    let wide = BRANCHES * HIDDEN;
                    // Gate logits [n][4].
                    let gates_host: Vec<f32> = (0..n * BRANCHES)
                        .map(|_| (rng.unit() - 0.5) * 6.0)
                        .collect();
                    let gates = gpu
                        .upload_f32(&gates_host, &[gates_host.len()])
                        .expect("gates");
                    let mixed_bytes: Vec<u8> = proj.iter().flat_map(|w| w.to_le_bytes()).collect();
                    let mixed = tensor(&gpu, &mixed_bytes, DType::F32, n * HIDDEN);
                    for bf16 in [false, true] {
                        // Streams: F32 values, or BF16 bits in the first half of the buffer.
                        let mut init = vec![NAN_F32; n * wide];
                        if bf16 {
                            for p in 0..n * wide / 2 {
                                let lo =
                                    bf16_rne((((rng.unit() - 0.5) * 8.0).to_bits()) as u32) as u32;
                                let hi =
                                    bf16_rne((((rng.unit() - 0.5) * 8.0).to_bits()) as u32) as u32;
                                init[p] = lo | (hi << 16);
                            }
                        } else {
                            for v in init.iter_mut() {
                                *v = (((rng.unit() - 0.5) * 8.0).to_bits()) & 0xffff_0000;
                            }
                        }
                        let unfused = guarded_f32(&gpu, Some(&init), 0);
                        let fused = guarded_f32(&gpu, Some(&init), 0);
                        let uview = unfused.sub_offset(GUARD, n * wide);
                        let fview = fused.sub_offset(GUARD, n * wide);
                        hyper_write(
                            &mut gpu,
                            &HyperWrite {
                                input: &uview,
                                normalized: &uview,
                                mixed: &mixed,
                                gates: &gates,
                                output: &uview,
                                branches: BRANCHES,
                                hidden: HIDDEN,
                                state_bf16: bf16,
                            },
                        )
                        .expect("unfused hyper_write");
                        let xf16 = gpu.rotate_x_mq_batched_f16(&x32, k, n).expect("rotate f16");
                        gpu.gemm_mq4g256v2_hcw(
                            &mat.w, &xf16, None, mat.m, k, n, &fview, &gates, bf16,
                        )
                        .expect("hcw launch");
                        let (want, got) = (fetch_u32(&gpu, &unfused), fetch_u32(&gpu, &fused));
                        let mut start = vec![NAN_F32; GUARD];
                        start.extend_from_slice(&init);
                        start.extend(std::iter::repeat_n(NAN_F32, GUARD));
                        let mut bad = diff(&got, &want);
                        // The write must have changed the streams (not a vacuous pair).
                        if diff(&want, &start) == 0 {
                            eprintln!("unfused write changed nothing");
                            bad += 1;
                        }
                        report.add(
                            &format!(
                                "(d) hcw {} {} streams",
                                mat.name,
                                if bf16 { "bf16" } else { "f32" }
                            ),
                            bad,
                        );
                        cases += 1;
                        gpu.free_tensor(unfused).expect("free");
                        gpu.free_tensor(fused).expect("free");
                    }
                    // The F32 mirror receives the projection as the unfused route stores it.
                    let mirror = guarded_f32(&gpu, None, n * HIDDEN);
                    let mview = mirror.sub_offset(GUARD, n * HIDDEN);
                    let streams = guarded_f32(&gpu, Some(&vec![0u32; n * wide]), 0);
                    let sview = streams.sub_offset(GUARD, n * wide);
                    let xf16 = gpu.rotate_x_mq_batched_f16(&x32, k, n).expect("rotate f16");
                    gpu.gemm_mq4g256v2_hcw(
                        &mat.w,
                        &xf16,
                        Some(&mview),
                        mat.m,
                        k,
                        n,
                        &sview,
                        &gates,
                        false,
                    )
                    .expect("hcw mirror launch");
                    report.add(
                        &format!("(d) hcw {} mirror", mat.name),
                        diff_guarded(&fetch_u32(&gpu, &mirror), proj, NAN_F32),
                    );
                    cases += 1;
                    for t in [mirror, streams, mixed, gates] {
                        gpu.free_tensor(t).expect("free");
                    }
                }
            }
            gpu.free_tensor(xrot).expect("free xrot");
            gpu.free_tensor(x32).expect("free x");
        }
        println!("qwen4_mq4_x4_pm_oracle: N={n} done");
    }

    // Negative control: perturbed weights must disagree with the unperturbed reference.
    let control_bad = {
        let (m, k, n) = (mats[ai].m, mats[ai].k, 256usize);
        let x32 = random_x(&mut gpu, n, k, &mut rng);
        let xrot = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc xrot");
        gpu.rotate_x_mq_batched(&x32, &xrot, k, n)
            .expect("rotate f32");
        let want = reference(&mut gpu, &mats[ai].w, &xrot, m, k, n);
        let mut bytes = gpu
            .download_raw_bytes(&mats[ai].w)
            .expect("download weights");
        for group in 0..m * (k / 256) {
            // Flip every nibble of the first 16 code bytes of every group.
            for b in 0..16 {
                bytes[group * 136 + 8 + b] ^= 0xff;
            }
        }
        let perturbed = gpu
            .upload_raw(&bytes, &[bytes.len()])
            .expect("upload perturbed");
        let xf16 = gpu.rotate_x_mq_batched_f16(&x32, k, n).expect("rotate f16");
        let y = guarded_f32(&gpu, None, n * m);
        let view = y.sub_offset(GUARD, n * m);
        gpu.gemm_mq4g256v2_xf16(&perturbed, &xf16, &view, m, k, n)
            .expect("control launch");
        let bad = diff_guarded(&fetch_u32(&gpu, &y), &want, NAN_F32);
        for t in [y, perturbed, xrot, x32] {
            gpu.free_tensor(t).expect("free");
        }
        bad
    };
    println!("qwen4_mq4_x4_pm_oracle: negative control (perturbed weights) mismatch words = {control_bad}");
    if control_bad == 0 {
        eprintln!(
            "negative control FAILED: perturbed weights agreed with the reference (vacuous oracle)"
        );
        std::process::exit(1);
    }

    println!("qwen4_mq4_x4_pm_oracle: per-case summary (runs = N in {NS:?})");
    let mut bad_words = 0usize;
    for (label, runs, bad) in &report.rows {
        println!("  {label:<40} runs={runs:<3} mismatch_words={bad}");
        bad_words += bad;
    }
    if bad_words != 0 {
        println!("qwen4_mq4_x4_pm_oracle: FAIL cases={cases} mismatch_words={bad_words}");
        std::process::exit(1);
    }
    println!(
        "qwen4_mq4_x4_pm_oracle: PASS cases={cases} mismatch_bytes=0 pm={}",
        if pm {
            "on"
        } else {
            "off (hipcc bt4 fallback; bf16out/regions/hcw skipped)"
        }
    );
}
