// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab gate: the lane-batched A3B `wo` residual GEMV
//! (`Gpu::gemv_mq4g256v2_residual_lanes`, `y[b] += W . x[b]`, B <= 8) against
//! the REAL singleton launch of the one-row decode
//! (`Gpu::gemv_hfq4g256_residual_mq4v2`, the PM-native twin of
//! `gemv_mq4g256v2_residual` under `kernel.pm_decode`) and against the PM
//! x-batch residual it replaces (`Gpu::gemv_mq4g256v2_residual_xbatch`).
//!
//! `a3b_lane_wo_matches_singleton_and_pm_xbatch_hw`: per lane, raw output bits
//! (NaN payloads included) must equal both references for N in 1..=8 lanes,
//! the A3B shape (m 2048, K 4096) plus row counts that leave a partial
//! workgroup, random + adversarial activations (zeros, +-large,
//! overflow to inf/NaN, denormals, ties, duplicated lanes) and residuals
//! (random, zeros, -0.0, large), random + extreme-header + all-0/all-15 weights
//! in the exact MQ4G256V2 layout. Rows past `lanes` of the output stay
//! untouched; a one-bit change of one lane's activation changes that lane only.
//!
//! `a3b_lane_wo_timing_hw`: warmup + 200 timed samples (hipEvents) of OLD
//! (PM x-batch residual) vs NEW at the production shape with the weights
//! cycled over 40 distinct copies (180 MB, beyond the last-level cache) as the
//! 40 layers of the model do. `LANE_WO_SWEEP=4x4,2x4,...` also times other
//! workgroup geometries (waves = `KS_NSPL` chain splits x rows per workgroup =
//! `KS_R`), checking each is byte-identical first.
//!
//! `#[ignore]`d: needs an exact gfx1201 device, no model files.
//!
//!   cargo test -p rdna-compute --release --features lab --test a3b_lane_wo \
//!       -- --ignored --nocapture --test-threads=1
#![cfg(feature = "lab")]

use rdna_compute::residual_lanes::{
    residual_lanes_source_with, RESIDUAL_LANES_MAX, RESIDUAL_LANES_MODULE,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const POISON: u32 = 0x7FC0_DEAD;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f32_unit(&mut self) -> f32 {
        let a = self.next_u32() & 0x7F_FFFF;
        f32::from_bits(0x3F80_0000 | a) - 1.0
    }
    fn f32_signed(&mut self) -> f32 {
        let v = self.f32_unit();
        if self.next_u32() & 1 == 1 {
            -v
        } else {
            v
        }
    }
    fn f16_normal(&mut self, lo: u32, hi: u32, signed: bool) -> u16 {
        let exp = lo + self.next_u32() % (hi - lo + 1);
        let mant = self.next_u32() & 0x3FF;
        let sign = if signed { self.next_u32() & 1 } else { 0 };
        ((sign << 15) | (exp << 10) | mant) as u16
    }
}

fn f32_bytes(d: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) }
}
fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc u8");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod u8");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc f32");
    gpu.hip
        .memcpy_htod(&t.buf, f32_bytes(data))
        .expect("htod f32");
    t
}
fn put_f32(gpu: &mut Gpu, t: &GpuTensor, off: usize, data: &[f32]) {
    let view = t.sub_offset(off, data.len());
    gpu.hip
        .memcpy_htod(&view.buf, f32_bytes(data))
        .expect("put");
}
fn dl(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}
fn bits(d: &[f32]) -> Vec<u32> {
    d.iter().map(|v| v.to_bits()).collect()
}

#[derive(Clone, Copy, Debug)]
enum Wk {
    /// scale 2^-7..2^-2 style magnitudes, signed zero points
    Normal,
    /// scales spanning 2^-14..2^12 (rounding at every magnitude)
    Wide,
    AllZero,
    AllFifteen,
}

/// MQ4G256V2 rows: `groups` x 136 B `[sc0:f16][zp0:f16][sc1:f16][zp1:f16][128 B]`.
fn synth_qt44(rows: usize, groups: usize, seed: u64, kind: Wk) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut out = vec![0u8; rows * groups * 136];
    for r in 0..rows * groups {
        let off = r * 136;
        for half in 0..2 {
            let (sc, zp) = match kind {
                Wk::Normal | Wk::AllZero | Wk::AllFifteen => {
                    (rng.f16_normal(8, 13, false), rng.f16_normal(7, 13, true))
                }
                Wk::Wide => (rng.f16_normal(1, 26, false), rng.f16_normal(1, 26, true)),
            };
            out[off + 4 * half..off + 4 * half + 2].copy_from_slice(&sc.to_le_bytes());
            out[off + 4 * half + 2..off + 4 * half + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for w in 0..32 {
            let pk = match kind {
                Wk::AllZero => 0,
                Wk::AllFifteen => u32::MAX,
                _ => rng.next_u32() ^ (rng.next_u32() << 16),
            };
            out[off + 8 + 4 * w..off + 12 + 4 * w].copy_from_slice(&pk.to_le_bytes());
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
enum Xk {
    Rand,
    RandBig,
    Zeros,
    /// +-1e30..: products and sums stay finite but are huge
    Large,
    /// reaches inf / NaN inside the dot product
    Overflow,
    Denormal,
    /// every element 1.0 (all partial sums tie)
    Ones,
    /// +1/-1 alternating
    Alternating,
    /// the same vector in every lane
    DupLanes,
    /// mixed zeros, tiny and huge
    Mixed,
}
const XKINDS: [Xk; 10] = [
    Xk::Rand,
    Xk::RandBig,
    Xk::Zeros,
    Xk::Large,
    Xk::Overflow,
    Xk::Denormal,
    Xk::Ones,
    Xk::Alternating,
    Xk::DupLanes,
    Xk::Mixed,
];

fn make_x(kind: Xk, rng: &mut Lcg, lanes: usize, k: usize) -> Vec<Vec<f32>> {
    let one = |rng: &mut Lcg, kind: Xk| -> Vec<f32> {
        (0..k)
            .map(|i| match kind {
                Xk::Rand => rng.f32_signed(),
                Xk::RandBig => rng.f32_signed() * 64.0,
                Xk::Zeros => 0.0,
                Xk::Large => rng.f32_signed() * 1.0e30,
                Xk::Overflow => rng.f32_signed() * 3.0e38,
                Xk::Denormal => {
                    f32::from_bits(rng.next_u32() & 0x007F_FFFF)
                        * if rng.next_u32() & 1 == 1 { -1.0 } else { 1.0 }
                }
                Xk::Ones => 1.0,
                Xk::Alternating => {
                    if i & 1 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                Xk::DupLanes => rng.f32_signed(),
                Xk::Mixed => match rng.next_u32() % 5 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => rng.f32_signed() * 1.0e-30,
                    3 => rng.f32_signed() * 1.0e8,
                    _ => rng.f32_signed(),
                },
            })
            .collect()
    };
    if matches!(kind, Xk::DupLanes) {
        let v = one(rng, kind);
        return (0..lanes).map(|_| v.clone()).collect();
    }
    (0..lanes).map(|_| one(rng, kind)).collect()
}

#[derive(Clone, Copy, Debug)]
enum Yk {
    Rand,
    Zeros,
    NegZeros,
    Large,
}
const YKINDS: [Yk; 4] = [Yk::Rand, Yk::Zeros, Yk::NegZeros, Yk::Large];

fn make_y(kind: Yk, rng: &mut Lcg, lanes: usize, m: usize) -> Vec<f32> {
    (0..lanes * m)
        .map(|_| match kind {
            Yk::Rand => rng.f32_signed() * 4.0,
            Yk::Zeros => 0.0,
            Yk::NegZeros => -0.0,
            Yk::Large => rng.f32_signed() * 1.0e12,
        })
        .collect()
}

struct Probe {
    w: GpuTensor,
    m: usize,
    k: usize,
    /// batched y scratch: `RESIDUAL_LANES_MAX + 1` lanes so lane-count guards
    /// can be checked
    y_new: GpuTensor,
    y_pm: GpuTensor,
    x: GpuTensor,
    /// singleton scratch
    x1: GpuTensor,
    y1: GpuTensor,
}

fn probe(gpu: &mut Gpu, wbytes: &[u8], m: usize, k: usize) -> Probe {
    let w = upload_u8(gpu, wbytes);
    let y_new = upload_f32(
        gpu,
        &vec![f32::from_bits(POISON); (RESIDUAL_LANES_MAX + 1) * m],
    );
    let y_pm = upload_f32(
        gpu,
        &vec![f32::from_bits(POISON); (RESIDUAL_LANES_MAX + 1) * m],
    );
    let x = upload_f32(gpu, &vec![f32::from_bits(POISON); RESIDUAL_LANES_MAX * k]);
    let x1 = upload_f32(gpu, &vec![0.0; k]);
    let y1 = upload_f32(gpu, &vec![0.0; m]);
    Probe {
        w,
        m,
        k,
        y_new,
        y_pm,
        x,
        x1,
        y1,
    }
}

/// The singleton launch on lane `r` of (`xs`, `ys`): the product's launch.
fn singleton(gpu: &mut Gpu, p: &Probe, x: &[f32], y: &[f32]) -> Vec<u32> {
    put_f32(gpu, &p.x1, 0, x);
    put_f32(gpu, &p.y1, 0, y);
    gpu.gemv_hfq4g256_residual_mq4v2(&p.w, &p.x1, &p.y1, p.m, p.k)
        .expect("singleton residual");
    gpu.hip.device_synchronize().expect("sync");
    dl(gpu, &p.y1, p.m)
}

fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// One full comparison; returns mismatched words.
fn compare_case(
    gpu: &mut Gpu,
    p: &Probe,
    tag: &str,
    xs: &[Vec<f32>],
    y0: &[f32],
    variant: Option<(&str, usize, usize)>,
) -> usize {
    let (m, k, lanes) = (p.m, p.k, xs.len());
    let flat_x: Vec<f32> = xs.iter().flatten().copied().collect();
    let mut bad = 0usize;

    // reference: singleton per lane
    let want: Vec<Vec<u32>> = (0..lanes)
        .map(|r| singleton(gpu, p, &xs[r], &y0[r * m..(r + 1) * m]))
        .collect();

    // PM x-batch
    put_f32(gpu, &p.x, 0, &flat_x);
    let mut ypm = y0.to_vec();
    ypm.resize((RESIDUAL_LANES_MAX + 1) * m, f32::from_bits(POISON));
    put_f32(gpu, &p.y_pm, 0, &ypm);
    gpu.gemv_mq4g256v2_residual_xbatch(&p.w, &p.x, &p.y_pm, m, k, lanes)
        .expect("pm residual xbatch");
    gpu.hip.device_synchronize().expect("sync");
    let got_pm = dl(gpu, &p.y_pm, (RESIDUAL_LANES_MAX + 1) * m);

    // NEW
    put_f32(gpu, &p.y_new, 0, &ypm);
    match variant {
        None => gpu
            .gemv_mq4g256v2_residual_lanes(&p.w, &p.x, &p.y_new, m, k, lanes)
            .expect("lanes residual"),
        Some((module, w, r)) => gpu
            .launch_residual_lanes(module, w, r, &p.w, &p.x, &p.y_new, m, k, lanes)
            .expect("lanes residual variant"),
    }
    gpu.hip.device_synchronize().expect("sync");
    let got = dl(gpu, &p.y_new, (RESIDUAL_LANES_MAX + 1) * m);

    for r in 0..lanes {
        let g = &got[r * m..(r + 1) * m];
        if let Some(i) = first_diff(g, &want[r]) {
            bad += g.iter().zip(&want[r]).filter(|(a, b)| a != b).count();
            eprintln!(
                "MISMATCH {tag} lane {r}/{lanes}: new != singleton at row {i}: new {:#010x} singleton {:#010x} (y0 {:#010x}, {} words differ)",
                g[i],
                want[r][i],
                y0[r * m + i].to_bits(),
                g.iter().zip(&want[r]).filter(|(a, b)| a != b).count()
            );
        }
        let gp = &got_pm[r * m..(r + 1) * m];
        if let Some(i) = first_diff(g, gp) {
            bad += g.iter().zip(gp).filter(|(a, b)| a != b).count();
            eprintln!(
                "MISMATCH {tag} lane {r}/{lanes}: new != pm_xbatch at row {i}: new {:#010x} pm {:#010x}",
                g[i], gp[i]
            );
        }
        if let Some(i) = first_diff(gp, &want[r]) {
            // The PM reference itself disagrees with the singleton: report, count.
            bad += 1;
            eprintln!(
                "MISMATCH {tag} lane {r}/{lanes}: pm_xbatch != singleton at row {i} (reference problem)"
            );
        }
    }
    // rows past `lanes` untouched
    for (i, &v) in got[lanes * m..].iter().enumerate() {
        if v != POISON {
            bad += 1;
            eprintln!("MISMATCH {tag}: guard word {i} past lane {lanes} written ({v:#010x})");
            break;
        }
    }
    bad
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn a3b_lane_wo_matches_singleton_and_pm_xbatch_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    assert!(
        gpu.arch_caps.is_gfx1201(),
        "this gate requires an exact gfx1201 device (got {})",
        gpu.arch
    );
    assert!(
        gpu.flags.pm_decode,
        "kernel.pm_decode is off: the singleton/x-batch references would not be the PM objects"
    );
    // (m, k): the A3B wo shape first, then row counts that leave partial
    // wave / workgroup tiles (K is fixed at 4096 by the kernel).
    let shapes: [(usize, usize); 6] = [
        (2048, 4096),
        (37, 4096),
        (1, 4096),
        (2050, 4096),
        (333, 4096),
        (96, 4096),
    ];
    let lane_counts = [1usize, 2, 3, 4, 5, 6, 7, 8];
    let variants = load_variants(&mut gpu);
    let only_variants = std::env::var("LANE_WO_ONLY_VARIANTS").is_ok();
    let mut total_bad = 0usize;
    let mut compared = 0usize;
    for (si, &(m, k)) in shapes.iter().enumerate() {
        for (wi, wk) in [Wk::Normal, Wk::Wide, Wk::AllZero, Wk::AllFifteen]
            .into_iter()
            .enumerate()
        {
            // the full matrix of activation x residual kinds on the production
            // shape; a rotating subset elsewhere
            let wbytes = synth_qt44(m, k / 256, 0xA3B0 + (si * 8 + wi) as u64, wk);
            let p = probe(&mut gpu, &wbytes, m, k);
            for (xi, &xk) in XKINDS.iter().enumerate() {
                for (yi, &yk) in YKINDS.iter().enumerate() {
                    if si != 0 && (xi + yi + wi) % 3 != 0 {
                        continue;
                    }
                    for &lanes in &lane_counts {
                        if si != 0 && lanes % 2 == 0 && lanes != 8 {
                            continue;
                        }
                        let mut rng = Lcg(0x1234_5678
                            ^ ((si as u64) << 40)
                            ^ ((wi as u64) << 32)
                            ^ ((xi as u64) << 24)
                            ^ ((yi as u64) << 16)
                            ^ lanes as u64);
                        let xs = make_x(xk, &mut rng, lanes, k);
                        let y0 = make_y(yk, &mut rng, lanes, m);
                        let tag = format!("m{m} k{k} w{wk:?} x{xk:?} y{yk:?} n{lanes}");
                        if !only_variants {
                            total_bad += compare_case(&mut gpu, &p, &tag, &xs, &y0, None);
                            compared += 1;
                        }
                        for (name, w, r) in &variants {
                            let tag = format!("{tag} variant W{w} R{r}");
                            total_bad += compare_case(
                                &mut gpu,
                                &p,
                                &tag,
                                &xs,
                                &y0,
                                Some((name.as_str(), *w, *r)),
                            );
                            compared += 1;
                        }
                    }
                }
            }
        }
    }

    // ---- negative control: one input bit of one lane changes that lane only
    {
        let (m, k, lanes) = (2048usize, 4096usize, 5usize);
        let wbytes = synth_qt44(m, k / 256, 0xBEEF, Wk::Normal);
        let p = probe(&mut gpu, &wbytes, m, k);
        let mut rng = Lcg(0xC0FFEE);
        let xs = make_x(Xk::Rand, &mut rng, lanes, k);
        let y0 = make_y(Yk::Rand, &mut rng, lanes, m);
        let run = |gpu: &mut Gpu, xs: &[Vec<f32>]| -> Vec<u32> {
            let flat: Vec<f32> = xs.iter().flatten().copied().collect();
            put_f32(gpu, &p.x, 0, &flat);
            put_f32(gpu, &p.y_new, 0, &y0);
            gpu.gemv_mq4g256v2_residual_lanes(&p.w, &p.x, &p.y_new, m, k, lanes)
                .unwrap();
            gpu.hip.device_synchronize().unwrap();
            dl(gpu, &p.y_new, lanes * m)
        };
        let base = run(&mut gpu, &xs);
        let mut pert = xs.clone();
        pert[2][1234] = f32::from_bits(pert[2][1234].to_bits() ^ (1 << 20));
        let got = run(&mut gpu, &pert);
        let want2 = singleton(&mut gpu, &p, &pert[2], &y0[2 * m..3 * m]);
        let mut changed_other = 0usize;
        for r in 0..lanes {
            let (g, b) = (&got[r * m..(r + 1) * m], &base[r * m..(r + 1) * m]);
            let n = g.iter().zip(b).filter(|(a, c)| a != c).count();
            if r == 2 {
                assert!(n > 0, "a one-bit activation change left lane 2 unchanged");
                assert_eq!(first_diff(g, &want2), None, "perturbed lane != singleton");
                eprintln!("negative control: lane 2 differs from baseline in {n} rows");
            } else {
                changed_other += n;
            }
        }
        assert_eq!(changed_other, 0, "the perturbation bled into other lanes");
    }

    // ---- 200 back-to-back launches stay identical (no race between launches)
    {
        let (m, k, lanes) = (2048usize, 4096usize, 8usize);
        let wbytes = synth_qt44(m, k / 256, 0xFACE, Wk::Normal);
        let p = probe(&mut gpu, &wbytes, m, k);
        let mut rng = Lcg(0xD00D);
        let xs = make_x(Xk::Rand, &mut rng, lanes, k);
        let y0 = make_y(Yk::Rand, &mut rng, lanes, m);
        let flat: Vec<f32> = xs.iter().flatten().copied().collect();
        put_f32(&mut gpu, &p.x, 0, &flat);
        let want: Vec<Vec<u32>> = (0..lanes)
            .map(|r| singleton(&mut gpu, &p, &xs[r], &y0[r * m..(r + 1) * m]))
            .collect();
        for rep in 0..200 {
            put_f32(&mut gpu, &p.y_new, 0, &y0);
            gpu.gemv_mq4g256v2_residual_lanes(&p.w, &p.x, &p.y_new, m, k, lanes)
                .unwrap();
            if rep % 20 == 19 || rep == 0 {
                gpu.hip.device_synchronize().unwrap();
                let got = dl(&gpu, &p.y_new, lanes * m);
                for r in 0..lanes {
                    assert_eq!(
                        first_diff(&got[r * m..(r + 1) * m], &want[r]),
                        None,
                        "repeat {rep} lane {r}"
                    );
                }
            }
            gpu.hip.device_synchronize().unwrap();
        }
    }

    eprintln!("compared {compared} cases; total mismatched words = {total_bad}");
    assert_eq!(
        total_bad, 0,
        "lane-batched residual is not byte-identical to the singleton / PM x-batch"
    );
    let _ = bits(&[]);
}

/// Module source of one sweep variant. `LANE_WO_SRC=<file>` substitutes the
/// kernel file (policy preamble + file) so a kernel edit needs no rebuild.
fn variant_source(entry: &str, w: usize, r: usize) -> String {
    match std::env::var("LANE_WO_SRC") {
        Ok(path) => {
            let inc = std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../kernels/src/gfx12_weight_cache_policy.inc"
            ))
            .expect("policy inc");
            let body = std::fs::read_to_string(&path).expect("LANE_WO_SRC file");
            format!(
                "#define HIPFIRE_RESIDUAL_LANES_KERNEL {entry}\n#define KS_NSPL {w}\n#define KS_R {r}\n#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n{inc}{body}"
            )
        }
        Err(_) => residual_lanes_source_with(entry, w, r),
    }
}

/// `LANE_WO_SWEEP=4x4,2x4,...`: compiled sweep variants (name, waves, rows per WG).
fn load_variants(gpu: &mut Gpu) -> Vec<(String, usize, usize)> {
    let mut variants = vec![];
    if let Ok(spec) = std::env::var("LANE_WO_SWEEP") {
        for (i, item) in spec.split(',').filter(|s| !s.is_empty()).enumerate() {
            let (w, r) = item.split_once('x').expect("WxR");
            let (w, r): (usize, usize) = (w.parse().unwrap(), r.parse().unwrap());
            let name = format!("residual_lanes_sweep_{i}");
            gpu.ensure_kernel_public(&name, &variant_source(&name, w, r), &name)
                .expect("variant compile");
            variants.push((name, w, r));
        }
    }
    variants
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "needs an exact gfx1201 GPU; timing run (exclusive card)"]
fn a3b_lane_wo_timing_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    assert!(gpu.arch_caps.is_gfx1201(), "exact gfx1201 required");
    assert!(gpu.flags.pm_decode, "kernel.pm_decode must be on");
    const COPIES: usize = 40;
    const BATCH: usize = 40; // launches per timed sample = one pass over the copies
    const SAMPLES: usize = 200;
    let (m, k) = (2048usize, 4096usize);
    let wbytes = synth_qt44(m, k / 256, 0x71CE, Wk::Normal);
    let ws: Vec<GpuTensor> = (0..COPIES).map(|_| upload_u8(&mut gpu, &wbytes)).collect();
    let mut rng = Lcg(0x7E57);
    let xs: Vec<f32> = (0..RESIDUAL_LANES_MAX * k)
        .map(|_| rng.f32_signed())
        .collect();
    let x = upload_f32(&mut gpu, &xs);
    let y_old = upload_f32(&mut gpu, &vec![0.0; RESIDUAL_LANES_MAX * m]);
    let y_new = upload_f32(&mut gpu, &vec![0.0; RESIDUAL_LANES_MAX * m]);

    let variants = load_variants(&mut gpu);

    // `LANE_WO_FLUSH=1`: a 2 x 96 MB device copy after every launch (evicts L2, the
    // last-level cache, TLB entries and instruction caches, as the other ~25 kernels
    // between two wo launches of a decode step do).
    let flush = std::env::var("LANE_WO_FLUSH").is_ok();
    let fa = gpu.alloc_tensor(&[96 << 18], DType::F32).expect("flush a");
    let fb = gpu.alloc_tensor(&[96 << 18], DType::F32).expect("flush b");
    let bytes_w = (m * (k / 256) * 136) as f64;
    for &lanes in &[8usize, 7, 6, 5, 4, 2, 1] {
        let bytes = bytes_w + (lanes * k * 4) as f64 + (lanes * m * 8) as f64;
        let time = |gpu: &mut Gpu, f: &mut dyn FnMut(&mut Gpu, usize)| -> Vec<f64> {
            let start = gpu.hip.event_create().unwrap();
            let stop = gpu.hip.event_create().unwrap();
            for i in 0..BATCH {
                f(gpu, i % COPIES);
                if flush {
                    gpu.hip.memcpy_dtod(&fb.buf, &fa.buf, 96 << 20).unwrap();
                }
            }
            gpu.hip.device_synchronize().unwrap();
            let mut us = Vec::with_capacity(SAMPLES);
            for _ in 0..SAMPLES {
                gpu.hip.event_record(&start, None).unwrap();
                for i in 0..BATCH {
                    f(gpu, i % COPIES);
                    if flush {
                        gpu.hip.memcpy_dtod(&fb.buf, &fa.buf, 96 << 20).unwrap();
                    }
                }
                gpu.hip.event_record(&stop, None).unwrap();
                gpu.hip.event_synchronize(&stop).unwrap();
                us.push(
                    gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1000.0 / BATCH as f64,
                );
            }
            gpu.hip.event_destroy(start).unwrap();
            gpu.hip.event_destroy(stop).unwrap();
            us
        };
        let report = |name: &str, mut us: Vec<f64>| {
            let min = us.iter().cloned().fold(f64::INFINITY, f64::min);
            let med = median(&mut us);
            eprintln!(
                "TIMING m{m} k{k} N={lanes} {name:<28} median {med:7.2} us ({:6.1} GB/s)  min {min:7.2} us ({:6.1} GB/s)",
                bytes / med / 1e3,
                bytes / min / 1e3
            );
        };
        let us = time(&mut gpu, &mut |g, i| {
            g.gemv_mq4g256v2_residual_xbatch(&ws[i], &x, &y_old, m, k, lanes)
                .unwrap()
        });
        report("OLD pm_xbatch residual", us);
        let us = time(&mut gpu, &mut |g, i| {
            g.gemv_mq4g256v2_residual_lanes(&ws[i], &x, &y_new, m, k, lanes)
                .unwrap()
        });
        report(&format!("NEW {RESIDUAL_LANES_MODULE:.12}.."), us);
        for (name, w, r) in &variants {
            let us = time(&mut gpu, &mut |g, i| {
                g.launch_residual_lanes(name, *w, *r, &ws[i], &x, &y_new, m, k, lanes)
                    .unwrap()
            });
            report(&format!("variant W{w} R{r}"), us);
        }
    }
}
