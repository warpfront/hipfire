// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab gate: the lane-batched multi-weight projection GEMV
//! (`Gpu::gemv_mq4g256v2_lane_proj`) against the REAL singleton launches of the
//! one-row A3B decode and against the existing `gemv_mq4g256v2_xbatch_pm` loop
//! it replaces, at the three production projection groups (K = 2048):
//!
//! - DeltaNetMoe QKVZA: wqkv 8192 + wz 4096 + w_beta 32 + w_alpha 32
//!   (singleton `fused_qkvza_mq4g256v2`),
//! - FullAttnMoe QKV: wq 8192 + wk 512 + wv 512 (singleton `fused_qkv_mq4g256v2`),
//! - MoE quartet: router 256 + selector 1 + shared gate 512 + shared up 512
//!   (singleton 1281-row `fused_qkvza_mq4g256v2`).
//!
//! Every output word of every lane (N in {1,2,3,5,8}) must equal the singleton
//! launch on that lane's column alone AND the x-batch GEMV's, on random,
//! zero, +-large, denormal-range, tied and outlier activations, with MQ4G256V2
//! weights in the real layout (sane scale/zero values at two magnitudes, and
//! fully random header bytes compared as raw bytes). Also: a chain test through
//! the real norm paths (batched norm -> lane_proj vs the singleton norm-fold
//! launch), poison guards (rows beyond `lanes` untouched), and a negative
//! control (one flipped activation bit changes exactly that lane's column).
//!
//! `#[ignore]`d: needs an exact gfx1201 device, no model files.
//!
//!   cargo test -p rdna-compute --release --features lab --test lane_proj \
//!       -- --ignored --nocapture --test-threads=1
//!
//! Timing (`--class timing`): `lane_proj_timing_hw` (OLD x-batch loop vs NEW,
//! median / min over >= 20 x 200 back-to-back launches cycling weight sets that
//! exceed the last-level cache); `LANE_PROJ_SWEEP=r:w:wg,...` also times
//! tuning variants.
#![cfg(feature = "lab")]

use rdna_compute::lane_proj::{LaneProjSegment, LaneProjVariant};
use rdna_compute::norm_fold::{norm_fold_grid, NormFoldArgs};
use rdna_compute::{DType, Gpu, GpuTensor};

const HIDDEN: usize = 2048;
const MAXL: usize = 8;
const POISON: u32 = 0x7FC0_DEAD;
const EPS: f32 = 1e-6;

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
fn poison(gpu: &mut Gpu, t: &GpuTensor, n: usize) {
    let v = vec![POISON; n];
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("poison");
}
fn poisoned(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).expect("alloc poison");
    poison(gpu, &t, n);
    t
}
fn dl(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WKind {
    /// Scales ~2^-9..2^-7: dequantised values ~1e-2.
    Small,
    /// Scales ~2^-1..2^1.
    Large,
    /// Fully random header bytes (inf / nan / denormal scales and zeros).
    Wild,
}

/// MQ4G256V2 rows: 8 groups x 136 B `[sc0:f16][zp0:f16][sc1:f16][zp1:f16][128 nibble bytes]`.
fn synth_qt44(rows: usize, seed: u64, kind: WKind) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let groups = 8;
    let mut out = vec![0u8; rows * groups * 136];
    for r in 0..rows * groups {
        let off = r * 136;
        for half in 0..2 {
            let (sc, zp) = match kind {
                WKind::Small => (rng.f16_normal(6, 8, false), rng.f16_normal(5, 8, true)),
                WKind::Large => (rng.f16_normal(13, 16, false), rng.f16_normal(12, 16, true)),
                WKind::Wild => (
                    (rng.next_u32() & 0xFFFF) as u16,
                    (rng.next_u32() & 0xFFFF) as u16,
                ),
            };
            out[off + 4 * half..off + 4 * half + 2].copy_from_slice(&sc.to_le_bytes());
            out[off + 4 * half + 2..off + 4 * half + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for w in 0..32 {
            let pk = rng.next_u32() ^ (rng.next_u32() << 16);
            out[off + 8 + 4 * w..off + 12 + 4 * w].copy_from_slice(&pk.to_le_bytes());
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Group {
    Dn,
    Fa,
    Moe,
}

impl Group {
    fn name(self) -> &'static str {
        match self {
            Group::Dn => "dn",
            Group::Fa => "fa",
            Group::Moe => "moe",
        }
    }
    fn rows(self) -> Vec<usize> {
        match self {
            Group::Dn => vec![8192, 4096, 32, 32],
            Group::Fa => vec![8192, 512, 512],
            Group::Moe => vec![256, 1, 512, 512],
        }
    }
}

struct Weights {
    group: Group,
    rows: Vec<usize>,
    w: Vec<GpuTensor>,
}

fn build_weights(gpu: &mut Gpu, group: Group, seed: u64, kind: WKind) -> Weights {
    let rows = group.rows();
    let w = rows
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let bytes = synth_qt44(m, seed ^ ((i as u64 + 1) << 32), kind);
            upload_u8(gpu, &bytes)
        })
        .collect();
    Weights { group, rows, w }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum XKind {
    Random,
    Zeros,
    Large,
    Denormal,
    Ties,
    Outliers,
    /// +-inf and nan elements (invalid-operation paths).
    NonFinite,
}

fn make_x(kind: XKind, rng: &mut Lcg) -> Vec<f32> {
    (0..HIDDEN)
        .map(|i| match kind {
            XKind::Random => rng.f32_signed() * 2.0,
            XKind::Zeros => 0.0,
            XKind::Large => {
                if rng.next_u32() & 1 == 1 {
                    1.0e30
                } else {
                    -1.0e30 * rng.f32_unit()
                }
            }
            XKind::Denormal => match rng.next_u32() % 3 {
                0 => f32::from_bits(rng.next_u32() & 0x007F_FFFF),
                1 => rng.f32_signed() * 1.0e-38,
                _ => -f32::from_bits(rng.next_u32() & 0x007F_FFFF),
            },
            XKind::Ties => [1.0f32, -1.0, 0.5, -0.5][i % 4],
            XKind::Outliers => {
                if rng.next_u32() % 64 == 0 {
                    rng.f32_signed() * 1.0e4
                } else {
                    rng.f32_signed()
                }
            }
            XKind::NonFinite => match rng.next_u32() % 97 {
                0 => f32::INFINITY,
                1 => f32::NEG_INFINITY,
                2 => f32::NAN,
                3 => -0.0,
                _ => rng.f32_signed(),
            },
        })
        .collect()
}

/// Real singleton launch of the one-row decode projection group on `x` (one
/// column); returns each segment's output words.
fn singleton(gpu: &mut Gpu, w: &Weights, x: &GpuTensor) -> Vec<Vec<u32>> {
    let ys: Vec<GpuTensor> = w.rows.iter().map(|&m| poisoned(gpu, m)).collect();
    match w.group {
        Group::Dn | Group::Moe => gpu
            .fused_qkvza_hfq4g256_mq4v2(
                &w.w[0], &w.w[1], &w.w[2], &w.w[3], x, &ys[0], &ys[1], &ys[2], &ys[3], w.rows[0],
                w.rows[1], w.rows[2], w.rows[3], HIDDEN,
            )
            .expect("singleton qkvza"),
        Group::Fa => gpu
            .fused_qkv_hfq4g256_mq4v2(
                &w.w[0], &w.w[1], &w.w[2], x, &ys[0], &ys[1], &ys[2], w.rows[0], w.rows[1],
                w.rows[2], HIDDEN,
            )
            .expect("singleton qkv"),
    }
    w.rows
        .iter()
        .zip(&ys)
        .map(|(&m, y)| dl(gpu, y, m))
        .collect()
}

/// The pm_xbatch loop the lane kernel replaces.
fn old_loop(gpu: &mut Gpu, w: &Weights, x: &GpuTensor, ys: &[GpuTensor], n: usize) {
    for (i, &m) in w.rows.iter().enumerate() {
        gpu.gemv_mq4g256v2_xbatch_pm(&w.w[i], x, &ys[i], m, HIDDEN, n)
            .expect("xbatch");
    }
}

fn new_launch(
    gpu: &mut Gpu,
    w: &Weights,
    x: &GpuTensor,
    ys: &[GpuTensor],
    n: usize,
    v: LaneProjVariant,
) {
    let segs: Vec<LaneProjSegment> = (0..w.rows.len())
        .map(|i| LaneProjSegment {
            w: &w.w[i],
            rows: w.rows[i],
            y: &ys[i],
        })
        .collect();
    gpu.gemv_mq4g256v2_lane_proj_variant(&segs, x, n, v)
        .expect("lane_proj");
}

/// NaN payload / sign differences between two NaN results (inf - inf, 0 * inf
/// or an input NaN reaching the sum by a different commutation of the final
/// adds) are tallied but not counted as mismatches: every non-NaN word must be
/// bit-equal, and a NaN must meet a NaN.
static NAN_PAYLOAD_DIFFS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn is_nan_bits(v: u32) -> bool {
    (v & 0x7F80_0000) == 0x7F80_0000 && (v & 0x007F_FFFF) != 0
}

fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter().zip(b).position(|(&x, &y)| {
        if x == y {
            return false;
        }
        if is_nan_bits(x) && is_nan_bits(y) {
            NAN_PAYLOAD_DIFFS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        true
    })
}

struct Rig {
    xs_lane: GpuTensor,
    x_batch: GpuTensor,
}

fn alloc_rig(gpu: &mut Gpu) -> Rig {
    Rig {
        xs_lane: poisoned(gpu, HIDDEN),
        x_batch: poisoned(gpu, MAXL * HIDDEN),
    }
}

/// One identity case: `n` lanes of `kinds[lane]` activations.
fn check_case(
    gpu: &mut Gpu,
    rig: &Rig,
    w: &Weights,
    n: usize,
    kinds: &[XKind],
    seed: u64,
    tag: &str,
) -> usize {
    let mut rng = Lcg(seed);
    let xs: Vec<Vec<f32>> = (0..n)
        .map(|b| make_x(kinds[b % kinds.len()], &mut rng))
        .collect();
    for (b, x) in xs.iter().enumerate() {
        gpu.hip
            .memcpy_htod(
                &rig.x_batch.sub_offset(b * HIDDEN, HIDDEN).buf,
                f32_bytes(x),
            )
            .expect("put x");
    }
    // The rest of the staging buffer is poison: a kernel reading a lane >= n
    // would show up as poison-dependent output only when the lane count is wrong.
    let ys_new: Vec<GpuTensor> = w.rows.iter().map(|&m| poisoned(gpu, MAXL * m)).collect();
    let ys_old: Vec<GpuTensor> = w.rows.iter().map(|&m| poisoned(gpu, MAXL * m)).collect();
    new_launch(gpu, w, &rig.x_batch, &ys_new, n, test_variant());
    old_loop(gpu, w, &rig.x_batch, &ys_old, n);
    let mut bad = 0usize;
    let mut new_dump: Vec<Vec<u32>> = Vec::new();
    let mut old_dump: Vec<Vec<u32>> = Vec::new();
    for (i, &m) in w.rows.iter().enumerate() {
        new_dump.push(dl(gpu, &ys_new[i], MAXL * m));
        old_dump.push(dl(gpu, &ys_old[i], MAXL * m));
    }
    for b in 0..n {
        gpu.hip
            .memcpy_htod(&rig.xs_lane.buf, f32_bytes(&xs[b]))
            .expect("lane x");
        let want = singleton(gpu, w, &rig.xs_lane);
        for (i, &m) in w.rows.iter().enumerate() {
            let got = &new_dump[i][b * m..(b + 1) * m];
            if let Some(p) = first_diff(got, &want[i]) {
                bad += 1;
                eprintln!(
                    "[{tag}] MISMATCH vs singleton: seg {i} lane {b} word {p}: got {:08x} want {:08x}",
                    got[p], want[i][p]
                );
            }
            let old = &old_dump[i][b * m..(b + 1) * m];
            if let Some(p) = first_diff(got, old) {
                bad += 1;
                eprintln!(
                    "[{tag}] MISMATCH vs xbatch_pm: seg {i} lane {b} word {p}: got {:08x} want {:08x}",
                    got[p], old[p]
                );
            }
            if got.iter().any(|&v| v == POISON) {
                bad += 1;
                eprintln!("[{tag}] poison left in seg {i} lane {b}");
            }
        }
    }
    // rows beyond `n` untouched
    for (i, &m) in w.rows.iter().enumerate() {
        if new_dump[i][n * m..].iter().any(|&v| v != POISON) {
            bad += 1;
            eprintln!("[{tag}] lane rows >= {n} of seg {i} were written");
        }
    }
    bad
}

/// The variant under test: the product unless `LANE_PROJ_TEST_VARIANT=r:w:wg`.
fn test_variant() -> LaneProjVariant {
    match std::env::var("LANE_PROJ_TEST_VARIANT") {
        Ok(s) => {
            let p: Vec<usize> = s.split(':').map(|t| t.parse().unwrap()).collect();
            LaneProjVariant {
                rows_per_wave: p[0],
                waves: p[1],
                wgs_per_wgp: p[2],
            }
        }
        Err(_) => LaneProjVariant::PRODUCT,
    }
}

fn require_gfx1201(gpu: &Gpu) {
    assert!(
        gpu.arch_caps.is_gfx1201(),
        "this gate requires an exact gfx1201 device (got {})",
        gpu.arch
    );
    assert!(
        gpu.flags.pm_decode,
        "kernel.pm_decode is off: the x-batch GEMV would run the hipcc fallback, not the PM object"
    );
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn lane_proj_matches_singleton_and_xbatch_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    require_gfx1201(&gpu);
    gpu.ensure_mq_signs().expect("signs");
    let rig = alloc_rig(&mut gpu);
    let mut bad = 0usize;
    let mut cases = 0usize;
    let kinds_all = [
        XKind::Random,
        XKind::Zeros,
        XKind::Large,
        XKind::Denormal,
        XKind::Ties,
        XKind::Outliers,
        XKind::NonFinite,
    ];
    for group in [Group::Dn, Group::Fa, Group::Moe] {
        for (wi, wkind) in [WKind::Small, WKind::Large, WKind::Wild]
            .into_iter()
            .enumerate()
        {
            let w = build_weights(
                &mut gpu,
                group,
                0x1000 + wi as u64 * 17 + group as u64,
                wkind,
            );
            for &n in &[1usize, 2, 3, 5, 8] {
                // each activation kind on all lanes
                for (ki, &k) in kinds_all.iter().enumerate() {
                    let tag = format!("{} {:?} n{n} {:?}", group.name(), wkind, k);
                    bad += check_case(
                        &mut gpu,
                        &rig,
                        &w,
                        n,
                        &[k],
                        0xA11CE ^ (n as u64) << 8 ^ ki as u64,
                        &tag,
                    );
                    cases += 1;
                }
                // every kind mixed across lanes
                let tag = format!("{} {:?} n{n} mixed", group.name(), wkind);
                bad += check_case(&mut gpu, &rig, &w, n, &kinds_all, 0xB0B ^ n as u64, &tag);
                cases += 1;
            }
        }
    }
    eprintln!(
        "lane_proj identity: {cases} cases, {bad} mismatches, {} NaN-vs-NaN payload/sign differences",
        NAN_PAYLOAD_DIFFS.load(std::sync::atomic::Ordering::Relaxed)
    );
    assert_eq!(bad, 0, "lane_proj is not byte-identical");
}

/// The real norm paths: pre-norm lanes -> batched norm -> lane_proj, against
/// the singleton norm-fold launch per lane.
#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn lane_proj_chain_matches_norm_fold_singleton_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    require_gfx1201(&gpu);
    gpu.ensure_mq_signs().expect("signs");
    let mut rng = Lcg(0xFEED);
    let norm_w: Vec<f32> = (0..HIDDEN).map(|_| 0.5 + rng.f32_unit()).collect();
    let norm_w = upload_f32(&mut gpu, &norm_w);
    let x_batch = poisoned(&mut gpu, MAXL * HIDDEN);
    let x_rot_batch = poisoned(&mut gpu, MAXL * HIDDEN);
    let x1 = poisoned(&mut gpu, HIDDEN);
    let x1_rot = poisoned(&mut gpu, HIDDEN);
    let mut bad = 0usize;
    for group in [Group::Dn, Group::Fa, Group::Moe] {
        let w = build_weights(&mut gpu, group, 0x77 + group as u64, WKind::Small);
        for &n in &[1usize, 3, 8] {
            for seed in 0..4u64 {
                let mut r = Lcg(0xD00D ^ seed << 12 ^ n as u64);
                let xs: Vec<Vec<f32>> = (0..n)
                    .map(|_| {
                        (0..HIDDEN)
                            .map(|_| r.f32_signed() * (0.5 + seed as f32 * 2.0))
                            .collect()
                    })
                    .collect();
                for (b, x) in xs.iter().enumerate() {
                    gpu.hip
                        .memcpy_htod(&x_batch.sub_offset(b * HIDDEN, HIDDEN).buf, f32_bytes(x))
                        .unwrap();
                }
                gpu.fused_rmsnorm_rotate_mq_batched(
                    &x_batch,
                    &norm_w,
                    &x_rot_batch,
                    HIDDEN,
                    EPS,
                    n,
                )
                .expect("batched norm");
                let ys: Vec<GpuTensor> = w
                    .rows
                    .iter()
                    .map(|&m| poisoned(&mut gpu, MAXL * m))
                    .collect();
                new_launch(&mut gpu, &w, &x_rot_batch, &ys, n, test_variant());
                let dumps: Vec<Vec<u32>> = w
                    .rows
                    .iter()
                    .zip(&ys)
                    .map(|(&m, y)| dl(&gpu, y, MAXL * m))
                    .collect();
                for (b, x) in xs.iter().enumerate() {
                    gpu.hip.memcpy_htod(&x1.buf, f32_bytes(x)).unwrap();
                    let outs: Vec<GpuTensor> =
                        w.rows.iter().map(|&m| poisoned(&mut gpu, m)).collect();
                    let o = |i: usize| &outs[i.min(outs.len() - 1)];
                    let args = NormFoldArgs {
                        x: &x1,
                        norm_weight: &norm_w,
                        x_rot: &x1_rot,
                        weights: [&w.w[0], &w.w[1], &w.w[2], &w.w[3.min(w.w.len() - 1)]],
                        outs: [o(0), o(1), o(2), o(3)],
                        m: [
                            w.rows[0],
                            w.rows[1],
                            w.rows[2],
                            if w.rows.len() > 3 { w.rows[3] } else { 0 },
                        ],
                        eps: EPS,
                    };
                    let total: usize = w.rows.iter().sum();
                    gpu.fused_rmsnorm_proj4_mq4g256v2_fold(&args, norm_fold_grid(total, 3))
                        .expect("fold");
                    for (i, &m) in w.rows.iter().enumerate() {
                        let want = dl(&gpu, &outs[i], m);
                        let got = &dumps[i][b * m..(b + 1) * m];
                        if let Some(p) = first_diff(got, &want) {
                            bad += 1;
                            eprintln!(
                                "[{} chain n{n} seed{seed}] seg {i} lane {b} word {p}: got {:08x} want {:08x}",
                                group.name(),
                                got[p],
                                want[p]
                            );
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        bad, 0,
        "lane_proj chain is not byte-identical to the norm-fold singleton"
    );
}

/// A one-bit flip of one lane's activation changes exactly that lane's column.
#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn lane_proj_negative_control_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    require_gfx1201(&gpu);
    let rig = alloc_rig(&mut gpu);
    for group in [Group::Dn, Group::Fa, Group::Moe] {
        let w = build_weights(&mut gpu, group, 0x99 + group as u64, WKind::Small);
        let n = 5usize;
        let mut rng = Lcg(0x5EED);
        let xs: Vec<Vec<f32>> = (0..n).map(|_| make_x(XKind::Random, &mut rng)).collect();
        let run = |gpu: &mut Gpu, xs: &[Vec<f32>]| -> Vec<Vec<u32>> {
            for (b, x) in xs.iter().enumerate() {
                gpu.hip
                    .memcpy_htod(
                        &rig.x_batch.sub_offset(b * HIDDEN, HIDDEN).buf,
                        f32_bytes(x),
                    )
                    .unwrap();
            }
            let ys: Vec<GpuTensor> = w.rows.iter().map(|&m| poisoned(gpu, MAXL * m)).collect();
            new_launch(gpu, &w, &rig.x_batch, &ys, n, LaneProjVariant::PRODUCT);
            w.rows
                .iter()
                .zip(&ys)
                .map(|(&m, y)| dl(gpu, y, MAXL * m))
                .collect()
        };
        let base = run(&mut gpu, &xs);
        let mut pert = xs.clone();
        pert[2][700] = f32::from_bits(pert[2][700].to_bits() ^ (1 << 20));
        let got = run(&mut gpu, &pert);
        for b in 0..n {
            let mut changed = 0usize;
            for (i, &m) in w.rows.iter().enumerate() {
                changed += got[i][b * m..(b + 1) * m]
                    .iter()
                    .zip(&base[i][b * m..(b + 1) * m])
                    .filter(|(a, c)| a != c)
                    .count();
            }
            if b == 2 {
                assert!(changed > 0, "{}: flipped lane unchanged", group.name());
            } else {
                assert_eq!(changed, 0, "{}: lane {b} bled", group.name());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

fn median_min(v: &mut [f64]) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0])
}

/// Mean per-launch time (us) of `reps` x `per` back-to-back launches of `f`,
/// returned as (median, min) over reps.
fn time_launches(
    gpu: &mut Gpu,
    per: usize,
    reps: usize,
    mut f: impl FnMut(&mut Gpu, usize),
) -> (f64, f64) {
    let start = gpu.hip.event_create().expect("ev");
    let stop = gpu.hip.event_create().expect("ev");
    for i in 0..per {
        f(gpu, i); // warmup
    }
    gpu.hip.device_synchronize().expect("sync");
    let mut samples = Vec::new();
    for _ in 0..reps {
        gpu.hip.event_record(&start, None).expect("rec");
        for i in 0..per {
            f(gpu, i);
        }
        gpu.hip.event_record(&stop, None).expect("rec");
        gpu.hip.event_synchronize(&stop).expect("sync");
        let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed");
        samples.push(ms as f64 * 1000.0 / per as f64);
    }
    gpu.hip.event_destroy(start).ok();
    gpu.hip.event_destroy(stop).ok();
    median_min(&mut samples)
}

fn parse_variants() -> Vec<LaneProjVariant> {
    let mut out = vec![LaneProjVariant::PRODUCT];
    if let Ok(s) = std::env::var("LANE_PROJ_SWEEP") {
        for item in s.split(',').filter(|t| !t.is_empty()) {
            let p: Vec<usize> = item.split(':').map(|t| t.parse().unwrap()).collect();
            let v = LaneProjVariant {
                rows_per_wave: p[0],
                waves: p[1],
                wgs_per_wgp: p[2],
            };
            if !out.contains(&v) {
                out.push(v);
            }
        }
    }
    out
}

#[test]
#[ignore = "needs an exact gfx1201 GPU; run with fabric --class timing"]
fn lane_proj_timing_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    require_gfx1201(&gpu);
    let variants = parse_variants();
    let rig = alloc_rig(&mut gpu);
    let mut rng = Lcg(0x7131);
    let xs = make_x(XKind::Random, &mut rng);
    let all: Vec<f32> = (0..MAXL).flat_map(|_| xs.iter().copied()).collect();
    gpu.hip
        .memcpy_htod(&rig.x_batch.buf, f32_bytes(&all))
        .unwrap();
    let lanes: Vec<usize> = std::env::var("LANE_PROJ_LANES")
        .ok()
        .map(|s| s.split(',').map(|t| t.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![8, 1]);
    let only_group = std::env::var("LANE_PROJ_GROUPS").ok();
    let skip_old = std::env::var("LANE_PROJ_SKIP_OLD").is_ok();
    for group in [Group::Dn, Group::Fa, Group::Moe] {
        if only_group.as_deref().is_some_and(|g| g != group.name()) {
            continue;
        }
        let bytes: usize = group.rows().iter().sum::<usize>() * 8 * 136;
        // enough distinct weight sets that the cycle exceeds the last-level cache
        let sets = (192usize << 20).div_ceil(bytes).clamp(2, 160);
        let pool: Vec<Weights> = (0..sets)
            .map(|s| build_weights(&mut gpu, group, 0x500 + s as u64, WKind::Small))
            .collect();
        for &n in &lanes {
            let ys: Vec<GpuTensor> = group
                .rows()
                .iter()
                .map(|&m| poisoned(&mut gpu, MAXL * m))
                .collect();
            let (om, omin) = if skip_old {
                (1.0, 1.0)
            } else {
                time_launches(&mut gpu, 200, 25, |g, i| {
                    old_loop(g, &pool[i % sets], &rig.x_batch, &ys, n)
                })
            };
            eprintln!(
                "TIMING {} N={n} bytes={:.2}MB OLD(xbatch_pm loop x{}): median {om:.2} us min {omin:.2} us ({:.0} GB/s median)",
                group.name(),
                bytes as f64 / 1e6,
                group.rows().len(),
                bytes as f64 / om / 1e3
            );
            if !skip_old {
                // per-weight OLD launches (the production shapes the group replaces)
                for (si, &m) in group.rows().iter().enumerate() {
                    let seg_bytes = m * 8 * 136;
                    let (sm, smin) = time_launches(&mut gpu, 200, 25, |g, i| {
                        g.gemv_mq4g256v2_xbatch_pm(
                            &pool[i % sets].w[si],
                            &rig.x_batch,
                            &ys[si],
                            m,
                            HIDDEN,
                            n,
                        )
                        .expect("xbatch")
                    });
                    eprintln!(
                        "TIMING {} N={n} OLD seg{si} m={m}: median {sm:.2} us min {smin:.2} us ({:.0} GB/s median)",
                        group.name(),
                        seg_bytes as f64 / sm / 1e3
                    );
                }
            }
            for v in &variants {
                let (nm, nmin) = time_launches(&mut gpu, 200, 25, |g, i| {
                    new_launch(g, &pool[i % sets], &rig.x_batch, &ys, n, *v)
                });
                eprintln!(
                    "TIMING {} N={n} NEW r{}w{}wg{}: median {nm:.2} us min {nmin:.2} us ({:.0} GB/s median) speedup {:.2}x",
                    group.name(),
                    v.rows_per_wave,
                    v.waves,
                    v.wgs_per_wgp,
                    bytes as f64 / nm / 1e3,
                    om / nm
                );
            }
        }
    }
}
