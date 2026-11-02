// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Per-(layer, tag, weight) error-concentration analysis.
//! Tokens: calibration = first T/2, evaluation = last T/2. All matmuls f32 (faer, Par::Seq per rayon task),
//! all norm accumulations f64.
use crate::{quant, rot};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::sync::Mutex;

pub const PS: [f64; 4] = [0.05, 0.10, 0.20, 0.40];
pub const CS: [f64; 5] = [0.005, 0.01, 0.02, 0.05, 0.20];
pub const KS: [usize; 3] = [16, 32, 64];
const NC: usize = 64; // output rows per task
const NCFG: usize = 44; // 32 selective + yall + y2all + 5 rot outlier + 5 orig outlier
const CFG_YALL: usize = 32;
const CFG_Y2ALL: usize = 33;
const CFG_ROT: usize = 34;
const CFG_ORIG: usize = 39;
const RANK_NAMES: [&str; 4] = ["static_calib", "static_insample", "dyn_oracle", "dyn_proxy"];

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(&self) -> *mut f32 {
        self.0
    }
}

/// Per-(layer, tag) data shared by all consumer weights.
pub struct Ctx {
    pub k: usize,
    pub t: usize,
    pub tc: usize,
    pub nb: usize,
    pub tt: usize,
    pub xrot: Vec<f32>,
    pub r: Vec<f32>,
    pub r2: Vec<f32>,
    pub rn2: Vec<f32>,        // [t][b] ||r'_{t,b}||^2
    pub rcol2_cal: Vec<f64>,  // [k] sum over calib tokens of r'^2
    pub ro_eval: Vec<f32>,    // [Te][k] original-domain residual of eval tokens
    pub rocol2_cal: Vec<f64>, // [k]
    pub s1: Vec<f32>,
    pub s2: Vec<f32>,
}

fn tile_width(tc: usize) -> usize {
    (1..=128.min(tc)).rev().find(|d| tc % d == 0).unwrap()
}

impl Ctx {
    pub fn new(xrot: Vec<f32>, r: Vec<f32>, k: usize, t: usize, s1: &[f32], s2: &[f32]) -> Ctx {
        let nb = k / 128;
        let tc = t / 2;
        let mut r2 = vec![0f32; t * k];
        let mut rn2 = vec![0f32; t * nb];
        r2.par_chunks_mut(k)
            .zip(rn2.par_chunks_mut(nb))
            .enumerate()
            .for_each(|(tok, (o, n))| {
                for e in 0..nb {
                    let rb = &r[tok * k + e * 128..tok * k + (e + 1) * 128];
                    quant::a8_residual(rb, &mut o[e * 128..(e + 1) * 128]);
                    n[e] = rb.iter().map(|v| v * v).sum();
                }
            });
        let rcol2_cal = col_sumsq(&r[..tc * k], k);
        let mut ro_eval = r[tc * k..].to_vec();
        ro_eval
            .par_chunks_mut(k)
            .for_each(|row| rot::rotate_inv(row, s1, s2));
        let rocol2_cal = {
            let mut tmp = r[..tc * k].to_vec();
            tmp.par_chunks_mut(k)
                .for_each(|row| rot::rotate_inv(row, s1, s2));
            col_sumsq(&tmp, k)
        };
        Ctx {
            k,
            t,
            tc,
            nb,
            tt: tile_width(tc),
            xrot,
            r,
            r2,
            rn2,
            rcol2_cal,
            ro_eval,
            rocol2_cal,
            s1: s1.to_vec(),
            s2: s2.to_vec(),
        }
    }
}

/// Column sums of squares of a row-major [rows][k] matrix (f64).
fn col_sumsq(m: &[f32], k: usize) -> Vec<f64> {
    m.par_chunks((m.len() / k / 64).max(1) * k)
        .map(|blk| {
            let mut acc = vec![0f64; k];
            for row in blk.chunks(k) {
                for (a, v) in acc.iter_mut().zip(row) {
                    *a += (*v as f64) * (*v as f64);
                }
            }
            acc
        })
        .reduce(
            || vec![0f64; k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        )
}

fn cnt(p: f64, nb: usize) -> usize {
    ((p * nb as f64).round() as usize).clamp(1, nb)
}

/// Descending argsort (stable: ties favour the lower index).
fn argsort_desc(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap().then(a.cmp(&b)));
    idx
}
/// Tier per item: 0..3 = nested set that first contains it (top 5/10/20/40%), 4 = none.
fn tiers_from_order(order: &[usize], counts: &[usize; 4]) -> Vec<u8> {
    let mut t = vec![4u8; order.len()];
    for (rank, &i) in order.iter().enumerate() {
        if let Some(j) = (0..4).find(|&j| rank < counts[j]) {
            t[i] = j as u8;
        }
    }
    t
}

struct Gather {
    cc: [usize; 5],
    width: usize,
    wg: Vec<f32>,
    rg: Vec<f32>,
}

fn pick_cols(score: &[f64], k: usize) -> (Vec<usize>, [usize; 5]) {
    let order = argsort_desc(score);
    let mut cc = [0usize; 5];
    for c in 0..5 {
        cc[c] = cnt(CS[c], k);
    }
    (order[..cc[4]].to_vec(), cc)
}
fn make_gather(
    w: &[f32],
    n: usize,
    k: usize,
    cols: &[usize],
    cc: [usize; 5],
    r_eval: &[f32],
) -> Gather {
    let width = cols.len();
    let mut wg = vec![0f32; n * width];
    wg.par_chunks_mut(width).enumerate().for_each(|(i, o)| {
        for (c, &col) in cols.iter().enumerate() {
            o[c] = w[i * k + col];
        }
    });
    let te = r_eval.len() / k;
    let mut rg = vec![0f32; te * width];
    rg.par_chunks_mut(width).enumerate().for_each(|(t, o)| {
        for (c, &col) in cols.iter().enumerate() {
            o[c] = r_eval[t * k + col];
        }
    });
    Gather { cc, width, wg, rg }
}

fn mm(dst: &mut [f32], nr: usize, tw: usize, replace: bool, lhs: MatRef<f32>, rhs: MatRef<f32>) {
    matmul(
        MatMut::from_column_major_slice_mut(dst, nr, tw),
        if replace { Accum::Replace } else { Accum::Add },
        lhs,
        rhs,
        1.0f32,
        Par::Seq,
    );
}
fn add_into(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += *y;
    }
}

struct StageA {
    g: Vec<f64>,
    yx2: Vec<f64>,
    e: Vec<f32>,
}

fn stage_a(ctx: &Ctx, w: &[f32], n: usize) -> StageA {
    let (k, t, nb, tc, tt) = (ctx.k, ctx.t, ctx.nb, ctx.tc, ctx.tt);
    let nchunks = n.div_ceil(NC);
    let ntiles = t / tt;
    let g = Mutex::new(vec![0f64; t * nb]);
    let yx2 = Mutex::new(vec![0f64; t]);
    let mut e = vec![0f32; n * t];
    let eptr = SendPtr(e.as_mut_ptr());
    let tasks: Vec<(usize, usize)> = (0..nchunks)
        .flat_map(|c| (0..ntiles).map(move |ti| (c, ti)))
        .collect();
    tasks.par_iter().for_each(|&(c, ti)| {
        let row0 = c * NC;
        let nr = NC.min(n - row0);
        let t0 = ti * tt;
        let tw = tt;
        let mut yall = vec![0f32; nr * tw];
        let mut ytmp = vec![0f32; nr * tw];
        let mut gloc = vec![0f64; tw * nb];
        for b in 0..nb {
            let a = MatRef::from_row_major_slice_with_stride(&w[row0 * k + b * 128..], nr, 128, k);
            let rhs =
                MatRef::from_column_major_slice_with_stride(&ctx.r[t0 * k + b * 128..], 128, tw, k);
            mm(&mut ytmp, nr, tw, true, a, rhs);
            for tj in 0..tw {
                gloc[tj * nb + b] = ytmp[tj * nr..(tj + 1) * nr]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>() as f64;
            }
            add_into(&mut yall, &ytmp);
        }
        for tj in 0..tw {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    yall[tj * nr..].as_ptr(),
                    eptr.get().add((t0 + tj) * n + row0),
                    nr,
                );
            }
        }
        {
            let mut gg = g.lock().unwrap();
            for tj in 0..tw {
                for b in 0..nb {
                    gg[(t0 + tj) * nb + b] += gloc[tj * nb + b];
                }
            }
        }
        if t0 >= tc {
            let a = MatRef::from_row_major_slice_with_stride(&w[row0 * k..], nr, k, k);
            let rhs = MatRef::from_column_major_slice_with_stride(&ctx.xrot[t0 * k..], k, tw, k);
            mm(&mut ytmp, nr, tw, true, a, rhs);
            let mut y = yx2.lock().unwrap();
            for tj in 0..tw {
                y[t0 + tj] += ytmp[tj * nr..(tj + 1) * nr]
                    .iter()
                    .map(|v| (*v as f64) * (*v as f64))
                    .sum::<f64>();
            }
        }
    });
    StageA {
        g: g.into_inner().unwrap(),
        yx2: yx2.into_inner().unwrap(),
        e,
    }
}

struct Plan {
    stat: [Vec<u8>; 2],
    dynt: [Vec<u8>; 2],
    rot: Gather,
    orig: Gather,
}

fn stage_b(ctx: &Ctx, w: &[f32], n: usize, plan: &Plan) -> Vec<f64> {
    let (k, t, nb, tc, tt) = (ctx.k, ctx.t, ctx.nb, ctx.tc, ctx.tt);
    let nchunks = n.div_ceil(NC);
    let ntiles = t / tt;
    let res = Mutex::new(vec![0f64; NCFG * t]);
    let tasks: Vec<(usize, usize)> = (0..nchunks)
        .flat_map(|c| (0..ntiles).map(move |ti| (c, ti)))
        .collect();
    tasks.par_iter().for_each(|&(c, ti)| {
        let row0 = c * NC;
        let nr = NC.min(n - row0);
        let t0 = ti * tt;
        let tw = tt;
        let sz = nr * tw;
        let eval = t0 >= tc;
        let active: &[usize] = if eval { &[0, 1, 2, 3] } else { &[1] };
        let mut ay: Vec<Vec<f32>> = vec![Vec::new(); 32];
        for &r in active {
            for j in 0..4 {
                for kind in 0..2 {
                    ay[(r * 4 + j) * 2 + kind] = vec![0f32; sz];
                }
            }
        }
        let (mut yall, mut y2all, mut y, mut y2) = (
            vec![0f32; sz],
            vec![0f32; sz],
            vec![0f32; sz],
            vec![0f32; sz],
        );
        for b in 0..nb {
            let a = MatRef::from_row_major_slice_with_stride(&w[row0 * k + b * 128..], nr, 128, k);
            mm(
                &mut y,
                nr,
                tw,
                true,
                a,
                MatRef::from_column_major_slice_with_stride(&ctx.r[t0 * k + b * 128..], 128, tw, k),
            );
            mm(
                &mut y2,
                nr,
                tw,
                true,
                a,
                MatRef::from_column_major_slice_with_stride(
                    &ctx.r2[t0 * k + b * 128..],
                    128,
                    tw,
                    k,
                ),
            );
            add_into(&mut yall, &y);
            add_into(&mut y2all, &y2);
            for &r in active {
                if r < 2 {
                    let j = plan.stat[r][b] as usize;
                    if j < 4 {
                        add_into(&mut ay[(r * 4 + j) * 2], &y);
                        add_into(&mut ay[(r * 4 + j) * 2 + 1], &y2);
                    }
                } else {
                    let tiers = &plan.dynt[r - 2];
                    for tj in 0..tw {
                        let j = tiers[(t0 + tj - tc) * nb + b] as usize;
                        if j < 4 {
                            add_into(
                                &mut ay[(r * 4 + j) * 2][tj * nr..(tj + 1) * nr],
                                &y[tj * nr..(tj + 1) * nr],
                            );
                            add_into(
                                &mut ay[(r * 4 + j) * 2 + 1][tj * nr..(tj + 1) * nr],
                                &y2[tj * nr..(tj + 1) * nr],
                            );
                        }
                    }
                }
            }
        }
        let mut local = vec![0f64; NCFG * tw];
        let colsum = |z: &dyn Fn(usize) -> f32, out: &mut [f64]| {
            for tj in 0..tw {
                let mut s = 0f32;
                for i in 0..nr {
                    let v = z(tj * nr + i);
                    s += v * v;
                }
                out[tj] += s as f64;
            }
        };
        colsum(&|i| yall[i], &mut local[CFG_YALL * tw..(CFG_YALL + 1) * tw]);
        colsum(
            &|i| y2all[i],
            &mut local[CFG_Y2ALL * tw..(CFG_Y2ALL + 1) * tw],
        );
        for &r in active {
            let (mut cy, mut cy2) = (vec![0f32; sz], vec![0f32; sz]);
            for pi in 0..4 {
                add_into(&mut cy, &ay[(r * 4 + pi) * 2]);
                add_into(&mut cy2, &ay[(r * 4 + pi) * 2 + 1]);
                let (ce, ca) = ((r * 4 + pi) * 2, (r * 4 + pi) * 2 + 1);
                for tj in 0..tw {
                    let (mut se, mut sa) = (0f32, 0f32);
                    for i in tj * nr..(tj + 1) * nr {
                        let ze = yall[i] - cy[i];
                        let za = ze + cy2[i];
                        se += ze * ze;
                        sa += za * za;
                    }
                    local[ce * tw + tj] += se as f64;
                    local[ca * tw + tj] += sa as f64;
                }
            }
        }
        if eval {
            let te0 = t0 - tc;
            for (dom, g, base) in [(0, &plan.rot, CFG_ROT), (1, &plan.orig, CFG_ORIG)] {
                let _ = dom;
                let mut cum = vec![0f32; sz];
                for cidx in 0..5 {
                    let lo = if cidx == 0 { 0 } else { g.cc[cidx - 1] };
                    let hi = g.cc[cidx];
                    if hi > lo {
                        let a = MatRef::from_row_major_slice_with_stride(
                            &g.wg[row0 * g.width + lo..],
                            nr,
                            hi - lo,
                            g.width,
                        );
                        let rhs = MatRef::from_column_major_slice_with_stride(
                            &g.rg[te0 * g.width + lo..],
                            hi - lo,
                            tw,
                            g.width,
                        );
                        mm(&mut y, nr, tw, true, a, rhs);
                        add_into(&mut cum, &y);
                    }
                    let out = &mut local[(base + cidx) * tw..(base + cidx + 1) * tw];
                    for tj in 0..tw {
                        let mut s = 0f32;
                        for i in tj * nr..(tj + 1) * nr {
                            let z = yall[i] - cum[i];
                            s += z * z;
                        }
                        out[tj] += s as f64;
                    }
                }
            }
        }
        let mut rr = res.lock().unwrap();
        for cfg in 0..NCFG {
            for tj in 0..tw {
                rr[cfg * t + t0 + tj] += local[cfg * tw + tj];
            }
        }
    });
    res.into_inner().unwrap()
}

// ---------------- low rank ----------------
fn randn(rows: usize, cols: usize, seed: u64) -> Mat<f32> {
    let mut s = seed | 1;
    let mut unit = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    Mat::from_fn(rows, cols, |_, _| {
        ((0..12).map(|_| unit()).sum::<f64>() - 6.0) as f32
    })
}
fn gemm(a: MatRef<f32>, b: MatRef<f32>) -> Mat<f32> {
    let mut c = Mat::<f32>::zeros(a.nrows(), b.ncols());
    matmul(c.as_mut(), Accum::Replace, a, b, 1.0f32, Par::rayon(0));
    c
}
/// QR/SVD of the tall-skinny panels run sequentially: faer's parallel path costs ~300x more here (measured 0.57 s vs 2 ms).
fn orth(a: Mat<f32>) -> Mat<f32> {
    faer::set_global_parallelism(Par::Seq);
    a.qr().compute_thin_Q()
}

/// Data-aware: U_k = top-k left singular vectors of E_cal (randomized subspace iteration);
/// returns (recovery per KS on E_eval, leading singular value estimates).
pub fn lowrank_da(
    ecal: MatRef<f32>,
    eeval: MatRef<f32>,
    iters: usize,
    os: usize,
) -> (Vec<f64>, Vec<f64>) {
    faer::set_global_parallelism(Par::Seq);
    let (n, tc) = (ecal.nrows(), ecal.ncols());
    let kmax = (*KS.iter().max().unwrap()).min(n).min(tc);
    let l = (kmax + os).min(n).min(tc);
    let mut q = orth(randn(tc, l, 0xA11CE));
    for _ in 0..iters {
        let y = orth(gemm(ecal, q.as_ref()));
        q = orth(gemm(ecal.transpose(), y.as_ref()));
    }
    let y = orth(gemm(ecal, q.as_ref()));
    let b = gemm(y.as_ref().transpose(), ecal);
    let svd = b.thin_svd().unwrap();
    let u = gemm(y.as_ref(), svd.U().subcols(0, kmax));
    let m = gemm(u.as_ref().transpose(), eeval);
    let rowsq: Vec<f64> = (0..kmax)
        .map(|i| (0..m.ncols()).map(|j| (m[(i, j)] as f64).powi(2)).sum())
        .collect();
    let tot: f64 = (0..eeval.ncols())
        .into_par_iter()
        .map(|j| (0..n).map(|i| (eeval[(i, j)] as f64).powi(2)).sum::<f64>())
        .sum();
    let rec = KS
        .iter()
        .map(|&k| rowsq[..k.min(kmax)].iter().sum::<f64>() / tot)
        .collect();
    let sv = (0..kmax)
        .map(|i| svd.S().column_vector()[i] as f64)
        .collect();
    (rec, sv)
}

/// Data-free: V_k = top-k right singular vectors of W' (randomized subspace iteration);
/// recovery = 1 - ||W'(I - V_k^T V_k) R||^2 / ||W' R||^2 on eval.
pub fn lowrank_df(
    w: MatRef<f32>,
    eeval: MatRef<f32>,
    reval_t: MatRef<f32>,
    iters: usize,
    os: usize,
) -> Vec<f64> {
    faer::set_global_parallelism(Par::Seq);
    let (n, k) = (w.nrows(), w.ncols());
    let kmax = (*KS.iter().max().unwrap()).min(n).min(k);
    let l = (kmax + os).min(n).min(k);
    let mut q = orth(randn(k, l, 0xB0B));
    for _ in 0..iters {
        let y = orth(gemm(w, q.as_ref()));
        q = orth(gemm(w.transpose(), y.as_ref()));
    }
    let b = gemm(w, q.as_ref());
    let svd = b.thin_svd().unwrap();
    let vk = gemm(q.as_ref(), svd.V().subcols(0, kmax));
    let wv = gemm(b.as_ref(), svd.V().subcols(0, kmax));
    let p = gemm(vk.as_ref().transpose(), reval_t);
    let tot: f64 = (0..eeval.ncols())
        .into_par_iter()
        .map(|j| (0..n).map(|i| (eeval[(i, j)] as f64).powi(2)).sum::<f64>())
        .sum();
    KS.iter()
        .map(|&kk| {
            let kk = kk.min(kmax);
            let mut z = eeval.to_owned();
            matmul(
                z.as_mut(),
                Accum::Add,
                wv.as_ref().subcols(0, kk),
                p.as_ref().subrows(0, kk),
                -1.0f32,
                Par::rayon(0),
            );
            let resid: f64 = (0..z.ncols())
                .into_par_iter()
                .map(|j| (0..n).map(|i| (z[(i, j)] as f64).powi(2)).sum::<f64>())
                .sum();
            1.0 - resid / tot
        })
        .collect()
}

// ---------------- driver ----------------
fn sum_range(v: &[f64], a: usize, b: usize) -> f64 {
    v[a..b].iter().sum()
}
fn pkey(p: f64) -> String {
    format!("{}", (p * 1000.0).round() / 10.0)
}

pub fn analyze_weight(ctx: &Ctx, w: &[f32], n: usize, lowrank_iters: (usize, usize)) -> Value {
    let (k, t, nb, tc) = (ctx.k, ctx.t, ctx.nb, ctx.tc);
    let t0 = std::time::Instant::now();
    let wcol2 = col_sumsq(w, k);
    let wb2: Vec<f64> = (0..nb)
        .map(|b| wcol2[b * 128..(b + 1) * 128].iter().sum())
        .collect();

    // ---- stage A: per-(token,block) energies G, E store, S_eval
    let sa = stage_a(ctx, w, n);
    let g = &sa.g;
    let t_a = t0.elapsed().as_secs_f64();

    // ---- rankings
    let counts = [
        cnt(PS[0], nb),
        cnt(PS[1], nb),
        cnt(PS[2], nb),
        cnt(PS[3], nb),
    ];
    let mut eb_cal = vec![0f64; nb];
    let mut eb_all = vec![0f64; nb];
    let mut eb_eval = vec![0f64; nb];
    for tok in 0..t {
        for b in 0..nb {
            let v = g[tok * nb + b];
            eb_all[b] += v;
            if tok < tc {
                eb_cal[b] += v
            } else {
                eb_eval[b] += v
            }
        }
    }
    let tier_sc = tiers_from_order(&argsort_desc(&eb_cal), &counts);
    let tier_sa = tiers_from_order(&argsort_desc(&eb_all), &counts);
    let mut tier_do = vec![4u8; (t - tc) * nb];
    let mut tier_dp = vec![4u8; (t - tc) * nb];
    tier_do
        .par_chunks_mut(nb)
        .zip(tier_dp.par_chunks_mut(nb))
        .enumerate()
        .for_each(|(te, (o, p))| {
            let tok = tc + te;
            o.copy_from_slice(&tiers_from_order(
                &argsort_desc(&g[tok * nb..(tok + 1) * nb]),
                &counts,
            ));
            let prox: Vec<f64> = (0..nb)
                .map(|b| ctx.rn2[tok * nb + b] as f64 * wb2[b])
                .collect();
            p.copy_from_slice(&tiers_from_order(&argsort_desc(&prox), &counts));
        });
    // outlier channels: rotated domain
    let rscore: Vec<f64> = (0..k).map(|c| wcol2[c] * ctx.rcol2_cal[c]).collect();
    let (rcols, rcc) = pick_cols(&rscore, k);
    let rot_g = make_gather(w, n, k, &rcols, rcc, &ctx.r[tc * k..]);
    // original domain: W'_o rows = P W'^T, r_o = P r'
    let mut wo = w.to_vec();
    wo.par_chunks_mut(k)
        .for_each(|row| rot::rotate_inv(row, &ctx.s1, &ctx.s2));
    let wocol2 = col_sumsq(&wo, k);
    let oscore: Vec<f64> = (0..k).map(|c| wocol2[c] * ctx.rocol2_cal[c]).collect();
    let (ocols, occ) = pick_cols(&oscore, k);
    let orig_g = make_gather(&wo, n, k, &ocols, occ, &ctx.ro_eval);
    // numerical identity check W' r' == W'_o r_o on the first eval tile, first <=64 rows
    let identity_err = {
        let nr = n.min(64);
        let tw = ctx.tt;
        let mut ya = vec![0f32; nr * tw];
        let mut yb = vec![0f32; nr * tw];
        mm(
            &mut ya,
            nr,
            tw,
            true,
            MatRef::from_row_major_slice_with_stride(&w[..], nr, k, k),
            MatRef::from_column_major_slice_with_stride(&ctx.r[tc * k..], k, tw, k),
        );
        mm(
            &mut yb,
            nr,
            tw,
            true,
            MatRef::from_row_major_slice_with_stride(&wo[..], nr, k, k),
            MatRef::from_column_major_slice_with_stride(&ctx.ro_eval[..], k, tw, k),
        );
        let m = ya.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = ya
            .iter()
            .zip(&yb)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        (d / m.max(1e-30)) as f64
    };
    drop(wo);
    let plan = Plan {
        stat: [tier_sc.clone(), tier_sa.clone()],
        dynt: [tier_do.clone(), tier_dp.clone()],
        rot: rot_g,
        orig: orig_g,
    };

    // ---- stage B
    let res = stage_b(ctx, w, n, &plan);
    let t_b = t0.elapsed().as_secs_f64();
    let row = |cfg: usize| &res[cfg * t..(cfg + 1) * t];
    let e_eval = sum_range(row(CFG_YALL), tc, t);
    let e_all_tok = sum_range(row(CFG_YALL), 0, t);
    let e_cal = sum_range(row(CFG_YALL), 0, tc);
    let s_eval: f64 = sum_range(&sa.yx2, tc, t);
    let sel = |tiers: &[u8], pi: usize| -> f64 {
        (0..nb)
            .filter(|&b| (tiers[b] as usize) <= pi)
            .map(|b| eb_eval[b])
            .sum()
    };
    let sel_all = |tiers: &[u8], pi: usize| -> f64 {
        (0..nb)
            .filter(|&b| (tiers[b] as usize) <= pi)
            .map(|b| eb_all[b])
            .sum()
    };
    let g_eval_total: f64 = eb_eval.iter().sum();
    let g_all_total: f64 = eb_all.iter().sum();
    let mut rankings = serde_json::Map::new();
    for r in 0..4 {
        let mut per_p = serde_json::Map::new();
        for pi in 0..4 {
            let (lo, hi, etot, diag) = if r == 1 {
                (0, t, e_all_tok, sel_all(&tier_sa, pi) / g_all_total)
            } else {
                (
                    tc,
                    t,
                    e_eval,
                    match r {
                        0 => sel(&tier_sc, pi) / g_eval_total,
                        _ => {
                            let tiers = if r == 2 { &tier_do } else { &tier_dp };
                            let mut s = 0f64;
                            for te in 0..t - tc {
                                for b in 0..nb {
                                    if (tiers[te * nb + b] as usize) <= pi {
                                        s += g[(tc + te) * nb + b];
                                    }
                                }
                            }
                            s / g_eval_total
                        }
                    },
                )
            };
            let re = sum_range(row((r * 4 + pi) * 2), lo, hi);
            let ra = sum_range(row((r * 4 + pi) * 2 + 1), lo, hi);
            per_p.insert(pkey(PS[pi]), json!({"nsel": counts[pi], "diag_share": diag, "f_exact": 1.0 - re / etot, "f_a8": 1.0 - ra / etot, "resid_exact": re, "resid_a8": ra, "e_tot": etot}));
        }
        rankings.insert(RANK_NAMES[r].to_string(), Value::Object(per_p));
    }
    let r2sum = sum_range(row(CFG_Y2ALL), tc, t);
    let fullk = json!({"f_a8": 1.0 - r2sum / e_eval, "resid": r2sum, "e_tot": e_eval});
    let mut outl = serde_json::Map::new();
    for (name, base, cc) in [
        ("outlier_rot", CFG_ROT, rcc),
        ("outlier_orig", CFG_ORIG, occ),
    ] {
        let mut m = serde_json::Map::new();
        for c in 0..5 {
            let rs = sum_range(row(base + c), tc, t);
            m.insert(
                pkey(CS[c]),
                json!({"ncols": cc[c], "f_exact": 1.0 - rs / e_eval, "resid": rs, "e_tot": e_eval}),
            );
        }
        outl.insert(name.to_string(), Value::Object(m));
    }

    // ---- low rank (E from stage A is col-major N x T; calib = first tc columns)
    let ecal = MatRef::from_column_major_slice(&sa.e[..n * tc], n, tc);
    let eeval = MatRef::from_column_major_slice(&sa.e[n * tc..], n, t - tc);
    let wmat = MatRef::from_row_major_slice(w, n, k);
    let reval_t = MatRef::from_column_major_slice_with_stride(&ctx.r[tc * k..], k, t - tc, k);
    let (it_lo, it_hi) = lowrank_iters;
    let (da_lo, sv_lo) = lowrank_da(ecal, eeval, it_lo, 32);
    let (da_hi, _) = lowrank_da(ecal, eeval, it_hi, 32);
    let df_lo = lowrank_df(wmat, eeval, reval_t, it_lo, 32);
    let df_hi = lowrank_df(wmat, eeval, reval_t, it_hi, 32);
    let mut lr_da = serde_json::Map::new();
    let mut lr_df = serde_json::Map::new();
    let mut conv = 0f64;
    for (i, &kk) in KS.iter().enumerate() {
        lr_da.insert(
            kk.to_string(),
            json!({"f": da_lo[i], "f_hi_iter": da_hi[i], "rank_eff": kk.min(n)}),
        );
        lr_df.insert(
            kk.to_string(),
            json!({"f": df_lo[i], "f_hi_iter": df_hi[i], "rank_eff": kk.min(n)}),
        );
        conv = conv
            .max((da_lo[i] - da_hi[i]).abs())
            .max((df_lo[i] - df_hi[i]).abs());
    }
    let t_c = t0.elapsed().as_secs_f64();
    json!({
        "n": n, "k": k, "nb": nb, "t": t, "tc": tc,
        "e_eval": e_eval, "s_eval": s_eval, "nsr_eval": e_eval / s_eval, "e_cal": e_cal, "e_all": e_all_tok,
        "rankings": Value::Object(rankings), "fullk_a8": fullk,
        "outlier": Value::Object(outl),
        "lowrank_da": Value::Object(lr_da), "lowrank_df": Value::Object(lr_df),
        "lowrank_conv": {"iters": [it_lo, it_hi], "oversample": 32, "max_abs_diff_recovery": conv, "top_sv_da": &sv_lo[..sv_lo.len().min(4)]},
        "rot_identity_rel_err": identity_err,
        "diag_check": {"sum_g_eval_over_e_eval": g_eval_total / e_eval},
        "secs": {"stage_a": t_a, "stage_b": t_b - t_a, "lowrank": t_c - t_b},
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // W (n x k) with geometrically decaying spectrum, R (k x te) random.
    fn setup() -> (Mat<f32>, Mat<f32>) {
        let (n, k, te) = (120, 200, 192);
        let g1 = randn(n, 100, 1);
        let g2 = randn(100, k, 2);
        let mut w = Mat::<f32>::zeros(n, k);
        let mut scaled = g1.clone();
        for j in 0..100 {
            let s = 0.8f32.powi(j as i32);
            for i in 0..n {
                scaled[(i, j)] *= s;
            }
        }
        matmul(
            w.as_mut(),
            Accum::Replace,
            scaled.as_ref(),
            g2.as_ref(),
            1.0f32,
            Par::Seq,
        );
        (w, randn(k, te, 3))
    }
    #[test]
    fn randomized_df_matches_exact_svd() {
        let (w, r) = setup();
        let mut e = Mat::<f32>::zeros(w.nrows(), r.ncols());
        matmul(
            e.as_mut(),
            Accum::Replace,
            w.as_ref(),
            r.as_ref(),
            1.0f32,
            Par::Seq,
        );
        let got = lowrank_df(w.as_ref(), e.as_ref(), r.as_ref(), 8, 32);
        let svd = w.thin_svd().unwrap();
        let tot = sumsq_m(&e);
        for (i, &k) in KS.iter().enumerate() {
            // exact: project W's right-singular subspace
            let vk = svd.V().subcols(0, k);
            let p = gemm(vk.transpose(), r.as_ref());
            let wv = gemm(w.as_ref(), vk);
            let mut z = e.clone();
            matmul(
                z.as_mut(),
                Accum::Add,
                wv.as_ref(),
                p.as_ref(),
                -1.0f32,
                Par::Seq,
            );
            let exact = 1.0 - sumsq_m(&z) / tot;
            assert!(
                (got[i] - exact).abs() < 2e-3,
                "k={k}: randomized {} vs exact {}",
                got[i],
                exact
            );
        }
    }
    #[test]
    fn randomized_da_matches_exact_svd() {
        let (w, r) = setup();
        let te = 96;
        let mut e = Mat::<f32>::zeros(w.nrows(), r.ncols());
        matmul(
            e.as_mut(),
            Accum::Replace,
            w.as_ref(),
            r.as_ref(),
            1.0f32,
            Par::Seq,
        );
        let (ecal, eeval) = (
            e.as_ref().subcols(0, te).to_owned(),
            e.as_ref().subcols(te, te).to_owned(),
        );
        let (got, _) = lowrank_da(ecal.as_ref(), eeval.as_ref(), 8, 32);
        let svd = ecal.thin_svd().unwrap();
        let tot = sumsq_m(&eeval);
        for (i, &k) in KS.iter().enumerate() {
            let uk = svd.U().subcols(0, k);
            let m = gemm(uk.transpose(), eeval.as_ref());
            let exact = sumsq_m(&m) / tot;
            assert!(
                (got[i] - exact).abs() < 2e-3,
                "k={k}: randomized {} vs exact {}",
                got[i],
                exact
            );
        }
    }

    /// Brute-force f64 reference of the stage-A/B machinery on a tiny case.
    #[test]
    fn analyze_weight_matches_bruteforce() {
        let (n, k, t) = (70usize, 512usize, 16usize);
        let (tc, nb) = (t / 2, k / 128);
        let mut seed = 99u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        };
        let s1: Vec<f32> = (0..256)
            .map(|_| if rnd() > 0.0 { 1.0 } else { -1.0 })
            .collect();
        let s2: Vec<f32> = (0..256)
            .map(|_| if rnd() > 0.0 { 1.0 } else { -1.0 })
            .collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| rnd() * if (i % k) % 37 == 3 { 4.0 } else { 1.0 })
            .collect();
        let r: Vec<f32> = (0..t * k)
            .map(|i| rnd() * if (i % k) / 128 == 2 { 3.0 } else { 0.5 })
            .collect();
        let xrot: Vec<f32> = (0..t * k).map(|_| rnd()).collect();
        let ctx = Ctx::new(xrot, r.clone(), k, t, &s1, &s2);
        let out = analyze_weight(&ctx, &w, n, (8, 12));
        let y = |wm: &[f32],
                 rm: &dyn Fn(usize, usize) -> f64,
                 tok: usize,
                 mask: &dyn Fn(usize) -> bool|
         -> f64 {
            (0..n)
                .map(|i| {
                    let v: f64 = (0..k)
                        .filter(|&c| mask(c))
                        .map(|c| wm[i * k + c] as f64 * rm(tok, c))
                        .sum();
                    v * v
                })
                .sum()
        };
        let rr = |tok: usize, c: usize| r[tok * k + c] as f64;
        let r2 = |tok: usize, c: usize| ctx.r2[tok * k + c] as f64;
        let e_eval: f64 = (tc..t).map(|tok| y(&w, &rr, tok, &|_| true)).sum();
        assert!((out["e_eval"].as_f64().unwrap() / e_eval - 1.0).abs() < 1e-4);
        // per-(token,block) energy
        let gtb = |tok: usize, b: usize| y(&w, &rr, tok, &|c| c / 128 == b);
        let counts = [
            cnt(PS[0], nb),
            cnt(PS[1], nb),
            cnt(PS[2], nb),
            cnt(PS[3], nb),
        ];
        let ecal: Vec<f64> = (0..nb)
            .map(|b| (0..tc).map(|tok| gtb(tok, b)).sum())
            .collect();
        let order = argsort_desc(&ecal);
        for (pi, key) in ["5", "10", "20", "40"].iter().enumerate() {
            let sel: Vec<usize> = order[..counts[pi]].to_vec();
            let resid_exact: f64 = (tc..t)
                .map(|tok| y(&w, &rr, tok, &|c| !sel.contains(&(c / 128))))
                .sum();
            let mixed = |tok: usize, c: usize| {
                if sel.contains(&(c / 128)) {
                    r2(tok, c)
                } else {
                    rr(tok, c)
                }
            };
            let resid_a8: f64 = (tc..t).map(|tok| y(&w, &mixed, tok, &|_| true)).sum();
            let node = &out["rankings"]["static_calib"][*key];
            assert!(
                ((1.0 - resid_exact / e_eval) - node["f_exact"].as_f64().unwrap()).abs() < 1e-4,
                "static exact p={key}"
            );
            assert!(
                ((1.0 - resid_a8 / e_eval) - node["f_a8"].as_f64().unwrap()).abs() < 1e-4,
                "static a8 p={key}"
            );
            // dynamic oracle per token
            let mut resid_dyn = 0f64;
            let mut resid_dyn_a8 = 0f64;
            for tok in tc..t {
                let gs: Vec<f64> = (0..nb).map(|b| gtb(tok, b)).collect();
                let o = argsort_desc(&gs);
                let s: Vec<usize> = o[..counts[pi]].to_vec();
                resid_dyn += y(&w, &rr, tok, &|c| !s.contains(&(c / 128)));
                resid_dyn_a8 += y(
                    &w,
                    &|tk, c| {
                        if s.contains(&(c / 128)) {
                            r2(tk, c)
                        } else {
                            rr(tk, c)
                        }
                    },
                    tok,
                    &|_| true,
                );
            }
            let node = &out["rankings"]["dyn_oracle"][*key];
            assert!(
                ((1.0 - resid_dyn / e_eval) - node["f_exact"].as_f64().unwrap()).abs() < 1e-4,
                "dyn exact p={key}"
            );
            assert!(
                ((1.0 - resid_dyn_a8 / e_eval) - node["f_a8"].as_f64().unwrap()).abs() < 1e-4,
                "dyn a8 p={key}"
            );
        }
        // full-K A8
        let r2all: f64 = (tc..t).map(|tok| y(&w, &r2, tok, &|_| true)).sum();
        assert!(((1.0 - r2all / e_eval) - out["fullk_a8"]["f_a8"].as_f64().unwrap()).abs() < 1e-4);
        // outliers, rotated and original domain
        let mut wo = w.clone();
        rot::rotate_inv(&mut wo, &s1, &s2);
        let mut ro = r.clone();
        rot::rotate_inv(&mut ro, &s1, &s2);
        for (name, wm, rm) in [("outlier_rot", &w, &r), ("outlier_orig", &wo, &ro)] {
            let score: Vec<f64> = (0..k)
                .map(|c| {
                    (0..n).map(|i| (wm[i * k + c] as f64).powi(2)).sum::<f64>()
                        * (0..tc)
                            .map(|tok| (rm[tok * k + c] as f64).powi(2))
                            .sum::<f64>()
                })
                .collect();
            let ord = argsort_desc(&score);
            for (ci, key) in ["0.5", "1", "2", "5", "20"].iter().enumerate() {
                let sel: Vec<usize> = ord[..cnt(CS[ci], k)].to_vec();
                let rmf = |tok: usize, c: usize| rm[tok * k + c] as f64;
                let resid: f64 = (tc..t)
                    .map(|tok| y(wm, &rmf, tok, &|c| !sel.contains(&c)))
                    .sum();
                assert!(
                    ((1.0 - resid / e_eval)
                        - out["outlier"][name][*key]["f_exact"].as_f64().unwrap())
                    .abs()
                        < 2e-4,
                    "{name} c={key}"
                );
            }
        }
        assert!(out["rot_identity_rel_err"].as_f64().unwrap() < 1e-4);
    }
    fn sumsq_m(m: &Mat<f32>) -> f64 {
        let mut s = 0f64;
        for j in 0..m.ncols() {
            for i in 0..m.nrows() {
                s += (m[(i, j)] as f64).powi(2);
            }
        }
        s
    }
}
