// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Correctness gate for the absorbed-latent kernels (`latent_attention_absorb_*`)
// against an independent host F64 reference. No timing.
//
// Geometries: Raon-like (H=40, nope=128, tail=64, v=128, latent=512), a
// non-Raon one (H=3, nope=32, tail=16, v=48, latent=64) and an odd one with a
// non-vector tail (H=2, nope=16, tail=6, v=24, latent=40).
// Per geometry/row count:
//   * pack: byte-exact against a host re-packing of RNE-BF16(w_t) (the weights are
//     deliberately NOT BF16-valued before packing);
//   * absorb_q / absorb_v against F64 on BF16-rounded weights (rel_l2 <= 1e-4 and
//     cosine >= 0.99999 for the FP32 route; rounded-y route: rel_l2 <= 1e-2,
//     cosine >= 0.9999 and every element exactly BF16-representable);
//   * output buffers pre-poisoned with NaN and a NaN/1e30 guard region behind the
//     outputs, the inputs and the packed weights (guards must stay bit-intact, no
//     output element may stay poisoned: proves nothing outside the visible range is
//     read or skipped);
//   * byte-identical repeat runs (poison pattern changed between runs);
//   * the tail of q_abs is a bit-exact copy of the q tail.
// Plus the pack -> v roundtrip identity (one-hot z returns Av[:,c] exactly), and
// the host algebra check `(qabs . c) + (qt . kpe) == qnope . (Ak c) + qt . kpe`.
// rows == 4096 on the Raon geometry is checked on a sampled row set.
//
// Run through the fabric wrapper on card E.

use rayon::prelude::*;
use rdna_compute::latent_attention_absorb::latent_attention_absorb_packed_elems;
use rdna_compute::{DType, Gpu, GpuTensor};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bf16_bits(f: f32) -> u16 {
    let u = f.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        return ((u >> 16) | 0x40) as u16;
    }
    let r = u.wrapping_add(0x7fff + ((u >> 16) & 1));
    (r >> 16) as u16
}
fn bf16_val(f: f32) -> f32 {
    f32::from_bits((bf16_bits(f) as u32) << 16)
}

const NAN_BITS: u32 = 0x7fc0_0000;
const BIG_BITS: u32 = 0x7149_0000 | 0x0000_7149; // 1e30-class in both BF16 halves
const GUARD: usize = 1024;

fn guard_pattern(i: usize) -> u32 {
    if i % 2 == 0 {
        NAN_BITS
    } else {
        BIG_BITS
    }
}

fn htod(gpu: &Gpu, t: &GpuTensor, bytes: &[u8]) {
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod");
}

/// F32 tensor = `data` followed by a poisoned guard.
fn up_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let n = data.len() + GUARD;
    let t = gpu.alloc_tensor(&[n], DType::F32).expect("alloc f32");
    let mut bits: Vec<u32> = data.iter().map(|f| f.to_bits()).collect();
    bits.extend((0..GUARD).map(guard_pattern));
    htod(gpu, &t, bytemuck_u32(&bits));
    t
}

/// BF16 tensor = `data` followed by a poisoned guard.
fn up_bf16(gpu: &mut Gpu, data: &[u16], guard: bool) -> GpuTensor {
    let g = if guard { GUARD } else { 0 };
    let t = gpu
        .alloc_tensor(&[data.len() + g], DType::BF16)
        .expect("alloc bf16");
    let mut v: Vec<u16> = data.to_vec();
    v.extend((0..g).map(|i| if i % 2 == 0 { 0x7fc0 } else { 0x7149 }));
    htod(gpu, &t, bytemuck_u16(&v));
    t
}

/// Output tensor of `n` elements poisoned with `fill` bits, plus the guard.
fn out_f32(gpu: &mut Gpu, n: usize, fill: u32) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[n + GUARD], DType::F32)
        .expect("alloc out");
    let mut bits = vec![fill; n];
    bits.extend((0..GUARD).map(guard_pattern));
    htod(gpu, &t, bytemuck_u32(&bits));
    t
}

fn bytemuck_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn bytemuck_u16(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}

fn dl_f32(gpu: &Gpu, t: &GpuTensor) -> Vec<f32> {
    gpu.hip.device_synchronize().expect("sync");
    gpu.download_f32(t).expect("download")
}

fn guard_ok(v: &[f32], n: usize) -> bool {
    (0..GUARD).all(|i| v[n + i].to_bits() == guard_pattern(i))
}

fn stats(got: &[f32], want: &[f64]) -> (f64, f64) {
    let (mut dd, mut ww, mut gg, mut gw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        let g = *g as f64;
        dd += (g - w) * (g - w);
        ww += w * w;
        gg += g * g;
        gw += g * w;
    }
    let rel = if ww > 0.0 {
        (dd / ww).sqrt()
    } else {
        dd.sqrt()
    };
    let cos = if gg > 0.0 && ww > 0.0 {
        gw / (gg.sqrt() * ww.sqrt())
    } else {
        1.0
    };
    (rel, cos)
}

struct Geo {
    name: &'static str,
    h: usize,
    nope: usize,
    tail: usize,
    v: usize,
    latent: usize,
}

struct Tally {
    checks: usize,
    fails: usize,
}
impl Tally {
    fn check(&mut self, ok: bool, what: impl AsRef<str>) {
        self.checks += 1;
        if !ok {
            self.fails += 1;
            println!("FAIL {}", what.as_ref());
        }
    }
}

/// Weights: `w` row-major `[h*(nope+v), latent]` F32 (random, not BF16-valued);
/// returns (`w_t` plane `[latent, ncols]`, host-packed BF16 expectation, BF16-valued W).
fn make_weights(g: &Geo, rng: &mut Rng) -> (Vec<f32>, Vec<u16>, Vec<f32>) {
    let ncols = g.h * (g.nope + g.v);
    let w: Vec<f32> = (0..ncols * g.latent).map(|_| rng.unit() * 0.7).collect();
    let mut w_t = vec![0.0f32; ncols * g.latent];
    for col in 0..ncols {
        for c in 0..g.latent {
            w_t[c * ncols + col] = w[col * g.latent + c];
        }
    }
    let wb: Vec<f32> = w.iter().map(|f| bf16_val(*f)).collect();
    // Independent host re-packing: per head [nope][latent] then [latent][v].
    let mut packed = Vec::with_capacity(ncols * g.latent);
    for h in 0..g.h {
        for d in 0..g.nope {
            for c in 0..g.latent {
                packed.push(bf16_bits(w[(h * (g.nope + g.v) + d) * g.latent + c]));
            }
        }
        for c in 0..g.latent {
            for e in 0..g.v {
                packed.push(bf16_bits(
                    w[(h * (g.nope + g.v) + g.nope + e) * g.latent + c],
                ));
            }
        }
    }
    (w_t, packed, wb)
}

fn sample_rows(rows: usize) -> Vec<usize> {
    if rows <= 512 {
        (0..rows).collect()
    } else {
        let mut v: Vec<usize> = [0, 1, 7, 8, 9, 31, 32, 33, 2047, 2048, 4000, rows - 1]
            .iter()
            .copied()
            .filter(|r| *r < rows)
            .collect();
        v.sort();
        v.dedup();
        v
    }
}

fn ref_q(g: &Geo, wb: &[f32], q: &[f32], sel: &[usize]) -> Vec<Vec<f64>> {
    let xw = g.nope + g.tail;
    sel.par_iter()
        .map(|&r| {
            let mut out = vec![0.0f64; g.h * g.latent];
            for h in 0..g.h {
                let acc = &mut out[h * g.latent..(h + 1) * g.latent];
                for d in 0..g.nope {
                    let qd = q[(r * g.h + h) * xw + d] as f64;
                    let wrow = &wb[(h * (g.nope + g.v) + d) * g.latent..][..g.latent];
                    for c in 0..g.latent {
                        acc[c] += wrow[c] as f64 * qd;
                    }
                }
            }
            out
        })
        .collect()
}

fn ref_v(g: &Geo, wb: &[f32], z: &[f32], sel: &[usize]) -> Vec<Vec<f64>> {
    sel.par_iter()
        .map(|&r| {
            let mut out = vec![0.0f64; g.h * g.v];
            for h in 0..g.h {
                let zr = &z[(r * g.h + h) * g.latent..][..g.latent];
                for e in 0..g.v {
                    let wrow = &wb[(h * (g.nope + g.v) + g.nope + e) * g.latent..][..g.latent];
                    let mut s = 0.0f64;
                    for c in 0..g.latent {
                        s += wrow[c] as f64 * zr[c] as f64;
                    }
                    out[h * g.v + e] = s;
                }
            }
            out
        })
        .collect()
}

fn run_geometry(gpu: &mut Gpu, g: &Geo, rows_list: &[usize], t: &mut Tally, seed: u64) {
    let mut rng = Rng(seed);
    let (w_t, packed_host, wb) = make_weights(g, &mut rng);
    let elems = latent_attention_absorb_packed_elems(g.h, g.nope, g.v, g.latent);
    t.check(
        elems == packed_host.len(),
        format!("{} packed_elems", g.name),
    );

    // ---- pack: byte-exact ----
    let w_t_dev = up_f32(gpu, &w_t);
    let packed_dev = up_bf16(gpu, &vec![0xdeadu16; elems], true);
    gpu.latent_attention_absorb_pack(&w_t_dev, &packed_dev, g.h, g.nope, g.v, g.latent)
        .expect("pack");
    gpu.hip.device_synchronize().expect("sync");
    let raw = gpu.download_raw_bytes(&packed_dev).expect("dl packed");
    let got: Vec<u16> = raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    t.check(
        got[..elems] == packed_host[..],
        format!("{} pack bytes", g.name),
    );
    t.check(
        (0..GUARD).all(|i| got[elems + i] == if i % 2 == 0 { 0x7fc0 } else { 0x7149 }),
        format!("{} pack guard (written past packed_elems)", g.name),
    );
    // Re-upload packed with a poisoned guard for the matrix kernels.
    let packed_dev = up_bf16(gpu, &packed_host, true);

    for &rows in rows_list {
        let sel = sample_rows(rows);
        let xw = g.nope + g.tail;
        let ow = g.latent + g.tail;
        // BF16-valued activations.
        let q: Vec<f32> = (0..rows * g.h * xw).map(|_| bf16_val(rng.unit())).collect();
        let z: Vec<f32> = (0..rows * g.h * g.latent)
            .map(|_| bf16_val(rng.unit()))
            .collect();
        let q_dev = up_f32(gpu, &q);
        let z_dev = up_f32(gpu, &z);

        // ---- absorb_q ----
        let nq = rows * g.h * ow;
        let qa0 = out_f32(gpu, nq, NAN_BITS);
        gpu.latent_attention_absorb_q(
            &q_dev,
            &packed_dev,
            &qa0,
            rows,
            g.h,
            g.nope,
            g.tail,
            g.v,
            g.latent,
        )
        .expect("absorb_q");
        let a = dl_f32(gpu, &qa0);
        let qa1 = out_f32(gpu, nq, BIG_BITS);
        gpu.latent_attention_absorb_q(
            &q_dev,
            &packed_dev,
            &qa1,
            rows,
            g.h,
            g.nope,
            g.tail,
            g.v,
            g.latent,
        )
        .expect("absorb_q repeat");
        let a1 = dl_f32(gpu, &qa1);
        let tag = format!("{} rows={rows}", g.name);
        t.check(
            guard_ok(&a, nq) && guard_ok(&a1, nq),
            format!("{tag} q_abs guard"),
        );
        t.check(
            a[..nq]
                .iter()
                .zip(&a1[..nq])
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            format!("{tag} q_abs repeat byte-identical (different poison)"),
        );
        t.check(
            a[..nq].iter().all(|x| x.is_finite()),
            format!("{tag} q_abs fully written"),
        );
        // tail copy exact
        let mut tail_ok = true;
        for r in 0..rows {
            for h in 0..g.h {
                for j in 0..g.tail {
                    let src = q[(r * g.h + h) * xw + g.nope + j];
                    let dst = a[(r * g.h + h) * ow + g.latent + j];
                    tail_ok &= src.to_bits() == dst.to_bits();
                }
            }
        }
        t.check(tail_ok, format!("{tag} q tail copy exact"));
        let want = ref_q(g, &wb, &q, &sel);
        let (mut got_all, mut want_all) = (Vec::new(), Vec::new());
        for (i, &r) in sel.iter().enumerate() {
            for h in 0..g.h {
                got_all.extend_from_slice(&a[(r * g.h + h) * ow..][..g.latent]);
                want_all.extend_from_slice(&want[i][h * g.latent..(h + 1) * g.latent]);
            }
        }
        let (rel, cos) = stats(&got_all, &want_all);
        println!("  {tag} absorb_q  rel_l2={rel:.3e} cos={cos:.8}");
        t.check(
            rel <= 1e-4 && cos >= 0.99999,
            format!("{tag} absorb_q band"),
        );

        // ---- host algebra: (qabs . c) + qt.kpe == qn . (Ak c) + qt.kpe ----
        {
            let mut alg_rng = Rng(seed ^ 0xa5a5 ^ rows as u64);
            let mut worst = 0.0f64;
            for _ in 0..64 {
                let r = sel[(alg_rng.next() as usize) % sel.len()];
                let h = (alg_rng.next() as usize) % g.h;
                let crow: Vec<f32> = (0..g.latent).map(|_| bf16_val(alg_rng.unit())).collect();
                let kpe: Vec<f32> = (0..g.tail).map(|_| bf16_val(alg_rng.unit())).collect();
                let qa = &a[(r * g.h + h) * ow..][..ow];
                let qh = &q[(r * g.h + h) * xw..][..xw];
                let lhs: f64 = (0..g.latent)
                    .map(|c| qa[c] as f64 * crow[c] as f64)
                    .sum::<f64>()
                    + (0..g.tail)
                        .map(|j| qa[g.latent + j] as f64 * kpe[j] as f64)
                        .sum::<f64>();
                let mut rhs = 0.0f64;
                let mut mag = 0.0f64;
                for d in 0..g.nope {
                    let wrow = &wb[(h * (g.nope + g.v) + d) * g.latent..][..g.latent];
                    let ak_c: f64 = (0..g.latent).map(|c| wrow[c] as f64 * crow[c] as f64).sum();
                    rhs += qh[d] as f64 * ak_c;
                    mag += (qh[d] as f64 * ak_c).abs();
                }
                for j in 0..g.tail {
                    rhs += qh[g.nope + j] as f64 * kpe[j] as f64;
                    mag += (qh[g.nope + j] as f64 * kpe[j] as f64).abs();
                }
                worst = worst.max((lhs - rhs).abs() / mag.max(1e-30));
            }
            println!("  {tag} score algebra worst |dlhs-rhs|/sum|terms| = {worst:.3e}");
            t.check(worst <= 1e-5, format!("{tag} score algebra equality"));
        }

        // ---- absorb_v (round off / on) ----
        let ny = rows * g.h * g.v;
        let want = ref_v(g, &wb, &z, &sel);
        let mut want_all = Vec::new();
        for (i, _) in sel.iter().enumerate() {
            want_all.extend_from_slice(&want[i]);
        }
        for round in [false, true] {
            let y0 = out_f32(gpu, ny, NAN_BITS);
            gpu.latent_attention_absorb_v(
                &z_dev,
                &packed_dev,
                &y0,
                rows,
                g.h,
                g.nope,
                g.v,
                g.latent,
                round,
            )
            .expect("absorb_v");
            let y = dl_f32(gpu, &y0);
            let y1 = out_f32(gpu, ny, BIG_BITS);
            gpu.latent_attention_absorb_v(
                &z_dev,
                &packed_dev,
                &y1,
                rows,
                g.h,
                g.nope,
                g.v,
                g.latent,
                round,
            )
            .expect("absorb_v repeat");
            let yb = dl_f32(gpu, &y1);
            let tag = format!("{tag} v round={round}");
            t.check(
                guard_ok(&y, ny) && guard_ok(&yb, ny),
                format!("{tag} y guard"),
            );
            t.check(
                y[..ny]
                    .iter()
                    .zip(&yb[..ny])
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                format!("{tag} repeat byte-identical (different poison)"),
            );
            t.check(
                y[..ny].iter().all(|x| x.is_finite()),
                format!("{tag} y fully written"),
            );
            let mut got_all = Vec::new();
            for &r in &sel {
                got_all.extend_from_slice(&y[r * g.h * g.v..(r + 1) * g.h * g.v]);
            }
            let (rel, cos) = stats(&got_all, &want_all);
            println!("  {tag} absorb_v  rel_l2={rel:.3e} cos={cos:.8}");
            if round {
                t.check(rel <= 1e-2 && cos >= 0.9999, format!("{tag} band"));
                t.check(
                    y[..ny].iter().all(|x| x.to_bits() & 0xffff == 0),
                    format!("{tag} outputs are BF16-representable"),
                );
            } else {
                t.check(rel <= 1e-4 && cos >= 0.99999, format!("{tag} band"));
            }
        }
    }

    // ---- roundtrip identity: pack -> v with one-hot z returns Av[:, c] exactly ----
    for rows in [1usize, 2, 9, 17] {
        let mut z = vec![0.0f32; rows * g.h * g.latent];
        let mut hot = vec![0usize; rows * g.h];
        for r in 0..rows {
            for h in 0..g.h {
                let c = (r * 7 + h * 3 + 1) % g.latent;
                hot[r * g.h + h] = c;
                z[(r * g.h + h) * g.latent + c] = 1.0;
            }
        }
        let z_dev = up_f32(gpu, &z);
        let ny = rows * g.h * g.v;
        for round in [false, true] {
            let y0 = out_f32(gpu, ny, NAN_BITS);
            gpu.latent_attention_absorb_v(
                &z_dev,
                &packed_dev,
                &y0,
                rows,
                g.h,
                g.nope,
                g.v,
                g.latent,
                round,
            )
            .expect("absorb_v onehot");
            let y = dl_f32(gpu, &y0);
            let mut exact = guard_ok(&y, ny);
            for r in 0..rows {
                for h in 0..g.h {
                    let c = hot[r * g.h + h];
                    for e in 0..g.v {
                        let want = wb[(h * (g.nope + g.v) + g.nope + e) * g.latent + c];
                        exact &= y[(r * g.h + h) * g.v + e] == want;
                    }
                }
            }
            t.check(
                exact,
                format!("{} roundtrip one-hot rows={rows} round={round}", g.name),
            );
        }
    }
}

fn expect_err(t: &mut Tally, what: &str, r: Result<(), hip_bridge::HipError>) {
    match r {
        Err(e) => {
            println!("  refused ({what}): {e:?}");
            t.check(true, what);
        }
        Ok(()) => t.check(false, format!("{what}: expected a refusal")),
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch {}", gpu.arch);
    let mut t = Tally {
        checks: 0,
        fails: 0,
    };

    let geos: [(Geo, &[usize]); 4] = [
        (
            Geo {
                name: "raon",
                h: 40,
                nope: 128,
                tail: 64,
                v: 128,
                latent: 512,
            },
            &[1, 2, 15, 16, 17, 33, 64, 129, 512, 4096],
        ),
        (
            Geo {
                name: "H8-fixture",
                h: 8,
                nope: 64,
                tail: 32,
                v: 96,
                latent: 256,
            },
            &[1, 2, 8, 9, 17, 64, 129, 512],
        ),
        (
            Geo {
                name: "H3-w32-v48",
                h: 3,
                nope: 32,
                tail: 16,
                v: 48,
                latent: 64,
            },
            &[1, 2, 8, 9, 15, 16, 17, 33, 64, 65, 129, 512],
        ),
        (
            Geo {
                name: "H2-odd-tail",
                h: 2,
                nope: 16,
                tail: 6,
                v: 24,
                latent: 40,
            },
            &[1, 3, 8, 9, 33, 129],
        ),
    ];
    for (i, (g, rows)) in geos.iter().enumerate() {
        println!(
            "geometry {} (H={} nope={} tail={} v={} latent={})",
            g.name, g.h, g.nope, g.tail, g.v, g.latent
        );
        run_geometry(
            &mut gpu,
            g,
            rows,
            &mut t,
            0x1234_5678_9abc_def1 ^ ((i as u64 + 1) * 0x9e37),
        );
    }

    // ---- refusals ----
    {
        let w = up_f32(&mut gpu, &vec![0.0; 64 * 64]);
        let pk = up_bf16(&mut gpu, &vec![0u16; 64 * 64], false);
        expect_err(
            &mut t,
            "pack nope%8",
            gpu.latent_attention_absorb_pack(&w, &pk, 1, 12, 8, 64),
        );
        expect_err(
            &mut t,
            "pack latent=0",
            gpu.latent_attention_absorb_pack(&w, &pk, 1, 8, 8, 0),
        );
        expect_err(
            &mut t,
            "pack too-small packed",
            gpu.latent_attention_absorb_pack(&w, &pk, 5, 8, 8, 64),
        );
        let q = up_f32(&mut gpu, &vec![0.0; 1024]);
        let out = up_f32(&mut gpu, &vec![0.0; 1024]);
        expect_err(
            &mut t,
            "q v%8",
            gpu.latent_attention_absorb_q(&q, &pk, &out, 1, 1, 8, 8, 9, 64),
        );
        expect_err(
            &mut t,
            "v wrong dtype",
            gpu.latent_attention_absorb_v(&q, &pk, &pk, 1, 1, 8, 8, 64, false),
        );
        expect_err(
            &mut t,
            "q too-small q_abs",
            gpu.latent_attention_absorb_q(&q, &pk, &out, 200, 1, 8, 8, 8, 64),
        );
    }

    println!("checks={} fails={}", t.checks, t.fails);
    if t.fails != 0 {
        std::process::exit(1);
    }
    println!("PASS");
}
