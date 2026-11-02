// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness check for `Gpu::latent_attention_flash_prefill` against an
//! independent host reference (F64 softmax/accumulation over BF16-valued
//! inputs from a fixed-seed PRNG).
//!
//! Geometries: Raon-like decompressed (192/128, token-major mirrors), absorbed
//! (wa=512, wb=64, v=512, head_stride 0), a non-Raon strided shape, and the
//! smaller oracle-fixture / multi-chunk shapes. Swept: `*pos`, `pos_offset`,
//! rows, `max_rows` truncation, rows with no visible key (negative `*pos`),
//! `round_y_bf16` on/off, poisoned memory beyond the visible range (and in
//! every unused gap of a view), byte-identical repeats, and byte-identical
//! results for a different non-truncating `max_rows`.
//!
//! On a gfx12 device the shipped kernel (gfx12 fragment maps) is checked, then
//! the gfx11 fragment maps are exercised through the kernel's `LAF_EMU_GFX11`
//! fixture build (gfx11 operand layouts driven by gfx12 matrix ops + lane
//! exchanges); that is layout evidence only, not a gfx11 execution.
//!
//! Usage: latent_attention_flash_check [--quick] [--no-emu]
//! Bands (BF16 matrix-unit route): rel_l2 <= 1e-2, cosine >= 0.9999.

use rayon::prelude::*;
use rdna_compute::latent_attention_flash::LATENT_ATTENTION_FLASH_SRC;
use rdna_compute::latent_attention_tiled::{LatentRows, TiledAttention};
use rdna_compute::{DType, Gpu, GpuTensor};

const REL_L2_MAX: f64 = 1e-2;
const COS_MIN: f64 = 0.9999;
const POISON_BIG: u16 = 0x7149; // ~1e30-class
const POISON_NAN: u16 = 0x7fc1;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bf16_rne(f: f32) -> u16 {
    let u = f.to_bits();
    ((u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16) as u16
}
fn bf16_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[derive(Clone, Copy)]
struct ViewSpec {
    buf: usize,
    off: usize,
    tok: usize,
    head: usize,
    width: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Role {
    K,
    V,
    Shared,
}

struct Geo {
    name: &'static str,
    h: usize,
    /// (token stride in elements, role) per buffer.
    bufs: Vec<(usize, Role)>,
    ka: ViewSpec,
    kb: ViewSpec,
    v: ViewSpec,
}

impl Geo {
    fn dq(&self) -> usize {
        self.ka.width + self.kb.width
    }
}

fn geo_raon(h: usize, name: &'static str) -> Geo {
    Geo {
        name,
        h,
        bufs: vec![(h * 128, Role::K), (576, Role::K), (h * 128, Role::V)],
        ka: ViewSpec {
            buf: 0,
            off: 0,
            tok: h * 128,
            head: 128,
            width: 128,
        },
        kb: ViewSpec {
            buf: 1,
            off: 512,
            tok: 576,
            head: 0,
            width: 64,
        },
        v: ViewSpec {
            buf: 2,
            off: 0,
            tok: h * 128,
            head: 128,
            width: 128,
        },
    }
}

fn geo_absorbed(h: usize, latent: usize, tail: usize, name: &'static str) -> Geo {
    let ts = latent + tail;
    Geo {
        name,
        h,
        bufs: vec![(ts, Role::Shared)],
        ka: ViewSpec {
            buf: 0,
            off: 0,
            tok: ts,
            head: 0,
            width: latent,
        },
        kb: ViewSpec {
            buf: 0,
            off: latent,
            tok: ts,
            head: 0,
            width: tail,
        },
        v: ViewSpec {
            buf: 0,
            off: 0,
            tok: ts,
            head: 0,
            width: latent,
        },
    }
}

/// Strided, padded, non-Raon shape: every view has an offset, token stride and
/// head stride with unused gaps.
fn geo_generic(name: &'static str, h: usize, wa: usize, wb: usize, dv: usize) -> Geo {
    let (ha, hv) = (wa + 8, dv + 8);
    Geo {
        name,
        h,
        bufs: vec![
            (h * ha + 16, Role::K),
            (wb + 24, Role::K),
            (h * hv + 8, Role::V),
        ],
        ka: ViewSpec {
            buf: 0,
            off: 8,
            tok: h * ha + 16,
            head: ha,
            width: wa,
        },
        kb: ViewSpec {
            buf: 1,
            off: 16,
            tok: wb + 24,
            head: 0,
            width: wb,
        },
        v: ViewSpec {
            buf: 2,
            off: 8,
            tok: h * hv + 8,
            head: hv,
            width: dv,
        },
    }
}

fn geo_oracle() -> Geo {
    // H=8, nope=64, tail=32, v=96, latent=256 fixtures.
    let h = 8;
    Geo {
        name: "oracle H8 64+32/96",
        h,
        bufs: vec![(h * 64, Role::K), (256 + 32, Role::K), (h * 96, Role::V)],
        ka: ViewSpec {
            buf: 0,
            off: 0,
            tok: h * 64,
            head: 64,
            width: 64,
        },
        kb: ViewSpec {
            buf: 1,
            off: 256,
            tok: 288,
            head: 0,
            width: 32,
        },
        v: ViewSpec {
            buf: 2,
            off: 0,
            tok: h * 96,
            head: 96,
            width: 96,
        },
    }
}

struct Case {
    pos: i32,
    po: usize,
    rows: usize,
    max_rows: Option<usize>,
    amp: f32,
}

struct HostBufs {
    bits: Vec<Vec<u16>>,
    f: Vec<Vec<f32>>,
}

fn view_len_needed(v: &ViewSpec, h: usize, rows: usize) -> usize {
    v.off + (rows - 1) * v.tok + (h - 1) * v.head + v.width
}

fn build_bufs(geo: &Geo, cap: usize, vis_end: usize, mode: usize, rng: &mut Rng) -> HostBufs {
    let mut bits: Vec<Vec<u16>> = geo
        .bufs
        .iter()
        .enumerate()
        .map(|(bi, &(ts, _))| {
            let mut need = cap * ts;
            for v in [&geo.ka, &geo.kb, &geo.v] {
                if v.buf == bi {
                    assert_eq!(v.tok, ts);
                    need = need.max(view_len_needed(v, geo.h, cap));
                }
            }
            vec![POISON_NAN; need + 64]
        })
        .collect();
    // Visible rows: random BF16 in the viewed elements; gaps stay NaN.
    for v in [&geo.ka, &geo.kb, &geo.v] {
        for t in 0..vis_end {
            for hd in 0..geo.h {
                for d in 0..v.width {
                    let idx = v.off + t * v.tok + hd * v.head + d;
                    if bits[v.buf][idx] == POISON_NAN {
                        bits[v.buf][idx] = bf16_rne(rng.unit());
                    }
                }
            }
        }
    }
    // Rows beyond the visible range: poison everything in the row.
    for (bi, &(ts, role)) in geo.bufs.iter().enumerate() {
        let (kp, vp) = if mode == 0 {
            (POISON_BIG, POISON_NAN)
        } else {
            (POISON_NAN, POISON_BIG)
        };
        for t in vis_end..cap {
            let lo = t * ts;
            let hi = ((t + 1) * ts).min(bits[bi].len());
            for (e, slot) in bits[bi][lo..hi].iter_mut().enumerate() {
                *slot = match role {
                    Role::K => kp,
                    Role::V => vp,
                    Role::Shared => {
                        if e % 2 == 0 {
                            kp
                        } else {
                            vp
                        }
                    }
                };
            }
        }
    }
    let f = bits
        .iter()
        .map(|b| b.iter().map(|&x| bf16_f32(x)).collect())
        .collect();
    HostBufs { bits, f }
}

/// F64 host reference, `[rows][H][dv]`.
fn reference(
    geo: &Geo,
    hb: &HostBufs,
    q: &[f32],
    rows: usize,
    pos: i32,
    po: usize,
    max_rows: usize,
    scale: f32,
) -> Vec<f64> {
    let (h, dq, dv) = (geo.h, geo.dq(), geo.v.width);
    let mut y = vec![0.0f64; rows * h * dv];
    y.par_chunks_mut(dv).enumerate().for_each(|(ih, out)| {
        let (i, hd) = (ih / h, ih % h);
        let lim = (pos as i64 + po as i64 + i as i64).min(max_rows as i64 - 1);
        if lim < 0 {
            return;
        }
        let qrow = &q[ih * dq..(ih + 1) * dq];
        let n = lim as usize + 1;
        let mut s = vec![0.0f64; n];
        for (j, sj) in s.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (view, qo) in [(&geo.ka, 0usize), (&geo.kb, geo.ka.width)] {
                let base = view.off + j * view.tok + hd * view.head;
                let row = &hb.f[view.buf][base..base + view.width];
                for (d, &kv) in row.iter().enumerate() {
                    acc += qrow[qo + d] as f64 * kv as f64;
                }
            }
            *sj = acc * scale as f64;
        }
        let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut l = 0.0f64;
        for sj in s.iter_mut() {
            *sj = (*sj - mx).exp();
            l += *sj;
        }
        for (j, &p) in s.iter().enumerate() {
            let base = geo.v.off + j * geo.v.tok + hd * geo.v.head;
            let row = &hb.f[geo.v.buf][base..base + dv];
            for (d, &vv) in row.iter().enumerate() {
                out[d] += p * vv as f64;
            }
        }
        for o in out.iter_mut() {
            *o /= l;
        }
    });
    y
}

fn bf16_tensor(gpu: &mut Gpu, bits: &[u16]) -> GpuTensor {
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let mut t = gpu.upload_raw(&bytes, &[bits.len()]).expect("bf16 upload");
    t.dtype = DType::BF16;
    t.shape = vec![bits.len()];
    t
}

struct Metrics {
    rel_l2: f64,
    cos: f64,
    max_abs: f64,
}

fn metrics(got: &[f32], want: &[f64]) -> Metrics {
    let (mut dd, mut ww, mut gg, mut gw, mut mx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want) {
        let g = g as f64;
        dd += (g - w) * (g - w);
        ww += w * w;
        gg += g * g;
        gw += g * w;
        mx = mx.max((g - w).abs());
    }
    if ww == 0.0 {
        return Metrics {
            rel_l2: dd.sqrt(),
            cos: if gg == 0.0 { 1.0 } else { 0.0 },
            max_abs: mx,
        };
    }
    let cos = if gg == 0.0 {
        0.0
    } else {
        gw / (gg.sqrt() * ww.sqrt())
    };
    Metrics {
        rel_l2: (dd / ww).sqrt(),
        cos,
        max_abs: mx,
    }
}

struct Runner<'a> {
    gpu: &'a mut Gpu,
    emu: bool,
}

impl Runner<'_> {
    fn launch(&mut self, a: &TiledAttention<'_>) {
        if self.emu {
            let src = format!(
                "#define LAF_EMU_GFX11 1\n#define latent_attention_flash_prefill \
                 latent_attention_flash_prefill_emu11\n#define latent_attention_flash_prefill_wide \
                 latent_attention_flash_prefill_emu11_wide\n{LATENT_ATTENTION_FLASH_SRC}"
            );
            self.gpu
                .latent_attention_flash_prefill_with_source(
                    a,
                    "latent_attention_flash_emu11",
                    "latent_attention_flash_prefill_emu11",
                    &src,
                )
                .expect("emu launch");
        } else {
            self.gpu.latent_attention_flash_prefill(a).expect("launch");
        }
    }
}

struct Totals {
    cases: usize,
    launches: usize,
    fails: Vec<String>,
    worst_rel: f64,
    worst_cos: f64,
    worst_abs: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_case(r: &mut Runner<'_>, geo: &Geo, c: &Case, idx: usize, seed: u64, tot: &mut Totals) {
    let (h, dq, dv) = (geo.h, geo.dq(), geo.v.width);
    let scale = (dq as f32).powf(-0.5);
    let reach = (c.pos as i64 + c.po as i64 + c.rows as i64).max(0) as usize;
    let (max_rows_main, truncating) = match c.max_rows {
        Some(m) => (m, m < reach),
        None => (reach.max(1) + 37, false),
    };
    let vis_end = reach.min(max_rows_main);
    let cap = max_rows_main
        + if truncating || c.max_rows.is_some() {
            37
        } else {
            0
        };
    let mut rng = Rng(seed ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let hb = build_bufs(geo, cap, vis_end, idx % 2, &mut rng);
    let qv: Vec<f32> = (0..c.rows * h * dq)
        .map(|_| bf16_f32(bf16_rne(rng.unit() * c.amp)))
        .collect();
    let want = reference(geo, &hb, &qv, c.rows, c.pos, c.po, max_rows_main, scale);
    let label = format!(
        "{} pos={} po={} rows={} max_rows={:?} amp={}",
        geo.name, c.pos, c.po, c.rows, c.max_rows, c.amp
    );

    let bufs: Vec<GpuTensor> = hb.bits.iter().map(|b| bf16_tensor(r.gpu, b)).collect();
    let d_q = r.gpu.upload_f32(&qv, &[qv.len()]).unwrap();
    let mut d_pos = r.gpu.upload_raw(&c.pos.to_ne_bytes(), &[4]).unwrap();
    d_pos.dtype = DType::F32;
    d_pos.shape = vec![1];

    let view = |v: &ViewSpec| LatentRows {
        buf: &bufs[v.buf],
        offset: v.off,
        tok_stride: v.tok,
        head_stride: v.head,
        width: v.width,
    };
    let run = |r: &mut Runner<'_>, max_rows: usize, round: bool, tot: &mut Totals| -> Vec<f32> {
        let n = c.rows * h * dv;
        let d_y = r.gpu.upload_f32(&vec![f32::NAN; n], &[n]).unwrap();
        let a = TiledAttention {
            q: &d_q,
            y: &d_y,
            k_a: view(&geo.ka),
            k_b: view(&geo.kb),
            v: view(&geo.v),
            pos: &d_pos,
            pos_offset: c.po,
            n_rows: c.rows,
            n_heads: h,
            scale,
            max_rows,
            round_y_bf16: round,
        };
        r.launch(&a);
        tot.launches += 1;
        let y = r.gpu.download_f32(&d_y).unwrap();
        r.gpu.free_tensor(d_y).unwrap();
        y
    };

    let note = |tot: &mut Totals, msg: String| {
        tot.fails.push(format!("{label}: {msg}"));
    };
    let bytes =
        |y: &[f32]| -> Vec<u8> { y.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect() };

    tot.cases += 1;
    let mut plain: Option<Vec<f32>> = None;
    for round in [false, true] {
        let y1 = run(r, max_rows_main, round, tot);
        let y2 = run(r, max_rows_main, round, tot);
        if bytes(&y1) != bytes(&y2) {
            note(
                tot,
                format!("round={round}: repeat run is not byte-identical"),
            );
        }
        if !truncating {
            let y3 = run(r, vis_end.max(1), round, tot);
            if bytes(&y1) != bytes(&y3) {
                note(
                    tot,
                    format!(
                        "round={round}: bytes differ for max_rows={}",
                        vis_end.max(1)
                    ),
                );
            }
        }
        if y1.iter().any(|v| !v.is_finite()) {
            let bad = y1.iter().filter(|v| !v.is_finite()).count();
            let first = y1.iter().position(|v| !v.is_finite()).unwrap_or(0);
            note(
                tot,
                format!(
                    "round={round}: non-finite/unwritten outputs ({bad} of {}, first at {first} = row {} head {} col {})",
                    y1.len(),
                    first / (geo.h * dv),
                    first / dv % geo.h,
                    first % dv
                ),
            );
            continue;
        }
        // Rows with no visible key are exact zeros.
        for i in 0..c.rows {
            let lim = (c.pos as i64 + c.po as i64 + i as i64).min(max_rows_main as i64 - 1);
            if lim < 0 && y1[i * h * dv..(i + 1) * h * dv].iter().any(|&v| v != 0.0) {
                note(
                    tot,
                    format!("round={round}: row {i} has no visible key but is nonzero"),
                );
                break;
            }
        }
        let m = metrics(&y1, &want);
        if m.rel_l2 > REL_L2_MAX || m.cos < COS_MIN {
            note(
                tot,
                format!(
                    "round={round}: rel_l2={:.3e} cos={:.7} max_abs={:.3e}",
                    m.rel_l2, m.cos, m.max_abs
                ),
            );
        }
        if !round {
            tot.worst_rel = tot.worst_rel.max(m.rel_l2);
            tot.worst_cos = tot.worst_cos.min(m.cos);
            tot.worst_abs = tot.worst_abs.max(m.max_abs);
            plain = Some(y1);
        } else {
            if y1.iter().any(|v| v.to_bits() & 0xffff != 0) {
                note(tot, "round=true: output is not BF16-valued".into());
            }
            if let Some(p) = &plain {
                let want_bits: Vec<u32> = p.iter().map(|&v| (bf16_rne(v) as u32) << 16).collect();
                let got_bits: Vec<u32> = y1.iter().map(|v| v.to_bits()).collect();
                if want_bits != got_bits {
                    note(
                        tot,
                        "round=true differs from bf16(round=false) output".into(),
                    );
                }
            }
        }
    }
    for t in bufs {
        r.gpu.free_tensor(t).unwrap();
    }
    r.gpu.free_tensor(d_q).unwrap();
    r.gpu.free_tensor(d_pos).unwrap();
}

fn cases_for(quick: bool) -> Vec<Case> {
    let pos_list: &[i32] = if quick {
        &[0, 17, 64, 200]
    } else {
        &[0, 1, 15, 16, 17, 62, 63, 64, 65, 126, 127, 128, 200, 1000]
    };
    let rows_list: &[usize] = if quick {
        &[1, 16, 33, 129]
    } else {
        &[1, 2, 15, 16, 17, 33, 64, 65, 129]
    };
    let mut v = Vec::new();
    for &pos in pos_list {
        for &rows in rows_list {
            v.push(Case {
                pos,
                po: 0,
                rows,
                max_rows: None,
                amp: 1.0,
            });
        }
    }
    // pos_offset != 0
    for &(pos, po, rows) in &[
        (0, 17, 33),
        (15, 64, 65),
        (200, 33, 129),
        (1000, 5, 17),
        (0, 1, 1),
        (7, 0, 16),
    ] {
        v.push(Case {
            pos,
            po,
            rows,
            max_rows: None,
            amp: 1.0,
        });
    }
    // max_rows truncation (and max_rows below the call's first row).
    for &(pos, po, rows, mr) in &[
        (100, 0, 65, 130),
        (0, 0, 129, 70),
        (200, 0, 33, 20),
        (64, 0, 64, 100),
        (30, 7, 17, 40),
        (0, 0, 16, 16),
    ] {
        v.push(Case {
            pos,
            po,
            rows,
            max_rows: Some(mr),
            amp: 1.0,
        });
    }
    // Rows with no visible key.
    for &(pos, rows) in &[(-3, 33), (-70, 65), (-20, 65), (-1, 1), (-16, 17)] {
        v.push(Case {
            pos,
            po: 0,
            rows,
            max_rows: None,
            amp: 1.0,
        });
    }
    // Peaked attention.
    v.push(Case {
        pos: 200,
        po: 0,
        rows: 65,
        max_rows: None,
        amp: 6.0,
    });
    v.push(Case {
        pos: 17,
        po: 0,
        rows: 33,
        max_rows: None,
        amp: 6.0,
    });
    v
}

fn run_suite(r: &mut Runner<'_>, quick: bool, tag: &str) -> bool {
    let mut geos: Vec<Geo> = vec![
        geo_raon(4, "raon-like H4 128+64/128"),
        geo_absorbed(4, 512, 64, "absorbed H4 512+64/512 head_stride 0"),
        geo_generic("generic H3 32+16/48", 3, 32, 16, 48),
        geo_oracle(),
        geo_generic(
            "multi-chunk H2 48+16/192 (wide kernel, 4 x 48 cols)",
            2,
            48,
            16,
            192,
        ),
        geo_generic(
            "multi-chunk H2 48+16/160 (chunked kernel, 128 + 32 cols)",
            2,
            48,
            16,
            160,
        ),
        geo_generic(
            "wide H2 64+64/512 (wide kernel, 4 x 128 cols)",
            2,
            64,
            64,
            512,
        ),
        geo_absorbed(2, 256, 32, "absorbed-small H2 256+32/256"),
    ];
    if !quick {
        geos.push(geo_raon(40, "raon-like H40 128+64/128"));
    }
    let mut all_ok = true;
    for (gi, geo) in geos.iter().enumerate() {
        let cases = if geo.h == 40 {
            vec![
                Case {
                    pos: 1000,
                    po: 0,
                    rows: 65,
                    max_rows: None,
                    amp: 1.0,
                },
                Case {
                    pos: 0,
                    po: 0,
                    rows: 129,
                    max_rows: None,
                    amp: 1.0,
                },
                Case {
                    pos: 200,
                    po: 17,
                    rows: 33,
                    max_rows: Some(150),
                    amp: 1.0,
                },
            ]
        } else {
            cases_for(quick)
        };
        let mut tot = Totals {
            cases: 0,
            launches: 0,
            fails: vec![],
            worst_rel: 0.0,
            worst_cos: 1.0,
            worst_abs: 0.0,
        };
        for (ci, c) in cases.iter().enumerate() {
            run_case(r, geo, c, ci, 0xA5A5_0000 + gi as u64, &mut tot);
        }
        println!(
            "[{tag}] {:<42} cases={:>3} launches={:>4} worst rel_l2={:.3e} cos={:.7} max_abs={:.3e} {}",
            geo.name,
            tot.cases,
            tot.launches,
            tot.worst_rel,
            tot.worst_cos,
            tot.worst_abs,
            if tot.fails.is_empty() { "PASS" } else { "FAIL" }
        );
        for f in &tot.fails {
            println!("    FAIL {f}");
        }
        all_ok &= tot.fails.is_empty();
    }
    all_ok
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let no_emu = args.iter().any(|a| a == "--no-emu");
    let mut gpu = Gpu::init().expect("GPU init");
    println!("GPU arch: {}", gpu.arch);
    let is_gfx12 = gpu.arch.starts_with("gfx12");
    let mut ok = true;
    {
        let mut r = Runner {
            gpu: &mut gpu,
            emu: false,
        };
        ok &= run_suite(&mut r, quick, if is_gfx12 { "gfx12" } else { "native" });
    }
    if is_gfx12 && !no_emu {
        let mut r = Runner {
            gpu: &mut gpu,
            emu: true,
        };
        ok &= run_suite(&mut r, true, "gfx11-layout-emu");
    }
    println!("{}", if ok { "ALL PASS" } else { "FAILURES" });
    if !ok {
        std::process::exit(1);
    }
}
