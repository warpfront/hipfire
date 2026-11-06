// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Gpu::latent_attention_decode_split_rows{,_q8}` K = 1..=16 oracle: every
//! output row of a K-row call (one launch pair, grid z = row) must be
//! BIT-IDENTICAL (`f32::to_bits`) to the rows == 1 split-KV decode call
//! (`latent_attention_decode_split{,_q8}`) made at that row's length
//! `L + r + 1` (device scalar `*pos = L + r`, `pos_offset = 0`, i.e. exactly what
//! a decode step at that position runs). The K new latent / mirror rows are
//! already in the cache: row r attends keys `[0, L + r]`.
//!
//! Geometries (MLA, kv_lora 512 + rope 64):
//! * `absorbed`: the latent-space route, `k_a` = latent cache `[0, 512)`,
//!   `k_b` = rope tail `[512, 576)`, `v` = latent `[0, 512)`, all heads sharing
//!   the row (`head_stride 0`), dq 576, dv 512, 32 heads;
//! * `bf16_mirror`: expanded BF16 K_nope / V planes (128 / 128 per head) + the
//!   64-wide rope tail from the same latent cache, 16 heads;
//! * `q8_mirror`: the same with INT8 + per-(row, head) F16-scale planes.
//!
//! Lengths `L` in {1, 37, 511, 2048, 8000}, K = 1..=16 on every one, with
//! `round_y_bf16` on (the pipeline default) and, for K in {1, 5, 16}, off; plus a
//! `max_rows` truncation case (rows whose length passes `max_rows`). Random
//! deterministic data (fixed seed); the output starts as a NaN sentinel with a
//! guard row past the end, the workspace as a NaN fill.
//!
//! Prints per-(geometry, L, K) PASS/FAIL with mismatching words and max abs
//! diff, then timings (median of launch batches) of rows_exact vs K singleton
//! decode calls vs the current flash-prefill route at K = 1, 2, 4, 8, 16 (L =
//! 2048 and 8000). Asserts every comparison is equal.
//!
//! `#[ignore]`d: needs a HIP GPU. Run explicitly (through the fabric lock):
//!
//!   cargo test -p rdna-compute --release --test latent_attention_decode_rows_exact -- --ignored --nocapture

use rdna_compute::latent_attention_decode::{
    latent_attention_decode_workspace_elems, LATENT_DECODE_ROWS_CAP,
};
use rdna_compute::latent_attention_tiled::{q8_plane_row_bytes, LatentRows, TiledAttention};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const SENTINEL: u32 = 0x7fc0_dead;
const LATENT: usize = 512;
const TAIL: usize = 64;
const NOPE: usize = 128;
const VDIM: usize = 128;
const LENGTHS: [usize; 5] = [1, 37, 511, 2048, 8000];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }
    fn unit(&mut self) -> f32 {
        (self.next() as u32 & 0xff_ffff) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

fn bf16_vec(rng: &mut Rng, n: usize, amp: f32) -> Vec<u16> {
    (0..n)
        .map(|_| ((rng.unit() * amp).to_bits() >> 16) as u16)
        .collect()
}

fn upload_bf16(gpu: &mut Gpu, data: &[u16]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::BF16)
        .expect("alloc bf16");
    // SAFETY: u16 -> u8 view of the same bytes; the host is little endian.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    gpu.hip.device_synchronize().expect("sync");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("upload bf16");
    t
}

fn set_pos(gpu: &Gpu, pos: &GpuTensor, v: i32) {
    gpu.hip.device_synchronize().expect("sync before pos write");
    gpu.hip
        .memcpy_htod(&pos.buf, &v.to_le_bytes())
        .expect("pos write");
}

fn sentinel(gpu: &mut Gpu, n: usize) -> GpuTensor {
    gpu.upload_f32(&vec![f32::from_bits(SENTINEL); n], &[n])
        .expect("upload sentinel")
}

fn bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.hip.device_synchronize().expect("sync");
    gpu.download_f32(t)
        .expect("download")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn compare(got: &[u32], want: &[u32]) -> (usize, f32) {
    let mut bad = 0;
    let mut max_abs = 0.0f32;
    for (&g, &w) in got.iter().zip(want) {
        if g != w {
            bad += 1;
        }
        let d = (f32::from_bits(g) - f32::from_bits(w)).abs();
        if d.is_finite() && d > max_abs {
            max_abs = d;
        }
    }
    (bad, max_abs)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Absorbed,
    Bf16Mirror,
    Q8Mirror,
}

struct Geo {
    name: &'static str,
    kind: Kind,
    heads: usize,
    cap: usize,
    /// Latent cache `[cap, LATENT + TAIL]` BF16 (the rope tail lives here).
    cache: GpuTensor,
    /// Mirror planes (K_nope, V) for the mirror kinds.
    planes: Option<(GpuTensor, GpuTensor)>,
}

impl Geo {
    fn dq(&self) -> usize {
        match self.kind {
            Kind::Absorbed => LATENT + TAIL,
            _ => NOPE + TAIL,
        }
    }
    fn dv(&self) -> usize {
        match self.kind {
            Kind::Absorbed => LATENT,
            _ => VDIM,
        }
    }
    fn q8(&self) -> bool {
        self.kind == Kind::Q8Mirror
    }

    fn build(
        gpu: &mut Gpu,
        rng: &mut Rng,
        name: &'static str,
        kind: Kind,
        heads: usize,
        cap: usize,
    ) -> Geo {
        let row = LATENT + TAIL;
        let cache = upload_bf16(gpu, &bf16_vec(rng, cap * row, 1.0));
        let planes = match kind {
            Kind::Absorbed => None,
            Kind::Bf16Mirror => Some((
                upload_bf16(gpu, &bf16_vec(rng, cap * heads * NOPE, 1.0)),
                upload_bf16(gpu, &bf16_vec(rng, cap * heads * VDIM, 1.0)),
            )),
            Kind::Q8Mirror => {
                let mut plane = |w: usize| {
                    let rb = q8_plane_row_bytes(heads, w);
                    let mut bytes = vec![0u8; cap * rb];
                    for r in 0..cap {
                        let base = r * rb;
                        for b in &mut bytes[base..base + heads * w] {
                            *b = rng.next() as u8;
                        }
                        for h in 0..heads {
                            // f16 scale in [2^-5, 2^-3): exponent field 10..12
                            let s = 0x2800u16 + (rng.next() % 0x0800) as u16;
                            let o = base + heads * w + 2 * h;
                            bytes[o..o + 2].copy_from_slice(&s.to_le_bytes());
                        }
                    }
                    gpu.upload_raw(&bytes, &[bytes.len()])
                        .expect("upload plane")
                };
                Some((plane(NOPE), plane(VDIM)))
            }
        };
        Geo {
            name,
            kind,
            heads,
            cap,
            cache,
            planes,
        }
    }

    fn attention<'a>(
        &'a self,
        q: &'a GpuTensor,
        y: &'a GpuTensor,
        pos: &'a GpuTensor,
        pos_offset: usize,
        n_rows: usize,
        max_rows: usize,
        round: bool,
    ) -> TiledAttention<'a> {
        let width = LATENT + TAIL;
        let tail = LatentRows {
            buf: &self.cache,
            offset: LATENT,
            tok_stride: width,
            head_stride: 0,
            width: TAIL,
        };
        let (k_a, k_b, v) = match (&self.kind, &self.planes) {
            (Kind::Absorbed, _) => (
                LatentRows {
                    buf: &self.cache,
                    offset: 0,
                    tok_stride: width,
                    head_stride: 0,
                    width: LATENT,
                },
                tail,
                LatentRows {
                    buf: &self.cache,
                    offset: 0,
                    tok_stride: width,
                    head_stride: 0,
                    width: LATENT,
                },
            ),
            (Kind::Bf16Mirror, Some((kn, vm))) => (
                LatentRows {
                    buf: kn,
                    offset: 0,
                    tok_stride: self.heads * NOPE,
                    head_stride: NOPE,
                    width: NOPE,
                },
                tail,
                LatentRows {
                    buf: vm,
                    offset: 0,
                    tok_stride: self.heads * VDIM,
                    head_stride: VDIM,
                    width: VDIM,
                },
            ),
            (Kind::Q8Mirror, Some((kn, vm))) => (
                LatentRows {
                    buf: kn,
                    offset: 0,
                    tok_stride: q8_plane_row_bytes(self.heads, NOPE),
                    head_stride: NOPE,
                    width: NOPE,
                },
                tail,
                LatentRows {
                    buf: vm,
                    offset: 0,
                    tok_stride: q8_plane_row_bytes(self.heads, VDIM),
                    head_stride: VDIM,
                    width: VDIM,
                },
            ),
            _ => unreachable!("mirror kinds carry planes"),
        };
        TiledAttention {
            q,
            y,
            k_a,
            k_b,
            v,
            pos,
            pos_offset,
            n_rows,
            n_heads: self.heads,
            scale: 1.0 / (self.dq() as f32).sqrt(),
            max_rows,
            round_y_bf16: round,
        }
    }
}

/// The rows == 1 decode call, `r` rows past `*pos` (`pos_offset = r`).
fn singleton(gpu: &mut Gpu, geo: &Geo, a: &TiledAttention<'_>, ws: &GpuTensor) {
    if geo.q8() {
        gpu.latent_attention_decode_split_q8(a, ws)
    } else {
        gpu.latent_attention_decode_split(a, ws)
    }
    .expect("singleton decode");
}

fn rows_exact(gpu: &mut Gpu, geo: &Geo, a: &TiledAttention<'_>, ws: &GpuTensor) {
    if geo.q8() {
        gpu.latent_attention_decode_split_rows_q8(a, ws)
    } else {
        gpu.latent_attention_decode_split_rows(a, ws)
    }
    .expect("rows decode");
}

fn flash(gpu: &mut Gpu, geo: &Geo, a: &TiledAttention<'_>) {
    if geo.q8() {
        gpu.latent_attention_flash_prefill_q8(a)
    } else {
        gpu.latent_attention_flash_prefill(a)
    }
    .expect("flash prefill");
}

#[test]
#[ignore = "needs a HIP GPU"]
fn latent_attention_decode_rows_are_bit_identical_to_the_decode_singleton() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut rng = Rng(0x0a7a_0004);
    println!("arch {}", gpu.arch);
    let lmax = *LENGTHS.last().unwrap();
    let cap = lmax + LATENT_DECODE_ROWS_CAP;
    let geos = [
        Geo::build(&mut gpu, &mut rng, "absorbed", Kind::Absorbed, 32, cap),
        Geo::build(&mut gpu, &mut rng, "bf16_mirror", Kind::Bf16Mirror, 16, cap),
        Geo::build(&mut gpu, &mut rng, "q8_mirror", Kind::Q8Mirror, 16, cap),
    ];
    let pos = gpu.alloc_tensor(&[1], DType::F32).expect("alloc pos");

    let mut failures: Vec<String> = Vec::new();
    let mut cases = 0usize;
    for geo in &geos {
        let (dq, dv, heads) = (geo.dq(), geo.dv(), geo.heads);
        let q_all = gpu
            .upload_f32(
                &(0..LATENT_DECODE_ROWS_CAP * heads * dq)
                    .map(|_| rng.unit())
                    .collect::<Vec<_>>(),
                &[LATENT_DECODE_ROWS_CAP * heads * dq],
            )
            .expect("upload q");
        let ws_one = latent_attention_decode_workspace_elems(heads, dv);
        let ws_rows = gpu
            .upload_f32(
                &vec![f32::NAN; ws_one * LATENT_DECODE_ROWS_CAP],
                &[ws_one * LATENT_DECODE_ROWS_CAP],
            )
            .expect("ws rows");
        let ws_single = gpu
            .upload_f32(&vec![f32::NAN; ws_one], &[ws_one])
            .expect("ws single");

        // (L, max_rows override): the last case truncates rows past max_rows.
        let mut scenarios: Vec<(usize, usize)> = LENGTHS.iter().map(|&l| (l, cap)).collect();
        scenarios.push((511, 520));
        for (l, max_rows) in scenarios {
            for k in 1..=LATENT_DECODE_ROWS_CAP {
                let rounds: &[bool] = if [1, 5, 16].contains(&k) {
                    &[true, false]
                } else {
                    &[true]
                };
                for &round in rounds {
                    let q = q_all.sub_offset(0, k * heads * dq);
                    let y = sentinel(&mut gpu, (k + 1) * heads * dv);
                    set_pos(&gpu, &pos, l as i32);
                    let a = geo.attention(&q, &y, &pos, 0, k, max_rows, round);
                    rows_exact(&mut gpu, geo, &a, &ws_rows);
                    let got = bits(&gpu, &y);

                    // Reference: row r is the rows == 1 decode at *pos = l + r.
                    let yref = sentinel(&mut gpu, (k + 1) * heads * dv);
                    for r in 0..k {
                        set_pos(&gpu, &pos, (l + r) as i32);
                        let q_r = q_all.sub_offset(r * heads * dq, heads * dq);
                        let y_r = yref.sub_offset(r * heads * dv, heads * dv);
                        let a1 = geo.attention(&q_r, &y_r, &pos, 0, 1, max_rows, round);
                        singleton(&mut gpu, geo, &a1, &ws_single);
                    }
                    let want = bits(&gpu, &yref);

                    let n = k * heads * dv;
                    let guard_ok = got[n..].iter().all(|&v| v == SENTINEL);
                    let holes = want[..n].iter().filter(|&&v| v == SENTINEL).count();
                    let (bad, max_abs) = compare(&got[..n], &want[..n]);
                    let pass = bad == 0 && guard_ok && holes == 0;
                    cases += 1;
                    println!(
                        "{} {} L={l} max_rows={max_rows} K={k} round={round}: mismatching words {bad}/{n}, max abs diff {max_abs:e}{}{}",
                        if pass { "PASS" } else { "FAIL" },
                        geo.name,
                        if guard_ok { "" } else { ", WROTE PAST THE END" },
                        if holes == 0 { "" } else { ", singleton left holes" },
                    );
                    if !pass {
                        failures.push(format!(
                            "{} L={l} max_rows={max_rows} K={k} round={round} bad={bad}",
                            geo.name
                        ));
                    }
                    gpu.free_tensor(y).expect("free");
                    gpu.free_tensor(yref).expect("free");
                }
            }
        }
    }
    println!("{cases} comparisons, {} failed", failures.len());

    // Timing: rows_exact (one launch pair) vs K singleton decode calls vs the
    // current flash-prefill route over the same K rows.
    println!("== timing (us per call, median of 9 batches x 20 calls; K singletons = K x (partials+merge))");
    println!(
        "geometry     L     K  rows_exact  K singletons  flash prefill  vs singletons  vs flash"
    );
    for geo in &geos {
        let (dq, dv, heads) = (geo.dq(), geo.dv(), geo.heads);
        let ws_one = latent_attention_decode_workspace_elems(heads, dv);
        let ws_rows = gpu
            .upload_f32(
                &vec![0.0; ws_one * LATENT_DECODE_ROWS_CAP],
                &[ws_one * LATENT_DECODE_ROWS_CAP],
            )
            .expect("ws");
        let ws_single = gpu.upload_f32(&vec![0.0; ws_one], &[ws_one]).expect("ws");
        let q_all = gpu
            .upload_f32(
                &(0..LATENT_DECODE_ROWS_CAP * heads * dq)
                    .map(|_| rng.unit())
                    .collect::<Vec<_>>(),
                &[LATENT_DECODE_ROWS_CAP * heads * dq],
            )
            .expect("q");
        for l in [2048usize, 8000] {
            set_pos(&gpu, &pos, l as i32);
            for k in [1usize, 2, 4, 8, 16] {
                let q = q_all.sub_offset(0, k * heads * dq);
                let y = sentinel(&mut gpu, k * heads * dv);
                let a = geo.attention(&q, &y, &pos, 0, k, cap, true);
                let mut time = |gpu: &mut Gpu, which: u8| -> f64 {
                    let run = |gpu: &mut Gpu| match which {
                        0 => rows_exact(gpu, geo, &a, &ws_rows),
                        1 => {
                            for r in 0..k {
                                let q_r = q_all.sub_offset(r * heads * dq, heads * dq);
                                let y_r = y.sub_offset(r * heads * dv, heads * dv);
                                let a1 = geo.attention(&q_r, &y_r, &pos, r, 1, cap, true);
                                singleton(gpu, geo, &a1, &ws_single);
                            }
                        }
                        _ => flash(gpu, geo, &a),
                    };
                    for _ in 0..3 {
                        run(gpu);
                    }
                    gpu.hip.device_synchronize().expect("sync");
                    let samples = (0..9)
                        .map(|_| {
                            let t = Instant::now();
                            for _ in 0..20 {
                                run(gpu);
                            }
                            gpu.hip.device_synchronize().expect("sync");
                            t.elapsed().as_secs_f64() * 1e6 / 20.0
                        })
                        .collect();
                    median(samples)
                };
                let (t_rows, t_single, t_flash) =
                    (time(&mut gpu, 0), time(&mut gpu, 1), time(&mut gpu, 2));
                println!(
                    "{:12} {l:5} {k:3} {t_rows:11.1} {t_single:13.1} {t_flash:14.1} {:13.2}x {:8.2}x",
                    geo.name,
                    t_single / t_rows,
                    t_flash / t_rows
                );
            }
        }
    }

    assert!(
        failures.is_empty(),
        "multi-row decode differs from the rows == 1 decode call: {failures:?}"
    );
}
