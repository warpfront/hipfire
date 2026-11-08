// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! X1 oracle: the vocab-sharded lane draw (`lane_leviathan_verify` modes 1 + 2 through
//! the staged `LaneDrawArena` API) equals the whole-row draw (`lane_draws`, mode 0),
//! byte for byte: every lane's tokens, accepted count and advanced RNG, and every
//! row's p table.
//!
//! Setup per launch: lanes of full rows `[rows][V]`; arena A stages the first halves
//! (packed `[rows][V/2]`, base 0), arena B the second halves (base V/2). Both run
//! `launch_select`; A then runs `launch_merged` with `cand_peer` = B's candidate table
//! address (a plain device pointer here; UVA across ranks in the mesh); A reads back.
//!
//! Coverage (cycled over the lanes so the product is visited, then randomised):
//! temperature {0, 0.7, 1.0} x top_k {None, 0, 1, 20, 21, 64} x top_p {0.95, 1.0} x
//! min_p {None, 0.05}; repeat {1, 1.3, 2}, presence {0, +, -}, frequency {0, +, -},
//! windows {8, 64, 200}; blocked tokens (hot ids in both halves), scope tokens in both
//! halves, EOS, drafts with q and without, AR (1-row) lanes; equal top logits at
//! V/2-1 and V/2 in every row; exact ties from BF16 rounding; NaN / -inf planted; and
//! a mass-tie lane (2,200 equal top values per shard, so both shards take the
//! tile path). V = 248,320, 5,000 (aligned shards) and 4,098 (odd shard width, the
//! unaligned scalar path).
//!
//! `#[ignore]`d: needs an exact gfx12 HIP GPU. Run explicitly:
//!
//!   cargo test -p hipfire-runtime --release --test x1_lane_draw_shard_gpu -- --ignored --nocapture
//!
//! The `leviathan_twin_*` tests run the same matrix (same generator, seeds and launch
//! shapes) through the COMMITTED objects only (`kernels/lane_leviathan_{hipcc,pm}_gfx120{0,1}`,
//! `include_bytes!`, loaded with `module_load_data`: no hipcc, no JIT, no dispatch) and
//! require the PM twin byte-identical to the hipcc object in all three modes (raw out
//! records, p tables, mode-1 candidate tables), and for each object modes 1+2 equal mode 0.
//! They fail, not skip, on a device that is not gfx1200 or gfx1201.

mod x1_lane_draw_twin;

use hipfire_runtime::lane_draw::{
    cand_table_words, lane_draws, lane_draws_supported, LaneDrawArena, LaneDrawJob,
};
use hipfire_runtime::sampler::SamplerConfig;
use rdna_compute::lane_leviathan::{LLV_CAND_WORDS, LLV_OUT_WORDS, LLV_P_WORDS};
use rdna_compute::{DType, Gpu, GpuTensor};
use x1_lane_draw_twin as twin;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
    fn chance(&mut self, pct: u32) -> bool {
        self.next() % 100 < pct
    }
}

fn bf16(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xffff_0000)
}

struct LaneSpec {
    rows: usize,
    /// `[rows][V]`.
    full: Vec<f32>,
    drafts: Vec<u32>,
    q: Vec<Vec<(u32, f32)>>,
    history: Vec<u32>,
    cfg: SamplerConfig,
    rng: u32,
    eos: Option<u32>,
}

const TEMPS: [f32; 3] = [0.0, 0.7, 1.0];
const TOPK: [Option<u32>; 6] = [None, Some(0), Some(1), Some(20), Some(21), Some(64)];
const TOPP: [f32; 2] = [0.95, 1.0];
const MINP: [Option<f32>; 2] = [None, Some(0.05)];
const REP: [f32; 3] = [1.0, 1.3, 2.0];
const PRES: [f32; 3] = [0.0, 0.5, -0.5];
const FREQ: [f32; 3] = [0.0, 0.3, -0.3];
const WINDOW: [usize; 3] = [8, 64, 200];

fn make_lane(g: &mut Lcg, v: usize, lane_global: usize, mass_tie: bool) -> LaneSpec {
    let w = v / 2;
    let n_drafts = if g.chance(25) { 0 } else { 1 + g.below(5) };
    let rows = n_drafts + 1;
    // Hot ids: both sides of the split, then random ones; the first two tie exactly.
    let mut hot: Vec<u32> = vec![(w - 1) as u32, w as u32];
    while hot.len() < 8 {
        let t = g.below(v) as u32;
        if !hot.contains(&t) {
            hot.push(t);
        }
    }
    let mut full = Vec::with_capacity(rows * v);
    for _ in 0..rows {
        let quant = g.chance(50);
        let mut row: Vec<f32> = (0..v)
            .map(|_| {
                let x = g.unit() * 3.0 - 1.0;
                if quant {
                    bf16(x)
                } else {
                    x
                }
            })
            .collect();
        if mass_tie {
            for x in &mut row {
                *x = -2.0 - x.abs();
            }
            for i in 0..2200usize.min(w) {
                row[i] = 7.0;
                row[w + i] = 7.0;
            }
        }
        row[hot[0] as usize] = 9.0;
        row[hot[1] as usize] = 9.0;
        let all_tied = g.chance(40);
        for &t in &hot[2..] {
            row[t as usize] = if all_tied {
                9.0
            } else {
                5.0 + 3.0 * g.unit().abs()
            };
        }
        if g.chance(25) {
            for _ in 0..3 {
                let i = g.below(v);
                if !hot.contains(&(i as u32)) {
                    row[i] = f32::NAN;
                }
                let j = g.below(v);
                if !hot.contains(&(j as u32)) {
                    row[j] = f32::NEG_INFINITY;
                }
            }
        }
        full.extend(row);
    }
    let pick_tok = |g: &mut Lcg, hot_pct: u32| -> u32 {
        if g.chance(hot_pct) {
            hot[g.below(hot.len())]
        } else {
            g.below(v) as u32
        }
    };
    let hist_len = [0usize, 3, 20, 80][g.below(4)];
    let history: Vec<u32> = (0..hist_len).map(|_| pick_tok(g, 70)).collect();
    let drafts: Vec<u32> = (0..n_drafts).map(|_| pick_tok(g, 80)).collect();
    let q: Vec<Vec<(u32, f32)>> = (0..n_drafts)
        .map(|_| {
            if g.chance(40) {
                Vec::new()
            } else {
                let k = 2 + g.below(3);
                let mut toks = Vec::new();
                while toks.len() < k {
                    let t = hot[g.below(hot.len())];
                    if !toks.contains(&t) {
                        toks.push(t);
                    }
                }
                let wts = [0.5f32, 0.3, 0.2, 0.1];
                let sum: f32 = wts[..k].iter().sum();
                toks.iter().zip(wts).map(|(&t, wt)| (t, wt / sum)).collect()
            }
        })
        .collect();
    let n_blocked = g.below(4);
    let blocked_tokens: Vec<u32> = (0..n_blocked).map(|_| pick_tok(g, 90)).collect();
    let cfg = SamplerConfig {
        temperature: TEMPS[lane_global % 3],
        top_k: TOPK[(lane_global / 3) % 6],
        top_p: TOPP[(lane_global / 18) % 2],
        min_p: MINP[(lane_global / 36) % 2],
        repeat_penalty: REP[(lane_global / 2) % 3],
        presence_penalty: PRES[(lane_global / 5) % 3],
        frequency_penalty: FREQ[(lane_global / 7) % 3],
        repeat_window: WINDOW[(lane_global / 11) % 3],
        blocked_tokens,
    };
    let eos = if g.chance(60) {
        Some(pick_tok(g, 90))
    } else {
        None
    };
    LaneSpec {
        rows,
        full,
        drafts,
        q,
        history,
        cfg,
        rng: g.next() | 1,
        eos,
    }
}

fn ptab_bits(gpu: &Gpu, t: &GpuTensor, rows: usize) -> Vec<u32> {
    gpu.download_f32(t).expect("ptab download")[..rows * LLV_P_WORDS]
        .iter()
        .map(|f| f.to_bits())
        .collect()
}

fn job<'a>(l: &'a LaneSpec, logits: &'a GpuTensor, vocab: usize, base: usize) -> LaneDrawJob<'a> {
    LaneDrawJob {
        logits,
        rows: l.rows,
        vocab,
        base,
        drafts: &l.drafts,
        q: &l.q,
        history: &l.history,
        cfg: &l.cfg,
        rng_state: l.rng,
        eos: l.eos,
    }
}

/// One launch of `lanes`: whole-row reference vs the sharded A+B merge.
fn compare_launch(gpu: &mut Gpu, v: usize, lanes: &[LaneSpec], ctx: &str) {
    let w = v / 2;
    // Device rows: the full rows, and each half packed [rows][w].
    let mut full_t = Vec::new();
    let mut half0 = Vec::new();
    let mut half1 = Vec::new();
    for l in lanes {
        full_t.push(gpu.upload_f32(&l.full, &[l.full.len()]).expect("full rows"));
        let (mut h0, mut h1) = (Vec::new(), Vec::new());
        for r in 0..l.rows {
            h0.extend_from_slice(&l.full[r * v..r * v + w]);
            h1.extend_from_slice(&l.full[r * v + w..(r + 1) * v]);
        }
        half0.push(gpu.upload_f32(&h0, &[h0.len()]).expect("half 0"));
        half1.push(gpu.upload_f32(&h1, &[h1.len()]).expect("half 1"));
    }
    let fjobs: Vec<LaneDrawJob<'_>> = lanes
        .iter()
        .zip(&full_t)
        .map(|(l, t)| job(l, t, v, 0))
        .collect();
    let ajobs: Vec<LaneDrawJob<'_>> = lanes
        .iter()
        .zip(&half0)
        .map(|(l, t)| job(l, t, w, 0))
        .collect();
    let bjobs: Vec<LaneDrawJob<'_>> = lanes
        .iter()
        .zip(&half1)
        .map(|(l, t)| job(l, t, w, w))
        .collect();
    let rows: usize = lanes.iter().map(|l| l.rows).sum();

    let mut arena_f = LaneDrawArena::new();
    let want = lane_draws(gpu, &mut arena_f, &fjobs).expect("whole-row draws");

    let mut arena_a = LaneDrawArena::new();
    let mut arena_b = LaneDrawArena::new();
    let staged_a = arena_a.stage(gpu, &ajobs).expect("stage A");
    let staged_b = arena_b.stage(gpu, &bjobs).expect("stage B");
    assert_eq!(staged_a.rows(), rows);
    let cand_a = gpu
        .zeros(&[cand_table_words(rows)], DType::F32)
        .expect("cand A");
    let cand_b = gpu
        .zeros(&[cand_table_words(rows)], DType::F32)
        .expect("cand B");
    arena_a
        .launch_select(gpu, &staged_a, &cand_a)
        .expect("select A");
    arena_b
        .launch_select(gpu, &staged_b, &cand_b)
        .expect("select B");
    arena_a
        .launch_merged(gpu, &staged_a, &cand_a, cand_b.buf.as_ptr() as u64)
        .expect("merged A");
    let got = arena_a.readback(gpu, &staged_a).expect("read back A");
    gpu.hip.device_synchronize().expect("sync");

    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g, w,
            "{ctx}: lane {i} draw differs (cfg {:?})",
            lanes[i].cfg
        );
    }
    assert_eq!(got.len(), want.len(), "{ctx}: lane count");
    let want_p = ptab_bits(gpu, arena_f.ptab().expect("full ptab"), rows);
    let got_p = ptab_bits(gpu, arena_a.ptab().expect("merged ptab"), rows);
    if got_p != want_p {
        let at = got_p.iter().zip(&want_p).position(|(a, b)| a != b).unwrap();
        panic!(
            "{ctx}: p tables differ at row {} word {}: sharded {:#x} whole {:#x}",
            at / LLV_P_WORDS,
            at % LLV_P_WORDS,
            got_p[at],
            want_p[at]
        );
    }

    for t in full_t.into_iter().chain(half0).chain(half1) {
        let _ = gpu.free_tensor(t);
    }
    for t in [cand_a, cand_b] {
        let _ = gpu.free_tensor(t);
    }
    arena_f.free_gpu(gpu);
    arena_a.free_gpu(gpu);
    arena_b.free_gpu(gpu);
}

fn sweep(gpu: &mut Gpu, v: usize, launches: usize, seed: u64) {
    let mut g = Lcg(seed);
    const LANES: usize = 6;
    let mut lane_global = 0usize;
    for launch in 0..launches {
        let lanes: Vec<LaneSpec> = (0..LANES)
            .map(|k| {
                let lane_id = lane_global;
                lane_global += 1;
                // One mass-tie lane per launch (the first lane of every third launch).
                make_lane(&mut g, v, lane_id, k == 0 && launch % 3 == 0)
            })
            .collect();
        compare_launch(gpu, v, &lanes, &format!("V {v} launch {launch}"));
    }
    eprintln!("lane draw shard: V {v}: {launches} launches x {LANES} lanes byte-identical");
}

#[test]
#[ignore = "needs an exact gfx12 GPU: run through the GPU fabric"]
fn sharded_lane_draw_equals_whole_row_flash_vocab() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !lane_draws_supported(&gpu) {
        eprintln!("skip: lane-wide verify does not run on {}", gpu.arch);
        return;
    }
    // 12 launches x 6 lanes = 72 lanes: every (temperature, top_k, top_p, min_p) cell.
    sweep(&mut gpu, 248_320, 12, 0x51_0a);
}

#[test]
#[ignore = "needs an exact gfx12 GPU: run through the GPU fabric"]
fn sharded_lane_draw_equals_whole_row_small_vocabs() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !lane_draws_supported(&gpu) {
        eprintln!("skip: lane-wide verify does not run on {}", gpu.arch);
        return;
    }
    sweep(&mut gpu, 5_000, 24, 0x51_0b);
    // Odd shard width (2,049): rows past the first start off 16-byte alignment.
    sweep(&mut gpu, 4_098, 24, 0x51_0c);
}

// ───────────────────────── committed-object twin oracle ─────────────────────────

/// Every observable output word of one implementation over one launch set.
struct TwinRun {
    /// Mode 0 over the full rows: raw out records and p tables.
    out_full: Vec<u32>,
    p_full: Vec<u32>,
    /// Mode 1 candidate tables of shard A (`[0, V/2)`) and shard B (`[V/2, V)`).
    cand_a: Vec<u32>,
    cand_b: Vec<u32>,
    /// Mode 2 on A (peer = B's table): raw out records and p tables.
    out_merged: Vec<u32>,
    p_merged: Vec<u32>,
}

/// Run `imp` in all three modes over the lanes' full rows and A/B halves.
#[allow(clippy::too_many_arguments)]
fn twin_run(
    gpu: &mut Gpu,
    imp: &twin::Impl,
    packed_f: &twin::Packed,
    packed_a: &twin::Packed,
    packed_b: &twin::Packed,
    rows_per_lane: &[usize],
    ctx: &str,
) -> TwinRun {
    let ctx = format!("{ctx} [{}]", imp.name);
    let rows = packed_f.layout.rows;
    let out_words = packed_f.layout.out_words();
    assert_eq!(packed_a.layout, packed_b.layout);
    assert_eq!(packed_a.layout.rows, rows);

    // Mode 0: the whole row.
    let region_f = twin::upload_region(gpu, packed_f);
    let ptab_f = twin::zeros_words(gpu, rows * LLV_P_WORDS);
    imp.launch(
        gpu,
        &region_f,
        packed_f.layout,
        Some(&ptab_f),
        twin::MODE_FULL,
        0,
        0,
    );
    let out_full = twin::download_words(gpu, &region_f, out_words);
    let p_full = twin::download_words(gpu, &ptab_f, rows * LLV_P_WORDS);
    let draws_full = twin::decode_out(&out_full, rows_per_lane, &format!("{ctx} mode 0"));

    // Mode 1 on both shards: candidate tables; the out records stay untouched.
    let region_a = twin::upload_region(gpu, packed_a);
    let region_b = twin::upload_region(gpu, packed_b);
    let cand_a = twin::zeros_words(gpu, cand_table_words(rows));
    let cand_b = twin::zeros_words(gpu, cand_table_words(rows));
    imp.launch(
        gpu,
        &region_a,
        packed_a.layout,
        None,
        twin::MODE_SELECT,
        cand_a.buf.as_ptr() as u64,
        0,
    );
    imp.launch(
        gpu,
        &region_b,
        packed_b.layout,
        None,
        twin::MODE_SELECT,
        cand_b.buf.as_ptr() as u64,
        0,
    );
    for (name, region) in [("A", &region_a), ("B", &region_b)] {
        let out = twin::download_words(gpu, region, out_words);
        assert!(
            out.iter().all(|&w| w == 0),
            "{ctx}: mode 1 on shard {name} wrote an out record"
        );
    }
    let cand_a_w = twin::download_words(gpu, &cand_a, cand_table_words(rows));
    let cand_b_w = twin::download_words(gpu, &cand_b, cand_table_words(rows));

    // Mode 2 on A with B's table as the peer.
    let ptab_m = twin::zeros_words(gpu, rows * LLV_P_WORDS);
    imp.launch(
        gpu,
        &region_a,
        packed_a.layout,
        Some(&ptab_m),
        twin::MODE_MERGED,
        cand_a.buf.as_ptr() as u64,
        cand_b.buf.as_ptr() as u64,
    );
    let out_merged = twin::download_words(gpu, &region_a, out_words);
    let p_merged = twin::download_words(gpu, &ptab_m, rows * LLV_P_WORDS);
    let draws_merged = twin::decode_out(&out_merged, rows_per_lane, &format!("{ctx} mode 2"));

    // The X1 oracle on this object: modes 1+2 equal mode 0, draws and p tables.
    for (i, (m, f)) in draws_merged.iter().zip(&draws_full).enumerate() {
        assert_eq!(m, f, "{ctx}: lane {i} merged draw differs from whole-row");
    }
    twin::assert_words_eq(
        &p_merged,
        &p_full,
        LLV_P_WORDS,
        "p tables (mode 2 vs mode 0)",
        &ctx,
    );

    for t in [region_f, ptab_f, region_a, region_b, cand_a, cand_b, ptab_m] {
        let _ = gpu.free_tensor(t);
    }
    TwinRun {
        out_full,
        p_full,
        cand_a: cand_a_w,
        cand_b: cand_b_w,
        out_merged,
        p_merged,
    }
}

/// One launch set: hipcc object vs PM object, every observable word, all three modes.
fn twin_compare_launch(
    gpu: &mut Gpu,
    impls: &[twin::Impl; 2],
    v: usize,
    lanes: &[LaneSpec],
    ctx: &str,
) {
    let w = v / 2;
    let mut full_t = Vec::new();
    let mut half0 = Vec::new();
    let mut half1 = Vec::new();
    for l in lanes {
        full_t.push(gpu.upload_f32(&l.full, &[l.full.len()]).expect("full rows"));
        let (mut h0, mut h1) = (Vec::new(), Vec::new());
        for r in 0..l.rows {
            h0.extend_from_slice(&l.full[r * v..r * v + w]);
            h1.extend_from_slice(&l.full[r * v + w..(r + 1) * v]);
        }
        half0.push(gpu.upload_f32(&h0, &[h0.len()]).expect("half 0"));
        half1.push(gpu.upload_f32(&h1, &[h1.len()]).expect("half 1"));
    }
    let fjobs: Vec<LaneDrawJob<'_>> = lanes
        .iter()
        .zip(&full_t)
        .map(|(l, t)| job(l, t, v, 0))
        .collect();
    let ajobs: Vec<LaneDrawJob<'_>> = lanes
        .iter()
        .zip(&half0)
        .map(|(l, t)| job(l, t, w, 0))
        .collect();
    let bjobs: Vec<LaneDrawJob<'_>> = lanes
        .iter()
        .zip(&half1)
        .map(|(l, t)| job(l, t, w, w))
        .collect();
    let packed_f = twin::pack(&fjobs);
    let packed_a = twin::pack(&ajobs);
    let packed_b = twin::pack(&bjobs);
    let rows_per_lane: Vec<usize> = lanes.iter().map(|l| l.rows).collect();

    let hipcc = twin_run(
        gpu,
        &impls[0],
        &packed_f,
        &packed_a,
        &packed_b,
        &rows_per_lane,
        ctx,
    );
    let pm = twin_run(
        gpu,
        &impls[1],
        &packed_f,
        &packed_a,
        &packed_b,
        &rows_per_lane,
        ctx,
    );

    let pair = |what: &str, a: &[u32], b: &[u32], row_words: usize| {
        twin::assert_words_eq(a, b, row_words, &format!("hipcc vs pm: {what}"), ctx);
    };
    pair(
        "mode 0 out records",
        &hipcc.out_full,
        &pm.out_full,
        LLV_OUT_WORDS,
    );
    pair("mode 0 p tables", &hipcc.p_full, &pm.p_full, LLV_P_WORDS);
    pair(
        "mode 1 shard A candidates",
        &hipcc.cand_a,
        &pm.cand_a,
        LLV_CAND_WORDS,
    );
    pair(
        "mode 1 shard B candidates",
        &hipcc.cand_b,
        &pm.cand_b,
        LLV_CAND_WORDS,
    );
    pair(
        "mode 2 out records",
        &hipcc.out_merged,
        &pm.out_merged,
        LLV_OUT_WORDS,
    );
    pair(
        "mode 2 p tables",
        &hipcc.p_merged,
        &pm.p_merged,
        LLV_P_WORDS,
    );

    for t in full_t.into_iter().chain(half0).chain(half1) {
        let _ = gpu.free_tensor(t);
    }
}

/// Plant the edge values the incumbent generator lacks (it plants NaN and -inf only) into
/// every row of `lane`: +inf, +0 / -0, quiet / negative / signalling NaN payloads and -inf,
/// in both shard halves and next to the shard boundary. Positions come from `e`, a stream
/// separate from the matrix generator, so the base matrix is exactly the incumbent's.
fn plant_edges(e: &mut Lcg, lane: &mut LaneSpec, v: usize) {
    const NANS: [u32; 4] = [0x7fc0_0000, 0xffc1_2345, 0x7f80_0001, 0x7fff_ffff];
    let w = v / 2;
    for r in 0..lane.rows {
        let row = &mut lane.full[r * v..(r + 1) * v];
        if r % 2 == 0 {
            row[e.below(w)] = f32::INFINITY;
            row[w + e.below(w)] = f32::INFINITY;
        }
        row[w - 2] = -0.0;
        row[w + 1] = 0.0;
        row[e.below(v)] = 0.0;
        row[e.below(v)] = -0.0;
        for (k, bits) in NANS.iter().enumerate() {
            let i = if k % 2 == 0 {
                e.below(w)
            } else {
                w + e.below(w)
            };
            row[i] = f32::from_bits(*bits);
        }
        row[e.below(v)] = f32::NEG_INFINITY;
    }
}

/// The incumbent sweep's lane generator and launch shapes, run through the committed
/// objects, with one edge-value lane per launch.
fn twin_sweep(gpu: &mut Gpu, impls: &[twin::Impl; 2], v: usize, launches: usize, seed: u64) {
    let mut g = Lcg(seed);
    let mut e = Lcg(seed ^ 0xed9e_5eed_0000_0001);
    const LANES: usize = 6;
    let mut lane_global = 0usize;
    for launch in 0..launches {
        let mut lanes: Vec<LaneSpec> = (0..LANES)
            .map(|k| {
                let lane_id = lane_global;
                lane_global += 1;
                make_lane(&mut g, v, lane_id, k == 0 && launch % 3 == 0)
            })
            .collect();
        plant_edges(&mut e, &mut lanes[launch % LANES], v);
        twin_compare_launch(gpu, impls, v, &lanes, &format!("V {v} launch {launch}"));
    }
    eprintln!(
        "lane draw twin: V {v}: {launches} launches x {LANES} lanes: PM == hipcc in modes 0/1/2, modes 1+2 == mode 0"
    );
}

/// Init the GPU and load both committed objects of its exact architecture.
fn twin_setup() -> (Gpu, [twin::Impl; 2]) {
    let mut gpu = Gpu::init().expect("gpu init");
    let objects = twin::objects_for(&gpu);
    assert_eq!(
        objects.arch, gpu.arch,
        "object arch selected from the device"
    );
    let impls = [
        twin::Impl::load(&mut gpu, "hipcc", objects.hipcc),
        twin::Impl::load(&mut gpu, "pm", objects.pm),
    ];
    (gpu, impls)
}

#[test]
#[ignore = "needs an exact gfx1200/gfx1201 GPU: run through the GPU fabric"]
fn leviathan_twin_pm_equals_hipcc_flash_vocab() {
    let (mut gpu, impls) = twin_setup();
    // The shard boundary: hot ids V/2 - 1 and V/2 tie exactly at 9.0 in every row.
    assert_eq!(248_320 / 2, 124_160);
    twin_sweep(&mut gpu, &impls, 248_320, 12, 0x51_0a);
}

#[test]
#[ignore = "needs an exact gfx1200/gfx1201 GPU: run through the GPU fabric"]
fn leviathan_twin_pm_equals_hipcc_small_vocabs() {
    let (mut gpu, impls) = twin_setup();
    twin_sweep(&mut gpu, &impls, 5_000, 24, 0x51_0b);
    // Odd shard width (2,049): rows past the first start off 16-byte alignment.
    twin_sweep(&mut gpu, &impls, 4_098, 24, 0x51_0c);
}
