// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Wave-3 C8 MoE glue folds X8a / X8b: byte ORACLE + timing bench (exact gfx1201; developer
//! example, `HIPFIRE_QWEN4_MOE_GLUE_FOLD` a / b).
//!
//! The reference is the served unfused chain on the ep_a8_wide route: `moe_router_softmax_top10_f32`
//! (fast router), `moe_scatter_fused_top10`, `ep_tile_mask_pm_gfx1201` (`Gpu::ep_tile_mask`).
//! Candidates: **A** = the same router, then `moe_scatter_fused_top10_pmask` (scatter + mask, one
//! launch); **B** = `moe_route_scatter_top10_wide_f32` (router + scatter + mask, one launch).
//!
//! `oracle`: every row count 9..=64 x seeded logits (uniform, heavy ties, all-equal, BF16-grid, shared
//! peaked rows that pile onto the same experts, +-80 spreads, rows with NaN / +Inf / -Inf) x the four
//! softmax recipes (normalize x BF16 round) x owner tables {e % 2 rank 0, e % 2 rank 1, e % 5 rank 0,
//! all foreign, all owned} into canary-filled buffers (whole allocations, trailing canary included).
//! `topk_idx`, `topk_w`, `expert_token_counts`, `expert_offsets`, `expert_tile_ids`, `tile_ids_owned`
//! and the router fault word are compared WORD FOR WORD; `sorted_slot_index` / `inverse_perm` are
//! compared as the permutation they are: the shipped scatter's atomic bucket fill leaves the order
//! of the slots inside one expert's bucket to a wave-arrival race (the reference does not repeat
//! itself; measured below as `chain != chain rerun`), so equal bucket contents, `sorted[inverse[i]]
//! == i`, equal -1 cells and equal canary tails are required. Controls: a 1-ulp flip in `topk_w`, a
//! changed owned tile, a bumped offset and a swapped sorted pair in a copy of the candidate dump
//! must each be detected.
//!
//! `bench`: reference chain / A / B per layer (event-timed single sequence and a 48-layer
//! back-to-back loop incl. launch gaps), rows 9..=64.
//!
//! Usage: qwen4_moe_glue_fold oracle | bench

use rdna_compute::moe::TileMaskFold;
use rdna_compute::{FeatureFlags, Gpu, GpuTensor};
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const E: usize = 512;
const K: usize = 10;
const BLOCK_M: usize = 16;
const PAD: usize = 64;
const CANARY: u32 = 0x7FC0_DEAD;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f32_signed(&mut self) -> f32 {
        let a = self.next_u32() & 0x7F_FFFF;
        let sign = self.next_u32() & 1;
        let v = f32::from_bits(0x3F80_0000 | a) - 1.0;
        if sign == 1 {
            -v
        } else {
            v
        }
    }
}

fn m_total_bound(slots: usize) -> usize {
    (slots + slots.min(E) * (BLOCK_M - 1)).div_ceil(BLOCK_M) * BLOCK_M
}

fn upload_u32(gpu: &mut Gpu, data: &[u32]) -> Res<GpuTensor> {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    Ok(gpu.upload_raw(&bytes, &[bytes.len()])?)
}

fn download_u32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Res<Vec<u32>> {
    let mut b = vec![0u8; n * 4];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf)?;
    Ok(b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// One set of outputs, every buffer `len + PAD` words of canary.
struct Outputs {
    t: [GpuTensor; 8],
    lens: [usize; 8],
    grouped_rows: usize,
}
impl Outputs {
    fn new(gpu: &mut Gpu, rows: usize) -> Res<Self> {
        let slots = rows * K;
        let m = m_total_bound(slots);
        let lens = [slots, slots, E, E + 1, m, m / BLOCK_M, slots, m / BLOCK_M];
        let mut v = Vec::new();
        for &n in &lens {
            v.push(upload_u32(gpu, &vec![CANARY; n + PAD])?);
        }
        let t: [GpuTensor; 8] = v.try_into().map_err(|_| "eight output tensors")?;
        Ok(Self {
            t,
            lens,
            grouped_rows: m,
        })
    }
    fn dump(&self, gpu: &Gpu) -> Res<Vec<Vec<u32>>> {
        let mut out = Vec::new();
        for (t, &n) in self.t.iter().zip(&self.lens) {
            out.push(download_u32(gpu, t, n + PAD)?);
        }
        Ok(out)
    }
}

fn gen_row(rng: &mut Lcg, mode: u32, shared: &[f32]) -> Vec<f32> {
    let mut r: Vec<f32> = match mode {
        0 | 4 | 5 | 6 => (0..E).map(|_| rng.f32_signed() * 4.0).collect(),
        1 => (0..E).map(|_| (rng.next_u32() % 6) as f32 * 0.5).collect(),
        2 => vec![1.25; E],
        3 => (0..E)
            .map(|_| f32::from_bits((rng.f32_signed() * 3.0).to_bits() & 0xFFFF_0000))
            .collect(),
        7 => shared.iter().map(|v| v + rng.f32_signed() * 1e-3).collect(),
        8 => (0..E).map(|_| rng.f32_signed() * 80.0).collect(),
        _ => vec![-1.0e30; E],
    };
    let at = (rng.next_u32() as usize) % E;
    match mode {
        4 => r[at] = f32::NAN,
        5 => r[at] = f32::INFINITY,
        6 => r[at] = f32::NEG_INFINITY,
        _ => {}
    }
    r
}

fn check_arch(gpu: &Gpu) -> Res<()> {
    if !gpu.arch_caps.is_gfx1201() {
        return Err("requires exact gfx1201".into());
    }
    Ok(())
}

/// Enable every glue fold on this device (the launchers' admission reads the flags).
fn enable_folds(gpu: &mut Gpu) {
    let mut f: FeatureFlags = (*gpu.flags).clone();
    f.qwen4_moe_glue_fold = 0b111;
    gpu.flags = Arc::new(f);
}

struct Table {
    name: &'static str,
    owner: Vec<i32>,
    rank: usize,
}

fn tables() -> Vec<Table> {
    vec![
        Table {
            name: "mod2_r0",
            owner: (0..E).map(|e| (e % 2) as i32).collect(),
            rank: 0,
        },
        Table {
            name: "mod2_r1",
            owner: (0..E).map(|e| (e % 2) as i32).collect(),
            rank: 1,
        },
        Table {
            name: "mod5_r0",
            owner: (0..E).map(|e| (e % 5) as i32).collect(),
            rank: 0,
        },
        Table {
            name: "all_foreign",
            owner: vec![1; E],
            rank: 0,
        },
        Table {
            name: "all_owned",
            owner: vec![0; E],
            rank: 0,
        },
    ]
}

/// Compare a candidate dump to the reference dump: every output word-for-word except
/// `sorted_slot_index` / `inverse_perm`, which are compared as permutations (see the header).
/// Returns the number of differing words / violated permutation facts.
fn diff(got: &[Vec<u32>], want: &[Vec<u32>], lens: &[usize; 8], slots: usize) -> usize {
    let mut bad = 0usize;
    for i in [0usize, 1, 2, 3, 5, 7] {
        bad += got[i].iter().zip(&want[i]).filter(|(g, w)| g != w).count();
    }
    // Canary tails of the permutation tables.
    for i in [4usize, 6] {
        bad += got[i][lens[i]..]
            .iter()
            .zip(&want[i][lens[i]..])
            .filter(|(g, w)| g != w)
            .count();
    }
    let (offsets, sorted, inverse) = (&want[3], &got[4], &got[6]);
    let (wsorted, winverse) = (&want[4], &want[6]);
    let m_used = offsets[E] as usize;
    for e in 0..E {
        let (lo, hi) = (offsets[e] as usize, offsets[e + 1] as usize);
        let mut x: Vec<u32> = sorted[lo..hi].to_vec();
        let mut y: Vec<u32> = wsorted[lo..hi].to_vec();
        x.sort_unstable();
        y.sort_unstable();
        if x != y {
            bad += 1;
        }
    }
    // Cells past the used grouped rows (the -1 fill up to m_total_max) are strictly equal.
    bad += sorted[m_used..lens[4]]
        .iter()
        .zip(&wsorted[m_used..lens[4]])
        .filter(|(g, w)| g != w)
        .count();
    for i in 0..slots {
        let ok = if inverse[i] == u32::MAX {
            winverse[i] == u32::MAX
        } else {
            (inverse[i] as usize) < sorted.len() && sorted[inverse[i] as usize] == i as u32
        };
        if !ok {
            bad += 1;
        }
    }
    bad
}

#[derive(Default)]
struct Tally {
    cases: usize,
    words: usize,
    bad: usize,
    fault_cases: usize,
    chain_self_diff: usize,
    control_detected: [usize; 5],
    control_total: usize,
}

fn run_reference(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    out: &Outputs,
    rows: usize,
    recipe: (bool, bool),
    layer: usize,
    row_base: usize,
    table: &GpuTensor,
    tab: &Table,
) -> Res<()> {
    let slots = rows * K;
    gpu.moe_router_softmax_top10_f32(
        logits, &out.t[0], &out.t[1], rows, recipe.0, recipe.1, layer, row_base,
    )?;
    gpu.moe_scatter_fused_top10(
        &out.t[0],
        &out.t[2],
        &out.t[3],
        &out.t[4],
        &out.t[5],
        &out.t[6],
        slots,
        E,
        out.grouped_rows,
        BLOCK_M,
    )?;
    gpu.ep_tile_mask(
        &out.t[5],
        &out.t[7],
        table,
        out.grouped_rows / BLOCK_M,
        E,
        tab.rank,
    )?;
    Ok(())
}

fn run_a(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    out: &Outputs,
    rows: usize,
    recipe: (bool, bool),
    layer: usize,
    row_base: usize,
    table: &GpuTensor,
    tab: &Table,
) -> Res<()> {
    let slots = rows * K;
    gpu.moe_router_softmax_top10_f32(
        logits, &out.t[0], &out.t[1], rows, recipe.0, recipe.1, layer, row_base,
    )?;
    gpu.moe_scatter_fused_top10_pmask(
        &out.t[0],
        &out.t[2],
        &out.t[3],
        &out.t[4],
        &out.t[5],
        &out.t[6],
        slots,
        E,
        out.grouped_rows,
        BLOCK_M,
        TileMaskFold {
            tile_ids_owned: &out.t[7],
            owner_table: table,
            tiles: out.grouped_rows / BLOCK_M,
            rank: tab.rank,
        },
    )?;
    Ok(())
}

fn run_b(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    out: &Outputs,
    rows: usize,
    recipe: (bool, bool),
    layer: usize,
    row_base: usize,
    table: Option<(&GpuTensor, usize)>,
) -> Res<bool> {
    Ok(gpu.moe_route_scatter_top10_wide_f32(
        logits,
        &out.t[0],
        &out.t[1],
        rows,
        recipe.0,
        recipe.1,
        layer,
        row_base,
        &out.t[2],
        &out.t[3],
        &out.t[4],
        &out.t[5],
        &out.t[6],
        E,
        K,
        out.grouped_rows,
        BLOCK_M,
        table.map(|(t, rank)| TileMaskFold {
            tile_ids_owned: &out.t[7],
            owner_table: t,
            tiles: out.grouped_rows / BLOCK_M,
            rank,
        }),
    )?)
}

fn oracle() -> Res<()> {
    let mut gpu = Gpu::init()?;
    check_arch(&gpu)?;
    enable_folds(&mut gpu);
    gpu.ensure_router_fault_word()?;
    let seeds: u64 = std::env::var("GLUE_FOLD_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let real: Option<Vec<f32>> = std::env::var("MOE_ROUTE_SCATTER_LOGITS").ok().map(|path| {
        let b = std::fs::read(&path).expect("logits file");
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    });
    let tabs = tables();
    let mut tab_t = Vec::new();
    for t in &tabs {
        let words: Vec<u32> = t.owner.iter().map(|&v| v as u32).collect();
        tab_t.push(upload_u32(&mut gpu, &words)?);
    }

    // Admission edges: nothing is launched outside 9..=64 rows / E512 / top-10 (the wide launch),
    // and with the flag part off.
    {
        let o = Outputs::new(&mut gpu, 64)?;
        let logits = upload_u32(&mut gpu, &vec![0u32; 80 * E])?;
        for (rows, e, k) in [
            (0usize, E, K),
            (8, E, K),
            (65, E, K),
            (16, 256, K),
            (16, E, 8),
        ] {
            let launched = gpu.moe_route_scatter_top10_wide_f32(
                &logits,
                &o.t[0],
                &o.t[1],
                rows,
                true,
                true,
                0,
                0,
                &o.t[2],
                &o.t[3],
                &o.t[4],
                &o.t[5],
                &o.t[6],
                e,
                k,
                o.grouped_rows,
                BLOCK_M,
                None,
            )?;
            assert!(!launched, "rows {rows} e {e} k {k} must decline");
        }
        let mut f: FeatureFlags = (*gpu.flags).clone();
        f.qwen4_moe_glue_fold = 0b101; // a + c: no router fold
        gpu.flags = Arc::new(f);
        assert!(!gpu.moe_glue_fold_route_admitted(16, E, K));
        assert!(gpu.moe_glue_fold_mask_admitted());
        let mut f: FeatureFlags = (*gpu.flags).clone();
        f.qwen4_moe_glue_fold = 0b010; // b only: no mask fold
        gpu.flags = Arc::new(f);
        assert!(gpu.moe_glue_fold_route_admitted(16, E, K));
        assert!(!gpu.moe_glue_fold_mask_admitted());
        enable_folds(&mut gpu);
        assert!(
            gpu.moe_glue_fold_route_admitted(9, E, K) && gpu.moe_glue_fold_route_admitted(64, E, K)
        );
        assert!(
            !gpu.moe_glue_fold_route_admitted(8, E, K)
                && !gpu.moe_glue_fold_route_admitted(65, E, K)
        );
    }

    let mut tally = [Tally::default(), Tally::default()];
    let mut b_nomask_bad = 0usize;
    for rows in 9usize..=64 {
        let slots = rows * K;
        for seed in 0..seeds {
            let mut rng = Lcg(0x57A7_0000 + 1000 * rows as u64 + seed);
            let shared: Vec<f32> = (0..E).map(|_| rng.f32_signed() * 3.0).collect();
            let mut modes: Vec<u32> = (0..rows)
                .map(|r| {
                    // Finite row modes only (0..=3, 7..=9); seeds' last two cases force bad rows below.
                    const FINITE: [u32; 7] = [0, 1, 2, 3, 7, 8, 9];
                    FINITE[((seed as usize / 2) + r * 5 + (rng.next_u32() % 3) as usize) % 7]
                })
                .collect();
            if seed + 2 >= seeds {
                let bad_rows = if seed + 1 == seeds { 3.min(rows) } else { 1 };
                for b in 0..bad_rows {
                    modes[(rng.next_u32() as usize + b) % rows] = 4 + (seed as u32 + b as u32) % 3;
                }
            }
            let mut logits: Vec<f32> = Vec::with_capacity(rows * E);
            for (r, &mode) in modes.iter().enumerate() {
                if let Some(real) = real.as_ref() {
                    let src = ((r + seed as usize) % (real.len() / E)) * E;
                    logits.extend_from_slice(&real[src..][..E]);
                } else {
                    logits.extend(gen_row(&mut rng, mode, &shared));
                }
            }
            let bits: Vec<u32> = logits.iter().map(|v| v.to_bits()).collect();
            let logits_t = upload_u32(&mut gpu, &bits)?;
            let nonfinite_rows: Vec<usize> = (0..rows)
                .filter(|&r| logits[r * E..][..E].iter().any(|v| !v.is_finite()))
                .collect();
            let layer = 3 + (seed as usize % 40);
            let row_base = 17 * (seed as usize % 5);
            for (ri, recipe) in [(true, false), (true, true), (false, false), (false, true)]
                .into_iter()
                .enumerate()
            {
                for (ti, tab) in tabs.iter().enumerate() {
                    // Reference chain (twice: the within-bucket order is an atomic race).
                    let want = Outputs::new(&mut gpu, rows)?;
                    gpu.clear_router_fault();
                    run_reference(
                        &mut gpu, &logits_t, &want, rows, recipe, layer, row_base, &tab_t[ti], tab,
                    )?;
                    gpu.hip.device_synchronize()?;
                    let want_fault = gpu.take_router_fault();
                    let want_d = want.dump(&gpu)?;
                    let want2 = Outputs::new(&mut gpu, rows)?;
                    run_reference(
                        &mut gpu, &logits_t, &want2, rows, recipe, layer, row_base, &tab_t[ti], tab,
                    )?;
                    gpu.hip.device_synchronize()?;
                    let _ = gpu.take_router_fault();
                    let want2_d = want2.dump(&gpu)?;
                    let chain_diff = (want_d[4] != want2_d[4]) || (want_d[6] != want2_d[6]);

                    for (arm, name) in [(0usize, "A"), (1, "B")] {
                        let got = Outputs::new(&mut gpu, rows)?;
                        gpu.clear_router_fault();
                        if arm == 0 {
                            run_a(
                                &mut gpu, &logits_t, &got, rows, recipe, layer, row_base,
                                &tab_t[ti], tab,
                            )?;
                        } else {
                            let launched = run_b(
                                &mut gpu,
                                &logits_t,
                                &got,
                                rows,
                                recipe,
                                layer,
                                row_base,
                                Some((&tab_t[ti], tab.rank)),
                            )?;
                            assert!(launched, "wide launch declined");
                        }
                        gpu.hip.device_synchronize()?;
                        let got_fault = gpu.take_router_fault();
                        let got_d = got.dump(&gpu)?;
                        let t = &mut tally[arm];
                        let n = diff(&got_d, &want_d, &got.lens, slots);
                        if n != 0 {
                            eprintln!(
                                "MISMATCH arm {name} rows {rows} seed {seed} recipe {ri} table {}: {n}",
                                tab.name
                            );
                        }
                        t.bad += n;
                        t.words += got_d.iter().map(|d| d.len()).sum::<usize>();
                        t.cases += 1;
                        if chain_diff {
                            t.chain_self_diff += 1;
                        }
                        // Router fault word: one bad row is deterministic; several race (last
                        // store wins in both), so either must name one of the bad rows.
                        let key = |f: &Option<rdna_compute::router_fault::RouterFault>| {
                            f.as_ref().map(|f| (f.layer, f.row))
                        };
                        if arm == 0 && !nonfinite_rows.is_empty() {
                            t.fault_cases += 1;
                        }
                        if arm == 1 && !nonfinite_rows.is_empty() {
                            t.fault_cases += 1;
                        }
                        if nonfinite_rows.len() <= 1 {
                            if key(&got_fault) != key(&want_fault) {
                                eprintln!(
                                    "MISMATCH arm {name} rows {rows} seed {seed}: fault {:?} vs {:?}",
                                    key(&got_fault),
                                    key(&want_fault)
                                );
                                t.bad += 1;
                            }
                            if got_fault.is_some() != !nonfinite_rows.is_empty() {
                                eprintln!(
                                    "MISMATCH arm {name} rows {rows} seed {seed}: fault presence"
                                );
                                t.bad += 1;
                            }
                        } else {
                            for f in [&got_fault, &want_fault] {
                                let Some(f) = f.as_ref() else {
                                    t.bad += 1;
                                    continue;
                                };
                                if !(nonfinite_rows
                                    .iter()
                                    .any(|&r| f.row == (row_base + r) as u64)
                                    && f.layer as usize == layer)
                                {
                                    eprintln!("MISMATCH arm {name}: fault names no bad row");
                                    t.bad += 1;
                                }
                            }
                        }

                        // Controls (first table of the first seed of every row count, arm A and B).
                        if ti == 0 && seed == 0 && ri == 1 {
                            t.control_total += 1;
                            let mut c;
                            c = got_d.clone();
                            c[1][0] ^= 1; // 1-ulp flip of one routing weight
                            if diff(&c, &want_d, &got.lens, slots) != 0 {
                                t.control_detected[0] += 1;
                            }
                            c = got_d.clone();
                            let live = c[7][..got.lens[7]].iter().position(|&v| v != u32::MAX);
                            if let Some(p) = live {
                                c[7][p] = u32::MAX; // a live owned tile masked out
                            } else {
                                c[7][0] ^= 1;
                            }
                            if diff(&c, &want_d, &got.lens, slots) != 0 {
                                t.control_detected[1] += 1;
                            }
                            c = got_d.clone();
                            c[3][E] += 16; // M_total off by one tile
                            if diff(&c, &want_d, &got.lens, slots) != 0 {
                                t.control_detected[2] += 1;
                            }
                            c = got_d.clone();
                            // swap two slots of DIFFERENT experts inside sorted (breaks buckets)
                            let (mut a, mut b) = (None, None);
                            for (i, &v) in c[4][..got.lens[4]].iter().enumerate() {
                                if v == u32::MAX {
                                    continue;
                                }
                                match a {
                                    None => a = Some(i),
                                    Some(ai) => {
                                        let ea = want_d[0][c[4][ai] as usize];
                                        let ev = want_d[0][v as usize];
                                        if ea != ev {
                                            b = Some(i);
                                            break;
                                        }
                                    }
                                }
                            }
                            if let (Some(a), Some(b)) = (a, b) {
                                c[4].swap(a, b);
                                if diff(&c, &want_d, &got.lens, slots) != 0 {
                                    t.control_detected[3] += 1;
                                }
                            }
                            c = got_d.clone();
                            c[0][0] ^= 1; // a changed expert id
                            if diff(&c, &want_d, &got.lens, slots) != 0 {
                                t.control_detected[4] += 1;
                            }
                        }
                    }

                    // B without a mask: the unmasked tables still equal the reference and the
                    // owned buffer stays canary (nothing writes it).
                    if ti == 0 && ri == 0 && seed == 0 {
                        let got = Outputs::new(&mut gpu, rows)?;
                        gpu.clear_router_fault();
                        let launched = run_b(
                            &mut gpu, &logits_t, &got, rows, recipe, layer, row_base, None,
                        )?;
                        assert!(launched);
                        gpu.hip.device_synchronize()?;
                        let _ = gpu.take_router_fault();
                        let got_d = got.dump(&gpu)?;
                        let mut w = want_d.clone();
                        w[7] = got_d[7].clone();
                        b_nomask_bad += diff(&got_d, &w, &got.lens, slots);
                        if got_d[7].iter().any(|&v| v != CANARY) {
                            b_nomask_bad += 1;
                        }
                    }
                }
            }
        }
    }
    // Repeat stress of arm B (the last-arriving-workgroup scatter): the same call launched back to
    // back, every launch compared to the reference chain (counter hygiene, arrival races).
    let repeats: usize = std::env::var("GLUE_FOLD_REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let mut stress_bad = 0usize;
    let mut stress_launches = 0usize;
    for &rows in &[9usize, 16, 17, 33, 48, 64] {
        let slots = rows * K;
        let mut rng = Lcg(0x5712_E550 + rows as u64);
        let shared: Vec<f32> = (0..E).map(|_| rng.f32_signed() * 3.0).collect();
        let mut logits: Vec<f32> = Vec::new();
        for r in 0..rows {
            logits.extend(gen_row(&mut rng, [0u32, 1, 3, 7, 8][r % 5], &shared));
        }
        let bits: Vec<u32> = logits.iter().map(|v| v.to_bits()).collect();
        let logits_t = upload_u32(&mut gpu, &bits)?;
        let tab = &tabs[0];
        let want = Outputs::new(&mut gpu, rows)?;
        run_reference(
            &mut gpu,
            &logits_t,
            &want,
            rows,
            (true, true),
            5,
            0,
            &tab_t[0],
            tab,
        )?;
        gpu.hip.device_synchronize()?;
        let want_d = want.dump(&gpu)?;
        for _ in 0..repeats {
            let got = Outputs::new(&mut gpu, rows)?;
            let launched = run_b(
                &mut gpu,
                &logits_t,
                &got,
                rows,
                (true, true),
                5,
                0,
                Some((&tab_t[0], tab.rank)),
            )?;
            assert!(launched);
            gpu.hip.device_synchronize()?;
            let got_d = got.dump(&gpu)?;
            stress_bad += diff(&got_d, &want_d, &got.lens, slots);
            stress_launches += 1;
        }
    }
    eprintln!("arm B repeat stress: {stress_launches} launches, mismatches {stress_bad}");
    let mut failed = false;
    for (arm, name) in [
        (0usize, "A (router, scatter+mask)"),
        (1, "B (router+scatter+mask)"),
    ] {
        let t = &tally[arm];
        eprintln!(
            "arm {name}: {} cases (rows 9..=64 x {seeds} seeds x 4 recipes x 5 owner tables, {} with non-finite rows), {} words compared, mismatches {}; chain != chain rerun (sorted/inverse bucket order) in {} cases",
            t.cases, t.fault_cases, t.words, t.bad, t.chain_self_diff
        );
        eprintln!(
            "  controls ({} cells x 5: topk_w 1-ulp, owned tile, M_total, sorted swap, expert id): detected {:?}",
            t.control_total, t.control_detected
        );
        failed |= t.bad != 0 || t.control_detected.iter().any(|&d| d != t.control_total);
    }
    eprintln!("arm B without mask: mismatches {b_nomask_bad}");
    failed |= b_nomask_bad != 0;
    failed |= stress_bad != 0;
    if failed {
        return Err("glue fold oracle FAILED".into());
    }
    eprintln!("glue fold oracle: PASS");
    Ok(())
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn bench() -> Res<()> {
    const SETS: usize = 8;
    const WARM: usize = 30;
    const ITERS: usize = 200;
    const LAYERS: usize = 48;
    let mut gpu = Gpu::init()?;
    check_arch(&gpu)?;
    enable_folds(&mut gpu);
    gpu.ensure_router_fault_word()?;
    let e0 = gpu.hip.event_create()?;
    let e1 = gpu.hip.event_create()?;
    let tabs = tables();
    let table = {
        let words: Vec<u32> = tabs[0].owner.iter().map(|&v| v as u32).collect();
        upload_u32(&mut gpu, &words)?
    };
    let mut json = Vec::new();
    println!("rows | ref chain (router+scatter+mask) | A (router + scatter&mask) | B (wide route+scatter+mask) | A-saved | B-saved   (us/layer, 48-layer loop, median; single-sequence event times after)");
    for &rows in &[9usize, 12, 16, 24, 32, 40, 48, 56, 64] {
        let mut rng = Lcg(0xB3_0000 + rows as u64);
        let shared: Vec<f32> = (0..E).map(|_| rng.f32_signed() * 3.0).collect();
        let mut logits_set = Vec::new();
        let mut outs = Vec::new();
        for s in 0..SETS {
            let mut logits: Vec<f32> = Vec::new();
            for r in 0..rows {
                logits.extend(gen_row(&mut rng, [0u32, 3, 7][(s + r) % 3], &shared));
            }
            let bits: Vec<u32> = logits.iter().map(|v| v.to_bits()).collect();
            logits_set.push(upload_u32(&mut gpu, &bits)?);
            outs.push(Outputs::new(&mut gpu, rows)?);
        }
        let mut ref48 = Vec::new();
        let mut a48 = Vec::new();
        let mut b48 = Vec::new();
        let mut ref1 = Vec::new();
        let mut a1 = Vec::new();
        let mut b1 = Vec::new();
        for it in 0..WARM + ITERS {
            for arm in 0..3 {
                let run = |gpu: &mut Gpu, g: usize, arm: usize| -> Res<()> {
                    let out = &outs[g];
                    match arm {
                        0 => run_reference(
                            gpu,
                            &logits_set[g],
                            out,
                            rows,
                            (true, true),
                            3,
                            0,
                            &table,
                            &tabs[0],
                        ),
                        1 => run_a(
                            gpu,
                            &logits_set[g],
                            out,
                            rows,
                            (true, true),
                            3,
                            0,
                            &table,
                            &tabs[0],
                        ),
                        _ => run_b(
                            gpu,
                            &logits_set[g],
                            out,
                            rows,
                            (true, true),
                            3,
                            0,
                            Some((&table, 0)),
                        )
                        .map(|_| ()),
                    }
                };
                // single sequence
                gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
                run(&mut gpu, (it + arm) % SETS, arm)?;
                gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
                gpu.hip.event_synchronize(&e1)?;
                let one = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0;
                // 48-layer loop
                gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
                for l in 0..LAYERS {
                    run(&mut gpu, (it + l) % SETS, arm)?;
                }
                gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
                gpu.hip.event_synchronize(&e1)?;
                let loopv = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0 / LAYERS as f64;
                if it >= WARM {
                    match arm {
                        0 => {
                            ref1.push(one);
                            ref48.push(loopv)
                        }
                        1 => {
                            a1.push(one);
                            a48.push(loopv)
                        }
                        _ => {
                            b1.push(one);
                            b48.push(loopv)
                        }
                    }
                }
            }
        }
        let (r48, a48m, b48m) = (pct(&mut ref48, 0.5), pct(&mut a48, 0.5), pct(&mut b48, 0.5));
        let (r1, a1m, b1m) = (pct(&mut ref1, 0.5), pct(&mut a1, 0.5), pct(&mut b1, 0.5));
        println!(
            "{rows} | {r48:.1} | {a48m:.1} | {b48m:.1} | {:.1} | {:.1}   (single: {r1:.1} / {a1m:.1} / {b1m:.1})",
            r48 - a48m,
            r48 - b48m
        );
        json.push(format!(
            "{{\"rows\":{rows},\"ref_x48\":{r48:.2},\"a_x48\":{a48m:.2},\"b_x48\":{b48m:.2},\"ref_single\":{r1:.2},\"a_single\":{a1m:.2},\"b_single\":{b1m:.2}}}"
        ));
    }
    println!(
        "{{\"unit\":\"us\",\"stat\":\"median\",\"iters\":{ITERS},\"warmups\":{WARM},\"sets\":{SETS},\"layers\":{LAYERS},\"cells\":[{}]}}",
        json.join(",")
    );
    let _ = gpu.hip.event_destroy(e0);
    let _ = gpu.hip.event_destroy(e1);
    Ok(())
}

fn main() -> Res<()> {
    match std::env::args().nth(1).as_deref() {
        Some("oracle") => oracle(),
        Some("bench") => bench(),
        _ => {
            eprintln!("usage: qwen4_moe_glue_fold oracle | bench");
            std::process::exit(2);
        }
    }
}
