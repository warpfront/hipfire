// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Wide-verify MoE combine with its rank order inline: byte ORACLE + timing bench (exact
//! gfx1201; developer example).
//!
//! `moe_down_combine_grouped_top10_bf16in_ordered[_zinit]` (one launch,
//! `HIPFIRE_QWEN4_MOE_COMBINE_INLINE_ORDER`) against the two-launch pair it replaces
//! (`moe_combine_order_top10_owned` + `moe_down_combine_grouped_top10_bf16in[_zinit]`, i.e.
//! `Gpu::moe_down_combine_grouped_top10_bf16in_owned`).
//!
//! `oracle`: owner tables {mod2 rank0, mod2 rank1, mod5 rank0, none owned, all owned} x tokens
//! {1, 9, 24, 32, 47, 48, 64, 100} x routing {distinct, tied (duplicate experts in a token),
//! hostile (-1 / out-of-range ids, invalid inverse rows, 0 / -0 / large / NaN weights)} x
//! {accumulate (finite dirty residual), zinit (NaN-poisoned residual)}; foreign and padding
//! grouped rows NaN-poisoned (BF16 0xFFFF); every output word (u32 bits) compared to the
//! two-launch pair, and (clean routing) to the unmasked pair over foreign rows cleared to +0;
//! `tokens*hidden` sentinel tail words checked untouched; one negative control (the candidate's
//! owner table has one live expert flipped) proving the comparer is live.
//! `bench`: order kernel, combine kernel, the pair (event-timed), the fused launch, and a
//! 48-layer back-to-back loop of each (per-layer us including the launch gap).
//!
//! Usage: qwen4_moe_combine_ordered oracle | bench

use rdna_compute::hc_row_fold::CombineOwners;
use rdna_compute::{DType, Gpu, GpuTensor};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const HIDDEN: usize = 2560;
const N_EXP: usize = 512;
const TAIL_WORDS: usize = 4096;
const SENTINEL: u32 = 0xDEAD_BEEF;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Routing {
    Distinct,
    Tied,
    Hostile,
}

struct Route {
    experts: Vec<i32>,
    weights: Vec<f32>,
    inverse: Vec<i32>,
    /// owning expert of each grouped row (-1 padding)
    row_expert: Vec<i32>,
    grouped_rows: usize,
}

fn build_route(tokens: usize, routing: Routing, seed: u64) -> Route {
    let slots = tokens * 10;
    let mut r = Rng(seed ^ (tokens as u64) << 8);
    let mut experts = vec![-1i32; slots];
    for t in 0..tokens {
        for k in 0..10 {
            let slot = t * 10 + k;
            if routing != Routing::Distinct && slot % 17 == 0 {
                continue; // stays -1: unrouted
            }
            loop {
                let e = r.below(N_EXP) as i32;
                let dup_ok = routing == Routing::Tied && k % 3 == 2 && k > 0;
                if dup_ok {
                    experts[slot] = experts[slot - 1 - r.below(k)];
                    if experts[slot] < 0 {
                        experts[slot] = e;
                    }
                    break;
                }
                if !experts[t * 10..slot].contains(&e) {
                    experts[slot] = e;
                    break;
                }
            }
        }
    }
    if routing == Routing::Hostile {
        for s in (5..slots).step_by(29) {
            experts[s] = N_EXP as i32 + (s % 7) as i32; // out of range: unrouted
        }
    }
    let mut weights: Vec<f32> = (0..slots)
        .map(|_| (r.below(3001) as f32 - 1500.0) / 1000.0)
        .collect();
    if routing == Routing::Hostile {
        for (i, w) in weights.iter_mut().enumerate() {
            match i % 31 {
                3 => *w = 0.0,
                7 => *w = -0.0,
                11 => *w = 3.0e38,
                19 => *w = f32::NAN,
                23 => *w = f32::INFINITY,
                _ => {}
            }
        }
    }
    // Grouped layout: experts ascending, groups padded to 16 rows.
    let mut inverse = vec![-1i32; slots];
    let mut row_expert: Vec<i32> = Vec::new();
    for e in 0..N_EXP as i32 {
        let start = row_expert.len();
        for s in 0..slots {
            if experts[s] == e {
                inverse[s] = row_expert.len() as i32;
                row_expert.push(e);
            }
        }
        if row_expert.len() > start {
            row_expert.resize(start + (row_expert.len() - start).div_ceil(16) * 16, -1);
        }
    }
    let grouped_rows = row_expert.len().max(16);
    row_expert.resize(grouped_rows, -1);
    if routing == Routing::Hostile {
        for s in (2..slots).step_by(37) {
            inverse[s] = if s % 2 == 0 { -1 } else { grouped_rows as i32 + 5 };
        }
    }
    Route {
        experts,
        weights,
        inverse,
        row_expert,
        grouped_rows,
    }
}

fn bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn upload_words(gpu: &mut Gpu, d: &[u32]) -> Res<GpuTensor> {
    let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    Ok(gpu.upload_raw(&bytes, &[bytes.len()])?)
}

fn download_words(gpu: &Gpu, t: &GpuTensor, words: usize) -> Res<Vec<u32>> {
    let mut b = vec![0u8; words * 4];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf)?;
    Ok(b
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// BF16 grouped rows: owned rows deterministic finite, everything else `foreign`.
fn grouped_bf16(route: &Route, table: &[i32], rank: i32, foreign: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(route.grouped_rows * HIDDEN * 2);
    for (row, &e) in route.row_expert.iter().enumerate() {
        let owned = e >= 0 && table[e as usize] == rank;
        for c in 0..HIDDEN {
            let bits: u16 = if owned {
                let h = (row * 7919 + c * 31 + 1) % 4093;
                let v = (h as f32 + 1.0) / 1024.0 * if h % 3 == 0 { -1.0 } else { 1.0 };
                (v.to_bits() >> 16) as u16
            } else {
                foreign
            };
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    out
}

fn residual_words(tokens: usize, kind: &str) -> Vec<u32> {
    let mut w: Vec<u32> = (0..tokens * HIDDEN)
        .map(|i| match kind {
            "poison" => 0xFFFF_FFFFu32,
            _ => {
                let v = (i % 977) as f32 / 61.0 - 7.0;
                if i % 1013 == 5 {
                    (-0.0f32).to_bits()
                } else {
                    v.to_bits()
                }
            }
        })
        .collect();
    w.extend(std::iter::repeat(SENTINEL).take(TAIL_WORDS));
    w
}

#[derive(Default)]
struct Totals {
    cells: usize,
    words: u64,
    mismatches: u64,
    tail_words: u64,
    tail_violations: u64,
    nan_words: u64,
}

fn check_arch(gpu: &Gpu) -> Res<()> {
    if !gpu.arch_caps.is_gfx1201() {
        return Err("requires exact gfx1201".into());
    }
    Ok(())
}

fn cmp(
    what: &str,
    label: &str,
    n: usize,
    want: &[u32],
    got: &[u32],
    t: &mut Totals,
    fails: &mut Vec<String>,
) {
    t.cells += 1;
    t.words += n as u64;
    let mm = want[..n].iter().zip(&got[..n]).filter(|(a, b)| a != b).count();
    t.mismatches += mm as u64;
    let tv = got[n..].iter().filter(|&&w| w != SENTINEL).count();
    t.tail_words += TAIL_WORDS as u64;
    t.tail_violations += tv as u64;
    if mm != 0 || tv != 0 {
        fails.push(format!("{what} {label}: {mm} mismatches, {tv} tail"));
    }
}

fn oracle() -> Res<()> {
    let mut gpu = Gpu::init()?;
    check_arch(&gpu)?;
    type OwnerFn = fn(usize) -> i32;
    let tables: [(&str, OwnerFn, i32); 5] = [
        ("mod2_r0", |e| (e % 2) as i32, 0),
        ("mod2_r1", |e| (e % 2) as i32, 1),
        ("mod5_r0", |e| if e % 5 == 0 { 0 } else { 1 }, 0),
        ("none_owned", |_| 1, 0),
        ("all_owned", |_| 0, 0),
    ];
    let mut t = Totals::default();
    let mut fails: Vec<String> = Vec::new();
    for &tokens in &[1usize, 9, 24, 32, 47, 48, 64, 100] {
        for (ri, routing) in [Routing::Distinct, Routing::Tied, Routing::Hostile]
            .into_iter()
            .enumerate()
        {
            let route = build_route(tokens, routing, 0x1000 + ri as u64);
            let slots = tokens * 10;
            let inverse_t = gpu.upload_raw(&bytes_i32(&route.inverse), &[slots * 4])?;
            let experts_t = gpu.upload_raw(&bytes_i32(&route.experts), &[slots * 4])?;
            let weights_t = gpu.upload_f32(&route.weights, &[slots])?;
            let order_t = gpu.upload_raw(&vec![0u8; slots * 8], &[slots * 8])?;
            for (name, owner_of, rank) in tables {
                let table: Vec<i32> = (0..N_EXP).map(owner_of).collect();
                let owner_t = gpu.upload_raw(&bytes_i32(&table), &[N_EXP * 4])?;
                let poisoned = grouped_bf16(&route, &table, rank, 0xFFFF);
                let cleared = grouped_bf16(&route, &table, rank, 0x0000);
                let poisoned_t = gpu.upload_raw(&poisoned, &[poisoned.len()])?;
                let cleared_t = gpu.upload_raw(&cleared, &[cleared.len()])?;
                let owners = CombineOwners {
                    table: &owner_t,
                    rank: rank as usize,
                };
                for initial_zero in [false, true] {
                    let kind = if initial_zero { "poison" } else { "dirty" };
                    let init = residual_words(tokens, kind);
                    let label = format!(
                        "{name} tokens={tokens} routing={} zinit={initial_zero}",
                        ["distinct", "tied", "hostile"][ri]
                    );
                    // Reference: the two-launch owned pair over the poisoned grouped rows.
                    let pair_t = upload_words(&mut gpu, &init)?;
                    gpu.moe_down_combine_grouped_top10_bf16in_owned(
                        &poisoned_t,
                        &inverse_t,
                        &experts_t,
                        &weights_t,
                        &pair_t,
                        &order_t,
                        HIDDEN,
                        route.grouped_rows,
                        tokens,
                        initial_zero,
                        N_EXP,
                        Some(owners),
                    )?;
                    // Candidate: the fused launch over the same poisoned rows.
                    let fused_t = upload_words(&mut gpu, &init)?;
                    gpu.moe_down_combine_grouped_top10_bf16in_ordered(
                        &poisoned_t,
                        &inverse_t,
                        &experts_t,
                        &weights_t,
                        &fused_t,
                        HIDDEN,
                        route.grouped_rows,
                        tokens,
                        initial_zero,
                        N_EXP,
                        owners,
                    )?;
                    gpu.hip.device_synchronize()?;
                    let n = tokens * HIDDEN;
                    let pair = download_words(&gpu, &pair_t, n + TAIL_WORDS)?;
                    let fused = download_words(&gpu, &fused_t, n + TAIL_WORDS)?;
                    cmp("fused-vs-pair", &label, n, &pair, &fused, &mut t, &mut fails);
                    // The fused result carries no NaN of the poisoned rows unless a hostile
                    // weight (NaN / inf x 0) put it there; count them only for clean routing.
                    if routing != Routing::Hostile {
                        let nans = fused[..n].iter().filter(|w| f32::from_bits(**w).is_nan()).count();
                        t.nan_words += nans as u64;
                        if nans != 0 {
                            fails.push(format!("NaN leaked {label}: {nans}"));
                        }
                    }
                    // Clean routing: also the unmasked pair over foreign rows cleared to +0.
                    if routing != Routing::Hostile {
                        let base_t = upload_words(&mut gpu, &init)?;
                        gpu.moe_down_combine_grouped_top10_bf16in(
                            &cleared_t,
                            &inverse_t,
                            &experts_t,
                            &weights_t,
                            &base_t,
                            &order_t,
                            HIDDEN,
                            route.grouped_rows,
                            tokens,
                            initial_zero,
                            N_EXP,
                        )?;
                        gpu.hip.device_synchronize()?;
                        let base = download_words(&gpu, &base_t, n + TAIL_WORDS)?;
                        cmp("fused-vs-unmasked-cleared", &label, n, &base, &fused, &mut t, &mut fails);
                        gpu.free_tensor(base_t)?;
                    }
                    gpu.free_tensor(pair_t)?;
                    gpu.free_tensor(fused_t)?;
                }
                gpu.free_tensor(owner_t)?;
                gpu.free_tensor(poisoned_t)?;
                gpu.free_tensor(cleared_t)?;
            }
            gpu.free_tensor(inverse_t)?;
            gpu.free_tensor(experts_t)?;
            gpu.free_tensor(weights_t)?;
            gpu.free_tensor(order_t)?;
        }
    }

    // Negative control: the candidate sees an owner table with one live expert flipped; the
    // comparer must report mismatches (so a green run above means something).
    let route = build_route(48, Routing::Distinct, 0x1000);
    let slots = 480;
    let table: Vec<i32> = (0..N_EXP).map(|e| (e % 2) as i32).collect();
    let live_expert = route
        .experts
        .iter()
        .copied()
        .find(|&e| e >= 0 && table[e as usize] == 0)
        .ok_or("no live owned expert")?;
    let mut flipped = table.clone();
    flipped[live_expert as usize] = 1;
    let inverse_t = gpu.upload_raw(&bytes_i32(&route.inverse), &[slots * 4])?;
    let experts_t = gpu.upload_raw(&bytes_i32(&route.experts), &[slots * 4])?;
    let weights_t = gpu.upload_f32(&route.weights, &[slots])?;
    let order_t = gpu.upload_raw(&vec![0u8; slots * 8], &[slots * 8])?;
    let owner_t = gpu.upload_raw(&bytes_i32(&table), &[N_EXP * 4])?;
    let owner_flip_t = gpu.upload_raw(&bytes_i32(&flipped), &[N_EXP * 4])?;
    let g = grouped_bf16(&route, &table, 0, 0xFFFF);
    let grouped_t = gpu.upload_raw(&g, &[g.len()])?;
    let init = residual_words(48, "dirty");
    let pair_t = upload_words(&mut gpu, &init)?;
    let fused_t = upload_words(&mut gpu, &init)?;
    gpu.moe_down_combine_grouped_top10_bf16in_owned(
        &grouped_t,
        &inverse_t,
        &experts_t,
        &weights_t,
        &pair_t,
        &order_t,
        HIDDEN,
        route.grouped_rows,
        48,
        false,
        N_EXP,
        Some(CombineOwners {
            table: &owner_t,
            rank: 0,
        }),
    )?;
    gpu.moe_down_combine_grouped_top10_bf16in_ordered(
        &grouped_t,
        &inverse_t,
        &experts_t,
        &weights_t,
        &fused_t,
        HIDDEN,
        route.grouped_rows,
        48,
        false,
        N_EXP,
        CombineOwners {
            table: &owner_flip_t,
            rank: 0,
        },
    )?;
    gpu.hip.device_synchronize()?;
    let n = 48 * HIDDEN;
    let a = download_words(&gpu, &pair_t, n)?;
    let b = download_words(&gpu, &fused_t, n)?;
    let control = a.iter().zip(&b).filter(|(x, y)| x != y).count();

    println!(
        "ORACLE cells={} words={} mismatches={} tail_words={} tail_violations={} nan_words={} negative_control_mismatches={} control={}",
        t.cells,
        t.words,
        t.mismatches,
        t.tail_words,
        t.tail_violations,
        t.nan_words,
        control,
        if control > 0 { "PASS" } else { "FAIL" }
    );
    for f in fails.iter().take(20) {
        println!("FAIL {f}");
    }
    if t.mismatches != 0 || t.tail_violations != 0 || t.nan_words != 0 || control == 0 {
        return Err("oracle failed".into());
    }
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
    let e0 = gpu.hip.event_create()?;
    let e1 = gpu.hip.event_create()?;
    let table: Vec<i32> = (0..N_EXP).map(|e| (e % 2) as i32).collect();
    let owner_t = gpu.upload_raw(&bytes_i32(&table), &[N_EXP * 4])?;
    let owners = CombineOwners {
        table: &owner_t,
        rank: 0,
    };
    let mut json: Vec<String> = Vec::new();
    println!("rows zinit | order | combine | pair(events) | fused | saved | pair x48/layer | fused x48/layer | saved/layer   (us, median)");
    for &tokens in &[24usize, 32, 48] {
        let route = build_route(tokens, Routing::Distinct, 0x77);
        let slots = tokens * 10;
        // SETS rotating (route, grouped) buffers so each launch reads cold-ish rows.
        let grouped: Vec<u8> = grouped_bf16(&route, &table, 0, 0x0000);
        let mut gs = Vec::new();
        let mut inv = Vec::new();
        let mut exp = Vec::new();
        let mut wts = Vec::new();
        for _ in 0..SETS {
            gs.push(gpu.upload_raw(&grouped, &[grouped.len()])?);
            inv.push(gpu.upload_raw(&bytes_i32(&route.inverse), &[slots * 4])?);
            exp.push(gpu.upload_raw(&bytes_i32(&route.experts), &[slots * 4])?);
            wts.push(gpu.upload_f32(&route.weights, &[slots])?);
        }
        let order_t = gpu.upload_raw(&vec![0u8; slots * 8], &[slots * 8])?;
        let target = gpu.alloc_tensor(&[tokens * HIDDEN], DType::F32)?;
        for initial_zero in [false, true] {
            let mut orders = Vec::new();
            let mut combines = Vec::new();
            let mut pairs = Vec::new();
            let mut fuseds = Vec::new();
            let mut pair48 = Vec::new();
            let mut fused48 = Vec::new();
            for it in 0..WARM + ITERS {
                let s = it % SETS;
                let sf = (it + 5) % SETS;
                // order alone
                gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
                gpu.moe_combine_order_top10_owned(
                    &inv[s], &exp[s], &wts[s], &order_t, route.grouped_rows, tokens, N_EXP,
                    Some(owners),
                )?;
                gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
                gpu.hip.event_synchronize(&e1)?;
                let order_us = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0;
                // the pair (`_owned` launches the order kernel, then the combine); the
                // combine alone is pair - order
                let sc = (it + 1) % SETS;
                gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
                gpu.moe_down_combine_grouped_top10_bf16in_owned(
                    &gs[sc], &inv[s], &exp[s], &wts[s], &target, &order_t, HIDDEN,
                    route.grouped_rows, tokens, initial_zero, N_EXP, Some(owners),
                )?;
                gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
                gpu.hip.event_synchronize(&e1)?;
                let pair_us = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0;
                gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
                gpu.moe_down_combine_grouped_top10_bf16in_ordered(
                    &gs[sf], &inv[s], &exp[s], &wts[s], &target, HIDDEN, route.grouped_rows,
                    tokens, initial_zero, N_EXP, owners,
                )?;
                gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
                gpu.hip.event_synchronize(&e1)?;
                let fused_us = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0;
                // 48-layer back-to-back loops
                gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
                for l in 0..LAYERS {
                    let g = (it + l) % SETS;
                    gpu.moe_down_combine_grouped_top10_bf16in_owned(
                        &gs[g], &inv[g], &exp[g], &wts[g], &target, &order_t, HIDDEN,
                        route.grouped_rows, tokens, initial_zero, N_EXP, Some(owners),
                    )?;
                }
                gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
                gpu.hip.event_synchronize(&e1)?;
                let p48 = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0 / LAYERS as f64;
                gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
                for l in 0..LAYERS {
                    let g = (it + l) % SETS;
                    gpu.moe_down_combine_grouped_top10_bf16in_ordered(
                        &gs[g], &inv[g], &exp[g], &wts[g], &target, HIDDEN, route.grouped_rows,
                        tokens, initial_zero, N_EXP, owners,
                    )?;
                }
                gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
                gpu.hip.event_synchronize(&e1)?;
                let f48 = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0 / LAYERS as f64;
                if it >= WARM {
                    orders.push(order_us);
                    combines.push(pair_us - order_us);
                    pairs.push(pair_us);
                    fuseds.push(fused_us);
                    pair48.push(p48);
                    fused48.push(f48);
                }
            }
            let (o, c, p, f, p48, f48) = (
                pct(&mut orders, 0.5),
                pct(&mut combines, 0.5),
                pct(&mut pairs, 0.5),
                pct(&mut fuseds, 0.5),
                pct(&mut pair48, 0.5),
                pct(&mut fused48, 0.5),
            );
            println!(
                "{tokens} {initial_zero} | {o:.1} | {c:.1} | {p:.1} | {f:.1} | {:.1} | {p48:.1} | {f48:.1} | {:.1}",
                p - f,
                p48 - f48
            );
            json.push(format!(
                "{{\"rows\":{tokens},\"zinit\":{initial_zero},\"order\":{o:.2},\"combine\":{c:.2},\"pair\":{p:.2},\"fused\":{f:.2},\"pair_x48\":{p48:.2},\"fused_x48\":{f48:.2}}}"
            ));
        }
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
            eprintln!("usage: qwen4_moe_combine_ordered oracle | bench");
            std::process::exit(2);
        }
    }
}
