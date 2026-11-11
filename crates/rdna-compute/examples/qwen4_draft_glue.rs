// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MTP draft-head launch-glue fold (`HIPFIRE_QWEN4_DRAFT_GLUE_FOLD`): byte ORACLE + timing bench
//! (exact gfx1201; developer example). Every fold replaces N existing launches by one launch of an
//! EXISTING entry, so the oracle compares the folded launch to the N launches it replaces, word
//! for word, over the served head shapes:
//!
//! * `broadcast`: one `broadcast_add_rows_f32` over `lanes` lane rows (4 HC branch rows of 2560
//!   each, one 2560-wide embedding row per lane) against `4 * lanes` per-branch `add_f32`
//!   launches (`execute_broadcast_add`'s unfolded loop), lanes 1..=8.
//! * `rotate`: one `rotate_x_mq_batched` over `len` consecutive 2560-wide rows against `len`
//!   `rotate_x_mq` launches (the draft ranking's per-row rotation), len 1..=8.
//! * `copy`: one `copy_regions` over N tensors against N `copy_d2d` launches: the batched round's
//!   per-lane staging (lane tensor -> row of the rows scratch, row -> lane tensor, 40960 B, lanes
//!   2..=8) and the head snapshot / restore triple (16384 B + 4 B + 40960 B).
//!
//! Each cell: three data classes (random, special words: NaN payloads / inf / +-0 / subnormals,
//! cancellation pairs) x two dirty output fills (all-ones NaN, an alternating pattern); `TAIL`
//! sentinel words after every output checked untouched; every output word (u32 bits) compared;
//! one injected-change control per family (the folded output with one 1-ulp bit flipped must be
//! detected, and the folded launch with the wrong lane order must mismatch).
//! `bench`: per-round us (median of 200, event-timed, back-to-back) of the N-launch sequence and
//! the folded launch.
//!
//! Usage: qwen4_draft_glue oracle | bench

use rdna_compute::tensor_ops::{copy_regions, CopyRegion};
use rdna_compute::{Gpu, GpuTensor};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const HIDDEN: usize = 2560;
const BRANCHES: usize = 4;
const WIDE: usize = BRANCHES * HIDDEN;
const TAIL: usize = 4096;
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
}

#[derive(Clone, Copy)]
enum Data {
    Random,
    Special,
    Cancel,
}

const DATA: [(&str, Data); 3] = [
    ("random", Data::Random),
    ("special", Data::Special),
    ("cancel", Data::Cancel),
];

const SPECIALS: [u32; 12] = [
    0x7FC0_0000, // qNaN
    0xFFC0_0001, // -qNaN payload
    0x7F80_0001, // sNaN
    0x7F80_0000, // +inf
    0xFF80_0000, // -inf
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x0000_0001, // smallest subnormal
    0x807F_FFFF, // -largest subnormal
    0x7F7F_FFFF, // max finite
    0x0080_0000, // min normal
    0x3F80_0000, // 1.0
];

/// `n` words of `data`; `seed` varies per call. `Cancel` pairs +x / -x so sums cancel exactly.
fn words(n: usize, data: Data, seed: u64) -> Vec<u32> {
    let mut r = Rng(seed);
    (0..n)
        .map(|i| match data {
            Data::Random => {
                let v = (r.next() % 20001) as f32 / 1000.0 - 10.0;
                v.to_bits()
            }
            Data::Special => {
                if i % 3 == 0 {
                    SPECIALS[(r.next() % SPECIALS.len() as u64) as usize]
                } else {
                    let v = (r.next() % 2001) as f32 / 100.0 - 10.0;
                    v.to_bits()
                }
            }
            Data::Cancel => {
                let base = ((i / 2) % 4093) as f32 / 8.0 + 0.125;
                if i % 2 == 0 {
                    base.to_bits()
                } else {
                    (-base).to_bits()
                }
            }
        })
        .collect()
}

fn fill_words(n: usize, kind: usize) -> Vec<u32> {
    let mut w: Vec<u32> = (0..n)
        .map(|i| match kind {
            0 => 0xFFFF_FFFF,
            _ => {
                if i % 2 == 0 {
                    0xFFFF_FFFF
                } else {
                    0x7FC0_0001
                }
            }
        })
        .collect();
    w.extend(std::iter::repeat(SENTINEL).take(TAIL));
    w
}

fn upload_words(gpu: &mut Gpu, w: &[u32]) -> Res<GpuTensor> {
    let f: Vec<f32> = w.iter().map(|&b| f32::from_bits(b)).collect();
    Ok(gpu.upload_f32(&f, &[w.len()])?)
}

fn download_words(gpu: &Gpu, t: &GpuTensor) -> Res<Vec<u32>> {
    Ok(gpu.download_f32(t)?.into_iter().map(f32::to_bits).collect())
}

#[derive(Default)]
struct Totals {
    cells: usize,
    words: u64,
    mismatches: u64,
    tail_words: u64,
    tail_violations: u64,
    controls: usize,
    controls_detected: usize,
}

fn compare(
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
    let mm = want[..n]
        .iter()
        .zip(&got[..n])
        .filter(|(a, b)| a != b)
        .count();
    t.mismatches += mm as u64;
    let tv = got[n..n + TAIL].iter().filter(|&&w| w != SENTINEL).count();
    t.tail_words += TAIL as u64;
    t.tail_violations += tv as u64;
    if mm != 0 || tv != 0 {
        fails.push(format!("{what} {label}: {mm} mismatches, {tv} tail"));
    }
}

/// Controls: the folded output with one 1-ulp bit flipped, and a lane-order swap, must differ.
fn control(what: &str, n: usize, want: &[u32], got: &[u32], lane_words: usize, t: &mut Totals) {
    let mut flipped = got[..n].to_vec();
    let at = n / 2;
    flipped[at] ^= 1;
    t.controls += 1;
    if flipped != want[..n] {
        t.controls_detected += 1;
    } else {
        eprintln!("control missed: {what} 1-ulp flip");
    }
    if n >= 2 * lane_words {
        let mut swapped = got[..n].to_vec();
        for i in 0..lane_words {
            swapped.swap(i, lane_words + i);
        }
        // A swap of two identical lane rows changes nothing: it is no control then.
        if swapped != got[..n] {
            t.controls += 1;
            if swapped != want[..n] {
                t.controls_detected += 1;
            } else {
                eprintln!("control missed: {what} lane swap");
            }
        }
    }
}

fn check_arch(gpu: &Gpu) -> Res<()> {
    if !gpu.arch_caps.is_gfx1201() {
        return Err("requires exact gfx1201".into());
    }
    Ok(())
}

fn oracle_broadcast(gpu: &mut Gpu, t: &mut Totals, fails: &mut Vec<String>) -> Res<()> {
    for lanes in 1..=8usize {
        for (dname, data) in DATA {
            for fill in 0..2usize {
                let a_w = words(lanes * WIDE, data, 0xA0 + lanes as u64);
                let b_w = words(lanes * HIDDEN, data, 0xB0 + lanes as u64);
                let a = upload_words(gpu, &a_w)?;
                let b = upload_words(gpu, &b_w)?;
                let n = lanes * WIDE;
                let fill_w = fill_words(n, fill);
                let want_t = upload_words(gpu, &fill_w)?;
                let got_t = upload_words(gpu, &fill_w)?;
                // The unfolded loop of `execute_broadcast_add`: one `add_f32` per branch row.
                for l in 0..lanes {
                    let row = b.sub_offset(l * HIDDEN, HIDDEN);
                    for r in 0..BRANCHES {
                        let off = (l * BRANCHES + r) * HIDDEN;
                        gpu.add_f32(
                            &a.sub_offset(off, HIDDEN),
                            &row,
                            &want_t.sub_offset(off, HIDDEN),
                        )?;
                    }
                }
                gpu.broadcast_add_rows_f32(&a, &b, &got_t, BRANCHES, HIDDEN, lanes)?;
                gpu.hip.device_synchronize()?;
                let want = download_words(gpu, &want_t)?;
                let got = download_words(gpu, &got_t)?;
                compare(
                    "broadcast",
                    &format!("lanes {lanes} {dname} fill{fill}"),
                    n,
                    &want,
                    &got,
                    t,
                    fails,
                );
                if fill == 0 {
                    control("broadcast", n, &want, &got, WIDE, t);
                }
            }
        }
    }
    Ok(())
}

fn oracle_rotate(gpu: &mut Gpu, t: &mut Totals, fails: &mut Vec<String>) -> Res<()> {
    for len in 1..=8usize {
        for (dname, data) in DATA {
            for fill in 0..2usize {
                let x_w = words(len * HIDDEN, data, 0xC0 + len as u64);
                let x = upload_words(gpu, &x_w)?;
                let n = len * HIDDEN;
                let fill_w = fill_words(n, fill);
                let want_t = upload_words(gpu, &fill_w)?;
                let got_t = upload_words(gpu, &fill_w)?;
                for row in 0..len {
                    gpu.rotate_x_mq(
                        &x.sub_offset(row * HIDDEN, HIDDEN),
                        &want_t.sub_offset(row * HIDDEN, HIDDEN),
                        HIDDEN,
                    )?;
                }
                gpu.rotate_x_mq_batched(&x, &got_t, HIDDEN, len)?;
                gpu.hip.device_synchronize()?;
                let want = download_words(gpu, &want_t)?;
                let got = download_words(gpu, &got_t)?;
                compare(
                    "rotate",
                    &format!("len {len} {dname} fill{fill}"),
                    n,
                    &want,
                    &got,
                    t,
                    fails,
                );
                if fill == 0 {
                    control("rotate", n, &want, &got, HIDDEN, t);
                }
            }
        }
    }
    Ok(())
}

fn oracle_copy(gpu: &mut Gpu, t: &mut Totals, fails: &mut Vec<String>) -> Res<()> {
    // Lane staging: N lane tensors -> N rows of one tensor, then N rows -> N lane tensors.
    for lanes in 2..=8usize {
        for (dname, data) in DATA {
            for fill in 0..2usize {
                let mut srcs = Vec::new();
                let mut src_w = Vec::new();
                for l in 0..lanes {
                    let w = words(WIDE, data, 0xD0 + (lanes * 16 + l) as u64);
                    srcs.push(upload_words(gpu, &w)?);
                    src_w.extend(w);
                }
                let n = lanes * WIDE;
                let fill_w = fill_words(n, fill);
                let want_t = upload_words(gpu, &fill_w)?;
                let got_t = upload_words(gpu, &fill_w)?;
                for (l, src) in srcs.iter().enumerate() {
                    gpu.copy_d2d(src, &want_t.sub_offset(l * WIDE, WIDE), WIDE * 4)?;
                }
                let dsts: Vec<GpuTensor> = (0..lanes)
                    .map(|l| got_t.sub_offset(l * WIDE, WIDE))
                    .collect();
                let regions: Vec<CopyRegion<'_>> = srcs
                    .iter()
                    .zip(&dsts)
                    .map(|(s, d)| CopyRegion {
                        dst: &d.buf,
                        dst_offset: 0,
                        src: &s.buf,
                        src_offset: 0,
                        bytes: WIDE * 4,
                    })
                    .collect();
                copy_regions(gpu, &regions)?;
                gpu.hip.device_synchronize()?;
                let want = download_words(gpu, &want_t)?;
                let got = download_words(gpu, &got_t)?;
                compare(
                    "stage-in",
                    &format!("lanes {lanes} {dname} fill{fill}"),
                    n,
                    &want,
                    &got,
                    t,
                    fails,
                );
                if fill == 0 {
                    control("stage-in", n, &want, &got, WIDE, t);
                }
                // The words are the sources' (the unfolded copy is exact by construction).
                if got[..n] != src_w[..] {
                    fails.push(format!(
                        "stage-in lanes {lanes} {dname}: not the source words"
                    ));
                    t.mismatches += 1;
                }

                // Epilogue direction: rows of a source tensor -> N separate lane tensors.
                let rows_t = upload_words(gpu, &src_w)?;
                let mut want_d = Vec::new();
                let mut got_d = Vec::new();
                for _ in 0..lanes {
                    want_d.push(upload_words(gpu, &fill_words(WIDE, fill))?);
                    got_d.push(upload_words(gpu, &fill_words(WIDE, fill))?);
                }
                let srows: Vec<GpuTensor> = (0..lanes)
                    .map(|l| rows_t.sub_offset(l * WIDE, WIDE))
                    .collect();
                for (s, d) in srows.iter().zip(&want_d) {
                    gpu.copy_d2d(s, d, WIDE * 4)?;
                }
                let regions: Vec<CopyRegion<'_>> = srows
                    .iter()
                    .zip(&got_d)
                    .map(|(s, d)| CopyRegion {
                        dst: &d.buf,
                        dst_offset: 0,
                        src: &s.buf,
                        src_offset: 0,
                        bytes: WIDE * 4,
                    })
                    .collect();
                copy_regions(gpu, &regions)?;
                gpu.hip.device_synchronize()?;
                for l in 0..lanes {
                    let want = download_words(gpu, &want_d[l])?;
                    let got = download_words(gpu, &got_d[l])?;
                    compare(
                        "stage-out",
                        &format!("lanes {lanes} lane {l} {dname} fill{fill}"),
                        WIDE,
                        &want,
                        &got,
                        t,
                        fails,
                    );
                }
            }
        }
    }
    // Head snapshot / restore triple: selected indices (16384 B), the length word (4 B), the wide
    // hidden (40960 B), both directions.
    let sizes = [4096usize, 1, WIDE];
    for (dname, data) in DATA {
        for fill in 0..2usize {
            let mut live = Vec::new();
            let mut live_w = Vec::new();
            for (i, &n) in sizes.iter().enumerate() {
                let w = words(n, data, 0xE0 + i as u64);
                live.push(upload_words(gpu, &w)?);
                live_w.push(w);
            }
            let mut want = Vec::new();
            let mut got = Vec::new();
            for &n in &sizes {
                want.push(upload_words(gpu, &fill_words(n, fill))?);
                got.push(upload_words(gpu, &fill_words(n, fill))?);
            }
            for (i, &n) in sizes.iter().enumerate() {
                gpu.copy_d2d(&live[i], &want[i], n * 4)?;
            }
            let regions: Vec<CopyRegion<'_>> = (0..sizes.len())
                .map(|i| CopyRegion {
                    dst: &got[i].buf,
                    dst_offset: 0,
                    src: &live[i].buf,
                    src_offset: 0,
                    bytes: sizes[i] * 4,
                })
                .collect();
            copy_regions(gpu, &regions)?;
            gpu.hip.device_synchronize()?;
            for (i, &n) in sizes.iter().enumerate() {
                let w = download_words(gpu, &want[i])?;
                let g = download_words(gpu, &got[i])?;
                compare(
                    "snapshot",
                    &format!("tensor {i} {dname} fill{fill}"),
                    n,
                    &w,
                    &g,
                    t,
                    fails,
                );
                if g[..n] != live_w[i][..] {
                    fails.push(format!("snapshot tensor {i} {dname}: not the source words"));
                    t.mismatches += 1;
                }
            }
        }
    }
    Ok(())
}

fn oracle() -> Res<()> {
    let mut gpu = Gpu::init()?;
    check_arch(&gpu)?;
    let mut t = Totals::default();
    let mut fails: Vec<String> = Vec::new();
    oracle_broadcast(&mut gpu, &mut t, &mut fails)?;
    let b = (t.cells, t.words, t.mismatches);
    oracle_rotate(&mut gpu, &mut t, &mut fails)?;
    let r = (t.cells - b.0, t.words - b.1, t.mismatches - b.2);
    oracle_copy(&mut gpu, &mut t, &mut fails)?;
    println!(
        "draft glue oracle: broadcast {} cells {} words; rotate {} cells {} words; copy {} cells {} words",
        b.0,
        b.1,
        r.0,
        r.1,
        t.cells - b.0 - r.0,
        t.words - b.1 - r.1
    );
    println!(
        "cells {} words {} mismatches {} sentinel tail words {} violations {} controls {}/{} detected",
        t.cells, t.words, t.mismatches, t.tail_words, t.tail_violations, t.controls_detected, t.controls
    );
    for f in fails.iter().take(40) {
        println!("FAIL {f}");
    }
    let pass = t.mismatches == 0
        && t.tail_violations == 0
        && fails.is_empty()
        && t.controls > 0
        && t.controls_detected == t.controls;
    println!("{}", if pass { "PASS" } else { "FAIL" });
    if pass {
        Ok(())
    } else {
        Err("draft glue oracle failed".into())
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// Median us per sequence over `ITERS` event-timed runs after `WARM` warmups.
fn time_us(
    gpu: &mut Gpu,
    e0: &hip_bridge::Event,
    e1: &hip_bridge::Event,
    mut run: impl FnMut(&mut Gpu) -> Res<()>,
) -> Res<f64> {
    const WARM: usize = 30;
    const ITERS: usize = 200;
    let mut v = Vec::new();
    for it in 0..WARM + ITERS {
        gpu.hip.event_record(e0, gpu.active_stream.as_ref())?;
        run(gpu)?;
        gpu.hip.event_record(e1, gpu.active_stream.as_ref())?;
        gpu.hip.event_synchronize(e1)?;
        if it >= WARM {
            v.push(gpu.hip.event_elapsed_ms(e0, e1)? as f64 * 1000.0);
        }
    }
    Ok(pct(&mut v, 0.5))
}

fn bench() -> Res<()> {
    let mut gpu = Gpu::init()?;
    check_arch(&gpu)?;
    let e0 = gpu.hip.event_create()?;
    let e1 = gpu.hip.event_create()?;
    println!("family lanes | unfolded us | folded us | saved us   (median of 200, event-timed, one round)");
    let mut json: Vec<String> = Vec::new();
    for &lanes in &[1usize, 2, 4, 8] {
        // broadcast add
        let a = upload_words(&mut gpu, &words(lanes * WIDE, Data::Random, 1))?;
        let b = upload_words(&mut gpu, &words(lanes * HIDDEN, Data::Random, 2))?;
        let c = upload_words(&mut gpu, &words(lanes * WIDE, Data::Random, 3))?;
        let unf = time_us(&mut gpu, &e0, &e1, |g| {
            for l in 0..lanes {
                let row = b.sub_offset(l * HIDDEN, HIDDEN);
                for r in 0..BRANCHES {
                    let off = (l * BRANCHES + r) * HIDDEN;
                    g.add_f32(&a.sub_offset(off, HIDDEN), &row, &c.sub_offset(off, HIDDEN))?;
                }
            }
            Ok(())
        })?;
        let fol = time_us(&mut gpu, &e0, &e1, |g| {
            Ok(g.broadcast_add_rows_f32(&a, &b, &c, BRANCHES, HIDDEN, lanes)?)
        })?;
        println!("broadcast {lanes} | {unf:.2} | {fol:.2} | {:.2}", unf - fol);
        json.push(format!(
            "{{\"family\":\"broadcast\",\"lanes\":{lanes},\"unfolded\":{unf:.2},\"folded\":{fol:.2}}}"
        ));

        // rotate
        let x = upload_words(&mut gpu, &words(lanes * HIDDEN, Data::Random, 4))?;
        let xr = upload_words(&mut gpu, &words(lanes * HIDDEN, Data::Random, 5))?;
        let unf = time_us(&mut gpu, &e0, &e1, |g| {
            for row in 0..lanes {
                g.rotate_x_mq(
                    &x.sub_offset(row * HIDDEN, HIDDEN),
                    &xr.sub_offset(row * HIDDEN, HIDDEN),
                    HIDDEN,
                )?;
            }
            Ok(())
        })?;
        let fol = time_us(&mut gpu, &e0, &e1, |g| {
            Ok(g.rotate_x_mq_batched(&x, &xr, HIDDEN, lanes)?)
        })?;
        println!("rotate {lanes} | {unf:.2} | {fol:.2} | {:.2}", unf - fol);
        json.push(format!(
            "{{\"family\":\"rotate\",\"lanes\":{lanes},\"unfolded\":{unf:.2},\"folded\":{fol:.2}}}"
        ));

        // staging copies (one direction; the round does two: in and out)
        if lanes >= 2 {
            let srcs: Vec<GpuTensor> = (0..lanes)
                .map(|l| upload_words(&mut gpu, &words(WIDE, Data::Random, 10 + l as u64)))
                .collect::<Res<_>>()?;
            let rows = upload_words(&mut gpu, &words(lanes * WIDE, Data::Random, 99))?;
            let dsts: Vec<GpuTensor> = (0..lanes)
                .map(|l| rows.sub_offset(l * WIDE, WIDE))
                .collect();
            let unf = time_us(&mut gpu, &e0, &e1, |g| {
                for (s, d) in srcs.iter().zip(&dsts) {
                    g.copy_d2d(s, d, WIDE * 4)?;
                }
                Ok(())
            })?;
            let regions: Vec<CopyRegion<'_>> = srcs
                .iter()
                .zip(&dsts)
                .map(|(s, d)| CopyRegion {
                    dst: &d.buf,
                    dst_offset: 0,
                    src: &s.buf,
                    src_offset: 0,
                    bytes: WIDE * 4,
                })
                .collect();
            let fol = time_us(&mut gpu, &e0, &e1, |g| Ok(copy_regions(g, &regions)?))?;
            println!(
                "stage-copy {lanes} | {unf:.2} | {fol:.2} | {:.2}",
                unf - fol
            );
            json.push(format!(
                "{{\"family\":\"stage_copy\",\"lanes\":{lanes},\"unfolded\":{unf:.2},\"folded\":{fol:.2}}}"
            ));
        }
    }
    // snapshot / restore triple
    let sizes = [4096usize, 1, WIDE];
    let live: Vec<GpuTensor> = sizes
        .iter()
        .map(|&n| upload_words(&mut gpu, &words(n, Data::Random, 77)))
        .collect::<Res<_>>()?;
    let arena: Vec<GpuTensor> = sizes
        .iter()
        .map(|&n| upload_words(&mut gpu, &words(n, Data::Random, 78)))
        .collect::<Res<_>>()?;
    let unf = time_us(&mut gpu, &e0, &e1, |g| {
        for i in 0..3 {
            g.copy_d2d(&live[i], &arena[i], sizes[i] * 4)?;
        }
        Ok(())
    })?;
    let regions: Vec<CopyRegion<'_>> = (0..3)
        .map(|i| CopyRegion {
            dst: &arena[i].buf,
            dst_offset: 0,
            src: &live[i].buf,
            src_offset: 0,
            bytes: sizes[i] * 4,
        })
        .collect();
    let fol = time_us(&mut gpu, &e0, &e1, |g| Ok(copy_regions(g, &regions)?))?;
    println!("snapshot-triple 3 | {unf:.2} | {fol:.2} | {:.2}", unf - fol);
    json.push(format!(
        "{{\"family\":\"snapshot_triple\",\"lanes\":3,\"unfolded\":{unf:.2},\"folded\":{fol:.2}}}"
    ));
    println!(
        "{{\"unit\":\"us\",\"stat\":\"median\",\"iters\":200,\"warmups\":30,\"cells\":[{}]}}",
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
            eprintln!("usage: qwen4_draft_glue oracle | bench");
            std::process::exit(2);
        }
    }
}
