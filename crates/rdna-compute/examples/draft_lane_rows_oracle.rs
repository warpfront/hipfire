// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness oracle of the lane-batched ctx/fc row backend of the DFlash
//! continuous-batching draft (`rdna_compute::draft_lane_rows`, plan section C):
//!
//! 1. `Gpu::gemv_mq4g256v2_multirow_r2_lanes` (batch 1..=8, plus 9/16/33/72 on
//!    smaller shapes) against today's singleton R2 route
//!    (`Gpu::gemv_mq4g256v2` once per segment on `x[z]` / `y[z]` sub-views,
//!    which is `gemv_mq4g256v2_multirow_r2` on exact gfx1201), bit for bit over
//!    the whole output buffer including poisoned guard words. Shapes: the real
//!    draft shapes ((m,k) = (5120,5120), (1024,5120), (5120,25600), i.e. hidden
//!    / kv_dim by hidden / num_extract*hidden; `DflashConfig` lives in the
//!    runtime crate, which this crate's examples cannot see), plus odd M, K
//!    group-count tails (groups % 4 in {1,2,3}), random headers/payloads, two
//!    signed-zero rows, and per-segment x modes (random, all zero, F16-hostile
//!    mantissas, +-0, wide dynamic range). Repeat determinism is checked.
//! 2. The OLD singleton R2 output is unchanged: it is run twice (determinism)
//!    and its output hashes are compared with a stored golden file:
//!    `--golden-write FILE` on a build of the pre-change tree writes it,
//!    `--golden-check FILE` compares against it (case labels and RNG are
//!    deterministic). The old R2 is also the reference of check 1, so a
//!    refactor that moved it would show up as a twin mismatch as well.
//! 3. `draft_rows_gather_f32` / `draft_rows_scatter_f32`: pure bit copies of
//!    random u32 payloads (NaN payloads, signed zeros), positions copied once
//!    per row, unused packed rows / destination words / positions keep their
//!    poison (guard bytes), 1..=32 spans, shuffled packed layouts, overlapping
//!    gather sources, K-only and K+V scatter; plus API validation refusals.
//! 4. Negative control from plan section C: K=256, weight headers scale=0
//!    zero=1 (every decoded weight 1), every F32 x element `1 + 2^-12`. R2 sums
//!    to exactly `256 + 1/16`; the 2-row overwrite WMMA class rounds X to F16
//!    (= 1.0) and returns 256, so the two arithmetic classes MUST DIFFER. A
//!    random-data comparison of the two classes is printed for information.
//!
//! The old R2 route is the PM decode twin image when `kernel.pm_decode` is on
//! (default on exact gfx1201), else the HIP source; the lane twin is always the
//! HIP source. Run with `HIPFIRE_PM_DECODE=0` and default to cover both.
//!
//! Prints PASS/FAIL per case; exits nonzero on any FAIL. Requires exact gfx1201.
//!
//! `cargo run --release -p rdna-compute --features lab --example draft_lane_rows_oracle`

use rdna_compute::draft_lane_rows::{DraftRowSpan, DRAFT_ROW_SPANS_MAX};
use rdna_compute::{Gpu, GpuTensor};
use std::collections::HashMap;

const POISON: u32 = 0x7fc0_dead;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    fn bits(&mut self) -> u32 {
        let b = (self.next() >> 16) as u32;
        if b == POISON {
            b ^ 1
        } else {
            b
        }
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

/// Round-to-nearest-even f16 bits of a finite value in the normal f16 range.
fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    let mut m10 = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m10 & 1 == 1) {
        m10 += 1;
    }
    if m10 == 0x400 {
        m10 = 0;
        exp += 1;
    }
    assert!((1..=30).contains(&exp), "{v} is not a normal f16");
    sign | ((exp as u16) << 10) | m10 as u16
}

fn random_scale(rng: &mut Rng) -> u16 {
    let mag = 2f32.powi(-9) + rng.unit() * 0.2;
    f16_bits(if rng.next() & 1 == 0 { mag } else { -mag })
}

fn random_zero(rng: &mut Rng) -> u16 {
    let mut v = (rng.unit() - 0.5) * 4.0;
    if v.abs() < 2f32.powi(-12) {
        v = 0.125;
    }
    f16_bits(v)
}

/// `[m][k / 256]` groups of 136 bytes: hA (scale s0, zero z0), hB (s1, z1),
/// then 128 payload bytes. Row 0: zero payload, +0.0 headers. Row 1: zero
/// payload, -0.0 headers (0x8000).
fn pack_mq4g256v2(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let groups = k / 256;
    let mut out = Vec::with_capacity(m * groups * 136);
    for row in 0..m {
        for _ in 0..groups {
            if row < 2 {
                let h: u16 = if row == 0 { 0x0000 } else { 0x8000 };
                for _half in 0..2 {
                    out.extend_from_slice(&h.to_le_bytes());
                    out.extend_from_slice(&h.to_le_bytes());
                }
                out.extend(std::iter::repeat_n(0u8, 128));
                continue;
            }
            for _half in 0..2 {
                out.extend_from_slice(&random_scale(rng).to_le_bytes());
                out.extend_from_slice(&random_zero(rng).to_le_bytes());
            }
            for _ in 0..128 {
                out.push((rng.next() & 0xff) as u8);
            }
        }
    }
    out
}

fn bits_to_f32(v: &[u32]) -> Vec<f32> {
    v.iter().map(|&b| f32::from_bits(b)).collect()
}

fn upload_bits(gpu: &mut Gpu, v: &[u32]) -> GpuTensor {
    gpu.upload_f32(&bits_to_f32(v), &[v.len()])
        .expect("upload bits")
}

fn poisoned(gpu: &mut Gpu, n: usize) -> GpuTensor {
    upload_bits(gpu, &vec![POISON; n])
}

fn fetch(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.hip.device_synchronize().expect("sync before download");
    gpu.download_f32(t)
        .expect("download")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn fnv(bits: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bits {
        for byte in b.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn diff(a: &[u32], b: &[u32]) -> (usize, Option<usize>) {
    let mut n = 0;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x != y {
            n += 1;
            first.get_or_insert(i);
        }
    }
    (n, first)
}

struct Tally {
    cases: usize,
    failures: usize,
}

impl Tally {
    fn report(&mut self, label: &str, r: Result<(), String>) {
        self.cases += 1;
        match r {
            Ok(()) => println!("PASS {label}"),
            Err(why) => {
                self.failures += 1;
                println!("FAIL {label}: {why}");
            }
        }
    }
}

// ── 1/2: R2 lanes twin vs old R2 ────────────────────────────────────────────

/// One segment of x. Modes cover random, all-zero, F16-hostile mantissas,
/// signed zeros and wide dynamic range.
fn gen_x(rng: &mut Rng, k: usize, mode: u64) -> Vec<f32> {
    (0..k)
        .map(|i| match mode % 5 {
            0 => {
                if i % 5 == 0 {
                    0.0
                } else {
                    (rng.unit() + rng.unit() - 1.0) * 1.5
                }
            }
            1 => 0.0,
            2 => {
                // (1 + 2^-12) or the exact F16 tie 1 + 2^-11, scaled by 2^e.
                let m = if rng.next() & 1 == 0 {
                    1.0 + 2f32.powi(-12)
                } else {
                    1.0 + 2f32.powi(-11)
                };
                let e = (rng.next() % 13) as i32 - 6;
                let s = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
                s * m * 2f32.powi(e)
            }
            3 => match rng.next() % 4 {
                0 => 0.0,
                1 => -0.0,
                2 => (rng.unit() - 0.5) * 1e-3,
                _ => (rng.unit() - 0.5) * 3.0,
            },
            _ => (rng.unit() - 0.5) * 2f32.powi((rng.next() % 20) as i32 - 10),
        })
        .collect()
}

fn run_r2_case(
    gpu: &mut Gpu,
    rng: &mut Rng,
    w: &GpuTensor,
    m: usize,
    k: usize,
    batch: usize,
    label: &str,
    golden: &mut Vec<(String, u64)>,
) -> Result<(), String> {
    const GUARD: usize = 64;
    let mut x_h: Vec<f32> = Vec::with_capacity(batch * k);
    for z in 0..batch {
        let mode = rng.next() % 5 + z as u64;
        x_h.extend(gen_x(rng, k, mode));
    }
    let x = gpu.upload_f32(&x_h, &[batch * k]).expect("x");
    let y_old = poisoned(gpu, batch * m + GUARD);
    let y_old2 = poisoned(gpu, batch * m + GUARD);
    let y_new = poisoned(gpu, batch * m + GUARD);
    let y_new2 = poisoned(gpu, batch * m + GUARD);

    let mut result: Result<(), String> = Ok(());
    let mut fail = |msg: String| {
        if result.is_ok() {
            result = Err(msg);
        }
    };

    // Old: one singleton launch per segment, twice.
    for y in [&y_old, &y_old2] {
        for z in 0..batch {
            let xs = x.sub_offset(z * k, k);
            let ys = y.sub_offset(z * m, m);
            gpu.gemv_mq4g256v2(w, &xs, &ys, m, k).expect("old r2");
        }
    }
    // New: one lanes launch, twice.
    for y in [&y_new, &y_new2] {
        let ys = y.sub_offset(0, batch * m);
        gpu.gemv_mq4g256v2_multirow_r2_lanes(w, &x, &ys, m, k, batch)
            .expect("r2 lanes");
    }
    let old = fetch(gpu, &y_old);
    let old2 = fetch(gpu, &y_old2);
    let new = fetch(gpu, &y_new);
    let new2 = fetch(gpu, &y_new2);

    if old != old2 {
        fail("old singleton R2 is not repeat-deterministic".into());
    }
    golden.push((label.to_string(), fnv(&old)));
    let (n, first) = diff(&old, &new);
    if n != 0 {
        let i = first.unwrap();
        let (seg, row) = (i / m.max(1), i % m.max(1));
        fail(format!(
            "{n} of {} output bits differ (first at segment {seg} row {row}: old {:#010x} lanes {:#010x})",
            old.len(),
            old[i],
            new[i]
        ));
    }
    if new != new2 {
        fail("lane twin is not repeat-deterministic".into());
    }
    if new[batch * m..].iter().any(|&b| b != POISON) {
        fail("guard words past y were written".into());
    }
    if new[..batch * m].iter().any(|&b| b == POISON) {
        fail("an output word was not written".into());
    }
    for t in [x, y_old, y_old2, y_new, y_new2] {
        gpu.free_tensor(t).expect("free");
    }
    result
}

// ── 3: gather / scatter ─────────────────────────────────────────────────────

fn addr(t: &GpuTensor, elem_off: usize) -> u64 {
    t.buf.as_ptr() as u64 + elem_off as u64 * 4
}

/// Packed layout: spans in `order`, each preceded by `gap` rows, plus 2
/// trailing guard rows. Returns (packed_row0 per span, total packed rows).
fn packed_layout(rows: &[usize], gap: usize, rev: bool) -> (Vec<usize>, usize) {
    let mut p0 = vec![0usize; rows.len()];
    let mut cursor = 0usize;
    let order: Vec<usize> = if rev {
        (0..rows.len()).rev().collect()
    } else {
        (0..rows.len()).collect()
    };
    for i in order {
        cursor += gap;
        p0[i] = cursor;
        cursor += rows[i];
    }
    (p0, cursor + 2)
}

fn run_gather(
    gpu: &mut Gpu,
    rng: &mut Rng,
    width: usize,
    rows: &[usize],
    gap: usize,
    rev: bool,
    with_pos: bool,
    overlap_src: bool,
) -> Result<(), String> {
    let (p0, total) = packed_layout(rows, gap, rev);
    // Source arena: disjoint consecutive spans (1 separator row) or all at 0.
    let mut src_off = vec![0usize; rows.len()];
    let mut cursor = 0usize;
    if !overlap_src {
        for (i, &r) in rows.iter().enumerate() {
            src_off[i] = cursor;
            cursor += r + 1;
        }
    }
    let arena_rows = if overlap_src {
        *rows.iter().max().unwrap() + 1
    } else {
        cursor + 1
    };
    let arena: Vec<u32> = (0..arena_rows * width).map(|_| rng.bits()).collect();
    let pos_src: Vec<u32> = (0..arena_rows).map(|_| rng.bits()).collect();
    let d_arena = upload_bits(gpu, &arena);
    let d_pos_src = upload_bits(gpu, &pos_src);
    let d_packed = poisoned(gpu, total * width);
    let d_pos_out = poisoned(gpu, total);

    let spans: Vec<DraftRowSpan> = rows
        .iter()
        .enumerate()
        .map(|(i, &r)| DraftRowSpan {
            src: addr(&d_arena, src_off[i] * width),
            dst0: 0,
            dst1: 0,
            positions: if with_pos {
                addr(&d_pos_src, src_off[i])
            } else {
                0
            },
            packed_row0: p0[i] as i32,
            rows: r as i32,
        })
        .collect();
    gpu.draft_rows_gather_f32(&d_packed, with_pos.then_some(&d_pos_out), &spans, width)
        .map_err(|e| format!("launch refused: {e}"))?;
    let packed = fetch(gpu, &d_packed);
    let pos_out = fetch(gpu, &d_pos_out);

    let mut expect = vec![POISON; total * width];
    let mut expect_pos = vec![POISON; total];
    for (i, &r) in rows.iter().enumerate() {
        for row in 0..r {
            let s = (src_off[i] + row) * width;
            let d = (p0[i] + row) * width;
            expect[d..d + width].copy_from_slice(&arena[s..s + width]);
            if with_pos {
                expect_pos[p0[i] + row] = pos_src[src_off[i] + row];
            }
        }
    }
    let mut result = Ok(());
    let (n, first) = diff(&expect, &packed);
    if n != 0 {
        result = Err(format!(
            "{n} packed bits differ (first at word {})",
            first.unwrap()
        ));
    }
    let (pn, pfirst) = diff(&expect_pos, &pos_out);
    if result.is_ok() && pn != 0 {
        result = Err(format!(
            "{pn} position words differ (first at row {}: want {:#010x} got {:#010x})",
            pfirst.unwrap(),
            expect_pos[pfirst.unwrap()],
            pos_out[pfirst.unwrap()]
        ));
    }
    for t in [d_arena, d_pos_src, d_packed, d_pos_out] {
        gpu.free_tensor(t).expect("free");
    }
    result
}

fn run_scatter(
    gpu: &mut Gpu,
    rng: &mut Rng,
    width: usize,
    rows: &[usize],
    gap: usize,
    rev: bool,
    with_v: bool,
) -> Result<(), String> {
    let (p0, total) = packed_layout(rows, gap, rev);
    let packed0: Vec<u32> = (0..total * width).map(|_| rng.bits()).collect();
    let packed1: Vec<u32> = (0..total * width).map(|_| rng.bits()).collect();
    let d_p0 = upload_bits(gpu, &packed0);
    let d_p1 = upload_bits(gpu, &packed1);
    // Destination arenas: disjoint spans, `gap` rows before each, 2 guard rows.
    let mut dst_off = vec![0usize; rows.len()];
    let mut cursor = 0usize;
    for (i, &r) in rows.iter().enumerate() {
        cursor += gap;
        dst_off[i] = cursor;
        cursor += r;
    }
    let dst_rows = cursor + 2;
    let d_k = poisoned(gpu, dst_rows * width);
    let d_v = poisoned(gpu, dst_rows * width);
    let spans: Vec<DraftRowSpan> = rows
        .iter()
        .enumerate()
        .map(|(i, &r)| DraftRowSpan {
            src: 0,
            dst0: addr(&d_k, dst_off[i] * width),
            dst1: if with_v {
                addr(&d_v, dst_off[i] * width)
            } else {
                0
            },
            positions: 0,
            packed_row0: p0[i] as i32,
            rows: r as i32,
        })
        .collect();
    gpu.draft_rows_scatter_f32(&d_p0, with_v.then_some(&d_p1), &spans, width)
        .map_err(|e| format!("launch refused: {e}"))?;
    let k_out = fetch(gpu, &d_k);
    let v_out = fetch(gpu, &d_v);

    let mut expect_k = vec![POISON; dst_rows * width];
    let mut expect_v = vec![POISON; dst_rows * width];
    for (i, &r) in rows.iter().enumerate() {
        for row in 0..r {
            let s = (p0[i] + row) * width;
            let d = (dst_off[i] + row) * width;
            expect_k[d..d + width].copy_from_slice(&packed0[s..s + width]);
            if with_v {
                expect_v[d..d + width].copy_from_slice(&packed1[s..s + width]);
            }
        }
    }
    let mut result = Ok(());
    let (n, first) = diff(&expect_k, &k_out);
    if n != 0 {
        result = Err(format!(
            "{n} K destination bits differ (first at word {})",
            first.unwrap()
        ));
    }
    let (vn, vfirst) = diff(&expect_v, &v_out);
    if result.is_ok() && vn != 0 {
        result = Err(format!(
            "{vn} V destination bits differ (first at word {})",
            vfirst.unwrap()
        ));
    }
    for t in [d_p0, d_p1, d_k, d_v] {
        gpu.free_tensor(t).expect("free");
    }
    result
}

fn expect_refused(t: &mut Tally, name: &str, r: Result<(), hip_bridge::HipError>) {
    t.cases += 1;
    match r {
        Err(e) => println!("PASS validation {name}: refused ({e})"),
        Ok(()) => {
            t.failures += 1;
            println!("FAIL validation {name}: accepted an invalid call");
        }
    }
}

fn validation(gpu: &mut Gpu, t: &mut Tally) {
    let width = 16usize;
    let arena = poisoned(gpu, 64 * width);
    let pos = poisoned(gpu, 64);
    let packed = poisoned(gpu, 16 * width);
    let pos_out = poisoned(gpu, 16);
    let dk = poisoned(gpu, 32 * width);
    let dv = poisoned(gpu, 32 * width);
    let ok_g = [
        DraftRowSpan {
            src: addr(&arena, 0),
            dst0: 0,
            dst1: 0,
            positions: addr(&pos, 0),
            packed_row0: 0,
            rows: 3,
        },
        DraftRowSpan {
            src: addr(&arena, 10 * width),
            dst0: 0,
            dst1: 0,
            positions: addr(&pos, 10),
            packed_row0: 3,
            rows: 2,
        },
    ];
    t.cases += 1;
    match gpu.draft_rows_gather_f32(&packed, Some(&pos_out), &ok_g, width) {
        Ok(()) => println!("PASS validation gather baseline: valid call accepted"),
        Err(e) => {
            t.failures += 1;
            println!("FAIL validation gather baseline: valid call refused ({e})");
        }
    }
    let mut bad = ok_g;
    bad[1].packed_row0 = 2;
    expect_refused(
        t,
        "gather overlapping packed ranges",
        gpu.draft_rows_gather_f32(&packed, Some(&pos_out), &bad, width),
    );
    let mut bad = ok_g;
    bad[0].rows = 0;
    expect_refused(
        t,
        "gather zero-row span",
        gpu.draft_rows_gather_f32(&packed, Some(&pos_out), &bad, width),
    );
    let mut bad = ok_g;
    bad[0].src = 0;
    expect_refused(
        t,
        "gather null src",
        gpu.draft_rows_gather_f32(&packed, Some(&pos_out), &bad, width),
    );
    let mut bad = ok_g;
    bad[1].positions = 0;
    expect_refused(
        t,
        "gather null positions with positions output",
        gpu.draft_rows_gather_f32(&packed, Some(&pos_out), &bad, width),
    );
    expect_refused(
        t,
        "gather no spans",
        gpu.draft_rows_gather_f32(&packed, None, &[], width),
    );
    let many: Vec<DraftRowSpan> = (0..DRAFT_ROW_SPANS_MAX + 1)
        .map(|i| DraftRowSpan {
            src: addr(&arena, 0),
            dst0: 0,
            dst1: 0,
            positions: 0,
            packed_row0: i as i32,
            rows: 1,
        })
        .collect();
    let many_packed = poisoned(gpu, 40 * width);
    expect_refused(
        t,
        "gather too many spans",
        gpu.draft_rows_gather_f32(&many_packed, None, &many, width),
    );
    gpu.free_tensor(many_packed).expect("free");
    let mut bad = ok_g;
    bad[1].packed_row0 = 15;
    bad[1].rows = 4;
    expect_refused(
        t,
        "gather packed buffer too small",
        gpu.draft_rows_gather_f32(&packed, Some(&pos_out), &bad, width),
    );
    expect_refused(
        t,
        "gather zero width",
        gpu.draft_rows_gather_f32(&packed, None, &ok_g, 0),
    );
    let mut bad = ok_g;
    bad[0].src = addr(&packed, 0);
    expect_refused(
        t,
        "gather src overlaps packed",
        gpu.draft_rows_gather_f32(&packed, None, &bad, width),
    );

    let ok_s = [
        DraftRowSpan {
            src: 0,
            dst0: addr(&dk, 0),
            dst1: addr(&dv, 0),
            positions: 0,
            packed_row0: 0,
            rows: 3,
        },
        DraftRowSpan {
            src: 0,
            dst0: addr(&dk, 8 * width),
            dst1: addr(&dv, 8 * width),
            positions: 0,
            packed_row0: 3,
            rows: 2,
        },
    ];
    t.cases += 1;
    match gpu.draft_rows_scatter_f32(&packed, Some(&packed), &ok_s, width) {
        Ok(()) => println!("PASS validation scatter baseline: valid call accepted"),
        Err(e) => {
            t.failures += 1;
            println!("FAIL validation scatter baseline: valid call refused ({e})");
        }
    }
    let mut bad = ok_s;
    bad[1].dst0 = addr(&dk, 2 * width);
    expect_refused(
        t,
        "scatter overlapping destinations",
        gpu.draft_rows_scatter_f32(&packed, Some(&packed), &bad, width),
    );
    let mut bad = ok_s;
    bad[0].dst1 = bad[1].dst0;
    expect_refused(
        t,
        "scatter dst1 overlaps another dst0",
        gpu.draft_rows_scatter_f32(&packed, Some(&packed), &bad, width),
    );
    let mut bad = ok_s;
    bad[1].dst1 = 0;
    expect_refused(
        t,
        "scatter null dst1 with packed1",
        gpu.draft_rows_scatter_f32(&packed, Some(&packed), &bad, width),
    );
    let mut bad = ok_s;
    bad[0].dst0 = 0;
    expect_refused(
        t,
        "scatter null dst0",
        gpu.draft_rows_scatter_f32(&packed, None, &bad, width),
    );
    let mut bad = ok_s;
    bad[0].dst0 = addr(&packed, 0);
    expect_refused(
        t,
        "scatter dst overlaps packed",
        gpu.draft_rows_scatter_f32(&packed, None, &bad, width),
    );
    let mut bad = ok_s;
    bad[1].packed_row0 = 15;
    expect_refused(
        t,
        "scatter packed buffer too small",
        gpu.draft_rows_scatter_f32(&packed, None, &bad, width),
    );

    // R2 lanes API.
    let (m, k) = (4usize, 256usize);
    let w = gpu.upload_raw(&vec![0u8; m * 136], &[m * 136]).expect("w");
    let x = poisoned(gpu, 4 * k);
    let y = poisoned(gpu, 4 * m);
    t.cases += 1;
    match gpu.gemv_mq4g256v2_multirow_r2_lanes(&w, &x, &y, m, k, 4) {
        Ok(()) => println!("PASS validation r2 baseline: valid call accepted"),
        Err(e) => {
            t.failures += 1;
            println!("FAIL validation r2 baseline: valid call refused ({e})");
        }
    }
    expect_refused(
        t,
        "r2 batch 0",
        gpu.gemv_mq4g256v2_multirow_r2_lanes(&w, &x, &y, m, k, 0),
    );
    expect_refused(
        t,
        "r2 batch too large",
        gpu.gemv_mq4g256v2_multirow_r2_lanes(&w, &x, &y, m, k, 73),
    );
    expect_refused(
        t,
        "r2 k not multiple of 256",
        gpu.gemv_mq4g256v2_multirow_r2_lanes(&w, &x, &y, m, 255, 1),
    );
    expect_refused(
        t,
        "r2 rotated too small",
        gpu.gemv_mq4g256v2_multirow_r2_lanes(&w, &x, &y, m, k, 5),
    );
    expect_refused(
        t,
        "r2 y too small",
        gpu.gemv_mq4g256v2_multirow_r2_lanes(&w, &x, &y.sub_offset(0, 3 * m), m, k, 4),
    );
    expect_refused(
        t,
        "r2 weight too small",
        gpu.gemv_mq4g256v2_multirow_r2_lanes(&w, &x, &y, m + 1, k, 1),
    );
    expect_refused(
        t,
        "r2 y aliases rotated",
        gpu.gemv_mq4g256v2_multirow_r2_lanes(&w, &x, &x.sub_offset(0, 4 * m), m, k, 4),
    );
    for tn in [arena, pos, packed, pos_out, dk, dv, w, x, y] {
        gpu.free_tensor(tn).expect("free");
    }
}

// ── 4: R2 vs overwrite-WMMA negative control ────────────────────────────────

fn negative_control(gpu: &mut Gpu, rng: &mut Rng) -> Result<(), String> {
    // (a) Analytical counterexample: K=256, scale=0, zero=1, x = 1 + 2^-12.
    let (m, k) = (32usize, 256usize);
    let mut bytes = Vec::with_capacity(m * 136);
    for _ in 0..m {
        for _half in 0..2 {
            bytes.extend_from_slice(&0x0000u16.to_le_bytes()); // scale 0
            bytes.extend_from_slice(&0x3C00u16.to_le_bytes()); // zero 1.0
        }
        bytes.extend(std::iter::repeat_n(0u8, 128));
    }
    let w = gpu.upload_raw(&bytes, &[bytes.len()]).expect("w");
    let xv = 1.0 + 2f32.powi(-12);
    let x = gpu.upload_f32(&vec![xv; k], &[k]).expect("x1");
    let x2 = gpu.upload_f32(&vec![xv; 2 * k], &[2 * k]).expect("x2");
    let y_r2 = poisoned(gpu, m);
    let y_lanes = poisoned(gpu, m);
    let y_wmma = poisoned(gpu, 2 * m);
    gpu.gemv_mq4g256v2(&w, &x, &y_r2, m, k).expect("old r2");
    gpu.gemv_mq4g256v2_multirow_r2_lanes(&w, &x, &y_lanes, m, k, 1)
        .expect("lanes");
    gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
    let admitted = gpu
        .gemm_mq4g256v2_overwrite_gfx12_dflash(&w, &x2, &y_wmma, m, k, 2)
        .expect("overwrite launch");
    let r2 = fetch(gpu, &y_r2);
    let lanes = fetch(gpu, &y_lanes);
    let wmma = fetch(gpu, &y_wmma);
    println!(
        "  control (a): R2 row0 = {} ({:#010x}), lanes row0 = {} ({:#010x}), WMMA row0 = {} ({:#010x}), WMMA admitted = {admitted}",
        f32::from_bits(r2[0]),
        r2[0],
        f32::from_bits(lanes[0]),
        lanes[0],
        f32::from_bits(wmma[0]),
        wmma[0]
    );
    for t in [w, x, x2, y_r2, y_lanes, y_wmma] {
        gpu.free_tensor(t).expect("free");
    }
    if !admitted {
        return Err("the overwrite WMMA class was not admitted (control inconclusive)".into());
    }
    if r2[0] != (256.0f32 + 0.0625).to_bits() {
        return Err(format!(
            "R2 row0 is {:#010x}, expected 256.0625 ({:#010x})",
            r2[0],
            (256.0f32 + 0.0625).to_bits()
        ));
    }
    if r2 != lanes {
        return Err("R2 lanes twin differs from old R2 on the control".into());
    }
    if r2.iter().zip(&wmma[..m]).all(|(a, b)| a == b) {
        return Err("R2 and the overwrite WMMA class are bit-identical on the F16-rounding counterexample (control cannot see class mixing)".into());
    }

    // (b) Random realistic data: informational mismatch count between classes.
    let (m, k) = (1024usize, 5120usize);
    let wb = pack_mq4g256v2(m, k, rng);
    let w = gpu.upload_raw(&wb, &[wb.len()]).expect("w");
    let xh: Vec<f32> = (0..k)
        .map(|_| (rng.unit() + rng.unit() - 1.0) * 1.5)
        .collect();
    let mut x2h = xh.clone();
    x2h.extend_from_slice(&xh);
    let x = gpu.upload_f32(&xh, &[k]).expect("x");
    let x2 = gpu.upload_f32(&x2h, &[2 * k]).expect("x2");
    let y_r2 = poisoned(gpu, m);
    let y_wmma = poisoned(gpu, 2 * m);
    gpu.gemv_mq4g256v2(&w, &x, &y_r2, m, k).expect("old r2");
    gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
    let admitted = gpu
        .gemm_mq4g256v2_overwrite_gfx12_dflash(&w, &x2, &y_wmma, m, k, 2)
        .expect("overwrite launch");
    let r2 = fetch(gpu, &y_r2);
    let wmma = fetch(gpu, &y_wmma);
    let (n, _) = diff(&r2, &wmma[..m]);
    println!("  control (b): random (m={m}, k={k}) R2 vs WMMA row0: {n} of {m} outputs differ (admitted={admitted}; informational)");
    for t in [w, x, x2, y_r2, y_wmma] {
        gpu.free_tensor(t).expect("free");
    }
    Ok(())
}

// ── main ────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let golden_write = flag("--golden-write");
    let golden_check = flag("--golden-check");

    let mut gpu = Gpu::init().expect("Gpu::init");
    if gpu.arch != "gfx1201" {
        eprintln!(
            "draft_lane_rows_oracle: needs exact gfx1201, got {}",
            gpu.arch
        );
        std::process::exit(2);
    }
    println!(
        "arch={} pm_decode={} draft_lane_rows_admitted={}",
        gpu.arch,
        gpu.flags.pm_decode,
        gpu.draft_lane_rows_admitted()
    );

    let mut rng = Rng(0xD0F1_C0A7_0B5E_1234);
    let mut t = Tally {
        cases: 0,
        failures: 0,
    };
    let mut golden: Vec<(String, u64)> = Vec::new();

    // 1/2: R2 twin vs old R2.
    let shapes: [(usize, usize); 13] = [
        (5120, 5120),
        (1024, 5120),
        (5120, 25600),
        (1, 256),
        (2, 256),
        (3, 768),
        (7, 1280),
        (33, 1792),
        (65, 1536),
        (1025, 5120),
        (4099, 2304),
        (5121, 5376),
        (17, 512),
    ];
    for &(m, k) in &shapes {
        let wb = pack_mq4g256v2(m, k, &mut rng);
        let w = gpu.upload_raw(&wb, &[wb.len()]).expect("weights");
        let mut batches: Vec<usize> = (1..=8).collect();
        if m * k <= 5120 * 5120 {
            batches.extend([9, 16, 33, 72]);
        }
        for &batch in &batches {
            let label = format!("r2_lanes m={m} k={k} batch={batch}");
            let mut g = Vec::new();
            let r = run_r2_case(&mut gpu, &mut rng, &w, m, k, batch, &label, &mut g);
            golden.append(&mut g);
            t.report(&label, r);
        }
        gpu.free_tensor(w).expect("free weights");
    }

    // Golden: the OLD singleton R2 output is unchanged.
    if let Some(path) = &golden_write {
        let body: String = golden
            .iter()
            .map(|(l, h)| format!("{l}\t{h:016x}\n"))
            .collect();
        std::fs::write(path, body).expect("write golden");
        println!("golden written: {} entries -> {path}", golden.len());
    }
    if let Some(path) = &golden_check {
        let text = std::fs::read_to_string(path).expect("read golden");
        let want: HashMap<&str, &str> = text.lines().filter_map(|l| l.split_once('\t')).collect();
        let mut bad = 0usize;
        for (label, h) in &golden {
            match want.get(label.as_str()) {
                Some(w) if *w == format!("{h:016x}") => {}
                Some(w) => {
                    bad += 1;
                    println!("  old R2 hash differs for {label}: golden {w} now {h:016x}");
                }
                None => {
                    bad += 1;
                    println!("  old R2 golden has no entry for {label}");
                }
            }
        }
        t.report(
            &format!("old R2 unchanged vs golden {path} ({} cases)", golden.len()),
            if bad == 0 {
                Ok(())
            } else {
                Err(format!("{bad} case(s) changed"))
            },
        );
    } else if golden_write.is_none() {
        println!("note: no --golden-check FILE: old R2 'unchanged' is covered only by the old-vs-twin comparison above");
    }

    // 3: gather / scatter.
    let widths = [1usize, 5, 64, 1024, 5120];
    let row_lists: Vec<Vec<usize>> = vec![
        vec![1],
        vec![9],
        vec![3, 1, 9],
        vec![1; 8],
        vec![8; 8],
        vec![2, 1, 7, 1, 1, 9, 4, 1],
        vec![1; DRAFT_ROW_SPANS_MAX],
        (0..DRAFT_ROW_SPANS_MAX).map(|i| 1 + i % 3).collect(),
    ];
    for &width in &widths {
        for rows in &row_lists {
            if width == 5120 && rows.iter().sum::<usize>() > 40 {
                continue;
            }
            for &(gap, rev) in &[(0usize, false), (1, true), (2, false)] {
                for &with_pos in &[false, true] {
                    let label = format!("gather width={width} rows={rows:?} gap={gap} rev={rev} positions={with_pos}");
                    let r = run_gather(&mut gpu, &mut rng, width, rows, gap, rev, with_pos, false);
                    t.report(&label, r);
                }
                for &with_v in &[false, true] {
                    let label = format!(
                        "scatter width={width} rows={rows:?} gap={gap} rev={rev} v={with_v}"
                    );
                    let r = run_scatter(&mut gpu, &mut rng, width, rows, gap, rev, with_v);
                    t.report(&label, r);
                }
            }
        }
        for rows in [vec![3usize, 9, 1], vec![8; 8]] {
            let label =
                format!("gather width={width} rows={rows:?} overlapping sources positions=true");
            let r = run_gather(&mut gpu, &mut rng, width, &rows, 1, false, true, true);
            t.report(&label, r);
        }
    }

    validation(&mut gpu, &mut t);

    // 4: negative control.
    t.report(
        "negative control: R2 class != overwrite WMMA class (F16-rounding counterexample)",
        negative_control(&mut gpu, &mut rng),
    );

    println!("{} cases, {} FAIL", t.cases, t.failures);
    if t.failures != 0 {
        std::process::exit(1);
    }
}
