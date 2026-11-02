// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness oracle of the lane-batched DFlash2 dynamic causal
//! convolution of the continuous-batching draft
//! (`Gpu::dynamic_causal_conv_f32_lanes`, kernel
//! `dynamic_causal_conv_f32_lanes`) against today's per-lane launch
//! sequence (`Gpu::dynamic_causal_conv_f32` once per lane on that lane's row
//! sub-views).
//!
//! Every case lays all lanes' rows in ONE shared row scratch (lane `i` owns
//! rows `[row0_i, row0_i + rows_i)`, with optional unowned gap rows between
//! lanes and a trailing guard) filled with random f32 including signed zeros
//! and exactly cancelling `base + dynamic` weights. Arm A (reference) runs the
//! existing kernel per lane on sub-views into a POISONED output; arm B runs
//! one lane launch into a second POISONED output, and arm B is run once more
//! to check repeat determinism. Every f32 bit of the whole output buffers
//! (owned rows, gap rows and guards) must be identical.
//!
//! Matrix: hidden/group_size {(64,16), (96,16), (64,1), (128,8)} plus the
//! draft's (5120,16); kernel_size {1,2,3,5} (the generic loop; 2 is the
//! unrolled path); lane row lists (unequal rows in {1,3,4,8,9,16}, up to 16
//! lanes); gap rows {0,2}; phases: base subview at 0 / dynamic offset 0 and
//! base subview at k*hidden / dynamic offset k*groups (2-phase stride), plus
//! the compact stride.
//!
//! Negative controls: with contiguous lanes, the existing kernel run ONCE over
//! all rows (a globally causal convolution that lets a lane read its
//! neighbour's last row) MUST differ from the per-lane reference — in the
//! impulse variant (zero input except each lane's last row = 1.0) as well as
//! on random data — proving the comparison can see a lane-boundary leak
//! that a lane-local guard prevents. API validation cases must be refused.
//!
//! Prints PASS/FAIL per case; exits nonzero on any FAIL.
//!
//! `cargo run --release -p rdna-compute --features lab --example draft_lane_conv_oracle`

use rdna_compute::draft_lane_conv::{DraftConvLane, DRAFT_CONV_LANES_MAX};
use rdna_compute::{Gpu, GpuTensor};

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
    /// Uniform in [-1, 1).
    fn sym(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

/// Random f32s; about 1 in 16 entries is an exact +0.0 or -0.0.
fn gen(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            if rng.next() % 16 == 0 {
                if rng.next() & 1 == 0 {
                    0.0
                } else {
                    -0.0
                }
            } else {
                rng.sym() * amp
            }
        })
        .collect()
}

const POISON: u32 = 0x7fc0_dead;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    /// Base subview at 0, dynamic offset 0, stride 2*k*groups (+ pad).
    Zero,
    /// Base subview at k*hidden, dynamic offset k*groups, stride 2*k*groups (+ pad).
    One,
    /// Compact dynamic: stride k*groups, offset 0.
    Compact,
}

struct Case<'a> {
    hidden: usize,
    gs: usize,
    k: usize,
    rows: &'a [usize],
    gap: usize,
    phase: Phase,
    /// Impulse input: zero except each lane's last row = 1.0.
    impulse: bool,
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn poisoned(gpu: &mut Gpu, n: usize) -> GpuTensor {
    gpu.upload_f32(&vec![f32::from_bits(POISON); n], &[n])
        .expect("poison upload")
}

/// First differing index and the mismatch count between two bit vectors.
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

/// Returns `Ok(())` on PASS, `Err(reason)` on FAIL.
fn run_case(gpu: &mut Gpu, rng: &mut Rng, c: &Case) -> Result<(), String> {
    let (hidden, gs, k) = (c.hidden, c.gs, c.k);
    let groups = hidden / gs;
    let window = k * groups;
    let (stride, offset) = match c.phase {
        Phase::Zero => (2 * window + 3, 0),
        Phase::One => (2 * window + 3, window),
        Phase::Compact => (window, 0),
    };
    let base_off = if c.phase == Phase::One { k * hidden } else { 0 };

    // Lane geometry: lane i owns [row0_i, row0_i + rows_i); `gap` unowned rows
    // precede each lane and 2 guard rows trail the last.
    let mut lanes = Vec::new();
    let mut cursor = 0usize;
    for &r in c.rows {
        cursor += c.gap;
        lanes.push(DraftConvLane {
            row0: cursor,
            rows: r,
        });
        cursor += r;
    }
    let end = cursor;
    let total_rows = end + 2;

    let base_h = gen(rng, 2 * k * hidden, 1.0);
    let mut dyn_h = gen(rng, total_rows * stride, 1.0);
    // Exactly cancelling weights (b + d == 0) at the group's first channel.
    if c.phase != Phase::Compact {
        for r in 0..total_rows {
            for p in 0..2 {
                for off in 0..k {
                    for g in 0..groups {
                        if rng.next() % 16 == 0 {
                            dyn_h[r * stride + p * window + off * groups + g] =
                                -base_h[(p * k + off) * hidden + g * gs];
                        }
                    }
                }
            }
        }
    } else {
        for r in 0..total_rows {
            for off in 0..k {
                for g in 0..groups {
                    if rng.next() % 16 == 0 {
                        dyn_h[r * stride + off * groups + g] = -base_h[off * hidden + g * gs];
                    }
                }
            }
        }
    }
    let mut in_h = gen(rng, total_rows * hidden, 1.5);
    if c.impulse {
        in_h.iter_mut().for_each(|v| *v = 0.0);
        for ln in &lanes {
            let last = ln.row0 + ln.rows - 1;
            in_h[last * hidden..(last + 1) * hidden]
                .iter_mut()
                .for_each(|v| *v = 1.0);
        }
    }

    let input = gpu
        .upload_f32(&in_h, &[total_rows * hidden])
        .expect("input");
    let base = gpu.upload_f32(&base_h, &[2 * k * hidden]).expect("base");
    let dynamic = gpu
        .upload_f32(&dyn_h, &[total_rows * stride])
        .expect("dynamic");
    let out_ref = poisoned(gpu, total_rows * hidden);
    let out_lanes = poisoned(gpu, total_rows * hidden);
    let out_again = poisoned(gpu, total_rows * hidden);
    let base_view = base.sub_offset(base_off, k * hidden);

    let mut result: Result<(), String> = Ok(());
    let mut fail = |msg: String| {
        if result.is_ok() {
            result = Err(msg);
        }
    };

    // Arm A: the existing launch per lane on that lane's sub-views.
    for ln in &lanes {
        let inp = input.sub_offset(ln.row0 * hidden, ln.rows * hidden);
        let out = out_ref.sub_offset(ln.row0 * hidden, ln.rows * hidden);
        let dy = dynamic.sub_offset(ln.row0 * stride, ln.rows * stride);
        gpu.dynamic_causal_conv_f32(
            &inp, &base_view, &dy, &out, ln.rows, hidden, k, gs, stride, offset,
        )
        .expect("per-lane conv");
    }
    // Arm B: one lane launch (twice, into separate poisoned outputs).
    gpu.dynamic_causal_conv_f32_lanes(
        &input, &base_view, &dynamic, &out_lanes, &lanes, hidden, k, gs, stride, offset,
    )
    .expect("lane conv");
    gpu.dynamic_causal_conv_f32_lanes(
        &input, &base_view, &dynamic, &out_again, &lanes, hidden, k, gs, stride, offset,
    )
    .expect("lane conv repeat");

    let ref_bits = bits(&gpu.download_f32(&out_ref).expect("dl ref"));
    let lane_bits = bits(&gpu.download_f32(&out_lanes).expect("dl lanes"));
    let again_bits = bits(&gpu.download_f32(&out_again).expect("dl again"));

    let (n, first) = diff(&ref_bits, &lane_bits);
    if n != 0 {
        let i = first.unwrap();
        fail(format!(
            "{n} of {} output bits differ (first at row {} col {}: ref {:#010x} lanes {:#010x})",
            ref_bits.len(),
            i / hidden,
            i % hidden,
            ref_bits[i],
            lane_bits[i]
        ));
    }
    if lane_bits != again_bits {
        fail("repeated lane launch is not deterministic".to_string());
    }
    // Poison survives in every row no lane owns (explicit guard check, so a
    // reference that also corrupted a guard cannot hide it).
    let mut owned = vec![false; total_rows];
    for ln in &lanes {
        owned[ln.row0..ln.row0 + ln.rows]
            .iter_mut()
            .for_each(|o| *o = true);
    }
    for (r, &o) in owned.iter().enumerate() {
        if !o
            && lane_bits[r * hidden..(r + 1) * hidden]
                .iter()
                .any(|&b| b != POISON)
        {
            fail(format!("unowned guard row {r} was written"));
            break;
        }
    }
    // Owned rows must have been written (a sentinel NaN payload never results
    // from a finite computation).
    for ln in &lanes {
        if lane_bits[ln.row0 * hidden..(ln.row0 + ln.rows) * hidden]
            .iter()
            .any(|&b| b == POISON)
        {
            fail(format!(
                "owned rows of lane at row0 {} were not written",
                ln.row0
            ));
            break;
        }
    }

    // Negative control: a single globally causal launch over contiguous lanes
    // starting at row 0 must differ from the per-lane reference (the lane
    // boundary leak the lane-local guard prevents).
    if c.gap == 0 && k >= 2 && lanes.len() >= 2 {
        let out_g = poisoned(gpu, total_rows * hidden);
        gpu.dynamic_causal_conv_f32(
            &input, &base_view, &dynamic, &out_g, end, hidden, k, gs, stride, offset,
        )
        .expect("global conv");
        let g_bits = bits(&gpu.download_f32(&out_g).expect("dl global"));
        let (gn, _) = diff(&ref_bits[..end * hidden], &g_bits[..end * hidden]);
        if gn == 0 {
            fail("negative control: globally causal conv matches the per-lane reference (oracle cannot see a boundary leak)".to_string());
        }
        gpu.free_tensor(out_g).expect("free out_g");
    }

    for t in [input, base, dynamic, out_ref, out_lanes, out_again] {
        gpu.free_tensor(t).expect("free");
    }
    result
}

/// API validation: the call must be refused.
fn expect_refused(
    name: &str,
    r: Result<(), hip_bridge::HipError>,
    failures: &mut usize,
    cases: &mut usize,
) {
    *cases += 1;
    match r {
        Err(e) => println!("PASS validation {name}: refused ({e})"),
        Ok(()) => {
            *failures += 1;
            println!("FAIL validation {name}: accepted an invalid call");
        }
    }
}

fn validation(gpu: &mut Gpu, failures: &mut usize, cases: &mut usize) {
    let (hidden, gs, k) = (32usize, 16usize, 2usize);
    let groups = hidden / gs;
    let stride = k * groups;
    let rows = 16usize;
    let input = poisoned(gpu, rows * hidden);
    let output = poisoned(gpu, rows * hidden);
    let base = poisoned(gpu, k * hidden);
    let dynamic = poisoned(gpu, rows * stride);
    let ok = [
        DraftConvLane { row0: 0, rows: 4 },
        DraftConvLane { row0: 4, rows: 4 },
    ];

    // Sanity: the same operands are accepted.
    *cases += 1;
    match gpu.dynamic_causal_conv_f32_lanes(
        &input, &base, &dynamic, &output, &ok, hidden, k, gs, stride, 0,
    ) {
        Ok(()) => println!("PASS validation baseline: valid call accepted"),
        Err(e) => {
            *failures += 1;
            println!("FAIL validation baseline: valid call refused ({e})");
        }
    }
    let overlap = [
        DraftConvLane { row0: 0, rows: 4 },
        DraftConvLane { row0: 3, rows: 4 },
    ];
    expect_refused(
        "overlapping lanes",
        gpu.dynamic_causal_conv_f32_lanes(
            &input, &base, &dynamic, &output, &overlap, hidden, k, gs, stride, 0,
        ),
        failures,
        cases,
    );
    let zero = [DraftConvLane { row0: 0, rows: 0 }];
    expect_refused(
        "zero-row lane",
        gpu.dynamic_causal_conv_f32_lanes(
            &input, &base, &dynamic, &output, &zero, hidden, k, gs, stride, 0,
        ),
        failures,
        cases,
    );
    expect_refused(
        "no lanes",
        gpu.dynamic_causal_conv_f32_lanes(
            &input,
            &base,
            &dynamic,
            &output,
            &[],
            hidden,
            k,
            gs,
            stride,
            0,
        ),
        failures,
        cases,
    );
    let many: Vec<DraftConvLane> = (0..DRAFT_CONV_LANES_MAX + 1)
        .map(|i| DraftConvLane { row0: i, rows: 1 })
        .collect();
    expect_refused(
        "too many lanes",
        gpu.dynamic_causal_conv_f32_lanes(
            &input, &base, &dynamic, &output, &many, hidden, k, gs, stride, 0,
        ),
        failures,
        cases,
    );
    let beyond = [DraftConvLane { row0: 12, rows: 8 }];
    expect_refused(
        "lane beyond buffers",
        gpu.dynamic_causal_conv_f32_lanes(
            &input, &base, &dynamic, &output, &beyond, hidden, k, gs, stride, 0,
        ),
        failures,
        cases,
    );
    expect_refused(
        "output aliases input",
        gpu.dynamic_causal_conv_f32_lanes(
            &input, &base, &dynamic, &input, &ok, hidden, k, gs, stride, 0,
        ),
        failures,
        cases,
    );
    expect_refused(
        "window overruns stride",
        gpu.dynamic_causal_conv_f32_lanes(
            &input, &base, &dynamic, &output, &ok, hidden, k, gs, stride, 1,
        ),
        failures,
        cases,
    );
    expect_refused(
        "hidden not divisible by group_size",
        gpu.dynamic_causal_conv_f32_lanes(
            &input, &base, &dynamic, &output, &ok, hidden, k, 5, stride, 0,
        ),
        failures,
        cases,
    );
    expect_refused(
        "short base",
        gpu.dynamic_causal_conv_f32_lanes(
            &input,
            &base.sub_offset(0, hidden),
            &dynamic,
            &output,
            &ok,
            hidden,
            k,
            gs,
            stride,
            0,
        ),
        failures,
        cases,
    );
    for t in [input, output, base, dynamic] {
        gpu.free_tensor(t).expect("free");
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    println!(
        "arch={} draft_lane_conv_admitted={}",
        gpu.arch,
        gpu.draft_lane_conv_admitted()
    );
    if gpu.arch != "gfx1201" {
        eprintln!("draft_lane_conv_oracle: note: product admission is gfx1201-only; running the kernel on {}", gpu.arch);
    }

    let mut rng = Rng(0x0D1F_C0A7);
    let row_lists: Vec<Vec<usize>> = vec![
        vec![1],
        vec![3],
        vec![4, 8],
        vec![9],
        vec![16],
        vec![1, 3, 4, 8, 9, 16],
        vec![8; 8],
        vec![16, 9, 1, 3, 8, 4, 1, 16, 9, 3, 4, 8, 1, 16, 9, 3],
    ];
    let shapes: [(usize, usize); 4] = [(64, 16), (96, 16), (64, 1), (128, 8)];

    let mut failures = 0usize;
    let mut cases = 0usize;
    let run = |gpu: &mut Gpu, rng: &mut Rng, c: Case, failures: &mut usize, cases: &mut usize| {
        *cases += 1;
        let label = format!(
            "hidden={} gs={} k={} rows={:?} gap={} phase={:?}{}",
            c.hidden,
            c.gs,
            c.k,
            c.rows,
            c.gap,
            c.phase,
            if c.impulse { " impulse" } else { "" }
        );
        match run_case(gpu, rng, &c) {
            Ok(()) => println!("PASS {label}"),
            Err(why) => {
                *failures += 1;
                println!("FAIL {label}: {why}");
            }
        }
    };

    for &(hidden, gs) in &shapes {
        for &k in &[1usize, 2, 3, 5] {
            for rows in &row_lists {
                for &gap in &[0usize, 2] {
                    for &phase in &[Phase::Zero, Phase::One, Phase::Compact] {
                        run(
                            &mut gpu,
                            &mut rng,
                            Case {
                                hidden,
                                gs,
                                k,
                                rows,
                                gap,
                                phase,
                                impulse: false,
                            },
                            &mut failures,
                            &mut cases,
                        );
                    }
                }
            }
        }
    }
    // Impulse controls (contiguous lanes): k >= 2 only, both phases.
    for &(hidden, gs) in &shapes {
        for &k in &[2usize, 3, 5] {
            for rows in [vec![4usize, 8], vec![1, 3, 4, 8, 9, 16], vec![8; 8]] {
                for &phase in &[Phase::Zero, Phase::One] {
                    run(
                        &mut gpu,
                        &mut rng,
                        Case {
                            hidden,
                            gs,
                            k,
                            rows: &rows,
                            gap: 0,
                            phase,
                            impulse: true,
                        },
                        &mut failures,
                        &mut cases,
                    );
                }
            }
        }
    }
    // The draft's real dimensions (h=5120, group 16, k=2), C8 shape and unequal lanes.
    for rows in [vec![8usize; 8], vec![1, 3, 4, 8, 9, 16]] {
        for &phase in &[Phase::Zero, Phase::One] {
            run(
                &mut gpu,
                &mut rng,
                Case {
                    hidden: 5120,
                    gs: 16,
                    k: 2,
                    rows: &rows,
                    gap: 0,
                    phase,
                    impulse: false,
                },
                &mut failures,
                &mut cases,
            );
        }
    }

    validation(&mut gpu, &mut failures, &mut cases);

    println!("{cases} cases, {failures} FAIL");
    if failures != 0 {
        std::process::exit(1);
    }
}
