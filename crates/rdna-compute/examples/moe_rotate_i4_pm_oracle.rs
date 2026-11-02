// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab oracle for the exact PeaceMaker twin of the down A4 producer
//! (`Gpu::qwen4_moe_rotate128_i4_pm`, flag `HIPFIRE_MOE_ROTATE_I4_PM`, bundle
//! `kernels/moe_rotate_i4_pm_gfx1201.hxaco`) against the HIP incumbent
//! (`Gpu::qwen4_moe_rotate128_i4_incumbent`, module `qwen4_moe_rotate128_i4`).
//!
//! Both launchers have one signature `(h, sorted, k, grouped_rows, total_slots)` and write the
//! same compact flat-slot sidecar of `block_i4_128` records (`d` F32, `s` I32, 64 `qs` bytes,
//! 72 B): record `g * L + sorted[p]` for K128 group `g` of grouped row `p`; a pad row
//! (`sorted[p] < 0`) writes nothing. The twin changes only the launch geometry (grid
//! `[P, K/128, 1]` against the incumbent's folded `[P * K/128, 1, 1]`), so the sidecar must be
//! byte-identical, not merely close.
//!
//! Matrix (A3B down profile): `K = 512` (`--k` takes another K128 multiple; an unsupported K
//! errors, it is never approximated), top-8, `rows` in {512, 4096, 8192} tokens so `L = rows * 8`
//! flat slots and `P` grouped rows, four data kinds each:
//! - `zero`: every BF16 value is +0 or -0 (the unit-block path), `P = L`, `sorted` a permutation;
//! - `random`: finite BF16 over 2^-8..2^8 magnitudes with some all-zero groups, `P = L`,
//!   `sorted` a shuffled permutation;
//! - `padded`: random data, `P = L - L/10 + 256 * 15`: a tenth of the slots are never
//!   referenced and the rest of the rows are pad rows with negative sentinels (`-1`, `-2`,
//!   `i32::MIN`, `-L`) scattered through `sorted`; pad rows hold random data;
//! - `empty`: random data but every row is a pad row (`P = L`, all `sorted < 0`): no record
//!   may be written.
//!
//! Each arm starts from a sidecar poisoned with 0xA5, so a missing, extra or misplaced record
//! shows. The whole sidecar (`K/128 * L * 72` bytes: every record, written or not) is downloaded
//! per arm and byte-compared; the first mismatch (case, byte offset, record, group, slot, grouped
//! row, byte within record, both bytes) is printed. Pass criteria: `all_byte_identical`; every
//! record not named by `sorted` still poison in both arms (`padding_untouched`); and the launch
//! effect is visible (the `random` and `zero` kinds change records, `empty` changes none), so a
//! no-op pair cannot pass.
//!
//! Timing (every `random` case): hipEvent pair around one launch, 5 warmups and 25 timed reps
//! per arm (>= 20), median / p10 / p90 in microseconds, arm order alternating per rep.
//!
//! The whole GPU mode runs under a 120 s watchdog thread: past it the process prints a failed
//! summary and exits 3. Exit 1 on a failed check, 2 on usage error (including an unsupported
//! `--k`); non-gfx1201 prints a skipped summary and exits 0.
//!
//! CPU-only modes initialise no GPU and never call `Gpu::init`: `--plan` prints the matrix and
//! geometry; `--self-check` verifies the launch-geometry reindexing (the incumbent's folded grid
//! and the twin's `[P, K/128]` grid visit every (row, group) once and agree), the address law
//! `g * L + sorted[p]` and the write set of each kind against an independent model, the case
//! generators' invariants, the host shape validator (supported and refused shapes), hand-built
//! controls that distinguish the law from three wrong layouts, and a repeat-run digest. The GPU
//! mode runs the self-check first and aborts if it fails.
//!
//! ```text
//! cargo build --release -p rdna-compute --features lab --example moe_rotate_i4_pm_oracle
//! cargo run --release -p rdna-compute --features lab --example moe_rotate_i4_pm_oracle -- --plan
//! cargo run --release -p rdna-compute --features lab --example moe_rotate_i4_pm_oracle -- --self-check
//! cargo run --release -p rdna-compute --features lab --example moe_rotate_i4_pm_oracle   # GPU, gfx1201
//! ```
//! `--rows 512,4096` restricts the GPU/plan matrix to a subset.

use rdna_compute::moe_rotate_i4_pm::{validate_moe_rotate_i4_pm, MoeRotateI4PmGeometry};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::{Duration, Instant};

const ORACLE: &str = "moe_rotate_i4_pm_oracle";
const K_DEFAULT: usize = 512;
const TOP_K: usize = 8;
const ROWS: [usize; 3] = [512, 4096, 8192];
/// `256 * 15`: the expert pad slack of an A3B grouped layout.
const PAD_ROWS: usize = 256 * 15;
const GROUP: usize = 128;
const RECORD: usize = 72;
const POISON: u8 = 0xA5;
const WARMUPS: usize = 5;
const REPS: usize = 25;
const DEADLINE_SECS: u64 = 120;
const SEED: u64 = 0x5EED_A4C2_0F2B_17A3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Zero,
    Random,
    Padded,
    Empty,
}

impl Kind {
    const ALL: [Kind; 4] = [Kind::Zero, Kind::Random, Kind::Padded, Kind::Empty];
    fn name(self) -> &'static str {
        match self {
            Kind::Zero => "zero",
            Kind::Random => "random",
            Kind::Padded => "padded",
            Kind::Empty => "empty",
        }
    }
}

struct Args {
    plan: bool,
    self_check: bool,
    rows: Vec<usize>,
    k: usize,
}

fn parse_args() -> Args {
    let mut args = Args {
        plan: false,
        self_check: false,
        rows: ROWS.to_vec(),
        k: K_DEFAULT,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--plan" => args.plan = true,
            "--self-check" => args.self_check = true,
            "--rows" => {
                let list = it.next().unwrap_or_default();
                let parsed: Option<Vec<usize>> = list
                    .split(',')
                    .map(|s| s.trim().parse::<usize>().ok().filter(|n| ROWS.contains(n)))
                    .collect();
                match parsed {
                    Some(v) if !v.is_empty() => args.rows = v,
                    _ => {
                        eprintln!("--rows expects a comma list drawn from {ROWS:?}, got {list:?}");
                        std::process::exit(2);
                    }
                }
            }
            "--k" => {
                let raw = it.next().unwrap_or_default();
                match raw.trim().parse::<usize>() {
                    Ok(k) => args.k = k,
                    Err(_) => {
                        eprintln!("--k expects a positive multiple of 128, got {raw:?}");
                        std::process::exit(2);
                    }
                }
            }
            other => {
                eprintln!(
                    "unknown argument {other:?}; usage: [--plan] [--self-check] [--rows 512,4096,8192] [--k 512]"
                );
                std::process::exit(2);
            }
        }
    }
    // An unsupported K is an error, never an approximation.
    let widest = ROWS.iter().copied().max().unwrap() * TOP_K + PAD_ROWS;
    if let Err(why) = validate_moe_rotate_i4_pm(args.k, widest, widest) {
        eprintln!("unsupported shape: {why}");
        std::process::exit(2);
    }
    args
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// One generated launch: `h` BF16 bits `[p, k]`, `sorted` `[p]`, `l` total flat slots.
struct Case {
    k: usize,
    p: usize,
    l: usize,
    h: Vec<u16>,
    sorted: Vec<i32>,
}

/// Finite BF16: sign random, biased exponent 119..=135 (2^-8..2^8), random mantissa. Never zero,
/// denormal, Inf or NaN, so the rotation cannot overflow f32.
fn fill_bf16(rng: &mut Rng, h: &mut [u16]) {
    for chunk in h.chunks_mut(4) {
        let bits = rng.next();
        for (j, v) in chunk.iter_mut().enumerate() {
            let b = (bits >> (16 * j)) as u16;
            let exp = 119 + ((b >> 7) & 0xFF) % 17;
            *v = (b & 0x8000) | (exp << 7) | (b & 0x7F);
        }
    }
}

fn shuffle<T>(rng: &mut Rng, v: &mut [T]) {
    for i in (1..v.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

fn gen_case(kind: Kind, l: usize, pad: usize, k: usize, seed: u64) -> Case {
    let mut rng = Rng(seed);
    let groups = k / GROUP;
    let mut perm: Vec<i32> = (0..l as i32).collect();
    shuffle(&mut rng, &mut perm);
    let (p, sorted) = match kind {
        Kind::Zero | Kind::Random => (l, perm),
        Kind::Padded => {
            let valid = l - l / 10;
            let p = valid + pad;
            let mut v: Vec<i32> = perm[..valid].to_vec();
            let neg = [-1, -2, i32::MIN, -(l as i32)];
            for _ in 0..pad {
                v.push(neg[(rng.next() % 4) as usize]);
            }
            shuffle(&mut rng, &mut v);
            (p, v)
        }
        Kind::Empty => {
            let neg = [-1, -2, i32::MIN, -(l as i32)];
            let v: Vec<i32> = (0..l).map(|_| neg[(rng.next() % 4) as usize]).collect();
            (l, v)
        }
    };
    let mut h = vec![0u16; p * k];
    match kind {
        Kind::Zero => {
            for (i, v) in h.iter_mut().enumerate() {
                if i % 5 == 0 {
                    *v = 0x8000;
                }
            }
        }
        _ => {
            fill_bf16(&mut rng, &mut h);
            if kind == Kind::Random {
                // Some all-zero groups: the unit-block (amax == 0) path inside random data.
                for row in (0..p).filter(|r| r % 13 == 0) {
                    let g = (row / 13) % groups;
                    h[row * k + g * GROUP..row * k + (g + 1) * GROUP].fill(0);
                }
            }
        }
    }
    Case { k, p, l, h, sorted }
}

// ----------------------------------------------------------------- CPU law

/// The record index of (group `g`, grouped row `p`) under a layout law; `None` writes nothing.
/// `Exact` is the sidecar law; the others are deliberately wrong layouts used as controls.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Law {
    Exact,
    SlotMajor,
    RowIndexed,
    PadWraps,
}

fn record_index(law: Law, c: &Case, p: usize, g: usize) -> Option<usize> {
    let groups = c.k / GROUP;
    let slot = c.sorted[p];
    match law {
        Law::Exact => (slot >= 0).then(|| g * c.l + slot as usize),
        Law::SlotMajor => (slot >= 0).then(|| slot as usize * groups + g),
        Law::RowIndexed => Some(g * c.l + p),
        Law::PadWraps => Some(g * c.l + (slot as u32 as usize) % c.l),
    }
}

/// Write set of a law: per record, whether some (p, g) writes it, plus the count of writers.
fn write_set(law: Law, c: &Case) -> (Vec<u32>, bool) {
    let groups = c.k / GROUP;
    let mut writers = vec![0u32; groups * c.l];
    let mut in_range = true;
    for p in 0..c.p {
        for g in 0..groups {
            if let Some(i) = record_index(law, c, p, g) {
                if i < writers.len() {
                    writers[i] += 1;
                } else {
                    in_range = false;
                }
            }
        }
    }
    (writers, in_range)
}

/// The (row, group) each launch geometry assigns to a workgroup: the incumbent folds them into
/// grid x (`p = x / groups`, `g = x % groups`), the twin uses `(x, y)` directly.
fn incumbent_pg(x: usize, groups: usize) -> (usize, usize) {
    (x / groups, x % groups)
}

fn fnv(h: &mut u64, v: u64) {
    *h ^= v;
    *h = h.wrapping_mul(0x0000_0100_0000_01B3);
}

fn fail(failures: &mut Vec<String>, ok: &mut bool, msg: String) {
    *ok = false;
    if failures.len() < 20 {
        failures.push(msg);
    }
}

fn self_check_once() -> (bool, u64, serde_json::Value) {
    let mut ok = true;
    let mut digest = 0xCBF2_9CE4_8422_2325u64;
    let mut failures: Vec<String> = Vec::new();
    let mut cases = 0usize;

    for &k in &[128usize, 512, 640, 1024] {
        let groups = k / GROUP;
        for &l in &[1usize, 5, 64, 300] {
            for kind in Kind::ALL {
                cases += 1;
                let tag = format!("k{k}/l{l}/{}", kind.name());
                let seed = SEED ^ ((k as u64) << 32) ^ ((l as u64) << 8) ^ (kind as u64);
                let pad = 7;
                let c = gen_case(kind, l, pad, k, seed);
                let valid = c.sorted.iter().filter(|&&s| s >= 0).count();

                // Generator invariants: sorted values < L, valid slots unique, kind structure.
                let mut seen = vec![false; l];
                for &s in c.sorted.iter().filter(|&&s| s >= 0) {
                    if s as usize >= l || std::mem::replace(&mut seen[s as usize], true) {
                        fail(
                            &mut failures,
                            &mut ok,
                            format!("{tag}: slot {s} repeated or >= L"),
                        );
                    }
                }
                if c.h.len() != c.p * k || c.sorted.len() != c.p {
                    fail(&mut failures, &mut ok, format!("{tag}: operand extents"));
                }
                let want_valid = match kind {
                    Kind::Zero | Kind::Random => l,
                    Kind::Padded => l - l / 10,
                    Kind::Empty => 0,
                };
                if valid != want_valid {
                    fail(
                        &mut failures,
                        &mut ok,
                        format!("{tag}: {valid} valid rows, want {want_valid}"),
                    );
                }
                if kind == Kind::Padded && c.p != valid + pad {
                    fail(&mut failures, &mut ok, format!("{tag}: padded P {}", c.p));
                }
                if kind == Kind::Zero && c.h.iter().any(|&v| v & 0x7FFF != 0) {
                    fail(
                        &mut failures,
                        &mut ok,
                        format!("{tag}: zero kind has a nonzero value"),
                    );
                }
                if kind != Kind::Zero && c.h.iter().any(|&v| (v >> 7) & 0xFF == 0xFF) {
                    fail(&mut failures, &mut ok, format!("{tag}: non-finite BF16"));
                }

                // Address law against the independent write-set model: one record per valid
                // (slot, group), nothing from pad rows, nothing out of range.
                let (writers, in_range) = write_set(Law::Exact, &c);
                let written = writers.iter().filter(|&&w| w > 0).count();
                if !in_range || writers.iter().any(|&w| w > 1) || written != valid * groups {
                    fail(
                        &mut failures,
                        &mut ok,
                        format!(
                            "{tag}: write set {written} (want {}), in_range {in_range}",
                            valid * groups
                        ),
                    );
                }
                for (s, &seen) in seen.iter().enumerate() {
                    for g in 0..groups {
                        if (writers[g * l + s] == 1) != seen {
                            fail(
                                &mut failures,
                                &mut ok,
                                format!("{tag}: record g{g} slot {s} coverage"),
                            );
                        }
                    }
                }
                if kind == Kind::Empty && written != 0 {
                    fail(
                        &mut failures,
                        &mut ok,
                        format!("{tag}: empty kind writes {written} records"),
                    );
                }

                // Geometry: both grids visit every (row, group) once and agree.
                let mut visited = vec![0u8; c.p * groups];
                for x in 0..c.p * groups {
                    let (p, g) = incumbent_pg(x, groups);
                    if p >= c.p || g >= groups {
                        fail(
                            &mut failures,
                            &mut ok,
                            format!("{tag}: incumbent grid x {x} out of range"),
                        );
                        continue;
                    }
                    visited[p * groups + g] += 1;
                }
                for by in 0..groups {
                    for bx in 0..c.p {
                        let x = bx * groups + by;
                        if incumbent_pg(x, groups) != (bx, by) {
                            fail(
                                &mut failures,
                                &mut ok,
                                format!("{tag}: grids disagree at ({bx},{by})"),
                            );
                        }
                    }
                }
                if visited.iter().any(|&v| v != 1) {
                    fail(
                        &mut failures,
                        &mut ok,
                        format!("{tag}: incumbent grid is not a bijection"),
                    );
                }
                match validate_moe_rotate_i4_pm(k, c.p, l) {
                    Ok(geo) => {
                        let want = MoeRotateI4PmGeometry {
                            grid: [c.p as u32, groups as u32, 1],
                            groups,
                            h_bytes: c.p * k * 2,
                            xq_bytes: groups * l * RECORD,
                        };
                        if geo != want {
                            fail(
                                &mut failures,
                                &mut ok,
                                format!("{tag}: geometry {geo:?} != {want:?}"),
                            );
                        }
                    }
                    Err(why) => fail(&mut failures, &mut ok, format!("{tag}: refused: {why}")),
                }

                for (i, &w) in writers.iter().enumerate() {
                    if w > 0 {
                        fnv(&mut digest, i as u64);
                    }
                }
                fnv(&mut digest, valid as u64);
            }
        }
    }

    // Controls: a non-identity permutation with pad rows must distinguish the law from the
    // three wrong layouts (slot-major records, row-indexed records, pad rows wrapping in).
    {
        cases += 1;
        let k = 640;
        let groups = k / GROUP;
        let c = Case {
            k,
            p: 6,
            l: 5,
            h: vec![0; 6 * k],
            sorted: vec![3, -1, 0, 4, i32::MIN, 1],
        };
        let exact = |p: usize, g: usize| record_index(Law::Exact, &c, p, g);
        // Hand-computed: row 0 slot 3 group 2 -> 2*5 + 3; row 3 slot 4 group 4 -> 4*5 + 4.
        if exact(0, 2) != Some(13)
            || exact(3, 4) != Some(24)
            || exact(1, 0).is_some()
            || exact(4, 3).is_some()
        {
            fail(
                &mut failures,
                &mut ok,
                "control: exact law hand values".into(),
            );
        }
        for wrong in [Law::SlotMajor, Law::RowIndexed, Law::PadWraps] {
            let differs =
                (0..c.p).any(|p| (0..groups).any(|g| record_index(wrong, &c, p, g) != exact(p, g)));
            if !differs {
                fail(
                    &mut failures,
                    &mut ok,
                    format!("control: {wrong:?} not distinguished"),
                );
            }
        }
        // The write sets also differ in count: the pad-wrapping variant writes pad rows.
        let (exact_set, _) = write_set(Law::Exact, &c);
        let (wrap_set, _) = write_set(Law::PadWraps, &c);
        if exact_set.iter().sum::<u32>() as usize != 4 * groups
            || wrap_set.iter().sum::<u32>() as usize != c.p * groups
        {
            fail(&mut failures, &mut ok, "control: writer counts".into());
        }
    }

    // Host shape validator: refused shapes error, supported ones carry the exact byte spans.
    for (k, p, l) in [
        (0usize, 8usize, 8usize),
        (64, 8, 8),
        (576, 8, 8),
        (512, 0, 8),
        (512, 8, 0),
        (GROUP * 65_536, 8, 8),
        (512, i32::MAX as usize + 1, 8),
        (512, 8, i32::MAX as usize + 1),
        (512, 8, (1 << 30) + 1),
        (1024, 8, 1 << 30),
    ] {
        cases += 1;
        if validate_moe_rotate_i4_pm(k, p, l).is_ok() {
            fail(
                &mut failures,
                &mut ok,
                format!("validator accepted K={k} P={p} L={l}"),
            );
        }
    }
    for &rows in &ROWS {
        cases += 1;
        let l = rows * TOP_K;
        let p = l + PAD_ROWS;
        match validate_moe_rotate_i4_pm(K_DEFAULT, p, l) {
            Ok(g)
                if g.grid == [p as u32, 4, 1]
                    && g.xq_bytes == 4 * l * RECORD
                    && g.h_bytes == p * 1024 => {}
            other => fail(
                &mut failures,
                &mut ok,
                format!("A3B rows {rows}: {other:?}"),
            ),
        }
    }

    let json = serde_json::json!({
        "cases": cases,
        "digest": format!("{digest:016x}"),
        "failures": failures,
    });
    (ok, digest, json)
}

fn self_check() -> (bool, serde_json::Value) {
    let (ok1, d1, mut json) = self_check_once();
    let (ok2, d2, _) = self_check_once();
    let deterministic = d1 == d2;
    let pass = ok1 && ok2 && deterministic;
    json["oracle"] = ORACLE.into();
    json["record"] = "self_check".into();
    json["gpu_initialized"] = false.into();
    json["deterministic_digest"] = deterministic.into();
    json["pass"] = pass.into();
    (pass, json)
}

fn plan_json(args: &Args) -> serde_json::Value {
    let k = args.k;
    let cases: Vec<_> = args
        .rows
        .iter()
        .flat_map(|&rows| {
            Kind::ALL.into_iter().map(move |kind| {
                let l = rows * TOP_K;
                let p = match kind {
                    Kind::Padded => l - l / 10 + PAD_ROWS,
                    _ => l,
                };
                let geo = validate_moe_rotate_i4_pm(k, p, l).ok();
                serde_json::json!({
                    "rows": rows,
                    "kind": kind.name(),
                    "grouped_rows": p,
                    "total_slots": l,
                    "pm_grid": geo.map(|g| g.grid),
                    "incumbent_grid": [p * (k / GROUP), 1, 1],
                    "h_bytes": p * k * 2,
                    "sidecar_bytes": k / GROUP * l * RECORD,
                    "timed": kind == Kind::Random,
                })
            })
        })
        .collect();
    serde_json::json!({
        "oracle": ORACLE,
        "record": "plan",
        "plan_only": true,
        "gpu_initialized": false,
        "k": k,
        "top_k": TOP_K,
        "arms": {
            "incumbent": "Gpu::qwen4_moe_rotate128_i4_incumbent",
            "pm": "Gpu::qwen4_moe_rotate128_i4_pm",
        },
        "record_law": "record g*L + sorted[p] of 72 B {f32 d, i32 s, 64 qs}; sorted[p] < 0 writes nothing",
        "kernarg": "h@0 sorted@8 xq@16 signs1@24 signs2@32 K@40 P@44 L@48 (56 B)",
        "pass": "all_byte_identical (whole sidecar, every record, poisoned 0xA5 before each arm) && padding_untouched && launch_effect_ok",
        "timing": { "warmups": WARMUPS, "reps": REPS, "stats": ["median", "p10", "p90"], "clock": "hipEvent", "unit": "us", "kind": "random" },
        "deadline_secs": DEADLINE_SECS,
        "cases": cases,
    })
}

// ---------------------------------------------------------------- GPU mode

struct Dev {
    h: GpuTensor,
    sorted: GpuTensor,
}

fn htod<T>(gpu: &Gpu, t: &GpuTensor, data: &[T]) {
    // SAFETY: plain-old-data host slices reinterpreted as bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod");
}

fn upload_case(gpu: &mut Gpu, c: &Case) -> Dev {
    // `sorted` is i32, a 4-byte element: the launchers only check byte spans, so it is tagged F32.
    let h = gpu.alloc_tensor(&[c.p, c.k], DType::BF16).expect("alloc h");
    let sorted = gpu.alloc_tensor(&[c.p], DType::F32).expect("alloc sorted");
    htod(gpu, &h, &c.h);
    htod(gpu, &sorted, &c.sorted);
    Dev { h, sorted }
}

fn free_dev(gpu: &mut Gpu, d: Dev) {
    for t in [d.h, d.sorted] {
        gpu.free_tensor(t).expect("free");
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Incumbent,
    Pm,
    /// The grouped-position twin (`qwen4_moe_rotate128_i4_gp_pm`): record `g * P + row`.
    Gp,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Incumbent => "incumbent",
            Arm::Pm => "pm",
            Arm::Gp => "gp",
        }
    }
}

fn launch(g: &mut Gpu, arm: Arm, d: &Dev, c: &Case) -> Result<(), String> {
    let r = match arm {
        Arm::Incumbent => g.qwen4_moe_rotate128_i4_incumbent(&d.h, &d.sorted, c.k, c.p, c.l),
        Arm::Pm => g.qwen4_moe_rotate128_i4_pm(&d.h, &d.sorted, c.k, c.p, c.l),
        Arm::Gp => g.qwen4_moe_rotate128_i4_gp_pm(&d.h, &d.sorted, c.k, c.p, c.l),
    };
    r.map(|_| ())
        .map_err(|e| format!("{} launch: {e}", arm.name()))
}

/// Poison the first `bytes` of the down sidecar, launch `arm`, return those bytes.
fn run_arm(gpu: &mut Gpu, arm: Arm, d: &Dev, c: &Case, bytes: usize) -> Result<Vec<u8>, String> {
    let sidecar = gpu
        .scratch
        .qwen4_moe_down_i4_scratch
        .as_ref()
        .ok_or("down sidecar scratch not allocated")?;
    gpu.bind_thread().map_err(|e| e.to_string())?;
    gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
    gpu.hip
        .memset(sidecar, POISON as i32, bytes)
        .map_err(|e| e.to_string())?;
    gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
    launch(gpu, arm, d, c)?;
    gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
    // The launch may have regrown nothing (the prime launch sized the scratch); re-borrow anyway.
    let sidecar = gpu
        .scratch
        .qwen4_moe_down_i4_scratch
        .as_ref()
        .ok_or("down sidecar scratch vanished")?;
    let mut out = vec![0u8; bytes];
    gpu.hip
        .memcpy_dtoh(&mut out, sidecar)
        .map_err(|e| e.to_string())?;
    Ok(out)
}

fn event_us(gpu: &mut Gpu, launch: impl FnOnce(&mut Gpu)) -> f64 {
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    gpu.hip
        .event_record(&start, gpu.active_stream.as_ref())
        .expect("record start");
    launch(gpu);
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("synchronize stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed");
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    ms as f64 * 1_000.0
}

/// Nearest-rank percentile over a sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn median(sorted: &[f64]) -> f64 {
    let m = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[m]
    } else {
        (sorted[m - 1] + sorted[m]) / 2.0
    }
}

fn stats_json(samples: &mut [f64]) -> serde_json::Value {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite timing"));
    serde_json::json!({
        "median_us": median(samples),
        "p10_us": percentile(samples, 0.10),
        "p90_us": percentile(samples, 0.90),
        "min_us": samples[0],
        "reps": samples.len(),
    })
}

fn time_arm(gpu: &mut Gpu, arm: Arm, d: &Dev, c: &Case) -> f64 {
    gpu.hip
        .device_synchronize()
        .expect("sync before timed launch");
    event_us(gpu, |g| launch(g, arm, d, c).expect("timed launch"))
}

fn time_case(gpu: &mut Gpu, d: &Dev, c: &Case) -> serde_json::Value {
    let (mut inc, mut pm, mut gp) = (Vec::new(), Vec::new(), Vec::new());
    for rep in 0..WARMUPS + REPS {
        let (a, b, e) = if rep % 2 == 0 {
            let a = time_arm(gpu, Arm::Incumbent, d, c);
            let b = time_arm(gpu, Arm::Pm, d, c);
            (a, b, time_arm(gpu, Arm::Gp, d, c))
        } else {
            let e = time_arm(gpu, Arm::Gp, d, c);
            let b = time_arm(gpu, Arm::Pm, d, c);
            (time_arm(gpu, Arm::Incumbent, d, c), b, e)
        };
        if rep >= WARMUPS {
            inc.push(a);
            pm.push(b);
            gp.push(e);
        }
    }
    let inc = stats_json(&mut inc);
    let pm = stats_json(&mut pm);
    let gp = stats_json(&mut gp);
    let ratio = inc["median_us"].as_f64().unwrap() / pm["median_us"].as_f64().unwrap();
    serde_json::json!({ "incumbent": inc, "pm": pm, "gp": gp, "incumbent_over_pm_median": ratio })
}

fn hx(b: u8) -> String {
    format!("{b:02x}")
}

fn hex_record(r: &[u8]) -> String {
    r.iter().map(|&b| hx(b)).collect()
}

struct CaseResult {
    json: serde_json::Value,
    identical: bool,
    padding_untouched: bool,
    effect_ok: bool,
    first_mismatch: Option<serde_json::Value>,
}

/// Whether record `r` (`g * L + slot`) is named by `sorted` (some valid row writes it).
fn named_records(c: &Case) -> Vec<bool> {
    let groups = c.k / GROUP;
    let mut named = vec![false; groups * c.l];
    for &s in c.sorted.iter().filter(|&&s| s >= 0) {
        for g in 0..groups {
            named[g * c.l + s as usize] = true;
        }
    }
    named
}

fn run_case(
    gpu: &mut Gpu,
    kind: Kind,
    rows: usize,
    k: usize,
    seed: u64,
) -> Result<CaseResult, String> {
    let l = rows * TOP_K;
    let c = gen_case(kind, l, PAD_ROWS, k, seed);
    let geometry = validate_moe_rotate_i4_pm(k, c.p, l)?;
    let groups = geometry.groups;
    let bytes = geometry.xq_bytes;
    let d = upload_case(gpu, &c);

    // Prime the sidecar scratch to its final size (and the sign tables / kernels) with the
    // incumbent; both measured arms then reuse one allocation and one poison pattern.
    launch(gpu, Arm::Incumbent, &d, &c)?;
    launch(gpu, Arm::Gp, &d, &c)?;
    gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
    let inc = run_arm(gpu, Arm::Incumbent, &d, &c, bytes)?;
    let pm = run_arm(gpu, Arm::Pm, &d, &c, bytes)?;
    let gp_bytes = groups * c.p * RECORD;
    let gp = run_arm(gpu, Arm::Gp, &d, &c, gp_bytes)?;

    let named = named_records(&c);
    let mut mismatches = 0usize;
    let mut first = None;
    for (i, (a, b)) in inc.iter().zip(&pm).enumerate() {
        if a != b {
            mismatches += 1;
            if first.is_none() {
                let (rec, byte) = (i / RECORD, i % RECORD);
                let (g, slot) = (rec / l, rec % l);
                let row = c.sorted.iter().position(|&s| s == slot as i32);
                first = Some(serde_json::json!({
                    "rows": rows,
                    "kind": kind.name(),
                    "byte_offset": i,
                    "record": rec,
                    "group": g,
                    "slot": slot,
                    "grouped_row": row,
                    "byte_in_record": byte,
                    "field": if byte < 4 { "d" } else if byte < 8 { "s" } else { "qs" },
                    "incumbent": hx(*a),
                    "pm": hx(*b),
                }));
            }
        }
    }
    let mut identical = mismatches == 0 && inc.len() == pm.len();

    // Grouped-position twin: record `g * P + row` of every valid row equals the flat record
    // `g * L + sorted[row]` byte for byte; every pad row's record stays poison.
    let mut gp_mismatches = 0usize;
    let mut gp_pad_touched = 0usize;
    let mut gp_first = None;
    for g in 0..groups {
        for (row, &slot) in c.sorted.iter().enumerate() {
            let at = (g * c.p + row) * RECORD;
            let got = &gp[at..at + RECORD];
            if slot < 0 {
                gp_pad_touched += usize::from(got.iter().any(|&b| b != POISON));
            } else {
                let want = &inc[(g * l + slot as usize) * RECORD..][..RECORD];
                let bad = got.iter().zip(want).filter(|(a, b)| a != b).count();
                gp_mismatches += bad;
                if bad > 0 && gp_first.is_none() {
                    gp_first = Some(serde_json::json!({
                        "group": g, "grouped_row": row, "slot": slot,
                        "gp_record": hex_record(got), "incumbent_record": hex_record(want),
                    }));
                }
            }
        }
    }
    let gp_identical = gp_mismatches == 0 && gp.len() == groups * c.p * RECORD;
    identical &= gp_identical;

    let record_poison = |buf: &[u8], r: usize| {
        buf[r * RECORD..(r + 1) * RECORD]
            .iter()
            .all(|&b| b == POISON)
    };
    let mut unnamed = 0usize;
    let mut touched = [0usize; 2];
    let mut changed_named = [0usize; 2];
    for r in 0..groups * l {
        for (arm, buf) in [&inc, &pm].into_iter().enumerate() {
            let poison = record_poison(buf, r);
            if named[r] {
                if !poison {
                    changed_named[arm] += 1;
                }
            } else if !poison {
                touched[arm] += 1;
            }
        }
        if !named[r] {
            unnamed += 1;
        }
    }
    let padding_untouched = touched == [0, 0] && gp_pad_touched == 0;
    // A no-op pair must not pass: random / zero / padded kinds have to change records, empty none.
    let want_named = named.iter().filter(|&&n| n).count();
    let effect_ok = match kind {
        Kind::Empty => changed_named == [0, 0] && want_named == 0,
        _ => changed_named.iter().all(|&n| n > 0),
    };

    let mut json = serde_json::json!({
        "oracle": ORACLE,
        "record": "case",
        "rows": rows,
        "kind": kind.name(),
        "k": k,
        "grouped_rows": c.p,
        "total_slots": l,
        "sidecar_bytes": bytes,
        "records": groups * l,
        "named_records": want_named,
        "unnamed_records": unnamed,
        "byte_identical": identical,
        "mismatching_bytes": mismatches,
        "first_mismatch": first.clone(),
        "padding_untouched": padding_untouched,
        "unnamed_records_touched": { "incumbent": touched[0], "pm": touched[1] },
        "named_records_changed": { "incumbent": changed_named[0], "pm": changed_named[1] },
        "launch_effect_ok": effect_ok,
        "pm_selected_by_flag": gpu.moe_rotate_i4_pm_selected(k, c.p, l),
        "gp_plane_bytes": gp_bytes,
        "gp_byte_identical": gp_identical,
        "gp_mismatching_bytes": gp_mismatches,
        "gp_first_mismatch": gp_first,
        "gp_pad_records_touched": gp_pad_touched,
    });
    if kind == Kind::Random {
        json["timing"] = time_case(gpu, &d, &c);
    }
    free_dev(gpu, d);
    Ok(CaseResult {
        json,
        identical,
        padding_untouched,
        effect_ok,
        first_mismatch: first,
    })
}

fn start_watchdog() {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(DEADLINE_SECS));
        println!(
            "{}",
            serde_json::json!({
                "oracle": ORACLE,
                "record": "summary",
                "pass": false,
                "all_byte_identical": serde_json::Value::Null,
                "error": format!("{DEADLINE_SECS}s deadline exceeded"),
            })
        );
        eprintln!("FAIL: {ORACLE} exceeded {DEADLINE_SECS}s");
        std::process::exit(3);
    });
}

fn gpu_main(args: &Args) -> i32 {
    let (sc_pass, sc_json) = self_check();
    println!("{sc_json}");
    if !sc_pass {
        eprintln!("FAIL: CPU self-check failed; not touching the GPU");
        return 1;
    }
    let t0 = Instant::now();
    start_watchdog();
    let mut gpu = Gpu::init().expect("Gpu::init");
    gpu.bind_thread().expect("bind");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: {ORACLE} requires exact gfx1201");
        println!(
            "{}",
            serde_json::json!({
                "oracle": ORACLE,
                "record": "summary",
                "skipped": true,
                "arch": gpu.arch,
                "pass": serde_json::Value::Null,
            })
        );
        return 0;
    }
    if !gpu.iu4_a4_c2_candidates_exact() {
        eprintln!(
            "FAIL: the frozen C2 producer candidate set (-DIU4_A4_CANDIDATES=2) is not in force"
        );
        println!(
            "{}",
            serde_json::json!({
                "oracle": ORACLE,
                "record": "summary",
                "arch": gpu.arch,
                "pass": false,
                "error": "C2 producer candidate set not in force",
            })
        );
        return 1;
    }

    let mut all_identical = true;
    let mut all_padding = true;
    let mut effect_ok = true;
    let mut first_mismatch: Option<serde_json::Value> = None;
    let mut timings = Vec::new();
    for &rows in &args.rows {
        for kind in Kind::ALL {
            eprintln!("case rows={rows} kind={} k={}", kind.name(), args.k);
            let seed = SEED ^ ((rows as u64) << 8) ^ (kind as u64);
            let r = match run_case(&mut gpu, kind, rows, args.k, seed) {
                Ok(r) => r,
                Err(why) => {
                    eprintln!("FAIL: rows={rows} kind={}: {why}", kind.name());
                    println!(
                        "{}",
                        serde_json::json!({
                            "oracle": ORACLE,
                            "record": "summary",
                            "arch": gpu.arch,
                            "pass": false,
                            "all_byte_identical": serde_json::Value::Null,
                            "error": format!("rows={rows} kind={}: {why}", kind.name()),
                        })
                    );
                    return 1;
                }
            };
            println!("{}", r.json);
            all_identical &= r.identical;
            all_padding &= r.padding_untouched;
            effect_ok &= r.effect_ok;
            if kind == Kind::Random {
                timings.push(serde_json::json!({
                    "rows": rows,
                    "timing": r.json["timing"].clone(),
                }));
            }
            if first_mismatch.is_none() {
                first_mismatch = r.first_mismatch;
            }
        }
    }
    let pass = all_identical && all_padding && effect_ok;
    eprintln!("all_byte_identical: {all_identical}");
    println!(
        "{}",
        serde_json::json!({
            "oracle": ORACLE,
            "record": "summary",
            "arch": gpu.arch,
            "k": args.k,
            "all_byte_identical": all_identical,
            "first_mismatch": first_mismatch,
            "padding_untouched": all_padding,
            "launch_effect_ok": effect_ok,
            "timing": timings,
            "elapsed_s": t0.elapsed().as_secs_f64(),
            "pass": pass,
        })
    );
    if pass {
        0
    } else {
        1
    }
}

fn main() {
    let args = parse_args();
    if args.plan || args.self_check {
        if args.plan {
            println!("{}", plan_json(&args));
        }
        if args.self_check {
            let (pass, json) = self_check();
            println!("{json}");
            if !pass {
                std::process::exit(1);
            }
        }
        return;
    }
    std::process::exit(gpu_main(&args));
}
