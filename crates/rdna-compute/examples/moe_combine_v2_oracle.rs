// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab oracle for the MoE slot-ops combine v2
//! (`Gpu::moe_down_combine_grouped_bf16in_v2_pm`, flag `HIPFIRE_MOE_COMBINE_V2`)
//! against the certified v1 launch
//! (`Gpu::moe_down_combine_grouped_bf16in_pm`).
//!
//! Both launchers have one signature
//! `(y_grouped_bf16, inverse_perm, topk_weights, x_residual, dim, k_top,
//! n_tokens, grouped_rows)` and one arithmetic law, per output column `m` of
//! token `t`:
//!
//! ```text
//! acc = +0.0
//! for k in 0..k_top ascending:
//!     slot = inverse_perm[t*k_top + k]
//!     skip when (slot as u32) >= grouped_rows      // unsigned: negative = skip
//!     acc = fma(w[t*k_top + k], bf16_to_f32(y[slot*dim + m]), acc)
//! x_residual[t*dim + m] = x_residual[t*dim + m] + acc   // one RN add
//! ```
//!
//! v2 only changes the lane mapping (block 256, eight contiguous columns per
//! lane, grid `[n_tokens, dim/2048, 1]`, `dim` a positive multiple of 2048),
//! so the result must be byte-identical to v1, not merely close.
//!
//! Matrix (A3B shape): `n_tokens` in {512, 4096, 8192}, `dim` 2048,
//! `k_top` 8, `grouped_rows = n_tokens * k_top + 3840` (the `256 * 15` expert
//! pad slack), four data kinds each:
//! - `zero`: every BF16 value is +0 or -0, weights and residual finite random;
//! - `random`: NaN/Inf-free BF16 over 2^-8..2^8 magnitudes, finite weights
//!   (including +/-0) and residual (including +/-0);
//! - `slot_ge_rows`: random plus skipped slots `>= grouped_rows` (`rows`,
//!   `rows + 1`, `i32::MAX`); token 0 is skipped entirely, token 1 keeps only
//!   its last slot;
//! - `negative_sentinel`: the same skip pattern with `-1`, `-2`, `i32::MIN`,
//!   `-grouped_rows` in place of the `>= rows` values.
//!
//! Every kind also references the last grouped row (`rows - 1`) and some pad
//! rows. `all_byte_identical` is true only if every residual word of v1 and v2
//! match for every case; the first mismatch (case, flat word index, token,
//! column, both words) is printed. The CPU law is also evaluated and reported
//! (`v1_matches_cpu_law`, `v2_matches_cpu_law`), reported only: the pass
//! criterion is v1 == v2, plus the `random` v1 launch having visibly changed
//! the residual (a no-op v1/v2 pair cannot pass).
//!
//! Timing (random kind only): hipEvent pair around one launch, 5 warmups and
//! 25 timed reps per arm, median / p10 / p90 in microseconds. Each arm owns a
//! residual tensor that is restored from a pristine device copy
//! (device-to-device, bracketed by device synchronisation) OUTSIDE every event
//! pair, so v1 and v2 each see the identical initial state every launch. Arm
//! order alternates per rep.
//!
//! The whole GPU mode runs under a 120 s watchdog thread: past it the process
//! prints a failed summary and exits 3. Exit 1 on a failed check, 2 on usage
//! error; non-gfx1201 prints a skipped summary and exits 0.
//!
//! CPU-only modes initialise no GPU: `--plan` prints the matrix and the law;
//! `--self-check` verifies the CPU law against an independent model of the v2
//! vector lane mapping (wave-uniform slot/weight reads, one b128 BF16 group
//! load per slot, two b128 residual loads and stores per lane) on a small
//! matrix (dim 2048/4096, `k_top` 1..=8, all kinds, skipped slots), checks
//! hand-built controls that distinguish the law from four wrong variants
//! (descending order, mul-then-add, -0 accumulator init, residual-folded
//! accumulator), and requires a repeat run to produce the same digest. The
//! GPU mode runs the self-check first and aborts if it fails.
//!
//! ```text
//! cargo build --release -p rdna-compute --features lab --example moe_combine_v2_oracle
//! cargo run --release -p rdna-compute --features lab --example moe_combine_v2_oracle -- --plan
//! cargo run --release -p rdna-compute --features lab --example moe_combine_v2_oracle -- --self-check
//! cargo run --release -p rdna-compute --features lab --example moe_combine_v2_oracle   # GPU, gfx1201
//! ```
//! `--n-tokens 512,4096` restricts the GPU/plan matrix to a subset.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::{Duration, Instant};

const ORACLE: &str = "moe_combine_v2_oracle";
const DIM: usize = 2048;
const K_TOP: usize = 8;
const N_TOKENS: [usize; 3] = [512, 4096, 8192];
/// `256 * 15`: the expert pad slack of an A3B grouped layout.
const GPU_PAD_ROWS: usize = 256 * 15;
const BLOCK: usize = 256;
const WAVE: usize = 32;
const LANE_COLS: usize = 8;
/// Columns one workgroup covers: 256 lanes x 8 columns.
const TILE_COLS: usize = BLOCK * LANE_COLS;
const WARMUPS: usize = 5;
const REPS: usize = 25;
const DEADLINE_SECS: u64 = 120;
const SEED: u64 = 0x5EED_C0DE_0F2B_17A3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Zero,
    Random,
    SlotGeRows,
    NegativeSentinel,
}

impl Kind {
    const ALL: [Kind; 4] = [
        Kind::Zero,
        Kind::Random,
        Kind::SlotGeRows,
        Kind::NegativeSentinel,
    ];
    fn name(self) -> &'static str {
        match self {
            Kind::Zero => "zero",
            Kind::Random => "random",
            Kind::SlotGeRows => "slot_ge_rows",
            Kind::NegativeSentinel => "negative_sentinel",
        }
    }
    fn skips(self) -> bool {
        matches!(self, Kind::SlotGeRows | Kind::NegativeSentinel)
    }
}

struct Args {
    plan: bool,
    self_check: bool,
    n_tokens: Vec<usize>,
}

fn parse_args() -> Args {
    let mut args = Args {
        plan: false,
        self_check: false,
        n_tokens: N_TOKENS.to_vec(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--plan" => args.plan = true,
            "--self-check" => args.self_check = true,
            "--n-tokens" => {
                let list = it.next().unwrap_or_default();
                let parsed: Option<Vec<usize>> = list
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .ok()
                            .filter(|n| N_TOKENS.contains(n))
                    })
                    .collect();
                match parsed {
                    Some(v) if !v.is_empty() => args.n_tokens = v,
                    _ => {
                        eprintln!(
                            "--n-tokens expects a comma list drawn from {N_TOKENS:?}, got {list:?}"
                        );
                        std::process::exit(2);
                    }
                }
            }
            other => {
                eprintln!(
                    "unknown argument {other:?}; usage: [--plan] [--self-check] [--n-tokens 512,4096,8192]"
                );
                std::process::exit(2);
            }
        }
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
    /// Uniform in [-1, 1).
    fn sym(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

struct Case {
    rows: usize,
    /// BF16 bits, `[rows, dim]`.
    y: Vec<u16>,
    /// `[n * k]`, flat slot to grouped row.
    perm: Vec<i32>,
    /// `[n * k]`.
    w: Vec<f32>,
    /// `[n, dim]`.
    res: Vec<f32>,
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Finite BF16: sign random, biased exponent 119..=135 (2^-8..2^8), random
/// mantissa. Never zero, denormal, Inf or NaN.
fn fill_bf16(rng: &mut Rng, y: &mut [u16]) {
    for chunk in y.chunks_mut(4) {
        let bits = rng.next();
        for (j, v) in chunk.iter_mut().enumerate() {
            let b = (bits >> (16 * j)) as u16;
            let exp = 119 + ((b >> 7) & 0xFF) % 17;
            *v = (b & 0x8000) | (exp << 7) | (b & 0x7F);
        }
    }
}

/// Independent of the law's unsigned compare: negative or `>= rows`.
fn skipped(slot: i32, rows: usize) -> bool {
    slot < 0 || slot as usize >= rows
}

fn gen_case(kind: Kind, n: usize, dim: usize, k: usize, pad: usize, seed: u64) -> Case {
    let mut rng = Rng(seed);
    let slots = n * k;
    let rows = slots + pad;

    let mut y = vec![0u16; rows * dim];
    match kind {
        Kind::Zero => {
            for (i, v) in y.iter_mut().enumerate() {
                if i % 5 == 0 {
                    *v = 0x8000;
                }
            }
        }
        _ => fill_bf16(&mut rng, &mut y),
    }

    // Valid slots: a shuffled permutation of 0..slots, then some references to
    // the pad rows and the very last grouped row.
    let mut perm: Vec<i32> = (0..slots as i32).collect();
    for i in (1..slots).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    for p in perm.iter_mut() {
        if rng.next() % 16 == 0 {
            *p = (slots + (rng.next() as usize) % pad) as i32;
        }
    }
    perm[slots - 1] = (rows - 1) as i32;

    if kind.skips() {
        let rows_i = rows as i32;
        for (idx, p) in perm.iter_mut().enumerate() {
            let (t, kk) = (idx / k, idx % k);
            let skip = t == 0 || (t == 1 && kk + 1 < k) || rng.next() % 4 == 0;
            if !skip {
                continue;
            }
            let pick = (rng.next() % 4) as usize;
            *p = if kind == Kind::SlotGeRows {
                [rows_i, rows_i + 1, i32::MAX, rows_i + 257][pick]
            } else {
                [-1, -2, i32::MIN, -rows_i][pick]
            };
        }
    }

    let mut w: Vec<f32> = (0..slots).map(|_| rng.sym() * 0.5).collect();
    for (i, v) in w.iter_mut().enumerate() {
        if i % 11 == 0 {
            *v = if (i / 11) % 2 == 0 { 0.0 } else { -0.0 };
        }
    }
    let mut res: Vec<f32> = (0..n * dim).map(|_| rng.sym() * 4.0).collect();
    for (i, v) in res.iter_mut().enumerate() {
        if i % 7 == 0 {
            *v = -0.0;
        } else if i % 13 == 0 {
            *v = 0.0;
        }
    }
    Case {
        rows,
        y,
        perm,
        w,
        res,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Law {
    Exact,
    DescendingK,
    MulThenAdd,
    NegZeroInit,
    FoldResidual,
}

/// The arithmetic law (and four deliberately wrong variants used as controls),
/// token-major with a per-token accumulator row. Returns residual bit words.
fn law_ref(c: &Case, dim: usize, k: usize, n: usize, law: Law) -> Vec<u32> {
    let rows = c.rows as u32;
    let mut out = vec![0u32; n * dim];
    let mut acc = vec![0f32; dim];
    let order: Vec<usize> = if law == Law::DescendingK {
        (0..k).rev().collect()
    } else {
        (0..k).collect()
    };
    for t in 0..n {
        let base = t * dim;
        for (m, a) in acc.iter_mut().enumerate() {
            *a = match law {
                Law::NegZeroInit => -0.0,
                Law::FoldResidual => c.res[base + m],
                _ => 0.0,
            };
        }
        for &kk in &order {
            let slot = c.perm[t * k + kk] as u32;
            if slot >= rows {
                continue;
            }
            let wv = c.w[t * k + kk];
            let row = &c.y[slot as usize * dim..][..dim];
            for (a, &b) in acc.iter_mut().zip(row) {
                let v = bf16_to_f32(b);
                *a = if law == Law::MulThenAdd {
                    *a + wv * v
                } else {
                    wv.mul_add(v, *a)
                };
            }
        }
        for (m, a) in acc.iter().enumerate() {
            out[base + m] = if law == Law::FoldResidual {
                a.to_bits()
            } else {
                (c.res[base + m] + a).to_bits()
            };
        }
    }
    out
}

#[derive(Default, Debug)]
struct LaneStats {
    slot_loads: usize,
    weight_loads: usize,
    y_group_loads: usize,
    res_loads: usize,
    res_stores: usize,
    columns_written_once: bool,
}

/// One 16-byte vector load (little endian), 16-byte aligned.
fn load_b128(buf: &[u8], off: usize) -> [u32; 4] {
    assert!(off % 16 == 0 && off + 16 <= buf.len(), "b128 load at {off}");
    let mut o = [0u32; 4];
    for (i, w) in o.iter_mut().enumerate() {
        *w = u32::from_le_bytes(buf[off + 4 * i..off + 4 * i + 4].try_into().unwrap());
    }
    o
}

fn store_b128(buf: &mut [u8], off: usize, v: [u32; 4]) {
    assert!(
        off % 16 == 0 && off + 16 <= buf.len(),
        "b128 store at {off}"
    );
    for (i, w) in v.iter().enumerate() {
        buf[off + 4 * i..off + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
}

/// Model of the v2 lane mapping. Grid `[n, dim / 2048]`, block 256 = 8 waves of
/// 32; lane `tid` owns the eight contiguous columns
/// `gy * 2048 + tid * 8 ..`. Slots and weights are read once per wave per `k`
/// (scalar / wave-uniform) and reused across the lane's eight columns; each
/// non-skipped slot costs one b128 BF16 group load; the residual is two b128
/// loads and two b128 stores. Works on little-endian byte images only.
fn lane_model(c: &Case, dim: usize, k: usize, n: usize) -> (Vec<u32>, LaneStats) {
    assert!(dim % TILE_COLS == 0 && dim > 0 && (1..=8).contains(&k));
    let yb: Vec<u8> = c.y.iter().flat_map(|v| v.to_le_bytes()).collect();
    let rb: Vec<u8> = c.res.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut ob = vec![0u8; rb.len()];
    let mut written = vec![0u8; n * dim];
    let mut st = LaneStats::default();
    for t in 0..n {
        for gy in 0..dim / TILE_COLS {
            for wave in 0..BLOCK / WAVE {
                // Wave-uniform scalar loads: once per k, shared by all 32 lanes
                // and all eight columns of each lane.
                let slots: [u32; 8] =
                    std::array::from_fn(|kk| if kk < k { c.perm[t * k + kk] as u32 } else { 0 });
                let wts: [f32; 8] =
                    std::array::from_fn(|kk| if kk < k { c.w[t * k + kk] } else { 0.0 });
                st.slot_loads += k;
                st.weight_loads += k;
                for lane in 0..WAVE {
                    let tid = wave * WAVE + lane;
                    let col0 = gy * TILE_COLS + tid * LANE_COLS;
                    let mut acc = [0f32; 8];
                    for (&slot, &wt) in slots.iter().zip(&wts).take(k) {
                        if slot >= c.rows as u32 {
                            continue;
                        }
                        let g = load_b128(&yb, (slot as usize * dim + col0) * 2);
                        st.y_group_loads += 1;
                        for (j, a) in acc.iter_mut().enumerate() {
                            let word = g[j / 2];
                            let bits = if j % 2 == 0 {
                                word << 16
                            } else {
                                word & 0xFFFF_0000
                            };
                            *a = wt.mul_add(f32::from_bits(bits), *a);
                        }
                    }
                    let roff = (t * dim + col0) * 4;
                    let lo = load_b128(&rb, roff);
                    let hi = load_b128(&rb, roff + 16);
                    st.res_loads += 2;
                    let r = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
                    let mut o = [0u32; 8];
                    for (j, (oj, (&rj, &aj))) in o.iter_mut().zip(r.iter().zip(&acc)).enumerate() {
                        *oj = (f32::from_bits(rj) + aj).to_bits();
                        written[t * dim + col0 + j] += 1;
                    }
                    store_b128(&mut ob, roff, [o[0], o[1], o[2], o[3]]);
                    store_b128(&mut ob, roff + 16, [o[4], o[5], o[6], o[7]]);
                    st.res_stores += 2;
                }
            }
        }
    }
    st.columns_written_once = written.iter().all(|&w| w == 1);
    (words(&ob).collect(), st)
}

fn words(b: &[u8]) -> impl Iterator<Item = u32> + '_ {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
}

fn hx(w: u32) -> String {
    format!("0x{w:08x}")
}

fn fnv(h: &mut u64, v: u32) {
    for b in v.to_le_bytes() {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

type Control = (&'static str, Case, usize, Vec<u32>, Law);

/// Hand-built controls: (name, case, k, expected exact words, variant that
/// must differ from the law).
fn controls() -> Vec<Control> {
    let mut v: Vec<Control> = Vec::new();
    // Order / residual-folded accumulator: 1, 2^-24, 2^-24 with w = 1 and
    // residual -1. Ascending: 1 -> 1 -> 1 -> residual add +0. Descending sums
    // the small terms first (2^-23) and ends at 1 + 2^-23.
    let one = 0x3F80u16;
    let tiny = 0x3380u16; // 2^-24
    let mut y = vec![tiny; 3 * DIM];
    y[..DIM].fill(one);
    v.push((
        "order_and_residual_fold",
        Case {
            rows: 3,
            y,
            perm: vec![0, 1, 2],
            w: vec![1.0; 3],
            res: vec![-1.0; DIM],
        },
        3,
        vec![0u32; DIM],
        Law::DescendingK,
    ));
    // Fused multiply-add vs mul-then-add: slot 0 gives -(1 + 2^-7); slot 1
    // multiplies (1 + 2^-23) by (1 + 2^-7), whose 2^-30 tail only an fma keeps
    // before the add. Exact result 2^-23 * (1 + 2^-7); unfused gives 2^-23.
    let mut y = vec![0x3F81u16; 2 * DIM];
    y[..DIM].fill(0xBF81);
    v.push((
        "fma_vs_mul_then_add",
        Case {
            rows: 2,
            y,
            perm: vec![0, 1],
            w: vec![1.0, f32::from_bits(0x3F80_0001)],
            res: vec![0.0; DIM],
        },
        2,
        vec![0x3401_0000u32; DIM],
        Law::MulThenAdd,
    ));
    // +0 accumulator init and skipped slots (-1 and 1 >= rows = 1): every slot
    // is skipped, so the output is -0 + (+0) = +0; a -0 init would keep -0.
    v.push((
        "plus_zero_init_all_skipped",
        Case {
            rows: 1,
            y: vec![0u16; DIM],
            perm: vec![-1, 1],
            w: vec![1.0, 1.0],
            res: vec![-0.0; DIM],
        },
        2,
        vec![0u32; DIM],
        Law::NegZeroInit,
    ));
    v
}

fn self_check_once() -> (bool, u64, serde_json::Value) {
    let mut ok = true;
    let mut digest = 0xCBF2_9CE4_8422_2325u64;
    let mut failures: Vec<String> = Vec::new();
    let mut cases = 0usize;
    let mut skipped_slots = 0usize;
    let mut fail = |ok: &mut bool, msg: String| {
        *ok = false;
        if failures.len() < 20 {
            failures.push(msg);
        }
    };
    for dim in [DIM, 2 * DIM] {
        for n in [1usize, 5] {
            for k in 1..=K_TOP {
                for kind in Kind::ALL {
                    cases += 1;
                    let tag = format!("dim{dim}/n{n}/k{k}/{}", kind.name());
                    let seed = SEED ^ ((dim as u64) << 32) ^ ((n as u64) << 16) ^ (k as u64);
                    let c = gen_case(kind, n, dim, k, 64, seed);
                    let law = law_ref(&c, dim, k, n, Law::Exact);
                    let (lane, st) = lane_model(&c, dim, k, n);
                    for &w in &law {
                        fnv(&mut digest, w);
                    }
                    if let Some(i) = (0..law.len()).find(|&i| law[i] != lane[i]) {
                        fail(
                            &mut ok,
                            format!(
                                "{tag}: lane model != law at word {i}: {} vs {}",
                                hx(lane[i]),
                                hx(law[i])
                            ),
                        );
                    }
                    // Independent expected operation counts.
                    let valid = c.perm.iter().filter(|&&p| !skipped(p, c.rows)).count();
                    skipped_slots += c.perm.len() - valid;
                    let want_slot = n * (dim / TILE_COLS) * (BLOCK / WAVE) * k;
                    if st.slot_loads != want_slot
                        || st.weight_loads != want_slot
                        || st.y_group_loads != valid * (dim / LANE_COLS)
                        || st.res_loads != n * (dim / LANE_COLS) * 2
                        || st.res_stores != n * (dim / LANE_COLS) * 2
                        || !st.columns_written_once
                    {
                        fail(
                            &mut ok,
                            format!("{tag}: lane op counts {st:?} (valid {valid})"),
                        );
                    }
                    // Kind-specific structure.
                    let neg = c.perm.iter().any(|&p| p < 0);
                    let ge = c.perm.iter().any(|&p| p >= 0 && p as usize >= c.rows);
                    let want = match kind {
                        Kind::Zero | Kind::Random => (false, false),
                        Kind::SlotGeRows => (false, true),
                        Kind::NegativeSentinel => (true, false),
                    };
                    if (neg, ge) != want {
                        fail(&mut ok, format!("{tag}: sentinel mix neg={neg} ge={ge}"));
                    }
                    if kind.skips() && (law[0] != 0 || lane[0] != 0) {
                        // Token 0 is fully skipped and residual[0] = -0.0: +0 out.
                        fail(
                            &mut ok,
                            format!("{tag}: skipped token 0 col 0 = {}", hx(law[0])),
                        );
                    }
                    if kind == Kind::Zero {
                        // Closed form: acc == +0, so out == residual with -0 -> +0.
                        for (i, &w) in law.iter().enumerate() {
                            let r = c.res[i];
                            let want = if r == 0.0 { 0 } else { r.to_bits() };
                            if w != want {
                                fail(&mut ok, format!("{tag}: zero closed form word {i}"));
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    for (name, c, k, expected, wrong) in controls() {
        cases += 1;
        let law = law_ref(&c, DIM, k, 1, Law::Exact);
        let (lane, st) = lane_model(&c, DIM, k, 1);
        let bad = law_ref(&c, DIM, k, 1, wrong);
        for &w in &law {
            fnv(&mut digest, w);
        }
        if law != expected {
            fail(
                &mut ok,
                format!(
                    "control {name}: law {} != expected {}",
                    hx(law[0]),
                    hx(expected[0])
                ),
            );
        }
        if lane != law || !st.columns_written_once {
            fail(&mut ok, format!("control {name}: lane model != law"));
        }
        if bad == law {
            fail(
                &mut ok,
                format!("control {name}: {wrong:?} not distinguished"),
            );
        }
        // The residual-folded variant is distinguished by the order control too.
        if name == "order_and_residual_fold" && law_ref(&c, DIM, k, 1, Law::FoldResidual) == law {
            fail(
                &mut ok,
                "control order_and_residual_fold: FoldResidual not distinguished".into(),
            );
        }
    }
    let json = serde_json::json!({
        "cases": cases,
        "skipped_slots_total": skipped_slots,
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
    let cases: Vec<_> = args
        .n_tokens
        .iter()
        .flat_map(|&n| {
            Kind::ALL.into_iter().map(move |kind| {
                let rows = n * K_TOP + GPU_PAD_ROWS;
                serde_json::json!({
                    "n_tokens": n,
                    "kind": kind.name(),
                    "grouped_rows": rows,
                    "y_bytes": rows * DIM * 2,
                    "residual_bytes": n * DIM * 4,
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
        "dim": DIM,
        "k_top": K_TOP,
        "arms": {
            "v1": "Gpu::moe_down_combine_grouped_bf16in_pm",
            "v2": "Gpu::moe_down_combine_grouped_bf16in_v2_pm",
        },
        "law": "acc=+0; k ascending; skip (slot as u32)>=grouped_rows; acc=fma(w,bf16(y[slot*dim+m]),acc); out=residual+acc",
        "v2_mapping": {
            "block": BLOCK,
            "wave": WAVE,
            "columns_per_lane": LANE_COLS,
            "grid": "[n_tokens, dim/2048, 1]",
            "lane_columns": "gy*2048 + tid*8 .. +8 contiguous",
        },
        "pass": "all_byte_identical (v1 == v2 on every residual word of every case)",
        "reported_only": ["v1_matches_cpu_law", "v2_matches_cpu_law"],
        "timing": { "warmups": WARMUPS, "reps": REPS, "stats": ["median", "p10", "p90"], "clock": "hipEvent", "unit": "us", "kind": "random" },
        "deadline_secs": DEADLINE_SECS,
        "cases": cases,
    })
}

// ---------------------------------------------------------------- GPU mode

struct Dev {
    y: GpuTensor,
    perm: GpuTensor,
    w: GpuTensor,
    res0: GpuTensor,
    res1: GpuTensor,
    res2: GpuTensor,
}

fn htod<T>(gpu: &Gpu, t: &GpuTensor, data: &[T]) {
    // SAFETY: plain-old-data host slices reinterpreted as bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod");
}

fn upload_case(gpu: &mut Gpu, c: &Case, n: usize, k: usize) -> Dev {
    // `inverse_perm` is i32 and `topk_weights` f32, both 4-byte elements; the
    // launchers only check byte spans, so the tensors are tagged F32.
    let y = gpu
        .alloc_tensor(&[c.rows, DIM], DType::BF16)
        .expect("alloc y");
    let perm = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc perm");
    let w = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc w");
    let res0 = gpu.alloc_tensor(&[n, DIM], DType::F32).expect("alloc res0");
    let res1 = gpu.alloc_tensor(&[n, DIM], DType::F32).expect("alloc res1");
    let res2 = gpu.alloc_tensor(&[n, DIM], DType::F32).expect("alloc res2");
    htod(gpu, &y, &c.y);
    htod(gpu, &perm, &c.perm);
    htod(gpu, &w, &c.w);
    htod(gpu, &res0, &c.res);
    Dev {
        y,
        perm,
        w,
        res0,
        res1,
        res2,
    }
}

fn free_dev(gpu: &mut Gpu, d: Dev) {
    for t in [d.y, d.perm, d.w, d.res0, d.res1, d.res2] {
        gpu.free_tensor(t).expect("free");
    }
}

/// Restore `dst` from the pristine `src`, bracketed by device synchronisation.
/// Always called outside timed regions.
fn reset_res(gpu: &Gpu, dst: &GpuTensor, src: &GpuTensor) {
    gpu.bind_thread().expect("bind");
    gpu.hip.device_synchronize().expect("sync before reset");
    gpu.hip
        .memcpy_dtod(&dst.buf, &src.buf, src.byte_size())
        .expect("reset dtod");
    gpu.hip.device_synchronize().expect("sync after reset");
}

type Launch = fn(&mut Gpu, &Dev, &GpuTensor, usize, usize);

fn launch_v1(g: &mut Gpu, d: &Dev, res: &GpuTensor, n: usize, rows: usize) {
    g.moe_down_combine_grouped_bf16in_pm(&d.y, &d.perm, &d.w, res, DIM, K_TOP, n, rows)
        .expect("v1 launch");
}

fn launch_v2(g: &mut Gpu, d: &Dev, res: &GpuTensor, n: usize, rows: usize) {
    g.moe_down_combine_grouped_bf16in_v2_pm(&d.y, &d.perm, &d.w, res, DIM, K_TOP, n, rows)
        .expect("v2 launch");
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

/// One timed launch of an arm: restore its residual outside the event pair.
fn time_arm(gpu: &mut Gpu, d: &Dev, v2: bool, n: usize, rows: usize) -> f64 {
    let dst = if v2 { &d.res2 } else { &d.res1 };
    let run: Launch = if v2 { launch_v2 } else { launch_v1 };
    reset_res(gpu, dst, &d.res0);
    event_us(gpu, |g| run(g, d, dst, n, rows))
}

fn time_case(gpu: &mut Gpu, d: &Dev, n: usize, rows: usize) -> serde_json::Value {
    let (mut t1, mut t2) = (Vec::new(), Vec::new());
    for rep in 0..WARMUPS + REPS {
        let (a, b) = if rep % 2 == 0 {
            let a = time_arm(gpu, d, false, n, rows);
            (a, time_arm(gpu, d, true, n, rows))
        } else {
            let b = time_arm(gpu, d, true, n, rows);
            (time_arm(gpu, d, false, n, rows), b)
        };
        if rep >= WARMUPS {
            t1.push(a);
            t2.push(b);
        }
    }
    let v1 = stats_json(&mut t1);
    let v2 = stats_json(&mut t2);
    let ratio = v1["median_us"].as_f64().unwrap() / v2["median_us"].as_f64().unwrap();
    serde_json::json!({ "v1": v1, "v2": v2, "v1_over_v2_median": ratio })
}

struct CaseResult {
    json: serde_json::Value,
    identical: bool,
    effect_ok: bool,
    first_mismatch: Option<serde_json::Value>,
}

fn run_case(gpu: &mut Gpu, kind: Kind, n: usize, seed: u64) -> CaseResult {
    let c = gen_case(kind, n, DIM, K_TOP, GPU_PAD_ROWS, seed);
    let rows = c.rows;
    let d = upload_case(gpu, &c, n, K_TOP);
    reset_res(gpu, &d.res1, &d.res0);
    reset_res(gpu, &d.res2, &d.res0);
    launch_v1(gpu, &d, &d.res1, n, rows);
    gpu.hip.device_synchronize().expect("sync v1");
    launch_v2(gpu, &d, &d.res2, n, rows);
    gpu.hip.device_synchronize().expect("sync v2");
    let b1 = gpu.download_raw_bytes(&d.res1).expect("download v1");
    let b2 = gpu.download_raw_bytes(&d.res2).expect("download v2");
    let w1: Vec<u32> = words(&b1).collect();
    let w2: Vec<u32> = words(&b2).collect();
    let pristine: Vec<u32> = c.res.iter().map(|v| v.to_bits()).collect();

    let mut mismatches = 0usize;
    let mut first = None;
    for (i, (a, b)) in w1.iter().zip(&w2).enumerate() {
        if a != b {
            mismatches += 1;
            if first.is_none() {
                first = Some(serde_json::json!({
                    "n_tokens": n,
                    "kind": kind.name(),
                    "word_index": i,
                    "token": i / DIM,
                    "column": i % DIM,
                    "v1": hx(*a),
                    "v2": hx(*b),
                    "pristine_residual": hx(pristine[i]),
                }));
            }
        }
    }
    let identical = mismatches == 0 && w1.len() == w2.len();
    // A no-op v1/v2 pair must not pass: the random kind has to move the residual.
    let moved = w1.iter().zip(&pristine).filter(|(a, b)| a != b).count();
    let effect_ok = kind != Kind::Random || moved > 0;

    let law = law_ref(&c, DIM, K_TOP, n, Law::Exact);
    let v1_law = w1 == law;
    let v2_law = w2 == law;

    let mut json = serde_json::json!({
        "oracle": ORACLE,
        "record": "case",
        "n_tokens": n,
        "kind": kind.name(),
        "grouped_rows": rows,
        "byte_identical": identical,
        "mismatching_words": mismatches,
        "first_mismatch": first.clone(),
        "words_changed_vs_pristine": moved,
        "launch_effect_ok": effect_ok,
        "v1_matches_cpu_law": v1_law,
        "v2_matches_cpu_law": v2_law,
    });
    if kind == Kind::Random {
        json["timing"] = time_case(gpu, &d, n, rows);
    }
    free_dev(gpu, d);
    CaseResult {
        json,
        identical,
        effect_ok,
        first_mismatch: first,
    }
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

    let mut all_identical = true;
    let mut effect_ok = true;
    let mut first_mismatch: Option<serde_json::Value> = None;
    let mut timings = Vec::new();
    for &n in &args.n_tokens {
        for kind in Kind::ALL {
            eprintln!("case n_tokens={n} kind={}", kind.name());
            let seed = SEED ^ ((n as u64) << 8) ^ (kind as u64);
            let r = run_case(&mut gpu, kind, n, seed);
            println!("{}", r.json);
            all_identical &= r.identical;
            effect_ok &= r.effect_ok;
            if kind == Kind::Random {
                timings.push(serde_json::json!({
                    "n_tokens": n,
                    "timing": r.json["timing"].clone(),
                }));
            }
            if first_mismatch.is_none() {
                first_mismatch = r.first_mismatch;
            }
        }
    }
    let pass = all_identical && effect_ok;
    eprintln!("all_byte_identical: {all_identical}");
    println!(
        "{}",
        serde_json::json!({
            "oracle": ORACLE,
            "record": "summary",
            "arch": gpu.arch,
            "all_byte_identical": all_identical,
            "first_mismatch": first_mismatch,
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
