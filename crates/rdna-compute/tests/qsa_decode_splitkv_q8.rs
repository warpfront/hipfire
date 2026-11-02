// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Q8 QSA hardware diagnostics and strict production-dispatch G1 row-byte gate.
//! Direct split-KV-vs-per-head tolerance checks below are diagnostics only.
//! The wide gate invokes `indexed_attention_attention_batch` at every R=1..64
//! under actual `Qwen4VerifyReference::{Eager, Recorded}` bindings and compares
//! every output byte to sequential singleton calls through that same dispatcher.
//! Both references launch eagerly; Recorded binds recorded singleton arithmetic,
//! not a graph replay. All calls share an immutable, seeded, nonzero cache S0
//! and advance the singleton position. This attention-only fixture does not
//! exercise cache append, pooling, or selector-state transitions.
//!
//! The Q8 cache is the dense q8 KV row: per token and KV head eight 34-byte
//! Q8_0 blocks (little-endian f16 scale, then 32 int8 codes; 272 bytes per head,
//! 544 per token row of K or of V), encoded here by `q8_row` (amax / 127 scale,
//! round-to-nearest codes; a few all-zero blocks).
//!
//! * `qsa_splitk_q8_rows1_matches_exact`: single-row launches at a union of
//!   lengths (around the 64-key tile and split boundaries, 1131, 2048, 2049,
//!   2051, 8195), four compress/budget/capacity formulas, random q/K/V, a
//!   pure-valid selection and a mixed one that carries negative tokens, tokens
//!   >= visible and tokens >= full_capacity, for splits 1, 2, 4, 8, 16, 32.
//!   Per head: maxdiff <= 1e-5 * maxabsref + 1e-7. A row with no valid token
//!   must be exactly +0, as in the reference. The selected input bytes must be
//!   unchanged by every launch and every output must be finite.
//! * `qsa_splitk_q8_all_invalid_zero`: all-invalid selections (rows 1 and
//!   batched 1..8) write exactly +0.0 for every split count.
//! * `qsa_splitk_q8_batched_rows_bitwise_individual`: rows 1..8 batched launches
//!   equal, bit for bit, the single-row launches of the same q row, selected row
//!   and position_start + row, for heterogeneous selected_len (compress 8,
//!   budgets around the block boundary, empty slices, S > len, capacity clamp,
//!   an all-invalid row inside every batch of two or more rows). Every output
//!   starts as a NaN sentinel; batched outputs are also compared with the exact
//!   reference.
//! * `qsa_splitk_q8_batched_rows_bitwise_individual_wide`: strict production
//!   parity for all widths 1..64, multiple seeds, compress 1/4/8 and every
//!   starting residue, empty/short/capacity-max selections and contexts spanning
//!   the per-head LDS boundary. Split enabled/disabled arms assert the reached
//!   policy; Q8 split-KV has fixed LDS, not a context-dependent split threshold.
//!   Output prefix/tail guards and inactive rows are poisoned; query, selections,
//!   and K/V must remain immutable. Candidate refusal/mismatch, missing device,
//!   and unsupported wide architecture are fatal. Physical 64 is a lab scope,
//!   not promotion of the production-certified eight-row capability.
//! * `qsa_splitk_q8_selected_tiles_rows_1_2_8`: launches of rows 1, 2 and 8 with
//!   partially filled selected tiles (selected_len 2051 = 32 tiles + 3 tokens
//!   through a capacity clamp and through a block budget, 63..70 around one
//!   tile, 2045..2052 around 2048, empty slices, 8K tails), all-valid, mixed,
//!   all-invalid and row-heterogeneous selections, against an 8K-capacity cache
//!   (8300 tokens) and a 256K-like cache (262144 tokens, selected tokens drawn
//!   from the whole range).
//!
//! Diagnostic tests print tolerance summaries. The strict wide gate prints
//! mode/width/policy and terminal-launch module/object SHA receipts (not a
//! provenance claim for every stage), and accepts raw bytes only. These are
//! finite synthetic operands, not an exceptional-value or full-model proof.
//!
//! `#[ignore]`d: needs a working HIP device. Run explicitly, one test thread
//! (the tests share the one GPU):
//!
//!   cargo test -p rdna-compute --release --features lab --test qsa_decode_splitkv_q8 \
//!       -- --ignored --nocapture --test-threads=1
//!
//! No environment variable is set by this file.

#![cfg(feature = "lab")]

use rdna_compute::prefill_invariant::Qwen4VerifyReference;
use rdna_compute::tensor_ops::{
    indexed_attention_attention_batch, indexed_attention_attention_batch_exact,
    indexed_attention_attention_splitkv_q8_for_test, qsa_decode_attention_policy_for_row_for_test,
    IndexedAttentionAttentionBatch, QsaDecodeAttentionPolicy, QsaKvFormat,
    QSA_DECODE_SPLITK_SPLITS,
};
use rdna_compute::{Gpu, GpuTensor};

const NH: usize = 24;
const NKV: usize = 2;
const HD: usize = 256;
/// KV cache rows: above the longest context (8196 visible) with slack, so
/// tokens in [visible, FULL_CAP) are in-bounds cache rows the kernel must still
/// reject.
const FULL_CAP: usize = 8300;
/// 256K-like KV cache rows for the Q8 selected-tile test (2^18 tokens, 142.6 MB
/// of Q8 K and as much V).
const FULL_CAP_256K: usize = 262_144;
/// Budget large enough never to bind.
const HUGE: usize = 1 << 20;
const SENTINEL: u32 = 0x7fc0_dead;
/// Split counts under test (the kernels accept 1..=32).
const SPLITS: [usize; 6] = [1, 2, 4, 8, 16, 32];

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// Deterministic value in [-scale, scale).
    fn value(&mut self, scale: f32) -> f32 {
        (self.next() as u32 & 0xff_ffff) as f32 / (1u32 << 24) as f32 * 2.0 * scale - scale
    }
}

/// `rows` query rows: per head 256 query values (+/-2) then 256 gate values (+/-3).
fn make_q(rng: &mut Rng, rows: usize) -> Vec<f32> {
    let mut q = Vec::with_capacity(rows * 2 * NH * HD);
    for _ in 0..rows * NH {
        q.extend((0..HD).map(|_| rng.value(2.0)));
        q.extend((0..HD).map(|_| rng.value(3.0)));
    }
    q
}

/// IEEE binary16 bits of a finite non-negative `x`, round to nearest even
/// (subnormals and overflow to infinity included).
fn f32_to_f16_bits(x: f32) -> u16 {
    assert!(x.is_finite() && x >= 0.0, "f16 encode of {x}");
    let b = x.to_bits();
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    if exp >= 31 {
        return 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return 0;
        }
        let full = mant | 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut half = full >> shift;
        let rem = full & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if rem > halfway || (rem == halfway && half & 1 == 1) {
            half += 1;
        }
        return half as u16;
    }
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    h as u16
}

/// One Q8 token row of K or of V: per KV head eight Q8_0 blocks of 34 bytes (the
/// little-endian f16 scale, then 32 int8 codes), `scale = amax / 127`,
/// `code = round(x / scale)`. Block magnitudes vary (0.1..1.5, one block in 64
/// all zero) so the scales are exercised.
fn q8_row(rng: &mut Rng, out: &mut Vec<u8>) {
    for _ in 0..NKV * (HD / 32) {
        let amp = if rng.below(64) == 0 {
            0.0
        } else {
            0.8 + rng.value(0.7)
        };
        let x: Vec<f32> = (0..32).map(|_| rng.value(amp)).collect();
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend(x.iter().map(|&v| {
            if d == 0.0 {
                0u8
            } else {
                (v / d).round().clamp(-127.0, 127.0) as i8 as u8
            }
        }));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Valid, negative, >= visible and >= full_cap tokens (pure valid when the
    /// selected length is 1).
    Mixed,
    AllValid,
    AllInvalid,
}

fn valid_token(rng: &mut Rng, visible: usize) -> i32 {
    rng.below(visible as u64) as i32
}

/// kind 0: negative, 1: in [visible, full_cap), 2: >= full_cap.
fn invalid_token(rng: &mut Rng, kind: usize, visible: usize, full_cap: usize) -> i32 {
    match kind % 3 {
        0 => match rng.below(4) {
            0 => -1,
            1 => i32::MIN,
            2 => -1 - rng.below(100_000) as i32,
            _ => -2,
        },
        1 => visible as i32 + rng.below((full_cap - visible) as u64) as i32,
        _ => match rng.below(3) {
            0 => i32::MAX,
            1 => full_cap as i32,
            _ => full_cap as i32 + rng.below(1000) as i32,
        },
    }
}

/// The kernel's active selected length of the row with `visible` keys.
fn selected_len(visible: usize, budget: usize, compress: usize, capacity: usize) -> usize {
    let block_count = visible / compress;
    let chosen = budget.min(block_count);
    (chosen * compress + visible - block_count * compress).min(capacity)
}

/// One `capacity`-wide selected row. Slots at or past `sel_len` hold valid-looking
/// tokens (reading them would change the result) unless the mode is AllInvalid.
fn gen_row(
    rng: &mut Rng,
    mode: Mode,
    visible: usize,
    sel_len: usize,
    capacity: usize,
    full_cap: usize,
) -> Vec<i32> {
    let mut row: Vec<i32> = (0..capacity)
        .map(|i| match mode {
            Mode::AllInvalid => {
                let kind = rng.below(3) as usize;
                invalid_token(rng, kind, visible, full_cap)
            }
            Mode::AllValid => valid_token(rng, visible),
            Mode::Mixed => {
                if i >= sel_len {
                    return valid_token(rng, visible);
                }
                match rng.below(10) {
                    0 => invalid_token(rng, 0, visible, full_cap),
                    1 => invalid_token(rng, 1, visible, full_cap),
                    2 => invalid_token(rng, 2, visible, full_cap),
                    _ => valid_token(rng, visible),
                }
            }
        })
        .collect();
    if mode == Mode::Mixed && sel_len > 0 {
        // One guaranteed valid slot and, as far as the length permits, one
        // guaranteed invalid token of each kind.
        let k = rng.below(sel_len as u64) as usize;
        row[k] = valid_token(rng, visible);
        for t in 1..=3.min(sel_len - 1) {
            row[(k + t) % sel_len] = invalid_token(rng, t - 1, visible, full_cap);
        }
    }
    row
}

fn valid_count(row: &[i32], sel_len: usize, visible: usize, full_cap: usize) -> usize {
    row.iter()
        .take(sel_len)
        .filter(|&&t| t >= 0 && (t as usize) < visible && (t as usize) < full_cap)
        .count()
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

struct Dev {
    format: QsaKvFormat,
    /// Cache rows (`full_capacity`).
    full_cap: usize,
    keys: GpuTensor,
    values: GpuTensor,
}

/// Q8 K/V of `full_cap` rows (`kv_row_bytes` = 544 per token and tensor).
fn make_dev_q8(gpu: &mut Gpu, full_cap: usize, seed: u64) -> Dev {
    let row_bytes = QsaKvFormat::Q8.kv_row_bytes(NKV, HD);
    assert_eq!(row_bytes, NKV * (HD / 32) * 34);
    assert_eq!(row_bytes, 544);
    let mut rng = Rng(seed);
    let mut tensors = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut bytes = Vec::with_capacity(full_cap * row_bytes);
        for _ in 0..full_cap {
            q8_row(&mut rng, &mut bytes);
        }
        assert_eq!(bytes.len(), full_cap * row_bytes);
        tensors.push(
            gpu.upload_raw(&bytes, &[bytes.len()])
                .expect("upload q8 kv"),
        );
    }
    let values = tensors.pop().unwrap();
    let keys = tensors.pop().unwrap();
    Dev {
        format: QsaKvFormat::Q8,
        full_cap,
        keys,
        values,
    }
}

fn sentinel(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let data = vec![f32::from_bits(SENTINEL); n];
    gpu.upload_f32(&data, &[n]).expect("upload sentinel")
}

fn upload_q(gpu: &mut Gpu, q: &[f32]) -> GpuTensor {
    gpu.upload_f32(q, &[q.len()]).expect("upload q")
}

fn upload_selected(gpu: &Gpu, rows: &[Vec<i32>]) -> GpuTensor {
    let bytes: Vec<u8> = rows
        .iter()
        .flatten()
        .flat_map(|t| t.to_le_bytes())
        .collect();
    gpu.upload_raw(&bytes, &[bytes.len()])
        .expect("upload selected")
}

fn download(gpu: &Gpu, t: &GpuTensor) -> Vec<f32> {
    gpu.download_f32(t).expect("download")
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    rows: usize,
    position_start: usize,
    budget: usize,
    compress: usize,
    capacity: usize,
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    /// `indexed_attention_attention_batch_exact`: the per-head kernels.
    Exact,
    /// `indexed_attention_attention_splitkv_q8_for_test(splits)`.
    Split(usize),
}

fn launch(
    gpu: &mut Gpu,
    kind: Kind,
    dev: &Dev,
    q: &GpuTensor,
    selected: &GpuTensor,
    out: &GpuTensor,
    sh: &Shape,
) {
    let p = IndexedAttentionAttentionBatch {
        q_with_gate: q,
        full_keys: &dev.keys,
        full_values: &dev.values,
        selected,
        output: out,
        rows: sh.rows,
        position_start: sh.position_start,
        n_heads: NH,
        n_kv_heads: NKV,
        head_dim: HD,
        budget_blocks: sh.budget,
        compress: sh.compress,
        capacity: sh.capacity,
        full_capacity: dev.full_cap,
        format: dev.format,
        shape_selected: sh.capacity,
    };
    let result = match kind {
        Kind::Exact => indexed_attention_attention_batch_exact(gpu, &p),
        Kind::Split(splits) => indexed_attention_attention_splitkv_q8_for_test(gpu, &p, splits),
    };
    if let Err(e) = result {
        panic!("{kind:?} {} failed for {sh:?}: {e:?}", dev.format.name());
    }
}

/// Every split count under test; the production default must be among them.
fn splits() -> Vec<usize> {
    assert!(
        SPLITS.contains(&QSA_DECODE_SPLITK_SPLITS),
        "the production split count {QSA_DECODE_SPLITK_SPLITS} must be under test"
    );
    SPLITS.to_vec()
}

/// Visible lengths: around the 64-key tile boundary and the 32-split tile
/// boundary (2048 = 32 tiles), the long timing contexts and 8195, unioned with
/// {S - 1, S} of every split count S > 1 (so every visible length stays well
/// inside FULL_CAP).
fn lengths(splits: &[usize]) -> Vec<usize> {
    let mut lens = vec![1, 63, 64, 65, 128, 129, 1131, 2048, 2049, 2051, 2113, 8195];
    for &s in splits {
        if s > 1 {
            lens.push(s - 1);
        }
        lens.push(s);
    }
    lens.sort_unstable();
    lens.dedup();
    lens
}

// ---------------------------------------------------------------------------
// Comparison against the exact reference
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Worst {
    diff: f64,
    tol: f64,
    ratio: f64,
}

impl Worst {
    fn update(&mut self, diff: f64, tol: f64) {
        let ratio = diff / tol;
        if ratio >= self.ratio {
            *self = Worst { diff, tol, ratio };
        }
    }

    fn merge(&mut self, other: Worst) {
        if other.ratio >= self.ratio {
            *self = other;
        }
    }
}

/// Aggregate counters of one test.
#[derive(Default)]
struct Counts {
    /// `check` calls (one exact launch plus every split count each).
    fixtures: usize,
    /// Rows launched / rows with a valid token / rows without one.
    rows: usize,
    valid_rows: usize,
    zero_rows: usize,
    /// Split launches, valid-row heads compared against the reference, rows
    /// that must be +0, and rows whose selected_len is 2051.
    split_launches: usize,
    heads: usize,
    zero_rows_checked: usize,
    rows_sel2051: usize,
    /// Mismatches.
    head_mismatches: usize,
    nonfinite_heads: usize,
    zero_row_mismatches: usize,
    reference_defects: usize,
    selected_modified: usize,
    /// Batched-vs-single bitwise comparisons and their mismatches.
    bitwise_rows: usize,
    bitwise_mismatches: usize,
}

struct Stats {
    fails: Vec<String>,
    worst: Vec<Worst>,
    counts: Counts,
}

impl Stats {
    fn new(combos: usize) -> Self {
        Stats {
            fails: Vec::new(),
            worst: vec![Worst::default(); combos],
            counts: Counts::default(),
        }
    }

    fn fail(&mut self, msg: String) {
        self.fails.push(msg);
    }

    fn finish(&self, title: &str, splits: &[usize]) {
        let c = &self.counts;
        println!("=== {title}: aggregate ===");
        println!(
            "fixtures={} rows={} (valid={} all-invalid={} selected_len=2051: {}) split launches={}",
            c.fixtures, c.rows, c.valid_rows, c.zero_rows, c.rows_sel2051, c.split_launches
        );
        println!(
            "heads compared={} zero rows checked={} batched-vs-single rows={}",
            c.heads, c.zero_rows_checked, c.bitwise_rows
        );
        println!(
            "mismatches: heads={} (non-finite={}) zero-rows={} reference-defects={} \
             selected-byte-modifications={} batched-vs-single={}; failures={}",
            c.head_mismatches,
            c.nonfinite_heads,
            c.zero_row_mismatches,
            c.reference_defects,
            c.selected_modified,
            c.bitwise_mismatches,
            self.fails.len()
        );
        let mut overall = Worst::default();
        println!("--- worst per-head maxdiff / tolerance per split count ---");
        for (c, &s) in splits.iter().enumerate() {
            let w = self.worst[c];
            overall.merge(w);
            println!(
                "splits={s:>2} maxdiff={:.3e} tol={:.3e} ratio={:.3}",
                w.diff, w.tol, w.ratio
            );
        }
        println!(
            "worst ratio overall={:.3} (maxdiff={:.3e} tol={:.3e})",
            overall.ratio, overall.diff, overall.tol
        );
        if !self.fails.is_empty() {
            let shown: Vec<&str> = self.fails.iter().take(40).map(String::as_str).collect();
            panic!(
                "{title}: {} failure(s); first {}:\n{}",
                self.fails.len(),
                shown.len(),
                shown.join("\n")
            );
        }
    }
}

/// Launches the exact reference and every split count on one `Shape` (rows
/// `sel.len()`), compares them, and returns each split count's output. A row
/// whose selection holds no valid token inside its selected length must be
/// exactly +0 in the reference and in every split count; any other row is
/// compared per head with the tolerance `1e-5 * maxabsref + 1e-7`, and the
/// reference must be nonzero there. Every output must be finite. The selected
/// input bytes must come back unchanged after all launches.
#[allow(clippy::too_many_arguments)]
fn check(
    gpu: &mut Gpu,
    dev: &Dev,
    sh: &Shape,
    q: &[f32],
    sel: &[Vec<i32>],
    splits: &[usize],
    label: &str,
    stats: &mut Stats,
) -> Vec<Vec<f32>> {
    assert_eq!(sel.len(), sh.rows);
    assert_eq!(q.len(), sh.rows * 2 * NH * HD);
    let q_t = upload_q(gpu, q);
    let sel_t = upload_selected(gpu, sel);
    let n_out = sh.rows * NH * HD;
    let sel_lens: Vec<usize> = (0..sh.rows)
        .map(|r| {
            selected_len(
                sh.position_start + r + 1,
                sh.budget,
                sh.compress,
                sh.capacity,
            )
        })
        .collect();
    let valid: Vec<bool> = (0..sh.rows)
        .map(|r| {
            let visible = sh.position_start + r + 1;
            valid_count(&sel[r], sel_lens[r], visible, dev.full_cap) > 0
        })
        .collect();
    stats.counts.fixtures += 1;
    stats.counts.rows += sh.rows;
    stats.counts.valid_rows += valid.iter().filter(|&&v| v).count();
    stats.counts.zero_rows += valid.iter().filter(|&&v| !v).count();
    stats.counts.rows_sel2051 += sel_lens.iter().filter(|&&l| l == 2051).count();

    let out_ref = sentinel(gpu, n_out);
    launch(gpu, Kind::Exact, dev, &q_t, &sel_t, &out_ref, sh);
    let want = download(gpu, &out_ref);
    let row_len = NH * HD;
    for r in 0..sh.rows {
        let row = &want[r * row_len..(r + 1) * row_len];
        if row.iter().any(|x| !x.is_finite()) {
            stats.counts.reference_defects += 1;
            stats.fail(format!(
                "{label}: exact reference row {r} not finite/unwritten"
            ));
        } else if valid[r] && row.iter().all(|&x| x == 0.0) {
            stats.counts.reference_defects += 1;
            stats.fail(format!(
                "{label}: exact reference row {r} is all zero despite valid tokens"
            ));
        } else if !valid[r] && row.iter().any(|x| x.to_bits() != 0) {
            stats.counts.reference_defects += 1;
            stats.fail(format!(
                "{label}: exact reference row {r} is not +0 without valid tokens"
            ));
        }
    }

    let mut outs = Vec::with_capacity(splits.len());
    let mut fixture = Worst::default();
    for (c, &s) in splits.iter().enumerate() {
        let out = sentinel(gpu, n_out);
        launch(gpu, Kind::Split(s), dev, &q_t, &sel_t, &out, sh);
        stats.counts.split_launches += 1;
        let got = download(gpu, &out);
        let mut worst = Worst::default();
        for r in 0..sh.rows {
            if !valid[r] {
                stats.counts.zero_rows_checked += 1;
                if let Some(i) = (r * row_len..(r + 1) * row_len).find(|&i| got[i].to_bits() != 0) {
                    stats.counts.zero_row_mismatches += 1;
                    stats.fail(format!(
                        "{label} splits={s}: row {r} has no valid token, expected +0, got bits {:#010x} at {}",
                        got[i].to_bits(),
                        i - r * row_len
                    ));
                }
                continue;
            }
            for h in 0..NH {
                let lo = (r * NH + h) * HD;
                let (g, w) = (&got[lo..lo + HD], &want[lo..lo + HD]);
                let maxabs = w.iter().fold(0.0f64, |m, &x| m.max((x as f64).abs()));
                let tol = 1e-5 * maxabs + 1e-7;
                let mut diff = 0.0f64;
                let nonfinite = g.iter().any(|x| !x.is_finite());
                let mut bad = nonfinite;
                for (a, b) in g.iter().zip(w) {
                    diff = diff.max((*a as f64 - *b as f64).abs());
                }
                if !(diff <= tol) {
                    bad = true;
                }
                stats.counts.heads += 1;
                if nonfinite {
                    stats.counts.nonfinite_heads += 1;
                }
                if bad {
                    stats.counts.head_mismatches += 1;
                    stats.fail(format!(
                        "{label} splits={s}: row {r} head {h} maxdiff={diff:.3e} > tol={tol:.3e} (maxabsref={maxabs:.3e})"
                    ));
                }
                worst.update(diff, tol);
            }
        }
        stats.worst[c].merge(worst);
        fixture.merge(worst);
        outs.push(got);
    }
    let selected_after = gpu.download_raw_bytes(&sel_t).expect("download selected");
    let selected_before: Vec<u8> = sel.iter().flatten().flat_map(|t| t.to_le_bytes()).collect();
    if selected_after != selected_before {
        stats.counts.selected_modified += 1;
        stats.fail(format!(
            "{label}: a launch modified the selected input bytes"
        ));
    }
    println!(
        "{label}: valid rows {}/{} worst maxdiff={:.3e} tol={:.3e} ratio={:.3}",
        valid.iter().filter(|&&v| v).count(),
        sh.rows,
        fixture.diff,
        fixture.tol,
        fixture.ratio
    );
    outs
}

// ---------------------------------------------------------------------------
// Single-row configurations
// ---------------------------------------------------------------------------

const ROWS1_CFGS: [&str; 4] = [
    "c1/huge/cap=len",
    "c8/huge/cap=len+5",
    "c8/budget=blocks-1",
    "c8/budget=blocks/3/clamped-cap",
];

/// (compress, budget, capacity) of single-row configuration `cfg` at `len` visible keys.
fn rows1_cfg(cfg: usize, len: usize) -> (usize, usize, usize) {
    let blocks = len / 8;
    match cfg {
        0 => (1, HUGE, len),
        1 => (8, HUGE, len + 5),
        2 => (8, blocks.saturating_sub(1), len),
        _ => {
            let budget = blocks / 3;
            let unclamped = selected_len(len, budget, 8, usize::MAX);
            (8, budget, unclamped.saturating_sub(1).max(1))
        }
    }
}

fn rows1_fixture(
    gpu: &mut Gpu,
    dev: &Dev,
    splits: &[usize],
    len: usize,
    cfg: usize,
    mode: Mode,
    stats: &mut Stats,
) {
    let (compress, budget, capacity) = rows1_cfg(cfg, len);
    let sh = Shape {
        rows: 1,
        position_start: len - 1,
        budget,
        compress,
        capacity,
    };
    let mut rng = Rng(0x1000_0000 + (len * 31 + cfg * 7) as u64 * 977 + mode as u64);
    let q = make_q(&mut rng, 1);
    let sel_len = selected_len(len, budget, compress, capacity);
    let row = gen_row(&mut rng, mode, len, sel_len, capacity, dev.full_cap);
    let label = format!(
        "{} rows1 len={len:<4} {:<31} {mode:?} sel_len={sel_len}",
        dev.format.name(),
        ROWS1_CFGS[cfg]
    );
    check(gpu, dev, &sh, &q, &[row], splits, &label, stats);
}

fn rows1_sweep(gpu: &mut Gpu, dev: &Dev, title: &str) {
    let splits = splits();
    let mut stats = Stats::new(splits.len());
    for len in lengths(&splits) {
        for cfg in 0..ROWS1_CFGS.len() {
            for mode in [Mode::AllValid, Mode::Mixed] {
                rows1_fixture(gpu, dev, &splits, len, cfg, mode, &mut stats);
            }
        }
    }
    stats.finish(title, &splits);
}

#[test]
#[ignore = "needs a HIP device"]
fn qsa_splitk_q8_rows1_matches_exact() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!(
        "arch={} default splits={}",
        gpu.arch, QSA_DECODE_SPLITK_SPLITS
    );
    let dev = make_dev_q8(&mut gpu, FULL_CAP, 0x5a5a_0008);
    rows1_sweep(&mut gpu, &dev, "q8 rows1 matches exact (8K cache)");
}

// ---------------------------------------------------------------------------
// Batched configurations
// ---------------------------------------------------------------------------

struct BatchCfg {
    name: &'static str,
    compress: usize,
    budget: usize,
    position_start: usize,
    capacity: usize,
}

/// Eight-row windows (`rows` 1..8 take the prefix) with heterogeneous
/// `selected_len`: S > len, budget at the block boundary (selected_len drops when a
/// block completes), empty slices, capacity clamp, and the long contexts.
const BATCH_CFGS: [BatchCfg; 9] = [
    BatchCfg {
        name: "c8 b2 ps14 cap64",
        compress: 8,
        budget: 2,
        position_start: 14,
        capacity: 64,
    },
    BatchCfg {
        name: "c8 b2 ps20 cap64 (drop at block 3)",
        compress: 8,
        budget: 2,
        position_start: 20,
        capacity: 64,
    },
    BatchCfg {
        name: "c8 huge ps0 cap16 (S>len)",
        compress: 8,
        budget: HUGE,
        position_start: 0,
        capacity: 16,
    },
    BatchCfg {
        name: "c8 b0 ps0 cap16 (empty slice)",
        compress: 8,
        budget: 0,
        position_start: 0,
        capacity: 16,
    },
    BatchCfg {
        name: "c8 b0 ps6 cap16 (empty slice)",
        compress: 8,
        budget: 0,
        position_start: 6,
        capacity: 16,
    },
    BatchCfg {
        name: "c1 b5 ps0 cap8",
        compress: 1,
        budget: 5,
        position_start: 0,
        capacity: 8,
    },
    BatchCfg {
        name: "c8 b140 ps1124 cap1200",
        compress: 8,
        budget: 140,
        position_start: 1124,
        capacity: 1200,
    },
    BatchCfg {
        name: "c8 huge ps8188 cap8200",
        compress: 8,
        budget: HUGE,
        position_start: 8188,
        capacity: 8200,
    },
    BatchCfg {
        name: "c8 huge ps20 cap24 (clamp)",
        compress: 8,
        budget: HUGE,
        position_start: 20,
        capacity: 24,
    },
];

/// Row `r` mode: row 1 is all invalid (so every batch of 2+ rows has one),
/// every third row from 2 is pure valid, the rest mixed.
fn row_mode(r: usize, force: Option<Mode>) -> Mode {
    if let Some(m) = force {
        return m;
    }
    match r {
        1 => Mode::AllInvalid,
        _ if r % 3 == 2 => Mode::AllValid,
        _ => Mode::Mixed,
    }
}

/// Host data of a config's first `rows` rows (eight in the direct diagnostic).
fn batch_host(
    cfg: &BatchCfg,
    force: Option<Mode>,
    full_cap: usize,
    rows: usize,
) -> (Vec<f32>, Vec<Vec<i32>>, Vec<usize>) {
    assert!(
        cfg.position_start + rows < full_cap,
        "{}: {rows} rows must leave an in-bounds invalid token range below {full_cap}",
        cfg.name
    );
    let mut rng = Rng(0x2000_0000
        + (cfg.position_start * 13 + cfg.compress) as u64 * 7919
        + cfg.budget as u64 % 1009);
    let q = make_q(&mut rng, rows);
    let mut sel = Vec::new();
    let mut lens = Vec::new();
    for r in 0..rows {
        let visible = cfg.position_start + r + 1;
        let sel_len = selected_len(visible, cfg.budget, cfg.compress, cfg.capacity);
        lens.push(sel_len);
        sel.push(gen_row(
            &mut rng,
            row_mode(r, force),
            visible,
            sel_len,
            cfg.capacity,
            full_cap,
        ));
    }
    (q, sel, lens)
}

fn batch_shape(cfg: &BatchCfg, rows: usize) -> Shape {
    Shape {
        rows,
        position_start: cfg.position_start,
        budget: cfg.budget,
        compress: cfg.compress,
        capacity: cfg.capacity,
    }
}

/// Batched launches of `row_counts` rows (each at most `host_rows`) against
/// the single-row launches of the same rows.
fn batched_bitwise_sweep(
    gpu: &mut Gpu,
    dev: &Dev,
    title: &str,
    host_rows: usize,
    row_counts: &[usize],
) {
    assert!(row_counts.iter().all(|&n| (1..=host_rows).contains(&n)));
    let splits = splits();
    let mut stats = Stats::new(splits.len());
    for cfg in &BATCH_CFGS {
        let (q, sel, lens) = batch_host(cfg, None, dev.full_cap, host_rows);
        println!(
            "== {} {} selected_len per row {lens:?}",
            dev.format.name(),
            cfg.name
        );
        assert!(
            lens.iter().any(|&l| l != lens[0]),
            "{}: selected_len must be heterogeneous: {lens:?}",
            cfg.name
        );
        // The single-row tensors of all `host_rows` rows.
        let singles: Vec<(GpuTensor, GpuTensor)> = (0..host_rows)
            .map(|r| {
                let q_row = &q[r * 2 * NH * HD..(r + 1) * 2 * NH * HD];
                (upload_q(gpu, q_row), upload_selected(gpu, &sel[r..r + 1]))
            })
            .collect();
        for &rows in row_counts {
            let sh = batch_shape(cfg, rows);
            let label = format!("{} batched {} rows={rows}", dev.format.name(), cfg.name);
            let outs = check(
                gpu,
                dev,
                &sh,
                &q[..rows * 2 * NH * HD],
                &sel[..rows],
                &splits,
                &label,
                &mut stats,
            );
            let row_len = NH * HD;
            for (c, &s) in splits.iter().enumerate() {
                for r in 0..rows {
                    let single = Shape {
                        rows: 1,
                        position_start: cfg.position_start + r,
                        ..sh
                    };
                    let out = sentinel(gpu, row_len);
                    launch(
                        gpu,
                        Kind::Split(s),
                        dev,
                        &singles[r].0,
                        &singles[r].1,
                        &out,
                        &single,
                    );
                    let alone = download(gpu, &out);
                    let batched = &outs[c][r * row_len..(r + 1) * row_len];
                    stats.counts.bitwise_rows += 1;
                    if alone.iter().any(|x| !x.is_finite()) {
                        stats.counts.bitwise_mismatches += 1;
                        stats.fail(format!(
                            "{label} splits={s}: single-row launch {r} not finite"
                        ));
                    }
                    if let Some(i) =
                        (0..row_len).find(|&i| alone[i].to_bits() != batched[i].to_bits())
                    {
                        stats.counts.bitwise_mismatches += 1;
                        stats.fail(format!(
                            "{label} splits={s}: row {r} differs from its single-row launch at {i}: batched {:#010x} single {:#010x}",
                            batched[i].to_bits(),
                            alone[i].to_bits()
                        ));
                    }
                }
            }
        }
    }
    stats.finish(title, &splits);
}

fn all_invalid_sweep(gpu: &mut Gpu, dev: &Dev, title: &str) {
    let splits = splits();
    let mut stats = Stats::new(splits.len());
    for len in lengths(&splits) {
        for cfg in 0..ROWS1_CFGS.len() {
            rows1_fixture(gpu, dev, &splits, len, cfg, Mode::AllInvalid, &mut stats);
        }
    }
    for cfg in &BATCH_CFGS {
        let (q, sel, _) = batch_host(cfg, Some(Mode::AllInvalid), dev.full_cap, 8);
        for rows in 1..=8usize {
            let sh = batch_shape(cfg, rows);
            let label = format!(
                "{} all-invalid batched {} rows={rows}",
                dev.format.name(),
                cfg.name
            );
            check(
                gpu,
                dev,
                &sh,
                &q[..rows * 2 * NH * HD],
                &sel[..rows],
                &splits,
                &label,
                &mut stats,
            );
        }
    }
    stats.finish(title, &splits);
}

#[test]
#[ignore = "needs a HIP device"]
fn qsa_splitk_q8_batched_rows_bitwise_individual() {
    let mut gpu = Gpu::init().expect("gpu init");
    let dev = make_dev_q8(&mut gpu, FULL_CAP, 0x5a5a_0008);
    let rows: Vec<usize> = (1..=8).collect();
    batched_bitwise_sweep(
        &mut gpu,
        &dev,
        "q8 batched rows vs single-row launches (8K cache)",
        8,
        &rows,
    );
}

fn assert_raw_equal(got: &[u8], want: &[u8], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: byte extent");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "{label}: first mismatch byte={i} row={} head={} word={} got={:#04x} want={:#04x}",
            i / (NH * HD * 4),
            (i / (HD * 4)) % NH,
            (i / 4) % HD,
            got[i],
            want[i],
        );
    }
}

fn assert_poison(bytes: &[u8], label: &str) {
    assert_eq!(bytes.len() % 4, 0);
    let poison = SENTINEL.to_le_bytes();
    if let Some(i) = bytes.chunks_exact(4).position(|word| word != poison) {
        panic!("{label}: guard/inactive word {i} overwritten");
    }
}

#[allow(clippy::too_many_arguments)]
fn strict_dispatch(
    gpu: &mut Gpu,
    dev: &Dev,
    cfg: &BatchCfg,
    q: &GpuTensor,
    selected: &GpuTensor,
    out: &GpuTensor,
    rows: usize,
    position_start: usize,
    reference: Qwen4VerifyReference,
    split_enabled: bool,
    label: &str,
) -> QsaDecodeAttentionPolicy {
    let p = IndexedAttentionAttentionBatch {
        q_with_gate: q,
        full_keys: &dev.keys,
        full_values: &dev.values,
        selected,
        output: out,
        rows,
        position_start,
        n_heads: NH,
        n_kv_heads: NKV,
        head_dim: HD,
        budget_blocks: cfg.budget,
        compress: cfg.compress,
        capacity: cfg.capacity,
        full_capacity: dev.full_cap,
        format: dev.format,
        shape_selected: cfg.capacity,
    };
    assert_eq!(
        gpu.qwen4_verify_reference(),
        Some(reference),
        "{label}: reference scope"
    );
    let expected =
        if reference == Qwen4VerifyReference::Eager && split_enabled && gpu.arch == "gfx1151" {
            QsaDecodeAttentionPolicy::SplitKvQ8 {
                splits: QSA_DECODE_SPLITK_SPLITS,
            }
        } else {
            QsaDecodeAttentionPolicy::PerHead
        };
    for row in 0..rows {
        assert_eq!(
            qsa_decode_attention_policy_for_row_for_test(gpu, &p, row),
            Some(expected),
            "{label}: row {row} refused or policy mismatch",
        );
    }
    assert_eq!(
        qsa_decode_attention_policy_for_row_for_test(gpu, &p, rows),
        None
    );
    indexed_attention_attention_batch(gpu, &p)
        .unwrap_or_else(|e| panic!("{label}: production candidate refused: {e:?}"));
    let expected_terminal = if matches!(expected, QsaDecodeAttentionPolicy::SplitKvQ8 { .. }) {
        "qsa_decode_splitkv_merge_f32"
    } else {
        let bound = cfg.budget * cfg.compress + cfg.compress - 1;
        let selected_extent = if reference == Qwen4VerifyReference::Recorded {
            cfg.capacity
        } else {
            (position_start + rows).min(cfg.capacity).min(bound)
        };
        let parallel = gpu.arch_caps.has_qwen4_parallel_qsa_gdn()
            && selected_extent > 0
            && selected_extent * 8 + 32 <= 64 * 1024;
        if parallel {
            "indexed_attention_attention_q8_batched"
        } else {
            "indexed_attention_attention_q8_batched_serial"
        }
    };
    assert_eq!(
        gpu.last_launched_kernel(),
        Some(expected_terminal),
        "{label}: actual terminal policy/LDS subroute",
    );
    expected
}

#[allow(clippy::too_many_arguments)]
fn strict_production_fixture(
    gpu: &mut Gpu,
    dev: &Dev,
    cfg: &BatchCfg,
    seed: u64,
    reference: Qwen4VerifyReference,
    split_enabled: bool,
) {
    const ROWS: usize = 64;
    const GUARD: usize = 64;
    const Q_ROW: usize = 2 * NH * HD;
    const O_ROW: usize = NH * HD;
    let label = format!(
        "arch={} seed={seed:#x} mode={reference:?} split={split_enabled} {} ps={} c={} budget={} cap={}",
        gpu.arch, cfg.name, cfg.position_start, cfg.compress, cfg.budget, cfg.capacity,
    );
    assert!(cfg.position_start > 0);
    assert!(cfg.position_start + ROWS < dev.full_cap, "{label}");
    let mut rng = Rng(seed ^ ((cfg.position_start as u64) << 16) ^ cfg.budget as u64);
    let q_host = make_q(&mut rng, ROWS);
    let selected_host: Vec<Vec<i32>> = (0..ROWS)
        .map(|r| {
            let visible = cfg.position_start + r + 1;
            gen_row(
                &mut rng,
                row_mode(r, None),
                visible,
                selected_len(visible, cfg.budget, cfg.compress, cfg.capacity),
                cfg.capacity,
                dev.full_cap,
            )
        })
        .collect();
    let q = upload_q(gpu, &q_host);
    let selected = upload_selected(gpu, &selected_host);
    let q_before = gpu.download_raw_bytes(&q).expect("query S0 bytes");
    let selected_before = gpu
        .download_raw_bytes(&selected)
        .expect("selection S0 bytes");
    let keys_before = gpu.download_raw_bytes(&dev.keys).expect("K S0 bytes");
    let values_before = gpu.download_raw_bytes(&dev.values).expect("V S0 bytes");

    // A persistent S0 is shared by the sequential reference calls. Attention is
    // read-only: only visibility/position advances; no append pipeline is claimed.
    let oracle_owner = sentinel(gpu, 2 * GUARD + (ROWS + 1) * O_ROW);
    for r in 0..ROWS {
        let qr = q.sub_offset(r * Q_ROW, Q_ROW);
        // Raw selected tensors are byte-addressed (each i32 occupies four).
        let sr = selected.sub_offset(r * cfg.capacity * 4, cfg.capacity * 4);
        let out = oracle_owner.sub_offset(GUARD + r * O_ROW, O_ROW);
        strict_dispatch(
            gpu,
            dev,
            cfg,
            &qr,
            &sr,
            &out,
            1,
            cfg.position_start + r,
            reference,
            split_enabled,
            &format!("{label} singleton row={r}"),
        );
        let bytes = gpu
            .download_raw_bytes(&oracle_owner)
            .expect("singleton output bytes");
        assert_poison(&bytes[..GUARD * 4], &label);
        assert_poison(&bytes[(GUARD + (r + 1) * O_ROW) * 4..], &label);
        assert!(
            bytes[(GUARD + r * O_ROW) * 4..(GUARD + (r + 1) * O_ROW) * 4]
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "{label}: singleton row={r} nonfinite/unwritten",
        );
    }
    let oracle = gpu
        .download_raw_bytes(&oracle_owner)
        .expect("sequential oracle bytes");
    assert_raw_equal(
        &gpu.download_raw_bytes(&dev.keys)
            .expect("singleton K final"),
        &keys_before,
        &format!("{label} singleton immutable K"),
    );
    assert_raw_equal(
        &gpu.download_raw_bytes(&dev.values)
            .expect("singleton V final"),
        &values_before,
        &format!("{label} singleton immutable V"),
    );
    assert_raw_equal(
        &gpu.download_raw_bytes(&q).expect("singleton query final"),
        &q_before,
        &format!("{label} singleton immutable query"),
    );
    assert_raw_equal(
        &gpu.download_raw_bytes(&selected)
            .expect("singleton selection final"),
        &selected_before,
        &format!("{label} singleton immutable selections"),
    );
    // 8188 entries fit the 64KiB per-head LDS budget (8 bytes/entry + 32).
    // Q8 split-KV itself has fixed LDS. The split-disabled Eager arm crosses
    // parallel/serial within a window; Recorded preserves shape_selected instead.
    if cfg.name == "lds-boundary" {
        let first = (cfg.position_start + 1).min(cfg.capacity);
        let last = (cfg.position_start + ROWS).min(cfg.capacity);
        assert!(
            first <= 8192 && last > 8188,
            "{label}: LDS context coverage"
        );
        if cfg.position_start < 8188 {
            assert!(first * 8 + 32 <= 65536 && last * 8 + 32 > 65536);
        }
    }
    for rows in 1..=ROWS {
        let candidate_label = format!("{label} R={rows} output_bytes={}", rows * O_ROW * 4);
        let owner = sentinel(gpu, 2 * GUARD + (ROWS + 1) * O_ROW);
        let out = owner.sub_offset(GUARD, ROWS * O_ROW);
        let policy = strict_dispatch(
            gpu,
            dev,
            cfg,
            &q,
            &selected,
            &out,
            rows,
            cfg.position_start,
            reference,
            split_enabled,
            &candidate_label,
        );
        let terminal = gpu
            .last_launched_kernel()
            .expect("production dispatch must launch a kernel")
            .to_owned();
        let (module, container_sha, sha) = gpu
            .lab_kernel_device_identity(&terminal)
            .expect("strict G1 requires terminal launched-object provenance");
        let got = gpu.download_raw_bytes(&owner).expect("candidate bytes");
        assert_poison(&got[..GUARD * 4], &candidate_label);
        assert_poison(&got[(GUARD + rows * O_ROW) * 4..], &candidate_label);
        assert_raw_equal(
            &got[GUARD * 4..(GUARD + rows * O_ROW) * 4],
            &oracle[GUARD * 4..(GUARD + rows * O_ROW) * 4],
            &candidate_label,
        );
        for r in 0..rows {
            let visible = cfg.position_start + r + 1;
            let len = selected_len(visible, cfg.budget, cfg.compress, cfg.capacity);
            if valid_count(&selected_host[r], len, visible, dev.full_cap) == 0 {
                assert!(
                    got[(GUARD + r * O_ROW) * 4..(GUARD + (r + 1) * O_ROW) * 4]
                        .iter()
                        .all(|&b| b == 0),
                    "{candidate_label}: empty/all-invalid row {r} must be +0 bytes",
                );
            }
        }
        assert_raw_equal(
            &gpu.download_raw_bytes(&q).expect("candidate query final"),
            &q_before,
            &format!("{candidate_label} immutable query"),
        );
        assert_raw_equal(
            &gpu.download_raw_bytes(&selected)
                .expect("candidate selection final"),
            &selected_before,
            &format!("{candidate_label} immutable selections"),
        );
        println!(
            "{candidate_label} policy={policy:?} terminal={terminal} module={module} container_sha256={container_sha} object_sha256={sha} q_bytes={} selected_bytes={} kv_bytes_each={} shape_selected={} byte_equal=true",
            q_before.len(), selected_before.len(), keys_before.len(), cfg.capacity,
        );
        gpu.free_tensor(owner)
            .expect("free guarded candidate owner");
    }
    gpu.free_tensor(oracle_owner)
        .expect("free guarded oracle owner");
    gpu.free_tensor(q).expect("free query owner");
    gpu.free_tensor(selected).expect("free selected owner");
}

/// Strict G1 production-dispatch gate; the direct tolerance oracle is not used.
#[test]
#[ignore = "strict G1: needs a supported HIP device; refusal is fatal"]
fn qsa_splitk_q8_batched_rows_bitwise_individual_wide() {
    let mut gpu = Gpu::init().expect("strict G1 requires a working HIP device");
    assert!(
        matches!(gpu.arch.as_str(), "gfx1151" | "gfx1201"),
        "strict wide QSA admission unavailable on {}",
        gpu.arch
    );
    let prior_rows = gpu.set_wide_verify_rows(64);
    let prior_reference = gpu.qwen4_verify_reference();
    let prior_flags = gpu.flags.clone();
    // Restore host scope even if a candidate refusal/mismatch panics.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_eq!(gpu.few_row_limit(), 64, "strict candidate was not admitted");
        for seed in [0x5a5a_0040, 0xd3c0_de64] {
            let dev = make_dev_q8(&mut gpu, FULL_CAP, seed);
            let keys = gpu.download_raw_bytes(&dev.keys).expect("cache K S0");
            let values = gpu.download_raw_bytes(&dev.values).expect("cache V S0");
            assert!(keys.iter().any(|&b| b != 0) && values.iter().any(|&b| b != 0));
            for split_enabled in [true, false] {
                let mut flags = (*prior_flags).clone();
                flags.qwen4_qsa_decode_splitk = split_enabled;
                gpu.flags = std::sync::Arc::new(flags);
                for reference in [Qwen4VerifyReference::Eager, Qwen4VerifyReference::Recorded] {
                    gpu.set_qwen4_verify_reference(Some(reference));
                    for compress in [1, 4, 8] {
                        for residue in 0..compress {
                            // Nonzero initial prefix in every fixture; all compress residues.
                            let start = 16 * compress + residue;
                            for (name, position_start, budget, capacity) in [
                                ("empty-short", start, 0, 16),
                                ("short-budget", start, 1, 16),
                                ("capacity-max", start, HUGE, 64),
                                ("lds-boundary", 8184 + residue, HUGE, 8256),
                            ] {
                                let cfg = BatchCfg {
                                    name,
                                    compress,
                                    budget,
                                    position_start,
                                    capacity,
                                };
                                strict_production_fixture(
                                    &mut gpu,
                                    &dev,
                                    &cfg,
                                    seed,
                                    reference,
                                    split_enabled,
                                );
                                assert_raw_equal(
                                    &gpu.download_raw_bytes(&dev.keys).expect("cache K final"),
                                    &keys,
                                    "immutable K cache S0",
                                );
                                assert_raw_equal(
                                    &gpu.download_raw_bytes(&dev.values).expect("cache V final"),
                                    &values,
                                    "immutable V cache S0",
                                );
                            }
                        }
                    }
                }
            }
            gpu.free_tensor(dev.keys).expect("free cache K");
            gpu.free_tensor(dev.values).expect("free cache V");
        }
    }));
    gpu.flags = prior_flags;
    gpu.set_qwen4_verify_reference(prior_reference);
    gpu.set_wide_verify_rows(prior_rows);
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}

#[test]
#[ignore = "needs a HIP device"]
fn qsa_splitk_q8_all_invalid_zero() {
    let mut gpu = Gpu::init().expect("gpu init");
    let dev = make_dev_q8(&mut gpu, FULL_CAP, 0x5a5a_0008);
    all_invalid_sweep(
        &mut gpu,
        &dev,
        "q8 all-invalid selections are +0 (8K cache)",
    );
}

// ---------------------------------------------------------------------------
// Q8 selected tiles at 8K and 256K-like cache capacities
// ---------------------------------------------------------------------------

/// Eight-row windows run at rows 1, 2 and 8. Selected lengths that fill the last
/// 64-key tile only partially: 2051 (32 tiles + 3) by capacity clamp and by block
/// budget, 63..70 and 2045..2052 around tile boundaries, empty slices, and an 8K
/// tail; the 256K configs draw tokens from the whole 2^18 range. A config runs
/// on every cache with `position_start + 8 < full_cap`.
const Q8_TILE_CFGS: [BatchCfg; 11] = [
    BatchCfg {
        name: "c8 huge ps2050 cap2051 (clamp 2051)",
        compress: 8,
        budget: HUGE,
        position_start: 2050,
        capacity: 2051,
    },
    BatchCfg {
        name: "c8 b256 ps4098 cap2200 (2048+3 = 2051..2058)",
        compress: 8,
        budget: 256,
        position_start: 4098,
        capacity: 2200,
    },
    BatchCfg {
        name: "c1 huge ps2044 cap4096 (2045..2052)",
        compress: 1,
        budget: HUGE,
        position_start: 2044,
        capacity: 4096,
    },
    BatchCfg {
        name: "c1 huge ps62 cap64 (63..70 clamp 64)",
        compress: 1,
        budget: HUGE,
        position_start: 62,
        capacity: 64,
    },
    BatchCfg {
        name: "c8 b0 ps1000 cap16 (empty slice)",
        compress: 8,
        budget: 0,
        position_start: 1000,
        capacity: 16,
    },
    BatchCfg {
        name: "c8 huge ps8180 cap8200 (8K tail)",
        compress: 8,
        budget: HUGE,
        position_start: 8180,
        capacity: 8200,
    },
    BatchCfg {
        name: "c8 huge ps262000 cap2051 (256K clamp 2051)",
        compress: 8,
        budget: HUGE,
        position_start: 262000,
        capacity: 2051,
    },
    BatchCfg {
        name: "c8 b256 ps262098 cap4096 (256K 2048+3)",
        compress: 8,
        budget: 256,
        position_start: 262098,
        capacity: 4096,
    },
    BatchCfg {
        name: "c1 huge ps262130 cap2051 (256K c1)",
        compress: 1,
        budget: HUGE,
        position_start: 262130,
        capacity: 2051,
    },
    BatchCfg {
        name: "c8 b0 ps262000 cap16 (256K empty slice)",
        compress: 8,
        budget: 0,
        position_start: 262000,
        capacity: 16,
    },
    BatchCfg {
        name: "c1 b5 ps262000 cap8 (256K tiny)",
        compress: 1,
        budget: 5,
        position_start: 262000,
        capacity: 8,
    },
];

fn q8_selected_tiles(gpu: &mut Gpu, dev: &Dev, stats: &mut Stats) {
    let splits = splits();
    for cfg in &Q8_TILE_CFGS {
        if cfg.position_start + 8 >= dev.full_cap {
            continue;
        }
        for force in [
            None,
            Some(Mode::AllValid),
            Some(Mode::Mixed),
            Some(Mode::AllInvalid),
        ] {
            let (q, sel, lens) = batch_host(cfg, force, dev.full_cap, 8);
            for rows in [1usize, 2, 8] {
                let sh = batch_shape(cfg, rows);
                let label = format!(
                    "q8 cap={} {} {} rows={rows} sel_len={:?}",
                    dev.full_cap,
                    cfg.name,
                    force.map_or("RowMix".to_string(), |m| format!("{m:?}")),
                    &lens[..rows]
                );
                check(
                    gpu,
                    dev,
                    &sh,
                    &q[..rows * 2 * NH * HD],
                    &sel[..rows],
                    &splits,
                    &label,
                    stats,
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a HIP device"]
fn qsa_splitk_q8_selected_tiles_rows_1_2_8() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!(
        "arch={} default splits={}",
        gpu.arch, QSA_DECODE_SPLITK_SPLITS
    );
    let splits = splits();
    let mut stats = Stats::new(splits.len());
    let dev_8k = make_dev_q8(&mut gpu, FULL_CAP, 0x5a5a_0108);
    q8_selected_tiles(&mut gpu, &dev_8k, &mut stats);
    let after_8k = (stats.counts.fixtures, stats.counts.rows_sel2051);
    println!(
        "8K cache: fixtures={} rows with selected_len 2051={}",
        after_8k.0, after_8k.1
    );
    let dev_256k = make_dev_q8(&mut gpu, FULL_CAP_256K, 0x5a5a_0256);
    q8_selected_tiles(&mut gpu, &dev_256k, &mut stats);
    println!(
        "256K cache: fixtures={} rows with selected_len 2051={}",
        stats.counts.fixtures - after_8k.0,
        stats.counts.rows_sel2051 - after_8k.1
    );
    assert!(after_8k.1 > 0, "8K cache: no row had selected_len 2051");
    assert!(
        stats.counts.rows_sel2051 > after_8k.1,
        "256K cache: no row had selected_len 2051"
    );
    stats.finish(
        "q8 selected tiles rows 1/2/8 (8K and 256K-like caches)",
        &splits,
    );
}
