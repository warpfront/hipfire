// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact QSA filter selector (`indexed_attention_select_batch_filter_arm`,
//! `qsa_filter_select_from_scores_lab`) against the incumbent selector.
//!
//! * End to end (production score + select, F32 pooled keys, decode geometry
//!   `budget_blocks = 512`, `compress = 4`, `capacity = 2051`): the filter arm
//!   with the hipcc score (gfx1151 and gfx1201) and with the gfx1151 PM score
//!   must write the bytes of the hipcc incumbent
//!   (`indexed_attention_select_batch_pm_arm(.., Hipcc)`) for 65,536 blocks
//!   and the 4095/4096/4097 boundary, 1..=8 and 16 rows, every tail 0..3, over
//!   tie-rich keys, with and without a mirror, with canaries past `selected`
//!   and `mirror`. Repeats are bit-identical (selection and row stats).
//! * Forced underfill (`target = 1`) and overflow (`cap = 600`,
//!   `target = 4096`) must report every row repaired and still match exactly.
//! * From-scores: all-equal, zero, mostly-zero, `+inf` and ULP-boundary score
//!   rows at budgets 0/1/511/512 against a CPU oracle, the hipcc incumbent and
//!   (gfx1151) the PM incumbent.
//!
//! Every GPU test skips without a gfx1151/gfx1201 device. The `#[ignore]`d
//! benchmark times the select alone (hipEvent, warmed, per-call median of 50).
//!
//!   cargo test -p rdna-compute --release --features lab --test qsa_filter_select -- --nocapture
//!   cargo test -p rdna-compute --release --features lab --test qsa_filter_select -- --ignored --nocapture qsa_select_only_bench
#![cfg(feature = "lab")]

use hip_bridge::KernargBlob;
use rdna_compute::tensor_ops::{
    indexed_attention_select_batch_filter_arm, indexed_attention_select_batch_pm_arm,
    qsa_filter_select_from_scores_lab, qsa_incumbent_select_from_scores_lab,
    IndexedAttentionSelectBatch, QsaFilterKnobs, QsaFilterRowStat, QsaFromScores, QsaSelectPmArm,
};
use rdna_compute::{Gpu, GpuTensor};

const FILTER_SRC: &str = include_str!("../../../kernels/src/qsa_filter_select.hip");
const FILTER_MODULE: &str = "qsa_filter_select_bench";

const HEADS: usize = 4;
const DIM: usize = 128;
const COMPRESS: usize = 4;
const QUERY_STRIDE: usize = HEADS * DIM;
const BUDGET_BLOCKS: usize = 512;
const CAPACITY: usize = BUDGET_BLOCKS * COMPRESS + COMPRESS - 1;
const MAX_ROWS: usize = 16;
const POISON: u8 = 0xa5;
const CANARY: usize = 4096;

// ---------------------------------------------------------------------------
// helpers

fn gpu_or_skip() -> Option<Gpu> {
    let gpu = match Gpu::init() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("skip: no GPU ({e})");
            return None;
        }
    };
    if !(gpu.arch_caps.is_gfx1151() || gpu.arch_caps.is_gfx1201()) {
        eprintln!("skip: needs gfx1151 or gfx1201 (got {})", gpu.arch);
        return None;
    }
    Some(gpu)
}

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 16_777_216.0
    }
}

fn mix(x: usize) -> u32 {
    let mut h = (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= h >> 29;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    (h >> 32) as u32
}

fn poisoned(gpu: &Gpu, len: usize) -> GpuTensor {
    gpu.upload_raw(&vec![POISON; len], &[len])
        .expect("poisoned upload")
}

fn poison(gpu: &Gpu, t: &GpuTensor) {
    gpu.hip
        .memset(&t.buf, POISON as i32, t.byte_size())
        .expect("poison outputs");
}

fn stat_tuples(stats: &[QsaFilterRowStat]) -> Vec<(u64, u32, bool)> {
    stats
        .iter()
        .map(|s| (s.threshold, s.count, s.repaired))
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn all_poison(bytes: &[u8]) -> bool {
    bytes.iter().all(|b| *b == POISON)
}

fn underfill_knobs() -> QsaFilterKnobs {
    QsaFilterKnobs {
        target: 1,
        ..QsaFilterKnobs::default()
    }
}

fn overflow_knobs() -> QsaFilterKnobs {
    QsaFilterKnobs {
        cap: 600,
        target: 4096,
        ..QsaFilterKnobs::default()
    }
}

// ---------------------------------------------------------------------------
// end to end: production score + select

/// Pooled keys with exact score ties at the budget boundary: 20% of blocks are
/// two top-level constant blocks (10% each, so the 512th key sits inside a tie
/// run at 4096+ blocks), 20% copy two random blocks, 10% are all-negative
/// (ReLU zero), the rest random.
fn tie_rich_pooled(blocks: usize) -> Vec<f32> {
    let mut rng = Lcg(0x7d3b_91a5 ^ blocks as u32);
    let palette: Vec<Vec<f32>> = (0..4)
        .map(|k| match k {
            0 => vec![1.0; DIM],
            1 => vec![0.9; DIM],
            _ => (0..DIM).map(|_| rng.unit()).collect(),
        })
        .collect();
    let mut keys = vec![0f32; blocks * DIM];
    for block in 0..blocks {
        let h = mix(block);
        let row = &mut keys[block * DIM..(block + 1) * DIM];
        match h % 10 {
            0..=3 => row.copy_from_slice(&palette[((h >> 8) % 4) as usize]),
            4 => row.iter_mut().for_each(|v| *v = -rng.unit() - 0.01),
            _ => row.iter_mut().for_each(|v| *v = rng.unit() - 0.25),
        }
    }
    keys
}

struct Fixture {
    query: GpuTensor,
    selected: GpuTensor,
    mirror: GpuTensor,
}

impl Fixture {
    fn new(gpu: &mut Gpu) -> Self {
        let mut rng = Lcg(0x51ca_17b3);
        // Positive queries: an all-negative block scores 0.
        let query: Vec<f32> = (0..MAX_ROWS * QUERY_STRIDE).map(|_| rng.unit()).collect();
        Self {
            query: gpu
                .upload_f32(&query, &[query.len()])
                .expect("query upload"),
            selected: poisoned(gpu, MAX_ROWS * CAPACITY * 4 + CANARY),
            mirror: poisoned(gpu, CAPACITY * 4 + CANARY),
        }
    }

    fn params<'a>(
        &'a self,
        pooled: &'a GpuTensor,
        blocks: usize,
        rows: usize,
        position_start: usize,
    ) -> IndexedAttentionSelectBatch<'a> {
        IndexedAttentionSelectBatch {
            query: &self.query,
            pooled,
            pooled_bf16: None,
            pooled_bf16_valid_blocks: 0,
            selected: &self.selected,
            rows,
            query_row_stride: QUERY_STRIDE,
            block_count: blocks,
            index_heads: HEADS,
            index_dim: DIM,
            budget_blocks: BUDGET_BLOCKS,
            compress: COMPRESS,
            position_start,
            capacity: CAPACITY,
            shape_blocks: blocks,
        }
    }

    fn poison(&self, gpu: &Gpu) {
        poison(gpu, &self.selected);
        poison(gpu, &self.mirror);
    }

    fn download(&self, gpu: &Gpu, persisted: bool, stats: Vec<QsaFilterRowStat>) -> Out {
        Out {
            persisted,
            selected: gpu
                .download_raw_bytes(&self.selected)
                .expect("selected download"),
            mirror: gpu
                .download_raw_bytes(&self.mirror)
                .expect("mirror download"),
            stats,
        }
    }

    fn free(self, gpu: &mut Gpu) {
        for t in [self.query, self.selected, self.mirror] {
            gpu.free_tensor(t).expect("free");
        }
    }
}

struct Out {
    persisted: bool,
    selected: Vec<u8>,
    mirror: Vec<u8>,
    stats: Vec<QsaFilterRowStat>,
}

fn run_reference(
    gpu: &mut Gpu,
    fx: &Fixture,
    p: &IndexedAttentionSelectBatch<'_>,
    with_mirror: bool,
) -> Out {
    fx.poison(gpu);
    let persisted = indexed_attention_select_batch_pm_arm(
        gpu,
        p,
        with_mirror.then_some(&fx.mirror),
        QsaSelectPmArm::Hipcc,
    )
    .unwrap_or_else(|e| {
        panic!(
            "hipcc reference rows={} blocks={} ps={}: {e}",
            p.rows, p.block_count, p.position_start
        )
    });
    fx.download(gpu, persisted, Vec::new())
}

fn run_filter(
    gpu: &mut Gpu,
    fx: &Fixture,
    p: &IndexedAttentionSelectBatch<'_>,
    with_mirror: bool,
    score_pm: bool,
    knobs: QsaFilterKnobs,
) -> Out {
    fx.poison(gpu);
    let (persisted, stats) = indexed_attention_select_batch_filter_arm(
        gpu,
        p,
        with_mirror.then_some(&fx.mirror),
        score_pm,
        knobs,
    )
    .unwrap_or_else(|e| {
        panic!(
            "filter pm_score={score_pm} rows={} blocks={} ps={} mirror={with_mirror}: {e}",
            p.rows, p.block_count, p.position_start
        )
    });
    fx.download(gpu, persisted, stats)
}

/// `out` must be the reference's bytes, flag included, with intact canaries
/// and a mirror only when one was given.
fn check_against(label: &str, out: &Out, reference: &Out, rows: usize, with_mirror: bool) {
    let used = rows * CAPACITY * 4;
    assert!(
        out.selected == reference.selected,
        "{label}: selection differs from the hipcc incumbent"
    );
    assert!(
        out.mirror == reference.mirror,
        "{label}: mirror differs from the hipcc incumbent"
    );
    assert_eq!(
        out.persisted, reference.persisted,
        "{label}: persisted flag"
    );
    assert_eq!(
        out.persisted, with_mirror,
        "{label}: persisted without/with mirror"
    );
    assert!(
        all_poison(&out.selected[used..]),
        "{label}: wrote past the selected rows"
    );
    assert!(
        all_poison(&out.mirror[CAPACITY * 4..]),
        "{label}: wrote past the mirror"
    );
    if with_mirror {
        assert!(
            out.mirror[..CAPACITY * 4] == out.selected[used - CAPACITY * 4..used],
            "{label}: mirror is not the final row's selection"
        );
    } else {
        assert!(
            all_poison(&out.mirror),
            "{label}: wrote a mirror it was not given"
        );
    }
    assert_eq!(out.stats.len(), rows, "{label}: one stat per row");
}

fn score_arms(gpu: &Gpu) -> Vec<bool> {
    if gpu.arch_caps.is_gfx1151() {
        vec![false, true]
    } else {
        vec![false]
    }
}

/// `(position_start + rows) / compress == blocks`, with `tail` extra tokens.
fn position_start(blocks: usize, rows: usize, tail: usize) -> usize {
    blocks * COMPRESS + tail - rows
}

#[test]
fn qsa_filter_selection_matches_hipcc_incumbent_byte_for_byte() {
    let Some(mut gpu) = gpu_or_skip() else { return };
    let fx = Fixture::new(&mut gpu);
    let arms = score_arms(&gpu);
    let mut cases = 0usize;
    let mut repaired_rows = 0usize;
    let mut total_rows = 0usize;
    for blocks in [4095usize, 4096, 4097, 8192, 16384, 65536] {
        let keys = tie_rich_pooled(blocks);
        let pooled = gpu.upload_f32(&keys, &[keys.len()]).expect("pooled upload");
        for rows in [1usize, 2, 3, 4, 5, 6, 7, 8, 16] {
            for tail in 0..COMPRESS {
                let p = fx.params(&pooled, blocks, rows, position_start(blocks, rows, tail));
                let with = run_reference(&mut gpu, &fx, &p, true);
                let without = run_reference(&mut gpu, &fx, &p, false);
                assert!(with.persisted && !without.persisted);
                assert!(
                    with.selected == without.selected,
                    "reference differs with a mirror"
                );
                for &score_pm in &arms {
                    for with_mirror in [true, false] {
                        let label = format!(
                            "blocks={blocks} rows={rows} tail={tail} pm_score={score_pm} mirror={with_mirror}"
                        );
                        let reference = if with_mirror { &with } else { &without };
                        let out = run_filter(
                            &mut gpu,
                            &fx,
                            &p,
                            with_mirror,
                            score_pm,
                            QsaFilterKnobs::default(),
                        );
                        check_against(&label, &out, reference, rows, with_mirror);
                        repaired_rows += out.stats.iter().filter(|s| s.repaired).count();
                        total_rows += out.stats.len();
                        // Repeat: selection, mirror and per-row stats are
                        // bit-identical run to run.
                        let again = run_filter(
                            &mut gpu,
                            &fx,
                            &p,
                            with_mirror,
                            score_pm,
                            QsaFilterKnobs::default(),
                        );
                        check_against(
                            &format!("{label} repeat"),
                            &again,
                            reference,
                            rows,
                            with_mirror,
                        );
                        assert_eq!(
                            stat_tuples(&again.stats),
                            stat_tuples(&out.stats),
                            "{label}: stats differ between repeats"
                        );
                        cases += 1;
                    }
                }
            }
        }
        gpu.free_tensor(pooled).expect("free pooled");
    }
    fx.free(&mut gpu);
    println!("{cases} filter cases exact; repaired rows under default knobs: {repaired_rows}/{total_rows}");
}

#[test]
fn qsa_filter_forced_underfill_and_overflow_repair_to_the_exact_selection() {
    let Some(mut gpu) = gpu_or_skip() else { return };
    let fx = Fixture::new(&mut gpu);
    let arms = score_arms(&gpu);
    // Every row's visible block count must exceed the default cap (4096):
    // `rbc <= cap` takes threshold 0 and never repairs.
    for blocks in [8192usize, 16384, 65536] {
        let keys = tie_rich_pooled(blocks);
        let pooled = gpu.upload_f32(&keys, &[keys.len()]).expect("pooled upload");
        for rows in [1usize, 4, 16] {
            for tail in [0usize, 3] {
                let p = fx.params(&pooled, blocks, rows, position_start(blocks, rows, tail));
                let with = run_reference(&mut gpu, &fx, &p, true);
                let without = run_reference(&mut gpu, &fx, &p, false);
                for (name, knobs) in [
                    ("underfill", underfill_knobs()),
                    ("overflow", overflow_knobs()),
                ] {
                    for &score_pm in &arms {
                        for with_mirror in [true, false] {
                            let label = format!(
                                "{name} blocks={blocks} rows={rows} tail={tail} pm_score={score_pm} mirror={with_mirror}"
                            );
                            let reference = if with_mirror { &with } else { &without };
                            let out = run_filter(&mut gpu, &fx, &p, with_mirror, score_pm, knobs);
                            check_against(&label, &out, reference, rows, with_mirror);
                            for (row, stat) in out.stats.iter().enumerate() {
                                assert!(
                                    stat.repaired,
                                    "{label}: row {row} was not reported repaired"
                                );
                            }
                            let again = run_filter(&mut gpu, &fx, &p, with_mirror, score_pm, knobs);
                            check_against(
                                &format!("{label} repeat"),
                                &again,
                                reference,
                                rows,
                                with_mirror,
                            );
                            assert_eq!(
                                stat_tuples(&again.stats),
                                stat_tuples(&out.stats),
                                "{label}: stats differ between repeats"
                            );
                        }
                    }
                }
            }
        }
        gpu.free_tensor(pooled).expect("free pooled");
    }
    fx.free(&mut gpu);
}

// ---------------------------------------------------------------------------
// from scores

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Equal,
    Zero,
    MostlyZero,
    Inf,
    /// `budget - 1` blocks at B+1, the rest at B (a tie run straddles the
    /// budget boundary by one ULP), every eighth of those at B-1.
    UlpBelow,
    /// `budget` blocks at B+1.
    UlpAt,
    /// `budget + 1` blocks at B+1.
    UlpAbove,
}

const PATTERNS: [Pattern; 7] = [
    Pattern::Equal,
    Pattern::Zero,
    Pattern::MostlyZero,
    Pattern::Inf,
    Pattern::UlpBelow,
    Pattern::UlpAt,
    Pattern::UlpAbove,
];

const ONE: u32 = 0x3f80_0000;
const POS_INF: u32 = 0x7f80_0000;

fn fill_score_row(pattern: Pattern, budget: usize, blocks: usize, row: usize, out: &mut [u32]) {
    const PRIME: usize = 1_000_003; // coprime to every block count used here
    let hi = match pattern {
        Pattern::UlpBelow => budget.saturating_sub(1),
        Pattern::UlpAt => budget,
        Pattern::UlpAbove => budget + 1,
        _ => 0,
    };
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = match pattern {
            Pattern::Equal => 1.5f32.to_bits(),
            Pattern::Zero => 0,
            Pattern::MostlyZero => {
                let h = mix(row * 1_000_003 + i);
                if h % 61 == 0 {
                    [ONE, ONE + 1, 2.0f32.to_bits(), 0.5f32.to_bits()][((h >> 8) % 4) as usize]
                } else {
                    0
                }
            }
            Pattern::Inf => POS_INF,
            Pattern::UlpBelow | Pattern::UlpAt | Pattern::UlpAbove => {
                let j = (i * PRIME + row * 131) % blocks;
                if j < hi {
                    ONE + 1
                } else if j % 8 == 0 {
                    ONE - 1
                } else {
                    ONE
                }
            }
        };
    }
}

/// `rows * blocks` score bits; indices at or past the last row's visible block
/// count are `+inf` so a read past it changes the selection.
fn score_matrix(
    pattern: Pattern,
    budget: usize,
    blocks: usize,
    rows: usize,
    ps: usize,
) -> Vec<u32> {
    let max_rbc = ((ps + rows) / COMPRESS).min(blocks);
    let mut bits = vec![0u32; rows * blocks];
    for row in 0..rows {
        let slice = &mut bits[row * blocks..(row + 1) * blocks];
        fill_score_row(pattern, budget, blocks, row, slice);
        slice[max_rbc..].fill(POS_INF);
    }
    bits
}

/// The selector's exact contract on the CPU: key = `(bits << 32) | (0xffffffff
/// - block)`, top `chosen` keys descending expanded by `compress`, then the
/// causal tail, `-1` padding, everything bounded by `capacity`.
fn oracle(
    bits: &[u32],
    blocks: usize,
    rows: usize,
    budget: usize,
    ps: usize,
    capacity: usize,
) -> Vec<i32> {
    let mut out = vec![-1i32; rows * capacity];
    for row in 0..rows {
        let visible = ps + row + 1;
        let rbc = (visible / COMPRESS).min(blocks);
        let chosen = budget.min(rbc);
        let selected = &mut out[row * capacity..(row + 1) * capacity];
        let mut keys: Vec<u64> = (0..rbc)
            .map(|b| ((bits[row * blocks + b] as u64) << 32) | (0xffff_ffffu64 - b as u64))
            .collect();
        if chosen > 0 {
            if chosen < rbc {
                keys.select_nth_unstable_by(chosen - 1, |a, b| b.cmp(a));
                keys.truncate(chosen);
            }
            keys.sort_unstable_by(|a, b| b.cmp(a));
            for (rank, key) in keys.iter().enumerate() {
                let block = (0xffff_ffffu32 - (*key as u32)) as usize;
                for slot in 0..COMPRESS {
                    let index = rank * COMPRESS + slot;
                    let token = block * COMPRESS + slot;
                    if index < capacity && token < visible {
                        selected[index] = token as i32;
                    }
                }
            }
        }
        let tail_start = rbc * COMPRESS;
        for token in tail_start..visible {
            let index = chosen * COMPRESS + (token - tail_start);
            if index < capacity {
                selected[index] = token as i32;
            }
        }
    }
    out
}

struct ScoreFixture {
    selected: GpuTensor,
    mirror: GpuTensor,
}

impl ScoreFixture {
    fn new(gpu: &Gpu, selected_rows: usize) -> Self {
        Self {
            selected: poisoned(gpu, selected_rows * CAPACITY * 4 + CANARY),
            mirror: poisoned(gpu, CAPACITY * 4 + CANARY),
        }
    }

    fn poison(&self, gpu: &Gpu) {
        poison(gpu, &self.selected);
        poison(gpu, &self.mirror);
    }

    fn params<'a>(
        &'a self,
        scores: &'a GpuTensor,
        blocks: usize,
        rows: usize,
        budget: usize,
        ps: usize,
        capacity: usize,
    ) -> QsaFromScores<'a> {
        QsaFromScores {
            scores,
            score_stride: blocks,
            selected: &self.selected,
            rows,
            block_count: blocks,
            budget_blocks: budget,
            compress: COMPRESS,
            position_start: ps,
            capacity,
        }
    }

    fn download(&self, gpu: &Gpu) -> (Vec<u8>, Vec<u8>) {
        (
            gpu.download_raw_bytes(&self.selected)
                .expect("selected download"),
            gpu.download_raw_bytes(&self.mirror)
                .expect("mirror download"),
        )
    }

    fn free(self, gpu: &mut Gpu) {
        gpu.free_tensor(self.selected).expect("free");
        gpu.free_tensor(self.mirror).expect("free");
    }
}

/// Expected full buffers: the oracle's bytes, poison after them; the mirror is
/// the last row's `capacity` entries.
fn expected_buffers(
    expected: &[i32],
    rows: usize,
    capacity: usize,
    selected_rows: usize,
    with_mirror: bool,
) -> (Vec<u8>, Vec<u8>) {
    let mut selected = i32_bytes(expected);
    selected.resize(selected_rows * CAPACITY * 4 + CANARY, POISON);
    let mut mirror = if with_mirror {
        i32_bytes(&expected[(rows - 1) * capacity..rows * capacity])
    } else {
        Vec::new()
    };
    mirror.resize(CAPACITY * 4 + CANARY, POISON);
    (selected, mirror)
}

#[test]
fn qsa_filter_from_scores_matches_oracle_and_incumbents() {
    let Some(mut gpu) = gpu_or_skip() else { return };
    let fx = ScoreFixture::new(&gpu, MAX_ROWS);
    let pm_available = gpu.arch_caps.is_gfx1151();
    let knob_sets = [
        ("default", QsaFilterKnobs::default()),
        ("underfill", underfill_knobs()),
        ("overflow", overflow_knobs()),
    ];
    let mut cases = 0usize;
    for blocks in [513usize, 4096, 4097, 65536] {
        for rows in [1usize, 3, MAX_ROWS] {
            // Full tail, ramp (row_block_count < block_count) and start at 0
            // (rows with no visible block, then a growing prefix).
            for ps in [blocks * COMPRESS + 2 - rows, blocks * COMPRESS / 2, 0] {
                for pattern in PATTERNS {
                    for budget in [0usize, 1, 511, 512] {
                        let bits = score_matrix(pattern, budget, blocks, rows, ps);
                        let floats: Vec<f32> = bits.iter().map(|b| f32::from_bits(*b)).collect();
                        let scores = gpu
                            .upload_f32(&floats, &[floats.len()])
                            .expect("scores upload");
                        for capacity in [
                            budget * COMPRESS + COMPRESS - 1,
                            (budget * COMPRESS) / 2 + 1,
                        ] {
                            let expected = oracle(&bits, blocks, rows, budget, ps, capacity);
                            let want_with =
                                expected_buffers(&expected, rows, capacity, MAX_ROWS, true);
                            let want_without =
                                expected_buffers(&expected, rows, capacity, MAX_ROWS, false);
                            let p = fx.params(&scores, blocks, rows, budget, ps, capacity);
                            let label = format!(
                                "{pattern:?} blocks={blocks} rows={rows} ps={ps} budget={budget} cap={capacity}"
                            );

                            // Incumbents validate the oracle (selection + mirror).
                            for pm in [false, true] {
                                if pm && !pm_available {
                                    continue;
                                }
                                fx.poison(&gpu);
                                qsa_incumbent_select_from_scores_lab(
                                    &mut gpu,
                                    &p,
                                    Some(&fx.mirror),
                                    pm,
                                )
                                .unwrap_or_else(|e| panic!("incumbent pm={pm} {label}: {e}"));
                                let got = fx.download(&gpu);
                                assert!(
                                    got.0 == want_with.0,
                                    "incumbent pm={pm} {label}: selection != oracle"
                                );
                                assert!(
                                    got.1 == want_with.1,
                                    "incumbent pm={pm} {label}: mirror != oracle"
                                );
                            }

                            for (knob_name, knobs) in knob_sets {
                                let label = format!("{label} knobs={knob_name}");
                                fx.poison(&gpu);
                                let stats = qsa_filter_select_from_scores_lab(
                                    &mut gpu,
                                    &p,
                                    Some(&fx.mirror),
                                    knobs,
                                )
                                .unwrap_or_else(|e| panic!("filter {label}: {e}"));
                                let got = fx.download(&gpu);
                                assert_eq!(stats.len(), rows, "{label}: one stat per row");
                                assert!(
                                    got.0 == want_with.0,
                                    "filter {label}: selection != oracle"
                                );
                                assert!(got.1 == want_with.1, "filter {label}: mirror != oracle");

                                fx.poison(&gpu);
                                let again = qsa_filter_select_from_scores_lab(
                                    &mut gpu,
                                    &p,
                                    Some(&fx.mirror),
                                    knobs,
                                )
                                .unwrap_or_else(|e| panic!("filter repeat {label}: {e}"));
                                let repeat = fx.download(&gpu);
                                assert!(repeat == got, "filter {label}: repeat differs");
                                assert_eq!(
                                    stat_tuples(&again),
                                    stat_tuples(&stats),
                                    "{label}: stats differ between repeats"
                                );

                                // No mirror: same selection, mirror untouched.
                                fx.poison(&gpu);
                                qsa_filter_select_from_scores_lab(&mut gpu, &p, None, knobs)
                                    .unwrap_or_else(|e| panic!("filter no-mirror {label}: {e}"));
                                let bare = fx.download(&gpu);
                                assert!(
                                    bare.0 == want_without.0,
                                    "filter no-mirror {label}: selection != oracle"
                                );
                                assert!(
                                    bare.1 == want_without.1,
                                    "filter no-mirror {label}: wrote a mirror"
                                );
                                cases += 1;
                            }
                        }
                        gpu.free_tensor(scores).expect("free scores");
                    }
                }
            }
        }
    }
    fx.free(&mut gpu);
    println!("{cases} from-scores filter cases exact");
}

// ---------------------------------------------------------------------------
// select-only benchmark

const BENCH_WARM: usize = 5;
const BENCH_RUNS: usize = 50;

/// Per-call hipEvent median (ms): each timed call is its own event pair.
fn median_event_ms(gpu: &mut Gpu, mut launch: impl FnMut(&mut Gpu)) -> f32 {
    for _ in 0..BENCH_WARM {
        launch(gpu);
    }
    gpu.hip.device_synchronize().expect("sync");
    let start = gpu.hip.event_create().expect("event");
    let stop = gpu.hip.event_create().expect("event");
    let mut times = Vec::with_capacity(BENCH_RUNS);
    for _ in 0..BENCH_RUNS {
        gpu.hip.event_record(&start, None).expect("record");
        launch(gpu);
        gpu.hip.event_record(&stop, None).expect("record");
        gpu.hip.event_synchronize(&stop).expect("event sync");
        times.push(gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed"));
    }
    gpu.hip.event_destroy(start).expect("destroy");
    gpu.hip.event_destroy(stop).expect("destroy");
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[BENCH_RUNS / 2]
}

const HIPCC_SELECT: &str = "indexed_attention_select_from_scores";
const PM_SELECT: &str = "indexed_attention_select_from_scores_pm_gfx1151";

/// Select only, scores resident, tie-rich rows: the hipcc incumbent, the PM
/// incumbent (gfx1151) and the filter trio, each timed as raw
/// `launch_kernel_blob` launches on prepared kernarg blobs (incumbent: one
/// select kernel; trio: threshold, compact, finish) with images loaded and
/// control/candidate buffers allocated before the timed window, so no host
/// helper checks, arg setup, stats download, JIT or scratch growth is inside
/// it. Before timing, the lab filter, the raw trio and each raw incumbent image
/// must reproduce the lab incumbent's bytes.
#[test]
#[ignore = "benchmark; needs gfx1151/gfx1201: cargo test -p rdna-compute --release --features lab --test qsa_filter_select -- --ignored --nocapture qsa_select_only_bench"]
fn qsa_select_only_bench() {
    let Some(mut gpu) = gpu_or_skip() else { return };
    let pm_available = gpu.arch_caps.is_gfx1151();
    let fx = ScoreFixture::new(&gpu, 256);
    let knobs = QsaFilterKnobs::default();
    let (samples, target, cap) = (knobs.samples as i32, knobs.target as i32, knobs.cap as i32);
    for kernel in [
        "qsa_filter_threshold",
        "qsa_filter_compact",
        "qsa_filter_finish",
    ] {
        gpu.ensure_kernel_public(FILTER_MODULE, FILTER_SRC, kernel)
            .expect("load filter kernel");
    }
    println!(
        "select-only median of {BENCH_RUNS} (ms), per call after {BENCH_WARM} warm; budget_blocks={BUDGET_BLOCKS} compress={COMPRESS} samples={samples} target={target} cap={cap}"
    );
    for rows in [1usize, 4, 16, 256] {
        for blocks in [2048usize, 8192, 32768, 65536] {
            let ps = blocks * COMPRESS - rows;
            let bits: Vec<u32> = (0..rows * blocks)
                .map(|i| {
                    let h = mix(i);
                    if h % 3 == 0 {
                        (1.0f32 + ((h >> 8) % 16) as f32 / 16.0).to_bits()
                    } else {
                        0x3f00_0000 | (h >> 9)
                    }
                })
                .collect();
            let floats: Vec<f32> = bits.iter().map(|b| f32::from_bits(*b)).collect();
            let scores = gpu
                .upload_f32(&floats, &[floats.len()])
                .expect("scores upload");
            let p = fx.params(&scores, blocks, rows, BUDGET_BLOCKS, ps, CAPACITY);
            let used = rows * CAPACITY * 4;

            // Equality gates before timing: lab filter and raw trio vs incumbent.
            fx.poison(&gpu);
            qsa_incumbent_select_from_scores_lab(&mut gpu, &p, Some(&fx.mirror), false)
                .expect("incumbent");
            let reference = fx.download(&gpu);
            fx.poison(&gpu);
            qsa_filter_select_from_scores_lab(&mut gpu, &p, Some(&fx.mirror), knobs)
                .expect("filter");
            assert!(
                fx.download(&gpu) == reference,
                "rows={rows} blocks={blocks}: lab filter != incumbent"
            );

            let ctl = gpu
                .upload_raw(&vec![0u8; rows * 16], &[rows * 16])
                .expect("ctl");
            let cand_bytes = rows * cap as usize * 8;
            let cand = gpu
                .upload_raw(&vec![0u8; cand_bytes], &[cand_bytes])
                .expect("cand");
            let mut threshold = KernargBlob::new();
            threshold.push_ptr(scores.buf.as_ptr());
            threshold.push_i32(blocks as i32);
            threshold.push_ptr(ctl.buf.as_ptr());
            for v in [
                rows as i32,
                blocks as i32,
                COMPRESS as i32,
                ps as i32,
                samples,
                target,
                cap,
            ] {
                threshold.push_i32(v);
            }
            threshold.pad_to(16);
            let mut compact = KernargBlob::new();
            compact.push_ptr(scores.buf.as_ptr());
            compact.push_i32(blocks as i32);
            compact.push_ptr(ctl.buf.as_ptr());
            compact.push_ptr(cand.buf.as_ptr());
            for v in [rows as i32, blocks as i32, COMPRESS as i32, ps as i32, cap] {
                compact.push_i32(v);
            }
            compact.pad_to(16);
            let mut finish = KernargBlob::new();
            finish.push_ptr(scores.buf.as_ptr());
            finish.push_i32(blocks as i32);
            finish.push_ptr(ctl.buf.as_ptr());
            finish.push_ptr(cand.buf.as_ptr());
            finish.push_ptr(fx.selected.buf.as_ptr());
            finish.push_ptr(fx.mirror.buf.as_ptr());
            for v in [
                rows as i32,
                blocks as i32,
                BUDGET_BLOCKS as i32,
                COMPRESS as i32,
                ps as i32,
                CAPACITY as i32,
                cap,
            ] {
                finish.push_i32(v);
            }
            finish.pad_to(16);
            let tiles = blocks.div_ceil(4096) as u32;
            let mut trio = |g: &mut Gpu| {
                g.launch_kernel_blob(
                    "qsa_filter_threshold",
                    [rows as u32, 1, 1],
                    [256, 1, 1],
                    0,
                    threshold.as_mut_slice(),
                )
                .expect("threshold");
                g.launch_kernel_blob(
                    "qsa_filter_compact",
                    [tiles, rows as u32, 1],
                    [256, 1, 1],
                    0,
                    compact.as_mut_slice(),
                )
                .expect("compact");
                g.launch_kernel_blob(
                    "qsa_filter_finish",
                    [rows as u32, 1, 1],
                    [256, 1, 1],
                    0,
                    finish.as_mut_slice(),
                )
                .expect("finish");
            };
            fx.poison(&gpu);
            trio(&mut gpu);
            gpu.hip.device_synchronize().expect("sync");
            assert!(
                fx.download(&gpu) == reference,
                "rows={rows} blocks={blocks}: raw trio != incumbent"
            );
            assert!(
                all_poison(&reference.0[used..]),
                "rows={rows} blocks={blocks}: reference wrote past rows"
            );

            // Raw incumbent: same kernarg layout and grid as
            // `indexed_attention_select_rows8` (scores, stride, selected, rows,
            // blocks, budget, compress, position_start, capacity, mirror).
            // The lab calls load the hipcc and PM images before any timing; the
            // timed window is then one `launch_kernel_blob` on a prepared blob,
            // exactly like the trio, and each image must reproduce `reference`.
            let mut incumbent = KernargBlob::new();
            incumbent.push_ptr(scores.buf.as_ptr());
            incumbent.push_i32(blocks as i32);
            incumbent.push_ptr(fx.selected.buf.as_ptr());
            for v in [
                rows as i32,
                blocks as i32,
                BUDGET_BLOCKS as i32,
                COMPRESS as i32,
                ps as i32,
                CAPACITY as i32,
            ] {
                incumbent.push_i32(v);
            }
            incumbent.push_ptr(fx.mirror.buf.as_ptr());
            incumbent.pad_to(16);
            let mut images = vec![HIPCC_SELECT];
            if pm_available {
                qsa_incumbent_select_from_scores_lab(&mut gpu, &p, Some(&fx.mirror), true)
                    .expect("load PM image");
                images.push(PM_SELECT);
            }
            for image in images {
                fx.poison(&gpu);
                gpu.launch_kernel_blob(
                    image,
                    [rows as u32, 1, 1],
                    [256, 1, 1],
                    0,
                    incumbent.as_mut_slice(),
                )
                .unwrap_or_else(|e| panic!("raw {image}: {e}"));
                gpu.hip.device_synchronize().expect("sync");
                assert!(
                    fx.download(&gpu) == reference,
                    "rows={rows} blocks={blocks}: raw {image} != reference"
                );
            }

            let hipcc = median_event_ms(&mut gpu, |g| {
                g.launch_kernel_blob(
                    HIPCC_SELECT,
                    [rows as u32, 1, 1],
                    [256, 1, 1],
                    0,
                    incumbent.as_mut_slice(),
                )
                .expect("hipcc incumbent");
            });
            let pm = pm_available.then(|| {
                median_event_ms(&mut gpu, |g| {
                    g.launch_kernel_blob(
                        PM_SELECT,
                        [rows as u32, 1, 1],
                        [256, 1, 1],
                        0,
                        incumbent.as_mut_slice(),
                    )
                    .expect("pm incumbent");
                })
            });
            let filter = median_event_ms(&mut gpu, &mut trio);
            let best = pm.map_or(hipcc, |pm| pm.min(hipcc));
            println!(
                "rows={rows:>3} blocks={blocks:>6}  hipcc={hipcc:>8.4}  pm={}  trio={filter:>8.4}  trio/best_incumbent={:.3}",
                pm.map_or("     n/a".to_string(), |v| format!("{v:>8.4}")),
                filter / best
            );
            for t in [ctl, cand, scores] {
                gpu.free_tensor(t).expect("free");
            }
        }
    }
    fx.free(&mut gpu);
}
