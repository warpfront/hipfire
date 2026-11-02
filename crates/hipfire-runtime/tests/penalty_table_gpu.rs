// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The GPU penalty prepass (`Gpu::apply_penalty_table`, kernel
//! `logit_penalty_table_rows`) must leave every logit bit-identical to the
//! host policy `sampler::apply_logit_policy_cpu` over the same history and
//! config: the Qwen4 AR greedy row and every sampled-MTP verify row take the
//! prepass instead of the host policy, so any difference changes the target
//! distribution (or a greedy id) against the host AR producer.
//!
//! Covers repeat > 1, repeat < 1, the 1.5 cap, presence, frequency and all
//! combined; windows 1, short, 128 and 2048 (stage regrowth); histories
//! shorter than, equal to and past the window; out-of-vocabulary ids; several
//! rows per launch with sliding windows (the verify-row layout); logits that
//! are positive, negative, ±0, subnormal, huge and ±infinity; and the
//! Flash-Next vocabulary width.
//!
//! `#[ignore]`d: needs a GPU with a working HIP toolchain. Run explicitly:
//!
//!   cargo test -p hipfire-runtime --release --test penalty_table_gpu -- --ignored

use hipfire_runtime::sampler::{apply_logit_policy_cpu, PenaltyTable, SamplerConfig};
use rdna_compute::Gpu;

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
}

/// Logits drawn across the f32 range a logit row can hold.
fn synth_logits(rng: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| match rng.below(16) {
            0 => 0.0,
            1 => -0.0,
            2 => f32::from_bits(rng.below(0x0080_0000)), // positive subnormal
            3 => -f32::from_bits(rng.below(0x0080_0000)),
            4 => f32::INFINITY,
            5 => f32::NEG_INFINITY,
            6 => f32::from_bits(0x7f00_0000 + rng.below(0x007f_ffff)), // huge
            7 => -f32::from_bits(0x7f00_0000 + rng.below(0x007f_ffff)),
            _ => (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32 * 64.0 - 32.0,
        })
        .collect()
}

struct Case {
    name: &'static str,
    repeat: f32,
    window: usize,
    presence: f32,
    frequency: f32,
}

const CASES: &[Case] = &[
    Case {
        name: "presence_tc",
        repeat: 1.0,
        window: 128,
        presence: 1.5,
        frequency: 0.0,
    },
    Case {
        name: "repeat_1_1",
        repeat: 1.1,
        window: 128,
        presence: 0.0,
        frequency: 0.0,
    },
    Case {
        name: "repeat_lt_1",
        repeat: 0.87,
        window: 64,
        presence: 0.0,
        frequency: 0.0,
    },
    Case {
        name: "repeat_capped",
        repeat: 2.3,
        window: 32,
        presence: 0.0,
        frequency: 0.0,
    },
    Case {
        name: "frequency_odd",
        repeat: 1.0,
        window: 128,
        presence: 0.0,
        frequency: 0.3,
    },
    Case {
        name: "combined",
        repeat: 1.07,
        window: 128,
        presence: 0.7,
        frequency: 0.13,
    },
    Case {
        name: "combined_w1",
        repeat: 1.3,
        window: 1,
        presence: 1.5,
        frequency: 0.5,
    },
    Case {
        name: "combined_w2048",
        repeat: 1.05,
        window: 2048,
        presence: 0.4,
        frequency: 0.21,
    },
];

fn cfg(case: &Case) -> SamplerConfig {
    let mut c = SamplerConfig::greedy();
    c.repeat_penalty = case.repeat;
    c.repeat_window = case.window;
    c.presence_penalty = case.presence;
    c.frequency_penalty = case.frequency;
    c
}

/// `rows` verify-style histories: a shared base of `base_len` tokens, row `r`
/// sees the base plus the first `r` drafts, clipped to the window by the
/// table builder exactly like `PenaltyHistory::row`.
fn sliding_rows(rng: &mut Lcg, vocab: usize, base_len: usize, rows: usize) -> Vec<Vec<u32>> {
    let pool = (vocab as u32 / 8).clamp(2, 4096);
    let tokens: Vec<u32> = (0..base_len + rows)
        .map(|i| {
            if i % 23 == 11 {
                vocab as u32 + rng.below(64) // out of vocabulary
            } else {
                rng.below(pool)
            }
        })
        .collect();
    (0..rows).map(|r| tokens[..base_len + r].to_vec()).collect()
}

fn check(
    gpu: &mut Gpu,
    stage: &mut Option<rdna_compute::sampling::PenaltyTableStage>,
    rng: &mut Lcg,
    case: &Case,
    vocab: usize,
    base_len: usize,
    rows: usize,
) {
    let cfg = cfg(case);
    let hists = sliding_rows(rng, vocab, base_len, rows);
    let mut table = PenaltyTable::default();
    table.reset(&cfg);
    for hist in &hists {
        table.push_row(hist, &cfg, vocab);
    }
    let orig = synth_logits(rng, rows * vocab);
    let logits = gpu
        .upload_f32(&orig, &[rows * vocab])
        .expect("upload logits");
    gpu.apply_penalty_table(
        stage,
        &logits,
        vocab,
        table.flags(),
        table.row_ends(),
        table.entries(),
    )
    .expect("apply_penalty_table");
    let got = gpu.download_f32(&logits).expect("download logits");
    gpu.free_tensor(logits).expect("free logits");
    let mut touched = 0usize;
    for (row, hist) in hists.iter().enumerate() {
        let mut want = orig[row * vocab..(row + 1) * vocab].to_vec();
        apply_logit_policy_cpu(&mut want, hist, &cfg);
        let got = &got[row * vocab..(row + 1) * vocab];
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{} vocab={vocab} base={base_len} row={row} logit[{i}]: gpu {g:e} ({:#010x}) cpu {w:e} ({:#010x}), orig {:e}",
                case.name,
                g.to_bits(),
                w.to_bits(),
                orig[row * vocab + i],
            );
        }
        touched += want
            .iter()
            .zip(&orig[row * vocab..(row + 1) * vocab])
            .filter(|(w, o)| w.to_bits() != o.to_bits())
            .count();
    }
    assert!(
        touched > 0,
        "{} vocab={vocab}: the case penalized nothing",
        case.name
    );
}

#[test]
#[ignore = "needs a GPU"]
fn penalty_table_prepass_matches_host_policy_bit_for_bit() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut stage = None;
    let mut rng = Lcg(0x9e17_ab1e);
    for case in CASES {
        for &(vocab, base_len, rows) in &[
            (248_320usize, 1500usize, 1usize), // Qwen4 AR row
            (248_320, 1500, 4),                // K=3 verify block
            (4096, 40, 11),                    // short history, 11 rows
            (4096, case.window, 3),            // exactly the window
            (4096, case.window + 7, 11),       // past the window
            (300, 3, 2),
        ] {
            check(&mut gpu, &mut stage, &mut rng, case, vocab, base_len, rows);
        }
    }
    if let Some(stage) = stage.take() {
        gpu.free_penalty_table_stage(stage).expect("free stage");
    }
    eprintln!("penalty table prepass bit-exact on {}", gpu.arch);
}
