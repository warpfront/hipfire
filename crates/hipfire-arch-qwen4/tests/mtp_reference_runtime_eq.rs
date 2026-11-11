// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU pin of `reference_mtp::mtp_forward_step` against the runtime head
//! equations (`mtp_gpu::head_block`): per-branch `pre_fc_norm_hidden`, per-branch
//! normalised HC write gates, the learned MLP HC read as the MoE input, and the
//! per-head interleaved `[q | gate]` `q_proj` layout.
//!
//! The golden (`fixtures/mtp_reference_runtime_eq.json`) is the independent
//! torch head (`mtrain/mi300x/fn_mtp_head.py`, `use_runtime_equations()`,
//! `qgate_layout = interleaved`, CPU fp32) run on the same compact fixture
//! (four HC branches, 4 query / 2 KV heads, a 4-head indexer, 16 experts with
//! normalized top-10, a shared expert, six fill steps then two chained steps).
//! The previous reference equations (joint 4H prefc norm, joint 4H write-gate
//! norm, plain-mean MoE input, split Q/gate rows) route to different experts
//! on every one of the eight steps (collapsed hidden rel diff 0.13-0.94), so
//! this fixture fails on them.
//!
//! `cargo test -p hipfire-arch-qwen4 --features reference-parity --test mtp_reference_runtime_eq`

#![cfg(feature = "reference-parity")]

use hipfire_arch_qwen4::reference_mtp::{
    mtp_forward_step, MtpAttentionWeights, MtpFinalHyperWeights, MtpForwardInput, MtpHyperWeights,
    MtpMatrix, MtpMoeWeights, MtpQsaGeometry, Qwen4MtpState, ReferenceMtpWeights, MTP_BRANCHES,
};
use rdna_compute::rope::{Rope, RopeScaling};
use serde_json::Value;

const HIDDEN: usize = 16;
const HC_RANK: usize = 6;
const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const INDEX_HEADS: usize = 4;
const INDEX_DIM: usize = 8;
const EXPERTS: usize = 16;
const EXPERT_INTERMEDIATE: usize = 6;
const SHARED_INTERMEDIATE: usize = 6;
const TOL: f64 = 1.0e-4;

/// SplitMix64 + Box-Muller: deterministic, dependency-free normals.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self, n: usize, std: f64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let (u1, u2) = (self.uniform(), self.uniform());
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                (z * std) as f32
            })
            .collect()
    }
}

fn matrix(rng: &mut Rng, rows: usize, cols: usize, std: f64) -> MtpMatrix {
    MtpMatrix::new(rows, cols, rng.normal(rows * cols, std)).unwrap()
}

fn hyper(rng: &mut Rng, wide: usize) -> MtpHyperWeights {
    MtpHyperWeights {
        norm: rng.normal(wide, 0.1),
        down: matrix(rng, HC_RANK, wide, 0.5 / (wide as f64).sqrt() * 4.0),
        up: matrix(rng, wide, HC_RANK, 0.7),
        block_inject: matrix(rng, MTP_BRANCHES, wide, 0.5),
    }
}

fn geometry() -> MtpQsaGeometry {
    MtpQsaGeometry {
        q_heads: Q_HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        index_heads: INDEX_HEADS,
        index_dim: INDEX_DIM,
        compress_ratio: 2,
        budget: 8,
        max_seq_len: 16,
        rotary_dim: 4,
        rope: Rope {
            theta: 10_000_000.0,
            scaling: RopeScaling::None,
        },
    }
}

/// The weights of the golden's source fixture (same stream, same draw order).
fn fixture() -> ReferenceMtpWeights {
    let wide = MTP_BRANCHES * HIDDEN;
    let q_width = Q_HEADS * HEAD_DIM;
    let kv_width = KV_HEADS * HEAD_DIM;
    let mut rng = Rng(0x4D54_5052_4F4F_4601);
    let s = 1.0 / (HIDDEN as f64).sqrt();
    let weights = ReferenceMtpWeights {
        pre_fc_norm_embedding: rng.normal(HIDDEN, 0.1),
        pre_fc_norm_hidden: rng.normal(wide, 0.1),
        fc_embedding: matrix(&mut rng, HIDDEN, HIDDEN, s),
        fc_hidden: matrix(&mut rng, HIDDEN, HIDDEN, s),
        attention: MtpAttentionWeights {
            q_proj: matrix(&mut rng, 2 * q_width, HIDDEN, s),
            k_proj: matrix(&mut rng, kv_width, HIDDEN, s),
            v_proj: matrix(&mut rng, kv_width, HIDDEN, s),
            o_proj: matrix(&mut rng, HIDDEN, q_width, 1.0 / (q_width as f64).sqrt()),
            q_norm: rng.normal(HEAD_DIM, 0.1),
            k_norm: rng.normal(HEAD_DIM, 0.1),
            index_qk_proj: matrix(&mut rng, (INDEX_HEADS + 1) * INDEX_DIM, HIDDEN, s),
            index_q_norm: rng.normal(INDEX_DIM, 0.1),
            index_k_norm: rng.normal(INDEX_DIM, 0.1),
        },
        attention_hyper: hyper(&mut rng, wide),
        moe: MtpMoeWeights {
            router: matrix(&mut rng, EXPERTS, HIDDEN, 1.5),
            routed_gate_up: (0..EXPERTS)
                .map(|_| matrix(&mut rng, 2 * EXPERT_INTERMEDIATE, HIDDEN, s))
                .collect(),
            routed_down: (0..EXPERTS)
                .map(|_| {
                    matrix(
                        &mut rng,
                        HIDDEN,
                        EXPERT_INTERMEDIATE,
                        1.0 / (EXPERT_INTERMEDIATE as f64).sqrt(),
                    )
                })
                .collect(),
            shared_gate_scalar: rng.normal(HIDDEN, s),
            shared_gate: matrix(&mut rng, SHARED_INTERMEDIATE, HIDDEN, s),
            shared_up: matrix(&mut rng, SHARED_INTERMEDIATE, HIDDEN, s),
            shared_down: matrix(
                &mut rng,
                HIDDEN,
                SHARED_INTERMEDIATE,
                1.0 / (SHARED_INTERMEDIATE as f64).sqrt(),
            ),
        },
        mlp_hyper: hyper(&mut rng, wide),
        final_hyper: MtpFinalHyperWeights {
            norm: rng.normal(wide, 0.1),
            down: matrix(&mut rng, HC_RANK, wide, 0.5 / (wide as f64).sqrt() * 4.0),
            up: matrix(&mut rng, wide, HC_RANK, 0.7),
        },
        qsa_geometry: geometry(),
    };
    weights.validate().expect("fixture weights validate");
    weights
}

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_f64().expect("number") as f32)
        .collect()
}

/// max|a-b| / max|b|.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let diff = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0.0, f64::max);
    let scale = b.iter().map(|&y| (y as f64).abs()).fold(1.0e-30, f64::max);
    diff / scale
}

#[test]
fn reference_step_matches_the_runtime_equation_golden() {
    let golden: Value =
        serde_json::from_str(include_str!("fixtures/mtp_reference_runtime_eq.json"))
            .expect("golden json");
    let steps = golden["steps"].as_array().expect("steps");
    assert_eq!(steps.len(), 8);
    let weights = fixture();
    let mut state = Qwen4MtpState::new(geometry()).expect("state");
    for step in steps {
        let position = step["position"].as_u64().unwrap() as usize;
        let out = mtp_forward_step(
            &weights,
            &mut state,
            MtpForwardInput {
                token_embedding: &floats(&step["token_embedding"]),
                backbone_hidden: &floats(&step["backbone_hidden"]),
                position,
                step_index: step["step_index"].as_u64().unwrap() as usize,
            },
        )
        .expect("mtp_forward_step");
        let expected_experts: Vec<usize> = step["selected_experts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(
            out.selected_experts.to_vec(),
            expected_experts,
            "position {position}: routed experts"
        );
        let checks = [
            ("collapsed_hidden", &out.collapsed_hidden[..]),
            ("wide_hidden", &out.wide_hidden[..]),
            ("routing_weights", &out.routing_weights[..]),
        ];
        for (name, actual) in checks {
            let r = rel(actual, &floats(&step[name]));
            assert!(r <= TOL, "position {position}: {name} rel {r:e} > {TOL:e}");
        }
    }
}
