// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pins the CPU oracle `reference_mtp::mtp_forward_step` to the production HIP
//! head lowering (`mtp_gpu::head_block`) on a real Flash-Next artifact.
//!
//! The CPU side is built from the artifact's own head tensors (BF16 / Q8F16
//! decoded by the runtime's host dequantizer, the routed experts the router
//! selects decoded from their qt44 / qt53 groups by `hipfire-cpu`) and a fixed
//! deterministic input (a `[4H]` backbone hidden per prompt-fill row and the
//! trunk embedding row of each token). The GPU side is the real bundle's head
//! (`state_parity::run_mtp_head_reference_pin_gpu`): four step-0 rows with a
//! fresh QSA selection plus one chained step on the head's own wide output.
//!
//! Asserted per step: the draft the production draft head ranks from the
//! reference collapsed hidden is the draft the GPU head produces (draft_1 on
//! the four fill rows, the chained draft_2 on the fifth), and the collapsed and
//! wide hiddens agree within the head's BF16-rounding / activation-quantization
//! noise (cosine, relative L2). The old reference equations (joint `4H`
//! pre-fc and write-gate norms, plain-mean MoE input, split Q/gate rows)
//! diverge from the GPU head by far more than these tolerances.
//!
//! `#[ignore]`d: needs a real HIP GPU and the artifact named by
//! `HIPFIRE_MTP_REF_PIN_MODEL`, plus a fixed
//! `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` (a discrete card cannot hold every
//! expert). Run on one card through the fabric:
//!
//! ```text
//! HIPFIRE_MTP_REF_PIN_MODEL=/path/to/qwen3.8-flash-next-gptq3.mq4 \
//! HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=5 \
//! cargo test -p hipfire-arch-qwen4 --features reference-parity \
//!     --test mtp_head_reference_pin_hw -- --ignored --exact \
//!     mtp_head_reference_pin_hw --nocapture
//! ```

#![cfg(feature = "reference-parity")]

use hipfire_arch_qwen4::reference_mtp::{
    mtp_forward_step, MtpAttentionWeights, MtpFinalHyperWeights, MtpForwardInput, MtpHyperWeights,
    MtpMatrix, MtpMoeWeights, MtpQsaGeometry, Qwen4MtpState, ReferenceMtpWeights, MTP_TOP_K,
};
use hipfire_arch_qwen4::state_parity::{run_mtp_head_reference_pin_gpu, MtpHeadPinCase};
use hipfire_cpu::quant::{dequant_group, CpuQuant};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_backend::dequantize_weight_to_f32;
use std::path::PathBuf;

const MODEL_ENV: &str = "HIPFIRE_MTP_REF_PIN_MODEL";
const MAX_SEQ: usize = 4096;
const HIDDEN: usize = 2560;
const BRANCHES: usize = 4;
const L0: &str = "mtp.layers.0.";
/// Fill rows (step-0 head steps) then one chained step.
const FILL_TOKENS: [u32; 4] = [1_000, 21_763, 5, 99_999];
const CHAIN_TOKEN: u32 = 4_242;

/// Cosine / relative-L2 gates for the collapsed and wide hiddens. The GPU head
/// rounds its HC path to BF16 and quantizes activations for the Q8 / MQ
/// projections; the reference is exact fp32 over the same decoded weights.
const MIN_COSINE: f64 = 0.99;
const MAX_REL_L2: f64 = 0.15;

fn model_path() -> PathBuf {
    let path = std::env::var_os(MODEL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{MODEL_ENV} must name a Flash-Next artifact"));
    assert!(
        path.is_file(),
        "{MODEL_ENV}={} is not a file",
        path.display()
    );
    path
}

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

/// A dense head tensor decoded to f32 (BF16 / F16 / F32 / Q8F16).
fn dense(hfq: &HfqFile, name: &str) -> (Vec<f32>, Vec<usize>) {
    let (info, bytes) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("artifact has no tensor {name}"));
    let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
    let n: usize = shape.iter().product();
    assert!(
        matches!(info.quant_type, 1 | 2 | 3 | 16),
        "{name}: quant type {} is not a dense head format",
        info.quant_type
    );
    let values = dequantize_weight_to_f32(info.quant_type, bytes, n);
    assert_eq!(values.len(), n, "{name}: decoded length");
    (values, shape)
}

fn matrix(hfq: &HfqFile, name: &str) -> MtpMatrix {
    let (values, shape) = dense(hfq, name);
    assert_eq!(shape.len(), 2, "{name}: not a matrix");
    MtpMatrix::new(shape[0], shape[1], values).expect("matrix")
}

fn vector(hfq: &HfqFile, name: &str) -> Vec<f32> {
    dense(hfq, name).0
}

fn hyper(hfq: &HfqFile, prefix: &str) -> MtpHyperWeights {
    MtpHyperWeights {
        norm: vector(hfq, &format!("{prefix}.hc_norm.weight")),
        down: matrix(hfq, &format!("{prefix}.input_mix_weight_down.weight")),
        up: matrix(hfq, &format!("{prefix}.input_mix_weight_up.weight")),
        block_inject: matrix(hfq, &format!("{prefix}.block_inject_weight.weight")),
    }
}

/// Rows `[0, rows)` of expert `expert` of a fused `[experts, rows, cols]` qt44 /
/// qt53 tensor, decoded to the original basis.
fn expert_matrix(hfq: &HfqFile, name: &str, expert: usize) -> MtpMatrix {
    let (info, bytes) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("artifact has no tensor {name}"));
    let [experts, rows, cols] = info.shape[..] else {
        panic!(
            "{name}: shape {:?} is not [experts, rows, cols]",
            info.shape
        );
    };
    let (experts, rows, cols) = (experts as usize, rows as usize, cols as usize);
    let quant = CpuQuant::from_quant_type(info.quant_type)
        .unwrap_or_else(|| panic!("{name}: qt {} has no host decoder", info.quant_type));
    assert!(!quant.is_planar(), "{name}: planar qt {}", info.quant_type);
    let (group_bytes, group_elems) = (quant.group_bytes(), quant.group_elems());
    assert_eq!(bytes.len() % experts, 0, "{name}: bytes per expert");
    let per_expert = bytes.len() / experts;
    assert_eq!(
        per_expert,
        rows * cols / group_elems * group_bytes,
        "{name}: per-expert bytes vs shape"
    );
    let slab = &bytes[expert * per_expert..(expert + 1) * per_expert];
    let mut values = vec![0.0f32; rows * cols];
    for (group, out) in slab
        .chunks_exact(group_bytes)
        .zip(values.chunks_exact_mut(group_elems))
    {
        dequant_group(quant, group, out);
    }
    MtpMatrix::new(rows, cols, values).expect("expert matrix")
}

/// The trunk embedding row of `token` (Q8F16 / BF16 table).
fn embedding_row(hfq: &HfqFile, token: u32) -> Vec<f32> {
    let info = hfq
        .tensors()
        .iter()
        .find(|t| t.name.ends_with("embed_tokens.weight") && !t.name.starts_with("mtp."))
        .expect("artifact has no trunk embedding");
    let (_, bytes) = hfq.tensor_data(&info.name).expect("embedding bytes");
    let rows = info.shape[0] as usize;
    assert_eq!(info.shape[1] as usize, HIDDEN);
    assert!((token as usize) < rows);
    let stride = bytes.len() / rows;
    let row = &bytes[token as usize * stride..(token as usize + 1) * stride];
    assert!(
        matches!(info.quant_type, 3 | 16),
        "embedding qt {} is not a dense format",
        info.quant_type
    );
    let values = dequantize_weight_to_f32(info.quant_type, row, HIDDEN);
    assert_eq!(values.len(), HIDDEN);
    values
}

/// Every head tensor except the routed experts, which are zero placeholders
/// until the router has chosen (`experts` below).
fn head_weights(hfq: &HfqFile) -> ReferenceMtpWeights {
    let sa = format!("{L0}self_attn");
    let mlp = format!("{L0}mlp");
    let router = matrix(hfq, &format!("{mlp}.gate.weight"));
    let experts = router.rows;
    let placeholder_gate_up = || MtpMatrix::new(2, HIDDEN, vec![0.0; 2 * HIDDEN]).unwrap();
    let placeholder_down = || MtpMatrix::new(HIDDEN, 1, vec![0.0; HIDDEN]).unwrap();
    ReferenceMtpWeights {
        pre_fc_norm_embedding: vector(hfq, "mtp.pre_fc_norm_embedding.weight"),
        pre_fc_norm_hidden: vector(hfq, "mtp.pre_fc_norm_hidden.weight"),
        fc_embedding: matrix(hfq, "mtp.fc_embedding.weight"),
        fc_hidden: matrix(hfq, "mtp.fc_hidden.weight"),
        attention: MtpAttentionWeights {
            q_proj: matrix(hfq, &format!("{sa}.q_proj.weight")),
            k_proj: matrix(hfq, &format!("{sa}.k_proj.weight")),
            v_proj: matrix(hfq, &format!("{sa}.v_proj.weight")),
            o_proj: matrix(hfq, &format!("{sa}.o_proj.weight")),
            q_norm: vector(hfq, &format!("{sa}.q_norm.weight")),
            k_norm: vector(hfq, &format!("{sa}.k_norm.weight")),
            index_qk_proj: matrix(hfq, &format!("{sa}.indexer.index_qk_proj.weight")),
            index_q_norm: vector(hfq, &format!("{sa}.indexer.q_layernorm.weight")),
            index_k_norm: vector(hfq, &format!("{sa}.indexer.k_layernorm.weight")),
        },
        attention_hyper: hyper(hfq, &format!("{L0}attn_hyper_connection")),
        moe: MtpMoeWeights {
            router,
            routed_gate_up: (0..experts).map(|_| placeholder_gate_up()).collect(),
            routed_down: (0..experts).map(|_| placeholder_down()).collect(),
            shared_gate_scalar: vector(hfq, &format!("{mlp}.shared_expert_gate.weight")),
            shared_gate: matrix(hfq, &format!("{mlp}.shared_expert.gate_proj.weight")),
            shared_up: matrix(hfq, &format!("{mlp}.shared_expert.up_proj.weight")),
            shared_down: matrix(hfq, &format!("{mlp}.shared_expert.down_proj.weight")),
        },
        mlp_hyper: hyper(hfq, &format!("{L0}mlp_hyper_connection")),
        final_hyper: MtpFinalHyperWeights {
            norm: vector(hfq, "mtp.hyper_connection_mixer.hc_norm.weight"),
            down: matrix(
                hfq,
                "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
            ),
            up: matrix(hfq, "mtp.hyper_connection_mixer.input_mix_weight_up.weight"),
        },
        qsa_geometry: MtpQsaGeometry::native(MAX_SEQ),
    }
}

struct Step {
    token: u32,
    /// `Some` for a fill row, `None` for the chained step.
    backbone: Option<Vec<f32>>,
}

/// The reference over the whole sequence: `(collapsed, wide, experts)` per step.
fn run_reference(
    weights: &ReferenceMtpWeights,
    hfq: &HfqFile,
    steps: &[Step],
) -> Vec<(Vec<f32>, Vec<f32>, [usize; MTP_TOP_K])> {
    let mut state = Qwen4MtpState::new(weights.qsa_geometry).expect("state");
    let mut previous_wide: Vec<f32> = Vec::new();
    let mut out = Vec::new();
    for (position, step) in steps.iter().enumerate() {
        let backbone = step
            .backbone
            .clone()
            .unwrap_or_else(|| previous_wide.clone());
        let result = mtp_forward_step(
            weights,
            &mut state,
            MtpForwardInput {
                token_embedding: &embedding_row(hfq, step.token),
                backbone_hidden: &backbone,
                position,
                step_index: usize::from(step.backbone.is_none()),
            },
        )
        .expect("mtp_forward_step");
        previous_wide = result.wide_hidden.clone();
        out.push((
            result.collapsed_hidden,
            result.wide_hidden,
            result.selected_experts,
        ));
    }
    out
}

fn cosine_and_rel_l2(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb, mut diff) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        diff += (x - y) * (x - y);
    }
    (dot / (na.sqrt() * nb.sqrt()), (diff / nb).sqrt())
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_REF_PIN_MODEL"]
fn mtp_head_reference_pin_hw() {
    assert!(
        std::env::var_os("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS").is_some(),
        "set a fixed HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS (a discrete card cannot hold every expert)"
    );
    let path = model_path();
    let hfq = HfqFile::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));

    let mut rng = Rng(0x4D54_5052_4546_5049);
    let mut steps: Vec<Step> = FILL_TOKENS
        .iter()
        .map(|&token| Step {
            token,
            backbone: Some(rng.normal(BRANCHES * HIDDEN, 1.5)),
        })
        .collect();
    steps.push(Step {
        token: CHAIN_TOKEN,
        backbone: None,
    });

    // Routing depends on the MLP HC read, which precedes every expert but not
    // the previous step's expert output (the chained step consumes the head's
    // own wide). So run with placeholder experts, decode exactly the experts
    // the latest run selected, and repeat until every selected expert is the
    // real one.
    let mut weights = head_weights(&hfq);
    let (gate_up, down) = (
        format!("{L0}mlp.experts.gate_up_proj"),
        format!("{L0}mlp.experts.down_proj"),
    );
    let mut loaded = std::collections::BTreeSet::new();
    let reference = loop {
        let run = run_reference(&weights, &hfq, &steps);
        let missing: Vec<usize> = run
            .iter()
            .flat_map(|(_, _, experts)| experts.iter().copied())
            .filter(|expert| !loaded.contains(expert))
            .collect();
        if missing.is_empty() {
            break run;
        }
        assert!(
            loaded.len() <= steps.len() * MTP_TOP_K,
            "expert selection did not converge"
        );
        for expert in missing {
            if loaded.insert(expert) {
                weights.moe.routed_gate_up[expert] = expert_matrix(&hfq, &gate_up, expert);
                weights.moe.routed_down[expert] = expert_matrix(&hfq, &down, expert);
            }
        }
    };
    weights.validate().expect("reference weights validate");
    println!("MTP_REF_PIN experts_loaded={}", loaded.len());

    let cases: Vec<MtpHeadPinCase> = steps
        .iter()
        .zip(&reference)
        .map(|(step, (collapsed, _, _))| MtpHeadPinCase {
            token: step.token,
            backbone: step.backbone.clone(),
            reference_collapsed: collapsed.clone(),
        })
        .collect();
    let gpu = run_mtp_head_reference_pin_gpu(&path, MAX_SEQ, &cases)
        .unwrap_or_else(|e| panic!("GPU head: {e}"));
    assert_eq!(gpu.len(), steps.len());

    let mut failures = Vec::new();
    for (index, (gpu, (ref_collapsed, ref_wide, experts))) in gpu.iter().zip(&reference).enumerate()
    {
        let (collapsed_cos, collapsed_rel) = cosine_and_rel_l2(&gpu.collapsed, ref_collapsed);
        let (wide_cos, wide_rel) = cosine_and_rel_l2(&gpu.wide, ref_wide);
        let depth = if steps[index].backbone.is_some() {
            1
        } else {
            2
        };
        println!(
            "MTP_REF_PIN step={index} depth={depth} token={} draft_gpu={} draft_reference={} \
             collapsed_cos={collapsed_cos:.6} collapsed_rel_l2={collapsed_rel:.6} \
             wide_cos={wide_cos:.6} wide_rel_l2={wide_rel:.6} experts={experts:?}",
            steps[index].token, gpu.draft, gpu.draft_from_reference
        );
        if gpu.draft != gpu.draft_from_reference {
            failures.push(format!(
                "step {index} draft_{depth}: GPU head {} vs draft head on the reference hidden {}",
                gpu.draft, gpu.draft_from_reference
            ));
        }
        for (what, cos, rel) in [
            ("collapsed", collapsed_cos, collapsed_rel),
            ("wide", wide_cos, wide_rel),
        ] {
            if !(cos >= MIN_COSINE && rel <= MAX_REL_L2) {
                failures.push(format!(
                    "step {index} {what}: cosine {cos:.6} (>= {MIN_COSINE}), rel L2 {rel:.6} (<= {MAX_REL_L2})"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "reference vs GPU head:\n{}",
        failures.join("\n")
    );
}
