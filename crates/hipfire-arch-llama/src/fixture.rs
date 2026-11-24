// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dense Q8 attention fixture (B2.x): a tiny seeded LLaMA-family HFQM that the
//! production carrier loads (`load_bundle`, route `ManifestPlainLlama`) with
//! every 2D weight in `Q8F16` (quant_type 3), the layout `hipfire-quantize
//! --format q8` writes.
//!
//! # Q8F16 payload layout
//!
//! Source of truth: `hipfire-quantize` `quantize_q8f16` / `dequantize_hfq_q8f16`
//! and the `Q8_0` GEMV (`gemv_q8_0`). A tensor of `n` elements (row-major, rows
//! of `K` elements, `K % 32 == 0`) is `n / 32` blocks of 34 bytes:
//!
//! ```text
//! [0..2)   f16 scale, little endian
//! [2..34)  32 x i8 quants
//! element = quant as f32 * scale          (exact in f32: 7 + 11 bits)
//! ```
//!
//! Blocks run along the flattened row-major element order, so with
//! `K % 32 == 0` no block straddles a row. HFQM index entry: `quant_type = 3`,
//! `group_size = 32`, shape `[rows, K]`. Norm vectors are F16 (`quant_type 1`,
//! `group_size 0`), as the quantizer writes them.
//!
//! # What stays real-size and what shrinks
//!
//! Kept at the dimensions a real LLaMA-class model has (B2.0 §1): `head_dim`
//! 128, GQA ratio 2 (4 query / 2 KV heads), every projection `K % 32 == 0`,
//! llama half-split RoPE with a non-default `rope_theta`. Shrunk: depth (2
//! layers), vocabulary (64) and context (64 positions). `hidden_size` 256 is
//! not tied to `heads * head_dim` (512), as in Qwen3-0.6B.
//!
//! # Determinism
//!
//! Every byte comes from integer arithmetic on a SplitMix64 stream keyed by
//! `(FIXTURE_SEED, FNV-1a(tensor name))`. There is no float math, no clock, no
//! host-dependent ordering, so the file is byte-identical on every host.

use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
use std::path::Path;

/// HFQM `arch_id` of the LLaMA family (`arch_mapping::quantize_arch_id("llama")`).
pub const ARCH_ID_LLAMA: u32 = 0;
/// Q8F16 quant type (`QuantType::Q8F16`).
pub const QT_Q8F16: u8 = 3;
/// F16 quant type (`QuantType::F16`).
pub const QT_F16: u8 = 1;
/// Weights per Q8F16 block.
pub const Q8_BLOCK: usize = 32;
/// Bytes per Q8F16 block.
pub const Q8_BLOCK_BYTES: usize = 34;

/// Seed of the whole fixture. Changing it changes the pinned hash.
pub const FIXTURE_SEED: u64 = 0x42_32_78_51_38_00_00_01;

pub const DIM: usize = 256;
pub const N_LAYERS: usize = 2;
pub const N_HEADS: usize = 4;
pub const N_KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 128;
pub const INTERMEDIATE: usize = 512;
pub const VOCAB: usize = 64;
pub const MAX_POSITIONS: usize = 64;
pub const ROPE_THETA: f64 = 500000.0;
pub const RMS_NORM_EPS: f64 = 1e-5;
pub const BOS_TOKEN: u32 = 1;
pub const EOS_TOKEN: u32 = 2;

/// Weight-scale exponents (`scale = 2^-e * (1 + m/16)`, `m` in 0..16): chosen so
/// the decoded weights have std ~ `gain / sqrt(K)` and the residual stream and
/// logits stay O(1) through the two layers.
const PROJ_SCALE_EXP: u32 = 10;
const EMBED_SCALE_EXP: u32 = 6;
const HEAD_SCALE_EXP: u32 = 9;

/// The tiny tokenizer's merge table (priority order). Every merge result is a
/// vocabulary entry; see [`tokenizer_json`].
const MERGES: [(&str, &str); 24] = [
    ("\u{2581}", "t"),
    ("t", "h"),
    ("\u{2581}t", "h"),
    ("h", "e"),
    ("\u{2581}th", "e"),
    ("i", "n"),
    ("e", "r"),
    ("a", "n"),
    ("r", "e"),
    ("o", "n"),
    ("a", "t"),
    ("e", "n"),
    ("n", "d"),
    ("t", "i"),
    ("e", "s"),
    ("o", "r"),
    ("t", "e"),
    ("o", "f"),
    ("e", "d"),
    ("i", "s"),
    ("i", "t"),
    ("a", "l"),
    ("a", "r"),
    ("s", "t"),
];

/// Fixture tensors and metadata, before they are written to an HFQM file.
pub struct DenseQ8Fixture {
    pub metadata_json: String,
    pub tensors: Vec<HfqMemTensor>,
}

impl DenseQ8Fixture {
    /// Tensor by exact name.
    pub fn tensor(&self, name: &str) -> &HfqMemTensor {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("dense Q8 fixture has no tensor {name}"))
    }
}

/// SplitMix64 (Steele, Lea, Flood): integer only, so identical on every host.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn tensor_rng(name: &str) -> SplitMix64 {
    SplitMix64(FIXTURE_SEED ^ fnv1a64(name.as_bytes()))
}

/// F16 bits of `2^-exp * (1 + mant/16)`: a normal f16 built from its fields, so
/// no float conversion is involved.
fn scale_bits(exp: u32, mant: u64) -> u16 {
    debug_assert!(mant < 16 && (1..=14).contains(&exp));
    (((15 - exp) as u16) << 10) | ((mant as u16) << 6)
}

/// One Q8F16 tensor `[rows, k]`. Quants are the sum of three uniform draws in
/// `[-40, 39]` (bell shaped, |q| <= 120), with one element per block forced to
/// +-127 so each block is full scale, as `quantize_q8f16` (`scale = max|w|/127`)
/// produces. The scale is `2^-scale_exp * (1 + m/16)` with `m` drawn per block.
fn q8f16_tensor(name: &str, rows: usize, k: usize, scale_exp: u32) -> HfqMemTensor {
    assert_eq!(k % Q8_BLOCK, 0, "{name}: Q8F16 needs K % 32 == 0");
    let blocks = rows * k / Q8_BLOCK;
    let mut rng = tensor_rng(name);
    let mut data = Vec::with_capacity(blocks * Q8_BLOCK_BYTES);
    for _ in 0..blocks {
        let mant = rng.next() % 16;
        data.extend_from_slice(&scale_bits(scale_exp, mant).to_le_bytes());
        let pin = (rng.next() % Q8_BLOCK as u64) as usize;
        let pin_sign = rng.next() & 1;
        let mut quants = [0i8; Q8_BLOCK];
        for (i, q) in quants.iter_mut().enumerate() {
            let bits = rng.next();
            let draw = |shift: u32| ((bits >> shift) % 80) as i32 - 40;
            *q = (draw(0) + draw(20) + draw(40)) as i8;
            if i == pin {
                *q = if pin_sign == 0 { 127 } else { -127 };
            }
        }
        data.extend(quants.iter().map(|&q| q as u8));
    }
    HfqMemTensor {
        name: name.into(),
        quant_type: QT_Q8F16,
        shape: vec![rows as u32, k as u32],
        group_size: Q8_BLOCK as u32,
        data,
    }
}

/// One F16 norm vector with weights `1 + n/16`, `n` in `-4..=4`: exact in f16.
fn f16_norm_tensor(name: &str, len: usize) -> HfqMemTensor {
    let mut rng = tensor_rng(name);
    let mut data = Vec::with_capacity(len * 2);
    for _ in 0..len {
        let n = (rng.next() % 9) as i32 - 4;
        // 1 + n/16 lies in [0.75, 1.25]: exponent 14 below 1.0, 15 from 1.0.
        let bits: u16 = if n >= 0 {
            (15 << 10) | ((n as u16) << 6)
        } else {
            // 0.75 = 1.5 * 2^-1 -> mant 512, 0.8125 -> 640, 0.875 -> 768, 0.9375 -> 896.
            (14 << 10) | (((16 + 2 * n) as u16) << 6)
        };
        data.extend_from_slice(&bits.to_le_bytes());
    }
    HfqMemTensor {
        name: name.into(),
        quant_type: QT_F16,
        shape: vec![len as u32],
        group_size: 0,
        data,
    }
}

/// The fixture's tokenizer, as an HF `tokenizer.json` string: 64 ids =
/// `<unk>`, `<s>`, `</s>`, `▁`, `a`..`z`, `0`..`9`, then one entry per merge.
/// SentencePiece-style (`▁`) BPE, so the daemon can tokenize a text prompt.
pub fn tokenizer_json() -> String {
    let mut vocab: Vec<String> = vec![
        "<unk>".into(),
        "<s>".into(),
        "</s>".into(),
        "\u{2581}".into(),
    ];
    vocab.extend(('a'..='z').map(String::from));
    vocab.extend(('0'..='9').map(String::from));
    for (left, right) in MERGES {
        vocab.push(format!("{left}{right}"));
    }
    assert_eq!(
        vocab.len(),
        VOCAB,
        "tokenizer vocabulary must fill the model vocabulary"
    );
    let entries: Vec<String> = vocab
        .iter()
        .enumerate()
        .map(|(id, token)| format!("\"{token}\":{id}"))
        .collect();
    let merges: Vec<String> = MERGES
        .iter()
        .map(|(left, right)| format!("\"{left} {right}\""))
        .collect();
    let added: Vec<String> = ["<unk>", "<s>", "</s>"]
        .iter()
        .enumerate()
        .map(|(id, token)| {
            format!(
                "{{\"id\":{id},\"content\":\"{token}\",\"special\":true,\"single_word\":false,\
                 \"lstrip\":false,\"rstrip\":false,\"normalized\":false}}"
            )
        })
        .collect();
    format!(
        "{{\"version\":\"1.0\",\"added_tokens\":[{}],\"normalizer\":null,\"pre_tokenizer\":null,\
         \"model\":{{\"type\":\"BPE\",\"unk_token\":\"<unk>\",\"vocab\":{{{}}},\"merges\":[{}]}}}}",
        added.join(","),
        entries.join(","),
        merges.join(",")
    )
}

/// Minimal JSON string escaping for the embedded tokenizer document.
fn json_string(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 2);
    out.push('"');
    for ch in raw.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// HFQM `metadata_json`: the LLaMA `config` block the loader parses
/// (`llama_config_from_value`), the embedded tokenizer, and fixture provenance.
pub fn metadata_json() -> String {
    format!(
        "{{\"architecture\":\"llama\",\"config\":{{\"model_type\":\"llama\",\
         \"hidden_size\":{DIM},\"num_hidden_layers\":{N_LAYERS},\
         \"num_attention_heads\":{N_HEADS},\"num_key_value_heads\":{N_KV_HEADS},\
         \"intermediate_size\":{INTERMEDIATE},\"vocab_size\":{VOCAB},\"head_dim\":{HEAD_DIM},\
         \"rms_norm_eps\":0.00001,\"max_position_embeddings\":{MAX_POSITIONS},\
         \"rope_theta\":{ROPE_THETA:.1},\"bos_token_id\":{BOS_TOKEN},\"eos_token_id\":{EOS_TOKEN}}},\
         \"generation_config\":{{\"bos_token_id\":{BOS_TOKEN},\"eos_token_id\":{EOS_TOKEN}}},\
         \"fixture\":{{\"id\":\"dense-q8-llama\",\"format\":\"q8f16\",\"seed\":{FIXTURE_SEED},\
         \"generator\":\"hipfire-arch-llama/src/fixture.rs\"}},\
         \"tokenizer\":{}}}",
        json_string(&tokenizer_json())
    )
}

/// Build the fixture in memory. Pure function of the constants in this module.
pub fn dense_q8_fixture() -> DenseQ8Fixture {
    let kv_dim = N_KV_HEADS * HEAD_DIM;
    let q_dim = N_HEADS * HEAD_DIM;
    let mut tensors = vec![
        q8f16_tensor("model.embed_tokens.weight", VOCAB, DIM, EMBED_SCALE_EXP),
        f16_norm_tensor("model.norm.weight", DIM),
        q8f16_tensor("lm_head.weight", VOCAB, DIM, HEAD_SCALE_EXP),
    ];
    for layer in 0..N_LAYERS {
        let p = |suffix: &str| format!("model.layers.{layer}.{suffix}");
        tensors.extend([
            f16_norm_tensor(&p("input_layernorm.weight"), DIM),
            q8f16_tensor(&p("self_attn.q_proj.weight"), q_dim, DIM, PROJ_SCALE_EXP),
            q8f16_tensor(&p("self_attn.k_proj.weight"), kv_dim, DIM, PROJ_SCALE_EXP),
            q8f16_tensor(&p("self_attn.v_proj.weight"), kv_dim, DIM, PROJ_SCALE_EXP),
            q8f16_tensor(&p("self_attn.o_proj.weight"), DIM, q_dim, PROJ_SCALE_EXP),
            f16_norm_tensor(&p("post_attention_layernorm.weight"), DIM),
            q8f16_tensor(
                &p("mlp.gate_proj.weight"),
                INTERMEDIATE,
                DIM,
                PROJ_SCALE_EXP,
            ),
            q8f16_tensor(&p("mlp.up_proj.weight"), INTERMEDIATE, DIM, PROJ_SCALE_EXP),
            q8f16_tensor(
                &p("mlp.down_proj.weight"),
                DIM,
                INTERMEDIATE,
                PROJ_SCALE_EXP,
            ),
        ]);
    }
    DenseQ8Fixture {
        metadata_json: metadata_json(),
        tensors,
    }
}

/// Write the fixture as an HFQM package at `path` through the production
/// writer (`write_hfqm_package_mem`, `arch_id` 0).
pub fn write_dense_q8_fixture(path: &Path) -> std::io::Result<()> {
    let fixture = dense_q8_fixture();
    write_hfqm_package_mem(
        path,
        ARCH_ID_LLAMA,
        &fixture.metadata_json,
        &fixture.tensors,
    )
}

/// The three fixed prompts of the GPU-vs-CPU row (I10), as token ids. Lengths
/// 5, 11 and 19 sit off every kernel tile grid; all three start with BOS. The
/// ids are fixed lists; `cpu_reference::tests::prompt_set_has_near_tie_and_separated_picks`
/// pins that, scored by the CPU reference, the set holds both near-tie picks
/// (top-1/top-2 gap under 0.15) and well-separated picks (gap over 1.0).
pub const PROMPTS: [&[u32]; 3] = [
    &[1, 9, 22, 41, 7],
    &[1, 15, 33, 4, 58, 27, 12, 50, 3, 36, 19],
    &[
        1, 44, 8, 61, 25, 13, 30, 5, 47, 18, 53, 10, 39, 23, 6, 56, 31, 14, 42,
    ],
];

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;

    fn write_bytes() -> Vec<u8> {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        write_dense_q8_fixture(file.path()).expect("write fixture");
        std::fs::read(file.path()).expect("read fixture back")
    }

    #[test]
    fn fixture_bytes_are_deterministic() {
        let first = write_bytes();
        let second = write_bytes();
        assert_eq!(first, second, "same bytes twice");
        // The in-memory build is a pure function too.
        let a = dense_q8_fixture();
        let b = dense_q8_fixture();
        assert_eq!(a.metadata_json, b.metadata_json);
        for (x, y) in a.tensors.iter().zip(&b.tensors) {
            assert_eq!(x.data, y.data, "{}", x.name);
        }
    }

    /// The digest pinned in `fixtures/pins.toml` (`model/dense-q8-llama.hfq`).
    /// Any change to the generator, the constants or the HFQM writer moves it,
    /// and must re-pin in the same commit.
    const PINNED_MD5: &str = "fa76f3ee8c330b82086ae3f4d471182a";
    const PINNED_SIZE: usize = 1_712_640;

    #[test]
    fn fixture_matches_the_pinned_digest() {
        let bytes = write_bytes();
        assert_eq!(bytes.len(), PINNED_SIZE);
        assert_eq!(format!("{:x}", md5::compute(&bytes)), PINNED_MD5);
    }

    #[test]
    fn q8_payload_size_and_every_linear_is_q8() {
        let fixture = dense_q8_fixture();
        let mut q8 = 0usize;
        for t in &fixture.tensors {
            let elements: usize = t.shape.iter().map(|&d| d as usize).product();
            match t.quant_type {
                QT_Q8F16 => {
                    q8 += 1;
                    assert_eq!(t.shape.len(), 2, "{}", t.name);
                    assert_eq!(t.shape[1] as usize % Q8_BLOCK, 0, "{}", t.name);
                    assert_eq!(t.group_size, 32, "{}", t.name);
                    assert_eq!(
                        t.data.len(),
                        elements / Q8_BLOCK * Q8_BLOCK_BYTES,
                        "{}",
                        t.name
                    );
                }
                QT_F16 => {
                    assert_eq!(t.shape.len(), 1, "{}", t.name);
                    assert_eq!(t.data.len(), elements * 2, "{}", t.name);
                }
                other => panic!("{}: unexpected quant_type {other}", t.name),
            }
        }
        // embed + lm_head + 7 projections per layer.
        assert_eq!(q8, 2 + 7 * N_LAYERS);
    }

    #[test]
    fn q8_blocks_are_full_scale_with_finite_f16_scales() {
        let fixture = dense_q8_fixture();
        for t in fixture.tensors.iter().filter(|t| t.quant_type == QT_Q8F16) {
            for (i, block) in t.data.chunks_exact(Q8_BLOCK_BYTES).enumerate() {
                let bits = u16::from_le_bytes([block[0], block[1]]);
                let exp = (bits >> 10) & 0x1f;
                assert!(
                    (1..31).contains(&exp),
                    "{} block {i}: f16 scale not normal",
                    t.name
                );
                assert_eq!(bits >> 15, 0, "{} block {i}: negative scale", t.name);
                let max = block[2..]
                    .iter()
                    .map(|&b| (b as i8 as i32).abs())
                    .max()
                    .unwrap();
                assert_eq!(max, 127, "{} block {i}: not full scale", t.name);
            }
        }
    }

    #[test]
    fn container_round_trips_and_config_parses_through_the_runtime() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        write_dense_q8_fixture(file.path()).expect("write fixture");
        let hfq = HfqFile::open(file.path()).expect("open fixture");
        assert_eq!(hfq.arch_id, ARCH_ID_LLAMA);
        let fixture = dense_q8_fixture();
        for t in &fixture.tensors {
            let (info, data) = hfq
                .tensor_data_vec(&t.name)
                .unwrap_or_else(|| panic!("{} missing from the container", t.name));
            assert_eq!(info.quant_type, t.quant_type, "{}", t.name);
            assert_eq!(data, t.data, "{}", t.name);
        }
        let config = <crate::Llama as Architecture>::config_from_hfq(&hfq).expect("config");
        assert_eq!(config.dim, DIM);
        assert_eq!(config.hidden_dim, INTERMEDIATE);
        assert_eq!(config.n_layers, N_LAYERS);
        assert_eq!(config.n_heads, N_HEADS);
        assert_eq!(config.n_kv_heads, N_KV_HEADS);
        assert_eq!(config.head_dim, HEAD_DIM);
        assert_eq!(config.vocab_size, VOCAB);
        assert_eq!(config.max_seq_len, MAX_POSITIONS);
        assert_eq!(config.rope_freq_base, ROPE_THETA as f32);
        assert_eq!(config.bos_token, BOS_TOKEN);
        assert_eq!(config.eos_token, EOS_TOKEN);
        assert!(!config.has_qk_norm);
    }

    #[test]
    fn tokenizer_loads_from_the_metadata_and_round_trips() {
        let tokenizer = Tokenizer::from_hfq_metadata(&metadata_json()).expect("tokenizer");
        let ids = tokenizer.encode("the cat and the dog");
        assert!(!ids.is_empty());
        assert!(ids.iter().all(|&id| (id as usize) < VOCAB), "{ids:?}");
        assert_eq!(tokenizer.decode(&ids).trim(), "the cat and the dog");
    }

    #[test]
    fn prompts_fit_the_context_and_vocabulary() {
        for prompt in PROMPTS {
            assert!(prompt.len() >= 4, "batched prefill needs >= 4 tokens");
            assert!(prompt.len() < MAX_POSITIONS);
            assert_eq!(prompt[0], BOS_TOKEN);
            assert!(prompt.iter().all(|&t| (t as usize) < VOCAB));
        }
        let lens: Vec<usize> = PROMPTS.iter().map(|p| p.len()).collect();
        assert_eq!(lens, [5, 11, 19], "off-grid lengths");
    }
}
