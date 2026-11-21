// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gates for the cached multi-token CPU path (brief 04).
//!
//! * Gate 1: at `pos == 0` on a fresh state, `forward_token_cached` is
//!   bit-identical to the untouched `forward_token` (oracle parity itself is
//!   owned by `dump_raw_logits_token0` in the lib: r = 0.9999999968,
//!   argmax 31274).
//! * Gate 2: a 3-token prefix decoded incrementally equals a fresh-state
//!   replay of the full prefix (catches cache indexing / conv-history /
//!   window errors without an oracle).
//! * Gate 3: greedy 8-token generation on F32 for `The capital of France is`
//!   reproduces the golden thinking-phase prefix of
//!   `D:\hipfire-models\inkling2-golden\g1.txt`.
//!
//! The artifact-gated tests need `D:\hipfire-models\gguf-f32.bin` (104 GB)
//! and run in `--release` (F32 matvec is ~10x vectorised there).

use hipfire_arch_inkling2::config::Inkling2Config;
use hipfire_arch_inkling2::forward::{self, CpuDecodeState};
use hipfire_arch_inkling2::weights::{GgufF32, WeightSource};

/// Copy of the crate's `TEST_METADATA` (integration tests cannot see
/// `pub(crate)` items); the parse test below guards drift.
const TEST_METADATA: &str = r#"{"config":{"eos_token_id":200006,"text_config":{
    "num_hidden_layers":24,"hidden_size":2304,"vocab_size":201024,
    "unpadded_vocab_size":200058,"num_attention_heads":24,"num_key_value_heads":8,
    "head_dim":128,"sliding_window_size":1024,"d_rel":16,"rel_extent":1024,
    "qk_l2_clip_threshold":22.627416997969522,"rel_l2_clip_threshold":8.0,
    "log_scaling_n_floor":8192,"log_scaling_alpha":0.1,"sconv_kernel_size":4,
    "n_layers_per_attn_res_block":6,"final_logit_softcapping":30.0,
    "n_routed_experts":128,"num_experts_per_tok":8,"intermediate_size":1152,
    "route_scale":8.0,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"rms_norm_eps":1e-6,
    "local_layer_ids":[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
    "loop_config":{"core_start":6,"core_end":18,"num_passes":3,"gate_rank":64,"is_varloop":false}
}}}"#;

const F32_BIN: &str = r"D:\hipfire-models\gguf-f32.bin";

fn load() -> (Inkling2Config, GgufF32) {
    let cfg = Inkling2Config::from_metadata_json(TEST_METADATA).unwrap();
    let w = GgufF32::open(std::path::Path::new(F32_BIN)).unwrap();
    (cfg, w)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn test_metadata_matches_crate_shape() {
    let cfg = Inkling2Config::from_metadata_json(TEST_METADATA).unwrap();
    assert_eq!(cfg.loop_cfg.n_slot(), 48);
    assert_eq!(cfg.n_head, 24);
    assert_eq!(cfg.n_head_kv, 8);
    assert_eq!(cfg.n_extent, 1024);
    assert_eq!(cfg.sliding_window, 1024);
    let st = CpuDecodeState::new(&cfg, 128).unwrap();
    assert_eq!(st.slots.len(), 48);
}

#[test]
#[ignore = "requires D:\\hipfire-models\\gguf-f32.bin (104 GB); pos-0 cached-vs-uncached identity"]
fn cached_matches_forward_at_pos0() {
    let (cfg, w) = load();
    // First embed row: the same row the committed dump_raw_logits_token0
    // parity run feeds as "token 0".
    let embed = w.f32("model.llm.embed.weight").unwrap().1;
    let tok = embed[..cfg.n_embd].to_vec();
    let reference = forward::forward_token(&w, &cfg, &tok, 0, None, None).unwrap();
    let mut st = CpuDecodeState::new(&cfg, 128).unwrap();
    let t0 = std::time::Instant::now();
    let cached = forward::forward_token_cached(&w, &cfg, &mut st, &tok, 0).unwrap();
    let dt = t0.elapsed();
    let m = max_abs(&reference, &cached);
    eprintln!("pos0 cached vs forward: max|d|={m:e} wall={dt:?}");
    assert!(
        m == 0.0,
        "pos-0 cached path must be bit-identical to forward_token, got max|d|={m:e}"
    );
}

#[test]
#[ignore = "requires D:\\hipfire-models\\gguf-f32.bin (104 GB); 3-token incremental-vs-scratch consistency"]
fn incremental_matches_scratch_3_token_prefix() {
    // First three ids of the gate-3 prompt ("The capital of").
    let prefix: [u32; 3] = [371, 151451, 149879];
    let (cfg, w) = load();
    let embed = w.f32("model.llm.embed.weight").unwrap().1;
    let row = |id: u32| embed[id as usize * cfg.n_embd..(id as usize + 1) * cfg.n_embd].to_vec();
    // Incremental: one persistent state over the prefix.
    let mut st = CpuDecodeState::new(&cfg, 128).unwrap();
    let mut inc = Vec::new();
    for (i, &id) in prefix.iter().enumerate() {
        inc.push(forward::forward_token_cached(&w, &cfg, &mut st, &row(id), i).unwrap());
    }
    // Scratch: a fresh state replaying the full prefix (never touches the
    // incremental state's ring buffer or conv history).
    let mut st2 = CpuDecodeState::new(&cfg, 128).unwrap();
    let mut scr = Vec::new();
    for (i, &id) in prefix.iter().enumerate() {
        scr.push(forward::forward_token_cached(&w, &cfg, &mut st2, &row(id), i).unwrap());
    }
    for (pos, (a, b)) in inc.iter().zip(&scr).enumerate() {
        let m = max_abs(a, b);
        eprintln!("pos{pos} incremental vs scratch: max|d|={m:e}");
        assert!(
            m <= 1e-4,
            "pos{pos} incremental/scratch diverge: max|d|={m:e} (bar 1e-4)"
        );
    }
}

#[test]
#[ignore = "requires D:\\hipfire-models\\gguf-f32.bin (104 GB); greedy golden reproduction, run in --release"]
fn greedy_8_tokens_match_golden() {
    // "The capital of France is" (md5 1aacd3c05cf9695cc799acc59581938d).
    let prompt: [u32; 5] = [371, 151451, 149879, 151194, 149884];
    let (cfg, w) = load();
    let t0 = std::time::Instant::now();
    let ids = forward::generate_cpu(&w, &cfg, &prompt, 8, 128).unwrap();
    let wall = t0.elapsed();
    eprintln!(
        "greedy ids={ids:?} wall={wall:?} per-token={:?} ({} forwards)",
        wall / 13,
        13
    );
    // Expected: the golden continuation right after the prompt,
    // "\n\n[Start thinking]\n\nThe user", encoded with the model's
    // tokenizer.json: [68014, 58, 3214, 154762, 60, 68014, 371, 150409].
    assert_eq!(
        ids,
        vec![68014, 58, 3214, 154762, 60, 68014, 371, 150409],
        "greedy output diverged from the golden thinking-phase prefix"
    );
}
