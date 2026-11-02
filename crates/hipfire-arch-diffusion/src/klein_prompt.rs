// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FLUX.2 Klein prompt template, tokenization and right-padding.
//!
//! Klein conditions on a Qwen3 causal-LM text encoder (see [`crate::qwen3`]),
//! which expects the ComfyUI chat-template wrapping and a right-padded,
//! never-truncated id/mask pair (ComfyUI pads to at least a minimum length
//! rather than a fixed one, and never truncates a long prompt).

use hipfire_runtime::tokenizer::Tokenizer;

/// Production pad id: Qwen3 tokenizer `<|endoftext|>`.
pub const KLEIN_PAD_ID: u32 = 151643;
/// Production minimum sequence length ComfyUI's Klein node pads to.
pub const KLEIN_MIN_LEN: usize = 512;

/// Wrap a raw prompt in the ComfyUI Klein chat template (Qwen3 chatml with
/// an empty `<think>` block — Klein does not use chain-of-thought).
pub fn klein_template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// Tokenized, right-padded Klein prompt: `ids`/`mask` are always the same
/// length, at least `min_len`. `mask[i] == 0` marks a pad position.
pub struct KleinPrompt {
    pub ids: Vec<u32>,
    pub mask: Vec<u8>,
}

/// Encode with the runtime tokenizer, right-pad with `pad_id` to at least
/// `min_len`. No truncation (ComfyUI rule) — a prompt encoding longer than
/// `min_len` is returned in full, unpadded.
pub fn encode_klein_prompt(
    tok: &Tokenizer,
    prompt: &str,
    pad_id: u32,
    min_len: usize,
) -> KleinPrompt {
    let mut ids = tok.encode(&klein_template(prompt));
    let real = ids.len();
    let mut mask = vec![1u8; real];
    if real < min_len {
        ids.resize(min_len, pad_id);
        mask.resize(min_len, 0);
    }
    KleinPrompt { ids, mask }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_matches_comfyui_klein_tokenizer() {
        assert_eq!(
            klein_template("a cat"),
            "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn encode_pads_right_to_min_len_and_masks_pads() {
        let tok = hipfire_runtime::tokenizer::Tokenizer::from_hf_json(
            r#"{"model":{"type":"BPE","vocab":{"a":0,"b":1,"<|im_start|>":2,"<|im_end|>":3,"<pad>":4},"merges":[]},
                "added_tokens":[{"id":2,"content":"<|im_start|>"},{"id":3,"content":"<|im_end|>"},{"id":4,"content":"<pad>"}]}"#).unwrap();
        let p = encode_klein_prompt(&tok, "ab", 4, 8);
        assert_eq!(p.ids.len(), 8);
        assert_eq!(p.mask.len(), 8);
        let real = p.mask.iter().filter(|m| **m == 1).count();
        assert!(real >= 4 && real < 8);
        assert!(p.ids[real..].iter().all(|&i| i == 4));
        assert!(p.mask[real..].iter().all(|&m| m == 0));
        assert_eq!(p.ids[0], 2, "starts with <|im_start|>");
    }

    #[test]
    fn encode_does_not_truncate_long_prompts() {
        let tok = hipfire_runtime::tokenizer::Tokenizer::from_hf_json(
            r#"{"model":{"type":"BPE","vocab":{"a":0,"b":1,"<|im_start|>":2,"<|im_end|>":3,"<pad>":4},"merges":[]},
                "added_tokens":[{"id":2,"content":"<|im_start|>"},{"id":3,"content":"<|im_end|>"},{"id":4,"content":"<pad>"}]}"#).unwrap();
        let long = "a ".repeat(40);
        let p = encode_klein_prompt(&tok, &long, 4, 8);
        assert!(p.ids.len() > 8);
        assert!(p.mask.iter().all(|&m| m == 1));
    }
}
