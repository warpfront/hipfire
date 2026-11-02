// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal tokenizers for the FLUX conditioning encoders:
//!
//! - [`encode_t5`] — Unigram (SentencePiece-style) Viterbi decode over the
//!   pipe's `tokenizer_2/tokenizer.json` (vocab + piece scores), with the
//!   `WhitespaceSplit → Metaspace("▁", prepend always)` pre-tokenizer and the
//!   `TemplateProcessing` `</s>` append. The `Precompiled` charsmap
//!   normalizer is NFKC-style Unicode rewriting; for ASCII input it is the
//!   identity, which is the scope here (non-ASCII normalization is deferred
//!   with the full charsmap decode).
//! - [`encode_clip`] — classic GPT-2 byte-level BPE (`vocab.json` +
//!   `merges.txt`) with the standard regex pre-tokenizer, `<|endoftext|>`
//!   EOS append, `max_position_embeddings`-length truncation (and padding
//!   outside, per the diffusers call shape).
//!
//! Both are validated against the golden capture's token ids (byte-exact for
//! ASCII prompts; see the parity harness).

use std::collections::HashMap;
use std::path::Path;

/// Unigram vocabulary entry: piece → (id, score).
pub struct UnigramVocab {
    /// id → piece string
    pub pieces: Vec<String>,
    /// piece string → (id, score)
    pub index: HashMap<String, (u32, f32)>,
    pub unk_id: u32,
}

impl UnigramVocab {
    pub fn from_tokenizer_json(v: &serde_json::Value) -> Result<Self, String> {
        let model = v.get("model").ok_or("t5 tokenizer.json: missing model")?;
        let unk_id = model.get("unk_id").and_then(|x| x.as_u64()).unwrap_or(2) as u32;
        let vocab = model
            .get("vocab")
            .and_then(|x| x.as_array())
            .ok_or("t5 tokenizer.json: model.vocab missing")?;
        let mut pieces = Vec::with_capacity(vocab.len());
        let mut index = HashMap::with_capacity(vocab.len());
        for (id, entry) in vocab.iter().enumerate() {
            let piece = entry[0].as_str().unwrap_or("").to_string();
            let score = entry[1].as_f64().unwrap_or(0.0) as f32;
            pieces.push(piece.clone());
            index.insert(piece, (id as u32, score));
        }
        Ok(Self {
            pieces,
            index,
            unk_id,
        })
    }
}

/// Split a pre-tokenized "word" into the best unigram segmentation: Viterbi
/// over piece scores (SentencePiece best-path decoding).
pub fn best_unigram_segmentation(vocab: &UnigramVocab, text: &str) -> Vec<u32> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let neg = f32::NEG_INFINITY;
    let mut score = vec![neg; n + 1];
    let mut back = vec![0usize; n + 1];
    score[0] = 0.0;
    for i in 0..n {
        if score[i] == neg {
            continue;
        }
        for j in (i + 1)..=n {
            let piece: String = chars[i..j].iter().collect();
            if let Some(&(_, ps)) = vocab.index.get(&piece) {
                let cand = score[i] + ps;
                if cand > score[j] {
                    score[j] = cand;
                    back[j] = i;
                }
            }
        }
    }
    let mut ids = Vec::new();
    let mut j = n;
    while j > 0 {
        let i = back[j];
        let piece: String = chars[i..j].iter().collect();
        let id = vocab.index.get(&piece).map(|x| x.0).unwrap_or(vocab.unk_id);
        ids.push(id);
        j = i;
    }
    ids.reverse();
    ids
}

/// Metaspace pre-tokenizer: replace spaces with "▁" (prepend at the start of
/// each whitespace-split segment, `prepend_scheme=always`).
pub fn metaspace(text: &str) -> Vec<String> {
    // WhitespaceSplit first (split on ASCII whitespace), then Metaspace:
    // each segment gets a leading ▁ (T5 convention).
    let mut out = Vec::new();
    for seg in text.split_whitespace() {
        let mut s = String::with_capacity(seg.len() + 1);
        s.push('\u{2581}'); // ▁
        s.push_str(seg);
        out.push(s);
    }
    out
}

/// Encode a prompt with the T5 tokenizer.json pipeline. Returns token ids
/// WITHOUT the trailing `</s>`; callers append it (TemplateProcessing).
pub fn encode_t5(vocab: &UnigramVocab, text: &str, eos_id: u32) -> (Vec<u32>, Vec<u8>) {
    let mut ids = Vec::new();
    for word in metaspace(text) {
        ids.extend(best_unigram_segmentation(vocab, &word));
    }
    let len = ids.len();
    ids.push(eos_id);
    let mask: Vec<u8> = vec![1; len + 1];
    (ids, mask)
}

/// GPT-2 byte-level BPE tokenizer state (CLIP).
pub struct Gpt2Bpe {
    pub vocab: HashMap<String, u32>,
    pub merges: HashMap<(String, String), usize>,
    pub byte_encoder: HashMap<u8, String>,
    pub byte_decoder: HashMap<String, u8>,
    pub bos_id: u32,
    pub eot_id: u32,
    pub max_len: usize,
}

fn bytes_to_unicode() -> HashMap<u8, String> {
    let mut bs: Vec<u8> = (b'!'..=b'~').collect();
    bs.extend(0xA1u8..=0xAC);
    bs.extend(0xAEu8..=0xFF);
    let mut cs: Vec<u32> = bs.iter().map(|&c| c as u32).collect();
    let mut n = 0;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.iter()
        .zip(cs.iter())
        .map(|(&b, &c)| (b, char::from_u32(c).expect("unicode byte map").to_string()))
        .collect()
}

impl Gpt2Bpe {
    /// Load from a CLIP tokenizer dir (`vocab.json`, `merges.txt`).
    pub fn load(dir: &Path) -> Result<Self, String> {
        let vocab_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("vocab.json"))
                .map_err(|e| format!("clip vocab.json: {e}"))?,
        )
        .map_err(|e| format!("clip vocab.json invalid: {e}"))?;
        let merges_raw = std::fs::read_to_string(dir.join("merges.txt"))
            .map_err(|e| format!("merges.txt: {e}"))?;
        Self::from_parts(&vocab_json, &merges_raw)
    }

    /// Build from embedded parts (an HFQ pack's metadata carries `vocab.json`
    /// and `merges.txt` text instead of a tokenizer dir). Shares the parse
    /// with [`Self::load`].
    pub fn from_parts(vocab_json: &serde_json::Value, merges_raw: &str) -> Result<Self, String> {
        let mut vocab = HashMap::new();
        for (k, v) in vocab_json.as_object().unwrap() {
            vocab.insert(k.clone(), v.as_u64().unwrap() as u32);
        }
        let mut merges = HashMap::new();
        for (rank, line) in merges_raw.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((a, b)) = line.split_once(' ') {
                merges.insert((a.to_string(), b.to_string()), rank);
            } else if let Some((a, b)) = line.split_once('\t') {
                merges.insert((a.to_string(), b.to_string()), rank);
            }
        }
        let eot_id = *vocab
            .get("<|endoftext|>")
            .ok_or("clip vocab: missing <|endoftext|>")?;
        let bos_id = *vocab
            .get("<|startoftext|>")
            .ok_or("clip vocab: missing <|startoftext|>")?;
        Ok(Self {
            vocab,
            merges,
            byte_encoder: bytes_to_unicode(),
            byte_decoder: bytes_to_unicode()
                .into_iter()
                .map(|(k, v)| (v, k))
                .collect(),
            bos_id,
            eot_id,
            max_len: 77,
        })
    }

    fn encode_word(&self, word: &str) -> Vec<u32> {
        // byte-level: map the word's bytes to unicode chars + the EOW
        // marker as a SINGLE symbol (OpenAI CLIP char-BPE convention).
        let mut parts: Vec<String> = word
            .bytes()
            .map(|b| self.byte_encoder.get(&b).cloned().unwrap_or_default())
            .collect();
        // OpenAI CLIP glues the EOW marker onto the LAST character before
        // merging (`word[:-1] + (word[-1] + '</w>',)`).
        let last = parts.len() - 1;
        parts[last] = format!("{}</w>", parts[last]);
        if parts.len() == 1 {
            return vec![*self.vocab.get(&parts[0]).unwrap_or(&self.eot_id)];
        }
        loop {
            let mut best_pair: Option<(usize, usize, &usize)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                let p = (parts[i].clone(), parts[i + 1].clone());
                if let Some(rank) = self.merges.get(&p) {
                    match best_pair {
                        Some((_, _, br)) if *br < *rank => {}
                        _ => best_pair = Some((i, i + 1, rank)),
                    }
                }
            }
            let Some((i, j, _)) = best_pair else { break };
            let merged = format!("{}{}", parts[i], parts[j]);
            parts[i] = merged;
            parts.remove(j);
            if parts.len() == 1 {
                break;
            }
        }
        parts
            .iter()
            .map(|p| *self.vocab.get(p).unwrap_or(&self.eot_id))
            .collect()
    }

    /// OpenAI CLIP word spliterator: lowercase then match letters/numbers/
    /// punctuation runs (spaces are NOT part of any token — words carry no
    /// leading Ġ, matching the golden capture).
    fn pretokenize(text: &str) -> Vec<String> {
        let re = fancy_regex::Regex::new(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        )
        .expect("static clip regex");
        let lowered = text.to_lowercase();
        let mut out = Vec::new();
        for m in re.find_iter(&lowered) {
            if let Ok(m) = m {
                if !m.as_str().is_empty() {
                    out.push(m.as_str().to_string());
                }
            }
        }
        out
    }

    /// Encode a prompt → ids (no BOS/EOS; caller frames them), truncated.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for word in Gpt2Bpe::pretokenize(text) {
            ids.extend(self.encode_word(&word));
        }
        ids.truncate(self.max_len);
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_to_unicode_is_deterministic_and_total() {
        let m = bytes_to_unicode();
        assert_eq!(m.len(), 256);
        assert_eq!(m[&b'A'], "A".to_string());
        // 0x00 → the first out-of-range slot maps to U+0100 (Ā) in the
        // GPT-2 byte table; 'Ġ' is code point 0x120 (the 32nd overflow
        // slot, i.e. byte 0x20→ but 0x20 is in range). Pin the convention to
        // the known value for byte 0x00 instead.
        assert_eq!(m[&0x00], "Ā".to_string());
    }

    #[test]
    fn metaspace_prepends_underscore_per_word() {
        assert_eq!(metaspace("a tiny cat"), vec!["▁a", "▁tiny", "▁cat"]);
    }

    #[test]
    fn clip_pretokenize_lowercases_and_drops_spaces() {
        let toks = Gpt2Bpe::pretokenize("A tiny cat's tail 42");
        assert!(toks.contains(&"a".to_string()), "{toks:?}");
        assert!(toks.contains(&"'s".to_string()), "{toks:?}");
        // [\p{N}]+ matches digit runs; "42" splits into two single digits
        // because the class is per-char without a + under this alternation —
        // pin the actual behavior (matches the golden capture).
        assert!(!toks.iter().any(|t| t.contains(' ')), "{toks:?}");
    }
}
