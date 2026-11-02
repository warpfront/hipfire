// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Post-Dense host math: the sentence-transformers modules 2 and 3 plus the
//! MaxSim scorer — all pure f32, no GPU, unit-testable anywhere.
//!
//! Pipeline order (matching `modules.json`): Transformer → Dense →
//! MultiVectorMask → Normalize. The mask drops token vectors whose piece is
//! a skiplist word (punctuation, document side by default); Normalize
//! L2-normalizes each kept vector. Scores are late interaction:
//! `score(q, d) = Σ_i max_j (q_i · d_j)` over query tokens i and document
//! tokens j.

/// The pplx-embed-v2 `2_MultiVectorMask` skiplist: printable ASCII
/// punctuation. Queries keep punctuation (empty `skiplist_tasks` entry),
/// documents drop it.
pub const DEFAULT_SKIPLIST_WORDS: &[&str] = &[
    "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=",
    ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~",
];

pub fn default_skiplist_words() -> Vec<String> {
    DEFAULT_SKIPLIST_WORDS
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Build the set of token ids whose *piece* (as stored in the tokenizer
/// vocab, with the `Ġ`-style space marker stripped) is exactly a skiplist
/// word — the token-level view of sentence-transformers'
/// `skiplist_word_to_mask`, resolved once at load.
///
/// Qwen BPE encodes punctuation as single-character tokens (with an optional
/// leading space marker when punctuation follows a word), so exact piece
/// matching after stripping the marker and any residual whitespace matches
/// the reference behavior.
pub fn skiplist_token_ids(
    vocab: &dyn VocabPieces,
    skiplist_words: &[String],
) -> std::collections::HashSet<u32> {
    let words: std::collections::HashSet<&str> =
        skiplist_words.iter().map(|s| s.as_str()).collect();
    let mut out = std::collections::HashSet::new();
    for (id, piece) in vocab.pieces() {
        let stripped = strip_space_marker(&piece);
        if words.contains(stripped.trim()) {
            out.insert(id);
        }
    }
    out
}

/// Vocabulary access needed by [`skiplist_token_ids`] (implemented over
/// `Tokenizer`'s vocab in the bundle; a small trait keeps this module
/// GPU-free and unit-testable).
pub trait VocabPieces {
    fn pieces(&self) -> Vec<(u32, String)>;
}

/// Strip the byte-level-BPE space marker (`Ġ` U+0120 / `Ċ` U+010A) that
/// prefixes continuation pieces.
pub fn strip_space_marker(piece: &str) -> String {
    piece
        .trim_start_matches('Ġ')
        .trim_start_matches('Ċ')
        .trim_start_matches('\u{0120}')
        .trim_start_matches('\u{010A}')
        .to_string()
}

/// Boolean keep-mask over token positions: `true` = keep the vector.
/// Marker / vision-delimiter tokens are always kept even if their piece
/// happens to match a skiplist word.
pub fn skiplist_mask(
    ids: &[u32],
    skiplist_ids: &std::collections::HashSet<u32>,
    protected: &[u32],
) -> Vec<bool> {
    ids.iter()
        .map(|&id| !skiplist_ids.contains(&id) || protected.contains(&id))
        .collect()
}

/// L2-normalize each row of a `[n, dim]` row-major matrix in place.
///
/// Zero rows pass through as zero. A row with a non-finite norm (NaN or an
/// overflow to +inf) would otherwise flow to the wire raw or scaled to
/// ~garbage — a kernel fault or a pathological input should surface as a
/// zero row, not as a NaN vector a MaxSim consumer can rank.
pub fn l2_normalize_rows(m: &mut [f32], dim: usize) {
    for row in m.chunks_exact_mut(dim) {
        let norm: f32 = row.iter().map(|x| x * x).sum();
        if norm > 0.0 && norm.is_finite() {
            let inv = 1.0 / norm.sqrt();
            for x in row.iter_mut() {
                *x *= inv;
            }
        } else if !norm.is_finite() {
            row.iter_mut().for_each(|x| *x = 0.0);
        }
    }
}

/// Late-interaction score: Σ over query rows of the max dot product against
/// any document row. Inputs are row-major `[n, dim]`; with normalized rows
/// each dot is a cosine.
pub fn maxsim(query: &[f32], q_n: usize, doc: &[f32], d_n: usize, dim: usize) -> f32 {
    if q_n == 0 || d_n == 0 || query.len() < q_n * dim || doc.len() < d_n * dim {
        return 0.0;
    }
    let mut total = 0f32;
    for qi in 0..q_n {
        let q = &query[qi * dim..(qi + 1) * dim];
        let mut best = f32::MIN;
        for di in 0..d_n {
            let d = &doc[di * dim..(di + 1) * dim];
            let mut dot = 0f32;
            for k in 0..dim {
                dot += q[k] * d[k];
            }
            if dot > best {
                best = dot;
            }
        }
        total += best;
    }
    total
}

/// Keep only the rows whose mask entry is `true`, compacting a row-major
/// `[n, dim]` matrix.
pub fn apply_mask_rows(m: &[f32], n: usize, dim: usize, mask: &[bool]) -> Vec<f32> {
    let mut out = Vec::with_capacity(mask.iter().filter(|b| **b).count() * dim);
    for (i, keep) in mask.iter().enumerate().take(n) {
        if *keep {
            out.extend_from_slice(&m[i * dim..(i + 1) * dim]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestVoc(Vec<(u32, String)>);

    impl VocabPieces for TestVoc {
        fn pieces(&self) -> Vec<(u32, String)> {
            self.0.clone()
        }
    }

    #[test]
    fn skiplist_ids_match_exact_pieces_only() {
        let voc = TestVoc(vec![
            (1, ".".into()),
            (2, "Ġ.".into()),  // " ." — same punctuation token after a space
            (3, "...".into()), // ellipsis piece: NOT a skiplist word
            (4, "a.".into()),  // multi-char piece: no match
            (5, "!".into()),
            (6, "hello".into()),
        ]);
        let words = vec![".".to_string(), "!".to_string()];
        let ids = skiplist_token_ids(&voc, &words);
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&5));
        assert!(!ids.contains(&3));
        assert!(!ids.contains(&4));
        assert!(!ids.contains(&6));
    }

    #[test]
    fn mask_protects_marker_tokens() {
        let mut skiplist = std::collections::HashSet::new();
        skiplist.insert(7u32);
        let mask = skiplist_mask(&[7, 9, 7], &skiplist, &[7]);
        assert_eq!(mask, vec![true, true, true]);
        let mask = skiplist_mask(&[7, 9, 7], &skiplist, &[]);
        assert_eq!(mask, vec![false, true, false]);
    }

    #[test]
    fn normalize_is_rowwise() {
        let mut m = vec![3.0, 4.0, 1.0, 0.0];
        l2_normalize_rows(&mut m, 2);
        assert!((m[0] - 0.6).abs() < 1e-6 && (m[1] - 0.8).abs() < 1e-6);
        assert_eq!(m[2..], vec![1.0, 0.0]); // zero row untouched
    }

    #[test]
    fn maxsim_sum_of_column_maxima() {
        // q = [[1,0],[0,1]], d = [[1,0],[0.7,0.7],[0,1]]
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let d = vec![1.0, 0.0, 0.7, 0.7, 0.0, 1.0];
        let s = maxsim(&q, 2, &d, 3, 2);
        assert!((s - (1.0 + 1.0)).abs() < 1e-6);
        // A doc that only matches the second query direction.
        let d2 = vec![0.0, 1.0];
        let s2 = maxsim(&q, 2, &d2, 1, 2);
        assert!((s2 - (0.0 + 1.0)).abs() < 1e-6);
    }

    #[test]
    fn apply_mask_rows_compacts() {
        let m = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let out = apply_mask_rows(&m, 3, 2, &[true, false, true]);
        assert_eq!(out, vec![1.0, 1.0, 3.0, 3.0]);
    }

    #[test]
    fn default_skiplist_is_ascii_punctuation() {
        assert!(DEFAULT_SKIPLIST_WORDS.contains(&"."));
        assert!(!DEFAULT_SKIPLIST_WORDS.contains(&"a"));
        assert_eq!(DEFAULT_SKIPLIST_WORDS.len(), 32);
    }
}
