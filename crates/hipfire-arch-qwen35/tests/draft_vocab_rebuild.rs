// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! The committed MTP draft-vocabulary subset (`data/qwen35-draft-vocab-65536.bin`)
//! is reproducible from public files committed to this repository: rebuilding
//! it from the committed tokenizer fixture, `benchmarks/calib/bartowski_v5.txt`,
//! the wikitext-2 eval slice and the HF tokenizer corpus ids must give the
//! embedded bytes exactly.
//! CPU only.

use hipfire_arch_qwen35::draft_vocab::{
    committed_sources, rank_draft_vocab, tokenizer_identity, DraftVocab, RankParams,
    QWEN35_DRAFT_VOCAB, QWEN35_DRAFT_VOCAB_K, QWEN35_LM_HEAD_ROWS, SOURCE_IDS_JSONL, SOURCE_TEXTS,
    SOURCE_TOKENIZER,
};
use hipfire_runtime::tokenizer::Tokenizer;
use std::path::Path;

#[test]
fn committed_draft_vocab_rebuilds_from_repo_sources() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let read = |rel: &str| {
        std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
    };
    let tok = Tokenizer::from_hf_json(&read(SOURCE_TOKENIZER)).expect("tokenizer fixture");
    let sources = committed_sources(
        &tok,
        QWEN35_LM_HEAD_ROWS,
        &SOURCE_TEXTS.map(read),
        &read(SOURCE_IDS_JSONL),
    )
    .expect("count sources");
    let built = DraftVocab {
        rows: QWEN35_LM_HEAD_ROWS,
        tokenizer_sha256: tokenizer_identity(&tok),
        ids: rank_draft_vocab(
            &tok,
            QWEN35_LM_HEAD_ROWS,
            &sources,
            QWEN35_DRAFT_VOCAB_K,
            RankParams::COMMITTED,
        ),
    };
    let embedded = DraftVocab::parse(QWEN35_DRAFT_VOCAB).expect("embedded subset");
    assert_eq!(
        embedded.tokenizer_sha256, built.tokenizer_sha256,
        "tokenizer identity"
    );
    assert_eq!(embedded.ids, built.ids, "subset ids");
    assert_eq!(built.to_bytes(), QWEN35_DRAFT_VOCAB, "file bytes");
    // Every special token and byte-level symbol stays draftable.
    for (s, id) in tok.special_tokens() {
        assert!(
            built.ids.binary_search(id).is_ok(),
            "special {s:?} ({id}) missing"
        );
    }
}
