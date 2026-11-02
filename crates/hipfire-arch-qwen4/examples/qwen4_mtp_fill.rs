// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native-MTP prompt-fill digest probe: runs an `mtp_prefill` per prompt
//! length and writes the target/MTP state family digests, the final prompt
//! row's logits, the pending hidden row, the seed and one draft window as
//! JSON. Run it once per route setting (e.g. `HIPFIRE_QWEN4_MTP_BATCHED_FILL`;
//! routes are read once per process) and `diff` the outputs: the batched fill
//! must match the per-row fill byte for byte.
//!
//! Without WARM_CHUNKS (or with 0) every fill is cold. With WARM_CHUNKS = K,
//! each LEN is the suffix of a session-cache hit: a warm prompt of
//! P = K * chunk rows tokens (plus one row) publishes the snapshot at P, and
//! a prompt sharing those P tokens then restores it and fills the LEN fresh
//! tokens from `start_pos = P`. PROMPT_FILE (cold fills only) supplies the
//! prompt: each LEN fills the first LEN tokens of the file's text, tokenized
//! without a template; without it the tokens are a deterministic spread over
//! the vocabulary.
//!
//!   cargo run --release -p hipfire-arch-qwen4 --features reference-parity \
//!     --example qwen4_mtp_fill -- MODEL OUT.json MAX_SEQ LEN[,LEN...] [WARM_CHUNKS [PROMPT_FILE]]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=6).contains(&args.len()) {
        eprintln!(
            "usage: qwen4_mtp_fill MODEL OUT.json MAX_SEQ LEN[,LEN...] [WARM_CHUNKS [PROMPT_FILE]]"
        );
        std::process::exit(2);
    }
    let max_seq: usize = args[2].parse().expect("MAX_SEQ");
    let lengths: Vec<usize> = args[3]
        .split(',')
        .map(|length| length.parse().expect("LEN"))
        .collect();
    let warm_chunks: usize = args
        .get(4)
        .map_or(0, |chunks| chunks.parse().expect("WARM_CHUNKS"));
    let prompt_text = args
        .get(5)
        .map(|path| std::fs::read_to_string(path).expect("read PROMPT_FILE"));
    if prompt_text.is_some() && warm_chunks != 0 {
        eprintln!("qwen4_mtp_fill: PROMPT_FILE needs WARM_CHUNKS = 0");
        std::process::exit(2);
    }
    match hipfire_arch_qwen4::state_parity::run_mtp_fill_digest(
        std::path::Path::new(&args[0]),
        &lengths,
        max_seq,
        warm_chunks,
        prompt_text.as_deref(),
    ) {
        Ok(report) => {
            std::fs::write(&args[1], serde_json::to_string_pretty(&report).unwrap())
                .expect("write output");
        }
        Err(error) => {
            eprintln!("qwen4_mtp_fill: {error}");
            std::process::exit(1);
        }
    }
}
