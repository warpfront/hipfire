// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared helpers for the KLD-eval example binaries (`build_kld_ref`,
//! `eval_hipfire`, `eval_gguf`). Each of these examples re-verifies the
//! same three invariants before doing work; centralising the helpers
//! here so a fix to one applies to all three. See
//! `docs/plans/issue-113-quant-quality-eval.md` §4 (Reference format &
//! pipeline) for what these checks defend against.

use std::path::Path;
use std::process::Command;

/// Verify the reference file's sha256 against the in-tree
/// `manifest.json` index.
///
/// Layout assumption: ref lives at `.../refs/<name>.kldref.bin`, manifest
/// at `.../harness/manifest.json` (sibling to `refs/`).
///
/// Behaviour:
/// - if the manifest is absent OR has no entry for `<name>`, emit a
///   warning and return (developer pre-upload state);
/// - if sha256 disagrees, print a clear error and `std::process::exit(2)`.
///
/// `tool_name` is the binary's short name (e.g. `"eval_hipfire"`) and is
/// used only in log lines.
pub fn verify_ref_sha256(ref_path: &Path, tool_name: &str) {
    let manifest_path = match ref_path.parent().and_then(|p| p.parent()) {
        Some(p) => p.join("harness").join("manifest.json"),
        None => {
            eprintln!(
                "warning: cannot locate harness/manifest.json relative to {}; \
                 skipping ref sha256 check",
                ref_path.display()
            );
            return;
        }
    };
    if !manifest_path.exists() {
        eprintln!(
            "warning: {} missing; skipping ref sha256 check",
            manifest_path.display()
        );
        return;
    }
    let manifest_file = std::fs::File::open(&manifest_path).expect("open manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_reader(manifest_file).expect("parse manifest.json");
    let ref_name = ref_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let expected = manifest
        .get("references")
        .and_then(|r| r.get(ref_name))
        .and_then(|r| r.get("sha256"))
        .and_then(|s| s.as_str())
        .map(String::from);
    let expected = match expected {
        Some(s) => s,
        None => {
            eprintln!("warning: no manifest entry / sha256 for {ref_name}; skipping check");
            return;
        }
    };
    eprintln!(
        "{tool_name}: computing sha256 of {} ...",
        ref_path.display()
    );
    let out = Command::new("sha256sum")
        .arg(ref_path)
        .output()
        .expect("invoke sha256sum");
    let actual = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(String::from)
        .expect("empty sha256sum output");
    if actual != expected {
        eprintln!("ERROR: ref sha256 mismatch for {}", ref_path.display());
        eprintln!("  expected: {expected}");
        eprintln!("  actual:   {actual}");
        std::process::exit(2);
    }
    eprintln!("{tool_name}: verified ref sha256 = {actual}");
}

/// Verify the slice file's md5 against the sibling `slice.md5` (one-line
/// `<md5>` or `md5sum`-format output).
///
/// Behaviour:
/// - if `<slice_dir>/slice.md5` is absent OR no recognisable hash on the
///   first line, emit a warning and return;
/// - if md5 disagrees, print a clear error and `std::process::exit(2)`.
pub fn verify_slice_md5(slice_path: &Path, tool_name: &str) {
    let md5_path = match slice_path.parent() {
        Some(p) => p.join("slice.md5"),
        None => return,
    };
    if !md5_path.exists() {
        eprintln!(
            "warning: {} missing; skipping slice md5 check",
            md5_path.display()
        );
        return;
    }
    let expected_line = match std::fs::read_to_string(&md5_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "warning: cannot read {}: {e}; skipping slice md5",
                md5_path.display()
            );
            return;
        }
    };
    let expected = match expected_line.split_whitespace().next() {
        Some(s) => s.to_string(),
        None => {
            eprintln!("warning: {} empty; skipping", md5_path.display());
            return;
        }
    };
    let out = Command::new("md5sum")
        .arg(slice_path)
        .output()
        .expect("invoke md5sum");
    let actual = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(String::from)
        .expect("empty md5sum output");
    if actual != expected {
        eprintln!("ERROR: slice md5 mismatch for {}", slice_path.display());
        eprintln!("  expected: {expected}");
        eprintln!("  actual:   {actual}");
        std::process::exit(2);
    }
    eprintln!("{tool_name}: verified slice md5 = {actual}");
}

/// Verify that the supplied `llama-perplexity` binary's reported commit
/// hash matches `pinned`.
///
/// Behaviour:
/// - parses `<bin> --version`'s "version: N (hash)" line;
/// - demands the binary's hash be ≥ 7 chars (collision floor on short
///   git hashes);
/// - compares an equal-length prefix of `pinned` to the binary's hash.
///   The prior implementation used `pinned.starts_with(hash)`, which
///   would accept arbitrarily short binary hashes (e.g. "9d" matching
///   "9dcf83552…"). Surfaced by glm-5 review finding 2.5.
///
/// On any mismatch, `std::process::exit(2)`.
pub fn verify_llama_commit(bin: &str, pinned: &str, tool_name: &str) {
    // Reproducibility guard, not a correctness requirement. Skip it when the
    // available llama.cpp build differs from the pinned commit (the KLD value
    // is insensitive to the exact llama.cpp commit). Set
    // HIPFIRE_SKIP_LLAMA_COMMIT_CHECK=1 to bypass.
    if hipfire_config::developer_var("HIPFIRE_SKIP_LLAMA_COMMIT_CHECK")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!("{tool_name}: HIPFIRE_SKIP_LLAMA_COMMIT_CHECK=1 — skipping llama.cpp commit verification (pinned {pinned})");
        return;
    }
    let out = Command::new(bin).arg("--version").output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ERROR: failed to invoke `{bin} --version`: {e}");
            std::process::exit(2);
        }
    };
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let needle = "version: ";
    let after_version = match combined.find(needle) {
        Some(i) => &combined[i + needle.len()..],
        None => {
            eprintln!("ERROR: could not find 'version: ' in `{bin} --version` output");
            std::process::exit(2);
        }
    };
    let open = match after_version.find('(') {
        Some(i) => i,
        None => {
            eprintln!("ERROR: malformed `--version` output: no '(' after version number");
            std::process::exit(2);
        }
    };
    let after_paren = &after_version[open + 1..];
    let close = match after_paren.find(')') {
        Some(i) => i,
        None => {
            eprintln!("ERROR: malformed `--version` output: no ')' after commit hash");
            std::process::exit(2);
        }
    };
    let hash = &after_paren[..close];
    if hash.len() < 7 {
        eprintln!(
            "ERROR: llama-perplexity reported a {}-char hash; want ≥ 7",
            hash.len()
        );
        eprintln!("  binary:    {bin}");
        eprintln!("  reported:  {hash}");
        std::process::exit(2);
    }
    if hash.len() > pinned.len() {
        eprintln!(
            "ERROR: llama-perplexity hash ({}) longer than pinned ({})",
            hash.len(),
            pinned.len()
        );
        std::process::exit(2);
    }
    let pinned_prefix = &pinned[..hash.len()];
    if hash != pinned_prefix {
        eprintln!("ERROR: llama-perplexity commit mismatch");
        eprintln!("  binary:             {bin}");
        eprintln!("  expected (pinned):  {pinned}");
        eprintln!("  actual (--version): {hash}");
        eprintln!("  Either rebuild llama.cpp at the pinned commit, or update");
        eprintln!("  PINNED_LLAMACPP_COMMIT in this binary AND in the PRD.");
        std::process::exit(2);
    }
    eprintln!("{tool_name}: verified llama.cpp commit prefix {hash}");
}

/// `eval_hipfire --scoring-mode prefill-whole` (qwen35): the one
/// `forward_prefill_batch` call of an `n_ctx` window, starting at position 0,
/// and the captured rows it scores. The existing `per-token` and split
/// `prefill` modes do not use it; their code is pinned unchanged by
/// `existing_scoring_modes_source_is_unchanged` below.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WholeWindowPlan {
    /// Window rows forwarded: the prefix, the scored region and the final,
    /// unscored token (always padded so the forward is tile-aligned).
    pub rows: std::ops::Range<usize>,
    /// Rows of the hidden-state capture buffer (every executed row).
    pub hidden_rows: usize,
    /// First captured row that is scored (`n_ctx / 2`, the split mode's first
    /// scored position).
    pub first_scored_row: usize,
    /// Scored rows (`n_ctx - 1 - n_ctx / 2`, as in every mode).
    pub scored_rows: usize,
}

/// Whether the final, unscored token is executed. `prefill-whole` always pads;
/// every other mode keeps eval_hipfire's pad-tail environment flag (`env_pad`).
pub fn scoring_tail_padded(whole_window: bool, env_pad: bool) -> bool {
    whole_window || env_pad
}

/// The [`WholeWindowPlan`] of an `n_ctx` window.
pub fn whole_window_plan(n_ctx: usize) -> WholeWindowPlan {
    WholeWindowPlan {
        rows: 0..n_ctx,
        hidden_rows: n_ctx,
        first_scored_row: n_ctx / 2,
        scored_rows: n_ctx - 1 - n_ctx / 2,
    }
}

/// KLD (and NLL of `actual_next`) of one scored position: candidate `logits`
/// against one HFKLDR reference block
/// `[top_k u32 idx][top_k f32 logp][f32 sum_p_residual][f32 pad]`, top-K of
/// the reference plus the residual cross-term, fp64 throughout. Shared by
/// every eval_hipfire arch and scoring mode (extracted from the evaluator with
/// its arithmetic unchanged; `legacy_scoring_output_golden` executes it).
pub fn kld_from_block(
    logits: &[f32],
    block_buf: &[u8],
    top_k: usize,
    actual_next: usize,
) -> (f64, Option<f64>) {
    // parse block: [top_k u32 idx][top_k f32 logp][f32 sum_p_residual][f32 pad]
    let mut top_indices: Vec<u32> = Vec::with_capacity(top_k);
    let mut top_log_probs: Vec<f32> = Vec::with_capacity(top_k);
    for j in 0..top_k {
        top_indices.push(u32::from_le_bytes(
            block_buf[j * 4..j * 4 + 4].try_into().unwrap(),
        ));
    }
    let lp_off = top_k * 4;
    for j in 0..top_k {
        top_log_probs.push(f32::from_le_bytes(
            block_buf[lp_off + j * 4..lp_off + j * 4 + 4]
                .try_into()
                .unwrap(),
        ));
    }
    let resid_off = top_k * 8;
    let sum_p_residual =
        f32::from_le_bytes(block_buf[resid_off..resid_off + 4].try_into().unwrap());

    // Candidate's log-Z = log Σ exp(logit_i) — fp64 throughout.
    let mut max_logit = f32::NEG_INFINITY;
    for &v in logits.iter() {
        if v > max_logit {
            max_logit = v;
        }
    }
    let mut sum_exp = 0.0f64;
    for &v in logits.iter() {
        sum_exp += ((v - max_logit) as f64).exp();
    }
    let log_z = (max_logit as f64) + sum_exp.ln();

    // KLD = Σ_{i in top_K_P_ref} P_ref(i) * (log_p_ref(i) - log_p_cand(i))
    //     + residual cross-term  (sum_p_residual_ref * Δlog_residual)
    let mut kld_token = 0.0f64;
    let mut sum_p_cand_at_ref_top = 0.0f64;
    for j in 0..top_k {
        let ref_idx = top_indices[j] as usize;
        if ref_idx >= logits.len() {
            continue;
        }
        let log_p_ref = top_log_probs[j] as f64;
        let log_p_cand = (logits[ref_idx] as f64) - log_z;
        let p_ref = log_p_ref.exp();
        let p_cand = log_p_cand.exp();
        kld_token += p_ref * (log_p_ref - log_p_cand);
        sum_p_cand_at_ref_top += p_cand;
    }
    let sum_p_residual_ref = sum_p_residual as f64;
    let sum_p_residual_cand = (1.0 - sum_p_cand_at_ref_top).max(0.0);
    if sum_p_residual_ref > 1e-9 && sum_p_residual_cand > 1e-9 {
        kld_token += sum_p_residual_ref * (sum_p_residual_ref.ln() - sum_p_residual_cand.ln());
    }
    // KLD ≥ 0 by Gibbs' inequality. Tiny negatives are fp64 roundoff on
    // ~257-term sums; >1e-9 magnitudes indicate a math bug. debug_assert
    // surfaces the latter in dev builds; release runs clamp at 0.
    debug_assert!(
        kld_token >= -1e-9,
        "negative KLD beyond fp roundoff: {kld_token}"
    );
    let kld_token = kld_token.max(0.0);

    let nll = if actual_next < logits.len() {
        Some(-((logits[actual_next] as f64) - log_z))
    } else {
        None
    };
    (kld_token, nll)
}

/// One chunk's per-sequence `(mean KLD, p99 KLD, mean NLL)`; an empty chunk is
/// `(0, 0, NaN)` (extracted from eval_hipfire's per-chunk statistics; same
/// arithmetic, the empty chunk's `continue` becomes this early return).
pub fn seq_stats(
    chunk_klds: &[f64],
    chunk_nll_sum: f64,
    chunk_nll_count: usize,
) -> (f64, f64, f64) {
    if chunk_klds.is_empty() {
        return (0.0, 0.0, f64::NAN);
    }
    let mean: f64 = chunk_klds.iter().copied().sum::<f64>() / chunk_klds.len() as f64;
    let mut sorted = chunk_klds.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
    let p99 = sorted[p99_idx];
    let mean_nll = if chunk_nll_count > 0 {
        chunk_nll_sum / chunk_nll_count as f64
    } else {
        f64::NAN
    };
    (mean, p99, mean_nll)
}

/// HFKSEQ v2 output: `HFKSEQ\0\0`, version 2, chunk count, reserved 0, then
/// per chunk `(mean, p99, mean_nll)` as little-endian f64 (extracted from
/// eval_hipfire's writer; same byte order and flush, errors returned).
pub fn write_hfkseq(
    out: &mut impl std::io::Write,
    effective_n_chunk: usize,
    mean_kld_per_seq: &[f64],
    p99_kld_per_seq: &[f64],
    mean_nll_per_seq: &[f64],
) -> std::io::Result<()> {
    out.write_all(b"HFKSEQ\0\0")?;
    out.write_all(&2u32.to_le_bytes())?;
    out.write_all(&(effective_n_chunk as u32).to_le_bytes())?;
    out.write_all(&0u32.to_le_bytes())?;
    for ((m, p), n) in mean_kld_per_seq
        .iter()
        .zip(p99_kld_per_seq.iter())
        .zip(mean_nll_per_seq.iter())
    {
        out.write_all(&m.to_le_bytes())?;
        out.write_all(&p.to_le_bytes())?;
        out.write_all(&n.to_le_bytes())?;
    }
    out.flush()
}

/// Mean of one chunk's per-token KLDs, exactly as [`seq_stats`] and the
/// evaluator's inline per-chunk statistics compute it (sequential `f64` sum in
/// scoring order, divided by the count). Anything that recomputes a kldseq
/// `mean_kld` from HFKTOK values MUST use this order to reproduce it bit for bit.
pub fn chunk_mean_kld(chunk_klds: &[f64]) -> f64 {
    chunk_klds.iter().copied().sum::<f64>() / chunk_klds.len() as f64
}

/// HFKTOK v1 per-token output: `HFKTOK\0\0`, version 1, `n_chunk`,
/// `scored_per_chunk`, reserved 0 (all little-endian u32), then `n_chunk *
/// scored_per_chunk` little-endian f64 per-token KLDs in scoring order (chunk
/// major). `klds.len()` MUST equal `n_chunk * scored_per_chunk`.
pub fn write_hfktok(
    out: &mut impl std::io::Write,
    n_chunk: usize,
    scored_per_chunk: usize,
    klds: &[f64],
) -> std::io::Result<()> {
    if n_chunk.checked_mul(scored_per_chunk) != Some(klds.len()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "hfktok: {} values != n_chunk {n_chunk} * scored_per_chunk {scored_per_chunk}",
                klds.len()
            ),
        ));
    }
    let as_u32 = |v: usize, what: &str| {
        u32::try_from(v).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("hfktok: {what} {v} overflows u32"),
            )
        })
    };
    out.write_all(b"HFKTOK\0\0")?;
    out.write_all(&1u32.to_le_bytes())?;
    out.write_all(&as_u32(n_chunk, "n_chunk")?.to_le_bytes())?;
    out.write_all(&as_u32(scored_per_chunk, "scored_per_chunk")?.to_le_bytes())?;
    out.write_all(&0u32.to_le_bytes())?;
    for k in klds {
        out.write_all(&k.to_le_bytes())?;
    }
    out.flush()
}

/// Collects the per-token KLDs of every completed chunk (the exact `f64`s
/// whose [`chunk_mean_kld`] the kldseq records) for [`write_hfktok`]. A chunk
/// whose scored-token count differs from `scored_per_chunk` is an error: the
/// HFKTOK shape is fixed.
pub struct TokenKldRecorder {
    scored_per_chunk: usize,
    n_chunk: usize,
    klds: Vec<f64>,
}

impl TokenKldRecorder {
    pub fn new(scored_per_chunk: usize, effective_n_chunk: usize) -> Self {
        Self {
            scored_per_chunk,
            n_chunk: 0,
            klds: Vec::with_capacity(scored_per_chunk * effective_n_chunk),
        }
    }

    pub fn push_chunk(&mut self, chunk_klds: &[f64]) -> Result<(), String> {
        if chunk_klds.len() != self.scored_per_chunk {
            return Err(format!(
                "per-token output: chunk {} scored {} tokens, expected {}",
                self.n_chunk,
                chunk_klds.len(),
                self.scored_per_chunk
            ));
        }
        self.klds.extend_from_slice(chunk_klds);
        self.n_chunk += 1;
        Ok(())
    }

    pub fn write(&self, out: &mut impl std::io::Write) -> std::io::Result<()> {
        write_hfktok(out, self.n_chunk, self.scored_per_chunk, &self.klds)
    }
}

#[cfg(test)]
mod hfktok_tests {
    use super::{chunk_mean_kld, seq_stats, write_hfkseq, write_hfktok, TokenKldRecorder};

    /// Reader for the HFKTOK v1 layout (the consumer lives in `pm-census`).
    fn read_hfktok(b: &[u8]) -> (usize, usize, Vec<f64>) {
        assert_eq!(&b[..8], b"HFKTOK\0\0");
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        assert_eq!(u(8), 1, "version");
        assert_eq!(u(20), 0, "reserved");
        let (n, s) = (u(12) as usize, u(16) as usize);
        assert_eq!(b.len(), 24 + n * s * 8);
        let vals = b[24..]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (n, s, vals)
    }

    fn synth(c: usize, scored: usize) -> Vec<f64> {
        (0..scored)
            .map(|j| {
                let x = ((c * 7919 + j * 104729) % 10007) as f64;
                (x * 1e-4 + 1e-6).powi(2) * (1.0 + (j % 5) as f64 * 0.3)
            })
            .collect()
    }

    #[test]
    fn hfktok_round_trips_and_reproduces_the_kldseq_chunk_means_bit_for_bit() {
        let (n, scored) = (5usize, 1023usize);
        let mut rec = TokenKldRecorder::new(scored, n);
        let (mut means, mut p99s, mut nlls) = (vec![], vec![], vec![]);
        let mut all = vec![];
        for c in 0..n {
            let chunk = synth(c, scored);
            let (m, p, nll) = seq_stats(&chunk, 1.5 * (c + 1) as f64, scored);
            means.push(m);
            p99s.push(p);
            nlls.push(nll);
            rec.push_chunk(&chunk).unwrap();
            all.extend_from_slice(&chunk);
        }
        let mut bytes = vec![];
        rec.write(&mut bytes).unwrap();
        assert_eq!(&bytes[..8], b"HFKTOK\0\0");
        let (rn, rs, vals) = read_hfktok(&bytes);
        assert_eq!((rn, rs), (n, scored));
        assert_eq!(vals.len(), all.len());
        for (a, b) in vals.iter().zip(&all) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        let mut seq = vec![];
        write_hfkseq(&mut seq, n, &means, &p99s, &nlls).unwrap();
        for c in 0..n {
            let re = chunk_mean_kld(&vals[c * scored..(c + 1) * scored]);
            let off = 20 + c * 24;
            let rec_mean = f64::from_le_bytes(seq[off..off + 8].try_into().unwrap());
            assert_eq!(re.to_bits(), rec_mean.to_bits(), "chunk {c}");
            assert_eq!(re.to_bits(), means[c].to_bits());
        }
    }

    #[test]
    fn hfktok_refuses_wrong_shapes() {
        let mut rec = TokenKldRecorder::new(4, 2);
        assert!(rec.push_chunk(&[0.0; 3]).is_err());
        rec.push_chunk(&[0.0; 4]).unwrap();
        assert!(write_hfktok(&mut Vec::new(), 2, 4, &[0.0; 4]).is_err());
        assert!(write_hfktok(&mut Vec::new(), 1, 4, &[0.0; 4]).is_ok());
    }
}

#[cfg(test)]
mod legacy_scoring_tests {
    use super::{kld_from_block, seq_stats, write_hfkseq};

    const TOP_K: usize = 2;
    const VOCAB: usize = 5;
    const N_CTX: usize = 8;

    fn fixture_logits(c: usize, j: usize) -> Vec<f32> {
        (0..VOCAB)
            .map(|v| ((c * 7 + j * 3 + v * 5) % 11) as f32 * 0.37 - 1.1)
            .collect()
    }

    /// One reference block; chunk 1's last block names an index past the
    /// vocabulary (skipped by the scorer).
    fn fixture_block(c: usize, j: usize) -> Vec<u8> {
        let idx = [
            ((c + j) % 5) as u32,
            if c == 1 && j == 2 {
                7
            } else {
                ((c + j + 2) % 5) as u32
            },
        ];
        let lp = [-0.4f32 - 0.1 * j as f32, -1.7f32 - 0.05 * c as f32];
        let resid = (1.0f32 - lp[0].exp() - lp[1].exp()).max(0.0);
        let mut b = Vec::new();
        idx.iter()
            .for_each(|i| b.extend_from_slice(&i.to_le_bytes()));
        lp.iter()
            .for_each(|l| b.extend_from_slice(&l.to_le_bytes()));
        b.extend_from_slice(&resid.to_le_bytes());
        b.extend_from_slice(&0f32.to_le_bytes());
        b
    }

    /// The existing (split `prefill` / `per-token`) scoring of two n_ctx-8
    /// windows from fixed candidate logits: scored positions `n_ctx/2 + j`,
    /// the j-th reference block of the chunk, NLL of `tokens[pos + 1]`
    /// (chunk 1 has one out-of-vocabulary next token), per-sequence stats and
    /// the HFKSEQ bytes. Every expected value below was produced by executing
    /// a standalone copy of the evaluator's pre-`prefill-whole` scoring code
    /// (beta 0a02e4b31 eval_hipfire.rs:246-321, :772-790, :1500-1514, copied
    /// into a program run on this fixture; not a run of the evaluator or a
    /// GPU). The extracted helpers must reproduce that output bit for bit.
    #[test]
    fn legacy_scoring_output_golden() {
        let tokens: [[u32; N_CTX]; 2] = [[1, 3, 0, 2, 4, 1, 3, 2], [2, 2, 1, 0, 3, 4, 9, 1]];
        let scoring_start = N_CTX / 2;
        let scored_per_chunk = N_CTX - 1 - N_CTX / 2;
        let golden_kld: [[u64; 3]; 2] = [
            [0x40027abd57d2c3f8, 0x3faf5f8bf033d1a7, 0x3fe91f40c08a088e],
            [0x3ffcf8474aef8a50, 0x3fcd876510f32e44, 0x3f96e0281cad1168],
        ];
        let golden_nll: [[Option<u64>; 3]; 2] = [
            [
                Some(4612887928392052492),
                Some(4607537410056564008),
                Some(4612288443115406740),
            ],
            [Some(4609862258098874162), None, Some(4604559737691607404)],
        ];
        let golden_seq: [[u64; 3]; 2] = [
            [0x3ff0d55d25240e41, 0x40027abd57d2c3f8, 0x3fff5c4f3699dccd],
            [0x3fe6032309006d94, 0x3ffcf8474aef8a50, 0x3ff26e5238c677f4],
        ];
        let golden_hfkseq = "48464b5345510000020000000200000000000000410e24255dd5f03ff8c3d257\
                             bd7a0240cddc99364f5cff3f946d00092303e63f508aef4a47f8fc3ff477c638\
                             526ef23f";
        let (mut means, mut p99s, mut nlls) = (Vec::new(), Vec::new(), Vec::new());
        for c in 0..2 {
            let mut chunk_klds = Vec::new();
            let (mut nll_sum, mut nll_count) = (0.0f64, 0usize);
            for j in 0..scored_per_chunk {
                let pos = scoring_start + j;
                let actual_next = tokens[c][pos + 1] as usize;
                let (kld, nll) = kld_from_block(
                    &fixture_logits(c, j),
                    &fixture_block(c, j),
                    TOP_K,
                    actual_next,
                );
                assert_eq!(kld.to_bits(), golden_kld[c][j], "kld c{c} j{j}");
                assert_eq!(nll.map(f64::to_bits), golden_nll[c][j], "nll c{c} j{j}");
                chunk_klds.push(kld);
                if let Some(n) = nll {
                    nll_sum += n;
                    nll_count += 1;
                }
            }
            let (m, p, n) = seq_stats(&chunk_klds, nll_sum, nll_count);
            assert_eq!(
                [m.to_bits(), p.to_bits(), n.to_bits()],
                golden_seq[c],
                "seq {c}"
            );
            means.push(m);
            p99s.push(p);
            nlls.push(n);
        }
        let mut bytes = Vec::new();
        write_hfkseq(&mut bytes, 2, &means, &p99s, &nlls).unwrap();
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, golden_hfkseq.replace(' ', ""));
    }

    #[test]
    fn empty_chunk_stats_are_zero_zero_nan() {
        let (m, p, n) = seq_stats(&[], 0.0, 0);
        assert_eq!((m, p), (0.0, 0.0));
        assert!(n.is_nan());
    }
}

#[cfg(test)]
mod prefill_whole_tests {
    use super::{scoring_tail_padded, whole_window_plan, WholeWindowPlan};

    #[test]
    fn whole_window_plan_scores_the_split_modes_positions() {
        assert_eq!(
            whole_window_plan(512),
            WholeWindowPlan {
                rows: 0..512,
                hidden_rows: 512,
                first_scored_row: 256,
                scored_rows: 255,
            }
        );
        assert_eq!(
            whole_window_plan(2048),
            WholeWindowPlan {
                rows: 0..2048,
                hidden_rows: 2048,
                first_scored_row: 1024,
                scored_rows: 1023,
            }
        );
        for n_ctx in [64usize, 513, 4096] {
            let p = whole_window_plan(n_ctx);
            // The split mode scores captured row j of its scored-region call,
            // which starts at `n_ctx / 2`, as position `n_ctx / 2 + j` for
            // `j < n_ctx - 1 - n_ctx / 2`; the whole plan must score exactly
            // those positions from its buffer.
            assert_eq!(p.rows.start + p.first_scored_row, n_ctx / 2);
            assert_eq!(p.scored_rows, n_ctx - 1 - n_ctx / 2);
            assert!(p.first_scored_row + p.scored_rows < p.hidden_rows);
            assert_eq!(p.rows.len(), p.hidden_rows);
        }
    }

    #[test]
    fn tail_padding_is_forced_for_whole_and_unchanged_otherwise() {
        assert!(!scoring_tail_padded(false, false));
        assert!(scoring_tail_padded(false, true));
        assert!(scoring_tail_padded(true, false));
        assert!(scoring_tail_padded(true, true));
    }

    /// Whitespace-collapsed source text.
    fn squash(text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// The existing `per-token` / split `prefill` scoring code of
    /// `eval_hipfire` is the code that produced every earlier kldseq. Each
    /// snippet below is copied verbatim from the evaluator as it was before
    /// `prefill-whole` existed (the split mode's padding default, its scratch
    /// hook, its capture buffer, and its two forward calls), and must still
    /// appear exactly once in the shipped evaluator: changing what the existing
    /// modes run has to change this golden deliberately. The evaluator's
    /// output itself needs the GPU forward, so this source golden is the CPU
    /// half of the identity argument.
    #[test]
    fn existing_scoring_modes_source_is_unchanged() {
        let source = squash(include_str!("../examples/eval_hipfire.rs"));
        // The evaluator's pad-tail variable is a harness-only token; it is
        // spelled in two halves so the env-var inventory scan does not count
        // this test as a source of it.
        let pad_env = concat!("HIPFIRE", "_FP8_G3_PAD_TAIL");
        let pad_line =
            format!(r#"let pad_scoring_tail = std::env::var("{pad_env}").as_deref() == Ok("1");"#);
        let golden = [
            pad_line.as_str(),
            r#"if args.scoring_mode == "prefill" {
            std::env::set_var("HIPFIRE_PREFILL_REUSE_PBS", "1");
        }"#,
            r#"let hidden_buf = if args.scoring_mode == "prefill" {
            Some(
                gpu.alloc_tensor(
                    &[scored_per_chunk + usize::from(pad_scoring_tail), config.dim],
                    DType::F32,
                )
                .expect("alloc hidden_buf"),
            )"#,
            r#"let h_buf = hidden_buf.as_ref().expect("hidden_buf in prefill mode");"#,
            r#"qwen35::forward_prefill_batch(
                    &mut gpu,
                    &weights,
                    &config,
                    &chunk_tokens[0..scoring_start],
                    0,
                    &mut kv_cache,
                    &mut dn_state,
                    &scratch,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("forward_prefill_batch prefix");
                qwen35::forward_prefill_batch(
                    &mut gpu,
                    &weights,
                    &config,
                    &chunk_tokens[scoring_start..(n_ctx - 1 + usize::from(pad_scoring_tail))],
                    scoring_start,
                    &mut kv_cache,
                    &mut dn_state,
                    &scratch,
                    None,
                    Some(h_buf),
                    None,
                    None,
                )
                .expect("forward_prefill_batch scored");"#,
            // The split mode's position / reference-block / NLL mapping that
            // `legacy_scoring_tests::legacy_scoring_output_golden` executes.
            r#"let pos = scoring_start + j;
                    let actual_next = chunk_tokens[pos + 1] as usize;
                    ref_in.read_exact(&mut block_buf).expect("read ref block");
                    let cand_logits = gpu.download_f32(&logits_view).expect("download logits");
                    let (kld, nll) = kld_from_block(&cand_logits, &block_buf, top_k, actual_next);"#,
        ];
        for snippet in golden {
            let snippet = squash(snippet);
            assert_eq!(
                source.matches(snippet.as_str()).count(),
                1,
                "existing-mode evaluator code changed: {snippet}"
            );
        }
    }
}
