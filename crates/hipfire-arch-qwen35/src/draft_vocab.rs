// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Compressed MTP draft vocabulary sliced from the trunk lm_head (exact gfx1100).
//!
//! A Qwen3.5/3.6/3.8 MTP head without a compressed sidecar drafts through the
//! trunk's full `[248320, n_embd]` lm_head: on the 27B MQ4G256V2 trunk that is
//! ~675 MB of weights per draft step plus a single-workgroup argmax over 248K
//! logits. [`attach_trunk_draft_vocab`] gives such a head a FastMTP-style
//! `lm_head_draft` at load instead: the rows of a fixed token subset
//! ([`QWEN35_DRAFT_VOCAB`], 65,536 ids) copied device-to-device out of the
//! trunk lm_head, in ascending id order, plus the draft->full `vocab_map`. The
//! existing compressed paths of [`crate::mtp_spec::mtp_draft_phase_inner`]
//! (greedy device token chain, p_min, cvs-sampled) then draft over the subset.
//!
//! Only the draft changes. Verify still runs the full-vocab trunk lm_head and
//! the unchanged accept rule, so greedy output is the trunk's greedy output
//! (a token outside the subset can only be rejected, which lowers tau), and
//! the cvs-sampled path stays distribution-exact: the draft distribution `q`
//! is the softmax over the subset, scattered to full width with zeros outside
//! it, which is a valid proposal for the residual rule.
//!
//! The slice rows are the trunk's own row bytes, read by the same
//! `weight_gemv` dispatch, so an in-subset draft logit is computed from
//! exactly the weights the full-vocab draft uses for that id.
//!
//! The subset is data, not weights: [`QWEN35_DRAFT_VOCAB`] is built only from
//! files committed to this repository (see `examples/mtp_draft_vocab.rs` and
//! `tests/draft_vocab_rebuild.rs`, which rebuilds it byte-for-byte):
//! token frequencies over `benchmarks/calib/bartowski_v5.txt` (the public
//! bartowski imatrix calibration text), the wikitext-2 eval slice
//! `benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt` and the
//! HF reference ids of
//! `crates/hipfire-runtime/tests/fixtures/hf_tokenizer_corpus/cases.jsonl`
//! (rendered chat requests), with the special tokens and the 256 byte tokens
//! forced in and a BPE merge-rank prior for the tail. It is keyed to a tokenizer
//! by a SHA-256 of the vocabulary strings, so a model with a different
//! tokenizer keeps the full-vocab draft.
//!
//! Default on for exact gfx1100 (`HIPFIRE_GFX1100_DRAFT_VOCAB=0` keeps the
//! full-vocab draft; `=1` also enables it on other arches, untested there).

use crate::mtp_head::Qwen35MtpHead;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu};
use sha2::{Digest, Sha256};

/// Process switch: unset = on for exact gfx1100 only, `0` = off, `1` = on.
pub const DRAFT_VOCAB_ENV: &str = "HIPFIRE_GFX1100_DRAFT_VOCAB";

/// The committed Qwen3.5-family subset (65,536 of 248,320 lm_head rows).
pub static QWEN35_DRAFT_VOCAB: &[u8] = include_bytes!("../data/qwen35-draft-vocab-65536.bin");

const MAGIC: &[u8; 4] = b"HFDV";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 4 + 4 + 4 + 4 + 32;

/// A draft-vocabulary subset: `ids` (strictly ascending) out of `rows`
/// lm_head rows, for the tokenizer whose [`tokenizer_identity`] is
/// `tokenizer_sha256`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftVocab {
    pub rows: usize,
    pub tokenizer_sha256: [u8; 32],
    pub ids: Vec<u32>,
}

impl DraftVocab {
    /// File layout (little-endian): `HFDV`, version u32 (1), rows u32, k u32,
    /// tokenizer SHA-256 (32 B), then a bitmap of `rows` bits (bit `i` of
    /// byte `i / 8` set = token id `i` is in the subset).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_BYTES + self.rows.div_ceil(8));
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.rows as u32).to_le_bytes());
        out.extend_from_slice(&(self.ids.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.tokenizer_sha256);
        let mut bits = vec![0u8; self.rows.div_ceil(8)];
        for &id in &self.ids {
            bits[id as usize / 8] |= 1 << (id % 8);
        }
        out.extend_from_slice(&bits);
        out
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_BYTES || &bytes[..4] != MAGIC {
            return Err("not a draft-vocab file (magic)".into());
        }
        let u32_at =
            |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        if u32_at(4) != VERSION {
            return Err(format!(
                "draft-vocab version {} (expected {VERSION})",
                u32_at(4)
            ));
        }
        let rows = u32_at(8) as usize;
        let k = u32_at(12) as usize;
        let bits = &bytes[HEADER_BYTES..];
        if rows == 0 || bits.len() != rows.div_ceil(8) {
            return Err(format!(
                "draft-vocab bitmap is {} B for {rows} rows",
                bits.len()
            ));
        }
        let ids: Vec<u32> = (0..rows)
            .filter(|&i| bits[i / 8] & (1 << (i % 8)) != 0)
            .map(|i| i as u32)
            .collect();
        if ids.len() != k || k == 0 {
            return Err(format!(
                "draft-vocab bitmap holds {} ids, header says {k}",
                ids.len()
            ));
        }
        let mut tokenizer_sha256 = [0u8; 32];
        tokenizer_sha256.copy_from_slice(&bytes[16..48]);
        Ok(Self {
            rows,
            tokenizer_sha256,
            ids,
        })
    }
}

/// SHA-256 over the vocabulary strings of every real token id
/// (`0..=max_token_id`), length-prefixed. Unlike
/// [`Tokenizer::vocab_digest`] it ignores the trailing padding slots, which
/// differ between the HF-JSON and `.hfq`-metadata constructors.
pub fn tokenizer_identity(tok: &Tokenizer) -> [u8; 32] {
    let vocab = tok.vocab();
    let n = tok
        .max_token_id()
        .map_or(0, |m| m as usize + 1)
        .min(vocab.len());
    let mut h = Sha256::new();
    h.update((n as u64).to_le_bytes());
    for s in &vocab[..n] {
        h.update((s.len() as u64).to_le_bytes());
        h.update(s.as_bytes());
    }
    h.finalize().into()
}

/// One token-count source for [`rank_draft_vocab`]: `counts[id]` occurrences
/// of each id. Sources are weighted equally by relative frequency.
pub struct CountSource {
    pub counts: Vec<u64>,
}

impl CountSource {
    pub fn new(rows: usize) -> Self {
        Self {
            counts: vec![0; rows],
        }
    }
    pub fn add_ids(&mut self, ids: impl IntoIterator<Item = u32>) {
        for id in ids {
            if let Some(c) = self.counts.get_mut(id as usize) {
                *c += 1;
            }
        }
    }
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }
}

/// Scoring knobs of [`rank_draft_vocab`]; [`RankParams::COMMITTED`] built
/// the committed subset.
#[derive(Clone, Copy, Debug)]
pub struct RankParams {
    /// Weight of the BPE merge-rank prior.
    pub merge_rank_prior: f64,
    /// Share of a word's pooled frequency (summed over its leading-space and
    /// letter-case variants: `Sentence`, ` sentence`, ...) credited to each
    /// variant. The corpora are mostly running text, while generations put
    /// many words right after markdown or a newline, where the tokenizer
    /// uses the no-space form.
    pub variant_pool: f64,
}

impl RankParams {
    pub const COMMITTED: RankParams = RankParams {
        merge_rank_prior: 0.05,
        variant_pool: 0.5,
    };
}

/// ASCII word key of a decoded token (one leading space dropped,
/// lower-cased), for [`RankParams::variant_pool`].
fn word_key(s: &str) -> Option<String> {
    let w = s.strip_prefix(' ').unwrap_or(s);
    (w.len() > 1 && w.bytes().all(|b| b.is_ascii_alphabetic())).then(|| w.to_ascii_lowercase())
}

/// Prior scale by script of the decoded token: the merge order follows the
/// tokenizer's multilingual training mix, so the prior mostly promotes ASCII
/// (English, code) tokens a small corpus happened to miss.
fn script_factor(s: &str) -> f64 {
    const TYPO: [char; 9] = [
        '\u{2014}', '\u{2013}', '\u{2019}', '\u{2018}', '\u{201c}', '\u{201d}', '\u{2026}',
        '\u{2022}', '\u{2192}',
    ];
    if s.is_ascii() {
        1.0
    } else if s
        .chars()
        .all(|c| (c as u32) < 0x250 || c.is_whitespace() || TYPO.contains(&c))
    {
        0.5
    } else {
        0.1
    }
}

/// The `k` draft ids for a `rows`-row lm_head. The special/added tokens and
/// the single-symbol byte tokens are always in. The rest are ordered by
/// score = summed relative frequency over `sources`
/// + `variant_pool` x the pooled frequency of the token's word variants
/// + `merge_rank_prior` x script factor x a Zipf weight `1 / (rank + 10)`
/// of the BPE merge rank (normalized to sum 1; earlier merge = more frequent
/// in the tokenizer's own training data), then merge rank, then id. Ids past
/// the tokenizer's last real token (lm_head padding rows) are never chosen.
/// Returned ascending.
pub fn rank_draft_vocab(
    tok: &Tokenizer,
    rows: usize,
    sources: &[CountSource],
    k: usize,
    params: RankParams,
) -> Vec<u32> {
    let real = tok.max_token_id().map_or(0, |m| m as usize + 1).min(rows);
    let vocab = tok.vocab();
    let merge_rank = tok.build_merge_rank_table();
    let mut forced = vec![false; rows];
    for (_, id) in tok.special_tokens() {
        if (*id as usize) < real {
            forced[*id as usize] = true;
        }
    }
    for (id, f) in forced.iter_mut().enumerate().take(real) {
        if vocab.get(id).is_some_and(|s| s.chars().count() == 1) {
            *f = true;
        }
    }
    let mut score = vec![0f64; rows];
    for src in sources {
        let n = src.total();
        if n == 0 {
            continue;
        }
        for (s, &c) in score.iter_mut().zip(&src.counts) {
            *s += c as f64 / n as f64;
        }
    }
    let rank = |id: usize| -> usize {
        if vocab.get(id).is_some_and(|s| s.chars().count() == 1) {
            0
        } else {
            merge_rank.get(&(id as u32)).map_or(usize::MAX, |r| r + 1)
        }
    };
    let decoded: Vec<String> = (0..real).map(|id| tok.decode(&[id as u32])).collect();
    let keys: Vec<Option<String>> = decoded.iter().map(|s| word_key(s)).collect();
    let mut pooled = std::collections::HashMap::<&str, f64>::new();
    for (id, key) in keys.iter().enumerate() {
        if let Some(key) = key {
            *pooled.entry(key.as_str()).or_default() += score[id];
        }
    }
    let zipf_sum: f64 = (0..=merge_rank.len())
        .map(|r| 1.0 / (r as f64 + 10.0))
        .sum();
    let mut prior = vec![0f64; real];
    for (id, p) in prior.iter_mut().enumerate() {
        let r = rank(id);
        if r != usize::MAX {
            *p = params.merge_rank_prior * script_factor(&decoded[id])
                / (r as f64 + 10.0)
                / zipf_sum;
        }
    }
    for (id, s) in score.iter_mut().enumerate().take(real) {
        if let Some(key) = &keys[id] {
            *s += params.variant_pool * pooled[key.as_str()];
        }
        *s += prior[id];
    }
    let mut order: Vec<usize> = (0..real).collect();
    order.sort_by(|&a, &b| {
        forced[b]
            .cmp(&forced[a])
            .then(score[b].total_cmp(&score[a]))
            .then(rank(a).cmp(&rank(b)))
            .then(a.cmp(&b))
    });
    let mut ids: Vec<u32> = order.into_iter().take(k).map(|i| i as u32).collect();
    ids.sort_unstable();
    ids
}

/// lm_head rows of the Qwen3.5/3.6/3.8 family (248,320: the tokenizer's
/// ids plus padding) and the committed subset size.
pub const QWEN35_LM_HEAD_ROWS: usize = 248_320;
pub const QWEN35_DRAFT_VOCAB_K: usize = 65_536;

/// Repository paths (relative to the repo root) of the inputs the committed
/// subset is built from; `tests/draft_vocab_rebuild.rs` checks the result.
pub const SOURCE_TOKENIZER: &str =
    "crates/hipfire-runtime/tests/fixtures/hf_tokenizer_corpus/tokenizer.json";
pub const SOURCE_TEXTS: [&str; 2] = [
    "benchmarks/calib/bartowski_v5.txt",
    "benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt",
];
pub const SOURCE_IDS_JSONL: &str =
    "crates/hipfire-runtime/tests/fixtures/hf_tokenizer_corpus/cases.jsonl";

/// The count sources of the committed subset: each of `texts` (the
/// [`SOURCE_TEXTS`] files) encoded with `tok`, then the `ids` arrays of every
/// line of `ids_jsonl` (the HF tokenizer corpus cases), one source each.
pub fn committed_sources(
    tok: &Tokenizer,
    rows: usize,
    texts: &[String],
    ids_jsonl: &str,
) -> Result<Vec<CountSource>, String> {
    let mut out = Vec::with_capacity(texts.len() + 1);
    for text in texts {
        let mut a = CountSource::new(rows);
        a.add_ids(tok.encode(text));
        out.push(a);
    }
    let mut b = CountSource::new(rows);
    for (n, line) in ids_jsonl.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("ids line {}: {e}", n + 1))?;
        let ids = v["ids"]
            .as_array()
            .ok_or_else(|| format!("ids line {}: no ids array", n + 1))?;
        b.add_ids(ids.iter().filter_map(|x| x.as_u64()).map(|x| x as u32));
    }
    out.push(b);
    Ok(out)
}

fn enabled(gpu: &Gpu) -> bool {
    hipfire_config::developer_bool(DRAFT_VOCAB_ENV, gpu.arch_caps.is_gfx1100())
}

/// Give `head` a compressed draft lm_head sliced from the trunk lm_head
/// `output` when the switch is on, the head has none of its own, the trunk
/// lm_head has a plain row layout and `tokenizer` matches the subset. Any
/// refusal or allocation failure leaves the head on the full-vocab draft
/// (logged once); this never fails the load.
pub fn attach_trunk_draft_vocab(
    gpu: &mut Gpu,
    head: &mut Qwen35MtpHead,
    output: &WeightTensor,
    tokenizer: &Tokenizer,
) {
    if !enabled(gpu) || head.weights.lm_head_draft.is_some() {
        return;
    }
    match try_attach(gpu, head, output, tokenizer) {
        Ok(()) => {}
        Err(reason) => eprintln!("  MTP draft vocab: full-vocab draft kept ({reason})"),
    }
}

fn try_attach(
    gpu: &mut Gpu,
    head: &mut Qwen35MtpHead,
    output: &WeightTensor,
    tokenizer: &Tokenizer,
) -> Result<(), String> {
    let vocab = DraftVocab::parse(QWEN35_DRAFT_VOCAB)?;
    if output.m != head.config.vocab_size || output.k != head.config.n_embd {
        return Err(format!(
            "trunk lm_head [{} x {}] != head vocab {} x n_embd {}",
            output.m, output.k, head.config.vocab_size, head.config.n_embd
        ));
    }
    if output.m != vocab.rows {
        return Err(format!(
            "lm_head has {} rows, subset is for {}",
            output.m, vocab.rows
        ));
    }
    if output.paro.is_some() || output.row_stride != 0 {
        return Err(format!(
            "{:?} lm_head has no plain row layout",
            output.gpu_dtype
        ));
    }
    let row_bytes = output
        .gpu_dtype
        .row_bytes(output.k)
        .ok_or_else(|| format!("{:?} lm_head has no fixed row size", output.gpu_dtype))?;
    if output.buf.buf.size() < output.m * row_bytes {
        return Err("lm_head buffer smaller than its rows".into());
    }
    if tokenizer_identity(tokenizer) != vocab.tokenizer_sha256 {
        return Err("tokenizer differs from the one the subset was built for".into());
    }
    let ids = vocab.ids;
    let k = ids.len();
    let slice = gpu
        .alloc_tensor(&[k * row_bytes], DType::Raw)
        .map_err(|e| format!("alloc {} B: {e}", k * row_bytes))?;
    let map_bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
    let copied = (|| -> hip_bridge::HipResult<rdna_compute::GpuTensor> {
        // Ascending ids: one D2D copy per run of consecutive rows.
        let mut j = 0;
        while j < k {
            let start = ids[j] as usize;
            let mut len = 1;
            while j + len < k && ids[j + len] as usize == start + len {
                len += 1;
            }
            gpu.memcpy_dtod_at_auto(
                &slice.buf,
                j * row_bytes,
                &output.buf.buf,
                start * row_bytes,
                len * row_bytes,
            )?;
            j += len;
        }
        gpu.upload_raw(&map_bytes, &[map_bytes.len()])
    })();
    let map_gpu = match copied {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(slice);
            return Err(format!("row copy: {e}"));
        }
    };
    eprintln!(
        "  MTP draft vocab: {k} of {} trunk lm_head rows ({:.1} MiB, {:?}); {DRAFT_VOCAB_ENV}=0 keeps the full-vocab draft",
        output.m,
        (k * row_bytes) as f64 / (1 << 20) as f64,
        output.gpu_dtype
    );
    head.weights.lm_head_draft = Some(WeightTensor {
        buf: slice,
        gpu_dtype: output.gpu_dtype,
        m: k,
        k: output.k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
        lloyd_lut_e4m3: output.lloyd_lut_e4m3,
        lloyd_lut_f16: output.lloyd_lut_f16,
        lloyd_lut_c16: output.lloyd_lut_c16,
    });
    head.weights.lm_head_draft_vocab_map = Some(ids);
    head.weights.lm_head_draft_vocab_map_gpu = Some(map_gpu);
    head.weights.compressed_vocab_size = Some(k);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_vocab_file_round_trips() {
        let v = DraftVocab {
            rows: 21,
            tokenizer_sha256: [7; 32],
            ids: vec![0, 3, 8, 9, 20],
        };
        let bytes = v.to_bytes();
        assert_eq!(bytes.len(), HEADER_BYTES + 3);
        assert_eq!(DraftVocab::parse(&bytes).unwrap(), v);
        let mut bad = bytes.clone();
        bad[12] = 4; // k no longer matches the bitmap
        assert!(DraftVocab::parse(&bad).is_err());
        assert!(DraftVocab::parse(&bytes[..HEADER_BYTES]).is_err());
    }

    #[test]
    fn draft_vocab_committed_subset_parses() {
        let v = DraftVocab::parse(QWEN35_DRAFT_VOCAB).unwrap();
        assert_eq!(v.rows, 248_320);
        assert_eq!(v.ids.len(), 65_536);
        assert!(v.ids.windows(2).all(|w| w[0] < w[1]));
    }
}
