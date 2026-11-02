// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire-quantize ple-hot-rank`: offline, no-GPU frequency rank of the
//! Qwen4 / Flash-Next PLE rows, written as the `<model>.plehot` bucket sidecar
//! that the runtime's PLE hot-row store consumes.
//!
//! The corpus is tokenized with the artifact's embedded tokenizer and hashed
//! with the runtime's own `PleHistory` / `PleHashMetadata`.  For every window
//! of `--window` tokens each DISTINCT global row id the window touches gets
//! +1 (history carries across the windows of one document; every document
//! starts from `PleHistory::new(eos)`).  Rows are then ranked by score
//! (descending, ties by ascending row index) and `bucket = rank * 256 /
//! padded_rows` is written per row.

use clap::Parser;
use hipfire_arch_qwen4::{PleHashMetadata, PleHistory, Qwen4Config};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::ple_hot_store::write_ple_hot_rank;
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message};
use hipfire_runtime::tokenizer::Tokenizer;
use rayon::slice::ParallelSliceMut;
use serde_json::{json, Value};
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Bucket count of the sidecar format (`buckets` in the header).
const BUCKETS: u64 = 256;

#[derive(Debug, Parser)]
#[command(
    name = "hipfire-quantize ple-hot-rank",
    about = "Rank Qwen4 PLE rows by document frequency over a corpus and write the \
             `<model>.plehot` hot-row bucket sidecar",
    long_about = "Rank Qwen4 / Flash-Next PLE rows by how many token windows touch them and write \
                  the 256-bucket `<model>.plehot` sidecar consumed by the runtime PLE hot-row \
                  store (bucket 0 = hottest).\n\n\
                  Corpus files are JSONL, one document per line:\n  \
                  {\"text\": \"...\"}                         encoded raw, no template\n  \
                  {\"messages\": [...], \"tools\": [...]?}    rendered with the model's embedded \
                  chat template (enable_thinking=false)\n\
                  Lines that fail to parse or render are counted as `bad` and skipped.\n\n\
                  A precomputed per-row count file (u32 little-endian, one entry per padded row) \
                  may be added with --counts-in; its counts are summed into the score."
)]
pub(crate) struct HotRankArgs {
    /// Qwen4 / Flash-Next HFQM model file whose PLE layout and tokenizer are used.
    #[arg(long, value_name = "MODEL.mq4")]
    pub model: PathBuf,

    /// Destination sidecar (conventionally `<model file>.plehot`).
    #[arg(long, value_name = "MODEL.mq4.plehot")]
    pub out: PathBuf,

    /// Tokens per counting window; each distinct row in a window counts once.
    #[arg(long, value_name = "TOKENS", default_value_t = 8192,
          value_parser = clap::value_parser!(u64).range(1..))]
    pub window: u64,

    /// Counting threads (default: available parallelism).
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,

    /// Precomputed per-row counts (u32 LE, padded_rows entries) added to the
    /// score. May be repeated.
    #[arg(long = "counts-in", value_name = "FILE.u32")]
    pub counts_in: Vec<PathBuf>,

    /// JSONL corpus files (one document per line).
    #[arg(value_name = "CORPUS.jsonl")]
    pub corpus: Vec<PathBuf>,
}

/// Entry point for `hipfire-quantize ple-hot-rank …` (`argv` excludes the
/// program name and the subcommand word). Returns the process exit code.
pub(crate) fn run(argv: Vec<String>) -> i32 {
    let args = HotRankArgs::parse_from(
        std::iter::once("hipfire-quantize ple-hot-rank".to_string()).chain(argv),
    );
    if args.corpus.is_empty() && args.counts_in.is_empty() {
        eprintln!("error: give at least one <corpus.jsonl> or --counts-in <file.u32>");
        return 2;
    }
    match run_inner(&args) {
        Ok(summary) => {
            println!("{summary}");
            0
        }
        Err(e) => {
            eprintln!("error: ple-hot-rank: {e}");
            1
        }
    }
}

/// Per-corpus-file totals, returned by the counting thread that read it.
struct SourceStat {
    path: PathBuf,
    documents: u64,
    tokens: u64,
    windows: u64,
    bad: u64,
}

fn run_inner(args: &HotRankArgs) -> Result<Value, String> {
    let t0 = std::time::Instant::now();
    let window = usize::try_from(args.window).map_err(|_| "--window too large".to_string())?;
    let hfq = HfqFile::open(&args.model)
        .map_err(|e| format!("open model {}: {e}", args.model.display()))?;
    // The rank needs only the model's PLE hash metadata, its EOS id and its
    // tokenizer/template, all in the base's metadata JSON: reading them
    // directly (rather than through full artifact admission) makes the tool
    // work on a split base, whose PLE rows live in a sidecar it never reads.
    let meta = &PleHashMetadata::from_json(&hfq.metadata_json)
        .map_err(|e| format!("PLE metadata of {}: {e}", args.model.display()))?;
    if *meta != PleHashMetadata::qwen4() {
        return Err(format!(
            "{}: PLE metadata does not match the pinned Qwen4 contract",
            args.model.display()
        ));
    }
    let eos = Qwen4Config::from_metadata_json(&hfq.metadata_json)
        .map_err(|e| format!("config of {}: {e}", args.model.display()))?
        .eos_token_id;
    let padded_rows = meta.padded_rows();
    let padded = usize::try_from(padded_rows).map_err(|_| "padded_rows overflow".to_string())?;
    if padded_rows > u64::from(u32::MAX) {
        return Err(format!(
            "padded_rows {padded_rows} exceeds the u32 row index range"
        ));
    }
    let threads = args
        .threads
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get()))
        .min(args.corpus.len().max(1));
    eprintln!(
        "ple-hot-rank: padded_rows={padded_rows} eos={eos} window={window} threads={threads} \
         corpus_files={} counts_in={}",
        args.corpus.len(),
        args.counts_in.len()
    );

    let (counts, sources) = if args.corpus.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer: {e}"))?;
        let template = hfq
            .chat_template()
            .ok_or_else(|| "model has no embedded chat template".to_string())?;
        count_corpus(
            &args.corpus,
            threads,
            window,
            meta,
            eos,
            &tokenizer,
            &template,
            padded,
            t0,
        )?
    };

    // Score = u64 sum of corpus counts and every --counts-in file.
    let mut scores: Vec<u64> = counts
        .iter()
        .map(|c| u64::from(c.load(Ordering::Relaxed)))
        .collect();
    drop(counts);
    scores.resize(padded, 0);
    for path in &args.counts_in {
        add_counts_file(path, &mut scores)?;
        eprintln!(
            "[{:.0}s] counts-in {}",
            t0.elapsed().as_secs_f64(),
            path.display()
        );
    }

    let buckets = assign_buckets(&scores);
    drop(scores);

    let documents: u64 = sources.iter().map(|s| s.documents).sum();
    let tokens: u64 = sources.iter().map(|s| s.tokens).sum();
    let windows: u64 = sources.iter().map(|s| s.windows).sum();
    let bad: u64 = sources.iter().map(|s| s.bad).sum();
    let rank_meta = json!({
        "window_tokens": window,
        "tokens": tokens,
        "documents": documents,
        "windows": windows,
        "sources": sources.iter().map(|s| json!({
            "path": s.path.display().to_string(),
            "documents": s.documents,
            "tokens": s.tokens,
        })).collect::<Vec<_>>(),
        "counts_in": args.counts_in.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "generator": "hipfire-quantize ple-hot-rank",
    });
    let fingerprint = hex(&meta.fingerprint());
    write_ple_hot_rank(&args.out, padded_rows, &fingerprint, &buckets, rank_meta)?;
    Ok(json!({
        "out": args.out.display().to_string(),
        "model": args.model.display().to_string(),
        "padded_rows": padded_rows,
        "ple_fingerprint": fingerprint,
        "window": window,
        "documents": documents,
        "tokens": tokens,
        "windows": windows,
        "bad_lines": bad,
        "counts_in_files": args.counts_in.len(),
        "elapsed_s": t0.elapsed().as_secs_f64(),
    }))
}

/// Count distinct rows per window over every corpus file. Files are handed to
/// `threads` scoped workers through an atomic cursor; per-file stats come back
/// through the join handles (no shared locks).
#[allow(clippy::too_many_arguments)]
fn count_corpus(
    files: &[PathBuf],
    threads: usize,
    window: usize,
    meta: &PleHashMetadata,
    eos: u32,
    tokenizer: &Tokenizer,
    template: &str,
    padded: usize,
    t0: std::time::Instant,
) -> Result<(Vec<AtomicU32>, Vec<SourceStat>), String> {
    let counts: Vec<AtomicU32> = (0..padded).map(|_| AtomicU32::new(0)).collect();
    let next = AtomicUsize::new(0);
    let results: Vec<Result<Vec<SourceStat>, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| -> Result<Vec<SourceStat>, String> {
                    let mut seen: Vec<u64> =
                        Vec::with_capacity(window.saturating_mul(16).min(1 << 24));
                    let mut stats = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(path) = files.get(i) else { break };
                        let stat = count_file(
                            path, window, meta, eos, tokenizer, template, &counts, &mut seen,
                        )?;
                        eprintln!(
                            "[{:.0}s] {}: docs={} tokens={} windows={} bad={}",
                            t0.elapsed().as_secs_f64(),
                            path.display(),
                            stat.documents,
                            stat.tokens,
                            stat.windows,
                            stat.bad
                        );
                        stats.push(stat);
                    }
                    Ok(stats)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("counting thread panicked".to_string()))
            })
            .collect()
    });
    let mut sources = Vec::new();
    for r in results {
        sources.extend(r?);
    }
    // Stable, input-order listing regardless of which thread read which file.
    sources.sort_by_key(|s| files.iter().position(|f| f == &s.path));
    Ok((counts, sources))
}

#[allow(clippy::too_many_arguments)]
fn count_file(
    path: &Path,
    window: usize,
    meta: &PleHashMetadata,
    eos: u32,
    tokenizer: &Tokenizer,
    template: &str,
    counts: &[AtomicU32],
    seen: &mut Vec<u64>,
) -> Result<SourceStat, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let reader = std::io::BufReader::with_capacity(1 << 20, file);
    let mut stat = SourceStat {
        path: path.to_path_buf(),
        documents: 0,
        tokens: 0,
        windows: 0,
        bad: 0,
    };
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read {}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let Some(tokens) = tokenize_line(&line, tokenizer, template) else {
            stat.bad += 1;
            continue;
        };
        stat.windows += count_document(meta, eos, &tokens, window, counts, seen);
        stat.documents += 1;
        stat.tokens += tokens.len() as u64;
    }
    Ok(stat)
}

/// One JSONL line → token ids, or `None` when the line is not a usable document.
fn tokenize_line(line: &str, tokenizer: &Tokenizer, template: &str) -> Option<Vec<u32>> {
    let v: Value = serde_json::from_str(line).ok()?;
    if let Some(text) = v.get("text").and_then(Value::as_str) {
        return Some(tokenizer.encode(text));
    }
    let messages: Vec<Message> = serde_json::from_value(v.get("messages")?.clone()).ok()?;
    let tools = match v.get("tools") {
        None | Some(Value::Null) => None,
        Some(Value::Array(a)) => Some(a.as_slice()),
        Some(_) => return None,
    };
    let frame = JinjaChatFrame {
        tokenizer,
        template,
        system: None,
        user: "",
        enable_thinking: false,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    };
    let rendered = frame.render_messages(&messages, tools, None).ok()?;
    Some(tokenizer.encode(&rendered))
}

/// Add 1 per distinct row per `window`-token window of one document; returns
/// the number of windows. The history carries across windows.
fn count_document(
    meta: &PleHashMetadata,
    eos: u32,
    tokens: &[u32],
    window: usize,
    counts: &[AtomicU32],
    seen: &mut Vec<u64>,
) -> u64 {
    let mut history = PleHistory::new(eos);
    let mut windows = 0;
    for chunk in tokens.chunks(window) {
        seen.clear();
        for &t in chunk {
            seen.extend_from_slice(&history.hash_token(meta, t));
        }
        seen.sort_unstable();
        seen.dedup();
        for &r in seen.iter() {
            counts[r as usize].fetch_add(1, Ordering::Relaxed);
        }
        windows += 1;
    }
    windows
}

/// Add a u32-LE per-row count file into `scores`; the file must hold exactly
/// `scores.len()` entries.
fn add_counts_file(path: &Path, scores: &mut [u64]) -> Result<(), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let len = file
        .metadata()
        .map_err(|e| format!("stat {}: {e}", path.display()))?
        .len();
    let expected = scores.len() as u64 * 4;
    if len != expected {
        return Err(format!(
            "--counts-in {}: {len} bytes, expected {expected} ({} u32 rows)",
            path.display(),
            scores.len()
        ));
    }
    let mut reader = std::io::BufReader::with_capacity(1 << 24, file);
    let mut buf = vec![0u8; 1 << 20];
    for chunk in scores.chunks_mut(buf.len() / 4) {
        let bytes = &mut buf[..chunk.len() * 4];
        reader
            .read_exact(bytes)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        for (score, c) in chunk.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *score += u64::from(u32::from_le_bytes(*c));
        }
    }
    Ok(())
}

/// Rank rows by score descending, ties by row index ascending, and return the
/// per-row bucket `rank * 256 / rows` (bucket 0 = hottest).
fn assign_buckets(scores: &[u64]) -> Vec<u8> {
    let rows = scores.len();
    assert!(rows as u64 <= u64::from(u32::MAX), "row index must fit u32");
    let mut order: Vec<u32> = (0..rows as u32).collect();
    order
        .par_sort_unstable_by(|&a, &b| scores[b as usize].cmp(&scores[a as usize]).then(a.cmp(&b)));
    let mut buckets = vec![0u8; rows];
    for (rank, &row) in order.iter().enumerate() {
        buckets[row as usize] = (rank as u64 * BUCKETS / rows as u64) as u8;
    }
    buckets
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_rank_by_score_desc_then_row_asc() {
        // 8 rows → bucket = rank * 256 / 8 = rank * 32.
        let scores = [3u64, 7, 7, 1, 7, 0, 3, 9];
        // Order: row7(9), row1, row2, row4 (7, ties by row), row0, row6 (3), row3(1), row5(0).
        let buckets = assign_buckets(&scores);
        assert_eq!(buckets, vec![128, 32, 64, 192, 96, 224, 160, 0]);
    }

    #[test]
    fn all_equal_scores_bucket_by_row_index() {
        let rows = 512usize;
        let buckets = assign_buckets(&vec![5u64; rows]);
        for (row, &b) in buckets.iter().enumerate() {
            assert_eq!(u64::from(b), row as u64 * 256 / rows as u64);
        }
        assert_eq!(buckets[rows - 1], 255);
    }

    #[test]
    fn bucket_math_uses_floor_for_non_multiple_row_counts() {
        // 5 rows: rank*256/5 = 0, 51, 102, 153, 204.
        let buckets = assign_buckets(&[50u64, 40, 30, 20, 10]);
        assert_eq!(buckets, vec![0, 51, 102, 153, 204]);
    }

    #[test]
    fn counts_file_adds_into_scores_and_rejects_wrong_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.u32");
        let mut bytes = Vec::new();
        for v in [1u32, 0, u32::MAX, 4] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, &bytes).unwrap();
        let mut scores = vec![10u64, 10, 10, 10];
        add_counts_file(&path, &mut scores).unwrap();
        assert_eq!(scores, vec![11, 10, 10 + u64::from(u32::MAX), 14]);
        let mut short = vec![0u64; 3];
        assert!(add_counts_file(&path, &mut short).is_err());
    }

    #[test]
    fn hex_encodes_lowercase_two_digits() {
        assert_eq!(hex(&[0x00, 0xab, 0x0f]), "00ab0f");
    }
}
