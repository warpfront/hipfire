// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Offline cost-curve fixtures, generated from corpus messages, NOT byte-identical
//! replicas of the 22-cell benchmark. No GPU, weights copy, or inference is used.
//! The HFQ reader maps the source read-only; only metadata/tokenizer/template are
//! consumed. Each trial renders the WHOLE real Jinja frame before tokenization.
//!
//! cargo build -p hipfire-arch-qwen4 --example qwen4_cost_prompts --release
//! cargo test -p hipfire-arch-qwen4 --example qwen4_cost_prompts
//! target/release/examples/qwen4_cost_prompts --model MODEL.hfq \
//!   --corpus /path/to/phase3-kit-v1/kit/corpus.json --out NEW_DIRECTORY \
//!   --targets 1536,6144,12288,24576 --n-ctx 32768
//!
//! Feed the eight manifest-listed .txt files directly to the cost example's
//! --prompt-files: they already contain the entire rendered chat frame. Do NOT
//! apply another chat template. Verify their actual buckets again in that runner.
//! manifest.json is published last and is the completion marker. A failed write
//! can leave a partial new directory; it is never reused or overwritten.

use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::prompt_frame::JinjaChatFrame;
use hipfire_runtime::tokenizer::Tokenizer;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const COPY_LABEL: &str = "W1-2K-quote";
const PROSE_LABEL: &str = "W4-prose-river";
const CONTEXT_RESERVE: usize = 64;
const LOCAL_RADIUS: usize = 96;
const USAGE: &str = "qwen4_cost_prompts --model MODEL.hfq --corpus CORPUS.json --out NEWDIR [--targets 1536,6144,12288,24576] [--n-ctx 32768]";

type Result<T, E = String> = std::result::Result<T, E>;

struct Args {
    model: PathBuf,
    corpus: PathBuf,
    out: PathBuf,
    targets: [usize; 4],
    n_ctx: usize,
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args> {
    let (mut model, mut corpus, mut out) = (None, None, None);
    let mut targets = [1536, 6144, 12288, 24576];
    let mut n_ctx = 32768;
    while let Some(key) = args.next() {
        if !matches!(
            key.as_str(),
            "--model" | "--corpus" | "--out" | "--targets" | "--n-ctx"
        ) {
            return Err(format!("unknown argument {key}\n{USAGE}"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--corpus" => corpus = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--targets" => {
                let values: Vec<usize> = value
                    .split(',')
                    .map(|s| s.parse().map_err(|_| format!("invalid target {s:?}")))
                    .collect::<Result<_>>()?;
                targets = values
                    .try_into()
                    .map_err(|_| "--targets needs four integers".to_string())?;
            }
            "--n-ctx" => n_ctx = value.parse().map_err(|_| "invalid --n-ctx".to_string())?,
            _ => unreachable!(),
        }
    }
    for (index, target) in targets.iter().enumerate() {
        if *target == 0 || bucket_index(*target) != index {
            return Err(format!(
                "target {target} is not in bucket {}",
                bucket_name(index)
            ));
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        corpus: corpus.ok_or("--corpus is required")?,
        out: out.ok_or("--out is required")?,
        targets,
        n_ctx,
    })
}

fn bucket_index(tokens: usize) -> usize {
    match tokens {
        0..=2048 => 0,
        2049..=8192 => 1,
        8193..=16384 => 2,
        _ => 3,
    }
}

fn bucket_name(index: usize) -> &'static str {
    ["Le2K", "Le8K", "Le16K", "Over16K"][index]
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

// Stable, documented token hashing: consecutive unsigned 32-bit little-endian IDs.
fn token_sha256(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn required_str<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value[field]
        .as_str()
        .ok_or_else(|| format!("missing string field {field}"))
}

struct Source<'a> {
    class: &'static str,
    cell: &'a Value,
    original_user: &'a str,
    background: &'a str,
    fixed_prompt: String,
    derivation: &'static str,
    decode_tokens: usize,
}

fn cell<'a>(corpus: &'a Value, label: &str) -> Result<&'a Value> {
    let cells = corpus.as_array().ok_or("corpus must be a JSON array")?;
    let mut matches = cells
        .iter()
        .filter(|cell| cell["label"].as_str() == Some(label));
    let selected = matches
        .next()
        .ok_or_else(|| format!("missing corpus cell {label}"))?;
    if matches.next().is_some() {
        return Err(format!("duplicate corpus cell {label}"));
    }
    Ok(selected)
}

fn single_user(cell: &Value) -> Result<&str> {
    let messages = cell["messages"]
        .as_array()
        .ok_or("cell has no messages array")?;
    if messages.len() != 1 || messages[0]["role"] != "user" {
        return Err("selected source must contain exactly one user message".into());
    }
    required_str(&messages[0], "content")
}

fn decode_tokens(cell: &Value) -> Result<usize> {
    let count = cell["max_tokens"]
        .as_u64()
        .ok_or("cell has no max_tokens")?;
    let count = usize::try_from(count).map_err(|_| "max_tokens exceeds usize")?;
    if count == 0 {
        return Err("max_tokens must be positive".into());
    }
    Ok(count)
}

fn sources(corpus: &Value) -> Result<[Source<'_>; 2]> {
    let copy = cell(corpus, COPY_LABEL)?;
    let prose = cell(corpus, PROSE_LABEL)?;
    let copy_user = single_user(copy)?;
    let prose_user = single_user(prose)?;
    let opening = copy_user
        .find("<document ")
        .ok_or("COPY source has no document opening")?;
    let body_start = copy_user[opening..]
        .find('>')
        .map(|i| opening + i + 1)
        .ok_or("COPY document opening is unclosed")?;
    let close = copy_user[body_start..]
        .find("</document>")
        .map(|i| body_start + i)
        .ok_or("COPY source has no document closing")?;
    let document = &copy_user[body_start..close];
    let heading = required_str(&copy["quote"], "heading")?;
    let level = heading.bytes().take_while(|byte| *byte == b'#').count();
    if level == 0 {
        return Err("COPY quote heading is not a Markdown heading".into());
    }
    let section_start = document
        .find(heading)
        .filter(|start| {
            (*start == 0 || document.as_bytes()[*start - 1] == b'\n')
                && document[*start + heading.len()..].starts_with('\n')
        })
        .ok_or("COPY heading missing as a complete document line")?;
    let rest = &document[section_start..];
    let mut section_end = rest.len();
    let mut offset = 0;
    for (index, line) in rest.split_inclusive('\n').enumerate() {
        let hashes = line.bytes().take_while(|byte| *byte == b'#').count();
        if index > 0 && hashes > 0 && hashes <= level && line.as_bytes().get(hashes) == Some(&b' ')
        {
            section_end = offset;
            break;
        }
        offset += line.len();
    }
    let section = &rest[..section_end];
    if section.trim().len() <= heading.len() {
        return Err("COPY quote section is empty".into());
    }
    // Keep the real quote section unmodified, the original document wrapper, and
    // the entire original final instruction. Only unrelated document sections
    // are omitted, allowing the COPY task to fit the smallest bucket as well.
    let fixed_copy = format!(
        "{}\n{}{}",
        &copy_user[..body_start],
        section,
        &copy_user[close..]
    );
    if copy_user[close + "</document>".len()..].trim().is_empty() {
        return Err("COPY final instruction is missing".into());
    }
    Ok([
        Source {
            class: "COPY", cell: copy, original_user: copy_user,
            background: document, fixed_prompt: fixed_copy,
            derivation: "Original document reduced to its complete requested quote section; original final instruction retained verbatim. Background repeats original document text.",
            decode_tokens: decode_tokens(copy)?,
        },
        Source {
            class: "PROSE", cell: prose, original_user: prose_user,
            background: prose_user, fixed_prompt: prose_user.to_string(),
            derivation: "Original river prose instruction retained verbatim at the end. Background repeats original river message text.",
            decode_tokens: decode_tokens(prose)?,
        },
    ])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fit {
    background_bytes: usize,
    actual_tokens: usize,
    trials: usize,
}

// This is bounded search, NOT a claim that BPE counts are globally monotone.
// Exponential bracketing + bisection locates a crossing, then every byte length
// near that crossing is measured. Keep the closest actually measured in-bucket
// candidate, tie-breaking toward fewer tokens, then fewer background bytes.
fn fit(
    mut count: impl FnMut(usize) -> Result<usize>,
    target: usize,
    max_tokens: usize,
    max_background: usize,
) -> Result<Fit> {
    let wanted_bucket = bucket_index(target);
    let mut best: Option<Fit> = None;
    let mut trials = 0;
    let mut measure = |bytes: usize| -> Result<usize> {
        let tokens = count(bytes)?;
        trials += 1;
        if tokens > 0 && tokens <= max_tokens && bucket_index(tokens) == wanted_bucket {
            let candidate = Fit {
                background_bytes: bytes,
                actual_tokens: tokens,
                trials: 0,
            };
            let score = |v: Fit| {
                (
                    v.actual_tokens.abs_diff(target),
                    v.actual_tokens,
                    v.background_bytes,
                )
            };
            if best.map_or(true, |old| score(candidate) < score(old)) {
                best = Some(candidate);
            }
        }
        Ok(tokens)
    };
    let baseline = measure(0)?;
    if baseline != target {
        let mut low = 0;
        let mut high = 1.min(max_background);
        let mut high_count = baseline;
        if baseline < target && high > 0 {
            high_count = measure(high)?;
            while high_count < target && high < max_background {
                low = high;
                high = high.saturating_mul(2).min(max_background);
                high_count = measure(high)?;
            }
            while low + 1 < high {
                let mid = low + (high - low) / 2;
                if measure(mid)? < target {
                    low = mid;
                } else {
                    high = mid;
                }
            }
        }
        // A too-long baseline cannot be repaired by truncating the fixed task.
        // Still consider local BPE boundary changes without assuming monotonicity.
        let center = if baseline < target { high } else { 0 };
        let start = center.saturating_sub(LOCAL_RADIUS);
        let end = center.saturating_add(LOCAL_RADIUS).min(max_background);
        for bytes in start..=end {
            if measure(bytes)? == target {
                break;
            }
        }
        let _ = high_count;
    }
    let mut best = best.ok_or_else(|| format!(
        "no measured candidate fits bucket {} and prompt limit {max_tokens} (target {target}, {trials} trials)",
        bucket_name(wanted_bucket)
    ))?;
    best.trials = trials;
    Ok(best)
}

fn prefix(text: &str, mut bytes: usize) -> &str {
    bytes = bytes.min(text.len());
    while !text.is_char_boundary(bytes) {
        bytes -= 1;
    }
    &text[..bytes]
}

fn user_prompt(source: &Source<'_>, background: &str, bytes: usize) -> String {
    let selected = prefix(background, bytes);
    if selected.is_empty() {
        return source.fixed_prompt.clone();
    }
    format!(
        "Background reference material (deterministic corpus repetition; the final task below controls the answer):\n{selected}\n\nFinal task:\n{}",
        source.fixed_prompt
    )
}

fn render(tokenizer: &Tokenizer, template: &str, user: &str) -> Result<String> {
    JinjaChatFrame {
        tokenizer,
        template,
        system: Some(""),
        user,
        enable_thinking: false,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    }
    .render()
}

// create_dir, unlike create_dir_all, is an atomic claim and refuses even an
// existing empty directory, regular file, or dangling symlink.
fn reserve_output(path: &Path) -> Result<()> {
    fs::create_dir(path)
        .map_err(|e| format!("cannot create NEW output directory {}: {e}", path.display()))
}

// Publish complete file contents atomically without a rename-overwrite race.
// Both links are in our newly claimed directory and on the same filesystem.
fn publish(path: &Path, bytes: &[u8]) -> Result<()> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or("output name is not UTF-8")?;
    let pending = path.with_file_name(format!(".{name}.pending"));
    let operation = || -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pending)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::hard_link(&pending, path)?;
        fs::remove_file(&pending)?;
        Ok(())
    };
    operation().map_err(|e| format!("cannot publish {} without overwrite: {e}", path.display()))
}

fn run(args: Args) -> Result<()> {
    match fs::symlink_metadata(&args.out) {
        Ok(_) => return Err(format!("output already exists: {}", args.out.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "cannot inspect output {}: {error}",
                args.out.display()
            ))
        }
    }
    let corpus_bytes = fs::read(&args.corpus).map_err(|e| format!("read corpus: {e}"))?;
    let corpus: Value =
        serde_json::from_slice(&corpus_bytes).map_err(|e| format!("parse corpus: {e}"))?;
    let sources = sources(&corpus)?;
    let model_path = fs::canonicalize(&args.model).map_err(|e| format!("resolve model: {e}"))?;
    if !model_path.is_file() {
        return Err("--model must be an HFQ file, not a model directory".into());
    }
    let model_str = model_path.to_str().ok_or("model path is not UTF-8")?;
    let source = ModelSource::from_path(model_str)?;
    let ModelSource::Hfq(hfq) = &source else {
        return Err("model is not an HFQ artifact".into());
    };
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| e.to_string())?;
    let template = hfq
        .chat_template()
        .ok_or("HFQ artifact has no chat template")?;
    let mut files = Vec::new();
    let mut entries = Vec::new();
    for source in &sources {
        let max_prompt = args
            .n_ctx
            .checked_sub(CONTEXT_RESERVE)
            .and_then(|n| n.checked_sub(source.decode_tokens))
            .ok_or("--n-ctx cannot hold the 64-token reserve plus corpus decode budget")?;
        for (index, target) in args.targets.iter().copied().enumerate() {
            if target > max_prompt {
                return Err(format!(
                    "target {target} exceeds prompt limit {max_prompt} for {}",
                    source.class
                ));
            }
            let max_background = target
                .checked_mul(64)
                .ok_or("background search bound overflow")?;
            if source.background.is_empty() {
                return Err("source background must not be empty".into());
            }
            let repeats = max_background / source.background.len() + 1;
            let background = source.background.repeat(repeats);
            let fitted = fit(
                |bytes| {
                    let user = user_prompt(source, &background, bytes);
                    Ok(tokenizer
                        .encode(&render(&tokenizer, &template, &user)?)
                        .len())
                },
                target,
                max_prompt,
                max_background,
            )?;
            let user = user_prompt(source, &background, fitted.background_bytes);
            let raw = render(&tokenizer, &template, &user)?;
            let tokens = tokenizer.encode(&raw);
            if tokens.len() != fitted.actual_tokens
                || bucket_index(tokens.len()) != index
                || tokens.len() > max_prompt
            {
                return Err(
                    "final whole-frame tokenization disagrees with fit or context budget".into(),
                );
            }
            let filename = format!(
                "{}-{}.txt",
                source.class.to_ascii_lowercase(),
                bucket_name(index)
            );
            let messages =
                json!([{"role": "system", "content": ""}, {"role": "user", "content": user}]);
            entries.push(json!({
                "file": filename, "class": source.class,
                "bucket": bucket_name(index), "target_tokens": target,
                "actual_tokens": tokens.len(), "exact_target": tokens.len() == target,
                "target_delta": tokens.len() as i128 - target as i128,
                "prompt_sha256": sha256(raw.as_bytes()),
                "token_sha256": token_sha256(&tokens),
                "token_hash_encoding": "u32 little-endian concatenated, no length prefix",
                "n_ctx": args.n_ctx, "context_reserve_tokens": CONTEXT_RESERVE,
                "decode_tokens": source.decode_tokens,
                "fit": {
                    "method": "bounded exponential bracketing, bisection, exhaustive local byte-prefix search",
                    "selection": "closest actually measured candidate in the requested bucket; ties use fewer tokens then fewer background bytes; not a proof of global BPE monotonicity",
                    "trials": fitted.trials, "local_radius_bytes": LOCAL_RADIUS,
                    "max_background_bytes": max_background,
                    "background_bytes": prefix(&background, fitted.background_bytes).len(),
                    "background_repetition": "repeat source text byte-for-byte; retain UTF-8-safe prefix in user content only"
                },
                "source_cells": [source.cell["label"].clone()],
                "source_cell": source.cell,
                "source_messages_sha256": sha256(&serde_json::to_vec(&source.cell["messages"]).map_err(|e| e.to_string())?),
                "source_user_sha256": sha256(source.original_user.as_bytes()),
                "background_source_sha256": sha256(source.background.as_bytes()),
                "fixed_task_sha256": sha256(source.fixed_prompt.as_bytes()),
                "derivation": source.derivation,
                "messages": messages,
                "messages_sha256": sha256(&serde_json::to_vec(&messages).map_err(|e| e.to_string())?),
                "render": {"api": "JinjaChatFrame::render", "system": "", "enable_thinking": false,
                           "bos_token": null, "reasoning_strength": null, "reasoning_effort": null}
            }));
            files.push((filename, raw));
        }
    }
    let manifest = json!({
        "schema": "qwen4-cost-prompts-v1",
        "fixture_law": "Generated COPY/PROSE cost fixtures derived from real corpus messages; NOT byte-identical to the original 22-cell benchmark and NOT benchmark measurements.",
        "corpus": {"path": fs::canonicalize(&args.corpus).map_err(|e| e.to_string())?,
                   "sha256": sha256(&corpus_bytes)},
        "model": {"path": model_path, "metadata_sha256": sha256(hfq.metadata_json.as_bytes()),
                  "chat_template_sha256": sha256(template.as_bytes()), "arch_id": hfq.arch_id,
                  "identity_scope": "model path and metadata/template hashes; weight bytes were not hashed or copied"},
        "tokenizer": "Tokenizer::from_hfq_metadata; Tokenizer::encode on complete raw rendered text",
        "targets": args.targets, "n_ctx": args.n_ctx,
        "prompt_files": files.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        "fixtures": entries
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    reserve_output(&args.out)?;
    for (filename, raw) in files {
        publish(&args.out.join(filename), raw.as_bytes())?;
    }
    publish(&args.out.join("manifest.json"), &manifest_bytes)?;
    fs::File::open(&args.out)
        .and_then(|file| file.sync_all())
        .map_err(|e| format!("sync output directory: {e}"))?;
    println!("{}", args.out.join("manifest.json").display());
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args == ["--help"] || args == ["-h"] {
        println!("{USAGE}");
        return;
    }
    if let Err(error) = parse_args(args.into_iter()).and_then(run) {
        eprintln!("qwen4_cost_prompts: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    // SYNTHETIC tests only: these are not real tokenizer or model measurements.
    #[test]
    fn synthetic_bucket_edges_match_cost_runner() {
        for (tokens, bucket) in [
            (1, 0),
            (2048, 0),
            (2049, 1),
            (8192, 1),
            (8193, 2),
            (16384, 2),
            (16385, 3),
            (24576, 3),
        ] {
            assert_eq!(bucket_index(tokens), bucket);
        }
    }

    #[test]
    fn synthetic_fit_exact_and_nearest_stay_in_bucket() {
        let exact = fit(|n| Ok(80 + n), 1536, 2000, 4096).unwrap();
        assert_eq!(exact.actual_tokens, 1536);
        let nearest = fit(|n| Ok(80 + 2 * n), 1537, 2000, 4096).unwrap();
        assert_eq!(nearest.actual_tokens, 1536);
        assert!(nearest.trials < 250);
        let edge = fit(|n| Ok(2048 + 2 * n), 2049, 8192, 4096).unwrap();
        assert_eq!(edge.actual_tokens, 2050); // 2048 is closer but in the wrong bucket.
        assert!(fit(|_| Ok(8193), 6144, 8192, 4096).is_err());
    }

    #[test]
    fn synthetic_fit_handles_local_nonmonotone_counts_and_utf8() {
        let fit = fit(
            |n| Ok(if n == 723 { 1537 } else { 80 + 2 * n }),
            1537,
            2048,
            4096,
        )
        .unwrap();
        assert_eq!(fit.actual_tokens, 1537);
        assert_eq!(fit.background_bytes, 723);
        assert_eq!(prefix("aé河z", 2), "a");
        assert_eq!(prefix("aé河z", 5), "aé");
    }

    #[test]
    fn synthetic_output_claim_and_publication_never_overwrite() {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "qwen4-cost-prompts-synthetic-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        reserve_output(&dir).unwrap();
        assert!(reserve_output(&dir).is_err());
        let file = dir.join("fixture.txt");
        publish(&file, b"first").unwrap();
        assert!(publish(&file, b"second").is_err());
        assert_eq!(fs::read(&file).unwrap(), b"first");
        assert!(reserve_output(&file).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
}
