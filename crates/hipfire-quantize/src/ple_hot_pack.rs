// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire-quantize ple-hot-pack`: offline, no-GPU conversion of a Qwen4 /
//! Flash-Next model plus its `<model>.plehot` rank sidecar into the
//! `<model>.plehotrows` hot-row pack.
//!
//! The pack holds the hottest PLE rows (buckets `0..kept_buckets` of the
//! sidecar), copied verbatim from the model file's PLE shards into one file in
//! the order the runtime's hot-row store keeps them, so the runtime can serve
//! them through the kernel page cache.  The format is documented in
//! `hipfire_runtime::ple_hot_pack`.

use clap::Parser;
use hipfire_arch_qwen4::admit_hfqm_artifact;
use hipfire_arch_qwen4::bundle::ple_hot_plan;
use hipfire_arch_qwen4::Qwen4PleRows;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::ple_hot_pack::write_ple_hot_pack_with_progress;
use hipfire_runtime::ple_hot_store::{PleHotRank, BUCKETS, DEFAULT_TIER_BUCKETS};
use hipfire_runtime::ple_sidecar::{resolve_and_attach_ple_rows, ResolvePolicy};
use serde_json::{json, Value};
use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(
    name = "hipfire-quantize ple-hot-pack",
    about = "Cut the `<model>.plehotrows` PLE hot-row pack from a Qwen4 model and its \
             `<model>.plehot` rank sidecar",
    long_about = "Copy the hottest PLE rows of a Qwen4 / Flash-Next model (buckets 0..kept of the \
                  `<model>.plehot` sidecar written by `ple-hot-rank`) verbatim into a separate \
                  `<model>.plehotrows` file, grouped into tiers of --tier-group buckets, so the \
                  runtime reads them through the kernel page cache instead of anonymous RAM.\n\n\
                  The number of kept buckets is derived from --gib: the largest K <= 256 whose \
                  rows fit that many GiB."
)]
pub(crate) struct HotPackArgs {
    /// Qwen4 / Flash-Next HFQM model file the rows are cut from.
    #[arg(long, value_name = "MODEL.mq4")]
    pub model: PathBuf,

    /// Rank sidecar (default `<model>.plehot`).
    #[arg(long, value_name = "MODEL.mq4.plehot")]
    pub rank: Option<PathBuf>,

    /// Destination pack (default `<rows source>.plehotrows`: the model file,
    /// or the chosen PLE sidecar).
    #[arg(long, value_name = "ROWS.plehotrows")]
    pub out: Option<PathBuf>,

    /// PLE rows to cut the pack from: `q4` / `q8` use the model's
    /// `<model>.ple-q4` / `<model>.ple-q8` sidecar (a monolithic model's q8 rows
    /// are its own shards).  Default: the model's embedded rows.  A rank-order
    /// q4 table is its own pack and is refused.
    #[arg(long, value_name = "q4|q8", value_parser = ["q4", "q8"])]
    pub ple_format: Option<String>,

    /// Size budget of the pack's row data, GiB (2^30 bytes).
    #[arg(long, value_name = "GIB", default_value_t = 40.0)]
    pub gib: f64,

    /// Buckets per tier (2..=256).
    #[arg(long, value_name = "N", default_value_t = DEFAULT_TIER_BUCKETS as u64,
          value_parser = clap::value_parser!(u64).range(2..=256))]
    pub tier_group: u64,
}

/// `<model><suffix>` (suffix appended to the file name, extension kept).
fn sibling(model: &std::path::Path, suffix: &str) -> PathBuf {
    let mut name: OsString = model.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Entry point for `hipfire-quantize ple-hot-pack …` (`argv` excludes the
/// program name and the subcommand word). Returns the process exit code.
pub(crate) fn run(argv: Vec<String>) -> i32 {
    let args = HotPackArgs::parse_from(
        std::iter::once("hipfire-quantize ple-hot-pack".to_string()).chain(argv),
    );
    match run_inner(&args) {
        Ok(summary) => {
            println!("{summary}");
            0
        }
        Err(e) => {
            eprintln!("error: ple-hot-pack: {e}");
            1
        }
    }
}

/// Largest K <= 256 whose rows fit `gib` GiB:
/// `floor(gib * 2^30 * 256 / (row_bytes * rows))`.
fn kept_buckets_for(gib: f64, padded_rows: u64, row_bytes: usize) -> usize {
    let bytes = gib * (1u64 << 30) as f64;
    let kept = (bytes * BUCKETS as f64 / (row_bytes as f64 * padded_rows as f64)).floor();
    kept.min(BUCKETS as f64) as usize
}

fn run_inner(args: &HotPackArgs) -> Result<Value, String> {
    let t0 = Instant::now();
    if !args.gib.is_finite() || args.gib <= 0.0 {
        return Err(format!("--gib must be a positive number, got {}", args.gib));
    }
    let rank_path = args
        .rank
        .clone()
        .unwrap_or_else(|| sibling(&args.model, ".plehot"));

    let mut hfq = HfqFile::open(&args.model)
        .map_err(|e| format!("open model {}: {e}", args.model.display()))?;
    // Default: the model's own (embedded) PLE rows, as before sidecars
    // existed.  `--ple-format` resolves a sidecar through the loader's rules
    // (never deriving: run `hipfire ple derive` first); the GPU arch only
    // matters for `auto`, which is not offered here.
    if let Some(request) = args
        .ple_format
        .as_deref()
        .and_then(hipfire_config::memory::PleFormatRequest::parse)
    {
        resolve_and_attach_ple_rows(
            &mut hfq,
            &args.model,
            request,
            "",
            ResolvePolicy {
                derive_missing_q4: false,
                progress: None,
            },
        )
        .map_err(|e| format!("PLE rows of {}: {e}", args.model.display()))?;
    }
    let receipt =
        admit_hfqm_artifact(&hfq).map_err(|e| format!("admit {}: {e}", args.model.display()))?;
    if let Qwen4PleRows::Sidecar { path, .. } = &receipt.ple_rows {
        if receipt.ple_rows.rank_table().is_some() {
            return Err(format!(
                "{} is a rank-order PLE table: its rows are already stored hottest-first with \
                 their rank, so it is its own hot-row pack and there is nothing to cut",
                path.display()
            ));
        }
    }
    let metadata = receipt.ple.as_ref().ok_or_else(|| {
        "PLE hot-pack generation requires PLE hash metadata; this profile has no PLE".to_string()
    })?;
    let descriptors = receipt
        .ple_row_descriptors(&hfq)
        .map_err(|e| format!("PLE rows of {}: {e}", args.model.display()))?;
    let (rows_path, rows_bytes) = receipt
        .ple_rows_source(&hfq)
        .map_err(|e| format!("PLE rows of {}: {e}", args.model.display()))?;
    let plan = ple_hot_plan(metadata, &descriptors, &rows_path, rows_bytes)?
        .ok_or_else(|| "PLE rows are neither Q8F16 nor Q4LM".to_string())?;
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| sibling(&plan.source_path, ".plehotrows"));
    let row_bytes = plan.encoded_row_bytes;

    let rank = PleHotRank::open(&rank_path, plan.padded_rows, &plan.fingerprint_hex)?;
    rank.verify()?;
    let kept = kept_buckets_for(args.gib, plan.padded_rows, row_bytes);
    if kept == 0 {
        return Err(format!(
            "--gib {} holds no bucket ({} rows of {} B need {:.2} GiB per 1/256 of the table)",
            args.gib,
            plan.padded_rows,
            row_bytes,
            (plan.padded_rows * row_bytes as u64) as f64 / BUCKETS as f64 / (1u64 << 30) as f64
        ));
    }
    eprintln!(
        "ple-hot-pack: padded_rows={} shards={} kept_buckets={kept}/256 tier_group={} -> {}",
        plan.padded_rows,
        plan.shard_offsets.len(),
        args.tier_group,
        out.display()
    );

    // Progress every ~5% of the rows scanned.
    let total_scan_bytes = plan.padded_rows as f64 * row_bytes as f64;
    let mut next_percent = 5u64;
    let t_write = Instant::now();
    let header = write_ple_hot_pack_with_progress(
        &plan.source_path,
        &plan.shard_offsets,
        plan.rows_per_shard,
        plan.row_encoding,
        plan.row_width,
        &plan.fingerprint_hex,
        &rank,
        kept,
        args.tier_group as usize,
        &out,
        &mut |done, total| {
            let percent = done * 100 / total.max(1);
            if percent >= next_percent {
                next_percent = (percent / 5 + 1) * 5;
                let elapsed = t_write.elapsed().as_secs_f64();
                eprintln!(
                    "ple-hot-pack: {done}/{total} rows ({percent}%) {elapsed:.0}s {:.2} GB/s",
                    total_scan_bytes * done as f64 / total as f64 / 1e9 / elapsed.max(1e-9)
                );
            }
        },
    )?;
    let elapsed = t_write.elapsed().as_secs_f64();
    Ok(json!({
        "out": out.display().to_string(),
        "model": args.model.display().to_string(),
        "rows_source": plan.source_path.display().to_string(),
        "row_encoding": plan.row_encoding.dtype_name(),
        "rank": rank_path.display().to_string(),
        "kept_buckets": header.kept_buckets,
        "tier_group": header.tier_group,
        "tiers": header.tier_rows.len(),
        "padded_rows": header.padded_rows,
        "data_offset": header.data_offset,
        "data_bytes": header.data_bytes,
        "elapsed_s": t0.elapsed().as_secs_f64(),
        "read_gb_per_s": total_scan_bytes / 1e9 / elapsed.max(1e-9),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kept_buckets_follow_the_budget() {
        // 320M rows * 170 B = 54.4 GB ≈ 50.67 GiB for all 256 buckets.
        let rows = 320_000_000u64;
        assert_eq!(kept_buckets_for(1000.0, rows, 170), 256);
        let k = kept_buckets_for(40.0, rows, 170);
        assert!((1..256).contains(&k));
        assert!(k as f64 * rows as f64 * 170.0 / 256.0 <= 40.0 * (1u64 << 30) as f64);
        assert!((k + 1) as f64 * rows as f64 * 170.0 / 256.0 > 40.0 * (1u64 << 30) as f64);
        assert_eq!(kept_buckets_for(0.0001, rows, 170), 0);
        // Wider rows keep fewer buckets in the same budget.
        assert!(kept_buckets_for(40.0, rows, 340) < k);
    }

    #[test]
    fn default_paths_append_to_the_model_name() {
        assert_eq!(
            sibling(std::path::Path::new("/m/x.mq4"), ".plehotrows"),
            PathBuf::from("/m/x.mq4.plehotrows")
        );
    }

    #[test]
    fn ple_format_accepts_q4_and_q8_only() {
        let parse = |extra: &[&str]| {
            HotPackArgs::try_parse_from(["ple-hot-pack", "--model", "m.mq4"].iter().chain(extra))
        };
        assert_eq!(parse(&[]).unwrap().ple_format, None);
        assert_eq!(
            parse(&["--ple-format", "q4"])
                .unwrap()
                .ple_format
                .as_deref(),
            Some("q4")
        );
        assert!(parse(&["--ple-format", "q8"]).is_ok());
        assert!(parse(&["--ple-format", "auto"]).is_err());
        assert!(parse(&["--ple-format", "bf16"]).is_err());
    }
}
