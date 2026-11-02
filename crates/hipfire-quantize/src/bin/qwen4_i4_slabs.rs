//! Build the Flash-Next MQ3-I4 / MQ2-I4 expert-slab artifact (DESIGN F2).
//!
//! Takes the pinned MQ4-XTS HFQ, an R1 allocation point (or a uniform choice)
//! and the per-expert final-code NPZ records, and writes the slab artifact plus
//! a JSON receipt. Verification is built in and cannot be disabled; see
//! `hipfire_quantize::i4_slabs`.
//!
//! Experts can also be carried as E8P-I4 (qt135 gate_up R256 / qt136 down R128,
//! wire codec 3) from the existing solved `vq-e8p` records:
//!
//! ```text
//! qwen4_i4_slabs --source qwen3.8-flash-next-gptq3.mq4 \
//!     --alloc alloc.c24.json --point c24 --cold-e8p \
//!     --records-e8p arms/vq-e8p --manifest-e8p arms/vq-e8p/manifest.json \
//!     --manifest-e8p-sha256 a664a9e8509aa1e98a4d25f3057b95f0a71b54c3c3b9cb505bc98d7f558b8f3b \
//!     --out c24.hfq --receipt c24.receipt.json --threads 8
//! ```
//!
//! `--cold-e8p` reads allocation index 0 (cold) as E8P-I4 and index 2 (hot) as
//! the copied MQ4-XTS slice. Experts whose scale trips the writer guard
//! (`d < 2^-14` or non-finite) or that the solver left MQ4 are written as
//! MQ4-XTS and counted in the receipt's `e8p_i4` block; never reinterpreted.
//!
//! ```text
//! qwen4_i4_slabs --source qwen3.8-flash-next-gptq3.mq4 \
//!     --alloc alloc.ep2.json --point free-1GiB-per-rank \
//!     --records-w3 q/w3-gptq3 --records-w2 q/w2-gptq3 \
//!     --manifests-w3 pack/w3-gptq3 --manifests-w2 pack/w2-gptq3 \
//!     --cross-check-packs-w3 pack/w3-gptq3 --cross-check-packs-w2 pack/w2-gptq3 \
//!     --out mixed.hfq --receipt mixed.receipt.json --threads 8
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use hipfire_arch_qwen4::expert_slabs::ExpertFormat;
use hipfire_arch_qwen4::ExpertKeep;
use hipfire_quantize::i4_records::I4RecordShapes;
use hipfire_quantize::i4_slabs::{
    apply_keep_tiers, load_allocation_cold_e8p_with_keep, load_allocation_with_keep,
    size_i4_expert_slabs, uniform_formats, write_i4_expert_slabs_with_keep, AllocationSource,
    E8pRecordProvider, E8pVqDirProvider, ExpertRecordProvider, NpzDirProvider, SlabGeometry,
    SlabWriteConfig,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Uniform {
    W3,
    W2,
    E8p,
    Mq4,
}

#[derive(Parser, Debug)]
#[command(
    name = "qwen4_i4_slabs",
    about = "Pack MQ3-I4/MQ2-I4 final-code records and E8P-I4 solved records into a Flash-Next expert-slab HFQ"
)]
struct Args {
    /// Pinned MQ4-XTS source HFQ.
    #[arg(long)]
    source: PathBuf,
    /// hipfire.coder.keep.v1 original-id membership; optional tiers override allocation.
    #[arg(long)]
    expert_keep: Option<PathBuf>,
    /// Print exact base bytes without writing output or receipt (use a PLE-split source for a split base).
    #[arg(long)]
    size_only: bool,
    /// R1 allocation JSON (`alloc.ep2.json`); needs --point.
    #[arg(long, requires = "point", conflicts_with = "uniform")]
    alloc: Option<PathBuf>,
    /// Allocation name, e.g. `free-1GiB-per-rank`.
    #[arg(long, requires = "alloc")]
    point: Option<String>,
    /// With --alloc: read profile index 0 as E8P-I4 and 2 as MQ4-XTS (index 1 is an error).
    #[arg(long, requires = "alloc")]
    cold_e8p: bool,
    /// Use one format for every expert instead of --alloc/--point.
    #[arg(long, value_enum)]
    uniform: Option<Uniform>,
    /// Directory with `L{L}/E{e}.npz` MQ3-I4 records.
    #[arg(long)]
    records_w3: Option<PathBuf>,
    /// Directory with `L{L}/E{e}.npz` MQ2-I4 records.
    #[arg(long)]
    records_w2: Option<PathBuf>,
    /// Directory with the solved E8P records `L{L}.gate_up_proj.bin` / `L{L}.down_proj.bin`.
    #[arg(long, requires_all = ["manifest_e8p", "manifest_e8p_sha256"])]
    records_e8p: Option<PathBuf>,
    /// The `vq-e8p` `manifest.json` that describes --records-e8p.
    #[arg(long, requires = "records_e8p")]
    manifest_e8p: Option<PathBuf>,
    /// Pinned sha256 (hex) of --manifest-e8p.
    #[arg(long, requires = "records_e8p")]
    manifest_e8p_sha256: Option<String>,
    /// Directory with `L{L}/manifest.json` (npz_sha256 per expert), W3.
    #[arg(long, requires = "records_w3")]
    manifests_w3: Option<PathBuf>,
    /// Directory with `L{L}/manifest.json` (npz_sha256 per expert), W2.
    #[arg(long, requires = "records_w2")]
    manifests_w2: Option<PathBuf>,
    /// Python R1 packs (`L{L}/E{e}.bin` + manifest) to compare byte-for-byte, W3.
    #[arg(long, requires = "records_w3")]
    cross_check_packs_w3: Option<PathBuf>,
    /// Python R1 packs to compare byte-for-byte, W2.
    #[arg(long, requires = "records_w2")]
    cross_check_packs_w2: Option<PathBuf>,
    /// Output artifact path (must not exist).
    #[arg(long)]
    out: PathBuf,
    /// JSON receipt path (must not exist).
    #[arg(long)]
    receipt: PathBuf,
    /// Packing worker threads.
    #[arg(long, default_value_t = 8)]
    threads: usize,
    /// Assert the sha256 (hex) of --source.
    #[arg(long)]
    source_sha256: Option<String>,
}

fn provider(
    records: Option<PathBuf>,
    manifests: Option<PathBuf>,
    packs: Option<PathBuf>,
    format: ExpertFormat,
) -> Option<NpzDirProvider> {
    let mut p = NpzDirProvider::new(records?, format, I4RecordShapes::FN);
    if let Some(m) = manifests {
        p = p.with_manifests(m);
    }
    if let Some(c) = packs {
        p = p.with_cross_check_packs(c);
    }
    Some(p)
}

fn run(args: Args) -> Result<(), String> {
    for (what, path) in [("--out", &args.out), ("--receipt", &args.receipt)] {
        if path.symlink_metadata().is_ok() {
            return Err(format!(
                "refusing to overwrite existing {what} {}",
                path.display()
            ));
        }
    }
    let geometry = SlabGeometry::FN;
    let keep_json = args
        .expert_keep
        .as_ref()
        .map(|path| {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|e| e.to_string())
        })
        .transpose()?;
    let keep = keep_json
        .as_ref()
        .map(ExpertKeep::from_coder_keep_file)
        .transpose()?;
    if keep
        .as_ref()
        .is_some_and(|k| k.n_experts() != geometry.experts || k.n_layers() != geometry.layers)
    {
        return Err("keep geometry must be 48 layers x 512 original ids".into());
    }
    if keep.as_ref().is_some_and(|k| {
        (0..k.n_layers()).any(|l| k.kept(l).len() < 10)
            || k.mtp_kept().is_some_and(|ids| ids.len() < 10)
    }) {
        return Err("every kept trunk/MTP layer must retain at least top_k=10 experts".into());
    }
    let (mut formats, allocation) = match (&args.alloc, &args.point, args.uniform) {
        (Some(alloc), Some(point), None) => {
            if args.cold_e8p {
                let a = load_allocation_cold_e8p_with_keep(alloc, point, geometry, keep.as_ref())
                    .map_err(|e| e.to_string())?;
                (
                    a.formats,
                    AllocationSource::ColdE8p {
                        file_sha256: a.file_sha256,
                        point: a.point,
                    },
                )
            } else {
                let a = load_allocation_with_keep(alloc, point, geometry, keep.as_ref())
                    .map_err(|e| e.to_string())?;
                (
                    a.formats,
                    AllocationSource::File {
                        file_sha256: a.file_sha256,
                        point: a.point,
                    },
                )
            }
        }
        (None, None, Some(u)) => {
            let f = match u {
                Uniform::W3 => ExpertFormat::Mq3I4,
                Uniform::W2 => ExpertFormat::Mq2I4,
                Uniform::E8p => ExpertFormat::E8pI4,
                Uniform::Mq4 => ExpertFormat::Mq4Xts,
            };
            (uniform_formats(geometry, f), AllocationSource::Uniform(f))
        }
        _ => return Err("give either --alloc with --point, or --uniform w3|w2|mq4".into()),
    };
    if let (Some(root), Some(keep)) = (&keep_json, &keep) {
        apply_keep_tiers(root, keep, &mut formats).map_err(|e| e.to_string())?;
    }
    // Pruned packs only: a dense run keeps its former error timing and never
    // stats the full 48 x 512 record grid up front.
    if let Some(k) = keep.as_ref() {
        for item in 0..formats.len() {
            let (layer, expert) = (item / geometry.experts, item % geometry.experts);
            if !k.is_kept(layer, expert) {
                continue;
            }
            let records = match formats[item] {
                ExpertFormat::Mq3I4 => args.records_w3.as_ref(),
                ExpertFormat::Mq2I4 => args.records_w2.as_ref(),
                // E8P records are one pair of per-layer files, verified by the provider at open.
                ExpertFormat::E8pI4 | ExpertFormat::Mq4Xts => continue,
            }
            .ok_or_else(|| format!("missing records directory for L{layer}/E{expert}"))?;
            let path = records.join(format!("L{layer}/E{expert}.npz"));
            if !path.is_file() {
                return Err(format!("missing record {}", path.display()));
            }
        }
    }
    let w3 = provider(
        args.records_w3.clone(),
        args.manifests_w3.clone(),
        args.cross_check_packs_w3.clone(),
        ExpertFormat::Mq3I4,
    );
    let w2 = provider(
        args.records_w2.clone(),
        args.manifests_w2.clone(),
        args.cross_check_packs_w2.clone(),
        ExpertFormat::Mq2I4,
    );
    let e8p = match (
        &args.records_e8p,
        &args.manifest_e8p,
        &args.manifest_e8p_sha256,
    ) {
        (Some(dir), Some(manifest), Some(sha)) => Some(
            E8pVqDirProvider::open(dir.clone(), manifest.clone(), sha, I4RecordShapes::FN)
                .map_err(|e| e.to_string())?,
        ),
        _ => None,
    };
    let config = SlabWriteConfig {
        geometry,
        source: &args.source,
        out: &args.out,
        formats: &formats,
        allocation,
        w3: w3.as_ref().map(|p| p as &dyn ExpertRecordProvider),
        w2: w2.as_ref().map(|p| p as &dyn ExpertRecordProvider),
        e8p: e8p.as_ref().map(|p| p as &dyn E8pRecordProvider),
        threads: args.threads,
        expected_source_sha256: args.source_sha256.as_deref(),
        progress: true,
    };
    let receipt = if args.size_only {
        size_i4_expert_slabs(&config, keep.as_ref())
    } else {
        write_i4_expert_slabs_with_keep(&config, keep.as_ref())
    }
    .map_err(|e| e.to_string())?;
    if args.size_only {
        println!(
            "{}",
            serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    let text = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
    std::fs::write(&args.receipt, text + "\n")
        .map_err(|e| format!("writing {}: {e}", args.receipt.display()))?;
    println!(
        "wrote {} ({} bytes, sha256 {})\nreceipt {}",
        args.out.display(),
        receipt["output"]["bytes"],
        receipt["output"]["sha256"].as_str().unwrap_or("?"),
        args.receipt.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("qwen4_i4_slabs: {e}");
            ExitCode::FAILURE
        }
    }
}
