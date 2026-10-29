// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_placement_receipt — CPU-only receipt of the product's expert placement on a real
//! Flash-Next artifact, for the behavior-preserving move of the unset-layer-count
//! placement into one shared resolver (`expert_residency::incumbent_placement_layers`).
//!
//! No GPU is initialised (run it with `HIP_VISIBLE_DEVICES=-1`): the artifact is opened,
//! admitted and its manifest sized exactly as the product loader sizes it. For each GDN
//! state format (`q8` and `q8_ef`) the base reserve of every case is COMPUTED from the
//! artifact's own config with the CURRENT accounting the product loader calls
//! (`state_census` -> `Qwen4State::device_bytes`, which charges the `q8_ef` live
//! residual plane once, plus `auto_vram_reserve`, the prefill workspace terms with
//! `qwen4_dense_gdn_admits`, the runtime overhead and the F16 shadows); only free VRAM,
//! the chunk and the stage switch are fixed CASES taken from real product placement logs
//! (`qwen4 auto expert placement: ... MiB free ...`). For every (format, case) the receipt
//! records
//!   * `legacy`: the product loader's pre-extraction inline composition of the placement
//!     primitives (fits-fully-resident, else `auto`), as `carriers.rs` wrote it at e1,
//!   * `shared`: `residency::incumbent_placement_layers`, the function the product loader
//!     and the lab loaders now resolve through,
//!   * the placement each implies on the REAL manifest (`place_routed_experts`): the
//!     host-placed trunk layers, host tensor count/bytes and `placement_digest`.
//! Per format it also records the state census and, for `q8_ef`, proves the live residual
//! is exactly one F16 plane per GDN layer over `q8` (not mirrored in the snapshot arena).
//! Exit 0 only if `legacy == shared` on every case and every requested binding holds
//! (model header sha256 == the layout proof; the git source proof of `--bind-source`).
//!
//! Real limits: this proves the resolver and the placement of the real layout under the
//! current accounting, not a card's free VRAM (the case inputs are logged MiB, floored).
//! `placement_inputs` itself takes a live `Gpu` and is not executed here: the reserve is
//! the CPU composition of the same accounting primitives for the fixture below (QSA q8
//! K/V, legacy backend, native MTP off, no pooled-BF16/F16 arenas, so the QSA selector
//! and gather terms are 0; the gfx1201 F16 shadows assume `HIPFIRE_QWEN4_F16_WMMA_GFX1201`
//! at its default). `--bind-identity` (a retired binary identity) is only a historical
//! label: no baseline binary is executed.
//!
//! Usage:
//!   HIP_VISIBLE_DEVICES=-1 cargo run --release -p hipfire-arch-qwen4 --features lab \
//!     --example qwen4_placement_receipt -- MODEL.mq4 [--arch gfx1100|gfx1201] \
//!     [--max-seq 8224] [--gdn q8,q8_ef] \
//!     [--case LABEL,FREE_MIB,CHUNK_ROWS,STAGE(0|1)]... \
//!     [--layout-proof layout-proof.json] [--bind-source baseline-source.json] \
//!     [--git-dir REPO] [--bind-identity HISTORICAL-IDENTITY.json \
//!      --source-head-prefix ad25c6c1f] [--json OUT.json]

use hipfire_arch_qwen4::admit_hfqm_artifact_on;
use hipfire_arch_qwen4::expert_residency as residency;
use hipfire_arch_qwen4::gpu_forward::Qwen4ForwardScratchMode;
use hipfire_arch_qwen4::{
    GdnStateFormat, QsaKvFormat, Qwen4Config, Qwen4ContextCommit, Qwen4KvBackend, Qwen4StateFormat,
};
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::ple_sidecar::{resolve_and_attach_ple_rows, ResolvePolicy};
use hipfire_runtime::weight_manifest::WeightEntry;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::PathBuf;

type R<T> = Result<T, String>;

fn e<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

const MIB: u64 = 1 << 20;
/// Granularity of a VMM arena; the legacy backend this fixture charges ignores it.
const GRANULARITY: usize = 2 << 20;

struct Case {
    label: String,
    free: u64,
    chunk_rows: usize,
    stage: bool,
}

struct Args {
    model: PathBuf,
    arch: String,
    max_seq: usize,
    formats: Vec<(String, GdnStateFormat)>,
    cases: Vec<Case>,
    layout_proof: Option<PathBuf>,
    bind_source: Option<PathBuf>,
    git_dir: PathBuf,
    bind_identity: Option<PathBuf>,
    source_head_prefix: String,
    json: Option<PathBuf>,
}

/// The real XTX hold receipts' placement logs (`g2-xtx-default/hold-20261017T102413Z`):
/// base at the 1536-row chunk, candidate at the 8192-row chunk, floored MiB.
fn default_cases() -> Vec<Case> {
    let case = |label: &str, free_mib: u64, chunk_rows: usize| Case {
        label: label.into(),
        free: free_mib * MIB,
        chunk_rows,
        stage: false,
    };
    vec![
        case("xtx-base-1536", 24522, 1536),
        case("xtx-cand-8192", 24376, 8192),
        case("everything-fits", 160 * 1024, 8192),
    ]
}

fn parse_case(text: &str) -> R<Case> {
    let parts: Vec<&str> = text.split(',').collect();
    let [label, free, chunk, stage] = parts[..] else {
        return Err(format!(
            "--case '{text}': need LABEL,FREE_MIB,CHUNK_ROWS,STAGE"
        ));
    };
    let num = |v: &str| {
        v.parse::<u64>()
            .map_err(|x| format!("--case '{text}': {x}"))
    };
    Ok(Case {
        label: label.into(),
        free: num(free)? * MIB,
        chunk_rows: usize::try_from(num(chunk)?).map_err(e)?,
        stage: match stage {
            "0" => false,
            "1" => true,
            other => return Err(format!("--case '{text}': STAGE is 0 or 1, got '{other}'")),
        },
    })
}

fn parse_formats(text: &str) -> R<Vec<(String, GdnStateFormat)>> {
    text.split(',')
        .map(|name| match name {
            "q8" => Ok((name.to_string(), GdnStateFormat::Q8)),
            "q8_ef" => Ok((name.to_string(), GdnStateFormat::Q8Ef)),
            other => Err(format!("--gdn: '{other}' is q8 or q8_ef")),
        })
        .collect()
}

fn parse_args() -> R<Args> {
    let mut model = None;
    let mut arch = "gfx1100".to_string();
    let mut max_seq = 8224usize;
    let mut formats = parse_formats("q8,q8_ef")?;
    let mut cases = Vec::new();
    let (mut layout_proof, mut bind_source, mut bind_identity, mut json) = (None, None, None, None);
    let mut git_dir = PathBuf::from(".");
    let mut source_head_prefix = "ad25c6c1f".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--arch" => arch = value("--arch")?,
            "--max-seq" => max_seq = value("--max-seq")?.parse().map_err(e)?,
            "--gdn" => formats = parse_formats(&value("--gdn")?)?,
            "--case" => cases.push(parse_case(&value("--case")?)?),
            "--layout-proof" => layout_proof = Some(PathBuf::from(value("--layout-proof")?)),
            "--bind-source" => bind_source = Some(PathBuf::from(value("--bind-source")?)),
            "--git-dir" => git_dir = PathBuf::from(value("--git-dir")?),
            "--bind-identity" => bind_identity = Some(PathBuf::from(value("--bind-identity")?)),
            "--source-head-prefix" => source_head_prefix = value("--source-head-prefix")?,
            "--json" => json = Some(PathBuf::from(value("--json")?)),
            other if other.starts_with("--") => return Err(format!("unknown argument {other}")),
            other if model.is_none() => model = Some(PathBuf::from(other)),
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    if !matches!(arch.as_str(), "gfx1100" | "gfx1201") {
        return Err(format!(
            "--arch {arch}: the CPU reserve fixture is exact for gfx1100 and gfx1201 only"
        ));
    }
    if cases.is_empty() {
        cases = default_cases();
    }
    Ok(Args {
        model: model.ok_or("usage: qwen4_placement_receipt MODEL.mq4 [options]")?,
        arch,
        max_seq,
        formats,
        cases,
        layout_proof,
        bind_source,
        git_dir,
        bind_identity,
        source_head_prefix,
        json,
    })
}

fn file_sha256(path: &std::path::Path) -> R<String> {
    let mut file =
        std::fs::File::open(path).map_err(|x| format!("open {}: {x}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(e)?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// sha256 of the bytes before the data region and the header's data offset (the
/// `sha256_of_bytes_before_data_offset` of `read_layout_proof.py`).
fn header_sha256(path: &std::path::Path) -> R<(String, u64)> {
    let mut file =
        std::fs::File::open(path).map_err(|x| format!("open {}: {x}", path.display()))?;
    let mut head = [0u8; 40];
    file.read_exact(&mut head).map_err(e)?;
    if &head[..4] != b"HFQM" {
        return Err("not an HFQM artifact".into());
    }
    let data_offset = u64::from_le_bytes(head[24..32].try_into().map_err(e)?);
    let mut file = std::fs::File::open(path).map_err(e)?;
    let mut digest = Sha256::new();
    let mut remaining = data_offset;
    let mut buf = vec![0u8; 1 << 20];
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        file.read_exact(&mut buf[..want]).map_err(e)?;
        digest.update(&buf[..want]);
        remaining -= want as u64;
    }
    Ok((format!("{:x}", digest.finalize()), data_offset))
}

/// `git -C dir <args>` stdout, trimmed. Read-only object queries only.
fn git(dir: &std::path::Path, args: &[&str]) -> R<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .map_err(|x| format!("git {args:?}: {x}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Git-proof binding of the baseline SOURCE (no binary is executed): the commit's tree,
/// the runtime commit's tree and the blobs of the files that own the accounting must
/// resolve to the ids the proof names, in the repository at `git_dir`.
fn bind_source(path: &std::path::Path, git_dir: &std::path::Path) -> R<(Value, bool)> {
    let doc: Value = serde_json::from_slice(&std::fs::read(path).map_err(e)?).map_err(e)?;
    let text = |key: &str| doc[key].as_str().unwrap_or("").to_string();
    let (head, tree) = (text("source_head"), text("source_tree"));
    let (runtime, runtime_tree) = (text("runtime_commit"), text("runtime_tree"));
    let tree_of = |rev: &str| {
        git(
            git_dir,
            &["rev-parse", "--verify", &format!("{rev}^{{tree}}")],
        )
    };
    let head_tree = tree_of(&head);
    let runtime_resolved = tree_of(&runtime);
    let mut blobs = Vec::new();
    let mut blobs_ok = true;
    if let Some(map) = doc["git_blob_ids"].as_object() {
        for (file, want) in map {
            let got = git(
                git_dir,
                &["rev-parse", "--verify", &format!("{head}:{file}")],
            );
            let pass = got.as_deref().ok() == want.as_str();
            blobs_ok &= pass;
            blobs.push(json!({"path": file, "want": want, "got": got.ok(), "pass": pass}));
        }
    } else {
        blobs_ok = false;
    }
    let receipt_path = doc["queue_build_receipt"].as_str().map(PathBuf::from);
    let receipt_sha = receipt_path.as_deref().and_then(|p| file_sha256(p).ok());
    let want_receipt_sha = doc["queue_build_receipt_sha256"].as_str();
    let receipt_ok = want_receipt_sha.is_some() && receipt_sha.as_deref() == want_receipt_sha;
    let source_only =
        doc["source_only"] == json!(true) && doc["binary_execution_claim"] == json!(false);
    let pass = head_tree.as_deref().ok() == Some(tree.as_str())
        && runtime_resolved.as_deref().ok() == Some(runtime_tree.as_str())
        && blobs_ok
        && receipt_ok
        && source_only;
    Ok((
        json!({
            "binding": "baseline-source-git-proof", "path": path.display().to_string(),
            "file_sha256": file_sha256(path)?, "git_dir": git_dir.display().to_string(),
            "source_head": head, "source_tree_want": tree, "source_tree_got": head_tree.ok(),
            "runtime_commit": runtime, "runtime_tree_want": runtime_tree,
            "runtime_tree_got": runtime_resolved.ok(),
            "blobs": blobs,
            "queue_build_receipt": doc["queue_build_receipt"],
            "queue_build_receipt_sha256_want": want_receipt_sha,
            "queue_build_receipt_sha256_got": receipt_sha,
            "source_only": source_only, "binary_executed": false,
            "extraction_checkout_head": git(git_dir, &["rev-parse", "HEAD"]).ok(),
            "extraction_checkout_tree": git(git_dir, &["rev-parse", "HEAD^{tree}"]).ok(),
            "pass": pass,
        }),
        pass,
    ))
}

/// What the real artifact and the fixture fix for the reserve.
struct Fixture<'a> {
    config: &'a Qwen4Config,
    weights: &'a [WeightEntry],
    arch: &'a str,
    max_seq: usize,
}

/// Whether the dense GDN scan's scratch is charged for `gdn` on `arch`: exact gfx1201, a Q8 or
/// Q8Ef state (`qwen4_dense_gdn_admits`) and the pinned 16-key/48-value-head, 128-dim,
/// 10240-wide geometry, the product loader's own condition.
fn dense_gdn_applies(arch: &str, gdn: GdnStateFormat, config: &Qwen4Config) -> bool {
    rdna_compute::scratch::qwen4_dense_gdn_admits(arch, gdn)
        && config.linear_num_key_heads == 16
        && config.linear_num_value_heads == 48
        && config.linear_key_head_dim == 128
        && config.linear_value_head_dim == 128
        && 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
            == 10240
}

/// The `auto` reserve the product loader sizes for one rank on a discrete single device with
/// no native MTP head, from the CURRENT accounting (the CPU composition of what
/// `expert_residency::placement_inputs` computes with a live `Gpu`), and the state census
/// behind it.
fn fixture_reserve(
    fx: &Fixture<'_>,
    gdn: GdnStateFormat,
    chunk_rows: usize,
) -> R<(residency::AutoReserve, residency::StateCensus)> {
    let config = fx.config;
    let mode = Qwen4ForwardScratchMode::Full;
    let backend = Qwen4KvBackend::Legacy;
    let format = Qwen4StateFormat {
        qsa: QsaKvFormat::Q8,
        gdn,
        qsa_f16_arena: false,
        qsa_pooled_bf16: false,
    };
    let context =
        Qwen4ContextCommit::for_expert_reserve(backend, false, fx.max_seq, chunk_rows, GRANULARITY);
    let census_context = Qwen4ContextCommit::new(backend, fx.max_seq, fx.max_seq, GRANULARITY);
    let runtime = residency::runtime_overhead_bytes(fx.arch, false, 1);
    let historical = if runtime.is_some() {
        None
    } else {
        Some(
            residency::auto_vram_reserve(
                config,
                &context,
                chunk_rows,
                mode,
                format.layout(),
                None,
                None,
            )
            .map_err(|x| format!("qwen4: {x}"))?,
        )
    };
    let state = residency::state_census(config, format, &census_context, chunk_rows, mode)
        .map_err(|x| format!("qwen4: {x}"))?;
    let f16_shadows =
        if fx.arch == "gfx1201" && chunk_rows >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS {
            residency::f16_shadow_bytes(fx.weights, config.num_hidden_layers)
                .map_err(|x| format!("qwen4: {x}"))?
        } else {
            0
        };
    // The shared-expert INT8 shadows (default on: the process flag, exact gfx1201).
    let shared_i8_shadows = residency::shared_i8_charge(
        rdna_compute::shared_i8::shared_i8_route_enabled(
            std::env::var("HIPFIRE_QWEN4_SHARED_INT8").map_or(true, |v| v != "0"),
            fx.arch,
        ),
        chunk_rows,
        fx.weights,
        config.num_hidden_layers,
    )
    .map_err(|x| format!("qwen4: {x}"))?;
    let dense_gdn = dense_gdn_applies(fx.arch, gdn, config);
    let routed_experts = !config.ffn.is_dense();
    let routed_i4 = routed_experts
        && config.num_experts > 0
        && config.num_experts_per_tok > 0
        && config.moe_intermediate_size > 0
        && config.hidden_size % 256 == 0
        && config.moe_intermediate_size % 128 == 0;
    let dims = rdna_compute::scratch::Qwen4PrefillWorkspaceDims {
        hidden_size: config.hidden_size,
        moe_intermediate_size: config.moe_intermediate_size,
        num_experts: config.num_experts,
        top_k: config.num_experts_per_tok,
        hc_count: config.hc_count,
        hc_lowrank: config.hc_lowrank,
        gdn_format: dense_gdn.then_some(gdn),
        routed_i4,
        include_short_tails: true,
    };
    let workspace = rdna_compute::scratch::qwen4_prefill_workspace_terms(chunk_rows, &dims)
        .ok_or("qwen4: prefill workspace bytes cannot be computed")?;
    let census = residency::ReserveCensus {
        qsa_context: state.qsa_context,
        state: state.state,
        forward: state.forward,
        f16_shadows,
        shared_i8_shadows,
        workspace,
        qsa_selector: 0,
        mtp: 0,
        gather: 0,
        runtime: runtime.unwrap_or(0),
    };
    let reserve =
        residency::AutoReserve::new(historical, census).map_err(|x| format!("qwen4: {x}"))?;
    Ok((reserve, state))
}

/// The product loader's composition for an unset layer count, exactly as `carriers.rs`
/// wrote it before the extraction (`resolve_expert_vram_layers` closure with
/// `placement_charges(layer, false)` then the `Auto` arm with `placement_charges(layer,
/// stage)` and `auto_placement_layers`).
fn legacy_inline(
    free: u64,
    non_expert: u64,
    experts: u64,
    layer: u64,
    layers: usize,
    base: u64,
    stage: bool,
) -> R<Option<usize>> {
    let charges = residency::PlacementCharges::new(layer, false)?;
    let reserve = charges.reserve(base)?;
    if residency::fits_fully_resident(free, non_expert, experts, reserve) {
        return Ok(None);
    }
    let charges = residency::PlacementCharges::new(layer, stage)?;
    residency::auto_placement_layers(free, non_expert, layer, layers, base, &charges).map(Some)
}

fn describe(result: &R<Option<usize>>) -> Value {
    match result {
        Ok(None) => json!({"outcome": "fully-resident"}),
        Ok(Some(layers)) => json!({"outcome": "resident-layers", "resident_layers": layers}),
        Err(error) => json!({"outcome": "error", "error": error}),
    }
}

fn placed(
    weights: &[WeightEntry],
    bytes_of: &dyn Fn(&WeightEntry) -> Option<u64>,
    layers: usize,
    result: &R<Option<usize>>,
) -> R<Value> {
    let Ok(outcome) = result else {
        return Ok(Value::Null);
    };
    let mut weights = weights.to_vec();
    let moved = match outcome {
        Some(resident) => residency::place_routed_experts(&mut weights, *resident),
        None => 0,
    };
    let host = residency::host_placed_trunk_layers(&weights);
    let host_bytes = residency::host_mapped_bytes(&weights, bytes_of)?;
    let resident_bytes = residency::resident_routed_expert_bytes(&weights, bytes_of)?;
    Ok(json!({
        "host_tensors": moved,
        "host_placed_trunk_layers": host,
        "placement_digest": residency::placement_digest(layers, &host),
        "host_expert_bytes": host_bytes,
        "resident_expert_bytes": resident_bytes,
    }))
}

fn real_main() -> R<bool> {
    let args = parse_args()?;
    let mut ok = true;
    let mut bindings = Vec::new();

    // ── model binding ────────────────────────────────────────────────
    let model_size = std::fs::metadata(&args.model).map_err(e)?.len();
    let (header_sha, data_offset) = header_sha256(&args.model)?;
    if let Some(proof) = &args.layout_proof {
        let doc: Value = serde_json::from_slice(&std::fs::read(proof).map_err(e)?).map_err(e)?;
        let want_sha = doc["sha256_of_bytes_before_data_offset"]
            .as_str()
            .unwrap_or("");
        let want_size = doc["model_size_bytes"].as_u64();
        let pass = want_sha == header_sha && want_size == Some(model_size);
        ok &= pass;
        bindings.push(json!({
            "binding": "layout-proof", "path": proof.display().to_string(),
            "file_sha256": file_sha256(proof)?, "header_sha256_match": want_sha == header_sha,
            "size_match": want_size == Some(model_size), "pass": pass,
        }));
    }
    if let Some(source) = &args.bind_source {
        let (binding, pass) = bind_source(source, &args.git_dir)?;
        ok &= pass;
        bindings.push(binding);
    }
    if let Some(identity) = &args.bind_identity {
        let doc: Value = serde_json::from_slice(&std::fs::read(identity).map_err(e)?).map_err(e)?;
        let head = doc["source_head"].as_str().unwrap_or("");
        let pass = head.starts_with(&args.source_head_prefix);
        ok &= pass;
        bindings.push(json!({
            "binding": "historical-binary-identity", "binary_executed": false,
            "path": identity.display().to_string(),
            "file_sha256": file_sha256(identity)?, "source_head": head,
            "source_tree": doc["source_tree"], "build_command": doc["build_command"],
            "binaries": doc["files"], "expected_source_head_prefix": args.source_head_prefix,
            "pass": pass,
        }));
    }

    // ── real artifact, admitted and sized as the product loader does ──
    let mut hfq = HfqFile::open(&args.model).map_err(|x| format!("open HFQM: {x}"))?;
    resolve_and_attach_ple_rows(
        &mut hfq,
        &args.model,
        hipfire_config::memory::ple_format(),
        &args.arch,
        ResolvePolicy {
            derive_missing_q4: false,
            progress: None,
        },
    )
    .map_err(|x| format!("qwen4 PLE rows: {x}"))?;
    let receipt = admit_hfqm_artifact_on(&hfq, Some(&args.arch))
        .map_err(|x| format!("qwen4 admission: {x}"))?;
    let recipe = receipt.manifest.identity.recipe.clone();
    let layers = receipt.config.num_hidden_layers;
    let source = HfqModelSource::from_hfq(hfq);
    // Entry sizes as the product loader sizes them: a slab layer arena from the
    // recipe's plan, anything else from the mapped tensor.
    let bytes_of = |entry: &WeightEntry| -> Option<u64> {
        if let hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { arenas, .. } = &recipe
        {
            if entry
                .name
                .ends_with(hipfire_arch_qwen4::expert_arena::ARENA_SUFFIX)
            {
                return entry
                    .layer
                    .and_then(|layer| arenas.layer(layer))
                    .map(|layer| layer.bytes as u64);
            }
        }
        source
            .tensor_data(&entry.name)
            .map(|(_, bytes)| bytes.len() as u64)
    };
    let weights = receipt.manifest.weights.clone();
    let (non_expert, layer_experts) = residency::resident_split(&weights, &bytes_of).map_err(e)?;
    let routed = residency::routed_expert_bytes(&weights, &bytes_of).map_err(e)?;
    let fixture = Fixture {
        config: &receipt.config,
        weights: &weights,
        arch: &args.arch,
        max_seq: args.max_seq,
    };

    // ── formats x cases ──────────────────────────────────────────────
    let mut format_docs = Vec::new();
    let mut all_equal = true;
    let live_plane = GdnStateFormat::Q8Ef.residual_bytes(
        fixture.config.linear_num_value_heads,
        fixture.config.linear_key_head_dim,
        fixture.config.linear_value_head_dim,
    ) as u64;
    let live_total = live_plane * fixture.config.n_linear_layers() as u64;
    for (name, gdn) in &args.formats {
        let mut case_docs = Vec::new();
        for case in &args.cases {
            let (reserve, state) = fixture_reserve(&fixture, *gdn, case.chunk_rows)?;
            let base = reserve.bytes();
            let legacy = legacy_inline(
                case.free,
                non_expert,
                routed,
                layer_experts,
                layers,
                base,
                case.stage,
            );
            let shared = residency::incumbent_placement_layers(
                case.free,
                non_expert,
                routed,
                layer_experts,
                layers,
                base,
                case.stage,
            );
            let equal = legacy == shared;
            let legacy_placed = placed(&weights, &bytes_of, layers, &legacy)?;
            let shared_placed = placed(&weights, &bytes_of, layers, &shared)?;
            let placed_equal = legacy_placed == shared_placed;
            let identical = equal && placed_equal;
            all_equal &= identical;
            println!(
                "gdn {name} case {}: free {} MiB chunk {} base reserve {} MiB ({}) stage {} -> \
                 legacy {:?} shared {:?} {}",
                case.label,
                case.free / MIB,
                case.chunk_rows,
                base / MIB,
                reserve.winner(),
                case.stage,
                legacy,
                shared,
                if identical { "IDENTICAL" } else { "DIFFER" }
            );
            case_docs.push(json!({
                "label": case.label, "free_vram_bytes": case.free,
                "chunk_rows": case.chunk_rows, "dedicated_stage": case.stage,
                "base_reserve_bytes": base, "reserve_winner": reserve.winner(),
                "reserve_historical_bytes": reserve.historical,
                "reserve_census_total_bytes": reserve.census_total,
                "reserve_census": reserve.census.describe(),
                "state_census": {
                    "qsa_context": state.qsa_context, "state": state.state, "forward": state.forward,
                },
                "legacy": describe(&legacy), "shared": describe(&shared),
                "placed_legacy": legacy_placed, "placed_shared": shared_placed,
                "identical": identical,
            }));
        }
        format_docs.push(json!({"gdn_state": name, "cases": case_docs}));
    }
    ok &= all_equal;

    // ── live-EF accounting: one residual plane per GDN layer over q8 ──
    let accounting = if args.formats.iter().any(|(_, f)| *f == GdnStateFormat::Q8)
        && args.formats.iter().any(|(_, f)| *f == GdnStateFormat::Q8Ef)
    {
        let mut rows = Vec::new();
        let mut pass = true;
        for case in &args.cases {
            let (q8, q8_state) = fixture_reserve(&fixture, GdnStateFormat::Q8, case.chunk_rows)?;
            let (ef, ef_state) = fixture_reserve(&fixture, GdnStateFormat::Q8Ef, case.chunk_rows)?;
            // The one workspace term the formats legitimately disagree on: plain Q8 charges
            // the dense scan's thread-local zero-EF plane, Q8Ef scans against its live
            // residual (charged once in `state`) and charges none
            // (`fn_gdn_dense::workspace_ef_bytes_for`, asked of the allocator itself).
            let q8_ef_scratch =
                if dense_gdn_applies(fixture.arch, GdnStateFormat::Q8, fixture.config) {
                    rdna_compute::fn_gdn_dense::workspace_ef_bytes_for(
                        GdnStateFormat::Q8,
                        case.chunk_rows,
                    ) as i128
                } else {
                    0
                };
            let mut terms = Vec::new();
            let mut row_pass = ef.census.workspace.len() == q8.census.workspace.len();
            let mut total_delta = 0i128;
            for ((name, q8_bytes), (ef_name, ef_bytes)) in q8.census.terms().zip(ef.census.terms())
            {
                let delta = ef_bytes as i128 - q8_bytes as i128;
                // Every term is equal except the state (+ one live plane per GDN layer, once)
                // and the dense scan's EF scratch (- the plain-Q8 plane).
                let expected = match name {
                    "state" => live_total as i128,
                    "gdn_dense_ef" => -q8_ef_scratch,
                    _ => 0,
                };
                let term_pass = name == ef_name
                    && delta == expected
                    && (name != "gdn_dense_ef" || ef_bytes == 0);
                row_pass &= term_pass;
                total_delta += delta;
                if delta != 0 || expected != 0 {
                    terms.push(json!({
                        "term": name, "q8_bytes": q8_bytes, "q8_ef_bytes": ef_bytes,
                        "delta_bytes": delta as i64, "expected_delta_bytes": expected as i64,
                        "pass": term_pass,
                    }));
                }
            }
            let census_delta = ef.census_total as i128 - q8.census_total as i128;
            row_pass &= census_delta == total_delta
                && census_delta == live_total as i128 - q8_ef_scratch
                && ef_state.state as i128 - q8_state.state as i128 == live_total as i128
                && ef_state.qsa_context == q8_state.qsa_context
                && ef_state.forward == q8_state.forward;
            pass &= row_pass;
            rows.push(json!({
                "case": case.label, "chunk_rows": case.chunk_rows,
                "q8_state_bytes": q8_state.state, "q8_ef_state_bytes": ef_state.state,
                "state_delta_bytes": ef_state.state as i64 - q8_state.state as i64,
                "q8_dense_scan_ef_scratch_bytes": q8_ef_scratch as i64,
                "census_total_delta_bytes": census_delta as i64,
                "expected_census_total_delta_bytes": (live_total as i128 - q8_ef_scratch) as i64,
                "differing_terms": terms,
                "pass": row_pass,
            }));
        }
        ok &= pass;
        json!({
            "live_residual_plane_bytes": live_plane,
            "gdn_layers": fixture.config.n_linear_layers(),
            "live_residual_total_bytes": live_total,
            "note": "per term, q8_ef minus q8: `state` is exactly +one live F16 residual plane per \
                     GDN layer (Qwen4State::device_bytes adds it once, not mirrored in the \
                     snapshot arena, not per ring row); the dense GDN scan's `gdn_dense_ef` \
                     workspace term is exactly -the plain-Q8 zero-EF plane \
                     (fn_gdn_dense::workspace_ef_bytes_for: Q8Ef charges none because its \
                     residual is the live state); every other census term is equal",
            "rows": rows, "pass": pass,
        })
    } else {
        Value::Null
    };

    let env = |name: &str| std::env::var(name).ok();
    let doc = json!({
        "tool": "qwen4_placement_receipt",
        "model": {
            "path": args.model.display().to_string(), "size_bytes": model_size,
            "header_sha256": header_sha, "data_offset": data_offset,
            "arch_admitted_as": args.arch, "trunk_layers": layers,
            "non_expert_bytes": non_expert, "layer_expert_bytes": layer_experts,
            "routed_expert_bytes": routed,
        },
        "fixture": {
            "max_seq": args.max_seq, "qsa_kv": "q8", "kv_backend": "legacy",
            "native_mtp": false, "ranks": 1, "scratch_mode": "full",
            "accounting": "current: state_census (live-EF once), auto_vram_reserve, \
                           qwen4_prefill_workspace_terms with qwen4_dense_gdn_admits, \
                           runtime_overhead_bytes, f16_shadow_bytes",
        },
        "settings": {
            "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": env("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"),
            "HIPFIRE_QWEN4_EXPERT_STAGE": env("HIPFIRE_QWEN4_EXPERT_STAGE"),
            "HIPFIRE_PREFILL_CHUNK_ROWS": env("HIPFIRE_PREFILL_CHUNK_ROWS"),
            "HIP_VISIBLE_DEVICES": env("HIP_VISIBLE_DEVICES"),
        },
        "bindings": bindings,
        "formats": format_docs,
        "live_ef_accounting": accounting,
        "pass": ok,
    });
    let text = serde_json::to_string_pretty(&doc).map_err(e)? + "\n";
    if let Some(path) = &args.json {
        std::fs::write(path, &text).map_err(|x| format!("write {}: {x}", path.display()))?;
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(ok)
}

fn main() {
    match real_main() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(message) => {
            eprintln!("qwen4_placement_receipt: {message}");
            std::process::exit(2);
        }
    }
}
