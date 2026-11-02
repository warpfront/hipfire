// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `qwen4_e8p_gp`: R8 / K-E1 real-layer qualification and Gp timing of the E8P-on-IU4 PM kernels (NT1, gfx1201).
//!
//! Raw PM module launches only (`module_load_data` + `launch_kernel_blob`; the 64-byte W3 ABI, `n_experts@60` = the pointer table's actual extent); never runtime dispatch,
//! never a production module/bundle/manifest edit, never a new loader admission. Two modes over ONE resident layer
//! of REAL solved E8P records (`vq-e8p` arm, BF16-source solved, all 512 experts; nothing is re-encoded here):
//!
//! `--mode g1`      real-layer G1 (`g1.e8p.real`): all 512 experts resident, the shipped gate/up + down kernels and their
//!                  `_acc` twins compared bit for bit with the K-E0 CPU twin (`unpack_word_ref` nibbles, MAGIC seed,
//!                  `fold128`, `epoch_sum` for F32; the frozen BF16 epilogue DAG with the device `v_exp_f32` /
//!                  `v_ldexp_f32` / IEEE-divide probes for BF16). Scenario set: one tile per expert for ALL 512 experts
//!                  (bounded + stress activations), the real-G1 per-expert parity set on hashed experts, and (with
//!                  `--trace-dir`) the head of the recorded real route. Receipt `g1.e8p.real.json` carries
//!                  per-scenario / per-output / per-slot mismatch counts and the real scope.
//! `--mode timing`  Gp NT1 tile-epoch timing of E8P (`--variant rev1[,alt]`) against XTS MQ4 and W3 on the SAME recorded
//!                  route, SAME selected expert ids, SAME padded tiles (R6 method), warmup 5 / 30 HIP-event reps, arms
//!                  rotated per rep, one card/session. Gp receipt `gp.e8p.timing.receipt.json`: raw samples, per-kind
//!                  (gate_up, down) and aggregate (pass) ratios with bootstrap CIs, tile epochs, the 0.85 verdict
//!                  (`xts_median_s / e8p_median_s` on the aggregate tiling, pass label), identity/inputs/code/commands.
//!                  Timing refuses to start unless each variant's `g1.e8p.real` receipt is a pass for the same code object.
//!
//! Everything is pinned and hashed on the CPU before any device is touched: the manifest, both record files (whole-file
//! SHA-256 against the CLI pin and the manifest), every record and repacked blob (`e8p_records.json`), code objects
//! and their bundles, the G1 guards. `--plan-only` runs every CPU step (record read/repack validation, K-E0 table
//! self-check over all 65 536 wire words, scenario/tile construction, guards) and exits before the GPU.
//! CPU repack: record `idx | signs << 8` u16 words (K-E0 `pack_word`) planar, then FP16 scales; gate_up 870 400 B and down
//! 435 200 B per expert. Refuses `HSA_CU_MASK` / `ROC_GLOBAL_CU_MASK`. Device runs only under `fabric-run.sh`.
//!
//! Variants: `rev1` = `--e8p-codeobject`; `alt` = `--e8p-alt-codeobject` (the single bounded LDS alternative; same
//! symbols, own sha pin, own G1 receipt). No fake entries: a variant without its code object is an error.
//!
//! Exit codes: 0 ok (g1: pass), 3 g1 ran but not exact/complete (see receipt), 1 error.
//!
//! Build (parent): `cargo build -p hipfire-arch-qwen4 --release --features lab --example qwen4_e8p_gp`
//! G1 run (class correct):
//!   fabric-run.sh --class correct --card A --mem-gb 6 --priority P1 --owner Subq4WeekLead3.R8E8pGp --max-secs N -- \
//!     target/release/examples/qwen4_e8p_gp --mode g1 --variant rev1 --layer 0 \
//!       --e8p-codeobject E8P.co [--e8p-sha256 HEX] [--e8p-bundle B.hxaco] \
//!       --probe-codeobject $G/qwen4_moe_sym_g1_probe_gfx1201.co --probe-sha256 4b810b2f11d82937afe1ac12d3e96c36a53c082c7041349230681e61a8d3d92c \
//!       --e8p-manifest /home/kaden/qcal/release-0.4.2/subq4/vq/rcpt/vq-e8p.manifest.json \
//!       --e8p-manifest-sha256 a664a9e8509aa1e98a4d25f3057b95f0a71b54c3c3b9cb505bc98d7f558b8f3b \
//!       --e8p-gate-up $W/artifacts/r8-input/L0.gate_up_proj.bin --e8p-gate-up-sha256 b62e40c7119b8aee26c6204937bcd80666f3705aab58fe1702ca336fcef14dfc \
//!       --e8p-down $W/artifacts/r8-input/L0.down_proj.bin --e8p-down-sha256 7ccdc29ae35fb2775ac292c49ebcc0e7e5ac9db40c344d5dd0aa70ecfb643bac \
//!       [--trace-dir $W/artifacts/r6-route/run1 --route-tokens 64] --card A --fabric-owner NAME \
//!       --fabric-wrapper /home/kaden/qcal/locks/fabric/fabric-run.sh --out-dir OUT [--expect-bdf BDF]
//! Gp run (class timing): the same record/e8p flags plus
//!       --mode timing --variant rev1 --e8p-g1-receipt OUT/g1.e8p.real.json
//!       --w3-codeobject $G/qwen4_moe_iu4_sym_w3_pm_gfx1201.co --w3-sha256 60144976aa1c5b83e1000b2568909182126f21995a2830c4575d119e9d327f86
//!       --mq4-codeobject $G/qwen4_moe_iu4_sym_pm_gfx1201.co --mq4-sha256 ff22e97eb31183cb8220d4c856f33e621b5cc13fed484419cd1cca91b31f33a8
//!       --g1-receipt $W/rcpt/g1.k2.real.json --artifact $W/artifacts/ctl3-hot/ctl3-hot.hfq --artifact-sha256 e8fb196deed493a1c1d6a825f44e934465135bb85d6325c38c2e022de33b9674 --artifact-bytes 113653405208
//!       --source /home/kaden/qcal/models/qwen3.8-flash-next-gptq3.mq4 --source-sha256 8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972
//!       --trace-dir $W/artifacts/r6-route/run1 --scope w3_pool --tiling both --reps 30 --warmup 5
//!     (alternative after a revision-1 FAIL: `--variant rev1,alt --e8p-alt-codeobject ALT.co --e8p-alt-g1-receipt ALT_G1.json` re-times all arms in one session)

#![allow(dead_code)]

#[path = "e8p_gp/common.rs"]
mod common;
#[path = "e8p_gp/g1.rs"]
mod g1;
#[path = "e8p_gp/gp.rs"]
mod gp;
#[path = "e8p_gp/hw.rs"]
mod hw;
#[path = "e8p_gp/records.rs"]
mod records;
#[path = "e8p_gp/route.rs"]
mod route;
#[path = "e8p_gp/twin.rs"]
mod twin;

use std::path::PathBuf;

use common::*;
use route::Tiling;
use serde_json::{json, Value};

const USAGE: &str = "usage: qwen4_e8p_gp --mode g1|timing [--variant rev1[,alt]] [--layer 0] \
--e8p-codeobject P [--e8p-sha256 HEX] [--e8p-bundle P] [--e8p-alt-codeobject P [--e8p-alt-sha256 HEX] [--e8p-alt-bundle P]] \
[--e8p-gate-up-symbol S] [--e8p-down-symbol S] \
--e8p-manifest P --e8p-manifest-sha256 HEX --e8p-gate-up P --e8p-gate-up-sha256 HEX --e8p-down P --e8p-down-sha256 HEX \
--out-dir DIR [--card A|B|C|E --fabric-owner NAME --fabric-wrapper PATH [--expect-bdf BDF]] [--plan-only] [--overwrite] [--seed N] \
g1: --probe-codeobject P [--probe-sha256 HEX] [--probe-source P] [--probe-kernarg-bytes N] [--g1-experts 6] [--trace-dir DIR [--route-bin P] --route-tokens 64] [--negative-control 1|2] \
timing: --e8p-g1-receipt P [--e8p-alt-g1-receipt P] --w3-codeobject P --w3-sha256 HEX --mq4-codeobject P --mq4-sha256 HEX [--w3-bundle P] [--mq4-bundle P] \
--g1-receipt g1.k2.real.json --artifact HFQ --artifact-sha256 HEX [--artifact-bytes N] --source XTS.mq4 --source-sha256 HEX [--source-bytes N] \
--trace-dir DIR [--route-bin P] [--scope w3_pool|g1_w3|ids|full_layer] [--expert-ids a,b,..] [--per-codec 3] [--tiling aggregate|chunked|both] [--reps 30] [--warmup 5]";

const SYM_GATE_UP: &str = "qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1201";
const SYM_DOWN: &str = "qwen4_moe_down_iu4_sym_e8p_pm_gfx1201";

const HARNESS_SOURCES: [(&str, &[u8]); 8] = [
    (
        "examples/qwen4_e8p_gp.rs",
        include_bytes!("qwen4_e8p_gp.rs"),
    ),
    (
        "examples/e8p_gp/common.rs",
        include_bytes!("e8p_gp/common.rs"),
    ),
    (
        "examples/e8p_gp/records.rs",
        include_bytes!("e8p_gp/records.rs"),
    ),
    ("examples/e8p_gp/twin.rs", include_bytes!("e8p_gp/twin.rs")),
    ("examples/e8p_gp/hw.rs", include_bytes!("e8p_gp/hw.rs")),
    (
        "examples/e8p_gp/route.rs",
        include_bytes!("e8p_gp/route.rs"),
    ),
    ("examples/e8p_gp/g1.rs", include_bytes!("e8p_gp/g1.rs")),
    ("examples/e8p_gp/gp.rs", include_bytes!("e8p_gp/gp.rs")),
];

fn harness_block() -> Value {
    let files: Vec<Value> = HARNESS_SOURCES
        .iter()
        .map(|(n, b)| json!({"source": n, "sha256": sha(b)}))
        .collect();
    let mut all = Vec::new();
    for (n, b) in HARNESS_SOURCES {
        all.extend_from_slice(n.as_bytes());
        all.push(0);
        all.extend_from_slice(&sha(b).into_bytes());
        all.push(b'\n');
    }
    json!({"files": files, "combined_sha256": sha(&all), "embedded_at_build": true})
}

pub struct VariantCfg {
    pub co: PathBuf,
    pub sha: Option<String>,
    pub bundle: Option<PathBuf>,
    pub g1_receipt: Option<PathBuf>,
}

#[derive(Default)]
pub struct Args {
    mode: String,
    variants: Vec<String>,
    layer: usize,
    e8p_co: Option<PathBuf>,
    e8p_sha: Option<String>,
    e8p_bundle: Option<PathBuf>,
    e8p_g1_receipt: Option<PathBuf>,
    e8p_alt_co: Option<PathBuf>,
    e8p_alt_sha: Option<String>,
    e8p_alt_bundle: Option<PathBuf>,
    e8p_alt_g1_receipt: Option<PathBuf>,
    e8p_gate_up_symbol: Option<String>,
    e8p_down_symbol: Option<String>,
    e8p_manifest: Option<PathBuf>,
    e8p_manifest_sha: Option<String>,
    e8p_gate_up: Option<PathBuf>,
    e8p_gate_up_sha: Option<String>,
    e8p_down: Option<PathBuf>,
    e8p_down_sha: Option<String>,
    probe_co: Option<PathBuf>,
    probe_sha: Option<String>,
    probe_source: Option<PathBuf>,
    probe_ka: Option<usize>,
    g1_experts: usize,
    route_tokens: usize,
    neg: u8,
    w3_co: Option<PathBuf>,
    mq4_co: Option<PathBuf>,
    w3_sha: Option<String>,
    mq4_sha: Option<String>,
    w3_bundle: Option<PathBuf>,
    mq4_bundle: Option<PathBuf>,
    g1_receipt: Option<PathBuf>,
    artifact: Option<PathBuf>,
    artifact_sha: Option<String>,
    artifact_bytes: Option<u64>,
    source: Option<PathBuf>,
    source_sha: Option<String>,
    source_bytes: Option<u64>,
    trace_dir: Option<PathBuf>,
    route_bin: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    card: Option<String>,
    scope: String,
    expert_ids: Vec<usize>,
    tilings: Vec<Tiling>,
    reps: usize,
    warmup: usize,
    fabric_wrapper: Option<String>,
    fabric_owner: Option<String>,
    expect_bdf: Option<String>,
    seed: u64,
    per_codec: usize,
    plan_only: bool,
    overwrite: bool,
}

impl Args {
    fn single_variant(&self) -> Result<String> {
        match self.variants.as_slice() {
            [v] => Ok(v.clone()),
            _ => Err(format!(
                "--mode g1 qualifies exactly one --variant, got {:?}",
                self.variants
            )),
        }
    }
    fn variant_cfg(&self, name: &str) -> Result<VariantCfg> {
        match name {
            "rev1" => Ok(VariantCfg {
                co: self
                    .e8p_co
                    .clone()
                    .ok_or("variant rev1 needs --e8p-codeobject")?,
                sha: self.e8p_sha.clone(),
                bundle: self.e8p_bundle.clone(),
                g1_receipt: self.e8p_g1_receipt.clone(),
            }),
            "alt" => Ok(VariantCfg {
                co: self
                    .e8p_alt_co
                    .clone()
                    .ok_or("variant alt needs --e8p-alt-codeobject")?,
                sha: self.e8p_alt_sha.clone(),
                bundle: self.e8p_alt_bundle.clone(),
                g1_receipt: self.e8p_alt_g1_receipt.clone(),
            }),
            other => Err(format!("--variant {other}: expected rev1|alt")),
        }
    }
    fn e8p_symbols(&self) -> [String; 2] {
        [
            self.e8p_gate_up_symbol
                .clone()
                .unwrap_or_else(|| SYM_GATE_UP.into()),
            self.e8p_down_symbol
                .clone()
                .unwrap_or_else(|| SYM_DOWN.into()),
        ]
    }
}

fn parse_list(v: &str) -> Result<Vec<usize>> {
    v.split(',')
        .map(|s| s.trim().parse::<usize>().map_err(err))
        .collect()
}

fn parse_args(argv: &[String]) -> Result<Args> {
    let mut a = Args {
        variants: vec!["rev1".into()],
        scope: "w3_pool".into(),
        tilings: vec![Tiling::Aggregate, Tiling::Chunked],
        reps: 30,
        warmup: 5,
        seed: 1,
        per_codec: 3,
        g1_experts: 6,
        route_tokens: 64,
        ..Default::default()
    };
    let hex_arg = |flag: &str, v: &str| -> Result<String> {
        if is_sha256_hex(v) {
            Ok(v.to_owned())
        } else {
            Err(format!("{flag} must be 64 lower-case hex digits"))
        }
    };
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--plan-only" => {
                a.plan_only = true;
                continue;
            }
            "--overwrite" => {
                a.overwrite = true;
                continue;
            }
            _ => {}
        }
        let v = it.next().ok_or(USAGE)?.clone();
        match flag.as_str() {
            "--mode" => {
                if !["g1", "timing"].contains(&v.as_str()) {
                    return Err(format!("--mode {v}: expected g1|timing"));
                }
                a.mode = v;
            }
            "--variant" => {
                a.variants = v.split(',').map(|s| s.trim().to_owned()).collect();
                if a.variants.is_empty()
                    || a.variants.iter().any(|x| x != "rev1" && x != "alt")
                    || a.variants
                        .iter()
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        != a.variants.len()
                {
                    return Err(format!(
                        "--variant {v}: expected unique names from rev1,alt"
                    ));
                }
            }
            "--layer" => a.layer = v.parse().map_err(err)?,
            "--e8p-codeobject" => a.e8p_co = Some(v.into()),
            "--e8p-sha256" => a.e8p_sha = Some(hex_arg(flag, &v)?),
            "--e8p-bundle" => a.e8p_bundle = Some(v.into()),
            "--e8p-g1-receipt" => a.e8p_g1_receipt = Some(v.into()),
            "--e8p-alt-codeobject" => a.e8p_alt_co = Some(v.into()),
            "--e8p-alt-sha256" => a.e8p_alt_sha = Some(hex_arg(flag, &v)?),
            "--e8p-alt-bundle" => a.e8p_alt_bundle = Some(v.into()),
            "--e8p-alt-g1-receipt" => a.e8p_alt_g1_receipt = Some(v.into()),
            "--e8p-gate-up-symbol" => a.e8p_gate_up_symbol = Some(v),
            "--e8p-down-symbol" => a.e8p_down_symbol = Some(v),
            "--e8p-manifest" => a.e8p_manifest = Some(v.into()),
            "--e8p-manifest-sha256" => a.e8p_manifest_sha = Some(hex_arg(flag, &v)?),
            "--e8p-gate-up" => a.e8p_gate_up = Some(v.into()),
            "--e8p-gate-up-sha256" => a.e8p_gate_up_sha = Some(hex_arg(flag, &v)?),
            "--e8p-down" => a.e8p_down = Some(v.into()),
            "--e8p-down-sha256" => a.e8p_down_sha = Some(hex_arg(flag, &v)?),
            "--probe-codeobject" => a.probe_co = Some(v.into()),
            "--probe-sha256" => a.probe_sha = Some(hex_arg(flag, &v)?),
            "--probe-source" => a.probe_source = Some(v.into()),
            "--probe-kernarg-bytes" => a.probe_ka = Some(v.parse().map_err(err)?),
            "--g1-experts" => a.g1_experts = v.parse().map_err(err)?,
            "--route-tokens" => a.route_tokens = v.parse().map_err(err)?,
            "--negative-control" => a.neg = v.parse().map_err(err)?,
            "--w3-codeobject" => a.w3_co = Some(v.into()),
            "--mq4-codeobject" => a.mq4_co = Some(v.into()),
            "--w3-sha256" => a.w3_sha = Some(hex_arg(flag, &v)?),
            "--mq4-sha256" => a.mq4_sha = Some(hex_arg(flag, &v)?),
            "--w3-bundle" => a.w3_bundle = Some(v.into()),
            "--mq4-bundle" => a.mq4_bundle = Some(v.into()),
            "--g1-receipt" => a.g1_receipt = Some(v.into()),
            "--artifact" => a.artifact = Some(v.into()),
            "--artifact-sha256" => a.artifact_sha = Some(hex_arg(flag, &v)?),
            "--artifact-bytes" => a.artifact_bytes = Some(v.parse().map_err(err)?),
            "--source" => a.source = Some(v.into()),
            "--source-sha256" => a.source_sha = Some(hex_arg(flag, &v)?),
            "--source-bytes" => a.source_bytes = Some(v.parse().map_err(err)?),
            "--trace-dir" => a.trace_dir = Some(v.into()),
            "--route-bin" => a.route_bin = Some(v.into()),
            "--out-dir" => a.out_dir = Some(v.into()),
            "--card" => a.card = Some(v),
            "--scope" => {
                if !["w3_pool", "g1_w3", "ids", "full_layer"].contains(&v.as_str()) {
                    return Err(format!(
                        "--scope {v}: expected w3_pool|g1_w3|ids|full_layer"
                    ));
                }
                a.scope = v;
            }
            "--expert-ids" => a.expert_ids = parse_list(&v)?,
            "--tiling" => {
                a.tilings = match v.as_str() {
                    "aggregate" => vec![Tiling::Aggregate],
                    "chunked" => vec![Tiling::Chunked],
                    "both" => vec![Tiling::Aggregate, Tiling::Chunked],
                    _ => return Err(format!("--tiling {v}: expected aggregate|chunked|both")),
                }
            }
            "--reps" => a.reps = v.parse().map_err(err)?,
            "--warmup" => a.warmup = v.parse().map_err(err)?,
            "--fabric-wrapper" => a.fabric_wrapper = Some(v),
            "--fabric-owner" => a.fabric_owner = Some(v),
            "--expect-bdf" => a.expect_bdf = Some(v.to_ascii_lowercase()),
            "--seed" => a.seed = v.parse().map_err(err)?,
            "--per-codec" => a.per_codec = v.parse().map_err(err)?,
            _ => return Err(USAGE.into()),
        }
    }
    if a.mode.is_empty() {
        return Err(format!("--mode is required\n{USAGE}"));
    }
    if a.scope == "ids" && a.expert_ids.is_empty() {
        return Err("--scope ids needs --expert-ids".into());
    }
    if a.neg > 2 {
        return Err("--negative-control must be 1 or 2".into());
    }
    Ok(a)
}

fn run() -> Result<i32> {
    let argv: Vec<String> = std::env::args().collect();
    let a = parse_args(&argv[1..])?;
    refuse_cu_masks()?;
    let out_dir = a.out_dir.clone().ok_or("--out-dir is required")?;
    let (manifest, manifest_sha, gate_up, gate_up_sha, down, down_sha) = (
        a.e8p_manifest.clone().ok_or("--e8p-manifest is required")?,
        a.e8p_manifest_sha
            .clone()
            .ok_or("--e8p-manifest-sha256 is required")?,
        a.e8p_gate_up.clone().ok_or("--e8p-gate-up is required")?,
        a.e8p_gate_up_sha
            .clone()
            .ok_or("--e8p-gate-up-sha256 is required")?,
        a.e8p_down.clone().ok_or("--e8p-down is required")?,
        a.e8p_down_sha
            .clone()
            .ok_or("--e8p-down-sha256 is required")?,
    );
    let receipt_name = match (a.mode.as_str(), a.plan_only) {
        ("g1", false) => "g1.e8p.real.json",
        ("g1", true) => "g1.e8p.real.plan.json",
        _ => "gp.e8p.timing.receipt.json",
    };
    if !a.overwrite && out_dir.join(receipt_name).exists() {
        return Err(format!(
            "{} exists; pass --overwrite",
            out_dir.join(receipt_name).display()
        ));
    }
    std::fs::create_dir_all(&out_dir).map_err(err)?;
    let out_dir = std::fs::canonicalize(&out_dir).map_err(err)?;

    // ---- CPU: pin, hash, read, repack, validate the real records; K-E0 table self-check ----------------------------
    let layer = records::load_layer(&records::RecordPins {
        layer: a.layer,
        manifest: &manifest,
        manifest_sha256: &manifest_sha,
        gate_up: &gate_up,
        gate_up_sha256: &gate_up_sha,
        down: &down,
        down_sha256: &down_sha,
    })?;
    let (oracle, mut oracle_check) = twin::Oracle::new()?;
    oracle_check["launch_geometry_known_answer"] = launch_geometry_known_answer()?;
    let rec_doc = json!({"schema": "r8.e8p.records.v1", "layer": a.layer, "manifest_sha256": manifest_sha, "gate_up_sha256": gate_up_sha, "down_sha256": down_sha, "experts": layer.per_expert});
    let rec_path = out_dir.join("e8p_records.json");
    let rec_bytes = write_json(&rec_path, &rec_doc)?;
    let records_file = json!({"path": rec_path.display().to_string(), "bytes": rec_bytes.len(), "sha256": sha(&rec_bytes)});
    let harness = harness_block();

    match a.mode.as_str() {
        "g1" => {
            let ok = g1::run_g1(&g1::Prep {
                args: &a,
                argv: &argv,
                layer: &layer,
                oracle: &oracle,
                oracle_check: &oracle_check,
                harness: &harness,
                out_dir: &out_dir,
                records_file: &records_file,
            })?;
            Ok(if ok { 0 } else { 3 })
        }
        _ => {
            gp::run_timing(&gp::TimingPrep {
                args: &a,
                argv: &argv,
                layer: &layer,
                harness: &harness,
                out_dir: &out_dir,
                records_file: &records_file,
            })?;
            Ok(0)
        }
    }
}

fn main() {
    match run() {
        Ok(0) => {}
        Ok(rc) => {
            eprintln!("qwen4_e8p_gp: not exact/complete (see receipt)");
            std::process::exit(rc);
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
