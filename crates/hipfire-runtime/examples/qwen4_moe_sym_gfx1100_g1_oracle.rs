// SPDX-License-Identifier: Apache-2.0
//! B4-C independent G1 oracle for the gfx1100 (RDNA3, wave32) NT1 grouped IU4 K2 kernels
//! (`qwen4_moe_{gate_up_silu,down}_iu4_sym{,_w2,_w3}_pm_gfx1100[_acc]`). Raw PM module launches only; no runtime
//! dispatch, and the candidate kernel never produces an expected value.
//!
//! Expected values come from: the signed MQ4 fold (`rdna-compute` `fold_reference` algorithm, full -8..7 domain) or
//! `mq_i4_reference::iu4_gemm_f32_parts` (W2/W3), then the gfx1100 epilogue adapter (`qwen4_moe_sym_gfx1100_g1/epilogue.rs`:
//! the public `grouped_prefill_epilogue_with` plus the reviewed device MulUp NaN rule) driven by hooks that answer
//! only from the independent gfx1100 intrinsic probe table (exact operand keys; a missing key fails the case).
//! A path-loaded MQ4 QT44/QT53 anchor module (`--mq4-anchor`) runs the bit-equivalent MQ4 tensors on the same card;
//! anchor equality is required IN ADDITION to CPU equality, never instead of it.
//!
//!
//! E8P (`--codec e8p`, `--mixed-codeobject e8p=PATH`; entries `qwen4_moe_{gate_up_silu,down}_iu4_sym_e8p_pm_gfx1100[_acc]`, quant types
//! qt135 gate_up / qt136 down): the expected F32 `_acc` is the frozen K-E0 arithmetic of `hipfire_runtime::e8p_i4_reference`
//! (`unpack_word_ref` nibbles, `MAGIC + C`, `fold128`, `epoch_sum` from +0 over ascending epochs; see `qwen4_moe_sym_gfx1100_g1/e8p.rs`),
//! the BF16 expectation is the UNCHANGED reviewed MulUp adapter over those sums. There is no MQ4 anchor (|v| up to 11): `--mq4-anchor`
//! is refused, the mixed MQ4 anchor union is recorded as not applicable, and CPU equality is the sole expectation. The six negatives are
//! `wrong-lut`, `wrong-sign-mask`, `reversed-epoch`, `wrong-fold-epoch` (activation sum `s` dropped in each epoch, signed dot and quarter
//! multiplier kept: `2C/4` instead of `(2C+s)/4`), `wrong-scale-row`, `wrong-expert-pointer` (all F32-gated; no pack-selector negative).
//! Synthetic E8P generators: random words, `EveryCode` (every one of the 65536 wire words in each expert), extreme `|v|` words, f16 max /
//! subnormal / zero scales (the receipt counts `guard_trips`; real artifacts must have none), scale-plane sentinels, targeted exact dots.
//! Build (parent builds; no CU masks anywhere):
//!   cargo build -p hipfire-runtime --release --features lab --example qwen4_moe_sym_gfx1100_g1_oracle
//!
//! Usage:
//!   CPU-only harness validation (no device, no table needed; table optional and then only CPU-checked):
//!     qwen4_moe_sym_gfx1100_g1_oracle --self-check [--codec mq4|w2|w3] [--probe-table OBS.json] [--sample-dir DIR]
//!         [--seed N] [--only SUBSTR] --out RECEIPT.json
//!   MulUp CPU recheck (no device): qwen4_moe_sym_gfx1100_g1_oracle --cpu-recheck-fixture OUT.json --probe-table OBS.json
//!         [--divide-table DIV.json] [--intrinsic-table OBS.json]... [--seed N]
//!     runs every GateUp pass of the standalone fixture (same `fixture_values`/seeds/slots as the device run), captures the
//!     actual q/up per cell and compares the adapter with the historical SSE (Up-destination `mulss`) expectation.
//!   Down chain, step 1 (CPU only, on hiptrx; add `--self-check` for the explicit CPU-only hooks instead of the table):
//!     qwen4_moe_sym_gfx1100_g1_oracle --prepare-chain DIR --codec mq4|w2|w3 --sample-dir DIR_S --probe-table OBS.json
//!         [--seed N] [--only SUBSTR] --out PREP.json          (writes DIR/*.h.bf16.bin and DIR/chain-h.json)
//!   Down chain, step 2: python3 crates/peacemaker-isa/tools/qwen4_moe_gfx1100_chain_a4.py --chain-dir DIR
//!   Down chain, step 3: pass `--chain-dir DIR` to the device run or to `--self-check` (a mixed run uses the `mixed-*` entries
//!     prepared for the same --codec w2|w3; sources, probe table, sample manifests, hooks kind, seed and every entry's CPU H are
//!     validated; the preparing binary's sha256 is recorded with an `oracle_binary_match` flag).
//!   Device (run ON hipx through fabric, `--class correct --card xtx`):
//!     qwen4_moe_sym_gfx1100_g1_oracle --codeobject K2.hxaco --codec mq4|w2|w3 --mq4-anchor MQ4.hxaco
//!         --probe-table OBS.json [--divide-table DIV.json] --card xtx --expect-uuid GPU-43390a851e296ee5
//!         --expect-bdf 0000:67:00.0 --out RECEIPT.json
//!         [--sample-dir DIR]                       real expert weights (synthetic activations/routes), real scope
//!         [--negative-codeobject NAME=PATH]...     NAME in wrong-scale-row|reversed-epoch|wrong-pack-selector|wrong-expert-pointer
//!                                                  (e8p: wrong-lut|wrong-sign-mask|reversed-epoch|wrong-fold-epoch|wrong-scale-row|wrong-expert-pointer)
//!         --epilogue-codeobject FIXTURE.hxaco      required for a complete device PASS (+ its wrongsel negative)
//!         [--seed N] [--only SUBSTR]               `--only` marks the receipt `complete: false`
//!   Exit status: 0 pass, 2 gate failure (receipt written), 1 harness/setup error.
//!   Hook-key census (CPU only; one process, every planned candidate-independent CPU case; NO `--only`; see census.rs):
//!     qwen4_moe_sym_gfx1100_g1_oracle --collect-hook-keys OUT.json --probe-table BASE.json [--divide-table DIV.json]
//!         [--intrinsic-table OBS.json]... --census-sample-dir w2=C24_SAMPLES --census-sample-dir w3=DEQUANT_SAMPLES
//!         [--census-chain-dir mq4=DIR --census-chain-dir w2=DIR --census-chain-dir w3=DIR] [--seed N]
//!     writes OUT.json (`complete`, `missing_count`, `first_missing`, `keys` = FULL required set, `missing` = frontier, both
//!     `consumer24.gfx1100.intrinsic-keys.v1`, `scopes`, `chain_sources`, `chain_stage`), OUT.json.required-keys.json and
//!     OUT.json.missing-keys.json (pure key documents for the observer `--keys`); exit 0 whenever written, 1 on error.
//!   `--intrinsic-table PATH` (repeatable; every device run, `--prepare-chain` and the census): generic exact-key
//!     `consumer24.gfx1100.intrinsic-observations.v1`, merged conflict-checked after the base table and `--divide-table`;
//!     each must carry device identity (gfx1100/xtx, identity_match), `selfcheck.ok` and `probe.denorm` modes 3/3. Their
//!     sha256s (sorted) and the divide table's are pinned in the chain root and the receipts.
//!
//! Residency arms per scenario: VRAM, device-mapped pinned host alias, and a VMM mapping whose last expert tensor
//! ends exactly at the mapped end (the next granule is reserved but unmapped, so any read past the final byte faults).
//! Scope: synthetic runs say `synthetic`. Real-sample runs (`--sample-dir`) say exactly `real weights; synthetic A4/routes;
//! Down input twin-chained`: GateUp runs real expert weights over synthetic A4 activations/routes, and every Down input is
//! the chain synthetic A4 -> CPU twin GateUp (same real expert weights, qualified hooks) -> external K3 producer
//! (`--prepare-chain DIR` writes the CPU H; the Python helper `qwen4_moe_gfx1100_chain_a4.py --chain-dir DIR` rotates and
//! quantizes it; `--chain-dir DIR` loads and validates the result; this oracle contains no K3 code) -> A4 rows indexed by
//! grouped row (Down slots are `p` / -1, x_row_div 1, tile order kept; a missing or rejected chain is a reported failure);
//! the candidate never produces any chain input. The receipt's `g1c` object passes (`measured_pass`, verdict
//! `G1c PASS, artifact binding pending` or `... artifact bound`) ONLY for a mixed-codec run (`--mixed-codeobject
//! mq4=PATH --mixed-codeobject w2=PATH`, or `w3=PATH` as the separate W3 reference mode) on real samples with zero
//! mismatches; `artifact_binding` is `pending|bound` metadata (provisional hash-checked samples count as real weights)
//! and `closed` additionally needs `bound`. Per-codec `--sample-dir` runs list `mixed_codec_run` as a blocker.
//! Mixed mode: one global expert-ID space, one tile list/slot array and one initially poisoned output; each codec's
//! kernel runs with its tile ids masked to -1 for the other codec's experts and null pointer-table entries for
//! experts it does not own; the MQ4 anchor (`--mq4-anchor`, else the mq4 mixed module) runs the QT44/QT53-equivalent
//! union unmasked and must equal the combined output and the CPU twin.
//!
//! # gfx1151 (Strix Halo; `--expect-arch gfx1151`)
//! `--expect-arch gfx1100` (the default) keeps every behaviour above. `--expect-arch gfx1151` retargets the whole oracle at the gfx1151 images: every entry is
//! `qwen4_moe_{gate_up_silu,down}_iu4_sym{,_w2,_w3,_e8p}_pm_gfx1151[_acc]` (E8P: `kernels/qwen4_moe_iu4_sym_e8p_pm_gfx1151.hxaco`, the gfx1100 arm with only
//! arch-forced deltas, so the F32 `_acc` sums must equal the K-E0 twin bit for bit exactly as on gfx1100), the epilogue-fixture symbols end in `_gfx1151`, the
//! device must be gfx1151, `--card halo`, `--expect-bdf` is REQUIRED and `--expect-uuid` OPTIONAL (Halo has no ROCr UUID: checked case-insensitively only when
//! given, then also required of every probe / divide / intrinsic table; receipt `device.uuid_authoritative: false`), and the receipt schema is
//! `consumer24.gfx1151.g1-oracle.v1`. Qualification inputs must be gfx1151 observations (`--probe-table`, `--divide-table`, `--intrinsic-table`:
//! `device.arch == gfx1151`, `identity_match`, the bdf, passing selfcheck, denorm modes 3/3; their schemas keep the historical `consumer24.gfx1100.*` names)
//! and `--epilogue-codeobject` the gfx1151 fixture. The `Finish::MulUp` NaN rule of the epilogue adapter is the gfx1100 device-pinned one; on gfx1151 the
//! device run is what confirms it. `--cpu-recheck-fixture` and `--collect-hook-keys` stay gfx1100 CPU modes and refuse `--expect-arch`.
//! Real layer (E8P) on Halo: `--sample-dir` is the extractor output over the slab artifact (this loader records `artifact_sha256`; it does not pin it):
//!   python3 tools/e8p_k1_extract_samples.py --artifact /home/kaden/pm-wave/consumer24/models/c24h6306-d2/qwen3.8-flash-next.mq2-xts \
//!     --expect-sha256 d3d536c040f834699b5a261a5122b67d076461a9493eca16669803757928a949 --output-dir /abs/halo-e8p-k1-samples
//!   /home/kaden/qcal/locks/fabric/fabric-run.sh --class correct --card halo --mem-gb N --owner NAME -- \
//!     target/release/examples/qwen4_moe_sym_gfx1100_g1_oracle --codeobject /abs/qwen4_moe_iu4_sym_e8p_pm_gfx1151.hxaco --codec e8p \
//!     --sample-dir /abs/halo-e8p-k1-samples --chain-dir DIR --probe-table OBS.json [--divide-table DIV.json] [--intrinsic-table OBS.json]... \
//!     [--epilogue-codeobject FIXTURE.hxaco] [--negative-codeobject NAME=PATH]... --card halo --expect-arch gfx1151 --expect-bdf 0000:XX:00.0 \
//!     [--expect-uuid U] --out RECEIPT.json         (without `--sample-dir`: the synthetic scenario set)
//!   The gfx1151 probe tables and fixture need gfx1151 builds of the PM probe emitters (`peacemaker-isa` examples `gfx1100_*_probe`, `*_fixture_gfx1100_emit`)
//!   and observers (`gfx1100_*_observe`); none exists. This table-qualified path is therefore NOT gate eligible for gfx1151: it is arch plumbing only, no
//!   gfx1151 table qualification is added here. The Halo K2 gate is the self-contained `qwen4_moe_sym_e8p_g1_oracle --expect-arch gfx1151 --samples-dir ...`
//!   (device probes built from `crates/hipfire-arch-qwen4/examples/oracle_kernels/qwen4_moe_sym_g1_probe.gfx1151.hip`, no tables).
#![allow(dead_code)]

use hip_bridge::{DeviceBuffer, Function, HipRuntime, Module};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

#[path = "qwen4_moe_sym_gfx1100_g1/census.rs"]
mod census;
#[path = "qwen4_moe_sym_gfx1100_g1/chain.rs"]
mod chain;
#[path = "qwen4_moe_sym_gfx1100_g1/device.rs"]
mod device;
#[path = "qwen4_moe_sym_gfx1100_g1/e8p.rs"]
mod e8p;
#[path = "qwen4_moe_sym_gfx1100_g1/epilogue.rs"]
mod epilogue;
#[path = "qwen4_moe_sym_gfx1100_g1/model.rs"]
mod model;
#[path = "qwen4_moe_sym_gfx1100_g1/probe_table.rs"]
mod probe_table;
#[path = "qwen4_moe_sym_gfx1100_g1/samples.rs"]
mod samples;
#[path = "qwen4_moe_sym_gfx1100_g1/scenarios.rs"]
mod scenarios;

use chain::{chain_name, ChainDir, PrepEntry, RootExpect, HOOKS_CPU_ONLY, HOOKS_QUALIFIED};
use device::{kernel_denorm, upload, with_ptr_table, Arm, DeviceCase, Resident};
use hipfire_runtime::mq_i4_reference::EpilogueIntrinsics;
use model::*;
use probe_table::{CpuOnlySelfCheckHooks, QualifiedHooks};
use scenarios::{Class, RealRef, Scenario};

const SCOPE_REAL: &str = "real weights; synthetic A4/routes; Down input twin-chained";
const SCHEMA: &str = "consumer24.gfx1100.g1-oracle.v1";
const SCHEMA_GFX1151: &str = "consumer24.gfx1151.g1-oracle.v1";
const XTX_UUID: &str = "GPU-43390a851e296ee5";
const XTX_BDF: &str = "0000:67:00.0";
/// Every negative-control name any codec accepts; `Codec::negative_names` is the per-codec required subset.
const NEG_NAMES: [&str; 7] = [
    "wrong-scale-row",
    "reversed-epoch",
    "wrong-pack-selector",
    "wrong-expert-pointer",
    "wrong-lut",
    "wrong-sign-mask",
    "wrong-fold-epoch",
];

/// `--expect-arch`: gfx1100 (the default; the xtx) or gfx1151 (Strix Halo, fabric label `halo`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Target {
    Gfx1100,
    Gfx1151,
}
impl Target {
    fn of_name(arch: &str) -> Option<Self> {
        match arch {
            "gfx1100" => Some(Self::Gfx1100),
            "gfx1151" => Some(Self::Gfx1151),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Gfx1100 => "gfx1100",
            Self::Gfx1151 => "gfx1151",
        }
    }
    fn schema(self) -> &'static str {
        match self {
            Self::Gfx1100 => SCHEMA,
            Self::Gfx1151 => SCHEMA_GFX1151,
        }
    }
    fn card(self) -> &'static str {
        match self {
            Self::Gfx1100 => "xtx",
            Self::Gfx1151 => "halo",
        }
    }
    fn link_caveat(self) -> &'static str {
        match self {
            Self::Gfx1100 => "xtx sits behind a PCIe 4.0 x4 root port (00:03.1); this is a correctness oracle and carries no speed or bandwidth claim",
            Self::Gfx1151 => "gfx1151 is the Strix Halo APU (UMA: VRAM and host memory are one pool); this is a correctness oracle and carries no speed or bandwidth claim",
        }
    }
}

/// The device identity a run, its probe tables and its fixtures must carry: gfx1100 pins the xtx uuid and bdf; gfx1151 has no ROCr UUID, so its uuid is
/// optional (checked when given) and a device run requires `--expect-bdf`.
struct Ident {
    target: Target,
    uuid: Option<String>,
    bdf: Option<String>,
}
impl Ident {
    fn arch(&self) -> &'static str {
        self.target.name()
    }
    /// Whether a recorded `device` object (probe / observation / divide table) is this identity with `identity_match: true`.
    fn matches(&self, d: &Value) -> bool {
        d["arch"] == json!(self.arch())
            && self.uuid.as_ref().is_none_or(|u| d["uuid"] == json!(u))
            && self.bdf.as_ref().is_none_or(|b| d["bdf"] == json!(b))
            && d["identity_match"] == json!(true)
    }
}
impl Args {
    fn target_or_default(&self) -> Target {
        self.expect_arch
            .as_deref()
            .and_then(Target::of_name)
            .unwrap_or(Target::Gfx1100)
    }
    /// Identity expected of the qualification tables (CPU modes included): the xtx for gfx1100, the optional flags for gfx1151.
    fn table_ident(&self) -> Ident {
        match self.target_or_default() {
            Target::Gfx1100 => Ident {
                target: Target::Gfx1100,
                uuid: Some(XTX_UUID.into()),
                bdf: Some(XTX_BDF.into()),
            },
            Target::Gfx1151 => Ident {
                target: Target::Gfx1151,
                uuid: self.expect_uuid.clone(),
                bdf: self.expect_bdf.clone(),
            },
        }
    }
}
fn schema_of(a: &Args) -> &'static str {
    a.target_or_default().schema()
}
const USAGE: &str = "usage: qwen4_moe_sym_gfx1100_g1_oracle (--self-check | (--codeobject PATH --codec mq4|w2|w3|e8p | --mixed-codeobject mq4=PATH --mixed-codeobject w2=PATH|w3=PATH|e8p=PATH) [--mq4-anchor PATH (required for w2/w3 per-codec runs and the w2/w3 anchor union; REFUSED for e8p)] --probe-table PATH --card xtx|halo [--expect-arch gfx1100|gfx1151 (default gfx1100)] --expect-uuid UUID (optional for gfx1151) --expect-bdf BDF) --out PATH [--sample-dir DIR (required for mixed)] [--divide-table PATH] [--intrinsic-table PATH]... [--negative-codeobject NAME=PATH]... [--epilogue-codeobject PATH] [--seed N] [--only SUBSTR] | qwen4_moe_sym_gfx1100_g1_oracle --collect-hook-keys OUT.json --probe-table PATH [--divide-table PATH] [--intrinsic-table PATH]... --census-sample-dir w2=DIR --census-sample-dir w3=DIR | --census-sample-dir e8p=DIR... [--census-chain-dir mq4=DIR --census-chain-dir w2=DIR --census-chain-dir w3=DIR | e8p=DIR] [--seed N] | qwen4_moe_sym_gfx1100_g1_oracle --cpu-recheck-fixture OUT.json --probe-table PATH [--divide-table PATH] [--intrinsic-table PATH]... [--seed N]";

fn source_hashes() -> Value {
    let files: [(&str, &[u8]); 10] = [
        (
            "examples/qwen4_moe_sym_gfx1100_g1_oracle.rs",
            include_bytes!("qwen4_moe_sym_gfx1100_g1_oracle.rs"),
        ),
        (
            "examples/qwen4_moe_sym_gfx1100_g1/device.rs",
            include_bytes!("qwen4_moe_sym_gfx1100_g1/device.rs"),
        ),
        (
            "examples/qwen4_moe_sym_gfx1100_g1/model.rs",
            include_bytes!("qwen4_moe_sym_gfx1100_g1/model.rs"),
        ),
        (
            "examples/qwen4_moe_sym_gfx1100_g1/chain.rs",
            include_bytes!("qwen4_moe_sym_gfx1100_g1/chain.rs"),
        ),
        (
            "examples/qwen4_moe_sym_gfx1100_g1/probe_table.rs",
            include_bytes!("qwen4_moe_sym_gfx1100_g1/probe_table.rs"),
        ),
        (
            "examples/qwen4_moe_sym_gfx1100_g1/samples.rs",
            include_bytes!("qwen4_moe_sym_gfx1100_g1/samples.rs"),
        ),
        (
            "examples/qwen4_moe_sym_gfx1100_g1/scenarios.rs",
            include_bytes!("qwen4_moe_sym_gfx1100_g1/scenarios.rs"),
        ),
        (
            "examples/qwen4_moe_sym_gfx1100_g1/census.rs",
            include_bytes!("qwen4_moe_sym_gfx1100_g1/census.rs"),
        ),
        (
            "examples/qwen4_moe_sym_gfx1100_g1/epilogue.rs",
            include_bytes!("qwen4_moe_sym_gfx1100_g1/epilogue.rs"),
        ),
        (
            "examples/qwen4_moe_sym_gfx1100_g1/e8p.rs",
            include_bytes!("qwen4_moe_sym_gfx1100_g1/e8p.rs"),
        ),
    ];
    let mut m = serde_json::Map::new();
    for (n, b) in files {
        m.insert(n.to_string(), json!(sha256_hex(b)));
    }
    Value::Object(m)
}

#[derive(Default)]
struct Args {
    codeobject: Option<String>,
    codec: Option<String>,
    anchor: Option<String>,
    probe_table: Option<String>,
    divide_table: Option<String>,
    card: Option<String>,
    expect_uuid: Option<String>,
    expect_bdf: Option<String>,
    expect_arch: Option<String>,
    out: Option<String>,
    self_check: bool,
    sample_dir: Option<String>,
    negatives: Vec<(String, String)>,
    mixed: Vec<(String, String)>,
    epilogue_fixture: Option<String>,
    prepare_chain: Option<String>,
    chain_dir: Option<String>,
    seed: u32,
    only: Option<String>,
    collect_hook_keys: Option<String>,
    cpu_recheck_fixture: Option<String>,
    census_chains: Vec<(String, String)>,
    intrinsic_tables: Vec<String>,
    census_samples: Vec<(String, String)>,
    argv: Vec<String>,
}

fn parse_args() -> Res<Args> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        seed: 1,
        argv: argv.clone(),
        ..Default::default()
    };
    let mut it = argv.into_iter();
    while let Some(f) = it.next() {
        if f == "--self-check" {
            a.self_check = true;
            continue;
        }
        if f == "--help" || f == "-h" {
            println!("{USAGE}");
            std::process::exit(0);
        }
        let v = it
            .next()
            .ok_or_else(|| format!("{f} needs a value\n{USAGE}"))?;
        match f.as_str() {
            "--codeobject" => a.codeobject = Some(v),
            "--codec" => a.codec = Some(v),
            "--mq4-anchor" => a.anchor = Some(v),
            "--probe-table" => a.probe_table = Some(v),
            "--divide-table" => a.divide_table = Some(v),
            "--card" => a.card = Some(v),
            "--expect-uuid" => a.expect_uuid = Some(v),
            "--expect-bdf" => a.expect_bdf = Some(v.to_ascii_lowercase()),
            "--expect-arch" => a.expect_arch = Some(v),
            "--out" => a.out = Some(v),
            "--sample-dir" => a.sample_dir = Some(v),
            "--epilogue-codeobject" => a.epilogue_fixture = Some(v),
            "--prepare-chain" => a.prepare_chain = Some(v),
            "--chain-dir" => a.chain_dir = Some(v),
            "--collect-hook-keys" => a.collect_hook_keys = Some(v),
            "--cpu-recheck-fixture" => a.cpu_recheck_fixture = Some(v),
            "--census-chain-dir" => {
                let (c, d) = v
                    .split_once('=')
                    .ok_or("--census-chain-dir needs mq4=DIR, w2=DIR, w3=DIR or e8p=DIR")?;
                if !["mq4", "w2", "w3", "e8p"].contains(&c) {
                    return Err(format!(
                        "--census-chain-dir key must be mq4, w2, w3 or e8p, got {c:?}"
                    ));
                }
                if a.census_chains.iter().any(|(x, y)| x == c && y == d) {
                    return Err(format!("--census-chain-dir {c}={d} given twice"));
                }
                a.census_chains.push((c.to_string(), d.to_string()));
            }
            "--intrinsic-table" => a.intrinsic_tables.push(v),
            "--census-sample-dir" => {
                let (c, d) = v
                    .split_once('=')
                    .ok_or("--census-sample-dir needs w2=DIR, w3=DIR, mq4=DIR or e8p=DIR")?;
                if !["mq4", "w2", "w3", "e8p"].contains(&c) {
                    return Err(format!(
                        "--census-sample-dir key must be mq4, w2, w3 or e8p, got {c:?}"
                    ));
                }
                // e8p may name several sample dirs (one per artifact: c24-e8p, c24h6306-e8p); every other key exactly once
                if a.census_samples
                    .iter()
                    .any(|(x, y)| x == c && (c != "e8p" || y == d))
                {
                    return Err(format!("--census-sample-dir {c} given twice"));
                }
                a.census_samples.push((c.to_string(), d.to_string()));
            }
            "--only" => a.only = Some(v),
            "--seed" => a.seed = v.parse().map_err(err)?,
            "--mixed-codeobject" => {
                let (k, p) = v
                    .split_once('=')
                    .ok_or("--mixed-codeobject needs mq4=PATH or w2=PATH or w3=PATH or e8p=PATH")?;
                if !["mq4", "w2", "w3", "e8p"].contains(&k) {
                    return Err(format!(
                        "--mixed-codeobject key must be mq4, w2, w3 or e8p, got {k:?}"
                    ));
                }
                if a.mixed.iter().any(|(x, _)| x == k) {
                    return Err(format!("--mixed-codeobject {k} given twice"));
                }
                a.mixed.push((k.to_string(), p.to_string()));
            }
            "--negative-codeobject" => {
                let (n, p) = v
                    .split_once('=')
                    .ok_or("--negative-codeobject needs NAME=PATH")?;
                if !NEG_NAMES.contains(&n) {
                    return Err(format!(
                        "unknown negative control {n:?}; expected one of {NEG_NAMES:?}"
                    ));
                }
                if a.negatives.iter().any(|(x, _)| x == n) {
                    return Err(format!("negative control {n} given twice"));
                }
                a.negatives.push((n.to_string(), p.to_string()));
            }
            _ => return Err(format!("unknown flag {f}\n{USAGE}")),
        }
    }
    let target = match a.expect_arch.as_deref() {
        None => Target::Gfx1100,
        Some(n) => Target::of_name(n).ok_or_else(|| {
            format!("--expect-arch {n:?}: this oracle serves gfx1100 and gfx1151\n{USAGE}")
        })?,
    };
    model::set_arch(target.name());
    Ok(a)
}

fn write_json(path: &str, v: &Value) -> Res<()> {
    let bytes = serde_json::to_vec_pretty(v).map_err(err)?;
    std::fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))
}

fn read_file(p: &str) -> Res<Vec<u8>> {
    std::fs::read(p).map_err(|e| format!("{p}: {e}"))
}

fn file_json(p: &str) -> Res<(Vec<u8>, Value)> {
    let b = read_file(p)?;
    let v: Value = serde_json::from_slice(&b).map_err(|e| format!("{p}: {e}"))?;
    Ok((b, v))
}

fn scenario_list(
    a: &Args,
    codec: Codec,
    kind: Kind,
    set: &Option<samples::SampleSet>,
) -> Vec<Scenario> {
    let mut v = match set {
        Some(s) => scenarios::real(
            if kind == Kind::GateUp {
                &s.gate_up
            } else {
                &s.down
            },
            a.seed,
        ),
        None => scenarios::synthetic(a.seed),
    };
    let _ = codec;
    if let Some(f) = &a.only {
        v.retain(|s| s.name.contains(f.as_str()));
    }
    v
}

fn coverage_gaps(sc: &Scenario, kind: Kind, cov: &Value) -> Vec<String> {
    let g = |k: &str| cov[k].as_u64().unwrap_or(0);
    let mut v = Vec::new();
    if sc.name == "targeted_ties" {
        if g("f32_sum_ties_round_down_even") == 0 {
            v.push(format!(
                "{}: no BF16 tie rounding down on even LSB reached",
                sc.name
            ));
        }
        if g("f32_sum_ties_round_up_odd") == 0 {
            v.push(format!(
                "{}: no BF16 tie rounding up on odd LSB reached",
                sc.name
            ));
        }
    }
    if sc.name == "targeted_signed_zero" && kind == Kind::GateUp && g("bf16_out_negative_zero") == 0
    {
        v.push(format!("{}: no negative-zero BF16 output reached", sc.name));
    }
    if sc.class == Class::Exceptional && g("f32_sum_nonfinite") == 0 {
        v.push(format!("{}: no non-finite F32 sum reached", sc.name));
    }
    v
}

fn experts_json(experts: &[Expert]) -> Value {
    Value::Array(experts.iter().map(|e| {
        let mut j = json!({"origin": e.origin, "bytes": e.bytes.len(), "sha256": e.sha256, "asym_headers": e.asym_headers});
        if e.codec == Codec::E8p {
            j["e8p"] = e8p::word_stats(e);
        }
        j
    }).collect())
}

fn dims_json(sc: &Scenario, kind: Kind) -> Value {
    let live: usize = sc
        .tiles
        .iter()
        .filter(|t| t.0 >= 0)
        .map(|t| t.1.iter().filter(|&&s| s >= 0).count())
        .sum();
    json!({"m": kind.m(), "k": kind.k(), "out_cols": kind.out_cols(), "tiles": sc.tiles.len(), "m_total": sc.tiles.len() * 16,
           "x_rows": sc.rows, "x_row_div": sc.div, "live_slots": live,
           "sentinel_tiles": sc.tiles.iter().filter(|t| t.0 < 0).count(),
           "padding_slots": sc.tiles.iter().filter(|t| t.0 >= 0).map(|t| t.1.iter().filter(|&&s| s < 0).count()).sum::<usize>()})
}

#[derive(Default)]
struct NegAcc {
    f32_bad: u64,
    h_bad: u64,
    scenarios: u64,
    error: Option<String>,
}

#[derive(Default)]
struct Tally {
    finite_total: u64,
    finite_fail: u64,
    exceptional_total: u64,
    exceptional_fail: u64,
    coverage_gaps: Vec<String>,
    chain_down_total: u64,
    mixed_total: u64,
    arm_counts: BTreeMap<String, u64>,
    neg: BTreeMap<(String, String, String), NegAcc>,
}

impl Tally {
    fn record(&mut self, class: Class, pass: bool) {
        match class {
            Class::Finite => {
                self.finite_total += 1;
                if !pass {
                    self.finite_fail += 1;
                }
            }
            Class::Exceptional => {
                self.exceptional_total += 1;
                if !pass {
                    self.exceptional_fail += 1;
                }
            }
        }
    }
    fn neg_json(&self, required: &[String]) -> (Vec<Value>, bool) {
        let mut out = Vec::new();
        let mut all_ok = true;
        for ((codec, kind, name), n) in &self.neg {
            let gate = if name == "wrong-pack-selector" {
                "bf16"
            } else {
                "f32"
            };
            let bad = if gate == "bf16" { n.h_bad } else { n.f32_bad };
            let detected = n.error.is_none() && bad > 0;
            all_ok &= detected;
            out.push(json!({"codec": codec, "kind": kind, "negative": name, "required_gate": gate, "scenarios_run": n.scenarios,
                "f32_mismatches": n.f32_bad, "bf16_mismatches": n.h_bad, "detected": detected,
                "status": if let Some(e) = &n.error { format!("error: {e}") } else if detected { "detected".to_string() } else { "NOT_DETECTED".to_string() }}));
        }
        for r in required {
            if !self.neg.keys().any(|(_, _, n)| n == r) {
                all_ok = false;
                out.push(json!({"negative": r, "status": "not_provided", "detected": false}));
            }
        }
        (out, all_ok)
    }
}

fn load_table(a: &Args) -> Res<(QualifiedHooks, Value, Vec<String>)> {
    let path = a.probe_table.as_ref().ok_or("--probe-table is required")?;
    let id = a.table_ident();
    let (bytes, doc) = file_json(path)?;
    let mut hooks = QualifiedHooks::from_json(&doc)?;
    let mut problems = Vec::new();
    let mut divide_meta = Value::Null;
    if let Some(dp) = &a.divide_table {
        let (db, dd) = file_json(dp)?;
        if dd["selfcheck"]["ok"] != json!(true) || !id.matches(&dd["device"]) {
            problems.push(format!(
                "supplementary divide table lacks a passing {} qualification",
                id.arch()
            ));
        }
        let added = hooks.merge_divide_extras(&dd)?;
        divide_meta = json!({"path": dp, "sha256": sha256_hex(&db), "new_divide_keys": added});
    }
    // generic exact-key intrinsic observation tables (repeatable `--intrinsic-table`): each must carry its own xtx
    // qualification (identity, selfcheck, denorm); keys are conflict-checked against everything merged so far.
    let mut intr: Vec<(String, Value)> = Vec::new();
    for ip in &a.intrinsic_tables {
        let (ib, idoc) = file_json(ip)?;
        let sha = sha256_hex(&ib);
        if intr.iter().any(|(s, _)| *s == sha) {
            return Err(format!(
                "--intrinsic-table {ip} repeats an earlier table (sha256 {sha})"
            ));
        }
        for p in probe_table::intrinsic_qualification(
            &idoc,
            id.arch(),
            id.uuid.as_deref(),
            id.bdf.as_deref(),
        ) {
            problems.push(format!("intrinsic table {ip}: {p}"));
        }
        let merged = hooks
            .merge_intrinsic_observations(&idoc)
            .map_err(|e| format!("intrinsic table {ip}: {e}"))?;
        intr.push((sha.clone(), json!({"path": ip, "sha256": sha, "bytes": ib.len(), "merged": merged, "device": idoc.get("device"),
            "selfcheck": idoc.get("selfcheck"), "denorm": idoc["probe"].get("denorm").or_else(|| idoc.get("denorm")),
            "probe_co_sha256": idoc["probe"].get("co_sha256"), "probe_manifest_sha256": idoc["probe"].get("manifest_sha256"), "batches": idoc.get("batches")})));
    }
    intr.sort_by(|x, y| x.0.cmp(&y.0));
    let intr_shas: Vec<String> = intr.iter().map(|x| x.0.clone()).collect();
    let intr_meta: Vec<Value> = intr.into_iter().map(|x| x.1).collect();
    let dag = hooks.dag_consistency();
    if dag["mismatches"].as_u64().unwrap_or(1) != 0 {
        problems.push(format!(
            "probe table disagrees with the epilogue DAG operands: {}",
            dag["first"]
        ));
    }
    let top = hooks.top_level.clone();
    if top["selfcheck"]["ok"] != json!(true) {
        problems.push("probe table lacks a passing independent selfcheck".into());
    }
    if !id.matches(&top["device"]) {
        problems.push(format!(
            "probe table was not qualified on the expected {} {}",
            id.arch(),
            id.target.card()
        ));
    }
    let observed: std::collections::BTreeSet<u32> = hooks
        .entries
        .iter()
        .filter(|e| !e.from_exception)
        .map(|e| e.gate_bits)
        .collect();
    let required: std::collections::BTreeSet<u32> = (0..=u16::MAX)
        .filter(|b| b & 0x7f80 != 0x7f80)
        .map(|b| u32::from(b) << 16)
        .collect();
    if observed != required {
        problems
            .push("probe table must observe every finite BF16 gate, with no missing values".into());
    }
    let meta = json!({"path": path, "sha256": sha256_hex(&bytes), "bytes": bytes.len(), "key_counts": hooks.key_counts(), "dag_consistency": dag,
        "divide_table": divide_meta, "probe": top.get("probe"), "device": top.get("device"), "gate_sets": top.get("gate_sets"),
        "selfcheck": top.get("selfcheck"), "counts": top.get("counts"), "records_sha256": top.get("records_sha256"),
        "intrinsic_tables": intr_meta, "intrinsic_tables_sha256": intr_shas});
    Ok((hooks, meta, problems))
}

fn head_common(a: &Args, mode: &str, scope: &str) -> Value {
    json!({"schema": schema_of(a), "target_arch": a.target_or_default().name(), "mode": mode, "scope": scope, "command": format!("qwen4_moe_sym_gfx1100_g1_oracle {}", a.argv.join(" ")),
           "source_sha256": source_hashes(), "seed": a.seed, "only_filter": a.only,
           "complete": a.only.is_none(),
           "g1c_terms": "closes only for a mixed-codec run over artifact-bound real samples with every Down input twin-chained and zero mismatches; see the final `g1c` object",
           "link_caveat": a.target_or_default().link_caveat()})
}

// ---------------------------------------------------------------------------------------------------------------
// Self-check (CPU only)
// ---------------------------------------------------------------------------------------------------------------

struct ScState {
    results: Vec<Value>,
    skipped: Vec<String>,
    tally: Tally,
    base_bad: u64,
    mask_bad: u64,
    chain_items: Vec<Scenario>,
    downs: HashMap<(u64, u64, Codec), RealRef>,
    notes: Vec<Value>,
    chain: Option<ChainDir>,
}

/// CPU-only validation of one scenario: twin expectation vs the independent CPU model (`fold_generic` on raw
/// values), coverage, the per-codec tile-mask partition (mixed scenarios), the Down twin chain, and (neg subset)
/// detection of mutated CPU kernels.
fn self_check_one(
    sc: &Scenario,
    codec: Codec,
    kind: Kind,
    hooks: &CpuOnlySelfCheckHooks,
    st: &mut ScState,
) -> Res<()> {
    let t0 = Instant::now();
    let experts = sc.build_experts(codec, kind)?;
    let x = sc.x_set(kind);
    let expert_of = |e: i32| e as usize;
    let pairs = needed_pairs(&sc.tiles, sc.div, &expert_of);
    let dense = build_dense(kind, &experts, &x, &pairs, &|k, e, r, x| {
        fold_twin(k, e, r, x)
    });
    let exp = assemble(kind, &sc.tiles, sc.div, &dense, &expert_of, hooks, false);
    let sim = simulate(Mutation::None, kind, &experts, &sc.tiles, sc.div, &x, hooks);
    let (m, oc) = (kind.m(), kind.out_cols());
    let (ca, ch) = (cmp_acc(&sim.acc, &exp, m), cmp_h(&sim.h, &exp, oc));
    let mut gaps = coverage_gaps(sc, kind, &exp.coverage);
    gaps.extend(experts.iter().flat_map(|e| e8p::word_gaps(&sc.name, e)));
    let mut mask_bad = 0u64;
    let mut mask_json = Value::Null;
    if sc.mixed {
        // each codec's masked launch writes only its own tiles; merging them over one poisoned output must equal the full expectation
        let mut acc = vec![POISON32; exp.m_total * m];
        let mut h = vec![POISON16; exp.m_total * oc];
        let mut codecs: Vec<Codec> = experts.iter().map(|e| e.codec).collect();
        codecs.sort_by_key(|c| c.name());
        codecs.dedup();
        for c in &codecs {
            let ids = mask_tiles(&sc.tiles, &experts, *c);
            let tiles_c: Vec<Tile> = sc
                .tiles
                .iter()
                .zip(&ids)
                .map(|(t, &id)| if id >= 0 { (id, t.1) } else { (-1, [0; 16]) })
                .collect();
            let pairs_c = needed_pairs(&tiles_c, sc.div, &expert_of);
            let dense_c = build_dense(kind, &experts, &x, &pairs_c, &|k, e, r, x| {
                fold_twin(k, e, r, x)
            });
            let e_c = assemble(kind, &tiles_c, sc.div, &dense_c, &expert_of, hooks, false);
            for (ti, &id) in ids.iter().enumerate() {
                if id >= 0 {
                    for lane in 0..16 {
                        let p = ti * 16 + lane;
                        acc[p * m..(p + 1) * m].copy_from_slice(&e_c.acc[p * m..(p + 1) * m]);
                        h[p * oc..(p + 1) * oc].copy_from_slice(&e_c.h[p * oc..(p + 1) * oc]);
                    }
                }
            }
        }
        let (ma, mh) = (cmp_acc(&acc, &exp, m), cmp_h(&h, &exp, oc));
        mask_bad = ma.bad_total() + mh.bad_total();
        mask_json = json!({"codecs": codecs.iter().map(|c| c.name()).collect::<Vec<_>>(), "f32": ma.to_json(), "bf16": mh.to_json()});
    }
    let pass = ca.bad_total() == 0 && ch.bad_total() == 0 && gaps.is_empty() && mask_bad == 0;
    st.base_bad += ca.bad_total() + ch.bad_total();
    st.mask_bad += mask_bad;
    st.tally.record(sc.class, pass);
    st.tally.coverage_gaps.extend(gaps.iter().cloned());
    if sc.chain_x.is_some() {
        st.tally.chain_down_total += 1;
    }
    if sc.mixed {
        st.tally.mixed_total += 1;
    }
    if kind == Kind::GateUp && sc.feeds_chain && !st.downs.is_empty() {
        let name = chain_name(if sc.mixed { "mixed" } else { "percodec" }, codec, &sc.name);
        let tok = match &st.chain {
            None => Err("no --chain-dir: the Down chain input was not prepared (run --prepare-chain, then the A4 helper)".to_string()),
            Some(dir) => dir.tok_for(&name, &exp.h, &exp.row_class, &chain::tiles_sha(&sc.tiles)),
        };
        match tok.and_then(|t| scenarios::chain_down(sc, &st.downs, t)) {
            Ok(Some(s)) => st.chain_items.push(s),
            Ok(None) => st.notes.push(json!({"scenario": sc.name, "status": "skipped: an expert has no Down tensor in the sample set"})),
            Err(e) => {
                st.notes.push(json!({"scenario": sc.name, "chain": name, "status": format!("error: {e}")}));
                st.tally.record(Class::Finite, false);
            }
        }
    }
    if sc.neg {
        for &mu in codec.mutations() {
            let s2 = simulate(mu, kind, &experts, &sc.tiles, sc.div, &x, hooks);
            let n = st
                .tally
                .neg
                .entry((codec.name().into(), kind.name().into(), mu.name().into()))
                .or_default();
            n.scenarios += 1;
            n.f32_bad += cmp_acc(&s2.acc, &exp, m).bad_total();
            n.h_bad += cmp_h(&s2.h, &exp, oc).bad_total();
        }
    }
    st.results.push(json!({"codec": codec.name(), "kind": kind.name(), "scenario": sc.name, "class": format!("{:?}", sc.class),
        "weights": if sc.real { "real" } else { "synthetic" }, "chain_down": sc.chain_x.is_some(), "mixed": sc.mixed,
        "dims": dims_json(sc, kind), "recipes": sc.recipe_json(), "experts": experts_json(&experts), "coverage": exp.coverage, "coverage_gaps": gaps,
        "expected_vs_independent_cpu_model": {"f32": ca.to_json(), "bf16": ch.to_json()}, "mask_partition_union_vs_full": mask_json, "pass": pass,
        "seconds": t0.elapsed().as_secs_f64()}));
    eprintln!(
        "[self-check] {} {} {} pass={pass}",
        codec.name(),
        kind.name(),
        sc.name
    );
    Ok(())
}

fn run_self_check(a: &Args) -> Res<bool> {
    let out = a.out.as_ref().ok_or("--out is required")?;
    if !a.mixed.is_empty() {
        return Err("--self-check validates the mask/chain logic on the CPU itself; --mixed-codeobject is a device flag".into());
    }
    let codecs: Vec<Codec> = match &a.codec {
        Some(c) => vec![Codec::parse(c)?],
        None => vec![Codec::Mq4, Codec::W2, Codec::W3, Codec::E8p],
    };
    let hooks = CpuOnlySelfCheckHooks::default();
    let table = if a.probe_table.is_some() {
        Some(load_table(a)?)
    } else {
        None
    };
    let proto = probe_table::protocol_selftest();
    let mut head = head_common(
        a,
        "self-check",
        if a.sample_dir.is_some() {
            "cpu-only; real weights; synthetic A4/routes; Down input twin-chained"
        } else {
            "cpu-only; synthetic"
        },
    );
    head["device"] = Value::Null;
    head["hooks"] =
        json!({"kind": "CpuOnlySelfCheckHooks (host math; NEVER a device expectation)"});
    head["probe_table"] = table.as_ref().map(|t| t.1.clone()).unwrap_or(Value::Null);
    head["hook_protocol_selftest"] = proto.clone();
    let mut problems: Vec<String> = table.as_ref().map(|t| t.2.clone()).unwrap_or_default();
    if proto["pass"] != json!(true) {
        problems.push("hook protocol selftest failed".into());
    }
    let chain_loaded: Option<ChainDir> = match &a.chain_dir {
        None => {
            head["chain"] = json!({"dir": Value::Null, "status": "not provided: real-sample Down chain scenarios are reported as failures"});
            None
        }
        Some(d) => match ChainDir::load(std::path::Path::new(d)) {
            Ok(c) => Some(c),
            Err(e) => {
                problems.push(e);
                None
            }
        },
    };
    let mut st = ScState {
        results: Vec::new(),
        skipped: Vec::new(),
        tally: Tally::default(),
        base_bad: 0,
        mask_bad: 0,
        chain_items: Vec::new(),
        downs: HashMap::new(),
        notes: Vec::new(),
        chain: chain_loaded,
    };
    let mut set_meta = Vec::new();
    for codec in &codecs {
        let codec = *codec;
        let set = match &a.sample_dir {
            Some(d) => Some(samples::load(std::path::Path::new(d), codec)?),
            None => None,
        };
        // hot MQ4 companion of the same directory (for the mixed scenarios) when the run codec is a cold one
        let hot_set = match (&a.sample_dir, codec) {
            (Some(d), Codec::W2 | Codec::W3 | Codec::E8p) => {
                samples::load(std::path::Path::new(d), Codec::Mq4).ok()
            }
            _ => None,
        };
        st.downs.clear();
        for s in set.iter().chain(hot_set.iter()) {
            for r in &s.down {
                st.downs.insert((r.layer, r.expert, r.codec), r.clone());
            }
        }
        if let Some(s) = &set {
            set_meta.push(json!({"codec": codec.name(), "manifest_sha256": s.manifest_sha256, "provisional": s.provisional, "artifact_sha256": s.artifact_sha256, "meta": s.meta}));
        }
        if let Some(c) = &st.chain {
            let exp_root = RootExpect {
                source_sha256: source_hashes(),
                probe_table_sha256: None,
                divide_table_sha256: None,
                intrinsic_tables_sha256: Vec::new(),
                hooks: HOOKS_CPU_ONLY,
                sample_manifests: set
                    .iter()
                    .map(|s| (codec.name().to_string(), s.manifest_sha256.clone()))
                    .chain(
                        hot_set
                            .iter()
                            .map(|s| ("mq4".to_string(), s.manifest_sha256.clone())),
                    )
                    .collect(),
                seed: a.seed,
                oracle_binary_sha256: exe_sha(),
            };
            problems.extend(c.root_problems(&exp_root));
            head["chain"] = c.root_info(&exp_root);
        }
        for kind in [Kind::GateUp, Kind::Down] {
            let mut list = scenario_list(a, codec, kind, &set);
            if let (Kind::GateUp, Some(cs), Some(hs)) = (kind, &set, &hot_set) {
                let mut v = scenarios::mixed(&cs.gate_up, &hs.gate_up, a.seed);
                if let Some(f) = &a.only {
                    v.retain(|s| s.name.contains(f.as_str()));
                }
                list.extend(v);
            }
            let chain: Vec<Scenario> = st.chain_items.drain(..).collect();
            list.extend(chain);
            for sc in list {
                if sc.production {
                    st.skipped
                        .push(format!("{}/{}/{}", codec.name(), kind.name(), sc.name));
                    continue;
                }
                self_check_one(&sc, codec, kind, &hooks, &mut st)?;
            }
        }
    }
    let mut required: Vec<String> = codecs
        .iter()
        .flat_map(|c| c.mutations().iter().map(|m| m.name().to_string()))
        .collect();
    required.sort();
    required.dedup();
    let (negs, neg_ok) = st.tally.neg_json(&required);
    let pass = problems.is_empty()
        && st.base_bad == 0
        && st.mask_bad == 0
        && st.tally.finite_fail == 0
        && st.tally.exceptional_fail == 0
        && neg_ok;
    let mut doc = head;
    doc["status"] = json!("complete");
    doc["sample_dirs"] = Value::Array(set_meta);
    doc["results"] = Value::Array(st.results);
    doc["negative_controls"] = Value::Array(negs);
    doc["skipped_in_self_check"] = json!({"scenarios": st.skipped, "reason": "production-size cases are CPU-heavy and only meaningful against a device"});
    doc["problems"] = json!(problems);
    doc["coverage_gaps"] = json!(st.tally.coverage_gaps);
    doc["chain_notes"] = Value::Array(st.notes);
    doc["g1c"] = json!({"closed": false, "blockers": ["cpu-only run: no device launch"], "note": "G1c closes only in a device mixed-codec run (see --mixed-codeobject)"});
    doc["summary"] = json!({"finite_total": st.tally.finite_total, "finite_fail": st.tally.finite_fail, "exceptional_total": st.tally.exceptional_total,
        "exceptional_fail": st.tally.exceptional_fail, "baseline_mismatches": st.base_bad, "mask_partition_mismatches": st.mask_bad,
        "mixed_scenarios": st.tally.mixed_total, "chain_down_scenarios": st.tally.chain_down_total, "negative_controls_all_detected": neg_ok});
    doc["pass"] = json!(pass);
    write_json(out, &doc)?;
    println!(
        "{}",
        json!({"schema": schema_of(a), "mode": "self-check", "out": out, "pass": pass, "summary": doc["summary"], "problems": doc["problems"]})
    );
    Ok(pass)
}

// ---------------------------------------------------------------------------------------------------------------
// Device run
// ---------------------------------------------------------------------------------------------------------------

struct KindFns {
    acc: Function,
    h: Function,
    anchor_acc: Option<Function>,
    anchor_h: Option<Function>,
    negs: Vec<(String, Result<(Function, Function), String>)>,
}

struct MixedFns {
    hot: (Function, Function),
    cold: (Function, Function),
    cold_codec: Codec,
    anchor_acc: Option<Function>,
    /// `None` for e8p: its codewords (|v| up to 11) have no MQ4 anchor encoding.
    anchor_h: Option<Function>,
    anchor_note: &'static str,
}

/// (cold-codec residency, hot-codec residency): hot VRAM + cold in each arm, then both on mapped / guarded storage.
const MIXED_ARMS: [(Arm, Arm); 5] = [
    (Arm::Vram, Arm::Vram),
    (Arm::Mapped, Arm::Vram),
    (Arm::Guarded, Arm::Vram),
    (Arm::Mapped, Arm::Mapped),
    (Arm::Guarded, Arm::Guarded),
];

fn mixed_arm_label(cold: Arm, hot: Arm) -> String {
    format!("cold={},hot={}", cold.name(), hot.name())
}

struct Ctx<'a> {
    hip: &'a HipRuntime,
    dev: i32,
    codec: Codec,
    hooks: &'a QualifiedHooks,
    head: Value,
    out: String,
    results: Vec<Value>,
    tally: Tally,
    sentinel_arm_notes: Vec<Value>,
    extra: BTreeMap<String, Value>,
    required_negs: Vec<String>,
    mixed: bool,
    real_samples: bool,
    provisional: bool,
    artifact_bound: bool,
    down_lookup: HashMap<(u64, u64, Codec), RealRef>,
    chain_items: Vec<Scenario>,
    chain_notes: Vec<Value>,
    chain_dir: Option<ChainDir>,
    scenarios_since_flush: usize,
}

impl<'a> Ctx<'a> {
    fn doc(&self, status: &str) -> Value {
        let (negs, neg_ok) = self.tally.neg_json(&self.required_negs);
        let fixture_ok = self
            .extra
            .get("epilogue_fixture")
            .is_some_and(|f| f["pass"] == json!(true));
        let positive_pass = self.tally.finite_fail == 0
            && self.tally.exceptional_fail == 0
            && self.tally.coverage_gaps.is_empty()
            && fixture_ok;
        let mut d = self.head.clone();
        d["status"] = json!(status);
        for (k, v) in &self.extra {
            d[k.as_str()] = v.clone();
        }
        d["results"] = Value::Array(self.results.clone());
        d["negative_controls"] = Value::Array(negs);
        d["coverage_gaps"] = json!(self.tally.coverage_gaps);
        d["summary"] = json!({"finite_total": self.tally.finite_total, "finite_fail": self.tally.finite_fail,
            "exceptional_total": self.tally.exceptional_total, "exceptional_fail": self.tally.exceptional_fail,
            "positive_pass": positive_pass, "negative_controls_all_detected": neg_ok,
            "finite_domain_pass": self.tally.finite_fail == 0, "exceptional_pass": self.tally.exceptional_fail == 0,
            "mixed_scenarios": self.tally.mixed_total, "chain_down_scenarios": self.tally.chain_down_total, "arm_counts": self.tally.arm_counts});
        let denorm_ok = d["denorm"]["ok"] == json!(true);
        let complete = d["complete"] == json!(true);
        let arms_ok = MIXED_ARMS.iter().all(|(c, h)| {
            self.tally
                .arm_counts
                .get(&mixed_arm_label(*c, *h))
                .copied()
                .unwrap_or(0)
                > 0
        });
        let req = json!({"mixed_codec_run": self.mixed, "real_weight_samples": self.real_samples,
            "mixed_scenarios_ran": self.tally.mixed_total > 0, "down_twin_chain_ran": self.tally.chain_down_total > 0,
            "all_mixed_residency_arms_ran": arms_ok, "positive_pass": positive_pass, "denorm_ok": denorm_ok,
            "epilogue_fixture_pass": fixture_ok, "complete_unfiltered": complete, "run_complete": status == "complete"});
        let blockers: Vec<String> = req
            .as_object()
            .map(|o| {
                o.iter()
                    .filter(|(_, v)| **v != json!(true))
                    .map(|(k, _)| k.clone())
                    .collect()
            })
            .unwrap_or_default();
        let measured_pass = blockers.is_empty();
        // Artifact binding is recorded, not a pass condition: provisional hash-checked source/R1-pack-bin samples count as real weights.
        let binding = if self.artifact_bound {
            "bound"
        } else {
            "pending"
        };
        let verdict = match (measured_pass, self.artifact_bound) {
            (true, false) => "G1c PASS, artifact binding pending",
            (true, true) => "G1c PASS, artifact bound",
            (false, _) => "G1c NOT PASSED",
        };
        d["g1c"] = json!({"measured_pass": measured_pass, "verdict": verdict, "closed": measured_pass && self.artifact_bound,
            "scope": if self.real_samples { SCOPE_REAL } else { "synthetic" }, "requirements": req, "blockers": blockers,
            "artifact_binding": {"status": binding, "provisional_samples": self.provisional,
                "artifact_sha256": {"cold_or_run_codec": self.head["inputs"]["sample_dir"]["artifact_sha256"], "hot_companion": self.head["inputs"]["sample_dir_hot_companion"]["artifact_sha256"]},
                "closure_rule": "closed = measured_pass and artifact_binding.status == bound; once the final c24 artifact exists, a CPU check that every sample's expert tensor bytes (by sha256) equal the artifact's slices for the same layer/expert/projection binds this result without a GPU rerun; if any slice differs, rerun on the device"},
            "chain": {"definition": "synthetic A4 -> CPU twin GateUp (same real expert weights, qualified hooks) -> prepared chain-h.json -> external helper (pilot_solve a4_emulate over the R128 rotation, candidates=2) -> chain-a4.json loaded as the Down XSet; Down A4 rows indexed by grouped row p; Down slots p / -1, x_row_div 1, expert/tile order preserved; the candidate and Python are never run by this binary for chain inputs",
                      "down_scenarios_run": self.tally.chain_down_total, "notes": self.chain_notes},
            "note": "per-codec runs never pass mixed-codec G1c; this object is the only place G1c status is stated"});
        d["pass"] = json!(status == "complete" && positive_pass && neg_ok && denorm_ok && complete);
        d
    }

    fn flush(&self, status: &str) -> Res<()> {
        write_json(&self.out, &self.doc(status))
    }

    fn progress(&self, kind: Kind, scenario: &str, phase: &str) {
        let p = json!({"status": "in_progress", "kind": kind.name(), "scenario": scenario, "phase": phase,
            "finite_fail": self.tally.finite_fail, "exceptional_fail": self.tally.exceptional_fail, "results": self.results.len()});
        let _ = std::fs::write(
            format!("{}.progress", self.out),
            serde_json::to_vec(&p).unwrap_or_default(),
        );
    }

    fn scenario(&mut self, kind: Kind, f: &KindFns, sc: &Scenario) -> Res<()> {
        let hip = self.hip;
        let hooks = self.hooks;
        let codec = self.codec;
        let t0 = Instant::now();
        self.progress(kind, &sc.name, "cpu_expectation");
        if sc.chain_x.is_some() {
            self.tally.chain_down_total += 1;
        }
        let experts = sc.build_experts(codec, kind)?;
        let x = sc.x_set(kind);
        let referenced = sc.referenced();
        let xq = epoch_major(&x, &referenced);
        let expert_of = |e: i32| e as usize;
        let pairs = needed_pairs(&sc.tiles, sc.div, &expert_of);
        let dense = build_dense(kind, &experts, &x, &pairs, &|k, e, r, x| {
            fold_twin(k, e, r, x)
        });
        hooks.clear_missing();
        let exp = assemble(kind, &sc.tiles, sc.div, &dense, &expert_of, hooks, false);
        let (missing, missing_samples) = (hooks.missing_count(), hooks.missing_samples());
        drop(dense);
        let mut gaps = coverage_gaps(sc, kind, &exp.coverage);
        gaps.extend(experts.iter().flat_map(|e| e8p::word_gaps(&sc.name, e)));
        self.tally.coverage_gaps.extend(gaps.iter().cloned());
        if kind == Kind::GateUp && sc.feeds_chain {
            self.queue_chain(sc, &exp, missing)?;
        }
        let cpu_secs = t0.elapsed().as_secs_f64();
        let dc = DeviceCase::new(hip, kind, &sc.tiles, sc.div, sc.rows, &xq)?;
        let cand: Vec<&[u8]> = experts.iter().map(|e| e.bytes.as_slice()).collect();
        let anchor: Vec<&[u8]> = experts.iter().map(|e| e.anchor.as_slice()).collect();
        let (m, oc) = (kind.m(), kind.out_cols());
        let (mut vram_acc, mut vram_h) = (Vec::new(), Vec::new());
        let class_s = format!("{:?}", sc.class);
        for arm in Arm::ALL {
            self.progress(kind, &sc.name, &format!("launch:{}", arm.name()));
            let res = Resident::new(hip, self.dev, arm, &cand)?;
            let info = res.info.clone();
            let r = with_ptr_table(hip, &res.ptrs, |pt| {
                let a = dc.run(&f.acc, pt, dc.acc_bytes())?;
                let h = dc.run(&f.h, pt, dc.h_bytes())?;
                Ok((a, h))
            });
            res.release(hip)?;
            let ((acc_b, acc_tail), (h_b, h_tail)) = r?;
            let (acc, h) = (le_u32s(&acc_b), le_u16s(&h_b));
            let (ca, ch) = (cmp_acc(&acc, &exp, m), cmp_h(&h, &exp, oc));
            let pass = ca.bad_total() == 0
                && ch.bad_total() == 0
                && acc_tail == 0
                && h_tail == 0
                && missing == 0;
            self.tally.record(sc.class, pass);
            *self
                .tally
                .arm_counts
                .entry(arm.name().to_string())
                .or_default() += 1;
            self.results.push(json!({"check": "candidate_vs_cpu", "chain_down": sc.chain_x.is_some(), "codec": codec.name(), "kind": kind.name(), "scenario": sc.name, "class": class_s,
                "weights": if sc.real { "real" } else { "synthetic" }, "residency": info, "dims": dims_json(sc, kind), "recipes": sc.recipe_json(),
                "experts": experts_json(&experts), "coverage": exp.coverage, "coverage_gaps": gaps,
                "f32_acc": ca.to_json(), "bf16": ch.to_json(),
                "poison_tail_bytes_changed": {"f32_acc": acc_tail, "bf16": h_tail},
                "padding_and_sentinel": {"f32_pad_mismatches": ca.pad.bad, "f32_sentinel_mismatches": ca.sentinel.bad, "bf16_pad_mismatches": ch.pad.bad, "bf16_sentinel_mismatches": ch.sentinel.bad},
                "hook_missing_keys": missing, "hook_missing_samples": missing_samples, "cpu_seconds": cpu_secs, "pass": pass}));
            if arm == Arm::Vram {
                vram_acc = acc;
                vram_h = h;
            }
        }
        // same-device MQ4 anchor on bit-equivalent QT44/QT53 tensors (required in addition to CPU equality)
        if let Some(ah) = &f.anchor_h {
            self.progress(kind, &sc.name, "anchor");
            let res = Resident::new(hip, self.dev, Arm::Vram, &anchor)?;
            let r = with_ptr_table(hip, &res.ptrs, |pt| {
                let h = dc.run(ah, pt, dc.h_bytes())?;
                let a = match &f.anchor_acc {
                    Some(fa) => Some(dc.run(fa, pt, dc.acc_bytes())?),
                    None => None,
                };
                Ok((h, a))
            });
            res.release(hip)?;
            let ((ah_b, ah_tail), a_acc) = r?;
            let ah_v = le_u16s(&ah_b);
            let (bad_vs_cand, first_vs_cand) = cmp_pair_u16(&vram_h, &ah_v);
            let anchor_vs_cpu = cmp_h(&ah_v, &exp, oc);
            let acc_part = match &a_acc {
                Some((b, tail)) => {
                    let v = le_u32s(b);
                    let (bad, first) = cmp_pair_u32(&vram_acc, &v);
                    json!({"present": true, "acc_vs_candidate_mismatches": bad, "first": first, "poison_tail_bytes_changed": tail})
                }
                None => json!({"present": false}),
            };
            let acc_bad = a_acc
                .as_ref()
                .map_or(0, |(b, t)| cmp_pair_u32(&vram_acc, &le_u32s(b)).0 + *t);
            let pass = bad_vs_cand == 0
                && anchor_vs_cpu.bad_total() == 0
                && ah_tail == 0
                && acc_bad == 0
                && missing == 0;
            self.tally.record(sc.class, pass);
            self.results.push(json!({"check": "mq4_anchor", "codec": codec.name(), "kind": kind.name(), "scenario": sc.name, "class": class_s,
                "bf16_candidate_vs_anchor_mismatches": bad_vs_cand, "first": first_vs_cand, "anchor_bf16_vs_cpu": anchor_vs_cpu.to_json(),
                "anchor_poison_tail_bytes_changed": ah_tail, "anchor_acc": acc_part,
                "note": "anchor equality is required in addition to, never instead of, CPU equality", "pass": pass}));
        }
        // negative controls: broken candidate objects must make the relevant gate fail
        if sc.neg {
            for (name, fns) in &f.negs {
                let key = (
                    codec.name().to_string(),
                    kind.name().to_string(),
                    name.clone(),
                );
                match fns {
                    Err(e) => {
                        self.tally.neg.entry(key).or_default().error = Some(e.clone());
                    }
                    Ok((nacc, nh)) => {
                        self.progress(kind, &sc.name, &format!("negative:{name}"));
                        let res = Resident::new(hip, self.dev, Arm::Vram, &cand)?;
                        let r = with_ptr_table(hip, &res.ptrs, |pt| {
                            let a = dc.run(nacc, pt, dc.acc_bytes())?;
                            let h = dc.run(nh, pt, dc.h_bytes())?;
                            Ok((a, h))
                        });
                        res.release(hip)?;
                        let ((acc_b, _), (h_b, _)) = r?;
                        let (ca, ch) = (
                            cmp_acc(&le_u32s(&acc_b), &exp, m),
                            cmp_h(&le_u16s(&h_b), &exp, oc),
                        );
                        let n = self.tally.neg.entry(key).or_default();
                        n.scenarios += 1;
                        n.f32_bad += ca.bad_total();
                        n.h_bad += ch.bad_total();
                    }
                }
            }
        }
        dc.free()?;
        self.scenarios_since_flush += 1;
        eprintln!(
            "[g1] {} {} {} ({:.1}s)",
            codec.name(),
            kind.name(),
            sc.name,
            t0.elapsed().as_secs_f64()
        );
        if self.scenarios_since_flush >= 8 {
            self.scenarios_since_flush = 0;
            self.flush("in_progress")?;
        }
        Ok(())
    }

    /// Build the twin-chained Down scenario of a GateUp scenario from its CPU expectation (never from the candidate).
    fn queue_chain(&mut self, sc: &Scenario, exp: &Expectation, missing: u64) -> Res<()> {
        if self.down_lookup.is_empty() {
            return Ok(());
        }
        if missing > 0 {
            self.chain_notes.push(json!({"scenario": sc.name, "status": "blocked: GateUp expectation has unresolved probe-table keys"}));
            self.tally.record(Class::Finite, false);
            self.results.push(json!({"check": "chain_blocked", "scenario": sc.name, "reason": "unresolved hook keys in the GateUp twin; Down chain input cannot be formed", "pass": false}));
            return Ok(());
        }
        let name = chain_name(
            if sc.mixed { "mixed" } else { "percodec" },
            self.codec,
            &sc.name,
        );
        let tok = match &self.chain_dir {
            None => Err("no --chain-dir: the Down chain input was not prepared (run --prepare-chain on hiptrx, then the A4 helper)".to_string()),
            Some(dir) => dir.tok_for(&name, &exp.h, &exp.row_class, &chain::tiles_sha(&sc.tiles)),
        };
        match tok.and_then(|t| scenarios::chain_down(sc, &self.down_lookup, t)) {
            Ok(Some(s)) => self.chain_items.push(s),
            Ok(None) => self.chain_notes.push(json!({"scenario": sc.name, "status": "skipped: an expert has no Down tensor in the sample set"})),
            Err(e) => {
                self.chain_notes.push(json!({"scenario": sc.name, "chain": name, "status": format!("error: {e}")}));
                self.tally.record(Class::Finite, false);
                self.results.push(json!({"check": "chain_input_rejected", "scenario": sc.name, "chain": name, "error": e, "pass": false}));
            }
        }
        Ok(())
    }

    /// Mixed-codec scenario: global expert IDs, one tile list / slot array / poisoned output; per-codec masked tile
    /// ids and pointer tables (null for experts the codec does not own); sequential path-loaded launches; the MQ4
    /// anchor runs the unmasked QT44/QT53-equivalent union.
    fn scenario_mixed(&mut self, kind: Kind, f: &MixedFns, sc: &Scenario) -> Res<()> {
        let hip = self.hip;
        let hooks = self.hooks;
        let t0 = Instant::now();
        self.progress(kind, &sc.name, "cpu_expectation");
        if sc.chain_x.is_some() {
            self.tally.chain_down_total += 1;
        }
        self.tally.mixed_total += 1;
        let experts = sc.build_experts(Codec::Mq4, kind)?;
        let x = sc.x_set(kind);
        let referenced = sc.referenced();
        let xq = epoch_major(&x, &referenced);
        let expert_of = |e: i32| e as usize;
        let pairs = needed_pairs(&sc.tiles, sc.div, &expert_of);
        let dense = build_dense(kind, &experts, &x, &pairs, &|k, e, r, x| {
            fold_twin(k, e, r, x)
        });
        hooks.clear_missing();
        let exp = assemble(kind, &sc.tiles, sc.div, &dense, &expert_of, hooks, false);
        let (missing, missing_samples) = (hooks.missing_count(), hooks.missing_samples());
        drop(dense);
        let mut gaps = coverage_gaps(sc, kind, &exp.coverage);
        gaps.extend(experts.iter().flat_map(|e| e8p::word_gaps(&sc.name, e)));
        self.tally.coverage_gaps.extend(gaps.iter().cloned());
        if kind == Kind::GateUp && sc.feeds_chain {
            self.queue_chain(sc, &exp, missing)?;
        }
        let n = experts.len();
        let hot_idx: Vec<usize> = (0..n).filter(|&i| experts[i].codec == Codec::Mq4).collect();
        let cold_idx: Vec<usize> = (0..n)
            .filter(|&i| experts[i].codec == f.cold_codec)
            .collect();
        if hot_idx.len() + cold_idx.len() != n {
            return Err(format!(
                "mixed scenario {} holds an expert that is neither mq4 nor {}",
                sc.name,
                f.cold_codec.name()
            ));
        }
        let cpu_secs = t0.elapsed().as_secs_f64();
        let dc = DeviceCase::new(hip, kind, &sc.tiles, sc.div, sc.rows, &xq)?;
        let (m, oc) = (kind.m(), kind.out_cols());
        let le_i32 =
            |ids: &[i32]| -> Vec<u8> { ids.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let (hot_ids, cold_ids) = (
            mask_tiles(&sc.tiles, &experts, Codec::Mq4),
            mask_tiles(&sc.tiles, &experts, f.cold_codec),
        );
        let (hot_tiles, cold_tiles) = (
            upload(hip, &le_i32(&hot_ids))?,
            upload(hip, &le_i32(&cold_ids))?,
        );
        let (hot_live, cold_live) = (
            hot_ids.iter().filter(|&&v| v >= 0).count(),
            cold_ids.iter().filter(|&&v| v >= 0).count(),
        );
        let hot_t: Vec<&[u8]> = hot_idx
            .iter()
            .map(|&i| experts[i].bytes.as_slice())
            .collect();
        let cold_t: Vec<&[u8]> = cold_idx
            .iter()
            .map(|&i| experts[i].bytes.as_slice())
            .collect();
        let class_s = format!("{:?}", sc.class);
        let (mut vram_acc, mut vram_h) = (Vec::new(), Vec::new());
        for (cold_arm, hot_arm) in MIXED_ARMS {
            let label = mixed_arm_label(cold_arm, hot_arm);
            self.progress(kind, &sc.name, &format!("launch:{label}"));
            let res_hot = if hot_t.is_empty() {
                None
            } else {
                Some(Resident::new(hip, self.dev, hot_arm, &hot_t)?)
            };
            let res_cold = if cold_t.is_empty() {
                None
            } else {
                Some(Resident::new(hip, self.dev, cold_arm, &cold_t)?)
            };
            let table = |idx: &[usize], res: &Option<Resident>| -> Vec<u8> {
                let mut t = vec![0u64; n];
                if let Some(r) = res {
                    for (pos, &i) in idx.iter().enumerate() {
                        t[i] = r.ptrs[pos];
                    }
                }
                t.iter().flat_map(|p| p.to_le_bytes()).collect()
            };
            let (pt_hot, pt_cold) = (
                upload(hip, &table(&hot_idx, &res_hot))?,
                upload(hip, &table(&cold_idx, &res_cold))?,
            );
            let (mut acc_l, mut h_l): (
                Vec<(&Function, &DeviceBuffer, &DeviceBuffer)>,
                Vec<(&Function, &DeviceBuffer, &DeviceBuffer)>,
            ) = (Vec::new(), Vec::new());
            if !hot_idx.is_empty() {
                acc_l.push((&f.hot.0, &pt_hot, &hot_tiles));
                h_l.push((&f.hot.1, &pt_hot, &hot_tiles));
            }
            if !cold_idx.is_empty() {
                acc_l.push((&f.cold.0, &pt_cold, &cold_tiles));
                h_l.push((&f.cold.1, &pt_cold, &cold_tiles));
            }
            let r = (|| -> Res<_> {
                let a = dc.run_multi(&acc_l, dc.acc_bytes())?;
                let h = dc.run_multi(&h_l, dc.h_bytes())?;
                Ok((a, h))
            })();
            let info = json!({"cold": res_cold.as_ref().map(|r| r.info.clone()), "hot": res_hot.as_ref().map(|r| r.info.clone())});
            hip.free(pt_hot).map_err(err)?;
            hip.free(pt_cold).map_err(err)?;
            if let Some(r) = res_hot {
                r.release(hip)?;
            }
            if let Some(r) = res_cold {
                r.release(hip)?;
            }
            let ((acc_b, acc_tail), (h_b, h_tail)) = r?;
            let (acc, h) = (le_u32s(&acc_b), le_u16s(&h_b));
            let (ca, ch) = (cmp_acc(&acc, &exp, m), cmp_h(&h, &exp, oc));
            let pass = ca.bad_total() == 0
                && ch.bad_total() == 0
                && acc_tail == 0
                && h_tail == 0
                && missing == 0;
            self.tally.record(sc.class, pass);
            *self.tally.arm_counts.entry(label.clone()).or_default() += 1;
            self.results.push(json!({"check": "mixed_candidates_vs_cpu", "chain_down": sc.chain_x.is_some(), "mixed_codecs": {"hot": "mq4", "cold": f.cold_codec.name()},
                "arm": label, "residency": info, "kind": kind.name(), "scenario": sc.name, "class": class_s, "weights": "real",
                "tiles": {"total": sc.tiles.len(), "hot_codec_tiles": hot_live, "cold_codec_tiles": cold_live},
                "dims": dims_json(sc, kind), "recipes": sc.recipe_json(), "experts": experts_json(&experts), "coverage": exp.coverage, "coverage_gaps": gaps,
                "f32_acc": ca.to_json(), "bf16": ch.to_json(), "poison_tail_bytes_changed": {"f32_acc": acc_tail, "bf16": h_tail},
                "padding_and_sentinel": {"f32_pad_mismatches": ca.pad.bad, "f32_sentinel_mismatches": ca.sentinel.bad, "bf16_pad_mismatches": ch.pad.bad, "bf16_sentinel_mismatches": ch.sentinel.bad},
                "hook_missing_keys": missing, "hook_missing_samples": missing_samples, "cpu_seconds": cpu_secs, "pass": pass}));
            if (cold_arm, hot_arm) == (Arm::Vram, Arm::Vram) {
                vram_acc = acc;
                vram_h = h;
            }
        }
        // anchor union: every expert as its QT44/QT53-equivalent tensor, unmasked tile ids, one MQ4 kernel (not for e8p)
        if let Some(anchor_h) = &f.anchor_h {
            self.progress(kind, &sc.name, "anchor_union");
            let anchor: Vec<&[u8]> = experts.iter().map(|e| e.anchor.as_slice()).collect();
            let res = Resident::new(hip, self.dev, Arm::Vram, &anchor)?;
            let r = with_ptr_table(hip, &res.ptrs, |pt| {
                let h = dc.run(anchor_h, pt, dc.h_bytes())?;
                let a = match &f.anchor_acc {
                    Some(fa) => Some(dc.run(fa, pt, dc.acc_bytes())?),
                    None => None,
                };
                Ok((h, a))
            });
            res.release(hip)?;
            let ((ah_b, ah_tail), a_acc) = r?;
            let ah_v = le_u16s(&ah_b);
            let (bad_vs_cand, first_vs_cand) = cmp_pair_u16(&vram_h, &ah_v);
            let anchor_vs_cpu = cmp_h(&ah_v, &exp, oc);
            let (acc_bad, acc_first) = match &a_acc {
                Some((b, tail)) => {
                    let (bad, first) = cmp_pair_u32(&vram_acc, &le_u32s(b));
                    (bad + *tail, first)
                }
                None => (0, None),
            };
            let pass = bad_vs_cand == 0
                && anchor_vs_cpu.bad_total() == 0
                && ah_tail == 0
                && acc_bad == 0
                && missing == 0;
            self.tally.record(sc.class, pass);
            self.results.push(json!({"check": "mixed_mq4_anchor_union", "chain_down": sc.chain_x.is_some(), "kind": kind.name(), "scenario": sc.name, "class": class_s,
                "bf16_combined_vs_anchor_mismatches": bad_vs_cand, "first": first_vs_cand, "anchor_bf16_vs_cpu": anchor_vs_cpu.to_json(),
                "anchor_poison_tail_bytes_changed": ah_tail, "anchor_acc_present": a_acc.is_some(), "acc_combined_vs_anchor_mismatches_plus_tail": acc_bad, "acc_first": acc_first,
                "anchor_module": f.anchor_note, "note": "anchor equality is required in addition to, never instead of, CPU equality", "pass": pass}));
        } else {
            self.results.push(json!({"check": "mixed_mq4_anchor_union", "kind": kind.name(), "scenario": sc.name, "applicable": false,
                "reason": "e8p cold experts have no QT44/QT53 anchor encoding; the CPU twin (K-E0 arithmetic + reviewed MulUp BF16 adapter) is the sole expectation", "pass": true}));
        }
        dc.free()?;
        hip.free(hot_tiles).map_err(err)?;
        hip.free(cold_tiles).map_err(err)?;
        self.scenarios_since_flush += 1;
        eprintln!(
            "[g1-mixed] {} {} ({:.1}s)",
            kind.name(),
            sc.name,
            t0.elapsed().as_secs_f64()
        );
        if self.scenarios_since_flush >= 8 {
            self.scenarios_since_flush = 0;
            self.flush("in_progress")?;
        }
        Ok(())
    }
}

fn fixture_values(
    kind: Kind,
    pass: &str,
    p_rows: usize,
    seed: u32,
    finite_bf16: &[u32],
) -> Vec<f32> {
    let m = kind.m();
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b1) | 1);
    let mut v = vec![0f32; p_rows * m];
    let specials: [u32; 20] = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x0080_0000,
        0x3f80_8000,
        0x3f81_8000,
        0xbf80_8000,
        0xbf81_8000,
        0x42ce_8ed0,
        0x42ce_8ed1,
        0x42ce_8ecf,
        0xc2b1_7218,
        0xc2b1_7219,
        0xc2b1_7217,
        0x7f7f_0000,
        0xff7f_0000,
        0x7f7e_8000,
    ];
    for p in 0..p_rows {
        for c in 0..m {
            let n = p * m + c;
            let is_gate_col = kind == Kind::GateUp && c < m / 2;
            let bits: u32 = match pass {
                "all_finite_bf16_gates" => {
                    if kind == Kind::Down || is_gate_col {
                        let idx = if kind == Kind::GateUp {
                            p * (m / 2) + c
                        } else {
                            n
                        };
                        finite_bf16[idx % finite_bf16.len()]
                    } else {
                        ((100 + rng.below(51)) << 23)
                            | (rng.below(0x80) << 16)
                            | (rng.below(2) << 31)
                    }
                }
                "random_f32" => {
                    let (lo, span) = if kind == Kind::GateUp {
                        (90u32, 51u32)
                    } else {
                        (1, 253)
                    };
                    (rng.below(2) << 31)
                        | ((lo + rng.below(span)) << 23)
                        | (rng.next() & 0x007f_ffff)
                }
                "ties_specials" => match rng.below(3) {
                    0 => specials[rng.below(specials.len() as u32) as usize],
                    1 => {
                        (rng.below(2) << 31)
                            | ((100 + rng.below(40)) << 23)
                            | (rng.below(0x80) << 16)
                            | 0x8000
                    }
                    _ => {
                        (rng.below(2) << 31)
                            | ((100 + rng.below(40)) << 23)
                            | (rng.next() & 0x007f_ffff)
                    }
                },
                _ => {
                    // exceptional: inf / NaN payloads / values rounding up to infinity
                    const EX: [u32; 14] = [
                        0x7f80_0000,
                        0xff80_0000,
                        0x7fc0_0000,
                        0xffc0_0000,
                        0x7fc1_2345,
                        0x7f81_2345,
                        0xffc5_4321,
                        0x7f7f_ffff,
                        0xff7f_ffff,
                        0x7f7f_8000,
                        0xff7f_8000,
                        0x7f7f_7fff,
                        0x3f80_0000,
                        0xbf80_0000,
                    ];
                    if rng.below(3) == 0 {
                        EX[rng.below(EX.len() as u32) as usize]
                    } else {
                        (rng.below(2) << 31)
                            | ((100 + rng.below(40)) << 23)
                            | (rng.next() & 0x007f_ffff)
                    }
                }
            };
            v[n] = f32::from_bits(bits);
        }
    }
    v
}

fn finite_bf16_list() -> Vec<u32> {
    (0u32..0x10000)
        .map(|h| h << 16)
        .filter(|b| b & 0x7f80_0000 != 0x7f80_0000)
        .collect()
}

/// Standalone-fixture passes, grouped-row counts and slot layout; shared by `run_fixture` and the hook-key census so
/// both walk exactly the same sums.
const FIXTURE_PASSES: [&str; 4] = [
    "all_finite_bf16_gates",
    "random_f32",
    "ties_specials",
    "exceptional_inputs",
];

fn fixture_p_rows(kind: Kind) -> usize {
    if kind == Kind::GateUp {
        112
    } else {
        128
    }
}

/// Padding at a rotating position in odd tiles, a fully padded tile at the end.
fn fixture_slots(p_rows: usize) -> Vec<i32> {
    (0..p_rows)
        .map(|p| {
            let t = p / 16;
            if t == p_rows / 16 - 1 || (t % 2 == 1 && p % 16 == (t * 5) % 16) {
                -1
            } else {
                p as i32
            }
        })
        .collect()
}

/// B4-A standalone epilogue fixture: sums -> BF16, versus the gfx1100 epilogue adapter (public `grouped_prefill_epilogue_with` + reviewed MulUp NaN rule) + qualified hooks.
/// Covers every finite BF16 gate value, random F32 (input rounding), ties/thresholds/subnormals, exceptional inputs,
/// padding positions, and requires its `wrongsel` negative to mismatch.
fn run_fixture(hip: &HipRuntime, co: &[u8], hooks: &QualifiedHooks, seed: u32) -> Res<Value> {
    let module = hip.module_load_data(co).map_err(err)?;
    let finite = finite_bf16_list();
    let mut cases = Vec::new();
    let mut all_pass = true;
    let mut neg_bad = [0u64; 2];
    let neg_names = ["wrongsel_gate_up", "wrongsel_down"];
    for (ki, kind) in [Kind::GateUp, Kind::Down].into_iter().enumerate() {
        let tag = if kind == Kind::GateUp {
            "gate_up"
        } else {
            "down"
        };
        let f = hip
            .module_get_function(
                &module,
                &format!("qwen4_moe_epilogue_fixture_{tag}_{}", model::arch_name()),
            )
            .map_err(err)?;
        let fneg = hip
            .module_get_function(
                &module,
                &format!(
                    "qwen4_moe_epilogue_fixture_{tag}_wrongsel_{}",
                    model::arch_name()
                ),
            )
            .map_err(err)?;
        let (m, oc) = (kind.m(), kind.out_cols());
        let p_rows = fixture_p_rows(kind);
        for (pi, pass_name) in FIXTURE_PASSES.iter().enumerate() {
            let class = if *pass_name == "exceptional_inputs" {
                Class::Exceptional
            } else {
                Class::Finite
            };
            let sums = fixture_values(kind, pass_name, p_rows, seed ^ (pi as u32 * 977), &finite);
            let slots = fixture_slots(p_rows);
            hooks.clear_missing();
            let want = mq_i4_reference_epilogue(kind, m, &sums, &slots, hooks);
            let (missing, missing_samples) = (hooks.missing_count(), hooks.missing_samples());
            let sums_b: Vec<u8> = sums
                .iter()
                .flat_map(|f| f.to_bits().to_le_bytes())
                .collect();
            let slots_b: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
            let (sd, ld) = (upload(hip, &sums_b)?, upload(hip, &slots_b)?);
            let mut run_one = |func: &Function| -> Res<Vec<u16>> {
                let y = hip.malloc(p_rows * oc * 2).map_err(err)?;
                hip.memset(&y, 0xa5, p_rows * oc * 2).map_err(err)?;
                let mut a = [0u8; 32];
                for (off, p) in [
                    (0usize, sd.as_ptr() as u64),
                    (8, ld.as_ptr() as u64),
                    (16, y.as_ptr() as u64),
                ] {
                    a[off..off + 8].copy_from_slice(&p.to_le_bytes());
                }
                a[24..28].copy_from_slice(&(m as u32).to_le_bytes());
                // SAFETY: 32-byte fixture ABI {sums, slots, Y, M}; buffers outlive the synchronize.
                unsafe {
                    hip.launch_kernel_blob(
                        func,
                        [(m / 64) as u32, (p_rows / 16) as u32, 1],
                        [kind.threads(), 1, 1],
                        0,
                        None,
                        &mut a,
                    )
                }
                .map_err(err)?;
                hip.device_synchronize().map_err(err)?;
                let mut b = vec![0u8; p_rows * oc * 2];
                hip.memcpy_dtoh(&mut b, &y).map_err(err)?;
                hip.free(y).map_err(err)?;
                Ok(le_u16s(&b))
            };
            let got = run_one(&f)?;
            let got_neg = run_one(&fneg)?;
            hip.free(sd).map_err(err)?;
            hip.free(ld).map_err(err)?;
            let bad = got.iter().zip(&want).filter(|(g, w)| g != w).count();
            let first = got.iter().zip(&want).enumerate().find(|(_, (g, w))| g != w).map(|(i, (g, w))| json!({"p": i / oc, "col": i % oc, "got": format!("{g:#06x}"), "want": format!("{w:#06x}")}));
            let nbad = got_neg.iter().zip(&want).filter(|(g, w)| g != w).count() as u64;
            neg_bad[ki] += nbad;
            let pass = bad == 0 && missing == 0;
            all_pass &= pass;
            cases.push(json!({"kind": tag, "pass_name": pass_name, "class": format!("{class:?}"), "grouped_rows": p_rows, "outputs_checked": got.len(),
                "bf16_mismatches": bad, "first": first, "hook_missing_keys": missing, "hook_missing_samples": missing_samples,
                "wrongsel_mismatches": nbad, "pass": pass}));
        }
    }
    let neg: Vec<Value> = neg_names
        .iter()
        .zip(neg_bad)
        .map(|(n, b)| json!({"negative": n, "mismatches": b, "detected": b > 0}))
        .collect();
    let neg_ok = neg_bad.iter().all(|&b| b > 0);
    Ok(
        json!({"module_sha256": sha256_hex(co), "cases": cases, "negative_controls": neg, "pass": all_pass && neg_ok,
        "note": "standalone epilogue fixture; expected values from the gfx1100 epilogue adapter + qualified hooks, never from the fixture"}),
    )
}

fn mq_i4_reference_epilogue(
    kind: Kind,
    m: usize,
    sums: &[f32],
    slots: &[i32],
    hooks: &QualifiedHooks,
) -> Vec<u16> {
    epilogue::grouped_prefill_epilogue(kind.decode_kind(), m, sums, slots, hooks)
}

/// `--cpu-recheck-fixture OUT.json` (CPU only, no HIP): every GateUp pass of the standalone fixture (the exact
/// `fixture_values` / seeds / slots / pass list of `run_fixture`) through the adapter's trace, comparing per live cell
///   old  = bf16 of the historical SSE product (`mulss` Up-dest, `q` src; old binary function 0x224d80, instruction 0x22569c),
///   twin = the public twin's own output with the same hooks,
///   new  = the adapter's output (reviewed MulUp rule).
/// The historical GPU vectors were not retained: the receipt carries the historical 959 only as comparison metadata.
#[cfg(target_arch = "x86_64")]
fn run_cpu_recheck_fixture(a: &Args) -> Res<i32> {
    let out = a
        .cpu_recheck_fixture
        .as_ref()
        .ok_or("--cpu-recheck-fixture needs OUTJSON")?;
    for (flag, given) in [
        ("--self-check", a.self_check),
        ("--only", a.only.is_some()),
        ("--collect-hook-keys", a.collect_hook_keys.is_some()),
        ("--codeobject", a.codeobject.is_some()),
        ("--codec", a.codec.is_some()),
        ("--mq4-anchor", a.anchor.is_some()),
        ("--sample-dir", a.sample_dir.is_some()),
        ("--chain-dir", a.chain_dir.is_some()),
        ("--prepare-chain", a.prepare_chain.is_some()),
        ("--out", a.out.is_some()),
        ("--epilogue-codeobject", a.epilogue_fixture.is_some()),
        ("--mixed-codeobject", !a.mixed.is_empty()),
        ("--negative-codeobject", !a.negatives.is_empty()),
        ("--card", a.card.is_some()),
        ("--expect-uuid", a.expect_uuid.is_some()),
        ("--expect-bdf", a.expect_bdf.is_some()),
        ("--expect-arch", a.expect_arch.is_some()),
        ("--census-sample-dir", !a.census_samples.is_empty()),
        ("--census-chain-dir", !a.census_chains.is_empty()),
    ] {
        if given {
            return Err(format!(
                "--cpu-recheck-fixture is the CPU-only fixture recheck; {flag} is not accepted"
            ));
        }
    }
    let (hooks, meta, problems) = load_table(a)?;
    if !problems.is_empty() {
        return Err(format!("qualified tables rejected: {problems:?}"));
    }
    const HIST_TOTAL: u64 = 959;
    const HIST_Q_NAN_UP_NAN_PAYLOAD: u64 = 571;
    const HIST_Q_FFC00000: u64 = 388;
    let finite = finite_bf16_list();
    let kind = Kind::GateUp;
    let (m, oc) = (kind.m(), kind.out_cols());
    let p_rows = fixture_p_rows(kind);
    let (mut passes, mut problems_run) = (Vec::new(), Vec::new());
    let (
        mut tot_cells,
        mut tot_delta,
        mut tot_a,
        mut tot_b,
        mut tot_c,
        mut tot_nonnan,
        mut tot_twin_vs_hist,
        mut tot_missing,
    ) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut tot_stats = epilogue::MulUpStats::default();
    let mut first_overall: Option<Value> = None;
    let bf16 = |bits: u32| {
        (hipfire_runtime::mq_i4_reference::bf16_rne_finite_passthrough_bits(bits) >> 16) as u16
    };
    for (pi, pass) in FIXTURE_PASSES.iter().enumerate() {
        let sums = fixture_values(kind, pass, p_rows, a.seed ^ (pi as u32 * 977), &finite);
        let slots = fixture_slots(p_rows);
        hooks.clear_missing();
        let run =
            epilogue::grouped_prefill_epilogue_trace(kind.decode_kind(), m, &sums, &slots, &hooks);
        let (missing, missing_samples) = (hooks.missing_count(), hooks.missing_samples());
        let twin = run
            .twin
            .as_ref()
            .ok_or("trace run lost the public twin output")?;
        let (mut delta, mut ca, mut cb, mut cc, mut nonnan, mut twin_vs_hist) =
            (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        let mut first: Option<Value> = None;
        for c in &run.cells {
            let idx = c.row * oc + c.col;
            let hist_bits = epilogue::historical_mulss_up_dest_bits(c.q, c.u);
            let old = bf16(hist_bits);
            let (new, tw) = (run.out[idx], twin[idx]);
            if tw != old {
                twin_vs_hist += 1;
            }
            if new == old {
                continue;
            }
            delta += 1;
            let (q_nan, u_nan) = (f32::from_bits(c.q).is_nan(), f32::from_bits(c.u).is_nan());
            if !f32::from_bits(hist_bits).is_nan()
                && !(f32::from_bits(c.q) * f32::from_bits(c.u)).is_nan()
            {
                nonnan += 1;
            }
            let gate_raw = sums[c.row * m + c.col].to_bits();
            let gate = if gate_raw & 0x7f80_0000 == 0x7f80_0000 {
                gate_raw
            } else {
                u32::from(bf16(gate_raw)) << 16
            };
            if gate & 0x7fff_ffff > 0x7f80_0000 && q_nan && u_nan {
                ca += 1;
            } else if gate == 0xff80_0000 && q_nan && u_nan {
                cb += 1;
            } else {
                cc += 1;
            }
            if first.is_none() {
                first = Some(
                    json!({"p": c.row, "col": c.col, "q": format!("{:#010x}", c.q), "up": format!("{:#010x}", c.u), "old_sse": format!("{old:#06x}"),
                    "public_twin": format!("{tw:#06x}"), "new": format!("{new:#06x}"), "q_nan": q_nan, "up_nan": u_nan}),
                );
            }
        }
        tot_cells += run.cells.len() as u64;
        tot_delta += delta;
        tot_a += ca;
        tot_b += cb;
        tot_c += cc;
        tot_nonnan += nonnan;
        tot_twin_vs_hist += twin_vs_hist;
        tot_missing += missing;
        tot_stats += run.stats;
        if first_overall.is_none() {
            first_overall = first.clone().map(|mut f| {
                f["pass_name"] = json!(pass);
                f
            });
        }
        if missing > 0 {
            problems_run.push(format!(
                "{pass}: {missing} unresolved probe-table keys, e.g. {missing_samples:?}"
            ));
        }
        eprintln!("[cpu-recheck] gate_up {pass} cells={} delta_vs_historical={delta} (gate_nan&up_nan {ca}, rounded_gate=-inf&up_nan {cb}, other {cc}) nonnan_delta={nonnan}", run.cells.len());
        passes.push(json!({"pass_name": pass, "grouped_rows": p_rows, "live_cells": run.cells.len(), "mul_up_audit": run.stats.to_json(),
            "delta_vs_historical_sse": delta, "delta_gate_nan_up_nan_payload": ca, "delta_gate_rounded_neg_inf_up_nan": cb, "delta_other": cc,
            "nonnan_product_delta": nonnan, "public_twin_vs_historical_sse": twin_vs_hist, "hook_missing_keys": missing,
            "hook_missing_samples": missing_samples, "first_delta": first}));
    }
    if a.seed == 1
        && (tot_delta != HIST_TOTAL
            || tot_a != HIST_Q_NAN_UP_NAN_PAYLOAD
            || tot_b != HIST_Q_FFC00000
            || tot_c != 0)
    {
        problems_run.push(format!("j587 seed-1 delta classification differs: total={tot_delta}, gate_nan={tot_a}, rounded_gate_neg_inf={tot_b}, other={tot_c}"));
    }
    let pass = problems_run.is_empty() && tot_nonnan == 0 && tot_stats.nonnan_changed == 0;
    let doc = json!({"schema": "consumer24.gfx1100.mulup-cpu-recheck.v1", "mode": "cpu-recheck-fixture", "scope": "cpu-only; standalone GateUp fixture, every pass, same fixture_values/seeds/slots as the device run; Down has no MulUp and is unchanged",
        "command": format!("qwen4_moe_sym_gfx1100_g1_oracle {}", a.argv.join(" ")), "source_sha256": source_hashes(), "binary_sha256": exe_sha(), "seed": a.seed,
        "probe_table": meta, "rule": "q NaN -> q|0x00400000; else up NaN -> up|0x00400000; else 0*inf/inf*0 -> 0xffc00000 (device pin rcpt/b4.mulup-pin.json); else the public twin's output",
        "historical_model": {"old_binary_sha256": "0ef743fe7790e46faa2b6229a5b15c1f055c9492c493d71352ccc5c18352e9db",
            "op": "mulss -0x4(%rax),%xmm0 at function 0x224d80, instruction 0x22569c: destination = Up, source = q (frozen here with inline asm)",
            "note": "the historical GPU vectors were NOT retained; this recheck replays the historical CPU expectation from the actual q/up of the same fixture"},
        "passes": passes,
        "totals": {"live_cells": tot_cells, "delta_vs_historical_sse": tot_delta, "delta_gate_nan_up_nan_payload": tot_a, "delta_gate_rounded_neg_inf_up_nan": tot_b, "delta_other": tot_c,
            "nonnan_product_delta": tot_nonnan, "public_twin_vs_historical_sse": tot_twin_vs_hist, "hook_missing_keys": tot_missing, "mul_up_audit": tot_stats.to_json()},
        "first_delta": first_overall,
        "historical_receipt_comparison": {"metadata_only": true, "old_receipt_mismatches": HIST_TOTAL, "old_receipt_gate_nan_up_nan_payload": HIST_Q_NAN_UP_NAN_PAYLOAD,
            "old_receipt_gate_neg_inf_q_ffc00000": HIST_Q_FFC00000,
            "cpu_delta_total_equals_old_receipt": tot_delta == HIST_TOTAL, "cpu_delta_gate_nan_up_nan_equals_571": tot_a == HIST_Q_NAN_UP_NAN_PAYLOAD,
            "cpu_delta_gate_rounded_neg_inf_up_nan_equals_388": tot_b == HIST_Q_FFC00000,
            "old_receipt_j587": {"artifact": "artifacts/b4-nan-pair/j587.synthetic-mq4.json", "gate_up_exception_mismatches": HIST_TOTAL,
                "gate_up_finite_random_ties_mismatches": 0, "down_mismatches": 0, "wrongsel_gate_up_total": 29367, "wrongsel_gate_up_exception": 8003, "wrongsel_down": 138621,
                "wrongsel_gate_up_total_lower_bound_if_only_the_cpu_expected_delta_changes": 29367u64.saturating_sub(tot_delta),
                "wrongsel_gate_up_exception_lower_bound_if_only_the_cpu_expected_delta_changes": 8003u64.saturating_sub(tot_delta),
                "bound_note": "logical bound from the old raw counts (changing at most `delta` expected cells can change at most `delta` wrongsel mismatches); NOT an exact new GPU negative count"},
            "label": "CPU expected-delta, no historical GPU vector",
            "label_explanation": "these counts are the CPU expected-output delta against the historical SSE expectation; the historical GPU vectors were not retained, so no exact new GPU count is claimed",
            "note": "comparison against the old receipt's counts only; the old GPU vectors are not retained, so this is not a retrospective GPU claim"},
        "problems": problems_run, "pass": pass});
    write_json(out, &doc)?;
    println!(
        "{}",
        json!({"out": out, "pass": pass, "live_cells": tot_cells, "delta_vs_historical_sse": tot_delta, "delta_q_nan_and_up_nan_payload": tot_a,
        "delta_q_ffc00000": tot_b, "delta_other": tot_c, "nonnan_product_delta": tot_nonnan, "public_twin_vs_historical_sse": tot_twin_vs_hist, "mul_up_audit": tot_stats.to_json(),
        "first_delta": first_overall})
    );
    Ok(if pass { 0 } else { 2 })
}

#[cfg(not(target_arch = "x86_64"))]
fn run_cpu_recheck_fixture(_a: &Args) -> Res<i32> {
    Err(
        "--cpu-recheck-fixture replays the historical x86 SSE mulss and needs an x86_64 host"
            .into(),
    )
}

fn denorm_audit(images: &[(&str, &[u8], Vec<String>)]) -> Value {
    let mut entries = Vec::new();
    let mut ok = true;
    for (label, img, names) in images {
        for n in names {
            match kernel_denorm(img, n) {
                Ok((f32m, f16m)) => {
                    let good = f32m == 3 && f16m == 3;
                    ok &= good;
                    entries.push(json!({"object": label, "kernel": n, "fp32_denorm_mode": f32m, "fp16_64_denorm_mode": f16m, "ok": good}));
                }
                Err(e) => {
                    ok = false;
                    entries.push(json!({"object": label, "kernel": n, "error": e, "ok": false}));
                }
            }
        }
    }
    json!({"required": "FLOAT_DENORM_MODE_32 = 3 and FLOAT_DENORM_MODE_16_64 = 3 (kernel descriptor COMPUTE_PGM_RSRC1)", "entries": entries, "ok": ok})
}

fn run_device(a: &Args) -> Res<bool> {
    let out = a.out.clone().ok_or("--out is required")?;
    let mixed = !a.mixed.is_empty();
    let mixed_path = |k: &str| a.mixed.iter().find(|(x, _)| x == k).map(|(_, p)| p.clone());
    // `codec` is the run's candidate codec; in mixed mode it is the cold codec (w2 or w3) and `hot_path` the MQ4 module.
    let (codec, co_path, hot_path): (Codec, String, Option<String>) = if mixed {
        if a.codeobject.is_some() || a.codec.is_some() {
            return Err("mixed mode replaces --codeobject/--codec (use --mixed-codeobject mq4=PATH and w2=PATH or w3=PATH)".into());
        }
        if !a.negatives.is_empty() {
            return Err("--negative-codeobject belongs to per-codec runs, not mixed mode".into());
        }
        if a.sample_dir.is_none() {
            return Err(
                "mixed mode needs --sample-dir with both the MQ4 hot and the W2/W3 cold experts"
                    .into(),
            );
        }
        let hot = mixed_path("mq4").ok_or("--mixed-codeobject mq4=PATH is required")?;
        match (mixed_path("w2"), mixed_path("w3"), mixed_path("e8p")) {
            (Some(p), None, None) => (Codec::W2, p, Some(hot)),
            (None, Some(p), None) => (Codec::W3, p, Some(hot)),
            (None, None, Some(p)) => (Codec::E8p, p, Some(hot)),
            (None, None, None) => {
                return Err(
                    "mixed mode needs --mixed-codeobject w2=PATH, w3=PATH or e8p=PATH".into(),
                )
            }
            _ => {
                return Err(
                    "give exactly one of w2= / w3= / e8p= (each is its own mixed run)".into(),
                )
            }
        }
    } else {
        (
            Codec::parse(a.codec.as_deref().ok_or("--codec is required")?)?,
            a.codeobject.clone().ok_or("--codeobject is required")?,
            None,
        )
    };
    let card = a.card.clone().ok_or("--card is required")?;
    let target = a.target_or_default();
    if card != target.card() {
        return Err(format!("--card must be {}, got {card:?}", target.card()));
    }
    let ident = match target {
        Target::Gfx1100 => {
            let expect_uuid = a.expect_uuid.clone().ok_or("--expect-uuid is required")?;
            let expect_bdf = a.expect_bdf.clone().ok_or("--expect-bdf is required")?;
            if expect_uuid != XTX_UUID || expect_bdf != XTX_BDF {
                return Err(format!(
                    "xtx identity is {XTX_UUID} / {XTX_BDF}; got {expect_uuid} / {expect_bdf}"
                ));
            }
            Ident {
                target,
                uuid: Some(expect_uuid),
                bdf: Some(expect_bdf),
            }
        }
        Target::Gfx1151 => Ident {
            target,
            uuid: a.expect_uuid.clone(),
            bdf: Some(a.expect_bdf.clone().ok_or(
                "gfx1151 has no ROCr UUID: --expect-bdf is required (--expect-uuid is optional)",
            )?),
        },
    };
    if !mixed && matches!(codec, Codec::W2 | Codec::W3) && a.anchor.is_none() {
        return Err("--mq4-anchor is required for w2/w3 (the same-device QT44/QT53 anchor)".into());
    }
    if codec == Codec::E8p && a.anchor.is_some() {
        return Err("--mq4-anchor does not apply to e8p: E8P codewords (|v| up to 11) have no QT44/QT53 anchor encoding, the CPU twin is the only expectation".into());
    }
    for (n, _) in &a.negatives {
        if !codec.negative_names().contains(&n.as_str()) {
            return Err(format!(
                "negative control {n:?} does not apply to codec {}; expected one of {:?}",
                codec.name(),
                codec.negative_names()
            ));
        }
    }
    for v in ["HSA_CU_MASK", "ROC_GLOBAL_CU_MASK"] {
        if std::env::var_os(v).is_some() {
            return Err(format!(
                "{v} is set; CU masks are banned on RDNA devices (they hung the MES)"
            ));
        }
    }
    let hip = HipRuntime::load().map_err(err)?;
    let id = hip.current_device().map_err(err)?;
    let uuid = hip.device_uuid(id).map_err(err)?;
    let bdf = hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase();
    let arch = hip.get_arch(id).map_err(err)?;
    let identity_match = arch == ident.arch()
        && ident
            .uuid
            .as_deref()
            .is_none_or(|u| uuid.eq_ignore_ascii_case(u))
        && ident.bdf.as_deref().is_none_or(|b| bdf == b);
    let device = json!({"card": card, "uuid": uuid, "bdf": bdf, "arch": arch, "expected_arch": ident.arch(), "expected_uuid": ident.uuid, "expected_bdf": ident.bdf,
        "uuid_authoritative": target == Target::Gfx1100,
        "identity_match": identity_match, "device_id": id});
    if !identity_match {
        return Err(format!("device identity mismatch: {device}"));
    }
    let driver = json!({"hip_runtime": hip.runtime_version().ok().map(|(a, b)| format!("{a}.{b}")),
        "amdgpu_module_version": std::fs::read_to_string("/sys/module/amdgpu/version").ok().map(|s| s.trim().to_string()),
        "kernel_osrelease": std::fs::read_to_string("/proc/sys/kernel/osrelease").ok().map(|s| s.trim().to_string()),
        "cu_mask_env": "HSA_CU_MASK / ROC_GLOBAL_CU_MASK verified unset"});

    let set = match &a.sample_dir {
        Some(d) => Some(samples::load(std::path::Path::new(d), codec)?),
        None => None,
    };
    let hot_set = match (&a.sample_dir, mixed) {
        (Some(d), true) => Some(samples::load(std::path::Path::new(d), Codec::Mq4)?),
        _ => None,
    };
    let provisional = set.iter().chain(hot_set.iter()).any(|s| s.provisional);
    let artifact_bound = set.is_some()
        && set
            .iter()
            .chain(hot_set.iter())
            .all(|s| !s.provisional && s.artifact_sha256.is_some());
    let mut down_lookup: HashMap<(u64, u64, Codec), RealRef> = HashMap::new();
    for s in set.iter().chain(hot_set.iter()) {
        for r in &s.down {
            down_lookup.insert((r.layer, r.expert, r.codec), r.clone());
        }
    }
    let scope = if set.is_some() {
        SCOPE_REAL
    } else {
        "synthetic"
    };
    let mut head = head_common(a, "device", scope);
    head["device"] = device;
    head["driver"] = driver;
    head["codec"] = if mixed {
        json!(format!("mixed:mq4+{}", codec.name()))
    } else {
        json!(codec.name())
    };
    head["mixed_codec_run"] = json!(mixed);
    head["profile"] = json!({"gate_up": format!("{:?}/R256 qt{}", codec, codec.qt(Kind::GateUp)), "down": format!("{:?}/R128 qt{}", codec, codec.qt(Kind::Down)),
        "entries": {"gate_up": Kind::GateUp.entry(codec, false), "down": Kind::Down.entry(codec, false)},
        "hot_entries": if mixed { json!({"gate_up": Kind::GateUp.entry(Codec::Mq4, false), "down": Kind::Down.entry(Codec::Mq4, false)}) } else { Value::Null }});
    head["arms"] = if mixed {
        json!(MIXED_ARMS
            .iter()
            .map(|(c, h)| mixed_arm_label(*c, *h))
            .collect::<Vec<_>>())
    } else {
        json!(Arm::ALL.iter().map(|a| a.name()).collect::<Vec<_>>())
    };

    let co = read_file(&co_path)?;
    let hot_bytes = match &hot_path {
        Some(p) => Some((p.clone(), read_file(p)?)),
        None => None,
    };
    let anchor_bytes = match &a.anchor {
        Some(p) => Some((p.clone(), read_file(p)?)),
        None => None,
    };
    let mut neg_bytes = Vec::new();
    for (n, p) in &a.negatives {
        neg_bytes.push((n.clone(), p.clone(), read_file(p)?));
    }
    let fixture_bytes = match &a.epilogue_fixture {
        Some(p) => Some((p.clone(), read_file(p)?)),
        None => None,
    };
    let set_json = |s: &Option<samples::SampleSet>| {
        s.as_ref().map(|s| json!({"path": a.sample_dir, "manifest_sha256": s.manifest_sha256, "provisional": s.provisional, "artifact_sha256": s.artifact_sha256, "meta": s.meta}))
    };
    head["inputs"] = json!({
        "candidate_codeobject": {"path": co_path, "sha256": sha256_hex(&co), "bytes": co.len(), "role": if mixed { "mixed cold-codec module" } else { "per-codec candidate" }},
        "mixed_mq4_codeobject": hot_bytes.as_ref().map(|(p, b)| json!({"path": p, "sha256": sha256_hex(b), "bytes": b.len()})),
        "mq4_anchor_codeobject": anchor_bytes.as_ref().map(|(p, b)| json!({"path": p, "sha256": sha256_hex(b), "bytes": b.len()})),
        "negative_codeobjects": neg_bytes.iter().map(|(n, p, b)| json!({"name": n, "path": p, "sha256": sha256_hex(b), "bytes": b.len()})).collect::<Vec<_>>(),
        "epilogue_fixture_codeobject": fixture_bytes.as_ref().map(|(p, b)| json!({"path": p, "sha256": sha256_hex(b), "bytes": b.len()})),
        "sample_dir": set_json(&set),
        "sample_dir_hot_companion": set_json(&hot_set),
    });

    let (hooks, table_meta, problems) = load_table(a)?;
    head["probe_table"] = table_meta.clone();
    head["intrinsic_tables_sha256"] = table_meta["intrinsic_tables_sha256"].clone();
    head["hooks"] =
        json!({"kind": "QualifiedHooks (exact-key probe observations only; no portable fallback)"});
    let mut problems = problems;
    if let Some(d) = table_meta.get("device") {
        if !d.is_null() {
            if d.get("arch").and_then(Value::as_str) != Some(ident.arch()) {
                problems.push(format!(
                    "probe table arch {:?} is not {}",
                    d.get("arch"),
                    ident.arch()
                ));
            }
            if let Some(eu) = &ident.uuid {
                if d.get("uuid")
                    .and_then(Value::as_str)
                    .is_some_and(|u| !u.eq_ignore_ascii_case(eu))
                {
                    problems.push(format!(
                        "probe table was observed on {:?}, not {eu}",
                        d.get("uuid")
                    ));
                }
            }
            if d.get("identity_match").and_then(Value::as_bool) == Some(false) {
                problems.push("probe table device.identity_match is false".into());
            }
        }
    }
    let exp_root = RootExpect {
        source_sha256: source_hashes(),
        probe_table_sha256: table_meta
            .get("sha256")
            .and_then(Value::as_str)
            .map(str::to_string),
        divide_table_sha256: table_binding(&table_meta).0,
        intrinsic_tables_sha256: table_binding(&table_meta).1,
        hooks: HOOKS_QUALIFIED,
        sample_manifests: set
            .iter()
            .map(|s| (codec.name().to_string(), s.manifest_sha256.clone()))
            .chain(
                hot_set
                    .iter()
                    .map(|s| ("mq4".to_string(), s.manifest_sha256.clone())),
            )
            .collect(),
        seed: a.seed,
        oracle_binary_sha256: exe_sha(),
    };
    let chain_dir: Option<ChainDir> = match &a.chain_dir {
        None => {
            head["chain"] = json!({"dir": Value::Null, "status": "not provided: real-sample Down chain scenarios are reported as failures"});
            None
        }
        Some(d) => match ChainDir::load(std::path::Path::new(d)) {
            Ok(c) => {
                problems.extend(c.root_problems(&exp_root));
                head["chain"] = c.root_info(&exp_root);
                Some(c)
            }
            Err(e) => {
                problems.push(e);
                None
            }
        },
    };
    if !problems.is_empty() {
        head["status"] = json!("aborted");
        head["problems"] = json!(problems);
        head["pass"] = json!(false);
        write_json(&out, &head)?;
        return Err(format!("probe table / twin rejected: {problems:?}"));
    }

    // modules + denorm audit
    let m_cand: Module = hip.module_load_data(&co).map_err(err)?;
    let m_hot: Option<Module> = match &hot_bytes {
        Some((_, b)) => Some(hip.module_load_data(b).map_err(err)?),
        None => None,
    };
    let m_anchor: Option<Module> = match &anchor_bytes {
        Some((_, b)) => Some(hip.module_load_data(b).map_err(err)?),
        None => None,
    };
    let mut m_negs: Vec<(String, Module)> = Vec::new();
    for (n, _, b) in &neg_bytes {
        m_negs.push((n.clone(), hip.module_load_data(b).map_err(err)?));
    }
    let mut audit_in: Vec<(&str, &[u8], Vec<String>)> = Vec::new();
    let cand_names: Vec<String> = [Kind::GateUp, Kind::Down]
        .iter()
        .flat_map(|k| [k.entry(codec, false), k.entry(codec, true)])
        .collect();
    audit_in.push((
        if mixed { "mixed_cold" } else { "candidate" },
        co.as_slice(),
        cand_names,
    ));
    if let Some((_, b)) = &hot_bytes {
        audit_in.push((
            "mixed_mq4",
            b.as_slice(),
            [Kind::GateUp, Kind::Down]
                .iter()
                .flat_map(|k| [k.entry(Codec::Mq4, false), k.entry(Codec::Mq4, true)])
                .collect(),
        ));
    }
    if let Some((_, b)) = &anchor_bytes {
        audit_in.push((
            "mq4_anchor",
            b.as_slice(),
            [Kind::GateUp, Kind::Down]
                .iter()
                .map(|k| k.entry(Codec::Mq4, false))
                .collect(),
        ));
    }
    for (n, _, b) in &neg_bytes {
        audit_in.push((
            n.as_str(),
            b.as_slice(),
            [Kind::GateUp, Kind::Down]
                .iter()
                .flat_map(|k| [k.entry(codec, false), k.entry(codec, true)])
                .collect(),
        ));
    }
    if let Some((_, b)) = &fixture_bytes {
        let mut v = Vec::new();
        for t in ["gate_up", "down"] {
            v.push(format!(
                "qwen4_moe_epilogue_fixture_{t}_{}",
                model::arch_name()
            ));
            v.push(format!(
                "qwen4_moe_epilogue_fixture_{t}_wrongsel_{}",
                model::arch_name()
            ));
        }
        audit_in.push(("epilogue_fixture", b.as_slice(), v));
    }
    head["denorm"] = denorm_audit(&audit_in);

    let mut ctx = Ctx {
        hip: &hip,
        dev: id,
        codec,
        hooks: &hooks,
        head,
        out: out.clone(),
        results: Vec::new(),
        tally: Tally::default(),
        sentinel_arm_notes: Vec::new(),
        extra: BTreeMap::new(),
        required_negs: if mixed {
            Vec::new()
        } else {
            codec
                .negative_names()
                .iter()
                .map(|s| s.to_string())
                .collect()
        },
        mixed,
        real_samples: set.is_some(),
        provisional,
        artifact_bound,
        down_lookup,
        chain_items: Vec::new(),
        chain_notes: Vec::new(),
        chain_dir,
        scenarios_since_flush: 0,
    };
    ctx.flush("in_progress")?;

    if let Some((_, b)) = &fixture_bytes {
        ctx.progress(Kind::GateUp, "epilogue_fixture", "fixture");
        let r = run_fixture(&hip, b, &hooks, a.seed)?;
        ctx.extra.insert("epilogue_fixture".into(), r);
        ctx.flush("in_progress")?;
    }

    for kind in [Kind::GateUp, Kind::Down] {
        let get = |m: &Module, n: &str| hip.module_get_function(m, n).map_err(err);
        let mut list: Vec<Scenario> = if mixed {
            match (kind, &set, &hot_set) {
                (Kind::GateUp, Some(cs), Some(hs)) => {
                    let mut v = scenarios::mixed(&cs.gate_up, &hs.gate_up, a.seed);
                    if let Some(f) = &a.only {
                        v.retain(|s| s.name.contains(f.as_str()));
                    }
                    v
                }
                _ => Vec::new(),
            }
        } else {
            scenario_list(a, codec, kind, &set)
        };
        // Down chain items were produced while the GateUp scenarios ran
        let chain: Vec<Scenario> = ctx.chain_items.drain(..).collect();
        list.extend(chain);
        if mixed {
            let hot_m = m_hot
                .as_ref()
                .ok_or("internal: mixed run without the mq4 module")?;
            let (anchor_m, anchor_note) = match &m_anchor {
                Some(m) => (m, "--mq4-anchor module"),
                None => (hot_m, "mixed mq4 module (no --mq4-anchor given: the anchor union shares the hot kernel)"),
            };
            let f = MixedFns {
                hot: (
                    get(hot_m, &kind.entry(Codec::Mq4, true))?,
                    get(hot_m, &kind.entry(Codec::Mq4, false))?,
                ),
                cold: (
                    get(&m_cand, &kind.entry(codec, true))?,
                    get(&m_cand, &kind.entry(codec, false))?,
                ),
                cold_codec: codec,
                anchor_acc: if codec == Codec::E8p {
                    None
                } else {
                    get(anchor_m, &kind.entry(Codec::Mq4, true)).ok()
                },
                anchor_h: if codec == Codec::E8p {
                    None
                } else {
                    Some(get(anchor_m, &kind.entry(Codec::Mq4, false))?)
                },
                anchor_note,
            };
            for sc in list {
                ctx.scenario_mixed(kind, &f, &sc)?;
            }
        } else {
            let f = KindFns {
                acc: get(&m_cand, &kind.entry(codec, true))?,
                h: get(&m_cand, &kind.entry(codec, false))?,
                anchor_acc: m_anchor
                    .as_ref()
                    .and_then(|m| get(m, &kind.entry(Codec::Mq4, true)).ok()),
                anchor_h: match &m_anchor {
                    Some(m) => Some(get(m, &kind.entry(Codec::Mq4, false))?),
                    None => None,
                },
                negs: m_negs
                    .iter()
                    .map(|(n, m)| {
                        let both = (|| -> Res<(Function, Function)> {
                            Ok((
                                get(m, &kind.entry(codec, true))?,
                                get(m, &kind.entry(codec, false))?,
                            ))
                        })();
                        (n.clone(), both)
                    })
                    .collect(),
            };
            for sc in list {
                ctx.scenario(kind, &f, &sc)?;
            }
        }
        ctx.flush("in_progress")?;
    }
    ctx.flush("complete")?;
    let d = ctx.doc("complete");
    let _ = std::fs::remove_file(format!("{out}.progress"));
    println!(
        "{}",
        json!({"schema": schema_of(a), "mode": "device", "out": out, "scope": scope, "mixed_codec_run": mixed, "pass": d["pass"], "g1c_verdict": d["g1c"]["verdict"],
        "g1c_measured_pass": d["g1c"]["measured_pass"], "g1c_closed": d["g1c"]["closed"], "artifact_binding": d["g1c"]["artifact_binding"]["status"],
        "g1c_blockers": d["g1c"]["blockers"], "summary": d["summary"], "denorm_ok": d["denorm"]["ok"],
        "negative_controls": d["negative_controls"], "coverage_gaps": d["coverage_gaps"], "epilogue_fixture_pass": d.get("epilogue_fixture").map(|f| f["pass"].clone())})
    );
    Ok(d["pass"] == json!(true))
}

fn exe_sha() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .map(|b| sha256_hex(&b))
        .unwrap_or_default()
}

/// `--prepare-chain DIR` (CPU only): for every chain-source GateUp scenario (per-codec and, when the sample directory
/// holds the MQ4 hot companion, mixed) write the CPU-expected grouped BF16 H and `DIR/chain-h.json`.
fn prepare_with<I: EpilogueIntrinsics>(
    a: &Args,
    hooks: &I,
    hooks_kind: &'static str,
    probe_sha: Option<String>,
    bind: (Option<String>, Vec<String>),
    miss: &dyn Fn() -> (u64, Vec<String>),
    clear: &dyn Fn(),
) -> Res<bool> {
    let dir_s = a
        .prepare_chain
        .as_ref()
        .ok_or("--prepare-chain needs DIR")?;
    let out = a.out.as_ref().ok_or("--out is required")?;
    if a.chain_dir.is_some() || !a.mixed.is_empty() {
        return Err(
            "--prepare-chain is CPU-only and takes neither --chain-dir nor --mixed-codeobject"
                .into(),
        );
    }
    let codec = Codec::parse(a.codec.as_deref().ok_or("--codec is required")?)?;
    let sd = a.sample_dir.as_ref().ok_or("--sample-dir is required")?;
    let dir = std::path::Path::new(dir_s);
    std::fs::create_dir_all(dir).map_err(|e| format!("{dir_s}: {e}"))?;
    let set_o = Some(samples::load(std::path::Path::new(sd), codec)?);
    let set = set_o.as_ref().ok_or("internal: sample set")?;
    let hot_set = match codec {
        Codec::Mq4 => None,
        _ => samples::load(std::path::Path::new(sd), Codec::Mq4).ok(),
    };
    let mut downs: HashMap<(u64, u64, Codec), RealRef> = HashMap::new();
    for s in std::iter::once(set).chain(hot_set.iter()) {
        for r in &s.down {
            downs.insert((r.layer, r.expert, r.codec), r.clone());
        }
    }
    let mut jobs: Vec<(&'static str, Scenario)> = scenario_list(a, codec, Kind::GateUp, &set_o)
        .into_iter()
        .filter(|s| s.feeds_chain && !s.production)
        .map(|s| ("percodec", s))
        .collect();
    if let Some(hs) = &hot_set {
        let mut v = scenarios::mixed(&set.gate_up, &hs.gate_up, a.seed);
        if let Some(f) = &a.only {
            v.retain(|s| s.name.contains(f.as_str()));
        }
        jobs.extend(
            v.into_iter()
                .filter(|s| s.feeds_chain && !s.production)
                .map(|s| ("mixed", s)),
        );
    }
    let (mut entries, mut problems, mut names) = (Vec::new(), Vec::new(), Vec::new());
    let kind = Kind::GateUp;
    for (mode, sc) in &jobs {
        let experts = sc.build_experts(codec, kind)?;
        let x = sc.x_set(kind);
        let expert_of = |e: i32| e as usize;
        let pairs = needed_pairs(&sc.tiles, sc.div, &expert_of);
        let dense = build_dense(kind, &experts, &x, &pairs, &|k, e, r, x| {
            fold_twin(k, e, r, x)
        });
        clear();
        let exp = assemble(kind, &sc.tiles, sc.div, &dense, &expert_of, hooks, false);
        let (missing, samples_missing) = miss();
        if missing > 0 {
            problems.push(format!(
                "{}: {missing} unresolved probe-table keys, e.g. {samples_missing:?}",
                sc.name
            ));
            continue;
        }
        let live = chain::live_rows(&exp.row_class);
        if live.iter().any(|&p| {
            exp.h[p * 640..(p + 1) * 640]
                .iter()
                .any(|v| v & 0x7f80 == 0x7f80)
        }) {
            problems.push(format!(
                "{}: a live GateUp H row is non-finite; the A4 helper rejects BF16 inf/NaN",
                sc.name
            ));
            continue;
        }
        let name = chain_name(mode, codec, &sc.name);
        let experts_json: Vec<Value> = sc
            .recipes
            .iter()
            .map(|r| match r {
                scenarios::Recipe::Real(rr) => json!({"layer": rr.layer, "expert": rr.expert, "codec": rr.codec.name(), "gate_up_sha256": rr.sha256,
                    "down_sha256": downs.get(&(rr.layer, rr.expert, rr.codec)).map(|d| d.sha256.clone())}),
                scenarios::Recipe::Synth(..) => Value::Null,
            })
            .collect();
        let e = PrepEntry {
            name: name.clone(),
            mode: *mode,
            codec,
            scenario: sc.name.clone(),
            rows: exp.m_total,
            live,
            h: exp.h.clone(),
            tiles_sha256: chain::tiles_sha(&sc.tiles),
            div: sc.div,
            x_seed: sc.x_seed,
            experts: Value::Array(experts_json),
        };
        entries.push(chain::write_entry(dir, &e)?);
        names.push(name);
        eprintln!("[prepare-chain] {} {}", mode, sc.name);
    }
    let mut manifests = serde_json::Map::new();
    manifests.insert(codec.name().to_string(), json!(set.manifest_sha256));
    if let Some(hs) = &hot_set {
        manifests.insert("mq4".to_string(), json!(hs.manifest_sha256));
    }
    let mut receipt = json!({"schema": "consumer24.gfx1100.chain-prep.v1", "dir": dir_s, "hooks": hooks_kind, "codec": codec.name(), "seed": a.seed,
        "command": format!("qwen4_moe_sym_gfx1100_g1_oracle {}", a.argv.join(" ")), "entries": names, "problems": problems,
        "next": format!("python3 crates/peacemaker-isa/tools/qwen4_moe_gfx1100_chain_a4.py --chain-dir {dir_s}")});
    let pass = problems.is_empty() && !entries.is_empty();
    if pass {
        let mut root = serde_json::Map::new();
        root.insert("oracle_binary_sha256".into(), json!(exe_sha()));
        root.insert("source_sha256".into(), source_hashes());
        root.insert("probe_table_sha256".into(), json!(probe_sha));
        root.insert("divide_table_sha256".into(), json!(bind.0));
        root.insert("intrinsic_tables_sha256".into(), json!(bind.1));
        root.insert("hooks".into(), json!(hooks_kind));
        root.insert("sample_manifest_sha256".into(), Value::Object(manifests));
        root.insert("seed".into(), json!(a.seed));
        root.insert(
            "mixed_mode".into(),
            json!(jobs.iter().any(|(m, _)| *m == "mixed")),
        );
        root.insert("codec".into(), json!(codec.name()));
        root.insert("scope".into(), json!(SCOPE_REAL));
        root.insert(
            "created_by".into(),
            json!("qwen4_moe_sym_gfx1100_g1_oracle --prepare-chain"),
        );
        receipt["chain_h_json_sha256"] = json!(chain::write_chain_h(dir, root, entries)?);
    } else if entries.is_empty() && problems.is_empty() {
        receipt["problems"] =
            json!(["no chain-source scenarios selected (check --only / the sample set)"]);
    }
    receipt["divide_table_sha256"] = json!(bind.0);
    receipt["intrinsic_tables_sha256"] = json!(bind.1);
    receipt["pass"] = json!(pass);
    write_json(out, &receipt)?;
    println!("{receipt}");
    Ok(pass)
}

/// `(divide table sha256, sorted intrinsic table sha256s)` recorded by `load_table`; pinned in the chain root.
fn table_binding(meta: &Value) -> (Option<String>, Vec<String>) {
    (
        meta["divide_table"]
            .get("sha256")
            .and_then(Value::as_str)
            .map(str::to_string),
        meta["intrinsic_tables_sha256"]
            .as_array()
            .map(|v| {
                v.iter()
                    .filter_map(|s| s.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
    )
}

fn run_prepare(a: &Args) -> Res<bool> {
    if a.self_check {
        let h = CpuOnlySelfCheckHooks::default();
        prepare_with(
            a,
            &h,
            HOOKS_CPU_ONLY,
            None,
            (None, Vec::new()),
            &|| (0, Vec::new()),
            &|| {},
        )
    } else {
        let (hooks, meta, problems) = load_table(a)?;
        if !problems.is_empty() {
            return Err(format!("probe table rejected: {problems:?}"));
        }
        let sha = meta
            .get("sha256")
            .and_then(Value::as_str)
            .map(str::to_string);
        prepare_with(
            a,
            &hooks,
            HOOKS_QUALIFIED,
            sha,
            table_binding(&meta),
            &|| (hooks.missing_count(), hooks.missing_samples()),
            &|| hooks.clear_missing(),
        )
    }
}

fn run() -> Res<i32> {
    let a = parse_args()?;
    let code = |ok: bool| if ok { 0 } else { 2 };
    if a.cpu_recheck_fixture.is_some() {
        run_cpu_recheck_fixture(&a)
    } else if a.collect_hook_keys.is_some() {
        run_census(&a)
    } else if a.prepare_chain.is_some() {
        run_prepare(&a).map(code)
    } else if a.self_check {
        run_self_check(&a).map(code)
    } else {
        run_device(&a).map(code)
    }
}

/// `--collect-hook-keys OUT.json` (CPU only): one process, every planned candidate-independent CPU case; see
/// `census.rs`. Requires the qualified base `--probe-table`; no `CpuOnlySelfCheckHooks`/portable math is involved.
fn run_census(a: &Args) -> Res<i32> {
    let out = a
        .collect_hook_keys
        .as_ref()
        .ok_or("--collect-hook-keys needs OUTJSON")?;
    for (flag, given) in [
        ("--self-check", a.self_check),
        ("--only", a.only.is_some()),
        ("--codeobject", a.codeobject.is_some()),
        ("--codec", a.codec.is_some()),
        ("--mq4-anchor", a.anchor.is_some()),
        ("--sample-dir", a.sample_dir.is_some()),
        ("--chain-dir", a.chain_dir.is_some()),
        ("--prepare-chain", a.prepare_chain.is_some()),
        ("--out", a.out.is_some()),
        ("--epilogue-codeobject", a.epilogue_fixture.is_some()),
        ("--mixed-codeobject", !a.mixed.is_empty()),
        ("--negative-codeobject", !a.negatives.is_empty()),
        ("--card", a.card.is_some()),
        ("--expect-uuid", a.expect_uuid.is_some()),
        ("--expect-bdf", a.expect_bdf.is_some()),
        ("--expect-arch", a.expect_arch.is_some()),
    ] {
        if given {
            return Err(format!("--collect-hook-keys is the CPU-only full census; {flag} is not accepted (scopes come from --census-sample-dir)"));
        }
    }
    let (hooks, meta, problems) = load_table(a)?;
    if !problems.is_empty() {
        return Err(format!("qualified tables rejected: {problems:?}"));
    }
    let head = json!({"schema": census::CENSUS_SCHEMA, "mode": "collect-hook-keys", "scope": "cpu-only; census of every planned candidate-independent G1 CPU epilogue key",
        "command": format!("qwen4_moe_sym_gfx1100_g1_oracle {}", a.argv.join(" ")), "source_sha256": source_hashes(), "binary_sha256": exe_sha(), "seed": a.seed,
        "hooks": {"kind": "CensusHooks over QualifiedHooks (exact-key qualified tables; typed MissingQuery suspension; no portable fallback, no CpuOnlySelfCheckHooks)"},
        "tables": {"probe_sha256": meta.get("sha256"), "divide_sha256": meta["divide_table"].get("sha256"), "intrinsic_sha256": meta.get("intrinsic_tables_sha256")},
        "probe_table": meta});
    let (divide_sha, intrinsic_sha) = table_binding(&meta);
    let probe_sha = meta
        .get("sha256")
        .and_then(Value::as_str)
        .map(str::to_string);
    census::run(census::Inputs {
        table: &hooks,
        seed: a.seed,
        sample_dirs: &a.census_samples,
        chains: &a.census_chains,
        probe_sha,
        divide_sha,
        intrinsic_sha,
        out,
        head,
    })
}

fn main() {
    match run() {
        Ok(0) => {}
        Ok(c) => std::process::exit(c),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
