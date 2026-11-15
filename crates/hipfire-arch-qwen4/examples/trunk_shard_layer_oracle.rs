// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Trunk-shard layer oracle (lab): ONE layer of the REAL Flash-Next weights, run unsharded
//! and as the TP2 trunk shard on the same injected HC streams, with every HC-stream element
//! checked against the documented contract.
//!
//! ```text
//! trunk_shard_layer_oracle <plan|selftest|run|compare> [options]      (see --help)
//!
//! run --arm ref  --model M --out DIR [--layer 0,3] [--rows 1,8,32] [--block mixer|ffn|both]
//!                [--input synthetic[:SEED]|real] [--repeat 2] [--device 0]
//! run --arm tp2  --model M --out DIR --input from:DIR/ref|synthetic[:SEED] --ref DIR/ref
//!                [--devices 0,1] [--no-qualify] [--ep-ar-wire bf16|f32]
//! compare --ref DIR/ref --tp DIR/tp2 [--out FILE.json] [--tol-acc X --tol-rms X --tol-scale X]
//!         [--ep-ar-wire bf16|f32]
//! plan    [--model M | synthetic config] [--layer .. --rows .. --route ..]   (no GPU)
//! selftest                                                                   (no GPU)
//! ```
//!
//! # What it proves
//!
//! A case is `(layer, rows, route)`. The block under test (`mixer` = HC read, GDN/QSA, out/o
//! projection, attention HC write; `ffn` = MLP HC read, MoE, MLP HC write) is executed by the
//! PRODUCT forward program (`forward_mesh_chunk` over the op range of the layer; the same
//! kernels, route selection, scratch and fusions as serving) on HC streams that are INJECTED
//! between program segments, so the block sees byte-identical input on every arm and rank
//! (deterministic synthetic streams, or the real streams the layers before it produced on a
//! prompt: `--input real` on the reference, `--input from:DIR` on the TP2 arm; real streams
//! give the router real routes). Nothing is committed (no head, no state commit): each case
//! starts from `Qwen4Bundle::reset`.
//!
//! * **reference arm** (`--arm ref`, one card): the unsharded AR-only layer, both stream
//!   kinds (`rows < 512`: F32 words holding BF16-exact values, decode / verify / short
//!   chunks; `prefill` rows >= `QWEN4_F16_WMMA_MIN_TOKENS`: BF16 bits). With
//!   `HIPFIRE_QWEN4_HC_FUSE=1` the mixer's HC write stays unfused and the full block output
//!   `m_ref` is recorded; otherwise (level 2 fuses the write into the projection epilogue)
//!   only the streams are. `--repeat N` re-runs every case from a reset and checks the bytes
//!   do not change (determinism of the arm itself).
//! * **TP2 arm** (`--arm tp2`, two cards, two threads): the same layer with the weights of
//!   `Qwen4Weights::shard_trunk` (cut after the product load's overlay/requant), the
//!   rank-local `Qwen4Config::trunk_shard_local` (8/24 GDN heads, 12 q heads over 2 KV
//!   heads), a state/forward rebuilt on it, and the mixer's HC write carrying the fused
//!   all-reduce (`ArHcExchange`, installed through the forward's own `HyperWriteOp.ar` hook),
//!   mailbox qualified at start. It does not read `HIPFIRE_QWEN4_EP_TRUNK_SHARD` and does
//!   not depend on the switch.
//! * **checks** per block (`evaluate_block`, CPU-side and unit-tested; `compare` re-runs
//!   them from the dumps on any box):
//!   C0 identical block input on both ranks and on the reference;
//!   C1 rank0 == rank1 bytes (output streams, gate logits);
//!   C2 fold contract on the REAL partials: `|m-(p0+p1)| <= ArHcPartial::fold_bound` of the
//!   mixer wire under test (`--ep-ar-wire`, default bf16: `(2^-8+2^-23)(|p0|+|p1|)`; f32:
//!   `2^-24(|p0|+|p1|)`); pass the same `--ep-ar-wire` to `run --arm tp2` and to `compare`;
//!   C3 the device's fused write == host model (`ArHcPartial::fold` + `hc_inject[_f32]`) of the
//!   captured input/partials/gates, word for word (one-bf16-ulp gate neighbours, host libm vs
//!   device `expf`, are counted as benign);
//!   C4 `p0+p1` vs the reference block output `m_ref` within
//!   `fold_bound + acc(|p0|+|p1|) + row_rms * rms(m_ref)` (this is the cut/kernel-shape
//!   check: a wrong head order or cut is O(|m|));
//!   C5 EVERY HC stream element TP2 vs reference within the documented envelope (see
//!   `check_stream_bound`);
//!   C6 gate logits equal the reference's.
//!   Reports max/mean relative differences (relative to the reference RMS) and the largest
//!   bf16-ulp distance per stream branch and per block, and the fold's own error.
//!
//! # What it does not prove
//!
//! * No KLD, no end-to-end quality: one block on injected streams, fresh state (position 0,
//!   zero recurrent/conv state, `rows` tokens of context). Decode with long context, state
//!   evolution across steps and the MTP head are not exercised.
//! * The TP2 arm runs the MIXER only. The FFN partial route (owned routed slots on the
//!   zero-dummy tables + the K-half shared expert through the fused all-reduce) needs the EP2
//!   expert placement, which a single-device load does not have; `--block ffn|both` runs the
//!   unsharded reference FFN (a baseline and determinism check) and the TP2 arm reports the
//!   FFN as skipped. Checking the FFN partial (zero-row claim on the real dummy tables,
//!   384/256 shared K-halves, the prefill chunk partial) needs the product EP2 load with the
//!   shard on (the developer override of the load slice): the block runner here would then
//!   have to drive `forward_mesh_chunk` on an EP forward, which the mesh entry refuses
//!   today; the existing end-to-end probe for it is `fn_ep_oracle --chunk-dump` with
//!   `HIPFIRE_CHUNK_DUMP_RAW=1` (per-layer raw HC streams per rank).
//! * The shard is applied to a single-device product load by hand (`localize_bundle`). The
//!   product EP2 load path (`apply_trunk_shard`, rank-local config at state allocation, the
//!   EP replica check, expert units) is NOT exercised here, and neither is the Q8 trunk
//!   overlay recipe beyond what the product load applied before the cut.
//! * The injected input is a fixed deterministic stream or a real prefix; it is not a
//!   sample of the activation distribution, so the checks are contract/bug detectors (a bad
//!   cut, kernel shape, head order, gate or rounding order), not statistical equivalence.
//! * The all-reduce kernel itself is qualified at start (both stream kinds) and compared
//!   word for word here on real partials; its fault handling is the separate
//!   `ar_hc_write_oracle`.
//!
//! Exit code: 0 all checks pass, 3 any check fails, 2 usage or setup error.

use hipfire_arch_qwen4::lab_trunk_oracle::{
    self as oracle, Block, BlockReport, CaseSpec, Check, InputSource, RouteSel, Tolerances, TpHook,
};
use hipfire_runtime::ar_hc_exchange::ArHcExchange;
use rdna_compute::ar_hc_write::{ArHcPartial, StreamKind, AR_HC_ROW_CAP};
use rdna_compute::Gpu;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const DEFAULT_MODEL: &str =
    "/home/kaden/qcal/release-0.4.3/mgfn/model/qwen3.8-flash-next-base.mq4-xts";

type R<T> = Result<T, String>;

const HELP: &str = "\
trunk_shard_layer_oracle: one real Flash-Next layer, unsharded vs TP2 trunk shard (lab)

commands:
  plan       print the case plan and op segments of the real program (no GPU)
  selftest   run the contract analysis on synthetic captures incl. injected defects (no GPU)
  run        run an arm on the GPU and write its dump: --arm ref | --arm tp2
  compare    evaluate a tp2 dump against a ref dump (no GPU)

common options:
  --model PATH             artifact (default the mq4-xts under release-0.4.3/mgfn/model)
  --layer 0,3              layers to test (default 0,3: a GDN and a QSA layer)
  --block mixer|ffn|both   blocks (default mixer; tp2 runs the mixer only)
  --rows 1,8,32            rows per case (default 1,8,32; rows >= 512 run BF16 streams)
  --route auto|decode|verify|prefill   (default auto: 1 decode, 2..=16 verify, else prefill)
  --input synthetic[:SEED]|real|from:DIR   block input (default synthetic:1)
  --repeat N               run every case N times from a reset (default 2)
  --out DIR                dump directory (default ./trunk-shard-oracle)
  --max-seq N              context capacity of the load (default 1024)
  --prompt-ids A,B,..      token ids of the prefix/embedding (default deterministic ids)
  --prompt-file F          text tokenized with the artifact's tokenizer
run options:
  --arm ref|tp2            --device N (ref, default 0)   --devices A,B (tp2, default 0,1)
  --ref DIR                tp2: also compare against this ref dump when done
  --no-qualify             tp2: skip the start-up qualification of the exchange
  --dry-run                print the plan and exit (no GPU)
compare/run tolerances:
  --tol-acc X              fp32 accumulation-order allowance (default 2^-14)
  --tol-rms X              cancellation floor, fraction of the reference row RMS (default 2^-12)
  --tol-scale X            scales the contract bounds (default 1)
compare options:
  --ref DIR --tp DIR [--json FILE]
";

// ================================================================================================
// Arguments
// ================================================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Ref,
    Tp2,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Ref => "ref",
            Arm::Tp2 => "tp2",
        }
    }
}

#[derive(Clone, Debug)]
enum InputArg {
    Synthetic(u64),
    Real,
    From(PathBuf),
}

#[derive(Clone, Debug)]
struct Args {
    command: String,
    model: Option<String>,
    arm: Option<Arm>,
    out: PathBuf,
    ref_dir: Option<PathBuf>,
    tp_dir: Option<PathBuf>,
    json: Option<PathBuf>,
    layers: Vec<usize>,
    blocks: Vec<Block>,
    rows: Vec<usize>,
    route: Option<RouteSel>,
    input: InputArg,
    repeat: usize,
    max_seq: usize,
    device: i32,
    devices: [i32; 2],
    prompt_ids: Option<Vec<u32>>,
    prompt_file: Option<PathBuf>,
    no_qualify: bool,
    dry_run: bool,
    tol: Tolerances,
}

fn parse_list<T: std::str::FromStr>(text: &str, what: &str) -> R<Vec<T>> {
    text.split(',')
        .map(|item| {
            item.trim()
                .parse::<T>()
                .map_err(|_| format!("{what}: `{item}` is not a valid value"))
        })
        .collect()
}

fn parse_blocks(text: &str) -> R<Vec<Block>> {
    match text {
        "mixer" => Ok(vec![Block::Mixer]),
        "ffn" => Ok(vec![Block::Ffn]),
        "both" => Ok(vec![Block::Mixer, Block::Ffn]),
        other => Err(format!("--block `{other}` is not mixer|ffn|both")),
    }
}

fn parse_args(raw: &[String]) -> R<Args> {
    let mut args = Args {
        command: String::new(),
        model: None,
        arm: None,
        out: PathBuf::from("trunk-shard-oracle"),
        ref_dir: None,
        tp_dir: None,
        json: None,
        layers: vec![0, 3],
        blocks: vec![Block::Mixer],
        rows: vec![1, 8, 32],
        route: None,
        input: InputArg::Synthetic(1),
        repeat: 2,
        max_seq: 1024,
        device: 0,
        devices: [0, 1],
        prompt_ids: None,
        prompt_file: None,
        no_qualify: false,
        dry_run: false,
        tol: Tolerances::default(),
    };
    let mut it = raw.iter().peekable();
    args.command = it.next().cloned().unwrap_or_default();
    while let Some(flag) = it.next() {
        let mut value = |name: &str| -> R<String> {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match flag.as_str() {
            "--model" => args.model = Some(value(flag)?),
            "--arm" => {
                args.arm = Some(match value(flag)?.as_str() {
                    "ref" => Arm::Ref,
                    "tp2" => Arm::Tp2,
                    other => return Err(format!("--arm `{other}` is not ref|tp2")),
                })
            }
            "--out" => args.out = PathBuf::from(value(flag)?),
            "--ref" => args.ref_dir = Some(PathBuf::from(value(flag)?)),
            "--tp" => args.tp_dir = Some(PathBuf::from(value(flag)?)),
            "--json" => args.json = Some(PathBuf::from(value(flag)?)),
            "--layer" => args.layers = parse_list(&value(flag)?, "--layer")?,
            "--block" => args.blocks = parse_blocks(&value(flag)?)?,
            "--rows" => args.rows = parse_list(&value(flag)?, "--rows")?,
            "--route" => {
                let v = value(flag)?;
                args.route = if v == "auto" {
                    None
                } else {
                    Some(RouteSel::parse(&v)?)
                };
            }
            "--input" => {
                let v = value(flag)?;
                args.input = if v == "real" {
                    InputArg::Real
                } else if let Some(seed) = v.strip_prefix("synthetic:") {
                    InputArg::Synthetic(
                        seed.parse()
                            .map_err(|_| "--input synthetic:SEED".to_string())?,
                    )
                } else if v == "synthetic" {
                    InputArg::Synthetic(1)
                } else if let Some(dir) = v.strip_prefix("from:") {
                    InputArg::From(PathBuf::from(dir))
                } else {
                    return Err(format!(
                        "--input `{v}` is not synthetic[:SEED]|real|from:DIR"
                    ));
                };
            }
            "--repeat" => {
                args.repeat = value(flag)?.parse().map_err(|_| "--repeat N".to_string())?
            }
            "--max-seq" => {
                args.max_seq = value(flag)?
                    .parse()
                    .map_err(|_| "--max-seq N".to_string())?
            }
            "--device" => {
                args.device = value(flag)?.parse().map_err(|_| "--device N".to_string())?
            }
            "--devices" => {
                let d: Vec<i32> = parse_list(&value(flag)?, "--devices")?;
                if d.len() != 2 || d[0] == d[1] {
                    return Err("--devices wants two distinct device ids".into());
                }
                args.devices = [d[0], d[1]];
            }
            "--prompt-ids" => args.prompt_ids = Some(parse_list(&value(flag)?, "--prompt-ids")?),
            "--prompt-file" => args.prompt_file = Some(PathBuf::from(value(flag)?)),
            "--no-qualify" => args.no_qualify = true,
            "--dry-run" => args.dry_run = true,
            "--tol-acc" => {
                args.tol.acc = value(flag)?
                    .parse()
                    .map_err(|_| "--tol-acc X".to_string())?
            }
            "--tol-rms" => {
                args.tol.row_rms = value(flag)?
                    .parse()
                    .map_err(|_| "--tol-rms X".to_string())?
            }
            "--ep-ar-wire" => {
                args.tol.fold = match value(flag)?.as_str() {
                    "bf16" => ArHcPartial::Bf16,
                    "f32" => ArHcPartial::F32,
                    other => return Err(format!("--ep-ar-wire `{other}` is not bf16|f32")),
                }
            }
            "--tol-scale" => {
                args.tol.scale = value(flag)?
                    .parse()
                    .map_err(|_| "--tol-scale X".to_string())?
            }
            "-h" | "--help" | "help" => {
                args.command = "help".into();
            }
            other => return Err(format!("unknown option `{other}` (see --help)")),
        }
    }
    if args.repeat == 0 {
        return Err("--repeat must be at least 1".into());
    }
    if args.rows.is_empty() || args.rows.contains(&0) || args.layers.is_empty() {
        return Err("--rows and --layer need non-zero / non-empty lists".into());
    }
    Ok(args)
}

impl Args {
    fn model_path(&self) -> String {
        self.model
            .clone()
            .unwrap_or_else(|| DEFAULT_MODEL.to_string())
    }

    fn cases(&self) -> Vec<CaseSpec> {
        self.layers
            .iter()
            .flat_map(|&layer| self.rows.iter().map(move |&rows| (layer, rows)))
            .map(|(layer, rows)| CaseSpec {
                layer,
                rows,
                route: self.route.unwrap_or_else(|| RouteSel::auto(rows)),
            })
            .collect()
    }
}

// ================================================================================================
// Config without a GPU (plan, dry-run)
// ================================================================================================

fn load_config(args: &Args) -> R<(hipfire_arch_qwen4::config::Qwen4Config, String)> {
    match &args.model {
        Some(path) => {
            let hfq = hipfire_runtime::hfq::HfqFile::open(Path::new(path))
                .map_err(|e| format!("{path}: {e}"))?;
            let receipt =
                hipfire_arch_qwen4::admit_hfqm_artifact(&hfq).map_err(|e| e.to_string())?;
            Ok((receipt.config, path.clone()))
        }
        None => Ok((
            oracle::synthetic_flash_next_config(),
            "(synthetic compact Flash-Next config: pass --model for the artifact's)".to_string(),
        )),
    }
}

fn print_plan(args: &Args) -> R<()> {
    let (config, source) = load_config(args)?;
    println!("trunk_shard_layer_oracle plan");
    println!("  config source: {source}");
    println!(
        "  global: {} layers, GDN heads {}/{} (key/value), q heads {} kv {}, shared {}, profile {}",
        config.num_hidden_layers,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.shared_expert_intermediate_size,
        config.profile.id()
    );
    let ar = config.clone().into_ar_only()?;
    for rank in 0..2 {
        let local = ar.trunk_shard_local(rank, 2)?;
        println!(
            "  rank {rank} local config: GDN heads {}/{}, q heads {} kv {}, shared {}  (validated against its global)",
            local.linear_num_key_heads,
            local.linear_num_value_heads,
            local.num_attention_heads,
            local.num_key_value_heads,
            local.shared_expert_intermediate_size
        );
        local.validate()?;
    }
    println!(
        "  stream kind: F32 words for rows < {} (route-dependent: see the geometry line at run time), BF16 bits from there",
        rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS
    );
    println!("  exchange: up to {AR_HC_ROW_CAP} rows per launch (longer cases launch in slices)");
    for case in args.cases() {
        let plan = oracle::layer_plan(&config, case.layer)?;
        println!(
            "  case {}: {} layer, {} rows, route {}; op spans prefix {:?} mixer {:?} ffn {:?}",
            case.dir_name(),
            oracle::layer_type_name(plan.layer_type),
            case.rows,
            case.route.name(),
            plan.prefix,
            plan.mixer,
            plan.ffn
        );
        for block in &args.blocks {
            println!(
                "      block {:<5} ref: run   tp2: {}",
                block.name(),
                match block {
                    Block::Mixer => "run (fused all-reduce on the attention HC write)",
                    Block::Ffn => "SKIPPED (needs the EP2 expert partial route)",
                }
            );
        }
    }
    println!(
        "  checks per block: C0 identical input, C1 rank0==rank1, C2 fold contract, C3 device==host model, C4 partials vs reference m, C5 every stream element vs reference, C6 gates"
    );
    Ok(())
}

// ================================================================================================
// Tokens, meta
// ================================================================================================

fn default_tokens(vocab: usize, n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 7919 + 17) % vocab) as u32).collect()
}

/// The prompt tokens of every case: `--prompt-ids`, the encoded `--prompt-file` (`encoded`, done
/// by the caller against the loaded model's tokenizer) or deterministic ids.
fn resolve_tokens(
    args: &Args,
    encoded: Option<Vec<u32>>,
    vocab: usize,
    need: usize,
) -> R<Vec<u32>> {
    let tokens = if let Some(ids) = &args.prompt_ids {
        ids.clone()
    } else if let Some(ids) = encoded {
        ids
    } else {
        default_tokens(vocab, need)
    };
    if tokens.is_empty() {
        return Err("empty token list".into());
    }
    if let Some(bad) = tokens.iter().find(|t| **t as usize >= vocab) {
        return Err(format!(
            "token id {bad} is outside the {vocab}-entry vocabulary"
        ));
    }
    // Cycle a short prompt so every case has `need` rows of tokens.
    Ok(tokens
        .iter()
        .copied()
        .cycle()
        .take(need.max(tokens.len()))
        .collect())
}

/// Encode `--prompt-file` with the loaded model's own tokenizer.
fn encode_prompt_file(args: &Args, model: &hipfire_loader::LoadedModel) -> R<Option<Vec<u32>>> {
    let Some(file) = &args.prompt_file else {
        return Ok(None);
    };
    let text = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(Some(
        model
            .tokenizer
            .as_ref()
            .ok_or("the loaded model has no tokenizer")?
            .encode(&text),
    ))
}

fn env_snapshot() -> Value {
    let map: serde_json::Map<String, Value> = std::env::vars()
        .filter(|(k, _)| {
            k.starts_with("HIPFIRE_")
                || matches!(
                    k.as_str(),
                    "ROCR_VISIBLE_DEVICES" | "HIP_VISIBLE_DEVICES" | "FABRIC_CARDS" | "FABRIC_JOB"
                )
        })
        .map(|(k, v)| (k, Value::String(v)))
        .collect();
    Value::Object(map)
}

fn write_json(path: &Path, value: &Value) -> R<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    std::fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", path.display()))
}

// ================================================================================================
// Lockstep between the two rank threads
// ================================================================================================

/// A two-party barrier that can be poisoned: a rank that fails calls [`Rendezvous::fail`] and
/// the other rank's next `wait` returns `Err` instead of hanging.
struct Rendezvous {
    state: Mutex<(usize, usize, bool)>,
    cv: Condvar,
}

impl Rendezvous {
    fn new() -> Self {
        Self {
            state: Mutex::new((0, 0, false)),
            cv: Condvar::new(),
        }
    }

    fn fail(&self) {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).2 = true;
        self.cv.notify_all();
    }

    fn wait(&self) -> R<()> {
        let mut guard = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if guard.2 {
            return Err("the peer rank failed".into());
        }
        guard.0 += 1;
        if guard.0 == 2 {
            guard.0 = 0;
            guard.1 += 1;
            self.cv.notify_all();
            return Ok(());
        }
        let generation = guard.1;
        while guard.1 == generation && !guard.2 {
            let (g, timeout) = self
                .cv
                .wait_timeout(guard, Duration::from_secs(3600))
                .unwrap_or_else(|p| p.into_inner());
            guard = g;
            if timeout.timed_out() {
                guard.2 = true;
                self.cv.notify_all();
                return Err("timed out waiting for the peer rank".into());
            }
        }
        if guard.2 {
            return Err("the peer rank failed".into());
        }
        Ok(())
    }
}

// ================================================================================================
// Running an arm
// ================================================================================================

/// Per case: the outcomes of each repeat.
type CaseRuns = Vec<(CaseSpec, Vec<oracle::RunOutcome>)>;

struct ArmResult {
    runs: CaseRuns,
    localized: String,
}

fn load_loaded_model(args: &Args, gpu: &mut Gpu) -> R<hipfire_loader::LoadedModel> {
    let spec = hipfire_runtime::loader_api::SpecLoadCfg::default();
    hipfire_loader::load_model(
        &args.model_path(),
        args.max_seq,
        None,
        None,
        None,
        None,
        1,
        spec,
        gpu,
    )
}

fn blocks_for(arm: Arm, requested: &[Block]) -> Vec<Block> {
    match arm {
        Arm::Ref => requested.to_vec(),
        Arm::Tp2 => requested
            .iter()
            .copied()
            .filter(|b| *b == Block::Mixer)
            .collect(),
    }
}

fn input_for(args: &Args, arm: Arm, case: &CaseSpec, blocks: &[Block]) -> R<InputSource> {
    match (&args.input, arm) {
        (InputArg::Synthetic(seed), _) => Ok(InputSource::Synthetic(*seed)),
        (InputArg::Real, Arm::Ref) => Ok(InputSource::Real),
        (InputArg::Real, Arm::Tp2) => Err(
            "--input real on the tp2 arm would run the sharded prefix with unreduced FFN partials; \
             record the reference's real input (`run --arm ref --input real`) and pass \
             `--input from:<ref dump>` here"
                .into(),
        ),
        (InputArg::From(dir), _) => {
            let first = *blocks.first().ok_or("no blocks")?;
            let path = oracle::block_dir(dir, case, first).join("in.bin");
            std::fs::read(&path)
                .map(InputSource::Fixed)
                .map_err(|e| format!("--input from: {}: {e}", path.display()))
        }
    }
}

fn run_ref(args: &Args) -> R<ArmResult> {
    let mut gpu = Gpu::init_with_device(args.device).map_err(|e| format!("Gpu::init: {e}"))?;
    eprintln!("[oracle] ref: loading {}", args.model_path());
    let mut model = load_loaded_model(args, &mut gpu)?;
    let encoded = encode_prompt_file(args, &model)?;
    let bundle = model
        .qwen4_mut()
        .ok_or("loaded model is not a Qwen4 (Flash-Next) bundle")?;
    let localized = oracle::localize_bundle(bundle, &mut gpu, None, args.max_seq)?;
    eprintln!("[oracle] ref: {}", localized.line);
    let cases = args.cases();
    let need = cases.iter().map(|c| c.rows).max().unwrap_or(1);
    let tokens = resolve_tokens(args, encoded, bundle.config.vocab_size, need)?;
    let blocks = blocks_for(Arm::Ref, &args.blocks);
    let mut runs: CaseRuns = Vec::new();
    for case in cases {
        let input = input_for(args, Arm::Ref, &case, &blocks)?;
        let mut outcomes = Vec::new();
        for rep in 0..args.repeat {
            let outcome =
                oracle::run_case(bundle, &mut gpu, &case, &blocks, &tokens, &input, None)?;
            eprintln!(
                "[oracle] ref {} rep {rep}: {}",
                case.dir_name(),
                outcome.notes.join("; ")
            );
            outcomes.push(outcome);
        }
        runs.push((case, outcomes));
    }
    Ok(ArmResult {
        runs,
        localized: localized.line,
    })
}

fn tp_rank(
    args: &Args,
    rank: usize,
    exchange: Arc<ArHcExchange>,
    rendezvous: Arc<Rendezvous>,
) -> R<ArmResult> {
    let fail = |e: String| {
        rendezvous.fail();
        format!("rank {rank}: {e}")
    };
    let mut gpu =
        Gpu::init_with_device(args.devices[rank]).map_err(|e| fail(format!("Gpu::init: {e}")))?;
    oracle::register_exchange(&exchange, rank, &mut gpu).map_err(&fail)?;
    rendezvous.wait()?;
    oracle::enable_exchange(&exchange, rank, &mut gpu).map_err(&fail)?;
    rendezvous.wait()?;
    if !args.no_qualify {
        oracle::qualify_exchange(&exchange, rank, &mut gpu).map_err(&fail)?;
    }
    rendezvous.wait()?;
    eprintln!(
        "[oracle] rank {rank}: exchange ready; loading {}",
        args.model_path()
    );
    let mut model = load_loaded_model(args, &mut gpu).map_err(&fail)?;
    let encoded = encode_prompt_file(args, &model).map_err(&fail)?;
    let bundle = model
        .qwen4_mut()
        .ok_or_else(|| fail("loaded model is not a Qwen4 (Flash-Next) bundle".into()))?;
    let localized =
        oracle::localize_bundle(bundle, &mut gpu, Some(rank), args.max_seq).map_err(&fail)?;
    eprintln!("[oracle] rank {rank}: {}", localized.line);
    let cases = args.cases();
    let need = cases.iter().map(|c| c.rows).max().unwrap_or(1);
    let tokens = resolve_tokens(args, encoded, bundle.config.vocab_size, need).map_err(&fail)?;
    let blocks = blocks_for(Arm::Tp2, &args.blocks);
    if blocks.is_empty() {
        return Err(fail(
            "the tp2 arm runs the mixer only: pass --block mixer|both".into(),
        ));
    }
    let hook = TpHook {
        exchange: exchange.clone(),
        rank,
        mixer: args.tol.fold,
    };
    let mut runs: CaseRuns = Vec::new();
    for case in cases {
        let input = input_for(args, Arm::Tp2, &case, &blocks).map_err(&fail)?;
        let mut outcomes = Vec::new();
        for rep in 0..args.repeat {
            rendezvous.wait()?;
            let outcome = oracle::run_case(
                bundle,
                &mut gpu,
                &case,
                &blocks,
                &tokens,
                &input,
                Some(&hook),
            )
            .map_err(|e| {
                // Drain, then zero the generation words so a later case is not poisoned.
                let _ = gpu.hip.device_synchronize();
                fail(format!("{} rep {rep}: {e}", case.dir_name()))
            })?;
            eprintln!(
                "[oracle] rank {rank} {} rep {rep}: {}",
                case.dir_name(),
                outcome.notes.join("; ")
            );
            outcomes.push(outcome);
        }
        runs.push((case, outcomes));
    }
    Ok(ArmResult {
        runs,
        localized: localized.line,
    })
}

fn run_tp2(args: &Args) -> R<[ArmResult; 2]> {
    let exchange = Arc::new(ArHcExchange::new(AR_HC_ROW_CAP).map_err(|e| e.to_string())?);
    let rendezvous = Arc::new(Rendezvous::new());
    let handles: Vec<_> = (0..2)
        .map(|rank| {
            let (args, exchange, rendezvous) = (args.clone(), exchange.clone(), rendezvous.clone());
            std::thread::Builder::new()
                .name(format!("oracle-rank{rank}"))
                .stack_size(64 << 20)
                .spawn(move || tp_rank(&args, rank, exchange, rendezvous))
                .map_err(|e| e.to_string())
        })
        .collect::<R<Vec<_>>>()?;
    let mut results = Vec::new();
    let mut errors = Vec::new();
    for handle in handles {
        match handle.join() {
            Ok(Ok(result)) => results.push(result),
            Ok(Err(e)) => errors.push(e),
            Err(_) => errors.push("a rank thread panicked".to_string()),
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("; "));
    }
    let r1 = results.pop().ok_or("no rank 1 result")?;
    let r0 = results.pop().ok_or("no rank 0 result")?;
    Ok([r0, r1])
}

// ================================================================================================
// Dumps
// ================================================================================================

fn case_json(
    case: &CaseSpec,
    layer_type: &str,
    kind: StreamKind,
    geometry: &oracle::Geometry,
    notes: &[String],
) -> Value {
    json!({
        "layer": case.layer,
        "layer_type": layer_type,
        "rows": case.rows,
        "route": case.route.name(),
        "stream_kind": kind.name(),
        "routing_rows": geometry.route_rows,
        "max_chunk": geometry.max_chunk,
        "notes": notes,
    })
}

fn layer_type_of(outcome: &oracle::RunOutcome) -> String {
    outcome
        .notes
        .iter()
        .find_map(|n| {
            let rest = n.strip_prefix("layer ")?;
            let open = rest.find('(')?;
            Some(rest[open + 1..rest.len() - 1].to_string())
        })
        .unwrap_or_else(|| "?".to_string())
}

/// Write one arm's dump. Rank 0's file names carry `rank0`; the reference arm is rank 0 only.
fn write_dump(arm_dir: &Path, results: &[&ArmResult]) -> R<Vec<Check>> {
    let mut checks = Vec::new();
    for (rank, result) in results.iter().enumerate() {
        for (case, outcomes) in &result.runs {
            let first = outcomes.first().ok_or("no outcome")?;
            let case_dir = arm_dir.join(case.dir_name());
            if rank == 0 {
                write_json(
                    &case_dir.join("case.json"),
                    &case_json(
                        case,
                        &layer_type_of(first),
                        first.geometry.kind,
                        &first.geometry,
                        &first.notes,
                    ),
                )?;
            }
            for (block, capture) in &first.blocks {
                oracle::write_capture(&oracle::block_dir(arm_dir, case, *block), rank, capture)?;
                // Determinism of this rank across repeats.
                for (rep, other) in outcomes.iter().enumerate().skip(1) {
                    if let Some((_, again)) = other.blocks.iter().find(|(b, _)| b == block) {
                        let mut c = oracle::check_bytes_equal(
                            "R0 repeat determinism (output streams)",
                            &capture.output,
                            &again.output,
                            &format!(
                                "{} {} rank {rank} rep 0 vs {rep}",
                                case.dir_name(),
                                block.name()
                            ),
                        );
                        if let (Some(a), Some(b)) = (&capture.partial, &again.partial) {
                            let p = oracle::check_bytes_equal(
                                "R0 repeat determinism (block output)",
                                &oracle::f32_bytes(a),
                                &oracle::f32_bytes(b),
                                &format!(
                                    "{} {} rank {rank} rep 0 vs {rep}",
                                    case.dir_name(),
                                    block.name()
                                ),
                            );
                            if !p.pass {
                                c = p;
                            }
                        }
                        checks.push(c);
                    }
                }
            }
        }
    }
    Ok(checks)
}

// ================================================================================================
// Compare
// ================================================================================================

fn read_case_json(dir: &Path) -> R<Value> {
    let path = dir.join("case.json");
    serde_json::from_slice(&std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn compare_dirs(ref_dir: Option<&Path>, tp_dir: &Path, tol: &Tolerances) -> R<Vec<BlockReport>> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(tp_dir)
        .map_err(|e| format!("{}: {e}", tp_dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("case.json").exists())
        .collect();
    entries.sort();
    let mut reports = Vec::new();
    for case_path in entries {
        let name = case_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string();
        let meta = read_case_json(&case_path)?;
        let layer_type: &'static str = match meta["layer_type"].as_str() {
            Some("gdn") => "gdn",
            Some("qsa") => "qsa",
            _ => "?",
        };
        for block in [Block::Mixer, Block::Ffn] {
            let block_path = case_path.join(block.name());
            if !block_path.exists() {
                continue;
            }
            let (Some(r0), Some(r1)) = (
                oracle::read_capture(&block_path, 0)?,
                oracle::read_capture(&block_path, 1)?,
            ) else {
                continue;
            };
            let reference = match ref_dir {
                Some(dir) => Some(
                    oracle::read_capture(&dir.join(&name).join(block.name()), 0)?.ok_or_else(
                        || {
                            format!(
                                "reference dump {} lacks {name}/{}",
                                dir.display(),
                                block.name()
                            )
                        },
                    )?,
                ),
                None => None,
            };
            reports.push(oracle::evaluate_block(
                &name,
                block,
                layer_type,
                [&r0, &r1],
                reference.as_ref(),
                tol,
            )?);
        }
    }
    Ok(reports)
}

fn print_reports(reports: &[BlockReport], extra: &[Check]) -> bool {
    let mut pass = true;
    for report in reports {
        print!("{}", report.render());
        pass &= report.pass();
    }
    for check in extra {
        println!(
            "[{}] {}  {}",
            if check.pass { "PASS" } else { "FAIL" },
            check.name,
            check.detail
        );
        pass &= check.pass;
    }
    println!(
        "trunk_shard_layer_oracle: {} ({} blocks, {} extra checks)",
        if pass { "PASS" } else { "FAIL" },
        reports.len(),
        extra.len()
    );
    pass
}

// ================================================================================================
// Commands
// ================================================================================================

fn finish_run(args: &Args, arm: Arm, results: &[&ArmResult], extra: Vec<Check>) -> R<bool> {
    let arm_dir = args.out.join(arm.name());
    let mut checks = extra;
    checks.extend(write_dump(&arm_dir, results)?);
    let mut reports = Vec::new();
    if arm == Arm::Tp2 {
        reports = compare_dirs(args.ref_dir.as_deref(), &arm_dir, &args.tol)?;
    }
    let pass = print_reports(&reports, &checks);
    let meta = json!({
        "oracle": "trunk_shard_layer_oracle",
        "arm": arm.name(),
        "model": args.model_path(),
        "args": format!("{args:?}"),
        "env": env_snapshot(),
        "localized": results.iter().map(|r| r.localized.clone()).collect::<Vec<_>>(),
        "pass": pass,
        "extra_checks": checks.iter().map(Check::to_json).collect::<Vec<_>>(),
        "blocks": reports.iter().map(BlockReport::to_json).collect::<Vec<_>>(),
        "ffn": if arm == Arm::Tp2 { "tp2 FFN skipped: needs the EP2 owner partial route" } else { "ref FFN recorded when --block ffn|both" },
    });
    write_json(&arm_dir.join("report.json"), &meta)?;
    println!("dump + report: {}", arm_dir.display());
    Ok(pass)
}

fn run(args: &Args) -> R<bool> {
    let arm = args.arm.ok_or("run needs --arm ref|tp2")?;
    if args.dry_run {
        print_plan(args)?;
        return Ok(true);
    }
    match arm {
        Arm::Ref => {
            let result = run_ref(args)?;
            finish_run(args, arm, &[&result], Vec::new())
        }
        Arm::Tp2 => {
            if args.ref_dir.is_none() {
                eprintln!("[oracle] note: no --ref DIR: C4/C5/C6 (against the reference) will be skipped; use `compare` later");
            }
            let [r0, r1] = run_tp2(args)?;
            finish_run(args, arm, &[&r0, &r1], Vec::new())
        }
    }
}

fn compare(args: &Args) -> R<bool> {
    let tp = args.tp_dir.as_deref().ok_or("compare needs --tp DIR")?;
    let reports = compare_dirs(args.ref_dir.as_deref(), tp, &args.tol)?;
    if reports.is_empty() {
        return Err(format!("{}: no TP2 case dumps found", tp.display()));
    }
    let pass = print_reports(&reports, &[]);
    if let Some(path) = &args.json {
        write_json(
            path,
            &json!({"pass": pass, "blocks": reports.iter().map(BlockReport::to_json).collect::<Vec<_>>()}),
        )?;
    }
    Ok(pass)
}

/// The file path of the analysis: synthetic dumps (clean and with a wrong-cut defect) written
/// to a temporary tree and evaluated by [`compare_dirs`] exactly as a real run's would be.
fn selftest_dumps() -> R<usize> {
    let root = std::env::temp_dir().join(format!("tslo-selftest-{}", std::process::id()));
    let mut cases = 0;
    for (defect, expect_pass) in [
        (oracle::Defect::None, true),
        (oracle::Defect::CutSlice, false),
    ] {
        let base = root.join(format!("{defect:?}"));
        for kind in StreamKind::ALL {
            let case = CaseSpec {
                layer: 3,
                rows: 2,
                route: RouteSel::Verify,
            };
            let (reference, r0, r1) =
                oracle::synthetic_tp_case(7, 2, kind, ArHcPartial::Bf16, defect);
            for (arm, ranks) in [("ref", vec![&reference]), ("tp2", vec![&r0, &r1])] {
                let arm_dir = base.join(kind.name()).join(arm);
                let geometry = oracle::Geometry {
                    route_rows: 2,
                    kind,
                    max_chunk: 2,
                };
                write_json(
                    &arm_dir.join(case.dir_name()).join("case.json"),
                    &case_json(&case, "qsa", kind, &geometry, &[]),
                )?;
                for (rank, capture) in ranks.into_iter().enumerate() {
                    oracle::write_capture(
                        &oracle::block_dir(&arm_dir, &case, Block::Mixer),
                        rank,
                        capture,
                    )?;
                }
            }
            let reports = compare_dirs(
                Some(&base.join(kind.name()).join("ref")),
                &base.join(kind.name()).join("tp2"),
                &Tolerances::default(),
            )?;
            let pass = !reports.is_empty() && reports.iter().all(BlockReport::pass);
            if pass != expect_pass {
                let _ = std::fs::remove_dir_all(&root);
                return Err(format!(
                    "selftest dump compare: {defect:?} {} expected pass={expect_pass}, got {pass}\n{}",
                    kind.name(),
                    reports.iter().map(BlockReport::render).collect::<String>()
                ));
            }
            cases += 1;
        }
    }
    let _ = std::fs::remove_dir_all(&root);
    Ok(cases)
}

fn selftest() -> R<bool> {
    let lines = oracle::selftest(&[1, 8, 32])?;
    for line in &lines {
        println!("{line}");
    }
    let dumps = selftest_dumps()?;
    println!(
        "trunk_shard_layer_oracle selftest: PASS ({} synthetic captures, {dumps} dump trees through `compare`)",
        lines.len()
    );
    Ok(true)
}

fn real_main() -> R<bool> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() {
        print!("{HELP}");
        return Err("no command".into());
    }
    if matches!(raw[0].as_str(), "-h" | "--help" | "help") {
        print!("{HELP}");
        return Ok(true);
    }
    let args = parse_args(&raw)?;
    match args.command.as_str() {
        "help" => {
            print!("{HELP}");
            Ok(true)
        }
        "plan" => {
            print_plan(&args)?;
            Ok(true)
        }
        "selftest" => selftest(),
        "run" => run(&args),
        "compare" => compare(&args),
        other => Err(format!(
            "unknown command `{other}` (plan|selftest|run|compare; --help)"
        )),
    }
}

fn main() {
    match real_main() {
        Ok(true) => {}
        Ok(false) => std::process::exit(3),
        Err(error) => {
            eprintln!("trunk_shard_layer_oracle: {error}");
            std::process::exit(2);
        }
    }
}
