// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! fn_ep_oracle: Flash-Next Single, expert-parallel and pipeline-parallel exactness oracle (lab).
//!
//! Loads a Qwen4 model through `hipfire_loader::load_model` exactly as the
//! daemon does (EP comes from `topology.ep` / the S1 env var, not from a flag
//! here), prefills each prompt with `prefill_final`, then runs the daemon's
//! greedy decode (`forward_token_or_argmax(None, logits)`) for `--tokens`
//! steps. Everything the single-vs-EP comparison needs is written under
//! `--out`:
//!
//! * `logits.jsonl` — one record per prompt per step (`step` 0 = after prefill,
//!   `step` k = after k decode forwards): token, position, SHA-256 of the rank-0
//!   logits row and (EP) of every peer's logits mirror.
//! * `state.jsonl` — at each `--state-steps` step one record per rank with the
//!   `Qwen4State::debug_digest` rows.
//! * `logits-p<prompt>-s<step>.f32` — raw rank-0 logits at the state steps (and
//!   step 0 / `--save-logits-steps`), for max-abs reporting.
//! * `meta.json` — binary, model, environment, devices, prompt identity.
//!
//! Compare arms with `fn-ep/oracle/compare.py`.
//!
//! Usage:
//! ```text
//! fn_ep_oracle --model M.mq4 --prompt-file a.txt [--prompt-file b.ids ...] \
//!     --max-seq 4096 --tokens 256 --out DIR --arm single \
//!     [--state-steps 1,64,256] [--save-logits-steps 5,6] \
//!     [--moe-dump-positions 0,1] [--perturb-rank R --perturb-step S] \
//!     [--kv-mode M] [--state-quant Q] [--skip-model-hash] \
//!     [--truncate-prompt-tokens N] [--chunk-dump LAYERS|default] [--canonical-state]
//! ```
//!
//! `--canonical-state` emits one sorted logical ownership projection per snapshot
//! (the union of all PP bands, with exactly one owner per key), and saves every
//! output logits row. Without it, existing Single/EP state and logits formats
//! are unchanged. The PP arm is selected by `--pp N` (N > 1, the daemon's
//! `params.pp`), with `HIPFIRE_PP_LAYERS` supplying band counts; `--arm` is
//! only a record label.
//! PP uses explicit mesh prefill/decode/reset and output-owner logits, never EP
//! mirrors. Perturbation is not admitted with PP; native MTP is (`--spec mtp`).
//!
//! `--moe-dump-positions` labels decode forwards 0-based (`position-<P>` under
//! `HIPFIRE_MOE_DUMP_DIR`); with the dump enabled exactly one prompt is
//! accepted and decoding stops after the last listed position (the dump hooks
//! keep writing under the last context otherwise).
//!
//! `--perturb-*` (negative control): after step S is recorded, one mantissa bit
//! of the first nonzero word of rank R's GDN layer-0 recurrent state is flipped
//! (every following token reads it), so that rank must diverge from step S+1.
//!
//! `--spec mtp` (S5): the model loads with native MTP and each prompt runs
//! through the load's MTP speculator exactly as the daemon drives it
//! (`configure_request` with `--temp/--top-p/--top-k/--seed/--ngram`,
//! `prefill`, `step` windows) until `--tokens` tokens are emitted. Then
//! `logits.jsonl` holds one `window` record per window (emitted tokens,
//! proposed/accepted, positions) and a final `tokens` record per prompt, and
//! `state.jsonl` holds per-rank digests (trunk `debug_digest` + every MTP head
//! byte family as `mtp.<name>`) whose `step` is the window index, after the
//! prefill (window 0), every `--state-every` windows (default 8) and after the
//! last window. Window-boundary state comparisons across arms need the same
//! window routes on every arm (`HIPFIRE_MTP_AR_FLOOR=0`; the floor is timed).
//!
//! `--spec mtp` under PP (S6b) drives the same loop through the native drafter's
//! mesh entries with the model's `Gpus` (the daemon's `generate_spec` calls the
//! same entries), so the `window`, `tokens` and `state` records have the Single
//! format. `--canonical-state` adds the output owner's MTP head digest rows
//! (`mtp.<name>`, the absolute keys Single emits) to the one sorted union;
//! without it they join the output owner's band record. PP MTP is greedy only
//! and runs without the n-gram modifier (`--ngram on` is refused, `meta.spec.ngram`
//! records the effective `false`); every arm of a comparison must also run the
//! same route/depth/floor policy (`HIPFIRE_MTP_AR_FLOOR=0`, or a forced
//! `HIPFIRE_MTP_INCREMENTAL=0|1`), which the PP arm requires. `meta.json` adds
//! `numerical_policy` and `placement` (band counts, rank order, edge transports,
//! output owner) for a PP arm.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::config::LayerType;
use hipfire_dispatch::pipeline::moe_dump;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::spec::{PrefillOutcome, SpecRequestConfig, Speculator};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

type R<T> = Result<T, String>;

fn e<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

// ---------------------------------------------------------------- hashing

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn file_sha256(path: &Path) -> R<String> {
    let mut file =
        std::fs::File::open(path).map_err(|x| format!("open {}: {x}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let n = file.read(&mut buffer).map_err(e)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// RFC 1321 MD5 (identity tags only, never a security boundary).
fn md5_hex(data: &[u8]) -> String {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64)
        .map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32)
        .collect();
    let (mut a0, mut b0, mut c0, mut d0) =
        (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_le_bytes());
    for chunk in message.chunks_exact(64) {
        let m: Vec<u32> = chunk
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let rotated = a
                .wrapping_add(f)
                .wrapping_add(k[i])
                .wrapping_add(m[g])
                .rotate_left(S[i]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(rotated);
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = Vec::with_capacity(16);
    for word in [a0, b0, c0, d0] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    hex(&out)
}

// ------------------------------------------------------------------- args

struct Args {
    model: String,
    prompt_files: Vec<String>,
    max_seq: usize,
    /// `--seed` (default `0x1357_9BDF`): the MTP request seed, and the seed the
    /// reserve receipt names in AR mode too.
    seed: u64,
    tokens: usize,
    out: PathBuf,
    arm: String,
    state_steps: BTreeSet<usize>,
    save_logits_steps: BTreeSet<usize>,
    moe_dump_positions: BTreeSet<usize>,
    perturb: Option<(usize, usize)>,
    kv_mode: Option<String>,
    state_quant: Option<String>,
    skip_model_hash: bool,
    canonical_state: bool,
    /// `--pp N`: the pipeline-parallel degree handed to `load_model` (the
    /// daemon's `params.pp`); 1 = Single/EP.
    pp: usize,
    truncate_prompt_tokens: Option<usize>,
    /// `--chunk-dump LAYERS|default`: digest the prefill chunks of these layers
    /// (`hipfire_arch_qwen4::chunk_dump`).
    chunk_dump: Option<String>,
    /// `--spec mtp|dflash`: drive the load's speculator (native MTP, or the
    /// DFlash2 draft `--draft`) instead of AR.
    mtp: Option<MtpArgs>,
    /// `--spec dflash`: the DFlash2 draft and `--dflash-block` (`None` = the
    /// load's `auto` width).
    dflash: Option<(String, Option<usize>)>,
}

/// The `--spec mtp` request: sampling (`--temp 0` = greedy), seed, the
/// n-gram modifier (`auto` = the daemon's arch default) and how often the
/// per-rank state digest is taken (every `state_every` windows, plus the
/// window after prefill and the last one).
struct MtpArgs {
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    seed: u64,
    ngram: Option<bool>,
    state_every: usize,
}

fn usize_list(value: &str) -> R<BTreeSet<usize>> {
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|x| format!("bad number {part:?}: {x}"))
        })
        .collect()
}

fn parse_args() -> R<Args> {
    let mut model = None;
    let mut prompt_files = Vec::new();
    let mut max_seq = None;
    let mut tokens = 256usize;
    let mut out = None;
    let mut arm = None;
    let mut state_steps = BTreeSet::new();
    let mut save_logits_steps = BTreeSet::new();
    let mut moe_dump_positions = BTreeSet::new();
    let mut perturb_rank = None;
    let mut perturb_step = None;
    let mut kv_mode = None;
    let mut state_quant = None;
    let mut skip_model_hash = false;
    let mut canonical_state = false;
    let mut pp = 1usize;
    let mut truncate_prompt_tokens = None;
    let mut chunk_dump: Option<String> = None;
    let mut spec_mtp = false;
    let mut temp = 0.0f32;
    let mut top_p = 1.0f32;
    let mut top_k = None;
    let mut seed = 0x1357_9BDFu64;
    let mut ngram = None;
    let mut state_every = 8usize;
    let mut spec_dflash = false;
    let mut draft = None;
    let mut dflash_block = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match flag.as_str() {
            "--model" => model = Some(value("--model")?),
            "--prompt-file" => prompt_files.push(value("--prompt-file")?),
            "--max-seq" => max_seq = Some(value("--max-seq")?.parse::<usize>().map_err(e)?),
            "--tokens" => tokens = value("--tokens")?.parse::<usize>().map_err(e)?,
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--arm" => arm = Some(value("--arm")?),
            "--state-steps" => state_steps = usize_list(&value("--state-steps")?)?,
            "--save-logits-steps" => {
                save_logits_steps = usize_list(&value("--save-logits-steps")?)?
            }
            "--moe-dump-positions" => {
                moe_dump_positions = usize_list(&value("--moe-dump-positions")?)?
            }
            "--perturb-rank" => {
                perturb_rank = Some(value("--perturb-rank")?.parse::<usize>().map_err(e)?)
            }
            "--perturb-step" => {
                perturb_step = Some(value("--perturb-step")?.parse::<usize>().map_err(e)?)
            }
            "--kv-mode" => kv_mode = Some(value("--kv-mode")?),
            "--state-quant" => state_quant = Some(value("--state-quant")?),
            "--skip-model-hash" => skip_model_hash = true,
            "--canonical-state" => canonical_state = true,
            "--pp" => pp = value("--pp")?.parse::<usize>().map_err(e)?,
            "--help" | "-h" => {
                println!(
                    "fn_ep_oracle --model M.mq4 --prompt-file P [--prompt-file P ...] \
                     --max-seq N --tokens N --out DIR --arm LABEL\n\
                     [--pp N] [--canonical-state] [--state-steps 0,1,64,256] \
                     [--save-logits-steps 5,6] [--kv-mode M] [--state-quant Q]\n\
                     [--truncate-prompt-tokens N] [--skip-model-hash] \
                     [--chunk-dump LAYERS|default] [--moe-dump-positions 0,1]\n\
                     [--perturb-rank R --perturb-step S] [--spec off|mtp] \
                     [--temp T --top-p P --top-k K --seed S --ngram auto|on|off --state-every N]\n\
                     --canonical-state: sorted absolute-layer state keys; save every logits row.\n\
                     PP arm: --pp N (the daemon's params.pp) and HIPFIRE_PP_LAYERS=<comma-separated band counts>.\n\
                     Single cut-safe: HIPFIRE_QWEN4_CUT_LAYERS=<comma-separated band starts>.\n\
                     PP forwards/reset use the mesh and logits use its output owner. \
                     PP --spec mtp: greedy, no --ngram on (the PP drafter has no n-gram pool; pass \
                     --ngram off on every arm of a comparison), with \
                     HIPFIRE_MTP_AR_FLOOR=0 (or a forced HIPFIRE_MTP_INCREMENTAL) on every arm. \
                     PP does not admit --perturb-*."
                );
                std::process::exit(0);
            }
            "--truncate-prompt-tokens" => {
                truncate_prompt_tokens = Some(
                    value("--truncate-prompt-tokens")?
                        .parse::<usize>()
                        .map_err(e)?,
                )
            }
            "--spec" => match value("--spec")?.as_str() {
                "mtp" => (spec_mtp, spec_dflash) = (true, false),
                "dflash" => (spec_mtp, spec_dflash) = (true, true),
                "off" => (spec_mtp, spec_dflash) = (false, false),
                other => return Err(format!("--spec takes mtp, dflash or off, got {other:?}")),
            },
            "--draft" => draft = Some(value("--draft")?),
            "--dflash-block" => {
                dflash_block = Some(value("--dflash-block")?.parse::<usize>().map_err(e)?)
            }
            "--temp" => temp = value("--temp")?.parse::<f32>().map_err(e)?,
            "--top-p" => top_p = value("--top-p")?.parse::<f32>().map_err(e)?,
            "--top-k" => top_k = Some(value("--top-k")?.parse::<u32>().map_err(e)?),
            "--seed" => seed = value("--seed")?.parse::<u64>().map_err(e)?,
            "--ngram" => {
                ngram = match value("--ngram")?.as_str() {
                    "on" => Some(true),
                    "off" => Some(false),
                    "auto" => None,
                    other => return Err(format!("--ngram takes on, off or auto, got {other:?}")),
                }
            }
            "--state-every" => {
                state_every = value("--state-every")?.parse::<usize>().map_err(e)?.max(1)
            }
            "--chunk-dump" => chunk_dump = Some(value("--chunk-dump")?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let perturb = match (perturb_rank, perturb_step) {
        (Some(rank), Some(step)) => Some((rank, step)),
        (None, None) => None,
        _ => return Err("--perturb-rank and --perturb-step go together".into()),
    };
    if prompt_files.is_empty() {
        return Err("at least one --prompt-file is required".into());
    }
    if spec_mtp && (perturb.is_some() || !moe_dump_positions.is_empty()) {
        return Err("--perturb-* and --moe-dump-positions are AR-mode options".into());
    }
    if spec_dflash != draft.is_some() {
        return Err("--spec dflash and --draft go together".into());
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        prompt_files,
        max_seq: max_seq.ok_or("--max-seq is required")?,
        tokens,
        seed,
        out: out.ok_or("--out is required")?,
        arm: arm.ok_or("--arm is required")?,
        state_steps,
        save_logits_steps,
        moe_dump_positions,
        perturb,
        kv_mode,
        state_quant,
        skip_model_hash,
        canonical_state,
        pp,
        truncate_prompt_tokens,
        chunk_dump,
        mtp: spec_mtp.then_some(MtpArgs {
            temp,
            top_p,
            top_k,
            seed,
            ngram,
            state_every,
        }),
        dflash: draft.map(|draft| (draft, dflash_block)),
    })
}

// ---------------------------------------------------------------- prompts

struct Prompt {
    path: String,
    ids: Vec<u32>,
    file_md5: String,
    ids_md5: String,
    kind: &'static str,
    /// Token count before `--truncate-prompt-tokens` cut it.
    full_token_count: usize,
}

fn ids_md5(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    md5_hex(&bytes)
}

fn load_prompt(
    path: &str,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    truncate: Option<usize>,
) -> R<Prompt> {
    let bytes = std::fs::read(path).map_err(|x| format!("read {path}: {x}"))?;
    let file_md5 = md5_hex(&bytes);
    let (ids, kind) = if path.ends_with(".ids") {
        let text = std::str::from_utf8(&bytes).map_err(|x| format!("{path}: {x}"))?;
        let ids = text
            .split_whitespace()
            .map(|word| {
                word.parse::<u32>()
                    .map_err(|x| format!("{path}: bad id {word:?}: {x}"))
            })
            .collect::<R<Vec<u32>>>()?;
        (ids, "ids")
    } else {
        let text = std::str::from_utf8(&bytes).map_err(|x| format!("{path}: not UTF-8: {x}"))?;
        (tokenizer.encode(text), "text")
    };
    let full_token_count = ids.len();
    let mut ids = ids;
    if let Some(limit) = truncate {
        ids.truncate(limit);
    }
    if ids.is_empty() {
        return Err(format!("{path}: empty prompt"));
    }
    Ok(Prompt {
        full_token_count,
        path: path.to_string(),
        file_md5,
        ids_md5: ids_md5(&ids),
        ids,
        kind,
    })
}

// ----------------------------------------------------------------- device

fn tensor_bytes(gpu: &Gpu, tensor: &GpuTensor) -> R<Vec<u8>> {
    gpu.hip.device_synchronize().map_err(e)?;
    gpu.download_raw_bytes(tensor).map_err(e)
}

fn device_info(gpu: &Gpu) -> Value {
    json!({
        "device_id": gpu.device_id,
        "arch": gpu.arch,
        "pci_bus_id": gpu.hip.device_pci_bus_id(gpu.device_id).unwrap_or_else(|x| format!("error: {x}")),
    })
}

/// Flip one mantissa bit of the first nonzero 4-byte word of GDN layer 0's
/// recurrent state (or plant `2.0f32` when the state is all zero).
fn perturb_state(bundle: &mut Qwen4Bundle, gpu: &mut Gpu) -> R<Value> {
    let layer = bundle
        .state
        .gdn
        .first()
        .ok_or("perturb: model has no GDN layer")?;
    let mut bytes = tensor_bytes(gpu, &layer.recurrent)?;
    let words = bytes.len() / 4;
    let index = (0..words)
        .find(|&i| bytes[4 * i..4 * i + 4] != [0, 0, 0, 0])
        .unwrap_or(0);
    let before = bytes[4 * index..4 * index + 4].to_vec();
    if before == [0, 0, 0, 0] {
        bytes[4 * index..4 * index + 4].copy_from_slice(&0x4000_0000u32.to_le_bytes());
    } else {
        bytes[4 * index + 2] ^= 0x40;
    }
    let after = bytes[4 * index..4 * index + 4].to_vec();
    gpu.hip
        .memcpy_htod(&layer.recurrent.buf, &bytes)
        .map_err(e)?;
    gpu.hip.device_synchronize().map_err(e)?;
    Ok(json!({
        "tensor": "gdn[0].recurrent",
        "word_index": index,
        "before": hex(&before),
        "after": hex(&after),
    }))
}

// ------------------------------------------------------------------- main

fn main() {
    if let Err(error) = run() {
        eprintln!("[fn_ep_oracle] ERROR: {error}");
        std::process::exit(1);
    }
}

struct Sinks {
    logits: std::fs::File,
    state: std::fs::File,
}

fn emit(file: &mut std::fs::File, record: &Value) -> R<()> {
    writeln!(file, "{record}").map_err(e)?;
    file.flush().map_err(e)
}

/// S5's whole-model canonical schema, without allocating a reference model.
/// Capture families are optional, but when live must cover every GDN layer.
fn canonical_rows(bundle: &Qwen4Bundle, digests: Vec<Vec<(String, String)>>) -> R<Vec<Value>> {
    let mut union = BTreeMap::new();
    for (rank, digest) in digests.into_iter().enumerate() {
        for (key, value) in digest {
            if union.insert(key.clone(), value).is_some() {
                return Err(format!(
                    "canonical state: duplicate key {key:?} on band {rank}"
                ));
            }
        }
    }
    let capture = union.keys().any(|key| key.starts_with("row_capture["));
    let mut expected: BTreeSet<String> =
        ["ple_history.previous", "ple_history.eos", "hyper_feedback"]
            .into_iter()
            .map(str::to_owned)
            .collect();
    if bundle.config.ple.is_some() {
        expected.insert("ple_conv".into());
    }
    for (layer, kind) in bundle.config.layer_types.iter().enumerate() {
        expected.insert(format!("layer[L{layer}].position"));
        let (prefix, fields): (&str, &[&str]) = match kind {
            LayerType::LinearAttention => ("gdn", &["live", "recurrent", "conv"]),
            LayerType::FullAttention => (
                "qsa",
                &[
                    "marks",
                    "full_keys",
                    "full_values",
                    "raw_index_keys",
                    "pooled_keys",
                    "partial_keys",
                    "partial_values",
                    "selected_indices",
                ],
            ),
        };
        for field in fields {
            expected.insert(format!("{prefix}[L{layer}].{field}"));
        }
        if capture && matches!(kind, LayerType::LinearAttention) {
            for field in ["ring", "inputs", "recurrence"] {
                expected.insert(format!("row_capture[L{layer}].{field}"));
            }
        }
    }
    let actual: BTreeSet<String> = union.keys().cloned().collect();
    if actual != expected {
        return Err(format!(
            "canonical state: missing Single keys {:?}; unexpected keys {:?}",
            expected.difference(&actual).collect::<Vec<_>>(),
            actual.difference(&expected).collect::<Vec<_>>(),
        ));
    }
    Ok(union
        .into_iter()
        .map(|(key, value)| json!([key, value]))
        .collect())
}

fn pp_state_rows(bundle: &Qwen4Bundle, gpus: &mut Gpus, canonical: bool) -> R<Vec<Vec<Value>>> {
    let states: Vec<_> = bundle
        .pp_band_states()
        .ok_or("PP mesh has no band states")?
        .collect();
    if states.len() != gpus.devices.len() {
        return Err("PP state/device band counts disagree".into());
    }
    let digests = states
        .into_iter()
        .zip(&mut gpus.devices)
        .map(|(state, gpu)| {
            gpu.bind_thread().map_err(e)?;
            if canonical {
                state.canonical_digest_rows(gpu).map_err(e)
            } else {
                state.debug_digest(gpu).map_err(e)
            }
        })
        .collect::<R<Vec<_>>>()?;
    if canonical {
        Ok(vec![canonical_rows(bundle, digests)?])
    } else {
        Ok(digests
            .into_iter()
            .map(|digest| {
                digest
                    .into_iter()
                    .map(|(key, value)| json!([key, value]))
                    .collect()
            })
            .collect())
    }
}

/// One `--spec mtp` prompt.
struct MtpRun<'a> {
    prompt_index: usize,
    prompt: &'a Prompt,
    tokens: usize,
    state_every: usize,
    ranks: usize,
    canonical_state: bool,
}

/// Rank `rank`'s state digest rows: `Qwen4State::debug_digest` (trunk) then
/// the SHA-256 of every resident MTP head byte family (`mtp.<name>`).
fn mtp_state_rows(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    rank: usize,
    canonical: bool,
) -> R<Vec<Value>> {
    let rows = |bundle: &mut Qwen4Bundle, gpu: &mut Gpu| -> R<Vec<Value>> {
        let mut rows: Vec<Value> = if canonical {
            canonical_rows(
                bundle,
                vec![bundle.state.canonical_digest_rows(gpu).map_err(e)?],
            )?
        } else {
            bundle
                .state
                .debug_digest(gpu)
                .map_err(e)?
                .into_iter()
                .map(|(name, value)| json!([name, value]))
                .collect()
        };
        for (name, bytes) in bundle.lab_resident_head_items(gpu)? {
            rows.push(json!([format!("mtp.{name}"), sha256_hex(&bytes)]));
        }
        if canonical {
            rows.sort_by(|a, b| a[0].as_str().cmp(&b[0].as_str()));
        }
        Ok(rows)
    };
    if rank == 0 {
        rows(bundle, gpu)
    } else {
        bundle.with_rank_mut(gpu, rank, |peer, peer_gpu, _| rows(peer, peer_gpu))
    }
}

/// Every rank's state rows of a PP MTP snapshot: the band digests
/// ([`pp_state_rows`]) plus the output owner's MTP head byte families
/// (`mtp.<name>`, the same absolute keys Single emits). Canonical state is one
/// sorted union in the rank-0 record; otherwise the head rows join the output
/// owner's own band record.
fn mtp_pp_state_rows(bundle: &Qwen4Bundle, gpus: &mut Gpus, canonical: bool) -> R<Vec<Vec<Value>>> {
    let owner = gpus
        .mesh_output_owner()
        .ok_or("PP mesh has no output owner")?;
    let mut rank_rows = pp_state_rows(bundle, gpus, canonical)?;
    let gpu = gpus
        .devices
        .get_mut(owner)
        .ok_or("PP output owner is not a mesh device")?;
    gpu.bind_thread().map_err(e)?;
    let head_rows: Vec<Value> = bundle
        .lab_resident_head_items(gpu)?
        .into_iter()
        .map(|(name, bytes)| json!([format!("mtp.{name}"), sha256_hex(&bytes)]))
        .collect();
    let rows = rank_rows
        .get_mut(if canonical { 0 } else { owner })
        .ok_or("PP state rows miss the head's record")?;
    rows.extend(head_rows);
    if canonical {
        rows.sort_by(|a, b| a[0].as_str().cmp(&b[0].as_str()));
    }
    Ok(rank_rows)
}

/// One window's `state` records: per-rank on Single/EP, the PP projection on
/// the mesh.
fn mtp_digest(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    pp: Option<&mut Gpus>,
    run: &MtpRun<'_>,
    window: usize,
    sinks: &mut Sinks,
) -> R<()> {
    let rank_rows = match pp {
        Some(gpus) => mtp_pp_state_rows(bundle, gpus, run.canonical_state)?,
        None => (0..run.ranks)
            .map(|rank| mtp_state_rows(bundle, gpu, rank, run.canonical_state))
            .collect::<R<Vec<_>>>()?,
    };
    for (rank, rows) in rank_rows.into_iter().enumerate() {
        emit(
            &mut sinks.state,
            &json!({
                "kind": "state",
                "prompt": run.prompt_index,
                "step": window,
                "rank": rank,
                "digest": rows,
            }),
        )?;
    }
    Ok(())
}

/// The load's native MTP drafter, for the mesh entries.
fn pp_drafter(
    speculator: &mut dyn Speculator,
) -> R<&mut hipfire_arch_qwen4::mtp_spec::Qwen4MtpDrafter> {
    speculator
        .drafter_any_mut()
        .and_then(|any| any.downcast_mut::<hipfire_arch_qwen4::mtp_spec::Qwen4MtpDrafter>())
        .ok_or_else(|| {
            "--spec mtp: the load's speculator is not the native Qwen4 MTP drafter".into()
        })
}

/// Prefill and decode one prompt through the native MTP speculator the way
/// the daemon drives it (`configure_request`, `prefill`, then `step` windows
/// with `position += emit.len()`, `seed = next_seed`) until `tokens` tokens
/// (the prefill seed included) are emitted. Under PP (`pp`) the same loop runs
/// the drafter's mesh entries with the model's `Gpus`, exactly as the daemon's
/// spec loop does (`MtpSpeculator`'s window budget and lowering included).
/// Writes one `window` record per window to the logits sink, the state digest
/// (as `state` records whose `step` is the window index) after the prefill,
/// every `state_every` windows and after the last window, and a final `tokens`
/// record. Returns the emitted token ids (prefill seed included).
fn mtp_prompt(
    gpu: &mut Gpu,
    mut pp: Option<&mut Gpus>,
    bundle: &mut Qwen4Bundle,
    speculator: &mut dyn Speculator,
    request: SpecRequestConfig,
    run: &MtpRun<'_>,
    sinks: &mut Sinks,
) -> R<Vec<u32>> {
    let ids = &run.prompt.ids;
    match pp.as_deref_mut() {
        Some(gpus) => {
            bundle.pp_reset(gpus).map_err(e)?;
            pp_drafter(speculator)?.pp_mtp_reset(gpus, bundle)?;
        }
        None => {
            bundle.reset(gpu).map_err(e)?;
            speculator.reset(gpu)?;
        }
    }
    speculator.configure_request(request);
    let first = match pp.as_deref_mut() {
        Some(gpus) => pp_drafter(speculator)?
            .pp_mtp_prefill(gpus, bundle, ids, ids, 0, false, &|| false)?
            .into_first_token()?,
        None => match speculator.prefill(gpu, bundle, ids, ids, 0, false, None, &|| false)? {
            PrefillOutcome::Ready { first_token } => first_token,
            PrefillOutcome::Aborted => return Err("MTP prefill aborted".into()),
        },
    };
    let mut position = ids.len();
    let mut seed = first;
    let mut emitted = vec![first];
    emit(
        &mut sinks.logits,
        &json!({
            "kind": "window",
            "prompt": run.prompt_index,
            "window": 0,
            "position": position,
            "target_position": bundle.state.position,
            "emit": [first],
        }),
    )?;
    // DFlash2 owns no MTP head: its arm records the windows only.
    let digests = speculator.name() == "mtp";
    if digests {
        mtp_digest(bundle, gpu, pp.as_deref_mut(), run, 0, sinks)?;
    }
    let mut window = 0usize;
    let mut digested = 0usize;
    while emitted.len() < run.tokens {
        window += 1;
        let max_emit = run.tokens - emitted.len();
        let step = match pp.as_deref_mut() {
            Some(gpus) => {
                let eos = bundle.config.eos_token_id;
                let drafter = pp_drafter(speculator)?;
                let k = max_emit.saturating_sub(1).min(
                    hipfire_runtime::spec::MtpDrafter::proposal_capacity(&*drafter),
                );
                let result = drafter.pp_mtp_step(gpus, bundle, position, seed, &emitted, k, eos)?;
                hipfire_arch_qwen4::mtp_spec::window_to_spec_step(result)?.cap_emit(max_emit)
            }
            None => speculator.step(
                gpu,
                bundle,
                position,
                seed,
                &emitted,
                None,
                request.temp,
                max_emit,
            )?,
        };
        position += step.emit.len();
        emitted.extend(step.emit.iter().copied());
        seed = step.next_seed;
        emit(
            &mut sinks.logits,
            &json!({
                "kind": "window",
                "prompt": run.prompt_index,
                "window": window,
                "position": position,
                "target_position": bundle.state.position,
                "emit": step.emit.to_vec(),
                "proposed": step.proposed,
                "accepted": step.accepted,
            }),
        )?;
        if digests && window % run.state_every == 0 {
            mtp_digest(bundle, gpu, pp.as_deref_mut(), run, window, sinks)?;
            digested = window;
        }
        if window % 32 == 0 {
            eprintln!(
                "[fn_ep_oracle] prompt {}: window {window}, {}/{} tokens",
                run.prompt_index,
                emitted.len(),
                run.tokens
            );
        }
    }
    if digests && digested != window {
        mtp_digest(bundle, gpu, pp.as_deref_mut(), run, window, sinks)?;
    }
    emit(
        &mut sinks.logits,
        &json!({
            "kind": "tokens",
            "prompt": run.prompt_index,
            "windows": window,
            "count": emitted.len(),
            "ids_md5": ids_md5(&emitted),
            "ids": emitted,
        }),
    )?;
    Ok(emitted)
}

/// Developer env: print the `EP_RESERVE_RECEIPT <json>` lines the EP2 reserve
/// gate harvests (`scripts/fn_ep_compare.py reserve harvest`); the forward
/// prints the per-rank `header`/`rank`/`prompt_rank` records the same way.
const RESERVE_RECEIPT_ENV: &str = "HIPFIRE_EP_RESERVE_RECEIPT";

fn reserve_receipt(record: &Value) {
    eprintln!("EP_RESERVE_RECEIPT {record}");
}

fn run() -> R<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out)
        .map_err(|x| format!("create {}: {x}", args.out.display()))?;
    if let Some(layers) = args.chunk_dump.as_deref() {
        // Chunk-row dumps (`hipfire_arch_qwen4::chunk_dump`): the arm's prefill chunks of the
        // selected layers are digested under `<out>/chunk-dump/`; `fn-ep/oracle/compare.py
        // chunk` compares arms. Set before the model loads (no thread exists yet).
        // SAFETY: single-threaded at this point of `run`.
        unsafe {
            std::env::set_var(
                hipfire_arch_qwen4::chunk_dump::CHUNK_DUMP_DIR_ENV,
                args.out.join("chunk-dump"),
            );
            if layers != "default" {
                std::env::set_var(
                    hipfire_arch_qwen4::chunk_dump::CHUNK_DUMP_LAYERS_ENV,
                    layers,
                );
            }
        }
    }
    let dump = moe_dump::enabled();
    if dump {
        if args.moe_dump_positions.is_empty() {
            return Err("HIPFIRE_MOE_DUMP_DIR is set: pass --moe-dump-positions".into());
        }
        if args.prompt_files.len() != 1 {
            return Err("MoE dump runs take exactly one --prompt-file".into());
        }
    }

    eprintln!("[fn_ep_oracle] loading {}", args.model);
    let mut gpu = Gpu::init().map_err(|x| format!("Gpu::init: {x:?}"))?;
    let cask = hipfire_runtime::loader_api::CaskConfig::default();
    let mut spec_cfg = hipfire_runtime::loader_api::SpecLoadCfg::default();
    if args.mtp.is_some() {
        spec_cfg.mtp = Some(args.dflash.is_none());
    }
    if let Some((_, block)) = &args.dflash {
        spec_cfg.dflash = Some(true);
        spec_cfg.dflash_block = *block;
    }
    let mut model = hipfire_loader::load_model(
        &args.model,
        args.max_seq,
        args.dflash.as_ref().map(|(draft, _)| draft.as_str()),
        args.kv_mode.as_deref(),
        None,
        args.state_quant.as_deref(),
        &cask,
        args.pp,
        spec_cfg,
        &mut gpu,
    )?;
    let pp = model.pp;
    if pp > 1 && args.perturb.is_some() {
        return Err("PP oracle does not admit --perturb-*".into());
    }

    let prompts = {
        let tokenizer = model
            .tokenizer
            .as_ref()
            .ok_or("loaded model has no tokenizer")?;
        args.prompt_files
            .iter()
            .map(|path| load_prompt(path, tokenizer, args.truncate_prompt_tokens))
            .collect::<R<Vec<Prompt>>>()?
    };
    for prompt in &prompts {
        if prompt.ids.len() + args.tokens > args.max_seq {
            return Err(format!(
                "{}: {} prompt tokens + {} decode tokens exceed --max-seq {}",
                prompt.path,
                prompt.ids.len(),
                args.tokens,
                args.max_seq
            ));
        }
    }

    let mut speculator = match &args.mtp {
        Some(_) => Some(
            model
                .speculator
                .take()
                .ok_or("--spec: the load attached no speculator")?,
        ),
        None => None,
    };
    // Under PP the load builds the native drafter without an n-gram pool, so
    // the effective modifier is off whatever `auto` resolves to elsewhere; an
    // explicit `--ngram on` is refused. `meta.spec.ngram` records the effective
    // value, so an arm that really ran the pool never compares equal to it.
    let ngram = args.mtp.as_ref().map(|mtp| {
        if pp > 1 || args.dflash.is_some() {
            return false;
        }
        mtp.ngram.unwrap_or_else(|| {
            hipfire_config::mtp_ngram_enabled_for_arch(hipfire_arch_qwen4::ARCH_ID, &gpu.arch)
        })
    });
    if pp > 1 {
        if let Some(mtp) = &args.mtp {
            // Native MTP on the mesh is greedy, without the n-gram modifier, and
            // only under a route/depth/floor policy that does not depend on
            // timing: every arm of a comparison must run the same windows.
            if !(mtp.temp.is_finite() && mtp.temp <= 1.0e-6) {
                return Err("PP --spec mtp is greedy only: pass --temp 0".into());
            }
            if mtp.ngram == Some(true) {
                return Err(
                    "PP --spec mtp runs without the n-gram modifier: drop --ngram on".into(),
                );
            }
            let floor_off =
                hipfire_config::developer_var("HIPFIRE_MTP_AR_FLOOR").as_deref() == Ok("0");
            let route_forced = matches!(
                hipfire_config::developer_var("HIPFIRE_MTP_INCREMENTAL").as_deref(),
                Ok("0" | "1")
            );
            if !floor_off && !route_forced {
                return Err(
                    "PP --spec mtp needs a timing-independent window policy: set \
                     HIPFIRE_MTP_AR_FLOOR=0 (or force HIPFIRE_MTP_INCREMENTAL=0|1) on every arm"
                        .into(),
                );
            }
        }
    }
    let (bundle, mut pp_gpus) = if pp > 1 {
        let (bundle, gpus) = model
            .qwen4_mesh_mut()
            .ok_or("PP oracle requires a Qwen4 bundle with caller-owned pp_gpus")?;
        if !bundle.pp_mesh_active() {
            return Err("PP oracle requires an active Qwen4 PP mesh".into());
        }
        (bundle, Some(gpus))
    } else {
        (
            model
                .qwen4_mut()
                .ok_or("loaded model is not a Qwen4 (Flash-Next) bundle")?,
            None,
        )
    };
    let ranks = pp_gpus
        .as_ref()
        .map_or_else(|| bundle.ranks(), |gpus| gpus.devices.len());
    if let Some((rank, _)) = args.perturb {
        if rank >= ranks {
            return Err(format!(
                "--perturb-rank {rank} but the bundle has {ranks} rank(s)"
            ));
        }
    }
    let vocab = bundle.config.vocab_size;
    gpu.bind_thread().map_err(e)?;
    let logits = gpu.zeros(&[vocab], DType::F32).map_err(e)?;

    let mut devices = Vec::new();
    if let Some(gpus) = pp_gpus.as_ref() {
        let owner = gpus
            .mesh_output_owner()
            .ok_or("PP mesh has no output owner")?;
        if gpus.devices.get(owner).map(|owner| owner.device_id) != Some(gpu.device_id) {
            return Err("PP output owner is not the daemon logits device".into());
        }
        devices.extend(gpus.devices.iter().enumerate().map(|(rank, gpu)| {
            let mut info = device_info(gpu);
            info["rank"] = json!(rank);
            info
        }));
    } else {
        devices.push(device_info(&gpu));
        for rank in 1..ranks {
            devices.push(bundle.with_rank_mut(&mut gpu, rank, |_, peer, _| Ok(device_info(peer)))?);
        }
    }

    let model_sha = if args.skip_model_hash {
        Value::Null
    } else {
        eprintln!("[fn_ep_oracle] hashing model (use --skip-model-hash to skip)");
        Value::String(file_sha256(Path::new(&args.model))?)
    };
    let exe = std::fs::read("/proc/self/exe").map_err(e)?;
    let env: serde_json::Map<String, Value> = std::env::vars()
        .filter(|(key, _)| {
            key.starts_with("HIPFIRE_")
                || matches!(
                    key.as_str(),
                    "ROCR_VISIBLE_DEVICES" | "HIP_VISIBLE_DEVICES" | "FABRIC_CARDS" | "HOME"
                )
        })
        .map(|(key, value)| (key, Value::String(value)))
        .collect();
    let prompt_meta: Vec<Value> = prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| {
            json!({
                "prompt": index,
                "path": prompt.path,
                "kind": prompt.kind,
                "file_md5": prompt.file_md5,
                "token_count": prompt.ids.len(),
                "full_token_count": prompt.full_token_count,
                "ids_md5": prompt.ids_md5,
            })
        })
        .collect();
    let mut meta = json!({
        "arm": args.arm,
        "binary": std::fs::read_link("/proc/self/exe").map(|p| p.display().to_string()).unwrap_or_default(),
        "binary_md5": md5_hex(&exe),
        "model_path": args.model,
        "model_sha256": model_sha,
        "env": env,
        "ranks": ranks,
        "devices": devices,
        "max_seq": args.max_seq,
        "tokens": args.tokens,
        "seed": args.seed,
        "vocab": vocab,
        "state_steps": args.state_steps,
        "moe_dump_positions": args.moe_dump_positions,
        "chunk_dump": args.chunk_dump,
        "perturb": args.perturb.map(|(rank, step)| json!({"rank": rank, "step": step})),
        "prompts": prompt_meta,
        "completed": false,
        "spec": args.mtp.as_ref().map(|mtp| json!({
            "kind": if args.dflash.is_some() { "dflash" } else { "mtp" },
            "draft": args.dflash.as_ref().map(|(draft, _)| draft),
            "dflash_block": args.dflash.as_ref().map(|(_, block)| block),
            "temp": mtp.temp,
            "top_p": mtp.top_p,
            "top_k": mtp.top_k,
            "seed": mtp.seed,
            "ngram": ngram,
            "state_every": mtp.state_every,
        })),
    });
    if args.canonical_state {
        meta["canonical_state"] = json!(true);
    }
    if let Some(gpus) = pp_gpus.as_ref() {
        let band_counts: Vec<_> = bundle
            .pp_band_states()
            .ok_or("PP mesh has no states")?
            .map(|state| state.scope_map().band().len())
            .collect();
        let edges: Vec<_> = (0..ranks - 1).map(|src| (src, src + 1)).collect();
        // Production admission has no pre-side-effect peer qualification;
        // resolve the same policy here (auto/host => host-mapped, peer refuses).
        let policy = hipfire_loader::admission::parse_mesh_transport(
            hipfire_config::developer_var(hipfire_loader::admission::MESH_TRANSPORT_ENV)
                .ok()
                .as_deref(),
        )?;
        let transports = hipfire_runtime::mesh_placement::resolve_edge_transports(
            policy,
            &|_, _| false,
            &edges,
        )?;
        let mut edge_meta = Vec::new();
        for edge in transports {
            let epoch = gpus
                .mesh_edge_epoch(hipfire_runtime::multi_gpu::transport::EdgeId {
                    src_device: edge.src,
                    dst_device: edge.dst,
                })
                .filter(|epoch| *epoch != 0)
                .ok_or("PP edge has no live transport lease")?;
            edge_meta.push(json!({
                "src": edge.src, "dst": edge.dst, "transport": edge.transport.to_string(),
                "epoch": epoch,
            }));
        }
        meta["pp"] = json!(pp);
        meta["band_counts"] = json!(band_counts);
        meta["rank_order"] = meta["devices"].clone();
        meta["edge_transports"] = json!(edge_meta);
        meta["output_owner"] = json!(gpus.mesh_output_owner());
        meta["numerical_policy"] = json!(hipfire_loader::admission::QWEN4_PP_NUMERICAL_POLICY);
        meta["placement"] = json!({
            "pp": pp,
            "band_counts": meta["band_counts"].clone(),
            "rank_order": meta["rank_order"].clone(),
            "edge_transports": meta["edge_transports"].clone(),
            "output_owner": meta["output_owner"].clone(),
        });
    }
    let write_meta = |meta: &Value| -> R<()> {
        std::fs::write(
            args.out.join("meta.json"),
            serde_json::to_vec_pretty(meta).map_err(e)?,
        )
        .map_err(e)
    };
    write_meta(&meta)?;
    // Reserve receipts (`HIPFIRE_EP_RESERVE_RECEIPT=1`): the oracle's half. The header repeats
    // the values meta.json carries (binary md5, model sha256, seed, mode, max_seq); the PLE
    // hash is not known to the oracle and is omitted in every arm. The forward prints the
    // per-rank records, keyed by the prompt scope set around each prompt below.
    let receipts = hipfire_config::developer_bool(RESERVE_RECEIPT_ENV, false);
    if receipts {
        reserve_receipt(&json!({
            "kind": "header",
            "binary_id": meta["binary_md5"].clone(),
            "model_sha256": meta["model_sha256"].clone(),
            "seed": meta["seed"].clone(),
            "mode": if args.mtp.is_some() { "mtp" } else { "ar" },
            "max_seq": meta["max_seq"].clone(),
        }));
    }

    let mut sinks = Sinks {
        logits: std::fs::File::create(args.out.join("logits.jsonl")).map_err(e)?,
        state: std::fs::File::create(args.out.join("state.jsonl")).map_err(e)?,
    };
    let last_dump = args.moe_dump_positions.iter().next_back().copied();
    let mut perturb_note = Value::Null;
    let mut steps_run = Vec::new();

    for (prompt_index, prompt) in prompts.iter().enumerate() {
        eprintln!(
            "[fn_ep_oracle] prompt {prompt_index}: {} ({} tokens)",
            prompt.path,
            prompt.ids.len()
        );
        hipfire_arch_qwen4::ep_reserve_receipt_scope(Some(prompt_index as u64));
        if let (Some(mtp), Some(speculator)) = (&args.mtp, speculator.as_mut()) {
            let request = SpecRequestConfig {
                temp: mtp.temp,
                top_p: mtp.top_p,
                top_k: mtp.top_k,
                rng_seed: mtp.seed,
                allow_ngram_modifier: ngram.unwrap_or(false),
                ..SpecRequestConfig::default()
            };
            let run = MtpRun {
                prompt_index,
                prompt,
                tokens: args.tokens,
                state_every: mtp.state_every,
                ranks,
                canonical_state: args.canonical_state,
            };
            let emitted = mtp_prompt(
                &mut gpu,
                pp_gpus.as_deref_mut(),
                bundle,
                speculator.as_mut(),
                request,
                &run,
                &mut sinks,
            )?;
            hipfire_arch_qwen4::ep_reserve_receipt_scope(None);
            if receipts {
                reserve_receipt(&json!({"kind": "ids", "prompt": prompt_index, "ids": emitted}));
            }
            steps_run.push(emitted.len());
            continue;
        }
        if let Some(gpus) = pp_gpus.as_deref_mut() {
            // Cold PP prefill resets every band and uses the canonical tile schedule.
            bundle
                .pp_prefill_final(gpus, &prompt.ids, 0, &logits)
                .map_err(e)?;
        } else {
            bundle.reset(&mut gpu).map_err(e)?;
            bundle
                .prefill_final(&mut gpu, &prompt.ids, 0, &logits)
                .map_err(e)?;
        }
        let mut completed = 0usize;
        let mut generated = Vec::new();
        for step in 0..=args.tokens {
            let token = if step == 0 {
                None
            } else {
                let forward = step - 1;
                if dump {
                    if Some(forward) > last_dump {
                        break;
                    }
                    if args.moe_dump_positions.contains(&forward) {
                        moe_dump::set_context(&args.arm, forward);
                    } else {
                        // Positions not listed would overwrite the last
                        // context's files; a dump run must be contiguous.
                        return Err(format!(
                            "MoE dump: decode forward {forward} is not in --moe-dump-positions \
                             (positions must be 0..=last, contiguous)"
                        ));
                    }
                }
                Some(if let Some(gpus) = pp_gpus.as_deref_mut() {
                    bundle
                        .pp_forward_token_or_argmax(gpus, None, &logits)
                        .map_err(e)?
                } else {
                    bundle
                        .forward_token_or_argmax(&mut gpu, None, &logits)
                        .map_err(e)?
                })
            };
            completed = step;
            if let (true, Some(token)) = (step >= 1, token) {
                generated.push(token);
            }

            gpu.bind_thread().map_err(e)?;
            let rank0 = tensor_bytes(&gpu, &logits)?;
            let rank0_sha = sha256_hex(&rank0);
            let mut record = json!({
                "kind": "logits",
                "prompt": prompt_index,
                "step": step,
                "token": token,
                "position": bundle.state.position,
                "logits_sha256": rank0_sha,
            });
            if pp_gpus.is_none() && ranks > 1 {
                let mut rank_sha = vec![rank0_sha.clone()];
                for rank in 1..ranks {
                    let peer_bytes =
                        bundle.with_rank_mut(&mut gpu, rank, |_, peer_gpu, mirrors| {
                            let mirror = mirrors
                                .logits()
                                .ok_or_else(|| format!("rank {rank} has no logits mirror"))?;
                            tensor_bytes(peer_gpu, mirror)
                        })?;
                    rank_sha.push(sha256_hex(&peer_bytes));
                }
                if rank_sha.iter().any(|sha| *sha != rank0_sha) {
                    eprintln!(
                        "[fn_ep_oracle] prompt {prompt_index} step {step}: RANK LOGITS DISAGREE {rank_sha:?}"
                    );
                }
                record["rank_logits_sha256"] = json!(rank_sha);
            }
            emit(&mut sinks.logits, &record)?;
            if args.canonical_state
                || pp_gpus.is_some()
                || step == 0
                || args.state_steps.contains(&step)
                || args.save_logits_steps.contains(&step)
            {
                std::fs::write(
                    args.out.join(format!("logits-p{prompt_index}-s{step}.f32")),
                    &rank0,
                )
                .map_err(e)?;
            }

            if args.state_steps.contains(&step) {
                let rank_rows = if let Some(gpus) = pp_gpus.as_deref_mut() {
                    pp_state_rows(bundle, gpus, args.canonical_state)?
                } else {
                    let mut rank_rows = Vec::new();
                    for rank in 0..ranks {
                        let rows = |bundle: &mut Qwen4Bundle, gpu: &mut Gpu| -> R<Vec<Value>> {
                            if args.canonical_state {
                                canonical_rows(
                                    bundle,
                                    vec![bundle.state.canonical_digest_rows(gpu).map_err(e)?],
                                )
                            } else {
                                Ok(bundle
                                    .state
                                    .debug_digest(gpu)
                                    .map_err(e)?
                                    .into_iter()
                                    .map(|(name, value)| json!([name, value]))
                                    .collect())
                            }
                        };
                        rank_rows.push(if rank == 0 {
                            rows(bundle, &mut gpu)?
                        } else {
                            bundle.with_rank_mut(&mut gpu, rank, |peer, peer_gpu, _| {
                                rows(peer, peer_gpu)
                            })?
                        });
                    }
                    rank_rows
                };
                for (rank, rows) in rank_rows.into_iter().enumerate() {
                    emit(
                        &mut sinks.state,
                        &json!({
                            "kind": "state",
                            "prompt": prompt_index,
                            "step": step,
                            "rank": rank,
                            "digest": rows,
                        }),
                    )?;
                }
            }

            if let Some((rank, perturb_step)) = args.perturb {
                if step == perturb_step {
                    perturb_note = if rank == 0 {
                        perturb_state(bundle, &mut gpu)?
                    } else {
                        bundle.with_rank_mut(&mut gpu, rank, |peer, peer_gpu, _| {
                            perturb_state(peer, peer_gpu)
                        })?
                    };
                    eprintln!(
                        "[fn_ep_oracle] prompt {prompt_index}: perturbed rank {rank} after step {step}: {perturb_note}"
                    );
                }
            }
            if step % 32 == 0 && step > 0 {
                eprintln!(
                    "[fn_ep_oracle] prompt {prompt_index}: step {step}/{}",
                    args.tokens
                );
            }
        }
        hipfire_arch_qwen4::ep_reserve_receipt_scope(None);
        if receipts {
            reserve_receipt(&json!({"kind": "ids", "prompt": prompt_index, "ids": generated}));
        }
        steps_run.push(completed);
    }

    meta["completed"] = json!(true);
    meta["steps_run"] = json!(steps_run);
    meta["perturb_applied"] = perturb_note;
    write_meta(&meta)?;
    if receipts {
        reserve_receipt(&json!({"kind": "end", "complete": true}));
    }
    eprintln!("[fn_ep_oracle] done: {}", args.out.display());
    Ok(())
}
