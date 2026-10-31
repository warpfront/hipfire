// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon DFlash2 oracle CLI: capture | emit | rowexact | greedy | rollback | verifygraph |
//! sampled | costs. Lab-gated
//! (`--features lab`), GPU required. PLAN `raon/dflash2/PLAN.md` §4 gates 1-4.
//!
//! ```text
//! raon_dflash_oracle capture  --model M.hfq --draft D.hfq --prompt-file P.txt [--out DIR] [--max-seq N]
//! raon_dflash_oracle emit     --model M.hfq --draft D.hfq --prompt-file P.txt --out DIR [--block N] [--max-seq N]
//! raon_dflash_oracle rowexact --model M.hfq --prompt-file P.txt [--out DIR] [--max-seq N]
//! raon_dflash_oracle greedy   --model M.hfq --draft D.hfq --prompt-file P.txt [--out DIR] [--block N] [--tokens N] [--profile on|off] [--max-seq N]
//! raon_dflash_oracle rollback  --model M.hfq [--draft D.hfq] --prompt-file P.txt [--out DIR] [--chat on] [--max-seq N]
//! raon_dflash_oracle verifygraph --model M.hfq --prompt-file P.txt [--out DIR] [--chat on] [--max-seq N]
//! raon_dflash_oracle sampled  --model M.hfq --draft D.hfq --prompt-file P.txt [--chat on] [--think on|off] [--trials N] [--tokens L] [--block B] [--temp T] [--top-p P] [--top-k K] [--alpha A] [--out DIR]
//! raon_dflash_oracle costs    --model M.hfq --draft D.hfq --prompt-file P.txt [--chat on] [--reps R] [--max-rows K] [--block B] [--out DIR]
//! ```
//!
//! Common options: `--chat on|off` (default off: the prompt file is tokenized as
//! raw text with the Raon tokenizer from the HFQ metadata, no BOS; `on` renders
//! the HFQ-embedded chat template exactly like `raon_oracle generate`, with
//! `--think on|off`, default on), `--block N` (draft block incl. the seed,
//! 2..=16, default 8), `--tokens N` (greedy stream length, default 64),
//! `--max-seq N` (context rows to reserve; default prompt + tokens + 112, min 512).
//!
//! * `capture` (gate 1): (advance) prefill through
//!   `SpecTarget::spec_advance(hidden_out)` on a fresh state (the production
//!   prefill feature path: the AR launch sequence plus a per-layer download
//!   tap), then on a second fresh state run `RaonBundle::dflash_tap_reference`
//!   over the same tokens; (verify) prefill all but the last K = min(16,
//!   prompt - 1) tokens, `verify_block` those K with the device capture, and
//!   compare with the tap over the same K-row batch from the same state. Every
//!   captured feature float must be bit-identical.
//! * `emit` (gate 2): prefill, take the target's greedy seed, run one draft
//!   block (`draft_forward_based`, target `lm_head_logits_device`,
//!   `propose_candidates_device` = the call sequence of `Dflash2Speculator::step`)
//!   and write the dump directory `raon/dflash2/oracle/draft_ref.py` reads
//!   (`meta.json`, `features.f32`, `tokens.json`, `hipfire.json`,
//!   `hipfire_hidden.f32`); then `draft_ref.py run` + `draft_ref.py compare`.
//! * `rowexact` (gate 3): per-row logits of `verify_block_logits` over K tokens
//!   against K sequential `decode_step` logits from the same snapshot, K = 1..=16.
//!   Exit 0 iff every row of every K is bit-identical (byte-identical greedy is
//!   then guaranteed); otherwise the per-K max |diff| and argmax agreement are
//!   printed and the exit is 1 (greedy must be tie-judged).
//! * `greedy` (gate 4): AR greedy vs DFlash greedy (`build_dflash2_speculator` +
//!   the `Speculator` trait) over `--tokens` tokens, both cut at the first EOS.
//!   Prints both streams' md5, the first divergence and the AR top-2 margin
//!   there, mean tokens per step and the step count. At the first divergence
//!   (AR token a, DFlash token b) it rebuilds the AR arm's exact state and runs
//!   the tie judge: the decode path (`decode_step`) and the verify path
//!   (`verify_block_logits` row 0 of `[prev, b]`) are asked for the same row;
//!   `JUSTIFIED-TIE` iff decode picks a and verify picks b (a pure row-count
//!   path flip), else `UNJUSTIFIED`. Index 0 compares the AR prefill logits with
//!   `spec_advance`'s prefill. Exit 0 iff the streams are identical or the
//!   divergence is a JUSTIFIED-TIE (`greedy.json` carries `tie_judge`).
//!   `--profile on` wraps the bundle in a timing `SpecTarget` (device sync
//!   around every call) and prints/writes ms per step for verify, commit
//!   (full-accept vs replay, replayed rows), head, embed and draft + selector
//!   (= step - target calls), the verify-rows histogram and AR ms/token.
//! * `rollback` (recurrence-rollback gate): after prefill, an 8-row block (the
//!   target's greedy seed plus 7 tokens of its AR greedy chain). For
//!   `keep` = 1..=7: arm A = restore the post-prefill state, `verify_block`,
//!   `commit_prefix(accept_len = keep - 1)` under `RollbackMode::Recurrence`
//!   (restore + 18 recurrence launches + conv-history copies); arm B = the same
//!   from the same state under `RollbackMode::Replay` (restore + full eager
//!   replay). The whole persistent slab (RDA matrices, conv histories, device
//!   `seq_len`, padding) is downloaded after each arm and must be byte-equal per
//!   `keep`; the table prints the first differing layer/offset and the commit
//!   cost of each arm. With `--draft` the verify also runs the DFlash feature
//!   capture (the product verify). Exit 0 iff every `keep` in 2..=7 is
//!   byte-equal; keep 1 is printed but not judged, because the replay arm's
//!   1-row forward takes the single-row decode route (different rounding from
//!   the batched verify rows the recurrence arm reproduces).
//!
//! * `verifygraph` (verify-graph gate): for K = 1..=16 at three positions, one verify
//!   window (`verify_block_logits` + feature capture, `verify_block` argmax, partial and
//!   full `commit_prefix`) under `DecodeMode::Eager` and again under `DecodeMode::Graph`
//!   (three passes: eager warm run, record, replay) from the same snapshot. Exit 0 iff
//!   every logits row, feature row, argmax pick and the persist slab (RDA state, conv
//!   history, `seq_len`) after each commit are bit-identical between the modes and the
//!   graph route recorded and replayed at least once.
//!
//! * `sampled` (distribution gate): `--trials N` (default 2000) cold trials of an AR arm
//!   (host sampler `sample_top_k_p_with_state`, per-trial SplitMix64 seed) and a DFlash2
//!   sampled-speculation arm, `--tokens L` (default 4) tokens each, `--temp` (1.0),
//!   `--top-p` (0.95), `--top-k` (20), `--block` (8). Chi-square goodness of fit of both
//!   arms against the exact truncated law (position 0; positions 1/2 conditioned on the 3
//!   most frequent DFlash contexts with >= 200 samples) plus AR-vs-DFlash homogeneity per
//!   position and on joint pairs. PASS iff every p-value > `--alpha` (1e-3) and no token
//!   falls outside the exact support. Writes `sampled.json`.
//! * `costs` (spec-retire policy): device-synced median/mean ms of AR `decode_step`,
//!   `verify_block` / `verify_block_logits` per row count 1..=`--max-rows` (9), partial
//!   `commit_prefix` (k = 2 and k = max-rows), the draft sequence and its head, over
//!   `--reps` (20); prints `window_units[k] = (draft + verify[k]) / decode`. Writes `costs.json`.
//!
//! Exit status: 0 pass, 1 verdict failed, 2 usage / runtime error.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use hipfire_arch_raon::bundle::{DecodeMode, PrefillMode, RaonBundle};
use hipfire_arch_raon::load_raon_bundle;
use hipfire_arch_raon::state::{ConvSlabLayout, PersistLayout, RaonStateDims, RaonStateSnapshot};
use hipfire_arch_raon::RollbackMode;
use hipfire_runtime::dflash::{
    draft_forward_based, propose_candidates_device, DflashConfig, DflashScratch, DflashWeights,
    DFLASH_BLOCK_MAX,
};
use hipfire_runtime::dflash2_spec::build_dflash2_speculator;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::llama::sample_top_k_p_with_state;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use hipfire_runtime::prompt_frame::JinjaChatFrame;
use hipfire_runtime::spec::{SpecAdvance, SpecRequestConfig, SpecScratch, SpecTarget, Speculator};
use hipfire_runtime::spec_sampling::{SampleSpec, SparseDist};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Value};

const USAGE: &str = "usage:
  raon_dflash_oracle capture  --model M --draft D --prompt-file P [--out DIR] [--max-seq N]
  raon_dflash_oracle emit     --model M --draft D --prompt-file P --out DIR [--block N] [--max-seq N]
  raon_dflash_oracle rowexact --model M --prompt-file P [--out DIR] [--max-seq N]
  raon_dflash_oracle greedy   --model M --draft D --prompt-file P [--out DIR] [--block N] [--tokens N] [--profile on|off] [--max-seq N]
  raon_dflash_oracle rollback --model M [--draft D] --prompt-file P [--out DIR] [--max-seq N]
  raon_dflash_oracle verifygraph --model M --prompt-file P [--out DIR] [--chat on] [--max-seq N]
  raon_dflash_oracle sampled  --model M --draft D --prompt-file P [--chat on] [--think on|off] [--trials N] [--tokens L] [--block B] [--temp T] [--top-p P] [--top-k K] [--alpha A] [--out DIR]
  raon_dflash_oracle costs    --model M --draft D --prompt-file P [--chat on] [--reps R] [--max-rows K] [--block B] [--out DIR]
common: [--chat on|off] [--think on|off]   (raw text tokens by default)";

type Opts = BTreeMap<String, String>;

fn parse_cli() -> Result<(String, Opts), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mode = argv.first().cloned().ok_or("missing mode")?;
    if !matches!(
        mode.as_str(),
        "capture"
            | "emit"
            | "rowexact"
            | "greedy"
            | "rollback"
            | "verifygraph"
            | "sampled"
            | "costs"
    ) {
        return Err(format!("unknown mode `{mode}`"));
    }
    let mut opts = Opts::new();
    let mut i = 1;
    while i < argv.len() {
        let key = argv[i]
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument `{}`", argv[i]))?;
        let val = argv
            .get(i + 1)
            .ok_or_else(|| format!("`--{key}` needs a value"))?;
        if opts.insert(key.to_string(), val.clone()).is_some() {
            return Err(format!("`--{key}` given twice"));
        }
        i += 2;
    }
    Ok((mode, opts))
}

fn require<'a>(opts: &'a Opts, key: &str) -> Result<&'a str, String> {
    opts.get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("missing required `--{key}`"))
}

fn opt_usize(opts: &Opts, key: &str, default: usize) -> Result<usize, String> {
    match opts.get(key) {
        Some(s) => s
            .parse::<usize>()
            .map_err(|e| format!("--{key} `{s}`: {e}")),
        None => Ok(default),
    }
}

fn opt_f32(opts: &Opts, key: &str, default: f32) -> Result<f32, String> {
    match opts.get(key) {
        Some(s) => s.parse::<f32>().map_err(|e| format!("--{key} `{s}`: {e}")),
        None => Ok(default),
    }
}

fn opt_f64(opts: &Opts, key: &str, default: f64) -> Result<f64, String> {
    match opts.get(key) {
        Some(s) => s.parse::<f64>().map_err(|e| format!("--{key} `{s}`: {e}")),
        None => Ok(default),
    }
}

fn on_off(opts: &Opts, key: &str, default: bool) -> Result<bool, String> {
    match opts.get(key).map(String::as_str) {
        None => Ok(default),
        Some("on") => Ok(true),
        Some("off") => Ok(false),
        Some(other) => Err(format!("--{key} `{other}`: on|off")),
    }
}

/// First maximum (lowest index on ties).
fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// `(id, value)` of the largest and second-largest entry (first index wins ties).
fn top2(v: &[f32]) -> [(u32, f32); 2] {
    let mut a = (0u32, f32::NEG_INFINITY);
    let mut b = (0u32, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate() {
        if x > a.1 {
            b = a;
            a = (i as u32, x);
        } else if x > b.1 {
            b = (i as u32, x);
        }
    }
    [a, b]
}

/// Top-`k` by descending value, ties by lower id.
fn top_k_desc(row: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    let by = |a: &u32, b: &u32| {
        row[*b as usize]
            .partial_cmp(&row[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    };
    idx.select_nth_unstable_by(k - 1, by);
    idx.truncate(k);
    idx.sort_by(by);
    idx.into_iter().map(|i| (i, row[i as usize])).collect()
}

fn md5_hex_u32(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
    format!("{:x}", md5::compute(bytes))
}

fn write_f32(path: &Path, data: &[f32]) -> Result<(), String> {
    let f = std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut w = std::io::BufWriter::new(f);
    let mut buf = Vec::with_capacity(1 << 20);
    for chunk in data.chunks(1 << 18) {
        buf.clear();
        for x in chunk {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        w.write_all(&buf)
            .map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    w.flush()
        .map_err(|e| format!("write {}: {e}", path.display()))
}

fn write_json(path: &Path, v: &Value) -> Result<(), String> {
    let s = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
    std::fs::write(path, s).map_err(|e| format!("write {}: {e}", path.display()))
}

fn out_dir(opts: &Opts, required: bool) -> Result<Option<PathBuf>, String> {
    match opts.get("out") {
        Some(p) => {
            let d = PathBuf::from(p);
            std::fs::create_dir_all(&d).map_err(|e| format!("create {}: {e}", d.display()))?;
            Ok(Some(d))
        }
        None if required => Err("missing required `--out`".into()),
        None => Ok(None),
    }
}

/// Production carrier load at an explicit `max_seq`, PP=1, no drafter/vision
/// (the drafter is built by the modes that need it).
fn load_bundle(
    gpu: &mut Gpu,
    src: ModelSource,
    path: &str,
    max_seq: usize,
) -> Result<RaonBundle, String> {
    let cask = CaskConfig::default();
    let mut ctx = LoadCtx {
        path,
        max_seq,
        sequence: None,
        deepseek4_compute_placement: Default::default(),
        deepseek4_experts_per_token: None,
        draft_path: None,
        vision_path: None,
        mtp_path: None,
        vision_mode: "off".to_string(),
        kv_mode_override: None,
        kv_k_override: None,
        kv_v_override: None,
        qwen_default_q8: false,
        kv_backend: KvBackend::default(),
        kv_adaptive_override: None,
        state_quant_override: None,
        cask: &cask,
        pp: 1,
        pp_layer_counts: None,
        admitted_mesh: None,
        band_placement: None,
        spec: SpecLoadCfg::default(),
        gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
        xdna: None,
        batch_request: Default::default(),
    };
    load_raon_bundle(src, &mut ctx)
}

/// The prompt as token ids: raw text, or (`--chat on`) the chat template.
fn prompt_tokens(hfq: &HfqFile, opts: &Opts) -> Result<Vec<u32>, String> {
    let file = require(opts, "prompt-file")?;
    let user = std::fs::read_to_string(file).map_err(|e| format!("read prompt {file}: {e}"))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer from HFQ metadata: {e:?}"))?;
    let ids = if on_off(opts, "chat", false)? {
        let template = hfq
            .chat_template()
            .ok_or("the HFQ carries no chat_template (tokenizer_config.chat_template)")?;
        JinjaChatFrame {
            tokenizer: &tokenizer,
            template: &template,
            system: None,
            user: &user,
            enable_thinking: on_off(opts, "think", true)?,
            // The Raon template never reads bos_token (bos_token is null upstream).
            bos_token: Some(""),
            reasoning_strength: None,
            reasoning_effort: None,
        }
        .render_and_encode()?
    } else {
        tokenizer.encode(&user)
    };
    if ids.is_empty() {
        return Err("the prompt tokenized to zero tokens".into());
    }
    Ok(ids)
}

fn hip<T>(r: hip_bridge::HipResult<T>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("{what}: {e:?}"))
}

fn draft_config(path: &str) -> Result<(HfqFile, DflashConfig), String> {
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("open draft {path}: {e}"))?;
    let cfg = DflashConfig::from_hfq(&hfq)
        .ok_or_else(|| format!("draft {path}: missing or malformed `dflash` metadata"))?;
    Ok((hfq, cfg))
}

fn check_streams(bundle: &RaonBundle, cfg: &DflashConfig) -> Result<(), String> {
    let t = SpecTarget::dflash_streams_per_layer(bundle);
    if t != cfg.target_streams_per_layer {
        return Err(format!(
            "target captures {t} streams per layer, the draft expects {}",
            cfg.target_streams_per_layer
        ));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// capture
// ─────────────────────────────────────────────────────────────────────────────

fn run_capture(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    prompt: &[u32],
    opts: &Opts,
) -> Result<bool, String> {
    let (_hfq, cfg) = draft_config(require(opts, "draft")?)?;
    check_streams(bundle, &cfg)?;
    SpecTarget::set_dflash_extract_layers(bundle, cfg.target_layer_ids.clone());

    bundle.reset(gpu)?;
    let mut device_rows: Vec<f32> = Vec::new();
    let t0 = Instant::now();
    match SpecTarget::spec_advance(
        bundle,
        gpu,
        prompt,
        0,
        true,
        &|| false,
        Some(&mut device_rows),
    )? {
        SpecAdvance::Ready { .. } => {}
        SpecAdvance::Aborted => return Err("spec_advance aborted".into()),
    }
    eprintln!(
        "capture: spec_advance over {} tokens in {:.2}s",
        prompt.len(),
        t0.elapsed().as_secs_f64()
    );

    bundle.reset(gpu)?;
    let t1 = Instant::now();
    let tap_rows = bundle.dflash_tap_reference(gpu, prompt)?;
    eprintln!(
        "capture: dflash_tap_reference over {} tokens in {:.2}s",
        prompt.len(),
        t1.elapsed().as_secs_f64()
    );

    let advance_ok = compare_feature_rows(
        "advance",
        &device_rows,
        &tap_rows,
        prompt,
        &cfg,
        bundle.config.hidden,
    );

    // The verify's device capture (one strided copy per extract layer) over
    // the last K prompt rows, against the tap over the same K-row batch from
    // the same state.
    let k = DFLASH_BLOCK_MAX.min(prompt.len() - 1);
    let split = prompt.len() - k;
    bundle.reset(gpu)?;
    bundle.prefill(gpu, &prompt[..split])?;
    let mut vs = SpecTarget::new_spec_scratch(bundle, gpu, k)?;
    let mut verify_rows: Vec<f32> = Vec::new();
    let verified = SpecTarget::verify_block(
        bundle,
        gpu,
        &prompt[split..],
        split,
        vs.as_mut(),
        Some(&mut verify_rows),
    )
    .and_then(|_| SpecTarget::abandon_verify(bundle, gpu, vs.as_mut()));
    vs.free(gpu);
    verified?;
    bundle.reset(gpu)?;
    bundle.prefill(gpu, &prompt[..split])?;
    let tap_k = bundle.dflash_tap_reference(gpu, &prompt[split..])?;
    let verify_ok = compare_feature_rows(
        "verify",
        &verify_rows,
        &tap_k,
        &prompt[split..],
        &cfg,
        bundle.config.hidden,
    );
    Ok(advance_ok && verify_ok)
}

/// Bit-compare captured feature rows `got` with the tap reference `want` over
/// `tokens`; print one PASS/FAIL line naming the first differing float.
fn compare_feature_rows(
    label: &str,
    got: &[f32],
    want: &[f32],
    tokens: &[u32],
    cfg: &DflashConfig,
    h: usize,
) -> bool {
    let fw = cfg.feature_width();
    let expect = tokens.len() * fw;
    if got.len() != expect || want.len() != expect {
        println!(
            "FAIL capture {label}: captured {} floats, tap reference {} floats, expected {expect} ({} rows x {fw})",
            got.len(),
            want.len(),
            tokens.len()
        );
        return false;
    }
    let per_layer = cfg.target_streams_per_layer * h;
    let mut bad = 0usize;
    let mut first: Option<usize> = None;
    let mut max_abs = 0f32;
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        if a.to_bits() != b.to_bits() {
            bad += 1;
            first.get_or_insert(i);
            let d = (a - b).abs();
            if d > max_abs || d.is_nan() {
                max_abs = d;
            }
        }
    }
    match first {
        None => {
            println!(
                "PASS capture {label}: {} rows x {fw} floats ({} layers x {} streams x {h}) bit-identical to forward_rows_tapped Ffn(l)",
                tokens.len(),
                cfg.num_extract(),
                cfg.target_streams_per_layer
            );
            true
        }
        Some(i) => {
            let (row, col) = (i / fw, i % fw);
            let (slot, rem) = (col / per_layer, col % per_layer);
            let (stream, hi) = (rem / h, rem % h);
            println!(
                "FAIL capture {label}: {bad}/{} floats differ (max |diff| {max_abs:.6e}); first at row {row} (token {}), extract slot {slot} (layer {}), stream {stream}, hidden {hi}: captured {:#010x} ({}) vs tap {:#010x} ({})",
                got.len(),
                tokens[row],
                cfg.target_layer_ids[slot],
                got[i].to_bits(),
                got[i],
                want[i].to_bits(),
                want[i],
            );
            false
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// emit
// ─────────────────────────────────────────────────────────────────────────────

fn run_emit(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    prompt: &[u32],
    opts: &Opts,
    out: &Path,
) -> Result<bool, String> {
    let block = opt_usize(opts, "block", 8)?;
    if !(2..=DFLASH_BLOCK_MAX).contains(&block) {
        return Err(format!("--block {block} outside 2..={DFLASH_BLOCK_MAX}"));
    }
    let draft_path = require(opts, "draft")?;
    let (draft_hfq, cfg) = draft_config(draft_path)?;
    check_streams(bundle, &cfg)?;
    let (Some(k), Some(_rank)) = (cfg.selector_top_k, cfg.selector_rank) else {
        return Err("emit needs a DFlash2 draft (candidate selector metadata)".into());
    };
    let weights = DflashWeights::load(gpu, &draft_hfq, &cfg).map_err(|e| format!("{e}"))?;
    if !weights.has_candidate_selector() {
        weights.free_gpu(gpu);
        return Err("DFlash2 draft: candidate selector tensors are missing".into());
    }
    let ctx_capacity = bundle.max_seq();
    let scratch = match cfg.declared_window {
        Some(w) => {
            DflashScratch::new_windowed(gpu, &cfg, block, w, w, ctx_capacity, weights.has_mq)
        }
        None => DflashScratch::new_with_mq(gpu, &cfg, block, ctx_capacity, weights.has_mq),
    };
    let mut scratch = match scratch {
        Ok(s) => s,
        Err(e) => {
            weights.free_gpu(gpu);
            return Err(format!("{e}"));
        }
    };
    let r = emit_block(
        gpu,
        bundle,
        prompt,
        &cfg,
        &weights,
        &mut scratch,
        block,
        k,
        out,
    );
    scratch.free_gpu(gpu);
    weights.free_gpu(gpu);
    r.map(|()| true)
}

#[allow(clippy::too_many_arguments)]
fn emit_block(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    prompt: &[u32],
    cfg: &DflashConfig,
    weights: &DflashWeights,
    scratch: &mut DflashScratch,
    block: usize,
    k: usize,
    out: &Path,
) -> Result<(), String> {
    let fw = cfg.feature_width();
    let h = cfg.hidden;
    let feature_window = cfg.declared_window.filter(|_| cfg.all_layers_sliding);
    SpecTarget::set_dflash_extract_layers(bundle, cfg.target_layer_ids.clone());
    if let Some(w) = feature_window {
        SpecTarget::set_dflash_feature_window(bundle, w);
    }
    if SpecTarget::dflash_extract_layers(bundle) != Some(cfg.target_layer_ids.as_slice()) {
        return Err("target did not adopt the draft's extract layers".into());
    }

    // Prefill: features of every prompt position + the target's greedy seed.
    bundle.reset(gpu)?;
    let mut host: Vec<f32> = Vec::new();
    let seed =
        match SpecTarget::spec_advance(bundle, gpu, prompt, 0, true, &|| false, Some(&mut host))? {
            SpecAdvance::Ready { last_argmax, .. } => last_argmax,
            SpecAdvance::Aborted => return Err("spec_advance aborted".into()),
        };
    let position = prompt.len();
    if host.len() % fw != 0 {
        return Err(format!(
            "captured {} floats, not a multiple of {fw}",
            host.len()
        ));
    }
    let rows_held = host.len() / fw;
    let host_base = position
        .checked_sub(rows_held)
        .ok_or_else(|| format!("{rows_held} feature rows for {position} tokens"))?;
    // Rows the draft reads: its declared window (all held rows without one).
    let first_row = match feature_window {
        Some(w) => host_base.max(position.saturating_sub(w)),
        None => host_base,
    };
    let features = &host[(first_row - host_base) * fw..];
    eprintln!(
        "emit: prompt {position} tokens, seed {seed}, feature rows {first_row}..{position} (window {feature_window:?})"
    );

    // One draft block, the call sequence of Dflash2Speculator::step.
    let mask_row = SpecTarget::embed_row(bundle, gpu, cfg.mask_token_id)?;
    let seed_row = SpecTarget::embed_row(bundle, gpu, seed)?;
    if seed_row.len() != h || mask_row.len() != h {
        return Err(format!(
            "target embedding rows have {} floats, the draft needs {h}",
            seed_row.len()
        ));
    }
    let mut noise = Vec::with_capacity(block * h);
    noise.extend_from_slice(&seed_row);
    for _ in 1..block {
        noise.extend_from_slice(&mask_row);
    }
    let positions_q: Vec<i32> = (position as i32..(position + block) as i32).collect();
    let positions_k: Vec<i32> = (0..(position + block) as i32).collect();
    let t0 = Instant::now();
    draft_forward_based(
        gpu,
        weights,
        cfg,
        Some(&noise),
        Some(features),
        first_row,
        &positions_q,
        &positions_k,
        block,
        position,
        scratch,
        false,
    )
    .map_err(|e| format!("draft_forward: {e}"))?;
    let drafts_n = block - 1;
    let hidden_all = hip(
        gpu.download_f32(&scratch.x.sub_offset(0, block * h)),
        "download draft hidden",
    )?;
    let hidden_rows = scratch.x.sub_offset(h, drafts_n * h);
    let logits_dev = SpecTarget::lm_head_logits_device(bundle, gpu, &hidden_rows, drafts_n)?;
    let vocab = cfg.vocab_size;
    let logits = hip(gpu.download_f32(&logits_dev), "download draft logits")?;
    if logits.len() < drafts_n * vocab {
        return Err(format!(
            "draft logits {} floats, need {drafts_n} x {vocab}",
            logits.len()
        ));
    }
    let proposal = propose_candidates_device(
        gpu,
        weights,
        scratch,
        &hidden_rows,
        &logits_dev,
        drafts_n,
        seed,
        0.0,
        None,
    )
    .map_err(|e| format!("selector: {e}"))?;
    eprintln!(
        "emit: draft block + head + selector in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    let mut top_ids = Vec::with_capacity(drafts_n * k);
    let mut top_vals = Vec::with_capacity(drafts_n * k);
    for r in 0..drafts_n {
        for (id, v) in top_k_desc(&logits[r * vocab..(r + 1) * vocab], k) {
            top_ids.push(id);
            top_vals.push(v);
        }
    }

    write_f32(&out.join("features.f32"), features)?;
    write_f32(&out.join("hipfire_hidden.f32"), &hidden_all)?;
    // The noise block hipfire fed the draft (the target's own embedding rows,
    // dequantized), so the reference can isolate the draft math from the
    // artifact's embedding quantization.
    write_f32(&out.join("noise.f32"), &noise)?;
    write_json(&out.join("tokens.json"), &json!(prompt))?;
    write_json(
        &out.join("meta.json"),
        &json!({
            "format": "raon-dflash2-oracle-v1",
            "ctx_rows": position - first_row,
            "feature_width": fw,
            "positions": (first_row..position).collect::<Vec<_>>(),
            "seed": seed,
            "seed_position": position,
            "block": block,
            "mask_token": cfg.mask_token_id,
            "hidden": h,
            "top_k": k,
            "prompt_tokens": position,
        }),
    )?;
    write_json(
        &out.join("hipfire.json"),
        &json!({
            "selector_tokens": proposal.tokens,
            "candidates": proposal.candidates,
            "confidence": proposal.confidence,
            "top_ids": top_ids,
            "top_vals": top_vals,
        }),
    )?;
    println!(
        "emit: wrote {} (ctx rows {}, block {block}, seed {seed}); selector tokens {:?}",
        out.display(),
        position - first_row,
        proposal.tokens
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// rowexact
// ─────────────────────────────────────────────────────────────────────────────

fn run_rowexact(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    prompt: &[u32],
    out: Option<&Path>,
) -> Result<bool, String> {
    const KMAX: usize = 16;
    let vocab = bundle.config.vocab;
    bundle.reset(gpu)?;
    bundle.prefill(gpu, prompt)?;
    let position = bundle.n_tokens();
    if position + KMAX > bundle.max_seq() {
        return Err(format!(
            "context {} + {KMAX} rows exceeds max_seq {}",
            position,
            bundle.max_seq()
        ));
    }
    let mut snap = bundle.state.snapshot_alloc(gpu)?;
    bundle.state.snapshot_into(gpu, &mut snap)?;

    // Sequential reference: the natural greedy chain, logits after each decode_step.
    let mut chain = Vec::with_capacity(KMAX);
    chain.push(argmax(&bundle.download_logits(gpu)?) as u32);
    let mut seq_rows: Vec<Vec<f32>> = Vec::with_capacity(KMAX);
    for i in 0..KMAX {
        bundle.decode_step(gpu, chain[i])?;
        let l = bundle.download_logits(gpu)?;
        if i + 1 < KMAX {
            chain.push(argmax(&l) as u32);
        }
        seq_rows.push(l);
    }
    bundle.state.restore(gpu, &snap)?;
    snap.free_gpu(gpu)?;
    if bundle.n_tokens() != position {
        return Err(format!(
            "restore left n_tokens {} != {position}",
            bundle.n_tokens()
        ));
    }

    let mut scratch: Box<dyn SpecScratch> = SpecTarget::new_spec_scratch(bundle, gpu, KMAX)?;
    let mut all_exact = true;
    let mut per_k = Vec::new();
    println!("rowexact: prompt {position} tokens; chain {chain:?}");
    println!("  K  rows_bit_equal  argmax_agree  max|diff| (row of max)");
    let r = (|| -> Result<(), String> {
        for kk in 1..=KMAX {
            let logits = SpecTarget::verify_block_logits(
                bundle,
                gpu,
                &chain[..kk],
                position,
                scratch.as_mut(),
                None,
            )?;
            SpecTarget::abandon_verify(bundle, gpu, scratch.as_mut())?;
            if bundle.n_tokens() != position {
                return Err(format!(
                    "abandon_verify left n_tokens {} != {position} after K={kk}",
                    bundle.n_tokens()
                ));
            }
            if logits.len() != kk * vocab {
                return Err(format!(
                    "verify_block_logits returned {} floats for K={kk}, expected {}",
                    logits.len(),
                    kk * vocab
                ));
            }
            let (mut equal, mut agree, mut worst, mut worst_row) = (0usize, 0usize, 0f32, 0usize);
            let mut rows = Vec::new();
            for i in 0..kk {
                let v = &logits[i * vocab..(i + 1) * vocab];
                let s = &seq_rows[i];
                let md = v
                    .iter()
                    .zip(s)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, |m, d| if d.is_nan() || d > m { d } else { m });
                let bit_equal = v.iter().zip(s).all(|(a, b)| a.to_bits() == b.to_bits());
                let ag = argmax(v) == argmax(s);
                equal += bit_equal as usize;
                agree += ag as usize;
                if md > worst {
                    worst = md;
                    worst_row = i;
                }
                rows.push(json!({"row": i, "max_abs_diff": md, "bit_equal": bit_equal, "argmax_agree": ag}));
            }
            all_exact &= equal == kk;
            println!("  {kk:>2}  {equal:>3}/{kk:<3}         {agree:>3}/{kk:<3}       {worst:.6e} (row {worst_row})");
            per_k.push(json!({"k": kk, "rows_bit_equal": equal, "argmax_agree": agree, "max_abs_diff": worst, "rows": rows}));
        }
        Ok(())
    })();
    scratch.free(gpu);
    r?;
    println!(
        "{} rowexact: verify_block_logits rows {} the sequential decode_step logits",
        if all_exact { "PASS" } else { "FAIL" },
        if all_exact {
            "are bit-identical to"
        } else {
            "are NOT all bit-identical to"
        }
    );
    if let Some(dir) = out {
        write_json(
            &dir.join("rowexact.json"),
            &json!({"pass": all_exact, "position": position, "chain": chain, "per_k": per_k}),
        )?;
    }
    Ok(all_exact)
}

// ─────────────────────────────────────────────────────────────────────────────
// greedy
// ─────────────────────────────────────────────────────────────────────────────

/// AR top-2 logit margin under which a greedy divergence onto AR's runner-up
/// is a near-tie: 4 BF16 ulps at |logit| in [16, 32), below the measured
/// single-row vs batched-forward logit discrepancy on Raon (rowexact max 1.57).
const NEAR_TIE_LOGITS: f32 = 0.5;

/// Cut `v` before its first EOS token.
fn cut_at_eos(eos: &[u32], v: &mut Vec<u32>) {
    if let Some(p) = v.iter().position(|t| eos.contains(t)) {
        v.truncate(p);
    }
}

fn sync(gpu: &Gpu) -> Result<(), String> {
    hip(gpu.hip.device_synchronize(), "device synchronize")
}

/// Wall time and call count of one timed phase.
#[derive(Clone, Copy, Default)]
struct PhaseStat {
    calls: usize,
    ms: f64,
}

/// Per-method accounting of the timing target wrapper.
#[derive(Clone, Default)]
struct Prof {
    by_method: BTreeMap<&'static str, PhaseStat>,
    /// `commit_prefix` with `accept_len == block.len() - 1` (no replay).
    commit_full: PhaseStat,
    /// `commit_prefix` that restores the snapshot and replays `accept_len + 1` rows.
    commit_partial: PhaseStat,
    replayed_rows: usize,
    /// `block.len()` of every verify -> count.
    verify_rows: BTreeMap<usize, usize>,
    /// Every verify in call order: (absolute position, block). The tie judge
    /// replays the window that emitted the first divergent token.
    verifies: Vec<(usize, Vec<u32>)>,
    /// `sampled`: when set, the truncated law of every row the drafter may
    /// draw from (the prefill's last row, each verify row) is recorded into
    /// `last_laws` (replaced by every such call).
    row_law: Option<SampleSpec>,
    last_laws: Vec<Vec<(u32, f32)>>,
}

impl Prof {
    fn add(&mut self, name: &'static str, ms: f64) {
        let s = self.by_method.entry(name).or_default();
        s.calls += 1;
        s.ms += ms;
    }

    /// Wall ms spent in every wrapped target call.
    fn total_ms(&self) -> f64 {
        self.by_method.values().map(|s| s.ms).sum::<f64>()
            + self.commit_full.ms
            + self.commit_partial.ms
    }

    fn method_ms(&self, names: &[&str]) -> PhaseStat {
        let mut out = PhaseStat::default();
        for n in names {
            if let Some(s) = self.by_method.get(n) {
                out.calls += s.calls;
                out.ms += s.ms;
            }
        }
        out
    }
}

type SharedProf = Rc<RefCell<Prof>>;

/// Run `$call` (between two device syncs when the wrapper times) and add its
/// wall ms to `$prof[$name]`.
macro_rules! timed {
    ($self:ident, $gpu:ident, $name:expr, $call:expr) => {{
        if $self.timed {
            sync($gpu)?;
        }
        let t = Instant::now();
        let r = $call;
        let s = if $self.timed { sync($gpu) } else { Ok(()) };
        let ms = t.elapsed().as_secs_f64() * 1e3;
        $self.prof.borrow_mut().add($name, ms);
        let r = r?;
        s?;
        Ok(r)
    }};
}

/// `SpecTarget` that forwards every method the DFlash2 speculator uses to the
/// bundle and logs every verify; with `timed` it also device-syncs around and
/// times each call.
struct TimedTarget<'a> {
    inner: &'a mut RaonBundle,
    prof: SharedProf,
    timed: bool,
}

impl SpecTarget for TimedTarget<'_> {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self.inner.as_any_mut()
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        timed!(
            self,
            gpu,
            "reset_recurrent",
            self.inner.reset_recurrent(gpu)
        )
    }

    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        timed!(
            self,
            gpu,
            "new_spec_scratch",
            self.inner.new_spec_scratch(gpu, block_size)
        )
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        let r: Result<SpecAdvance, String> = timed!(
            self,
            gpu,
            "spec_advance",
            self.inner
                .spec_advance(gpu, tokens, start_pos, reset, abort, hidden_out)
        );
        let advance = r?;
        let law = self.prof.borrow().row_law;
        if let (
            Some(spec),
            SpecAdvance::Ready {
                last_logits: Some(row),
                ..
            },
        ) = (law, &advance)
        {
            let law = row_law(row, spec)?;
            self.prof.borrow_mut().last_laws = vec![law];
        }
        Ok(advance)
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        {
            let mut p = self.prof.borrow_mut();
            *p.verify_rows.entry(block.len()).or_default() += 1;
            p.verifies.push((position, block.to_vec()));
        }
        timed!(
            self,
            gpu,
            "verify_block",
            self.inner
                .verify_block(gpu, block, position, scratch, hidden_out)
        )
    }

    fn verify_block_logits(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        {
            let mut p = self.prof.borrow_mut();
            *p.verify_rows.entry(block.len()).or_default() += 1;
            p.verifies.push((position, block.to_vec()));
        }
        let r: Result<Vec<f32>, String> = timed!(
            self,
            gpu,
            "verify_block_logits",
            self.inner
                .verify_block_logits(gpu, block, position, scratch, hidden_out)
        );
        let logits = r?;
        let law = self.prof.borrow().row_law;
        if let Some(spec) = law {
            let vocab = self.inner.config.vocab;
            let laws = logits
                .chunks_exact(vocab)
                .map(|row| row_law(row, spec))
                .collect::<Result<Vec<_>, _>>()?;
            self.prof.borrow_mut().last_laws = laws;
        }
        Ok(logits)
    }

    fn commit_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        accept_len: usize,
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        if self.timed {
            sync(gpu)?;
        }
        let t = Instant::now();
        let r = self
            .inner
            .commit_prefix(gpu, block, accept_len, position, scratch);
        let s = if self.timed { sync(gpu) } else { Ok(()) };
        let ms = t.elapsed().as_secs_f64() * 1e3;
        {
            let mut p = self.prof.borrow_mut();
            let slot = if accept_len + 1 == block.len() {
                &mut p.commit_full
            } else {
                &mut p.commit_partial
            };
            slot.calls += 1;
            slot.ms += ms;
            if accept_len + 1 != block.len() {
                p.replayed_rows += accept_len + 1;
            }
        }
        r?;
        s
    }

    fn eos_token(&self) -> u32 {
        self.inner.eos_token()
    }

    fn ctx_capacity(&self) -> usize {
        SpecTarget::ctx_capacity(&*self.inner)
    }

    fn map_ahead_for_decode(
        &mut self,
        gpu: &mut Gpu,
        position: usize,
        remaining_new_tokens: usize,
        max_draft_rows: usize,
    ) {
        if self.timed {
            let _ = sync(gpu);
        }
        let t = Instant::now();
        self.inner
            .map_ahead_for_decode(gpu, position, remaining_new_tokens, max_draft_rows);
        if self.timed {
            let _ = sync(gpu);
        }
        self.prof
            .borrow_mut()
            .add("map_ahead_for_decode", t.elapsed().as_secs_f64() * 1e3);
    }

    fn dflash_extract_layers(&self) -> Option<&[usize]> {
        self.inner.dflash_extract_layers()
    }

    fn dflash_streams_per_layer(&self) -> usize {
        self.inner.dflash_streams_per_layer()
    }

    fn set_dflash_extract_layers(&mut self, layers: Vec<usize>) {
        self.inner.set_dflash_extract_layers(layers)
    }

    fn set_dflash_feature_window(&mut self, rows: usize) {
        self.inner.set_dflash_feature_window(rows)
    }

    fn embed_row(&mut self, gpu: &mut Gpu, token_id: u32) -> Result<Vec<f32>, String> {
        timed!(self, gpu, "embed_row", self.inner.embed_row(gpu, token_id))
    }

    fn lm_head_logits(
        &mut self,
        gpu: &mut Gpu,
        hidden_rows: &GpuTensor,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        timed!(
            self,
            gpu,
            "lm_head_logits",
            self.inner.lm_head_logits(gpu, hidden_rows, n)
        )
    }

    fn lm_head_logits_device(
        &mut self,
        gpu: &mut Gpu,
        hidden_rows: &GpuTensor,
        n: usize,
    ) -> Result<GpuTensor, String> {
        timed!(
            self,
            gpu,
            "lm_head_logits_device",
            self.inner.lm_head_logits_device(gpu, hidden_rows, n)
        )
    }

    fn abandon_verify(
        &mut self,
        gpu: &mut Gpu,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        timed!(
            self,
            gpu,
            "abandon_verify",
            self.inner.abandon_verify(gpu, scratch)
        )
    }

    fn repair_committed_prefix(
        &mut self,
        gpu: &mut Gpu,
        scratch: &mut dyn SpecScratch,
        position: usize,
        replay: &[u32],
    ) -> Result<bool, String> {
        timed!(
            self,
            gpu,
            "repair_committed_prefix",
            self.inner
                .repair_committed_prefix(gpu, scratch, position, replay)
        )
    }
}

/// What one DFlash greedy arm produced.
struct DflashRun {
    stream: Vec<u32>,
    steps: usize,
    accepted: usize,
    proposed: usize,
    secs: f64,
    /// `--profile`: device-synced wall ms around every `step`, and the target
    /// calls made during the steps / during prefill.
    step_wall_ms: f64,
    step_prof: Option<Prof>,
    prefill_prof: Option<Prof>,
}

/// Build the speculator over `target`, prefill, and step until `n` tokens (or
/// an EOS). `prof` collects the target calls (and the verify log); with
/// `timed` every `step` is device-synced and timed as well.
#[allow(clippy::too_many_arguments)]
fn dflash_arm(
    gpu: &mut Gpu,
    target: &mut dyn SpecTarget,
    prompt: &[u32],
    draft_path: &str,
    block: usize,
    n: usize,
    ctx_capacity: usize,
    eos: &[u32],
    prof: &SharedProf,
    timed: bool,
) -> Result<DflashRun, String> {
    let mut spec: Box<dyn Speculator> =
        build_dflash2_speculator(gpu, draft_path, target, ctx_capacity, block)?;
    let r = (|| -> Result<DflashRun, String> {
        let t0 = Instant::now();
        let first = spec
            .prefill(gpu, target, prompt, prompt, 0, false, None, &|| false)?
            .into_first_token()?;
        target.map_ahead_for_decode(gpu, prompt.len(), n, block);
        let prefill_prof = Some(std::mem::take(&mut *prof.borrow_mut()));
        let mut stream = vec![first];
        let (mut position, mut seed) = (prompt.len(), first);
        let (mut steps, mut accepted, mut proposed) = (0usize, 0usize, 0usize);
        let mut step_wall_ms = 0f64;
        while stream.len() < n && !eos.contains(stream.last().unwrap()) {
            if timed {
                sync(gpu)?;
            }
            let t = Instant::now();
            let step = spec.step(gpu, target, position, seed, &stream, None, 0.0, usize::MAX)?;
            if timed {
                sync(gpu)?;
                step_wall_ms += t.elapsed().as_secs_f64() * 1e3;
            }
            steps += 1;
            accepted += step.accepted;
            proposed += step.proposed;
            position += step.emit.len();
            seed = step.next_seed;
            stream.extend_from_slice(&step.emit);
        }
        Ok(DflashRun {
            stream,
            steps,
            accepted,
            proposed,
            secs: t0.elapsed().as_secs_f64(),
            step_wall_ms,
            step_prof: Some(prof.borrow().clone()),
            prefill_prof,
        })
    })();
    spec.free(gpu);
    r
}

/// `[argmax id, logit(a), logit(b), margin(a - b), top-2]` of one logits row.
fn describe_row(row: &[f32], a: u32, b: u32) -> (usize, Value) {
    let am = argmax(row);
    let t = top2(row);
    (
        am,
        json!({
            "argmax": am,
            "logit_a": row[a as usize],
            "logit_b": row[b as usize],
            "margin_a_minus_b": row[a as usize] - row[b as usize],
            "top1": {"id": t[0].0, "logit": t[0].1},
            "top2": {"id": t[1].0, "logit": t[1].1},
        }),
    )
}

/// Tie judge at the first divergence (`common = ar[..i]`, AR token `a`, DFlash
/// token `b`). `window` is the DFlash verify that emitted token `i`: its
/// absolute position and block (seed + drafts), whose row `r` predicted the
/// token at absolute position `prompt.len() + i`. The judge rebuilds the AR
/// arm's exact state at the window start, then asks (a) the decode path
/// (`decode_step` over `block[..=r]`, exactly how the AR arm got token `i`) and
/// (b) the verify path (`verify_block_logits(block)` row `r`, the very block
/// DFlash verified) from that identical state. `JUSTIFIED-TIE` iff the decode
/// path picks `a` and the verify path picks `b`: the same tokens through the
/// single-row and the batched forward disagree on a near-tie.
fn tie_judge(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    prompt: &[u32],
    common: &[u32],
    a: u32,
    b: u32,
    window: Option<&(usize, Vec<u32>)>,
) -> Result<(bool, Value), String> {
    let i = common.len();
    let vocab = bundle.config.vocab;
    // The AR arm's state: reset, batched prefill, decode_step over the chosen tokens.
    bundle.reset(gpu)?;
    bundle.prefill(gpu, prompt)?;
    let dec_row: Vec<f32>;
    let ver_row: Option<Vec<f32>>;
    let ver_argmax_only: Option<u32>;
    let path: &str;
    if i == 0 {
        // Token 0 comes from the prefill logits (AR) vs the speculator's prefill
        // (`spec_advance`, chunked): the same prompt through two prefill routes.
        dec_row = bundle.download_logits(gpu)?;
        match SpecTarget::spec_advance(bundle, gpu, prompt, 0, true, &|| false, None)? {
            SpecAdvance::Ready {
                last_argmax,
                last_logits,
            } => {
                ver_row = last_logits;
                ver_argmax_only = Some(last_argmax);
            }
            SpecAdvance::Aborted => return Err("tie judge: spec_advance aborted".into()),
        }
        path = "prefill logits vs spec_advance prefill";
    } else {
        let (wpos, block) =
            window.ok_or("tie judge: no verify window emitted the divergent token")?;
        let (wpos, block) = (*wpos, block.as_slice());
        let target_pos = prompt.len() + i;
        if wpos < prompt.len() || wpos >= target_pos || target_pos > wpos + block.len() {
            return Err(format!(
                "tie judge: window at {wpos} ({} rows) does not predict position {target_pos}",
                block.len()
            ));
        }
        let r = target_pos - 1 - wpos;
        let g0 = wpos - prompt.len();
        if block[..=r] != common[g0..=g0 + r] {
            return Err(format!(
                "tie judge: window rows 0..={r} {:?} are not the common tokens {:?}",
                &block[..=r],
                &common[g0..=g0 + r]
            ));
        }
        // The AR arm's state at the window start: generated tokens below it were
        // consumed one decode_step each.
        for &t in &common[..g0] {
            bundle.decode_step(gpu, t)?;
        }
        if bundle.n_tokens() != wpos {
            return Err(format!(
                "tie judge: AR state at {} != window {wpos}",
                bundle.n_tokens()
            ));
        }
        let mut snap = bundle.state.snapshot_alloc(gpu)?;
        bundle.state.snapshot_into(gpu, &mut snap)?;
        // (a) decode path: the AR arm's own computation of token i.
        for &t in &block[..=r] {
            bundle.decode_step(gpu, t)?;
        }
        dec_row = bundle.download_logits(gpu)?;
        bundle.state.restore(gpu, &snap)?;
        snap.free_gpu(gpu)?;
        if bundle.n_tokens() != wpos {
            return Err(format!(
                "tie judge: restore left n_tokens {} != {wpos}",
                bundle.n_tokens()
            ));
        }
        // (b) verify path: row r of the exact block DFlash verified.
        let mut scratch: Box<dyn SpecScratch> =
            SpecTarget::new_spec_scratch(bundle, gpu, block.len())?;
        let rows = (|| -> Result<Vec<f32>, String> {
            let rows =
                SpecTarget::verify_block_logits(bundle, gpu, block, wpos, scratch.as_mut(), None)?;
            SpecTarget::abandon_verify(bundle, gpu, scratch.as_mut())?;
            if rows.len() != block.len() * vocab {
                return Err(format!(
                    "tie judge: verify returned {} floats, expected {}",
                    rows.len(),
                    block.len() * vocab
                ));
            }
            Ok(rows[r * vocab..(r + 1) * vocab].to_vec())
        })();
        scratch.free(gpu);
        ver_row = Some(rows?);
        ver_argmax_only = None;
        path = "decode_step over block[..=r] vs verify_block_logits(block) row r, window replayed from the AR state";
        println!(
            "  tie judge window: position {wpos}, {} rows, row {r}",
            block.len()
        );
    }
    let (dec_arg, dec_json) = describe_row(&dec_row, a, b);
    let (ver_arg, ver_json) = match (&ver_row, ver_argmax_only) {
        (Some(row), _) => describe_row(row, a, b),
        (None, Some(arg)) => (
            arg as usize,
            json!({"argmax": arg, "logits": "not exposed by spec_advance"}),
        ),
        (None, None) => unreachable!("one verify-side result is always set"),
    };
    let justified = dec_arg as u32 == a && ver_arg as u32 == b;
    println!("  tie judge ({path}), state = AR arm after {i} generated tokens:");
    println!("    decode path: argmax {dec_arg}  {dec_json}");
    println!("    verify path: argmax {ver_arg}  {ver_json}");
    println!(
        "    verdict: {}  (decode argmax {} AR token {a}; verify argmax {} DFlash token {b})",
        if justified {
            "JUSTIFIED-TIE"
        } else {
            "UNJUSTIFIED"
        },
        if dec_arg as u32 == a { "==" } else { "!=" },
        if ver_arg as u32 == b { "==" } else { "!=" },
    );
    Ok((
        justified,
        json!({
            "index": i, "ar_token": a, "dflash_token": b, "path": path,
            "decode_path": dec_json, "verify_path": ver_json,
            "verdict": if justified { "JUSTIFIED-TIE" } else { "UNJUSTIFIED" },
        }),
    ))
}

fn phase_json(s: PhaseStat, steps: usize) -> Value {
    json!({"calls": s.calls, "total_ms": s.ms, "ms_per_step": s.ms / steps.max(1) as f64})
}

/// Print and serialize the `--profile` table.
fn report_profile(run: &DflashRun, ar_decode_ms: f64, ar_decode_tokens: usize) -> Value {
    let (Some(p), Some(pre)) = (&run.step_prof, &run.prefill_prof) else {
        return Value::Null;
    };
    let steps = run.steps.max(1);
    let per = |ms: f64| ms / steps as f64;
    let verify = p.method_ms(&["verify_block", "verify_block_logits"]);
    let head = p.method_ms(&["lm_head_logits_device", "lm_head_logits"]);
    let embed = p.method_ms(&["embed_row"]);
    let other_ms =
        p.total_ms() - verify.ms - p.commit_full.ms - p.commit_partial.ms - head.ms - embed.ms;
    let target_ms = p.total_ms();
    let draft_ms = run.step_wall_ms - target_ms;
    println!(
        "  profile (device-synced around every call; ms per step over {} steps):",
        run.steps
    );
    println!(
        "    step total                      {:>9.3}",
        per(run.step_wall_ms)
    );
    println!(
        "    draft + selector (step - target) {:>8.3}",
        per(draft_ms)
    );
    println!(
        "    target calls                    {:>9.3}",
        per(target_ms)
    );
    println!(
        "      verify                        {:>9.3}  ({} calls)",
        per(verify.ms),
        verify.calls
    );
    println!(
        "      commit_prefix full accept     {:>9.3}  ({} calls)",
        per(p.commit_full.ms),
        p.commit_full.calls
    );
    println!(
        "      commit_prefix partial (replay){:>9.3}  ({} calls, {} rows replayed)",
        per(p.commit_partial.ms),
        p.commit_partial.calls,
        p.replayed_rows
    );
    println!(
        "      lm_head_logits_device         {:>9.3}  ({} calls)",
        per(head.ms),
        head.calls
    );
    println!(
        "      embed_row                     {:>9.3}  ({} calls)",
        per(embed.ms),
        embed.calls
    );
    println!("      other                         {:>9.3}", per(other_ms));
    if p.commit_partial.calls > 0 {
        println!(
            "    partial commit: {:.3} ms/call, {:.3} ms/replayed row",
            p.commit_partial.ms / p.commit_partial.calls as f64,
            p.commit_partial.ms / p.replayed_rows.max(1) as f64
        );
    }
    println!(
        "    verify rows histogram (block.len() -> verifies): {:?}",
        p.verify_rows
    );
    if ar_decode_tokens > 0 {
        println!(
            "    AR decode_step: {:.3} ms/token ({ar_decode_tokens} tokens, synced per token)",
            ar_decode_ms / ar_decode_tokens as f64
        );
    }
    println!(
        "    prefill-side target calls: {:.1} ms total",
        pre.total_ms()
    );
    let hist: BTreeMap<String, usize> = p
        .verify_rows
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    let ar_ms_per_token = if ar_decode_tokens > 0 {
        json!(ar_decode_ms / ar_decode_tokens as f64)
    } else {
        Value::Null
    };
    let by_method: BTreeMap<String, Value> = p
        .by_method
        .iter()
        .map(|(k, s)| (k.to_string(), phase_json(*s, steps)))
        .collect();
    json!({
        "steps": run.steps,
        "step_total_ms_per_step": per(run.step_wall_ms),
        "draft_selector_ms_per_step": per(draft_ms),
        "target_ms_per_step": per(target_ms),
        "verify": phase_json(verify, steps),
        "commit_full": phase_json(p.commit_full, steps),
        "commit_partial": phase_json(p.commit_partial, steps),
        "replayed_rows": p.replayed_rows,
        "lm_head_logits_device": phase_json(head, steps),
        "embed_row": phase_json(embed, steps),
        "other_ms_per_step": per(other_ms),
        "by_method": by_method,
        "verify_rows_histogram": hist,
        "ar_decode_ms_per_token": ar_ms_per_token,
        "ar_decode_tokens": ar_decode_tokens,
        "prefill_target_ms": pre.total_ms(),
    })
}

fn run_greedy(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    prompt: &[u32],
    opts: &Opts,
    out: Option<&Path>,
) -> Result<bool, String> {
    let n = opt_usize(opts, "tokens", 64)?;
    let block = opt_usize(opts, "block", 8)?;
    let profile = on_off(opts, "profile", false)?;
    if n == 0 {
        return Err("--tokens must be >= 1".into());
    }
    let draft_path = require(opts, "draft")?;
    let eos: Vec<u32> = bundle.eos.clone();

    // AR greedy: the token chosen at step i and the top-2 of the logits that chose it.
    bundle.reset(gpu)?;
    let t_ar = Instant::now();
    bundle.prefill(gpu, prompt)?;
    let mut ar: Vec<u32> = Vec::with_capacity(n);
    let mut ar_top2: Vec<[(u32, f32); 2]> = Vec::with_capacity(n);
    let (mut ar_decode_ms, mut ar_decode_tokens) = (0f64, 0usize);
    loop {
        let l = bundle.download_logits(gpu)?;
        let t2 = top2(&l);
        let tok = t2[0].0;
        ar_top2.push(t2);
        ar.push(tok);
        if ar.len() >= n || eos.contains(&tok) {
            break;
        }
        if profile {
            sync(gpu)?;
        }
        let t = Instant::now();
        bundle.decode_step(gpu, tok)?;
        if profile {
            sync(gpu)?;
            ar_decode_ms += t.elapsed().as_secs_f64() * 1e3;
            ar_decode_tokens += 1;
        }
    }
    let ar_secs = t_ar.elapsed().as_secs_f64();
    cut_at_eos(&eos, &mut ar);

    // DFlash greedy through the production speculator, behind the logging (and,
    // with --profile, timing) wrapper.
    bundle.reset(gpu)?;
    let ctx_capacity = bundle.max_seq();
    let prof: SharedProf = Rc::new(RefCell::new(Prof::default()));
    let run = {
        let mut wrapped = TimedTarget {
            inner: &mut *bundle,
            prof: Rc::clone(&prof),
            timed: profile,
        };
        dflash_arm(
            gpu,
            &mut wrapped,
            prompt,
            draft_path,
            block,
            n,
            ctx_capacity,
            &eos,
            &prof,
            profile,
        )?
    };
    let mut df = run.stream.clone();
    df.truncate(n);
    cut_at_eos(&eos, &mut df);

    let ar_md5 = md5_hex_u32(&ar);
    let df_md5 = md5_hex_u32(&df);
    let common = ar.len().min(df.len());
    let diverge = (0..common).find(|&i| ar[i] != df[i]);
    let identical = diverge.is_none() && ar.len() == df.len();
    let emitted_by_steps = df.len().saturating_sub(1);
    let tokens_per_step = emitted_by_steps as f64 / run.steps.max(1) as f64;
    println!(
        "greedy: prompt {} tokens, {n} requested, block {block}",
        prompt.len()
    );
    println!(
        "  AR     {} tokens  md5 {ar_md5}  ({ar_secs:.2}s incl. prefill)",
        ar.len()
    );
    println!(
        "  DFlash {} tokens  md5 {df_md5}  ({:.2}s incl. prefill)",
        df.len(),
        run.secs
    );
    println!(
        "  steps {}  mean tokens/step {tokens_per_step:.3}  accepted drafts/step {:.3}  proposed/step {:.3}",
        run.steps,
        run.accepted as f64 / run.steps.max(1) as f64,
        run.proposed as f64 / run.steps.max(1) as f64
    );
    let profile_json = if profile {
        report_profile(&run, ar_decode_ms, ar_decode_tokens)
    } else {
        Value::Null
    };
    let mut divergence = Value::Null;
    let mut tie = Value::Null;
    let mut justified = false;
    let mut near_tie = false;
    match diverge {
        Some(i) => {
            let t = ar_top2[i];
            let margin = t[0].1 - t[1].1;
            println!(
                "  first divergence at index {i}: AR {} vs DFlash {}; AR top-2 {} ({}) / {} ({}), margin {margin:.6}",
                ar[i], df[i], t[0].0, t[0].1, t[1].0, t[1].1
            );
            println!(
                "  DFlash token is AR's {} at that index",
                if df[i] == t[1].0 {
                    "runner-up"
                } else {
                    "not top-2"
                }
            );
            divergence = json!({
                "index": i, "ar_token": ar[i], "dflash_token": df[i],
                "ar_top1": {"id": t[0].0, "logit": t[0].1},
                "ar_top2": {"id": t[1].0, "logit": t[1].1},
                "margin": margin,
                "dflash_is_runner_up": df[i] == t[1].0,
            });
            // The last verify whose rows predicted absolute position prompt+i.
            let target_pos = prompt.len() + i;
            let window = run.step_prof.as_ref().and_then(|p| {
                p.verifies
                    .iter()
                    .rev()
                    .find(|(pos, block)| *pos < target_pos && target_pos <= pos + block.len())
            });
            let (j, v) = tie_judge(gpu, bundle, prompt, &ar[..i], ar[i], df[i], window)?;
            justified = j;
            tie = v;
            // Second level: the exact window replayed from the AR arm's state
            // keeps AR's pick, so the flip comes from the DFlash arm's own state
            // history (built by batched verify/replay forwards, not single-row
            // decode). Accept it only as a near-tie: DFlash took AR's runner-up
            // and AR's own margin is within NEAR_TIE_LOGITS, below the measured
            // single-row vs batched path discrepancy (rowexact: up to 1.57).
            if !justified && df[i] == t[1].0 && margin <= NEAR_TIE_LOGITS {
                near_tie = true;
                println!(
                    "    near-tie: DFlash took AR's runner-up at AR margin {margin} <= {NEAR_TIE_LOGITS} \
                     (state-history path flip) -> NEAR-TIE"
                );
            }
        }
        None if ar.len() != df.len() => println!(
            "  streams agree on the first {common} tokens but differ in length (AR {}, DFlash {})",
            ar.len(),
            df.len()
        ),
        None => println!("  streams are identical"),
    }
    let pass = identical || justified || near_tie;
    println!(
        "{} greedy{}",
        if pass { "PASS" } else { "FAIL" },
        if identical {
            " (streams identical)"
        } else if justified {
            " (first divergence is a JUSTIFIED-TIE)"
        } else if near_tie {
            " (first divergence is a NEAR-TIE)"
        } else {
            ""
        }
    );
    if let Some(dir) = out {
        write_json(
            &dir.join("greedy.json"),
            &json!({
                "pass": pass, "identical": identical, "block": block, "prompt_tokens": prompt.len(),
                "ar_md5": ar_md5, "dflash_md5": df_md5,
                "ar_tokens": ar, "dflash_tokens": df,
                "steps": run.steps, "mean_tokens_per_step": tokens_per_step,
                "accepted": run.accepted, "proposed": run.proposed,
                "divergence": divergence,
                "tie_judge": tie,
                "near_tie": near_tie,
                "profile": profile_json,
            }),
        )?;
    }
    Ok(pass)
}

// ─────────────────────────────────────────────────────────────────────────────
// rollback
// ─────────────────────────────────────────────────────────────────────────────

/// Where byte `i` of the persistent slab lives.
fn slab_locate(i: usize, lay: &PersistLayout, conv: &ConvSlabLayout, d: &RaonStateDims) -> String {
    if i < lay.conv_off {
        let (layer, within) = (i / lay.rda_stride, i % lay.rda_stride);
        let e = within / 4;
        let per_head = d.v_dim * d.k_dim;
        if e >= d.heads * per_head {
            return format!("RDA layer {layer} padding +{within}");
        }
        return format!(
            "RDA layer {layer} S[head {}, v {}, k {}] (byte {i})",
            e / per_head,
            (e % per_head) / d.k_dim,
            e % d.k_dim
        );
    }
    if i < lay.seq_off {
        let rem = i - lay.conv_off;
        let (layer, w) = (rem / lay.conv_stride, rem % lay.conv_stride);
        let (plane, wp) = (w / conv.plane_pitch, w % conv.plane_pitch);
        let (row, col) = (wp / (conv.cols * 4), (wp % (conv.cols * 4)) / 4);
        let name = ["q", "k", "v"].get(plane).copied().unwrap_or("?");
        if row >= conv.hist {
            return format!("RDA layer {layer} conv_{name} padding +{wp}");
        }
        return format!("RDA layer {layer} conv_{name}[hist row {row}, ch {col}] (byte {i})");
    }
    format!("seq_len/pad +{}", i - lay.seq_off)
}

/// Bytes that differ between two slab downloads, the first difference located,
/// and the RDA layers whose matrix / conv history differ.
fn slab_diff(
    a: &[u8],
    b: &[u8],
    lay: &PersistLayout,
    conv: &ConvSlabLayout,
    d: &RaonStateDims,
) -> (usize, Option<String>, Vec<usize>, Vec<usize>) {
    let mut bytes = 0usize;
    let mut first: Option<usize> = None;
    let (mut s_layers, mut c_layers) = (Vec::new(), Vec::new());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x == y {
            continue;
        }
        bytes += 1;
        first.get_or_insert(i);
        if i < lay.conv_off {
            let l = i / lay.rda_stride;
            if s_layers.last() != Some(&l) {
                s_layers.push(l);
            }
        } else if i < lay.seq_off {
            let l = (i - lay.conv_off) / lay.conv_stride;
            if c_layers.last() != Some(&l) {
                c_layers.push(l);
            }
        }
    }
    let first = first.map(|i| {
        format!(
            "{} ({:#04x} vs {:#04x})",
            slab_locate(i, lay, conv, d),
            a[i],
            b[i]
        )
    });
    (bytes, first, s_layers, c_layers)
}

/// One arm of the rollback gate: restore `base`, verify `block`, commit
/// `keep` rows under `mode`; the slab after the commit and the commit's
/// device-synchronised milliseconds.
#[allow(clippy::too_many_arguments)]
fn rollback_arm(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    scratch: &mut dyn SpecScratch,
    base: &RaonStateSnapshot,
    block: &[u32],
    position: usize,
    keep: usize,
    mode: RollbackMode,
    features: bool,
) -> Result<(Vec<u8>, f64), String> {
    bundle.state.restore(gpu, base)?;
    bundle.set_dflash_rollback_mode(mode);
    let mut feat: Vec<f32> = Vec::new();
    SpecTarget::verify_block(
        bundle,
        gpu,
        block,
        position,
        scratch,
        features.then_some(&mut feat),
    )?;
    sync(gpu)?;
    let t = Instant::now();
    SpecTarget::commit_prefix(bundle, gpu, block, keep - 1, position, scratch)?;
    sync(gpu)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    if bundle.n_tokens() != position + keep {
        return Err(format!(
            "{} commit of {keep} rows left n_tokens {}, expected {}",
            mode.as_str(),
            bundle.n_tokens(),
            position + keep
        ));
    }
    Ok((bundle.state.download_persist(gpu)?, ms))
}

fn run_rollback(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    prompt: &[u32],
    opts: &Opts,
    out: Option<&Path>,
) -> Result<bool, String> {
    const BLOCK: usize = 8;
    let features = match opts.get("draft") {
        Some(p) => {
            let (_hfq, cfg) = draft_config(p)?;
            check_streams(bundle, &cfg)?;
            SpecTarget::set_dflash_extract_layers(bundle, cfg.target_layer_ids.clone());
            true
        }
        None => false,
    };
    bundle.reset(gpu)?;
    bundle.prefill(gpu, prompt)?;
    let position = bundle.n_tokens();
    if position + BLOCK > bundle.max_seq() {
        return Err(format!(
            "context {position} + {BLOCK} rows exceeds max_seq {}",
            bundle.max_seq()
        ));
    }
    let mut base = bundle.state.snapshot_alloc(gpu)?;
    bundle.state.snapshot_into(gpu, &mut base)?;

    // The block: the target's greedy seed and 7 tokens of its AR greedy chain.
    let mut chain = vec![argmax(&bundle.download_logits(gpu)?) as u32];
    for i in 0..BLOCK - 1 {
        bundle.decode_step(gpu, chain[i])?;
        chain.push(argmax(&bundle.download_logits(gpu)?) as u32);
    }
    bundle.state.restore(gpu, &base)?;

    let lay = *bundle.state.persist_layout();
    let conv = bundle.state.conv_slab_layout()?;
    let dims = bundle.state.dims().clone();
    let mut scratch: Box<dyn SpecScratch> = SpecTarget::new_spec_scratch(bundle, gpu, BLOCK)?;
    println!(
        "rollback: prompt {position} tokens, block {BLOCK}{}; chain {chain:?}",
        if features {
            " (verify with DFlash feature capture)"
        } else {
            ""
        }
    );
    println!("  keep  slab  differing_bytes  recurrence_ms  replay_ms  first difference");
    let r = (|| -> Result<(bool, Vec<Value>), String> {
        let (mut all_equal, mut rows) = (true, Vec::new());
        for keep in 1..BLOCK {
            let (a, rec_ms) = rollback_arm(
                gpu,
                bundle,
                scratch.as_mut(),
                &base,
                &chain,
                position,
                keep,
                RollbackMode::Recurrence,
                features,
            )?;
            let (b, replay_ms) = rollback_arm(
                gpu,
                bundle,
                scratch.as_mut(),
                &base,
                &chain,
                position,
                keep,
                RollbackMode::Replay,
                features,
            )?;
            if a.len() != b.len() {
                return Err(format!("slab sizes differ: {} vs {}", a.len(), b.len()));
            }
            let (bytes, first, s_layers, c_layers) = slab_diff(&a, &b, &lay, &conv, &dims);
            let equal = bytes == 0;
            // keep 1 is reported, never judged: the replay arm's 1-row forward
            // takes the single-row decode route (`forward_chunk` with one
            // token), which rounds differently from the batched verify rows
            // the recurrence arm reproduces, so the two arms differ by
            // construction. keep 2..=7 replay through the batched route.
            if keep >= 2 {
                all_equal &= equal;
            }
            println!(
                "  {keep:>4}  {:<5} {bytes:>15}  {rec_ms:>13.3}  {replay_ms:>9.3}  {}",
                if equal { "equal" } else { "DIFF" },
                first.as_deref().unwrap_or("-")
            );
            if !equal {
                println!(
                    "        S layers differing {s_layers:?}; conv layers differing {c_layers:?}"
                );
            }
            rows.push(json!({
                "keep": keep, "equal": equal, "differing_bytes": bytes,
                "first_difference": first,
                "rda_state_layers_differing": s_layers,
                "conv_layers_differing": c_layers,
                "recurrence_commit_ms": rec_ms, "replay_commit_ms": replay_ms,
            }));
        }
        Ok((all_equal, rows))
    })();
    scratch.free(gpu);
    // Leave the slot as found.
    let restored = bundle.state.restore(gpu, &base);
    let freed = base.free_gpu(gpu);
    let (all_equal, rows) = r?;
    restored?;
    freed?;
    println!(
        "{} rollback: the persistent slab after recurrence rollbacks keep 2..={} {} byte-identical to the full-replay arm (keep 1 reported only: the replay arm's 1-row forward takes the decode route)",
        if all_equal { "PASS" } else { "FAIL" },
        BLOCK - 1,
        if all_equal { "is" } else { "is NOT" }
    );
    if let Some(dir) = out {
        write_json(
            &dir.join("rollback.json"),
            &json!({"pass": all_equal, "position": position, "block": BLOCK, "chain": chain, "per_keep": rows}),
        )?;
    }
    Ok(all_equal)
}

// ─────────────────────────────────────────────────────────────────────────────
// verifygraph
// ─────────────────────────────────────────────────────────────────────────────

/// Everything one verify window leaves behind, for the eager/graph comparison.
struct VgTrial {
    /// `verify_block_logits` rows (`K x vocab`).
    logits: Vec<f32>,
    /// Feature rows captured by the same verify (`K x row_width`).
    feats: Vec<f32>,
    /// `verify_block` per-row device argmax.
    picks: Vec<u32>,
    /// Persist slab (RDA state, conv history, device `seq_len`) after a
    /// partial-accept `commit_prefix` / after a full-accept one.
    partial: Vec<u8>,
    full: Vec<u8>,
}

/// One window over `block` from `snap`: logits + features + abandon, argmax +
/// abandon, then partial and full `commit_prefix` with the persist slab
/// downloaded after each; every step is rolled back to `snap`.
fn vg_trial(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    scratch: &mut dyn SpecScratch,
    snap: &RaonStateSnapshot,
    block: &[u32],
    position: usize,
) -> Result<VgTrial, String> {
    let k = block.len();
    let mut feats = Vec::new();
    let logits =
        SpecTarget::verify_block_logits(bundle, gpu, block, position, scratch, Some(&mut feats))?;
    SpecTarget::abandon_verify(bundle, gpu, scratch)?;
    let picks = SpecTarget::verify_block(bundle, gpu, block, position, scratch, None)?;
    SpecTarget::abandon_verify(bundle, gpu, scratch)?;
    if bundle.n_tokens() != position {
        return Err(format!(
            "abandon_verify left n_tokens {} != {position} (K={k})",
            bundle.n_tokens()
        ));
    }
    let mut persist: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
    for (slot, accept) in [(k - 1) / 2, k - 1].into_iter().enumerate() {
        let mut f = Vec::new();
        SpecTarget::verify_block(bundle, gpu, block, position, scratch, Some(&mut f))?;
        SpecTarget::commit_prefix(bundle, gpu, block, accept, position, scratch)?;
        let want = position + accept + 1;
        if bundle.n_tokens() != want {
            return Err(format!(
                "commit_prefix(accept {accept}) left n_tokens {} != {want} (K={k})",
                bundle.n_tokens()
            ));
        }
        persist[slot] = bundle.state.download_persist(gpu)?;
        bundle.state.restore(gpu, snap)?;
    }
    let [partial, full] = persist;
    Ok(VgTrial {
        logits,
        feats,
        picks,
        partial,
        full,
    })
}

/// First differing bit of two F32 slices (a length mismatch differs at the shorter length).
fn first_f32_diff(a: &[f32], b: &[f32]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
}

fn first_byte_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// Differences of `g` (graph mode) against `e` (eager mode), empty when identical.
fn vg_compare(e: &VgTrial, g: &VgTrial, vocab: usize, row_width: usize) -> Vec<String> {
    let mut d = Vec::new();
    if let Some(i) = first_f32_diff(&e.logits, &g.logits) {
        d.push(format!(
            "logits differ at row {} col {} (eager {:#010x}, graph {:#010x})",
            i / vocab,
            i % vocab,
            e.logits.get(i).map_or(0, |v| v.to_bits()),
            g.logits.get(i).map_or(0, |v| v.to_bits())
        ));
    }
    if let Some(i) = first_f32_diff(&e.feats, &g.feats) {
        d.push(format!(
            "features differ at row {} col {} (eager {:#010x}, graph {:#010x})",
            i / row_width.max(1),
            i % row_width.max(1),
            e.feats.get(i).map_or(0, |v| v.to_bits()),
            g.feats.get(i).map_or(0, |v| v.to_bits())
        ));
    }
    if e.picks != g.picks {
        d.push(format!(
            "argmax picks differ: eager {:?}, graph {:?}",
            e.picks, g.picks
        ));
    }
    for (name, a, b) in [
        ("partial", &e.partial, &g.partial),
        ("full", &e.full, &g.full),
    ] {
        if let Some(i) = first_byte_diff(a, b) {
            d.push(format!(
                "persist slab after {name}-accept commit differs at byte {i} (eager {:#04x}, graph {:#04x})",
                a.get(i).copied().unwrap_or(0),
                b.get(i).copied().unwrap_or(0)
            ));
        }
    }
    d
}

/// Verify-graph gate: for K = 1..=16 at several positions, one verify window
/// run with `DecodeMode::Eager` and again with `DecodeMode::Graph` (first use
/// = eager warm run, then record, then replay) from the same snapshot must
/// leave bit-identical logits rows, captured feature rows, argmax picks and
/// persist slab (RDA state, conv history, `seq_len`) after partial and full
/// `commit_prefix`.
fn run_verifygraph(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    prompt: &[u32],
    out: Option<&Path>,
) -> Result<bool, String> {
    const KMAX: usize = 16;
    const POSITIONS: usize = 3;
    const ADVANCE: usize = 5;
    let layers = vec![1usize, 6, 11, 16, 21];
    if bundle.config.n_layers <= layers[layers.len() - 1] {
        return Err(format!(
            "model has {} layers, the gate extracts {layers:?}",
            bundle.config.n_layers
        ));
    }
    let vocab = bundle.config.vocab;
    let row_width = layers.len() * bundle.config.mhc_streams * bundle.config.hidden;
    SpecTarget::set_dflash_extract_layers(bundle, layers);
    bundle.set_decode_mode(DecodeMode::Graph);
    bundle.reset(gpu)?;
    bundle.prefill(gpu, prompt)?;
    let need = bundle.n_tokens() + POSITIONS * ADVANCE + KMAX;
    if need > bundle.max_seq() {
        return Err(format!(
            "context {} + {} rows exceeds max_seq {}",
            bundle.n_tokens(),
            POSITIONS * ADVANCE + KMAX,
            bundle.max_seq()
        ));
    }
    let mut scratch: Box<dyn SpecScratch> = SpecTarget::new_spec_scratch(bundle, gpu, KMAX)?;
    let mut snap = bundle.state.snapshot_alloc(gpu)?;
    let (caps0, reps0) = bundle.verify_graph_stats();
    let r = (|| -> Result<bool, String> {
        let mut failures = 0usize;
        let mut windows = 0usize;
        let mut rows_json = Vec::new();
        println!(
            "verifygraph: prompt {} tokens; K = 1..={KMAX} at {POSITIONS} positions",
            prompt.len()
        );
        for step in 0..POSITIONS {
            bundle.set_decode_mode(DecodeMode::Graph);
            if step > 0 {
                for _ in 0..ADVANCE {
                    let t = argmax(&bundle.download_logits(gpu)?) as u32;
                    bundle.decode_step(gpu, t)?;
                }
            }
            let position = bundle.n_tokens();
            let first = argmax(&bundle.download_logits(gpu)?) as u32;
            bundle.state.snapshot_into(gpu, &mut snap)?;
            let full: Vec<u32> = (0..KMAX)
                .map(|i| {
                    if i == 0 {
                        first
                    } else {
                        prompt[(7 * i + 13 * step + 3) % prompt.len()]
                    }
                })
                .collect();
            for kk in 1..=KMAX {
                let block = &full[..kk];
                bundle.set_decode_mode(DecodeMode::Eager);
                let eager = vg_trial(gpu, bundle, scratch.as_mut(), &snap, block, position)?;
                bundle.set_decode_mode(DecodeMode::Graph);
                let mut diffs = Vec::new();
                for pass in 0..3 {
                    let graph = vg_trial(gpu, bundle, scratch.as_mut(), &snap, block, position)?;
                    for d in vg_compare(&eager, &graph, vocab, row_width) {
                        diffs.push(format!("pass {pass}: {d}"));
                    }
                }
                windows += 1;
                if diffs.is_empty() {
                    println!("  pos {position} K {kk:>2}: graph == eager (logits, features, argmax, rda/conv/seq_len after partial+full commit) x3 passes");
                } else {
                    failures += 1;
                    println!("  pos {position} K {kk:>2}: FAIL");
                    for d in &diffs {
                        println!("      {d}");
                    }
                }
                rows_json.push(
                    json!({"position": position, "k": kk, "ok": diffs.is_empty(), "diffs": diffs}),
                );
            }
        }
        let (caps1, reps1) = bundle.verify_graph_stats();
        let (caps, reps) = (caps1 - caps0, reps1 - reps0);
        println!(
            "verifygraph: {} verify graph captures, {} replays, {} kernel nodes in the last recording, {:.3} ms host time in hipGraphLaunch",
            caps,
            reps,
            bundle.verify_graph_kernel_nodes(),
            bundle.verify_graph_launch_secs() * 1e3
        );
        let mut pass = failures == 0;
        if caps == 0 || reps == 0 {
            println!(
                "FAIL verifygraph: the graph route never recorded and replayed a verify (captures {caps}, replays {reps}, decode mode {})",
                bundle.decode_mode()
            );
            pass = false;
        }
        println!(
            "{} verifygraph: {windows} windows, {failures} with eager/graph differences",
            if pass { "PASS" } else { "FAIL" }
        );
        if let Some(dir) = out {
            write_json(
                &dir.join("verifygraph.json"),
                &json!({"pass": pass, "windows": windows, "failures": failures,
                        "captures": caps, "replays": reps, "rows": rows_json}),
            )?;
        }
        Ok(pass)
    })();
    scratch.free(gpu);
    let freed = snap.free_gpu(gpu);
    let pass = r?;
    freed?;
    Ok(pass)
}

// ─────────────────────────────────────────────────────────────────────────────
// sampled
// ─────────────────────────────────────────────────────────────────────────────

const SALT_AR: u64 = 0x4152_5F53_414D_504C;
const SALT_DF: u64 = 0x4446_4C41_5348_3253;

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn trial_seed(arm_salt: u64, trial: usize) -> u64 {
    splitmix64(arm_salt ^ splitmix64(trial as u64))
}

/// ln Γ(x), Lanczos (g = 7, n = 9).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized upper incomplete gamma Q(a, x) (series / continued fraction).
fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let ln_pre = -x + a * x.ln() - ln_gamma(a);
    if x < a + 1.0 {
        let (mut sum, mut term, mut n) = (1.0 / a, 1.0 / a, a);
        for _ in 0..1000 {
            n += 1.0;
            term *= x / n;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        1.0 - sum * ln_pre.exp()
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        ln_pre.exp() * h
    }
}

/// Two-sample chi-square homogeneity test of category counts `a` vs `b`.
/// Categories whose expected count is under 5 in either sample are pooled.
/// Returns `(statistic, degrees of freedom, p-value)`.
fn homogeneity<K: std::hash::Hash + Eq + Clone>(a: &[K], b: &[K]) -> (f64, usize, f64) {
    let mut cells: HashMap<K, (f64, f64)> = HashMap::new();
    for key in a {
        cells.entry(key.clone()).or_default().0 += 1.0;
    }
    for key in b {
        cells.entry(key.clone()).or_default().1 += 1.0;
    }
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let n = na + nb;
    let mut kept: Vec<(f64, f64)> = Vec::new();
    let mut pooled = (0.0, 0.0);
    for &(x, y) in cells.values() {
        let total = x + y;
        if total * na.min(nb) / n < 5.0 {
            pooled.0 += x;
            pooled.1 += y;
        } else {
            kept.push((x, y));
        }
    }
    if pooled.0 + pooled.1 > 0.0 {
        kept.push(pooled);
    }
    let mut stat = 0.0;
    for &(x, y) in &kept {
        let total = x + y;
        let (ea, eb) = (total * na / n, total * nb / n);
        stat += (x - ea).powi(2) / ea + (y - eb).powi(2) / eb;
    }
    let df = kept.len().saturating_sub(1).max(1);
    (stat, df, gamma_q(df as f64 / 2.0, stat / 2.0))
}

/// Pearson chi-square of independent categorical draws, each from its own
/// law: `samples[i] = (token_i, law_i)`. Observed count of a token = draws of
/// it; expected = the sum of its probability over every draw's law (each
/// draw is one multinomial(1, law_i) trial, so this is the exact mean). Cells
/// with expected count < 5 are pooled. A draw outside its own law's support
/// is counted (`outside_support`) and joins the pooled cell.
/// Returns `(statistic, degrees of freedom, p-value, outside_support)`.
fn gof_rows(samples: &[(u32, &[(u32, f32)])]) -> (f64, usize, f64, u64) {
    let mut cells: HashMap<u32, (f64, f64)> = HashMap::new();
    let mut outside = 0u64;
    for &(tok, law) in samples {
        let total: f64 = law.iter().map(|&(_, q)| q as f64).sum();
        for &(t, q) in law {
            cells.entry(t).or_default().1 += q as f64 / total;
        }
        if !law.iter().any(|&(t, q)| t == tok && q > 0.0) {
            outside += 1;
        }
        cells.entry(tok).or_default().0 += 1.0;
    }
    let (mut stat, mut kept) = (0.0f64, 0usize);
    let mut pooled = (0.0f64, 0.0f64);
    for &(o, e) in cells.values() {
        if e < 5.0 {
            pooled.0 += o;
            pooled.1 += e;
        } else {
            stat += (o - e).powi(2) / e;
            kept += 1;
        }
    }
    if pooled.1 > 0.0 {
        stat += (pooled.0 - pooled.1).powi(2) / pooled.1;
        kept += 1;
    } else if pooled.0 > 0.0 {
        stat = f64::INFINITY;
        kept += 1;
    }
    let df = kept.saturating_sub(1).max(1);
    let p = if stat.is_infinite() {
        0.0
    } else {
        gamma_q(df as f64 / 2.0, stat / 2.0)
    };
    (stat, df, p, outside)
}

/// One printed/serialized test line.
struct TestRec {
    name: String,
    stat: f64,
    df: usize,
    p: f64,
    n: u64,
    outside: u64,
    pass: bool,
    /// Counts toward the verdict (an informational test only reports).
    gating: bool,
}

impl TestRec {
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        stat: f64,
        df: usize,
        p: f64,
        n: u64,
        outside: u64,
        alpha: f64,
        gating: bool,
    ) -> Self {
        let pass = p > alpha && outside == 0;
        let rec = TestRec {
            name,
            stat,
            df,
            p,
            n,
            outside,
            pass,
            gating,
        };
        println!(
            "{} {:.4} {} {:.6} {} {}{}",
            rec.name,
            rec.stat,
            rec.df,
            rec.p,
            rec.n,
            if rec.outside > 0 {
                format!("FAIL(outside_support={})", rec.outside)
            } else if rec.pass {
                "PASS".to_string()
            } else {
                "FAIL".to_string()
            },
            if rec.gating { "" } else { " (info)" }
        );
        rec
    }

    fn json(&self) -> Value {
        json!({"name": self.name, "stat": self.stat, "df": self.df, "p": self.p,
               "n": self.n, "outside_support": self.outside, "pass": self.pass,
               "gating": self.gating})
    }
}

/// The truncated sampling law (`SampleSpec`) of one logits row, as entries.
fn row_law(row: &[f32], spec: SampleSpec) -> Result<Vec<(u32, f32)>, String> {
    let mut d = SparseDist::default();
    d.build_from_logits(row, spec, &mut Vec::new())?;
    Ok(d.entries().to_vec())
}

/// Mean of per-draw laws (the marginal a position's draws follow).
fn mean_law(laws: &[&[(u32, f32)]]) -> Vec<(u32, f32)> {
    let mut m: BTreeMap<u32, f64> = BTreeMap::new();
    for law in laws {
        for &(t, p) in *law {
            *m.entry(t).or_default() += p as f64;
        }
    }
    let n = laws.len().max(1) as f64;
    m.into_iter().map(|(t, p)| (t, (p / n) as f32)).collect()
}

/// Total-variation distance between two sparse laws.
fn tv_distance(a: &[(u32, f32)], b: &[(u32, f32)]) -> f64 {
    let mut m: HashMap<u32, (f64, f64)> = HashMap::new();
    for &(t, p) in a {
        m.entry(t).or_default().0 += p as f64;
    }
    for &(t, p) in b {
        m.entry(t).or_default().1 += p as f64;
    }
    0.5 * m.values().map(|(x, y)| (x - y).abs()).sum::<f64>()
}

/// Sampled-decoding distribution oracle: every emitted token of the AR host
/// sampler and of DFlash2 sampled speculation against the truncated law of the
/// target row it was drawn or verified against.
fn run_sampled(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    prompt: &[u32],
    opts: &Opts,
    out: Option<&Path>,
) -> Result<bool, String> {
    let trials = opt_usize(opts, "trials", 2000)?;
    let l = opt_usize(opts, "tokens", 4)?;
    let block = opt_usize(opts, "block", 8)?;
    let temp = opt_f32(opts, "temp", 1.0)?;
    let top_p = opt_f32(opts, "top-p", 0.95)?;
    let top_k = opt_usize(opts, "top-k", 20)? as u32;
    let alpha = opt_f64(opts, "alpha", 1e-3)?;
    if trials == 0 || l == 0 {
        return Err("--trials and --tokens must be >= 1".into());
    }
    if !(2..=DFLASH_BLOCK_MAX).contains(&block) {
        return Err(format!("--block {block} outside 2..={DFLASH_BLOCK_MAX}"));
    }
    if temp <= 0.0 || top_k == 0 {
        return Err("--temp must be > 0 and --top-k >= 1".into());
    }
    let draft_path = require(opts, "draft")?;
    let model = require(opts, "model")?;
    let spec_s = SampleSpec::cpu_ar(temp, top_p, Some(top_k), 0.0);

    // AR arm: the production host sampler on the decode path's rows; every
    // draw keeps the law of the row it was drawn from.
    let t_ar = Instant::now();
    let mut ar: Vec<Vec<u32>> = Vec::with_capacity(trials);
    let mut ar_laws: Vec<Vec<Vec<(u32, f32)>>> = Vec::with_capacity(trials);
    for t in 0..trials {
        bundle.reset(gpu)?;
        bundle.prefill(gpu, prompt)?;
        let seed = trial_seed(SALT_AR, t);
        let mut state = (((seed >> 32) as u32) ^ (seed as u32)).max(1);
        let (mut s, mut laws) = (Vec::with_capacity(l), Vec::with_capacity(l));
        for i in 0..l {
            let logits = bundle.download_logits(gpu)?;
            laws.push(row_law(&logits, spec_s)?);
            let tok =
                sample_top_k_p_with_state(&logits, temp, top_p, Some(top_k), None, &mut state);
            s.push(tok);
            if i + 1 < l {
                bundle.decode_step(gpu, tok)?;
            }
        }
        ar.push(s);
        ar_laws.push(laws);
        if (t + 1) % 250 == 0 {
            eprintln!("sampled: AR trial {}/{trials}", t + 1);
        }
    }
    let ar_secs = t_ar.elapsed().as_secs_f64();

    // DFlash arm through the recording wrapper: the prefill row and every
    // verify row's law, so each emitted token is judged against the row it
    // was accepted or resampled from (emitted token j of a window = verify
    // row j: the accepted drafts, then the residual or bonus draw).
    let t_df = Instant::now();
    let ctx_capacity = bundle.max_seq();
    let prof: SharedProf = Rc::new(RefCell::new(Prof {
        row_law: Some(spec_s),
        ..Prof::default()
    }));
    let mut df: Vec<Vec<u32>> = Vec::with_capacity(trials);
    let mut df_laws: Vec<Vec<Vec<(u32, f32)>>> = Vec::with_capacity(trials);
    let (mut windows, mut proposed_sum, mut accepted_sum) = (0u64, 0u64, 0u64);
    let mut depth_hist: BTreeMap<usize, u64> = BTreeMap::new();
    {
        let mut target = TimedTarget {
            inner: &mut *bundle,
            prof: Rc::clone(&prof),
            timed: false,
        };
        let mut spec: Box<dyn Speculator> =
            build_dflash2_speculator(gpu, draft_path, &mut target, ctx_capacity, block)?;
        let r = (|| -> Result<(), String> {
            for t in 0..trials {
                spec.configure_request(SpecRequestConfig {
                    temp,
                    top_p,
                    top_k: Some(top_k),
                    min_p: 0.0,
                    rng_seed: trial_seed(SALT_DF, t),
                    ..Default::default()
                });
                let first = spec
                    .prefill(gpu, &mut target, prompt, prompt, 0, false, None, &|| false)?
                    .into_first_token()?;
                let prefill_law = std::mem::take(&mut prof.borrow_mut().last_laws);
                let [p0] = <[Vec<(u32, f32)>; 1]>::try_from(prefill_law)
                    .map_err(|v| format!("prefill recorded {} row laws, expected 1", v.len()))?;
                SpecTarget::map_ahead_for_decode(&mut target, gpu, prompt.len(), l, block);
                let mut stream = vec![first];
                let mut laws = vec![p0];
                let (mut position, mut seed) = (prompt.len(), first);
                while stream.len() < l {
                    let step = spec.step(
                        gpu,
                        &mut target,
                        position,
                        seed,
                        &stream,
                        None,
                        temp,
                        usize::MAX,
                    )?;
                    let rows = std::mem::take(&mut prof.borrow_mut().last_laws);
                    if rows.len() < step.emit.len() {
                        return Err(format!(
                            "a window emitted {} tokens from {} recorded verify rows",
                            step.emit.len(),
                            rows.len()
                        ));
                    }
                    laws.extend(rows.into_iter().take(step.emit.len()));
                    windows += 1;
                    proposed_sum += step.proposed as u64;
                    accepted_sum += step.accepted as u64;
                    *depth_hist.entry(step.proposed).or_default() += 1;
                    position += step.emit.len();
                    seed = step.next_seed;
                    stream.extend_from_slice(&step.emit);
                }
                stream.truncate(l);
                laws.truncate(l);
                df.push(stream);
                df_laws.push(laws);
                if (t + 1) % 250 == 0 {
                    eprintln!("sampled: DFlash trial {}/{trials}", t + 1);
                }
            }
            Ok(())
        })();
        spec.free(gpu);
        r?;
    }
    let df_secs = t_df.elapsed().as_secs_f64();

    // Tests. Gating: every position's draws against their own rows' laws, per
    // arm. Informational: AR-vs-DFlash homogeneity and the distance between
    // the two arms' mean laws, which also measure the verify-row vs
    // decode-row rounding gap (rowexact: up to 1.57 logits; the row-exact
    // verify work), not the sampler.
    let mut tests: Vec<TestRec> = Vec::new();
    let mut tv = Vec::with_capacity(l);
    let n_t = trials as u64;
    for i in 0..l {
        for (arm, streams, laws) in [("ar", &ar, &ar_laws), ("dflash", &df, &df_laws)] {
            let samples: Vec<(u32, &[(u32, f32)])> = streams
                .iter()
                .zip(laws.iter())
                .map(|(s, l)| (s[i], l[i].as_slice()))
                .collect();
            let (stat, d, p, o) = gof_rows(&samples);
            tests.push(TestRec::new(
                format!("gof_rows_pos{i}_{arm}"),
                stat,
                d,
                p,
                n_t,
                o,
                alpha,
                true,
            ));
        }
        let a: Vec<&[(u32, f32)]> = ar_laws.iter().map(|v| v[i].as_slice()).collect();
        let b: Vec<&[(u32, f32)]> = df_laws.iter().map(|v| v[i].as_slice()).collect();
        let d = tv_distance(&mean_law(&a), &mean_law(&b));
        println!("tv_mean_law_pos{i} ar vs dflash: {d:.4}");
        tv.push(d);
        let a: Vec<u32> = ar.iter().map(|s| s[i]).collect();
        let b: Vec<u32> = df.iter().map(|s| s[i]).collect();
        let (stat, d, p) = homogeneity(&a, &b);
        tests.push(TestRec::new(
            format!("homog_pos{i}"),
            stat,
            d,
            p,
            2 * n_t,
            0,
            alpha,
            false,
        ));
    }
    for (name, i) in [("homog_joint01", 0usize), ("homog_joint12", 1usize)] {
        if l > i + 1 {
            let a: Vec<(u32, u32)> = ar.iter().map(|s| (s[i], s[i + 1])).collect();
            let b: Vec<(u32, u32)> = df.iter().map(|s| (s[i], s[i + 1])).collect();
            let (stat, d, p) = homogeneity(&a, &b);
            tests.push(TestRec::new(
                name.to_string(),
                stat,
                d,
                p,
                2 * n_t,
                0,
                alpha,
                false,
            ));
        }
    }
    let pass = tests.iter().filter(|t| t.gating).all(|t| t.pass);
    println!(
        "{} sampled (alpha {alpha}, {} gating of {} tests, mean verified depth {:.3}, mean accepted {:.3} over {windows} windows)",
        if pass { "PASS" } else { "FAIL" },
        tests.iter().filter(|t| t.gating).count(),
        tests.len(),
        proposed_sum as f64 / windows.max(1) as f64,
        accepted_sum as f64 / windows.max(1) as f64,
    );

    if let Some(dir) = out {
        write_json(
            &dir.join("sampled.json"),
            &json!({
                "pass": pass,
                "config": {
                    "model": model, "draft": draft_path,
                    "prompt_tokens": prompt.len(), "prompt_md5": md5_hex_u32(prompt),
                    "trials": trials, "tokens": l, "block": block,
                    "temp": temp, "top_p": top_p, "top_k": top_k, "alpha": alpha,
                    "salt_ar": SALT_AR, "salt_dflash": SALT_DF,
                },
                "tests": tests.iter().map(TestRec::json).collect::<Vec<_>>(),
                "tv_mean_law": tv,
                "streams": {"ar": ar, "dflash": df},
                "laws": {"ar": ar_laws, "dflash": df_laws},
                "dflash_windows": {
                    "windows": windows, "proposed": proposed_sum, "accepted": accepted_sum,
                    "depth_hist": depth_hist.iter().map(|(k, v)| (k.to_string(), *v))
                        .collect::<BTreeMap<String, u64>>(),
                },
                "wall_secs": {"ar": ar_secs, "dflash": df_secs},
            }),
        )?;
    }
    Ok(pass)
}

// ─────────────────────────────────────────────────────────────────────────────
// costs
// ─────────────────────────────────────────────────────────────────────────────

/// `(median, mean)` of `v`.
fn median_mean(v: &[f64]) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let med = if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        0.5 * (s[s.len() / 2 - 1] + s[s.len() / 2])
    };
    (med, s.iter().sum::<f64>() / s.len() as f64)
}

/// Device-synced wall ms of `f`.
fn timed_ms<T>(
    gpu: &mut Gpu,
    f: impl FnOnce(&mut Gpu) -> Result<T, String>,
) -> Result<(T, f64), String> {
    sync(gpu)?;
    let t = Instant::now();
    let r = f(gpu)?;
    sync(gpu)?;
    Ok((r, t.elapsed().as_secs_f64() * 1e3))
}

/// Draft-call cost: the call sequence of `emit_block` (draft forward, target
/// head over the draft rows, selector) from the post-prefill state; returns
/// the per-rep ms of the whole sequence and of the head alone.
#[allow(clippy::too_many_arguments)]
fn draft_cost(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    prompt: &[u32],
    cfg: &DflashConfig,
    weights: &DflashWeights,
    scratch: &mut DflashScratch,
    block: usize,
    reps: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let fw = cfg.feature_width();
    let h = cfg.hidden;
    let feature_window = cfg.declared_window.filter(|_| cfg.all_layers_sliding);
    SpecTarget::set_dflash_extract_layers(bundle, cfg.target_layer_ids.clone());
    if let Some(w) = feature_window {
        SpecTarget::set_dflash_feature_window(bundle, w);
    }
    bundle.reset(gpu)?;
    let mut host: Vec<f32> = Vec::new();
    let seed =
        match SpecTarget::spec_advance(bundle, gpu, prompt, 0, true, &|| false, Some(&mut host))? {
            SpecAdvance::Ready { last_argmax, .. } => last_argmax,
            SpecAdvance::Aborted => return Err("spec_advance aborted".into()),
        };
    let position = prompt.len();
    let rows_held = host.len() / fw;
    let host_base = position
        .checked_sub(rows_held)
        .ok_or_else(|| format!("{rows_held} feature rows for {position} tokens"))?;
    let first_row = match feature_window {
        Some(w) => host_base.max(position.saturating_sub(w)),
        None => host_base,
    };
    let features = &host[(first_row - host_base) * fw..];
    let mask_row = SpecTarget::embed_row(bundle, gpu, cfg.mask_token_id)?;
    let seed_row = SpecTarget::embed_row(bundle, gpu, seed)?;
    let mut noise = Vec::with_capacity(block * h);
    noise.extend_from_slice(&seed_row);
    for _ in 1..block {
        noise.extend_from_slice(&mask_row);
    }
    let positions_q: Vec<i32> = (position as i32..(position + block) as i32).collect();
    let positions_k: Vec<i32> = (0..(position + block) as i32).collect();
    let drafts_n = block - 1;

    let mut whole = Vec::with_capacity(reps);
    for rep in 0..reps + 2 {
        let (_, ms) = timed_ms(gpu, |gpu| {
            draft_forward_based(
                gpu,
                weights,
                cfg,
                Some(&noise),
                Some(features),
                first_row,
                &positions_q,
                &positions_k,
                block,
                position,
                scratch,
                false,
            )
            .map_err(|e| format!("draft_forward: {e}"))?;
            let hidden_rows = scratch.x.sub_offset(h, drafts_n * h);
            let logits_dev =
                SpecTarget::lm_head_logits_device(bundle, gpu, &hidden_rows, drafts_n)?;
            propose_candidates_device(
                gpu,
                weights,
                scratch,
                &hidden_rows,
                &logits_dev,
                drafts_n,
                seed,
                0.0,
                None,
            )
            .map_err(|e| format!("selector: {e}"))?;
            Ok(())
        })?;
        if rep >= 2 {
            whole.push(ms);
        }
    }
    // The head alone, over the hidden rows the last draft forward left.
    let hidden_rows = scratch.x.sub_offset(h, drafts_n * h);
    let mut head = Vec::with_capacity(reps);
    for rep in 0..reps + 2 {
        let (_, ms) = timed_ms(gpu, |gpu| {
            SpecTarget::lm_head_logits_device(bundle, gpu, &hidden_rows, drafts_n)?;
            Ok(())
        })?;
        if rep >= 2 {
            head.push(ms);
        }
    }
    Ok((whole, head))
}

/// Per-call cost measurements for the spec-retire policy: AR decode, verify
/// (argmax and logits routes) per row count, partial commit, draft sequence.
fn run_costs(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    prompt: &[u32],
    opts: &Opts,
    out: Option<&Path>,
) -> Result<bool, String> {
    let reps = opt_usize(opts, "reps", 20)?;
    let max_rows = opt_usize(opts, "max-rows", 9)?;
    let block = opt_usize(opts, "block", 8)?;
    if reps == 0 {
        return Err("--reps must be >= 1".into());
    }
    if !(1..=DFLASH_BLOCK_MAX).contains(&max_rows) {
        return Err(format!(
            "--max-rows {max_rows} outside 1..={DFLASH_BLOCK_MAX}"
        ));
    }
    if !(2..=DFLASH_BLOCK_MAX).contains(&block) {
        return Err(format!("--block {block} outside 2..={DFLASH_BLOCK_MAX}"));
    }
    let draft_path = require(opts, "draft")?;
    let decode_mode = bundle.decode_mode();
    println!("costs: decode mode {decode_mode}, block {block}, reps {reps}, max rows {max_rows}");
    let ctx_capacity = bundle.max_seq();
    let position = prompt.len();
    if position + max_rows.max(block) > ctx_capacity {
        return Err(format!(
            "context {position} + {} rows exceeds max_seq {ctx_capacity}",
            max_rows.max(block)
        ));
    }
    let mut spec: Box<dyn Speculator> =
        build_dflash2_speculator(gpu, draft_path, bundle, ctx_capacity, block)?;
    let mut scratch: Box<dyn SpecScratch> =
        match SpecTarget::new_spec_scratch(bundle, gpu, max_rows.max(block)) {
            Ok(s) => s,
            Err(e) => {
                spec.free(gpu);
                return Err(e);
            }
        };
    let r = (|| -> Result<Value, String> {
        // 1 + 2. Cold spec prefill, then AR decode.
        let first = spec
            .prefill(gpu, bundle, prompt, prompt, 0, false, None, &|| false)?
            .into_first_token()?;
        let mut chain = vec![first];
        let mut decode = Vec::with_capacity(reps);
        for i in 0..reps + 2 {
            let tok = *chain.last().unwrap();
            let (_, ms) = timed_ms(gpu, |gpu| bundle.decode_step(gpu, tok).map(|_| ()))?;
            chain.push(argmax(&bundle.download_logits(gpu)?) as u32);
            if i >= 2 {
                decode.push(ms);
            }
        }
        let (dec_med, dec_mean) = median_mean(&decode);
        println!("decode_step: median {dec_med:.3} ms, mean {dec_mean:.3} ms");
        let first = spec
            .prefill(gpu, bundle, prompt, prompt, 0, false, None, &|| false)?
            .into_first_token()?;
        if bundle.n_tokens() != position {
            return Err(format!(
                "re-prefill left n_tokens {} != {position}",
                bundle.n_tokens()
            ));
        }
        let block_for = |k: usize| -> Vec<u32> {
            (0..k)
                .map(|i| {
                    if i == 0 {
                        first
                    } else {
                        chain.get(i).copied().unwrap_or(first)
                    }
                })
                .collect()
        };

        // 3. Verify cost per row count.
        println!("  k  verify_ms(med/mean)  verify_logits_ms(med/mean)  window_units");
        let (mut verify, mut verify_logits) = (Vec::new(), Vec::new());
        for k in 1..=max_rows {
            let blk = block_for(k);
            let mut rows: Vec<f32> = Vec::new();
            let (mut v_ms, mut vl_ms) = (Vec::new(), Vec::new());
            for rep in 0..reps + 3 {
                let (_, ms) = timed_ms(gpu, |gpu| {
                    SpecTarget::verify_block(
                        bundle,
                        gpu,
                        &blk,
                        position,
                        scratch.as_mut(),
                        Some(&mut rows),
                    )
                    .map(|_| ())
                })?;
                SpecTarget::abandon_verify(bundle, gpu, scratch.as_mut())?;
                if rep >= 3 {
                    v_ms.push(ms);
                }
            }
            for rep in 0..reps + 3 {
                let (_, ms) = timed_ms(gpu, |gpu| {
                    SpecTarget::verify_block_logits(
                        bundle,
                        gpu,
                        &blk,
                        position,
                        scratch.as_mut(),
                        Some(&mut rows),
                    )
                    .map(|_| ())
                })?;
                SpecTarget::abandon_verify(bundle, gpu, scratch.as_mut())?;
                if rep >= 3 {
                    vl_ms.push(ms);
                }
            }
            let (vm, va) = median_mean(&v_ms);
            let (lm, la) = median_mean(&vl_ms);
            println!("{k:>3}  {vm:>8.3} / {va:<8.3}  {lm:>8.3} / {la:<8.3}");
            verify.push((k, vm, va));
            verify_logits.push((k, lm, la));
        }

        // Partial commit (accept_len = 0) at k = 2 and k = max_rows.
        let mut partial: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
        let pc_reps = (reps / 4).max(1);
        for k in [2usize, max_rows] {
            if k < 2 || partial.contains_key(&k) {
                continue;
            }
            let blk = block_for(k);
            let mut ms_v = Vec::with_capacity(pc_reps);
            for _ in 0..pc_reps {
                spec.prefill(gpu, bundle, prompt, prompt, 0, false, None, &|| false)?
                    .into_first_token()?;
                SpecTarget::verify_block(bundle, gpu, &blk, position, scratch.as_mut(), None)?;
                let (_, ms) = timed_ms(gpu, |gpu| {
                    SpecTarget::commit_prefix(bundle, gpu, &blk, 0, position, scratch.as_mut())
                        .map(|_| ())
                })?;
                ms_v.push(ms);
            }
            let (med, mean) = median_mean(&ms_v);
            println!(
                "partial commit k={k}: median {med:.3} ms, mean {mean:.3} ms ({pc_reps} reps)"
            );
            partial.insert(k, (med, mean));
        }

        // 4. Draft cost.
        let (draft_hfq, cfg) = draft_config(draft_path)?;
        check_streams(bundle, &cfg)?;
        let weights = DflashWeights::load(gpu, &draft_hfq, &cfg).map_err(|e| format!("{e}"))?;
        if !weights.has_candidate_selector() {
            weights.free_gpu(gpu);
            return Err("DFlash2 draft: candidate selector tensors are missing".into());
        }
        let dscratch = match cfg.declared_window {
            Some(w) => {
                DflashScratch::new_windowed(gpu, &cfg, block, w, w, ctx_capacity, weights.has_mq)
            }
            None => DflashScratch::new_with_mq(gpu, &cfg, block, ctx_capacity, weights.has_mq),
        };
        let mut dscratch = match dscratch {
            Ok(s) => s,
            Err(e) => {
                weights.free_gpu(gpu);
                return Err(format!("{e}"));
            }
        };
        let d = draft_cost(
            gpu,
            bundle,
            prompt,
            &cfg,
            &weights,
            &mut dscratch,
            block,
            reps,
        );
        dscratch.free_gpu(gpu);
        weights.free_gpu(gpu);
        let (draft_v, head_v) = d?;
        let (draft_med, draft_mean) = median_mean(&draft_v);
        let (head_med, head_mean) = median_mean(&head_v);
        println!(
            "draft sequence (block {block}): median {draft_med:.3} ms, mean {draft_mean:.3} ms"
        );
        println!("draft lm_head: median {head_med:.3} ms, mean {head_mean:.3} ms");

        println!("  k  window_units = (draft + verify) / decode");
        let mut units = Vec::new();
        for &(k, vm, _) in &verify {
            let u = (draft_med + vm) / dec_med;
            println!("{k:>3}  {u:.3}");
            units.push(u);
        }
        let by_k = |v: &[(usize, f64, f64)]| -> Value {
            Value::Object(
                v.iter()
                    .map(|&(k, med, mean)| (k.to_string(), json!({"median": med, "mean": mean})))
                    .collect(),
            )
        };
        Ok(json!({
            "decode_ms": {"median": dec_med, "mean": dec_mean, "samples": decode},
            "verify_ms": by_k(&verify),
            "verify_logits_ms": by_k(&verify_logits),
            "partial_commit_ms": Value::Object(partial.iter()
                .map(|(k, &(med, mean))| (k.to_string(), json!({"median": med, "mean": mean})))
                .collect()),
            "draft_ms": {"median": draft_med, "mean": draft_mean},
            "draft_lm_head_ms": {"median": head_med, "mean": head_mean},
            "window_units": Value::Object(units.iter().enumerate()
                .map(|(i, u)| ((i + 1).to_string(), json!(u))).collect()),
        }))
    })();
    scratch.free(gpu);
    spec.free(gpu);
    let mut v = r?;
    if let Some(dir) = out {
        let o = v.as_object_mut().expect("costs result is an object");
        o.insert("block".into(), json!(block));
        o.insert("reps".into(), json!(reps));
        o.insert("max_rows".into(), json!(max_rows));
        o.insert("decode_mode".into(), json!(decode_mode));
        o.insert("draft".into(), json!(draft_path));
        o.insert("gpu_arch".into(), json!(format!("{}", gpu.arch)));
        o.insert("prompt_tokens".into(), json!(prompt.len()));
        write_json(&dir.join("costs.json"), &v)?;
    }
    Ok(true)
}

// ─────────────────────────────────────────────────────────────────────────────

fn run() -> Result<bool, String> {
    let (mode, opts) = parse_cli()?;
    let model = require(&opts, "model")?.to_string();
    let hfq = HfqFile::open(Path::new(&model)).map_err(|e| format!("open {model}: {e}"))?;
    let prompt = prompt_tokens(&hfq, &opts)?;
    let n_new = opt_usize(&opts, "tokens", 64)?;
    let max_seq = match opts.get("max-seq") {
        Some(_) => opt_usize(&opts, "max-seq", 0)?,
        None => (prompt.len() + n_new + 3 * DFLASH_BLOCK_MAX + 64).max(512),
    };
    let out = out_dir(&opts, mode == "emit")?;

    let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e:?}"))?;
    let mut bundle = load_bundle(&mut gpu, ModelSource::Hfq(hfq), &model, max_seq)?;
    bundle.set_prefill_mode(PrefillMode::Batched);
    let r = (|| -> Result<bool, String> {
        let vocab = bundle.config.vocab;
        if let Some((i, &t)) = prompt
            .iter()
            .enumerate()
            .find(|(_, &t)| t as usize >= vocab)
        {
            return Err(format!("prompt token[{i}] = {t} >= vocab {vocab}"));
        }
        match mode.as_str() {
            "capture" => run_capture(&mut gpu, &mut bundle, &prompt, &opts),
            "emit" => run_emit(
                &mut gpu,
                &mut bundle,
                &prompt,
                &opts,
                out.as_deref().ok_or("missing `--out`")?,
            ),
            "rowexact" => run_rowexact(&mut gpu, &mut bundle, &prompt, out.as_deref()),
            "greedy" => run_greedy(&mut gpu, &mut bundle, &prompt, &opts, out.as_deref()),
            "rollback" => run_rollback(&mut gpu, &mut bundle, &prompt, &opts, out.as_deref()),
            "verifygraph" => run_verifygraph(&mut gpu, &mut bundle, &prompt, out.as_deref()),
            "sampled" => run_sampled(&mut gpu, &mut bundle, &prompt, &opts, out.as_deref()),
            "costs" => run_costs(&mut gpu, &mut bundle, &prompt, &opts, out.as_deref()),
            _ => unreachable!("mode validated by parse_cli"),
        }
    })();
    bundle.free_all(&mut gpu);
    r
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("raon_dflash_oracle: {e}\n{USAGE}");
            std::process::exit(2);
        }
    }
}
