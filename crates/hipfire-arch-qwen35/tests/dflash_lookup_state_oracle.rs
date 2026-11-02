// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! DFlash2 verbatim-lookup splice (`spec_step_dflash`'s `lookup` argument)
//! state oracle, singleton dense 27B target + paired DFlash2 draft.
//!
//! The splice verifies `b_eff = 1 + K + L <= b` rows and every later stage
//! (verify, accept, hidden ring scatter, GDN tape / DeltaNet rollback, thlog
//! cursor) must run at `b_eff`. This oracle drives `spec_step_dflash` itself
//! with scripted lookups and compares complete committed state:
//!
//! * Determinism: every scenario runs twice from a fresh prefill of the same
//!   prompt; after the prefill and after every window the committed tokens,
//!   bonus, verified block, RNG state, KV rows `[0, pos)` of every attention
//!   layer, every DeltaNet buffer (state, scales, error feedback, conv), the
//!   most recent hidden-ring rows (all extract layers), the draft's
//!   `target_hidden` rows, the GDN tape rows `[0, b_eff)` and the thlog
//!   cursors must be byte-identical between the two runs. Each scenario is one
//!   lookup window followed by three incumbent windows (so the "next-window
//!   drafted block" and the rolled-back state are exercised), or, for the
//!   `tail_*` scenarios, a few incumbent windows to reach a window whose learned
//!   block is known, then the lookup window, then three incumbent windows.
//! * Incumbent identity: a schedule whose Tail source always declines
//!   (`splice` returns `None`) is byte-identical to `lookup = None`, greedy and
//!   sampled (same rows, kernels, RNG draws).
//! * Splice-boundary coverage (accepted = 0 / partial / full for Copy; tail
//!   rejected at its first token / partial / full, learned-prefix rejection
//!   below K, EOS inside the lookup tokens, `max_accept` clamp) with the
//!   forced `accepted` asserted against the reference continuation.
//!
//! `ref` column: the committed stream of each scenario against the plain
//! incumbent greedy reference run. Different verify row counts may flip a
//! near-tie logit, so a DIFF is reported but is not a failure; every other
//! column is.
//!
//! ```bash
//! HIPFIRE_DEVICES=<uuid> HIPFIRE_TEST_TARGET=<27b.mq4> HIPFIRE_TEST_DRAFT=<dflash2.hfq> \
//!   cargo test --release -p hipfire-arch-qwen35 --features deltanet \
//!   --test dflash_lookup_state_oracle -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Each `#[ignore]` test loads its own 27B pair and does not free it: run them
//! one per process (`--exact dflash_lookup_state_oracle`, then
//! `--exact dflash_lookup_sampled_law`).

use hipfire_arch_qwen35::dflash_spec::{load_dflash_state, DflashState};
use hipfire_arch_qwen35::qwen35::LayerType;
use hipfire_arch_qwen35::speculative::{
    scatter_hidden_block_to_interleaved, seed_target_hidden_from_prompt, spec_step_dflash,
    DeltaNetSnapshot, DflashLookup, DflashTailSource, KvMode, ModelSlot, ModelSlotConfig,
    SpecStepResult,
};
use hipfire_runtime::dflash::TargetHiddenLogMark;
use hipfire_runtime::llama::KvCache;
use rdna_compute::{Gpu, GpuTensor};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;

type R<T> = Result<T, Box<dyn std::error::Error>>;

const TARGET_ENV: &str = "HIPFIRE_TEST_TARGET";
const DRAFT_ENV: &str = "HIPFIRE_TEST_DRAFT";
/// Target KV / draft context capacity.
const CTX: usize = 4096;
/// Incumbent windows after the lookup window of every scenario.
const TRAILING: usize = 3;
/// Most recent ring / target_hidden rows digested per window.
const RECENT_ROWS: usize = 17;
const PASSAGE: &str = "The harbor bell rang twice before dawn, and the gulls answered in kind. \
Lanterns swayed along the pier while the tide turned slowly toward the old stones. \
A fisherman coiled his rope, counted the nets twice, and walked home without a word.";

// ── device helpers ───────────────────────────────────────────────────────

fn read_dev(gpu: &Gpu, t: &GpuTensor, offset: usize, bytes: usize) -> R<Vec<u8>> {
    if offset + bytes > t.buf.size() {
        return Err(format!("read {offset}+{bytes} exceeds allocation {}", t.buf.size()).into());
    }
    let mut raw = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh_at(&mut raw, &t.buf, offset)?;
    Ok(raw)
}

fn hex(d: impl AsRef<[u8]>) -> String {
    d.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

/// Per-position KV row stride for the loaded cache encoding.
fn kv_row_bytes(kv: &KvCache) -> R<(usize, usize)> {
    if kv.quant_fp8 {
        let r = KvCache::fp8_row_bytes(kv.n_kv_heads, kv.head_dim)?;
        return Ok((r, r));
    }
    if kv.quant_bf16 {
        let r = KvCache::bf16_row_bytes(kv.n_kv_heads, kv.head_dim)?;
        return Ok((r, r));
    }
    let rotated = kv.quant_asym2 || kv.quant_asym3 || kv.quant_asym4 || kv.quant_fwht;
    if kv.quant_q8 && !rotated {
        let r = kv.n_kv_heads * (kv.head_dim / 32) * 34;
        return Ok((r, r));
    }
    Err("oracle supports the fp8/q8/bf16 KV encodings only".into())
}

// ── rig ──────────────────────────────────────────────────────────────────

struct Rig {
    gpu: Gpu,
    slot: ModelSlot,
    df: DflashState,
    host_hidden: Vec<f32>,
    prompt: Vec<u32>,
    eos: u32,
    vocab: usize,
}

impl Rig {
    fn open() -> R<Option<Rig>> {
        let (Some(target), Some(draft)) = (
            std::env::var(TARGET_ENV).ok().map(PathBuf::from),
            std::env::var(DRAFT_ENV).ok().map(PathBuf::from),
        ) else {
            eprintln!("skipping: {TARGET_ENV} / {DRAFT_ENV} not set");
            return Ok(None);
        };
        let mut gpu = Gpu::init()?;
        let mut cfg = ModelSlotConfig::default();
        cfg.max_seq = CTX + 64;
        cfg.kv_mode = KvMode::Q8;
        let slot = ModelSlot::load(&mut gpu, &target, "lookup-oracle", cfg)
            .map_err(|e| format!("load target: {e:?}"))?;
        if slot.config.num_experts > 0 {
            return Err("dense targets only".into());
        }
        let kv_q8 = {
            let kv = &slot.kv_cache;
            kv.quant_q8
                && !kv.quant_fwht
                && !kv.quant_asym2
                && !kv.quant_asym3
                && !kv.quant_asym4
                && matches!(kv.v_mode, hipfire_runtime::llama::VMode::Q8)
        };
        let draft_str = draft.to_str().ok_or("draft path is not UTF-8")?;
        let df = load_dflash_state(
            draft_str,
            CTX,
            &slot.config,
            &slot.dn_state,
            &mut gpu,
            None,
            None,
            false,
            &slot.weights,
            kv_q8,
            slot.kv_cache.uses_vmm_backend(),
            slot.scratch.flash_partials.numel(),
            true,
            false,
            16,
        )?;
        if !df.draft_weights.has_candidate_selector() {
            eprintln!(
                "skipping: draft has no DFlash2 candidate selector (lookup is selector-mode only)"
            );
            return Ok(None);
        }
        let tok = slot
            .load_tokenizer()
            .map_err(|e| format!("tokenizer: {e:?}"))?;
        let prompt: Vec<u32> = [
            "<|im_start|>",
            "user",
            "\n",
            "Repeat the following text exactly, with no other words:\n\n",
            PASSAGE,
            "<|im_end|>",
            "\n",
            "<|im_start|>",
            "assistant",
            "\n",
            "<think>\n\n</think>\n\n",
        ]
        .iter()
        .flat_map(|s| tok.encode(s))
        .collect();
        // The chat turn ends with <|im_end|>; fall back to the config EOS.
        let im_end = tok.encode("<|im_end|>");
        let eos = if im_end.len() == 1 {
            im_end[0]
        } else {
            slot.config.eos_token
        };
        let vocab = slot.config.vocab_size;
        eprintln!(
            "[lookup-oracle] gpu={} prompt={} tokens, draft block {} (runtime {}), eos={eos}",
            gpu.arch,
            prompt.len(),
            df.draft_config.block_size,
            df.block_size
        );
        Ok(Some(Rig {
            gpu,
            slot,
            df,
            host_hidden: Vec::new(),
            prompt,
            eos,
            vocab,
        }))
    }

    /// Fresh request: reset every per-request owner exactly like the demo's
    /// per-row reset, prefill the prompt, prime the draft. Returns the first
    /// (greedy) token.
    fn fresh_prefill(&mut self) -> R<u32> {
        self.host_hidden.clear();
        self.slot.kv_cache.compact_offset = 0;
        self.df.draft_scratch.reset_upload_tracking();
        self.df.hidden_rb.reset();
        // With GDN error feedback off, Q8 DeltaNet requant rounds off a
        // process-global frame counter: pin it so two fresh runs are bytewise
        // comparable.
        if self.slot.dn_state.s_ef_residual.is_empty() {
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(0);
        }
        seed_target_hidden_from_prompt(
            &mut self.gpu,
            &mut self.slot,
            &mut self.df.hidden_rb,
            &mut self.host_hidden,
            &self.prompt,
        )?;
        let n = self.prompt.len();
        scatter_hidden_block_to_interleaved(
            &self.gpu,
            &self.df.hidden_rb,
            &self.df.draft_scratch.target_hidden,
            0,
            n,
            n,
            self.df.draft_scratch.ctx_modulus(),
        )?;
        hipfire_runtime::dflash::draft_seed_backfill(
            &mut self.gpu,
            &self.df.draft_weights,
            &self.df.draft_config,
            &mut self.df.draft_scratch,
            &self.host_hidden,
            n,
        )?;
        self.df.draft_scratch.thlog.seed_prompt(n);
        let logits = self.gpu.download_f32(&self.slot.scratch.logits)?;
        let mut best = (0u32, f32::NEG_INFINITY);
        for (i, &v) in logits.iter().enumerate() {
            if v > best.1 {
                best = (i as u32, v);
            }
        }
        Ok(best.0)
    }
}

// ── scripted lookups ─────────────────────────────────────────────────────

/// Tail source: always keeps the first `k` learned candidates and appends
/// `tail`; `k == 0` declines (incumbent window). Not a sequential-stopping
/// rule on purpose: this is a state oracle (greedy forced blocks), not a law
/// test; the sampled-law proof is the CPU `lookup_law_*` suite.
struct Script {
    k: usize,
    tail: Vec<u32>,
    calls: usize,
}

impl DflashTailSource for Script {
    fn splice(&mut self, _drafted: &[u32], tail: &mut Vec<u32>) -> Option<usize> {
        self.calls += 1;
        if self.k == 0 {
            return None;
        }
        tail.clear();
        tail.extend_from_slice(&self.tail);
        Some(self.k)
    }
}

/// Tail source that splices `tail` after K = 1 kept learned row only when the
/// first learned candidate is exactly `want` (a sequential stopping rule: the
/// decision reads `drafted[..1]` only), otherwise declines.
struct GatedTail {
    want: u32,
    tail: Vec<u32>,
}

impl DflashTailSource for GatedTail {
    fn splice(&mut self, drafted: &[u32], tail: &mut Vec<u32>) -> Option<usize> {
        if drafted.first() == Some(&self.want) {
            tail.clear();
            tail.extend_from_slice(&self.tail);
            Some(1)
        } else {
            None
        }
    }
}

enum Look {
    None,
    Copy(Vec<u32>),
    Tail(Script),
    Gated(GatedTail),
}

struct Step {
    look: Look,
    max_accept: Option<usize>,
}

impl Step {
    fn none() -> Self {
        Step {
            look: Look::None,
            max_accept: None,
        }
    }
    fn copy(tokens: Vec<u32>, max_accept: Option<usize>) -> Self {
        Step {
            look: Look::Copy(tokens),
            max_accept,
        }
    }
    fn tail(k: usize, tail: Vec<u32>, max_accept: Option<usize>) -> Self {
        Step {
            look: Look::Tail(Script { k, tail, calls: 0 }),
            max_accept,
        }
    }
}

#[derive(Clone, Copy)]
struct Samp {
    temp: f32,
    top_p: f32,
    top_k: usize,
}

const GREEDY: Samp = Samp {
    temp: 0.0,
    top_p: 1.0,
    top_k: 0,
};
const SAMPLED: Samp = Samp {
    temp: 0.7,
    top_p: 0.95,
    top_k: 20,
};

struct Cursor {
    pos: usize,
    seed: u32,
    rng: u64,
    emitted: Vec<u32>,
}

fn step_once(rig: &mut Rig, cur: &mut Cursor, samp: Samp, step: &mut Step) -> R<SpecStepResult> {
    let lookup = match &mut step.look {
        Look::None => None,
        Look::Copy(v) => Some(DflashLookup::Copy(&v[..])),
        Look::Tail(s) => Some(DflashLookup::Tail(s)),
        Look::Gated(s) => Some(DflashLookup::Tail(s)),
    };
    let r = spec_step_dflash(
        &mut rig.gpu,
        &mut rig.slot,
        &rig.df.draft_weights,
        &rig.df.draft_config,
        &mut rig.df.draft_scratch,
        &mut rig.df.hidden_rb,
        &mut rig.host_hidden,
        &mut rig.df.target_snap,
        &rig.df.verify_scratch,
        cur.pos,
        cur.seed,
        None,
        Some(&mut rig.df.gdn_tape),
        samp.temp,
        samp.top_p,
        samp.top_k,
        &mut cur.rng,
        Some(rig.df.block_size),
        None,
        &cur.emitted,
        0.0,
        None,
        1.0,
        0,
        step.max_accept,
        Some(&mut rig.df.verify_pm4),
        lookup,
    )?;
    cur.pos += r.accepted + 1;
    cur.seed = r.bonus_token;
    cur.emitted.extend_from_slice(&r.committed[1..]);
    Ok(r)
}

// ── state capture ────────────────────────────────────────────────────────

#[derive(PartialEq, Debug)]
struct Rec {
    accepted: usize,
    bonus: u32,
    drafted: Vec<u32>,
    committed: Vec<u32>,
    pos: usize,
    rng: u64,
    parts: Vec<(&'static str, String)>,
}

fn capture(rig: &Rig, r: Option<&SpecStepResult>, pos: usize, rng: u64) -> R<Rec> {
    rig.gpu.hip.device_synchronize()?;
    let gpu = &rig.gpu;
    let mut parts: Vec<(&'static str, String)> = Vec::new();

    // KV rows [0, pos) of every attention layer.
    {
        let kv = &rig.slot.kv_cache;
        let (kr, vr) = kv_row_bytes(kv)?;
        let mut h = Sha256::new();
        for (layer, ty) in rig.slot.config.layer_types.iter().enumerate() {
            if *ty == LayerType::FullAttention {
                h.update(read_dev(gpu, &kv.k_gpu[layer], 0, pos * kr)?);
                h.update(read_dev(gpu, &kv.v_gpu[layer], 0, pos * vr)?);
            }
        }
        parts.push(("kv", hex(h.finalize())));
    }
    // DeltaNet state, scales, error feedback and conv state (whole buffers).
    {
        let dn = &rig.slot.dn_state;
        let mut h = Sha256::new();
        for t in dn
            .s_matrices
            .iter()
            .chain(&dn.s_scales)
            .chain(&dn.s_ef_residual)
            .chain(&dn.conv_states)
        {
            h.update(read_dev(gpu, t, 0, t.buf.size())?);
        }
        parts.push(("deltanet", hex(h.finalize())));
    }
    // Most recent hidden-ring rows of every extract layer.
    {
        let rb = &rig.df.hidden_rb;
        let row = rb.hidden_dim * 4;
        let mut h = Sha256::new();
        for i in 0..rb.written.min(RECENT_ROWS) {
            let slot = (rb.head + rb.max_positions - 1 - i) % rb.max_positions;
            for t in &rb.layer_bufs {
                h.update(read_dev(gpu, t, slot * row, row)?);
            }
        }
        h.update(format!("head={} written={}", rb.head, rb.written).as_bytes());
        parts.push(("ring", hex(h.finalize())));
    }
    // The draft's interleaved target_hidden rows of the most recent positions.
    {
        let ds = &rig.df.draft_scratch;
        let row = rig.df.draft_config.num_extract() * rig.df.draft_config.hidden * 4;
        let modulus = ds.ctx_modulus();
        let mut h = Sha256::new();
        for p in pos.saturating_sub(RECENT_ROWS)..pos {
            let r = if modulus == usize::MAX {
                p
            } else {
                p % modulus
            };
            h.update(read_dev(gpu, &ds.target_hidden, r * row, row)?);
        }
        parts.push(("target_hidden", hex(h.finalize())));
    }
    // GDN tape rows [0, b_eff) (the verified window).
    if let Some(r) = r {
        let tape = &rig.df.gdn_tape;
        let n = r.drafted.len();
        let mut h = Sha256::new();
        for ((q, a), b) in tape
            .qkv_bufs
            .iter()
            .zip(&tape.alpha_bufs)
            .zip(&tape.beta_bufs)
        {
            h.update(read_dev(gpu, q, 0, n * tape.qkv_dim * 4)?);
            h.update(read_dev(gpu, a, 0, n * tape.n_v_heads * 4)?);
            h.update(read_dev(gpu, b, 0, n * tape.n_v_heads * 4)?);
        }
        parts.push(("tape", hex(h.finalize())));
    }
    let th = &rig.df.draft_scratch.thlog;
    parts.push((
        "thlog",
        format!(
            "uploaded={} abs_len={} abs_last={:?} proj={} full={}",
            th.uploaded_rows(),
            th.abs_positions().len(),
            th.abs_positions().last(),
            th.proj_cached_rows(),
            th.full_cached_rows()
        ),
    ));
    Ok(match r {
        Some(r) => Rec {
            accepted: r.accepted,
            bonus: r.bonus_token,
            drafted: r.drafted.clone(),
            committed: r.committed.clone(),
            pos,
            rng,
            parts,
        },
        None => Rec {
            accepted: 0,
            bonus: 0,
            drafted: vec![],
            committed: vec![],
            pos,
            rng,
            parts,
        },
    })
}

struct Run {
    /// recs[0] = after prefill, recs[1 + i] = after step i.
    recs: Vec<Rec>,
    emitted: Vec<u32>,
    /// `splice` calls per Tail step.
    tail_calls: Vec<usize>,
}

fn run_schedule(rig: &mut Rig, samp: Samp, mut steps: Vec<Step>) -> R<Run> {
    let first = rig.fresh_prefill()?;
    let mut cur = Cursor {
        pos: rig.prompt.len(),
        seed: first,
        rng: 0x1D2C_3B4A_5968_7786 | 1,
        emitted: vec![first],
    };
    let mut recs = vec![capture(rig, None, cur.pos, cur.rng)?];
    for step in steps.iter_mut() {
        let r = step_once(rig, &mut cur, samp, step)?;
        recs.push(capture(rig, Some(&r), cur.pos, cur.rng)?);
    }
    let tail_calls = steps
        .iter()
        .filter_map(|s| match &s.look {
            Look::Tail(t) => Some(t.calls),
            _ => None,
        })
        .collect();
    Ok(Run {
        recs,
        emitted: cur.emitted,
        tail_calls,
    })
}

fn first_diff(a: &Run, b: &Run) -> Option<String> {
    if a.recs.len() != b.recs.len() {
        return Some(format!("window count {} vs {}", a.recs.len(), b.recs.len()));
    }
    for (i, (x, y)) in a.recs.iter().zip(&b.recs).enumerate() {
        let what = if x.accepted != y.accepted {
            "accepted"
        } else if x.bonus != y.bonus {
            "bonus"
        } else if x.drafted != y.drafted {
            "drafted block"
        } else if x.committed != y.committed {
            "committed"
        } else if x.pos != y.pos {
            "position"
        } else if x.rng != y.rng {
            "rng state"
        } else if x.parts != y.parts {
            let p = x.parts.iter().zip(&y.parts).find(|(p, q)| p != q);
            return Some(format!(
                "state after window {i}: {}",
                p.map_or("parts".to_string(), |(p, _)| p.0.to_string())
            ));
        } else {
            continue;
        };
        return Some(format!("window {i}: {what}"));
    }
    None
}

// ── reference continuation ───────────────────────────────────────────────

struct RefWin {
    seed_idx: usize,
    accepted: usize,
}

struct Reference {
    /// Greedy stream incl. the first token, truncated after the first EOS.
    all: Vec<u32>,
    wins: Vec<RefWin>,
    eos_at: Option<usize>,
}

impl Reference {
    fn cont(&self, w: usize) -> &[u32] {
        &self.all[(self.wins[w].seed_idx + 1).min(self.all.len())..]
    }
}

fn reference_run(rig: &mut Rig) -> R<Reference> {
    let first = rig.fresh_prefill()?;
    let mut cur = Cursor {
        pos: rig.prompt.len(),
        seed: first,
        rng: 0x1D2C_3B4A_5968_7786 | 1,
        emitted: vec![first],
    };
    let mut wins = Vec::new();
    for _ in 0..40 {
        if cur.emitted.contains(&rig.eos) || cur.emitted.len() >= 120 {
            break;
        }
        let seed_idx = cur.emitted.len() - 1;
        let r = step_once(rig, &mut cur, GREEDY, &mut Step::none())?;
        wins.push(RefWin {
            seed_idx,
            accepted: r.accepted,
        });
    }
    let mut all = cur.emitted;
    let eos_at = all.iter().position(|&t| t == rig.eos);
    if let Some(e) = eos_at {
        all.truncate(e + 1);
    }
    eprintln!(
        "[lookup-oracle] reference: {} tokens, eos_at={eos_at:?}, window accepts {:?}",
        all.len(),
        wins.iter().map(|w| w.accepted).collect::<Vec<_>>()
    );
    Ok(Reference { all, wins, eos_at })
}

// ── scenarios ────────────────────────────────────────────────────────────

struct Scenario {
    name: String,
    samp: Samp,
    /// Index of the lookup step.
    at: usize,
    expect_accepted: Option<usize>,
    expect_rows: Option<usize>,
    build: Box<dyn Fn() -> Vec<Step>>,
}

fn wrong(t: u32, vocab: usize) -> u32 {
    ((t as usize + 1) % vocab) as u32
}

/// `pre` incumbent windows, the lookup step, `TRAILING` incumbent windows.
fn schedule(pre: usize, lookup: impl Fn() -> Step + 'static) -> Box<dyn Fn() -> Vec<Step>> {
    Box::new(move || {
        let mut v: Vec<Step> = (0..pre).map(|_| Step::none()).collect();
        v.push(lookup());
        v.extend((0..TRAILING).map(|_| Step::none()));
        v
    })
}

fn scenarios(rig: &Rig, rf: &Reference) -> (Vec<Scenario>, Vec<(String, String)>) {
    let vocab = rig.vocab;
    let eos = rig.eos;
    let mut out = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    let w0: Vec<u32> = rf.cont(0).to_vec();

    // ── Copy windows at the first window ─────────────────────────────────
    let mut copy =
        |name: &str, pre: usize, tokens: Vec<u32>, max_accept: Option<usize>, expect: usize| {
            let rows = 1 + tokens.len();
            out.push(Scenario {
                name: name.into(),
                samp: GREEDY,
                at: pre,
                expect_accepted: Some(expect),
                expect_rows: Some(rows),
                build: schedule(pre, move || Step::copy(tokens.clone(), max_accept)),
            });
        };
    if w0.len() >= 8 {
        copy("copy_zero", 0, vec![wrong(w0[0], vocab)], None, 0);
        let mut p = w0[..7].to_vec();
        p.push(wrong(w0[7], vocab));
        copy("copy_partial7", 0, p, None, 7);
    } else {
        skipped.push((
            "copy_zero/partial".into(),
            format!("reference continuation too short ({})", w0.len()),
        ));
    }
    if w0.len() >= 15 {
        copy("copy_full15_b16", 0, w0[..15].to_vec(), None, 15);
        copy("copy_full15_clamp5", 0, w0[..15].to_vec(), Some(5), 5);
    } else {
        skipped.push((
            "copy_full15".into(),
            format!("reference continuation too short ({})", w0.len()),
        ));
    }
    if w0.len() >= 4 {
        copy("copy_full4_b5", 0, w0[..4].to_vec(), None, 4);
    }
    // EOS inside the lookup tokens: the real EOS at the first reference
    // window whose continuation reaches it within 15 tokens (the copy window
    // is placed there, accepted runs through the EOS), plus a synthetic EOS
    // the target rejects at the first window.
    let eos_win = rf.eos_at.and_then(|e| {
        rf.wins.iter().enumerate().find_map(|(w, r)| {
            let idx = e.checked_sub(r.seed_idx + 1)?;
            (idx < 15).then_some((w, idx))
        })
    });
    match eos_win {
        Some((w, idx)) => {
            let t = rf.cont(w)[..=idx].to_vec();
            copy(&format!("copy_eos_real@w{w}"), w, t, None, idx + 1);
        }
        None => skipped.push((
            "copy_eos_real".into(),
            "the reference never reached EOS within one 15-token window".into(),
        )),
    }
    if w0.len() >= 4 {
        copy(
            "copy_eos_synthetic",
            0,
            vec![w0[0], w0[1], eos, w0[3]],
            None,
            2,
        );
    } else {
        skipped.push((
            "copy_eos_synthetic".into(),
            "reference continuation too short".into(),
        ));
    }

    // ── Tail windows at a window whose learned block is known ───────────
    let rem = |w: usize| rf.cont(w).len();
    let tail_w = rf
        .wins
        .iter()
        .enumerate()
        .position(|(w, r)| r.accepted >= 2 && rem(w) >= r.accepted.min(5) + 6);
    match tail_w {
        Some(w) => {
            let m = rf.wins[w].accepted;
            let k = m.min(5);
            let cont: Vec<u32> = rf.cont(w).to_vec();
            let l_full = 8usize.min(15 - k).min(cont.len() - k);
            let mut tail =
                |name: &str, kk: usize, t: Vec<u32>, max_accept: Option<usize>, expect: usize| {
                    let rows = 1 + kk + t.len();
                    out.push(Scenario {
                        name: format!("{name}@w{w}"),
                        samp: GREEDY,
                        at: w,
                        expect_accepted: Some(expect),
                        expect_rows: Some(rows),
                        build: schedule(w, move || Step::tail(kk, t.clone(), max_accept)),
                    });
                };
            tail(
                "tail_full",
                k,
                cont[k..k + l_full].to_vec(),
                None,
                k + l_full,
            );
            let mut p = cont[k..k + 4].to_vec();
            p.push(wrong(cont[k + 4], vocab));
            tail("tail_partial", k, p, None, k + 4);
            tail("tail_zero_in_tail", k, vec![wrong(cont[k], vocab)], None, k);
            if l_full >= 3 {
                tail(
                    "tail_clamp",
                    k,
                    cont[k..k + l_full].to_vec(),
                    Some(k + 2),
                    k + 2,
                );
            }
            // EOS inside the tail: real when the reference's EOS falls in it.
            let eos_rel = rf.eos_at.and_then(|e| {
                let idx = e.checked_sub(rf.wins[w].seed_idx + 1)?;
                (idx >= k && idx - k + 1 <= l_full).then_some(idx)
            });
            match eos_rel {
                Some(idx) => tail("tail_eos_real", k, cont[k..=idx].to_vec(), None, idx + 1),
                None => {
                    let t = vec![cont[k], eos, cont[k + 2]];
                    tail("tail_eos_synthetic", k, t, None, k + 1);
                }
            }
        }
        None => skipped.push((
            "tail_full/partial/zero/clamp/eos".into(),
            "no reference window with accepted >= 2 and a long enough continuation".into(),
        )),
    }
    // Rejection inside the kept learned prefix (K = m + 1 keeps the learned
    // candidate the incumbent window rejected): any reference window with a
    // learned rejection (m <= 13 leaves at least one tail row in 16) and a
    // continuation reaching past K; the lowest-accept such window is used.
    let learned_w = rf
        .wins
        .iter()
        .enumerate()
        .filter(|(w, r)| r.accepted <= 13 && rem(*w) >= r.accepted + 2)
        .min_by_key(|(_, r)| r.accepted)
        .map(|(w, _)| w);
    match learned_w {
        Some(w) => {
            let m = rf.wins[w].accepted;
            let k = m + 1;
            let cont: Vec<u32> = rf.cont(w).to_vec();
            let l = 6usize.min(15 - k).min(cont.len() - k);
            let t = cont[k..k + l].to_vec();
            let rows = 1 + k + t.len();
            out.push(Scenario {
                name: format!("tail_learned_reject@w{w}"),
                samp: GREEDY,
                at: w,
                expect_accepted: Some(m),
                expect_rows: Some(rows),
                build: schedule(w, move || Step::tail(k, t.clone(), None)),
            });
        }
        None => skipped.push((
            "tail_learned_reject".into(),
            "no reference window with a learned rejection (accepted <= 13) and a continuation past it"
                .into(),
        )),
    }

    // ── Sampled determinism (CPU tests carry the law) ───────────────────
    if w0.len() >= 8 {
        let c = w0[..8].to_vec();
        out.push(Scenario {
            name: "sampled_copy8".into(),
            samp: SAMPLED,
            at: 0,
            expect_accepted: None,
            expect_rows: Some(9),
            build: schedule(0, move || Step::copy(c.clone(), None)),
        });
        let t = w0[2..8].to_vec();
        out.push(Scenario {
            name: "sampled_tail_k2".into(),
            samp: SAMPLED,
            at: 0,
            expect_accepted: None,
            expect_rows: Some(9),
            build: schedule(0, move || Step::tail(2, t.clone(), None)),
        });
    }
    (out, skipped)
}

struct Row {
    name: String,
    rows: String,
    accepted: String,
    expect: String,
    det: Result<(), String>,
    refcmp: String,
    ok: bool,
}

fn ref_cmp(emitted: &[u32], rf: &Reference) -> String {
    let n = emitted.len().min(rf.all.len());
    match (0..n).find(|&i| emitted[i] != rf.all[i]) {
        None => "MATCH".into(),
        Some(i) => format!("DIFF@{i}"),
    }
}

fn run_scenario(rig: &mut Rig, sc: &Scenario, rf: &Reference) -> R<Row> {
    let a = run_schedule(rig, sc.samp, (sc.build)())?;
    let b = run_schedule(rig, sc.samp, (sc.build)())?;
    let det = match first_diff(&a, &b) {
        None => Ok(()),
        Some(d) => Err(d),
    };
    let rec = &a.recs[1 + sc.at];
    let rows = rec.drafted.len();
    let mut ok = det.is_ok();
    let rows_ok = sc.expect_rows.map_or(true, |r| r == rows);
    let acc_ok = sc.expect_accepted.map_or(true, |x| x == rec.accepted);
    ok &= rows_ok && acc_ok;
    // The rejected-boundary bonus / clamp: every window stays inside its
    // verified block.
    ok &= a.recs.iter().skip(1).all(|r| r.accepted < r.drafted.len());
    Ok(Row {
        name: sc.name.clone(),
        rows: format!("{rows}{}", if rows_ok { "" } else { "!" }),
        accepted: format!("{}", rec.accepted),
        expect: sc.expect_accepted.map_or("-".into(), |x| {
            format!("{x}{}", if acc_ok { "" } else { "!" })
        }),
        det,
        refcmp: if sc.samp.temp > 0.0 {
            "n/a".into()
        } else {
            ref_cmp(&a.emitted, rf)
        },
        ok,
    })
}

/// A schedule whose Tail source always declines is byte-identical to
/// `lookup = None` (rows, kernels, RNG draws, state).
fn run_decline_equivalence(rig: &mut Rig, samp: Samp, name: &str) -> R<Row> {
    let windows = 4;
    let plain = run_schedule(rig, samp, (0..windows).map(|_| Step::none()).collect())?;
    let declined = run_schedule(
        rig,
        samp,
        (0..windows).map(|_| Step::tail(0, vec![], None)).collect(),
    )?;
    let mut det = match first_diff(&plain, &declined) {
        None => Ok(()),
        Some(d) => Err(d),
    };
    if det.is_ok() && declined.tail_calls != vec![1; windows] {
        det = Err(format!(
            "source consulted {:?} times per window, expected once",
            declined.tail_calls
        ));
    }
    let ok = det.is_ok();
    Ok(Row {
        name: name.into(),
        rows: "16".into(),
        accepted: format!(
            "{:?}",
            plain
                .recs
                .iter()
                .skip(1)
                .map(|r| r.accepted)
                .collect::<Vec<_>>()
        ),
        expect: "== None".into(),
        det,
        refcmp: "n/a".into(),
        ok,
    })
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_TEST_TARGET (27B dense) + HIPFIRE_TEST_DRAFT (paired DFlash2)"]
fn dflash_lookup_state_oracle() {
    let Some(mut rig) = Rig::open().expect("open rig") else {
        return;
    };
    let rf = reference_run(&mut rig).expect("reference run");
    let (scs, skipped) = scenarios(&rig, &rf);
    let mut rows: Vec<Row> = Vec::new();
    for sc in &scs {
        match run_scenario(&mut rig, sc, &rf) {
            Ok(r) => rows.push(r),
            Err(e) => rows.push(Row {
                name: sc.name.clone(),
                rows: "-".into(),
                accepted: "-".into(),
                expect: "-".into(),
                det: Err(format!("error: {e}")),
                refcmp: "-".into(),
                ok: false,
            }),
        }
    }
    for (name, samp) in [
        ("decline_eq_greedy", GREEDY),
        ("decline_eq_sampled", SAMPLED),
    ] {
        match run_decline_equivalence(&mut rig, samp, name) {
            Ok(r) => rows.push(r),
            Err(e) => rows.push(Row {
                name: name.into(),
                rows: "-".into(),
                accepted: "-".into(),
                expect: "-".into(),
                det: Err(format!("error: {e}")),
                refcmp: "-".into(),
                ok: false,
            }),
        }
    }

    println!(
        "{:<28} {:>5} {:>9} {:>7} {:>9}  {:<6}  {}",
        "scenario", "rows", "accepted", "expect", "A==B", "ref", "verdict"
    );
    for r in &rows {
        println!(
            "{:<28} {:>5} {:>9} {:>7} {:>9}  {:<6}  {}{}",
            r.name,
            r.rows,
            r.accepted,
            r.expect,
            if r.det.is_ok() {
                "identical"
            } else {
                "DIFFERS"
            },
            r.refcmp,
            if r.ok { "PASS" } else { "FAIL" },
            match &r.det {
                Err(d) => format!("  ({d})"),
                Ok(()) => String::new(),
            }
        );
    }
    for (name, why) in &skipped {
        println!("{name:<28} SKIPPED: {why}");
    }
    let failed: Vec<&str> = rows
        .iter()
        .filter(|r| !r.ok)
        .map(|r| r.name.as_str())
        .collect();
    println!(
        "DFLASH_LOOKUP_VERDICT {} ({} scenarios, {} skipped, failed: {failed:?})",
        if failed.is_empty() { "PASS" } else { "FAIL" },
        rows.len(),
        skipped.len()
    );
    // The forced coverage must exist: a skipped splice boundary is not a pass.
    assert!(
        skipped.is_empty(),
        "reference continuation did not provide every splice-boundary scenario: {skipped:?}"
    );
    assert!(
        failed.is_empty(),
        "lookup state oracle failures: {failed:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Sampled mixed-window law (hardware): chi-square of the emitted tokens
// ═══════════════════════════════════════════════════════════════════════════
//
// One prefill snapshot, one sampled window repeated `TRIALS` times from the
// restored snapshot with distinct RNG seeds. The target law is computed on the
// host from the window's own verify logits (a forced Copy window with the same
// rows) under the request's truncation, and compared with the emitted tokens:
//
// * Tail: K = 1 real learned row (`d1`, the first learned candidate that
//   maximizes the pilot's estimated reach P(d1 = mode) * P(d1 accepted), on
//   the best of several built-in prompts) + 3 point-mass rows (the target's
//   2nd-ranked token after the prefix, 1st-ranked if the 2nd carries < 2%).
//   The source splices only when the learned candidate equals `d1` (a
//   sequential rule reading `drafted[..1]` only); every other trial is an
//   incumbent window. Position 0 uses all trials (row 0 is the incumbent
//   procedure with or without the splice), positions >= 1 use spliced trials
//   whose emitted prefix reached them. Trials run until the first point-mass
//   row has >= 300 and the second >= 100 conditioned trials (cap 8000).
// * Copy: 3 point-mass rows chosen the same way; positions 0..3 conditional
//   on the emitted prefix.
//
// Position j's law: P(emitted_j = x | positions < j emitted as in the prefix)
// = p_j(x | prefix). Restore between trials covers what the window mutates:
// DeltaNet state (snapshot), hidden-ring head/written, thlog cursors (mark).
// KV rows >= pos and draft/target_hidden rows >= pos are rewritten by the next
// window before being read; every 100 trials the full committed state
// (`capture`) must equal the snapshot byte for byte.

const TRIALS: usize = 2000;
/// Incumbent windows per candidate prompt for the Tail reach estimate.
const SELECT_WINDOWS: usize = 60;
/// Tail trials: at least `TAIL_MIN_TRIALS`, then until the conditioned counts
/// of the first / second point-mass rows reach their targets, at most the cap.
const TAIL_MIN_TRIALS: usize = 2000;
const TAIL_CAP_TRIALS: usize = 8000;
const TAIL_N_FIRST: usize = 300;
const TAIL_N_SECOND: usize = 100;
/// A 2nd-ranked point-mass candidate with less target mass than this is
/// replaced by the 1st-ranked token (keeps p(candidate) testable).
const MIN_CANDIDATE_P: f64 = 0.02;
/// Candidate prompts of the Tail scenario (the story prompt also drives Copy).
const TAIL_PROMPTS: [(&str, &str); 4] = [
    (
        "story",
        "Write the opening paragraph of a story about a lighthouse keeper.",
    ),
    (
        "list",
        "List the planets of the solar system in order from the sun, one per line, each with one short fact.",
    ),
    (
        "code",
        "Write a Python function that reads a CSV file and returns a list of dictionaries, with a short docstring.",
    ),
    (
        "explain",
        "Explain in a few sentences how a refrigerator keeps food cold.",
    ),
];
/// Significance level of every chi-square test (declared, not tuned).
const ALPHA: f64 = 0.001;
/// Bins with expected count below this are pooled.
const MIN_BIN_EXPECTED: f64 = 5.0;
/// A (position, scenario) with fewer conditioned trials is reported, not tested.
const MIN_TRIALS: usize = 50;
/// Registry-like sampling: T = 1, top_p 0.95, top_k 20.
const LAW: Samp = Samp {
    temp: 1.0,
    top_p: 0.95,
    top_k: 20,
};

struct Snap {
    dn: DeltaNetSnapshot,
    ring_head: usize,
    ring_written: usize,
    th: TargetHiddenLogMark,
    pos: usize,
    seed: u32,
    parts: Vec<(&'static str, String)>,
}

fn take_snapshot(rig: &mut Rig) -> R<Snap> {
    let seed = rig.fresh_prefill()?;
    let pos = rig.prompt.len();
    rig.gpu.hip.device_synchronize()?;
    let mut dn = DeltaNetSnapshot::new_for(&mut rig.gpu, &rig.slot.dn_state)?;
    dn.save_from(&rig.slot.dn_state, &mut rig.gpu)?;
    let parts = capture(rig, None, pos, 0)?.parts;
    Ok(Snap {
        dn,
        ring_head: rig.df.hidden_rb.head,
        ring_written: rig.df.hidden_rb.written,
        th: rig.df.draft_scratch.thlog.mark(),
        pos,
        seed,
        parts,
    })
}

fn restore(rig: &mut Rig, s: &Snap) -> R<()> {
    rig.gpu.hip.device_synchronize()?;
    s.dn.restore_to(&mut rig.slot.dn_state, &mut rig.gpu)?;
    rig.df.hidden_rb.head = s.ring_head;
    rig.df.hidden_rb.written = s.ring_written;
    rig.df.draft_scratch.thlog.restore(s.th)?;
    Ok(())
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One window from the restored snapshot with RNG seed `seed`.
fn window_from(rig: &mut Rig, snap: &Snap, seed: u64, step: &mut Step) -> R<SpecStepResult> {
    restore(rig, snap)?;
    let mut cur = Cursor {
        pos: snap.pos,
        seed: snap.seed,
        rng: splitmix(seed) | 1,
        emitted: vec![snap.seed],
    };
    step_once(rig, &mut cur, LAW, step)
}

/// The request's truncated target law of one logits row (f64, independent of
/// the production kernels): temperature-1 softmax, the `top_k` most probable
/// tokens, then the nucleus over their own mass (cut at the first token whose
/// cumulative mass reaches `top_p`, inclusive), renormalized.
fn target_law(logits: &[f32], top_k: usize, top_p: f64) -> Vec<f64> {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut p: Vec<f64> = logits.iter().map(|&l| (l as f64 - mx).exp()).collect();
    let z: f64 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= z);
    let mut order: Vec<usize> = (0..p.len()).collect();
    order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(a.cmp(&b)));
    let cap = if top_k == 0 || top_k > 64 { 64 } else { top_k };
    order.truncate(cap);
    let kept: f64 = order.iter().map(|&i| p[i]).sum();
    let mut cum = 0.0;
    let mut cut = order.len();
    for (rank, &i) in order.iter().enumerate() {
        cum += p[i] / kept;
        if cum >= top_p {
            cut = rank + 1;
            break;
        }
    }
    order.truncate(cut);
    let mass: f64 = order.iter().map(|&i| p[i]).sum();
    let mut law = vec![0.0; p.len()];
    for &i in &order {
        law[i] = p[i] / mass;
    }
    law
}

fn ranked(law: &[f64], rank: usize) -> u32 {
    let mut order: Vec<usize> = (0..law.len()).collect();
    order.sort_by(|&a, &b| law[b].partial_cmp(&law[a]).unwrap().then(a.cmp(&b)));
    order[rank - 1] as u32
}

/// Laws of verify rows `0..=tokens.len()` of the window `[seed, tokens..]`,
/// read from its own verify logits (a forced Copy window).
fn pilot_laws(rig: &mut Rig, snap: &Snap, tokens: &[u32], seed: u64) -> R<Vec<Vec<f64>>> {
    window_from(rig, snap, seed, &mut Step::copy(tokens.to_vec(), None))?;
    rig.gpu.hip.device_synchronize()?;
    let rows = tokens.len() + 1;
    let vocab = rig.vocab;
    let all = rig.gpu.download_f32(&rig.df.verify_scratch.logits)?;
    if all.len() < rows * vocab {
        return Err(format!("verify logits hold {} < {} floats", all.len(), rows * vocab).into());
    }
    Ok((0..rows)
        .map(|r| {
            target_law(
                &all[r * vocab..(r + 1) * vocab],
                LAW.top_k,
                LAW.top_p as f64,
            )
        })
        .collect())
}

/// Grow `prefix` by the target's 2nd-ranked token at row `prefix.len()`
/// (1st-ranked when the 2nd carries less than `MIN_CANDIDATE_P`).
fn extend_with_second(
    rig: &mut Rig,
    snap: &Snap,
    prefix: &mut Vec<u32>,
    probe: u32,
    seed: u64,
) -> R<()> {
    let mut forced = prefix.clone();
    forced.push(probe);
    let laws = pilot_laws(rig, snap, &forced, seed)?;
    let law = &laws[prefix.len()];
    let second = ranked(law, 2);
    let pick = if law[second as usize] >= MIN_CANDIDATE_P {
        second
    } else {
        ranked(law, 1)
    };
    prefix.push(pick);
    Ok(())
}

fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        (-1_259.139_216_722_402_8),
        771.323_428_777_653_1,
        (-176.615_029_162_140_6),
        12.507_343_278_686_905,
        (-0.138_571_095_265_720_12),
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized upper incomplete gamma Q(a, x) (series / continued fraction).
fn gammq(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let lg = ln_gamma(a);
    if x < a + 1.0 {
        let (mut ap, mut sum) = (a, 1.0 / a);
        let mut del = sum;
        for _ in 0..10_000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        1.0 - sum * (-x + a * x.ln() - lg).exp()
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
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
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        (-x + a * x.ln() - lg).exp() * h
    }
}

struct Chi {
    n: usize,
    bins: usize,
    stat: f64,
    p_value: f64,
    violations: usize,
    table: Vec<(String, f64, f64)>,
}

fn chi_square(counts: &HashMap<u32, usize>, law: &[f64]) -> Chi {
    let n: usize = counts.values().sum();
    let nf = n as f64;
    let mut idx: Vec<usize> = (0..law.len()).filter(|&i| law[i] > 0.0).collect();
    idx.sort_by(|&a, &b| law[b].partial_cmp(&law[a]).unwrap().then(a.cmp(&b)));
    let mut bins: Vec<(String, f64, f64)> = Vec::new();
    let (mut listed_exp, mut listed_obs) = (0.0f64, 0usize);
    for &i in &idx {
        let e = nf * law[i];
        if e < MIN_BIN_EXPECTED {
            break;
        }
        let o = counts.get(&(i as u32)).copied().unwrap_or(0);
        bins.push((format!("{i}"), e, o as f64));
        listed_exp += e;
        listed_obs += o;
    }
    let violations: usize = counts
        .iter()
        .filter(|(&t, _)| law.get(t as usize).map_or(true, |&p| p <= 0.0))
        .map(|(_, &c)| c)
        .sum();
    let (rest_exp, rest_obs) = (nf - listed_exp, (n - listed_obs) as f64);
    if rest_exp >= MIN_BIN_EXPECTED {
        bins.push(("(pooled)".into(), rest_exp, rest_obs));
    } else if let Some(last) = bins.last_mut() {
        last.1 += rest_exp;
        last.2 += rest_obs;
    }
    let stat: f64 = bins.iter().map(|(_, e, o)| (o - e) * (o - e) / e).sum();
    let dof = bins.len().saturating_sub(1);
    let p_value = if dof >= 1 {
        gammq(dof as f64 / 2.0, stat / 2.0)
    } else {
        f64::NAN
    };
    Chi {
        n,
        bins: bins.len(),
        stat,
        p_value,
        violations,
        table: bins,
    }
}

/// Two-sided exact binomial p-value of `k` successes in `n` trials at rate
/// `p`: the summed probability of every outcome no more likely than `k`.
fn binomial_two_sided(n: usize, k: usize, p: f64) -> f64 {
    if p <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if p >= 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }
    let nf = n as f64;
    let ln_pmf = |i: usize| {
        let i_f = i as f64;
        ln_gamma(nf + 1.0) - ln_gamma(i_f + 1.0) - ln_gamma(nf - i_f + 1.0)
            + i_f * p.ln()
            + (nf - i_f) * (1.0 - p).ln()
    };
    let at_k = ln_pmf(k);
    let total: f64 = (0..=n)
        .map(ln_pmf)
        .filter(|&l| l <= at_k + 1e-9)
        .map(f64::exp)
        .sum();
    total.min(1.0)
}

struct Trial {
    accepted: usize,
    committed: Vec<u32>,
    rows: usize,
}

/// Chi-square of position `j` over the trials `keep` selects. Returns
/// (verdict ok, evaluated).
fn test_position(
    label: &str,
    j: usize,
    trials: &[Trial],
    law: &[f64],
    keep: impl Fn(&Trial) -> bool,
) -> (bool, bool) {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for t in trials.iter().filter(|t| t.accepted >= j && keep(t)) {
        *counts.entry(t.committed[j + 1]).or_default() += 1;
    }
    let chi = chi_square(&counts, law);
    if chi.n < MIN_TRIALS {
        println!("{label} pos {j}: n={} (< {MIN_TRIALS}): not tested", chi.n);
        return (true, false);
    }
    if chi.bins < 2 {
        // The truncated law leaves one chi-square bin. Test it exactly
        // instead: a single-token support means every draw must be that
        // token (support violations are the whole test); a wider support
        // whose tail pools into the one bin gets a two-sided exact binomial
        // test of the off-mode count against its law mass.
        let support = law.iter().filter(|&&p| p > 0.0).count();
        let (mode, p_mode) =
            law.iter().enumerate().fold(
                (0usize, 0.0f64),
                |m, (i, &p)| if p > m.1 { (i, p) } else { m },
            );
        let off = chi.n - counts.get(&(mode as u32)).copied().unwrap_or(0);
        let p_value = if support <= 1 {
            if off == 0 {
                1.0
            } else {
                0.0
            }
        } else {
            binomial_two_sided(chi.n, off, 1.0 - p_mode)
        };
        let ok = chi.violations == 0 && p_value >= ALPHA;
        println!(
            "{label} pos {j}: n={} one-bin law (support {support}, mode {mode} p={p_mode:.6}): off-mode {off}, exact p={p_value:.4} support_violations={} (alpha={ALPHA}) {}",
            chi.n,
            chi.violations,
            if ok { "PASS" } else { "FAIL" }
        );
        return (ok, true);
    }
    let ok = chi.violations == 0 && chi.p_value >= ALPHA;
    println!(
        "{label} pos {j}: n={} bins={} chi2={:.2} dof={} p={:.4} support_violations={} (alpha={ALPHA}) {}",
        chi.n,
        chi.bins,
        chi.stat,
        chi.bins - 1,
        chi.p_value,
        chi.violations,
        if ok { "PASS" } else { "FAIL" }
    );
    println!("    {:<12} {:>10} {:>10}", "token", "expected", "observed");
    for (tok, e, o) in chi.table.iter().take(12) {
        println!("    {tok:<12} {e:>10.1} {o:>10.0}");
    }
    if chi.table.len() > 12 {
        println!("    ... {} more bins", chi.table.len() - 12);
    }
    (ok, true)
}

fn user_prompt(slot: &ModelSlot, user: &str) -> R<Vec<u32>> {
    let tok = slot
        .load_tokenizer()
        .map_err(|e| format!("tokenizer: {e:?}"))?;
    Ok([
        "<|im_start|>",
        "user",
        "\n",
        user,
        "<|im_end|>",
        "\n",
        "<|im_start|>",
        "assistant",
        "\n",
        "<think>\n\n</think>\n\n",
    ]
    .iter()
    .flat_map(|s| tok.encode(s))
    .collect())
}

/// The trial emitted `prefix[..j]` at positions `0..j` (and reached position `j`).
fn prefix_ok(t: &Trial, prefix: &[u32], j: usize) -> bool {
    t.accepted >= j && t.committed[1..=j] == prefix[..j]
}

/// Run one chi-square position test; record coverage and rejections.
#[allow(clippy::too_many_arguments)]
fn check(
    evaluated: &mut Vec<String>,
    failures: &mut Vec<String>,
    label: &str,
    j: usize,
    trials: &[Trial],
    law: &[f64],
    keep: impl Fn(&Trial) -> bool,
) {
    let (ok, ran) = test_position(label, j, trials, law, keep);
    if ran {
        evaluated.push(format!("{label}@{j}"));
    }
    if !ok {
        failures.push(format!(
            "{label}@{j}: chi-square / support check rejected the target law"
        ));
    }
}

/// Restore + full-state byte check against the snapshot.
fn assert_restored(rig: &mut Rig, snap: &Snap, trials_done: usize) {
    restore(rig, snap).expect("restore");
    let parts = capture(rig, None, snap.pos, 0).expect("capture").parts;
    assert!(
        parts == snap.parts,
        "restored state differs from the snapshot after {trials_done} trials"
    );
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_TEST_TARGET (27B dense) + HIPFIRE_TEST_DRAFT (paired DFlash2); ~2000 Copy + 2000..8000 Tail sampled windows"]
fn dflash_lookup_sampled_law() {
    let Some(mut rig) = Rig::open().expect("open rig") else {
        return;
    };
    let t_all = std::time::Instant::now();
    let mut failures: Vec<String> = Vec::new();
    let mut evaluated: Vec<String> = Vec::new();

    // ── Copy window (story prompt): 3 point-mass rows, 2nd-ranked chain ──
    rig.prompt = user_prompt(&rig.slot, TAIL_PROMPTS[0].1).expect("prompt");
    let snap = take_snapshot(&mut rig).expect("snapshot");
    let mut copy_tokens: Vec<u32> = Vec::new();
    for i in 0..3 {
        extend_with_second(&mut rig, &snap, &mut copy_tokens, 0, 10 + i).expect("copy pilot");
    }
    let copy_laws = pilot_laws(&mut rig, &snap, &copy_tokens, 20).expect("copy laws");
    println!(
        "[sampled-law] copy tokens {copy_tokens:?}, p(candidate) per row {:?}",
        copy_tokens
            .iter()
            .enumerate()
            .map(|(r, &c)| copy_laws[r][c as usize])
            .collect::<Vec<_>>()
    );
    let mut trials: Vec<Trial> = Vec::new();
    for i in 0..TRIALS {
        let r = window_from(
            &mut rig,
            &snap,
            1_000 + i as u64,
            &mut Step::copy(copy_tokens.clone(), None),
        )
        .expect("copy window");
        trials.push(Trial {
            accepted: r.accepted,
            committed: r.committed,
            rows: r.drafted.len(),
        });
        if (i + 1) % 100 == 0 {
            assert_restored(&mut rig, &snap, i + 1);
        }
    }
    assert!(
        trials.iter().all(|t| t.rows == 4),
        "every Copy trial must verify exactly 4 rows"
    );
    for j in 0..=3 {
        check(
            &mut evaluated,
            &mut failures,
            "copy",
            j,
            &trials,
            &copy_laws[j],
            |t| prefix_ok(t, &copy_tokens, j),
        );
    }
    snap.dn.free_gpu(&mut rig.gpu);

    // ── Tail: pick the prompt / first learned candidate by estimated reach ─
    // reach = P(learned d1 = d1*) * P(d1* accepted), from SELECT_WINDOWS
    // incumbent windows per prompt (d1* = the d1 with most accepted windows).
    let mut best: Option<(usize, u32, usize, usize)> = None; // (prompt, d1, accepted, count)
    for (ci, (name, text)) in TAIL_PROMPTS.iter().enumerate() {
        rig.prompt = user_prompt(&rig.slot, text).expect("prompt");
        let s = take_snapshot(&mut rig).expect("snapshot");
        let mut freq: HashMap<u32, (usize, usize)> = HashMap::new(); // d1 -> (windows, accepted>=1)
        for i in 0..SELECT_WINDOWS {
            let r = window_from(
                &mut rig,
                &s,
                200_000 + (ci * 1_000 + i) as u64,
                &mut Step::none(),
            )
            .expect("selection window");
            let e = freq.entry(r.drafted[1]).or_default();
            e.0 += 1;
            if r.accepted >= 1 {
                e.1 += 1;
            }
        }
        s.dn.free_gpu(&mut rig.gpu);
        let (&d1, &(count, acc)) = freq
            .iter()
            .max_by(|a, b| {
                a.1 .1
                    .cmp(&b.1 .1)
                    .then(a.1 .0.cmp(&b.1 .0))
                    .then(b.0.cmp(a.0))
            })
            .expect("selection windows");
        println!(
            "[sampled-law] prompt {name}: d1*={d1} learned in {count}/{SELECT_WINDOWS} windows, accepted in {acc} -> est. reach {:.3}",
            acc as f64 / SELECT_WINDOWS as f64
        );
        if best.map_or(true, |(_, _, a, c)| (acc, count) > (a, c)) {
            best = Some((ci, d1, acc, count));
        }
    }
    let (pi, d1, acc, count) = best.expect("a candidate prompt");
    println!(
        "[sampled-law] tail prompt chosen: {} (d1*={d1}, est. reach {:.3}, d1* in {count}/{SELECT_WINDOWS}, accepted {acc})",
        TAIL_PROMPTS[pi].0,
        acc as f64 / SELECT_WINDOWS as f64
    );
    rig.prompt = user_prompt(&rig.slot, TAIL_PROMPTS[pi].1).expect("prompt");
    let snap = take_snapshot(&mut rig).expect("snapshot");

    let mut prefix: Vec<u32> = vec![d1];
    for i in 0..3 {
        extend_with_second(&mut rig, &snap, &mut prefix, 0, 60 + i).expect("tail pilot");
    }
    let tail_tokens: Vec<u32> = prefix[1..].to_vec();
    let tail_laws = pilot_laws(&mut rig, &snap, &prefix, 70).expect("tail laws");
    println!(
        "[sampled-law] tail d1={d1} p(d1)={:.4}; point-mass tokens {tail_tokens:?}, p(candidate) rows 1..4 {:?}",
        tail_laws[0][d1 as usize],
        tail_tokens
            .iter()
            .enumerate()
            .map(|(r, &c)| tail_laws[1 + r][c as usize])
            .collect::<Vec<_>>()
    );
    let mut trials: Vec<Trial> = Vec::new();
    let (mut n_first, mut n_second) = (0usize, 0usize);
    let mut i = 0usize;
    while i < TAIL_CAP_TRIALS
        && !(i >= TAIL_MIN_TRIALS && n_first >= TAIL_N_FIRST && n_second >= TAIL_N_SECOND)
    {
        let mut step = Step {
            look: Look::Gated(GatedTail {
                want: d1,
                tail: tail_tokens.clone(),
            }),
            max_accept: None,
        };
        let r = window_from(&mut rig, &snap, 100_000 + i as u64, &mut step).expect("tail window");
        let t = Trial {
            accepted: r.accepted,
            committed: r.committed,
            rows: r.drafted.len(),
        };
        if t.rows == 5 {
            n_first += prefix_ok(&t, &prefix, 1) as usize;
            n_second += prefix_ok(&t, &prefix, 2) as usize;
        }
        trials.push(t);
        i += 1;
        if i % 100 == 0 {
            assert_restored(&mut rig, &snap, i);
        }
    }
    let spliced = trials.iter().filter(|t| t.rows == 5).count();
    println!(
        "[sampled-law] tail: {i} trials, {spliced} spliced; conditioned trials first point-mass row {n_first} (target {TAIL_N_FIRST}), second {n_second} (target {TAIL_N_SECOND})"
    );
    check(
        &mut evaluated,
        &mut failures,
        "tail",
        0,
        &trials,
        &tail_laws[0],
        |_| true,
    );
    for j in 1..=4 {
        check(
            &mut evaluated,
            &mut failures,
            "tail",
            j,
            &trials,
            &tail_laws[j],
            |t| t.rows == 5 && prefix_ok(t, &prefix, j),
        );
    }
    snap.dn.free_gpu(&mut rig.gpu);

    // Coverage is decided BEFORE the verdict line is printed.
    for must in ["copy@0", "copy@1", "tail@0", "tail@1", "tail@2"] {
        if !evaluated.iter().any(|e| e == must) {
            failures.push(format!(
                "{must} was not tested (too few conditioned trials or < 2 bins); first/second point-mass conditioned counts {n_first}/{n_second} of {i} tail trials"
            ));
        }
    }
    println!(
        "DFLASH_LOOKUP_SAMPLED_LAW {} (alpha={ALPHA}, copy trials={TRIALS}, tail trials={i}, tested: {evaluated:?}, {:.0}s){}",
        if failures.is_empty() { "PASS" } else { "FAIL" },
        t_all.elapsed().as_secs_f64(),
        if failures.is_empty() {
            String::new()
        } else {
            format!(" reasons: {failures:?}")
        }
    );
    assert!(failures.is_empty(), "sampled-law failures: {failures:?}");
}

#[test]
fn lookup_law_exact_binomial_helper() {
    // Symmetric: every outcome is no more likely than the mode at n/2.
    assert!((binomial_two_sided(10, 5, 0.5) - 1.0).abs() < 1e-9);
    // k = 0 at p = 1/2: P(0) + P(10) = 2 / 1024.
    assert!((binomial_two_sided(10, 0, 0.5) - 2.0 / 1024.0).abs() < 1e-12);
    // Degenerate rates.
    assert_eq!(binomial_two_sided(50, 0, 0.0), 1.0);
    assert_eq!(binomial_two_sided(50, 1, 0.0), 0.0);
    // 100 draws at 1%: k = 0 excludes only the more likely k = 1.
    let p1 = 100.0 * 0.01 * 0.99f64.powi(99);
    assert!((binomial_two_sided(100, 0, 0.01) - (1.0 - p1)).abs() < 1e-9);
}

// ── window-boundary carry-over diagnostic ───────────────────────────────

/// Greedy windows from a fresh prefill. `at` = (window index, step) replaces
/// that one window; every other window is the incumbent. Stops once
/// `min_tokens` are emitted or EOS. Returns the cursor.
fn boundary_arm(rig: &mut Rig, at: Option<(usize, Step)>, min_tokens: usize) -> R<Cursor> {
    let first = rig.fresh_prefill()?;
    let mut cur = Cursor {
        pos: rig.prompt.len(),
        seed: first,
        rng: 0x1D2C_3B4A_5968_7786 | 1,
        emitted: vec![first],
    };
    let mut at = at;
    let mut w = 0usize;
    while cur.emitted.len() < min_tokens && !cur.emitted.contains(&rig.eos) && w < 64 {
        let mut step = match at.take() {
            Some((i, s)) if i == w => s,
            other => {
                at = other;
                Step::none()
            }
        };
        step_once(rig, &mut cur, GREEDY, &mut step)?;
        w += 1;
    }
    Ok(cur)
}

/// Diagnostic for cross-window state carry-over, no product claim by itself:
/// A = incumbent windows; B = the same, but window `w` clamped (`max_accept`,
/// the incumbent max_emit path) to commit `m - 2` instead of `m`; C = window
/// `w` is a Copy window whose candidate `m - 2` is wrong, so it commits the
/// same tokens at the same boundary as B. Greedy streams must agree. Reports
/// whether the committed KV [0, P) of A and B agree (window-boundary
/// dependence of the incumbent dense verifier), and ASSERTS that B and C agree
/// on every captured state part (a lookup window leaves exactly the state an
/// incumbent window with the same boundary leaves).
#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_TEST_TARGET (27B dense) + HIPFIRE_TEST_DRAFT (paired DFlash2)"]
fn dflash_window_boundary_carryover() {
    let Some(mut rig) = Rig::open().expect("open rig") else {
        return;
    };
    let rf = reference_run(&mut rig).expect("reference run");
    let Some(w) = rf
        .wins
        .iter()
        .position(|r| r.accepted >= 4 && rf.all.len() >= r.seed_idx + 1 + 15)
    else {
        println!("WINDOW_BOUNDARY_CARRYOVER SKIPPED: no reference window with accepted >= 4 and 15 continuation tokens");
        panic!("no usable reference window");
    };
    let m = rf.wins[w].accepted;
    let cont = rf.cont(w).to_vec();
    let min_tokens = rf.all.len().min(100);

    let a = boundary_arm(&mut rig, None, min_tokens).expect("arm A");
    let pa = a.pos;

    let mut clamp = Step::none();
    clamp.max_accept = Some(m - 2);
    let b = boundary_arm(&mut rig, Some((w, clamp)), min_tokens).expect("arm B");
    let pb = b.pos;
    let b_at_pb = capture(&rig, None, pb, 0).expect("capture B");

    let mut copy_tokens = cont[..15.min(cont.len())].to_vec();
    copy_tokens[m - 2] = wrong(copy_tokens[m - 2], rig.vocab);
    let c = boundary_arm(
        &mut rig,
        Some((w, Step::copy(copy_tokens, None))),
        min_tokens,
    )
    .expect("arm C");
    let pc = c.pos;
    let c_at_pc = capture(&rig, None, pc, 0).expect("capture C");

    let n = a.emitted.len().min(b.emitted.len()).min(c.emitted.len());
    let streams_equal = a.emitted[..n] == b.emitted[..n] && b.emitted[..n] == c.emitted[..n];
    // KV [0, P) of A vs B at the common committed prefix: recapture A by
    // re-running it (state is only readable at the arm's end).
    let p = pa.min(pb);
    let a2 = boundary_arm(&mut rig, None, min_tokens).expect("arm A again");
    assert_eq!(a2.emitted, a.emitted, "arm A is not deterministic");
    let a_at_p = capture(&rig, None, p, 0).expect("capture A at P");
    let b2 = boundary_arm(
        &mut rig,
        Some((w, {
            let mut s = Step::none();
            s.max_accept = Some(m - 2);
            s
        })),
        min_tokens,
    )
    .expect("arm B again");
    assert_eq!(b2.emitted, b.emitted, "arm B is not deterministic");
    let b_at_p = capture(&rig, None, p, 0).expect("capture B at P");
    let kv = |r: &Rec| {
        r.parts
            .iter()
            .find(|(k, _)| *k == "kv")
            .map(|(_, v)| v.clone())
    };
    let kv_a_b_equal = kv(&a_at_p) == kv(&b_at_p);

    let b_vs_c = pb == pc && b.emitted == c.emitted && b_at_pb.parts == c_at_pc.parts;
    println!(
        "WINDOW_BOUNDARY_CARRYOVER window={w} m={m} streams_equal={streams_equal} \
         kv_incumbent_boundary_shift_equal={kv_a_b_equal} (P={p}) lookup_vs_clamped_incumbent_equal={b_vs_c}"
    );
    if !b_vs_c {
        for ((k, x), (_, y)) in b_at_pb.parts.iter().zip(&c_at_pc.parts) {
            println!(
                "    {k}: B={x} C={y} {}",
                if x == y { "same" } else { "DIFF" }
            );
        }
    }
    assert!(streams_equal, "greedy streams differ across arms");
    assert!(
        b_vs_c,
        "Copy window left different state than the clamped incumbent window with the same boundary"
    );
}
