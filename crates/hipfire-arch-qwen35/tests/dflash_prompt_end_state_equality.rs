// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! DFlash prompt-end payload state-equality hardware proof (#1198, 27B dense +
//! paired DFlash2 draft).
//!
//! ONE `#[ignore]`d test driving the real [`DflashSpeculator`] in-process on a
//! real HIP GPU. A cold exact retry (the same rendered prompt re-sent after the
//! previous turn committed) must reuse its WHOLE prompt: zero prefill rows, with
//! the last-row logits, the speculator seed (draft context + cursors) and the
//! first token byte-identical to a fresh prefill.
//!
//! Per arm (greedy and fixed-seed sampled; a short prompt whose rows decode
//! overwrites after the draft window wraps, and a prompt longer than the window):
//!
//! 1. Fresh COLD prefill (`cache_hit = false`). The speculator must advertise
//!    exactly `(len, prefix_fingerprint(prompt))` as its prompt-end checkpoint.
//! 2. Digest the FRESH state: raw last-row logits, DeltaNet bytes and stream
//!    metadata, target-KV prefix `[0, len)`, the draft's `target_hidden` window
//!    rows, the `target_hidden` cursors (uploaded rows, absolute positions,
//!    projection watermarks) and the request RNG after the first-token draw.
//! 3. Decode (greedy: argmax windows; sampled: seeded rejection sampling) far
//!    enough that, on the windowed draft, the `target_hidden` ring wraps over the
//!    prompt's own rows. The digest taken then is asserted to DIFFER from the
//!    fresh one (the wrap is real, so equality after the retry is not vacuous).
//! 4. Re-configure the request (fresh RNG) and EXACT-RETRY: `prefill` with
//!    `prefill_tokens = []`, `prefill_start = prompt.len()`, `cache_hit = true`
//!    (the planner's contract). Digest again: byte-equal to the fresh digest,
//!    same first token, same RNG state.
//! 5. Decode the same number of tokens again: the committed stream, the final
//!    position and the end-of-decode digest (KV `[0, pos)`, `target_hidden`
//!    window, cursors, DeltaNet, RNG) must equal the first decode's.
//! 6. CONTROL: reset, a second independent fresh cold prefill must digest equal
//!    to the first (the restore equals a fresh prefill, not merely the first
//!    capture).
//!
//! The raw per-layer `HiddenStateRingBuffer` is deliberately NOT part of the
//! digest: it is write-before-read staging (a verify window advances it by the
//! whole block while only `accepted + 1` rows are scattered into
//! `target_hidden`), so after decode it is no image of committed rows and its
//! cursor depends on request history even for two fresh prefills.
//!
//! Negative controls: a same-length prompt with one token changed, and a shorter
//! prompt, both fail closed (and leave the payload usable); a cold prefill of a
//! different prompt moves the descriptor and an exact retry of the old prompt
//! then fails closed; `reset` drops the payload (no descriptor, retry fails
//! closed); `rewind_to` below the prompt end drops it.
//!
//! The windowed payload bytes are asserted against `4 x min(len, W) x E x D`
//! from the loaded dims; capture / restore wall time and the cold-prefill vs
//! exact-retry wall time are printed (`[dflash-prompt-end] ...`) for the cost
//! gate (copy cost vs the prefill rows saved).
//!
//! ```bash
//! HIPFIRE_DEVICES=<uuid> HIPFIRE_TEST_TARGET=<27b.mq4> HIPFIRE_TEST_DRAFT=<dflash2.hfq> \
//!   cargo test --release -p hipfire-arch-qwen35 --features deltanet \
//!   --test dflash_prompt_end_state_equality -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Without both variables the test reports a skip and returns before any GPU
//! initialisation (same convention as `dflash_lookup_state_oracle`); with them
//! set a GPU initialisation or load failure fails the test. It loads its own
//! 27B pair and does not free it: run it in its own process.

#![allow(clippy::all)]

use hipfire_arch_qwen35::dflash_spec::{load_dflash_state, DflashSpeculator};
use hipfire_arch_qwen35::qwen35::LayerType;
use hipfire_arch_qwen35::speculative::{KvMode, ModelSlot, ModelSlotConfig};
use hipfire_runtime::checkpoint_pool::prefix_fingerprint;
use hipfire_runtime::dflash::DraftCtxMode;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::spec::{SpecRequestConfig, SpecTarget, Speculator};
use rdna_compute::{Gpu, GpuTensor};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

type R<T> = Result<T, Box<dyn std::error::Error>>;

const TARGET_ENV: &str = "HIPFIRE_TEST_TARGET";
const DRAFT_ENV: &str = "HIPFIRE_TEST_DRAFT";
/// Target KV / draft context capacity.
const CTX: usize = 4096;
/// Tokens decoded in the warm-up arm (verify graphs and kernels warm once).
const WARMUP_DECODE: usize = 48;
/// Extra decoded tokens past the draft window so decode overwrites the prompt's
/// oldest `target_hidden` rows several times over.
const WRAP_MARGIN: usize = 320;
/// Tokens decoded after the long (longer than the window) prompt.
const LONG_DECODE: usize = 384;
const PASSAGE: &str = "The harbor bell rang twice before dawn, and the gulls answered in kind. \
Lanterns swayed along the pier while the tide turned slowly toward the old stones. \
A fisherman coiled his rope, counted the nets twice, and walked home without a word.";
const TASK: &str = "Write a long, detailed story about a lighthouse keeper and keep \
going without stopping.\n\nHarbor notes:\n";

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
    spec: DflashSpeculator,
    tok: hipfire_runtime::tokenizer::Tokenizer,
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
        let slot = ModelSlot::load(&mut gpu, &target, "prompt-end-oracle", cfg)
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
            &slot.weights,
            kv_q8,
            slot.kv_cache.uses_vmm_backend(),
            slot.scratch.flash_partials.numel(),
            true,
            false,
            16,
        )?;
        let tok = slot
            .load_tokenizer()
            .map_err(|e| format!("tokenizer: {e:?}"))?;
        // The daemon's checkpoint cadence defaults (interval 2048, cap 8); fixed
        // block, no floor call log.
        let spec = DflashSpeculator::new(df, 2048, 8, false, false);
        Ok(Some(Rig {
            gpu,
            slot,
            spec,
            tok,
        }))
    }

    /// Window rows of the draft's `target_hidden` ring (`None`: Legacy,
    /// absolute rows).
    fn window(&self) -> Option<usize> {
        match self.spec.lane_state()?.draft_scratch.ctx_mode {
            DraftCtxMode::Windowed { w, .. } => Some(w),
            DraftCtxMode::Legacy => None,
        }
    }

    /// A clean slate for a cold prefill: drafter reset (drops the payload), the
    /// target's recurrent state and KV offset.
    fn cold_reset(&mut self) -> R<()> {
        self.spec.reset(&mut self.gpu)?;
        self.slot.reset_recurrent(&mut self.gpu)?;
        self.slot.kv_cache.compact_offset = 0;
        // With GDN error feedback off, Q8 DeltaNet requant rounds off a
        // process-global frame counter: pin it so two fresh runs are bytewise
        // comparable.
        if self.slot.dn_state.s_ef_residual.is_empty() {
            rdna_compute::norm::restore_gdn_requant_frame_checkpoint(0);
        }
        Ok(())
    }

    /// The ChatML turn the oracle prompts use, with `body` as the user context.
    fn prompt(&self, body: &str) -> Vec<u32> {
        [
            "<|im_start|>",
            "user",
            "\n",
            TASK,
            body,
            "<|im_end|>",
            "\n",
            "<|im_start|>",
            "assistant",
            "\n",
            "<think>\n\n</think>\n\n",
        ]
        .iter()
        .flat_map(|s| self.tok.encode(s))
        .collect()
    }

    /// The smallest repetition of the passage whose prompt has at least `min_len`
    /// tokens (an irregular, non chunk-aligned natural end).
    fn long_prompt(&self, min_len: usize) -> Vec<u32> {
        let mut body = String::new();
        for i in 0.. {
            body.push_str(&format!("[{i}] {PASSAGE}\n"));
            let p = self.prompt(&body);
            if p.len() >= min_len {
                return p;
            }
        }
        unreachable!()
    }

    fn prefill_cold(&mut self, prompt: &[u32], cfg: SpecRequestConfig) -> R<u32> {
        self.spec.configure_request(cfg);
        let never = || false;
        Ok(self
            .spec
            .prefill(
                &mut self.gpu,
                &mut self.slot,
                prompt,
                prompt,
                0,
                false,
                None,
                &never,
            )?
            .into_first_token()?)
    }

    /// The planner's exact edge: nothing to prefill, whole prompt cached.
    fn prefill_exact(
        &mut self,
        prompt: &[u32],
        cfg: SpecRequestConfig,
    ) -> Result<hipfire_runtime::spec::PrefillOutcome, String> {
        self.spec.configure_request(cfg);
        let never = || false;
        self.spec.prefill(
            &mut self.gpu,
            &mut self.slot,
            prompt,
            &[],
            prompt.len(),
            true,
            None,
            &never,
        )
    }

    /// `n` tokens after `first` (greedy / sampled windows through the real
    /// `Speculator::step`); returns every committed token (first included) and
    /// the final position.
    fn decode(&mut self, len: usize, first: u32, n: usize, temp: f32) -> R<(Vec<u32>, usize)> {
        let mut emitted = vec![first];
        let mut position = len;
        let mut seed = first;
        while emitted.len() <= n {
            let max_emit = n + 1 - emitted.len();
            let step = self.spec.step(
                &mut self.gpu,
                &mut self.slot,
                position,
                seed,
                &emitted,
                None,
                temp,
                max_emit,
            )?;
            position += step.emit.len();
            seed = step.next_seed;
            emitted.extend_from_slice(&step.emit);
        }
        Ok((emitted, position))
    }
}

// ── state digest ─────────────────────────────────────────────────────────

/// Everything the restore must reproduce, as named digests so a failure names
/// the diverging surface.
struct State {
    parts: Vec<(&'static str, String)>,
}

fn capture_state(rig: &Rig, pos: usize) -> R<State> {
    rig.gpu.hip.device_synchronize()?;
    let gpu = &rig.gpu;
    let slot = &rig.slot;
    let df = rig.spec.lane_state().ok_or("DFlash lane state is vacant")?;
    let mut parts: Vec<(&'static str, String)> = Vec::new();

    // Last-row logits (raw bytes of the target scratch row).
    {
        let lg = &slot.scratch.logits;
        parts.push((
            "logits",
            hex(Sha256::digest(read_dev(gpu, lg, 0, lg.numel() * 4)?)),
        ));
    }
    // DeltaNet state, scales, error feedback and conv state (logical extents).
    {
        let dn = &slot.dn_state;
        let mut h = Sha256::new();
        for t in dn
            .s_matrices
            .iter()
            .chain(&dn.s_scales)
            .chain(&dn.s_ef_residual)
            .chain(&dn.conv_states)
        {
            let n = t.byte_size().min(t.buf.size());
            h.update(read_dev(gpu, t, 0, n)?);
        }
        parts.push(("deltanet", hex(h.finalize())));
    }
    // Stream identity cursors (owner generation is the live owner's, not state).
    {
        let mut meta = slot.dn_state.stream.image().meta;
        for m in meta.iter_mut() {
            m.owner_generation = 0;
        }
        parts.push(("dn_stream", format!("{meta:?}")));
    }
    // Target KV rows [0, pos) of every attention layer.
    {
        let kv = &slot.kv_cache;
        let (kr, vr) = kv_row_bytes(kv)?;
        let mut h = Sha256::new();
        for (layer, ty) in slot.config.layer_types.iter().enumerate() {
            if *ty == LayerType::FullAttention {
                h.update(read_dev(gpu, &kv.k_gpu[layer], 0, pos * kr)?);
                h.update(read_dev(gpu, &kv.v_gpu[layer], 0, pos * vr)?);
            }
        }
        parts.push(("kv", hex(h.finalize())));
    }
    // The draft's interleaved `target_hidden` rows of every live position:
    // the whole ring when it has wrapped, else rows [0, pos).
    let ds = &df.draft_scratch;
    {
        let row = df.draft_config.feature_width() * 4;
        let rows = match ds.ctx_mode {
            DraftCtxMode::Windowed { w, .. } => pos.min(w),
            DraftCtxMode::Legacy => pos,
        };
        parts.push((
            "target_hidden",
            hex(Sha256::digest(read_dev(
                gpu,
                &ds.target_hidden,
                0,
                rows * row,
            )?)),
        ));
    }
    // `target_hidden` cursors: uploaded rows, absolute positions, watermarks.
    {
        let th = &ds.thlog;
        let abs: Vec<u8> = th
            .abs_positions()
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect();
        parts.push((
            "thlog",
            format!(
                "uploaded={} abs_len={} abs_sha={} proj={} full={}",
                th.uploaded_rows(),
                th.abs_positions().len(),
                hex(Sha256::digest(&abs)),
                th.proj_cached_rows(),
                th.full_cached_rows()
            ),
        ));
    }
    parts.push(("rng", format!("{:#018x}", rig.spec.request_rng_state())));
    Ok(State { parts })
}

fn report(tag: &str, s: &State) {
    let line: Vec<String> = s
        .parts
        .iter()
        .map(|(n, d)| format!("{n}={}", &d[..d.len().min(16)]))
        .collect();
    eprintln!("[dflash-prompt-end] {tag}: {}", line.join(" "));
}

/// Field-by-field comparison so a failure names every diverging surface.
fn assert_state_equal(tag: &str, want: &State, got: &State) {
    assert_eq!(want.parts.len(), got.parts.len(), "{tag}: part count");
    let bad: Vec<String> = want
        .parts
        .iter()
        .zip(&got.parts)
        .filter(|(w, g)| w != g)
        .map(|(w, g)| format!("{}: want {} got {}", w.0, w.1, g.1))
        .collect();
    assert!(
        bad.is_empty(),
        "{tag}: state diverged:\n  {}",
        bad.join("\n  ")
    );
}

fn part<'a>(s: &'a State, name: &str) -> &'a str {
    &s.parts.iter().find(|(n, _)| *n == name).expect("part").1
}

// ── arms ─────────────────────────────────────────────────────────────────

fn greedy() -> SpecRequestConfig {
    SpecRequestConfig::default()
}

fn sampled(seed: u64) -> SpecRequestConfig {
    SpecRequestConfig {
        temp: 0.7,
        top_p: 0.95,
        top_k: Some(20),
        rng_seed: seed,
        ..SpecRequestConfig::default()
    }
}

/// One full arm (steps 1-6 of the module doc) on `prompt`.
fn run_arm(rig: &mut Rig, name: &str, prompt: &[u32], cfg: SpecRequestConfig, n: usize) -> R<()> {
    let len = prompt.len();
    let window = rig.window();
    let want_desc = Some((len, prefix_fingerprint(prompt)));

    // 1. fresh cold prefill.
    rig.cold_reset()?;
    let t = Instant::now();
    let first = rig.prefill_cold(prompt, cfg)?;
    let cold_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(
        rig.spec.prompt_end_checkpoint(),
        want_desc,
        "{name}: a cold prefill must advertise (len, prefix_fingerprint(prompt)) \
         (payload not retained? the admitted-context memory gate or an allocation refused it)"
    );
    let stats = rig
        .spec
        .prompt_end_stats()
        .ok_or("no prompt-end stats after capture")?;
    let df = rig.spec.lane_state().ok_or("vacant")?;
    let expect_feature_bytes = window.map_or(0, |w| {
        (4 * len.min(w)
            * df.draft_config.num_extract()
            * df.draft_config.hidden
            * df.draft_config.target_streams_per_layer) as u64
    });
    assert_eq!(
        stats.feature_bytes, expect_feature_bytes,
        "{name}: saved feature window bytes (4 x min(len, W) x E x D)"
    );
    // 2. digest the fresh state.
    let fresh = capture_state(rig, len)?;
    report(&format!("{name} fresh"), &fresh);

    // 3. decode far enough to wrap the draft window over the prompt's own rows.
    let (toks_a, pos_a) = rig.decode(len, first, n, cfg.temp)?;
    let after_a = capture_state(rig, pos_a)?;
    let dirty = capture_state(rig, len)?;
    assert_ne!(
        part(&fresh, "deltanet"),
        part(&dirty, "deltanet"),
        "{name}: decode did not move the DeltaNet state"
    );
    assert_eq!(
        part(&fresh, "kv"),
        part(&dirty, "kv"),
        "{name}: the target KV prefix [0, len) must survive decode (append-only)"
    );
    if window.is_some() {
        assert_ne!(
            part(&fresh, "target_hidden"),
            part(&dirty, "target_hidden"),
            "{name}: decode did not overwrite the prompt's target_hidden window rows \
             (the wrap this oracle exists to cover)"
        );
    }
    assert_eq!(
        rig.spec.prompt_end_checkpoint(),
        want_desc,
        "{name}: decode at or above the prompt end must keep the payload"
    );

    // 4. exact retry, then compare against the fresh digest.
    let t = Instant::now();
    let first2 = rig
        .prefill_exact(prompt, cfg)
        .map_err(|e| format!("{name}: exact retry: {e}"))?
        .into_first_token()?;
    let retry_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(first2, first, "{name}: first token");
    let retry = capture_state(rig, len)?;
    report(&format!("{name} retry"), &retry);
    assert_state_equal(
        &format!("{name}: exact retry vs fresh prefill"),
        &fresh,
        &retry,
    );
    assert_eq!(
        rig.spec.prompt_end_checkpoint(),
        want_desc,
        "{name}: the payload stays valid after its own restore"
    );
    let stats = rig
        .spec
        .prompt_end_stats()
        .ok_or("no stats after restore")?;
    eprintln!(
        "[dflash-prompt-end] {name}: window={window:?} len={len} dn_snapshot={} B features={} B \
         logits={} B | capture {} us, restore {} us | cold prefill {cold_ms:.1} ms vs exact retry \
         {retry_ms:.1} ms (both include the first-token fold)",
        stats.dn_snapshot_bytes,
        stats.feature_bytes,
        stats.logits_bytes,
        stats.capture_us,
        stats.restore_us,
    );

    // 5. the continuation is the same stream, ending in the same state.
    let (toks_b, pos_b) = rig.decode(len, first2, n, cfg.temp)?;
    if toks_b != toks_a || pos_b != pos_a {
        // Attribute it: is a second FRESH decode of this arm equal to the first?
        rig.cold_reset()?;
        let f = rig.prefill_cold(prompt, cfg)?;
        let (toks_c, _) = rig.decode(len, f, n, cfg.temp)?;
        let first_diff = toks_a.iter().zip(&toks_b).position(|(a, b)| a != b);
        panic!(
            "{name}: continuation after the exact retry diverged (first differing index {first_diff:?}, \
             lens {} vs {}); a second fresh decode {} the first",
            toks_a.len(),
            toks_b.len(),
            if toks_c == toks_a {
                "equals"
            } else {
                "ALSO differs from (harness nondeterminism)"
            }
        );
    }
    let after_b = capture_state(rig, pos_b)?;
    assert_state_equal(
        &format!("{name}: end of continuation vs end of first decode"),
        &after_a,
        &after_b,
    );

    // 6. control: an independent fresh cold prefill digests equal to the first.
    rig.cold_reset()?;
    let first3 = rig.prefill_cold(prompt, cfg)?;
    assert_eq!(first3, first, "{name}: second fresh first token");
    let fresh2 = capture_state(rig, len)?;
    assert_state_equal(
        &format!("{name}: second fresh prefill vs first"),
        &fresh,
        &fresh2,
    );
    assert_eq!(
        rig.spec.prompt_end_checkpoint(),
        want_desc,
        "{name}: recaptured"
    );

    // Negative controls on the live payload.
    let mut other = prompt.to_vec();
    *other.last_mut().unwrap() ^= 1;
    let err = rig
        .prefill_exact(&other, cfg)
        .expect_err(&format!("{name}: a changed token must fail closed"));
    assert!(err.contains("fingerprint"), "{name}: {err}");
    let short = &prompt[..len - 1];
    let err = rig
        .prefill_exact(short, cfg)
        .expect_err(&format!("{name}: a shorter prompt must fail closed"));
    assert!(err.contains("fingerprint"), "{name}: {err}");
    assert_eq!(
        rig.spec.prompt_end_checkpoint(),
        want_desc,
        "{name}: a refused exact retry leaves the payload usable"
    );
    eprintln!("[dflash-prompt-end] {name}: OK");
    Ok(())
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_TEST_TARGET (27b dense) + HIPFIRE_TEST_DRAFT (dflash2 draft)"]
fn dflash_prompt_end_exact_retry_state_equality() -> R<()> {
    let Some(mut rig) = Rig::open()? else {
        return Ok(());
    };
    let window = rig.window();
    eprintln!(
        "[dflash-prompt-end] gpu={} draft ctx mode: {}, supports_temp_verify={}",
        rig.gpu.arch,
        window.map_or("Legacy".to_string(), |w| format!("Windowed W={w}")),
        rig.spec.supports_temp_verify(),
    );

    let short = rig.prompt(PASSAGE);
    let long = rig.long_prompt(window.unwrap_or(1500) + 573);
    eprintln!(
        "[dflash-prompt-end] prompts: short={} long={} tokens",
        short.len(),
        long.len()
    );

    // Warm-up (verify graphs / kernels); its payload is superseded below.
    rig.cold_reset()?;
    let first = rig.prefill_cold(&short, greedy())?;
    rig.decode(short.len(), first, WARMUP_DECODE, 0.0)?;

    let wrap_n = window.map_or(256, |w| w + WRAP_MARGIN);
    let arms: Vec<(&str, &Vec<u32>, SpecRequestConfig, usize)> = {
        let mut a = vec![
            ("short/greedy", &short, greedy(), wrap_n),
            ("long/greedy", &long, greedy(), LONG_DECODE),
        ];
        if rig.spec.supports_temp_verify() {
            a.push(("short/sampled", &short, sampled(0xC0FF_EE11_98), wrap_n));
            a.push(("long/sampled", &long, sampled(0x1198_5EED), LONG_DECODE));
        } else {
            eprintln!("[dflash-prompt-end] sampled arms skipped: draft has no candidate selector");
        }
        a
    };
    for (name, prompt, cfg, n) in arms {
        run_arm(&mut rig, name, prompt, cfg, n)?;
    }

    // A cold prefill of a different prompt moves the descriptor: the older
    // prompt's exact retry fails closed.
    rig.cold_reset()?;
    rig.prefill_cold(&short, greedy())?;
    rig.prefill_cold(&long, greedy())?;
    assert_eq!(
        rig.spec.prompt_end_checkpoint(),
        Some((long.len(), prefix_fingerprint(&long))),
        "descriptor follows the newest cold prompt"
    );
    let err = rig
        .prefill_exact(&short, greedy())
        .expect_err("the superseded prompt must fail closed");
    assert!(err.contains("fingerprint"), "{err}");

    // A write below the prompt end (a divergent-render rewind) drops the payload.
    rig.spec
        .rewind_to(&mut rig.gpu, &mut rig.slot, long.len() - 1)?;
    assert_eq!(
        rig.spec.prompt_end_checkpoint(),
        None,
        "rewind below the prompt end"
    );
    // A cold prefill re-establishes it; reset drops it again.
    rig.cold_reset()?;
    rig.prefill_cold(&short, greedy())?;
    assert!(rig.spec.prompt_end_checkpoint().is_some());
    rig.cold_reset()?;
    assert_eq!(rig.spec.prompt_end_checkpoint(), None, "reset drops it");
    let err = rig
        .prefill_exact(&short, greedy())
        .expect_err("no payload: the exact retry must fail closed");
    assert!(err.contains("without a resident payload"), "{err}");

    // The suffix prefill of a longer conversation (a hit at or above the prompt
    // end) keeps the payload; one starting below it drops it.
    rig.cold_reset()?;
    let first = rig.prefill_cold(&short, greedy())?;
    let desc = rig.spec.prompt_end_checkpoint();
    assert!(desc.is_some());
    let (toks, pos) = rig.decode(short.len(), first, 16, 0.0)?;
    let mut conversation = short.clone();
    conversation.extend_from_slice(&toks[..toks.len() - 1]);
    assert_eq!(conversation.len(), pos);
    let mut extended = conversation.clone();
    extended.extend_from_slice(&rig.tok.encode("\nContinue.\n"));
    rig.spec.configure_request(greedy());
    let never = || false;
    rig.spec.prefill(
        &mut rig.gpu,
        &mut rig.slot,
        &extended,
        &extended[pos..],
        pos,
        true,
        None,
        &never,
    )?;
    assert_eq!(
        rig.spec.prompt_end_checkpoint(),
        desc,
        "a suffix prefill above the prompt end keeps the payload"
    );

    // A suffix prefill that starts below the prompt end rewrites rows the
    // payload describes: it drops the payload.
    let below = short.len() - 8;
    rig.spec.configure_request(greedy());
    rig.spec.prefill(
        &mut rig.gpu,
        &mut rig.slot,
        &short,
        &short[below..],
        below,
        true,
        Some(below),
        &never,
    )?;
    assert_eq!(
        rig.spec.prompt_end_checkpoint(),
        None,
        "a suffix prefill below the prompt end drops the payload"
    );
    Ok(())
}
