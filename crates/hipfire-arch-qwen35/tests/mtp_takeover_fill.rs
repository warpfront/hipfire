// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MTP-head KV after an ngram-mod/PLD takeover window.
//!
//! A takeover commits rows the MTP head never proposed. It must leave the head
//! KV exactly as one-token native MTP decode over the same tokens would: slot
//! `p` written from `(tok_p, h_{p-1})`, where `h_{p-1}` is the trunk hidden of
//! the previous position. Only the rows the trunk kept may be written.
//!
//! Reference arm: prompt fill, then one-token windows. Each window first runs
//! the head write a native window's first proposal step makes
//! (`mtp_head_forward_block_only(seed, prev_hidden, cur_pos)`), then a k=0
//! verify commits the bonus. Takeover arm: the same prompt fill, then
//! takeover windows over the reference continuation (partial accept, full
//! accept, zero accept).
//!
//! Checks:
//! 1. Head route: every filled row is byte-identical to the decode-route
//!    per-row forward of the same token and hidden.
//! 2. Pairing: fill row `i` holds the pre-window hidden (i = 0) or verify
//!    row `i - 1`, i.e. the hidden decode would pair with that token.
//! 3. End-to-end vs the reference arm. Trunk hiddens differ in low bits
//!    between a 1-row and an m-row verify, so rows are compared dequantized
//!    and must be far closer to the reference than an off-by-one pairing.
//! 4. Rows past the committed prefix are never written, and native MTP
//!    drafts again after the takeovers.
//!
//! ```bash
//! HIPFIRE_DEVICES=<uuid> HIPFIRE_MTP_BYTE_IDENTITY_MODEL=<trunk> \
//!   HIPFIRE_MTP_BYTE_IDENTITY_HEAD=<head.mtp> cargo test --release \
//!   -p hipfire-arch-qwen35 --test mtp_takeover_fill -- --ignored --nocapture \
//!   --test-threads=1 --exact takeover_fills_head_kv_like_one_token_decode
//! ```
//!
//! # Prompt-fill fast/off identity oracle (`HIPFIRE_MTP_FILL_FAST`)
//!
//! The head's prompt fill has a kill switch, `HIPFIRE_MTP_FILL_FAST` (default
//! on, `0` = legacy fill). The switch is a process-snapshot read, so a mode
//! cannot be toggled inside a process: the oracle runs ONE ARM PER PROCESS and
//! compares the two artifacts. Nothing here mutates the environment; the spawn
//! test sets the switch on the child `Command` only.
//!
//! - `mtp_fill_oracle_arm` (GPU): runs one arm with whatever
//!   `HIPFIRE_MTP_FILL_FAST` the process started with and writes a JSON
//!   artifact to `HIPFIRE_MTP_FILL_ORACLE_OUT`. Reuses one loaded trunk/head
//!   for every case, sequentially. `max_seq` = 16448 (needs >= 16400).
//!   * State phase (direct `prefill_trunk_and_mtp_cache`, `ArRoute`, Q8 head
//!     KV): cold fills of 1, 2, 383, 384, 385, 512, 513, 2055, 8058 and 16059
//!     tokens of a deterministic repeated-sentence fixture (every prompt is a
//!     prefix of one stream), and warm suffix fills (prefix filled cold, then
//!     the suffix at `start_pos = prefix`, i.e. the cache-hit protocol) for
//!     prefix/total 1/2, 383/385, 512/2055, 8058/16059. Per stage it records
//!     sha256 of the head K and V rows the stage wrote (cold: `[0, L)`;
//!     suffix: stages `prefix` `[0, P)`, `suffix` `[P, L)`, `full` `[0, L)`),
//!     64-row block digests, sha256 of `prev_hidden` bytes and of the trunk
//!     logits, the argmax token, and the fill time (informational only, not
//!     compared).
//!   * Device witness (right after `Gpu::init`): the daemon's HIP-observation
//!     + `hipfire_config::devices` verification, without taking `GpuLocks`.
//!     Requires exactly one HIP-visible device, verifies it against the
//!     `HIPFIRE_DEVICES` selection `Gpu::init` applied, and prints the daemon's
//!     `[devices] verified logical 0: HIP <arch> <pci> = index N <lock identity>
//!     rocr=...` line on stderr (without a selection it prints the daemon's
//!     `[devices] logical ...` line and records `verified=false`; the compare
//!     then fails). The observed arch, PCI bus id, UUID, index and lock
//!     identity go in the artifact's `device` object, are compared across arms
//!     and printed as `MTP_FILL_DEVICE`.
//!   * Reset proof (first thing in the state phase, before any fill): the
//!     logical head K/V span of every layer buffer (`numel() * 4` bytes, the
//!     span reset zeros; allocator padding beyond it is ignored) is poisoned with 0xA5 via host upload
//!     (read back to prove the poison landed, so a freshly zero-allocated
//!     cache cannot mask a broken reset), `MtpSpecState::reset` runs, the
//!     logical K/V span is read back and asserted all-zero, and its
//!     sha256 is recorded as the artifact's `reset` object (compared across
//!     arms, printed as `MTP_FILL_ROW kind=reset`). Uninitialized trunk
//!     logits and `prev_hidden` are never read before the first fill.
//!   * Generation phase (production `build_qwen35_mtp_speculator`, K = 3,
//!     `Speculator::configure_request`/`prefill`/`step`, EOS disabled so the
//!     length is fixed at 64 tokens, the first drawn by prefill): greedy and
//!     sampled (T 0.7, top-p 0.95, top-k 20, fixed seed) over the same matrix
//!     as the state phase: a chat prompt, every cold length (`cold-<L>`), and
//!     every warm suffix case (`suffix-<P>-<L>`: prefix prefill, then suffix
//!     `prefill(.., start = P, cache_hit = true)`). 30 records (2 modes x 15
//!     cases); the compare fails if any case is missing. Records the exact
//!     token ids and the draft counts (`proposed`, `accepted`, `steps`,
//!     `windows`).
//! - `mtp_fill_oracle_compare` (no GPU): reads `HIPFIRE_MTP_FILL_ORACLE_A` and
//!   `_B`, prints the table, fails on any mismatch (with first differing
//!   token, or first differing 64-row block, as context).
//! - `mtp_fill_oracle_spawn` (GPU): one command; re-executes this test binary
//!   twice (`HIPFIRE_MTP_FILL_FAST=0` = `a`, then `=1` = `b`, sequentially so
//!   only one 27B session is resident), then compares. Artifacts and per-arm
//!   logs land in `HIPFIRE_MTP_FILL_ORACLE_DIR` (default
//!   `$TMPDIR/mtp_fill_oracle_<pid>`).
//!
//! Output table, one parseable `key=value` line per case/stage on stdout
//! (`MTP_FILL_ARMS`, `MTP_FILL_ROW kind=state|gen ... identical=true|false`,
//! `MTP_FILL_VERDICT identical=...`); `_a`/`_b` suffixes are the two arms and
//! digests are shown as their first 16 hex chars (full sha256 in the JSON).
//! The `fill_ms_a`/`fill_ms_b` columns show the two modes really ran
//! different fills; a fast arm that is not faster than the off arm on the long
//! rows means the switch did not take effect.
//!
//! ```bash
//! # build once (single thread of the build; no GPU needed to build)
//! cargo test --release -p hipfire-arch-qwen35 --test mtp_takeover_fill --no-run
//! BIN=$(ls -t target/release/deps/mtp_takeover_fill-* | grep -v '\.d$' | head -1)
//! export HIPFIRE_DEVICES=<uuid> \
//!   HIPFIRE_MTP_BYTE_IDENTITY_MODEL=/home/kaden/qcal/release-0.4.0/fp8-tp-ep-geo/models/qwen3.8-27b.mq4-xts \
//!   HIPFIRE_MTP_BYTE_IDENTITY_HEAD=<sidecar .mtp next to the model>
//! # one command: spawns off then fast, compares
//! "$BIN" --exact mtp_fill_oracle_spawn --ignored --nocapture --test-threads=1
//! # or manually, one arm per process, then compare
//! HIPFIRE_MTP_FILL_FAST=0 HIPFIRE_MTP_FILL_ORACLE_OUT=/tmp/fill_off.json \
//!   "$BIN" --exact mtp_fill_oracle_arm --ignored --nocapture --test-threads=1
//! HIPFIRE_MTP_FILL_FAST=1 HIPFIRE_MTP_FILL_ORACLE_OUT=/tmp/fill_fast.json \
//!   "$BIN" --exact mtp_fill_oracle_arm --ignored --nocapture --test-threads=1
//! HIPFIRE_MTP_FILL_ORACLE_A=/tmp/fill_off.json HIPFIRE_MTP_FILL_ORACLE_B=/tmp/fill_fast.json \
//!   "$BIN" --exact mtp_fill_oracle_compare --ignored --nocapture --test-threads=1
//! ```
//! The same manual arms also certify the PM GEMM: keep `HIPFIRE_MTP_FILL_FAST=1`
//! in both processes, change ONLY `HIPFIRE_MTP_FILL_PM_GEMM=0` / `=1`, then
//! run `mtp_fill_oracle_compare`. The artifact records both effective switches;
//! comparison refuses identical configurations and changing both switches.
//!
//! Single test thread is REQUIRED (`--test-threads=1`, and run GPU tests one
//! at a time): every GPU test loads the full trunk and head, and two sessions
//! do not fit. Do not set `HIPFIRE_MTP_NGRAM`: the oracle disables the n-gram
//! modifier per request, but its pool allocation would change the state
//! allocation sequence.

#![allow(clippy::all)]

use hipfire_arch_qwen35::mtp_head::{self, MtpKvMode};
use hipfire_arch_qwen35::mtp_spec::{
    prefill_trunk_and_mtp_cache, spec_step_mtp_compressed_serial_with_k,
    spec_step_mtp_compressed_serial_with_takeover_candidates, MtpPromptRoute, MtpSpecState,
};
use hipfire_arch_qwen35::mtp_speculator::build_qwen35_mtp_speculator;
use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
use hipfire_runtime::spec::{PrefillOutcome, SpecRequestConfig, SpecTarget, Speculator};
use rdna_compute::{Gpu, GpuTensor};
use std::path::PathBuf;

const CTX: usize = 2048;
const MAX_N: usize = 3;
const VERIFY_CAPACITY: usize = 64;
const REF_STEPS: usize = 72;
const PROMPT: &str = "Write a Rust function `fn parse_kv(line: &str) -> Option<(String, String)>` that splits \
a `key=value` line on the first '=', trims both sides, and rejects an empty key. Add three unit tests.";

fn tensor_bytes(gpu: &Gpu, t: &GpuTensor, offset_bytes: usize, len_bytes: usize) -> Vec<u8> {
    let mut all = vec![0u8; offset_bytes + len_bytes];
    gpu.hip.memcpy_dtoh(&mut all, &t.buf).expect("dtoh");
    all.split_off(offset_bytes)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 => sign * f32::INFINITY,
        _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Dequantize Q8_0 blocks (`f16 scale, 32 × i8`).
fn dequant_q8(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 34 * 32);
    for block in bytes.chunks_exact(34) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        out.extend(block[2..].iter().map(|&q| d * (q as i8) as f32));
    }
    out
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

/// Head KV rows `[start, start + n)`: K bytes then V bytes, per row.
struct HeadRows {
    row_bytes: usize,
    k: Vec<u8>,
    v: Vec<u8>,
}

impl HeadRows {
    fn read(gpu: &Gpu, state: &MtpSpecState, start: usize, n: usize) -> Self {
        let kv = &state.mtp_kv;
        let row_bytes = kv.n_head_kv * (kv.head_dim / 32) * 34;
        let k = tensor_bytes(gpu, &kv.inner.k_gpu[0], start * row_bytes, n * row_bytes);
        let v = tensor_bytes(gpu, &kv.inner.v_gpu[0], start * row_bytes, n * row_bytes);
        Self { row_bytes, k, v }
    }

    fn row(&self, i: usize) -> (&[u8], &[u8]) {
        let r = i * self.row_bytes..(i + 1) * self.row_bytes;
        (&self.k[r.clone()], &self.v[r])
    }

    fn row_f32(&self, i: usize) -> Vec<f32> {
        let (k, v) = self.row(i);
        let mut out = dequant_q8(k);
        out.extend(dequant_q8(v));
        out
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0 as u32
}

fn prefill(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    head: &mtp_head::Qwen35MtpHead,
    state: &mut MtpSpecState,
    prompt: &[u32],
) -> u32 {
    slot.reset_recurrent(gpu).expect("reset trunk");
    state.reset(gpu).expect("reset head");
    prefill_trunk_and_mtp_cache(gpu, slot, head, state, prompt, 0, MtpPromptRoute::ArRoute)
        .expect("prefill");
    argmax(&gpu.download_f32(&slot.scratch.logits).expect("logits"))
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_MTP_BYTE_IDENTITY_MODEL (qwen3.5/3.8 trunk) + MTP sidecar"]
fn takeover_fills_head_kv_like_one_token_decode() {
    let Some(model) = std::env::var("HIPFIRE_MTP_BYTE_IDENTITY_MODEL")
        .ok()
        .map(PathBuf::from)
    else {
        eprintln!("skipping: HIPFIRE_MTP_BYTE_IDENTITY_MODEL not set");
        return;
    };
    let head_path = std::env::var("HIPFIRE_MTP_BYTE_IDENTITY_HEAD")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model.with_extension("mtp"));
    let mut gpu = Gpu::init().expect("Gpu::init");
    let mut cfg = ModelSlotConfig::default();
    cfg.max_seq = CTX;
    let mut slot = ModelSlot::load(&mut gpu, &model, "takeover-fill", cfg).expect("load trunk");
    let head = mtp_head::load_mtp_head(&head_path, &mut gpu, CTX).expect("load head");
    let tokenizer = slot.load_tokenizer().expect("tokenizer");
    let prompt: Vec<u32> = [
        "<|im_start|>",
        "user",
        "\n",
        PROMPT,
        "<|im_end|>",
        "\n",
        "<|im_start|>",
        "assistant",
        "\n",
        "<think>\n\n</think>\n\n",
    ]
    .iter()
    .flat_map(|s| tokenizer.encode(s))
    .collect();
    let n = prompt.len();
    let eos = slot.config.eos_token;
    let dim = slot.config.dim;
    let mut state = MtpSpecState::new_for_slot_with_kv_mode_and_verify_capacity(
        &mut gpu,
        &slot,
        &head,
        MAX_N,
        VERIFY_CAPACITY,
        MtpKvMode::Q8,
    )
    .expect("state");
    eprintln!("[fill] gpu={} prompt={n} tokens", gpu.arch);

    // ── Reference arm: one-token native decode ──────────────────────────
    let seed0 = prefill(&mut gpu, &mut slot, &head, &mut state, &prompt);
    let mut toks = vec![seed0];
    let mut ref_hidden: Vec<Vec<f32>> = Vec::new();
    for i in 0..REF_STEPS {
        let pos = n + i;
        let seed = toks[i];
        ref_hidden.push(gpu.download_f32(&state.prev_hidden).expect("prev_hidden"));
        mtp_head::mtp_head_forward_block_only(
            &mut gpu,
            &head,
            &state.mtp_scratch,
            &mut state.mtp_kv,
            seed,
            &state.prev_hidden,
            None,
            pos,
            &slot.weights,
        )
        .expect("decode head row");
        let r = spec_step_mtp_compressed_serial_with_k(
            &mut gpu, &mut slot, &head, &mut state, pos, seed, eos, 0,
        )
        .expect("k=0 window");
        assert_eq!(r.advance, 1);
        assert_ne!(
            r.committed[0], eos,
            "reference continuation hit EOS; use a longer prompt"
        );
        toks.push(r.committed[0]);
    }
    let reference = HeadRows::read(&gpu, &state, n, REF_STEPS);
    eprintln!(
        "[fill] reference continuation: {:?}",
        tokenizer.decode(&toks)
    );

    // ── Takeover arm ─────────────────────────────────────────────────────
    let seed = prefill(&mut gpu, &mut slot, &head, &mut state, &prompt);
    assert_eq!(seed, seed0, "prompt fill must be deterministic");
    let untouched = HeadRows::read(&gpu, &state, n, VERIFY_CAPACITY + 1);
    let wrong = |t: u32| if t == 0 { 1 } else { t - 1 };

    // (label, candidates relative to the window seed index, expected accept)
    let mut idx = 0usize; // index into `toks` of the current seed
    let mut filled = 0usize;
    let windows: [(&str, usize, bool); 3] = [
        ("partial", 20, true),
        ("full", 30, false),
        ("zero", 0, true),
    ];
    for (label, accept, append_wrong) in windows {
        let pos = n + idx;
        let mut cands: Vec<u32> = toks[idx + 1..=idx + accept].to_vec();
        if append_wrong {
            cands.push(wrong(toks[idx + accept + 1]));
        }
        let r = spec_step_mtp_compressed_serial_with_takeover_candidates(
            &mut gpu, &mut slot, &head, &mut state, pos, toks[idx], eos, &cands,
        )
        .expect("takeover");
        eprintln!(
            "[fill] {label}: pos={pos} cands={} accept={} advance={}",
            cands.len(),
            r.accept_count,
            r.advance
        );
        assert_eq!(
            r.accept_count, accept,
            "{label}: takeover must accept the reference prefix"
        );
        assert_eq!(
            &r.committed[..],
            &toks[idx + 1..=idx + r.advance],
            "{label}: committed tokens"
        );
        let adv = r.advance;

        // 4. Nothing past the kept rows was written by this window.
        if label == "partial" {
            let after = HeadRows::read(&gpu, &state, pos + adv, cands.len() + 1 - adv);
            let before_rows = cands.len() + 1 - adv;
            for j in 0..before_rows {
                assert_eq!(
                    after.row(j),
                    untouched.row(adv + j),
                    "{label}: rejected row {j} written"
                );
            }
        }

        // 2. Pairing: staged hidden row i is what decode pairs with token i.
        let staged = gpu
            .download_f32(&state.takeover_fill_hidden)
            .expect("staged");
        let verify = gpu
            .download_f32(&state.verify_hidden)
            .expect("verify hidden");
        for i in 0..adv {
            let row = &staged[i * dim..(i + 1) * dim];
            if i > 0 {
                assert_eq!(
                    row,
                    &verify[(i - 1) * dim..i * dim],
                    "{label}: fill row {i} != verify row {}",
                    i - 1
                );
            }
            let hidden_rel = rel_l2(row, &ref_hidden[idx + i]);
            let shifted_rel = if idx + i + 1 < ref_hidden.len() {
                rel_l2(row, &ref_hidden[idx + i + 1])
            } else {
                f64::NAN
            };
            if i < 3 || i + 1 == adv {
                eprintln!("[fill] {label} row {i}: hidden rel-L2 vs decode h_(p-1)={hidden_rel:.3e}  vs h_p={shifted_rel:.3e}");
            }
            assert!(
                hidden_rel < 0.05,
                "{label} row {i}: hidden is not decode's h_(p-1) (rel {hidden_rel})"
            );
        }

        // 1. Head route: recompute each row with the decode-route forward.
        let filled_rows = HeadRows::read(&gpu, &state, pos, adv);
        let mut max_route_rel = 0f64;
        let mut route_identical = 0usize;
        for i in 0..adv {
            let hidden_row = state.takeover_fill_hidden.sub_offset(i * dim, dim);
            let tok = toks[idx + i];
            mtp_head::mtp_head_forward_block_only(
                &mut gpu,
                &head,
                &state.mtp_scratch,
                &mut state.mtp_kv,
                tok,
                &hidden_row,
                None,
                pos + i,
                &slot.weights,
            )
            .expect("decode-route recompute");
        }
        let decode_route = HeadRows::read(&gpu, &state, pos, adv);
        for i in 0..adv {
            if filled_rows.row(i) == decode_route.row(i) {
                route_identical += 1;
            }
            max_route_rel =
                max_route_rel.max(rel_l2(&filled_rows.row_f32(i), &decode_route.row_f32(i)));
        }
        eprintln!(
            "[fill] {label}: head route byte-identical rows {route_identical}/{adv}, max rel-L2 {max_route_rel:.3e}"
        );

        // 3. End-to-end: the rows the takeover wrote vs the reference arm.
        let mut max_e2e = 0f64;
        let mut min_offby1 = f64::INFINITY;
        for i in 0..adv {
            let got = filled_rows.row_f32(i);
            let want = reference.row_f32(idx + i);
            max_e2e = max_e2e.max(rel_l2(&got, &want));
            if idx + i + 1 < REF_STEPS {
                min_offby1 = min_offby1.min(rel_l2(&got, &reference.row_f32(idx + i + 1)));
            }
        }
        eprintln!("[fill] {label}: rows vs one-token decode max rel-L2 {max_e2e:.3e}; nearest off-by-one row {min_offby1:.3e}");
        assert_eq!(
            route_identical, adv,
            "{label}: batched head route is not byte-identical to decode route ({max_route_rel})"
        );
        assert!(
            max_e2e < 0.1 * min_offby1,
            "{label}: filled rows not decode-paired ({max_e2e} vs off-by-one {min_offby1})"
        );

        idx += adv;
        filled += adv;
    }
    assert!(filled >= 50, "takeover windows filled only {filled} rows");

    // Native MTP drafts again after the takeovers.
    let pos = n + idx;
    let r = spec_step_mtp_compressed_serial_with_k(
        &mut gpu, &mut slot, &head, &mut state, pos, toks[idx], eos, MAX_N,
    )
    .expect("native window after takeovers");
    eprintln!(
        "[fill] native window after takeovers: drafts={} accept={} committed={:?}",
        r.drafts_generated, r.accept_count, r.committed
    );
    assert_eq!(
        r.drafts_generated, MAX_N,
        "native MTP must draft after a takeover"
    );
    assert_eq!(
        r.committed[0],
        toks[idx + 1],
        "greedy output diverged after takeovers"
    );

    state.free_gpu(&mut gpu);
    head.free_gpu(&mut gpu);
    slot.kv_cache.free_gpu(&mut gpu).expect("free kv");
    slot.dn_state.free_gpu(&mut gpu);
    slot.scratch.free_gpu(&mut gpu).expect("free scratch");
    slot.weights.free_gpu(&mut gpu);
}

// ═══════════════════════════════════════════════════════════════════════════
// Fill fast/off identity oracle (`HIPFIRE_MTP_FILL_FAST` = 1 vs 0)
// ═══════════════════════════════════════════════════════════════════════════

const FILL_ENV: &str = "HIPFIRE_MTP_FILL_FAST";
const MODEL_ENV: &str = "HIPFIRE_MTP_BYTE_IDENTITY_MODEL";
const HEAD_ENV: &str = "HIPFIRE_MTP_BYTE_IDENTITY_HEAD";
const OUT_ENV: &str = "HIPFIRE_MTP_FILL_ORACLE_OUT";
const ARM_A_ENV: &str = "HIPFIRE_MTP_FILL_ORACLE_A";
const ARM_B_ENV: &str = "HIPFIRE_MTP_FILL_ORACLE_B";
const DIR_ENV: &str = "HIPFIRE_MTP_FILL_ORACLE_DIR";
const ARM_TEST: &str = "mtp_fill_oracle_arm";

/// Trunk + head KV capacity: the longest fixture prompt (16059) plus decode
/// room, rounded to a multiple of 64.
const ORACLE_CTX: usize = 16448;
/// Cold prompt-fill lengths (chunk-boundary and long-context coverage).
const COLD_LENS: [usize; 10] = [1, 2, 383, 384, 385, 512, 513, 2055, 8058, 16059];
/// Warm suffix fills `(cached prefix, total length)`: the prefix is filled
/// cold, then the suffix is filled at `start_pos = prefix` with the cache hit
/// protocol (head KV and `prev_hidden` carried over).
const SUFFIX_CASES: [(usize, usize); 4] = [(1, 2), (383, 385), (512, 2055), (8058, 16059)];
const FIXTURE_LEN: usize = 16059;
const GEN_TOKENS: usize = 64;
const GEN_K: usize = 3;
/// State records: one per cold length, three stages per suffix case.
const EXPECTED_STATE_RECORDS: usize = COLD_LENS.len() + 3 * SUFFIX_CASES.len();
/// Generation records: per sampling mode, chat + every cold length + every suffix case.
const EXPECTED_GEN_RECORDS: usize = 2 * (1 + COLD_LENS.len() + SUFFIX_CASES.len());
const SAMPLED_SEED: u64 = 0x4D54_5046_494C_4C31;
/// Rows per digest block (localizes a mismatch to a row range).
const BLOCK_ROWS: usize = 64;
const FIXTURE_TEXT: &str =
    "The harbor bell rang twice before dawn, and the gulls answered in kind. ";

fn sha_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Deterministic safe-repetition token stream: one encoded sentence repeated.
/// Prompts of every length are prefixes of the same stream.
fn fixture(tokenizer: &hipfire_runtime::tokenizer::Tokenizer, vocab: usize, eos: u32) -> Vec<u32> {
    let unit = tokenizer.encode(FIXTURE_TEXT);
    assert!(!unit.is_empty(), "fixture sentence encoded to no tokens");
    assert!(
        unit.iter().all(|&t| (t as usize) < vocab && t != eos),
        "fixture sentence encodes to out-of-vocab or EOS tokens"
    );
    (0..FIXTURE_LEN).map(|i| unit[i % unit.len()]).collect()
}

/// Timed fill without the cold reset (warm suffix) or after it (cold).
fn fill_ms(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    head: &mtp_head::Qwen35MtpHead,
    state: &mut MtpSpecState,
    tokens: &[u32],
    start_pos: usize,
) -> f64 {
    let t = prefill_trunk_and_mtp_cache(
        gpu,
        slot,
        head,
        state,
        tokens,
        start_pos,
        MtpPromptRoute::ArRoute,
    )
    .expect("prefill_trunk_and_mtp_cache");
    t.mtp_prompt_fill_secs * 1e3
}

fn cold_reset(gpu: &mut Gpu, slot: &mut ModelSlot, state: &mut MtpSpecState) {
    slot.reset_recurrent(gpu).expect("reset trunk");
    state.reset(gpu).expect("reset head");
}

/// Digest record of head KV rows `[start, end)` (K and V written bytes, plus
/// per-block digests), `prev_hidden` bytes and the trunk logits.
fn state_record(
    gpu: &Gpu,
    slot: &ModelSlot,
    state: &MtpSpecState,
    case: &str,
    stage: &str,
    start: usize,
    end: usize,
    fill_ms: f64,
) -> serde_json::Value {
    assert_eq!(
        state.prev_hidden_pos,
        Some(end - 1),
        "{case}/{stage}: prev_hidden_pos must be the last filled position"
    );
    // Oracle observation boundary (not production timing): wait for every
    // queued fill/copy before reading cache, prev_hidden and logits.
    gpu.hip
        .device_synchronize()
        .expect("device_synchronize before state read");
    let rows = HeadRows::read(gpu, state, start, end - start);
    let blocks = |bytes: &[u8]| -> Vec<String> {
        bytes
            .chunks(BLOCK_ROWS * rows.row_bytes)
            .map(|c| sha_hex(c)[..16].to_string())
            .collect()
    };
    let prev = tensor_bytes(gpu, &state.prev_hidden, 0, slot.config.dim * 4);
    let logits = gpu.download_f32(&slot.scratch.logits).expect("logits");
    serde_json::json!({
        "case": case,
        "stage": stage,
        "start": start,
        "end": end,
        "k_sha": sha_hex(&rows.k),
        "v_sha": sha_hex(&rows.v),
        "k_blocks": blocks(&rows.k),
        "v_blocks": blocks(&rows.v),
        "prev_sha": sha_hex(&prev),
        "logits_sha": sha_hex(&f32_bytes(&logits)),
        "argmax": argmax(&logits),
        "prev_hidden_pos": end - 1,
        "fill_ms": fill_ms,
    })
}

/// Poisons the logical head K/V cache (`numel() * 4` bytes of every layer
/// buffer, the span `KvCache` reset zeros; allocator padding past it is left
/// alone) with a non-zero pattern, verifies the poison landed, calls
/// `MtpSpecState::reset`, reads the logical cache back, asserts every byte is
/// zero, and returns the digests of those bytes. Trunk logits/`prev_hidden`
/// are never read here (they are uninitialized before the first fill).
fn reset_proof(gpu: &mut Gpu, state: &mut MtpSpecState) -> serde_json::Value {
    const POISON: u8 = 0xA5;
    let layers = state.mtp_kv.inner.k_gpu.len();
    assert_eq!(
        layers,
        state.mtp_kv.inner.v_gpu.len(),
        "K/V layer count mismatch"
    );
    // Logical span: exactly what `KvCache` reset zeros (`numel() * 4`). The
    // device allocation may be larger (pool over-allocation / padding) and
    // that tail is intentionally neither poisoned, read nor hashed.
    let span = |t: &GpuTensor| -> usize {
        let n = t.numel() * 4;
        assert!(
            n <= t.buf.size(),
            "logical span {n} exceeds allocation {}",
            t.buf.size()
        );
        n
    };
    let read_all = |gpu: &Gpu, bufs: &[GpuTensor]| -> Vec<u8> {
        let mut all = Vec::new();
        for t in bufs {
            all.extend_from_slice(&tensor_bytes(gpu, t, 0, span(t)));
        }
        all
    };
    for t in state
        .mtp_kv
        .inner
        .k_gpu
        .iter()
        .chain(state.mtp_kv.inner.v_gpu.iter())
    {
        gpu.hip
            .memcpy_htod(&t.buf, &vec![POISON; span(t)])
            .expect("poison head kv");
    }
    gpu.hip
        .device_synchronize()
        .expect("device_synchronize after poison");
    let (pk, pv) = (
        read_all(gpu, &state.mtp_kv.inner.k_gpu),
        read_all(gpu, &state.mtp_kv.inner.v_gpu),
    );
    assert!(
        !pk.is_empty() && pk.iter().chain(pv.iter()).all(|&b| b == POISON),
        "poison did not land in the whole head KV; the reset proof would be vacuous"
    );
    let poison_sha = sha_hex(&[pk.as_slice(), pv.as_slice()].concat());
    state.reset(gpu).expect("reset head");
    // The fast reset is stream-async; observe only after every queued memset
    // completed, so a stream-vs-DTOH ordering gap cannot read pre-reset bytes.
    gpu.hip
        .device_synchronize()
        .expect("device_synchronize after reset");
    let (k, v) = (
        read_all(gpu, &state.mtp_kv.inner.k_gpu),
        read_all(gpu, &state.mtp_kv.inner.v_gpu),
    );
    let nonzero_k = k.iter().position(|&b| b != 0);
    let nonzero_v = v.iter().position(|&b| b != 0);
    assert!(
        nonzero_k.is_none() && nonzero_v.is_none(),
        "MtpSpecState::reset left non-zero head KV bytes: first nonzero K byte {nonzero_k:?} (of {}), V byte {nonzero_v:?} (of {})",
        k.len(),
        v.len()
    );
    eprintln!(
        "[oracle] reset proof: {} K + {} V bytes poisoned then zeroed",
        k.len(),
        v.len()
    );
    serde_json::json!({
        "layers": layers,
        "k_bytes": k.len(),
        "v_bytes": v.len(),
        "poison_sha": poison_sha,
        "k_sha": sha_hex(&k),
        "v_sha": sha_hex(&v),
        "all_zero": true,
    })
}

fn state_phase(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    head: &mtp_head::Qwen35MtpHead,
    tokens: &[u32],
) -> (Vec<serde_json::Value>, serde_json::Value) {
    let mut state = MtpSpecState::new_for_slot_with_kv_mode_and_verify_capacity(
        gpu,
        slot,
        head,
        MAX_N,
        MAX_N,
        MtpKvMode::Q8,
    )
    .expect("state");
    // Reset proof before any fill: the freshly allocated cache is zero, which
    // would mask a broken reset, so poison every K/V byte first.
    let reset = reset_proof(gpu, &mut state);
    let mut out = Vec::new();
    for len in COLD_LENS {
        cold_reset(gpu, slot, &mut state);
        let ms = fill_ms(gpu, slot, head, &mut state, &tokens[..len], 0);
        let case = format!("cold-{len}");
        out.push(state_record(gpu, slot, &state, &case, "cold", 0, len, ms));
        eprintln!("[oracle] state {case}: fill {ms:.1} ms");
    }
    for (p, len) in SUFFIX_CASES {
        let case = format!("suffix-{p}-{len}");
        cold_reset(gpu, slot, &mut state);
        let ms = fill_ms(gpu, slot, head, &mut state, &tokens[..p], 0);
        out.push(state_record(gpu, slot, &state, &case, "prefix", 0, p, ms));
        let ms = fill_ms(gpu, slot, head, &mut state, &tokens[p..len], p);
        // Rows the suffix wrote, then the whole resident prefix + suffix.
        out.push(state_record(gpu, slot, &state, &case, "suffix", p, len, ms));
        out.push(state_record(gpu, slot, &state, &case, "full", 0, len, ms));
        eprintln!("[oracle] state {case}: suffix fill {ms:.1} ms");
    }
    state.free_gpu(gpu);
    (out, reset)
}

/// Test-only device identity witness, the daemon's `observe_hip_devices` +
/// `reserve_and_verify_gpus` verification WITHOUT taking `GpuLocks` (the
/// caller's fabric owns the lock). Queries the HIP devices this process
/// actually sees (arch and PCI bus id from HIP; UUID only where identity comes
/// from HIP), requires exactly one (the single-card oracle), maps it to its
/// physical identity through `hipfire_config::devices`, and prints the same
/// `[devices] verified logical ...` line the daemon logs. With a resolved
/// `HIPFIRE_DEVICES` selection (applied by `Gpu::init`) the observed device
/// must equal the resolved card (`verify_visible`); with none, the identity is
/// derived from the observation (`identify_observed`) and the daemon's
/// `[devices] logical ...` line is printed instead, and `verified` is false.
fn device_witness(gpu: &Gpu) -> serde_json::Value {
    use hipfire_config::devices::{self, DeviceRoot, IdentitySource, ObservedDevice};
    let root = DeviceRoot::system();
    let source = root.identity_source();
    let count = gpu.hip.device_count().expect("hip device_count");
    assert_eq!(
        count, 1,
        "oracle expects exactly one HIP-visible device, saw {count}"
    );
    let observed: Vec<ObservedDevice> = (0..count)
        .map(|id| ObservedDevice {
            logical: id as usize,
            arch: gpu.hip.get_arch(id).expect("hip get_arch"),
            pci_bus_id: gpu
                .hip
                .device_pci_bus_id(id)
                .expect("hip device_pci_bus_id"),
            uuid: (source == IdentitySource::Hip)
                .then(|| gpu.hip.device_uuid(id).expect("hip device_uuid")),
        })
        .collect();
    let seen = &observed[0];
    let (device, verified) = if let Some(active) = devices::active_devices() {
        active
            .verify_visible(&observed)
            .unwrap_or_else(|e| panic!("[devices] verification failed: {e}"));
        let device = active.devices[0].clone();
        eprintln!(
            "[devices] verified logical {}: HIP {} {} = index {} {} rocr={}",
            seen.logical,
            seen.arch,
            seen.pci_bus_id,
            device.index,
            device.lock_identity(),
            device.rocr_selector()
        );
        (device, true)
    } else {
        assert!(
            !hipfire_config::legacy_devices_env_set(),
            "HIPFIRE_DEVICES/HIPFIRE_DEVICE is set but Gpu::init applied no device selection; cannot verify the card"
        );
        let device = devices::identify_observed(&root, &observed)
            .unwrap_or_else(|e| panic!("[devices] identification failed: {e}"))
            .remove(0);
        let note = match source {
            IdentitySource::Kfd => "",
            IdentitySource::Hip => " (identity from HIP: no KFD topology)",
        };
        eprintln!(
            "[devices] logical {}: HIP {} {} = index {} {}{note}",
            seen.logical,
            seen.arch,
            seen.pci_bus_id,
            device.index,
            device.lock_identity()
        );
        (device, false)
    };
    serde_json::json!({
        "count": count,
        "logical": seen.logical,
        "arch": seen.arch,
        "pci_bus_id": seen.pci_bus_id,
        "uuid": device.uuid(),
        "index": device.index,
        "lock_identity": device.lock_identity(),
        "rocr": device.rocr_selector(),
        "identity_source": format!("{source:?}"),
        "verified": verified,
    })
}

fn ready(outcome: PrefillOutcome) -> u32 {
    match outcome {
        PrefillOutcome::Ready { first_token } => first_token,
        PrefillOutcome::Aborted => panic!("prefill aborted"),
    }
}

/// One generation through the production speculator: `configure_request`,
/// `prefill` (cold, or prefix + warm suffix), then `step` until `GEN_TOKENS`
/// tokens (the first drawn by prefill included).
fn generation_record(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    spec: &mut dyn Speculator,
    case: &str,
    mode: &str,
    req: SpecRequestConfig,
    prompt: &[u32],
    prefix: Option<usize>,
) -> serde_json::Value {
    spec.configure_request(req);
    let first = match prefix {
        None => ready(
            spec.prefill(gpu, slot, prompt, prompt, 0, false, None, &|| false)
                .expect("prefill"),
        ),
        Some(p) => {
            ready(
                spec.prefill(
                    gpu,
                    slot,
                    &prompt[..p],
                    &prompt[..p],
                    0,
                    false,
                    None,
                    &|| false,
                )
                .expect("prefix prefill"),
            );
            ready(
                spec.prefill(gpu, slot, prompt, &prompt[p..], p, true, None, &|| false)
                    .expect("suffix prefill"),
            )
        }
    };
    let mut generated = vec![first];
    let (mut position, mut seed) = (prompt.len(), first);
    let (mut steps, mut proposed, mut accepted) = (0usize, 0usize, 0usize);
    while generated.len() < GEN_TOKENS {
        let remaining = GEN_TOKENS - generated.len();
        let step = spec
            .step(
                gpu, slot, position, seed, &generated, None, req.temp, remaining,
            )
            .expect("spec step");
        assert!(
            !step.emit.is_empty(),
            "{case}/{mode}: empty step at {position}"
        );
        assert!(
            step.emit.len() <= remaining,
            "{case}/{mode}: step overshoots max_emit"
        );
        steps += 1;
        proposed += step.proposed;
        accepted += step.accepted;
        position += step.emit.len();
        seed = step.next_seed;
        generated.extend(step.emit.iter().copied());
    }
    let stats = spec.request_stats();
    eprintln!(
        "[oracle] gen {case}/{mode}: tokens={} steps={steps} proposed={proposed} accepted={accepted}",
        generated.len()
    );
    serde_json::json!({
        "case": case,
        "mode": mode,
        "prompt_len": prompt.len(),
        "prefill_start": prefix.unwrap_or(0),
        "tokens": generated,
        "token_sha": sha_hex(&u32_bytes(&generated)),
        "generated": generated.len(),
        "steps": steps,
        "proposed": proposed,
        "accepted": accepted,
        "windows": stats.mtp_windows,
    })
}

fn chat_prompt(tokenizer: &hipfire_runtime::tokenizer::Tokenizer) -> Vec<u32> {
    [
        "<|im_start|>",
        "user",
        "\n",
        PROMPT,
        "<|im_end|>",
        "\n",
        "<|im_start|>",
        "assistant",
        "\n",
        "<think>\n\n</think>\n\n",
    ]
    .iter()
    .flat_map(|s| tokenizer.encode(s))
    .collect()
}

/// One arm in THIS process. The fill mode is whatever `HIPFIRE_MTP_FILL_FAST`
/// held at process start (the production switch is a process-snapshot read);
/// this test never mutates the environment. Writes the artifact to
/// `HIPFIRE_MTP_FILL_ORACLE_OUT`.
#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_MTP_BYTE_IDENTITY_MODEL + MTP sidecar + HIPFIRE_MTP_FILL_ORACLE_OUT"]
fn mtp_fill_oracle_arm() {
    let Some(model) = std::env::var(MODEL_ENV).ok().map(PathBuf::from) else {
        eprintln!("skipping: {MODEL_ENV} not set");
        return;
    };
    let out_path = PathBuf::from(
        std::env::var(OUT_ENV)
            .unwrap_or_else(|_| panic!("{OUT_ENV} must name the artifact JSON to write")),
    );
    let head_path = std::env::var(HEAD_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| model.with_extension("mtp"));
    let fill_env = std::env::var(FILL_ENV).unwrap_or_else(|_| "unset".to_string());

    let mut gpu = Gpu::init().expect("Gpu::init");
    let device = device_witness(&gpu);
    let mut cfg = ModelSlotConfig::default();
    cfg.max_seq = ORACLE_CTX;
    let mut slot = ModelSlot::load(&mut gpu, &model, "mtp-fill-oracle", cfg).expect("load trunk");
    let head = mtp_head::load_mtp_head(&head_path, &mut gpu, ORACLE_CTX).expect("load head");
    assert_eq!(
        head.config.vocab_size, slot.config.vocab_size,
        "trunk/head vocab mismatch"
    );
    let tokenizer = slot.load_tokenizer().expect("tokenizer");
    let real_eos = slot.config.eos_token;
    let tokens = fixture(&tokenizer, slot.config.vocab_size, real_eos);
    eprintln!(
        "[oracle] arm {FILL_ENV}={fill_env} gpu={} ctx={ORACLE_CTX}",
        gpu.arch
    );

    // ── State oracle (fill digests), head borrowed ───────────────────────
    let (state, reset) = state_phase(&mut gpu, &mut slot, &head, &tokens);

    // ── Generation oracle through the production speculator ──────────────
    // Fixed-length: EOS disabled for both sampling modes, restored after.
    let mut spec = build_qwen35_mtp_speculator(head, GEN_K, ORACLE_CTX);
    slot.config.eos_token = u32::MAX;
    let greedy = SpecRequestConfig::default();
    let sampled = SpecRequestConfig {
        temp: 0.7,
        top_p: 0.95,
        top_k: Some(20),
        rng_seed: SAMPLED_SEED,
        ..SpecRequestConfig::default()
    };
    assert!(
        spec.supports_temp_verify(),
        "speculator must advertise sampled verification"
    );
    let chat = chat_prompt(&tokenizer);
    let mut generation = Vec::new();
    for (mode, req) in [("greedy", greedy), ("sampled", sampled)] {
        generation.push(generation_record(
            &mut gpu,
            &mut slot,
            spec.as_mut(),
            "chat",
            mode,
            req,
            &chat,
            None,
        ));
        for len in COLD_LENS {
            let case = format!("cold-{len}");
            generation.push(generation_record(
                &mut gpu,
                &mut slot,
                spec.as_mut(),
                &case,
                mode,
                req,
                &tokens[..len],
                None,
            ));
        }
        for (p, len) in SUFFIX_CASES {
            let case = format!("suffix-{p}-{len}");
            generation.push(generation_record(
                &mut gpu,
                &mut slot,
                spec.as_mut(),
                &case,
                mode,
                req,
                &tokens[..len],
                Some(p),
            ));
        }
    }
    slot.config.eos_token = real_eos;

    let artifact = serde_json::json!({
        "schema": 1,
        "fill_fast_env": fill_env,
        "fill_pm_gemm": hipfire_config::developer_bool("HIPFIRE_MTP_FILL_PM_GEMM", true),
        "gpu": format!("{}", gpu.arch),
        "model": model.display().to_string(),
        "head": head_path.display().to_string(),
        "max_seq": ORACLE_CTX,
        "fixture_sha": sha_hex(&u32_bytes(&tokens)),
        "device": device,
        "reset": reset,
        "state": state,
        "generation": generation,
    });
    std::fs::write(
        &out_path,
        serde_json::to_vec_pretty(&artifact).expect("serialize"),
    )
    .expect("write artifact");
    eprintln!("[oracle] wrote {}", out_path.display());

    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    spec.free(&mut gpu);
    slot.kv_cache.free_gpu(&mut gpu).expect("free kv");
    slot.dn_state.free_gpu(&mut gpu);
    slot.scratch.free_gpu(&mut gpu).expect("free scratch");
    slot.weights.free_gpu(&mut gpu);
    gpu.drain_pool();
}

fn short(v: &serde_json::Value) -> String {
    match v.as_str() {
        Some(s) => s.chars().take(16).collect(),
        None => v.to_string(),
    }
}

/// First index where two token arrays differ, with a window of context.
fn token_diff_context(a: &serde_json::Value, b: &serde_json::Value) -> String {
    let (a, b) = (
        a.as_array().cloned().unwrap_or_default(),
        b.as_array().cloned().unwrap_or_default(),
    );
    match (0..a.len().max(b.len())).find(|&i| a.get(i) != b.get(i)) {
        None => "no token difference".into(),
        Some(i) => {
            let lo = i.saturating_sub(4);
            let win = |v: &Vec<serde_json::Value>| {
                v[lo.min(v.len())..(i + 5).min(v.len())]
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            };
            format!(
                "first token diff at index {i}: a[{lo}..]=[{}] b[{lo}..]=[{}]",
                win(&a),
                win(&b)
            )
        }
    }
}

/// First differing digest block (rows `start + i * BLOCK_ROWS ..`).
fn block_diff_context(
    a: &serde_json::Value,
    b: &serde_json::Value,
    key: &str,
    start: u64,
) -> String {
    let (ab, bb) = (
        a[key].as_array().cloned().unwrap_or_default(),
        b[key].as_array().cloned().unwrap_or_default(),
    );
    match (0..ab.len().max(bb.len())).find(|&i| ab.get(i) != bb.get(i)) {
        None => format!("{key}: no block difference"),
        Some(i) => {
            let first = start + (i * BLOCK_ROWS) as u64;
            format!(
                "{key}: first differing block {i} = rows {first}..{} (a={} b={})",
                first + BLOCK_ROWS as u64,
                ab.get(i).map(short).unwrap_or_default(),
                bb.get(i).map(short).unwrap_or_default()
            )
        }
    }
}

/// Prints one `MTP_FILL_ROW` line per case/stage and returns failures.
/// `a` is the first artifact, `b` the second (the spawn test passes
/// `a = HIPFIRE_MTP_FILL_FAST=0`, `b = 1`).
fn compare_artifacts(a: &serde_json::Value, b: &serde_json::Value) -> Vec<String> {
    let mut failures: Vec<String> = Vec::new();
    println!(
        "MTP_FILL_ARMS a_env={} b_env={} gpu={} max_seq={} fixture_sha={}",
        a["fill_fast_env"].as_str().unwrap_or("?"),
        b["fill_fast_env"].as_str().unwrap_or("?"),
        short(&a["gpu"]),
        a["max_seq"],
        short(&a["fixture_sha"]),
    );
    for key in ["gpu", "model", "head", "max_seq", "fixture_sha"] {
        if a[key] != b[key] {
            failures.push(format!("arms differ in {key}: a={} b={}", a[key], b[key]));
        }
    }
    let fast_diff = a["fill_fast_env"] != b["fill_fast_env"];
    let pm_diff = a["fill_pm_gemm"] != b["fill_pm_gemm"];
    if fast_diff == pm_diff {
        failures.push("oracle arms must differ in exactly one fill switch".into());
    }
    if pm_diff && (a["fill_fast_env"] != "1" || b["fill_fast_env"] != "1") {
        failures.push("PM GEMM oracle requires fast fill enabled in both arms".into());
    }
    println!(
        "MTP_FILL_PM_GEMM a={} b={}",
        a["fill_pm_gemm"], b["fill_pm_gemm"]
    );
    let (da, db) = (&a["device"], &b["device"]);
    {
        const DEVICE_FIELDS: [&str; 9] = [
            "logical",
            "arch",
            "pci_bus_id",
            "uuid",
            "index",
            "lock_identity",
            "rocr",
            "identity_source",
            "verified",
        ];
        let bad: Vec<&str> = DEVICE_FIELDS
            .iter()
            .copied()
            .filter(|f| da[*f] != db[*f] || da[*f].is_null() && *f != "uuid")
            .collect();
        println!(
            "MTP_FILL_DEVICE identical={} count_a={} count_b={} arch={} pci={} index={} lock_identity={} rocr={} uuid={} verified={} source={}",
            bad.is_empty(), da["count"], db["count"], da["arch"], da["pci_bus_id"], da["index"],
            da["lock_identity"], da["rocr"], da["uuid"], da["verified"], da["identity_source"],
        );
        if !bad.is_empty()
            || da["count"] != 1
            || db["count"] != 1
            || da["verified"] != true
            || db["verified"] != true
        {
            let mut ctx = format!(
                "device witness: differing/missing {bad:?}, count a={} b={}, verified a={} b={}",
                da["count"], db["count"], da["verified"], db["verified"]
            );
            for f in &bad {
                ctx += &format!("\n    {f}: a={} b={}", da[*f], db[*f]);
            }
            failures.push(ctx);
        }
    }

    let (ra, rb) = (&a["reset"], &b["reset"]);
    {
        const RESET_FIELDS: [&str; 7] = [
            "layers",
            "k_bytes",
            "v_bytes",
            "poison_sha",
            "k_sha",
            "v_sha",
            "all_zero",
        ];
        let bad: Vec<&str> = RESET_FIELDS
            .iter()
            .copied()
            .filter(|f| ra[*f] != rb[*f] || ra[*f].is_null())
            .collect();
        let zero = ra["all_zero"] == true && rb["all_zero"] == true;
        println!(
            "MTP_FILL_ROW kind=reset case=head-kv-reset stage=poison-then-reset rows=0..{} identical={} all_zero_a={} all_zero_b={} k_bytes={} v_bytes={} poison_a={} poison_b={} k_a={} k_b={} v_a={} v_b={}",
            a["max_seq"], bad.is_empty() && zero, ra["all_zero"], rb["all_zero"], ra["k_bytes"], ra["v_bytes"],
            short(&ra["poison_sha"]), short(&rb["poison_sha"]),
            short(&ra["k_sha"]), short(&rb["k_sha"]), short(&ra["v_sha"]), short(&rb["v_sha"]),
        );
        if !bad.is_empty() || !zero {
            let mut ctx = format!(
                "head KV reset: differing/missing {bad:?}, all_zero a={} b={}",
                ra["all_zero"], rb["all_zero"]
            );
            for f in &bad {
                ctx += &format!("\n    {f}: a={} b={}", ra[*f], rb[*f]);
            }
            failures.push(ctx);
        }
    }

    let (sa, sb) = (
        a["state"].as_array().cloned().unwrap_or_default(),
        b["state"].as_array().cloned().unwrap_or_default(),
    );
    if sa.len() != sb.len() || sa.len() != EXPECTED_STATE_RECORDS {
        failures.push(format!(
            "state record counts: a={} b={}, expected {EXPECTED_STATE_RECORDS} (missing or extra cases)",
            sa.len(),
            sb.len()
        ));
    }
    const STATE_FIELDS: [&str; 8] = [
        "start",
        "end",
        "k_sha",
        "v_sha",
        "prev_sha",
        "logits_sha",
        "argmax",
        "prev_hidden_pos",
    ];
    for (ra, rb) in sa.iter().zip(&sb) {
        let (case, stage) = (
            ra["case"].as_str().unwrap_or("?"),
            ra["stage"].as_str().unwrap_or("?"),
        );
        let mut bad: Vec<&str> = Vec::new();
        if ra["case"] != rb["case"] || ra["stage"] != rb["stage"] {
            bad.push("case/stage");
        }
        for f in STATE_FIELDS {
            if ra[f] != rb[f] {
                bad.push(f);
            }
        }
        println!(
            "MTP_FILL_ROW kind=state case={case} stage={stage} rows={}..{} identical={} k_a={} k_b={} v_a={} v_b={} prev_a={} prev_b={} logits_a={} logits_b={} argmax_a={} argmax_b={} fill_ms_a={:.1} fill_ms_b={:.1}",
            ra["start"], ra["end"], bad.is_empty(),
            short(&ra["k_sha"]), short(&rb["k_sha"]), short(&ra["v_sha"]), short(&rb["v_sha"]),
            short(&ra["prev_sha"]), short(&rb["prev_sha"]), short(&ra["logits_sha"]), short(&rb["logits_sha"]),
            ra["argmax"], rb["argmax"],
            ra["fill_ms"].as_f64().unwrap_or(f64::NAN), rb["fill_ms"].as_f64().unwrap_or(f64::NAN),
        );
        if !bad.is_empty() {
            let start = ra["start"].as_u64().unwrap_or(0);
            let mut ctx = format!(
                "state {case}/{stage} rows {}..{}: differing {bad:?}",
                ra["start"], ra["end"]
            );
            for f in &bad {
                ctx += &format!("\n    {f}: a={} b={}", ra[*f], rb[*f]);
            }
            if bad.contains(&"k_sha") {
                ctx += &format!("\n    {}", block_diff_context(ra, rb, "k_blocks", start));
            }
            if bad.contains(&"v_sha") {
                ctx += &format!("\n    {}", block_diff_context(ra, rb, "v_blocks", start));
            }
            failures.push(ctx);
        }
    }

    let (ga, gb) = (
        a["generation"].as_array().cloned().unwrap_or_default(),
        b["generation"].as_array().cloned().unwrap_or_default(),
    );
    if ga.len() != gb.len() || ga.len() != EXPECTED_GEN_RECORDS {
        failures.push(format!(
            "generation record counts: a={} b={}, expected {EXPECTED_GEN_RECORDS} (missing or extra cases)",
            ga.len(),
            gb.len()
        ));
    }
    const GEN_FIELDS: [&str; 9] = [
        "case",
        "mode",
        "prompt_len",
        "prefill_start",
        "tokens",
        "generated",
        "steps",
        "proposed",
        "accepted",
    ];
    for (ra, rb) in ga.iter().zip(&gb) {
        let (case, mode) = (
            ra["case"].as_str().unwrap_or("?"),
            ra["mode"].as_str().unwrap_or("?"),
        );
        let mut bad: Vec<&str> = GEN_FIELDS
            .iter()
            .copied()
            .filter(|f| ra[*f] != rb[*f])
            .collect();
        if ra["windows"] != rb["windows"] {
            bad.push("windows");
        }
        println!(
            "MTP_FILL_ROW kind=gen case={case} mode={mode} prompt_len={} prefill_start={} identical={} generated_a={} generated_b={} steps_a={} steps_b={} proposed_a={} proposed_b={} accepted_a={} accepted_b={} windows_a={} windows_b={} token_sha_a={} token_sha_b={}",
            ra["prompt_len"], ra["prefill_start"], bad.is_empty(),
            ra["generated"], rb["generated"], ra["steps"], rb["steps"], ra["proposed"], rb["proposed"],
            ra["accepted"], rb["accepted"], ra["windows"], rb["windows"],
            short(&ra["token_sha"]), short(&rb["token_sha"]),
        );
        if !bad.is_empty() {
            let mut ctx = format!("generation {case}/{mode}: differing {bad:?}");
            for f in bad.iter().filter(|f| **f != "tokens") {
                ctx += &format!("\n    {f}: a={} b={}", ra[*f], rb[*f]);
            }
            if bad.contains(&"tokens") {
                ctx += &format!("\n    {}", token_diff_context(&ra["tokens"], &rb["tokens"]));
            }
            failures.push(ctx);
        }
    }
    let identical = failures.is_empty();
    println!(
        "MTP_FILL_VERDICT identical={identical} failures={}",
        failures.len()
    );
    failures
}

fn read_artifact(env: &str) -> serde_json::Value {
    let path =
        std::env::var(env).unwrap_or_else(|_| panic!("{env} must name an arm artifact JSON"));
    serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}")))
        .unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// Compares two arm artifacts (no GPU, no model needed).
#[test]
#[ignore = "post-processing of two mtp_fill_oracle_arm artifacts: HIPFIRE_MTP_FILL_ORACLE_A / _B"]
fn mtp_fill_oracle_compare() {
    let (a, b) = (read_artifact(ARM_A_ENV), read_artifact(ARM_B_ENV));
    let failures = compare_artifacts(&a, &b);
    assert!(
        failures.is_empty(),
        "fill fast/off oracle mismatches:\n{}",
        failures.join("\n")
    );
}

/// One-command oracle: re-executes this test binary twice, each a fresh
/// process running `mtp_fill_oracle_arm` with `HIPFIRE_MTP_FILL_FAST=0` (a)
/// then `=1` (b) set on the CHILD command only (this process's environment is
/// never mutated), then compares the artifacts.
#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_MTP_BYTE_IDENTITY_MODEL + MTP sidecar; spawns two child processes"]
fn mtp_fill_oracle_spawn() {
    if std::env::var_os(MODEL_ENV).is_none() {
        eprintln!("skipping: {MODEL_ENV} not set");
        return;
    }
    let dir = std::env::var_os(DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("mtp_fill_oracle_{}", std::process::id()))
        });
    std::fs::create_dir_all(&dir).expect("create oracle dir");
    let exe = std::env::current_exe().expect("test binary path");
    let mut artifacts = Vec::new();
    for (label, value) in [("off", "0"), ("fast", "1")] {
        let out = dir.join(format!("{label}.json"));
        let _ = std::fs::remove_file(&out);
        let output = std::process::Command::new(&exe)
            .args([
                "--exact",
                ARM_TEST,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(FILL_ENV, value)
            .env(OUT_ENV, &out)
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap_or_else(|e| panic!("spawn {label} arm: {e}"));
        std::fs::write(dir.join(format!("{label}.stdout.log")), &output.stdout)
            .expect("write stdout log");
        std::fs::write(dir.join(format!("{label}.stderr.log")), &output.stderr)
            .expect("write stderr log");
        assert!(
            output.status.success(),
            "{label} arm failed ({}); logs in {}",
            output.status,
            dir.display()
        );
        assert!(
            out.exists(),
            "{label} arm wrote no artifact (is {MODEL_ENV} visible to the child?); logs in {}",
            dir.display()
        );
        artifacts.push(
            serde_json::from_slice::<serde_json::Value>(
                &std::fs::read(&out).expect("read artifact"),
            )
            .expect("parse artifact"),
        );
    }
    eprintln!("[oracle] artifacts in {}", dir.display());
    let failures = compare_artifacts(&artifacts[0], &artifacts[1]);
    assert!(
        failures.is_empty(),
        "fill fast/off oracle mismatches (artifacts in {}):\n{}",
        dir.display(),
        failures.join("\n")
    );
}
