// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MTP prompt-end payload state-equality hardware proof (#1198, 27B dense).
//!
//! ONE `#[ignore]`d test driving the real [`Qwen35MtpDrafter`] in-process on a
//! real HIP GPU + a real Qwen3.5/3.6 dense trunk/MTP-head pair. A cold exact
//! retry (the same rendered prompt re-sent after the previous turn committed)
//! must reuse its WHOLE prompt: zero prefill rows, with the last-row logits, the
//! speculator seed and the first token byte-identical to a fresh prefill.
//!
//! Per arm (greedy and fixed-seed sampled, eager and deferred-tail prefill):
//!
//! 1. Fresh cold prefill of an irregular-length prompt. With a deferred tail the
//!    drafter must advertise NO `prompt_end_checkpoint` while head fill is owed,
//!    and exactly `(len, prefix_fingerprint(prompt))` once it settles.
//! 2. Digest the FRESH state: first token, raw logits row bytes, DeltaNet bytes
//!    and stream metadata, retained target-KV prefix, MTP-head KV prefix,
//!    `prev_hidden` bytes and position, owed rows.
//! 3. Decode several windows (this rewrites target/head KV above the prompt end,
//!    the DeltaNet state and `prev_hidden`).
//! 4. Re-configure the request (fresh RNG) and EXACT-RETRY: `mtp_prefill` with
//!    `start_pos == prompt.len()`, empty fill, `cache_hit = true`. Digest again
//!    and require the digests byte-equal, the first token equal, no owed rows.
//! 5. Decode the same windows again: the committed stream must equal the fresh
//!    arm's (the continuation surface where a silent mismatch would show).
//!
//! Eager and deferred prefills must converge to the same fresh digest.
//!
//! Negative controls: a cold prefill of a different prompt moves the descriptor
//! and an exact retry of the old prompt then fails closed; `mtp_reset` drops the
//! payload (no descriptor, retry fails closed).
//!
//! Run explicitly:
//!
//! ```bash
//! HIP_VISIBLE_DEVICES=1 HIPFIRE_MTP_PROMPT_END_MODEL=~/.hipfire/models/<27b>.mq4 \
//!   cargo test --release -p hipfire-arch-qwen35 --test mtp_prompt_end_state_equality \
//!   -- --ignored --nocapture
//! ```
//!
//! `HIPFIRE_MTP_PROMPT_END_HEAD` names the MTP sidecar (default: the model path
//! with its extension replaced by `.mtp`); `HIPFIRE_MTP_PROMPT_END_TOKENS`
//! overrides the prompt length (default 1237, deliberately not chunk aligned).
//! Without the model variable the test reports a skip and returns before any GPU
//! initialisation (same convention as `mtp_cache_byte_identity`); with it set a
//! GPU initialisation failure fails the test.

#![allow(clippy::all)]

use hipfire_arch_qwen35::mtp_head::{self, Qwen35MtpHead};
use hipfire_arch_qwen35::mtp_speculator::Qwen35MtpDrafter;
use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
use hipfire_runtime::checkpoint_pool::prefix_fingerprint;
use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig};
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};

const MAX_N: usize = 4;
const CTX: usize = 2048;
const DEFAULT_PROMPT_TOKENS: usize = 1237;
const DECODE_WINDOWS: usize = 6;
const PROMPT_SEED: &str =
    "The harbor bell rang twice before dawn, and the gulls answered in kind. ";

fn model_path() -> Option<PathBuf> {
    std::env::var("HIPFIRE_MTP_PROMPT_END_MODEL")
        .ok()
        .map(PathBuf::from)
}

fn head_path(model: &Path) -> PathBuf {
    if let Ok(h) = std::env::var("HIPFIRE_MTP_PROMPT_END_HEAD") {
        return PathBuf::from(h);
    }
    model.with_extension("mtp")
}

fn prompt_tokens_target() -> usize {
    std::env::var("HIPFIRE_MTP_PROMPT_END_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_PROMPT_TOKENS)
}

/// FNV-1a/64 for the human-readable record; assertions compare full byte vecs.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Raw device bytes of one tensor: `min(logical bytes, allocation bytes)` (the
/// min matters for Q8/Q4 DeltaNet state, whose `GpuTensor` shape describes the
/// logical F32 view while the allocation holds the packed bytes).
fn tensor_bytes(gpu: &Gpu, t: &rdna_compute::GpuTensor) -> Vec<u8> {
    let n = t.byte_size().min(t.buf.size());
    let mut out = vec![0u8; n];
    gpu.hip
        .memcpy_dtoh(&mut out, &t.buf)
        .expect("dtoh tensor bytes");
    out
}

/// First `n` device bytes of one tensor (retained-prefix KV rows).
fn tensor_prefix_bytes(gpu: &Gpu, t: &rdna_compute::GpuTensor, n: usize) -> Vec<u8> {
    assert!(
        n <= t.buf.size(),
        "prefix {n} exceeds allocation {}",
        t.buf.size()
    );
    let mut out = vec![0u8; n];
    gpu.hip
        .memcpy_dtoh(&mut out, &t.buf)
        .expect("dtoh tensor prefix");
    out
}

fn load_session(
    gpu: &mut Gpu,
    model: &Path,
    head: &Path,
    max_seq: usize,
) -> (ModelSlot, Qwen35MtpHead) {
    let mut cfg = ModelSlotConfig::default();
    cfg.max_seq = max_seq;
    let slot = ModelSlot::load(gpu, model, "prompt-end", cfg).expect("load trunk");
    let mtp = mtp_head::load_mtp_head(head, gpu, max_seq).expect("load mtp head");
    assert_eq!(
        mtp.config.n_embd, slot.config.dim,
        "trunk/head dim mismatch"
    );
    assert_eq!(
        mtp.config.vocab_size, slot.config.vocab_size,
        "trunk/head vocab mismatch"
    );
    (slot, mtp)
}

fn free_session(gpu: &mut Gpu, slot: ModelSlot, drafter: Qwen35MtpDrafter) {
    // Mirror the daemon's unload_model: invalidate the weight-pointer-keyed
    // caches and captured graphs BEFORE freeing the weights they point into.
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    // `mtp_free` releases the drafter's MTP state, prompt-end payload and head.
    Box::new(drafter).mtp_free(gpu);
    slot.kv_cache.free_gpu(gpu).expect("free kv");
    slot.dn_state.free_gpu(gpu);
    slot.scratch.free_gpu(gpu).expect("free scratch");
    slot.weights.free_gpu(gpu);
    gpu.drain_pool();
}

/// Q8_0 KV row stride in bytes: blocks of 32 int8 + 2 B scale per head row.
fn kv_row_bytes(n_kv_heads: usize, head_dim: usize) -> usize {
    assert!(head_dim % 32 == 0, "q8 kv head_dim {head_dim} not /32");
    n_kv_heads * (head_dim / 32) * 34
}

/// Retained-prefix trunk KV bytes: per FullAttention layer, K rows
/// `[0..positions)` then V rows `[0..positions)`.
fn kv_prefix_bytes(gpu: &Gpu, slot: &ModelSlot, positions: usize) -> Vec<u8> {
    let kv = &slot.kv_cache;
    let row = kv_row_bytes(kv.n_kv_heads, kv.head_dim);
    let mut out = Vec::new();
    for layer in 0..kv.k_gpu.len() {
        for buf in [&kv.k_gpu[layer], &kv.v_gpu[layer]] {
            if buf.buf.size() < row {
                continue; // non-KV-layer placeholder
            }
            out.extend_from_slice(&tensor_prefix_bytes(gpu, buf, positions * row));
        }
    }
    out
}

/// Full target-DeltaNet live bytes: S matrices + scales + conv ring + EF
/// residual (when active), one entry per family.
fn dn_bytes(gpu: &Gpu, slot: &ModelSlot) -> Vec<Vec<u8>> {
    let mut fams: Vec<Vec<u8>> = Vec::with_capacity(4);
    for set in [
        &slot.dn_state.s_matrices,
        &slot.dn_state.s_scales,
        &slot.dn_state.conv_states,
        &slot.dn_state.s_ef_residual,
    ] {
        let mut fam = Vec::new();
        for t in set {
            fam.extend_from_slice(&tensor_bytes(gpu, t));
        }
        fams.push(fam);
    }
    fams
}

fn chatml_wrap(tok: &hipfire_runtime::tokenizer::Tokenizer, body: &str) -> Vec<u32> {
    let im_start = tok.encode("<|im_start|>");
    let im_end = tok.encode("<|im_end|>");
    let user = tok.encode("user");
    let asst = tok.encode("assistant");
    let nl = tok.encode("\n");
    assert!(im_start.len() == 1, "tokenizer has no <|im_start|> special");
    let q = tok.encode(body);
    let mut t = Vec::new();
    t.extend_from_slice(&im_start);
    t.extend_from_slice(&user);
    t.extend_from_slice(&nl);
    t.extend_from_slice(&q);
    t.extend_from_slice(&im_end);
    t.extend_from_slice(&nl);
    t.extend_from_slice(&im_start);
    t.extend_from_slice(&asst);
    t.extend_from_slice(&nl);
    t
}

/// Everything a prompt-end restore must reproduce, as raw bytes.
#[derive(PartialEq)]
struct SeedState {
    first_token: u32,
    logits: Vec<u8>,
    dn: Vec<Vec<u8>>,
    stream_meta: String,
    trunk_kv_prefix: Vec<u8>,
    head_kv_prefix: Vec<u8>,
    prev_hidden: Vec<u8>,
    prev_hidden_pos: Option<usize>,
    owed_rows: usize,
}

fn digest(
    gpu: &Gpu,
    slot: &ModelSlot,
    drafter: &Qwen35MtpDrafter,
    len: usize,
    first_token: u32,
) -> SeedState {
    let st = drafter
        .mtp_live_state()
        .expect("mtp state allocated after prefill");
    let mtp_row = kv_row_bytes(st.mtp_kv.n_head_kv, st.mtp_kv.head_dim);
    let mut head_kv_prefix = Vec::new();
    for buf in st
        .mtp_kv
        .inner
        .k_gpu
        .iter()
        .chain(st.mtp_kv.inner.v_gpu.iter())
    {
        if buf.buf.size() < mtp_row {
            continue;
        }
        head_kv_prefix.extend_from_slice(&tensor_prefix_bytes(gpu, buf, len * mtp_row));
    }
    SeedState {
        first_token,
        logits: tensor_bytes(gpu, &slot.scratch.logits),
        dn: dn_bytes(gpu, slot),
        // `owner_generation` names the live stream owner (bumped by every
        // reset), not state: two fresh prefills differ in it.
        stream_meta: {
            let mut meta = slot.dn_state.stream.image().meta.clone();
            for m in meta.iter_mut() {
                m.owner_generation = 0;
            }
            format!("{meta:?}")
        },
        trunk_kv_prefix: kv_prefix_bytes(gpu, slot, len),
        head_kv_prefix,
        prev_hidden: tensor_bytes(gpu, &st.prev_hidden),
        prev_hidden_pos: st.prev_hidden_pos,
        owed_rows: drafter.owed_fill_rows(),
    }
}

fn report(tag: &str, s: &SeedState) {
    eprintln!(
        "[prompt-end {tag}] first_token={} logits fnv={:016x} dn fnv={:016x} meta fnv={:016x} \
         trunk_kv bytes={} fnv={:016x} head_kv bytes={} fnv={:016x} prev_hidden fnv={:016x} pos={:?} owed={}",
        s.first_token,
        fnv1a(&s.logits),
        fnv1a(&s.dn.concat()),
        fnv1a(s.stream_meta.as_bytes()),
        s.trunk_kv_prefix.len(),
        fnv1a(&s.trunk_kv_prefix),
        s.head_kv_prefix.len(),
        fnv1a(&s.head_kv_prefix),
        fnv1a(&s.prev_hidden),
        s.prev_hidden_pos,
        s.owed_rows,
    );
}

/// Field-by-field comparison so a failure names the diverging surface.
fn assert_seed_equal(tag: &str, fresh: &SeedState, other: &SeedState) {
    assert_eq!(fresh.first_token, other.first_token, "{tag}: first token");
    assert_eq!(fresh.logits, other.logits, "{tag}: last-row logits bytes");
    assert_eq!(fresh.dn.len(), other.dn.len(), "{tag}: dn families");
    for (i, (a, b)) in fresh.dn.iter().zip(other.dn.iter()).enumerate() {
        assert_eq!(a, b, "{tag}: dn family {i} bytes");
    }
    assert_eq!(
        fresh.stream_meta, other.stream_meta,
        "{tag}: dn stream meta"
    );
    assert_eq!(
        fresh.trunk_kv_prefix, other.trunk_kv_prefix,
        "{tag}: target KV prefix bytes"
    );
    assert_eq!(
        fresh.head_kv_prefix, other.head_kv_prefix,
        "{tag}: MTP head KV prefix bytes"
    );
    assert_eq!(
        fresh.prev_hidden, other.prev_hidden,
        "{tag}: prev_hidden bytes"
    );
    assert_eq!(
        fresh.prev_hidden_pos, other.prev_hidden_pos,
        "{tag}: prev_hidden position"
    );
    assert_eq!(
        fresh.owed_rows, other.owed_rows,
        "{tag}: owed head-fill rows"
    );
}

/// `DECODE_WINDOWS` greedy/sampled windows from the first token; returns the
/// committed stream after the prompt (first token included).
fn decode(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    drafter: &mut Qwen35MtpDrafter,
    prompt: &[u32],
    first_token: u32,
) -> Vec<u32> {
    let eos = slot.config.eos_token;
    let mut history: Vec<u32> = prompt.to_vec();
    history.push(first_token);
    let mut pos = prompt.len();
    let mut seed = first_token;
    for _ in 0..DECODE_WINDOWS {
        let w = drafter
            .mtp_step(gpu, slot, pos, seed, &history, MAX_N, eos, None)
            .expect("decode window");
        history.extend_from_slice(&w.committed);
        pos += w.committed.len();
        seed = *history.last().unwrap();
    }
    history[prompt.len()..].to_vec()
}

struct Arm {
    name: &'static str,
    cfg: SpecRequestConfig,
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_MTP_PROMPT_END_MODEL (qwen3.5/3.6 dense trunk) + MTP sidecar"]
fn mtp_prompt_end_exact_retry_state_equality() {
    let Some(model) = model_path() else {
        eprintln!("skipping: HIPFIRE_MTP_PROMPT_END_MODEL not set");
        return;
    };
    let head_p = head_path(&model);
    eprintln!("[prompt-end] model: {}", model.display());
    eprintln!("[prompt-end] head:  {}", head_p.display());

    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!("[prompt-end] gpu: {}", gpu.arch);
    let no_abort = || false;

    let (mut slot, head) = load_session(&mut gpu, &model, &head_p, CTX);
    let tokenizer = slot.load_tokenizer().expect("tokenizer");
    let seed_tokens = chatml_wrap(&tokenizer, PROMPT_SEED);
    let target_len = prompt_tokens_target();
    assert!(
        target_len + 256 < CTX,
        "prompt {target_len} leaves no decode room in {CTX}"
    );
    let mut prompt: Vec<u32> = Vec::with_capacity(target_len + seed_tokens.len());
    while prompt.len() < target_len {
        prompt.extend_from_slice(&seed_tokens);
    }
    prompt.truncate(target_len);
    let len = prompt.len();
    let fingerprint = prefix_fingerprint(&prompt);
    eprintln!("[prompt-end] prompt tokens: {len}");

    let mut drafter = Qwen35MtpDrafter::new(head, MAX_N, CTX);
    let arms = [
        Arm {
            name: "greedy",
            cfg: SpecRequestConfig::default(),
        },
        Arm {
            name: "sampled",
            cfg: SpecRequestConfig {
                temp: 0.8,
                top_p: 0.95,
                top_k: Some(20),
                rng_seed: 0x5EED_1198,
                ..SpecRequestConfig::default()
            },
        },
    ];

    for arm in &arms {
        let mut fresh_by_defer: Vec<SeedState> = Vec::new();
        for defer in [false, true] {
            let tag = format!("{} defer={defer}", arm.name);

            // ── 1. Fresh cold prefill ───────────────────────────────────
            drafter.configure_request(arm.cfg);
            drafter.mtp_defer_prefill_tail(defer);
            let t_fresh = std::time::Instant::now();
            let first_fresh = drafter
                .mtp_prefill(&mut gpu, &mut slot, &prompt, &prompt, 0, false, &no_abort)
                .and_then(|o| o.into_first_token())
                .expect("fresh prefill");
            let fresh_ms = t_fresh.elapsed().as_secs_f64() * 1e3;
            if drafter.owed_fill_rows() > 0 {
                assert!(defer, "{tag}: eager prefill left head fill owed");
                assert_eq!(
                    drafter.mtp_prompt_end_checkpoint(),
                    None,
                    "{tag}: a prompt-end payload was advertised while head fill is owed"
                );
            }
            drafter
                .mtp_finish_prefill(&mut gpu, &mut slot)
                .expect("settle owed fill");
            assert_eq!(drafter.owed_fill_rows(), 0, "{tag}: owed rows after settle");
            assert_eq!(
                drafter.mtp_prompt_end_checkpoint(),
                Some((len, fingerprint)),
                "{tag}: settled prefill must advertise the whole-prompt descriptor"
            );

            // ── 2. Fresh digest ─────────────────────────────────────────
            let fresh = digest(&gpu, &slot, &drafter, len, first_fresh);
            report(&format!("{tag} fresh"), &fresh);

            // ── 3. Decode (rewrites state above the prompt end) ─────────
            let decoded_fresh = decode(&mut gpu, &mut slot, &mut drafter, &prompt, first_fresh);
            assert_eq!(
                drafter.mtp_prompt_end_checkpoint(),
                Some((len, fingerprint)),
                "{tag}: decode must not invalidate the prompt-end payload"
            );

            // ── 4. Exact retry: whole prompt reused, zero prefill rows ──
            drafter.configure_request(arm.cfg);
            drafter.mtp_defer_prefill_tail(defer);
            let t_retry = std::time::Instant::now();
            let first_retry = drafter
                .mtp_prefill(&mut gpu, &mut slot, &prompt, &[], len, true, &no_abort)
                .and_then(|o| o.into_first_token())
                .expect("exact retry prefill");
            let retry_ms = t_retry.elapsed().as_secs_f64() * 1e3;
            drafter
                .mtp_finish_prefill(&mut gpu, &mut slot)
                .expect("settle after retry");
            eprintln!(
                "[prompt-end {tag}] fresh_prefill_ms={fresh_ms:.1} exact_retry_ms={retry_ms:.2}"
            );
            let retry = digest(&gpu, &slot, &drafter, len, first_retry);
            report(&format!("{tag} retry"), &retry);
            assert_seed_equal(&format!("{tag} fresh-vs-retry"), &fresh, &retry);
            assert_eq!(
                drafter.mtp_prompt_end_checkpoint(),
                Some((len, fingerprint)),
                "{tag}: the payload must survive its own restore"
            );

            // ── 5. Continuation from the restored state ─────────────────
            let decoded_retry = decode(&mut gpu, &mut slot, &mut drafter, &prompt, first_retry);
            assert_eq!(
                decoded_fresh, decoded_retry,
                "{tag}: continuation after the exact retry diverges from the fresh stream \
                 (fresh={decoded_fresh:?} retry={decoded_retry:?})"
            );
            eprintln!(
                "[prompt-end {tag}] continuation identical: {:?}",
                tokenizer.decode(&decoded_retry)
            );
            fresh_by_defer.push(fresh);
        }
        // Eager and deferred prefills converge to the same sealed seed.
        assert_seed_equal(
            &format!("{} eager-vs-deferred fresh", arm.name),
            &fresh_by_defer[0],
            &fresh_by_defer[1],
        );
    }

    // ── Negative controls ───────────────────────────────────────────────
    // A cold prefill of a different (shorter) prompt moves the descriptor; an
    // exact retry of the old prompt then fails closed instead of restoring a
    // seed that does not describe it.
    {
        drafter.configure_request(SpecRequestConfig::default());
        let other = &prompt[..len - 37];
        let other_fp = prefix_fingerprint(other);
        drafter
            .mtp_prefill(&mut gpu, &mut slot, other, other, 0, false, &no_abort)
            .and_then(|o| o.into_first_token())
            .expect("other-prompt prefill");
        assert_eq!(
            drafter.mtp_prompt_end_checkpoint(),
            Some((other.len(), other_fp)),
            "descriptor must follow the newest whole-prompt prefill"
        );
        assert_ne!(
            drafter.mtp_prompt_end_checkpoint(),
            Some((len, fingerprint)),
            "stale descriptor still advertised"
        );
        assert!(
            drafter
                .mtp_prefill(&mut gpu, &mut slot, &prompt, &[], len, true, &no_abort)
                .is_err(),
            "exact retry of a superseded prompt must fail closed"
        );
    }
    // A reset drops the payload entirely.
    {
        drafter.mtp_reset(&mut gpu).expect("reset");
        assert_eq!(
            drafter.mtp_prompt_end_checkpoint(),
            None,
            "reset must drop the prompt-end payload"
        );
        assert!(
            drafter
                .mtp_prefill(&mut gpu, &mut slot, &prompt, &[], len, true, &no_abort)
                .is_err(),
            "exact retry after reset must fail closed"
        );
    }

    free_session(&mut gpu, slot, drafter);
    eprintln!(
        "[prompt-end] PASS: exact-retry state equality proven (greedy+sampled, eager+deferred)"
    );
}
