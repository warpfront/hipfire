// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MTP recurrent-state byte-identity hardware proofs (#749 + #751).
//!
//! ONE `#[ignore]`d test driving the real [`Qwen35MtpDrafter`] paths
//! in-process on a real HIP GPU + real Qwen3.5-MoE trunk/MTP-head pair:
//!
//! - Path R (strict-prefix terminal repair, #749): run an MTP session to a
//!   multi-token window, repair a strict-prefix terminal from `trunk_snap`,
//!   and require the resident target DeltaNet bytes, retained-prefix target
//!   KV bytes, MTP-head KV bytes, MTP `prev_hidden` bytes, and resident token
//!   ids to equal a cold render of the identical token prefix in a fresh
//!   session. A greedy continuation window must then commit identically on
//!   both arms (the "later cached turn" surface where a silent state mismatch
//!   would first surface).
//! - Path C (checkpoint-ring rewind, #751): prefill a long prompt with the
//!   ring on, rewind to an advertised checkpoint, and require the restored
//!   DeltaNet bytes (+ retained KV prefix) to equal a cold prefill of
//!   `[0..P)`; then re-prefill the divergent suffix on the rewound arm and
//!   require FULL state (DeltaNet + trunk KV + MTP KV + `prev_hidden` + ids)
//!   to equal a single cold prefill of the same tokens, plus an identical
//!   greedy continuation window.
//!
//! Cursor convention (mirrors `generate_spec` in `hipfire-generate/src/qwen.rs`
//! and the `SpecPrefixRealignPlan` contract): GPU state covers `[0..pos)`;
//! the host-baked stream runs one ahead with the last token pending
//! (unprocessed) AT `pos`. `step(pos, seed)` WRITES `seed` at slot `pos`.
//! `MtpWindow.committed` carries fresh tokens only (no seed re-echo); the
//! repair helper rebuilds `[seed, consumed[..keep-1]]` at the window start.
//!
//! Run explicitly (GPU 1 per the v0.3.1 lane discipline):
//!
//! ```bash
//! HIP_VISIBLE_DEVICES=1 flock /tmp/hipfire-gpu1.lock cargo test --release \
//!   -p hipfire-arch-qwen35 --test mtp_cache_byte_identity -- --ignored --nocapture
//! ```
//!
//! with `HIPFIRE_MTP_BYTE_IDENTITY_MODEL` naming the trunk
//! (`~/.hipfire/models/ornith-1.5-35b-a3b.mq4r`) and optionally
//! `HIPFIRE_MTP_BYTE_IDENTITY_HEAD` naming the MTP sidecar (defaults to the
//! model path with its extension replaced by `.mtp`). Without the model var
//! the test reports a skip and returns (same convention as `pp_parity`).

#![allow(clippy::all)]

use hipfire_arch_qwen35::mtp_head::{self, Qwen35MtpHead};
use hipfire_arch_qwen35::mtp_speculator::Qwen35MtpDrafter;
use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
use hipfire_runtime::spec::{terminal_prefix_replay, MtpDrafter, SpecRequestConfig};
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};

const CTX_PATH1: usize = 1024;
const CTX_PATH2: usize = 2048;
const MAX_N: usize = 4;
const PROMPT_PATH1: &str = "Explain why the sky is blue in one short paragraph.";
const PROMPT_PATH2_SEED: &str =
    "The harbor bell rang twice before dawn, and the gulls answered in kind. ";
const PATH2_MIN_TOKENS: usize = 700;

fn model_path() -> Option<PathBuf> {
    std::env::var("HIPFIRE_MTP_BYTE_IDENTITY_MODEL")
        .ok()
        .map(PathBuf::from)
}

fn head_path(model: &Path) -> PathBuf {
    if let Ok(h) = std::env::var("HIPFIRE_MTP_BYTE_IDENTITY_HEAD") {
        return PathBuf::from(h);
    }
    // Default: sibling sidecar — same stem, `.mtp` extension.
    model.with_extension("mtp")
}

/// FNV-1a/64 for compact display. The assertions compare full byte vecs;
/// hashes are only the human-readable record.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Raw device bytes of one tensor: the live region is
/// `min(logical bytes, allocation bytes)`. The min matters for Q8/Q4 DeltaNet
/// state, whose `GpuTensor` shape/dtype describe the logical F32 view while
/// the allocation holds the packed bytes.
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
    let slot = ModelSlot::load(gpu, model, "byte-identity", cfg).expect("load trunk");
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
    // caches (mmq_screen, fp16 shadows own device tensors) and captured
    // graphs BEFORE freeing the weights they point into; otherwise every
    // session leaks device memory and the third load OOMs.
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    // `mtp_free` releases the drafter's MTP state AND its owned head.
    Box::new(drafter).mtp_free(gpu);
    slot.kv_cache.free_gpu(gpu).expect("free kv");
    slot.dn_state.free_gpu(gpu);
    slot.scratch.free_gpu(gpu).expect("free scratch");
    slot.weights.free_gpu(gpu);
    gpu.drain_pool();
}

fn greedy_drafter(head: Qwen35MtpHead, ctx: usize) -> Qwen35MtpDrafter {
    let mut d = Qwen35MtpDrafter::new(head, MAX_N, ctx);
    d.configure_request(SpecRequestConfig::default());
    d
}

/// Full target-DeltaNet live bytes: S matrices + scales + conv ring +
/// EF residual (when active), one entry per family.
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

/// Q8_0 KV row stride in bytes: blocks of 32 int8 + 2 B scale per head row.
fn kv_row_bytes(n_kv_heads: usize, head_dim: usize) -> usize {
    assert!(head_dim % 32 == 0, "q8 kv head_dim {head_dim} not /32");
    n_kv_heads * (head_dim / 32) * 34
}

/// Retained-prefix trunk KV bytes: per FullAttention layer (placeholder
/// layers for DeltaNet-only levels are smaller than one row and skipped),
/// K rows `[0..positions)` then V rows `[0..positions)`.
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

/// Resident MTP-head state bytes: `prev_hidden` (full) + single-layer Q8 KV
/// prefix rows `[0..positions)`.
fn mtp_bytes(gpu: &Gpu, drafter: &Qwen35MtpDrafter, positions: usize) -> (Vec<u8>, Vec<u8>) {
    let st = drafter
        .mtp_live_state()
        .expect("mtp state allocated after prefill");
    let prev = tensor_bytes(gpu, &st.prev_hidden);
    let kv = &st.mtp_kv.inner;
    let row = kv_row_bytes(st.mtp_kv.n_head_kv, st.mtp_kv.head_dim);
    let mut kvb = Vec::new();
    for buf in kv.k_gpu.iter().chain(kv.v_gpu.iter()) {
        if buf.buf.size() < row {
            continue;
        }
        kvb.extend_from_slice(&tensor_prefix_bytes(gpu, buf, positions * row));
    }
    (prev, kvb)
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
/// First differing row index between two packed buffers, or `None` when the
/// compared extents are identical. Row granularity keeps the report in token
/// positions for KV families.
fn first_diff_row(a: &[u8], b: &[u8], row_bytes: usize) -> Option<usize> {
    let n = a.len().min(b.len()) / row_bytes;
    for r in 0..n {
        if a[r * row_bytes..(r + 1) * row_bytes] != b[r * row_bytes..(r + 1) * row_bytes] {
            return Some(r);
        }
    }
    if a.len() != b.len() {
        return Some(n);
    }
    None
}
/// First differing DeltaNet layer index within one concatenated family, or
/// `None` when identical. Layers are equal-sized contiguous splits.
fn first_diff_layer(a: &[u8], b: &[u8], n_layers: usize) -> Option<usize> {
    if n_layers == 0 || a.len() != b.len() || a.len() % n_layers != 0 {
        return None;
    }
    let per = a.len() / n_layers;
    for l in 0..n_layers {
        if a[l * per..(l + 1) * per] != b[l * per..(l + 1) * per] {
            return Some(l);
        }
    }
    None
}

fn report_fams(tag: &str, arm: &str, dn: &[Vec<u8>], kv: &[u8], mtp_kv: &[u8], prev: &[u8]) {
    let names = ["dn_s", "dn_scales", "dn_conv", "dn_ef"];
    for (n, fam) in names.iter().zip(dn.iter()) {
        eprintln!(
            "[byte-id {tag} {arm}] {n}: bytes={} fnv={:016x}",
            fam.len(),
            fnv1a(fam)
        );
    }
    eprintln!(
        "[byte-id {tag} {arm}] trunk_kv_prefix: bytes={} fnv={:016x}",
        kv.len(),
        fnv1a(kv)
    );
    eprintln!(
        "[byte-id {tag} {arm}] mtp_kv_prefix: bytes={} fnv={:016x}",
        mtp_kv.len(),
        fnv1a(mtp_kv)
    );
    eprintln!(
        "[byte-id {tag} {arm}] mtp_prev_hidden: bytes={} fnv={:016x}",
        prev.len(),
        fnv1a(prev)
    );
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_MTP_BYTE_IDENTITY_MODEL (qwen3.5-MoE trunk) + MTP sidecar"]
fn mtp_recurrent_state_byte_identity() {
    let Some(model) = model_path() else {
        eprintln!("skipping: HIPFIRE_MTP_BYTE_IDENTITY_MODEL not set");
        return;
    };
    let head_p = head_path(&model);
    eprintln!("[byte-id] model: {}", model.display());
    eprintln!("[byte-id] head:  {}", head_p.display());
    // Small checkpoint interval so the rewind path exercises the real ring
    // on a short prompt. Read once per drafter construction.
    std::env::set_var("HIPFIRE_CACHE_CKPT_INTERVAL", "256");

    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!("[byte-id] gpu: {}", gpu.arch);
    let no_abort = || false;

    // ── Path R: strict-prefix terminal repair (#749) ────────────────
    // Daemon-faithful cursors: `pos` = next write slot (GPU covers
    // `[0..pos)`), `seed` = pending token AT `pos` (baked in host history,
    // unprocessed on device). `MtpWindow.committed` is fresh tokens only.
    {
        let (mut slot_a, head_a) = load_session(&mut gpu, &model, &head_p, CTX_PATH1);
        let tokenizer = slot_a.load_tokenizer().expect("tokenizer");
        let prompt = chatml_wrap(&tokenizer, PROMPT_PATH1);
        eprintln!("[byte-id repair] prompt tokens: {}", prompt.len());

        // Arm A (repair): prefill, step windows, repair a strict prefix.
        let mut drafter_a = greedy_drafter(head_a, CTX_PATH1);
        let mut seed = drafter_a
            .mtp_prefill(&mut gpu, &mut slot_a, &prompt, &prompt, 0, false, &no_abort)
            .expect("arm A prefill");
        let eos_a = slot_a.config.eos_token;
        // Host history: prompt + every baked token (last = pending seed).
        let mut history: Vec<u32> = prompt.clone();
        history.push(seed);
        let mut pos = prompt.len();
        // Find a multi-token window to terminate mid-window.
        let (wpos, wseed, wcommitted) = loop {
            let mut found = None;
            for _ in 0..8 {
                let w = drafter_a
                    .mtp_step(
                        &mut gpu,
                        &mut slot_a,
                        pos,
                        seed,
                        &history,
                        MAX_N,
                        eos_a,
                        None,
                    )
                    .expect("arm A step");
                history.extend_from_slice(&w.committed);
                pos += w.committed.len();
                seed = *history.last().unwrap();
                if w.committed.len() >= 2 {
                    found = Some((
                        pos - w.committed.len(),
                        history[history.len() - w.committed.len() - 1],
                        w.committed.clone(),
                    ));
                    break;
                }
            }
            let f = found.expect("no multi-token window in 8 steps");
            break f;
        };
        // Simulate a terminal that consumed a strict prefix of the window:
        // the last committed token stays pending (daemon terminal flush).
        let keep = wcommitted.len() - 1;
        let consumed = wcommitted[..keep].to_vec();
        let repaired = drafter_a
            .mtp_repair_terminal_prefix(&mut gpu, &mut slot_a, wpos, wseed, &consumed)
            .expect("repair must not error on its own window");
        assert!(repaired, "repair declined its own window");
        let replay = terminal_prefix_replay(wseed, &consumed);
        let p1 = wpos + replay.len();
        eprintln!(
            "[byte-id repair] window_start={wpos} committed={} keep={keep} resumed_positions={p1}",
            wcommitted.len()
        );
        // Resident processed prefix: everything before the window plus the
        // replay ([seed @ wpos, consumed prefix after]). The last consumed
        // token stays pending at p1.
        let mut resident: Vec<u32> = history[..wpos].to_vec();
        resident.extend_from_slice(&replay);
        assert_eq!(resident.len(), p1, "resident length must equal p1");
        assert_eq!(
            resident[wpos], wseed,
            "replay must re-commit the seed at wpos"
        );
        let pending = *consumed.last().expect("keep >= 1");
        let dn_a = dn_bytes(&gpu, &slot_a);
        let kv_a = kv_prefix_bytes(&gpu, &slot_a, p1);
        let (prev_a, mtp_kv_a) = mtp_bytes(&gpu, &drafter_a, p1);
        report_fams("repair", "A(repaired)", &dn_a, &kv_a, &mtp_kv_a, &prev_a);

        // Continuation on the repaired arm (the "later cached turn").
        let wcont_a = drafter_a
            .mtp_step(
                &mut gpu,
                &mut slot_a,
                p1,
                pending,
                &resident,
                MAX_N,
                eos_a,
                None,
            )
            .expect("arm A continuation step");
        let cont_text_a = tokenizer.decode(&wcont_a.committed);
        eprintln!("[byte-id repair] A continuation: {cont_text_a:?}");
        free_session(&mut gpu, slot_a, drafter_a);

        // Arm B (cold): single prefill of the identical resident prefix.
        let (mut slot_b, head_b) = load_session(&mut gpu, &model, &head_p, CTX_PATH1);
        let mut drafter_b = greedy_drafter(head_b, CTX_PATH1);
        let seed_b = drafter_b
            .mtp_prefill(
                &mut gpu,
                &mut slot_b,
                &resident,
                &resident,
                0,
                false,
                &no_abort,
            )
            .expect("arm B cold prefill");
        assert_eq!(
            seed_b, pending,
            "cold prefill must predict the same pending token"
        );
        let dn_b = dn_bytes(&gpu, &slot_b);
        let kv_b = kv_prefix_bytes(&gpu, &slot_b, p1);
        let (prev_b, mtp_kv_b) = mtp_bytes(&gpu, &drafter_b, p1);
        report_fams("repair", "B(cold)", &dn_b, &kv_b, &mtp_kv_b, &prev_b);
        // NOTE: the stepped-no-repair control (arm D) lives in the separate
        // `mtp_step_oracle` test binary: a third full 35B session does not fit
        // sequentially in one process (residue OOMs the third load), while two
        // sessions fit. Each binary is its own process; cross-process
        // determinism is proven (identical fnv across runs), so fnv equality
        // across the two logs decides the oracle.
        // First-divergence diagnostics for A-vs-B (cold vs repaired).
        let kv_row = kv_row_bytes(slot_b.kv_cache.n_kv_heads, slot_b.kv_cache.head_dim);
        eprintln!(
            "[byte-id repair] A-vs-B trunk_kv_prefix first_diff_row={:?} (prompt_len=20, wpos={wpos})",
            first_diff_row(&kv_a, &kv_b, kv_row)
        );
        let mtp_row = {
            let st = drafter_b.mtp_live_state().expect("mtp state");
            kv_row_bytes(st.mtp_kv.n_head_kv, st.mtp_kv.head_dim)
        };
        eprintln!(
            "[byte-id repair] A-vs-B mtp_kv_prefix first_diff_row={:?}",
            first_diff_row(&mtp_kv_a, &mtp_kv_b, mtp_row)
        );
        let n_dn = slot_b.dn_state.s_matrices.len();
        eprintln!(
            "[byte-id repair] A-vs-B dn_s first_diff_layer={:?} (of {n_dn} layers)",
            first_diff_layer(&dn_a[0], &dn_b[0], n_dn)
        );

        for (i, (a, b)) in dn_a.iter().zip(dn_b.iter()).enumerate() {
            assert_eq!(a.len(), b.len(), "repair dn family {i} length");
            assert_eq!(a, b, "repair dn family {i} bytes differ (cold vs repaired)");
        }
        assert_eq!(kv_a, kv_b, "repair trunk-KV retained-prefix bytes differ");
        assert_eq!(
            mtp_kv_a, mtp_kv_b,
            "repair MTP-KV retained-prefix bytes differ"
        );
        assert_eq!(prev_a, prev_b, "repair MTP prev_hidden bytes differ");

        let wcont_b = drafter_b
            .mtp_step(
                &mut gpu,
                &mut slot_b,
                p1,
                pending,
                &resident,
                MAX_N,
                eos_a,
                None,
            )
            .expect("arm B continuation step");
        assert_eq!(
            wcont_a.committed, wcont_b.committed,
            "post-repair continuation diverges: A={:?} B={:?}",
            wcont_a.committed, wcont_b.committed
        );
        eprintln!(
            "[byte-id repair] continuation identical: {:?} ({:?})",
            wcont_b.committed,
            tokenizer.decode(&wcont_b.committed)
        );
        free_session(&mut gpu, slot_b, drafter_b);
    }

    // ── Path C: checkpoint-ring rewind (#751) ───────────────────────
    {
        let (mut slot_a, head_a) = load_session(&mut gpu, &model, &head_p, CTX_PATH2);
        let tokenizer = slot_a.load_tokenizer().expect("tokenizer");
        let seed_tokens = chatml_wrap(&tokenizer, PROMPT_PATH2_SEED);
        let mut long: Vec<u32> = Vec::with_capacity(PATH2_MIN_TOKENS + 64);
        while long.len() < PATH2_MIN_TOKENS {
            long.extend_from_slice(&seed_tokens);
        }
        long.truncate(PATH2_MIN_TOKENS);
        eprintln!("[byte-id rewind] prompt tokens: {}", long.len());

        // Arm A: prefill all, rewind to a mid checkpoint.
        let mut drafter_a = greedy_drafter(head_a, CTX_PATH2);
        drafter_a
            .mtp_prefill(&mut gpu, &mut slot_a, &long, &long, 0, false, &no_abort)
            .expect("arm A long prefill");
        let positions = drafter_a.mtp_checkpoint_positions();
        eprintln!("[byte-id rewind] checkpoints: {positions:?}");
        assert!(
            positions.len() >= 2,
            "need ≥2 checkpoints for a mid rewind, got {positions:?}"
        );
        let p = positions[positions.len() / 2];
        let back = drafter_a
            .mtp_rewind_to(&mut gpu, &mut slot_a, p)
            .expect("rewind to advertised checkpoint");
        assert_eq!(back, p, "rewind must land exactly on {p}");
        // Pure-restore check: DeltaNet + retained KV prefix vs cold [0..P).
        let dn_rw = dn_bytes(&gpu, &slot_a);
        let kv_rw = kv_prefix_bytes(&gpu, &slot_a, p);
        report_fams("rewind-restore", "A(rewound)", &dn_rw, &kv_rw, &[], &[]);

        // Resume protocol: re-prefill the divergent suffix, then compare
        // FULL state against one cold prefill of the same tokens.
        let suffix = long[p..].to_vec();
        let seed_a = drafter_a
            .mtp_prefill(&mut gpu, &mut slot_a, &long, &suffix, p, true, &no_abort)
            .expect("arm A suffix re-prefill");
        let dn_a = dn_bytes(&gpu, &slot_a);
        let kv_a = kv_prefix_bytes(&gpu, &slot_a, long.len());
        let (prev_a, mtp_kv_a) = mtp_bytes(&gpu, &drafter_a, long.len());
        report_fams("rewind", "A(resumed)", &dn_a, &kv_a, &mtp_kv_a, &prev_a);
        let eos_a = slot_a.config.eos_token;
        let wcont_a = drafter_a
            .mtp_step(
                &mut gpu,
                &mut slot_a,
                long.len(),
                seed_a,
                &long,
                MAX_N,
                eos_a,
                None,
            )
            .expect("arm A post-resume step");
        free_session(&mut gpu, slot_a, drafter_a);

        // Arm B (cold): single prefill of [0..L).
        let (mut slot_b, head_b) = load_session(&mut gpu, &model, &head_p, CTX_PATH2);
        let mut drafter_b = greedy_drafter(head_b, CTX_PATH2);
        let seed_b = drafter_b
            .mtp_prefill(&mut gpu, &mut slot_b, &long, &long, 0, false, &no_abort)
            .expect("arm B cold prefill");
        assert_eq!(
            seed_a, seed_b,
            "resumed and cold prefills must predict the same pending token"
        );
        let dn_b = dn_bytes(&gpu, &slot_b);
        let kv_b = kv_prefix_bytes(&gpu, &slot_b, long.len());
        let (prev_b, mtp_kv_b) = mtp_bytes(&gpu, &drafter_b, long.len());
        report_fams("rewind", "B(cold)", &dn_b, &kv_b, &mtp_kv_b, &prev_b);

        // Arm C (cold prefix [0..P)): reference for the pure-restore check.
        let (mut slot_c, head_c) = load_session(&mut gpu, &model, &head_p, CTX_PATH2);
        let mut drafter_c = greedy_drafter(head_c, CTX_PATH2);
        drafter_c
            .mtp_prefill(
                &mut gpu,
                &mut slot_c,
                &long[..p],
                &long[..p],
                0,
                false,
                &no_abort,
            )
            .expect("arm C cold prefix prefill");
        let dn_c = dn_bytes(&gpu, &slot_c);
        let kv_c = kv_prefix_bytes(&gpu, &slot_c, p);
        report_fams("rewind-restore", "C(cold-prefix)", &dn_c, &kv_c, &[], &[]);

        for (i, (a, c)) in dn_rw.iter().zip(dn_c.iter()).enumerate() {
            assert_eq!(a.len(), c.len(), "rewind dn family {i} length");
            assert_eq!(a, c, "rewind dn family {i} bytes differ (restored vs cold)");
        }
        assert_eq!(kv_rw, kv_c, "rewind trunk-KV retained-prefix bytes differ");
        for (i, (a, b)) in dn_a.iter().zip(dn_b.iter()).enumerate() {
            assert_eq!(a.len(), b.len(), "resumed dn family {i} length");
            assert_eq!(a, b, "resumed dn family {i} bytes differ (resumed vs cold)");
        }
        assert_eq!(kv_a, kv_b, "resumed trunk-KV bytes differ");
        assert_eq!(mtp_kv_a, mtp_kv_b, "resumed MTP-KV bytes differ");
        assert_eq!(prev_a, prev_b, "resumed MTP prev_hidden bytes differ");

        let eos_b = slot_b.config.eos_token;
        let wcont_b = drafter_b
            .mtp_step(
                &mut gpu,
                &mut slot_b,
                long.len(),
                seed_b,
                &long,
                MAX_N,
                eos_b,
                None,
            )
            .expect("arm B post-resume step");
        assert_eq!(
            wcont_a.committed, wcont_b.committed,
            "post-resume continuation diverges: A={:?} B={:?}",
            wcont_a.committed, wcont_b.committed
        );
        eprintln!(
            "[byte-id rewind] continuation identical: {:?} ({:?})",
            wcont_b.committed,
            tokenizer.decode(&wcont_b.committed)
        );
        free_session(&mut gpu, slot_b, drafter_b);
        free_session(&mut gpu, slot_c, drafter_c);
    }

    eprintln!("[byte-id] PASS: repair + rewind byte-identity proven");
}
