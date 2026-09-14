// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Step-path oracle for the MTP byte-identity proof (`mtp_cache_byte_identity`).
//!
//! Arm D (control, NO repair) + Arm B2 (cold prefill of D's exact full token
//! stream) in ONE process (two sequential 35B sessions — the proven envelope;
//! a third session in the main binary OOMs on residue). Separate binary =
//! separate process; cross-process determinism is proven (identical fnv
//! across runs), so fnv equality across the two logs decides the oracle:
//!
//! - D == B2 on every family → normal stepping is prefill-exact, and the
//!   A-vs-B divergence in the main log isolates the repair → #749 defect.
//! - D != B2 but D-KV-fnv == A-KV-fnv from the main log → stepping itself is
//!   not prefill-exact (oracle too strict); repair reproduces stepping.
//! - else → repair diverges from both.
//!
//! Cursor convention mirrors `generate_spec`: GPU covers `[0..pos)`, the last
//! baked token is pending AT `pos`. B2 prefills D's processed prefix
//! `history[..pos_end]`; its predicted pending token must equal D's actual
//! pending token (deterministic-argmax cross-check).
//!
//! Run explicitly (GPU 1 per the v0.3.1 lane discipline):
//!
//! ```bash
//! HIP_VISIBLE_DEVICES=1 flock /tmp/hipfire-gpu1.lock cargo test --release \
//!   -p hipfire-arch-qwen35 --test mtp_step_oracle -- --ignored --nocapture
//! ```
//!
//! Gated on `HIPFIRE_MTP_BYTE_IDENTITY_MODEL` (+ optional
//! `HIPFIRE_MTP_BYTE_IDENTITY_HEAD`), same as `mtp_cache_byte_identity`.

#![allow(clippy::all)]

use hipfire_arch_qwen35::mtp_head::{self, Qwen35MtpHead};
use hipfire_arch_qwen35::mtp_speculator::Qwen35MtpDrafter;
use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig};
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};

const CTX: usize = 1024;
const MAX_N: usize = 4;
const PROMPT: &str = "Explain why the sky is blue in one short paragraph.";

fn model_path() -> Option<PathBuf> {
    std::env::var("HIPFIRE_MTP_BYTE_IDENTITY_MODEL")
        .ok()
        .map(PathBuf::from)
}

fn head_path(model: &Path) -> PathBuf {
    if let Ok(h) = std::env::var("HIPFIRE_MTP_BYTE_IDENTITY_HEAD") {
        return PathBuf::from(h);
    }
    model.with_extension("mtp")
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn tensor_bytes(gpu: &Gpu, t: &rdna_compute::GpuTensor) -> Vec<u8> {
    let n = t.byte_size().min(t.buf.size());
    let mut out = vec![0u8; n];
    gpu.hip
        .memcpy_dtoh(&mut out, &t.buf)
        .expect("dtoh tensor bytes");
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
    let slot = ModelSlot::load(gpu, model, "step-oracle", cfg).expect("load trunk");
    let mtp = mtp_head::load_mtp_head(head, gpu, max_seq).expect("load mtp head");
    (slot, mtp)
}

fn free_session(gpu: &mut Gpu, slot: ModelSlot, drafter: Qwen35MtpDrafter) {
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
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

fn kv_row_bytes(n_kv_heads: usize, head_dim: usize) -> usize {
    assert!(head_dim % 32 == 0, "q8 kv head_dim {head_dim} not /32");
    n_kv_heads * (head_dim / 32) * 34
}
/// Retained-prefix trunk KV bytes: per FullAttention layer (placeholders
/// skipped), K rows `[0..positions)` then V rows `[0..positions)`.
fn kv_prefix_bytes(gpu: &Gpu, slot: &ModelSlot, positions: usize) -> Vec<u8> {
    let kv = &slot.kv_cache;
    let row = kv_row_bytes(kv.n_kv_heads, kv.head_dim);
    let mut out = Vec::new();
    for layer in 0..kv.k_gpu.len() {
        for buf in [&kv.k_gpu[layer], &kv.v_gpu[layer]] {
            if buf.buf.size() < row {
                continue;
            }
            let n = positions * row;
            let mut pre = vec![0u8; n];
            gpu.hip
                .memcpy_dtoh(&mut pre, &buf.buf)
                .expect("dtoh kv prefix");
            out.extend_from_slice(&pre);
        }
    }
    out
}
/// MTP-head KV prefix rows `[0..positions)`: single-layer K then V.
fn mtp_prefix_bytes(gpu: &Gpu, drafter: &Qwen35MtpDrafter, positions: usize) -> Vec<u8> {
    let st = drafter.mtp_live_state().expect("mtp state");
    let kv = &st.mtp_kv.inner;
    let row = kv_row_bytes(st.mtp_kv.n_head_kv, st.mtp_kv.head_dim);
    let mut out = Vec::new();
    for buf in kv.k_gpu.iter().chain(kv.v_gpu.iter()) {
        if buf.buf.size() < row {
            continue;
        }
        let n = positions * row;
        let mut pre = vec![0u8; n];
        gpu.hip
            .memcpy_dtoh(&mut pre, &buf.buf)
            .expect("dtoh mtp kv prefix");
        out.extend_from_slice(&pre);
    }
    out
}

fn kv_full_bytes(gpu: &Gpu, slot: &ModelSlot) -> Vec<u8> {
    let kv = &slot.kv_cache;
    let mut out = Vec::new();
    for layer in 0..kv.k_gpu.len() {
        for buf in [&kv.k_gpu[layer], &kv.v_gpu[layer]] {
            out.extend_from_slice(&tensor_bytes(gpu, buf));
        }
    }
    out
}

fn mtp_full_bytes(gpu: &Gpu, drafter: &Qwen35MtpDrafter) -> (Vec<u8>, Vec<u8>) {
    let st = drafter
        .mtp_live_state()
        .expect("mtp state allocated after prefill");
    let prev = tensor_bytes(gpu, &st.prev_hidden);
    let kv = &st.mtp_kv.inner;
    let mut kvb = Vec::new();
    for buf in kv.k_gpu.iter().chain(kv.v_gpu.iter()) {
        kvb.extend_from_slice(&tensor_bytes(gpu, buf));
    }
    (prev, kvb)
}

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

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_MTP_BYTE_IDENTITY_MODEL (qwen3.5-MoE trunk) + MTP sidecar"]
fn mtp_step_oracle_no_repair() {
    let Some(model) = model_path() else {
        eprintln!("skipping: HIPFIRE_MTP_BYTE_IDENTITY_MODEL not set");
        return;
    };
    let head_p = head_path(&model);
    std::env::set_var("HIPFIRE_CACHE_CKPT_INTERVAL", "256");
    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!("[oracle] gpu: {}", gpu.arch);
    let no_abort = || false;

    // Arm D: prefill + normal window stepping, NO repair.
    let (mut slot_d, head_d) = load_session(&mut gpu, &model, &head_p, CTX);
    let tokenizer = slot_d.load_tokenizer().expect("tokenizer");
    let prompt = {
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        let user = tokenizer.encode("user");
        let asst = tokenizer.encode("assistant");
        let nl = tokenizer.encode("\n");
        let q = tokenizer.encode(PROMPT);
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
    };
    let mut drafter_d = greedy_drafter(head_d, CTX);
    let mut seed_d = drafter_d
        .mtp_prefill(&mut gpu, &mut slot_d, &prompt, &prompt, 0, false, &no_abort)
        .expect("arm D prefill");
    let eos_d = slot_d.config.eos_token;
    let mut history_d: Vec<u32> = prompt.clone();
    history_d.push(seed_d);
    let mut pos_d = prompt.len();
    let (wpos_d, wcommitted_d) = loop {
        let mut found = None;
        for _ in 0..8 {
            let w = drafter_d
                .mtp_step(
                    &mut gpu,
                    &mut slot_d,
                    pos_d,
                    seed_d,
                    &history_d,
                    MAX_N,
                    eos_d,
                    None,
                )
                .expect("arm D step");
            history_d.extend_from_slice(&w.committed);
            pos_d += w.committed.len();
            seed_d = *history_d.last().unwrap();
            if w.committed.len() >= 2 {
                found = Some((pos_d - w.committed.len(), w.committed.clone()));
                break;
            }
        }
        break found.expect("arm D: no multi-token window in 8 steps");
    };
    // Same resumed position the main test repairs to.
    let keep = wcommitted_d.len() - 1;
    let p1 = wpos_d + keep;
    eprintln!(
        "[oracle] window_start={wpos_d} committed={} keep={keep} p1={p1} pos_end={pos_d}",
        wcommitted_d.len()
    );
    let dn_d = dn_bytes(&gpu, &slot_d);
    let kv_d = kv_full_bytes(&gpu, &slot_d);
    let (prev_d, mtp_kv_d) = mtp_full_bytes(&gpu, &drafter_d);
    for (n, fam) in ["dn_s", "dn_scales", "dn_conv", "dn_ef"]
        .iter()
        .zip(dn_d.iter())
    {
        eprintln!(
            "[oracle D(stepped)] {n}: bytes={} fnv={:016x}",
            fam.len(),
            fnv1a(fam)
        );
    }
    eprintln!(
        "[oracle D(stepped)] trunk_kv_full: bytes={} fnv={:016x}",
        kv_d.len(),
        fnv1a(&kv_d)
    );
    eprintln!(
        "[oracle D(stepped)] mtp_kv_full: bytes={} fnv={:016x}",
        mtp_kv_d.len(),
        fnv1a(&mtp_kv_d)
    );
    eprintln!(
        "[oracle D(stepped)] mtp_prev_hidden: bytes={} fnv={:016x}",
        prev_d.len(),
        fnv1a(&prev_d)
    );
    let kv_pre_d = kv_prefix_bytes(&gpu, &slot_d, p1);
    eprintln!(
        "[oracle D(stepped)] trunk_kv_prefix_p1: bytes={} fnv={:016x}",
        kv_pre_d.len(),
        fnv1a(&kv_pre_d)
    );
    let mtp_pre_d = mtp_prefix_bytes(&gpu, &drafter_d, p1);
    eprintln!(
        "[oracle D(stepped)] mtp_kv_prefix_p1: bytes={} fnv={:016x}",
        mtp_pre_d.len(),
        fnv1a(&mtp_pre_d)
    );
    free_session(&mut gpu, slot_d, drafter_d);

    // Arm B2: cold prefill of D's exact processed stream.
    let full: Vec<u32> = history_d[..pos_d].to_vec();
    let (mut slot_b, head_b) = load_session(&mut gpu, &model, &head_p, CTX);
    let mut drafter_b = greedy_drafter(head_b, CTX);
    let seed_b = drafter_b
        .mtp_prefill(&mut gpu, &mut slot_b, &full, &full, 0, false, &no_abort)
        .expect("arm B2 cold prefill");
    assert_eq!(
        seed_b, history_d[pos_d],
        "cold prefill must predict D's actual pending token"
    );
    let dn_b = dn_bytes(&gpu, &slot_b);
    let kv_b = kv_full_bytes(&gpu, &slot_b);
    let (prev_b, mtp_kv_b) = mtp_full_bytes(&gpu, &drafter_b);
    for (n, fam) in ["dn_s", "dn_scales", "dn_conv", "dn_ef"]
        .iter()
        .zip(dn_b.iter())
    {
        eprintln!(
            "[oracle B2(cold)] {n}: bytes={} fnv={:016x}",
            fam.len(),
            fnv1a(fam)
        );
    }
    eprintln!(
        "[oracle B2(cold)] trunk_kv_full: bytes={} fnv={:016x}",
        kv_b.len(),
        fnv1a(&kv_b)
    );
    eprintln!(
        "[oracle B2(cold)] mtp_kv_full: bytes={} fnv={:016x}",
        mtp_kv_b.len(),
        fnv1a(&mtp_kv_b)
    );
    eprintln!(
        "[oracle B2(cold)] mtp_prev_hidden: bytes={} fnv={:016x}",
        prev_b.len(),
        fnv1a(&prev_b)
    );

    // Three-way oracle report. Full-buffer compare covers every prefix;
    // first-diff rows locate divergence in token positions.
    let kv_row = kv_row_bytes(slot_b.kv_cache.n_kv_heads, slot_b.kv_cache.head_dim);
    let mtp_row = {
        let st = drafter_b.mtp_live_state().expect("mtp state");
        kv_row_bytes(st.mtp_kv.n_head_kv, st.mtp_kv.head_dim)
    };
    eprintln!(
        "[oracle] D-vs-B2 trunk_kv equal={} first_diff_row={:?}",
        kv_d == kv_b,
        first_diff_row(&kv_d, &kv_b, kv_row)
    );
    eprintln!(
        "[oracle] D-vs-B2 mtp_kv equal={} first_diff_row={:?}",
        mtp_kv_d == mtp_kv_b,
        first_diff_row(&mtp_kv_d, &mtp_kv_b, mtp_row)
    );
    eprintln!("[oracle] D-vs-B2 prev_hidden equal={}", prev_d == prev_b);
    for (i, (d, b)) in dn_d.iter().zip(dn_b.iter()).enumerate() {
        eprintln!(
            "[oracle] D-vs-B2 dn_fam{i} equal={} fnv D={:016x} B2={:016x}",
            d == b,
            fnv1a(d),
            fnv1a(b)
        );
    }
    free_session(&mut gpu, slot_b, drafter_b);
    eprintln!("[oracle] done");

    assert_eq!(kv_d, kv_b, "step-path trunk KV differs from cold prefill");
    assert_eq!(
        mtp_kv_d, mtp_kv_b,
        "step-path MTP KV differs from cold prefill"
    );
    assert_eq!(
        prev_d, prev_b,
        "step-path prev_hidden differs from cold prefill"
    );
    for (i, (d, b)) in dn_d.iter().zip(dn_b.iter()).enumerate() {
        assert_eq!(d, b, "step-path dn family {i} differs from cold prefill");
    }
}
