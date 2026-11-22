// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Deferred-tail MTP prompt fill: state equality against the eager prefill.
//!
//! `Speculator::defer_prefill_tail` lets the MTP drafter return the trunk's
//! first token while the LAST ordinary outer chunk's head fill stays owed
//! (`PendingMtpFill`), settled by `finish_prefill`. The settled state must be
//! byte-identical to the eager prefill of the same prompt:
//!
//! * DeltaNet S / scales / conv / EF residual, trunk KV `[0, P)`, trunk logits
//! * head Q8 KV `[0, P)` (K and V), `prev_hidden`, `prev_hidden_pos`
//! * the checkpoint ring's positions and prefix hashes
//! * the first token (greedy and sampled, same seed) and three following
//!   windows of drafted tokens.
//!
//! Controls that must hold (each would fail on a wrong implementation):
//! * Before `finish_prefill` the head KV of the owed rows is NOT the eager one
//!   (the debt is real; an omitted fill would be caught), and the prompt-end
//!   checkpoint is not advertised.
//! * `mtp_reset` / a cold prefill discards the debt and its checkpoint.
//! * A window run without an explicit `finish_prefill` settles the debt first
//!   and lands in the same state as eager + the same window.
//!
//! Run (one hiptrx/hipx card, fabric job):
//!
//! ```bash
//! HIPFIRE_DEVICES=<uuid> HIPFIRE_MTP_BYTE_IDENTITY_MODEL=<qwen3.8-27b.mq4-xts> \
//!   HIPFIRE_MTP_BYTE_IDENTITY_HEAD=<qwen3.8-27b.mtp> cargo test --release \
//!   -p hipfire-arch-qwen35 --test mtp_auto_generic_hw -- --ignored --nocapture
//! ```

#![allow(clippy::all)]

use hipfire_arch_qwen35::mtp_head::{self, Qwen35MtpHead};
use hipfire_arch_qwen35::mtp_speculator::Qwen35MtpDrafter;
use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig};
use rdna_compute::Gpu;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const CTX: usize = 8192;
const MAX_N: usize = 3;
const WINDOWS: usize = 3;
const SEED: u64 = 7;
const PROMPT_LENS: [usize; 4] = [37, 700, 2600, 5000];
const FIXTURE_TEXT: &str =
    "The harbor bell rang twice before dawn, and the gulls answered in kind while \
    the nets dried on the quay and the tide turned slowly toward the open water. ";

fn sha(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())[..16].to_string()
}

fn model_path() -> Option<PathBuf> {
    std::env::var("HIPFIRE_MTP_BYTE_IDENTITY_MODEL")
        .ok()
        .map(PathBuf::from)
}

fn head_path(model: &Path) -> PathBuf {
    std::env::var("HIPFIRE_MTP_BYTE_IDENTITY_HEAD")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model.with_extension("mtp"))
}

fn tensor_bytes(gpu: &Gpu, t: &rdna_compute::GpuTensor) -> Vec<u8> {
    let n = t.byte_size().min(t.buf.size());
    let mut out = vec![0u8; n];
    gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("dtoh");
    out
}

fn prefix_bytes(gpu: &Gpu, t: &rdna_compute::GpuTensor, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n.min(t.buf.size())];
    gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("dtoh prefix");
    out
}

/// Digest of everything the prefill must leave identical.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Fingerprint {
    dn: Vec<String>,
    trunk_kv: String,
    logits: String,
    head_kv_k: String,
    head_kv_v: String,
    prev_hidden: String,
    prev_hidden_pos: Option<usize>,
    checkpoints: Vec<(usize, u64)>,
}

fn capture(gpu: &Gpu, slot: &ModelSlot, d: &Qwen35MtpDrafter, positions: usize) -> Fingerprint {
    gpu.hip.device_synchronize().expect("sync");
    let dn = [
        &slot.dn_state.s_matrices,
        &slot.dn_state.s_scales,
        &slot.dn_state.conv_states,
        &slot.dn_state.s_ef_residual,
    ]
    .iter()
    .map(|set| {
        let mut fam = Vec::new();
        for t in set.iter() {
            fam.extend_from_slice(&tensor_bytes(gpu, t));
        }
        sha(&fam)
    })
    .collect();
    let kv = &slot.kv_cache;
    let row = kv.n_kv_heads * (kv.head_dim / 32) * 34;
    let mut trunk = Vec::new();
    for layer in 0..kv.k_gpu.len() {
        for buf in [&kv.k_gpu[layer], &kv.v_gpu[layer]] {
            if buf.buf.size() < row {
                continue;
            }
            trunk.extend_from_slice(&prefix_bytes(gpu, buf, positions * row));
        }
    }
    let logits: Vec<u8> = gpu
        .download_f32(&slot.scratch.logits)
        .expect("logits")
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let st = d.mtp_live_state().expect("state");
    let hrow = st.mtp_kv.n_head_kv * (st.mtp_kv.head_dim / 32) * 34;
    let hk = prefix_bytes(gpu, &st.mtp_kv.inner.k_gpu[0], positions * hrow);
    let hv = prefix_bytes(gpu, &st.mtp_kv.inner.v_gpu[0], positions * hrow);
    let positions_ck = d.mtp_checkpoint_positions();
    let hashes = d.mtp_checkpoint_prefix_hashes();
    assert_eq!(positions_ck.len(), hashes.len());
    Fingerprint {
        dn,
        trunk_kv: sha(&trunk),
        logits: sha(&logits),
        head_kv_k: sha(&hk),
        head_kv_v: sha(&hv),
        prev_hidden: sha(&tensor_bytes(gpu, &st.prev_hidden)),
        prev_hidden_pos: st.prev_hidden_pos,
        checkpoints: hashes,
    }
}

fn load(gpu: &mut Gpu, model: &Path, head: &Path) -> (ModelSlot, Qwen35MtpHead) {
    let mut cfg = ModelSlotConfig::default();
    cfg.max_seq = CTX;
    let slot = ModelSlot::load(gpu, model, "mtp-auto-generic", cfg).expect("load trunk");
    let h = mtp_head::load_mtp_head(head, gpu, CTX).expect("load head");
    (slot, h)
}

fn fixture(slot: &ModelSlot, len: usize) -> Vec<u32> {
    let tok = slot.load_tokenizer().expect("tokenizer");
    let unit = tok.encode(FIXTURE_TEXT);
    assert!(!unit.is_empty());
    (0..len).map(|i| unit[i % unit.len()]).collect()
}

/// `WINDOWS` windows from the prefill's seed; returns every committed token.
fn windows(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    d: &mut Qwen35MtpDrafter,
    prompt: &[u32],
    first: u32,
    n: usize,
) -> Vec<u32> {
    let eos = u32::MAX;
    let mut history = prompt.to_vec();
    history.push(first);
    let mut pos = prompt.len();
    let mut seed = first;
    let mut out = vec![first];
    for _ in 0..n {
        let w = d
            .mtp_step(gpu, slot, pos, seed, &history, MAX_N, eos, None)
            .expect("window");
        history.extend_from_slice(&w.committed);
        pos += w.committed.len();
        seed = *history.last().unwrap();
        out.extend_from_slice(&w.committed);
    }
    out
}

fn request(sampled: bool) -> SpecRequestConfig {
    if sampled {
        SpecRequestConfig {
            temp: 0.7,
            top_p: 0.95,
            top_k: Some(20),
            rng_seed: SEED,
            ..SpecRequestConfig::default()
        }
    } else {
        SpecRequestConfig::default()
    }
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_MTP_BYTE_IDENTITY_MODEL (qwen3.8 dense trunk) + MTP head"]
fn deferred_tail_prefill_equals_eager() {
    let Some(model) = model_path() else {
        eprintln!("skipping: HIPFIRE_MTP_BYTE_IDENTITY_MODEL not set");
        return;
    };
    let head_p = head_path(&model);
    // Small interval so the ring takes a checkpoint at the prompt end too.
    std::env::set_var("HIPFIRE_CACHE_CKPT_INTERVAL", "256");
    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!("[deferred-tail] gpu {}", gpu.arch);
    let (mut slot, head) = load(&mut gpu, &model, &head_p);
    let mut d = Qwen35MtpDrafter::new(head, MAX_N, CTX);
    let no_abort = || false;
    let mut failures: Vec<String> = Vec::new();

    for sampled in [false, true] {
        for &len in &PROMPT_LENS {
            let tag = format!("{}-{len}", if sampled { "sampled" } else { "greedy" });
            let prompt = fixture(&slot, len);

            // ── Eager reference ──────────────────────────────────────
            d.configure_request(request(sampled));
            let first_e = d
                .mtp_prefill(&mut gpu, &mut slot, &prompt, &prompt, 0, false, &no_abort)
                .and_then(|o| o.into_first_token())
                .expect("eager prefill");
            assert_eq!(d.owed_fill_rows(), 0, "{tag}: eager prefill left debt");
            let fp_e = capture(&gpu, &slot, &d, len);
            let toks_e = windows(&mut gpu, &mut slot, &mut d, &prompt, first_e, WINDOWS);
            let fp_e_after = capture(&gpu, &slot, &d, len + 8);

            // ── Deferred tail + explicit finish_prefill ──────────────
            d.configure_request(request(sampled));
            d.mtp_defer_prefill_tail(true);
            let first_d = d
                .mtp_prefill(&mut gpu, &mut slot, &prompt, &prompt, 0, false, &no_abort)
                .and_then(|o| o.into_first_token())
                .expect("deferred prefill");
            let owed = d.owed_fill_rows();
            assert!(owed > 0, "{tag}: deferring prefill left no debt");
            // Control: the owed rows are not there yet, and the prompt-end
            // checkpoint is not advertised.
            let fp_pre = capture(&gpu, &slot, &d, len);
            if fp_pre.head_kv_k == fp_e.head_kv_k || fp_pre.head_kv_v == fp_e.head_kv_v {
                failures.push(format!(
                    "{tag}: head KV already complete before finish_prefill (no debt?)"
                ));
            }
            if d.mtp_checkpoint_positions().contains(&len) {
                failures.push(format!(
                    "{tag}: prompt-end checkpoint advertised while fill is owed"
                ));
            }
            d.mtp_finish_prefill(&mut gpu, &mut slot)
                .expect("finish_prefill");
            assert_eq!(d.owed_fill_rows(), 0, "{tag}: debt after finish");
            d.mtp_finish_prefill(&mut gpu, &mut slot)
                .expect("finish idempotent");
            let fp_d = capture(&gpu, &slot, &d, len);
            if first_d != first_e {
                failures.push(format!("{tag}: first token {first_d} != eager {first_e}"));
            }
            if fp_d != fp_e {
                failures.push(format!(
                    "{tag}: settled state != eager\n  eager   {fp_e:?}\n  settled {fp_d:?}"
                ));
            }
            let toks_d = windows(&mut gpu, &mut slot, &mut d, &prompt, first_d, WINDOWS);
            let fp_d_after = capture(&gpu, &slot, &d, len + 8);
            if toks_d != toks_e {
                failures.push(format!(
                    "{tag}: window tokens differ {toks_d:?} vs {toks_e:?}"
                ));
            }
            if fp_d_after != fp_e_after {
                failures.push(format!("{tag}: state after windows differs"));
            }

            // ── Control: debt discarded by reset, nothing left behind ──
            d.configure_request(request(sampled));
            d.mtp_defer_prefill_tail(true);
            d.mtp_prefill(&mut gpu, &mut slot, &prompt, &prompt, 0, false, &no_abort)
                .expect("deferred prefill (discard)");
            assert!(d.owed_fill_rows() > 0);
            d.mtp_reset(&mut gpu).expect("reset");
            if d.owed_fill_rows() != 0 || d.mtp_checkpoint_positions().contains(&len) {
                failures.push(format!("{tag}: reset left debt or the guarded checkpoint"));
            }

            // ── Control: a window settles the debt itself ─────────────
            d.configure_request(request(sampled));
            d.mtp_defer_prefill_tail(true);
            let first_s = d
                .mtp_prefill(&mut gpu, &mut slot, &prompt, &prompt, 0, false, &no_abort)
                .and_then(|o| o.into_first_token())
                .expect("deferred prefill (implicit settle)");
            let toks_s = windows(&mut gpu, &mut slot, &mut d, &prompt, first_s, WINDOWS);
            let fp_s_after = capture(&gpu, &slot, &d, len + 8);
            if first_s != first_e || toks_s != toks_e || fp_s_after != fp_e_after {
                failures.push(format!("{tag}: implicit settle by mtp_step != eager"));
            }
            eprintln!(
                "[deferred-tail] {tag}: owed {owed} rows, settled state byte-equal: {}",
                fp_d == fp_e
            );
        }
    }
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    Box::new(d).mtp_free(&mut gpu);
    assert!(
        failures.is_empty(),
        "deferred-tail oracle failures:\n{}",
        failures.join("\n")
    );
}
