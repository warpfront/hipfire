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
//!   -p hipfire-arch-qwen35 --test mtp_takeover_fill -- --ignored --nocapture
//! ```

#![allow(clippy::all)]

use hipfire_arch_qwen35::mtp_head::{self, MtpKvMode};
use hipfire_arch_qwen35::mtp_spec::{
    prefill_trunk_and_mtp_cache, spec_step_mtp_compressed_serial_with_k,
    spec_step_mtp_compressed_serial_with_takeover_candidates, MtpPromptRoute, MtpSpecState,
};
use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
use hipfire_runtime::spec::SpecTarget;
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
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
        .0 as u32
}

fn prefill(gpu: &mut Gpu, slot: &mut ModelSlot, head: &mtp_head::Qwen35MtpHead, state: &mut MtpSpecState, prompt: &[u32]) -> u32 {
    slot.reset_recurrent(gpu).expect("reset trunk");
    state.reset(gpu).expect("reset head");
    prefill_trunk_and_mtp_cache(gpu, slot, head, state, prompt, 0, MtpPromptRoute::ArRoute).expect("prefill");
    argmax(&gpu.download_f32(&slot.scratch.logits).expect("logits"))
}

#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_MTP_BYTE_IDENTITY_MODEL (qwen3.5/3.8 trunk) + MTP sidecar"]
fn takeover_fills_head_kv_like_one_token_decode() {
    let Some(model) = std::env::var("HIPFIRE_MTP_BYTE_IDENTITY_MODEL").ok().map(PathBuf::from) else {
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
        let r = spec_step_mtp_compressed_serial_with_k(&mut gpu, &mut slot, &head, &mut state, pos, seed, eos, 0)
            .expect("k=0 window");
        assert_eq!(r.advance, 1);
        assert_ne!(r.committed[0], eos, "reference continuation hit EOS; use a longer prompt");
        toks.push(r.committed[0]);
    }
    let reference = HeadRows::read(&gpu, &state, n, REF_STEPS);
    eprintln!("[fill] reference continuation: {:?}", tokenizer.decode(&toks));

    // ── Takeover arm ─────────────────────────────────────────────────────
    let seed = prefill(&mut gpu, &mut slot, &head, &mut state, &prompt);
    assert_eq!(seed, seed0, "prompt fill must be deterministic");
    let untouched = HeadRows::read(&gpu, &state, n, VERIFY_CAPACITY + 1);
    let wrong = |t: u32| if t == 0 { 1 } else { t - 1 };

    // (label, candidates relative to the window seed index, expected accept)
    let mut idx = 0usize; // index into `toks` of the current seed
    let mut filled = 0usize;
    let windows: [(&str, usize, bool); 3] = [("partial", 20, true), ("full", 30, false), ("zero", 0, true)];
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
        assert_eq!(r.accept_count, accept, "{label}: takeover must accept the reference prefix");
        assert_eq!(&r.committed[..], &toks[idx + 1..=idx + r.advance], "{label}: committed tokens");
        let adv = r.advance;

        // 4. Nothing past the kept rows was written by this window.
        if label == "partial" {
            let after = HeadRows::read(&gpu, &state, pos + adv, cands.len() + 1 - adv);
            let before_rows = cands.len() + 1 - adv;
            for j in 0..before_rows {
                assert_eq!(after.row(j), untouched.row(adv + j), "{label}: rejected row {j} written");
            }
        }

        // 2. Pairing: staged hidden row i is what decode pairs with token i.
        let staged = gpu.download_f32(&state.takeover_fill_hidden).expect("staged");
        let verify = gpu.download_f32(&state.verify_hidden).expect("verify hidden");
        for i in 0..adv {
            let row = &staged[i * dim..(i + 1) * dim];
            if i > 0 {
                assert_eq!(row, &verify[(i - 1) * dim..i * dim], "{label}: fill row {i} != verify row {}", i - 1);
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
            assert!(hidden_rel < 0.05, "{label} row {i}: hidden is not decode's h_(p-1) (rel {hidden_rel})");
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
            max_route_rel = max_route_rel.max(rel_l2(&filled_rows.row_f32(i), &decode_route.row_f32(i)));
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
        assert_eq!(route_identical, adv, "{label}: batched head route is not byte-identical to decode route ({max_route_rel})");
        assert!(max_e2e < 0.1 * min_offby1, "{label}: filled rows not decode-paired ({max_e2e} vs off-by-one {min_offby1})");

        idx += adv;
        filled += adv;
    }
    assert!(filled >= 50, "takeover windows filled only {filled} rows");

    // Native MTP drafts again after the takeovers.
    let pos = n + idx;
    let r = spec_step_mtp_compressed_serial_with_k(&mut gpu, &mut slot, &head, &mut state, pos, toks[idx], eos, MAX_N)
        .expect("native window after takeovers");
    eprintln!(
        "[fill] native window after takeovers: drafts={} accept={} committed={:?}",
        r.drafts_generated, r.accept_count, r.committed
    );
    assert_eq!(r.drafts_generated, MAX_N, "native MTP must draft after a takeover");
    assert_eq!(r.committed[0], toks[idx + 1], "greedy output diverged after takeovers");

    state.free_gpu(&mut gpu);
    head.free_gpu(&mut gpu);
    slot.kv_cache.free_gpu(&mut gpu).expect("free kv");
    slot.dn_state.free_gpu(&mut gpu);
    slot.scratch.free_gpu(&mut gpu).expect("free scratch");
    slot.weights.free_gpu(&mut gpu);
}
