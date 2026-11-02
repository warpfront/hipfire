// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase oracle for Design A slice A4: the eager DFlash draft versus its
//! emitted HIP twin (`dflash_draft_emit::plan_dflash_draft` +
//! `Gpu::run_planned_ops`), on identical preimages.
//!
//! Per case `(L0, delta)` (every `delta = 1..=16`, ten context lengths per
//! delta around the ring seams of the draft window `W`: `W/2`, `W-1`, `W`,
//! `W+1`, new rows wrapping the slot boundary, `W+300`, `2W-1`, `2W`, `2W+1`,
//! new rows wrapping at `2W`):
//!
//! 1. Prime: deterministic synthetic accepted-hidden rows are written into
//!    the draft's `target_hidden` ring (slot = row % W), `thlog` is seeded and
//!    one eager `draft_dflash_block_rank` at `L0` fills every K/V cache and
//!    projection (watermarks = `L0`). `delta` fresh rows are appended.
//! 2. Snapshot the complete draft state (every scratch allocation except the
//!    read-only `target_hidden` ring, the verify logits/argmax/rot planes, the
//!    selector projection / top-K scratch, the staging and token buffers and
//!    the selector block) = S0.
//! 3. EAGER arm: the production `draft_dflash_block_rank` (host scorer),
//!    snapshot E, CPU selector scores recomputed from E's projection/top-K
//!    with `propose_candidates_from_rows_capture`.
//! 4. Restore S0 + the exact `thlog` cursors. TWIN arm A: plan at a
//!    DIFFERENT shape of the same family (`L_A != L`), pack the window inputs
//!    into `PinnedWindowInputs`, one async H2D + fence, bind the declared patches
//!    to the case shape (`bind_draft_ops`), `Gpu::run_planned_ops`, snapshot
//!    T_A. TWIN arm S: the same with a plan made at the case shape itself.
//! 5. Gates (byte exact): (a) bound plan A == plan S op for op (kernarg bytes,
//!    grids, blocks, LDS, copy addresses/sizes) — the declared patch table is
//!    complete and correct; (b) T_A == E and T_S == E on every compared
//!    allocation (final hidden `x`, `target_hidden_proj`, every per-layer K/V
//!    context cache, `k_noise`/`v_noise`/`k_cat`/`v_cat`/`attn_out`, selector
//!    projection and top-K scratch, verify logits, ...); (c) the device
//!    proposal block == the eager `drafted` chain, status 0, the device scores
//!    == the CPU scorer's scores, and the verify-token words 1..16 == the
//!    eager proposals; (d) nothing but the declared host delta changed in
//!    `thlog` while planning.
//!
//! Controls: out-of-family / out-of-domain binds must refuse, and a twin run
//! bound with `draft_seed + 1` must NOT reproduce the eager state (the seed
//! patch is live).
//!
//! Per-layer note: the draft layer hiddens live only in the shared `x` plane
//! and are overwritten layer by layer, so the per-layer evidence is the
//! per-layer K/V caches plus the final hidden after five chained layers (a
//! divergence at any layer would flip `x`).
//!
//! Needs an exact-gfx1201 box, a dense DFlash2 target + selector draft:
//!
//! ```text
//! HIPFIRE_DEVICES=<uuid> cargo run --release -p hipfire-arch-qwen35 \
//!     --features deltanet --example dflash_draft_emit_oracle -- \
//!     --target <27b.mq4> --draft <dflash2.hfq> [--out DIR] [--deltas 1,2,15,16] [--quick]
//! ```
//! (`HIPFIRE_TEST_TARGET` / `HIPFIRE_TEST_DRAFT` also work.) Exit 0 on PASS.

use hip_bridge::DeviceBuffer;
use hipfire_arch_qwen35::dflash_draft_emit::{
    bind_draft_ops, draft_family_key, draft_staged_words, draft_state_tensors, plan_dflash_draft,
    DraftEmission, DraftEmitArgs, DraftStagingLayout, DRAFT_ROWS,
};
use hipfire_arch_qwen35::dflash_spec::{load_dflash_state, DflashState};
use hipfire_arch_qwen35::speculative::{
    draft_dflash_block_rank, KvMode, ModelSlot, ModelSlotConfig,
};
use hipfire_runtime::dflash::{
    self, PinnedWindowInputs, SelectorDeviceCodebooks, SelectorGreedyBlock,
};
use railgun::window::WindowShape;
use rdna_compute::plan_sink::PlannedOp;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::fmt::Write as _;
use std::path::PathBuf;

type R<T> = Result<T, Box<dyn std::error::Error>>;

/// Draft context capacity (rows); the window ring is the draft's declared W.
const CTX: usize = 8192;

fn le_u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn fnv(b: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &x in b {
        h ^= u64::from(x);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Deterministic accepted-hidden row `r` (`n` f32), magnitude ~0.5.
fn row_vec(r: usize, n: usize) -> Vec<f32> {
    let mut s = (r as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            ((u - 0.5) * 1.0) as f32
        })
        .collect()
}

fn as_bytes_f32(v: &[f32]) -> &[u8] {
    // SAFETY: f32 slice viewed as its initialized bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

struct Rig {
    gpu: Gpu,
    slot: ModelSlot,
    df: DflashState,
    books: SelectorDeviceCodebooks,
    blk: SelectorGreedyBlock,
    staging: GpuTensor,
    tokens: GpuTensor,
    pinned: PinnedWindowInputs,
    stream: hip_bridge::Stream,
    w: usize,
}

/// All compared / snapshotted allocations: draft scratch (minus the read-only
/// `target_hidden` ring) + the twin-only buffers.
fn tracked<'a>(r: &'a Rig) -> Vec<(String, &'a DeviceBuffer)> {
    let mut v: Vec<(String, &DeviceBuffer)> =
        draft_state_tensors(&r.df.draft_scratch, &r.df.verify_scratch)
            .into_iter()
            .filter(|(n, _)| n != "target_hidden")
            .map(|(n, t)| (n, &t.buf))
            .collect();
    v.push(("staging".into(), &r.staging.buf));
    v.push(("tokens".into(), &r.tokens.buf));
    v.push(("selector_seed".into(), r.blk.seed_buffer()));
    v.push(("selector_scores".into(), r.blk.scores_buffer()));
    v.push(("selector_block".into(), r.blk.block_buffer()));
    v.push(("selector_status".into(), r.blk.status_buffer()));
    v
}

/// Buffers only the twin writes: not compared against the eager arm.
fn twin_only(name: &str) -> bool {
    matches!(
        name,
        "staging"
            | "tokens"
            | "selector_seed"
            | "selector_scores"
            | "selector_block"
            | "selector_status"
    )
}

type Snapshot = Vec<(String, Vec<u8>)>;

impl Rig {
    fn snapshot(&self) -> R<Snapshot> {
        let mut out = Vec::new();
        for (n, b) in tracked(self) {
            let mut bytes = vec![0u8; b.size()];
            self.gpu.hip.memcpy_dtoh(&mut bytes, b)?;
            out.push((n, bytes));
        }
        Ok(out)
    }

    fn restore(&self, s: &Snapshot) -> R<()> {
        for ((n, b), (n2, bytes)) in tracked(self).into_iter().zip(s) {
            assert_eq!(&n, n2);
            self.gpu.hip.memcpy_htod(b, bytes)?;
        }
        Ok(())
    }

    /// Reproduce the `thlog` cursors of a window at `(l, c, f)` with contiguous
    /// absolute positions (production `compact_offset == 0`).
    fn set_thlog(&mut self, l: usize, c: usize, f: usize) {
        let t = &mut self.df.draft_scratch.thlog;
        t.reset();
        t.seed_prompt(l);
        t.set_resume_checkpoint(c);
        t.mark_full_cached(f);
    }

    fn write_rows(&self, rows: std::ops::Range<usize>) -> R<()> {
        let cfg = &self.df.draft_config;
        let row_f32 = cfg.num_extract() * cfg.hidden;
        let th = &self.df.draft_scratch.target_hidden;
        for r in rows {
            let v = row_vec(r, row_f32);
            self.gpu.hip.memcpy_htod_offset(
                &th.buf,
                (r % self.w) * row_f32 * 4,
                as_bytes_f32(&v),
            )?;
        }
        Ok(())
    }

    fn eager(&mut self, l: usize, seed: u32) -> R<Vec<u32>> {
        let mask = self.df.draft_config.mask_token_id;
        let block: Vec<u32> = std::iter::once(seed)
            .chain(std::iter::repeat(mask).take(DRAFT_ROWS - 1))
            .collect();
        let vocab = self.df.draft_config.vocab_size;
        Ok(draft_dflash_block_rank(
            &mut self.gpu,
            &self.slot.weights,
            &self.df.draft_weights,
            &self.df.draft_config,
            &mut self.df.draft_scratch,
            &self.df.verify_scratch,
            &[],
            &block,
            l,
            0,
            None,
            false,
            seed,
            vocab,
        )?)
    }

    /// Eager prime: ring rows, thlog, one eager draft at `l0`.
    fn prime(&mut self, l0: usize) -> R<()> {
        self.write_rows(l0.saturating_sub(self.w)..l0)?;
        self.set_thlog(l0, 0, 0);
        self.eager(l0, 7)?;
        self.gpu.hip.device_synchronize()?;
        Ok(())
    }

    fn plan_at(&mut self, l: usize, c: usize, seed: u32) -> Result<DraftEmission, String> {
        self.set_thlog(l, c, c);
        let layout = DraftStagingLayout {
            seed_off: 0,
            ids_off: 4,
            q_off: 4 + DRAFT_ROWS * 4,
            k_off: 4 + 2 * DRAFT_ROWS * 4,
        };
        let a = DraftEmitArgs {
            target_weights: &self.slot.weights,
            draft_weights: &self.df.draft_weights,
            draft_cfg: &self.df.draft_config,
            draft_scratch: &mut self.df.draft_scratch,
            verify_scratch: &self.df.verify_scratch,
            selector: Some((&self.books, &self.blk)),
            staging: &self.staging.buf,
            layout,
            verify_tokens: &self.tokens.buf,
            position: l,
            compact_offset: 0,
            seed_token: seed,
            mask_token: self.df.draft_config.mask_token_id,
            max_position: (CTX * 2) as u64,
        };
        plan_dflash_draft(&mut self.gpu, a)
    }

    /// Pack the window inputs, one async H2D + fence.
    fn stage(&mut self, l: usize, seed: u32) -> R<()> {
        let words = draft_staged_words(
            self.df.draft_scratch.thlog.abs_positions(),
            l,
            0,
            seed,
            self.df.draft_config.mask_token_id,
            DRAFT_ROWS,
            self.w,
        );
        let offs = self.pinned.pack(&[
            &[words.seed],
            &words.noise_ids,
            &words.q_positions,
            &words.k_positions,
        ])?;
        assert_eq!(
            offs,
            vec![0, 4, 4 + DRAFT_ROWS * 4, 4 + 2 * DRAFT_ROWS * 4],
            "staging layout"
        );
        self.pinned
            .submit(&self.gpu, &self.staging.buf, &self.stream)?;
        self.pinned.complete(&self.gpu)?;
        Ok(())
    }
}

fn shape_of(l: usize, c: usize, w: usize) -> WindowShape {
    WindowShape {
        position: l as u64,
        logical_ctx_rows: l as u64,
        projected_rows: c as u64,
        full_cached_rows: c as u64,
        swa_w: w as u64,
        full_w: w as u64,
        max_position: (CTX * 2) as u64,
        verify_rows: 16,
        draft_rows: 16,
        ..WindowShape::default()
    }
}

/// First op where two op lists differ (None = identical incl. LDS).
fn first_op_diff(a: &[PlannedOp], b: &[PlannedOp]) -> Option<String> {
    if a.len() != b.len() {
        return Some(format!("{} ops vs {} ops", a.len(), b.len()));
    }
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        match (x, y) {
            (PlannedOp::Kernel(p), PlannedOp::Kernel(q)) => {
                if p.symbol != q.symbol {
                    return Some(format!("op {i}: {} vs {}", p.symbol, q.symbol));
                }
                if (p.grid, p.block, p.dynamic_lds) != (q.grid, q.block, q.dynamic_lds) {
                    return Some(format!(
                        "op {i} {}: grid/block/lds {:?}/{:?}/{} vs {:?}/{:?}/{}",
                        p.symbol, p.grid, p.block, p.dynamic_lds, q.grid, q.block, q.dynamic_lds
                    ));
                }
                if p.kernarg != q.kernarg {
                    let at = p
                        .kernarg
                        .iter()
                        .zip(&q.kernarg)
                        .position(|(u, v)| u != v)
                        .unwrap_or(p.kernarg.len().min(q.kernarg.len()));
                    return Some(format!("op {i} {}: kernarg differs at byte {at}", p.symbol));
                }
            }
            (
                PlannedOp::Copy { dst, src, bytes },
                PlannedOp::Copy {
                    dst: d2,
                    src: s2,
                    bytes: b2,
                },
            ) => {
                if (dst, src, bytes) != (d2, s2, b2) {
                    return Some(format!(
                        "op {i} copy {src:#x}->{dst:#x} {bytes}B vs {s2:#x}->{d2:#x} {b2}B"
                    ));
                }
            }
            (
                PlannedOp::Fill { dst, value, words },
                PlannedOp::Fill {
                    dst: d2,
                    value: v2,
                    words: w2,
                },
            ) => {
                if (dst, value, words) != (d2, v2, w2) {
                    return Some(format!("op {i}: fill differs"));
                }
            }
            _ => return Some(format!("op {i}: kind differs")),
        }
    }
    None
}

fn compare(e: &Snapshot, t: &Snapshot) -> Vec<String> {
    let mut bad = Vec::new();
    for ((n, a), (_, b)) in e.iter().zip(t) {
        if twin_only(n) {
            continue;
        }
        if a != b {
            let at = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0);
            let nd = a.iter().zip(b).filter(|(x, y)| x != y).count();
            bad.push(format!(
                "{n}: {nd} bytes differ, first at {at} (eager {:016x} twin {:016x})",
                fnv(a),
                fnv(b)
            ));
        }
    }
    bad
}

fn snap_get<'a>(s: &'a Snapshot, name: &str) -> &'a [u8] {
    &s.iter().find(|(n, _)| n == name).expect("tracked name").1
}

struct CaseOutcome {
    line: String,
    fails: Vec<String>,
}

fn run_case(r: &mut Rig, l0: usize, d: usize, idx: usize) -> R<CaseOutcome> {
    let l = l0 + d;
    let seed = 1000 + idx as u32;
    let w = r.w;
    let mut fails = Vec::new();
    r.prime(l0)?;
    r.write_rows(l0..l)?;
    r.set_thlog(l, l0, l0);
    let s0 = r.snapshot()?;

    // ---- eager arm
    let drafted = r.eager(l, seed)?;
    r.gpu.hip.device_synchronize()?;
    let e = r.snapshot()?;
    let cfg = &r.df.draft_config;
    let rows = DRAFT_ROWS - 1;
    let rank = r.df.draft_weights.selector_rank.unwrap_or(256);
    let k = r.df.draft_weights.selector_top_k.unwrap_or(16);
    let proj: Vec<f32> = le_u32s(snap_get(&e, "selector_proj")[..rows * rank * 4].as_ref())
        .into_iter()
        .map(f32::from_bits)
        .collect();
    let ids = le_u32s(&snap_get(&e, "topk_ids")[..rows * k * 4]);
    let vals: Vec<f32> = le_u32s(&snap_get(&e, "topk_vals")[..rows * k * 4])
        .into_iter()
        .map(f32::from_bits)
        .collect();
    let mut cpu_scores: Vec<f32> = Vec::new();
    let cpu = dflash::propose_candidates_from_rows_capture(
        &r.df.draft_weights,
        &proj,
        &ids,
        &vals,
        rows,
        seed,
        0.0,
        None,
        Some(&mut cpu_scores),
    )?;
    let mut cpu_block = vec![seed];
    cpu_block.extend_from_slice(&cpu.tokens);
    if cpu_block != drafted {
        fails.push("eager drafted != CPU scorer recomputation (harness)".into());
    }
    let _ = cfg;

    // ---- plans: at a different shape of the same family, and at the case shape
    r.restore(&s0)?;
    let shape_s = shape_of(l, l0, w);
    let key_s = draft_family_key(&shape_s)?;
    let em_s = r
        .plan_at(l, l0, seed)
        .map_err(|x| format!("plan at case shape: {x}"))?;
    let mut em_a: Option<(usize, DraftEmission)> = None;
    for off in [37isize, -37, 101, -101, 1, -1, 211, -211, 509, -509] {
        let la = l as isize + off;
        if la <= d as isize || la as usize > CTX - 32 || off == 0 {
            continue;
        }
        let la = la as usize;
        if draft_family_key(&shape_of(la, la - d, w)).ok().as_ref() != Some(&key_s) {
            continue;
        }
        em_a = Some((
            la,
            r.plan_at(la, la - d, seed)
                .map_err(|x| format!("plan at {la}: {x}"))?,
        ));
        break;
    }
    // planning never touches GPU state or cursors; rebuild the case cursors.
    r.set_thlog(l, l0, l0);
    let (la_used, em_a) = em_a.unwrap_or_else(|| (l, em_s.clone()));
    em_s.accepts(&em_s.shape)?;
    let bound = bind_draft_ops(&em_a, &em_s.shape, &em_a.scalars(seed))?;
    if let Some(diff) = first_op_diff(&bound, &em_s.ops) {
        fails.push(format!(
            "bound plan(L_A={la_used}) != plan at case shape: {diff}"
        ));
    }

    // ---- twin arms
    let run_twin =
        |r: &mut Rig, ops: &[PlannedOp], label: &str, fails: &mut Vec<String>| -> R<Snapshot> {
            r.restore(&s0)?;
            r.set_thlog(l, l0, l0);
            r.stage(l, seed)?;
            r.gpu.run_planned_ops(ops)?;
            r.gpu.hip.device_synchronize()?;
            let t = r.snapshot()?;
            for m in compare(&e, &t) {
                fails.push(format!("{label}: {m}"));
            }
            Ok(t)
        };
    let t_a = run_twin(r, &bound, "twinA", &mut fails)?;
    let _t_s = run_twin(r, &em_s.ops, "twinS", &mut fails)?;

    // ---- proposal block / scores / verify tokens (twin A state)
    let status = le_u32s(snap_get(&t_a, "selector_status"))[0];
    // The selector block/score buffers are allocated with headroom; only the
    // live prefix (16 block ids, rows x k scores) is the window's output.
    let block_all = le_u32s(snap_get(&t_a, "selector_block"));
    let scores_all: Vec<f32> = le_u32s(snap_get(&t_a, "selector_scores"))
        .into_iter()
        .map(f32::from_bits)
        .collect();
    let tokens = le_u32s(&snap_get(&t_a, "tokens")[..DRAFT_ROWS * 4]);
    if status != 0 {
        fails.push(format!("selector status {status:#x}"));
    }
    let block = &block_all[..drafted.len().min(block_all.len())];
    if block != drafted.as_slice() {
        fails.push(format!("proposal block {block:?} != eager {drafted:?}"));
    }
    let scores = &scores_all[..cpu_scores.len().min(scores_all.len())];
    if scores.len() != cpu_scores.len()
        || scores
            .iter()
            .zip(&cpu_scores)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        let first = scores
            .iter()
            .zip(&cpu_scores)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        let k = (cpu_scores.len() / drafted.len().saturating_sub(1).max(1)).max(1);
        let detail = first.map(|i| {
            format!(
                " (row {} lane {}: gpu {:#010x} {:e} vs cpu {:#010x} {:e}; differing {} of {})",
                i / k,
                i % k,
                scores[i].to_bits(),
                scores[i],
                cpu_scores[i].to_bits(),
                cpu_scores[i],
                scores
                    .iter()
                    .zip(&cpu_scores)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count(),
                scores.len()
            )
        });
        fails.push(format!(
            "device scores != CPU scorer scores (bytes): len {} vs {}, first diff {first:?}{}",
            scores.len(),
            cpu_scores.len(),
            detail.unwrap_or_default()
        ));
    }
    if tokens[1..] != drafted[1..] {
        fails.push(format!(
            "verify tokens[1..16] {:?} != eager {:?}",
            &tokens[1..],
            &drafted[1..]
        ));
    }

    let line = format!(
        "L0={l0:5} d={d:2} L={l:5} W={w} fc_rows={:?} concat_segs={} planA@{la_used} kernels={} ops={} lds_sat={} dom={:?} {}",
        em_s.family.projection_rows,
        em_s.concat_segments,
        em_s.kernel_count(),
        em_s.ops.len(),
        em_s.attention_lds_saturated,
        em_s.l_domain,
        if fails.is_empty() { "OK" } else { "FAIL" }
    );
    Ok(CaseOutcome { line, fails })
}

/// Host-only and live-seed controls on one case.
fn controls(r: &mut Rig, report: &mut String) -> R<bool> {
    let w = r.w;
    let (l0, d) = (w / 2, 3usize);
    let l = l0 + d;
    let seed = 4242u32;
    let mut ok = true;
    r.prime(l0)?;
    r.write_rows(l0..l)?;
    r.set_thlog(l, l0, l0);
    let s0 = r.snapshot()?;
    let em = r.plan_at(l, l0, seed)?;
    r.set_thlog(l, l0, l0);
    let mut line = |name: &str, pass: bool| {
        ok &= pass;
        let _ = writeln!(
            report,
            "control {name}: {}",
            if pass { "PASS" } else { "FAIL" }
        );
    };
    // out-of-family: delta 5 is a different projection family
    line(
        "refuse other delta",
        em.accepts(&shape_of(l0 + 5, l0, w)).is_err(),
    );
    // cold seed: delta > 16
    line(
        "refuse delta 17",
        draft_family_key(&shape_of(l0 + 17, l0, w)).is_err(),
    );
    // out of domain bind
    line(
        "refuse out-of-domain bind",
        bind_draft_ops(
            &em,
            &shape_of(*em.l_domain.end() as usize + 1 + d, l0, w),
            &em.scalars(seed),
        )
        .is_err(),
    );
    // missing scalar
    line(
        "refuse missing declared scalars",
        em.scalars_required.is_empty()
            || bind_draft_ops(&em, &em.shape, &Default::default()).is_err(),
    );
    // live seed patch: seed+1 must not reproduce the eager state
    let drafted = r.eager(l, seed)?;
    r.gpu.hip.device_synchronize()?;
    let e = r.snapshot()?;
    r.restore(&s0)?;
    r.set_thlog(l, l0, l0);
    let bound = bind_draft_ops(&em, &em.shape, &em.scalars(seed + 1))?;
    // Batched route: the seed lives in the staged id plane; scalar route: the
    // `draft_seed` patch. Either way a different window seed must diverge.
    r.stage(l, seed + 1)?;
    r.gpu.run_planned_ops(&bound)?;
    r.gpu.hip.device_synchronize()?;
    let t = r.snapshot()?;
    line("seed+1 diverges (patch live)", !compare(&e, &t).is_empty());
    let _ = drafted;
    // and the right seed reproduces
    r.restore(&s0)?;
    r.set_thlog(l, l0, l0);
    let bound = bind_draft_ops(&em, &em.shape, &em.scalars(seed))?;
    r.stage(l, seed)?;
    r.gpu.run_planned_ops(&bound)?;
    r.gpu.hip.device_synchronize()?;
    let t = r.snapshot()?;
    line("seed reproduces", compare(&e, &t).is_empty());
    Ok(ok)
}

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
}

fn main() -> R<()> {
    let target = arg("--target")
        .or_else(|| std::env::var("HIPFIRE_TEST_TARGET").ok())
        .map(PathBuf::from)
        .ok_or("--target / HIPFIRE_TEST_TARGET required")?;
    let draft = arg("--draft")
        .or_else(|| std::env::var("HIPFIRE_TEST_DRAFT").ok())
        .ok_or("--draft / HIPFIRE_TEST_DRAFT required")?;
    let out = arg("--out").map(PathBuf::from);
    let quick = std::env::args().any(|a| a == "--quick");
    let deltas: Vec<usize> = match arg("--deltas") {
        Some(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        None if quick => vec![1, 2, 15, 16],
        None => (1..=16).collect(),
    };

    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1201" {
        return Err(format!("exact gfx1201 only (this GPU is {})", gpu.arch).into());
    }
    let mut cfg = ModelSlotConfig::default();
    cfg.max_seq = 512;
    cfg.kv_mode = KvMode::Q8;
    let slot = ModelSlot::load(&mut gpu, &target, "draft-emit-oracle", cfg)
        .map_err(|e| format!("load target: {e:?}"))?;
    let kv_q8 = {
        let kv = &slot.kv_cache;
        kv.quant_q8
            && !kv.quant_fwht
            && !kv.quant_asym2
            && !kv.quant_asym3
            && !kv.quant_asym4
            && matches!(kv.v_mode, hipfire_runtime::llama::VMode::Q8)
    };
    let df = load_dflash_state(
        &draft,
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
        return Err("draft has no candidate selector".into());
    }
    let w = df.draft_scratch.ctx_modulus();
    if w == usize::MAX || !df.draft_config.all_layers_sliding {
        return Err("needs an all-sliding windowed DFlash2 draft".into());
    }
    let books = SelectorDeviceCodebooks::upload(&mut gpu, &df.draft_weights)?;
    let blk = SelectorGreedyBlock::new(&mut gpu, &df.draft_weights, DRAFT_ROWS - 1)?;
    let staging = gpu.alloc_tensor_exact(&[4 + DRAFT_ROWS + w + DRAFT_ROWS + 64], DType::F32)?;
    let tokens = gpu.alloc_tensor_exact(&[DRAFT_ROWS], DType::F32)?;
    let pinned = PinnedWindowInputs::new(&gpu, (4 + DRAFT_ROWS + w + DRAFT_ROWS + 64) * 4)?;
    let stream = gpu.hip.stream_create()?;
    let mut rig = Rig {
        gpu,
        slot,
        df,
        books,
        blk,
        staging,
        tokens,
        pinned,
        stream,
        w,
    };
    eprintln!(
        "[draft-emit-oracle] gpu={} W={w} layers={} hidden={} deltas={deltas:?}",
        rig.gpu.arch, rig.df.draft_config.n_layers, rig.df.draft_config.hidden
    );

    let mut report = String::new();
    let mut failed = 0usize;
    let mut total = 0usize;
    let mut idx = 0usize;
    for &d in &deltas {
        let mut targets = vec![
            w / 2,
            w - 1,
            w,
            w + 1,
            w + d.div_ceil(2),
            w + 300,
            2 * w - 1,
            2 * w,
            2 * w + 1,
            2 * w + d.div_ceil(2),
        ];
        if quick {
            targets.retain(|t| [w / 2, w, w + d.div_ceil(2), 2 * w + 1].contains(t));
        }
        targets.sort_unstable();
        targets.dedup();
        for t in targets {
            if t <= d + 1 || t + 32 > CTX {
                continue;
            }
            idx += 1;
            total += 1;
            match run_case(&mut rig, t - d, d, idx) {
                Ok(o) => {
                    eprintln!("{}", o.line);
                    let _ = writeln!(report, "{}", o.line);
                    for f in &o.fails {
                        eprintln!("    {f}");
                        let _ = writeln!(report, "    {f}");
                    }
                    if !o.fails.is_empty() {
                        failed += 1;
                    }
                }
                Err(e) => {
                    failed += 1;
                    let m = format!("L0={} d={d}: ERROR {e}", t - d);
                    eprintln!("{m}");
                    let _ = writeln!(report, "{m}");
                }
            }
        }
    }
    let ctl_ok = if std::env::args().any(|a| a == "--no-controls") {
        true
    } else {
        controls(&mut rig, &mut report)?
    };
    let pass = failed == 0 && ctl_ok && total > 0;
    let verdict = format!(
        "VERDICT {}: {total} cases, {failed} failed, controls {}",
        if pass { "PASS" } else { "FAIL" },
        if ctl_ok { "ok" } else { "FAILED" }
    );
    eprintln!("{verdict}");
    let _ = writeln!(report, "{verdict}");
    if let Some(dir) = out {
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("report.txt"), &report)?;
    }
    std::process::exit(if pass { 0 } else { 1 });
}
