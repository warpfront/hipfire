// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SP3 Task 3 — the correctness gate for `forward_batch_slots` (SP3 Task 2).
// Task 2 was build-only: nothing before this harness has checked that the
// N-slot forward produces the same numbers as the existing single-sequence
// path. This is that check.
//
// COMPARISON IS ON LOGITS, NOT TOKENS. Sampling makes runs diverge, and
// greedy decoding on a quantised model can diverge on a near-tie even when
// both paths are correct — token-level equivalence is not a sound check
// here. For `n_slots` in 1..=4, with per-slot prompts of differing length,
// each slot's full step sequence (one prefill + a few decodes) is run twice:
// once alone through the existing single-sequence `forward_prefill_batch`,
// and once as part of an n_slots-wide batch through `forward_batch_slots`.
// Per-step logits are compared with a tolerance-based `assert_close` —
// copied from `rdna-compute/examples/test_batched_attn_slots.rs` (SP1's
// harness) rather than reinvented, including its two hard-won guards: a
// finiteness check (two independently-NaN arrays would otherwise "agree" at
// a perfect 0.000x) and a non-degeneracy check that rejects an all-zero
// reference (SP1 found two all-zero arrays passing at 0.000x tolerance).
//
// KV-MODE MATRIX. The sweep runs once per requested KV tier (default: q8,
// fwht{2,3,4}), resolving each name through `QWEN35_SLOTS_POLICY`
// exactly as the rig does. The reference arm builds its sequential KvCache
// with the SAME mode-generic production constructor the carrier uses
// (`KvCacheExt::from_mode_with_backend`), and the candidate arm builds its
// `SlotKvTier` tables with the same seeds as the sequential constructors
// (`gen_givens_angles(42,·)` / `gen_fwht_signs(42|1042,·)`), which is the
// bit-identical-packed-bytes contract `kv_write_slots` rests on. f16 is NOT
// in the matrix: it is a slots-only tier by design (no sequential qwen35
// f16 constructor exists to reference against). fp8 is not either: it
// resolves cleanly through the slots policy but the kernels are
// gfx1201-only (`#error` below gfx1201) and the rig gate refuses it on
// every other arch — this gfx1101 host cannot exercise it. The `asymN`
// names resolve to their post-0.4.0 `fwhtN` aliases.
//
// NEGATIVE CONTROL. `run_negative_control` redirects one row's `row_slot`
// entry in the CANDIDATE arm's `SlotBatch` — the harness-level analogue of
// "two slots pointed at the same SlotId" — and asserts the resulting
// mismatch against the (uncorrupted) single-sequence reference. The
// corruption is applied at a DECODE step where the two slots' absolute
// positions have already diverged (6 vs 9). This is deliberate: if the
// redirected row happened to share its target slot's KV write offset with a
// real, un-redirected row from that slot (the case at equal-length
// equal-position steps), the two rows would race to write the same slab
// bytes and, depending on which one lands last, the corrupted read could
// accidentally come back correct — defeating the control the way SP1's
// *first* attempt at this kind of check did (see
// `test_batched_attn_slots.rs`'s `maybe_corrupt` doc comment: corrupting
// both arms let a wrong answer agree with itself). Distinct lengths ensure
// every step's absolute positions are distinct across slots, so a redirected
// write always lands at an offset no real row is also writing to that step,
// and the redirected read is unambiguously wrong. (Before the KV-matrix
// rework this control had gone VACUOUS: it corrupted at step 2 while
// DECODE_STEPS had been set to 0, so the "expected failure" was an
// index-out-of-bounds panic, not a tolerance mismatch. It now runs its own
// explicit prefill + one-decode sequence under every mode.)
//
// SCOPE. `forward_batch_slots` admits dense and MoE layers per SITE via
// `plan_proj_group` (mixed tier/fixed-tier recipes included; see its module
// doc), so any DENSE Qwen3.5/3.6 checkpoint works here — uniform Q8_0, MQ4,
// MQ4V2 or a mixed recipe. MoE checkpoints also load but are exercised by
// the serve harnesses instead. Verified against `qwen3.5-4b-q8.hf4` (Q8)
// and `qwen3.6-27b.mq4` (MQ4, 64 layers). If a different model is passed
// that doesn't meet that bar, `forward_batch_slots` returns a precise
// `HipError` describing exactly which requirement failed, which this
// harness surfaces via `.expect(..)` rather than papering over.
//
// Usage:
//   cargo run --release -p hipfire-runtime --features deltanet,arch-qwen35 \
//     --example test_forward_slots_golden -- <model.hf4> [kv-modes]
//
//   kv-modes  comma list from {q8,asym2,asym3,asym4,fwht2,fwht3,fwht4,
//             bf16,f16}; default: all of them.
//   SLOTS_GOLDEN_PAGED=1  run the whole matrix against a paged pool
//             (PagePool block tables) instead of the legacy slab pool.
//
// Run only through `scripts/run-bounded.sh`, and only when no daemon holds a
// model resident and MemAvailable is comfortably above what this harness
// plans to use (see the preflight computation in `main` below) — an
// under-provisioned run on this box has previously triggered a GLOBAL OOM
// that killed unrelated user processes, because the cgroup does not contain
// amdgpu GTT allocations.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet,arch-qwen35");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::forward_slots::{forward_batch_slots, SlotDescStaging, SlotKvTier};
    use hipfire_arch_qwen35::qwen35::{
        self, DeltaNetState, LayerType, PrefillBatchScratch, Qwen35Config, Qwen35Scratch,
        Qwen35Weights,
    };
    use hipfire_runtime::kv_backend::KvBackend;
    use hipfire_runtime::kv_mode::{self, KvMode, SlotKvTierPlan};
    use hipfire_runtime::llama::{KvCache, KvCacheExt, KvDims, KvLayers, KvTarget};
    use hipfire_runtime::slot_batch::SlotBatch;
    use hipfire_runtime::hfq::HfqFile;
    use rdna_compute::kv_slots::{preflight_alloc, R9700_VRAM_BYTES};
    use rdna_compute::page_pool::PAGE_TOKENS;
    use rdna_compute::slot_pool::{SlotId, SlotPool};
    use rdna_compute::{DType, Gpu, GpuTensor};
    use std::path::Path;

    const N_SLOTS_MAX: usize = 4;
    const DECODE_STEPS: usize = 0; // was 3 — see the mrope decode note below
                                   // SlotPool rounds cap_tokens up to a multiple of 128 internally, so this
                                   // just needs to clear every prompt length + DECODE_STEPS (max 9 + 3).
    const CAP_TOKENS: usize = 64;
    // Distinct, deliberately non-tile-aligned per-slot prompt lengths —
    // exercises the LDS-decode kernel's M>1 ("verify"-shaped) path on the
    // prefill step for every n_slots in the sweep.
    const PROMPT_LENS: [usize; N_SLOTS_MAX] = [5, 9, 3, 7];
    // Default excludes bf16: the slots bf16 kernels are bit-exact vs the
    // sequential ones at kernel level (rdna-compute/examples/
    // test_bf16_slots_parity.rs: attend zero-base/non-zero-base/mixed and
    // write parity all BIT-EXACT on gfx1101), but end-to-end logits still
    // diverge ~6x this harness's tolerance on one element of the vocab
    // (stable across the pre- and post-parity-fix attend arm — measured
    // 2026-10-04, gfx1101, qwen3.5-4b.mq4v2.hfq). The sequential reference's
    // bf16 attend selection is itself a heuristic (kv_tier.rs
    // `bf16_attend_key`: flash_mode/capture-dependent), so matching arms is
    // not sufficient for bit-parity; root-cause is a tracked follow-up
    // (docs/plans/slots-quant-recipe-admission.md §9). Requesting bf16
    // explicitly still runs it — expect the tolerance assert to fire until
    // that follow-up lands.
    const DEFAULT_MODES: &str = "q8,fwht2,fwht3,fwht4";

    /// One KV tier the matrix sweeps: the policy-resolved mode plus its slot
    /// arena geometry. Built exactly the way `Rig::build` builds them.
    struct ModeFixture {
        name: String,
        mode: KvMode,
        plan: SlotKvTierPlan,
    }

    /// Resolve mode NAMES through the slots policy only (host-side, no
    /// geometry yet); the tier plan is resolved in `main` once the config's
    /// head geometry is known. F16 is skipped with a note: it is a
    /// slots-only tier BY DESIGN (saddle-core kv.rs: "F16 flat rows exist
    /// only on the slots engine") — no sequential qwen35 f16 cache
    /// constructor exists, so no golden reference can be built for it.
    /// (The `asymN` names are still accepted: they alias to `fwhtN` since
    /// 0.4.0 and exercise the alias table on their way through resolve.)
    fn resolve_mode_names(spec: &str) -> Vec<(String, KvMode)> {
        spec.split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .filter_map(|name| {
                let kv_mode::ResolveResult { mode, warning } =
                    kv_mode::resolve(name, &kv_mode::QWEN35_SLOTS_POLICY);
                assert!(
                    warning.is_none(),
                    "mode {name:?} did not cleanly resolve through QWEN35_SLOTS_POLICY: {:?}",
                    warning
                );
                if mode == KvMode::F16 {
                    eprintln!(
                        "note: skipping kv mode {name:?} — F16 is slots-only by design; \
                         no sequential reference exists to compare against"
                    );
                    return None;
                }
                Some((name.to_string(), mode))
            })
            .collect()
    }

    fn prompt_lens(n_slots: usize) -> Vec<usize> {
        PROMPT_LENS[..n_slots].to_vec()
    }

    /// Small, varied, collision-avoiding token ids. `salt` lets the negative
    /// control use a dataset disjoint from the golden sweep's, though
    /// nothing depends on that beyond making printed output easier to read.
    fn deterministic_token(slot: usize, idx: usize, salt: u32) -> u32 {
        ((slot as u32)
            .wrapping_mul(733)
            .wrapping_add((idx as u32).wrapping_mul(131))
            .wrapping_add(salt)
            % 900)
            + 1
    }

    /// One token stream per slot, long enough to cover its prompt plus every
    /// decode step. Reference and candidate both slice from the SAME stream,
    /// so any divergence between them is attributable to the forward path,
    /// never to the two arms seeing different input.
    fn build_token_stream(lens: &[usize], salt: u32) -> Vec<Vec<u32>> {
        lens.iter()
            .enumerate()
            .map(|(s, &plen)| {
                (0..plen + DECODE_STEPS + 1)
                    .map(|i| deterministic_token(s, i, salt))
                    .collect()
            })
            .collect()
    }

    fn build_prefill_batch(lens: &[usize], streams: &[Vec<u32>]) -> SlotBatch {
        let triples: Vec<(SlotId, &[u32], usize)> = (0..lens.len())
            .map(|s| (SlotId(s), &streams[s][..lens[s]], 0usize))
            .collect();
        SlotBatch::build(&triples)
    }

    fn build_decode_batch(
        lens: &[usize],
        streams: &[Vec<u32>],
        positions: &[usize],
    ) -> SlotBatch {
        let triples: Vec<(SlotId, &[u32], usize)> = (0..lens.len())
            .map(|s| {
                let start = positions[s];
                (SlotId(s), &streams[s][start..start + 1], start)
            })
            .collect();
        SlotBatch::build(&triples)
    }

    /// SP1's hardened comparator (`test_batched_attn_slots.rs::assert_close`),
    /// reused verbatim rather than reinvented per the brief. Two guards
    /// beyond a plain tolerance loop: a finiteness check (NaN vs NaN
    /// compares false in both directions and would otherwise silently
    /// "pass"), and a non-degeneracy check that refuses an all-zero
    /// reference (SP1 found two all-zero arrays passing at 0.000x).
    fn assert_close(label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        if let Some(i) = got.iter().position(|v| !v.is_finite()) {
            panic!(
                "{label}: candidate[{i}]={} is non-finite (want[{i}]={})",
                got[i], want[i]
            );
        }
        if let Some(i) = want.iter().position(|v| !v.is_finite()) {
            panic!(
                "{label}: reference[{i}]={} is non-finite (got[{i}]={})",
                want[i], got[i]
            );
        }
        if !want.is_empty() {
            assert!(
                want.iter().any(|v| v.abs() > 0.0),
                "{label}: reference array is all-zero ({} elements) — a kernel that wrote \
                 nothing would pass this comparison by accident; refusing to treat an \
                 all-zero pair as agreement",
                want.len()
            );
        }
        let mut worst = 0.0f32;
        let mut worst_i = 0usize;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let tol = 1e-3 * w.abs().max(1.0);
            let err = (g - w).abs() / tol;
            if err > worst {
                worst = err;
                worst_i = i;
            }
        }
        assert!(
            worst <= 1.0,
            "{label}: worst element {worst_i} at {worst:.2}x tolerance (got {}, want {})",
            got[worst_i],
            want[worst_i]
        );
        println!("  {label}: OK (worst {worst:.3}x tolerance)");
    }

    /// Build the rig's `SlotKvTier` for one mode: rotation tables uploaded
    /// with the same seeds the sequential constructors use (see
    /// `serve_engine`'s identical block). Caller frees via
    /// `SlotKvTier::free_gpu`.
    fn build_slot_kv_tier(gpu: &mut Gpu, fx: &ModeFixture) -> SlotKvTier {
        let upload_f32 = |gpu: &mut Gpu, vals: &[f32]| -> GpuTensor {
            let t = gpu
                .alloc_tensor(&[vals.len()], DType::F32)
                .expect("tier table alloc");
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
            gpu.hip
                .memcpy_htod(&t.buf, &bytes)
                .expect("tier table upload");
            t
        };
        let (cos, sin, s1, s2) = if let Some(len) = fx.plan.givens_len {
            let (c, si) = KvCache::gen_givens_angles(42, len);
            (
                Some(upload_f32(gpu, &c)),
                Some(upload_f32(gpu, &si)),
                None,
                None,
            )
        } else if let Some(len) = fx.plan.fwht_len {
            (
                None,
                None,
                Some(upload_f32(gpu, &KvCache::gen_fwht_signs(42, len))),
                Some(upload_f32(gpu, &KvCache::gen_fwht_signs(1042, len))),
            )
        } else {
            (None, None, None, None)
        };
        SlotKvTier {
            mode: fx.mode,
            givens_cos: cos,
            givens_sin: sin,
            fwht_signs1: s1,
            fwht_signs2: s2,
        }
    }

    /// Run one slot's full step sequence (prefill + DECODE_STEPS decodes)
    /// alone through the existing single-sequence path, recording per-step
    /// last-token logits. Owns and frees its own KvCache/DeltaNetState/
    /// Qwen35Scratch — callers run this once per slot, never more than one
    /// live at a time (unlike the candidate arm, which genuinely needs all
    /// slots live at once).
    fn run_reference_for_slot(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        fx: &ModeFixture,
        is_kv_layer: &[bool],
        stream: &[u32],
        prompt_len: usize,
        extra_decode: bool,
    ) -> Vec<Vec<f32>> {
        let kv_seq = (prompt_len + DECODE_STEPS + 16).max(CAP_TOKENS).max(512);
        // The SAME mode-generic constructor the sequential carrier uses
        // (carrier.rs `construct_kv_cache`), so the reference is production-
        // true for every tier in the matrix — not a hand-picked per-mode
        // constructor that could drift from what users actually run.
        let mut kv_cache = <KvCache as KvCacheExt>::from_mode_with_backend(
            fx.mode,
            KvBackend::Legacy,
            KvTarget::Single(gpu),
            &KvDims {
                layers: KvLayers::Mask(is_kv_layer.to_vec()),
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
                max_seq: kv_seq,
                physical_cap: Some(kv_seq),
            },
        )
        .expect("reference: KvCache::from_mode_with_backend");
        let mut dn_state = DeltaNetState::new(gpu, config).expect("reference: DeltaNetState::new");
        let scratch = Qwen35Scratch::new_with_kv_max(gpu, config, 128, kv_seq)
            .expect("reference: Qwen35Scratch::new_with_kv_max");

        let mut steps = Vec::with_capacity(1 + DECODE_STEPS + usize::from(extra_decode));

        qwen35::forward_prefill_batch(
            gpu,
            weights,
            config,
            &stream[..prompt_len],
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            None,
            None,
            None,
        )
        .expect("reference: prefill forward");
        gpu.hip.device_synchronize().expect("sync");
        steps.push(
            gpu.download_f32(&scratch.logits)
                .expect("reference: download logits"),
        );

        // DECODE_STEPS is currently 0 ("was 3"), so this range folds to 0..0;
        // the loop must stay for the day the mrope decode steps come back.
        // `extra_decode` (the negative control) adds ONE explicit decode step
        // through the canonical decode entry point.
        #[allow(clippy::reversed_empty_ranges)]
        for k in 0..DECODE_STEPS + usize::from(extra_decode) {
            // `forward_scratch`, NOT forward_prefill_batch with a 1-token slice.
            //
            // Both accept a single token, but they are not interchangeable: the
            // single-token batched path segfaults on this model inside
            // gated_delta_net_q8_compact2. `forward_scratch` is the canonical
            // decode entry point -- it is what daemon.rs uses (see its decode
            // sites) -- and it does the `ensure_mapped_capacity` growth that the
            // batched path does not do for a 1-token call.
            qwen35::forward_scratch(
                gpu,
                weights,
                config,
                stream[prompt_len + k],
                prompt_len + k,
                &mut kv_cache,
                &mut dn_state,
                &scratch,
            )
            .expect("reference: decode forward");
            gpu.hip.device_synchronize().expect("sync");
            steps.push(
                gpu.download_f32(&scratch.logits)
                    .expect("reference: download logits"),
            );
        }

        kv_cache.free_gpu(gpu).expect("reference: free kv_cache");
        dn_state.free_gpu(gpu);
        scratch.free_gpu(gpu);
        steps
    }

    /// Run every slot together through `forward_batch_slots`, one call per
    /// step (prefill, then DECODE_STEPS decodes), recording per-slot,
    /// per-step last-token logits.
    ///
    /// `corrupt = Some((step_idx, victim, target))` relabels every row
    /// belonging to slot `victim` as slot `target` in ONLY the `step_idx`-th
    /// call's `SlotBatch.row_slot` — the negative control's sole hook, and
    /// it touches nothing the reference arm reads (the reference never
    /// builds a `SlotBatch` at all).
    ///
    /// All candidate-arm GPU tensors (arenas, tier tables, DeltaNet state,
    /// scratch) are allocated fresh at entry and freed before returning —
    /// nothing is held live across calls, per the "free per-iteration GPU
    /// tensors" rule that a prior sweep in this project violated its way
    /// into an OOM.
    #[allow(clippy::too_many_arguments)]
    fn run_candidate(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        fx: &ModeFixture,
        n_fa_layers: usize,
        lens: &[usize],
        streams: &[Vec<u32>],
        corrupt: Option<(usize, usize, usize)>,
        extra_decode: bool,
        paged: bool,
    ) -> Vec<Vec<Vec<f32>>> {
        let n_slots = lens.len();
        let max_batch = lens.iter().sum::<usize>();

        // The pool geometry follows the tier plan exactly as `Rig::build`
        // does: independent K/V strides, and (paged mode) a page pool whose
        // capacity matches the legacy total.
        let mut pool = if paged {
            let n_pages = n_slots * CAP_TOKENS.div_ceil(PAGE_TOKENS);
            SlotPool::new_paged_with_strides(
                n_slots,
                CAP_TOKENS,
                fx.plan.k_bytes_per_pos,
                fx.plan.v_bytes_per_pos,
                n_pages,
            )
            .expect("candidate: SlotPool::new_paged_with_strides")
        } else {
            SlotPool::new_with_strides(
                n_slots,
                CAP_TOKENS,
                fx.plan.k_bytes_per_pos,
                fx.plan.v_bytes_per_pos,
            )
            .expect("candidate: SlotPool::new_with_strides")
        };
        for s in 0..n_slots {
            let id = pool.acquire().expect("candidate: SlotPool::acquire");
            assert_eq!(
                id.0, s,
                "SlotPool handed out slots out of order — this harness's row_slot values \
                 assume acquire() returns 0, 1, 2, ... in order"
            );
        }

        let k_arena_bytes = pool.k_arena_bytes();
        let v_arena_bytes = pool.v_arena_bytes();
        let k_arenas: Vec<GpuTensor> = (0..n_fa_layers)
            .map(|_| {
                gpu.zeros(&[k_arena_bytes], DType::Raw)
                    .expect("candidate: alloc k_arena")
            })
            .collect();
        let v_arenas: Vec<GpuTensor> = (0..n_fa_layers)
            .map(|_| {
                gpu.zeros(&[v_arena_bytes], DType::Raw)
                    .expect("candidate: alloc v_arena")
            })
            .collect();
        let mut dn_states: Vec<DeltaNetState> = (0..n_slots)
            .map(|_| DeltaNetState::new(gpu, config).expect("candidate: DeltaNetState::new"))
            .collect();
        let max_pages_per_slot = if paged {
            pool.cap_tokens().div_ceil(PAGE_TOKENS)
        } else {
            0
        };
        let mut desc_staging =
            SlotDescStaging::new(gpu, n_slots, max_batch, max_pages_per_slot)
                .expect("candidate: SlotDescStaging::new");
        let kv_tier = build_slot_kv_tier(gpu, fx);
        let pbs = PrefillBatchScratch::new(gpu, config, max_batch)
            .expect("candidate: PrefillBatchScratch::new");
        let scratch = Qwen35Scratch::new_with_kv_max(gpu, config, 64, CAP_TOKENS)
            .expect("candidate: Qwen35Scratch::new_with_kv_max");
        let logits_out = gpu
            .zeros(&[n_slots * config.vocab_size], DType::F32)
            .expect("candidate: alloc logits_out");

        let mut per_slot_steps: Vec<Vec<Vec<f32>>> =
            vec![Vec::with_capacity(1 + DECODE_STEPS); n_slots];

        for step_idx in 0..=DECODE_STEPS + usize::from(extra_decode) {
            let mut batch = if step_idx == 0 {
                build_prefill_batch(lens, streams)
            } else {
                let positions: Vec<usize> =
                    lens.iter().map(|&l| l + step_idx - 1).collect();
                build_decode_batch(lens, streams, &positions)
            };
            if let Some((cstep, victim, target)) = corrupt {
                if step_idx == cstep {
                    // Redirect victim's rows to the target's descriptor AND
                    // keep the batch internally consistent (the engine's
                    // `advance_slot_seq_lens` asserts every slot with
                    // m_per_slot > 0 owns at least one row): the victim's
                    // row count moves to the target. The corruption is that
                    // the victim's row — carrying the victim's absolute
                    // position — is now written and attended under the
                    // TARGET's slot descriptor, and the victim itself runs
                    // no rows that step, so its logits go stale.
                    for rs in batch.row_slot.iter_mut() {
                        if *rs == victim as i32 {
                            *rs = target as i32;
                        }
                    }
                    let m = batch.m_per_slot[victim];
                    batch.m_per_slot[victim] = 0;
                    batch.m_per_slot[target] += m;
                }
            }

            forward_batch_slots(
                gpu,
                weights,
                config,
                &batch,
                &mut pool,
                &mut dn_states,
                &k_arenas,
                &v_arenas,
                &mut desc_staging,
                &kv_tier,
                &pbs,
                &scratch,
                &logits_out,
            )
            .expect("candidate: forward_batch_slots");
            gpu.hip.device_synchronize().expect("sync");

            // Keep desc.seq_len in sync with each slot's true history,
            // regardless of any row_slot corruption applied above — this
            // reflects the REAL logical length of each slot's own KV, per
            // kv_slot_desc.h's documented invariant ("kernel reads
            // [0, seq_len)"), not whatever the corrupted call happened to
            // address this step.
            for s in 0..n_slots {
                pool.set_seq_len(SlotId(s), lens[s] + step_idx)
                    .expect("candidate: set_seq_len");
            }

            let flat = gpu
                .download_f32(&logits_out)
                .expect("candidate: download logits_out");
            for s in 0..n_slots {
                per_slot_steps[s]
                    .push(flat[s * config.vocab_size..(s + 1) * config.vocab_size].to_vec());
            }
        }

        for t in k_arenas {
            gpu.free_tensor(t).expect("candidate: free k_arena");
        }
        for t in v_arenas {
            gpu.free_tensor(t).expect("candidate: free v_arena");
        }
        for dn in dn_states {
            dn.free_gpu(gpu);
        }
        kv_tier.free_gpu(gpu);
        desc_staging.free_gpu(gpu);
        pbs.free_gpu(gpu);
        scratch.free_gpu(gpu);
        gpu.free_tensor(logits_out)
            .expect("candidate: free logits_out");

        per_slot_steps
    }

    #[allow(clippy::too_many_arguments)]
    fn run_golden_equivalence(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        fx: &ModeFixture,
        is_kv_layer: &[bool],
        n_slots: usize,
        n_fa_layers: usize,
        paged: bool,
    ) -> (usize, usize) {
        let lens = prompt_lens(n_slots);
        let streams = build_token_stream(&lens, 0);
        println!("-- n_slots={n_slots} prompt_lens={lens:?}");

        let reference: Vec<Vec<Vec<f32>>> = (0..n_slots)
            .map(|s| {
                run_reference_for_slot(
                    gpu,
                    weights,
                    config,
                    fx,
                    is_kv_layer,
                    &streams[s],
                    lens[s],
                    false,
                )
            })
            .collect();
        let candidate = run_candidate(
            gpu,
            weights,
            config,
            fx,
            n_fa_layers,
            &lens,
            &streams,
            None,
            false,
            paged,
        );

        let mut n_ok = 0usize;
        let n_total = n_slots * (1 + DECODE_STEPS);
        for s in 0..n_slots {
            for step in 0..=DECODE_STEPS {
                let label = format!("n_slots={n_slots} slot={s} step={step}");
                assert_close(&label, &candidate[s][step], &reference[s][step]);
                n_ok += 1;
            }
        }
        (n_ok, n_total)
    }

    /// Prove the comparison can actually fail, under the CURRENT tier. Two
    /// slots of UNEQUAL length (6 and 9), prefill then ONE decode step, with
    /// slot 1's decode row redirected to slot 0's descriptor (absolute
    /// positions 6 vs 9 by then — see this file's header comment for why
    /// unequal positions matter). The assertion failure must be a TOLERANCE
    /// mismatch: an index-out-of-bounds or other harness panic is NOT a
    /// passing control (that vacuity is exactly what the pre-matrix version
    /// of this check had regressed into).
    #[allow(clippy::too_many_arguments)]
    fn run_negative_control(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        fx: &ModeFixture,
        is_kv_layer: &[bool],
        n_fa_layers: usize,
        paged: bool,
    ) {
        println!(
            "\n=== negative control (kv={}): candidate arm's row_slot corrupted (slot 1 -> slot 0 at the decode step) ===",
            fx.name
        );
        let lens = vec![6usize, 9usize];
        let streams = build_token_stream(&lens, 1); // salt=1: a dataset distinct from the golden sweep's

        let reference_slot1 = run_reference_for_slot(
            gpu,
            weights,
            config,
            fx,
            is_kv_layer,
            &streams[1],
            lens[1],
            true,
        );

        let corrupt_step = 1usize; // the decode step: slot 0 @ pos 6, slot 1 @ pos 9
        let candidate = run_candidate(
            gpu,
            weights,
            config,
            fx,
            n_fa_layers,
            &lens,
            &streams,
            Some((corrupt_step, 1, 0)),
            true,
            paged,
        );

        // This comparison is EXPECTED to fail on tolerance — suppress the
        // default panic hook's stderr spam for the duration of the probe,
        // then restore it.
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_close(
                "negative control (slot 1, row_slot redirected to slot 0 at the decode step)",
                &candidate[1][corrupt_step],
                &reference_slot1[corrupt_step],
            );
        }));
        std::panic::set_hook(default_hook);

        match result {
            Ok(()) => panic!(
                "negative control did NOT fail: a candidate arm with slot 1's row \
                 redirected to slot 0's KV descriptor still matched slot 1's true \
                 single-sequence reference within tolerance. The comparison is not \
                 sensitive to a misrouted slot — see SP1's task-7 review (91.44x \
                 tolerance on the corrected control) for the shape of what this is \
                 supposed to catch."
            ),
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "<non-string panic payload>".to_string());
                assert!(
                    msg.contains("tolerance") || msg.contains("worst"),
                    "negative control failed for the WRONG reason (expected a \
                     tolerance mismatch, got): {msg}"
                );
                println!("  negative control correctly failed: {msg}");
            }
        }
    }

    // ────────────────────────────────── main ──────────────────────────────────

    let mut args = std::env::args().skip(1);
    let model_path = args.next().unwrap_or_else(|| {
        eprintln!(
            "Usage: test_forward_slots_golden <model.hf4> [kv-modes]  (a DENSE \
             Qwen3.5 checkpoint; kv-modes: comma list from q8,asym2,asym3,asym4,\
             fwht2,fwht3,fwht4,bf16,f16 — default all)"
        );
        std::process::exit(1);
    });
    let modes_spec = args.next().unwrap_or_else(|| DEFAULT_MODES.to_string());
    let paged = std::env::var("SLOTS_GOLDEN_PAGED").ok().as_deref() == Some("1");

    // ---- host-only setup: open the file and parse config before any GPU
    // allocation, so the preflight check below can be computed from real
    // config values rather than guessed constants. ----
    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("parse Qwen3.5 config");
    let n_fa_layers = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::FullAttention)
        .count();
    let n_delta_layers = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();

    // Resolve every mode up front (host-side); the tier plans come from the
    // config's real head geometry. Under the PAGED pool, q8 is skipped: the
    // single-slot/multi-slot WMMA fast paths are legacy-slab-only
    // (`!pool.is_paged()` in q8_attend_slots), so a paged q8 run attends
    // through the tiled kernel while the sequential reference uses the WMMA
    // family — mathematically equivalent, numerically different (~18x this
    // harness's tolerance on the worst element, measured 2026-10-04), so a
    // bit-parity comparison cannot pass and would only mask the property.
    // Paged q8's kernels are covered at the kernel level by
    // rdna-compute/examples/test_kv_slot_desc_ports.rs and
    // test_batched_attn_slots.rs.
    let fixtures: Vec<ModeFixture> = resolve_mode_names(&modes_spec)
        .into_iter()
        .filter(|(name, mode)| {
            if paged && *mode == KvMode::Q8 {
                eprintln!(
                    "note: skipping kv mode {name:?} under the paged pool — the q8 WMMA attend \
                     family is legacy-slab-only, so paged q8 uses the tiled kernel and is \
                     not bit-comparable to the sequential WMMA reference"
                );
                return false;
            }
            true
        })
        .map(|(name, mode)| {
            let plan = SlotKvTierPlan::resolve(mode, config.n_kv_heads, config.head_dim)
                .unwrap_or_else(|e| panic!("kv mode {name}: {e}"));
            ModeFixture { name, mode, plan }
        })
        .collect();
    println!(
        "kv matrix: {} (pool: {})",
        fixtures.iter().map(|f| f.name.clone()).collect::<Vec<_>>().join(", "),
        if paged { "paged" } else { "legacy slab" }
    );
    assert!(
        !fixtures.is_empty(),
        "no testable kv modes left after resolution (all skipped?)"
    );

    // ---- preflight: itemized, not a magic number. A prior harness in this
    // project undercounted by ~30% by omitting host-side Vecs; this adds up
    // every device AND host allocation this run holds live at once, at its
    // worst case (n_slots=N_SLOTS_MAX, at the widest tier strides). ----
    let weight_bytes = std::fs::metadata(&model_path)
        .expect("stat model file")
        .len();
    let cap_rounded = CAP_TOKENS.div_ceil(128) * 128;

    let max_k_per_pos = fixtures.iter().map(|f| f.plan.k_bytes_per_pos).max().unwrap();
    let max_v_per_pos = fixtures.iter().map(|f| f.plan.v_bytes_per_pos).max().unwrap();

    // Candidate arm: K+V arenas across every FullAttention layer, sized for
    // N_SLOTS_MAX at the widest tier's strides and held live for that
    // iteration of the sweep.
    let candidate_kv_bytes = (n_fa_layers as u64)
        * (max_k_per_pos as u64 + max_v_per_pos as u64)
        * (N_SLOTS_MAX as u64)
        * (cap_rounded as u64);

    // Candidate arm: one DeltaNetState per slot (s_matrices Q8 1B/elem +
    // s_scales f32 + s_ef_residual f16 + conv_states f32), N_SLOTS_MAX held
    // live at once — mirrors DeltaNetState::new_with_quant's own arithmetic.
    let dn_s_dim = config.linear_key_head_dim;
    let dn_heads = config.linear_num_value_heads;
    let dn_s_size = dn_heads * dn_s_dim * dn_s_dim;
    let dn_conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let dn_conv_state_size = dn_conv_channels * config.conv_kernel_dim.saturating_sub(1);
    let per_slot_dn_bytes = (n_delta_layers as u64)
        * (dn_s_size as u64
            + (dn_heads * dn_s_dim) as u64 * 4
            + dn_s_size as u64 * 2
            + dn_conv_state_size as u64 * 4);
    let candidate_dn_bytes = (N_SLOTS_MAX as u64) * per_slot_dn_bytes;

    // Reference arm: one single-sequence KvCache (masked to the FA layers,
    // as the carrier builds it) + DeltaNetState + Qwen35Scratch alive at a
    // time (freed between slots) — budget one at the widest strides.
    let ref_kv_bytes = (n_fa_layers as u64)
        * (max_k_per_pos as u64 + max_v_per_pos as u64)
        * (cap_rounded as u64);
    let reference_bytes = ref_kv_bytes + per_slot_dn_bytes + 64 * 1024 * 1024;

    // Host-side logits downloads, summed across the whole run rather than
    // assumed O(1) — every `download_f32(&scratch.logits)` /
    // `download_f32(&logits_out)` call this harness makes. Two extra
    // reference decodes + candidate steps per mode come from the negative
    // controls.
    let n_modes = fixtures.len();
    let golden_ref_downloads: usize =
        (1..=N_SLOTS_MAX).sum::<usize>() * (1 + DECODE_STEPS) * n_modes;
    let golden_cand_downloads: usize =
        N_SLOTS_MAX * (1 + DECODE_STEPS) * n_modes + 2 * 2 * n_modes;
    let host_logit_bytes = (golden_ref_downloads + golden_cand_downloads) as u64
        * (config.vocab_size as u64)
        * 4;

    let planned = weight_bytes
        + candidate_kv_bytes
        + candidate_dn_bytes
        + reference_bytes
        + host_logit_bytes
        + 256 * 1024 * 1024; // PrefillBatchScratch / SlotDescStaging / Qwen35Scratch misc, flat slop

    eprintln!(
        "preflight: weights={:.2} GiB, candidate_kv={:.1} MiB, candidate_dn={:.1} MiB, \
         reference={:.1} MiB, host_logits={:.1} MiB, planned={:.2} GiB",
        weight_bytes as f64 / 1073741824.0,
        candidate_kv_bytes as f64 / 1048576.0,
        candidate_dn_bytes as f64 / 1048576.0,
        reference_bytes as f64 / 1048576.0,
        host_logit_bytes as f64 / 1048576.0,
        planned as f64 / 1073741824.0,
    );
    preflight_alloc(planned, R9700_VRAM_BYTES, "test_forward_slots_golden")
        .expect("preflight_alloc refused this configuration");

    let mut gpu = Gpu::init().expect("gpu init");
    let weights: Qwen35Weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");

    println!(
        "model: {} layers ({} FullAttention / {} LinearAttention), vocab={}",
        config.n_layers, n_fa_layers, n_delta_layers, config.vocab_size
    );

    let mut n_ok = 0usize;
    let mut n_total = 0usize;
    for fx in &fixtures {
        println!("\n===== kv tier: {} ({:?}) =====", fx.name, fx.mode);
        for n_slots in 1..=N_SLOTS_MAX {
            let (ok, total) = run_golden_equivalence(
                &mut gpu,
                &weights,
                &config,
                fx,
                &is_kv_layer,
                n_slots,
                n_fa_layers,
                paged,
            );
            n_ok += ok;
            n_total += total;
        }

        run_negative_control(
            &mut gpu,
            &weights,
            &config,
            fx,
            &is_kv_layer,
            n_fa_layers,
            paged,
        );
    }

    weights.free_gpu(&mut gpu);

    println!("\n{n_ok}/{n_total} slot-steps passed golden equivalence across {} kv tiers.",
        fixtures.len());
    println!("ALL CHECKS PASS");
}
