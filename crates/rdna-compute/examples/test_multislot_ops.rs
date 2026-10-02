// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SP2 Task 6 — the correctness gate for SP2 (multi-slot state and per-slot
// ops). Tasks 3 (KV cache write slot axis) and 4 (DeltaNet slot stride) were
// COMPILE-ONLY when they landed — the box had a serve daemon resident and no
// GPU harness was run. This is the first thing to exercise either of those
// two components on real hardware.
//
// Scope: KV write and DeltaNet only, per the brief's Step 1 text ("For slot
// counts 1-8, and for each of KV write and DeltaNet: ..."). `sample_per_slot`
// (Task 5) already has its own unit-level coverage (`params_struct_is_16_bytes_repr_c`,
// `all_greedy_is_detectable_as_a_fast_path` in sampling.rs) and doesn't touch
// GPU-addressed shared state the way KV write's descriptor table or
// DeltaNet's strided S buffer do, so it is not re-tested here. RoPE (Task 2)
// needed no code change. `SlotPool` (Task 1) is exercised directly below —
// the KV-write descriptor table is built through it, not hand-rolled.
//
// Four layers per component, mirroring `test_batched_attn_slots.rs` (SP1
// Task 7) structurally, but each one re-derived for what these two ops
// actually are rather than copied verbatim:
//
//   1. Golden equivalence — multi-slot op in one call (or, for DeltaNet,
//      the slot-scoped launch sequence SP2 actually specifies) vs. the
//      existing single-sequence op run per slot. `assert_close` rejects an
//      all-zero reference (SP1 found two all-zero arrays pass at 0.000x)
//      and panics loudly on any non-finite element instead of silently
//      scoring a shared-NaN pair as a match.
//
//      DeltaNet note: the original design gave every slot's launch a
//      *shared*, `s_stride_elems`-strided S buffer plus a device-side
//      `row_slot` selecting the offset. That design was found unsound
//      before any caller shipped (`s_q8` and `s_scales` differ by a factor
//      of HD, and `s_ef_residual` was never strided at all — see
//      `gated_delta_net_q8_batch_seq_slots`'s doc in norm.rs) and was
//      retired: `s_stride_elems` must now be 0. DeltaNet state is
//      fixed-size and per-slot independent, so each slot instead gets its
//      OWN `s_q8`/`s_scales`/EF device buffers and the harness calls the
//      plain `gated_delta_net_q8_batch_seq` (no row_slot, no stride) —
//      exactly what SP3's one-`DeltaNetState`-per-slot model already does.
//
//   2. Cross-slot isolation, with a positive poison control. KV write has
//      no arena READ path (it only ever writes), so "poison every slot
//      except the target" is translated as: pre-fill the WHOLE destination
//      arena with an NaN Q8_0 sentinel, write only the target slot's rows,
//      and require every OTHER slot's slab to still decode as 100% NaN —
//      i.e. the write did not leak outside its own k_base..k_base+cap
//      window. DeltaNet's slot-scoped launch DOES read its own S state on
//      entry, so its isolation test poisons every OTHER slot's OWN device
//      buffer, runs the target's launch against the target's OWN
//      (unpoisoned) buffer, and requires the output stays finite. With
//      genuinely separate per-slot buffers there is no shared array left
//      for a neighbour's poison to leak through, so this no longer probes
//      address-stride isolation the way the retired design did — it now
//      checks that the harness wires each slot's own buffer handle to its
//      own launch (and that neighbouring, simultaneously-live poisoned
//      allocations don't get misrouted in), which is still worth having.
//      Each component gets its own positive control proving the poison
//      mechanism is actually live (SP1's review caught a version of this
//      check that would have passed vacuously against an inert poison).
//
//   3. Negative control, corrupting the CANDIDATE ARM ONLY. Both components
//      reuse the same mechanism: misroute one slot's device-side addressing
//      so the write or read lands on a DIFFERENT, CLEAN (non-poisoned,
//      numerically distinct) slot's data, while the reference computation
//      still uses the slot's own true data. KV write forces every uploaded
//      descriptor to slot 0's. DeltaNet, post-retirement, has no row_slot
//      left to corrupt, so the equivalent corruption is at the call-site
//      level: the candidate launch is wired to the BYSTANDER slot's own
//      (clean, numerically distinct) S buffers while every other argument
//      (q/k/v/gate/beta) stays the target's — the same class of bug a wrong
//      buffer handle in a real caller would produce. This produces a
//      genuine finite numeric mismatch — not a crash and not the
//      all-zero/non-finite guard rails — exactly the shape SP1's task-7
//      review demanded ("91.44x tolerance", not an assertion trip). Neither
//      kernel has a device-side bounds guard that could abort before the
//      comparison runs (kv_cache_write's only host guard is the
//      slot_descs/row_slot both-or-neither assert, which that control
//      satisfies), so there is nothing to route around.
//
//   4. Generator variance. Both components' synthetic-data generators are
//      probed directly — hold slot fixed and vary position/token, then hold
//      position/token fixed and vary slot — and asserted non-constant
//      BEFORE any GPU work runs. This is the direct fix for the SP1 near
//      miss: a generator computing `pos * 7 % 7` is always zero, and a
//      whole-array "isn't literally constant" check does not catch a term
//      that is degenerate in exactly the ONE dimension an addressing bug
//      would need to vary along.
//
// BOX STATE: see the top-level task report
// (.superpowers/sdd/sp2-task-6-report.md) for whether this ran.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("test_multislot_ops requires --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    dn::run();
}

#[cfg(feature = "deltanet")]
mod dn {
    use rdna_compute::kv_slots::{self, KvSlotDesc, R9700_VRAM_BYTES};
    use rdna_compute::page_pool::PAGE_TOKENS;
    use rdna_compute::slot_pool::{SlotId, SlotPool};
    use rdna_compute::{DType, Gpu, GpuTensor};

    // ─────────────────────────── shared helpers ────────────────────────────

    fn i32_bytes(v: &[i32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_ne_bytes()).collect()
    }

    fn i8_bytes(v: &[i8]) -> Vec<u8> {
        v.iter().map(|&x| x as u8).collect()
    }

    /// Pack KvSlotDesc records byte-identically to kernels/src/kv_slot_desc.h
    /// (block_table u64, legacy_k_base u64, legacy_v_base u64, seq_len i32,
    /// page_tokens i32 = 32 bytes).
    fn pack_descs(descs: &[KvSlotDesc]) -> Vec<u8> {
        let mut out = Vec::with_capacity(descs.len() * 32);
        for d in descs {
            out.extend_from_slice(&d.block_table.to_ne_bytes());
            out.extend_from_slice(&d.legacy_k_base.to_ne_bytes());
            out.extend_from_slice(&d.legacy_v_base.to_ne_bytes());
            out.extend_from_slice(&d.seq_len.to_ne_bytes());
            out.extend_from_slice(&d.page_tokens.to_ne_bytes());
        }
        out
    }

    /// IEEE binary16 -> f32, including NaN/Inf (unlike `kv_slots::half_from_f32`,
    /// which only needs to ENCODE small positive scales, this harness needs to
    /// DECODE arbitrary bit patterns — specifically the 0x7E00 NaN sentinel
    /// used as the poison marker below). Copied from
    /// `crates/hipfire-runtime/examples/test_q8kv.rs::f16_to_f32`.
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as i32;
        let frac = (bits & 0x3FF) as u32;
        if exp == 0 {
            if frac == 0 {
                return if sign == 1 { -0.0 } else { 0.0 };
            }
            let v = (frac as f32) / 1024.0 * 2.0f32.powi(-14);
            return if sign == 1 { -v } else { v };
        }
        if exp == 31 {
            return if frac == 0 {
                if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            };
        }
        let v = 2.0f32.powi(exp - 15) * (1.0 + frac as f32 / 1024.0);
        if sign == 1 {
            -v
        } else {
            v
        }
    }

    /// Decode a byte slice as consecutive 34-byte Q8_0 blocks (2-byte f16
    /// scale + 32 int8 values) into flat f32. `bytes.len()` must be a
    /// multiple of 34.
    fn decode_q8_0(bytes: &[u8]) -> Vec<f32> {
        assert_eq!(
            bytes.len() % 34,
            0,
            "decode_q8_0: not a whole number of 34-byte blocks"
        );
        let mut out = Vec::with_capacity(bytes.len() / 34 * 32);
        for blk in bytes.chunks_exact(34) {
            let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            for &b in &blk[2..34] {
                out.push((b as i8) as f32 * scale);
            }
        }
        out
    }

    fn download_raw(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
        let mut buf = vec![0u8; t.buf.size()];
        gpu.hip.memcpy_dtoh(&mut buf, &t.buf).expect("download raw");
        buf
    }

    fn assert_varying_f32(data: &[f32], label: &str) {
        assert!(!data.is_empty(), "{label}: empty — nothing to vary");
        let all_same = data.windows(2).all(|w| w[0] == w[1]);
        assert!(
            !all_same,
            "{label}: generator produced constant values — not a real test"
        );
    }

    fn assert_varying_i8(data: &[i8], label: &str) {
        assert!(!data.is_empty(), "{label}: empty — nothing to vary");
        let all_same = data.windows(2).all(|w| w[0] == w[1]);
        assert!(
            !all_same,
            "{label}: generator produced constant values — not a real test"
        );
    }

    /// `assert_close`, `assert_varying_*` and the non-degeneracy/finiteness
    /// guards below are the exact instruments `test_batched_attn_slots.rs`
    /// built and SP1's review hardened (an earlier version silently passed
    /// two all-zero arrays at "0.000x tolerance", and separately silently
    /// passed a shared-NaN candidate/reference pair). Reused verbatim rather
    /// than re-derived, per the brief's "reuse its structure" instruction.
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

    /// Run `f`, expecting it to panic (i.e. expecting `assert_close` to
    /// report a real mismatch). Used by the negative controls, which must
    /// prove the harness's own numeric comparison is sensitive to a
    /// candidate-arm-only corruption — not merely that *some* assertion
    /// somewhere fires. Prints the panic payload so the mismatch (worst
    /// element / tolerance ratio) is visible in the report, per the brief's
    /// "report each control's actual output" instruction.
    fn expect_mismatch<F: FnOnce() + std::panic::UnwindSafe>(label: &str, f: F) {
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {})); // we print our own message below
        let result = std::panic::catch_unwind(f);
        std::panic::set_hook(prev_hook);
        match result {
            Ok(()) => panic!(
                "{label}: negative control did NOT produce a mismatch — the corruption was \
                 ineffective (or, worse, a real defect is already masking it)"
            ),
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "<non-string panic payload>".to_string());
                println!("  {label}: OK — corruption produced the expected mismatch:\n    {msg}");
            }
        }
    }

    // ═══════════════════════════ KV write component ═══════════════════════

    const KV_N_KV_HEADS: usize = 2;
    const KV_HEAD_DIM: usize = 256; // multiple of 32, matches SP1's shapes
    const KV_PER_POS_BYTES: usize = KV_N_KV_HEADS * (KV_HEAD_DIM / 32) * 34; // 544
    const KV_DIM: usize = KV_N_KV_HEADS * KV_HEAD_DIM; // 512
    const KV_MAX_SLOTS: usize = 8;
    const KV_PAGE_TOKENS: usize = 128; // mirrors SlotPool's internal rounding

    /// Per-slot seq_len, deliberately slot-varying (a uniform seq_len across
    /// slots would let a wrong-but-plausible k_base pass by symmetry in the
    /// golden check).
    fn kv_seq_len(slot: usize) -> usize {
        191 + slot * 137
    }

    fn kv_cap_tokens(seq_len: usize) -> usize {
        seq_len.div_ceil(KV_PAGE_TOKENS) * KV_PAGE_TOKENS
    }

    /// Three positions per slot, spread across its length: first, middle,
    /// last. Exercises small and large offsets within the same slab in one
    /// call, not just position 0.
    fn kv_write_positions(seq_len: usize) -> Vec<usize> {
        vec![0, seq_len / 2, seq_len - 1]
    }

    /// Synthetic KV-write source value. Every term (elem, slot, pos) has a
    /// distinct, coprime-ish multiplier so no single dimension can degenerate
    /// to a constant the way `pos * 7 % 7` did in SP1's near miss.
    fn kv_src_value(slot: usize, pos: usize, elem: usize) -> f32 {
        let x = (elem as u32)
            .wrapping_mul(37)
            .wrapping_add((slot as u32).wrapping_mul(53))
            .wrapping_add((pos as u32).wrapping_mul(11));
        (((x % 101) as f32) - 50.0) * 0.01
    }

    /// Step 4 for KV write: prove `kv_src_value` actually varies along BOTH
    /// the slot and the position axis before trusting anything downstream.
    fn assert_kv_generator_varies() {
        let a = kv_src_value(0, 500, 3);
        let b = kv_src_value(1, 500, 3); // slot varies, pos/elem fixed
        let c = kv_src_value(0, 900, 3); // pos varies, slot/elem fixed
        assert_ne!(
            a, b,
            "KV-write src generator: slot index does not perturb the output"
        );
        assert_ne!(
            a, c,
            "KV-write src generator: position does not perturb the output"
        );
    }

    /// One arena-worth of NaN-poisoned Q8_0 bytes: f16 scale = 0x7E00 (NaN),
    /// quant bytes arbitrary (irrelevant — `q * NaN == NaN` for every q,
    /// including 0). Matches the poison convention `kv_slots::build_arena`
    /// uses for the attention harness.
    fn poison_bytes(n_bytes: usize) -> Vec<u8> {
        assert_eq!(n_bytes % 34, 0);
        let mut out = Vec::with_capacity(n_bytes);
        for i in 0..(n_bytes / 34) {
            out.push(0x00);
            out.push(0x7E);
            for j in 0..32 {
                out.push(((i * 7 + j) % 256) as u8);
            }
        }
        out
    }

    /// Build a `SlotPool`-backed descriptor table for `seq_lens`. Uses
    /// `SlotPool` directly (Task 1) rather than hand-rolling slab math, per
    /// the brief's "Consumes: SlotPool" — this is real integration, not a
    /// re-implementation of Task 1's layout.
    fn build_kv_pool(seq_lens: &[usize]) -> (SlotPool, Vec<KvSlotDesc>) {
        let cap_tokens = *seq_lens.iter().max().expect("seq_lens non-empty");
        let mut pool =
            SlotPool::new(seq_lens.len(), cap_tokens, KV_PER_POS_BYTES).expect("SlotPool::new");
        for (s, &sl) in seq_lens.iter().enumerate() {
            let id = pool.acquire().expect("SlotPool::acquire");
            assert_eq!(
                id.0, s,
                "SlotPool handed out slots out of order — row_slot indices below assume acquire() \
                 returns 0..n_slots in sequence"
            );
            pool.set_seq_len(id, sl).expect("SlotPool::set_seq_len");
        }
        let descs = pool.descriptors().to_vec();
        (pool, descs)
    }

    /// Run the legacy single-sequence KV write for one slot's rows into a
    /// fresh, isolated destination arena, and decode the requested positions.
    /// This is the "reference" arm for every KV-write check below.
    fn kv_legacy_reference(
        gpu: &mut Gpu,
        cap: usize,
        src_slice: &[f32],
        positions: &[usize],
    ) -> Vec<f32> {
        let dst_ref = gpu
            .zeros(&[cap * KV_PER_POS_BYTES], DType::Raw)
            .expect("dst_ref");
        let src_dev = gpu
            .upload_f32(src_slice, &[positions.len() * KV_DIM])
            .expect("src upload");
        let pos_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let pos_dev = gpu
            .upload_raw(&i32_bytes(&pos_i32), &[positions.len()])
            .expect("pos upload");
        gpu.kv_cache_write_q8_0_batched(
            &dst_ref,
            &src_dev,
            &pos_dev,
            KV_N_KV_HEADS,
            KV_HEAD_DIM,
            positions.len(),
        )
        .expect("legacy kv write");
        gpu.hip.device_synchronize().expect("sync");
        let bytes = download_raw(gpu, &dst_ref);
        let decoded = decode_q8_0(&bytes);
        gpu.free_tensor(dst_ref).expect("free dst_ref");
        gpu.free_tensor(src_dev).expect("free src_dev");
        gpu.free_tensor(pos_dev).expect("free pos_dev");
        let mut out = Vec::with_capacity(positions.len() * KV_DIM);
        for &p in positions {
            out.extend_from_slice(&decoded[p * KV_DIM..(p + 1) * KV_DIM]);
        }
        out
    }

    /// Step 1 for KV write: multi-slot write over ALL slots in one launch,
    /// checked slot-by-slot against `kv_legacy_reference`.
    fn test_kv_write_golden(gpu: &mut Gpu, n_slots: usize) {
        let seq_lens: Vec<usize> = (0..n_slots).map(kv_seq_len).collect();
        let (pool, descs) = build_kv_pool(&seq_lens);
        let positions_per_slot: Vec<Vec<usize>> =
            seq_lens.iter().map(|&sl| kv_write_positions(sl)).collect();

        let mut src = Vec::new();
        let mut positions_flat = Vec::new();
        let mut row_slot = Vec::new();
        for (s, positions) in positions_per_slot.iter().enumerate() {
            for &p in positions {
                for e in 0..KV_DIM {
                    src.push(kv_src_value(s, p, e));
                }
                positions_flat.push(p as i32);
                row_slot.push(s as i32);
            }
        }
        assert_varying_f32(&src, &format!("KV golden src n_slots={n_slots}"));

        let arena_bytes = pool.arena_bytes();
        let dst = gpu.zeros(&[arena_bytes], DType::Raw).expect("dst arena");
        let src_dev = gpu
            .upload_f32(&src, &[positions_flat.len() * KV_DIM])
            .expect("src upload");
        let positions_dev = gpu
            .upload_raw(&i32_bytes(&positions_flat), &[positions_flat.len()])
            .expect("positions upload");
        let row_slot_dev = gpu
            .upload_raw(&i32_bytes(&row_slot), &[row_slot.len()])
            .expect("row_slot upload");
        let descs_dev = gpu
            .upload_raw(&pack_descs(&descs), &[descs.len()])
            .expect("descs upload");

        gpu.kv_cache_write_q8_0_batched_slots(
            &dst,
            &src_dev,
            &positions_dev,
            KV_N_KV_HEADS,
            KV_HEAD_DIM,
            positions_flat.len(),
            Some(&descs_dev),
            Some(&row_slot_dev),
            /*use_v_base=*/ false,
        )
        .expect("multi-slot kv write");
        gpu.hip.device_synchronize().expect("sync");
        let arena = download_raw(gpu, &dst);

        gpu.free_tensor(dst).expect("free dst");
        gpu.free_tensor(src_dev).expect("free src_dev");
        gpu.free_tensor(positions_dev).expect("free positions_dev");
        gpu.free_tensor(row_slot_dev).expect("free row_slot_dev");
        gpu.free_tensor(descs_dev).expect("free descs_dev");

        for (s, positions) in positions_per_slot.iter().enumerate() {
            let desc = descs[s];
            let cap = rdna_compute::kv_slots::legacy_cap(desc.seq_len as usize);
            let region = &arena
                [desc.legacy_k_base as usize..desc.legacy_k_base as usize + cap * KV_PER_POS_BYTES];
            let decoded = decode_q8_0(region);
            let mut candidate = Vec::with_capacity(positions.len() * KV_DIM);
            for &p in positions {
                candidate.extend_from_slice(&decoded[p * KV_DIM..(p + 1) * KV_DIM]);
            }
            let src_slice: Vec<f32> = positions
                .iter()
                .flat_map(|&p| (0..KV_DIM).map(move |e| kv_src_value(s, p, e)))
                .collect();
            let reference = kv_legacy_reference(gpu, cap, &src_slice, positions);
            assert_close(
                &format!("KV golden n_slots={n_slots} slot={s}"),
                &candidate,
                &reference,
            );
        }
    }

    /// Step 2 for KV write: pre-poison the WHOLE arena, write only the
    /// TARGET slot's rows, and require every OTHER slot's slab to still
    /// decode as 100% NaN — the write must not leak outside its own
    /// k_base..k_base+cap window. Then a positive control: corrupt the
    /// DEVICE-side row_slot for the same rows to point at a bystander
    /// instead, and confirm the bystander's slab DOES flip out of poison —
    /// proving the "still poison" check above is sensitive to a real
    /// misroute, not vacuously true because nothing ever gets written.
    fn test_kv_write_isolation(gpu: &mut Gpu, n_slots: usize) {
        if n_slots < 2 {
            println!("  KV isolation n_slots=1: skipped (no neighbour to poison)");
            return;
        }
        let seq_lens: Vec<usize> = (0..n_slots).map(kv_seq_len).collect();
        let (pool, descs) = build_kv_pool(&seq_lens);
        let arena_bytes = pool.arena_bytes();
        let target = n_slots - 1;
        let bystander = 0usize;
        assert_ne!(target, bystander, "isolation needs target != bystander");

        let positions = kv_write_positions(seq_lens[target]);
        let mut src = Vec::new();
        for &p in &positions {
            for e in 0..KV_DIM {
                src.push(kv_src_value(target, p, e));
            }
        }
        let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let row_slot_true = vec![target as i32; positions.len()];

        // ---- Isolation: correctly-routed write to `target` only. ----
        {
            let poisoned = poison_bytes(arena_bytes);
            let dst = gpu
                .upload_raw(&poisoned, &[arena_bytes])
                .expect("poisoned dst upload");
            let src_dev = gpu
                .upload_f32(&src, &[positions.len() * KV_DIM])
                .expect("src upload");
            let positions_dev = gpu
                .upload_raw(&i32_bytes(&positions_i32), &[positions.len()])
                .expect("positions upload");
            let row_slot_dev = gpu
                .upload_raw(&i32_bytes(&row_slot_true), &[positions.len()])
                .expect("row_slot upload");
            let descs_dev = gpu
                .upload_raw(&pack_descs(&descs), &[descs.len()])
                .expect("descs upload");

            gpu.kv_cache_write_q8_0_batched_slots(
                &dst,
                &src_dev,
                &positions_dev,
                KV_N_KV_HEADS,
                KV_HEAD_DIM,
                positions.len(),
                Some(&descs_dev),
                Some(&row_slot_dev),
                false,
            )
            .expect("isolation write");
            gpu.hip.device_synchronize().expect("sync");
            let arena = download_raw(gpu, &dst);

            gpu.free_tensor(dst).expect("free dst");
            gpu.free_tensor(src_dev).expect("free src_dev");
            gpu.free_tensor(positions_dev).expect("free positions_dev");
            gpu.free_tensor(row_slot_dev).expect("free row_slot_dev");
            gpu.free_tensor(descs_dev).expect("free descs_dev");

            for (s, &desc) in descs.iter().enumerate() {
                let cap = rdna_compute::kv_slots::legacy_cap(desc.seq_len as usize);
                let region = &arena[desc.legacy_k_base as usize
                    ..desc.legacy_k_base as usize + cap * KV_PER_POS_BYTES];
                let decoded = decode_q8_0(region);
                if s == target {
                    let mut candidate = Vec::with_capacity(positions.len() * KV_DIM);
                    for &p in &positions {
                        candidate.extend_from_slice(&decoded[p * KV_DIM..(p + 1) * KV_DIM]);
                    }
                    assert!(
                        candidate.iter().all(|v| v.is_finite()),
                        "KV isolation n_slots={n_slots}: target slot {target}'s own write came back non-finite"
                    );
                    let reference = kv_legacy_reference(gpu, cap, &src, &positions);
                    assert_close(
                        &format!("KV isolation n_slots={n_slots} target={target} (own write)"),
                        &candidate,
                        &reference,
                    );
                } else {
                    assert!(
                        decoded.iter().all(|v| !v.is_finite()),
                        "KV isolation n_slots={n_slots}: writing target slot {target} changed \
                         neighbour slot {s}'s slab — addressing leaked across a slot boundary"
                    );
                }
            }
            println!("  KV isolation n_slots={n_slots} target={target}: OK (neighbours untouched, target finite+correct)");
        }

        // ---- Positive control: misroute the SAME rows onto `bystander`. ----
        {
            let poisoned = poison_bytes(arena_bytes);
            let dst = gpu
                .upload_raw(&poisoned, &[arena_bytes])
                .expect("poisoned dst upload (positive control)");
            let src_dev = gpu
                .upload_f32(&src, &[positions.len() * KV_DIM])
                .expect("src upload");
            let positions_dev = gpu
                .upload_raw(&i32_bytes(&positions_i32), &[positions.len()])
                .expect("positions upload");
            let row_slot_corrupt = vec![bystander as i32; positions.len()];
            let row_slot_dev = gpu
                .upload_raw(&i32_bytes(&row_slot_corrupt), &[positions.len()])
                .expect("row_slot upload (corrupt)");
            let descs_dev = gpu
                .upload_raw(&pack_descs(&descs), &[descs.len()])
                .expect("descs upload");

            gpu.kv_cache_write_q8_0_batched_slots(
                &dst,
                &src_dev,
                &positions_dev,
                KV_N_KV_HEADS,
                KV_HEAD_DIM,
                positions.len(),
                Some(&descs_dev),
                Some(&row_slot_dev),
                false,
            )
            .expect("positive control write");
            gpu.hip.device_synchronize().expect("sync");
            let arena = download_raw(gpu, &dst);

            gpu.free_tensor(dst).expect("free dst");
            gpu.free_tensor(src_dev).expect("free src_dev");
            gpu.free_tensor(positions_dev).expect("free positions_dev");
            gpu.free_tensor(row_slot_dev).expect("free row_slot_dev");
            gpu.free_tensor(descs_dev).expect("free descs_dev");

            let desc_b = descs[bystander];
            let cap_b = rdna_compute::kv_slots::legacy_cap(desc_b.seq_len as usize);
            let region_b = &arena[desc_b.legacy_k_base as usize
                ..desc_b.legacy_k_base as usize + cap_b * KV_PER_POS_BYTES];
            let decoded_b = decode_q8_0(region_b);
            let n_finite = decoded_b.iter().filter(|v| v.is_finite()).count();
            assert!(
                n_finite > 0,
                "KV positive control n_slots={n_slots}: rows misrouted from target {target} to \
                 bystander {bystander} but the bystander's slab is STILL 100% poison — the \
                 isolation check above cannot be trusted, since it would pass identically \
                 whether or not addressing actually leaked"
            );

            let desc_t = descs[target];
            let cap_t = rdna_compute::kv_slots::legacy_cap(desc_t.seq_len as usize);
            let region_t = &arena[desc_t.legacy_k_base as usize
                ..desc_t.legacy_k_base as usize + cap_t * KV_PER_POS_BYTES];
            let decoded_t = decode_q8_0(region_t);
            assert!(
                decoded_t.iter().all(|v| !v.is_finite()),
                "KV positive control n_slots={n_slots}: target {target}'s slab changed even \
                 though every row was rerouted to bystander {bystander} — the write landed in \
                 neither the expected nor the misrouted location"
            );
            println!(
                "  KV positive control n_slots={n_slots}: OK (misrouted rows flipped bystander \
                 {bystander}'s slab out of poison: {n_finite}/{} elements finite)",
                decoded_b.len()
            );
        }
    }

    /// Step 3 for KV write: corrupt the CANDIDATE arm's device-side
    /// descriptor table only (every descriptor forced to slot 0's, exactly
    /// SP1's `maybe_corrupt`), leaving the reference computation on the true
    /// descriptors. Target != slot 0, so the corruption actually changes
    /// where the write lands. The arena starts all-ZERO (not poisoned) so
    /// the mismatch this produces is a genuine finite tolerance-ratio
    /// failure, not the non-finite guard rail.
    fn test_kv_write_negative_control(gpu: &mut Gpu) {
        let seq_lens = vec![kv_seq_len(0), kv_seq_len(1), kv_seq_len(2)];
        let (pool, descs) = build_kv_pool(&seq_lens);
        let target = 1usize; // != 0, so "force every desc to slot 0" actually misroutes it
        let positions = kv_write_positions(seq_lens[target]);
        let src: Vec<f32> = positions
            .iter()
            .flat_map(|&p| (0..KV_DIM).map(move |e| kv_src_value(target, p, e)))
            .collect();
        let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let row_slot = vec![target as i32; positions.len()];

        let descs_corrupt: Vec<KvSlotDesc> = descs.iter().map(|_| descs[0]).collect();

        let arena_bytes = pool.arena_bytes();
        let dst = gpu
            .zeros(&[arena_bytes], DType::Raw)
            .expect("dst arena (zeroed, not poisoned)");
        let src_dev = gpu
            .upload_f32(&src, &[positions.len() * KV_DIM])
            .expect("src upload");
        let positions_dev = gpu
            .upload_raw(&i32_bytes(&positions_i32), &[positions.len()])
            .expect("positions upload");
        let row_slot_dev = gpu
            .upload_raw(&i32_bytes(&row_slot), &[positions.len()])
            .expect("row_slot upload");
        // Device-side table only: descs_corrupt uploads, but `descs` (the true
        // table) is what the reference call and the readback offset below use.
        let descs_dev = gpu
            .upload_raw(&pack_descs(&descs_corrupt), &[descs_corrupt.len()])
            .expect("descs upload (corrupt)");

        gpu.kv_cache_write_q8_0_batched_slots(
            &dst,
            &src_dev,
            &positions_dev,
            KV_N_KV_HEADS,
            KV_HEAD_DIM,
            positions.len(),
            Some(&descs_dev),
            Some(&row_slot_dev),
            false,
        )
        .expect("negative-control write");
        gpu.hip.device_synchronize().expect("sync");
        let arena = download_raw(gpu, &dst);

        gpu.free_tensor(dst).expect("free dst");
        gpu.free_tensor(src_dev).expect("free src_dev");
        gpu.free_tensor(positions_dev).expect("free positions_dev");
        gpu.free_tensor(row_slot_dev).expect("free row_slot_dev");
        gpu.free_tensor(descs_dev).expect("free descs_dev");

        // Read back at target's TRUE offset (uncorrupted) — the corrupted
        // launch actually wrote to slot 0's offset instead, so this region
        // never got touched and still reads the zero fill.
        let desc_t = descs[target];
        let cap_t = rdna_compute::kv_slots::legacy_cap(desc_t.seq_len as usize);
        let region_t = &arena[desc_t.legacy_k_base as usize
            ..desc_t.legacy_k_base as usize + cap_t * KV_PER_POS_BYTES];
        let decoded_t = decode_q8_0(region_t);
        let mut candidate = Vec::with_capacity(positions.len() * KV_DIM);
        for &p in &positions {
            candidate.extend_from_slice(&decoded_t[p * KV_DIM..(p + 1) * KV_DIM]);
        }
        let reference = kv_legacy_reference(gpu, cap_t, &src, &positions);

        expect_mismatch("KV negative control", move || {
            assert_close(
                "KV negative control (corrupted descriptor table, candidate arm only)",
                &candidate,
                &reference,
            );
        });
    }

    // ═══════════════════════ Paged KV component (SP4) ══════════════════════

    /// Build a paged `SlotPool` whose block tables are deliberately
    /// NON-monotonic. Allocate A, then B, release A, then take C: the free
    /// list is LIFO, so C's logical page 0 lands in A's OLD physical page 2
    /// and its table reads [2, 1, 0] — a kernel that ignored the block table
    /// and read C's KV contiguously would get pages in the wrong order (and
    /// B's live pages sitting right after them). That is exactly the failure
    /// mode this section exists to catch.
    fn build_paged_pool_scattered() -> (SlotPool, SlotId, SlotId) {
        let mut pool = SlotPool::new_paged(3, 384, KV_PER_POS_BYTES, 6).expect("paged pool");
        let a = pool.acquire().expect("acquire A");
        let b = pool.acquire().expect("acquire B");
        pool.set_seq_len(a, 300).expect("A seq_len"); // pages [0, 1, 2]
        pool.set_seq_len(b, 130).expect("B seq_len"); // pages [3, 4]
        pool.release(a); // frees 0, 1, 2 (B keeps 3, 4)
        let c = pool.acquire().expect("acquire C"); // LIFO: pages [2, 1, 0]
        pool.set_seq_len(c, 257).expect("C seq_len");
        assert_eq!(
            pool.block_table(c).unwrap().page_indices(),
            &[2u32, 1, 0],
            "scatter setup broke: C's table must be non-monotonic for this test to mean anything"
        );
        (pool, b, c)
    }

    /// Upload paged descriptors for the WHOLE pool — one descriptor entry
    /// per pool slot, indexed by pool slot id, exactly like production's
    /// `descs_dev` (the kernels do `slot_descs[row_slot[row]]` with raw pool
    /// ids, so a table sized only to the live slots would be indexed out of
    /// range). Released or never-filled slots get the legacy zero-base
    /// zero-length descriptor `SlotPool::reset` leaves behind; no row may
    /// target them.
    fn upload_paged_pool_descs(
        gpu: &mut Gpu,
        pool: &SlotPool,
    ) -> (Vec<KvSlotDesc>, Vec<GpuTensor>) {
        let n = pool.descriptors().len();
        let mut descs = Vec::with_capacity(n);
        let mut tables: Vec<GpuTensor> = Vec::with_capacity(n);
        for i in 0..n {
            match pool.block_table(SlotId(i)) {
                Some(bt) if bt.num_pages() > 0 => {
                    let indices = bt.page_indices();
                    let bytes: Vec<u8> = indices.iter().flat_map(|x| x.to_ne_bytes()).collect();
                    let dev = gpu
                        .upload_raw(&bytes, &[indices.len()])
                        .expect("block table upload");
                    descs.push(KvSlotDesc {
                        block_table: dev.buf.as_ptr() as u64,
                        legacy_k_base: 0,
                        legacy_v_base: 0,
                        seq_len: bt.live_tokens() as i32,
                        page_tokens: PAGE_TOKENS as i32,
                    });
                    tables.push(dev);
                }
                _ => {
                    descs.push(KvSlotDesc {
                        block_table: 0,
                        legacy_k_base: 0,
                        legacy_v_base: 0,
                        seq_len: 0,
                        page_tokens: 0,
                    });
                }
            }
        }
        (descs, tables)
    }

    /// Decode one position out of a paged arena, translating through the
    /// HOST copy of the block table — the ground truth the kernel must agree
    /// with.
    fn decode_paged_position(
        arena: &[u8],
        pages: &[u32],
        per_pos_bytes: usize,
        pos: usize,
    ) -> Vec<f32> {
        let page_bytes = PAGE_TOKENS * per_pos_bytes;
        let lp = pos / PAGE_TOKENS;
        let off = pos % PAGE_TOKENS;
        let phys = pages[lp] as usize;
        let start = phys * page_bytes + off * per_pos_bytes;
        decode_q8_0(&arena[start..start + per_pos_bytes])
    }

    /// Step 1 for paged KV: positions deliberately chosen to CROSS page
    /// boundaries (127|128, 255|256), written through paged descriptors into
    /// an arena whose free-list layout puts C's pages in REVERSE order.
    /// Every written position must decode — via the host block table — to
    /// exactly what the legacy single-sequence kernel wrote for the same
    /// values, and every UNwritten position in both live slots' pages must
    /// still be NaN poison (the write went precisely where the table said,
    /// and nowhere else).
    fn test_kv_write_paged_golden(gpu: &mut Gpu) {
        let (pool, b, c) = build_paged_pool_scattered();
        let slots = vec![b, c];
        let arena_bytes = pool.arena_bytes();

        // Positions crossing page boundaries in both slots.
        let positions_per_slot: Vec<Vec<usize>> = vec![
            vec![0, 100, 127, 128, 129],      // B: 130 live, 2 pages
            vec![0, 127, 128, 200, 255, 256], // C: 257 live, 3 pages
        ];

        let mut src = Vec::new();
        let mut positions_flat = Vec::new();
        let mut row_slot = Vec::new();
        for (i, &s) in slots.iter().enumerate() {
            let live = pool.block_table(s).unwrap().live_tokens();
            for &p in &positions_per_slot[i] {
                assert!(
                    p < live,
                    "test position {p} exceeds slot {}'s live length",
                    s.0
                );
                for e in 0..KV_DIM {
                    src.push(kv_src_value(s.0, p, e));
                }
                positions_flat.push(p as i32);
                row_slot.push(s.0 as i32);
            }
        }

        let poisoned = poison_bytes(arena_bytes);
        let dst = gpu
            .upload_raw(&poisoned, &[arena_bytes])
            .expect("poisoned paged arena");
        let src_dev = gpu
            .upload_f32(&src, &[positions_flat.len() * KV_DIM])
            .expect("src upload");
        let positions_dev = gpu
            .upload_raw(&i32_bytes(&positions_flat), &[positions_flat.len()])
            .expect("positions upload");
        let row_slot_dev = gpu
            .upload_raw(&i32_bytes(&row_slot), &[row_slot.len()])
            .expect("row_slot upload");
        let (descs, table_devs) = upload_paged_pool_descs(gpu, &pool);
        let descs_dev = gpu
            .upload_raw(&pack_descs(&descs), &[descs.len()])
            .expect("paged descs upload");

        gpu.kv_cache_write_q8_0_batched_slots(
            &dst,
            &src_dev,
            &positions_dev,
            KV_N_KV_HEADS,
            KV_HEAD_DIM,
            positions_flat.len(),
            Some(&descs_dev),
            Some(&row_slot_dev),
            false,
        )
        .expect("paged multi-slot kv write");
        gpu.hip.device_synchronize().expect("sync");
        let arena = download_raw(gpu, &dst);

        for (i, &s) in slots.iter().enumerate() {
            let bt = pool.block_table(s).unwrap();
            let pages = bt.page_indices();
            let live = bt.live_tokens();
            let written: Vec<usize> = positions_per_slot[i].clone();
            // Reference slab must hold every position the slot's pages cover.
            let ref_cap = live.div_ceil(PAGE_TOKENS) * PAGE_TOKENS;

            // Every written position decodes to the legacy reference's bytes.
            for &p in &written {
                let candidate = decode_paged_position(&arena, pages, KV_PER_POS_BYTES, p);
                let src_slice: Vec<f32> = (0..KV_DIM).map(|e| kv_src_value(s.0, p, e)).collect();
                let reference = kv_legacy_reference(gpu, ref_cap, &src_slice, &[p]);
                assert_close(
                    &format!(
                        "KV paged golden slot={} pos={p} (page {})",
                        s.0,
                        pages[p / PAGE_TOKENS]
                    ),
                    &candidate,
                    &reference,
                );
            }

            // Every UNwritten position in the slot's own pages is still NaN.
            let mut checked = 0usize;
            for p in 0..live {
                if written.contains(&p) {
                    continue;
                }
                let decoded = decode_paged_position(&arena, pages, KV_PER_POS_BYTES, p);
                assert!(
                    decoded.iter().all(|v| !v.is_finite()),
                    "KV paged golden slot {}: position {p} was not written this call but its \
                     page bytes changed — the write leaked outside the block-table translation",
                    s.0
                );
                checked += 1;
            }
            println!(
                "  KV paged golden slot={} (pages {:?}, {live} live): OK ({} written positions \
                 match reference, {checked} unwritten positions still poison)",
                s.0,
                pages,
                written.len()
            );
        }

        gpu.free_tensor(dst).expect("free dst");
        gpu.free_tensor(src_dev).expect("free src_dev");
        gpu.free_tensor(positions_dev).expect("free positions_dev");
        gpu.free_tensor(row_slot_dev).expect("free row_slot_dev");
        gpu.free_tensor(descs_dev).expect("free descs_dev");
        for t in table_devs {
            gpu.free_tensor(t).expect("free block table");
        }
    }

    /// Cross-slot isolation for paged KV: write ONLY slot C's rows into a
    /// fully poisoned arena; B's live pages (physical 3 and 4 — note C's
    /// REVERSED table puts its own logical page 0 at physical 2, adjacent to
    /// B's first page) must still be 100% NaN. A kernel that fell back to
    /// contiguous-slab addressing for C would write into physical pages
    /// 0..2* — physical 3 is B's — and flip B out of poison.
    fn test_kv_write_paged_isolation(gpu: &mut Gpu) {
        let (pool, b, c) = build_paged_pool_scattered();
        let arena_bytes = pool.arena_bytes();

        let live_c = pool.block_table(c).unwrap().live_tokens();
        let positions: Vec<usize> = (0..live_c).collect();
        let src: Vec<f32> = positions
            .iter()
            .flat_map(|&p| (0..KV_DIM).map(move |e| kv_src_value(c.0, p, e)))
            .collect();
        let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let row_slot = vec![c.0 as i32; positions.len()];

        let poisoned = poison_bytes(arena_bytes);
        let dst = gpu
            .upload_raw(&poisoned, &[arena_bytes])
            .expect("poisoned paged arena (isolation)");
        let src_dev = gpu
            .upload_f32(&src, &[positions.len() * KV_DIM])
            .expect("src upload");
        let positions_dev = gpu
            .upload_raw(&i32_bytes(&positions_i32), &[positions.len()])
            .expect("positions upload");
        let row_slot_dev = gpu
            .upload_raw(&i32_bytes(&row_slot), &[row_slot.len()])
            .expect("row_slot upload");
        let (descs, table_devs) = upload_paged_pool_descs(gpu, &pool);
        let descs_dev = gpu
            .upload_raw(&pack_descs(&descs), &[descs.len()])
            .expect("paged descs upload");

        gpu.kv_cache_write_q8_0_batched_slots(
            &dst,
            &src_dev,
            &positions_dev,
            KV_N_KV_HEADS,
            KV_HEAD_DIM,
            positions.len(),
            Some(&descs_dev),
            Some(&row_slot_dev),
            false,
        )
        .expect("paged isolation write");
        gpu.hip.device_synchronize().expect("sync");
        let arena = download_raw(gpu, &dst);

        // Bystander B: every position of every live page still poison.
        let bt_b = pool.block_table(b).unwrap();
        for p in 0..bt_b.live_tokens() {
            let decoded = decode_paged_position(&arena, bt_b.page_indices(), KV_PER_POS_BYTES, p);
            assert!(
                decoded.iter().all(|v| !v.is_finite()),
                "KV paged isolation: writing ALL of slot {}'s positions flipped bystander \
                 slot {}'s page bytes at position {p} — paged addressing leaked across slots",
                c.0,
                b.0
            );
        }

        // Target C: its full live length decodes finite and correct.
        for &p in &positions {
            let candidate = decode_paged_position(
                &arena,
                pool.block_table(c).unwrap().page_indices(),
                KV_PER_POS_BYTES,
                p,
            );
            assert!(
                candidate.iter().all(|v| v.is_finite()),
                "KV paged isolation: target slot {} position {p} came back non-finite",
                c.0
            );
        }
        println!(
            "  KV paged isolation: OK (slot {} wrote {} positions via a reversed block table; \
             bystander slot {}'s pages untouched)",
            c.0,
            positions.len(),
            b.0
        );

        gpu.free_tensor(dst).expect("free dst");
        gpu.free_tensor(src_dev).expect("free src_dev");
        gpu.free_tensor(positions_dev).expect("free positions_dev");
        gpu.free_tensor(row_slot_dev).expect("free row_slot_dev");
        gpu.free_tensor(descs_dev).expect("free descs_dev");
        for t in table_devs {
            gpu.free_tensor(t).expect("free block table");
        }
    }

    /// Paged attention read path: the SAME logical KV content, attended once
    /// through paged descriptors over the scattered arena and once through
    /// legacy descriptors over a gathered contiguous arena. Both arms run
    /// `attention_q8_0_kv_batched_masked_slots`; they read byte-identical
    /// KV, so the outputs must match. A kernel that ignored the block table
    /// reads the wrong physical pages and diverges here.
    fn test_attn_paged_matches_legacy(gpu: &mut Gpu) {
        let (pool, b, c) = build_paged_pool_scattered();
        let slots = vec![b, c];
        let arena_bytes = pool.arena_bytes();

        // Write EVERY live position of both slots (attention reads [0, pos]).
        let mut src = Vec::new();
        let mut positions_flat = Vec::new();
        let mut row_slot = Vec::new();
        for &s in &slots {
            let live = pool.block_table(s).unwrap().live_tokens();
            for p in 0..live {
                for e in 0..KV_DIM {
                    src.push(kv_src_value(s.0, p, e));
                }
                positions_flat.push(p as i32);
                row_slot.push(s.0 as i32);
            }
        }

        let zeroed = vec![0u8; arena_bytes];
        let dst = gpu
            .upload_raw(&zeroed, &[arena_bytes])
            .expect("paged arena");
        let src_dev = gpu
            .upload_f32(&src, &[positions_flat.len() * KV_DIM])
            .expect("src upload");
        let positions_dev = gpu
            .upload_raw(&i32_bytes(&positions_flat), &[positions_flat.len()])
            .expect("positions upload");
        let row_slot_dev = gpu
            .upload_raw(&i32_bytes(&row_slot), &[row_slot.len()])
            .expect("row_slot upload");
        let (descs, table_devs) = upload_paged_pool_descs(gpu, &pool);
        let descs_dev = gpu
            .upload_raw(&pack_descs(&descs), &[descs.len()])
            .expect("paged descs upload");

        gpu.kv_cache_write_q8_0_batched_slots(
            &dst,
            &src_dev,
            &positions_dev,
            KV_N_KV_HEADS,
            KV_HEAD_DIM,
            positions_flat.len(),
            Some(&descs_dev),
            Some(&row_slot_dev),
            false,
        )
        .expect("paged kv write (attn setup)");
        gpu.hip.device_synchronize().expect("sync");
        let arena = download_raw(gpu, &dst);
        gpu.free_tensor(dst).expect("free dst");
        gpu.free_tensor(src_dev).expect("free src_dev");
        gpu.free_tensor(positions_dev).expect("free positions_dev");
        gpu.free_tensor(row_slot_dev).expect("free row_slot_dev");

        // Decode row per slot, at each slot's LAST live position. row_slot
        // carries raw POOL slot ids (b, c) indexing the FULL n_slots-entry
        // descriptor tables both arms upload — the same contract production
        // kernels run under.
        let rows = slots.len();
        let pos_data: Vec<i32> = slots
            .iter()
            .map(|&s| (pool.block_table(s).unwrap().live_tokens() - 1) as i32)
            .collect();
        let row_slot_attn: Vec<i32> = slots.iter().map(|s| s.0 as i32).collect();
        let positions_attn = gpu
            .upload_raw(&i32_bytes(&pos_data), &[rows])
            .expect("attn positions");
        let row_slot_attn_dev = gpu
            .upload_raw(&i32_bytes(&row_slot_attn), &[rows])
            .expect("attn row_slot");
        let q_data: Vec<f32> = (0..rows * KV_DIM)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let q = gpu.upload_f32(&q_data, &[rows * KV_DIM]).expect("q");
        let out_paged = gpu.zeros(&[rows * KV_DIM], DType::F32).expect("out paged");

        // Paged arm: the scattered arena as uploaded.
        let k_paged = gpu
            .upload_raw(&arena, &[arena.len()])
            .expect("paged k arena");
        let v_paged = gpu
            .upload_raw(&arena, &[arena.len()])
            .expect("paged v arena");
        gpu.attention_q8_0_kv_batched_masked_slots(
            &q,
            &k_paged,
            &v_paged,
            &out_paged,
            &positions_attn,
            KV_N_KV_HEADS,
            KV_N_KV_HEADS,
            KV_HEAD_DIM,
            pool.cap_tokens(),
            *pos_data.iter().max().unwrap() as usize + 1,
            rows,
            None,
            0,
            0,
            Some(&descs_dev),
            Some(&row_slot_attn_dev),
        )
        .expect("paged attention");
        gpu.hip.device_synchronize().expect("sync paged attn");

        // Legacy arm: gather every slot's pages into contiguous per-slot
        // slabs (the host block table is the ground-truth translator) and
        // build a FULL n_slots-entry legacy descriptor table over the
        // gathered arena — entry 0 (released slot A) stays a zero-length
        // dummy, so row_slot [b, c] lands on B's and C's slabs in both arms.
        let page_bytes = PAGE_TOKENS * KV_PER_POS_BYTES;
        let mut legacy_arena: Vec<u8> = Vec::with_capacity(arena.len());
        let mut legacy_descs: Vec<KvSlotDesc> = Vec::with_capacity(pool.descriptors().len());
        for i in 0..pool.descriptors().len() {
            match pool.block_table(SlotId(i)) {
                Some(bt) if bt.num_pages() > 0 => {
                    let base = legacy_arena.len() as u64;
                    for &phys in bt.page_indices() {
                        let start = phys as usize * page_bytes;
                        legacy_arena.extend_from_slice(&arena[start..start + page_bytes]);
                    }
                    legacy_descs.push(KvSlotDesc {
                        block_table: 0,
                        legacy_k_base: base,
                        legacy_v_base: base,
                        seq_len: bt.live_tokens() as i32,
                        page_tokens: 0,
                    });
                }
                _ => {
                    legacy_descs.push(KvSlotDesc {
                        block_table: 0,
                        legacy_k_base: 0,
                        legacy_v_base: 0,
                        seq_len: 0,
                        page_tokens: 0,
                    });
                }
            }
        }
        let k_legacy = gpu
            .upload_raw(&legacy_arena, &[legacy_arena.len()])
            .expect("legacy k arena");
        let v_legacy = gpu
            .upload_raw(&legacy_arena, &[legacy_arena.len()])
            .expect("legacy v arena");
        let legacy_descs_dev = gpu
            .upload_raw(&pack_descs(&legacy_descs), &[legacy_descs.len()])
            .expect("legacy descs upload");
        let out_legacy = gpu.zeros(&[rows * KV_DIM], DType::F32).expect("out legacy");

        gpu.attention_q8_0_kv_batched_masked_slots(
            &q,
            &k_legacy,
            &v_legacy,
            &out_legacy,
            &positions_attn,
            KV_N_KV_HEADS,
            KV_N_KV_HEADS,
            KV_HEAD_DIM,
            pool.cap_tokens(),
            *pos_data.iter().max().unwrap() as usize + 1,
            rows,
            None,
            0,
            0,
            Some(&legacy_descs_dev),
            Some(&row_slot_attn_dev),
        )
        .expect("legacy attention");
        gpu.hip.device_synchronize().expect("sync legacy attn");

        let got = download_f32(gpu, &out_paged);
        let want = download_f32(gpu, &out_legacy);
        assert_close("attention paged vs gathered-legacy", &got, &want);

        gpu.free_tensor(k_paged).expect("free k_paged");
        gpu.free_tensor(v_paged).expect("free v_paged");
        gpu.free_tensor(k_legacy).expect("free k_legacy");
        gpu.free_tensor(v_legacy).expect("free v_legacy");
        gpu.free_tensor(legacy_descs_dev)
            .expect("free legacy descs");
        gpu.free_tensor(positions_attn)
            .expect("free positions_attn");
        gpu.free_tensor(row_slot_attn_dev)
            .expect("free row_slot_attn");
        gpu.free_tensor(q).expect("free q");
        gpu.free_tensor(out_paged).expect("free out_paged");
        gpu.free_tensor(out_legacy).expect("free out_legacy");
        gpu.free_tensor(descs_dev).expect("free descs_dev");
        for t in table_devs {
            gpu.free_tensor(t).expect("free block table");
        }
        println!(
            "  attention paged vs legacy: OK (slots {:?} attended via reversed block tables \
             match the gathered-contiguous arm)",
            slots.iter().map(|s| s.0).collect::<Vec<_>>()
        );
    }

    /// Download a full f32 tensor.
    fn download_f32(gpu: &Gpu, t: &GpuTensor) -> Vec<f32> {
        let bytes = download_raw(gpu, t);
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // ═══════════════════════════ DeltaNet component ════════════════════════

    const DN_N_HEADS: usize = 2; // smaller than production (16), fast is enough
    const DN_HD: usize = 128; // fixed by the kernel's HD macro
    const DN_N_TOKENS: usize = 3;
    const DN_MAX_SLOTS: usize = 8;
    const DN_S_STRIDE: usize = DN_N_HEADS * DN_HD * DN_HD;

    fn dn_gen(slot: usize, n_tokens: usize, n_heads: usize, hd: usize, seed: u32) -> Vec<f32> {
        let mut v = Vec::with_capacity(n_tokens * n_heads * hd);
        for t in 0..n_tokens {
            for h in 0..n_heads {
                for d in 0..hd {
                    let x = (d as u32)
                        .wrapping_mul(seed)
                        .wrapping_add((t as u32).wrapping_mul(131))
                        .wrapping_add((h as u32).wrapping_mul(17))
                        .wrapping_add((slot as u32).wrapping_mul(53));
                    v.push((((x % 101) as f32) - 50.0) * 0.01);
                }
            }
        }
        v
    }

    fn dn_gate(slot: usize, n_tokens: usize, n_heads: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n_tokens * n_heads);
        for t in 0..n_tokens {
            for h in 0..n_heads {
                let x = ((t * 7 + h * 3 + slot * 11) % 23) as f32;
                v.push(-0.1 - x * 0.02); // small negative decay, bounded
            }
        }
        v
    }

    fn dn_beta(slot: usize, n_tokens: usize, n_heads: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n_tokens * n_heads);
        for t in 0..n_tokens {
            for h in 0..n_heads {
                let x = ((t * 5 + h * 13 + slot * 7) % 19) as f32 / 19.0 - 0.5;
                v.push(1.0 / (1.0 + (-x).exp())); // sigmoid, bounded (0,1)
            }
        }
        v
    }

    fn dn_state_q(slot: usize, i: usize) -> i8 {
        (((i * 31 + slot * 101) % 251) as i32 - 125) as i8
    }

    /// Small positive varying scale — mirrors `kv_slots::build_arena`'s
    /// filler convention (`0.02 + (i % 13) * 0.005`-shaped).
    fn dn_state_scale(slot: usize, i: usize) -> f32 {
        0.01 + (((i + slot * 7) % 13) as f32) * 0.003
    }

    /// Step 4 for DeltaNet: prove the Q/K/V/gate/beta/state generators vary
    /// along BOTH the slot axis and the token/element axis.
    fn assert_dn_generators_vary() {
        let a = dn_gen(0, DN_N_TOKENS, DN_N_HEADS, DN_HD, 37);
        let b = dn_gen(1, DN_N_TOKENS, DN_N_HEADS, DN_HD, 37);
        assert_ne!(
            a, b,
            "DeltaNet qkv generator: slot index does not perturb the output"
        );
        assert_ne!(
            a[0], a[DN_HD],
            "DeltaNet qkv generator: token/head index does not perturb the output"
        );

        let ga = dn_gate(0, DN_N_TOKENS, DN_N_HEADS);
        let gb = dn_gate(1, DN_N_TOKENS, DN_N_HEADS);
        assert_ne!(
            ga, gb,
            "DeltaNet gate generator: slot index does not perturb the output"
        );
        assert_varying_f32(&ga, "DeltaNet gate generator (own array)");

        let sa = dn_state_q(0, 5);
        let sb = dn_state_q(1, 5);
        assert_ne!(
            sa, sb,
            "DeltaNet initial-state generator: slot index does not perturb q8 state"
        );
        assert_ne!(
            dn_state_q(0, 5),
            dn_state_q(0, 6),
            "DeltaNet initial-state generator: index does not perturb q8 state"
        );
    }

    /// Build ONE slot's OWN, independent s_q8/s_scales host buffers —
    /// sized exactly to what the (now unstrided) kernel reads, since each
    /// slot gets its own device buffer rather than a region carved out of a
    /// shared, strided array. `poisoned`: when true, s_scales is filled
    /// with NaN (int8 body irrelevant — poison propagates through the
    /// S_f4 load, `scale * (float)src[..]`, which is NaN regardless of
    /// `src`'s value, including 0).
    fn build_dn_state(slot: usize, poisoned: bool) -> (Vec<i8>, Vec<f32>) {
        let mut s_q8 = vec![0i8; DN_S_STRIDE];
        let mut s_scales = vec![0f32; DN_N_HEADS * DN_HD];
        for i in 0..DN_S_STRIDE {
            s_q8[i] = if poisoned { 0 } else { dn_state_q(slot, i) };
        }
        for i in 0..(DN_N_HEADS * DN_HD) {
            s_scales[i] = if poisoned {
                f32::NAN
            } else {
                dn_state_scale(slot, i)
            };
        }
        (s_q8, s_scales)
    }

    /// Decode one slot's [n_heads x HD x HD] state into flat f32, applying
    /// each row's own scale (`s_q8[h,row,col] as f32 * s_scales[h,row]`).
    fn decode_dn_state(s_q8: &[i8], s_scales: &[f32], n_heads: usize, hd: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n_heads * hd * hd);
        for h in 0..n_heads {
            for row in 0..hd {
                let scale = s_scales[h * hd + row];
                for col in 0..hd {
                    out.push(s_q8[h * hd * hd + row * hd + col] as f32 * scale);
                }
            }
        }
        out
    }

    /// One slot's private, zero-initialized EF-residual scratch buffer.
    /// `kernels/src/gated_delta_net_q8_fast.hip`'s EF pointer arithmetic
    /// (`s_ef_residual + h*HD*HD + rr*HD`) has NO slot offset applied —
    /// Task 4 only strided s_q8/s_scales, not the EF residual. Sharing one
    /// EF buffer across multiple slots' launches would silently alias
    /// their error-feedback state; this harness sidesteps that (out of
    /// scope for Task 6, which verifies the S-state stride) by giving every
    /// slot-call its own EF buffer, exactly as a caller who wanted
    /// per-slot EF would have to.
    fn dn_ef_buffer(gpu: &mut Gpu) -> GpuTensor {
        gpu.zeros(&[DN_N_HEADS * DN_HD * DN_HD], DType::F16)
            .expect("ef residual")
    }

    struct DnRefResult {
        out: Vec<f32>,
        state: Vec<f32>,
    }

    /// Legacy single-slot reference: fresh unstrided state, own EF buffer.
    #[allow(clippy::too_many_arguments)]
    fn dn_legacy_reference(
        gpu: &mut Gpu,
        slot: usize,
        s_q8_init: &[i8],
        s_scales_init: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
    ) -> DnRefResult {
        let sq_ref = gpu
            .upload_raw(&i8_bytes(s_q8_init), &[DN_N_HEADS * DN_HD * DN_HD])
            .expect("sq_ref upload");
        let sc_ref = gpu
            .upload_f32(s_scales_init, &[DN_N_HEADS * DN_HD])
            .expect("sc_ref upload");
        let q_dev = gpu
            .upload_f32(q, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
            .expect("q upload");
        let k_dev = gpu
            .upload_f32(k, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
            .expect("k upload");
        let v_dev = gpu
            .upload_f32(v, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
            .expect("v upload");
        let gate_dev = gpu
            .upload_f32(gate, &[DN_N_TOKENS * DN_N_HEADS])
            .expect("gate upload");
        let beta_dev = gpu
            .upload_f32(beta, &[DN_N_TOKENS * DN_N_HEADS])
            .expect("beta upload");
        let out_ref = gpu
            .zeros(&[DN_N_TOKENS * DN_N_HEADS * DN_HD], DType::F32)
            .expect("out_ref");
        let ef_ref = dn_ef_buffer(gpu);

        gpu.gated_delta_net_q8_batch_seq(
            &q_dev,
            &k_dev,
            &v_dev,
            &gate_dev,
            &beta_dev,
            &sq_ref,
            &sc_ref,
            &out_ref,
            DN_N_TOKENS,
            DN_N_HEADS,
            DN_HD,
            Some(&ef_ref),
        )
        .expect("legacy gated_delta_net_q8_batch_seq");
        gpu.hip.device_synchronize().expect("sync");

        let out_host = gpu.download_f32(&out_ref).expect("download out_ref");
        let sq_bytes = download_raw(gpu, &sq_ref);
        let sq_host: Vec<i8> = sq_bytes.iter().map(|&b| b as i8).collect();
        let sc_host = gpu.download_f32(&sc_ref).expect("download sc_ref");
        let state = decode_dn_state(&sq_host, &sc_host, DN_N_HEADS, DN_HD);

        gpu.free_tensor(sq_ref).expect("free sq_ref");
        gpu.free_tensor(sc_ref).expect("free sc_ref");
        gpu.free_tensor(q_dev).expect("free q_dev");
        gpu.free_tensor(k_dev).expect("free k_dev");
        gpu.free_tensor(v_dev).expect("free v_dev");
        gpu.free_tensor(gate_dev).expect("free gate_dev");
        gpu.free_tensor(beta_dev).expect("free beta_dev");
        gpu.free_tensor(out_ref).expect("free out_ref");
        gpu.free_tensor(ef_ref).expect("free ef_ref");
        let _ = slot; // slot only feeds the (already-baked-in) generator args above
        DnRefResult {
            out: out_host,
            state,
        }
    }

    /// Step 1 for DeltaNet: for each slot 0..n_slots, issue its OWN
    /// slot-scoped launch into its OWN, independent S buffers (the
    /// stride-retirement design SP2 Task 4 now specifies — one launch per
    /// active slot against that slot's own state, mirroring SP3's
    /// one-`DeltaNetState`-per-slot model), and compare both the output and
    /// the persisted post-update state against the legacy per-slot
    /// reference.
    fn test_dn_golden(gpu: &mut Gpu, n_slots: usize) {
        let states: Vec<(Vec<i8>, Vec<f32>)> = (0..n_slots)
            .map(|slot| build_dn_state(slot, false))
            .collect();
        let all_q8: Vec<i8> = states.iter().flat_map(|(q, _)| q.iter().copied()).collect();
        let all_scales: Vec<f32> = states.iter().flat_map(|(_, s)| s.iter().copied()).collect();
        assert_varying_i8(
            &all_q8,
            &format!("DN golden initial q8 state n_slots={n_slots}"),
        );
        assert_varying_f32(
            &all_scales,
            &format!("DN golden initial scales n_slots={n_slots}"),
        );

        for slot in 0..n_slots {
            let (s_q8_init, s_scales_init) = &states[slot];
            let q = dn_gen(slot, DN_N_TOKENS, DN_N_HEADS, DN_HD, 3);
            let k = dn_gen(slot, DN_N_TOKENS, DN_N_HEADS, DN_HD, 5);
            let v = dn_gen(slot, DN_N_TOKENS, DN_N_HEADS, DN_HD, 7);
            let gate = dn_gate(slot, DN_N_TOKENS, DN_N_HEADS);
            let beta = dn_beta(slot, DN_N_TOKENS, DN_N_HEADS);

            let q_dev = gpu
                .upload_f32(&q, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
                .expect("q upload");
            let k_dev = gpu
                .upload_f32(&k, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
                .expect("k upload");
            let v_dev = gpu
                .upload_f32(&v, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
                .expect("v upload");
            let gate_dev = gpu
                .upload_f32(&gate, &[DN_N_TOKENS * DN_N_HEADS])
                .expect("gate upload");
            let beta_dev = gpu
                .upload_f32(&beta, &[DN_N_TOKENS * DN_N_HEADS])
                .expect("beta upload");
            let out = gpu
                .zeros(&[DN_N_TOKENS * DN_N_HEADS * DN_HD], DType::F32)
                .expect("out");
            let ef = dn_ef_buffer(gpu);
            let s_q8_dev = gpu
                .upload_raw(&i8_bytes(s_q8_init), &[DN_S_STRIDE])
                .expect("s_q8 upload");
            let s_scales_dev = gpu
                .upload_f32(s_scales_init, &[DN_N_HEADS * DN_HD])
                .expect("s_scales upload");

            gpu.gated_delta_net_q8_batch_seq(
                &q_dev,
                &k_dev,
                &v_dev,
                &gate_dev,
                &beta_dev,
                &s_q8_dev,
                &s_scales_dev,
                &out,
                DN_N_TOKENS,
                DN_N_HEADS,
                DN_HD,
                Some(&ef),
            )
            .expect("gated_delta_net_q8_batch_seq");
            gpu.hip.device_synchronize().expect("sync");
            let out_host = gpu.download_f32(&out).expect("download out");

            let sq_bytes = download_raw(gpu, &s_q8_dev);
            let sq_host: Vec<i8> = sq_bytes.iter().map(|&b| b as i8).collect();
            let sc_host = gpu.download_f32(&s_scales_dev).expect("download s_scales");
            let state = decode_dn_state(&sq_host, &sc_host, DN_N_HEADS, DN_HD);

            gpu.free_tensor(q_dev).expect("free q_dev");
            gpu.free_tensor(k_dev).expect("free k_dev");
            gpu.free_tensor(v_dev).expect("free v_dev");
            gpu.free_tensor(gate_dev).expect("free gate_dev");
            gpu.free_tensor(beta_dev).expect("free beta_dev");
            gpu.free_tensor(out).expect("free out");
            gpu.free_tensor(ef).expect("free ef");
            gpu.free_tensor(s_q8_dev).expect("free s_q8_dev");
            gpu.free_tensor(s_scales_dev).expect("free s_scales_dev");

            let reference = dn_legacy_reference(
                gpu,
                slot,
                s_q8_init,
                s_scales_init,
                &q,
                &k,
                &v,
                &gate,
                &beta,
            );

            assert_close(
                &format!("DN golden output n_slots={n_slots} slot={slot}"),
                &out_host,
                &reference.out,
            );
            assert_close(
                &format!("DN golden post-state n_slots={n_slots} slot={slot}"),
                &state,
                &reference.state,
            );
        }
    }

    /// Step 2 for DeltaNet: poison every OTHER slot's OWN device buffer,
    /// run the target's launch against the target's OWN (unpoisoned)
    /// buffer, and require its output/state stay finite and match the
    /// reference. With the shared strided buffer retired there is no
    /// address space left for a neighbour's poison to leak through — this
    /// no longer probes stride addressing. What it still verifies: the
    /// harness allocates every slot's buffer independently, keeps the
    /// poisoned neighbours simultaneously live on the device (not freed
    /// before the target's launch), and wires the TARGET's own handle —
    /// not a neighbour's — into the target's call. That's a real, if
    /// narrower, thing to get wrong (e.g. an off-by-one in which `states[i]`
    /// pairs with which device buffer), so it stays worth checking even
    /// though it can no longer catch an addressing-stride bug.
    fn test_dn_isolation(gpu: &mut Gpu, n_slots: usize) {
        if n_slots < 2 {
            println!("  DN isolation n_slots=1: skipped (no neighbour to poison)");
            return;
        }
        let target = n_slots - 1;
        let states: Vec<(Vec<i8>, Vec<f32>)> = (0..n_slots)
            .map(|slot| build_dn_state(slot, slot != target))
            .collect();
        // Upload every slot's buffer (not just target's) and keep them all
        // alive through the target's launch, so a wrong-handle bug has
        // live, poisoned neighbours to misroute into.
        let device_bufs: Vec<(GpuTensor, GpuTensor)> = states
            .iter()
            .map(|(s_q8, s_scales)| {
                let s_q8_dev = gpu
                    .upload_raw(&i8_bytes(s_q8), &[DN_S_STRIDE])
                    .expect("s_q8 upload");
                let s_scales_dev = gpu
                    .upload_f32(s_scales, &[DN_N_HEADS * DN_HD])
                    .expect("s_scales upload");
                (s_q8_dev, s_scales_dev)
            })
            .collect();

        let q = dn_gen(target, DN_N_TOKENS, DN_N_HEADS, DN_HD, 3);
        let k = dn_gen(target, DN_N_TOKENS, DN_N_HEADS, DN_HD, 5);
        let v = dn_gen(target, DN_N_TOKENS, DN_N_HEADS, DN_HD, 7);
        let gate = dn_gate(target, DN_N_TOKENS, DN_N_HEADS);
        let beta = dn_beta(target, DN_N_TOKENS, DN_N_HEADS);

        let q_dev = gpu
            .upload_f32(&q, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
            .expect("q upload");
        let k_dev = gpu
            .upload_f32(&k, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
            .expect("k upload");
        let v_dev = gpu
            .upload_f32(&v, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
            .expect("v upload");
        let gate_dev = gpu
            .upload_f32(&gate, &[DN_N_TOKENS * DN_N_HEADS])
            .expect("gate upload");
        let beta_dev = gpu
            .upload_f32(&beta, &[DN_N_TOKENS * DN_N_HEADS])
            .expect("beta upload");
        let out = gpu
            .zeros(&[DN_N_TOKENS * DN_N_HEADS * DN_HD], DType::F32)
            .expect("out");
        let ef = dn_ef_buffer(gpu);
        let (target_s_q8_dev, target_s_scales_dev) = &device_bufs[target];

        gpu.gated_delta_net_q8_batch_seq(
            &q_dev,
            &k_dev,
            &v_dev,
            &gate_dev,
            &beta_dev,
            target_s_q8_dev,
            target_s_scales_dev,
            &out,
            DN_N_TOKENS,
            DN_N_HEADS,
            DN_HD,
            Some(&ef),
        )
        .expect("isolation launch");
        gpu.hip.device_synchronize().expect("sync");
        let out_host = gpu.download_f32(&out).expect("download out");

        gpu.free_tensor(q_dev).expect("free q_dev");
        gpu.free_tensor(k_dev).expect("free k_dev");
        gpu.free_tensor(v_dev).expect("free v_dev");
        gpu.free_tensor(gate_dev).expect("free gate_dev");
        gpu.free_tensor(beta_dev).expect("free beta_dev");
        gpu.free_tensor(out).expect("free out");
        gpu.free_tensor(ef).expect("free ef");
        for (s_q8_dev, s_scales_dev) in device_bufs {
            gpu.free_tensor(s_q8_dev).expect("free s_q8_dev");
            gpu.free_tensor(s_scales_dev).expect("free s_scales_dev");
        }

        assert!(
            out_host.iter().all(|x| x.is_finite()),
            "DN isolation n_slots={n_slots}: target {target}'s output went non-finite with every \
             OTHER slot's OWN buffer poisoned — a neighbour's buffer got misrouted into target's \
             own S read"
        );

        let (s_q8_init, s_scales_init) = &states[target];
        let reference = dn_legacy_reference(
            gpu,
            target,
            s_q8_init,
            s_scales_init,
            &q,
            &k,
            &v,
            &gate,
            &beta,
        );
        assert_close(
            &format!("DN isolation n_slots={n_slots} target={target}"),
            &out_host,
            &reference.out,
        );
        println!("  DN isolation n_slots={n_slots} target={target}: OK (neighbours poisoned, target finite+correct)");
    }

    /// Positive poison control for DeltaNet, direct SP1-style translation:
    /// poison the TARGET's own S buffer, run the target's own
    /// correctly-routed launch, and assert its output DOES go non-finite.
    /// Proves the NaN scale really does reach this kernel's S read and
    /// really does propagate to output — the fact the isolation check
    /// above stayed finite is therefore meaningful, not a byproduct of an
    /// inert poison mechanism.
    ///
    /// Pre-retirement this poisoned "every slot but a bystander" inside a
    /// shared array and relied on `target != bystander` to poison target.
    /// With per-slot buffers there's no shared array for a clean
    /// bystander to matter to this check — `build_dn_state(target, true)`
    /// is the direct, equivalent replacement: only the target's own
    /// buffer needs to exist, and it needs to be poisoned.
    fn test_dn_poison_is_live(gpu: &mut Gpu) {
        let target = 1usize;
        let (s_q8_host, s_scales_host) = build_dn_state(target, true);
        let s_q8_dev = gpu
            .upload_raw(&i8_bytes(&s_q8_host), &[DN_S_STRIDE])
            .expect("s_q8 upload");
        let s_scales_dev = gpu
            .upload_f32(&s_scales_host, &[DN_N_HEADS * DN_HD])
            .expect("s_scales upload");

        let q = dn_gen(target, DN_N_TOKENS, DN_N_HEADS, DN_HD, 3);
        let k = dn_gen(target, DN_N_TOKENS, DN_N_HEADS, DN_HD, 5);
        let v = dn_gen(target, DN_N_TOKENS, DN_N_HEADS, DN_HD, 7);
        let gate = dn_gate(target, DN_N_TOKENS, DN_N_HEADS);
        let beta = dn_beta(target, DN_N_TOKENS, DN_N_HEADS);

        let q_dev = gpu
            .upload_f32(&q, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
            .expect("q upload");
        let k_dev = gpu
            .upload_f32(&k, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
            .expect("k upload");
        let v_dev = gpu
            .upload_f32(&v, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
            .expect("v upload");
        let gate_dev = gpu
            .upload_f32(&gate, &[DN_N_TOKENS * DN_N_HEADS])
            .expect("gate upload");
        let beta_dev = gpu
            .upload_f32(&beta, &[DN_N_TOKENS * DN_N_HEADS])
            .expect("beta upload");
        let out = gpu
            .zeros(&[DN_N_TOKENS * DN_N_HEADS * DN_HD], DType::F32)
            .expect("out");
        let ef = dn_ef_buffer(gpu);

        gpu.gated_delta_net_q8_batch_seq(
            &q_dev,
            &k_dev,
            &v_dev,
            &gate_dev,
            &beta_dev,
            &s_q8_dev,
            &s_scales_dev,
            &out,
            DN_N_TOKENS,
            DN_N_HEADS,
            DN_HD,
            Some(&ef),
        )
        .expect("poison-is-live launch");
        gpu.hip.device_synchronize().expect("sync");
        let out_host = gpu.download_f32(&out).expect("download out");

        gpu.free_tensor(q_dev).expect("free q_dev");
        gpu.free_tensor(k_dev).expect("free k_dev");
        gpu.free_tensor(v_dev).expect("free v_dev");
        gpu.free_tensor(gate_dev).expect("free gate_dev");
        gpu.free_tensor(beta_dev).expect("free beta_dev");
        gpu.free_tensor(out).expect("free out");
        gpu.free_tensor(ef).expect("free ef");
        gpu.free_tensor(s_q8_dev).expect("free s_q8_dev");
        gpu.free_tensor(s_scales_dev).expect("free s_scales_dev");

        let n_bad = out_host.iter().filter(|x| !x.is_finite()).count();
        assert!(
            n_bad > 0,
            "DN positive poison control: target slot {target}'s OWN S state was poisoned with \
             NaN, but its output stayed fully finite \
             ({} elements, 0 non-finite) — the poison mechanism is not reaching this kernel's \
             device reads, which means the isolation layer's passing checks are meaningless",
            out_host.len()
        );
        println!("  DN positive poison control: OK (target {target}'s poisoned output has {n_bad}/{} non-finite elements)", out_host.len());
    }

    /// Step 3 for DeltaNet: corrupt the CANDIDATE arm's device-side
    /// addressing only. Pre-retirement this rerouted the device-side
    /// `row_slot` value; with `row_slot` gone, the equivalent mistake a
    /// real caller could make is passing the WRONG slot's S buffer handles
    /// into the launch — target's own q/k/v/gate/beta, but wired to
    /// bystander's CLEAN-but-different S buffers — while the reference
    /// uses target's true state. Bystander's state is numerically
    /// distinct, not poisoned, so this produces a finite tolerance-ratio
    /// mismatch rather than the non-finite guard rail.
    fn test_dn_negative_control(gpu: &mut Gpu) {
        let target = 1usize;
        let bystander = 0usize;
        let (target_s_q8, target_s_scales) = build_dn_state(target, false);
        let (bystander_s_q8, bystander_s_scales) = build_dn_state(bystander, false); // clean, numerically distinct

        let q = dn_gen(target, DN_N_TOKENS, DN_N_HEADS, DN_HD, 3);
        let k = dn_gen(target, DN_N_TOKENS, DN_N_HEADS, DN_HD, 5);
        let v = dn_gen(target, DN_N_TOKENS, DN_N_HEADS, DN_HD, 7);
        let gate = dn_gate(target, DN_N_TOKENS, DN_N_HEADS);
        let beta = dn_beta(target, DN_N_TOKENS, DN_N_HEADS);

        let q_dev = gpu
            .upload_f32(&q, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
            .expect("q upload");
        let k_dev = gpu
            .upload_f32(&k, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
            .expect("k upload");
        let v_dev = gpu
            .upload_f32(&v, &[DN_N_TOKENS * DN_N_HEADS * DN_HD])
            .expect("v upload");
        let gate_dev = gpu
            .upload_f32(&gate, &[DN_N_TOKENS * DN_N_HEADS])
            .expect("gate upload");
        let beta_dev = gpu
            .upload_f32(&beta, &[DN_N_TOKENS * DN_N_HEADS])
            .expect("beta upload");
        let out = gpu
            .zeros(&[DN_N_TOKENS * DN_N_HEADS * DN_HD], DType::F32)
            .expect("out");
        let ef = dn_ef_buffer(gpu);
        // Device-side corruption: target's own inputs, but wired to
        // bystander's S buffers.
        let s_q8_dev = gpu
            .upload_raw(&i8_bytes(&bystander_s_q8), &[DN_S_STRIDE])
            .expect("s_q8 upload (corrupt)");
        let s_scales_dev = gpu
            .upload_f32(&bystander_s_scales, &[DN_N_HEADS * DN_HD])
            .expect("s_scales upload (corrupt)");

        gpu.gated_delta_net_q8_batch_seq(
            &q_dev,
            &k_dev,
            &v_dev,
            &gate_dev,
            &beta_dev,
            &s_q8_dev,
            &s_scales_dev,
            &out,
            DN_N_TOKENS,
            DN_N_HEADS,
            DN_HD,
            Some(&ef),
        )
        .expect("negative-control launch");
        gpu.hip.device_synchronize().expect("sync");
        let out_host = gpu.download_f32(&out).expect("download out");

        gpu.free_tensor(q_dev).expect("free q_dev");
        gpu.free_tensor(k_dev).expect("free k_dev");
        gpu.free_tensor(v_dev).expect("free v_dev");
        gpu.free_tensor(gate_dev).expect("free gate_dev");
        gpu.free_tensor(beta_dev).expect("free beta_dev");
        gpu.free_tensor(out).expect("free out");
        gpu.free_tensor(ef).expect("free ef");
        gpu.free_tensor(s_q8_dev).expect("free s_q8_dev");
        gpu.free_tensor(s_scales_dev).expect("free s_scales_dev");

        let reference = dn_legacy_reference(
            gpu,
            target,
            &target_s_q8,
            &target_s_scales,
            &q,
            &k,
            &v,
            &gate,
            &beta,
        );

        expect_mismatch("DN negative control", move || {
            assert_close(
                "DN negative control (S buffers rerouted to a clean neighbour, candidate arm only)",
                &out_host,
                &reference.out,
            );
        });
    }

    // ═══════════════════════════════ main ═══════════════════════════════

    /// Total bytes this run holds live at once, at its largest point
    /// (n_slots=8 for both components). Computed from the real per-buffer
    /// formulas rather than a guessed round number — a prior SP2 harness
    /// review found a ~30% undercount from omitting host-side `Vec`s, so
    /// this deliberately budgets more than one live copy per buffer (device,
    /// host build, and host download-back) even though the actual peak, with
    /// this harness's per-iteration frees, is smaller.
    fn total_planned_bytes() -> u64 {
        let kv_cap = kv_cap_tokens(kv_seq_len(KV_MAX_SLOTS - 1));
        let kv_arena = KV_MAX_SLOTS * kv_cap * KV_PER_POS_BYTES;
        // 1 device arena + up to 3 host-side copies (poison build, upload
        // staging, decode-back) live at various points across the sweep.
        let kv_total = kv_arena as u64 * 4
            // Per-slot legacy-reference arena (device+host), summed across
            // the whole n_slots=8 sweep rather than tracked as O(1) live.
            + (KV_MAX_SLOTS as u64) * (kv_cap * KV_PER_POS_BYTES) as u64 * 2;

        let dn_shared = DN_MAX_SLOTS * DN_S_STRIDE + DN_MAX_SLOTS * DN_S_STRIDE * 4; // s_q8 + s_scales bytes
        let dn_total = dn_shared as u64 * 3 // device + host build + host download-back
            + (DN_MAX_SLOTS as u64) * (DN_S_STRIDE as u64) * 6; // per-slot q/k/v/out/ef/ref-state scratch, summed conservatively

        kv_total + dn_total
    }

    pub fn run() {
        println!("### Step 4: generator variance (must hold before trusting any result) ###");
        assert_kv_generator_varies();
        assert_dn_generators_vary();
        println!("  KV write and DeltaNet generators vary by slot AND position/token: OK\n");

        let planned = total_planned_bytes();
        kv_slots::preflight_alloc(planned, R9700_VRAM_BYTES, "test_multislot_ops")
            .expect("preflight_alloc refused this configuration");

        let mut gpu = Gpu::init().expect("gpu init");

        println!("### KV write (SP2 Task 3): golden equivalence, slots 1..=8 ###");
        for n in 1..=KV_MAX_SLOTS {
            test_kv_write_golden(&mut gpu, n);
        }
        println!("\n### KV write: cross-slot isolation + positive control, slots 2..=8 ###");
        for n in 2..=KV_MAX_SLOTS {
            test_kv_write_isolation(&mut gpu, n);
        }
        println!("\n### KV write: negative control (candidate arm only) ###");
        test_kv_write_negative_control(&mut gpu);

        println!(
            "\n### KV write PAGED (SP4 paged block tables): golden across page boundaries ###"
        );
        test_kv_write_paged_golden(&mut gpu);
        println!("\n### KV write PAGED: cross-slot isolation over scattered pages ###");
        test_kv_write_paged_isolation(&mut gpu);
        println!("\n### KV write PAGED: attention read path vs gathered-legacy arm ###");
        test_attn_paged_matches_legacy(&mut gpu);

        println!("\n### DeltaNet (SP2 Task 4): golden equivalence, slots 1..=8 ###");
        for n in 1..=DN_MAX_SLOTS {
            test_dn_golden(&mut gpu, n);
        }
        println!("\n### DeltaNet: cross-slot isolation, slots 2..=8 ###");
        for n in 2..=DN_MAX_SLOTS {
            test_dn_isolation(&mut gpu, n);
        }
        println!("\n### DeltaNet: positive poison control ###");
        test_dn_poison_is_live(&mut gpu);
        println!("\n### DeltaNet: negative control (candidate arm only) ###");
        test_dn_negative_control(&mut gpu);

        println!("\nALL COMPONENTS PASS");
    }
}
