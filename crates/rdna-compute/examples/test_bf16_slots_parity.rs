// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// GPU validation for the slots bf16 tier's parity route and the fp8 tier's
// fail-closed gate. Companion to `test_kv_slot_desc_ports.rs` (rotated
// tiers); this example covers the two native tiers the slot engine admits:
//
//   1. bf16 scalar slots attend — `attention_bf16_kv_batched_slots` (the
//      kernel the slots Bf16 arm routes through for bit-parity with the
//      sequential `AttnBf16KvBatchedMasked`):
//        a. zero-base parity    — null descriptors vs a zero-base descriptor
//                                 launch must be BIT-identical (the
//                                 port's backwards-compatibility contract).
//        b. non-zero-base       — a 2-slot arena whose slot-0 slabs are
//           understudy            NaN-poisoned while slot 1 holds the real
//                                 data at `legacy_*_base = slab_bytes`; every
//                                 row assigned slot 1. If the kernel ignored
//                                 the base fields the output picks up slot 0's
//                                 poison and diverges hard.
//        c. two-slot mix        — rows alternate between two REAL slots;
//                                 each row's output must equal a plain-cache
//                                 run over that slot's data, bit for bit.
//   2. bf16 slots write parity — `kv_cache_write_bf16_batched` with real
//      descriptors must land bit-identical bytes at the slot base as the
//      plain legacy write, leaving the poisoned slot-0 slab untouched.
//   3. fp8 fail-closed gate    — off gfx1201, both fp8 slots entry points
//      (`kv_cache_write_fp8_e4m3_batched_slots`,
//      `attention_flash_fp8_e4m3_batched_masked_slots`) must return an Err:
//      the kernel TU carries a `#error requires --offload-arch=gfx1201`
//      guard, so the JIT compile refuses before any launch. On a gfx1201
//      host this check is skipped (it cannot prove the refusal there).
//
// Run (GPU required):
//   cargo run --release -p rdna-compute --example test_bf16_slots_parity

use rdna_compute::kv_slots::KvSlotDesc;
use rdna_compute::{Gpu, GpuTensor};

const N_KV_HEADS: usize = 2;
const N_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const BATCH: usize = 4;
const SLOTS: usize = 2;
const SLAB_TOKENS: usize = 256;
const MAX_CTX: usize = 256;
const POSITIONS: [i32; BATCH] = [31, 63, 127, 200];

/// bf16 token row: 2 bytes/element, no scales.
fn row_bytes() -> usize {
    N_KV_HEADS * HEAD_DIM * 2
}

struct Rng(u32);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        self.0
    }
    fn f32_unit(&mut self) -> f32 {
        (self.next_u32() % 10_000) as f32 / 10_000.0
    }
}

/// Random FINITE bf16 cache bytes: truncation-converted f32 in [-2, 2) —
/// same bytes both arms read, so parity never depends on the conversion.
fn bf16_slab(rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(SLAB_TOKENS * row_bytes());
    for _ in 0..SLAB_TOKENS * N_KV_HEADS * HEAD_DIM {
        let v = rng.f32_unit() * 4.0 - 2.0;
        out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_ne_bytes());
    }
    out
}

/// NaN-poisoned bf16 slab (0x7FC0 quiet NaN pattern) — any read of it shows
/// up in the attention output, which is what the understudy check relies on.
fn poisoned_slab() -> Vec<u8> {
    vec![0xFCu8, 0x7F].repeat(SLAB_TOKENS * row_bytes() / 2)
}

fn rand_f32_vec(n: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|_| rng.f32_unit() * 2.0 - 1.0).collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn download_raw(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    let mut buf = vec![0u8; t.buf.size()];
    gpu.hip.memcpy_dtoh(&mut buf, &t.buf).expect("download raw");
    buf
}

fn download_f32(gpu: &Gpu, t: &GpuTensor) -> Vec<f32> {
    download_raw(gpu, t)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

fn upload_raw(gpu: &Gpu, bytes: &[u8]) -> GpuTensor {
    gpu.upload_raw(bytes, &[bytes.len()]).expect("upload raw")
}

fn upload_f32(gpu: &mut Gpu, vals: &[f32]) -> GpuTensor {
    gpu.upload_f32(vals, &[vals.len()]).expect("upload f32")
}

/// Legacy-mode (page_tokens = 0) descriptors; K and V share the base because
/// the bf16 (and fp8) strides are equal.
fn slab_descs(gpu: &Gpu, bases: [u64; SLOTS]) -> GpuTensor {
    let descs: Vec<KvSlotDesc> = (0..SLOTS)
        .map(|s| KvSlotDesc {
            block_table: 0,
            legacy_k_base: bases[s],
            legacy_v_base: bases[s],
            seq_len: SLAB_TOKENS as i32,
            page_tokens: 0,
        })
        .collect();
    let mut bytes = Vec::with_capacity(SLOTS * 32);
    for d in &descs {
        bytes.extend_from_slice(&d.block_table.to_ne_bytes());
        bytes.extend_from_slice(&d.legacy_k_base.to_ne_bytes());
        bytes.extend_from_slice(&d.legacy_v_base.to_ne_bytes());
        bytes.extend_from_slice(&d.seq_len.to_ne_bytes());
        bytes.extend_from_slice(&d.page_tokens.to_ne_bytes());
    }
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    upload_raw(gpu, &bytes)
}

fn positions_dev(gpu: &Gpu) -> GpuTensor {
    upload_raw(gpu, &i32_bytes(&POSITIONS))
}

fn row_slot(gpu: &Gpu, slots: [i32; BATCH]) -> GpuTensor {
    upload_raw(gpu, &i32_bytes(&slots))
}

fn out_dev(gpu: &mut Gpu) -> GpuTensor {
    upload_f32(gpu, &vec![0.0f32; BATCH * N_HEADS * HEAD_DIM])
}

/// One `attention_bf16_kv_batched_slots` launch with the full explicit-arg
/// shape the slots engine passes (tree verify out of scope: None, 0, 0).
#[allow(clippy::too_many_arguments)]
fn bf16_attn_slots(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    out: &GpuTensor,
    pos: &GpuTensor,
    descs: Option<&GpuTensor>,
    rows: Option<&GpuTensor>,
) {
    gpu.attention_bf16_kv_batched_slots(
        q,
        k,
        v,
        out,
        pos,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        SLAB_TOKENS,   // max_seq (arena cap)
        MAX_CTX,       // max_ctx_len
        BATCH,
        None,          // tree_bias
        0,
        0,
        descs,
        rows,
    )
    .expect("bf16 slots attend");
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("arch: {}", gpu.arch);
    let mut rng = Rng(0x5eed_1234);

    let pos = positions_dev(&gpu);
    let q_data = rand_f32_vec(BATCH * N_HEADS * HEAD_DIM, &mut rng);
    let q = upload_f32(&mut gpu, &q_data);

    // ── Check 1a: zero-base descriptor parity ────────────────────────────
    // Real K/V in a single plain cache; the reference runs the legacy path
    // (null descriptors), the candidate passes a zero-base descriptor table.
    let k_real = bf16_slab(&mut rng);
    let v_real = bf16_slab(&mut rng);
    let k_plain = upload_raw(&gpu, &k_real);
    let v_plain = upload_raw(&gpu, &v_real);
    let out_ref = out_dev(&mut gpu);
    bf16_attn_slots(&mut gpu, &q, &k_plain, &v_plain, &out_ref, &pos, None, None);
    let descs_zero = slab_descs(&gpu, [0; SLOTS]);
    let rows1 = row_slot(&gpu, [1; BATCH]);
    let out_zero = out_dev(&mut gpu);
    bf16_attn_slots(
        &mut gpu,
        &q,
        &k_plain,
        &v_plain,
        &out_zero,
        &pos,
        Some(&descs_zero),
        Some(&rows1),
    );
    let (dr, dz) = (download_f32(&gpu, &out_ref), download_f32(&gpu, &out_zero));
    assert!(
        dr == dz,
        "check 1a: zero-base descriptor launch diverged from the legacy path"
    );
    eprintln!("check 1a zero-base parity: BIT-EXACT ({} elements)", dr.len());

    // ── Check 1b: non-zero-base understudy ───────────────────────────────
    // Arena = [NaN-poisoned slot-0 slabs][real data at slab_bytes]; every
    // row reads slot 1. A base-ignoring kernel swallows poison → mismatch.
    let k_arena_bytes = [poisoned_slab(), k_real.clone()].concat();
    let v_arena_bytes = [poisoned_slab(), v_real.clone()].concat();
    let k_arena = upload_raw(&gpu, &k_arena_bytes);
    let v_arena = upload_raw(&gpu, &v_arena_bytes);
    let descs_off = slab_descs(&gpu, [0, SLAB_TOKENS as u64 * row_bytes() as u64]);
    let out_under = out_dev(&mut gpu);
    bf16_attn_slots(
        &mut gpu,
        &q,
        &k_arena,
        &v_arena,
        &out_under,
        &pos,
        Some(&descs_off),
        Some(&rows1),
    );
    let du = download_f32(&gpu, &out_under);
    assert!(du == dr, "check 1b: slot-1 base honoring diverged from the plain-cache run");
    eprintln!("check 1b non-zero-base understudy: BIT-EXACT, poison untouched");

    // ── Check 1c: two-slot mix ───────────────────────────────────────────
    // Slot 0 and slot 1 hold DIFFERENT real data; rows alternate. Each row's
    // output must equal a plain-cache run over ITS slot's data.
    let k0 = bf16_slab(&mut rng);
    let v0 = bf16_slab(&mut rng);
    let k_mix = [k0.clone(), k_real.clone()].concat();
    let v_mix = [v0.clone(), v_real.clone()].concat();
    let k_mix_dev = upload_raw(&gpu, &k_mix);
    let v_mix_dev = upload_raw(&gpu, &v_mix);
    let descs_mix = slab_descs(&gpu, [0, SLAB_TOKENS as u64 * row_bytes() as u64]);
    let rows_alt = row_slot(&gpu, [0, 1, 0, 1]);
    let out_mix = out_dev(&mut gpu);
    bf16_attn_slots(
        &mut gpu,
        &q,
        &k_mix_dev,
        &v_mix_dev,
        &out_mix,
        &pos,
        Some(&descs_mix),
        Some(&rows_alt),
    );
    // Per-row references: plain-cache runs over each slot's data alone.
    let k_slot0 = upload_raw(&gpu, &k0);
    let v_slot0 = upload_raw(&gpu, &v0);
    let out_ref0 = out_dev(&mut gpu);
    bf16_attn_slots(&mut gpu, &q, &k_slot0, &v_slot0, &out_ref0, &pos, None, None);
    let out_ref1 = out_dev(&mut gpu);
    bf16_attn_slots(&mut gpu, &q, &k_plain, &v_plain, &out_ref1, &pos, None, None);
    let d_mix = download_f32(&gpu, &out_mix);
    let d0 = download_f32(&gpu, &out_ref0);
    let d1 = download_f32(&gpu, &out_ref1);
    let stride = N_HEADS * HEAD_DIM;
    for (row, want0) in [true, false, true, false].iter().enumerate() {
        let want = if *want0 { &d0 } else { &d1 };
        let got = &d_mix[row * stride..(row + 1) * stride];
        let refv = &want[row * stride..(row + 1) * stride];
        assert!(
            got == refv,
            "check 1c: row {row} (slot {}) diverged from its plain-cache reference",
            if *want0 { 0 } else { 1 }
        );
    }
    eprintln!("check 1c two-slot mix: BIT-EXACT per row");

    // ── Check 2: bf16 slots WRITE parity ─────────────────────────────────
    // The descriptor write into slot 1's slab must land the same bytes as
    // the plain write into a zeroed slab, leaving slot 0's poison intact.
    let src_data: Vec<f32> = (0..BATCH * N_KV_HEADS * HEAD_DIM)
        .map(|_| rng.f32_unit() * 2.0 - 1.0)
        .collect();
    let src = upload_f32(&mut gpu, &src_data);
    let ref_slab = vec![0u8; SLAB_TOKENS * row_bytes()];
    let k_ref = upload_raw(&gpu, &ref_slab);
    gpu.kv_cache_write_bf16_batched(
        &k_ref,
        &src,
        &pos,
        N_KV_HEADS,
        HEAD_DIM,
        BATCH,
        None,
        None,
    )
    .expect("bf16 plain write");
    let cand_arena = [poisoned_slab(), vec![0u8; SLAB_TOKENS * row_bytes()]].concat();
    let k_cand = upload_raw(&gpu, &cand_arena);
    gpu.kv_cache_write_bf16_batched(
        &k_cand,
        &src,
        &pos,
        N_KV_HEADS,
        HEAD_DIM,
        BATCH,
        Some(&descs_off),
        Some(&rows1),
    )
    .expect("bf16 slots write");
    let ref_bytes = download_raw(&gpu, &k_ref);
    let cand_bytes = download_raw(&gpu, &k_cand);
    let slab = SLAB_TOKENS * row_bytes();
    assert!(
        cand_bytes[slab..] == ref_bytes[..],
        "check 2: descriptor write bytes at the slot base differ from the legacy write"
    );
    let cand0 = &cand_bytes[..slab];
    let poison = poisoned_slab();
    assert!(cand0 == &poison[..], "check 2: slot-0 poison slab was written through");
    eprintln!("check 2 bf16 slots write parity: BIT-EXACT, slot 0 untouched");

    // ── Check 3: fp8 fail-closed off gfx1201 ─────────────────────────────
    // The fp8 kernel TU carries `#error "… requires --offload-arch=gfx1201"`,
    // so off gfx1201 the JIT compile must refuse before any launch. On a
    // gfx1201 host this check cannot prove the refusal and is skipped.
    if gpu.arch == "gfx1201" {
        eprintln!("check 3 fp8 fail-closed: skipped on gfx1201 (admission host)");
    } else {
        let fp8_cache = upload_raw(&gpu, &vec![0u8; SLAB_TOKENS * row_bytes()]);
        let err = gpu.kv_cache_write_fp8_e4m3_batched_slots(
            &fp8_cache,
            &src,
            &pos,
            N_KV_HEADS,
            HEAD_DIM,
            BATCH,
            Some(&descs_off),
            Some(&rows1),
        )
        .err()
        .expect("check 3: fp8 slots write must fail closed off gfx1201");
        eprintln!("check 3 fp8 slots write refused: {err}");
        let fp8_out = out_dev(&mut gpu);
        let fp8_partials = out_dev(&mut gpu);
        let err = gpu.attention_flash_fp8_e4m3_batched_masked_slots(
            &q,
            &fp8_cache,
            &fp8_cache,
            &fp8_out,
            &pos,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            SLAB_TOKENS,
            MAX_CTX,
            BATCH,
            &fp8_partials,
            None,
            0,
            0,
            Some(&descs_off),
            Some(&rows1),
        )
        .err()
        .expect("check 3: fp8 slots attend must fail closed off gfx1201");
        eprintln!("check 3 fp8 slots attend refused: {err}");
    }

    eprintln!("ALL CHECKS PASSED on {}", gpu.arch);
}
