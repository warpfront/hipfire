// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// GPU validation for the multi-slot descriptor ports of the rotated KV
// tiers — the kernels the multi-slot engine routes through when the slots
// KV tier is not q8:
//
//   tile-batched attention:  attention_flash_{asym2,asym3,asym4,fwht2,fwht3,fwht4}_tile_batched
//   batched K writers:       kv_cache_write_asym_k_{givens2,givens3,givens4,fwht2,fwht3,fwht4}_batched
//   Q8 V writer:             kv_cache_write_q8_0_batched (use_v_base arm)
//
// Every check runs against the tier's own LEGACY (descriptor-free) entry
// points, which predate the ports and have GPU coverage of their own:
//
//   1. zero-base parity         — the `_slots` path with null-base
//                                 legacy-mode descriptors must be
//                                 BIT-identical to the plain path (the
//                                 port's backwards-compatibility contract).
//   2. non-zero-base understudy — a 2-slot arena whose slot-0 slab is
//                                 NaN-poisoned while slot 1's slab holds the
//                                 real data at `legacy_*_base = slab_bytes`,
//                                 every row assigned slot 1. If the kernel
//                                 ignored the base fields (the failure mode
//                                 that motivated the ports) the output picks
//                                 up slot 0's poison and diverges hard.
//   3. composite K/V write      — the `_batched_slots` writer against the
//                                 plain `_batched` writer over ZEROED slabs:
//                                 the bytes landing at the slot base must be
//                                 bit-identical to the legacy write, and the
//                                 poisoned slot-0 slab must be untouched.
//   4. q8 use_v_base arm        — with legacy_v_base != legacy_k_base, the V
//                                 write must land at the V base, and the flag
//                                 must actually flip addressing (writing with
//                                 use_v_base=false lands at the K base).
//
// Run (GPU required — bare-metal ROCm box or the ROCm container with
// /dev/kfd passed through):
//   cargo run --release -p rdna-compute --example test_kv_slot_desc_ports

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
/// Tile size the flash path walks KV in — keep MAX_CTX a multiple of it.
const TILE: usize = 128;
const TOL: f32 = 1e-5;

const TIERS: [&str; 6] = ["asym2", "asym3", "asym4", "fwht2", "fwht3", "fwht4"];

fn k_bytes_per_head(tier: &str) -> usize {
    match tier {
        "asym2" | "fwht2" => 4 + HEAD_DIM / 4,
        "asym3" | "fwht3" => 4 + (HEAD_DIM * 3) / 8,
        "asym4" | "fwht4" => 4 + HEAD_DIM / 2,
        other => panic!("unknown tier {other}"),
    }
}

fn k_bytes_per_pos(tier: &str) -> usize {
    N_KV_HEADS * k_bytes_per_head(tier)
}

fn v_bytes_per_pos() -> usize {
    N_KV_HEADS * (HEAD_DIM / 32) * 34
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
    fn byte(&mut self) -> u8 {
        (self.next_u32() >> 16) as u8
    }
}

/// K slab bytes for one slot: finite, varying f32 norm headers (a NaN header
/// read as `cnorm` poisons the whole dot-product — check 2 relies on that)
/// over arbitrary packed bodies. Arbitrary body bytes are safe: they only
/// ever select entries from the tiers' bounded dequant tables.
fn k_slab(tier: &str, rng: &mut Rng, poison: bool) -> Vec<u8> {
    let bph = k_bytes_per_head(tier);
    let mut out = Vec::with_capacity(SLAB_TOKENS * k_bytes_per_pos(tier));
    for _ in 0..SLAB_TOKENS {
        for _ in 0..N_KV_HEADS {
            let header: f32 = if poison {
                f32::NAN
            } else {
                0.05 + rng.f32_unit() * 0.9
            };
            out.extend_from_slice(&header.to_ne_bytes());
            for _ in 0..bph - 4 {
                out.push(rng.byte());
            }
        }
    }
    out
}

/// Q8_0 V slab bytes: f16 scale (finite, or the 0x7E00 NaN sentinel when
/// poisoned) + arbitrary i8 payload.
fn v_slab(rng: &mut Rng, poison: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(SLAB_TOKENS * v_bytes_per_pos());
    for _ in 0..SLAB_TOKENS {
        for _ in 0..N_KV_HEADS * (HEAD_DIM / 32) {
            let scale_bits: u16 = if poison { 0x7E00 } else { 0x3800 };
            out.extend_from_slice(&scale_bits.to_ne_bytes());
            for _ in 0..32 {
                out.push(rng.byte());
            }
        }
    }
    out
}

fn rand_f32_vec(n: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|_| rng.f32_unit() * 2.0 - 1.0).collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

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

/// Legacy-mode (page_tokens = 0) descriptors for the 2-slot slab arena.
fn slab_descs(gpu: &Gpu, k_bases: [u64; SLOTS], v_bases: [u64; SLOTS]) -> GpuTensor {
    let descs: Vec<KvSlotDesc> = (0..SLOTS)
        .map(|s| KvSlotDesc {
            block_table: 0,
            legacy_k_base: k_bases[s],
            legacy_v_base: v_bases[s],
            seq_len: SLAB_TOKENS as i32,
            page_tokens: 0,
        })
        .collect();
    let mut bytes = pack_descs(&descs);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    upload_raw(gpu, &bytes)
}

struct Tables {
    cos: GpuTensor,
    sin: GpuTensor,
    signs1: GpuTensor,
    signs2: GpuTensor,
}

fn upload_tables(gpu: &mut Gpu, rng: &mut Rng) -> Tables {
    // Table CONTENT is arbitrary here — the same tables feed the candidate
    // and the reference, so parity does not depend on the canonical seeds
    // (those live on KvCache: gen_givens_angles / gen_fwht_signs).
    let cos = upload_f32(gpu, &rand_f32_vec(HEAD_DIM / 2, rng));
    let sin = upload_f32(gpu, &rand_f32_vec(HEAD_DIM / 2, rng));
    let signs1 = upload_f32(
        gpu,
        &(0..128)
            .map(|_| {
                if rng.next_u32() & 1 == 0 {
                    1.0f32
                } else {
                    -1.0f32
                }
            })
            .collect::<Vec<_>>(),
    );
    let signs2 = upload_f32(
        gpu,
        &(0..128)
            .map(|_| {
                if rng.next_u32() & 1 == 0 {
                    1.0f32
                } else {
                    -1.0f32
                }
            })
            .collect::<Vec<_>>(),
    );
    Tables {
        cos,
        sin,
        signs1,
        signs2,
    }
}

fn positions_dev(gpu: &Gpu) -> GpuTensor {
    upload_raw(gpu, &i32_bytes(&POSITIONS))
}

fn row_slot_all(gpu: &Gpu, slot: i32) -> GpuTensor {
    upload_raw(gpu, &i32_bytes(&[slot; BATCH]))
}

fn row_slot_alt(gpu: &Gpu) -> GpuTensor {
    upload_raw(gpu, &i32_bytes(&[0, 1, 0, 1]))
}

fn partials_dev(gpu: &mut Gpu) -> GpuTensor {
    let max_tiles = MAX_CTX.div_ceil(TILE);
    upload_f32(
        gpu,
        &vec![0.0f32; BATCH * N_HEADS * max_tiles * (2 + HEAD_DIM)],
    )
}

fn out_dev(gpu: &mut Gpu) -> GpuTensor {
    upload_f32(gpu, &vec![0.0f32; BATCH * N_HEADS * HEAD_DIM])
}

fn max_abs_diff(gpu: &Gpu, a: &GpuTensor, b: &GpuTensor) -> f32 {
    let (da, db) = (download_f32(gpu, a), download_f32(gpu, b));
    da.iter()
        .zip(db.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Check 1 + 2 for one tier's attention. Returns (parity_diff, understudy_diff).
fn attn_checks(gpu: &mut Gpu, tier: &str, t: &Tables, rng: &mut Rng) -> (f32, f32) {
    let q_data = rand_f32_vec(BATCH * N_HEADS * HEAD_DIM, rng);
    let q = upload_f32(gpu, &q_data);
    let pos = positions_dev(gpu);
    let partials = partials_dev(gpu);

    // ── Check 1: zero-base parity on a clean single-slab arena ──────────
    let k_clean = upload_raw(gpu, &k_slab(tier, rng, false));
    let v_clean = upload_raw(gpu, &v_slab(rng, false));
    // descs: both slots at base 0 (rows map to slots 0/1 alternately; both
    // resolve to the same zero-base arena, exactly the legacy contract).
    let descs_zero = slab_descs(gpu, [0; SLOTS], [0; SLOTS]);
    let rs_alt = row_slot_alt(gpu);
    let out_slots = out_dev(gpu);
    let out_ref = out_dev(gpu);
    let parity = {
        match tier {
            "asym2" => gpu.attention_flash_asym2_batched_slots(
                &q,
                &k_clean,
                &v_clean,
                &out_slots,
                &pos,
                &t.cos,
                &t.sin,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                Some(&descs_zero),
                Some(&rs_alt),
            ),
            "asym3" => gpu.attention_flash_asym3_batched_masked_slots(
                &q,
                &k_clean,
                &v_clean,
                &out_slots,
                &pos,
                &t.cos,
                &t.sin,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
                Some(&descs_zero),
                Some(&rs_alt),
            ),
            "asym4" => gpu.attention_flash_asym4_batched_masked_slots(
                &q,
                &k_clean,
                &v_clean,
                &out_slots,
                &pos,
                &t.cos,
                &t.sin,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
                Some(&descs_zero),
                Some(&rs_alt),
            ),
            "fwht2" => gpu.attention_flash_fwht2_batched_slots(
                &q,
                &k_clean,
                &v_clean,
                &out_slots,
                &pos,
                &t.signs1,
                &t.signs2,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                Some(&descs_zero),
                Some(&rs_alt),
            ),
            "fwht3" => gpu.attention_flash_fwht3_batched_masked_slots(
                &q,
                &k_clean,
                &v_clean,
                &out_slots,
                &pos,
                &t.signs1,
                &t.signs2,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
                8,
                Some(&descs_zero),
                Some(&rs_alt),
            ),
            "fwht4" => gpu.attention_flash_fwht4_batched_masked_slots(
                &q,
                &k_clean,
                &v_clean,
                &out_slots,
                &pos,
                &t.signs1,
                &t.signs2,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
                Some(&descs_zero),
                Some(&rs_alt),
            ),
            other => panic!("unknown tier {other}"),
        }
        .expect("slots attention launch");
        match tier {
            "asym2" => gpu.attention_flash_asym2_batched(
                &q,
                &k_clean,
                &v_clean,
                &out_ref,
                &pos,
                &t.cos,
                &t.sin,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
            ),
            "asym3" => gpu.attention_flash_asym3_batched_masked(
                &q,
                &k_clean,
                &v_clean,
                &out_ref,
                &pos,
                &t.cos,
                &t.sin,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
            ),
            "asym4" => gpu.attention_flash_asym4_batched_masked(
                &q,
                &k_clean,
                &v_clean,
                &out_ref,
                &pos,
                &t.cos,
                &t.sin,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
            ),
            "fwht2" => gpu.attention_flash_fwht2_batched(
                &q,
                &k_clean,
                &v_clean,
                &out_ref,
                &pos,
                &t.signs1,
                &t.signs2,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                8,
            ),
            "fwht3" => gpu.attention_flash_fwht3_batched_masked(
                &q,
                &k_clean,
                &v_clean,
                &out_ref,
                &pos,
                &t.signs1,
                &t.signs2,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
                8,
            ),
            "fwht4" => gpu.attention_flash_fwht4_batched_masked(
                &q,
                &k_clean,
                &v_clean,
                &out_ref,
                &pos,
                &t.signs1,
                &t.signs2,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
                8,
            ),
            other => panic!("unknown tier {other}"),
        }
        .expect("reference attention launch");
        gpu.hip.device_synchronize().expect("sync");
        max_abs_diff(gpu, &out_slots, &out_ref)
    };

    // ── Check 2: non-zero-base understudy — slot 0 poisoned, slot 1 clean
    // at base = slab, ALL rows on slot 1. ─────────────────────────────────
    let slab_k = SLAB_TOKENS * k_bytes_per_pos(tier);
    let slab_v = SLAB_TOKENS * v_bytes_per_pos();
    let mut k_two = k_slab(tier, rng, true);
    k_two.extend(k_slab(tier, rng, false));
    let mut v_two = v_slab(rng, true);
    v_two.extend(v_slab(rng, false));
    let k_two_dev = upload_raw(gpu, &k_two);
    let v_two_dev = upload_raw(gpu, &v_two);
    let descs_slab = slab_descs(gpu, [0, slab_k as u64], [0, slab_v as u64]);
    let rs_one = row_slot_all(gpu, 1);
    let out_two = out_dev(gpu);
    let out_ref2 = out_dev(gpu);
    let understudy = {
        match tier {
            "asym2" => gpu.attention_flash_asym2_batched_slots(
                &q,
                &k_two_dev,
                &v_two_dev,
                &out_two,
                &pos,
                &t.cos,
                &t.sin,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                Some(&descs_slab),
                Some(&rs_one),
            ),
            "asym3" => gpu.attention_flash_asym3_batched_masked_slots(
                &q,
                &k_two_dev,
                &v_two_dev,
                &out_two,
                &pos,
                &t.cos,
                &t.sin,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
                Some(&descs_slab),
                Some(&rs_one),
            ),
            "asym4" => gpu.attention_flash_asym4_batched_masked_slots(
                &q,
                &k_two_dev,
                &v_two_dev,
                &out_two,
                &pos,
                &t.cos,
                &t.sin,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
                Some(&descs_slab),
                Some(&rs_one),
            ),
            "fwht2" => gpu.attention_flash_fwht2_batched_slots(
                &q,
                &k_two_dev,
                &v_two_dev,
                &out_two,
                &pos,
                &t.signs1,
                &t.signs2,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                Some(&descs_slab),
                Some(&rs_one),
            ),
            "fwht3" => gpu.attention_flash_fwht3_batched_masked_slots(
                &q,
                &k_two_dev,
                &v_two_dev,
                &out_two,
                &pos,
                &t.signs1,
                &t.signs2,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
                8,
                Some(&descs_slab),
                Some(&rs_one),
            ),
            "fwht4" => gpu.attention_flash_fwht4_batched_masked_slots(
                &q,
                &k_two_dev,
                &v_two_dev,
                &out_two,
                &pos,
                &t.signs1,
                &t.signs2,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
                Some(&descs_slab),
                Some(&rs_one),
            ),
            other => panic!("unknown tier {other}"),
        }
        .expect("understudy slots launch");
        // Reference: the SAME clean slab data, at base 0, via the plain path.
        let k_ref2 = upload_raw(gpu, &k_two[slab_k..]);
        let v_ref2 = upload_raw(gpu, &v_two[slab_v..]);
        match tier {
            "asym2" => gpu.attention_flash_asym2_batched(
                &q,
                &k_ref2,
                &v_ref2,
                &out_ref2,
                &pos,
                &t.cos,
                &t.sin,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
            ),
            "asym3" => gpu.attention_flash_asym3_batched_masked(
                &q,
                &k_ref2,
                &v_ref2,
                &out_ref2,
                &pos,
                &t.cos,
                &t.sin,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
            ),
            "asym4" => gpu.attention_flash_asym4_batched_masked(
                &q,
                &k_ref2,
                &v_ref2,
                &out_ref2,
                &pos,
                &t.cos,
                &t.sin,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
            ),
            "fwht2" => gpu.attention_flash_fwht2_batched(
                &q,
                &k_ref2,
                &v_ref2,
                &out_ref2,
                &pos,
                &t.signs1,
                &t.signs2,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                8,
            ),
            "fwht3" => gpu.attention_flash_fwht3_batched_masked(
                &q,
                &k_ref2,
                &v_ref2,
                &out_ref2,
                &pos,
                &t.signs1,
                &t.signs2,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
                8,
            ),
            "fwht4" => gpu.attention_flash_fwht4_batched_masked(
                &q,
                &k_ref2,
                &v_ref2,
                &out_ref2,
                &pos,
                &t.signs1,
                &t.signs2,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SLAB_TOKENS,
                MAX_CTX,
                BATCH,
                &partials,
                None,
                0,
                0,
                8,
            ),
            other => panic!("unknown tier {other}"),
        }
        .expect("understudy reference launch");
        gpu.hip.device_synchronize().expect("sync");
        max_abs_diff(gpu, &out_two, &out_ref2)
    };
    // Example-exit hygiene only: the process tears the GPU context down
    // right after, so per-tensor frees are best-effort.
    let _ = (
        pos, partials, q, k_clean, v_clean, descs_zero, rs_alt, out_slots, out_ref, k_two_dev,
        v_two_dev, descs_slab, rs_one, out_two, out_ref2,
    );
    (parity, understudy)
}

/// Check 3 for one tier's composite K/V writer. Returns the number of
/// mismatched bytes between the descriptor write's slot-1 slab and the
/// legacy write's slab (expect 0), and whether slot 0 stayed untouched.
fn write_check(gpu: &mut Gpu, tier: &str, t: &Tables, rng: &mut Rng) -> (usize, bool) {
    let k_src = upload_f32(gpu, &rand_f32_vec(BATCH * N_KV_HEADS * HEAD_DIM, rng));
    let v_src = upload_f32(gpu, &rand_f32_vec(BATCH * N_KV_HEADS * HEAD_DIM, rng));
    let pos = positions_dev(gpu);
    let rs_one = row_slot_all(gpu, 1);

    let slab_k = SLAB_TOKENS * k_bytes_per_pos(tier);
    let slab_v = SLAB_TOKENS * v_bytes_per_pos();

    // Candidate arenas: [poison slab][zero slab]; rows write through slot 1.
    let k_poison_snapshot = k_slab(tier, rng, true);
    let mut k_cand = k_poison_snapshot.clone();
    k_cand.extend(std::iter::repeat(0u8).take(slab_k));
    let v_poison_snapshot = v_slab(rng, true);
    let mut v_cand = v_poison_snapshot.clone();
    v_cand.extend(std::iter::repeat(0u8).take(slab_v));
    let k_cand_dev = upload_raw(gpu, &k_cand);
    let v_cand_dev = upload_raw(gpu, &v_cand);

    // Reference arenas: one zeroed slab.
    let k_ref = upload_raw(gpu, &vec![0u8; slab_k]);
    let v_ref = upload_raw(gpu, &vec![0u8; slab_v]);

    let descs = slab_descs(gpu, [0, slab_k as u64], [0, slab_v as u64]);
    match tier {
        "asym2" => gpu.kv_cache_write_asym2_batched_slots(
            &k_cand_dev,
            &v_cand_dev,
            &k_src,
            &v_src,
            &pos,
            &t.cos,
            &t.sin,
            N_KV_HEADS,
            HEAD_DIM,
            BATCH,
            Some(&descs),
            Some(&rs_one),
        ),
        "asym3" => gpu.kv_cache_write_asym3_batched_slots(
            &k_cand_dev,
            &v_cand_dev,
            &k_src,
            &v_src,
            &pos,
            &t.cos,
            &t.sin,
            N_KV_HEADS,
            HEAD_DIM,
            BATCH,
            Some(&descs),
            Some(&rs_one),
        ),
        "asym4" => gpu.kv_cache_write_asym4_batched_slots(
            &k_cand_dev,
            &v_cand_dev,
            &k_src,
            &v_src,
            &pos,
            &t.cos,
            &t.sin,
            N_KV_HEADS,
            HEAD_DIM,
            BATCH,
            Some(&descs),
            Some(&rs_one),
        ),
        "fwht2" => gpu.kv_cache_write_fwht2_batched_slots(
            &k_cand_dev,
            &v_cand_dev,
            &k_src,
            &v_src,
            &pos,
            &t.signs1,
            &t.signs2,
            N_KV_HEADS,
            HEAD_DIM,
            BATCH,
            Some(&descs),
            Some(&rs_one),
        ),
        "fwht3" => gpu.kv_cache_write_fwht3_batched_slots(
            &k_cand_dev,
            &v_cand_dev,
            &k_src,
            &v_src,
            &pos,
            &t.signs1,
            &t.signs2,
            N_KV_HEADS,
            HEAD_DIM,
            BATCH,
            Some(&descs),
            Some(&rs_one),
        ),
        "fwht4" => gpu.kv_cache_write_fwht4_batched_slots(
            &k_cand_dev,
            &v_cand_dev,
            &k_src,
            &v_src,
            &pos,
            &t.signs1,
            &t.signs2,
            N_KV_HEADS,
            HEAD_DIM,
            BATCH,
            Some(&descs),
            Some(&rs_one),
        ),
        other => panic!("unknown tier {other}"),
    }
    .expect("candidate composite write");
    match tier {
        "asym2" => gpu.kv_cache_write_asym2_batched(
            &k_ref, &v_ref, &k_src, &v_src, &pos, &t.cos, &t.sin, N_KV_HEADS, HEAD_DIM, BATCH,
        ),
        "asym3" => gpu.kv_cache_write_asym3_batched(
            &k_ref, &v_ref, &k_src, &v_src, &pos, &t.cos, &t.sin, N_KV_HEADS, HEAD_DIM, BATCH,
        ),
        "asym4" => gpu.kv_cache_write_asym4_batched(
            &k_ref, &v_ref, &k_src, &v_src, &pos, &t.cos, &t.sin, N_KV_HEADS, HEAD_DIM, BATCH,
        ),
        "fwht2" => gpu.kv_cache_write_fwht2_batched(
            &k_ref, &v_ref, &k_src, &v_src, &pos, &t.signs1, &t.signs2, N_KV_HEADS, HEAD_DIM,
            BATCH, 8,
        ),
        "fwht3" => gpu.kv_cache_write_fwht3_batched(
            &k_ref, &v_ref, &k_src, &v_src, &pos, &t.signs1, &t.signs2, N_KV_HEADS, HEAD_DIM,
            BATCH, 8,
        ),
        "fwht4" => gpu.kv_cache_write_fwht4_batched(
            &k_ref, &v_ref, &k_src, &v_src, &pos, &t.signs1, &t.signs2, N_KV_HEADS, HEAD_DIM,
            BATCH, 8,
        ),
        other => panic!("unknown tier {other}"),
    }
    .expect("reference composite write");
    gpu.hip.device_synchronize().expect("sync");

    let kc = download_raw(gpu, &k_cand_dev);
    let vc = download_raw(gpu, &v_cand_dev);
    let kr = download_raw(gpu, &k_ref);
    let vr = download_raw(gpu, &v_ref);
    let mismatches = kc[slab_k..]
        .iter()
        .zip(kr.iter())
        .chain(vc[slab_v..].iter().zip(vr.iter()))
        .filter(|(a, b)| a != b)
        .count();
    let poison_intact =
        kc[..slab_k] == k_poison_snapshot[..] && vc[..slab_v] == v_poison_snapshot[..];
    let _ = (
        k_src, v_src, pos, rs_one, k_cand_dev, v_cand_dev, k_ref, v_ref,
    );
    (mismatches, poison_intact)
}

/// Check 4: the Q8 batched writer's `use_v_base` arm.
fn q8_v_base_check(gpu: &mut Gpu, rng: &mut Rng) {
    let src = upload_f32(gpu, &rand_f32_vec(BATCH * N_KV_HEADS * HEAD_DIM, rng));
    let pos = positions_dev(gpu);
    let rs_one = row_slot_all(gpu, 1);
    let slab = SLAB_TOKENS * v_bytes_per_pos();
    let descs = slab_descs(gpu, [0; SLOTS], [0, slab as u64]);

    // Candidate: V base = slab, use_v_base = true → bytes must land in
    // slot 1's slab. Two slabs: [poison][zero].
    let cand_poison = v_slab(rng, true);
    let mut cand = cand_poison.clone();
    cand.extend(std::iter::repeat(0u8).take(slab));
    let cand_dev = upload_raw(gpu, &cand);
    gpu.kv_cache_write_q8_0_batched_slots(
        &cand_dev,
        &src,
        &pos,
        N_KV_HEADS,
        HEAD_DIM,
        BATCH,
        Some(&descs),
        Some(&rs_one),
        true,
    )
    .expect("candidate v-base write");

    // Reference: zero-base write into a fresh slab.
    let ref_dev = upload_raw(gpu, &vec![0u8; slab]);
    gpu.kv_cache_write_q8_0_batched_slots(
        &ref_dev, &src, &pos, N_KV_HEADS, HEAD_DIM, BATCH, None, None, false,
    )
    .expect("reference q8 write");

    // Negative control: use_v_base = FALSE with v_base = slab must land at
    // the K base (0) — proving the flag actually flips addressing. Two
    // slabs again, so the slot-1 half exists even though the write goes to
    // slot 0.
    let neg_poison = v_slab(rng, true);
    let mut neg = neg_poison.clone();
    neg.extend(std::iter::repeat(0u8).take(slab));
    let neg_dev = upload_raw(gpu, &neg);
    gpu.kv_cache_write_q8_0_batched_slots(
        &neg_dev,
        &src,
        &pos,
        N_KV_HEADS,
        HEAD_DIM,
        BATCH,
        Some(&descs),
        Some(&rs_one),
        false,
    )
    .expect("negative-control write");
    gpu.hip.device_synchronize().expect("sync");

    let c = download_raw(gpu, &cand_dev);
    let r = download_raw(gpu, &ref_dev);
    let n = download_raw(gpu, &neg_dev);
    assert!(
        c[slab..] == r[..],
        "use_v_base=true did not reproduce the zero-base write at the V base"
    );
    assert!(
        c[..slab] == cand_poison[..],
        "use_v_base=true disturbed slot 0's slab — wrote through the K base"
    );
    assert!(
        n[..slab] != neg_poison[..],
        "use_v_base=false negative control did not write at the K base — \
         the flag is not flipping addressing"
    );
    assert!(
        n[slab..].iter().all(|&b| b == 0),
        "use_v_base=false negative control wrote into slot 1's slab"
    );
    let _ = (src, pos, rs_one, cand_dev, ref_dev, neg_dev);
}

// ─────────────────────────────────────────────────────────────────────────────
// Paged (block-table) arm
//
// The checks above all run LEGACY (page_tokens = 0) descriptors, so nothing
// here proved that the rotated tiers' paged indirection works: the batched K
// writers and the tile-batched attention kernels resolve their destination
// through `kv_offset_for_k/v(desc, pos, stride)`, which switches to a block
// table lookup the moment `page_tokens != 0`. This arm drives that path for
// real: scattered, non-monotonic physical page tables, a write through the
// paged descriptors, a gather of the referenced pages into an equivalent
// contiguous arena, and a comparison of the paged attention output against
// the legacy output over those identical logical bytes. It also asserts that
// pages NO table references stay byte-for-byte untouched.
// ─────────────────────────────────────────────────────────────────────────────

/// Tokens per physical page for the paged arm (the production page size).
const PAGE_TOKENS: usize = 128;
/// Physical pages per arena; slots reference 4 of them, 2 stay bystanders.
const PAGED_PAGES: usize = 6;
/// Non-monotonic page tables: logical page 0 maps AWAY from physical 0, so a
/// kernel that ignored the table (pos * stride) lands in the wrong bytes.
const SLOT_TABLES: [[i32; 2]; SLOTS] = [[4, 1], [3, 0]];

fn paged_descs(gpu: &Gpu, tables: &[GpuTensor; SLOTS]) -> GpuTensor {
    let descs: Vec<KvSlotDesc> = (0..SLOTS)
        .map(|s| KvSlotDesc {
            block_table: tables[s].buf.as_ptr() as u64,
            legacy_k_base: 0,
            legacy_v_base: 0,
            seq_len: (2 * PAGE_TOKENS) as i32,
            page_tokens: PAGE_TOKENS as i32,
        })
        .collect();
    let mut bytes = pack_descs(&descs);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    upload_raw(gpu, &bytes)
}

/// Gather the physical pages a table references into logical page order.
fn gather_logical(raw: &[u8], table: &[i32; 2], page_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 * page_bytes);
    for &phys in table.iter() {
        let start = phys as usize * page_bytes;
        out.extend_from_slice(&raw[start..start + page_bytes]);
    }
    out
}

/// One page's worth of poison for a tier (NaN norm header / NaN Q8 scale), so
/// any stray write into an unreferenced page is observable.
fn poison_page_k(tier: &str, rng: &mut Rng) -> Vec<u8> {
    let page_bytes = PAGE_TOKENS * k_bytes_per_pos(tier);
    let slab = k_slab(tier, rng, true);
    let mut out = Vec::with_capacity(page_bytes);
    for t in 0..PAGE_TOKENS {
        let row = &slab[t * k_bytes_per_pos(tier)..(t + 1) * k_bytes_per_pos(tier)];
        out.extend_from_slice(row);
    }
    out
}

fn poison_page_v(rng: &mut Rng) -> Vec<u8> {
    let page_bytes = PAGE_TOKENS * v_bytes_per_pos();
    let slab = v_slab(rng, true);
    let mut out = Vec::with_capacity(page_bytes);
    for t in 0..PAGE_TOKENS {
        let row = &slab[t * v_bytes_per_pos()..(t + 1) * v_bytes_per_pos()];
        out.extend_from_slice(row);
    }
    out
}

/// Write the step's rows through the tier's composite slots writer with a
/// PAGED descriptor (block table, page_tokens > 0).
#[allow(clippy::too_many_arguments)]
fn paged_write(
    gpu: &mut Gpu,
    tier: &str,
    t: &Tables,
    k_dst: &GpuTensor,
    v_dst: &GpuTensor,
    k_src: &GpuTensor,
    v_src: &GpuTensor,
    pos: &GpuTensor,
    row_slot: &GpuTensor,
    descs: &GpuTensor,
) {
    match tier {
        "asym2" => gpu.kv_cache_write_asym2_batched_slots(
            k_dst,
            v_dst,
            k_src,
            v_src,
            pos,
            &t.cos,
            &t.sin,
            N_KV_HEADS,
            HEAD_DIM,
            BATCH,
            Some(descs),
            Some(row_slot),
        ),
        "asym3" => gpu.kv_cache_write_asym3_batched_slots(
            k_dst,
            v_dst,
            k_src,
            v_src,
            pos,
            &t.cos,
            &t.sin,
            N_KV_HEADS,
            HEAD_DIM,
            BATCH,
            Some(descs),
            Some(row_slot),
        ),
        "asym4" => gpu.kv_cache_write_asym4_batched_slots(
            k_dst,
            v_dst,
            k_src,
            v_src,
            pos,
            &t.cos,
            &t.sin,
            N_KV_HEADS,
            HEAD_DIM,
            BATCH,
            Some(descs),
            Some(row_slot),
        ),
        "fwht2" => gpu.kv_cache_write_fwht2_batched_slots(
            k_dst,
            v_dst,
            k_src,
            v_src,
            pos,
            &t.signs1,
            &t.signs2,
            N_KV_HEADS,
            HEAD_DIM,
            BATCH,
            Some(descs),
            Some(row_slot),
        ),
        "fwht3" => gpu.kv_cache_write_fwht3_batched_slots(
            k_dst,
            v_dst,
            k_src,
            v_src,
            pos,
            &t.signs1,
            &t.signs2,
            N_KV_HEADS,
            HEAD_DIM,
            BATCH,
            Some(descs),
            Some(row_slot),
        ),
        "fwht4" => gpu.kv_cache_write_fwht4_batched_slots(
            k_dst,
            v_dst,
            k_src,
            v_src,
            pos,
            &t.signs1,
            &t.signs2,
            N_KV_HEADS,
            HEAD_DIM,
            BATCH,
            Some(descs),
            Some(row_slot),
        ),
        other => panic!("unknown tier {other}"),
    }
    .expect("paged composite write");
    gpu.hip.device_synchronize().expect("sync");
}

/// Attention over the given (paged or legacy) descriptors for one tier.
#[allow(clippy::too_many_arguments)]
fn attend(
    gpu: &mut Gpu,
    tier: &str,
    t: &Tables,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    out: &GpuTensor,
    pos: &GpuTensor,
    partials: &GpuTensor,
    seq: usize,
    descs: &GpuTensor,
    row_slot: &GpuTensor,
) {
    match tier {
        "asym2" => gpu.attention_flash_asym2_batched_slots(
            q,
            k,
            v,
            out,
            pos,
            &t.cos,
            &t.sin,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            seq,
            seq,
            BATCH,
            partials,
            Some(descs),
            Some(row_slot),
        ),
        "asym3" => gpu.attention_flash_asym3_batched_masked_slots(
            q,
            k,
            v,
            out,
            pos,
            &t.cos,
            &t.sin,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            seq,
            seq,
            BATCH,
            partials,
            None,
            0,
            0,
            Some(descs),
            Some(row_slot),
        ),
        "asym4" => gpu.attention_flash_asym4_batched_masked_slots(
            q,
            k,
            v,
            out,
            pos,
            &t.cos,
            &t.sin,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            seq,
            seq,
            BATCH,
            partials,
            None,
            0,
            0,
            Some(descs),
            Some(row_slot),
        ),
        "fwht2" => gpu.attention_flash_fwht2_batched_slots(
            q,
            k,
            v,
            out,
            pos,
            &t.signs1,
            &t.signs2,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            seq,
            seq,
            BATCH,
            partials,
            Some(descs),
            Some(row_slot),
        ),
        "fwht3" => gpu.attention_flash_fwht3_batched_masked_slots(
            q,
            k,
            v,
            out,
            pos,
            &t.signs1,
            &t.signs2,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            seq,
            seq,
            BATCH,
            partials,
            None,
            0,
            0,
            8,
            Some(descs),
            Some(row_slot),
        ),
        "fwht4" => gpu.attention_flash_fwht4_batched_masked_slots(
            q,
            k,
            v,
            out,
            pos,
            &t.signs1,
            &t.signs2,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            seq,
            seq,
            BATCH,
            partials,
            None,
            0,
            0,
            Some(descs),
            Some(row_slot),
        ),
        other => panic!("unknown tier {other}"),
    }
    .expect("paged attention");
    gpu.hip.device_synchronize().expect("sync");
}

/// Returns (attention parity, stray-write free, gather parity).
fn paged_rotated_check(gpu: &mut Gpu, tier: &str, t: &Tables, rng: &mut Rng) -> (f32, bool, bool) {
    let page_k = PAGE_TOKENS * k_bytes_per_pos(tier);
    let page_v = PAGE_TOKENS * v_bytes_per_pos();

    // Poison every physical page; the write must only touch pages a table
    // references, and only at the offsets the table maps.
    let k_page = poison_page_k(tier, rng);
    let v_page = poison_page_v(rng);
    let mut k_arena: Vec<u8> = Vec::with_capacity(PAGED_PAGES * page_k);
    let mut v_arena: Vec<u8> = Vec::with_capacity(PAGED_PAGES * page_v);
    for _ in 0..PAGED_PAGES {
        k_arena.extend_from_slice(&k_page);
        v_arena.extend_from_slice(&v_page);
    }
    let k_poison_snapshot = k_arena.clone();
    let v_poison_snapshot = v_arena.clone();

    let k_paged = upload_raw(gpu, &k_arena);
    let v_paged = upload_raw(gpu, &v_arena);

    // One device block table per slot.
    let tables: [GpuTensor; SLOTS] =
        std::array::from_fn(|s| upload_raw(gpu, &i32_bytes(&SLOT_TABLES[s])));
    let descs_paged = paged_descs(gpu, &tables);
    // Rows alternate slots; positions span both logical pages.
    let row_slot = upload_raw(gpu, &i32_bytes(&[0, 0, 1, 1]));
    let pos = positions_dev(gpu);

    let k_src = upload_f32(gpu, &rand_f32_vec(BATCH * N_KV_HEADS * HEAD_DIM, rng));
    let v_src = upload_f32(gpu, &rand_f32_vec(BATCH * N_KV_HEADS * HEAD_DIM, rng));

    paged_write(
        gpu,
        tier,
        t,
        &k_paged,
        &v_paged,
        &k_src,
        &v_src,
        &pos,
        &row_slot,
        &descs_paged,
    );

    let k_after = download_raw(gpu, &k_paged);
    let v_after = download_raw(gpu, &v_paged);

    // Bystander pages (2, 5) are referenced by no table: any change is a
    // stray write through unmapped addressing.
    let mut stray = false;
    for bystander in [2usize, 5] {
        let ks = bystander * page_k;
        let vs = bystander * page_v;
        if k_after[ks..ks + page_k] != k_poison_snapshot[ks..ks + page_k]
            || v_after[vs..vs + page_v] != v_poison_snapshot[vs..vs + page_v]
        {
            stray = true;
        }
    }

    // Gather the referenced pages into logical order: the same logical bytes
    // a legacy contiguous arena would hold.
    let mut legacy_k: Vec<u8> = Vec::with_capacity(SLOTS * 2 * page_k);
    let mut legacy_v: Vec<u8> = Vec::with_capacity(SLOTS * 2 * page_v);
    for s in 0..SLOTS {
        legacy_k.extend_from_slice(&gather_logical(&k_after, &SLOT_TABLES[s], page_k));
        legacy_v.extend_from_slice(&gather_logical(&v_after, &SLOT_TABLES[s], page_v));
    }
    let k_legacy = upload_raw(gpu, &legacy_k);
    let v_legacy = upload_raw(gpu, &legacy_v);
    let descs_legacy = slab_descs(gpu, [0, (2 * page_k) as u64], [0, (2 * page_v) as u64]);

    // Parity: paged read over the scattered arena vs legacy read over the
    // gathered copy of exactly those bytes.
    let q = upload_f32(gpu, &rand_f32_vec(BATCH * N_HEADS * HEAD_DIM, rng));
    let partials = partials_dev(gpu);
    let out_paged = out_dev(gpu);
    let out_legacy = out_dev(gpu);
    let seq = 2 * PAGE_TOKENS;
    attend(
        gpu,
        tier,
        t,
        &q,
        &k_paged,
        &v_paged,
        &out_paged,
        &pos,
        &partials,
        seq,
        &descs_paged,
        &row_slot,
    );
    attend(
        gpu,
        tier,
        t,
        &q,
        &k_legacy,
        &v_legacy,
        &out_legacy,
        &pos,
        &partials,
        seq,
        &descs_legacy,
        &row_slot,
    );
    let parity = max_abs_diff(gpu, &out_paged, &out_legacy);

    // Gather parity: the logical bytes really are where the table says. A
    // writer that ignored the table would have put the rows at their logical
    // offsets instead, so re-deriving each written row through the table's
    // own mapping is the check that the indirection was honored.
    let mut gather_ok = true;
    let slot_of_row = [0usize, 0, 1, 1];
    let row_bytes = k_bytes_per_pos(tier);
    for (row, &p_raw) in POSITIONS.iter().enumerate() {
        let p = p_raw as usize;
        let s = slot_of_row[row];
        let gathered = gather_logical(&k_after, &SLOT_TABLES[s], page_k);
        let off = p * row_bytes;
        let written = &gathered[off..off + row_bytes];
        let poison_row = &k_page[(p % PAGE_TOKENS) * row_bytes..(p % PAGE_TOKENS + 1) * row_bytes];
        if written == poison_row {
            gather_ok = false;
        }
    }

    (parity, !stray, gather_ok)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut rng = Rng(0xC0FFEE);
    let mut failures = 0usize;

    println!(
        "== descriptor-port GPU validation (hd={HEAD_DIM}, nkv={N_KV_HEADS}, batch={BATCH}) =="
    );
    for tier in TIERS {
        let t = upload_tables(&mut gpu, &mut rng);
        let (parity, understudy) = attn_checks(&mut gpu, tier, &t, &mut rng);
        let (mm, intact) = write_check(&mut gpu, tier, &t, &mut rng);
        let (paged_parity, no_stray, gather_ok) = paged_rotated_check(&mut gpu, tier, &t, &mut rng);
        let ok = parity <= TOL
            && understudy <= TOL
            && mm == 0
            && intact
            && paged_parity <= TOL
            && no_stray
            && gather_ok;
        if !ok {
            failures += 1;
        }
        println!(
            "  {tier:6} attn parity {parity:.3e} | understudy {understudy:.3e} | \
             write mismatches {mm} | slot0 intact {intact} | \
             paged parity {paged_parity:.3e} | no stray {no_stray} | gather {gather_ok}  {}",
            if ok { "OK" } else { "FAIL" }
        );
        drop(t);
    }
    q8_v_base_check(&mut gpu, &mut rng);
    println!("  q8     use_v_base arm OK (V write lands at legacy_v_base; flag flips addressing)");

    if failures > 0 {
        eprintln!("FAILED: {failures} tier(s) failed");
        std::process::exit(1);
    }
    println!("ALL TIERS PASS");
}
