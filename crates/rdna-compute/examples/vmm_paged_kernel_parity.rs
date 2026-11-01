// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 1 oracle for the `_vmm_paged` kernel variants.
//!
//! Both routes use the same logical block table and the same token bytes, but
//! different physical placement:
//! - reference: arena-paged bytes at `phys_page * page_bytes`
//! - candidate: VMM-paged bytes at
//!   `(phys_page / P) * granule_bytes + (phys_page % P) * page_bytes`
//!
//! The candidate is compared against the arena-paged twin for:
//! - Q8 KV writes,
//! - Q8 LDS decode attention,
//! - Q8 tile attention.
//!
//! The block table, positions, row_slot, and inputs are shared, so a mismatch
//! is a descriptor/address-translation bug, not a different test.
//!
//! Run:
//!   HIPFIRE_VMM_SMOKE_DEVICE=2 cargo run -p rdna-compute --example vmm_paged_kernel_parity

use rdna_compute::kv_slots::{KvSlotDesc, VmmPagedKvSlotDesc};
use rdna_compute::{DType, Gpu, GpuTensor};

const PAGE_TOKENS: usize = 4;
const PAGES_PER_GRANULE: u32 = 2;
const GRANULE_BYTES: usize = 1024;
const N_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 32;
const N_HEADS: usize = 1;
const STRIDE: usize = N_KV_HEADS * (HEAD_DIM / 32) * 34;
const PAGE_BYTES: usize = PAGE_TOKENS * STRIDE;
const N_PAGES: usize = 10;
const PAGES_PER_SLOT: usize = 5;
const N_GRANULES: usize = (N_PAGES + PAGES_PER_GRANULE as usize - 1) / PAGES_PER_GRANULE as usize;
const VMM_ARENA_BYTES: usize = N_GRANULES * GRANULE_BYTES;
const REF_ARENA_BYTES: usize = N_PAGES * PAGE_BYTES;
const SEQ_LEN: usize = 18;

fn dev() -> i32 {
    std::env::var("HIPFIRE_VMM_SMOKE_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(0)
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn page_offset(phys: u32) -> usize {
    let p = PAGES_PER_GRANULE as usize;
    (phys as usize / p) * GRANULE_BYTES + (phys as usize % p) * PAGE_BYTES
}

fn q8_row(salt: u8, position: usize) -> Vec<u8> {
    let mut row = vec![0u8; STRIDE];
    // f16 1.0 scale, then 32 signed quants. Keep every value finite; the
    // comparison is exact, not tolerance-based.
    row[0] = 0x00;
    row[1] = 0x3c;
    for (i, byte) in row[2..].iter_mut().enumerate() {
        *byte = (salt as usize)
            .wrapping_mul(37)
            .wrapping_add(position.wrapping_mul(19))
            .wrapping_add(i.wrapping_mul(7)) as u8;
    }
    row
}

fn page_data(salt: u8, logical_page: usize, valid_positions: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(PAGE_BYTES);
    for t in 0..PAGE_TOKENS {
        let pos = logical_page * PAGE_TOKENS + t;
        if pos < valid_positions {
            bytes.extend_from_slice(&q8_row(salt, pos));
        } else {
            // Poison the never-read tail; a kernel that over-reads must fail
            // the exact comparison rather than silently see zeros.
            bytes.extend(std::iter::repeat_n(0x7f, STRIDE));
        }
    }
    bytes
}

/// Two slots, two disjoint physical page sets. Logical page `lp` of slot 0
/// lives at phys `lp`; logical page `lp` of slot 1 lives at phys
/// `PAGES_PER_SLOT + lp`. The salts differ so a kernel that pins slot 1 to
/// slot 0's table fails the exact comparison.
fn build_two_slot_arena(salt0: u8, salt1: u8, vmm: bool, valid_positions: usize) -> Vec<u8> {
    let mut bytes = vec![
        0u8;
        if vmm {
            VMM_ARENA_BYTES
        } else {
            REF_ARENA_BYTES
        }
    ];
    for lp in 0..PAGES_PER_SLOT {
        let physical = lp;
        let offset = if vmm {
            page_offset(physical as u32)
        } else {
            physical * PAGE_BYTES
        };
        bytes[offset..offset + PAGE_BYTES].copy_from_slice(&page_data(salt0, lp, valid_positions));
    }
    for lp in 0..PAGES_PER_SLOT {
        let physical = PAGES_PER_SLOT + lp;
        let offset = if vmm {
            page_offset(physical as u32)
        } else {
            physical * PAGE_BYTES
        };
        bytes[offset..offset + PAGE_BYTES].copy_from_slice(&page_data(salt1, lp, valid_positions));
    }
    bytes
}

fn upload_desc_tables(
    gpu: &Gpu,
    block_tables: &[&GpuTensor],
    vmm: bool,
    seq_lens: &[usize],
) -> GpuTensor {
    assert_eq!(block_tables.len(), seq_lens.len());
    if vmm {
        let descs: Vec<VmmPagedKvSlotDesc> = block_tables
            .iter()
            .zip(seq_lens)
            .map(|(table, &seq_len)| VmmPagedKvSlotDesc {
                block_table: table.buf.as_ptr() as u64,
                legacy_k_base: 0,
                legacy_v_base: 0,
                seq_len: seq_len as i32,
                page_tokens: PAGE_TOKENS as i32,
                granule_bytes: GRANULE_BYTES as u64,
                pages_per_granule: PAGES_PER_GRANULE,
                _pad: 0,
            })
            .collect();
        for d in &descs {
            d.check_paged_layout().unwrap();
        }
        gpu.upload_raw(VmmPagedKvSlotDesc::as_bytes(&descs), &[descs.len()])
            .expect("upload VMM descs")
    } else {
        let mut raw = Vec::with_capacity(seq_lens.len() * 32);
        for (table, &seq_len) in block_tables.iter().zip(seq_lens) {
            let d = KvSlotDesc {
                block_table: table.buf.as_ptr() as u64,
                legacy_k_base: 0,
                legacy_v_base: 0,
                seq_len: seq_len as i32,
                page_tokens: PAGE_TOKENS as i32,
            };
            raw.extend_from_slice(&d.block_table.to_ne_bytes());
            raw.extend_from_slice(&d.legacy_k_base.to_ne_bytes());
            raw.extend_from_slice(&d.legacy_v_base.to_ne_bytes());
            raw.extend_from_slice(&d.seq_len.to_ne_bytes());
            raw.extend_from_slice(&d.page_tokens.to_ne_bytes());
        }
        gpu.upload_raw(&raw, &[seq_lens.len()])
            .expect("upload arena descs")
    }
}

fn q8_write_parity(gpu: &mut Gpu) {
    let table0: Vec<i32> = (0..PAGES_PER_SLOT as i32).collect();
    let table1: Vec<i32> = (PAGES_PER_SLOT as i32..N_PAGES as i32).collect();
    let block_table0 = gpu
        .upload_raw(&i32_bytes(&table0), &[PAGES_PER_SLOT])
        .expect("write block table 0");
    let block_table1 = gpu
        .upload_raw(&i32_bytes(&table1), &[PAGES_PER_SLOT])
        .expect("write block table 1");
    let ref_descs = upload_desc_tables(
        gpu,
        &[&block_table0, &block_table1],
        false,
        &[SEQ_LEN, SEQ_LEN],
    );
    let vmm_descs = upload_desc_tables(
        gpu,
        &[&block_table0, &block_table1],
        true,
        &[SEQ_LEN, SEQ_LEN],
    );
    let ref_arena = gpu
        .zeros(&[REF_ARENA_BYTES], DType::Raw)
        .expect("ref arena");
    let vmm_arena = gpu
        .zeros(&[VMM_ARENA_BYTES], DType::Raw)
        .expect("vmm arena");
    let src: Vec<f32> = (0..2 * HEAD_DIM)
        .map(|i| ((i as f32) * 0.0625) - 2.0)
        .collect();
    let src = gpu.upload_f32(&src, &[2 * HEAD_DIM]).expect("src");
    let pos_values = [5i32, 6];
    let positions = gpu
        .upload_raw(&i32_bytes(&pos_values), &[2])
        .expect("positions");
    let row_slot = gpu.upload_raw(&i32_bytes(&[0, 1]), &[2]).expect("row_slot");

    gpu.kv_cache_write_q8_0_batched_slots(
        &ref_arena,
        &src,
        &positions,
        N_KV_HEADS,
        HEAD_DIM,
        2,
        Some(&ref_descs),
        Some(&row_slot),
        false,
    )
    .expect("reference write");
    gpu.kv_cache_write_q8_0_batched_slots_vmm_paged(
        &vmm_arena,
        &src,
        &positions,
        N_KV_HEADS,
        HEAD_DIM,
        2,
        Some(&vmm_descs),
        Some(&row_slot),
        false,
    )
    .expect("vmm write");
    gpu.hip.device_synchronize().expect("sync");
    let ref_bytes = gpu.download_raw_bytes(&ref_arena).expect("read ref");
    let vmm_bytes = gpu.download_raw_bytes(&vmm_arena).expect("read vmm");
    for (row, pos) in pos_values.iter().enumerate() {
        let lp = *pos as usize / PAGE_TOKENS;
        let t = *pos as usize % PAGE_TOKENS;
        let phys = if row == 0 { lp } else { PAGES_PER_SLOT + lp };
        let ref_start = phys * PAGE_BYTES + t * STRIDE;
        let vmm_start = page_offset(phys as u32) + t * STRIDE;
        assert_eq!(
            &ref_bytes[ref_start..ref_start + STRIDE],
            &vmm_bytes[vmm_start..vmm_start + STRIDE],
            "Q8 write mismatch at row {row} position {pos}"
        );
    }
    // The candidate's poison/tail pages must be untouched, not written at a
    // wrong physical offset. Each active slot writes the same logical page.
    let written_page = 5 / PAGE_TOKENS;
    for phys in 0..N_PAGES {
        if phys != written_page && phys != PAGES_PER_SLOT + written_page {
            let off = page_offset(phys as u32);
            assert!(
                vmm_bytes[off..off + PAGE_BYTES].iter().all(|&b| b == 0),
                "VMM write touched unaddressed physical page {phys}"
            );
        }
    }
    gpu.free_tensor(block_table0).unwrap();
    gpu.free_tensor(block_table1).unwrap();
    gpu.free_tensor(ref_descs).unwrap();
    gpu.free_tensor(vmm_descs).unwrap();
    gpu.free_tensor(ref_arena).unwrap();
    gpu.free_tensor(vmm_arena).unwrap();
    gpu.free_tensor(src).unwrap();
    gpu.free_tensor(positions).unwrap();
    gpu.free_tensor(row_slot).unwrap();
    println!("vmm_paged_kernel_parity: Q8_WRITE PASS");
}

fn attention_parity(gpu: &mut Gpu) {
    let table0: Vec<i32> = (0..PAGES_PER_SLOT as i32).collect();
    let table1: Vec<i32> = (PAGES_PER_SLOT as i32..N_PAGES as i32).collect();
    let block_table0 = gpu
        .upload_raw(&i32_bytes(&table0), &[PAGES_PER_SLOT])
        .expect("attn block table 0");
    let block_table1 = gpu
        .upload_raw(&i32_bytes(&table1), &[PAGES_PER_SLOT])
        .expect("attn block table 1");
    let seq_lens = [SEQ_LEN, SEQ_LEN];
    let ref_descs = upload_desc_tables(gpu, &[&block_table0, &block_table1], false, &seq_lens);
    let vmm_descs = upload_desc_tables(gpu, &[&block_table0, &block_table1], true, &seq_lens);
    let ref_k = gpu
        .upload_raw(
            &build_two_slot_arena(0x31, 0x51, false, SEQ_LEN),
            &[REF_ARENA_BYTES],
        )
        .expect("ref k");
    let ref_v = gpu
        .upload_raw(
            &build_two_slot_arena(0x91, 0xb1, false, SEQ_LEN),
            &[REF_ARENA_BYTES],
        )
        .expect("ref v");
    let vmm_k = gpu
        .upload_raw(
            &build_two_slot_arena(0x31, 0x51, true, SEQ_LEN),
            &[VMM_ARENA_BYTES],
        )
        .expect("vmm k");
    let vmm_v = gpu
        .upload_raw(
            &build_two_slot_arena(0x91, 0xb1, true, SEQ_LEN),
            &[VMM_ARENA_BYTES],
        )
        .expect("vmm v");

    let pos_values = [7i32, 17];
    let positions = gpu
        .upload_raw(&i32_bytes(&pos_values), &[2])
        .expect("attn positions");
    let row_slot = gpu.upload_raw(&i32_bytes(&[0, 1]), &[2]).expect("row_slot");
    let q_data: Vec<f32> = (0..2 * N_HEADS * HEAD_DIM)
        .map(|i| ((i as f32) * 0.023) - 0.7)
        .collect();
    let q = gpu
        .upload_f32(&q_data, &[2 * N_HEADS * HEAD_DIM])
        .expect("q");

    let run_lds = |gpu: &mut Gpu, k: &GpuTensor, v: &GpuTensor, descs: &GpuTensor, vmm: bool| {
        let out = gpu
            .zeros(&[2 * N_HEADS * HEAD_DIM], DType::F32)
            .expect("out");
        if vmm {
            gpu.attention_q8_0_kv_batched_masked_slots_vmm_paged(
                &q,
                k,
                v,
                &out,
                &positions,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SEQ_LEN,
                SEQ_LEN,
                2,
                None,
                0,
                0,
                Some(descs),
                Some(&row_slot),
            )
            .expect("vmm LDS");
        } else {
            gpu.attention_q8_0_kv_batched_masked_slots(
                &q,
                k,
                v,
                &out,
                &positions,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SEQ_LEN,
                SEQ_LEN,
                2,
                None,
                0,
                0,
                Some(descs),
                Some(&row_slot),
            )
            .expect("arena LDS");
        }
        gpu.hip.device_synchronize().expect("sync");
        let values = gpu.download_f32(&out).expect("read out");
        gpu.free_tensor(out).unwrap();
        values
    };
    let ref_out = run_lds(gpu, &ref_k, &ref_v, &ref_descs, false);
    let vmm_out = run_lds(gpu, &vmm_k, &vmm_v, &vmm_descs, true);
    assert_eq!(ref_out.len(), vmm_out.len(), "LDS output length mismatch");
    for (i, (a, b)) in ref_out.iter().zip(&vmm_out).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "LDS bit mismatch at {i}: {a} vs {b}"
        );
    }

    // Tile/flash path: one tile of context, same inputs and block tables.
    let run_tile = |gpu: &mut Gpu, k: &GpuTensor, v: &GpuTensor, descs: &GpuTensor, vmm: bool| {
        let out = gpu
            .zeros(&[2 * N_HEADS * HEAD_DIM], DType::F32)
            .expect("out");
        let max_tiles = SEQ_LEN.div_ceil(128);
        let partials = gpu
            .zeros(&[2 * N_HEADS * max_tiles * (2 + HEAD_DIM)], DType::F32)
            .expect("partials");
        if vmm {
            gpu.attention_flash_q8_0_batched_masked_slots_vmm_paged(
                &q,
                k,
                v,
                &out,
                &positions,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SEQ_LEN,
                SEQ_LEN,
                2,
                &partials,
                None,
                0,
                0,
                Some(descs),
                Some(&row_slot),
            )
            .expect("vmm tile");
        } else {
            gpu.attention_flash_q8_0_batched_masked_slots(
                &q,
                k,
                v,
                &out,
                &positions,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SEQ_LEN,
                SEQ_LEN,
                2,
                &partials,
                None,
                0,
                0,
                Some(descs),
                Some(&row_slot),
            )
            .expect("arena tile");
        }
        gpu.hip.device_synchronize().expect("sync");
        let values = gpu.download_f32(&out).expect("read tile out");
        gpu.free_tensor(out).unwrap();
        gpu.free_tensor(partials).unwrap();
        values
    };
    let ref_tile = run_tile(gpu, &ref_k, &ref_v, &ref_descs, false);
    let vmm_tile = run_tile(gpu, &vmm_k, &vmm_v, &vmm_descs, true);
    for (i, (a, b)) in ref_tile.iter().zip(&vmm_tile).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "tile bit mismatch at {i}: {a} vs {b}"
        );
    }

    for t in [
        block_table0,
        block_table1,
        ref_descs,
        vmm_descs,
        ref_k,
        ref_v,
        vmm_k,
        vmm_v,
        positions,
        row_slot,
        q,
    ] {
        gpu.free_tensor(t).unwrap();
    }
    println!("vmm_paged_kernel_parity: LDS PASS, TILE PASS");
}

fn wmma_parity(gpu: &mut Gpu) {
    // 16 query rows per slot, two slots. Positions 2..=17 cross every granule
    // the slot owns (page = 4 tokens, granule = 2 pages). Arena and VMM run
    // the same WMMA body; only the descriptor header differs, so the outputs
    // are bit-identical when granule translation is right.
    const ROWS: usize = 16;
    const BATCH: usize = ROWS * 2;
    let table0: Vec<i32> = (0..PAGES_PER_SLOT as i32).collect();
    let table1: Vec<i32> = (PAGES_PER_SLOT as i32..N_PAGES as i32).collect();
    let block_table0 = gpu
        .upload_raw(&i32_bytes(&table0), &[PAGES_PER_SLOT])
        .expect("wmma block table 0");
    let block_table1 = gpu
        .upload_raw(&i32_bytes(&table1), &[PAGES_PER_SLOT])
        .expect("wmma block table 1");
    let seq_lens = [SEQ_LEN, SEQ_LEN];
    let ref_descs = upload_desc_tables(gpu, &[&block_table0, &block_table1], false, &seq_lens);
    let vmm_descs = upload_desc_tables(gpu, &[&block_table0, &block_table1], true, &seq_lens);
    let ref_k = gpu
        .upload_raw(
            &build_two_slot_arena(0x31, 0x51, false, SEQ_LEN),
            &[REF_ARENA_BYTES],
        )
        .expect("wmma ref k");
    let ref_v = gpu
        .upload_raw(
            &build_two_slot_arena(0x91, 0xb1, false, SEQ_LEN),
            &[REF_ARENA_BYTES],
        )
        .expect("wmma ref v");
    let vmm_k = gpu
        .upload_raw(
            &build_two_slot_arena(0x31, 0x51, true, SEQ_LEN),
            &[VMM_ARENA_BYTES],
        )
        .expect("wmma vmm k");
    let vmm_v = gpu
        .upload_raw(
            &build_two_slot_arena(0x91, 0xb1, true, SEQ_LEN),
            &[VMM_ARENA_BYTES],
        )
        .expect("wmma vmm v");
    let pos_values: Vec<i32> = (0..ROWS)
        .map(|row| 2 + row as i32)
        .chain((0..ROWS).map(|row| 2 + row as i32))
        .collect();
    let positions = gpu
        .upload_raw(&i32_bytes(&pos_values), &[BATCH])
        .expect("wmma positions");
    let tile_slot = gpu
        .upload_raw(&i32_bytes(&[0, 1]), &[2])
        .expect("tile_slot");
    let tile_row0 = gpu
        .upload_raw(&i32_bytes(&[0, 0]), &[2])
        .expect("tile_row0");
    let tile_qbase = gpu
        .upload_raw(&i32_bytes(&[0, ROWS as i32]), &[2])
        .expect("tile_qbase");
    let q_data: Vec<f32> = (0..BATCH * N_HEADS * HEAD_DIM)
        .map(|i| ((i as f32) * 0.019) - 0.5)
        .collect();
    let q = gpu
        .upload_f32(&q_data, &[BATCH * N_HEADS * HEAD_DIM])
        .expect("wmma q");
    let run = |gpu: &mut Gpu, k: &GpuTensor, v: &GpuTensor, descs: &GpuTensor, vmm: bool| {
        let out = gpu
            .zeros(&[BATCH * N_HEADS * HEAD_DIM], DType::F32)
            .expect("wmma out");
        let launch = if vmm {
            gpu.attention_q8_0_flash_prefill_wmma_slots_vmm_paged(
                &q,
                k,
                v,
                &out,
                &positions,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SEQ_LEN,
                BATCH,
                Some(descs),
                Some(&tile_slot),
                Some(&tile_row0),
                Some(&tile_qbase),
            )
        } else {
            gpu.attention_q8_0_flash_prefill_wmma_slots(
                &q,
                k,
                v,
                &out,
                &positions,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                SEQ_LEN,
                BATCH,
                Some(descs),
                Some(&tile_slot),
                Some(&tile_row0),
                Some(&tile_qbase),
            )
        };
        launch.expect("wmma launch");
        gpu.hip.device_synchronize().expect("wmma sync");
        let values = gpu.download_f32(&out).expect("wmma read");
        gpu.free_tensor(out).unwrap();
        values
    };
    let ref_out = run(gpu, &ref_k, &ref_v, &ref_descs, false);
    let vmm_out = run(gpu, &vmm_k, &vmm_v, &vmm_descs, true);
    assert!(
        ref_out.iter().any(|v| v.to_bits() != 0),
        "WMMA reference output is all zeros"
    );
    // Slot 1 must not be a copy of slot 0: the salts differ, so a kernel that
    // pins every tile to slot 0 matches on the first 16 rows and diverges after.
    let row = N_HEADS * HEAD_DIM;
    assert_ne!(
        &ref_out[..row],
        &ref_out[ROWS * row..(ROWS + 1) * row],
        "slot 1 WMMA output matches slot 0; the block table was ignored"
    );
    for (i, (a, b)) in ref_out.iter().zip(&vmm_out).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "WMMA bit mismatch at {i}: {a} vs {b}"
        );
    }
    for t in [
        block_table0,
        block_table1,
        ref_descs,
        vmm_descs,
        ref_k,
        ref_v,
        vmm_k,
        vmm_v,
        positions,
        tile_slot,
        tile_row0,
        tile_qbase,
        q,
    ] {
        gpu.free_tensor(t).unwrap();
    }
    println!("vmm_paged_kernel_parity: WMMA PASS");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init_with_device(dev())?;
    q8_write_parity(&mut gpu);
    attention_parity(&mut gpu);
    wmma_parity(&mut gpu);
    println!("vmm_paged_kernel_parity: PASS");
    Ok(())
}
