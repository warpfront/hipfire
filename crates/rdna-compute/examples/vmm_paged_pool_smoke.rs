// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 2 single-slot oracle: PagePool page lifecycle + VMM mapper.
//!
//! - a PagePool allocates 16 logical q8 pages for a 2048-token sequence;
//! - the mapper commits only their two granule groups (pages 0..14 and 15);
//! - q8 writes and LDS attention through the `_vmm_paged` kernels are
//!   bit-identical to the arena-paged twin;
//! - releasing the table unmaps nothing: granules keep their physical
//!   allocation for the mapper's life (stable binding);
//! - a fresh sequence reuses the logical pages in place, with no new map.
//!
//! Run:
//!   HIPFIRE_VMM_SMOKE_DEVICE=2 cargo run -p rdna-compute --example vmm_paged_pool_smoke

use rdna_compute::kv_slots::{KvSlotDesc, VmmPagedKvSlotDesc};
use rdna_compute::page_pool::{BlockTable, PagePool, PAGE_TOKENS};
use rdna_compute::vmm_page_pool::{VmmPageArenas, VmmPageMapper};
use rdna_compute::{DType, Gpu};

const PER_POS: usize = 1088;
const N_PAGES: usize = 20;
const SEQ_TOKENS: usize = 16 * PAGE_TOKENS;
const PAGE_BYTES: usize = PAGE_TOKENS * PER_POS;
const REF_ARENA_BYTES: usize = N_PAGES * PAGE_BYTES;
const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const N_HEADS: usize = 8;
const STRIDE: usize = N_KV_HEADS * (HEAD_DIM / 32) * 34;
const SRC_ELEMS: usize = N_KV_HEADS * HEAD_DIM;

fn dev() -> i32 {
    std::env::var("HIPFIRE_VMM_SMOKE_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(0)
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn q8_row(salt: u8, position: usize) -> Vec<u8> {
    let mut row = vec![0u8; STRIDE];
    for block in 0..STRIDE / 34 {
        let base = block * 34;
        row[base] = 0x00;
        row[base + 1] = 0x3c; // f16 1.0 scale
        for i in 0..32 {
            row[base + 2 + i] = (salt as usize)
                .wrapping_mul(37)
                .wrapping_add(position.wrapping_mul(19))
                .wrapping_add(block.wrapping_mul(11))
                .wrapping_add(i.wrapping_mul(7)) as u8;
        }
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
            bytes.extend(std::iter::repeat_n(0x7f, STRIDE));
        }
    }
    bytes
}

fn build_arena(
    salt: u8,
    vmm: bool,
    mapper: &VmmPageMapper,
    arenas: &VmmPageArenas,
    valid_positions: usize,
) -> Vec<u8> {
    let mut bytes = vec![
        0u8;
        if vmm {
            arenas.k[0].buf.size()
        } else {
            REF_ARENA_BYTES
        }
    ];
    for lp in 0..N_PAGES {
        let offset = if vmm {
            mapper.page_offset(lp as u32).expect("page offset")
        } else {
            lp * PAGE_BYTES
        };
        if offset + PAGE_BYTES > bytes.len() {
            // A VMM arena's owner safe length can end below the last free
            // page; only pages the pool allocated are written.
            continue;
        }
        bytes[offset..offset + PAGE_BYTES].copy_from_slice(&page_data(salt, lp, valid_positions));
    }
    bytes
}

fn arena_desc(block_table: u64, seq_len: usize) -> KvSlotDesc {
    KvSlotDesc {
        block_table,
        legacy_k_base: 0,
        legacy_v_base: 0,
        seq_len: seq_len as i32,
        page_tokens: PAGE_TOKENS as i32,
    }
}

fn pack_arena_desc(d: &KvSlotDesc) -> Vec<u8> {
    let mut raw = Vec::with_capacity(32);
    raw.extend_from_slice(&d.block_table.to_ne_bytes());
    raw.extend_from_slice(&d.legacy_k_base.to_ne_bytes());
    raw.extend_from_slice(&d.legacy_v_base.to_ne_bytes());
    raw.extend_from_slice(&d.seq_len.to_ne_bytes());
    raw.extend_from_slice(&d.page_tokens.to_ne_bytes());
    raw
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init_with_device(dev())?;
    let mut pool = PagePool::new_external_with_strides(N_PAGES, PER_POS, PER_POS)
        .map_err(|e| format!("page pool: {e}"))?;
    let (mut mapper, mut arenas) =
        VmmPageMapper::new(&mut gpu, 1, N_PAGES, PAGE_TOKENS, PER_POS, 0, 0)?;
    assert_eq!(mapper.mapped_bytes(), 0, "pool load must be VA-only");

    let mut table = BlockTable::new();
    pool.ensure_capacity(&mut table, SEQ_TOKENS)
        .map_err(|e| format!("allocate pages: {e}"))?;
    assert_eq!(table.num_pages(), 16);
    mapper.sync(&mut gpu, &pool.page_states())?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    let layout = mapper.page_layout();
    assert_eq!(layout.n_granules, 2);
    let granule_cost = layout.granule_bytes * 2;
    assert_eq!(
        mapper.mapped_bytes(),
        2 * granule_cost,
        "pages 0..15 must commit two granule groups"
    );
    mapper
        .check_table(table.page_indices(), table.num_pages())
        .map_err(|e| format!("table check: {e}"))?;

    let block_table = gpu
        .upload_raw(
            &i32_bytes(
                &table
                    .page_indices()
                    .iter()
                    .map(|&p| p as i32)
                    .collect::<Vec<_>>(),
            ),
            &[table.num_pages()],
        )
        .expect("block table upload");
    let block_addr = block_table.buf.as_ptr() as u64;
    let ref_descs = gpu
        .upload_raw(&pack_arena_desc(&arena_desc(block_addr, SEQ_TOKENS)), &[1])
        .expect("ref desc");
    let vmm_desc = mapper.descriptor(block_addr, SEQ_TOKENS);
    vmm_desc
        .check_paged_layout()
        .expect("VMM descriptor layout");
    let vmm_descs = gpu
        .upload_raw(VmmPagedKvSlotDesc::as_bytes(&[vmm_desc]), &[1])
        .expect("vmm desc");

    let ref_k_data = build_arena(0x31, false, &mapper, &arenas, SEQ_TOKENS);
    let ref_v_data = build_arena(0x91, false, &mapper, &arenas, SEQ_TOKENS);
    let vmm_k_data = build_arena(0x31, true, &mapper, &arenas, SEQ_TOKENS);
    let vmm_v_data = build_arena(0x91, true, &mapper, &arenas, SEQ_TOKENS);
    let ref_k = gpu
        .upload_raw(&ref_k_data, &[REF_ARENA_BYTES])
        .expect("ref k");
    let ref_v = gpu
        .upload_raw(&ref_v_data, &[REF_ARENA_BYTES])
        .expect("ref v");
    // Candidate bytes go straight into the mapped VMM owners at their
    // granule-grouped offsets. The shallow clones alias the owners; only the
    // mapper may free them.
    gpu.hip
        .memcpy_htod(&arenas.k[0].buf, &vmm_k_data)
        .expect("fill VMM K arena");
    gpu.hip
        .memcpy_htod(&arenas.v[0].buf, &vmm_v_data)
        .expect("fill VMM V arena");
    let vmm_k = arenas.k[0].shallow_clone();
    let vmm_v = arenas.v[0].shallow_clone();

    // q8 write parity at the last token of page 15 (second granule).
    let positions = gpu
        .upload_raw(&i32_bytes(&[(SEQ_TOKENS - 1) as i32]), &[1])
        .expect("positions");
    let row_slot = gpu.upload_raw(&i32_bytes(&[0]), &[1]).expect("row_slot");
    let src_data: Vec<f32> = (0..SRC_ELEMS)
        .map(|i| ((i as f32) * 0.03125) - 1.0)
        .collect();
    let src = gpu.upload_f32(&src_data, &[SRC_ELEMS]).expect("src");
    gpu.kv_cache_write_q8_0_batched_slots(
        &ref_k,
        &src,
        &positions,
        N_KV_HEADS,
        HEAD_DIM,
        1,
        Some(&ref_descs),
        Some(&row_slot),
        false,
    )
    .expect("reference write");
    gpu.kv_cache_write_q8_0_batched_slots_vmm_paged(
        &vmm_k,
        &src,
        &positions,
        N_KV_HEADS,
        HEAD_DIM,
        1,
        Some(&vmm_descs),
        Some(&row_slot),
        false,
    )
    .expect("vmm write");
    gpu.hip.device_synchronize().expect("write sync");
    let ref_bytes = gpu.download_raw_bytes(&ref_k).expect("read ref k");
    let vmm_bytes = gpu.download_raw_bytes(&vmm_k).expect("read vmm k");
    let last_page = (SEQ_TOKENS - 1) / PAGE_TOKENS;
    let last_tok = (SEQ_TOKENS - 1) % PAGE_TOKENS;
    let ref_tok = last_page * PAGE_BYTES + last_tok * STRIDE;
    let vmm_tok = mapper.page_offset(last_page as u32).unwrap() + last_tok * STRIDE;
    assert_eq!(
        &ref_bytes[ref_tok..ref_tok + STRIDE],
        &vmm_bytes[vmm_tok..vmm_tok + STRIDE],
        "VMM q8 write at page 15 differs"
    );

    // LDS attention parity over the full 2048-token sequence. Both routes read
    // the same logical page bytes from their own physical layout.
    let q_data: Vec<f32> = (0..N_HEADS * HEAD_DIM)
        .map(|i| ((i as f32) * 0.017) - 0.4)
        .collect();
    let q = gpu.upload_f32(&q_data, &[N_HEADS * HEAD_DIM]).expect("q");
    let run_lds = |gpu: &mut Gpu,
                   k: &rdna_compute::GpuTensor,
                   v: &rdna_compute::GpuTensor,
                   descs: &rdna_compute::GpuTensor,
                   vmm: bool| {
        let out = gpu.zeros(&[N_HEADS * HEAD_DIM], DType::F32).expect("out");
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
                SEQ_TOKENS,
                SEQ_TOKENS,
                1,
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
                SEQ_TOKENS,
                SEQ_TOKENS,
                1,
                None,
                0,
                0,
                Some(descs),
                Some(&row_slot),
            )
            .expect("arena LDS");
        }
        gpu.hip.device_synchronize().expect("attn sync");
        let values = gpu.download_f32(&out).expect("read out");
        gpu.free_tensor(out).unwrap();
        values
    };
    let ref_out = run_lds(&mut gpu, &ref_k, &ref_v, &ref_descs, false);
    let vmm_out = run_lds(&mut gpu, &vmm_k, &vmm_v, &vmm_descs, true);
    for (i, (a, b)) in ref_out.iter().zip(&vmm_out).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "LDS bit mismatch at {i}");
    }

    // WMMA over the same real VMM mapping. The query is the last token of
    // page 15, which is the first page of the second granule (P=15).
    let tile_slot = gpu
        .upload_raw(&i32_bytes(&[0]), &[1])
        .expect("wmma tile slot");
    let tile_row0 = gpu
        .upload_raw(&i32_bytes(&[0]), &[1])
        .expect("wmma tile row0");
    let tile_qbase = gpu
        .upload_raw(&i32_bytes(&[0]), &[1])
        .expect("wmma tile qbase");
    let run_wmma = |gpu: &mut Gpu,
                    k: &rdna_compute::GpuTensor,
                    v: &rdna_compute::GpuTensor,
                    descs: &rdna_compute::GpuTensor,
                    vmm: bool| {
        let out = gpu
            .zeros(&[N_HEADS * HEAD_DIM], DType::F32)
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
                SEQ_TOKENS,
                1,
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
                SEQ_TOKENS,
                1,
                Some(descs),
                Some(&tile_slot),
                Some(&tile_row0),
                Some(&tile_qbase),
            )
        };
        launch.expect("wmma");
        gpu.hip.device_synchronize().expect("wmma sync");
        let values = gpu.download_f32(&out).expect("wmma read");
        gpu.free_tensor(out).unwrap();
        values
    };
    let ref_wmma = run_wmma(&mut gpu, &ref_k, &ref_v, &ref_descs, false);
    let vmm_wmma = run_wmma(&mut gpu, &vmm_k, &vmm_v, &vmm_descs, true);
    assert!(
        ref_wmma.iter().any(|v| v.to_bits() != 0),
        "WMMA reference output is all zeros"
    );
    for (i, (a, b)) in ref_wmma.iter().zip(&vmm_wmma).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "WMMA bit mismatch at {i}");
    }

    // Prefix sharing + COW: a second table shares a full 8-page prefix, then
    // copies away one shared page. The mapper must keep the shared granules
    // mapped, map the COW destination, and the copy must use page offsets.
    let mut shared = BlockTable::new();
    pool.share_prefix(&table, &mut shared, 8)
        .expect("share sealed prefix");
    assert_eq!(shared.num_pages(), 8);
    mapper.sync(&mut gpu, &pool.page_states())?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    mapper
        .check_table(shared.page_indices(), shared.num_pages())
        .expect("shared table check");
    let plan = pool
        .plan_cow(&shared, 0, PAGE_TOKENS)
        .expect("plan COW on shared page");
    assert_eq!(plan.copies().len(), 1, "one shared page is copied");
    let copy = plan.copies()[0];
    mapper.sync(&mut gpu, &pool.page_states())?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    for (k, v) in arenas.k.iter().zip(arenas.v.iter()) {
        let src_off = mapper.page_offset(copy.src_phys).expect("src page offset");
        let dst_off = mapper.page_offset(copy.dst_phys).expect("dst page offset");
        gpu.hip
            .memcpy_dtod_at(&k.buf, dst_off, &k.buf, src_off, copy.k_copy_bytes)
            .expect("COW K copy");
        gpu.hip
            .memcpy_dtod_at(&v.buf, dst_off, &v.buf, src_off, copy.v_copy_bytes)
            .expect("COW V copy");
    }
    pool.commit_cow(&mut shared, plan).expect("commit COW");
    assert_eq!(shared.physical(0), Some(copy.dst_phys));
    assert_ne!(copy.dst_phys, copy.src_phys);
    // Divergence: the copied private page must byte-match the sealed source,
    // and writing it afterward must not change the source slot's page.
    let src_off = mapper.page_offset(copy.src_phys).unwrap();
    let dst_off = mapper.page_offset(copy.dst_phys).unwrap();
    let mut src_before = vec![0u8; copy.k_copy_bytes];
    gpu.hip
        .memcpy_dtoh_at(&mut src_before, &arenas.k[0].buf, src_off)
        .expect("read COW source");
    let mut dst_after = vec![0u8; copy.k_copy_bytes];
    gpu.hip
        .memcpy_dtoh_at(&mut dst_after, &arenas.k[0].buf, dst_off)
        .expect("read COW destination");
    assert_eq!(dst_after, src_before, "COW copy does not match source");
    gpu.hip
        .memcpy_htod_offset(&arenas.k[0].buf, dst_off, &vec![0xA5u8; 4096])
        .expect("diverge COW destination");
    let mut src_after = vec![0u8; copy.k_copy_bytes];
    gpu.hip
        .memcpy_dtoh_at(&mut src_after, &arenas.k[0].buf, src_off)
        .expect("read source after divergence");
    assert_eq!(
        src_after, src_before,
        "writing the COW destination mutated the shared source"
    );
    mapper
        .check_table(shared.page_indices(), shared.num_pages())
        .expect("post-COW table check");
    pool.check_invariants();
    pool.release_table(&mut shared)
        .expect("release shared table");
    mapper.sync(&mut gpu, &pool.page_states())?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    pool.check_invariants();

    // Release keeps every granule bound; a fresh table reuses them.
    let mapped_hwm = mapper.mapped_bytes();
    pool.release_table(&mut table).expect("release table");
    mapper.sync(&mut gpu, &pool.page_states())?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    assert_eq!(
        mapper.mapped_bytes(),
        mapped_hwm,
        "sync must never unmap a used granule"
    );
    let mut table2 = BlockTable::new();
    pool.ensure_capacity(&mut table2, 8 * PAGE_TOKENS)
        .expect("reallocate pages");
    mapper.sync(&mut gpu, &pool.page_states())?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    // The free list hands back pages 8..15, which straddle the granule
    // boundary (pages 0..14 / 15..19), so both groups are resident.
    assert_eq!(
        mapper.mapped_bytes(),
        mapped_hwm,
        "reused pages must reuse the bound granules"
    );
    mapper
        .check_table(table2.page_indices(), table2.num_pages())
        .expect("reused table check");
    pool.release_table(&mut table2).expect("release table2");
    mapper.sync(&mut gpu, &pool.page_states())?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    assert_eq!(mapper.mapped_bytes(), mapped_hwm);

    for t in [
        block_table,
        ref_descs,
        vmm_descs,
        ref_k,
        ref_v,
        positions,
        row_slot,
        q,
        tile_slot,
        tile_row0,
        tile_qbase,
    ] {
        gpu.free_tensor(t).unwrap();
    }
    drop(vmm_k);
    drop(vmm_v);
    mapper.release();
    for t in arenas.k.drain(..).chain(arenas.v.drain(..)) {
        gpu.free_tensor(t)?;
    }
    assert_eq!(gpu.vmm_allocation_count(), 0);
    println!("vmm_paged_pool_smoke: PASS");
    Ok(())
}
