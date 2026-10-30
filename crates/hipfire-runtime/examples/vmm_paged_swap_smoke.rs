// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 4 oracle: VMM-paged capture/restore uses granule page offsets.
//!
//! Fills a paged VMM arena, captures a slot snapshot, releases the table,
//! restores into a freshly allocated table, and requires the second capture to
//! be byte-identical. Run:
//!   HIPFIRE_VMM_SMOKE_DEVICE=2 cargo run -p hipfire-runtime --example vmm_paged_swap_smoke

use hipfire_runtime::swap::snapshot::{capture_slot_mapped, restore_slot_mapped, SnapshotStamp};
use rdna_compute::page_pool::PAGE_TOKENS;
use rdna_compute::slot_pool::SlotPool;
use rdna_compute::vmm_page_pool::VmmPageMapper;
use rdna_compute::Gpu;

const PER_POS: usize = 1088;
const N_PAGES: usize = 20;
const SEQ_TOKENS: usize = 16 * PAGE_TOKENS;
const PAGE_BYTES: usize = PAGE_TOKENS * PER_POS;

fn dev() -> i32 {
    std::env::var("HIPFIRE_VMM_SMOKE_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(0)
}

fn page_bytes(salt: u8, logical_page: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(PAGE_BYTES);
    for t in 0..PAGE_TOKENS {
        for i in 0..PER_POS {
            bytes.push(
                (salt as usize)
                    .wrapping_mul(31)
                    .wrapping_add(logical_page.wrapping_mul(17))
                    .wrapping_add(t.wrapping_mul(13))
                    .wrapping_add(i.wrapping_mul(7)) as u8,
            );
        }
    }
    bytes
}

fn build_arena(salt: u8, mapper: &VmmPageMapper, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    for lp in 0..SEQ_TOKENS / PAGE_TOKENS {
        let off = mapper.page_offset(lp as u32).expect("page offset");
        bytes[off..off + PAGE_BYTES].copy_from_slice(&page_bytes(salt, lp));
    }
    bytes
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init_with_device(dev())?;
    let mut pool = SlotPool::new_paged_with_strides(1, 4096, PER_POS, PER_POS, N_PAGES)
        .map_err(|e| format!("pool: {e}"))?;
    let (mut mapper, mut arenas) =
        VmmPageMapper::new(&mut gpu, 1, N_PAGES, PAGE_TOKENS, PER_POS, 0, 0)?;
    let layout = mapper.page_layout();
    let slot0 = pool.acquire().expect("slot0");
    pool.set_seq_len(slot0, SEQ_TOKENS)
        .map_err(|e| format!("pages: {e}"))?;
    mapper.sync(&mut gpu, &pool.page_pool().unwrap().page_states())?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;

    let k_data = build_arena(0x31, &mapper, arenas.k[0].buf.size());
    let v_data = build_arena(0x91, &mapper, arenas.v[0].buf.size());
    gpu.hip.memcpy_htod(&arenas.k[0].buf, &k_data)?;
    gpu.hip.memcpy_htod(&arenas.v[0].buf, &v_data)?;
    gpu.hip.device_synchronize()?;

    let stamp = SnapshotStamp {
        model_hash: 0x1234,
        instance_id: 7,
        kv_dtype_tag: 1,
        per_pos_bytes: PER_POS as u32,
        per_pos_v_bytes: PER_POS as u32,
        n_fa_layers: 1,
        dn_layout_version: 1,
        cap: 4096,
        dn_bytes: 0,
    };
    let first = capture_slot_mapped(
        &mut gpu,
        &pool,
        slot0,
        &arenas.k,
        &arenas.v,
        &[],
        &vec![1u32; SEQ_TOKENS],
        stamp,
        Some(layout),
    )?;
    assert_eq!(first.seq_len, SEQ_TOKENS);

    // Return the pages, then allocate a fresh table and restore. The mapper
    // must map the destination granules before the restore writes them.
    pool.release(slot0);
    mapper.sync(&mut gpu, &pool.page_pool().unwrap().page_states())?;
    let slot1 = pool.acquire().expect("slot1");
    pool.provision_pages_for(slot1, SEQ_TOKENS)
        .map_err(|e| format!("reallocate: {e}"))?;
    mapper.sync(&mut gpu, &pool.page_pool().unwrap().page_states())?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    restore_slot_mapped(
        &mut gpu,
        &mut pool,
        slot1,
        &arenas.k,
        &arenas.v,
        &[],
        &first,
        stamp,
        Some(layout),
    )?;
    gpu.hip.device_synchronize()?;
    let second = capture_slot_mapped(
        &mut gpu,
        &pool,
        slot1,
        &arenas.k,
        &arenas.v,
        &[],
        &vec![1u32; SEQ_TOKENS],
        stamp,
        Some(layout),
    )?;
    assert_eq!(
        second.payload, first.payload,
        "VMM-paged capture/restore payload changed"
    );

    pool.release(slot1);
    mapper.sync(&mut gpu, &pool.page_pool().unwrap().page_states())?;
    mapper.release();
    for t in arenas.k.drain(..).chain(arenas.v.drain(..)) {
        gpu.free_tensor(t)?;
    }
    println!("vmm_paged_swap_smoke: PASS");
    Ok(())
}
