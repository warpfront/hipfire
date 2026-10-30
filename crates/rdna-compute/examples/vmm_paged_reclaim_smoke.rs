// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Physical reclaim for the shared VMM page pool under a stable binding.
//!
//! One granule group is the budget. The pool first resides on the high
//! granule. Freeing it does not unmap: a used VA keeps its physical
//! allocation for the mapper's life, because remapping a used VA onto a new
//! allocation loses GPU writes on gfx1100 (see `VmmPageMapper::sync`). So
//! moving residency to the low granule is a capacity refusal, not an
//! unmap-then-map, and the high granule still holds what was written to it.
//! The physical bytes return at teardown, when the arena owners are freed.
//!
//! Run:
//!   cargo run -p rdna-compute --example vmm_paged_reclaim_smoke --features lab

use rdna_compute::page_pool::PageState;
use rdna_compute::vmm_page_pool::{VmmGranuleGeometry, VmmPageMapper};
use rdna_compute::Gpu;

fn dev() -> i32 {
    std::env::var("HIPFIRE_VMM_SMOKE_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init_with_device(dev())?;
    // 128 * 16 KiB = 2 MiB, so the default granule holds exactly one page.
    // Two pages are two granule groups. The budget is one group.
    const PAGE_TOKENS: usize = 128;
    const PER_POS: usize = 16 * 1024;
    let granularity = gpu.vmm_recommended_granularity()?;
    let geom = VmmGranuleGeometry::plan(granularity, PAGE_TOKENS, PER_POS, 0)?;
    assert_eq!(
        geom.pages_per_granule, 1,
        "expected one page per granule, got {} (page {} B, granule {} B)",
        geom.pages_per_granule, geom.page_bytes, geom.granule_bytes
    );
    let group = geom.group_bytes(1)?;
    let (mut mapper, mut arenas) =
        VmmPageMapper::new(&mut gpu, 1, 2, PAGE_TOKENS, PER_POS, 0, group)?;
    assert_eq!(mapper.page_layout().n_granules, 2);
    assert_eq!(mapper.mapped_bytes(), 0, "load is VA-only");

    let high = vec![PageState::Free, PageState::Private];
    mapper.sync(&mut gpu, &high)?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    assert_eq!(
        mapper.mapped_bytes(),
        group,
        "only the high granule is resident"
    );
    mapper.check_table(&[1], 1)?;
    assert!(
        mapper.check_table(&[0], 1).is_err(),
        "the low granule is not mapped yet"
    );

    let offset = mapper.page_offset(1).expect("page 1");
    let pattern = vec![0x5Au8; 4096];
    gpu.hip
        .memcpy_htod_offset(&arenas.k[0].buf, offset, &pattern)?;

    // Same budget, opposite residency: the high granule stays bound, so the
    // low one does not fit. A refusal, not a poison.
    let low = vec![PageState::Private, PageState::Free];
    let err = mapper
        .sync(&mut gpu, &low)
        .expect_err("a bound granule must not be unmapped to make room");
    assert!(err.contains("budget"), "unexpected error: {err}");
    assert!(!mapper.is_poisoned());
    assert_eq!(mapper.mapped_bytes(), group, "residency did not move");
    assert!(
        mapper.check_table(&[0], 1).is_err(),
        "the low granule was never mapped"
    );

    // Every page Free: nothing unmaps, and the bound page kept its bytes.
    let none = vec![PageState::Free, PageState::Free];
    mapper.sync(&mut gpu, &none)?;
    mapper.refresh_owners(&mut gpu, &mut arenas)?;
    assert_eq!(mapper.mapped_bytes(), group, "sync must never unmap");
    mapper.check_table(&[1], 1)?;
    let mut got = vec![0u8; 4096];
    gpu.hip.memcpy_dtoh_at(&mut got, &arenas.k[0].buf, offset)?;
    assert_eq!(got, pattern, "a bound page lost its contents");

    mapper.release();
    for tensor in arenas.k.drain(..).chain(arenas.v.drain(..)) {
        gpu.free_tensor(tensor)?;
    }
    assert_eq!(
        gpu.vmm_allocation_count(),
        0,
        "teardown must return every physical granule"
    );
    println!("vmm_paged_reclaim_smoke: PASS");
    Ok(())
}
