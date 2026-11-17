// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Phase 0 smoke test for arbitrary VMM page mapping.
//!
//! Proves on hardware:
//! - two pages can be mapped at non-adjacent offsets (a hole is unmapped)
//! - a buffer view rejects an unmapped hole
//! - a page can be unmapped, remapped, and written/read again
//! - overlapping maps are refused without changing the mapped set
//! - release tears down every mapped page
//!
//! Device selection:
//!   HIPFIRE_VMM_SMOKE_DEVICE=2 cargo run -p hip-bridge --example vmm_page_arena_smoke

use hip_bridge::{clear_vmm_faults, inject_vmm_fault, HipRuntime, VmmArena, VmmFaultKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = std::env::var("HIPFIRE_VMM_SMOKE_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(0);
    let hip = HipRuntime::load()?;
    let mut arena = VmmArena::reserve(&hip, device, 64 << 20)?;
    let g = arena.granule_bytes();
    if 3 * g > arena.reserved_bytes() {
        return Err(format!(
            "reserved {} B is smaller than three {g} B granules",
            arena.reserved_bytes()
        )
        .into());
    }
    println!(
        "reserved={} granularity={} granule={} owner={}",
        arena.reserved_bytes(),
        arena.granularity(),
        g,
        arena.owner_device()
    );

    arena.map_at(&hip, 0, g, &[])?;
    arena.map_at(&hip, 2 * g, g, &[])?;
    assert_eq!(arena.mapped_bytes(), 2 * g);
    assert!(
        arena.buffer_at(0, 3 * g).is_err(),
        "hole must not be viewable"
    );
    assert!(arena.buffer_at(0, g).is_ok());

    let first = vec![0x11u8; g];
    let third = vec![0x33u8; g];
    hip.memcpy_htod(&arena.buffer_at(0, g)?, &first)?;
    hip.memcpy_htod(&arena.buffer_at(2 * g, g)?, &third)?;
    let mut readback = vec![0u8; g];
    hip.memcpy_dtoh(&mut readback, &arena.buffer_at(2 * g, g)?)?;
    assert_eq!(readback, third);

    // Overlap is refused and leaves both pages mapped.
    assert!(arena.map_at(&hip, 0, g, &[]).is_err());
    assert!(arena.map_at(&hip, 2 * g, g, &[]).is_err());
    assert_eq!(arena.mapped_bytes(), 2 * g);
    // The hole itself is a valid destination; filling it makes a contiguous view.
    arena.map_at(&hip, g, g, &[])?;
    assert_eq!(arena.mapped_bytes(), 3 * g);
    assert!(arena.buffer_at(0, 3 * g).is_ok());
    arena.unmap_at(&hip, g, g)?;
    assert_eq!(arena.mapped_bytes(), 2 * g);

    // Unmap the first page, remap it, and verify fresh contents.
    arena.unmap_at(&hip, 0, g)?;
    assert_eq!(arena.mapped_bytes(), g);
    assert!(arena.buffer_at(0, 3 * g).is_err());
    arena.map_at(&hip, 0, g, &[])?;
    let replacement = vec![0x55u8; g];
    hip.memcpy_htod(&arena.buffer_at(0, g)?, &replacement)?;
    let mut readback = vec![0u8; g];
    hip.memcpy_dtoh(&mut readback, &arena.buffer_at(0, g)?)?;
    assert_eq!(readback, replacement);
    assert_eq!(arena.mapped_bytes(), 2 * g);

    // Every allocation/map/access failure rolls back without changing the
    // mapped set, and a later map at the same offset succeeds.
    for kind in [
        VmmFaultKind::Create,
        VmmFaultKind::Map,
        VmmFaultKind::AccessReset,
    ] {
        inject_vmm_fault(kind, 1);
        assert!(arena.map_at(&hip, 4 * g, g, &[]).is_err());
        clear_vmm_faults();
        assert_eq!(
            arena.mapped_bytes(),
            2 * g,
            "failed {kind:?} changed mapped set"
        );
    }
    arena.map_at(&hip, 4 * g, g, &[])?;
    assert_eq!(arena.mapped_bytes(), 3 * g);

    // Unmapping an already-unmapped range is refused.
    arena.unmap_at(&hip, 2 * g, g)?;
    assert!(arena.unmap_at(&hip, 2 * g, g).is_err());
    arena.unmap_at(&hip, 0, g)?;
    arena.unmap_at(&hip, 4 * g, g)?;
    assert_eq!(arena.mapped_bytes(), 0);
    arena.release(&hip)?;
    assert!(arena.is_released());

    // An unmap failure leaves the arena poisoned but release still retries
    // the teardown; no handle is leaked.
    let mut fault_arena = VmmArena::reserve(&hip, device, 64 << 20)?;
    fault_arena.map_at(&hip, 0, g, &[])?;
    inject_vmm_fault(VmmFaultKind::Unmap, 1);
    assert!(fault_arena.unmap_at(&hip, 0, g).is_err());
    clear_vmm_faults();
    fault_arena.release(&hip)?;
    assert!(fault_arena.is_released());

    // The same tracked physical granule can be mapped at two VAs and both
    // views alias the same bytes; dropping one mapping leaves the other live.
    let mut alias_arena = VmmArena::reserve(&hip, device, 64 << 20)?;
    alias_arena.map_next_granules(&hip, g, &[])?;
    let id = alias_arena
        .physical_granules()
        .first()
        .map(|&(id, _)| id)
        .expect("one tracked granule");
    let original = vec![0x7au8; g];
    hip.memcpy_htod(&alias_arena.buffer_at(0, g)?, &original)?;
    alias_arena.map_granule_at(&hip, 2 * g, g, id, &[])?;
    let mut readback = vec![0u8; g];
    hip.memcpy_dtoh(&mut readback, &alias_arena.buffer_at(2 * g, g)?)?;
    assert_eq!(readback, original, "alias view reads the original bytes");
    let replacement = vec![0xa5u8; g];
    hip.memcpy_htod(&alias_arena.buffer_at(2 * g, g)?, &replacement)?;
    hip.memcpy_dtoh(&mut readback, &alias_arena.buffer_at(0, g)?)?;
    assert_eq!(
        readback, replacement,
        "alias write is visible at the original"
    );
    alias_arena.unmap_at(&hip, 2 * g, g)?;
    hip.memcpy_dtoh(&mut readback, &alias_arena.buffer_at(0, g)?)?;
    assert_eq!(
        readback, replacement,
        "original mapping survives alias unmap"
    );
    alias_arena.unmap_at(&hip, 0, g)?;
    alias_arena.release(&hip)?;
    assert!(alias_arena.is_released());

    println!("vmm_page_arena_smoke: PASS");
    Ok(())
}
