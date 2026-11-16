// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.

//! Grow-and-touch soak of the VMM TLB flush poke (`HIPFIRE_VMM_TLB_POKE`).
//!
//! Grows a `VmmArena` one chunk at a time, as a VMM KV context grows, and
//! queues a device memset over the mapped prefix straight after each map, so
//! GPU work touches freshly mapped pages with no host delay. Every map is at a
//! fresh VA, so this exercises the fresh-map race of SDMA-written page tables
//! (reported as page faults on a USB4-attached gfx1201); on a large-BAR card
//! the default poke stays off here and the soak passes either way. The VA-reuse
//! defect is `vmm::tests::vmm_va_reuse_reads_back_its_own_pages_hw`.
//!
//!   HIPFIRE_VMM_SMOKE_DEVICE=1 cargo run --release -p hip-bridge --features lab --example vmm_tlb_flush_soak -- [CHUNK_BYTES] [CHUNKS] [ROUNDS] [TOUCH]
//!   HIPFIRE_VMM_TLB_POKE=0 ...   # never poke; =1 pokes after every map
//!
//! Prints the mean wall time of a map (`map_next`, which includes any poke)
//! and the number of pokes. `TOUCH=0` maps without the memsets: the per-map
//! cost of the poke is the difference between `HIPFIRE_VMM_TLB_POKE=1` and
//! `=0` in that mode.

use hip_bridge::{HipRuntime, VmmArena};

fn arg(index: usize, default: usize) -> usize {
    std::env::args()
        .nth(index)
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = std::env::var("HIPFIRE_VMM_SMOKE_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(0);
    let chunk = arg(1, 2 << 20);
    let chunks = arg(2, 128);
    let rounds = arg(3, 20);
    let touch = arg(4, 1) != 0;
    let hip = HipRuntime::load()?;
    hip.set_device(device)?;
    println!("vmm_tlb_flush_soak: device {device}, {chunks} x {chunk} B chunks, {rounds} rounds, touch {touch}");
    let mut map_time = std::time::Duration::ZERO;
    for round in 0..rounds {
        let mut arena = VmmArena::reserve(&hip, device, chunk * chunks)?;
        let words = |bytes: usize| bytes / 4;
        for c in 0..chunks {
            let t = std::time::Instant::now();
            arena
                .map_next(&hip, chunk, &[device])
                .map_err(|err| format!("round {round} chunk {c}: map: {err}"))?;
            map_time += t.elapsed();
            if !touch {
                continue;
            }
            let prefix = arena.buffer(chunk * (c + 1))?;
            // No host delay: the memset runs over the pages mapped just now.
            hip.memset_d32_async(
                &prefix,
                (round * 7919 + c) as i32,
                words(prefix.size()),
                None,
            )
            .map_err(|err| format!("round {round} chunk {c}: memset: {err}"))?;
        }
        hip.device_synchronize()
            .map_err(|err| format!("round {round}: sync: {err}"))?;
        if !touch {
            arena.release(&hip)?;
            continue;
        }
        let full = arena.buffer(chunk * chunks)?;
        let mut readback = vec![0u8; full.size()];
        hip.memcpy_dtoh(&mut readback, &full)?;
        let expect = ((round * 7919 + chunks - 1) as i32).to_ne_bytes();
        if let Some(bad) = readback
            .as_chunks::<4>()
            .0
            .iter()
            .position(|word| *word != expect)
        {
            return Err(format!("round {round}: word {bad} differs from the last memset").into());
        }
        arena.release(&hip)?;
        if (round + 1) % 10 == 0 {
            println!("  {} rounds verified", round + 1);
        }
    }
    let maps = rounds * chunks;
    println!(
        "vmm_tlb_flush_soak: PASS ({maps} maps{}; mean map {:.1} us; {} TLB pokes)",
        if touch {
            ", each touched right away"
        } else {
            ", not touched"
        },
        map_time.as_secs_f64() * 1e6 / maps as f64,
        hip_bridge::vmm_tlb_pokes()
    );
    Ok(())
}
