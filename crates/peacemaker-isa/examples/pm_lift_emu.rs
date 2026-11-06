// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab, CPU only: the o4 gate/up body (`gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4`)
//! standalone and as a region of the P0 pool dispatcher, both on the PeaceMaker
//! emulator over the same small synthetic problem; prints the first output words
//! that differ. `pm_lift_emu STANDALONE_SA_CO OUT.s`

use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::lifted_dispatch_gfx1201 as ld;
use peacemaker_isa::{native, Arch};
use peacemaker_lift::rewrite::region::{lift_region, RegionInputs};

const SYMBOL: &str = "gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let object = std::fs::read(&args[1]).expect("object");
    let (m, k, n_experts, grouped_rows, tokens) = (16usize, 256usize, 2usize, 16usize, 2usize);
    let mut seed = 0x1234_5678u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let groups = k / 256;
    let (ptrs, w0, tiles, slots, x, y, karg, tasks, q) = (
        0x10_0000u64,
        0x20_0000u64,
        0x40_0000u64,
        0x41_0000u64,
        0x50_0000u64,
        0x60_0000u64,
        0x70_0000u64,
        0x80_0000u64,
        0x90_0000u64,
    );
    let expert_bytes = m * (k / 2) + m * groups * 4;
    let mut weights = Vec::new();
    for _ in 0..n_experts {
        for _ in 0..m * (k / 2) {
            weights.push(next() as u8);
        }
        for _ in 0..m * groups * 2 {
            let h = ((next() & 0x8000) | ((8 + next() % 6) << 10) | (next() & 0x3ff)) as u16;
            weights.extend(h.to_le_bytes());
        }
    }
    let ptr_table: Vec<u8> = (0..n_experts)
        .flat_map(|e| (w0 + (e * expert_bytes) as u64).to_le_bytes())
        .collect();
    let tile_ids: Vec<u8> = [1i32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let slot_index: Vec<u8> = (0..grouped_rows)
        .flat_map(|i| {
            (if i % 5 == 3 {
                -1i32
            } else {
                (next() % (tokens as u32 * 10)) as i32
            })
            .to_le_bytes()
        })
        .collect();
    let xs: Vec<u8> = (0..tokens * k)
        .flat_map(|_| ((next() % 2001) as f32 / 1000.0 - 1.0).to_le_bytes())
        .collect();
    let mut explicit = Vec::new();
    for p in [ptrs, tiles, slots, x, y] {
        explicit.extend(p.to_le_bytes());
    }
    for v in [
        m as i32,
        k as i32,
        10,
        grouped_rows as i32,
        n_experts as i32,
    ] {
        explicit.extend(v.to_le_bytes());
    }
    explicit.resize(64, 0);
    let grid = [1u32, (grouped_rows / 2) as u32];
    let base_mem = || {
        let mut mem = Memory::default();
        mem.map("ptrs".into(), ptrs, ptr_table.clone()).unwrap();
        mem.map("w".into(), w0, weights.clone()).unwrap();
        mem.map("tiles".into(), tiles, tile_ids.clone()).unwrap();
        mem.map("slots".into(), slots, slot_index.clone()).unwrap();
        mem.map("x".into(), x, xs.clone()).unwrap();
        mem.map("y".into(), y, vec![0xa5; grouped_rows * m * 4])
            .unwrap();
        mem.map("karg".into(), karg, explicit.clone()).unwrap();
        mem
    };
    // standalone
    let lifted = peacemaker_lift::lift_object(
        &object,
        peacemaker_lift::Options {
            frontend: Frontend::Hipcc,
        },
    )
    .expect("lift");
    let emu = Emulator::new(&lifted.program, SYMBOL).expect("emu");
    let mut mem = base_mem();
    for gy in 0..grid[1] {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: karg,
                workgroup: [0, gy, 0],
                block: [128, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )
        .unwrap_or_else(|e| panic!("standalone wg {gy}: {e}"));
    }
    let want = mem.read(y, grouped_rows * m * 4).unwrap().to_vec();
    // pool
    let region = lift_region(
        &object,
        SYMBOL,
        &RegionInputs {
            prefix: ".Lt16b".into(),
            workgroup_x: ld::WG_X,
            workgroup_yz: ld::WG_YZ,
            block: 128,
            dynamic_lds: 0,
        },
    )
    .expect("region");

    let e = ld::pool(
        "pm_lift_o4",
        128,
        &[ld::Hosted {
            tmpl: 16,
            region: &region,
        }],
        None,
    )
    .expect("pool");
    std::fs::write(&args[2], &e.s_text).unwrap();
    let elf = native::assemble(&e.s_text, Arch::Gfx1201).expect("assemble");
    let pool = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .expect("lift pool");
    let emu = Emulator::new(&pool.program, "pm_lift_o4").expect("emu pool");
    let mut mem = base_mem();
    let mut t = Vec::new();
    for gy in 0..grid[1] {
        let mut d = [0u32; 32];
        d[..12].copy_from_slice(&[16, 0, gy, 0, 1, grid[1], 1, 0, u32::MAX, 0, u32::MAX, 1]);
        t.extend(d.iter().flat_map(|w| w.to_le_bytes()));
    }
    mem.map("tasks".into(), tasks, t).unwrap();
    mem.map("q".into(), q, vec![0; 64 + 4096]).unwrap();
    let mut params = Vec::new();
    for p in [tasks, q, q + 16, karg, q + 64, q + 32] {
        params.extend(p.to_le_bytes());
    }
    params.extend(1000u64.to_le_bytes());
    params.extend(grid[1].to_le_bytes());
    params.extend(0u32.to_le_bytes());
    params.resize(72, 0);
    mem.map("params".into(), 0xa0_0000, params).unwrap();
    emu.run_workgroup(
        &Launch {
            kernarg_ptr: 0xa0_0000,
            workgroup: [0, 0, 0],
            block: [128, 1, 1],
            dynamic_lds: 0,
        },
        &mut mem,
    )
    .unwrap_or_else(|e| panic!("pool: {e}"));
    let got = mem.read(y, grouped_rows * m * 4).unwrap().to_vec();
    let w32 = |b: &[u8]| -> Vec<u32> {
        b.chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let (w, g) = (w32(&want), w32(&got));
    let bad: Vec<usize> = (0..w.len()).filter(|&i| w[i] != g[i]).collect();
    println!("{} of {} words differ", bad.len(), w.len());
    for &i in bad.iter().take(16) {
        println!("word {i}: standalone {:08x} pool {:08x}", w[i], g[i]);
    }
}
