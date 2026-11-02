// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! npu-hello: milestone 1 of railgun::npu / peacemaker-npu.
//!
//! Compute tile (col 0, row 2) runs a hand-encoded AIE2P program: write N words
//! `BASE + i` into its data memory, release lock 0, `done`. Its MM2S0 DMA (BD0 waits on lock 0)
//! streams the buffer south through the mem tile to the shim, whose S2MM0 writes it into a host
//! buffer. Every byte submitted (core program, CDO, PDI, transaction stream) is produced here;
//! submission is raw amdxdna ioctls via railgun::npu.
//!
//! usage: npu-hello query | dump <dir> | run [--repeat K]
use peacemaker_npu::{cdo::Cdo, dma, isa::*, pdi, regs, regs::tile, txn::Txn};
use railgun::npu::{Device, HwCtx, ERT_STATE_COMPLETED};

const N: u32 = 64;
const BASE: u32 = 0xC0DE_0000;
const BUF_DMA: u32 = 0x8000; // tile data-memory byte address (DMA view)
const COL: u32 = 0;
const ROW: u32 = 2;
const LOCK: u32 = 0;

fn core_program() -> Vec<u8> {
    let mut p = Program::new();
    p.push(movxm(P(0), regs::core::CORE_VIEW_OWN_MEM + BUF_DMA))
        .push(movxm(R(0), BASE))
        .push(mova(R(1), 1))
        .nops(4);
    for _ in 0..N {
        p.push(st_post_imm(R(0), P(0), 4))
            .push(add_ri(R(0), R(0), 1));
    }
    p.nops(8) // let the last store retire before handing the buffer to the DMA
        .push(rel(regs::core::CORE_LOCK_OWN_BASE + LOCK as u64, R(1)))
        .nops(8)
        .push(done())
        .nops(16);
    p.finish()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| {
            let mut a = [0u8; 4];
            a[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(a)
        })
        .collect()
}

fn build_cdo(prog: &[u8]) -> Cdo {
    use regs::{core as c, mem as m, shim as s};
    let t = |off| tile(COL, ROW, off);
    let mut cdo = Cdo::new();
    // core off, MM2S0 in reset while program memory is loaded
    cdo.mask_write(t(c::CORE_CONTROL), 0b01, 0)
        .mask_write(t(c::DMA_MM2S_0_CTRL), 0b10, 0b10)
        .dma_write(t(c::PROGRAM_MEMORY), &words(prog))
        .mask_write(t(c::DMA_MM2S_0_CTRL), 0b10, 0)
        // pulse core reset
        .mask_write(t(c::CORE_CONTROL), 0b10, 0b10)
        .mask_write(t(c::CORE_CONTROL), 0b10, 0)
        .write(t(c::LOCK0_VALUE + 0x10 * LOCK), 0);
    // MM2S0 BD0: wait for lock >= 1 (acquire -1), send N words from BUF_DMA
    let bd = dma::tile_bd(
        BUF_DMA,
        N,
        dma::BdLocks {
            acq: Some((LOCK, -1)),
            rel: None,
        },
        None,
    );
    cdo.dma_write(t(c::DMA_BD0), &bd)
        .write(t(c::DMA_MM2S_0_START_QUEUE), 0);
    // circuit route: core DMA_0 -> SOUTH0 -> mem NORTH_0 -> SOUTH0 -> shim NORTH_0 -> SOUTH2 -> S2MM0
    cdo.write(
        t(c::SS_MASTER_SOUTH0),
        regs::ss_master(c::SS_SLAVE_DMA_0, c::SS_SLAVE_BASE),
    )
    .write(t(c::SS_SLAVE_DMA_0), regs::SS_ENABLE)
    .write(
        tile(COL, 1, m::SS_MASTER_SOUTH0),
        regs::ss_master(m::SS_SLAVE_NORTH_0, m::SS_SLAVE_BASE),
    )
    .write(tile(COL, 1, m::SS_SLAVE_NORTH_0), regs::SS_ENABLE)
    .write(
        tile(COL, 0, s::SS_MASTER_SOUTH2),
        regs::ss_master(s::SS_SLAVE_NORTH_0, s::SS_SLAVE_BASE),
    )
    .write(tile(COL, 0, s::SS_SLAVE_NORTH_0), regs::SS_ENABLE)
    .mask_write(tile(COL, 0, s::DEMUX_CONFIG), 0x30, 0x10); // SOUTH2 -> shim DMA
                                                            // task-complete token path (TILE_CTRL packets -> SOUTH0 -> controller), needed by TXN sync
    for (a, v) in regs::shim_token_route(COL) {
        cdo.write(a, v);
    }
    // go
    cdo.mask_write(t(c::CORE_CONTROL), 0b01, 0b01);
    cdo
}

fn build_txn() -> Txn {
    use regs::shim as s;
    let mut t = Txn::aie2p_8col();
    t.block_write(tile(COL, 0, s::DMA_BD0), COL, 0, &dma::shim_bd(N, 0))
        .ddr_patch(tile(COL, 0, s::DMA_BD0 + 4), 0, 0)
        .mask_write(tile(COL, 0, s::DMA_S2MM_0_CTRL), 0xf00, 0x1f00) // controller id for the token
        .write32(tile(COL, 0, s::DMA_S2MM_0_TASK_QUEUE), 0x8000_0000) // BD0, issue token
        .sync(COL, 0, 0, 0, 1, 1);
    t
}

fn artifacts() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let prog = core_program();
    let pdi = pdi::build(&build_cdo(&prog).to_words());
    let insts = build_txn().to_bytes();
    (prog, pdi, insts)
}

fn query(dev: &Device) -> Result<(), String> {
    println!("aie metadata: {:?}", dev.meta);
    println!("firmware: {:?}", dev.firmware_version()?);
    println!("clocks: {:?}", dev.clocks()?);
    Ok(())
}

fn run(repeat: usize) -> Result<bool, String> {
    let (prog, pdi_bytes, insts_bytes) = artifacts();
    println!(
        "program {} B, pdi {} B, insts {} B",
        prog.len(),
        pdi_bytes.len(),
        insts_bytes.len()
    );
    let mut dev = Device::open()?;
    query(&dev)?;
    dev.map_heap(64 << 20)?;
    let ntiles = dev.meta.cols as u32 * dev.meta.core.0 as u32;
    let mut all_ok = true;
    for it in 0..repeat {
        // fresh context per iteration: the core runs once per PDI load
        let ctx = HwCtx::create(&dev, ntiles, 2048)?;
        let pdi = dev.dev_bo(&pdi_bytes)?;
        ctx.config_cu(&pdi)?;
        let insts = dev.dev_bo(&insts_bytes)?;
        let mut out = dev.shmem_bo((N * 4) as usize)?;
        for (i, w) in out.as_mut_slice().chunks_mut(4).enumerate() {
            w.copy_from_slice(&(0xDEAD_0000u32 | i as u32 | ((it as u32) << 8)).to_le_bytes());
        }
        out.flush();
        let mut cmd = dev.cmd_bo()?;
        let t0 = std::time::Instant::now();
        let seq = ctx.submit(&mut cmd, &insts, &[&out])?;
        let state = ctx.wait(&cmd, seq, 5000);
        let dt = t0.elapsed();
        out.flush();
        let got = words(out.as_slice());
        let bad = got
            .iter()
            .enumerate()
            .filter(|&(i, &v)| v != BASE + i as u32)
            .count();
        println!(
            "iter {it}: wait={state:?} ({:.1} us) out[0..4]={:08x?} out[N-1]={:08x} mismatches={bad}/{N}",
            dt.as_secs_f64() * 1e6,
            &got[..4],
            got[N as usize - 1]
        );
        let ok = matches!(state, Ok(ERT_STATE_COMPLETED)) && bad == 0;
        if !ok {
            let mut st = vec![0u8; 8192];
            match dev.aie_status(&mut st) {
                Ok(cols) => println!(
                    "aie_status cols_filled={cols} first 256 B: {:02x?}",
                    &st[..256]
                ),
                Err(e) => println!("aie_status: {e}"),
            }
        }
        // Destroy the context while its PDI/insts/arg BOs still exist (a timed-out context is
        // restarted by the driver, which re-reads the PDI BO).
        drop(ctx);
        all_ok &= ok;
        if !ok {
            break;
        }
    }
    Ok(all_ok)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("dump") => {
            let dir = args.get(2).expect("dump <dir>");
            std::fs::create_dir_all(dir).unwrap();
            let (prog, pdi, insts) = artifacts();
            std::fs::write(format!("{dir}/core_0_2.bin"), &prog).unwrap();
            std::fs::write(format!("{dir}/hello.pdi"), &pdi).unwrap();
            std::fs::write(format!("{dir}/hello.insts"), &insts).unwrap();
            println!(
                "wrote {dir}: program {} B, pdi {} B, insts {} B",
                prog.len(),
                pdi.len(),
                insts.len()
            );
        }
        Some("query") => {
            let dev = Device::open().expect("open");
            query(&dev).expect("query");
        }
        Some("run") => {
            let repeat = args
                .iter()
                .position(|a| a == "--repeat")
                .and_then(|i| args.get(i + 1))
                .map(|s| s.parse().unwrap())
                .unwrap_or(1);
            match run(repeat) {
                Ok(true) => println!("RESULT: PASS"),
                Ok(false) => {
                    println!("RESULT: FAIL");
                    std::process::exit(1)
                }
                Err(e) => {
                    println!("RESULT: ERROR {e}");
                    std::process::exit(2)
                }
            }
        }
        _ => {
            eprintln!("usage: npu-hello query | dump <dir> | run [--repeat K]");
            std::process::exit(2)
        }
    }
}
