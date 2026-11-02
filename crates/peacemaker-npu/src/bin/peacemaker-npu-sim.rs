// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Run a stand-alone encoded core image without opening an accelerator.
use peacemaker_npu::sim::{decode::PreparedDecoder, Step, Tile};
fn run() -> std::result::Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|s| s == "array") {
        use peacemaker_npu::kernels::{gemm_array as array, gemm_i8::cpu_reference};
        if args.len() != 5 {
            return Err("array <M> <N> <K>".into());
        }
        let mut shape = [0usize; 3];
        for (v, s) in shape.iter_mut().zip(&args[2..]) {
            *v = s
                .parse()
                .map_err(|e: std::num::ParseIntError| e.to_string())?
        }
        let [m, n, k] = shape;
        if m == 0
            || n == 0
            || k % array::CHUNK_K != 0
            || !(array::CHUNK_K..=array::MAX_KC * array::CHUNK_K).contains(&k)
            || m.div_ceil(array::WAVE_M)
                .checked_mul(n.div_ceil(array::WAVE_N))
                .is_none_or(|waves| waves > array::MAX_WAVES)
        {
            return Err("array requires positive M,N; K=64..2560 divisible by64; at most256 padded512x512waves".into());
        }
        let d = array::design(m, n, k);
        let mut state = 0xc001d00du32;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as i8
        };
        let a: Vec<i8> = (0..m * k).map(|_| random()).collect();
        let b: Vec<i8> = (0..k * n).map(|_| random()).collect();
        let [a_host, b_host] = d.pack_in(&a, &b);
        let mut host = vec![a_host, b_host, vec![0xa5; d.args[2].bytes]];
        let mut sim = peacemaker_npu::sim::config::Config::from_pdi(&d.pdi)?;
        sim.submit(&d.insts, &mut host)?;
        if d.unpack_out(&host[2]) != cpu_reference(&a, &b, m, n, k) {
            return Err("submitted array GEMM output differs from CPU".into());
        }
        println!("PASS array {m}x{n}x{k}: {} CPU-exact host words, {} PDI bytes, {} TXN bytes, {} functional ticks (not hardware cycles)",m*n,d.pdi.len(),d.insts.len(),sim.ticks);
        for c in sim.shim_stats() {
            println!(
                "shim({},0) {:?} channel{}: {} words, {} DRAM bytes",
                c.col,
                c.direction,
                c.channel,
                c.words,
                c.dram_bytes()
            )
        }
        for c in sim.core_stats() {
            println!(
                "core({},{}) busy={} lock_wait={} functional steps",
                c.col, c.row, c.busy, c.lock_wait
            )
        }
        return Ok(());
    }
    if args.get(1).is_some_and(|s| s == "gemm-design") {
        use peacemaker_npu::kernels::gemm_i8::{cpu_reference, design};
        let k = args
            .get(2)
            .map(|s| s.parse::<usize>())
            .transpose()
            .map_err(|e| e.to_string())?
            .unwrap_or(64);
        if ![64, 128, 256].contains(&k) {
            return Err("gemm-design K must be 64,128,256".into());
        }
        let d = design(&[0], 64, 64, k);
        let a: Vec<i8> = (0..64 * k).map(|i| (i * 73 + 128) as i8).collect();
        let b: Vec<i8> = (0..k * 64).map(|i| (i * 29 + 128) as i8).collect();
        let mut host = d.pack_in(&a, &b);
        host.push(vec![0xa5; d.args[1].bytes]);
        let mut sim = peacemaker_npu::sim::config::Config::from_pdi(&d.pdi)?;
        sim.submit(&d.insts, &mut host)?;
        if d.unpack_out(&host[1]) != cpu_reference(&a, &b, 64, 64, k) {
            return Err("submitted GEMM output differs from CPU".into());
        }
        println!("PASS exact GEMM Design: {} PDI bytes, {} TXN bytes, 64x64x{k}, 4096 CPU-exact host words, {} ticks",d.pdi.len(),d.insts.len(),sim.ticks);
        return Ok(());
    }
    if args.get(1).is_some_and(|s| s == "submit") {
        let pdi = args.get(2).ok_or("submit <pdi> <insts> <arg-file>...")?;
        let insts = args.get(3).ok_or("submit <pdi> <insts> <arg-file>...")?;
        let mut host: Vec<Vec<u8>> = args[4..]
            .iter()
            .map(std::fs::read)
            .collect::<std::io::Result<_>>()
            .map_err(|e| e.to_string())?;
        let mut sim = peacemaker_npu::sim::config::Config::from_pdi(
            &std::fs::read(pdi).map_err(|e| e.to_string())?,
        )?;
        sim.submit(&std::fs::read(insts).map_err(|e| e.to_string())?, &mut host)?;
        println!(
            "SYNC complete after {} ticks (input files unchanged)",
            sim.ticks
        );
        for (arg, bytes) in host.iter().enumerate() {
            println!("arg {arg}: {} bytes", bytes.len());
            for (i, w) in bytes.chunks_exact(4).enumerate() {
                println!("{i:04}: {:08x}", u32::from_le_bytes(w.try_into().unwrap()));
            }
        }
        return Ok(());
    }
    if args.get(1).is_some_and(|s| s == "gemm") {
        use peacemaker_npu::kernels::gemm_i8::{cpu_reference, pack_job};
        let k = args
            .get(2)
            .map(|s| s.parse::<usize>())
            .transpose()
            .map_err(|e| e.to_string())?
            .unwrap_or(64);
        if ![64, 128, 256].contains(&k) {
            return Err("gemm K must be 64,128,256".into());
        }
        let a: Vec<i8> = (0..8 * k).map(|i| (i * 73 + 128) as i8).collect();
        let b: Vec<i8> = (0..k * 8).map(|i| (i * 29 + 128) as i8).collect();
        let jobs = vec![
            pack_job(&a, &b, 8, 8, k, 0, 0),
            pack_job(&a, &b, 8, 8, k, 0, 0),
            pack_job(&a, &b, 8, 8, k, 0, 0),
        ];
        let output = peacemaker_npu::sim::gemm::simulate(k, &jobs).map_err(|e| format!("{e:?}"))?;
        let expected: Vec<u8> = cpu_reference(&a, &b, 8, 8, k)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        if output.chunks_exact(256).any(|c| c != expected) {
            return Err("GEMM memory differs from CPU".into());
        }
        println!("PASS encoded GEMM: 3 jobs, 8x{k}x8, 192 output words identical to CPU");
        return Ok(());
    }
    let path = args
        .get(1)
        .ok_or("usage: peacemaker-npu-sim <program.bin> [dma-offset words]")?;
    let offset = args
        .get(2)
        .map(|s| usize::from_str_radix(s.trim_start_matches("0x"), 16))
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(0x8000);
    let words = args
        .get(3)
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(64);
    let end = offset
        .checked_add(words.checked_mul(4).ok_or("size overflow")?)
        .ok_or("size overflow")?;
    let mut tile =
        Tile::new(std::fs::read(path).map_err(|e| e.to_string())?).map_err(|e| format!("{e:?}"))?;
    let decoder = PreparedDecoder::new(&tile.program).map_err(|e| format!("{e:?}"))?;
    let step = tile
        .run(&decoder, 1_000_000)
        .map_err(|e| format!("pc={:#x}: {e:?}", tile.pc))?;
    if step != Step::Done {
        return Err(format!(
            "core blocked at pc={:#x}; locks={:?}",
            tile.pc, tile.locks
        ));
    }
    let output = tile
        .memory
        .get(offset..end)
        .ok_or("output outside tile memory")?;
    println!("DONE status={:#x} pc={:#x}", tile.status, tile.pc);
    for (i, w) in output.chunks_exact(4).enumerate() {
        println!("{i:04}: {:08x}", u32::from_le_bytes(w.try_into().unwrap()));
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
