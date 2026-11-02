// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use peacemaker_npu::sim::{decode::PreparedDecoder, dma::*, Error, Step, Tile};
use peacemaker_npu::{dma as bd, isa::*};
fn hello_program() -> Vec<u8> {
    let mut p = Program::new();
    p.push(movxm(P(0), 0x78000))
        .push(movxm(R(0), 0xc0de0000))
        .push(mova(R(1), 1))
        .nops(4);
    for _ in 0..64 {
        p.push(st_post_imm(R(0), P(0), 4))
            .push(add_ri(R(0), R(0), 1));
    }
    p.nops(8).push(rel(48, R(1))).nops(8).push(done()).nops(16);
    p.finish()
}
fn run_hello(program: Vec<u8>) {
    let mut tile = Tile::new(program).unwrap();
    let decoder = PreparedDecoder::new(&tile.program).unwrap();
    let mut dma = TileDma::new(Direction::Mm2s);
    dma.descriptors[0] = Some(bd::tile_bd(
        0x8000,
        64,
        bd::BdLocks {
            acq: Some((0, -1)),
            rel: None,
        },
        None,
    ));
    dma.start(0).unwrap();
    let mut shim = ShimDma::new(Direction::S2mm, bd::shim_bd(64, 0)).unwrap();
    let mut core = Fifo::new(2);
    let mut mem = Fifo::new(2);
    let mut south = Fifo::new(2);
    let mut host = vec![0xff; 256];
    assert_eq!(dma.step(&mut tile, &mut core).unwrap(), Step::Blocked);
    let mut complete = false;
    for _ in 0..10000 {
        tile.step(&decoder).unwrap();
        dma.step(&mut tile, &mut core).unwrap();
        core.route(&mut mem);
        mem.route(&mut south);
        if shim.step(&mut host, &mut south).unwrap() == Step::Done && tile.status & (1 << 20) != 0 {
            complete = true;
            break;
        }
    }
    assert!(complete, "hello deadlock");
    let expected: Vec<u8> = (0..64)
        .flat_map(|i| (0xc0de0000u32 + i).to_le_bytes())
        .collect();
    assert_eq!(host, expected);
    assert_eq!(tile.locks[0], 0);
}
#[test]
fn npu_hello_64_words() {
    run_hello(hello_program());
}
#[test]
fn npu_hello_actual_dump() {
    let Ok(path) = std::env::var("NPU_HELLO_PROGRAM") else {
        return;
    };
    run_hello(std::fs::read(path).unwrap());
}
fn vload(x: u64, p: u64) -> Inst {
    Inst::Lda((p << 17) | (1 << 13) | (3 << 11) | (x << 7) | 0b0110111)
}
fn multiply(acc: u64) -> Inst {
    Inst::Vec((2 << 21) | (acc << 15) | (1 << 7) | 4)
}
fn vstore(quarter: u64) -> Inst {
    Inst::St((2 << 17) | (1 << 13) | (3 << 11) | (quarter << 6) | 0b001101)
}
#[test]
fn int8_matmul_matches_cpu_memory() {
    // Two K=8 chunks exercise both initial VMUL and subsequent VMAC.
    for signs in [0, 0x100, 0x200, 0x300] {
        let mut p = Program::new();
        p.push(movxm(P(0), 0x70000))
            .push(movxm(P(1), 0x71000))
            .push(movxm(P(2), 0x72000))
            .push(movxm(R(2), 8 | signs))
            .nops(16);
        // Conservative retirement spacing: this gate checks all signedness
        // modes, not the software-pipelined kernel's issue schedule.
        for acc in [7, 0] {
            p.push(vload(0, 0))
                .push(vload(1, 1))
                .nops(16)
                .push(multiply(acc))
                .nops(16);
        }
        for q in 0..4 {
            p.push(vstore(q));
        }
        p.nops(16).push(done());
        let mut tile = Tile::new(p.finish()).unwrap();
        let decoder = PreparedDecoder::new(&tile.program).unwrap();
        let mut reference = [0i32; 64];
        for chunk in 0..2 {
            for i in 0..64 {
                tile.memory[chunk * 64 + i] = (i * 73 + chunk * 19 + 128) as u8;
                tile.memory[0x1000 + chunk * 64 + i] = (i * 29 + chunk * 113 + 128) as u8;
            }
        }
        for r in 0..8 {
            for c in 0..8 {
                for k in 0..16 {
                    let a = tile.memory[k / 8 * 64 + r * 8 + k % 8];
                    let b = tile.memory[0x1000 + k / 8 * 64 + k % 8 * 8 + c];
                    let a = if signs & 0x200 != 0 {
                        a as i8 as i32
                    } else {
                        a as i32
                    };
                    let b = if signs & 0x100 != 0 {
                        b as i8 as i32
                    } else {
                        b as i32
                    };
                    reference[r * 8 + c] += a * b;
                }
            }
        }
        assert_eq!(tile.run(&decoder, 200).unwrap(), Step::Done);
        let expected: Vec<u8> = reference.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(&tile.memory[0x2000..0x2100], expected);
    }
}
#[test]
fn locks_and_dma_backpressure() {
    let mut tile = Tile::new(vec![]).unwrap();
    assert!(!tile.acquire(0, -1).unwrap());
    tile.release(0, 2).unwrap();
    assert!(tile.acquire(0, -1).unwrap());
    assert_eq!(tile.locks[0], 1);
    assert!(!tile.acquire(0, 2).unwrap());
    assert!(tile.acquire(0, 1).unwrap());
    assert_eq!(tile.locks[0], 1);
    let mut dma = TileDma::new(Direction::S2mm);
    dma.descriptors[0] = Some(bd::tile_bd(
        0,
        2,
        bd::BdLocks {
            acq: Some((0, -1)),
            rel: Some((1, 1)),
        },
        Some(1),
    ));
    dma.descriptors[1] = Some(bd::tile_bd(8, 1, bd::BdLocks::default(), None));
    dma.start(0).unwrap();
    let mut fifo = Fifo::new(1);
    assert_eq!(dma.step(&mut tile, &mut fifo).unwrap(), Step::Blocked);
    assert_eq!(tile.locks[0], 0);
    for v in [123u32, 456, 789] {
        assert!(fifo.push(v));
        dma.step(&mut tile, &mut fifo).unwrap();
    }
    assert_eq!(tile.locks[1], 1);
    assert_eq!(dma.step(&mut tile, &mut fifo).unwrap(), Step::Done);
    assert_eq!(
        &tile.memory[..12],
        [123u32, 456, 789]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    );
}
#[test]
fn scalar_load_and_branch_delay() {
    let mut p = Program::new();
    p.push(movxm(P(0), 0x70000))
        .push(lda_idx_imm(R(0), P(0), 0));
    let branch = p.pc();
    // The target is the 16-byte-aligned DONE after the five delay-slot ADDs (isa::rules BranchTarget).
    let target = (branch + 6 + 5 * 4).next_multiple_of(16);
    p.push(jnz(R(0), target as u64));
    for _ in 0..5 {
        p.push(add_ri(R(1), R(1), 1));
    }
    while p.pc() < target {
        p.nops(1);
    }
    p.push(done());
    let mut tile = Tile::new(p.finish()).unwrap();
    let decoder = PreparedDecoder::new(&tile.program).unwrap();
    tile.memory[..4].copy_from_slice(&1u32.to_le_bytes());
    tile.run(&decoder, 100).unwrap();
    assert_eq!(tile.r[1], 5);
    assert_eq!(Tile::new(vec![0; 16385]).err(), Some(Error::Bounds));
}

#[test]
fn gemm_i8_27_shapes_dma_cpu_exact() {
    use peacemaker_npu::kernels::gemm_i8::{cpu_reference, pack_job};
    let mut state = 0x12345678u32;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as i8
    };
    let mut shapes = 0;
    for m in [64, 128, 256] {
        for n in [64, 128, 256] {
            for k in [64, 128, 256] {
                let mut a: Vec<i8> = (0..m * k).map(|_| random()).collect();
                let mut b: Vec<i8> = (0..k * n).map(|_| random()).collect();
                a[0] = -128;
                b[0] = -128;
                a[1] = 127;
                b[1] = 127;
                let expected = cpu_reference(&a, &b, m, n, k);
                let coordinates: Vec<_> = (0..m / 8)
                    .flat_map(|mb| (0..n / 8).map(move |nb| (mb, nb)))
                    .collect();
                let mut got = vec![0i32; m * n];
                for batch in coordinates.chunks(peacemaker_npu::kernels::gemm_i8::MAX_JOBS) {
                    let packed: Vec<_> = batch
                        .iter()
                        .map(|&(mb, nb)| pack_job(&a, &b, m, n, k, mb, nb))
                        .collect();
                    let output = peacemaker_npu::sim::gemm::simulate(k, &packed).unwrap();
                    for (job, &(mb, nb)) in batch.iter().enumerate() {
                        for r in 0..8 {
                            for c in 0..8 {
                                let off = job * 256 + (r * 8 + c) * 4;
                                got[(mb * 8 + r) * n + nb * 8 + c] =
                                    i32::from_le_bytes(output[off..off + 4].try_into().unwrap());
                            }
                        }
                    }
                }
                assert_eq!(got, expected, "M={m} N={n} K={k}");
                shapes += 1;
                eprintln!("PASS M={m} N={n} K={k}, {} output words", m * n);
            }
        }
    }
    assert_eq!(shapes, 27);
}
