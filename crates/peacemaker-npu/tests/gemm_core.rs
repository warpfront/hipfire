// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use peacemaker_npu::isa::{self, Inst, Program, Reg};
use peacemaker_npu::kernels::{gemm_core as core, gemm_i8};
use peacemaker_npu::sim::{decode::PreparedDecoder, Decoder, Op, Operations, Step, Tile};

fn random(state: &mut u64) -> i8 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as u8 as i8
}
fn refill(tile: &mut Tile, inputs: &[(Vec<i8>, Vec<i8>)], kc: usize, next: &mut usize) {
    while *next < inputs.len() * kc {
        let slot = *next % 2;
        if tile.locks[core::A_EMPTY[slot] as usize] < 1
            || tile.locks[core::B_EMPTY[slot] as usize] < 1
        {
            break;
        }
        assert!(tile.acquire(core::A_EMPTY[slot] as usize, -1).unwrap());
        assert!(tile.acquire(core::B_EMPTY[slot] as usize, -1).unwrap());
        let (a, b) = &inputs[*next / kc];
        let chunk = *next % kc;
        let k = kc * 64;
        let aa = core::A_ADDR[slot] as usize;
        let bb = core::B_ADDR[slot] as usize;
        for mb in 0..16 {
            for kb in 0..8 {
                for row in 0..8 {
                    for kk in 0..8 {
                        tile.memory[aa + (mb * 8 + kb) * 64 + row * 8 + kk] =
                            a[(mb * 8 + row) * k + chunk * 64 + kb * 8 + kk] as u8;
                    }
                }
            }
        }
        for kb in 0..8 {
            for nb in 0..8 {
                for kk in 0..8 {
                    for col in 0..8 {
                        tile.memory[bb + (kb * 8 + nb) * 64 + kk * 8 + col] =
                            b[(chunk * 64 + kb * 8 + kk) * 64 + nb * 8 + col] as u8;
                    }
                }
            }
        }
        tile.release(core::A_FULL[slot] as usize, 1).unwrap();
        tile.release(core::B_FULL[slot] as usize, 1).unwrap();
        *next += 1;
    }
}
fn output(tile: &Tile) -> Vec<i32> {
    let mut result = vec![0; 128 * 64];
    for mb in 0..16 {
        for nb in 0..8 {
            for row in 0..8 {
                for col in 0..8 {
                    let address = core::C_ADDR as usize + (mb * 8 + nb) * 256 + (row * 8 + col) * 4;
                    result[(mb * 8 + row) * 64 + nb * 8 + col] =
                        i32::from_le_bytes(tile.memory[address..address + 4].try_into().unwrap());
                }
            }
        }
    }
    result
}
#[test]
fn gemm_core_lock_protocol_cpu_exact() {
    cpu_exact(core::CoreVariant::Fast);
}
#[test]
fn gemm_core_serial_lock_protocol_cpu_exact() {
    cpu_exact(core::CoreVariant::Serial);
}
fn cpu_exact(variant: core::CoreVariant) {
    for kc in [1, 2, 10, 40] {
        for tiles in [1, 2, 3] {
            let bytes = core::program_variant(kc, tiles, variant).finish();
            let decoder = PreparedDecoder::new(&bytes).unwrap();
            let mut tile = Tile::new(bytes).unwrap();
            tile.locks = core::initial_locks();
            tile.memory[core::C_ADDR as usize..core::C_ADDR as usize + core::C_BYTES].fill(0xa5);
            tile.dm.fill([0x31415926; 64]); // VMUL must not retain a prior output tile.
            let mut state = 0x123456789abcdefu64;
            let inputs: Vec<_> = (0..tiles)
                .map(|_| {
                    let mut a: Vec<_> = (0..128 * kc * 64).map(|_| random(&mut state)).collect();
                    let mut b: Vec<_> = (0..kc * 64 * 64).map(|_| random(&mut state)).collect();
                    a[0] = -128;
                    a[1] = 127;
                    b[0] = 127;
                    b[1] = -128;
                    (a, b)
                })
                .collect();
            let expected: Vec<_> = inputs
                .iter()
                .map(|(a, b)| gemm_i8::cpu_reference(a, b, 128, 64, kc * 64))
                .collect();
            let mut outputs = Vec::new();
            let mut next = 0;
            let mut busy = 0;
            let mut blocked = 0;
            let mut mac_run = 0;
            let mut max_mac_run = 0;
            let mut matrix_ops = 0;
            let mut compute_start = None;
            let mut chunk_cycles = Vec::new();
            for _ in 0..12_000_000 {
                refill(&mut tile, &inputs, kc, &mut next);
                if tile.locks[core::C_FULL as usize] > 0 {
                    assert!(tile.acquire(core::C_FULL as usize, -1).unwrap());
                    outputs.push(output(&tile));
                    tile.release(core::C_EMPTY as usize, 1).unwrap();
                }
                let decoded = decoder.bundle(&tile.program, tile.pc).unwrap();
                let ops = match &decoded.operations {
                    Operations::Single(op) => std::slice::from_ref(op),
                    Operations::Prepared(ops) => ops,
                };
                let mac = ops
                    .iter()
                    .any(|op| matches!(op.op,Op::Multiply {acc,..} if acc!=7));
                matrix_ops += ops
                    .iter()
                    .filter(|op| matches!(op.op, Op::Multiply { .. }))
                    .count();
                if ops.iter().any(
                    |op| matches!(op.op,Op::Move(Reg::P(4),v) if v==core::CORE_BASE+core::C_ADDR),
                ) {
                    compute_start = Some(busy);
                }
                if ops
                    .iter()
                    .any(|op| matches!(op.op, Op::Jump(_, Some(4), false)))
                {
                    if let Some(start) = compute_start.take() {
                        chunk_cycles.push(busy - start);
                    }
                }
                let step = tile
                    .step(&decoder)
                    .unwrap_or_else(|e| panic!("kc={kc} tiles={tiles} pc={:#x}: {e:?}", tile.pc));
                if step == Step::Blocked {
                    blocked += 1;
                } else {
                    busy += 1;
                    if mac {
                        mac_run += 1;
                        max_mac_run = max_mac_run.max(mac_run);
                    } else {
                        mac_run = 0;
                    }
                }
                if step == Step::Done {
                    break;
                }
            }
            assert_ne!(tile.status & (1 << 20), 0, "core did not finish");
            if tile.locks[core::C_FULL as usize] > 0 {
                outputs.push(output(&tile));
            }
            assert_eq!(outputs, expected, "kc={kc} tiles={tiles}");
            assert_eq!(matrix_ops, tiles * kc * 1024);
            println!("gemm_core {variant:?} kc={kc} tiles={tiles}: {} bytes; busy_bundles={busy}; lock_wait={blocked}; matrix_ops={matrix_ops}; longest_1_vmac_per_cycle_run={max_mac_run}",tile.program.len());
            assert_eq!(chunk_cycles.len(), tiles * kc);
            for (chunk, &cycles) in chunk_cycles.iter().enumerate() {
                if variant == core::CoreVariant::Fast && chunk % kc != 0 {
                    assert!(cycles <= 1300, "later chunk {chunk}: {cycles} issue cycles");
                }
            }
            println!(
                "gemm_core compute issues first={} later={:?}",
                chunk_cycles[0],
                chunk_cycles.get(1).filter(|_| kc > 1)
            );
        }
    }
    println!(
        "gemm_core {variant:?} maximum deployment: {} bytes",
        core::program_variant(40, 256, variant).finish().len()
    );
}

#[test]
fn gemm_core_lock_only_publishes_pattern() {
    for kc in [1, 2, 10, 40] {
        for tiles in [1, 2, 3] {
            let bytes = core::program_variant(kc, tiles, core::CoreVariant::LockOnly).finish();
            let decoder = PreparedDecoder::new(&bytes).unwrap();
            let mut tile = Tile::new(bytes).unwrap();
            tile.locks = core::initial_locks();
            tile.memory[core::C_ADDR as usize..core::C_ADDR as usize + core::C_BYTES].fill(0xa5);
            let inputs: Vec<_> = (0..tiles)
                .map(|_| (vec![1i8; 128 * kc * 64], vec![1i8; kc * 64 * 64]))
                .collect();
            let mut next = 0;
            let mut outputs = Vec::new();
            let mut busy = 0;
            for _ in 0..100_000 {
                refill(&mut tile, &inputs, kc, &mut next);
                if tile.locks[core::C_FULL as usize] > 0 {
                    assert!(tile.acquire(core::C_FULL as usize, -1).unwrap());
                    let words: Vec<u32> = tile.memory
                        [core::C_ADDR as usize..core::C_ADDR as usize + 256]
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    outputs.push(words);
                    tile.release(core::C_EMPTY as usize, 1).unwrap();
                }
                let step = tile.step(&decoder).unwrap();
                if step != Step::Blocked {
                    busy += 1;
                }
                if step == Step::Done {
                    break;
                }
            }
            assert_ne!(tile.status & (1 << 20), 0, "lock-only did not finish");
            let expected: Vec<Vec<u32>> = (0..tiles)
                .map(|tile| {
                    (0..64)
                        .map(|i| 0xc0de0000 + tile as u32 * 0x100 + i)
                        .collect()
                })
                .collect();
            assert_eq!(outputs, expected, "LockOnly kc={kc} tiles={tiles}");
            println!(
                "gemm_core LockOnly kc={kc} tiles={tiles}: {} bytes; busy_bundles={busy}",
                tile.program.len()
            );
        }
    }
}

fn packed(instructions: &[Inst]) -> Vec<u8> {
    packed_sized(instructions, false)
}
fn packed_sized(instructions: &[Inst], full: bool) -> Vec<u8> {
    use isa::gen::Slot;
    let mut slots = [None; 8];
    let mut mask = 0u8;
    for &instruction in instructions {
        let (slot, bits) = match instruction {
            Inst::Alu(v) => (Slot::Alu, v),
            Inst::St(v) => (Slot::St, v),
            _ => panic!("fixture slot"),
        };
        slots[slot as usize] = Some(bits);
        mask |= 1 << slot as usize;
    }
    let format = isa::gen::FORMATS
        .iter()
        .filter(|f| {
            (!full || f.bits == 128)
                && f.slots.iter().fold(0, |m, s| m | (1 << s.slot as usize)) & mask == mask
        })
        .min_by_key(|f| f.bits)
        .unwrap();
    let mut bundle = isa::bundle::Bundle::new(format);
    for field in format.slots {
        let nop = || {
            isa::gen::ENCODINGS
                .iter()
                .find(|e| {
                    e.slot == field.slot && e.mnemonic.starts_with("nop") && e.fields.is_empty()
                })
                .unwrap()
                .value
        };
        bundle
            .set(field.slot, slots[field.slot as usize].unwrap_or_else(nop))
            .unwrap();
    }
    bundle.pack().unwrap().as_slice().to_vec()
}
#[test]
fn gemm_core_bundle_reads_are_atomic() {
    let mut bytes = packed(&[
        isa::add_ri(Reg::R(0), Reg::R(0), 1),
        isa::st_idx_imm(Reg::R(0), Reg::P(0), 0),
    ]);
    bytes.extend(isa::done().encode());
    let decoder = PreparedDecoder::new(&bytes).unwrap();
    let mut tile = Tile::new(bytes).unwrap();
    tile.r[0] = 41;
    tile.p[0] = 0x70000;
    assert_eq!(tile.run(&decoder, 16).unwrap(), Step::Done);
    assert_eq!(tile.r[0], 42);
    assert_eq!(u32::from_le_bytes(tile.memory[..4].try_into().unwrap()), 41);
}
#[test]
fn gemm_core_blocked_bundle_has_no_partial_effects() {
    let mut bytes = packed(&[
        isa::acq(48, Reg::R(0)),
        isa::st_post_imm(Reg::R(1), Reg::P(0), 4),
    ]);
    bytes.extend(isa::done().encode());
    let decoder = PreparedDecoder::new(&bytes).unwrap();
    let mut tile = Tile::new(bytes).unwrap();
    tile.r[0] = (-1i32) as u32;
    tile.r[1] = 77;
    tile.p[0] = 0x70000;
    for _ in 0..5 {
        assert_eq!(tile.step(&decoder).unwrap(), Step::Blocked);
    }
    assert_eq!(tile.p[0], 0x70000);
    assert_eq!(&tile.memory[..4], &[0; 4]);
    tile.release(0, 1).unwrap();
    assert_eq!(tile.run(&decoder, 16).unwrap(), Step::Done);
    assert_eq!(tile.p[0], 0x70004);
    assert_eq!(tile.locks[0], 0);
    assert_eq!(u32::from_le_bytes(tile.memory[..4].try_into().unwrap()), 77);
}
#[test]
fn gemm_core_load_latency_observes_old_values() {
    for (wait, expected) in [(0, 0), (6, 8)] {
        let mut p = Program::new();
        p.push(gemm_i8::load_x(0, 0))
            .push(gemm_i8::load_x(1, 1))
            .nops(wait);
        p.push(gemm_i8::matrix_mac(0, None, 0, 1, 2))
            .push(isa::done());
        let decoder = PreparedDecoder::new(&p.bytes).unwrap();
        let mut tile = Tile::new(p.bytes).unwrap();
        tile.p[0] = 0x70000;
        tile.p[1] = 0x70040;
        tile.r[2] = core::SIGNED_8X8;
        tile.memory[..128].fill(1);
        assert_eq!(tile.run(&decoder, 32).unwrap(), Step::Done);
        assert_eq!(tile.dm[0], [expected; 64]);
    }
}
#[test]
fn gemm_core_zero_overhead_loop_counts_retired_iterations() {
    // llvm.set.loop.iterations(N) lowers unchanged to LC: decrement, then
    // compare. A blocked LE bundle must not consume an iteration.
    for trips in [1u32, 2, 6] {
        let mut bytes = Vec::new();
        let enc = isa::gen::ENCODINGS
            .iter()
            .find(|e| e.name == "MOVXM")
            .unwrap();
        for (name, value) in [("ls", 128), ("le", 176), ("lc", trips)] {
            let dst = enc.operands[0]
                .registers
                .iter()
                .find(|r| r.name == name)
                .unwrap()
                .value;
            let bits =
                isa::bundle::encode_slot(enc, &[("dst", dst), ("i", value as u64)], 0).unwrap();
            bytes.extend(
                isa::bundle::pack_slots(&[(enc.slot, bits)])
                    .unwrap()
                    .as_slice(),
            );
        }
        let mut prelude = Program::new();
        prelude.bytes = bytes;
        prelude.nops((128 - prelude.bytes.len()) / 2);
        bytes = prelude.bytes;
        bytes.extend(packed_sized(&[isa::add_ri(Reg::R(2), Reg::R(2), 1)], true));
        bytes.extend(packed_sized(&[], true));
        bytes.extend(packed_sized(&[], true));
        bytes.extend(packed_sized(&[isa::acq(48, Reg::R(0))], true));
        bytes.extend(isa::done().encode());
        let decoder = PreparedDecoder::new(&bytes).unwrap();
        let mut tile = Tile::new(bytes).unwrap();
        tile.r[0] = (-1i32) as u32;
        assert_eq!(tile.run(&decoder, 128).unwrap(), Step::Blocked);
        assert_eq!(tile.r[2], 1);
        for _ in 0..5 {
            assert_eq!(tile.step(&decoder).unwrap(), Step::Blocked);
        }
        tile.release(0, trips as i32).unwrap();
        assert_eq!(tile.run(&decoder, 128).unwrap(), Step::Done);
        assert_eq!(tile.r[2], trips);
        assert_eq!(tile.locks[0], 0);
    }
}
