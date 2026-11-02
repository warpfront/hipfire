// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Two-core V8 pair (lower core row below upper core row, neighbour memory/lock views) CPU-exact checks:
//! the pair shares one 128-row A block through E/O halves and computes 128 x 128 outputs (L = cols 0..64,
//! U = cols 64..128 of the pair's B block). DMA is emulated by the harness exactly as `tile_bds_pair` /
//! `initial_locks_pair` describe it.
use peacemaker_npu::kernels::{gemm_core as core, gemm_i8};
use peacemaker_npu::sim::{decode::PreparedDecoder, Neighbour, Neighbours, Step, Tile};

const M: usize = 128;
const N: usize = 128;

fn random(state: &mut u64) -> i8 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as u8 as i8
}
struct Core {
    tile: Tile,
    half: usize,
    next: usize,
    outputs: Vec<Vec<i32>>,
    outputs8: Vec<Vec<i8>>,
}
fn refill(
    core: &mut Core,
    parity: usize,
    inputs: &[(Vec<i8>, Vec<i8>)],
    kc: usize,
    layout: &core::PairLayout,
) {
    let tile = &mut core.tile;
    while core.next < inputs.len() * kc {
        let slot = core.next % 2;
        if tile.locks[core::A_EMPTY[slot] as usize] < 1
            || tile.locks[core::B_EMPTY[slot] as usize] < 1
        {
            break;
        }
        assert!(tile.acquire(core::A_EMPTY[slot] as usize, -1).unwrap());
        assert!(tile.acquire(core::B_EMPTY[slot] as usize, -1).unwrap());
        let (a, b) = &inputs[core.next / kc];
        let chunk = core.next % kc;
        let k = kc * 64;
        let aa = layout.a[slot] as usize;
        let bb = layout.b[slot] as usize;
        for local in 0..8 {
            let mb = 2 * local + parity;
            for kb in 0..8 {
                for row in 0..8 {
                    for kk in 0..8 {
                        tile.memory[aa + (local * 8 + kb) * 64 + row * 8 + kk] =
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
                            b[(chunk * 64 + kb * 8 + kk) * N + core.half * 64 + nb * 8 + col] as u8;
                    }
                }
            }
        }
        tile.release(core::A_FULL[slot] as usize, 1).unwrap();
        tile.release(core::B_FULL[slot] as usize, 1).unwrap();
        core.next += 1;
    }
}
fn drain(core: &mut Core, epi: core::Epilogue, layout: &core::PairLayout) {
    let tile = &mut core.tile;
    if tile.locks[core::C_FULL as usize] < 1 {
        return;
    }
    assert!(tile.acquire(core::C_FULL as usize, -1).unwrap());
    match epi {
        core::Epilogue::I32 => {
            let mut result = vec![0; 128 * 64];
            for mb in 0..16 {
                for nb in 0..8 {
                    for row in 0..8 {
                        for col in 0..8 {
                            let address =
                                layout.c as usize + (mb * 8 + nb) * 256 + (row * 8 + col) * 4;
                            result[(mb * 8 + row) * 64 + nb * 8 + col] = i32::from_le_bytes(
                                tile.memory[address..address + 4].try_into().unwrap(),
                            );
                        }
                    }
                }
            }
            core.outputs.push(result);
        }
        core::Epilogue::Int8 { .. } => {
            let mut result = vec![0i8; 128 * 64];
            for mb in 0..16 {
                for nb in 0..8 {
                    for row in 0..8 {
                        for col in 0..8 {
                            result[(mb * 8 + row) * 64 + nb * 8 + col] = tile.memory
                                [layout.cout as usize + (mb * 8 + nb) * 64 + row * 8 + col]
                                as i8;
                        }
                    }
                }
            }
            core.outputs8.push(result);
        }
        // The pair core refuses direct BF16 at program-assembly time
        // ("direct BF16 is a G80-only epilogue"), so no run can get here.
        core::Epilogue::Bf16 => unreachable!("direct BF16 is a G80-only epilogue"),
    }
    tile.release(core::C_EMPTY as usize, 1).unwrap();
}
fn pair_exact(kc: usize, tiles: usize, epi: core::Epilogue, ctl: core::Control) {
    let layout = core::pair_layout(ctl.probe().layout);
    let label = format!("kc={kc} tiles={tiles} {epi:?} {ctl:?}");
    let lower_bytes = core::program_pair(kc, tiles, core::PairRole::Lower, epi, ctl).finish();
    let upper_bytes = core::program_pair(kc, tiles, core::PairRole::Upper, epi, ctl).finish();
    let (ld, ud) = (
        PreparedDecoder::new(&lower_bytes).unwrap(),
        PreparedDecoder::new(&upper_bytes).unwrap(),
    );
    let new_core = |bytes: Vec<u8>, half| {
        let mut tile = Tile::new(bytes).unwrap();
        tile.locks = core::initial_locks_pair();
        tile.memory[layout.c as usize..layout.c as usize + core::C_BYTES].fill(0xa5);
        tile.memory[layout.cout as usize..layout.cout as usize + core::COUT_BYTES].fill(0x5a);
        tile.dm.fill([0x31415926; 64]);
        Core {
            tile,
            half,
            next: 0,
            outputs: vec![],
            outputs8: vec![],
        }
    };
    let (mut l, mut u) = (new_core(lower_bytes, 0), new_core(upper_bytes, 1));
    let mut state = 0x123456789abcdefu64 ^ (kc as u64 * 977 + tiles as u64);
    let inputs: Vec<_> = (0..tiles)
        .map(|_| {
            let mut a: Vec<_> = (0..M * kc * 64).map(|_| random(&mut state)).collect();
            let mut b: Vec<_> = (0..kc * 64 * N).map(|_| random(&mut state)).collect();
            a[0] = -128;
            a[1] = 127;
            b[0] = 127;
            b[1] = -128;
            (a, b)
        })
        .collect();
    let expected: Vec<_> = inputs
        .iter()
        .map(|(a, b)| gemm_i8::cpu_reference(a, b, M, N, kc * 64))
        .collect();
    let (mut l_done, mut u_done) = (false, false);
    let mut idle = 0u64;
    for _ in 0..40_000_000u64 {
        refill(&mut l, 0, &inputs, kc, &layout);
        refill(&mut u, 1, &inputs, kc, &layout);
        drain(&mut l, epi, &layout);
        drain(&mut u, epi, &layout);
        let mut progressed = false;
        if !l_done {
            let mut nb = Neighbours {
                south: None,
                west: None,
                north: Some(Neighbour {
                    memory: &mut u.tile.memory,
                    locks: &mut u.tile.locks,
                }),
            };
            let step = l
                .tile
                .step_with(&ld, &mut nb)
                .unwrap_or_else(|e| panic!("{label}: lower pc={:#x}: {e:?}", l.tile.pc));
            l_done = step == Step::Done;
            progressed |= step != Step::Blocked;
        }
        if !u_done {
            let mut nb = Neighbours {
                south: Some(Neighbour {
                    memory: &mut l.tile.memory,
                    locks: &mut l.tile.locks,
                }),
                west: None,
                north: None,
            };
            let step = u
                .tile
                .step_with(&ud, &mut nb)
                .unwrap_or_else(|e| panic!("{label}: upper pc={:#x}: {e:?}", u.tile.pc));
            u_done = step == Step::Done;
            progressed |= step != Step::Blocked;
        }
        if l_done && u_done {
            break;
        }
        idle = if progressed { 0 } else { idle + 1 };
        assert!(
            idle < 1000,
            "{label}: deadlock lower pc={:#x} locks={:?} upper pc={:#x} locks={:?}",
            l.tile.pc,
            l.tile.locks,
            u.tile.pc,
            u.tile.locks
        );
    }
    assert!(l_done && u_done, "{label}: did not finish");
    drain(&mut l, epi, &layout);
    drain(&mut u, epi, &layout);
    for (core, half) in [(&l, 0usize), (&u, 1)] {
        let count = match epi {
            core::Epilogue::I32 => core.outputs.len(),
            core::Epilogue::Int8 { .. } => core.outputs8.len(),
            core::Epilogue::Bf16 => unreachable!("direct BF16 is a G80-only epilogue"),
        };
        assert_eq!(count, tiles, "{label}: core {half} outputs");
        for (wave, reference) in expected.iter().enumerate() {
            for row in 0..M {
                for col in 0..64 {
                    let c = reference[row * N + half * 64 + col];
                    match epi {
                        core::Epilogue::I32 => assert_eq!(
                            core.outputs[wave][row * 64 + col],
                            c,
                            "{label}: core {half} wave {wave} ({row},{col})"
                        ),
                        core::Epilogue::Int8 { shift } => assert_eq!(
                            core.outputs8[wave][row * 64 + col],
                            (c >> shift).clamp(-128, 127) as i8,
                            "{label}: core {half} wave {wave} ({row},{col}) c={c}"
                        ),
                        core::Epilogue::Bf16 => unreachable!("direct BF16 is a G80-only epilogue"),
                    }
                }
            }
        }
    }
}
#[test]
fn pair_core_i32_exact() {
    for ctl in [core::Control::Fast, core::Control::Slow] {
        for (kc, tiles) in [(1, 1), (2, 2), (3, 2)] {
            pair_exact(kc, tiles, core::Epilogue::I32, ctl);
        }
    }
}
#[test]
fn pair_core_int8_exact() {
    for ctl in [core::Control::Fast, core::Control::Slow] {
        for shift in [6u8, 10, 14] {
            for (kc, tiles) in [(1, 1), (2, 2), (3, 2)] {
                pair_exact(kc, tiles, core::Epilogue::Int8 { shift }, ctl);
            }
        }
    }
}
#[test]
fn pair_core_int8_exact_k2560() {
    pair_exact(
        40,
        1,
        core::Epilogue::Int8 { shift: 16 },
        core::Control::Fast,
    );
}
/// Every data-memory layout knob value, both controls and both epilogues: the same exact GEMM comes out of the
/// layout's own C / Cout region. kc=2 exercises the A/B ping-pong, tiles=2 the consecutive-wave C / Cout reuse.
#[test]
fn pair_core_layouts_exact() {
    for layout in 0..=3u8 {
        for serial in [false, true] {
            let ctl = core::Control::Probe(core::Probe {
                layout,
                serial,
                ..Default::default()
            });
            for epi in [core::Epilogue::I32, core::Epilogue::Int8 { shift: 12 }] {
                pair_exact(2, 2, epi, ctl);
            }
        }
    }
}
