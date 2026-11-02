// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use peacemaker_npu::sim::{
    config::{Config, TxnOp},
    dma::{Direction, Fifo, ShimDma, TileDma},
    Error, Step, Tile,
};
use peacemaker_npu::{
    cdo::Cdo,
    dma::{lock_write, Bd, BdLocks, Dim, Direction as D, Iteration, Location, ShimAxi, Task},
    pdi,
    route::{Circuit, PacketMaster, PacketRule, Port, ShimDma as RouteDma},
};
fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn circuit(cdo: &mut Cdo, tile: Location, slave: Port, master: Port) {
    Circuit {
        tile,
        slave,
        master,
    }
    .emit_cdo(cdo)
}
fn task(direction: D, channel: u32, bd: u32, repeat: u32, token: bool) -> Task {
    Task {
        direction,
        channel,
        bd,
        repeat,
        issue_token: token,
    }
}
fn enqueue(ops: &mut Vec<TxnOp>, t: Task, loc: Location) {
    let (a, v) = t.write(loc);
    ops.push(TxnOp::Write(a, v));
}
fn shim(cdo: &mut Cdo, loc: Location, channel: u8) {
    RouteDma {
        tile: loc,
        direction: D::S2mm,
        channel,
    }
    .emit_cdo(cdo);
    cdo.write(loc.address(0x1d200 + channel as u32 * 8), 15 << 8);
    PacketMaster {
        tile: loc,
        port: Port::South(0),
        arbiter: 5,
        selects: 8,
        drop_header: true,
    }
    .emit_cdo(cdo);
    PacketRule {
        tile: loc,
        slave: Port::TileCtrl,
        slot: 0,
        id: 15,
        mask: 31,
        select: 3,
        arbiter: 5,
    }
    .emit_cdo(cdo);
}
#[test]
fn compute_and_shim_tensor_addressing() {
    let mut bd = Bd::new(16, 12);
    bd.dims = [
        Dim { step: 1, wrap: 2 },
        Dim { step: 5, wrap: 2 },
        Dim { step: 12, wrap: 0 },
        Dim::default(),
    ];
    bd.iteration = Iteration {
        step: 40,
        wrap: 2,
        current: 1,
    };
    let positions = [44, 45, 49, 50, 56, 57, 61, 62, 68, 69, 73, 74];
    let mut tile = Tile::new(Vec::new()).unwrap();
    for i in 0..100 {
        tile.memory[i * 4..i * 4 + 4].copy_from_slice(&(1000 + i as u32).to_le_bytes());
    }
    let mut dma = TileDma::new(Direction::Mm2s);
    dma.descriptors[0] = Some(bd.tile_words());
    dma.start(0).unwrap();
    let mut fifo = Fifo::new(1);
    let mut actual = Vec::new();
    for _ in 0..12 {
        let step = dma.step(&mut tile, &mut fifo).unwrap();
        assert_ne!(step, Step::Blocked);
        actual.push(fifo.pop().unwrap());
    }
    assert_eq!(actual, positions.map(|i| 1000 + i));
    assert_eq!(dma.step(&mut tile, &mut fifo).unwrap(), Step::Done);
    let mut host = words(&(0..100).collect::<Vec<_>>());
    let mut dma = ShimDma::new(Direction::Mm2s, bd.shim_words()).unwrap();
    let mut actual = Vec::new();
    for _ in 0..12 {
        dma.step(&mut host, &mut fifo).unwrap();
        actual.push(fifo.pop().unwrap());
    }
    assert_eq!(actual, positions);
}
/// Latched AXI attributes read back from the BD words, with MM2S and S2MM tensor addressing unchanged: the same
/// strided pattern (D0 2x1, D1 2x5, iteration at +40 words, base word 4) is read and written exactly for every
/// attribute tuple, including the largest legal fields.
#[test]
fn shim_axi_roundtrips_and_leaves_tensor_addressing_exact() {
    let mut bd = Bd::new(16, 12);
    bd.dims = [
        Dim { step: 1, wrap: 2 },
        Dim { step: 5, wrap: 2 },
        Dim { step: 12, wrap: 0 },
        Dim::default(),
    ];
    bd.iteration = Iteration {
        step: 40,
        wrap: 2,
        current: 1,
    };
    let positions = [44usize, 45, 49, 50, 56, 57, 61, 62, 68, 69, 73, 74];
    for axi in [
        ShimAxi::default(),
        ShimAxi {
            burst: 0,
            cache: 0,
            qos: 0,
        },
        ShimAxi {
            burst: 1,
            cache: 3,
            qos: 5,
        },
        ShimAxi {
            burst: 2,
            cache: 0xb,
            qos: 9,
        },
        ShimAxi {
            burst: 3,
            cache: 0xf,
            qos: 0xf,
        },
        ShimAxi {
            burst: 0,
            cache: 0xf,
            qos: 0,
        },
        ShimAxi {
            burst: 3,
            cache: 0,
            qos: 0xf,
        },
    ] {
        bd.axi = axi;
        let words_bd = bd.shim_words_for_hazardous_probe();
        // MM2S: read the pattern from host memory.
        let mut host = words(&(0..100).collect::<Vec<_>>());
        let mut fifo = Fifo::new(1);
        let mut dma = ShimDma::new(Direction::Mm2s, words_bd).unwrap();
        assert_eq!(dma.axi(), axi);
        let mut actual = Vec::new();
        for _ in 0..12 {
            dma.step(&mut host, &mut fifo).unwrap();
            actual.push(fifo.pop().unwrap());
        }
        assert_eq!(actual, positions.map(|p| p as u32), "MM2S {axi:?}");
        assert_eq!(dma.axi(), axi, "attrs changed by running MM2S");
        assert_eq!(
            host,
            words(&(0..100).collect::<Vec<_>>()),
            "MM2S modified host {axi:?}"
        );
        // S2MM: scatter 12 stream words to the same positions and nowhere else.
        let mut host = vec![0xa5u8; 100 * 4];
        let mut dma = ShimDma::new(Direction::S2mm, words_bd).unwrap();
        assert_eq!(dma.axi(), axi);
        for i in 0..12u32 {
            assert!(fifo.push(1000 + i));
            for _ in 0..4 {
                if fifo.is_empty() {
                    break;
                }
                dma.step(&mut host, &mut fifo).unwrap();
            }
            assert!(fifo.is_empty(), "S2MM {axi:?} left word {i} unconsumed");
        }
        assert_eq!(dma.step(&mut host, &mut fifo).unwrap(), Step::Done);
        let mut expected = vec![0xa5a5_a5a5u32; 100];
        for (i, p) in positions.iter().enumerate() {
            expected[*p] = 1000 + i as u32;
        }
        assert_eq!(host, words(&expected), "S2MM {axi:?}");
        assert_eq!(dma.axi(), axi, "attrs changed by running S2MM");
    }
}
/// AxCACHE/AxQoS/burst are accepted; the neighbouring reserved fields (SMID in BDi_5 bits 28..31, SecureAccess
/// in BDi_3 bits 30..31) still fail explicitly instead of being silently dropped.
#[test]
fn shim_reserved_smid_and_secure_access_are_rejected_explicitly() {
    let mut bd = Bd::new(16, 12);
    bd.dims = [
        Dim { step: 1, wrap: 2 },
        Dim { step: 5, wrap: 2 },
        Dim { step: 12, wrap: 0 },
        Dim::default(),
    ];
    bd.axi = ShimAxi {
        burst: 3,
        cache: 0xf,
        qos: 0xf,
    };
    let good = bd.shim_words_for_hazardous_probe();
    ShimDma::new(Direction::Mm2s, good).unwrap();
    for bit in 28..32 {
        let mut w = good;
        w[5] |= 1 << bit;
        assert!(
            matches!(ShimDma::new(Direction::Mm2s, w), Err(Error::Unsupported)),
            "SMID bit {bit}"
        );
    }
    for bit in 30..32 {
        let mut w = good;
        w[3] |= 1 << bit;
        assert!(
            matches!(ShimDma::new(Direction::S2mm, w), Err(Error::Unsupported)),
            "BD3 bit {bit}"
        );
    }
}
#[test]
fn memtile_four_dimensions_chain_and_iteration_repeat() {
    let s = Location::new(0, 0);
    let m = Location::new(0, 1);
    let mut cdo = Cdo::new();
    for (loc, slave, master) in [
        (s, Port::South(3), Port::North(0)),
        (m, Port::South(0), Port::Dma(0)),
        (m, Port::Dma(0), Port::South(0)),
        (s, Port::North(0), Port::South(2)),
    ] {
        circuit(&mut cdo, loc, slave, master)
    }
    RouteDma {
        tile: s,
        direction: D::Mm2s,
        channel: 0,
    }
    .emit_cdo(&mut cdo);
    shim(&mut cdo, s, 0);
    Bd::new(0, 32).emit_cdo(s, 0, &mut cdo);
    Bd::new(0, 96).emit_cdo(s, 1, &mut cdo);
    let mut input = Bd::new(0x80000, 16);
    input.dims = [
        Dim { step: 1, wrap: 2 },
        Dim { step: 5, wrap: 2 },
        Dim { step: 12, wrap: 2 },
        Dim { step: 24, wrap: 0 },
    ];
    input.locks = BdLocks {
        acq: None,
        rel: Some((64, 1)),
    };
    input.next = Some(2);
    input.emit_cdo(m, 0, &mut cdo);
    input.addr += 48 * 4;
    input.next = None;
    input.emit_cdo(m, 2, &mut cdo);
    let mut output = Bd::new(0x80000, 48);
    output.iteration = Iteration {
        step: 48,
        wrap: 2,
        current: 0,
    };
    output.locks = BdLocks {
        acq: Some((64, -1)),
        rel: None,
    };
    output.emit_cdo(m, 4, &mut cdo);
    for t in [task(D::S2mm, 0, 0, 1, false), task(D::Mm2s, 0, 4, 2, false)] {
        t.emit_cdo(m, &mut cdo)
    }
    let mut sim = Config::from_pdi(&pdi::build(&cdo.to_words())).unwrap();
    let input: Vec<u32> = (1..=32).collect();
    let mut args = vec![words(&input), vec![0xa5; 96 * 4]];
    let mut ops = vec![
        TxnOp::DdrPatch {
            address: Bd::address(s, 0) + 4,
            arg: 0,
            plus: 0,
        },
        TxnOp::DdrPatch {
            address: Bd::address(s, 1) + 4,
            arg: 1,
            plus: 0,
        },
    ];
    enqueue(&mut ops, task(D::Mm2s, 0, 0, 1, false), s);
    enqueue(&mut ops, task(D::S2mm, 0, 1, 1, true), s);
    ops.push(TxnOp::Sync {
        col: 0,
        row: 0,
        direction: 0,
        channel: 0,
        cols: 1,
        rows: 1,
    });
    sim.execute(&ops, &mut args).unwrap();
    let positions = [0, 1, 5, 6, 12, 13, 17, 18, 24, 25, 29, 30, 36, 37, 41, 42];
    let mut expected = vec![0u32; 96];
    for (i, p) in positions.iter().enumerate() {
        expected[*p] = input[i];
        expected[48 + *p] = input[16 + i];
    }
    assert_eq!(args[1], words(&expected));
    assert_eq!(args[0], words(&input));
    assert_eq!(
        sim.shim_stats().map(|c| c.dram_bytes()).sum::<u64>(),
        (32 + 96) * 4
    );
}
#[test]
fn multicast_waits_for_every_destination() {
    let s = Location::new(0, 0);
    let m = Location::new(0, 1);
    let mut cdo = Cdo::new();
    for (loc, slave, master) in [
        (s, Port::South(3), Port::North(0)),
        (m, Port::South(0), Port::Dma(0)),
        (m, Port::South(0), Port::Dma(2)),
        (m, Port::Dma(0), Port::South(0)),
        (m, Port::Dma(2), Port::South(1)),
        (s, Port::North(0), Port::South(2)),
        (s, Port::North(1), Port::South(3)),
    ] {
        circuit(&mut cdo, loc, slave, master)
    }
    RouteDma {
        tile: s,
        direction: D::Mm2s,
        channel: 0,
    }
    .emit_cdo(&mut cdo);
    shim(&mut cdo, s, 0);
    shim(&mut cdo, s, 1);
    for id in 0..3 {
        Bd::new(0, 16).emit_cdo(s, id, &mut cdo)
    }
    for (channel, base, full, empty) in [(0, 0x80000, 64, None), (2, 0x80100, 66, Some((65, -1)))] {
        let mut input = Bd::new(base, 16);
        input.locks = BdLocks {
            acq: empty,
            rel: Some((full, 1)),
        };
        input.emit_cdo(m, channel, &mut cdo);
        let mut output = Bd::new(base, 16);
        output.locks = BdLocks {
            acq: Some((full, -1)),
            rel: None,
        };
        output.emit_cdo(m, channel + 4, &mut cdo);
        for t in [
            task(D::S2mm, channel, channel, 1, false),
            task(D::Mm2s, channel, channel + 4, 1, false),
        ] {
            t.emit_cdo(m, &mut cdo)
        }
    }
    let mut sim = Config::from_pdi(&pdi::build(&cdo.to_words())).unwrap();
    let input = words(&(1..=16).collect::<Vec<_>>());
    let mut args = vec![input.clone(), vec![0xa5; 64], vec![0xa5; 64]];
    let mut ops = Vec::new();
    for id in 0..3 {
        ops.push(TxnOp::DdrPatch {
            address: Bd::address(s, id) + 4,
            arg: id as usize,
            plus: 0,
        })
    }
    enqueue(&mut ops, task(D::Mm2s, 0, 0, 1, false), s);
    enqueue(&mut ops, task(D::S2mm, 0, 1, 1, true), s);
    enqueue(&mut ops, task(D::S2mm, 1, 2, 1, true), s);
    sim.execute(&ops, &mut args).unwrap();
    for _ in 0..100 {
        sim.tick(&mut args).unwrap()
    }
    // The blocked destination has four FIFO slots: the fifth word cannot be
    // delivered to the otherwise-ready destination either.
    sim.tick_limit = 0;
    sim.execute(
        &[
            TxnOp::MaskPoll(m.address(0), u32::MAX, 1),
            TxnOp::MaskPoll(m.address(16), u32::MAX, 0),
        ],
        &mut args,
    )
    .unwrap();
    assert_eq!(args[1], vec![0xa5; 64]);
    let (a, v) = lock_write(m, 1, 1);
    sim.tick_limit = 1000;
    sim.execute(
        &[
            TxnOp::Write(a, v),
            TxnOp::Sync {
                col: 0,
                row: 0,
                direction: 0,
                channel: 0,
                cols: 1,
                rows: 1,
            },
            TxnOp::Sync {
                col: 0,
                row: 0,
                direction: 0,
                channel: 1,
                cols: 1,
                rows: 1,
            },
        ],
        &mut args,
    )
    .unwrap();
    assert_eq!(args[1], input);
    assert_eq!(args[2], input);
}

#[test]
fn unsupported_register_modes_fail_explicitly() {
    let m = Location::new(0, 1);
    let c = Location::new(0, 2);
    for (addr, value) in [
        (m.address(0x90000), 0),
        (c.address(0x1d018), 0),
        (m.address(0xa060c), 0),
        (m.address(0xa0600), 1),
        (0x1d200, 2),
        (c.address(0x1de04), 1 << 31),
        (m.address(0xb0050), 1 << 31),
        (m.address(0xb0000), 3 << 30),
        (c.address(0x3f010), 1 << 31),
    ] {
        assert!(
            Config::new()
                .load_cdo(&[peacemaker_npu::cdo::Cmd::Write(addr, value)])
                .is_err(),
            "accepted unsupported {addr:#x}={value:#x}"
        );
    }
}

#[test]
fn iteration_advances_once_per_bd_load_under_backpressure() {
    let mut bd = Bd::new(0, 2);
    bd.iteration = Iteration {
        step: 16,
        wrap: 2,
        current: 0,
    };
    bd.next = Some(0);
    let mut tile = Tile::new(Vec::new()).unwrap();
    tile.memory[..128].copy_from_slice(&words(&(0..32).collect::<Vec<_>>()));
    let mut dma = TileDma::new(Direction::Mm2s);
    dma.descriptors[0] = Some(bd.tile_words());
    dma.start(0).unwrap();
    let mut fifo = Fifo::new(1);
    let mut actual = Vec::new();
    for _ in 0..5 {
        dma.step(&mut tile, &mut fifo).unwrap();
        for _ in 0..5 {
            assert_eq!(dma.step(&mut tile, &mut fifo).unwrap(), Step::Blocked)
        }
        actual.push(fifo.pop().unwrap());
    }
    assert_eq!(actual, [0, 1, 16, 17, 0]);
}
