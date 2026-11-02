// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use peacemaker_npu::kernels::bw_probe::{self, BwConfig, WriteCheck};
use peacemaker_npu::sim::{config::Config, dma::Direction};

/// Runs `submits` consecutive submits of the exact PDI+TXN bytes on one simulator context. Every submit must
/// finish (all SYNCs satisfied), count exactly the configured DRAM bytes per shim channel, and (after poisoning
/// the write buffer) leave the memtile pattern repeated in every write segment.
fn run(cfg: BwConfig, submits: usize) {
    run_design(bw_probe::design(cfg), submits);
}

fn run_design(d: bw_probe::BwDesign, submits: usize) {
    let cfg = d.cfg;
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    let mut args: Vec<Vec<u8>> = d.args.iter().map(|a| vec![0xa5; a.bytes]).collect();
    // The read buffer carries a distinct, never-checked pattern; it must come back unmodified.
    for (i, b) in args[0].iter_mut().enumerate() {
        *b = (i * 7 + 1) as u8;
    }
    let read_before = args[0].clone();
    for s in 0..submits {
        args[1].fill(0xa5);
        sim.submit(&d.insts, &mut args)
            .unwrap_or_else(|e| panic!("{cfg:?} submit {s}: {e}"));
        assert_eq!(
            d.check_write(&args[1]),
            WriteCheck {
                bad_words: 0,
                first_bad: None
            },
            "{cfg:?} submit {s}"
        );
        assert_eq!(args[0], read_before, "{cfg:?} read buffer modified");
        // Counters accumulate across submits: every configured shim channel moved exactly `bytes` per submit.
        for c in sim.shim_stats() {
            let active = (c.col as usize) < cfg.cols
                && (c.channel as usize)
                    < match c.direction {
                        Direction::Mm2s => cfg.read_ch,
                        Direction::S2mm => cfg.write_ch,
                    };
            let want = if active {
                ((s + 1) * cfg.bytes) as u64
            } else {
                0
            };
            assert_eq!(
                c.dram_bytes(),
                want,
                "{cfg:?} shim col {} {:?} ch {} after submit {s}",
                c.col,
                c.direction,
                c.channel
            );
        }
    }
    eprintln!("PASS bw_probe {cfg:?} x{submits}: {} PDI B, {} TXN B, {} functional ticks (not hardware cycles)", d.pdi.len(), d.insts.len(), sim.ticks);
}

fn cfg(cols: usize, read_ch: usize, write_ch: usize, bytes: usize) -> BwConfig {
    BwConfig {
        cols,
        read_ch,
        write_ch,
        bytes,
        axi: Default::default(),
    }
}

#[test]
fn one_column_read_16k() {
    run(cfg(1, 1, 0, 16 << 10), 2);
}
#[test]
fn one_column_read_64k() {
    run(cfg(1, 1, 0, 64 << 10), 2);
}
#[test]
fn one_column_write_16k() {
    run(cfg(1, 0, 1, 16 << 10), 2);
}
#[test]
fn one_column_write_64k() {
    run(cfg(1, 0, 1, 64 << 10), 2);
}
#[test]
fn eight_columns_read2_write2_16k() {
    run(cfg(8, 2, 2, 16 << 10), 2);
}
#[test]
fn eight_columns_read2_write2_64k() {
    run(cfg(8, 2, 2, 64 << 10), 2);
}
/// Hardware-only AXI knobs (burst length, AxCACHE, AxQoS) must not leak into the neighbouring BD fields: the probe
/// stays exact (read and write) for the conservative AxCACHE 0/2/3 experiment candidates (not a hardware safety
/// guarantee). This is a functional-model check only: the simulator has no AXI timing or coherence model.
#[test]
fn shim_axi_variants_move_the_same_data() {
    for (burst, cache, qos) in [
        (0, 2, 0),
        (1, 2, 0),
        (2, 2, 0),
        (3, 0, 0),
        (3, 3, 0),
        (3, 2, 0xf),
    ] {
        run_design(
            bw_probe::design_for_hazardous_probe(BwConfig {
                axi: peacemaker_npu::dma::ShimAxi { burst, cache, qos },
                ..cfg(2, 2, 2, 20 << 10)
            }),
            2,
        );
    }
}
/// Every legal field value, including AxCACHE allocate values (0xb, 0xf) that must not be assumed correct on real
/// host buffers,
/// is accepted and leaves the data path untouched. Proves BD field acceptance in the functional model only; it says
/// nothing about coherence or correctness on hardware.
#[test]
fn shim_axi_all_legal_field_values_are_accepted_by_the_functional_model() {
    for (burst, cache, qos) in [(0, 0xb, 0), (3, 0xf, 0xf), (2, 0xf, 7)] {
        run_design(
            bw_probe::design_for_hazardous_probe(BwConfig {
                axi: peacemaker_npu::dma::ShimAxi { burst, cache, qos },
                ..cfg(2, 2, 2, 20 << 10)
            }),
            2,
        );
    }
}
/// Sizes that are not a multiple of the 16 KiB source buffer exercise the short tail BD, the chain suffix and
/// the repeated chain (CHAIN = 16 BDs: 2 repeats + 3 buffers + 4 KiB).
#[test]
fn write_tail_and_repeat_shapes() {
    run(cfg(1, 0, 1, 20 << 10), 2);
    run(cfg(2, 0, 2, (3 * 16 + 1) * (16 << 10) + (8 << 10)), 2);
    run(cfg(1, 1, 1, 36 << 10), 2);
}

/// Model the observed corruption signature as a mutation of low memtile word 0 between submits.
/// This does not model or identify the proprietary firmware operation that caused the hardware failure.
/// The persistent write sources must remain exact, including repeated chains, chain suffixes and short tails.
#[test]
fn low_memtile_word_mutation_does_not_corrupt_repeated_writes() {
    let d = bw_probe::design(cfg(
        2,
        2,
        2,
        (2 * bw_probe::CHAIN + 3) * bw_probe::BUF_BYTES + 4096,
    ));
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    let mut args: Vec<Vec<u8>> = d.args.iter().map(|a| vec![0xa5; a.bytes]).collect();
    for submit in 0..3 {
        if submit > 0 {
            let mutations: Vec<_> = (0..d.cfg.cols as u32)
                .map(|col| {
                    peacemaker_npu::sim::config::TxnOp::Write(
                        peacemaker_npu::dma::Location::new(col, 1).address(0),
                        0x00cd_0cd0,
                    )
                })
                .collect();
            sim.execute(&mutations, &mut args).unwrap();
        }
        args[1].fill(0xa5);
        sim.submit(&d.insts, &mut args).unwrap();
        assert_eq!(
            d.check_write(&args[1]),
            WriteCheck {
                bad_words: 0,
                first_bad: None
            },
            "submit {submit}"
        );
    }
}

fn concurrency_cfg(
    read_channels: usize,
    write_channels: usize,
    queued_tasks: usize,
    bd_bytes: usize,
) -> bw_probe::ConcurrencyConfig {
    bw_probe::ConcurrencyConfig {
        read_channels,
        write_channels,
        queued_tasks,
        bd_bytes,
        axi: Default::default(),
    }
}

/// Run the actual PDI/TXN, including final-task-only completion tokens. Exact per-channel counters and
/// independent checks of every DDR task range rule out counting repeat traffic to the same address.
fn run_concurrency(cfg: bw_probe::ConcurrencyConfig) {
    let d = bw_probe::concurrency_design(cfg);
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    let mut args: Vec<Vec<u8>> = d.args.iter().map(|a| vec![0xa5; a.bytes]).collect();
    for (i, b) in args[0].iter_mut().enumerate() {
        *b = (i * 7 + 1) as u8;
    }
    let read_before = args[0].clone();
    for submit in 0..2 {
        args[1].fill(if submit == 0 { 0xa5 } else { 0x5a });
        sim.submit(&d.insts, &mut args)
            .unwrap_or_else(|e| panic!("{cfg:?} submit {submit}: {e}"));
        assert_eq!(args[0], read_before, "{cfg:?} read buffer modified");
        assert_eq!(
            d.check_write(&args[1]),
            WriteCheck {
                bad_words: 0,
                first_bad: None
            },
            "{cfg:?} submit {submit}"
        );
        for active in 0..cfg.write_channels {
            let (col, lane) = cfg.channel_location(active);
            for task in 0..cfg.queued_tasks {
                let range = d.write_segment(active, task);
                assert_eq!(
                    range.start,
                    (active * cfg.queued_tasks + task) * cfg.bd_bytes
                );
                assert_eq!(range.len(), cfg.bd_bytes);
                for (word, bytes) in args[1][range].chunks_exact(4).enumerate() {
                    assert_eq!(
                        u32::from_le_bytes(bytes.try_into().unwrap()),
                        bw_probe::pattern_word(
                            col as u32,
                            lane as u32,
                            task * cfg.bd_bytes / 4 + word
                        ),
                        "{cfg:?} submit {submit} active {active} task {task} word {word}",
                    );
                }
            }
        }
        for c in sim.shim_stats() {
            let count = match c.direction {
                Direction::Mm2s => cfg.read_channels,
                Direction::S2mm => cfg.write_channels,
            };
            let index = c.channel as usize * 8 + c.col as usize;
            let want = if index < count {
                ((submit + 1) * cfg.queued_tasks * cfg.bd_bytes) as u64
            } else {
                0
            };
            assert_eq!(
                c.dram_bytes(),
                want,
                "{cfg:?} col {} {:?} ch {} submit {submit}",
                c.col,
                c.direction,
                c.channel
            );
        }
    }
    // Corruption anywhere in any independent task range must be detected with its actual DDR offset.
    for active in 0..cfg.write_channels {
        let (col, lane) = cfg.channel_location(active);
        for task in 0..cfg.queued_tasks {
            let offset = d.write_segment(active, task).end - 4;
            let saved: [u8; 4] = args[1][offset..offset + 4].try_into().unwrap();
            args[1][offset] ^= 1;
            let want =
                bw_probe::pattern_word(col as u32, lane as u32, (task + 1) * cfg.bd_bytes / 4 - 1);
            assert_eq!(
                d.check_write(&args[1]),
                WriteCheck {
                    bad_words: 1,
                    first_bad: Some((offset, want ^ 1, want))
                }
            );
            args[1][offset..offset + 4].copy_from_slice(&saved);
        }
    }
}

#[test]
fn concurrency_active_channels_are_column_first() {
    for count in [1, 2, 4, 8, 16] {
        let cfg = concurrency_cfg(count, count, 1, 4096);
        let positions: Vec<_> = (0..count).map(|i| cfg.channel_location(i)).collect();
        let expected: Vec<_> = (0..count.min(8))
            .map(|col| (col, 0))
            .chain((0..count.saturating_sub(8)).map(|col| (col, 1)))
            .collect();
        assert_eq!(positions, expected);
    }
}

#[test]
fn concurrency_small_ranges_all_channel_and_queue_shapes() {
    for active in [1, 2, 4, 8, 16] {
        for queued in [1, 2, 4] {
            for (read, write) in [(active, 0), (0, active), (active, active)] {
                run_concurrency(concurrency_cfg(read, write, queued, 4096));
            }
        }
    }
}

#[test]
fn concurrency_stream_wrap_and_large_bd_boundaries() {
    // The first task can end inside the source period; the next must continue, not restart word zero.
    for (queued, bytes) in [
        (2, 4096),
        (4, 8192),
        (1, 16384),
        (2, 32768),
        (4, 262144),
        (4, 1048576),
    ] {
        run_concurrency(concurrency_cfg(1, 1, queued, bytes));
    }
}

#[test]
fn concurrency_mixed_maximum_shim_bd_capacity() {
    // 2 directions * 2 lanes * 4 independent tasks = all 16 shim BDs in each of eight columns.
    run_concurrency(concurrency_cfg(16, 16, 4, 32768));
}

#[test]
fn concurrency_asymmetric_directions_complete_only_active_columns() {
    run_concurrency(concurrency_cfg(16, 4, 2, 8192));
    run_concurrency(concurrency_cfg(1, 16, 4, 4096));
}

#[test]
fn concurrency_rejects_invalid_geometry_and_unsafe_axi() {
    let good = concurrency_cfg(16, 16, 4, 1048576);
    assert!(good.validate().is_ok());
    for queued_tasks in [0, 3, 5, 256] {
        assert!(bw_probe::ConcurrencyConfig {
            queued_tasks,
            ..good
        }
        .validate()
        .is_err());
    }
    for channels in [3, 5, 9, 17] {
        assert!(bw_probe::ConcurrencyConfig {
            read_channels: channels,
            ..good
        }
        .validate()
        .is_err());
        assert!(bw_probe::ConcurrencyConfig {
            write_channels: channels,
            ..good
        }
        .validate()
        .is_err());
    }
    assert!(concurrency_cfg(0, 0, 1, 4096).validate().is_err());
    for bd_bytes in [0, 2048, 12288, 2097152] {
        assert!(bw_probe::ConcurrencyConfig { bd_bytes, ..good }
            .validate()
            .is_err());
    }
    for (cache, qos) in [(0, 0), (3, 0), (2, 1), (2, 15)] {
        assert!(bw_probe::ConcurrencyConfig {
            axi: peacemaker_npu::dma::ShimAxi {
                burst: 3,
                cache,
                qos
            },
            ..good
        }
        .validate()
        .is_err());
    }
    for burst in 0..=3 {
        assert!(bw_probe::ConcurrencyConfig {
            axi: peacemaker_npu::dma::ShimAxi {
                burst,
                cache: 2,
                qos: 0
            },
            ..good
        }
        .validate()
        .is_ok());
    }
}
