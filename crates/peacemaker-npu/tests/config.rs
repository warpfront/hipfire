// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use peacemaker_npu::sim::config::{parse_txn, Config, TxnOp};
use peacemaker_npu::{
    cdo::Cmd,
    kernels::gemm_i8::{cpu_reference, design},
    regs,
};
fn matrices(m: usize, n: usize, k: usize) -> (Vec<i8>, Vec<i8>) {
    let mut state = 0xc001d00du32;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as i8
    };
    let mut a: Vec<_> = (0..m * k).map(|_| random()).collect();
    let mut b: Vec<_> = (0..k * n).map(|_| random()).collect();
    a[0] = -128;
    b[0] = -128;
    (a, b)
}
#[test]
fn npu_hello_exact_pdi_insts_64_words() {
    let Ok(root) = std::env::var("NPU_HELLO_ARTIFACT_DIR") else {
        eprintln!("skip: set NPU_HELLO_ARTIFACT_DIR from npu-hello dump <dir>");
        return;
    };
    let root = std::path::Path::new(&root);
    let pdi = std::fs::read(root.join("hello.pdi")).unwrap();
    let insts = std::fs::read(root.join("hello.insts")).unwrap();
    let mut sim = Config::from_pdi(&pdi).unwrap();
    let mut args = vec![vec![0xa5; 256]];
    sim.submit(&insts, &mut args).unwrap();
    let expected: Vec<u8> = (0..64)
        .flat_map(|i| (0xc0de0000u32 + i).to_le_bytes())
        .collect();
    assert_eq!(args[0], expected);
    eprintln!(
        "PASS exact hello PDI {} B + TXN {} B: all64hostwords, {}ticks",
        pdi.len(),
        insts.len(),
        sim.ticks
    );
}
#[test]
fn gemm_design_exact_pdi_insts_cpu_memory() {
    for (col, m, n, k) in [(0, 64, 64, 64), (7, 8, 24, 256), (3, 16, 16, 128)] {
        let d = design(&[col], m, n, k);
        let (a, b) = matrices(m, n, k);
        let mut args = d.pack_in(&a, &b);
        let input = args[0].clone();
        args.push(vec![0xa5; d.args[1].bytes]);
        let mut sim = Config::from_pdi(&d.pdi).unwrap();
        sim.submit(&d.insts, &mut args).unwrap();
        assert_eq!(d.unpack_out(&args[1]), cpu_reference(&a, &b, m, n, k));
        assert_eq!(args[0], input);
        eprintln!(
            "PASS exact GEMM col{col} {m}x{n}x{k}: {}PDIbytes {}TXNbytes {}hostwords {}ticks",
            d.pdi.len(),
            d.insts.len(),
            m * n,
            sim.ticks
        );
    }
}
#[test]
fn missing_token_route_moves_data_but_sync_times_out() {
    let d = design(&[0], 8, 8, 64);
    let (a, b) = matrices(8, 8, 64);
    let mut args = d.pack_in(&a, &b);
    args.push(vec![0xa5; 256]);
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    sim.tick_limit = 20000;
    sim.load_cdo(&[Cmd::Write(regs::shim::SS_MASTER_SOUTH0, 0)])
        .unwrap();
    assert!(sim
        .submit(&d.insts, &mut args)
        .unwrap_err()
        .contains("SYNC timed out"));
    assert_eq!(d.unpack_out(&args[1]), cpu_reference(&a, &b, 8, 8, 64));
}
#[test]
fn disabled_input_slave_obeys_backpressure() {
    let d = design(&[0], 8, 8, 64);
    let (a, b) = matrices(8, 8, 64);
    let mut args = d.pack_in(&a, &b);
    args.push(vec![0xa5; 256]);
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    sim.tick_limit = 10000;
    // Disable memtile South0 slave; a configured master alone cannot carry input.
    sim.load_cdo(&[Cmd::MaskWrite(
        regs::tile(0, 1, regs::mem::SS_SLAVE_BASE + 7 * 4),
        1 << 31,
        0,
    )])
    .unwrap();
    assert!(sim
        .submit(&d.insts, &mut args)
        .unwrap_err()
        .contains("SYNC timed out"));
    assert_eq!(args[1], vec![0xa5; 256]);
}
#[test]
fn ddr_patch_plus_preserves_host_guards() {
    let d = design(&[0], 8, 8, 64);
    let (a, b) = matrices(8, 8, 64);
    let packed = d.pack_in(&a, &b).remove(0);
    let mut input = vec![0xcc; 16];
    input.extend(&packed);
    input.extend([0xcc; 16]);
    let input_before = input.clone();
    let mut args = vec![input, vec![0xee; 288]];
    let mut ops = parse_txn(&d.insts).unwrap();
    for op in &mut ops {
        if let TxnOp::DdrPatch { plus, .. } = op {
            *plus = 16
        }
    }
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    sim.execute(&ops, &mut args).unwrap();
    assert_eq!(
        d.unpack_out(&args[1][16..272]),
        cpu_reference(&a, &b, 8, 8, 64)
    );
    assert_eq!(&args[1][..16], [0xee; 16]);
    assert_eq!(&args[1][272..], [0xee; 16]);
    assert_eq!(args[0], input_before);
}
#[test]
fn malformed_txn_and_invalid_binding_return_errors() {
    let d = design(&[0], 8, 8, 64);
    for cut in 0..d.insts.len() {
        assert!(parse_txn(&d.insts[..cut]).is_err());
    }
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    let mut args = vec![vec![0; 4]];
    assert!(sim
        .execute(
            &[TxnOp::DdrPatch {
                address: regs::shim::DMA_BD0 + 4,
                arg: 2,
                plus: 0
            }],
            &mut args
        )
        .is_err());
    assert!(sim
        .execute(
            &[TxnOp::DdrPatch {
                address: regs::shim::DMA_BD0 + 4,
                arg: 0,
                plus: 2
            }],
            &mut args
        )
        .is_err());
    assert!(sim
        .execute(
            &[TxnOp::DdrPatch {
                address: regs::shim::DMA_BD0 + 4,
                arg: 0,
                plus: 8
            }],
            &mut args
        )
        .is_err());
}
#[test]
fn sync_completion_at_exact_tick_limit() {
    let d = design(&[0], 8, 8, 64);
    let (a, b) = matrices(8, 8, 64);
    let host = || {
        let mut args = d.pack_in(&a, &b);
        args.push(vec![0xa5; 256]);
        args
    };
    let mut baseline = Config::from_pdi(&d.pdi).unwrap();
    baseline.submit(&d.insts, &mut host()).unwrap();
    let mut exact = Config::from_pdi(&d.pdi).unwrap();
    exact.tick_limit = baseline.ticks;
    let mut args = host();
    exact.submit(&d.insts, &mut args).unwrap();
    assert_eq!(d.unpack_out(&args[1]), cpu_reference(&a, &b, 8, 8, 64));
    let mut short = Config::from_pdi(&d.pdi).unwrap();
    short.tick_limit = baseline.ticks - 1;
    assert!(short
        .submit(&d.insts, &mut host())
        .unwrap_err()
        .contains("SYNC timed out"));
}
#[test]
fn malformed_cdo_payloads_return_errors() {
    for body in [
        vec![(255 << 16) | 0x105],
        vec![0x103],
        vec![(1 << 16) | 0x102, 0],
        vec![(2 << 16) | 0x105, 0],
    ] {
        let cdo = peacemaker_npu::cdo::Cdo::from_body(body);
        assert!(Config::from_pdi(&peacemaker_npu::pdi::build(&cdo.to_words())).is_err());
    }
}
