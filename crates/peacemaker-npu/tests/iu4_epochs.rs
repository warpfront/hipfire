// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! The epoch-batched V8 deployment ([`peacemaker_npu::kernels::iu4::EpochGemm`]) run on the exact PDI/TXN simulator returns
//! hipfire's IU4 per-K128-epoch int32 partials bit for bit, and the GPU fold of those partials equals the fold of
//! the CPU reference partials bit for bit.
use peacemaker_npu::kernels::gemm_core::Control;
use peacemaker_npu::kernels::iu4::{self, Acts, EpochGemm, Weights};
use peacemaker_npu::sim::config::Config;

fn epoch_shape(tokens: usize, rows: usize, k: usize, ctl: Control, seed: u64) {
    let (wb, xb) = iu4::random_operands(tokens, rows, k, seed, true);
    let (w, x) = (Weights::new(rows, k, &wb), Acts::new(tokens, k, &xb));
    let g = EpochGemm::new(tokens, rows, k, ctl);
    let [a, b] = g.pack_in(&w, &x);
    let mut args = vec![a, b, vec![0xa5; g.design.args[2].bytes]];
    let mut sim = Config::from_pdi(&g.design.pdi).unwrap();
    sim.submit(&g.design.insts, &mut args).unwrap();
    let want = iu4::partials(&w, &x);
    let got = g.unpack_out(&args[2]);
    assert_eq!(got.len(), want.len());
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        let (e, t, r) = (i / (tokens * rows), i / rows % tokens, i % rows);
        panic!(
            "{tokens}x{rows}x{k} {ctl:?}: epoch {e} token {t} feature {r}: got {} want {}",
            got[i], want[i]
        );
    }
    assert_eq!(
        g.design.unpack_out(&args[2]),
        g.padded(&want),
        "padded token rows must be zero partials"
    );
    assert!(
        want.contains(&8192) && want.contains(&-7168),
        "extreme partials exercised"
    );
    let (f_npu, f_cpu) = (iu4::fold_set(&w, &x, &got), iu4::fold_set(&w, &x, &want));
    assert!(f_npu
        .iter()
        .zip(&f_cpu)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    println!(
        "iu4 epochs {tokens}x{rows}x{k} {ctl:?}: {} waves, {} partials exact, fold bit-identical",
        g.design.waves(),
        got.len()
    );
}

/// Two epochs (one MQ4G256V2 group), one wave per epoch.
#[test]
fn iu4_epochs_512_512_256() {
    for ctl in [Control::Fast, Control::Slow] {
        epoch_shape(512, 512, 256, ctl, 0x1u64)
    }
}
/// Token and feature padding (tokens 300 -> 512, features 700 -> 1024), four epochs, 8 waves.
#[test]
fn iu4_epochs_padding_300_700_512() {
    epoch_shape(300, 700, 512, Control::Fast, 0x2u64)
}
/// Two token blocks per epoch (MW = 2 per epoch, B selected per M-wave) and 8 epochs: 32 waves.
#[test]
fn iu4_epochs_1024_1024_1024() {
    epoch_shape(1024, 1024, 1024, Control::Fast, 0x3u64)
}
/// One context, three submits with different operands (the dynamic TXN restarts every ring).
#[test]
fn iu4_epochs_reuse_context() {
    let (tokens, rows, k) = (512, 512, 512);
    let g = EpochGemm::new(tokens, rows, k, Control::Fast);
    let mut sim = Config::from_pdi(&g.design.pdi).unwrap();
    for seed in [11u64, 12, 13] {
        let (wb, xb) = iu4::random_operands(tokens, rows, k, seed, seed == 12);
        let (w, x) = (Weights::new(rows, k, &wb), Acts::new(tokens, k, &xb));
        let [a, b] = g.pack_in(&w, &x);
        let mut args = vec![a, b, vec![0xa5; g.design.args[2].bytes]];
        sim.submit(&g.design.insts, &mut args).unwrap();
        assert_eq!(g.unpack_out(&args[2]), iu4::partials(&w, &x), "seed {seed}");
    }
}
