// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use peacemaker_npu::kernels::{
    gemm_array,
    gemm_core::CoreVariant,
    gemm_i8::{self, cpu_reference},
};
use peacemaker_npu::sim::{config::Config, dma::Direction};
use std::hash::Hasher;
fn matrices(m: usize, n: usize, k: usize, seed: u32) -> (Vec<i8>, Vec<i8>) {
    let mut state = seed;
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
fn signature(bytes: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(bytes);
    h.finish()
}
fn array_shape(m: usize, n: usize, k: usize) {
    variant_shape(gemm_array::Variant::V6, m, n, k)
}
fn variant_shape(v: gemm_array::Variant, m: usize, n: usize, k: usize) {
    core_shape(v, CoreVariant::Fast, m, n, k)
}
/// Runs the generated PDI/TXN of `v` with `core` in the simulator. Fast/Serial must equal the CPU reference; LockOnly is not a GEMM and
/// must instead write the documented `0xC0DE0000 + wave*0x100 + word` pattern into the first 64 words of every physical tile's C frame.
fn core_shape(v: gemm_array::Variant, core: CoreVariant, m: usize, n: usize, k: usize) {
    core_probe_shape(v, core, m, n, k, 0);
}
/// `core_shape` with a `ClockProbe` iteration count; returns the summed simulated core busy cycles over all cores.
fn core_probe_shape(
    v: gemm_array::Variant,
    core: CoreVariant,
    m: usize,
    n: usize,
    k: usize,
    probe_iters: u32,
) -> u64 {
    let lock_only = matches!(core, CoreVariant::LockOnly | CoreVariant::ClockProbe);
    let d = match core {
        CoreVariant::Fast => gemm_array::design_variant(m, n, k, v),
        _ => gemm_array::design_variant_probe(m, n, k, v, core, probe_iters),
    };
    let (a, b) = matrices(m, n, k, 0xc001d00d);
    let [a_host, b_host] = d.pack_in(&a, &b);
    let before = [signature(&a_host), signature(&b_host)];
    let mut args = vec![a_host, b_host, vec![0xa5; d.args[2].bytes]];
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    sim.submit(&d.insts, &mut args).unwrap();
    if !lock_only {
        assert_eq!(
            d.unpack_out(&args[2]),
            cpu_reference(&a, &b, m, n, k),
            "{v} {core:?} {m}x{n}x{k}"
        )
    }
    assert_eq!([signature(&args[0]), signature(&args[1])], before);
    let (rows, cols) = (d.active_rows(), d.active_cols());
    let one_column = v == gemm_array::Variant::V3;
    let mut read = 0;
    let mut written = 0;
    for c in sim.shim_stats() {
        let per_wave = d.waves() * (k / 64);
        // V3 streams the chunk/row interleaved A on shim MM2S0 and B on MM2S1; the others B on MM2S0 and A (col<rows) on MM2S1.
        let expected = match (c.direction, c.channel) {
            (Direction::Mm2s, 0) if one_column => {
                if c.col == 0 {
                    per_wave * rows * gemm_array::A_BYTES
                } else {
                    0
                }
            }
            (Direction::Mm2s, 1) if one_column => {
                if c.col == 0 {
                    per_wave * gemm_array::B_BYTES
                } else {
                    0
                }
            }
            (Direction::Mm2s, 0) => {
                if (c.col as usize) < cols {
                    per_wave * gemm_array::B_BYTES
                } else {
                    0
                }
            }
            (Direction::Mm2s, 1) => {
                if (c.col as usize) < rows {
                    per_wave * gemm_array::A_BYTES
                } else {
                    0
                }
            }
            (Direction::S2mm, 0) => {
                if (c.col as usize) < cols {
                    d.waves() * rows * gemm_array::C_BYTES
                } else {
                    0
                }
            }
            (Direction::S2mm, 1) => 0,
            _ => panic!("unexpected shim channel"),
        };
        assert_eq!(
            c.dram_bytes(),
            expected as u64,
            "{v} shim col {} {:?} ch {}",
            c.col,
            c.direction,
            c.channel
        );
        match c.direction {
            Direction::Mm2s => read += c.dram_bytes(),
            Direction::S2mm => written += c.dram_bytes(),
        }
    }
    assert_eq!(read, d.host_bytes_read());
    assert_eq!(written, d.host_bytes_written());
    // Every active tile delivered every wave (no tile range is still the 0xa5 poison).
    let tiles = d.output_tile_ranges();
    assert_eq!(tiles.len(), rows * cols);
    for ((col, row), ranges) in &tiles {
        assert_eq!(ranges.len(), d.waves());
        for (w, r) in ranges.iter().enumerate() {
            assert!(
                args[2][r.clone()].iter().any(|&x| x != 0xa5),
                "{v} {core:?} tile ({col},{row}) wave {w} not written"
            );
            if lock_only {
                for i in 0..64 {
                    let at = r.start + 4 * i;
                    let got = u32::from_le_bytes(args[2][at..at + 4].try_into().unwrap());
                    assert_eq!(
                        got,
                        0xC0DE0000 + (w as u32) * 0x100 + i as u32,
                        "{v} LockOnly tile ({col},{row}) wave {w} word {i}"
                    )
                }
            }
        }
    }
    let (busy, wait) = sim.core_stats().fold((0u64, 0u64), |(busy, wait), c| {
        (busy + c.busy, wait + c.lock_wait)
    });
    let checked = if lock_only {
        format!(
            "{} LockOnly pattern words (tiles {}x waves {}x 64)",
            rows * cols * d.waves() * 64,
            rows * cols,
            d.waves()
        )
    } else {
        format!("{} CPU-exact words", m * n)
    };
    eprintln!("PASS {} {v} {core:?} {m}x{n}x{k}: {}PDIbytes {}TXNbytes {checked}; {}functional ticks (not hardware cycles); DRAM read={read} write={written}; core busy={busy} lock_wait={wait}",if lock_only {"lock pattern"} else {"exact"},d.pdi.len(),d.insts.len(),sim.ticks);
    busy
}
#[test]
fn clock_probe_core_pattern_and_busy_scale_with_iters() {
    use gemm_array::Variant::V2;
    let (m, n, k) = (128, 64, 64);
    let b0 = core_probe_shape(V2, CoreVariant::ClockProbe, m, n, k, 0);
    let b2 = core_probe_shape(V2, CoreVariant::ClockProbe, m, n, k, 2);
    let d = gemm_array::design_variant_probe(m, n, k, V2, CoreVariant::ClockProbe, 2);
    let per_iter = peacemaker_npu::kernels::gemm_core::clock_probe_cycles_per_iter();
    let expected = (d.active_rows() * d.active_cols() * d.waves()) as u64 * 2 * per_iter;
    let got = b2 - b0;
    // N=0 skips the loop via jz; the taken jz also skips a few NOPs, so allow a small constant slack per core tile.
    let slack = (d.active_rows() * d.active_cols() * d.waves()) as u64 * 16;
    assert!(
        got >= expected.saturating_sub(slack) && got <= expected + slack,
        "busy delta {got} vs expected {expected} (+-{slack}), per_iter={per_iter}"
    );
    eprintln!("ClockProbe busy N=0: {b0} N=2: {b2} delta {got} expected {expected}");
}
#[test]
fn serial_core_exact_v2_v5_k64() {
    use gemm_array::Variant::*;
    for v in [V2, V5] {
        let (m, n) = v.shape();
        core_shape(v, CoreVariant::Serial, m, n, 64)
    }
}
#[test]
fn serial_core_exact_multichunk_multiwave() {
    use gemm_array::Variant::*;
    // V2: 4 waves (2x2) x 3 chunks; V5: 2 waves x 2 chunks, padded M.
    core_shape(V2, CoreVariant::Serial, 256, 128, 192);
    core_shape(V5, CoreVariant::Serial, 300, 1024, 128);
}
#[test]
fn lock_only_core_writes_pattern_in_every_tile_frame() {
    for v in gemm_array::Variant::ALL {
        let (m, n) = v.shape();
        core_shape(v, CoreVariant::LockOnly, m, n, 64)
    }
}
#[test]
fn lock_only_core_multichunk_multiwave_pattern() {
    use gemm_array::Variant::*;
    core_shape(V2, CoreVariant::LockOnly, 256, 128, 192);
    core_shape(V5, CoreVariant::LockOnly, 512, 1024, 128);
}
#[test]
fn fast_slow_ctl_core_exact_v2_v5_k64() {
    use gemm_array::Variant::*;
    for v in [V2, V5] {
        let (m, n) = v.shape();
        core_shape(v, CoreVariant::FastSlowCtl, m, n, 64)
    }
}
#[test]
fn fast_slow_ctl_core_exact_multichunk_multiwave() {
    core_shape(
        gemm_array::Variant::V2,
        CoreVariant::FastSlowCtl,
        256,
        128,
        192,
    );
}
#[test]
fn every_variant_exact_at_its_shape_k64() {
    for v in gemm_array::Variant::ALL {
        let (m, n) = v.shape();
        variant_shape(v, m, n, 64)
    }
}
#[test]
fn reduced_variants_exact_k192_and_k256() {
    use gemm_array::Variant::*;
    for v in [V1, V2, V3, V4] {
        let (m, n) = v.shape();
        for k in [192, 256] {
            variant_shape(v, m, n, k)
        }
    }
}
#[test]
fn full_static_and_split_sync_variants_exact_k192() {
    use gemm_array::Variant::*;
    for v in [V5, V6b] {
        let (m, n) = v.shape();
        variant_shape(v, m, n, 192)
    }
}
#[test]
fn reduced_variants_exact_with_padding_and_waves() {
    use gemm_array::Variant::*;
    for (v, m, n, k) in [
        (V1, 129, 65, 64),
        (V2, 128, 100, 128),
        (V3, 513, 65, 64),
        (V3, 512, 130, 192),
        (V4, 129, 512, 64),
        (V4, 100, 1100, 64),
    ] {
        variant_shape(v, m, n, k)
    }
}
#[test]
fn array_exact_512_512_64() {
    array_shape(512, 512, 64)
}
#[test]
fn array_exact_512_512_256() {
    array_shape(512, 512, 256)
}
#[test]
fn array_exact_512_2560_640() {
    array_shape(512, 2560, 640)
}
#[test]
fn array_exact_512_1280_2560() {
    array_shape(512, 1280, 2560)
}
#[test]
fn array_reuses_context_with_odd_ping_phase() {
    let (m, n, k) = (512, 512, 64);
    let d = gemm_array::design(m, n, k);
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    for seed in [0x12345678, 0x98765432] {
        let (a, b) = matrices(m, n, k, seed);
        let [a_host, b_host] = d.pack_in(&a, &b);
        let mut args = vec![a_host, b_host, vec![0xa5; d.args[2].bytes]];
        sim.submit(&d.insts, &mut args).unwrap();
        assert_eq!(d.unpack_out(&args[2]), cpu_reference(&a, &b, m, n, k));
    }
    eprintln!(
        "PASS same-context odd kc1*waves1 submits: 512x512x64, {}functional ticks total",
        sim.ticks
    );
}
#[test]
fn single_core_full_designs() {
    for size in [128, 256] {
        let d = gemm_i8::design(&[0], size, size, size);
        let (a, b) = matrices(size, size, size, 0xc001d00d);
        let mut args = d.pack_in(&a, &b);
        args.push(vec![0xa5; d.args[1].bytes]);
        let mut sim = Config::from_pdi(&d.pdi).unwrap();
        sim.submit(&d.insts, &mut args).unwrap();
        assert_eq!(
            d.unpack_out(&args[1]),
            cpu_reference(&a, &b, size, size, size)
        );
        eprintln!(
            "PASS exact single-core {size}^3: {}CPU-exact words, {}functional ticks",
            size * size,
            sim.ticks
        );
    }
}

// ---------------------------------------------------------------- V8 (pair A-sharing, int8 SRS epilogue)
use peacemaker_npu::kernels::gemm_core::{Control, Epilogue};
const INT8: Epilogue = Epilogue::Int8 { shift: 12 };
/// Runs the generated V8 PDI/TXN in the whole-array simulator and requires the result to equal the CPU reference
/// (exact GEMM for `I32`; `clamp(floor(c / 2^shift), -128, 127)` for `Int8`), unchanged inputs, the exact shim
/// traffic of both C channels and a written (non-poison) C range for every core tile and wave.
fn v8_shape(epi: Epilogue, ctl: Control, m: usize, n: usize, k: usize) {
    let d = gemm_array::design_v8(m, n, k, epi, ctl);
    let (a, b) = matrices(m, n, k, 0xc001d00d);
    let [a_host, b_host] = d.pack_in(&a, &b);
    let before = [signature(&a_host), signature(&b_host)];
    let mut args = vec![a_host, b_host, vec![0xa5; d.args[2].bytes]];
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    sim.submit(&d.insts, &mut args).unwrap();
    let want = d.reference(&a, &b);
    if epi == Epilogue::I32 {
        assert_eq!(want, cpu_reference(&a, &b, m, n, k))
    }
    let got = d.unpack_out(&args[2]);
    assert_eq!(got.len(), want.len());
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "V8 {epi:?} {ctl:?} {m}x{n}x{k}: first mismatch at ({},{}) got {} want {}",
            i / n,
            i % n,
            got[i],
            want[i]
        )
    }
    assert_eq!([signature(&args[0]), signature(&args[1])], before);
    let cb = if matches!(epi, Epilogue::I32) {
        gemm_array::C_BYTES
    } else {
        gemm_array::COUT_BYTES
    };
    let per_wave = d.waves() * (k / 64);
    let (mut read, mut written) = (0, 0);
    for c in sim.shim_stats() {
        let expected = match (c.direction, c.channel) {
            (Direction::Mm2s, 0) | (Direction::Mm2s, 1) => per_wave * gemm_array::B_BYTES,
            (Direction::S2mm, 0) | (Direction::S2mm, 1) => d.waves() * 2 * cb,
            _ => panic!("unexpected shim channel"),
        };
        assert_eq!(
            c.dram_bytes(),
            expected as u64,
            "V8 shim col {} {:?} ch {}",
            c.col,
            c.direction,
            c.channel
        );
        match c.direction {
            Direction::Mm2s => read += c.dram_bytes(),
            Direction::S2mm => written += c.dram_bytes(),
        }
    }
    assert_eq!(read, d.host_bytes_read());
    assert_eq!(written, d.host_bytes_written());
    let tiles = d.output_tile_ranges();
    assert_eq!(tiles.len(), 32);
    for ((col, row), ranges) in &tiles {
        assert_eq!(ranges.len(), d.waves());
        for (w, r) in ranges.iter().enumerate() {
            assert_eq!(r.len(), cb);
            assert!(
                args[2][r.clone()].iter().any(|&x| x != 0xa5),
                "V8 tile ({col},{row}) wave {w} not written"
            );
        }
    }
    let (busy, wait) = sim.core_stats().fold((0u64, 0u64), |(busy, wait), c| {
        (busy + c.busy, wait + c.lock_wait)
    });
    eprintln!("PASS exact V8 {epi:?} {ctl:?} {m}x{n}x{k}: {} PDI bytes {} TXN bytes {} CPU-exact words; {} functional ticks (not hardware cycles); DRAM read={read} write={written}; core busy={busy} lock_wait={wait}",d.pdi.len(),d.insts.len(),m*n,sim.ticks);
}
/// All four epilogue x control combinations.
fn v8_all(m: usize, n: usize, k: usize) {
    for epi in [Epilogue::I32, INT8] {
        for ctl in [Control::Fast, Control::Slow] {
            v8_shape(epi, ctl, m, n, k)
        }
    }
}
#[test]
fn v8_exact_512_512_64() {
    v8_all(512, 512, 64)
}
#[test]
fn v8_exact_512_512_256() {
    v8_all(512, 512, 256)
}
#[test]
fn v8_exact_padding_and_waves_k192() {
    v8_all(513, 513, 192)
}
#[test]
fn v8_int8_shift_sweep_saturates_and_floors() {
    for shift in [0, 6, 20, 31] {
        v8_shape(Epilogue::Int8 { shift }, Control::Fast, 512, 512, 64)
    }
}
#[test]
fn v8_reuses_context_with_odd_and_even_ping_phase() {
    for (epi, k) in [(Epilogue::I32, 64), (INT8, 64), (INT8, 128)] {
        let (m, n) = (512, 512);
        let d = gemm_array::design_v8(m, n, k, epi, Control::Fast);
        let mut sim = Config::from_pdi(&d.pdi).unwrap();
        for seed in [0x12345678, 0x98765432, 0x2468ace1] {
            let (a, b) = matrices(m, n, k, seed);
            let [a_host, b_host] = d.pack_in(&a, &b);
            let mut args = vec![a_host, b_host, vec![0xa5; d.args[2].bytes]];
            sim.submit(&d.insts, &mut args).unwrap();
            assert_eq!(
                d.unpack_out(&args[2]),
                d.reference(&a, &b),
                "V8 {epi:?} k{k} seed {seed:#x}"
            );
        }
        eprintln!("PASS V8 same-context {epi:?} 512x512x{k} (kc*waves={}) three submits, {} functional ticks total",k/64,sim.ticks);
    }
}
#[test]
fn v8_exact_512_2560_640_int8_fast() {
    v8_shape(INT8, Control::Fast, 512, 2560, 640)
}
#[test]
fn v8_exact_512_2560_640_int8_slow() {
    v8_shape(INT8, Control::Slow, 512, 2560, 640)
}
#[test]
fn v8_exact_512_2560_640_i32_fast() {
    v8_shape(Epilogue::I32, Control::Fast, 512, 2560, 640)
}
#[test]
fn v8_exact_512_2560_640_i32_slow() {
    v8_shape(Epilogue::I32, Control::Slow, 512, 2560, 640)
}
#[test]
fn v8_exact_512_1280_2560_int8_fast() {
    v8_shape(INT8, Control::Fast, 512, 1280, 2560)
}
#[test]
fn v8_exact_512_1280_2560_int8_slow() {
    v8_shape(INT8, Control::Slow, 512, 1280, 2560)
}
#[test]
fn v8_exact_512_1280_2560_i32_fast() {
    v8_shape(Epilogue::I32, Control::Fast, 512, 1280, 2560)
}
#[test]
fn v8_exact_512_1280_2560_i32_slow() {
    v8_shape(Epilogue::I32, Control::Slow, 512, 1280, 2560)
}
#[test]
fn v8_exact_1024_1280_2560_int8_fast() {
    v8_shape(INT8, Control::Fast, 1024, 1280, 2560)
}
#[test]
fn v8_exact_1024_1280_2560_i32_slow() {
    v8_shape(Epilogue::I32, Control::Slow, 1024, 1280, 2560)
}
#[test]
fn v8_exact_1024_1280_2560_int8_slow() {
    v8_shape(INT8, Control::Slow, 1024, 1280, 2560)
}
#[test]
fn v8_exact_1024_1280_2560_i32_fast() {
    v8_shape(Epilogue::I32, Control::Fast, 1024, 1280, 2560)
}
/// K beyond the old 2560 limit (kc = 80): chunk counter, host layouts and shim BD lengths.
#[test]
fn v8_exact_512_512_5120() {
    for ctl in [Control::Fast, Control::Slow] {
        v8_shape(INT8, ctl, 512, 512, 5120)
    }
}

// ---------------------------------------------------------------- V9 (resident B, nw-outer, int8 SRS epilogue)
/// Runs the generated V9 PDI/TXN in the whole-array simulator (`Epilogue::Int8{shift:12}` only) and requires the result to
/// equal the CPU reference `clamp(floor(c / 2^12), -128, 127)`, unchanged inputs, the exact per-channel shim traffic
/// (MM2S0 = resident B: `NW*kc*4096` B, loaded once per `nw`; MM2S1 = A: `waves*kc*4096` B; S2MM0/1 = `waves*2*COUT_BYTES`).
fn v9_shape(ctl: Control, m: usize, n: usize, k: usize) {
    let epi = INT8;
    let d = gemm_array::design_v9(m, n, k, epi, ctl);
    let (a, b) = matrices(m, n, k, 0xc001d00d);
    let [a_host, b_host] = d.pack_in(&a, &b);
    let before = [signature(&a_host), signature(&b_host)];
    let (kc, nw) = (k / 64, n.div_ceil(512));
    let waves = m.div_ceil(512) * nw;
    assert_eq!(d.waves(), waves);
    let mut args = vec![a_host, b_host, vec![0xa5; d.args[2].bytes]];
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    sim.submit(&d.insts, &mut args).unwrap();
    let want = d.reference(&a, &b);
    let got = d.unpack_out(&args[2]);
    assert_eq!(got.len(), want.len());
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "V9 {epi:?} {ctl:?} {m}x{n}x{k}: first mismatch at ({},{}) got {} want {}",
            i / n,
            i % n,
            got[i],
            want[i]
        )
    }
    assert_eq!([signature(&args[0]), signature(&args[1])], before);
    let cb = gemm_array::COUT_BYTES;
    let (mut read, mut written) = (0, 0);
    for c in sim.shim_stats() {
        let expected = match (c.direction, c.channel) {
            (Direction::Mm2s, 0) => nw * kc * 4096,
            (Direction::Mm2s, 1) => waves * kc * 4096,
            (Direction::S2mm, 0) | (Direction::S2mm, 1) => waves * 2 * cb,
            _ => panic!("unexpected shim channel"),
        };
        assert_eq!(
            c.dram_bytes(),
            expected as u64,
            "V9 shim col {} {:?} ch {}",
            c.col,
            c.direction,
            c.channel
        );
        match c.direction {
            Direction::Mm2s => read += c.dram_bytes(),
            Direction::S2mm => written += c.dram_bytes(),
        }
    }
    assert_eq!(read, d.host_bytes_read());
    assert_eq!(written, d.host_bytes_written());
    let (busy, wait) = sim.core_stats().fold((0u64, 0u64), |(busy, wait), c| {
        (busy + c.busy, wait + c.lock_wait)
    });
    eprintln!("PASS exact V9 {epi:?} {ctl:?} {m}x{n}x{k}: {} PDI bytes {} TXN bytes {} CPU-exact words; {} functional ticks (not hardware cycles); DRAM read={read} write={written}; core busy={busy} lock_wait={wait}",d.pdi.len(),d.insts.len(),m*n,sim.ticks);
}
/// Derived V9 K limit: kc = 54 (K = 3456); K = 3520 (kc = 55) is rejected.
#[test]
fn v9_exact_512_512_3456() {
    v9_all(512, 512, 3456)
}
#[test]
#[should_panic]
fn v9_rejects_i32_epilogue() {
    gemm_array::design_v9(512, 512, 64, Epilogue::I32, Control::Fast);
}
#[test]
#[should_panic]
fn v9_rejects_k_above_derived_limit() {
    gemm_array::design_v9(512, 512, 3520, INT8, Control::Fast);
}
/// Both control disciplines.
fn v9_all(m: usize, n: usize, k: usize) {
    for ctl in [Control::Fast, Control::Slow] {
        v9_shape(ctl, m, n, k)
    }
}
#[test]
fn v9_exact_512_512_64() {
    v9_all(512, 512, 64)
}
/// Non-default shim AXI attributes keep V9 CPU-exact through the simulator, in the same context across repeated
/// setter transitions (non-default, back to default, another non-default), and invalid attributes are rejected
/// without modifying the command stream.
#[test]
fn v9_shim_axi_transitions_stay_cpu_exact_and_invalid_is_atomic() {
    use peacemaker_npu::dma::ShimAxi;
    let (m, n, k) = (512, 512, 64);
    let mut d = gemm_array::design_v9(m, n, k, INT8, Control::Fast);
    let odd = ShimAxi {
        burst: 1,
        cache: 2,
        qos: 0,
    };
    let snapshot = d.insts.clone();
    for bad in [
        ShimAxi { burst: 4, ..odd },
        ShimAxi { cache: 16, ..odd },
        ShimAxi { qos: 16, ..odd },
        ShimAxi { cache: 3, ..odd },
        ShimAxi { qos: 15, ..odd },
    ] {
        assert!(d.set_shim_axi(bad).is_err());
        assert_eq!(d.insts, snapshot, "rejected {bad:?} modified the TXN");
    }
    let (a, b) = matrices(m, n, k, 0xc001d00d);
    let want = d.reference(&a, &b);
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    for axi in [
        odd,
        ShimAxi::default(),
        ShimAxi {
            burst: 0,
            cache: 2,
            qos: 0,
        },
        odd,
    ] {
        d.set_shim_axi(axi).unwrap();
        let [a_host, b_host] = d.pack_in(&a, &b);
        let mut args = vec![a_host, b_host, vec![0xa5; d.args[2].bytes]];
        sim.submit(&d.insts, &mut args).unwrap();
        assert_eq!(d.unpack_out(&args[2]), want, "V9 {axi:?}");
    }
}
#[test]
fn v9_exact_1024_1024_128() {
    v9_all(1024, 1024, 128)
}
#[test]
fn v9_exact_1536_1024_192() {
    v9_all(1536, 1024, 192)
}
#[test]
fn v9_exact_1024_1280_2560() {
    v9_all(1024, 1280, 2560)
}
#[test]
fn v9_exact_2048_2560_640() {
    v9_all(2048, 2560, 640)
}
#[test]
fn v9_exact_4096_1280_2560() {
    v9_all(4096, 1280, 2560)
}
#[test]
fn v9_exact_4096_512_128() {
    v9_all(4096, 512, 128)
}
/// Boundary wave grids: MW=16 (NW=1) and NW=8 (MW=1).
#[test]
fn v9_exact_8192_512_64() {
    v9_all(8192, 512, 64)
}
#[test]
fn v9_exact_512_4096_64() {
    v9_all(512, 4096, 64)
}
/// Flash-Next routed experts at the average row count M = 160 (padded into one 512-row M-wave): gate_up and down.
#[test]
fn v9_exact_expert_rows_m160() {
    v9_all(160, 1280, 2560);
    v9_all(160, 2560, 640)
}
#[test]
fn v10_exact_expert_down_m160() {
    v10_all(160, 2560, 640)
}
#[test]
fn v9_reuses_context_with_odd_and_even_kc_times_waves() {
    // (m,n,k): kc*waves = 1 (odd), 2 (even), 3 (odd), 8 (even), 9 (odd), 12 (even), 9 (odd, NW=3).
    for (m, n, k) in [
        (512, 512, 64),
        (512, 512, 128),
        (1536, 512, 64),
        (1024, 1024, 128),
        (1536, 512, 192),
        (1024, 1536, 128),
        (1536, 1536, 64),
    ] {
        for ctl in [Control::Fast, Control::Slow] {
            let d = gemm_array::design_v9(m, n, k, INT8, ctl);
            let mut sim = Config::from_pdi(&d.pdi).unwrap();
            for seed in [0x12345678, 0x98765432, 0x2468ace1] {
                let (a, b) = matrices(m, n, k, seed);
                let [a_host, b_host] = d.pack_in(&a, &b);
                let mut args = vec![a_host, b_host, vec![0xa5; d.args[2].bytes]];
                sim.submit(&d.insts, &mut args).unwrap();
                assert_eq!(
                    d.unpack_out(&args[2]),
                    d.reference(&a, &b),
                    "V9 {ctl:?} {m}x{n}x{k} seed {seed:#x}"
                );
            }
            eprintln!("PASS V9 same-context {ctl:?} {m}x{n}x{k} (kc*waves={}) three submits, {} functional ticks total",k/64*d.waves(),sim.ticks);
        }
    }
}
/// Whole-array V8 under every pair-core data-memory layout: both controls and both epilogues stay CPU-exact.
#[test]
fn v8_exact_512_512_128_all_layouts() {
    for layout in 0..=3u8 {
        for serial in [false, true] {
            for epi in [Epilogue::I32, INT8] {
                v8_shape(
                    epi,
                    Control::Probe(peacemaker_npu::kernels::gemm_core::Probe {
                        layout,
                        serial,
                        ..Default::default()
                    }),
                    512,
                    512,
                    128,
                )
            }
        }
    }
}
/// V9 under every pair-core data-memory layout (both controls).
#[test]
fn v9_exact_1024_1024_128_all_layouts() {
    for layout in 0..=3u8 {
        for serial in [false, true] {
            v9_shape(
                Control::Probe(peacemaker_npu::kernels::gemm_core::Probe {
                    layout,
                    serial,
                    ..Default::default()
                }),
                1024,
                1024,
                128,
            )
        }
    }
}

// ---------------------------------------------------------------- V10 (resident A and B, mw-outer, int8 SRS epilogue)
/// Runs the generated V10 PDI/TXN in the whole-array simulator (`Epilogue::Int8{shift:12}` only) and requires the result to
/// equal the CPU reference `clamp(floor(c / 2^12), -128, 127)`, unchanged inputs and the exact per-channel shim traffic
/// (MM2S0 = resident B: `NW*kc*4096` B; MM2S1 = resident A: `MW*kc*4096` B; S2MM0/1 = `waves*2*COUT_BYTES`, unchanged from V9).
fn v10_shape(ctl: Control, m: usize, n: usize, k: usize) {
    let epi = INT8;
    let d = gemm_array::design_v10(m, n, k, epi, ctl);
    let (a, b) = matrices(m, n, k, 0xc001d00d);
    let [a_host, b_host] = d.pack_in(&a, &b);
    let before = [signature(&a_host), signature(&b_host)];
    let (kc, mw, nw) = (k / 64, m.div_ceil(512), n.div_ceil(512));
    let waves = mw * nw;
    assert_eq!(d.waves(), waves);
    let mut args = vec![a_host, b_host, vec![0xa5; d.args[2].bytes]];
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    sim.submit(&d.insts, &mut args).unwrap();
    let want = d.reference(&a, &b);
    let got = d.unpack_out(&args[2]);
    assert_eq!(got.len(), want.len());
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "V10 {epi:?} {ctl:?} {m}x{n}x{k}: first mismatch at ({},{}) got {} want {}",
            i / n,
            i % n,
            got[i],
            want[i]
        )
    }
    assert_eq!([signature(&args[0]), signature(&args[1])], before);
    let cb = gemm_array::COUT_BYTES;
    let (mut read, mut written) = (0, 0);
    for c in sim.shim_stats() {
        let expected = match (c.direction, c.channel) {
            (Direction::Mm2s, 0) => nw * kc * 4096,
            (Direction::Mm2s, 1) => mw * kc * 4096,
            (Direction::S2mm, 0) | (Direction::S2mm, 1) => waves * 2 * cb,
            _ => panic!("unexpected shim channel"),
        };
        assert_eq!(
            c.dram_bytes(),
            expected as u64,
            "V10 shim col {} {:?} ch {}",
            c.col,
            c.direction,
            c.channel
        );
        match c.direction {
            Direction::Mm2s => read += c.dram_bytes(),
            Direction::S2mm => written += c.dram_bytes(),
        }
    }
    assert_eq!(read, d.host_bytes_read());
    assert_eq!(written, d.host_bytes_written());
    let (busy, wait) = sim.core_stats().fold((0u64, 0u64), |(busy, wait), c| {
        (busy + c.busy, wait + c.lock_wait)
    });
    eprintln!("PASS exact V10 {epi:?} {ctl:?} {m}x{n}x{k}: {} PDI bytes {} TXN bytes {} CPU-exact words; {} functional ticks (not hardware cycles); DRAM read={read} write={written}; core busy={busy} lock_wait={wait}",d.pdi.len(),d.insts.len(),m*n,sim.ticks);
}
/// Both control disciplines.
fn v10_all(m: usize, n: usize, k: usize) {
    for ctl in [Control::Fast, Control::Slow] {
        v10_shape(ctl, m, n, k)
    }
}
#[test]
fn v10_exact_512_512_64() {
    v10_all(512, 512, 64)
}
#[test]
fn v10_exact_1024_1024_128() {
    v10_all(1024, 1024, 128)
}
#[test]
fn v10_exact_1536_1024_192() {
    v10_all(1536, 1024, 192)
}
#[test]
fn v10_exact_2048_2560_640() {
    v10_all(2048, 2560, 640)
}
#[test]
fn v10_exact_4096_2560_640() {
    v10_all(4096, 2560, 640)
}
/// Boundary wave grid NW=8 (MW=2, kc=1).
#[test]
fn v10_exact_1024_4096_64() {
    v10_all(1024, 4096, 64)
}
#[test]
fn v10_reuses_context_with_odd_and_even_kc_times_waves() {
    // (m,n,k): kc*waves = 1 (odd), 2 (even), 3 (odd), 8 (even), 9 (odd), 12 (even), 9 (odd, NW=3).
    for (m, n, k) in [
        (512, 512, 64),
        (512, 512, 128),
        (1536, 512, 64),
        (1024, 1024, 128),
        (1536, 512, 192),
        (1024, 1536, 128),
        (1536, 1536, 64),
    ] {
        for ctl in [Control::Fast, Control::Slow] {
            let d = gemm_array::design_v10(m, n, k, INT8, ctl);
            let mut sim = Config::from_pdi(&d.pdi).unwrap();
            for seed in [0x12345678, 0x98765432, 0x2468ace1] {
                let (a, b) = matrices(m, n, k, seed);
                let [a_host, b_host] = d.pack_in(&a, &b);
                let mut args = vec![a_host, b_host, vec![0xa5; d.args[2].bytes]];
                sim.submit(&d.insts, &mut args).unwrap();
                assert_eq!(
                    d.unpack_out(&args[2]),
                    d.reference(&a, &b),
                    "V10 {ctl:?} {m}x{n}x{k} seed {seed:#x}"
                );
            }
            eprintln!("PASS V10 same-context {ctl:?} {m}x{n}x{k} (kc*waves={}) three submits, {} functional ticks total",k/64*d.waves(),sim.ticks);
        }
    }
}
#[test]
#[should_panic]
fn v10_rejects_i32_epilogue() {
    gemm_array::design_v10(512, 512, 64, Epilogue::I32, Control::Fast);
}
/// gate_up 4096x1280x2560: kc = 40, NW = 3 -> kc*(2+NW) = 200 > 112 resident words budget.
#[test]
#[should_panic]
fn v10_rejects_oversize_gate_up_shape() {
    gemm_array::design_v10(4096, 1280, 2560, INT8, Control::Fast);
}
