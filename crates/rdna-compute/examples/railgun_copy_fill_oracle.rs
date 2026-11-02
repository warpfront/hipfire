// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Byte oracle for the railgun D2D primitives the DFlash window uses
// (Design A, slice A3): `railgun_copy` and the typed `railgun_fill_u32`.
// Every destination is a poisoned allocation with 256-byte canary margins on
// both sides; the checks are byte equality over the WHOLE allocation. Copy
// also verifies the source is unchanged. Exit status is non-zero on any
// mismatch.
//
// Run (gfx1201):
//   cargo run --release -p rdna-compute --example railgun_copy_fill_oracle

use railgun::copy;

const MARGIN: usize = 256;
const POISON: u8 = 0xA5;

fn pattern(n: usize, salt: u32) -> Vec<u8> {
    let mut x = 0x9E37_79B9u32 ^ salt.wrapping_mul(0x85EB_CA6B);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x >> 8) as u8
        })
        .collect()
}

fn pad16(mut v: Vec<u8>) -> Vec<u8> {
    while v.len() % 16 != 0 {
        v.push(0);
    }
    v
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    gpu.ensure_kernel_public(copy::MODULE, copy::SOURCE, copy::COPY_SYMBOL)
        .expect("copy kernel");
    gpu.ensure_kernel_public(copy::FILL_MODULE, copy::FILL_SOURCE, copy::FILL_SYMBOL)
        .expect("fill kernel");
    let mut failures = 0usize;
    let mut cases = 0usize;

    // ---- copy: sizes x (src offset, dst offset) misalignments -------------
    let sizes = [
        1usize, 2, 15, 16, 17, 31, 32, 33, 255, 256, 257, 511, 8192, 20480, 20481, 40000, 262_144,
        262_145, 1_000_003,
    ];
    let offs = [
        (0usize, 0usize),
        (1, 0),
        (0, 3),
        (5, 11),
        (16, 16),
        (7, 7),
        (15, 1),
    ];
    for &n in &sizes {
        for &(so, dof) in &offs {
            let src_total = MARGIN + so + n + MARGIN;
            let dst_total = MARGIN + dof + n + MARGIN;
            let src_host = pattern(src_total, (n * 31 + so) as u32);
            let src = gpu.hip.malloc(src_total).expect("malloc src");
            let dst = gpu.hip.malloc(dst_total).expect("malloc dst");
            gpu.hip.memcpy_htod(&src, &src_host).expect("up src");
            gpu.hip
                .memcpy_htod(&dst, &vec![POISON; dst_total])
                .expect("up dst");
            let l = copy::copy_launch(
                src.as_ptr() as u64 + (MARGIN + so) as u64,
                dst.as_ptr() as u64 + (MARGIN + dof) as u64,
                n as u32,
            );
            let mut ka = pad16(l.kernarg.clone());
            gpu.launch_kernel_blob(l.symbol, l.grid, l.block, 0, &mut ka)
                .expect("launch copy");
            gpu.hip.device_synchronize().expect("sync");
            let mut got = vec![0u8; dst_total];
            gpu.hip.memcpy_dtoh(&mut got, &dst).expect("dl dst");
            let mut src_back = vec![0u8; src_total];
            gpu.hip.memcpy_dtoh(&mut src_back, &src).expect("dl src");
            let mut want = vec![POISON; dst_total];
            want[MARGIN + dof..MARGIN + dof + n]
                .copy_from_slice(&src_host[MARGIN + so..MARGIN + so + n]);
            cases += 1;
            if got != want || src_back != src_host {
                failures += 1;
                let first = got.iter().zip(&want).position(|(a, b)| a != b);
                eprintln!(
                    "COPY FAIL n={n} so={so} dof={dof} first_diff={first:?} src_changed={}",
                    src_back != src_host
                );
            }
            gpu.hip.free(src).ok();
            gpu.hip.free(dst).ok();
        }
    }

    // ---- fill: words x byte offsets x values ------------------------------
    let words_set = [1usize, 2, 3, 4, 63, 64, 65, 1000, 4097, 65_537, 300_001];
    let dst_offsets = [0usize, 4, 12, 28];
    let values = [0u32, 0xFFFF_FFFF, 0xDEAD_BEEF, 0x8000_0000, 1];
    for &w in &words_set {
        for &dof in &dst_offsets {
            for &v in &values {
                let total = MARGIN + dof + w * 4 + MARGIN;
                let dst = gpu.hip.malloc(total).expect("malloc dst");
                gpu.hip
                    .memcpy_htod(&dst, &vec![POISON; total])
                    .expect("up dst");
                // The extent the host checked: the launch writes inside it.
                copy::check_fill((MARGIN + dof) as u64, w as u64, (total - MARGIN) as u64)
                    .expect("in-bounds fill rejected");
                let l = copy::fill_launch(dst.as_ptr() as u64 + (MARGIN + dof) as u64, v, w as u32);
                let mut ka = pad16(l.kernarg.clone());
                gpu.launch_kernel_blob(l.symbol, l.grid, l.block, 0, &mut ka)
                    .expect("launch fill");
                gpu.hip.device_synchronize().expect("sync");
                let mut got = vec![0u8; total];
                gpu.hip.memcpy_dtoh(&mut got, &dst).expect("dl");
                let mut want = vec![POISON; total];
                for i in 0..w {
                    let o = MARGIN + dof + i * 4;
                    want[o..o + 4].copy_from_slice(&v.to_le_bytes());
                }
                cases += 1;
                if got != want {
                    failures += 1;
                    let first = got.iter().zip(&want).position(|(a, b)| a != b);
                    eprintln!("FILL FAIL words={w} dof={dof} v={v:#x} first_diff={first:?}");
                }
                gpu.hip.free(dst).ok();
            }
        }
    }

    // ---- host bound checks refuse out-of-extent fills ----------------------
    let refusals = [
        copy::check_fill(0, 0, 64).is_err(),
        copy::check_fill(0, 17, 64).is_err(),
        copy::check_fill(2, 1, 64).is_err(),
        copy::check_fill(60, 2, 64).is_err(),
        copy::check_fill(u64::MAX - 3, 1, u64::MAX).is_err(),
        copy::check_fill(0, copy::MAX_FILL_WORDS + 1, u64::MAX).is_err(),
    ];
    for (i, ok) in refusals.iter().enumerate() {
        cases += 1;
        if !ok {
            failures += 1;
            eprintln!("BOUNDS FAIL: refusal case {i} accepted");
        }
    }

    println!("railgun_copy_fill_oracle: {cases} cases, {failures} failures");
    if failures != 0 {
        std::process::exit(1);
    }
}
