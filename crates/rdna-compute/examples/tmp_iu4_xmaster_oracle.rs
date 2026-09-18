// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MQ4E8 slice-B oracle: standalone `quantize_int4_mmq_ds128` (+ optional
//! `HIPFIRE_IU4_XMASTER` reduce/fixup) device dump, host verifier, and quant
//! TIME throwaway.
//!
//! Modes:
//!   dump <prefix> [--k K] [--n N]  — upload deterministic X, run
//!       `ensure_int4_mmq_x`, write `<prefix>.x.bin` (f32 LE) + `<prefix>.y.bin`
//!       (raw 72 B blocks). Flag on/off comes from the process env.
//!   verify <a_prefix> <b_prefix>    — gate 2 oracle: every block of B has
//!       stored d == master*2^e bitwise (e in 0..2), q/s consistent with the
//!       stored d, and e matches a CPU half-SSE argmin replica. Exit 1 on
//!       any gate failure.
//!   time [--k K]                    — HIPFIRE wall TIME around
//!       `ensure_int4_mmq_x` at N=512 (sync-delimited median over 20 calls).
//!
//! Gfx1201 only; other arches SKIP cleanly (exit 0).

use rdna_compute::{DType, Gpu};

const BLOCK: usize = 128;
const BLOCK_BYTES: usize = 72;

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// Deterministic X [n][k]: token 0 all-zero; 1 constant; 2 single spike;
/// 3 tiny uniform; 4 prng ±1; 5 prng ±32; 6 alternating ±0.25/±8 blocks;
/// 7 prng ±1 with the last block zeroed. Higher tokens repeat the t4 shape
/// with a shifted salt.
fn build_x(n: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % BLOCK, 0);
    let nb = k / BLOCK;
    let mut x = vec![0.0f32; n * k];
    for t in 0..n {
        let row = &mut x[t * k..(t + 1) * k];
        match t {
            0 => {}
            1 => {
                for v in row.iter_mut() {
                    *v = 0.5;
                }
            }
            2 => {
                row[0] = 3.0;
            }
            3 => {
                for (i, v) in row.iter_mut().enumerate() {
                    *v = (prng(i, 0x3000_0003) * 2.0 - 1.0) * 1e-3;
                }
            }
            4 => {
                for (i, v) in row.iter_mut().enumerate() {
                    *v = prng(i, 0x4000_0004) * 2.0 - 1.0;
                }
            }
            5 => {
                for (i, v) in row.iter_mut().enumerate() {
                    *v = (prng(i, 0x5000_0005) * 2.0 - 1.0) * 32.0;
                }
            }
            6 => {
                for b in 0..nb {
                    let s = if b % 2 == 0 { 0.25 } else { 8.0 };
                    for i in 0..BLOCK {
                        let j = b * BLOCK + i;
                        row[j] = (prng(j, 0x6000_0006) * 2.0 - 1.0) * s;
                    }
                }
            }
            7 => {
                for (i, v) in row.iter_mut().enumerate() {
                    *v = prng(i, 0x7000_0007) * 2.0 - 1.0;
                }
                for v in row[(nb - 1) * BLOCK..].iter_mut() {
                    *v = 0.0;
                }
            }
            _ => {
                let salt = 0x8000_0000u32 ^ ((t as u32).wrapping_mul(0x9E37_79B9));
                for (i, v) in row.iter_mut().enumerate() {
                    *v = prng(i, salt) * 2.0 - 1.0;
                }
            }
        }
    }
    x
}

fn write_le(path: &str, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("write {path}: {e}"));
}

fn read_bin(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn block_d(y: &[u8], nb: usize, n: usize, b: usize, t: usize) -> f32 {
    let off = (b * n + t) * BLOCK_BYTES;
    f32::from_le_bytes([y[off], y[off + 1], y[off + 2], y[off + 3]])
}

fn block_s(y: &[u8], nb: usize, n: usize, b: usize, t: usize) -> i32 {
    let off = (b * n + t) * BLOCK_BYTES + 4;
    i32::from_le_bytes([y[off], y[off + 1], y[off + 2], y[off + 3]])
}

fn block_q(y: &[u8], nb: usize, n: usize, b: usize, t: usize, j: usize) -> i32 {
    let off = (b * n + t) * BLOCK_BYTES + 8;
    let lane = j / 4;
    let sub = j % 4;
    let byte = y[off + lane * 2 + (sub >= 2) as usize];
    let nib = if sub % 2 == 0 { byte & 15 } else { byte >> 4 };
    (nib as i32) - if nib >= 8 { 16 } else { 0 }
}

fn cpu_quant(x: f32, d: f32) -> i32 {
    ((x / d).round_ties_even().clamp(-8.0, 7.0)) as i32
}

/// Exact replica of the fixup wave-tree half SSE for one 128-block.
fn cpu_block_sse(xs: &[f32], d: f32) -> f32 {
    let mut lanes = [0.0f32; 32];
    for lane in 0..32 {
        let mut sse = 0.0f32;
        for k in 0..4 {
            let x = xs[lane * 4 + k];
            let q = cpu_quant(x, d) as f32;
            let err = (-q).mul_add(d, x);
            sse = err.mul_add(err, sse);
        }
        lanes[lane] = sse;
    }
    for off in [16, 8, 4, 2, 1] {
        let prev = lanes;
        for l in 0..32 {
            lanes[l] += prev[l ^ off];
        }
    }
    lanes[0]
}

fn do_dump(gpu: &mut Gpu, prefix: &str, k: usize, n: usize) {
    let x = build_x(n, k);
    let d_x = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
    gpu.hip
        .memcpy_htod(
            &d_x.buf,
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) },
        )
        .expect("htod x");
    gpu.hip.device_synchronize().expect("sync x");
    let ptr = gpu.ensure_int4_mmq_x(&d_x, n, k).expect("ensure int4");
    gpu.hip.device_synchronize().expect("sync quant");
    let nb = k / BLOCK;
    let y_len = nb * n * BLOCK_BYTES;
    let mut y = vec![0u8; y_len];
    let view = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, y_len) };
    gpu.hip.memcpy_dtoh(&mut y, &view).expect("dtoh y");
    let mut xb = vec![0u8; x.len() * 4];
    for (i, v) in x.iter().enumerate() {
        xb[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    write_le(&format!("{prefix}.x.bin"), &xb);
    write_le(&format!("{prefix}.y.bin"), &y);
    std::fs::write(format!("{prefix}.shape"), format!("{n} {k}\n"))
        .unwrap_or_else(|e| panic!("write {prefix}.shape: {e}"));
    eprintln!("dump {prefix}: n={n} k={k} nb={nb} y={y_len}B xmaster={}", gpu.flags.iu4_xmaster_enabled());
}

fn do_verify(a_prefix: &str, b_prefix: &str) -> bool {
    let ax = as_f32(&read_bin(&format!("{a_prefix}.x.bin")));
    let bx = as_f32(&read_bin(&format!("{b_prefix}.x.bin")));
    assert_eq!(ax, bx, "X differs between dumps");
    let ay = read_bin(&format!("{a_prefix}.y.bin"));
    let by = read_bin(&format!("{b_prefix}.y.bin"));
    assert_eq!(ay.len(), by.len(), "Y length differs");
    let x = ax;
    // (n, k) from the dump sidecar: length factorization is ambiguous
    // (e.g. n=64/nb=1 vs n=8/nb=8 give the same byte counts).
    let shape_a = std::fs::read_to_string(format!("{a_prefix}.shape")).expect("read a.shape");
    let shape_b = std::fs::read_to_string(format!("{b_prefix}.shape")).expect("read b.shape");
    assert_eq!(shape_a, shape_b, "shapes differ between dumps");
    let mut it = shape_a.split_whitespace();
    let n: usize = it.next().expect("shape n").parse().expect("parse n");
    let k: usize = it.next().expect("shape k").parse().expect("parse k");
    assert_eq!(k % BLOCK, 0, "k not a multiple of 128");
    let nb = k / BLOCK;
    assert_eq!(x.len(), n * k, "x length vs shape");
    assert_eq!(by.len(), nb * n * BLOCK_BYTES, "y length vs shape");
    eprintln!("verify: n={n} k={k} nb={nb}");
    let mut ok = true;
    let mut e_hist = [0usize; 3];
    let mut d_fail = 0;
    let mut q_fail = 0;
    let mut s_fail = 0;
    let mut e_fail = 0;
    for t in 0..n {
        let row = &x[t * k..(t + 1) * k];
        let zero = row.iter().all(|v| *v == 0.0);
        let master = if zero {
            1.0f32
        } else {
            let mut m = 0.0f32;
            for b in 0..nb {
                m = m.max(block_d(&ay, nb, n, b, t));
            }
            m * 0.25
        };
        let cands = [master, master * 2.0, master * 4.0];
        for b in 0..nb {
            let got_d = block_d(&by, nb, n, b, t);
            let e = (0..3).find(|e| cands[*e].to_bits() == got_d.to_bits());
            match e {
                Some(e) => {
                    e_hist[e] += 1;
                    // q/s consistency against the stored scale.
                    let xs = &row[b * BLOCK..(b + 1) * BLOCK];
                    let mut s = 0i32;
                    for j in 0..BLOCK {
                        let want = cpu_quant(xs[j], got_d);
                        s += want;
                        let got = block_q(&by, nb, n, b, t, j);
                        if got != want {
                            q_fail += 1;
                            if q_fail < 5 {
                                eprintln!("  Q MISMATCH t={t} b={b} j={j}: got={got} want={want} d={got_d:e} x={}", xs[j]);
                            }
                        }
                    }
                    if block_s(&by, nb, n, b, t) != s {
                        s_fail += 1;
                        if s_fail < 5 {
                            eprintln!("  S MISMATCH t={t} b={b}: got={} want={s}", block_s(&by, nb, n, b, t));
                        }
                    }
                    // e-optimality vs CPU half-SSE replica (skipped for zero rows:
                    // fixup preserves pass-1 bytes there by construction).
                    if !zero {
                        let mut best = 0;
                        let mut best_sse = f32::INFINITY;
                        for (ce, dc) in cands.iter().enumerate() {
                            let sse = cpu_block_sse(xs, *dc);
                            if sse < best_sse {
                                best_sse = sse;
                                best = ce;
                            }
                        }
                        if best != e {
                            e_fail += 1;
                            if e_fail < 5 {
                                eprintln!("  E MISMATCH t={t} b={b}: stored e={e} cpu-argmin e={best}");
                            }
                        }
                    }
                }
                None => {
                    d_fail += 1;
                    if d_fail < 5 {
                        eprintln!("  D MISMATCH t={t} b={b}: stored {got_d:e} ({:08x}) not in master {master:e} * {{1,2,4}}", got_d.to_bits());
                    }
                }
            }
        }
        if zero {
            // Zero-row preservation: pass-1 bytes d=1/q=0/s=0 must survive.
            for b in 0..nb {
                assert_eq!(block_d(&ay, nb, n, b, t).to_bits(), 1.0f32.to_bits(), "pass-1 zero d != 1");
                assert_eq!(block_d(&by, nb, n, b, t).to_bits(), 1.0f32.to_bits(), "zero row d changed t={t} b={b}");
                assert_eq!(block_s(&by, nb, n, b, t), 0, "zero row s != 0 t={t} b={b}");
            }
        }
    }
    eprintln!("e histogram: e0={} e1={} e2={}", e_hist[0], e_hist[1], e_hist[2]);
    eprintln!("d_fail={d_fail} q_fail={q_fail} s_fail={s_fail} e_fail={e_fail}");
    if d_fail > 0 {
        eprintln!("GATE FAIL: stored d != master*2^e bitwise");
        ok = false;
    }
    if q_fail > 0 || s_fail > 0 {
        eprintln!("GATE FAIL: q/s inconsistent with stored d");
        ok = false;
    }
    if e_fail > 0 {
        eprintln!("GATE FAIL: stored e != CPU half-SSE argmin");
        ok = false;
    }
    eprintln!("{}", if ok { "ORACLE PASS" } else { "ORACLE FAIL" });
    ok
}

fn do_time(gpu: &mut Gpu, k: usize) {
    let n = 512usize;
    let x: Vec<f32> = (0..n * k)
        .map(|i| prng(i, 0x71E0_0001) * 2.0 - 1.0)
        .collect();
    let d_x = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
    gpu.hip
        .memcpy_htod(
            &d_x.buf,
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) },
        )
        .expect("htod x");
    gpu.hip.device_synchronize().expect("sync x");
    for _ in 0..5 {
        gpu.ensure_int4_mmq_x(&d_x, n, k).expect("ensure");
    }
    gpu.hip.device_synchronize().expect("sync warmup");
    let mut dts = Vec::with_capacity(20);
    for _ in 0..20 {
        let t0 = std::time::Instant::now();
        gpu.ensure_int4_mmq_x(&d_x, n, k).expect("ensure");
        gpu.hip.device_synchronize().expect("sync");
        dts.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    dts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = dts[dts.len() / 2];
    let mean = dts.iter().sum::<f64>() / dts.len() as f64;
    eprintln!(
        "TIME ensure_int4_mmq_x n={n} k={k} xmaster={} med={med:.1}us mean={mean:.1}us min={:.1} max={:.1} (sync-delimited, 20 calls)",
        gpu.flags.iu4_xmaster_enabled(),
        dts[0],
        dts[dts.len() - 1],
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} {{dump <prefix>|verify <a> <b>|time}} [--k K] [--n N]", args[0]);
        std::process::exit(2);
    }
    let mut k = 1024usize;
    let mut n = 8usize;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--k" => {
                k = args[i + 1].parse().expect("--k");
                i += 2;
            }
            "--n" => {
                n = args[i + 1].parse().expect("--n");
                i += 2;
            }
            _ => break,
        }
    }
    let rest = &args[i..];
    // verify is host-only (no GPU needed).
    if rest[0] == "verify" {
        if !do_verify(&rest[1], &rest[2]) {
            std::process::exit(1);
        }
        return;
    }
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    if !(gpu.arch_caps.is_gfx1201() && gpu.arch == "gfx1201") {
        eprintln!("SKIP: arch {} is not exact gfx1201", gpu.arch);
        return;
    }
    eprintln!(
        "arch gfx1201 flags: iu4_prefill={} iu4_xmaster={}",
        gpu.flags.iu4_prefill_enabled(),
        gpu.flags.iu4_xmaster_enabled()
    );
    match rest[0].as_str() {
        "dump" => do_dump(&mut gpu, &rest[1], k, n),
        "verify" => {
            if !do_verify(&rest[1], &rest[2]) {
                std::process::exit(1);
            }
        }
        "time" => do_time(&mut gpu, k),
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
    }
}
