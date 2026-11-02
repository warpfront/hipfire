// SPDX-License-Identifier: MIT OR Apache-2.0
// gfx1201 symmetric qt49 A8 iu8/fp8 GEMM, production fp8 MQ4 and old f16.
// Synthetic inputs stand in for the already-rotated activation domain;
// actual captured-site activation quality is measured separately on CPU.
use hip_bridge::KernargBlob;
use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{DType, Gpu, GpuTensor};

const PRELUDE: &str = include_str!("../../../kernels/src/block_i8_128_quant.hip");
const IU8: &str = include_str!("../../../kernels/src/gemm_mq3v25_residual_mmq_iu8.gfx12.hip");
const FP8_SYM3: &str = include_str!("../../../kernels/src/gemm_mq3v25_residual_fp8.gfx12.hip");
const FP8_PACK: &str = include_str!("../../../kernels/src/pack_f16_to_fp8_mq4v2.gfx12.hip");
const FP8: &str = include_str!("../../../kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip");
const F16: &str = include_str!("../../../kernels/src/gemm_mq3g256v2_residual_wmma.gfx12.hip");
const I8_NAME: &str = "gemm_mq3v25_residual_mmq_iu8";
const FP8_SYM3_NAME: &str = "gemm_mq3v25_residual_fp8";
const FP8_NAME: &str = "gemm_mq4g256v2_residual_wmma_fp8_v2_b128x128_gfx1201_symfold";
const F16_NAME: &str = "gemm_mq3g256v2_residual_wmma_gfx12";
const SHAPES: &[(&str, usize, usize)] = &[
    ("qkv", 10240, 5120),
    ("gate", 17408, 5120),
    ("up", 17408, 5120),
    ("out", 5120, 6144),
    ("down", 5120, 17408),
];
fn f16(h: u16) -> f32 {
    let exp = ((h >> 10) & 31) as u32;
    assert!((1..31).contains(&exp));
    f32::from_bits(((h as u32 & 0x8000) << 16) | ((exp + 112) << 23) | ((h as u32 & 1023) << 13))
}
fn scale(row: usize, group: usize, half: usize) -> u16 {
    half_from_f32(0.003 + ((row * 17 + group * 13 + half * 29) % 107) as f32 * 0.000031)
}
fn code(row: usize, group: usize, k: usize, salt: usize) -> u8 {
    ((row * 13 + group * 23 + k * 17 + (k >> 3) * 7 + salt) & 7) as u8
}
fn weights(m: usize, k: usize, salt: usize) -> (Vec<u8>, Vec<u8>) {
    let gpr = k / 256;
    let mut w3 = vec![0u8; m * gpr * 104];
    let mut w4 = vec![0u8; m * gpr * 136];
    for row in 0..m {
        for g in 0..gpr {
            let a = (row * gpr + g) * 104;
            let b = (row * gpr + g) * 136;
            for h in 0..2 {
                let d = scale(row, g, h);
                let z3 = half_from_f32(-4.0 * f16(d));
                let z4 = half_from_f32(-8.0 * f16(d));
                assert_eq!(f16(z3), -4.0 * f16(d));
                w3[a + h * 4..a + h * 4 + 2].copy_from_slice(&d.to_le_bytes());
                w3[a + h * 4 + 2..a + h * 4 + 4].copy_from_slice(&z3.to_le_bytes());
                w4[b + h * 4..b + h * 4 + 2].copy_from_slice(&d.to_le_bytes());
                w4[b + h * 4 + 2..b + h * 4 + 4].copy_from_slice(&z4.to_le_bytes());
            }
            for octet in 0..32 {
                let mut bits = 0u32;
                for j in 0..8 {
                    let ix = octet * 8 + j;
                    let u = code(row, g, ix, salt);
                    bits |= (u as u32) << (3 * j);
                    w4[b + 8 + ix / 2] |= (u + 4) << (4 * (ix & 1));
                }
                w3[a + 8 + octet * 3..a + 11 + octet * 3].copy_from_slice(&bits.to_le_bytes()[..3]);
            }
        }
    }
    (w3, w4)
}
fn input(token: usize, k: usize) -> f32 {
    let v = ((token * 11 + k * 13 + (k >> 2) * 5 + (token >> 3) * 17) % 255) as i32 - 127;
    (v as f32) * 0.0078125
}
// Byte offset inside `qs` of K element `i` in beta's nibble-pair order: within
// each 8-element group, qs[8g+0..3] = q[8g+0,2,4,6], qs[8g+4..7] = q[8g+1,3,5,7].
fn qs_pos(i: usize) -> usize {
    let (g, j) = (i / 8, i % 8);
    8 * g + j / 2 + 4 * (j % 2)
}
// block_i8_128 { float d; int s; signed char qs[128]; } = 136 B, qs at byte 8.
fn cpu_block(x: &[f32]) -> [u8; 136] {
    let mut out = [0u8; 136];
    let amax = x.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let zero = amax == 0.0;
    let d = if zero { 0.0f32 } else { amax / 127.0 };
    let mut s = 0i32;
    for (i, &v) in x.iter().enumerate() {
        let q = if zero {
            0
        } else {
            (v / d).round_ties_even().clamp(-127.0, 127.0) as i32
        };
        s += q;
        out[8 + qs_pos(i)] = (q as i8) as u8;
    }
    out[..4].copy_from_slice(&d.to_le_bytes());
    out[4..8].copy_from_slice(&s.to_le_bytes());
    out
}
fn args3(a: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize, n: usize) -> Vec<u8> {
    let mut b = KernargBlob::new();
    for t in [a, x, y] {
        b.push_ptr(t.buf.as_ptr());
    }
    for v in [m, k, n, 0] {
        b.push_i32(v as i32);
    }
    b.into_vec()
}
fn args_f16(a: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize, n: usize) -> Vec<u8> {
    let mut b = KernargBlob::new();
    for t in [a, x, y] {
        b.push_ptr(t.buf.as_ptr());
    }
    for v in [m, k, n] {
        b.push_i32(v as i32);
    }
    b.into_vec()
}
fn args_fp8(
    a: &GpuTensor,
    x: &GpuTensor,
    sums: &GpuTensor,
    scales: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<u8> {
    let mut b = KernargBlob::new();
    for t in [a, x, sums, scales, y] {
        b.push_ptr(t.buf.as_ptr());
    }
    for v in [m, k, n] {
        b.push_i32(v as i32);
    }
    b.into_vec()
}
fn launch(gpu: &Gpu, name: &str, arg: &mut [u8], m: usize, n: usize) {
    let (rm, rn, threads, lds) = match name {
        I8_NAME => (128, 128, 256, 19456),
        FP8_NAME | FP8_SYM3_NAME => (128, 128, 256, 18944),
        _ => (16, 16, 32, 0),
    };
    gpu.launch_kernel_blob(
        name,
        [m.div_ceil(rm) as u32, n.div_ceil(rn) as u32, 1],
        [threads, 1, 1],
        lds,
        arg,
    )
    .unwrap();
}
fn measure(gpu: &Gpu, name: &str, arg: &mut [u8], m: usize, n: usize) -> f64 {
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    launch(gpu, name, arg, m, n);
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
    gpu.hip.event_destroy(start).unwrap();
    gpu.hip.event_destroy(stop).unwrap();
    ms
}
fn oracle(gpu: &Gpu, y: &GpuTensor, m: usize, k: usize, n: usize, salt: usize, blocks: &[u8]) {
    let got = gpu.download_f32(y).unwrap();
    let rows = [0, 1, 15, 16, 31, 63, 64, 127, m / 2, m - 1];
    let cols = [0, 1, 15, 16, 31, 63, 64, n / 2, n - 1];
    let mut max_abs = 0.0f32;
    for &r in &rows {
        for &col in &cols {
            if col >= n {
                continue;
            }
            let mut expected = 0.0f32;
            for kb in 0..k / 128 {
                let d =
                    f32::from_le_bytes(blocks[((kb * n + col) * 136)..][..4].try_into().unwrap());
                let mut dot = 0i32;
                for i in 0..128 {
                    let a = blocks[(kb * n + col) * 136 + 8 + qs_pos(i)] as i8 as i32;
                    dot += (code(r, kb / 2, (kb & 1) * 128 + i, salt) as i32 - 4) * a;
                }
                expected += (f16(scale(r, kb / 2, kb & 1)) * d) * dot as f32;
            }
            let value = got[col * m + r];
            let err = (value - expected).abs();
            max_abs = max_abs.max(err);
            assert!(
                err <= 0.00003 * expected.abs().max(1.0),
                "IU8 fold row={r} token={col} GPU={value} CPU={expected} diff={err}"
            );
        }
    }
    println!(
        "CPU i32-dot + K128 fold parity sampled={} max_abs={max_abs:.3e}",
        rows.len() * cols.len()
    );
}
fn e4m3(b: u8) -> f32 {
    let v = if b & 0x78 == 0 {
        (b & 7) as f32 * 2f32.powi(-9)
    } else {
        (1.0 + (b & 7) as f32 / 8.0) * 2f32.powi(((b >> 3) & 15) as i32 - 7)
    };
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}
fn cpu_e4m3(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 0x80 } else { 0 };
    let a = v.abs();
    let (mut low, mut high) = (0u8, 126u8);
    while high - low > 1 {
        let mid = low + (high - low) / 2;
        if e4m3(mid) <= a {
            low = mid;
        } else {
            high = mid;
        }
    }
    let dlow = a - e4m3(low);
    let dhigh = e4m3(high) - a;
    sign | if dhigh < dlow || (dhigh == dlow && low & 1 != 0) {
        high
    } else {
        low
    }
}
fn fp8_oracle(
    gpu: &Gpu,
    y: &GpuTensor,
    fp: &GpuTensor,
    scales: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    salt: usize,
) {
    let got = gpu.download_f32(y).unwrap();
    let mut bytes = vec![0u8; n * k];
    gpu.hip.memcpy_dtoh(&mut bytes, &fp.buf).unwrap();
    let row_scales = gpu.download_f32(scales).unwrap();
    assert!(
        bytes.iter().any(|&v| v & 0x7f != 0),
        "fp8 producer emitted only zeroes"
    );
    assert!(
        row_scales.iter().all(|&v| v > 0.0 && v.is_finite()),
        "fp8 row scales invalid"
    );
    let rows = [0, 1, 15, 16, 63, 64, 127, 128, m / 2, m - 1];
    let cols = [0, 1, 15, 16, 63, 64, 127, 128, n / 2, n - 1];
    // Independently reproduce production f16 -> scaled e4m3 (RN-even).
    let mut cpu_rows = Vec::with_capacity(cols.len());
    for &col in &cols {
        if col >= n {
            cpu_rows.push(Vec::new());
            continue;
        }
        let half: Vec<f32> = (0..k)
            .map(|i| {
                let h = half_from_f32(input(col, i));
                if h & 0x7fff == 0 {
                    0.0
                } else {
                    f16(h)
                }
            })
            .collect();
        let amax = half.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let a = if amax == 0.0 {
            1.0
        } else {
            2f32.powi(((amax / 224.0).log2().ceil() as i32).clamp(-31, 9))
        };
        assert_eq!(
            row_scales[col].to_bits(),
            a.to_bits(),
            "fp8 producer row scale"
        );
        let mut decoded = Vec::with_capacity(k);
        for (i, &v) in half.iter().enumerate() {
            let b = cpu_e4m3(v / a);
            assert_eq!(
                bytes[col * k + i],
                b,
                "fp8 producer RN-even token={col} k={i}"
            );
            decoded.push(e4m3(b));
        }
        cpu_rows.push(decoded);
    }
    let mut max_abs = 0f32;
    let mut max_reference = 0f32;
    for &r in &rows {
        for (ci, &col) in cols.iter().enumerate() {
            if col >= n {
                continue;
            }
            let mut expected = 0f32;
            for kb in 0..k / 128 {
                let mut dot = 0f32;
                for i in 0..128 {
                    let a = cpu_rows[ci][kb * 128 + i];
                    dot += (code(r, kb / 2, (kb & 1) * 128 + i, salt) as i32 - 4) as f32 * a;
                }
                expected = f16(scale(r, kb / 2, kb & 1)).mul_add(dot, expected);
            }
            expected *= row_scales[col];
            max_reference = max_reference.max(expected.abs());
            let value = got[col * m + r];
            let err = (value - expected).abs();
            max_abs = max_abs.max(err);
            assert!(
                err <= 0.001 * expected.abs().max(1.0),
                "FP8 sym3 row={r} token={col} GPU={value} CPU={expected} diff={err}"
            );
        }
    }
    assert!(
        max_reference > 0.01 && max_abs.is_finite(),
        "fp8 CPU oracle was degenerate"
    );
    println!(
        "CPU e4m3 sym3 fold sampled={} max_abs={max_abs:.3e} max_reference={max_reference:.3e}",
        rows.len() * cols.len()
    );
}
fn producer_boundaries(gpu: &mut Gpu) {
    const K: usize = 256;
    const N: usize = 3;
    let mut x = vec![0f32; K * N];
    for token in 0..N {
        x[token * K] = 127.0;
        x[token * K + 1] = -127.0;
        x[token * K + 2] = 0.5;
        x[token * K + 3] = 1.5;
        x[token * K + 4] = 2.5;
        x[token * K + 5] = -0.5;
        x[token * K + 6] = -1.5;
        x[token * K + 7] = -2.5;
        // The second K128 is all zero; its scale is the specified 1.
    }
    let raw: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tx = gpu.upload_raw(&raw, &[raw.len()]).unwrap();
    let y = gpu.zeros(&[K / 128 * N * 34], DType::F32).unwrap();
    let mut arg = KernargBlob::new();
    arg.push_ptr(tx.buf.as_ptr());
    arg.push_ptr(y.buf.as_ptr());
    arg.push_i32(K as i32);
    arg.push_i32(N as i32);
    gpu.launch_kernel_blob(
        "quantize_int8_mmq_ds128",
        [(K.div_ceil(1024) * N) as u32, 1, 1],
        [256, 1, 1],
        0,
        &mut arg.into_vec(),
    )
    .unwrap();
    let mut actual = vec![0u8; K / 128 * N * 136];
    gpu.hip.memcpy_dtoh(&mut actual, &y.buf).unwrap();
    for kb in 0..K / 128 {
        for token in 0..N {
            let expected = cpu_block(&x[token * K + kb * 128..token * K + (kb + 1) * 128]);
            assert_eq!(
                &actual[(kb * N + token) * 136..(kb * N + token + 1) * 136],
                &expected
            );
        }
    }
    assert_eq!(actual[8 + qs_pos(2)] as i8, 0); // 0.5 -> even zero
    assert_eq!(actual[8 + qs_pos(3)] as i8, 2); // 1.5 -> even two
    assert_eq!(actual[8 + qs_pos(4)] as i8, 2); // 2.5 -> even two
    assert_eq!(actual[8 + qs_pos(5)] as i8, 0); // -0.5 -> even zero
    assert_eq!(actual[(N) * 136..(N) * 136 + 4], 0f32.to_le_bytes()); // zero block: d = 0
    assert_eq!(actual[(N) * 136 + 4..(N) * 136 + 8], 0i32.to_le_bytes()); // zero block: s = 0
    println!(
        "producer boundary RN ties/zero/saturation bit-exact blocks={}",
        K / 128 * N
    );
}

fn exact_i32_oracle(gpu: &mut Gpu) {
    // Scale=1 on BOTH sides: every output is the actual signed K256 integer
    // WMMA dot (including both independently folded K128 groups). Exercise
    // the two CTA tails, SET and ADD, rather than inferring dot parity from
    // a tolerant floating-point comparison.
    const M: usize = 129;
    const N: usize = 129;
    const K: usize = 256;
    let (mut w, _) = weights(M, K, 0);
    for row in 0..M {
        for h in 0..2 {
            let base = row * 104 + h * 4;
            w[base..base + 2].copy_from_slice(&half_from_f32(1.0).to_le_bytes());
            w[base + 2..base + 4].copy_from_slice(&half_from_f32(-4.0).to_le_bytes());
        }
    }
    let mut q = vec![0u8; 2 * N * 136];
    for kb in 0..2 {
        for token in 0..N {
            let base = (kb * N + token) * 136;
            q[base..base + 4].copy_from_slice(&1f32.to_le_bytes());
            let mut s = 0i32;
            for i in 0..128 {
                let value = if i == 0 {
                    127
                } else {
                    ((token * 11 + kb * 19 + i * 13) % 255) as i32 - 127
                };
                s += value;
                q[base + 8 + qs_pos(i)] = value as i8 as u8;
            }
            q[base + 4..base + 8].copy_from_slice(&s.to_le_bytes());
        }
    }
    let tw = gpu.upload_raw(&w, &[w.len()]).unwrap();
    let tq = gpu.upload_raw(&q, &[q.len()]).unwrap();
    let y = gpu.zeros(&[N, M], DType::F32).unwrap();
    let mut set = args3(&tw, &tq, &y, M, K, N);
    launch(gpu, I8_NAME, &mut set, M, N);
    let got = gpu.download_f32(&y).unwrap();
    for token in 0..N {
        for row in 0..M {
            let mut expected = 0i32;
            for kb in 0..2 {
                for i in 0..128 {
                    let a = q[(kb * N + token) * 136 + 8 + qs_pos(i)] as i8 as i32;
                    expected += (code(row, 0, kb * 128 + i, 0) as i32 - 4) * a;
                }
            }
            assert_eq!(
                got[token * M + row].to_bits(),
                (expected as f32).to_bits(),
                "exact i32 WMMA row={row} token={token}"
            );
        }
    }
    let mut add = KernargBlob::new();
    for t in [&tw, &tq, &y] {
        add.push_ptr(t.buf.as_ptr());
    }
    for v in [M, K, N, 1] {
        add.push_i32(v as i32);
    }
    launch(gpu, I8_NAME, &mut add.into_vec(), M, N);
    let doubled = gpu.download_f32(&y).unwrap();
    for (a, b) in doubled.iter().zip(got.iter()) {
        assert_eq!(a.to_bits(), (2.0 * b).to_bits(), "ADD integer fold");
    }
    println!("exact i32 WMMA SET+ADD outputs={}", N * M);
}

fn main() {
    let mut family = "all".to_owned();
    let mut n = 8192usize;
    let mut it = std::env::args().skip(1);
    while let Some(key) = it.next() {
        let value = it.next().expect("option value");
        match key.as_str() {
            "--family" => family = value,
            "--n" => n = value.parse().unwrap(),
            "--arch" => assert_eq!(value, "gfx1201"),
            _ => panic!("unknown flag {key}"),
        }
    }
    assert!(n > 0);
    let mut gpu = Gpu::init().unwrap();
    assert_eq!(gpu.arch, "gfx1201");
    let gemm_i8 = format!("#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n{PRELUDE}\n{IU8}");
    gpu.ensure_kernel_public("mqv25_a8_producer", PRELUDE, "quantize_int8_mmq_ds128")
        .unwrap();
    gpu.ensure_kernel_public("mqv25_iu8", &gemm_i8, I8_NAME)
        .unwrap();
    gpu.ensure_kernel_public("mqv25_fp8_sym3", FP8_SYM3, FP8_SYM3_NAME)
        .unwrap();
    gpu.ensure_kernel_public("mqv25_fp8_pack", FP8_PACK, "pack_f16_to_fp8_mq4v2_gfx12")
        .unwrap();
    let fp8_src = format!("#define HIPFIRE_FP8_SYMFOLD 1\n#define HIPFIRE_FP8_V2_TILE 1\n#define HIPFIRE_FP8_V2_BM 128\n#define HIPFIRE_FP8_V2_BN 128\n#define HIPFIRE_FP8_V2_BK 64\n#define HIPFIRE_FP8_V2_WAVES 8\n#define HIPFIRE_FP8_RESIDUAL 1\n#define HIPFIRE_FP8_GATEUP_KERNEL {FP8_NAME}\n{FP8}");
    gpu.ensure_kernel_public("mqv25_fp8_prod", &fp8_src, FP8_NAME)
        .unwrap();
    gpu.ensure_kernel_public("mqv25_old_f16", F16, F16_NAME)
        .unwrap();
    println!("arch={} IU8=128x128/8w/19456B occ={} FP8-sym3=128x128x64/8w/18944B occ={} FP8-MQ4 occ={} f16={F16_NAME}",
        gpu.arch, gpu.occupancy_max_active_blocks(I8_NAME, [256,1,1], 19456).unwrap(),
        gpu.occupancy_max_active_blocks(FP8_SYM3_NAME, [256,1,1], 18944).unwrap(),
        gpu.occupancy_max_active_blocks(FP8_NAME, [256,1,1], 18944).unwrap());
    producer_boundaries(&mut gpu);
    exact_i32_oracle(&mut gpu);
    println!("family,M,K,N,iu8_ms,fp8_sym3_ms,fp8_mq4_ms,mq3_f16_ms,iu8/mq4,sym3/mq4,iu8/sym3");
    for &(label, m, k) in SHAPES {
        if family != "all" && family != label {
            continue;
        }
        let salt = if label == "up" { 3 } else { 0 };
        let (w3, w4) = weights(m, k, salt);
        let mut x = Vec::with_capacity(n * k);
        for col in 0..n {
            for i in 0..k {
                x.push(input(col, i));
            }
        }
        let x_half: Vec<u16> = x.iter().map(|v| half_from_f32(*v)).collect();
        let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let h_bytes: Vec<u8> = x_half.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tw3 = gpu.upload_raw(&w3, &[w3.len()]).unwrap();
        let tw4 = gpu.upload_raw(&w4, &[w4.len()]).unwrap();
        let tx = gpu.upload_raw(&x_bytes, &[x_bytes.len()]).unwrap();
        let txh = gpu.upload_raw(&h_bytes, &[h_bytes.len()]).unwrap();
        drop((x_bytes, h_bytes, x_half));
        // Byte views of f32 scratch: only allocation size/pointer matter to HIP.
        let q = gpu.zeros(&[(k / 128) * n * 34], DType::F32).unwrap();
        let fp = gpu.zeros(&[n * k / 4], DType::F32).unwrap();
        let sums = gpu.zeros(&[n * (k / 128)], DType::F32).unwrap();
        let scales = gpu.zeros(&[n], DType::F32).unwrap();
        let yi = gpu.zeros(&[n, m], DType::F32).unwrap();
        let ys = gpu.zeros(&[n, m], DType::F32).unwrap();
        let yf = gpu.zeros(&[n, m], DType::F32).unwrap();
        let yh = gpu.zeros(&[n, m], DType::F32).unwrap();
        let mut quant = KernargBlob::new();
        quant.push_ptr(tx.buf.as_ptr());
        quant.push_ptr(q.buf.as_ptr());
        quant.push_i32(k as i32);
        quant.push_i32(n as i32);
        let mut quant = quant.into_vec();
        gpu.launch_kernel_blob(
            "quantize_int8_mmq_ds128",
            [(k.div_ceil(1024) * n) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut quant,
        )
        .unwrap();
        let mut blocks = vec![0u8; (k / 128) * n * 136];
        gpu.hip.memcpy_dtoh(&mut blocks, &q.buf).unwrap();
        let mut checked = 0usize;
        for kb in 0..k / 128 {
            for col in 0..n {
                let expected = cpu_block(&x[col * k + kb * 128..col * k + kb * 128 + 128]);
                let base = (kb * n + col) * 136;
                assert_eq!(
                    &blocks[base..base + 136],
                    &expected,
                    "producer kb={kb} col={col}"
                );
                checked += 1;
            }
        }
        println!(
            "producer {label} bit-exact K128 blocks={checked} bytes={}",
            blocks.len()
        );
        let mut pack = KernargBlob::new();
        for t in [&txh, &fp, &sums, &scales] {
            pack.push_ptr(t.buf.as_ptr());
        }
        pack.push_i32(k as i32);
        pack.push_i32(n as i32);
        pack.push_i32(1);
        gpu.launch_kernel_blob(
            "pack_f16_to_fp8_mq4v2_gfx12",
            [n as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut pack.into_vec(),
        )
        .unwrap();
        let mut ai = args3(&tw3, &q, &yi, m, k, n);
        let mut as3 = args_fp8(&tw3, &fp, &sums, &scales, &ys, m, k, n);
        let mut af = args_fp8(&tw4, &fp, &sums, &scales, &yf, m, k, n);
        let mut ah = args_f16(&tw3, &txh, &yh, m, k, n);
        launch(&gpu, I8_NAME, &mut ai, m, n);
        gpu.hip.device_synchronize().unwrap();
        oracle(&gpu, &yi, m, k, n, salt, &blocks);
        launch(&gpu, FP8_SYM3_NAME, &mut as3, m, n);
        gpu.hip.device_synchronize().unwrap();
        fp8_oracle(&gpu, &ys, &fp, &scales, m, k, n, salt);
        // All four arms time GEMM only; each has a distinct output buffer.
        for round in 0..24 {
            for arm in [round % 4, (round + 1) % 4, (round + 2) % 4, (round + 3) % 4] {
                let (name, arg) = match arm {
                    0 => (I8_NAME, &mut ai),
                    1 => (FP8_SYM3_NAME, &mut as3),
                    2 => (FP8_NAME, &mut af),
                    _ => (F16_NAME, &mut ah),
                };
                launch(&gpu, name, arg, m, n);
            }
        }
        gpu.hip.device_synchronize().unwrap();
        let mut times = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for round in 0..7 {
            for arm in [round % 4, (round + 1) % 4, (round + 2) % 4, (round + 3) % 4] {
                let (name, arg) = match arm {
                    0 => (I8_NAME, &mut ai),
                    1 => (FP8_SYM3_NAME, &mut as3),
                    2 => (FP8_NAME, &mut af),
                    _ => (F16_NAME, &mut ah),
                };
                times[arm].push(measure(&gpu, name, arg, m, n));
            }
        }
        for samples in &mut times {
            samples.sort_by(|a, b| a.total_cmp(b));
        }
        let (i, s, f, h) = (times[0][3], times[1][3], times[2][3], times[3][3]);
        println!(
            "{label},{m},{k},{n},{i:.5},{s:.5},{f:.5},{h:.5},{:.4},{:.4},{:.4}",
            i / f,
            s / f,
            i / s
        );
        eprintln!(
            "samples {label}: IU8={:?} SYM3={:?} MQ4={:?} F16={:?}",
            times[0], times[1], times[2], times[3]
        );
    }
}
