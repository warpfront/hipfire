// SPDX-License-Identifier: MIT OR Apache-2.0
// Symmetric qt49 A4 against production V2C (gfx1100) / V2B (gfx1151) MQ4.
use hip_bridge::KernargBlob;
use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{DType, Gpu, GpuTensor};

const PROBE_V2C: &str = include_str!("../../../kernels/src/gemm_mq3v25_residual_iu4_v2c.gfx11.hip");
const CONTROL_V2C: &str =
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_iu4_v2c.gfx11.hip");
const PROBE_V2B: &str = include_str!("../../../kernels/src/gemm_mq3v25_residual_iu4_v2b.gfx11.hip");
const CONTROL_V2B: &str =
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_iu4_v2b.gfx11.hip");
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
    f32::from_bits(
        (((h & 0x8000) as u32) << 16) | ((exp + 112) << 23) | (((h & 1023) as u32) << 13),
    )
}
fn scale(row: usize, group: usize, half: usize) -> u16 {
    half_from_f32(0.003 + ((row * 17 + group * 13 + half * 29) % 107) as f32 * 0.000031)
}
fn code(row: usize, group: usize, k: usize, salt: usize) -> u8 {
    ((row * 13 + group * 23 + k * 17 + (k >> 3) * 7 + salt) & 7) as u8
}
fn activation(token: usize, kb: usize, i: usize) -> i32 {
    (((token * 11 + kb * 19 + i * 13 + (i >> 2) * 5) & 15) as i32) - 8
}
fn act_scale(token: usize, kb: usize) -> f32 {
    0.021 + ((token * 7 + kb * 17) % 43) as f32 * 0.00043
}
// Same signed codes and identical half scales in the 104-byte qt49 and 136-byte qt44 groups.
fn weights(m: usize, k: usize, salt: usize) -> (Vec<u8>, Vec<u8>) {
    let gpr = k / 256;
    let mut w3 = vec![0; m * gpr * 104];
    let mut w4 = vec![0; m * gpr * 136];
    for row in 0..m {
        for g in 0..gpr {
            let a = (row * gpr + g) * 104;
            let b = (row * gpr + g) * 136;
            for h in 0..2 {
                let d = scale(row, g, h);
                let v = f16(d);
                let z3 = half_from_f32(-4.0 * v);
                let z4 = half_from_f32(-8.0 * v);
                assert_eq!(f16(z3), -4.0 * v);
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
                    w4[b + 8 + ix / 2] |= (u + 4) << (4 * (ix % 2));
                }
                w3[a + 8 + octet * 3..a + 11 + octet * 3].copy_from_slice(&bits.to_le_bytes()[..3]);
            }
        }
    }
    (w3, w4)
}
// Produces the unchanged block_i4_128 ABI: [f32 d, i32 s, 64 signed-nibble bytes].
fn activations(k: usize, n: usize) -> Vec<u8> {
    let mut x = vec![0u8; (k / 128) * n * 72];
    for kb in 0..k / 128 {
        for token in 0..n {
            let base = (kb * n + token) * 72;
            x[base..base + 4].copy_from_slice(&act_scale(token, kb).to_le_bytes());
            let mut sum = 0i32;
            for i in 0..128 {
                let q = activation(token, kb, i);
                sum += q;
                x[base + 8 + i / 2] |= ((q & 15) as u8) << (4 * (i % 2));
            }
            x[base + 4..base + 8].copy_from_slice(&sum.to_le_bytes());
        }
    }
    x
}
// Mirrors the staged spreading and independently asserts every signed code.
fn check_codes() {
    for octet in 0..8u32 {
        let bits = (0..8).fold(0u32, |v, j| v | (((octet + j) & 7) << (3 * j)));
        let mut wide = (bits & 0xfff) | ((bits & 0xfff000) << 4);
        wide = (wide & 0x003f003f) | ((wide & 0x0fc00fc0) << 2);
        wide = (wide & 0x07070707) | ((wide & 0x38383838) << 1);
        wide = (wide.wrapping_add(0x44444444)) ^ 0x88888888;
        for j in 0..8 {
            assert_eq!((wide >> (4 * j)) & 15, ((((octet + j) & 7) + 4) ^ 8) & 15);
            assert_eq!(
                (wide >> (4 * j)) & 15,
                ((octet + j) & 7).wrapping_sub(4) & 15
            );
        }
    }
}
fn expected_at(row: usize, token: usize, k: usize, salt: usize) -> f32 {
    let mut sum = 0f32;
    for kb in 0..k / 128 {
        let g = kb / 2;
        let h = kb % 2;
        let mut dot = 0i32;
        for i in 0..128 {
            dot += (code(row, g, h * 128 + i, salt) as i32 - 4) * activation(token, kb, i);
        }
        // Production V2C uses RN(sc*d), exact i32 dot, and fused RN(t1*dot + sum).
        sum = (f16(scale(row, g, h)) * act_scale(token, kb)).mul_add(dot as f32, sum);
    }
    sum
}
fn check_output(label: &str, y: &[f32], m: usize, k: usize, n: usize, salt: usize, add: bool) {
    let rows: Vec<_> = [0, 1, 7, 15, 16, 31, 63, 127, m / 2, m - 1]
        .into_iter()
        .filter(|&row| row < m)
        .collect();
    let tokens: Vec<_> = [0, 1, 15, 64, 127, 128, n / 2, n - 1]
        .into_iter()
        .filter(|&token| token < n)
        .collect();
    let mut max_abs = 0f32;
    for &row in &rows {
        for &token in &tokens {
            let base = expected_at(row, token, k, salt);
            let expected = if add { base + base } else { base };
            let got = y[token * m + row];
            let err = (got - expected).abs();
            max_abs = max_abs.max(err);
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "{label}: token={token} row={row} GPU={got} CPU={expected} err={err}"
            );
        }
    }
    println!(
        "oracle {label} checked={} max_abs={max_abs:.3e}",
        rows.len() * tokens.len()
    );
}
fn args(
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    add: Option<i32>,
) -> Vec<u8> {
    let mut b = KernargBlob::new();
    b.push_ptr(w.buf.as_ptr());
    b.push_ptr(x.buf.as_ptr());
    b.push_ptr(y.buf.as_ptr());
    b.push_i32(m as i32);
    b.push_i32(k as i32);
    b.push_i32(n as i32);
    if let Some(value) = add {
        b.push_i32(value);
    }
    b.into_vec()
}
fn launch(gpu: &Gpu, name: &str, arg: &mut [u8], m: usize, n: usize) {
    let (tile, waves, lds) = if gpu.arch == "gfx1151" {
        (256, 16, 65536)
    } else {
        (128, 8, 32768)
    };
    gpu.launch_kernel_blob(
        name,
        [(n / tile) as u32, (m / tile) as u32, 1],
        [32, waves, 1],
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
fn median(v: &mut [f64; 7]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[3]
}
fn timing_pair(
    gpu: &Gpu,
    probe: &str,
    control: &str,
    probe_args: &mut [u8],
    control_args: &mut [u8],
    m: usize,
    n: usize,
) -> (f64, f64) {
    for _ in 0..16 {
        launch(gpu, probe, probe_args, m, n);
        launch(gpu, control, control_args, m, n);
    }
    gpu.hip.device_synchronize().unwrap();
    let mut p = [0f64; 7];
    let mut c = [0f64; 7];
    for i in 0..7 {
        if i % 2 == 0 {
            p[i] = measure(gpu, probe, probe_args, m, n);
            c[i] = measure(gpu, control, control_args, m, n);
        } else {
            c[i] = measure(gpu, control, control_args, m, n);
            p[i] = measure(gpu, probe, probe_args, m, n);
        }
    }
    eprintln!("samples probe={p:?} MQ4={c:?}");
    (median(&mut p), median(&mut c))
}
fn main() {
    let mut family = "all".to_string();
    let mut arch = "gfx1100".to_string();
    let mut n = 8192usize;
    let mut m_override = None;
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        let value = iter.next().expect("option needs value");
        match flag.as_str() {
            "--arch" => arch = value,
            "--family" => family = value,
            "--n" => n = value.parse().expect("--n integer"),
            "--m" => m_override = Some(value.parse::<usize>().expect("--m integer")),
            _ => panic!("unknown option {flag}"),
        }
    }
    check_codes();
    let mut gpu = Gpu::init().expect("HIP init");
    assert_eq!(gpu.arch, arch, "must use the requested exact gfx11 GPU");
    let (
        probe_src,
        control_src,
        probe_name,
        control_set,
        control_add,
        probe_module,
        control_module,
        tile,
        waves,
        lds,
    ) = match arch.as_str() {
        "gfx1100" => (
            PROBE_V2C,
            CONTROL_V2C,
            "gemm_mq3v25_residual_iu4_v2c",
            "gemm_mq4g256v2_residual_iu4_v2c_set_gfx11",
            "gemm_mq4g256v2_residual_iu4_v2c_add_touch_gfx11",
            "mq3v25_g11_v2c",
            "mq4v2_g11_v2c",
            128,
            8,
            32768,
        ),
        "gfx1151" => (
            PROBE_V2B,
            CONTROL_V2B,
            "gemm_mq3v25_residual_iu4_v2b",
            "gemm_mq4g256v2_residual_iu4_v2b_set_gfx11",
            "gemm_mq4g256v2_residual_iu4_v2b_add_touch_gfx11",
            "mq3v25_halo_v2b",
            "mq4v2_halo_v2b",
            256,
            16,
            65536,
        ),
        _ => panic!("unsupported arch {arch}"),
    };
    assert!(n > 0 && n % tile == 0, "full token tiles required");
    gpu.ensure_kernel_public(probe_module, probe_src, probe_name)
        .expect("JIT qt49 V2");
    gpu.ensure_kernel_public(control_module, control_src, control_set)
        .expect("JIT MQ4 V2 SET");
    gpu.ensure_kernel_public(control_module, control_src, control_add)
        .expect("JIT MQ4 V2 ADD_TOUCH");
    let block = [32, waves, 1];
    let occupancy_probe = gpu
        .occupancy_max_active_blocks(probe_name, block, lds)
        .unwrap();
    let occupancy_control = gpu
        .occupancy_max_active_blocks(control_set, block, lds)
        .unwrap();
    println!("PID={} arch={} occupancy CTA/CU probe={occupancy_probe} MQ4={occupancy_control} LDS={lds} tile={tile}x{tile}", std::process::id(), gpu.arch);
    println!("family,mode,M,K,N,qt49_ms,mq4_ms,ratio");
    let mut checked = 0usize;
    for &(name, default_m, k) in SHAPES {
        if family != "all" && family != name {
            continue;
        }
        let m = m_override.unwrap_or(default_m);
        assert!(
            m > 0 && m % tile == 0 && k % 256 == 0,
            "full M/K tiles required"
        );
        let salt = if name == "up" { 3 } else { 0 };
        let (w3, w4) = weights(m, k, salt);
        let x = activations(k, n);
        eprintln!(
            "producer {name} CPU (excluded), qt49={}B qt44={}B Xq={}B",
            w3.len(),
            w4.len(),
            x.len()
        );
        let t3 = gpu.upload_raw(&w3, &[w3.len()]).unwrap();
        let t4 = gpu.upload_raw(&w4, &[w4.len()]).unwrap();
        let tx = gpu.upload_raw(&x, &[x.len()]).unwrap();
        let y3 = gpu.zeros(&[n, m], DType::F32).unwrap();
        let y4 = gpu.zeros(&[n, m], DType::F32).unwrap();
        let mut a3 = args(&t3, &tx, &y3, m, k, n, Some(0));
        let mut a4 = args(&t4, &tx, &y4, m, k, n, None);
        launch(&gpu, probe_name, &mut a3, m, n);
        launch(&gpu, control_set, &mut a4, m, n);
        gpu.hip.device_synchronize().unwrap();
        let got3 = gpu.download_f32(&y3).unwrap();
        let got4 = gpu.download_f32(&y4).unwrap();
        check_output(name, &got3, m, k, n, salt, false);
        for (i, (a, b)) in got3.iter().zip(&got4).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{name}: SET mismatch at output {i}"
            );
        }
        checked += got3.len();
        println!("parity {name} SET bit-exact outputs={}", got3.len());
        let (p, c) = timing_pair(&gpu, probe_name, control_set, &mut a3, &mut a4, m, n);
        println!("{name},SET,{m},{k},{n},{p:.5},{c:.5},{:.4}", p / c);

        launch(&gpu, probe_name, &mut a3, m, n);
        launch(&gpu, control_set, &mut a4, m, n);
        let mut a3_add = args(&t3, &tx, &y3, m, k, n, Some(1));
        launch(&gpu, probe_name, &mut a3_add, m, n);
        launch(&gpu, control_add, &mut a4, m, n);
        gpu.hip.device_synchronize().unwrap();
        let got3 = gpu.download_f32(&y3).unwrap();
        let got4 = gpu.download_f32(&y4).unwrap();
        check_output(name, &got3, m, k, n, salt, true);
        for (i, (a, b)) in got3.iter().zip(&got4).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{name}: ADD mismatch at output {i}"
            );
        }
        checked += got3.len();
        println!("parity {name} ADD bit-exact outputs={}", got3.len());
        let (p, c) = timing_pair(&gpu, probe_name, control_add, &mut a3_add, &mut a4, m, n);
        println!("{name},ADD,{m},{k},{n},{p:.5},{c:.5},{:.4}", p / c);
    }
    assert!(checked > 0, "unknown family: {family}");
    println!("total cross-kernel bit-exact outputs={checked}");
}
