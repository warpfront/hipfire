// SPDX-License-Identifier: MIT OR Apache-2.0
// Stage-0 gfx1201 symmetric qt49 -> signed IU4 probe against production MQ4.
// Run with ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES restricted to card D.
use hip_bridge::KernargBlob;
use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const PRELUDE: &str = include_str!("../../../kernels/src/block_i4_128_quant.hip");
const PROBE: &str = include_str!("../../../kernels/src/gemm_mq3v25_iu4_probe.gfx12.hip");
const PRODUCTION: &str =
    include_str!("../../../kernels/src/gemm_mq3v25_residual_mmq_iu4.gfx12.hip");
const CONTROL: &str =
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip");
const PROBE_NAME: &str = "gemm_mq3v25_probe_iu4";
const CONTROL_NAME: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold_g12r";
const CONTROL_ADD: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold_g12r";
const PROBE_FUSED: &str = "gemm_mq3v25_gate_up_silu_mmq_iu4";
const CONTROL_FUSED: &str = "gemm_mq4g256v2_gate_up_silu_mmq_iu4_symfold_g12r";
const SHAPES: &[(&str, usize, usize)] = &[
    ("qkv", 10240, 5120),
    ("gate", 17408, 5120),
    ("up", 17408, 5120),
    ("out", 5120, 6144),
    ("down", 5120, 17408),
];

fn f16(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 31) as u32;
    let mant = (h & 1023) as u32;
    assert!(
        (1..31).contains(&exp),
        "fixture requires normal finite scales"
    );
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
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

// Both buffers derive from the same u and the same fp16 d. MQ4 stores code
// u+4, which its production symmetric-fold producer XORs with 8 to obtain
// the signed nibble (u-4)&15; qt49 stores u in 3 packed bits.
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
                let d4 = f16(d);
                let z3 = half_from_f32(-4.0 * d4);
                let z4 = half_from_f32(-8.0 * d4);
                assert_eq!(f16(z3), -4.0 * d4);
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
                    let wi = b + 8 + ix / 2;
                    w4[wi] |= (u + 4) << (4 * (ix % 2));
                }
                let bytes = bits.to_le_bytes();
                w3[a + 8 + octet * 3..a + 11 + octet * 3].copy_from_slice(&bytes[..3]);
            }
        }
    }
    (w3, w4)
}
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
fn check_codes() {
    for u in 0u8..8 {
        assert_eq!(((u + 4) ^ 8) & 15, ((u as i32 - 4) & 15) as u8);
    }
}
fn expected_at(row: usize, token: usize, k: usize, salt: usize) -> f32 {
    let mut expected = 0f32;
    for kb in 0..k / 128 {
        let g = kb / 2;
        let h = kb % 2;
        let sc = f16(scale(row, g, h));
        let d = act_scale(token, kb);
        let mut dot = 0i32;
        for i in 0..128 {
            dot += (code(row, g, h * 128 + i, salt) as i32 - 4) * activation(token, kb, i);
        }
        expected += (sc * d) * dot as f32;
    }
    expected
}
fn check_output(label: &str, y: &[f32], m: usize, k: usize, n: usize, salt: usize, factor: f32) {
    // Independent CPU per-K128 i32 dot, then the production symmetric
    // RN DAG: t1=RN(d_w*d_x), p=RN(t1*float(dot)), sum=RN(sum+p).
    // Strided rows and tokens exercise every CTA and both K-half scales.
    let rows: Vec<usize> = [0, 1, 7, 15, 16, 31, 63, 127, m / 2, m - 1]
        .into_iter()
        .filter(|&row| row < m)
        .collect();
    let tokens: Vec<usize> = [0, 1, 15, 64, 127, 128, n / 2, n - 1]
        .into_iter()
        .filter(|&token| token < n)
        .collect();
    let mut max_abs = 0f32;
    for &row in &rows {
        for &token in &tokens {
            let expected = factor * expected_at(row, token, k, salt);
            let got = y[token * m + row];
            assert!(
                got.is_finite(),
                "{label}: nonfinite token={token} row={row}"
            );
            let err = (got - expected).abs();
            max_abs = max_abs.max(err);
            assert!(
                err <= 0.000025 * expected.abs().max(1.0),
                "{label}: token={token} row={row}: GPU={got} CPU={expected} err={err}"
            );
        }
    }
    println!(
        "oracle {label} checked={} max_abs={max_abs:.3e}",
        rows.len() * tokens.len()
    );
}
fn check_fused(label: &str, y: &[f32], m: usize, k: usize, n: usize) {
    let rows: Vec<usize> = [0, 1, 15, 16, 63, 64, m / 2, m - 1]
        .into_iter()
        .filter(|&row| row < m)
        .collect();
    let tokens: Vec<usize> = [0, 1, 64, 127, 128, n - 1]
        .into_iter()
        .filter(|&token| token < n)
        .collect();
    let mut max_abs = 0f32;
    for &row in &rows {
        for &token in &tokens {
            let g = expected_at(row, token, k, 0);
            let u = expected_at(row, token, k, 3);
            let expected = g / (1.0 + (-g).exp()) * u;
            let got = y[token * m + row];
            let err = (got - expected).abs();
            max_abs = max_abs.max(err);
            assert!(
                got.is_finite() && err <= 0.00005 * expected.abs().max(1.0),
                "{label} CPU mismatch token={token} row={row} got={got} expected={expected}"
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
    add: i32,
) -> Vec<u8> {
    let mut b = KernargBlob::new();
    b.push_ptr(w.buf.as_ptr());
    b.push_ptr(x.buf.as_ptr());
    b.push_ptr(y.buf.as_ptr());
    b.push_i32(m as i32);
    b.push_i32(k as i32);
    b.push_i32(n as i32);
    b.push_i32(add);
    b.into_vec()
}
fn fused_args(
    g: &GpuTensor,
    u: &GpuTensor,
    x: &GpuTensor,
    h: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<u8> {
    let mut b = KernargBlob::new();
    b.push_ptr(g.buf.as_ptr());
    b.push_ptr(u.buf.as_ptr());
    b.push_ptr(x.buf.as_ptr());
    b.push_ptr(h.buf.as_ptr());
    b.push_i32(m as i32);
    b.push_i32(k as i32);
    b.push_i32(n as i32);
    b.into_vec()
}
fn launch(gpu: &Gpu, name: &str, arg: &mut [u8], m: usize, n: usize) {
    gpu.launch_kernel_blob(
        name,
        [(m.div_ceil(128)) as u32, (n.div_ceil(128)) as u32, 1],
        [256, 1, 1],
        20480,
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
fn median(v: &mut [f64; 5]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[2]
}
fn timing_pair(
    gpu: &Gpu,
    probe: &str,
    control: &str,
    a3: &mut [u8],
    a4: &mut [u8],
    grid_m: usize,
    n: usize,
) -> (f64, f64) {
    for _ in 0..32 {
        launch(gpu, probe, a3, grid_m, n);
        launch(gpu, control, a4, grid_m, n);
    }
    gpu.hip.device_synchronize().unwrap();
    let mut t_probe = [0.0; 5];
    let mut t_control = [0.0; 5];
    for i in 0..5 {
        if i % 2 == 0 {
            t_probe[i] = measure(gpu, probe, a3, grid_m, n);
            t_control[i] = measure(gpu, control, a4, grid_m, n);
        } else {
            t_control[i] = measure(gpu, control, a4, grid_m, n);
            t_probe[i] = measure(gpu, probe, a3, grid_m, n);
        }
    }
    eprintln!("samples {probe}={t_probe:?} {control}={t_control:?}");
    (median(&mut t_probe), median(&mut t_control))
}
fn main() {
    let mut arch = "gfx1201".to_string();
    let mut family = "all".to_string();
    let mut n = 8192usize;
    let mut m_override = None;
    let mut variant = "original".to_string();
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        let value = iter.next().expect("option needs value");
        match flag.as_str() {
            "--arch" => arch = value,
            "--family" => family = value,
            "--n" => n = value.parse().expect("--n integer"),
            "--m" => m_override = Some(value.parse::<usize>().expect("--m integer")),
            "--variant" => variant = value,
            _ => panic!("unknown option {flag}"),
        }
    }
    assert_eq!(arch, "gfx1201", "stage-0 is exact gfx1201 only");
    assert!(n > 0);
    check_codes();
    let mut gpu = Gpu::init().expect("HIP init");
    assert_eq!(gpu.arch, arch, "must use exact requested GPU arch");
    let unpack = match variant.as_str() {
        "original" => "",
        "staged" => "#define IU4_UNPACK_STAGED 1\n",
        "direct" => "#define IU4_UNPACK_DIRECT 1\n",
        "production" => "",
        _ => panic!("unknown unpack variant {variant}"),
    };
    let kernel_source = if variant == "production" {
        PRODUCTION
    } else {
        PROBE
    };
    let probe_name = if variant == "production" {
        "gemm_mq3v25_residual_mmq_iu4"
    } else {
        PROBE_NAME
    };
    let source_probe = format!(
        "#define IU4_SYMMETRIC_FOLD 1\n#define IU4_G12_RASTER 1\n{unpack}{PRELUDE}{kernel_source}"
    );
    let source_control = format!("#define IU4_SYMMETRIC_FOLD 1\n#define IU4_G12_RASTER 1\n#define gemm_mq4g256v2_residual_mmq_iu4_full_set {CONTROL_NAME}\n#define gemm_mq4g256v2_residual_mmq_iu4_full_add {CONTROL_ADD}\n#define gemm_mq4g256v2_gate_up_silu_mmq_iu4 {CONTROL_FUSED}\n{PRELUDE}{CONTROL}");
    let probe_module = format!("mq3v25_probe_sym_g12r_{variant}");
    gpu.ensure_kernel_public(&probe_module, &source_probe, probe_name)
        .expect("compile qt49 SET/ADD");
    gpu.ensure_kernel_public(&probe_module, &source_probe, PROBE_FUSED)
        .expect("compile qt49 fused");
    gpu.ensure_kernel_public("mq4v2_prod_sym_g12r", &source_control, CONTROL_NAME)
        .expect("compile tuned MQ4");
    gpu.ensure_kernel_public("mq4v2_prod_sym_g12r", &source_control, CONTROL_ADD)
        .expect("compile MQ4 ADD");
    gpu.ensure_kernel_public("mq4v2_prod_sym_g12r", &source_control, CONTROL_FUSED)
        .expect("compile MQ4 fused");
    let occupancy_probe = gpu
        .occupancy_max_active_blocks(probe_name, [256, 1, 1], 20480)
        .expect("probe occupancy");
    let occupancy_control = gpu
        .occupancy_max_active_blocks(CONTROL_NAME, [256, 1, 1], 20480)
        .expect("control occupancy");
    println!("HIP occupancy active CTA/CU probe={occupancy_probe} MQ4={occupancy_control}");
    println!("arch={} N={n} variant={variant} probe={probe_name} control={CONTROL_NAME} tile=128x128 block=256 LDS=20480 symfold_g12r", gpu.arch);
    println!("family,mode,M,K,N,qt49_ms,mq4_ms,ratio");
    for &(name, default_m, k) in SHAPES {
        if family != "all" && family != name {
            continue;
        }
        let m = m_override.unwrap_or(default_m);
        assert!(m > 0);
        let salt = if name == "up" { 3 } else { 0 };
        let producer = Instant::now();
        let (w3, w4) = weights(m, k, salt);
        let weight_ms = producer.elapsed().as_secs_f64() * 1e3;
        let producer = Instant::now();
        let x = activations(k, n);
        let activation_ms = producer.elapsed().as_secs_f64() * 1e3;
        eprintln!("producer {name} CPU weight={weight_ms:.2}ms activation={activation_ms:.2}ms (excluded from GPU timing), qt49={}B qt44={}B Xq={}B", w3.len(), w4.len(), x.len());
        let t3 = gpu.upload_raw(&w3, &[w3.len()]).unwrap();
        let t4 = gpu.upload_raw(&w4, &[w4.len()]).unwrap();
        let tx = gpu.upload_raw(&x, &[x.len()]).unwrap();
        let y3 = gpu.zeros(&[n, m], DType::F32).unwrap();
        let y4 = gpu.zeros(&[n, m], DType::F32).unwrap();
        let mut a3 = args(&t3, &tx, &y3, m, k, n, 0);
        let mut a4 = args(&t4, &tx, &y4, m, k, n, 0);
        launch(&gpu, probe_name, &mut a3, m, n);
        launch(&gpu, CONTROL_NAME, &mut a4, m, n);
        gpu.hip.device_synchronize().unwrap();
        let got3 = gpu.download_f32(&y3).unwrap();
        let got4 = gpu.download_f32(&y4).unwrap();
        check_output(name, &got3, m, k, n, salt, 1.0);
        check_output("MQ4 control", &got4, m, k, n, salt, 1.0);
        for (i, (a, b)) in got3.iter().zip(&got4).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{name}: SET mismatch at output {i}"
            );
        }
        println!("cross-kernel {name} SET bit-exact outputs={}", got3.len());
        let (p, c) = timing_pair(&gpu, probe_name, CONTROL_NAME, &mut a3, &mut a4, m, n);
        println!("{name},SET,{m},{k},{n},{p:.5},{c:.5},{:.4}", p / c);

        // Reinitialize both outputs with SET before checking ADD. The
        // production control has a dedicated ADD entry; qt49 uses add=1.
        launch(&gpu, probe_name, &mut a3, m, n);
        launch(&gpu, CONTROL_NAME, &mut a4, m, n);
        let mut a3_add = args(&t3, &tx, &y3, m, k, n, 1);
        let mut a4_add = args(&t4, &tx, &y4, m, k, n, 1);
        launch(&gpu, probe_name, &mut a3_add, m, n);
        launch(&gpu, CONTROL_ADD, &mut a4_add, m, n);
        gpu.hip.device_synchronize().unwrap();
        let got3 = gpu.download_f32(&y3).unwrap();
        let got4 = gpu.download_f32(&y4).unwrap();
        check_output(name, &got3, m, k, n, salt, 2.0);
        for (i, (a, b)) in got3.iter().zip(&got4).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{name}: ADD mismatch at output {i}"
            );
        }
        println!("cross-kernel {name} ADD bit-exact outputs={}", got3.len());
        let (p, c) = timing_pair(
            &gpu,
            probe_name,
            CONTROL_ADD,
            &mut a3_add,
            &mut a4_add,
            m,
            n,
        );
        println!("{name},ADD,{m},{k},{n},{p:.5},{c:.5},{:.4}", p / c);

        if name == "gate" {
            let (w3u, w4u) = weights(m, k, 3);
            let t3u = gpu.upload_raw(&w3u, &[w3u.len()]).unwrap();
            let t4u = gpu.upload_raw(&w4u, &[w4u.len()]).unwrap();
            let h3 = gpu.zeros(&[n, m], DType::F32).unwrap();
            let h4 = gpu.zeros(&[n, m], DType::F32).unwrap();
            let mut f3 = fused_args(&t3, &t3u, &tx, &h3, m, k, n);
            let mut f4 = fused_args(&t4, &t4u, &tx, &h4, m, k, n);
            launch(&gpu, PROBE_FUSED, &mut f3, 2 * m, n);
            if m % 64 != 0 {
                gpu.hip.device_synchronize().unwrap();
                let got3 = gpu.download_f32(&h3).unwrap();
                check_fused("qt49 fused tail", &got3, m, k, n);
                println!("gate/up fused tail CPU-checked outputs={}", got3.len());
                continue;
            }
            launch(&gpu, CONTROL_FUSED, &mut f4, 2 * m, n);
            gpu.hip.device_synchronize().unwrap();
            let got3 = gpu.download_f32(&h3).unwrap();
            let got4 = gpu.download_f32(&h4).unwrap();
            check_fused("qt49 fused", &got3, m, k, n);
            check_fused("MQ4 fused", &got4, m, k, n);
            for (i, (a, b)) in got3.iter().zip(&got4).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "gate/up fused mismatch at output {i}"
                );
            }
            println!(
                "cross-kernel gate/up fused bit-exact outputs={}",
                got3.len()
            );
            let (p, c) = timing_pair(&gpu, PROBE_FUSED, CONTROL_FUSED, &mut f3, &mut f4, 2 * m, n);
            println!("gate/up,FUSED,{m},{k},{n},{p:.5},{c:.5},{:.4}", p / c);
        }
    }
}
