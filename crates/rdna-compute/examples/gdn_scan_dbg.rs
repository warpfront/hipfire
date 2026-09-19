// SPDX-License-Identifier: Apache-2.0
// Temporary scan debug probe (tiny shapes, value dumps).

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("requires --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    use rdna_compute::DType;

    const HD: usize = 128;
    const H: usize = 1;
    let t_arg = std::env::args().nth(1).and_then(|s| s.parse::<usize>().ok());
    let T: usize = t_arg.unwrap_or(8);

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    // One-hot probe: q/k = e_0, v = e_1, gate = 0 (alpha 1), beta = 1, S = 0.
    // Exact expectation: M[0,0] = 1, out_0 = e_1 (out_0[1] = 1, else 0).
    let mut q = vec![0f32; T * H * HD];
    let mut k = vec![0f32; T * H * HD];
    let mut v = vec![0f32; T * H * HD];
    for t in 0..T {
        q[t * HD] = 1.0;
        k[t * HD] = 1.0;
        v[t * HD + 1] = 1.0;
    }
    let gate = vec![0f32; T * H];
    let beta = vec![1f32; T * H];
    // Zero initial state -> codes all 0, scales 1, ef 0.
    let s_q8 = vec![0i8; H * HD * HD];
    let s_scales = vec![1f32; H * HD];
    let s_ef = vec![0u16; H * HD * HD];

    let q_gpu = gpu.upload_f32(&q, &[T, H * HD]).unwrap();
    let k_gpu = gpu.upload_f32(&k, &[T, H * HD]).unwrap();
    let v_gpu = gpu.upload_f32(&v, &[T, H * HD]).unwrap();
    let g_gpu = gpu.upload_f32(&gate, &[T, H]).unwrap();
    let b_gpu = gpu.upload_f32(&beta, &[T, H]).unwrap();

    let run_serial = || {
        let mut g = rdna_compute::Gpu::init().expect("GPU init");
        (g, 0)
    };
    let _ = run_serial;

    let out_s = gpu.zeros(&[T, H * HD], DType::F32).unwrap();
    let sq_s = upload_i8(&mut gpu, &s_q8, &[H * HD * HD]);
    let sc_s = gpu.upload_f32(&s_scales, &[H * HD]).unwrap();
    let ef_s = gpu.upload_f16_bits(&s_ef, &[H * HD * HD]).unwrap();
    gpu.gated_delta_net_q8_batch_seq(
        &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &sq_s, &sc_s, &out_s, T, H, HD,
        Some(&ef_s),
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let out_s_h = gpu.download_f32(&out_s).unwrap();
    let last = T - 1;
    println!("serial out[0][0..8]: {:?}", &out_s_h[0..8]);
    println!("serial out[L][0..8]: {:?}", &out_s_h[last * HD..last * HD + 8]);
    let out_w = gpu.zeros(&[T, H * HD], DType::F32).unwrap();
    let sq_w = upload_i8(&mut gpu, &s_q8, &[H * HD * HD]);
    let sc_w = gpu.upload_f32(&s_scales, &[H * HD]).unwrap();
    let ef_w = gpu.upload_f16_bits(&s_ef, &[H * HD * HD]).unwrap();
    gpu.gated_delta_net_q8_scan(
        &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &sq_w, &sc_w, &out_w, T, H, HD,
        &ef_w,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let out_w_h = gpu.download_f32(&out_w).unwrap();
    println!("scan   out[0][0..8]: {:?}", &out_w_h[0..8]);
    println!("scan   out[L][0..8]: {:?}", &out_w_h[last * HD..last * HD + 8]);
    let traj_s: Vec<f32> = (0..T).map(|t| out_s_h[t * HD + 1]).collect();
    let traj_w: Vec<f32> = (0..T).map(|t| out_w_h[t * HD + 1]).collect();
    println!("serial traj[1]: {:?}", traj_s);
    println!("scan   traj[1]: {:?}", traj_w);
}

#[cfg(feature = "deltanet")]
fn upload_i8(gpu: &mut rdna_compute::Gpu, data: &[i8], shape: &[usize]) -> rdna_compute::GpuTensor {
    let t = gpu.alloc_tensor(shape, rdna_compute::DType::Raw).unwrap();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
    t
}
