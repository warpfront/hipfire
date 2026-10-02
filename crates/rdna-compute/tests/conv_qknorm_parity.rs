// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201, gfx1100 and gfx1151: the decode-only fusion
//! `conv1d_silu_split_qknorm` must produce the same bytes as the unfused pair
//! it replaces (`conv1d_silu_split_f32` then `fused_qk_l2_norm_scale_f32`).
//! The Qwen3.5 lowered decode uses the fusion and the hand decode
//! (`HIPFIRE_FORWARD_LOWERED=0`, DFlash's per-token hidden-extracting
//! forward) uses the pair, so any difference makes the two decode paths
//! diverge. It did: the compiler contracted the conv tap sum and the first Q/K
//! square-sum add into FMAs in a different order in each kernel.
//!
//! Q, K, V and the conv state are compared bit for bit over four chained
//! decode steps (the state ring shifts each step), direct and replayed from a
//! captured hipGraph, at the Qwen3.6/3.8-27B shape (16 key heads, 48 value
//! heads), the Qwen3.5-A3B shape (16 / 32), and head counts one either side
//! of those (a value width of an odd number of 128-wide heads leaves the last
//! 256-thread V block half full).
//!
//! On gfx1100 the 27B decode runs the scalar-prep variant, which also folds
//! `fused_sigmoid_alpha_gate_f32` (beta sigmoid, alpha softplus gate) into one
//! extra workgroup. It is checked the same way against the three reference
//! kernels, including beta and alpha.
//!
//! Other arches run the original fused expressions, so the test skips there.
//!
//! `#[ignore]`d: needs a GPU with a working HIP toolchain. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --features deltanet \
//!       --test conv_qknorm_parity -- --ignored

#![cfg(feature = "deltanet")]

use rdna_compute::{Gpu, GpuTensor};

const HEAD_DIM: usize = 128;
const EPS: f32 = 1e-6;
const STEPS: usize = 4;

/// (key heads, value heads): production shapes, then ±1 head around them.
const SHAPES: [(usize, usize); 6] = [(16, 48), (16, 32), (15, 47), (17, 49), (16, 33), (1, 1)];

/// Arches whose fused kernel spells out the reference FMA order.
const EXACT_ARCHES: [&str; 3] = ["gfx1201", "gfx1100", "gfx1151"];

/// Deterministic values in [-scale, scale).
fn fill(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.download_f32(t)
        .expect("download")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn write(gpu: &Gpu, t: &GpuTensor, data: &[f32]) {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod");
}

/// Scalar-prep operands (gfx1100 variant): beta/alpha are transformed in place.
struct ScalarPrep<'a> {
    beta: &'a GpuTensor,
    alpha: &'a GpuTensor,
    dt_bias: &'a GpuTensor,
    a_log: &'a GpuTensor,
}

#[derive(Clone, Copy)]
struct Fused<'a> {
    q: &'a GpuTensor,
    k: &'a GpuTensor,
    v: &'a GpuTensor,
    input: &'a GpuTensor,
    weight: &'a GpuTensor,
    state: &'a GpuTensor,
    n_key_heads: usize,
    n_value_heads: usize,
    scalar_prep: Option<&'a ScalarPrep<'a>>,
}

fn launch_fused(gpu: &mut Gpu, f: &Fused) {
    let (k_dim, v_dim) = (f.n_key_heads * HEAD_DIM, f.n_value_heads * HEAD_DIM);
    let q_scale = 1.0 / (HEAD_DIM as f32).sqrt();
    match f.scalar_prep {
        None => gpu.conv1d_silu_split_qknorm(
            f.q, f.k, f.v, f.input, f.weight, f.state, k_dim, v_dim, f.n_key_heads, HEAD_DIM,
            q_scale, EPS,
        ),
        Some(p) => gpu.conv1d_silu_split_qknorm_scalar_prep_gfx1100(
            f.q, f.k, f.v, f.input, f.weight, f.state, p.beta, p.alpha, p.dt_bias, p.a_log,
            k_dim, v_dim, f.n_key_heads, HEAD_DIM, q_scale, EPS, f.n_value_heads,
        ),
    }
    .expect("conv1d_silu_split_qknorm");
}

fn check_shape(
    gpu: &mut Gpu,
    n_key_heads: usize,
    n_value_heads: usize,
    graph: bool,
    scalar_prep: bool,
) {
    let k_dim = n_key_heads * HEAD_DIM;
    let v_dim = n_value_heads * HEAD_DIM;
    let channels = 2 * k_dim + v_dim;
    let mut seed = 0x5eed_0000 + (n_key_heads * 1000 + n_value_heads) as u64;
    let alloc = |gpu: &mut Gpu, n: usize| gpu.upload_f32(&vec![0.0; n], &[n]).unwrap();

    let weight = alloc(gpu, channels * 4);
    write(gpu, &weight, &fill(&mut seed, channels * 4, 0.5));
    let state0 = fill(&mut seed, channels * 3, 2.0);
    let (state_ref, state_fused) = (alloc(gpu, channels * 3), alloc(gpu, channels * 3));
    write(gpu, &state_ref, &state0);
    write(gpu, &state_fused, &state0);
    let input = alloc(gpu, channels);
    let (q_ref, k_ref, v_ref) = (alloc(gpu, k_dim), alloc(gpu, k_dim), alloc(gpu, v_dim));
    let (q_fused, k_fused, v_fused) = (alloc(gpu, k_dim), alloc(gpu, k_dim), alloc(gpu, v_dim));
    // Scalar-prep operands. dt_bias and a_log are constant; the raw beta and
    // alpha are rewritten each step. Alpha spans all three softplus branches.
    let (dt_bias, a_log) = (alloc(gpu, n_value_heads), alloc(gpu, n_value_heads));
    write(gpu, &dt_bias, &fill(&mut seed, n_value_heads, 2.0));
    write(gpu, &a_log, &fill(&mut seed, n_value_heads, 2.0));
    let (beta_ref, alpha_ref) = (alloc(gpu, n_value_heads), alloc(gpu, n_value_heads));
    let (beta_fused, alpha_fused) = (alloc(gpu, n_value_heads), alloc(gpu, n_value_heads));
    let prep = ScalarPrep { beta: &beta_fused, alpha: &alpha_fused, dt_bias: &dt_bias, a_log: &a_log };
    let fused = Fused {
        q: &q_fused,
        k: &k_fused,
        v: &v_fused,
        input: &input,
        weight: &weight,
        state: &state_fused,
        n_key_heads,
        n_value_heads,
        scalar_prep: scalar_prep.then_some(&prep),
    };

    if graph {
        // JIT outside the capture on throwaway state, then record one launch.
        let scratch = alloc(gpu, channels * 3);
        let (scratch_beta, scratch_alpha) = (alloc(gpu, n_value_heads), alloc(gpu, n_value_heads));
        let scratch_prep =
            ScalarPrep { beta: &scratch_beta, alpha: &scratch_alpha, dt_bias: &dt_bias, a_log: &a_log };
        launch_fused(
            gpu,
            &Fused { state: &scratch, scalar_prep: scalar_prep.then_some(&scratch_prep), ..fused },
        );
        gpu.hip.device_synchronize().unwrap();
        for t in [scratch, scratch_beta, scratch_alpha] {
            gpu.free_tensor(t).unwrap();
        }
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create().unwrap());
        }
        let stream = gpu.active_stream.take().unwrap();
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, &stream)
            .unwrap();
        gpu.active_stream = Some(stream);
        launch_fused(gpu, &fused);
        let stream = gpu.active_stream.take().unwrap();
        gpu.graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, &stream)
            .unwrap();
        gpu.active_stream = Some(stream);
    }

    let variant = match (scalar_prep, graph) {
        (false, false) => "direct",
        (false, true) => "graph replay",
        (true, false) => "scalar prep, direct",
        (true, true) => "scalar prep, graph replay",
    };
    for step in 0..STEPS {
        gpu.hip.device_synchronize().unwrap();
        write(gpu, &input, &fill(&mut seed, channels, 2.0));
        let beta_raw = fill(&mut seed, n_value_heads, 8.0);
        let alpha_raw = fill(&mut seed, n_value_heads, 30.0);
        for (reference, fused, raw) in
            [(&beta_ref, &beta_fused, &beta_raw), (&alpha_ref, &alpha_fused, &alpha_raw)]
        {
            write(gpu, reference, raw);
            write(gpu, fused, raw);
        }
        if scalar_prep {
            gpu.fused_sigmoid_alpha_gate_f32(&beta_ref, &alpha_ref, &dt_bias, &a_log, n_value_heads)
                .unwrap();
        }
        gpu.conv1d_silu_split_f32(
            &q_ref, &k_ref, &v_ref, &input, &weight, &state_ref, k_dim, v_dim,
        )
        .unwrap();
        gpu.fused_qk_l2_norm_scale_f32(
            &q_ref,
            &k_ref,
            n_key_heads,
            HEAD_DIM,
            1.0 / (HEAD_DIM as f32).sqrt(),
            EPS,
        )
        .unwrap();
        if graph {
            let stream = gpu.active_stream.as_ref().unwrap();
            gpu.graphs
                .graph_launch(&gpu.hip, gpu.device_id, stream)
                .unwrap();
        } else {
            launch_fused(gpu, &fused);
        }
        gpu.hip.device_synchronize().unwrap();
        let mut tensors = vec![
            ("q", &q_ref, &q_fused),
            ("k", &k_ref, &k_fused),
            ("v", &v_ref, &v_fused),
            ("conv state", &state_ref, &state_fused),
        ];
        if scalar_prep {
            tensors.extend([("beta", &beta_ref, &beta_fused), ("alpha", &alpha_ref, &alpha_fused)]);
        }
        for (name, reference, got) in tensors {
            let (r, f) = (bits(gpu, reference), bits(gpu, got));
            let differing = r.iter().zip(&f).filter(|(a, b)| a != b).count();
            assert_eq!(
                differing,
                0,
                "{} {n_key_heads}/{n_value_heads} heads, {variant}, step {step}: {differing} of {} \
                 {name} floats differ between the fused kernel and the unfused reference kernels",
                gpu.arch,
                r.len()
            );
        }
    }
    if graph {
        gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
        if let Some(stream) = gpu.active_stream.take() {
            gpu.hip.stream_destroy(stream).unwrap();
        }
    }
    for t in [
        weight, state_ref, state_fused, input, q_ref, k_ref, v_ref, q_fused, k_fused, v_fused,
        dt_bias, a_log, beta_ref, alpha_ref, beta_fused, alpha_fused,
    ] {
        gpu.free_tensor(t).unwrap();
    }
}

#[test]
#[ignore = "needs a GPU"]
fn conv_qknorm_fusion_matches_unfused_pair_bit_for_bit() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !EXACT_ARCHES.contains(&gpu.arch.as_str()) {
        eprintln!("skip: {} keeps the original fused expressions", gpu.arch);
        return;
    }
    // The scalar-prep variant is dispatched on gfx1100 only.
    let scalar_prep_modes: &[bool] = if gpu.arch_caps.is_gfx1100() { &[false, true] } else { &[false] };
    for &scalar_prep in scalar_prep_modes {
        for (n_key_heads, n_value_heads) in SHAPES {
            check_shape(&mut gpu, n_key_heads, n_value_heads, false, scalar_prep);
            check_shape(&mut gpu, n_key_heads, n_value_heads, true, scalar_prep);
        }
    }
}
