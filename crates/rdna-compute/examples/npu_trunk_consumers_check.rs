// SPDX-License-Identifier: Apache-2.0
//! gfx1151 byte oracle for every direct-arena trunk reader; no NPU required.
use hip_bridge::KernargBlob;
use rdna_compute::{
    rope::{Rope, RopeScaling, RopeYarn, Yarn},
    tensor_ops::*,
    DType, Gpu, GpuTensor,
};

const TENSOR_SRC: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/rope_yarn.h"),
    include_str!("../../../kernels/src/tensor_ops.hip")
);
const CHUNK_SRC: &str = include_str!("../../../kernels/src/gated_delta_chunk_wmma.gfx1151.hip");
const Q8_SRC: &str = include_str!("../../../kernels/src/gated_delta_chunk_q8_wmma.gfx1151.hip");
const CHUNK_ARENA_SRC: &str = concat!("#define NPU_TRUNK_ARENA 1\n#define gated_delta_chunk_gate_wmma gated_delta_chunk_gate_wmma_arena\n", include_str!("../../../kernels/src/gated_delta_chunk_wmma.gfx1151.hip"));
const Q8_ARENA_SRC: &str = concat!("#define NPU_TRUNK_ARENA 1\n#define gated_delta_chunk_gate_q8_wmma gated_delta_chunk_gate_q8_wmma_arena\n", include_str!("../../../kernels/src/gated_delta_chunk_q8_wmma.gfx1151.hip"));

fn typed(g: &mut Gpu, bytes: &[u8], n: usize, dtype: DType) -> GpuTensor {
    assert_eq!(bytes.len(), n * dtype.size());
    let mut t = g.upload_raw(bytes, &[n]).unwrap();
    t.dtype = dtype;
    t
}
fn filled(g: &mut Gpu, n: usize, dtype: DType) -> GpuTensor {
    typed(g, &vec![0xa5; n * dtype.size()], n, dtype)
}
fn weights(g: &mut Gpu, n: usize, value: u16) -> GpuTensor {
    typed(g, &value.to_le_bytes().repeat(n), n, DType::BF16)
}
fn random(g: &mut Gpu, n: usize, dtype: DType, seed: u32) -> GpuTensor {
    let bits = random_bits(n, seed);
    let bytes: Vec<_> = bits
        .into_iter()
        .flat_map(|b| {
            if dtype == DType::BF16 {
                b.to_le_bytes().to_vec()
            } else {
                ((b as u32) << 16).to_le_bytes().to_vec()
            }
        })
        .collect();
    typed(g, &bytes, n, dtype)
}
fn random_bits(n: usize, mut seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            // Finite signed values, signed zero and subnormals; never arbitrary NaNs.
            if i % 257 == 0 {
                (seed & 0x8001) as u16
            } else {
                ((seed & 0x807f) | (0x3c00 + ((seed >> 8) % 7) * 128)) as u16
            }
        })
        .collect()
}
fn view(t: &GpuTensor, pitch: usize, col: usize) -> Bf16Arena<'_> {
    Bf16Arena {
        tensor: t,
        row_pitch_elems: pitch,
        col0_elems: col,
    }
}
fn snapshot(g: &Gpu, t: &GpuTensor) -> Vec<u8> {
    g.download_raw_bytes(t).unwrap()
}
fn equal(label: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len(), "{label}: byte length");
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        panic!("{label}: byte {i}: old={:02x} new={:02x}", a[i], b[i]);
    }
}
fn compare(g: &Gpu, label: &str, a: &GpuTensor, b: &GpuTensor) {
    equal(label, &snapshot(g, a), &snapshot(g, b));
}
fn timed(g: &mut Gpu, f: impl Fn(&mut Gpu)) -> f32 {
    // Warm the exact same operation once: JIT and allocations are excluded.
    // Stateful twins each advance twice from identical initial state.
    f(g);
    g.hip.device_synchronize().unwrap();
    let a = g.hip.event_create().unwrap();
    let b = g.hip.event_create().unwrap();
    g.hip.event_record(&a, g.active_stream.as_ref()).unwrap();
    f(g);
    g.hip.event_record(&b, g.active_stream.as_ref()).unwrap();
    g.hip.event_synchronize(&b).unwrap();
    let ms = g.hip.event_elapsed_ms(&a, &b).unwrap();
    g.hip.event_destroy(a).unwrap();
    g.hip.event_destroy(b).unwrap();
    ms
}
fn report(name: &str, rows: usize, pitch: usize, old: f32, new: f32) {
    println!("PASS kernel={name} rows={rows} pitch={pitch} old_with_materialize_ms={old:.4} direct_ms={new:.4}");
}
fn materialize(g: &mut Gpu, src: Bf16Arena<'_>, dst: &GpuTensor, rows: usize, cols: usize) {
    npu_trunk_copy_projection_arena(g, src, dst, rows, cols, cols, 0, 0).unwrap();
}
fn check_materialize(
    g: &mut Gpu,
    cpu: &[u16],
    arena: &GpuTensor,
    rows: usize,
    pitch: usize,
    col: usize,
    cols: usize,
    dtype: DType,
) -> GpuTensor {
    let dst = filled(g, rows * cols, dtype);
    materialize(g, view(arena, pitch, col), &dst, rows, cols);
    g.hip.device_synchronize().unwrap();
    let mut expected = Vec::with_capacity(rows * cols * dtype.size());
    for r in 0..rows {
        for c in col..col + cols {
            let b = cpu[r * pitch + c];
            if dtype == DType::BF16 {
                expected.extend(b.to_le_bytes());
            } else {
                expected.extend(((b as u32) << 16).to_le_bytes());
            }
        }
    }
    equal(
        "independent CPU exact materialization",
        &snapshot(g, &dst),
        &expected,
    );
    dst
}
fn conv<'a>(
    input: &'a GpuTensor,
    kernel: &'a GpuTensor,
    history: &'a GpuTensor,
    output: &'a GpuTensor,
    rows: usize,
) -> GatedDeltaConvBatched<'a> {
    GatedDeltaConvBatched {
        input,
        kernel,
        history,
        output,
        next_history: history,
        rows,
        channels: 10240,
        history_rows: 3,
        kernel_size: 4,
        start_cursor: 2,
    }
}
fn params<'a>(
    a: &'a GpuTensor,
    b: &'a GpuTensor,
    al: &'a GpuTensor,
    dt: &'a GpuTensor,
    gate: &'a GpuTensor,
    beta: &'a GpuTensor,
    rows: usize,
) -> GatedDeltaParamsBatched<'a> {
    GatedDeltaParamsBatched {
        a,
        b,
        a_log: al,
        dt_bias: dt,
        gate,
        beta,
        rows,
        heads: 48,
    }
}
fn step<'a>(
    projection: &'a GpuTensor,
    gate: &'a GpuTensor,
    beta: &'a GpuTensor,
    state: &'a GpuTensor,
    output: &'a GpuTensor,
    rows: usize,
) -> GatedDeltaStepBatched<'a> {
    GatedDeltaStepBatched {
        projection,
        gate,
        beta,
        state,
        format: GdnStateFormat::F32,
        residual: None,
        output,
        row_states: None,
        rows,
        qkv_width: 10240,
        key_heads: 16,
        value_heads: 48,
        key_dim: 128,
        value_dim: 128,
        position: 3,
    }
}
fn gdn(g: &mut Gpu, rows: usize) {
    let pitch = 16640;
    let cpu = random_bits(rows * pitch, 0x7345);
    let bytes: Vec<_> = cpu.iter().flat_map(|b| b.to_le_bytes()).collect();
    let arena = typed(g, &bytes, cpu.len(), DType::BF16);
    let qkv = check_materialize(g, &cpu, &arena, rows, pitch, 0, 10240, DType::BF16);
    let z = check_materialize(g, &cpu, &arena, rows, pitch, 10240, 6144, DType::F32);
    // ColumnMap's fixed family order is qkv, z, a, b.
    let a = check_materialize(g, &cpu, &arena, rows, pitch, 16384, 48, DType::F32);
    let b = check_materialize(g, &cpu, &arena, rows, pitch, 16432, 48, DType::F32);
    let al = weights(g, 48, 0xc040);
    let dt = weights(g, 48, 0x3e00);
    let norm = weights(g, 128, 0x3f80);
    let gates = [
        filled(g, rows * 48, DType::F32),
        filled(g, rows * 48, DType::F32),
    ];
    let betas = [
        filled(g, rows * 48, DType::F32),
        filled(g, rows * 48, DType::F32),
    ];
    let old = timed(g, |g| {
        materialize(g, view(&arena, pitch, 16384), &a, rows, 48);
        materialize(g, view(&arena, pitch, 16432), &b, rows, 48);
        gated_delta_params_batched(g, &params(&a, &b, &al, &dt, &gates[0], &betas[0], rows))
            .unwrap();
    });
    let new = timed(g, |g| {
        gated_delta_params_batched_arena(
            g,
            &params(&a, &b, &al, &dt, &gates[1], &betas[1], rows),
            view(&arena, pitch, 16384),
            view(&arena, pitch, 16432),
        )
        .unwrap()
    });
    compare(g, "gate parameters", &gates[0], &gates[1]);
    compare(g, "beta parameters", &betas[0], &betas[1]);
    report("params_flat/params_batched", rows, pitch, old, new);
    let kernel = random(g, 10240 * 4, DType::BF16, 17);
    let mut projection = None;
    for (dtype, fused) in [
        (DType::F32, false),
        (DType::BF16, false),
        (DType::F32, true),
        (DType::BF16, true),
    ] {
        let outputs = [
            filled(g, rows * 10240, dtype),
            filled(g, rows * 10240, dtype),
        ];
        let history = [
            random(g, 3 * 10240, DType::F32, 19),
            random(g, 3 * 10240, DType::F32, 19),
        ];
        let old = timed(g, |g| {
            materialize(g, view(&arena, pitch, 0), &qkv, rows, 10240);
            if fused {
                materialize(g, view(&arena, pitch, 16384), &a, rows, 48);
                materialize(g, view(&arena, pitch, 16432), &b, rows, 48);
                gated_delta_conv_params_batched(
                    g,
                    &conv(&qkv, &kernel, &history[0], &outputs[0], rows),
                    &params(&a, &b, &al, &dt, &gates[0], &betas[0], rows),
                )
                .unwrap();
            } else {
                gated_delta_conv_batched(g, &conv(&qkv, &kernel, &history[0], &outputs[0], rows))
                    .unwrap();
            }
        });
        let new = timed(g, |g| {
            if fused {
                gated_delta_conv_params_batched_arena(
                    g,
                    &conv(&arena, &kernel, &history[1], &outputs[1], rows),
                    &params(&a, &b, &al, &dt, &gates[1], &betas[1], rows),
                    view(&arena, pitch, 0),
                    view(&arena, pitch, 16384),
                    view(&arena, pitch, 16432),
                )
                .unwrap();
            } else {
                gated_delta_conv_batched_arena(
                    g,
                    &conv(&arena, &kernel, &history[1], &outputs[1], rows),
                    view(&arena, pitch, 0),
                )
                .unwrap();
            }
        });
        compare(g, "conv output", &outputs[0], &outputs[1]);
        compare(g, "conv in-place history", &history[0], &history[1]);
        if fused {
            compare(g, "fused parameters gate", &gates[0], &gates[1]);
            compare(g, "fused parameters beta", &betas[0], &betas[1]);
        }
        report(
            &format!("conv_batched {dtype:?} fused={fused}"),
            rows,
            pitch,
            old,
            new,
        );
        for t in history {
            g.free_tensor(t).unwrap();
        }
        let [o, n] = outputs;
        g.free_tensor(n).unwrap();
        if dtype == DType::BF16 && fused {
            projection = Some(o);
        } else {
            g.free_tensor(o).unwrap();
        }
    }
    let projection = projection.unwrap();
    // Fused Q/K normalization: compare scratch QN/KN as well as all V/Q/K,
    // parameter outputs and aliased convolution-history slots.
    for keep in [false, true] {
        let outputs = [
            filled(g, rows * 10240, DType::BF16),
            filled(g, rows * 10240, DType::BF16),
        ];
        let history = [
            random(g, 3 * 10240, DType::F32, 23),
            random(g, 3 * 10240, DType::F32, 23),
        ];
        let state = g.zeros(&[48 * 128 * 128], DType::F32).unwrap();
        let sp = step(&projection, &gates[0], &betas[0], &state, &projection, rows);
        let old = timed(g, |g| {
            materialize(g, view(&arena, pitch, 0), &qkv, rows, 10240);
            materialize(g, view(&arena, pitch, 16384), &a, rows, 48);
            materialize(g, view(&arena, pitch, 16432), &b, rows, 48);
            gated_delta_conv_params_qknorm_batched(
                g,
                &conv(&qkv, &kernel, &history[0], &outputs[0], rows),
                &params(&a, &b, &al, &dt, &gates[0], &betas[0], rows),
                &sp,
                keep,
            )
            .unwrap();
        });
        let scratch = g.qwen4_f16_x_scratch(2 * rows * 2048).unwrap();
        let old_qnk = snapshot(g, &scratch);
        let new = timed(g, |g| {
            gated_delta_conv_params_qknorm_batched_arena(
                g,
                &conv(&arena, &kernel, &history[1], &outputs[1], rows),
                &params(&a, &b, &al, &dt, &gates[1], &betas[1], rows),
                &sp,
                keep,
                view(&arena, pitch, 0),
                view(&arena, pitch, 16384),
                view(&arena, pitch, 16432),
            )
            .unwrap()
        });
        equal("qknorm QN/KN", &old_qnk, &snapshot(g, &scratch));
        compare(
            g,
            "qknorm output including unwritten Q/K",
            &outputs[0],
            &outputs[1],
        );
        compare(g, "qknorm history", &history[0], &history[1]);
        compare(g, "qknorm gate", &gates[0], &gates[1]);
        compare(g, "qknorm beta", &betas[0], &betas[1]);
        report(&format!("conv_qknorm keep={keep}"), rows, pitch, old, new);
        for t in outputs.into_iter().chain(history).chain([state]) {
            g.free_tensor(t).unwrap();
        }
    }
    let recurrent = [
        random(g, rows * 6144, DType::F32, 31),
        random(g, rows * 6144, DType::F32, 31),
    ];
    let outputs = [
        filled(g, rows * 6144, DType::F32),
        filled(g, rows * 6144, DType::F32),
    ];
    let gate = |i| GatedDeltaGateBatched {
        recurrent_output: &recurrent[i],
        z: &z,
        norm: &norm,
        output: &outputs[i],
        rows,
        value_heads: 48,
        value_dim: 128,
    };
    let old = timed(g, |g| {
        materialize(g, view(&arena, pitch, 10240), &z, rows, 6144);
        gated_delta_gate_batched(g, &gate(0)).unwrap();
    });
    let new = timed(g, |g| {
        gated_delta_gate_batched_arena(g, &gate(1), view(&arena, pitch, 10240)).unwrap()
    });
    compare(g, "gate output", &outputs[0], &outputs[1]);
    compare(g, "rounded recurrent output", &recurrent[0], &recurrent[1]);
    report("gate_batched", rows, pitch, old, new);
    let rotated = [
        filled(g, rows * 6144, DType::F32),
        filled(g, rows * 6144, DType::F32),
    ];
    let old = timed(g, |g| {
        materialize(g, view(&arena, pitch, 10240), &z, rows, 6144);
        gated_delta_gate_batched_rotate(g, &gate(0), &rotated[0]).unwrap();
    });
    let new = timed(g, |g| {
        gated_delta_gate_batched_rotate_arena(g, &gate(1), &rotated[1], view(&arena, pitch, 10240))
            .unwrap()
    });
    compare(g, "rotating gate output", &outputs[0], &outputs[1]);
    compare(
        g,
        "rotating gate recurrent state",
        &recurrent[0],
        &recurrent[1],
    );
    compare(g, "all FWHT rotated outputs", &rotated[0], &rotated[1]);
    report("gate_batched_rotate", rows, pitch, old, new);
    for t in rotated {
        g.free_tensor(t).unwrap();
    }
    for t in outputs.into_iter().chain(recurrent) {
        g.free_tensor(t).unwrap();
    }
    let scalar_z = filled(g, 6144, DType::F32);
    let recurrent = random(g, 6144, DType::F32, 37);
    let outputs = [filled(g, 6144, DType::F32), filled(g, 6144, DType::F32)];
    let gate = |i| GatedDeltaGate {
        recurrent_output: &recurrent,
        z: &scalar_z,
        norm: &norm,
        output: &outputs[i],
        value_heads: 48,
        value_dim: 128,
    };
    let before = snapshot(g, &recurrent);
    let old = timed(g, |g| {
        materialize(g, view(&arena, pitch, 10240), &scalar_z, 1, 6144);
        gated_delta_gate(g, &gate(0)).unwrap();
    });
    let new = timed(g, |g| {
        gated_delta_gate_arena(g, &gate(1), view(&arena, pitch, 10240)).unwrap()
    });
    compare(g, "scalar gate", &outputs[0], &outputs[1]);
    equal(
        "scalar recurrent remains immutable",
        &before,
        &snapshot(g, &recurrent),
    );
    report("gate", 1, pitch, old, new);
    for t in outputs.into_iter().chain([scalar_z, recurrent]) {
        g.free_tensor(t).unwrap();
    }
    // Exercise both chunk symbols, F32 and packed output, and Q8 codes/scales
    // plus the inline kernel's complete diagnostic recurrent-state capture.
    let qnk = g.qwen4_f16_x_scratch(2 * rows * 2048).unwrap();
    let qp = qnk.buf.as_ptr();
    let kp = unsafe { (qp as *mut u8).add(rows * 2048 * 2) } as *mut std::ffi::c_void;
    g.ensure_kernel_public("tensor_ops", TENSOR_SRC, "gated_delta_qk_norm_bf16_batched")
        .unwrap();
    let mut args = KernargBlob::new();
    args.push_ptr(projection.buf.as_ptr());
    args.push_ptr(qp);
    args.push_ptr(kp);
    for n in [rows, 10240, 16] {
        args.push_i32(n as i32);
    }
    args.pad_to(16);
    g.launch_kernel_blob(
        "gated_delta_qk_norm_bf16_batched",
        [rows as u32, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
    .unwrap();
    for (q8, mode) in [(false, 0), (true, 0), (true, 1), (true, 2)] {
        for dtype in [DType::F32, DType::BF16] {
            let state_bytes = if mode == 0 {
                random_bits(48 * 128 * 128, 41)
                    .into_iter()
                    .flat_map(|b| ((b as u32) << 16).to_le_bytes())
                    .collect()
            } else {
                let mut b: Vec<u8> = (0..48 * 128 * 128)
                    .map(|i| ((i * 73 % 255) as i32 - 127) as i8 as u8)
                    .collect();
                b.extend(0.001f32.to_le_bytes().repeat(48 * 128));
                if mode == 2 {
                    b.extend(vec![0xa5; 48 * 128 * 128 * 4]);
                }
                b
            };
            let states = [
                typed(g, &state_bytes, state_bytes.len(), DType::Raw),
                typed(g, &state_bytes, state_bytes.len(), DType::Raw),
            ];
            let outputs = [filled(g, rows * 6144, dtype), filled(g, rows * 6144, dtype)];
            let launch = |g: &mut Gpu, direct: bool, i: usize| {
                let (module, source, name) = match (q8, direct) {
                    (false, false) => (
                        "gated_delta_chunk_wmma",
                        CHUNK_SRC,
                        "gated_delta_chunk_gate_wmma",
                    ),
                    (false, true) => (
                        "gated_delta_chunk_wmma_arena",
                        CHUNK_ARENA_SRC,
                        "gated_delta_chunk_gate_wmma_arena",
                    ),
                    (true, false) => (
                        "gated_delta_chunk_q8_wmma",
                        Q8_SRC,
                        "gated_delta_chunk_gate_q8_wmma",
                    ),
                    (true, true) => (
                        "gated_delta_chunk_q8_wmma_arena",
                        Q8_ARENA_SRC,
                        "gated_delta_chunk_gate_q8_wmma_arena",
                    ),
                };
                g.ensure_kernel_public(module, source, name).unwrap();
                let mut args = KernargBlob::new();
                args.push_ptr(projection.buf.as_ptr());
                args.push_ptr(qp);
                args.push_ptr(kp);
                for t in [
                    &gates[0],
                    &betas[0],
                    &states[i],
                    if direct { &arena } else { &z },
                    &norm,
                    &outputs[i],
                ] {
                    args.push_ptr(t.buf.as_ptr());
                }
                for n in [rows, 10240, 16, 48] {
                    args.push_i32(n as i32);
                }
                args.push_f32(128f32.sqrt().recip());
                args.push_i32(i32::from(dtype == DType::BF16));
                if q8 {
                    args.push_i32(mode);
                    args.push_u32((rows + 2) as u32);
                }
                if direct {
                    args.push_i32(pitch as i32);
                    args.push_i32(10240);
                }
                args.pad_to(16);
                g.launch_kernel_blob(name, [48, 1, 1], [256, 1, 1], 0, args.as_mut_slice())
                    .unwrap();
            };
            let old = timed(g, |g| {
                materialize(g, view(&arena, pitch, 10240), &z, rows, 6144);
                launch(g, false, 0);
            });
            let new = timed(g, |g| launch(g, true, 1));
            compare(g, "chunk output", &outputs[0], &outputs[1]);
            compare(
                g,
                "complete recurrent state incl Q8 capture",
                &states[0],
                &states[1],
            );
            report(
                &format!(
                    "chunk{} state_mode={mode} {dtype:?}",
                    if q8 { "_q8" } else { "" }
                ),
                rows,
                pitch,
                old,
                new,
            );
            for t in outputs.into_iter().chain(states) {
                g.free_tensor(t).unwrap();
            }
        }
    }
    equal(
        "GDN source arena remains immutable",
        &bytes,
        &snapshot(g, &arena),
    );
    for t in [arena, qkv, z, a, b, al, dt, norm, kernel, projection]
        .into_iter()
        .chain(gates)
        .chain(betas)
    {
        g.free_tensor(t).unwrap();
    }
}

// qkv-only compact arena plus genuinely F32 native z/a/b. Keep low mantissa
// bits in scratch and check the incumbent core's BF16 rounding at consumption.
fn native_f32(g: &mut Gpu, n: usize, seed: u32) -> GpuTensor {
    let bytes: Vec<_> = random_bits(n, seed)
        .into_iter()
        .enumerate()
        .flat_map(|(i, b)| {
            (f32::from_bits((b as u32) << 16) + (i % 31) as f32 / 777.0).to_le_bytes()
        })
        .collect();
    typed(g, &bytes, n, DType::F32)
}
fn gdn_mixed(g: &mut Gpu, rows: usize) {
    let pitch = 10240;
    let cpu = random_bits(rows * pitch, 0xabc7);
    let bytes: Vec<_> = cpu.iter().flat_map(|b| b.to_le_bytes()).collect();
    let arena = typed(g, &bytes, cpu.len(), DType::BF16);
    let qkv = check_materialize(g, &cpu, &arena, rows, pitch, 0, 10240, DType::BF16);
    let a = native_f32(g, rows * 48, 47);
    let b = native_f32(g, rows * 48, 53);
    let z = native_f32(g, rows * 6144, 59);
    let al = weights(g, 48, 0xc040);
    let dt = weights(g, 48, 0x3e00);
    let norm = weights(g, 128, 0x3f80);
    let kernel = random(g, 10240 * 4, DType::BF16, 61);
    let gates = [
        filled(g, rows * 48, DType::F32),
        filled(g, rows * 48, DType::F32),
    ];
    let betas = [
        filled(g, rows * 48, DType::F32),
        filled(g, rows * 48, DType::F32),
    ];
    for (dtype, qknorm, keep) in [
        (DType::F32, false, false),
        (DType::BF16, false, false),
        (DType::BF16, true, false),
        (DType::BF16, true, true),
    ] {
        let outputs = [
            filled(g, rows * 10240, dtype),
            filled(g, rows * 10240, dtype),
        ];
        let history = [
            random(g, 3 * 10240, DType::F32, 67),
            random(g, 3 * 10240, DType::F32, 67),
        ];
        let state = g.zeros(&[48 * 128 * 128], DType::F32).unwrap();
        let sp = step(&qkv, &gates[0], &betas[0], &state, &qkv, rows);
        let old = timed(g, |g| {
            let c = conv(&qkv, &kernel, &history[0], &outputs[0], rows);
            let p = params(&a, &b, &al, &dt, &gates[0], &betas[0], rows);
            if qknorm {
                gated_delta_conv_params_qknorm_batched(g, &c, &p, &sp, keep).unwrap();
            } else {
                gated_delta_conv_params_batched(g, &c, &p).unwrap();
            }
        });
        let old_qnk = if qknorm {
            let t = g.qwen4_f16_x_scratch(2 * rows * 2048).unwrap();
            Some(snapshot(g, &t))
        } else {
            None
        };
        let new = timed(g, |g| {
            let c = conv(&qkv, &kernel, &history[1], &outputs[1], rows);
            let p = params(&a, &b, &al, &dt, &gates[1], &betas[1], rows);
            if qknorm {
                gated_delta_conv_params_qknorm_batched_mixed(
                    g,
                    &c,
                    &p,
                    &sp,
                    keep,
                    Some(view(&arena, pitch, 0)),
                    None,
                    None,
                )
                .unwrap();
            } else {
                gated_delta_conv_params_batched_mixed(
                    g,
                    &c,
                    &p,
                    Some(view(&arena, pitch, 0)),
                    None,
                    None,
                )
                .unwrap();
            }
        });
        if let Some(expected) = old_qnk {
            let t = g.qwen4_f16_x_scratch(2 * rows * 2048).unwrap();
            equal("mixed qknorm QN/KN", &expected, &snapshot(g, &t));
        }
        compare(g, "mixed convolution", &outputs[0], &outputs[1]);
        compare(g, "mixed convolution history", &history[0], &history[1]);
        compare(g, "mixed native alpha gate", &gates[0], &gates[1]);
        compare(g, "mixed native beta", &betas[0], &betas[1]);
        report(
            &format!("mixed qkv-arena native-zab {dtype:?} qknorm={qknorm} keep={keep}"),
            rows,
            pitch,
            old,
            new,
        );
        for t in outputs.into_iter().chain(history).chain([state]) {
            g.free_tensor(t).unwrap();
        }
    }
    let recurrent = [
        random(g, rows * 6144, DType::F32, 71),
        random(g, rows * 6144, DType::F32, 71),
    ];
    let outputs = [
        filled(g, rows * 6144, DType::F32),
        filled(g, rows * 6144, DType::F32),
    ];
    for i in 0..2 {
        let p = GatedDeltaGateBatched {
            recurrent_output: &recurrent[i],
            z: &z,
            norm: &norm,
            output: &outputs[i],
            rows,
            value_heads: 48,
            value_dim: 128,
        };
        gated_delta_gate_batched(g, &p).unwrap();
    }
    g.hip.device_synchronize().unwrap();
    compare(g, "mixed native z gate", &outputs[0], &outputs[1]);
    compare(
        g,
        "mixed native z recurrent rounding",
        &recurrent[0],
        &recurrent[1],
    );
    equal(
        "mixed immutable compact qkv arena",
        &bytes,
        &snapshot(g, &arena),
    );
    for t in [arena, qkv, a, b, z, al, dt, norm, kernel]
        .into_iter()
        .chain(gates)
        .chain(betas)
        .chain(outputs)
        .chain(recurrent)
    {
        g.free_tensor(t).unwrap();
    }
}
fn qsa(g: &mut Gpu, rows: usize) {
    let pitch = 14080;
    let cpu = random_bits(rows * pitch, 0x9713);
    let bytes: Vec<_> = cpu.iter().flat_map(|b| b.to_le_bytes()).collect();
    let arena = typed(g, &bytes, cpu.len(), DType::BF16);
    let qgate = check_materialize(g, &cpu, &arena, rows, pitch, 0, 12288, DType::F32);
    let key = check_materialize(g, &cpu, &arena, rows, pitch, 12288, 512, DType::F32);
    let value = check_materialize(g, &cpu, &arena, rows, pitch, 12800, 512, DType::F32);
    let index = check_materialize(g, &cpu, &arena, rows, pitch, 13312, 640, DType::F32);
    let mut normalized_key = None;
    let mut normalized_index = None;
    // YaRN factor 4 over 262,144 (Qwen3.8 README override), rotary width 64.
    let yarn64 = Rope {
        theta: 1e7,
        scaling: RopeScaling::Yarn(Yarn::new(4.0, 262_144, None, None, None, None)),
    }
    .yarn(64);
    for (label, col, width, heads, dim, stride, input) in [
        ("qgate", 0, 12288, 24, 256, 512, &qgate),
        ("k", 12288, 512, 2, 256, 256, &key),
        ("idx", 13312, 640, 4, 128, 128, &index),
    ] {
        let norm = weights(g, dim, 0x3e80);
        let out = filled(g, rows * width, DType::F32);
        let p = |values| IndexedAttentionNormRopeBatch {
            values,
            norm: &norm,
            rows,
            row_stride: width,
            heads,
            head_dim: dim,
            head_stride: stride,
            position_start: 3,
            rotary_dim: 64,
            yarn: RopeYarn::OFF,
        };
        // YaRN past the native window first; the plain pass below re-materializes `input`.
        let py = |values| IndexedAttentionNormRopeBatch {
            position_start: 300_003,
            yarn: yarn64,
            ..p(values)
        };
        let scaled = filled(g, rows * width, DType::F32);
        materialize(g, view(&arena, pitch, col), input, rows, width);
        indexed_attention_norm_rope_batch(g, &py(input)).unwrap();
        indexed_attention_norm_rope_batch_arena(g, &py(&scaled), view(&arena, pitch, col)).unwrap();
        compare(
            g,
            "YaRN norm/RoPE columns incl untouched gates/index key",
            input,
            &scaled,
        );
        let old = timed(g, |g| {
            materialize(g, view(&arena, pitch, col), input, rows, width);
            indexed_attention_norm_rope_batch(g, &p(input)).unwrap();
        });
        let new = timed(g, |g| {
            indexed_attention_norm_rope_batch_arena(g, &p(&out), view(&arena, pitch, col)).unwrap()
        });
        compare(
            g,
            "all norm/RoPE columns incl untouched gates/index key",
            input,
            &out,
        );
        report(&format!("norm_rope {label}"), rows, pitch, old, new);
        g.free_tensor(norm).unwrap();
        g.free_tensor(scaled).unwrap();
        if label == "k" {
            normalized_key = Some(out);
        } else if label == "idx" {
            normalized_index = Some(out);
        } else {
            g.free_tensor(out).unwrap();
        }
    }
    let normalized_key = normalized_key.unwrap();
    let normalized_index = normalized_index.unwrap();
    for format in [QsaKvFormat::F32, QsaKvFormat::Q8] {
        for direct_key in [false, true] {
            let units = format.kv_row_units(2, 256);
            let dtype = if format == QsaKvFormat::F32 {
                DType::F32
            } else {
                DType::Raw
            };
            let keys = [
                filled(g, (rows + 7) * units, dtype),
                filled(g, (rows + 7) * units, dtype),
            ];
            let values = [
                filled(g, (rows + 7) * units, dtype),
                filled(g, (rows + 7) * units, dtype),
            ];
            let p = |i| IndexedAttentionCacheAppendBatch {
                key: if direct_key { &key } else { &normalized_key },
                value: &value,
                full_keys: &keys[i],
                full_values: &values[i],
                rows,
                position_start: 3,
                kv_heads: 2,
                head_dim: 256,
                format,
            };
            let old = timed(g, |g| {
                if direct_key {
                    materialize(g, view(&arena, pitch, 12288), &key, rows, 512);
                }
                materialize(g, view(&arena, pitch, 12800), &value, rows, 512);
                indexed_attention_cache_append_batch(g, &p(0)).unwrap();
            });
            let new = timed(g, |g| {
                indexed_attention_cache_append_batch_arena(
                    g,
                    &p(1),
                    direct_key.then(|| view(&arena, pitch, 12288)),
                    view(&arena, pitch, 12800),
                )
                .unwrap()
            });
            compare(
                g,
                "all K cache bytes incl untouched prefix/suffix",
                &keys[0],
                &keys[1],
            );
            compare(
                g,
                "all V cache bytes incl scales and guards",
                &values[0],
                &values[1],
            );
            report(
                &format!("cache_append {format:?} direct_k={direct_key}"),
                rows,
                pitch,
                old,
                new,
            );
            for t in keys.into_iter().chain(values) {
                g.free_tensor(t).unwrap();
            }
        }
    }
    for dtype in [DType::BF16, DType::F32] {
        let raw = [
            filled(g, (rows + 7) * 128, dtype),
            filled(g, (rows + 7) * 128, dtype),
        ];
        let p = |i, index_rows| IndexedAttentionIndexKeyAppendBatch {
            index_rows,
            raw_index_keys: &raw[i],
            rows,
            index_q_width: 512,
            index_kv_width: 128,
            position_start: 3,
        };
        let old = timed(g, |g| {
            // Re-materialize only the key tail; query columns remain transformed.
            npu_trunk_copy_projection_arena(
                g,
                view(&arena, pitch, 13824),
                &index,
                rows,
                128,
                640,
                512,
                0,
            )
            .unwrap();
            if dtype == DType::BF16 {
                indexed_attention_index_key_append_batch(g, &p(0, &index)).unwrap();
            } else {
                g.bf16_round_trip_f32_strided(&index, rows, 512, 640, 128)
                    .unwrap();
                let tail = index.sub_offset(512, rows * 640 - 512);
                g.copy_rows_strided_f32(&tail, &raw[0], rows, 128, 640, 128, 3 * 128, Some(128))
                    .unwrap();
            }
        });
        let new = timed(g, |g| {
            indexed_attention_index_key_append_batch_arena(
                g,
                &p(1, &normalized_index),
                view(&arena, pitch, 13312),
            )
            .unwrap()
        });
        compare(g, "all index keys incl prefix/suffix", &raw[0], &raw[1]);
        compare(g, "index query+key output state", &index, &normalized_index);
        report(
            &format!("index_key_append {dtype:?}"),
            rows,
            pitch,
            old,
            new,
        );
        for t in raw {
            g.free_tensor(t).unwrap();
        }
    }
    equal(
        "source arena remains immutable",
        &bytes,
        &snapshot(g, &arena),
    );
    for t in [
        arena,
        qgate,
        key,
        value,
        index,
        normalized_key,
        normalized_index,
    ] {
        g.free_tensor(t).unwrap();
    }
}
fn run() {
    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx1151", "Halo-only oracle");
    for rows in [8192, 4647] {
        gdn(&mut gpu, rows);
        gdn_mixed(&mut gpu, rows);
        qsa(&mut gpu, rows);
    }
    println!("RESULT PASS: all consumer outputs/state byte-exact");
}
fn main() {
    if std::panic::catch_unwind(run).is_err() {
        eprintln!("RESULT FAIL");
        std::process::exit(1);
    }
}
