// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Q8Ef request-state lifecycle oracle (CHUNK-CANONICAL-DESIGN §4.3, G2).
//!
//! A real `Qwen4State` in Q8Ef with a nonzero slot, residual and convolution
//! history takes a snapshot ticket, an armed verify window of `R` rows
//! (`gated_delta_step_batched` capture + `commit_row_capture`), and then, per
//! `keep` in {0, 1, mid, R-1, R}, either keeps the whole window (`keep == R`),
//! rolls it back (`rollback_rows_retain`) or restores the ticket (`keep == 0`).
//! Checks (layer 0, byte exact):
//!   * the kept state equals `keep` single-row AR steps from the initial
//!     state: Q8 slot, scales and the REAL F16 residual (not zeroed), and the
//!     convolution history;
//!   * `restore` of the ticket returns slot, residual and conv to the initial
//!     bytes exactly;
//!   * one further AR step on the restored state equals the uninterrupted
//!     chain's next step.
//! Fails on the base tree (restore and partial rollback zero the residual).

use super::*;
use rdna_compute::tensor_ops::{gated_delta_step_batched, GatedDeltaStepBatched};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xffff) as f32 / 32768.0 - 1.0
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn up(gpu: &Gpu, t: &GpuTensor, bytes: &[u8]) {
    gpu.hip.device_synchronize().unwrap();
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
}

fn down(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    gpu.hip.device_synchronize().unwrap();
    let mut host = vec![0u8; t.byte_size()];
    gpu.hip.memcpy_dtoh(&mut host, &t.buf).unwrap();
    host
}

#[test]
fn q8ef_snapshot_restore_rollback_continue_is_byte_exact_hw() {
    let Ok(mut gpu) = Gpu::init() else {
        eprintln!("skip: needs a GPU");
        return;
    };
    const R: usize = 8;
    let mut config = crate::config::compact_test_config();
    config.ple = None;
    let (kh, vh) = (config.linear_num_key_heads, config.linear_num_value_heads);
    let qkv = 2 * kh * 128 + vh * 128;
    let value = vh * 128;
    let format = Qwen4StateFormat {
        gdn: GdnStateFormat::Q8Ef,
        ..Qwen4StateFormat::F32
    };
    let mut state = Qwen4State::new(&mut gpu, &config, 64, format).expect("state");
    state.ensure_row_capture(&mut gpu, R, 3).expect("ring");
    let mut rng = Rng(0x0ef0_5eed);
    // Initial layer state: random codes, positive scales, small residual, conv.
    let units = state.gdn[0].recurrent.numel();
    let mut slot0: Vec<u8> = (0..vh * 128 * 128).map(|_| (rng.next() & 0xff) as u8).collect();
    slot0.extend(f32_bytes(
        &(0..vh * 128).map(|_| 0.002 + 0.001 * (rng.unit() * 0.5 + 0.5)).collect::<Vec<_>>(),
    ));
    assert_eq!(slot0.len(), units);
    let ef0: Vec<u8> = (0..vh * 128 * 128)
        .flat_map(|_| half::f16::from_f32(rng.unit() * 0.001).to_bits().to_le_bytes())
        .collect();
    let conv0 = f32_bytes(
        &(0..state.gdn[0].conv.numel()).map(|_| rng.unit()).collect::<Vec<_>>(),
    );
    for layer in &state.gdn {
        up(&gpu, &layer.recurrent, &slot0);
        up(&gpu, layer.ef_residual.as_ref().unwrap(), &ef0);
        up(&gpu, &layer.conv, &conv0);
    }
    // Window inputs (convolution output rows, gate, beta) and conv input rows.
    let proj = f32_bytes(&(0..R * qkv).map(|_| rng.unit()).collect::<Vec<_>>());
    let gate = f32_bytes(
        &(0..R * vh).map(|_| -0.02 - 0.5 * (rng.unit() * 0.5 + 0.5)).collect::<Vec<_>>(),
    );
    let beta = f32_bytes(
        &(0..R * vh).map(|_| 0.2 + 0.7 * (rng.unit() * 0.5 + 0.5)).collect::<Vec<_>>(),
    );
    let inputs = f32_bytes(&(0..R * qkv).map(|_| rng.unit()).collect::<Vec<_>>());
    let proj_t = gpu.zeros(&[R * qkv], DType::F32).unwrap();
    let gate_t = gpu.zeros(&[R * vh], DType::F32).unwrap();
    let beta_t = gpu.zeros(&[R * vh], DType::F32).unwrap();
    let out_t = gpu.zeros(&[R * value], DType::F32).unwrap();
    up(&gpu, &proj_t, &proj);
    up(&gpu, &gate_t, &gate);
    up(&gpu, &beta_t, &beta);
    // AR reference chain: refs[k] = (slot, residual) after k single-row steps.
    let slot_ref = gpu.zeros(&[units], DType::Raw).unwrap();
    let ef_ref = gpu.zeros(&[vh * 128 * 128], DType::F16).unwrap();
    up(&gpu, &slot_ref, &slot0);
    up(&gpu, &ef_ref, &ef0);
    let ar_step = |gpu: &mut Gpu, slot: &GpuTensor, ef: &GpuTensor, row: usize| {
        gated_delta_step_batched(
            gpu,
            &GatedDeltaStepBatched {
                projection: &proj_t.sub_offset(row * qkv, qkv),
                gate: &gate_t.sub_offset(row * vh, vh),
                beta: &beta_t.sub_offset(row * vh, vh),
                state: slot,
                format: GdnStateFormat::Q8Ef,
                residual: Some(ef),
                output: &out_t.sub_offset(0, value),
                row_states: None,
                rows: 1,
                qkv_width: qkv,
                key_heads: kh,
                value_heads: vh,
                key_dim: 128,
                value_dim: 128,
                position: row,
            },
        )
        .expect("AR step");
    };
    let mut refs = vec![(slot0.clone(), ef0.clone())];
    for row in 0..R {
        ar_step(&mut gpu, &slot_ref, &ef_ref, row);
        refs.push((down(&gpu, &slot_ref), down(&gpu, &ef_ref)));
    }
    assert_ne!(refs[R].1, ef0, "the window moves the residual");
    let conv_after = |keep: usize| {
        let mut conv = conv0.clone();
        let channel = qkv * 4;
        for pos in 0..keep {
            let slot = pos % 3;
            conv[slot * channel..(slot + 1) * channel]
                .copy_from_slice(&inputs[pos * channel..(pos + 1) * channel]);
        }
        conv
    };
    let ple_normed = gpu.zeros(&[1], DType::F32).unwrap();
    let tokens = vec![1u32; R];
    let mut checked = 0;
    for keep in [0usize, 1, R / 2, R - 1, R] {
        state.row_capture_armed = true;
        let ticket = state.snapshot(&mut gpu).expect("snapshot");
        {
            let capture = state.gdn_row_capture(0, R).expect("armed capture");
            up(&gpu, capture.recurrence, &[proj.clone(), gate.clone(), beta.clone()].concat());
            up(&gpu, capture.inputs, &inputs);
            gated_delta_step_batched(
                &mut gpu,
                &GatedDeltaStepBatched {
                    projection: &proj_t,
                    gate: &gate_t,
                    beta: &beta_t,
                    state: &state.gdn[0].recurrent,
                    format: GdnStateFormat::Q8Ef,
                    residual: state.gdn[0].ef_residual.as_ref(),
                    output: &out_t,
                    row_states: Some(&capture.states),
                    rows: R,
                    qkv_width: qkv,
                    key_heads: kh,
                    value_heads: vh,
                    key_dim: 128,
                    value_dim: 128,
                    position: 0,
                },
            )
            .expect("capture");
        }
        state.commit_row_capture(R);
        if keep > 0 && keep < R {
            state
                .rollback_rows_retain(
                    &mut gpu,
                    ticket,
                    keep,
                    R,
                    &tokens,
                    &ple_normed,
                    0,
                    3,
                    config.indexer_compress_ratio,
                    1024,
                )
                .expect("rollback");
        }
        if keep > 0 {
            let (slot, ef) = &refs[keep];
            assert_eq!(&down(&gpu, &state.gdn[0].recurrent), slot, "keep {keep}: slot");
            assert_eq!(
                &down(&gpu, state.gdn[0].ef_residual.as_ref().unwrap()),
                ef,
                "keep {keep}: the real residual (not zeroed)"
            );
            if keep < R {
                assert_eq!(down(&gpu, &state.gdn[0].conv), conv_after(keep), "keep {keep}: conv");
            }
        }
        state.row_capture_armed = false;
        state.restore(&mut gpu, ticket).expect("restore");
        assert_eq!(down(&gpu, &state.gdn[0].recurrent), slot0, "keep {keep}: restored slot");
        assert_eq!(
            down(&gpu, state.gdn[0].ef_residual.as_ref().unwrap()),
            ef0,
            "keep {keep}: restored residual"
        );
        assert_eq!(down(&gpu, &state.gdn[0].conv), conv0, "keep {keep}: restored conv");
        // Continue: one more AR step on the restored state equals the chain's.
        let ef = state.gdn[0].ef_residual.as_ref().unwrap();
        ar_step(&mut gpu, &state.gdn[0].recurrent, ef, 0);
        assert_eq!(down(&gpu, &state.gdn[0].recurrent), refs[1].0, "keep {keep}: continued slot");
        assert_eq!(down(&gpu, ef), refs[1].1, "keep {keep}: continued residual");
        // Back to the initial bytes for the next case.
        up(&gpu, &state.gdn[0].recurrent, &slot0);
        up(&gpu, ef, &ef0);
        checked += 1;
    }
    println!("q8ef_state_oracle: arch={} cases={checked} failures=0", gpu.arch);
    let _ = state.free_gpu(&mut gpu);
}
