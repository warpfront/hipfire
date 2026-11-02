// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Railgun E0 / L6c parity gate (gfx1201): the two-launch multi-layer GDN
//! replay (`dflash_gdn_replay_pre_ml` + `gated_delta_net_q8_fast_ml`) against
//! the exact per-layer sequence `GdnTape::replay_gdn_inner` runs on gfx1201
//! (`conv1d_silu_split_f32_n` + `fused_qk_l2_norm_scale_f32_batched` +
//! `repeat_interleave_qk_f32_batched` + `gated_delta_net_q8_batch_seq`).
//!
//! Real Qwen3.8-27B DeltaNet dims (16 key heads, 48 value heads, head_dim
//! 128), several layers, nonzero Q8 state + scales + EF residual, and a
//! chain of replays with varying step counts that carries state between
//! calls. Compares every layer's conv ring, Q8 state, scales and EF residual
//! byte-for-byte after every replay, plus the last layer's q/k/v and
//! recurrence output. Runs with EF on and with EF off (stochastic rounding:
//! both arms consume identical frame IDs via the frame checkpoint). Any
//! mismatch aborts nonzero. Non-gfx1201 exits 0 with a skip note.

use rdna_compute::dflash_gdn_replay::{table_bytes, DflashReplayPreLayer, GdnLayerTable};
use rdna_compute::norm::{gdn_requant_frame_checkpoint, restore_gdn_requant_frame_checkpoint};
use rdna_compute::{DType, Gpu, GpuTensor};

const HD: usize = 128;
const N_KEY: usize = 16;
const N_V: usize = 48;
const RATIO: usize = N_V / N_KEY;
const K_DIM: usize = N_KEY * HD;
const V_DIM: usize = N_V * HD;
const QKV_DIM: usize = 2 * K_DIM + V_DIM;
const N_CH: usize = QKV_DIM;
const MAX_N: usize = 16;
const S_SIZE: usize = N_V * HD * HD;
const LAYERS: usize = 6;
const EPS: f32 = 1e-6;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn f(&mut self, scale: f32) -> f32 {
        ((self.next() as f64 / u32::MAX as f64 - 0.5) as f32) * scale
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_ne_bytes()).collect()
}

fn upload(gpu: &Gpu, t: &GpuTensor, bytes: &[u8]) {
    assert_eq!(bytes.len(), t.byte_size());
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("upload");
}

fn download(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    let mut b = vec![0u8; t.byte_size()];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).expect("download");
    b
}

fn raw(gpu: &mut Gpu, bytes: usize) -> GpuTensor {
    let buf = gpu.hip.malloc(bytes).expect("malloc");
    GpuTensor {
        buf,
        shape: vec![bytes],
        dtype: DType::Raw,
    }
}

fn check_eq(name: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    if a != b {
        let first = a.iter().zip(b).position(|(x, y)| x != y).unwrap();
        let diffs = a.iter().zip(b).filter(|(x, y)| x != y).count();
        panic!(
            "{name}: byte mismatch at byte {first}; {diffs} of {} bytes differ",
            a.len()
        );
    }
}

/// One layer's mutable recurrent state for one arm.
struct LayerState {
    conv: GpuTensor,
    s: GpuTensor,
    scales: GpuTensor,
    ef: GpuTensor,
}

/// One layer's tape + conv weight, shared by both arms (read-only).
struct LayerTape {
    qkv: GpuTensor,
    alpha: GpuTensor,
    beta: GpuTensor,
    conv_w: GpuTensor,
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: test_dflash_replay_ml requires exact gfx1201");
        return;
    }
    eprintln!("=== multi-layer GDN replay parity (gfx1201, {LAYERS} layers) ===");
    let mut rng = Lcg(0x5EED_1201);
    let q_scale = 1.0 / (HD as f32).sqrt();

    let tapes: Vec<LayerTape> = (0..LAYERS)
        .map(|_| {
            let qkv = gpu.alloc_tensor(&[MAX_N * QKV_DIM], DType::F32).unwrap();
            let alpha = gpu.alloc_tensor(&[MAX_N * N_V], DType::F32).unwrap();
            let beta = gpu.alloc_tensor(&[MAX_N * N_V], DType::F32).unwrap();
            let conv_w = gpu.alloc_tensor(&[N_CH * 4], DType::F32).unwrap();
            upload(
                &gpu,
                &qkv,
                &f32_bytes(&(0..MAX_N * QKV_DIM).map(|_| rng.f(1.2)).collect::<Vec<_>>()),
            );
            // Gate = log-decay (<= 0), beta = sigmoid output in (0, 1).
            upload(
                &gpu,
                &alpha,
                &f32_bytes(
                    &(0..MAX_N * N_V)
                        .map(|_| -rng.f(1.0).abs() * 2.0)
                        .collect::<Vec<_>>(),
                ),
            );
            upload(
                &gpu,
                &beta,
                &f32_bytes(
                    &(0..MAX_N * N_V)
                        .map(|_| 0.5 + rng.f(0.98))
                        .collect::<Vec<_>>(),
                ),
            );
            upload(
                &gpu,
                &conv_w,
                &f32_bytes(&(0..N_CH * 4).map(|_| rng.f(0.5)).collect::<Vec<_>>()),
            );
            LayerTape {
                qkv,
                alpha,
                beta,
                conv_w,
            }
        })
        .collect();

    // Shared legacy scratch (the per-layer path reuses one set).
    let q_raw = gpu.alloc_tensor(&[MAX_N * K_DIM], DType::F32).unwrap();
    let k_raw = gpu.alloc_tensor(&[MAX_N * K_DIM], DType::F32).unwrap();
    let v_s = gpu.alloc_tensor(&[MAX_N * V_DIM], DType::F32).unwrap();
    let q_s = gpu.alloc_tensor(&[MAX_N * V_DIM], DType::F32).unwrap();
    let k_s = gpu.alloc_tensor(&[MAX_N * V_DIM], DType::F32).unwrap();
    let attn_s = gpu.alloc_tensor(&[MAX_N * V_DIM], DType::F32).unwrap();
    // Multi-layer scratch: one [MAX_N x V_DIM] row block per layer.
    let ml_q = gpu
        .alloc_tensor(&[LAYERS * MAX_N * V_DIM], DType::F32)
        .unwrap();
    let ml_k = gpu
        .alloc_tensor(&[LAYERS * MAX_N * V_DIM], DType::F32)
        .unwrap();
    let ml_v = gpu
        .alloc_tensor(&[LAYERS * MAX_N * V_DIM], DType::F32)
        .unwrap();
    let ml_out = gpu
        .alloc_tensor(&[LAYERS * MAX_N * V_DIM], DType::F32)
        .unwrap();
    let row = (MAX_N * V_DIM * 4) as u64;

    for ef_on in [true, false] {
        eprintln!(
            "--- EF residual {} ---",
            if ef_on { "on" } else { "off (stochastic)" }
        );
        let mut arms: [Vec<LayerState>; 2] = [Vec::new(), Vec::new()];
        for _ in 0..LAYERS {
            let conv0: Vec<f32> = (0..N_CH * 3).map(|_| rng.f(0.6)).collect();
            let s0: Vec<u8> = (0..S_SIZE)
                .map(|_| (rng.next() % 255) as u8 ^ 0x80)
                .collect();
            let sc0: Vec<f32> = (0..N_V * HD).map(|_| 0.004 + rng.f(0.006).abs()).collect();
            // Small f16 residuals: exponent 5..6, random sign/mantissa.
            let ef0: Vec<u8> = (0..S_SIZE)
                .flat_map(|_| {
                    let r = rng.next() as u16;
                    ((r & 0x83FF) | 0x1400).to_ne_bytes()
                })
                .collect();
            for arm in arms.iter_mut() {
                let st = LayerState {
                    conv: gpu.alloc_tensor(&[N_CH * 3], DType::F32).unwrap(),
                    s: raw(&mut gpu, S_SIZE),
                    scales: gpu.alloc_tensor(&[N_V * HD], DType::F32).unwrap(),
                    ef: gpu.alloc_tensor(&[S_SIZE], DType::F16).unwrap(),
                };
                upload(&gpu, &st.conv, &f32_bytes(&conv0));
                upload(&gpu, &st.s, &s0);
                upload(&gpu, &st.scales, &f32_bytes(&sc0));
                upload(&gpu, &st.ef, &ef0);
                arm.push(st);
            }
        }
        let ml = &arms[1];
        let pre_rows: Vec<DflashReplayPreLayer> = (0..LAYERS)
            .map(|l| DflashReplayPreLayer {
                qkv_tape: tapes[l].qkv.buf.as_ptr() as u64,
                conv_w: tapes[l].conv_w.buf.as_ptr() as u64,
                conv_state: ml[l].conv.buf.as_ptr() as u64,
                v_out: ml_v.buf.as_ptr() as u64 + l as u64 * row,
                q_dst: ml_q.buf.as_ptr() as u64 + l as u64 * row,
                k_dst: ml_k.buf.as_ptr() as u64 + l as u64 * row,
            })
            .collect();
        let gdn_rows: Vec<GdnLayerTable> = (0..LAYERS)
            .map(|l| GdnLayerTable {
                q: ml_q.buf.as_ptr() as u64 + l as u64 * row,
                k: ml_k.buf.as_ptr() as u64 + l as u64 * row,
                v: ml_v.buf.as_ptr() as u64 + l as u64 * row,
                gate: tapes[l].alpha.buf.as_ptr() as u64,
                beta: tapes[l].beta.buf.as_ptr() as u64,
                s_q8: ml[l].s.buf.as_ptr() as u64,
                s_scales: ml[l].scales.buf.as_ptr() as u64,
                output: ml_out.buf.as_ptr() as u64 + l as u64 * row,
                ef: if ef_on {
                    ml[l].ef.buf.as_ptr() as u64
                } else {
                    0
                },
            })
            .collect();
        let pre_table = gpu
            .hip
            .malloc(std::mem::size_of_val(&pre_rows[..]))
            .unwrap();
        let gdn_table = gpu
            .hip
            .malloc(std::mem::size_of_val(&gdn_rows[..]))
            .unwrap();
        gpu.hip
            .memcpy_htod(&pre_table, table_bytes(&pre_rows))
            .unwrap();
        gpu.hip
            .memcpy_htod(&gdn_table, table_bytes(&gdn_rows))
            .unwrap();

        for (call, n) in [1usize, 2, 5, 16, 15, 3, 7, 16, 1, 12]
            .into_iter()
            .enumerate()
        {
            assert!(gpu.gdn_replay_ml_eligible(N_V, N_KEY, HD, HD, n));
            let frame0 = gdn_requant_frame_checkpoint();
            // Legacy: exactly replay_gdn_inner's gfx1201 per-layer sequence.
            for l in 0..LAYERS {
                let st = &arms[0][l];
                gpu.conv1d_silu_split_f32_n(
                    &q_raw,
                    &k_raw,
                    &v_s,
                    &tapes[l].qkv,
                    &tapes[l].conv_w,
                    &st.conv,
                    K_DIM,
                    V_DIM,
                    n,
                )
                .unwrap();
                gpu.fused_qk_l2_norm_scale_f32_batched(&q_raw, &k_raw, N_KEY, HD, q_scale, EPS, n)
                    .unwrap();
                gpu.repeat_interleave_qk_f32_batched(
                    &q_raw, &k_raw, &q_s, &k_s, N_KEY, RATIO, HD, n,
                )
                .unwrap();
                gpu.gated_delta_net_q8_batch_seq(
                    &q_s,
                    &k_s,
                    &v_s,
                    &tapes[l].alpha,
                    &tapes[l].beta,
                    &st.s,
                    &st.scales,
                    &attn_s,
                    n,
                    N_V,
                    HD,
                    if ef_on { Some(&st.ef) } else { None },
                )
                .unwrap();
            }
            let frame_after_legacy = gdn_requant_frame_checkpoint();
            restore_gdn_requant_frame_checkpoint(frame0);
            // Multi-layer: two launches.
            gpu.dflash_gdn_replay_pre_ml(
                pre_table.as_ptr() as *const _,
                LAYERS,
                N_V,
                N_KEY,
                K_DIM,
                V_DIM,
                QKV_DIM,
                n,
                q_scale,
                EPS,
            )
            .unwrap();
            gpu.gated_delta_net_q8_fast_ml(gdn_table.as_ptr() as *const _, LAYERS, n, N_V, HD)
                .unwrap();
            assert_eq!(
                gdn_requant_frame_checkpoint(),
                frame_after_legacy,
                "frame reservation differs"
            );
            gpu.hip.device_synchronize().unwrap();

            for l in 0..LAYERS {
                let (o, w) = (&arms[0][l], &arms[1][l]);
                check_eq(
                    &format!("call{call} n={n} L{l} conv_state"),
                    &download(&gpu, &o.conv),
                    &download(&gpu, &w.conv),
                );
                check_eq(
                    &format!("call{call} n={n} L{l} s_q8"),
                    &download(&gpu, &o.s),
                    &download(&gpu, &w.s),
                );
                check_eq(
                    &format!("call{call} n={n} L{l} s_scales"),
                    &download(&gpu, &o.scales),
                    &download(&gpu, &w.scales),
                );
                if ef_on {
                    check_eq(
                        &format!("call{call} n={n} L{l} ef"),
                        &download(&gpu, &o.ef),
                        &download(&gpu, &w.ef),
                    );
                }
            }
            let last = (LAYERS - 1) * MAX_N * V_DIM * 4;
            let used = n * V_DIM * 4;
            for (name, legacy, multi) in [
                ("q", &q_s, &ml_q),
                ("k", &k_s, &ml_k),
                ("v", &v_s, &ml_v),
                ("attn", &attn_s, &ml_out),
            ] {
                let a = download(&gpu, legacy);
                let b = download(&gpu, multi);
                check_eq(
                    &format!("call{call} n={n} last-layer {name}"),
                    &a[..used],
                    &b[last..last + used],
                );
            }
            eprintln!("  ok call{call} n_steps={n}: {LAYERS} layers conv/S/scales{} + last-layer q/k/v/attn identical", if ef_on { "/EF" } else { "" });
        }
        let _ = gpu.hip.free(pre_table);
        let _ = gpu.hip.free(gdn_table);
        for arm in arms {
            for st in arm {
                for t in [st.conv, st.s, st.scales, st.ef] {
                    let _ = gpu.free_tensor(t);
                }
            }
        }
    }
    assert!(
        !gpu.gdn_replay_ml_eligible(N_V, N_KEY, HD, HD, 17),
        "n_steps=17 must decline"
    );
    assert!(
        !gpu.gdn_replay_ml_eligible(N_V, N_KEY, HD, HD, 0),
        "n_steps=0 must decline"
    );
    eprintln!("  ok ineligible step counts decline");

    // Device timing at the real replay shape: 48 LA layers, EF on, one layer
    // tape reused (read-only), per-layer state. Back-to-back replays.
    const TL: usize = 48;
    let states: Vec<LayerState> = (0..TL)
        .map(|_| LayerState {
            conv: gpu.zeros(&[N_CH * 3], DType::F32).unwrap(),
            s: gpu.zeros(&[S_SIZE / 4], DType::F32).unwrap(),
            scales: gpu.zeros(&[N_V * HD], DType::F32).unwrap(),
            ef: gpu.zeros(&[S_SIZE], DType::F16).unwrap(),
        })
        .collect();
    let big = |gpu: &mut Gpu| gpu.alloc_tensor(&[TL * MAX_N * V_DIM], DType::F32).unwrap();
    let (tq, tk, tv, to) = (big(&mut gpu), big(&mut gpu), big(&mut gpu), big(&mut gpu));
    let t = &tapes[0];
    let pre_rows: Vec<DflashReplayPreLayer> = (0..TL)
        .map(|l| DflashReplayPreLayer {
            qkv_tape: t.qkv.buf.as_ptr() as u64,
            conv_w: t.conv_w.buf.as_ptr() as u64,
            conv_state: states[l].conv.buf.as_ptr() as u64,
            v_out: tv.buf.as_ptr() as u64 + l as u64 * row,
            q_dst: tq.buf.as_ptr() as u64 + l as u64 * row,
            k_dst: tk.buf.as_ptr() as u64 + l as u64 * row,
        })
        .collect();
    let gdn_rows: Vec<GdnLayerTable> = (0..TL)
        .map(|l| GdnLayerTable {
            q: tq.buf.as_ptr() as u64 + l as u64 * row,
            k: tk.buf.as_ptr() as u64 + l as u64 * row,
            v: tv.buf.as_ptr() as u64 + l as u64 * row,
            gate: t.alpha.buf.as_ptr() as u64,
            beta: t.beta.buf.as_ptr() as u64,
            s_q8: states[l].s.buf.as_ptr() as u64,
            s_scales: states[l].scales.buf.as_ptr() as u64,
            output: to.buf.as_ptr() as u64 + l as u64 * row,
            ef: states[l].ef.buf.as_ptr() as u64,
        })
        .collect();
    let pre_table = gpu
        .hip
        .malloc(std::mem::size_of_val(&pre_rows[..]))
        .unwrap();
    let gdn_table = gpu
        .hip
        .malloc(std::mem::size_of_val(&gdn_rows[..]))
        .unwrap();
    gpu.hip
        .memcpy_htod(&pre_table, table_bytes(&pre_rows))
        .unwrap();
    gpu.hip
        .memcpy_htod(&gdn_table, table_bytes(&gdn_rows))
        .unwrap();
    let reps = 50;
    for n in [1usize, 8, 15, 16] {
        let mut legacy = |gpu: &mut Gpu| {
            for st in &states {
                gpu.conv1d_silu_split_f32_n(
                    &q_raw, &k_raw, &v_s, &t.qkv, &t.conv_w, &st.conv, K_DIM, V_DIM, n,
                )
                .unwrap();
                gpu.fused_qk_l2_norm_scale_f32_batched(&q_raw, &k_raw, N_KEY, HD, q_scale, EPS, n)
                    .unwrap();
                gpu.repeat_interleave_qk_f32_batched(
                    &q_raw, &k_raw, &q_s, &k_s, N_KEY, RATIO, HD, n,
                )
                .unwrap();
                gpu.gated_delta_net_q8_batch_seq(
                    &q_s,
                    &k_s,
                    &v_s,
                    &t.alpha,
                    &t.beta,
                    &st.s,
                    &st.scales,
                    &attn_s,
                    n,
                    N_V,
                    HD,
                    Some(&st.ef),
                )
                .unwrap();
            }
        };
        let mut multi = |gpu: &mut Gpu| {
            gpu.dflash_gdn_replay_pre_ml(
                pre_table.as_ptr() as *const _,
                TL,
                N_V,
                N_KEY,
                K_DIM,
                V_DIM,
                QKV_DIM,
                n,
                q_scale,
                EPS,
            )
            .unwrap();
            gpu.gated_delta_net_q8_fast_ml(gdn_table.as_ptr() as *const _, TL, n, N_V, HD)
                .unwrap();
        };
        let mut time = |gpu: &mut Gpu, f: &mut dyn FnMut(&mut Gpu)| -> f64 {
            for _ in 0..3 {
                f(gpu);
            }
            gpu.hip.device_synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                f(gpu);
            }
            gpu.hip.device_synchronize().unwrap();
            t0.elapsed().as_secs_f64() * 1e6 / reps as f64
        };
        let a = time(&mut gpu, &mut legacy);
        let b = time(&mut gpu, &mut multi);
        eprintln!("  timing 48 layers n_steps={n}: per-layer {a:.1} us/replay (192 launches) -> multi-layer {b:.1} us/replay (2 launches)");
    }
    eprintln!("PASS: multi-layer GDN replay byte-identical to the per-layer gfx1201 sequence");
}
