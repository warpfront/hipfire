// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Railgun D8 parity gate (gfx1201): the snapshot-source multi-layer GDN
//! replay (`dflash_gdn_replay_pre_ml_from` + `gated_delta_net_q8_fast_ml_from`,
//! state read from snapshot buffers, written to the live buffers) against the
//! E0 route it replaces in the DFlash rollback: restore the snapshot into the
//! live buffers (byte copy), then the in-place two-launch replay
//! (`dflash_gdn_replay_pre_ml` + `gated_delta_net_q8_fast_ml`).
//!
//! Real Qwen3.8-27B DeltaNet dims, several layers, nonzero Q8 state + scales
//! + EF residual, and a chain of cycles with varying step counts. Each cycle:
//! both arms start from the same pre-verify state; the D8 arm's live buffers
//! are first overwritten with a different poison per cycle (standing in for
//! the verify-advanced state), so any live byte the D8 kernels fail to write
//! shows up as a mismatch. After every cycle, every layer's conv ring, Q8
//! state, scales and EF residual must be byte-identical between the arms, the
//! last layer's q/k/v/recurrence output too, frame reservations equal, and the
//! D8 snapshot must be unchanged (it stays the pre-window state). Runs with EF
//! on and EF off (stochastic rounding, identical frames). Any mismatch aborts
//! nonzero. Non-gfx1201 exits 0 with a skip note.

use rdna_compute::dflash_gdn_replay::{
    table_bytes, DflashReplayPreLayer, DflashReplayPreLayerFrom, GdnLayerTable, GdnLayerTableFrom,
};
use rdna_compute::norm::{gdn_requant_frame_checkpoint, restore_gdn_requant_frame_checkpoint};
use rdna_compute::{DType, Gpu, GpuTensor};

const HD: usize = 128;
const N_KEY: usize = 16;
const N_V: usize = 48;
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

/// One layer's recurrent state (one buffer set).
struct LayerState {
    conv: GpuTensor,
    s: GpuTensor,
    scales: GpuTensor,
    ef: GpuTensor,
}

impl LayerState {
    fn alloc(gpu: &mut Gpu) -> Self {
        Self {
            conv: gpu.alloc_tensor(&[N_CH * 3], DType::F32).unwrap(),
            s: raw(gpu, S_SIZE),
            scales: gpu.alloc_tensor(&[N_V * HD], DType::F32).unwrap(),
            ef: gpu.alloc_tensor(&[S_SIZE], DType::F16).unwrap(),
        }
    }
    fn tensors(&self) -> [&GpuTensor; 4] {
        [&self.conv, &self.s, &self.scales, &self.ef]
    }
    /// Byte copy `src` -> `self` (the snapshot save / restore).
    fn copy_from(&self, gpu: &Gpu, src: &LayerState) {
        for (d, s) in self.tensors().into_iter().zip(src.tensors()) {
            gpu.hip.memcpy_dtod(&d.buf, &s.buf, s.buf.size()).unwrap();
        }
    }
    fn fill(&self, gpu: &Gpu, byte: u8) {
        for t in self.tensors() {
            upload(gpu, t, &vec![byte; t.byte_size()]);
        }
    }
    fn bytes(&self, gpu: &Gpu) -> [Vec<u8>; 4] {
        self.tensors().map(|t| download(gpu, t))
    }
}

/// One layer's tape + conv weight (read-only).
struct LayerTape {
    qkv: GpuTensor,
    alpha: GpuTensor,
    beta: GpuTensor,
    conv_w: GpuTensor,
}

struct Scratch {
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    out: GpuTensor,
}

fn pre_row(t: &LayerTape, st: &LayerState, sc: &Scratch, l: usize) -> DflashReplayPreLayer {
    let row = (MAX_N * V_DIM * 4 * l) as u64;
    DflashReplayPreLayer {
        qkv_tape: t.qkv.buf.as_ptr() as u64,
        conv_w: t.conv_w.buf.as_ptr() as u64,
        conv_state: st.conv.buf.as_ptr() as u64,
        v_out: sc.v.buf.as_ptr() as u64 + row,
        q_dst: sc.q.buf.as_ptr() as u64 + row,
        k_dst: sc.k.buf.as_ptr() as u64 + row,
    }
}

fn gdn_row(t: &LayerTape, st: &LayerState, sc: &Scratch, l: usize, ef_on: bool) -> GdnLayerTable {
    let row = (MAX_N * V_DIM * 4 * l) as u64;
    GdnLayerTable {
        q: sc.q.buf.as_ptr() as u64 + row,
        k: sc.k.buf.as_ptr() as u64 + row,
        v: sc.v.buf.as_ptr() as u64 + row,
        gate: t.alpha.buf.as_ptr() as u64,
        beta: t.beta.buf.as_ptr() as u64,
        s_q8: st.s.buf.as_ptr() as u64,
        s_scales: st.scales.buf.as_ptr() as u64,
        output: sc.out.buf.as_ptr() as u64 + row,
        ef: if ef_on { st.ef.buf.as_ptr() as u64 } else { 0 },
    }
}

fn scratch(gpu: &mut Gpu, layers: usize) -> Scratch {
    let mut big = || {
        gpu.alloc_tensor(&[layers * MAX_N * V_DIM], DType::F32)
            .unwrap()
    };
    Scratch {
        q: big(),
        k: big(),
        v: big(),
        out: big(),
    }
}

fn upload_table<T: Copy>(gpu: &Gpu, rows: &[T]) -> hip_bridge::DeviceBuffer {
    let t = gpu.hip.malloc(std::mem::size_of_val(rows)).unwrap();
    gpu.hip.memcpy_htod(&t, table_bytes(rows)).unwrap();
    t
}

fn make_tape(gpu: &Gpu, t: &LayerTape, rng: &mut Lcg) {
    upload(
        gpu,
        &t.qkv,
        &f32_bytes(&(0..MAX_N * QKV_DIM).map(|_| rng.f(1.2)).collect::<Vec<_>>()),
    );
    // Gate = log-decay (<= 0), beta = sigmoid output in (0, 1).
    upload(
        gpu,
        &t.alpha,
        &f32_bytes(
            &(0..MAX_N * N_V)
                .map(|_| -rng.f(1.0).abs() * 2.0)
                .collect::<Vec<_>>(),
        ),
    );
    upload(
        gpu,
        &t.beta,
        &f32_bytes(
            &(0..MAX_N * N_V)
                .map(|_| 0.5 + rng.f(0.98))
                .collect::<Vec<_>>(),
        ),
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: test_dflash_replay_from_snapshot requires exact gfx1201");
        return;
    }
    eprintln!("=== D8 snapshot-source GDN replay parity (gfx1201, {LAYERS} layers) ===");
    let mut rng = Lcg(0xD8_1201);
    let q_scale = 1.0 / (HD as f32).sqrt();
    let tapes: Vec<LayerTape> = (0..LAYERS)
        .map(|_| {
            let t = LayerTape {
                qkv: gpu.alloc_tensor(&[MAX_N * QKV_DIM], DType::F32).unwrap(),
                alpha: gpu.alloc_tensor(&[MAX_N * N_V], DType::F32).unwrap(),
                beta: gpu.alloc_tensor(&[MAX_N * N_V], DType::F32).unwrap(),
                conv_w: gpu.alloc_tensor(&[N_CH * 4], DType::F32).unwrap(),
            };
            upload(
                &gpu,
                &t.conv_w,
                &f32_bytes(&(0..N_CH * 4).map(|_| rng.f(0.5)).collect::<Vec<_>>()),
            );
            t
        })
        .collect();
    let sc_e0 = scratch(&mut gpu, LAYERS);
    let sc_d8 = scratch(&mut gpu, LAYERS);

    for ef_on in [true, false] {
        eprintln!(
            "--- EF residual {} ---",
            if ef_on { "on" } else { "off (stochastic)" }
        );
        // E0 arm: live + snapshot (restore then in-place replay).
        // D8 arm: live + snapshot (replay reads the snapshot, writes live).
        let e0_live: Vec<LayerState> = (0..LAYERS).map(|_| LayerState::alloc(&mut gpu)).collect();
        let e0_snap: Vec<LayerState> = (0..LAYERS).map(|_| LayerState::alloc(&mut gpu)).collect();
        let d8_live: Vec<LayerState> = (0..LAYERS).map(|_| LayerState::alloc(&mut gpu)).collect();
        let d8_snap: Vec<LayerState> = (0..LAYERS).map(|_| LayerState::alloc(&mut gpu)).collect();
        for l in 0..LAYERS {
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
            for st in [&e0_live[l], &d8_live[l]] {
                upload(&gpu, &st.conv, &f32_bytes(&conv0));
                upload(&gpu, &st.s, &s0);
                upload(&gpu, &st.scales, &f32_bytes(&sc0));
                upload(&gpu, &st.ef, &ef0);
            }
        }
        let e0_pre = upload_table(
            &gpu,
            &(0..LAYERS)
                .map(|l| pre_row(&tapes[l], &e0_live[l], &sc_e0, l))
                .collect::<Vec<_>>(),
        );
        let e0_gdn = upload_table(
            &gpu,
            &(0..LAYERS)
                .map(|l| gdn_row(&tapes[l], &e0_live[l], &sc_e0, l, ef_on))
                .collect::<Vec<_>>(),
        );
        let d8_pre = upload_table(
            &gpu,
            &(0..LAYERS)
                .map(|l| DflashReplayPreLayerFrom {
                    base: pre_row(&tapes[l], &d8_live[l], &sc_d8, l),
                    conv_state_src: d8_snap[l].conv.buf.as_ptr() as u64,
                })
                .collect::<Vec<_>>(),
        );
        let d8_gdn = upload_table(
            &gpu,
            &(0..LAYERS)
                .map(|l| GdnLayerTableFrom {
                    base: gdn_row(&tapes[l], &d8_live[l], &sc_d8, l, ef_on),
                    s_q8_src: d8_snap[l].s.buf.as_ptr() as u64,
                    s_scales_src: d8_snap[l].scales.buf.as_ptr() as u64,
                    ef_src: if ef_on {
                        d8_snap[l].ef.buf.as_ptr() as u64
                    } else {
                        0
                    },
                })
                .collect::<Vec<_>>(),
        );

        // Accept lengths 0..=15 plus the clamp-shaped short steps; n = accept + 1.
        let steps = [
            1usize, 16, 2, 5, 16, 15, 3, 7, 16, 1, 12, 16, 16, 4, 9, 13, 1, 16,
        ];
        for (cycle, &n) in steps.iter().enumerate() {
            for t in &tapes {
                make_tape(&gpu, t, &mut rng);
            }
            // Snapshot save on both arms (the cycle's pre-verify state).
            for l in 0..LAYERS {
                e0_snap[l].copy_from(&gpu, &e0_live[l]);
                d8_snap[l].copy_from(&gpu, &d8_live[l]);
            }
            // "Verify" clobbers both live sets; a different poison per cycle
            // on the D8 side proves every live byte is rewritten.
            let poison = [0xA5u8, 0x5A, 0xFF, 0x00, 0x7F, 0x81][cycle % 6];
            for l in 0..LAYERS {
                e0_live[l].fill(&gpu, 0x3C);
                d8_live[l].fill(&gpu, poison);
            }
            let snap_before: Vec<[Vec<u8>; 4]> = d8_snap.iter().map(|s| s.bytes(&gpu)).collect();

            let frame0 = gdn_requant_frame_checkpoint();
            // E0: restore + in-place replay.
            for l in 0..LAYERS {
                e0_live[l].copy_from(&gpu, &e0_snap[l]);
            }
            gpu.dflash_gdn_replay_pre_ml(
                e0_pre.as_ptr() as *const _,
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
            gpu.gated_delta_net_q8_fast_ml(e0_gdn.as_ptr() as *const _, LAYERS, n, N_V, HD)
                .unwrap();
            let frame_e0 = gdn_requant_frame_checkpoint();
            restore_gdn_requant_frame_checkpoint(frame0);
            // D8: replay from the snapshot, no restore.
            gpu.dflash_gdn_replay_pre_ml_from(
                d8_pre.as_ptr() as *const _,
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
            gpu.gated_delta_net_q8_fast_ml_from(d8_gdn.as_ptr() as *const _, LAYERS, n, N_V, HD)
                .unwrap();
            assert_eq!(
                gdn_requant_frame_checkpoint(),
                frame_e0,
                "frame reservation differs"
            );
            gpu.hip.device_synchronize().unwrap();

            let names = ["conv_state", "s_q8", "s_scales", "ef"];
            for l in 0..LAYERS {
                let (a, b) = (e0_live[l].bytes(&gpu), d8_live[l].bytes(&gpu));
                for (i, name) in names.iter().enumerate() {
                    if i == 3 && !ef_on {
                        continue;
                    }
                    check_eq(
                        &format!("cycle{cycle} n={n} L{l} live {name}"),
                        &a[i],
                        &b[i],
                    );
                }
                let after = d8_snap[l].bytes(&gpu);
                for (i, name) in names.iter().enumerate() {
                    check_eq(
                        &format!("cycle{cycle} n={n} L{l} snapshot {name} unchanged"),
                        &snap_before[l][i],
                        &after[i],
                    );
                }
            }
            let used = n * V_DIM * 4;
            for (name, a, b) in [
                ("q", &sc_e0.q, &sc_d8.q),
                ("k", &sc_e0.k, &sc_d8.k),
                ("v", &sc_e0.v, &sc_d8.v),
                ("attn", &sc_e0.out, &sc_d8.out),
            ] {
                let (a, b) = (download(&gpu, a), download(&gpu, b));
                for l in 0..LAYERS {
                    let off = l * MAX_N * V_DIM * 4;
                    check_eq(
                        &format!("cycle{cycle} n={n} L{l} {name}"),
                        &a[off..off + used],
                        &b[off..off + used],
                    );
                }
            }
            eprintln!(
                "  ok cycle{cycle} n_steps={n} poison={poison:#04x}: {LAYERS} layers live conv/S/scales{} identical, snapshot unchanged, q/k/v/attn identical",
                if ef_on { "/EF" } else { "" }
            );
        }
        for t in [e0_pre, e0_gdn, d8_pre, d8_gdn] {
            let _ = gpu.hip.free(t);
        }
        for set in [e0_live, e0_snap, d8_live, d8_snap] {
            for st in set {
                for t in [st.conv, st.s, st.scales, st.ef] {
                    let _ = gpu.free_tensor(t);
                }
            }
        }
    }

    // Device timing at the real rollback shape: 48 LA layers, EF on, one layer
    // tape reused. E0 = restore copy (one memcpy per tensor here, not the bulk
    // kernel) + in-place replay; D8 = snapshot-source replay only.
    const TL: usize = 48;
    let live: Vec<LayerState> = (0..TL).map(|_| LayerState::alloc(&mut gpu)).collect();
    let snap: Vec<LayerState> = (0..TL).map(|_| LayerState::alloc(&mut gpu)).collect();
    for st in live.iter().chain(&snap) {
        st.fill(&gpu, 0);
    }
    make_tape(&gpu, &tapes[0], &mut rng);
    let sc = scratch(&mut gpu, TL);
    let t = &tapes[0];
    let e0_pre = upload_table(
        &gpu,
        &(0..TL)
            .map(|l| pre_row(t, &live[l], &sc, l))
            .collect::<Vec<_>>(),
    );
    let e0_gdn = upload_table(
        &gpu,
        &(0..TL)
            .map(|l| gdn_row(t, &live[l], &sc, l, true))
            .collect::<Vec<_>>(),
    );
    let d8_pre = upload_table(
        &gpu,
        &(0..TL)
            .map(|l| DflashReplayPreLayerFrom {
                base: pre_row(t, &live[l], &sc, l),
                conv_state_src: snap[l].conv.buf.as_ptr() as u64,
            })
            .collect::<Vec<_>>(),
    );
    let d8_gdn = upload_table(
        &gpu,
        &(0..TL)
            .map(|l| GdnLayerTableFrom {
                base: gdn_row(t, &live[l], &sc, l, true),
                s_q8_src: snap[l].s.buf.as_ptr() as u64,
                s_scales_src: snap[l].scales.buf.as_ptr() as u64,
                ef_src: snap[l].ef.buf.as_ptr() as u64,
            })
            .collect::<Vec<_>>(),
    );
    let reps = 50;
    for n in [1usize, 8, 16] {
        let mut e0 = |gpu: &mut Gpu| {
            for l in 0..TL {
                live[l].copy_from(gpu, &snap[l]);
            }
            gpu.dflash_gdn_replay_pre_ml(
                e0_pre.as_ptr() as *const _,
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
            gpu.gated_delta_net_q8_fast_ml(e0_gdn.as_ptr() as *const _, TL, n, N_V, HD)
                .unwrap();
        };
        let mut d8 = |gpu: &mut Gpu| {
            gpu.dflash_gdn_replay_pre_ml_from(
                d8_pre.as_ptr() as *const _,
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
            gpu.gated_delta_net_q8_fast_ml_from(d8_gdn.as_ptr() as *const _, TL, n, N_V, HD)
                .unwrap();
        };
        let time = |gpu: &mut Gpu, f: &mut dyn FnMut(&mut Gpu)| -> f64 {
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
        let a = time(&mut gpu, &mut e0);
        let b = time(&mut gpu, &mut d8);
        eprintln!("  timing 48 layers n_steps={n}: memcpy restore + in-place replay {a:.1} us -> snapshot-source replay {b:.1} us");
    }
    eprintln!("PASS: snapshot-source GDN replay byte-identical to restore + in-place replay; snapshot untouched");
}
