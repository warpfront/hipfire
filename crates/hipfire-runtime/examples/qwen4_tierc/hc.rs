// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 Tier C cases for the Hyper-Connection (HC) family on gfx942.
//!
//! What the gfx942 C1 route launches for HC (census: `local://census-hc.md`):
//!   - stream init      `hc_streams_init_from_embed_batched` (F32 streams)
//!   - HC read          `hyper_norm_f32` -> BF16 down projection (trunk MFMA GEMM)
//!                      -> `bf16_roundtrip_f32` (all rows) / `scale_f32` / `silu_f32`
//!                      -> BF16 up projection -> `hyper_read_projected_f32`
//!                      (`hyper_read_projected_rotate_f32` when the next mixer
//!                      consumes `mixed` through the read's rotation scratch)
//!   - HC write         `hyper_norm_f32` + BF16 gate projection + `hyper_write_f32`
//!   - decode write+read `hyper_write_norm_f32` (rows 1..=32, F32 streams)
//!   - final HC read    `execute_final_hyper` == the HC read above
//! `has_gfx11_plus_simt()` is false on gfx942, so `state_bf16` is false for the
//! whole forward and every `hc_activation_fused_f32` / `hyper_norm_gate*` /
//! `hyper_read_up_*` / `hc_row_fold_*` / BF16-state kernel is unreachable.
//!
//! CPU references transcribe the kernels' math (BF16 rounding at every source
//! boundary) from `kernels/src/tensor_ops.hip` and
//! `kernels/src/qwen4_gemv_bf16_xf32.hip`; the equation-level twin is
//! `hipfire-arch-qwen4/src/ops.rs:174-276` (`hc_read`, `hc_write`), which does
//! not round to BF16 per stage and therefore cannot be compared bitwise.
//!
//! Tolerances (frozen, common.rs): pure copies / BF16 round trips / scalings are
//! bit-exact. Every output that passes through `expf` / `rsqrtf` followed by a
//! BF16 rounding is compared with `BF16_INPUT_F32_ACCUM`: the device libm
//! `expf` / `rsqrtf` can differ from the host's by an ulp, which in rare
//! elements flips one BF16 rounding (one BF16 ulp = 2^-8 relative, inside
//! rtol 2^-5). The host mirrors the kernels' reduction trees (lane-strided
//! partials + 256-wide LDS tree; 32-lane shuffle tree) so only that flip
//! remains. hipcc contracts `a += b * c` into an FMA by default for these
//! modules (`compiler.rs` only disables it for the GDN modules), so the host
//! mirrors with `mul_add`.

use super::common::*;
use super::Q4Case;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{
    execute_final_hyper, execute_hyper_read, execute_hyper_write, execute_steps, HyperReadOp,
    HyperWriteOp, Step,
};
use rdna_compute::tensor_ops::{
    bf16_roundtrip_f32, hc_activation_fused_f32, hyper_norm, hyper_read_projected, hyper_write,
    hyper_write_norm, scale_f32, Bf16Roundtrip, HcActivationFused, HyperNextGates, HyperNorm,
    HyperReadProjected, HyperWrite, ScaleF32,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// `hc_lowrank` of the pinned Flash-Next config (`config.rs:1112`).
const HC_LOW: usize = 320;
/// Row counts for the element-wise / row-grid HC kernels: decode, small
/// prefill, a tail-y count, and the default non-gfx1151 prefill chunk.
const HC_ROWS: [usize; 4] = [1, 7, 33, 1536];
/// Rows the decode-fused `hyper_write_norm_f32` accepts (`rows <=
/// WIDE_VERIFY_MAX_ROWS` = 32: block `(branch, row)` is the whole computation
/// of its row, so the one-row decode and every wider verify block run the same
/// per-row chain; the route gates on `route_rows <= Gpu::few_row_limit()`):
/// the decode and 8-row shapes, the wide-verify widths 9/15/16/17/31/32.
const DECODE_ROWS: [usize; 9] = [1, 7, 8, 9, 15, 16, 17, 31, 32];
const EPS: f32 = 1.0e-6;

pub const CASES: &[Q4Case] = &[
    Q4Case {
        name: "qwen4_hc_streams_init_from_embed_batched",
        timeout_s: 90,
        run: case_streams_init,
    },
    Q4Case {
        name: "qwen4_hyper_norm_f32",
        timeout_s: 90,
        run: case_hyper_norm,
    },
    Q4Case {
        name: "qwen4_hc_read_activation_chain_bf16_roundtrip_scale_silu",
        timeout_s: 90,
        run: case_activation_chain,
    },
    Q4Case {
        name: "qwen4_hc_activation_fused_f32_mtp_only",
        timeout_s: 90,
        run: case_activation_fused,
    },
    Q4Case {
        name: "qwen4_hyper_read_projected_f32",
        timeout_s: 90,
        run: case_read_projected,
    },
    Q4Case {
        name: "qwen4_hyper_read_projected_rotate_f32",
        timeout_s: 90,
        run: case_read_projected_rotate,
    },
    Q4Case {
        name: "qwen4_hyper_write_f32",
        timeout_s: 90,
        run: case_hyper_write,
    },
    Q4Case {
        name: "qwen4_hyper_write_norm_f32",
        timeout_s: 120,
        run: case_hyper_write_norm,
    },
    Q4Case {
        name: "qwen4_hc_g942_unfused_gates_bitwise_quarters",
        timeout_s: 120,
        run: case_unfused_gates_bitwise_quarters,
    },
    Q4Case {
        name: "qwen4_hc_read_route_execute_hyper_read",
        timeout_s: 180,
        run: case_route_read,
    },
    Q4Case {
        name: "qwen4_hc_final_read_route_execute_final_hyper",
        timeout_s: 180,
        run: case_route_final_read,
    },
    Q4Case {
        name: "qwen4_g942_fold_elem_hc_down_act_kernel_bytes",
        timeout_s: 300,
        run: case_fold_elem_hc_down_act,
    },
    Q4Case {
        name: "qwen4_g942_fold_elem_hc_read_route_digest",
        timeout_s: 240,
        run: case_fold_elem_route_digest,
    },
    Q4Case {
        name: "qwen4_hc_write_route_execute_hyper_write",
        timeout_s: 180,
        run: case_route_write,
    },
    Q4Case {
        name: "qwen4_hc_g942_lane_wide_write_bitwise_singles",
        timeout_s: 300,
        run: case_lane_wide_write,
    },
    Q4Case {
        name: "qwen4_hc_g942_lane_wide_read_bitwise_singles",
        timeout_s: 300,
        run: case_lane_wide_read,
    },
    Q4Case {
        name: "qwen4_hc_step_program_write_read_pairs",
        timeout_s: 240,
        run: case_step_program,
    },
];

// ── CPU twins ─────────────────────────────────────────────────────────────────

fn bf(x: f32) -> f32 {
    bf16_round(x)
}

/// `1.0f / (1.0f + expf(-x))` (tensor_ops.hip:1410, 1486, 1519).
fn sigmoid_k(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One (row, branch) of `hyper_norm_rows` (tensor_ops.hip:1535-1613, the
/// `single` hidden==2560 shape): thread `tid` sums `bf16(x[tid + 256 u])^2` for
/// u in order, 256-wide LDS tree with stride 128..1, `rsqrtf(sum/hidden + 1e-6)`,
/// then `bf16(bf16(x) * inv * (1 + w))`. Returns the normalized branch.
fn cpu_norm_branch(raw: &[f32], w: &[f32], out: &mut [f32]) {
    debug_assert_eq!(raw.len(), HIDDEN);
    let mut s = [0.0f32; 256];
    for (tid, slot) in s.iter_mut().enumerate() {
        let mut p = 0.0f32;
        for u in 0..HIDDEN / 256 {
            let v = bf(raw[tid + u * 256]);
            p = v.mul_add(v, p);
        }
        *slot = p;
    }
    let mut stride = 128;
    while stride > 0 {
        for t in 0..stride {
            s[t] += s[t + stride];
        }
        stride >>= 1;
    }
    let inv = 1.0 / (s[0] / HIDDEN as f32 + EPS).sqrt();
    for j in 0..HIDDEN {
        out[j] = bf(bf(raw[j]) * inv * (1.0 + w[j]));
    }
}

/// `hyper_norm_f32` over `rows` rows of `[4, 2560]`.
fn cpu_norm_rows(x: &[f32], w: &[f32], rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * HC_WIDE];
    for r in 0..rows {
        for b in 0..HC_STREAMS {
            let o = r * HC_WIDE + b * HIDDEN;
            cpu_norm_branch(
                &x[o..o + HIDDEN],
                &w[b * HIDDEN..(b + 1) * HIDDEN],
                &mut out[o..o + HIDDEN],
            );
        }
    }
    out
}

/// The write gate (tensor_ops.hip:1480-1486 / qwen4_gemv_bf16_xf32.hip:420-423).
fn cpu_gate(logit: f32) -> f32 {
    let gl = bf(logit);
    let gs = bf(gl / HC_STREAMS as f32);
    bf(2.0 * bf(sigmoid_k(gs)))
}

/// `hyper_write_f32` (tensor_ops.hip:1464-1493), any rows.
fn cpu_write(input: &[f32], mixed: &[f32], gates: &[f32], rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * HC_WIDE];
    for r in 0..rows {
        for col in 0..HC_WIDE {
            let b = col / HIDDEN;
            let gate = cpu_gate(gates[r * HC_STREAMS + b]);
            let inj = bf(gate * bf(mixed[r * HIDDEN + col % HIDDEN]));
            out[r * HC_WIDE + col] = bf(bf(input[r * HC_WIDE + col]) + inj);
        }
    }
    out
}

/// `hyper_read_projected_f32` (tensor_ops.hip:1390-1420), any rows.
fn cpu_read_mix(normalized: &[f32], up: &[f32], rows: usize) -> Vec<f32> {
    let mut mixed = vec![0.0f32; rows * HIDDEN];
    for r in 0..rows {
        for c in 0..HIDDEN {
            let mut v = 0.0f32;
            for b in 0..HC_STREAMS {
                let o = r * HC_WIDE + b * HIDDEN + c;
                let gate = bf(sigmoid_k(bf(up[o])));
                v += bf(gate * bf(normalized[o]));
            }
            mixed[r * HIDDEN + c] = bf(v / HC_STREAMS as f32);
        }
    }
    mixed
}

/// The route's HC-read activation on one low-rank value after the down
/// projection: round, `* 1/branches`, round, SiLU, round (layer_ops.rs HC read
/// non-gfx11 arm == `hc_activation_fused_f32`, tensor_ops.hip:5108-5124).
fn cpu_hc_act(v: f32) -> f32 {
    let v = bf(v);
    let v = bf(v * (1.0 / HC_STREAMS as f32));
    bf(v / (1.0 + (-v).exp()))
}

fn dot64(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum::<f64>() as f32
}

/// Whole HC read for `rows` rows (weights `down [320, wide]`, `up [wide, 320]`).
fn cpu_read(x: &[f32], wn: &[f32], down: &[f32], up: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>) {
    let normalized = cpu_norm_rows(x, wn, rows);
    let mut logits = vec![0.0f32; rows * HC_WIDE];
    for r in 0..rows {
        let n = &normalized[r * HC_WIDE..(r + 1) * HC_WIDE];
        let low: Vec<f32> = (0..HC_LOW)
            .map(|j| cpu_hc_act(dot64(&down[j * HC_WIDE..(j + 1) * HC_WIDE], n)))
            .collect();
        for c in 0..HC_WIDE {
            logits[r * HC_WIDE + c] = dot64(&up[c * HC_LOW..(c + 1) * HC_LOW], &low);
        }
    }
    let mixed = cpu_read_mix(&normalized, &logits, rows);
    (normalized, mixed)
}

/// Gate logits of the write: `inject [4, wide] . normalized` per row.
fn cpu_gates(inject: &[f32], normalized: &[f32], rows: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; rows * HC_STREAMS];
    for r in 0..rows {
        for b in 0..HC_STREAMS {
            g[r * HC_STREAMS + b] = dot64(
                &inject[b * HC_WIDE..(b + 1) * HC_WIDE],
                &normalized[r * HC_WIDE..(r + 1) * HC_WIDE],
            );
        }
    }
    g
}

/// `gemv_bf16_xf32_row`-order quarter dot of one 2560-wide branch
/// (qwen4_gemv_bf16_xf32.hip:451-468): 32 lanes at `lane*8 + s*256`, four
/// chains `(a0 a1 a2 a3)` fed e0..e7, `(a0+a1)+(a2+a3)`, then the lane-0 tree of
/// `sum += __shfl_down(sum, off)`, off = 16..1.
fn cpu_quarter(w: &[f32], x: &[f32]) -> f32 {
    let mut lane_sum = [0.0f32; 32];
    for (lane, out) in lane_sum.iter_mut().enumerate() {
        let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for s in 0..HIDDEN / 256 {
            let o = lane * 8 + s * 256;
            a0 = w[o].mul_add(x[o], a0);
            a1 = w[o + 1].mul_add(x[o + 1], a1);
            a2 = w[o + 2].mul_add(x[o + 2], a2);
            a3 = w[o + 3].mul_add(x[o + 3], a3);
            a0 = w[o + 4].mul_add(x[o + 4], a0);
            a1 = w[o + 5].mul_add(x[o + 5], a1);
            a2 = w[o + 6].mul_add(x[o + 6], a2);
            a3 = w[o + 7].mul_add(x[o + 7], a3);
        }
        *out = (a0 + a1) + (a2 + a3);
    }
    let mut v = lane_sum;
    for off in [16usize, 8, 4, 2, 1] {
        let prev = v;
        for l in 0..32 {
            if l + off < 32 {
                v[l] = prev[l] + prev[l + off];
            }
        }
    }
    v[0]
}

/// 256-point signed FWHT of `mq_fwht256_lane` (mq_fwht256.h) + the epilogue
/// `v * 0.0625 * signs2` of `hyper_read_projected_rotate_f32`
/// (qwen4_gemv_mq4g256.hip:52-59), for one 256-column group.
fn cpu_rotate_group(group: &[f32], s1: &[f32], s2: &[f32]) -> Vec<f32> {
    let mut v: Vec<f32> = (0..256).map(|i| group[i] * s1[i]).collect();
    let mut h = 1;
    while h < 256 {
        for i in 0..256 {
            if i & h == 0 {
                let (a, b) = (v[i], v[i + h]);
                v[i] = a + b;
                v[i + h] = a - b;
            }
        }
        h <<= 1;
    }
    (0..256).map(|i| v[i] * 0.0625 * s2[i]).collect()
}

fn wref(buf: &GpuTensor, m: usize, k: usize) -> WeightRef<'_> {
    WeightRef {
        buf,
        dtype: DType::BF16,
        m,
        k,
        row_stride: k,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

fn de<E: std::fmt::Debug>(err: E) -> String {
    format!("{err:?}")
}

// ── Cases ─────────────────────────────────────────────────────────────────────

/// `hc_streams_init_from_embed_batched` (`kernels/src/hc_streams_init_from_embed_batched.hip:29`):
/// pure broadcast, bit-exact. Launcher: `Gpu::hc_streams_init_from_embed_batched`
/// (the F32-stream arm: `bf16_state` is false on gfx942).
fn case_streams_init(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_1001);
    let mut out = Vec::new();
    for &rows in &HC_ROWS {
        let emb = rng.vec(rows * HIDDEN, -4.0, 4.0);
        let et = up_f32(gpu, &emb)?;
        let st = sentinel_f32(gpu, rows * HC_WIDE)?;
        gpu.hc_streams_init_from_embed_batched(
            &et,
            &st,
            HIDDEN as i32,
            HC_STREAMS as i32,
            rows as i32,
        )
        .map_err(e)?;
        let got = down_f32(gpu, &st)?;
        let mut want = vec![0.0f32; rows * HC_WIDE];
        for r in 0..rows {
            for h in 0..HC_STREAMS {
                want[r * HC_WIDE + h * HIDDEN..r * HC_WIDE + (h + 1) * HIDDEN]
                    .copy_from_slice(&emb[r * HIDDEN..(r + 1) * HIDDEN]);
            }
        }
        out.push(exact_f32(&format!("streams_init rows={rows}"), &got, &want));
        free_all(gpu, vec![et, st])?;
    }
    all(out)
}

/// `hyper_norm_f32` (tensor_ops.hip:1616), F32 streams, grid `[4, rows]`.
fn case_hyper_norm(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_2001);
    let w = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
    let wt = up_bf16(gpu, &w)?;
    let mut out = Vec::new();
    for &rows in &HC_ROWS {
        let x = rng.vec(rows * HC_WIDE, -3.0, 3.0);
        let xt = up_f32(gpu, &x)?;
        let nt = sentinel_f32(gpu, rows * HC_WIDE)?;
        hyper_norm(
            gpu,
            &HyperNorm {
                input: &xt,
                norm_weight: &wt,
                normalized: &nt,
                branches: HC_STREAMS,
                hidden: HIDDEN,
                state_bf16: false,
            },
        )
        .map_err(e)?;
        let got = down_f32(gpu, &nt)?;
        out.push(close_f32(
            &format!("hyper_norm rows={rows}"),
            &got,
            &cpu_norm_rows(&x, &w, rows),
            BF16_INPUT_F32_ACCUM,
        ));
        free_all(gpu, vec![xt, nt])?;
    }
    free_all(gpu, vec![wt])?;
    all(out)
}

/// The non-gfx11 arm of the HC read activation, launch for launch as
/// `execute_hyper_read_inner` issues it (`bf16_roundtrip_f32` over all rows,
/// flat `scale_f32`, round trip, flat `silu_f32`, round trip; the round trip's
/// BF16 staging is a `rows * low_rank` buffer, as the read's `up`).
/// Round trips and the scale are exact; the SiLU stage carries `expf`.
fn case_activation_chain(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_3001);
    let mut out = Vec::new();
    for &rows in &ROW_COUNTS {
        let v = rng.vec(rows * HC_LOW, -6.0, 6.0);
        let low = up_f32(gpu, &v)?;
        let staging = up_bf16(gpu, &vec![0.0f32; rows * HC_LOW])?;
        let roundtrip = |gpu: &mut Gpu| -> Result<(), String> {
            bf16_roundtrip_f32(
                gpu,
                &Bf16Roundtrip {
                    input: &low,
                    scratch: &staging,
                    output: &low,
                    elements: rows * HC_LOW,
                },
            )
            .map_err(e)
        };
        roundtrip(gpu)?;
        let s1: Vec<f32> = v.iter().map(|&x| bf(x)).collect();
        out.push(exact_f32(
            &format!("roundtrip#1 rows={rows}"),
            &down_f32(gpu, &low)?,
            &s1,
        ));
        scale_f32(
            gpu,
            &ScaleF32 {
                values: &low,
                scale: 1.0 / HC_STREAMS as f32,
            },
        )
        .map_err(e)?;
        roundtrip(gpu)?;
        let s2: Vec<f32> = s1.iter().map(|&x| bf(x * 0.25)).collect();
        out.push(exact_f32(
            &format!("scale+roundtrip#2 rows={rows}"),
            &down_f32(gpu, &low)?,
            &s2,
        ));
        gpu.silu_f32(&low, &low).map_err(e)?;
        roundtrip(gpu)?;
        let s3: Vec<f32> = s2.iter().map(|&x| bf(x / (1.0 + (-x).exp()))).collect();
        out.push(close_f32(
            &format!("silu+roundtrip#3 rows={rows}"),
            &down_f32(gpu, &low)?,
            &s3,
            BF16_INPUT_F32_ACCUM,
        ));
        free_all(gpu, vec![low, staging])?;
    }
    all(out)
}

/// `hc_activation_fused_f32` (tensor_ops.hip:5108). NOT on the gfx942 AR route
/// (its caller is gfx11+-gated); the MTP head (`mtp_gpu.rs:2977`, non-goal)
/// launches it unconditionally. Both outputs: F32 in place and packed BF16.
fn case_activation_fused(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_3101);
    let mut out = Vec::new();
    for &rows in &ROW_COUNTS {
        let v = rng.vec(rows * HC_LOW, -6.0, 6.0);
        let want: Vec<f32> = v.iter().map(|&x| cpu_hc_act(x)).collect();
        let low = up_f32(gpu, &v)?;
        hc_activation_fused_f32(
            gpu,
            &HcActivationFused {
                values: &low,
                scale: 0.25,
                bf16_out: None,
            },
        )
        .map_err(e)?;
        out.push(close_f32(
            &format!("activation_fused f32 rows={rows}"),
            &down_f32(gpu, &low)?,
            &want,
            BF16_INPUT_F32_ACCUM,
        ));
        let low2 = up_f32(gpu, &v)?;
        let packed = up_bf16(gpu, &vec![0.0f32; rows * HC_LOW])?;
        hc_activation_fused_f32(
            gpu,
            &HcActivationFused {
                values: &low2,
                scale: 0.25,
                bf16_out: Some(&packed),
            },
        )
        .map_err(e)?;
        let got: Vec<f32> = down_bf16(gpu, &packed)?
            .into_iter()
            .map(bf16_bits_to_f32)
            .collect();
        out.push(close_f32(
            &format!("activation_fused bf16_out rows={rows}"),
            &got,
            &want,
            BF16_INPUT_F32_ACCUM,
        ));
        out.push(exact_f32(
            &format!("activation_fused bf16_out leaves values rows={rows}"),
            &down_f32(gpu, &low2)?,
            &v,
        ));
        free_all(gpu, vec![low, low2, packed])?;
    }
    all(out)
}

/// `hyper_read_projected_f32` (tensor_ops.hip:1390), grid `[ceil(wide/256), rows]`.
/// Only `mixed` is written.
fn case_read_projected(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_4001);
    let w = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
    let wt = up_bf16(gpu, &w)?;
    let mut out = Vec::new();
    for &rows in &HC_ROWS {
        let x = rng.vec(rows * HC_WIDE, -2.0, 2.0);
        let normalized = rng.vec_bf16_exact(rows * HC_WIDE, -3.0, 3.0);
        let up = rng.vec(rows * HC_WIDE, -5.0, 5.0);
        let (xt, nt, ut) = (
            up_f32(gpu, &x)?,
            up_f32(gpu, &normalized)?,
            up_f32(gpu, &up)?,
        );
        let mt = sentinel_f32(gpu, rows * HIDDEN)?;
        hyper_read_projected(
            gpu,
            &HyperReadProjected {
                input: &xt,
                norm_weight: &wt,
                up: &ut,
                normalized: &nt,
                mixed: &mt,
                branches: HC_STREAMS,
                hidden: HIDDEN,
            },
        )
        .map_err(e)?;
        out.push(close_f32(
            &format!("read_projected rows={rows}"),
            &down_f32(gpu, &mt)?,
            &cpu_read_mix(&normalized, &up, rows),
            BF16_INPUT_F32_ACCUM,
        ));
        free_all(gpu, vec![xt, nt, ut, mt])?;
    }
    free_all(gpu, vec![wt])?;
    all(out)
}

/// `hyper_read_projected_rotate_f32` (qwen4_gemv_mq4g256.hip:23): the read's
/// branch mix plus `mq_rotate_x` of `mixed`, one launch (`Gpu::hyper_read_projected_rotate`,
/// reached from `execute_hyper_read_inner` when `rotate_into` is set, i.e. the
/// decode write+read pair followed by a mixer/MoE consuming `mixed`; the FWHT is
/// `ds_swizzle` xor 1..16 inside a 32-lane group, wave 0 lanes 0..31 only).
/// `mixed` vs the host mix (BF16 tol); `rotated` bit-exact against the host FWHT
/// of the kernel's own `mixed`.
fn case_read_projected_rotate(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_5001);
    gpu.ensure_mq_signs().map_err(e)?;
    let s1 = down_f32(
        gpu,
        gpu.scratch.mq_signs1.as_ref().ok_or("mq_signs1 missing")?,
    )?;
    let s2 = down_f32(
        gpu,
        gpu.scratch.mq_signs2.as_ref().ok_or("mq_signs2 missing")?,
    )?;
    let mut out = Vec::new();
    for &rows in &ROW_COUNTS {
        let normalized = rng.vec_bf16_exact(rows * HC_WIDE, -3.0, 3.0);
        let up = rng.vec(rows * HC_WIDE, -5.0, 5.0);
        let (nt, ut) = (up_f32(gpu, &normalized)?, up_f32(gpu, &up)?);
        let mt = sentinel_f32(gpu, rows * HIDDEN)?;
        let rt = sentinel_f32(gpu, rows * HIDDEN)?;
        gpu.hyper_read_projected_rotate(&nt, &ut, &mt, &rt, HIDDEN, rows)
            .map_err(e)?;
        gpu.scratch.prerotated = None;
        let got_mixed = down_f32(gpu, &mt)?;
        out.push(close_f32(
            &format!("read_projected_rotate mixed rows={rows}"),
            &got_mixed,
            &cpu_read_mix(&normalized, &up, rows),
            BF16_INPUT_F32_ACCUM,
        ));
        let mut want_rot = Vec::with_capacity(rows * HIDDEN);
        for g in 0..rows * HIDDEN / 256 {
            want_rot.extend(cpu_rotate_group(
                &got_mixed[g * 256..(g + 1) * 256],
                &s1,
                &s2,
            ));
        }
        out.push(exact_f32(
            &format!("read_projected_rotate rotated rows={rows}"),
            &down_f32(gpu, &rt)?,
            &want_rot,
        ));
        free_all(gpu, vec![nt, ut, mt, rt])?;
    }
    all(out)
}

/// `hyper_write_f32` (tensor_ops.hip:1464), in place (`input == output` as the
/// route passes `scratch.streams` twice), grid `[ceil(wide/256), rows]`.
fn case_hyper_write(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_6001);
    let mut out = Vec::new();
    for &rows in &HC_ROWS {
        let x = rng.vec(rows * HC_WIDE, -3.0, 3.0);
        let mixed = rng.vec(rows * HIDDEN, -1.0, 1.0);
        let gates = rng.vec(rows * HC_STREAMS, -6.0, 6.0);
        let st = up_f32(gpu, &x)?;
        let nt = zeros_f32(gpu, rows * HC_WIDE)?;
        let (mt, gt) = (up_f32(gpu, &mixed)?, up_f32(gpu, &gates)?);
        hyper_write(
            gpu,
            &HyperWrite {
                input: &st,
                normalized: &nt,
                mixed: &mt,
                gates: &gt,
                output: &st,
                branches: HC_STREAMS,
                hidden: HIDDEN,
                state_bf16: false,
            },
        )
        .map_err(e)?;
        out.push(close_f32(
            &format!("hyper_write rows={rows}"),
            &down_f32(gpu, &st)?,
            &cpu_write(&x, &mixed, &gates, rows),
            BF16_INPUT_F32_ACCUM,
        ));
        free_all(gpu, vec![st, nt, mt, gt])?;
    }
    all(out)
}

/// `hyper_write_norm_f32` (qwen4_gemv_bf16_xf32.hip:361), the decode write +
/// next-read-norm kernel (rows 1..=8; the only HC kernel with 32-lane `wave`/`lane`
/// arithmetic and a width-less `__shfl_down` reachable on gfx942: lane 0 / lane 32
/// of each 64-lane wave each reduce their own 32-lane half).
/// Stage A: gates given, `next` (quarters out) + `clear`. Stage B: gates taken from
/// stage A's quarters (`quarters_in`), `gates` NaN-poisoned (must not be read).
fn case_hyper_write_norm(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_7001);
    const CLEAR_N: usize = 20_000;
    let mut out = Vec::new();
    for &rows in &DECODE_ROWS {
        let wn = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
        let nn = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
        let inj = rng.vec_bf16_exact(4 * HC_WIDE, -0.05, 0.05);
        let (wnt, nnt, injt) = (up_bf16(gpu, &wn)?, up_bf16(gpu, &nn)?, up_bf16(gpu, &inj)?);

        // Stage A.
        let s0 = rng.vec(rows * HC_WIDE, -2.0, 2.0);
        let mix = rng.vec(rows * HIDDEN, -1.0, 1.0);
        let gl = rng.vec(rows * HC_STREAMS, -6.0, 6.0);
        let st = up_f32(gpu, &s0)?;
        let (mt, gt) = (up_f32(gpu, &mix)?, up_f32(gpu, &gl)?);
        let nt = sentinel_f32(gpu, rows * HC_WIDE)?;
        let qt = sentinel_f32(gpu, 16 * rows)?;
        let clr = up_f32(gpu, &vec![7.0f32; CLEAR_N])?;
        let next = HyperNextGates {
            norm_weight: &nnt,
            inject: &injt,
            quarters: &qt,
        };
        hyper_write_norm(
            gpu,
            &HyperWrite {
                input: &st,
                normalized: &nt,
                mixed: &mt,
                gates: &gt,
                output: &st,
                branches: HC_STREAMS,
                hidden: HIDDEN,
                state_bf16: false,
            },
            &wnt,
            &nt,
            None,
            Some(&next),
            Some(&clr),
        )
        .map_err(e)?;
        let want_s1 = cpu_write(&s0, &mix, &gl, rows);
        out.push(close_f32(
            &format!("write_norm A streams rows={rows}"),
            &down_f32(gpu, &st)?,
            &want_s1,
            BF16_INPUT_F32_ACCUM,
        ));
        out.push(close_f32(
            &format!("write_norm A normalized rows={rows}"),
            &down_f32(gpu, &nt)?,
            &cpu_norm_rows(&want_s1, &wn, rows),
            BF16_INPUT_F32_ACCUM,
        ));
        // Quarters: bf16-quantized normalized input (next norm) x BF16 inject, F32 accumulation.
        let xn = cpu_norm_rows(&want_s1, &nn, rows);
        let mut want_q = vec![0.0f32; 16 * rows];
        for r in 0..rows {
            for g in 0..4 {
                for b in 0..HC_STREAMS {
                    want_q[r * 16 + g * 4 + b] = cpu_quarter(
                        &inj[g * HC_WIDE + b * HIDDEN..g * HC_WIDE + (b + 1) * HIDDEN],
                        &xn[r * HC_WIDE + b * HIDDEN..r * HC_WIDE + (b + 1) * HIDDEN],
                    );
                }
            }
        }
        let q_gpu = down_f32(gpu, &qt)?;
        out.push(close_f32(
            &format!("write_norm A quarters_out rows={rows}"),
            &q_gpu,
            &want_q,
            BF16_INPUT_F32_ACCUM,
        ));
        out.push(exact_f32(
            &format!("write_norm A clear rows={rows}"),
            &down_f32(gpu, &clr)?,
            &vec![0.0f32; CLEAR_N],
        ));

        // Stage B: gates from stage A's quarters.
        let s1 = rng.vec(rows * HC_WIDE, -2.0, 2.0);
        let mix2 = rng.vec(rows * HIDDEN, -1.0, 1.0);
        let st2 = up_f32(gpu, &s1)?;
        let mt2 = up_f32(gpu, &mix2)?;
        let gt2 = sentinel_f32(gpu, rows * HC_STREAMS)?;
        let nt2 = sentinel_f32(gpu, rows * HC_WIDE)?;
        hyper_write_norm(
            gpu,
            &HyperWrite {
                input: &st2,
                normalized: &nt2,
                mixed: &mt2,
                gates: &gt2,
                output: &st2,
                branches: HC_STREAMS,
                hidden: HIDDEN,
                state_bf16: false,
            },
            &wnt,
            &nt2,
            Some(&qt),
            None,
            None,
        )
        .map_err(e)?;
        // `gates` for the host: the kernel's (q0 + q1) + (q2 + q3) fold of the device quarters.
        let mut gl2 = vec![0.0f32; rows * HC_STREAMS];
        for r in 0..rows {
            for b in 0..HC_STREAMS {
                let q = &q_gpu[r * 16 + b * 4..r * 16 + b * 4 + 4];
                gl2[r * HC_STREAMS + b] = (q[0] + q[1]) + (q[2] + q[3]);
            }
        }
        let want_s2 = cpu_write(&s1, &mix2, &gl2, rows);
        out.push(close_f32(
            &format!("write_norm B(quarters_in) streams rows={rows}"),
            &down_f32(gpu, &st2)?,
            &want_s2,
            BF16_INPUT_F32_ACCUM,
        ));
        out.push(close_f32(
            &format!("write_norm B(quarters_in) normalized rows={rows}"),
            &down_f32(gpu, &nt2)?,
            &cpu_norm_rows(&want_s2, &wn, rows),
            BF16_INPUT_F32_ACCUM,
        ));
        free_all(
            gpu,
            vec![wnt, nnt, injt, st, mt, gt, nt, qt, clr, st2, mt2, gt2, nt2],
        )?;
    }
    all(out)
}

/// `HIPFIRE_QWEN4_G942_HC_QUARTERS_GATES`: the gates an UNFUSED write computes
/// (`execute_hyper_write` -> `hyper_write_gates`) are bitwise the gates the fused
/// `hyper_write_norm_f32` produces as the next write's quarters, rows 1..=8 (the route's
/// few-row range on gfx942). Per row count: `hyper_write_norm` with `next` writes the streams
/// and the quarters for the next write (norm weight `nn`, inject `inj`); the next write then runs
/// through `execute_hyper_write` on those streams with the same weights and its gates, a
/// sentinel-poisoned buffer the route must fill, are compared bit for bit with the quarters'
/// `(q0 + q1) + (q2 + q3)` fold (`hyper_write_norm` stage B's own gate expression).
fn case_unfused_gates_bitwise_quarters(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_7B01);
    let mut out = Vec::new();
    for rows in 1..=8usize {
        let wn = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
        let nn = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
        let inj = rng.vec_bf16_exact(4 * HC_WIDE, -0.05, 0.05);
        let (wnt, nnt, injt) = (up_bf16(gpu, &wn)?, up_bf16(gpu, &nn)?, up_bf16(gpu, &inj)?);
        let s0 = rng.vec(rows * HC_WIDE, -2.0, 2.0);
        let mix = rng.vec(rows * HIDDEN, -1.0, 1.0);
        let gl = rng.vec(rows * HC_STREAMS, -6.0, 6.0);
        let st = up_f32(gpu, &s0)?;
        let (mt, gt) = (up_f32(gpu, &mix)?, up_f32(gpu, &gl)?);
        let nt = sentinel_f32(gpu, rows * HC_WIDE)?;
        let qt = sentinel_f32(gpu, 16 * rows)?;
        let next = HyperNextGates {
            norm_weight: &nnt,
            inject: &injt,
            quarters: &qt,
        };
        hyper_write_norm(
            gpu,
            &HyperWrite {
                input: &st,
                normalized: &nt,
                mixed: &mt,
                gates: &gt,
                output: &st,
                branches: HC_STREAMS,
                hidden: HIDDEN,
                state_bf16: false,
            },
            &wnt,
            &nt,
            None,
            Some(&next),
            None,
        )
        .map_err(e)?;
        let q_gpu = down_f32(gpu, &qt)?;
        let mut want = vec![0.0f32; rows * HC_STREAMS];
        for r in 0..rows {
            for g in 0..HC_STREAMS {
                let q = &q_gpu[r * 16 + g * 4..r * 16 + g * 4 + 4];
                want[r * HC_STREAMS + g] = (q[0] + q[1]) + (q[2] + q[3]);
            }
        }
        // The next write, unfused, on the streams the fused launch just wrote.
        let mix2 = rng.vec(rows * HIDDEN, -1.0, 1.0);
        let mt2 = up_f32(gpu, &mix2)?;
        let nt2 = sentinel_f32(gpu, rows * HC_WIDE)?;
        let gt2 = sentinel_f32(gpu, rows * HC_STREAMS)?;
        let rot = zeros_f32(gpu, rows * HC_WIDE)?;
        let op = HyperWriteOp {
            state_bf16: false,
            input: &st,
            norm_weight: &nnt,
            block_inject: wref(&injt, HC_STREAMS, HC_WIDE),
            normalized: &nt2,
            mixed: &mt2,
            gates: &gt2,
            output: &st,
            rows,
            branches: HC_STREAMS,
            hidden: HIDDEN,
            rotation: &rot,
            ar: None,
        };
        execute_hyper_write(gpu, &op).map_err(de)?;
        out.push(exact_f32(
            &format!("unfused gates == fused quarters fold rows={rows}"),
            &down_f32(gpu, &gt2)?,
            &want,
        ));
        free_all(
            gpu,
            vec![wnt, nnt, injt, st, mt, gt, nt, qt, mt2, nt2, gt2, rot],
        )?;
    }
    all(out)
}

/// Shared by the HC read route cases: `execute_hyper_read` / `execute_final_hyper`
/// (the final HC read is the same `HyperReadOp`, `layer_ops.rs execute_final_hyper`).
/// Exercises, on gfx942: `hyper_norm_f32`, the BF16 down GEMM, the per-row
/// round-trip / scale / SiLU arm, the BF16 up GEMM and `hyper_read_projected_f32`.
fn run_route_read(gpu: &mut Gpu, final_read: bool) -> Result<String, String> {
    let mut rng = Rng::new(if final_read { 0x4843_8101 } else { 0x4843_8001 });
    let wn = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
    let down = rng.vec_bf16_exact(HC_LOW * HC_WIDE, -0.02, 0.02);
    let upw = rng.vec_bf16_exact(HC_WIDE * HC_LOW, -0.1, 0.1);
    let (wnt, downt, upt) = (
        up_bf16(gpu, &wn)?,
        up_bf16(gpu, &down)?,
        up_bf16(gpu, &upw)?,
    );
    let mut out = Vec::new();
    for &rows in &ROW_COUNTS {
        let x = rng.vec(rows * HC_WIDE, -2.0, 2.0);
        let xt = up_f32(gpu, &x)?;
        let nt = sentinel_f32(gpu, rows * HC_WIDE)?;
        let lowt = sentinel_f32(gpu, rows * HC_LOW)?;
        let uot = sentinel_f32(gpu, rows * HC_WIDE)?;
        let mt = sentinel_f32(gpu, rows * HIDDEN)?;
        let rot = zeros_f32(gpu, rows * HC_WIDE)?;
        let op = HyperReadOp {
            state_bf16: false,
            input: &xt,
            norm_weight: &wnt,
            input_mix_down: wref(&downt, HC_LOW, HC_WIDE),
            input_mix_up: wref(&upt, HC_WIDE, HC_LOW),
            normalized: &nt,
            low: &lowt,
            up: &uot,
            mixed: &mt,
            rows,
            branches: HC_STREAMS,
            hidden: HIDDEN,
            low_rank: HC_LOW,
            rotation: &rot,
        };
        if final_read {
            execute_final_hyper(gpu, &op).map_err(de)?;
        } else {
            execute_hyper_read(gpu, &op).map_err(de)?;
        }
        let (want_n, want_m) = cpu_read(&x, &wn, &down, &upw, rows);
        out.push(close_f32(
            &format!("read normalized rows={rows}"),
            &down_f32(gpu, &nt)?,
            &want_n,
            BF16_INPUT_F32_ACCUM,
        ));
        out.push(close_f32(
            &format!("read mixed rows={rows}"),
            &down_f32(gpu, &mt)?,
            &want_m,
            BF16_INPUT_F32_ACCUM,
        ));
        free_all(gpu, vec![xt, nt, lowt, uot, mt, rot])?;
    }
    free_all(gpu, vec![wnt, downt, upt])?;
    all(out)
}

fn case_route_read(gpu: &mut Gpu) -> Result<String, String> {
    run_route_read(gpu, false)
}

fn case_route_final_read(gpu: &mut Gpu) -> Result<String, String> {
    run_route_read(gpu, true)
}

/// `HIPFIRE_QWEN4_G942_FOLD_ELEM` kernel pin: `qwen4_g942_gemm_bf16_rows16_hc_act`
/// (`qwen4_g942_glue_elem.hip`, via `Gfx942Device::qwen4_hc_down_act_rows16`) must be
/// BYTE-equal to the unfused launches it replaces on the same inputs: the rows16 BF16
/// GEMM into F32, then `bf16_roundtrip_f32`, `scale_f32`, `bf16_roundtrip_f32`,
/// `silu_f32`, `bf16_roundtrip_f32` (as `execute_hyper_read_inner` issues them). Every
/// row count 1..=16 (both entries: `_x2` takes an F32 X of 9..=16 rows), F32 and BF16
/// X, three magnitude regimes (silu tails, mid range, near-zero) and a poisoned row
/// (Inf in X: NaN / Inf accumulators go through the same casts).
fn case_fold_elem_hc_down_act(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_F001);
    let scale = 1.0 / HC_STREAMS as f32;
    let regimes: [(&str, f32, f32); 3] = [
        ("mid", 0.05, 4.0),
        ("wide", 2.0, 64.0),
        ("tiny", 1.0e-4, 1.0e-2),
    ];
    let mut out = Vec::new();
    for (name, w_amp, x_amp) in regimes {
        let down = rng.vec_bf16_exact(HC_LOW * HC_WIDE, -w_amp, w_amp);
        let downt = up_bf16(gpu, &down)?;
        for rows in 1..=16usize {
            for x_bf16 in [false, true] {
                let mut x = rng.vec(rows * HC_WIDE, -x_amp, x_amp);
                if rows == 3 {
                    x[HC_WIDE + 5] = f32::INFINITY;
                }
                let xt = if x_bf16 {
                    up_bf16(gpu, &x)?
                } else {
                    up_f32(gpu, &x)?
                };
                let staging = up_bf16(gpu, &vec![0.0f32; rows * HC_LOW])?;
                let want_t = sentinel_f32(gpu, rows * HC_LOW)?;
                let launched = gpu
                    .gemm_bf16_mfma_gfx942_rows16(
                        &xt,
                        &[(&downt.buf, &want_t, HC_LOW)],
                        HC_WIDE,
                        rows,
                    )
                    .map_err(e)?;
                if !launched {
                    return Err(format!("unfused rows16 declined: {name} rows={rows}"));
                }
                let roundtrip = |gpu: &mut Gpu| -> Result<(), String> {
                    bf16_roundtrip_f32(
                        gpu,
                        &Bf16Roundtrip {
                            input: &want_t,
                            scratch: &staging,
                            output: &want_t,
                            elements: rows * HC_LOW,
                        },
                    )
                    .map_err(e)
                };
                roundtrip(gpu)?;
                scale_f32(
                    gpu,
                    &ScaleF32 {
                        values: &want_t,
                        scale,
                    },
                )
                .map_err(e)?;
                roundtrip(gpu)?;
                gpu.silu_f32(&want_t, &want_t).map_err(e)?;
                roundtrip(gpu)?;
                let got_t = sentinel_f32(gpu, rows * HC_LOW)?;
                let folded = gpu
                    .try_gfx942()
                    .ok_or_else(|| "not gfx942".to_string())?
                    .qwen4_hc_down_act_rows16(&downt.buf, &xt, &got_t, HC_LOW, HC_WIDE, rows, scale)
                    .map_err(e)?;
                if !folded {
                    return Err(format!("folded entry declined: {name} rows={rows}"));
                }
                let (got, want) = (down_f32(gpu, &got_t)?, down_f32(gpu, &want_t)?);
                out.push(exact_f32(
                    &format!(
                        "fold_elem hc_down_act {name} rows={rows} x={}",
                        if x_bf16 { "bf16" } else { "f32" }
                    ),
                    &got,
                    &want,
                ));
                free_all(gpu, vec![xt, staging, want_t, got_t])?;
            }
        }
        free_all(gpu, vec![downt])?;
    }
    all(out)
}

/// Route digest for the `HIPFIRE_QWEN4_G942_FOLD_ELEM` on/off A/B: `execute_hyper_read`
/// at rows 1..=17 (17 declines the fold: its rows16 entry takes 16) with the production
/// F32 streams; SHA-256 over `normalized`, `low` and `mixed` bytes of every row count.
/// Run the case under each setting of the switch: the two summaries must carry the same
/// digest (the harness is one supervised process per case, so the switch is read once).
fn case_fold_elem_route_digest(gpu: &mut Gpu) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut rng = Rng::new(0x4843_F101);
    let wn = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
    let down = rng.vec_bf16_exact(HC_LOW * HC_WIDE, -0.02, 0.02);
    let upw = rng.vec_bf16_exact(HC_WIDE * HC_LOW, -0.1, 0.1);
    let (wnt, downt, upt) = (
        up_bf16(gpu, &wn)?,
        up_bf16(gpu, &down)?,
        up_bf16(gpu, &upw)?,
    );
    let mut digest = Sha256::new();
    for rows in 1..=17usize {
        let x = rng.vec(rows * HC_WIDE, -2.0, 2.0);
        let xt = up_f32(gpu, &x)?;
        let nt = sentinel_f32(gpu, rows * HC_WIDE)?;
        let lowt = sentinel_f32(gpu, rows * HC_LOW)?;
        let uot = sentinel_f32(gpu, rows * HC_WIDE)?;
        let mt = sentinel_f32(gpu, rows * HIDDEN)?;
        let rot = zeros_f32(gpu, rows * HC_WIDE)?;
        let op = HyperReadOp {
            state_bf16: false,
            input: &xt,
            norm_weight: &wnt,
            input_mix_down: wref(&downt, HC_LOW, HC_WIDE),
            input_mix_up: wref(&upt, HC_WIDE, HC_LOW),
            normalized: &nt,
            low: &lowt,
            up: &uot,
            mixed: &mt,
            rows,
            branches: HC_STREAMS,
            hidden: HIDDEN,
            low_rank: HC_LOW,
            rotation: &rot,
        };
        execute_hyper_read(gpu, &op).map_err(de)?;
        for t in [&nt, &lowt, &mt] {
            digest.update(down_bytes(gpu, t)?);
        }
        free_all(gpu, vec![xt, nt, lowt, uot, mt, rot])?;
    }
    free_all(gpu, vec![wnt, downt, upt])?;
    let digest = digest.finalize();
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    Ok(format!("fold_elem route digest rows 1..=17 sha256={hex}"))
}

/// `execute_hyper_write` (unfused path: gates = `hyper_norm` + BF16 gate GEMM,
/// then `hyper_write_f32`), in place on the streams as the route does.
fn case_route_write(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_9001);
    let wn = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
    let inj = rng.vec_bf16_exact(4 * HC_WIDE, -0.05, 0.05);
    let (wnt, injt) = (up_bf16(gpu, &wn)?, up_bf16(gpu, &inj)?);
    let mut out = Vec::new();
    for &rows in &ROW_COUNTS {
        let x = rng.vec(rows * HC_WIDE, -2.0, 2.0);
        let mix = rng.vec(rows * HIDDEN, -1.0, 1.0);
        let st = up_f32(gpu, &x)?;
        let nt = sentinel_f32(gpu, rows * HC_WIDE)?;
        let mt = up_f32(gpu, &mix)?;
        let gt = sentinel_f32(gpu, rows * HC_STREAMS)?;
        let rot = zeros_f32(gpu, rows * HC_WIDE)?;
        let op = HyperWriteOp {
            ar: None,
            state_bf16: false,
            input: &st,
            norm_weight: &wnt,
            block_inject: wref(&injt, HC_STREAMS, HC_WIDE),
            normalized: &nt,
            mixed: &mt,
            gates: &gt,
            output: &st,
            rows,
            branches: HC_STREAMS,
            hidden: HIDDEN,
            rotation: &rot,
        };
        execute_hyper_write(gpu, &op).map_err(de)?;
        let normalized = cpu_norm_rows(&x, &wn, rows);
        let gates = cpu_gates(&inj, &normalized, rows);
        out.push(close_f32(
            &format!("write gates rows={rows}"),
            &down_f32(gpu, &gt)?,
            &gates,
            BF16_INPUT_F32_ACCUM,
        ));
        out.push(close_f32(
            &format!("write streams rows={rows}"),
            &down_f32(gpu, &st)?,
            &cpu_write(&x, &mix, &gates, rows),
            BF16_INPUT_F32_ACCUM,
        ));
        free_all(gpu, vec![st, nt, mt, gt, rot])?;
    }
    free_all(gpu, vec![wnt, injt])?;
    all(out)
}

/// The step-program path: `[HyperWrite A, HyperRead A, HyperWrite B, HyperRead B]`
/// on one stream tensor through `execute_steps`. At rows 1..=8 (F32 streams,
/// hidden 2560) the interpreter takes `execute_hyper_write_then_read`:
/// `hyper_write_norm_f32` (A produces B's gate quarters, B consumes them) followed
/// by the read tail; at rows 33 it runs the unfused write + read. The final HC
/// streams and both reads' `mixed` are compared with the host chain.
fn case_step_program(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_A001);
    let wn_a = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
    let wn_b = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
    let rn_a = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
    let rn_b = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
    let inj_a = rng.vec_bf16_exact(4 * HC_WIDE, -0.05, 0.05);
    let inj_b = rng.vec_bf16_exact(4 * HC_WIDE, -0.05, 0.05);
    let down = rng.vec_bf16_exact(HC_LOW * HC_WIDE, -0.02, 0.02);
    let upw = rng.vec_bf16_exact(HC_WIDE * HC_LOW, -0.1, 0.1);
    let (wn_at, wn_bt, rn_at, rn_bt) = (
        up_bf16(gpu, &wn_a)?,
        up_bf16(gpu, &wn_b)?,
        up_bf16(gpu, &rn_a)?,
        up_bf16(gpu, &rn_b)?,
    );
    let (inj_at, inj_bt, downt, upt) = (
        up_bf16(gpu, &inj_a)?,
        up_bf16(gpu, &inj_b)?,
        up_bf16(gpu, &down)?,
        up_bf16(gpu, &upw)?,
    );
    let mut out = Vec::new();
    // (rows, prefill_invariant_scope): gfx942 prefill pins the invariant scope
    // (`gpu_forward.rs trunk_prefill_invariant_scope`), so `route_rows` is 8192
    // and the write+read pair is never fused there; decode (scope off) fuses at <= 8 rows.
    for &(rows, invariant) in &[
        (1usize, false),
        (7, false),
        (8, false),
        (1, true),
        (7, true),
        (33, true),
    ] {
        let s0 = rng.vec(rows * HC_WIDE, -2.0, 2.0);
        let mix_a = rng.vec(rows * HIDDEN, -1.0, 1.0);
        let mix_b = rng.vec(rows * HIDDEN, -1.0, 1.0);
        let st = up_f32(gpu, &s0)?;
        let (mat, mbt) = (up_f32(gpu, &mix_a)?, up_f32(gpu, &mix_b)?);
        // Gate scratch with room for two quarter slots (`hyper_gate_quarters`).
        let gat = sentinel_f32(gpu, rows * HC_STREAMS + 32 * rows)?;
        let gbt = sentinel_f32(gpu, rows * HC_STREAMS + 32 * rows)?;
        let nt = sentinel_f32(gpu, rows * HC_WIDE)?;
        let lowt = sentinel_f32(gpu, rows * HC_LOW)?;
        let uot = sentinel_f32(gpu, rows * HC_WIDE)?;
        let out_a = sentinel_f32(gpu, rows * HIDDEN)?;
        let out_b = sentinel_f32(gpu, rows * HIDDEN)?;
        let rot = zeros_f32(gpu, rows * HC_WIDE)?;
        let write = |norm_weight, inject, mixed, gates| HyperWriteOp {
            ar: None,
            state_bf16: false,
            input: &st,
            norm_weight,
            block_inject: wref(inject, HC_STREAMS, HC_WIDE),
            normalized: &nt,
            mixed,
            gates,
            output: &st,
            rows,
            branches: HC_STREAMS,
            hidden: HIDDEN,
            rotation: &rot,
        };
        let read = |norm_weight, mixed| HyperReadOp {
            state_bf16: false,
            input: &st,
            norm_weight,
            input_mix_down: wref(&downt, HC_LOW, HC_WIDE),
            input_mix_up: wref(&upt, HC_WIDE, HC_LOW),
            normalized: &nt,
            low: &lowt,
            up: &uot,
            mixed,
            rows,
            branches: HC_STREAMS,
            hidden: HIDDEN,
            low_rank: HC_LOW,
            rotation: &rot,
        };
        let steps = vec![
            Step::HyperWrite(write(&wn_at, &inj_at, &mat, &gat)),
            Step::HyperRead(read(&rn_at, &out_a)),
            Step::HyperWrite(write(&wn_bt, &inj_bt, &mbt, &gbt)),
            Step::HyperRead(read(&rn_bt, &out_b)),
        ];
        let ctx = DispatchCtx::new(gpu);
        let prior_scope = std::mem::replace(&mut gpu.prefill_invariant_scope, invariant);
        let ran = execute_steps(gpu, &ctx, &steps);
        gpu.prefill_invariant_scope = prior_scope;
        ran.map_err(de)?;
        drop(steps);

        // Host chain.
        let g_a = cpu_gates(&inj_a, &cpu_norm_rows(&s0, &wn_a, rows), rows);
        let s1 = cpu_write(&s0, &mix_a, &g_a, rows);
        let (_, m_a) = cpu_read(&s1, &rn_a, &down, &upw, rows);
        let g_b = cpu_gates(&inj_b, &cpu_norm_rows(&s1, &wn_b, rows), rows);
        let s2 = cpu_write(&s1, &mix_b, &g_b, rows);
        let (_, m_b) = cpu_read(&s2, &rn_b, &down, &upw, rows);
        out.push(close_f32(
            &format!("program read A mixed rows={rows} invariant={invariant}"),
            &down_f32(gpu, &out_a)?,
            &m_a,
            BF16_INPUT_F32_ACCUM,
        ));
        out.push(close_f32(
            &format!("program final streams rows={rows} invariant={invariant}"),
            &down_f32(gpu, &st)?,
            &s2,
            BF16_INPUT_F32_ACCUM,
        ));
        out.push(close_f32(
            &format!("program read B mixed rows={rows} invariant={invariant}"),
            &down_f32(gpu, &out_b)?,
            &m_b,
            BF16_INPUT_F32_ACCUM,
        ));
        free_all(
            gpu,
            vec![st, mat, mbt, gat, gbt, nt, lowt, uot, out_a, out_b, rot],
        )?;
    }
    free_all(
        gpu,
        vec![wn_at, wn_bt, rn_at, rn_bt, inj_at, inj_bt, downt, upt],
    )?;
    all(out)
}

/// Rows an exact-gfx942 lane step's all-rows HC segment covers (and the counts around the 16-row
/// slice and 8-row tile boundaries).
const LANE_WIDE_ROWS: [usize; 8] = [9, 16, 17, 31, 32, 33, 48, 64];

/// Runs `f` under the scopes `forward_lanes` opens for its all-rows HC steps (the lane-rows
/// scope and the wide verify scope of `rows` rows) and restores them.
fn with_lane_scope<T>(
    gpu: &mut Gpu,
    rows: usize,
    f: impl FnOnce(&mut Gpu) -> Result<T, String>,
) -> Result<T, String> {
    let prior_lane = std::mem::replace(&mut gpu.lane_rows_scope, true);
    let prior_wide = gpu.set_wide_verify_rows(rows);
    let result = f(gpu);
    gpu.set_wide_verify_rows(prior_wide);
    gpu.lane_rows_scope = prior_lane;
    result
}

/// A lane step's all-rows `execute_hyper_write` (gates + norm + streams) is bitwise the one-row
/// singleton call of every row (the pinned quarters gates), at every width past the few-row
/// window; with the lane scope closed the chunk keeps its BF16 MFMA gates, which differ from
/// the singleton's (the control that the comparison can see the arithmetic).
fn case_lane_wide_write(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_C101);
    let wn = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
    let inj = rng.vec_bf16_exact(4 * HC_WIDE, -0.05, 0.05);
    let (wnt, injt) = (up_bf16(gpu, &wn)?, up_bf16(gpu, &inj)?);
    let mut out = Vec::new();
    for &rows in &LANE_WIDE_ROWS {
        let x = rng.vec(rows * HC_WIDE, -2.0, 2.0);
        let mix = rng.vec(rows * HIDDEN, -1.0, 1.0);
        let run = |gpu: &mut Gpu,
                   single: bool,
                   lane: bool|
         -> Result<(Vec<f32>, Vec<f32>), String> {
            let st = up_f32(gpu, &x)?;
            let nt = sentinel_f32(gpu, rows * HC_WIDE)?;
            let mt = up_f32(gpu, &mix)?;
            let gt = sentinel_f32(gpu, rows * HC_STREAMS)?;
            let rot = zeros_f32(gpu, rows * HC_WIDE)?;
            let call = |gpu: &mut Gpu, r0: usize, n: usize| -> Result<(), String> {
                let (sv, nv, mv, gv) = (
                    st.sub_offset(r0 * HC_WIDE, n * HC_WIDE),
                    nt.sub_offset(r0 * HC_WIDE, n * HC_WIDE),
                    mt.sub_offset(r0 * HIDDEN, n * HIDDEN),
                    gt.sub_offset(r0 * HC_STREAMS, n * HC_STREAMS),
                );
                let op = HyperWriteOp {
                    ar: None,
                    state_bf16: false,
                    input: &sv,
                    norm_weight: &wnt,
                    block_inject: wref(&injt, HC_STREAMS, HC_WIDE),
                    normalized: &nv,
                    mixed: &mv,
                    gates: &gv,
                    output: &sv,
                    rows: n,
                    branches: HC_STREAMS,
                    hidden: HIDDEN,
                    rotation: &rot,
                };
                execute_hyper_write(gpu, &op).map_err(de)
            };
            if single {
                for r in 0..rows {
                    call(gpu, r, 1)?;
                }
            } else if lane {
                with_lane_scope(gpu, rows, |gpu| call(gpu, 0, rows))?;
            } else {
                call(gpu, 0, rows)?;
            }
            let got = (down_f32(gpu, &gt)?, down_f32(gpu, &st)?);
            free_all(gpu, vec![st, nt, mt, gt, rot])?;
            Ok(got)
        };
        let (g_single, s_single) = run(gpu, true, false)?;
        let (g_lane, s_lane) = run(gpu, false, true)?;
        let (g_chunk, _) = run(gpu, false, false)?;
        out.push(exact_f32(
            &format!("lane-wide write gates rows={rows} == singles"),
            &g_lane,
            &g_single,
        ));
        out.push(exact_f32(
            &format!("lane-wide write streams rows={rows} == singles"),
            &s_lane,
            &s_single,
        ));
        let differing = g_chunk
            .iter()
            .zip(&g_single)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        out.push(if differing > 0 {
            Ok(format!(
                "chunk MFMA gates rows={rows} differ from the singleton in {differing} words (control)"
            ))
        } else {
            Err(format!(
                "chunk gates rows={rows} equal the singleton's: the control has no power"
            ))
        });
    }
    free_all(gpu, vec![wnt, injt])?;
    all(out)
}

/// A lane step's all-rows `execute_hyper_read` / `execute_final_hyper` (norm, down + activation, up, mix) is bitwise the one-row singleton call of every row.
fn case_lane_wide_read(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4843_C201);
    let wn = rng.vec_bf16_exact(HC_WIDE, -0.5, 0.5);
    let down = rng.vec_bf16_exact(HC_LOW * HC_WIDE, -0.02, 0.02);
    let upw = rng.vec_bf16_exact(HC_WIDE * HC_LOW, -0.1, 0.1);
    let (wnt, downt, upt) = (
        up_bf16(gpu, &wn)?,
        up_bf16(gpu, &down)?,
        up_bf16(gpu, &upw)?,
    );
    let mut out = Vec::new();
    for final_read in [false, true] {
        for &rows in &LANE_WIDE_ROWS {
            let x = rng.vec(rows * HC_WIDE, -2.0, 2.0);
            let run = |gpu: &mut Gpu, single: bool| -> Result<[Vec<f32>; 3], String> {
                let xt = up_f32(gpu, &x)?;
                let nt = sentinel_f32(gpu, rows * HC_WIDE)?;
                let lowt = sentinel_f32(gpu, rows * HC_LOW)?;
                let uot = sentinel_f32(gpu, rows * HC_WIDE)?;
                let mt = sentinel_f32(gpu, rows * HIDDEN)?;
                let rot = zeros_f32(gpu, rows * HC_WIDE)?;
                let call = |gpu: &mut Gpu, r0: usize, n: usize| -> Result<(), String> {
                    let (xv, nv, lv, uv, mv) = (
                        xt.sub_offset(r0 * HC_WIDE, n * HC_WIDE),
                        nt.sub_offset(r0 * HC_WIDE, n * HC_WIDE),
                        lowt.sub_offset(r0 * HC_LOW, n * HC_LOW),
                        uot.sub_offset(r0 * HC_WIDE, n * HC_WIDE),
                        mt.sub_offset(r0 * HIDDEN, n * HIDDEN),
                    );
                    let op = HyperReadOp {
                        state_bf16: false,
                        input: &xv,
                        norm_weight: &wnt,
                        input_mix_down: wref(&downt, HC_LOW, HC_WIDE),
                        input_mix_up: wref(&upt, HC_WIDE, HC_LOW),
                        normalized: &nv,
                        low: &lv,
                        up: &uv,
                        mixed: &mv,
                        rows: n,
                        branches: HC_STREAMS,
                        hidden: HIDDEN,
                        low_rank: HC_LOW,
                        rotation: &rot,
                    };
                    if final_read {
                        execute_final_hyper(gpu, &op).map_err(de)
                    } else {
                        execute_hyper_read(gpu, &op).map_err(de)
                    }
                };
                if single {
                    for r in 0..rows {
                        call(gpu, r, 1)?;
                    }
                } else {
                    with_lane_scope(gpu, rows, |gpu| call(gpu, 0, rows))?;
                }
                let got = [
                    down_f32(gpu, &nt)?,
                    down_f32(gpu, &lowt)?,
                    down_f32(gpu, &mt)?,
                ];
                free_all(gpu, vec![xt, nt, lowt, uot, mt, rot])?;
                Ok(got)
            };
            let single = run(gpu, true)?;
            let lane = run(gpu, false)?;
            for (i, name) in ["normalized", "low", "mixed"].iter().enumerate() {
                out.push(exact_f32(
                    &format!(
                        "lane-wide {} {name} rows={rows} == singles",
                        if final_read { "final read" } else { "read" }
                    ),
                    &lane[i],
                    &single[i],
                ));
            }
        }
    }
    free_all(gpu, vec![wnt, downt, upt])?;
    all(out)
}
