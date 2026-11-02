// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 Tier C cases: Gated DeltaNet (GDN) linear-attention kernels at the
//! Qwen4 geometry (16 key heads, 48 value heads, 128 x 128, conv kernel 4,
//! qkv width 2 * 2048 + 6144 = 10,240).
//!
//! Route of record on gfx942 (census: `local://census-gdn.md`):
//! `execute_gated_delta_net_hc_impl` (`hipfire-dispatch/src/pipeline/layer_ops.rs`)
//! takes `persistent_batch` (`gdn_persistent_batch`: `has_qwen4_parallel_qsa_gdn()
//! && (route_rows(rows) > 1 || is_gfx942()) && 128x128 heads && conv kernel 4`,
//! `has_qwen4_parallel_qsa_gdn() = has_gfx11_plus_simt() || is_gfx942()`). On
//! gfx942 every GDN step takes it, decode's one row included, so the per-row
//! chain `gated_delta_conv_params` -> `gated_delta_step_gated` (unfused:
//! `gated_delta_step` + `gated_delta_gate`; cases 1-4) is the reference the
//! batched route must equal, no longer a gfx942 route. Prefill and decode run
//! `gated_delta_conv_params_batched` -> `gated_delta_step_batched` (the F32
//! persistent256 recurrence) -> `gated_delta_gate_batched`: cases 5-7. Only
//! that F32 persistent route is covered here: native gfx942 MTP verify / row
//! capture is not claimed (its gate is the original gfx11+ one) and the
//! Q8-state persistent kernels are not covered by these cases. The WMMA chunk
//! route is gfx11 only (`has_wmma_w32`), so gfx942 never takes it. On gfx942
//! the F32 persistent256 body pins the sequential kernels' expression order
//! with explicit `__fmul_rn` / `__fmaf_rn` chains (hipcc otherwise packs
//! independent multiplies), so a batched F32 launch must equal `rows`
//! sequential single-row launches bit for bit.
//!
//! CPU twin: the kernels' math transcribed from `kernels/src/tensor_ops.hip`
//! (cited per function). `hipfire-arch-qwen4/src/ops.rs:391 gdn_step` and
//! `reference_forward.rs:575 gdn_layer` are the F32 source-model reference:
//! they l2-normalize q/k in plain F32, whereas the kernels reproduce the source
//! framework's BF16 l2norm (`tensor_ops.hip:5-8,78-81`), so they are not the
//! element-wise twin of these kernels. fp-contract: the `tensor_ops` module is
//! built without `-ffp-contract=off` (`compiler.rs:690-720`), so the twin uses
//! `mul_add` where the kernel's `acc += a * b` contracts to an FMA.

use super::common::*;
use super::Q4Case;
use rdna_compute::tensor_ops::{
    gated_delta_conv, gated_delta_conv_params, gated_delta_conv_params_batched, gated_delta_gate,
    gated_delta_gate_batched, gated_delta_params, gated_delta_step, gated_delta_step_batched,
    gated_delta_step_gated, GatedDeltaConv, GatedDeltaConvBatched, GatedDeltaGate,
    GatedDeltaGateBatched, GatedDeltaParams, GatedDeltaParamsBatched, GatedDeltaStep,
    GatedDeltaStepBatched, GdnStateFormat,
};
use rdna_compute::{DType, Gpu, GpuTensor};

pub const CASES: &[Q4Case] = &[
    Q4Case {
        name: "qwen4_gated_delta_conv_params_bf16_f32",
        timeout_s: 120,
        run: conv_params,
    },
    Q4Case {
        name: "qwen4_gated_delta_gate_bf16_f32",
        timeout_s: 120,
        run: gate_decode,
    },
    Q4Case {
        name: "qwen4_gated_delta_step_f32_gated",
        timeout_s: 300,
        run: step_f32_gated,
    },
    Q4Case {
        name: "qwen4_gated_delta_step_norm128_q8_gated",
        timeout_s: 300,
        run: step_q8_gated,
    },
    Q4Case {
        name: "qwen4_gated_delta_conv_params_bf16_f32_batched_k4",
        timeout_s: 120,
        run: conv_params_batched,
    },
    Q4Case {
        name: "qwen4_gated_delta_step_halves_state128_persistent256_f32",
        timeout_s: 300,
        run: step_batched,
    },
    Q4Case {
        name: "qwen4_gated_delta_gate_bf16_f32_batched",
        timeout_s: 120,
        run: gate_batched,
    },
];

// ── Geometry ──────────────────────────────────────────────────────────────────
const KH: usize = GDN_KEY_HEADS; // 16
const VH: usize = GDN_VALUE_HEADS; // 48
const HD: usize = GDN_HEAD_DIM; // 128
const REPEAT: usize = VH / KH; // 3
const QK: usize = KH * HD; // 2048
const VW: usize = VH * HD; // 6144
const QKV: usize = 2 * QK + VW; // 10_240
const HISTORY: usize = GDN_CONV_KERNEL - 1; // 3
const STATE_F32: usize = VH * HD * HD;
/// Absolute position of row 0: start_cursor = 5 % 3 = 2 exercises the ring wrap.
const START_POSITION: usize = 5;
/// Q8 GDN state slot (`tensor_ops.hip:31-51`): `VH * 128` rows of 128 i8 codes,
/// then `VH * 128` F32 scales.
const Q8_CODES: usize = VH * HD * HD;
const Q8_ROWS: usize = VH * HD;
const Q8_BYTES: usize = Q8_CODES + Q8_ROWS * 4;

fn bf(x: f32) -> f32 {
    bf16_round(x)
}

fn view(t: &GpuTensor, row: usize, width: usize) -> GpuTensor {
    t.sub_offset(row * width, width)
}

// ═══════════════════════════ CPU twins ════════════════════════════════════════

/// `gated_delta_conv_channel` (`tensor_ops.hip:2088-2127`), one row, all
/// channels, ring in place. `kernel` holds the BF16-exact weights `[c * 4 + tap]`.
fn conv_row_cpu(input: &[f32], kernel: &[f32], ring: &mut [f32], cursor: usize, out: &mut [f32]) {
    for c in 0..QKV {
        let current = bf(input[c]);
        let mut value = kernel[c * GDN_CONV_KERNEL + HISTORY] * current;
        for tap in 0..HISTORY {
            let h = bf(ring[((cursor + tap) % HISTORY) * QKV + c]);
            value = kernel[c * GDN_CONV_KERNEL + tap].mul_add(h, value);
        }
        out[c] = bf(value * (1.0 / (1.0 + (-value).exp())));
        for row in 0..HISTORY {
            ring[row * QKV + c] = bf(ring[row * QKV + c]);
        }
        ring[cursor * QKV + c] = current;
    }
}

/// `gated_delta_params_head` (`tensor_ops.hip:2584-2598`).
fn params_cpu(
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    gate: &mut [f32],
    beta: &mut [f32],
) {
    for flat in 0..a.len() {
        let head = flat % VH;
        let value = bf(a[flat]) + dt_bias[head];
        let softplus = if value > 20.0 {
            value
        } else if value < -20.0 {
            value.exp()
        } else {
            value.exp().ln_1p()
        };
        gate[flat] = -a_log[head].exp() * softplus;
        beta[flat] = 1.0 / (1.0 + (-bf(b[flat])).exp());
    }
}

/// `gated_delta_gate_channel` (`tensor_ops.hip:2667-2688`) for every head of
/// one row (`rec`, `z`, `out` are `VW` wide, `norm` is `HD` BF16-exact).
fn gate_cpu(rec: &[f32], z: &[f32], norm: &[f32], out: &mut [f32]) {
    for h in 0..VH {
        let s = h * HD;
        let mut sum = 0.0f32;
        for i in 0..HD {
            let v = bf(rec[s + i]);
            sum = v.mul_add(v, sum);
        }
        let inv = 1.0 / (sum / HD as f32 + 1.0e-6).sqrt();
        for c in 0..HD {
            let normalized = bf(bf(rec[s + c]) * inv * norm[c]);
            let g = bf(z[s + c]);
            out[s + c] = bf(normalized / (1.0 + (-g).exp()));
        }
    }
}

/// BF16 l2norm factor of one key head's q or k (`tensor_ops.hip:101-117`):
/// serial kd-order sum of BF16-rounded squares, BF16 sum, BF16 `rsqrt(sum + eps)`.
fn bf16_l2_inv(values: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in values {
        let b = bf(x);
        sum += bf(b * b);
    }
    bf(1.0 / (bf(sum) + bf(1.0e-6)).sqrt())
}

/// Normalized q (already times `query_scale`) and k of key head `kh`
/// (`tensor_ops.hip:126-149`).
fn qk_norms(q: &[f32], k: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let q_inv = bf16_l2_inv(q);
    let k_inv = bf16_l2_inv(k);
    let scale = (HD as f32).sqrt().recip();
    let qn = q.iter().map(|&x| bf(bf(x) * q_inv) * scale).collect();
    let kn = k.iter().map(|&x| bf(bf(x) * k_inv)).collect();
    (qn, kn)
}

struct RowIn<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    gate: &'a [f32],
    beta: &'a [f32],
}

/// `gated_delta_step_f32` (`tensor_ops.hip:82-152`) for one row: F32 state
/// `[head][key][value]`.
fn step_f32_cpu(state: &mut [f32], r: &RowIn, out: &mut [f32]) {
    for h in 0..VH {
        let kh = h / REPEAT;
        let (qn, kn) = qk_norms(&r.q[kh * HD..(kh + 1) * HD], &r.k[kh * HD..(kh + 1) * HD]);
        let decay = r.gate[h].exp();
        let beta = r.beta[h];
        let base = h * HD * HD;
        for i in 0..HD * HD {
            state[base + i] *= decay;
        }
        for vi in 0..HD {
            let mut kv_mem = 0.0f32;
            for kd in 0..HD {
                kv_mem = state[base + kd * HD + vi].mul_add(kn[kd], kv_mem);
            }
            let delta = (bf(r.v[h * HD + vi]) - kv_mem) * beta;
            for kd in 0..HD {
                let i = base + kd * HD + vi;
                state[i] = kn[kd].mul_add(delta, state[i]);
            }
            let mut o = 0.0f32;
            for kd in 0..HD {
                o = state[base + kd * HD + vi].mul_add(qn[kd], o);
            }
            out[h * HD + vi] = o;
        }
    }
}

/// `gdn_q8_round` (`tensor_ops.hip:53-60`), split into the pre-floor value
/// `fma(value, inv, noise)` and the clamp/floor so mismatches can report how
/// close the twin sat to a code boundary.
fn q8_prefloor(value: f32, inv: f32, head: usize, row: usize, column: usize, frame: u32) -> f32 {
    let mut rng = (head as u32)
        .wrapping_mul(1664525)
        .wrapping_add((row as u32).wrapping_mul(1013904223))
        .wrapping_add((column as u32).wrapping_mul(214013))
        .wrapping_add(frame.wrapping_mul(2531011));
    rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
    let noise = (rng >> 8) as f32 * (1.0 / 16777216.0);
    value.mul_add(inv, noise)
}
fn q8_code(prefloor: f32) -> i8 {
    prefloor.floor().max(-128.0).min(127.0) as i8
}

/// `gated_delta_step_norm128_body<true>` (`tensor_ops.hip:254-306`): the F32
/// recurrence on the dequantized rows, each row requantized by its absmax.
fn step_q8_cpu(
    codes: &mut [i8],
    scales: &mut [f32],
    r: &RowIn,
    frame: u32,
    out: &mut [f32],
    prefloor: &mut [f32],
) {
    for h in 0..VH {
        let kh = h / REPEAT;
        let (qn, kn) = qk_norms(&r.q[kh * HD..(kh + 1) * HD], &r.k[kh * HD..(kh + 1) * HD]);
        let decay = r.gate[h].exp();
        let beta = r.beta[h];
        for vi in 0..HD {
            let row = h * HD + vi;
            let scale = scales[row];
            let code = |kd: usize| (scale * codes[row * HD + kd] as f32) * decay;
            let mut kv_mem = 0.0f32;
            for kd in 0..HD {
                kv_mem = code(kd).mul_add(kn[kd], kv_mem);
            }
            let delta = (bf(r.v[h * HD + vi]) - kv_mem) * beta;
            let mut decayed = [0.0f32; HD];
            let mut o = 0.0f32;
            let mut peak = 0.0f32;
            for kd in 0..HD {
                decayed[kd] = kn[kd].mul_add(delta, code(kd));
                peak = peak.max(decayed[kd].abs());
                o = decayed[kd].mul_add(qn[kd], o);
            }
            let inv = if peak > 0.0 { 127.0 / peak } else { 0.0 };
            for kd in 0..HD {
                let p = q8_prefloor(decayed[kd], inv, h, vi, kd, frame);
                prefloor[row * HD + kd] = p;
                codes[row * HD + kd] = q8_code(p);
            }
            scales[row] = if peak > 0.0 { peak / 127.0 } else { 1.0 };
            out[h * HD + vi] = o;
        }
    }
}

fn split_q8(bytes: &[u8]) -> (Vec<i8>, Vec<f32>) {
    let codes = bytes[..Q8_CODES].iter().map(|&b| b as i8).collect();
    let scales = bytes[Q8_CODES..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (codes, scales)
}

fn join_q8(codes: &[i8], scales: &[f32]) -> Vec<u8> {
    let mut bytes: Vec<u8> = codes.iter().map(|&c| c as u8).collect();
    for s in scales {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    bytes
}

// ═══════════════════════════ input generators ═════════════════════════════════

/// Per-row recurrence inputs: `proj` is `[rows, QKV]` laid out q | k | v per
/// row (the convolution output the route's step reads), `gate` the log decay
/// and `beta` the update rate, both `[rows, VH]`. `zero_even_gates` makes
/// every even head's decay exactly `expf(0) = 1` (so the Q8 requantization of
/// those heads is bit-reproducible on the host).
struct Steps {
    proj: Vec<f32>,
    gate: Vec<f32>,
    beta: Vec<f32>,
}

fn make_steps(rows: usize, seed: u64, zero_even_gates: bool) -> Steps {
    let mut rng = Rng::new(seed);
    let mut proj = Vec::with_capacity(rows * QKV);
    for _ in 0..rows {
        proj.extend(rng.vec(2 * QK, -2.0, 2.0));
        proj.extend(rng.vec(VW, -1.0, 1.0));
    }
    let mut gate = Vec::with_capacity(rows * VH);
    for _ in 0..rows {
        for h in 0..VH {
            let g = rng.range(-0.3, -0.02);
            gate.push(if zero_even_gates && h % 2 == 0 {
                0.0
            } else {
                g
            });
        }
    }
    let beta = rng.vec(rows * VH, 0.1, 0.9);
    Steps { proj, gate, beta }
}

impl Steps {
    fn row(&self, r: usize) -> RowIn<'_> {
        let p = &self.proj[r * QKV..(r + 1) * QKV];
        RowIn {
            q: &p[..QK],
            k: &p[QK..2 * QK],
            v: &p[2 * QK..],
            gate: &self.gate[r * VH..(r + 1) * VH],
            beta: &self.beta[r * VH..(r + 1) * VH],
        }
    }
}

fn gdn_step_args<'a>(
    q: &'a GpuTensor,
    k: &'a GpuTensor,
    v: &'a GpuTensor,
    gate: &'a GpuTensor,
    beta: &'a GpuTensor,
    state: &'a GpuTensor,
    format: GdnStateFormat,
    output: &'a GpuTensor,
    position: usize,
) -> GatedDeltaStep<'a> {
    GatedDeltaStep {
        q,
        k,
        v,
        gate,
        beta,
        state,
        format,
        residual: None,
        output,
        key_heads: KH,
        value_heads: VH,
        key_dim: HD,
        value_dim: HD,
        position,
    }
}

// ═══════════════════ convolution + parameters (decode / per-row) ══════════════

struct ConvIn {
    input: Vec<f32>,
    kernel: Vec<f32>,
    history: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    a_log: Vec<f32>,
    dt: Vec<f32>,
}

fn make_conv_in(rows: usize, seed: u64) -> ConvIn {
    let mut rng = Rng::new(seed);
    ConvIn {
        input: rng.vec(rows * QKV, -3.0, 3.0),
        kernel: rng.vec_bf16_exact(QKV * GDN_CONV_KERNEL, -0.8, 0.8),
        history: rng.vec(HISTORY * QKV, -3.0, 3.0),
        a: rng.vec(rows * VH, -4.0, 4.0),
        b: rng.vec(rows * VH, -4.0, 4.0),
        a_log: rng.vec_bf16_exact(VH, -0.5, 1.5),
        dt: rng.vec_bf16_exact(VH, -1.0, 1.0),
    }
}

struct ConvOut {
    out: Vec<f32>,
    hist: Vec<f32>,
    gate: Vec<f32>,
    beta: Vec<f32>,
}

/// The gfx942 decode route's per-row launch (`layer_ops.rs`, `gated_delta_conv_params`
/// with `history == next_history`; also the sequential baseline the batched
/// kernels are compared with). `fused == false` runs the two standalone
/// kernels (`gated_delta_conv_bf16_f32`, `gated_delta_params_bf16_f32`) the
/// fused kernel claims to be bitwise equal to.
fn gpu_conv_params_rows(
    gpu: &mut Gpu,
    rows: usize,
    c: &ConvIn,
    fused: bool,
) -> Result<ConvOut, String> {
    let input = up_f32(gpu, &c.input)?;
    let kernel = up_bf16(gpu, &c.kernel)?;
    let hist = up_f32(gpu, &c.history)?;
    let out = sentinel_f32(gpu, rows * QKV)?;
    let a = up_f32(gpu, &c.a)?;
    let b = up_f32(gpu, &c.b)?;
    let a_log = up_bf16(gpu, &c.a_log)?;
    let dt = up_bf16(gpu, &c.dt)?;
    let gate = sentinel_f32(gpu, rows * VH)?;
    let beta = sentinel_f32(gpu, rows * VH)?;
    for row in 0..rows {
        let cursor = (START_POSITION + row) % HISTORY;
        let in_row = view(&input, row, QKV);
        let out_row = view(&out, row, QKV);
        let params = GatedDeltaParams {
            a: &view(&a, row, VH),
            b: &view(&b, row, VH),
            a_log: &a_log,
            dt_bias: &dt,
            gate: &view(&gate, row, VH),
            beta: &view(&beta, row, VH),
        };
        let conv = GatedDeltaConv {
            input: &in_row,
            kernel: &kernel,
            history: &hist,
            output: &out_row,
            next_history: &hist,
            channels: QKV,
            history_rows: HISTORY,
            kernel_size: GDN_CONV_KERNEL,
            cursor,
            row_index: row,
        };
        if fused {
            gated_delta_conv_params(gpu, &conv, &params, VH).map_err(e)?;
        } else {
            gated_delta_conv(gpu, &conv).map_err(e)?;
            gated_delta_params(gpu, &params, VH).map_err(e)?;
        }
    }
    let result = ConvOut {
        out: down_f32(gpu, &out)?,
        hist: down_f32(gpu, &hist)?,
        gate: down_f32(gpu, &gate)?,
        beta: down_f32(gpu, &beta)?,
    };
    free_all(
        gpu,
        vec![input, kernel, hist, out, a, b, a_log, dt, gate, beta],
    )?;
    Ok(result)
}

/// Kernels: `gated_delta_conv_params_bf16_f32` (the decode route's per-row
/// launch: one row outside the prefill scope) plus, cross-checked bitwise, `gated_delta_conv_bf16_f32`
/// and `gated_delta_params_bf16_f32`. `ROW_COUNTS` sequential rows carry the
/// conv ring state in and out (start cursor 2, so the ring wraps), aliasing
/// `history` and `next_history` exactly like the route.
fn conv_params(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for &rows in &ROW_COUNTS {
        let c = make_conv_in(rows, 0x6764_6e01 + rows as u64);
        let mut ring = c.history.clone();
        let mut want_out = vec![0.0f32; rows * QKV];
        for row in 0..rows {
            conv_row_cpu(
                &c.input[row * QKV..(row + 1) * QKV],
                &c.kernel,
                &mut ring,
                (START_POSITION + row) % HISTORY,
                &mut want_out[row * QKV..(row + 1) * QKV],
            );
        }
        let mut want_gate = vec![0.0f32; rows * VH];
        let mut want_beta = vec![0.0f32; rows * VH];
        params_cpu(&c.a, &c.b, &c.a_log, &c.dt, &mut want_gate, &mut want_beta);

        let fused = gpu_conv_params_rows(gpu, rows, &c, true)?;
        let split = gpu_conv_params_rows(gpu, rows, &c, false)?;
        // Conv outputs are BF16-rounded SiLU values; the exp differs by an ulp
        // between ocml `expf` and libm, which can flip one BF16 rounding.
        checks.push(close_f32(
            &format!("rows={rows} conv out"),
            &fused.out,
            &want_out,
            BF16_INPUT_F32_ACCUM,
        ));
        checks.push(exact_f32(
            &format!("rows={rows} conv ring out"),
            &fused.hist,
            &ring,
        ));
        checks.push(close_f32(
            &format!("rows={rows} gate"),
            &fused.gate,
            &want_gate,
            F32_ACCUM,
        ));
        checks.push(close_f32(
            &format!("rows={rows} beta"),
            &fused.beta,
            &want_beta,
            F32_ACCUM,
        ));
        checks.push(exact_f32(
            &format!("rows={rows} fused==split conv"),
            &fused.out,
            &split.out,
        ));
        checks.push(exact_f32(
            &format!("rows={rows} fused==split ring"),
            &fused.hist,
            &split.hist,
        ));
        checks.push(exact_f32(
            &format!("rows={rows} fused==split gate"),
            &fused.gate,
            &split.gate,
        ));
        checks.push(exact_f32(
            &format!("rows={rows} fused==split beta"),
            &fused.beta,
            &split.beta,
        ));
    }
    all(checks)
}

// ═══════════════════════════ gated RMSNorm (decode) ═══════════════════════════

fn gate_inputs(rows: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    (
        rng.vec(rows * VW, -3.0, 3.0),
        rng.vec(rows * VW, -6.0, 6.0),
        rng.vec_bf16_exact(HD, 0.5, 1.5),
    )
}

/// `gated_delta_gate_bf16_f32` (the route's per-row gate, `tensor_ops.hip:2690`):
/// 48 heads x 128, one launch per row.
fn gate_decode(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for &rows in &ROW_COUNTS {
        let (rec, z, norm) = gate_inputs(rows, 0x6764_6e02 + rows as u64);
        let mut want = vec![0.0f32; rows * VW];
        for row in 0..rows {
            gate_cpu(
                &rec[row * VW..(row + 1) * VW],
                &z[row * VW..(row + 1) * VW],
                &norm,
                &mut want[row * VW..(row + 1) * VW],
            );
        }
        let rec_t = up_f32(gpu, &rec)?;
        let z_t = up_f32(gpu, &z)?;
        let norm_t = up_bf16(gpu, &norm)?;
        let out_t = sentinel_f32(gpu, rows * VW)?;
        for row in 0..rows {
            let g = GatedDeltaGate {
                recurrent_output: &view(&rec_t, row, VW),
                z: &view(&z_t, row, VW),
                norm: &norm_t,
                output: &view(&out_t, row, VW),
                value_heads: VH,
                value_dim: HD,
            };
            gated_delta_gate(gpu, &g).map_err(e)?;
        }
        let got = down_f32(gpu, &out_t)?;
        // The output boundary is BF16: every stored F32 is BF16-exact.
        let rounded: Vec<f32> = got.iter().map(|&x| bf(x)).collect();
        checks.push(close_f32(
            &format!("rows={rows} gate out"),
            &got,
            &want,
            BF16_INPUT_F32_ACCUM,
        ));
        checks.push(exact_f32(
            &format!("rows={rows} gate out is BF16-exact"),
            &got,
            &rounded,
        ));
        free_all(gpu, vec![rec_t, z_t, norm_t, out_t])?;
    }
    all(checks)
}

// ═══════════════════ decode step + gate (route: gated_delta_step_gated) ═══════

/// Decode route launch (one row outside the prefill scope; the single-row
/// `gated_delta_step_gated` keeps its gfx11+ fused gate, so it stays unfused on
/// gfx942) `gated_delta_step_gated` on an F32 state. gfx942 branch
/// (`has_gfx11_plus_simt()` false): `gated_delta_step_f32` (block 256, grid
/// [48, 1]) then `gated_delta_gate_bf16_f32`; the gfx11+ fused kernel
/// (`..._gate_norm128_gfx1151`, atomics) is not taken, and when it is (other
/// archs) the recurrent output is not stored, so that comparison is skipped.
/// 33 sequential rows carry the state (rows 1, 7 and 33 are prefixes of the
/// sequence; the first offending row is `worst / 6144` in the summary).
fn step_f32_gated(gpu: &mut Gpu) -> Result<String, String> {
    let rows = *ROW_COUNTS.last().unwrap();
    let fused_arch = gpu.arch_caps.has_gfx11_plus_simt();
    let steps = make_steps(rows, 0x6764_6e03, false);
    let mut rng = Rng::new(0x6764_6e13);
    let state0 = rng.vec(STATE_F32, -0.5, 0.5);
    let z = rng.vec(rows * VW, -6.0, 6.0);
    let norm = rng.vec_bf16_exact(HD, 0.5, 1.5);

    let proj = up_f32(gpu, &steps.proj)?;
    let gate = up_f32(gpu, &steps.gate)?;
    let beta = up_f32(gpu, &steps.beta)?;
    let state = up_f32(gpu, &state0)?;
    let z_t = up_f32(gpu, &z)?;
    let norm_t = up_bf16(gpu, &norm)?;
    let rec = sentinel_f32(gpu, rows * VW)?;
    let out = sentinel_f32(gpu, rows * VW)?;
    for row in 0..rows {
        let (q, k, v) = (
            proj.sub_offset(row * QKV, QK),
            proj.sub_offset(row * QKV + QK, QK),
            proj.sub_offset(row * QKV + 2 * QK, VW),
        );
        let (g_row, b_row) = (view(&gate, row, VH), view(&beta, row, VH));
        let rec_row = view(&rec, row, VW);
        let step = gdn_step_args(
            &q,
            &k,
            &v,
            &g_row,
            &b_row,
            &state,
            GdnStateFormat::F32,
            &rec_row,
            START_POSITION + row,
        );
        let gate_args = GatedDeltaGate {
            recurrent_output: &rec_row,
            z: &view(&z_t, row, VW),
            norm: &norm_t,
            output: &view(&out, row, VW),
            value_heads: VH,
            value_dim: HD,
        };
        gated_delta_step_gated(gpu, &step, &gate_args, None).map_err(e)?;
    }
    let got_rec = down_f32(gpu, &rec)?;
    let got_out = down_f32(gpu, &out)?;
    let got_state = down_f32(gpu, &state)?;

    let mut cpu_state = state0.clone();
    let mut want_rec = vec![0.0f32; rows * VW];
    for row in 0..rows {
        step_f32_cpu(
            &mut cpu_state,
            &steps.row(row),
            &mut want_rec[row * VW..(row + 1) * VW],
        );
    }
    // Isolate the gate from recurrence rounding: feed it the recurrent output
    // the kernel actually stored (or the twin's where the fused kernel keeps it in LDS).
    let gate_in = if fused_arch { &want_rec } else { &got_rec };
    let mut want_out = vec![0.0f32; rows * VW];
    for row in 0..rows {
        gate_cpu(
            &gate_in[row * VW..(row + 1) * VW],
            &z[row * VW..(row + 1) * VW],
            &norm,
            &mut want_out[row * VW..(row + 1) * VW],
        );
    }
    let mut checks = Vec::new();
    if !fused_arch {
        checks.push(close_f32(
            "recurrent out (F32)",
            &got_rec,
            &want_rec,
            F32_ACCUM,
        ));
    }
    checks.push(close_f32(
        "final state (F32)",
        &got_state,
        &cpu_state,
        F32_ACCUM,
    ));
    checks.push(close_f32(
        "gated out (BF16 boundary)",
        &got_out,
        &want_out,
        BF16_INPUT_F32_ACCUM,
    ));
    free_all(gpu, vec![proj, gate, beta, state, z_t, norm_t, rec, out])?;
    all(checks)
}

/// Same route launch on a Q8 GDN state (`gated_delta_step_norm128_q8`, block
/// 128): the default state of a Qwen4 load (`state_quant` auto -> Q8,
/// `state.rs:591`). Each step is verified from the GPU's own pre-step state
/// (so stochastic-rounding code flips cannot drift the twin). Even heads run
/// with decay exactly 1 (`expf(0)`): their requantized codes and scales must
/// match bit for bit (this pins the noise hash, absmax and slot layout);
/// odd heads (decay differs by an ulp between `expf` implementations, which
/// may move a code by one step <= peak/127 < atol) compare the dequantized
/// state under the BF16-input tolerance and the scales under F32_ACCUM.
fn step_q8_gated(gpu: &mut Gpu) -> Result<String, String> {
    let rows = *ROW_COUNTS.last().unwrap();
    let fused_arch = gpu.arch_caps.has_gfx11_plus_simt();
    let steps = make_steps(rows, 0x6764_6e04, true);
    let mut rng = Rng::new(0x6764_6e14);
    let codes0: Vec<i8> = (0..Q8_CODES)
        .map(|_| (rng.below(255) as i32 - 127) as i8)
        .collect();
    let scales0: Vec<f32> = (0..Q8_ROWS).map(|_| rng.range(0.002, 0.01)).collect();
    let z = rng.vec(rows * VW, -6.0, 6.0);
    let norm = rng.vec_bf16_exact(HD, 0.5, 1.5);

    let proj = up_f32(gpu, &steps.proj)?;
    let gate = up_f32(gpu, &steps.gate)?;
    let beta = up_f32(gpu, &steps.beta)?;
    debug_assert_eq!(GdnStateFormat::Q8.state_units(VH, HD, HD), Q8_BYTES);
    let state = gpu
        .upload_raw(&join_q8(&codes0, &scales0), &[Q8_BYTES])
        .map_err(e)?;
    if state.dtype != DType::Raw || state.byte_size() != Q8_BYTES {
        return Err(format!(
            "Q8 state upload: dtype {:?} bytes {}",
            state.dtype,
            state.byte_size()
        ));
    }
    let z_t = up_f32(gpu, &z)?;
    let norm_t = up_bf16(gpu, &norm)?;
    let rec = sentinel_f32(gpu, rows * VW)?;
    let out = sentinel_f32(gpu, rows * VW)?;
    let mut last = String::new();
    let mut row_results = Vec::new();
    let mut prefloor = vec![0.0f32; Q8_CODES];
    for row in 0..rows {
        let pre = down_bytes(gpu, &state)?;
        let (mut codes, mut scales) = split_q8(&pre);
        let mut want_rec = vec![0.0f32; VW];
        let frame = (START_POSITION + row) as u32;
        step_q8_cpu(
            &mut codes,
            &mut scales,
            &steps.row(row),
            frame,
            &mut want_rec,
            &mut prefloor,
        );

        let (q, k, v) = (
            proj.sub_offset(row * QKV, QK),
            proj.sub_offset(row * QKV + QK, QK),
            proj.sub_offset(row * QKV + 2 * QK, VW),
        );
        let (g_row, b_row) = (view(&gate, row, VH), view(&beta, row, VH));
        let rec_row = view(&rec, row, VW);
        let out_row = view(&out, row, VW);
        let step = gdn_step_args(
            &q,
            &k,
            &v,
            &g_row,
            &b_row,
            &state,
            GdnStateFormat::Q8,
            &rec_row,
            START_POSITION + row,
        );
        let gate_args = GatedDeltaGate {
            recurrent_output: &rec_row,
            z: &view(&z_t, row, VW),
            norm: &norm_t,
            output: &out_row,
            value_heads: VH,
            value_dim: HD,
        };
        gated_delta_step_gated(gpu, &step, &gate_args, None).map_err(e)?;
        let got_rec = down_f32(gpu, &rec_row)?;
        let got_out = down_f32(gpu, &out_row)?;
        let (got_codes, got_scales) = split_q8(&down_bytes(gpu, &state)?);

        let mut checks = Vec::new();
        if !fused_arch {
            checks.push(close_f32(
                &format!("row {row} recurrent out"),
                &got_rec,
                &want_rec,
                F32_ACCUM,
            ));
        }
        let gate_in = if fused_arch { &want_rec } else { &got_rec };
        let mut want_out = vec![0.0f32; VW];
        gate_cpu(gate_in, &z[row * VW..(row + 1) * VW], &norm, &mut want_out);
        checks.push(close_f32(
            &format!("row {row} gated out"),
            &got_out,
            &want_out,
            BF16_INPUT_F32_ACCUM,
        ));
        // State: exact on the decay-1 heads, tolerance on the others.
        let (mut ec_got, mut ec_want, mut es_got, mut es_want) = (vec![], vec![], vec![], vec![]);
        let (mut dq_got, mut dq_want, mut sc_got, mut sc_want) = (vec![], vec![], vec![], vec![]);
        for h in 0..VH {
            for vi in 0..HD {
                let r = h * HD + vi;
                let span = r * HD..(r + 1) * HD;
                if h % 2 == 0 {
                    ec_got.extend_from_slice(&got_codes[span.clone()]);
                    ec_want.extend_from_slice(&codes[span]);
                    es_got.push(got_scales[r]);
                    es_want.push(scales[r]);
                } else {
                    dq_got.extend(
                        got_codes[span.clone()]
                            .iter()
                            .map(|&c| got_scales[r] * c as f32),
                    );
                    dq_want.extend(codes[span].iter().map(|&c| scales[r] * c as f32));
                    sc_got.push(got_scales[r]);
                    sc_want.push(scales[r]);
                }
            }
        }
        let codes_check = exact_ints(
            &format!("row {row} decay-1 heads Q8 codes"),
            &ec_got,
            &ec_want,
        );
        // Classify code mismatches: report the twin's pre-floor value
        // fma(decayed, inv, noise) and its distance to the floor boundary, and
        // whether the row scale (absmax) matched. A ±1 code with the twin
        // within a few F32 ulps of an integer and an exact scale is a last-ulp
        // F32 difference in `decayed`/`delta` (operation order / contraction),
        // not a lane-mapping defect.
        let codes_check = codes_check.map_err(|msg| {
            let mut notes = Vec::new();
            for h in (0..VH).step_by(2) {
                for vi in 0..HD {
                    let r = h * HD + vi;
                    for kd in 0..HD {
                        let i = r * HD + kd;
                        if got_codes[i] != codes[i] && notes.len() < 6 {
                            let p = prefloor[i];
                            notes.push(format!(
                                "[head {h} row {vi} key {kd}: got {} want {} twin_prefloor {p:.9} dist_to_boundary {:.3e} scale_got {:e} scale_want {:e}{}]",
                                got_codes[i],
                                codes[i],
                                p - p.floor(),
                                got_scales[r],
                                scales[r],
                                if got_scales[r].to_bits() == scales[r].to_bits() { " (scale exact)" } else { " (scale differs)" },
                            ));
                        }
                    }
                }
            }
            format!("{msg}; {}", notes.join(" "))
        });
        checks.push(codes_check);
        checks.push(exact_f32(
            &format!("row {row} decay-1 heads Q8 scales"),
            &es_got,
            &es_want,
        ));
        checks.push(close_f32(
            &format!("row {row} decaying heads dequantized state"),
            &dq_got,
            &dq_want,
            BF16_INPUT_F32_ACCUM,
        ));
        checks.push(close_f32(
            &format!("row {row} decaying heads scales"),
            &sc_got,
            &sc_want,
            F32_ACCUM,
        ));
        match all(checks) {
            Ok(msg) => last = msg,
            Err(msg) => row_results.push(Err(msg)),
        }
    }
    free_all(gpu, vec![proj, gate, beta, state, z_t, norm_t, rec, out])?;
    if !row_results.is_empty() {
        return all(row_results).map(|_| String::new());
    }
    Ok(format!("{rows} sequential Q8 steps ok; last: {last}"))
}

// ══════════ batched / persistent kernels (the gfx942 AR prefill route) ═════════════

/// `gated_delta_conv_bf16_f32_batched_k4` + the parameter blocks
/// (`gated_delta_conv_params_batched`, F32 in/out) against the per-row launch
/// chain: the kernel documents itself as the per-row ring unrolled
/// (`tensor_ops.hip:2144-2150`), so every output and the ring are bit-exact.
fn conv_params_batched(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for &rows in &ROW_COUNTS {
        let c = make_conv_in(rows, 0x6764_6e05 + rows as u64);
        let want = gpu_conv_params_rows(gpu, rows, &c, true)?;

        let input = up_f32(gpu, &c.input)?;
        let kernel = up_bf16(gpu, &c.kernel)?;
        let hist = up_f32(gpu, &c.history)?;
        let out = sentinel_f32(gpu, rows * QKV)?;
        let a = up_f32(gpu, &c.a)?;
        let b = up_f32(gpu, &c.b)?;
        let a_log = up_bf16(gpu, &c.a_log)?;
        let dt = up_bf16(gpu, &c.dt)?;
        let gate = sentinel_f32(gpu, rows * VH)?;
        let beta = sentinel_f32(gpu, rows * VH)?;
        let conv = GatedDeltaConvBatched {
            input: &input,
            kernel: &kernel,
            history: &hist,
            output: &out,
            next_history: &hist,
            rows,
            channels: QKV,
            history_rows: HISTORY,
            kernel_size: GDN_CONV_KERNEL,
            start_cursor: START_POSITION % HISTORY,
        };
        let params = GatedDeltaParamsBatched {
            a: &a,
            b: &b,
            a_log: &a_log,
            dt_bias: &dt,
            gate: &gate,
            beta: &beta,
            rows,
            heads: VH,
        };
        gated_delta_conv_params_batched(gpu, &conv, &params).map_err(e)?;
        checks.push(exact_f32(
            &format!("rows={rows} conv out"),
            &down_f32(gpu, &out)?,
            &want.out,
        ));
        checks.push(exact_f32(
            &format!("rows={rows} ring"),
            &down_f32(gpu, &hist)?,
            &want.hist,
        ));
        checks.push(exact_f32(
            &format!("rows={rows} gate"),
            &down_f32(gpu, &gate)?,
            &want.gate,
        ));
        checks.push(exact_f32(
            &format!("rows={rows} beta"),
            &down_f32(gpu, &beta)?,
            &want.beta,
        ));
        free_all(
            gpu,
            vec![input, kernel, hist, out, a, b, a_log, dt, gate, beta],
        )?;
    }
    all(checks)
}

/// `gated_delta_step_halves_state128_persistent256_f32` through
/// `gated_delta_step_batched` (gfx942 takes `gated_delta_step_pipe128_f32` by
/// default, bytewise this kernel under the same pinned arithmetic;
/// `HIPFIRE_QWEN4_GDN_PIPE=0` keeps persistent256). The kernel documents
/// bitwise equality with `rows` sequential
/// `gated_delta_step_f32` launches (`tensor_ops.hip:408-409`): outputs and the
/// final state are compared exactly against that chain, at the ROW_COUNTS and
/// at 257 (one past the kernel's 256-row parameter block). The CPU twin
/// (F32_ACCUM) runs for the ROW_COUNTS.
fn step_batched(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for rows in [ROW_COUNTS[0], ROW_COUNTS[1], ROW_COUNTS[2], 257] {
        let steps = make_steps(rows, 0x6764_6e06 + rows as u64, false);
        let mut rng = Rng::new(0x6764_6e16 + rows as u64);
        let state0 = rng.vec(STATE_F32, -0.5, 0.5);
        let proj = up_f32(gpu, &steps.proj)?;
        let gate = up_f32(gpu, &steps.gate)?;
        let beta = up_f32(gpu, &steps.beta)?;

        let state_b = up_f32(gpu, &state0)?;
        let out_b = sentinel_f32(gpu, rows * VW)?;
        gated_delta_step_batched(
            gpu,
            &GatedDeltaStepBatched {
                projection: &proj,
                gate: &gate,
                beta: &beta,
                state: &state_b,
                format: GdnStateFormat::F32,
                residual: None,
                output: &out_b,
                row_states: None,
                rows,
                qkv_width: QKV,
                key_heads: KH,
                value_heads: VH,
                key_dim: HD,
                value_dim: HD,
                position: START_POSITION,
            },
        )
        .map_err(e)?;

        let state_s = up_f32(gpu, &state0)?;
        let out_s = sentinel_f32(gpu, rows * VW)?;
        for row in 0..rows {
            let (q, k, v) = (
                proj.sub_offset(row * QKV, QK),
                proj.sub_offset(row * QKV + QK, QK),
                proj.sub_offset(row * QKV + 2 * QK, VW),
            );
            let (g_row, b_row) = (view(&gate, row, VH), view(&beta, row, VH));
            let o_row = view(&out_s, row, VW);
            gated_delta_step(
                gpu,
                &gdn_step_args(
                    &q,
                    &k,
                    &v,
                    &g_row,
                    &b_row,
                    &state_s,
                    GdnStateFormat::F32,
                    &o_row,
                    START_POSITION + row,
                ),
            )
            .map_err(e)?;
        }
        let got_out = down_f32(gpu, &out_b)?;
        let got_state = down_f32(gpu, &state_b)?;
        checks.push(exact_f32(
            &format!("rows={rows} batched==sequential out"),
            &got_out,
            &down_f32(gpu, &out_s)?,
        ));
        checks.push(exact_f32(
            &format!("rows={rows} batched==sequential state"),
            &got_state,
            &down_f32(gpu, &state_s)?,
        ));
        if rows <= *ROW_COUNTS.last().unwrap() {
            let mut cpu_state = state0.clone();
            let mut want_out = vec![0.0f32; rows * VW];
            for row in 0..rows {
                step_f32_cpu(
                    &mut cpu_state,
                    &steps.row(row),
                    &mut want_out[row * VW..(row + 1) * VW],
                );
            }
            checks.push(close_f32(
                &format!("rows={rows} out vs CPU"),
                &got_out,
                &want_out,
                F32_ACCUM,
            ));
            checks.push(close_f32(
                &format!("rows={rows} state vs CPU"),
                &got_state,
                &cpu_state,
                F32_ACCUM,
            ));
        }
        free_all(gpu, vec![proj, gate, beta, state_b, out_b, state_s, out_s])?;
    }
    all(checks)
}

/// `gated_delta_gate_bf16_f32_batched` (block 32 = half a wave64; the kernel is
/// barrier-synchronized, no shuffles) against the per-row gate, bit-exact (same
/// serial 128-term sum, same expressions), plus the CPU twin. The batched
/// kernel also writes the BF16-rounded recurrent values back in place, so it
/// runs on its own copy.
fn gate_batched(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for &rows in &ROW_COUNTS {
        let (rec, z, norm) = gate_inputs(rows, 0x6764_6e07 + rows as u64);
        let mut want = vec![0.0f32; rows * VW];
        for row in 0..rows {
            gate_cpu(
                &rec[row * VW..(row + 1) * VW],
                &z[row * VW..(row + 1) * VW],
                &norm,
                &mut want[row * VW..(row + 1) * VW],
            );
        }
        let rec_row = up_f32(gpu, &rec)?;
        let rec_batch = up_f32(gpu, &rec)?;
        let z_t = up_f32(gpu, &z)?;
        let norm_t = up_bf16(gpu, &norm)?;
        let out_row = sentinel_f32(gpu, rows * VW)?;
        let out_batch = sentinel_f32(gpu, rows * VW)?;
        for row in 0..rows {
            let g = GatedDeltaGate {
                recurrent_output: &view(&rec_row, row, VW),
                z: &view(&z_t, row, VW),
                norm: &norm_t,
                output: &view(&out_row, row, VW),
                value_heads: VH,
                value_dim: HD,
            };
            gated_delta_gate(gpu, &g).map_err(e)?;
        }
        gated_delta_gate_batched(
            gpu,
            &GatedDeltaGateBatched {
                recurrent_output: &rec_batch,
                z: &z_t,
                norm: &norm_t,
                output: &out_batch,
                rows,
                value_heads: VH,
                value_dim: HD,
            },
        )
        .map_err(e)?;
        let got_batch = down_f32(gpu, &out_batch)?;
        checks.push(exact_f32(
            &format!("rows={rows} batched==per-row"),
            &got_batch,
            &down_f32(gpu, &out_row)?,
        ));
        checks.push(close_f32(
            &format!("rows={rows} batched vs CPU"),
            &got_batch,
            &want,
            BF16_INPUT_F32_ACCUM,
        ));
        free_all(
            gpu,
            vec![rec_row, rec_batch, z_t, norm_t, out_row, out_batch],
        )?;
    }
    all(checks)
}
