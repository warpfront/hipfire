// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tier C cases: the BF16 (qt=16) dense trunk projections on gfx942.
//!
//! Census of record: `census-trunk.md` (family `trunk`). Covered kernels:
//!
//! * `gemm_bf16_mfma_gfx942` (`kernels/src/gemm_bf16_mfma.gfx942.hip`) through
//!   its public launcher [`Gpu::gemm_bf16_mfma_gfx942`] — the BF16 trunk GEMM
//!   the gfx942 route reaches for every real row count (1..=16 rows take the
//!   module's few-row entry `gemm_bf16_mfma_gfx942_rows16`, more rows the
//!   32-row-tile entry). The activation is staged F32 → BF16 by
//!   [`Gpu::convert_f32_to_bf16`] exactly as the route stages it; the staged
//!   bits are checked against the frozen RNE rule here too (the standalone
//!   convert case belongs to `io.rs`). The few-row entry is bit-compared
//!   against the tile entry ([`Gpu::gemm_bf16_mfma_gfx942_tiled`]) at 1..=16
//!   rows, and its fused form ([`Gpu::gemm_bf16_mfma_gfx942_rows16`]: F32 x
//!   staged in-kernel, F32 or BF16 output) against staging + GEMM + rounding.
//! * `gemv_bf16_xf32` ([`Gpu::gemv_bf16_xf32`], `kernels/src/gemv_bf16_xf32.hip`
//!   on every arch but gfx1151): the rows = 1 BF16 GEMV of the census-base
//!   `project_rotated` and of the final-row LM head.
//! * `gemm_bf16_xf32_multirow` ([`Gpu::gemm_bf16_xf32_multirow`],
//!   `kernels/src/gemm_bf16_xf32_multirow.hip`): what `project_rotated` launches
//!   for rows >= 2 on gfx942 on the census base (the F16-WMMA route and the
//!   gfx11 few-row grouped GEMVs are refused on gfx942).
//! * `gemv_bf16_xf32_k4` ([`Gpu::gemv_bf16_xf32_k4`],
//!   `kernels/src/qwen4_gemv_bf16_xf32.hip`): the four-wave long-K GEMV. The
//!   census base only launches it under `has_gfx11_plus_simt()`; it is covered
//!   here to prove it is wave64-safe if a future route reaches it, and a launch
//!   refusal is reported, not failed.
//!
//! Shapes are real Qwen4 projections (hidden 2560): GDN qkv 10240x2560, GDN /
//! QSA out-proj 2560x6144, GDN in_proj_a/b 48x2560 (M not a multiple of the
//! 32x32 tile), HC block-inject gate 4x10240, HC down 320x10240, LM-head slice.

use super::common::*;
use super::Q4Case;
use rdna_compute::{DType, Gpu, GpuTensor};

pub const CASES: &[Q4Case] = &[
    Q4Case {
        name: "qwen4_gemm_bf16_mfma_gfx942_trunk_shapes",
        timeout_s: 600,
        run: gemm_bf16_mfma_trunk,
    },
    Q4Case {
        name: "qwen4_gemm_bf16_mfma_gfx942_rows16_group",
        timeout_s: 300,
        run: gemm_bf16_mfma_rows16_group,
    },
    Q4Case {
        name: "qwen4_gemm_bf16_mfma_gfx942_rows16_chain",
        timeout_s: 900,
        run: gemm_bf16_mfma_rows16_chain,
    },
    Q4Case {
        name: "qwen4_gemm_bf16_mfma_gfx942_wide",
        timeout_s: 1800,
        run: gemm_bf16_mfma_wide,
    },
    Q4Case {
        name: "qwen4_gemv_bf16_xf32_1row",
        timeout_s: 120,
        run: gemv_bf16_xf32_1row,
    },
    Q4Case {
        name: "qwen4_gemm_bf16_xf32_multirow_trunk_shapes",
        timeout_s: 180,
        run: gemm_bf16_xf32_multirow_trunk,
    },
    Q4Case {
        name: "qwen4_gemv_bf16_xf32_k4_1row",
        timeout_s: 120,
        run: gemv_bf16_xf32_k4_1row,
    },
];

/// Rows of every X the cases generate: `ROW_COUNTS` plus one count above two
/// 32-row MFMA batch tiles (65 = 2 full tiles + a 1-row tail tile).
const MAX_ROWS: usize = 65;
const MFMA_ROWS: [usize; 5] = [1, 7, 16, 33, MAX_ROWS];
/// Row counts the few-row entry is bit-compared against the tile entry at.
const FEW_ROWS: [usize; 5] = [1, 2, 7, 15, 16];
/// Weight magnitude. With x in [-1, 1) the dots have |y| ~ 0.3-0.5, so F32
/// accumulation error (~1e-7) sits two orders under `F32_ACCUM`'s atol.
const W_RANGE: f32 = 0.02;

struct Shape {
    name: &'static str,
    m: usize,
    k: usize,
}

/// Qwen4 trunk projection shapes the BF16 MFMA route launches.
const MFMA_SHAPES: [Shape; 7] = [
    // GDN in-projection qkv: 2*(16*128) + 48*128.
    Shape {
        name: "gdn_qkv",
        m: 10240,
        k: HIDDEN,
    },
    // GDN out-proj (48*128) and QSA out-proj (24*256): K = 6144.
    Shape {
        name: "out_proj",
        m: HIDDEN,
        k: 6144,
    },
    // GDN in_proj_a / in_proj_b: M = 48 = 1.5 MFMA M tiles (tail tile).
    Shape {
        name: "gdn_ba",
        m: GDN_VALUE_HEADS,
        k: HIDDEN,
    },
    // HC block-inject gate: M = 4 branches over the wide stream (M % 16 != 0).
    Shape {
        name: "hc_gate",
        m: HC_STREAMS,
        k: HC_WIDE,
    },
    // HC mix down: 20 M tiles over the longest K (the few-row entry's
    // chain-bound shape).
    Shape {
        name: "hc_down",
        m: 320,
        k: HC_WIDE,
    },
    // HC mix up: K = 320 is not a multiple of the 128-K chunk, so both entries
    // run 24 MFMA steps, the last 4 over the zero-filled tail.
    Shape {
        name: "hc_up",
        m: HC_WIDE,
        k: 320,
    },
    // Shared-expert gate: a single output row.
    Shape {
        name: "shared_gate",
        m: 1,
        k: HIDDEN,
    },
];

/// F64 reference: `y[r, m] = sum_k w[m, k] * x[r, k]`, rounded to F32 once.
fn ref_f64(w: &[f32], x: &[f32], m: usize, k: usize, rows: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * m);
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        for mm in 0..m {
            let wr = &w[mm * k..(mm + 1) * k];
            let s: f64 = wr.iter().zip(xr).map(|(&a, &b)| a as f64 * b as f64).sum();
            out.push(s as f32);
        }
    }
    out
}

/// Number of differing F32 bit patterns and the max |a - b|.
fn bit_diff(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut n = 0;
    let mut max = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            n += 1;
        }
        let d = (x - y).abs();
        if d > max || d.is_nan() {
            max = if d.is_nan() { f32::INFINITY } else { d };
        }
    }
    (n, max)
}

/// F32 → BF16 staging the way the route does it (`convert_f32_to_bf16`).
fn stage_bf16(gpu: &mut Gpu, x: &[f32]) -> Result<GpuTensor, String> {
    let src = up_f32(gpu, x)?;
    let dst = gpu.alloc_tensor(&[x.len()], DType::BF16).map_err(e)?;
    gpu.convert_f32_to_bf16(&src, &dst, x.len()).map_err(e)?;
    sync(gpu)?;
    gpu.free_tensor(src).map_err(e)?;
    Ok(dst)
}

/// One `gemm_bf16_mfma_gfx942_tiled` launch (the 32-row-tile entry at any
/// row count) of `n` rows of `x` (BF16) against `w`.
fn run_mfma_tiled(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    let y = sentinel_f32(gpu, n * m)?;
    gpu.gemm_bf16_mfma_gfx942_tiled(&w.buf, &x.buf, &y.buf, m, k, n)
        .map_err(e)?;
    sync(gpu)?;
    let out = down_f32(gpu, &y)?;
    gpu.free_tensor(y).map_err(e)?;
    Ok(out)
}

/// One fused few-row launch (`gemm_bf16_mfma_gfx942_rows16`) of `n` rows of
/// `x` (F32 or BF16) into a fresh `out_dtype` tensor, returned as raw bytes.
fn run_rows16_fused(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    out_dtype: DType,
) -> Result<Vec<u8>, String> {
    let y = if out_dtype == DType::F32 {
        sentinel_f32(gpu, n * m)?
    } else {
        gpu.alloc_tensor(&[n * m], out_dtype).map_err(e)?
    };
    let ran = gpu
        .gemm_bf16_mfma_gfx942_rows16(x, &[(&w.buf, &y, m)], k, n)
        .map_err(e)?;
    if !ran {
        gpu.free_tensor(y).map_err(e)?;
        return Err(format!("rows16 entry declined m={m} k={k} rows={n}"));
    }
    sync(gpu)?;
    let out = down_bytes(gpu, &y)?;
    gpu.free_tensor(y).map_err(e)?;
    Ok(out)
}

/// One `gemm_bf16_mfma_gfx942` launch of `n` rows of `x` (BF16) against `w`.
fn run_mfma(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    let y = sentinel_f32(gpu, n * m)?;
    gpu.gemm_bf16_mfma_gfx942(&w.buf, &x.buf, &y.buf, m, k, n)
        .map_err(e)?;
    sync(gpu)?;
    let out = down_f32(gpu, &y)?;
    gpu.free_tensor(y).map_err(e)?;
    Ok(out)
}

// ── GemmBf16Mfma ─────────────────────────────────────────────────────────────
// Reference: F64 dot of the BF16-decoded weights and the BF16-rounded x, under
// `BF16_INPUT_F32_ACCUM` (the frozen tol for a BF16-input F32 accumulation).
// The stricter `F32_ACCUM` result is reported too: the inputs are exact BF16, so
// only the F32 accumulation order differs.
//
// gfx942 branch: `Gpu::gemm_bf16_mfma_gfx942` hard-errors off gfx942; the case
// is skipped there (it cannot run on any other arch).
fn gemm_bf16_mfma_trunk(gpu: &mut Gpu) -> Result<String, String> {
    if !gpu.arch_caps.is_gfx942() {
        return Ok("skipped: gemm_bf16_mfma_gfx942 requires gfx942".to_string());
    }
    let mut results = Vec::new();
    let mut rng = Rng::new(0x7121_0001);
    for shape in &MFMA_SHAPES {
        let (m, k) = (shape.m, shape.k);
        let label = format!("mfma_{}[{m}x{k}]", shape.name);
        let w_host = rng.vec_bf16_exact(m * k, -W_RANGE, W_RANGE);
        let x_host = rng.vec(MAX_ROWS * k, -1.0, 1.0);
        let x_bf_host: Vec<f32> = x_host.iter().map(|&v| bf16_round(v)).collect();
        let want = ref_f64(&w_host, &x_bf_host, m, k, MAX_ROWS);

        let w = up_bf16(gpu, &w_host)?;
        let xbf = stage_bf16(gpu, &x_host)?;
        // The staged bits are the frozen RNE rule's bits.
        let staged = down_bf16(gpu, &xbf)?;
        let want_bits: Vec<u16> = x_host.iter().map(|&v| f32_to_bf16_bits(v)).collect();
        results.push(exact_bf16(
            &format!("{label} staged x"),
            &staged,
            &want_bits,
        ));

        // Every row count, against the reference.
        let mut by_rows: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut strict = Vec::new();
        for &n in &MFMA_ROWS {
            let xv = xbf.sub_offset(0, n * k);
            let got = run_mfma(gpu, &w, &xv, m, k, n)?;
            let tag = format!("{label} rows={n}");
            results.push(close_f32(&tag, &got, &want[..n * m], BF16_INPUT_F32_ACCUM));
            strict.push(match close_f32(&tag, &got, &want[..n * m], F32_ACCUM) {
                Ok(_) => format!("rows={n} F32_ACCUM met"),
                Err(detail) => format!("rows={n} F32_ACCUM exceeded ({detail})"),
            });
            by_rows.push((n, got));
        }

        // Row independence: bytes(row r of an M=n launch) == bytes(M=1 launch).
        let probe: Vec<usize> = [0usize, 1, 31, 32, 33, 63, 64]
            .into_iter()
            .filter(|&r| r < MAX_ROWS)
            .collect();
        let mut single: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut bad = Vec::new();
        for &r in &probe {
            let xr = xbf.sub_offset(r * k, k);
            let one = run_mfma(gpu, &w, &xr, m, k, 1)?;
            for (n, got) in &by_rows {
                if r < *n {
                    let (diff, max_abs) = bit_diff(&got[r * m..(r + 1) * m], &one);
                    if diff != 0 {
                        bad.push(format!(
                            "row {r} of n={n}: {diff} bit diffs, max_abs {max_abs:e}"
                        ));
                    }
                }
            }
            single.push((r, one));
        }
        results.push(if bad.is_empty() {
            Ok(format!(
                "{label} row-invariant: rows {probe:?} bit-equal between M=1 and every M=n launch"
            ))
        } else {
            Err(format!("{label} rows-dependent bytes: {}", bad.join("; ")))
        });

        // Few-row entry == tile entry, bit for bit, at every row of 1..=16-row
        // launches; then the fused form (F32 x rounded in-kernel; F32 and BF16
        // output) == staging + GEMM (+ RNE rounding of the F32 result).
        let x32 = up_f32(gpu, &x_host)?;
        let mut few_bad = Vec::new();
        let mut fused_bad = Vec::new();
        for &n in &FEW_ROWS {
            let xv = xbf.sub_offset(0, n * k);
            let tiled = run_mfma_tiled(gpu, &w, &xv, m, k, n)?;
            let few = run_mfma(gpu, &w, &xv, m, k, n)?;
            let (diff, max_abs) = bit_diff(&few, &tiled);
            if diff != 0 {
                few_bad.push(format!("rows={n}: {diff} bit diffs, max_abs {max_abs:e}"));
            }
            let want_f32: Vec<u8> = tiled.iter().flat_map(|v| v.to_le_bytes()).collect();
            let want_bf16: Vec<u8> = tiled
                .iter()
                .flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes())
                .collect();
            let x32v = x32.sub_offset(0, n * k);
            for (xin, xname) in [(&x32v, "x_f32"), (&xv, "x_bf16")] {
                for (dt, want, oname) in [
                    (DType::F32, &want_f32, "y_f32"),
                    (DType::BF16, &want_bf16, "y_bf16"),
                ] {
                    let got = run_rows16_fused(gpu, &w, xin, m, k, n, dt)?;
                    if &got != want {
                        let diff = got
                            .chunks(dt.size())
                            .zip(want.chunks(dt.size()))
                            .filter(|(a, b)| a != b)
                            .count();
                        fused_bad.push(format!("rows={n} {xname} {oname}: {diff} elements differ"));
                    }
                }
            }
        }
        results.push(if few_bad.is_empty() {
            Ok(format!(
                "{label} few-row entry == tile entry bitwise at rows {FEW_ROWS:?}"
            ))
        } else {
            Err(format!(
                "{label} few-row entry != tile entry: {}",
                few_bad.join("; ")
            ))
        });
        results.push(if fused_bad.is_empty() {
            Ok(format!(
                "{label} fused few-row (x F32/BF16, y F32/BF16) == staged GEMM bitwise at rows {FEW_ROWS:?}"
            ))
        } else {
            Err(format!("{label} fused few-row mismatch: {}", fused_bad.join("; ")))
        });
        gpu.free_tensor(x32).map_err(e)?;

        // REPORT (never fails): the 1-row gemv on the same BF16-exact inputs
        // against the MFMA row. Expected to differ (two reduction orders).
        let xb32 = up_f32(gpu, &x_bf_host)?;
        let mut report = Vec::new();
        for (r, mfma_row) in single.iter().filter(|(r, _)| *r == 0 || *r == 33) {
            let xr = xb32.sub_offset(r * k, k);
            let y = sentinel_f32(gpu, m)?;
            gpu.gemv_bf16_xf32(&w, &xr, &y, m, k).map_err(e)?;
            sync(gpu)?;
            let g = down_f32(gpu, &y)?;
            gpu.free_tensor(y).map_err(e)?;
            let (diff, max_abs) = bit_diff(&g, mfma_row);
            report.push(format!(
                "row {r}: gemv==mfma bitwise {}/{m} equal (max_abs {max_abs:.3e})",
                m - diff
            ));
        }
        results.push(Ok(format!(
            "{label} REPORT gemv_bf16_xf32 vs mfma [{}]; strict: [{}]",
            report.join("; "),
            strict.join("; ")
        )));

        free_all(gpu, vec![w, xbf, xb32])?;
    }
    all(results)
}

// ── Few-row entry, one launch per projection group ───────────────────────────
// The GDN in-projection group (qkv, z, a, b: four weights over one x, K =
// HIDDEN) as ONE `gemm_bf16_mfma_gfx942_rows16` launch with mixed output
// dtypes (F32, BF16, F32, BF16), F32 x: each member's bytes must equal its own
// staged tile-entry GEMM (BF16 members: + RNE rounding), at 1, 9 (the F32
// `_x2` entry) and 16 rows.
fn gemm_bf16_mfma_rows16_group(gpu: &mut Gpu) -> Result<String, String> {
    if !gpu.arch_caps.is_gfx942() {
        return Ok("skipped: gemm_bf16_mfma_gfx942 requires gfx942".to_string());
    }
    let k = HIDDEN;
    let ms = [10240usize, 6144, GDN_VALUE_HEADS, GDN_VALUE_HEADS];
    let dts = [DType::F32, DType::BF16, DType::F32, DType::BF16];
    let mut rng = Rng::new(0x7121_0002);
    let mut ws = Vec::new();
    for &m in &ms {
        let host = rng.vec_bf16_exact(m * k, -W_RANGE, W_RANGE);
        ws.push(up_bf16(gpu, &host)?);
    }
    let x_host = rng.vec(16 * k, -1.0, 1.0);
    let x32 = up_f32(gpu, &x_host)?;
    let xbf = stage_bf16(gpu, &x_host)?;
    let mut results = Vec::new();
    for n in [1usize, 9, 16] {
        let mut ys = Vec::new();
        for (&m, &dt) in ms.iter().zip(&dts) {
            ys.push(gpu.alloc_tensor(&[n * m], dt).map_err(e)?);
        }
        let x = x32.sub_offset(0, n * k);
        let members: Vec<_> = ws
            .iter()
            .zip(&ys)
            .zip(&ms)
            .map(|((w, y), &m)| (&w.buf, y, m))
            .collect();
        let ran = gpu
            .gemm_bf16_mfma_gfx942_rows16(&x, &members, k, n)
            .map_err(e)?;
        if !ran {
            return Err(format!("group launch declined at rows={n}"));
        }
        sync(gpu)?;
        let mut bad = Vec::new();
        for (j, ((w, y), &m)) in ws.iter().zip(&ys).zip(&ms).enumerate() {
            let want = run_mfma_tiled(gpu, w, &xbf.sub_offset(0, n * k), m, k, n)?;
            let want: Vec<u8> = if dts[j] == DType::F32 {
                want.iter().flat_map(|v| v.to_le_bytes()).collect()
            } else {
                want.iter()
                    .flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes())
                    .collect()
            };
            let got = down_bytes(gpu, y)?;
            if got != want {
                let diff = got
                    .chunks(dts[j].size())
                    .zip(want.chunks(dts[j].size()))
                    .filter(|(a, b)| a != b)
                    .count();
                bad.push(format!(
                    "member {j} [{m}x{k} {:?}]: {diff} elements differ",
                    dts[j]
                ));
            }
        }
        results.push(if bad.is_empty() {
            Ok(format!(
                "gdn_in group rows={n}: 4 members == staged tile GEMM bitwise"
            ))
        } else {
            Err(format!("gdn_in group rows={n}: {}", bad.join("; ")))
        });
        free_all(gpu, ys)?;
    }
    ws.push(x32);
    ws.push(xbf);
    free_all(gpu, ws)?;
    all(results)
}

// ── Chained few-row entry past 16 rows (P1a) ────────────────────────────────
/// Row counts the chain is bit-compared against the tile entry at.
const CHAIN_ROWS: [usize; 6] = [17, 31, 57, 145, 456, 912];

// `Gpu::gemm_bf16_mfma_gfx942_rows16_chain` at 17 / 31 / 57 / 145 / 456 / 912
// rows: every trunk shape alone (F32 x, F32 and BF16 out) and the GDN
// in-projection group (F32 x, mixed out). Where the shape's threshold admits
// the chain every member's bytes must equal its staged tile-entry GEMM (BF16:
// + RNE rounding); where it does not, the call must decline without writing
// (the output keeps its sentinel).
fn gemm_bf16_mfma_rows16_chain(gpu: &mut Gpu) -> Result<String, String> {
    if !gpu.arch_caps.is_gfx942() {
        return Ok("skipped: gemm_bf16_mfma_gfx942 requires gfx942".to_string());
    }
    let max_rows = *CHAIN_ROWS.iter().max().unwrap();
    let mut rng = Rng::new(0x7121_0003);
    let mut groups: Vec<(String, usize, Vec<usize>, Vec<DType>)> = MFMA_SHAPES
        .iter()
        .flat_map(|s| {
            [DType::F32, DType::BF16]
                .into_iter()
                .map(move |dt| (format!("{} {dt:?}", s.name), s.k, vec![s.m], vec![dt]))
        })
        .collect();
    groups.push((
        "gdn_in group".to_string(),
        HIDDEN,
        vec![10240, 6144, GDN_VALUE_HEADS, GDN_VALUE_HEADS],
        vec![DType::F32, DType::BF16, DType::F32, DType::BF16],
    ));
    let mut results = Vec::new();
    for (name, k, ms, dts) in &groups {
        let k = *k;
        let mut ws = Vec::new();
        for &m in ms {
            let host = rng.vec_bf16_exact(m * k, -W_RANGE, W_RANGE);
            ws.push(up_bf16(gpu, &host)?);
        }
        let x_host = rng.vec(max_rows * k, -1.0, 1.0);
        let x32 = up_f32(gpu, &x_host)?;
        let xbf = stage_bf16(gpu, &x_host)?;
        for n in CHAIN_ROWS {
            let admitted = ms
                .iter()
                .all(|&m| n <= rdna_compute::cdna::gfx942::rows16_chain_max_rows(m, k));
            let mut ys = Vec::new();
            for (&m, &dt) in ms.iter().zip(dts) {
                let mut y = gpu
                    .upload_raw(&vec![0xA5u8; n * m * dt.size()], &[n * m])
                    .map_err(e)?;
                y.dtype = dt;
                ys.push(y);
            }
            let members: Vec<_> = ws
                .iter()
                .zip(&ys)
                .zip(ms)
                .map(|((w, y), &m)| (&w.buf, y, m))
                .collect();
            let ran = gpu
                .gemm_bf16_mfma_gfx942_rows16_chain(
                    &x32.sub_offset(0, n * k),
                    &members,
                    k,
                    n,
                    true,
                    false,
                )
                .map_err(e)?;
            sync(gpu)?;
            if ran != admitted {
                results.push(Err(format!(
                    "{name} rows={n}: chain ran={ran}, threshold admits={admitted}"
                )));
                free_all(gpu, ys)?;
                continue;
            }
            let mut bad = Vec::new();
            for (j, ((w, y), &m)) in ws.iter().zip(&ys).zip(ms).enumerate() {
                let got = down_bytes(gpu, y)?;
                let want: Vec<u8> = if !admitted {
                    vec![0xA5; n * m * dts[j].size()]
                } else {
                    let t = run_mfma_tiled(gpu, w, &xbf.sub_offset(0, n * k), m, k, n)?;
                    if dts[j] == DType::F32 {
                        t.iter().flat_map(|v| v.to_le_bytes()).collect()
                    } else {
                        t.iter()
                            .flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes())
                            .collect()
                    }
                };
                if got != want {
                    let diff = got
                        .chunks(dts[j].size())
                        .zip(want.chunks(dts[j].size()))
                        .filter(|(a, b)| a != b)
                        .count();
                    bad.push(format!(
                        "member {j} [{m}x{k} {:?}]: {diff} elements differ",
                        dts[j]
                    ));
                }
            }
            let what = if admitted {
                "chain == staged tile GEMM bitwise"
            } else {
                "declined, untouched"
            };
            results.push(if bad.is_empty() {
                Ok(format!("{name} rows={n}: {what}"))
            } else {
                Err(format!("{name} rows={n}: {}", bad.join("; ")))
            });
            free_all(gpu, ys)?;
        }
        ws.push(x32);
        ws.push(xbf);
        free_all(gpu, ws)?;
    }
    all(results)
}

// ── Wide entry (P1b) ─────────────────────────────────────────────────────────
/// Row counts the wide entry is bit-compared against the tile entry at: every
/// single-slice count 1..=64 and multi-slice counts that end on a partial
/// slice (65 / 257 / 1025).
fn wide_rows() -> Vec<usize> {
    (1..=64).chain([65, 257, 1025]).collect()
}

// `Gpu::gemm_bf16_mfma_gfx942_wide_slices` (one `gemm_bf16_mfma_gfx942_wide`
// launch per 64 rows) at every `wide_rows` count: every trunk shape alone (F32
// and BF16 x, F32 and BF16 out) and the GDN in-projection group (four members,
// F32 and BF16 x, mixed out). Every member's bytes must equal its staged
// tile-entry GEMM (BF16: + RNE rounding). Then the route
// (`gemm_bf16_mfma_gfx942_rows16_chain` with the wide switch on) at the same
// counts and x dtypes: where it runs, the same bytes; where it declines, the
// output keeps its sentinel.
fn gemm_bf16_mfma_wide(gpu: &mut Gpu) -> Result<String, String> {
    if !gpu.arch_caps.is_gfx942() {
        return Ok("skipped: gemm_bf16_mfma_gfx942 requires gfx942".to_string());
    }
    let rows_list = wide_rows();
    let max_rows = *rows_list.iter().max().unwrap();
    let mut rng = Rng::new(0x7121_0004);
    let mut groups: Vec<(String, usize, Vec<usize>, Vec<DType>)> = MFMA_SHAPES
        .iter()
        .flat_map(|s| {
            [DType::F32, DType::BF16]
                .into_iter()
                .map(move |dt| (format!("{} {dt:?}", s.name), s.k, vec![s.m], vec![dt]))
        })
        .collect();
    groups.push((
        "gdn_in group".to_string(),
        HIDDEN,
        vec![10240, 6144, GDN_VALUE_HEADS, GDN_VALUE_HEADS],
        vec![DType::F32, DType::BF16, DType::F32, DType::BF16],
    ));
    let mut results = Vec::new();
    for (name, k, ms, dts) in &groups {
        let k = *k;
        let mut ws = Vec::new();
        for &m in ms {
            let host = rng.vec_bf16_exact(m * k, -W_RANGE, W_RANGE);
            ws.push(up_bf16(gpu, &host)?);
        }
        let x_host = rng.vec(max_rows * k, -1.0, 1.0);
        let x32 = up_f32(gpu, &x_host)?;
        let xbf = stage_bf16(gpu, &x_host)?;
        for &n in &rows_list {
            // (label, expected to run, route): direct slices always run; the
            // route runs where a few-row entry, the wide threshold (any
            // member: a small member rides in a large one's launch) or the
            // chain threshold (every member) admits the group.
            let route_runs = n <= 16
                || ms
                    .iter()
                    .any(|&m| n <= rdna_compute::cdna::gfx942::wide_max_rows(m, k))
                || ms
                    .iter()
                    .all(|&m| n <= rdna_compute::cdna::gfx942::rows16_chain_max_rows(m, k));
            // Each member's staged tile-entry bytes (BF16: + RNE rounding).
            let mut tile_want: Vec<Vec<u8>> = Vec::new();
            for (j, (w, &m)) in ws.iter().zip(ms).enumerate() {
                let t = run_mfma_tiled(gpu, w, &xbf.sub_offset(0, n * k), m, k, n)?;
                tile_want.push(if dts[j] == DType::F32 {
                    t.iter().flat_map(|v| v.to_le_bytes()).collect()
                } else {
                    t.iter()
                        .flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes())
                        .collect()
                });
            }
            for (label, via_route, x_bf16) in [
                ("wide", false, false),
                ("wide xbf16", false, true),
                ("route", true, false),
                ("route xbf16", true, true),
            ] {
                let admitted = !via_route || route_runs;
                let mut ys = Vec::new();
                for (&m, &dt) in ms.iter().zip(dts) {
                    let mut y = gpu
                        .upload_raw(&vec![0xA5u8; n * m * dt.size()], &[n * m])
                        .map_err(e)?;
                    y.dtype = dt;
                    ys.push(y);
                }
                let members: Vec<_> = ws
                    .iter()
                    .zip(&ys)
                    .zip(ms)
                    .map(|((w, y), &m)| (&w.buf, y, m))
                    .collect();
                let x = if x_bf16 { &xbf } else { &x32 }.sub_offset(0, n * k);
                let ran = if via_route {
                    gpu.gemm_bf16_mfma_gfx942_rows16_chain(&x, &members, k, n, true, true)
                } else {
                    gpu.gemm_bf16_mfma_gfx942_wide_slices(&x, &members, k, n)
                }
                .map_err(e)?;
                sync(gpu)?;
                if ran != admitted {
                    results.push(Err(format!(
                        "{name} {label} rows={n}: ran={ran}, expected {admitted}"
                    )));
                    free_all(gpu, ys)?;
                    continue;
                }
                let mut bad = Vec::new();
                for (j, (y, &m)) in ys.iter().zip(ms).enumerate() {
                    let got = down_bytes(gpu, y)?;
                    let want: Vec<u8> = if admitted {
                        tile_want[j].clone()
                    } else {
                        vec![0xA5; n * m * dts[j].size()]
                    };
                    if got != want {
                        let diff = got
                            .chunks(dts[j].size())
                            .zip(want.chunks(dts[j].size()))
                            .filter(|(a, b)| a != b)
                            .count();
                        bad.push(format!(
                            "member {j} [{m}x{k} {:?}]: {diff} elements differ",
                            dts[j]
                        ));
                    }
                }
                let what = if admitted {
                    "== staged tile GEMM bitwise"
                } else {
                    "declined, untouched"
                };
                results.push(if bad.is_empty() {
                    Ok(format!("{name} {label} rows={n}: {what}"))
                } else {
                    Err(format!("{name} {label} rows={n}: {}", bad.join("; ")))
                });
                free_all(gpu, ys)?;
            }
        }
        ws.push(x32);
        ws.push(xbf);
        free_all(gpu, ws)?;
    }
    all(results)
}

// ── gemv_bf16_xf32 (rows = 1) ────────────────────────────────────────────────
// BF16 weights x F32 x, F32 accumulation in the kernel's own association
// (4 chains per lane, 32-lane shuffle tree); reference is the F64 dot under
// `F32_ACCUM` (weights are BF16-exact, x is not rounded, magnitudes keep the
// F32 accumulation error two orders under atol).
//
// gfx942 branch: `gemv_bf16_xf32_src(is_gfx1151 == false)` = gemv_bf16_xf32.hip,
// block 32 (half a wave64: lanes 32..63 are idle; the tree stays in lanes 0..31).
fn gemv_bf16_xf32_1row(gpu: &mut Gpu) -> Result<String, String> {
    const SHAPES: [Shape; 5] = [
        Shape {
            name: "gdn_qkv",
            m: 10240,
            k: HIDDEN,
        },
        Shape {
            name: "out_proj",
            m: HIDDEN,
            k: 6144,
        },
        Shape {
            name: "gdn_ba",
            m: GDN_VALUE_HEADS,
            k: HIDDEN,
        },
        Shape {
            name: "hc_gate",
            m: HC_STREAMS,
            k: HC_WIDE,
        },
        // 8192 rows of the 248,320-row LM head (the grid is one block per row).
        Shape {
            name: "lm_head_slice",
            m: 8192,
            k: HIDDEN,
        },
    ];
    let mut results = Vec::new();
    let mut rng = Rng::new(0x7121_0002);
    for shape in &SHAPES {
        let (m, k) = (shape.m, shape.k);
        let w_host = rng.vec_bf16_exact(m * k, -W_RANGE, W_RANGE);
        let x_host = rng.vec(k, -1.0, 1.0);
        let want = ref_f64(&w_host, &x_host, m, k, 1);
        let w = up_bf16(gpu, &w_host)?;
        let x = up_f32(gpu, &x_host)?;
        let y = sentinel_f32(gpu, m)?;
        gpu.gemv_bf16_xf32(&w, &x, &y, m, k).map_err(e)?;
        sync(gpu)?;
        let got = down_f32(gpu, &y)?;
        results.push(close_f32(
            &format!("gemv_bf16_xf32_{}[{m}x{k}]", shape.name),
            &got,
            &want,
            F32_ACCUM,
        ));
        free_all(gpu, vec![w, x, y])?;
    }
    all(results)
}

// ── gemm_bf16_xf32_multirow (rows >= 2, census-base route) ───────────────────
// Documented as bitwise the gemv per row; the reference is the F64 dot under
// `F32_ACCUM`, and the bitwise relation to `gemv_bf16_xf32` is REPORTED.
//
// gfx942 branch: neither gfx11 (`has_gfx11_plus_simt`) tile arm applies, so the
// four-token kernel `gemm_bf16_xf32_multirow` launches (grid [M, ceil(N/4)],
// block 32).
fn gemm_bf16_xf32_multirow_trunk(gpu: &mut Gpu) -> Result<String, String> {
    const ROWS: [usize; 4] = [2, 7, 33, MAX_ROWS];
    let mut results = Vec::new();
    let mut rng = Rng::new(0x7121_0003);
    let shapes: Vec<Shape> = MFMA_SHAPES
        .iter()
        .map(|s| Shape {
            name: s.name,
            m: s.m,
            k: s.k,
        })
        .chain([Shape {
            name: "router",
            m: MOE_EXPERTS,
            k: HIDDEN,
        }])
        .collect();
    for shape in &shapes {
        let (m, k) = (shape.m, shape.k);
        let label = format!("multirow_{}[{m}x{k}]", shape.name);
        let w_host = rng.vec_bf16_exact(m * k, -W_RANGE, W_RANGE);
        let x_host = rng.vec(MAX_ROWS * k, -1.0, 1.0);
        let want = ref_f64(&w_host, &x_host, m, k, MAX_ROWS);
        let w = up_bf16(gpu, &w_host)?;
        let x = up_f32(gpu, &x_host)?;
        // Reference rows of the single-row GEMV (the bitwise-twin claim).
        let mut gemv_rows: Vec<Vec<f32>> = Vec::new();
        for r in 0..MAX_ROWS {
            let xr = x.sub_offset(r * k, k);
            let y = sentinel_f32(gpu, m)?;
            gpu.gemv_bf16_xf32(&w, &xr, &y, m, k).map_err(e)?;
            sync(gpu)?;
            gemv_rows.push(down_f32(gpu, &y)?);
            gpu.free_tensor(y).map_err(e)?;
        }
        let mut report = Vec::new();
        for &n in &ROWS {
            let xv = x.sub_offset(0, n * k);
            let y = sentinel_f32(gpu, n * m)?;
            gpu.gemm_bf16_xf32_multirow(&w, &xv, &y, m, k, n)
                .map_err(e)?;
            sync(gpu)?;
            let got = down_f32(gpu, &y)?;
            gpu.free_tensor(y).map_err(e)?;
            results.push(close_f32(
                &format!("{label} rows={n}"),
                &got,
                &want[..n * m],
                F32_ACCUM,
            ));
            let diff: usize = (0..n)
                .map(|r| bit_diff(&got[r * m..(r + 1) * m], &gemv_rows[r]).0)
                .sum();
            report.push(format!(
                "rows={n}: {diff}/{} bits differ from gemv rows",
                n * m
            ));
        }
        results.push(Ok(format!(
            "{label} REPORT multirow vs gemv [{}]",
            report.join("; ")
        )));
        free_all(gpu, vec![w, x])?;
    }
    all(results)
}

// ── gemv_bf16_xf32_k4 (four waves per row; not reached by the gfx942 route) ──
// Quarter sums folded `(q0 + q1) + (q2 + q3)`; reference is the F64 dot under
// `F32_ACCUM`. A launch/module refusal on this arch is reported as a note (the
// kernel is unreachable on the census base), a numeric mismatch fails.
fn gemv_bf16_xf32_k4_1row(gpu: &mut Gpu) -> Result<String, String> {
    const SHAPES: [Shape; 3] = [
        // HC read down projection: low-rank 320 over the 10,240-wide stream.
        Shape {
            name: "hc_down",
            m: 320,
            k: HC_WIDE,
        },
        Shape {
            name: "gdn_ba",
            m: GDN_VALUE_HEADS,
            k: HIDDEN,
        },
        Shape {
            name: "hc_gate",
            m: HC_STREAMS,
            k: HC_WIDE,
        },
    ];
    let mut results = Vec::new();
    let mut rng = Rng::new(0x7121_0004);
    for shape in &SHAPES {
        let (m, k) = (shape.m, shape.k);
        let label = format!("gemv_bf16_xf32_k4_{}[{m}x{k}]", shape.name);
        let w_host = rng.vec_bf16_exact(m * k, -W_RANGE, W_RANGE);
        let x_host = rng.vec(k, -1.0, 1.0);
        let want = ref_f64(&w_host, &x_host, m, k, 1);
        let w = up_bf16(gpu, &w_host)?;
        let x = up_f32(gpu, &x_host)?;
        let y = sentinel_f32(gpu, m)?;
        let launched = gpu
            .gemv_bf16_xf32_k4(&w, &x, &y, m, k, None)
            .map_err(e)
            .and_then(|_| sync(gpu));
        match launched {
            Ok(()) => {
                let got = down_f32(gpu, &y)?;
                results.push(close_f32(&label, &got, &want, F32_ACCUM));
            }
            Err(detail) => results.push(Ok(format!(
                "{label}: launch refused on {} (unreachable on the gfx942 route): {detail}",
                gpu.arch
            ))),
        }
        free_all(gpu, vec![w, x, y])?;
    }
    all(results)
}
