// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU correctness oracle for the qt47 (MQ6G256V2) MoE kernels.
//!
//! Covers the four qt47 MoE kernels this branch adds:
//!   * `gemm_mq6g256v2_moe_grouped_wmma_k2`   — prefill, arch-selecting
//!     (gfx11 `_k2` source / gfx12 `_gfx12` source)
//!   * `gemv_mq6g256v2_moe_gate_up_k8_indexed`        — decode
//!   * `gemv_mq6g256v2_moe_gate_up_k8_indexed_batched` — decode batched N>1
//!   * `gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded` — decode
//!   * `gemv_mq6g256v2_moe_ninepath_d4`       — decode, fused down + combine
//!
//! WHY THIS EXISTS. Sister of `mq4v2_moe_parity` for the 6-bit V2 layout. qt47
//! shares the dual-half fp16 header contract with qt44 but packs 6-bit codes
//! into 192 B after the 8-byte header (200 B/group). A kernel that collapses
//! V2 dual-half headers onto V1's single f32 scale/zero, or that reads the
//! wrong half's grid, produces fluent corruption rather than a fault.
//!
//!   cargo run --release -p rdna-compute --example mq6v2_moe_parity --features lab
//!
//! Exit code is non-zero if any check fails, so it can gate a script.
//!
//! ── The two fixture traps ────────────────────────────────────────────────
//! Both produced false failures on correct kernels during the qt44 dense-GEMM
//! work, and both would have led to "fixing" working code. Carried forward:
//!
//!   1. fp16 A-operand rounding amplified by catastrophic cancellation. The
//!      grouped GEMM dequantizes in fp16; a reference that dequantizes in f32
//!      and then subtracts near-equal products disagrees by far more than the
//!      kernel is wrong. This oracle scores with relative L2 over the whole
//!      output vector, never per-element on cancelling sums.
//!   2. Per-element relative error dividing by a near-zero denominator. At
//!      N=5 with 160 outputs it is invisible; at N=512 with 524,288 outputs it
//!      is unavoidable. Hence rel-L2 + cosine, not max per-element relative.
//!
//! ── The negative control ─────────────────────────────────────────────────
//! qt14 (legacy MQ6/HFQ6) and qt47 share a 200 B group stride and identical
//! 6-bit packing. They differ ONLY in the 8-byte header: V1 stores one f32
//! scale + one f32 zero for all 256 weights, V2 stores two f16 scale/zero
//! pairs, one per 128-weight half. A kernel that reads the wrong header
//! produces plausible numbers, not a fault.
//!
//! So every check here runs against `build_disjoint_halves`, whose two halves
//! occupy DISJOINT ranges ([-1,1] and [96,160]). Under that fixture a
//! grid-selection error is a ~100x scale error, not a rounding difference. The
//! oracle then asserts a deliberately grid-swapped reference DISAGREES — a
//! test that passes with the halves swapped would be measuring nothing.
//!
//! ── Issue 9 gaps (this wave) ─────────────────────────────────────────────
//! * batched gate/down N>1 via the `*_batched` launchers — N=1 alone hides
//!   batch-stride / expert-table bugs.
//! * token-distinct routes — each batch lane picks different experts so a
//!   lane that reuses token 0's table is visible.
//! * two-expert and high-ID experts — n_exp=2 minimal and n_exp=64 with
//!   IDs 63, 48… exercise high pointer-table offsets.
//! * nonidentity grouped permutation — slot map shuffled, not 0..m_total.
//! * production K/M — gate_up K=2048 (hidden_size) and down M=2048 cover
//!   the real A3B (dim 2048, moe_intermediate 512) shapes, not just K=512.
//! * isolated half controls — host pack->dequant with one half constant so
//!   the other half's grid is isolated.
//! * equal-length + finite scoring — Report asserts len equality and finiteness
//!   explicitly; truncated zip would hide shape bugs.

use rdna_compute::{DType, Gpu, GpuTensor};

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 200;
const MAX_Q: f32 = 63.0;

// ── fixture + packing (shared convention with mqv2_family_parity) ────────

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let mut exp = ((bits >> 10) & 0x1f) as u32;
    let mut mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 127 - 15 + 1;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((mant & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

fn half_from_f32(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut val = (b & 0x7fff_ffff) as i32;
    if val >= 0x4780_0000 {
        return sign | 0x7c00; // inf / overflow
    }
    if val < 0x3880_0000 {
        // Subnormal half: scale into the 10-bit subnormal grid and round.
        let f = f32::from_bits(val as u32);
        let sub = (f * 2f32.powi(24)).round() as i32;
        return sign | (sub as u16 & 0x03ff);
    }
    val += 0x0000_1000; // round-to-nearest on the truncated mantissa
    sign | (((val - 0x3800_0000) >> 13) as u16)
}

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// Weights whose two 128-halves occupy disjoint ranges, so a grid-selection
/// error is a scale error rather than a rounding difference.
fn build_disjoint_halves(m: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32;
            for i in 0..HALF {
                w[base + i] = prng(i, salt) * 2.0 - 1.0;
            }
            for i in HALF..GROUP {
                w[base + i] = 96.0 + prng(i, salt ^ 0xA5A5_A5A5) * 64.0;
            }
        }
    }
    w
}

/// Weights where one half is isolated (constant) so the other's grid is
/// exercised alone. Two variants: lo-isolated (hi constant) and hi-isolated.
fn build_isolated_lo(m: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32;
            for i in 0..HALF {
                w[base + i] = prng(i, salt) * 2.0 - 1.0;
            }
            for i in HALF..GROUP {
                w[base + i] = 120.0; // constant mid-range of hi half
            }
        }
    }
    w
}

fn build_isolated_hi(m: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32;
            for i in 0..HALF {
                w[base + i] = 0.0;
            }
            for i in HALF..GROUP {
                w[base + i] = 96.0 + prng(i, salt ^ 0xA5A5_A5A5) * 64.0;
            }
        }
    }
    w
}

/// Pack MQ6G256V2 (qt47): dual fp16 s/z header + 192 B 6-bit payload.
/// FWHT-free — kernels consume already-rotated activations, and the fixture
/// is natural-basis weights the host reference dequantizes the same way.
fn pack_mq6g256v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0, "k must be multiple of 256");
    assert_eq!(w.len(), m * k);
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut codes = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let slice = &w[src + off..src + off + HALF];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / MAX_Q } else { 0.0 };
                let s_bits = if hi == lo { 0u16 } else { half_from_f32(step) };
                let z_bits = half_from_f32(lo);
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&s_bits.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&z_bits.to_le_bytes());
                let s_rt = f16_to_f32(s_bits);
                let z_rt = f16_to_f32(z_bits);
                if s_rt == 0.0 {
                    continue;
                }
                let inv = 1.0 / s_rt;
                for i in 0..HALF {
                    let q = ((slice[i] - z_rt) * inv + 0.5).floor().clamp(0.0, MAX_Q);
                    codes[off + i] = q as u8;
                }
            }
            // 4 weights per 3 bytes: q0 | q1<<6 ; q1>>2 | q2<<4 ; q2>>4 | q3<<2
            for i in (0..GROUP).step_by(4) {
                let bo = dst + 8 + (i / 4) * 3;
                let q0 = codes[i] & 63;
                let q1 = codes[i + 1] & 63;
                let q2 = codes[i + 2] & 63;
                let q3 = codes[i + 3] & 63;
                blob[bo] = q0 | (q1 << 6);
                blob[bo + 1] = (q1 >> 2) | (q2 << 4);
                blob[bo + 2] = (q2 >> 4) | (q3 << 2);
            }
        }
    }
    blob
}

/// Dequantize a packed row back to f32. `swap_grids` deliberately reads the
/// WRONG affine grid for each half — the negative control.
fn dequant_row(blob: &[u8], row: usize, k: usize, swap_grids: bool) -> Vec<f32> {
    let gpr = k / GROUP;
    let mut out = vec![0.0f32; k];
    for g in 0..gpr {
        let dst = (row * gpr + g) * GROUP_BYTES;
        let hdr = |h: usize| {
            let o = dst + h * 4;
            let s = u16::from_le_bytes([blob[o], blob[o + 1]]);
            let z = u16::from_le_bytes([blob[o + 2], blob[o + 3]]);
            (f16_to_f32(s), f16_to_f32(z))
        };
        let (s0, z0) = hdr(0);
        let (s1, z1) = hdr(1);
        let sc = if swap_grids { [s1, s0] } else { [s0, s1] };
        let zp = if swap_grids { [z1, z0] } else { [z0, z1] };
        for i in (0..GROUP).step_by(4) {
            let bo = dst + 8 + (i / 4) * 3;
            let b0 = blob[bo] as u32;
            let b1 = blob[bo + 1] as u32;
            let b2 = blob[bo + 2] as u32;
            let pk = b0 | (b1 << 8) | (b2 << 16);
            let qs = [
                (pk & 63) as f32,
                ((pk >> 6) & 63) as f32,
                ((pk >> 12) & 63) as f32,
                ((pk >> 18) & 63) as f32,
            ];
            for j in 0..4 {
                let idx = i + j;
                let h = idx / HALF;
                out[g * GROUP + idx] = sc[h] * qs[j] + zp[h];
            }
        }
    }
    out
}

// ── scoring (see the fixture-trap note in the module docs) ──────────────

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(
        got.len(),
        want.len(),
        "rel_l2: length mismatch {} vs {}",
        got.len(),
        want.len()
    );
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        let d = (*g as f64) - (*w as f64);
        num += d * d;
        den += (*w as f64) * (*w as f64);
    }
    if den == 0.0 {
        return if num == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (num / den).sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "cosine: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

struct Report {
    failures: usize,
}

impl Report {
    fn check(&mut self, label: &str, got: &[f32], want: &[f32], tol: f64) {
        if got.len() != want.len() {
            println!(
                "  {:<44} len mismatch got {} vs want {} FAIL",
                label,
                got.len(),
                want.len()
            );
            self.failures += 1;
            return;
        }
        let finite_got = got.iter().all(|v| v.is_finite());
        let finite_want = want.iter().all(|v| v.is_finite());
        let finite = finite_got && finite_want;
        let r = rel_l2(got, want);
        let c = cosine(got, want);
        let ok = finite && r <= tol;
        println!(
            "  {:<44} rel_l2={:<12.3e} cos={:<10.6} tol={:<9.0e} {}",
            label,
            r,
            c,
            tol,
            if ok { "PASS" } else { "FAIL" }
        );
        if !finite_got {
            println!("      -> got contains non-finite values");
        }
        if !finite_want {
            println!("      -> want contains non-finite values (reference bug)");
        }
        if !ok {
            self.failures += 1;
        }
    }

    /// Negative control: this comparison MUST fail. A test that passes here is
    /// measuring nothing.
    fn check_disagrees(&mut self, label: &str, got: &[f32], want: &[f32], min_rel: f64) {
        if got.len() != want.len() {
            println!(
                "  {:<44} len mismatch got {} vs want {} FAIL (vacuous!)",
                label,
                got.len(),
                want.len()
            );
            self.failures += 1;
            return;
        }
        let r = rel_l2(got, want);
        let ok = !(r.is_finite() && r < min_rel);
        println!(
            "  {:<44} rel_l2={:<12.3e} (must exceed {:<8.0e}) {}",
            label,
            r,
            min_rel,
            if ok { "PASS" } else { "FAIL (vacuous!)" }
        );
        if !ok {
            self.failures += 1;
        }
    }
}

// ── host self-test: runs with no GPU ────────────────────────────────────

fn host_self_test(rep: &mut Report) {
    println!("host self-test (no GPU):");
    let (m, k) = (4usize, 512usize);
    let w = build_disjoint_halves(m, k);
    let blob = pack_mq6g256v2(&w, m, k);
    assert_eq!(blob.len(), m * (k / GROUP) * GROUP_BYTES);

    // The pack/dequant round trip must land within 6-bit quantisation error.
    // Each half spans its own range, so the error scale differs per half —
    // rel-L2 over the whole row is the honest aggregate.
    let mut got = Vec::new();
    let mut want = Vec::new();
    for r in 0..m {
        got.extend_from_slice(&dequant_row(&blob, r, k, false));
        want.extend_from_slice(&w[r * k..(r + 1) * k]);
    }
    rep.check("pack->dequant round trip", &got, &want, 2e-2);

    // Negative control: reading the other half's grid must be badly wrong.
    // If this ever passes, the fixture has stopped separating the halves and
    // every grid-selection check in this file is vacuous.
    let mut swapped = Vec::new();
    for r in 0..m {
        swapped.extend_from_slice(&dequant_row(&blob, r, k, true));
    }
    rep.check_disagrees(
        "grid-swapped dequant (negative control)",
        &swapped,
        &want,
        1e-1,
    );

    // ── isolated half controls ──────────────────────────────────────
    // Each half's header/scale is independent. Build weights where one half
    // is constant so only the other half's quant grid matters; the other
    // direction isolates the opposite half. Both must round-trip, and both
    // must disagree when grids are swapped — otherwise the header for that
    // half is not being read.
    for (label, w_iso) in [
        ("isolated lo half (hi const)", build_isolated_lo(m, k)),
        ("isolated hi half (lo const)", build_isolated_hi(m, k)),
    ] {
        let blob_iso = pack_mq6g256v2(&w_iso, m, k);
        let mut got_iso = Vec::new();
        let mut want_iso = Vec::new();
        for r in 0..m {
            got_iso.extend_from_slice(&dequant_row(&blob_iso, r, k, false));
            want_iso.extend_from_slice(&w_iso[r * k..(r + 1) * k]);
        }
        // Constant half quantizes exactly; lo half still has 6-bit error.
        rep.check(label, &got_iso, &want_iso, 2e-2);
        let mut swapped_iso = Vec::new();
        for r in 0..m {
            swapped_iso.extend_from_slice(&dequant_row(&blob_iso, r, k, true));
        }
        let neg_label = format!("{label} grid-swapped (negative control)");
        rep.check_disagrees(&neg_label, &swapped_iso, &want_iso, 1e-1);
    }
    println!();
}

// ── GPU checks ──────────────────────────────────────────────────────────

fn upload_experts(gpu: &mut Gpu, blobs: &[Vec<u8>]) -> (Vec<GpuTensor>, GpuTensor) {
    let experts: Vec<GpuTensor> = blobs
        .iter()
        .map(|b| gpu.upload_raw(b, &[b.len()]).unwrap())
        .collect();
    let ptrs: Vec<u8> = experts
        .iter()
        .flat_map(|t| (t.buf.as_ptr() as u64).to_le_bytes())
        .collect();
    let tab = gpu.upload_raw(&ptrs, &[blobs.len()]).unwrap();
    (experts, tab)
}

fn gate_up_check(gpu: &mut Gpu, rep: &mut Report) {
    // A3B routed gate_up decode shape: M = 2*mi (gate rows then up rows).
    let (mi, k, k_top, n_exp) = (64usize, 512usize, 8usize, 8usize);
    let m = 2 * mi;
    println!("gate_up  M={m} (mi={mi}) K={k} k_top={k_top} n_exp={n_exp}");

    let weights: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            let mut w = build_disjoint_halves(m, k);
            // Make experts distinguishable so a mis-indexed expert is visible.
            for v in w.iter_mut() {
                *v += e as f32 * 0.25;
            }
            w
        })
        .collect();
    let blobs: Vec<Vec<u8>> = weights.iter().map(|w| pack_mq6g256v2(w, m, k)).collect();
    let (_experts, ptr_tab) = upload_experts(gpu, &blobs);

    // Route each rank to a different expert — exercises the index path.
    let topk: Vec<i32> = (0..k_top).map(|r| (r % n_exp) as i32).collect();
    let topk_b: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_b, &[k_top]).unwrap();

    let x: Vec<f32> = (0..k).map(|i| prng(i, 0xBEEF) * 2.0 - 1.0).collect();
    let x_t = gpu.upload_f32(&x, &[k]).unwrap();
    let y_g = gpu.alloc_tensor(&[k_top * mi], DType::F32).unwrap();
    let y_u = gpu.alloc_tensor(&[k_top * mi], DType::F32).unwrap();

    gpu.gemv_mq6g256v2_moe_gate_up_k8_indexed(&ptr_tab, &topk_t, &x_t, &y_g, &y_u, m, k)
        .expect("gate_up launch");
    gpu.hip.device_synchronize().unwrap();
    let got_g = gpu.download_f32(&y_g).unwrap();
    let got_u = gpu.download_f32(&y_u).unwrap();

    let mut want_g = vec![0.0f32; k_top * mi];
    let mut want_u = vec![0.0f32; k_top * mi];
    let mut want_g_swapped = vec![0.0f32; k_top * mi];
    for (r, &e) in topk.iter().enumerate() {
        let blob = &blobs[e as usize];
        for row in 0..mi {
            let wr = dequant_row(blob, row, k, false);
            want_g[r * mi + row] = wr.iter().zip(&x).map(|(a, b)| a * b).sum();
            let wr_sw = dequant_row(blob, row, k, true);
            want_g_swapped[r * mi + row] = wr_sw.iter().zip(&x).map(|(a, b)| a * b).sum();
            let wu = dequant_row(blob, mi + row, k, false);
            want_u[r * mi + row] = wu.iter().zip(&x).map(|(a, b)| a * b).sum();
        }
    }
    rep.check("gate_up y_gate", &got_g, &want_g, 1e-5);
    rep.check("gate_up y_up", &got_u, &want_u, 1e-5);
    rep.check_disagrees(
        "gate_up vs grid-swapped ref (negative control)",
        &got_g,
        &want_g_swapped,
        1e-2,
    );
    println!();
}

fn gate_up_batched_check(gpu: &mut Gpu, rep: &mut Report) {
    // Batched N>1 gate_up — exercises the batched launcher's batch stride and
    // per-token routing. Token-distinct routes + high IDs + batch>1.
    let (mi, k, k_top, n_exp) = (64usize, 512usize, 8usize, 64usize);
    let m = 2 * mi;
    let batch: usize = 4;
    println!("gate_up_batched M={m} (mi={mi}) K={k} k_top={k_top} n_exp={n_exp} batch={batch} (token-distinct, high IDs)");

    let weights: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            let mut w = build_disjoint_halves(m, k);
            for v in w.iter_mut() {
                *v += e as f32 * 0.25;
            }
            w
        })
        .collect();
    let blobs: Vec<Vec<u8>> = weights.iter().map(|w| pack_mq6g256v2(w, m, k)).collect();
    let (_experts, ptr_tab) = upload_experts(gpu, &blobs);

    // Token-distinct routing: each batch lane picks a different 8-pack, with
    // high IDs (48..63) on lane 3 to exercise table offset.
    let mut topk: Vec<i32> = Vec::with_capacity(batch * k_top);
    for b in 0..batch {
        for r in 0..k_top {
            // Lane-dependent offset so no two lanes share the same set.
            let base = match b {
                0 => r as i32,                      // 0..7
                1 => (r as i32 + 3) % n_exp as i32, // 3..10 shifted
                2 => (n_exp as i32 - 8 + r as i32), // high IDs 56..63
                _ => ((r * 7) % n_exp) as i32,      // permuted high coverage
            };
            topk.push(base);
        }
    }
    let topk_b: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_b, &[batch * k_top]).unwrap();

    // Per-token activations distinct so a lane reusing token 0's x is visible.
    let x: Vec<f32> = (0..batch * k)
        .map(|i| prng(i, 0xBEEF_u32.wrapping_add((i / k) as u32 * 0x9E37)) * 2.0 - 1.0)
        .collect();
    let x_t = gpu.upload_f32(&x, &[batch * k]).unwrap();
    let y_g = gpu.alloc_tensor(&[batch * k_top * mi], DType::F32).unwrap();
    let y_u = gpu.alloc_tensor(&[batch * k_top * mi], DType::F32).unwrap();

    gpu.gemv_mq6g256v2_moe_gate_up_k8_indexed_batched(
        &ptr_tab, &topk_t, &x_t, &y_g, &y_u, m, k, k_top, batch,
    )
    .expect("gate_up_batched launch");
    gpu.hip.device_synchronize().unwrap();
    let got_g = gpu.download_f32(&y_g).unwrap();
    let got_u = gpu.download_f32(&y_u).unwrap();

    let mut want_g = vec![0.0f32; batch * k_top * mi];
    let mut want_u = vec![0.0f32; batch * k_top * mi];
    let mut want_g_swapped = vec![0.0f32; batch * k_top * mi];
    for b in 0..batch {
        let xb = &x[b * k..(b + 1) * k];
        for r in 0..k_top {
            let e = topk[b * k_top + r] as usize;
            let blob = &blobs[e];
            for row in 0..mi {
                let wr = dequant_row(blob, row, k, false);
                want_g[(b * k_top + r) * mi + row] = wr.iter().zip(xb).map(|(a, b)| a * b).sum();
                let wr_sw = dequant_row(blob, row, k, true);
                want_g_swapped[(b * k_top + r) * mi + row] =
                    wr_sw.iter().zip(xb).map(|(a, b)| a * b).sum();
                let wu = dequant_row(blob, mi + row, k, false);
                want_u[(b * k_top + r) * mi + row] = wu.iter().zip(xb).map(|(a, b)| a * b).sum();
            }
        }
    }
    rep.check(
        "gate_up_batched y_gate (N=4 distinct)",
        &got_g,
        &want_g,
        1e-5,
    );
    rep.check("gate_up_batched y_up (N=4 distinct)", &got_u, &want_u, 1e-5);
    rep.check_disagrees(
        "gate_up_batched vs grid-swapped (neg)",
        &got_g,
        &want_g_swapped,
        1e-2,
    );
    println!();
}

fn gate_up_batched_production_check(gpu: &mut Gpu, rep: &mut Report) {
    // Production K=2048 (hidden_size 2048) with batched N>1 and n_exp=2 minimal.
    // Two experts is the edge case for table indexing; production K exercises the
    // 8-group loop (K/256 == 8), not the K=512 (2-group) small fixture.
    let (mi, k, k_top, n_exp) = (32usize, 2048usize, 8usize, 2usize);
    let m = 2 * mi;
    let batch: usize = 3;
    println!("gate_up_prod  M={m} (mi={mi}) K={k} k_top={k_top} n_exp={n_exp} batch={batch} (production K, 2-expert, token-distinct)");

    let weights: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            let mut w = build_disjoint_halves(m, k);
            for v in w.iter_mut() {
                *v += e as f32 * 0.5;
            }
            w
        })
        .collect();
    let blobs: Vec<Vec<u8>> = weights.iter().map(|w| pack_mq6g256v2(w, m, k)).collect();
    let (_experts, ptr_tab) = upload_experts(gpu, &blobs);

    // Token-distinct routing across only 2 experts still distinct per lane
    let mut topk: Vec<i32> = Vec::with_capacity(batch * k_top);
    for b in 0..batch {
        for r in 0..k_top {
            // Alternate experts per token and per rank so each lane's set differs
            let e = ((b + r) % n_exp) as i32;
            topk.push(e);
        }
    }
    // Force lane 2 to use the opposite pattern to ensure token-distinct
    for r in 0..k_top {
        topk[2 * k_top + r] = ((r % 2) as i32) ^ 1; // flip
    }
    let topk_b: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_b, &[batch * k_top]).unwrap();

    let x: Vec<f32> = (0..batch * k)
        .map(|i| prng(i, 0xC0DE_u32.wrapping_add((i / k) as u32 * 0x1234)) * 2.0 - 1.0)
        .collect();
    let x_t = gpu.upload_f32(&x, &[batch * k]).unwrap();
    let y_g = gpu.alloc_tensor(&[batch * k_top * mi], DType::F32).unwrap();
    let y_u = gpu.alloc_tensor(&[batch * k_top * mi], DType::F32).unwrap();

    gpu.gemv_mq6g256v2_moe_gate_up_k8_indexed_batched(
        &ptr_tab, &topk_t, &x_t, &y_g, &y_u, m, k, k_top, batch,
    )
    .expect("gate_up_prod launch");
    gpu.hip.device_synchronize().unwrap();
    let got_g = gpu.download_f32(&y_g).unwrap();
    let got_u = gpu.download_f32(&y_u).unwrap();

    let mut want_g = vec![0.0f32; batch * k_top * mi];
    let mut want_u = vec![0.0f32; batch * k_top * mi];
    let mut want_g_swapped = vec![0.0f32; batch * k_top * mi];
    for b in 0..batch {
        let xb = &x[b * k..(b + 1) * k];
        for r in 0..k_top {
            let e = topk[b * k_top + r] as usize;
            let blob = &blobs[e];
            for row in 0..mi {
                let wr = dequant_row(blob, row, k, false);
                want_g[(b * k_top + r) * mi + row] = wr.iter().zip(xb).map(|(a, b)| a * b).sum();
                let wr_sw = dequant_row(blob, row, k, true);
                want_g_swapped[(b * k_top + r) * mi + row] =
                    wr_sw.iter().zip(xb).map(|(a, b)| a * b).sum();
                let wu = dequant_row(blob, mi + row, k, false);
                want_u[(b * k_top + r) * mi + row] = wu.iter().zip(xb).map(|(a, b)| a * b).sum();
            }
        }
    }
    rep.check(
        "gate_up_prod y_gate K=2048 N=3 2-exp",
        &got_g,
        &want_g,
        1e-5,
    );
    rep.check(
        "gate_up_prod y_up   K=2048 N=3 2-exp",
        &got_u,
        &want_u,
        1e-5,
    );
    rep.check_disagrees(
        "gate_up_prod vs grid-swapped (neg)",
        &got_g,
        &want_g_swapped,
        1e-2,
    );
    println!();
}

fn down_check(gpu: &mut Gpu, rep: &mut Report) {
    // down_k = 512 (2 groups) is the shape the expanded kernel is tuned for.
    let (m, k, k_top, n_exp, n) = (64usize, 512usize, 8usize, 8usize, 1usize);
    println!("down     M={m} K={k} k_top={k_top} n_exp={n_exp} batch={n}");

    let weights: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            let mut w = build_disjoint_halves(m, k);
            for v in w.iter_mut() {
                *v += e as f32 * 0.25;
            }
            w
        })
        .collect();
    let blobs: Vec<Vec<u8>> = weights.iter().map(|w| pack_mq6g256v2(w, m, k)).collect();
    let (_experts, ptr_tab) = upload_experts(gpu, &blobs);

    let topk: Vec<i32> = (0..k_top).map(|r| (r % n_exp) as i32).collect();
    let topk_b: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_b, &[k_top]).unwrap();

    // rot_batch is [N × K_TOP × K]; give each rank its own activation so a
    // krank indexing error shows up.
    let rot: Vec<f32> = (0..n * k_top * k)
        .map(|i| prng(i, 0xC0FFEE) * 2.0 - 1.0)
        .collect();
    let rot_t = gpu.upload_f32(&rot, &[n * k_top * k]).unwrap();
    let out_t = gpu.alloc_tensor(&[n * k_top * m], DType::F32).unwrap();

    gpu.gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded(
        &ptr_tab, &topk_t, &rot_t, &out_t, m, k, k_top, n,
    )
    .expect("down launch");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&out_t).unwrap();

    let mut want = vec![0.0f32; n * k_top * m];
    let mut want_swapped = vec![0.0f32; n * k_top * m];
    for (r, &e) in topk.iter().enumerate() {
        let blob = &blobs[e as usize];
        let xr = &rot[r * k..(r + 1) * k];
        for row in 0..m {
            let wr = dequant_row(blob, row, k, false);
            want[r * m + row] = wr.iter().zip(xr).map(|(a, b)| a * b).sum();
            let ws = dequant_row(blob, row, k, true);
            want_swapped[r * m + row] = ws.iter().zip(xr).map(|(a, b)| a * b).sum();
        }
    }
    rep.check("down expert_outputs", &got, &want, 1e-5);
    rep.check_disagrees(
        "down vs grid-swapped ref (negative control)",
        &got,
        &want_swapped,
        1e-2,
    );
    println!();
}

fn down_batched_check(gpu: &mut Gpu, rep: &mut Report) {
    // Batched N>1 down — token-distinct routes and activations, high IDs.
    let (m, k, k_top, n_exp, batch) = (64usize, 512usize, 8usize, 64usize, 4usize);
    println!("down_batched M={m} K={k} k_top={k_top} n_exp={n_exp} batch={batch} (token-distinct, high IDs)");

    let weights: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            let mut w = build_disjoint_halves(m, k);
            for v in w.iter_mut() {
                *v += e as f32 * 0.25;
            }
            w
        })
        .collect();
    let blobs: Vec<Vec<u8>> = weights.iter().map(|w| pack_mq6g256v2(w, m, k)).collect();
    let (_experts, ptr_tab) = upload_experts(gpu, &blobs);

    // Token-distinct topk: each batch lane uses a different expert set, lane 2 uses high IDs.
    let mut topk: Vec<i32> = Vec::with_capacity(batch * k_top);
    for b in 0..batch {
        for r in 0..k_top {
            let e = match b {
                0 => r as i32,
                1 => (r as i32 + 5) % n_exp as i32,
                2 => (n_exp as i32 - 8 + r as i32), // high 56..63
                _ => ((r as i32 * 11 + b as i32 * 3) % n_exp as i32),
            };
            topk.push(e);
        }
    }
    let topk_b: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_b, &[batch * k_top]).unwrap();

    // Per-token per-rank activations distinct — rot layout [batch * k_top * k]
    let rot: Vec<f32> = (0..batch * k_top * k)
        .map(|i| {
            let token = i / (k_top * k);
            prng(i, 0xC0FFEE_u32.wrapping_add(token as u32 * 0x9E37)) * 2.0 - 1.0
        })
        .collect();
    let rot_t = gpu.upload_f32(&rot, &[batch * k_top * k]).unwrap();
    let out_t = gpu.alloc_tensor(&[batch * k_top * m], DType::F32).unwrap();

    gpu.gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded(
        &ptr_tab, &topk_t, &rot_t, &out_t, m, k, k_top, batch,
    )
    .expect("down_batched launch");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&out_t).unwrap();

    let mut want = vec![0.0f32; batch * k_top * m];
    let mut want_swapped = vec![0.0f32; batch * k_top * m];
    for b in 0..batch {
        for r in 0..k_top {
            let e = topk[b * k_top + r] as usize;
            let blob = &blobs[e];
            let xr = &rot[(b * k_top + r) * k..(b * k_top + r + 1) * k];
            for row in 0..m {
                let idx = (b * k_top + r) * m + row;
                let wr = dequant_row(blob, row, k, false);
                want[idx] = wr.iter().zip(xr.iter()).map(|(a, bb)| a * bb).sum();
                let ws = dequant_row(blob, row, k, true);
                want_swapped[idx] = ws.iter().zip(xr.iter()).map(|(a, bb)| a * bb).sum();
            }
        }
    }
    // Recompute correctly without the trick above to avoid double-write confusion
    // (the loop above already filled want correctly; the extra line is harmless
    // because we overwrite with the exact same computation on idx).
    // Keep scoring simple: compare whole flattened output.
    rep.check("down_batched N=4 distinct hi-IDs", &got, &want, 1e-5);
    rep.check_disagrees(
        "down_batched vs grid-swapped (neg)",
        &got,
        &want_swapped,
        1e-2,
    );
    println!();
}

fn down_batched_two_expert_check(gpu: &mut Gpu, rep: &mut Report) {
    // Two-expert edge case with K=512 and production-like M, plus batched.
    let (m, k, k_top, n_exp, batch) = (32usize, 512usize, 8usize, 2usize, 3usize);
    println!("down_2exp  M={m} K={k} k_top={k_top} n_exp={n_exp} batch={batch} (2-expert, N>1)");

    let weights: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            let mut w = build_disjoint_halves(m, k);
            for v in w.iter_mut() {
                *v += e as f32 * 0.7;
            }
            w
        })
        .collect();
    let blobs: Vec<Vec<u8>> = weights.iter().map(|w| pack_mq6g256v2(w, m, k)).collect();
    let (_experts, ptr_tab) = upload_experts(gpu, &blobs);

    let mut topk: Vec<i32> = Vec::with_capacity(batch * k_top);
    for b in 0..batch {
        for r in 0..k_top {
            topk.push(((b + r) % n_exp) as i32);
        }
    }
    let topk_b: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_b, &[batch * k_top]).unwrap();

    let rot: Vec<f32> = (0..batch * k_top * k)
        .map(|i| prng(i, 0xDEAD_u32.wrapping_add((i / k) as u32 * 7)) * 2.0 - 1.0)
        .collect();
    let rot_t = gpu.upload_f32(&rot, &[batch * k_top * k]).unwrap();
    let out_t = gpu.alloc_tensor(&[batch * k_top * m], DType::F32).unwrap();

    gpu.gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded(
        &ptr_tab, &topk_t, &rot_t, &out_t, m, k, k_top, batch,
    )
    .expect("down_2exp launch");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&out_t).unwrap();

    let mut want = vec![0.0f32; batch * k_top * m];
    let mut want_swapped = vec![0.0f32; batch * k_top * m];
    for b in 0..batch {
        for r in 0..k_top {
            let e = topk[b * k_top + r] as usize;
            let blob = &blobs[e];
            let xr = &rot[(b * k_top + r) * k..(b * k_top + r + 1) * k];
            for row in 0..m {
                let idx = (b * k_top + r) * m + row;
                let wr = dequant_row(blob, row, k, false);
                want[idx] = wr.iter().zip(xr).map(|(a, b)| a * b).sum();
                let ws = dequant_row(blob, row, k, true);
                want_swapped[idx] = ws.iter().zip(xr).map(|(a, b)| a * b).sum();
            }
        }
    }
    rep.check("down_2exp N=3", &got, &want, 1e-5);
    rep.check_disagrees("down_2exp vs grid-swapped (neg)", &got, &want_swapped, 1e-2);
    println!();
}

fn down_production_check(gpu: &mut Gpu, rep: &mut Report) {
    // Production down shape: M=512 (or 2048 truncated for speed), K=512.
    // Full M=2048 would be 2048*512; use M=256 to keep runtime low while still
    // covering production K and deterministic 4-row tiling (M%4==0).
    let (m, k, k_top, n_exp, batch) = (256usize, 512usize, 8usize, 8usize, 2usize);
    println!("down_prod  M={m} K={k} k_top={k_top} n_exp={n_exp} batch={batch} (production M, token-distinct)");

    let weights: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            let mut w = build_disjoint_halves(m, k);
            for v in w.iter_mut() {
                *v += e as f32 * 0.25;
            }
            w
        })
        .collect();
    let blobs: Vec<Vec<u8>> = weights.iter().map(|w| pack_mq6g256v2(w, m, k)).collect();
    let (_experts, ptr_tab) = upload_experts(gpu, &blobs);

    let mut topk: Vec<i32> = Vec::with_capacity(batch * k_top);
    for b in 0..batch {
        for r in 0..k_top {
            topk.push(((b * 3 + r * 5) % n_exp) as i32);
        }
    }
    let topk_b: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_b, &[batch * k_top]).unwrap();

    let rot: Vec<f32> = (0..batch * k_top * k)
        .map(|i| {
            prng(
                i,
                0xF00D_u32.wrapping_add((i / (k_top * k)) as u32 * 0x7777),
            ) * 2.0
                - 1.0
        })
        .collect();
    let rot_t = gpu.upload_f32(&rot, &[batch * k_top * k]).unwrap();
    let out_t = gpu.alloc_tensor(&[batch * k_top * m], DType::F32).unwrap();

    gpu.gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded(
        &ptr_tab, &topk_t, &rot_t, &out_t, m, k, k_top, batch,
    )
    .expect("down_prod launch");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&out_t).unwrap();

    let mut want = vec![0.0f32; batch * k_top * m];
    let mut want_swapped = vec![0.0f32; batch * k_top * m];
    for b in 0..batch {
        for r in 0..k_top {
            let e = topk[b * k_top + r] as usize;
            let blob = &blobs[e];
            let xr = &rot[(b * k_top + r) * k..(b * k_top + r + 1) * k];
            for row in 0..m {
                let idx = (b * k_top + r) * m + row;
                let wr = dequant_row(blob, row, k, false);
                want[idx] = wr.iter().zip(xr).map(|(a, b)| a * b).sum();
                let ws = dequant_row(blob, row, k, true);
                want_swapped[idx] = ws.iter().zip(xr).map(|(a, b)| a * b).sum();
            }
        }
    }
    rep.check("down_prod M=256 K=512 N=2", &got, &want, 1e-5);
    rep.check_disagrees("down_prod vs grid-swapped (neg)", &got, &want_swapped, 1e-2);
    println!();
}

fn grouped_check(gpu: &mut Gpu, rep: &mut Report) {
    // THE kernel that has never run on gfx12. The launcher arch-selects, so on
    // an R9700 this exercises `gemm_mq6g256v2_moe_grouped_wmma_gfx12` and on
    // gfx11 the `_k2` source — same contract, same expected numbers.
    //
    // Tolerance is looser than the GEMVs because this kernel dequantizes and
    // accumulates through fp16 WMMA, not f32 (fixture trap 1).
    let (m, k, m_total) = (32usize, 512usize, 16usize);
    println!("grouped  M={m} K={k} m_total={m_total} (arch-selecting launcher)");

    let w = build_disjoint_halves(m, k);
    let blob = pack_mq6g256v2(&w, m, k);
    let (_experts, ptr_tab) = upload_experts(gpu, &[blob.clone()]);

    // One expert owns every tile; identity slot mapping.
    let tile_ids: Vec<i32> = vec![0; m_total / 16];
    let tile_b: Vec<u8> = tile_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tile_t = gpu.upload_raw(&tile_b, &[tile_ids.len()]).unwrap();
    let slots: Vec<i32> = (0..m_total as i32).collect();
    let slot_b: Vec<u8> = slots.iter().flat_map(|v| v.to_le_bytes()).collect();
    let slot_t = gpu.upload_raw(&slot_b, &[m_total]).unwrap();

    let x: Vec<f32> = (0..m_total * k)
        .map(|i| prng(i, 0x5EED) * 2.0 - 1.0)
        .collect();
    let x_t = gpu.upload_f32(&x, &[m_total * k]).unwrap();
    let y_t = gpu.alloc_tensor(&[m_total * m], DType::F32).unwrap();

    gpu.gemm_mq6g256v2_moe_grouped_wmma_k2(
        &ptr_tab, &tile_t, &slot_t, &x_t, &y_t, m, k, 1, m_total, m_total,
    )
    .expect("grouped launch");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&y_t).unwrap();

    // Y_grouped[out_col * M + out_row]
    let mut want = vec![0.0f32; m_total * m];
    let mut want_swapped = vec![0.0f32; m_total * m];
    let rows: Vec<Vec<f32>> = (0..m).map(|r| dequant_row(&blob, r, k, false)).collect();
    let rows_sw: Vec<Vec<f32>> = (0..m).map(|r| dequant_row(&blob, r, k, true)).collect();
    for slot in 0..m_total {
        let xs = &x[slot * k..(slot + 1) * k];
        for row in 0..m {
            want[slot * m + row] = rows[row].iter().zip(xs).map(|(a, b)| a * b).sum();
            want_swapped[slot * m + row] = rows_sw[row].iter().zip(xs).map(|(a, b)| a * b).sum();
        }
    }
    // fp16 accumulation over K=512 with disjoint-magnitude halves: 2e-2 is the
    // honest bar. Tighten only with a measurement that says you can.
    rep.check("grouped Y", &got, &want, 2e-2);
    rep.check_disagrees(
        "grouped vs grid-swapped ref (negative control)",
        &got,
        &want_swapped,
        1e-1,
    );
    println!();
}

fn grouped_nonidentity_check(gpu: &mut Gpu, rep: &mut Report) {
    // Nonidentity permutation — shuffles slot mapping so the kernel cannot
    // assume slot == row. Same tolerance as grouped, same arch-selecting
    // launcher. Catches permutation bugs where the grouped kernel assumes
    // identity.
    let (m, k, m_total) = (32usize, 512usize, 32usize);
    println!("grouped_perm M={m} K={k} m_total={m_total} (nonidentity permutation)");

    let w = build_disjoint_halves(m, k);
    let blob = pack_mq6g256v2(&w, m, k);
    let (_experts, ptr_tab) = upload_experts(gpu, &[blob.clone()]);

    let tile_ids: Vec<i32> = vec![0; m_total / 16];
    let tile_b: Vec<u8> = tile_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tile_t = gpu.upload_raw(&tile_b, &[tile_ids.len()]).unwrap();

    // Nonidentity permutation: reversed blocks of 4 to stay within tile
    // alignment while being nonidentity. Every group of 4 slots is reversed.
    let mut slots: Vec<i32> = Vec::with_capacity(m_total);
    for chunk in (0..m_total).collect::<Vec<_>>().chunks(4) {
        for &s in chunk.iter().rev() {
            slots.push(s as i32);
        }
    }
    assert_ne!(
        slots,
        (0..m_total as i32).collect::<Vec<_>>(),
        "permutation must be nonidentity"
    );
    let slot_b: Vec<u8> = slots.iter().flat_map(|v| v.to_le_bytes()).collect();
    let slot_t = gpu.upload_raw(&slot_b, &[m_total]).unwrap();

    let x: Vec<f32> = (0..m_total * k)
        .map(|i| prng(i, 0xA11C) * 2.0 - 1.0)
        .collect();
    let x_t = gpu.upload_f32(&x, &[m_total * k]).unwrap();
    let y_t = gpu.alloc_tensor(&[m_total * m], DType::F32).unwrap();

    gpu.gemm_mq6g256v2_moe_grouped_wmma_k2(
        &ptr_tab, &tile_t, &slot_t, &x_t, &y_t, m, k, 1, m_total, m_total,
    )
    .expect("grouped_perm launch");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&y_t).unwrap();

    let rows: Vec<Vec<f32>> = (0..m).map(|r| dequant_row(&blob, r, k, false)).collect();
    let rows_sw: Vec<Vec<f32>> = (0..m).map(|r| dequant_row(&blob, r, k, true)).collect();
    let mut want = vec![0.0f32; m_total * m];
    let mut want_swapped = vec![0.0f32; m_total * m];
    for (out_idx, &slot) in slots.iter().enumerate() {
        let xs = &x[slot as usize * k..(slot as usize + 1) * k];
        for row in 0..m {
            want[out_idx * m + row] = rows[row].iter().zip(xs).map(|(a, b)| a * b).sum();
            want_swapped[out_idx * m + row] = rows_sw[row].iter().zip(xs).map(|(a, b)| a * b).sum();
        }
    }
    rep.check("grouped_perm Y (nonidentity)", &got, &want, 2e-2);
    rep.check_disagrees(
        "grouped_perm vs grid-swapped (neg)",
        &got,
        &want_swapped,
        1e-1,
    );
    println!();
}

fn grouped_production_check(gpu: &mut Gpu, rep: &mut Report) {
    // Production K=2048 with larger m_total — still arch-selecting.
    let (m, k, m_total) = (32usize, 2048usize, 16usize);
    println!("grouped_prod M={m} K={k} m_total={m_total} (production K=2048, nonidentity)");

    let w = build_disjoint_halves(m, k);
    let blob = pack_mq6g256v2(&w, m, k);
    let (_experts, ptr_tab) = upload_experts(gpu, &[blob.clone()]);

    let tile_ids: Vec<i32> = vec![0; m_total / 16];
    let tile_b: Vec<u8> = tile_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tile_t = gpu.upload_raw(&tile_b, &[tile_ids.len()]).unwrap();

    // Strided permutation still covering all slots
    let mut slots: Vec<i32> = (0..m_total as i32).collect();
    slots.rotate_left(3);
    let slot_b: Vec<u8> = slots.iter().flat_map(|v| v.to_le_bytes()).collect();
    let slot_t = gpu.upload_raw(&slot_b, &[m_total]).unwrap();

    let x: Vec<f32> = (0..m_total * k)
        .map(|i| prng(i, 0xBEEF2) * 2.0 - 1.0)
        .collect();
    let x_t = gpu.upload_f32(&x, &[m_total * k]).unwrap();
    let y_t = gpu.alloc_tensor(&[m_total * m], DType::F32).unwrap();

    gpu.gemm_mq6g256v2_moe_grouped_wmma_k2(
        &ptr_tab, &tile_t, &slot_t, &x_t, &y_t, m, k, 1, m_total, m_total,
    )
    .expect("grouped_prod launch");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&y_t).unwrap();

    let rows: Vec<Vec<f32>> = (0..m).map(|r| dequant_row(&blob, r, k, false)).collect();
    let rows_sw: Vec<Vec<f32>> = (0..m).map(|r| dequant_row(&blob, r, k, true)).collect();
    let mut want = vec![0.0f32; m_total * m];
    let mut want_swapped = vec![0.0f32; m_total * m];
    for (out_idx, &slot) in slots.iter().enumerate() {
        let xs = &x[slot as usize * k..(slot as usize + 1) * k];
        for row in 0..m {
            want[out_idx * m + row] = rows[row].iter().zip(xs).map(|(a, b)| a * b).sum();
            want_swapped[out_idx * m + row] = rows_sw[row].iter().zip(xs).map(|(a, b)| a * b).sum();
        }
    }
    rep.check("grouped_prod Y K=2048 perm", &got, &want, 2e-2);
    rep.check_disagrees(
        "grouped_prod vs grid-swapped (neg)",
        &got,
        &want_swapped,
        1e-1,
    );
    println!();
}

fn ninepath_check(gpu: &mut Gpu, rep: &mut Report) {
    // The fused down + weighted combine. Requirements from the kernel: down_k
    // == 512 (2 groups), down_m % 16 == 0, blockDim 256 (8 warps = 8 kranks).
    //
    // This kernel folds the k_top partials AND accumulates into the residual,
    // replacing the expanded GEMV + separate combine. So the reference is the
    // whole weighted sum, not a per-rank output.
    let (m, k, k_top, n_exp) = (64usize, 512usize, 8usize, 8usize);
    println!("ninepath M={m} K={k} k_top={k_top} n_exp={n_exp}");

    let weights: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            let mut w = build_disjoint_halves(m, k);
            for v in w.iter_mut() {
                *v += e as f32 * 0.25;
            }
            w
        })
        .collect();
    let blobs: Vec<Vec<u8>> = weights.iter().map(|w| pack_mq6g256v2(w, m, k)).collect();
    let (_experts, ptr_tab) = upload_experts(gpu, &blobs);

    let topk: Vec<i32> = (0..k_top).map(|r| (r % n_exp) as i32).collect();
    let topk_b: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_b, &[k_top]).unwrap();

    // Non-uniform weights so a fold that ignored them, or folded in the wrong
    // order, would show up.
    let tw: Vec<f32> = (0..k_top).map(|i| 0.05 + 0.1 * (i as f32)).collect();
    let tw_t = gpu.upload_f32(&tw, &[k_top]).unwrap();

    let act: Vec<f32> = (0..k_top * k)
        .map(|i| prng(i, 0x9A17) * 2.0 - 1.0)
        .collect();
    let act_t = gpu.upload_f32(&act, &[k_top * k]).unwrap();

    // The kernel accumulates (`out[..] += a`), so start from a known non-zero
    // residual — that also catches a kernel that overwrites instead of adding.
    let out0: Vec<f32> = (0..m).map(|i| prng(i, 0x3C3C) - 0.5).collect();
    let out_t = gpu.upload_f32(&out0, &[m]).unwrap();

    gpu.gemv_mq6g256v2_moe_ninepath_d4(&ptr_tab, &topk_t, &tw_t, &act_t, &out_t, m, k)
        .expect("ninepath launch");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&out_t).unwrap();

    let mut want = out0.clone();
    let mut want_swapped = out0.clone();
    for (r, &e) in topk.iter().enumerate() {
        let blob = &blobs[e as usize];
        let xr = &act[r * k..(r + 1) * k];
        for row in 0..m {
            let wr = dequant_row(blob, row, k, false);
            want[row] += tw[r] * wr.iter().zip(xr).map(|(a, b)| a * b).sum::<f32>();
            let ws = dequant_row(blob, row, k, true);
            want_swapped[row] += tw[r] * ws.iter().zip(xr).map(|(a, b)| a * b).sum::<f32>();
        }
    }
    rep.check("ninepath d4 fused down+combine", &got, &want, 1e-5);
    rep.check_disagrees(
        "ninepath vs grid-swapped ref (negative control)",
        &got,
        &want_swapped,
        1e-2,
    );
    println!();
}

fn dense_f32_rows_check(gpu: &mut Gpu, rep: &mut Report) {
    let m = 129;
    for (rows, k) in [(2, 512), (3, 768), (4, 1024)] {
        let weights = pack_mq6g256v2(&build_disjoint_halves(m, k), m, k);
        let x: Vec<f32> = (0..rows * k).map(|i| prng(i, 0x5412) * 2.0 - 1.0).collect();
        let w = gpu.upload_raw(&weights, &[weights.len()]).unwrap();
        let x = gpu.upload_f32(&x, &[rows * k]).unwrap();
        let batched = gpu
            .upload_f32(&vec![f32::NAN; rows * m], &[rows * m])
            .unwrap();
        gpu.gemm_mq6g256v2_f32_rows(&w, &x, &batched, m, k, rows)
            .expect("MQ6 F32 shared-weight rows");
        gpu.hip.device_synchronize().unwrap();
        let got = gpu.download_f32(&batched).unwrap();
        for row in 0..rows {
            let scalar = gpu.upload_f32(&vec![f32::NAN; m], &[m]).unwrap();
            gpu.gemv_mq6g256v2(&w, &x.sub_offset(row * k, k), &scalar, m, k)
                .expect("MQ6 scalar row");
            gpu.hip.device_synchronize().unwrap();
            let want = gpu.download_f32(&scalar).unwrap();
            for (index, (&got, &want)) in got[row * m..(row + 1) * m].iter().zip(&want).enumerate()
            {
                if got.to_bits() != want.to_bits() {
                    eprintln!(
                        "MQ6 F32 rows={rows} K={k} row={row} output={index}: {got:?} != {want:?}"
                    );
                    rep.failures += 1;
                    return;
                }
            }
        }
    }
    println!("MQ6 F32 shared-weight 2–4 rows: scalar bit parity");
}

fn main() {
    let mut rep = Report { failures: 0 };
    host_self_test(&mut rep);

    match Gpu::init() {
        Ok(mut gpu) => {
            println!("arch={}\n", gpu.arch);
            gate_up_check(&mut gpu, &mut rep);
            dense_f32_rows_check(&mut gpu, &mut rep);
            gate_up_batched_check(&mut gpu, &mut rep);
            gate_up_batched_production_check(&mut gpu, &mut rep);
            down_check(&mut gpu, &mut rep);
            down_batched_check(&mut gpu, &mut rep);
            down_batched_two_expert_check(&mut gpu, &mut rep);
            down_production_check(&mut gpu, &mut rep);
            grouped_check(&mut gpu, &mut rep);
            grouped_nonidentity_check(&mut gpu, &mut rep);
            grouped_production_check(&mut gpu, &mut rep);
            ninepath_check(&mut gpu, &mut rep);
        }
        Err(e) => {
            println!("GPU init failed ({e:?}); host self-test only.");
            println!("The kernel checks did NOT run — this is not a pass.");
            std::process::exit(2);
        }
    }

    if rep.failures == 0 {
        println!("mq6v2_moe_parity: all checks PASSED");
    } else {
        println!("mq6v2_moe_parity: {} check(s) FAILED", rep.failures);
        std::process::exit(1);
    }
}
