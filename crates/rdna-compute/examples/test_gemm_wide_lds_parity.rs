// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity test: every tile in `Gpu::LDS_TILE_VARIANTS` (the parameterised
//! `gemm_wmma_lds_*` family) vs `gemm_f16_x_f16_wmma` (LDS 0, 16×16 per wave)
//! + a separate `bias_add_f32` pass. The baseline kernel is the reference,
//! exactly as in `test_gemm_f16_x_f16_wmma_lds_parity`.
//!
//! Why the tolerance is tight: every kernel here walks K in ascending steps of
//! 16 and issues one `wmma_f32_16x16x16_f16_w32` per step into an F32
//! accumulator. The tiled kernels stage 32 or 64 K-elements at a time but still
//! consume them kt = 0,1,… in order, so the **accumulation order along K is
//! identical** to the baseline's for every tile. Only the bias add differs
//! (fused in the epilogue vs a separate f32 pass). Results must agree
//! bit-exactly. A nonzero delta means a real indexing or staging bug — fix the
//! kernel, do NOT widen the bound.
//!
//! Suites mirror the 128×128 gate, with the tile boundaries moved out to 512:
//!   1. Aligned shapes — multiples of the tile and of the K stage, including
//!      the three FLUX hot shapes at their real B = 4608.
//!   2. Tail shapes — M and/or B not a multiple of the tile, including B = 1
//!      (FLUX modulation GEMMs), M = 64 (FLUX `final_layer.linear`), and the
//!      128 < x < 256 band that is a tail for the wide tiles but was aligned
//!      for the narrow one.
//!   3. Bias — every shape is run with and without a non-zero bias.
//!   4. Padded row pitch — `lda`/`ldx` wider than K, with poison in the pad.
//!      Bit-exact against the packed reference is the whole point: the pitch
//!      changes where the operands live in DRAM (worth 1.25-1.65×, see the ROW
//!      PITCH note in the kernel source) and must change nothing numerically.
//!
//! Also checks `Gpu::lds_tile_for` (the per-arch auto-selector) without a GPU.
//!
//! Run: cargo run --release --features lab --example test_gemm_wide_lds_parity -p rdna-compute
//! Exits 0 on pass, 1 on any failure.

use rdna_compute::gemm::LdsTile;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Max relative error accepted. The kernels share K-accumulation order, so
/// anything above this is a bug, not float noise.
const TOL_REL: f32 = 1e-5;

/// Deterministic values in roughly [-1, 1). An LCG keeps the test reproducible
/// without pulling in a rand dependency.
fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// Upload f32 host data and convert it to an F16 device tensor.
fn upload_f16(gpu: &mut Gpu, host: &[f32], rows: usize, cols: usize) -> GpuTensor {
    let src = gpu.upload_f32(host, &[rows, cols]).expect("upload f32");
    let dst = gpu.zeros(&[rows, cols], DType::F16).expect("alloc f16");
    gpu.cast_f32_to_f16(&src, &dst).expect("cast f32->f16");
    gpu.free_tensor(src).expect("free f32 src");
    dst
}

/// The same data at a row pitch of `cols + pad`, with every padding element set
/// to [`PAD_POISON`]. Only the first `cols` of each row are legal to read.
fn upload_f16_padded(
    gpu: &mut Gpu,
    host: &[f32],
    rows: usize,
    cols: usize,
    pad: usize,
) -> GpuTensor {
    let ld = cols + pad;
    let mut wide = vec![PAD_POISON; rows * ld];
    for r in 0..rows {
        wide[r * ld..r * ld + cols].copy_from_slice(&host[r * cols..(r + 1) * cols]);
    }
    upload_f16(gpu, &wide, rows, ld)
}

struct Case {
    label: &'static str,
    m: usize,
    k: usize,
    b: usize,
    /// Extra elements on each row of `A` / `X`, i.e. `lda = k + pad_a` and
    /// `ldx = k + pad_x`. The padding is filled with `PAD_POISON`, so a kernel
    /// that reads past `k` in a row produces a result orders of magnitude off
    /// and the case fails loudly instead of drifting. `0` is the packed
    /// contract — `lda = ldx = k`, exactly what `gemm_f16_x_f16_wmma_lds_tiled`
    /// passes. Padded and packed must both be bit-exact against the same packed
    /// reference: the pitch moves where the operands live and nothing else.
    pad_a: usize,
    pad_x: usize,
}

/// Fill value for the padding columns. Large enough that one contribution
/// swamps the whole legitimate dot product (`|A|, |X| < 1`, `K <= 15360`), so
/// an over-read cannot hide inside the tolerance.
const PAD_POISON: f32 = 1.0e3;

const CASES: &[Case] = &[
    // Suite 1 — aligned to the 256×256 macro-tile.
    Case {
        label: "aligned/one-tile",
        m: 256,
        k: 64,
        b: 256,
        pad_a: 0,
        pad_x: 0,
    },
    Case {
        label: "aligned/small",
        m: 512,
        k: 256,
        b: 512,
        pad_a: 0,
        pad_x: 0,
    },
    Case {
        label: "aligned/flux-txt-qkv",
        m: 3072,
        k: 3072,
        b: 512,
        pad_a: 0,
        pad_x: 0,
    },
    // The three shapes that carry 67 % of a FLUX.1-dev step's GEMM FLOPs, at
    // their real B = n_img + n_txt = 4608. The gate must cover them at full
    // size, not a reduced B, because they are what the A/B bench times.
    Case {
        label: "hot/single-qkv",
        m: 3072,
        k: 3072,
        b: 4608,
        pad_a: 0,
        pad_x: 0,
    },
    Case {
        label: "hot/single-mlp-in",
        m: 12288,
        k: 3072,
        b: 4608,
        pad_a: 0,
        pad_x: 0,
    },
    Case {
        label: "hot/single-linear2",
        m: 3072,
        k: 15360,
        b: 4608,
        pad_a: 0,
        pad_x: 0,
    },
    // Suite 2 — tails against the 256 boundary.
    Case {
        label: "tail/M",
        m: 200,
        k: 128,
        b: 256,
        pad_a: 0,
        pad_x: 0,
    },
    Case {
        label: "tail/B",
        m: 256,
        k: 128,
        b: 100,
        pad_a: 0,
        pad_x: 0,
    },
    Case {
        label: "tail/both",
        m: 70,
        k: 64,
        b: 13,
        pad_a: 0,
        pad_x: 0,
    },
    // 128 < x < 256: aligned for the old tile, a tail for the new one. This is
    // the band the wider macro-tile newly has to clamp, so it is the case most
    // likely to expose a staging bug.
    Case {
        label: "tail/mid-band",
        m: 192,
        k: 192,
        b: 160,
        pad_a: 0,
        pad_x: 0,
    },
    Case {
        label: "tail/B=1 (flux mod)",
        m: 256,
        k: 128,
        b: 1,
        pad_a: 0,
        pad_x: 0,
    },
    Case {
        label: "tail/M=64 (flux final)",
        m: 64,
        k: 3072,
        b: 512,
        pad_a: 0,
        pad_x: 0,
    },
    Case {
        label: "tail/M<tile B<tile",
        m: 16,
        k: 64,
        b: 16,
        pad_a: 0,
        pad_x: 0,
    },
    // Suite 4 — padded row pitch (`gemm_f16_x_f16_wmma_lds_tiled_ld`). The pad
    // is poison, so these fail loudly if a tile reads past K in a row; the
    // result must still be bit-exact against the packed reference, since the
    // padding moves where the operands live and nothing else. `lda` and `ldx`
    // are exercised together and separately so a launcher that swapped them
    // could not pass. Every pad is a multiple of 16 because the pitch
    // precondition requires it; that rule is itself covered by
    // `check_pitch_rejections` below, not by a case here.
    Case {
        label: "pad/both",
        m: 512,
        k: 256,
        b: 512,
        pad_a: 64,
        pad_x: 64,
    },
    Case {
        label: "pad/A only",
        m: 512,
        k: 256,
        b: 512,
        pad_a: 128,
        pad_x: 0,
    },
    Case {
        label: "pad/X only",
        m: 512,
        k: 256,
        b: 512,
        pad_a: 0,
        pad_x: 192,
    },
    // The smallest legal pad, on a shape whose M and B are both tails, with
    // lda != ldx so a launcher that swapped them fails here. 16 elements is
    // the granularity floor: the staging loads are 32-byte `half16` vectors,
    // so the pitch must stay a multiple of 16 (asserted in
    // `Gpu::check_lds_pitch`) — a pad of 7 or 23 is REJECTED, not slow.
    Case {
        label: "pad/tails, min pad",
        m: 200,
        k: 128,
        b: 100,
        pad_a: 16,
        pad_x: 32,
    },
    // The pad the FLUX K values actually want (K % 512 == 0 is the camping
    // case), at a shape big enough to cover a multi-block grid.
    Case {
        label: "pad/flux-txt-qkv +64",
        m: 3072,
        k: 3072,
        b: 512,
        pad_a: 64,
        pad_x: 64,
    },
];

/// Run one shape against every tile variant and report the per-tile max
/// relative error. The reference is computed once and reused across tiles —
/// it does not depend on the tiling, and at the hot shapes the baseline kernel
/// plus a 226 MB readback is by far the slowest part of the gate.
fn run_case(gpu: &mut Gpu, c: &Case, with_bias: bool) -> Vec<(bool, f32)> {
    let a_host = pseudo_random(c.m * c.k, 0xA5A5_0000 ^ c.m as u64);
    let x_host = pseudo_random(c.b * c.k, 0x5A5A_0000 ^ c.b as u64);
    let a = upload_f16(gpu, &a_host, c.m, c.k);
    let x = upload_f16(gpu, &x_host, c.b, c.k);

    let bias_host = pseudo_random(c.m, 0xB1A5_0000 ^ c.k as u64);
    let bias = gpu.upload_f32(&bias_host, &[c.m]).expect("upload bias");

    // Reference: baseline kernel, then the separate bias pass the caller
    // currently performs.
    let y_ref = gpu.zeros(&[c.b, c.m], DType::F32).expect("alloc y_ref");
    gpu.gemm_f16_x_f16_wmma(&a, &x, &y_ref, c.m, c.k, c.b)
        .expect("baseline gemm");
    if with_bias {
        gpu.bias_add_f32(&y_ref, &bias, c.b, c.m).expect("bias_add");
    }
    gpu.hip.device_synchronize().expect("sync ref");
    let r = gpu.download_f32(&y_ref).expect("dl ref");

    // The padded operands, when the case asks for them. `a`/`x` stay packed so
    // the reference above is the packed contract in both arms.
    let padded = c.pad_a != 0 || c.pad_x != 0;
    let a_p = padded.then(|| upload_f16_padded(gpu, &a_host, c.m, c.k, c.pad_a));
    let x_p = padded.then(|| upload_f16_padded(gpu, &x_host, c.b, c.k, c.pad_x));

    let y_new = gpu.zeros(&[c.b, c.m], DType::F32).expect("alloc y_new");
    let mut out = Vec::with_capacity(Gpu::LDS_TILE_VARIANTS.len());

    for &tile in Gpu::LDS_TILE_VARIANTS {
        let bias_arg = if with_bias { Some(&bias) } else { None };
        gpu.gemm_f16_x_f16_wmma_lds_tiled_ld(
            a_p.as_ref().unwrap_or(&a),
            x_p.as_ref().unwrap_or(&x),
            &y_new,
            bias_arg,
            c.m,
            c.k,
            c.b,
            tile,
            c.k + c.pad_a,
            c.k + c.pad_x,
        )
        .expect("wide lds gemm");
        gpu.hip.device_synchronize().expect("sync");
        let n = gpu.download_f32(&y_new).expect("dl new");

        let mut max_rel = 0.0f32;
        let mut max_at = 0usize;
        let mut exact = true;
        for i in 0..r.len() {
            if r[i].to_bits() != n[i].to_bits() {
                exact = false;
            }
            let d = (r[i] - n[i]).abs();
            // Guard the denominator so a near-zero reference cannot manufacture
            // a huge ratio out of a tiny absolute difference.
            let rel = d / r[i].abs().max(1e-3);
            if rel > max_rel {
                max_rel = rel;
                max_at = i;
            }
        }
        if !exact {
            eprintln!(
                "      {} NOT bit-exact; worst at flat {max_at}: ref={} new={}",
                tile.label(),
                r[max_at],
                n[max_at]
            );
        }
        out.push((exact, max_rel));
    }

    for t in [a, x, bias, y_ref, y_new] {
        let _ = gpu.free_tensor(t);
    }
    for t in [a_p, x_p].into_iter().flatten() {
        let _ = gpu.free_tensor(t);
    }
    out
}

fn check_selector() -> usize {
    // (arch, M, B, cu_count) -> expected tile. The preference chain is measured
    // per arch (see Gpu::lds_tile_for); a tile is skipped when it is wider than
    // the operand it tiles, or when the grid would give fewer than cu_count/2
    // workgroups.
    let expect: &[(&str, usize, usize, usize, LdsTile)] = &[
        // FLUX hot shapes each take their arch's measured winner.
        (
            "gfx1150",
            3072,
            4608,
            16,
            LdsTile::new(128, 256, 32, 64, 64, false),
        ),
        (
            "gfx1151",
            3072,
            4608,
            40,
            LdsTile::new(256, 256, 64, 64, 64, false),
        ),
        (
            "gfx1100",
            3072,
            4608,
            96,
            LdsTile::new(128, 256, 64, 64, 32, false),
        ),
        (
            "gfx1100",
            12288,
            4608,
            96,
            LdsTile::new(128, 256, 64, 64, 32, false),
        ),
        // double/txt_qkv, B = 512: still wide enough for every arch's first
        // choice (gfx1151: 12 x 2 = 24 >= 20).
        (
            "gfx1151",
            3072,
            512,
            40,
            LdsTile::new(256, 256, 64, 64, 64, false),
        ),
        (
            "gfx1150",
            3072,
            512,
            16,
            LdsTile::new(128, 256, 32, 64, 64, false),
        ),
        // Degenerate B = 1 (FLUX modulation GEMMs): every tile is wider than B,
        // so the fallback — the shipped 128x128 / 32x64 tiling — is used.
        (
            "gfx1150",
            3072,
            1,
            16,
            LdsTile::new(128, 128, 32, 64, 64, false),
        ),
        (
            "gfx1100",
            3072,
            1,
            96,
            LdsTile::new(128, 128, 32, 64, 64, false),
        ),
        // M = 64 (FLUX final_layer.linear) is narrower than every bm.
        (
            "gfx1151",
            64,
            4608,
            40,
            LdsTile::new(128, 128, 32, 64, 64, false),
        ),
        // An unmeasured arch gets the tile that is positive on all three.
        (
            "gfx1201",
            3072,
            4608,
            64,
            LdsTile::new(128, 256, 32, 64, 64, false),
        ),
    ];
    // The main-loop form is decided after the tile, by Gpu::lds_pipe_gate, and
    // HIPFIRE_FLUX_GEMM_PIPE overrides the per-arch default. Print what this
    // process would actually dispatch, so a run with the env var set shows the
    // flag reaching the dispatch rather than being silently inert.
    eprintln!("  pipeline gate (HIPFIRE_FLUX_GEMM_PIPE as set for this process):");
    for arch in ["gfx1150", "gfx1151", "gfx1100", "gfx1201"] {
        let tile = Gpu::lds_tile_for(arch, 3072, 4608, 40);
        eprintln!(
            "    {arch}: {} -> {}  (arch default {})",
            tile.label(),
            Gpu::lds_pipe_gate(arch, tile).entry(),
            Gpu::lds_pipe_default(arch)
        );
    }

    let mut fails = 0;
    for &(arch, m, b, cu, want) in expect {
        let got = Gpu::lds_tile_for(arch, m, b, cu);
        let ok = got == want;
        if !ok {
            fails += 1;
        }
        eprintln!(
            "  {}  lds_tile_for({arch}, M={m}, B={b}, cu={cu}) = {}, want {}",
            if ok { "PASS" } else { "FAIL" },
            got.label(),
            want.label()
        );
    }
    fails
}

/// The three pitch preconditions must be *rejected*, not tolerated. None is
/// detectable by the kernel: a too-narrow pitch folds the next row into the dot
/// product, a pitch that is not a multiple of 16 misaligns the 32-byte staging
/// loads, and a pitch the allocation cannot back reads past the buffer. Each
/// would return plausible wrong numbers, which is exactly the failure mode the
/// rest of this gate cannot see.
fn check_pitch_rejections(gpu: &mut Gpu) -> usize {
    let (m, k, b) = (256usize, 64usize, 256usize);
    let tile = Gpu::LDS_TILE_VARIANTS[0];
    let a = upload_f16(gpu, &pseudo_random(m * k, 1), m, k);
    let x = upload_f16(gpu, &pseudo_random(b * k, 2), b, k);
    let y = gpu.zeros(&[b, m], DType::F32).expect("alloc y");

    // (label, lda, ldx) — each must panic. The operands above are packed, so
    // the last case is a legal pitch with an operand too short to back it.
    let bad: [(&str, usize, usize); 4] = [
        ("lda < K", k - 16, k),
        ("ldx < K", k, k - 16),
        ("lda % 16 != 0", k + 8, k),
        ("lda past the end of A", k + 16, k),
    ];

    let mut fails = 0usize;
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    for (label, lda, ldx) in bad {
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = gpu.gemm_f16_x_f16_wmma_lds_tiled_ld(&a, &x, &y, None, m, k, b, tile, lda, ldx);
        }))
        .is_err();
        if !caught {
            fails += 1;
        }
        eprintln!(
            "  {}  rejects {label} (lda={lda}, ldx={ldx}, K={k})",
            if caught { "PASS" } else { "FAIL" }
        );
    }
    std::panic::set_hook(hook);

    for t in [a, x, y] {
        let _ = gpu.free_tensor(t);
    }
    fails
}

fn main() {
    eprintln!("=== test_gemm_wide_lds_parity ===");
    eprintln!("\n--- tile selector (no GPU) ---");
    let mut fails = check_selector();

    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("\n  arch = {arch}");
    eprintln!("  tol  = {TOL_REL:e} relative (expect bit-exact)");
    if !arch.starts_with("gfx11") {
        eprintln!("  SKIPPED: gfx11 wave32 WMMA layout required, got {arch}");
        std::process::exit(if fails == 0 { 0 } else { 1 });
    }

    eprintln!("\n--- pitch preconditions ---");
    fails += check_pitch_rejections(&mut gpu);

    let mut cases = 0usize;
    let mut inexact = 0usize;
    for with_bias in [false, true] {
        eprintln!("\n--- bias = {with_bias} ---");
        for c in CASES {
            let per_tile = run_case(&mut gpu, c, with_bias);
            let mut worst = 0.0f32;
            let mut bad = 0usize;
            for (ti, &(exact, max_rel)) in per_tile.iter().enumerate() {
                cases += 1;
                if max_rel > TOL_REL {
                    fails += 1;
                    bad += 1;
                }
                if !exact {
                    inexact += 1;
                    eprintln!("      inexact on {}", Gpu::LDS_TILE_VARIANTS[ti].label());
                }
                worst = worst.max(max_rel);
            }
            eprintln!(
                "  {}  {:<24} M={:<6} K={:<6} B={:<5} lda={:<6} ldx={:<6} max_rel={worst:.3e} \
                 over {} tiles",
                if bad == 0 { "PASS" } else { "FAIL" },
                c.label,
                c.m,
                c.k,
                c.b,
                c.k + c.pad_a,
                c.k + c.pad_x,
                per_tile.len()
            );
        }
    }
    eprintln!(
        "\nbit-exact: {}/{cases} tile×shape×bias cells",
        cases - inexact
    );

    eprintln!();
    if fails == 0 {
        eprintln!("ALL PASS ({cases} GPU cases + selector + pitch preconditions)");
        std::process::exit(0);
    }
    eprintln!("{fails} FAILED");
    std::process::exit(1);
}
