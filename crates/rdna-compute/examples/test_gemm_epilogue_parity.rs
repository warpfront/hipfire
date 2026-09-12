// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity gate for the fused GEMM epilogues (`Gpu::gemm_f16_x_f16_wmma_lds_epi`).
//!
//! Each epilogue must be **bit-identical** to the pass it replaces, because
//! that is the whole claim: folding the GELU / gated-add / add-in / cast into
//! the GEMM store is a pure traffic optimisation, not a numerical change. The
//! reference for every cell is therefore the `EPI = 0` kernel plus the exact
//! elementwise expression the epilogue evaluates, in the same F32 order:
//!
//! | suffix  | reference                                                    |
//! |---------|--------------------------------------------------------------|
//! | `_o16`  | RNE(base)                                                    |
//! | `_o16g` | RNE(`gelu_tanh_f32`(base))                                    |
//! | `_a`    | (acc + addin) + bias                    — CPU, f32           |
//! | `_gr`   | fma(gate, base, residual)               — CPU, `f32::mul_add`|
//! | `_gra`  | fma(gate, (acc + addin) + bias, residual)                    |
//!
//! where `acc` is the `EPI = 0` kernel run with no bias and `base` is
//! `acc + bias` (asserted bit-equal to the bias-fused `EPI = 0` output, which
//! is what licenses computing it on the host).
//!
//! Two reference choices are deliberate:
//!   * **GELU is referenced against the GPU's own `gelu_tanh_f32`**, not a host
//!     `tanhf`. Device and host libm differ in the last ULP, so a host
//!     reference could only ever be a tolerance check; the device kernel is the
//!     pass Task 5 actually deletes, so equality against it is both stricter
//!     and the property that matters. A host GELU is still computed and its
//!     worst ULP distance reported, as a sanity check on the formula itself.
//!   * **The gated combine is an explicit fma on both sides** (`__builtin_fmaf`
//!     in the kernel, `f32::mul_add` here), so the contraction is pinned rather
//!     than left to `-ffp-contract`.
//!
//! Also covered: `residual` aliasing `y` (the in-place gated update Task 5
//! needs), the `mask == 0` delegation to the plain tiled entry, and the two
//! launcher rejections (uninstantiated combination, uninstantiated tile).
//!
//! **Every cell goes through the pitch-aware entry points**
//! (`gemm_f16_x_f16_wmma_lds_epi_ld` and `..._auto_epi_ld`), with
//! `lda = ldx = k` on the packed shapes — so the packed contract is covered by
//! the whole suite — and the `pad/*` shapes at the end run the same sweep with
//! `lda != ldx != k` and poison in the pad. That padded sweep is the only thing
//! that exercises the epilogue kernarg block's pitch slots, which sit AFTER
//! `C, R, Gate` in the kernel signature but BEFORE them in the device function
//! the macro calls; a wrong reconciliation is invisible at `lda = ldx = k`.
//!
//! Shapes are the four FLUX.1-dev hot GEMMs plus the tail suite from
//! `test_gemm_wide_lds_parity` — the hot shapes are exact tile multiples and so
//! never execute an epilogue bounds guard, which is where the staged and direct
//! paths' control flow differs. Tiles are every entry in `Gpu::LDS_EPI_TILES`,
//! not just this arch's winner, so the gate covers what any arch's selector can
//! reach.
//!
//! Run: cargo run --release --features lab --example test_gemm_epilogue_parity -p rdna-compute
//! Exits 0 on pass, 1 on any failure.

use rdna_compute::gemm::{GemmEpilogue, LdsTile};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Deterministic values in roughly [-1, 1). Same LCG as the sibling gates.
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

/// f32 → f16 with round-to-nearest-even, matching the `(_Float16)` conversion
/// the kernel emits. Written out rather than pulled from a crate so the gate
/// has no dependency the library does not already have; `check_rne_helper`
/// verifies it against the device's own cast kernel before it is trusted.
fn f32_to_f16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp_f32 == 0xff {
        // Inf, or a NaN kept quiet and non-zero.
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    if exp_f32 == 0 {
        // f32 zero or subnormal: far below the smallest f16 subnormal.
        return sign;
    }
    let e = exp_f32 - 127;
    if e > 15 {
        return sign | 0x7c00;
    }
    if e >= -14 {
        // Normal f16: keep 10 mantissa bits, round the 13 dropped ones.
        let keep = mant >> 13;
        let round = (mant >> 12) & 1;
        let sticky = (mant & 0xfff) != 0;
        let mut m = keep;
        if round == 1 && (sticky || (keep & 1) == 1) {
            m += 1;
        }
        let mut h_exp = (e + 15) as u32;
        if m == 0x400 {
            m = 0;
            h_exp += 1;
        }
        if h_exp >= 31 {
            return sign | 0x7c00;
        }
        return sign | ((h_exp as u16) << 10) | m as u16;
    }
    // Subnormal f16: restore the implicit 1 and shift it down into place.
    let shift = 13 + (-e - 14) as u32;
    if shift > 24 {
        return sign;
    }
    let m24 = mant | 0x0080_0000;
    let keep = m24 >> shift;
    let round = (m24 >> (shift - 1)) & 1;
    let sticky = (m24 & ((1u32 << (shift - 1)) - 1)) != 0;
    let mut m = keep;
    if round == 1 && (sticky || (keep & 1) == 1) {
        m += 1;
    }
    // A carry out of the mantissa lands in the exponent field, which is the
    // correct encoding for the smallest normal.
    sign | m as u16
}

/// Host GELU-tanh, same expression as `kernels/src/gelu_tanh.hip`. Used only to
/// report the host/device ULP gap, never as the pass/fail reference.
fn gelu_tanh_host(v: f32) -> f32 {
    let inner = 0.7978845608f32 * (v + 0.044715f32 * v * v * v);
    0.5f32 * v * (1.0f32 + inner.tanh())
}

fn upload_f16(gpu: &mut Gpu, host: &[f32], rows: usize, cols: usize) -> GpuTensor {
    let src = gpu.upload_f32(host, &[rows, cols]).expect("upload f32");
    let dst = gpu.zeros(&[rows, cols], DType::F16).expect("alloc f16");
    gpu.cast_f32_to_f16(&src, &dst).expect("cast f32->f16");
    gpu.free_tensor(src).expect("free f32 src");
    dst
}

/// The same data at a row pitch of `cols + pad`, every padding element set to
/// [`PAD_POISON`]. Only the first `cols` of each row are legal to read.
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

fn download_f16_bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u16> {
    let mut out = vec![0u16; t.numel()];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 2) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh f16");
    out
}

struct Shape {
    label: &'static str,
    m: usize,
    k: usize,
    b: usize,
    /// Run the no-bias sweep too. Reserved for the cheap shapes: `has_bias` is
    /// one runtime branch in the epilogue, identical for every tile and shape.
    both_bias: bool,
    /// Extra elements on each row of `A` / `X`, so the epilogue entries are
    /// called as `_epi_ld` with `lda = k + pad_a`, `ldx = k + pad_x`. The
    /// REFERENCE is always built from the packed operands, so a padded shape
    /// asserts exactly what the pitch feature claims: moving where the operands
    /// live changes nothing numerically. Must be multiples of 16 — see
    /// `Gpu::check_lds_pitch`. `pad_a != pad_x` on every padded shape, because
    /// the epilogue entry point's kernarg block appends `lda, ldx` AFTER the
    /// three epilogue pointers while the device function takes them BEFORE, and
    /// equal pitches cannot catch a launcher that swapped or misplaced them.
    pad_a: usize,
    pad_x: usize,
}

impl Shape {
    fn lda(&self) -> usize {
        self.k + self.pad_a
    }
    fn ldx(&self) -> usize {
        self.k + self.pad_x
    }
    fn padded(&self) -> bool {
        self.pad_a != 0 || self.pad_x != 0
    }
}

/// The four FLUX.1-dev shapes from the brief, then the tails.
///
/// B = 4608 = n_img + n_txt is the real single-block batch; 512 is the
/// double-block text stream. All four are exact multiples of every tile, so on
/// their own they never execute an epilogue bounds guard — neither the direct
/// path's `out_m < M` / `out_b >= B` nor, more importantly, the staged path's
/// `continue`/`break`, which sit in a different loop nest and are the only
/// place the two paths' control flow diverges. The tail suite below mirrors
/// `test_gemm_wide_lds_parity`'s: M and/or B off the tile boundary, the
/// 128 < x < 256 band that is a tail for the wide tiles only, B = 1 (the FLUX
/// modulation GEMMs) and M = 64 (`final_layer.linear`). K stays a multiple of
/// 64, which the kernel requires.
const SHAPES: &[Shape] = &[
    Shape {
        label: "single-qkv",
        m: 3072,
        k: 3072,
        b: 4608,
        both_bias: false,
        pad_a: 0,
        pad_x: 0,
    },
    Shape {
        label: "single-mlp-in",
        m: 12288,
        k: 3072,
        b: 4608,
        both_bias: false,
        pad_a: 0,
        pad_x: 0,
    },
    Shape {
        label: "linear2-half",
        m: 3072,
        k: 12288,
        b: 4608,
        both_bias: false,
        pad_a: 0,
        pad_x: 0,
    },
    Shape {
        label: "txt-qkv",
        m: 3072,
        k: 3072,
        b: 512,
        both_bias: true,
        pad_a: 0,
        pad_x: 0,
    },
    // --- tails ---
    Shape {
        label: "tail/M",
        m: 200,
        k: 128,
        b: 256,
        both_bias: true,
        pad_a: 0,
        pad_x: 0,
    },
    Shape {
        label: "tail/B",
        m: 256,
        k: 128,
        b: 100,
        both_bias: true,
        pad_a: 0,
        pad_x: 0,
    },
    Shape {
        label: "tail/both",
        m: 70,
        k: 64,
        b: 13,
        both_bias: true,
        pad_a: 0,
        pad_x: 0,
    },
    Shape {
        label: "tail/mid-band",
        m: 192,
        k: 192,
        b: 160,
        both_bias: true,
        pad_a: 0,
        pad_x: 0,
    },
    Shape {
        label: "tail/B=1 (flux mod)",
        m: 256,
        k: 128,
        b: 1,
        both_bias: true,
        pad_a: 0,
        pad_x: 0,
    },
    Shape {
        label: "tail/M=64 (flux final)",
        m: 64,
        k: 3072,
        b: 512,
        both_bias: true,
        pad_a: 0,
        pad_x: 0,
    },
    Shape {
        label: "tail/M<tile B<tile",
        m: 16,
        k: 64,
        b: 16,
        both_bias: true,
        pad_a: 0,
        pad_x: 0,
    },
    // --- padded row pitch, through `_epi_ld` ---
    //
    // These are the only cells that exercise the epilogue entry points with
    // `lda != k`. That matters more here than in the EPI = 0 gate, because the
    // epilogue kernarg block puts `lda, ldx` AFTER `C, R, Gate` while the
    // device function takes them BEFORE — the two orders are reconciled inside
    // the WLDS_KERNEL_EPI macro, and nothing but a padded epilogue run can tell
    // a correct reconciliation from a wrong one. `pad_a != pad_x` throughout so
    // a swap is caught too, and the pad is poison so an over-read cannot be
    // mistaken for a benign zero.
    //
    // One exact-tile-multiple shape (no epilogue bounds guard) and two tails
    // (both guards), each run against every EPI combination and every EPI tile.
    Shape {
        label: "pad/txt-qkv",
        m: 3072,
        k: 3072,
        b: 512,
        both_bias: true,
        pad_a: 64,
        pad_x: 128,
    },
    Shape {
        label: "pad/tail/M",
        m: 200,
        k: 128,
        b: 256,
        both_bias: true,
        pad_a: 128,
        pad_x: 16,
    },
    Shape {
        label: "pad/tail/both",
        m: 70,
        k: 64,
        b: 13,
        both_bias: true,
        pad_a: 16,
        pad_x: 64,
    },
];

/// Fill value for the padding columns, as in `test_gemm_wide_lds_parity`. Large
/// enough that one contribution swamps the whole legitimate dot product, so a
/// kernel that reads past K fails by orders of magnitude rather than drifting.
const PAD_POISON: f32 = 1.0e3;

#[derive(Clone, Copy, PartialEq)]
enum Combo {
    OutF16,
    OutF16Gelu,
    Gated,
    GatedAddin,
    Addin,
}

impl Combo {
    const ALL: [Combo; 5] = [
        Combo::OutF16,
        Combo::OutF16Gelu,
        Combo::Gated,
        Combo::GatedAddin,
        Combo::Addin,
    ];
    fn suffix(self) -> &'static str {
        match self {
            Combo::OutF16 => "_o16",
            Combo::OutF16Gelu => "_o16g",
            Combo::Gated => "_gr",
            Combo::GatedAddin => "_gra",
            Combo::Addin => "_a",
        }
    }
    fn out_f16(self) -> bool {
        matches!(self, Combo::OutF16 | Combo::OutF16Gelu)
    }
}

/// Everything the references for one (shape, with_bias) cell need.
struct Refs {
    /// `EPI = 0` accumulator, no bias — the device's raw `Σ_k A·X`.
    acc: Vec<f32>,
    /// `acc + bias` when `with_bias`, else `acc`. Bit-equal to the bias-fused
    /// `EPI = 0` output (asserted).
    base: Vec<f32>,
    /// Device `gelu_tanh_f32(base)`.
    gelu: Vec<f32>,
}

struct Operands {
    a: GpuTensor,
    x: GpuTensor,
    /// The same data at the shape's padded pitch, with [`PAD_POISON`] in the
    /// pad. `None` for an unpadded shape, where the packed operands are what
    /// the entry points get. The packed pair is ALWAYS what builds the
    /// reference, so a padded shape compares padded-input epilogue output
    /// against packed-input `EPI = 0` output.
    a_pad: Option<GpuTensor>,
    x_pad: Option<GpuTensor>,
    bias: GpuTensor,
    gate: GpuTensor,
    addin: GpuTensor,
    resid: GpuTensor,
    bias_host: Vec<f32>,
    gate_host: Vec<f32>,
    addin_host: Vec<f32>,
    resid_host: Vec<f32>,
}

impl Operands {
    /// What the epilogue entry points are called with: the padded copies when
    /// the shape asks for a pitch, the packed ones otherwise.
    fn a_in(&self) -> &GpuTensor {
        self.a_pad.as_ref().unwrap_or(&self.a)
    }
    fn x_in(&self) -> &GpuTensor {
        self.x_pad.as_ref().unwrap_or(&self.x)
    }
}

fn make_operands(gpu: &mut Gpu, s: &Shape) -> Operands {
    let a_host = pseudo_random(s.m * s.k, 0xA5A5_0000 ^ s.m as u64);
    let x_host = pseudo_random(s.b * s.k, 0x5A5A_0000 ^ s.b as u64);
    let bias_host = pseudo_random(s.m, 0xB1A5_0000 ^ s.k as u64);
    let gate_host = pseudo_random(s.m, 0x6A7E_0000 ^ s.m as u64);
    let addin_host = pseudo_random(s.b * s.m, 0xADD1_0000 ^ s.k as u64);
    let resid_host = pseudo_random(s.b * s.m, 0x8E51_0000 ^ s.b as u64);
    Operands {
        a: upload_f16(gpu, &a_host, s.m, s.k),
        x: upload_f16(gpu, &x_host, s.b, s.k),
        a_pad: (s.pad_a != 0 || s.pad_x != 0)
            .then(|| upload_f16_padded(gpu, &a_host, s.m, s.k, s.pad_a)),
        x_pad: (s.pad_a != 0 || s.pad_x != 0)
            .then(|| upload_f16_padded(gpu, &x_host, s.b, s.k, s.pad_x)),
        bias: gpu.upload_f32(&bias_host, &[s.m]).expect("upload bias"),
        gate: gpu.upload_f32(&gate_host, &[s.m]).expect("upload gate"),
        addin: gpu
            .upload_f32(&addin_host, &[s.b, s.m])
            .expect("upload addin"),
        resid: gpu
            .upload_f32(&resid_host, &[s.b, s.m])
            .expect("upload resid"),
        bias_host,
        gate_host,
        addin_host,
        resid_host,
    }
}

/// Build the references with the `EPI = 0` kernel and the standalone
/// elementwise kernels — the exact passes the fused epilogue replaces.
fn build_refs(gpu: &mut Gpu, s: &Shape, op: &Operands, with_bias: bool, tile: LdsTile) -> Refs {
    let y = gpu.zeros(&[s.b, s.m], DType::F32).expect("alloc y0");

    gpu.gemm_f16_x_f16_wmma_lds_tiled(&op.a, &op.x, &y, None, s.m, s.k, s.b, tile)
        .expect("EPI=0 gemm, no bias");
    gpu.hip.device_synchronize().expect("sync");
    let acc = gpu.download_f32(&y).expect("dl acc");

    // `base` on the host, then checked against the bias-fused kernel: the
    // epilogue adds the bias with the same single f32 add, so if these two
    // agree bitwise the host may stand in for the device everywhere below.
    let base: Vec<f32> = if with_bias {
        acc.iter()
            .enumerate()
            .map(|(i, &v)| v + op.bias_host[i % s.m])
            .collect()
    } else {
        acc.clone()
    };
    let bias_arg = if with_bias { Some(&op.bias) } else { None };
    gpu.gemm_f16_x_f16_wmma_lds_tiled(&op.a, &op.x, &y, bias_arg, s.m, s.k, s.b, tile)
        .expect("EPI=0 gemm, bias");
    gpu.hip.device_synchronize().expect("sync");
    let device_base = gpu.download_f32(&y).expect("dl base");
    let mismatch = device_base
        .iter()
        .zip(&base)
        .position(|(d, h)| d.to_bits() != h.to_bits());
    assert!(
        mismatch.is_none(),
        "host `acc + bias` disagrees with the fused-bias EPI=0 kernel at flat {} \
         — the host reference chain is invalid",
        mismatch.unwrap()
    );

    // GELU reference: the device kernel, in place on `base`.
    gpu.gelu_tanh_f32(&y, &y, s.b * s.m).expect("gelu ref");
    gpu.hip.device_synchronize().expect("sync");
    let gelu = gpu.download_f32(&y).expect("dl gelu");

    let _ = gpu.free_tensor(y);
    Refs { acc, base, gelu }
}

/// Expected f32 value at flat index `i` (row `b`, column `m = i % M`), in the
/// kernel's evaluation order: `acc`, then the add-in, then the bias, then the
/// gate. `bias` here is the same `bv` the kernel uses — `Bias[m]` or 0.
fn expect_f32(combo: Combo, i: usize, m_stride: usize, r: &Refs, op: &Operands, bias: f32) -> f32 {
    let m = i % m_stride;
    match combo {
        Combo::OutF16 => r.base[i],
        Combo::OutF16Gelu => r.gelu[i],
        Combo::Addin => (r.acc[i] + op.addin_host[i]) + bias,
        Combo::Gated => op.gate_host[m].mul_add(r.base[i], op.resid_host[i]),
        Combo::GatedAddin => {
            let v = (r.acc[i] + op.addin_host[i]) + bias;
            op.gate_host[m].mul_add(v, op.resid_host[i])
        }
    }
}

fn main() {
    eprintln!("=== test_gemm_epilogue_parity ===");
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("  arch = {arch}");
    if !arch.starts_with("gfx11") {
        eprintln!("  SKIPPED: gfx11 wave32 WMMA layout required, got {arch}");
        std::process::exit(0);
    }

    let mut fails = 0usize;
    let mut cells = 0usize;
    let mut worst_gelu_ulp = 0u32;

    fails += check_launcher_rejections(&mut gpu);
    fails += check_rne_helper(&mut gpu);

    for s in SHAPES {
        eprintln!(
            "\n--- {} (M={} K={} B={} lda={} ldx={}{}) ---",
            s.label,
            s.m,
            s.k,
            s.b,
            s.lda(),
            s.ldx(),
            if s.padded() { ", poison pad" } else { "" }
        );
        let op = make_operands(&mut gpu, s);
        let bias_settings: &[bool] = if s.both_bias { &[true, false] } else { &[true] };

        for &with_bias in bias_settings {
            // One reference set for every tile: the `EPI = 0` entries are
            // bit-exact across tiles (that is what test_gemm_wide_lds_parity
            // asserts), so a per-tile epilogue that disagrees with the tile-0
            // reference is a real defect either way.
            let refs = build_refs(&mut gpu, s, &op, with_bias, Gpu::LDS_EPI_TILES[0]);
            for &tile in Gpu::LDS_EPI_TILES {
                for combo in Combo::ALL {
                    cells += 1;
                    let bad = run_cell(&mut gpu, s, &op, &refs, tile, combo, with_bias);
                    if bad > 0 {
                        fails += 1;
                    }
                }
            }
            // GELU formula sanity: how far the host expression lands from the
            // device kernel. Reported, never asserted.
            let ulp = refs
                .gelu
                .iter()
                .zip(&refs.base)
                .map(|(&d, &b)| {
                    let h = gelu_tanh_host(b);
                    if d.is_sign_negative() != h.is_sign_negative() {
                        return u32::MAX;
                    }
                    (d.to_bits() as i64 - h.to_bits() as i64).unsigned_abs() as u32
                })
                .max()
                .unwrap_or(0);
            worst_gelu_ulp = worst_gelu_ulp.max(ulp);

            // In-place gated update: residual IS y.
            cells += 1;
            if check_aliased_gated(&mut gpu, s, &op, &refs, Gpu::LDS_EPI_TILES[0], with_bias) > 0 {
                fails += 1;
            }
            // mask == 0 must reach the plain tiled entry, bit-exact.
            cells += 1;
            if check_empty_epilogue(&mut gpu, s, &op, &refs, Gpu::LDS_EPI_TILES[0], with_bias) > 0 {
                fails += 1;
            }
            // The auto path — what Task 5 calls — must land on an instantiated
            // tile on this arch and produce the same bits.
            cells += 1;
            if check_auto(&mut gpu, s, &op, &refs, with_bias) > 0 {
                fails += 1;
            }
        }

        for t in [op.a, op.x, op.bias, op.gate, op.addin, op.resid] {
            let _ = gpu.free_tensor(t);
        }
        for t in [op.a_pad, op.x_pad].into_iter().flatten() {
            let _ = gpu.free_tensor(t);
        }
    }

    eprintln!(
        "\nGELU: worst host-vs-device distance {worst_gelu_ulp} ULP (reported only; \
         the device kernel is the reference)"
    );
    eprintln!();
    if fails == 0 {
        eprintln!("ALL PASS ({cells} cells)");
        std::process::exit(0);
    }
    eprintln!("{fails} FAILED of {cells} cells");
    std::process::exit(1);
}

/// One (shape, tile, combo, bias) cell. Returns the number of mismatches.
fn run_cell(
    gpu: &mut Gpu,
    s: &Shape,
    op: &Operands,
    refs: &Refs,
    tile: LdsTile,
    combo: Combo,
    with_bias: bool,
) -> usize {
    let dtype = if combo.out_f16() {
        DType::F16
    } else {
        DType::F32
    };
    let y = gpu.zeros(&[s.b, s.m], dtype).expect("alloc y");
    let epi = GemmEpilogue {
        out_f16: combo.out_f16(),
        gelu: combo == Combo::OutF16Gelu,
        addin: matches!(combo, Combo::Addin | Combo::GatedAddin).then_some(&op.addin),
        gate: matches!(combo, Combo::Gated | Combo::GatedAddin).then_some(&op.gate),
        residual: matches!(combo, Combo::Gated | Combo::GatedAddin).then_some(&op.resid),
    };
    let bias_arg = if with_bias { Some(&op.bias) } else { None };
    gpu.gemm_f16_x_f16_wmma_lds_epi_ld(
        op.a_in(),
        op.x_in(),
        &y,
        bias_arg,
        s.m,
        s.k,
        s.b,
        tile,
        &epi,
        s.lda(),
        s.ldx(),
    )
    .expect("epilogue gemm");
    gpu.hip.device_synchronize().expect("sync");

    let n = s.b * s.m;
    let mut bad = 0usize;
    let mut first = usize::MAX;
    let bias_at = |i: usize| {
        if with_bias {
            op.bias_host[i % s.m]
        } else {
            0.0
        }
    };
    if combo.out_f16() {
        let got = download_f16_bits(gpu, &y);
        for i in 0..n {
            let want = f32_to_f16_rne(expect_f32(combo, i, s.m, refs, op, bias_at(i)));
            if got[i] != want {
                bad += 1;
                first = first.min(i);
            }
        }
    } else {
        let got = gpu.download_f32(&y).expect("dl y");
        for i in 0..n {
            let want = expect_f32(combo, i, s.m, refs, op, bias_at(i));
            if got[i].to_bits() != want.to_bits() {
                bad += 1;
                first = first.min(i);
            }
        }
    }
    let _ = gpu.free_tensor(y);

    eprintln!(
        "  {}  {}{:<6} bias={:<5} {:<16} {}",
        if bad == 0 { "PASS" } else { "FAIL" },
        tile.label(),
        combo.suffix(),
        with_bias,
        format!("{n} elems"),
        if bad == 0 {
            "bit-exact".to_string()
        } else {
            format!("{bad} mismatches, first at flat {first}")
        }
    );
    bad
}

/// `residual` aliasing `y`: seed `y` with the residual, run the gated
/// epilogue in place, and require the same bits as the out-of-place run.
fn check_aliased_gated(
    gpu: &mut Gpu,
    s: &Shape,
    op: &Operands,
    refs: &Refs,
    tile: LdsTile,
    with_bias: bool,
) -> usize {
    let y = gpu
        .upload_f32(&op.resid_host, &[s.b, s.m])
        .expect("seed y with residual");
    let epi = GemmEpilogue {
        gate: Some(&op.gate),
        residual: Some(&y),
        ..Default::default()
    };
    let bias_arg = if with_bias { Some(&op.bias) } else { None };
    gpu.gemm_f16_x_f16_wmma_lds_epi_ld(
        op.a_in(),
        op.x_in(),
        &y,
        bias_arg,
        s.m,
        s.k,
        s.b,
        tile,
        &epi,
        s.lda(),
        s.ldx(),
    )
    .expect("aliased gated gemm");
    gpu.hip.device_synchronize().expect("sync");
    let got = gpu.download_f32(&y).expect("dl y");
    let mut bad = 0usize;
    let mut first = usize::MAX;
    for i in 0..s.b * s.m {
        let want = op.gate_host[i % s.m].mul_add(refs.base[i], op.resid_host[i]);
        if got[i].to_bits() != want.to_bits() {
            bad += 1;
            first = first.min(i);
        }
    }
    let _ = gpu.free_tensor(y);
    eprintln!(
        "  {}  {}_gr    residual ALIASES y (in-place) {}",
        if bad == 0 { "PASS" } else { "FAIL" },
        tile.label(),
        if bad == 0 {
            "bit-exact".to_string()
        } else {
            format!("{bad} mismatches, first at flat {first}")
        }
    );
    bad
}

/// `gemm_f16_x_f16_wmma_lds_auto_epi` must pick an instantiated tile on this
/// arch (no error) and give the same bits as the explicit call.
fn check_auto(gpu: &mut Gpu, s: &Shape, op: &Operands, refs: &Refs, with_bias: bool) -> usize {
    let y = gpu.zeros(&[s.b, s.m], DType::F16).expect("alloc y");
    let epi = GemmEpilogue {
        out_f16: true,
        gelu: true,
        ..Default::default()
    };
    let bias_arg = if with_bias { Some(&op.bias) } else { None };
    // `_auto_epi_ld` is what a pitch-aware caller uses; on an unpadded shape it
    // passes `lda = ldx = k` and is the same call as `_auto_epi`.
    if let Err(e) = gpu.gemm_f16_x_f16_wmma_lds_auto_epi_ld(
        op.a_in(),
        op.x_in(),
        &y,
        bias_arg,
        s.m,
        s.k,
        s.b,
        &epi,
        s.lda(),
        s.ldx(),
    ) {
        eprintln!("  FAIL  auto_epi_ld rejected its own tile choice: {e}");
        let _ = gpu.free_tensor(y);
        return 1;
    }
    gpu.hip.device_synchronize().expect("sync");
    let got = download_f16_bits(gpu, &y);
    let bad = got
        .iter()
        .zip(&refs.gelu)
        .filter(|(&g, &w)| g != f32_to_f16_rne(w))
        .count();
    let _ = gpu.free_tensor(y);
    eprintln!(
        "  {}  auto_epi_ld{:<6} _o16g  {}",
        if bad == 0 { "PASS" } else { "FAIL" },
        "",
        if bad == 0 {
            "bit-exact".to_string()
        } else {
            format!("{bad} mismatches")
        }
    );
    bad
}

/// An all-default epilogue must land on the plain `EPI = 0` entry.
fn check_empty_epilogue(
    gpu: &mut Gpu,
    s: &Shape,
    op: &Operands,
    refs: &Refs,
    tile: LdsTile,
    with_bias: bool,
) -> usize {
    let y = gpu.zeros(&[s.b, s.m], DType::F32).expect("alloc y");
    let bias_arg = if with_bias { Some(&op.bias) } else { None };
    gpu.gemm_f16_x_f16_wmma_lds_epi_ld(
        op.a_in(),
        op.x_in(),
        &y,
        bias_arg,
        s.m,
        s.k,
        s.b,
        tile,
        &GemmEpilogue::default(),
        s.lda(),
        s.ldx(),
    )
    .expect("empty epilogue");
    gpu.hip.device_synchronize().expect("sync");
    let got = gpu.download_f32(&y).expect("dl y");
    let bad = got
        .iter()
        .zip(&refs.base)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count();
    let _ = gpu.free_tensor(y);
    eprintln!(
        "  {}  {}       empty epilogue -> EPI=0 entry",
        if bad == 0 { "PASS" } else { "FAIL" },
        tile.label()
    );
    bad
}

/// The launcher must reject what is not instantiated, by name, instead of
/// launching something else.
fn check_launcher_rejections(gpu: &mut Gpu) -> usize {
    let (m, k, b) = (256usize, 64usize, 256usize);
    let a = gpu.zeros(&[m, k], DType::F16).expect("a");
    let x = gpu.zeros(&[b, k], DType::F16).expect("x");
    let y = gpu.zeros(&[b, m], DType::F32).expect("y");
    let gate = gpu.zeros(&[m], DType::F32).expect("gate");
    let tile = Gpu::LDS_EPI_TILES[0];
    let mut fails = 0;

    // GELU alone: a real mask with no instantiation.
    let epi = GemmEpilogue {
        gelu: true,
        ..Default::default()
    };
    match gpu.gemm_f16_x_f16_wmma_lds_epi(&a, &x, &y, None, m, k, b, tile, &epi) {
        Err(e) if e.to_string().contains(&tile.entry()) => {
            eprintln!("  PASS  uninstantiated combination rejected: {e}");
        }
        other => {
            fails += 1;
            eprintln!("  FAIL  GELU-only should have been rejected by name, got {other:?}");
        }
    }

    // A tile that exists as EPI = 0 but has no epilogue instantiation.
    let bare = LdsTile::new(128, 512, 32, 64, 32, false);
    assert!(Gpu::LDS_TILE_VARIANTS.contains(&bare) && !Gpu::LDS_EPI_TILES.contains(&bare));
    let epi = GemmEpilogue {
        out_f16: false,
        gate: Some(&gate),
        residual: Some(&y),
        ..Default::default()
    };
    match gpu.gemm_f16_x_f16_wmma_lds_epi(&a, &x, &y, None, m, k, b, bare, &epi) {
        Err(e) if e.to_string().contains(&bare.entry_epi("_gr")) => {
            eprintln!("  PASS  uninstantiated tile rejected: {e}");
        }
        other => {
            fails += 1;
            eprintln!(
                "  FAIL  tile {} should have been rejected, got {other:?}",
                bare.label()
            );
        }
    }

    // Half a gated epilogue is a caller bug, not a silent no-gate launch.
    let epi = GemmEpilogue {
        gate: Some(&gate),
        ..Default::default()
    };
    match gpu.gemm_f16_x_f16_wmma_lds_epi(&a, &x, &y, None, m, k, b, tile, &epi) {
        Err(e) if e.to_string().contains("residual") => {
            eprintln!("  PASS  gate-without-residual rejected: {e}");
        }
        other => {
            fails += 1;
            eprintln!("  FAIL  gate without residual should have been rejected, got {other:?}");
        }
    }

    for t in [a, x, y, gate] {
        let _ = gpu.free_tensor(t);
    }
    fails
}

/// Validate the host RNE helper against the device's own `(_Float16)` cast
/// before any f16 cell trusts it.
fn check_rne_helper(gpu: &mut Gpu) -> usize {
    let mut host: Vec<f32> = pseudo_random(1 << 16, 0xC0DE_0001);
    // Push some values onto the rounding boundaries the random draw misses.
    for (i, extra) in [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        65504.0,
        -65504.0,
        1.0009765625,  // exactly representable
        1.00048828125, // half-way: rounds to even
        1.0014648437,  // half-way up
        6.0e-8,        // below the smallest subnormal
        6.104e-5,      // smallest normal
        3.0e-5,        // subnormal
        1.0e-7,
        123456.0, // overflows f16
    ]
    .iter()
    .enumerate()
    {
        host[i] = *extra;
    }
    let src = gpu.upload_f32(&host, &[host.len()]).expect("upload");
    let dst = gpu.zeros(&[host.len()], DType::F16).expect("alloc f16");
    gpu.cast_f32_to_f16(&src, &dst).expect("device cast");
    gpu.hip.device_synchronize().expect("sync");
    let device = download_f16_bits(gpu, &dst);
    let mut bad = 0usize;
    for (i, &v) in host.iter().enumerate() {
        // NaN payloads are not compared; none are generated here.
        if device[i] != f32_to_f16_rne(v) {
            if bad < 4 {
                eprintln!(
                    "  FAIL  RNE helper: {v:e} -> host {:#06x}, device {:#06x}",
                    f32_to_f16_rne(v),
                    device[i]
                );
            }
            bad += 1;
        }
    }
    let _ = gpu.free_tensor(src);
    let _ = gpu.free_tensor(dst);
    if bad == 0 {
        eprintln!(
            "  PASS  host RNE helper matches the device cast over {} values",
            host.len()
        );
        0
    } else {
        eprintln!("  FAIL  host RNE helper disagrees on {bad} values");
        1
    }
}
