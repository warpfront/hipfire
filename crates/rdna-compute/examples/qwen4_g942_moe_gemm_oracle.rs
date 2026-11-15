// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-image byte oracle of the exact-gfx942 wide-lane FP8 grouped MoE GEMM module
//! (`kernels/src/qwen4_g942_moe_gemm.gfx942.hip`, module `qwen4_g942_moe_gemm`, entries
//! `qwen4_g942_moe_gemm_f32` (F32-X) and `qwen4_g942_moe_gemm_xbf16` (BF16-X)).
//!
//! Arms run on the same inputs; the WHOLE `Y` of every launch (poison-filled first, 0x7f guard
//! words on both sides that must survive; rows no live tile covers must keep the poison, pad
//! rows must be +0) must be byte-equal to the REFERENCE:
//!
//! - REFERENCE: the narrow production kernel `gemm_fp8b128_moe_grouped_bf16mfma` (F32-X) /
//!   `gemm_fp8b128_moe_grouped_bf16mfma_xbf16` (BF16-X) of its own hipcc module, launched
//!   directly at every tile count (the public launcher only reaches it below 80 tiles). A
//!   second REFERENCE launch must equal the first;
//! - BASELINE: the retained P1 snapshot, `Gfx942Device::gemm_fp8b128_moe_grouped_bf16mfma_reference`
//!   (the frozen byte baseline the production kernel is gated against);
//! - MODULE: the module entry launched directly through `Gpu::launch_kernel_blob`. The module is
//!   loaded the way the product loads it (`lab_ensure_moe_gemm_entry`): from the hipcc source, or,
//!   with `HIPFIRE_QWEN4_G942_MOE_GEMM_IMAGE=/absolute/device.elf`, from that code object (the
//!   twin proof: same module, same entry names, same kernarg ABI). It runs at EVERY tile count,
//!   also below the 80 the launcher routes to it;
//! - PRODUCT: the public launcher `Gfx942Device::gemm_fp8b128_moe_grouped_bf16mfma` (below 80
//!   tiles the narrow kernel, from 80 the module: the routing itself is under test).
//!
//! Shapes: gate_up R=1280 K=2560 (X row = slot / 10, X = token rows) and down R=2560 K=640
//! (X row = slot) of the Flash-Next stock-FP8 model, 12 experts of the stock block layout
//! (`b128_block_stride`); tile counts 1, 5, 10, 55, 79, 80, 81, 160, 320; X as BF16, as F32
//! holding BF16 values and as arbitrary F32 (the in-kernel RNE: ties, +-0, wide range); FP8
//! codes uniform over every non-NaN code, 0x80 / 0x00 heavy, small, and extreme (0x01, 0x7e, 0xfe,
//! 0x80); `s_inv` ordinary, wide-range and zero. Tile layouts: full padded-16 tiles with
//! pad rows, partial tiles (`nrows` < 16), mixed skip tiles (expert -1), an all-pad case (every
//! live row -1), an all-skip case and a single-expert case.
//!
//! Planted controls the oracle MUST detect (module output differs from the clean reference):
//! one flipped byte in the weight codes, in `s_inv`, in X, in a `sorted_rows` id; a wrong expert
//! id in a tile descriptor; a wrong `nrows`; a wrong tile count; and the OTHER X entry on the
//! same buffers (F32-X on BF16 data, BF16-X on F32 data). Exit 0 only when every case matches
//! and every control is detected. Prints PASS / FAIL counts.
//!
//! Build: `cargo build --release -p rdna-compute --features lab --example qwen4_g942_moe_gemm_oracle`
//! Run:   `qwen4_g942_moe_gemm_oracle [--out FILE.json] [--seed U64] [--quick]`
//! Twin image: `HIPFIRE_QWEN4_G942_MOE_GEMM_IMAGE=/absolute/device.elf qwen4_g942_moe_gemm_oracle`

mod g942_oracle_common;

use g942_oracle_common::*;
use hip_bridge::KernargBlob;
use rdna_compute::fp8_e4m3::b128_block_stride;
use rdna_compute::tensor_ops::{lab_ensure_moe_gemm_entry, lab_ensure_moe_gemm_narrow};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;

const N_EXP: usize = 12;
/// Output rows per workgroup of every entry (`64 * MG_NSUB`, pinned against the sources by the
/// launcher's unit tests).
const ROWS_PER_WG: usize = 64;
const F32_ENTRY: &str = "qwen4_g942_moe_gemm_f32";
const XBF16_ENTRY: &str = "qwen4_g942_moe_gemm_xbf16";

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    r: usize,
    k: usize,
    /// X row = sorted_rows[i] / x_row_div.
    x_row_div: usize,
    /// Flat slot ids are drawn from `0..slot_cap`.
    slot_cap: usize,
}

impl Shape {
    fn x_rows(&self) -> usize {
        self.slot_cap / self.x_row_div
    }
    fn stride(&self) -> usize {
        b128_block_stride(self.r, self.k).expect("block stride")
    }
}

const SHAPES: [Shape; 2] = [
    Shape {
        name: "gate_up",
        r: 1280,
        k: 2560,
        x_row_div: 10,
        slot_cap: 640,
    },
    Shape {
        name: "down",
        r: 2560,
        k: 640,
        x_row_div: 1,
        slot_cap: 640,
    },
];

#[derive(Clone, Copy, PartialEq)]
enum XMode {
    /// BF16 tensor, the BF16-X entry.
    Bf16,
    /// F32 tensor of BF16-exact values.
    F32Exact,
    /// F32 tensor of arbitrary values (the in-kernel RNE is exercised).
    F32Arb,
}

impl XMode {
    const ALL: [XMode; 3] = [XMode::Bf16, XMode::F32Exact, XMode::F32Arb];
    fn name(self) -> &'static str {
        match self {
            XMode::Bf16 => "x_bf16",
            XMode::F32Exact => "x_f32_bf16exact",
            XMode::F32Arb => "x_f32_arbitrary",
        }
    }
    fn bf16(self) -> bool {
        self == XMode::Bf16
    }
}

#[derive(Clone, Copy, PartialEq)]
enum TileMode {
    Full,
    Partial,
    MixedSkip,
    AllPad,
    AllSkip,
    OneExpert,
}

impl TileMode {
    fn name(self) -> &'static str {
        match self {
            TileMode::Full => "full",
            TileMode::Partial => "partial_nrows",
            TileMode::MixedSkip => "mixed_skip",
            TileMode::AllPad => "all_pad",
            TileMode::AllSkip => "all_skip",
            TileMode::OneExpert => "one_expert",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------------------------

fn bf16_bits(value: f32) -> u16 {
    let u = value.to_bits();
    if value.is_nan() {
        return ((u >> 16) | 0x0040) as u16;
    }
    ((u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// One FP8 code byte, never an e4m3fn NaN (0x7f / 0xff) of `family`.
fn code(rng: &mut Rng, family: usize) -> u8 {
    loop {
        let c = match family {
            0 => rng.next() as u8,
            1 => match rng.below(10) {
                0..=2 => 0x80,
                3 | 4 => 0x00,
                _ => rng.next() as u8,
            },
            2 => (rng.next() as u8 & 0x80) | (rng.next() as u8 & 0x1f),
            _ => rng.pick(&[0x01u8, 0x81, 0x7e, 0xfe, 0x00, 0x80, 0x08, 0x88, 0x38, 0xb8]),
        };
        if c & 0x7f != 0x7f {
            return c;
        }
    }
}

/// `s_inv` (bf16 bits) of kind 0 ordinary, 1 wide dynamic range, 2 with zeros.
fn scale(rng: &mut Rng, kind: usize) -> u16 {
    let (lo, span) = match kind {
        1 => (-30i32, 35usize),
        _ => (-8, 9),
    };
    if kind == 2 && rng.below(6) == 0 {
        return if rng.below(2) == 0 { 0x0000 } else { 0x8000 };
    }
    let exp = lo + rng.below(span) as i32;
    bf16_bits(((1.0 + rng.unit()) * (exp as f64).exp2()) as f32)
}

/// `N_EXP` expert blocks of the qt=55 layout: codes `[r, k]`, `s_inv` bf16 `[r/128, k/128]`, zero
/// padding to the stride.
fn weights(rng: &mut Rng, shape: Shape, code_family: usize, scale_kind: usize) -> Vec<u8> {
    let stride = shape.stride();
    let mut bytes = vec![0u8; N_EXP * stride];
    let n_scales = (shape.r / 128) * (shape.k / 128);
    for e in 0..N_EXP {
        let block = &mut bytes[e * stride..(e + 1) * stride];
        let (codes, rest) = block.split_at_mut(shape.r * shape.k);
        for c in codes.iter_mut() {
            *c = code(rng, code_family);
        }
        for s in 0..n_scales {
            rest[2 * s..2 * s + 2].copy_from_slice(&scale(rng, scale_kind).to_le_bytes());
        }
    }
    bytes
}

/// X values `[x_rows, k]` as F32 (`mode`'s content; BF16-exact unless `F32Arb`).
fn x_values(rng: &mut Rng, mode: XMode, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let magnitude = (rng.below(13) as i32 - 6) as f64;
            let v = (rng.gauss() * magnitude.exp2()) as f32;
            let v = match rng.below(40) {
                0 => 0.0,
                1 => -0.0,
                _ => v,
            };
            match mode {
                XMode::F32Arb => match rng.below(4) {
                    // Round-to-nearest-even ties and their neighbours.
                    0 => f32::from_bits((v.to_bits() & 0xffff_0000) | 0x8000),
                    1 => f32::from_bits((v.to_bits() & 0xffff_0000) | 0x7fff),
                    2 => f32::from_bits((v.to_bits() & 0xffff_0000) | 0x8001),
                    _ => v,
                },
                _ => bf16_to_f32(bf16_bits(v)),
            }
        })
        .collect()
}

struct Tiles {
    /// int4 per tile: `{expert, row0, nrows, 0}`.
    desc: Vec<i32>,
    sorted: Vec<i32>,
}

fn gen_tiles(rng: &mut Rng, shape: Shape, n_tiles: usize, mode: TileMode) -> Tiles {
    let mut desc = Vec::with_capacity(n_tiles * 4);
    let mut sorted = Vec::with_capacity(n_tiles * 16);
    let one = rng.below(N_EXP) as i32;
    for t in 0..n_tiles {
        let mut expert = rng.below(N_EXP) as i32;
        let mut nrows = 16usize;
        let mut pad = 0usize;
        match mode {
            TileMode::Full => {
                if rng.below(3) == 0 {
                    pad = rng.below(16);
                }
            }
            TileMode::Partial => {
                if rng.below(2) == 0 {
                    nrows = 1 + rng.below(15);
                }
                if rng.below(4) == 0 {
                    pad = rng.below(nrows);
                }
            }
            TileMode::MixedSkip => {
                if rng.below(5) < 2 {
                    expert = -1;
                } else if rng.below(3) == 0 {
                    pad = rng.below(16);
                }
            }
            TileMode::AllPad => pad = 16,
            TileMode::AllSkip => expert = -1,
            TileMode::OneExpert => expert = one,
        }
        desc.extend_from_slice(&[expert, (16 * t) as i32, nrows as i32, 0]);
        for i in 0..16 {
            // Rows past `nrows` and rows of skipped tiles keep valid ids: never read.
            let padding = i >= nrows.saturating_sub(pad) && i < nrows;
            sorted.push(if padding {
                -1
            } else {
                rng.below(shape.slot_cap) as i32
            });
        }
    }
    Tiles { desc, sorted }
}

// ---------------------------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------------------------

struct Dev<'a> {
    /// The expert blocks, shared across cases and released by their owner.
    w: &'a GpuTensor,
    desc: GpuTensor,
    sorted: GpuTensor,
    x: GpuTensor,
}

impl Dev<'_> {
    fn release(self, gpu: &Gpu) {
        for t in [self.desc, self.sorted, self.x] {
            release(gpu, t);
        }
    }
}

#[derive(Clone, Copy)]
enum Arm {
    /// The narrow production kernel, launched directly.
    Reference,
    /// The P1 snapshot through its public hidden launcher.
    Baseline,
    /// The wide module entry named by the symbol, launched directly.
    Module(&'static str),
    /// The public launcher.
    Product,
}

struct Run {
    shape: Shape,
    n_tiles: usize,
    /// Tile count passed to the launch (differs from `n_tiles` only in a control).
    launch_tiles: usize,
}

fn x_bf16_of(x: &GpuTensor) -> bool {
    x.dtype == DType::BF16
}

fn launch(gpu: &mut Gpu, arm: Arm, dev: &Dev, run: &Run, y: &GpuTensor) -> Res<()> {
    let shape = run.shape;
    let n_sorted = run.n_tiles * 16;
    let stride = shape.stride();
    match arm {
        Arm::Baseline | Arm::Product => {
            let mut device = gpu
                .try_gfx942()
                .ok_or_else(|| "not an exact gfx942 device".to_owned())?;
            let result = if matches!(arm, Arm::Baseline) {
                device.gemm_fp8b128_moe_grouped_bf16mfma_reference(
                    dev.w,
                    stride,
                    N_EXP,
                    &dev.desc,
                    run.launch_tiles,
                    &dev.sorted,
                    &dev.x,
                    shape.x_row_div,
                    shape.x_rows(),
                    y,
                    n_sorted,
                    shape.r,
                    shape.k,
                )
            } else {
                device.gemm_fp8b128_moe_grouped_bf16mfma(
                    dev.w,
                    stride,
                    N_EXP,
                    &dev.desc,
                    run.launch_tiles,
                    &dev.sorted,
                    &dev.x,
                    shape.x_row_div,
                    shape.x_rows(),
                    y,
                    n_sorted,
                    shape.r,
                    shape.k,
                )
            };
            result.map_err(err)
        }
        Arm::Reference | Arm::Module(_) => {
            let symbol = match arm {
                Arm::Module(entry) => lab_ensure_moe_gemm_entry(gpu, entry).map_err(err)?,
                _ => lab_ensure_moe_gemm_narrow(gpu, x_bf16_of(&dev.x)).map_err(err)?,
            };
            let mut args = KernargBlob::new();
            args.push_ptr(dev.w.buf.as_ptr());
            args.push_u64(stride as u64);
            args.push_ptr(dev.desc.buf.as_ptr());
            args.push_ptr(dev.sorted.buf.as_ptr());
            args.push_ptr(dev.x.buf.as_ptr());
            args.push_ptr(y.buf.as_ptr());
            for value in [shape.r, shape.k, shape.x_row_div, N_EXP] {
                args.push_i32(value as i32);
            }
            args.pad_to(16);
            gpu.launch_kernel_blob(
                symbol,
                [
                    shape.r.div_ceil(ROWS_PER_WG) as u32,
                    run.launch_tiles as u32,
                    1,
                ],
                [256, 1, 1],
                0,
                args.as_mut_slice(),
            )
            .map_err(err)
        }
    }
}

/// One launch into a fresh poisoned `Y` of `[n_tiles * 16, r]`: the whole body and the guards.
fn run_arm(gpu: &mut Gpu, arm: Arm, dev: &Dev, run: &Run) -> Res<(Vec<u8>, bool)> {
    let y = Out::poisoned(gpu, run.n_tiles * 16 * run.shape.r)?;
    launch(gpu, arm, dev, run, &y.body)?;
    let result = y.read(gpu)?;
    y.release(gpu);
    Ok(result)
}

fn module_entry(x_bf16: bool) -> &'static str {
    if x_bf16 {
        XBF16_ENTRY
    } else {
        F32_ENTRY
    }
}

/// X as a tensor of `mode`'s dtype from F32 values.
fn upload_x(gpu: &Gpu, mode: XMode, values: &[f32]) -> Res<GpuTensor> {
    match mode {
        XMode::Bf16 => {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|&v| bf16_bits(v).to_le_bytes())
                .collect();
            upload(gpu, &bytes, DType::BF16)
        }
        _ => upload_f32s(gpu, values),
    }
}

fn upload_tiles(gpu: &Gpu, tiles: &Tiles) -> Res<(GpuTensor, GpuTensor)> {
    Ok((
        upload_i32s(gpu, &tiles.desc)?,
        upload_i32s(gpu, &tiles.sorted)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn case(
    gpu: &mut Gpu,
    tally: &mut Tally,
    shape: Shape,
    w: &GpuTensor,
    xmode: XMode,
    n_tiles: usize,
    mode: TileMode,
    rng: &mut Rng,
    label_prefix: &str,
) -> Res<()> {
    let tiles = gen_tiles(rng, shape, n_tiles, mode);
    let x = upload_x(gpu, xmode, &x_values(rng, xmode, shape.x_rows() * shape.k))?;
    let (desc, sorted) = upload_tiles(gpu, &tiles)?;
    let dev = Dev { w, desc, sorted, x };
    let run = Run {
        shape,
        n_tiles,
        launch_tiles: n_tiles,
    };
    let (r, r_guard) = run_arm(gpu, Arm::Reference, &dev, &run)?;
    let (r2, r2_guard) = run_arm(gpu, Arm::Reference, &dev, &run)?;
    let (p1, p1_guard) = run_arm(gpu, Arm::Baseline, &dev, &run)?;
    let entry = module_entry(xmode.bf16());
    let (m, m_guard) = run_arm(gpu, Arm::Module(entry), &dev, &run)?;
    let (p, p_guard) = run_arm(gpu, Arm::Product, &dev, &run)?;
    dev.release(gpu);
    let diffs = vec![
        ("module".to_owned(), diff(&m, &r)),
        ("product".to_owned(), diff(&p, &r)),
        ("baseline_p1".to_owned(), diff(&p1, &r)),
        ("reference_repeat".to_owned(), diff(&r2, &r)),
    ];
    let guards = r_guard && r2_guard && p1_guard && m_guard && p_guard;
    tally.case(
        shape.name,
        &format!(
            "{label_prefix}{} {} tiles={n_tiles} layout={}",
            shape.name,
            xmode.name(),
            mode.name()
        ),
        &diffs,
        guards,
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------------------------

fn make_dev<'a>(gpu: &Gpu, w: &'a GpuTensor, tiles: &Tiles, x: GpuTensor) -> Res<Dev<'a>> {
    let (desc, sorted) = upload_tiles(gpu, tiles)?;
    Ok(Dev { w, desc, sorted, x })
}

fn controls(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally) -> Res<()> {
    let shape = SHAPES[0];
    let n_tiles = 80usize;
    let stride = shape.stride();
    let mut w_bytes = weights(rng, shape, 0, 0);
    let mut tiles = gen_tiles(rng, shape, n_tiles, TileMode::Full);
    // Tile 0: a full live tile of expert e0 whose first row reads X row `xr0`.
    let e0 = 3usize;
    tiles.desc[0] = e0 as i32;
    tiles.desc[2] = 16;
    for i in 0..16 {
        if tiles.sorted[i] < 0 {
            tiles.sorted[i] = rng.below(shape.slot_cap) as i32;
        }
    }
    let xr0 = tiles.sorted[0] as usize / shape.x_row_div;
    // Anchors: W[e0][0][0] = 1.0 (0x38), X[*][0] = 1.0, s_inv[e0][0][0] = 1.0.
    w_bytes[e0 * stride] = 0x38;
    let scale_at = e0 * stride + shape.r * shape.k;
    w_bytes[scale_at..scale_at + 2].copy_from_slice(&bf16_bits(1.0).to_le_bytes());
    let mut x_host = x_values(rng, XMode::F32Exact, shape.x_rows() * shape.k);
    for row in 0..shape.x_rows() {
        x_host[row * shape.k] = 1.0;
    }
    let run = Run {
        shape,
        n_tiles,
        launch_tiles: n_tiles,
    };
    let w = upload(gpu, &w_bytes, DType::Raw)?;
    // Both X dtypes carry the same BF16-exact values.
    for xmode in [XMode::F32Exact, XMode::Bf16] {
        let tag = xmode.name();
        let clean_x = upload_x(gpu, xmode, &x_host)?;
        let clean = make_dev(gpu, &w, &tiles, clean_x)?;
        let entry = module_entry(xmode.bf16());
        let (r, r_guard) = run_arm(gpu, Arm::Reference, &clean, &run)?;
        let (m, m_guard) = run_arm(gpu, Arm::Module(entry), &clean, &run)?;
        if diff(&m, &r) != 0 || !r_guard || !m_guard {
            return Err(format!(
                "gemm controls ({tag}): the clean module run differs from the reference"
            ));
        }
        // Launch-parameter controls on the clean buffers.
        let (got, guard) = run_arm(
            gpu,
            Arm::Module(entry),
            &clean,
            &Run {
                launch_tiles: n_tiles - 1,
                ..run
            },
        )?;
        tally.control(&format!("{tag} tile_count_minus_1"), diff(&got, &r), guard);
        // The other X entry on the same data. BF16 data under the F32 entry is zero-padded to
        // the F32 row size first so the over-read stays inside the allocation.
        let other = module_entry(!xmode.bf16());
        let wrong = if xmode.bf16() {
            let mut bytes: Vec<u8> = x_host
                .iter()
                .flat_map(|&v| bf16_bits(v).to_le_bytes())
                .collect();
            bytes.resize(x_host.len() * 4, 0);
            let padded = upload(gpu, &bytes, DType::BF16)?;
            Some(make_dev(gpu, &w, &tiles, padded)?)
        } else {
            None
        };
        let (got, guard) = run_arm(
            gpu,
            Arm::Module(other),
            wrong.as_ref().unwrap_or(&clean),
            &run,
        )?;
        tally.control(
            &format!("{tag} wrong_entry({other})"),
            diff(&got, &r),
            guard,
        );
        if let Some(wrong) = wrong {
            wrong.release(gpu);
        }
        clean.release(gpu);

        // Operand controls: each corrupts one input, the module runs on it.
        type Mutate = fn(&mut [u8], &mut Tiles, &mut [f32], Shape, usize, usize);
        let mutations: [(&str, Mutate); 7] = [
            ("flip_weight_code_byte", |w, _, _, shape, e0, _| {
                w[e0 * shape.stride()] ^= 0x40
            }),
            ("flip_s_inv_byte", |w, _, _, shape, e0, _| {
                w[e0 * shape.stride() + shape.r * shape.k + 1] ^= 0x10
            }),
            ("flip_x_byte", |_, _, x, shape, _, xr0| {
                let at = xr0 * shape.k;
                x[at] = f32::from_bits(x[at].to_bits() ^ 0x4000_0000)
            }),
            ("flip_sorted_row_id", |_, tiles, _, shape, _, _| {
                tiles.sorted[0] =
                    (tiles.sorted[0] + shape.x_row_div as i32 * 7) % shape.slot_cap as i32
            }),
            ("wrong_expert_id", |_, tiles, _, _, e0, _| {
                tiles.desc[0] = ((e0 + 1) % N_EXP) as i32
            }),
            ("wrong_nrows", |_, tiles, _, _, _, _| tiles.desc[2] = 15),
            ("tile_skipped", |_, tiles, _, _, _, _| tiles.desc[0] = -1),
        ];
        for (name, mutate) in mutations {
            let mut w2 = w_bytes.clone();
            let mut t2 = Tiles {
                desc: tiles.desc.clone(),
                sorted: tiles.sorted.clone(),
            };
            let mut x2 = x_host.clone();
            mutate(&mut w2, &mut t2, &mut x2, shape, e0, xr0);
            let w_dev = upload(gpu, &w2, DType::Raw)?;
            let x_dev = upload_x(gpu, xmode, &x2)?;
            let dev = make_dev(gpu, &w_dev, &t2, x_dev)?;
            let (got, guard) = run_arm(gpu, Arm::Module(entry), &dev, &run)?;
            tally.control(&format!("{tag} {name}"), diff(&got, &r), guard);
            dev.release(gpu);
            release(gpu, w_dev);
        }
    }
    release(gpu, w);
    Ok(())
}

fn main() {
    let args = Args::parse(0x4d47_454d_4d31_0001);
    let mut gpu = init_gfx942();
    println!(
        "{}",
        module_banner("qwen4_g942_moe_gemm", "HIPFIRE_QWEN4_G942_MOE_GEMM_IMAGE")
    );
    let mut rng = Rng(args.seed);
    let mut tally = Tally::default();
    let quick = args.quick;
    let tile_list: &[usize] = if quick {
        &[1, 10, 80, 320]
    } else {
        &[1, 5, 10, 55, 79, 80, 81, 160, 320]
    };
    let w_families = if quick { 1 } else { 3 };
    let ordinary = [TileMode::Full, TileMode::Partial, TileMode::MixedSkip];
    for shape in SHAPES {
        for family in 0..w_families {
            // Code family 0..3, `s_inv` kind 0..2: each (shape, family) pair its own weights.
            let w_bytes = weights(&mut rng, shape, family % 4, family % 3);
            let w = match upload(&gpu, &w_bytes, DType::Raw) {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            };
            let prefix = format!("w{family} ");
            for (xi, xmode) in XMode::ALL.into_iter().enumerate() {
                for (ti, &n_tiles) in tile_list.iter().enumerate() {
                    let mut modes = vec![ordinary[(family + xi + ti) % 3]];
                    if family == 0 {
                        let mut extra = vec![TileMode::Full];
                        if [1, 80, 320].contains(&n_tiles) {
                            extra.extend([
                                TileMode::AllPad,
                                TileMode::AllSkip,
                                TileMode::OneExpert,
                                TileMode::MixedSkip,
                                TileMode::Partial,
                            ]);
                        }
                        for mode in extra {
                            if !modes.contains(&mode) {
                                modes.push(mode);
                            }
                        }
                    }
                    for mode in modes {
                        if let Err(e) = case(
                            &mut gpu, &mut tally, shape, &w, xmode, n_tiles, mode, &mut rng,
                            &prefix,
                        ) {
                            eprintln!("error: {} {}: {e}", shape.name, xmode.name());
                            std::process::exit(1);
                        }
                    }
                }
            }
            release(&gpu, w);
            println!(
                "{} family {family}: cumulative PASS {} FAIL {}",
                shape.name, tally.pass, tally.fail
            );
        }
    }
    if let Err(e) = controls(&mut gpu, &mut rng, &mut tally) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    let code = finish(
        "qwen4_g942_moe_gemm_oracle",
        &args,
        &tally,
        "HIPFIRE_QWEN4_G942_MOE_GEMM_IMAGE",
        json!({"n_experts": N_EXP}),
    );
    std::process::exit(code);
}
