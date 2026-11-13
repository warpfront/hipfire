// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-image byte oracle of the exact-gfx942 row-grid module
//! (`kernels/src/qwen4_g942_rows16_grid.gfx942.hip`, module `qwen4_g942_rows16_grid`, entries
//! `qwen4_g942_rows16_grid` and `qwen4_g942_rows16_grid_x2`).
//!
//! The entries must produce, word for word, what `gemm_bf16_mfma_gfx942_rows16` produces per
//! 16-row slice (same kernarg ABI, the rows16 body run per `blockIdx.y`). Four arms run on the same
//! inputs; every output of every launch (with 0x7f guard words on both sides that must survive)
//! must be byte-equal:
//!
//! - REFERENCE: `Gpu::gemm_bf16_mfma_gfx942_rows16`, once per 16-row slice of `x` and of every
//!   `y_j` (the product's former route: the rows16 chain), last slice partial;
//! - MODULE: the module entries launched directly with the product's kernarg blob, grid
//!   `[sum_j ceil(m_j / 16), ceil(rows / 16)]`, block 576. The module is loaded the way the product
//!   loads it (`lab_ensure_rows16_grid_entry`): from the hipcc source, or, with
//!   `HIPFIRE_QWEN4_G942_ROWS16_GRID_IMAGE=/absolute/device.elf`, from that code object (the twin
//!   proof). The entry the product picks (`qwen4_g942_rows16_grid_x2` for an F32 `x` of more than
//!   eight rows, else `qwen4_g942_rows16_grid`) is launched on every case; the `_x2` entry,
//!   which takes any row count per slice, is also launched wherever the product picks the other
//!   one (a BF16 `x`, or an F32 `x` of at most eight rows);
//! - PRODUCT: `Gpu::gemm_bf16_mfma_gfx942_rows16_grid`, and, on the shapes the row grid admits
//!   (a BF16 `x` of 17..=256 rows, one narrow long-K member: `K >= 8192`, at most 32 16-row weight
//!   tiles), the public routing caller `Gpu::gemm_bf16_mfma_gfx942_rows16_chain` (`chain` and
//!   `wide` set), which routes there.
//!
//! A second REFERENCE run must equal the first. Shapes: the Flash-Next HC down projection
//! (`M = 320`, `K = 10240`) at 1 / 2 / 7 / 8 / 9 / 15 / 16 rows and the row-grid boundaries 17,
//! 31, 32, 33, 48, 63, 64, 65, 100, 128, 200, 255, 256, for an F32 and a BF16 `x` and for F32 /
//! BF16 outputs; one to four members of mixed `M` (320 / 48 / 17 / 33 / 16 / 4 / 1) with F32,
//! BF16 and mixed outputs; and `K` 8 / 136 / 264 / 640 / 1032 / 2560 (a `K` that is not a multiple
//! of 128 and an odd multiple of 128 + 8 included).
//!
//! Planted controls the oracle MUST detect (module output differs from the clean reference): one
//! flipped byte in `x`, in member 0's weight and in member 1's weight; the wrong row count
//! (`rows - 1`); swapped operands (weights of members 0 and 1, output pointers of members 0 and 1);
//! the wrong entry (the `_x2` case run on the non-`_x2` entry, an F32 `x` of 40 rows); the wrong `K`
//! (`K - 8`); and the wrong output-dtype mask. Exit 0 only when every case matches and every
//! control is detected. Prints PASS / FAIL counts.
//!
//! `qwen4_g942_rows16_grid_oracle [--out FILE.json] [--seed U64] [--quick]`
//!
//! `--bench [--reps N] [--out FILE.json]` instead times the HC down projection (`M = 320`,
//! `K = 10240`, BF16 `x`) at 17 / 32 / 64 / 128 / 256 rows through the row grid, the rows16 chain,
//! the wide64 entries (`gemm_bf16_mfma_gfx942_wide_slices`) and the public router, with hipEvent
//! pairs (median of `N >= 20`, default 25; warm weights, and cold weights: a 1 GiB memset before
//! every rep), one CSV row (`BENCH,...`) and one JSON row per cell, each arm's bytes checked
//! against the rows16 chain; then the 257-row decline (`DECLINE,...`): the full router leaves the
//! row grid (the last launch is not a row-grid entry) and a router with no fallback behind the
//! grid (`chain` only, a 48-row weight) launches at 256 rows and declines at 257 with the output
//! untouched. These cells are the evidence behind `ROW_GRID_MAX_ROWS`.

use std::ffi::c_void;

use hip_bridge::{DeviceBuffer, KernargBlob};
use rdna_compute::tensor_ops::lab_ensure_rows16_grid_entry;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;

type Res<T> = Result<T, String>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

const ENTRY: &str = "qwen4_g942_rows16_grid";
const ENTRY_X2: &str = "qwen4_g942_rows16_grid_x2";
const GUARD: usize = 64;
const MAX_MEMBERS: usize = 4;
/// LDS row pitch (BF16 elements) of the rows16 body: `R16_KC + 4`.
const LDS_PITCH: usize = 260;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A tensor of `n` `dtype` elements holding exactly `bytes` (`bytes.len() == n * dtype.size()`).
fn upload(gpu: &Gpu, bytes: &[u8], dtype: DType, n: usize) -> Res<GpuTensor> {
    assert_eq!(bytes.len(), n * dtype.size());
    let mut t = gpu.upload_raw(bytes, &[bytes.len()]).map_err(err)?;
    t.dtype = dtype;
    t.shape = vec![n];
    Ok(t)
}

/// An output tensor with 0x7f guards around a `[n]` body view.
struct Guarded {
    full: GpuTensor,
    body: GpuTensor,
    n: usize,
}

impl Guarded {
    fn new(gpu: &mut Gpu, n: usize, dtype: DType) -> Res<Self> {
        let full = gpu.alloc_tensor(&[n + 2 * GUARD], dtype).map_err(err)?;
        gpu.hip
            .memset(&full.buf, 0x7f, full.byte_size())
            .map_err(err)?;
        let body = full.sub_offset(GUARD, n);
        Ok(Guarded { full, body, n })
    }
    /// The body bytes, and whether both guard bands survived.
    fn read(&self, gpu: &Gpu) -> Res<(Vec<u8>, bool)> {
        gpu.hip.device_synchronize().map_err(err)?;
        let bytes = gpu.download_raw_bytes(&self.full).map_err(err)?;
        let esz = self.full.dtype.size();
        let (g, n) = (GUARD * esz, self.n * esz);
        let ok = bytes[..g].iter().all(|&b| b == 0x7f) && bytes[g + n..].iter().all(|&b| b == 0x7f);
        Ok((bytes[g..g + n].to_vec(), ok))
    }
}

/// The bodies of one arm's outputs and whether every guard band survived.
type Arm = (Vec<Vec<u8>>, bool);

fn diff(a: &Arm, b: &Arm) -> usize {
    let mut n = a.0.len().abs_diff(b.0.len());
    for (x, y) in a.0.iter().zip(&b.0) {
        n += x.iter().zip(y).filter(|(p, q)| p != q).count() + x.len().abs_diff(y.len());
    }
    n
}

#[derive(Default)]
struct Tally {
    pass: usize,
    fail: usize,
    detected: usize,
    missed: usize,
    rows: Vec<serde_json::Value>,
}

impl Tally {
    /// Every arm of one case against the reference (`r`); `r2` is the repeated reference.
    fn case(&mut self, label: &str, r: &Arm, r2: &Arm, arms: &[(&str, &Arm)]) {
        let d_r2 = diff(r2, r);
        let mut guards = r.1 && r2.1;
        let mut diffs = vec![("reference_repeat", d_r2)];
        for &(name, arm) in arms {
            guards &= arm.1;
            diffs.push((name, diff(arm, r)));
        }
        let ok = guards && diffs.iter().all(|&(_, d)| d == 0);
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
            eprintln!("FAIL {label}: guards_ok={guards} {diffs:?}");
        }
        if !ok || self.rows.len() < 8 {
            let by_arm: serde_json::Map<String, serde_json::Value> = diffs
                .iter()
                .map(|&(name, d)| (name.to_owned(), json!(d)))
                .collect();
            self.rows.push(json!({
                "case": label, "ok": ok, "guards_ok": guards, "differing_bytes_vs_reference": by_arm,
            }));
        }
    }

    /// A planted control: `got` MUST differ from the clean reference.
    fn control(&mut self, name: &str, got: &Arm, reference: &Arm) {
        let differing = diff(got, reference);
        if differing > 0 {
            self.detected += 1;
        } else {
            self.missed += 1;
            eprintln!("CONTROL MISSED {name}: no differing byte");
        }
        self.rows.push(
            json!({"control": name, "differing_bytes": differing, "detected": differing > 0}),
        );
    }
}

#[derive(Clone, Debug)]
struct Case {
    /// Weight rows of each member.
    ms: Vec<usize>,
    k: usize,
    rows: usize,
    x_f32: bool,
    /// Bit j set: member j's output is BF16.
    mask: u32,
}

impl Case {
    fn label(&self) -> String {
        format!(
            "rows16_grid ms={:?} k={} rows={} x={} d_mask={:#06b}",
            self.ms,
            self.k,
            self.rows,
            if self.x_f32 { "f32" } else { "bf16" },
            self.mask
        )
    }
    fn d_bf16(&self, j: usize) -> bool {
        self.mask >> j & 1 != 0
    }
    /// The entry the product picks.
    fn product_entry(&self) -> &'static str {
        if self.x_f32 && self.rows > 8 {
            ENTRY_X2
        } else {
            ENTRY
        }
    }
    /// The row-grid shape `Gpu::gemm_bf16_mfma_gfx942_rows16_chain` routes to the row grid
    /// (`row_grid_admitted`, BF16 `x`).
    fn routed_by_chain(&self) -> bool {
        !self.x_f32
            && self.rows > 16
            && self.rows <= 256
            && self.k >= 8192
            && self.ms.len() == 1
            && self.ms[0].div_ceil(16) <= 32
    }
}

/// Host operands: `x` as F32 values (a BF16 `x` takes their top 16 bits), the BF16 weight bits.
struct Host {
    x: Vec<f32>,
    a: Vec<Vec<u16>>,
}

fn host(rng: &mut Rng, case: &Case) -> Host {
    Host {
        x: (0..case.rows * case.k).map(|_| rng.f32() * 2.0).collect(),
        a: case.ms.iter().map(|&m| weights(rng, m * case.k)).collect(),
    }
}

fn weights(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| ((rng.f32() * 0.5).to_bits() >> 16) as u16)
        .collect()
}

fn clone_host(h: &Host) -> Host {
    Host {
        x: h.x.clone(),
        a: h.a.clone(),
    }
}

struct Dev {
    x: GpuTensor,
    a: Vec<GpuTensor>,
}

fn upload_x(gpu: &Gpu, x: &[f32], x_f32: bool) -> Res<GpuTensor> {
    if x_f32 {
        upload(gpu, &f32_bytes(x), DType::F32, x.len())
    } else {
        let bits: Vec<u16> = x.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        upload(gpu, &u16_bytes(&bits), DType::BF16, x.len())
    }
}

fn upload_a(gpu: &Gpu, a: &[Vec<u16>]) -> Res<Vec<GpuTensor>> {
    a.iter()
        .map(|w| upload(gpu, &u16_bytes(w), DType::BF16, w.len()))
        .collect()
}

fn dev(gpu: &Gpu, case: &Case, h: &Host) -> Res<Dev> {
    Ok(Dev {
        x: upload_x(gpu, &h.x, case.x_f32)?,
        a: upload_a(gpu, &h.a)?,
    })
}

struct Out {
    ys: Vec<Guarded>,
}

impl Out {
    fn new(gpu: &mut Gpu, case: &Case) -> Res<Self> {
        let mut ys = Vec::new();
        for (j, &m) in case.ms.iter().enumerate() {
            let dtype = if case.d_bf16(j) {
                DType::BF16
            } else {
                DType::F32
            };
            ys.push(Guarded::new(gpu, case.rows * m, dtype)?);
        }
        Ok(Out { ys })
    }
    fn read(&self, gpu: &Gpu) -> Res<Arm> {
        let mut parts = Vec::new();
        let mut guards = true;
        for y in &self.ys {
            let (bytes, ok) = y.read(gpu)?;
            parts.push(bytes);
            guards &= ok;
        }
        Ok((parts, guards))
    }
}

/// The rows16 chain: `gemm_bf16_mfma_gfx942_rows16` per 16-row slice of `x` and of every `y_j`.
fn reference(gpu: &mut Gpu, case: &Case, dev: &Dev, out: &Out) -> Res<()> {
    for row in (0..case.rows).step_by(16) {
        let sub = (case.rows - row).min(16);
        let xs = dev.x.sub_offset(row * case.k, sub * case.k);
        let ys: Vec<GpuTensor> = out
            .ys
            .iter()
            .zip(&case.ms)
            .map(|(y, &m)| y.body.sub_offset(row * m, sub * m))
            .collect();
        let members: Vec<(&DeviceBuffer, &GpuTensor, usize)> = dev
            .a
            .iter()
            .zip(&ys)
            .zip(&case.ms)
            .map(|((a, y), &m)| (&a.buf, y, m))
            .collect();
        if !gpu
            .gemm_bf16_mfma_gfx942_rows16(&xs, &members, case.k, sub)
            .map_err(err)?
        {
            return Err(format!("rows16 declined slice {row} of {}", case.label()));
        }
    }
    Ok(())
}

fn members<'a>(
    case: &Case,
    dev: &'a Dev,
    out: &'a Out,
) -> Vec<(&'a DeviceBuffer, &'a GpuTensor, usize)> {
    dev.a
        .iter()
        .zip(&out.ys)
        .zip(&case.ms)
        .map(|((a, y), &m)| (&a.buf, &y.body, m))
        .collect()
}

/// `Gpu::gemm_bf16_mfma_gfx942_rows16_grid`, the public launcher.
fn product_grid(gpu: &mut Gpu, case: &Case, dev: &Dev, out: &Out) -> Res<()> {
    let mem = members(case, dev, out);
    if !gpu
        .gemm_bf16_mfma_gfx942_rows16_grid(&dev.x, &mem, case.k, case.rows)
        .map_err(err)?
    {
        return Err(format!("rows16_grid declined {}", case.label()));
    }
    Ok(())
}

/// `Gpu::gemm_bf16_mfma_gfx942_rows16_chain` with `chain` and `wide` set: the routing caller.
fn product_chain(gpu: &mut Gpu, case: &Case, dev: &Dev, out: &Out) -> Res<()> {
    let mem = members(case, dev, out);
    if !gpu
        .gemm_bf16_mfma_gfx942_rows16_chain(&dev.x, &mem, case.k, case.rows, true, true)
        .map_err(err)?
    {
        return Err(format!("rows16_chain declined {}", case.label()));
    }
    Ok(())
}

/// The module entries' kernarg ABI (the rows16 blob), launched directly.
#[derive(Clone, Copy)]
struct GridLaunch {
    x: *const c_void,
    a: [*const c_void; MAX_MEMBERS],
    d: [*const c_void; MAX_MEMBERS],
    m: [i32; MAX_MEMBERS],
    k: i32,
    batch: i32,
    x_f32: i32,
    d_bf16_mask: i32,
}

impl GridLaunch {
    fn new(case: &Case, dev: &Dev, out: &Out) -> Self {
        let mut l = GridLaunch {
            x: dev.x.buf.as_ptr(),
            a: [std::ptr::null(); MAX_MEMBERS],
            d: [std::ptr::null(); MAX_MEMBERS],
            m: [0; MAX_MEMBERS],
            k: case.k as i32,
            batch: case.rows as i32,
            x_f32: i32::from(case.x_f32),
            d_bf16_mask: case.mask as i32,
        };
        for j in 0..case.ms.len() {
            l.a[j] = dev.a[j].buf.as_ptr();
            l.d[j] = out.ys[j].body.buf.as_ptr();
            l.m[j] = case.ms[j] as i32;
        }
        l
    }
    /// Launch the `_x2` entry (`x2`) or the plain entry, loaded the way the product loads it.
    fn launch(&self, gpu: &mut Gpu, x2: bool) -> Res<()> {
        let symbol = if x2 { ENTRY_X2 } else { ENTRY };
        lab_ensure_rows16_grid_entry(gpu, symbol).map_err(err)?;
        let tiles: usize = self.m.iter().map(|&m| (m as usize).div_ceil(16)).sum();
        let batch = self.batch as usize;
        let mut blob = KernargBlob::new();
        blob.push_ptr(self.x);
        for p in self.a.iter().chain(&self.d) {
            blob.push_ptr(*p);
        }
        for v in self.m {
            blob.push_i32(v);
        }
        blob.push_i32(self.k);
        blob.push_i32(self.batch);
        blob.push_i32(self.x_f32);
        blob.push_i32(self.d_bf16_mask);
        blob.pad_to(16);
        gpu.launch_kernel_blob(
            symbol,
            [tiles as u32, batch.div_ceil(16) as u32, 1],
            [576, 1, 1],
            (2 * (16 + batch.min(16)) * LDS_PITCH * 2) as u32,
            blob.as_mut_slice(),
        )
        .map_err(err)
    }
}

fn run_case(gpu: &mut Gpu, tally: &mut Tally, case: &Case, dev: &Dev) -> Res<()> {
    let r_out = Out::new(gpu, case)?;
    reference(gpu, case, dev, &r_out)?;
    let r = r_out.read(gpu)?;
    let r2_out = Out::new(gpu, case)?;
    reference(gpu, case, dev, &r2_out)?;
    let r2 = r2_out.read(gpu)?;
    let product_x2 = case.product_entry() == ENTRY_X2;
    let m_out = Out::new(gpu, case)?;
    GridLaunch::new(case, dev, &m_out).launch(gpu, product_x2)?;
    let m = m_out.read(gpu)?;
    let mut x2 = None;
    if !product_x2 {
        let out = Out::new(gpu, case)?;
        GridLaunch::new(case, dev, &out).launch(gpu, true)?;
        x2 = Some(out.read(gpu)?);
    }
    let p_out = Out::new(gpu, case)?;
    product_grid(gpu, case, dev, &p_out)?;
    let p = p_out.read(gpu)?;
    let mut chain = None;
    if case.routed_by_chain() {
        let out = Out::new(gpu, case)?;
        product_chain(gpu, case, dev, &out)?;
        chain = Some(out.read(gpu)?);
    }
    let mut arms: Vec<(&str, &Arm)> = vec![("module", &m), ("product_grid", &p)];
    if let Some(a) = &x2 {
        arms.push(("module_x2", a));
    }
    if let Some(a) = &chain {
        arms.push(("product_chain", a));
    }
    tally.case(&case.label(), &r, &r2, &arms);
    Ok(())
}

/// Every `(x dtype, output mask, rows)` of one weight set (`ms`, `k`): the weights are
/// generated and uploaded once.
fn group(
    gpu: &mut Gpu,
    rng: &mut Rng,
    tally: &mut Tally,
    ms: &[usize],
    k: usize,
    rows_list: &[usize],
    masks: &[u32],
) -> Res<()> {
    let weights_host: Vec<Vec<u16>> = ms.iter().map(|&m| weights(rng, m * k)).collect();
    let a = upload_a(gpu, &weights_host)?;
    let full = (1u32 << ms.len()) - 1;
    for x_f32 in [true, false] {
        for &rows in rows_list {
            let x: Vec<f32> = (0..rows * k).map(|_| rng.f32() * 2.0).collect();
            let x_dev = upload_x(gpu, &x, x_f32)?;
            let d = Dev {
                x: x_dev,
                a: a.iter().map(clone_tensor_view).collect(),
            };
            for &mask in masks {
                let case = Case {
                    ms: ms.to_vec(),
                    k,
                    rows,
                    x_f32,
                    mask: mask & full,
                };
                run_case(gpu, tally, &case, &d)?;
            }
        }
    }
    Ok(())
}

/// A non-owning view of a weight tensor (every arm only reads it).
fn clone_tensor_view(t: &GpuTensor) -> GpuTensor {
    t.sub_offset(0, t.numel())
}

fn sweep(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally, quick: bool) -> Res<()> {
    let hc_rows: &[usize] = if quick {
        &[1, 9, 16, 17, 33, 64, 65, 255, 256]
    } else {
        &[
            1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 100, 128, 200, 255, 256,
        ]
    };
    // The Flash-Next HC down projection.
    group(gpu, rng, tally, &[320], 10240, hc_rows, &[0, 1])?;
    println!(
        "HC down (M=320 K=10240): cumulative PASS {} FAIL {}",
        tally.pass, tally.fail
    );
    // Members of mixed M with F32, BF16 and mixed outputs.
    let member_sets: &[&[usize]] = &[
        &[320, 320],
        &[320, 48, 1],
        &[320, 16, 17, 33],
        &[4, 4, 4, 4],
        &[1],
        &[17],
    ];
    let member_rows: &[usize] = if quick {
        &[1, 17, 64, 256]
    } else {
        &[1, 16, 17, 33, 64, 65, 255, 256]
    };
    let ks: &[usize] = if quick { &[2560] } else { &[2560, 264] };
    for ms in member_sets {
        for &k in ks {
            group(gpu, rng, tally, ms, k, member_rows, &[0, 0b1111, 0b0101])?;
        }
        println!(
            "members {ms:?}: cumulative PASS {} FAIL {}",
            tally.pass, tally.fail
        );
    }
    if !quick {
        for ms in [&[320, 320][..], &[320, 48, 1][..], &[320, 16, 17, 33][..]] {
            group(
                gpu,
                rng,
                tally,
                ms,
                10240,
                &[17, 64, 256],
                &[0, 0b1111, 0b0101],
            )?;
        }
    }
    // K tails: not a multiple of 128, an odd multiple of 128 + 8, a single 8-wide step.
    let k_rows: &[usize] = if quick { &[17, 33] } else { &[17, 33, 64] };
    let tail_ks: &[usize] = if quick {
        &[8, 264]
    } else {
        &[8, 136, 264, 640, 1032]
    };
    for &k in tail_ks {
        group(gpu, rng, tally, &[320], k, k_rows, &[0, 1])?;
    }
    println!(
        "K tails: cumulative PASS {} FAIL {}",
        tally.pass, tally.fail
    );
    Ok(())
}

/// The planted controls: two members of 320 weight rows (so swapped operands read in bounds), 40
/// rows (three slices, the last partial), `K = 2560`, F32 outputs; once with a BF16 `x`, once
/// with an F32 `x` (the `_x2` entry).
fn controls(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally) -> Res<()> {
    for x_f32 in [false, true] {
        let tag = if x_f32 { "x_f32" } else { "x_bf16" };
        let case = Case {
            ms: vec![320, 320],
            k: 2560,
            rows: 40,
            x_f32,
            mask: 0,
        };
        let mut h = host(rng, &case);
        // Element 0 of every operand: 0.5 (a mantissa-MSB flip changes it to 0.75).
        h.x[0] = 0.5;
        h.a[0][0] = 0x3F00;
        h.a[1][0] = 0x3F00;
        let clean = dev(gpu, &case, &h)?;
        let ref_out = Out::new(gpu, &case)?;
        reference(gpu, &case, &clean, &ref_out)?;
        let reference = ref_out.read(gpu)?;
        let product_x2 = case.product_entry() == ENTRY_X2;
        let run =
            |gpu: &mut Gpu, x2: bool, dev: &Dev, edit: &dyn Fn(&mut GridLaunch)| -> Res<Arm> {
                let out = Out::new(gpu, &case)?;
                let mut launch = GridLaunch::new(&case, dev, &out);
                edit(&mut launch);
                launch.launch(gpu, x2)?;
                out.read(gpu)
            };
        if diff(&run(gpu, product_x2, &clean, &|_| {})?, &reference) != 0 {
            return Err(format!(
                "{tag}: clean control case: module differs from reference"
            ));
        }
        let mut flip_x = clone_host(&h);
        flip_x.x[0] = f32::from_bits(flip_x.x[0].to_bits() ^ 0x0040_0000);
        let mut flip_a0 = clone_host(&h);
        flip_a0.a[0][0] ^= 0x0040;
        let mut flip_a1 = clone_host(&h);
        flip_a1.a[1][0] ^= 0x0040;
        for (name, h) in [
            ("flip_x_byte", &flip_x),
            ("flip_a0_byte", &flip_a0),
            ("flip_a1_byte", &flip_a1),
        ] {
            let d = dev(gpu, &case, h)?;
            tally.control(
                &format!("rows16_grid {tag} {name}"),
                &run(gpu, product_x2, &d, &|_| {})?,
                &reference,
            );
        }
        tally.control(
            &format!("rows16_grid {tag} rows_minus_1"),
            &run(gpu, product_x2, &clean, &|l| l.batch -= 1)?,
            &reference,
        );
        tally.control(
            &format!("rows16_grid {tag} swap_a0_a1"),
            &run(gpu, product_x2, &clean, &|l| l.a.swap(0, 1))?,
            &reference,
        );
        tally.control(
            &format!("rows16_grid {tag} swap_d0_d1"),
            &run(gpu, product_x2, &clean, &|l| l.d.swap(0, 1))?,
            &reference,
        );
        tally.control(
            &format!("rows16_grid {tag} wrong_k"),
            &run(gpu, product_x2, &clean, &|l| l.k -= 8)?,
            &reference,
        );
        tally.control(
            &format!("rows16_grid {tag} wrong_d_mask"),
            &run(gpu, product_x2, &clean, &|l| l.d_bf16_mask ^= 0b11)?,
            &reference,
        );
        if x_f32 {
            // 40 F32 rows: slices of 16 need the `_x2` entry; the plain entry loads half of them.
            tally.control(
                &format!("rows16_grid {tag} wrong_entry({ENTRY})"),
                &run(gpu, false, &clean, &|_| {})?,
                &reference,
            );
        }
    }
    Ok(())
}

// ───────────────────────────────── --bench ─────────────────────────────────

type ArmFn = fn(&mut Gpu, &Case, &Dev, &Out) -> Res<()>;

/// `Gpu::gemm_bf16_mfma_gfx942_wide_slices`: one wide64-entry launch per 64-row slice.
fn product_wide(gpu: &mut Gpu, case: &Case, dev: &Dev, out: &Out) -> Res<()> {
    let mem = members(case, dev, out);
    if !gpu
        .gemm_bf16_mfma_gfx942_wide_slices(&dev.x, &mem, case.k, case.rows)
        .map_err(err)?
    {
        return Err(format!("wide_slices declined {}", case.label()));
    }
    Ok(())
}

/// The routes timed per cell, in print order.
const BENCH_ARMS: [(&str, ArmFn); 4] = [
    ("rows16_grid", product_grid),
    ("rows16_chain", reference),
    ("wide_slices", product_wide),
    ("router", product_chain),
];

/// One cell's launch times (ms): an untimed run first (module load), then `reps` events-bracketed
/// runs; with `flush`, the buffer is memset before each (the weights and `x` then come from HBM).
#[allow(clippy::too_many_arguments)]
fn time_arm(
    gpu: &mut Gpu,
    arm: ArmFn,
    case: &Case,
    dev: &Dev,
    out: &Out,
    flush: Option<&GpuTensor>,
    reps: usize,
) -> Res<Vec<f32>> {
    let start = gpu.hip.event_create().map_err(err)?;
    let stop = gpu.hip.event_create().map_err(err)?;
    arm(gpu, case, dev, out)?;
    gpu.hip.device_synchronize().map_err(err)?;
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        if let Some(f) = flush {
            gpu.hip.memset(&f.buf, 0, f.byte_size()).map_err(err)?;
        }
        gpu.hip.event_record(&start, None).map_err(err)?;
        arm(gpu, case, dev, out)?;
        gpu.hip.event_record(&stop, None).map_err(err)?;
        gpu.hip.event_synchronize(&stop).map_err(err)?;
        times.push(gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?);
    }
    gpu.hip.event_destroy(start).map_err(err)?;
    gpu.hip.event_destroy(stop).map_err(err)?;
    Ok(times)
}

fn median(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// The row-grid cells behind `ROW_GRID_MAX_ROWS`: the HC down projection (`M = 320`, `K = 10240`,
/// BF16 `x`) at rows 17 / 32 / 64 / 128 / 256 through the row grid, the rows16 chain, the wide
/// entries and the public router, warm and cold; plus the 257-row decline. Returns the rows (one
/// per cell) and whether every equality and decline check held.
fn bench(gpu: &mut Gpu, rng: &mut Rng, reps: usize) -> Res<(Vec<serde_json::Value>, bool)> {
    if gpu.active_stream.is_some() {
        return Err(
            "bench: events are recorded on the null stream; an active stream is set".into(),
        );
    }
    let (m, k) = (320usize, 10240usize);
    let a = upload_a(gpu, &[weights(rng, m * k)])?;
    let flush = gpu.alloc_tensor(&[1 << 28], DType::F32).map_err(err)?;
    let mut rows_out = Vec::new();
    let mut ok = true;
    println!(
        "BENCH,cell,rows,arm,weights,reps,median_us,min_us,max_us,equal_to_reference,last_kernel"
    );
    let make_dev = |gpu: &mut Gpu, rng: &mut Rng, case: &Case| -> Res<Dev> {
        let x: Vec<f32> = (0..case.rows * k).map(|_| rng.f32() * 2.0).collect();
        Ok(Dev {
            x: upload_x(gpu, &x, false)?,
            a: a.iter().map(clone_tensor_view).collect(),
        })
    };
    for rows in [17usize, 32, 64, 128, 256] {
        let case = Case {
            ms: vec![m],
            k,
            rows,
            x_f32: false,
            mask: 0,
        };
        let d = make_dev(gpu, rng, &case)?;
        let ref_out = Out::new(gpu, &case)?;
        reference(gpu, &case, &d, &ref_out)?;
        let expected = ref_out.read(gpu)?;
        for (name, arm) in BENCH_ARMS {
            let check = Out::new(gpu, &case)?;
            arm(gpu, &case, &d, &check)?;
            let got = check.read(gpu)?;
            let equal = diff(&got, &expected) == 0 && got.1;
            let last = gpu.last_launched_kernel().unwrap_or("").to_owned();
            ok &= equal;
            for (weights_state, f) in [("warm", None), ("cold", Some(&flush))] {
                let out = Out::new(gpu, &case)?;
                let mut times = time_arm(gpu, arm, &case, &d, &out, f, reps)?;
                times.sort_by(|x, y| x.total_cmp(y));
                let (med, min, max) = (median(&times), times[0], times[times.len() - 1]);
                println!(
                    "BENCH,hc_down_m320_k10240,{rows},{name},{weights_state},{reps},{:.2},{:.2},{:.2},{equal},{last}",
                    med * 1e3,
                    min * 1e3,
                    max * 1e3
                );
                rows_out.push(json!({
                    "cell": "hc_down_m320_k10240", "rows": rows, "arm": name,
                    "weights": weights_state, "reps": reps,
                    "median_us": med * 1e3, "min_us": min * 1e3, "max_us": max * 1e3,
                    "equal_to_reference": equal, "last_kernel": last,
                }));
            }
        }
    }
    // The router at 257 rows. (a) The HC down shape through the full router (`chain` and `wide`):
    // the row grid is not taken (the last launch is not a row-grid entry), another route runs and
    // its bytes equal the rows16 chain's.
    let case = Case {
        ms: vec![m],
        k,
        rows: 257,
        x_f32: false,
        mask: 0,
    };
    let d = make_dev(gpu, rng, &case)?;
    let ref_out = Out::new(gpu, &case)?;
    reference(gpu, &case, &d, &ref_out)?;
    let expected = ref_out.read(gpu)?;
    let out = Out::new(gpu, &case)?;
    product_chain(gpu, &case, &d, &out)?;
    let got = out.read(gpu)?;
    let last = gpu.last_launched_kernel().unwrap_or("").to_owned();
    let not_grid = !last.starts_with(ENTRY);
    let equal = diff(&got, &expected) == 0 && got.1;
    ok &= not_grid && equal;
    println!(
        "DECLINE,router_chain_wide,hc_down_m320_k10240,rows=257,row_grid_taken={},equal_to_reference={equal},last_kernel={last}",
        !not_grid
    );
    rows_out.push(json!({
        "cell": "decline_257_router_chain_wide", "rows": 257, "row_grid_taken": !not_grid,
        "equal_to_reference": equal, "last_kernel": last,
    }));
    // (b) The row grid's own gate with no other route behind it: a 48-row weight (3 tiles,
    // K = 10240) with `chain` set and `wide` clear has no chain route past 192 rows, so the
    // router launches the grid at 256 rows and declines at 257 with the output untouched.
    let m48 = 48usize;
    let a48 = upload_a(gpu, &[weights(rng, m48 * k)])?;
    for rows in [256usize, 257] {
        let case = Case {
            ms: vec![m48],
            k,
            rows,
            x_f32: false,
            mask: 0,
        };
        let x: Vec<f32> = (0..rows * k).map(|_| rng.f32() * 2.0).collect();
        let d = Dev {
            x: upload_x(gpu, &x, false)?,
            a: a48.iter().map(clone_tensor_view).collect(),
        };
        let out = Out::new(gpu, &case)?;
        let before = gpu.last_launched_kernel().unwrap_or("").to_owned();
        let launched = {
            let mem = members(&case, &d, &out);
            gpu.gemm_bf16_mfma_gfx942_rows16_chain(&d.x, &mem, k, rows, true, false)
                .map_err(err)?
        };
        let last = gpu.last_launched_kernel().unwrap_or("").to_owned();
        let after = out.read(gpu)?;
        let untouched = after.1 && after.0.iter().all(|y| y.iter().all(|&b| b == 0x7f));
        let pass = if rows <= 256 {
            launched && last.starts_with(ENTRY) && !untouched
        } else {
            !launched && untouched && last == before
        };
        ok &= pass;
        println!(
            "DECLINE,router_chain_only,m48_k10240,rows={rows},launched={launched},output_untouched={untouched},last_kernel={last},pass={pass}"
        );
        rows_out.push(json!({
            "cell": "router_chain_only_m48_k10240", "rows": rows, "launched": launched,
            "output_untouched": untouched, "last_kernel": last, "pass": pass,
        }));
    }
    Ok((rows_out, ok))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let quick = args.iter().any(|a| a == "--quick");
    let seed = flag("--seed")
        .map(|s| s.parse::<u64>().expect("--seed U64"))
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.try_gfx942().is_none() {
        eprintln!("skip: needs an exact gfx942 device");
        std::process::exit(0);
    }
    let image = std::env::var("HIPFIRE_QWEN4_G942_ROWS16_GRID_IMAGE")
        .ok()
        .filter(|p| !p.is_empty());
    println!(
        "module arm: {}",
        image.as_deref().map_or(
            "hipcc source (module qwen4_g942_rows16_grid)".to_owned(),
            |p| format!("code object {p}")
        )
    );
    let mut rng = Rng(seed);
    if args.iter().any(|a| a == "--bench") {
        let reps = flag("--reps")
            .map(|s| s.parse::<usize>().expect("--reps N"))
            .unwrap_or(25)
            .max(20);
        let (rows, ok) = match bench(&mut gpu, &mut Rng(seed), reps) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("bench: {e}");
                std::process::exit(1);
            }
        };
        if let Some(path) = flag("--out") {
            let report = json!({
                "oracle": "qwen4_g942_rows16_grid_oracle", "mode": "bench", "seed": seed,
                "image": image, "reps": reps, "rows": rows, "ok": ok,
            });
            std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap())
                .expect("write --out");
        }
        println!("{}", if ok { "PASS" } else { "FAIL" });
        std::process::exit(if ok { 0 } else { 1 });
    }
    let mut tally = Tally::default();
    if let Err(e) = sweep(&mut gpu, &mut rng, &mut tally, quick) {
        eprintln!("sweep: {e}");
        std::process::exit(1);
    }
    if let Err(e) = controls(&mut gpu, &mut rng, &mut tally) {
        eprintln!("controls: {e}");
        std::process::exit(1);
    }
    println!(
        "RESULT cases PASS {} FAIL {} | controls DETECTED {} MISSED {}",
        tally.pass, tally.fail, tally.detected, tally.missed
    );
    if let Some(path) = flag("--out") {
        let report = json!({
            "oracle": "qwen4_g942_rows16_grid_oracle", "seed": seed, "quick": quick,
            "image": image, "cases_pass": tally.pass, "cases_fail": tally.fail,
            "controls_detected": tally.detected, "controls_missed": tally.missed,
            "rows": tally.rows,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap()).expect("write --out");
    }
    let ok = tally.fail == 0 && tally.missed == 0 && tally.pass > 0 && tally.detected > 0;
    println!("{}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
