// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-image byte oracle of the exact-gfx942 convert-elimination module
//! (`kernels/src/qwen4_g942_conv_fuse.gfx942.hip`, module `qwen4_g942_conv_fuse`, entries
//! `qwen4_g942_hyper_norm_bf16x` and `qwen4_g942_hyper_write_norm_bf16x`).
//!
//! Three arms run on the same inputs; every output of every launch (with 0x7f guard words on both
//! sides that must survive) must be byte-equal, the F32 outputs AND the BF16-bits rows:
//!
//! - REFERENCE: the unfused route the entries replace. `hyper_norm_f32` (`tensor_ops::hyper_norm`)
//!   resp. the fused write + read `hyper_write_norm_f32` (`tensor_ops::hyper_write_norm`, next-gate
//!   quarters and folded Clear included), followed by `convert_f32_to_bf16` of the F32
//!   `normalized` rows (the down projection's staged activation);
//! - MODULE: the module entries launched directly with the product's kernarg blob. The module is
//!   loaded the way the product loads it (`lab_ensure_conv_fuse_entry`): from the hipcc source,
//!   or, with `HIPFIRE_QWEN4_G942_CONV_FUSE_IMAGE=/absolute/device.elf`, from that code object
//!   (the twin proof: same module, same entry names, same kernarg ABI);
//! - PRODUCT: `Gfx942Device::hyper_norm_bf16x` / `Gfx942Device::hyper_write_norm_bf16x`, the
//!   public methods `hipfire-dispatch` layer_ops.rs calls.
//!
//! A second REFERENCE launch must equal the first. Shapes: `hyper_norm_bf16x` on the Flash-Next HC
//! geometry (4 branches, hidden 2560) with rows 1..=64 (every lane-step count 4 / 8 / 12 / 16 / 24
//! / 32 included) and 65 / 96 / 128, BF16-bits input (`state_bf16`) on a subset, and the other
//! geometries the kernel admits (1 / 2 / 3 / 8 branches; hidden 1 / 3 / 100 / 257 / 1024 / 2048 /
//! 2049 / 5120 / 5121 / 7680 / 10241: the single-batch path at 2560, the batched strided path
//! elsewhere, hidden not a multiple of 256). `hyper_write_norm_bf16x` (4 branches, hidden 2560,
//! rows 1..=64 as the product admits) over all eight combinations of `quarters_in` (replacing
//! the gate GEMV), `next` (the next write's gate quarters) and folded `clear` (sizes 1 / 255 /
//! 4097 / one past the grid's first sweep).
//!
//! Planted controls the oracle MUST detect (module output differs from the clean reference): one
//! flipped byte in every input operand (`hyper_norm`: input, norm weight; `hyper_write_norm`:
//! input, mixed, gates, quarters_in, norm weight, next norm weight, inject), the wrong row count
//! (`rows - 1`), swapped operands (norm weight and input; norm weight and next norm weight; input
//! and output), a wrong state flag, a dropped optional operand (quarters_in, next, clear, the BF16
//! output), and the other entry: `hyper_norm_bf16x` on the pre-write streams against the
//! `hyper_write_norm` reference. Exit 0 only when every case matches and every control is
//! detected. Prints PASS / FAIL counts.
//!
//! `qwen4_g942_conv_fuse_oracle [--out FILE.json] [--seed U64] [--quick]`

use std::ffi::c_void;

use hip_bridge::KernargBlob;
use rdna_compute::tensor_ops::{
    hyper_norm, hyper_write_norm, lab_ensure_conv_fuse_entry, HyperNextGates, HyperNorm, HyperWrite,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;

type Res<T> = Result<T, String>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

const NORM_ENTRY: &str = "qwen4_g942_hyper_norm_bf16x";
const WRITE_NORM_ENTRY: &str = "qwen4_g942_hyper_write_norm_bf16x";
const GUARD: usize = 64;
const BRANCHES: usize = 4;
const HIDDEN: usize = 2560;
const WIDE: usize = BRANCHES * HIDDEN;
const MAX_ROWS: usize = 64;

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
    /// An HC stream value: mostly unit range, some large, some exact zeros.
    fn stream(&mut self) -> f32 {
        let v = self.f32() * if self.next() & 15 == 0 { 32.0 } else { 1.0 };
        if self.next() & 63 == 0 {
            0.0
        } else {
            v
        }
    }
    /// A BF16 weight in `[-0.5, 0.5)`.
    fn bf16(&mut self) -> u16 {
        ((self.f32() * 0.5).to_bits() >> 16) as u16
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

fn read_arm(gpu: &Gpu, outs: &[&Guarded]) -> Res<Arm> {
    let mut parts = Vec::new();
    let mut guards = true;
    for g in outs {
        let (bytes, ok) = g.read(gpu)?;
        parts.push(bytes);
        guards &= ok;
    }
    Ok((parts, guards))
}

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
    /// The three arms (and the repeated reference) of one case against the reference.
    fn case(&mut self, label: &str, r: &Arm, r2: &Arm, m: &Arm, p: &Arm) {
        let (d_m, d_p, d_r2) = (diff(m, r), diff(p, r), diff(r2, r));
        let guards = r.1 && r2.1 && m.1 && p.1;
        let ok = d_m == 0 && d_p == 0 && d_r2 == 0 && guards;
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
            eprintln!(
                "FAIL {label}: module_vs_ref={d_m} product_vs_ref={d_p} ref_repeat={d_r2} guards_ok={guards}"
            );
        }
        if !ok || self.rows.len() < 8 {
            self.rows.push(json!({
                "case": label, "ok": ok, "module_vs_reference": d_m,
                "product_vs_reference": d_p, "reference_repeat": d_r2, "guards_ok": guards,
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

// ───────────────────────────── hyper_norm_bf16x ─────────────────────────────

#[derive(Clone, Copy, Debug)]
struct NormCase {
    branches: usize,
    hidden: usize,
    rows: usize,
    /// The input holds BF16 bits (`HyperNorm::state_bf16`).
    state_bf16: bool,
}

impl NormCase {
    fn wide(self) -> usize {
        self.branches * self.hidden
    }
    fn elements(self) -> usize {
        self.rows * self.wide()
    }
}

struct NormHost {
    /// The uploaded input bytes: F32 rows, or BF16 bits padded to the F32 extent.
    input: Vec<u8>,
    weight: Vec<u16>,
}

fn norm_host(rng: &mut Rng, case: NormCase) -> NormHost {
    let n = case.elements();
    let input = if case.state_bf16 {
        let bits: Vec<u16> = (0..n)
            .map(|_| (rng.stream().to_bits() >> 16) as u16)
            .collect();
        let mut bytes = u16_bytes(&bits);
        bytes.resize(n * 4, 0);
        bytes
    } else {
        f32_bytes(&(0..n).map(|_| rng.stream()).collect::<Vec<_>>())
    };
    let weight = (0..case.wide()).map(|_| rng.bf16()).collect();
    NormHost { input, weight }
}

struct NormDev {
    input: GpuTensor,
    weight: GpuTensor,
}

fn norm_dev(gpu: &Gpu, case: NormCase, host: &NormHost) -> Res<NormDev> {
    Ok(NormDev {
        input: upload(gpu, &host.input, DType::F32, case.elements())?,
        weight: upload(gpu, &u16_bytes(&host.weight), DType::BF16, case.wide())?,
    })
}

struct NormOut {
    f32: Guarded,
    bf16: Guarded,
}

impl NormOut {
    fn new(gpu: &mut Gpu, case: NormCase) -> Res<Self> {
        Ok(NormOut {
            f32: Guarded::new(gpu, case.elements(), DType::F32)?,
            bf16: Guarded::new(gpu, case.elements(), DType::BF16)?,
        })
    }
    fn read(&self, gpu: &Gpu) -> Res<Arm> {
        read_arm(gpu, &[&self.f32, &self.bf16])
    }
}

fn hyper_norm_args<'a>(case: NormCase, dev: &'a NormDev, out: &'a NormOut) -> HyperNorm<'a> {
    HyperNorm {
        input: &dev.input,
        norm_weight: &dev.weight,
        normalized: &out.f32.body,
        branches: case.branches,
        hidden: case.hidden,
        state_bf16: case.state_bf16,
    }
}

/// `hyper_norm_f32` then `convert_f32_to_bf16` of its F32 rows.
fn norm_reference(gpu: &mut Gpu, case: NormCase, dev: &NormDev, out: &NormOut) -> Res<()> {
    hyper_norm(gpu, &hyper_norm_args(case, dev, out)).map_err(err)?;
    gpu.convert_f32_to_bf16(&out.f32.body, &out.bf16.body, case.elements())
        .map_err(err)
}

fn norm_product(gpu: &mut Gpu, case: NormCase, dev: &NormDev, out: &NormOut) -> Res<()> {
    let mut device = gpu.try_gfx942().ok_or("not an exact gfx942 device")?;
    device
        .hyper_norm_bf16x(&hyper_norm_args(case, dev, out), &out.bf16.body)
        .map_err(err)
}

/// The module entry's kernarg ABI (`hyper_norm_bf16x`'s blob), launched directly.
struct NormLaunch {
    input: *const c_void,
    weight: *const c_void,
    normalized: *const c_void,
    bf16: *const c_void,
    branches: usize,
    hidden: usize,
    /// The `rows` kernarg.
    rows: usize,
    /// The grid's row extent.
    grid_rows: usize,
    state_bf16: bool,
}

impl NormLaunch {
    fn new(case: NormCase, dev: &NormDev, out: &NormOut) -> Self {
        NormLaunch {
            input: dev.input.buf.as_ptr(),
            weight: dev.weight.buf.as_ptr(),
            normalized: out.f32.body.buf.as_ptr(),
            bf16: out.bf16.body.buf.as_ptr(),
            branches: case.branches,
            hidden: case.hidden,
            rows: case.rows,
            grid_rows: case.rows,
            state_bf16: case.state_bf16,
        }
    }
    fn launch(&self, gpu: &mut Gpu, symbol: &str) -> Res<()> {
        let mut args = KernargBlob::new();
        args.push_ptr(self.input);
        args.push_ptr(self.weight);
        args.push_ptr(self.normalized);
        args.push_i32(self.branches as i32);
        args.push_i32(self.hidden as i32);
        args.push_i32(self.rows as i32);
        args.push_ptr(self.bf16);
        args.push_i32(i32::from(self.state_bf16));
        args.pad_to(16);
        gpu.launch_kernel_blob(
            symbol,
            [self.branches as u32, self.grid_rows as u32, 1],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
        )
        .map_err(err)
    }
}

fn norm_case(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally, case: NormCase) -> Res<()> {
    let label = format!(
        "hyper_norm_bf16x b{} h{} rows{}{}",
        case.branches,
        case.hidden,
        case.rows,
        if case.state_bf16 { " state_bf16" } else { "" }
    );
    let host = norm_host(rng, case);
    let dev = norm_dev(gpu, case, &host)?;
    let module = lab_ensure_conv_fuse_entry(gpu, NORM_ENTRY).map_err(err)?;
    let r_out = NormOut::new(gpu, case)?;
    norm_reference(gpu, case, &dev, &r_out)?;
    let r = r_out.read(gpu)?;
    let r2_out = NormOut::new(gpu, case)?;
    norm_reference(gpu, case, &dev, &r2_out)?;
    let r2 = r2_out.read(gpu)?;
    let m_out = NormOut::new(gpu, case)?;
    NormLaunch::new(case, &dev, &m_out).launch(gpu, module)?;
    let m = m_out.read(gpu)?;
    let p_out = NormOut::new(gpu, case)?;
    norm_product(gpu, case, &dev, &p_out)?;
    let p = p_out.read(gpu)?;
    tally.case(&label, &r, &r2, &m, &p);
    Ok(())
}

fn norm_sweep(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally, quick: bool) -> Res<()> {
    let hc_rows: Vec<usize> = if quick {
        vec![1, 2, 4, 8, 12, 16, 17, 24, 32, 33, 48, 64, 65, 128]
    } else {
        (1..=MAX_ROWS).chain([65, 96, 128]).collect()
    };
    for &rows in &hc_rows {
        let case = NormCase {
            branches: BRANCHES,
            hidden: HIDDEN,
            rows,
            state_bf16: false,
        };
        norm_case(gpu, rng, tally, case)?;
    }
    for rows in [1, 16, 17, 64] {
        let case = NormCase {
            branches: BRANCHES,
            hidden: HIDDEN,
            rows,
            state_bf16: true,
        };
        norm_case(gpu, rng, tally, case)?;
    }
    let others: &[(usize, usize)] = &[
        (1, 2560),
        (2, 2560),
        (3, 2560),
        (8, 2560),
        (4, 1),
        (4, 3),
        (4, 100),
        (3, 257),
        (4, 1024),
        (4, 2048),
        (4, 2049),
        (4, 5120),
        (4, 5121),
        (4, 7680),
        (2, 10241),
    ];
    let other_rows: &[usize] = if quick { &[1, 17] } else { &[1, 3, 17, 64] };
    for &(branches, hidden) in others {
        for &rows in other_rows {
            for state_bf16 in [false, true] {
                if state_bf16 && rows != 17 {
                    continue;
                }
                let case = NormCase {
                    branches,
                    hidden,
                    rows,
                    state_bf16,
                };
                norm_case(gpu, rng, tally, case)?;
            }
        }
    }
    Ok(())
}

/// The planted controls of `hyper_norm_bf16x` on the HC geometry, 4 rows.
fn norm_controls(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally) -> Res<()> {
    let case = NormCase {
        branches: BRANCHES,
        hidden: HIDDEN,
        rows: 4,
        state_bf16: false,
    };
    let module = lab_ensure_conv_fuse_entry(gpu, NORM_ENTRY).map_err(err)?;
    let mut host = norm_host(rng, case);
    // Element 0: input 0.5, weight 0.25 (one mantissa-MSB flip changes each visibly).
    host.input[..4].copy_from_slice(&0.5f32.to_le_bytes());
    host.weight[0] = 0x3E80;
    let clean = norm_dev(gpu, case, &host)?;
    let ref_out = NormOut::new(gpu, case)?;
    norm_reference(gpu, case, &clean, &ref_out)?;
    let reference = ref_out.read(gpu)?;
    let run = |gpu: &mut Gpu, dev: &NormDev, edit: &dyn Fn(&mut NormLaunch)| -> Res<Arm> {
        let out = NormOut::new(gpu, case)?;
        let mut launch = NormLaunch::new(case, dev, &out);
        edit(&mut launch);
        launch.launch(gpu, module)?;
        out.read(gpu)
    };
    // Sanity: the clean module launch equals the reference.
    let clean_run = run(gpu, &clean, &|_| {})?;
    if diff(&clean_run, &reference) != 0 {
        return Err("hyper_norm_bf16x: clean control case: module differs from reference".into());
    }
    let mut flip_input = NormHost {
        input: host.input.clone(),
        weight: host.weight.clone(),
    };
    flip_input.input[2] ^= 0x40;
    let mut flip_weight = NormHost {
        input: host.input.clone(),
        weight: host.weight.clone(),
    };
    flip_weight.weight[0] ^= 0x0040;
    for (name, h) in [
        ("hyper_norm flip_input_byte", &flip_input),
        ("hyper_norm flip_weight_byte", &flip_weight),
    ] {
        let dev = norm_dev(gpu, case, h)?;
        tally.control(name, &run(gpu, &dev, &|_| {})?, &reference);
    }
    tally.control(
        "hyper_norm rows_minus_1",
        &run(gpu, &clean, &|l| {
            l.rows -= 1;
            l.grid_rows -= 1;
        })?,
        &reference,
    );
    tally.control(
        "hyper_norm wrong_state_flag",
        &run(gpu, &clean, &|l| l.state_bf16 = true)?,
        &reference,
    );
    tally.control(
        "hyper_norm null_bf16_output",
        &run(gpu, &clean, &|l| l.bf16 = std::ptr::null())?,
        &reference,
    );
    // Swapped operands on one row: the weight tensor padded to the F32 input's extent so each
    // operand reads in bounds as the other's type.
    let one = NormCase { rows: 1, ..case };
    let one_host = NormHost {
        input: host.input[..one.elements() * 4].to_vec(),
        weight: host.weight.clone(),
    };
    let one_dev = norm_dev(gpu, one, &one_host)?;
    let mut padded = u16_bytes(&one_host.weight);
    padded.resize(one.elements() * 4, 0);
    let weight_wide = upload(gpu, &padded, DType::F32, one.elements())?;
    let one_ref = NormOut::new(gpu, one)?;
    norm_reference(gpu, one, &one_dev, &one_ref)?;
    let one_reference = one_ref.read(gpu)?;
    let out = NormOut::new(gpu, one)?;
    let mut launch = NormLaunch::new(one, &one_dev, &out);
    launch.input = weight_wide.buf.as_ptr();
    launch.weight = one_dev.input.buf.as_ptr();
    launch.launch(gpu, module)?;
    tally.control(
        "hyper_norm swap_input_weight",
        &out.read(gpu)?,
        &one_reference,
    );
    Ok(())
}

// ─────────────────────────── hyper_write_norm_bf16x ───────────────────────────

/// Which optional operands a `hyper_write_norm` call carries.
#[derive(Clone, Copy, Debug)]
struct Var {
    quarters_in: bool,
    next: bool,
    /// The folded Clear's element count.
    clear: Option<usize>,
}

struct WHost {
    input: Vec<f32>,
    mixed: Vec<f32>,
    gates: Vec<f32>,
    quarters_in: Vec<f32>,
    norm_w: Vec<u16>,
    next_w: Vec<u16>,
    inject: Vec<u16>,
}

fn scaled(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.f32() * scale).collect()
}

fn weights(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n).map(|_| rng.bf16()).collect()
}

fn w_host(rng: &mut Rng, rows: usize) -> WHost {
    let input: Vec<f32> = (0..rows * WIDE).map(|_| rng.stream()).collect();
    WHost {
        input,
        mixed: scaled(rng, rows * HIDDEN, 1.0),
        gates: scaled(rng, rows * BRANCHES, 6.0),
        quarters_in: scaled(rng, rows * 16, 3.0),
        norm_w: weights(rng, WIDE),
        next_w: weights(rng, WIDE),
        inject: weights(rng, 4 * WIDE),
    }
}

/// Plant the control values the one-bit flips of [`w_controls`] act on (element 0 of row 0).
fn w_plant(h: &mut WHost) {
    h.input[0] = 0.5;
    h.mixed[0] = 0.75;
    h.gates[0] = 1.0;
    h.quarters_in[0] = 0.25;
    h.norm_w[0] = 0x3E80;
    h.next_w[0] = 0x3E80;
    h.inject[0] = 0x3E80;
}

struct WDev {
    input: GpuTensor,
    mixed: GpuTensor,
    gates: GpuTensor,
    quarters_in: GpuTensor,
    norm_w: GpuTensor,
    next_w: GpuTensor,
    inject: GpuTensor,
}

fn w_dev(gpu: &Gpu, rows: usize, h: &WHost) -> Res<WDev> {
    let f = |v: &[f32]| upload(gpu, &f32_bytes(v), DType::F32, v.len());
    let b = |v: &[u16]| upload(gpu, &u16_bytes(v), DType::BF16, v.len());
    assert_eq!(h.input.len(), rows * WIDE);
    Ok(WDev {
        input: f(&h.input)?,
        mixed: f(&h.mixed)?,
        gates: f(&h.gates)?,
        quarters_in: f(&h.quarters_in)?,
        norm_w: b(&h.norm_w)?,
        next_w: b(&h.next_w)?,
        inject: b(&h.inject)?,
    })
}

struct WOut {
    output: Guarded,
    normalized: Guarded,
    bf16: Guarded,
    quarters: Option<Guarded>,
    clear: Option<Guarded>,
}

impl WOut {
    fn new(gpu: &mut Gpu, rows: usize, var: Var) -> Res<Self> {
        Ok(WOut {
            output: Guarded::new(gpu, rows * WIDE, DType::F32)?,
            normalized: Guarded::new(gpu, rows * WIDE, DType::F32)?,
            bf16: Guarded::new(gpu, rows * WIDE, DType::BF16)?,
            quarters: if var.next {
                Some(Guarded::new(gpu, rows * 16, DType::F32)?)
            } else {
                None
            },
            clear: match var.clear {
                Some(n) => Some(Guarded::new(gpu, n, DType::F32)?),
                None => None,
            },
        })
    }
    /// output, normalized, BF16 rows, then quarters and clear when present.
    fn read(&self, gpu: &Gpu) -> Res<Arm> {
        let mut outs: Vec<&Guarded> = vec![&self.output, &self.normalized, &self.bf16];
        outs.extend(self.quarters.iter());
        outs.extend(self.clear.iter());
        read_arm(gpu, &outs)
    }
}

fn hyper_write<'a>(dev: &'a WDev, out: &'a WOut) -> HyperWrite<'a> {
    HyperWrite {
        input: &dev.input,
        normalized: &out.normalized.body,
        mixed: &dev.mixed,
        gates: &dev.gates,
        output: &out.output.body,
        branches: BRANCHES,
        hidden: HIDDEN,
        state_bf16: false,
    }
}

fn w_reference(gpu: &mut Gpu, rows: usize, var: Var, dev: &WDev, out: &WOut) -> Res<()> {
    let next = out.quarters.as_ref().map(|q| HyperNextGates {
        norm_weight: &dev.next_w,
        inject: &dev.inject,
        quarters: &q.body,
    });
    hyper_write_norm(
        gpu,
        &hyper_write(dev, out),
        &dev.norm_w,
        &out.normalized.body,
        var.quarters_in.then_some(&dev.quarters_in),
        next.as_ref(),
        out.clear.as_ref().map(|c| &c.body),
    )
    .map_err(err)?;
    gpu.convert_f32_to_bf16(&out.normalized.body, &out.bf16.body, rows * WIDE)
        .map_err(err)
}

fn w_product(gpu: &mut Gpu, var: Var, dev: &WDev, out: &WOut) -> Res<()> {
    let next = out.quarters.as_ref().map(|q| HyperNextGates {
        norm_weight: &dev.next_w,
        inject: &dev.inject,
        quarters: &q.body,
    });
    let mut device = gpu.try_gfx942().ok_or("not an exact gfx942 device")?;
    device
        .hyper_write_norm_bf16x(
            &hyper_write(dev, out),
            &dev.norm_w,
            &out.normalized.body,
            &out.bf16.body,
            var.quarters_in.then_some(&dev.quarters_in),
            next.as_ref(),
            out.clear.as_ref().map(|c| &c.body),
        )
        .map_err(err)
}

/// The module entry's kernarg ABI (`hyper_write_norm_bf16x`'s blob), launched directly.
#[derive(Clone, Copy)]
struct WLaunch {
    input: *const c_void,
    mixed: *const c_void,
    gates: *const c_void,
    quarters_in: *const c_void,
    output: *const c_void,
    norm_w: *const c_void,
    normalized: *const c_void,
    bf16: *const c_void,
    next_norm: *const c_void,
    inject: *const c_void,
    quarters_out: *const c_void,
    clear: *const c_void,
    clear_elements: usize,
    grid_rows: usize,
}

impl WLaunch {
    fn new(rows: usize, var: Var, dev: &WDev, out: &WOut) -> Self {
        let null = std::ptr::null();
        let p = |t: &GpuTensor| t.buf.as_ptr() as *const c_void;
        WLaunch {
            input: p(&dev.input),
            mixed: p(&dev.mixed),
            gates: p(&dev.gates),
            quarters_in: if var.quarters_in {
                p(&dev.quarters_in)
            } else {
                null
            },
            output: p(&out.output.body),
            norm_w: p(&dev.norm_w),
            normalized: p(&out.normalized.body),
            bf16: p(&out.bf16.body),
            next_norm: if var.next { p(&dev.next_w) } else { null },
            inject: if var.next { p(&dev.inject) } else { null },
            quarters_out: out.quarters.as_ref().map_or(null, |q| p(&q.body)),
            clear: out.clear.as_ref().map_or(null, |c| p(&c.body)),
            clear_elements: var.clear.unwrap_or(0),
            grid_rows: rows,
        }
    }
    fn launch(&self, gpu: &mut Gpu, symbol: &str) -> Res<()> {
        let mut args = KernargBlob::new();
        for ptr in [
            self.input,
            self.mixed,
            self.gates,
            self.quarters_in,
            self.output,
            self.norm_w,
            self.normalized,
            self.bf16,
            self.next_norm,
            self.inject,
            self.quarters_out,
            self.clear,
        ] {
            args.push_ptr(ptr);
        }
        args.push_i32(self.clear_elements as i32);
        args.push_i32(BRANCHES as i32);
        args.push_i32(HIDDEN as i32);
        args.pad_to(16);
        gpu.launch_kernel_blob(
            symbol,
            [BRANCHES as u32, self.grid_rows as u32, 1],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
        )
        .map_err(err)
    }
}

fn w_case(gpu: &mut Gpu, tally: &mut Tally, rows: usize, var: Var, dev: &WDev) -> Res<()> {
    let label = format!(
        "hyper_write_norm_bf16x rows{rows} quarters_in={} next={} clear={:?}",
        var.quarters_in, var.next, var.clear
    );
    let module = lab_ensure_conv_fuse_entry(gpu, WRITE_NORM_ENTRY).map_err(err)?;
    let r_out = WOut::new(gpu, rows, var)?;
    w_reference(gpu, rows, var, dev, &r_out)?;
    let r = r_out.read(gpu)?;
    let r2_out = WOut::new(gpu, rows, var)?;
    w_reference(gpu, rows, var, dev, &r2_out)?;
    let r2 = r2_out.read(gpu)?;
    let m_out = WOut::new(gpu, rows, var)?;
    WLaunch::new(rows, var, dev, &m_out).launch(gpu, module)?;
    let m = m_out.read(gpu)?;
    let p_out = WOut::new(gpu, rows, var)?;
    w_product(gpu, var, dev, &p_out)?;
    let p = p_out.read(gpu)?;
    tally.case(&label, &r, &r2, &m, &p);
    Ok(())
}

fn w_sweep(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally, quick: bool) -> Res<()> {
    let rows_list: Vec<usize> = if quick {
        vec![1, 2, 4, 8, 12, 16, 17, 24, 32, 33, 48, 64]
    } else {
        (1..=MAX_ROWS).collect()
    };
    for (idx, &rows) in rows_list.iter().enumerate() {
        let host = w_host(rng, rows);
        let dev = w_dev(gpu, rows, &host)?;
        let clear_sizes = [1, 255, 4097, rows * BRANCHES * 256 + 13];
        for bits in 0..8usize {
            let var = Var {
                quarters_in: bits & 1 != 0,
                next: bits & 2 != 0,
                clear: (bits & 4 != 0).then(|| clear_sizes[(idx + bits) % clear_sizes.len()]),
            };
            w_case(gpu, tally, rows, var, &dev)?;
        }
    }
    Ok(())
}

/// The planted controls of `hyper_write_norm_bf16x` (17 rows; once with `quarters_in`, once
/// with the gate GEMV's gates).
fn w_controls(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally) -> Res<()> {
    let rows = 17usize;
    let module = lab_ensure_conv_fuse_entry(gpu, WRITE_NORM_ENTRY).map_err(err)?;
    let norm_module = lab_ensure_conv_fuse_entry(gpu, NORM_ENTRY).map_err(err)?;
    for quarters_in in [false, true] {
        let tag = if quarters_in { "qin" } else { "gates" };
        let var = Var {
            quarters_in,
            next: true,
            clear: Some(5000),
        };
        let mut host = w_host(rng, rows);
        w_plant(&mut host);
        let clone = |h: &WHost| WHost {
            input: h.input.clone(),
            mixed: h.mixed.clone(),
            gates: h.gates.clone(),
            quarters_in: h.quarters_in.clone(),
            norm_w: h.norm_w.clone(),
            next_w: h.next_w.clone(),
            inject: h.inject.clone(),
        };
        let clean = w_dev(gpu, rows, &host)?;
        let ref_out = WOut::new(gpu, rows, var)?;
        w_reference(gpu, rows, var, &clean, &ref_out)?;
        let reference = ref_out.read(gpu)?;
        let run = |gpu: &mut Gpu, dev: &WDev, edit: &dyn Fn(&mut WLaunch)| -> Res<Arm> {
            let out = WOut::new(gpu, rows, var)?;
            let mut launch = WLaunch::new(rows, var, dev, &out);
            edit(&mut launch);
            launch.launch(gpu, module)?;
            out.read(gpu)
        };
        let clean_run = run(gpu, &clean, &|_| {})?;
        if diff(&clean_run, &reference) != 0 {
            return Err(format!(
                "hyper_write_norm_bf16x {tag}: clean control case: module differs from reference"
            ));
        }
        // One flipped byte (the mantissa MSB of element 0) in every operand the case reads.
        type Flip = (&'static str, fn(&mut WHost));
        let flips: [Flip; 7] = [
            ("input", |h| {
                h.input[0] = f32::from_bits(h.input[0].to_bits() ^ 0x0040_0000)
            }),
            ("mixed", |h| {
                h.mixed[0] = f32::from_bits(h.mixed[0].to_bits() ^ 0x0040_0000)
            }),
            ("gates", |h| {
                h.gates[0] = f32::from_bits(h.gates[0].to_bits() ^ 0x0040_0000)
            }),
            ("quarters_in", |h| {
                h.quarters_in[0] = f32::from_bits(h.quarters_in[0].to_bits() ^ 0x0040_0000)
            }),
            ("norm_weight", |h| h.norm_w[0] ^= 0x0040),
            ("next_norm_weight", |h| h.next_w[0] ^= 0x0040),
            ("inject", |h| h.inject[0] ^= 0x0040),
        ];
        for (what, flip) in flips {
            // The gate source not read by this case cannot show a flip.
            if (what == "gates" && quarters_in) || (what == "quarters_in" && !quarters_in) {
                continue;
            }
            let mut h = clone(&host);
            flip(&mut h);
            let dev = w_dev(gpu, rows, &h)?;
            tally.control(
                &format!("hyper_write_norm {tag} flip_{what}_byte"),
                &run(gpu, &dev, &|_| {})?,
                &reference,
            );
        }
        tally.control(
            &format!("hyper_write_norm {tag} rows_minus_1"),
            &run(gpu, &clean, &|l| l.grid_rows -= 1)?,
            &reference,
        );
        tally.control(
            &format!("hyper_write_norm {tag} swap_norm_weight_next_norm"),
            &run(gpu, &clean, &|l| {
                std::mem::swap(&mut l.norm_w, &mut l.next_norm)
            })?,
            &reference,
        );
        // Input and output swapped: on a fresh device copy (the launch writes through `input`).
        {
            let dev = w_dev(gpu, rows, &host)?;
            tally.control(
                &format!("hyper_write_norm {tag} swap_input_output"),
                &run(gpu, &dev, &|l| std::mem::swap(&mut l.input, &mut l.output))?,
                &reference,
            );
        }
        if quarters_in {
            tally.control(
                "hyper_write_norm drop_quarters_in",
                &run(gpu, &clean, &|l| l.quarters_in = std::ptr::null())?,
                &reference,
            );
        }
        tally.control(
            &format!("hyper_write_norm {tag} drop_next"),
            &run(gpu, &clean, &|l| {
                l.next_norm = std::ptr::null();
                l.inject = std::ptr::null();
                l.quarters_out = std::ptr::null();
            })?,
            &reference,
        );
        tally.control(
            &format!("hyper_write_norm {tag} drop_clear"),
            &run(gpu, &clean, &|l| {
                l.clear = std::ptr::null();
                l.clear_elements = 0;
            })?,
            &reference,
        );
        tally.control(
            &format!("hyper_write_norm {tag} null_bf16_output"),
            &run(gpu, &clean, &|l| l.bf16 = std::ptr::null())?,
            &reference,
        );
        // The other entry: `hyper_norm_bf16x` on the PRE-write streams (what the next read would
        // norm without the write's injection) against the write-norm reference's `normalized`.
        let norm_case = NormCase {
            branches: BRANCHES,
            hidden: HIDDEN,
            rows,
            state_bf16: false,
        };
        let norm_out = NormOut::new(gpu, norm_case)?;
        let norm_dev = NormDev {
            input: upload(gpu, &f32_bytes(&host.input), DType::F32, rows * WIDE)?,
            weight: upload(gpu, &u16_bytes(&host.norm_w), DType::BF16, WIDE)?,
        };
        NormLaunch::new(norm_case, &norm_dev, &norm_out).launch(gpu, norm_module)?;
        let wrong = norm_out.read(gpu)?;
        let wrong_entry: Arm = (wrong.0.clone(), wrong.1);
        let expected: Arm = (
            vec![reference.0[1].clone(), reference.0[2].clone()],
            reference.1,
        );
        tally.control(
            &format!("hyper_write_norm {tag} wrong_entry(hyper_norm_bf16x)"),
            &wrong_entry,
            &expected,
        );
    }
    Ok(())
}

type SweepFn = fn(&mut Gpu, &mut Rng, &mut Tally, bool) -> Res<()>;

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
    let image = std::env::var("HIPFIRE_QWEN4_G942_CONV_FUSE_IMAGE")
        .ok()
        .filter(|p| !p.is_empty());
    println!(
        "module arm: {}",
        image.as_deref().map_or(
            "hipcc source (module qwen4_g942_conv_fuse)".to_owned(),
            |p| format!("code object {p}")
        )
    );
    let mut rng = Rng(seed);
    let mut tally = Tally::default();
    let steps: [(&str, SweepFn); 2] = [
        ("hyper_norm_bf16x", norm_sweep),
        ("hyper_write_norm_bf16x", w_sweep),
    ];
    for (name, sweep) in steps {
        if let Err(e) = sweep(&mut gpu, &mut rng, &mut tally, quick) {
            eprintln!("{name}: {e}");
            std::process::exit(1);
        }
        println!("{name}: cumulative PASS {} FAIL {}", tally.pass, tally.fail);
    }
    for (name, controls) in [
        (
            "hyper_norm_bf16x",
            norm_controls as fn(&mut Gpu, &mut Rng, &mut Tally) -> Res<()>,
        ),
        ("hyper_write_norm_bf16x", w_controls),
    ] {
        if let Err(e) = controls(&mut gpu, &mut rng, &mut tally) {
            eprintln!("controls {name}: {e}");
            std::process::exit(1);
        }
    }
    println!(
        "RESULT cases PASS {} FAIL {} | controls DETECTED {} MISSED {}",
        tally.pass, tally.fail, tally.detected, tally.missed
    );
    if let Some(path) = flag("--out") {
        let report = json!({
            "oracle": "qwen4_g942_conv_fuse_oracle", "seed": seed, "quick": quick,
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
