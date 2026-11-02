// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Strict G1 BF16 PM row-tile byte gate for exact gfx1151 and gfx1201.
//!
//! Every ordinary x4 and HC k4 fixture executes every logical R=1..64,
//! including partial physical eight-row tiles. References are the actual runtime
//! singleton single/x4/k4 launches, never PM C1 or a CPU approximation.
//! All output bytes, inactive rows, head/tail guards and read-only operands are
//! checked. Inputs beyond R are NaN poisoned. Setup, missing identity, route and
//! candidate failures are fatal; unsupported devices are refusals.
//!
//! Production router/shared and HC shapes, ragged M, every K%8 scalar-tail
//! residue, all nonempty x4 slot masks, finite extreme operands and extreme HC
//! scales are mandatory. Ordinary packed-quarter fixtures remain separate from
//! five-plane device observations of actual fused HC arithmetic. No CPU claim.
//! Receipts identify the actual loaded module/symbol/object SHA and byte extents.
//! Direct blobs use the recorder-aware launch funnel so receipts observe the
//! successfully submitted tile rather than a preceding singleton reference.
//!
//! ```text
//! HIPFIRE_QWEN4_WIDE_TWIN_BF16=1 cargo run --release -p rdna-compute \
//!     --features lab --example bf16_rows_pm_oracle -- --launcher
//! ```
//! `--hc` and `--router512` are retained as redundant explicit coverage requests;
//! their coverage is always mandatory. Sampled `--quick` qualification is refused.
//! Fixtures retain stable operand addresses through every replay and control.
//! Pool-backed uploads and consuming releases recycle the real owners after each
//! fixture (including HC quarters and the empty-mask refusal), not after a seed.

use rdna_compute::gemv::WideTwinFamily;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::rc::Rc;

type Result<T, E = String> = std::result::Result<T, E>;

const MAXR: usize = 64;
/// Guard floats on each side of every output allocation.
const GUARD: usize = 64;
/// Output prefill: `0xFFFFFFFF` is a NaN no kernel produces.
const SENTINEL: u8 = 0xff;
/// NaN activation rows before and after the `[MAXR, K]` X view.
const PRE_ROWS: usize = 3;
/// NaN-poisoned BF16 rows after each weight matrix.
const PAD_ROWS: usize = 8;
const ABI_BYTES: usize = 96;
const NAN_BF16: u16 = 0x7fc0;
/// Per-row magnitudes cycle.
const AMPS: [f32; 8] = [1.0, 1.0 / 1024.0, 1.0 / 16.0, 4.0, 64.0, 0.5, 0.0, 1024.0];

/// Frozen total-R exports; the physical tile remains eight rows.
fn tile_symbol(gpu: &Gpu, hc: bool) -> Result<String> {
    let arch = gpu.arch.as_str();
    if !matches!(arch, "gfx1151" | "gfx1201") {
        return Err(format!(
            "BF16 rowtiles unavailable on exact architecture {arch}"
        ));
    }
    Ok(format!(
        "gemv_bf16_xf32_{}_t8_pm_{arch}",
        if hc { "k4_rows" } else { "rows_x4" }
    ))
}

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// Keep the gate error, but never hide a real owner-release failure.
fn finish_release<T>(result: Result<T>, released: Result<()>) -> Result<T> {
    match (result, released) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(reason), Ok(())) | (Ok(_), Err(reason)) => Err(reason),
        (Err(reason), Err(release)) => Err(format!("{reason}; release failed: {release}")),
    }
}

fn release_tensors(gpu: &mut Gpu, tensors: impl IntoIterator<Item = GpuTensor>) -> Result<()> {
    let mut result = Ok(());
    for tensor in tensors {
        result = finish_release(result, gpu.free_tensor(tensor).map_err(err));
    }
    result
}

fn release_weights(gpu: &mut Gpu, weights: impl IntoIterator<Item = Rc<GpuTensor>>) -> Result<()> {
    let mut result = Ok(());
    for weight in weights {
        let released = match Rc::try_unwrap(weight) {
            Ok(owner) => gpu.free_tensor(owner).map_err(err),
            Err(_) => Err("weight owner still shared at fixture release".into()),
        };
        result = finish_release(result, released);
    }
    result
}

/// Unlike upload_raw, this allocation can reuse prior free_tensor slots.
fn upload_raw_pooled(gpu: &mut Gpu, bytes: &[u8]) -> Result<Rc<GpuTensor>> {
    let tensor = gpu.alloc_tensor(&[bytes.len()], DType::Raw).map_err(err)?;
    if let Err(reason) = gpu.hip.memcpy_htod(&tensor.buf, bytes).map_err(err) {
        return finish_release(Err(reason), gpu.free_tensor(tensor).map_err(err));
    }
    Ok(Rc::new(tensor))
}

fn release_fixture_parts(
    gpu: &mut Gpu,
    ws: Vec<Rc<GpuTensor>>,
    ws_alt: Vec<Rc<GpuTensor>>,
    outs: Vec<Out>,
    x_full: Option<GpuTensor>,
) -> Result<()> {
    let weights = release_weights(gpu, ws.into_iter().chain(ws_alt));
    let tensors = release_tensors(gpu, outs.into_iter().map(|out| out.full).chain(x_full));
    finish_release(weights, tensors)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// One BF16 weight matrix `[m, k]` (little-endian u16) followed by [`PAD_ROWS`]
/// NaN rows. Row kinds: ordinary mixed exponents; `+-0` only rows; subnormal-heavy
/// rows; in a `wide` fixture also large and tiny exponent rows.
fn gen_weights(m: usize, k: usize, wide: bool, rng: &mut Rng) -> Vec<u8> {
    let mut words: Vec<u16> = Vec::with_capacity((m + PAD_ROWS) * k);
    for r in 0..m {
        for _ in 0..k {
            let sign = (rng.below(2) as u16) << 15;
            let w = if r % 17 == 3 {
                // signed zeros only
                sign
            } else if r % 13 == 5 {
                // subnormal-heavy
                match rng.below(4) {
                    0 => sign,
                    1 => sign | (0x40 + rng.below(0x40) as u16),
                    _ => sign | (1 + rng.below(0x7f) as u16),
                }
            } else {
                match rng.below(64) {
                    0 => 0x0000,
                    1 => 0x8000,
                    2 => sign | (1 + rng.below(0x7f) as u16),
                    _ => {
                        let (lo, span) = if wide {
                            match r % 3 {
                                0 => (127 - 30, 61),
                                1 => (127 + 10, 20),
                                _ => (127 - 40, 20),
                            }
                        } else {
                            (127 - 12 + (r as u64 % 5), 21)
                        };
                        sign | (((lo + rng.below(span)) as u16) << 7) | rng.below(0x80) as u16
                    }
                }
            };
            words.push(w);
        }
    }
    words.extend(std::iter::repeat(NAN_BF16).take(PAD_ROWS * k));
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// `[MAXR][k]` activations. Row magnitude cycles through [`AMPS`]; row kinds add
/// alternating signs (`r % 11 == 3`), `+-2^20` cancellation around a small residue
/// (`5`), F32 subnormals with `+-0` (`7`), all `-0.0` (`9`), alternating `+-0.0` (`10`).
fn gen_x(k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(MAXR * k);
    for r in 0..MAXR {
        let amp = AMPS[r % AMPS.len()];
        for i in 0..k {
            let v = match r % 11 {
                3 => {
                    let m = rng.unit().abs() * amp.max(1.0 / 64.0);
                    if i % 2 == 0 {
                        m
                    } else {
                        -m
                    }
                }
                5 => {
                    if i == k / 3 {
                        5.0
                    } else if i % 2 == 0 {
                        (1u32 << 20) as f32
                    } else {
                        -((1u32 << 20) as f32)
                    }
                }
                7 => match rng.below(8) {
                    0 => 0.0,
                    1 => -0.0,
                    _ => {
                        let sign = (rng.below(2) as u32) << 31;
                        f32::from_bits(sign | (1 + rng.below(0x7f_ffff) as u32))
                    }
                },
                9 => -0.0,
                10 => {
                    if i % 2 == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                _ => rng.unit() * amp,
            };
            x.push(v);
        }
    }
    x
}

/// One output allocation: `GUARD | MAXR * m f32 | GUARD`.
struct Out {
    full: GpuTensor,
    m: usize,
}

impl Out {
    fn new(gpu: &mut Gpu, m: usize) -> Result<Self> {
        let full = gpu
            .zeros(&[2 * GUARD + MAXR * m], DType::F32)
            .map_err(err)?;
        Ok(Self { full, m })
    }
    fn view(&self) -> GpuTensor {
        self.full.sub_offset(GUARD, MAXR * self.m)
    }
    fn bytes(&self) -> usize {
        (2 * GUARD + MAXR * self.m) * 4
    }
    fn reset(&self, gpu: &Gpu) -> Result<()> {
        gpu.hip
            .memset(&self.full.buf, i32::from(SENTINEL), self.bytes())
            .map_err(err)
    }
    fn download(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        gpu.hip.device_synchronize().map_err(err)?;
        let mut v = vec![0u8; self.full.buf.size()];
        gpu.hip.memcpy_dtoh(&mut v, &self.full.buf).map_err(err)?;
        v.truncate(self.bytes());
        Ok(v)
    }
    /// Expected bytes: sentinel everywhere, `reference`'s first `rows` rows in place.
    fn expected(&self, reference: &[u8], rows: usize) -> Vec<u8> {
        let mut e = vec![SENTINEL; self.bytes()];
        let (lo, hi) = (GUARD * 4, GUARD * 4 + rows * self.m * 4);
        e[lo..hi].copy_from_slice(&reference[lo..hi]);
        e
    }
    fn locate(&self, idx: usize) -> String {
        let (lo, hi) = (GUARD * 4, GUARD * 4 + MAXR * self.m * 4);
        if idx < lo {
            format!("head guard byte {idx}")
        } else if idx >= hi {
            format!("tail guard byte {}", idx - hi)
        } else {
            let off = idx - lo;
            let (row, rem) = (off / (self.m * 4), off % (self.m * 4));
            format!("row {row} column {} byte {}", rem / 4, rem % 4)
        }
    }
}

/// One x4 shape: four matrices `ms` (zero slots allowed, at least one non-zero), shared K.
struct Fixture {
    name: String,
    ms: [usize; 4],
    k: usize,
    ws: Vec<Rc<GpuTensor>>,
    /// Same shapes, different bytes: the swapped-weights negative control.
    ws_alt: Vec<Rc<GpuTensor>>,
    outs: Vec<Out>,
    /// `[PRE_ROWS NaN rows | MAXR rows | PRE_ROWS NaN rows] x k`.
    x_full: GpuTensor,
    x_host: Vec<f32>,
}

fn upload_weights(
    gpu: &mut Gpu,
    ms: [usize; 4],
    k: usize,
    wide: bool,
    rng: &mut Rng,
) -> Result<Vec<Rc<GpuTensor>>> {
    let mut ws = Vec::new();
    let result = (|| {
        for &m in &ms {
            // An unused slot still needs a valid pointer (NaN-poisoned, never read).
            let bytes = if m == 0 {
                vec![0xffu8; 256]
            } else {
                gen_weights(m, k, wide, rng)
            };
            ws.push(upload_raw_pooled(gpu, &bytes)?);
        }
        Ok(())
    })();
    match result {
        Ok(()) => Ok(ws),
        Err(reason) => finish_release(Err(reason), release_weights(gpu, ws)),
    }
}

impl Fixture {
    fn new(gpu: &mut Gpu, ms: [usize; 4], k: usize, wide: bool, rng: &mut Rng) -> Result<Self> {
        assert!(ms.iter().any(|&m| m > 0), "a fixture needs one non-zero M");
        let mut ws = Vec::new();
        let mut ws_alt = Vec::new();
        let mut outs = Vec::new();
        let built = (|| {
            ws = upload_weights(gpu, ms, k, wide, rng)?;
            ws_alt = upload_weights(gpu, ms, k, wide, rng)?;
            for &m in &ms {
                outs.push(Out::new(gpu, m)?);
            }
            let x_host = gen_x(k, rng);
            let zeros = vec![0f32; (2 * PRE_ROWS + MAXR) * k];
            let x_full = gpu.upload_f32(&zeros, &[zeros.len()]).map_err(err)?;
            Ok((x_host, x_full))
        })();
        let (x_host, x_full) = match built {
            Ok(parts) => parts,
            Err(reason) => {
                return finish_release(
                    Err(reason),
                    release_fixture_parts(gpu, ws, ws_alt, outs, None),
                );
            }
        };
        let name = format!(
            "bf16 x4 ms={ms:?} k={k}{}",
            if wide { " wide-range" } else { "" }
        );
        let fx = Self {
            name,
            ms,
            k,
            ws,
            ws_alt,
            outs,
            x_full,
            x_host,
        };
        if let Err(reason) = fx.set_x(gpu, &fx.x_host) {
            return finish_release(Err(reason), fx.release(gpu));
        }
        Ok(fx)
    }

    fn release(self, gpu: &mut Gpu) -> Result<()> {
        release_fixture_parts(gpu, self.ws, self.ws_alt, self.outs, Some(self.x_full))
    }

    /// Write `rows` (`[MAXR, k]`) into the stable X allocation between NaN rows.
    fn set_x(&self, gpu: &Gpu, rows: &[f32]) -> Result<()> {
        let nan = f32::NAN.to_bits().to_le_bytes();
        let mut bytes = Vec::with_capacity((2 * PRE_ROWS + MAXR) * self.k * 4);
        for _ in 0..PRE_ROWS * self.k {
            bytes.extend_from_slice(&nan);
        }
        for v in rows {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for _ in 0..PRE_ROWS * self.k {
            bytes.extend_from_slice(&nan);
        }
        gpu.hip.memcpy_htod(&self.x_full.buf, &bytes).map_err(err)
    }

    /// Exact-size view of the first `rows` activation rows.
    fn x_view(&self, rows: usize) -> GpuTensor {
        self.x_full.sub_offset(PRE_ROWS * self.k, rows * self.k)
    }

    fn reset(&self, gpu: &Gpu) -> Result<()> {
        self.outs.iter().try_for_each(|o| o.reset(gpu))
    }

    fn download(&self, gpu: &Gpu) -> Result<Vec<Vec<u8>>> {
        self.outs.iter().map(|o| o.download(gpu)).collect()
    }

    /// The actual singleton decode launch, once per matrix and activation row.
    fn ref_single(&self, gpu: &mut Gpu) -> Result<()> {
        let inputs: Vec<&GpuTensor> = self.ws.iter().map(|w| &**w).collect();
        guarded_launch(gpu, &self.x_full, self.k, MAXR, &inputs, |gpu| {
            let x = self.x_view(MAXR);
            for r in 0..MAXR {
                for i in 0..4 {
                    let (m, k) = (self.ms[i], self.k);
                    if m > 0 {
                        let y = self.outs[i].view().sub_offset(r * m, m);
                        gpu.gemv_bf16_xf32(&self.ws[i], &x.sub_offset(r * k, k), &y, m, k)
                            .map_err(err)?;
                        let symbol = gpu
                            .last_launched_kernel()
                            .ok_or("singleton launch has no route receipt")?;
                        receipt(
                            gpu,
                            symbol,
                            "actual-singleton",
                            1,
                            &format!("matrix={i} row={r} M={m} K={k} output_bytes={}", m * 4),
                        )?;
                    }
                }
            }
            Ok(())
        })
    }

    /// The actual grouped decode launch (`gemv_bf16_xf32_x4`), once per activation row.
    fn ref_x4(&self, gpu: &mut Gpu) -> Result<()> {
        let inputs: Vec<&GpuTensor> = self.ws.iter().map(|w| &**w).collect();
        guarded_launch(gpu, &self.x_full, self.k, MAXR, &inputs, |gpu| {
            let x = self.x_view(MAXR);
            for r in 0..MAXR {
                let ys: Vec<GpuTensor> = (0..4)
                    .map(|i| self.outs[i].view().sub_offset(r * self.ms[i], self.ms[i]))
                    .collect();
                let parts: [(&GpuTensor, &GpuTensor, usize); 4] =
                    std::array::from_fn(|i| (&*self.ws[i], &ys[i], self.ms[i]));
                gpu.gemv_bf16_xf32_x4(parts, &x.sub_offset(r * self.k, self.k), self.k)
                    .map_err(err)?;
                let symbol = gpu
                    .last_launched_kernel()
                    .ok_or("singleton x4 has no route receipt")?;
                receipt(
                    gpu,
                    symbol,
                    "actual-singleton-x4",
                    1,
                    &format!("row={r} M={:?} K={}", self.ms, self.k),
                )?;
            }
            Ok(())
        })
    }

    /// Reference bytes (guards and unused rows still sentinel): the grouped decode
    /// launch, after comparing it with the singleton launches. Bytes that differ
    /// between the two are added to `tally.ref_disagree`.
    fn reference(
        &self,
        gpu: &mut Gpu,
        tally: &mut Tally,
        notes: &mut Vec<String>,
    ) -> Result<Vec<Vec<u8>>> {
        self.reset(gpu)?;
        self.ref_single(gpu)?;
        let by_single = self.download(gpu)?;
        self.reset(gpu)?;
        self.ref_x4(gpu)?;
        let by_x4 = self.download(gpu)?;
        for (kind, actual) in [("single", &by_single), ("x4", &by_x4)] {
            let expected: Vec<Vec<u8>> = self
                .outs
                .iter()
                .zip(actual)
                .map(|(o, bytes)| o.expected(bytes, MAXR))
                .collect();
            if self.compare(actual, &expected).0 != 0 {
                return Err(format!("actual {kind} singleton overwrote output guards"));
            }
        }
        let (bad, first) = self.compare(&by_x4, &by_single);
        tally.ref_disagree += bad;
        if let Some(first) = first {
            note(
                notes,
                format!("decode x4 differs from singleton decode at {first}"),
            );
        }
        Ok(by_x4)
    }

    /// Public rows launcher, including the ordinary singleton dispatch at R=1.
    fn run_launcher(&self, gpu: &mut Gpu, rows: usize) -> Result<()> {
        let x = self.x_view(rows);
        let ys: Vec<GpuTensor> = self.outs.iter().map(|o| o.view()).collect();
        let parts: [(&GpuTensor, &GpuTensor, usize); 4] =
            std::array::from_fn(|i| (&*self.ws[i], &ys[i], self.ms[i]));
        let inputs: Vec<&GpuTensor> = self.ws.iter().map(|w| &**w).collect();
        guarded_launch(gpu, &self.x_full, self.k, rows, &inputs, |gpu| {
            if rows == 1 {
                gpu.gemv_bf16_xf32_x4(parts, &x, self.k).map_err(err)?;
            } else {
                gpu.gemv_bf16_xf32_x4_rows(parts, &x, self.k, rows)
                    .map_err(err)?;
            }
            let symbol = gpu
                .last_launched_kernel()
                .ok_or("rows launcher has no route receipt")?;
            receipt(
                gpu,
                symbol,
                "routed-x4",
                rows,
                &format!("M={:?} K={}", self.ms, self.k),
            )
        })
    }

    /// Direct total-R tile launch with the frozen ABI.
    fn run_direct(&self, gpu: &mut Gpu, cap: usize, rows: usize, alt: bool) -> Result<()> {
        if cap != MAXR || !(1..=MAXR).contains(&rows) {
            return Err("invalid BF16 rowtile extent".into());
        }
        let name = tile_symbol(gpu, false)?;
        load_pm(gpu, &name)?;
        let ws = if alt { &self.ws_alt } else { &self.ws };
        let x = self.x_view(rows);
        let ys: Vec<GpuTensor> = self.outs.iter().map(|o| o.view()).collect();
        let mut kb = hip_bridge::KernargBlob::new();
        for w in ws.iter() {
            kb.push_ptr(w.buf.as_ptr() as *const std::ffi::c_void);
        }
        kb.push_ptr(x.buf.as_ptr() as *const std::ffi::c_void);
        for y in &ys {
            kb.push_ptr(y.buf.as_ptr() as *const std::ffi::c_void);
        }
        for m in self.ms {
            kb.push_i32(m as i32);
        }
        kb.push_i32(self.k as i32);
        kb.push_i32(rows as i32);
        if kb.len() != ABI_BYTES {
            return Err(format!(
                "kernarg blob is {} bytes, the frozen ABI is {ABI_BYTES}",
                kb.len()
            ));
        }
        let blocks: usize = self.ms.iter().sum();
        let inputs: Vec<&GpuTensor> = ws.iter().map(|w| &**w).collect();
        guarded_launch(gpu, &self.x_full, self.k, rows, &inputs, |gpu| {
            gpu.launch_blob_recorded(
                &name,
                [blocks as u32, rows.div_ceil(8) as u32, 1],
                [32, 1, 1],
                0,
                kb.as_mut_slice(),
                Default::default(),
            )
            .map_err(err)?;
            receipt(
                gpu,
                &name,
                "candidate-x4",
                rows,
                &format!(
                    "M={:?} K={} output_bytes={}",
                    self.ms,
                    self.k,
                    rows * self.ms.iter().sum::<usize>() * 4
                ),
            )
        })
    }

    /// Diff every output of `got` against `want`; the first difference is described.
    fn compare(&self, got: &[Vec<u8>], want: &[Vec<u8>]) -> (usize, Option<String>) {
        let mut bad = 0;
        let mut first = None;
        for (i, ((g, w), out)) in got.iter().zip(want).zip(&self.outs).enumerate() {
            let (n, at) = diff(g, w);
            bad += n;
            if let (None, Some(at)) = (&first, at) {
                first = Some(format!(
                    "matrix {i} {} (got {:#04x}, want {:#04x})",
                    out.locate(at),
                    g[at],
                    w[at]
                ));
            }
        }
        (bad, first)
    }

    /// Download and compare against the first `rows` rows of `reference`.
    fn check(
        &self,
        gpu: &Gpu,
        reference: &[Vec<u8>],
        rows: usize,
    ) -> Result<(usize, Option<String>)> {
        let expected: Vec<Vec<u8>> = self
            .outs
            .iter()
            .zip(reference)
            .map(|(o, r)| o.expected(r, rows))
            .collect();
        let got = self.download(gpu)?;
        Ok(self.compare(&got, &expected))
    }
}

/// Load the architecture-specific PM image through the strict runtime loader.
fn load_pm(gpu: &mut Gpu, symbol: &str) -> Result<()> {
    gpu.ensure_row_tile_pm_entry(symbol).map_err(err)
}

fn raw(gpu: &Gpu, tensor: &GpuTensor) -> Result<Vec<u8>> {
    gpu.hip.device_synchronize().map_err(err)?;
    let mut bytes = vec![0; tensor.buf.size()];
    gpu.hip.memcpy_dtoh(&mut bytes, &tensor.buf).map_err(err)?;
    Ok(bytes)
}

fn receipt(gpu: &Gpu, symbol: &str, mode: &str, rows: usize, shape: &str) -> Result<()> {
    if gpu.last_launched_kernel() != Some(symbol) {
        return Err(format!(
            "route receipt expected {symbol}, observed {:?}",
            gpu.last_launched_kernel()
        ));
    }
    let (module, container_sha, sha) = gpu.lab_kernel_device_identity(symbol).map_err(err)?;
    if module.is_empty()
        || sha.len() != 64
        || !sha.bytes().all(|b| b.is_ascii_hexdigit())
        || container_sha.len() != 64
        || !container_sha.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(format!(
            "missing actual loaded object identity for {symbol}"
        ));
    }
    println!("G1 arch={} device={} module={module} symbol={symbol} object_sha256={sha} container_sha256={container_sha} R={rows} mode={mode:?} {shape}",
        gpu.arch.as_str(), gpu.device_id);
    Ok(())
}

/// Poison inactive rows without changing live inputs; check all read-only bytes
/// and restore the stable activation allocation even when the launch fails.
fn guarded_launch(
    gpu: &mut Gpu,
    x: &GpuTensor,
    k: usize,
    rows: usize,
    weights: &[&GpuTensor],
    launch: impl FnOnce(&mut Gpu) -> Result<()>,
) -> Result<()> {
    let original = raw(gpu, x)?;
    let mut poisoned = original.clone();
    poisoned[(PRE_ROWS + rows) * k * 4..(PRE_ROWS + MAXR) * k * 4].fill(0xff);
    let before: Vec<Vec<u8>> = weights.iter().map(|w| raw(gpu, w)).collect::<Result<_>>()?;
    gpu.hip.memcpy_htod(&x.buf, &poisoned).map_err(err)?;
    let result = launch(gpu);
    let intact = (|| {
        if raw(gpu, x)? != poisoned {
            return Err("candidate mutated activation bytes/guards".into());
        }
        for (i, (w, expected)) in weights.iter().zip(&before).enumerate() {
            if raw(gpu, w)? != *expected {
                return Err(format!("candidate mutated weight matrix {i} bytes/padding"));
            }
        }
        Ok(())
    })();
    gpu.hip.device_synchronize().map_err(err)?;
    gpu.hip.memcpy_htod(&x.buf, &original).map_err(err)?;
    result?;
    intact
}

/// `(mismatching byte count, first mismatching index)`.
fn diff(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
    let mut n = 0;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x != y {
            n += 1;
            first.get_or_insert(i);
        }
    }
    (n + a.len().abs_diff(b.len()), first)
}

#[derive(Default)]
struct Tally {
    cases: usize,
    launches: usize,
    bad_bytes: usize,
    ref_disagree: usize,
    bad_symbols: usize,
    negative_control_blind: usize,
}

impl Tally {
    fn state(&self) -> (usize, usize, usize, usize) {
        (
            self.bad_bytes,
            self.ref_disagree,
            self.bad_symbols,
            self.negative_control_blind,
        )
    }
    fn failed(&self) -> bool {
        self.bad_bytes + self.ref_disagree + self.bad_symbols + self.negative_control_blind > 0
    }
}

struct Opts {
    router512: bool,
    launcher: bool,
    hc: bool,
}

fn row_set(max: usize) -> Vec<usize> {
    (1..=max).collect()
}

fn note(notes: &mut Vec<String>, text: String) {
    if notes.len() < 6 {
        notes.push(text);
    }
}

/// Every PM wide direct width (and, with `--launcher`, every public launcher width,
/// asserting the launched PM symbol) against `reference`.
fn sweep(
    gpu: &mut Gpu,
    fx: &Fixture,
    reference: &[Vec<u8>],
    launcher: bool,
    tag: &str,
    tally: &mut Tally,
    notes: &mut Vec<String>,
) -> Result<()> {
    let symbol = tile_symbol(gpu, false)?;
    for cap in [MAXR] {
        for rows in row_set(cap) {
            fx.reset(gpu)?;
            fx.run_direct(gpu, cap, rows, false)?;
            let (bad, first) = fx.check(gpu, reference, rows)?;
            tally.launches += 1;
            tally.bad_bytes += bad;
            if let Some(first) = first {
                note(notes, format!("{tag}{symbol} R={rows}: {first}"));
            }
        }
    }
    if launcher {
        let mut launched_symbols: Vec<String> = Vec::new();
        for rows in row_set(MAXR) {
            fx.reset(gpu)?;
            fx.run_launcher(gpu, rows)?;
            let launched = gpu.last_launched_kernel().unwrap_or("<none>").to_string();
            let (bad, first) = fx.check(gpu, reference, rows)?;
            tally.launches += 1;
            tally.bad_bytes += bad;
            if let Some(first) = first {
                note(
                    notes,
                    format!("{tag}launcher R={rows} ({launched}): {first}"),
                );
            }
            if rows > 8 && launched != symbol {
                tally.bad_symbols += 1;
                note(
                    notes,
                    format!("{tag}x4 launcher R={rows} launched {launched}, expected {symbol}"),
                );
            }
            if !launched_symbols.contains(&launched) {
                launched_symbols.push(launched);
            }
        }
        note(
            notes,
            format!("{tag}x4 launcher symbols {launched_symbols:?}"),
        );
    }
    Ok(())
}

fn verify(gpu: &mut Gpu, fx: &Fixture, opt: &Opts, tally: &mut Tally) -> Result<()> {
    let before = tally.state();
    let mut notes: Vec<String> = Vec::new();

    // 1: the fixture's own X.
    let reference = fx.reference(gpu, tally, &mut notes)?;
    sweep(gpu, fx, &reference, opt.launcher, "", tally, &mut notes)?;

    // 2a: row permutation (rows reversed) through the same buffers.
    let k = fx.k;
    let reversed: Vec<f32> = (0..MAXR)
        .rev()
        .flat_map(|r| fx.x_host[r * k..(r + 1) * k].to_vec())
        .collect();
    fx.set_x(gpu, &reversed)?;
    let reference_rev = fx.reference(gpu, tally, &mut notes)?;
    sweep(
        gpu,
        fx,
        &reference_rev,
        false,
        "reversed X: ",
        tally,
        &mut notes,
    )?;

    // 2b: replay with changed contents at stable addresses (a fresh X in the same
    // allocation). The stale reference must now differ for a full-width launch.
    let mut rng = Rng(0x00ba_d5ee_d00d_f00d
        ^ ((fx.ms[0] as u64) << 8)
        ^ ((fx.ms[3] as u64) << 20)
        ^ k as u64);
    let changed = gen_x(k, &mut rng);
    fx.set_x(gpu, &changed)?;
    let reference_new = fx.reference(gpu, tally, &mut notes)?;
    sweep(
        gpu,
        fx,
        &reference_new,
        false,
        "changed X: ",
        tally,
        &mut notes,
    )?;
    fx.reset(gpu)?;
    fx.run_direct(gpu, MAXR, MAXR, false)?;
    let (stale_bad, _) = fx.check(gpu, &reference, MAXR)?;
    tally.launches += 1;
    if stale_bad == 0 {
        tally.negative_control_blind += 1;
        note(
            &mut notes,
            "negative control blind: changed X reproduced the stale reference".to_string(),
        );
    }

    // 3: different weights must not reproduce the original-weights reference.
    fx.set_x(gpu, &fx.x_host)?;
    fx.reset(gpu)?;
    fx.run_direct(gpu, MAXR, MAXR, true)?;
    let (swapped_bad, _) = fx.check(gpu, &reference, MAXR)?;
    tally.launches += 1;
    if swapped_bad == 0 {
        tally.negative_control_blind += 1;
        note(
            &mut notes,
            "negative control blind: swapped weights reproduced the original reference".to_string(),
        );
    }

    tally.cases += 1;
    let status = if before == tally.state() {
        "ok"
    } else {
        "MISMATCH"
    };
    println!(
        "{:<56} {status}{}",
        fx.name,
        if notes.is_empty() {
            String::new()
        } else {
            format!(" [{}]", notes.join("; "))
        }
    );
    Ok(())
}

/// Frozen HC ABI: W@0 X@8 Y@16 M@24 K@28 scale@32 hc_act@36 R@40,
/// padded to 48 bytes; fixed quarter owners, block128, grid.y=ceil(R/8).
const HC_ABI_BYTES: usize = 48;

/// One HC launch mode. `scale`/`hc_act` are the kernarg values of the direct PM
/// launch; `api` is the `hc_act_scale` argument of the actual singleton
/// `Gpu::gemv_bf16_xf32_k4` that is the reference (`None` is scale 1.0,
/// `hc_act = 0`; `Some(s)` is `hc_act = 1`).
struct HcMode {
    name: &'static str,
    scale: f32,
    hc_act: i32,
    api: Option<f32>,
}

const HC_MODES: [HcMode; 9] = [
    HcMode {
        name: "plain",
        scale: 1.0,
        hc_act: 0,
        api: None,
    },
    HcMode {
        name: "scale .25, hc_act 0",
        scale: 0.25,
        hc_act: 0,
        api: None,
    },
    HcMode {
        name: "hc_act scale .25",
        scale: 0.25,
        hc_act: 1,
        api: Some(0.25),
    },
    HcMode {
        name: "hc_act scale 1",
        scale: 1.0,
        hc_act: 1,
        api: Some(1.0),
    },
    HcMode {
        name: "hc negative scale",
        scale: -1.0,
        hc_act: 1,
        api: Some(-1.0),
    },
    HcMode {
        name: "hc zero scale",
        scale: 0.0,
        hc_act: 1,
        api: Some(0.0),
    },
    HcMode {
        name: "hc tiny scale",
        scale: f32::MIN_POSITIVE,
        hc_act: 1,
        api: Some(f32::MIN_POSITIVE),
    },
    HcMode {
        name: "hc large scale",
        scale: 1.0e20,
        hc_act: 1,
        api: Some(1.0e20),
    },
    HcMode {
        name: "hc extreme scale",
        scale: f32::MAX,
        hc_act: 1,
        api: Some(f32::MAX),
    },
];

/// The HC k4 matrix `[m, k]` (one BF16 weight, `k % 32 == 0`), output `[R, m]`.
struct HcFixture {
    name: String,
    m: usize,
    k: usize,
    w: Rc<GpuTensor>,
    w_alt: Rc<GpuTensor>,
    /// Host bytes of `w` / `w_alt` (`[m, k]` BF16 rows plus the NaN pad), kept for
    /// the independently packed quarter matrices.
    w_bytes: Vec<u8>,
    w_alt_bytes: Vec<u8>,
    out: Out,
    x_full: GpuTensor,
    x_host: Vec<f32>,
}

impl HcFixture {
    fn new(gpu: &mut Gpu, m: usize, k: usize, wide: bool, rng: &mut Rng) -> Result<Self> {
        assert!(m > 0 && k % 32 == 0, "HC k4 needs M > 0 and K % 32 == 0");
        let w_bytes = gen_weights(m, k, wide, rng);
        let w = upload_raw_pooled(gpu, &w_bytes)?;
        let w_alt_bytes = gen_weights(m, k, wide, rng);
        let w_alt = match upload_raw_pooled(gpu, &w_alt_bytes) {
            Ok(owner) => owner,
            Err(reason) => {
                return finish_release(Err(reason), release_weights(gpu, [w]));
            }
        };
        let out = match Out::new(gpu, m) {
            Ok(out) => out,
            Err(reason) => {
                return finish_release(Err(reason), release_weights(gpu, [w, w_alt]));
            }
        };
        let x_host = gen_x(k, rng);
        let zeros = vec![0f32; (2 * PRE_ROWS + MAXR) * k];
        let x_full = match gpu.upload_f32(&zeros, &[zeros.len()]).map_err(err) {
            Ok(owner) => owner,
            Err(reason) => {
                let weights = release_weights(gpu, [w, w_alt]);
                let output = release_tensors(gpu, [out.full]);
                return finish_release(Err(reason), finish_release(weights, output));
            }
        };
        let name = format!(
            "bf16 k4 hc m={m} k={k}{}",
            if wide { " wide-range" } else { "" }
        );
        let fx = Self {
            name,
            m,
            k,
            w,
            w_alt,
            w_bytes,
            w_alt_bytes,
            out,
            x_full,
            x_host,
        };
        if let Err(reason) = fx.set_x(gpu, &fx.x_host) {
            return finish_release(Err(reason), fx.release(gpu));
        }
        Ok(fx)
    }

    fn release(self, gpu: &mut Gpu) -> Result<()> {
        let weights = release_weights(gpu, [self.w, self.w_alt]);
        let tensors = release_tensors(gpu, [self.out.full, self.x_full]);
        finish_release(weights, tensors)
    }

    fn set_x(&self, gpu: &Gpu, rows: &[f32]) -> Result<()> {
        let nan = f32::NAN.to_bits().to_le_bytes();
        let mut bytes = Vec::with_capacity((2 * PRE_ROWS + MAXR) * self.k * 4);
        for _ in 0..PRE_ROWS * self.k {
            bytes.extend_from_slice(&nan);
        }
        for v in rows {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for _ in 0..PRE_ROWS * self.k {
            bytes.extend_from_slice(&nan);
        }
        gpu.hip.memcpy_htod(&self.x_full.buf, &bytes).map_err(err)
    }

    fn x_view(&self, rows: usize) -> GpuTensor {
        self.x_full.sub_offset(PRE_ROWS * self.k, rows * self.k)
    }

    /// Reference bytes of `mode`: the actual singleton `Gpu::gemv_bf16_xf32_k4`
    /// per activation row (guards and unused rows sentinel).
    fn reference(&self, gpu: &mut Gpu, mode: &HcMode) -> Result<Vec<u8>> {
        guarded_launch(
            gpu,
            &self.x_full,
            self.k,
            MAXR,
            &[&self.w, &self.w_alt],
            |gpu| {
                self.out.reset(gpu)?;
                let x = self.x_view(MAXR);
                for r in 0..MAXR {
                    let y = self.out.view().sub_offset(r * self.m, self.m);
                    gpu.gemv_bf16_xf32_k4(
                        &self.w,
                        &x.sub_offset(r * self.k, self.k),
                        &y,
                        self.m,
                        self.k,
                        mode.api,
                    )
                    .map_err(err)?;
                    let symbol = gpu
                        .last_launched_kernel()
                        .ok_or("HC singleton has no route receipt")?;
                    receipt(
                        gpu,
                        symbol,
                        mode.name,
                        1,
                        &format!(
                            "actual-singleton-k4 row={r} M={} K={} scale_bits={:08x}",
                            self.m,
                            self.k,
                            mode.scale.to_bits()
                        ),
                    )?;
                }
                Ok(())
            },
        )?;
        let actual = self.out.download(gpu)?;
        if diff(&actual, &self.out.expected(&actual, MAXR)).0 != 0 {
            return Err("actual HC singleton overwrote output guards".into());
        }
        Ok(actual)
    }

    /// Public HC rows launcher, including singleton dispatch at R=1.
    fn run_launcher(&self, gpu: &mut Gpu, rows: usize, mode: &HcMode) -> Result<()> {
        let x = self.x_view(rows);
        let y = self.out.view();
        guarded_launch(
            gpu,
            &self.x_full,
            self.k,
            rows,
            &[&self.w, &self.w_alt],
            |gpu| {
                gpu.gemv_bf16_xf32_k4_rows(&self.w, &x, &y, self.m, self.k, mode.api, rows)
                    .map_err(err)?;
                let symbol = gpu
                    .last_launched_kernel()
                    .ok_or("HC rows launcher has no route receipt")?;
                receipt(
                    gpu,
                    symbol,
                    mode.name,
                    rows,
                    &format!("routed-k4 M={} K={}", self.m, self.k),
                )
            },
        )
    }

    /// Direct HC PM rowtile launch, ABI48 built here.
    fn run_direct(
        &self,
        gpu: &mut Gpu,
        cap: usize,
        rows: usize,
        mode: &HcMode,
        alt: bool,
    ) -> Result<()> {
        if cap != MAXR || !(1..=MAXR).contains(&rows) {
            return Err("invalid BF16 HC rowtile extent".into());
        }
        let name = tile_symbol(gpu, true)?;
        load_pm(gpu, &name)?;
        let w = if alt { &self.w_alt } else { &self.w };
        let x = self.x_view(rows);
        let y = self.out.view();
        let mut kb = hip_bridge::KernargBlob::new();
        kb.push_ptr(w.buf.as_ptr() as *const std::ffi::c_void);
        kb.push_ptr(x.buf.as_ptr() as *const std::ffi::c_void);
        kb.push_ptr(y.buf.as_ptr() as *const std::ffi::c_void);
        kb.push_i32(self.m as i32);
        kb.push_i32(self.k as i32);
        kb.push_f32(mode.scale);
        kb.push_i32(mode.hc_act);
        kb.push_i32(rows as i32);
        kb.pad_to(8);
        if kb.len() != HC_ABI_BYTES {
            return Err(format!(
                "HC kernarg blob is {} bytes, the frozen ABI is {HC_ABI_BYTES}",
                kb.len()
            ));
        }
        guarded_launch(
            gpu,
            &self.x_full,
            self.k,
            rows,
            &[&self.w, &self.w_alt],
            |gpu| {
                gpu.launch_blob_recorded(
                    &name,
                    [self.m as u32, rows.div_ceil(8) as u32, 1],
                    [128, 1, 1],
                    0,
                    kb.as_mut_slice(),
                    Default::default(),
                )
                .map_err(err)?;
                receipt(
                    gpu,
                    &name,
                    mode.name,
                    rows,
                    &format!(
                        "M={} K={} output_bytes={} scale_bits={:08x} hc_act={}",
                        self.m,
                        self.k,
                        rows * self.m * 4,
                        mode.scale.to_bits(),
                        mode.hc_act
                    ),
                )
            },
        )
    }

    /// Download and compare against the first `rows` rows of `reference`.
    fn check(&self, gpu: &Gpu, reference: &[u8], rows: usize) -> Result<(usize, Option<String>)> {
        let expected = self.out.expected(reference, rows);
        let got = self.out.download(gpu)?;
        let (bad, at) = diff(&got, &expected);
        Ok((
            bad,
            at.map(|at| {
                format!(
                    "{} (got {:#04x}, want {:#04x})",
                    self.out.locate(at),
                    got[at],
                    expected[at]
                )
            }),
        ))
    }

    /// Quarter `q` as an ordinary one-matrix x4 fixture of width `K / 4`: the
    /// columns `[q*K/4, (q+1)*K/4)` of every weight row packed into their own
    /// matrix (plus the NaN pad), the same columns of every X row packed into a
    /// `[MAXR, K/4]` activation, and the slice of `w_alt` for the negative control.
    fn quarter_fixture(&self, gpu: &mut Gpu, q: usize) -> Result<Fixture> {
        let kq = self.k / 4;
        let pack = |bytes: &[u8]| -> Vec<u8> {
            let mut out = Vec::with_capacity((self.m + PAD_ROWS) * kq * 2);
            for r in 0..self.m {
                let o = (r * self.k + q * kq) * 2;
                out.extend_from_slice(&bytes[o..o + kq * 2]);
            }
            out.extend(
                std::iter::repeat(NAN_BF16.to_le_bytes())
                    .take(PAD_ROWS * kq)
                    .flatten(),
            );
            out
        };
        let x: Vec<f32> = (0..MAXR)
            .flat_map(|r| self.x_host[r * self.k + q * kq..r * self.k + (q + 1) * kq].to_vec())
            .collect();
        Fixture::from_data(
            gpu,
            format!("{} quarter {q}", self.name),
            self.m,
            kq,
            &pack(&self.w_bytes),
            &pack(&self.w_alt_bytes),
            x,
        )
    }
}

impl Fixture {
    /// One-matrix fixture (`ms = [m, 0, 0, 0]`) over caller-packed weights.
    fn from_data(
        gpu: &mut Gpu,
        name: String,
        m: usize,
        k: usize,
        w: &[u8],
        w_alt: &[u8],
        x_host: Vec<f32>,
    ) -> Result<Self> {
        let mut ws = Vec::new();
        let mut ws_alt = Vec::new();
        let ms = [m, 0, 0, 0];
        let mut outs = Vec::new();
        let built = (|| {
            for slot in 0..4 {
                let (a, b): (&[u8], &[u8]) = if slot == 0 {
                    (w, w_alt)
                } else {
                    (&[0xffu8; 256], &[0xffu8; 256])
                };
                ws.push(upload_raw_pooled(gpu, a)?);
                ws_alt.push(upload_raw_pooled(gpu, b)?);
            }
            for &mi in &ms {
                outs.push(Out::new(gpu, mi)?);
            }
            let zeros = vec![0f32; (2 * PRE_ROWS + MAXR) * k];
            gpu.upload_f32(&zeros, &[zeros.len()]).map_err(err)
        })();
        let x_full = match built {
            Ok(owner) => owner,
            Err(reason) => {
                return finish_release(
                    Err(reason),
                    release_fixture_parts(gpu, ws, ws_alt, outs, None),
                );
            }
        };
        let fx = Self {
            name,
            ms,
            k,
            ws,
            ws_alt,
            outs,
            x_full,
            x_host,
        };
        if let Err(reason) = fx.set_x(gpu, &fx.x_host) {
            return finish_release(Err(reason), fx.release(gpu));
        }
        Ok(fx)
    }
}

/// Private lab objects: no diagnostic is registered in a production module.
struct HcDiagnostic {
    _pm_module: hip_bridge::Module,
    _c1_module: hip_bridge::Module,
    pm: hip_bridge::Function,
    c1: hip_bridge::Function,
    pm_symbol: String,
    pm_sha: String,
    c1_sha: String,
    c1_container_sha: String,
}

fn diagnostic_sha(bytes: &[u8]) -> String {
    use sha2::Digest;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

impl HcDiagnostic {
    fn load(gpu: &Gpu) -> Result<Self> {
        use peacemaker_isa::{native, Arch};
        let (arch, emitted) = match gpu.arch.as_str() {
            "gfx1151" => (
                Arch::Gfx1151,
                peacemaker_isa::kernels::qwen4_bf16_rows_wide::emit_hc_quarter_diagnostic(
                    Arch::Gfx1151,
                )?,
            ),
            "gfx1201" => (
                Arch::Gfx1201,
                peacemaker_isa::kernels::lane_rows_bf16_gfx1201::emit_hc_quarter_diagnostic()?,
            ),
            other => return Err(format!("HC diagnostic refuses {other}")),
        };
        let image = native::assemble(&emitted.s_text, arch)?;
        let pm_sha = diagnostic_sha(&image);
        println!("G1 HC_DIAGNOSTIC_RESOURCE proof={} block=128 static_lds_bytes=128 ABI=48 production_registry=false",
            serde_json::to_string(&emitted.proof).map_err(err)?);
        let pm_symbol = format!("gemv_bf16_xf32_k4_rows_t8_quarters_pm_{}", gpu.arch);
        let pm_module = gpu.hip.module_load_data(&image).map_err(err)?;
        let pm = gpu
            .hip
            .module_get_function(&pm_module, &pm_symbol)
            .map_err(err)?;
        const SOURCE: &str = include_str!("../../../kernels/src/qwen4_gemv_bf16_xf32.hip");
        const DEFINITION: &str = "extern \"C\" __global__ void gemv_bf16_xf32_k4(";
        const FOLD: &str =
            "    float value = (quarter[0] + quarter[1]) + (quarter[2] + quarter[3]);";
        if diagnostic_sha(SOURCE.as_bytes())
            != "8a4b2c8402369979f8432599150ad24dbbfe337490d16aa893c516608ae6831b"
        {
            return Err(
                "actual C1 diagnostic source hash changed; re-anchor instrumentation".into(),
            );
        }
        if SOURCE.matches(DEFINITION).count() != 1 || SOURCE.matches(FOLD).count() != 1 {
            return Err("C1 diagnostic source anchors are not unique".into());
        }
        let start = SOURCE
            .find(DEFINITION)
            .ok_or("missing actual C1 definition")?;
        let fold = SOURCE.find(FOLD).ok_or("missing actual C1 fold")?;
        let end = SOURCE[start..]
            .find("\n}\n")
            .map(|n| start + n)
            .ok_or("missing actual C1 body terminator")?;
        if !(start < fold && fold < end)
            || !SOURCE[start..fold].contains("    __shared__ float quarter[4];")
            || !SOURCE[start..fold].contains("    if (threadIdx.x != 0) return;\n")
        {
            return Err("actual C1 quarter/fold body changed".into());
        }
        // Preserve the entire actual source and every arithmetic/epilogue byte.
        // Only this export name and four pre-fold observations are changed.
        let source = SOURCE.replacen(DEFINITION,
            "extern \"C\" __global__ void gemv_bf16_xf32_k4_quarters_c1_lab(", 1)
            .replacen(FOLD, &format!(
                "    y[(size_t)64 * M + row] = quarter[0];\n    y[(size_t)128 * M + row] = quarter[1];\n    y[(size_t)192 * M + row] = quarter[2];\n    y[(size_t)256 * M + row] = quarter[3];\n{FOLD}"), 1);
        let c1_symbol = "gemv_bf16_xf32_k4_quarters_c1_lab";
        let mut compiler =
            rdna_compute::KernelCompiler::new(&gpu.arch, String::new()).map_err(err)?;
        let path = compiler
            .compile_for_symbol("bf16_hc_quarters_c1_lab", &source, c1_symbol)
            .map_err(err)?
            .to_owned();
        let container = std::fs::read(path).map_err(err)?;
        let device = rdna_compute::lab_extract_device_bytes(&container, &gpu.arch)
            .ok_or("C1 diagnostic has no actual device code object")?;
        let c1_sha = diagnostic_sha(device);
        let c1_container_sha = diagnostic_sha(&container);
        let c1_module = gpu.hip.module_load_data(&container).map_err(err)?;
        let c1 = gpu
            .hip
            .module_get_function(&c1_module, c1_symbol)
            .map_err(err)?;
        println!("G1 HC_DIAGNOSTIC_SOURCE actual_source_sha256={} transformed_source_sha256={} unique_definition=1 unique_fold=1 transform=rename-export+four-pre-fold-stores arithmetic=actual-source ABI_pm=48 ABI_c1=40 planes=5 plane_stride_rows=64 pm_object_sha256={pm_sha} c1_object_sha256={c1_sha} c1_container_sha256={c1_container_sha}",
            diagnostic_sha(SOURCE.as_bytes()), diagnostic_sha(source.as_bytes()));
        Ok(Self {
            _pm_module: pm_module,
            _c1_module: c1_module,
            pm,
            c1,
            pm_symbol,
            pm_sha,
            c1_sha,
            c1_container_sha,
        })
    }

    fn launch(
        &self,
        gpu: &Gpu,
        fx: &HcFixture,
        out: &Out,
        mode: &HcMode,
        rows: usize,
        singleton: Option<usize>,
        alt: bool,
    ) -> Result<()> {
        let w = if alt { &fx.w_alt } else { &fx.w };
        let row = singleton.unwrap_or(0);
        let x = fx.x_view(rows).sub_offset(
            row * fx.k,
            if singleton.is_some() {
                fx.k
            } else {
                rows * fx.k
            },
        );
        let y = out
            .view()
            .sub_offset(row * fx.m, 5 * MAXR * fx.m - row * fx.m);
        let mut args = hip_bridge::KernargBlob::new();
        args.push_ptr(w.buf.as_ptr() as *const std::ffi::c_void);
        args.push_ptr(x.buf.as_ptr() as *const std::ffi::c_void);
        args.push_ptr(y.buf.as_ptr() as *const std::ffi::c_void);
        args.push_i32(fx.m as i32);
        args.push_i32(fx.k as i32);
        args.push_f32(mode.scale);
        args.push_i32(mode.hc_act);
        if singleton.is_none() {
            args.push_i32(rows as i32);
        }
        args.pad_to(8);
        if args.len()
            != if singleton.is_some() {
                40
            } else {
                HC_ABI_BYTES
            }
        {
            return Err("HC diagnostic actual argument ABI changed".into());
        }
        // SAFETY: both objects' actual argument layouts are reproduced above;
        // the five-plane owner and every operand outlive synchronization.
        unsafe {
            gpu.hip
                .launch_kernel_blob(
                    if singleton.is_some() {
                        &self.c1
                    } else {
                        &self.pm
                    },
                    [
                        fx.m as u32,
                        if singleton.is_some() {
                            1
                        } else {
                            rows.div_ceil(8) as u32
                        },
                        1,
                    ],
                    [128, 1, 1],
                    0,
                    None,
                    args.as_mut_slice(),
                )
                .map_err(err)?;
        }
        gpu.hip.device_synchronize().map_err(err)?;
        println!("G1 arch={} device={} module={} symbol={} object_sha256={} container_sha256={} R={rows} singleton_row={singleton:?} mode={:?} M={} K={} observed_planes=5",
            gpu.arch, gpu.device_id,
            if singleton.is_some() { "bf16_hc_quarters_c1_lab" } else { "bf16_hc_quarters_pm_lab" },
            if singleton.is_some() { "gemv_bf16_xf32_k4_quarters_c1_lab" } else { &self.pm_symbol },
            if singleton.is_some() { &self.c1_sha } else { &self.pm_sha },
            if singleton.is_some() { &self.c1_container_sha } else { &self.pm_sha },
            mode.name, fx.m, fx.k);
        Ok(())
    }
}

fn fused_quarter_gate(
    gpu: &mut Gpu,
    fx: &HcFixture,
    production: &[u8],
    mode: &HcMode,
    tally: &mut Tally,
    diagnostic: &HcDiagnostic,
) -> Result<()> {
    // Out's single extent is five consecutive 64*M planes, sharing one owner.
    let trace = Out::new(gpu, 5 * fx.m)?;
    let result = (|| {
        trace.reset(gpu)?;
        guarded_launch(gpu, &fx.x_full, fx.k, MAXR, &[&fx.w, &fx.w_alt], |gpu| {
            for r in 0..MAXR {
                diagnostic.launch(gpu, fx, &trace, mode, MAXR, Some(r), false)?;
                tally.launches += 1;
            }
            Ok(())
        })?;
        let reference = trace.download(gpu)?;
        let final_lo = GUARD * 4;
        let final_hi = final_lo + MAXR * fx.m * 4;
        tally.ref_disagree += diff(
            &reference[final_lo..final_hi],
            &production[final_lo..final_hi],
        )
        .0;
        // Every reference plane must preserve the same surrounding guards.
        let mut guarded = vec![SENTINEL; trace.bytes()];
        guarded[final_lo..final_lo + 5 * MAXR * fx.m * 4]
            .copy_from_slice(&reference[final_lo..final_lo + 5 * MAXR * fx.m * 4]);
        tally.ref_disagree += diff(&reference, &guarded).0;
        for rows in 1..=MAXR {
            trace.reset(gpu)?;
            guarded_launch(gpu, &fx.x_full, fx.k, rows, &[&fx.w, &fx.w_alt], |gpu| {
                diagnostic.launch(gpu, fx, &trace, mode, rows, None, false)
            })?;
            tally.launches += 1;
            let actual = trace.download(gpu)?;
            let mut expected = vec![SENTINEL; trace.bytes()];
            for plane in 0..5 {
                let lo = final_lo + plane * MAXR * fx.m * 4;
                let hi = lo + rows * fx.m * 4;
                expected[lo..hi].copy_from_slice(&reference[lo..hi]);
            }
            tally.bad_bytes += diff(&actual, &expected).0;
            // Compare the diagnostic final against the actual normal native
            // launch at the identical R/mode, independently of the C1 match.
            fx.out.reset(gpu)?;
            fx.run_direct(gpu, MAXR, rows, mode, false)?;
            tally.launches += 1;
            let native = fx.out.download(gpu)?;
            tally.bad_bytes += diff(
                &actual[final_lo..final_lo + rows * fx.m * 4],
                &native[final_lo..final_lo + rows * fx.m * 4],
            )
            .0;
        }
        trace.reset(gpu)?;
        guarded_launch(gpu, &fx.x_full, fx.k, MAXR, &[&fx.w, &fx.w_alt], |gpu| {
            diagnostic.launch(gpu, fx, &trace, mode, MAXR, None, true)
        })?;
        tally.launches += 1;
        let swapped = trace.download(gpu)?;
        let quarter_lo = final_hi;
        let quarter_hi = quarter_lo + 4 * MAXR * fx.m * 4;
        tally.bad_bytes += swapped[..final_lo]
            .iter()
            .filter(|&&b| b != SENTINEL)
            .count()
            + swapped[quarter_hi..]
                .iter()
                .filter(|&&b| b != SENTINEL)
                .count();
        let negative = diff(
            &swapped[quarter_lo..quarter_hi],
            &reference[quarter_lo..quarter_hi],
        )
        .0;
        if negative == 0 {
            tally.negative_control_blind += 1;
        }
        println!("G1 HC_FUSED_QUARTER_GATE mode={:?} R_range=1..64 planes=5 active_quarter_bytes={} active_final_bytes={} singleton_launches=64 diagnostic_launches=65 production_final_launches=64 quarter_negative_bad_bytes={negative} owner_count=1 release=consuming-real-pool",
            mode.name, 4 * (1..=MAXR).sum::<usize>() * fx.m * 4,
            (1..=MAXR).sum::<usize>() * fx.m * 4);
        Ok(())
    })();
    finish_release(result, gpu.free_tensor(trace.full).map_err(err))
}

/// Every HC PM direct width (and, with `launcher`, every public `k4_rows`
/// launcher width, asserting the launched PM symbol) against `reference`.
#[allow(clippy::too_many_arguments)]
fn sweep_hc(
    gpu: &mut Gpu,
    fx: &HcFixture,
    reference: &[u8],
    mode: &HcMode,
    launcher: bool,
    tag: &str,
    tally: &mut Tally,
    notes: &mut Vec<String>,
    diagnostic: &HcDiagnostic,
) -> Result<()> {
    fused_quarter_gate(gpu, fx, reference, mode, tally, diagnostic)?;
    let symbol = tile_symbol(gpu, true)?;
    for cap in [MAXR] {
        for rows in row_set(cap) {
            fx.out.reset(gpu)?;
            fx.run_direct(gpu, cap, rows, mode, false)?;
            let (bad, first) = fx.check(gpu, reference, rows)?;
            tally.launches += 1;
            tally.bad_bytes += bad;
            if let Some(first) = first {
                note(
                    notes,
                    format!("{tag}{} {symbol} R={rows}: {first}", mode.name),
                );
            }
        }
    }
    if launcher {
        let mut launched_symbols: Vec<String> = Vec::new();
        for rows in row_set(MAXR) {
            fx.out.reset(gpu)?;
            fx.run_launcher(gpu, rows, mode)?;
            let launched = gpu.last_launched_kernel().unwrap_or("<none>").to_string();
            let (bad, first) = fx.check(gpu, reference, rows)?;
            tally.launches += 1;
            tally.bad_bytes += bad;
            if let Some(first) = first {
                note(
                    notes,
                    format!("{tag}{} launcher R={rows} ({launched}): {first}", mode.name),
                );
            }
            if rows > 8 && launched != symbol {
                tally.bad_symbols += 1;
                note(
                    notes,
                    format!("{tag}k4 launcher R={rows} launched {launched}, expected {symbol}"),
                );
            }
            if !launched_symbols.contains(&launched) {
                launched_symbols.push(launched);
            }
        }
        note(
            notes,
            format!("{tag}k4 launcher symbols {launched_symbols:?}"),
        );
    }
    Ok(())
}

/// Additional ordinary packed-quarter gate, distinct from fused_quarter_gate:
/// this does not observe HC internals. Each of the four K/4
/// quarters, packed independently as its own `[M, K/4]` matrix with its own
/// `[R, K/4]` X, runs through the ordinary PM x4 rowtiles at every R1..64
/// and must equal the actual singleton `Gpu::gemv_bf16_xf32` (and grouped
/// `gemv_bf16_xf32_x4`) at local `K/4`, byte for byte.
fn quarter_gate(
    gpu: &mut Gpu,
    fx: &HcFixture,
    tally: &mut Tally,
    notes: &mut Vec<String>,
) -> Result<()> {
    for q in 0..4 {
        let qf = fx.quarter_fixture(gpu, q)?;
        let result = (|| {
            let reference = qf.reference(gpu, tally, notes)?;
            sweep(
                gpu,
                &qf,
                &reference,
                false,
                &format!("quarter {q}: "),
                tally,
                notes,
            )
        })();
        finish_release(result, qf.release(gpu))?;
    }
    Ok(())
}

fn verify_hc(gpu: &mut Gpu, fx: &HcFixture, opt: &Opts, tally: &mut Tally) -> Result<()> {
    let before = tally.state();
    let mut notes: Vec<String> = Vec::new();
    let diagnostic = HcDiagnostic::load(gpu)?;
    // Modes whose activation is not saturating carry the replay/control passes:
    // `plain` (a raw dot) and the `.25` HC activation. The negative controls use
    // `plain` only (a saturated activation could reproduce a different dot).
    const CONTROL_MODES: [usize; 2] = [0, 2];

    // 0: quarter results first (ordinary per-quarter DAG, see `quarter_gate`).
    quarter_gate(gpu, fx, tally, &mut notes)?;

    // 1: the fixture's own X, every mode.
    let mut references: Vec<Vec<u8>> = Vec::new();
    for mode in &HC_MODES {
        references.push(fx.reference(gpu, mode)?);
    }
    for (i, (mode, reference)) in HC_MODES.iter().zip(&references).enumerate() {
        sweep_hc(
            gpu,
            fx,
            reference,
            mode,
            opt.launcher && CONTROL_MODES.contains(&i),
            "",
            tally,
            &mut notes,
            &diagnostic,
        )?;
    }

    // 2a: rows reversed through the same buffers.
    let k = fx.k;
    let reversed: Vec<f32> = (0..MAXR)
        .rev()
        .flat_map(|r| fx.x_host[r * k..(r + 1) * k].to_vec())
        .collect();
    fx.set_x(gpu, &reversed)?;
    for i in CONTROL_MODES {
        let reference = fx.reference(gpu, &HC_MODES[i])?;
        sweep_hc(
            gpu,
            fx,
            &reference,
            &HC_MODES[i],
            false,
            "reversed X: ",
            tally,
            &mut notes,
            &diagnostic,
        )?;
    }

    // 2b: a fresh X in the same allocation; the stale reference must now differ.
    let mut rng = Rng(0x00ba_d5ee_d00d_f00d ^ ((fx.m as u64) << 8) ^ ((fx.k as u64) << 24));
    let changed = gen_x(k, &mut rng);
    fx.set_x(gpu, &changed)?;
    for i in CONTROL_MODES {
        let reference = fx.reference(gpu, &HC_MODES[i])?;
        sweep_hc(
            gpu,
            fx,
            &reference,
            &HC_MODES[i],
            false,
            "changed X: ",
            tally,
            &mut notes,
            &diagnostic,
        )?;
    }
    fx.out.reset(gpu)?;
    fx.run_direct(gpu, MAXR, MAXR, &HC_MODES[0], false)?;
    let (stale_bad, _) = fx.check(gpu, &references[0], MAXR)?;
    tally.launches += 1;
    if stale_bad == 0 {
        tally.negative_control_blind += 1;
        note(
            &mut notes,
            "negative control blind: changed X reproduced the stale reference".to_string(),
        );
    }

    // 3: different weights must not reproduce the original-weights reference.
    fx.set_x(gpu, &fx.x_host)?;
    fx.out.reset(gpu)?;
    fx.run_direct(gpu, MAXR, MAXR, &HC_MODES[0], true)?;
    let (swapped_bad, _) = fx.check(gpu, &references[0], MAXR)?;
    tally.launches += 1;
    if swapped_bad == 0 {
        tally.negative_control_blind += 1;
        note(
            &mut notes,
            "negative control blind: swapped weights reproduced the original reference".to_string(),
        );
    }

    tally.cases += 1;
    let status = if before == tally.state() {
        "ok"
    } else {
        "MISMATCH"
    };
    println!(
        "{:<56} {status}{}",
        fx.name,
        if notes.is_empty() {
            String::new()
        } else {
            format!(" [{}]", notes.join("; "))
        }
    );
    Ok(())
}

/// `(M, K, wide-range weights)`: M tails (odd M leaves half a workgroup), small
/// and production K (`K % 32 == 0`); `--hc` adds the real shapes: the M = 4 gates,
/// the M = 320 low-rank HC down and the M = 2560 down at `K = 10240`.
fn hc_specs(hc: bool) -> Vec<(usize, usize, bool)> {
    let mut v = vec![
        (1, 32, false),
        (2, 32, false),
        (3, 96, false),
        (5, 288, false),
        (7, 2560, false),
        (8, 32, false),
        (9, 1056, false),
        (15, 4096, false),
        (17, 10240, false),
        (33, 2560, false),
        (7, 2560, true),
        (9, 1056, true),
    ];
    if hc {
        v.extend([
            (4, 10240, false),
            (320, 10240, false),
            (2560, 10240, false),
            (4, 10240, true),
        ]);
    }
    v
}

fn hc_fixtures(gpu: &mut Gpu, opt: &Opts, seed_index: u64, tally: &mut Tally) -> Result<()> {
    let mut seed = 0x6263_3136_6863_6b34u64 ^ seed_index;
    let mut next_rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        Rng(seed | 1)
    };
    for (m, k, wide) in hc_specs(opt.hc) {
        let fx = HcFixture::new(gpu, m, k, wide, &mut next_rng())?;
        let result = verify_hc(gpu, &fx, opt, tally);
        finish_release(result, fx.release(gpu))?;
    }
    Ok(())
}

/// M slot patterns (zero slots, odd and tail sizes).
const PATTERNS: [[usize; 4]; 11] = [
    [1, 0, 0, 0],
    [0, 2, 0, 0],
    [3, 0, 7, 0],
    [7, 8, 9, 0],
    [8, 9, 0, 1],
    [9, 8, 7, 3],
    [1, 2, 3, 7],
    [0, 0, 0, 9],
    [16, 17, 31, 0],
    [8, 8, 8, 8],
    [9, 1, 0, 15],
];

/// `(K, pattern index, wide-range weights)`: scalar `K % 8 != 0`, partial-256
/// multiples of eight, then production `K = 2560` over every pattern.
/// Router (512-expert, top-10) device gate: the `--router512` fixture arm
/// keeps the three original x4 projection fixtures (selector + shared gate/up
/// rows) and feeds each fixture's paired slot-0 logits — actual native-PM
/// direct tile vs actual R1 grouped singleton decode — into the native-PM
/// router vs actual R1 router-singleton gate at every logical R=1..64 and
/// both seeds, plus synthetic direct tie/signed-zero/nonfinite controls. Full
/// ID+weight bytes are compared, with guarded inactive inputs/outputs, the
/// device faultword, and tie/signed-zero/nonfinite-policy controls.
/// Round-trip (`round_logits`) and renormalisation (`normalize_topk_prob`)
/// policies are covered. Native rows ride the actual batched path
/// (`route_rows(tokens) > 1`), and the reference is the R1 decode path (the
/// same symbol, identical bytes); `route_rows` identities are asserted. Real
/// owners are released, including failure paths; no CPU router or fake receipts.
const ROUTER_EXPERTS: usize = 512;
const ROUTER_TOPK: usize = 10;
const ROUTER_GUARD: usize = 16;
/// `0xFF` prefill: the router never stores NaN; fault rows store `-1`/`0.0`.
const ROUTER_SENTINEL: u8 = 0xff;
const ROUTER_LAYERS: [usize; 2] = [7, 129];
const ROUTER_ROW_BASES: [usize; 2] = [0, 1000];
const ROUTER_CONTROLS: [(&str, usize, usize); 4] = [
    ("incumbent", 0, 0),
    ("fast", 1, 0),
    ("round", 0, 1),
    ("roundfast", 1, 1),
];
fn router_logits(rng: &mut Rng) -> Vec<f32> {
    // Mirror the `router_fast_is_bit_identical_to_incumbent` tie/nonfinite
    // pattern (proven byte-identical across both symbols); the per-seed `rng`
    // draw below keeps the two gate seeds distinct without touching NaN/Inf.
    let jitter = (rng.next() % 5) as f32 / 1024.0;
    let mut x = Vec::with_capacity(MAXR * ROUTER_EXPERTS);
    for r in 0..MAXR {
        for e in 0..ROUTER_EXPERTS {
            let i = (r * ROUTER_EXPERTS + e) as u64;
            let base = ((i.wrapping_mul(7919) % 4093) as f32 - 2046.0) / 311.0;
            let amp = AMPS[(r + e) % AMPS.len()];
            let mut v: f32 = base * amp.max(1.0 / 64.0) + jitter;
            match r % 6 {
                // Full-width tie rows: every expert equal, so the top-10 cut is
                // entirely tied and the lowest-index policy is observable; the
                // wide tie-back keeps the cut tied even under the small jitter.
                1 => v = 9.5 + jitter,
                2 if e < 140 => v = -7.25 + jitter,
                // Signed-zero policy: alternate -0.0/+0.0; finite rows hold no NaN/Inf.
                0 if e == 0 => v = if r % 2 == 0 { -0.0 } else { 0.0 },
                3 if e % 5 == 0 => v = f32::NEG_INFINITY,
                4 if e == 17 => v = f32::INFINITY,
                5 if e == 300 => v = f32::NAN,
                _ => {}
            }
            x.push(v);
        }
    }
    x
}
/// `(incumbent, expected native)`: the batched wave-per-token symbol only when
/// `route_rows(rows) > 1` and the caller asked for the fast path.
fn router_expected_symbols(gpu: &Gpu, rows: usize, fast: bool) -> (&'static str, &'static str) {
    let batched = gpu.route_rows(rows) > 1 && fast;
    (
        "moe_router_softmax_top10_f32",
        if batched {
            "moe_router_softmax_top10_f32_fast"
        } else {
            "moe_router_softmax_top10_f32"
        },
    )
}
#[allow(clippy::too_many_arguments)]
fn router_launch(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    ids: &GpuTensor,
    weights: &GpuTensor,
    rows: usize,
    normalize: bool,
    round: bool,
    layer: usize,
    row_base: usize,
    mode: &str,
) -> Result<String> {
    gpu.moe_router_softmax_top10_f32(
        logits, ids, weights, rows, normalize, round, layer, row_base,
    )
    .map_err(err)?;
    let symbol = gpu
        .last_launched_kernel()
        .ok_or("router launch has no route receipt")?
        .to_string();
    receipt(
        gpu,
        &symbol,
        mode,
        rows,
        &format!(
            "router512 experts={ROUTER_EXPERTS} topk={ROUTER_TOPK} normalize={} round={} layer={layer} row_base={row_base}",
            normalize as u8,
            round as u8,
        ),
    )?;
    Ok(symbol)
}
fn router_reset_out(gpu: &Gpu, ids_full: &GpuTensor, w_full: &GpuTensor) -> Result<()> {
    let bytes = (2 * ROUTER_GUARD + MAXR * ROUTER_TOPK) * 4;
    gpu.hip
        .memset(&ids_full.buf, i32::from(ROUTER_SENTINEL), bytes)
        .map_err(err)?;
    gpu.hip
        .memset(&w_full.buf, i32::from(ROUTER_SENTINEL), bytes)
        .map_err(err)
}
fn router_download_out(gpu: &Gpu, full: &GpuTensor) -> Result<Vec<u8>> {
    gpu.hip.device_synchronize().map_err(err)?;
    let mut v = vec![0u8; full.buf.size()];
    gpu.hip.memcpy_dtoh(&mut v, &full.buf).map_err(err)?;
    v.truncate((2 * ROUTER_GUARD + MAXR * ROUTER_TOPK) * 4);
    Ok(v)
}
fn router_expected_live(reference: &[u8], rows: usize) -> Vec<u8> {
    let mut e = vec![ROUTER_SENTINEL; (2 * ROUTER_GUARD + MAXR * ROUTER_TOPK) * 4];
    let (lo, hi) = (ROUTER_GUARD * 4, ROUTER_GUARD * 4 + rows * ROUTER_TOPK * 4);
    e[lo..hi].copy_from_slice(&reference[lo..hi]);
    e
}
fn router_row_desc(idx: usize) -> String {
    let (lo, hi) = (ROUTER_GUARD * 4, ROUTER_GUARD * 4 + MAXR * ROUTER_TOPK * 4);
    if idx < lo {
        format!("head guard byte {idx}")
    } else if idx >= hi {
        format!("tail guard byte {}", idx - hi)
    } else {
        let off = idx - lo;
        format!(
            "row {} slot {} byte {}",
            off / (ROUTER_TOPK * 4),
            (off % (ROUTER_TOPK * 4)) / 4,
            off % 4
        )
    }
}
/// A router logit row faults closed iff any expert logit is non-finite.
fn row_is_nonfinite(row: &[f32]) -> bool {
    row.iter().any(|v| !v.is_finite())
}
/// Live `[MAXR, 512]` f32 logits out of a downloaded router-projection output,
/// refusing guard damage.
fn router_f32_from_out(bytes: &[u8], m: usize) -> Result<Vec<f32>> {
    let expect = (2 * GUARD + MAXR * m) * 4;
    if bytes.len() != expect || m != ROUTER_EXPERTS {
        return Err(format!(
            "router projection extent is {} bytes (M={m}), expected {expect}",
            bytes.len()
        ));
    }
    if bytes[..GUARD * 4].iter().any(|&b| b != SENTINEL) {
        return Err("router projection overwrote head guard".into());
    }
    if bytes[GUARD * 4 + MAXR * m * 4..]
        .iter()
        .any(|&b| b != SENTINEL)
    {
        return Err("router projection overwrote tail guard".into());
    }
    Ok(bytes[GUARD * 4..GUARD * 4 + MAXR * m * 4]
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
/// Paired router-projection logits from one original x4 fixture slot 0
/// (`M=512, K=2560`): `(actual native-PM direct tile, actual R1 grouped
/// singleton decode)`, each `[MAXR, 512]` f32. Both launches carry their own
/// route receipts (`run_direct`/`ref_x4`); the two extra launches are tallied.
fn router_projection_pair(
    gpu: &mut Gpu,
    fx: &Fixture,
    tally: &mut Tally,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if fx.ms[0] != ROUTER_EXPERTS || fx.k != 2560 {
        return Err(format!(
            "router projection needs slot-0 M=512 at K=2560, got M={:?} K={}",
            fx.ms, fx.k
        ));
    }
    fx.reset(gpu)?;
    fx.run_direct(gpu, MAXR, MAXR, false)?;
    let native_bytes = fx.outs[0].download(gpu)?;
    fx.reset(gpu)?;
    fx.ref_x4(gpu)?;
    let ref_bytes = fx.outs[0].download(gpu)?;
    tally.launches += 2;
    Ok((
        router_f32_from_out(&native_bytes, fx.outs[0].m)?,
        router_f32_from_out(&ref_bytes, fx.outs[0].m)?,
    ))
}

/// Compare the actual native-PM batched router rows against the actual R1 GPU
/// singleton reference at one R: full ID+weight bytes, guarded inactive
/// inputs/outputs/faultword, for one (normalize, round, layer, row_base).
#[allow(clippy::too_many_arguments)]
fn router_verify_rows(
    gpu: &mut Gpu,
    host_native: &[f32],
    host_ref: &[f32],
    rows: usize,
    normalize: bool,
    round: bool,
    fast_native: bool,
    layer: usize,
    row_base: usize,
    tally: &mut Tally,
    notes: &mut Vec<String>,
) -> Result<()> {
    if !(1..=MAXR).contains(&rows) {
        return Err("invalid router row count".into());
    }
    if gpu.route_rows(1) != 1 {
        return Err(format!(
            "route_rows(1) is {}, expected 1 (R1 decode identity)",
            gpu.route_rows(1)
        ));
    }
    if fast_native && rows > 1 && gpu.route_rows(rows) <= 1 {
        return Err(format!(
            "route_rows({rows}) is {}, expected batched",
            gpu.route_rows(rows)
        ));
    }
    let (incumbent, native_expected) = router_expected_symbols(gpu, rows, fast_native);
    gpu.clear_router_fault();
    if gpu.take_router_fault().is_some() {
        return Err("router fault word not clear before launch".into());
    }
    // Fresh NaN-prefilled device owners per R; every owner below is freed on
    // both the success and the failure path via `finish_release`. Two paired
    // logit pages are uploaded: the native-PM projection rows feed the native
    // router launch, the R1 singleton projection rows feed the R1 reference.
    // Genuine projection rows stay tag-free (fed identically untouched); the
    // tie/signed-zero/nonfinite kinds live on designated synthetic controls.
    let native_live = &host_native[..rows * ROUTER_EXPERTS];
    let ref_live = &host_ref[..rows * ROUTER_EXPERTS];
    let poison_row = vec![f32::NAN; ROUTER_EXPERTS];
    let mut native_page = Vec::with_capacity((2 * PRE_ROWS + MAXR) * ROUTER_EXPERTS);
    for _ in 0..PRE_ROWS * ROUTER_EXPERTS {
        native_page.push(f32::NAN);
    }
    native_page.extend_from_slice(&native_live);
    for _ in rows..MAXR {
        native_page.extend_from_slice(&poison_row);
    }
    for _ in 0..PRE_ROWS * ROUTER_EXPERTS {
        native_page.push(f32::NAN);
    }
    let mut ref_page = Vec::with_capacity((2 * PRE_ROWS + MAXR) * ROUTER_EXPERTS);
    for _ in 0..PRE_ROWS * ROUTER_EXPERTS {
        ref_page.push(f32::NAN);
    }
    ref_page.extend_from_slice(&ref_live);
    for _ in rows..MAXR {
        ref_page.extend_from_slice(&poison_row);
    }
    for _ in 0..PRE_ROWS * ROUTER_EXPERTS {
        ref_page.push(f32::NAN);
    }
    // Per-row fault expectations come from the windowed paired rows
    // themselves (either side non-finite faults closed).
    let fault_here: Vec<bool> = (0..rows)
        .map(|r| {
            row_is_nonfinite(&native_live[r * ROUTER_EXPERTS..(r + 1) * ROUTER_EXPERTS])
                || row_is_nonfinite(&ref_live[r * ROUTER_EXPERTS..(r + 1) * ROUTER_EXPERTS])
        })
        .collect();
    let logits = gpu
        .upload_f32(&native_page, &[native_page.len()])
        .map_err(err)?;
    let result = (|| -> Result<()> {
        // Read-only inputs: exact live prefix plus guards/poison before launch.
        let before = raw(gpu, &logits)?;
        // Native batched path over the live native-projection prefix.
        let live_view = logits.sub_offset(PRE_ROWS * ROUTER_EXPERTS, rows * ROUTER_EXPERTS);
        let ids_full = gpu
            .upload_f32(
                &vec![f32::from_bits(0x7fc0_4321); 2 * ROUTER_GUARD + MAXR * ROUTER_TOPK],
                &[2 * ROUTER_GUARD + MAXR * ROUTER_TOPK],
            )
            .map_err(err)?;
        let w_full = match (|| -> Result<GpuTensor> {
            gpu.upload_f32(
                &vec![f32::from_bits(0x7fc0_1234); 2 * ROUTER_GUARD + MAXR * ROUTER_TOPK],
                &[2 * ROUTER_GUARD + MAXR * ROUTER_TOPK],
            )
            .map_err(err)
        })() {
            Ok(t) => t,
            Err(reason) => {
                return finish_release(Err(reason), gpu.free_tensor(ids_full).map_err(err));
            }
        };
        let result = (|| -> Result<()> {
            let ids_view = ids_full.sub_offset(ROUTER_GUARD, rows * ROUTER_TOPK);
            let w_view = w_full.sub_offset(ROUTER_GUARD, rows * ROUTER_TOPK);
            router_reset_out(gpu, &ids_full, &w_full)?;
            let saved_fast = gpu.flags.qwen4_router_fast_enabled();
            std::sync::Arc::make_mut(&mut gpu.flags).qwen4_router_fast = fast_native;
            let launched = router_launch(
                gpu,
                &live_view,
                &ids_view,
                &w_view,
                rows,
                normalize,
                round,
                layer,
                row_base,
                "native-pm-router",
            )?;
            std::sync::Arc::make_mut(&mut gpu.flags).qwen4_router_fast = saved_fast;
            if launched != native_expected {
                tally.bad_symbols += 1;
                note(
                    notes,
                    format!("R={rows} native launched {launched}, expected {native_expected}"),
                );
            }
            gpu.hip.device_synchronize().map_err(err)?;
            let native_ids = router_download_out(gpu, &ids_full)?;
            let native_w = router_download_out(gpu, &w_full)?;
            let observed = gpu.take_router_fault();
            let any_fault_row = fault_here.iter().any(|&b| b);
            match (any_fault_row, observed) {
                (true, None) => {
                    return Err(format!(
                        "R={rows} non-finite rows produced no faultword record"
                    ));
                }
                (true, Some(f)) => {
                    if f.layer as usize != layer {
                        return Err(format!("fault layer {} != {layer}", f.layer));
                    }
                    let rel = f.row.saturating_sub(row_base as u64) as usize;
                    if rel >= rows || !fault_here[rel] {
                        return Err(format!("fault row {} not a non-finite row", f.row));
                    }
                }
                (false, Some(f)) => {
                    return Err(format!(
                        "finite R={rows} faulted layer {} row {}",
                        f.layer, f.row
                    ));
                }
                (false, None) => {}
            }
            if gpu.take_router_fault().is_some() {
                return Err("a fault take must clear the word".into());
            }
            // Fault rows fail closed: ids -1, weights +0.0; finite rows hold valid ids.
            for r in 0..rows {
                let (lo, hi) = (
                    ROUTER_GUARD * 4 + r * ROUTER_TOPK * 4,
                    ROUTER_GUARD * 4 + (r + 1) * ROUTER_TOPK * 4,
                );
                if fault_here[r] {
                    if !native_ids[lo..hi].iter().all(|&b| b == 0xff) {
                        return Err(format!("R={rows} non-finite row {r} ids are not -1"));
                    }
                    if !native_w[lo..hi].iter().all(|&b| b == 0) {
                        return Err(format!("R={rows} non-finite row {r} weights are not 0"));
                    }
                } else {
                    for s in 0..ROUTER_TOPK {
                        let o = ROUTER_GUARD * 4 + (r * ROUTER_TOPK + s) * 4;
                        let id = i32::from_ne_bytes([
                            native_ids[o],
                            native_ids[o + 1],
                            native_ids[o + 2],
                            native_ids[o + 3],
                        ]);
                        if !(0..ROUTER_EXPERTS as i32).contains(&id) {
                            return Err(format!("R={rows} finite row {r} id {id} out of range"));
                        }
                    }
                }
            }
            // Actual R1 GPU reference: same symbol, decode path, one row per launch.
            let ref_ids_full = gpu
                .upload_f32(
                    &vec![f32::from_bits(0x7fc0_4321); 2 * ROUTER_GUARD + MAXR * ROUTER_TOPK],
                    &[2 * ROUTER_GUARD + MAXR * ROUTER_TOPK],
                )
                .map_err(err)?;
            let ref_w_full = match (|| -> Result<GpuTensor> {
                gpu.upload_f32(
                    &vec![f32::from_bits(0x7fc0_1234); 2 * ROUTER_GUARD + MAXR * ROUTER_TOPK],
                    &[2 * ROUTER_GUARD + MAXR * ROUTER_TOPK],
                )
                .map_err(err)
            })() {
                Ok(t) => t,
                Err(reason) => {
                    return finish_release(Err(reason), gpu.free_tensor(ref_ids_full).map_err(err));
                }
            };
            let ref_result = (|| -> Result<()> {
                // Separate R1 page upload: the R1 reference consumes the R1
                // projection rows (same guarded layout), never the native page.
                let ref_logits = gpu.upload_f32(&ref_page, &[ref_page.len()]).map_err(err)?;
                let ref_run = (|| -> Result<()> {
                    // Consensus assertion: the paired projection rows agree
                    // bit-for-bit through the live prefix before routing; both
                    // launches below still run unconditionally.
                    // SAFETY: live, immutable contiguous f32 slices back these
                    // byte views for this scope; u8 accepts every bit pattern,
                    // including NaN payloads, and neither view outlives its owner.
                    let (native_bytes, ref_bytes) = unsafe {
                        (
                            std::slice::from_raw_parts(
                                native_page.as_ptr().cast::<u8>(),
                                std::mem::size_of_val(native_page.as_slice()),
                            ),
                            std::slice::from_raw_parts(
                                ref_page.as_ptr().cast::<u8>(),
                                std::mem::size_of_val(ref_page.as_slice()),
                            ),
                        )
                    };
                    let agree = diff(native_bytes, ref_bytes).0 == 0;
                    if !agree {
                        tally.ref_disagree += diff(
                            &native_bytes[PRE_ROWS * ROUTER_EXPERTS * 4
                                ..(PRE_ROWS + rows) * ROUTER_EXPERTS * 4],
                            &ref_bytes[PRE_ROWS * ROUTER_EXPERTS * 4
                                ..(PRE_ROWS + rows) * ROUTER_EXPERTS * 4],
                        )
                        .0;
                    }
                    router_reset_out(gpu, &ref_ids_full, &ref_w_full)?;
                    let saved = gpu.flags.qwen4_router_fast_enabled();
                    std::sync::Arc::make_mut(&mut gpu.flags).qwen4_router_fast = false;
                    if gpu.route_rows(1) != 1 {
                        return Err("route_rows(1) drifted mid-reference".into());
                    }
                    for r in 0..rows {
                        let lrow =
                            ref_logits.sub_offset((PRE_ROWS + r) * ROUTER_EXPERTS, ROUTER_EXPERTS);
                        let irow =
                            ref_ids_full.sub_offset(ROUTER_GUARD + r * ROUTER_TOPK, ROUTER_TOPK);
                        let wrow =
                            ref_w_full.sub_offset(ROUTER_GUARD + r * ROUTER_TOPK, ROUTER_TOPK);
                        let got = router_launch(
                            gpu,
                            &lrow,
                            &irow,
                            &wrow,
                            1,
                            normalize,
                            round,
                            layer,
                            row_base + r,
                            "actual-singleton-r1",
                        )?;
                        if got != incumbent {
                            tally.bad_symbols += 1;
                            note(
                                notes,
                                format!("R={rows} R1 row {r} launched {got}, expected {incumbent}"),
                            );
                        }
                        gpu.hip.device_synchronize().map_err(err)?;
                        match (fault_here[r], gpu.take_router_fault()) {
                            (true, None) => {
                                return Err(format!(
                                    "R={rows} R1 row {r} produced no faultword record"
                                ));
                            }
                            (true, Some(f)) => {
                                if f.layer as usize != layer || f.row as usize != row_base + r {
                                    return Err(format!(
                                        "R={rows} R1 row {r} fault ({}, {}) mismatch",
                                        f.layer, f.row
                                    ));
                                }
                            }
                            (false, Some(f)) => {
                                return Err(format!(
                                    "R={rows} finite R1 row {r} faulted ({}, {})",
                                    f.layer, f.row
                                ));
                            }
                            (false, None) => {}
                        }
                    }
                    std::sync::Arc::make_mut(&mut gpu.flags).qwen4_router_fast = saved;
                    gpu.hip.device_synchronize().map_err(err)
                })();
                let freed = gpu.free_tensor(ref_logits).map_err(err);
                finish_release(ref_run, freed)?;
                gpu.hip.device_synchronize().map_err(err)?;
                let ref_ids = router_download_out(gpu, &ref_ids_full)?;
                let ref_w = router_download_out(gpu, &ref_w_full)?;
                // Guarded inactive outputs: full ID+weight bytes, live prefix in place.
                let want_ids = router_expected_live(&ref_ids, rows);
                let want_w = router_expected_live(&ref_w, rows);
                let (bad_ids, at_ids) = diff(&native_ids, &want_ids);
                let (bad_w, at_w) = diff(&native_w, &want_w);
                // Exact launch census: 1 native batched + `rows` R1 reference.
                tally.launches += 1 + rows;
                tally.bad_bytes += bad_ids + bad_w;
                if let Some(at) = at_ids {
                    note(
                        notes,
                        format!(
                            "R={rows} norm={} round={} layer={layer} base={row_base} ids differ at {} (got {:#04x}, want {:#04x})",
                            normalize as u8,
                            round as u8,
                            router_row_desc(at),
                            native_ids[at],
                            want_ids[at]
                        ),
                    );
                }
                if let Some(at) = at_w {
                    note(
                        notes,
                        format!(
                            "R={rows} norm={} round={} layer={layer} base={row_base} weights differ at {} (got {:#04x}, want {:#04x})",
                            normalize as u8,
                            round as u8,
                            router_row_desc(at),
                            native_w[at],
                            want_w[at]
                        ),
                    );
                }
                Ok(())
            })();
            let freed = release_tensors(gpu, [ref_ids_full, ref_w_full]);
            finish_release(ref_result, freed)?;
            // Read-only inputs intact: the guarded page matches the pre-launch bytes.
            if raw(gpu, &logits)? != before {
                return Err("router candidate mutated activation bytes/guards".into());
            }
            gpu.hip.device_synchronize().map_err(err)
        })();
        let freed = release_tensors(gpu, [ids_full, w_full]);
        finish_release(result, freed)
    })();
    let freed = gpu.free_tensor(logits).map_err(err);
    finish_release(result, freed)
}
/// Full router gate for one seed: paired router-projection logits from each of
/// the three original `--router512` x4 fixtures plus the synthetic direct
/// tie/signed-zero/nonfinite controls, every R=1..64, both fast/native
/// controls, both normalize policies, both round policies, two layers and two
/// row bases. Native router rows consume the native-PM projection logits; the
/// R1 reference consumes the R1 projection logits.
fn router_gate_from_pair(
    gpu: &mut Gpu,
    pair: &(Vec<f32>, Vec<f32>),
    src: &str,
    tally: &mut Tally,
) -> Result<()> {
    let (host_native, host_ref) = (&pair.0, &pair.1);
    for rows in row_set(MAXR) {
        for &(tag, fast_ctl, round_ctl) in &ROUTER_CONTROLS {
            let fast_native = fast_ctl == 1;
            let round = round_ctl == 1;
            for normalize in [false, true] {
                for &layer in &ROUTER_LAYERS {
                    for &row_base in &ROUTER_ROW_BASES {
                        let mut notes: Vec<String> = Vec::new();
                        let before = tally.state();
                        router_verify_rows(
                            gpu,
                            host_native,
                            host_ref,
                            rows,
                            normalize,
                            round,
                            fast_native,
                            layer,
                            row_base,
                            tally,
                            &mut notes,
                        )?;
                        tally.cases += 1;
                        let status = if before == tally.state() {
                            "ok"
                        } else {
                            "MISMATCH"
                        };
                        println!(
                            "router512 {src} R={rows} {tag} norm={} round={} layer={layer} base={row_base} {status}{}",
                            normalize as u8,
                            round as u8,
                            if notes.is_empty() {
                                String::new()
                            } else {
                                format!(" [{}]", notes.join("; "))
                            }
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
fn router_gate(
    gpu: &mut Gpu,
    fixtures: &[(String, Vec<f32>, Vec<f32>)],
    seed_index: u64,
    tally: &mut Tally,
) -> Result<()> {
    // Designated synthetic controls: seeded `router_logits` (ties,
    // signed-zero, nonfinite per `r % 6`) stamped identically both sides,
    // distinct per gate seed; genuine projection pairs stay tag-free.
    // Rng has 64-bit state: retain the low 64 bits of the "router512" tag.
    let mut rng = Rng(0x6f75_7465_7235_3132 ^ seed_index);
    let synth_rows = router_logits(&mut rng);
    let synth = (synth_rows.clone(), synth_rows);
    router_gate_from_pair(gpu, &synth, "synthetic", tally)?;
    for (src, native, refr) in fixtures {
        router_gate_from_pair(gpu, &(native.clone(), refr.clone()), src, tally)?;
    }
    Ok(())
}

fn fixture_specs() -> Vec<(usize, usize, bool)> {
    let mut v: Vec<(usize, usize, bool)> = vec![
        // scalar domain
        (1, 3, false),
        (3, 6, false),
        (7, 0, false),
        (7, 5, false),
        (9, 8, false),
        (31, 2, false),
        (33, 4, false),
        (255, 9, false),
        (2559, 10, false),
        // K % 8 == 0, partial 256
        (8, 5, false),
        (8, 7, false),
        (16, 6, false),
        (24, 3, false),
        (264, 1, false),
        (264, 10, false),
        (520, 8, false),
        (1000, 4, false),
        (2056, 2, false),
        (2056, 9, false),
    ];
    // All seven scalar remainders, including nontrivial sweeps before the tail.
    v.extend((1..8).map(|r| (256 + r, 5, false)));
    // production K
    v.extend((0..PATTERNS.len()).map(|p| (2560, p, false)));
    // wide-range weights
    v.extend([(2560, 5, true), (1000, 6, true), (7, 3, true)]);
    v
}

fn fixtures(gpu: &mut Gpu, opt: &Opts, seed_index: u64, tally: &mut Tally) -> Result<()> {
    let mut seed = 0x6266_3136_7077_6964u64 ^ seed_index;
    let mut next_rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        Rng(seed | 1)
    };
    for (k, p, wide) in fixture_specs() {
        let fx = Fixture::new(gpu, PATTERNS[p], k, wide, &mut next_rng())?;
        let result = verify(gpu, &fx, opt, tally);
        finish_release(result, fx.release(gpu))?;
    }
    // Every nonempty/live x4 slot combination with differing ragged Mi.
    for mask in 1u8..16 {
        let sizes = [1, 3, 8, 9];
        let ms = std::array::from_fn(|i| if mask & (1 << i) != 0 { sizes[i] } else { 0 });
        let fx = Fixture::new(gpu, ms, 2560, false, &mut next_rng())?;
        let result = verify(gpu, &fx, opt, tally);
        finish_release(result, fx.release(gpu))?;
    }
    if opt.router512 {
        // Original router x4 fixtures (selector + shared gate/up rows) stay:
        // their x4 projection checks run, then the paired slot-0 logits feed
        // the native-PM vs R1 router gate.
        let mut pairs: Vec<(String, Vec<f32>, Vec<f32>)> = Vec::new();
        for ms in [[512, 1, 0, 0], [512, 1, 1024, 1024], [512, 1, 513, 511]] {
            let fx = Fixture::new(gpu, ms, 2560, false, &mut next_rng())?;
            let (native, refr) = match router_projection_pair(gpu, &fx, tally) {
                Ok(pair) => pair,
                Err(reason) => return finish_release(Err(reason), fx.release(gpu)),
            };
            let result = verify(gpu, &fx, opt, tally);
            finish_release(result, fx.release(gpu))?;
            pairs.push((format!("ms={ms:?}"), native, refr));
        }
        router_gate(gpu, &pairs, seed_index, tally)?;
    }
    Ok(())
}

/// The sixteenth x4 live-slot mask is an invalid empty grid: require refusal
/// before writes, not an unobserved "successful" zero-work candidate.
fn empty_mask_gate(gpu: &mut Gpu) -> Result<()> {
    let mut rng = Rng(0x656d_7074_792d_7834);
    let ws = upload_weights(gpu, [0; 4], 2560, false, &mut rng)?;
    let mut outs = Vec::new();
    let built = (|| {
        for _ in 0..4 {
            outs.push(Out::new(gpu, 0)?);
        }
        gpu.upload_f32(
            &vec![f32::NAN; (2 * PRE_ROWS + MAXR) * 2560],
            &[(2 * PRE_ROWS + MAXR) * 2560],
        )
        .map_err(err)
    })();
    let x = match built {
        Ok(owner) => owner,
        Err(reason) => {
            return finish_release(
                Err(reason),
                release_fixture_parts(gpu, ws, Vec::new(), outs, None),
            );
        }
    };
    let result = (|| {
        let weights: Vec<&GpuTensor> = ws.iter().map(|w| &**w).collect();
        for rows in 1..=MAXR {
            for out in &outs {
                out.reset(gpu)?;
            }
            let ys: Vec<GpuTensor> = outs.iter().map(Out::view).collect();
            let parts = std::array::from_fn(|i| (&*ws[i], &ys[i], 0));
            let xv = x.sub_offset(PRE_ROWS * 2560, rows * 2560);
            guarded_launch(gpu, &x, 2560, rows, &weights, |gpu| {
                match gpu.lab_gemv_bf16_xf32_rows_x4_t8(parts, &xv, 2560, rows) {
                    Ok(()) => Err(format!("all-empty x4 mask R={rows} was not refused")),
                    Err(reason) => {
                        println!("G1 control=all-empty-x4 R={rows} expected_refusal={reason}");
                        Ok(())
                    }
                }
            })?;
            for out in &outs {
                if out.download(gpu)? != vec![SENTINEL; out.bytes()] {
                    return Err(format!("all-empty x4 mask R={rows} modified output guards"));
                }
            }
        }
        Ok(())
    })();
    finish_release(
        result,
        release_fixture_parts(gpu, ws, Vec::new(), outs, Some(x)),
    )
}

fn run() -> Result<bool> {
    let mut opt = Opts {
        router512: true,
        launcher: false,
        hc: true,
    };
    for flag in std::env::args().skip(1) {
        match flag.as_str() {
            "--quick" => return Err("--quick is not a G1 qualification: every R1..64 is required".into()),
            "--router512" => opt.router512 = true,
            "--launcher" => opt.launcher = true,
            "--hc" => opt.hc = true,
            other => {
                return Err(format!(
                    "unknown argument {other} (usage: bf16_rows_pm_oracle [--router512] [--hc] [--launcher])"
                ))
            }
        }
    }
    let mut gpu = Gpu::init().map_err(err)?;
    gpu.lane_rows_scope = false;
    empty_mask_gate(&mut gpu)?;
    println!(
        "arch {} (few_row_limit {})",
        gpu.arch.as_str(),
        gpu.few_row_limit()
    );
    if !matches!(gpu.arch.as_str(), "gfx1151" | "gfx1201") {
        return Err(format!(
            "exact gfx1151/gfx1201 only, this device is {}",
            gpu.arch.as_str()
        ));
    }
    if opt.launcher {
        gpu.set_wide_verify_rows(MAXR);
        // The launcher assertions are the point of `--launcher`: fail closed when
        // the wide twin switch is off instead of silently asserting nothing.
        if !gpu.row_tile_twin_enabled(WideTwinFamily::Bf16, MAXR) {
            return Err(format!(
                "--launcher needs {}=1 (the BF16 wide twin switch)",
                WideTwinFamily::Bf16.env_var()
            ));
        }
    }
    for hc in [false, true] {
        let symbol = tile_symbol(&gpu, hc)?;
        load_pm(&mut gpu, &symbol)?;
        let block = if hc { 128 } else { 32 };
        let resident = gpu
            .occupancy_max_active_blocks(&symbol, [block, 1, 1], 0)
            .map_err(err)?;
        let (module, container_sha, sha) = gpu.lab_kernel_device_identity(&symbol).map_err(err)?;
        println!("module={module} symbol={symbol} object_sha256={sha} container_sha256={container_sha} resident_groups={resident} block={block}");
    }
    let mut tally = Tally::default();
    for seed in [0, 0x9e37_79b9_7f4a_7c15] {
        println!("fixture_seed_xor={seed:016x} domain=finite-synthetic cpu_claim=none");
        fixtures(&mut gpu, &opt, seed, &mut tally)?;
        hc_fixtures(&mut gpu, &opt, seed, &mut tally)?;
    }
    println!(
        "{} cases, {} launches: {} bad bytes vs actual decode, {} reference disagreement bytes, {} wrong launcher symbols, {} blind negative controls",
        tally.cases, tally.launches, tally.bad_bytes, tally.ref_disagree, tally.bad_symbols, tally.negative_control_blind
    );
    Ok(!tally.failed())
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("bf16_rows_pm_oracle: byte mismatch");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("bf16_rows_pm_oracle: {e}");
            std::process::exit(2);
        }
    }
}
