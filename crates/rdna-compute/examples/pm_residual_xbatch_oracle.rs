// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Per-column byte oracle and timing for the PM multi-column residual GEMV
//! `gemv_mq4g256v2_residual_xbatch` (gfx1201, B <= 8).
//!
//! Reference: one launch of the frozen hipcc `gemv_mq4g256v2_residual`
//! (inventory source snapshot, SHA-checked, `Gpu::ensure_kernel_public`) per
//! column `b` on `x[b]`, `y[b]`, at the launcher rule (grid M, block 32).
//! Candidate: one launch of the PM object through [`ResidualXbatch`], the
//! frozen launcher `Gpu::gemv_mq4g256v2_residual_xbatch(&w, &x, &y, m, k, b)`.
//! Every output column must be byte-identical to its singleton reference, and
//! the guard bytes around x and y must stay untouched.
//!
//! Inputs are real H2 tensors: the residual weights of the captured `wo`
//! (K=6144) and `down` (K=17408) callsites. Column 0 is the capture's own
//! (x, y); columns 1.. are distinct windows of the pool of every captured
//! real activation vector (residual, gate/up, qkv, qkvza, multirow inputs)
//! and of the captured residual streams (pre and post). Edge shapes re-slice
//! the same real matrices: group counts covering quads 0..2+ with tails 0..3
//! and the real K-3..K, row counts 1..7/9/13/33/65/M-1 (every remainder of the
//! 4-row workgroup, odd final rows), each at B = 1..8.
//!
//! Timing (`timing`): this launcher vs B singleton launches, the non-exact
//! WMMA batched GEMM (`gemm_hfq4g256_residual_mq4v2`) and the hipcc x-batch
//! GEMV, with the weights hot (one LLC-resident copy) and cold (copies rotated
//! per call).
//!
//! ```text
//! pm_residual_xbatch_oracle oracle --label run1     # build with --features lab
//! pm_residual_xbatch_oracle timing --label t1 [--batches 1,2,4,8]
//! ```

use hip_bridge::{HipResult, KernargBlob};
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::ffi::c_void;
use std::path::{Path, PathBuf};

type Result<T, E = String> = std::result::Result<T, E>;

const INVENTORY: &str = "/home/kaden/qcal/release-0.4.2/pm-decode-twins/inventory.json";
const CAPTURES: &str = "/home/kaden/qcal/release-0.4.2/pm-decode-twins/oracle-inputs";
const RESULTS: &str = "/home/kaden/qcal/release-0.4.2/pm-decode-twins/oracle-results/gemv_hfq4g256_residual_xbatch_mq4v2";
const INC_MODULE: &str = "gemv_hfq4g256_residual_mq4v2";
const INC_SYMBOL: &str = "gemv_mq4g256v2_residual";
const SYMBOL: &str = "gemv_mq4g256v2_residual_xbatch";
const GUARD: usize = 4096;
const GUARD_BYTE: u8 = 0xA5;

// ---------------------------------------------------------------------------
// Launcher (frozen signature; Main moves it into rdna-compute on merge)

/// Largest column count of one launch.
pub const RESIDUAL_XBATCH_MAX: usize = 8;
/// Output rows per workgroup; must equal
/// `peacemaker_isa::kernels::pm_decode::gemv_hfq4g256_residual_xbatch_mq4v2::ROWS_PER_WORKGROUP`
/// the object was built with. Grid = `ceil(m / rows)`.
pub const RESIDUAL_XBATCH_ROWS: usize = 4;
/// The PM code object, linked by `peacemaker-isa emit --kernel
/// pm_decode_gemv_hfq4g256_residual_xbatch_mq4v2`.
const RESIDUAL_XBATCH_CO: &[u8] =
    include_bytes!("../../../kernels/pm-decode/gfx1201/gemv_hfq4g256_residual_xbatch_mq4v2.co");

thread_local! {
    static PM_OBJECT: RefCell<Option<(hip_bridge::Module, hip_bridge::Function)>> = const { RefCell::new(None) };
}

/// `y[b] += W . x[b]` for b < batch, each column byte-identical to
/// `gemv_mq4g256v2_residual` on (`x[b]`, `y[b]`). x is `[batch][k]`, y is
/// `[batch][m]` (f32, row-major); W is MQ4G256 v2 (`m` rows of `k/256`
/// 136-byte groups). Exact gfx1201 only.
pub trait ResidualXbatch {
    fn gemv_mq4g256v2_residual_xbatch(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> HipResult<()>;
}

impl ResidualXbatch for Gpu {
    fn gemv_mq4g256v2_residual_xbatch(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> HipResult<()> {
        let fail = |msg: String| Err(hip_bridge::HipError::new(0, &msg));
        if self.arch != "gfx1201" {
            return fail(format!("{SYMBOL}: exact gfx1201 only, got {}", self.arch));
        }
        if !(1..=RESIDUAL_XBATCH_MAX).contains(&batch)
            || m == 0
            || k == 0
            || k % 256 != 0
            || m > i32::MAX as usize
            || k > i32::MAX as usize
        {
            return fail(format!("{SYMBOL}: batch {batch} (1..={RESIDUAL_XBATCH_MAX}), m {m} > 0, k {k} % 256 == 0 required"));
        }
        PM_OBJECT.with(|slot| -> HipResult<()> {
            let mut slot = slot.borrow_mut();
            if slot.is_none() {
                let module = self.hip.module_load_data(RESIDUAL_XBATCH_CO)?;
                let func = self.hip.module_get_function(&module, SYMBOL)?;
                *slot = Some((module, func));
            }
            let (_, func) = slot.as_ref().expect("loaded");
            let mut blob = KernargBlob::new();
            blob.push_ptr(w.buf.as_ptr() as *const c_void);
            blob.push_ptr(x.buf.as_ptr() as *const c_void);
            blob.push_ptr(y.buf.as_ptr() as *const c_void);
            blob.push_i32(m as i32);
            blob.push_i32(k as i32);
            blob.push_i32(batch as i32);
            blob.pad_to(16);
            // SAFETY: the blob follows the object's ABI (A/x/y at 0/8/16,
            // M/K/B at 24/28/32) and every pointer is a live device buffer.
            let grid = m.div_ceil(RESIDUAL_XBATCH_ROWS) as u32;
            unsafe {
                self.hip.launch_kernel_blob(
                    func,
                    [grid, 1, 1],
                    [32, 1, 1],
                    0,
                    None,
                    blob.as_mut_slice(),
                )
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn unhex(s: &str) -> Result<Vec<u8>> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(err))
        .collect()
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(err)
}

/// A SHA-checked pre-launch argument of a capture.
fn capture_arg(dir: &Path, off: usize, tag: &str) -> Result<Vec<u8>> {
    let meta = read_json(&dir.join("meta.json"))?;
    let name = format!("arg{off}.{tag}.bin");
    let bytes = std::fs::read(dir.join(&name)).map_err(err)?;
    let want = meta["files"][&name]["sha256"]
        .as_str()
        .ok_or(format!("{}: no hash for {name}", dir.display()))?;
    if sha256_hex(&bytes) != want {
        return Err(format!("{}/{name}: sha256 mismatch", dir.display()));
    }
    Ok(bytes)
}

struct Capture {
    dir: PathBuf,
    w: Vec<u8>,
    x: Vec<u8>,
    y: Vec<u8>,
    m: usize,
    k: usize,
}

fn residual_capture(occ: &str) -> Result<Capture> {
    let dir = Path::new(CAPTURES)
        .join("gemv_mq4g256v2_residual")
        .join(occ);
    let meta = read_json(&dir.join("meta.json"))?;
    if meta["symbol"] != INC_SYMBOL {
        return Err(format!("{}: not a {INC_SYMBOL} capture", dir.display()));
    }
    let ka = unhex(meta["kernarg_hex"].as_str().ok_or("kernarg_hex")?)?;
    let rd = |o: usize| i32::from_le_bytes(ka[o..o + 4].try_into().unwrap()) as usize;
    let (m, k) = (rd(24), rd(28));
    let c = Capture {
        w: capture_arg(&dir, 0, "pre")?,
        x: capture_arg(&dir, 8, "pre")?,
        y: capture_arg(&dir, 16, "pre")?,
        m,
        k,
        dir,
    };
    if c.w.len() != m * (k / 256) * 136 || c.x.len() != k * 4 || c.y.len() != m * 4 {
        return Err(format!(
            "{}: argument sizes disagree with M={m} K={k}",
            c.dir.display()
        ));
    }
    Ok(c)
}

/// Every captured real activation vector, concatenated (f32 bytes).
fn activation_pool() -> Result<Vec<u8>> {
    let mut pool = Vec::new();
    for (module, occ, off) in [
        ("gemv_mq4g256v2_residual", "occ0", 8),
        ("gemv_mq4g256v2_residual", "occ127", 8),
        ("fused_gate_up_mq4g256v2", "occ0", 16),
        ("fused_gate_up_mq4g256v2", "occ63", 16),
        ("fused_qkv_mq4g256v2", "occ0", 24),
        ("fused_qkv_mq4g256v2", "occ15", 24),
        ("fused_qkvza_mq4g256v2", "occ0", 32),
        ("fused_qkvza_mq4g256v2", "occ47", 32),
        ("gemv_mq4g256v2_multirow_r2", "occ0", 8),
    ] {
        pool.extend(capture_arg(
            &Path::new(CAPTURES).join(module).join(occ),
            off,
            "pre",
        )?);
    }
    Ok(pool)
}

/// The captured residual streams (pre and post of both callsites).
fn residual_pool() -> Result<Vec<u8>> {
    let mut pool = Vec::new();
    for occ in ["occ0", "occ127"] {
        let dir = Path::new(CAPTURES)
            .join("gemv_mq4g256v2_residual")
            .join(occ);
        pool.extend(capture_arg(&dir, 16, "pre")?);
        pool.extend(capture_arg(&dir, 16, "post")?);
    }
    Ok(pool)
}

/// `len` f32 of `pool` starting at f32 `start`, wrapping around.
fn window(pool: &[u8], start: usize, len: usize) -> Vec<u8> {
    let n = pool.len() / 4;
    (0..len)
        .flat_map(|i| {
            let j = (start + i) % n;
            pool[j * 4..j * 4 + 4].to_vec()
        })
        .collect()
}

/// The first `groups` groups of each of the first `rows` rows.
fn reslice(w: &[u8], real_groups: usize, rows: usize, groups: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * groups * 136);
    for r in 0..rows {
        out.extend_from_slice(&w[r * real_groups * 136..][..groups * 136]);
    }
    out
}

fn incumbent_source() -> Result<(String, String)> {
    let inv = read_json(Path::new(INVENTORY))?;
    let m = inv["modules"]
        .as_array()
        .ok_or("inventory.modules")?
        .iter()
        .find(|m| m["module"] == INC_MODULE)
        .ok_or(format!("{INC_MODULE} not in inventory"))?;
    let snap = m["source_snapshot"].as_str().ok_or("source_snapshot")?;
    let source = std::fs::read_to_string(snap).map_err(err)?;
    let sha = sha256_hex(source.as_bytes());
    if m["source_sha256"].as_str() != Some(sha.as_str()) {
        return Err(format!("{snap}: sha256 {sha} != inventory"));
    }
    Ok((source, sha))
}

struct Dev {
    t: GpuTensor,
}

impl Dev {
    /// `bytes` framed by guard bytes; the returned view addresses the payload.
    /// `GpuTensor` does not free on drop: every `Dev` ends in [`Dev::free`].
    fn framed(gpu: &mut Gpu, bytes: &[u8]) -> Result<(Self, GpuTensor)> {
        let mut framed = vec![GUARD_BYTE; GUARD + bytes.len() + GUARD];
        framed[GUARD..GUARD + bytes.len()].copy_from_slice(bytes);
        let t = gpu.upload_raw(&framed, &[framed.len()]).map_err(err)?;
        let view = t.sub_offset(GUARD, bytes.len());
        Ok((Dev { t }, view))
    }
    fn download(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        let mut v = vec![0u8; self.t.buf.size()];
        gpu.hip.memcpy_dtoh(&mut v, &self.t.buf).map_err(err)?;
        Ok(v)
    }
    fn free(self, gpu: &mut Gpu) -> Result<()> {
        gpu.free_tensor(self.t).map_err(err)
    }
}

fn guards_intact(framed: &[u8], len: usize) -> bool {
    framed[..GUARD]
        .iter()
        .chain(&framed[GUARD + len..])
        .all(|&b| b == GUARD_BYTE)
}

fn singleton(
    gpu: &Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
) -> Result<()> {
    let mut blob = KernargBlob::new();
    blob.push_ptr(w.buf.as_ptr() as *const c_void);
    blob.push_ptr(x.buf.as_ptr() as *const c_void);
    blob.push_ptr(y.buf.as_ptr() as *const c_void);
    blob.push_i32(m as i32);
    blob.push_i32(k as i32);
    gpu.launch_kernel_blob(
        INC_SYMBOL,
        [m as u32, 1, 1],
        [32, 1, 1],
        0,
        blob.as_mut_slice(),
    )
    .map_err(err)
}

// ---------------------------------------------------------------------------
// Oracle

struct Shape<'a> {
    name: String,
    w: &'a [u8],
    m: usize,
    k: usize,
    xs: Vec<Vec<u8>>,
    ys: Vec<Vec<u8>>,
}

/// One shape at column count `batch`: per-column mismatch bytes vs singleton.
fn check(gpu: &mut Gpu, s: &Shape, batch: usize, flip_weight: bool) -> Result<(usize, Value)> {
    let (m, k) = (s.m, s.k);
    let (wd, w) = Dev::framed(gpu, s.w)?;
    // Reference: one singleton launch per column on its own copies.
    let mut refs = Vec::new();
    for b in 0..batch {
        let (xd, x) = Dev::framed(gpu, &s.xs[b])?;
        let (yd, y) = Dev::framed(gpu, &s.ys[b])?;
        singleton(gpu, &w, &x, &y, m, k)?;
        refs.push((xd, yd));
    }
    // Candidate: one launch over [batch][k] / [batch][m].
    let mut w_cand = s.w.to_vec();
    if flip_weight {
        w_cand[8] ^= 0x10;
    }
    let (wcd, wc) = Dev::framed(gpu, &w_cand)?;
    let xcat: Vec<u8> = s.xs[..batch].concat();
    let ycat: Vec<u8> = s.ys[..batch].concat();
    let (xcd, xc) = Dev::framed(gpu, &xcat)?;
    let (ycd, yc) = Dev::framed(gpu, &ycat)?;
    gpu.gemv_mq4g256v2_residual_xbatch(&wc, &xc, &yc, m, k, batch)
        .map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    let cand_y = ycd.download(gpu)?;
    let cand_x = xcd.download(gpu)?;
    let mut total = 0;
    let mut cols = Vec::new();
    for (b, (xd, yd)) in refs.iter().enumerate() {
        let ry = yd.download(gpu)?;
        let rx = xd.download(gpu)?;
        if !guards_intact(&ry, m * 4)
            || !guards_intact(&rx, k * 4)
            || rx[GUARD..GUARD + k * 4] != s.xs[b][..]
        {
            return Err(format!(
                "{}: singleton reference wrote outside y[{b}]",
                s.name
            ));
        }
        let r = &ry[GUARD..GUARD + m * 4];
        let c = &cand_y[GUARD + b * m * 4..][..m * 4];
        let mism = r.iter().zip(c).filter(|(a, b)| a != b).count();
        let first = r.iter().zip(c).position(|(a, b)| a != b);
        // Non-vacuous: the reference changed y.
        let updated = r.iter().zip(&s.ys[b]).filter(|(a, b)| a != b).count();
        total += mism;
        let mut v =
            json!({"column": b, "mismatch_bytes": mism, "reference_updated_bytes": updated});
        if let Some(i) = first {
            v["first_index"] = json!(i);
        }
        cols.push(v);
    }
    let guards = guards_intact(&cand_y, batch * m * 4) && guards_intact(&cand_x, batch * k * 4);
    let x_intact = cand_x[GUARD..GUARD + batch * k * 4] == xcat[..];
    if !guards || !x_intact {
        total += 1;
    }
    for (xd, yd) in refs {
        xd.free(gpu)?;
        yd.free(gpu)?;
    }
    for d in [wd, wcd, xcd, ycd] {
        d.free(gpu)?;
    }
    Ok((
        total,
        json!({
            "shape": s.name, "M": m, "K": k, "groups": k / 256, "quads": k / 1024, "tail": (k / 256) % 4, "B": batch,
            "mismatch_bytes": total, "candidate_guards_intact": guards, "candidate_x_unchanged": x_intact,
            "columns": cols,
        }),
    ))
}

fn shape_inputs<'a>(
    name: String,
    w: &'a [u8],
    m: usize,
    k: usize,
    own: Option<(&[u8], &[u8])>,
    act: &[u8],
    res: &[u8],
) -> Shape<'a> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for b in 0..RESIDUAL_XBATCH_MAX {
        match own {
            Some((x, y)) if b == 0 => {
                xs.push(x[..k * 4].to_vec());
                ys.push(y[..m * 4].to_vec());
            }
            _ => {
                xs.push(window(act, 6151 * b + 1031, k));
                ys.push(window(res, 2309 * b + 517, m));
            }
        }
    }
    Shape {
        name,
        w,
        m,
        k,
        xs,
        ys,
    }
}

fn oracle(gpu: &mut Gpu) -> Result<(usize, Value)> {
    let act = activation_pool()?;
    let res = residual_pool()?;
    let mut failures = 0;
    let mut real = Vec::new();
    let mut edges = Vec::new();
    let mut controls = Vec::new();
    for occ in ["occ0", "occ127"] {
        let c = residual_capture(occ)?;
        let groups = c.k / 256;
        let s = shape_inputs(
            format!("{occ} real M={} K={}", c.m, c.k),
            &c.w,
            c.m,
            c.k,
            Some((&c.x, &c.y)),
            &act,
            &res,
        );
        for batch in 1..=RESIDUAL_XBATCH_MAX {
            let (n, v) = check(gpu, &s, batch, false)?;
            failures += n;
            real.push(v);
        }
        // Negative control: one perturbed weight byte must be detected.
        let (n, v) = check(gpu, &s, RESIDUAL_XBATCH_MAX, true)?;
        controls.push(json!({"capture": occ, "perturbed_weight_byte": 8, "xor": "0x10", "detected": n > 0, "result": v}));
        failures += usize::from(n == 0);
        // Edges: every quad/tail combination and odd final rows.
        let mut gs = vec![1, 2, 3, 4, 5, 6, 7, 9, groups - 3, groups - 2, groups - 1];
        gs.retain(|&g| g >= 1 && g <= groups);
        gs.sort_unstable();
        gs.dedup();
        for &g in &gs {
            // Every row remainder of the 4/6/8-row workgroups, odd final rows.
            for rows in [1usize, 2, 3, 4, 5, 6, 7, 9, 13, 33, 65, c.m - 1] {
                let w = reslice(&c.w, groups, rows, g);
                let s = shape_inputs(
                    format!("{occ} edge M={rows} K={}", g * 256),
                    &w,
                    rows,
                    g * 256,
                    None,
                    &act,
                    &res,
                );
                for batch in 1..=RESIDUAL_XBATCH_MAX {
                    let (n, v) = check(gpu, &s, batch, false)?;
                    failures += n;
                    if n > 0 {
                        edges.push(v);
                    } else {
                        edges.push(json!({"shape": s.name, "B": batch, "tail": g % 4, "mismatch_bytes": 0}));
                    }
                }
            }
        }
    }
    let real_mism: usize = real
        .iter()
        .map(|v| v["mismatch_bytes"].as_u64().unwrap_or(0) as usize)
        .sum();
    let edge_mism: usize = edges
        .iter()
        .map(|v| v["mismatch_bytes"].as_u64().unwrap_or(0) as usize)
        .sum();
    Ok((
        failures,
        json!({
            "real": {"launches": real.len(), "mismatch_bytes": real_mism, "results": real},
            "edges": {"launches": edges.len(), "mismatch_bytes": edge_mism, "results": edges},
            "negative_controls": controls,
        }),
    ))
}

// ---------------------------------------------------------------------------
// Timing

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median per-call ms of `f` over `samples` event-timed samples of `reps` calls.
fn time(
    gpu: &mut Gpu,
    samples: usize,
    reps: usize,
    mut f: impl FnMut(&mut Gpu) -> Result<()>,
) -> Result<(f64, Vec<f64>)> {
    for _ in 0..5 {
        f(gpu)?;
    }
    gpu.hip.device_synchronize().map_err(err)?;
    let start = gpu.hip.event_create().map_err(err)?;
    let stop = gpu.hip.event_create().map_err(err)?;
    let mut out = Vec::new();
    for _ in 0..samples {
        gpu.hip.event_record(&start, None).map_err(err)?;
        for _ in 0..reps {
            f(gpu)?;
        }
        gpu.hip.event_record(&stop, None).map_err(err)?;
        gpu.hip.event_synchronize(&stop).map_err(err)?;
        out.push(f64::from(gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?) / reps as f64);
    }
    gpu.hip.event_destroy(start).map_err(err)?;
    gpu.hip.event_destroy(stop).map_err(err)?;
    Ok((median(out.clone()), out))
}

/// The hipcc x-batch residual GEMV (`gemv_mq4g256v2_xbatch_residual`, B <= 4
/// per launch): the existing non-exact column-batched scalar path.
const XBATCH_RESIDUAL_SRC: &str = concat!(
    "#define HIPFIRE_MQ4G256V2_XBATCH_MAX 4\n",
    "#define HIPFIRE_MQ4G256V2_RESIDUAL_EPILOGUE 1\n",
    "#define HIPFIRE_MQ4G256V2_XBATCH_KERNEL gemv_mq4g256v2_xbatch_residual\n",
    include_str!("../../../kernels/src/gemv_mq4g256v2_xbatch.hip")
);

/// Cold regime: enough weight copies, rotated per call, that their bytes
/// exceed the 64 MiB last-level cache several times over (a decode step
/// streams every layer's weights once from HBM).
const COLD_BYTES: usize = 384 << 20;

fn timing(gpu: &mut Gpu, batches: &[usize]) -> Result<Value> {
    gpu.ensure_kernel_public(
        "gemv_mq4g256v2_xbatch_residual",
        XBATCH_RESIDUAL_SRC,
        "gemv_mq4g256v2_xbatch_residual",
    )
    .map_err(err)?;
    let act = activation_pool()?;
    let res = residual_pool()?;
    let mut rows = Vec::new();
    for occ in ["occ0", "occ127"] {
        let c = residual_capture(occ)?;
        let (m, k) = (c.m, c.k);
        let s = shape_inputs(String::new(), &c.w, m, k, Some((&c.x, &c.y)), &act, &res);
        let copies = COLD_BYTES.div_ceil(c.w.len()).max(2);
        let mut ws = Vec::with_capacity(copies);
        for _ in 0..copies {
            ws.push(gpu.upload_raw(&c.w, &[c.w.len()]).map_err(err)?);
        }
        let xcat: Vec<u8> = s.xs.concat();
        let ycat: Vec<u8> = s.ys.concat();
        let xf: Vec<f32> = xcat
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let yf: Vec<f32> = ycat
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let x = gpu
            .upload_f32(&xf, &[RESIDUAL_XBATCH_MAX, k])
            .map_err(err)?;
        let y = gpu
            .upload_f32(&yf, &[RESIDUAL_XBATCH_MAX, m])
            .map_err(err)?;
        for (regime, n_w) in [("hot", 1usize), ("cold", copies)] {
            let ws = &ws[..n_w];
            for &batch in batches {
                let mut i = 0usize;
                let (cand, cand_s) = time(gpu, 15, 20, |g| {
                    i += 1;
                    g.gemv_mq4g256v2_residual_xbatch(&ws[i % n_w], &x, &y, m, k, batch)
                        .map_err(err)
                })?;
                let (single, single_s) = time(gpu, 15, 20, |g| {
                    i += 1;
                    for b in 0..batch {
                        singleton(
                            g,
                            &ws[i % n_w],
                            &x.sub_offset(b * k, k),
                            &y.sub_offset(b * m, m),
                            m,
                            k,
                        )?;
                    }
                    Ok(())
                })?;
                let (wmma, wmma_s) = time(gpu, 15, 20, |g| {
                    i += 1;
                    g.gemm_hfq4g256_residual_mq4v2(&ws[i % n_w], &x, &y, m, k, batch)
                        .map_err(err)
                })?;
                let (xb4, xb4_s) = time(gpu, 15, 20, |g| {
                    i += 1;
                    let w = &ws[i % n_w];
                    for start in (0..batch).step_by(4) {
                        let n = 4.min(batch - start);
                        let mut blob = KernargBlob::new();
                        blob.push_ptr(w.buf.as_ptr() as *const c_void);
                        blob.push_ptr(x.sub_offset(start * k, n * k).buf.as_ptr() as *const c_void);
                        blob.push_ptr(y.sub_offset(start * m, n * m).buf.as_ptr() as *const c_void);
                        blob.push_i32(m as i32);
                        blob.push_i32(k as i32);
                        blob.push_i32(n as i32);
                        blob.pad_to(8);
                        g.launch_kernel_blob(
                            "gemv_mq4g256v2_xbatch_residual",
                            [m as u32, 1, 1],
                            [32, 1, 1],
                            0,
                            blob.as_mut_slice(),
                        )
                        .map_err(err)?;
                    }
                    Ok(())
                })?;
                let weight_gbs = c.w.len() as f64 / (cand * 1e-3) / 1e9;
                println!("{regime} {occ} M={m} K={k} B={batch}: pm_xbatch {:.2} us | {batch}x singleton {:.2} us ({:.2}x) | wmma batched {:.2} us | hipcc xbatch {:.2} us | {weight_gbs:.0} GB/s",
                    cand * 1e3, single * 1e3, single / cand, wmma * 1e3, xb4 * 1e3);
                rows.push(json!({
                    "regime": regime, "weight_copies": n_w,
                    "capture": c.dir.display().to_string(), "M": m, "K": k, "B": batch, "weight_bytes": c.w.len(),
                    "pm_residual_xbatch_ms": cand, "pm_residual_xbatch_samples_ms": cand_s,
                    "singleton_x_B_ms": single, "singleton_x_B_samples_ms": single_s,
                    "speedup_vs_singleton_x_B": single / cand,
                    "nonexact_wmma_batched_gemm_ms": wmma, "nonexact_wmma_batched_gemm_samples_ms": wmma_s,
                    "nonexact_hipcc_xbatch_residual_ms": xb4, "nonexact_hipcc_xbatch_residual_samples_ms": xb4_s,
                    "pm_weight_GBs": weight_gbs,
                }));
            }
        }
        for t in ws.into_iter().chain([x, y]) {
            gpu.free_tensor(t).map_err(err)?;
        }
    }
    Ok(
        json!({"rows": rows, "method": "hip events, null stream, 5 warmup calls, 15 samples x 20 calls, median per call; hot = one weight copy (LLC-resident), cold = copies rotated per call totalling >= 384 MiB"}),
    )
}

// ---------------------------------------------------------------------------

fn main() {
    match run() {
        Ok(0) => {}
        Ok(_) => std::process::exit(1),
        Err(e) => {
            eprintln!("pm_residual_xbatch_oracle: error: {e}");
            std::process::exit(2);
        }
    }
}

fn run() -> Result<usize> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().ok_or(
        "usage: pm_residual_xbatch_oracle <oracle|timing> [--label L] [--batches 1,2,4,8]",
    )?;
    let mut label = cmd.clone();
    let mut batches: Vec<usize> = (1..=RESIDUAL_XBATCH_MAX).collect();
    while let Some(k) = args.next() {
        let v = args.next().ok_or(format!("{k} needs a value"))?;
        match k.as_str() {
            "--label" => label = v,
            "--batches" => {
                batches = v
                    .split(',')
                    .map(|s| s.parse().map_err(err))
                    .collect::<Result<_>>()?
            }
            _ => return Err(format!("unknown option {k}")),
        }
    }
    let (source, source_sha) = incumbent_source()?;
    let mut gpu = Gpu::init().map_err(err)?;
    if gpu.arch != "gfx1201" {
        return Err(format!("gfx1201 only, got {}", gpu.arch));
    }
    gpu.ensure_kernel_public(INC_MODULE, &source, INC_SYMBOL)
        .map_err(err)?;
    let report_dir = Path::new(RESULTS).join(&label);
    std::fs::create_dir_all(&report_dir).map_err(err)?;
    let mut report = json!({
        "schema": "pm-residual-xbatch-oracle-v1",
        "command": std::env::args().collect::<Vec<_>>().join(" "),
        "candidate_symbol": SYMBOL,
        "candidate_object_sha256": sha256_hex(RESIDUAL_XBATCH_CO),
        "reference_symbol": INC_SYMBOL,
        "reference_source_sha256": source_sha,
        "reference_recipe": "inventory source snapshot via Gpu::ensure_kernel_public",
        "arch": gpu.arch,
    });
    let failures = match cmd.as_str() {
        "oracle" => {
            let (n, v) = oracle(&mut gpu)?;
            report["result"] = v;
            n
        }
        "timing" => {
            report["result"] = timing(&mut gpu, &batches)?;
            0
        }
        other => return Err(format!("unknown command {other}")),
    };
    report["pass"] = json!(failures == 0);
    let text = serde_json::to_string_pretty(&report).map_err(err)?;
    let path = report_dir.join("report.json");
    std::fs::write(&path, &text).map_err(err)?;
    if cmd == "oracle" {
        let r = &report["result"];
        println!("real: launches {} mismatch_bytes {} | edges: launches {} mismatch_bytes {} | controls detected {}",
            r["real"]["launches"], r["real"]["mismatch_bytes"], r["edges"]["launches"], r["edges"]["mismatch_bytes"],
            r["negative_controls"].as_array().map(|a| a.iter().all(|c| c["detected"] == true)).unwrap_or(false));
    }
    println!(
        "{}: {}",
        if failures == 0 { "PASS" } else { "FAIL" },
        path.display()
    );
    Ok(failures)
}
