// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! coop27: dense Qwen3.8-27B pp8192 FFN-down GPU+NPU co-op prototype, one layer (`docs/coop-down27.md`).
//!
//! Shape: `down` 8192 tokens x 5120 features x K 17408 with the ADD epilogue `x = RN(x + Y)`. The NPU owns the leading
//! feature window `[0, n)` (fixed column map, part of numeric identity) and runs the measured-exact IEF15 N0 kernel
//! (`peacemaker_npu::kernels::iu4_ief15::Ief15Gemm`, fold-contract §3); the GPU owns `[n, 5120)` on the incumbent route, the
//! certified PM V2B ADD entry `gemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151` (bytes vendored from hipfire wt-land-041m,
//! `native/*.hxaco`), launched on the window by pointer offset (`A + n*row_bytes`, `Y + n*4`), the true row stride
//! `M = 5120` and a row-tile grid of `(5120 - n)/256` tiles (`M` is only the Y stride in that kernel). Output columns of
//! the GPU window are therefore the incumbent's bytes.
//!
//! Per layer (CPU relay; the GPU-driven ring has no IEF15 body):
//! 1. GPU `ief_db`: activation sidecars D/b from the A4 Xq scales (exact `encode_act_scales`);
//! 2. GPU `ief_pack`: Xq codes + D/b -> the IEF15 A stream (replicated per N-wave) in shared host pages;
//! 3. GPU: the incumbent V2B ADD on its window (in flight while the NPU runs);
//! 4. CPU: waits for (2), submits the NPU IEF15 command (A, C in shared pages; B and S/a packed once per layer), waits;
//! 5. GPU `ief_addc`: `x = RN(x + Y)` on the NPU window from the NPU's tiled f32 C.
//!
//! Modes:
//! * `gpu`: GPU-only (no NPU device): windowed-V2B byte identity and time vs width (`--cols`), and the co-op GPU kernels
//!   (`ief_db`, `ief_pack`, `ief_addc`) byte-checked against the CPU oracle for each `--npu-cols` and timed.
//! * `npu`: NPU-only IEF15 down time vs `--npu-cols` on the co-op transport (A/C userptr pages, B shmem), real weights.
//! * `coop`: one `--npu-cols` split; GPU-alone (full V2B ADD), GPU window alone and the co-op layer interleaved
//!   (`--iters` rounds, order rotated), first and last co-op rounds verified byte for byte.
//!
//! Transport: anonymous host pages pinned by the NPU (userptr BO) and `hipHostRegister`ed; the three-way alias check
//! passes before any NPU command. Fabric clock: `railgun::npu::fclk::FabricClockGuard` (default `require`).
mod cpu;
mod fold;
mod gate_up;
mod hfqm;
mod maps;
mod parts;

use npu_tools::coop_gpu;
use peacemaker_npu::kernels::gemm_core::Control;
use peacemaker_npu::kernels::iu4::{Acts, Weights};
use peacemaker_npu::kernels::iu4_ief15::{self, Ief15Gemm};
use railgun::npu::hip_runtime::{
    HipBuffer, HipEvent, HipFunction, HipModule, HipRuntime, HostRegistration,
};
use railgun::npu::{clflush, Bo, Device, HwCtx, ERT_STATE_COMPLETED};
use std::time::Instant;

const PCI: &str = "0000:bf:00.0";
const T: usize = 8192;
const F: usize = 5120;
const K: usize = 17408;
const E: usize = K / 128;
const ROWB: usize = K / 256 * 136;
const XBLK: usize = 72;
const V2B_LDS: u32 = 65536;
const V2B_IMAGE: &[u8] =
    include_bytes!("../../../../../kernels/gemm_mq4g256v2_residual_iu4_pm_v2b_gfx1151.hxaco");
const COOP_CO: &[u8] = include_bytes!("../../../native/coop27/ief_coop_gfx1151.co");
const GLUE2_CO: &[u8] = include_bytes!("../../../native/coop27/ief_glue2_gfx1151.co");
const V2B_ADD: &str = "gemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151";
const POISON: u8 = 0xA5;
const DEFAULT_MODEL: &str = "/home/kaden/pm-wave/g11fa2/models/qwen3.8-27b.mq4-xts";
const USAGE: &str = "usage: coop27 --mode gpu|npu|coop|fold [--proj down|gate_up] --xq A4.bin [--model PATH] [--layer L] [--cols W,..] \
[--npu-cols N,..] [--hid H,..] [--iters I] [--blocks B] [--reps R] [--timeout-ms T] [--pack stream|wb] [--tail sep|wide|fold[,..]] [--drain WGS] [--claim UNITS]";

struct Cfg {
    mode: String,
    proj: String,
    model: String,
    layer: usize,
    xq: String,
    cols: Vec<usize>,
    npu_cols: Vec<usize>,
    hid: Vec<usize>,
    iters: usize,
    blocks: usize,
    reps: usize,
    timeout_ms: u64,
    wb: bool,
    part_tiles: Vec<usize>,
    reg: u32,
    glue2: bool,
    silu2: bool,
    tails: Vec<fold::Tail>,
    drain: usize,
    claim: usize,
}

fn parse(args: &[String]) -> Result<Cfg, String> {
    let mut c = Cfg {
        mode: String::new(),
        proj: "down".into(),
        model: DEFAULT_MODEL.into(),
        layer: 2,
        xq: String::new(),
        cols: vec![5120, 4864, 4608, 4352, 4096, 3840, 3584],
        npu_cols: vec![1024],
        hid: vec![1024, 2048, 3072, 4096],
        iters: 6,
        blocks: 4,
        reps: 4,
        timeout_ms: 5000,
        wb: false,
        part_tiles: vec![32, 16],
        reg: 0,
        glue2: false,
        silu2: false,
        tails: vec![fold::Tail::Sep],
        drain: 20,
        claim: 16,
    };
    let list = |s: Option<&String>| -> Result<Vec<usize>, String> {
        s.ok_or("list needs a value")?
            .split(',')
            .map(|p| {
                p.trim()
                    .parse::<usize>()
                    .map_err(|_| format!("bad integer {p:?}"))
            })
            .collect()
    };
    let num = |s: Option<&String>| -> Result<usize, String> {
        s.ok_or("flag needs a value")?
            .parse::<usize>()
            .map_err(|e| e.to_string())
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--mode" => c.mode = it.next().ok_or("--mode needs a value")?.clone(),
            "--model" => c.model = it.next().ok_or("--model needs a value")?.clone(),
            "--layer" => c.layer = num(it.next())?,
            "--xq" => c.xq = it.next().ok_or("--xq needs a value")?.clone(),
            "--cols" => c.cols = list(it.next())?,
            "--npu-cols" => c.npu_cols = list(it.next())?,
            "--iters" => c.iters = num(it.next())?,
            "--blocks" => c.blocks = num(it.next())?,
            "--reps" => c.reps = num(it.next())?,
            "--timeout-ms" => c.timeout_ms = num(it.next())? as u64,
            "--proj" => c.proj = it.next().ok_or("--proj needs a value")?.clone(),
            "--hid" => c.hid = list(it.next())?,
            "--part-tiles" => c.part_tiles = list(it.next())?,
            "--tail" => c.tails = fold::Tail::parse_list(it.next())?,
            "--drain" => c.drain = num(it.next())?,
            "--claim" => c.claim = num(it.next())?,
            "--silu" => {
                c.silu2 = match it.next().map(String::as_str) {
                    Some("v2") => true,
                    Some("v1") => false,
                    o => return Err(format!("--silu expects v1|v2, got {o:?}")),
                }
            }
            "--glue" => {
                c.glue2 = match it.next().map(String::as_str) {
                    Some("v2") => true,
                    Some("v1") => false,
                    o => return Err(format!("--glue expects v1|v2, got {o:?}")),
                }
            }
            "--reg" => {
                c.reg = match it.next().map(String::as_str) {
                    Some("default") => 0,
                    Some("coarse") => 0x8,
                    o => return Err(format!("--reg expects default|coarse, got {o:?}")),
                }
            }
            "--pack" => {
                c.wb = match it.next().map(String::as_str) {
                    Some("wb") => true,
                    Some("stream") => false,
                    o => return Err(format!("--pack expects stream|wb, got {o:?}")),
                }
            }
            o => return Err(format!("unknown argument {o:?}")),
        }
    }
    if !["gpu", "npu", "coop", "parts", "maps", "cpu", "fold"].contains(&c.mode.as_str()) {
        return Err("--mode must be gpu|npu|coop|parts|maps|cpu|fold".into());
    }
    if c.drain == 0 {
        return Err("--drain must be >= 1".into());
    }
    if c.claim == 0 || c.claim % 8 != 0 {
        return Err("--claim must be a positive multiple of 8 wave-units".into());
    }
    if c.xq.is_empty() && c.mode != "parts" {
        return Err("--xq is required".into());
    }
    for &w in &c.cols {
        if w == 0 || w > F || w % 256 != 0 {
            return Err(format!("--cols {w}: multiple of 256 in 256..={F}"));
        }
    }
    for &n in &c.npu_cols {
        if n == 0 || n >= F || n % 256 != 0 {
            return Err(format!("--npu-cols {n}: multiple of 256 in 256..{F}"));
        }
    }
    if c.iters == 0 || c.blocks == 0 || c.reps == 0 {
        return Err("--iters/--blocks/--reps must be >= 1".into());
    }
    Ok(c)
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
        (h ^ u64::from(x)).wrapping_mul(0x100_0000_01b3)
    })
}
fn put64(k: &mut [u8], at: usize, v: u64) {
    k[at..at + 8].copy_from_slice(&v.to_le_bytes())
}
fn put32(k: &mut [u8], at: usize, v: u32) {
    k[at..at + 4].copy_from_slice(&v.to_le_bytes())
}
fn f32_at(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap())
}
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}
fn fmt_ms(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{x:.3}"))
        .collect::<Vec<_>>()
        .join(",")
}

// ---------------------------------------------------------------------------------------------------------------------
// Inputs

struct Inputs {
    w: Vec<u8>,
    xq: Vec<u8>,
    old: Vec<u8>,
}

/// Captured A4 `block_i4_128 [e][tc][72]` tiled to `[e][T][72]` (token `t` takes captured token `t % tc`).
fn tile_xq(path: &str, e: usize) -> Result<Vec<u8>, String> {
    let raw = std::fs::read(path).map_err(|err| format!("{path}: {err}"))?;
    if raw.is_empty() || raw.len() % (e * XBLK) != 0 {
        return Err(format!("{path}: {} B is not [{e}][T][72]", raw.len()));
    }
    let tc = raw.len() / (e * XBLK);
    if tc > T || T % tc != 0 {
        return Err(format!("{path}: {tc} tokens do not tile {T}"));
    }
    let mut xq = vec![0u8; e * T * XBLK];
    for ep in 0..e {
        for t in 0..T {
            let s = (ep * tc + t % tc) * XBLK;
            xq[(ep * T + t) * XBLK..][..XBLK].copy_from_slice(&raw[s..s + XBLK]);
        }
    }
    println!(
        "inputs: xq {path} ({tc} tokens tiled to {T}, {e} epochs) fnv {:016x}",
        fnv(&xq)
    );
    Ok(xq)
}

fn load_inputs(cfg: &Cfg) -> Result<Inputs, String> {
    let t0 = Instant::now();
    let name = format!(
        "model.language_model.layers.{}.mlp.down_proj.weight",
        cfg.layer
    );
    let (t, w) = hfqm::read_tensor(&cfg.model, &name)?;
    if t.qt != 44 || t.dims != [F, K] || w.len() != F * ROWB {
        return Err(format!(
            "{name}: qt {} dims {:?} bytes {}",
            t.qt,
            t.dims,
            w.len()
        ));
    }
    let xq = tile_xq(&cfg.xq, E)?;
    // Deterministic residual stream in [-4, 4) (the ADD's old x; not captured).
    let mut old = vec![0u8; T * F * 4];
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    for c in old.chunks_exact_mut(4) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let v = ((s >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0) as f32;
        c.copy_from_slice(&v.to_le_bytes());
    }
    println!(
        "inputs: {name} qt {} {:?} fnv {:016x}; old residual fnv {:016x}; load {:.2} s",
        t.qt,
        t.dims,
        fnv(&w),
        fnv(&old),
        t0.elapsed().as_secs_f64()
    );
    Ok(Inputs { w, xq, old })
}

// ---------------------------------------------------------------------------------------------------------------------
// GPU

/// `reg`: `hipHostRegister` flags of every [`Shared`] buffer created from here on (0 default / 0x8 coarse-grained).
struct Gpu {
    rt: HipRuntime,
    _mods: [HipModule; 4],
    add: HipFunction,
    db: HipFunction,
    pack: HipFunction,
    addc: HipFunction,
    copy: HipFunction,
    pack2: HipFunction,
    db2: HipFunction,
    tail: HipFunction,
    reg: std::cell::Cell<u32>,
    fold: fold::FoldFns,
}

impl Gpu {
    fn new() -> Result<Gpu, String> {
        let rt = HipRuntime::load_for_pci(PCI)?;
        let arch = rt.device_arch()?;
        if arch != "gfx1151" {
            return Err(format!("device {PCI} is {arch}, not gfx1151"));
        }
        let m_v2b = rt.load_module(V2B_IMAGE)?;
        let m_coop = rt.load_module(COOP_CO)?;
        let m_ring = rt.load_module(coop_gpu::CODE_OBJECT)?;
        let m_glue2 = rt.load_module(GLUE2_CO)?;
        let (pack2, db2, tail) = (
            m_glue2.function("ief_pack2")?,
            m_glue2.function("ief_db2")?,
            m_glue2.function("ief_tail")?,
        );
        let add = m_v2b.function(V2B_ADD)?;
        let (db, pack, addc) = (
            m_coop.function("ief_db")?,
            m_coop.function("ief_pack")?,
            m_coop.function("ief_addc")?,
        );
        let copy = m_ring.function(coop_gpu::COPY)?;
        let fold = fold::FoldFns::new(&rt)?;
        println!("gpu: {arch} pci {PCI} CUs {}; V2B image fnv {:016x} ({} B), coop kernels fnv {:016x} ({} B), glue v2 fnv {:016x} ({} B), {}",
            rt.compute_units()?, fnv(V2B_IMAGE), V2B_IMAGE.len(), fnv(COOP_CO), COOP_CO.len(), fnv(GLUE2_CO), GLUE2_CO.len(), fold::FoldFns::describe());
        Ok(Gpu {
            rt,
            _mods: [m_v2b, m_coop, m_ring, m_glue2],
            add,
            db,
            pack,
            addc,
            copy,
            pack2,
            db2,
            tail,
            reg: std::cell::Cell::new(0),
            fold,
        })
    }
    /// The whole A-side glue of one call: sidecars D/b and the replicated A stream (codes + tails) for design `g`.
    /// v1: `ief_db` (two passes over Xq) + `ief_pack`; v2: `ief_pack2` (one pass over Xq, also extracts d into `dcomp`),
    /// `ief_db2` (over `dcomp`), `ief_tail`.
    #[allow(clippy::too_many_arguments)]
    fn prep(
        &self,
        v2: bool,
        xq: u64,
        a: u64,
        dc: u64,
        bc: u64,
        dcomp: u64,
        g: &Ief15Gemm,
        wb: bool,
    ) -> Result<(), String> {
        let e = g.epochs();
        if !v2 {
            self.db(xq, dc, bc, e)?;
            return self.pack(xq, a, dc, bc, g, wb);
        }
        let (mw, nw) = g.layout().wave_grid();
        let aseg = (g.design().args[0].bytes / 8) as u32;
        let mut k = [0u8; 48];
        put64(&mut k, 0, xq);
        put64(&mut k, 8, a);
        put64(&mut k, 16, dcomp);
        put32(&mut k, 24, T as u32);
        put32(&mut k, 28, e as u32);
        put32(&mut k, 32, nw as u32);
        put32(&mut k, 36, aseg);
        put32(&mut k, 40, u32::from(wb));
        self.rt.launch_grid(
            &self.pack2,
            [e as u32, mw as u32, 8],
            [128, 1, 1],
            0,
            &mut k,
        )?;
        let mut k2 = [0u8; 32];
        put64(&mut k2, 0, dcomp);
        put64(&mut k2, 8, dc);
        put64(&mut k2, 16, bc);
        put32(&mut k2, 24, T as u32);
        put32(&mut k2, 28, e as u32);
        self.rt
            .launch_grid(&self.db2, [(T / 256) as u32, 1, 1], [256, 1, 1], 0, &mut k2)?;
        let mut k3 = [0u8; 48];
        put64(&mut k3, 0, a);
        put64(&mut k3, 8, dc);
        put64(&mut k3, 16, bc);
        put32(&mut k3, 24, T as u32);
        put32(&mut k3, 28, e as u32);
        put32(&mut k3, 32, nw as u32);
        put32(&mut k3, 36, aseg);
        put32(&mut k3, 40, u32::from(wb));
        self.rt
            .launch_grid(&self.tail, [mw as u32, 8, 1], [32, 1, 1], 0, &mut k3)
    }
    fn up(&self, b: &[u8]) -> Result<HipBuffer, String> {
        let d = self.rt.allocate(b.len(), None)?;
        self.rt.upload(&d, b)?;
        Ok(d)
    }
    fn down(&self, d: &HipBuffer, n: usize) -> Result<Vec<u8>, String> {
        let mut v = vec![0u8; n];
        self.rt.download(d, &mut v)?;
        Ok(v)
    }
    /// Incumbent V2B ADD on features `[col0, col0 + cols)`: `A + col0*row_bytes`, `Y + col0*4`, `M = 5120` (Y row
    /// stride), `(cols/256)` row tiles, raster group `2^gs` (the incumbent's 4 when it divides the tiles).
    fn down_add(&self, w: u64, xq: u64, y: u64, col0: usize, cols: usize) -> Result<(), String> {
        let (mut k, grid) = Self::down_add_args(w, xq, y, col0, cols);
        self.rt
            .launch_grid(&self.add, grid, [512, 1, 1], V2B_LDS, &mut k)
    }
    fn down_add_args(w: u64, xq: u64, y: u64, col0: usize, cols: usize) -> ([u8; 40], [u32; 3]) {
        let tiles = cols / 256;
        let gs = if tiles % 4 == 0 {
            2
        } else if tiles % 2 == 0 {
            1
        } else {
            0
        };
        let mut k = [0u8; 40];
        put64(&mut k, 0, w + (col0 * ROWB) as u64);
        put64(&mut k, 8, xq);
        put64(&mut k, 16, y + (col0 * 4) as u64);
        put32(&mut k, 24, F as u32);
        put32(&mut k, 28, K as u32);
        put32(&mut k, 32, T as u32);
        put32(&mut k, 36, gs);
        (k, [((T / 256) << gs) as u32, (tiles >> gs) as u32, 1])
    }
    /// Workgroup grid (x, y) of the windowed V2B launch on `cols` features (the fold's drain set is its last workgroups).
    fn down_add_grid(cols: usize) -> [u32; 2] {
        let g = Self::down_add_args(0, 0, 0, 0, cols).1;
        [g[0], g[1]]
    }
    /// `ief_add_fold`: the incumbent V2B ADD window `[col0, col0 + cols)` plus the flag-gated addc of the NPU columns
    /// (`e.hbase` = x of feature 0, `e.c[0]` the NPU C, `e.ldy = F`).
    fn add_fold(
        &self,
        w: u64,
        xq: u64,
        y: u64,
        col0: usize,
        cols: usize,
        e: &fold::Ext,
    ) -> Result<(), String> {
        let (k, grid) = Self::down_add_args(w, xq, y, col0, cols);
        let mut k = fold::karg(&k, e);
        self.rt
            .launch_grid(&self.fold.add, grid, [512, 1, 1], V2B_LDS, &mut k)
    }
    /// `ief_addc_wide`: the addc of all `e.j` commands in one static launch.
    fn addc_wide(&self, e: &fold::Ext) -> Result<(), String> {
        let mut k = fold::karg(&[], e);
        self.rt.launch_grid(
            &self.fold.addc_wide,
            [e.j * e.geo.upc / 64, 1, 1],
            [256, 1, 1],
            0,
            &mut k,
        )
    }
    fn db(&self, xq: u64, dc: u64, bc: u64, e: usize) -> Result<(), String> {
        let mut k = [0u8; 32];
        put64(&mut k, 0, xq);
        put64(&mut k, 8, dc);
        put64(&mut k, 16, bc);
        put32(&mut k, 24, T as u32);
        put32(&mut k, 28, e as u32);
        self.rt
            .launch_grid(&self.db, [(T / 256) as u32, 1, 1], [256, 1, 1], 0, &mut k)
    }
    /// `wb`: plain (write-back) stores of the A stream instead of `glc slc dlc` streaming stores.
    fn pack(
        &self,
        xq: u64,
        a: u64,
        dc: u64,
        bc: u64,
        g: &Ief15Gemm,
        wb: bool,
    ) -> Result<(), String> {
        let (mw, nw) = g.layout().wave_grid();
        let aseg = g.design().args[0].bytes / 8;
        let mut k = [0u8; 56];
        put64(&mut k, 0, xq);
        put64(&mut k, 8, a);
        put64(&mut k, 16, dc);
        put64(&mut k, 24, bc);
        put32(&mut k, 32, T as u32);
        put32(&mut k, 36, g.epochs() as u32);
        put32(&mut k, 40, nw as u32);
        put32(&mut k, 44, aseg as u32);
        put32(&mut k, 48, u32::from(wb));
        self.rt.launch_grid(
            &self.pack,
            [g.epochs() as u32, mw as u32, 8],
            [96, 1, 1],
            0,
            &mut k,
        )
    }
    fn addc(&self, c: u64, x: u64, g: &Ief15Gemm) -> Result<(), String> {
        let (mw, nw) = g.layout().wave_grid();
        let mut k = [0u8; 32];
        put64(&mut k, 0, c);
        put64(&mut k, 8, x);
        put32(&mut k, 16, F as u32);
        put32(&mut k, 20, 0);
        put32(&mut k, 24, g.layout().waves() as u32);
        put32(&mut k, 28, nw as u32);
        self.rt.launch_grid(
            &self.addc,
            [2 * nw as u32, 2 * mw as u32, 8],
            [256, 1, 1],
            0,
            &mut k,
        )
    }
    /// `[cpu->gpu, gpu->cpu, npu map == cpu, npu map -> cpu, gpu re-read after a foreign write]` on a fresh 64 KiB shared
    /// buffer (`npu-coop --mode alias`), registered with the current flags; the last verdict catches a stale GPU cache.
    fn alias_check(&self, dev: &Device) -> Result<[bool; 5], String> {
        const BYTES: usize = 64 << 10;
        let sh = Shared::new(self, Some(dev), BYTES)?;
        let npu = sh.npu.as_ref().unwrap();
        let dev_buf = self.rt.allocate(BYTES, None)?;
        let pat = |salt: u32| -> Vec<u8> {
            (0..BYTES as u32)
                .map(|i| {
                    (i.wrapping_mul(2654435761)
                        .wrapping_add(salt.wrapping_mul(0x9e3779b9))
                        >> 13) as u8
                })
                .collect()
        };
        let copy = |dst: u64, src: u64| -> Result<(), String> {
            let mut a = coop_gpu::copy_args(
                dst,
                src,
                (BYTES / 16) as u64,
                (16 * coop_gpu::COPY_BLOCK) as u64,
            );
            self.rt
                .launch(&self.copy, 16, coop_gpu::COPY_BLOCK, &mut a)?;
            self.rt.synchronize()
        };
        let p1 = pat(1);
        sh.write(0, &p1);
        copy(dev_buf.ptr(), sh.gpu())?;
        let cpu_to_gpu = self.down(&dev_buf, BYTES)? == p1;
        let p2 = pat(2);
        self.rt.upload(&dev_buf, &p2)?;
        copy(sh.gpu(), dev_buf.ptr())?;
        let gpu_to_cpu = sh.read(0, BYTES) == p2;
        npu.flush();
        let npu_view = npu.as_slice()[..BYTES] == p2[..];
        let p3 = pat(3);
        unsafe { core::ptr::copy_nonoverlapping(p3.as_ptr(), npu.host, BYTES) };
        npu.flush();
        let npu_to_cpu = sh.read(0, BYTES) == p3;
        copy(dev_buf.ptr(), sh.gpu())?;
        let reread = self.down(&dev_buf, BYTES)? == p3;
        Ok([cpu_to_gpu, gpu_to_cpu, npu_view, npu_to_cpu, reread])
    }
}

/// Anonymous host pages: `hipHostRegister`ed for the GPU and (with a device) pinned by the NPU as a userptr BO; the CPU
/// accesses them directly with clflush. Every line is flushed once after population so no dirty CPU line from the
/// kernel's zero fill can later overwrite device writes. Drop order: HIP registration, NPU BO, munmap.
struct Shared {
    host: *mut u8,
    len: usize,
    gpu: Option<HostRegistration>,
    npu: Option<Bo>,
}

mod libc_mm {
    use core::ffi::c_void;
    extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
    }
    pub const PROT_RW: i32 = 3;
    pub const MAP_ANON_POPULATE: i32 = 0x02 | 0x20 | 0x8000;
}

impl Shared {
    fn new(gpu: &Gpu, dev: Option<&Device>, bytes: usize) -> Result<Shared, String> {
        let len = bytes.div_ceil(4096) * 4096;
        let p = unsafe {
            libc_mm::mmap(
                core::ptr::null_mut(),
                len,
                libc_mm::PROT_RW,
                libc_mm::MAP_ANON_POPULATE,
                -1,
                0,
            )
        };
        if p as usize == usize::MAX {
            return Err(format!(
                "mmap anonymous {len} B: {}",
                std::io::Error::last_os_error()
            ));
        }
        let mut s = Shared {
            host: p as *mut u8,
            len,
            gpu: None,
            npu: None,
        };
        unsafe { clflush(s.host, len) };
        if let Some(d) = dev {
            s.npu = Some(d.userptr_bo(s.host, len)?);
        }
        s.gpu = Some(gpu.rt.register_host_flags(s.host, len, gpu.reg.get())?);
        Ok(s)
    }
    fn gpu(&self) -> u64 {
        self.gpu.as_ref().unwrap().ptr()
    }
    fn bo(&self) -> &Bo {
        self.npu.as_ref().unwrap()
    }
    fn read(&self, at: usize, len: usize) -> Vec<u8> {
        assert!(at + len <= self.len);
        unsafe {
            clflush(self.host.add(at), len);
            core::slice::from_raw_parts(self.host.add(at), len).to_vec()
        }
    }
    fn write(&self, at: usize, src: &[u8]) {
        assert!(at + src.len() <= self.len);
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), self.host.add(at), src.len());
            clflush(self.host.add(at), src.len());
        }
    }
    fn fill(&self, v: u8) {
        unsafe {
            core::ptr::write_bytes(self.host, v, self.len);
            clflush(self.host, self.len);
        }
    }
}
impl Drop for Shared {
    fn drop(&mut self) {
        self.gpu = None;
        self.npu = None;
        unsafe { libc_mm::munmap(self.host as *mut core::ffi::c_void, self.len) };
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// CPU oracle for one NPU window

struct Oracle {
    g: Ief15Gemm,
    d: Vec<u16>,
    b: Vec<i16>,
    a_stream: Vec<u8>,
    b_stream: Vec<u8>,
    y: Vec<u8>,
}

fn oracle(inp: &Inputs, n: usize, with_reference: bool) -> Result<Oracle, String> {
    let t0 = Instant::now();
    let g = Ief15Gemm::new(T, n, K, Control::Fast)?;
    let w = Weights::new(n, K, &inp.w[..n * ROWB]);
    let x = Acts::new(T, K, &inp.xq);
    let (s, a) = iu4_ief15::encode_weight_scales(&w);
    let (d, b) = iu4_ief15::encode_act_scales(&x);
    let [a_stream, b_stream] = g.pack_in(&w, &x, &s, &a, &d, &b)?;
    let t1 = Instant::now();
    let y = if with_reference {
        g.reference(&w, &x, &s, &a, &d, &b)
    } else {
        Vec::new()
    };
    println!("oracle n={n}: {} epochs, waves {} (MW,NW)={:?}, args {:?}; sidecars+pack {:.2} s, reference {:.2} s; Y fnv {:016x}",
        g.epochs(), g.layout().waves(), g.layout().wave_grid(), g.design().args.iter().map(|a| a.bytes).collect::<Vec<_>>(),
        (t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64(), fnv(&y));
    Ok(Oracle {
        g,
        d,
        b,
        a_stream,
        b_stream,
        y,
    })
}

fn le16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn lei16(v: &[i16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn count_diff(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
    let mut n = 0;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x != y {
            n += 1;
            if first.is_none() {
                first = Some(i)
            }
        }
    }
    (n + a.len().abs_diff(b.len()), first)
}

/// Mismatching f32 words of `got` vs `want` in features `[c0, c1)` of `[T][F]` buffers.
fn cols_diff(got: &[u8], want: &[u8], c0: usize, c1: usize) -> usize {
    (0..T)
        .map(|t| {
            (c0..c1)
                .filter(|&r| got[(t * F + r) * 4..][..4] != want[(t * F + r) * 4..][..4])
                .count()
        })
        .sum()
}

/// `RN(old + Y)` on features `[0, n)` (Y `[T][n]`), as f32 LE bytes of the whole `[T][F]`, other columns `base`.
fn expect_add(base: &[u8], old: &[u8], y: &[u8], n: usize) -> Vec<u8> {
    let mut out = base.to_vec();
    for t in 0..T {
        for r in 0..n {
            let v = f32_at(old, t * F + r) + f32_at(y, t * n + r);
            out[(t * F + r) * 4..][..4].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Error census of `got` vs `refv` over features `[0, n)` of `[T][F]` f32 buffers: (NRMSE, max abs, mean rel, max rel).
fn err_stats(
    got: &[u8],
    refv: &[u8],
    stride_got: usize,
    stride_ref: usize,
    n: usize,
) -> (f64, f64, f64, f64) {
    let (mut se, mut sr, mut maxa, mut srel, mut maxrel, mut cnt) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0usize);
    for t in 0..T {
        for r in 0..n {
            let g = f64::from(f32_at(got, t * stride_got + r));
            let w = f64::from(f32_at(refv, t * stride_ref + r));
            let d = (g - w).abs();
            se += d * d;
            sr += w * w;
            maxa = maxa.max(d);
            if w != 0.0 {
                let rel = d / w.abs();
                srel += rel;
                maxrel = maxrel.max(rel);
                cnt += 1;
            }
        }
    }
    ((se / sr).sqrt(), maxa, srel / cnt.max(1) as f64, maxrel)
}

// ---------------------------------------------------------------------------------------------------------------------
// Modes

fn mode_gpu(cfg: &Cfg, inp: &Inputs) -> Result<bool, String> {
    let gpu = Gpu::new()?;
    let rt = &gpu.rt;
    let (w, xq, old) = (gpu.up(&inp.w)?, gpu.up(&inp.xq)?, gpu.up(&inp.old)?);
    let y_full = rt.allocate(T * F * 4, None)?;
    let y_win = rt.allocate(T * F * 4, None)?;
    let mut ok = true;
    // 1. Windowed incumbent: byte identity on its columns, untouched elsewhere.
    rt.copy_dtod(y_full.ptr(), old.ptr(), T * F * 4)?;
    gpu.down_add(w.ptr(), xq.ptr(), y_full.ptr(), 0, F)?;
    rt.synchronize()?;
    let full = gpu.down(&y_full, T * F * 4)?;
    println!("gpu full ADD: Y fnv {:016x}", fnv(&full));
    for &cw in &cfg.cols {
        if cw == F {
            continue;
        }
        let col0 = F - cw;
        rt.copy_dtod(y_win.ptr(), old.ptr(), T * F * 4)?;
        gpu.down_add(w.ptr(), xq.ptr(), y_win.ptr(), col0, cw)?;
        rt.synchronize()?;
        let win = gpu.down(&y_win, T * F * 4)?;
        let (own, rest) = (
            cols_diff(&win, &full, col0, F),
            cols_diff(&win, &inp.old, 0, col0),
        );
        println!("window identity cols [{col0},{F}): mismatches vs full ADD {own}/{}, untouched [0,{col0}) mismatches {rest}", T * cw);
        ok &= own == 0 && rest == 0;
    }
    // 2. Time vs width, ABBA blocks of `reps` launches each, against the full incumbent.
    let (e0, e1) = (rt.event()?, rt.event()?);
    let time = |col0: usize, cw: usize| -> Result<f64, String> {
        rt.record(&e0)?;
        for _ in 0..cfg.reps {
            gpu.down_add(w.ptr(), xq.ptr(), y_win.ptr(), col0, cw)?;
        }
        rt.record(&e1)?;
        Ok(f64::from(rt.elapsed_ms(&e0, &e1)?) / cfg.reps as f64)
    };
    for _ in 0..2 {
        time(0, F)?;
    }
    for &cw in &cfg.cols {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for blk in 0..cfg.blocks {
            if blk % 2 == 0 {
                a.push(time(0, F)?);
                b.push(time(F - cw, cw)?);
            } else {
                b.push(time(F - cw, cw)?);
                a.push(time(0, F)?);
            }
        }
        println!("GPUTIME width={cw} npu_cols={} full_ms={:.4} window_ms={:.4} ratio={:.4} full=[{}] window=[{}]",
            F - cw, median(&a), median(&b), median(&b) / median(&a), fmt_ms(&a), fmt_ms(&b));
    }
    // 3. Co-op GPU kernels vs the CPU oracle (no NPU: plain registered host pages), and their time.
    let dc = rt.allocate(E * T * 2, None)?;
    let bc = rt.allocate(T * 2, None)?;
    let dcomp = rt.allocate(E * T * 4, None)?;
    for &n in &cfg.npu_cols {
        let o = oracle(inp, n, false)?;
        let g = &o.g;
        let a_sh = Shared::new(&gpu, None, g.design().args[0].bytes)?;
        let c_sh = Shared::new(&gpu, None, g.design().args[2].bytes)?;
        for (v2, wb) in [(false, false), (false, true), (true, false), (true, true)] {
            a_sh.fill(POISON);
            rt.memset(&dc, POISON)?;
            gpu.prep(
                v2,
                xq.ptr(),
                a_sh.gpu(),
                dc.ptr(),
                bc.ptr(),
                dcomp.ptr(),
                g,
                wb,
            )?;
            rt.synchronize()?;
            let (dd, bd) = (
                count_diff(&gpu.down(&dc, E * T * 2)?, &le16(&o.d)).0,
                count_diff(&gpu.down(&bc, T * 2)?, &lei16(&o.b)).0,
            );
            let (ad, afirst) = count_diff(&a_sh.read(0, o.a_stream.len()), &o.a_stream);
            println!("kernels n={n} glue {}: D mismatches {dd}/{} B, b mismatches {bd}/{} B; A stream ({}) mismatches {ad}/{} B (first {afirst:?})",
                if v2 { "v2" } else { "v1" }, E * T * 2, T * 2, if wb { "wb" } else { "stream" }, o.a_stream.len());
            ok &= dd == 0 && bd == 0 && ad == 0;
        }
        // ief_addc: pseudo-random finite tiled C, expected RN(old + unpack(C)).
        let mut cbytes = vec![0u8; g.design().args[2].bytes];
        let mut s = 0x2545_f491_4f6c_dd1du64 ^ n as u64;
        for c in cbytes.chunks_exact_mut(4) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let v = ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
                * 2f32.powi((s % 9) as i32 - 4);
            c.copy_from_slice(&v.to_le_bytes());
        }
        c_sh.write(0, &cbytes);
        rt.copy_dtod(y_win.ptr(), old.ptr(), T * F * 4)?;
        gpu.addc(c_sh.gpu(), y_win.ptr(), g)?;
        rt.synchronize()?;
        let want = expect_add(&inp.old, &inp.old, &g.unpack_out(&cbytes), n);
        let (xd, _) = count_diff(&gpu.down(&y_win, T * F * 4)?, &want);
        println!("kernels n={n}: ief_addc x mismatches {xd}/{} B", T * F * 4);
        ok &= xd == 0;
        let (e2, e3) = (rt.event()?, rt.event()?);
        let mut tdb = Vec::new();
        let mut tpk = Vec::new();
        let mut tpw = Vec::new();
        let mut tac = Vec::new();
        let mut tv2 = Vec::new();
        for _ in 0..cfg.reps + 1 {
            rt.record(&e0)?;
            gpu.db(xq.ptr(), dc.ptr(), bc.ptr(), E)?;
            rt.record(&e1)?;
            gpu.pack(xq.ptr(), a_sh.gpu(), dc.ptr(), bc.ptr(), g, false)?;
            rt.record(&e2)?;
            gpu.addc(c_sh.gpu(), y_win.ptr(), g)?;
            rt.record(&e3)?;
            tdb.push(f64::from(rt.elapsed_ms(&e0, &e1)?));
            tpk.push(f64::from(rt.elapsed_ms(&e1, &e2)?));
            tac.push(f64::from(rt.elapsed_ms(&e2, &e3)?));
            rt.record(&e0)?;
            gpu.pack(xq.ptr(), a_sh.gpu(), dc.ptr(), bc.ptr(), g, true)?;
            rt.record(&e1)?;
            tpw.push(f64::from(rt.elapsed_ms(&e0, &e1)?));
            rt.record(&e0)?;
            gpu.prep(
                true,
                xq.ptr(),
                a_sh.gpu(),
                dc.ptr(),
                bc.ptr(),
                dcomp.ptr(),
                g,
                true,
            )?;
            rt.record(&e1)?;
            tv2.push(f64::from(rt.elapsed_ms(&e0, &e1)?));
        }
        println!("KTIME n={n} db_ms={:.4} pack_stream_ms={:.4} pack_wb_ms={:.4} v2_prep_wb_ms={:.4} addc_ms={:.4} (A stream {} B, C {} B) db=[{}] pack=[{}] pack_wb=[{}] v2=[{}] addc=[{}]",
            median(&tdb[1..]), median(&tpk[1..]), median(&tpw[1..]), median(&tv2[1..]), median(&tac[1..]), o.a_stream.len(), cbytes.len(), fmt_ms(&tdb),
            fmt_ms(&tpk), fmt_ms(&tpw), fmt_ms(&tv2), fmt_ms(&tac));
    }
    Ok(ok)
}

/// NPU side of one window `n`: hardware context, PDI, insts, B (packed once) and the shared A / C arenas.
struct Npu<'d> {
    ctx: HwCtx<'d>,
    _pdi: Bo,
    insts: Bo,
    b: Bo,
    cmd: Bo,
    a: Shared,
    c: Shared,
}

fn npu_side<'d>(dev: &'d Device, gpu: &Gpu, o: &Oracle) -> Result<Npu<'d>, String> {
    let d = o.g.design();
    let ntiles = dev.meta.cols as u32 * dev.meta.core.0 as u32;
    let ctx = HwCtx::create(dev, ntiles, 2048)?;
    let pdi = dev.dev_bo(&d.pdi)?;
    ctx.config_cu(&pdi)?;
    let insts = dev.dev_bo(&d.insts)?;
    let mut b = dev.shmem_bo(d.args[1].bytes)?;
    b.as_mut_slice().copy_from_slice(&o.b_stream);
    b.flush();
    let a = Shared::new(gpu, Some(dev), d.args[0].bytes)?;
    let c = Shared::new(gpu, Some(dev), d.args[2].bytes)?;
    let cmd = dev.cmd_bo()?;
    Ok(Npu {
        ctx,
        _pdi: pdi,
        insts,
        b,
        cmd,
        a,
        c,
    })
}

impl Npu<'_> {
    /// One submit + wait: (completed, seconds submit -> completion).
    fn run(&mut self, timeout_ms: u64) -> Result<(bool, f64), String> {
        let t0 = Instant::now();
        let seq = self.ctx.submit(
            &mut self.cmd,
            &self.insts,
            &[self.a.bo(), &self.b, self.c.bo()],
        )?;
        let st = self.ctx.wait(&self.cmd, seq, timeout_ms)?;
        Ok((st == ERT_STATE_COMPLETED, t0.elapsed().as_secs_f64()))
    }
}

fn open_npu(gpu: &Gpu) -> Result<Device, String> {
    let mut dev = Device::open()?;
    dev.map_heap(64 << 20)?;
    let v = gpu.alias_check(&dev)?;
    let s = |b: bool| if b { "ok" } else { "MISMATCH" };
    println!("alias: userptr (anon pages, NPU userptr BO + hipHostRegister flags {:#x}): cpu->gpu {} gpu->cpu {} npu-map==cpu {} npu-map->cpu {} \
        gpu-reread-after-foreign-write {}", gpu.reg.get(), s(v[0]), s(v[1]), s(v[2]), s(v[3]), s(v[4]));
    if !v.iter().all(|&b| b) {
        return Err("shared host pages do not alias across CPU / GPU / NPU mapping; stop before any NPU command".into());
    }
    Ok(dev)
}

fn mode_npu(cfg: &Cfg, inp: &Inputs) -> Result<bool, String> {
    let gpu = Gpu::new()?;
    let dev = open_npu(&gpu)?;
    let mut ok = true;
    for &n in &cfg.npu_cols {
        let o = oracle(inp, n, true)?;
        let mut npu = npu_side(&dev, &gpu, &o)?;
        npu.a.write(0, &o.a_stream);
        let mut busy = Vec::new();
        for it in 0..cfg.iters {
            let verify = it == 0 || it + 1 == cfg.iters;
            if verify {
                npu.c.fill(POISON);
            }
            let (done, dt) = npu.run(cfg.timeout_ms)?;
            busy.push(dt * 1e3);
            if !done {
                println!("npu n={n} iter {it}: NOT COMPLETED");
                return Ok(false);
            }
            if verify {
                let (bad, first) = count_diff(
                    &o.g.unpack_out(&npu.c.read(0, o.g.design().args[2].bytes)),
                    &o.y,
                );
                println!("npu n={n} iter {it}: {:.3} ms, C vs CPU IEF15 oracle mismatches {bad}/{} B (first {first:?})", dt * 1e3, o.y.len());
                ok &= bad == 0;
            }
        }
        let ms = if busy.len() > 1 {
            median(&busy[1..])
        } else {
            busy[0]
        };
        let ops = 2.0 * T as f64 * n as f64 * K as f64;
        println!(
            "NPUTIME n={n} median_ms={ms:.4} tops={:.3} first_ms={:.3} all=[{}]",
            ops / ms / 1e9,
            busy[0],
            fmt_ms(&busy)
        );
    }
    Ok(ok)
}

/// Labels of the `--mode fold` timing variants (both arms).
const FOLD_VARIANTS: [&str; 7] = [
    "window",
    "window+sep",
    "window+wide",
    "fold",
    "fold_j0",
    "tail_sep",
    "tail_wide",
];

/// ABBA timing of the 7 [`FOLD_VARIANTS`] (`iters` rounds after one warm-up, forward order on even rounds and reversed on
/// odd ones), GPU temperature/sclk sampled after every launch; prints `FOLDTIME` / `FOLDTHERM` / `FOLDRAW`.
fn fold_timing(
    rt: &HipRuntime,
    iters: usize,
    label: &str,
    prep: &dyn Fn() -> Result<(), String>,
    run: &dyn Fn(usize) -> Result<(), String>,
) -> Result<(), String> {
    let (e0, e1) = (rt.event()?, rt.event()?);
    let mut t: Vec<Vec<f64>> = vec![Vec::new(); 7];
    let mut th: Vec<Vec<[f64; 2]>> = vec![Vec::new(); 7];
    for it in 0..iters + 1 {
        for s in 0..7 {
            let v = if it % 2 == 0 { s } else { 6 - s };
            prep()?;
            rt.record(&e0)?;
            run(v)?;
            rt.record(&e1)?;
            let ms = f64::from(rt.elapsed_ms(&e0, &e1)?);
            if it > 0 {
                t[v].push(ms);
                th[v].push(fold::thermal());
            }
        }
    }
    let m: Vec<f64> = t.iter().map(|v| median(v)).collect();
    println!(
        "FOLDTIME {label} {}",
        FOLD_VARIANTS
            .iter()
            .zip(&m)
            .map(|(n, x)| format!("{n}={x:.4}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "FOLDTHERM {label} {}",
        FOLD_VARIANTS
            .iter()
            .zip(&th)
            .map(|(n, x)| format!("{n}:{}", fold::therm_summary(x)))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "FOLDRAW {label} {}",
        FOLD_VARIANTS
            .iter()
            .zip(&t)
            .map(|(n, x)| format!("{n}=[{}]", fmt_ms(x)))
            .collect::<Vec<_>>()
            .join(" ")
    );
    Ok(())
}

/// `--mode fold` (down, GPU only): `ief_add_fold` / `ief_addc_wide` on the window `n = --npu-cols[0]`. Checks: the
/// twin with no NPU work equals the full incumbent on its columns and leaves `[0, n)` untouched; the flag published
/// mid-kernel (3 rounds, a different C each, so a stale GPU line of an earlier round would show) and preset; the
/// wide tail. Then the [`FOLD_VARIANTS`] timing.
fn mode_fold(cfg: &Cfg, inp: &Inputs) -> Result<bool, String> {
    let n = cfg.npu_cols[0];
    let gpu = Gpu::new()?;
    gpu.reg.set(cfg.reg);
    let rt = &gpu.rt;
    let sync = fold::FoldSync::new(&gpu)?;
    let (w, xq, old) = (gpu.up(&inp.w)?, gpu.up(&inp.xq)?, gpu.up(&inp.old)?);
    let x_full = rt.allocate(T * F * 4, None)?;
    let x = rt.allocate(T * F * 4, None)?;
    rt.copy_dtod(x_full.ptr(), old.ptr(), T * F * 4)?;
    gpu.down_add(w.ptr(), xq.ptr(), x_full.ptr(), 0, F)?;
    rt.synchronize()?;
    let full = gpu.down(&x_full, T * F * 4)?;
    let g = Ief15Gemm::new(T, n, K, Control::Fast)?;
    let geo = fold::Geo::new(&g, false)?;
    let c_sh = Shared::new(&gpu, None, g.design().args[2].bytes)?;
    let grid = Gpu::down_add_grid(F - n);
    let ext = |round: u32, ctr: u64, j: u32, wide: bool| fold::Ext {
        c: [c_sh.gpu(), 0, 0, 0],
        flags: sync.flags(),
        ctr,
        hbase: x.ptr(),
        round,
        j,
        ldy: F as u32,
        geo,
        drain: cfg.drain as u32,
        grid,
        wide,
        chw: 0,
        claim: cfg.claim as u32,
    };
    let rand_c = |seed: u64| -> Vec<u8> {
        let mut cb = vec![0u8; g.design().args[2].bytes];
        let mut s = 0x2545_f491_4f6c_dd1du64 ^ (seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        for c in cb.chunks_exact_mut(4) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let v = ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
                * 2f32.powi((s % 9) as i32 - 4);
            c.copy_from_slice(&v.to_le_bytes());
        }
        cb
    };
    let (e0, e1) = (rt.event()?, rt.event()?);
    let mut ok = true;
    println!("fold down n={n}: upc {} lgx {} lgy {} waves {} NW {}; window grid {grid:?}, drain {} WGs, claim {}", geo.upc, geo.lgx, geo.lgy, geo.waves,
        geo.nw, cfg.drain, cfg.claim);
    // 1. The twin with no NPU work.
    rt.copy_dtod(x.ptr(), old.ptr(), T * F * 4)?;
    let (r, slot) = sync.next()?;
    gpu.add_fold(
        w.ptr(),
        xq.ptr(),
        x.ptr(),
        n,
        F - n,
        &ext(r, slot, 0, false),
    )?;
    rt.synchronize()?;
    let got = gpu.down(&x, T * F * 4)?;
    let (own, rest) = (
        cols_diff(&got, &full, n, F),
        cols_diff(&got, &inp.old, 0, n),
    );
    let (sok, s) = sync.check(&gpu, r, 0, geo.upc)?;
    println!("fold twin J=0: cols [{n},{F}) vs full incumbent mismatches {own}/{}, [0,{n}) untouched mismatches {rest}; slot {s}", T * (F - n));
    ok &= own == 0 && rest == 0 && sok;
    // 2. C written and the flag published while the kernel runs.
    for round_i in 0..3u64 {
        let cb = rand_c(round_i);
        c_sh.fill(POISON);
        rt.copy_dtod(x.ptr(), old.ptr(), T * F * 4)?;
        let (r, slot) = sync.next()?;
        rt.record(&e0)?;
        gpu.add_fold(
            w.ptr(),
            xq.ptr(),
            x.ptr(),
            n,
            F - n,
            &ext(r, slot, 1, false),
        )?;
        rt.record(&e1)?;
        let t0 = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(6));
        c_sh.write(0, &cb);
        sync.signal(0, r);
        let pub_ms = t0.elapsed().as_secs_f64() * 1e3;
        let k_ms = rt.elapsed_ms(&e0, &e1)?;
        let (bad, first) = count_diff(
            &gpu.down(&x, T * F * 4)?,
            &expect_add(&full, &inp.old, &g.unpack_out(&cb), n),
        );
        let (sok, s) = sync.check(&gpu, r, 1, geo.upc)?;
        println!("fold late flag {round_i}: kernel {k_ms:.3} ms, C + flag published {pub_ms:.3} ms after launch; x vs [RN(old + C) | full incumbent] \
            mismatches {bad} B (first {first:?}); slot {s}");
        ok &= bad == 0 && sok;
    }
    // 3. Flag preset, and 4. the wide tail, on one C.
    let cb = rand_c(7);
    c_sh.write(0, &cb);
    let y = g.unpack_out(&cb);
    rt.copy_dtod(x.ptr(), old.ptr(), T * F * 4)?;
    let (r, slot) = sync.next()?;
    sync.signal(0, r);
    gpu.add_fold(
        w.ptr(),
        xq.ptr(),
        x.ptr(),
        n,
        F - n,
        &ext(r, slot, 1, false),
    )?;
    rt.synchronize()?;
    let (bad, _) = count_diff(
        &gpu.down(&x, T * F * 4)?,
        &expect_add(&full, &inp.old, &y, n),
    );
    let (sok, s) = sync.check(&gpu, r, 1, geo.upc)?;
    rt.copy_dtod(x.ptr(), old.ptr(), T * F * 4)?;
    gpu.addc_wide(&ext(0, 0, 1, true))?;
    rt.synchronize()?;
    let (wbad, _) = count_diff(
        &gpu.down(&x, T * F * 4)?,
        &expect_add(&inp.old, &inp.old, &y, n),
    );
    println!("fold preset flag: mismatches {bad} B, slot {s}; ief_addc_wide: x vs [RN(old + C) | old] mismatches {wbad} B");
    ok &= bad == 0 && sok && wbad == 0;
    if !ok {
        return Ok(false);
    }
    let prep = || rt.copy_dtod(x.ptr(), old.ptr(), T * F * 4);
    let run = |v: usize| -> Result<(), String> {
        let win = || gpu.down_add(w.ptr(), xq.ptr(), x.ptr(), n, F - n);
        match v {
            0 => win(),
            1 => {
                win()?;
                gpu.addc(c_sh.gpu(), x.ptr(), &g)
            }
            2 => {
                win()?;
                gpu.addc_wide(&ext(0, 0, 1, true))
            }
            3 | 4 => {
                let (r, slot) = sync.next()?;
                sync.signal(0, r);
                gpu.add_fold(
                    w.ptr(),
                    xq.ptr(),
                    x.ptr(),
                    n,
                    F - n,
                    &ext(r, slot, if v == 3 { 1 } else { 0 }, false),
                )
            }
            5 => gpu.addc(c_sh.gpu(), x.ptr(), &g),
            _ => gpu.addc_wide(&ext(0, 0, 1, true)),
        }
    };
    fold_timing(
        rt,
        cfg.iters,
        &format!("proj=down n={n} reg={:#x}", cfg.reg),
        &prep,
        &run,
    )?;
    let (sok, s) = sync.check(&gpu, sync.last(), 0, geo.upc)?;
    println!("fold last timed round slot {s}");
    Ok(ok && sok)
}

fn mode_coop(cfg: &Cfg, inp: &Inputs) -> Result<bool, String> {
    let n = cfg.npu_cols[0];
    let gpu = Gpu::new()?;
    gpu.reg.set(cfg.reg);
    let rt = &gpu.rt;
    let dev = open_npu(&gpu)?;
    let o = oracle(inp, n, true)?;
    let g = &o.g;
    let mut npu = npu_side(&dev, &gpu, &o)?;
    let (w, xq, old) = (gpu.up(&inp.w)?, gpu.up(&inp.xq)?, gpu.up(&inp.old)?);
    let x_alone = rt.allocate(T * F * 4, None)?;
    let x_coop = rt.allocate(T * F * 4, None)?;
    let x_win = rt.allocate(T * F * 4, None)?;
    let dc = rt.allocate(E * T * 2, None)?;
    let dcomp = rt.allocate(E * T * 4, None)?;
    let bc = rt.allocate(T * 2, None)?;
    let ev: Vec<HipEvent> = (0..4).map(|_| rt.event()).collect::<Result<_, _>>()?;
    let sync = fold::FoldSync::new(&gpu)?;
    let geo = fold::Geo::new(g, false)?;
    let grid = Gpu::down_add_grid(F - n);
    let c_npu = npu.c.gpu();
    let ext = |round: u32, ctr: u64, wide: bool| fold::Ext {
        c: [c_npu, 0, 0, 0],
        flags: sync.flags(),
        ctr,
        hbase: x_coop.ptr(),
        round,
        j: 1,
        ldy: F as u32,
        geo,
        drain: cfg.drain as u32,
        grid,
        wide,
        chw: 0,
        claim: cfg.claim as u32,
    };
    let ext_wide = ext(0, 0, true);
    let last_round = std::cell::Cell::new(0u32);
    println!(
        "coop tails: {:?} (drain {} WGs, claim {}, window grid {grid:?})",
        cfg.tails, cfg.drain, cfg.claim
    );
    // Incumbent pre-epilogue Y (ADD onto zeros) for the error census.
    rt.memset(&x_win, 0)?;
    gpu.down_add(w.ptr(), xq.ptr(), x_win.ptr(), 0, F)?;
    rt.synchronize()?;
    let y_gpu = gpu.down(&x_win, T * F * 4)?;

    let alone = |x: &HipBuffer| -> Result<f64, String> {
        rt.copy_dtod(x.ptr(), old.ptr(), T * F * 4)?;
        rt.record(&ev[0])?;
        gpu.down_add(w.ptr(), xq.ptr(), x.ptr(), 0, F)?;
        rt.record(&ev[1])?;
        Ok(f64::from(rt.elapsed_ms(&ev[0], &ev[1])?))
    };
    let window = || -> Result<f64, String> {
        rt.copy_dtod(x_win.ptr(), old.ptr(), T * F * 4)?;
        rt.record(&ev[0])?;
        gpu.down_add(w.ptr(), xq.ptr(), x_win.ptr(), n, F - n)?;
        rt.record(&ev[1])?;
        Ok(f64::from(rt.elapsed_ms(&ev[0], &ev[1])?))
    };
    // One co-op layer. Returns (total, pack phase, GPU window GEMM (fold: + the folded addc), E2->E3, npu busy, relay us) ms.
    let coop = |npu: &mut Npu, tail: fold::Tail| -> Result<[f64; 6], String> {
        rt.copy_dtod(x_coop.ptr(), old.ptr(), T * F * 4)?;
        rt.synchronize()?;
        rt.record(&ev[0])?;
        gpu.prep(
            cfg.glue2,
            xq.ptr(),
            npu.a.gpu(),
            dc.ptr(),
            bc.ptr(),
            dcomp.ptr(),
            g,
            cfg.wb,
        )?;
        rt.record(&ev[1])?;
        let round = if tail == fold::Tail::Fold {
            let (r, slot) = sync.next()?;
            gpu.add_fold(
                w.ptr(),
                xq.ptr(),
                x_coop.ptr(),
                n,
                F - n,
                &ext(r, slot, false),
            )?;
            last_round.set(r);
            Some(r)
        } else {
            gpu.down_add(w.ptr(), xq.ptr(), x_coop.ptr(), n, F - n)?;
            None
        };
        rt.record(&ev[2])?;
        rt.sync_event(&ev[1])?;
        let t_obs = Instant::now();
        let run = npu.run(cfg.timeout_ms);
        let busy = match run {
            Ok((true, b)) => b,
            other => {
                if let Some(r) = round {
                    sync.release(r);
                    rt.synchronize()?;
                }
                return Err(format!(
                    "NPU command did not complete: {:?}",
                    other.map(|v| v.0)
                ));
            }
        };
        if let Some(r) = round {
            sync.signal(0, r);
        }
        let relay_us = (t_obs.elapsed().as_secs_f64() - busy) * 1e6;
        match tail {
            fold::Tail::Sep => gpu.addc(npu.c.gpu(), x_coop.ptr(), g)?,
            fold::Tail::Wide => gpu.addc_wide(&ext_wide)?,
            fold::Tail::Fold => {}
        }
        rt.record(&ev[3])?;
        let total = f64::from(rt.elapsed_ms(&ev[0], &ev[3])?);
        Ok([
            total,
            f64::from(rt.elapsed_ms(&ev[0], &ev[1])?),
            f64::from(rt.elapsed_ms(&ev[1], &ev[2])?),
            f64::from(rt.elapsed_ms(&ev[2], &ev[3])?),
            busy * 1e3,
            relay_us,
        ])
    };
    let mut ok = true;
    let verify = |npu: &Npu, label: &str, tail: fold::Tail| -> Result<bool, String> {
        let xa = gpu.down(&x_alone, T * F * 4)?;
        let xc = gpu.down(&x_coop, T * F * 4)?;
        let gpu_cols = cols_diff(&xc, &xa, n, F);
        let (dd, bd) = (
            count_diff(&gpu.down(&dc, E * T * 2)?, &le16(&o.d)).0,
            count_diff(&gpu.down(&bc, T * 2)?, &lei16(&o.b)).0,
        );
        let (ad, _) = count_diff(&npu.a.read(0, o.a_stream.len()), &o.a_stream);
        let (cd, cfirst) = count_diff(
            &g.unpack_out(&npu.c.read(0, g.design().args[2].bytes)),
            &o.y,
        );
        let want = expect_add(&xa, &inp.old, &o.y, n);
        let npu_cols = cols_diff(&xc, &want, 0, n);
        println!("verify {label}: GPU cols [{n},{F}) vs GPU-alone mismatches {gpu_cols}/{}; D/b vs CPU {dd}/{bd}; A stream vs CPU {ad}; \
            NPU C vs CPU IEF15 oracle {cd}/{} B (first {cfirst:?}); NPU cols x vs RN(old + oracle Y) {npu_cols}/{}",
            T * (F - n), o.y.len(), T * n);
        let mut all = gpu_cols == 0 && dd == 0 && bd == 0 && ad == 0 && cd == 0 && npu_cols == 0;
        if tail == fold::Tail::Fold {
            let (sok, s) = sync.check(&gpu, last_round.get(), 1, geo.upc)?;
            println!(
                "verify {label}: fold slot {s}: {}",
                if sok { "ok" } else { "BAD" }
            );
            all &= sok;
        }
        if label.starts_with("first") && tail == cfg.tails[0] {
            let (nr, ma, mr, xr) = err_stats(&o.y, &y_gpu, n, F, n);
            println!("ERR pre-epilogue Y, NPU cols (IEF15) vs incumbent GPU Y: nrmse={nr:.4e} maxabs={ma:.4e} meanrel={mr:.4e} maxrel={xr:.4e}");
            let (nr, ma, mr, xr) = err_stats(&xc, &xa, F, F, n);
            println!("ERR layer output x, NPU cols co-op vs GPU-alone: nrmse={nr:.4e} maxabs={ma:.4e} meanrel={mr:.4e} maxrel={xr:.4e}");
            let full_nrmse = {
                let (mut se, mut sr) = (0f64, 0f64);
                for i in 0..T * F {
                    let (c, a) = (f64::from(f32_at(&xc, i)), f64::from(f32_at(&xa, i)));
                    se += (c - a) * (c - a);
                    sr += a * a;
                }
                (se / sr).sqrt()
            };
            println!(
                "ERR layer output x, all 5120 cols co-op vs GPU-alone: nrmse={full_nrmse:.4e}"
            );
        }
        Ok(all)
    };
    // Warm-up, then the verified first co-op layer of every tail on poisoned arenas.
    alone(&x_alone)?;
    window()?;
    for &t in &cfg.tails {
        npu.a.fill(POISON);
        npu.c.fill(POISON);
        let first = coop(&mut npu, t)?;
        println!("coop first ({}): total {:.3} ms", t.name(), first[0]);
        alone(&x_alone)?;
        ok &= verify(&npu, &format!("first {}", t.name()), t)?;
    }
    if !ok {
        return Ok(false);
    }
    // Timed rounds: [alone, window, coop(tail)..] forward on even rounds, reversed on odd ones (ABBA).
    let nv = 2 + cfg.tails.len();
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); nv];
    let mut therm: Vec<Vec<[f64; 2]>> = vec![Vec::new(); nv];
    let mut parts: Vec<Vec<[f64; 6]>> = vec![Vec::new(); cfg.tails.len()];
    for it in 0..cfg.iters {
        for s in 0..nv {
            let v = if it % 2 == 0 { s } else { nv - 1 - s };
            let ms = match v {
                0 => alone(&x_alone)?,
                1 => window()?,
                _ => {
                    let p = coop(&mut npu, cfg.tails[v - 2])?;
                    parts[v - 2].push(p);
                    p[0]
                }
            };
            times[v].push(ms);
            therm[v].push(fold::thermal());
        }
    }
    // Last verified layer of every tail, again on a poisoned C.
    for &t in &cfg.tails {
        npu.c.fill(POISON);
        coop(&mut npu, t)?;
        alone(&x_alone)?;
        ok &= verify(&npu, &format!("last {}", t.name()), t)?;
    }
    let (ma, mw) = (median(&times[0]), median(&times[1]));
    println!(
        "COOPTHERM n={n} alone:{} window:{} {}",
        fold::therm_summary(&therm[0]),
        fold::therm_summary(&therm[1]),
        cfg.tails
            .iter()
            .enumerate()
            .map(|(i, t)| format!("coop_{}:{}", t.name(), fold::therm_summary(&therm[i + 2])))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "COOPRAW n={n} alone=[{}] window=[{}]",
        fmt_ms(&times[0]),
        fmt_ms(&times[1])
    );
    for (i, t) in cfg.tails.iter().enumerate() {
        let col = |k: usize| -> Vec<f64> { parts[i].iter().map(|p| p[k]).collect() };
        let mc = median(&times[i + 2]);
        println!("COOP n={n} tail={} gpu_alone_ms={ma:.4} window_alone_ms={mw:.4} coop_ms={mc:.4} delta_ms={:.4} speedup={:.4} \
            pack_ms={:.4} window_concurrent_ms={:.4} alpha={:.4} tail_ms={:.4} npu_busy_ms={:.4} relay_us={:.1}", t.name(),
            mc - ma, ma / mc, median(&col(1)), median(&col(2)), median(&col(2)) / mw, median(&col(3)), median(&col(4)), median(&col(5)));
        println!(
            "COOPRAW n={n} tail={} coop=[{}] pack=[{}] wconc=[{}] tail=[{}] npu=[{}]",
            t.name(),
            fmt_ms(&times[i + 2]),
            fmt_ms(&col(1)),
            fmt_ms(&col(2)),
            fmt_ms(&col(3)),
            fmt_ms(&col(4))
        );
    }
    Ok(ok)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = match parse(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("coop27: {e}\n{USAGE}");
            std::process::exit(2)
        }
    };
    let res = railgun::npu::fclk::FabricClockGuard::acquire("coop27").and_then(|_fclk| {
        if cfg.mode == "parts" {
            return parts::run(&cfg);
        }
        if cfg.mode == "maps" {
            return maps::run(&cfg);
        }
        if cfg.mode == "cpu" {
            return cpu::run(&cfg);
        }
        if cfg.proj == "gate_up" {
            return gate_up::run(&cfg);
        }
        if cfg.proj != "down" {
            return Err(format!("--proj must be down|gate_up, got {}", cfg.proj));
        }
        let inp = load_inputs(&cfg)?;
        match cfg.mode.as_str() {
            "gpu" => mode_gpu(&cfg, &inp),
            "npu" => mode_npu(&cfg, &inp),
            "fold" => mode_fold(&cfg, &inp),
            _ => mode_coop(&cfg, &inp),
        }
    });
    match res {
        Ok(true) => println!("RESULT: PASS"),
        Ok(false) => {
            println!("RESULT: FAIL");
            std::process::exit(1)
        }
        Err(e) => {
            println!("error: {e}");
            println!("RESULT: FAIL");
            std::process::exit(1)
        }
    }
}
