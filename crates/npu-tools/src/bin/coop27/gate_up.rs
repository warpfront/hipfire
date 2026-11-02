// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! gate_up arm: dense-27B FFN gate/up (two 17408 x 5120 MQ4G256V2 tensors, K = 5120) with the incumbent F1-lite SiLU
//! epilogue `h = g / (1 + expf(-g)) * u` into `h [8192][17408]`.
//!
//! Column map (fixed): the NPU owns hidden features `[0, n_h)`, `n_h = 1024 * J`, as J IEF15 commands of one design
//! `8192 x 2048 x 5120` whose features are `[G rows j*1024 .. +1024 | U rows j*1024 .. +1024]` (separate pre-SiLU f32 g
//! and u; one shared A stream since A depends only on Xq and NW = 8). The GPU owns `[n_h, 17408)` on the incumbent
//! `gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151` (window by pointer offset of G, U and Y, `M = 17408` as the Y row
//! stride, `2 (17408 - n_h) / 256` row tiles), and applies the incumbent SiLU op DAG to the NPU's g, u (`ief_silu`, the
//! verbatim instruction sequence of that entry's epilogue). The split is a multiple of 256 hidden features, so the
//! downstream A4 producer always sees whole 256-wide groups.
use super::*;

pub const H: usize = 17408;
pub const KG: usize = 5120;
pub const EG: usize = KG / 128;
pub const ROWG: usize = KG / 256 * 136;
/// Hidden features per NPU command (2048 IEF15 features: g and u halves).
pub const CH: usize = 1024;
const SILU_CO: &[u8] = include_bytes!("../../../native/coop27/ief_silu_gfx1151.co");
const SILU2_CO: &[u8] = include_bytes!("../../../native/coop27/ief_silu2_gfx1151.co");
const V2B_SET: &str = "gemm_mq4g256v2_residual_iu4_pm_v2b_set_gfx1151";
const V2B_SILU: &str = "gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151";

struct GuInputs {
    wg: Vec<u8>,
    wu: Vec<u8>,
    xq: Vec<u8>,
}

fn load(cfg: &Cfg) -> Result<GuInputs, String> {
    let t0 = Instant::now();
    let mut ws = Vec::new();
    for p in ["gate_proj", "up_proj"] {
        let name = format!("model.language_model.layers.{}.mlp.{p}.weight", cfg.layer);
        let (t, w) = hfqm::read_tensor(&cfg.model, &name)?;
        if t.qt != 44 || t.dims != [H, KG] || w.len() != H * ROWG {
            return Err(format!(
                "{name}: qt {} dims {:?} bytes {}",
                t.qt,
                t.dims,
                w.len()
            ));
        }
        println!(
            "inputs: {name} qt {} {:?} fnv {:016x}",
            t.qt,
            t.dims,
            fnv(&w)
        );
        ws.push(w);
    }
    let xq = tile_xq(&cfg.xq, EG)?;
    let wu = ws.pop().unwrap();
    let wg = ws.pop().unwrap();
    println!("inputs: load {:.2} s", t0.elapsed().as_secs_f64());
    Ok(GuInputs { wg, wu, xq })
}

/// Weights of command `j`: `[G rows j*CH.. | U rows j*CH..]`.
fn cmd_weights(inp: &GuInputs, j: usize) -> Vec<u8> {
    let r = j * CH * ROWG..(j + 1) * CH * ROWG;
    [&inp.wg[r.clone()], &inp.wu[r]].concat()
}

struct GuOracle {
    g: Ief15Gemm,
    d: Vec<u16>,
    b: Vec<i16>,
    a_stream: Vec<u8>,
    b_streams: Vec<Vec<u8>>,
    y: Vec<Vec<u8>>,
}

fn oracle(inp: &GuInputs, j_count: usize, with_reference: bool) -> Result<GuOracle, String> {
    let t0 = Instant::now();
    let g = Ief15Gemm::new(T, 2 * CH, KG, Control::Fast)?;
    let x = Acts::new(T, KG, &inp.xq);
    let (d, b) = iu4_ief15::encode_act_scales(&x);
    let (mut a_stream, mut b_streams, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..j_count {
        let wb = cmd_weights(inp, j);
        let w = Weights::new(2 * CH, KG, &wb);
        let (s, a) = iu4_ief15::encode_weight_scales(&w);
        let [ap, bp] = g.pack_in(&w, &x, &s, &a, &d, &b)?;
        if j == 0 {
            a_stream = ap;
        } else if ap != a_stream {
            return Err("A stream differs between commands".into());
        }
        b_streams.push(bp);
        if with_reference {
            y.push(g.reference(&w, &x, &s, &a, &d, &b));
        }
    }
    println!(
        "gate_up oracle J={j_count}: {} epochs, waves {} (MW,NW)={:?}, args {:?}; {:.2} s",
        g.epochs(),
        g.layout().waves(),
        g.layout().wave_grid(),
        g.design().args.iter().map(|a| a.bytes).collect::<Vec<_>>(),
        t0.elapsed().as_secs_f64()
    );
    Ok(GuOracle {
        g,
        d,
        b,
        a_stream,
        b_streams,
        y,
    })
}

/// Inverse of `Ief15Layout::unpack_out`: `y [T][nf]` LE f32 -> the C argument (`Geometry::pair_c_offset` tiles).
fn tile_in(g: &Ief15Gemm, y: &[u8]) -> Vec<u8> {
    let nf = 2 * CH;
    let (_, nw_n) = g.layout().wave_grid();
    let waves = g.layout().waves();
    let mut c = vec![0u8; g.design().args[2].bytes];
    let cseg = waves * 4 * 8192;
    for col in 0..8 {
        for wv in 0..waves {
            for rho in 0..4 {
                let (mw, nw) = (wv / nw_n, wv % nw_n);
                let base = col * cseg + (rho / 2) * waves * 2 * 8192 + (wv * 2 + rho % 2) * 8192;
                for mb in 0..8 {
                    for nb in 0..4 {
                        for i in 0..8 {
                            for jj in 0..8 {
                                let t = mw * 256 + (2 * (rho / 2) + col % 2) * 64 + mb * 8 + i;
                                let r = nw * 256 + (col / 2) * 64 + (rho % 2) * 32 + nb * 8 + jj;
                                let at = base + ((mb * 4 + nb) * 64 + i * 8 + jj) * 4;
                                c[at..at + 4].copy_from_slice(&y[(t * nf + r) * 4..][..4]);
                            }
                        }
                    }
                }
            }
        }
    }
    c
}

/// `v2`: launch `ief_silu2` (same DAG, low VGPR footprint) instead of `ief_silu`.
pub(super) struct GuGpu {
    m: [HipModule; 3],
    set: HipFunction,
    silu: HipFunction,
    epi: HipFunction,
    epi2: HipFunction,
    v2: std::cell::Cell<bool>,
}

/// The gate/up kernels (incumbent SET / SiLU entries and `ief_silu`) for other probes.
pub(super) fn silu_kernel(gpu: &Gpu) -> Result<GuGpu, String> {
    GuGpu::new(gpu)
}

impl GuGpu {
    fn new(gpu: &Gpu) -> Result<GuGpu, String> {
        let m_v2b = gpu.rt.load_module(V2B_IMAGE)?;
        let m_silu = gpu.rt.load_module(SILU_CO)?;
        let m_silu2 = gpu.rt.load_module(SILU2_CO)?;
        let (set, silu, epi, epi2) = (
            m_v2b.function(V2B_SET)?,
            m_v2b.function(V2B_SILU)?,
            m_silu.function("ief_silu")?,
            m_silu2.function("ief_silu2")?,
        );
        println!(
            "gate_up gpu: ief_silu fnv {:016x} ({} B), ief_silu2 fnv {:016x} ({} B)",
            fnv(SILU_CO),
            SILU_CO.len(),
            fnv(SILU2_CO),
            SILU2_CO.len()
        );
        Ok(GuGpu {
            m: [m_v2b, m_silu, m_silu2],
            set,
            silu,
            epi,
            epi2,
            v2: std::cell::Cell::new(false),
        })
    }
    /// Incumbent gate/up SiLU on hidden `[h0, H)`.
    fn silu(&self, gpu: &Gpu, wg: u64, wu: u64, xq: u64, y: u64, h0: usize) -> Result<(), String> {
        let (mut k, grid) = Self::silu_args(wg, wu, xq, y, h0);
        gpu.rt
            .launch_grid(&self.silu, grid, [512, 1, 1], V2B_LDS, &mut k)
    }
    fn silu_args(wg: u64, wu: u64, xq: u64, y: u64, h0: usize) -> ([u8; 44], [u32; 3]) {
        let mut k = [0u8; 44];
        put64(&mut k, 0, wg + (h0 * ROWG) as u64);
        put64(&mut k, 8, wu + (h0 * ROWG) as u64);
        put64(&mut k, 16, xq);
        put64(&mut k, 24, y + (h0 * 4) as u64);
        put32(&mut k, 32, H as u32);
        put32(&mut k, 36, KG as u32);
        put32(&mut k, 40, T as u32);
        (k, [(T / 256) as u32, (2 * (H - h0) / 256) as u32, 1])
    }
    /// Workgroup grid (x, y) of the windowed SiLU launch on hidden `[h0, H)` (the fold's drain set is its last workgroups).
    fn grid(h0: usize) -> [u32; 2] {
        let g = Self::silu_args(0, 0, 0, 0, h0).1;
        [g[0], g[1]]
    }
    /// `ief_gu_fold`: the incumbent SiLU window `[h0, H)` plus the flag-gated SiLU of the NPU's g|u (`e.c[j]`) into
    /// hidden `[j*CH, (j+1)*CH)` of `e.hbase`.
    #[allow(clippy::too_many_arguments)]
    fn fold(
        &self,
        gpu: &Gpu,
        wg: u64,
        wu: u64,
        xq: u64,
        y: u64,
        h0: usize,
        e: &fold::Ext,
    ) -> Result<(), String> {
        let (k, grid) = Self::silu_args(wg, wu, xq, y, h0);
        let mut k = fold::karg(&k, e);
        gpu.rt
            .launch_grid(&gpu.fold.gu, grid, [512, 1, 1], V2B_LDS, &mut k)
    }
    /// `ief_silu_wide`: the SiLU of all `e.j` NPU commands in one static launch.
    fn wide(&self, gpu: &Gpu, e: &fold::Ext) -> Result<(), String> {
        let mut k = fold::karg(&[], e);
        gpu.rt.launch_grid(
            &gpu.fold.silu_wide,
            [e.j * e.geo.upc / 64, 1, 1],
            [256, 1, 1],
            0,
            &mut k,
        )
    }
    /// Incumbent SET (pre-SiLU f32) of `rows` weight rows at `w` into `y [T][rows]`.
    fn set(&self, gpu: &Gpu, w: u64, xq: u64, y: u64, rows: usize) -> Result<(), String> {
        let mut k = [0u8; 40];
        put64(&mut k, 0, w);
        put64(&mut k, 8, xq);
        put64(&mut k, 16, y);
        put32(&mut k, 24, rows as u32);
        put32(&mut k, 28, KG as u32);
        put32(&mut k, 32, T as u32);
        gpu.rt.launch_grid(
            &self.set,
            [(T / 256) as u32, (rows / 256) as u32, 1],
            [512, 1, 1],
            V2B_LDS,
            &mut k,
        )
    }
    /// The incumbent SiLU DAG on one NPU command's tiled g / u (C arena `c`) into `h` hidden `[col0, col0 + CH)`.
    pub(super) fn epi(
        &self,
        gpu: &Gpu,
        c: u64,
        y: u64,
        col0: usize,
        g: &Ief15Gemm,
    ) -> Result<(), String> {
        let (mw, nw) = g.layout().wave_grid();
        let mut k = [0u8; 40];
        put64(&mut k, 0, c);
        put64(&mut k, 8, y);
        put32(&mut k, 16, H as u32);
        put32(&mut k, 20, col0 as u32);
        put32(&mut k, 24, g.layout().waves() as u32);
        put32(&mut k, 28, nw as u32);
        put32(&mut k, 32, (nw / 2) as u32);
        gpu.rt.launch_grid(
            if self.v2.get() { &self.epi2 } else { &self.epi },
            [nw as u32, 2 * mw as u32, 8],
            [256, 1, 1],
            0,
            &mut k,
        )
    }
}

fn cols_diff_h(got: &[u8], want: &[u8], c0: usize, c1: usize) -> usize {
    (0..T)
        .map(|t| {
            (c0..c1)
                .filter(|&r| got[(t * H + r) * 4..][..4] != want[(t * H + r) * 4..][..4])
                .count()
        })
        .sum()
}

pub fn run(cfg: &Cfg) -> Result<bool, String> {
    let inp = load(cfg)?;
    match cfg.mode.as_str() {
        "gpu" => gpu_mode(cfg, &inp),
        "coop" => coop_mode(cfg, &inp),
        "fold" => fold_mode(cfg, &inp),
        m => Err(format!("--proj gate_up has no mode {m}")),
    }
}

/// `--mode fold` (gate_up, GPU only), J = `--hid[0]` / 1024: `ief_gu_fold` / `ief_silu_wide` against the incumbent
/// window + J `ief_silu` passes, on the incumbent's own g, u of each chunk tiled into the NPU C layout (so the expected
/// h is the incumbent full launch's). Checks: the twin with no NPU work equals the full incumbent on `[n_h, H)` and
/// leaves `[0, n_h)` untouched; C written and flags published mid-kernel (3 rounds, the chunks rotated across the
/// arenas each round, so a stale GPU line of an earlier round would show) and preset; the wide tail. Then the
/// [`FOLD_VARIANTS`] timing.
fn fold_mode(cfg: &Cfg, inp: &GuInputs) -> Result<bool, String> {
    let j_count = cfg.hid[0] / CH;
    if j_count == 0 || j_count > 4 || cfg.hid[0] % CH != 0 {
        return Err(format!("--hid {}: 1024, 2048, 3072 or 4096", cfg.hid[0]));
    }
    let n_h = j_count * CH;
    let gpu = Gpu::new()?;
    gpu.reg.set(cfg.reg);
    let gu = GuGpu::new(&gpu)?;
    let rt = &gpu.rt;
    let sync = fold::FoldSync::new(&gpu)?;
    let (wg, wu, xq) = (gpu.up(&inp.wg)?, gpu.up(&inp.wu)?, gpu.up(&inp.xq)?);
    let y_full = rt.allocate(T * H * 4, None)?;
    let y = rt.allocate(T * H * 4, None)?;
    gu.silu(&gpu, wg.ptr(), wu.ptr(), xq.ptr(), y_full.ptr(), 0)?;
    rt.synchronize()?;
    let full = gpu.down(&y_full, T * H * 4)?;
    let g = Ief15Gemm::new(T, 2 * CH, KG, Control::Fast)?;
    let geo = fold::Geo::new(&g, true)?;
    // The incumbent's pre-SiLU g | u of every chunk, tiled into the NPU C layout.
    let mut tiled = Vec::new();
    {
        let (yg, yu) = (
            rt.allocate(T * CH * 4, None)?,
            rt.allocate(T * CH * 4, None)?,
        );
        for j in 0..j_count {
            gu.set(
                &gpu,
                wg.ptr() + (j * CH * ROWG) as u64,
                xq.ptr(),
                yg.ptr(),
                CH,
            )?;
            gu.set(
                &gpu,
                wu.ptr() + (j * CH * ROWG) as u64,
                xq.ptr(),
                yu.ptr(),
                CH,
            )?;
            rt.synchronize()?;
            let (gb, ub) = (gpu.down(&yg, T * CH * 4)?, gpu.down(&yu, T * CH * 4)?);
            let mut gu_y = vec![0u8; T * 2 * CH * 4];
            for t in 0..T {
                gu_y[t * 2 * CH * 4..][..CH * 4].copy_from_slice(&gb[t * CH * 4..][..CH * 4]);
                gu_y[(t * 2 * CH + CH) * 4..][..CH * 4]
                    .copy_from_slice(&ub[t * CH * 4..][..CH * 4]);
            }
            tiled.push(tile_in(&g, &gu_y));
        }
    }
    let cs: Vec<Shared> = (0..j_count)
        .map(|_| Shared::new(&gpu, None, g.design().args[2].bytes))
        .collect::<Result<_, _>>()?;
    let mut cp = [0u64; 4];
    for (j, c) in cs.iter().enumerate() {
        cp[j] = c.gpu();
    }
    let grid = GuGpu::grid(n_h);
    let ext = |round: u32, ctr: u64, j: usize, wide: bool| fold::Ext {
        c: cp,
        flags: sync.flags(),
        ctr,
        hbase: y.ptr(),
        round,
        j: j as u32,
        ldy: H as u32,
        geo,
        drain: cfg.drain as u32,
        grid,
        wide,
        chw: CH as u32,
        claim: cfg.claim as u32,
    };
    // NPU hidden of `got` vs the full incumbent h of chunk src[j] for arena j; whether [n_h, H) equals the incumbent.
    let npu_diff = |got: &[u8], src: &[usize]| -> usize {
        (0..T)
            .map(|t| {
                (0..j_count)
                    .map(|j| {
                        (0..CH * 4)
                            .filter(|&b| {
                                got[(t * H + j * CH) * 4 + b] != full[(t * H + src[j] * CH) * 4 + b]
                            })
                            .count()
                    })
                    .sum::<usize>()
            })
            .sum()
    };
    let ident: Vec<usize> = (0..j_count).collect();
    let mut ok = true;
    println!("fold gate_up J={j_count}: upc {} lgx {} lgy {} waves {} NW {} half {}; window grid {grid:?}, drain {} WGs, claim {}", geo.upc, geo.lgx, geo.lgy,
        geo.waves, geo.nw, geo.half, cfg.drain, cfg.claim);
    // 1. The twin with no NPU work.
    rt.memset(&y, POISON)?;
    let (r, slot) = sync.next()?;
    gu.fold(
        &gpu,
        wg.ptr(),
        wu.ptr(),
        xq.ptr(),
        y.ptr(),
        n_h,
        &ext(r, slot, 0, false),
    )?;
    rt.synchronize()?;
    let got = gpu.down(&y, T * H * 4)?;
    let own = cols_diff_h(&got, &full, n_h, H);
    let rest = (0..T)
        .map(|t| {
            got[t * H * 4..(t * H + n_h) * 4]
                .iter()
                .filter(|&&b| b != POISON)
                .count()
        })
        .sum::<usize>();
    let (sok, s) = sync.check(&gpu, r, 0, geo.upc)?;
    println!("fold twin J=0: hidden [{n_h},{H}) vs full incumbent mismatches {own}/{}, [0,{n_h}) bytes written {rest}; slot {s}", T * (H - n_h));
    ok &= own == 0 && rest == 0 && sok;
    let (e0, e1) = (rt.event()?, rt.event()?);
    // 2. C written and flags published while the kernel runs, chunks rotated across the arenas.
    for round_i in 0..3 {
        for c in &cs {
            c.fill(POISON);
        }
        rt.memset(&y, POISON)?;
        let src: Vec<usize> = (0..j_count).map(|j| (j + round_i) % j_count).collect();
        let (r, slot) = sync.next()?;
        rt.record(&e0)?;
        gu.fold(
            &gpu,
            wg.ptr(),
            wu.ptr(),
            xq.ptr(),
            y.ptr(),
            n_h,
            &ext(r, slot, j_count, false),
        )?;
        rt.record(&e1)?;
        let t0 = Instant::now();
        let mut pubs = Vec::new();
        for j in 0..j_count {
            std::thread::sleep(std::time::Duration::from_millis(3));
            cs[j].write(0, &tiled[src[j]]);
            sync.signal(j, r);
            pubs.push(format!("{:.2}", t0.elapsed().as_secs_f64() * 1e3));
        }
        let k_ms = rt.elapsed_ms(&e0, &e1)?;
        let got = gpu.down(&y, T * H * 4)?;
        let (own, npu) = (cols_diff_h(&got, &full, n_h, H), npu_diff(&got, &src));
        let (sok, s) = sync.check(&gpu, r, j_count, geo.upc)?;
        println!("fold late flags {round_i} (arena j <- chunk {src:?}): kernel {k_ms:.3} ms, flags published at [{}] ms after launch; \
            GPU hidden mismatches {own}, NPU hidden vs incumbent SiLU h of the chunk {npu}/{}; slot {s}", pubs.join(","), T * n_h);
        ok &= own == 0 && npu == 0 && sok;
    }
    // 3. Flags preset, and 4. the wide tail.
    for (j, c) in cs.iter().enumerate() {
        c.write(0, &tiled[j]);
    }
    rt.memset(&y, POISON)?;
    let (r, slot) = sync.next()?;
    for j in 0..j_count {
        sync.signal(j, r);
    }
    gu.fold(
        &gpu,
        wg.ptr(),
        wu.ptr(),
        xq.ptr(),
        y.ptr(),
        n_h,
        &ext(r, slot, j_count, false),
    )?;
    rt.synchronize()?;
    let got = gpu.down(&y, T * H * 4)?;
    let (own, npu) = (cols_diff_h(&got, &full, n_h, H), npu_diff(&got, &ident));
    let (sok, s) = sync.check(&gpu, r, j_count, geo.upc)?;
    rt.memset(&y, POISON)?;
    gu.wide(&gpu, &ext(0, 0, j_count, true))?;
    rt.synchronize()?;
    let got = gpu.down(&y, T * H * 4)?;
    let wnpu = npu_diff(&got, &ident);
    let wrest = (0..T)
        .map(|t| {
            got[(t * H + n_h) * 4..(t + 1) * H * 4]
                .iter()
                .filter(|&&b| b != POISON)
                .count()
        })
        .sum::<usize>();
    println!("fold preset flags: GPU hidden mismatches {own}, NPU hidden {npu}, slot {s}; ief_silu_wide: NPU hidden vs incumbent SiLU h {wnpu}/{}, \
        [{n_h},{H}) bytes written {wrest}", T * n_h);
    ok &= own == 0 && npu == 0 && sok && wnpu == 0 && wrest == 0;
    if !ok {
        return Ok(false);
    }
    let prep = || Ok(());
    let sep = || -> Result<(), String> {
        for (j, c) in cs.iter().enumerate() {
            gu.epi(&gpu, c.gpu(), y.ptr(), j * CH, &g)?;
        }
        Ok(())
    };
    let run = |v: usize| -> Result<(), String> {
        let win = || gu.silu(&gpu, wg.ptr(), wu.ptr(), xq.ptr(), y.ptr(), n_h);
        match v {
            0 => win(),
            1 => {
                win()?;
                sep()
            }
            2 => {
                win()?;
                gu.wide(&gpu, &ext(0, 0, j_count, true))
            }
            3 | 4 => {
                let (r, slot) = sync.next()?;
                for j in 0..j_count {
                    sync.signal(j, r);
                }
                gu.fold(
                    &gpu,
                    wg.ptr(),
                    wu.ptr(),
                    xq.ptr(),
                    y.ptr(),
                    n_h,
                    &ext(r, slot, if v == 3 { j_count } else { 0 }, false),
                )
            }
            5 => sep(),
            _ => gu.wide(&gpu, &ext(0, 0, j_count, true)),
        }
    };
    fold_timing(
        rt,
        cfg.iters,
        &format!("proj=gate_up J={j_count} reg={:#x}", cfg.reg),
        &prep,
        &run,
    )?;
    let (sok, s) = sync.check(&gpu, sync.last(), 0, geo.upc)?;
    println!("fold last timed round slot {s}");
    let _ = &gu.m;
    Ok(ok && sok)
}

fn gpu_mode(cfg: &Cfg, inp: &GuInputs) -> Result<bool, String> {
    let gpu = Gpu::new()?;
    let gu = GuGpu::new(&gpu)?;
    let rt = &gpu.rt;
    let (wg, wu, xq) = (gpu.up(&inp.wg)?, gpu.up(&inp.wu)?, gpu.up(&inp.xq)?);
    let y_full = rt.allocate(T * H * 4, None)?;
    let y_win = rt.allocate(T * H * 4, None)?;
    let mut ok = true;
    gu.silu(&gpu, wg.ptr(), wu.ptr(), xq.ptr(), y_full.ptr(), 0)?;
    rt.synchronize()?;
    let full = gpu.down(&y_full, T * H * 4)?;
    println!("gate_up full SiLU: h fnv {:016x}", fnv(&full));
    for &h0 in &cfg.hid {
        if h0 == 0 {
            continue;
        }
        rt.memset(&y_win, POISON)?;
        gu.silu(&gpu, wg.ptr(), wu.ptr(), xq.ptr(), y_win.ptr(), h0)?;
        rt.synchronize()?;
        let win = gpu.down(&y_win, T * H * 4)?;
        let own = cols_diff_h(&win, &full, h0, H);
        let rest = (0..T)
            .map(|t| {
                win[t * H * 4..(t * H + h0) * 4]
                    .iter()
                    .filter(|&&b| b != POISON)
                    .count()
            })
            .sum::<usize>();
        println!("gate_up window identity hidden [{h0},{H}): mismatches vs full {own}/{}, NPU region bytes written {rest}", T * (H - h0));
        ok &= own == 0 && rest == 0;
    }
    let (e0, e1) = (rt.event()?, rt.event()?);
    let time = |h0: usize| -> Result<f64, String> {
        rt.record(&e0)?;
        for _ in 0..cfg.reps {
            gu.silu(&gpu, wg.ptr(), wu.ptr(), xq.ptr(), y_win.ptr(), h0)?;
        }
        rt.record(&e1)?;
        Ok(f64::from(rt.elapsed_ms(&e0, &e1)?) / cfg.reps as f64)
    };
    time(0)?;
    for &h0 in &cfg.hid {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for blk in 0..cfg.blocks {
            if blk % 2 == 0 {
                a.push(time(0)?);
                b.push(time(h0)?);
            } else {
                b.push(time(h0)?);
                a.push(time(0)?);
            }
        }
        println!("GUTIME npu_hidden={h0} full_ms={:.4} window_ms={:.4} ratio={:.4} full=[{}] window=[{}]",
            median(&a), median(&b), median(&b) / median(&a), fmt_ms(&a), fmt_ms(&b));
    }
    // Kernels vs the CPU oracle: D/b and the A stream (E = 40, NW = 8), and ief_silu vs the incumbent's SiLU on the
    // incumbent's own g / u (SET of the chunk's G and U rows, tiled into the NPU C layout).
    let o = oracle(inp, 1, false)?;
    let g = &o.g;
    let dc = rt.allocate(EG * T * 2, None)?;
    let bc = rt.allocate(T * 2, None)?;
    let dcomp = rt.allocate(EG * T * 4, None)?;
    // Incumbent g / u (SET of G and U rows) of chunks 0 and 1, tiled into the NPU C layout.
    let yg = rt.allocate(T * CH * 4, None)?;
    let yu = rt.allocate(T * CH * 4, None)?;
    let mut tiled = Vec::new();
    for j in 0..2 {
        gu.set(
            &gpu,
            wg.ptr() + (j * CH * ROWG) as u64,
            xq.ptr(),
            yg.ptr(),
            CH,
        )?;
        gu.set(
            &gpu,
            wu.ptr() + (j * CH * ROWG) as u64,
            xq.ptr(),
            yu.ptr(),
            CH,
        )?;
        rt.synchronize()?;
        let (g_b, u_b) = (gpu.down(&yg, T * CH * 4)?, gpu.down(&yu, T * CH * 4)?);
        let mut gu_y = vec![0u8; T * 2 * CH * 4];
        for t in 0..T {
            gu_y[t * 2 * CH * 4..][..CH * 4].copy_from_slice(&g_b[t * CH * 4..][..CH * 4]);
            gu_y[(t * 2 * CH + CH) * 4..][..CH * 4].copy_from_slice(&u_b[t * CH * 4..][..CH * 4]);
        }
        let tl = tile_in(g, &gu_y);
        if g.unpack_out(&tl) != gu_y {
            return Err("tile_in is not the inverse of unpack_out".into());
        }
        tiled.push(tl);
    }
    // Both host registrations of the shared pages: default (coherent) and hipExtHostRegisterCoarseGrained.
    for reg in [0u32, 0x8] {
        gpu.reg.set(reg);
        let a_sh = Shared::new(&gpu, None, g.design().args[0].bytes)?;
        let c_sh = Shared::new(&gpu, None, tiled[0].len())?;
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
                count_diff(&gpu.down(&dc, EG * T * 2)?, &le16(&o.d)).0,
                count_diff(&gpu.down(&bc, T * 2)?, &lei16(&o.b)).0,
            );
            let (ad, _) = count_diff(&a_sh.read(0, o.a_stream.len()), &o.a_stream);
            println!("gate_up kernels reg {reg:#x} glue {}: D {dd} b {bd} mismatching B; A stream ({}) {ad}/{} mismatching B",
                if v2 { "v2" } else { "v1" }, if wb { "wb" } else { "stream" }, o.a_stream.len());
            ok &= dd == 0 && bd == 0 && ad == 0;
        }
        // ief_silu reads chunk 0, then the CPU overwrites the same pages with chunk 1 (a stale GPU cache line would
        // reproduce chunk 0's values): both must equal the incumbent SiLU h on their hidden columns.
        for v2 in [false, true] {
            gu.v2.set(v2);
            rt.memset(&y_win, POISON)?;
            for j in 0..2 {
                c_sh.write(0, &tiled[j]);
                gu.epi(&gpu, c_sh.gpu(), y_win.ptr(), j * CH, g)?;
                rt.synchronize()?;
                let sd = cols_diff_h(&gpu.down(&y_win, T * H * 4)?, &full, j * CH, (j + 1) * CH);
                println!("gate_up kernels reg {reg:#x} silu {}: on incumbent SET g/u (chunk {j}, same pages) vs incumbent SiLU h: mismatches {sd}/{}",
                    if v2 { "v2" } else { "v1" }, T * CH);
                ok &= sd == 0;
            }
        }
        let (e2, e3) = (rt.event()?, rt.event()?);
        let (mut tdb, mut tpk, mut tep, mut tv2, mut te2) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..cfg.reps + 1 {
            gu.v2.set(false);
            rt.record(&e0)?;
            gpu.db(xq.ptr(), dc.ptr(), bc.ptr(), EG)?;
            rt.record(&e1)?;
            gpu.pack(xq.ptr(), a_sh.gpu(), dc.ptr(), bc.ptr(), g, true)?;
            rt.record(&e2)?;
            gu.epi(&gpu, c_sh.gpu(), y_win.ptr(), 0, g)?;
            rt.record(&e3)?;
            tdb.push(f64::from(rt.elapsed_ms(&e0, &e1)?));
            tpk.push(f64::from(rt.elapsed_ms(&e1, &e2)?));
            tep.push(f64::from(rt.elapsed_ms(&e2, &e3)?));
            gu.v2.set(true);
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
            gu.epi(&gpu, c_sh.gpu(), y_win.ptr(), 0, g)?;
            rt.record(&e2)?;
            tv2.push(f64::from(rt.elapsed_ms(&e0, &e1)?));
            te2.push(f64::from(rt.elapsed_ms(&e1, &e2)?));
        }
        println!("GUKTIME reg={reg:#x} v1_db_ms={:.4} v1_pack_wb_ms={:.4} v2_prep_wb_ms={:.4} silu_epi_ms_per_cmd={:.4} silu2_epi_ms_per_cmd={:.4} (A stream {} B, C per cmd {} B)",
            median(&tdb[1..]), median(&tpk[1..]), median(&tv2[1..]), median(&tep[1..]), median(&te2[1..]), o.a_stream.len(), tiled[0].len());
    }
    Ok(ok)
}

fn coop_mode(cfg: &Cfg, inp: &GuInputs) -> Result<bool, String> {
    let j_count = cfg.hid[0] / CH;
    if j_count == 0 || cfg.hid[0] % CH != 0 {
        return Err(format!("--hid {}: a positive multiple of {CH}", cfg.hid[0]));
    }
    if cfg.tails.iter().any(|&t| t != fold::Tail::Sep) && j_count > 4 {
        return Err("--tail wide|fold supports J <= 4".into());
    }
    let n_h = j_count * CH;
    let gpu = Gpu::new()?;
    gpu.reg.set(cfg.reg);
    let gu = GuGpu::new(&gpu)?;
    gu.v2.set(cfg.silu2);
    let rt = &gpu.rt;
    let dev = open_npu(&gpu)?;
    let o = oracle(inp, j_count, true)?;
    let g = &o.g;
    let d = g.design();
    // NPU: one context and design, J commands (own B, C, cmd BO), one shared A arena.
    let ntiles = dev.meta.cols as u32 * dev.meta.core.0 as u32;
    let ctx = HwCtx::create(&dev, ntiles, 2048)?;
    let pdi = dev.dev_bo(&d.pdi)?;
    ctx.config_cu(&pdi)?;
    let insts = dev.dev_bo(&d.insts)?;
    let a_sh = Shared::new(&gpu, Some(&dev), d.args[0].bytes)?;
    let mut bs = Vec::new();
    let mut cs = Vec::new();
    let mut cmds = Vec::new();
    for j in 0..j_count {
        let mut b = dev.shmem_bo(d.args[1].bytes)?;
        b.as_mut_slice().copy_from_slice(&o.b_streams[j]);
        b.flush();
        bs.push(b);
        cs.push(Shared::new(&gpu, Some(&dev), d.args[2].bytes)?);
        cmds.push(dev.cmd_bo()?);
    }
    // Submits all J commands, then waits for them in order; `done(j)` right after command j completed.
    let run_npu = |cmds: &mut Vec<Bo>, done: &dyn Fn(usize)| -> Result<f64, String> {
        let t0 = Instant::now();
        let mut seqs = Vec::new();
        for (j, cmd) in cmds.iter_mut().enumerate() {
            seqs.push(ctx.submit(cmd, &insts, &[a_sh.bo(), &bs[j], cs[j].bo()])?);
        }
        for (j, cmd) in cmds.iter().enumerate() {
            if ctx.wait(cmd, seqs[j], cfg.timeout_ms)? != ERT_STATE_COMPLETED {
                return Err(format!("NPU command {j} did not complete"));
            }
            done(j);
        }
        Ok(t0.elapsed().as_secs_f64() * 1e3)
    };
    let (wg, wu, xq) = (gpu.up(&inp.wg)?, gpu.up(&inp.wu)?, gpu.up(&inp.xq)?);
    let h_alone = rt.allocate(T * H * 4, None)?;
    let h_coop = rt.allocate(T * H * 4, None)?;
    let h_win = rt.allocate(T * H * 4, None)?;
    let dc = rt.allocate(EG * T * 2, None)?;
    let bc = rt.allocate(T * 2, None)?;
    let dcomp = rt.allocate(EG * T * 4, None)?;
    let ev: Vec<HipEvent> = (0..4).map(|_| rt.event()).collect::<Result<_, _>>()?;
    let sync = fold::FoldSync::new(&gpu)?;
    let geo = fold::Geo::new(g, true)?;
    let grid = GuGpu::grid(n_h);
    let mut cp = [0u64; 4];
    for (j, c) in cs.iter().enumerate().take(4) {
        cp[j] = c.gpu();
    }
    let ext = |round: u32, ctr: u64, wide: bool| fold::Ext {
        c: cp,
        flags: sync.flags(),
        ctr,
        hbase: h_coop.ptr(),
        round,
        j: j_count as u32,
        ldy: H as u32,
        geo,
        drain: cfg.drain as u32,
        grid,
        wide,
        chw: CH as u32,
        claim: cfg.claim as u32,
    };
    let ext_wide = ext(0, 0, true);
    let last_round = std::cell::Cell::new(0u32);
    println!(
        "gate_up coop tails: {:?} (drain {} WGs, claim {}, window grid {grid:?})",
        cfg.tails, cfg.drain, cfg.claim
    );
    // h of the IEF15 oracle g/u through the incumbent SiLU DAG (expected NPU columns), and the incumbent's pre-SiLU g/u.
    let h_ref = rt.allocate(T * H * 4, None)?;
    rt.memset(&h_ref, 0)?;
    let mut g_inc = Vec::new();
    {
        let tmp = Shared::new(&gpu, None, d.args[2].bytes)?;
        let (yg, yu) = (
            rt.allocate(T * CH * 4, None)?,
            rt.allocate(T * CH * 4, None)?,
        );
        for j in 0..j_count {
            tmp.write(0, &tile_in(g, &o.y[j]));
            gu.epi(&gpu, tmp.gpu(), h_ref.ptr(), j * CH, g)?;
            gu.set(
                &gpu,
                wg.ptr() + (j * CH * ROWG) as u64,
                xq.ptr(),
                yg.ptr(),
                CH,
            )?;
            gu.set(
                &gpu,
                wu.ptr() + (j * CH * ROWG) as u64,
                xq.ptr(),
                yu.ptr(),
                CH,
            )?;
            rt.synchronize()?;
            let (gb, ub) = (gpu.down(&yg, T * CH * 4)?, gpu.down(&yu, T * CH * 4)?);
            let mut y = vec![0u8; T * 2 * CH * 4];
            for t in 0..T {
                y[t * 2 * CH * 4..][..CH * 4].copy_from_slice(&gb[t * CH * 4..][..CH * 4]);
                y[(t * 2 * CH + CH) * 4..][..CH * 4].copy_from_slice(&ub[t * CH * 4..][..CH * 4]);
            }
            g_inc.push(y);
        }
    }
    let h_ref_b = gpu.down(&h_ref, T * H * 4)?;
    // NPU alone (no GPU work), J commands per round.
    a_sh.write(0, &o.a_stream);
    let npu_alone: Vec<f64> = (0..4)
        .map(|_| run_npu(&mut cmds, &|_| {}))
        .collect::<Result<_, _>>()?;
    let alone = || -> Result<f64, String> {
        rt.record(&ev[0])?;
        gu.silu(&gpu, wg.ptr(), wu.ptr(), xq.ptr(), h_alone.ptr(), 0)?;
        rt.record(&ev[1])?;
        Ok(f64::from(rt.elapsed_ms(&ev[0], &ev[1])?))
    };
    let window = || -> Result<f64, String> {
        rt.record(&ev[0])?;
        gu.silu(&gpu, wg.ptr(), wu.ptr(), xq.ptr(), h_win.ptr(), n_h)?;
        rt.record(&ev[1])?;
        Ok(f64::from(rt.elapsed_ms(&ev[0], &ev[1])?))
    };
    let coop = |cmds: &mut Vec<Bo>, tail: fold::Tail| -> Result<[f64; 5], String> {
        rt.synchronize()?;
        rt.record(&ev[0])?;
        gpu.prep(
            cfg.glue2,
            xq.ptr(),
            a_sh.gpu(),
            dc.ptr(),
            bc.ptr(),
            dcomp.ptr(),
            g,
            cfg.wb,
        )?;
        rt.record(&ev[1])?;
        let round = if tail == fold::Tail::Fold {
            let (r, slot) = sync.next()?;
            gu.fold(
                &gpu,
                wg.ptr(),
                wu.ptr(),
                xq.ptr(),
                h_coop.ptr(),
                n_h,
                &ext(r, slot, false),
            )?;
            last_round.set(r);
            Some(r)
        } else {
            gu.silu(&gpu, wg.ptr(), wu.ptr(), xq.ptr(), h_coop.ptr(), n_h)?;
            None
        };
        rt.record(&ev[2])?;
        rt.sync_event(&ev[1])?;
        let busy = match run_npu(cmds, &|j| {
            if let Some(r) = round {
                sync.signal(j, r)
            }
        }) {
            Ok(b) => b,
            Err(e) => {
                if let Some(r) = round {
                    sync.release(r);
                    rt.synchronize()?;
                }
                return Err(e);
            }
        };
        match tail {
            fold::Tail::Sep => {
                for j in 0..j_count {
                    gu.epi(&gpu, cs[j].gpu(), h_coop.ptr(), j * CH, g)?;
                }
            }
            fold::Tail::Wide => gu.wide(&gpu, &ext_wide)?,
            fold::Tail::Fold => {}
        }
        rt.record(&ev[3])?;
        Ok([
            f64::from(rt.elapsed_ms(&ev[0], &ev[3])?),
            f64::from(rt.elapsed_ms(&ev[0], &ev[1])?),
            f64::from(rt.elapsed_ms(&ev[1], &ev[2])?),
            f64::from(rt.elapsed_ms(&ev[2], &ev[3])?),
            busy,
        ])
    };
    let verify = |label: &str, tail: fold::Tail| -> Result<bool, String> {
        let ha = gpu.down(&h_alone, T * H * 4)?;
        let hc = gpu.down(&h_coop, T * H * 4)?;
        let gpu_cols = cols_diff_h(&hc, &ha, n_h, H);
        let (dd, bd) = (
            count_diff(&gpu.down(&dc, EG * T * 2)?, &le16(&o.d)).0,
            count_diff(&gpu.down(&bc, T * 2)?, &lei16(&o.b)).0,
        );
        let (ad, _) = count_diff(&a_sh.read(0, o.a_stream.len()), &o.a_stream);
        let cd: usize = (0..j_count)
            .map(|j| count_diff(&g.unpack_out(&cs[j].read(0, d.args[2].bytes)), &o.y[j]).0)
            .sum();
        let npu_cols = cols_diff_h(&hc, &h_ref_b, 0, n_h);
        println!("gate_up verify {label}: GPU hidden [{n_h},{H}) vs GPU-alone mismatches {gpu_cols}/{}; D/b vs CPU {dd}/{bd}; A stream vs CPU {ad}; \
            NPU g/u C vs CPU IEF15 oracle {cd} B; NPU hidden h vs SiLU-DAG(oracle g,u) {npu_cols}/{}", T * (H - n_h), T * n_h);
        if label.starts_with("first") && tail == cfg.tails[0] {
            let (mut se, mut sr, mut ma) = (0f64, 0f64, 0f64);
            for j in 0..j_count {
                for i in 0..T * 2 * CH {
                    let (a, b) = (
                        f64::from(f32_at(&o.y[j], i)),
                        f64::from(f32_at(&g_inc[j], i)),
                    );
                    se += (a - b) * (a - b);
                    sr += b * b;
                    ma = ma.max((a - b).abs());
                }
            }
            println!(
                "GUERR pre-SiLU g,u NPU cols (IEF15) vs incumbent: nrmse={:.4e} maxabs={ma:.4e}",
                (se / sr).sqrt()
            );
            let (mut se, mut sr, mut ma, mut mr) = (0f64, 0f64, 0f64, 0f64);
            for t in 0..T {
                for r in 0..n_h {
                    let (a, b) = (
                        f64::from(f32_at(&hc, t * H + r)),
                        f64::from(f32_at(&ha, t * H + r)),
                    );
                    se += (a - b) * (a - b);
                    sr += b * b;
                    ma = ma.max((a - b).abs());
                    if b != 0.0 {
                        mr = mr.max((a - b).abs() / b.abs());
                    }
                }
            }
            let all = {
                let (mut se2, mut sr2) = (0f64, 0f64);
                for i in 0..T * H {
                    let (a, b) = (f64::from(f32_at(&hc, i)), f64::from(f32_at(&ha, i)));
                    se2 += (a - b) * (a - b);
                    sr2 += b * b;
                }
                (se2 / sr2).sqrt()
            };
            println!("GUERR h NPU cols co-op vs GPU-alone: nrmse={:.4e} maxabs={ma:.4e} maxrel={mr:.4e}; all {H} cols nrmse={all:.4e}", (se / sr).sqrt());
        }
        let mut all = gpu_cols == 0 && dd == 0 && bd == 0 && ad == 0 && cd == 0 && npu_cols == 0;
        if tail == fold::Tail::Fold {
            let (sok, s) = sync.check(&gpu, last_round.get(), j_count, geo.upc)?;
            println!(
                "gate_up verify {label}: fold slot {s}: {}",
                if sok { "ok" } else { "BAD" }
            );
            all &= sok;
        }
        Ok(all)
    };
    let mut ok = true;
    alone()?;
    window()?;
    for &t in &cfg.tails {
        a_sh.fill(POISON);
        for c in &cs {
            c.fill(POISON);
        }
        rt.memset(&h_coop, POISON)?;
        let first = coop(&mut cmds, t)?;
        println!(
            "gate_up coop first ({}): total {:.3} ms",
            t.name(),
            first[0]
        );
        ok &= verify(&format!("first {}", t.name()), t)?;
    }
    if !ok {
        return Ok(false);
    }
    // Timed rounds: [alone, window, coop(tail)..] forward on even rounds, reversed on odd ones (ABBA).
    let nv = 2 + cfg.tails.len();
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); nv];
    let mut therm: Vec<Vec<[f64; 2]>> = vec![Vec::new(); nv];
    let mut parts: Vec<Vec<[f64; 5]>> = vec![Vec::new(); cfg.tails.len()];
    for it in 0..cfg.iters {
        for s in 0..nv {
            let v = if it % 2 == 0 { s } else { nv - 1 - s };
            let ms = match v {
                0 => alone()?,
                1 => window()?,
                _ => {
                    let p = coop(&mut cmds, cfg.tails[v - 2])?;
                    parts[v - 2].push(p);
                    p[0]
                }
            };
            times[v].push(ms);
            therm[v].push(fold::thermal());
        }
    }
    // Last verified layer of every tail, on poisoned C arenas and h.
    for &t in &cfg.tails {
        for c in &cs {
            c.fill(POISON);
        }
        rt.memset(&h_coop, POISON)?;
        coop(&mut cmds, t)?;
        ok &= verify(&format!("last {}", t.name()), t)?;
    }
    let (ma, mw) = (median(&times[0]), median(&times[1]));
    println!(
        "GUCOOPTHERM J={j_count} alone:{} window:{} {}",
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
        "GUCOOPRAW J={j_count} alone=[{}] window=[{}] npu_alone=[{}]",
        fmt_ms(&times[0]),
        fmt_ms(&times[1]),
        fmt_ms(&npu_alone)
    );
    for (i, t) in cfg.tails.iter().enumerate() {
        let col = |k: usize| -> Vec<f64> { parts[i].iter().map(|p| p[k]).collect() };
        let mc = median(&times[i + 2]);
        println!("GUCOOP n_h={n_h} J={j_count} tail={} gpu_alone_ms={ma:.4} window_alone_ms={mw:.4} coop_ms={mc:.4} delta_ms={:.4} speedup={:.4} \
            pack_ms={:.4} window_concurrent_ms={:.4} alpha={:.4} tail_ms={:.4} npu_busy_ms={:.4} npu_alone_ms={:.4}",
            t.name(), mc - ma, ma / mc, median(&col(1)), median(&col(2)), median(&col(2)) / mw, median(&col(3)), median(&col(4)), median(&npu_alone));
        println!(
            "GUCOOPRAW J={j_count} tail={} coop=[{}] pack=[{}] wconc=[{}] tail=[{}] npu=[{}]",
            t.name(),
            fmt_ms(&times[i + 2]),
            fmt_ms(&col(1)),
            fmt_ms(&col(2)),
            fmt_ms(&col(3)),
            fmt_ms(&col(4))
        );
    }
    let _ = &gu.m;
    Ok(ok)
}
