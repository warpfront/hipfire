// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Host side of `native/ief_fold_gfx1151` (coop-epi-fuse): the NPU-column epilogue (gate_up SiLU / down addc) either
//! folded into a PM twin of the incumbent window kernel or run as one merged wide tail launch.
//!
//! * `ief_gu_fold` / `ief_add_fold`: the incumbent gate_up SiLU / down ADD entry, instruction for instruction (its code
//!   bytes up to the final dealloc+endpgm equal the vendored incumbent's), followed by `fold_sched.inc`: every wave
//!   that finishes its GEMM tile takes a ticket, then claims 4-wave-unit chunks of NPU command `j`'s epilogue once
//!   the host has published `flags[j] == round` (after the NPU command completed). The last `drain` tickets keep
//!   polling until every command is ready, so all NPU columns are written before the kernel ends.
//! * `ief_silu_wide` / `ief_addc_wide`: the same claim body in static mode, one launch for all J commands.
//!
//! The per-element arithmetic is the separate-pass kernels' (`ief_silu`: the incumbent SiLU DAG verbatim; `ief_addc`:
//! `RN(x + Y)`), and so is the C tile / output index math, so results are byte-equal to `ief_silu` / `ief_addc`.
use super::*;
use std::cell::Cell;

const FOLD_CO: &[u8] = include_bytes!("../../../native/coop27/ief_fold_gfx1151.co");
pub(super) const KARG: usize = 160;
/// Bytes per round slot (`claim[j]` at `j*128`, status at 512, one L2 line each) and rounds per [`FoldSync`].
const SLOT: usize = 640;
const SLOTS: usize = 4096;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Tail {
    Sep,
    Wide,
    Fold,
}

impl Tail {
    /// `sep|wide|fold[,..]`: the tails measured (interleaved) in one co-op process.
    pub(super) fn parse_list(s: Option<&String>) -> Result<Vec<Tail>, String> {
        let s = s.ok_or("--tail needs a value")?;
        let v = s
            .split(',')
            .map(|p| match p.trim() {
                "sep" => Ok(Tail::Sep),
                "wide" => Ok(Tail::Wide),
                "fold" => Ok(Tail::Fold),
                o => Err(format!("--tail expects sep|wide|fold[,..], got {o:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if v.is_empty() {
            return Err("--tail: empty list".into());
        }
        Ok(v)
    }
    pub(super) fn name(self) -> &'static str {
        match self {
            Tail::Sep => "sep",
            Tail::Wide => "wide",
            Tail::Fold => "fold",
        }
    }
}

/// GPU edge temperature (°C) and sclk (MHz) from the device's hwmon, sampled right after a timed launch (NaN when
/// unreadable). Halo throttles near 95 °C within ~1 s of sustained load, so every timed variant reports its own.
pub(super) fn thermal() -> [f64; 2] {
    let Ok(rd) = std::fs::read_dir(format!("/sys/bus/pci/devices/{PCI}/hwmon")) else {
        return [f64::NAN; 2];
    };
    for e in rd.flatten() {
        let p = e.path();
        let val = |f: &str| {
            std::fs::read_to_string(p.join(f))
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok())
        };
        if let (Some(t), Some(f)) = (val("temp1_input"), val("freq1_input")) {
            return [t / 1000.0, f / 1e6];
        }
    }
    [f64::NAN; 2]
}

/// `t=<median>/<max>C sclk=<median>/<min>MHz` of thermal samples.
pub(super) fn therm_summary(s: &[[f64; 2]]) -> String {
    let s: Vec<[f64; 2]> = s
        .iter()
        .copied()
        .filter(|x| x[0].is_finite() && x[1].is_finite())
        .collect();
    if s.is_empty() {
        return "t=-".into();
    }
    let t: Vec<f64> = s.iter().map(|x| x[0]).collect();
    let f: Vec<f64> = s.iter().map(|x| x[1]).collect();
    format!(
        "t={:.0}/{:.0}C sclk={:.0}/{:.0}MHz",
        median(&t),
        t.iter().cloned().fold(f64::MIN, f64::max),
        median(&f),
        f.iter().cloned().fold(f64::MAX, f64::min)
    )
}

pub(super) struct FoldFns {
    _m: HipModule,
    pub gu: HipFunction,
    pub add: HipFunction,
    pub silu_wide: HipFunction,
    pub addc_wide: HipFunction,
}

impl FoldFns {
    pub(super) fn new(rt: &HipRuntime) -> Result<FoldFns, String> {
        let m = rt.load_module(FOLD_CO)?;
        let (gu, add, silu_wide, addc_wide) = (
            m.function("ief_gu_fold")?,
            m.function("ief_add_fold")?,
            m.function("ief_silu_wide")?,
            m.function("ief_addc_wide")?,
        );
        Ok(FoldFns {
            _m: m,
            gu,
            add,
            silu_wide,
            addc_wide,
        })
    }
    pub(super) fn describe() -> String {
        format!(
            "fold kernels fnv {:016x} ({} B)",
            fnv(FOLD_CO),
            FOLD_CO.len()
        )
    }
}

/// Epilogue geometry of an IEF15 design: the separate-pass grid is `[GX, 2*MW, 8]` workgroups of 8 waves, `GX = NW`
/// for SiLU (g | u halves) and `2*NW` for addc; one wave-unit is one wave of that grid.
#[derive(Clone, Copy)]
pub(super) struct Geo {
    pub waves: u32,
    pub nw: u32,
    pub half: u32,
    pub lgx: u32,
    pub lgy: u32,
    pub upc: u32,
}

impl Geo {
    pub(super) fn new(g: &Ief15Gemm, silu: bool) -> Result<Geo, String> {
        let (mw, nw) = g.layout().wave_grid();
        let gx = if silu { nw } else { 2 * nw };
        let gy = 2 * mw;
        if !gx.is_power_of_two() || !gy.is_power_of_two() {
            return Err(format!(
                "fold geometry needs power-of-two grid, got {gx} x {gy}"
            ));
        }
        Ok(Geo {
            waves: g.layout().waves() as u32,
            nw: nw as u32,
            half: if silu { (nw / 2) as u32 } else { 0 },
            lgx: gx.trailing_zeros(),
            lgy: gy.trailing_zeros(),
            upc: (gx * gy * 64) as u32,
        })
    }
}

/// Kernarg extension (offsets 0x30.. of the 160 B kernarg; the first 48 B are the incumbent's own arguments).
/// `grid`: the launch's workgroup grid (x, y) — the fold's drain set is the last `drain` workgroups by linear id
/// `y*grid.x + x`; `claim`: wave-units per atomic claim (multiple of 8).
pub(super) struct Ext {
    pub c: [u64; 4],
    pub flags: u64,
    pub ctr: u64,
    pub hbase: u64,
    pub round: u32,
    pub j: u32,
    pub ldy: u32,
    pub geo: Geo,
    pub drain: u32,
    pub grid: [u32; 2],
    pub wide: bool,
    pub chw: u32,
    pub claim: u32,
}

pub(super) fn karg(base: &[u8], e: &Ext) -> [u8; KARG] {
    assert!(base.len() <= 48 && e.claim % 8 == 0 && e.claim > 0 && e.grid[0] < 1 << 16);
    let mut k = [0u8; KARG];
    k[..base.len()].copy_from_slice(base);
    for (i, c) in e.c.iter().enumerate() {
        put64(&mut k, 0x30 + 8 * i, *c);
    }
    put64(&mut k, 0x50, e.flags);
    put64(&mut k, 0x58, e.ctr);
    put64(&mut k, 0x60, e.hbase);
    let g = e.geo;
    let mode = u32::from(e.wide) | e.grid[0] << 16;
    for (i, v) in [
        e.round,
        e.j,
        e.ldy,
        g.waves,
        g.nw,
        g.half,
        g.lgx,
        g.lgy,
        g.upc,
        e.drain,
        e.grid[0] * e.grid[1],
        mode,
        e.chw,
        e.claim,
    ]
    .iter()
    .enumerate()
    {
        put32(&mut k, 0x68 + 4 * i, *v);
    }
    k
}

/// Flags page (anonymous host page, `hipHostRegister` default flags: fine-grained, polled glc dlc by the GPU; the CPU
/// stores + clflushes) and the per-round device slots. Rounds are 1-based and monotonic, so no flag is ever reset.
pub(super) struct FoldSync {
    flags: Shared,
    ctr: HipBuffer,
    round: Cell<u32>,
}

impl FoldSync {
    pub(super) fn new(gpu: &Gpu) -> Result<FoldSync, String> {
        let prev = gpu.reg.get();
        gpu.reg.set(0);
        let flags = Shared::new(gpu, None, 4096);
        gpu.reg.set(prev);
        let flags = flags?;
        flags.fill(0);
        let ctr = gpu.rt.allocate(SLOT * SLOTS, None)?;
        gpu.rt.memset(&ctr, 0)?;
        Ok(FoldSync {
            flags,
            ctr,
            round: Cell::new(0),
        })
    }
    /// A fresh round: (round, its slot address).
    pub(super) fn next(&self) -> Result<(u32, u64), String> {
        let r = self.round.get() + 1;
        if r as usize >= SLOTS {
            return Err("fold rounds exhausted".into());
        }
        self.round.set(r);
        Ok((r, self.ctr.ptr() + (r as usize * SLOT) as u64))
    }
    pub(super) fn flags(&self) -> u64 {
        self.flags.gpu()
    }
    /// The most recent round handed out by [`FoldSync::next`].
    pub(super) fn last(&self) -> u32 {
        self.round.get()
    }
    /// Publish "NPU command j of `round` is complete".
    pub(super) fn signal(&self, j: usize, round: u32) {
        self.flags.write(4 * j, &round.to_le_bytes());
    }
    /// Error path: release every waiting wave of `round` (the kernel then finishes with garbage NPU columns).
    pub(super) fn release(&self, round: u32) {
        for j in 0..4 {
            self.signal(j, round);
        }
    }
    /// Slot census of `round` after its kernel completed: (claims of commands `0..j` all >= upc, status == 0).
    /// Downloads the slot (synchronous).
    pub(super) fn check(
        &self,
        gpu: &Gpu,
        round: u32,
        j: usize,
        upc: u32,
    ) -> Result<(bool, String), String> {
        let mut s = vec![0u8; SLOT];
        gpu.rt
            .download_at(&self.ctr, round as usize * SLOT, &mut s)?;
        let w = |at: usize| u32::from_le_bytes(s[at..at + 4].try_into().unwrap());
        let ok = (0..j).all(|i| w(128 * i) >= upc) && w(512) == 0;
        Ok((
            ok,
            format!(
                "round {round}: claims [{}] (upc {upc}), status {}",
                (0..j)
                    .map(|i| w(128 * i).to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                w(512)
            ),
        ))
    }
}
