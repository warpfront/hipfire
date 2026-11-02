// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `lane_wo`: the exact-gfx1201 lane-batched A3B `wo` residual GEMV `gemv_mq4g256v2_residual_lanes_gfx1201`
//! (`y[b][row] += A[row] . x[b]`, wave32, K = 4096 = 16 groups of 136 B), judged on the R9700. The
//! incumbent is the shipped source exactly as the runtime compiles it: `hip_loop/incumbents/res_lanes.hip`
//! = `#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1` + `kernels/src/gfx12_weight_cache_policy.inc` +
//! `kernels/src/gemv_mq4g256v2_residual_lanes.gfx1201.hip` (sha256 036fdd2c...), built with
//! `-DIU4_A4_CANDIDATES=2` like the runtime.
//!
//! Production launch (`crates/rdna-compute/src/residual_lanes.rs` `launch_residual_lanes`): kernargs
//! `(A, x, y, M, B)` (3 pointers + 2 i32, 32 B), block 128 (4 wave32), grid `ceil(M / 8)` (two rows per
//! wave), no LDS; `x` is `[B][4096]` f32, `y` is `[B][M]` f32 and ACCUMULATES (`reset` restores it
//! from a pristine copy outside the timed interval).
//!
//! Shape grammar `lanes=1..8,m=2048` (defaults 8, 2048), optional `flush=0|1`. Preset `emu` (the
//! `--emu` small shape): m = 41 (ragged last workgroup, odd tail row), lanes 5.
//!
//! Candidate header (`// hip_loop: ...`): `block=` (threads, multiple of 32), `rows_per_wg=` (grid =
//! `ceil(M / rows_per_wg)`), `lds_dynamic=` (bytes).
//!
//! TIMING cycles a pool of distinct seeded weight sets whose total exceeds 192 MiB (dispatch `i` uses
//! set `i % sets`). IDENTITY runs the adversarial matrix of the product gate `tests/a3b_lane_wo.rs`
//! at the node's `m` (every weight kind x activation kind x residual kind x lanes 1..8) plus the
//! partial-tile row counts 1, 37, 96, 333, 2050 (a rotating subset), outputs pre-poisoned beyond
//! `lanes` (which must stay untouched). The product gate requires RAW bits: every word of every lane
//! of the candidate must equal the incumbent's, NaN payloads included (payload differences are also
//! tallied separately). Both arms are also compared with the CPU reference `lane_ref.rs`
//! (`cpu_ref_identical`, informational).
//!
//! Included with `#[path = "hip_loop/lane_wo.rs"] mod lane_wo;`.
#![allow(dead_code)]

use crate::lane_common::*;
use crate::lane_ref;
use crate::node::*;
use serde_json::{json, Value};

pub const K: usize = 4096;
const ROW_BYTES: usize = (K / 256) * GROUP_BYTES;
pub const GFX1201: &str = "gfx1201";
pub const MODULE: &str = "gemv_mq4g256v2_residual_lanes_gfx1201";
pub const SYMBOL: &str = "gemv_mq4g256v2_residual_lanes_gfx1201";
pub const EXTRA_FLAGS: &str = "-DIU4_A4_CANDIDATES=2";
pub const SOURCE: &str = include_str!("incumbents/res_lanes.hip");
const BLOCK: usize = 128;
const ROWS_PER_WG: usize = 8;
const EMU_M: usize = 41;
const PRODUCTION_M: usize = 2048;
const ARGS_BYTES: usize = 32;

struct State {
    sets: Vec<Buf>,
    x: Buf,
    /// `(MAX_LANES + 1) * m` words: lanes beyond `lanes` are the pre-poisoned guard.
    y: Buf,
    y0: Buf,
}

pub struct LaneWo {
    lanes: usize,
    m: usize,
    flush: bool,
    emu: bool,
    st: Option<State>,
}

impl LaneWo {
    pub fn new(shape: &ShapeSpec) -> Result<Self> {
        let mut s = shape.clone();
        let emu = match s.preset.as_deref() {
            None => false,
            Some("emu") => true,
            Some(p) => return Err(format!("lane_wo: unknown shape preset `{p}` (have emu)")),
        };
        let lanes = s.take_usize("lanes", Some(if emu { 5 } else { MAX_LANES }))?;
        let m = s.take_usize("m", Some(if emu { EMU_M } else { PRODUCTION_M }))?;
        let flush = s.take_usize("flush", Some(0))?;
        s.no_leftover()?;
        Self::build(lanes, m, flush != 0, emu)
    }

    fn build(lanes: usize, m: usize, flush: bool, emu: bool) -> Result<Self> {
        if !(1..=MAX_LANES).contains(&lanes) {
            return Err(format!("lane_wo: lanes={lanes} must be 1..={MAX_LANES}"));
        }
        if m == 0 || m > (i32::MAX as usize) / 64 {
            return Err(format!("lane_wo: m={m} out of range"));
        }
        Ok(Self {
            lanes,
            m,
            flush,
            emu,
            st: None,
        })
    }

    fn at(&self, lanes: usize, m: usize) -> Self {
        Self::build(lanes, m, false, false).expect("valid dims")
    }

    fn set_bytes(&self) -> usize {
        self.m * ROW_BYTES
    }

    fn st(&self) -> Result<&State> {
        self.st
            .as_ref()
            .ok_or_else(|| "lane_wo: setup has not run".to_owned())
    }

    /// Seeded host buffers: weights (Normal), activations `[lanes][4096]`, residual `[lanes][m]`.
    pub fn host_inputs(&self, seed: u64) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let w = synth_weights(self.m, K / 256, seed ^ 0xA3B0, WKind::Normal);
        (w, self.host_x(seed), self.host_y(seed))
    }

    fn host_x(&self, seed: u64) -> Vec<f32> {
        let mut r = Xs64::new(seed ^ 0x7E57);
        (0..self.lanes * K).map(|_| r.unit()).collect()
    }

    fn host_y(&self, seed: u64) -> Vec<f32> {
        let mut r = Xs64::new(seed ^ 0xBEEF);
        (0..self.lanes * self.m).map(|_| r.unit() * 4.0).collect()
    }

    /// The CPU reference of `y += W x` per lane, parallel over even row chunks (row parity picks the
    /// contraction form, so chunks start on even rows).
    fn cpu_reference(w: &[u8], x: &[f32], y0: &[f32], m: usize, lanes: usize) -> Vec<u32> {
        const CHUNK: usize = 128;
        let n = m.div_ceil(CHUNK);
        let parts = par_map(n, |c| {
            let (r0, nr) = (c * CHUNK, CHUNK.min(m - c * CHUNK));
            let mut y: Vec<f32> = (0..lanes)
                .flat_map(|b| y0[b * m + r0..b * m + r0 + nr].iter().copied())
                .collect();
            lane_ref::residual_lanes_ref(
                &w[r0 * ROW_BYTES..(r0 + nr) * ROW_BYTES],
                x,
                &mut y,
                nr,
                lanes,
            );
            y
        });
        let mut out = vec![0u32; lanes * m];
        for (c, part) in parts.into_iter().enumerate() {
            let (r0, nr) = (c * CHUNK, CHUNK.min(m - c * CHUNK));
            for b in 0..lanes {
                for j in 0..nr {
                    out[b * m + r0 + j] = part[b * nr + j].to_bits();
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Adversarial inputs (the product gate's kinds)

#[derive(Clone, Copy, Debug)]
enum Xk {
    Rand,
    RandBig,
    Zeros,
    /// +-1e30..: products and sums stay finite but are huge.
    Large,
    /// Reaches inf / NaN inside the dot product.
    Overflow,
    Denormal,
    /// Every element 1.0 (all partial sums tie).
    Ones,
    Alternating,
    /// The same vector in every lane.
    DupLanes,
    /// Mixed zeros, tiny and huge.
    Mixed,
}

const XKINDS: [Xk; 10] = [
    Xk::Rand,
    Xk::RandBig,
    Xk::Zeros,
    Xk::Large,
    Xk::Overflow,
    Xk::Denormal,
    Xk::Ones,
    Xk::Alternating,
    Xk::DupLanes,
    Xk::Mixed,
];

fn make_x(kind: Xk, r: &mut Xs64, lanes: usize) -> Vec<f32> {
    let one = |r: &mut Xs64| -> Vec<f32> {
        (0..K)
            .map(|i| match kind {
                Xk::Rand | Xk::DupLanes => r.unit(),
                Xk::RandBig => r.unit() * 64.0,
                Xk::Zeros => 0.0,
                Xk::Large => r.unit() * 1.0e30,
                Xk::Overflow => r.unit() * 3.0e38,
                Xk::Denormal => {
                    let v = f32::from_bits((r.next_u64() >> 32) as u32 & 0x007F_FFFF);
                    if r.next_u64() >> 40 & 1 == 1 {
                        -v
                    } else {
                        v
                    }
                }
                Xk::Ones => 1.0,
                Xk::Alternating => {
                    if i & 1 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                Xk::Mixed => match (r.next_u64() >> 33) % 5 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => r.unit() * 1.0e-30,
                    3 => r.unit() * 1.0e8,
                    _ => r.unit(),
                },
            })
            .collect()
    };
    if matches!(kind, Xk::DupLanes) {
        let v = one(r);
        return (0..lanes).flat_map(|_| v.iter().copied()).collect();
    }
    (0..lanes).flat_map(|_| one(r)).collect()
}

#[derive(Clone, Copy, Debug)]
enum Yk {
    Rand,
    Zeros,
    NegZeros,
    Large,
}

const YKINDS: [Yk; 4] = [Yk::Rand, Yk::Zeros, Yk::NegZeros, Yk::Large];

fn make_y(kind: Yk, r: &mut Xs64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| match kind {
            Yk::Rand => r.unit() * 4.0,
            Yk::Zeros => 0.0,
            Yk::NegZeros => -0.0,
            Yk::Large => r.unit() * 1.0e12,
        })
        .collect()
}

impl Node for LaneWo {
    fn name(&self) -> &'static str {
        "lane_wo"
    }
    fn arch(&self) -> &'static str {
        GFX1201
    }
    fn shape_label(&self) -> String {
        format!(
            "{}n{}-m{}",
            if self.emu { "emu-" } else { "" },
            self.lanes,
            self.m
        )
    }
    fn shape_json(&self) -> Value {
        let sets = pool_sets(self.set_bytes());
        json!({
            "lanes": self.lanes, "m": self.m, "k": K, "flush": self.flush,
            "weights": "MQ4G256V2 (16 groups x 136 B per row), seeded",
            "x": "f32 [lanes][4096], seeded random in [-1, 1)",
            "y": "f32 [lanes][m], accumulated (y += W x), restored from a pristine copy before every dispatch",
            "timing_pool": crate::cold::pool_json(sets, self.set_bytes()),
        })
    }
    fn incumbent(&self) -> Result<Incumbent> {
        Ok(Incumbent {
            module: MODULE.to_owned(),
            source: SOURCE.to_owned(),
            symbol: SYMBOL.to_owned(),
            extra_flags: EXTRA_FLAGS.to_owned(),
            selection: "crates/rdna-compute/src/residual_lanes.rs launch_residual_lanes \
                        (GEMV_MQ4G256V2_RESIDUAL_LANES_GFX1201_SRC = gfx12_weight_cache_policy.inc + \
                        gemv_mq4g256v2_residual_lanes.gfx1201.hip)"
                .to_owned(),
        })
    }
    fn geometry_with(&self, hdr: &mut Header) -> Result<Geometry> {
        let block = hdr.take_usize("block", BLOCK)?;
        let rows_per_wg = hdr.take_usize("rows_per_wg", ROWS_PER_WG)?;
        let lds = hdr.take_usize("lds_dynamic", 0)?;
        hdr.no_leftover()?;
        if block == 0 || block % 32 != 0 || block > 1024 {
            return Err(format!(
                "hip_loop header: block={block} must be a multiple of 32 in 32..=1024"
            ));
        }
        if rows_per_wg == 0 {
            return Err("hip_loop header: rows_per_wg must be positive".into());
        }
        if lds > 64 * 1024 {
            return Err(format!(
                "hip_loop header: lds_dynamic={lds} exceeds the 64 KiB workgroup LDS"
            ));
        }
        let grid = u32::try_from(self.m.div_ceil(rows_per_wg))
            .map_err(|_| "grid exceeds u32".to_owned())?;
        Ok(Geometry {
            grid: [grid, 1, 1],
            block: [block as u32, 1, 1],
            lds: lds as u32,
        })
    }
    fn work(&self) -> Work {
        let (m, l) = (self.m as f64, self.lanes as f64);
        Work {
            flops: 2.0 * m * K as f64 * l,
            bytes: m * ROW_BYTES as f64 + l * K as f64 * 4.0 + l * m * 8.0,
        }
    }
    fn record_extra(&self) -> Value {
        json!({"lanes": self.lanes, "m": self.m, "flush": self.flush})
    }
    fn flush_requested(&self) -> bool {
        self.flush
    }
    fn static_hazards(&self) -> bool {
        true
    }
    fn extent_full_default(&self) -> bool {
        true
    }
    fn extent_shapes(&self) -> Vec<String> {
        [1, 5, 8]
            .iter()
            .map(|n| format!("lanes={n},m={PRODUCTION_M}"))
            .collect()
    }
    fn out_owner<'a>(&'a self, hdr: &Header) -> Result<Box<dyn Fn(usize) -> [u32; 3] + 'a>> {
        // y is f32 [lanes][m]: row `e % m` belongs to workgroup `row / rows_per_wg`.
        let rows_per_wg = hdr.kv.get("rows_per_wg").map_or(Ok(ROWS_PER_WG), |v| {
            v.parse::<usize>()
                .map_err(|e| format!("hip_loop header rows_per_wg={v}: {e}"))
        })?;
        if rows_per_wg == 0 {
            return Err("hip_loop header: rows_per_wg must be positive".into());
        }
        let m = self.m;
        Ok(Box::new(move |e| [(e % m / rows_per_wg) as u32, 0, 0]))
    }
    fn emu_case(&self, seed: u64) -> Result<EmuCase> {
        let (w, x, y) = self.host_inputs(seed);
        Ok(EmuCase {
            bufs: vec![
                EmuBuf {
                    name: "w",
                    bytes: w,
                },
                EmuBuf {
                    name: "x",
                    bytes: f32_bytes(&x),
                },
                EmuBuf {
                    name: "y",
                    bytes: f32_bytes(&y),
                },
            ],
            args: vec![
                EmuArg::Buf(0),
                EmuArg::Buf(1),
                EmuArg::Buf(2),
                EmuArg::I32(self.m as i32),
                EmuArg::I32(self.lanes as i32),
            ],
            out: 2,
        })
    }

    fn setup(&mut self, gpu: &Gpu, seed: u64) -> Result<()> {
        let n_sets = pool_sets(self.set_bytes());
        let mut sets = Vec::with_capacity(n_sets);
        for s in 0..n_sets {
            let w = synth_weights(
                self.m,
                K / 256,
                seed ^ 0xA3B0 ^ ((s as u64) << 8),
                WKind::Normal,
            );
            sets.push(Buf::from_bytes(gpu, &w)?);
        }
        let x = Buf::from_bytes(gpu, &f32_bytes(&self.host_x(seed)))?;
        let mut y0 = word_bytes(
            &self
                .host_y(seed)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
        );
        y0.extend_from_slice(&word_bytes(&poison_words(
            (MAX_LANES + 1) * self.m - self.lanes * self.m,
        )));
        let y = Buf::from_bytes(gpu, &y0)?;
        let y0 = Buf::from_bytes(gpu, &y0)?;
        self.st = Some(State { sets, x, y, y0 });
        Ok(())
    }
    fn reset(&mut self, gpu: &Gpu) -> Result<()> {
        let s = self.st()?;
        s.y.copy_from(gpu, &s.y0)
    }
    fn kernarg(&self) -> Result<Vec<u8>> {
        let s = self.st()?;
        let mut a = Args::new();
        a.ptr(&s.sets[0])
            .ptr(&s.x)
            .ptr(&s.y)
            .i32(self.m as i32)
            .i32(self.lanes as i32);
        let k = a.finish();
        debug_assert_eq!(k.len(), ARGS_BYTES);
        Ok(k)
    }
    fn select_set(&self, karg: &mut [u8], i: usize) {
        if let Ok(s) = self.st() {
            let p = s.sets[i % s.sets.len()].ptr() as u64;
            karg[..8].copy_from_slice(&p.to_le_bytes());
        }
    }
    fn read_output(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        self.st()?.y.download(gpu)
    }
    fn teardown(&mut self, gpu: &Gpu) {
        if let Some(s) = self.st.take() {
            for b in s.sets {
                b.free(gpu);
            }
            s.x.free(gpu);
            s.y.free(gpu);
            s.y0.free(gpu);
        }
    }

    fn adversarial(
        &mut self,
        gpu: &Gpu,
        base: &ArmImage,
        cand: &ArmImage,
        seed: u64,
    ) -> Result<Option<Value>> {
        let arms = [
            LoadedArm::load(gpu, base.image)?,
            LoadedArm::load(gpu, cand.image)?,
        ];
        let funcs = [
            arms[0].function(gpu, SYMBOL)?,
            arms[1].function(gpu, SYMBOL)?,
        ];
        let lanes_all: Vec<usize> = (1..=MAX_LANES).collect();
        let mut mx = Matrix {
            lanes: lanes_all.clone(),
            ..Default::default()
        };
        let mut ms = vec![self.m];
        for m in [37usize, 1, 2050, 333, 96] {
            if !ms.contains(&m) {
                ms.push(m);
            }
        }
        let wkinds = [
            WKind::Normal,
            WKind::Wide,
            WKind::AllZero,
            WKind::AllFifteen,
            WKind::Wild,
        ];
        let max_m = *ms.iter().max().unwrap();
        let xbuf = Buf::alloc(gpu, MAX_LANES * K * 4)?;
        let ybuf = Buf::alloc(gpu, (MAX_LANES + 1) * max_m * 4)?;
        let result = (|| -> Result<()> {
            for (si, &m) in ms.iter().enumerate() {
                let g_dims = self.at(self.lanes, m);
                let geo = [
                    g_dims.geometry(),
                    g_dims.geometry_with(&mut cand.header.clone())?,
                ];
                for (wi, wk) in wkinds.into_iter().enumerate() {
                    if si == 0 {
                        mx.wkinds.push(wk.name());
                    }
                    let w = synth_weights(
                        m,
                        K / 256,
                        seed ^ 0xA3B0 ^ (((si * 8 + wi) as u64) << 16),
                        wk,
                    );
                    let wbuf = Buf::from_bytes(gpu, &w)?;
                    for (xi, &xk) in XKINDS.iter().enumerate() {
                        for (yi, &yk) in YKINDS.iter().enumerate() {
                            if si != 0 && (xi + yi + wi) % 3 != 0 {
                                continue;
                            }
                            for &lanes in &lanes_all {
                                if si != 0 && lanes % 2 == 0 && lanes != 8 {
                                    continue;
                                }
                                let mut r = Xs64::new(
                                    seed ^ 0x1234_5678
                                        ^ ((si as u64) << 40)
                                        ^ ((wi as u64) << 32)
                                        ^ ((xi as u64) << 24)
                                        ^ ((yi as u64) << 16)
                                        ^ lanes as u64,
                                );
                                let x = make_x(xk, &mut r, lanes);
                                let y0 = make_y(yk, &mut r, lanes * m);
                                let tag = format!("m{m} w{} x{xk:?} y{yk:?} n{lanes}", wk.name());
                                xbuf.upload(gpu, &f32_bytes(&x))?;
                                let mut y_init: Vec<u32> = y0.iter().map(|v| v.to_bits()).collect();
                                y_init.extend(poison_words((MAX_LANES + 1) * m - lanes * m));
                                let cpu = Self::cpu_reference(&w, &x, &y0, m, lanes);
                                let mut outs: Vec<Vec<u32>> = Vec::new();
                                for arm in 0..2 {
                                    ybuf.upload(gpu, &word_bytes(&y_init))?;
                                    let mut a = Args::new();
                                    a.ptr(&wbuf)
                                        .ptr(&xbuf)
                                        .ptr(&ybuf)
                                        .i32(m as i32)
                                        .i32(lanes as i32);
                                    let mut karg = a.finish();
                                    launch_sync(gpu, &funcs[arm], &geo[arm], &mut karg).map_err(
                                        |e| {
                                            format!(
                                                "{} arm, {tag}: {e}",
                                                ["baseline", "candidate"][arm]
                                            )
                                        },
                                    )?;
                                    let mut all = words(&ybuf.download(gpu)?);
                                    all.truncate((MAX_LANES + 1) * m);
                                    outs.push(all);
                                }
                                mx.cases += 1;
                                let written = lanes * m;
                                for arm in 0..2 {
                                    if outs[arm][written..].iter().any(|&v| v != POISON) {
                                        mx.guard_violations[arm] += 1;
                                        mx.fail(format!(
                                            "[{tag}] {} wrote past lane {lanes} (guard words)",
                                            ["baseline", "candidate"][arm]
                                        ));
                                    }
                                }
                                let (b, c) = (&outs[0][..written], &outs[1][..written]);
                                // The product gate (`tests/a3b_lane_wo.rs`): RAW bits, NaN payloads included, for
                                // every weight kind whose headers are finite. `Wild` headers carry random f16 NaN
                                // payloads, whose propagation depends on the compiler's free choice of commutative
                                // operand order (fma/mul src0 vs src1): NaN meets NaN there (the lane_proj policy).
                                let strict = wk != WKind::Wild;
                                let d = compare(c, b, strict);
                                if d.mismatches > 0 {
                                    let f = d.first.unwrap_or(0);
                                    mx.fail(format!(
                                        "[{tag}] candidate != incumbent: word {f} (lane {} row {}) cand {:#010x} base {:#010x} ({} words, {} NaN-payload)",
                                        f / m, f % m, c[f], b[f], d.mismatches, d.nan_payload_diffs
                                    ));
                                }
                                mx.cand_vs_base.add(&d);
                                let (db, dc) = (compare(b, &cpu, strict), compare(c, &cpu, strict));
                                mx.base_vs_cpu.add(&db);
                                mx.cand_vs_cpu.add(&dc);
                                if db.mismatches > 0 {
                                    mx.fail(format!(
                                        "[{tag}] INCUMBENT != CPU reference at word {} ({} words) — a reference bug or a hardware numerics fact",
                                        db.first.unwrap_or(0), db.mismatches
                                    ));
                                }
                            }
                        }
                    }
                    wbuf.free(gpu);
                }
            }
            Ok(())
        })();
        xbuf.free(gpu);
        ybuf.free(gpu);
        result?;
        let mut j = mx.json(true);
        j["policy"] = json!(
            "raw bits (NaN payloads included) for every finite-header weight kind; NaN meets NaN for Wild headers \
             (random f16 NaN payloads: payload propagation follows the compiler's commutative operand order)"
        );
        j["m"] = json!(ms);
        if j["cpu_ref_identical"]["baseline"] != json!(true) {
            eprintln!(
                "hip_loop: lane_wo: the INCUMBENT differs from the CPU reference ({} mismatching word(s), {} NaN-payload diff(s)): \
                 reference bug or hardware numerics fact — see identity_matrix.cpu_ref",
                mx.base_vs_cpu.mismatches, mx.base_vs_cpu.nan_payload_diffs
            );
        }
        Ok(Some(j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(shape: &str) -> LaneWo {
        LaneWo::new(&ShapeSpec::parse(Some(shape)).unwrap()).unwrap()
    }

    #[test]
    fn production_geometry_is_the_launcher_grid() {
        let g = node("lanes=8,m=2048").geometry();
        assert_eq!((g.grid, g.block, g.lds), ([256, 1, 1], [128, 1, 1], 0));
        let d = node("");
        assert_eq!((d.lanes, d.m), (8, 2048));
        assert_eq!(node("m=2050").geometry().grid, [257, 1, 1]);
    }

    #[test]
    fn kernarg_layout_is_the_production_blob() {
        // (A, x, y: 3 pointers) + (M, B: 2 i32) = 32 B.
        let mut a = Args::new();
        a.raw_ptr(1).raw_ptr(2).raw_ptr(3).i32(2048).i32(8);
        assert_eq!(a.finish().len(), ARGS_BYTES);
    }

    #[test]
    fn header_geometry() {
        let n = node("lanes=3,m=2048");
        let mut h =
            Header::parse("// hip_loop: block=256 rows_per_wg=16 lds_dynamic=1024").unwrap();
        let g = n.geometry_with(&mut h).unwrap();
        assert_eq!((g.grid, g.block, g.lds), ([128, 1, 1], [256, 1, 1], 1024));
        for bad in [
            "block=48",
            "block=2048",
            "rows_per_wg=0",
            "lds_dynamic=70000",
            "tile_m=8",
        ] {
            let mut h = Header::parse(&format!("// hip_loop: {bad}")).unwrap();
            assert!(n.geometry_with(&mut h).is_err(), "{bad}");
        }
    }

    #[test]
    fn shapes_parse_and_refuse() {
        for s in ["lanes=0", "lanes=9", "m=0", "q=1", "nope"] {
            assert!(
                LaneWo::new(&ShapeSpec::parse(Some(s)).unwrap()).is_err(),
                "{s}"
            );
        }
        let n = node("emu");
        assert_eq!((n.lanes, n.m), (5, EMU_M));
        assert!(node("lanes=2,flush=1").flush_requested());
        assert_eq!(node("emu").extent_shapes().len(), 3);
    }

    #[test]
    fn emu_case_and_owner_follow_the_kernel_layout() {
        let n = node("emu");
        let c = n.emu_case(3).unwrap();
        assert_eq!(c.bufs[0].bytes.len(), EMU_M * ROW_BYTES);
        assert_eq!(c.bufs[2].bytes.len(), 5 * EMU_M * 4);
        let own = n.out_owner(&Header::default()).unwrap();
        // y[b][row]: workgroup row / 8
        assert_eq!(own(0), [0, 0, 0]);
        assert_eq!(own(EMU_M + 9), [1, 0, 0]);
        assert_eq!(own(EMU_M - 1), [5, 0, 0]);
        let h = Header::parse("// hip_loop: rows_per_wg=16").unwrap();
        assert_eq!(n.out_owner(&h).unwrap()(EMU_M + 17), [1, 0, 0]);
    }

    #[test]
    fn chunked_cpu_reference_equals_the_direct_reference() {
        // m = 333 spans three 128-row chunks with an odd tail: parity is by absolute row.
        for (m, lanes) in [(41usize, 5usize), (333, 3)] {
            let n = LaneWo::build(lanes, m, false, false).unwrap();
            let (w, x, y0) = n.host_inputs(5);
            let mut y = y0.clone();
            lane_ref::residual_lanes_ref(&w, &x, &mut y, m, lanes);
            let want: Vec<u32> = y.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                LaneWo::cpu_reference(&w, &x, &y0, m, lanes),
                want,
                "m{m} n{lanes}"
            );
        }
    }

    /// Wall time of the CPU reference for one production-size matrix case (m 2048, 8 lanes); `--ignored --nocapture`.
    #[test]
    #[ignore = "timing probe"]
    fn cpu_reference_cost_m2048_8_lanes() {
        let n = node("lanes=8,m=2048");
        let (w, x, y0) = n.host_inputs(5);
        let t = std::time::Instant::now();
        let r = LaneWo::cpu_reference(&w, &x, &y0, 2048, 8);
        eprintln!(
            "lane_wo cpu reference m2048 n8: {:?} ({} words)",
            t.elapsed(),
            r.len()
        );
    }

    #[test]
    fn timing_pool_exceeds_the_last_level_cache_with_an_odd_set_count() {
        let n = node("lanes=8,m=2048");
        let sets = pool_sets(n.set_bytes());
        assert!(sets * n.set_bytes() > POOL_BYTES);
        assert_eq!(sets % 2, 1);
    }
}
