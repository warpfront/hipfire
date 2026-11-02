// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `lane_proj`: the exact-gfx1201 lane-batched multi-weight MQ4G256V2 projection GEMV of the A3B
//! batch route, `gemv_mq4g256v2_lane_proj_k2048_gfx1201_n{lanes}` (wave64, K = 2048), judged on
//! the R9700. The incumbent is the shipped source exactly as the runtime compiles it:
//! `hip_loop/incumbents/lane_proj.hip` = `// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64` + the
//! kernel source (sha256 32bec474...), built with `-DIU4_A4_CANDIDATES=2` like the runtime.
//!
//! Production launch (`crates/rdna-compute/src/lane_proj.rs` `Gpu::gemv_mq4g256v2_lane_proj_variant`,
//! `LaneProjVariant::PRODUCT`): up to four weight segments `A_s` (rows `m_s`, K = 2048, 8 groups of
//! 136 B per row) against `lanes` activation columns `x[b][2048]`, outputs `y_s[b * m_s + row]`;
//! kernargs `(A0..A3, y0..y3, x, m0..m3)` (13 args, 88 B; unused segments repeat `A0`/`y0` with
//! `m = 0`); block 512 (8 wave64), grid `clamp(ceil(total_rows / (8*2*3)), 1, WGPs * 3)`, dynamic
//! LDS `lanes * 2048 * 4`. `WGPs` is the device's MULTIPROCESSOR_COUNT attribute (32 on the R9700,
//! also what offline geometry assumes; the device run refuses another count).
//!
//! Shape grammar `group=dn|fa|moe,lanes=1..8` (default `dn`, 8), optional `flush=0|1` and
//! `rows=a:b:c:d` (segment rows override, tests/dev). Groups: `dn` 8192,4096,32,32 (30 launches per
//! step, DeltaNet QKVZA); `fa` 8192,512,512 (m3 = 0); `moe` 256,1,512,512. Preset `emu` (the
//! `--emu` small shape): ragged segments (dn 70,33,5,3; fa 70,19,19; moe 20,1,19,23), lanes 5.
//!
//! Candidate header (`// hip_loop: ...`): `block=` (threads, multiple of 64), `rows_per_wg=` (grid =
//! `clamp(ceil(total/rows_per_wg), 1, WGPs*wgs_per_wgp)`), `wgs_per_wgp=`, `lds_per_lane=` (dynamic
//! LDS = `lanes * lds_per_lane` bytes), `grid_fixed=` (absolute grid, overrides the formula).
//!
//! TIMING runs on a pool of distinct seeded weight sets whose total exceeds 192 MiB (cold last-level
//! cache as in production; dispatch `i` uses set `i % sets`). IDENTITY runs the adversarial matrix of
//! the product gate `tests/lane_proj.rs` for the node's group at the node's lane count and lanes
//! 1,2,3,5,8 (weights Small/Large/Wild x activations random/zeros/+-large/denormal/ties/outliers/
//! nonfinite/mixed per lane), outputs pre-poisoned (0x7FC0DEAD) including rows/lanes beyond `lanes`
//! (which must stay untouched), candidate vs incumbent under the lab NaN policy, both arms also vs
//! the CPU reference `lane_ref.rs` (`cpu_ref_identical`, informational).
//!
//! Included with `#[path = "hip_loop/lane_proj.rs"] mod lane_proj;`.
#![allow(dead_code)]

use crate::lane_common::*;
use crate::lane_ref;
use crate::node::*;
use serde_json::{json, Value};

pub const K: usize = 2048;
const ROW_BYTES: usize = (K / 256) * GROUP_BYTES;
pub const GFX1201: &str = "gfx1201";
pub const MODULE: &str = "gemv_mq4g256v2_lane_proj_k2048_gfx1201";
/// The runtime compiler's extra flags for gfx1201 (`hipfire-kernel-pack --extra-flags`).
pub const EXTRA_FLAGS: &str = "-DIU4_A4_CANDIDATES=2";
pub const SOURCE: &str = include_str!("incumbents/lane_proj.hip");
const BLOCK: usize = 512;
const ROWS_PER_WG: usize = 8 * 2 * 3;
const WGS_PER_WGP: usize = 3;
const LDS_PER_LANE: usize = K * 4;
const ARGS_BYTES: usize = 88;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Group {
    Dn,
    Fa,
    Moe,
}

impl Group {
    pub const ALL: [Group; 3] = [Group::Dn, Group::Fa, Group::Moe];

    pub fn name(self) -> &'static str {
        match self {
            Group::Dn => "dn",
            Group::Fa => "fa",
            Group::Moe => "moe",
        }
    }
    pub fn parse(s: &str) -> Result<Self> {
        Group::ALL
            .into_iter()
            .find(|g| g.name() == s)
            .ok_or_else(|| format!("lane_proj: unknown group `{s}` (have dn, fa, moe)"))
    }
    /// The production segment rows.
    pub fn rows(self) -> Vec<usize> {
        match self {
            Group::Dn => vec![8192, 4096, 32, 32],
            Group::Fa => vec![8192, 512, 512],
            Group::Moe => vec![256, 1, 512, 512],
        }
    }
    /// The small ragged `--emu` segments of the group.
    fn emu_rows(self) -> Vec<usize> {
        match self {
            Group::Dn => vec![70, 33, 5, 3],
            Group::Fa => vec![70, 19, 19],
            Group::Moe => vec![20, 1, 19, 23],
        }
    }
}

struct State {
    /// `sets[s][seg]`: the pool of weight sets.
    sets: Vec<Vec<Buf>>,
    /// Kernarg pointer values of every set: `[A0, A1, A2, A3]` (unused segments repeat A0).
    set_ptrs: Vec<[u64; 4]>,
    x: Buf,
    /// Per segment `MAX_LANES * m` words (rows beyond `lanes` are the guard).
    y: Vec<Buf>,
    y0: Vec<Buf>,
}

pub struct LaneProj {
    group: Group,
    lanes: usize,
    rows: Vec<usize>,
    flush: bool,
    emu: bool,
    st: Option<State>,
}

impl LaneProj {
    pub fn new(shape: &ShapeSpec) -> Result<Self> {
        let mut s = shape.clone();
        let emu = match s.preset.as_deref() {
            None => false,
            Some("emu") => true,
            Some(g) if Group::parse(g).is_ok() => {
                if s.kv.insert("group".to_owned(), g.to_owned()).is_some() {
                    return Err(format!(
                        "lane_proj: shape names the group twice (`{g}` and group=)"
                    ));
                }
                false
            }
            Some(p) => {
                return Err(format!(
                    "lane_proj: unknown shape preset `{p}` (have emu, dn, fa, moe)"
                ))
            }
        };
        let group = match s.kv.remove("group") {
            Some(g) => Group::parse(&g)?,
            None => Group::Dn,
        };
        let lanes = s.take_usize("lanes", Some(if emu { 5 } else { MAX_LANES }))?;
        let flush = s.take_usize("flush", Some(0))?;
        let rows = match s.kv.remove("rows") {
            Some(r) => r
                .split(':')
                .map(|t| {
                    t.parse::<usize>()
                        .map_err(|e| format!("--shape rows={r}: {e}"))
                })
                .collect::<Result<Vec<_>>>()?,
            None if emu => group.emu_rows(),
            None => group.rows(),
        };
        s.no_leftover()?;
        Self::build(group, lanes, rows, flush != 0, emu)
    }

    fn build(group: Group, lanes: usize, rows: Vec<usize>, flush: bool, emu: bool) -> Result<Self> {
        if !(1..=MAX_LANES).contains(&lanes) {
            return Err(format!("lane_proj: lanes={lanes} must be 1..={MAX_LANES}"));
        }
        if rows.is_empty() || rows.len() > 4 || rows.contains(&0) {
            return Err(format!(
                "lane_proj: segment rows {rows:?}: need 1..=4 positive segments"
            ));
        }
        Ok(Self {
            group,
            lanes,
            rows,
            flush,
            emu,
            st: None,
        })
    }

    /// The same node at another lane count (the identity matrix).
    fn at_lanes(&self, lanes: usize) -> Self {
        Self::build(self.group, lanes, self.rows.clone(), false, false).expect("lanes 1..=8")
    }

    pub fn symbol(&self) -> String {
        format!("{MODULE}_n{}", self.lanes)
    }

    fn total_rows(&self) -> usize {
        self.rows.iter().sum()
    }

    fn set_bytes(&self) -> usize {
        self.total_rows() * ROW_BYTES
    }

    /// Four-segment `m` array of the launch (unused segments 0).
    fn m4(&self) -> [i32; 4] {
        let mut m = [0i32; 4];
        for (i, r) in self.rows.iter().enumerate() {
            m[i] = *r as i32;
        }
        m
    }

    fn st(&self) -> Result<&State> {
        self.st
            .as_ref()
            .ok_or_else(|| "lane_proj: setup has not run".to_owned())
    }

    /// Seeded host buffers: the weights of every segment (Small), the activations `[lanes][2048]`.
    pub fn host_inputs(&self, seed: u64) -> (Vec<Vec<u8>>, Vec<f32>) {
        let w = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, &m)| synth_weights(m, K / 256, seed ^ ((i as u64 + 1) << 32), WKind::Small))
            .collect();
        (w, self.host_x(seed))
    }

    fn host_x(&self, seed: u64) -> Vec<f32> {
        let mut r = Xs64::new(seed ^ 0x5851_F42D_4C95_7F2D);
        (0..self.lanes * K).map(|_| r.unit() * 2.0).collect()
    }

    /// `(word offset, words)` of every segment in the emulator's packed output arena, pad after each.
    fn arena(&self) -> (Vec<usize>, usize) {
        let mut offs = Vec::new();
        let mut at = 0usize;
        for &m in &self.rows {
            offs.push(at);
            at += self.lanes * m + PAD_WORDS;
        }
        (offs, at)
    }

    /// The row partition of the incumbent (and of any candidate that keeps it): every half-wave
    /// owns one contiguous run of `ceil(total / n_hw)` rows; its workgroup is `hw / (waves * 2)`.
    fn row_owner(&self, hdr: &Header) -> Result<impl Fn(usize) -> u32> {
        let geo = self.geometry_with(&mut hdr.clone())?;
        let waves = (geo.block[0] / 64) as usize;
        let n_hw = geo.grid[0] as usize * waves * 2;
        let total = self.total_rows();
        let rows_per = total.div_ceil(n_hw);
        Ok(move |gid: usize| (gid / rows_per / (waves * 2)) as u32)
    }

    /// The matrix's CPU reference: per segment `[lanes][rows_s]` words.
    fn cpu_reference(w: &[Vec<u8>], rows: &[usize], x: &[f32], lanes: usize) -> Vec<Vec<u32>> {
        const CHUNK: usize = 96;
        let mut tasks = Vec::new();
        for (s, &m) in rows.iter().enumerate() {
            let mut r0 = 0;
            while r0 < m {
                tasks.push((s, r0, CHUNK.min(m - r0)));
                r0 += CHUNK;
            }
        }
        let parts = par_map(tasks.len(), |t| {
            let (s, r0, nr) = tasks[t];
            lane_ref::lane_proj_ref(
                &[&w[s][r0 * ROW_BYTES..(r0 + nr) * ROW_BYTES]],
                &[nr],
                x,
                lanes,
            )
            .remove(0)
        });
        let mut out: Vec<Vec<u32>> = rows.iter().map(|&m| vec![0u32; lanes * m]).collect();
        for ((s, r0, nr), part) in tasks.iter().zip(parts) {
            let m = rows[*s];
            for b in 0..lanes {
                out[*s][b * m + r0..b * m + r0 + nr].copy_from_slice(&part[b * nr..(b + 1) * nr]);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Adversarial activations (the product gate's kinds)

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum XKind {
    Random,
    Zeros,
    Large,
    Denormal,
    Ties,
    Outliers,
    /// +-inf and nan elements (invalid-operation paths).
    NonFinite,
}

const XKINDS: [XKind; 7] = [
    XKind::Random,
    XKind::Zeros,
    XKind::Large,
    XKind::Denormal,
    XKind::Ties,
    XKind::Outliers,
    XKind::NonFinite,
];

fn unit01(r: &mut Xs64) -> f32 {
    (r.unit() + 1.0) * 0.5
}

fn make_x(kind: XKind, r: &mut Xs64) -> Vec<f32> {
    (0..K)
        .map(|i| match kind {
            XKind::Random => r.unit() * 2.0,
            XKind::Zeros => 0.0,
            XKind::Large => {
                if r.next_u64() >> 40 & 1 == 1 {
                    1.0e30
                } else {
                    -1.0e30 * unit01(r)
                }
            }
            XKind::Denormal => match (r.next_u64() >> 33) % 3 {
                0 => f32::from_bits((r.next_u64() >> 32) as u32 & 0x007F_FFFF),
                1 => r.unit() * 1.0e-38,
                _ => -f32::from_bits((r.next_u64() >> 32) as u32 & 0x007F_FFFF),
            },
            XKind::Ties => [1.0f32, -1.0, 0.5, -0.5][i % 4],
            XKind::Outliers => {
                if (r.next_u64() >> 33) % 64 == 0 {
                    r.unit() * 1.0e4
                } else {
                    r.unit()
                }
            }
            XKind::NonFinite => match (r.next_u64() >> 33) % 97 {
                0 => f32::INFINITY,
                1 => f32::NEG_INFINITY,
                2 => f32::NAN,
                3 => -0.0,
                _ => r.unit(),
            },
        })
        .collect()
}

impl Node for LaneProj {
    fn name(&self) -> &'static str {
        "lane_proj"
    }
    fn arch(&self) -> &'static str {
        GFX1201
    }
    fn shape_label(&self) -> String {
        format!(
            "{}{}-n{}",
            if self.emu { "emu-" } else { "" },
            self.group.name(),
            self.lanes
        )
    }
    fn shape_json(&self) -> Value {
        let sets = pool_sets(self.set_bytes());
        json!({
            "group": self.group.name(), "lanes": self.lanes, "rows": self.rows,
            "segments": self.rows.len(), "k": K, "flush": self.flush,
            "weights": "MQ4G256V2 (8 groups x 136 B per row), Small scales, seeded",
            "x": "f32 [lanes][2048], seeded random in [-2, 2)",
            "y": "f32 [lanes][m_s] per segment (overwritten)",
            "timing_pool": crate::cold::pool_json(sets, self.set_bytes()),
        })
    }
    fn incumbent(&self) -> Result<Incumbent> {
        Ok(Incumbent {
            module: MODULE.to_owned(),
            source: SOURCE.to_owned(),
            symbol: self.symbol(),
            extra_flags: EXTRA_FLAGS.to_owned(),
            selection: "crates/rdna-compute/src/lane_proj.rs Gpu::gemv_mq4g256v2_lane_proj_variant \
                        (LaneProjVariant::PRODUCT); kernels/src/gemv_mq4g256v2_lane_proj_k2048.gfx1201.hip \
                        behind `// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64`"
                .to_owned(),
        })
    }
    fn geometry_with(&self, hdr: &mut Header) -> Result<Geometry> {
        let block = hdr.take_usize("block", BLOCK)?;
        let rows_per_wg = hdr.take_usize("rows_per_wg", ROWS_PER_WG)?;
        let wgs_per_wgp = hdr.take_usize("wgs_per_wgp", WGS_PER_WGP)?;
        let lds_per_lane = hdr.take_usize("lds_per_lane", LDS_PER_LANE)?;
        let fixed = match hdr.kv.remove("grid_fixed") {
            Some(v) => Some(
                v.parse::<usize>()
                    .map_err(|e| format!("hip_loop header grid_fixed={v}: {e}"))?,
            ),
            None => None,
        };
        hdr.no_leftover()?;
        if block == 0 || block % 64 != 0 || block > 1024 {
            return Err(format!("hip_loop header: block={block} must be a multiple of 64 in 64..=1024 (wave64 kernel)"));
        }
        if rows_per_wg == 0 || wgs_per_wgp == 0 {
            return Err("hip_loop header: rows_per_wg and wgs_per_wgp must be positive".into());
        }
        let lds = self.lanes * lds_per_lane;
        if lds > 64 * 1024 {
            return Err(format!(
                "hip_loop header: dynamic LDS lanes*lds_per_lane = {lds} exceeds the 64 KiB workgroup LDS"
            ));
        }
        let grid = match fixed {
            Some(0) => return Err("hip_loop header: grid_fixed must be positive".into()),
            Some(g) => g,
            None => self
                .total_rows()
                .div_ceil(rows_per_wg)
                .clamp(1, WGP_COUNT * wgs_per_wgp),
        };
        let grid = u32::try_from(grid).map_err(|_| format!("grid {grid} exceeds u32"))?;
        Ok(Geometry {
            grid: [grid, 1, 1],
            block: [block as u32, 1, 1],
            lds: lds as u32,
        })
    }
    fn work(&self) -> Work {
        let (t, l) = (self.total_rows() as f64, self.lanes as f64);
        Work {
            flops: 2.0 * t * K as f64 * l,
            bytes: t * ROW_BYTES as f64 + l * K as f64 * 4.0 + t * l * 4.0,
        }
    }
    fn record_extra(&self) -> Value {
        json!({"lanes": self.lanes, "group": self.group.name(), "flush": self.flush,
            "wgp_count_assumed": WGP_COUNT})
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
    fn identity_shapes(&self) -> Vec<String> {
        Group::ALL
            .iter()
            .map(|g| format!("group={},lanes={}", g.name(), self.lanes))
            .collect()
    }
    fn extent_shapes(&self) -> Vec<String> {
        let mut v = Vec::new();
        for g in Group::ALL {
            for n in [1, 3, 8] {
                v.push(format!("group={},lanes={n}", g.name()));
            }
        }
        v
    }
    fn out_owner<'a>(&'a self, hdr: &Header) -> Result<Box<dyn Fn(usize) -> [u32; 3] + 'a>> {
        // Elements of the packed emulator arena: segment s, lane b, row r at `off_s + b*m_s + r`,
        // `PAD_WORDS` pad after each segment.
        let own = self.row_owner(hdr)?;
        let (offs, _) = self.arena();
        let lanes = self.lanes;
        let rows = self.rows.clone();
        let starts: Vec<usize> = rows
            .iter()
            .scan(0usize, |a, &m| {
                let s = *a;
                *a += m;
                Some(s)
            })
            .collect();
        Ok(Box::new(move |e| {
            let s = offs.iter().rposition(|&o| o <= e).unwrap_or(0);
            let w = e - offs[s];
            let m = rows[s];
            if w >= lanes * m {
                return PAD_OWNER;
            }
            [own(starts[s] + w % m), 0, 0]
        }))
    }
    fn emu_case(&self, seed: u64) -> Result<EmuCase> {
        let (w, x) = self.host_inputs(seed);
        let (offs, total) = self.arena();
        let mut bufs: Vec<EmuBuf> = Vec::new();
        const WN: [&str; 4] = ["w0", "w1", "w2", "w3"];
        for (i, wb) in w.into_iter().enumerate() {
            bufs.push(EmuBuf {
                name: WN[i],
                bytes: wb,
            });
        }
        let xi = bufs.len();
        bufs.push(EmuBuf {
            name: "x",
            bytes: f32_bytes(&x),
        });
        let yi = bufs.len();
        bufs.push(EmuBuf {
            name: "y",
            bytes: word_bytes(&poison_words(total)),
        });
        let seg = |i: usize| if i < self.rows.len() { i } else { 0 };
        let mut args = Vec::new();
        for i in 0..4 {
            args.push(EmuArg::Buf(seg(i)));
        }
        for i in 0..4 {
            args.push(EmuArg::BufOff(yi, (offs[seg(i)] * 4) as u64));
        }
        args.push(EmuArg::Buf(xi));
        for m in self.m4() {
            args.push(EmuArg::I32(m));
        }
        Ok(EmuCase {
            bufs,
            args,
            out: yi,
        })
    }

    fn setup(&mut self, gpu: &Gpu, seed: u64) -> Result<()> {
        let wgp = gpu
            .hip
            .get_device_attribute(
                rdna_compute::profiler::HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                gpu.device_id,
            )
            .map_err(err)?;
        if wgp as usize != WGP_COUNT {
            return Err(format!(
                "the device reports {wgp} WGPs (MULTIPROCESSOR_COUNT), the production grid cap assumes {WGP_COUNT}: \
                 the offline geometry would not be the production launch"
            ));
        }
        let n_sets = pool_sets(self.set_bytes());
        let mut sets = Vec::with_capacity(n_sets);
        let mut set_ptrs = Vec::with_capacity(n_sets);
        for s in 0..n_sets {
            let mut bufs = Vec::new();
            for (i, &m) in self.rows.iter().enumerate() {
                let w = synth_weights(
                    m,
                    K / 256,
                    seed ^ ((i as u64 + 1) << 32) ^ ((s as u64) << 8),
                    WKind::Small,
                );
                bufs.push(Buf::from_bytes(gpu, &w)?);
            }
            let p = |i: usize| bufs[if i < self.rows.len() { i } else { 0 }].ptr() as u64;
            set_ptrs.push([p(0), p(1), p(2), p(3)]);
            sets.push(bufs);
        }
        let x = Buf::from_bytes(gpu, &f32_bytes(&self.host_x(seed)))?;
        let mut y = Vec::new();
        let mut y0 = Vec::new();
        for &m in &self.rows {
            let poison = word_bytes(&poison_words(MAX_LANES * m));
            y.push(Buf::from_bytes(gpu, &poison)?);
            y0.push(Buf::from_bytes(gpu, &poison)?);
        }
        self.st = Some(State {
            sets,
            set_ptrs,
            x,
            y,
            y0,
        });
        Ok(())
    }
    fn reset(&mut self, gpu: &Gpu) -> Result<()> {
        let s = self.st()?;
        for (y, y0) in s.y.iter().zip(&s.y0) {
            y.copy_from(gpu, y0)?;
        }
        Ok(())
    }
    fn kernarg(&self) -> Result<Vec<u8>> {
        let s = self.st()?;
        let mut a = Args::new();
        for p in s.set_ptrs[0] {
            a.raw_ptr(p);
        }
        for i in 0..4 {
            a.ptr(&s.y[if i < self.rows.len() { i } else { 0 }]);
        }
        a.ptr(&s.x);
        for m in self.m4() {
            a.i32(m);
        }
        let k = a.finish();
        debug_assert_eq!(k.len(), ARGS_BYTES);
        Ok(k)
    }
    fn select_set(&self, karg: &mut [u8], i: usize) {
        if let Ok(s) = self.st() {
            for (j, p) in s.set_ptrs[i % s.set_ptrs.len()].iter().enumerate() {
                karg[8 * j..8 * j + 8].copy_from_slice(&p.to_le_bytes());
            }
        }
    }
    fn read_output(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        let s = self.st()?;
        let mut out = Vec::new();
        for y in &s.y {
            out.extend_from_slice(&y.download(gpu)?);
        }
        Ok(out)
    }
    fn teardown(&mut self, gpu: &Gpu) {
        if let Some(s) = self.st.take() {
            for set in s.sets {
                for b in set {
                    b.free(gpu);
                }
            }
            s.x.free(gpu);
            for b in s.y.into_iter().chain(s.y0) {
                b.free(gpu);
            }
        }
    }

    fn adversarial(
        &mut self,
        gpu: &Gpu,
        base: &ArmImage,
        cand: &ArmImage,
        seed: u64,
    ) -> Result<Option<Value>> {
        let mut lanes = vec![self.lanes, 1, 2, 3, 5, 8];
        lanes.sort_unstable();
        lanes.dedup();
        let arms = [
            LoadedArm::load(gpu, base.image)?,
            LoadedArm::load(gpu, cand.image)?,
        ];
        let mut mx = Matrix {
            lanes: lanes.clone(),
            ..Default::default()
        };
        let rows = self.rows.clone();
        let xbuf = Buf::alloc(gpu, MAX_LANES * K * 4)?;
        let ybufs: Vec<Buf> = rows
            .iter()
            .map(|&m| Buf::alloc(gpu, MAX_LANES * m * 4))
            .collect::<Result<_>>()?;
        let result = (|| -> Result<()> {
            for (wi, wk) in [WKind::Small, WKind::Large, WKind::Wild]
                .into_iter()
                .enumerate()
            {
                mx.wkinds.push(wk.name());
                let w: Vec<Vec<u8>> = rows
                    .iter()
                    .enumerate()
                    .map(|(i, &m)| {
                        synth_weights(
                            m,
                            K / 256,
                            seed ^ 0x1000
                                ^ ((wi as u64) * 17 + self.group as u64)
                                ^ ((i as u64 + 1) << 32),
                            wk,
                        )
                    })
                    .collect();
                let wbufs: Vec<Buf> = w
                    .iter()
                    .map(|b| Buf::from_bytes(gpu, b))
                    .collect::<Result<_>>()?;
                for &n in &lanes {
                    let node = self.at_lanes(n);
                    let sym = node.symbol();
                    let geo = [
                        node.geometry(),
                        node.geometry_with(&mut cand.header.clone())?,
                    ];
                    let funcs = [arms[0].function(gpu, &sym)?, arms[1].function(gpu, &sym)?];
                    for case in 0..=XKINDS.len() {
                        let mut r = Xs64::new(
                            seed ^ 0xA11CE
                                ^ ((n as u64) << 8)
                                ^ ((case as u64) << 16)
                                ^ ((wi as u64) << 24),
                        );
                        let mut x = Vec::with_capacity(n * K);
                        for b in 0..n {
                            // cases 0..7: every lane that activation kind; the last: every kind mixed across lanes
                            let kind = if case < XKINDS.len() {
                                XKINDS[case]
                            } else {
                                XKINDS[b % XKINDS.len()]
                            };
                            x.extend(make_x(kind, &mut r));
                        }
                        let tag = format!(
                            "{} {} n{n} {}",
                            self.group.name(),
                            wk.name(),
                            if case < XKINDS.len() {
                                format!("{:?}", XKINDS[case])
                            } else {
                                "mixed".into()
                            }
                        );
                        xbuf.upload(gpu, &f32_bytes(&x))?;
                        let cpu = Self::cpu_reference(&w, &rows, &x, n);
                        let mut outs: Vec<Vec<Vec<u32>>> = Vec::new();
                        for arm in 0..2 {
                            for (y, &m) in ybufs.iter().zip(&rows) {
                                y.upload(gpu, &word_bytes(&poison_words(MAX_LANES * m)))?;
                            }
                            let mut a = Args::new();
                            for i in 0..4 {
                                a.ptr(&wbufs[if i < rows.len() { i } else { 0 }]);
                            }
                            for i in 0..4 {
                                a.ptr(&ybufs[if i < rows.len() { i } else { 0 }]);
                            }
                            a.ptr(&xbuf);
                            for m in node.m4() {
                                a.i32(m);
                            }
                            let mut karg = a.finish();
                            launch_sync(gpu, &funcs[arm], &geo[arm], &mut karg).map_err(|e| {
                                format!(
                                    "{} arm, {tag}: {e}",
                                    if arm == 0 { "baseline" } else { "candidate" }
                                )
                            })?;
                            let mut per_seg = Vec::new();
                            for y in &ybufs {
                                per_seg.push(words(&y.download(gpu)?));
                            }
                            outs.push(per_seg);
                        }
                        mx.cases += 1;
                        for (s, &m) in rows.iter().enumerate() {
                            let written = n * m;
                            for arm in 0..2 {
                                let o = &outs[arm][s];
                                if o[written..].iter().any(|&v| v != POISON) {
                                    mx.guard_violations[arm] += 1;
                                    mx.fail(format!(
                                        "[{tag}] {} wrote lane rows >= {n} of segment {s}",
                                        ["baseline", "candidate"][arm]
                                    ));
                                }
                                if o[..written].iter().any(|&v| v == POISON) {
                                    mx.poison_left[arm] += 1;
                                    mx.fail(format!(
                                        "[{tag}] {} left poison in segment {s}",
                                        ["baseline", "candidate"][arm]
                                    ));
                                }
                            }
                            let (b, c) = (&outs[0][s][..written], &outs[1][s][..written]);
                            let d = compare(c, b, false);
                            if d.mismatches > 0 {
                                mx.fail(format!(
                                    "[{tag}] candidate != incumbent: segment {s} word {} cand {:#010x} base {:#010x} ({} words)",
                                    d.first.unwrap_or(0), c[d.first.unwrap_or(0)], b[d.first.unwrap_or(0)], d.mismatches
                                ));
                            }
                            mx.cand_vs_base.add(&d);
                            let (db, dc) = (compare(b, &cpu[s], false), compare(c, &cpu[s], false));
                            mx.base_vs_cpu.add(&db);
                            mx.cand_vs_cpu.add(&dc);
                            if db.mismatches > 0 {
                                mx.fail(format!(
                                    "[{tag}] INCUMBENT != CPU reference: segment {s} word {} ({} words) — a reference bug or a hardware numerics fact",
                                    db.first.unwrap_or(0), db.mismatches
                                ));
                            }
                        }
                    }
                }
                for b in wbufs {
                    b.free(gpu);
                }
            }
            Ok(())
        })();
        xbuf.free(gpu);
        for b in ybufs {
            b.free(gpu);
        }
        result?;
        let mut j = mx.json(false);
        j["group"] = json!(self.group.name());
        if j["cpu_ref_identical"]["baseline"] != json!(true) {
            eprintln!(
                "hip_loop: lane_proj: the INCUMBENT differs from the CPU reference ({} mismatching word(s), {} NaN-payload diff(s)): \
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

    fn node(shape: &str) -> LaneProj {
        LaneProj::new(&ShapeSpec::parse(Some(shape)).unwrap()).unwrap()
    }

    #[test]
    fn production_geometry_is_the_launcher_grid() {
        // total 12384 rows: ceil(12384/48) = 258 -> clamp to 32 * 3 = 96.
        let g = node("group=dn,lanes=8").geometry();
        assert_eq!(
            (g.grid, g.block, g.lds),
            ([96, 1, 1], [512, 1, 1], 8 * 2048 * 4)
        );
        // fa: 9216 rows -> 192 -> 96; moe: 1281 rows -> 27 workgroups.
        assert_eq!(node("group=fa,lanes=8").geometry().grid, [96, 1, 1]);
        let g = node("group=moe,lanes=1").geometry();
        assert_eq!((g.grid, g.lds), ([27, 1, 1], 2048 * 4));
        // defaults: dn, 8 lanes
        let d = node("");
        assert_eq!(
            (d.group, d.lanes, d.symbol()),
            (
                Group::Dn,
                8,
                "gemv_mq4g256v2_lane_proj_k2048_gfx1201_n8".to_owned()
            )
        );
    }

    #[test]
    fn segments_match_the_production_groups() {
        assert_eq!(node("group=dn").m4(), [8192, 4096, 32, 32]);
        assert_eq!(node("group=fa").m4(), [8192, 512, 512, 0]);
        assert_eq!(node("group=moe").m4(), [256, 1, 512, 512]);
    }

    #[test]
    fn kernarg_layout_is_the_production_blob() {
        // (A0..A3, y0..y3, x: 9 pointers) + (m0..m3: 4 i32) = 88 B.
        let mut a = Args::new();
        for p in 0..9u64 {
            a.raw_ptr(p);
        }
        for v in [8192, 4096, 32, 32] {
            a.i32(v);
        }
        assert_eq!(a.finish().len(), ARGS_BYTES);
    }

    #[test]
    fn header_geometry() {
        let n = node("group=dn,lanes=4");
        let mut h = Header::parse("// hip_loop: block=256 rows_per_wg=24 wgs_per_wgp=2").unwrap();
        let g = n.geometry_with(&mut h).unwrap();
        // ceil(12384/24) = 516 -> clamp(1, 64)
        assert_eq!(
            (g.grid, g.block, g.lds),
            ([64, 1, 1], [256, 1, 1], 4 * 8192)
        );
        let mut h = Header::parse("// hip_loop: grid_fixed=7 lds_per_lane=4096").unwrap();
        let g = n.geometry_with(&mut h).unwrap();
        assert_eq!((g.grid, g.lds), ([7, 1, 1], 4 * 4096));
        for bad in [
            "block=100",
            "block=0",
            "rows_per_wg=0",
            "grid_fixed=0",
            "lds_per_lane=20000",
            "waves=4",
        ] {
            let mut h = Header::parse(&format!("// hip_loop: {bad}")).unwrap();
            assert!(n.geometry_with(&mut h).is_err(), "{bad}");
        }
    }

    #[test]
    fn shapes_parse_and_refuse() {
        for s in [
            "group=x",
            "lanes=0",
            "lanes=9",
            "group=dn,q=1",
            "rows=1:2:3:4:5",
            "rows=0",
            "nope",
        ] {
            assert!(
                LaneProj::new(&ShapeSpec::parse(Some(s)).unwrap()).is_err(),
                "{s}"
            );
        }
        let n = node("emu");
        assert_eq!((n.lanes, n.rows.clone()), (5, vec![70, 33, 5, 3]));
        assert_eq!(node("fa").group, Group::Fa);
        assert!(node("group=moe,flush=1").flush_requested());
        assert!(!node("group=moe").flush_requested());
    }

    #[test]
    fn extent_shapes_cover_every_group_at_lanes_1_3_8() {
        let v = node("emu").extent_shapes();
        assert_eq!(v.len(), 9);
        assert!(
            v.contains(&"group=fa,lanes=3".to_owned())
                && v.contains(&"group=moe,lanes=8".to_owned())
        );
        for s in &v {
            LaneProj::new(&ShapeSpec::parse(Some(s)).unwrap()).unwrap();
        }
    }

    #[test]
    fn timing_pool_exceeds_the_last_level_cache_with_distinct_odd_sets() {
        for g in Group::ALL {
            let n = node(&format!("group={}", g.name()));
            let sets = pool_sets(n.set_bytes());
            assert!(sets * n.set_bytes() > POOL_BYTES || sets == 161, "{g:?}");
            assert_eq!(sets % 2, 1);
        }
    }

    #[test]
    fn emu_case_packs_outputs_with_pads_and_the_owner_map_follows() {
        for shape in ["emu", "emu,group=fa", "emu,group=moe,lanes=3"] {
            let n = node(shape);
            let c = n.emu_case(1).unwrap();
            let (offs, total) = n.arena();
            assert_eq!(c.bufs[c.out].bytes.len(), total * 4);
            assert_eq!(c.args.len(), 13);
            let own = n.out_owner(&Header::default()).unwrap();
            let g = n.geometry();
            // every real element has an owner inside the grid; every pad element is PAD_OWNER
            for (s, &m) in n.rows.iter().enumerate() {
                for w in 0..n.lanes * m + PAD_WORDS {
                    let o = own(offs[s] + w);
                    if w < n.lanes * m {
                        assert!(
                            o[0] < g.grid[0] && o[1] == 0 && o[2] == 0,
                            "{shape} seg {s} word {w}: {o:?}"
                        );
                    } else {
                        assert_eq!(o, PAD_OWNER);
                    }
                }
            }
        }
    }

    #[test]
    fn chunked_cpu_reference_equals_the_direct_reference() {
        let n = node("emu,lanes=3");
        let (w, x) = n.host_inputs(5);
        let refs: Vec<&[u8]> = w.iter().map(|v| v.as_slice()).collect();
        let direct = lane_ref::lane_proj_ref(&refs, &n.rows, &x, 3);
        assert_eq!(LaneProj::cpu_reference(&w, &n.rows, &x, 3), direct);
    }

    /// Wall time of the CPU reference for one production-size matrix case (dn, 8 lanes); `--ignored --nocapture`.
    #[test]
    #[ignore = "timing probe"]
    fn cpu_reference_cost_dn_8_lanes() {
        let n = node("group=dn,lanes=8");
        let (w, x) = n.host_inputs(5);
        let t = std::time::Instant::now();
        let r = LaneProj::cpu_reference(&w, &n.rows, &x, 8);
        eprintln!(
            "lane_proj cpu reference dn n8: {:?} ({} words)",
            t.elapsed(),
            r.iter().map(Vec::len).sum::<usize>()
        );
    }

    #[test]
    fn weights_use_the_real_group_layout() {
        let w = synth_weights(3, 8, 7, WKind::Small);
        assert_eq!(w.len(), 3 * ROW_BYTES);
        let w = synth_weights(2, 8, 7, WKind::AllFifteen);
        assert!(w[8..GROUP_BYTES].iter().all(|&b| b == 0xFF));
    }
}
