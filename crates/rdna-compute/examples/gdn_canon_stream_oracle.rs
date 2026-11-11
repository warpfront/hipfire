// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LAB ONLY: byte-exact oracle of the canonical GDN prefill stream (Scan64 edge executor,
//! `Gpu::gdn_stream_append` / `Gpu::gdn_stream_seal`, CHUNK-CANONICAL-DESIGN §3.3, §7 G1).
//!
//! Claim under test: appending a row sequence in ANY partition (singletons, 1..63-row remainders,
//! both sides of every tile and 512-row seam, off-grid phase starts, nonzero incoming state/EF)
//! is byte-identical to one whole call: every published output row, the FP32 tile entry, the
//! conv ring at the tile entry and at the position, the saved raw prefix, the Q8 / scales / EF
//! planes at every seam, the metadata, and the sealed end state.
//!
//! Three independent references:
//!   * `legacy`   (aligned start only) the incumbent whole-call route exactly as
//!                `batched_deltanet::gdn_chunk_scan_layer` runs it: prep -> batched KKT ->
//!                `gdn_chunk_scan_layer_mseg` / 512-row `gdn_chunk_scan_segment`, state Q8(+EF) in,
//!                quantized at the call end. Proves the aligned bulk bytes are the base bytes.
//!   * `tiles`    a slow absolute-tile loop built only from the public typed kernels (one prep, one
//!                legacy KKT and one `gdn_chunk_scan_segment_ex` per complete tile, FP32 chained
//!                through an explicit plane, quantized only at absolute seams and the end). Shares
//!                no schedule, carry or replay code with the executor, so it also pins prefix
//!                stability: a tile's rows are computed from the complete tile.
//!   * `whole`    the executor itself, one append.
//! Output-prefix stability is tested directly: a fresh stream appended with only rows `[0, p)`
//! must publish exactly the first `p` rows of the whole-call output (every p in 1..=130 and the
//! tile edges beyond), starting from nonzero state/EF.
//!
//! Negative controls: a one-byte corruption of the saved tile-entry ring (and of the FP32 entry)
//! after the first append must change a later output; the carry planes start as NaN patterns so a
//! stream that read them before writing them fails. All comparisons are byte comparisons; any
//! difference is a failure, none is loosened.
//!
//! Build: cargo build --release -p rdna-compute --features lab,deltanet --example gdn_canon_stream_oracle
//! Run:   HIPFIRE_GRAPH=0 gdn_canon_stream_oracle --json <receipt.json> [--hv 48|32] [--quick]
//! Exit status 0 only if every comparison is byte identical and non-vacuous.

use rdna_compute::gdn_stream::{
    GdnCarryView, GdnLiveState, GdnPrefillPlan, GdnProjectionRecipe, GdnQ8Planes,
    GdnScanAlgorithm, GdnScanEntry, GdnScanExit, GdnSeqInputs, GdnStateQuant, GdnStreamConsts,
    GdnStreamMeta, GdnStreamPhase, GdnStreamRaw, GdnStreamScratch, GdnTensorFormat,
};
use rdna_compute::norm::GdnScanOut;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::collections::BTreeMap;

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

const HD: usize = 128;
const HK: usize = 16;
const EPS: f32 = 1e-6;
const SEED: u64 = 0x5eed_c0de_2026_1111;

fn q_scale() -> f32 {
    1.0 / (HD as f32).sqrt()
}

fn lcg(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 40) as u32) as f32 / (1u32 << 24) as f32;
            lo + (hi - lo) * u
        })
        .collect()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self, m: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % m.max(1)
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 {
        return sign << 15;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return (sign << 15) | 0x7bff;
    }
    if e <= 0 {
        return sign << 15;
    }
    (sign << 15) | ((e as u16) << 10) | (mant >> 13) as u16
}

fn fnv(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

fn download(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<u8>> {
    gpu.hip.device_synchronize()?;
    let mut data = vec![0u8; t.byte_size()];
    gpu.hip.memcpy_dtoh(&mut data, &t.buf)?;
    Ok(data)
}

fn raw(gpu: &mut Gpu, bytes: usize) -> Result<GpuTensor> {
    Ok(gpu.zeros(&[bytes.max(16)], DType::Raw)?)
}

fn upload(gpu: &mut Gpu, data: &[u8]) -> Result<GpuTensor> {
    let t = raw(gpu, data.len())?;
    if !data.is_empty() {
        gpu.memcpy_htod_auto(&t.buf, data)?;
    }
    Ok(t)
}

fn fill(gpu: &mut Gpu, t: &GpuTensor, byte: u8) -> Result<()> {
    gpu.memcpy_htod_auto(&t.buf, &vec![byte; t.byte_size()])?;
    Ok(())
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

fn next64(p: u64) -> u64 {
    (p / 64 + 1) * 64
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Quant {
    Q8,
    Q8Ef,
}

#[derive(Clone, Copy)]
struct Cfg {
    quant: Quant,
    fmt: GdnScanOut,
}

impl Cfg {
    fn name(&self) -> String {
        format!("{:?}/{:?}", self.quant, self.fmt)
    }
    fn out_elem(&self) -> usize {
        match self.fmt {
            GdnScanOut::F32 => 4,
            GdnScanOut::Bf16 => 2,
        }
    }
    fn plan(&self) -> GdnPrefillPlan {
        let mut p = GdnPrefillPlan {
            recipe_id: 0,
            algorithm: GdnScanAlgorithm::Scan64,
            micro_rows: 64,
            commit_rows: 512,
            projection_recipe: GdnProjectionRecipe { qkv: 0, z: 0, alpha: 0, beta: 0, out: 0 },
            scan_operand_format: GdnTensorFormat::F16,
            scan_output_format: match self.fmt {
                GdnScanOut::F32 => GdnTensorFormat::F32,
                GdnScanOut::Bf16 => GdnTensorFormat::Bf16,
            },
            state_quant: match self.quant {
                Quant::Q8 => GdnStateQuant::Q8,
                Quant::Q8Ef => GdnStateQuant::Q8Ef,
            },
            producer_tile_rows: 64,
        };
        p.recipe_id = p.compute_recipe_id();
        p
    }
}

struct Fixture {
    hv: usize,
    xdim: usize,
    rows: usize,
    qkv: Vec<f32>,
    beta_raw: Vec<f32>,
    alpha_raw: Vec<f32>,
    conv_w: GpuTensor,
    dt_bias: GpuTensor,
    a_log: GpuTensor,
    conv0: Vec<u8>,
    q80: Vec<u8>,
    scales0: Vec<u8>,
    ef0: Vec<u8>,
}

impl Fixture {
    fn new(gpu: &mut Gpu, hv: usize, rows: usize) -> Result<Self> {
        let xdim = (2 * HK + hv) * HD;
        let up = |gpu: &mut Gpu, d: Vec<f32>| -> Result<GpuTensor> {
            Ok(gpu.upload_f32(&d, &[d.len()])?)
        };
        let conv_w = up(gpu, lcg(5, 4 * xdim, -0.5, 0.5))?;
        let dt_bias = up(gpu, lcg(6, hv, -4.0, -2.0))?;
        let a_log = up(gpu, lcg(7, hv, 0.0, 2.0))?;
        Ok(Self {
            hv,
            xdim,
            rows,
            qkv: lcg(1, rows * xdim, -1.732_050_8, 1.732_050_8),
            beta_raw: lcg(3, rows * hv, -2.0, 2.0),
            alpha_raw: lcg(4, rows * hv, -1.732_050_8, 1.732_050_8),
            conv_w,
            dt_bias,
            a_log,
            conv0: f32_bytes(&lcg(9, 3 * xdim, -0.5, 0.5)),
            q80: lcg(10, hv * HD * HD, -64.0, 64.0).iter().map(|x| *x as i8 as u8).collect(),
            scales0: f32_bytes(&lcg(11, hv * HD, 0.005, 0.02)),
            ef0: lcg(12, hv * HD * HD, -0.004, 0.004)
                .iter()
                .flat_map(|x| f16_bits(*x).to_ne_bytes())
                .collect(),
        })
    }

    fn consts(&self) -> GdnStreamConsts<'_> {
        GdnStreamConsts {
            conv_weight: &self.conv_w,
            dt_bias: &self.dt_bias,
            a_log: &self.a_log,
            q_scale: q_scale(),
            eps: EPS,
            n_key_heads: HK,
            n_value_heads: self.hv,
        }
    }

    fn x_bytes(&self, row0: usize, rows: usize) -> Vec<u8> {
        f32_bytes(&self.qkv[row0 * self.xdim..(row0 + rows) * self.xdim])
    }
    fn a_bytes(&self, row0: usize, rows: usize) -> Vec<u8> {
        f32_bytes(&self.alpha_raw[row0 * self.hv..(row0 + rows) * self.hv])
    }
    fn b_bytes(&self, row0: usize, rows: usize) -> Vec<u8> {
        f32_bytes(&self.beta_raw[row0 * self.hv..(row0 + rows) * self.hv])
    }

    /// Conv ring (`[channel][3]`, index 0 newest) after absolute row `p`, for a stream whose
    /// row 0 is absolute position `s`: computed on the host from the raw rows, no kernel.
    fn ring_at(&self, s: u64, p: u64) -> Vec<u8> {
        let mut ring = vec![0f32; 3 * self.xdim];
        let c0: Vec<f32> = self
            .conv0
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        for c in 0..self.xdim {
            for j in 0..3usize {
                let r = p as i64 - 1 - j as i64;
                ring[c * 3 + j] = if r >= s as i64 {
                    self.qkv[(r as u64 - s) as usize * self.xdim + c]
                } else {
                    c0[c * 3 + (s as i64 - 1 - r) as usize]
                };
            }
        }
        f32_bytes(&ring)
    }
}

/// Live recurrent state of one layer.
struct Live {
    conv: GpuTensor,
    codes: GpuTensor,
    scales: GpuTensor,
    ef: Option<GpuTensor>,
}

impl Live {
    fn new(gpu: &mut Gpu, fx: &Fixture, quant: Quant) -> Result<Self> {
        let ef = match quant {
            Quant::Q8Ef => Some(upload(gpu, &fx.ef0)?),
            Quant::Q8 => None,
        };
        Ok(Self {
            conv: upload(gpu, &fx.conv0)?,
            codes: upload(gpu, &fx.q80)?,
            scales: upload(gpu, &fx.scales0)?,
            ef,
        })
    }
    fn view(&self) -> GdnLiveState<'_> {
        GdnLiveState { planes: self.planes(), conv: &self.conv }
    }
    fn planes(&self) -> GdnQ8Planes<'_> {
        GdnQ8Planes { codes: &self.codes, scales: &self.scales, ef: self.ef.as_ref() }
    }
    fn snap(&self, gpu: &Gpu) -> Result<StateBytes> {
        Ok(StateBytes {
            conv: download(gpu, &self.conv)?,
            codes: download(gpu, &self.codes)?,
            scales: download(gpu, &self.scales)?,
            ef: match &self.ef {
                Some(e) => Some(download(gpu, e)?),
                None => None,
            },
        })
    }
    fn free(self, gpu: &mut Gpu) -> Result<()> {
        gpu.free_tensor(self.conv)?;
        gpu.free_tensor(self.codes)?;
        gpu.free_tensor(self.scales)?;
        if let Some(e) = self.ef {
            gpu.free_tensor(e)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
struct StateBytes {
    conv: Vec<u8>,
    codes: Vec<u8>,
    scales: Vec<u8>,
    ef: Option<Vec<u8>>,
}

struct Carry {
    f32: GpuTensor,
    conv: GpuTensor,
    qkv: GpuTensor,
    alpha: GpuTensor,
    beta: GpuTensor,
}

impl Carry {
    /// Every plane starts as a NaN pattern: a stream that consumes carry before writing it fails.
    fn new(gpu: &mut Gpu, fx: &Fixture) -> Result<Self> {
        let hv = fx.hv;
        let c = Self {
            f32: raw(gpu, hv * HD * HD * 4)?,
            conv: raw(gpu, 3 * fx.xdim * 4)?,
            qkv: raw(gpu, 64 * fx.xdim * 4)?,
            alpha: raw(gpu, 64 * hv * 4)?,
            beta: raw(gpu, 64 * hv * 4)?,
        };
        for t in [&c.f32, &c.conv, &c.qkv, &c.alpha, &c.beta] {
            fill(gpu, t, 0xff)?;
        }
        Ok(c)
    }
    fn view(&self) -> GdnCarryView<'_> {
        GdnCarryView {
            tile_entry_f32: &self.f32,
            tile_entry_conv: &self.conv,
            pending_qkv: &self.qkv,
            pending_alpha: &self.alpha,
            pending_beta: &self.beta,
        }
    }
    fn free(self, gpu: &mut Gpu) -> Result<()> {
        for t in [self.f32, self.conv, self.qkv, self.alpha, self.beta] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
}

/// What the stream holds after an append (valid bytes only).
struct Boundary {
    meta: GdnStreamMeta,
    tile_f32: Vec<u8>,
    tile_conv: Vec<u8>,
    live_conv: Vec<u8>,
    pend_x: Vec<u8>,
    pend_a: Vec<u8>,
    pend_b: Vec<u8>,
    planes: StateBytes,
}

struct StreamRun {
    out: Vec<u8>,
    boundaries: Vec<Boundary>,
    sealed: StateBytes,
    final_meta: GdnStreamMeta,
}

/// `parts` appended in order from a stream sealed at absolute `s`; `x_row0` is the fixture row of
/// absolute position `s` (always 0: the fixture row index is `pos - s`). `corrupt`: after the
/// append with that index, flip one byte of the tile-entry ring (`1`) or FP32 entry (`2`).
#[allow(clippy::too_many_arguments)]
fn stream_run(
    gpu: &mut Gpu,
    fx: &Fixture,
    cfg: &Cfg,
    s: u64,
    parts: &[usize],
    collect: bool,
    seal: bool,
    corrupt: Option<(usize, u8)>,
) -> Result<StreamRun> {
    let plan = cfg.plan();
    let hv = fx.hv;
    let live = Live::new(gpu, fx, cfg.quant)?;
    let carry = Carry::new(gpu, fx)?;
    let ws = gpu.gdn_stream_workspace_alloc(HK, hv, cfg.fmt)?;
    let consts = fx.consts();
    let orow = hv * HD * cfg.out_elem();
    let mut meta = GdnStreamMeta::sealed_at(s, 1);
    let mut out = Vec::new();
    let mut boundaries = Vec::new();
    let mut done = 0usize;
    for (i, &n) in parts.iter().enumerate() {
        let alloc = n.div_ceil(64) * 64;
        let x = upload(gpu, &fx.x_bytes(done, n))?;
        let a = upload(gpu, &fx.a_bytes(done, n))?;
        let b = upload(gpu, &fx.b_bytes(done, n))?;
        // Scratch is poisoned too: the executor must not read prepared rows it did not write.
        let q = raw(gpu, alloc * HK * HD * 2)?;
        let k = raw(gpu, alloc * HK * HD * 2)?;
        let v = raw(gpu, alloc * hv * HD * 2)?;
        let aa = raw(gpu, alloc * hv * 64 * 2)?;
        for t in [&q, &k, &v, &aa] {
            fill(gpu, t, 0xff)?;
        }
        let o = raw(gpu, n * orow)?;
        let raw_in = GdnStreamRaw { qkv: &x, alpha: &a, beta: &b };
        let scratch = GdnStreamScratch { q: &q, k: &k, v: &v, a: &aa };
        let pos = s + done as u64;
        meta = gpu.gdn_stream_append(
            &plan,
            &meta,
            pos,
            n,
            &consts,
            &raw_in,
            &scratch,
            &o,
            &live.view(),
            &carry.view(),
            &ws,
        )?;
        out.extend_from_slice(&download(gpu, &o)?[..n * orow]);
        done += n;
        if collect {
            let m = meta.pending_rows as usize;
            let xr = fx.xdim * 4;
            boundaries.push(Boundary {
                meta,
                tile_f32: download(gpu, &carry.f32)?,
                tile_conv: download(gpu, &carry.conv)?,
                live_conv: download(gpu, &live.conv)?,
                pend_x: download(gpu, &carry.qkv)?[..m * xr].to_vec(),
                pend_a: download(gpu, &carry.alpha)?[..m * hv * 4].to_vec(),
                pend_b: download(gpu, &carry.beta)?[..m * hv * 4].to_vec(),
                planes: live.snap(gpu)?,
            });
        }
        if let Some((at, what)) = corrupt {
            if at == i {
                let t = if what == 1 { &carry.conv } else { &carry.f32 };
                let mut d = download(gpu, t)?;
                // Negate a whole head's worth of entries (the ring of the first three K heads,
                // which feed the recurrence state; or the first four state
                // rows): a single mantissa bit can round away in F16, and the Q ring only
                // touches the replayed (unpublished) rows' own outputs.
                let (first, span) = if what == 1 { (HK * HD * 3, (2 * HD + HD) * 3) } else { (0, 512) };
                for w in first..first + span {
                    d[w * 4 + 3] ^= 0x80;
                }
                gpu.memcpy_htod_auto(&t.buf, &d)?;
            }
        }
        for t in [x, a, b, q, k, v, aa, o] {
            gpu.free_tensor(t)?;
        }
    }
    let final_meta = if seal {
        gpu.gdn_stream_seal(&plan, &meta, &consts, &live.view(), &carry.view(), &ws)?
    } else {
        meta
    };
    let sealed = live.snap(gpu)?;
    live.free(gpu)?;
    carry.free(gpu)?;
    gpu.gdn_stream_workspace_free(ws)?;
    Ok(StreamRun { out, boundaries, sealed, final_meta })
}

/// Independent absolute-tile reference (module docs). Returns outputs, the FP32 plane after every
/// complete tile and the open state, the Q8 planes after every seam, and the sealed end state.
struct TileRef {
    out: Vec<u8>,
    f32_at: BTreeMap<u64, Vec<u8>>,
    planes_at: BTreeMap<u64, StateBytes>,
    end: StateBytes,
}

fn tile_ref(gpu: &mut Gpu, fx: &Fixture, cfg: &Cfg, s: u64, n: usize) -> Result<TileRef> {
    let hv = fx.hv;
    let live = Live::new(gpu, fx, cfg.quant)?;
    let h = raw(gpu, hv * HD * HD * 4)?;
    let orow = hv * HD * cfg.out_elem();
    let dummy = raw(gpu, 64)?;
    fn entry_live(l: &Live) -> GdnScanEntry<'_> {
        match &l.ef {
            Some(ef) => GdnScanEntry::Q8Ef { codes: &l.codes, scales: &l.scales, ef },
            None => GdnScanEntry::Q8 { codes: &l.codes, scales: &l.scales },
        }
    }
    // Open: Q8(+EF) -> FP32 once.
    gpu.gdn_chunk_scan_segment_ex(
        &dummy,
        &dummy,
        &dummy,
        &dummy,
        &dummy,
        &dummy,
        &entry_live(&live),
        &GdnScanExit::F32 { state: &h },
        &dummy,
        cfg.fmt,
        0,
        0,
        false,
        HK,
        hv,
    )?;
    let mut f32_at = BTreeMap::new();
    let mut planes_at = BTreeMap::new();
    f32_at.insert(s, download(gpu, &h)?);
    let ring = upload(gpu, &fx.conv0)?;
    let end_pos = s + n as u64;
    let mut out = Vec::new();
    let mut pos = s;
    while pos < end_pos {
        let t_end = next64(pos).min(end_pos);
        let rows = (t_end - pos) as usize;
        let complete = t_end == next64(pos);
        let off = (pos - s) as usize;
        let x = upload(gpu, &fx.x_bytes(off, rows))?;
        let a = upload(gpu, &fx.a_bytes(off, rows))?;
        let b = upload(gpu, &fx.b_bytes(off, rows))?;
        let q = raw(gpu, 64 * HK * HD * 2)?;
        let k = raw(gpu, 64 * HK * HD * 2)?;
        let v = raw(gpu, 64 * hv * HD * 2)?;
        let aa = raw(gpu, 64 * hv * 64 * 2)?;
        let o = raw(gpu, 64 * orow)?;
        gpu.gdn_chunk_prep(&x, &fx.conv_w, &ring, &a, &b, &fx.dt_bias, &fx.a_log, &q, &k, &v, rows, q_scale(), EPS, HK, hv)?;
        let last = t_end == end_pos;
        let seam = t_end % 512 == 0;
        let (planes_ef, planes_c, planes_s) = (live.ef.as_ref(), &live.codes, &live.scales);
        let exit = if last && !complete {
            GdnScanExit::Seal { codes: planes_c, scales: planes_s, ef: planes_ef }
        } else if seam {
            GdnScanExit::Seam { codes: planes_c, scales: planes_s, ef: planes_ef, next: &h }
        } else {
            GdnScanExit::F32 { state: &h }
        };
        gpu.gdn_chunk_scan_segment_ex(
            &q,
            &k,
            &v,
            &aa,
            &a,
            &b,
            &GdnScanEntry::F32 { state: &h },
            &exit,
            &o,
            cfg.fmt,
            0,
            rows,
            true,
            HK,
            hv,
        )?;
        out.extend_from_slice(&download(gpu, &o)?[..rows * orow]);
        if complete {
            f32_at.insert(t_end, download(gpu, &h)?);
            if seam {
                planes_at.insert(t_end, live.snap(gpu)?);
            }
        }
        for t in [x, a, b, q, k, v, aa, o] {
            gpu.free_tensor(t)?;
        }
        pos = t_end;
    }
    // Seal a complete final tile that is not a seam from its FP32 end state.
    if end_pos % 64 == 0 && end_pos % 512 != 0 && n > 0 {
        gpu.gdn_chunk_scan_segment_ex(
            &dummy,
            &dummy,
            &dummy,
            &dummy,
            &dummy,
            &dummy,
            &GdnScanEntry::F32 { state: &h },
            &GdnScanExit::Seal { codes: &live.codes, scales: &live.scales, ef: live.ef.as_ref() },
            &dummy,
            cfg.fmt,
            0,
            0,
            false,
            HK,
            hv,
        )?;
    }
    let mut end = live.snap(gpu)?;
    end.conv = download(gpu, &ring)?;
    gpu.free_tensor(ring)?;
    gpu.free_tensor(h)?;
    gpu.free_tensor(dummy)?;
    live.free(gpu)?;
    Ok(TileRef { out, f32_at, planes_at, end })
}

/// The incumbent whole-call route (`gdn_chunk_scan_layer`: prep, batched KKT when more than one
/// segment, mseg or 512-row segments), state entering as Q8(+EF) and leaving quantized.
fn legacy_run(gpu: &mut Gpu, fx: &Fixture, cfg: &Cfg, n: usize) -> Result<(Vec<u8>, StateBytes)> {
    let hv = fx.hv;
    let live = Live::new(gpu, fx, cfg.quant)?;
    let ef_zero = match cfg.quant {
        Quant::Q8 => Some(raw(gpu, hv * HD * HD * 2)?),
        Quant::Q8Ef => None,
    };
    let ef = live.ef.as_ref().or(ef_zero.as_ref()).unwrap();
    let alloc = n.div_ceil(64) * 64;
    let orow = hv * HD * cfg.out_elem();
    let x = upload(gpu, &fx.x_bytes(0, n))?;
    let a = upload(gpu, &fx.a_bytes(0, n))?;
    let b = upload(gpu, &fx.b_bytes(0, n))?;
    let q = raw(gpu, alloc * HK * HD * 2)?;
    let k = raw(gpu, alloc * HK * HD * 2)?;
    let v = raw(gpu, alloc * hv * HD * 2)?;
    let aa = raw(gpu, alloc * hv * 64 * 2)?;
    let o = raw(gpu, n * orow)?;
    let seg = 512usize;
    gpu.gdn_chunk_prepare(cfg.fmt)?;
    let kkt_batched = gpu.gdn_kkt_batched_enabled() && n > seg;
    gpu.gdn_chunk_prep(&x, &fx.conv_w, &live.conv, &a, &b, &fx.dt_bias, &fx.a_log, &q, &k, &v, n, q_scale(), EPS, HK, hv)?;
    if kkt_batched {
        gpu.gdn_chunk_kkt_solve_batched(&k, &a, &b, &aa, n, HK, hv)?;
    }
    if kkt_batched && gpu.gdn_scan_mseg_enabled(cfg.fmt, seg) {
        gpu.gdn_chunk_scan_layer_mseg(&q, &k, &v, &aa, &a, &b, &live.codes, &live.scales, ef, &o, n, HK, hv)?;
    } else {
        for row0 in (0..n).step_by(seg) {
            let rows = (n - row0).min(seg);
            let a_seg = if kkt_batched {
                aa.sub_offset(row0 * hv * 64 * 2, rows.div_ceil(64) * 64 * hv * 64 * 2)
            } else {
                aa.shallow_clone()
            };
            gpu.gdn_chunk_scan_segment(&q, &k, &v, &a_seg, &a, &b, &live.codes, &live.scales, ef, &o, cfg.fmt, row0, rows, !kkt_batched, HK, hv)?;
        }
    }
    let out = download(gpu, &o)?;
    let mut st = live.snap(gpu)?;
    if cfg.quant == Quant::Q8 {
        st.ef = None;
    }
    for t in [x, a, b, q, k, v, aa, o] {
        gpu.free_tensor(t)?;
    }
    if let Some(e) = ef_zero {
        gpu.free_tensor(e)?;
    }
    live.free(gpu)?;
    Ok((out[..n * orow].to_vec(), st))
}

struct Report {
    comparisons: usize,
    bad: Vec<String>,
}

impl Report {
    fn cmp(&mut self, ctx: &str, name: &str, a: &[u8], b: &[u8]) {
        self.comparisons += 1;
        if let Some(d) = first_diff(a, b) {
            if self.bad.len() < 400 {
                self.bad.push(format!(
                    "{ctx}/{name}: first diff at byte {d} (lens {}/{}, hashes {}/{})",
                    a.len(),
                    b.len(),
                    fnv(a),
                    fnv(b)
                ));
            } else if self.bad.len() == 400 {
                self.bad.push("... further failures suppressed".into());
            }
        }
    }
    fn check(&mut self, ctx: &str, name: &str, ok: bool) {
        self.comparisons += 1;
        if !ok {
            self.bad.push(format!("{ctx}/{name}: expectation violated"));
        }
    }
}

fn cmp_state(r: &mut Report, ctx: &str, a: &StateBytes, b: &StateBytes) {
    r.cmp(ctx, "codes", &a.codes, &b.codes);
    r.cmp(ctx, "scales", &a.scales, &b.scales);
    match (&a.ef, &b.ef) {
        (Some(x), Some(y)) => r.cmp(ctx, "ef", x, y),
        (None, None) => {}
        _ => r.check(ctx, "ef_presence", false),
    }
    r.cmp(ctx, "conv", &a.conv, &b.conv);
}

/// Tile start containing absolute position `p` for a phase starting at `s`.
fn tile_base(s: u64, p: u64) -> u64 {
    if p <= s {
        s
    } else {
        (p / 64 * 64).max(s)
    }
}

/// Compare every boundary of a partitioned stream against the references.
#[allow(clippy::too_many_arguments)]
fn check_boundaries(
    r: &mut Report,
    ctx: &str,
    fx: &Fixture,
    cfg: &Cfg,
    s: u64,
    parts: &[usize],
    run: &StreamRun,
    tref: &TileRef,
) {
    let xr = fx.xdim * 4;
    let orow = fx.hv * HD * cfg.out_elem();
    let total: usize = parts.iter().sum();
    r.cmp(ctx, "out_rows", &run.out, &tref.out[..total * orow]);
    let mut p = s;
    for (i, (&n, b)) in parts.iter().zip(&run.boundaries).enumerate() {
        p += n as u64;
        let c = format!("{ctx}/append{i}@{p}");
        let tb = tile_base(s, p);
        let pend = (p - tb) as usize;
        r.check(&c, "meta.position", b.meta.position == p);
        r.check(&c, "meta.tile_base", b.meta.tile_base == tb);
        r.check(&c, "meta.pending_rows", b.meta.pending_rows as usize == pend);
        r.check(&c, "meta.frame_base", b.meta.frame_base == s.max(p / 512 * 512));
        r.check(&c, "meta.phase_open", b.meta.phase == GdnStreamPhase::Open);
        match tref.f32_at.get(&tb) {
            Some(h) => r.cmp(&c, "tile_entry_f32", &b.tile_f32, h),
            None => r.check(&c, "tile_entry_f32_reference_missing", false),
        }
        r.cmp(&c, "tile_entry_conv", &b.tile_conv, &fx.ring_at(s, tb));
        r.cmp(&c, "live_conv", &b.live_conv, &fx.ring_at(s, p));
        let (xo, ao) = ((tb - s) as usize, (tb - s) as usize);
        r.cmp(&c, "pending_qkv", &b.pend_x, &fx.x_bytes(xo, pend));
        r.cmp(&c, "pending_alpha", &b.pend_a, &fx.a_bytes(ao, pend));
        r.cmp(&c, "pending_beta", &b.pend_b, &fx.b_bytes(ao, pend));
        if p % 512 == 0 {
            match tref.planes_at.get(&p) {
                Some(pl) => {
                    r.cmp(&c, "seam.codes", &b.planes.codes, &pl.codes);
                    r.cmp(&c, "seam.scales", &b.planes.scales, &pl.scales);
                    if let (Some(x), Some(y)) = (&b.planes.ef, &pl.ef) {
                        r.cmp(&c, "seam.ef", x, y);
                    }
                }
                None => r.check(&c, "seam_reference_missing", false),
            }
        }
    }
    let _ = xr;
    r.check(ctx, "sealed_meta", run.final_meta.phase == GdnStreamPhase::Sealed && run.final_meta.pending_rows == 0);
    cmp_state(r, &format!("{ctx}/sealed"), &run.sealed, &tref.end);
}

fn random_parts(rng: &mut Rng, total: usize) -> Vec<usize> {
    let mut parts = Vec::new();
    let mut left = total;
    while left > 0 {
        let pick = match rng.next(6) {
            0 => 1,
            1 => 1 + rng.next(63) as usize,
            2 => 63 + rng.next(3) as usize,
            3 => 1 + rng.next(130) as usize,
            4 => 1 + rng.next(700) as usize,
            _ => 508 + rng.next(10) as usize,
        };
        let n = pick.min(left);
        parts.push(n);
        left -= n;
    }
    parts
}

/// Prepared (post-conv, post-gate) inputs of the sequential algorithm: q/k expanded to the value
/// heads (`qk_head_ratio` 1, the legacy fast-kernel layout).
struct SeqFx {
    hv: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
    beta: Vec<f32>,
}

impl SeqFx {
    fn new(hv: usize, rows: usize) -> Self {
        Self {
            hv,
            q: lcg(21, rows * hv * HD, -0.2, 0.2),
            k: lcg(22, rows * hv * HD, -0.2, 0.2),
            v: lcg(23, rows * hv * HD, -1.0, 1.0),
            gate: lcg(24, rows * hv, -0.5, -0.01),
            beta: lcg(25, rows * hv, 0.1, 0.9),
        }
    }
}

fn seq_plan(quant: Quant) -> GdnPrefillPlan {
    let mut p = GdnPrefillPlan {
        recipe_id: 0,
        algorithm: GdnScanAlgorithm::Sequential,
        micro_rows: 1,
        commit_rows: 512,
        projection_recipe: GdnProjectionRecipe { qkv: 0, z: 0, alpha: 0, beta: 0, out: 0 },
        scan_operand_format: GdnTensorFormat::F32,
        scan_output_format: GdnTensorFormat::F32,
        state_quant: match quant {
            Quant::Q8 => GdnStateQuant::Q8,
            Quant::Q8Ef => GdnStateQuant::Q8Ef,
        },
        producer_tile_rows: 1,
    };
    p.recipe_id = p.compute_recipe_id();
    p
}

/// `parts` appended in order through `gdn_stream_append_sequential` (poisoned FP32 carry), then
/// sealed. `corrupt`: negate the first state rows of the FP32 carry after that append.
#[allow(clippy::too_many_arguments)]
fn seq_run(
    gpu: &mut Gpu,
    fx: &Fixture,
    sf: &SeqFx,
    quant: Quant,
    s: u64,
    parts: &[usize],
    seal: bool,
    corrupt: Option<usize>,
) -> Result<(Vec<u8>, StateBytes)> {
    let hv = sf.hv;
    let plan = seq_plan(quant);
    let live = Live::new(gpu, fx, quant)?;
    let carry = raw(gpu, hv * HD * HD * 4)?;
    fill(gpu, &carry, 0xff)?;
    let mut meta = GdnStreamMeta::sealed_at(s, 1);
    let mut out = Vec::new();
    let mut done = 0usize;
    for (i, &n) in parts.iter().enumerate() {
        let w = hv * HD;
        let q = upload(gpu, &f32_bytes(&sf.q[done * w..(done + n) * w]))?;
        let k = upload(gpu, &f32_bytes(&sf.k[done * w..(done + n) * w]))?;
        let v = upload(gpu, &f32_bytes(&sf.v[done * w..(done + n) * w]))?;
        let g = upload(gpu, &f32_bytes(&sf.gate[done * hv..(done + n) * hv]))?;
        let b = upload(gpu, &f32_bytes(&sf.beta[done * hv..(done + n) * hv]))?;
        let o = raw(gpu, n * w * 4)?;
        let inp = GdnSeqInputs { q: &q, k: &k, v: &v, gate: &g, beta: &b, qk_head_ratio: 1 };
        meta = gpu.gdn_stream_append_sequential(&plan, &meta, s + done as u64, n, &inp, &o, &live.planes(), &carry, hv)?;
        out.extend_from_slice(&download(gpu, &o)?[..n * w * 4]);
        done += n;
        if corrupt == Some(i) {
            let mut d = download(gpu, &carry)?;
            for x in 0..512 {
                d[x * 4 + 3] ^= 0x80;
            }
            gpu.memcpy_htod_auto(&carry.buf, &d)?;
        }
        for t in [q, k, v, g, b, o] {
            gpu.free_tensor(t)?;
        }
    }
    if seal {
        gpu.gdn_stream_seal_sequential(&plan, &meta, &live.planes(), &carry, hv)?;
    }
    let st = live.snap(gpu)?;
    live.free(gpu)?;
    gpu.free_tensor(carry)?;
    Ok((out, st))
}

/// Incumbent sequential route for Q8Ef: `gated_delta_net_q8` (fast kernel) per absolute 512-row
/// segment.
fn seq_legacy(gpu: &mut Gpu, fx: &Fixture, sf: &SeqFx, s: u64, n: usize) -> Result<(Vec<u8>, StateBytes)> {
    let hv = sf.hv;
    let live = Live::new(gpu, fx, Quant::Q8Ef)?;
    let w = hv * HD;
    let mut out = Vec::new();
    let (mut pos, end) = (s, s + n as u64);
    while pos < end {
        let e = end.min((pos / 512 + 1) * 512);
        let (off, len) = ((pos - s) as usize, (e - pos) as usize);
        let q = upload(gpu, &f32_bytes(&sf.q[off * w..(off + len) * w]))?;
        let k = upload(gpu, &f32_bytes(&sf.k[off * w..(off + len) * w]))?;
        let v = upload(gpu, &f32_bytes(&sf.v[off * w..(off + len) * w]))?;
        let g = upload(gpu, &f32_bytes(&sf.gate[off * hv..(off + len) * hv]))?;
        let b = upload(gpu, &f32_bytes(&sf.beta[off * hv..(off + len) * hv]))?;
        let o = raw(gpu, len * w * 4)?;
        gpu.gated_delta_net_q8(&q, &k, &v, &g, &b, &live.codes, &live.scales, &o, len, hv, HD, live.ef.as_ref())?;
        out.extend_from_slice(&download(gpu, &o)?[..len * w * 4]);
        for t in [q, k, v, g, b, o] {
            gpu.free_tensor(t)?;
        }
        pos = e;
    }
    let st = live.snap(gpu)?;
    live.free(gpu)?;
    Ok((out, st))
}

fn run() -> Result<()> {
    let mut json_path: Option<String> = None;
    let mut hv = 48usize;
    let mut quick = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--json" => json_path = args.next(),
            "--hv" => hv = args.next().ok_or("--hv value")?.parse()?,
            "--quick" => quick = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if !matches!(hv, 48 | 32) {
        return Err("--hv must be 48 or 32".into());
    }
    let mut gpu = Gpu::init()?;
    if hv != 48 && gpu.arch != "gfx1201" {
        return Err("hv 32 only runs on gfx1201".into());
    }
    if hipfire_config::developer_var("HIPFIRE_GDN_SCAN_OUT_EMU").is_ok() {
        return Err("unset HIPFIRE_GDN_SCAN_OUT_EMU".into());
    }
    let fmts: Vec<GdnScanOut> = if gpu.arch == "gfx1201" {
        vec![GdnScanOut::Bf16, GdnScanOut::F32]
    } else {
        vec![GdnScanOut::F32]
    };
    let starts: Vec<u64> = if quick {
        vec![0, 25, 512]
    } else {
        vec![0, 1, 25, 63, 64, 65, 511, 512, 513, 521, 45979]
    };
    let total_rows = 1100usize;
    let fx = Fixture::new(&mut gpu, hv, total_rows)?;
    let mut rng = Rng(SEED);
    let mut rep = Report { comparisons: 0, bad: Vec::new() };
    let mut cells = Vec::new();

    for &fmt in &fmts {
        for quant in [Quant::Q8Ef, Quant::Q8] {
            let cfg = Cfg { quant, fmt };
            let cname = cfg.name();

            // ---- legacy (incumbent whole call) == tiles == whole, aligned start 0 ----
            let legacy_ns: Vec<usize> = if quick {
                vec![1, 25, 63, 64, 65, 512, 575]
            } else {
                vec![1, 2, 3, 17, 25, 31, 32, 33, 62, 63, 64, 65, 100, 127, 128, 129, 300, 511, 512, 513, 575, 1024, 1100]
            };
            for &n in &legacy_ns {
                let ctx = format!("{cname}/legacy-n{n}");
                let (lo, ls) = legacy_run(&mut gpu, &fx, &cfg, n)?;
                let t = tile_ref(&mut gpu, &fx, &cfg, 0, n)?;
                rep.cmp(&ctx, "tiles_vs_legacy.out", &t.out, &lo);
                cmp_state(&mut rep, &format!("{ctx}/tiles_vs_legacy"), &t.end, &ls);
                let w = stream_run(&mut gpu, &fx, &cfg, 0, &[n], false, true, None)?;
                rep.cmp(&ctx, "whole_vs_legacy.out", &w.out, &lo);
                cmp_state(&mut rep, &format!("{ctx}/whole_vs_legacy"), &w.sealed, &ls);
                rep.check(&ctx, "nonvacuous_out", lo.iter().any(|b| *b != 0));
                rep.check(&ctx, "state_moved", ls.codes != fx.q80);
            }

            // ---- per start: partitions == whole == tiles ----
            for &s in &starts {
                let n_total = if s == 45979 { 700 } else { total_rows };
                let ctx0 = format!("{cname}/s{s}");
                let tref = tile_ref(&mut gpu, &fx, &cfg, s, n_total)?;
                let orow = hv * HD * cfg.out_elem();
                rep.check(&ctx0, "nonvacuous_out", tref.out.iter().any(|b| *b != 0));
                rep.check(&ctx0, "state_moved", tref.end.codes != fx.q80);
                let d_seam = (512 - (s % 512)) as usize;
                let d_tile = (64 - (s % 64)) as usize;
                let mut partitions: Vec<(String, Vec<usize>)> = vec![("whole".into(), vec![n_total])];
                for dd in [d_seam - 1, d_seam, d_seam + 1] {
                    if dd >= 1 && dd < n_total {
                        partitions.push((format!("cut-seam{dd}"), vec![dd, n_total - dd]));
                    }
                }
                for dd in [d_tile - 1, d_tile, d_tile + 1, d_tile + 63, d_tile + 64] {
                    if dd >= 1 && dd < n_total {
                        partitions.push((format!("cut-tile{dd}"), vec![dd, n_total - dd]));
                    }
                }
                let singles = 140usize.min(n_total);
                let mut p = vec![1usize; singles];
                if n_total > singles {
                    p.push(n_total - singles);
                }
                partitions.push(("singletons140+rest".into(), p));
                let mut eq64 = vec![64usize; n_total / 64];
                if n_total % 64 > 0 {
                    eq64.push(n_total % 64);
                }
                partitions.push(("rows64".into(), eq64));
                let nrand = if quick { 3 } else { 8 };
                for i in 0..nrand {
                    partitions.push((format!("rand{i}"), random_parts(&mut rng, n_total)));
                }
                for (name, parts) in &partitions {
                    let ctx = format!("{ctx0}/{name}");
                    let run = stream_run(&mut gpu, &fx, &cfg, s, parts, true, true, None)?;
                    check_boundaries(&mut rep, &ctx, &fx, &cfg, s, parts, &run, &tref);
                }

                // Whole-prefix runs at every boundary of every partition are covered by the
                // boundary checks above (the tile reference supplies the expected entry); the
                // output-prefix stability sweep appends only rows [0, p) to a fresh stream.
                let sweep: Vec<usize> = if quick {
                    vec![1, 2, 3, 63, 64, 65]
                } else if cfg.quant == Quant::Q8Ef && fmt == fmts[0] {
                    (1..=130).chain([191, 192, 193, 255, 256, 257, 447, 448, 449, 511, 512, 513]).collect()
                } else {
                    vec![1, 2, 3, 31, 32, 33, 62, 63, 64, 65, 127, 128, 129]
                };
                for &pfx in &sweep {
                    if pfx > n_total {
                        continue;
                    }
                    let ctx = format!("{ctx0}/prefix{pfx}");
                    let run = stream_run(&mut gpu, &fx, &cfg, s, &[pfx], false, false, None)?;
                    rep.cmp(&ctx, "out_prefix_vs_whole", &run.out, &tref.out[..pfx * orow]);
                }
                cells.push(json!({"cfg": cname, "start": s, "rows": n_total, "partitions": partitions.len(), "prefix_sweep": sweep.len()}));
            }

            // ---- negative controls: a corrupted carry must change a later output ----
            {
                let s = 25u64;
                let n_total = 300usize;
                let tref = tile_ref(&mut gpu, &fx, &cfg, s, n_total)?;
                let orow = hv * HD * cfg.out_elem();
                for (what, label) in [(1u8, "tile_entry_conv"), (2u8, "tile_entry_f32")] {
                    // first append stays inside the clipped tile (pending), the second completes it
                    let parts = [10usize, n_total - 10];
                    let run = stream_run(&mut gpu, &fx, &cfg, s, &parts, false, true, Some((0, what)))?;
                    let ctx = format!("{cname}/control-{label}");
                    rep.check(&ctx, "corruption_detected", run.out != tref.out[..n_total * orow]);
                    // The 25..35 rows were published before the corruption and must be unchanged.
                    rep.cmp(&ctx, "pre_corruption_rows_unchanged", &run.out[..10 * orow], &tref.out[..10 * orow]);
                }
            }
        }
    }

    // ---- Sequential (C = 1) algorithm: FP32 continuation, quantize only at absolute seams ----
    for quant in [Quant::Q8Ef, Quant::Q8] {
        let cname = format!("seq/{quant:?}");
        let sf = SeqFx::new(hv, total_rows);
        for &s in &[0u64, 25, 511, 512, 513, 45979] {
            let n_total = if s == 45979 { 700 } else { total_rows };
            let ctx0 = format!("{cname}/s{s}");
            let (whole_out, whole_state) = seq_run(&mut gpu, &fx, &sf, quant, s, &[n_total], true, None)?;
            rep.check(&ctx0, "nonvacuous_out", whole_out.iter().any(|b| *b != 0));
            rep.check(&ctx0, "state_moved", whole_state.codes != fx.q80);
            if quant == Quant::Q8Ef {
                // Incumbent route: one fast-kernel launch per absolute 512-row segment, Q8+EF
                // round trip between them and at the call end.
                let (lo, ls) = seq_legacy(&mut gpu, &fx, &sf, s, n_total)?;
                rep.cmp(&ctx0, "whole_vs_legacy.out", &whole_out, &lo);
                cmp_state(&mut rep, &format!("{ctx0}/whole_vs_legacy"), &whole_state, &ls);
            }
            let d_seam = (512 - (s % 512)) as usize;
            let mut partitions: Vec<(String, Vec<usize>)> = Vec::new();
            for dd in [d_seam - 1, d_seam, d_seam + 1, 1, 63, 64] {
                if dd >= 1 && dd < n_total {
                    partitions.push((format!("cut{dd}"), vec![dd, n_total - dd]));
                }
            }
            let mut p = vec![1usize; 140];
            p.push(n_total - 140);
            partitions.push(("singletons140+rest".into(), p));
            for i in 0..(if quick { 3 } else { 8 }) {
                partitions.push((format!("rand{i}"), random_parts(&mut rng, n_total)));
            }
            for (name, parts) in &partitions {
                let ctx = format!("{ctx0}/{name}");
                let (o, st) = seq_run(&mut gpu, &fx, &sf, quant, s, parts, true, None)?;
                rep.cmp(&ctx, "out_rows", &o, &whole_out);
                cmp_state(&mut rep, &format!("{ctx}/sealed"), &st, &whole_state);
            }
            // Negative control: a corrupted FP32 carry must change a later output.
            let (co, _) = seq_run(&mut gpu, &fx, &sf, quant, s, &[10, n_total - 10], true, Some(0))?;
            let orow = hv * HD * 4;
            rep.check(&ctx0, "control_corruption_detected", co != whole_out);
            rep.cmp(&ctx0, "control_pre_rows_unchanged", &co[..10 * orow], &whole_out[..10 * orow]);
            cells.push(json!({"cfg": cname, "start": s, "rows": n_total, "partitions": partitions.len()}));
        }
    }

    let receipt = json!({
        "oracle": "gdn_canon_stream_oracle",
        "arch": gpu.arch,
        "n_key_heads": HK,
        "n_value_heads": hv,
        "seed": format!("{SEED:#x}"),
        "starts": starts,
        "fmts": fmts.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>(),
        "comparisons": rep.comparisons,
        "all_pass": rep.bad.is_empty(),
        "failures": rep.bad,
        "cells": cells,
    });
    let text = serde_json::to_string_pretty(&receipt)?;
    if let Some(p) = json_path {
        std::fs::write(p, &text)?;
    }
    println!(
        "{}",
        serde_json::to_string(&json!({
            "oracle": "gdn_canon_stream_oracle",
            "arch": gpu.arch,
            "hv": hv,
            "comparisons": rep.comparisons,
            "all_pass": receipt["all_pass"],
            "failures": receipt["failures"].as_array().map(|a| a.len()),
        }))?
    );
    if !rep.bad.is_empty() {
        for b in rep.bad.iter().take(30) {
            eprintln!("FAIL {b}");
        }
        return Err(format!("{} failing comparison(s) of {}", rep.bad.len(), rep.comparisons).into());
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("gdn_canon_stream_oracle: {e}");
        std::process::exit(1);
    }
}
