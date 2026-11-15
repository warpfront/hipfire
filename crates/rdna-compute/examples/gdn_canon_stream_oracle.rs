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
//! `--fused-tail` (gfx1201 only; any other arch is a loud nonzero refusal, never a skip) replaces the
//! synthetic raw planes by the REAL fused production path of PLAN-prefill-tail: symmetric MQ4G256V2
//! weights and hidden rows, the independent separate-SET projection of every row as the reference
//! raw plane, and per append either `Gpu::gemm_qkvza_mq4g256v2_iu4_gdn_prepared` +
//! `GdnStreamInput::FusedPrepped` (schedule eligible: aligned start, no head, >= 1 complete tile) or
//! the SET projection + `Raw`. It runs the §9 G2 matrix (every tail r = 1..63 of B in
//! {64,128,512,576,1024}, every split of the final tile, the sequences [B+r,64-r] / [B+r,1,63-r],
//! snapshot/poison/restore continuation, Open pending0 and Sealed off-grid starts), counts the real
//! fused/raw routes per call and fails on an eligible call that did not take the fused route, checks the
//! producer's required raw rows against the independent rows, and runs the §9 negative controls
//! (omitted pending QKV / alpha-beta, wrong halo order, overall-end ring as entry, corrupted saved
//! entry, early gate mutation, sealing each append), each of which must be detected at a named plane.
//! `--graph` composes (captured FusedPrepped append); `--fused-r LO-HI`, `--fused-bases`,
//! `--fused-parts splits,seq,snap,offgrid,neg,reduced` and `--layout slab|token` shard/assert a run;
//! `--json` writes the receipt. The activation layout is per process (`HIPFIRE_A4_SLAB=0` = token).
//!
//! Build: cargo build --release -p rdna-compute --features lab,deltanet --example gdn_canon_stream_oracle
//! Run:   HIPFIRE_GRAPH=0 gdn_canon_stream_oracle --json <receipt.json> [--hv 48|32] [--quick]
//! Exit status 0 only if every comparison is byte identical and non-vacuous.

use rdna_compute::gdn_stream::{
    GdnCarryView, GdnLiveState, GdnPrefillPlan, GdnProjectionRecipe, GdnQ8Planes, GdnScanAlgorithm,
    GdnScanEntry, GdnScanExit, GdnSeqInputs, GdnStateQuant, GdnStreamConsts, GdnStreamMeta,
    GdnStreamPhase, GdnStreamRaw, GdnStreamScratch, GdnTensorFormat,
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
            projection_recipe: GdnProjectionRecipe {
                qkv: 0,
                z: 0,
                alpha: 0,
                beta: 0,
                out: 0,
            },
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
            q80: lcg(10, hv * HD * HD, -64.0, 64.0)
                .iter()
                .map(|x| *x as i8 as u8)
                .collect(),
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
        GdnLiveState {
            planes: self.planes(),
            conv: &self.conv,
        }
    }
    fn planes(&self) -> GdnQ8Planes<'_> {
        GdnQ8Planes {
            codes: &self.codes,
            scales: &self.scales,
            ef: self.ef.as_ref(),
        }
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
        let raw_in = GdnStreamRaw {
            qkv: &x,
            alpha: &a,
            beta: &b,
            input: rdna_compute::gdn_stream::GdnStreamInput::Raw,
        };
        let scratch = GdnStreamScratch {
            q: &q,
            k: &k,
            v: &v,
            a: &aa,
        };
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
                let (first, span) = if what == 1 {
                    (HK * HD * 3, (2 * HD + HD) * 3)
                } else {
                    (0, 512)
                };
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
    Ok(StreamRun {
        out,
        boundaries,
        sealed,
        final_meta,
    })
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
            Some(ef) => GdnScanEntry::Q8Ef {
                codes: &l.codes,
                scales: &l.scales,
                ef,
            },
            None => GdnScanEntry::Q8 {
                codes: &l.codes,
                scales: &l.scales,
            },
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
        gpu.gdn_chunk_prep(
            &x,
            &fx.conv_w,
            &ring,
            &a,
            &b,
            &fx.dt_bias,
            &fx.a_log,
            &q,
            &k,
            &v,
            rows,
            q_scale(),
            EPS,
            HK,
            hv,
        )?;
        let last = t_end == end_pos;
        let seam = t_end % 512 == 0;
        let (planes_ef, planes_c, planes_s) = (live.ef.as_ref(), &live.codes, &live.scales);
        let exit = if last && !complete {
            GdnScanExit::Seal {
                codes: planes_c,
                scales: planes_s,
                ef: planes_ef,
            }
        } else if seam {
            GdnScanExit::Seam {
                codes: planes_c,
                scales: planes_s,
                ef: planes_ef,
                next: &h,
            }
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
            &GdnScanExit::Seal {
                codes: &live.codes,
                scales: &live.scales,
                ef: live.ef.as_ref(),
            },
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
    Ok(TileRef {
        out,
        f32_at,
        planes_at,
        end,
    })
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
    gpu.gdn_chunk_prep(
        &x,
        &fx.conv_w,
        &live.conv,
        &a,
        &b,
        &fx.dt_bias,
        &fx.a_log,
        &q,
        &k,
        &v,
        n,
        q_scale(),
        EPS,
        HK,
        hv,
    )?;
    if kkt_batched {
        gpu.gdn_chunk_kkt_solve_batched(&k, &a, &b, &aa, n, HK, hv)?;
    }
    if kkt_batched && gpu.gdn_scan_mseg_enabled(cfg.fmt, seg) {
        gpu.gdn_chunk_scan_layer_mseg(
            &q,
            &k,
            &v,
            &aa,
            &a,
            &b,
            &live.codes,
            &live.scales,
            ef,
            &o,
            n,
            HK,
            hv,
        )?;
    } else {
        for row0 in (0..n).step_by(seg) {
            let rows = (n - row0).min(seg);
            let a_seg = if kkt_batched {
                aa.sub_offset(row0 * hv * 64 * 2, rows.div_ceil(64) * 64 * hv * 64 * 2)
            } else {
                aa.shallow_clone()
            };
            gpu.gdn_chunk_scan_segment(
                &q,
                &k,
                &v,
                &a_seg,
                &a,
                &b,
                &live.codes,
                &live.scales,
                ef,
                &o,
                cfg.fmt,
                row0,
                rows,
                !kkt_batched,
                HK,
                hv,
            )?;
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
        r.check(
            &c,
            "meta.pending_rows",
            b.meta.pending_rows as usize == pend,
        );
        r.check(
            &c,
            "meta.frame_base",
            b.meta.frame_base == s.max(p / 512 * 512),
        );
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
    r.check(
        ctx,
        "sealed_meta",
        run.final_meta.phase == GdnStreamPhase::Sealed && run.final_meta.pending_rows == 0,
    );
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
        projection_recipe: GdnProjectionRecipe {
            qkv: 0,
            z: 0,
            alpha: 0,
            beta: 0,
            out: 0,
        },
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
        let inp = GdnSeqInputs {
            q: &q,
            k: &k,
            v: &v,
            gate: &g,
            beta: &b,
            qk_head_ratio: 1,
        };
        meta = gpu.gdn_stream_append_sequential(
            &plan,
            &meta,
            s + done as u64,
            n,
            &inp,
            &o,
            &live.planes(),
            &carry,
            hv,
        )?;
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
fn seq_legacy(
    gpu: &mut Gpu,
    fx: &Fixture,
    sf: &SeqFx,
    s: u64,
    n: usize,
) -> Result<(Vec<u8>, StateBytes)> {
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
        gpu.gated_delta_net_q8(
            &q,
            &k,
            &v,
            &g,
            &b,
            &live.codes,
            &live.scales,
            &o,
            len,
            hv,
            HD,
            live.ef.as_ref(),
        )?;
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

// ---------------------------------------------------------------------------------------------
// hipGraph component parity (CHUNK-CANONICAL-DESIGN §4.5): one `gdn_stream_append` captured with
// preloaded modules, stable device pointers and a fixed shape, replayed against the same device
// state, must equal the eager call byte for byte. The executor's kernels take no absolute
// position: the cursor enters only through the schedule class (start modulo 64 / 512, pending
// rows, rows), so a graph captured at one absolute position replays correctly at any position of
// the same class; the host recomputes the candidate meta with `GdnAppendSchedule`. The oracle
// proves that by also replaying the graph (captured at `pos`) against the eager append at
// `pos + 2048` (a different absolute cursor, same class).
// ---------------------------------------------------------------------------------------------

/// Caller buffers of one append (raw rows from fixture row `row0`, poisoned scratch).
struct Bufs {
    row0: usize,
    n: usize,
    x: GpuTensor,
    a: GpuTensor,
    b: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    aa: GpuTensor,
    o: GpuTensor,
}

impl Bufs {
    fn new(gpu: &mut Gpu, fx: &Fixture, cfg: &Cfg, row0: usize, n: usize) -> Result<Self> {
        let alloc = n.div_ceil(64) * 64;
        let hv = fx.hv;
        let bufs = Self {
            row0,
            n,
            x: upload(gpu, &fx.x_bytes(row0, n))?,
            a: upload(gpu, &fx.a_bytes(row0, n))?,
            b: upload(gpu, &fx.b_bytes(row0, n))?,
            q: raw(gpu, alloc * HK * HD * 2)?,
            k: raw(gpu, alloc * HK * HD * 2)?,
            v: raw(gpu, alloc * hv * HD * 2)?,
            aa: raw(gpu, alloc * hv * 64 * 2)?,
            o: raw(gpu, n * hv * HD * cfg.out_elem())?,
        };
        bufs.reset(gpu, fx)?;
        Ok(bufs)
    }
    /// Restore the inputs the executor rewrites in place and re-poison scratch/output.
    fn reset(&self, gpu: &mut Gpu, fx: &Fixture) -> Result<()> {
        gpu.memcpy_htod_auto(&self.x.buf, &fx.x_bytes(self.row0, self.n))?;
        gpu.memcpy_htod_auto(&self.a.buf, &fx.a_bytes(self.row0, self.n))?;
        gpu.memcpy_htod_auto(&self.b.buf, &fx.b_bytes(self.row0, self.n))?;
        for t in [&self.q, &self.k, &self.v, &self.aa, &self.o] {
            fill(gpu, t, 0xff)?;
        }
        Ok(())
    }
    fn free(self, gpu: &mut Gpu) -> Result<()> {
        for t in [
            self.x, self.a, self.b, self.q, self.k, self.v, self.aa, self.o,
        ] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
}

struct GraphRig<'a> {
    fx: &'a Fixture,
    plan: GdnPrefillPlan,
    live: Live,
    carry: Carry,
    ws: rdna_compute::gdn_stream::GdnStreamWorkspace,
}

impl GraphRig<'_> {
    fn append(
        &self,
        gpu: &mut Gpu,
        meta: &GdnStreamMeta,
        pos: u64,
        b: &Bufs,
    ) -> Result<GdnStreamMeta> {
        let raw_in = GdnStreamRaw {
            qkv: &b.x,
            alpha: &b.a,
            beta: &b.b,
            input: rdna_compute::gdn_stream::GdnStreamInput::Raw,
        };
        let scratch = GdnStreamScratch {
            q: &b.q,
            k: &b.k,
            v: &b.v,
            a: &b.aa,
        };
        Ok(gpu.gdn_stream_append(
            &self.plan,
            meta,
            pos,
            b.n,
            &self.fx.consts(),
            &raw_in,
            &scratch,
            &b.o,
            &self.live.view(),
            &self.carry.view(),
            &self.ws,
        )?)
    }
    fn tensors(&self) -> Vec<&GpuTensor> {
        let mut v = vec![&self.live.conv, &self.live.codes, &self.live.scales];
        if let Some(e) = &self.live.ef {
            v.push(e);
        }
        v.extend([
            &self.carry.f32,
            &self.carry.conv,
            &self.carry.qkv,
            &self.carry.alpha,
            &self.carry.beta,
        ]);
        v
    }
    fn snap(&self, gpu: &Gpu) -> Result<Vec<Vec<u8>>> {
        self.tensors()
            .into_iter()
            .map(|t| download(gpu, t))
            .collect()
    }
    fn restore(&self, gpu: &Gpu, snap: &[Vec<u8>]) -> Result<()> {
        for (t, d) in self.tensors().into_iter().zip(snap) {
            gpu.memcpy_htod_auto(&t.buf, d)?;
        }
        Ok(())
    }
}

fn shifted(meta: &GdnStreamMeta, by: u64) -> GdnStreamMeta {
    GdnStreamMeta {
        position: meta.position + by,
        phase_start: meta.phase_start + by,
        frame_base: meta.frame_base + by,
        tile_base: meta.tile_base + by,
        ..*meta
    }
}

fn graph_main(
    gpu: &mut Gpu,
    hv: usize,
    fmts: &[GdnScanOut],
    json_path: Option<String>,
) -> Result<()> {
    let stream = gpu.hip.stream_create()?;
    gpu.active_stream = Some(stream);
    let fx = Fixture::new(gpu, hv, 1300)?;
    let mut rep = Report {
        comparisons: 0,
        bad: Vec::new(),
    };
    let mut cells = Vec::new();
    // (start, rows appended before, rows of the captured append)
    let cases: [(u64, &[usize], usize); 9] = [
        (25, &[10], 300),   // pending head completes, whole tiles, incomplete tail
        (0, &[], 1100),     // aligned multi-frame (mseg on gfx1201 bf16)
        (64, &[63], 1),     // singleton finishing a tile... then pending
        (500, &[], 40),     // clipped head crossing a seam
        (25, &[1], 1),      // singleton inside a tile
        (511, &[], 2),      // seam straddle
        (300, &[100], 700), // off-grid with pending, two seams
        (0, &[], 64),       // exactly one tile, aligned
        (130, &[], 37),     // clipped head, partial
    ];
    for &fmt in fmts {
        for quant in [Quant::Q8Ef, Quant::Q8] {
            let cfg = Cfg { quant, fmt };
            for &(s, pre, n) in &cases {
                let ctx = format!("graph/{}/s{s}/pre{pre:?}/n{n}", cfg.name());
                graph_case(gpu, &fx, &cfg, s, pre, n, &ctx, &mut rep)?;
                cells.push(json!({"cfg": cfg.name(), "start": s, "pre": pre, "rows": n}));
            }
        }
    }
    let receipt = json!({
        "oracle": "gdn_canon_stream_oracle --graph",
        "arch": gpu.arch,
        "n_value_heads": hv,
        "comparisons": rep.comparisons,
        "all_pass": rep.bad.is_empty(),
        "failures": rep.bad,
        "cells": cells,
    });
    if let Some(p) = json_path {
        std::fs::write(p, serde_json::to_string_pretty(&receipt)?)?;
    }
    println!(
        "{}",
        serde_json::to_string(
            &json!({"oracle": "gdn_canon_stream_oracle --graph", "arch": gpu.arch, "hv": hv, "comparisons": rep.comparisons, "all_pass": receipt["all_pass"], "failures": receipt["failures"].as_array().map(|a| a.len())})
        )?
    );
    if !rep.bad.is_empty() {
        for b in rep.bad.iter().take(30) {
            eprintln!("FAIL {b}");
        }
        return Err(format!(
            "{} failing comparison(s) of {}",
            rep.bad.len(),
            rep.comparisons
        )
        .into());
    }
    Ok(())
}

fn graph_case(
    gpu: &mut Gpu,
    fx: &Fixture,
    cfg: &Cfg,
    s: u64,
    pre: &[usize],
    n: usize,
    ctx: &str,
    rep: &mut Report,
) -> Result<()> {
    let hv = fx.hv;
    let rig = GraphRig {
        fx,
        plan: cfg.plan(),
        live: Live::new(gpu, fx, cfg.quant)?,
        carry: Carry::new(gpu, fx)?,
        ws: gpu.gdn_stream_workspace_alloc(HK, hv, cfg.fmt)?,
    };
    let orow = hv * HD * cfg.out_elem();
    // Eager prefix (also loads every module before capture).
    let mut meta = GdnStreamMeta::sealed_at(s, 1);
    let mut done = 0usize;
    for &p in pre {
        let b = Bufs::new(gpu, fx, cfg, done, p)?;
        meta = rig.append(gpu, &meta, s + done as u64, &b)?;
        gpu.hip.device_synchronize()?;
        b.free(gpu)?;
        done += p;
    }
    let pos = s + done as u64;
    let snap0 = rig.snap(gpu)?;
    let bufs = Bufs::new(gpu, fx, cfg, done, n)?;

    // 1. eager reference at `pos`
    let meta_e = rig.append(gpu, &meta, pos, &bufs)?;
    gpu.hip.device_synchronize()?;
    let out_e = download(gpu, &bufs.o)?;
    let st_e = rig.snap(gpu)?;

    // 2. capture the same call (nothing executes during capture)
    rig.restore(gpu, &snap0)?;
    bufs.reset(gpu, fx)?;
    gpu.hip.device_synchronize()?;
    gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
    {
        let st = gpu.active_stream.as_ref().ok_or("no active stream")?;
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, st)?;
    }
    let captured = rig.append(gpu, &meta, pos, &bufs);
    {
        let st = gpu.active_stream.as_ref().ok_or("no active stream")?;
        if captured.is_err() {
            gpu.graphs.abort_graph_capture(&gpu.hip, gpu.device_id, st);
        } else {
            gpu.graphs.end_graph_capture(&gpu.hip, gpu.device_id, st)?;
        }
    }
    let meta_c = captured?;
    rep.check(
        ctx,
        "capture_recorded_kernels",
        !gpu.graphs.ar_forward_blobs.is_empty() && gpu.graphs.graph_exec.is_some(),
    );
    rep.check(ctx, "capture_returns_same_candidate_meta", meta_c == meta_e);
    gpu.hip.device_synchronize()?;
    rep.check(ctx, "capture_executed_nothing", rig.snap(gpu)? == snap0);

    // 3. replay on restored state == eager
    rig.restore(gpu, &snap0)?;
    bufs.reset(gpu, fx)?;
    gpu.hip.device_synchronize()?;
    {
        let st = gpu.active_stream.as_ref().ok_or("no active stream")?;
        gpu.graphs.graph_launch(&gpu.hip, gpu.device_id, st)?;
    }
    gpu.hip.device_synchronize()?;
    let out_r = download(gpu, &bufs.o)?;
    rep.cmp(
        ctx,
        "replay_out_vs_eager",
        &out_r[..n * orow],
        &out_e[..n * orow],
    );
    let st_r = rig.snap(gpu)?;
    for (i, (a, b)) in st_r.iter().zip(&st_e).enumerate() {
        rep.cmp(ctx, &format!("replay_state{i}_vs_eager"), a, b);
    }
    rep.check(
        ctx,
        "nonvacuous_out",
        out_e[..n * orow].iter().any(|b| *b != 0),
    );

    // 4. a second replay from the same state is identical (no state baked into the graph)
    rig.restore(gpu, &snap0)?;
    bufs.reset(gpu, fx)?;
    gpu.hip.device_synchronize()?;
    {
        let st = gpu.active_stream.as_ref().ok_or("no active stream")?;
        gpu.graphs.graph_launch(&gpu.hip, gpu.device_id, st)?;
    }
    gpu.hip.device_synchronize()?;
    rep.cmp(
        ctx,
        "second_replay_out",
        &download(gpu, &bufs.o)?[..n * orow],
        &out_e[..n * orow],
    );

    // 5. a different absolute cursor of the same class: eager there == the graph captured at `pos`
    let by = 2048u64;
    rig.restore(gpu, &snap0)?;
    bufs.reset(gpu, fx)?;
    let meta_shift = if pre.is_empty() {
        GdnStreamMeta::sealed_at(pos + by, 1)
    } else {
        shifted(&meta, by)
    };
    let meta_e2 = rig.append(gpu, &meta_shift, pos + by, &bufs)?;
    gpu.hip.device_synchronize()?;
    rep.cmp(
        ctx,
        "eager_at_shifted_cursor_out",
        &download(gpu, &bufs.o)?[..n * orow],
        &out_e[..n * orow],
    );
    for (i, (a, b)) in rig.snap(gpu)?.iter().zip(&st_e).enumerate() {
        rep.cmp(ctx, &format!("eager_at_shifted_cursor_state{i}"), a, b);
    }
    rep.check(
        ctx,
        "shifted_meta_is_shifted_class",
        meta_e2.pending_rows == meta_e.pending_rows
            && meta_e2.tile_base == meta_e.tile_base + by
            && meta_e2.frame_base == meta_e.frame_base + by,
    );

    gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
    bufs.free(gpu)?;
    let GraphRig {
        live, carry, ws, ..
    } = rig;
    live.free(gpu)?;
    carry.free(gpu)?;
    gpu.gdn_stream_workspace_free(ws)?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// `--fused-tail` (PLAN-prefill-tail S0): the REAL fused producer + the real stream executor.
//
// Every fixture row is a hidden activation row of the 27B geometry (K 5120, symmetric MQ4G256V2
// weights, QKV 10240 / Z 6144 / beta 48 / alpha 48). The independent reference projection of ALL rows
// (separate prepared SETs, one bulk call) becomes the raw plane of the shared `Fixture`, so every
// existing independent arm (`tile_ref`, the Raw `stream_run`, host rings, host pending rows) is the
// reference of the candidate. The candidate arm projects each append on its own:
//   * schedule eligible (`!head && bulk_complete > 0`): `gemm_qkvza_mq4g256v2_iu4_gdn_prepared`
//     (one fused launch, q/k/v F16 in caller scratch, QKV rows only per the export mask) then
//     `gdn_stream_append(.., FusedPrepped)`;
//   * otherwise: the ordinary materializing SET projection and `GdnStreamInput::Raw`.
// The oracle counts the route of every call and FAILS when an eligible call did not take the fused
// route, when its own eligibility rule disagrees with the schedule, or when a requested part ran no
// call. Sparse QKV rows are NaN-poisoned before the producer: the executor cannot read what the
// export mask did not define.
// ---------------------------------------------------------------------------------------------

use rdna_compute::gdn_stream::GdnAppendSchedule;
use rdna_compute::F2GdnTargets;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const FT_K: usize = 5120;
const FT_Z: usize = 6144;
const FT_QKV: usize = 10240;
const FT_AB: usize = 48;
const FT_DIMS: [usize; 4] = [FT_QKV, FT_Z, FT_AB, FT_AB];
const FT_ROWS: usize = 1100;
const FT_GROUP_BYTES: usize = 136;
const FT_SEED: u64 = 0xf00d_7a11_5eed_0001;
const FT_BASES: [usize; 5] = [64, 128, 512, 576, 1024];

/// PLAN-prefill-tail §3.1: must the producer define raw QKV row `t` (call-local) of an `n`-row call?
fn ft_export_raw(t: usize, n: usize) -> bool {
    let t128 = t % 128;
    t < n && (t128 < 3 || t128 >= 125 || t + 3 >= (n & !63))
}

/// Maximal runs `(start, len)` of required raw rows of an `n`-row call.
fn ft_required_ranges(n: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for t in 0..n {
        if !ft_export_raw(t, n) {
            continue;
        }
        match out.last_mut() {
            Some((s, l)) if *s + *l == t => *l += 1,
            _ => out.push((t, 1)),
        }
    }
    out
}

struct W64(u64);
impl W64 {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }
}

/// Random valid SYMMETRIC MQ4G256V2 rows (`zp == -8 * sc` per half, both exact F16); scales are small
/// enough that projected values are O(1), like the lcg raw planes of the other modes.
fn ft_sym_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = W64(seed);
    let mut out = Vec::with_capacity(m * (k / 256) * FT_GROUP_BYTES);
    for _ in 0..m * (k / 256) {
        for _half in 0..2 {
            let scale = 0.0015 + (rng.next() & 0x3f) as f32 * 7.0e-5;
            let sc = half::f16::from_f32(scale);
            let zp = half::f16::from_f32(-8.0 * sc.to_f32());
            out.extend_from_slice(&sc.to_bits().to_le_bytes());
            out.extend_from_slice(&zp.to_bits().to_le_bytes());
        }
        for _ in 0..32 {
            out.extend_from_slice(&(rng.next() as u32).to_le_bytes());
        }
    }
    out
}

fn ft_hidden(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = W64(seed);
    let mut out = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let scale = 0.25 * (1 + r % 5) as f32;
        for _ in 0..k {
            out.push(((rng.next() & 0xff_ffff) as f32 / 8_388_608.0 - 1.0) * scale);
        }
    }
    out
}

/// Weights, norm and hidden rows of the fused-tail fixture.
struct FtRig {
    /// QKV, Z, beta, alpha (separate rows, the independent SET arm).
    w: Vec<GpuTensor>,
    /// The loader's Z fold (`Z`, beta in the next 128-row tile rows 0..47, alpha at rows 64..111).
    z_fold: GpuTensor,
    norm: GpuTensor,
    /// Hidden rows `[FT_ROWS][FT_K]` F32.
    x: GpuTensor,
    /// The rotated-F32 plane argument of the prepared SET consumer (unused on gfx1201 IU4).
    xr: GpuTensor,
    /// Production layer-input AWQ scale with reciprocal planes `[K, AWQ_RCP_ROWS]`. The gfx1201 slab
    /// producer twin (`_v2_slab_fdiv`, the only writer of slab activations, hence the only feeder of
    /// `_b1s`) is selected only with an AWQ tensor; `None` in the token layout (`HIPFIRE_A4_SLAB=0`).
    awq: Option<GpuTensor>,
}

impl FtRig {
    fn new(gpu: &mut Gpu) -> Result<Self> {
        let host: Vec<Vec<u8>> = FT_DIMS
            .iter()
            .enumerate()
            .map(|(i, m)| ft_sym_weights(*m, FT_K, FT_SEED + 0x100 * (i as u64 + 1)))
            .collect();
        let mut w = Vec::new();
        for b in &host {
            w.push(gpu.upload_raw(b, &[b.len()])?);
        }
        let row = FT_K / 256 * FT_GROUP_BYTES;
        let mut f = host[1].clone();
        f.extend_from_slice(&host[2]);
        f.extend(std::iter::repeat(0u8).take((64 - FT_AB) * row));
        f.extend_from_slice(&host[3]);
        f.extend(std::iter::repeat(0u8).take((64 - FT_AB) * row));
        assert_eq!(f.len(), (FT_Z + 128) * row);
        let z_fold = gpu.upload_raw(&f, &[f.len()])?;
        let norm = gpu.upload_f32(&lcg(FT_SEED ^ 2, FT_K, 0.5, 1.5), &[FT_K])?;
        let x = gpu.upload_f32(&ft_hidden(FT_ROWS, FT_K, FT_SEED ^ 3), &[FT_ROWS, FT_K])?;
        let xr = gpu.alloc_tensor(&[FT_ROWS, FT_K], DType::F32)?;
        let awq = if gpu.a4_slab_active() {
            let bytes: Vec<u8> =
                rdna_compute::gemv::awq_rcp_planes(&lcg(FT_SEED ^ 4, FT_K, 0.9, 1.1))
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect();
            Some(gpu.upload_raw(&bytes, &[FT_K, rdna_compute::gemv::AWQ_RCP_ROWS])?)
        } else {
            None
        };
        Ok(Self {
            w,
            z_fold,
            norm,
            x,
            xr,
            awq,
        })
    }
}

fn ft_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The shared `Fixture` with its raw planes replaced by the independent bulk SET projection of every
/// hidden row (one call of `FT_ROWS` rows: the bulk route, no per-append intent).
fn ft_fixture(gpu: &mut Gpu, rig: &FtRig) -> Result<Fixture> {
    let mut fx = Fixture::new(gpu, FT_AB, FT_ROWS)?;
    let mut planes = Vec::new();
    for m in FT_DIMS {
        planes.push(gpu.alloc_tensor(&[FT_ROWS, m], DType::F32)?);
    }
    let prev = gpu.ordinary_prefill_enter();
    let res = gpu.reserve_int4_mmq(FT_K, FT_ROWS)?;
    let prep = gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
        &rig.x,
        &rig.norm,
        rig.awq.as_ref(),
        None,
        res,
        FT_K,
        EPS,
        FT_ROWS,
    )?;
    gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared(
        &rig.w[0], &rig.w[1], &rig.w[2], &rig.w[3], &rig.xr, &prep, &planes[0], &planes[1],
        &planes[2], &planes[3], FT_QKV, FT_Z, FT_AB, FT_AB, FT_K, FT_ROWS,
    )?;
    gpu.ordinary_prefill_exit(prev);
    gpu.hip.device_synchronize()?;
    fx.qkv = ft_f32s(&download(gpu, &planes[0])?);
    fx.beta_raw = ft_f32s(&download(gpu, &planes[2])?);
    fx.alpha_raw = ft_f32s(&download(gpu, &planes[3])?);
    for t in planes {
        gpu.free_tensor(t)?;
    }
    if ![&fx.qkv, &fx.beta_raw, &fx.alpha_raw]
        .iter()
        .all(|v| v.iter().all(|x| x.is_finite()))
    {
        return Err("independent projection produced non-finite rows (fixture unusable)".into());
    }
    let nonzero = fx.qkv.iter().filter(|x| **x != 0.0).count();
    if nonzero < fx.qkv.len() / 2 {
        return Err("independent projection is mostly zero (vacuous fixture)".into());
    }
    Ok(fx)
}

/// Caller buffers of one candidate append, sized for up to `maxn` rows and reused across appends.
struct FtIo {
    qkv: GpuTensor,
    z: GpuTensor,
    beta: GpuTensor,
    alpha: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    aa: GpuTensor,
    o: GpuTensor,
}

impl FtIo {
    fn new(gpu: &mut Gpu, cfg: &Cfg, maxn: usize) -> Result<Self> {
        let alloc = maxn.div_ceil(64) * 64;
        let o_dt = if cfg.out_elem() == 4 {
            DType::F32
        } else {
            DType::F16
        };
        Ok(Self {
            qkv: gpu.alloc_tensor(&[maxn, FT_QKV], DType::F32)?,
            z: gpu.alloc_tensor(&[maxn, FT_Z], DType::F32)?,
            beta: gpu.alloc_tensor(&[maxn, FT_AB], DType::F32)?,
            alpha: gpu.alloc_tensor(&[maxn, FT_AB], DType::F32)?,
            q: raw(gpu, alloc * HK * HD * 2)?,
            k: raw(gpu, alloc * HK * HD * 2)?,
            v: raw(gpu, alloc * FT_AB * HD * 2)?,
            aa: raw(gpu, alloc * FT_AB * 64 * 2)?,
            o: gpu.alloc_tensor(&[maxn, FT_AB * HD], o_dt)?,
        })
    }
    fn free(self, gpu: &mut Gpu) -> Result<()> {
        for t in [
            self.qkv, self.z, self.beta, self.alpha, self.q, self.k, self.v, self.aa, self.o,
        ] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
}

struct FtRoute {
    fused: bool,
    /// Eligible but the fused launcher returned `Ok(false)`.
    declined: bool,
    slab: Option<bool>,
    /// Producer-level coverage failures (required raw rows / raw alpha-beta vs the independent rows).
    bad: Vec<String>,
}

/// Project the `n` hidden rows `[row0, row0 + n)` into `io`: the fused producer when `try_fused`
/// (live ring `live_conv` is its conv input), else the independent materializing SETs. Every plane is
/// NaN-poisoned first. When fused, the required raw rows and all raw alpha/beta are verified against
/// the independent projection before the executor sees them.
#[allow(clippy::too_many_arguments)]
fn ft_produce(
    gpu: &mut Gpu,
    rig: &FtRig,
    fx: &Fixture,
    io: &FtIo,
    live_conv: &GpuTensor,
    row0: usize,
    n: usize,
    try_fused: bool,
) -> Result<FtRoute> {
    for t in [
        &io.qkv, &io.z, &io.beta, &io.alpha, &io.q, &io.k, &io.v, &io.aa,
    ] {
        gpu.hip.memset(&t.buf, 0xFF, t.byte_size())?;
    }
    gpu.hip.device_synchronize()?;
    let x = rig.x.sub_offset(row0 * FT_K, n * FT_K);
    let xr = rig.xr.sub_offset(0, n * FT_K);
    let mut route = FtRoute {
        fused: false,
        declined: false,
        slab: None,
        bad: Vec::new(),
    };
    let prev = gpu.ordinary_prefill_enter();
    let res = gpu.reserve_int4_mmq(FT_K, n)?;
    let prep = gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
        &x,
        &rig.norm,
        rig.awq.as_ref(),
        None,
        res,
        FT_K,
        EPS,
        n,
    )?;
    if try_fused {
        route.slab = Some({
            let p = gpu.int4_mmq_prepared_ptr(&prep, FT_K, n)?;
            gpu.scratch.int4_mmq_slab_at(p)
        });
        let targets = F2GdnTargets {
            conv_weight: &fx.conv_w,
            conv_state: live_conv,
            q: &io.q,
            k: &io.k,
            v: &io.v,
            q_scale: q_scale(),
            eps: EPS,
        };
        let admitted = gpu.gemm_qkvza_mq4g256v2_iu4_gdn_prepared(
            &rig.w[0],
            &rig.z_fold,
            &prep,
            &io.qkv,
            &io.z,
            &io.beta,
            &io.alpha,
            FT_DIMS,
            FT_K,
            n,
            &targets,
        )?;
        route.fused = admitted;
        route.declined = !admitted;
    }
    if !route.fused {
        let v = |t: &GpuTensor, m: usize| t.sub_offset(0, n * m);
        gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared(
            &rig.w[0],
            &rig.w[1],
            &rig.w[2],
            &rig.w[3],
            &xr,
            &prep,
            &v(&io.qkv, FT_QKV),
            &v(&io.z, FT_Z),
            &v(&io.beta, FT_AB),
            &v(&io.alpha, FT_AB),
            FT_QKV,
            FT_Z,
            FT_AB,
            FT_AB,
            FT_K,
            n,
        )?;
    }
    gpu.ordinary_prefill_exit(prev);
    gpu.hip.device_synchronize()?;
    if route.fused {
        route.bad = ft_check_producer(gpu, io, fx, row0, n)?;
    }
    Ok(route)
}

/// Required raw QKV rows (from the frozen formula, never from a sentinel) and every raw alpha/beta
/// row of a fused call, against the independent projection of the same hidden rows.
fn ft_check_producer(
    gpu: &mut Gpu,
    io: &FtIo,
    fx: &Fixture,
    row0: usize,
    n: usize,
) -> Result<Vec<String>> {
    let mut bad = Vec::new();
    for (start, len) in ft_required_ranges(n) {
        let got = gpu.download_raw_bytes(&io.qkv.sub_offset(start * FT_QKV, len * FT_QKV))?;
        if let Some(d) = first_diff(&got, &fx.x_bytes(row0 + start, len)) {
            let row = start + d / 4 / FT_QKV;
            bad.push(format!(
                "producer.qkv_raw: row {row} col {} (required by export_raw at n={n}, rc={}, fixture row {}) differs from the independent SET row",
                d / 4 % FT_QKV,
                n & !63,
                row0 + row
            ));
            break;
        }
    }
    for (name, t, want) in [
        ("producer.alpha_raw", &io.alpha, fx.a_bytes(row0, n)),
        ("producer.beta_raw", &io.beta, fx.b_bytes(row0, n)),
    ] {
        let got = gpu.download_raw_bytes(&t.sub_offset(0, n * FT_AB))?;
        if let Some(d) = first_diff(&got, &want) {
            bad.push(format!(
                "{name}: row {} col {} (n={n}, fixture row {}) differs from the independent SET row",
                d / 4 / FT_AB,
                d / 4 % FT_AB,
                row0 + d / 4 / FT_AB
            ));
        }
    }
    Ok(bad)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fault {
    OmitPendingQkv,
    OmitPendingAb,
    WrongHaloOrder,
    OverallEndRingAsEntry,
    CorruptSavedEntry,
    EarlyGateMutation,
    SealEachAppend,
}

impl Fault {
    const ALL: [Fault; 7] = [
        Fault::OmitPendingQkv,
        Fault::OmitPendingAb,
        Fault::WrongHaloOrder,
        Fault::OverallEndRingAsEntry,
        Fault::CorruptSavedEntry,
        Fault::EarlyGateMutation,
        Fault::SealEachAppend,
    ];
    fn name(self) -> &'static str {
        match self {
            Fault::OmitPendingQkv => "omit_pending_qkv",
            Fault::OmitPendingAb => "omit_pending_alpha_beta",
            Fault::WrongHaloOrder => "wrong_halo_order",
            Fault::OverallEndRingAsEntry => "overall_end_ring_as_entry",
            Fault::CorruptSavedEntry => "corrupt_saved_fp32_entry",
            Fault::EarlyGateMutation => "early_gate_mutation",
            Fault::SealEachAppend => "seal_each_append",
        }
    }
    /// Planes (named failures) that must show the fault.
    fn planes(self) -> &'static [&'static str] {
        match self {
            Fault::OmitPendingQkv => &["pending_qkv"],
            Fault::OmitPendingAb | Fault::EarlyGateMutation => &["pending_alpha", "pending_beta"],
            Fault::WrongHaloOrder | Fault::OverallEndRingAsEntry => &["tile_entry_conv"],
            Fault::CorruptSavedEntry => &["tile_entry_f32"],
            Fault::SealEachAppend => {
                &["meta.phase_open", "meta.pending_rows", "out_rows", "sealed"]
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
struct FtOpts {
    fault: Option<Fault>,
    /// Append index after which the fault is applied (`SealEachAppend`: every append).
    fault_at: usize,
    /// After this append: snapshot live + carry + meta, poison all, re-create the workspace, restore.
    snap_at: Option<usize>,
}

struct FtCall {
    pos: u64,
    rows: usize,
    /// `!head && bulk_complete > 0` from the executor's own schedule.
    eligible: bool,
    /// This oracle's independent rule: the append starts on a C64 boundary and has a complete tile.
    own: bool,
    fused: bool,
    declined: bool,
    slab: Option<bool>,
    head: bool,
    bulk_complete: usize,
    bulk_tail: usize,
}

struct FtOut {
    run: StreamRun,
    calls: Vec<FtCall>,
    producer_bad: Vec<String>,
}

fn ft_apply_fault(
    gpu: &mut Gpu,
    fault: Fault,
    fx: &Fixture,
    live: &Live,
    carry: &Carry,
) -> Result<()> {
    match fault {
        Fault::OmitPendingQkv => fill(gpu, &carry.qkv, 0x00)?,
        Fault::OmitPendingAb => {
            fill(gpu, &carry.alpha, 0x00)?;
            fill(gpu, &carry.beta, 0x00)?;
        }
        Fault::WrongHaloOrder => {
            // Entry ring slots are newest first; swap slot 0 and slot 2 of every channel.
            let mut d = download(gpu, &carry.conv)?;
            for c in 0..fx.xdim {
                for b in 0..4 {
                    d.swap(c * 12 + b, c * 12 + 8 + b);
                }
            }
            gpu.memcpy_htod_auto(&carry.conv.buf, &d)?;
        }
        Fault::OverallEndRingAsEntry => {
            // The ring at the overall append end instead of at the complete-prefix end.
            let d = download(gpu, &live.conv)?;
            gpu.memcpy_htod_auto(&carry.conv.buf, &d)?;
        }
        Fault::CorruptSavedEntry => {
            let mut d = download(gpu, &carry.f32)?;
            for w in 0..512 {
                d[w * 4 + 3] ^= 0x80;
            }
            gpu.memcpy_htod_auto(&carry.f32.buf, &d)?;
        }
        Fault::EarlyGateMutation | Fault::SealEachAppend => {}
    }
    Ok(())
}

/// Run the in-place gate/sigmoid rewrite on the raw tail alpha/beta BEFORE the executor saves them
/// (the bug class "gate mutated before the pending copy").
fn ft_early_gate_mutation(
    gpu: &mut Gpu,
    fx: &Fixture,
    io: &FtIo,
    rc: usize,
    rt: usize,
) -> Result<()> {
    let ring = upload(gpu, &fx.conv0)?;
    let q = raw(gpu, 64 * HK * HD * 2)?;
    let k = raw(gpu, 64 * HK * HD * 2)?;
    let v = raw(gpu, 64 * FT_AB * HD * 2)?;
    let xv = io.qkv.sub_offset(rc * FT_QKV, rt * FT_QKV);
    let av = io.alpha.sub_offset(rc * FT_AB, rt * FT_AB);
    let bv = io.beta.sub_offset(rc * FT_AB, rt * FT_AB);
    gpu.gdn_chunk_prep(
        &xv,
        &fx.conv_w,
        &ring,
        &av,
        &bv,
        &fx.dt_bias,
        &fx.a_log,
        &q,
        &k,
        &v,
        rt,
        q_scale(),
        EPS,
        HK,
        FT_AB,
    )?;
    gpu.hip.device_synchronize()?;
    for t in [ring, q, k, v] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

/// `parts` appended in order from a stream sealed at absolute `s`, every append projected by the real
/// producer of its route (module section docs), then sealed. Boundaries are always collected.
#[allow(clippy::too_many_arguments)]
fn ft_run(
    gpu: &mut Gpu,
    fx: &Fixture,
    rig: &FtRig,
    io: &FtIo,
    cfg: &Cfg,
    s: u64,
    parts: &[usize],
    opts: FtOpts,
) -> Result<FtOut> {
    let plan = cfg.plan();
    let hv = fx.hv;
    let live = Live::new(gpu, fx, cfg.quant)?;
    let carry = Carry::new(gpu, fx)?;
    let mut ws = gpu.gdn_stream_workspace_alloc(HK, hv, cfg.fmt)?;
    let consts = fx.consts();
    let orow = hv * HD * cfg.out_elem();
    let xrow = fx.xdim * 4;
    let mut meta = GdnStreamMeta::sealed_at(s, 1);
    let mut out = Vec::new();
    let mut boundaries = Vec::new();
    let mut calls = Vec::new();
    let mut producer_bad = Vec::new();
    let mut done = 0usize;
    for (i, &n) in parts.iter().enumerate() {
        let pos = s + done as u64;
        let sched = GdnAppendSchedule::new(&meta, pos, n, 64, 512)?;
        let eligible = !sched.head && sched.bulk_complete > 0;
        let own = pos % 64 == 0 && n >= 64;
        let route = ft_produce(gpu, rig, fx, io, &live.conv, done, n, eligible)?;
        for b in &route.bad {
            if producer_bad.len() < 16 {
                producer_bad.push(format!("append{i}@{pos}/{b}"));
            }
        }
        if opts.fault == Some(Fault::EarlyGateMutation) && i == opts.fault_at && sched.bulk_tail > 0
        {
            ft_early_gate_mutation(gpu, fx, io, sched.bulk_complete, sched.bulk_tail)?;
        }
        let raw_in = GdnStreamRaw {
            qkv: &io.qkv,
            alpha: &io.alpha,
            beta: &io.beta,
            input: if route.fused {
                rdna_compute::gdn_stream::GdnStreamInput::FusedPrepped
            } else {
                rdna_compute::gdn_stream::GdnStreamInput::Raw
            },
        };
        let scratch = GdnStreamScratch {
            q: &io.q,
            k: &io.k,
            v: &io.v,
            a: &io.aa,
        };
        meta = gpu.gdn_stream_append(
            &plan,
            &meta,
            pos,
            n,
            &consts,
            &raw_in,
            &scratch,
            &io.o,
            &live.view(),
            &carry.view(),
            &ws,
        )?;
        gpu.hip.device_synchronize()?;
        out.extend_from_slice(&gpu.download_raw_bytes(&io.o.sub_offset(0, n * hv * HD))?);
        calls.push(FtCall {
            pos,
            rows: n,
            eligible,
            own,
            fused: route.fused,
            declined: route.declined,
            slab: route.slab,
            head: sched.head,
            bulk_complete: sched.bulk_complete,
            bulk_tail: sched.bulk_tail,
        });
        done += n;
        if let Some(fault) = opts.fault {
            if fault == Fault::SealEachAppend {
                meta =
                    gpu.gdn_stream_seal(&plan, &meta, &consts, &live.view(), &carry.view(), &ws)?;
            } else if i == opts.fault_at {
                ft_apply_fault(gpu, fault, fx, &live, &carry)?;
            }
        }
        let m = meta.pending_rows as usize;
        boundaries.push(Boundary {
            meta,
            tile_f32: download(gpu, &carry.f32)?,
            tile_conv: download(gpu, &carry.conv)?,
            live_conv: download(gpu, &live.conv)?,
            pend_x: download(gpu, &carry.qkv)?[..m * xrow].to_vec(),
            pend_a: download(gpu, &carry.alpha)?[..m * hv * 4].to_vec(),
            pend_b: download(gpu, &carry.beta)?[..m * hv * 4].to_vec(),
            planes: live.snap(gpu)?,
        });
        if opts.snap_at == Some(i) {
            // Everything a snapshot owns: live ring + Q8/scales/EF, tile-entry FP32 + ring, the valid
            // pending prefix, and the host meta. All device planes are poisoned, the workspace is
            // re-created, then the bytes are restored; unused pending capacity stays poisoned.
            let st = live.snap(gpu)?;
            let pad = |gpu: &Gpu, t: &GpuTensor, valid: usize| -> Result<Vec<u8>> {
                let mut d = download(gpu, t)?;
                for b in &mut d[valid..] {
                    *b = 0xff;
                }
                Ok(d)
            };
            let f32_b = download(gpu, &carry.f32)?;
            let conv_b = download(gpu, &carry.conv)?;
            let qkv_b = pad(gpu, &carry.qkv, m * xrow)?;
            let alpha_b = pad(gpu, &carry.alpha, m * hv * 4)?;
            let beta_b = pad(gpu, &carry.beta, m * hv * 4)?;
            for t in [
                &carry.f32,
                &carry.conv,
                &carry.qkv,
                &carry.alpha,
                &carry.beta,
                &live.conv,
                &live.codes,
                &live.scales,
            ] {
                fill(gpu, t, 0xff)?;
            }
            if let Some(e) = &live.ef {
                fill(gpu, e, 0xff)?;
            }
            gpu.gdn_stream_workspace_free(ws)?;
            ws = gpu.gdn_stream_workspace_alloc(HK, hv, cfg.fmt)?;
            for (t, d) in [
                (&carry.f32, &f32_b),
                (&carry.conv, &conv_b),
                (&carry.qkv, &qkv_b),
                (&carry.alpha, &alpha_b),
                (&carry.beta, &beta_b),
                (&live.conv, &st.conv),
                (&live.codes, &st.codes),
                (&live.scales, &st.scales),
            ] {
                gpu.memcpy_htod_auto(&t.buf, d)?;
            }
            if let (Some(e), Some(d)) = (&live.ef, &st.ef) {
                gpu.memcpy_htod_auto(&e.buf, d)?;
            }
        }
    }
    let final_meta =
        gpu.gdn_stream_seal(&plan, &meta, &consts, &live.view(), &carry.view(), &ws)?;
    let sealed = live.snap(gpu)?;
    live.free(gpu)?;
    carry.free(gpu)?;
    gpu.gdn_stream_workspace_free(ws)?;
    Ok(FtOut {
        run: StreamRun {
            out,
            boundaries,
            sealed,
            final_meta,
        },
        calls,
        producer_bad,
    })
}

impl Report {
    fn fail(&mut self, msg: String) {
        if self.bad.len() < 400 {
            self.bad.push(msg);
        } else if self.bad.len() == 400 {
            self.bad.push("... further failures suppressed".into());
        }
    }
    /// Byte compare naming the first differing row of `unit` (`row_bytes` wide).
    fn cmp_rows(
        &mut self,
        ctx: &str,
        name: &str,
        a: &[u8],
        b: &[u8],
        unit: &str,
        row_bytes: usize,
    ) {
        self.comparisons += 1;
        if let Some(d) = first_diff(a, b) {
            let rb = row_bytes.max(1);
            self.fail(format!(
                "{ctx}/{name}: first diff at {unit} {} (byte {d}, +{} in row; lens {}/{}, hashes {}/{})",
                d / rb,
                d % rb,
                a.len(),
                b.len(),
                fnv(a),
                fnv(b)
            ));
        }
    }
    fn expect(&mut self, ctx: &str, name: &str, ok: bool, why: String) {
        self.comparisons += 1;
        if !ok {
            self.fail(format!("{ctx}/{name}: {why}"));
        }
    }
}

fn ft_cmp_state(r: &mut Report, ctx: &str, a: &StateBytes, b: &StateBytes) {
    r.cmp_rows(ctx, "codes", &a.codes, &b.codes, "state head", HD * HD);
    r.cmp_rows(ctx, "scales", &a.scales, &b.scales, "state head", HD * 4);
    match (&a.ef, &b.ef) {
        (Some(x), Some(y)) => r.cmp_rows(ctx, "ef", x, y, "state head", HD * HD * 2),
        (None, None) => {}
        _ => r.check(ctx, "ef_presence", false),
    }
    r.cmp_rows(ctx, "conv", &a.conv, &b.conv, "channel", 12);
}

/// Every boundary of a candidate run against the independent references (same planes as
/// `check_boundaries`, each failure naming its row/head/channel).
#[allow(clippy::too_many_arguments)]
fn ft_check(
    r: &mut Report,
    ctx: &str,
    fx: &Fixture,
    cfg: &Cfg,
    s: u64,
    parts: &[usize],
    run: &StreamRun,
    tref: &TileRef,
) {
    let hv = fx.hv;
    let orow = hv * HD * cfg.out_elem();
    let total: usize = parts.iter().sum();
    r.cmp_rows(
        ctx,
        "out_rows",
        &run.out,
        &tref.out[..total * orow],
        "output row",
        orow,
    );
    let mut p = s;
    for (i, (&n, b)) in parts.iter().zip(&run.boundaries).enumerate() {
        p += n as u64;
        let c = format!("{ctx}/append{i}@{p}");
        let tb = tile_base(s, p);
        let pend = (p - tb) as usize;
        r.check(&c, "meta.position", b.meta.position == p);
        r.check(&c, "meta.tile_base", b.meta.tile_base == tb);
        r.check(
            &c,
            "meta.pending_rows",
            b.meta.pending_rows as usize == pend,
        );
        r.check(
            &c,
            "meta.frame_base",
            b.meta.frame_base == s.max(p / 512 * 512),
        );
        r.check(&c, "meta.phase_open", b.meta.phase == GdnStreamPhase::Open);
        match tref.f32_at.get(&tb) {
            Some(h) => r.cmp_rows(
                &c,
                "tile_entry_f32",
                &b.tile_f32,
                h,
                "state head",
                HD * HD * 4,
            ),
            None => r.check(&c, "tile_entry_f32_reference_missing", false),
        }
        r.cmp_rows(
            &c,
            "tile_entry_conv",
            &b.tile_conv,
            &fx.ring_at(s, tb),
            "channel",
            12,
        );
        r.cmp_rows(
            &c,
            "live_conv",
            &b.live_conv,
            &fx.ring_at(s, p),
            "channel",
            12,
        );
        let off = (tb - s) as usize;
        r.cmp_rows(
            &c,
            "pending_qkv",
            &b.pend_x,
            &fx.x_bytes(off, pend),
            "pending row",
            fx.xdim * 4,
        );
        r.cmp_rows(
            &c,
            "pending_alpha",
            &b.pend_a,
            &fx.a_bytes(off, pend),
            "pending row",
            hv * 4,
        );
        r.cmp_rows(
            &c,
            "pending_beta",
            &b.pend_b,
            &fx.b_bytes(off, pend),
            "pending row",
            hv * 4,
        );
        if p % 512 == 0 {
            match tref.planes_at.get(&p) {
                Some(pl) => {
                    r.cmp_rows(
                        &c,
                        "seam.codes",
                        &b.planes.codes,
                        &pl.codes,
                        "state head",
                        HD * HD,
                    );
                    r.cmp_rows(
                        &c,
                        "seam.scales",
                        &b.planes.scales,
                        &pl.scales,
                        "state head",
                        HD * 4,
                    );
                    if let (Some(x), Some(y)) = (&b.planes.ef, &pl.ef) {
                        r.cmp_rows(&c, "seam.ef", x, y, "state head", HD * HD * 2);
                    }
                }
                None => r.check(&c, "seam_reference_missing", false),
            }
        }
    }
    r.check(
        ctx,
        "sealed_meta",
        run.final_meta.phase == GdnStreamPhase::Sealed && run.final_meta.pending_rows == 0,
    );
    ft_cmp_state(r, &format!("{ctx}/sealed"), &run.sealed, &tref.end);
}

/// Candidate (producer-routed) run against the existing Raw-input stream run of the same partition:
/// outputs, every boundary plane, the sealed state, metadata.
fn ft_cmp_runs(r: &mut Report, ctx: &str, fx: &Fixture, cfg: &Cfg, a: &StreamRun, b: &StreamRun) {
    let orow = fx.hv * HD * cfg.out_elem();
    r.cmp_rows(ctx, "raw_arm.out_rows", &a.out, &b.out, "output row", orow);
    for (i, (x, y)) in a.boundaries.iter().zip(&b.boundaries).enumerate() {
        let c = format!("{ctx}/raw_arm/append{i}");
        r.check(&c, "meta", x.meta == y.meta);
        r.cmp_rows(
            &c,
            "tile_entry_f32",
            &x.tile_f32,
            &y.tile_f32,
            "state head",
            HD * HD * 4,
        );
        r.cmp_rows(
            &c,
            "tile_entry_conv",
            &x.tile_conv,
            &y.tile_conv,
            "channel",
            12,
        );
        r.cmp_rows(&c, "live_conv", &x.live_conv, &y.live_conv, "channel", 12);
        r.cmp_rows(
            &c,
            "pending_qkv",
            &x.pend_x,
            &y.pend_x,
            "pending row",
            fx.xdim * 4,
        );
        r.cmp_rows(
            &c,
            "pending_alpha",
            &x.pend_a,
            &y.pend_a,
            "pending row",
            fx.hv * 4,
        );
        r.cmp_rows(
            &c,
            "pending_beta",
            &x.pend_b,
            &y.pend_b,
            "pending row",
            fx.hv * 4,
        );
        ft_cmp_state(r, &format!("{c}/planes"), &x.planes, &y.planes);
    }
    r.check(ctx, "raw_arm.final_meta", a.final_meta == b.final_meta);
    ft_cmp_state(r, &format!("{ctx}/raw_arm/sealed"), &a.sealed, &b.sealed);
}

#[derive(Default)]
struct FtStats {
    runs: usize,
    calls: usize,
    fused: usize,
    raw: usize,
    eligible: usize,
    eligible_declined: usize,
    rule_mismatch: usize,
    producer_bad: usize,
    layouts: BTreeSet<&'static str>,
    /// Layout this process must exercise (`_b1s` slab / `_b1` token), from `Gpu::a4_slab_active`.
    want_layout: &'static str,
}

/// Record the activation layout a fused call really used; a call whose layout differs from the one this
/// process must exercise (`a4_slab_active` -> `_b1s`, else `_b1`) is a failure.
fn ft_note_layout(rep: &mut Report, st: &mut FtStats, ctx: &str, slab: Option<bool>) {
    let Some(s) = slab else { return };
    let got = if s { "slab" } else { "token" };
    st.layouts.insert(got);
    rep.expect(
        ctx,
        "activation_layout_matches_process",
        got == st.want_layout,
        format!(
            "fused call used {got} activations (expected {}): the {} launch was not exercised",
            st.want_layout,
            if st.want_layout == "slab" {
                "_b1s"
            } else {
                "_b1"
            }
        ),
    );
}

/// Route accounting of one candidate run: every eligible call must have taken the fused route and
/// the oracle's independent eligibility rule must agree with the executor's schedule.
fn ft_account(rep: &mut Report, st: &mut FtStats, ctx: &str, o: &FtOut) {
    st.runs += 1;
    for (i, c) in o.calls.iter().enumerate() {
        let cc = format!("{ctx}/call{i}@{}", c.pos);
        st.calls += 1;
        if c.own != c.eligible {
            st.rule_mismatch += 1;
        }
        rep.expect(
            &cc,
            "schedule_matches_independent_eligibility_rule",
            c.own == c.eligible,
            format!(
                "schedule eligible={} (head={}, complete rows {}, tail rows {}) but start%64==0 && rows>=64 is {}",
                c.eligible, c.head, c.bulk_complete, c.bulk_tail, c.own
            ),
        );
        if c.eligible {
            st.eligible += 1;
            if c.fused {
                st.fused += 1;
            } else {
                st.eligible_declined += 1;
            }
            rep.expect(
                &cc,
                "eligible_call_took_fused_route",
                c.fused && !c.declined,
                format!(
                    "eligible append of {} rows (complete {}, tail {}) did not take the fused producer (launcher declined: {})",
                    c.rows, c.bulk_complete, c.bulk_tail, c.declined
                ),
            );
        } else {
            st.raw += 1;
            rep.expect(
                &cc,
                "ineligible_call_stayed_raw",
                !c.fused,
                "an ineligible append took the fused route".into(),
            );
        }
        ft_note_layout(rep, st, &cc, c.slab);
    }
    for b in &o.producer_bad {
        st.producer_bad += 1;
        rep.fail(format!("{ctx}/{b}"));
    }
}

/// Splits of the final incomplete tile of `n = base + r` (PLAN §9 G2):
/// `({1..63} ∪ {base-63..n-1} ∪ {64, 128, 511, 512, 513}) ∩ [1, n-1]`.
fn ft_splits(base: usize, r: usize) -> Vec<usize> {
    let n = base + r;
    let mut s: BTreeSet<usize> = (1..=63).collect();
    s.extend(base - 63..n);
    s.extend([64, 128, 511, 512, 513]);
    s.into_iter().filter(|d| *d >= 1 && *d < n).collect()
}

/// Spot subset of [`ft_splits`] for the non-production configurations and `--quick`.
fn ft_splits_reduced(base: usize, r: usize) -> Vec<usize> {
    let n = base + r;
    [1, 2, r, 63, 64, base - 1, base, n - 1, 128, 511, 512, 513]
        .into_iter()
        .collect::<BTreeSet<usize>>()
        .into_iter()
        .filter(|d| *d >= 1 && *d < n)
        .collect()
}

struct FtArgs {
    r_lo: usize,
    r_hi: usize,
    bases: Vec<usize>,
    parts: BTreeSet<String>,
    layout: Option<String>,
}

impl Default for FtArgs {
    fn default() -> Self {
        Self {
            r_lo: 1,
            r_hi: 63,
            bases: FT_BASES.to_vec(),
            parts: ["splits", "seq", "snap", "offgrid", "neg", "reduced"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            layout: None,
        }
    }
}

impl FtArgs {
    fn has(&self, p: &str) -> bool {
        self.parts.contains(p)
    }
}

/// Embedded (or developer-overridden) `_b1` / `_b1s` bundle identity.
fn ft_bundle_info() -> Value {
    let one = |var: &str, embedded: &[u8]| -> Value {
        let (source, path, bytes): (&str, Option<String>, Vec<u8>) =
            match hipfire_config::developer_var(var) {
                Ok(p) => (
                    "override",
                    Some(p.clone()),
                    std::fs::read(&p).unwrap_or_default(),
                ),
                Err(_) => ("embedded", None, embedded.to_vec()),
            };
        json!({"env": var, "source": source, "path": path, "len": bytes.len(),
            "sha256": format!("{:x}", Sha256::digest(&bytes))})
    };
    json!({
        "b1_token": one("HIPFIRE_G12_IU4_B1_BUNDLE", rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1),
        "b1s_slab": one("HIPFIRE_G12_IU4_B1S_BUNDLE", rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S),
        "control_b1_in_use": hipfire_config::developer_bool("HIPFIRE_G12_IU4_B1_CONTROL", false),
    })
}

fn ft_write_receipt(json_path: &Option<String>, receipt: &Value) -> Result<()> {
    if let Some(p) = json_path {
        std::fs::write(p, serde_json::to_string_pretty(receipt)?)?;
    }
    Ok(())
}

/// Entry of `--fused-tail`: loud refusal on anything but the exact fused-producer configuration.
fn ft_entry(
    gpu: &mut Gpu,
    hv: usize,
    graph: bool,
    quick: bool,
    json_path: Option<String>,
    a: &FtArgs,
) -> Result<()> {
    let arch = gpu.arch.clone();
    let mode = if graph {
        "gdn_canon_stream_oracle --fused-tail --graph"
    } else {
        "gdn_canon_stream_oracle --fused-tail"
    };
    let refuse = |why: String| -> Result<()> {
        ft_write_receipt(
            &json_path,
            &json!({"oracle": mode, "arch": arch, "all_pass": false, "unsupported": why}),
        )?;
        Err(format!("--fused-tail is unsupported here: {why}").into())
    };
    if arch != "gfx1201" {
        return refuse(format!(
            "the fused qkvzagdn producer is gfx1201 only (this device is {arch}); run the Raw mode (no --fused-tail) on this arch"
        ));
    }
    if hv != 48 {
        return refuse(format!(
            "the fused producer is the 16/48-head geometry (--hv {hv})"
        ));
    }
    gpu.mq4v2_symmetric = true;
    if !gpu.iu4_gdn_fold_active(FT_DIMS, FT_K) {
        return refuse(
            "Gpu::iu4_gdn_fold_active is false (HIPFIRE_GDN_PREP_FUSED=0, IU4/symfold/B1_CONTROL flags, or not the certified _b1 bundle)"
                .into(),
        );
    }
    if let Some(l) = a.layout.as_deref() {
        let now = if gpu.a4_slab_active() {
            "slab"
        } else {
            "token"
        };
        if l != now {
            return refuse(format!(
                "--layout {l} requested but this process runs the {now} layout (HIPFIRE_A4_SLAB is read once per process)"
            ));
        }
    }
    if graph {
        ft_graph_main(gpu, json_path, a)
    } else {
        ft_main(gpu, quick, json_path, a)
    }
}

fn ft_main(gpu: &mut Gpu, quick: bool, json_path: Option<String>, a: &FtArgs) -> Result<()> {
    let rig = FtRig::new(gpu)?;
    let fx = ft_fixture(gpu, &rig)?;
    let mut rep = Report {
        comparisons: 0,
        bad: Vec::new(),
    };
    let mut st = FtStats {
        want_layout: if gpu.a4_slab_active() {
            "slab"
        } else {
            "token"
        },
        ..FtStats::default()
    };
    let prod = Cfg {
        quant: Quant::Q8Ef,
        fmt: GdnScanOut::Bf16,
    };
    let others = [
        Cfg {
            quant: Quant::Q8,
            fmt: GdnScanOut::Bf16,
        },
        Cfg {
            quant: Quant::Q8Ef,
            fmt: GdnScanOut::F32,
        },
        Cfg {
            quant: Quant::Q8,
            fmt: GdnScanOut::F32,
        },
    ];
    let (r_lo, r_hi) = (a.r_lo, a.r_hi);
    let bases: Vec<usize> = if quick {
        a.bases
            .iter()
            .copied()
            .filter(|b| *b == 64 || *b == 512)
            .collect()
    } else {
        a.bases.clone()
    };
    let r_list = |full: bool| -> Vec<usize> {
        if full && !quick {
            (r_lo..=r_hi).collect()
        } else {
            [1usize, 21, 42, 63]
                .into_iter()
                .filter(|r| (r_lo..=r_hi).contains(r))
                .collect()
        }
    };
    let mut cells: Vec<Value> = Vec::new();
    let mut coverage: Vec<Value> = Vec::new();
    // (base, r) -> (splits executed, splits expected)
    let mut split_cov: std::collections::BTreeMap<(usize, usize), (usize, usize)> =
        Default::default();
    let mut parts_run: BTreeSet<&'static str> = BTreeSet::new();
    let mut controls: Vec<Value> = Vec::new();
    let mut control_failures = 0usize;
    let mut offgrid: Vec<Value> = Vec::new();

    let cfgs: Vec<(Cfg, bool)> = std::iter::once((prod, true))
        .chain(others.iter().map(|c| (*c, false)))
        .collect();
    for (cfg, full) in cfgs {
        let cname = cfg.name();
        if !full && !a.has("reduced") {
            continue;
        }
        let io = FtIo::new(gpu, &cfg, FT_ROWS)?;
        let rr = r_list(full);

        // ---- G2 splits of every tail (and the independent Raw arm on a subset) ----
        if (full && a.has("splits")) || (!full && a.has("reduced")) {
            for &b in &bases {
                for &r in &rr {
                    let n = b + r;
                    let splits = if full && !quick {
                        ft_splits(b, r)
                    } else {
                        ft_splits_reduced(b, r)
                    };
                    let tref = tile_ref(gpu, &fx, &cfg, 0, n)?;
                    let before = (st.fused, st.raw);
                    let ctx0 = format!("{cname}/B{b}/r{r}");
                    // whole append: candidate vs references and vs the Raw-input arm
                    {
                        let ctx = format!("{ctx0}/whole");
                        let o = ft_run(gpu, &fx, &rig, &io, &cfg, 0, &[n], FtOpts::default())?;
                        ft_check(&mut rep, &ctx, &fx, &cfg, 0, &[n], &o.run, &tref);
                        ft_account(&mut rep, &mut st, &ctx, &o);
                        let rw = stream_run(gpu, &fx, &cfg, 0, &[n], true, true, None)?;
                        ft_cmp_runs(&mut rep, &ctx, &fx, &cfg, &o.run, &rw);
                        rep.check(&ctx, "whole_took_fused_route", o.calls[0].fused);
                    }
                    let mut executed = Vec::new();
                    for &d in &splits {
                        let parts = [d, n - d];
                        let ctx = format!("{ctx0}/split{d}");
                        let o = ft_run(gpu, &fx, &rig, &io, &cfg, 0, &parts, FtOpts::default())?;
                        ft_check(&mut rep, &ctx, &fx, &cfg, 0, &parts, &o.run, &tref);
                        ft_account(&mut rep, &mut st, &ctx, &o);
                        if [1, r, 63, 64, 65].contains(&d) || d + 1 == n {
                            let rw = stream_run(gpu, &fx, &cfg, 0, &parts, true, true, None)?;
                            ft_cmp_runs(&mut rep, &ctx, &fx, &cfg, &o.run, &rw);
                        }
                        executed.push(d);
                    }
                    let e = split_cov.entry((b, r)).or_insert((0, 0));
                    e.0 += executed.len();
                    e.1 += splits.len();
                    coverage.push(json!({"cfg": cname, "base": b, "r": r, "n": n,
                        "splits_expected": splits, "splits_executed": executed,
                        "fused_calls": st.fused - before.0, "raw_calls": st.raw - before.1}));
                }
            }
            parts_run.insert(if full { "splits" } else { "reduced" });
        }

        // ---- append sequences [B+r, 64-r], [B+r, 1, 63-r], [B+r, 1, 1, 1, 61-r] ----
        if full && a.has("seq") {
            for &b in &bases {
                let nref = b + 64;
                let tref = tile_ref(gpu, &fx, &cfg, 0, nref)?;
                for &r in &rr {
                    let mut seqs: Vec<Vec<usize>> = vec![vec![b + r, 64 - r]];
                    if r <= 62 {
                        seqs.push(vec![b + r, 1, 63 - r]);
                    }
                    if r <= 60 {
                        seqs.push(vec![b + r, 1, 1, 1, 61 - r]);
                    }
                    for parts in seqs {
                        let ctx = format!("{cname}/seq/B{b}/r{r}/{parts:?}");
                        let o = ft_run(gpu, &fx, &rig, &io, &cfg, 0, &parts, FtOpts::default())?;
                        ft_check(&mut rep, &ctx, &fx, &cfg, 0, &parts, &o.run, &tref);
                        ft_account(&mut rep, &mut st, &ctx, &o);
                        cells.push(json!({"cfg": cname, "phase": "seq", "start": 0, "partitions": parts,
                            "routes": o.calls.iter().map(|c| if c.fused {"fused"} else {"raw"}).collect::<Vec<_>>()}));
                    }
                }
            }
            parts_run.insert("seq");
        }

        // ---- snapshot after the first partial append, poison, restore, continue ----
        if full && a.has("snap") {
            for &b in &bases {
                let nref = b + 64;
                let tref = tile_ref(gpu, &fx, &cfg, 0, nref)?;
                for &r in &rr {
                    let mut cases: Vec<(Vec<usize>, usize)> = vec![(vec![b + r, 64 - r], 0)];
                    if r <= 62 {
                        cases.push((vec![b + r, 1, 63 - r], 0));
                        cases.push((vec![b + r, 1, 63 - r], 1));
                    }
                    for (parts, at) in cases {
                        let ctx = format!("{cname}/snap{at}/B{b}/r{r}/{parts:?}");
                        let opts = FtOpts {
                            snap_at: Some(at),
                            ..FtOpts::default()
                        };
                        let o = ft_run(gpu, &fx, &rig, &io, &cfg, 0, &parts, opts)?;
                        ft_check(&mut rep, &ctx, &fx, &cfg, 0, &parts, &o.run, &tref);
                        ft_account(&mut rep, &mut st, &ctx, &o);
                        cells.push(json!({"cfg": cname, "phase": "snapshot_poison_restore", "snapshot_after_append": at,
                            "start": 0, "partitions": parts}));
                    }
                }
            }
            parts_run.insert("snap");
        }

        // ---- Open pending0 aligned starts (p%128 = 64 / 0) and off-grid sealed starts: matched
        //      mode histories (the reference replays the same phase start) ----
        if full && a.has("offgrid") {
            let n_total = 700usize;
            let starts: Vec<u64> = if quick {
                vec![64, 25]
            } else {
                vec![64, 128, 1, 25, 63, 511, 45979]
            };
            for s in starts {
                let tref = tile_ref(gpu, &fx, &cfg, s, n_total)?;
                let h = (64 - s % 64) as usize;
                let d_seam = (512 - s % 512) as usize;
                let mut parts_list: Vec<Vec<usize>> = vec![vec![n_total]];
                let mut add = |p: Vec<usize>| {
                    if p.iter().all(|x| *x >= 1) && p.iter().sum::<usize>() == n_total {
                        parts_list.push(p);
                    }
                };
                add(vec![h, n_total - h]);
                if h >= 2 {
                    add(vec![h - 1, n_total - h + 1]);
                }
                add(vec![h + 1, n_total - h - 1]);
                add(vec![h, 64 + 21, n_total - h - 85]);
                add(vec![h, 128 + 37, n_total - h - 165]);
                add(vec![h, 85, 1, 63, n_total - h - 149]);
                add(vec![h + 64 + 5, n_total - h - 69]);
                for dd in [d_seam - 1, d_seam, d_seam + 1] {
                    add(vec![dd, n_total - dd]);
                }
                for parts in parts_list {
                    let ctx = format!("{cname}/offgrid/s{s}/{parts:?}");
                    let o = ft_run(gpu, &fx, &rig, &io, &cfg, s, &parts, FtOpts::default())?;
                    ft_check(&mut rep, &ctx, &fx, &cfg, s, &parts, &o.run, &tref);
                    ft_account(&mut rep, &mut st, &ctx, &o);
                    let phase = if s % 64 == 0 {
                        "sealed aligned start (producer tile origin p%128 as listed; later aligned calls Open pending0)"
                    } else {
                        "sealed off-grid start (clipped first tile stays Raw)"
                    };
                    offgrid.push(json!({"cfg": cname, "start": s, "start_mod_128": s % 128, "phase": phase,
                        "partitions": parts,
                        "calls": o.calls.iter().map(|c| json!({"pos": c.pos, "rows": c.rows,
                            "route": if c.fused {"fused"} else {"raw"}, "head": c.head,
                            "complete_rows": c.bulk_complete, "tail_rows": c.bulk_tail})).collect::<Vec<_>>()}));
                }
            }
            parts_run.insert("offgrid");
        }

        // ---- negative controls: each fault must be detected at a named plane/row ----
        if full && a.has("neg") {
            let neg_cases: [(usize, usize); 4] = [(64, 1), (128, 21), (512, 63), (1024, 37)];
            for fault in Fault::ALL {
                for &(b, r) in &neg_cases {
                    let n = b + r;
                    let parts = [n, 64 - r];
                    let nref = b + 64;
                    let tref = tile_ref(gpu, &fx, &cfg, 0, nref)?;
                    let opts = FtOpts {
                        fault: Some(fault),
                        fault_at: 0,
                        snap_at: None,
                    };
                    let o = ft_run(gpu, &fx, &rig, &io, &cfg, 0, &parts, opts)?;
                    let mut crep = Report {
                        comparisons: 0,
                        bad: Vec::new(),
                    };
                    let ctx = format!("{cname}/control/{}/B{b}/r{r}", fault.name());
                    ft_check(&mut crep, &ctx, &fx, &cfg, 0, &parts, &o.run, &tref);
                    let hit = |p: &str| {
                        crep.bad
                            .iter()
                            .any(|m| m.contains(&format!("/{p}:")) || m.contains(&format!("/{p}/")))
                    };
                    let planes: Vec<&str> =
                        fault.planes().iter().copied().filter(|p| hit(*p)).collect();
                    let detected = !planes.is_empty();
                    if !detected {
                        control_failures += 1;
                        rep.fail(format!(
                            "{ctx}: negative control {} was NOT detected at any of {:?} ({} other failures)",
                            fault.name(),
                            fault.planes(),
                            crep.bad.len()
                        ));
                    }
                    controls.push(json!({"control": fault.name(), "base": b, "r": r, "parts": parts,
                        "expected_planes": fault.planes(), "detected": detected, "detected_planes": planes,
                        "first_failure": crep.bad.first(), "failures_total": crep.bad.len(),
                        "routes": o.calls.iter().map(|c| if c.fused {"fused"} else {"raw"}).collect::<Vec<_>>()}));
                }
            }
            parts_run.insert("neg");
        }
        io.free(gpu)?;
    }

    // ---- absent coverage is a failure ----
    for p in a.parts.iter() {
        let want: &'static str = match p.as_str() {
            "splits" => "splits",
            "seq" => "seq",
            "snap" => "snap",
            "offgrid" => "offgrid",
            "neg" => "neg",
            "reduced" => "reduced",
            other => return Err(format!("unknown --fused-parts entry {other}").into()),
        };
        if !parts_run.contains(want) {
            rep.fail(format!("coverage/{want}: requested part ran no case"));
        }
    }
    let full_run = !quick && a.has("splits") && a.r_lo == 1 && a.r_hi == 63 && a.bases == FT_BASES;
    for (&(b, r), &(done, want)) in &split_cov {
        if done != want {
            rep.fail(format!(
                "coverage/splits/B{b}/r{r}: executed {done} splits of {want} required"
            ));
        }
    }
    if a.has("splits") {
        for &b in &bases {
            for r in r_list(true) {
                if !split_cov.contains_key(&(b, r)) {
                    rep.fail(format!("coverage/splits/B{b}/r{r}: tail never ran"));
                }
            }
        }
    }
    if st.fused == 0 {
        rep.fail("coverage/routes: no call took the fused route".into());
    }
    if st.eligible_declined != 0 || st.rule_mismatch != 0 {
        rep.fail(format!(
            "coverage/routes: {} eligible call(s) did not take the fused route, {} eligibility rule disagreement(s)",
            st.eligible_declined, st.rule_mismatch
        ));
    }
    if st.layouts.len() != 1 || st.layouts.iter().next() != Some(&st.want_layout) {
        rep.fail(format!(
            "coverage/layout: expected exactly the {} layout, saw {:?}",
            st.want_layout, st.layouts
        ));
    }
    let layout = st.layouts.iter().next().copied().unwrap_or("none");
    let receipt = json!({
        "oracle": "gdn_canon_stream_oracle --fused-tail",
        "arch": gpu.arch,
        "full_matrix": full_run,
        "quick": quick,
        "geometry": {"n_key_heads": HK, "n_value_heads": FT_AB, "head_dim": HD, "k": FT_K, "qkv": FT_QKV, "z": FT_Z,
            "quant": "MQ4G256V2 symmetric, IU4 A4 activations", "raw_format": "F32 [n][10240]",
            "operand_format": "F16 q/k/v", "scan_output": "Bf16 (production, full matrix) + F32 / plain Q8 (reduced)"},
        "activation_layout": layout,
        "bundles": ft_bundle_info(),
        "seeds": {"weights_hidden": format!("{FT_SEED:#x}"), "fixture": format!("{SEED:#x}"), "rows": FT_ROWS},
        "args": {"r": [a.r_lo, a.r_hi], "bases": a.bases, "parts": a.parts.iter().collect::<Vec<_>>(),
            "layout": a.layout},
        "export_raw": "valid(t) && (t%128 < 3 || t%128 >= 125 || t + 3 >= (n & !63))",
        "routes": {"runs": st.runs, "calls": st.calls, "fused_calls": st.fused, "raw_calls": st.raw,
            "eligible_calls": st.eligible, "eligible_declined": st.eligible_declined,
            "eligibility_rule_disagreements": st.rule_mismatch, "producer_row_failures": st.producer_bad},
        "tails": coverage,
        "split_coverage_counts": split_cov.iter().map(|((b, r), (d, w))| json!({"base": b, "r": r, "executed": d, "required": w})).collect::<Vec<_>>(),
        "cases": cells,
        "offgrid_cases": offgrid,
        "negative_controls": controls,
        "negative_control_failures": control_failures,
        "comparisons": rep.comparisons,
        "all_pass": rep.bad.is_empty(),
        "failures": rep.bad,
    });
    ft_write_receipt(&json_path, &receipt)?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"oracle": "gdn_canon_stream_oracle --fused-tail", "arch": gpu.arch,
            "layout": layout, "full_matrix": full_run, "comparisons": rep.comparisons,
            "fused_calls": st.fused, "raw_calls": st.raw, "eligible_declined": st.eligible_declined,
            "controls_detected": controls.len() - control_failures, "controls": controls.len(),
            "all_pass": receipt["all_pass"], "failures": receipt["failures"].as_array().map(|a| a.len())})
        )?
    );
    if !rep.bad.is_empty() {
        for b in rep.bad.iter().take(30) {
            eprintln!("FAIL {b}");
        }
        return Err(format!(
            "{} failing comparison(s) of {}",
            rep.bad.len(),
            rep.comparisons
        )
        .into());
    }
    Ok(())
}

// `--fused-tail --graph`: one `gdn_stream_append` with FusedPrepped inputs (producer completed
// eagerly before capture), captured with preloaded modules and stable pointers, replayed against the
// same device state; also compared against the Raw-route eager call and against an eager call at a
// different absolute cursor of the same class. Same contract as `graph_case`.

fn ft_g_append(
    gpu: &mut Gpu,
    fx: &Fixture,
    g: &GraphRig,
    io: &FtIo,
    meta: &GdnStreamMeta,
    pos: u64,
    n: usize,
    fused: bool,
) -> Result<GdnStreamMeta> {
    let raw_in = GdnStreamRaw {
        qkv: &io.qkv,
        alpha: &io.alpha,
        beta: &io.beta,
        input: if fused {
            rdna_compute::gdn_stream::GdnStreamInput::FusedPrepped
        } else {
            rdna_compute::gdn_stream::GdnStreamInput::Raw
        },
    };
    let scratch = GdnStreamScratch {
        q: &io.q,
        k: &io.k,
        v: &io.v,
        a: &io.aa,
    };
    Ok(gpu.gdn_stream_append(
        &g.plan,
        meta,
        pos,
        n,
        &fx.consts(),
        &raw_in,
        &scratch,
        &io.o,
        &g.live.view(),
        &g.carry.view(),
        &g.ws,
    )?)
}

fn ft_graph_main(gpu: &mut Gpu, json_path: Option<String>, _a: &FtArgs) -> Result<()> {
    let stream = gpu.hip.stream_create()?;
    gpu.active_stream = Some(stream);
    let rig = FtRig::new(gpu)?;
    let fx = ft_fixture(gpu, &rig)?;
    let mut rep = Report {
        comparisons: 0,
        bad: Vec::new(),
    };
    let mut st = FtStats {
        want_layout: if gpu.a4_slab_active() {
            "slab"
        } else {
            "token"
        },
        ..FtStats::default()
    };
    let mut cells = Vec::new();
    // (start, rows appended before, rows of the captured append, expected to take the fused route)
    let cases: [(u64, &[usize], usize, bool); 14] = [
        (0, &[], 100, true),       // one complete tile + 36-row tail
        (0, &[], 1100, true),      // aligned multi-frame (mseg) + 12-row tail
        (0, &[], 575, true),       // seam-1 prefix + 63-row tail
        (0, &[], 64, true),        // exactly one tile (rt == 0)
        (64, &[], 65, true),       // aligned start with p % 128 == 64
        (0, &[64], 85, true),      // Open pending0 at 64
        (0, &[128], 149, true),    // Open pending0 at 128
        (25, &[39], 130, true),    // sealed off-grid, clipped tile completed eagerly, then fused
        (25, &[10], 300, false),   // pending head: stays Raw
        (64, &[63], 1, false),     // singleton finishing a tile
        (500, &[], 40, false),     // clipped head crossing a seam
        (511, &[], 2, false),      // seam straddle
        (300, &[100], 700, false), // off-grid with pending, two seams
        (130, &[], 37, false),     // clipped head, partial
    ];
    let cfgs = [
        Cfg {
            quant: Quant::Q8Ef,
            fmt: GdnScanOut::Bf16,
        },
        Cfg {
            quant: Quant::Q8,
            fmt: GdnScanOut::Bf16,
        },
    ];
    for cfg in cfgs {
        for &(s, pre, n, expect_fused) in &cases {
            let ctx = format!("graph-fused/{}/s{s}/pre{pre:?}/n{n}", cfg.name());
            let fused = ft_graph_case(
                gpu,
                &fx,
                &rig,
                &cfg,
                s,
                pre,
                n,
                expect_fused,
                &ctx,
                &mut rep,
                &mut st,
            )?;
            cells.push(json!({"cfg": cfg.name(), "start": s, "pre": pre, "rows": n,
                "expected_route": if expect_fused {"fused"} else {"raw"},
                "route": if fused {"fused"} else {"raw"}}));
        }
    }
    if st.fused == 0 {
        rep.fail("coverage/routes: no graph case took the fused route".into());
    }
    if st.eligible_declined != 0 {
        rep.fail(format!(
            "coverage/routes: {} eligible graph call(s) did not take the fused route",
            st.eligible_declined
        ));
    }
    let layout = st.layouts.iter().next().copied().unwrap_or("none");
    let receipt = json!({
        "oracle": "gdn_canon_stream_oracle --fused-tail --graph",
        "arch": gpu.arch,
        "activation_layout": layout,
        "bundles": ft_bundle_info(),
        "seeds": {"weights_hidden": format!("{FT_SEED:#x}"), "fixture": format!("{SEED:#x}"), "rows": FT_ROWS},
        "routes": {"calls": st.calls, "fused_calls": st.fused, "raw_calls": st.raw,
            "eligible_declined": st.eligible_declined, "producer_row_failures": st.producer_bad},
        "comparisons": rep.comparisons,
        "all_pass": rep.bad.is_empty(),
        "failures": rep.bad,
        "cells": cells,
    });
    ft_write_receipt(&json_path, &receipt)?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"oracle": "gdn_canon_stream_oracle --fused-tail --graph", "arch": gpu.arch,
            "layout": layout, "comparisons": rep.comparisons, "fused_calls": st.fused, "raw_calls": st.raw,
            "all_pass": receipt["all_pass"], "failures": receipt["failures"].as_array().map(|a| a.len())})
        )?
    );
    if !rep.bad.is_empty() {
        for b in rep.bad.iter().take(30) {
            eprintln!("FAIL {b}");
        }
        return Err(format!(
            "{} failing comparison(s) of {}",
            rep.bad.len(),
            rep.comparisons
        )
        .into());
    }
    Ok(())
}

/// One graph case; returns whether the captured call took the fused route.
#[allow(clippy::too_many_arguments)]
fn ft_graph_case(
    gpu: &mut Gpu,
    fx: &Fixture,
    rig: &FtRig,
    cfg: &Cfg,
    s: u64,
    pre: &[usize],
    n: usize,
    expect_fused: bool,
    ctx: &str,
    rep: &mut Report,
    st: &mut FtStats,
) -> Result<bool> {
    let hv = fx.hv;
    let g = GraphRig {
        fx,
        plan: cfg.plan(),
        live: Live::new(gpu, fx, cfg.quant)?,
        carry: Carry::new(gpu, fx)?,
        ws: gpu.gdn_stream_workspace_alloc(HK, hv, cfg.fmt)?,
    };
    let maxn = pre.iter().copied().chain([n]).max().unwrap_or(n);
    let io = FtIo::new(gpu, cfg, maxn)?;
    let orow = hv * HD * cfg.out_elem();
    let note = |rep: &mut Report, st: &mut FtStats, route: &FtRoute, eligible: bool, what: &str| {
        st.calls += 1;
        if eligible {
            st.eligible += 1;
            if route.fused {
                st.fused += 1;
            } else {
                st.eligible_declined += 1;
            }
            rep.expect(
                ctx,
                &format!("{what}_eligible_took_fused_route"),
                route.fused,
                "eligible append did not take the fused producer".into(),
            );
        } else {
            st.raw += 1;
        }
        ft_note_layout(rep, st, ctx, route.slab);
        for b in &route.bad {
            st.producer_bad += 1;
            rep.fail(format!("{ctx}/{what}/{b}"));
        }
    };
    // Eager prefix (also loads every module before capture).
    let mut meta = GdnStreamMeta::sealed_at(s, 1);
    let mut done = 0usize;
    for &p in pre {
        let pos = s + done as u64;
        let sch = GdnAppendSchedule::new(&meta, pos, p, 64, 512)?;
        let el = !sch.head && sch.bulk_complete > 0;
        let route = ft_produce(gpu, rig, fx, &io, &g.live.conv, done, p, el)?;
        note(rep, st, &route, el, "prefix");
        meta = ft_g_append(gpu, fx, &g, &io, &meta, pos, p, route.fused)?;
        gpu.hip.device_synchronize()?;
        done += p;
    }
    let pos = s + done as u64;
    let sched = GdnAppendSchedule::new(&meta, pos, n, 64, 512)?;
    let eligible = !sched.head && sched.bulk_complete > 0;
    rep.expect(
        ctx,
        "eligibility_matches_case_table",
        eligible == expect_fused,
        format!("schedule eligible={eligible}, case table says {expect_fused}"),
    );
    let snap0 = g.snap(gpu)?;
    let out_view = |gpu: &Gpu| -> Result<Vec<u8>> {
        gpu.hip.device_synchronize()?;
        Ok(gpu.download_raw_bytes(&io.o.sub_offset(0, n * hv * HD))?)
    };

    // 1. eager reference at `pos` (producer route of the schedule)
    let route = ft_produce(gpu, rig, fx, &io, &g.live.conv, done, n, eligible)?;
    note(rep, st, &route, eligible, "captured_call");
    let fused = route.fused;
    let meta_e = ft_g_append(gpu, fx, &g, &io, &meta, pos, n, fused)?;
    let out_e = out_view(gpu)?;
    let st_e = g.snap(gpu)?;

    // 1b. the Raw-input arm (independent SET projection of the same rows) from the same state
    if fused {
        g.restore(gpu, &snap0)?;
        let rroute = ft_produce(gpu, rig, fx, &io, &g.live.conv, done, n, false)?;
        rep.check(ctx, "raw_arm_took_raw_route", !rroute.fused);
        let meta_r = ft_g_append(gpu, fx, &g, &io, &meta, pos, n, false)?;
        let out_r = out_view(gpu)?;
        rep.check(ctx, "raw_arm_meta_equals_fused_arm", meta_r == meta_e);
        rep.cmp_rows(
            ctx,
            "fused_vs_raw_arm_out",
            &out_e,
            &out_r,
            "output row",
            orow,
        );
        for (i, (a, b)) in g.snap(gpu)?.iter().zip(&st_e).enumerate() {
            rep.cmp(ctx, &format!("fused_vs_raw_arm_state{i}"), a, b);
        }
    }

    // 2. capture the same call (nothing executes during capture); the producer is NOT captured
    g.restore(gpu, &snap0)?;
    ft_produce(gpu, rig, fx, &io, &g.live.conv, done, n, eligible)?;
    gpu.hip.device_synchronize()?;
    gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
    {
        let stm = gpu.active_stream.as_ref().ok_or("no active stream")?;
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, stm)?;
    }
    let captured = ft_g_append(gpu, fx, &g, &io, &meta, pos, n, fused);
    {
        let stm = gpu.active_stream.as_ref().ok_or("no active stream")?;
        if captured.is_err() {
            gpu.graphs.abort_graph_capture(&gpu.hip, gpu.device_id, stm);
        } else {
            gpu.graphs.end_graph_capture(&gpu.hip, gpu.device_id, stm)?;
        }
    }
    let meta_c = captured?;
    rep.check(
        ctx,
        "capture_recorded_kernels",
        !gpu.graphs.ar_forward_blobs.is_empty() && gpu.graphs.graph_exec.is_some(),
    );
    rep.check(ctx, "capture_returns_same_candidate_meta", meta_c == meta_e);
    gpu.hip.device_synchronize()?;
    rep.check(ctx, "capture_executed_nothing", g.snap(gpu)? == snap0);

    // 3. replay on restored state (producer rerun eagerly) == eager
    g.restore(gpu, &snap0)?;
    ft_produce(gpu, rig, fx, &io, &g.live.conv, done, n, eligible)?;
    gpu.hip.device_synchronize()?;
    {
        let stm = gpu.active_stream.as_ref().ok_or("no active stream")?;
        gpu.graphs.graph_launch(&gpu.hip, gpu.device_id, stm)?;
    }
    let out_r = out_view(gpu)?;
    rep.cmp_rows(
        ctx,
        "replay_out_vs_eager",
        &out_r,
        &out_e,
        "output row",
        orow,
    );
    for (i, (a, b)) in g.snap(gpu)?.iter().zip(&st_e).enumerate() {
        rep.cmp(ctx, &format!("replay_state{i}_vs_eager"), a, b);
    }
    rep.check(ctx, "nonvacuous_out", out_e.iter().any(|b| *b != 0));

    // 4. a second replay from the same state is identical
    g.restore(gpu, &snap0)?;
    ft_produce(gpu, rig, fx, &io, &g.live.conv, done, n, eligible)?;
    gpu.hip.device_synchronize()?;
    {
        let stm = gpu.active_stream.as_ref().ok_or("no active stream")?;
        gpu.graphs.graph_launch(&gpu.hip, gpu.device_id, stm)?;
    }
    rep.cmp_rows(
        ctx,
        "second_replay_out",
        &out_view(gpu)?,
        &out_e,
        "output row",
        orow,
    );

    // 5. a different absolute cursor of the same class: eager there == the graph captured at `pos`
    let by = 2048u64;
    g.restore(gpu, &snap0)?;
    let route2 = ft_produce(gpu, rig, fx, &io, &g.live.conv, done, n, eligible)?;
    note(rep, st, &route2, eligible, "shifted_cursor_call");
    let meta_shift = if pre.is_empty() {
        GdnStreamMeta::sealed_at(pos + by, 1)
    } else {
        shifted(&meta, by)
    };
    let meta_e2 = ft_g_append(gpu, fx, &g, &io, &meta_shift, pos + by, n, route2.fused)?;
    rep.cmp_rows(
        ctx,
        "eager_at_shifted_cursor_out",
        &out_view(gpu)?,
        &out_e,
        "output row",
        orow,
    );
    for (i, (a, b)) in g.snap(gpu)?.iter().zip(&st_e).enumerate() {
        rep.cmp(ctx, &format!("eager_at_shifted_cursor_state{i}"), a, b);
    }
    rep.check(
        ctx,
        "shifted_meta_is_shifted_class",
        meta_e2.pending_rows == meta_e.pending_rows
            && meta_e2.tile_base == meta_e.tile_base + by
            && meta_e2.frame_base == meta_e.frame_base + by,
    );

    gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
    io.free(gpu)?;
    let GraphRig {
        live, carry, ws, ..
    } = g;
    live.free(gpu)?;
    carry.free(gpu)?;
    gpu.gdn_stream_workspace_free(ws)?;
    Ok(fused)
}

fn run() -> Result<()> {
    let mut fused_tail = false;
    let mut ft = FtArgs::default();
    let mut ft_flags = false;
    let mut json_path: Option<String> = None;
    let mut hv = 48usize;
    let mut quick = false;
    let mut graph = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--json" => json_path = args.next(),
            "--hv" => hv = args.next().ok_or("--hv value")?.parse()?,
            "--quick" => quick = true,
            "--graph" => graph = true,
            "--fused-tail" => fused_tail = true,
            "--fused-r" => {
                let v = args.next().ok_or("--fused-r LO-HI")?;
                let (lo, hi) = v.split_once('-').ok_or("--fused-r LO-HI")?;
                ft.r_lo = lo.parse()?;
                ft.r_hi = hi.parse()?;
                if !(1..=63).contains(&ft.r_lo) || !(ft.r_lo..=63).contains(&ft.r_hi) {
                    return Err("--fused-r must satisfy 1 <= LO <= HI <= 63".into());
                }
                ft_flags = true;
            }
            "--fused-bases" => {
                let v = args.next().ok_or("--fused-bases B[,B..]")?;
                ft.bases = v
                    .split(',')
                    .map(|b| b.parse::<usize>())
                    .collect::<std::result::Result<_, _>>()?;
                if ft.bases.is_empty() || ft.bases.iter().any(|b| !FT_BASES.contains(b)) {
                    return Err(format!("--fused-bases must be a subset of {FT_BASES:?}").into());
                }
                ft_flags = true;
            }
            "--fused-parts" => {
                let v = args.next().ok_or("--fused-parts P[,P..]")?;
                ft.parts = v.split(',').map(|p| p.to_string()).collect();
                ft_flags = true;
            }
            "--layout" => {
                let v = args.next().ok_or("--layout slab|token")?;
                if !matches!(v.as_str(), "slab" | "token") {
                    return Err("--layout must be slab or token".into());
                }
                ft.layout = Some(v);
                ft_flags = true;
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if ft_flags && !fused_tail {
        return Err("--fused-r/--fused-bases/--fused-parts/--layout need --fused-tail".into());
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
    if fused_tail {
        return ft_entry(&mut gpu, hv, graph, quick, json_path, &ft);
    }
    let fmts: Vec<GdnScanOut> = if gpu.arch == "gfx1201" {
        vec![GdnScanOut::Bf16, GdnScanOut::F32]
    } else {
        vec![GdnScanOut::F32]
    };
    if graph {
        return graph_main(&mut gpu, hv, &fmts, json_path);
    }
    let starts: Vec<u64> = if quick {
        vec![0, 25, 512]
    } else {
        vec![0, 1, 25, 63, 64, 65, 511, 512, 513, 521, 45979]
    };
    let total_rows = 1100usize;
    let fx = Fixture::new(&mut gpu, hv, total_rows)?;
    let mut rng = Rng(SEED);
    let mut rep = Report {
        comparisons: 0,
        bad: Vec::new(),
    };
    let mut cells = Vec::new();

    for &fmt in &fmts {
        for quant in [Quant::Q8Ef, Quant::Q8] {
            let cfg = Cfg { quant, fmt };
            let cname = cfg.name();

            // ---- legacy (incumbent whole call) == tiles == whole, aligned start 0 ----
            let legacy_ns: Vec<usize> = if quick {
                vec![1, 25, 63, 64, 65, 512, 575]
            } else {
                vec![
                    1, 2, 3, 17, 25, 31, 32, 33, 62, 63, 64, 65, 100, 127, 128, 129, 300, 511, 512,
                    513, 575, 1024, 1100,
                ]
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
                let mut partitions: Vec<(String, Vec<usize>)> =
                    vec![("whole".into(), vec![n_total])];
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
                    (1..=130)
                        .chain([191, 192, 193, 255, 256, 257, 447, 448, 449, 511, 512, 513])
                        .collect()
                } else {
                    vec![1, 2, 3, 31, 32, 33, 62, 63, 64, 65, 127, 128, 129]
                };
                for &pfx in &sweep {
                    if pfx > n_total {
                        continue;
                    }
                    let ctx = format!("{ctx0}/prefix{pfx}");
                    let run = stream_run(&mut gpu, &fx, &cfg, s, &[pfx], false, false, None)?;
                    rep.cmp(
                        &ctx,
                        "out_prefix_vs_whole",
                        &run.out,
                        &tref.out[..pfx * orow],
                    );
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
                    let run =
                        stream_run(&mut gpu, &fx, &cfg, s, &parts, false, true, Some((0, what)))?;
                    let ctx = format!("{cname}/control-{label}");
                    rep.check(
                        &ctx,
                        "corruption_detected",
                        run.out != tref.out[..n_total * orow],
                    );
                    // The 25..35 rows were published before the corruption and must be unchanged.
                    rep.cmp(
                        &ctx,
                        "pre_corruption_rows_unchanged",
                        &run.out[..10 * orow],
                        &tref.out[..10 * orow],
                    );
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
            let (whole_out, whole_state) =
                seq_run(&mut gpu, &fx, &sf, quant, s, &[n_total], true, None)?;
            rep.check(&ctx0, "nonvacuous_out", whole_out.iter().any(|b| *b != 0));
            rep.check(&ctx0, "state_moved", whole_state.codes != fx.q80);
            if quant == Quant::Q8Ef {
                // Incumbent route: one fast-kernel launch per absolute 512-row segment, Q8+EF
                // round trip between them and at the call end.
                let (lo, ls) = seq_legacy(&mut gpu, &fx, &sf, s, n_total)?;
                rep.cmp(&ctx0, "whole_vs_legacy.out", &whole_out, &lo);
                cmp_state(
                    &mut rep,
                    &format!("{ctx0}/whole_vs_legacy"),
                    &whole_state,
                    &ls,
                );
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
            let (co, _) = seq_run(
                &mut gpu,
                &fx,
                &sf,
                quant,
                s,
                &[10, n_total - 10],
                true,
                Some(0),
            )?;
            let orow = hv * HD * 4;
            rep.check(&ctx0, "control_corruption_detected", co != whole_out);
            rep.cmp(
                &ctx0,
                "control_pre_rows_unchanged",
                &co[..10 * orow],
                &whole_out[..10 * orow],
            );
            cells.push(
                json!({"cfg": cname, "start": s, "rows": n_total, "partitions": partitions.len()}),
            );
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
        return Err(format!(
            "{} failing comparison(s) of {}",
            rep.bad.len(),
            rep.comparisons
        )
        .into());
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("gdn_canon_stream_oracle: {e}");
        std::process::exit(1);
    }
}
