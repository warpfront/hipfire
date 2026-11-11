// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The **current production** GDN prefill scan route on the device as a
//! [`GdnStreamUnderTest`]: `gdn_chunk_prep` → (batched) KKT → `gdn_chunk_scan_*`
//! with the Q8Ef state quantized at every outer call's end and the 512-row
//! commit segmentation restarting at every call's row 0 (this mirrors
//! `hipfire_dispatch::pipeline::batched_deltanet::gdn_chunk_scan_layer`
//! exactly: segment rows = commit stride; batched KKT only when `n > segment`;
//! mseg only on gfx1201 bf16 with 512-row segments).
//!
//! It is the fail-before subject of kills 3 and 4: current production has no
//! carry, so it reports `reports_stream_meta() == false` and its `seal`,
//! `snapshot` and `restore` are the honest operations production has (state is
//! always the call-end-quantized Q8/EF/conv).

use super::q8ef as q8;
use super::stream::*;
use super::util::*;
use rdna_compute::gdn_stream::*;
use rdna_compute::norm::GdnScanOut;
use rdna_compute::{DType, Gpu, GpuTensor};

pub const HK: usize = 16;
pub const EPS: f32 = 1e-6;
pub const PAD_ROWS: usize = 64;
pub const SEG: usize = 512;

pub fn q_scale() -> f32 {
    1.0 / (HD as f32).sqrt()
}

pub fn gpu_geometry(hv: usize) -> StreamGeometry {
    StreamGeometry {
        hk: HK,
        hv,
        micro_rows: 64,
        commit_rows: SEG,
        conv_hist_rows: 3,
        state_quant: GdnStateQuant::Q8Ef,
    }
}

pub fn scan_out_fmt(gpu: &Gpu) -> GdnScanOut {
    if gpu.arch == "gfx1201" {
        GdnScanOut::Bf16
    } else {
        GdnScanOut::F32
    }
}

/// Nonzero incoming Q8 state, scales, EF (F16) and conv history.
pub fn gen_gpu_initial(hv: usize, position: u64) -> InitialState {
    let g = gpu_geometry(hv);
    let codes: Vec<i8> = lcg_vec(10, hv * HD * HD, -64.0, 64.0)
        .iter()
        .map(|x| *x as i8)
        .collect();
    let scales = lcg_vec(11, hv * HD, 0.005, 0.02);
    let ef: Vec<u16> = lcg_vec(12, hv * HD * HD, -0.004, 0.004)
        .iter()
        .map(|x| q8::f32_to_f16_bits(*x))
        .collect();
    InitialState {
        position,
        live: q8::LiveState {
            slot: q8::Q8Slot { codes, scales },
            ef,
        },
        conv: lcg_vec(9, 3 * g.qkv_width(), -0.5, 0.5),
    }
}

pub fn gen_gpu_rows(hv: usize, rows: usize) -> RawRows {
    let g = gpu_geometry(hv);
    RawRows {
        rows,
        qkv: lcg_vec(1, rows * g.qkv_width(), -1.732_050_8, 1.732_050_8),
        alpha: lcg_vec(4, rows * hv, -1.732_050_8, 1.732_050_8),
        beta: lcg_vec(3, rows * hv, -2.0, 2.0),
    }
}

pub struct GpuConsts {
    pub hv: usize,
    pub xdim: usize,
    pub conv_w: GpuTensor,
    pub dt_bias: GpuTensor,
    pub a_log: GpuTensor,
}

impl GpuConsts {
    pub fn new(gpu: &mut Gpu, hv: usize) -> R<Self> {
        let xdim = (2 * HK + hv) * HD;
        let mut up = |d: Vec<f32>| -> R<GpuTensor> { Ok(gpu.upload_f32(&d, &[d.len()])?) };
        Ok(Self {
            hv,
            xdim,
            conv_w: up(lcg_vec(5, 4 * xdim, -0.5, 0.5))?,
            dt_bias: up(lcg_vec(6, hv, -4.0, -2.0))?,
            a_log: up(lcg_vec(7, hv, 0.0, 2.0))?,
        })
    }
    pub fn free(self, gpu: &mut Gpu) -> R<()> {
        for t in [self.conv_w, self.dt_bias, self.a_log] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
}

pub fn download(gpu: &Gpu, t: &GpuTensor) -> R<Vec<u8>> {
    gpu.hip.device_synchronize()?;
    let mut data = vec![0u8; t.byte_size()];
    gpu.hip.memcpy_dtoh(&mut data, &t.buf)?;
    Ok(data)
}

pub fn raw(gpu: &mut Gpu, bytes: usize) -> R<GpuTensor> {
    Ok(gpu.zeros(&[bytes], DType::Raw)?)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pad {
    Zero,
    Nan,
    Huge,
}

impl Pad {
    pub fn byte(self) -> u8 {
        match self {
            Pad::Zero => 0x00,
            Pad::Nan => 0xff,
            Pad::Huge => 0x77,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Pad::Zero => "zero",
            Pad::Nan => "nan",
            Pad::Huge => "huge",
        }
    }
}

/// Persistent device state of one layer: conv ring + Q8 + scales + EF.
pub struct DevState {
    pub hv: usize,
    pub xdim: usize,
    pub conv: GpuTensor,
    pub s_q8: GpuTensor,
    pub s_scales: GpuTensor,
    pub ef: GpuTensor,
}

impl DevState {
    pub fn new(gpu: &mut Gpu, hv: usize) -> R<Self> {
        let xdim = (2 * HK + hv) * HD;
        Ok(Self {
            hv,
            xdim,
            conv: raw(gpu, 3 * xdim * 4)?,
            s_q8: raw(gpu, hv * HD * HD)?,
            s_scales: raw(gpu, hv * HD * 4)?,
            ef: raw(gpu, hv * HD * HD * 2)?,
        })
    }
    pub fn upload_init(&self, gpu: &Gpu, init: &InitialState) -> R<()> {
        let codes: Vec<u8> = init.live.slot.codes.iter().map(|&c| c as u8).collect();
        let ef: Vec<u8> = init.live.ef.iter().flat_map(|e| e.to_le_bytes()).collect();
        gpu.memcpy_htod_auto(&self.conv.buf, &f32_bytes(&init.conv))?;
        gpu.memcpy_htod_auto(&self.s_q8.buf, &codes)?;
        gpu.memcpy_htod_auto(&self.s_scales.buf, &f32_bytes(&init.live.slot.scales))?;
        gpu.memcpy_htod_auto(&self.ef.buf, &ef)?;
        Ok(())
    }
    pub fn read(&self, gpu: &Gpu, position: u64) -> R<SealedState> {
        let q8b = download(gpu, &self.s_q8)?;
        let sc = download(gpu, &self.s_scales)?;
        let ef = download(gpu, &self.ef)?;
        let cv = download(gpu, &self.conv)?;
        Ok(SealedState {
            position,
            codes: q8b.iter().map(|&b| b as i8).collect(),
            scales: sc
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            ef: ef
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect(),
            conv: cv
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        })
    }
    pub fn write(&self, gpu: &Gpu, s: &SealedState) -> R<()> {
        let codes: Vec<u8> = s.codes.iter().map(|&c| c as u8).collect();
        let ef: Vec<u8> = s.ef.iter().flat_map(|e| e.to_le_bytes()).collect();
        gpu.memcpy_htod_auto(&self.conv.buf, &f32_bytes(&s.conv))?;
        gpu.memcpy_htod_auto(&self.s_q8.buf, &codes)?;
        gpu.memcpy_htod_auto(&self.s_scales.buf, &f32_bytes(&s.scales))?;
        gpu.memcpy_htod_auto(&self.ef.buf, &ef)?;
        Ok(())
    }
    pub fn fill(&self, gpu: &Gpu, byte: u8) -> R<()> {
        gpu.hip.device_synchronize()?;
        for t in [&self.conv, &self.s_q8, &self.s_scales, &self.ef] {
            gpu.hip.memset(&t.buf, byte as i32, t.byte_size())?;
        }
        gpu.hip.device_synchronize()?;
        Ok(())
    }
    pub fn free(self, gpu: &mut Gpu) -> R<()> {
        for t in [self.conv, self.s_q8, self.s_scales, self.ef] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
}

/// Scratch of one call over `rows` valid rows with `PAD_ROWS` rows of slack and
/// the A / out planes rounded to whole 64-row blocks. Everything past the valid
/// rows is filled with `pad`.
pub struct Call {
    pub rows: usize,
    pub out_fmt: GdnScanOut,
    x: GpuTensor,
    alpha: GpuTensor,
    beta: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    a: GpuTensor,
    out: GpuTensor,
    hv: usize,
    out_elem: usize,
}

impl Call {
    pub fn new(gpu: &mut Gpu, hv: usize, data: &RawRows, out_fmt: GdnScanOut, pad: Pad) -> R<Self> {
        let rows = data.rows;
        let alloc_rows = rows.div_ceil(64) * 64 + PAD_ROWS;
        let out_elem = if out_fmt == GdnScanOut::Bf16 { 2 } else { 4 };
        let mut fill = |valid: &[u8], total: usize| -> R<GpuTensor> {
            let mut d = vec![pad.byte(); total];
            d[..valid.len()].copy_from_slice(valid);
            let t = raw(gpu, total)?;
            gpu.memcpy_htod_auto(&t.buf, &d)?;
            Ok(t)
        };
        let xdim = (2 * HK + hv) * HD;
        Ok(Self {
            rows,
            out_fmt,
            x: fill(&f32_bytes(&data.qkv), alloc_rows * xdim * 4)?,
            alpha: fill(&f32_bytes(&data.alpha), alloc_rows * hv * 4)?,
            beta: fill(&f32_bytes(&data.beta), alloc_rows * hv * 4)?,
            q: fill(&[], alloc_rows * HK * HD * 2)?,
            k: fill(&[], alloc_rows * HK * HD * 2)?,
            v: fill(&[], alloc_rows * hv * HD * 2)?,
            a: fill(&[], alloc_rows * hv * 64 * 2)?,
            out: fill(&[], alloc_rows * hv * HD * out_elem)?,
            hv,
            out_elem,
        })
    }
    pub fn free(self, gpu: &mut Gpu) -> R<()> {
        for t in [
            self.x, self.alpha, self.beta, self.q, self.k, self.v, self.a, self.out,
        ] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
    pub fn out_row_bytes(&self) -> usize {
        self.hv * HD * self.out_elem
    }
}

/// Route description (for receipts).
#[derive(Clone, Debug)]
pub struct RouteInfo {
    pub rows: usize,
    pub segment_rows: usize,
    pub kkt_batched: bool,
    pub mseg: bool,
    pub segments: usize,
}

/// Production route with the 512-row commit stride (`segment_rows = 512`).
pub fn run_route(gpu: &mut Gpu, consts: &GpuConsts, st: &DevState, c: &Call) -> R<RouteInfo> {
    let n = c.rows;
    let hv = consts.hv;
    gpu.gdn_chunk_prepare(c.out_fmt)?;
    let segment_rows = SEG;
    let kkt_batched = gpu.gdn_kkt_batched_enabled() && segment_rows % 64 == 0 && n > segment_rows;
    gpu.gdn_chunk_prep(
        &c.x,
        &consts.conv_w,
        &st.conv,
        &c.alpha,
        &c.beta,
        &consts.dt_bias,
        &consts.a_log,
        &c.q,
        &c.k,
        &c.v,
        n,
        q_scale(),
        EPS,
        HK,
        hv,
    )?;
    if kkt_batched {
        gpu.gdn_chunk_kkt_solve_batched(&c.k, &c.alpha, &c.beta, &c.a, n, HK, hv)?;
    }
    let mseg = kkt_batched && gpu.gdn_scan_mseg_enabled(c.out_fmt, segment_rows);
    if mseg {
        gpu.gdn_chunk_scan_layer_mseg(
            &c.q,
            &c.k,
            &c.v,
            &c.a,
            &c.alpha,
            &c.beta,
            &st.s_q8,
            &st.s_scales,
            &st.ef,
            &c.out,
            n,
            HK,
            hv,
        )?;
    } else {
        for row0 in (0..n).step_by(segment_rows) {
            let rows = (n - row0).min(segment_rows);
            let a_seg = if kkt_batched {
                c.a.sub_offset(row0 * hv * 64 * 2, rows.div_ceil(64) * 64 * hv * 64 * 2)
            } else {
                c.a.shallow_clone()
            };
            gpu.gdn_chunk_scan_segment(
                &c.q,
                &c.k,
                &c.v,
                &a_seg,
                &c.alpha,
                &c.beta,
                &st.s_q8,
                &st.s_scales,
                &st.ef,
                &c.out,
                c.out_fmt,
                row0,
                rows,
                !kkt_batched,
                HK,
                hv,
            )?;
        }
    }
    gpu.hip.device_synchronize()?;
    Ok(RouteInfo {
        rows: n,
        segment_rows,
        kkt_batched,
        mseg,
        segments: n.div_ceil(segment_rows),
    })
}

pub struct CallOut {
    pub route: RouteInfo,
    pub out_valid: Vec<u8>,
    /// Out rows past the valid rows still hold the poison pattern.
    pub out_pad_untouched: bool,
    /// Valid rows of the prep planes the scan consumes (bf16/f16 operand storage,
    /// `[row][heads*128]`), for localizing a prefix mismatch to prep vs scan.
    pub q_valid: Vec<u8>,
    pub k_valid: Vec<u8>,
    pub v_valid: Vec<u8>,
}

/// Run one production call over `data` (state in `st` is read and overwritten).
pub fn run_call(
    gpu: &mut Gpu,
    consts: &GpuConsts,
    st: &DevState,
    data: &RawRows,
    out_fmt: GdnScanOut,
    pad: Pad,
) -> R<CallOut> {
    let c = Call::new(gpu, consts.hv, data, out_fmt, pad)?;
    let route = run_route(gpu, consts, st, &c)?;
    let out = download(gpu, &c.out)?;
    let ob = c.out_row_bytes();
    let res = CallOut {
        route,
        out_valid: out[..data.rows * ob].to_vec(),
        out_pad_untouched: out[data.rows * ob..].iter().all(|b| *b == pad.byte()),
        q_valid: download(gpu, &c.q)?[..data.rows * HK * HD * 2].to_vec(),
        k_valid: download(gpu, &c.k)?[..data.rows * HK * HD * 2].to_vec(),
        v_valid: download(gpu, &c.v)?[..data.rows * consts.hv * HD * 2].to_vec(),
    };
    c.free(gpu)?;
    Ok(res)
}

pub fn decode_out(bytes: &[u8], fmt: GdnScanOut) -> Vec<f32> {
    match fmt {
        GdnScanOut::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        GdnScanOut::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

// ───────────────────────────── stream adapter ────────────────────────────────

pub struct GpuProductionStream<'a> {
    pub gpu: &'a mut Gpu,
    pub consts: &'a GpuConsts,
    pub st: DevState,
    pub g: StreamGeometry,
    pub out_fmt: GdnScanOut,
    pub position: u64,
    pub routes: Vec<RouteInfo>,
}

impl<'a> GpuProductionStream<'a> {
    pub fn new(gpu: &'a mut Gpu, consts: &'a GpuConsts) -> R<Self> {
        let st = DevState::new(gpu, consts.hv)?;
        let out_fmt = scan_out_fmt(gpu);
        Ok(Self {
            gpu,
            consts,
            st,
            g: gpu_geometry(consts.hv),
            out_fmt,
            position: 0,
            routes: vec![],
        })
    }
    pub fn free(self) -> R<()> {
        let Self { gpu, st, .. } = self;
        st.free(gpu)
    }
}

struct GpuSnap {
    position: u64,
    state: SealedState,
}

impl<'a> GdnStreamUnderTest for GpuProductionStream<'a> {
    fn label(&self) -> String {
        format!("production-call-end-quant[{}]", self.gpu.arch)
    }
    fn geometry(&self) -> StreamGeometry {
        self.g
    }
    fn reports_stream_meta(&self) -> bool {
        false
    }
    fn reset(&mut self, init: &InitialState) -> R<()> {
        self.st.upload_init(self.gpu, init)?;
        self.position = init.position;
        self.routes.clear();
        Ok(())
    }
    fn meta(&self) -> GdnStreamMeta {
        GdnStreamMeta::sealed_at(self.position, 0)
    }
    fn append(&mut self, rows: &RawRows) -> R<Vec<f32>> {
        if rows.rows == 0 {
            return Ok(vec![]);
        }
        let o = run_call(
            self.gpu,
            self.consts,
            &self.st,
            rows,
            self.out_fmt,
            Pad::Zero,
        )?;
        self.position += rows.rows as u64;
        self.routes.push(o.route);
        Ok(decode_out(&o.out_valid, self.out_fmt))
    }
    fn seal(&mut self) -> R<SealedState> {
        // Production has no seal: its state is always the call-end-quantized one.
        self.st.read(self.gpu, self.position)
    }
    fn peek_seal(&mut self) -> R<SealedState> {
        self.st.read(self.gpu, self.position)
    }
    fn snapshot(&mut self) -> R<OpaqueSnapshot> {
        Ok(OpaqueSnapshot(Box::new(GpuSnap {
            position: self.position,
            state: self.st.read(self.gpu, self.position)?,
        })))
    }
    fn restore(&mut self, snap: &OpaqueSnapshot) -> R<()> {
        let s = snap
            .0
            .downcast_ref::<GpuSnap>()
            .ok_or("snapshot type mismatch")?;
        self.st.write(self.gpu, &s.state)?;
        self.position = s.position;
        Ok(())
    }
    fn clobber_live(&mut self) -> R<()> {
        self.st.fill(self.gpu, 0xA5)?;
        self.position = 0xDEAD_BEEF;
        Ok(())
    }
}
