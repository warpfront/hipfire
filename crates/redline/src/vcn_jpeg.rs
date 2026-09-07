// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! VCN JPEG direct-ring decode (experiment `experiment/vcn-ring`).
//!
//! Programs the VCN JPEG engine with raw register packets on the
//! `AMDGPU_HW_IP_VCN_JPEG` ring, bypassing firmware session setup entirely:
//! Mesa creates firmware messages only in the non-JPEG branch
//! (`radeon_vcn_dec.c:3016-3023`); JPEG goes directly to register
//! programming. The JPEG stream itself is parsed with
//! [`va_bridge::parse_for_va`] and re-serialized into the canonical
//! header+entropy+EOI form Mesa feeds the engine — the parser is reused,
//! never duplicated, and libva is never touched here.
//!
//! Register values below are transcribed from the vendored Mesa 26.0.0
//! (`c10cba7e`) sources under `experiments/vcn-jpeg/upstream/mesa-c10cba7e/`.
//! Citations name those bytes. Two upstream paths are cited without a vendored
//! copy: the `{0x60000000, 0}` NOP pad
//! (`src/gallium/winsys/amdgpu/drm/amdgpu_cs.cpp:2211-2217`) and the 16-dword
//! JPEG pad mask (`src/gallium/drivers/radeonsi/ac_gpu_info.c:444-467`).
//!
//! Supported input: sequential baseline (SOF0), 8-bit, 3-component 4:2:0
//! (NV12 output) or 4:4:4 (planar 444 output). Everything else — VCN1,
//! unknown VCN revisions, 4:2:2, grayscale, progressive/lossless/arithmetic
//! JPEG, crop, packed-RGB conversion — fails closed with `Unsupported`.

use crate::device::{Device, GpuBuffer};
use crate::drm::AMDGPU_HW_IP_VCN_JPEG;
use crate::queue::{ComputeQueue, Engine, SubmissionFence, SubmitSync, SyncObj};
use crate::{RedlineError, Result};
use va_bridge::{parse_for_va, JpegVaParams};

// ---------------------------------------------------------------------------
// Frozen interface
// ---------------------------------------------------------------------------

/// Native output format of a VCN JPEG decode.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum JpegNativeFormat {
    Nv12,
    Yuv444p,
}

/// Linear surface geometry backing a decode.
///
/// `pitch` is `align(width, 256)`; the coded height `align(height, 16)` is
/// implicit in the offsets. `plane_offsets` are byte offsets into the
/// surface BO (all 256-byte aligned, all fit `u32`); `plane_rows` are the
/// valid rows per plane (`[h, ceil(h/2), 0]` for NV12, `[h, h, h]` for 444P).
/// Only valid rows are defined — padding bytes are whatever the engine left.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JpegSurfaceLayout {
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub plane_offsets: [u32; 3],
    pub plane_rows: [u32; 3],
    pub plane_count: u8,
    pub format: JpegNativeFormat,
}

// ---------------------------------------------------------------------------
// Register tables (audited transcription)
// ---------------------------------------------------------------------------
//
// Packet word: `(reg & 0x3ffff) | ((cond & 0xf) << 24) | ((type & 0xf) << 28)`
// (ac_vcn_dec.h:27-33); each register write is emitted as a two-dword
// (word, value) pair (radeon_vcn_dec_jpeg.c:107-112, `set_reg_jpeg`).
// Completion checks use COND3/TYPE3 exactly as Mesa does; the pair is kept
// opaque rather than assigned undocumented semantics.
//
// Register inventory: radeon_vcn_dec.h:32-64 (`struct jpeg_registers`,
// RDECODE_JPEG_REG_VER_V1/V2/V3). VCN-version to register-version selection:
// radeon_vcn_dec.c:2954-3014; concrete V2/V3 register assignment: 3023-3067.
// Numeric addresses: ac_vcn_dec.h:214-258.

const COND0: u32 = 0;
const COND3: u32 = 3;
const TYPE0: u32 = 0;
const TYPE3: u32 = 3;

/// JRBC/programming registers shared by V2 and V3 (ac_vcn_dec.h:215-216,
/// 219-221, 234-236).
const JRBC_COND: u32 = 0x408e;
const JRBC_REF: u32 = 0x408f;
const RB_BASE: u32 = 0x4001;
const RB_SIZE: u32 = 0x4004;
const RB_WPTR: u32 = 0x4002;
const INT_EN: u32 = 0x400a;
const CNTL: u32 = 0x4000;
const RB_RPTR: u32 = 0x4003;

/// V3-only addresses (ac_vcn_dec.h:238-258).
const ROI_START: u32 = 0x401b;
const ROI_STRIDE: u32 = 0x401c;
const FC_SPS: u32 = 0x4052;

/// FC_SPS_INFO for native (non-format-convert) output:
/// `1 | (1 << 5) | (255 << 8)` (radeon_vcn_dec_jpeg.c:369).
const FC_SPS_NATIVE: u32 = 1 | (1 << 5) | (255 << 8);
/// OUTBUF_CNTL = `(0x1587 & ~0x180) | (1 << 7) | (1 << 6)` = `0x14c7`
/// (radeon_vcn_dec_jpeg.c:375-376).
const OUTBUF_CNTL_VAL: u32 = 0x14c7;
/// JRBC timer magic programmed around the reset and the polls
/// (radeon_vcn_dec_jpeg.c:249, 393).
const JRBC_TIMER: u32 = 0x01400200;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RegVer {
    V2,
    V3,
}

/// Per-generation register addresses (`struct jpeg_registers` fields).
#[derive(Copy, Clone, Debug)]
struct RegSet {
    ver: RegVer,
    soft_rst: u32,
    read_hi: u32,
    read_lo: u32,
    pitch: u32,
    uv_pitch: u32,
    addr_mode: u32,
    y_tile: u32,
    uv_tile: u32,
    write_hi: u32,
    write_lo: u32,
    tier: u32,
    outbuf_rptr: u32,
    outbuf_cntl: u32,
    outbuf_wptr: u32,
    // V2 plane programming goes through INDEX/DATA; V3 writes the base
    // registers directly (radeon_vcn_dec_jpeg.c:327-339).
    index: u32,
    data: u32,
    luma: u32,
    chroma: u32,
    chromav: u32,
}

impl RegSet {
    /// V2: VCN 2.0/2.2/2.5/2.6, 3.0/3.1, 4.0.0/2/4/5/6
    /// (radeon_vcn_dec.c:2963-2986, 2992-3000, 3032-3048;
    /// ac_vcn_dec.h:214, 217-218, 222-237).
    const V2: Self = Self {
        ver: RegVer::V2,
        soft_rst: 0x402f,
        read_hi: 0x40e1,
        read_lo: 0x40e0,
        pitch: 0x401f,
        uv_pitch: 0x4020,
        addr_mode: 0x4027,
        y_tile: 0x4024,
        uv_tile: 0x4025,
        write_hi: 0x40e3,
        write_lo: 0x40e2,
        tier: 0x400f,
        outbuf_rptr: 0x401e,
        outbuf_cntl: 0x401c,
        outbuf_wptr: 0x401d,
        index: 0x402c,
        data: 0x402d,
        luma: 0,
        chroma: 0,
        chromav: 0,
    };
    /// V3: VCN 4.0.3 and VCN 5.0.0/5.0.1
    /// (radeon_vcn_dec.c:2987-2991, 3001-3010, 3049-3066;
    /// ac_vcn_dec.h:238-254).
    const V3: Self = Self {
        ver: RegVer::V3,
        soft_rst: 0x4051,
        read_hi: 0x40b3,
        read_lo: 0x40b2,
        pitch: 0x4043,
        uv_pitch: 0x4044,
        addr_mode: 0x404b,
        y_tile: 0x4048,
        uv_tile: 0x4049,
        write_hi: 0x40b5,
        write_lo: 0x40b4,
        tier: 0x400e,
        outbuf_rptr: 0x4042,
        outbuf_cntl: 0x4040,
        outbuf_wptr: 0x4041,
        index: 0,
        data: 0,
        luma: 0x41c0,
        chroma: 0x41c1,
        chromav: 0x41c2,
    };
}

/// Select the register set from the queried VCN IP discovery version
/// (`(v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff` = major, minor, revision).
/// VCN1 and unknown revisions return `None`: fail closed, never guess
/// (radeon_vcn_dec.c:2954-3014; VCN1 is an explicit non-goal for this lane).
fn select_reg_set(discovery: u32) -> Option<RegSet> {
    match discovery {
        0x050000 | 0x050001 | 0x040003 => Some(RegSet::V3),
        0x040000 | 0x040002 | 0x040004 | 0x040005 | 0x040006 | 0x030101 | 0x030102 | 0x030000
        | 0x030002 | 0x030010 | 0x030021 | 0x020500 | 0x020600 | 0x020000 | 0x020002 | 0x020003
        | 0x020200 => Some(RegSet::V2),
        _ => None,
    }
}

fn pkt(reg: u32, cond: u32, typ: u32, val: u32, out: &mut Vec<u32>) {
    out.push((reg & 0x3FFFF) | ((cond & 0xF) << 24) | ((typ & 0xF) << 28));
    out.push(val);
}

// ---------------------------------------------------------------------------
// Layout (unproven until the canary + five-fixture gates pass; plan §layout)
// ---------------------------------------------------------------------------

const PITCH_ALIGN: u32 = 256;
const CODED_H_ALIGN: u32 = 16;
/// Canary guard band before each plane and after the BO.
const CANARY_LEN: u64 = 256;

fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) & !(a - 1)
}

/// Native linear layout: pitch = align(w, 256), coded_h = align(h, 16);
/// NV12 chroma at pitch*coded_h; 444P Cb/Cr each pitch*coded_h. All offsets
/// 256-byte aligned, fit u32.
fn surface_layout(width: u32, height: u32, format: JpegNativeFormat) -> (JpegSurfaceLayout, u64) {
    let pitch = align_up(width as u64, PITCH_ALIGN as u64) as u32;
    let coded_h = align_up(height as u64, CODED_H_ALIGN as u64);
    let luma_size = pitch as u64 * coded_h;
    match format {
        JpegNativeFormat::Nv12 => {
            let chroma_size = luma_size / 2;
            let luma_off = CANARY_LEN;
            let chroma_off = luma_off + luma_size + CANARY_LEN;
            let total = chroma_off + chroma_size + CANARY_LEN;
            (
                JpegSurfaceLayout {
                    width,
                    height,
                    pitch,
                    plane_offsets: [luma_off as u32, chroma_off as u32, 0],
                    plane_rows: [height, (height + 1) / 2, 0],
                    plane_count: 2,
                    format,
                },
                total,
            )
        }
        JpegNativeFormat::Yuv444p => {
            let cb_off = CANARY_LEN + luma_size + CANARY_LEN;
            let cr_off = cb_off + luma_size + CANARY_LEN;
            let total = cr_off + luma_size + CANARY_LEN;
            (
                JpegSurfaceLayout {
                    width,
                    height,
                    pitch,
                    plane_offsets: [CANARY_LEN as u32, cb_off as u32, cr_off as u32],
                    plane_rows: [height, height, height],
                    plane_count: 3,
                    format,
                },
                total,
            )
        }
    }
}

/// `(offset, len, pattern)` for every canary band of a layout.
fn canary_ranges(layout: &JpegSurfaceLayout) -> Vec<(u64, u64, u8)> {
    let pitch = layout.pitch as u64;
    let coded_h = align_up(layout.height as u64, CODED_H_ALIGN as u64);
    let luma_size = pitch * coded_h;
    let mut out = Vec::new();
    out.push((0, CANARY_LEN, 0xC0));
    match layout.format {
        JpegNativeFormat::Nv12 => {
            let chroma_size = luma_size / 2;
            let luma_off = layout.plane_offsets[0] as u64;
            let chroma_off = layout.plane_offsets[1] as u64;
            out.push((luma_off + luma_size, CANARY_LEN, 0xC1));
            out.push((chroma_off + chroma_size, CANARY_LEN, 0xC2));
        }
        JpegNativeFormat::Yuv444p => {
            let cb_off = layout.plane_offsets[1] as u64;
            let cr_off = layout.plane_offsets[2] as u64;
            out.push((cb_off - CANARY_LEN, CANARY_LEN, 0xC1));
            out.push((cr_off - CANARY_LEN, CANARY_LEN, 0xC2));
            out.push((cr_off + luma_size, CANARY_LEN, 0xC3));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Canonical stream serialization
// ---------------------------------------------------------------------------

/// Classify the parsed stream into a native output format. 4:2:2,
/// grayscale, and anything the parser already rejected fail closed here.
fn classify(p: &JpegVaParams) -> Result<JpegNativeFormat> {
    let unsupported = |why: &str| RedlineError {
        code: -1,
        message: format!("vcn-jpeg: unsupported stream ({why})"),
    };
    if p.pic.num_components != 3 {
        return Err(unsupported("not 3-component"));
    }
    let c = |i: usize| {
        (
            p.pic.components[i].h_sampling_factor,
            p.pic.components[i].v_sampling_factor,
        )
    };
    match (p.max_h, p.max_v, c(0), c(1), c(2)) {
        (2, 2, (2, 2), (1, 1), (1, 1)) => Ok(JpegNativeFormat::Nv12),
        (1, 1, (1, 1), (1, 1), (1, 1)) => Ok(JpegNativeFormat::Yuv444p),
        _ => Err(unsupported("sampling factors are not 4:2:0 or 4:4:4")),
    }
}

/// Re-serialize the parsed tables/parameters into the canonical
/// SOI/DQT/DHT/[DRI]/SOF/SOS header Mesa builds in
/// `vlVaGetJpegSliceHeader` (picture_mjpeg.c:140-279), then append the
/// entropy bytes and EOI exactly as the VA frontend assembles the bitstream
/// BO (decode.c:245-265: slice header + slice data + `eoi_jpeg`).
fn canonical_stream(p: &JpegVaParams) -> Vec<u8> {
    let mut out = Vec::new();
    // SOI (picture_mjpeg.c:146-148).
    out.extend_from_slice(&[0xFF, 0xD8]);
    // DQT: one 65-byte table per loaded index, Pq = 0 always
    // (picture_mjpeg.c:150-168).
    out.extend_from_slice(&[0xFF, 0xDB]);
    let len_pos = out.len();
    out.extend_from_slice(&[0, 0]);
    for i in 0..4 {
        if p.iq.load_quantiser_table[i] == 0 {
            continue;
        }
        out.push(i as u8);
        out.extend_from_slice(&p.iq.quantiser_table[i]);
    }
    let len = (out.len() - len_pos) as u16;
    out[len_pos..len_pos + 2].copy_from_slice(&len.to_be_bytes());
    // DHT: DC tables then AC tables, values truncated to the bit counts
    // (picture_mjpeg.c:171-211).
    out.extend_from_slice(&[0xFF, 0xC4]);
    let len_pos = out.len();
    out.extend_from_slice(&[0, 0]);
    for i in 0..2 {
        if p.huff.load_huffman_table[i] == 0 {
            continue;
        }
        let t = &p.huff.huffman_table[i];
        let ndc: usize = t.num_dc_codes.iter().map(|&b| b as usize).sum();
        out.push(0x00 | i as u8);
        out.extend_from_slice(&t.num_dc_codes);
        out.extend_from_slice(&t.dc_values[..ndc]);
    }
    for i in 0..2 {
        if p.huff.load_huffman_table[i] == 0 {
            continue;
        }
        let t = &p.huff.huffman_table[i];
        let nac: usize = t.num_ac_codes.iter().map(|&b| b as usize).sum();
        out.push(0x10 | i as u8);
        out.extend_from_slice(&t.num_ac_codes);
        out.extend_from_slice(&t.ac_values[..nac]);
    }
    let len = (out.len() - len_pos) as u16;
    out[len_pos..len_pos + 2].copy_from_slice(&len.to_be_bytes());
    // DRI, only when a restart interval is set (picture_mjpeg.c:215-224).
    if p.slice.restart_interval != 0 {
        out.extend_from_slice(&[0xFF, 0xDD, 0x00, 0x04]);
        out.extend_from_slice(&p.slice.restart_interval.to_be_bytes());
    }
    // SOF0 baseline, always marker C0 (picture_mjpeg.c:226-253).
    out.extend_from_slice(&[0xFF, 0xC0]);
    let len_pos = out.len();
    out.extend_from_slice(&[0, 0]);
    out.push(0x08);
    out.extend_from_slice(&p.pic.picture_height.to_be_bytes());
    out.extend_from_slice(&p.pic.picture_width.to_be_bytes());
    out.push(p.pic.num_components);
    for i in 0..p.pic.num_components as usize {
        let c = &p.pic.components[i];
        out.push(c.component_id);
        out.push((c.h_sampling_factor << 4) | c.v_sampling_factor);
        out.push(c.quantiser_table_selector);
    }
    let len = (out.len() - len_pos) as u16;
    out[len_pos..len_pos + 2].copy_from_slice(&len.to_be_bytes());
    // SOS with spectral range 00 3F 00 (picture_mjpeg.c:257-277).
    out.extend_from_slice(&[0xFF, 0xDA]);
    let len_pos = out.len();
    out.extend_from_slice(&[0, 0]);
    out.push(p.slice.num_components);
    for i in 0..p.slice.num_components as usize {
        let c = &p.slice.components[i];
        out.push(c.component_selector);
        out.push((c.dc_table_selector << 4) | c.ac_table_selector);
    }
    out.extend_from_slice(&[0x00, 0x3F, 0x00]);
    let len = (out.len() - len_pos) as u16;
    out[len_pos..len_pos + 2].copy_from_slice(&len.to_be_bytes());
    // Slice data + EOI (decode.c:262-265).
    out.extend_from_slice(p.entropy);
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum State {
    /// Host may rewrite bitstream/IB; output is not GPU-owned.
    Idle,
    /// JPEG ioctl submitted; output must not be read or reused.
    DecodeSubmitted,
    /// A post-ioctl ambiguity, timeout, or abandoned flight. Only
    /// destroy/recreate clears it.
    Poisoned,
}

/// Direct-ring VCN JPEG decoder. Owns persistent GTT bitstream and JPEG-IB
/// BOs, one linear VRAM output BO, and one signal syncobj. (The submit BO
/// list is built and destroyed inside `submit_async` from the passed refs;
/// the decoder owns no BO list.) Single in-flight surface: the `&mut self`
/// borrow held by [`PendingJpeg`] prevents a second submit while output is
/// GPU-owned.
pub struct VcnJpegDecoder {
    reg: RegSet,
    engine: Engine,
    signal: SyncObj,
    va_align: u64,
    size_granule_dwords: u32,
    bs: Option<GpuBuffer>,
    ib: Option<GpuBuffer>,
    surf: Option<GpuBuffer>,
    layout: Option<JpegSurfaceLayout>,
    state: State,
}

impl VcnJpegDecoder {
    /// Query VCN_JPEG (IP 8) and fail closed on no rings or an unknown VCN
    /// IP discovery version. Allocates nothing yet: surface BOs are sized
    /// from the first submitted frame.
    pub fn new(dev: &Device) -> Result<Self> {
        let info = dev.query_hw_ip_info(AMDGPU_HW_IP_VCN_JPEG, 0)?;
        if info.available_rings & 0x1 == 0 {
            return Err(RedlineError {
                code: -1,
                message: "vcn-jpeg: IP 8 reports no available rings".into(),
            });
        }
        let v = info.ip_discovery_version;
        let reg = select_reg_set(v).ok_or_else(|| RedlineError {
            code: -1,
            message: format!(
                "vcn-jpeg: unsupported VCN IP discovery version 0x{v:06x} \
                 (major={} minor={} rev={})",
                (v >> 16) & 0xff,
                (v >> 8) & 0xff,
                v & 0xff,
            ),
        })?;
        // IB allocation/VA alignment is max(reported start alignment, 256);
        // IB size is padded to a 16-dword multiple, which also satisfies a
        // 64-byte size alignment (T0 measured both in bytes).
        let va_align = (info.ib_start_alignment as u64).max(256);
        let size_granule_dwords = (info.ib_size_alignment / 4).max(16);
        let signal = SyncObj::new(dev)?;
        Ok(Self {
            reg,
            engine: Engine {
                ip_type: AMDGPU_HW_IP_VCN_JPEG,
                ip_instance: 0,
                ring: 0,
            },
            signal,
            va_align,
            size_granule_dwords,
            bs: None,
            ib: None,
            surf: None,
            layout: None,
            state: State::Idle,
        })
    }

    /// Parse, serialize, upload, and submit one frame. Performs no host
    /// wait: the returned [`PendingJpeg`] must be terminally waited with
    /// [`PendingJpeg::wait_decode`] (decode-only) or
    /// [`PendingJpeg::complete_after`] (chained) before the output is
    /// touched. Reallocates persistent BOs only while Idle (guaranteed by
    /// the `&mut self` borrow: no flight can be outstanding); reallocation
    /// is setup cost, excluded from warmed per-frame timing.
    pub fn submit<'a>(
        &'a mut self,
        dev: &Device,
        queue: &ComputeQueue,
        jpeg: &[u8],
    ) -> Result<PendingJpeg<'a>> {
        if self.state == State::Poisoned {
            return Err(RedlineError {
                code: -1,
                message: "vcn-jpeg: decoder poisoned; destroy and recreate".into(),
            });
        }
        if self.state != State::Idle {
            return Err(RedlineError {
                code: -1,
                message: "vcn-jpeg: submit while a decode is in flight".into(),
            });
        }
        // Preparing failure rolls back to Idle without signaling: no ioctl
        // has run yet, so just propagate the error with state untouched.
        let p = parse_for_va(jpeg).map_err(|e| RedlineError {
            code: -1,
            message: format!("vcn-jpeg: parse failed: {e:?}"),
        })?;
        let format = classify(&p)?;
        let (width, height) = (p.pic.picture_width as u32, p.pic.picture_height as u32);
        if width == 0 || height == 0 {
            return Err(RedlineError {
                code: -1,
                message: "vcn-jpeg: zero dimensions".into(),
            });
        }
        let stream = canonical_stream(&p);
        let bsd_size = align_up(stream.len() as u64, 128);
        if bsd_size == 0 || bsd_size > u32::MAX as u64 {
            return Err(RedlineError {
                code: -1,
                message: "vcn-jpeg: degenerate bitstream size".into(),
            });
        }
        let (layout, surf_total) = surface_layout(width, height, format);

        // (Re)allocate persistent BOs while Idle.
        if self.bs.as_ref().map(|b| b.size).unwrap_or(0) < bsd_size {
            if let Some(old) = self.bs.take() {
                dev.free_buffer(old)?;
            }
            self.bs = Some(dev.alloc_gtt(bsd_size)?);
        }
        let bs_addr = self.bs.as_ref().expect("bitstream BO allocated").gpu_addr;
        let realloc_surf = self.layout.map(|l| l != layout).unwrap_or(true)
            || self.surf.as_ref().map(|b| b.size).unwrap_or(0) < surf_total;
        if realloc_surf {
            if let Some(old) = self.surf.take() {
                dev.free_buffer(old)?;
            }
            self.surf = Some(dev.alloc_vram(surf_total)?);
            self.layout = Some(layout);
        }
        let surf_buf = self.surf.as_ref().expect("surface BO allocated");
        let surf_addr = surf_buf.gpu_addr;

        // Upload the table-bearing stream, zero-filled through
        // align(length, 128); bsd_size is the aligned length and RB_WPTR is
        // bsd_size/4 (radeon_vcn_dec_jpeg.c:30, 421, 275).
        let mut bs_host = vec![0u8; bsd_size as usize];
        bs_host[..stream.len()].copy_from_slice(&stream);
        dev.upload(self.bs.as_ref().expect("bitstream BO allocated"), &bs_host)?;

        // Build + upload the register IB.
        let luma_off = layout.plane_offsets[0];
        let chroma_off = layout.plane_offsets[1];
        let chromav_off = match format {
            JpegNativeFormat::Nv12 => 0,
            JpegNativeFormat::Yuv444p => layout.plane_offsets[2],
        };
        let words = self.build_ib(
            bs_addr,
            surf_addr,
            bsd_size as u32,
            layout.pitch,
            luma_off,
            chroma_off,
            chromav_off,
        );
        debug_assert!(words.len() % 2 == 0, "packet must be an even dword count");
        let ib_bytes_needed = words.len() as u64 * 4;
        if self.ib.as_ref().map(|b| b.size).unwrap_or(0) < ib_bytes_needed {
            if let Some(old) = self.ib.take() {
                dev.free_buffer(old)?;
            }
            self.ib = Some(dev.alloc_gtt(ib_bytes_needed)?);
        }
        let ib_buf = self.ib.as_ref().expect("IB BO allocated");
        if ib_buf.gpu_addr % self.va_align != 0 {
            return Err(RedlineError {
                code: -1,
                message: format!(
                    "vcn-jpeg: IB VA 0x{:x} misaligned to {}",
                    ib_buf.gpu_addr, self.va_align
                ),
            });
        }
        let mut ib_host = Vec::with_capacity(words.len() * 4);
        for w in &words {
            ib_host.extend_from_slice(&w.to_le_bytes());
        }
        dev.upload(ib_buf, &ib_host)?;

        // Refresh canary guard bands before handing the surface to the GPU.
        write_canaries(dev, surf_buf, &layout)?;

        // Submit with no host wait. The BO list is built and destroyed
        // inside submit_async from these refs; the decoder owns no BO list.
        let sync = SubmitSync {
            wait: &[],
            signal: Some(self.signal),
            ib_flags: 0,
        };
        let bs_ref = self.bs.as_ref().expect("bitstream BO allocated");
        let fence = match queue.submit_async(
            dev,
            self.engine,
            ib_buf,
            words.len() as u32,
            &[bs_ref, ib_buf, surf_buf],
            &sync,
        ) {
            Ok(f) => f,
            Err(e) => {
                // Post-ioctl ambiguity poisons the decoder.
                self.state = State::Poisoned;
                return Err(e);
            }
        };
        self.state = State::DecodeSubmitted;
        Ok(PendingJpeg {
            dec: Some(self),
            fence,
            layout,
            ib_words: words,
        })
    }

    /// Emit the full decode packet: seven-write soft-reset/deassert
    /// sequence (radeon_vcn_dec_jpeg.c:245-258), table-bearing GTT stream
    /// address + RB_BASE/RB_SIZE/RB_WPTR (260-275), pitch/linear-mode/write
    /// BAR/plane offsets (305-339), TIER/OUTBUF/INT_EN/CNTL start (371-389),
    /// both completion polls then CNTL stop (391-407). V3 writes
    /// LUMA/CHROMA/CHROMAV directly plus no-crop ROI defaults and
    /// FC_SPS native (340-369); V2 programs planes through INDEX/DATA
    /// (327-335). Requires an even dword count, then pads to the size
    /// granule (16 dwords minimum) with two-dword `{0x60000000, 0}` NOPs.
    fn build_ib(
        &self,
        bs_addr: u64,
        surf_addr: u64,
        bsd_size: u32,
        pitch: u32,
        luma_off: u32,
        chroma_off: u32,
        chromav_off: u32,
    ) -> Vec<u32> {
        let r = self.reg;
        let mut w = Vec::with_capacity(96);
        // Bitstream prologue.
        pkt(r.soft_rst, COND0, TYPE0, 1, &mut w);
        pkt(JRBC_COND, COND0, TYPE0, JRBC_TIMER, &mut w);
        pkt(JRBC_REF, COND0, TYPE0, 1 << 16, &mut w);
        pkt(r.soft_rst, COND3, TYPE3, 1 << 16, &mut w);
        pkt(r.soft_rst, COND0, TYPE0, 0, &mut w);
        pkt(JRBC_REF, COND0, TYPE0, 0, &mut w);
        pkt(r.soft_rst, COND3, TYPE3, 1 << 16, &mut w);
        pkt(r.read_hi, COND0, TYPE0, (bs_addr >> 32) as u32, &mut w);
        pkt(r.read_lo, COND0, TYPE0, bs_addr as u32, &mut w);
        pkt(RB_BASE, COND0, TYPE0, 0, &mut w);
        pkt(RB_SIZE, COND0, TYPE0, 0xFFFF_FFF0, &mut w);
        pkt(RB_WPTR, COND0, TYPE0, bsd_size >> 2, &mut w);
        // Target: pitch in 16-byte units, linear address mode, no swizzle.
        pkt(r.pitch, COND0, TYPE0, pitch >> 4, &mut w);
        pkt(r.uv_pitch, COND0, TYPE0, pitch >> 4, &mut w);
        pkt(r.addr_mode, COND0, TYPE0, 0, &mut w);
        pkt(r.y_tile, COND0, TYPE0, 0, &mut w);
        pkt(r.uv_tile, COND0, TYPE0, 0, &mut w);
        pkt(r.write_hi, COND0, TYPE0, (surf_addr >> 32) as u32, &mut w);
        pkt(r.write_lo, COND0, TYPE0, surf_addr as u32, &mut w);
        match r.ver {
            RegVer::V2 => {
                pkt(r.index, COND0, TYPE0, 0, &mut w);
                pkt(r.data, COND0, TYPE0, luma_off, &mut w);
                pkt(r.index, COND0, TYPE0, 1, &mut w);
                pkt(r.data, COND0, TYPE0, chroma_off, &mut w);
                if chromav_off != 0 {
                    pkt(r.index, COND0, TYPE0, 2, &mut w);
                    pkt(r.data, COND0, TYPE0, chromav_off, &mut w);
                }
            }
            RegVer::V3 => {
                pkt(r.luma, COND0, TYPE0, luma_off, &mut w);
                pkt(r.chroma, COND0, TYPE0, chroma_off, &mut w);
                pkt(r.chromav, COND0, TYPE0, chromav_off, &mut w);
                pkt(ROI_START, COND0, TYPE0, (0 << 16) | 0, &mut w);
                pkt(ROI_STRIDE, COND0, TYPE0, (1 << 16) | 1, &mut w);
                pkt(FC_SPS, COND0, TYPE0, FC_SPS_NATIVE, &mut w);
            }
        }
        // tier_cntl2 uses type 0 exactly as Mesa's direct path does
        // (radeon_vcn_dec_jpeg.c:371).
        pkt(r.tier, COND0, 0, 0, &mut w);
        pkt(r.outbuf_rptr, COND0, TYPE0, 0, &mut w);
        pkt(r.outbuf_cntl, COND0, TYPE0, OUTBUF_CNTL_VAL, &mut w);
        pkt(INT_EN, COND0, TYPE0, 0xFFFF_FFFE, &mut w);
        pkt(CNTL, COND0, TYPE0, 0x6, &mut w);
        // Completion polls, then stop.
        pkt(JRBC_REF, COND0, TYPE0, bsd_size >> 2, &mut w);
        pkt(JRBC_COND, COND0, TYPE0, JRBC_TIMER, &mut w);
        pkt(RB_RPTR, COND3, TYPE3, 0xFFFF_FFFF, &mut w);
        pkt(JRBC_REF, COND0, TYPE0, 0xFFFF_FFFF, &mut w);
        pkt(r.outbuf_wptr, COND3, TYPE3, 0x0000_0001, &mut w);
        pkt(CNTL, COND0, TYPE0, 0x4, &mut w);
        // Pad to the size granule with two-dword NOPs.
        let gran = self.size_granule_dwords as usize;
        while w.len() % gran != 0 {
            w.push(0x6000_0000);
            w.push(0x0000_0000);
        }
        w
    }

    /// Re-check every canary band of the current surface. Catches
    /// under/over-allocation during the first full decode.
    pub fn verify_canaries(&self, dev: &Device) -> Result<()> {
        let (surf, layout) = match (self.surf.as_ref(), self.layout) {
            (Some(s), Some(l)) => (s, l),
            _ => {
                return Err(RedlineError {
                    code: -1,
                    message: "vcn-jpeg: no surface to verify".into(),
                });
            }
        };
        let mut host = vec![0u8; surf.size as usize];
        dev.download(surf, &mut host)?;
        for (off, len, pat) in canary_ranges(&layout) {
            let range = off as usize..(off + len) as usize;
            if host[range.clone()].iter().any(|&b| b != pat) {
                return Err(RedlineError {
                    code: -1,
                    message: format!(
                        "vcn-jpeg: canary corrupted at surface offset {off} (len {len})"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Release persistent BOs and the signal syncobj exactly once,
    /// matching Redline's explicit-destroy ownership style. Must be called;
    /// there is deliberately no `Drop` (freeing needs `&Device`).
    pub fn destroy(self, dev: &Device) {
        if let Some(b) = self.bs {
            let _ = dev.free_buffer(b);
        }
        if let Some(b) = self.ib {
            let _ = dev.free_buffer(b);
        }
        if let Some(b) = self.surf {
            let _ = dev.free_buffer(b);
        }
        self.signal.destroy(dev);
    }
}

fn write_canaries(dev: &Device, surf: &GpuBuffer, layout: &JpegSurfaceLayout) -> Result<()> {
    use std::ffi::c_void;
    use std::ptr;
    let mut cpu: *mut c_void = ptr::null_mut();
    // SAFETY: valid BO handle; unmapped once below on the success path.
    let ret = unsafe { (dev.drm.bo_cpu_map)(surf.handle, &mut cpu) };
    if ret != 0 {
        return Err(RedlineError {
            code: ret,
            message: format!("vcn-jpeg: bo_cpu_map for canaries failed: {ret}"),
        });
    }
    for (off, len, pat) in canary_ranges(layout) {
        // SAFETY: ranges are inside the BO by construction (the tail canary
        // ends exactly at the allocation size); `cpu` stays mapped for the
        // whole loop and is unmapped once below.
        unsafe {
            ptr::write_bytes((cpu as *mut u8).add(off as usize), pat, len as usize);
        }
    }
    // SAFETY: pairs the single map above; the pointer is not used afterwards.
    unsafe {
        (dev.drm.bo_cpu_unmap)(surf.handle);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// In-flight decode
// ---------------------------------------------------------------------------

/// A submitted decode: output is GPU-owned and must not be read or reused.
/// The `&mut` borrow of the decoder prevents a second submit while alive.
/// Dropping without a terminal wait poisons the decoder (loud, not corrupt).
pub struct PendingJpeg<'a> {
    dec: Option<&'a mut VcnJpegDecoder>,
    fence: SubmissionFence,
    layout: JpegSurfaceLayout,
    ib_words: Vec<u32>,
}

impl<'a> PendingJpeg<'a> {
    fn dec(&self) -> &VcnJpegDecoder {
        self.dec.as_ref().expect("flight holds decoder")
    }

    /// The output surface BO (GPU-owned until the terminal wait).
    pub fn surface(&self) -> &GpuBuffer {
        self.dec()
            .surf
            .as_ref()
            .expect("surface live during flight")
    }

    /// Geometry of the output surface.
    pub fn layout(&self) -> JpegSurfaceLayout {
        self.layout
    }

    /// Syncobj the JPEG submit signals. A consumer compute submit waits on
    /// this via `SYNCOBJ_IN` instead of any host wait.
    pub fn ready_syncobj(&self) -> &SyncObj {
        &self.dec().signal
    }

    /// The JPEG submission fence (decode-only terminal path).
    pub fn decode_fence(&self) -> &SubmissionFence {
        &self.fence
    }

    /// Host copy of the submitted IB dwords, for trace/packet-dump evidence.
    pub fn ib_dwords(&self) -> &[u32] {
        &self.ib_words
    }

    /// Decode-only/parity terminal path: wait the JPEG fence once. Success
    /// returns [`ReadyJpeg`] for CPU readback; timeout or error poisons the
    /// decoder. Must not be used after handing `ready_syncobj` to a
    /// consumer submit — that flight belongs to [`PendingJpeg::complete_after`].
    pub fn wait_decode(
        mut self,
        dev: &Device,
        queue: &ComputeQueue,
        timeout_ns: u64,
    ) -> Result<ReadyJpeg<'a>> {
        // Copy the fence first: `wait_fence` runs while no decoder borrow is
        // held, so poisoning below never aliases.
        let fence = self.fence;
        let done = match queue.wait_fence(dev, &fence, timeout_ns) {
            Ok(done) => done,
            Err(e) => {
                self.dec.as_mut().expect("flight holds decoder").state = State::Poisoned;
                return Err(e);
            }
        };
        if !done {
            self.dec.as_mut().expect("flight holds decoder").state = State::Poisoned;
            return Err(RedlineError {
                code: -1,
                message: "vcn-jpeg: decode fence timeout".into(),
            });
        }
        // Move the borrow into ReadyJpeg without triggering the abandon
        // poison: state stays DecodeSubmitted until ReadyJpeg drops to Idle.
        // `layout` is our own copy; the decoder borrow moves untouched.
        let layout = self.layout;
        let dec = self.dec.take().expect("flight holds decoder");
        std::mem::forget(self);
        Ok(ReadyJpeg { dec, layout })
    }

    /// Chained path: wait the consumer fence once and return to Idle. This
    /// is the only transition back to Idle for a consumer-chained flight.
    /// Timeout or error poisons the decoder.
    pub fn complete_after(
        mut self,
        dev: &Device,
        queue: &ComputeQueue,
        consumer: &SubmissionFence,
        timeout_ns: u64,
    ) -> Result<()> {
        let done = match queue.wait_fence(dev, consumer, timeout_ns) {
            Ok(done) => done,
            Err(e) => {
                self.dec.as_mut().expect("flight holds decoder").state = State::Poisoned;
                return Err(e);
            }
        };
        if !done {
            self.dec.as_mut().expect("flight holds decoder").state = State::Poisoned;
            return Err(RedlineError {
                code: -1,
                message: "vcn-jpeg: consumer fence timeout".into(),
            });
        }
        self.dec.as_mut().expect("flight holds decoder").state = State::Idle;
        // `self` is consumed; suppress the abandon-drop poison explicitly.
        std::mem::forget(self);
        Ok(())
    }
}

impl Drop for PendingJpeg<'_> {
    fn drop(&mut self) {
        if let Some(dec) = self.dec.as_mut() {
            if dec.state == State::DecodeSubmitted {
                dec.state = State::Poisoned;
            }
        }
    }
}

/// A fence-waited decode, safe for CPU readback via [`Device::download`]
/// on [`ReadyJpeg::surface`]. Dropping releases the decoder to Idle (GPU
/// work is already complete, so this cannot fail).
pub struct ReadyJpeg<'a> {
    dec: &'a mut VcnJpegDecoder,
    layout: JpegSurfaceLayout,
}

impl ReadyJpeg<'_> {
    pub fn surface(&self) -> &GpuBuffer {
        self.dec.surf.as_ref().expect("surface live while ready")
    }

    pub fn layout(&self) -> JpegSurfaceLayout {
        self.layout
    }
}

impl Drop for ReadyJpeg<'_> {
    fn drop(&mut self) {
        if self.dec.state == State::DecodeSubmitted {
            self.dec.state = State::Idle;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_v3_register_sets_match_plan() {
        // Common programming model (plan §packet_transcription).
        assert_eq!(
            (JRBC_COND, JRBC_REF, RB_BASE, RB_SIZE, RB_WPTR),
            (0x408e, 0x408f, 0x4001, 0x4004, 0x4002)
        );
        assert_eq!((INT_EN, CNTL, RB_RPTR), (0x400a, 0x4000, 0x4003));
        // V2 specifics.
        let v2 = RegSet::V2;
        assert_eq!(
            (v2.soft_rst, v2.read_hi, v2.read_lo, v2.pitch, v2.uv_pitch),
            (0x402f, 0x40e1, 0x40e0, 0x401f, 0x4020)
        );
        assert_eq!(
            (
                v2.addr_mode,
                v2.y_tile,
                v2.uv_tile,
                v2.write_hi,
                v2.write_lo
            ),
            (0x4027, 0x4024, 0x4025, 0x40e3, 0x40e2)
        );
        assert_eq!(
            (
                v2.tier,
                v2.outbuf_cntl,
                v2.outbuf_rptr,
                v2.outbuf_wptr,
                v2.index,
                v2.data
            ),
            (0x400f, 0x401c, 0x401e, 0x401d, 0x402c, 0x402d)
        );
        // V3 specifics.
        let v3 = RegSet::V3;
        assert_eq!(
            (v3.soft_rst, v3.read_hi, v3.read_lo, v3.pitch, v3.uv_pitch),
            (0x4051, 0x40b3, 0x40b2, 0x4043, 0x4044)
        );
        assert_eq!(
            (
                v3.addr_mode,
                v3.y_tile,
                v3.uv_tile,
                v3.write_hi,
                v3.write_lo
            ),
            (0x404b, 0x4048, 0x4049, 0x40b5, 0x40b4)
        );
        assert_eq!(
            (v3.tier, v3.outbuf_cntl, v3.outbuf_rptr, v3.outbuf_wptr),
            (0x400e, 0x4040, 0x4042, 0x4041)
        );
        assert_eq!(
            (v3.luma, v3.chroma, v3.chromav, ROI_START, ROI_STRIDE, FC_SPS),
            (0x41c0, 0x41c1, 0x41c2, 0x401b, 0x401c, 0x4052)
        );
        // Version selection incl. fail-closed unknowns.
        assert_eq!(select_reg_set(0x050000).unwrap().ver, RegVer::V3);
        assert_eq!(select_reg_set(0x050001).unwrap().ver, RegVer::V3);
        assert_eq!(select_reg_set(0x040003).unwrap().ver, RegVer::V3);
        assert_eq!(select_reg_set(0x030021).unwrap().ver, RegVer::V2);
        assert_eq!(select_reg_set(0x040006).unwrap().ver, RegVer::V2);
        assert!(select_reg_set(0x010000).is_none());
        assert!(select_reg_set(0x060000).is_none());
        assert!(select_reg_set(0xffffff).is_none());
    }

    #[test]
    fn doge_layout_matches_plan() {
        // doge.jpeg is 537x529 NV12: pitch = align(537,256) = 768,
        // coded_h = align(529,16) = 544.
        let (l, total) = surface_layout(537, 529, JpegNativeFormat::Nv12);
        assert_eq!(l.pitch, 768);
        assert_eq!(l.plane_offsets[0], 256);
        assert_eq!(l.plane_offsets[1], 256 + 768 * 544 + 256);
        assert_eq!(l.plane_rows, [529, 265, 0]);
        assert_eq!(l.plane_count, 2);
        assert_eq!(total, 256 + 768 * 544 + 256 + 768 * 272 + 256);
        assert_eq!(total, 627_456);
        for o in [l.plane_offsets[0], l.plane_offsets[1]] {
            assert_eq!(o % 256, 0);
        }
    }

    #[test]
    fn packet_word_encoding() {
        let mut w = Vec::new();
        pkt(0x4043, COND0, TYPE0, 0x30, &mut w);
        assert_eq!(w, vec![0x4043, 0x30]);
        pkt(0x4003, COND3, TYPE3, 0xFFFF_FFFF, &mut w);
        assert_eq!(w[2], (0x4003 | (3 << 24) | (3 << 28)) as u32);
        assert_eq!(w[3], 0xFFFF_FFFF);
    }
}
