// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! H.264 Annex-B decode through the VA-API bridge (`experiment/vcn-video`).
//!
//! Scope: baseline/high-profile **progressive** H.264, 4:2:0 8-bit, no B
//! slices, no FMO/ASO, no MBAFF/fields. The demuxer is out of scope: input is
//! an Annex-B elementary stream produced by ffmpeg
//! (`ffmpeg -i X -c:v copy -bsf:v h264_mp4toannexb -f h264 out.h264`).
//! The experiment fixture is additionally re-encoded with `-bf 0`, so decode
//! order == display order and no reordering buffer exists.
//!
//! Design authority is FFmpeg's `libavcodec/vaapi_h264.c` (buffer fill),
//! `h264_parse.c:ff_h264_init_poc` (POC), and `h264_ps.c` (scaling-list
//! fallbacks); struct layouts are libva 2.23 `va/va.h` (see `ffi.rs`).
//!
//! Per picture: one `VAPictureParameterBufferH264` + one
//! `VAIQMatrixBufferH264` + one (slice-param, slice-data) pair per slice NAL,
//! rendered in a single `vaRenderPicture` call. Reference surfaces live in a
//! pool (`max_num_ref_frames + 3`) so `ReferenceFrames`/`RefPicList0` stay
//! valid across pictures; every picture is synced before readback, so with
//! `-bf 0` pictures are returned in display order.

use crate::ffi::{
    VaIqMatrixH264, VaLib, VaPicParamH264, VaPictureH264, VaSliceParamH264,
    VaDrmPrimeDescriptor, VaImage, VaImageFormat, VA_EXPORT_SURFACE_READ_ONLY,
    VA_FOURCC_NV12, VA_MEM_TYPE_DRM_PRIME_2, VA_PICTURE_H264_INVALID,
    VA_PICTURE_H264_LONG_TERM_REFERENCE, VA_PICTURE_H264_SHORT_TERM_REFERENCE,
    VA_RT_FORMAT_YUV420, VA_SLICE_DATA_FLAG_ALL, VA_STATUS_SUCCESS,
    VA_INVALID_SURFACE, VA_IQ_MATRIX_TYPE, VA_PIC_PARAM_TYPE, VA_SLICE_DATA_TYPE,
    VA_SLICE_PARAM_TYPE, VaDisplay, VaError,
};
use crate::{DerivedFrame, VcnFrame};
use crate::interop::HipMapping;
use std::ffi::c_void;

// ── Default scaling lists (ITU-T H.264 Tables 7-3/7-4) ──────────────────────
// Raster-scan order, verbatim from FFmpeg `libavcodec/h264_ps.c`
// (`default_scaling4/8`), which is what `vaapi_h264.c` memcpys into the VA
// buffer when neither SPS nor PPS carries lists (our fixture: both absent).
const DEFAULT_SCALING_4X4: [[u8; 16]; 2] = [
    [
        6, 13, 20, 28, 13, 20, 28, 32, 20, 28, 32, 37, 28, 32, 37, 42,
    ],
    [
        10, 14, 20, 24, 14, 20, 24, 27, 20, 24, 27, 30, 24, 27, 30, 34,
    ],
];
const DEFAULT_SCALING_8X8: [[u8; 64]; 2] = [
    [
        6, 10, 13, 16, 18, 23, 25, 27, 10, 11, 16, 18, 23, 25, 27, 29, 13, 16,
        18, 23, 25, 27, 29, 31, 16, 18, 23, 25, 27, 29, 31, 33, 18, 23, 25, 27,
        29, 31, 33, 36, 23, 25, 27, 29, 31, 33, 36, 38, 25, 27, 29, 31, 33, 36,
        38, 40, 27, 29, 31, 33, 36, 38, 40, 42,
    ],
    [
        9, 13, 15, 17, 19, 21, 22, 24, 13, 13, 17, 19, 21, 22, 24, 25, 15, 17,
        19, 21, 22, 24, 25, 27, 17, 19, 21, 22, 24, 25, 27, 28, 19, 21, 22, 24,
        25, 27, 28, 30, 21, 22, 24, 25, 27, 28, 30, 32, 22, 24, 25, 27, 28, 30,
        32, 33, 24, 25, 27, 28, 30, 32, 33, 35,
    ],
];
// Zigzag scans (bitstream order → raster storage), used only when a stream
// carries explicit scaling lists.
const ZIGZAG_4X4: [usize; 16] =
    [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
const ZIGZAG_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33,
    40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43,
    36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60,
    61, 54, 47, 55, 62, 63,
];

// ── Bit reader ──────────────────────────────────────────────────────────────
struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize, // in bits
}
impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn pos(&self) -> usize {
        self.pos
    }
    fn get(&mut self, n: u32) -> Result<u32, VaError> {
        if n > 32 || self.pos + n as usize > self.buf.len() * 8 {
            return Err(VaError::Corrupt("h264: bitstream overread"));
        }
        let mut v = 0u32;
        for _ in 0..n {
            let b = (self.buf[self.pos >> 3] >> (7 - (self.pos & 7))) & 1;
            v = (v << 1) | b as u32;
            self.pos += 1;
        }
        Ok(v)
    }
    fn flag(&mut self) -> Result<bool, VaError> {
        Ok(self.get(1)? != 0)
    }
    fn ue(&mut self) -> Result<u32, VaError> {
        let mut zeros = 0u32;
        while self.get(1)? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err(VaError::Corrupt("h264: ue overflow"));
            }
        }
        Ok(if zeros == 0 {
            0
        } else {
            (1 << zeros) - 1 + self.get(zeros)?
        })
    }
    fn se(&mut self) -> Result<i32, VaError> {
        let v = self.ue()?;
        Ok(if v & 1 == 1 {
            ((v + 1) >> 1) as i32
        } else {
            -((v >> 1) as i32)
        })
    }
    /// CABAC slice-header terminator: byte-align on one-bits.
    fn align_ones(&mut self) -> Result<(), VaError> {
        while self.pos & 7 != 0 {
            if self.get(1)? != 1 {
                return Err(VaError::Corrupt("h264: bad cabac alignment bit"));
            }
        }
        Ok(())
    }
}

/// Strip emulation-prevention bytes (`00 00 03` → `00 00`) for parsing.
/// The slice *data* buffer keeps the original bytes (the driver unescapes).
fn rbsp(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0;
    for &b in nal {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

/// Split Annex-B into NAL units (start codes consumed, empty units dropped).
fn split_annexb(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 2 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push((i, 3));
            i += 3;
        } else if i + 3 < data.len()
            && data[i] == 0
            && data[i + 1] == 0
            && data[i + 2] == 0
            && data[i + 3] == 1
        {
            starts.push((i, 4));
            i += 4;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::new();
    for (k, &(off, len)) in starts.iter().enumerate() {
        let mut end = if k + 1 < starts.len() {
            starts[k + 1].0
        } else {
            data.len()
        };
        // A 4-byte start code's leading zero belongs to the code, not to
        // the preceding NAL (shim-verified: ffmpeg's slice data is 1 byte
        // shorter than a naive split here).
        if end > 0 && end < data.len() && data[end - 1] == 0 {
            end -= 1;
        }
        let nal = &data[off + len..end];
        if !nal.is_empty() {
            nals.push(nal);
        }
    }
    nals
}

// ── Parameter sets ──────────────────────────────────────────────────────────
#[derive(Clone)]
struct Sps {
    profile_idc: u8,
    level_idc: u8,
    chroma_format_idc: u32,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    log2_max_frame_num_minus4: u32,
    poc_type: u32,
    log2_max_poc_lsb_minus4: u32,
    max_num_ref_frames: u32,
    gaps_flag: bool,
    pic_width_in_mbs_minus1: u32,
    pic_height_in_map_units_minus1: u32,
    frame_mbs_only_flag: bool,
    direct_8x8_inference_flag: bool,
    crop_left: u32,
    crop_right: u32,
    crop_top: u32,
    crop_bottom: u32,
    scaling4: [[u8; 16]; 6],
    scaling8: [[u8; 64]; 6],
    scaling_present: bool,
}

impl Sps {
    fn max_frame_num(&self) -> u32 {
        1 << (self.log2_max_frame_num_minus4 + 4)
    }
    fn coded_width(&self) -> u32 {
        (self.pic_width_in_mbs_minus1 + 1) * 16
    }
    fn coded_height(&self) -> u32 {
        // frame-only (experiment scope): map units == MB rows.
        (self.pic_height_in_map_units_minus1 + 1) * 16
    }
    fn display_size(&self) -> (u32, u32) {
        // 4:2:0 frame crop units are 2x2 luma samples.
        let (w, h) = (self.coded_width(), self.coded_height());
        (
            w - 2 * (self.crop_left + self.crop_right),
            h - 2 * (self.crop_top + self.crop_bottom),
        )
    }
}

/// Parse one scaling list from the bitstream into raster-order `factors`.
#[allow(clippy::too_many_arguments)]
fn parse_scaling_list(
    br: &mut BitReader,
    factors: &mut [u8],
    scan: &[usize],
    jvt_default: &[u8],
    fallback: &[u8],
) -> Result<(), VaError> {
    if !br.flag()? {
        factors.copy_from_slice(fallback);
        return Ok(());
    }
    let (mut last, mut next) = (8i32, 8i32);
    for i in 0..factors.len() {
        if next != 0 {
            let v = br.se()?;
            if v < -128 || v > 127 {
                return Err(VaError::Corrupt("h264: delta scale out of range"));
            }
            next = (last + v) & 0xff;
        }
        if i == 0 && next == 0 {
            factors.copy_from_slice(jvt_default);
            return Ok(());
        }
        if next != 0 {
            last = next;
        }
        factors[scan[i]] = last as u8;
    }
    Ok(())
}

/// Parse the 8 SPS/PPS scaling lists (4:2:0 ⇒ no chroma 8x8).
/// `sps_fallback`: PPS-level call passes the SPS matrices when the SPS
/// carried them, else the spec defaults (mirrors FFmpeg `fallback[]`).
fn parse_scaling_matrices(
    br: &mut BitReader,
    s4: &mut [[u8; 16]; 6],
    s8: &mut [[u8; 64]; 6],
    want_8x8: bool,
    fb4: (&[u8], &[u8]),
    fb8: (&[u8], &[u8]),
) -> Result<(), VaError> {
    for i in 0..6 {
        let (jvt, fb) = if i < 3 {
            (&DEFAULT_SCALING_4X4[0][..], fb4.0)
        } else {
            (&DEFAULT_SCALING_4X4[1][..], fb4.1)
        };
        parse_scaling_list(br, &mut s4[i], &ZIGZAG_4X4, jvt, fb)?;
    }
    if want_8x8 {
        for (k, idx) in [0usize, 3usize].iter().enumerate() {
            let (jvt, fb) = if k == 0 {
                (&DEFAULT_SCALING_8X8[0][..], fb8.0)
            } else {
                (&DEFAULT_SCALING_8X8[1][..], fb8.1)
            };
            parse_scaling_list(br, &mut s8[*idx], &ZIGZAG_8X8, jvt, fb)?;
        }
    }
    Ok(())
}

fn default_matrices() -> ([[u8; 16]; 6], [[u8; 64]; 6]) {
    let mut s4 = [[0u8; 16]; 6];
    let mut s8 = [[0u8; 64]; 6];
    for i in 0..6 {
        s4[i] = DEFAULT_SCALING_4X4[if i < 3 { 0 } else { 1 }];
    }
    s8[0] = DEFAULT_SCALING_8X8[0];
    s8[3] = DEFAULT_SCALING_8X8[1];
    // 4:2:0 never transmits chroma 8x8; VA only reads [0] and [3].
    (s4, s8)
}
/// Flat-16 matrices: when neither SPS nor PPS carries scaling lists the VA
/// IQ buffer is flat (ffmpeg vaapi_h264 submit observed via LD_PRELOAD shim:
/// all 0x10), NOT the JVT default tables. The JVT tables apply only as
/// per-list fallbacks inside a present_flag=1 set that omits lists.
fn flat_matrices() -> ([[u8; 16]; 6], [[u8; 64]; 6]) {
    ([[16u8; 16]; 6], [[16u8; 64]; 6])
}

fn parse_sps(nal: &[u8]) -> Result<(u32, Sps), VaError> {
    let rb = rbsp(nal);
    let mut br = BitReader::new(&rb);
    let hdr = br.get(8)?;
    if hdr & 0x1f != 7 {
        return Err(VaError::Corrupt("h264: not an SPS NAL"));
    }
    let profile_idc = br.get(8)? as u8;
    let _constraints = br.get(8)?;
    let level_idc = br.get(8)? as u8;
    let id = br.ue()?;
    // High-profile extension (also 110/122/244/44/83/86/118/128/138/139/134/135).
    let high = matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    );
    let (mut chroma, mut depth_l, mut depth_c) = (1u32, 0u8, 0u8);
    let (mut s4, mut s8) = default_matrices();
    let mut scaling_present = false;
    if high {
        chroma = br.ue()?;
        if chroma != 1 {
            return Err(VaError::Unsupported("h264: only 4:2:0 supported"));
        }
        depth_l = br.ue()? as u8;
        depth_c = br.ue()? as u8;
        if depth_l != 0 || depth_c != 0 {
            return Err(VaError::Unsupported("h264: only 8-bit supported"));
        }
        let _bypass = br.flag()?;
        if br.flag()? {
            scaling_present = true;
            parse_scaling_matrices(
                &mut br,
                &mut s4,
                &mut s8,
                true,
                (
                    &DEFAULT_SCALING_4X4[0],
                    &DEFAULT_SCALING_4X4[1],
                ),
                (
                    &DEFAULT_SCALING_8X8[0],
                    &DEFAULT_SCALING_8X8[1],
                ),
            )?;
        }
    }
    let log2_max_frame_num_minus4 = br.ue()?;
    if log2_max_frame_num_minus4 > 12 {
        return Err(VaError::Corrupt("h264: log2_max_frame_num out of range"));
    }
    let poc_type = br.ue()?;
    if poc_type > 2 {
        return Err(VaError::Corrupt("h264: bad pic_order_cnt_type"));
    }
    let mut log2_max_poc_lsb_minus4 = 0;
    if poc_type == 0 {
        log2_max_poc_lsb_minus4 = br.ue()?;
        if log2_max_poc_lsb_minus4 > 12 {
            return Err(VaError::Corrupt("h264: log2_max_poc_lsb out of range"));
        }
    } else if poc_type == 1 {
        // Parsed (position-keeping); POC type 1 is rejected at slice time.
        let _always_zero = br.flag()?;
        let _off_nonref = br.se()?;
        let _off_top_bottom = br.se()?;
        let cycle = br.ue()?;
        if cycle > 255 {
            return Err(VaError::Corrupt("h264: poc cycle too long"));
        }
        for _ in 0..cycle {
            let _ = br.se()?;
        }
    }
    let max_num_ref_frames = br.ue()?;
    let gaps_flag = br.flag()?;
    let pic_width_in_mbs_minus1 = br.ue()?;
    let pic_height_in_map_units_minus1 = br.ue()?;
    if pic_width_in_mbs_minus1 > 255 || pic_height_in_map_units_minus1 > 255 {
        return Err(VaError::Corrupt("h264: picture too large"));
    }
    let frame_mbs_only_flag = br.flag()?;
    if !frame_mbs_only_flag {
        return Err(VaError::Unsupported("h264: interlaced/MBAFF"));
    }
    let direct_8x8_inference_flag = br.flag()?;
    let (mut cl, mut cr, mut ct, mut cb) = (0, 0, 0, 0);
    if br.flag()? {
        cl = br.ue()?;
        cr = br.ue()?;
        ct = br.ue()?;
        cb = br.ue()?;
    }
    // VUI, if present, is trailing: nothing after it is needed.
    Ok((
        id,
        Sps {
            profile_idc,
            level_idc,
            chroma_format_idc: chroma,
            bit_depth_luma_minus8: depth_l,
            bit_depth_chroma_minus8: depth_c,
            log2_max_frame_num_minus4,
            poc_type,
            log2_max_poc_lsb_minus4,
            max_num_ref_frames,
            gaps_flag,
            pic_width_in_mbs_minus1,
            pic_height_in_map_units_minus1,
            frame_mbs_only_flag,
            direct_8x8_inference_flag,
            crop_left: cl,
            crop_right: cr,
            crop_top: ct,
            crop_bottom: cb,
            scaling4: s4,
            scaling8: s8,
            scaling_present,
        },
    ))
}

#[derive(Clone)]
struct Pps {
    sps_id: u32,
    entropy_coding_mode_flag: bool,
    pic_order_present_flag: bool,
    num_ref_idx_l0_default_active_minus1: u32,
    num_ref_idx_l1_default_active_minus1: u32,
    weighted_pred_flag: bool,
    weighted_bipred_idc: u32,
    pic_init_qp_minus26: i32,
    pic_init_qs_minus26: i32,
    chroma_qp_index_offset: i32,
    second_chroma_qp_index_offset: i32,
    deblocking_filter_control_present_flag: bool,
    constrained_intra_pred_flag: bool,
    redundant_pic_cnt_present_flag: bool,
    transform_8x8_mode_flag: bool,
    scaling4: [[u8; 16]; 6],
    scaling8: [[u8; 64]; 6],
}

fn parse_pps(nal: &[u8], spss: &[Option<Sps>]) -> Result<(u32, Pps), VaError> {
    let rb = rbsp(nal);
    let mut br = BitReader::new(&rb);
    let hdr = br.get(8)?;
    if hdr & 0x1f != 8 {
        return Err(VaError::Corrupt("h264: not a PPS NAL"));
    }
    let id = br.ue()?;
    let sps_id = br.ue()?;
    let sps = spss
        .get(sps_id as usize)
        .and_then(|s| s.as_ref())
        .ok_or(VaError::Corrupt("h264: PPS refers to unknown SPS"))?;
    let entropy_coding_mode_flag = br.flag()?;
    let pic_order_present_flag = br.flag()?;
    if br.ue()? > 0 {
        return Err(VaError::Unsupported("h264: FMO slice groups"));
    }
    let num_ref_idx_l0_default_active_minus1 = br.ue()?;
    let num_ref_idx_l1_default_active_minus1 = br.ue()?;
    if num_ref_idx_l0_default_active_minus1 > 31 || num_ref_idx_l1_default_active_minus1 > 31 {
        return Err(VaError::Corrupt("h264: ref_idx_default out of range"));
    }
    let weighted_pred_flag = br.flag()?;
    let weighted_bipred_idc = br.get(2)?;
    let pic_init_qp_minus26 = br.se()?;
    let pic_init_qs_minus26 = br.se()?;
    let chroma_qp_index_offset = br.se()?;
    let deblocking_filter_control_present_flag = br.flag()?;
    let constrained_intra_pred_flag = br.flag()?;
    let redundant_pic_cnt_present_flag = br.flag()?;
    let transform_8x8_mode_flag = br.flag()?;
    // Resolved matrices: SPS lists when the SPS carried them, else flat-16
    // (NOT the JVT defaults — shim-verified against ffmpeg).
    let (mut s4, mut s8) = if sps.scaling_present {
        (sps.scaling4, sps.scaling8)
    } else {
        flat_matrices()
    };
    if br.flag()? {
        let (fb4a, fb4b) = if sps.scaling_present {
            (sps.scaling4[0].as_slice(), sps.scaling4[3].as_slice())
        } else {
            (
                DEFAULT_SCALING_4X4[0].as_slice(),
                DEFAULT_SCALING_4X4[1].as_slice(),
            )
        };
        let (fb8a, fb8b) = if sps.scaling_present {
            (sps.scaling8[0].as_slice(), sps.scaling8[3].as_slice())
        } else {
            (
                DEFAULT_SCALING_8X8[0].as_slice(),
                DEFAULT_SCALING_8X8[1].as_slice(),
            )
        };
        parse_scaling_matrices(
            &mut br,
            &mut s4,
            &mut s8,
            transform_8x8_mode_flag,
            (fb4a, fb4b),
            (fb8a, fb8b),
        )?;
    }
    let second_chroma_qp_index_offset = br.se()?;
    Ok((
        id,
        Pps {
            sps_id,
            entropy_coding_mode_flag,
            pic_order_present_flag,
            num_ref_idx_l0_default_active_minus1,
            num_ref_idx_l1_default_active_minus1,
            weighted_pred_flag,
            weighted_bipred_idc,
            pic_init_qp_minus26,
            pic_init_qs_minus26,
            chroma_qp_index_offset,
            second_chroma_qp_index_offset,
            deblocking_filter_control_present_flag,
            constrained_intra_pred_flag,
            redundant_pic_cnt_present_flag,
            transform_8x8_mode_flag,
            scaling4: s4,
            scaling8: s8,
        },
    ))
}

// ── Slice header ────────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
enum Mmco {
    End,
    MarkShortUnused { diff: u32 },
    MarkLongUnused { idx: u32 },
    MarkLong { diff: u32, idx: u32 },
    SetMaxLong { max: u32 },
    MarkAllUnused,
}

struct SliceHeader {
    first_mb_in_slice: u32,
    /// Mapped slice type: 0 = P, 2 = I (B/SP/SI rejected).
    slice_type: u8,
    pps_id: u32,
    frame_num: u32,
    nal_ref_idc: u8,
    is_idr: bool,
    idr_pic_id: u32,
    poc_lsb: u32,
    delta_poc_bottom: i32,
    num_ref_idx_l0_active_minus1: u32,
    list_mod_l0: bool,
    luma_log2_weight_denom: u32,
    chroma_log2_weight_denom: u32,
    lw_flag: bool,
    lw: Vec<i16>,
    lo: Vec<i16>,
    cw_flag: bool,
    cw: Vec<[i16; 2]>,
    co: Vec<[i16; 2]>,
    no_output_of_prior_pics: bool,
    idr_long_term: bool,
    sliding_window: bool,
    mmcos: Vec<Mmco>,
    cabac_init_idc: u32,
    slice_qp_delta: i32,
    disable_deblocking_filter_idc: u32,
    slice_alpha_c0_offset_div2: i32,
    slice_beta_offset_div2: i32,
    /// `slice_data_bit_offset`: RBSP bits consumed incl. the NAL header byte.
    bit_offset: u32,
}

fn parse_slice(
    nal: &[u8],
    ppss: &[Option<Pps>],
    spss: &[Option<Sps>],
) -> Result<(SliceHeader, u32), VaError> {
    let rb = rbsp(nal);
    let mut br = BitReader::new(&rb);
    let hdr = br.get(8)?;
    let (nal_ref_idc, nal_type) = ((hdr >> 5) as u8 & 3, (hdr & 0x1f) as u8);
    let is_idr = nal_type == 5;
    if nal_type != 1 && !is_idr {
        return Err(VaError::Corrupt("h264: not a slice NAL"));
    }
    let first_mb_in_slice = br.ue()?;
    let st = br.ue()?;
    if st > 9 {
        return Err(VaError::Corrupt("h264: bad slice_type"));
    }
    let slice_type = (st % 5) as u8;
    if slice_type == 1 {
        return Err(VaError::Unsupported("h264: B slices"));
    }
    if slice_type > 2 {
        return Err(VaError::Unsupported("h264: SP/SI slices"));
    }
    let pps_id = br.ue()?;
    let pps = ppss
        .get(pps_id as usize)
        .and_then(|p| p.as_ref())
        .ok_or(VaError::Corrupt("h264: slice refers to unknown PPS"))?;
    let sps = spss
        .get(pps.sps_id as usize)
        .and_then(|s| s.as_ref())
        .ok_or(VaError::Corrupt("h264: PPS refers to unknown SPS"))?;
    let max_fn_bits = sps.log2_max_frame_num_minus4 + 4;
    let frame_num = br.get(max_fn_bits)?;
    let mut idr_pic_id = 0;
    if is_idr {
        idr_pic_id = br.ue()?;
    }
    let (mut poc_lsb, mut delta_poc_bottom) = (0u32, 0i32);
    if sps.poc_type == 0 {
        poc_lsb = br.get(sps.log2_max_poc_lsb_minus4 + 4)?;
        if pps.pic_order_present_flag {
            delta_poc_bottom = br.se()?;
        }
    } else if sps.poc_type == 1 {
        return Err(VaError::Unsupported("h264: pic_order_cnt_type 1"));
    }
    if pps.redundant_pic_cnt_present_flag {
        let _ = br.ue()?;
    }
    let is_p = slice_type == 0;
    let mut num_l0 = pps.num_ref_idx_l0_default_active_minus1;
    if is_p && br.flag()? {
        num_l0 = br.ue()?;
        if num_l0 > 31 {
            return Err(VaError::Corrupt("h264: num_ref_idx_l0 out of range"));
        }
    }
    let mut list_mod_l0 = false;
    if is_p && br.flag()? {
        // Parsed for position; reordered lists are out of experiment scope.
        loop {
            let op = br.ue()?;
            if op == 3 {
                break;
            }
            if op > 3 {
                return Err(VaError::Corrupt("h264: bad list mod op"));
            }
            let _ = br.ue()?;
            if op == 2 {
                let _ = br.ue()?;
            }
        }
        list_mod_l0 = true;
    }
    let (mut luma_denom, mut chroma_denom) = (0u32, 0u32);
    let (mut lw_flag, mut cw_flag) = (false, false);
    let (mut lw, mut lo, mut cw, mut co) = (
        Vec::new(),
        Vec::new(),
        Vec::<[i16; 2]>::new(),
        Vec::<[i16; 2]>::new(),
    );
    if is_p && pps.weighted_pred_flag {
        luma_denom = br.ue()?;
        chroma_denom = br.ue()?;
        if luma_denom > 7 || chroma_denom > 7 {
            return Err(VaError::Corrupt("h264: weight denom out of range"));
        }
        for _ in 0..=num_l0 {
            if br.flag()? {
                lw_flag = true;
                lw.push(((1 << luma_denom) + br.se()?) as i16);
                lo.push(br.se()? as i16);
            } else {
                lw.push((1 << luma_denom) as i16);
                lo.push(0);
            }
            if br.flag()? {
                cw_flag = true;
                let mut w = [0i16; 2];
                let mut o = [0i16; 2];
                for j in 0..2 {
                    w[j] = ((1 << chroma_denom) + br.se()?) as i16;
                    o[j] = br.se()? as i16;
                }
                cw.push(w);
                co.push(o);
            } else {
                cw.push([(1 << chroma_denom) as i16; 2]);
                co.push([0, 0]);
            }
        }
    }
    let (mut no_output, mut idr_long, mut sliding, mut mmcos) =
        (false, false, true, Vec::new());
    if nal_ref_idc != 0 {
        if is_idr {
            no_output = br.flag()?;
            idr_long = br.flag()?;
        } else {
            if br.flag()? {
                sliding = false;
                loop {
                    match br.ue()? {
                        0 => {
                            mmcos.push(Mmco::End);
                            break;
                        }
                        1 => mmcos.push(Mmco::MarkShortUnused { diff: br.ue()? }),
                        2 => mmcos.push(Mmco::MarkLongUnused { idx: br.ue()? }),
                        3 => {
                            let diff = br.ue()?;
                            let idx = br.ue()?;
                            mmcos.push(Mmco::MarkLong { diff, idx });
                        }
                        4 => mmcos.push(Mmco::SetMaxLong { max: br.ue()? }),
                        5 => mmcos.push(Mmco::MarkAllUnused),
                        _ => return Err(VaError::Corrupt("h264: bad MMCO")),
                    }
                }
            }
        }
    }
    let mut cabac_init_idc = 0;
    if pps.entropy_coding_mode_flag && slice_type != 2 {
        cabac_init_idc = br.ue()?;
        if cabac_init_idc > 2 {
            return Err(VaError::Corrupt("h264: bad cabac_init_idc"));
        }
    }
    let slice_qp_delta = br.se()?;
    let (mut disable_idc, mut alpha, mut beta) = (0u32, 0i32, 0i32);
    if pps.deblocking_filter_control_present_flag {
        disable_idc = br.ue()?;
        if disable_idc > 2 {
            return Err(VaError::Corrupt("h264: bad deblock idc"));
        }
        if disable_idc != 1 {
            alpha = br.se()?;
            beta = br.se()?;
        }
    }
    // slice_data_bit_offset excludes CABAC alignment (shim-verified: ffmpeg
    // sends 34, not the aligned 40, for the fixture IDR).
    let bit_offset = br.pos() as u32;
    if pps.entropy_coding_mode_flag {
        br.align_ones()?;
    }
    Ok((
        SliceHeader {
            first_mb_in_slice,
            slice_type,
            pps_id,
            frame_num,
            nal_ref_idc,
            is_idr,
            idr_pic_id,
            poc_lsb,
            delta_poc_bottom,
            num_ref_idx_l0_active_minus1: num_l0,
            list_mod_l0,
            luma_log2_weight_denom: luma_denom,
            chroma_log2_weight_denom: chroma_denom,
            lw_flag,
            lw,
            lo,
            cw_flag,
            cw,
            co,
            no_output_of_prior_pics: no_output,
            idr_long_term: idr_long,
            sliding_window: sliding,
            mmcos,
            cabac_init_idc,
            slice_qp_delta,
            disable_deblocking_filter_idc: disable_idc,
            slice_alpha_c0_offset_div2: alpha,
            slice_beta_offset_div2: beta,
            bit_offset,
        },
        pps_id,
    ))
}

// ── Decoded picture buffer ──────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
enum RefKind {
    Short,
    Long(u32),
}
struct DpbEntry {
    pool_idx: usize,
    unwrap: i64,
    frame_num: u32,
    poc: i32,
    kind: RefKind,
}

struct Picture {
    slices: Vec<(SliceHeader, Vec<u8>)>, // header + original NAL bytes (EBSP)
    frame_num: u32,
    is_idr: bool,
}
/// Zeroed VA struct: repr(C) padding bytes must not carry stack garbage
/// (shim diff caught 0xeb/0x2f in slice-param pads).
fn bytes_zeroed<T>() -> T {
    // SAFETY: all VA structs are plain ints/byte arrays (no invalid patterns).
    unsafe { std::mem::zeroed() }
}

fn invalid_pic() -> VaPictureH264 {
    VaPictureH264 {
        picture_id: VA_INVALID_SURFACE,
        frame_idx: 0,
        flags: VA_PICTURE_H264_INVALID,
        top_field_order_cnt: 0,
        bottom_field_order_cnt: 0,
        va_reserved: [0; 4],
    }
}

struct VideoDecoder<'a> {
    sess: &'a crate::VaSession,
    lib: &'a VaLib,
    dpy: VaDisplay,
    ctx: u32,
    pool: Vec<u32>,
    width: u32,
    height: u32,
    spss: Vec<Option<Sps>>,
    ppss: Vec<Option<Pps>>,
    dpb: Vec<DpbEntry>,
    prev_frame_num: u32,
    frame_num_offset: i64,
    prev_poc_msb: i32,
    prev_poc_lsb: u32,
    max_poc_lsb: u32,
    poc_type: u32,
    first_picture: bool,
}

impl<'a> VideoDecoder<'a> {
    fn fail(&self, op: &'static str, code: i32) -> VaError {
        VaError::Status {
            op,
            code,
            msg: self.lib.error_str(code),
        }
    }

    /// POC for the picture's first slice (mirrors `ff_h264_init_poc`;
    /// frame-only, so top == bottom).
    fn compute_poc(&mut self, sps: &Sps, sh: &SliceHeader) -> Result<i32, VaError> {
        let max_fn = sps.max_frame_num() as i64;
        if sh.is_idr {
            self.frame_num_offset = 0;
            self.prev_poc_msb = 0;
            self.prev_poc_lsb = 0;
        } else if (sh.frame_num as i64) < self.prev_frame_num as i64 {
            self.frame_num_offset += max_fn;
        }
        let poc = match sps.poc_type {
            0 => {
                let max_lsb = 1 << (sps.log2_max_poc_lsb_minus4 + 4);
                let msb = if sh.poc_lsb < self.prev_poc_lsb
                    && self.prev_poc_lsb - sh.poc_lsb >= max_lsb / 2
                {
                    self.prev_poc_msb + max_lsb as i32
                } else if sh.poc_lsb > self.prev_poc_lsb
                    && (sh.poc_lsb - self.prev_poc_lsb) as i64
                        > max_lsb as i64 / 2
                {
                    // Matches FFmpeg's signed-difference form.
                    self.prev_poc_msb - max_lsb as i32
                } else {
                    self.prev_poc_msb
                };
                self.prev_poc_msb = msb;
                self.prev_poc_lsb = sh.poc_lsb;
                msb + sh.poc_lsb as i32
            }
            2 => {
                let mut p =
                    2 * (self.frame_num_offset + sh.frame_num as i64);
                if sh.nal_ref_idc == 0 {
                    p -= 1;
                }
                p as i32
            }
            _ => return Err(VaError::Unsupported("h264: pic_order_cnt_type 1")),
        };
        self.prev_frame_num = sh.frame_num;
        Ok(poc)
    }

    fn active_sps_pps(&self, sh: &SliceHeader) -> Result<(&Sps, &Pps), VaError> {
        let pps = self
            .ppss
            .get(sh.pps_id as usize)
            .and_then(|p| p.as_ref())
            .ok_or(VaError::Corrupt("h264: slice refers to unknown PPS"))?;
        let sps = self
            .spss
            .get(pps.sps_id as usize)
            .and_then(|s| s.as_ref())
            .ok_or(VaError::Corrupt("h264: PPS refers to unknown SPS"))?;
        Ok((sps, pps))
    }

    /// Surfaces holding references, most-recent first (short-term by
    /// descending PicNum, then long-term by ascending index) — the default
    /// RefPicList0 order (spec 8.2.4.2) and FFmpeg's DPB fill order.
    fn ordered_refs(&self) -> Vec<&DpbEntry> {
        let mut shorts: Vec<&DpbEntry> = self
            .dpb
            .iter()
            .filter(|e| e.kind == RefKind::Short)
            .collect();
        shorts.sort_by_key(|e| -e.unwrap);
        let mut longs: Vec<&DpbEntry> = self
            .dpb
            .iter()
            .filter(|e| matches!(e.kind, RefKind::Long(_)))
            .collect();
        longs.sort_by_key(|e| match e.kind {
            RefKind::Long(i) => i,
            _ => 0,
        });
        shorts.into_iter().chain(longs).collect()
    }

    fn entry_pic(&self, e: &DpbEntry) -> VaPictureH264 {
        VaPictureH264 {
            picture_id: self.pool[e.pool_idx],
            frame_idx: e.frame_num,
            flags: match e.kind {
                RefKind::Short => VA_PICTURE_H264_SHORT_TERM_REFERENCE,
                RefKind::Long(_) => VA_PICTURE_H264_LONG_TERM_REFERENCE,
            },
            top_field_order_cnt: e.poc,
            bottom_field_order_cnt: e.poc,
            va_reserved: [0; 4],
        }
    }

    fn build_pic_param(
        &self,
        sps: &Sps,
        pps: &Pps,
        target: u32,
        frame_num: u32,
        poc: i32,
        is_ref: bool,
        _is_long: bool,
    ) -> VaPicParamH264 {
        let mut rf = [invalid_pic(); 16];
        for (i, e) in self.ordered_refs().iter().take(16).enumerate() {
            rf[i] = self.entry_pic(e);
        }
        let log2_max_poc = if sps.poc_type == 0 {
            sps.log2_max_poc_lsb_minus4 & 15
        } else {
            // FFmpeg leaves log2_max_poc_lsb zero for poc_type != 0, so its
            // `(0 - 4)` wraps to 0xC in the 4-bit field (shim-verified 0x03
            // at seq_fields+2). Replicate exactly.
            12
        };
        let seq_fields: u32 = (sps.chroma_format_idc & 3)
            | ((sps.gaps_flag as u32) << 3)
            | (1 << 4) // frame_mbs_only_flag (experiment scope)
            | ((sps.direct_8x8_inference_flag as u32) << 6)
            | (((sps.level_idc >= 31) as u32) << 7) // MinLumaBiPredSize8x8 (A.3.3.2)
            | ((sps.log2_max_frame_num_minus4 & 15) << 8)
            | ((sps.poc_type & 3) << 12)
            | ((log2_max_poc & 15) << 14);
        let pic_fields: u32 = (pps.entropy_coding_mode_flag as u32)
            | ((pps.weighted_pred_flag as u32) << 1)
            | ((pps.weighted_bipred_idc & 3) << 2)
            | ((pps.transform_8x8_mode_flag as u32) << 4)
            | ((pps.constrained_intra_pred_flag as u32) << 6)
            | ((pps.pic_order_present_flag as u32) << 7)
            | ((pps.deblocking_filter_control_present_flag as u32) << 8)
            | ((pps.redundant_pic_cnt_present_flag as u32) << 9)
            | ((is_ref as u32) << 10);
        let mut p: VaPicParamH264 = bytes_zeroed();
        // Matches ffmpeg start_frame: CurrPic carries no reference flags
        // (Mesa only reads BOTTOM_FIELD from CurrPic).
        p.curr_pic = VaPictureH264 {
            picture_id: target,
            frame_idx: frame_num,
            flags: 0,
            top_field_order_cnt: poc,
            bottom_field_order_cnt: poc,
            va_reserved: [0; 4],
        };
        p.reference_frames = rf;
        p.picture_width_in_mbs_minus1 = sps.pic_width_in_mbs_minus1 as u16;
        p.picture_height_in_mbs_minus1 = sps.pic_height_in_map_units_minus1 as u16;
        p.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8;
        p.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8;
        p.num_ref_frames = sps.max_num_ref_frames as u8;
        p.seq_fields = seq_fields;
        p.pic_init_qp_minus26 = pps.pic_init_qp_minus26 as i8;
        p.pic_init_qs_minus26 = pps.pic_init_qs_minus26 as i8;
        p.chroma_qp_index_offset = pps.chroma_qp_index_offset as i8;
        p.second_chroma_qp_index_offset = pps.second_chroma_qp_index_offset as i8;
        p.pic_fields = pic_fields;
        p.frame_num = frame_num as u16;
        p
    }
    fn build_slice_param(&self, sh: &SliceHeader) -> Result<VaSliceParamH264, VaError> {
        if sh.list_mod_l0 {
            return Err(VaError::Unsupported("h264: ref pic list modification"));
        }
        let mut l0 = [invalid_pic(); 32];
        let l1 = [invalid_pic(); 32];
        let (mut n_l0, _n_l1) = (0u32, 0u32);
        if sh.slice_type == 0 {
            let refs = self.ordered_refs();
            n_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
            if refs.len() < n_l0 as usize {
                return Err(VaError::Corrupt("h264: DPB underflow for RefPicList0"));
            }
            for (i, e) in refs.iter().take(n_l0 as usize).enumerate() {
                l0[i] = self.entry_pic(e);
            }
        }
        // Inferred weight-table defaults (spec 7.4.3.2), mirroring FFmpeg's
        // `fill_vaapi_plain_pred_weight_table`.
        let mut lw = [0i16; 32];
        let mut lo = [0i16; 32];
        let mut cw = [[0i16; 2]; 32];
        let mut co = [[0i16; 2]; 32];
        for i in 0..n_l0 as usize {
            if sh.lw_flag {
                lw[i] = sh.lw[i];
                lo[i] = sh.lo[i];
            } else {
                lw[i] = (1 << sh.luma_log2_weight_denom) as i16;
            }
            if sh.cw_flag {
                cw[i] = sh.cw[i];
                co[i] = sh.co[i];
            } else {
                cw[i] = [(1 << sh.chroma_log2_weight_denom) as i16; 2];
            }
        }
        if sh.bit_offset > u16::MAX as u32 {
            return Err(VaError::Corrupt("h264: slice header too long"));
        }
        let mut s: VaSliceParamH264 = bytes_zeroed();
        s.slice_data_size = 0; // filled by the caller from the NAL length
        s.slice_data_flag = VA_SLICE_DATA_FLAG_ALL;
        s.slice_data_bit_offset = sh.bit_offset as u16;
        s.first_mb_in_slice = sh.first_mb_in_slice.min(0xffff) as u16;
        s.slice_type = sh.slice_type;
        s.num_ref_idx_l0_active_minus1 = if sh.slice_type == 0 {
            sh.num_ref_idx_l0_active_minus1 as u8
        } else {
            0
        };
        s.cabac_init_idc = sh.cabac_init_idc as u8;
        s.slice_qp_delta = sh.slice_qp_delta as i8;
        s.disable_deblocking_filter_idc = sh.disable_deblocking_filter_idc as u8;
        s.slice_alpha_c0_offset_div2 = sh.slice_alpha_c0_offset_div2 as i8;
        s.slice_beta_offset_div2 = sh.slice_beta_offset_div2 as i8;
        s.ref_pic_list0 = l0;
        s.ref_pic_list1 = l1;
        s.luma_log2_weight_denom = sh.luma_log2_weight_denom as u8;
        s.chroma_log2_weight_denom = sh.chroma_log2_weight_denom as u8;
        s.luma_weight_l0_flag = sh.lw_flag as u8;
        s.luma_weight_l0 = lw;
        s.luma_offset_l0 = lo;
        s.chroma_weight_l0_flag = sh.cw_flag as u8;
        s.chroma_weight_l0 = cw;
        s.chroma_offset_l0 = co;
        Ok(s)
    }

    /// Decode one picture into `target`: begin/render(all buffers)/end/sync.
    fn submit_picture(
        &self,
        pic: &Picture,
        target: u32,
        poc: i32,
    ) -> Result<(), VaError> {
        let mut bufs: Vec<u32> = Vec::with_capacity(2 + 2 * pic.slices.len());
        let new_buf = |ty: i32, data: *mut c_void, size: u32| -> Result<u32, VaError> {
            let mut id = 0;
            let st = unsafe {
                (self.lib.va_create_buffer)(self.dpy, self.ctx, ty, size, 1, data, &mut id)
            };
            if st != VA_STATUS_SUCCESS {
                Err(self.fail("vaCreateBuffer(h264)", st))
            } else {
                Ok(id)
            }
        };
        let (sh0, _) = &pic.slices[0];
        let (sps, pps) = self.active_sps_pps(sh0)?;
        if pic.slices[1..]
            .iter()
            .any(|(s, _)| s.frame_num != sh0.frame_num || s.is_idr != sh0.is_idr)
        {
            return Err(VaError::Corrupt("h264: mixed picture slices"));
        }
        let is_ref = sh0.nal_ref_idc != 0;
        let is_long = sh0.is_idr && sh0.idr_long_term;
        let mut param = self.build_pic_param(sps, pps, target, sh0.frame_num, poc, is_ref, is_long);
        let mut iq = VaIqMatrixH264 {
            scaling_list_4x4: pps.scaling4,
            scaling_list_8x8: [pps.scaling8[0], pps.scaling8[3]],
            va_reserved: [0; 4],
        };
        // Order mirrors vaapi_h264: pic param, IQ, then per-slice pairs.
        let r = (|| -> Result<(), VaError> {
            bufs.push(new_buf(
                VA_PIC_PARAM_TYPE,
                (&mut param as *mut VaPicParamH264).cast(),
                std::mem::size_of::<VaPicParamH264>() as u32,
            )?);
            bufs.push(new_buf(
                VA_IQ_MATRIX_TYPE,
                (&mut iq as *mut VaIqMatrixH264).cast(),
                std::mem::size_of::<VaIqMatrixH264>() as u32,
            )?);
            let mut params = Vec::with_capacity(pic.slices.len());
            for (sh, _) in &pic.slices {
                params.push(self.build_slice_param(sh)?);
            }
            for (i, (_, nal)) in pic.slices.iter().enumerate() {
                params[i].slice_data_size = nal.len() as u32;
                bufs.push(new_buf(
                    VA_SLICE_PARAM_TYPE,
                    (&mut params[i] as *mut VaSliceParamH264).cast(),
                    std::mem::size_of::<VaSliceParamH264>() as u32,
                )?);
                bufs.push(new_buf(
                    VA_SLICE_DATA_TYPE,
                    nal.as_ptr() as *mut c_void,
                    nal.len() as u32,
                )?);
            }
            if std::env::var_os("HIPFIRE_VCN_DUMP_BUFS").is_some() {
                use std::fmt::Write as _;
                let mut s = String::new();
                let hex = |b: &[u8], s: &mut String| {
                    for x in b {
                        write!(s, "{x:02x}").unwrap();
                    }
                };
                let ps = unsafe {
                    std::slice::from_raw_parts(
                        (&param as *const VaPicParamH264) as *const u8,
                        std::mem::size_of::<VaPicParamH264>(),
                    )
                };
                s.push_str("PIC ");
                hex(ps, &mut s);
                s.push('\n');
                let iq = unsafe {
                    std::slice::from_raw_parts(
                        (&iq as *const VaIqMatrixH264) as *const u8,
                        std::mem::size_of::<VaIqMatrixH264>(),
                    )
                };
                s.push_str("IQ ");
                hex(iq, &mut s);
                s.push('\n');
                for (i, pm) in params.iter().enumerate() {
                    let bs = unsafe {
                        std::slice::from_raw_parts(
                            (pm as *const VaSliceParamH264) as *const u8,
                            std::mem::size_of::<VaSliceParamH264>(),
                        )
                    };
                    s.push_str(&format!("SLICE{i} "));
                    hex(bs, &mut s);
                    s.push('\n');
                    let nal = &pic.slices[i].1;
                    let mut h = 0u32;
                    for b in nal {
                        h = h.wrapping_mul(31).wrapping_add(*b as u32);
                    }
                    s.push_str(&format!("DATA{i} HASH({})={h:08x}\n", nal.len()));
                }
                std::fs::write("/tmp/vamine.log", s).ok();
            }
            let mut st = unsafe { (self.lib.va_begin_picture)(self.dpy, self.ctx, target) };
            if st == VA_STATUS_SUCCESS {
                st = unsafe {
                    (self.lib.va_render_picture)(
                        self.dpy,
                        self.ctx,
                        bufs.as_mut_ptr(),
                        bufs.len() as i32,
                    )
                };
            }
            if st == VA_STATUS_SUCCESS {
                st = unsafe { (self.lib.va_end_picture)(self.dpy, self.ctx) };
            }
            if st != VA_STATUS_SUCCESS {
                return Err(self.fail("submit(begin/render/end)Picture(h264)", st));
            }
            st = unsafe { (self.lib.va_sync_surface)(self.dpy, target) };
            if st != VA_STATUS_SUCCESS {
                return Err(self.fail("vaSyncSurface(h264)", st));
            }
            Ok(())
        })();
        for b in bufs {
            unsafe {
                (self.lib.va_destroy_buffer)(self.dpy, b);
            }
        }
        r
    }

    /// Apply reference marking for a decoded picture (spec 8.2.5, frames).
    fn apply_marking(
        &mut self,
        pool_idx: usize,
        sh: &SliceHeader,
        unwrap: i64,
        poc: i32,
        max_refs: u32,
    ) -> Result<(), VaError> {
        if sh.nal_ref_idc == 0 && !sh.is_idr {
            return Ok(());
        }
        if sh.is_idr {
            // Decoded order == display order (-bf 0): earlier pictures were
            // already consumed, so eviction needs no output step.
            self.dpb.clear();
            if sh.no_output_of_prior_pics {
                // Nothing further: DPB already drained.
            }
            if sh.idr_long_term {
                self.dpb.push(DpbEntry {
                    pool_idx,
                    unwrap,
                    frame_num: sh.frame_num,
                    poc,
                    kind: RefKind::Long(0),
                });
            } else {
                self.dpb.push(DpbEntry {
                    pool_idx,
                    unwrap,
                    frame_num: sh.frame_num,
                    poc,
                    kind: RefKind::Short,
                });
            }
            return Ok(());
        }
        if sh.sliding_window {
            let nref = self.dpb.len() as u32;
            if nref >= max_refs.max(1) {
                // Unmark the short-term entry with the smallest FrameNum.
                let mut victim: Option<usize> = None;
                for (i, e) in self.dpb.iter().enumerate() {
                    if e.kind == RefKind::Short
                        && victim.map(|v| e.unwrap < self.dpb[v].unwrap).unwrap_or(true)
                    {
                        victim = Some(i);
                    }
                }
                if let Some(v) = victim {
                    self.dpb.remove(v);
                }
            }
            self.dpb.push(DpbEntry {
                pool_idx,
                unwrap,
                frame_num: sh.frame_num,
                poc,
                kind: RefKind::Short,
            });
            return Ok(());
        }
        // Adaptive marking (MMCO). Ops 3/4 (long-term conversion) are out of
        // experiment scope; 1/2/5 cover sliding-equivalent streams.
        let mut saw_op5 = false;
        for op in &sh.mmcos {
            match *op {
                Mmco::End => break,
                Mmco::MarkShortUnused { diff } => {
                    let pic_num = unwrap - diff as i64;
                    self.dpb.retain(|e| {
                        !(e.kind == RefKind::Short && e.unwrap == pic_num)
                    });
                }
                Mmco::MarkLongUnused { idx } => {
                    self.dpb
                        .retain(|e| e.kind != RefKind::Long(idx));
                }
                Mmco::MarkLong { .. } | Mmco::SetMaxLong { .. } => {
                    return Err(VaError::Unsupported("h264: long-term MMCO"));
                }
                Mmco::MarkAllUnused => {
                    self.dpb.clear();
                    saw_op5 = true;
                }
            }
        }
        if !saw_op5 {
            self.dpb.push(DpbEntry {
                pool_idx,
                unwrap,
                frame_num: sh.frame_num,
                poc,
                kind: RefKind::Short,
            });
        }
        Ok(())
    }
}

// ── Session entry points ────────────────────────────────────────────────────
impl crate::VaSession {
    /// Open a VA display + VLD config for a video profile
    /// (`VA_PROFILE_H264_HIGH = 7`, `VA_PROFILE_HEVC_MAIN = 17`,
    /// `VA_PROFILE_AV1_PROFILE0 = 32`). Generalises `open()` beyond JPEG.
    pub fn open_video(profile: i32) -> Result<Self, VaError> {
        Self::open_profile(profile)
    }

    /// Decode a complete Annex-B H.264 stream; one derived NV12 frame per
    /// picture, in decode (= display, `-bf 0`) order.
    pub fn decode_h264_annexb(&self, annexb: &[u8]) -> Result<Vec<DerivedFrame>, VaError> {
        let frames = self.decode_h264_collect(annexb, false)?;
        Ok(frames.derived)
    }

    /// Decode a complete Annex-B H.264 stream; one zero-copy dma-buf export +
    /// HIP import per picture, in decode order.
    pub fn decode_h264_annexb_zc(&self, annexb: &[u8]) -> Result<Vec<VcnFrame>, VaError> {
        let frames = self.decode_h264_collect(annexb, true)?;
        Ok(frames.zc)
    }

    fn decode_h264_collect(
        &self,
        annexb: &[u8],
        zerocopy: bool,
    ) -> Result<CollectedFrames, VaError> {
        let lib = &self.lib;
        let fail = |op: &'static str, code: i32| VaError::Status {
            op,
            code,
            msg: lib.error_str(code),
        };
        // ── pass 1: parameter sets + picture boundaries ──
        let mut spss: Vec<Option<Sps>> = vec![None; 32];
        let mut ppss: Vec<Option<Pps>> = vec![None; 256];
        let mut pictures: Vec<Picture> = Vec::new();
        let mut pending: Vec<(SliceHeader, Vec<u8>)> = Vec::new();
        let flush = |pending: &mut Vec<(SliceHeader, Vec<u8>)>,
                         pictures: &mut Vec<Picture>| {
            if pending.is_empty() {
                return;
            }
            let frame_num = pending[0].0.frame_num;
            let is_idr = pending[0].0.is_idr;
            pictures.push(Picture {
                slices: std::mem::take(pending),
                frame_num,
                is_idr,
            });
        };
        for nal in split_annexb(annexb) {
            match nal[0] & 0x1f {
                7 => {
                    flush(&mut pending, &mut pictures);
                    let (id, sps) = parse_sps(nal)?;
                    if id > 31 {
                        return Err(VaError::Corrupt("h264: SPS id too large"));
                    }
                    spss[id as usize] = Some(sps);
                }
                8 => {
                    flush(&mut pending, &mut pictures);
                    let (id, pps) = parse_pps(nal, &spss)?;
                    if id > 255 {
                        return Err(VaError::Corrupt("h264: PPS id too large"));
                    }
                    ppss[id as usize] = Some(pps);
                }
                1 | 5 => {
                    let (sh, _) = parse_slice(nal, &ppss, &spss)?;
                    if sh.first_mb_in_slice == 0 && !pending.is_empty() {
                        flush(&mut pending, &mut pictures);
                    }
                    pending.push((sh, nal.to_vec()));
                }
                _ => {
                    // SEI (6), AUD (9), filler (12): picture boundary, skip.
                    flush(&mut pending, &mut pictures);
                }
            }
        }
        flush(&mut pending, &mut pictures);
        if pictures.is_empty() {
            return Err(VaError::Corrupt("h264: no pictures found"));
        }
        // ── size the pool from the first SPS ──
        let first_sh = &pictures[0].slices[0].0;
        let (sps0, _) = {
            let pps = ppss
                .get(first_sh.pps_id as usize)
                .and_then(|p| p.as_ref())
                .ok_or(VaError::Corrupt("h264: slice refers to unknown PPS"))?;
            let sps = spss
                .get(pps.sps_id as usize)
                .and_then(|s| s.as_ref())
                .ok_or(VaError::Corrupt("h264: PPS refers to unknown SPS"))?;
            (sps.clone(), pps.clone())
        };
        let (dw, dh) = sps0.display_size();
        let max_refs = sps0.max_num_ref_frames.max(1);
        let pool_n = (max_refs + 3).min(16) as usize;
        // ── pool surfaces + one context ──
        let want_linear =
            hipfire_config::developer_var("HIPFIRE_VCN_LINEAR").is_ok();
        let mut mods = [crate::ffi::DRM_FORMAT_MOD_LINEAR];
        let mut mod_list = crate::ffi::VaDrmFormatModifierList {
            num_modifiers: 1,
            modifiers: mods.as_mut_ptr(),
        };
        let mut attribs = [crate::ffi::VaSurfaceAttrib {
            ty: crate::ffi::VA_SURFACE_ATTRIB_DRM_FORMAT_MODIFIERS,
            flags: crate::ffi::VA_SURFACE_ATTRIB_SETTABLE,
            value: crate::ffi::VaGenericValue {
                ty: crate::ffi::VA_GENERIC_VALUE_TYPE_POINTER,
                value: crate::ffi::VaGenericValueUnion {
                    p: (&mut mod_list as *mut crate::ffi::VaDrmFormatModifierList).cast(),
                },
            },
        }];
        let (attr_ptr, attr_n): (*mut c_void, u32) = if want_linear {
            (attribs.as_mut_ptr().cast(), 1)
        } else {
            (std::ptr::null_mut(), 0)
        };
        let mut pool = vec![0u32; pool_n];
        let mut st = unsafe {
            (lib.va_create_surfaces)(
                self.dpy,
                VA_RT_FORMAT_YUV420,
                sps0.coded_width(),
                sps0.coded_height(),
                pool.as_mut_ptr(),
                pool_n as u32,
                attr_ptr,
                attr_n,
            )
        };
        if st != VA_STATUS_SUCCESS {
            return Err(fail("vaCreateSurfaces(h264 pool)", st));
        }
        struct PoolGuard<'a> {
            lib: &'a VaLib,
            dpy: VaDisplay,
            pool: Vec<u32>,
        }
        impl Drop for PoolGuard<'_> {
            fn drop(&mut self) {
                unsafe {
                    (self.lib.va_destroy_surfaces)(
                        self.dpy,
                        self.pool.as_mut_ptr(),
                        self.pool.len() as i32,
                    );
                }
            }
        }
        let _pool_guard = PoolGuard {
            lib,
            dpy: self.dpy,
            pool: pool.clone(),
        };
        let mut ctx = 0;
        st = unsafe {
            (lib.va_create_context)(
                self.dpy,
                self.config,
                sps0.coded_width() as i32,
                sps0.coded_height() as i32,
                0,
                pool.as_mut_ptr(),
                pool_n as i32,
                &mut ctx,
            )
        };
        if st != VA_STATUS_SUCCESS {
            return Err(fail("vaCreateContext(h264)", st));
        }
        struct CtxGuard<'a> {
            lib: &'a VaLib,
            dpy: VaDisplay,
            ctx: u32,
        }
        impl Drop for CtxGuard<'_> {
            fn drop(&mut self) {
                unsafe {
                    (self.lib.va_destroy_context)(self.dpy, self.ctx);
                }
            }
        }
        let _ctx_guard = CtxGuard {
            lib,
            dpy: self.dpy,
            ctx,
        };
        // ── pass 2: decode pictures in order ──
        let mut dec = VideoDecoder {
            sess: self,
            lib,
            dpy: self.dpy,
            ctx,
            pool,
            width: dw,
            height: dh,
            spss,
            ppss,
            dpb: Vec::new(),
            prev_frame_num: 0,
            frame_num_offset: 0,
            prev_poc_msb: 0,
            prev_poc_lsb: 0,
            max_poc_lsb: 0,
            poc_type: 2,
            first_picture: true,
        };
        let mut out = CollectedFrames {
            derived: Vec::new(),
            zc: Vec::new(),
        };
        for (pi, pic) in pictures.iter().enumerate() {
            let sh0 = &pic.slices[0].0;
            let sps = dec.active_sps_pps(sh0)?.0.clone();
            let (cw, ch) = (sps.coded_width(), sps.coded_height());
            if cw != sps0.coded_width() || ch != sps0.coded_height() {
                return Err(VaError::Unsupported("h264: mid-stream resolution change"));
            }
            if !sh0.is_idr && !dec.first_picture {
                let max_fn = sps.max_frame_num();
                let expect = (dec.prev_frame_num + 1) % max_fn;
                if sh0.frame_num != expect && !sps.gaps_flag {
                    return Err(VaError::Corrupt("h264: frame_num gap"));
                }
            }
            if sh0.is_idr && sh0.frame_num != 0 {
                return Err(VaError::Corrupt("h264: IDR with frame_num != 0"));
            }
            let poc = dec.compute_poc(&sps, sh0)?;
            let unwrap = dec.frame_num_offset + sh0.frame_num as i64;
            // Free pool surface: synced every picture, so any surface not
            // currently a reference is reusable.
            let live: Vec<usize> = dec.dpb.iter().map(|e| e.pool_idx).collect();
            let slot = (0..dec.pool.len())
                .find(|i| !live.contains(i))
                .ok_or(VaError::Corrupt("h264: DPB pool exhausted"))?;
            let target = dec.pool[slot];
            dec.submit_picture(pic, target, poc)?;
            if zerocopy {
                out.zc.push(self.export_video_frame(
                    target,
                    dw,
                    dh,
                    &dec,
                    pi,
                )?);
            } else {
                out.derived.push(derive_surface(
                    lib,
                    self.dpy,
                    target,
                    dw,
                    dh,
                )?);
            }
            dec.apply_marking(slot, sh0, unwrap, poc, max_refs)?;
            dec.first_picture = false;
        }
        Ok(out)
    }

    /// Export a decoded pool surface as an NV12 dma-buf + HIP mapping.
    /// The mapping holds its own dma-buf reference; the caller must finish
    /// GPU work on frame N before the pool slot is reused (the parity
    /// example syncs per frame).
    fn export_video_frame(
        &self,
        surf: u32,
        width: u32,
        height: u32,
        _dec: &VideoDecoder,
        _pi: usize,
    ) -> Result<VcnFrame, VaError> {
        let lib = &self.lib;
        let fail = |op: &'static str, code: i32| VaError::Status {
            op,
            code,
            msg: lib.error_str(code),
        };
        let mut desc = VaDrmPrimeDescriptor {
            fourcc: 0,
            width: 0,
            height: 0,
            num_objects: 0,
            objects: Default::default(),
            num_layers: 0,
            layers: Default::default(),
        };
        let st = unsafe {
            (lib.va_export_surface_handle)(
                self.dpy,
                surf,
                VA_MEM_TYPE_DRM_PRIME_2,
                VA_EXPORT_SURFACE_READ_ONLY,
                (&mut desc as *mut VaDrmPrimeDescriptor).cast(),
            )
        };
        if st != VA_STATUS_SUCCESS {
            return Err(fail("vaExportSurfaceHandle(h264)", st));
        }
        if desc.num_objects == 0 {
            return Err(VaError::Corrupt("h264 export yielded no objects"));
        }
        let fd = desc.objects[0].fd;
        let size = desc.objects[0].size as usize;
        let mapping = HipMapping::import_dma_buf(fd, size);
        crate::libc_close(fd);
        for i in 1..desc.num_objects.min(4) as usize {
            crate::libc_close(desc.objects[i].fd);
        }
        let mapping = mapping?;
        Ok(VcnFrame {
            width,
            height,
            max_h: 2,
            max_v: 2,
            fourcc: desc.fourcc,
            layers: desc.layers,
            num_layers: desc.num_layers,
            mapping,
        })
    }
}

struct CollectedFrames {
    derived: Vec<DerivedFrame>,
    zc: Vec<VcnFrame>,
}

/// `vaDeriveImage` + map + pack a decoded video surface to NV12 planes.
fn derive_surface(
    lib: &VaLib,
    dpy: VaDisplay,
    surf: u32,
    width: u32,
    height: u32,
) -> Result<DerivedFrame, VaError> {
    let fail = |op: &'static str, code: i32| VaError::Status {
        op,
        code,
        msg: lib.error_str(code),
    };
    let mut img = VaImage {
        image_id: 0,
        format: VaImageFormat {
            fourcc: 0,
            byte_order: 0,
            bits_per_pixel: 0,
            depth: 0,
            red_mask: 0,
            green_mask: 0,
            blue_mask: 0,
            alpha_mask: 0,
            va_reserved: [0; 4],
        },
        buf: 0,
        width: 0,
        height: 0,
        data_size: 0,
        num_planes: 0,
        pitches: [0; 3],
        offsets: [0; 3],
        num_palette_entries: 0,
        entry_bytes: 0,
        component_order: [0; 4],
        va_reserved: [0; 4],
    };
    let mut st = unsafe { (lib.va_derive_image)(dpy, surf, &mut img) };
    if st != VA_STATUS_SUCCESS {
        return Err(fail("vaDeriveImage(h264)", st));
    }
    struct ImgGuard<'a> {
        lib: &'a VaLib,
        dpy: VaDisplay,
        id: u32,
    }
    impl Drop for ImgGuard<'_> {
        fn drop(&mut self) {
            unsafe {
                (self.lib.va_destroy_image)(self.dpy, self.id);
            }
        }
    }
    let _img_guard = ImgGuard {
        lib,
        dpy,
        id: img.image_id,
    };
    if img.format.fourcc != VA_FOURCC_NV12 || img.num_planes != 2 {
        return Err(VaError::Corrupt("h264 derived image is not 2-plane NV12"));
    }
    let mut ptr: *mut c_void = std::ptr::null_mut();
    st = unsafe { (lib.va_map_buffer)(dpy, img.buf, &mut ptr) };
    if st != VA_STATUS_SUCCESS || ptr.is_null() {
        return Err(fail("vaMapBuffer(h264)", st));
    }
    // SAFETY: mapped NV12 bytes; rows copied before unmap.
    let (y, uv) = unsafe {
        let base = ptr as *const u8;
        let (w, h) = (width as usize, height as usize);
        let (yp, up) = (img.pitches[0] as usize, img.pitches[1] as usize);
        let (yo, uo) = (img.offsets[0] as usize, img.offsets[1] as usize);
        let mut y = vec![0u8; w * h];
        for r in 0..h {
            let src = std::slice::from_raw_parts(base.add(yo + r * yp), w);
            y[r * w..(r + 1) * w].copy_from_slice(src);
        }
        let (cw, chh) = ((w + 1) / 2, (h + 1) / 2);
        let mut uv = vec![0u8; cw * chh * 2];
        for r in 0..chh {
            let src = std::slice::from_raw_parts(base.add(uo + r * up), cw * 2);
            uv[r * cw * 2..(r + 1) * cw * 2].copy_from_slice(src);
        }
        (y, uv)
    };
    unsafe {
        (lib.va_unmap_buffer)(dpy, img.buf);
    }
    Ok(DerivedFrame {
        width,
        height,
        fourcc: img.format.fourcc,
        y_pitch: img.pitches[0],
        uv_pitch: img.pitches[1],
        y,
        uv,
    })
}
