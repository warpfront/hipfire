// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Baseline-JPEG → VA-buffer parameter extraction.
//!
//! A purpose-built marker walker (SOF0 / DQT / DHT / DRI / SOS + entropy
//! extent). `libjpeg-turbo-rs` 0.8 — the experiment's CPU fallback/oracle —
//! exposes *decoded* Huffman/quant tables through its public API, not the
//! raw DHT bit-lengths / DQT zig-zag bytes the VA buffers require, so the
//! VA fill reads the stream directly while turbo owns the oracle pixels
//! (byte-identical to PIL per `docs/VALIDATION.md`).
//!
//! Scope: sequential baseline (SOF0), 8-bit, 1 or 3 components. Progressive
//! (SOF2), lossless (SOF3), arithmetic coding, 12-bit DQT, and 4-component
//! (CMYK/YCCK) streams are rejected with [`VaError::Unsupported`] — the
//! caller falls back to the turbo CPU path.

use crate::ffi::{
    VaError, VaJpegComponent, VaJpegHuffmanBuffer, VaJpegIQMatrix, VaJpegPicParam,
    VaJpegSliceComponent, VaJpegSliceParam, VA_RT_FORMAT_YUV400, VA_RT_FORMAT_YUV420,
    VA_RT_FORMAT_YUV422, VA_RT_FORMAT_YUV444, VA_SLICE_DATA_FLAG_ALL,
};

/// Everything `VaSession::decode_jpeg` needs to fill the five VA buffers,
/// plus the entropy slice borrowed from the input stream.
pub struct JpegVaParams<'a> {
    pub width: u16,
    pub height: u16,
    pub pic: VaJpegPicParam,
    pub iq: VaJpegIQMatrix,
    pub huff: VaJpegHuffmanBuffer,
    pub slice: VaJpegSliceParam,
    /// Raw entropy-coded segment (restart markers included, EOI excluded).
    pub entropy: &'a [u8],
    /// Max sampling factors (for MCU count + subsampling classification).
    pub max_h: u8,
    pub max_v: u8,
    /// VA render-target format the surface must be created with so the
    /// driver's chroma-format check accepts the stream (see SOF0 arm).
    pub rt_format: u32,
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}
impl<'a> Cursor<'a> {
    fn u8(&mut self) -> Result<u8, VaError> {
        if self.pos >= self.data.len() {
            return Err(VaError::Corrupt("unexpected EOF"));
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn u16(&mut self) -> Result<u16, VaError> {
        let hi = self.u8()? as u16;
        let lo = self.u8()? as u16;
        Ok((hi << 8) | lo)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], VaError> {
        if self.pos + n > self.data.len() {
            return Err(VaError::Corrupt("unexpected EOF in segment"));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    /// Segment length field (includes its own 2 bytes).
    fn seg_len(&mut self) -> Result<usize, VaError> {
        let l = self.u16()? as usize;
        if l < 2 {
            return Err(VaError::Corrupt("bad segment length"));
        }
        Ok(l - 2)
    }
}

/// Parse a baseline JPEG for VA-API submission.
pub fn parse_for_va(data: &[u8]) -> Result<JpegVaParams<'_>, VaError> {
    let mut c = Cursor { data, pos: 0 };
    if c.u8()? != 0xFF || c.u8()? != 0xD8 {
        return Err(VaError::Corrupt("missing SOI"));
    }

    let mut pic = VaJpegPicParam {
        picture_width: 0,
        picture_height: 0,
        components: [VaJpegComponent::default(); 255],
        num_components: 0,
        color_space: 0, // YUV
        rotation: 0,
        crop_x: 0,
        crop_y: 0,
        crop_width: 0,
        crop_height: 0,
        va_reserved: [0; 5],
    };
    let mut iq = VaJpegIQMatrix {
        load_quantiser_table: [0; 4],
        quantiser_table: [[0; 64]; 4],
        va_reserved: [0; 4],
    };
    let mut huff = VaJpegHuffmanBuffer {
        load_huffman_table: [0; 2],
        huffman_table: [crate::ffi::VaJpegHuffmanTable {
            num_dc_codes: [0; 16],
            dc_values: [0; 12],
            num_ac_codes: [0; 16],
            ac_values: [0; 162],
            pad: [0; 2],
        }; 2],
        va_reserved: [0; 4],
    };
    let mut restart_interval: u16 = 0;
    let mut saw_sof = false;
    let mut max_h = 1u8;
    let mut max_v = 1u8;
    let mut rt_format = VA_RT_FORMAT_YUV420;

    loop {
        // Markers: skip fill 0xFF bytes, then marker byte.
        let mut m = c.u8()?;
        if m != 0xFF {
            return Err(VaError::Corrupt("expected marker prefix"));
        }
        loop {
            m = c.u8()?;
            if m != 0xFF {
                break;
            }
        }
        match m {
            0xD8 => continue, // SOI (nested)
            0xD9 => return Err(VaError::Corrupt("EOI before SOS")),
            0x01 | 0xD0..=0xD7 => continue, // TEM / RSTn (standalone)
            0xC0 => {
                // SOF0 baseline
                let seg = c.seg_len()?;
                let end = c.pos + seg;
                let p = c.u8()?;
                if p != 8 {
                    return Err(VaError::Unsupported("only 8-bit baseline"));
                }
                let h = c.u16()?;
                let w = c.u16()?;
                if h == 0 || w == 0 {
                    return Err(VaError::Corrupt("zero SOF dimensions"));
                }
                let nf = c.u8()?;
                if nf != 1 && nf != 3 {
                    return Err(VaError::Unsupported("only 1- or 3-component frames"));
                }
                for i in 0..nf as usize {
                    let id = c.u8()?;
                    let hv = c.u8()?;
                    let tq = c.u8()?;
                    if tq > 3 {
                        return Err(VaError::Corrupt("bad Tqi"));
                    }
                    let h_i = hv >> 4;
                    let v_i = hv & 0x0F;
                    if h_i == 0 || h_i > 4 || v_i == 0 || v_i > 4 {
                        return Err(VaError::Corrupt("bad sampling factors"));
                    }
                    max_h = max_h.max(h_i);
                    max_v = max_v.max(v_i);
                    pic.components[i] = VaJpegComponent {
                        component_id: id,
                        h_sampling_factor: h_i,
                        v_sampling_factor: v_i,
                        quantiser_table_selector: tq,
                    };
                }
                pic.picture_width = w;
                pic.picture_height = h;
                pic.num_components = nf;
                // The VA surface must carry the stream's chroma format:
                // radeonsi's `radeon_dec_jpeg_end_frame` compares the two and
                // refuses a mismatch ("VCN - Decode format check failed").
                // The earlier "4:2:0 only" reading was this check firing on a
                // hard-coded YUV420 surface; rocJPEG decodes 4:4:4 on the same
                // driver by allocating a 444 surface. Classify here, allocate
                // accordingly in `submit`.
                let hv = |i: usize| {
                    (
                        pic.components[i].h_sampling_factor,
                        pic.components[i].v_sampling_factor,
                    )
                };
                let chroma_ok = nf == 1 || (hv(1) == (1, 1) && hv(2) == (1, 1));
                rt_format = match (nf, if nf == 3 { hv(0) } else { (1, 1) }) {
                    (1, _) => VA_RT_FORMAT_YUV400,
                    (3, (2, 2)) if chroma_ok => VA_RT_FORMAT_YUV420,
                    (3, (2, 1)) if chroma_ok => VA_RT_FORMAT_YUV422,
                    (3, (1, 1)) if chroma_ok => VA_RT_FORMAT_YUV444,
                    _ => {
                        return Err(VaError::Unsupported(
                            "sampling factors are not 4:2:0, 4:2:2, 4:4:4 or gray",
                        ))
                    }
                };
                saw_sof = true;
                c.pos = end; // skip any trailing bytes defensively
            }
            0xC2 => return Err(VaError::Unsupported("progressive JPEG (SOF2)")),
            0xC3 => return Err(VaError::Unsupported("lossless JPEG (SOF3)")),
            0xC4 => {
                // DHT
                let seg = c.seg_len()?;
                let end = c.pos + seg;
                while c.pos < end {
                    let tc_th = c.u8()?;
                    let tc = tc_th >> 4;
                    let th = (tc_th & 0x0F) as usize;
                    if th > 1 {
                        return Err(VaError::Unsupported("DHT table id > 1"));
                    }
                    if tc > 1 {
                        return Err(VaError::Corrupt("bad DHT Tc"));
                    }
                    let mut total = 0usize;
                    let mut bits = [0u8; 16];
                    for b in bits.iter_mut() {
                        *b = c.u8()?;
                        total += *b as usize;
                    }
                    let vals = c.bytes(total)?;
                    if tc == 0 {
                        if total > 12 {
                            return Err(VaError::Corrupt("DC table too long"));
                        }
                        huff.huffman_table[th].num_dc_codes = bits;
                        huff.huffman_table[th].dc_values[..total].copy_from_slice(vals);
                    } else {
                        if total > 162 {
                            return Err(VaError::Corrupt("AC table too long"));
                        }
                        huff.huffman_table[th].num_ac_codes = bits;
                        huff.huffman_table[th].ac_values[..total].copy_from_slice(vals);
                    }
                    huff.load_huffman_table[th] = 1;
                }
                c.pos = end;
            }
            0xDB => {
                // DQT
                let seg = c.seg_len()?;
                let end = c.pos + seg;
                while c.pos < end {
                    let pq_tq = c.u8()?;
                    let pq = pq_tq >> 4;
                    let tq = (pq_tq & 0x0F) as usize;
                    if pq != 0 {
                        return Err(VaError::Unsupported("12-bit DQT"));
                    }
                    if tq > 3 {
                        return Err(VaError::Corrupt("bad DQT Tq"));
                    }
                    let q = c.bytes(64)?;
                    iq.quantiser_table[tq].copy_from_slice(q);
                    iq.load_quantiser_table[tq] = 1;
                }
                c.pos = end;
            }
            0xDD => {
                // DRI
                let seg = c.seg_len()?;
                if seg != 2 {
                    return Err(VaError::Corrupt("bad DRI length"));
                }
                restart_interval = c.u16()?;
            }
            0xDA => {
                // SOS — the entropy segment follows the header.
                if !saw_sof {
                    return Err(VaError::Corrupt("SOS before SOF"));
                }
                let seg = c.seg_len()?;
                let end = c.pos + seg;
                let ns = c.u8()?;
                if ns == 0 || ns > 4 {
                    return Err(VaError::Corrupt("bad SOS Ns"));
                }
                let mut slice = VaJpegSliceParam {
                    slice_data_size: 0,
                    slice_data_offset: 0,
                    slice_data_flag: VA_SLICE_DATA_FLAG_ALL,
                    slice_horizontal_position: 0,
                    slice_vertical_position: 0,
                    components: [VaJpegSliceComponent::default(); 4],
                    num_components: ns,
                    _pad: 0,
                    restart_interval,
                    num_mcus: 0,
                    va_reserved: [0; 4],
                };
                for i in 0..ns as usize {
                    let cs = c.u8()?;
                    let td_ta = c.u8()?;
                    slice.components[i] = VaJpegSliceComponent {
                        component_selector: cs,
                        dc_table_selector: td_ta >> 4,
                        ac_table_selector: td_ta & 0x0F,
                    };
                }
                let _ss = c.u8()?;
                let _se = c.u8()?;
                let _ah_al = c.u8()?;
                c.pos = end;
                // Entropy extent: scan with byte-stuffing awareness to EOI.
                let entropy_start = c.pos;
                let entropy_end = scan_entropy_end(data, entropy_start)?;
                let entropy = &data[entropy_start..entropy_end];
                // MCU count from frame sampling.
                let mcu_w = (pic.picture_width as u32 + 8 * max_h as u32 - 1) / (8 * max_h as u32);
                let mcu_h = (pic.picture_height as u32 + 8 * max_v as u32 - 1) / (8 * max_v as u32);
                slice.num_mcus = mcu_w * mcu_h;
                slice.slice_data_size = entropy.len() as u32;
                return Ok(JpegVaParams {
                    width: pic.picture_width,
                    height: pic.picture_height,
                    pic,
                    iq,
                    huff,
                    slice,
                    entropy,
                    max_h,
                    max_v,
                    rt_format,
                });
            }
            0xCC => return Err(VaError::Unsupported("arithmetic coding (DAC)")),
            _ => {
                // APPn / COM / DNL / SOF others with length: skip.
                if (0xE0..=0xEF).contains(&m) || m == 0xFE || (0xC1..=0xCF).contains(&m) {
                    let seg = c.seg_len()?;
                    c.bytes(seg)?;
                } else {
                    return Err(VaError::Corrupt("unknown marker"));
                }
            }
        }
    }
}

/// Find the EOI terminating the entropy segment starting at `from`,
/// honouring `FF 00` stuffing and standalone `RSTn` markers.
fn scan_entropy_end(data: &[u8], from: usize) -> Result<usize, VaError> {
    let mut i = from;
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let m = data[i + 1];
        match m {
            0x00 | 0xD0..=0xD7 => {
                i += 2; // stuffed byte / restart: part of the stream
            }
            0xD9 => return Ok(i), // EOI: entropy ends here
            0xFF => {
                i += 1; // fill byte: re-examine
            }
            _ => {
                // A length-bearing marker (e.g. DQT after SOS in exotic
                // streams) or garbage. Multi-scan baseline is out of scope.
                return Err(VaError::Unsupported("data after first scan"));
            }
        }
    }
    Err(VaError::Corrupt("entropy runs past EOF"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_baseline() -> Vec<u8> {
        // 16x16 gray baseline: SOI + DQT + SOF0 + DHT(DC) + DHT(AC) + SOS +
        // 1 MCU of EOB-only data + EOI. Hand-built; decodes to flat gray.
        let mut v = vec![0xFF, 0xD8];
        // DQT: PqTq=0, 64 bytes of 8
        v.extend([0xFF, 0xDB, 0x00, 0x43, 0x00]);
        v.extend([8u8; 64]);
        // SOF0: P=8, 16x16, Nf=1, C1 H1V1 Tq0
        v.extend([
            0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x10, 0x00, 0x10, 0x01, 0x01, 0x11, 0x00,
        ]);
        // DHT DC table 0: 1 code of length 2 (category 0 => EOB-ish DC diff 0)
        v.extend([0xFF, 0xC4, 0x00, 0x14, 0x00]);
        v.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        v.push(0x00);
        // DHT AC table 0: 1 code of length 2 (EOB=0x00)
        v.extend([0xFF, 0xC4, 0x00, 0x14, 0x10]);
        v.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        v.push(0x00);
        // SOS: Ns=1, Cs=1 TdTa=0, SsSeAhAl=0,63,0
        v.extend([0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]);
        // entropy: DC code '00' (len2) + AC EOB '00' (len2) => 4 bits => 0x3F + pad
        v.extend([0x3F, 0xFF, 0xD9]);
        v
    }

    #[test]
    fn parses_minimal_gray_baseline() {
        let jpeg = minimal_baseline();
        let p = parse_for_va(&jpeg).expect("parse");
        assert_eq!((p.width, p.height), (16, 16));
        assert_eq!(p.pic.num_components, 1);
        assert_eq!(p.slice.num_mcus, 4); // 16x16, H=V=1 -> 2x2 MCUs
        assert_eq!(p.slice.slice_data_size, 1);
        assert_eq!(p.iq.load_quantiser_table[0], 1);
        assert_eq!(p.huff.load_huffman_table[0], 1);
    }

    #[test]
    fn rejects_progressive() {
        let mut jpeg = minimal_baseline();
        // SOF0 marker -> SOF2
        let pos = jpeg.iter().position(|w| *w == 0xC0).unwrap();
        jpeg[pos] = 0xC2;
        // fix: the 0xC0 byte sits right after an 0xFF
        assert!(matches!(parse_for_va(&jpeg), Err(VaError::Unsupported(_))));
    }

    #[test]
    fn parses_baseline_444_barney_cigar() {
        // Committed YUV444 baseline (vcn-zc-444 on hardware parity).
        // MD5 72f3437ea54e23ad2ff724ecef960471; 640×468.
        let jpeg = include_bytes!("../../../benchmarks/vision/images/barney_cigar.jpg");
        let p = parse_for_va(jpeg).expect("444 baseline must parse");
        assert_eq!((p.width, p.height), (640, 468));
        assert_eq!(p.pic.num_components, 3);
        assert_eq!((p.max_h, p.max_v), (1, 1));
        assert_eq!(p.rt_format, VA_RT_FORMAT_YUV444);
    }
}
