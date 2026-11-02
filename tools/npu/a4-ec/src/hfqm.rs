// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! HFQM reader/decoder (copied from tools/npu/fold-model/src/hfqm.rs; MQ4G256V2 qt=44).
use serde_json::Value;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub qt: u8,
    pub dims: Vec<usize>,
    pub off: u64,
    pub size: u64,
}

fn take<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> &'a [u8] {
    let v = &bytes[*pos..*pos + n];
    *pos += n;
    v
}
fn word(bytes: &[u8], pos: &mut usize) -> usize {
    u32::from_le_bytes(take(bytes, pos, 4).try_into().unwrap()) as usize
}

pub fn index(file: &mut File) -> (Value, Vec<Tensor>) {
    let mut header = [0; 32];
    file.read_exact(&mut header).unwrap();
    assert_eq!(&header[..4], b"HFQM");
    assert_eq!(u32::from_le_bytes(header[4..8].try_into().unwrap()), 1);
    let metadata = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let data = u64::from_le_bytes(header[24..32].try_into().unwrap());
    assert!(data - metadata < 32_000_000);
    file.seek(SeekFrom::Start(metadata)).unwrap();
    let mut bytes = vec![0; (data - metadata) as usize];
    file.read_exact(&mut bytes).unwrap();
    let mut stream = serde_json::Deserializer::from_slice(&bytes).into_iter::<Value>();
    let meta = stream.next().unwrap().unwrap();
    let mut pos = stream.byte_offset();
    let count = word(&bytes, &mut pos);
    assert_eq!(
        count,
        u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize
    );
    let mut tensors = Vec::new();
    let mut offset = data;
    for _ in 0..count {
        let n = u16::from_le_bytes(take(&bytes, &mut pos, 2).try_into().unwrap()) as usize;
        let name = String::from_utf8(take(&bytes, &mut pos, n).to_vec()).unwrap();
        let qt = take(&bytes, &mut pos, 1)[0];
        let ndim = take(&bytes, &mut pos, 1)[0];
        let dims = (0..ndim).map(|_| word(&bytes, &mut pos)).collect();
        let _group = word(&bytes, &mut pos);
        let size = u64::from_le_bytes(take(&bytes, &mut pos, 8).try_into().unwrap());
        tensors.push(Tensor {
            name,
            qt,
            dims,
            off: offset,
            size,
        });
        offset += size;
    }
    assert_eq!(offset, file.metadata().unwrap().len());
    (meta, tensors)
}

/// Exact f16 -> f32 widening (via the `half` crate).
pub fn half(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

/// Decode one weight row's raw bytes into per-epoch scales and (u-8) codes.
/// qt==44: 136 B per 256-group; f16 scale at header byte 4*(e&1); nibbles at +8+64*(e&1).
pub fn decode_row(qt: u8, bytes: &[u8], scales: &mut [f32], codes: &mut [i8]) {
    let epochs = bytes.len() / 68;
    for e in 0..epochs {
        let base = if qt == 44 { e / 2 * 136 } else { e * 68 };
        let hdr = base + if qt == 44 { 4 * (e % 2) } else { 0 };
        let nib = base + if qt == 44 { 8 + 64 * (e % 2) } else { 4 };
        let scale = half(u16::from_le_bytes(bytes[hdr..hdr + 2].try_into().unwrap()));
        assert!(scale.is_finite() && scale >= 0.0);
        scales[e] = scale;
        for j in 0..128 {
            let byte = bytes[nib + j / 2];
            codes[e * 128 + j] = (if j % 2 == 0 { byte & 15 } else { byte >> 4 }) as i8 - 8;
        }
    }
}

/// Read a whole tensor [N, K] as dense f32 W'[n][k] = sc_e * (u-8), row-major, parallel.
pub fn read_dense(path: &str, t: &Tensor) -> Vec<f32> {
    use rayon::prelude::*;
    assert!(
        t.qt == 44 || t.qt == 43,
        "unsupported qt {} for {}",
        t.qt,
        t.name
    );
    let n = t.dims[t.dims.len() - 2];
    let k = *t.dims.last().unwrap();
    let rb = k / 128 * 68;
    assert_eq!(t.size as usize, n * rb, "size mismatch {}", t.name);
    let mmap = unsafe { memmap2::Mmap::map(&File::open(path).unwrap()).unwrap() };
    let raw = &mmap[t.off as usize..(t.off + t.size) as usize];
    let mut w = vec![0f32; n * k];
    w.par_chunks_mut(k).enumerate().for_each(|(r, out)| {
        let mut sc = vec![0f32; k / 128];
        let mut cd = vec![0i8; k];
        decode_row(t.qt, &raw[r * rb..(r + 1) * rb], &mut sc, &mut cd);
        for e in 0..k / 128 {
            for j in 0..128 {
                out[e * 128 + j] = sc[e] * cd[e * 128 + j] as f32;
            }
        }
    });
    w
}
