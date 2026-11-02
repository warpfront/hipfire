// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use fold_contract_model::half;
use serde_json::Value;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

#[derive(Clone)]
pub struct Tensor {
    pub name: String,
    pub qt: u8,
    pub dims: Vec<usize>,
    pub off: u64,
    pub size: u64,
}
fn take<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> &'a [u8] {
    let value = &bytes[*pos..*pos + n];
    *pos += n;
    value
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
pub fn row(file: &mut File, t: &Tensor, expert: usize, feature: usize) -> Vec<u8> {
    let n = t.dims[t.dims.len() - 2];
    let k = *t.dims.last().unwrap();
    let size = k / 128 * 68;
    let mut bytes = vec![0; size];
    file.seek(SeekFrom::Start(
        t.off + ((expert * n + feature) * size) as u64,
    ))
    .unwrap();
    file.read_exact(&mut bytes).unwrap();
    bytes
}
pub fn decode(t: &Tensor, bytes: &[u8]) -> (Vec<f32>, Vec<i8>) {
    let epochs = bytes.len() / 68;
    let mut scales = Vec::new();
    let mut codes = Vec::new();
    for e in 0..epochs {
        let base = if t.qt == 44 { e / 2 * 136 } else { e * 68 };
        let hdr = base + if t.qt == 44 { 4 * (e % 2) } else { 0 };
        let nib = base + if t.qt == 44 { 8 + 64 * (e % 2) } else { 4 };
        let scale = half(u16::from_le_bytes(bytes[hdr..hdr + 2].try_into().unwrap()));
        let zero = half(u16::from_le_bytes(
            bytes[hdr + 2..hdr + 4].try_into().unwrap(),
        ));
        assert!(scale.is_finite() && scale >= 0.0);
        assert_eq!(zero, -8.0 * scale, "asymmetric {}", t.name);
        scales.push(scale);
        for j in 0..128 {
            let byte = bytes[nib + j / 2];
            codes.push((if j % 2 == 0 { byte & 15 } else { byte >> 4 }) as i8 - 8);
        }
    }
    (scales, codes)
}
pub fn audit(file: &mut File, tensors: &[Tensor], layers: &str) -> Value {
    let mut fallback = 0usize;
    let mut headers = 0usize;
    let mut asymmetric = 0usize;
    let mut per_layer = Vec::new();
    for layer in 0..49 {
        let report: Value =
            serde_json::from_slice(&std::fs::read(format!("{layers}/L{layer}.json")).unwrap())
                .unwrap();
        let prefix = report["prefix"].as_str().unwrap();
        let ids = report["fallback_experts"].as_array().unwrap();
        fallback += ids.len();
        let mut layer_asym = 0;
        let mut layer_headers = 0;
        for expert in ids {
            let expert = expert.as_u64().unwrap() as usize;
            for suffix in ["gate_up_proj", "down_proj"] {
                let name = format!("{prefix}{suffix}");
                let tensor = tensors.iter().find(|t| t.name == name).unwrap();
                let n = tensor.dims[1];
                let epochs = tensor.dims[2] / 128;
                for feature in 0..n {
                    let bytes = row(file, tensor, expert, feature);
                    for e in 0..epochs {
                        let hdr = if tensor.qt == 44 {
                            e / 2 * 136 + 4 * (e % 2)
                        } else {
                            e * 68
                        };
                        let scale =
                            half(u16::from_le_bytes(bytes[hdr..hdr + 2].try_into().unwrap()));
                        let zero = half(u16::from_le_bytes(
                            bytes[hdr + 2..hdr + 4].try_into().unwrap(),
                        ));
                        if !(scale.is_finite() && scale >= 0.0 && zero == -8.0 * scale) {
                            layer_asym += 1;
                        }
                        layer_headers += 1;
                    }
                }
            }
        }
        asymmetric += layer_asym;
        headers += layer_headers;
        per_layer.push(serde_json::json!({"layer":layer,"fallback_experts":ids.len(),"headers":layer_headers,"asymmetric_headers":layer_asym}));
    }
    assert_eq!(fallback, 553);
    serde_json::json!({"fallback_experts":fallback,"headers":headers,"asymmetric_headers":asymmetric,"layers":per_layer})
}
