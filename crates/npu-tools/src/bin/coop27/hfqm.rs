// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! HFQM index and raw tensor bytes (port of `tools/npu/a4-ec/src/hfqm.rs::index`, no decoding: the GPU and the NPU packer
//! read the stored MQ4G256V2 rows unchanged).
use serde_json::Value;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

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

pub fn index(file: &mut File) -> Result<Vec<Tensor>, String> {
    let mut header = [0; 32];
    file.read_exact(&mut header)
        .map_err(|e| format!("hfqm header: {e}"))?;
    if &header[..4] != b"HFQM" || u32::from_le_bytes(header[4..8].try_into().unwrap()) != 1 {
        return Err("not an HFQM v1 file".into());
    }
    let metadata = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let data = u64::from_le_bytes(header[24..32].try_into().unwrap());
    if data < metadata || data - metadata >= 32_000_000 {
        return Err("bad HFQM metadata range".into());
    }
    file.seek(SeekFrom::Start(metadata))
        .map_err(|e| e.to_string())?;
    let mut bytes = vec![0; (data - metadata) as usize];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    let mut stream = serde_json::Deserializer::from_slice(&bytes).into_iter::<Value>();
    stream
        .next()
        .ok_or("no HFQM json")?
        .map_err(|e| e.to_string())?;
    let mut pos = stream.byte_offset();
    let count = word(&bytes, &mut pos);
    let mut tensors = Vec::with_capacity(count);
    let mut offset = data;
    for _ in 0..count {
        let n = u16::from_le_bytes(take(&bytes, &mut pos, 2).try_into().unwrap()) as usize;
        let name =
            String::from_utf8(take(&bytes, &mut pos, n).to_vec()).map_err(|e| e.to_string())?;
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
    if offset != file.metadata().map_err(|e| e.to_string())?.len() {
        return Err("HFQM tensor sizes do not cover the file".into());
    }
    Ok(tensors)
}

/// Raw stored bytes of tensor `name`.
pub fn read_tensor(path: &str, name: &str) -> Result<(Tensor, Vec<u8>), String> {
    let mut f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let t = index(&mut f)?
        .into_iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("{name} not in {path}"))?;
    f.seek(SeekFrom::Start(t.off)).map_err(|e| e.to_string())?;
    let mut v = vec![0u8; t.size as usize];
    f.read_exact(&mut v).map_err(|e| e.to_string())?;
    Ok((t, v))
}
