// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal GGUF container reader for `--imatrix <file.gguf>` (llama.cpp
//! imatrix files: F32 `.in_sum2` / `.counts` tensors). Metadata values are
//! skipped except `general.alignment`. GGUF *weight* input was removed in
//! 0.5.0, so no tensor dequantizers live here.

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    BF16 = 30,
    Q1_0 = 41,
    Q2_0 = 42,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            30 => Some(Self::BF16),
            41 => Some(Self::Q1_0),
            42 => Some(Self::Q2_0),
            _ => None,
        }
    }

    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::Q1_0 | Self::Q2_0 => 128,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 290,
            Self::Q1_0 => 18,
            Self::Q2_0 => 34,
        }
    }

    pub fn tensor_bytes(self, n: usize) -> usize {
        let bs = self.block_size();
        let nblocks = (n + bs - 1) / bs;
        nblocks * self.block_bytes()
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GgmlType,
    pub offset: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn byte_size(&self) -> usize {
        self.dtype.tensor_bytes(self.numel())
    }
}

pub struct GgufFile {
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: usize,
    mmap: Mmap,
}

impl GgufFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cursor = Cursor::new(&mmap[..]);

        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid GGUF magic: 0x{magic:08x}"),
            ));
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        if !(2..=3).contains(&version) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version: {version}"),
            ));
        }

        let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()? as usize;

        let mut alignment = 32usize;
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut cursor)?;
            let vtype = cursor.read_u32::<LittleEndian>()?;
            if key == "general.alignment" {
                alignment = match vtype {
                    4 => cursor.read_u32::<LittleEndian>()? as usize,
                    5 => cursor.read_i32::<LittleEndian>()? as u32 as usize,
                    10 => cursor.read_u64::<LittleEndian>()? as u32 as usize,
                    _ => {
                        skip_typed_value(&mut cursor, vtype)?;
                        alignment
                    }
                };
            } else {
                skip_typed_value(&mut cursor, vtype)?;
            }
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64::<LittleEndian>()? as usize);
            }
            let dtype_raw = cursor.read_u32::<LittleEndian>()?;
            let dtype = GgmlType::from_u32(dtype_raw).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown GGML type: {dtype_raw}"),
                )
            })?;
            let offset = cursor.read_u64::<LittleEndian>()? as usize;
            tensors.push(TensorInfo {
                name,
                shape,
                dtype,
                offset,
            });
        }

        let pos = cursor.position() as usize;
        let tensor_data_offset = pos.div_ceil(alignment) * alignment;

        Ok(GgufFile {
            tensors,
            tensor_data_offset,
            mmap,
        })
    }

    pub fn tensor_data(&self, info: &TensorInfo) -> &[u8] {
        let start = self.tensor_data_offset + info.offset;
        let end = start + info.byte_size();
        &self.mmap[start..end]
    }
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> io::Result<String> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("invalid UTF-8: {e}")))
}

/// Advance past one metadata value of GGUF type `vtype` without keeping it.
fn skip_typed_value(cursor: &mut Cursor<&[u8]>, vtype: u32) -> io::Result<()> {
    let fixed = match vtype {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    };
    if let Some(n) = fixed {
        cursor.set_position(cursor.position() + n);
        return Ok(());
    }
    match vtype {
        8 => {
            let len = cursor.read_u64::<LittleEndian>()?;
            cursor.set_position(cursor.position() + len);
            Ok(())
        }
        9 => {
            let elem_type = cursor.read_u32::<LittleEndian>()?;
            let count = cursor.read_u64::<LittleEndian>()?;
            for _ in 0..count {
                skip_typed_value(cursor, elem_type)?;
            }
            Ok(())
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown metadata value type: {vtype}"),
        )),
    }
}
