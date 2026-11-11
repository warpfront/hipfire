// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal GGUF file parser. Reads header, metadata, tensor info.
//! Memory-maps the file for zero-copy access to tensor data.

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as LE u32 (bytes: 47 47 55 46)

/// GGML tensor types. Ids and block layouts are checked against upstream
/// `ggml/include/ggml.h` (`enum ggml_type`) and `ggml/src/ggml-common.h`
/// (`block_*` structs and their `static_assert` sizes).
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
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    MXFP4 = 39,
}

impl GgmlType {
    /// Every ggml type id this reader recognises, with its exact block layout.
    /// Any other id (TQ1_0/TQ2_0, NVFP4, Q1_0/Q2_0, the integer types, removed
    /// ids, future ids) is `None` and fails closed.
    ///
    /// Recognised is not the same as admitted by [`GgufFile::open`]: the
    /// pre-existing GGUF consumers (the llama loader, which only has CPU
    /// decoders for a handful of these) keep the original admitted set via
    /// [`GgmlType::legacy_admitted`]; the native GGUF path opens with
    /// [`GgufFile::open_native`] and admits per tensor itself.
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
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            29 => Some(Self::IQ1M),
            30 => Some(Self::BF16),
            39 => Some(Self::MXFP4),
            _ => None,
        }
    }

    /// The ids [`GgufFile::open`] has always admitted (before the ggml-native
    /// carrier): the llama loader's input set. Everything else that
    /// [`GgmlType::from_u32`] recognises is only readable through
    /// [`GgufFile::open_native`].
    pub fn legacy_admitted(self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F16
                | Self::Q4_0
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q8_1
                | Self::Q2K
                | Self::Q3K
                | Self::Q4K
                | Self::Q5K
                | Self::Q6K
                | Self::Q8K
                | Self::BF16
        )
    }

    /// The numeric ggml type id (`enum ggml_type`).
    pub fn ggml_id(self) -> u32 {
        self as u32
    }

    /// Upstream ggml name (`ggml_type_name`), for refusals and `inspect`.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ1S => "IQ1_S",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ3S => "IQ3_S",
            Self::IQ2S => "IQ2_S",
            Self::IQ4XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::IQ1M => "IQ1_M",
            Self::BF16 => "BF16",
            Self::MXFP4 => "MXFP4",
        }
    }

    /// Block size for quantized types (number of elements per block).
    pub fn block_size(self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4NL
            | Self::MXFP4 => 32,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ2S
            | Self::IQ3XXS
            | Self::IQ3S
            | Self::IQ1S
            | Self::IQ1M
            | Self::IQ4XS => 256,
        }
    }

    /// Bytes per block for quantized types (`sizeof(block_*)` upstream).
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 | Self::F64 => 8,
            Self::Q4_0 => 18, // 2 (f16 scale) + 16 (32 x 4-bit)
            Self::Q4_1 => 20, // 2 (f16 scale) + 2 (f16 min) + 16
            Self::Q5_0 => 22, // 2 + 4 (high bits) + 16
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,                   // 2 (f16 scale) + 32 (32 x 8-bit)
            Self::Q8_1 => 36,                   // 2 (f16 d) + 2 (f16 s) + 32 (32 x 8-bit)
            Self::MXFP4 => 17,                  // 1 (E8M0 scale) + 16 (32 x 4-bit)
            Self::Q2K => 2 + 2 + 16 + 64,       // 84: d(2) + dmin(2) + scales(16) + qs(64)
            Self::Q3K => 2 + 32 + 12 + 64,      // 110: d(2) + hmask(32) + scales(12) + qs(64)
            Self::Q4K => 2 + 2 + 12 + 128,      // 144: d(2) + dmin(2) + scales(12) + qs(128)
            Self::Q5K => 2 + 2 + 12 + 128 + 32, // 176: d(2) + dmin(2) + scales(12) + qs(128) + qh(32)
            Self::Q6K => 128 + 64 + 16 + 2,     // 210: ql(128) + qh(64) + scales(16) + d(2)
            Self::Q8K => 4 + 256 + 32,          // 292: d(f32) + qs(256) + bsums(16 x i16)
            Self::IQ2XXS => 2 + 64,             // 66: d(2) + qs(32 x u16)
            Self::IQ2XS => 2 + 64 + 8,          // 74: d(2) + qs(32 x u16) + scales(8)
            Self::IQ2S => 2 + 64 + 8 + 8,       // 82: d(2) + qs(64) + qh(8) + scales(8)
            Self::IQ3XXS => 2 + 96,             // 98: d(2) + qs(96)
            Self::IQ3S => 2 + 64 + 8 + 32 + 4, // 110: d(2) + qs(64) + qh(8) + signs(32) + scales(4)
            Self::IQ1S => 2 + 32 + 16,         // 50: d(2) + qs(32) + qh(16 B = 8 x u16)
            Self::IQ1M => 32 + 16 + 8,         // 56: qs(32) + qh(16) + scales(8); no d field
            Self::IQ4NL => 2 + 16,             // 18: d(2) + qs(16)
            Self::IQ4XS => 2 + 2 + 4 + 128,    // 136: d(2) + scales_h(2) + scales_l(4) + qs(128)
        }
    }

    /// Whether this is a float type (no quantization blocks).
    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::F64)
    }

    /// Bytes needed to store `n` elements.
    pub fn tensor_bytes(self, n: usize) -> usize {
        let bs = self.block_size();
        let nblocks = (n + bs - 1) / bs;
        nblocks * self.block_bytes()
    }

    /// Bytes of one row of `ne0` elements, or `None` when `ne0` is not a whole
    /// number of blocks (ggml stores whole blocks per row; a ragged row is a
    /// damaged or mis-typed tensor, not something to round up).
    pub fn row_bytes(self, ne0: usize) -> Option<usize> {
        let bs = self.block_size();
        if ne0 == 0 || ne0 % bs != 0 {
            return None;
        }
        (ne0 / bs).checked_mul(self.block_bytes())
    }
}

/// Metadata value types.
#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetaValue::U32(v) => Some(*v),
            MetaValue::I32(v) => Some(*v as u32),
            MetaValue::U64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetaValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Any non-negative integer-typed value (every GGUF integer width).
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetaValue::U8(v) => Some(u64::from(*v)),
            MetaValue::U16(v) => Some(u64::from(*v)),
            MetaValue::U32(v) => Some(u64::from(*v)),
            MetaValue::U64(v) => Some(*v),
            MetaValue::I8(v) => u64::try_from(*v).ok(),
            MetaValue::I16(v) => u64::try_from(*v).ok(),
            MetaValue::I32(v) => u64::try_from(*v).ok(),
            MetaValue::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// F32 or F64 value, widened to f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetaValue::F32(v) => Some(f64::from(*v)),
            MetaValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetaValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetaValue]> {
        match self {
            MetaValue::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// Info about a tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GgmlType,
    pub offset: usize, // offset from start of tensor data section
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.dtype.tensor_bytes(self.numel())
    }

    /// Exact stored bytes: `shape[0]` (ggml `ne0`, the contiguous axis) must be
    /// a whole number of blocks; the remaining axes multiply the row count.
    /// Errors name the tensor, its type and its shape.
    pub fn extent_bytes(&self) -> Result<usize, String> {
        let ne0 = self.shape.first().copied().unwrap_or(0);
        let row = self.dtype.row_bytes(ne0).ok_or_else(|| {
            format!(
                "tensor {}: {} row length {} is not a multiple of the {}-element block",
                self.name,
                self.dtype.name(),
                ne0,
                self.dtype.block_size()
            )
        })?;
        self.shape[1..]
            .iter()
            .try_fold(row, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| format!("tensor {}: byte extent overflows", self.name))
    }
}

/// Parsed GGUF file with memory-mapped tensor data.
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: usize,
    mmap: Mmap,
}

impl GgufFile {
    /// Open a GGUF for the pre-existing consumers (the llama loader): tensor
    /// types outside [`GgmlType::legacy_admitted`] fail closed at open, exactly
    /// as before the native carrier existed.
    pub fn open(path: &Path) -> io::Result<Self> {
        Self::open_with(path, false)
    }

    /// Open a GGUF for the native carrier: every [`GgmlType::from_u32`]
    /// recognised type parses (IQ family, MXFP4, ...); an unknown id still
    /// fails closed. Per-tensor admission is the caller's.
    pub fn open_native(path: &Path) -> io::Result<Self> {
        Self::open_with(path, true)
    }

    fn open_with(path: &Path, native: bool) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = Cursor::new(&mmap[..]);

        // Header
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid GGUF magic: 0x{magic:08x}; this is not a GGUF file (or it is corrupt) — re-download it"),
            ));
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        if version < 2 || version > 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version: {version}; hipfire reads GGUF versions 2 and 3 — re-export or re-download the model"),
            ));
        }

        let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()? as usize;

        // Metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut cursor)?;
            let value = read_meta_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        // Tensor info
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64::<LittleEndian>()? as usize);
            }
            let dtype_raw = cursor.read_u32::<LittleEndian>()?;
            let dtype = GgmlType::from_u32(dtype_raw)
                .filter(|t| native || t.legacy_admitted())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown GGML type: {dtype_raw}; this GGUF uses a tensor type hipfire does not support — use a different quantization of the model"),
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

        // Tensor data starts at next alignment boundary
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as usize;

        let pos = cursor.position() as usize;
        let tensor_data_offset = (pos + alignment - 1) / alignment * alignment;

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            tensor_data_offset,
            mmap,
        })
    }

    /// Get raw bytes for a tensor.
    pub fn tensor_data(&self, info: &TensorInfo) -> &[u8] {
        let start = self.tensor_data_offset + info.offset;
        let end = start + info.byte_size();
        &self.mmap[start..end]
    }

    /// Total bytes of the mapped file.
    pub fn file_len(&self) -> usize {
        self.mmap.len()
    }

    /// Raw bytes for a tensor, or `None` when the tensor's extent is
    /// malformed (ragged row) or lies outside the file (truncated download).
    /// Unlike [`GgufFile::tensor_data`] this never panics.
    pub fn try_tensor_data(&self, info: &TensorInfo) -> Option<&[u8]> {
        let bytes = info.extent_bytes().ok()?;
        let start = self.tensor_data_offset.checked_add(info.offset)?;
        let end = start.checked_add(bytes)?;
        self.mmap.get(start..end)
    }

    /// Find a tensor by name.
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get a metadata value.
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.meta(key).and_then(|v| v.as_u32())
    }

    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.meta(key).and_then(|v| v.as_f32())
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.meta(key).and_then(|v| v.as_str())
    }
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> io::Result<String> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("invalid UTF-8: {e}")))
}

fn read_meta_value(cursor: &mut Cursor<&[u8]>) -> io::Result<MetaValue> {
    let vtype = cursor.read_u32::<LittleEndian>()?;
    read_typed_value(cursor, vtype)
}

fn read_typed_value(cursor: &mut Cursor<&[u8]>, vtype: u32) -> io::Result<MetaValue> {
    match vtype {
        0 => Ok(MetaValue::U8(cursor.read_u8()?)),
        1 => Ok(MetaValue::I8(cursor.read_i8()?)),
        2 => Ok(MetaValue::U16(cursor.read_u16::<LittleEndian>()?)),
        3 => Ok(MetaValue::I16(cursor.read_i16::<LittleEndian>()?)),
        4 => Ok(MetaValue::U32(cursor.read_u32::<LittleEndian>()?)),
        5 => Ok(MetaValue::I32(cursor.read_i32::<LittleEndian>()?)),
        6 => Ok(MetaValue::F32(cursor.read_f32::<LittleEndian>()?)),
        7 => Ok(MetaValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(MetaValue::String(read_string(cursor)?)),
        9 => {
            // Array
            let elem_type = cursor.read_u32::<LittleEndian>()?;
            let count = cursor.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_typed_value(cursor, elem_type)?);
            }
            Ok(MetaValue::Array(arr))
        }
        10 => Ok(MetaValue::U64(cursor.read_u64::<LittleEndian>()?)),
        11 => Ok(MetaValue::I64(cursor.read_i64::<LittleEndian>()?)),
        12 => Ok(MetaValue::F64(cursor.read_f64::<LittleEndian>()?)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown metadata value type: {vtype}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(ggml id, name, elements per block, bytes per block)` from upstream
    /// `ggml.h` (`enum ggml_type`) and `ggml-common.h` (`sizeof(block_*)`).
    const TABLE: &[(u32, &str, usize, usize)] = &[
        (0, "F32", 1, 4),
        (1, "F16", 1, 2),
        (2, "Q4_0", 32, 18),
        (3, "Q4_1", 32, 20),
        (6, "Q5_0", 32, 22),
        (7, "Q5_1", 32, 24),
        (8, "Q8_0", 32, 34),
        (9, "Q8_1", 32, 36),
        (10, "Q2_K", 256, 84),
        (11, "Q3_K", 256, 110),
        (12, "Q4_K", 256, 144),
        (13, "Q5_K", 256, 176),
        (14, "Q6_K", 256, 210),
        (15, "Q8_K", 256, 292),
        (16, "IQ2_XXS", 256, 66),
        (17, "IQ2_XS", 256, 74),
        (18, "IQ3_XXS", 256, 98),
        (19, "IQ1_S", 256, 50),
        (20, "IQ4_NL", 32, 18),
        (21, "IQ3_S", 256, 110),
        (22, "IQ2_S", 256, 82),
        (23, "IQ4_XS", 256, 136),
        (29, "IQ1_M", 256, 56),
        (30, "BF16", 1, 2),
        (39, "MXFP4", 32, 17),
    ];

    #[test]
    fn recognised_ids_have_exact_names_and_block_layouts() {
        for &(id, name, elems, bytes) in TABLE {
            let t = GgmlType::from_u32(id).unwrap_or_else(|| panic!("id {id} ({name}) unknown"));
            assert_eq!(t.ggml_id(), id, "{name}");
            assert_eq!(t.name(), name, "id {id}");
            assert_eq!(t.block_size(), elems, "{name} block elems");
            assert_eq!(t.block_bytes(), bytes, "{name} block bytes");
        }
        // Spot-check the bpw the admission plan quotes.
        let bpw = |t: GgmlType| t.block_bytes() as f64 * 8.0 / t.block_size() as f64;
        assert_eq!(bpw(GgmlType::IQ4XS), 4.25);
        assert_eq!(bpw(GgmlType::Q4K), 4.5);
        assert_eq!(bpw(GgmlType::IQ3S), 3.4375);
        assert_eq!(bpw(GgmlType::IQ2XS), 2.3125);
        assert_eq!(bpw(GgmlType::IQ1M), 1.75);
        assert_eq!(bpw(GgmlType::MXFP4), 4.25);
    }

    #[test]
    fn unknown_ids_fail_closed() {
        // Removed/legacy ids, TQ/NVFP4/Q1_0/Q2_0, the integer types and a wild id.
        for id in [
            4, 5, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 255,
        ] {
            assert_eq!(
                GgmlType::from_u32(id),
                None,
                "id {id} must not be recognised"
            );
        }
    }

    #[test]
    fn legacy_admission_is_the_pre_native_set() {
        // 15 ids were admitted before the native carrier; the IQ family and
        // MXFP4 are recognised but not legacy-admitted.
        let legacy: Vec<u32> = TABLE
            .iter()
            .filter(|&&(id, ..)| GgmlType::from_u32(id).unwrap().legacy_admitted())
            .map(|&(id, ..)| id)
            .collect();
        assert_eq!(
            legacy,
            vec![0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 30]
        );
    }

    #[test]
    fn row_bytes_requires_whole_blocks() {
        assert_eq!(GgmlType::Q8_0.row_bytes(64), Some(68));
        assert_eq!(GgmlType::Q8_0.row_bytes(48), None);
        assert_eq!(GgmlType::Q4K.row_bytes(256), Some(144));
        assert_eq!(GgmlType::Q4K.row_bytes(128), None);
        assert_eq!(GgmlType::IQ4XS.row_bytes(5120), Some(20 * 136));
        assert_eq!(GgmlType::F16.row_bytes(7), Some(14));
        assert_eq!(GgmlType::Q4K.row_bytes(0), None);
    }

    /// One-tensor GGUF v3 with no metadata: `ne0 x 1` of `ty`.
    fn one_tensor_file(
        dir: &std::path::Path,
        ty: u32,
        ne0: u64,
        data: usize,
    ) -> std::path::PathBuf {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // tensors
        b.extend_from_slice(&0u64.to_le_bytes()); // kvs
        b.extend_from_slice(&1u64.to_le_bytes());
        b.push(b't');
        b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&ne0.to_le_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // offset
        b.resize(b.len().div_ceil(32) * 32, 0);
        b.resize(b.len() + data, 0);
        let path = dir.join(format!("t-{ty}-{ne0}-{data}.gguf"));
        std::fs::write(&path, b).unwrap();
        path
    }

    #[test]
    fn open_keeps_the_legacy_set_and_open_native_admits_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        // IQ4_XS (23): 1 block of 256 = 136 bytes.
        let p = one_tensor_file(dir.path(), 23, 256, 136);
        let err = GgufFile::open(&p)
            .err()
            .expect("legacy open fails closed on IQ4_XS");
        assert!(err.to_string().contains("unknown GGML type: 23"), "{err}");
        let g = GgufFile::open_native(&p).expect("native open admits IQ4_XS");
        let t = &g.tensors[0];
        assert_eq!(t.dtype, GgmlType::IQ4XS);
        assert_eq!(t.extent_bytes(), Ok(136));
        assert_eq!(g.try_tensor_data(t).map(<[u8]>::len), Some(136));
        // A type both modes admit parses in both.
        let p = one_tensor_file(dir.path(), 12, 256, 144);
        assert!(GgufFile::open(&p).is_ok());
        assert!(GgufFile::open_native(&p).is_ok());
        // An unknown id fails closed in both.
        let p = one_tensor_file(dir.path(), 250, 256, 0);
        assert!(GgufFile::open(&p).is_err());
        let err = GgufFile::open_native(&p).err().expect("unknown id refuses");
        assert!(err.to_string().contains("unknown GGML type: 250"), "{err}");
    }

    #[test]
    fn try_tensor_data_reports_truncation_and_ragged_rows() {
        let dir = tempfile::tempdir().unwrap();
        // Header says IQ4_XS x 256 (136 B) but only 100 bytes follow.
        let p = one_tensor_file(dir.path(), 23, 256, 100);
        let g = GgufFile::open_native(&p).unwrap();
        assert!(g.try_tensor_data(&g.tensors[0]).is_none());
        // 100 elements of Q4_K: not whole blocks.
        let p = one_tensor_file(dir.path(), 12, 100, 144);
        let g = GgufFile::open_native(&p).unwrap();
        let err = g.tensors[0].extent_bytes().unwrap_err();
        assert!(
            err.contains("tensor t") && err.contains("Q4_K") && err.contains("256"),
            "{err}"
        );
        assert!(g.try_tensor_data(&g.tensors[0]).is_none());
    }
}
