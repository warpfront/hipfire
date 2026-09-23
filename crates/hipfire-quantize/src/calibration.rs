// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.


#![allow(dead_code, unused_imports, unused_variables, non_snake_case, clippy::all)]

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};
use hipfire_quantize::hessian_io;
use crate::e8;
use crate::e8_gptq;
use crate::gguf_input;
use crate::reap_overlay;
use crate::dequant::*;

pub(crate) static IMATRIX: OnceLock<HashMap<String, Vec<f32>>> = OnceLock::new();
pub(crate) static AWQ_ALPHA: OnceLock<f32> = OnceLock::new();
pub(crate) static AWQ_A4_AWARE: OnceLock<bool> = OnceLock::new();
pub(crate) const AWQ_A4_CANDIDATE_ALPHAS: [f32; 5] = [0.35, 0.45, 0.55, 0.65, 0.75];
static AWQ_A4_ALPHA_BY_ACTIVATION: LazyLock<Mutex<HashMap<Vec<u32>, f32>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
/// Opt-in QAT producer rows; absent leaves the historical surrogate unchanged.
pub(crate) static AWQ_A4_SIGNED_CAPTURE: OnceLock<PathBuf> = OnceLock::new();
pub(crate) static AWQ_A4_ROUTE_C2: OnceLock<bool> = OnceLock::new();
pub(crate) static AWQ_A4_SOURCE_SHA: OnceLock<String> = OnceLock::new();
static AWQ_A4_CAPTURE_MANIFEST: OnceLock<serde_json::Value> = OnceLock::new();
static AWQ_A4_VERIFIED_FILES: LazyLock<Mutex<HashSet<PathBuf>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

pub(crate) const MQ4V2_FINAL_CODE_QTYPE: u8 = 44;

/// Borrowed view of one merge-aware QAT export record.
///
/// The three tensor payloads use safetensors' little-endian representation.
/// `d_z_f16` is flattened from `[M, K/256, 2, 2]` in row-major order, with
/// the last axis ordered `(d, z)`.
#[derive(Clone, Copy)]
pub(crate) struct Mq4v2FinalCodeRecord<'a> {
    pub(crate) name: &'a str,
    pub(crate) m: usize,
    pub(crate) k: usize,
    pub(crate) qt: u8,
    pub(crate) source_sha: &'a str,
    pub(crate) s_f16: &'a [u8],
    pub(crate) d_z_f16: &'a [u8],
    pub(crate) codes_u8: &'a [u8],
}

pub(crate) struct MappedMq4v2FinalCodeRecord {
    pub(crate) path: PathBuf,
    name: String,
    m: usize,
    k: usize,
    qt: u8,
    source_sha: String,
    s_range: Range<usize>,
    d_z_range: Range<usize>,
    codes_range: Range<usize>,
    mmap: Mmap,
}

impl MappedMq4v2FinalCodeRecord {
    pub(crate) fn as_record(&self) -> Mq4v2FinalCodeRecord<'_> {
        Mq4v2FinalCodeRecord {
            name: &self.name,
            m: self.m,
            k: self.k,
            qt: self.qt,
            source_sha: &self.source_sha,
            s_f16: &self.mmap[self.s_range.clone()],
            d_z_f16: &self.mmap[self.d_z_range.clone()],
            codes_u8: &self.mmap[self.codes_range.clone()],
        }
    }
}

fn checked_record_sizes(m: usize, k: usize) -> Result<(usize, usize), String> {
    if m == 0 || k == 0 {
        return Err(format!("MQ4V2 final-code dimensions must be nonzero, got M={m}, K={k}"));
    }
    if k % 256 != 0 {
        return Err(format!("MQ4V2 final-code K must be divisible by 256, got K={k}"));
    }
    let codes = m
        .checked_mul(k)
        .ok_or_else(|| format!("MQ4V2 final-code shape overflows: {m}x{k}"))?;
    let grid_values = codes
        .checked_div(256)
        .and_then(|groups| groups.checked_mul(4))
        .ok_or_else(|| format!("MQ4V2 final-code grid shape overflows: {m}x{k}"))?;
    Ok((codes, grid_values))
}

fn f16_bits_at(bytes: &[u8], index: usize) -> u16 {
    u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]])
}

/// Validate the frozen C3 record before any artifact byte is written.
pub(crate) fn validate_mq4v2_final_code_record(
    record: &Mq4v2FinalCodeRecord<'_>,
    expected_source_sha: &str,
) -> Result<(), String> {
    if record.name.is_empty() {
        return Err("MQ4V2 final-code record has an empty tensor name".to_string());
    }
    if record.qt != MQ4V2_FINAL_CODE_QTYPE {
        return Err(format!(
            "{}: final-code qt={} but MQ4V2 requires qt=44",
            record.name, record.qt
        ));
    }
    if record.source_sha.len() != 64
        || !record
            .source_sha
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "{}: source_sha must be a lowercase 64-digit SHA-256",
            record.name
        ));
    }
    if record.source_sha != expected_source_sha {
        return Err(format!(
            "{}: source_sha {} does not match input artifact {}",
            record.name, record.source_sha, expected_source_sha
        ));
    }

    let (expected_codes, expected_grid_values) = checked_record_sizes(record.m, record.k)?;
    if record.s_f16.len() != record.k * 2 {
        return Err(format!(
            "{}: S_f16 has {} bytes, expected {} for shape [{}]",
            record.name,
            record.s_f16.len(),
            record.k * 2,
            record.k
        ));
    }
    if record.d_z_f16.len() != expected_grid_values * 2 {
        return Err(format!(
            "{}: d_z_f16 has {} bytes, expected {} for shape [{},{},2,2]",
            record.name,
            record.d_z_f16.len(),
            expected_grid_values * 2,
            record.m,
            record.k / 256
        ));
    }
    if record.codes_u8.len() != expected_codes {
        return Err(format!(
            "{}: codes_u8 has {} bytes, expected {} for shape [{},{}]",
            record.name,
            record.codes_u8.len(),
            expected_codes,
            record.m,
            record.k
        ));
    }

    for index in 0..record.k {
        let value = f16_to_f32(f16_bits_at(record.s_f16, index));
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "{}: S_f16[{index}] must be finite and positive, got {value}",
                record.name
            ));
        }
    }
    for index in 0..expected_grid_values {
        let value = f16_to_f32(f16_bits_at(record.d_z_f16, index));
        if !value.is_finite() {
            return Err(format!(
                "{}: d_z_f16 flat index {index} is non-finite",
                record.name
            ));
        }
        if index % 2 == 0 && value < 0.0 {
            return Err(format!(
                "{}: d_z_f16 scale at flat index {index} is negative",
                record.name
            ));
        }
    }
    if let Some((index, code)) = record
        .codes_u8
        .iter()
        .enumerate()
        .find(|(_, code)| **code > 15)
    {
        return Err(format!(
            "{}: codes_u8[{index}]={code} exceeds the uint4 range",
            record.name
        ));
    }

    let groups = expected_codes / 256;
    for group in 0..groups {
        for half in 0..2 {
            let scale_index = group * 4 + half * 2;
            if f16_bits_at(record.d_z_f16, scale_index) == 0
                && record.codes_u8[group * 256 + half * 128..group * 256 + (half + 1) * 128]
                    .iter()
                    .any(|&code| code != 0)
            {
                return Err(format!(
                    "{}: group {group} half {half} has zero scale with nonzero codes",
                    record.name
                ));
            }
        }
    }
    Ok(())
}

fn safetensor_range(
    mmap: &Mmap,
    tensor: &safetensors::tensor::TensorView<'_>,
) -> Result<Range<usize>, String> {
    let start = (tensor.data().as_ptr() as usize)
        .checked_sub(mmap.as_ptr() as usize)
        .ok_or_else(|| "safetensors payload is outside its mmap".to_string())?;
    let end = start
        .checked_add(tensor.data().len())
        .ok_or_else(|| "safetensors payload range overflows".to_string())?;
    Ok(start..end)
}

fn parse_record_metadata_usize(
    metadata: &HashMap<String, String>,
    key: &str,
    path: &Path,
) -> Result<usize, String> {
    metadata
        .get(key)
        .ok_or_else(|| format!("{}: missing safetensors metadata `{key}`", path.display()))?
        .parse::<usize>()
        .map_err(|error| format!("{}: invalid metadata `{key}`: {error}", path.display()))
}

pub(crate) fn load_mq4v2_final_code_record(
    path: &Path,
    expected_source_sha: &str,
) -> Result<MappedMq4v2FinalCodeRecord, String> {
    let file = File::open(path)
        .map_err(|error| format!("open final-code record {}: {error}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|error| format!("mmap final-code record {}: {error}", path.display()))?;
    let (_, header) = SafeTensors::read_metadata(&mmap)
        .map_err(|error| format!("parse final-code record {}: {error}", path.display()))?;
    let parsed = SafeTensors::deserialize(&mmap)
        .map_err(|error| format!("parse final-code record {}: {error}", path.display()))?;
    let names: HashSet<&str> = parsed.names().into_iter().collect();
    let expected_names: HashSet<&str> = ["S_f16", "d_z_f16", "codes_u8"].into_iter().collect();
    if names != expected_names {
        let mut actual: Vec<&str> = names.into_iter().collect();
        actual.sort_unstable();
        return Err(format!(
            "{}: record tensors must be exactly S_f16,d_z_f16,codes_u8; got {:?}",
            path.display(),
            actual
        ));
    }
    let metadata = header
        .metadata()
        .as_ref()
        .ok_or_else(|| format!("{}: final-code record has no metadata", path.display()))?;
    let name = metadata
        .get("name")
        .cloned()
        .ok_or_else(|| format!("{}: missing safetensors metadata `name`", path.display()))?;
    let m = parse_record_metadata_usize(metadata, "M", path)?;
    let k = parse_record_metadata_usize(metadata, "K", path)?;
    let qt_usize = parse_record_metadata_usize(metadata, "qt", path)?;
    let qt = u8::try_from(qt_usize)
        .map_err(|_| format!("{}: metadata `qt` does not fit u8", path.display()))?;
    let source_sha = metadata
        .get("source_sha")
        .cloned()
        .ok_or_else(|| format!("{}: missing safetensors metadata `source_sha`", path.display()))?;

    let s = parsed
        .tensor("S_f16")
        .map_err(|error| format!("{}: read S_f16: {error}", path.display()))?;
    let d_z = parsed
        .tensor("d_z_f16")
        .map_err(|error| format!("{}: read d_z_f16: {error}", path.display()))?;
    let codes = parsed
        .tensor("codes_u8")
        .map_err(|error| format!("{}: read codes_u8: {error}", path.display()))?;
    if s.dtype() != Dtype::F16 || s.shape() != [k] {
        return Err(format!(
            "{}: S_f16 must have dtype F16 and shape [{k}], got {:?} {:?}",
            path.display(),
            s.dtype(),
            s.shape()
        ));
    }
    if d_z.dtype() != Dtype::F16 || d_z.shape() != [m, k / 256, 2, 2] {
        return Err(format!(
            "{}: d_z_f16 must have dtype F16 and shape [{m},{},2,2], got {:?} {:?}",
            path.display(),
            k / 256,
            d_z.dtype(),
            d_z.shape()
        ));
    }
    if codes.dtype() != Dtype::U8 || codes.shape() != [m, k] {
        return Err(format!(
            "{}: codes_u8 must have dtype U8 and shape [{m},{k}], got {:?} {:?}",
            path.display(),
            codes.dtype(),
            codes.shape()
        ));
    }
    let s_range = safetensor_range(&mmap, &s)?;
    let d_z_range = safetensor_range(&mmap, &d_z)?;
    let codes_range = safetensor_range(&mmap, &codes)?;
    drop(parsed);

    let mapped = MappedMq4v2FinalCodeRecord {
        path: path.to_path_buf(),
        name,
        m,
        k,
        qt,
        source_sha,
        s_range,
        d_z_range,
        codes_range,
        mmap,
    };
    validate_mq4v2_final_code_record(&mapped.as_record(), expected_source_sha)?;
    Ok(mapped)
}

pub(crate) fn load_mq4v2_final_code_records(
    path: &Path,
    expected_source_sha: &str,
) -> Result<Vec<MappedMq4v2FinalCodeRecord>, String> {
    let mut paths = if path.is_dir() {
        std::fs::read_dir(path)
            .map_err(|error| format!("read final-code directory {}: {error}", path.display()))?
            .map(|entry| {
                entry
                    .map(|entry| entry.path())
                    .map_err(|error| format!("read final-code directory entry: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|entry| entry.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
            .collect()
    } else {
        vec![path.to_path_buf()]
    };
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "{} contains no .safetensors final-code records",
            path.display()
        ));
    }

    let mut records = Vec::with_capacity(paths.len());
    let mut names = HashSet::with_capacity(paths.len());
    for record_path in paths {
        let record = load_mq4v2_final_code_record(&record_path, expected_source_sha)?;
        let name = record.as_record().name.to_string();
        if !names.insert(name.clone()) {
            return Err(format!("duplicate MQ4V2 final-code record for `{name}`"));
        }
        records.push(record);
    }
    Ok(records)
}

/// Shared runtime producer whose inverse-AWQ scale must be identical.
pub(crate) fn mq4v2_shared_scale_group(name: &str) -> Option<String> {
    const FULL_ATTN: [&str; 3] = ["q_proj.weight", "k_proj.weight", "v_proj.weight"];
    for suffix in FULL_ATTN {
        if let Some(prefix) = name.strip_suffix(suffix) {
            return Some(format!("{prefix}qkv"));
        }
    }
    for suffix in ["gate_proj.weight", "up_proj.weight"] {
        if let Some(prefix) = name.strip_suffix(suffix) {
            return Some(format!("{prefix}gate_up"));
        }
    }
    for suffix in [
        "in_proj_qkv.weight",
        "in_proj_z.weight",
        "in_proj_a.weight",
        "in_proj_b.weight",
    ] {
        if let Some(prefix) = name.strip_suffix(suffix) {
            return Some(format!("{prefix}in_proj_qkvzab"));
        }
    }
    None
}

struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    byte_len: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667,
                0xbb67ae85,
                0x3c6ef372,
                0xa54ff53a,
                0x510e527f,
                0x9b05688c,
                0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0; 64],
            block_len: 0,
            byte_len: 0,
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
            0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];
        let mut words = [0u32; 64];
        for (index, chunk) in block.chunks_exact(4).take(16).enumerate() {
            words[index] = u32::from_be_bytes(chunk.try_into().unwrap());
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(sum1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = sum0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (state, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *state = state.wrapping_add(value);
        }
    }

    fn update(&mut self, mut bytes: &[u8]) {
        self.byte_len = self.byte_len.wrapping_add(bytes.len() as u64);
        while !bytes.is_empty() {
            let take = (64 - self.block_len).min(bytes.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&bytes[..take]);
            self.block_len += take;
            bytes = &bytes[take..];
            if self.block_len == 64 {
                let block = self.block;
                self.compress(&block);
                self.block_len = 0;
            }
        }
    }

    fn finish(mut self) -> [u8; 32] {
        let bit_len = self.byte_len.wrapping_mul(8);
        self.block[self.block_len] = 0x80;
        self.block_len += 1;
        if self.block_len > 56 {
            self.block[self.block_len..].fill(0);
            let block = self.block;
            self.compress(&block);
            self.block = [0; 64];
        } else {
            self.block[self.block_len..56].fill(0);
        }
        self.block[56..].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.block;
        self.compress(&block);
        let mut digest = [0u8; 32];
        for (chunk, word) in digest.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }
}

pub(crate) fn sha256_file_hex(path: &Path) -> Result<String, String> {
    let file = File::open(path).map_err(|error| format!("open {} for SHA-256: {error}", path.display()))?;
    let mut reader = BufReader::with_capacity(4 * 1024 * 1024, file);
    let mut buffer = vec![0u8; 4 * 1024 * 1024];
    let mut hasher = Sha256::new();
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("read {} for SHA-256: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher
        .finish()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}


pub(crate) fn resolve_model_path(input: &str) -> String {
    let path = Path::new(input);

    // If it's already a valid local directory with config.json, use it directly
    if path.join("config.json").exists() {
        return input.to_string();
    }

    // Check if it looks like a HuggingFace model ID (contains exactly one /)
    if input.contains('/') && !input.contains(std::path::MAIN_SEPARATOR)
        || (cfg!(unix) && input.matches('/').count() == 1)
    {
        let parts: Vec<&str> = input.splitn(2, '/').collect();
        if parts.len() == 2 {
            let org = parts[0];
            let name = parts[1];

            // Check HF cache: ~/.cache/huggingface/hub/models--{org}--{name}/snapshots/*/
            let home = std::env::var("HOME").unwrap_or_default();
            let cache_dir = format!("{home}/.cache/huggingface/hub/models--{org}--{name}");
            let snapshots_dir = Path::new(&cache_dir).join("snapshots");

            if snapshots_dir.exists() {
                // Find the first snapshot directory
                if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                    for entry in entries.flatten() {
                        let snap_path = entry.path();
                        if snap_path.is_dir() && snap_path.join("config.json").exists() {
                            eprintln!("Resolved {input} -> {}", snap_path.display());
                            return snap_path.to_string_lossy().to_string();
                        }
                    }
                }
            }

            // Not in cache — try to download
            eprintln!("Model {input} not found locally. Downloading via huggingface-cli...");
            let status = std::process::Command::new("huggingface-cli")
                .args(["download", input])
                .status();

            match status {
                Ok(s) if s.success() => {
                    // Retry cache lookup after download
                    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                        for entry in entries.flatten() {
                            let snap_path = entry.path();
                            if snap_path.is_dir() && snap_path.join("config.json").exists() {
                                eprintln!("Downloaded {input} -> {}", snap_path.display());
                                return snap_path.to_string_lossy().to_string();
                            }
                        }
                    }
                }
                Ok(s) => eprintln!("huggingface-cli download failed with status {s}"),
                Err(e) => eprintln!(
                    "Failed to run huggingface-cli: {e}. Install with: pip install huggingface_hub"
                ),
            }
        }
    }

    // Fall through: return as-is, will fail at config.json read with a helpful error
    input.to_string()
}

// ─── GGUF input pipeline ────────────────────────────────────────────────────

/// True if the path points to a `.gguf` file on disk.
pub(crate) fn is_gguf_input(p: &Path) -> bool {
    p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("gguf")
}

/// Translate llama.cpp GGUF tensor names to the HuggingFace safetensors
/// names that `hipfire_runtime::hfq::load_weights_hfq` expects. The mapping is
/// the canonical llama.cpp ↔ HF convention.
///
/// Returns None for tensors that don't have a known safetensors equivalent
/// (we then keep them under their GGUF name; the future loader can decide
/// what to do, or they're skipped).
pub(crate) fn gguf_to_safetensors_name(gguf_name: &str, arch_id: u32) -> Option<String> {
    // Top-level tensors.
    match gguf_name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".to_string()),
        "output.weight" => return Some("lm_head.weight".to_string()),
        "output_norm.weight" => return Some("model.norm.weight".to_string()),
        _ => {}
    }
    // Per-layer: blk.{N}.<slot>.weight  →  model.layers.{N}.<slot>.weight
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        // rest = "{N}.<slot>.weight"
        let dot = rest.find('.')?;
        let layer_idx = &rest[..dot];
        let slot_full = &rest[dot + 1..]; // "<slot>.weight"
                                          // Drop the trailing ".weight" so we can rewrite slots like "attn_q"→"self_attn.q_proj".
        let slot = slot_full.strip_suffix(".weight")?;
        // Gemma 4 layers carry FOUR sandwich norms plus a per-layer scalar.
        // The generic Llama slot map below assumes two norms and maps
        // `ffn_norm` → post_attention_layernorm — correct for Llama, WRONG
        // for gemma4 (that is the pre-FFN norm). The loader reads
        // `{p}.layer_scalar` with NO `.weight` suffix (HF tensor name), so
        // `layer_output_scale` needs an early return.
        if arch_id == 13 {
            match slot {
                "post_attention_norm" => {
                    return Some(format!(
                        "model.layers.{layer_idx}.post_attention_layernorm.weight"
                    ));
                }
                "ffn_norm" => {
                    return Some(format!(
                        "model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
                    ));
                }
                "post_ffw_norm" => {
                    return Some(format!(
                        "model.layers.{layer_idx}.post_feedforward_layernorm.weight"
                    ));
                }
                "layer_output_scale" => {
                    return Some(format!("model.layers.{layer_idx}.layer_scalar"));
                }
                _ => {}
            }
        }
        let translated = match slot {
            "attn_norm" => "input_layernorm".to_string(),
            "ffn_norm" => "post_attention_layernorm".to_string(),
            "attn_q" => "self_attn.q_proj".to_string(),
            "attn_k" => "self_attn.k_proj".to_string(),
            "attn_v" => "self_attn.v_proj".to_string(),
            "attn_output" => "self_attn.o_proj".to_string(),
            "attn_q_norm" => "self_attn.q_norm".to_string(),
            "attn_k_norm" => "self_attn.k_norm".to_string(),
            "ffn_gate" => "mlp.gate_proj".to_string(),
            "ffn_up" => "mlp.up_proj".to_string(),
            "ffn_down" => "mlp.down_proj".to_string(),
            other => return Some(format!("model.layers.{layer_idx}.{other}.weight")),
        };
        return Some(format!("model.layers.{layer_idx}.{translated}.weight"));
    }
    None
}

/// True if the GGUF tensor's name is a 1D norm / RMSNorm scaling vector.
/// These stay F16 in the .hfq (no benefit from quantization, precision-sensitive).
pub(crate) fn gguf_is_norm_tensor(name: &str) -> bool {
    name.contains("_norm") || name.contains("norm.weight")
}

/// Translate a hipfire safetensors-style tensor name to the ggml-style name
/// used by llama.cpp's imatrix output (and the rest of llama.cpp's tooling).
///
/// Verified by shape-alignment on Qwen3.5-0.8B imatrix vs safetensors load log
/// (2026-05-11):
///   - K dims match for every covered tensor class (mlp.* , self_attn.* ,
///     linear_attn.in_proj_qkv/z/a/b, linear_attn.out_proj).
///   - Layer-pattern: FullAttention layers (3, 7, 11, ...) carry standard
///     `attn_q/k/v/output`; LinearAttention layers carry `attn_qkv`/
///     `attn_gate`/`ssm_alpha`/`ssm_beta`/`ssm_out` — the SSM-naming
///     convention llama.cpp uses for Mamba-style sub-blocks.
///
/// Returns `None` for tensors that don't have an imatrix counterpart
/// (norms / biases / 1D scalars / lookup-only tables). Those fall back to
/// non-imatrix-weighted quantization in the call site.
pub(crate) fn safetensors_to_ggml_name(name: &str) -> Option<String> {
    // Drop the architecture-specific "language_model." prefix (Qwen3.5
    // structure has model.language_model.layers.{N}.* — the linear-attn
    // crate uses this nested layout, llama.cpp flattens to blk.{N}.*).
    let normalized = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);

    // Top-level (currently no imatrix coverage; default is --process-output OFF).
    match normalized {
        "embed_tokens.weight" => return Some("token_embd.weight".to_string()),
        "lm_head.weight" => return Some("output.weight".to_string()),
        "norm.weight" => return Some("output_norm.weight".to_string()),
        _ => {}
    }

    // Per-layer: "layers.{N}.<slot>.weight"
    let rest = normalized.strip_prefix("layers.")?;
    let dot = rest.find('.')?;
    let layer_idx = &rest[..dot];
    let slot_full = &rest[dot + 1..];
    let slot = slot_full.strip_suffix(".weight")?;

    let translated = match slot {
        // MLP — present on every layer.
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        // FullAttention layer tensors (standard names).
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        // Glimmer gates attention output before o_proj under a name Qwen does
        // not use (see hipfire-arch-muse-glimmer lib.rs). llama.cpp exports it
        // as blk.{N}.attn_gate, so without this arm the 52 Glimmer gate tensors
        // silently miss AWQ despite `awq_eligible` matching `gate_proj.weight`
        // and the imatrix carrying the entry. No collision with the linear-attn
        // arm below: a layer is either full- or linear-attention, never both.
        "self_attn.gate_proj" => "attn_gate",
        // LinearAttention layer tensors (Mamba-2 / hybrid-arch SSM naming).
        "linear_attn.in_proj_qkv" => "attn_qkv",
        "linear_attn.in_proj_z" => "attn_gate",
        "linear_attn.in_proj_a" => "ssm_alpha",
        "linear_attn.in_proj_b" => "ssm_beta",
        "linear_attn.out_proj" => "ssm_out",
        // Unmapped: conv1d.weight (special-cased to HFQ4G128 at quantize
        // time; small, not multiplied by activation in the standard sense),
        // norm.weight, A_log, dt_bias (1D or scalars, no imatrix entry).
        _ => return None,
    };

    Some(format!("blk.{layer_idx}.{translated}.weight"))
}

/// Load an llama.cpp-compatible imatrix GGUF file and build a lookup
/// keyed by ggml-style tensor name. The GGUF stores per-linear-layer
/// pairs:
///   {name}.in_sum2     F32[k, n_mat]   sum of squared activations per channel
///   {name}.counts      F32[1, n_mat]   token count contributing per matrix
///
/// For non-MoE models n_mat=1; the [k] vector goes into the map directly.
/// For MoE we'd need per-expert handling — out of scope for Step 5a
/// (Qwen3.5 dense + Qwen3.6 dense are the first cohort targets; A3B MoE
/// is deferred to a future iteration that handles n_mat > 1).
///
/// Returns `HashMap<ggml_name, Vec<f32>>` with the .in_sum2 values keyed by
/// the BASE tensor name (the ".in_sum2" suffix stripped).
pub(crate) fn load_imatrix(path: &Path, fix_la_head_order: bool) -> HashMap<String, Vec<f32>> {
    use gguf_input::GgmlType;
    let gguf = gguf_input::GgufFile::open(path).unwrap_or_else(|e| {
        eprintln!("error: failed to open imatrix file {}: {e}", path.display());
        std::process::exit(1);
    });

    let mut map: HashMap<String, Vec<f32>> = HashMap::new();
    let mut fixed_la_out = 0usize;
    let mut total_entries = 0usize;
    let mut skipped_moe = 0usize;
    for t in &gguf.tensors {
        let name = match t.name.strip_suffix(".in_sum2") {
            Some(n) => n.to_string(),
            None => continue, // ignore .counts and any other entries
        };
        if t.dtype != GgmlType::F32 {
            eprintln!(
                "warning: imatrix entry {} has non-F32 dtype {:?}; skipping",
                t.name, t.dtype
            );
            continue;
        }
        // Shape is [k] (1D) for non-MoE; [k, n_mat] for MoE. Skip multi-mat
        // tensors with a warning — Step 5a doesn't handle them yet.
        let n_mat = if t.shape.len() >= 2 { t.shape[1] } else { 1 };
        if n_mat != 1 {
            skipped_moe += 1;
            continue;
        }
        let k = t.shape[0];

        // Read the F32 values from the tensor data segment.
        let data = gguf.tensor_data(t);
        let mut values = Vec::with_capacity(k);
        for i in 0..k {
            let off = i * 4;
            values.push(f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]));
        }
        if fix_la_head_order && (name.ends_with(".linear_attn.out_proj.weight")
            || name.ends_with(".ssm_out.weight"))
        {
            assert_eq!(k, 48 * 128, "{name}: Qwen3.8 LA out_proj must have 48 V-heads x 128");
            let raw = values;
            values = vec![0.0; k];
            // llama.cpp stores three V heads per K head (V-major/K-minor);
            // HF out_proj columns use K-major/V-minor. HF head h reads raw
            // head (h % 3)*16 + h/3. No reordering of W or runtime sidecar.
            for h in 0..48 {
                let raw_h = (h % 3) * 16 + h / 3;
                values[h * 128..(h + 1) * 128]
                    .copy_from_slice(&raw[raw_h * 128..(raw_h + 1) * 128]);
            }
            fixed_la_out += 1;
        }
        map.insert(name, values);
        total_entries += 1;
    }
    if fix_la_head_order {
        assert_eq!(fixed_la_out, 48, "Qwen3.8 LA imatrix head fix expected 48 out_proj tensors, found {fixed_la_out}");
        eprintln!("imatrix: corrected V-head order for {fixed_la_out} linear-attention out_proj tensors");
    }

    eprintln!(
        "imatrix: loaded {} entries from {} ({} MoE multi-matrix entries skipped — Step 5a is dense-only)",
        total_entries,
        path.display(),
        skipped_moe,
    );
    if total_entries == 0 {
        if skipped_moe > 0 {
            // MoE-only imatrix (e.g. MiniMax routed experts): no 1D dense
            // entries for the legacy dense-AWQ table, but the file IS valid.
            // The MiniMax AWQ-on-experts path reads the raw imatrix GGUF
            // (imatrix_gguf) directly, so an empty dense table is harmless —
            // dense tensors just fall back to non-imatrix quantization.
            eprintln!(
                "imatrix: 0 dense entries, {skipped_moe} MoE multi-matrix entries — \
                 dense table empty (MoE-only imatrix; expert AWQ uses the raw GGUF)"
            );
        } else {
            eprintln!("error: imatrix file contains no usable .in_sum2 entries");
            std::process::exit(1);
        }
    }
    map
}

/// Look up imatrix per-channel weights for a given safetensors tensor name.
/// Returns `None` (caller falls back to non-imatrix-weighted quantization) if:
///   - --imatrix wasn't passed (IMATRIX not initialized), OR
///   - the tensor name doesn't have a ggml-mapping (norms, small 1D, etc.), OR
///   - the imatrix file doesn't carry this tensor (rare; usually means the
///     tensor wasn't exercised by the calibration corpus).
pub(crate) fn imatrix_weights_for(safetensors_name: &str) -> Option<&'static [f32]> {
    let im = IMATRIX.get()?;
    // `load_imatrix` keys the map by the imatrix FILE's tensor names (`.in_sum2`
    // stripped). hipfire's `collect_imatrix` emits *safetensors* names
    // (`model.language_model.layers.N.linear_attn.in_proj_qkv.weight`), so try the
    // direct safetensors name FIRST — this was the AWQ no-op: the map is
    // safetensors-keyed but we only tried the GGML-converted name, which always
    // missed (and 27B-3.6 hybrid linear_attn names don't round-trip anyway).
    // Fall back to the GGML name for llama.cpp-style (blk.*) imatrices.
    if let Some(v) = im.get(safetensors_name) {
        return Some(v.as_slice());
    }
    let ggml_name = safetensors_to_ggml_name(safetensors_name)?;
    im.get(&ggml_name).map(|v| v.as_slice())
}

/// Compute AWQ per-channel scales `s[j]` for one linear-layer weight tensor.
///
/// Inputs:
///   - `in_sum2`: imatrix data — Σ_token act²[j] per input channel, length K.
///     Source: hipfire's `imatrix_collect` (llama.cpp `--imatrix` output).
///   - `alpha`: AWQ tuning parameter ∈ [0, 1]. Paper-original default = 0.5.
///
/// Output:
///   - `Vec<f32>` of length K, with geometric mean normalized to ≈ 1.0.
///
/// Formula (AWQ-paper-original simplified for hipfire's data shape):
///   1. RMS_act[j] = sqrt(in_sum2[j] / N_tok). The N_tok term is a global
///      constant for the tensor and gets absorbed by the geo-mean normalization
///      below, so we can omit it from the per-channel computation.
///      Equivalent: use sqrt(in_sum2[j]) directly.
///   2. s_raw[j] = (RMS_act[j])^alpha
///   3. Normalize: s[j] = s_raw[j] / exp(mean_j log(s_raw[j]))
///      This keeps the post-AWQ-scaled weight tensor's overall magnitude
///      in the same range as the input — important for the downstream MQ4
///      min-max scale fitter not to suddenly compress/expand its dynamic
///      range based on alpha.
///
/// Edge cases:
///   - Zero in_sum2[j] (channel never exercised by calibration): clamp to
///     a tiny floor (1e-12) before sqrt to avoid log(0). Practically rare;
///     would mean a channel is unused in the calibration corpus.
///   - alpha == 0 → all s[j] = 1.0 (AWQ disabled at this layer). Caller
///     can short-circuit before invoking this function.
///
/// Cost: O(K). For 9B Qwen3.5 ~32 calls × ~4096 elements = ~131K ops total
/// across the whole quantize. Negligible.
/// Parse the layer index N from a MiniMax expert tensor name
/// `…layers.N.block_sparse_moe.experts.E.wX.weight`.
pub(crate) fn minimax_layer_index(name: &str) -> Option<usize> {
    let after = name.split(".layers.").nth(1)?;
    after.split('.').next()?.parse::<usize>().ok()
}

/// True if layer `l` falls in the comma-separated range list held in process config `var`
/// (e.g. "12-45,50,55-60"; inclusive ranges or bare singles). Unset/empty →
/// false. Drives per-layer mixed-precision expert promotion for MiniMax.
pub(crate) fn minimax_layer_in_config_set(var: &str, l: usize) -> bool {
    let spec = match hipfire_config::developer_var(var) {
        Ok(v) => v,
        Err(_) => return false,
    };
    for tok in spec.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        if let Some((a, b)) = tok.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                if l >= a.min(b) && l <= a.max(b) {
                    return true;
                }
            }
        } else if let Ok(n) = tok.parse::<usize>() {
            if l == n {
                return true;
            }
        }
    }
    false
}

/// Shared-per-layer AWQ scales for MiniMax routed experts from an imatrix GGUF.
/// Aggregates per-expert activation energy (in_sum2) across ALL experts of
/// layer `n` into one shared per-input-channel scale: gate(w1)/up(w3) share the
/// MoE-input channels (s_gate_up, len hidden); down(w2) uses the intermediate
/// channels (s_down, len inter). The forward applies these via experts[0], so
/// one scale per layer is exactly what the runtime consumes. None if absent.
pub(crate) fn minimax_layer_awq_scales(
    gguf: &gguf_input::GgufFile,
    n: usize,
    alpha: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let agg = |kind: &str| -> Option<Vec<f32>> {
        let nm = format!("blk.{n}.ffn_{kind}_exps.weight.in_sum2");
        let t = gguf.tensors.iter().find(|t| t.name == nm)?;
        if t.shape.len() != 2 {
            return None;
        }
        let k = t.shape[0];
        let n_exp = t.shape[1];
        let flat: Vec<f32> = gguf
            .tensor_data(t)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if flat.len() != k * n_exp {
            return None;
        }
        let mut a = vec![0.0f32; k];
        for e in 0..n_exp {
            let off = e * k;
            for j in 0..k {
                a[j] += flat[off + j];
            }
        }
        Some(a)
    };
    let g = agg("gate")?;
    let gu: Vec<f32> = match agg("up") {
        Some(u) if u.len() == g.len() => g.iter().zip(&u).map(|(a, b)| a + b).collect(),
        _ => g.clone(),
    };
    let d = agg("down")?;
    Some((
        compute_awq_scales(&gu, alpha),
        compute_awq_scales(&d, alpha),
    ))
}

pub(crate) fn compute_awq_scales(in_sum2: &[f32], alpha: f32) -> Vec<f32> {
    let k = in_sum2.len();
    debug_assert!(k > 0, "empty imatrix vector");

    // Step 1+2: RMS_act^alpha, with the constant N_tok factor absorbed into
    // the geo-mean normalization. The sqrt and (·)^alpha combine into
    // (·)^(alpha/2) on the raw in_sum2 values.
    //
    // Implementation choice: compute log(s_raw) directly so we can do the
    // geo-mean normalization in log space (numerically more stable for
    // wide dynamic-range imatrix values).
    let half_alpha = (alpha as f64) * 0.5;
    let mut log_s_raw = Vec::with_capacity(k);
    let mut sum_log: f64 = 0.0;
    for &v in in_sum2 {
        // Floor dead channels to 1e-12 (NaN also maps here: f64::max returns the
        // non-NaN arg) AND cap non-finite / pathologically-large values to a
        // finite ceiling. An inf in_sum2 — f32 overflow during imatrix
        // collection, which the 27B tier1 imatrix actually contains — would
        // otherwise make this tensor's `mean_log = inf`, and then `l - mean_log`
        // = inf - inf = NaN for the inf channel. That NaN survives the output
        // clamp below (f32::clamp propagates NaN), poisoning the F16 sidecar and
        // NaN'ing the whole forward (37747 such values measured pre-fix).
        // Capping the input keeps mean_log finite; the output clamp then bounds
        // the final scale. 1e30 is well inside f64 range (ln ≈ 69).
        let v_clamped = (v as f64).max(1e-12).min(1e30);
        let log_s = half_alpha * v_clamped.ln(); // log(v^(alpha/2)) = (alpha/2) * log(v)
        log_s_raw.push(log_s);
        sum_log += log_s;
    }
    let mean_log = sum_log / (k as f64);

    // Step 3: subtract mean in log space, then exp back. After this,
    // geo_mean(s) = exp(0) = 1.0 exactly (within floating-point precision).
    //
    // Step 4 (CRITICAL — f16 safety): clamp to an f16-representable,
    // non-exploding range. The geo-mean is 1.0 by construction, so the bulk
    // of channels sit near 1; only pathological outliers reach the rails —
    // dead channels floored to 1e-12, or hot channels with huge activation
    // sums. Without this, exp() overflows to f32 inf and/or the F16 sidecar
    // under/overflows, and the inference-time `x / awq_scale` divide produces
    // inf → NaN. (Verified via dump_awq_scales on the 27B tier1 imatrix:
    // 49293 scales underflowed to 0.0 and 37747 stored as inf/NaN pre-clamp,
    // which NaN'd the whole forward — KLD 0.0 / PPL NaN on gfx11.)
    //
    // The SAME clamped vector is used for both the weight pre-scale (W*s) and
    // the emitted sidecar (x/s at inference), so the cancellation stays exact;
    // clamping only limits how aggressively pathological channels redistribute
    // quant difficulty. Real AWQ scales live in ~[0.2, 5]; [1e-2, 1e2] keeps
    // all genuine signal while removing the representability blow-ups.
    pub(crate) const AWQ_SCALE_MIN: f32 = 1e-2;
    pub(crate) const AWQ_SCALE_MAX: f32 = 1e2;
    log_s_raw
        .into_iter()
        .map(|l| ((l - mean_log).exp() as f32).clamp(AWQ_SCALE_MIN, AWQ_SCALE_MAX))
        .collect()
}

/// Host representation of the runtime's 72-byte `block_i4_128` activation block.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BlockI4_128 {
    pub(crate) d: f32,
    pub(crate) s: i32,
    pub(crate) qs: [u8; 64],
}

impl BlockI4_128 {
    pub(crate) fn to_bytes(&self) -> [u8; 72] {
        let mut bytes = [0u8; 72];
        bytes[..4].copy_from_slice(&self.d.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.s.to_le_bytes());
        bytes[8..].copy_from_slice(&self.qs);
        bytes
    }

    fn dequantize(&self) -> [f32; 128] {
        let mut values = [0.0f32; 128];
        for (i, value) in values.iter_mut().enumerate() {
            let nibble = if i & 1 == 0 {
                self.qs[i / 2] & 0x0f
            } else {
                self.qs[i / 2] >> 4
            };
            let q = ((nibble as i8) << 4) >> 4;
            *value = self.d * q as f32;
        }
        values
    }
}

/// CPU twin of the producer specialization in
/// `kernels/src/block_i4_128_quant.hip`. `candidates=2` is the fused
/// gfx1201 c2 route; the historical AWQ objective uses four candidates.
/// Standalone quantization is an eight-candidate route, not c2.
///
/// Four values are accumulated per simulated wave lane, then reduced with
/// the runtime's XOR 16,8,4,2,1 tree. Scale and error arithmetic order matters.
pub(crate) fn fake_quantize_block_i4_128(input: &[f32; 128], candidates: usize) -> BlockI4_128 {
    assert!(matches!(candidates, 1 | 2 | 4 | 8));
    let mut lane_amax = [0.0f32; 32];
    for lane in 0..32 {
        let base = lane * 4;
        lane_amax[lane] = input[base]
            .abs()
            .max(input[base + 1].abs())
            .max(input[base + 2].abs())
            .max(input[base + 3].abs());
    }
    for offset in [16usize, 8, 4, 2, 1] {
        let prior = lane_amax;
        for lane in 0..32 {
            lane_amax[lane] = lane_amax[lane].max(prior[lane ^ offset]);
        }
    }
    let amax = lane_amax[0];

    let mut best_d = 1.0f32;
    if amax != 0.0 {
        let candidate_base = (amax / 7.0f32) * 0.5f32;
        let multipliers: &[f32] = match candidates {
            1 => &[],
            2 => &[f32::from_bits(0x3fdb_6db7), 2.0],
            4 => &[1.0, f32::from_bits(0x3fa4_9249), f32::from_bits(0x3fdb_6db7), 2.0],
            8 => &[],
            _ => unreachable!(),
        };
        if candidates == 1 {
            best_d = amax / 7.0;
        }
        let mut best_mse = 1.0e30f32;
        for index in 0..if candidates == 8 { 8 } else { multipliers.len() } {
            let d = if candidates == 8 {
                (amax / 7.0) * 0.5 * (1.0 + index as f32 / 7.0)
            } else {
                candidate_base * multipliers[index]
            };
            let mut lane_mse = [0.0f32; 32];
            for lane in 0..32 {
                let base = lane * 4;
                let mut mse = 0.0f32;
                for e in 0..4 {
                    let q = (input[base + e] / d)
                        .round_ties_even()
                        .clamp(-8.0, 7.0);
                    let err = (-q).mul_add(d, input[base + e]);
                    mse = err.mul_add(err, mse);
                }
                lane_mse[lane] = mse;
            }
            for offset in [16usize, 8, 4, 2, 1] {
                let prior = lane_mse;
                for lane in 0..32 {
                    lane_mse[lane] += prior[lane ^ offset];
                }
            }
            if lane_mse[0] < best_mse {
                best_mse = lane_mse[0];
                best_d = d;
            }
        }
    }

    let mut q4 = [0i32; 128];
    for i in 0..128 {
        q4[i] = if amax == 0.0 {
            0
        } else {
            (input[i] / best_d)
                .round_ties_even()
                .clamp(-8.0, 7.0) as i32
        };
    }
    let s = q4.iter().sum();
    let mut qs = [0u8; 64];
    for i in 0..64 {
        qs[i] = ((q4[2 * i] & 15) | ((q4[2 * i + 1] & 15) << 4)) as u8;
    }
    BlockI4_128 { d: best_d, s, qs }
}

#[derive(Clone, Debug)]
pub(crate) struct A4AwareAwqResult {
    pub(crate) scales: Vec<f32>,
    pub(crate) alpha: f32,
    pub(crate) relative_output_mse: f64,
}

/// The QAT capture is post-producer, pre-AWQ and signed. Its manifest binds
/// every tensor to the parent and disjoint prompt stream; uncaptured layers
/// deliberately remain on the historical objective for a partial-layer screen.
fn signed_awq_rows(name: &str, k: usize) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let root = AWQ_A4_SIGNED_CAPTURE.get()?;
    let manifest = AWQ_A4_CAPTURE_MANIFEST.get_or_init(|| {
        serde_json::from_reader(File::open(root.join("manifest.json")).expect("AWQ capture manifest missing"))
            .expect("AWQ capture manifest invalid")
    });
    assert_eq!(
        manifest["schema_version"].as_str(),
        Some("hipfire.qat.spin_rotation.v1"),
        "unexpected AWQ capture schema"
    );
    assert_eq!(
        manifest["source"]["sha256"].as_str(),
        AWQ_A4_SOURCE_SHA.get().map(String::as_str),
        "AWQ capture parent SHA mismatch"
    );
    let prompt = &manifest["corpus"]["token_stream"];
    let prompt_path = root.join(prompt["file"].as_str().expect("capture token file missing"));
    {
        let mut verified = AWQ_A4_VERIFIED_FILES.lock().expect("capture hash cache poisoned");
        if verified.insert(prompt_path.clone()) {
            assert_eq!(
                sha256_file_hex(&prompt_path).expect("unable to hash capture prompt"),
                prompt["sha256"].as_str().expect("capture prompt SHA missing"),
                "AWQ capture prompt checksum mismatch"
            );
        }
    }
    let suffix = name.strip_prefix("model.language_model.layers.")
        .or_else(|| name.strip_prefix("model.layers."))?;
    let (block, projection) = suffix.split_once('.')?;
    let block: usize = block.parse().ok()?;
    let block_record = manifest["capture"]["blocks"].get(block.to_string())?;
    let site = if projection == "mlp.down_proj.weight" {
        "silu_mul_down"
    } else if projection == "mlp.gate_proj.weight" || projection == "mlp.up_proj.weight" {
        "post_attention_norm_gate_up"
    } else if projection == "self_attn.o_proj.weight"
        || projection == "linear_attn.out_proj.weight"
    {
        "mixer_out"
    } else if projection.starts_with("linear_attn.in_proj_") {
        "input_norm_qkvza"
    } else if projection == "self_attn.q_proj.weight"
        || projection == "self_attn.k_proj.weight"
        || projection == "self_attn.v_proj.weight"
    {
        if block_record["producer_sites"].get("input_norm_qkv").is_some() {
            "input_norm_qkv"
        } else {
            "input_norm_qkvza"
        }
    } else {
        return None;
    };
    let record = block_record["producer_sites"].get(site)?;
    assert_eq!(record["dtype"].as_str(), Some("float32-le"), "{name}: wrong capture dtype");
    assert_eq!(record["axes"], serde_json::json!(["sample", "channel"]), "{name}: wrong capture axes");
    let shape = record["shape"].as_array().expect("capture shape missing");
    let n_rows = shape[0].as_u64().expect("capture rows invalid") as usize;
    assert_eq!(shape[1].as_u64(), Some(k as u64), "{name}: capture K mismatch");
    let bytes = n_rows.checked_mul(k).and_then(|v| v.checked_mul(4)).expect("capture size overflow");
    assert_eq!(record["bytes"].as_u64(), Some(bytes as u64), "{name}: capture byte count mismatch");
    let indices = record["global_sample_indices"].as_array().expect("capture indices missing");
    assert_eq!(indices.len(), n_rows, "{name}: capture row count mismatch");
    let path = root.join(record["file"].as_str().expect("capture file missing"));
    let file = File::open(&path).expect("capture tensor missing");
    assert_eq!(file.metadata().expect("capture metadata unavailable").len(), bytes as u64, "{name}: capture file size mismatch");
    {
        let mut verified = AWQ_A4_VERIFIED_FILES.lock().expect("capture hash cache poisoned");
        if verified.insert(path.clone()) {
            assert_eq!(
                sha256_file_hex(&path).expect("unable to hash capture tensor"),
                record["sha256"].as_str().expect("capture SHA missing"),
                "{name}: capture checksum mismatch"
            );
        }
    }
    let mmap = unsafe { Mmap::map(&file).expect("unable to map capture tensor") };
    let train_limit = manifest["corpus"]["train_sequences"].as_u64().expect("train count missing")
        * manifest["corpus"]["sequence_length"].as_u64().expect("sequence length missing");
    let split = |train: bool, target: usize| {
        let eligible: Vec<usize> = indices.iter().enumerate().filter_map(|(i, index)| {
            ((index.as_u64().expect("sample index invalid") < train_limit) == train).then_some(i)
        }).collect();
        assert!(!eligible.is_empty(), "{name}: no {} captured rows", if train { "train" } else { "heldout" });
        (0..eligible.len().min(target)).map(|i| {
            let row = eligible[i * eligible.len() / eligible.len().min(target)];
            mmap[row * k * 4..(row + 1) * k * 4].chunks_exact(4).map(|b| {
                let value = f32::from_le_bytes(b.try_into().unwrap());
                assert!(value.is_finite(), "{name}: non-finite captured activation");
                value
            }).collect::<Vec<f32>>()
        }).collect()
    };
    let train: Vec<Vec<f32>> = split(true, 8);
    let heldout: Vec<Vec<f32>> = split(false, 4);
    assert!(
        train.iter().flatten().any(|&v| v < 0.0) && train.iter().flatten().any(|&v| v > 0.0),
        "{name}: capture rows must be signed"
    );
    Some((train, heldout))
}

fn representative_activation(in_sum2: &[f32]) -> Vec<f32> {
    let mut x: Vec<f32> = in_sum2
        .iter()
        .map(|&v| (v as f64).max(1e-12).min(1e30).sqrt() as f32)
        .collect();
    let rms = (x
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        / x.len() as f64)
        .sqrt() as f32;
    if rms.is_finite() && rms > 0.0 {
        for value in &mut x {
            *value /= rms;
        }
    }
    x
}

fn fake_quantize_runtime_activation(
    activation: &[f32],
    scales: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    candidates: usize,
    round_awq_sidecar: bool,
) -> Vec<f32> {
    use crate::quant_fwht::cpu_fwht_256;

    debug_assert_eq!(activation.len(), scales.len());
    debug_assert_eq!(activation.len() % 256, 0);
    let mut output = vec![0.0f32; activation.len()];
    for group_start in (0..activation.len()).step_by(256) {
        let mut group = [0.0f32; 256];
        for i in 0..256 {
            let scale = if round_awq_sidecar {
                f16_to_f32(f32_to_f16(scales[group_start + i]))
            } else {
                scales[group_start + i]
            };
            group[i] = activation[group_start + i] / scale;
        }
        cpu_fwht_256(&mut group, signs1, signs2);
        for half in 0..2 {
            let mut input = [0.0f32; 128];
            input.copy_from_slice(&group[half * 128..(half + 1) * 128]);
            let dequantized = fake_quantize_block_i4_128(&input, candidates).dequantize();
            output[group_start + half * 128..group_start + (half + 1) * 128]
                .copy_from_slice(&dequantized);
        }
    }
    output
}

fn fake_quantize_mq4v2_weight_group(group: &mut [f32; 256], symmetric: bool) {
    for half in 0..2 {
        if symmetric {
            // Keep the same f32/f16/f64 arithmetic order as the qt44 writer.
            let values = &group[half * 128..(half + 1) * 128];
            let amax = values.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            if amax == 0.0 {
                group[half * 128..(half + 1) * 128].fill(0.0);
                continue;
            }
            let base = (amax / 7.5) * 0.5;
            let mut best_mse = f64::INFINITY;
            let mut best_scale = 0u16;
            let mut best_zero = 0u16;
            for multiplier in [
                1.0f32,
                f32::from_bits(0x3fa4_9249),
                f32::from_bits(0x3fdb_6db7),
                2.0,
            ] {
                let scale_bits = f32_to_f16(base * multiplier);
                let scale = f16_to_f32(scale_bits);
                if scale == 0.0 {
                    continue;
                }
                let zero_bits = f32_to_f16(-8.0 * scale);
                let zero = f16_to_f32(zero_bits);
                let inverse = 1.0 / scale;
                let mut mse = 0.0f64;
                for &value in values {
                    let code = ((value - zero) * inverse + 0.5).floor().clamp(0.0, 15.0);
                    let error = value - code.mul_add(scale, zero);
                    mse += (error as f64) * (error as f64);
                }
                if mse < best_mse {
                    best_mse = mse;
                    best_scale = scale_bits;
                    best_zero = zero_bits;
                }
            }
            let scale = f16_to_f32(best_scale);
            let zero = f16_to_f32(best_zero);
            let inverse = 1.0 / scale;
            for value in &mut group[half * 128..(half + 1) * 128] {
                let code = ((*value - zero) * inverse + 0.5).floor().clamp(0.0, 15.0);
                *value = code.mul_add(scale, zero);
            }
            continue;
        }
        let values = &group[half * 128..(half + 1) * 128];
        let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let step_f32 = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
        let scale_bits = if hi == lo { 0 } else { f32_to_f16(step_f32) };
        let zero_bits = f32_to_f16(lo);
        let scale = f16_to_f32(scale_bits);
        let zero = f16_to_f32(zero_bits);
        let degenerate = hi == lo || step_f32 == 0.0 || scale == 0.0;
        for i in half * 128..(half + 1) * 128 {
            group[i] = if degenerate {
                zero
            } else {
                let q = ((group[i] - zero) * (1.0 / scale) + 0.5)
                    .floor()
                    .clamp(0.0, 15.0);
                q.mul_add(scale, zero)
            };
        }
    }
}

fn w4a4_relative_output_mse(
    weights: &[f32],
    m: usize,
    k: usize,
    activation: &[f32],
    reference_outputs: &[f64],
    sampled_rows: &[usize],
    scales: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    candidates: usize,
    symmetric: bool,
) -> f64 {
    use crate::quant_fwht::cpu_fwht_256;

    let quantized_activation =
        fake_quantize_runtime_activation(activation, scales, signs1, signs2, candidates, symmetric);
    let mut error2 = 0.0f64;
    let mut signal2 = 0.0f64;
    for (sample, &row) in sampled_rows.iter().enumerate() {
        let row_weights = &weights[row * k..(row + 1) * k];
        let mut output = 0.0f64;
        for group_start in (0..k).step_by(256) {
            let mut group = [0.0f32; 256];
            for i in 0..256 {
                group[i] = row_weights[group_start + i] * scales[group_start + i];
            }
            cpu_fwht_256(&mut group, signs1, signs2);
            fake_quantize_mq4v2_weight_group(&mut group, symmetric);
            for i in 0..256 {
                output +=
                    (group[i] as f64) * (quantized_activation[group_start + i] as f64);
            }
        }
        let reference = reference_outputs[sample];
        let error = output - reference;
        error2 += error * error;
        signal2 += reference * reference;
    }
    error2 / signal2.max(1e-30)
}

/// Search AWQ alpha against the selected A4/W recipe on each eligible tensor.
/// C2/symmetric runs across the entire model, using the legacy imatrix RMS
/// statistic where signed QAT producer rows are not captured. The old
/// c4/asymmetric/RMS search is preserved behind --awq-a4-aware.
pub(crate) fn compute_a4_aware_awq_scales(
    name: &str,
    in_sum2: &[f32],
    weights: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> A4AwareAwqResult {
    debug_assert_eq!(in_sum2.len(), k);
    debug_assert_eq!(weights.len(), m * k);
    debug_assert_eq!(k % 256, 0);

    let captured = signed_awq_rows(name, k);
    let activations = captured.as_ref().map_or_else(
        || vec![representative_activation(in_sum2)],
        |(train, _)| train.clone(),
    );
    let sample_count = m.min(if captured.is_some() { 32 } else { 128 });
    let sampled_rows: Vec<usize> = (0..sample_count)
        .map(|sample| sample * m / sample_count)
        .collect();
    let reference_outputs: Vec<Vec<f64>> = activations
        .iter()
        .map(|activation| {
            sampled_rows
                .iter()
                .map(|&row| {
                    weights[row * k..(row + 1) * k]
                        .iter()
                        .zip(activation)
                        .map(|(&w, &x)| (w as f64) * (x as f64))
                        .sum()
                })
                .collect()
        })
        .collect();
    let c2_symmetric = AWQ_A4_ROUTE_C2.get().copied().unwrap_or(false);
    if captured.is_some() {
        eprintln!("    AWQ capture {}: signed train rows, c2 symmetric", name);
    } else if c2_symmetric {
        eprintln!("    AWQ route {}: imatrix RMS statistic, c2 symmetric", name);
    }
    let score = |scales: &[f32], rows: &[Vec<f32>], references: &[Vec<f64>]| {
        rows.iter().zip(references).map(|(activation, reference)| {
            w4a4_relative_output_mse(
                weights, m, k, activation, reference, &sampled_rows,
                scales, signs1, signs2,
                if c2_symmetric { 2 } else { 4 },
                c2_symmetric,
            )
        }).sum::<f64>() / rows.len() as f64
    };
    let mut best: Option<A4AwareAwqResult> = None;
    for alpha in AWQ_A4_CANDIDATE_ALPHAS {
        let scales = compute_awq_scales(in_sum2, alpha);
        let relative_output_mse = score(&scales, &activations, &reference_outputs);
        if best
            .as_ref()
            .map_or(true, |current| relative_output_mse < current.relative_output_mse)
        {
            best = Some(A4AwareAwqResult {
                scales,
                alpha,
                relative_output_mse,
            });
        }
    }
    let best = best.expect("AWQ A4 alpha grid is non-empty");
    if let Some((_, heldout)) = captured {
        let heldout_reference: Vec<Vec<f64>> = heldout.iter().map(|activation| {
            sampled_rows.iter().map(|&row| {
                weights[row * k..(row + 1) * k].iter().zip(activation)
                    .map(|(&w, &x)| w as f64 * x as f64).sum()
            }).collect()
        }).collect();
        eprintln!(
            "    AWQ signed heldout {}: alpha={:.2}, relative_output_mse={:.8e}",
            name, best.alpha, score(&best.scales, &heldout, &heldout_reference)
        );
    }
    best
}

pub(crate) fn compute_awq_scales_for_weight(
    name: &str,
    in_sum2: &[f32],
    fallback_alpha: f32,
    weights: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> A4AwareAwqResult {
    if AWQ_A4_AWARE.get().copied().unwrap_or(false) {
        // Fused qkvza/gate-up producers inverse-scale one activation for several
        // weight matrices. Every matrix consuming the same calibration vector
        // must therefore use the same alpha, or its folded W*s no longer
        // cancels x/s at runtime. Tensor ordering puts the large fused anchor
        // first; subsequent siblings reuse its winning alpha.
        let activation_key: Vec<u32> = in_sum2.iter().map(|value| value.to_bits()).collect();
        if let Some(alpha) = AWQ_A4_ALPHA_BY_ACTIVATION
            .lock()
            .expect("A4 AWQ alpha cache poisoned")
            .get(&activation_key)
            .copied()
        {
            return A4AwareAwqResult {
                scales: compute_awq_scales(in_sum2, alpha),
                alpha,
                relative_output_mse: f64::NAN,
            };
        }
        let selected = compute_a4_aware_awq_scales(name, in_sum2, weights, m, k, signs1, signs2);
        AWQ_A4_ALPHA_BY_ACTIVATION
            .lock()
            .expect("A4 AWQ alpha cache poisoned")
            .insert(activation_key, selected.alpha);
        selected
    } else {
        A4AwareAwqResult {
            scales: compute_awq_scales(in_sum2, fallback_alpha),
            alpha: fallback_alpha,
            relative_output_mse: f64::NAN,
        }
    }
}

/// Apply AWQ pre-scaling to a row-major [m, k] weight tensor in place:
/// `W'[i,j] = W[i,j] * s[j]` for every (i, j).
///
/// AWQ scales are per-INPUT-channel (length K). The same s[j] vector
/// broadcasts across every output row i.
///
/// Done in-place to avoid allocating a second [m, k] buffer. The caller
/// owns the W slice and is responsible for ensuring this pre-scaling
/// happens BEFORE any subsequent transformation (e.g. FWHT rotation).
pub(crate) fn awq_pre_scale_weights(weights: &mut [f32], m: usize, k: usize, scales: &[f32]) {
    debug_assert_eq!(weights.len(), m * k, "weight buffer size mismatch");
    debug_assert_eq!(scales.len(), k, "AWQ scale vector must have length K");
    for r in 0..m {
        let row = &mut weights[r * k..(r + 1) * k];
        for j in 0..k {
            row[j] *= scales[j];
        }
    }
}

/// Helper: convert a `Vec<f32>` AWQ-scale vector into the F16 byte
/// payload that `HfqTensor` consumes for sidecar emission.
pub(crate) fn awq_scales_to_f16_bytes(scales: &[f32]) -> Vec<u8> {
    scales
        .iter()
        .flat_map(|&s| f32_to_f16(s).to_le_bytes())
        .collect()
}

/// AWQ pre-scaling is mathematically valid only for weights whose runtime
/// path applies the inverse divide-by-scale. As of F2 (2026-05-14), this
/// covers both the input-side projections (fed via the AWQ-aware variants
/// of `fused_rmsnorm_rotate_mq` from F1) AND the output-side projections
/// (`o_proj` / `out_proj` / `down_proj` / `w_down`, fed via the AWQ-aware
/// variants `rotate_x_mq_awq` and `fused_silu_mul_mq_rotate_awq` from F2).
///
/// Runtime path mapping for AWQ inverse divide-by-scale:
/// - `fused_rmsnorm_mq_rotate_awq`: post-RMSNorm input projections
///   (q/k/v/qkv, gate/up, in_proj_*, router, gate_up_proj)
/// - `rotate_x_mq_awq`: post-attention input to o_proj / out_proj
/// - `fused_silu_mul_mq_rotate_awq`: post-SwiGLU input to down_proj
///
/// Pre-F2 history: until 2026-05-14, output-side projections (o_proj /
/// out_proj / down_proj / w_down) were NOT on this whitelist because
/// their runtime path lacked AWQ-aware kernels. Pre-scaling them without
/// a runtime compensating divide produces `(W·s) · x ≠ W · x` — measured
/// 0.8B Qwen3.5 KLD blowup 0.6721 → 13.4893; see `awq_fix_claude.md`.
/// F2 added those kernels (`rotate_x_mq_awq` / `fused_silu_mul_mq_rotate_awq`)
/// plus `_for` helper routing in hipfire-runtime/llama.rs, so the whitelist
/// is now safe to expand.
///
/// Whitelist (vs blacklist) is still the safe default: a new tensor name
/// in a future arch fails closed (no AWQ) until someone confirms its
/// runtime path is AWQ-aware.
pub(crate) fn awq_eligible(name: &str) -> bool {
    // F1-vs-F2 A/B gate. When `HIPFIRE_AWQ_F1_ONLY=1` is set, the F2
    // additions below (o_proj / wo / out_proj / down_proj / w_down)
    // are excluded — produces an F1-equivalent quant for comparison
    // bench against the same binary's F2 quant. Default (env unset):
    // the full F2 whitelist applies.
    let f1_only = hipfire_config::developer_var("HIPFIRE_AWQ_F1_ONLY")
        .ok()
        .as_deref()
        == Some("1");
    let f1_match =
    // Full-attention input projections (HF naming + fused variants).
    name.ends_with("q_proj.weight")
        || name.ends_with("k_proj.weight")
        || name.ends_with("v_proj.weight")
        || name.ends_with("qkv_proj.weight")
        || name.ends_with("wqkv.weight")
        // MLP input projections (HF + hipfire-internal naming).
        || name.ends_with("gate_proj.weight")
        || name.ends_with("up_proj.weight")
        || name.ends_with("w_gate.weight")
        || name.ends_with("w_up.weight")
        // MoE fused expert gate+up projection (Qwen3-MoE convention —
        // experts.gate_up_proj is [num_experts, 2*intermediate, hidden]
        // with rows split between gate and up halves). Same input-side
        // semantics as gate_proj/up_proj: post-RMSNorm hidden state
        // routed via the MoE dispatch.
        || name.ends_with("gate_up_proj.weight")
        // Linear-attention input projections (Qwen3.5 Gated-DeltaNet).
        // Suffix varies (in_proj_qkv / _z / _a / _b); the substring is
        // anchored enough that no non-linear-attn tensor name should match.
        || name.contains(".in_proj_")
        // MoE router (HF naming for Qwen3-MoE / DeepSeek family — single
        // linear projecting post-RMSNorm hidden state to num_experts
        // logits). The quantizer's q8_router rule (set when is_moe)
        // promotes this to Q8 before reaching the MQ4G256 branch, so
        // this match is effectively dead code today. Kept for intent:
        // if Q8 auto-promotion is ever disabled, this preserves
        // correctness. `router.weight` would be a non-HF naming an
        // arch might choose; kept for safety.
        || name.ends_with("mlp.gate.weight")
        // MiniMax-M2 MoE router (block_sparse_moe.gate.weight). Same intent
        // as mlp.gate.weight: q8_router (set for is_minimax via is_moe_like)
        // keeps the router at Q8 so HFQ4 noise can't flip top-k selection.
        || name.ends_with("block_sparse_moe.gate.weight")
        || name.ends_with("router.weight")
        // Gemma4 26B-A4B MoE router: `router.proj.weight` (hidden_size × num_experts).
        // Same precision-sensitivity as Qwen3.5's `mlp.gate.weight`.
        || name.ends_with("router.proj.weight");
    if f1_only {
        return f1_match;
    }
    let f2_match =
        // ── F2 (2026-05-14): output-side projections ────────────────────
        // These now have AWQ-aware runtime kernels (rotate_x_mq_awq for
        // o_proj/out_proj/wo; fused_silu_mul_mq_rotate_awq for down_proj/w_down).
        // Runtime dispatch routes through _for helpers in llama.rs based on
        // WeightTensor.awq_scale.
        //
        // FullAttention output projection (HF + hipfire-internal naming).
        name.ends_with("o_proj.weight")
        || name.ends_with("wo.weight")
        // LinearAttention output projection (Qwen3.5 Gated-DeltaNet).
        || name.ends_with("out_proj.weight")
        // MLP down projection (HF + hipfire-internal naming).
        || name.ends_with("down_proj.weight")
        || name.ends_with("w_down.weight");
    f1_match || f2_match
}

/// True if the tensor is the token embedding. We Q8 these (matches the
/// safetensors path's `is_embed` rule — Q4 is too lossy for embedding tables).
pub(crate) fn gguf_is_embed_tensor(name: &str) -> bool {
    name == "token_embd.weight"
}

/// Build the `config` JSON object that `hipfire_runtime::hfq::config_from_hfq`
/// reads. Mirrors the field names HuggingFace uses in `config.json` for
/// LlamaForCausalLM / Qwen3ForCausalLM, populated from the GGUF
/// `<arch>.*` metadata keys.
pub(crate) fn config_json_from_gguf(
    gguf: &gguf_input::GgufFile,
    arch_str: &str,
    arch_id: u32,
) -> serde_json::Value {
    // GGUF prefixes its model hyperparameters with the architecture name —
    // e.g. for `general.architecture=llama` the keys live under `llama.*`.
    let prefix = arch_str;

    let read_u = |k: &str| -> Option<u64> {
        gguf.metadata.get(k).and_then(|v| match v {
            gguf_input::MetaValue::U8(x) => Some(*x as u64),
            gguf_input::MetaValue::I8(x) => Some(*x as u64),
            gguf_input::MetaValue::U16(x) => Some(*x as u64),
            gguf_input::MetaValue::I16(x) => Some(*x as u64),
            gguf_input::MetaValue::U32(x) => Some(*x as u64),
            gguf_input::MetaValue::I32(x) => Some(*x as u64),
            gguf_input::MetaValue::U64(x) => Some(*x),
            gguf_input::MetaValue::I64(x) => Some(*x as u64),
            _ => None,
        })
    };
    let read_f = |k: &str| -> Option<f64> {
        gguf.metadata.get(k).and_then(|v| match v {
            gguf_input::MetaValue::F32(x) => Some(*x as f64),
            gguf_input::MetaValue::F64(x) => Some(*x),
            _ => None,
        })
    };

    let dim = read_u(&format!("{prefix}.embedding_length"));
    let n_layers = read_u(&format!("{prefix}.block_count"));
    let n_heads = read_u(&format!("{prefix}.attention.head_count"));
    let n_kv_heads = read_u(&format!("{prefix}.attention.head_count_kv")).or(n_heads);
    let hidden_dim = read_u(&format!("{prefix}.feed_forward_length"));
    // vocab_size: prefer metadata, fall back to token_embd shape[1].
    let vocab_size = read_u(&format!("{prefix}.vocab_size")).or_else(|| {
        gguf.tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.shape.get(1).map(|&s| s as u64))
    });
    let max_seq_len = read_u(&format!("{prefix}.context_length"));
    let rope_theta = read_f(&format!("{prefix}.rope.freq_base"));
    let rms_eps = read_f(&format!("{prefix}.attention.layer_norm_rms_epsilon"));
    let head_dim = read_u(&format!("{prefix}.attention.key_length")).or_else(|| {
        // Fall back: head_dim = dim / n_heads.
        dim.zip(n_heads).map(|(d, h)| if h > 0 { d / h } else { d })
    });
    let bos = read_u("tokenizer.ggml.bos_token_id").unwrap_or(1);
    let eos = read_u("tokenizer.ggml.eos_token_id").unwrap_or(2);

    let mut cfg = serde_json::Map::new();
    cfg.insert(
        "model_type".to_string(),
        serde_json::Value::from(arch_str.to_string()),
    );
    if let Some(v) = dim {
        cfg.insert("hidden_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = n_layers {
        cfg.insert("num_hidden_layers".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = n_heads {
        cfg.insert(
            "num_attention_heads".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = n_kv_heads {
        cfg.insert(
            "num_key_value_heads".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = hidden_dim {
        cfg.insert("intermediate_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = vocab_size {
        cfg.insert("vocab_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = max_seq_len {
        cfg.insert(
            "max_position_embeddings".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = rope_theta {
        cfg.insert("rope_theta".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = rms_eps {
        cfg.insert("rms_norm_eps".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = head_dim {
        cfg.insert("head_dim".to_string(), serde_json::Value::from(v));
    }
    if arch_id == 13 {
        apply_gemma4_fields(gguf, prefix, &mut cfg);
    }
    cfg.insert("bos_token_id".to_string(), serde_json::Value::from(bos));
    cfg.insert("eos_token_id".to_string(), serde_json::Value::from(eos));
    serde_json::Value::Object(cfg)
}

/// Translate gemma4-specific GGUF metadata into the `text_config` fields the
/// `hipfire-arch-gemma4` loader expects. The generic `config_json_from_gguf`
/// path only reads Llama-style scalar keys; gemma4 stores its layout as arrays
/// and dual sliding/full keys that the generic reads miss or misread:
///
/// - `attention.sliding_window_pattern` (bool array) -> `layer_types`
///   (the loader hard-requires it; GGUF has no `layer_types` key).
/// - `attention.head_count_kv` is an ARRAY (8 sliding / 1 full layers);
///   the generic scalar read fails and falls back to n_heads. Split it into
///   `num_key_value_heads` (sliding) / `num_global_key_value_heads` (full).
/// - `attention.key_length` = 512 is the FULL head dim; sliding is
///   `key_length_swa` = 256. Generic wrote key_length into `head_dim`.
/// - Dual rope: `rope.freq_base_swa` (10k) vs `rope.freq_base` (1M) plus
///   proportional partial rotary on full layers -> `rope_parameters`.
/// - `attention_k_eq_v`: 12B-class checkpoints ship no `attn_v` tensor on
///   full layers (V = pre-k_norm K). Detect from the tensor table.
fn apply_gemma4_fields(
    gguf: &gguf_input::GgufFile,
    prefix: &str,
    cfg: &mut serde_json::Map<String, serde_json::Value>,
) {
    let read_u = |k: &str| -> Option<u64> {
        gguf.metadata.get(k).and_then(|v| match v {
            gguf_input::MetaValue::U8(x) => Some(*x as u64),
            gguf_input::MetaValue::I8(x) => Some(*x as u64),
            gguf_input::MetaValue::U16(x) => Some(*x as u64),
            gguf_input::MetaValue::I16(x) => Some(*x as u64),
            gguf_input::MetaValue::U32(x) => Some(*x as u64),
            gguf_input::MetaValue::I32(x) => Some(*x as u64),
            gguf_input::MetaValue::U64(x) => Some(*x),
            gguf_input::MetaValue::I64(x) => Some(*x as u64),
            _ => None,
        })
    };
    let read_f = |k: &str| -> Option<f64> {
        gguf.metadata.get(k).and_then(|v| match v {
            gguf_input::MetaValue::F32(x) => Some(*x as f64),
            gguf_input::MetaValue::F64(x) => Some(*x),
            _ => None,
        })
    };
    let read_arr_u = |k: &str| -> Option<Vec<u64>> {
        gguf.metadata.get(k).and_then(|v| match v {
            gguf_input::MetaValue::Array(arr) => arr
                .iter()
                .map(|item| match item {
                    gguf_input::MetaValue::U8(x) => Some(*x as u64),
                    gguf_input::MetaValue::I8(x) => Some(*x as u64),
                    gguf_input::MetaValue::U16(x) => Some(*x as u64),
                    gguf_input::MetaValue::I16(x) => Some(*x as u64),
                    gguf_input::MetaValue::U32(x) => Some(*x as u64),
                    gguf_input::MetaValue::I32(x) => Some(*x as u64),
                    gguf_input::MetaValue::U64(x) => Some(*x),
                    gguf_input::MetaValue::I64(x) => Some(*x as u64),
                    _ => None,
                })
                .collect::<Option<Vec<u64>>>(),
            _ => None,
        })
    };
    let ins_u = |cfg: &mut serde_json::Map<String, serde_json::Value>, k: &str, v: u64| {
        cfg.insert(k.to_string(), serde_json::Value::from(v));
    };
    let ins_f = |cfg: &mut serde_json::Map<String, serde_json::Value>, k: &str, v: f64| {
        cfg.insert(k.to_string(), serde_json::Value::from(v));
    };

    // ── layer_types from the sliding-window pattern (required by loader) ──
    // GGUF: true = sliding, false = full (global). Loader strings:
    // "sliding_attention" / "full_attention".
    let pattern = gguf
        .metadata
        .get(&format!("{prefix}.attention.sliding_window_pattern"))
        .and_then(|v| match v {
            gguf_input::MetaValue::Array(arr) => arr
                .iter()
                .map(|item| match item {
                    gguf_input::MetaValue::Bool(b) => Some(*b),
                    _ => None,
                })
                .collect::<Option<Vec<bool>>>(),
            _ => None,
        });
    let full_layer_indices: Vec<usize> = pattern
        .as_ref()
        .map(|p| {
            p.iter()
                .enumerate()
                .filter(|(_, b)| !**b)
                .map(|(i, _)| i)
                .collect()
        })
        .unwrap_or_default();
    if let Some(p) = &pattern {
        let layer_types: Vec<&str> = p
            .iter()
            .map(|&b| {
                if b {
                    "sliding_attention"
                } else {
                    "full_attention"
                }
            })
            .collect();
        cfg.insert(
            "layer_types".to_string(),
            serde_json::Value::Array(
                layer_types
                    .into_iter()
                    .map(serde_json::Value::from)
                    .collect(),
            ),
        );
    }

    // ── KV heads: array form is per-layer (sliding/full differ) ──
    let kv_arr = read_arr_u(&format!("{prefix}.attention.head_count_kv"));
    match (&kv_arr, &pattern) {
        (Some(arr), Some(p)) if arr.len() == p.len() && !arr.is_empty() => {
            // Sliding value: first element at a sliding index (pattern true).
            if let Some((_, &v)) = p.iter().zip(arr.iter()).find(|(&b, _)| b) {
                ins_u(cfg, "num_key_value_heads", v);
            }
            // Full value: element at a full index (pattern false).
            if let Some((_, &v)) = p.iter().zip(arr.iter()).find(|(&b, _)| !b) {
                ins_u(cfg, "num_global_key_value_heads", v);
            }
        }
        _ => {}
    }

    // ── Head dims: key_length is the FULL dim; key_length_swa is sliding ──
    if let Some(v) = read_u(&format!("{prefix}.attention.key_length_swa")) {
        ins_u(cfg, "head_dim", v);
    }
    if let Some(v) = read_u(&format!("{prefix}.attention.key_length")) {
        ins_u(cfg, "global_head_dim", v);
    }

    // ── Simple scalar carries the generic path missed ──
    if let Some(v) = read_u(&format!("{prefix}.attention.sliding_window")) {
        ins_u(cfg, "sliding_window", v);
    }
    if let Some(v) = read_u(&format!("{prefix}.attention.shared_kv_layers")) {
        ins_u(cfg, "num_kv_shared_layers", v);
    }
    if let Some(v) = read_u(&format!("{prefix}.embedding_length_per_layer_input")) {
        ins_u(cfg, "hidden_size_per_layer_input", v);
    }
    if let Some(v) = read_f(&format!("{prefix}.final_logit_softcapping")) {
        ins_f(cfg, "final_logit_softcapping", v);
    }

    // ── Dual rope parameters ──
    // Sliding: default rope at freq_base_swa (10k). Full: proportional
    // partial rotary at freq_base (1M). partial_rotary_factor has no GGUF
    // key in the wild; 0.25 is the family constant (arch config documents
    // it alongside the 1M full theta) — read the GGUF key if a exporter
    // ever ships one.
    let sliding_theta = read_f(&format!("{prefix}.rope.freq_base_swa"));
    let full_theta = read_f(&format!("{prefix}.rope.freq_base"));
    if sliding_theta.is_some() || full_theta.is_some() {
        let partial = read_f(&format!("{prefix}.rope.partial_rotary_factor"));
        let mut sliding = serde_json::Map::new();
        if let Some(t) = sliding_theta {
            sliding.insert("rope_theta".to_string(), serde_json::Value::from(t));
        }
        sliding.insert("rope_type".to_string(), serde_json::Value::from("default"));
        let mut full = serde_json::Map::new();
        if let Some(t) = full_theta {
            full.insert("rope_theta".to_string(), serde_json::Value::from(t));
        }
        full.insert(
            "rope_type".to_string(),
            serde_json::Value::from("proportional"),
        );
        full.insert(
            "partial_rotary_factor".to_string(),
            serde_json::Value::from(partial.unwrap_or(0.25)),
        );
        let mut rp = serde_json::Map::new();
        rp.insert(
            "sliding_attention".to_string(),
            serde_json::Value::Object(sliding),
        );
        rp.insert(
            "full_attention".to_string(),
            serde_json::Value::Object(full),
        );
        cfg.insert("rope_parameters".to_string(), serde_json::Value::Object(rp));
    }

    // ── attention_k_eq_v: no attn_v tensor on full layers (12B class) ──
    if let Some(&first_full) = full_layer_indices.first() {
        let has_v_on_full = gguf
            .tensors
            .iter()
            .any(|t| t.name == format!("blk.{first_full}.attn_v.weight"));
        if !has_v_on_full {
            cfg.insert(
                "attention_k_eq_v".to_string(),
                serde_json::Value::from(true),
            );
        }
    }
}

/// Translate the GGUF metadata HashMap into a JSON object that ends up in
/// the `.hfq` header's metadata blob. A future engine-side `from_hfq` for
/// Llama-style models can read these fields the same way the existing
/// `from_gguf` reads them today.
pub(crate) fn gguf_meta_to_json(meta: &HashMap<String, gguf_input::MetaValue>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, v) in meta {
        let json_v = mv_to_json(v);
        map.insert(k.clone(), json_v);
    }
    serde_json::Value::Object(map)
}

pub(crate) fn mv_to_json(v: &gguf_input::MetaValue) -> serde_json::Value {
    use gguf_input::MetaValue as MV;
    match v {
        MV::U8(x) => serde_json::Value::from(*x),
        MV::I8(x) => serde_json::Value::from(*x),
        MV::U16(x) => serde_json::Value::from(*x),
        MV::I16(x) => serde_json::Value::from(*x),
        MV::U32(x) => serde_json::Value::from(*x),
        MV::I32(x) => serde_json::Value::from(*x),
        MV::F32(x) => serde_json::Value::from(*x),
        MV::Bool(x) => serde_json::Value::from(*x),
        MV::String(s) => serde_json::Value::from(s.clone()),
        MV::U64(x) => serde_json::Value::from(*x),
        MV::I64(x) => serde_json::Value::from(*x),
        MV::F64(x) => serde_json::Value::from(*x),
        // Tokenizer arrays (tokens, scores, merges, ...) can be huge —
        // serialize them as JSON arrays so the engine side can re-parse.
        MV::Array(arr) => serde_json::Value::Array(arr.iter().map(mv_to_json).collect()),
    }
}
#[cfg(test)]
mod gemma4_config_tests {
    use crate::calibration::config_json_from_gguf;
    use crate::gguf_input::{GgufFile, MetaValue, TensorInfo};
    use std::collections::HashMap;

    /// Metadata + tensor table mirroring the real gemma-4-12b GGUF layout
    /// (48 layers, 5:1 sliding:full, per-layer kv array, dual rope, no
    /// attn_v on full layers). Values taken from a production file.
    fn gemma4_12b_gguf() -> GgufFile {
        let mut m: HashMap<String, MetaValue> = HashMap::new();
        let kv_arr: Vec<MetaValue> = (0..48)
            .map(|i| MetaValue::U32(if i % 6 == 5 { 1 } else { 8 }))
            .collect();
        let pattern: Vec<MetaValue> = (0..48).map(|i| MetaValue::Bool(i % 6 != 5)).collect();
        m.insert("gemma4.attention.head_count".into(), MetaValue::U32(16));
        m.insert(
            "gemma4.attention.head_count_kv".into(),
            MetaValue::Array(kv_arr),
        );
        m.insert("gemma4.attention.key_length".into(), MetaValue::U32(512));
        m.insert(
            "gemma4.attention.key_length_swa".into(),
            MetaValue::U32(256),
        );
        m.insert(
            "gemma4.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(1e-6),
        );
        m.insert(
            "gemma4.attention.shared_kv_layers".into(),
            MetaValue::U32(0),
        );
        m.insert(
            "gemma4.attention.sliding_window".into(),
            MetaValue::U32(1024),
        );
        m.insert(
            "gemma4.attention.sliding_window_pattern".into(),
            MetaValue::Array(pattern),
        );
        m.insert("gemma4.block_count".into(), MetaValue::U32(48));
        m.insert("gemma4.context_length".into(), MetaValue::U32(262144));
        m.insert("gemma4.embedding_length".into(), MetaValue::U32(3840));
        m.insert(
            "gemma4.embedding_length_per_layer_input".into(),
            MetaValue::U32(0),
        );
        m.insert("gemma4.feed_forward_length".into(), MetaValue::U32(15360));
        m.insert(
            "gemma4.final_logit_softcapping".into(),
            MetaValue::F32(30.0),
        );
        m.insert("gemma4.rope.freq_base".into(), MetaValue::F64(1_000_000.0));
        m.insert("gemma4.rope.freq_base_swa".into(), MetaValue::F64(10_000.0));
        m.insert(
            "general.architecture".into(),
            MetaValue::String("gemma4".into()),
        );

        let mut tensors = vec![TensorInfo {
            name: "token_embd.weight".into(),
            shape: vec![3840, 262144],
            ..fake_tensor()
        }];
        for i in 0..48 {
            tensors.push(TensorInfo {
                name: format!("blk.{i}.attn_k.weight"),
                ..fake_tensor()
            });
            if i % 6 != 5 {
                // v tensor only on sliding layers (12B k_eq_v layout).
                tensors.push(TensorInfo {
                    name: format!("blk.{i}.attn_v.weight"),
                    ..fake_tensor()
                });
            }
        }
        GgufFile::for_tests(m, tensors).unwrap()
    }

    fn fake_tensor() -> TensorInfo {
        TensorInfo {
            name: String::new(),
            shape: vec![1, 1],
            offset: 0,
            dtype: crate::gguf_input::GgmlType::F32,
        }
    }

    #[test]
    fn gemma4_gguf_config_has_layout_fields() {
        let cfg = config_json_from_gguf(&gemma4_12b_gguf(), "gemma4", 13);
        let lt = cfg["layer_types"].as_array().unwrap();
        assert_eq!(lt.len(), 48);
        for (i, v) in lt.iter().enumerate() {
            let expect = if i % 6 == 5 {
                "full_attention"
            } else {
                "sliding_attention"
            };
            assert_eq!(v.as_str().unwrap(), expect, "layer {i}");
        }
        assert_eq!(cfg["num_key_value_heads"].as_u64(), Some(8));
        assert_eq!(cfg["num_global_key_value_heads"].as_u64(), Some(1));
        assert_eq!(cfg["head_dim"].as_u64(), Some(256));
        assert_eq!(cfg["global_head_dim"].as_u64(), Some(512));
        assert_eq!(cfg["sliding_window"].as_u64(), Some(1024));
        assert_eq!(cfg["final_logit_softcapping"].as_f64(), Some(30.0));
        assert_eq!(cfg["attention_k_eq_v"].as_bool(), Some(true));
        let rp = &cfg["rope_parameters"];
        assert_eq!(
            rp["sliding_attention"]["rope_theta"].as_f64(),
            Some(10_000.0)
        );
        assert_eq!(
            rp["sliding_attention"]["rope_type"].as_str(),
            Some("default")
        );
        assert_eq!(
            rp["full_attention"]["rope_theta"].as_f64(),
            Some(1_000_000.0)
        );
        assert_eq!(
            rp["full_attention"]["rope_type"].as_str(),
            Some("proportional")
        );
        assert_eq!(
            rp["full_attention"]["partial_rotary_factor"].as_f64(),
            Some(0.25)
        );
        // Generic fields still populated.
        assert_eq!(cfg["hidden_size"].as_u64(), Some(3840));
        assert_eq!(cfg["num_hidden_layers"].as_u64(), Some(48));
        assert_eq!(cfg["num_attention_heads"].as_u64(), Some(16));
        assert_eq!(cfg["intermediate_size"].as_u64(), Some(15360));
        assert_eq!(cfg["vocab_size"].as_u64(), Some(262144));
    }

    #[test]
    fn gemma4_gguf_config_is_admitted_by_loader_parser() {
        let cfg = config_json_from_gguf(&gemma4_12b_gguf(), "gemma4", 13);
        let metadata_json = serde_json::to_string(&serde_json::json!({ "config": cfg })).unwrap();
        let parsed = hipfire_arch_gemma4::config::Gemma4Config::from_metadata_json(&metadata_json)
            .expect("loader parser must admit the generated config");
        assert_eq!(parsed.n_layers, 48);
        assert_eq!(parsed.n_full_layers(), 8);
        assert_eq!(parsed.n_sliding_layers(), 40);
        assert_eq!(parsed.sliding_head_dim, 256);
        assert_eq!(parsed.full_head_dim, 512);
        assert_eq!(parsed.sliding_n_kv_heads, 8);
        assert_eq!(parsed.full_n_kv_heads, 1);
        assert!(parsed.attention_k_eq_v);
        assert_eq!(
            parsed.layer_types[5],
            hipfire_arch_gemma4::config::LayerType::Full
        );
        assert_eq!(
            parsed.layer_types[0],
            hipfire_arch_gemma4::config::LayerType::Sliding
        );
    }

    #[test]
    fn gemma4_with_attn_v_on_full_layers_is_not_k_eq_v() {
        let mut gguf = gemma4_12b_gguf();
        gguf.tensors.push(TensorInfo {
            name: "blk.5.attn_v.weight".into(),
            ..fake_tensor()
        });
        let cfg = config_json_from_gguf(&gguf, "gemma4", 13);
        assert!(cfg.get("attention_k_eq_v").is_none());
    }

    #[test]
    fn non_gemma4_arch_is_untouched() {
        let mut m: HashMap<String, MetaValue> = HashMap::new();
        m.insert("llama.embedding_length".into(), MetaValue::U32(4096));
        m.insert("llama.block_count".into(), MetaValue::U32(32));
        m.insert("llama.attention.head_count".into(), MetaValue::U32(32));
        m.insert("llama.attention.head_count_kv".into(), MetaValue::U32(8));
        m.insert("llama.attention.key_length".into(), MetaValue::U32(128));
        m.insert("llama.feed_forward_length".into(), MetaValue::U32(11008));
        m.insert("llama.rope.freq_base".into(), MetaValue::F32(10000.0));
        let gguf = GgufFile::for_tests(m, vec![]).unwrap();
        let cfg = config_json_from_gguf(&gguf, "llama", 0);
        assert!(cfg.get("layer_types").is_none());
        assert!(cfg.get("rope_parameters").is_none());
        assert!(cfg.get("attention_k_eq_v").is_none());
        assert_eq!(cfg["num_key_value_heads"].as_u64(), Some(8));
        assert_eq!(cfg["head_dim"].as_u64(), Some(128));
    }

    /// gemma4_text is a distinct GGUF architecture string that still maps to
    /// arch_id 13. Metadata keys use the raw string as prefix (`gemma4_text.*`),
    /// but the translation gate must engage via arch_id — not `arch_str == "gemma4"`.
    fn gemma4_text_12b_gguf() -> GgufFile {
        let mut m: HashMap<String, MetaValue> = HashMap::new();
        let kv_arr: Vec<MetaValue> = (0..48)
            .map(|i| MetaValue::U32(if i % 6 == 5 { 1 } else { 8 }))
            .collect();
        let pattern: Vec<MetaValue> = (0..48).map(|i| MetaValue::Bool(i % 6 != 5)).collect();
        m.insert("gemma4_text.attention.head_count".into(), MetaValue::U32(16));
        m.insert(
            "gemma4_text.attention.head_count_kv".into(),
            MetaValue::Array(kv_arr),
        );
        m.insert("gemma4_text.attention.key_length".into(), MetaValue::U32(512));
        m.insert(
            "gemma4_text.attention.key_length_swa".into(),
            MetaValue::U32(256),
        );
        m.insert(
            "gemma4_text.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(1e-6),
        );
        m.insert(
            "gemma4_text.attention.shared_kv_layers".into(),
            MetaValue::U32(0),
        );
        m.insert(
            "gemma4_text.attention.sliding_window".into(),
            MetaValue::U32(1024),
        );
        m.insert(
            "gemma4_text.attention.sliding_window_pattern".into(),
            MetaValue::Array(pattern),
        );
        m.insert("gemma4_text.block_count".into(), MetaValue::U32(48));
        m.insert("gemma4_text.context_length".into(), MetaValue::U32(262144));
        m.insert("gemma4_text.embedding_length".into(), MetaValue::U32(3840));
        m.insert(
            "gemma4_text.embedding_length_per_layer_input".into(),
            MetaValue::U32(0),
        );
        m.insert("gemma4_text.feed_forward_length".into(), MetaValue::U32(15360));
        m.insert(
            "gemma4_text.final_logit_softcapping".into(),
            MetaValue::F32(30.0),
        );
        m.insert(
            "gemma4_text.rope.freq_base".into(),
            MetaValue::F64(1_000_000.0),
        );
        m.insert(
            "gemma4_text.rope.freq_base_swa".into(),
            MetaValue::F64(10_000.0),
        );
        m.insert(
            "general.architecture".into(),
            MetaValue::String("gemma4_text".into()),
        );

        let mut tensors = vec![TensorInfo {
            name: "token_embd.weight".into(),
            shape: vec![3840, 262144],
            ..fake_tensor()
        }];
        for i in 0..48 {
            tensors.push(TensorInfo {
                name: format!("blk.{i}.attn_k.weight"),
                ..fake_tensor()
            });
            if i % 6 != 5 {
                tensors.push(TensorInfo {
                    name: format!("blk.{i}.attn_v.weight"),
                    ..fake_tensor()
                });
            }
        }
        GgufFile::for_tests(m, tensors).unwrap()
    }

    #[test]
    fn gemma4_text_arch_engages_translation() {
        let cfg = config_json_from_gguf(&gemma4_text_12b_gguf(), "gemma4_text", 13);
        let lt = cfg["layer_types"].as_array().expect(
            "gemma4_text (arch_id 13) must emit layer_types; gating on raw arch_str skips it",
        );
        assert_eq!(lt.len(), 48);
        assert_eq!(cfg["num_key_value_heads"].as_u64(), Some(8));
        assert_eq!(cfg["num_global_key_value_heads"].as_u64(), Some(1));
        assert_eq!(cfg["head_dim"].as_u64(), Some(256));
        assert_eq!(cfg["global_head_dim"].as_u64(), Some(512));
        assert_eq!(cfg["attention_k_eq_v"].as_bool(), Some(true));
        assert_eq!(cfg["model_type"].as_str(), Some("gemma4_text"));
    }
}

#[cfg(test)]
mod gemma4_name_translation_tests {
    use crate::calibration::gguf_to_safetensors_name;

    #[test]
    fn gemma4_sandwich_norms_map_to_loader_names() {
        let f = |slot: &str| {
            gguf_to_safetensors_name(&format!("blk.7.{slot}.weight"), 13).unwrap()
        };
        assert_eq!(f("attn_norm"), "model.layers.7.input_layernorm.weight");
        assert_eq!(
            f("post_attention_norm"),
            "model.layers.7.post_attention_layernorm.weight"
        );
        assert_eq!(
            f("ffn_norm"),
            "model.layers.7.pre_feedforward_layernorm.weight"
        );
        assert_eq!(
            f("post_ffw_norm"),
            "model.layers.7.post_feedforward_layernorm.weight"
        );
        // Loader reads `{p}.layer_scalar` with NO `.weight` suffix.
        assert_eq!(f("layer_output_scale"), "model.layers.7.layer_scalar");
        // Generic slots unchanged.
        assert_eq!(f("attn_q"), "model.layers.7.self_attn.q_proj.weight");
        assert_eq!(f("ffn_down"), "model.layers.7.mlp.down_proj.weight");
    }

    #[test]
    fn llama_ffn_norm_still_maps_to_post_attention() {
        // The two-norm Llama layout must be byte-identical to before the
        // gemma4 arm existed — ffn_norm IS post_attention_layernorm there.
        assert_eq!(
            gguf_to_safetensors_name("blk.3.ffn_norm.weight", 0).unwrap(),
            "model.layers.3.post_attention_layernorm.weight"
        );
        // Gemma-4-only slots do not exist in Llama files; translation of an
        // unexpected slot passes through untouched (no gemma4 arm taken).
        assert_eq!(
            gguf_to_safetensors_name("blk.3.post_ffw_norm.weight", 0).unwrap(),
            "model.layers.3.post_ffw_norm.weight"
        );
    }

    #[test]
    fn gemma4_text_arch_id_engages_sandwich_norms() {
        // arch_id 13 covers gemma4_text / gemma4_unified / gemma4_unified_text;
        // the gate must not require the literal arch_str "gemma4".
        assert_eq!(
            gguf_to_safetensors_name("blk.2.ffn_norm.weight", 13).unwrap(),
            "model.layers.2.pre_feedforward_layernorm.weight"
        );
        assert_eq!(
            gguf_to_safetensors_name("blk.2.layer_output_scale.weight", 13).unwrap(),
            "model.layers.2.layer_scalar"
        );
    }
}

#[cfg(test)]
mod block_i4_128_tests {
    use super::fake_quantize_block_i4_128;

    /// Fixture dumped on gfx1201 by launching the shipping
    /// `quantize_int4_mmq_ds128` kernel from `block_i4_128_quant.hip`.
    #[test]
    fn cpu_fake_quant_matches_dumped_gfx1201_block_bit_for_bit() {
        let mut input = [0.0f32; 128];
        let mut state = 0x1234_5678u32;
        for value in &mut input {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let centered = ((state >> 8) & 0xffff) as i32 - 32_768;
            *value = centered as f32 * (1.0f32 / 4096.0f32);
        }
        let expected = [
            0xe5, 0x54, 0x7a, 0x3f, 0x05, 0x00, 0x00, 0x00, 0xbc, 0xe6, 0x36, 0x59, 0x5b,
            0xd0, 0x8f, 0x68, 0x01, 0x3e, 0x48, 0x9d, 0x72, 0x1e, 0x70, 0xfc, 0xc5, 0x1c,
            0xfa, 0x7b, 0x2f, 0x0a, 0xdc, 0x5a, 0x18, 0x1a, 0xb4, 0x7d, 0x15, 0xc2, 0xae,
            0x60, 0x3f, 0x17, 0x22, 0x43, 0x4a, 0xe5, 0x25, 0x70, 0xa1, 0x06, 0x27, 0x91,
            0xc8, 0x19, 0xd5, 0x2e, 0x5b, 0x6c, 0xb6, 0xbe, 0x1e, 0x45, 0x63, 0x25, 0x4c,
            0xa6, 0x4a, 0x07, 0xec, 0xa7, 0xec, 0x62,
        ];
        assert_eq!(fake_quantize_block_i4_128(&input, 4).to_bytes(), expected);
        // This block also matched a separate fused c2 gfx1201 kernel dump.
        assert_eq!(fake_quantize_block_i4_128(&input, 2).to_bytes(), expected);
    }
}

#[cfg(test)]
mod symmetric_awq_writer_tests {
    use super::fake_quantize_mq4v2_weight_group;
    use crate::quant_fwht::{cpu_fwht_256, quantize_mq4g256v2_symmetric};
    use hipfire_quantize::float16::f16_to_f32;

    #[test]
    fn symmetric_surrogate_decodes_exact_writer_headers_and_codes() {
        let signs1 = [1.0f32; 256];
        let signs2 = [1.0f32; 256];
        for seed in [0u32, 37, 0xffff_ffff] {
            let mut state = seed;
            let mut input = [0.0f32; 256];
            for x in &mut input {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *x = if seed == 0 { 0.0 } else { ((state >> 8) as i32 % 1021) as f32 / 137.0 };
            }
            let packed = quantize_mq4g256v2_symmetric(&input, 1, 256, &signs1, &signs2);
            let mut surrogate = input;
            cpu_fwht_256(&mut surrogate, &signs1, &signs2);
            fake_quantize_mq4v2_weight_group(&mut surrogate, true);
            for (i, &value) in surrogate.iter().enumerate() {
                let h = i / 128;
                let scale = f16_to_f32(u16::from_le_bytes(packed[h * 4..h * 4 + 2].try_into().unwrap()));
                let zero = f16_to_f32(u16::from_le_bytes(packed[h * 4 + 2..h * 4 + 4].try_into().unwrap()));
                let code_byte = packed[8 + i / 2];
                let code = if i & 1 == 0 { code_byte & 15 } else { code_byte >> 4 };
                assert_eq!(value.to_bits(), (code as f32).mul_add(scale, zero).to_bits(),
                    "writer mismatch seed={seed} index={i}");
            }
        }
    }
    #[test]
    fn c2_uses_the_fused_subset_not_the_legacy_four_point_grid() {
        let mut input = [0.0f32; 128];
        for (i, value) in input.iter_mut().enumerate() {
            *value = if i == 0 { 7.0 } else if i & 1 == 0 { 0.53 } else { -0.53 };
        }
        let c2 = super::fake_quantize_block_i4_128(&input, 2);
        let c4 = super::fake_quantize_block_i4_128(&input, 4);
        assert_ne!(c2.d.to_bits(), c4.d.to_bits(), "c2 must omit the c4-only scales");
        // Dumped by quantize_block_i4_128_wave<true> on gfx1201 with
        // IU4_A4_CANDIDATES=2, including the wave-reduced signed sum.
        let mut expected = [0xf1u8; 72];
        expected[..8].copy_from_slice(&[0xb7, 0x6d, 0x5b, 0x3f, 0x06, 0, 0, 0]);
        expected[8] = 0xf7;
        assert_eq!(c2.to_bytes(), expected);
    }
}

#[cfg(test)]
mod mq4v2_final_code_record_tests {
    use super::{
        sha256_file_hex, validate_mq4v2_final_code_record, Mq4v2FinalCodeRecord,
        MQ4V2_FINAL_CODE_QTYPE,
    };
    use hipfire_quantize::float16::f32_to_f16;
    use std::io::Write;

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&value| f32_to_f16(value).to_le_bytes())
            .collect()
    }

    #[test]
    fn final_code_record_validator_enforces_frozen_contract() {
        let source_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let s_f16 = f16_bytes(&vec![1.0; 256]);
        let d_z_f16 = f16_bytes(&[0.25, -1.0, 0.5, -2.0]);
        let codes_u8 = (0..256).map(|index| (index % 16) as u8).collect::<Vec<_>>();
        let valid = Mq4v2FinalCodeRecord {
            name: "model.layers.0.mlp.gate_proj.weight",
            m: 1,
            k: 256,
            qt: MQ4V2_FINAL_CODE_QTYPE,
            source_sha,
            s_f16: &s_f16,
            d_z_f16: &d_z_f16,
            codes_u8: &codes_u8,
        };
        validate_mq4v2_final_code_record(&valid, source_sha).unwrap();

        let mut bad_codes = codes_u8.clone();
        bad_codes[37] = 16;
        let bad_nibble = Mq4v2FinalCodeRecord {
            codes_u8: &bad_codes,
            ..valid
        };
        assert!(validate_mq4v2_final_code_record(&bad_nibble, source_sha)
            .unwrap_err()
            .contains("uint4 range"));

        let wrong_source = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        assert!(validate_mq4v2_final_code_record(&valid, wrong_source)
            .unwrap_err()
            .contains("does not match input artifact"));

        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"abc").unwrap();
        assert_eq!(
            sha256_file_hex(file.path()).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}