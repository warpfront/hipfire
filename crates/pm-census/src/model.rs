// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Model descriptor (see /home/kaden/qcal/release-0.4.3/pmrip/S1-SPEC.md): the
//! model `config.json` integers and the `.hfq` tensor directory (header only,
//! never the payload) as weight groups. Std + serde_json only; no model names.
//!
//! The `.hfq` container layout is the one `crates/hipfire-runtime/src/hfq.rs`
//! (`HfqFile::open*`) reads: 32 B header (`HFQM`, u32 version with bit 31 =
//! metadata overlay, u32 arch id, u32 tensor count, u64 metadata offset, u64
//! data offset), the metadata JSON at `metadata offset` (end found by brace
//! matching honouring strings and escapes), then the tensor index
//! (`u32 n; per tensor: u16 name_len, name, u8 quant_type, u8 n_dims,
//! n_dims x u32 shape, u32 group_size, u64 data_size`). An overlay replaces the
//! metadata wholesale and lives at the file tail (`metadata bytes, u64 len, u32
//! schema, 32 B sha256, "HFQMDOV1"`).

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde_json::Value;
use sha2::{Digest, Sha256};

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_METADATA_OVERLAY_FLAG: u32 = 1 << 31;
const HFQ_BASE_FORMAT_VERSION: u32 = 1;
const HFQ_METADATA_OVERLAY_SCHEMA: u32 = 1;
const HFQ_METADATA_OVERLAY_MAX_BYTES: u64 = 64 * 1024 * 1024;
const HFQ_METADATA_OVERLAY_MAGIC: &[u8; 8] = b"HFQMDOV1";
const HFQ_METADATA_OVERLAY_FOOTER_LEN: u64 = 8 + 4 + 32 + 8;
/// Refuse a header region larger than this (corrupt `data_offset`).
const HFQ_MAX_HEADER_BYTES: u64 = 1 << 30;

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// One tensor of the `.hfq` directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub quant_type: u8,
    pub codec: &'static str,
    /// Row-major shape as stored; 2-D = `[rows (output features), cols (K)]`.
    pub shape: Vec<u64>,
}

/// Header-only view of an `.hfq` file.
#[derive(Clone, Debug)]
pub struct HfqIndex {
    pub path: String,
    pub arch_id: u32,
    /// Metadata JSON (the overlay's when the file carries one).
    pub metadata: Value,
    pub tensors: Vec<TensorInfo>,
    /// SHA-256 of the header region (`[0, data_offset)`: header, metadata,
    /// tensor index).
    pub index_sha256: String,
    pub data_offset: u64,
}

/// Tensors that share a name pattern, shape and codec (or a fused sibling
/// row-concatenation of such groups).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightGroup {
    /// Output features (rows of `[rows, cols]`; summed over `members` if fused).
    pub rows: u64,
    pub cols: u64,
    /// Codec of the members; `"mixed"` for a fused group of unequal codecs.
    pub codec: &'static str,
    /// The name has a stack index: one launch reads the whole stack of a layer.
    pub stacked: bool,
    /// Row-concatenation of 2 or 3 sibling groups.
    pub fused: bool,
    /// Distinct layer indices (1 when the names carry no numeric component).
    pub layers: u64,
    /// Name patterns: first numeric component -> `*` (layer), any later one ->
    /// `#` (stack). One pattern unless `fused`.
    pub members: Vec<String>,
    /// Launch units per layer: tensors per layer when unstacked, **1** when
    /// `stacked` (the whole stack is one unit; its length is `stack_len`).
    pub count_per_layer: u64,
    /// Tensors per stack of one layer (1 when not stacked; the largest per-layer
    /// stack otherwise).
    pub stack_len: u64,
}

/// Config + optional `.hfq` header.
#[derive(Clone, Debug)]
pub struct ModelDesc {
    pub config: Value,
    pub config_sha256: String,
    pub hfq: Option<HfqIndex>,
}

/// hfq `quant_type` byte -> lower-case codec name. Reconciled with
/// `quant_type_to_dtype` (crates/hipfire-runtime/src/hfq.rs), the `QuantType`
/// enum (crates/hipfire-quantize/src/hfq.rs) and `QuantFormat` /
/// `QuantTensorGeometry` (crates/hipfire-arch-qwen4/src/admission.rs): the
/// dtype name lower-cased (`3` = `q8_0`, `44` = `mq4g256v2`, `1` = `f16`,
/// `2` = `f32`, `16` = `bf16`, ...). Unknown -> `"qt<N>"` from a fixed table, so
/// the result stays `&'static str` without leaking.
pub fn codec_name(quant_type: u8) -> &'static str {
    match quant_type {
        0 => "q4f16g64",
        1 => "f16",
        2 => "f32",
        3 => "q8_0",
        4 => "q4k",
        5 => "q8hfq",
        6 => "hfq4g256",
        7 => "hfq4g128",
        8 => "hfq6g256",
        9 => "hfq2g256",
        10 => "hfq2g128",
        11 => "hfq3g256",
        12 => "hfq3g128",
        13 => "mq4g256",
        14 => "mq8g256",
        15 => "mq6g256",
        16 => "bf16",
        17 => "mq3g256",
        18 => "mq2g256",
        19 => "mq2g256lloyd",
        20 => "mq3g256lloyd",
        21 => "hfp4g32",
        22 => "tidi32",
        24 => "mfp4g32",
        28 => "paro4g128",
        29 => "paro4g128t",
        30 => "mq4g256lloyd",
        31 => "mq5g256",
        32 => "mfp4g32lloyd",
        33 => "mfp4g32p",
        34 => "mfp4g32e8",
        35 => "mfp4g32e8soa",
        36 => "mfp3g32e8",
        37 => "mfp2g32e8",
        38 => "mq2g256gl",
        39 => "mq3g256gl",
        40 => "tq2g128",
        41 => "bq1g128",
        42 => "mfp4g32e8g128",
        44 => "mq4g256v2",
        45 => "mq4cg256",
        47 => "mq6g256v2",
        48 => "mq5g256v2",
        49 => "mq3g256v2",
        50 => "mq2g256v2",
        51 => "mq2g256lloydu",
        52 => "mq4g256v2l",
        53 => "mq4g128v2",
        54 => "i64",
        55 => "fp8e4m3b128",
        56 => "fp8e4m3pt",
        131 => "mq3i4g128r256",
        132 => "mq3i4g128r128",
        133 => "mq2i4g128r256",
        134 => "mq2i4g128r128",
        other => UNKNOWN_QT[other as usize],
    }
}

/// `qt<N>` for every byte (a `&'static str` per value without leaking).
const UNKNOWN_QT: [&str; 256] = {
    macro_rules! qt {
        ($($n:literal)*) => { [$(concat!("qt", $n)),*] };
    }
    qt!(
        0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
        32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60
        61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89
        90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113
        114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134
        135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155
        156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176
        177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197
        198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218
        219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239
        240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255
    )
};

fn rd_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn rd_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn rd_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

impl HfqIndex {
    /// Read the header region only (`[0, data_offset)`) plus, for an overlay
    /// file, the tail metadata. Never reads tensor payload.
    pub fn read(path: &Path) -> Result<HfqIndex, String> {
        let shown = path.display();
        let err = |m: String| format!("{shown}: {m}");
        let mut file = std::fs::File::open(path).map_err(|e| err(e.to_string()))?;
        let file_len = file.metadata().map_err(|e| err(e.to_string()))?.len();
        let mut head = [0u8; 32];
        file.read_exact(&mut head)
            .map_err(|e| err(format!("short HFQ header: {e}")))?;
        if &head[0..4] != HFQ_MAGIC {
            return Err(err("not an HFQ container (bad magic)".into()));
        }
        let raw_version = rd_u32(&head, 4);
        let has_overlay = raw_version & HFQ_METADATA_OVERLAY_FLAG != 0;
        let version = raw_version & !HFQ_METADATA_OVERLAY_FLAG;
        if version != HFQ_BASE_FORMAT_VERSION {
            return Err(err(format!(
                "unsupported HFQ format version {version} (raw header word {raw_version})"
            )));
        }
        let arch_id = rd_u32(&head, 8);
        let n_tensors = rd_u32(&head, 12) as usize;
        let metadata_offset = rd_u64(&head, 16);
        let data_offset = rd_u64(&head, 24);
        if metadata_offset < 32 || metadata_offset > data_offset || data_offset > file_len {
            return Err(err(format!(
                "invalid metadata/data offsets ({metadata_offset}, {data_offset}) for a {file_len}-byte file"
            )));
        }
        if data_offset > HFQ_MAX_HEADER_BYTES {
            return Err(err(format!(
                "header region of {data_offset} bytes exceeds the {HFQ_MAX_HEADER_BYTES}-byte cap"
            )));
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|e| err(e.to_string()))?;
        let mut region = vec![0u8; data_offset as usize];
        file.read_exact(&mut region)
            .map_err(|e| err(format!("short header region: {e}")))?;
        let index_sha256 = sha256_hex(&region);

        let meta_bytes = &region[metadata_offset as usize..];
        let json_end = json_object_end(meta_bytes).ok_or_else(|| {
            err(format!(
                "metadata JSON at offset {metadata_offset} is not brace-terminated"
            ))
        })?;
        let mut metadata_text = String::from_utf8_lossy(&meta_bytes[..json_end]).into_owned();
        let mut pos = metadata_offset as usize + json_end;

        let need = |pos: usize, n: usize, what: &str| -> Result<(), String> {
            if pos.checked_add(n).is_none_or(|e| e > region.len()) {
                Err(err(format!("header region ends inside {what}")))
            } else {
                Ok(())
            }
        };
        need(pos, 4, "the tensor-index count")?;
        let idx_n = rd_u32(&region, pos) as usize;
        if idx_n != n_tensors {
            return Err(err(format!(
                "tensor-index count {idx_n} does not match header count {n_tensors}"
            )));
        }
        pos += 4;
        let mut tensors = Vec::with_capacity(n_tensors);
        let mut payload_end = data_offset;
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for _ in 0..n_tensors {
            need(pos, 2, "a tensor name length")?;
            let name_len = rd_u16(&region, pos) as usize;
            pos += 2;
            need(pos, name_len, "a tensor name")?;
            let name = String::from_utf8_lossy(&region[pos..pos + name_len]).into_owned();
            pos += name_len;
            need(pos, 2, "a tensor quant type and dimension count")?;
            let quant_type = region[pos];
            let n_dims = region[pos + 1] as usize;
            pos += 2;
            need(pos, n_dims * 4, "a tensor shape")?;
            let shape: Vec<u64> = (0..n_dims)
                .map(|d| u64::from(rd_u32(&region, pos + d * 4)))
                .collect();
            pos += n_dims * 4;
            need(pos, 12, "a tensor group and data size")?;
            let data_size = rd_u64(&region, pos + 4);
            pos += 12;
            payload_end = payload_end.checked_add(data_size).ok_or_else(|| {
                err(format!(
                    "tensor {name:?} payload size overflows the file range"
                ))
            })?;
            if !seen.insert(name.clone()) {
                return Err(err(format!("duplicate tensor {name:?}")));
            }
            tensors.push(TensorInfo {
                name,
                quant_type,
                codec: codec_name(quant_type),
                shape,
            });
        }
        if payload_end > file_len {
            return Err(err(format!(
                "tensor payload ends at {payload_end}, past the {file_len}-byte file (truncated container)"
            )));
        }
        if has_overlay {
            metadata_text = read_overlay(&mut file, file_len, payload_end).map_err(err)?;
        }
        let metadata: Value =
            serde_json::from_str(&metadata_text).map_err(|e| err(format!("metadata JSON: {e}")))?;
        Ok(HfqIndex {
            path: path.display().to_string(),
            arch_id,
            metadata,
            tensors,
            index_sha256,
            data_offset,
        })
    }
}

/// End (exclusive) of the leading JSON object: brace matching honouring strings
/// and escapes, exactly as `HfqFile` does.
fn json_object_end(bytes: &[u8]) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if b == b'\\' && in_string {
            escape = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            if b == b'{' {
                depth += 1;
            } else if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
        }
    }
    None
}

/// The tail metadata overlay (`read_hfq_metadata_overlay` in the runtime).
fn read_overlay(
    file: &mut std::fs::File,
    file_len: u64,
    payload_end: u64,
) -> Result<String, String> {
    if file_len < HFQ_METADATA_OVERLAY_FOOTER_LEN {
        return Err("HFQ metadata overlay footer is truncated".into());
    }
    let footer_start = file_len - HFQ_METADATA_OVERLAY_FOOTER_LEN;
    file.seek(SeekFrom::Start(footer_start))
        .map_err(|e| e.to_string())?;
    let mut footer = [0u8; HFQ_METADATA_OVERLAY_FOOTER_LEN as usize];
    file.read_exact(&mut footer).map_err(|e| e.to_string())?;
    let len = rd_u64(&footer, 0);
    if len == 0 || len > HFQ_METADATA_OVERLAY_MAX_BYTES {
        return Err(format!(
            "HFQ metadata overlay length {len} is outside 1..={HFQ_METADATA_OVERLAY_MAX_BYTES}"
        ));
    }
    let schema = rd_u32(&footer, 8);
    if schema != HFQ_METADATA_OVERLAY_SCHEMA {
        return Err(format!("unsupported HFQ metadata overlay schema {schema}"));
    }
    if &footer[44..52] != HFQ_METADATA_OVERLAY_MAGIC {
        return Err("HFQ metadata overlay magic is missing or corrupt".into());
    }
    let start = footer_start
        .checked_sub(len)
        .ok_or("HFQ metadata overlay length underflows file bounds")?;
    if start < payload_end {
        return Err(format!(
            "HFQ metadata overlay starts at {start}, before payload end {payload_end}"
        ));
    }
    file.seek(SeekFrom::Start(start))
        .map_err(|e| e.to_string())?;
    let mut bytes = vec![0u8; len as usize];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    if Sha256::digest(&bytes).as_slice() != &footer[12..44] {
        return Err("HFQ metadata overlay checksum mismatch".into());
    }
    let text =
        String::from_utf8(bytes).map_err(|e| format!("HFQ metadata overlay is not UTF-8: {e}"))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|e| format!("HFQ metadata overlay JSON is invalid: {e}"))?;
    if !value.is_object() {
        return Err("HFQ metadata overlay JSON must be an object".into());
    }
    Ok(text)
}

impl ModelDesc {
    /// `config` is the model `config.json`; when that path does not exist (or is
    /// empty) the `hfq` metadata's `"config"` object is used instead.
    pub fn load(config: &Path, hfq: Option<&Path>) -> Result<Self, String> {
        let index = hfq.map(HfqIndex::read).transpose()?;
        let (config, config_sha256) = if !config.as_os_str().is_empty() && config.is_file() {
            let bytes = std::fs::read(config).map_err(|e| format!("{}: {e}", config.display()))?;
            let value: Value = serde_json::from_slice(&bytes)
                .map_err(|e| format!("{}: not JSON: {e}", config.display()))?;
            (value, sha256_hex(&bytes))
        } else if let Some(embedded) = index.as_ref().and_then(|i| i.metadata.get("config")) {
            (
                embedded.clone(),
                sha256_hex(embedded.to_string().as_bytes()),
            )
        } else {
            return Err(format!(
                "{}: no config file and the hfq metadata has no \"config\" object",
                config.display()
            ));
        };
        Ok(ModelDesc {
            config,
            config_sha256,
            hfq: index,
        })
    }

    /// Config taken from the hfq metadata only.
    pub fn from_hfq_only(hfq: &Path) -> Result<Self, String> {
        Self::load(Path::new(""), Some(hfq))
    }

    /// First present key wins. Keys are dotted paths (`rda_config.head_dim`);
    /// each is tried at the root and under `text_config.`. Accepts unsigned
    /// integers and integral floats.
    pub fn config_int(&self, keys: &[&str]) -> Option<u64> {
        for key in keys {
            for prefix in ["", "text_config."] {
                let mut cur = &self.config;
                let mut ok = true;
                for part in prefix
                    .split('.')
                    .chain(key.split('.'))
                    .filter(|p| !p.is_empty())
                {
                    match cur.get(part) {
                        Some(next) => cur = next,
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }
                if let Some(v) = cur.as_u64() {
                    return Some(v);
                }
                if let Some(f) = cur.as_f64() {
                    if f >= 0.0 && f.fract() == 0.0 && f < u64::MAX as f64 {
                        return Some(f as u64);
                    }
                }
            }
        }
        None
    }

    /// Weight groups of the 2-D tensors (empty without an hfq). See the module
    /// rules: names split on `.`; the first numeric component is the layer
    /// index, later numeric components are stack indices.
    pub fn weight_groups(&self) -> Vec<WeightGroup> {
        match &self.hfq {
            Some(h) => weight_groups_of(&h.tensors),
            None => Vec::new(),
        }
    }
}

/// `(pattern tokens, layer index, stack indices)` of a tensor name.
fn pattern_of(name: &str) -> (Vec<String>, Option<u64>, Vec<u64>) {
    let mut toks = Vec::new();
    let mut layer = None;
    let mut stack = Vec::new();
    for part in name.split('.') {
        match part.parse::<u64>() {
            Ok(n) if !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()) => {
                if layer.is_none() {
                    layer = Some(n);
                    toks.push("*".to_string());
                } else {
                    stack.push(n);
                    toks.push("#".to_string());
                }
            }
            _ => toks.push(part.to_string()),
        }
    }
    (toks, layer, stack)
}

/// Launch units of one output group: layer -> distinct stack tuples.
#[derive(Default)]
struct Acc {
    stacked: bool,
    per: BTreeMap<Option<u64>, BTreeSet<Vec<u64>>>,
}

impl Acc {
    fn add(&mut self, layer: Option<u64>, stack: Vec<u64>) {
        self.stacked |= !stack.is_empty();
        self.per.entry(layer).or_default().insert(stack);
    }

    fn finish(
        &self,
        rows: u64,
        cols: u64,
        codec: &'static str,
        fused: bool,
        members: &[String],
    ) -> WeightGroup {
        let per_layer = self.per.values().map(|s| s.len() as u64).max().unwrap_or(1);
        WeightGroup {
            rows,
            cols,
            codec,
            stacked: self.stacked,
            fused,
            layers: self.per.len() as u64,
            members: members.to_vec(),
            count_per_layer: if self.stacked { 1 } else { per_layer },
            stack_len: if self.stacked { per_layer } else { 1 },
        }
    }
}

/// Weight groups of the 2-D tensors. Singles are keyed by (pattern, shape,
/// codec). Fused groups are built per (layer, stack) *instance*. Siblings are
/// tensors of one instance with the same cols, the same stacking and the same
/// name tail after the leaf component (the last non-`weight` token, e.g.
/// `weight`), whose parent module is the same (`a.b.x`, `a.b.y`) or, as a
/// fallback for families split one level deeper (`a.b.x`, `a.b.sub.y`), one
/// component deeper than the other. Every 2- and 3-member combination of
/// siblings contributes (a combination with members from two levels needs at
/// least one member at the shallower level; purely-deeper combinations belong
/// to the deeper parent's own bucket). Instances with the same (members, summed
/// rows, merged codec) form one group, so a fusion exists on exactly the layers
/// where all its members exist, even when a member's codec differs between
/// layers.
pub(crate) fn weight_groups_of(tensors: &[TensorInfo]) -> Vec<WeightGroup> {
    type GroupKey = (Vec<String>, bool, u64, u64, &'static str);
    type BucketKey = (Vec<String>, Vec<String>, u64, bool);
    type Instance = (Option<u64>, Vec<u64>);
    // (pattern, rows, codec, depth below the bucket's parent: 0 = same parent, 1 = one deeper)
    type Member = (String, u64, &'static str, u8);
    let mut groups: BTreeMap<GroupKey, Acc> = BTreeMap::new();
    let mut buckets: BTreeMap<BucketKey, BTreeMap<Instance, Vec<Member>>> = BTreeMap::new();

    for t in tensors.iter().filter(|t| t.shape.len() == 2) {
        let (toks, layer, stack) = pattern_of(&t.name);
        let (rows, cols) = (t.shape[0], t.shape[1]);
        let pattern = toks.join(".");
        let stacked = !stack.is_empty();
        groups
            .entry((vec![pattern.clone()], false, rows, cols, t.codec))
            .or_default()
            .add(layer, stack.clone());
        let Some(leaf) = toks.iter().rposition(|x| x != "weight") else {
            continue;
        };
        if toks[leaf] == "*" || toks[leaf] == "#" {
            continue;
        }
        let parent = toks[..leaf].to_vec();
        let tail = toks[leaf + 1..].to_vec();
        let mut put = |parent: Vec<String>, depth: u8| {
            buckets
                .entry((parent, tail.clone(), cols, stacked))
                .or_default()
                .entry((layer, stack.clone()))
                .or_default()
                .push((pattern.clone(), rows, t.codec, depth));
        };
        // Climb one level only through a named module, never over a layer/stack index.
        if let Some(last) = parent.last() {
            if last != "*" && last != "#" {
                put(parent[..parent.len() - 1].to_vec(), 1);
            }
        }
        put(parent, 0);
    }

    for (bucket, instances) in buckets {
        let cols = bucket.2;
        for ((layer, stack), mut members) in instances {
            members.sort();
            let n = members.len();
            let mut combos: Vec<Vec<usize>> = Vec::new();
            for i in 0..n {
                for j in i + 1..n {
                    combos.push(vec![i, j]);
                    for k in j + 1..n {
                        combos.push(vec![i, j, k]);
                    }
                }
            }
            for combo in combos {
                if combo.iter().all(|&m| members[m].3 == 1) {
                    continue;
                }
                let first = members[combo[0]].2;
                let codec = if combo.iter().all(|&m| members[m].2 == first) {
                    first
                } else {
                    "mixed"
                };
                let rows: u64 = combo.iter().map(|&m| members[m].1).sum();
                let names: Vec<String> = combo.iter().map(|&m| members[m].0.clone()).collect();
                groups
                    .entry((names, true, rows, cols, codec))
                    .or_default()
                    .add(layer, stack.clone());
            }
        }
    }

    // BTreeMap order = (members, fused, rows, cols, codec): deterministic.
    let mut out: Vec<WeightGroup> = groups
        .iter()
        .map(|((members, fused, rows, cols, codec), acc)| {
            acc.finish(*rows, *cols, codec, *fused, members)
        })
        .collect();
    out.sort_by(|a, b| {
        (a.fused, &a.members, a.rows, a.cols, a.codec)
            .cmp(&(b.fused, &b.members, b.rows, b.cols, b.codec))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct T {
        name: String,
        qt: u8,
        shape: Vec<u32>,
    }

    fn t(name: &str, qt: u8, shape: &[u32]) -> T {
        T {
            name: name.to_string(),
            qt,
            shape: shape.to_vec(),
        }
    }

    /// Build a standalone HFQ container with `payload_per_tensor` zero bytes per
    /// tensor, optionally with a tail metadata overlay.
    fn build_hfq(
        metadata: &str,
        tensors: &[T],
        payload_per_tensor: u64,
        overlay: Option<&str>,
    ) -> Vec<u8> {
        let mut index = Vec::new();
        index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        for x in tensors {
            index.extend_from_slice(&(x.name.len() as u16).to_le_bytes());
            index.extend_from_slice(x.name.as_bytes());
            index.push(x.qt);
            index.push(x.shape.len() as u8);
            for d in &x.shape {
                index.extend_from_slice(&d.to_le_bytes());
            }
            index.extend_from_slice(&0u32.to_le_bytes());
            index.extend_from_slice(&payload_per_tensor.to_le_bytes());
        }
        let data_offset = 32 + metadata.len() + index.len();
        let mut out = Vec::new();
        out.extend_from_slice(b"HFQM");
        let version: u32 = if overlay.is_some() { 1 | (1 << 31) } else { 1 };
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&7u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        out.extend_from_slice(&32u64.to_le_bytes());
        out.extend_from_slice(&(data_offset as u64).to_le_bytes());
        out.extend_from_slice(metadata.as_bytes());
        out.extend_from_slice(&index);
        out.resize(data_offset + payload_per_tensor as usize * tensors.len(), 0);
        if let Some(ov) = overlay {
            out.extend_from_slice(ov.as_bytes());
            out.extend_from_slice(&(ov.len() as u64).to_le_bytes());
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(Sha256::digest(ov.as_bytes()).as_slice());
            out.extend_from_slice(b"HFQMDOV1");
        }
        out
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("pm-census-model-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn synthetic() -> Vec<T> {
        let mut v = vec![
            t("model.embed_tokens.weight", 3, &[1000, 64]),
            t("model.norm.weight", 2, &[64]),
        ];
        for l in 0..3u32 {
            v.push(t(
                &format!("model.layers.{l}.self_attn.q_proj.weight"),
                3,
                &[80, 64],
            ));
            v.push(t(
                &format!("model.layers.{l}.self_attn.k_proj.weight"),
                3,
                &[16, 64],
            ));
            v.push(t(
                &format!("model.layers.{l}.self_attn.v_proj.weight"),
                3,
                &[16, 64],
            ));
            v.push(t(
                &format!("model.layers.{l}.self_attn.q_norm.weight"),
                2,
                &[64],
            ));
            v.push(t(
                &format!("model.layers.{l}.mhc.phi_pre.weight"),
                1,
                &[4, 128],
            ));
            v.push(t(
                &format!("model.layers.{l}.mhc.phi_post.weight"),
                2,
                &[4, 128],
            ));
            v.push(t(
                &format!("model.layers.{l}.mhc.phi_z.weight"),
                2,
                &[9, 128],
            ));
            for e in 0..4u32 {
                v.push(t(
                    &format!("model.layers.{l}.mlp.experts.{e}.gate_proj.weight"),
                    44,
                    &[24, 64],
                ));
                v.push(t(
                    &format!("model.layers.{l}.mlp.experts.{e}.up_proj.weight"),
                    44,
                    &[24, 64],
                ));
                v.push(t(
                    &format!("model.layers.{l}.mlp.experts.{e}.down_proj.weight"),
                    44,
                    &[64, 24],
                ));
            }
        }
        v
    }

    fn find<'a>(g: &'a [WeightGroup], member: &str, fused: bool) -> &'a WeightGroup {
        g.iter()
            .find(|w| w.fused == fused && w.members.iter().any(|m| m == member))
            .unwrap_or_else(|| panic!("no group with member {member}"))
    }

    #[test]
    fn codec_names_match_runtime_table() {
        assert_eq!(codec_name(3), "q8_0");
        assert_eq!(codec_name(44), "mq4g256v2");
        assert_eq!(codec_name(1), "f16");
        assert_eq!(codec_name(2), "f32");
        assert_eq!(codec_name(16), "bf16");
        assert_eq!(codec_name(53), "mq4g128v2");
        assert_eq!(codec_name(55), "fp8e4m3b128");
        assert_eq!(codec_name(200), "qt200");
        assert_eq!(codec_name(23), "qt23");
        assert_eq!(codec_name(255), "qt255");
    }

    #[test]
    fn header_only_read_and_config_fallback() {
        let dir = scratch("hdr");
        let meta = r#"{"architecture":"toy","note":"a } \" { brace","config":{"hidden_size":64,"text_config":{"head_dim":16},"rda_config":{"conv_kernel_size":4},"num_experts_per_token":6}}"#;
        let tensors = synthetic();
        let bytes = build_hfq(meta, &tensors, 3, None);
        let path = dir.join("m.hfq");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        let idx = HfqIndex::read(&path).unwrap();
        assert_eq!(idx.arch_id, 7);
        assert_eq!(idx.tensors.len(), tensors.len());
        assert_eq!(idx.tensors[0].name, "model.embed_tokens.weight");
        assert_eq!(idx.tensors[0].codec, "q8_0");
        assert_eq!(idx.tensors[0].shape, vec![1000, 64]);
        assert_eq!(idx.metadata["architecture"], "toy");
        assert_eq!(idx.index_sha256.len(), 64);

        // config absent -> hfq metadata["config"]
        let md = ModelDesc::load(&dir.join("missing.json"), Some(&path)).unwrap();
        assert_eq!(md.config_int(&["hidden_size"]), Some(64));
        assert_eq!(md.config_int(&["rda_config.conv_kernel_size"]), Some(4));
        // text_config fallback + first-present-wins + absent
        assert_eq!(md.config_int(&["nope", "head_dim"]), Some(16));
        assert_eq!(
            md.config_int(&["num_experts_per_tok", "num_experts_per_token", "top_k"]),
            Some(6)
        );
        assert_eq!(md.config_int(&["absent"]), None);

        // explicit config.json wins; text_config prefix at the config root
        let cfg = dir.join("config.json");
        std::fs::write(&cfg, r#"{"text_config":{"hidden_size":128,"x":{"y":3.0}}}"#).unwrap();
        let md2 = ModelDesc::load(&cfg, Some(&path)).unwrap();
        assert_eq!(md2.config_int(&["hidden_size"]), Some(128));
        assert_eq!(md2.config_int(&["x.y"]), Some(3));
        assert_ne!(md2.config_sha256, md.config_sha256);
        assert!(ModelDesc::load(&cfg, None)
            .unwrap()
            .weight_groups()
            .is_empty());
    }

    #[test]
    fn overlay_metadata_replaces_header_metadata() {
        let dir = scratch("ovl");
        let bytes = build_hfq(
            r#"{"config":{"hidden_size":1}}"#,
            &synthetic(),
            2,
            Some(r#"{"config":{"hidden_size":99}}"#),
        );
        let path = dir.join("o.hfq");
        std::fs::write(&path, bytes).unwrap();
        let md = ModelDesc::from_hfq_only(&path).unwrap();
        assert_eq!(md.config_int(&["hidden_size"]), Some(99));
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        let dir = scratch("bad");
        let mut bytes = build_hfq("{}", &synthetic(), 8, None);
        let p = dir.join("t.hfq");
        std::fs::write(&p, &bytes[..bytes.len() - 5]).unwrap();
        assert!(HfqIndex::read(&p).unwrap_err().contains("truncated"));
        bytes[0] = b'X';
        std::fs::write(&p, &bytes).unwrap();
        assert!(HfqIndex::read(&p).unwrap_err().contains("bad magic"));
    }

    #[test]
    fn layer_and_stack_detection() {
        let g = weight_groups_of(&to_infos(&synthetic()));
        let q = find(&g, "model.layers.*.self_attn.q_proj.weight", false);
        assert_eq!((q.rows, q.cols, q.codec), (80, 64, "q8_0"));
        assert_eq!(
            (q.layers, q.count_per_layer, q.stack_len, q.stacked),
            (3, 1, 1, false)
        );
        let gate = find(&g, "model.layers.*.mlp.experts.#.gate_proj.weight", false);
        assert!(gate.stacked && !gate.fused);
        assert_eq!((gate.rows, gate.cols, gate.codec), (24, 64, "mq4g256v2"));
        assert_eq!(
            (gate.layers, gate.count_per_layer, gate.stack_len),
            (3, 1, 4)
        );
        let emb = find(&g, "model.embed_tokens.weight", false);
        assert_eq!(
            (emb.layers, emb.count_per_layer, emb.stacked),
            (1, 1, false)
        );
        // 1-D tensors never form groups
        assert!(g
            .iter()
            .all(|w| w.members.iter().all(|m| !m.contains("norm"))));
    }

    #[test]
    fn sibling_fusion_two_and_three_members() {
        let g = weight_groups_of(&to_infos(&synthetic()));
        // routed gate+up fused = 48 rows, stacked, mq4g256v2
        let gu = g
            .iter()
            .find(|w| {
                w.fused
                    && w.members
                        == vec![
                            "model.layers.*.mlp.experts.#.gate_proj.weight".to_string(),
                            "model.layers.*.mlp.experts.#.up_proj.weight".to_string(),
                        ]
            })
            .expect("gate+up fusion");
        assert_eq!(
            (gu.rows, gu.cols, gu.codec, gu.stacked),
            (48, 64, "mq4g256v2", true)
        );
        assert_eq!((gu.layers, gu.count_per_layer, gu.stack_len), (3, 1, 4));
        // down has different cols: never fused with gate/up
        assert!(g
            .iter()
            .filter(|w| w.fused)
            .all(|w| !w.members.iter().any(|m| m.contains("down_proj"))));
        // q+k+v (80+16+16) three-way and q+k two-way, same cols
        assert!(g
            .iter()
            .any(|w| w.fused && w.members.len() == 3 && w.rows == 112 && w.codec == "q8_0"));
        assert!(g
            .iter()
            .any(|w| w.fused && w.members.len() == 2 && w.rows == 96));
        // mHC phi_pre(f16)+phi_post(f32)+phi_z(f32) = 17 rows, mixed codec
        let mhc = g
            .iter()
            .find(|w| w.fused && w.members.len() == 3 && w.rows == 17)
            .expect("mhc fusion");
        assert_eq!((mhc.cols, mhc.codec, mhc.layers), (128, "mixed", 3));
        // post+z (both f32) = 13 rows keeps the codec
        assert!(g
            .iter()
            .any(|w| w.fused && w.rows == 13 && w.codec == "f32"));
        // no fusion across different parents
        assert!(g.iter().filter(|w| w.fused).all(|w| {
            let parent = |m: &String| m.rsplitn(3, '.').nth(2).unwrap_or("").to_string();
            w.members.iter().all(|m| parent(m) == parent(&w.members[0]))
        }));
    }

    #[test]
    fn irregular_layers_fuse_only_where_all_members_exist() {
        let mut v = synthetic();
        // layer 3 has q but no k/v: q group has 4 layers, q+k exists on 3.
        v.push(t("model.layers.3.self_attn.q_proj.weight", 3, &[80, 64]));
        // layer 4: mhc phi_pre is f32 there (f16 on layers 0..3): the 17-row
        // fusion splits by codec mix, never merges unequal tensors silently.
        v.push(t("model.layers.4.mhc.phi_pre.weight", 2, &[4, 128]));
        v.push(t("model.layers.4.mhc.phi_post.weight", 2, &[4, 128]));
        v.push(t("model.layers.4.mhc.phi_z.weight", 2, &[9, 128]));
        let g = weight_groups_of(&to_infos(&v));
        assert_eq!(
            find(&g, "model.layers.*.self_attn.q_proj.weight", false).layers,
            4
        );
        let qk = g
            .iter()
            .find(|w| w.fused && w.members.len() == 2 && w.rows == 96)
            .expect("q+k");
        assert_eq!(qk.layers, 3);
        let mixed = g
            .iter()
            .find(|w| w.fused && w.rows == 17 && w.codec == "mixed")
            .expect("f16+f32+f32 on layers 0..3");
        assert_eq!(mixed.layers, 3);
        let all32 = g
            .iter()
            .find(|w| w.fused && w.rows == 17 && w.codec == "f32")
            .expect("f32+f32+f32 on layer 4");
        assert_eq!(all32.layers, 1);
    }

    #[test]
    fn siblings_one_level_deeper_fuse_without_duplicates() {
        // Raon-shaped: phi_pre/phi_post under the module, phi_z one level deeper, no `.weight`.
        let mut v = Vec::new();
        for l in 0..2u32 {
            v.push(t(&format!("model.layers.{l}.mix.phi_pre"), 1, &[4, 128]));
            v.push(t(&format!("model.layers.{l}.mix.phi_post"), 2, &[4, 128]));
            v.push(t(&format!("model.layers.{l}.mix.res.phi_z"), 2, &[9, 128]));
            v.push(t(&format!("model.layers.{l}.mix.res.phi_y"), 2, &[5, 128]));
        }
        let g = weight_groups_of(&to_infos(&v));
        let fused: Vec<&WeightGroup> = g.iter().filter(|w| w.fused).collect();
        let triple = fused
            .iter()
            .find(|w| w.members.len() == 3 && w.rows == 17)
            .expect("pre+post+z");
        assert_eq!(
            (triple.codec, triple.layers, triple.cols),
            ("mixed", 2, 128)
        );
        // pre+post (same parent) = 8 rows; pre+z = 13; z+y (both deeper, same parent `mix.res`) = 14
        assert!(fused.iter().any(|w| w.rows == 8 && w.members.len() == 2));
        assert!(fused.iter().any(|w| w.rows == 13 && w.members.len() == 2));
        assert_eq!(
            fused
                .iter()
                .filter(|w| w.rows == 14 && w.members.len() == 2)
                .count(),
            1
        );
        // each combination appears exactly once
        let mut keys: Vec<_> = fused
            .iter()
            .map(|w| (w.members.clone(), w.rows, w.codec))
            .collect();
        let n = keys.len();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), n);
    }

    fn to_infos(v: &[T]) -> Vec<TensorInfo> {
        v.iter()
            .map(|x| TensorInfo {
                name: x.name.clone(),
                quant_type: x.qt,
                codec: codec_name(x.qt),
                shape: x.shape.iter().map(|&d| u64::from(d)).collect(),
            })
            .collect()
    }

    /// Real Raon file (17 GB; header read only). Run with `--ignored`.
    #[test]
    #[ignore]
    fn raon_hfq_directory() {
        let path = Path::new(
            "/home/kaden/qcal/release-0.4.3/raon/models/raon-1-mini-preview.mq4-xts-a8attn-s.hfq",
        );
        let md = ModelDesc::from_hfq_only(path).unwrap();
        let h = md.hfq.as_ref().unwrap();
        eprintln!(
            "tensors {} arch {} idx_sha {}",
            h.tensors.len(),
            h.arch_id,
            h.index_sha256
        );
        assert_eq!(h.tensors.len(), 19509);
        let g = md.weight_groups();
        for w in &g {
            eprintln!(
                "{} {}x{} {} stacked={} fused={} layers={} cpl={} stack={} {:?}",
                w.rows,
                w.rows,
                w.cols,
                w.codec,
                w.stacked,
                w.fused,
                w.layers,
                w.count_per_layer,
                w.stack_len,
                w.members
            );
        }
        let stack = g
            .iter()
            .find(|w| {
                w.stacked && !w.fused && w.rows == 1088 && w.cols == 2560 && w.codec == "mq4g256v2"
            })
            .expect("expert gate/up stack");
        assert_eq!((stack.layers, stack.stack_len), (24, 128));
        assert!(g
            .iter()
            .any(|w| w.stacked && w.fused && w.rows == 2176 && w.cols == 2560));
        // mHC phi_pre[4] + phi_post[4] + phi_z[9] = 17 rows x 10240 (f16/f32 file codecs: "mixed" or f32)
        assert!(g
            .iter()
            .any(|w| w.fused && w.members.len() == 3 && w.rows == 17 && w.cols == 10240));
        assert!(g
            .iter()
            .any(|w| !w.fused && w.codec == "q8_0" && w.rows == 5120 && w.cols == 2560));
        assert_eq!(
            md.config_int(&["num_experts_per_tok", "num_experts_per_token"]),
            Some(6)
        );
    }
}

#[cfg(test)]
mod rocpd_csv_tests {
    /// `scripts/rocpd_to_csv.py` output of the Raon pp4096 db loads through
    /// `rocprof::load` (needs the CSVs written by the script; run with `--ignored`).
    #[test]
    #[ignore]
    fn raon_rocpd_csv_loads() {
        let dir = std::path::Path::new("/home/kaden/qcal/release-0.4.3/pmrip/csv/raon-p1-rp4096");
        let trace = crate::rocprof::load(&dir.join("rp_results_kernel_trace.csv")).unwrap();
        eprintln!(
            "dispatches {} kernels {} agents {:?}",
            trace.dispatches.len(),
            trace.kernels.len(),
            trace
                .agents
                .iter()
                .map(|a| (a.id, a.kind.clone(), a.name.clone()))
                .collect::<Vec<_>>()
        );
        assert_eq!(trace.dispatches.len(), 27045);
        let gpu = trace
            .agent(trace.dispatches[0].agent)
            .expect("dispatch agent id resolves in agent_info");
        assert_eq!(gpu.name, "gfx1201");
    }
}
