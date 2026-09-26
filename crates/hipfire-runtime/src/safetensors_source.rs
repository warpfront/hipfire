//! SafetensorsSource: load HuggingFace safetensors models directly.
//!
//! Supports ParoQuant, AWQ, and unquantized safetensors models.
//! Reads config.json for architecture detection and quantization config.
//! Opens shard headers without mapping payloads; the incumbent borrowed-byte
//! API maps a shard lazily only when explicitly requested.

use crate::model_source::{
    capture_file_identity, read_file_exact_at, verify_path_identity, ModelSource, QuantConfig,
    SourceError, SourceFileIdentity, SourceFormat, SourceIdentity, SourceRangeDescriptor,
    SourceRangeIdentity, SourceReader, SourceReaderImpl, TensorInfo,
};
use half::bf16;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

struct SafetensorsFile {
    file: File,
    mmap: OnceLock<Result<Mmap, String>>,
}

impl SafetensorsFile {
    fn map(&self) -> io::Result<&Mmap> {
        match self
            .mmap
            .get_or_init(|| unsafe { Mmap::map(&self.file).map_err(|error| error.to_string()) })
        {
            Ok(mmap) => Ok(mmap),
            Err(error) => Err(io::Error::new(io::ErrorKind::Other, error.clone())),
        }
    }

    #[cfg(test)]
    fn is_mapped(&self) -> bool {
        self.mmap.get().is_some()
    }
}

const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100 * 1024 * 1024;

struct ParsedSafetensorsTensor {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    data_start: u64,
    relative_start: u64,
    relative_end: u64,
}
struct UniqueHeader(serde_json::Map<String, serde_json::Value>);

impl<'de> serde::Deserialize<'de> for UniqueHeader {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct HeaderVisitor;

        impl<'de> serde::de::Visitor<'de> for HeaderVisitor {
            type Value = UniqueHeader;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a safetensors header object")
            }

            fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut values = serde_json::Map::new();
                while let Some((key, value)) = access.next_entry::<String, serde_json::Value>()? {
                    if values.contains_key(&key) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate safetensors header entry {key}"
                        )));
                    }
                    values.insert(key, value);
                }
                Ok(UniqueHeader(values))
            }
        }

        deserializer.deserialize_map(HeaderVisitor)
    }
}

fn invalid_source(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn dtype_byte_width(dtype: &str) -> Option<u64> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => Some(1),
        "U16" | "I16" | "F16" | "BF16" => Some(2),
        "U32" | "I32" | "F32" => Some(4),
        "U64" | "I64" => Some(8),
        _ => None,
    }
}

fn parse_safetensors_header(
    file: &File,
    file_identity: &SourceFileIdentity,
) -> io::Result<Vec<ParsedSafetensorsTensor>> {
    if file_identity.len < 8 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "safetensors shard is shorter than its header length prefix",
        ));
    }
    let mut length_bytes = [0u8; 8];
    read_file_exact_at(file, file_identity, 0, &mut length_bytes)
        .map_err(|error| invalid_source(error.to_string()))?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(invalid_source(format!(
            "safetensors header is {header_len} bytes, above the {MAX_SAFETENSORS_HEADER_BYTES}-byte limit"
        )));
    }
    let data_start = 8u64
        .checked_add(header_len)
        .ok_or_else(|| invalid_source("safetensors header offset overflow"))?;
    if data_start > file_identity.len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "safetensors header ends at {data_start}, beyond shard length {}",
                file_identity.len
            ),
        ));
    }
    let header_len_usize = usize::try_from(header_len)
        .map_err(|_| invalid_source("safetensors header is too large"))?;
    let mut header_bytes = vec![0u8; header_len_usize];
    read_file_exact_at(file, file_identity, 8, &mut header_bytes)
        .map_err(|error| invalid_source(error.to_string()))?;
    let header: UniqueHeader =
        serde_json::from_slice(&header_bytes).map_err(|error| invalid_source(error.to_string()))?;
    let header = header.0;

    let payload_len = file_identity
        .len
        .checked_sub(data_start)
        .ok_or_else(|| invalid_source("safetensors data start exceeds shard length"))?;
    let mut tensors = Vec::with_capacity(header.len());
    for (name, value) in header {
        if name == "__metadata__" {
            if !value.is_object() {
                return Err(invalid_source("safetensors __metadata__ must be an object"));
            }
            continue;
        }
        let object = value
            .as_object()
            .ok_or_else(|| invalid_source(format!("tensor {name} metadata is not an object")))?;
        let dtype = object
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| invalid_source(format!("tensor {name} has no dtype")))?;
        let element_width = dtype_byte_width(dtype).ok_or_else(|| {
            invalid_source(format!("tensor {name} has unsupported dtype {dtype}"))
        })?;
        let shape_values = object
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| invalid_source(format!("tensor {name} has no shape")))?;
        let mut shape = Vec::with_capacity(shape_values.len());
        let mut element_count = 1u64;
        for dimension in shape_values {
            let dimension = dimension.as_u64().ok_or_else(|| {
                invalid_source(format!("tensor {name} has a non-integer shape dimension"))
            })?;
            if dimension == 0 {
                return Err(invalid_source(format!(
                    "tensor {name} has a zero shape dimension"
                )));
            }
            let dimension_usize = usize::try_from(dimension)
                .map_err(|_| invalid_source(format!("tensor {name} shape is too large")))?;
            element_count = element_count
                .checked_mul(dimension)
                .ok_or_else(|| invalid_source(format!("tensor {name} shape product overflows")))?;
            shape.push(dimension_usize);
        }
        let expected_bytes = element_count
            .checked_mul(element_width)
            .ok_or_else(|| invalid_source(format!("tensor {name} shape byte size overflows")))?;
        let offsets = object
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| invalid_source(format!("tensor {name} has no data_offsets")))?;
        if offsets.len() != 2 {
            return Err(invalid_source(format!(
                "tensor {name} data_offsets must have two entries"
            )));
        }
        let relative_start = offsets[0]
            .as_u64()
            .ok_or_else(|| invalid_source(format!("tensor {name} has an invalid start offset")))?;
        let relative_end = offsets[1]
            .as_u64()
            .ok_or_else(|| invalid_source(format!("tensor {name} has an invalid end offset")))?;
        if relative_end < relative_start {
            return Err(invalid_source(format!(
                "tensor {name} data_offsets are reversed"
            )));
        }
        let byte_len = relative_end - relative_start;
        if byte_len != expected_bytes {
            return Err(invalid_source(format!(
                "tensor {name} shape requires {expected_bytes} bytes but data_offsets span {byte_len}"
            )));
        }
        let absolute_end = data_start
            .checked_add(relative_end)
            .ok_or_else(|| invalid_source(format!("tensor {name} offset overflows")))?;
        if absolute_end > file_identity.len || relative_end > payload_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "tensor {name} data range [{relative_start}, {relative_end}) exceeds shard payload"
                ),
            ));
        }
        tensors.push(ParsedSafetensorsTensor {
            name: name.clone(),
            dtype: dtype.to_string(),
            shape,
            data_start,
            relative_start,
            relative_end,
        });
    }

    let mut ranges: Vec<_> = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.relative_start,
                tensor.relative_end,
                tensor.name.as_str(),
            )
        })
        .collect();
    ranges.sort_unstable_by_key(|(start, _, _)| *start);
    let mut previous_end = 0u64;
    for (start, end, name) in ranges {
        if start < previous_end {
            return Err(invalid_source(format!(
                "tensor {name} data range overlaps another tensor"
            )));
        }
        previous_end = previous_end.max(end);
    }
    Ok(tensors)
}
fn read_safetensors_index(
    dir: &Path,
    st_paths: &[PathBuf],
) -> io::Result<Option<HashMap<String, String>>> {
    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Ok(None);
    }
    let index_bytes = std::fs::read(&index_path)?;
    let index: serde_json::Value =
        serde_json::from_slice(&index_bytes).map_err(|error| invalid_source(error.to_string()))?;
    let weight_map = index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| invalid_source("safetensors index has no weight_map object"))?;
    let shard_names: HashMap<String, ()> = st_paths
        .iter()
        .filter_map(|path| {
            path.file_name()?
                .to_str()
                .map(|name| (name.to_string(), ()))
        })
        .collect();
    let mut result = HashMap::with_capacity(weight_map.len());
    for (name, shard) in weight_map {
        let shard = shard.as_str().ok_or_else(|| {
            invalid_source(format!("safetensors index entry {name} is not a string"))
        })?;
        if !shard_names.contains_key(shard) {
            return Err(invalid_source(format!(
                "safetensors index maps {name} to missing shard {shard}"
            )));
        }
        result.insert(name.clone(), shard.to_string());
    }
    Ok(Some(result))
}

/// Positional reader for one safetensors shard. The reader retains the
/// complete source seal but only one file handle: each descriptor identifies
/// its shard, and every read rechecks that shard's identity before pread.
struct SafetensorsRangeReader {
    identity: Arc<SourceIdentity>,
    file: File,
    file_identity: SourceFileIdentity,
}

impl SourceReaderImpl for SafetensorsRangeReader {
    fn identity(&self) -> &SourceIdentity {
        self.identity.as_ref()
    }

    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
        // The source seal covers the whole shard set. A changed non-selected
        // shard still invalidates descriptors because the source is one
        // immutable inventory, not an unbound collection of files.
        for expected in &self.identity.files {
            verify_path_identity(expected)?;
        }
        read_file_exact_at(&self.file, &self.file_identity, offset, dst)
    }
}

pub struct SafetensorsSource {
    dir: PathBuf,
    files: Vec<SafetensorsFile>,
    tensors: Vec<TensorInfo>,
    tensor_map: HashMap<String, (usize, usize)>, // name -> (file_idx, tensor_idx)
    metadata_json_cached: String,
    arch_id: u32,
    quant_config: Option<QuantConfig>,
    source_identity: Arc<SourceIdentity>,
    range_readers: Vec<SourceReader>,
}

impl SafetensorsSource {
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        // Read config.json.
        let config_path = dir.join("config.json");
        let mut config_str = String::new();
        File::open(&config_path)?.read_to_string(&mut config_str)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;

        let arch_id = derive_arch_id(&config);
        let quant_config = parse_quant_config(&config);
        let metadata_json_cached = build_metadata_json(&config, &config_str);

        // Open every shard and parse only its bounded header. Payload mappings
        // are deferred until an incumbent caller explicitly asks for borrowed
        // bytes through `tensor_data`.
        let mut st_paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .map_or(false, |extension| extension == "safetensors")
            })
            .collect();
        st_paths.sort();
        if st_paths.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{}: no .safetensors files found", dir.display()),
            ));
        }
        let indexed_shards = read_safetensors_index(dir, &st_paths)?;

        let mut files = Vec::with_capacity(st_paths.len());
        let mut file_identities = Vec::with_capacity(st_paths.len());
        let mut tensors = Vec::new();
        let mut tensor_map = HashMap::new();

        for (file_idx, st_path) in st_paths.iter().enumerate() {
            let file_identity = capture_file_identity(st_path)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
            let file = File::open(st_path)?;
            let parsed = parse_safetensors_header(&file, &file_identity)?;
            let shard_name = st_path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| invalid_source("safetensors shard has no UTF-8 file name"))?;

            for tensor in parsed {
                if tensor_map.contains_key(&tensor.name) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        SourceError::DuplicateTensor { name: tensor.name },
                    ));
                }
                if let Some(index) = &indexed_shards {
                    match index.get(&tensor.name) {
                        Some(indexed_shard) if indexed_shard == shard_name => {}
                        Some(indexed_shard) => {
                            return Err(invalid_source(format!(
                                "safetensors index maps {} to {indexed_shard}, but it is in {shard_name}",
                                tensor.name
                            )));
                        }
                        None => {
                            return Err(invalid_source(format!(
                                "safetensors index has no entry for {}",
                                tensor.name
                            )));
                        }
                    }
                }
                let data_offset_u64 = tensor
                    .data_start
                    .checked_add(tensor.relative_start)
                    .ok_or_else(|| invalid_source("safetensors tensor offset overflows"))?;
                let data_size_u64 = tensor
                    .relative_end
                    .checked_sub(tensor.relative_start)
                    .ok_or_else(|| invalid_source("safetensors tensor length underflows"))?;
                let data_offset = usize::try_from(data_offset_u64)
                    .map_err(|_| invalid_source("safetensors tensor offset is too large"))?;
                let data_size = usize::try_from(data_size_u64)
                    .map_err(|_| invalid_source("safetensors tensor is too large"))?;
                let tensor_idx = tensors.len();
                tensors.push(TensorInfo {
                    name: tensor.name.clone(),
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    quant_type: 0xFF,
                    data_offset,
                    data_size,
                });
                tensor_map.insert(tensor.name, (file_idx, tensor_idx));
            }

            file_identities.push(file_identity);
            files.push(SafetensorsFile {
                file,
                mmap: OnceLock::new(),
            });
        }
        if let Some(index) = &indexed_shards {
            if let Some(name) = index.keys().find(|name| !tensor_map.contains_key(*name)) {
                return Err(invalid_source(format!(
                    "safetensors index names missing tensor {name}"
                )));
            }
        }

        let source_identity = Arc::new(SourceIdentity {
            canonical_path: std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf()),
            format: SourceFormat::Safetensors,
            files: file_identities.clone(),
            metadata_json: metadata_json_cached.clone(),
            manifest: tensors
                .iter()
                .map(|info| {
                    let &(file_index, _) = tensor_map
                        .get(&info.name)
                        .expect("tensor map populated with every tensor");
                    SourceRangeIdentity {
                        name: info.name.clone(),
                        file_index,
                        offset: info.data_offset as u64,
                        length: info.data_size as u64,
                        dtype: info.dtype.clone(),
                        logical_shape: info.shape.clone(),
                    }
                })
                .collect(),
        });
        let range_readers = files
            .iter()
            .enumerate()
            .map(|(file_idx, file)| {
                let reader_file = file.file.try_clone()?;
                Ok(SourceReader::from_inner(SafetensorsRangeReader {
                    identity: source_identity.clone(),
                    file: reader_file,
                    file_identity: file_identities[file_idx].clone(),
                }))
            })
            .collect::<io::Result<Vec<_>>>()?;
        tracing::debug!(
            model_dir = %dir.display(),
            shard_count = files.len(),
            tensor_count = tensors.len(),
            arch_id,
            quantized = quant_config.is_some(),
            "opened safetensors model source"
        );

        Ok(Self {
            dir: dir.to_path_buf(),
            files,
            tensors,
            tensor_map,
            metadata_json_cached,
            arch_id,
            quant_config,
            source_identity,
            range_readers,
        })
    }

    /// Immutable source seal shared by all descriptors from this shard set.
    pub fn source_identity(&self) -> &SourceIdentity {
        self.source_identity.as_ref()
    }
    /// Public accessor so `loader_api` doesn't need the `ModelSource` trait in scope.
    pub fn arch_id(&self) -> u32 {
        self.arch_id
    }

    #[cfg(test)]
    fn mapped_file_count(&self) -> usize {
        self.files.iter().filter(|file| file.is_mapped()).count()
    }
}

impl ModelSource for SafetensorsSource {
    fn metadata_json(&self) -> &str {
        &self.metadata_json_cached
    }

    fn arch_id(&self) -> u32 {
        self.arch_id
    }

    fn quant_config(&self) -> Option<&QuantConfig> {
        self.quant_config.as_ref()
    }

    fn tensor_data(&self, name: &str) -> Option<(&TensorInfo, &[u8])> {
        let &(file_idx, tensor_idx) = self.tensor_map.get(name)?;
        let info = &self.tensors[tensor_idx];
        let mmap = self.files[file_idx].map().ok()?;
        let end = info.data_offset.checked_add(info.data_size)?;
        Some((info, &mmap[info.data_offset..end]))
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        let &(_file_idx, tensor_idx) = self.tensor_map.get(name)?;
        Some(&self.tensors[tensor_idx])
    }
    fn tensor_range(&self, name: &str) -> Result<Option<SourceRangeDescriptor>, SourceError> {
        let Some(&(file_idx, tensor_idx)) = self.tensor_map.get(name) else {
            return Ok(None);
        };
        let info = &self.tensors[tensor_idx];
        SourceRangeDescriptor::from_parts(
            self.source_identity.clone(),
            u64::try_from(info.data_offset).map_err(|_| SourceError::Overflow {
                offset: u64::MAX,
                length: info.data_size as u64,
            })?,
            u64::try_from(info.data_size).map_err(|_| SourceError::Overflow {
                offset: info.data_offset as u64,
                length: u64::MAX,
            })?,
            info.dtype.clone(),
            info.shape.clone(),
            self.range_readers[file_idx].clone(),
        )
        .map(Some)
    }

    /// `MADV_DONTNEED` over the tensor's mmap range.
    ///
    /// Reading a 24 GB checkpoint through an mmap makes every page it touches
    /// resident in THIS process, so a streaming loader that never allocates a
    /// host table still watches RSS climb to the size of the file. Dropping
    /// the page-table entries once a tensor has been handed to the GPU keeps
    /// the resident set flat at roughly one tensor.
    ///
    /// Safe on a read-only `MAP_SHARED` file mapping: the pages are clean and
    /// the kernel refaults them from the page cache (or the file) on the next
    /// read, so this loses performance at worst, never data. Best-effort — a
    /// `madvise` failure (unsupported filesystem, huge pages) is ignored, and
    /// the range is byte-exact rather than page-exact because `advise` rounds
    /// the start down and may spill into a neighbouring tensor's first page,
    /// which likewise only costs a refault.
    fn release_tensor_pages(&self, name: &str) {
        let Some(&(file_idx, tensor_idx)) = self.tensor_map.get(name) else {
            return;
        };
        let info = &self.tensors[tensor_idx];
        if info.data_size == 0 {
            return;
        }
        #[cfg(unix)]
        {
            let Some(Ok(mmap)) = self.files[file_idx].mmap.get() else {
                return;
            };
            // SAFETY: read-only file mapping, so every page is clean; a
            // discarded page is refaulted from the file with identical bytes.
            let _ = unsafe {
                mmap.unchecked_advise_range(
                    memmap2::UncheckedAdvice::DontNeed,
                    info.data_offset,
                    info.data_size,
                )
            };
        }
        #[cfg(not(unix))]
        let _ = file_idx;
    }

    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    fn path(&self) -> &Path {
        &self.dir
    }

    fn tokenizer_json_path(&self) -> Option<PathBuf> {
        let p = self.dir.join("tokenizer.json");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    fn chat_template(&self) -> Option<String> {
        // Newer HF convention (transformers 5.x): a standalone
        // `chat_template.jinja` file (e.g. North-Mini-Code / cohere2_moe).
        let jinja = self.dir.join("chat_template.jinja");
        if let Ok(mut f) = File::open(&jinja) {
            let mut s = String::new();
            if f.read_to_string(&mut s).is_ok() && !s.trim().is_empty() {
                return Some(s);
            }
        }
        // Older convention: a `chat_template` field in tokenizer_config.json.
        let p = self.dir.join("tokenizer_config.json");
        let mut s = String::new();
        File::open(p).ok()?.read_to_string(&mut s).ok()?;
        let v: serde_json::Value = serde_json::from_str(&s).ok()?;
        v.get("chat_template")?.as_str().map(|s| s.to_string())
    }
}

pub fn derive_arch_id(config: &serde_json::Value) -> u32 {
    let mut archs = config
        .get("architectures")
        .and_then(|a| a.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();

    // Diffusers component configs (e.g. `transformer/config.json`) carry no
    // `architectures` array — they name the class via `_class_name` instead.
    // When `architectures` is absent or empty, fall back to treating
    // `[_class_name]` as the architectures list so the same table-driven
    // substring match below resolves FLUX.1 vs FLUX.2 Klein.
    //
    // **Deliberately narrowed to the FLUX transformer classes.** `_class_name`
    // is a diffusers-wide key: an unrestricted fallback promotes it above
    // `model_type` for EVERY config in the workspace that happens to carry
    // both, which is a global routing change made for one feature's benefit.
    // `Flux*Transformer2DModel` is the whole set this feature needs
    // (`FluxTransformer2DModel` -> 40, `Flux2Transformer2DModel` -> 44), it
    // cannot collide with a text model's class name, and anything else keeps
    // the pre-existing `model_type` route.
    if archs.is_empty() {
        if let Some(class_name) = config
            .get("_class_name")
            .and_then(|v| v.as_str())
            .filter(|c| c.starts_with("Flux") && c.ends_with("Transformer2DModel"))
        {
            archs.push(class_name);
        }
    }

    // Check text_config for MoE indicators
    let text_config = config.get("text_config").unwrap_or(config);
    let has_experts = text_config
        .get("num_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
        > 0;

    // Architectures field takes priority over model_type (HF convention).
    // This loop is table-driven: any known model_type substring inside the
    // architecture string is resolved via the canonical table, longest key
    // wins so gemma4_unified_assistant (22) beats gemma4 (13) and
    // muse_glimmer_assistant (23) beats muse_glimmer (14). The table alone
    // cannot express the priority ordering nor the qwen3.5/3.6 dense-vs-MoE
    // has_experts decision, so those remain explicit.
    for arch in &archs {
        let arch_lower = arch.to_lowercase();
        // qwen3.5/3.6 family: dense (5) vs MoE (6) decided by has_experts.
        // This is the only per-arch substring check that must remain: the
        // table maps the four dense strings to 5, but a MoE checkpoint uses
        // the same strings with num_experts>0 to mean 6.
        if arch_lower.contains("qwen3_5")
            || arch_lower.contains("qwen3.5")
            || arch_lower.contains("qwen3_6")
            || arch_lower.contains("qwen3.6")
        {
            return if has_experts { 6 } else { 5 };
        }
        // Generic table-driven substring match for all other architectures.
        let mut best: Option<(&'static str, u32)> = None;
        for (k, v) in crate::arch_mapping::MODEL_TYPE_TO_ARCH_ID
            .iter()
            .chain(crate::arch_mapping::RESERVED_MODEL_TYPE_TO_ARCH_ID)
        {
            if arch_lower.contains(*k) {
                match best {
                    Some((bk, _)) if k.len() <= bk.len() => {}
                    _ => best = Some((*k, *v)),
                }
            }
        }
        if let Some((_, id)) = best {
            return id;
        }
    }

    // Fallback: check model_type via the canonical table (single source of truth).
    let model_type = config
        .get("model_type")
        .or_else(|| text_config.get("model_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if let Some(mut id) = crate::arch_mapping::lookup_model_type(model_type) {
        // qwen3.5/3.6 dense entries are 5 in the table; has_experts flips to 6.
        if id == 5 && has_experts {
            id = 6;
        }
        return id;
    }

    // C1: unrecognized model_type → an explicit unclaimed sentinel that NO
    // carrier matches, so `load_model` fails cleanly with "no carrier for
    // <dir>" instead of silently mis-routing to Qwen35 (arch_id=5) and dying
    // deep in weight loading with a confusing error.
    {
        let supported = crate::arch_mapping::supported_model_types_display();
        eprintln!(
            "warning: unrecognized model_type '{model_type}'; no carrier claims it              (supported model_types: {supported})"
        );
        UNCLAIMED_ARCH_ID
    }
}

/// Sentinel `arch_id` emitted by [`derive_arch_id`] for an unrecognized
/// `model_type`. No carrier's `claims_arch_id` matches it, so routing fails
/// loudly with a clean "no carrier" error rather than silently defaulting to
/// Qwen35. Far outside the assigned range (0..=64) the registry tests sweep.
pub const UNCLAIMED_ARCH_ID: u32 = u32::MAX;

fn parse_quant_config(config: &serde_json::Value) -> Option<QuantConfig> {
    let qc = config.get("quantization_config")?;
    let method = qc.get("quant_method")?.as_str()?.to_string();
    let bits = qc.get("bits").and_then(|v| v.as_u64()).unwrap_or(4) as u8;
    let group_size = qc.get("group_size").and_then(|v| v.as_u64()).unwrap_or(128) as u32;
    let krot = qc.get("krot").and_then(|v| v.as_u64()).unwrap_or(0) as u8;

    let dynamic_excludes = qc
        .get("dynamic")
        .and_then(|d| d.as_object())
        .map(|obj| {
            obj.keys()
                .filter(|k| k.starts_with("-:"))
                .map(|k| k.strip_prefix("-:").unwrap_or(k).to_string())
                .collect()
        })
        .unwrap_or_default();

    Some(QuantConfig {
        method,
        bits,
        group_size,
        krot,
        dynamic_excludes,
    })
}

fn build_metadata_json(config: &serde_json::Value, raw_config: &str) -> String {
    // Build HFQ-compatible metadata: { "architecture": "...", "config": {...} }
    // The Qwen35 config parser expects metadata_json to contain a "config" key.
    let mut meta = serde_json::Map::new();

    // Determine architecture string
    let text_config = config.get("text_config").unwrap_or(config);
    let model_type = text_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    meta.insert(
        "architecture".to_string(),
        serde_json::Value::String(model_type.to_string()),
    );

    // Embed the full config.json as the "config" key
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(raw_config) {
        meta.insert("config".to_string(), parsed);
    }

    serde_json::to_string(&serde_json::Value::Object(meta)).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// BF16 (bfloat16) decode helpers
// ---------------------------------------------------------------------------

/// Widen a BF16 (bfloat16) value to F32.
/// BF16 is the upper 16 bits of an IEEE-754 F32 number — widening is left-shifting by 16.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

/// Convert BF16 byte slice to F16 byte vector (owned).
/// Each BF16 value is widened to F32, then narrowed to F16.
pub fn bf16_bytes_to_f16(data: &[u8]) -> Vec<u8> {
    data.chunks_exact(2)
        .map(|c| {
            let bf16 = u16::from_le_bytes([c[0], c[1]]);
            let f32_val = f32::from_bits((bf16 as u32) << 16);
            crate::llama::f32_to_f16(f32_val).to_le_bytes()
        })
        .flatten()
        .collect()
}

/// Convert BF16 byte slice to F32 vector.
pub fn bf16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| {
            let bf16 = u16::from_le_bytes([c[0], c[1]]);
            f32::from_bits((bf16 as u32) << 16)
        })
        .collect()
}

/// Convert tensor bytes to F16 bytes based on dtype string.
/// Handles F16 (passthrough), BF16 (decode), F32 (narrow).
/// Panics on unknown dtype (fail-fast over silent wrong results).
/// NOTE: n_elements validation removed — caller responsibility.
pub fn source_bytes_to_f16_stream(source_dtype: &str, data: &[u8]) -> Vec<u8> {
    match source_dtype {
        "F16" => data.to_vec(),
        "BF16" => bf16_bytes_to_f16(data),
        "F32" => data
            .chunks_exact(4)
            .map(|c| {
                let f32_val = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                crate::llama::f32_to_f16(f32_val).to_le_bytes()
            })
            .flatten()
            .collect(),
        other => panic!(
            "unsupported source dtype '{other}' for fp-to-f16 conversion (expected F16/BF16/F32)"
        ),
    }
}

/// Convert tensor bytes to F32 vector based on dtype string.
/// Panics on unknown dtype.
pub fn source_bytes_to_f32_vec(source_dtype: &str, data: &[u8]) -> Vec<f32> {
    match source_dtype {
        "F16" => data
            .chunks_exact(2)
            .map(|c| crate::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "BF16" => bf16_bytes_to_f32(data),
        "F32" => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!(
            "unsupported source dtype '{other}' for fp-to-f32 conversion (expected F16/BF16/F32)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn known_model_types_route_as_expected() {
        assert_eq!(derive_arch_id(&json!({ "model_type": "llama" })), 0);
        assert_eq!(derive_arch_id(&json!({ "model_type": "mistral" })), 0);
        // qwen2 → 7 (Qwen2Carrier, loads attn biases); qwen3 → 1 (LlamaCarrier).
        assert_eq!(derive_arch_id(&json!({ "model_type": "qwen2" })), 7);
        assert_eq!(derive_arch_id(&json!({ "model_type": "qwen3" })), 1);
        assert_eq!(
            derive_arch_id(&json!({ "architectures": ["Qwen2ForCausalLM"] })),
            7
        );
        assert_eq!(
            derive_arch_id(&json!({ "architectures": ["Qwen3ForCausalLM"] })),
            1
        );
        assert_eq!(derive_arch_id(&json!({ "model_type": "qwen3.5" })), 5);
        assert_eq!(
            derive_arch_id(&json!({ "model_type": "qwen3.5", "num_experts": 8 })),
            6
        );
        assert_eq!(derive_arch_id(&json!({ "model_type": "minimax_m2" })), 10);
        // Per-expert / VLM Dir arches routed to dedicated carriers (mirrors the
        // quantizer ingest map). Lock the strings so a rename can't silently
        // unclaim a real checkpoint.
        assert_eq!(derive_arch_id(&json!({ "model_type": "dots_ocr" })), 8);
        assert_eq!(derive_arch_id(&json!({ "model_type": "deepseek_v4" })), 9);
        assert_eq!(derive_arch_id(&json!({ "model_type": "lfm2_moe" })), 11);
        assert_eq!(derive_arch_id(&json!({ "model_type": "lfm2" })), 11);
        assert_eq!(derive_arch_id(&json!({ "model_type": "cohere2_moe" })), 12);
        assert_eq!(derive_arch_id(&json!({ "model_type": "qwen4_exp" })), 16);
        assert_eq!(
            derive_arch_id(&json!({
                "model_type": "qwen4_exp",
                "text_config": { "model_type": "qwen4_exp_text" }
            })),
            16
        );
    }

    /// A diffusers FLUX transformer component config (`model_type: "flux"`)
    /// must route to arch 40 so the FluxDiffusionCarrier can claim it.
    #[test]
    fn flux_transformer_routes_to_arch_40() {
        assert_eq!(derive_arch_id(&json!({ "model_type": "flux" })), 40);
        // Diffusers layout has no `architectures`; the `_class_name` /
        // `model_index` style fields must not interfere with the model_type
        // fallback lookup.
        let cfg = serde_json::json!({
            "_class_name": "FluxTransformer2DModel",
            "model_type": "flux",
            "num_layers": 1,
            "num_single_layers": 1,
        });
        assert_eq!(derive_arch_id(&cfg), 40);
    }

    /// A diffusers FLUX.2 Klein transformer component config
    /// (`_class_name: "Flux2Transformer2DModel"` or `model_type: "flux2"`)
    /// must route to arch 45, and a FLUX.1 config with both `_class_name`
    /// and `model_type` present must still resolve to 40 (longest-key wins).
    #[test]
    fn flux2_transformer_routes_to_arch_45() {
        assert_eq!(derive_arch_id(&json!({ "model_type": "flux2" })), 45);
        assert_eq!(
            derive_arch_id(&json!({ "_class_name": "Flux2Transformer2DModel", "num_layers": 5 })),
            45
        );
        assert_eq!(
            derive_arch_id(
                &json!({ "_class_name": "FluxTransformer2DModel", "model_type": "flux" })
            ),
            40
        );
    }

    /// The `_class_name` fallback is scoped to `Flux*Transformer2DModel` and
    /// must NOT outrank `model_type` for anything else.
    ///
    /// `_class_name` is a diffusers-wide key, and the table match below is a
    /// SUBSTRING match, so an unrestricted fallback silently re-routes every
    /// config in the workspace that carries both fields. Both cases here were
    /// mis-routed by the unrestricted form: a Qwen3 text encoder's class name
    /// contains `qwen3` (arch 1) and would beat `model_type: qwen2` (arch 7);
    /// a FLUX.2 VAE's class name contains `flux2` (arch 45) and would beat
    /// `model_type: flux` (arch 40).
    #[test]
    fn non_flux_class_name_does_not_outrank_model_type() {
        assert_eq!(
            derive_arch_id(&json!({ "_class_name": "Qwen3ForCausalLM", "model_type": "qwen2" })),
            7,
            "a text encoder's _class_name must not beat its model_type"
        );
        assert_eq!(
            derive_arch_id(&json!({ "_class_name": "AutoencoderKLFlux2", "model_type": "flux" })),
            40,
            "a VAE component's _class_name must not beat its model_type"
        );
        // And the fallback still does nothing at all when there is no
        // `model_type` to fall through to: unclaimed, not a guess.
        assert_eq!(
            derive_arch_id(&json!({ "_class_name": "AutoencoderKLFlux2" })),
            UNCLAIMED_ARCH_ID
        );
    }

    /// C1: an unrecognized model_type must NOT silently become Qwen35 (arch_id=5).
    /// It returns the unclaimed sentinel so routing fails with a clean "no carrier".
    #[test]
    fn unrecognized_model_type_is_unclaimed_not_qwen35() {
        let id = derive_arch_id(&json!({ "model_type": "totally_unknown_arch" }));
        assert_eq!(id, UNCLAIMED_ARCH_ID);
        assert_ne!(id, 5, "must not default to Qwen35");
    }

    #[test]
    fn bf16_to_f32_basic_values() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0f32); // normal
        assert_eq!(bf16_to_f32(0xC000), -2.0f32); // normal negative
        assert_eq!(bf16_to_f32(0x0000), 0.0f32); // zero
        assert_eq!(bf16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits()); // neg zero
        assert!(bf16_to_f32(0x0001) > 0.0); // subnormal
        assert!(bf16_to_f32(0x7FC0).is_nan()); // NaN (quiet)
        assert!(bf16_to_f32(0x7F81).is_nan()); // NaN (signaling)
        assert!(bf16_to_f32(0xFFC0).is_nan()); // negative NaN
    }

    #[test]
    fn source_bytes_roundtrip() {
        // F16 passthrough: F16 1.0 → stays F16 1.0
        let f16_data = vec![0x00u8, 0x3C];
        let result = source_bytes_to_f16_stream("F16", &f16_data);
        assert_eq!(result, f16_data);

        // BF16→F16: BF16 1.0 (0x3F80 LE) → F16 1.0 (0x3C00 LE)
        let bf16_data = vec![0x80u8, 0x3F];
        let result = source_bytes_to_f16_stream("BF16", &bf16_data);
        assert_eq!(result, vec![0x00, 0x3C]);

        // F32→F16: F32 1.0 → F16 1.0
        let f32_data = vec![0x00u8, 0x00, 0x80, 0x3F];
        let result = source_bytes_to_f16_stream("F32", &f32_data);
        assert_eq!(result, vec![0x00, 0x3C]);

        // BF16→F32: BF16 -2.0 (0xC000 LE) → -2.0 F32
        let bf16_data = vec![0x00u8, 0xC0];
        let result = source_bytes_to_f32_vec("BF16", &bf16_data);
        assert_eq!(result, vec![-2.0f32]);
    }

    #[test]
    #[should_panic(expected = "unsupported source dtype")]
    fn source_bytes_to_f16_unknown_dtype_panics() {
        source_bytes_to_f16_stream("FP8", &[0u8; 4]);
    }

    /// Minimal hand-rolled safetensors writer: 8-byte LE header length, JSON
    /// header, concatenated little-endian tensor bytes. Keeps the test free of
    /// a writer dependency.
    fn write_shard(path: &Path, tensors: &[(&str, Vec<u8>, Vec<usize>)]) {
        use std::io::Write as _;
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, data, shape) in tensors {
            let mut meta = serde_json::Map::new();
            meta.insert("dtype".into(), "BF16".into());
            meta.insert(
                "shape".into(),
                serde_json::Value::Array(shape.iter().map(|&s| s.into()).collect()),
            );
            meta.insert(
                "data_offsets".into(),
                serde_json::json!([offset, offset + data.len()]),
            );
            offset += data.len();
            header.insert((*name).to_string(), meta.into());
        }
        let header_json = serde_json::Value::Object(header).to_string();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header_json.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header_json.as_bytes()).unwrap();
        for (_, data, _) in tensors {
            file.write_all(data).unwrap();
        }
    }

    fn write_source(dir: &Path, tensors: &[(&str, Vec<u8>, Vec<usize>)]) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({ "model_type": "llama" }).to_string(),
        )
        .unwrap();
        write_shard(&dir.join("model.safetensors"), tensors);
    }

    fn write_raw_shard(path: &Path, header: serde_json::Value, payload: &[u8]) {
        use std::io::Write as _;
        let header = header.to_string();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(payload).unwrap();
    }

    #[test]
    fn header_only_open_keeps_multi_shard_payloads_unmapped() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-st-header-only-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({ "model_type": "llama" }).to_string(),
        )
        .unwrap();
        let a = vec![1u8, 2];
        let b = vec![3u8, 4];
        write_shard(
            &dir.join("model-00001.safetensors"),
            &[("a", a.clone(), vec![1])],
        );
        write_shard(
            &dir.join("model-00002.safetensors"),
            &[("b", b.clone(), vec![1])],
        );
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::json!({
                "weight_map": {
                    "a": "model-00001.safetensors",
                    "b": "model-00002.safetensors"
                }
            })
            .to_string(),
        )
        .unwrap();

        let source = SafetensorsSource::open(&dir).expect("open multi-shard source");
        assert_eq!(source.mapped_file_count(), 0);
        let range = source.tensor_range("a").unwrap().expect("range a");
        let mut ranged = vec![0u8; a.len()];
        range.read_exact(&mut ranged).unwrap();
        assert_eq!(ranged, a);
        assert_eq!(source.mapped_file_count(), 0);

        assert_eq!(source.tensor_data("a").unwrap().1, &a[..]);
        assert_eq!(source.mapped_file_count(), 1);
        assert_eq!(source.tensor_data("b").unwrap().1, &b[..]);
        assert_eq!(source.mapped_file_count(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn duplicate_tensor_names_across_shards_are_rejected() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-st-duplicate-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({ "model_type": "llama" }).to_string(),
        )
        .unwrap();
        write_shard(
            &dir.join("model-00001.safetensors"),
            &[("same", vec![0, 0], vec![1])],
        );
        write_shard(
            &dir.join("model-00002.safetensors"),
            &[("same", vec![1, 1], vec![1])],
        );
        let error = SafetensorsSource::open(&dir)
            .err()
            .expect("duplicate must fail");
        assert!(error.to_string().contains("duplicate source tensor: same"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_offsets_and_shapes_are_rejected_before_mapping() {
        let cases = [
            (
                "shape",
                serde_json::json!({
                    "bad": { "dtype": "BF16", "shape": [2], "data_offsets": [0, 2] }
                }),
                vec![0u8, 0, 0, 0],
                "shape requires 4 bytes",
            ),
            (
                "offset",
                serde_json::json!({
                    "bad": { "dtype": "BF16", "shape": [1], "data_offsets": [0, 2] }
                }),
                vec![0u8],
                "exceeds shard payload",
            ),
        ];
        for (label, header, payload, expected) in cases {
            let dir = std::env::temp_dir().join(format!(
                "hipfire-st-malformed-{label}-{}-{}",
                std::process::id(),
                line!()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("config.json"),
                serde_json::json!({ "model_type": "llama" }).to_string(),
            )
            .unwrap();
            write_raw_shard(&dir.join("model.safetensors"), header, &payload);
            let error = SafetensorsSource::open(&dir)
                .err()
                .expect("malformed shard must fail");
            assert!(
                error.to_string().contains(expected),
                "{label} error: {error}"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn shard_index_membership_and_changed_identity_are_checked() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-st-identity-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({ "model_type": "llama" }).to_string(),
        )
        .unwrap();
        write_shard(
            &dir.join("model-00001.safetensors"),
            &[("a", vec![0, 0], vec![1])],
        );
        write_shard(
            &dir.join("model-00002.safetensors"),
            &[("b", vec![1, 1], vec![1])],
        );
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::json!({
                "weight_map": {
                    "a": "model-00002.safetensors",
                    "b": "model-00002.safetensors"
                }
            })
            .to_string(),
        )
        .unwrap();
        let error = SafetensorsSource::open(&dir)
            .err()
            .expect("index membership must fail");
        assert!(error
            .to_string()
            .contains("maps a to model-00002.safetensors"));

        std::fs::remove_file(dir.join("model.safetensors.index.json")).unwrap();
        let source = SafetensorsSource::open(&dir).expect("open source");
        let range = source.tensor_range("a").unwrap().expect("range a");
        use std::io::Write as _;
        std::fs::OpenOptions::new()
            .append(true)
            .open(dir.join("model-00002.safetensors"))
            .unwrap()
            .write_all(&[9])
            .unwrap();
        let mut bytes = vec![0u8; 2];
        let error = range.read_exact(&mut bytes).unwrap_err();
        assert!(matches!(error, SourceError::IdentityChanged { .. }));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn oversized_header_is_rejected_without_allocating_it() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-st-header-limit-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({ "model_type": "llama" }).to_string(),
        )
        .unwrap();
        use std::io::Write as _;
        let mut file = std::fs::File::create(dir.join("model.safetensors")).unwrap();
        file.write_all(&(MAX_SAFETENSORS_HEADER_BYTES + 1).to_le_bytes())
            .unwrap();
        let error = SafetensorsSource::open(&dir)
            .err()
            .expect("oversized header must fail");
        assert!(error.to_string().contains("above the"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `release_tensor_pages` is the crate's only `unsafe` madvise, and its
    /// trait contract is explicit: the hint is advisory, and a later
    /// `tensor_data` for the same name must still return the same bytes.
    ///
    /// That contract is what makes it safe to call while streaming a 24 GB
    /// checkpoint — the pages are clean file-backed pages, so `MADV_DONTNEED`
    /// only drops this process's page-table entries and the kernel refaults
    /// them from the file. If it ever silently zero-filled instead (which is
    /// what `MADV_DONTNEED` does to a PRIVATE ANONYMOUS mapping), every weight
    /// released before it was read would become zero and the failure would
    /// look like a model bug, not a memory bug. This test is the guard.
    #[test]
    fn releasing_tensor_pages_does_not_change_the_bytes() {
        // Several pages long, and deliberately NOT page-aligned in length, so
        // `a` and `b` share a page boundary: `advise` rounds the start down and
        // may spill into the neighbour, which must likewise only cost a
        // refault.
        let a_bytes: Vec<u8> = (0..40_000u32).map(|i| (i % 251) as u8 + 1).collect();
        let b_bytes: Vec<u8> = (0..8_000u32).map(|i| (i % 241) as u8 + 3).collect();
        let dir = std::env::temp_dir().join(format!(
            "hipfire-st-release-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        write_source(
            &dir,
            &[
                ("a", a_bytes.clone(), vec![a_bytes.len() / 2]),
                ("b", b_bytes.clone(), vec![b_bytes.len() / 2]),
            ],
        );

        let src = SafetensorsSource::open(&dir).expect("open source");
        let before = src.tensor_data("a").expect("tensor a").1.to_vec();
        assert_eq!(before, a_bytes, "fixture did not round-trip");

        src.release_tensor_pages("a");

        let after = src.tensor_data("a").expect("tensor a after release").1;
        assert_eq!(after, &a_bytes[..], "bytes changed after MADV_DONTNEED");
        // The neighbour sharing `a`'s trailing page must be intact too.
        let b_after = src.tensor_data("b").expect("tensor b after release").1;
        assert_eq!(b_after, &b_bytes[..], "neighbour tensor lost bytes");

        // Idempotent, and a name the source does not have is a no-op, not a
        // panic — callers treat the whole thing as best-effort.
        src.release_tensor_pages("a");
        src.release_tensor_pages("no-such-tensor");
        assert_eq!(
            src.tensor_data("a").unwrap().1,
            &a_bytes[..],
            "second release changed the bytes"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
