//! ModelSource: unified interface for loading model weights from different
//! container formats (HFQ, safetensors, etc.).
//!
//! The Architecture trait's `load_weights` and `config_from_source` consume
//! `&dyn ModelSource` so the same loading code works for hipfire's native HFQ
//! format and HuggingFace safetensors (ParoQuant, AWQ, etc.).

use std::fmt;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Metadata about a single tensor in a model file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Safetensors dtype string: "F16", "F32", "I32", "I16", "BF16", etc.
    pub dtype: String,
    pub shape: Vec<usize>,
    /// For HFQ: the quant_type byte. For safetensors: 0xFF (use dtype instead).
    pub quant_type: u8,
    /// Byte offset into the backing store.
    pub data_offset: usize,
    /// Byte size of the tensor data.
    pub data_size: usize,
}

/// The on-disk source format bound into a [`SourceIdentity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFormat {
    /// The HFQM container used by hipfire's native HFQ artifacts.
    Hfq,
    /// A HuggingFace safetensors shard set.
    Safetensors,
}

/// File identity captured when a source is opened.
///
/// The tuple is deliberately metadata-only: it is cheap to check before every
/// bounded read and catches replacement, truncation, extension, and timestamp
/// changes without hashing a multi-GB payload. The parsed source manifest is
/// stored separately in [`SourceIdentity`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceFileIdentity {
    pub canonical_path: PathBuf,
    pub dev: u64,
    pub ino: u64,
    pub len: u64,
    pub mtime_secs: i64,
    pub mtime_nanos: u32,
}

/// One source range recorded in an immutable source seal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRangeIdentity {
    pub name: String,
    /// Index into [`SourceIdentity::files`].
    pub file_index: usize,
    /// Absolute offset in the selected source file.
    pub offset: u64,
    pub length: u64,
    pub dtype: String,
    pub logical_shape: Vec<usize>,
}

/// Immutable source identity shared by all descriptors obtained from a source.
///
/// `files` binds every file that can satisfy a descriptor (including both the
/// base and overlay of an effective HFQ source). `metadata_json` and
/// `manifest` bind the parsed container/index, so a descriptor cannot silently
/// drift to another tensor with the same byte range. Keep this behind an
/// [`Arc`] when handing it to a loader or an external-row worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceIdentity {
    pub canonical_path: PathBuf,
    pub format: SourceFormat,
    pub files: Vec<SourceFileIdentity>,
    pub metadata_json: String,
    pub manifest: Vec<SourceRangeIdentity>,
}

/// Errors raised by checked source descriptors and bounded source reads.
#[derive(Debug)]
pub enum SourceError {
    /// The named tensor is absent from the source.
    NotFound { name: String },
    /// The descriptor's `offset + length` or a read's `offset + dst.len()`
    /// overflowed `u64`.
    Overflow { offset: u64, length: u64 },
    /// A checked range extends beyond its source file.
    RangeOutOfBounds {
        offset: u64,
        length: u64,
        source_len: u64,
    },
    /// A caller supplied a buffer whose size does not match an exact read.
    BufferLengthMismatch { expected: u64, actual: usize },
    /// The underlying positional read returned fewer bytes than requested.
    ShortRead {
        offset: u64,
        expected: usize,
        actual: usize,
    },
    /// The source file changed after the descriptor was sealed.
    IdentityChanged {
        expected: SourceFileIdentity,
        actual: SourceFileIdentity,
    },
    /// A source/index contains two entries for one logical tensor name.
    DuplicateTensor { name: String },
    /// An attached HFQ overlay cannot participate in one effective source seal.
    MissingEffectiveSeal { reason: String },
    /// The source/index is structurally invalid.
    InvalidSource { reason: String },
    /// The operating system rejected a positional read.
    Io { offset: u64, source: io::Error },
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound { name } => write!(f, "source tensor not found: {name}"),
            Self::Overflow { offset, length } => {
                write!(f, "source range overflows: offset={offset}, length={length}")
            }
            Self::RangeOutOfBounds {
                offset,
                length,
                source_len,
            } => write!(
                f,
                "source range [{offset}, {}) exceeds file length {source_len}",
                offset.saturating_add(*length)
            ),
            Self::BufferLengthMismatch { expected, actual } => {
                write!(f, "exact source read needs {expected} bytes, buffer has {actual}")
            }
            Self::ShortRead {
                offset,
                expected,
                actual,
            } => write!(
                f,
                "short source read at offset {offset}: expected {expected} bytes, got {actual}"
            ),
            Self::IdentityChanged { expected, actual } => write!(
                f,
                "source identity changed for {:?}: expected dev/ino={}/{} len={} mtime={}.{}, got dev/ino={}/{} len={} mtime={}.{}",
                expected.canonical_path,
                expected.dev,
                expected.ino,
                expected.len,
                expected.mtime_secs,
                expected.mtime_nanos,
                actual.dev,
                actual.ino,
                actual.len,
                actual.mtime_secs,
                actual.mtime_nanos,
            ),
            Self::DuplicateTensor { name } => write!(f, "duplicate source tensor: {name}"),
            Self::MissingEffectiveSeal { reason } => {
                write!(f, "HFQ overlay has no effective source seal: {reason}")
            }
            Self::InvalidSource { reason } => write!(f, "invalid source: {reason}"),
            Self::Io { offset, source } => write!(f, "source read at offset {offset}: {source}"),
        }
    }
}

impl std::error::Error for SourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Internal source-reader implementation. The public [`SourceReader`] wrapper
/// is intentionally constructible only by this crate's format readers, so
/// callers cannot forge a descriptor reader with a different source identity.
pub(crate) trait SourceReaderImpl: Send + Sync {
    fn identity(&self) -> &SourceIdentity;
    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), SourceError>;
}

/// A sealed, source-owned positional reader.
///
/// This is a handle rather than an externally implementable callback. HFQ and
/// safetensors create it from their own opened file handles and immutable
/// identity snapshots; every read rechecks those snapshots before issuing the
/// bounded positional I/O.
#[derive(Clone)]
pub struct SourceReader {
    inner: Arc<dyn SourceReaderImpl>,
}

impl fmt::Debug for SourceReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SourceReader")
            .field("identity", self.inner.identity())
            .finish()
    }
}

impl SourceReader {
    pub(crate) fn from_inner(inner: impl SourceReaderImpl + 'static) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Read exactly `dst.len()` bytes at an absolute source-file offset.
    pub fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
        self.inner.read_exact_at(offset, dst)
    }

    /// Immutable source seal checked by this reader.
    pub fn identity(&self) -> &SourceIdentity {
        self.inner.identity()
    }
}

/// A checked, format-neutral tensor byte range.
///
/// Offsets are absolute within the selected source file. The descriptor
/// retains its source-owned reader, enforces the descriptor bounds before every
/// read, and then delegates to the reader's identity-checked positional I/O.
#[derive(Clone, Debug)]
pub struct SourceRangeDescriptor {
    source_identity: Arc<SourceIdentity>,
    pub offset: u64,
    pub length: u64,
    pub dtype: String,
    pub logical_shape: Vec<usize>,
    reader: SourceReader,
}

impl SourceRangeDescriptor {
    pub(crate) fn from_parts(
        source_identity: Arc<SourceIdentity>,
        offset: u64,
        length: u64,
        dtype: String,
        logical_shape: Vec<usize>,
        reader: SourceReader,
    ) -> Result<Self, SourceError> {
        offset
            .checked_add(length)
            .ok_or(SourceError::Overflow { offset, length })?;
        if reader.identity() != source_identity.as_ref() {
            return Err(SourceError::InvalidSource {
                reason: "descriptor reader identity does not match descriptor seal".to_string(),
            });
        }
        Ok(Self {
            source_identity,
            offset,
            length,
            dtype,
            logical_shape,
            reader,
        })
    }

    pub fn source_identity(&self) -> &SourceIdentity {
        self.source_identity.as_ref()
    }

    pub fn reader(&self) -> &SourceReader {
        &self.reader
    }

    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    /// Read a sub-range at an absolute source-file offset. Reads outside this
    /// descriptor are rejected even when they are valid in the backing file.
    pub fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
        let desc_end = self
            .offset
            .checked_add(self.length)
            .ok_or(SourceError::Overflow {
                offset: self.offset,
                length: self.length,
            })?;
        let read_len = u64::try_from(dst.len()).map_err(|_| SourceError::Overflow {
            offset,
            length: u64::MAX,
        })?;
        let read_end = offset.checked_add(read_len).ok_or(SourceError::Overflow {
            offset,
            length: read_len,
        })?;
        if offset < self.offset || read_end > desc_end {
            return Err(SourceError::RangeOutOfBounds {
                offset,
                length: read_len,
                source_len: desc_end,
            });
        }
        self.reader.read_exact_at(offset, dst)
    }

    /// Read this descriptor's complete payload into a caller-owned buffer.
    pub fn read_exact(&self, dst: &mut [u8]) -> Result<(), SourceError> {
        if u64::try_from(dst.len()).ok() != Some(self.length) {
            return Err(SourceError::BufferLengthMismatch {
                expected: self.length,
                actual: dst.len(),
            });
        }
        self.read_exact_at(self.offset, dst)
    }
}

/// A source payload that preserves the old borrowed path for small tensors and
/// exposes a checked range for large tensors. `Owned` is used when an existing
/// source had to copy its small payload out of a reusable buffer.
#[derive(Debug, Clone)]
pub enum SourcePayload<'a> {
    Borrowed {
        info: &'a TensorInfo,
        bytes: &'a [u8],
    },
    Owned {
        info: TensorInfo,
        bytes: Vec<u8>,
    },
    Range(SourceRangeDescriptor),
}

impl SourcePayload<'_> {
    pub fn dtype(&self) -> &str {
        match self {
            Self::Borrowed { info, .. } => &info.dtype,
            Self::Owned { info, .. } => &info.dtype,
            Self::Range(range) => range.dtype(),
        }
    }

    pub fn logical_shape(&self) -> &[usize] {
        match self {
            Self::Borrowed { info, .. } => &info.shape,
            Self::Owned { info, .. } => &info.shape,
            Self::Range(range) => range.logical_shape(),
        }
    }
}

/// Read one exact file range after checking the expected file identity and
/// length. This helper is shared by the HFQ and safetensors source-owned
/// readers; it never allocates a whole tensor or shard.
pub(crate) fn read_file_exact_at(
    file: &File,
    expected: &SourceFileIdentity,
    offset: u64,
    dst: &mut [u8],
) -> Result<(), SourceError> {
    verify_path_identity(expected)?;
    let length = u64::try_from(dst.len()).map_err(|_| SourceError::Overflow {
        offset,
        length: u64::MAX,
    })?;
    let end = offset
        .checked_add(length)
        .ok_or(SourceError::Overflow { offset, length })?;
    if end > expected.len {
        return Err(SourceError::RangeOutOfBounds {
            offset,
            length,
            source_len: expected.len,
        });
    }
    if dst.is_empty() {
        return Ok(());
    }

    let mut done = 0usize;
    while done < dst.len() {
        let absolute = offset
            .checked_add(u64::try_from(done).expect("usize always fits u64 on supported targets"))
            .ok_or(SourceError::Overflow { offset, length })?;
        #[cfg(unix)]
        let n = {
            use std::os::unix::fs::FileExt;
            file.read_at(&mut dst[done..], absolute)
        };
        #[cfg(target_os = "windows")]
        let n = {
            use std::os::windows::fs::FileExt;
            file.seek_read(&mut dst[done..], absolute)
        };
        #[cfg(not(any(unix, target_os = "windows")))]
        let n: io::Result<usize> = Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "positional reads are unsupported on this target",
        ));
        let n = n.map_err(|source| SourceError::Io {
            offset: absolute,
            source,
        })?;
        if n == 0 {
            return Err(SourceError::ShortRead {
                offset,
                expected: dst.len(),
                actual: done,
            });
        }
        done += n;
    }
    Ok(())
}

/// Capture the cheap, immutable file identity used by source readers.
pub(crate) fn capture_file_identity(path: &Path) -> Result<SourceFileIdentity, SourceError> {
    let canonical_path =
        std::fs::canonicalize(path).map_err(|source| SourceError::Io { offset: 0, source })?;
    let metadata = std::fs::metadata(&canonical_path)
        .map_err(|source| SourceError::Io { offset: 0, source })?;
    Ok(identity_from_metadata(canonical_path, &metadata))
}

/// One file identity from an already-obtained [`std::fs::Metadata`].
fn identity_from_metadata(
    canonical_path: PathBuf,
    metadata: &std::fs::Metadata,
) -> SourceFileIdentity {
    #[cfg(unix)]
    let (dev, ino) = {
        use std::os::unix::fs::MetadataExt;
        (metadata.dev(), metadata.ino())
    };
    #[cfg(not(unix))]
    let (dev, ino) = (0, 0);
    let (mtime_secs, mtime_nanos) = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| (duration.as_secs() as i64, duration.subsec_nanos()))
        .unwrap_or((0, 0));
    SourceFileIdentity {
        canonical_path,
        dev,
        ino,
        len: metadata.len(),
        mtime_secs,
        mtime_nanos,
    }
}

/// True when `metadata` still describes exactly the sealed [`SourceFileIdentity`].
fn identity_matches(expected: &SourceFileIdentity, metadata: &std::fs::Metadata) -> bool {
    #[cfg(unix)]
    let (dev, ino) = {
        use std::os::unix::fs::MetadataExt;
        (metadata.dev(), metadata.ino())
    };
    #[cfg(not(unix))]
    let (dev, ino) = (0, 0);
    let (mtime_secs, mtime_nanos) = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| (duration.as_secs() as i64, duration.subsec_nanos()))
        .unwrap_or((0, 0));
    dev == expected.dev
        && ino == expected.ino
        && metadata.len() == expected.len
        && mtime_secs == expected.mtime_secs
        && mtime_nanos == expected.mtime_nanos
}

/// Re-check a sealed identity on the read path in one `stat`, with no path
/// resolution and no allocation.
///
/// The sealed identity already holds the canonical path, so re-resolving it per
/// read buys nothing: identity is compared by (dev, ino, length, mtime), and a
/// retargeted symlink or replaced ancestor directory shows up as a different
/// (dev, ino) on the same name. This is deliberately a `stat` by name rather
/// than an `fstat` on the open handle: the handle keeps pointing at the inode it
/// was opened from, so `fstat` alone cannot see a file replaced at the sealed
/// name — the case `descriptor_rejects_source_identity_change` and
/// `compact_qwen4_ple_range_rejects_identity_change_and_truncation` pin.
pub(crate) fn verify_path_identity(expected: &SourceFileIdentity) -> Result<(), SourceError> {
    let metadata = std::fs::metadata(&expected.canonical_path)
        .map_err(|source| SourceError::Io { offset: 0, source })?;
    if identity_matches(expected, &metadata) {
        return Ok(());
    }
    Err(SourceError::IdentityChanged {
        expected: expected.clone(),
        actual: identity_from_metadata(expected.canonical_path.clone(), &metadata),
    })
}

/// Quantization config parsed from HFQ metadata or HF config.json.
#[derive(Debug, Clone, Default)]
pub struct QuantConfig {
    pub method: String,  // "paroquant", "awq", "gptq", "" (HFQ native)
    pub bits: u8,        // 4
    pub group_size: u32, // 128
    pub krot: u8,        // 8 for ParoQuant, 0 otherwise
    /// Regex patterns for layers excluded from quantization (kept FP16).
    pub dynamic_excludes: Vec<String>,
}

/// Unified interface for reading model data from HFQ files or safetensors
/// directories. Both `HfqFile` and `SafetensorsSource` implement this.
pub trait ModelSource {
    /// JSON metadata blob. For HFQ: the embedded metadata. For safetensors:
    /// the contents of config.json formatted as HFQ-compatible metadata.
    fn metadata_json(&self) -> &str;

    /// Architecture ID for dispatch.
    /// 0 = LLaMA/Mistral, 1 = Qwen3/Qwen2, 5 = Qwen3.5 dense, 6 = MoE.
    fn arch_id(&self) -> u32;

    /// Quantization config (if detected from metadata).
    fn quant_config(&self) -> Option<&QuantConfig>;

    /// Look up a tensor by name. Returns metadata + byte slice.
    /// Returns None if the tensor doesn't exist or the mmap was dropped.
    fn tensor_data(&self, name: &str) -> Option<(&TensorInfo, &[u8])>;

    /// Look up tensor metadata without data (for pre-screening).
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo>;

    /// Open a bounded, source-owned range for a large tensor. The default keeps
    /// existing third-party ModelSource implementations source-compatible.
    fn tensor_range(&self, _name: &str) -> Result<Option<SourceRangeDescriptor>, SourceError> {
        Ok(None)
    }

    /// Return the old borrowed payload when available, otherwise a checked
    /// range payload. This default preserves existing small-tensor behavior.
    fn tensor_payload<'a>(&'a self, name: &str) -> Result<Option<SourcePayload<'a>>, SourceError> {
        if let Some((info, bytes)) = self.tensor_data(name) {
            return Ok(Some(SourcePayload::Borrowed { info, bytes }));
        }
        Ok(self.tensor_range(name)?.map(SourcePayload::Range))
    }

    /// All tensor names in the source.
    fn tensor_names(&self) -> Vec<&str>;

    /// Path to the model directory or file (for weight pager, logging).
    fn path(&self) -> &std::path::Path;

    /// Path to tokenizer.json (if available in the model directory).
    /// HFQ embeds the tokenizer in metadata; safetensors models ship it
    /// as a separate file.
    fn tokenizer_json_path(&self) -> Option<std::path::PathBuf> {
        None
    }

    /// Chat template string (Jinja) if available.
    fn chat_template(&self) -> Option<String> {
        None
    }

    /// Hint that one tensor's bytes will not be read again, so the source may
    /// drop whatever it is holding for them.
    ///
    /// Advisory and best-effort by contract: a caller that streams a 24 GB
    /// checkpoint tensor-by-tensor uses it to keep resident set size flat
    /// (see [`SafetensorsSource`](crate::safetensors_source::SafetensorsSource),
    /// which issues `MADV_DONTNEED` over the tensor's mmap range). A source
    /// with nothing to release — or one whose release fails — is not an
    /// error, and a later `tensor_data` for the same name must still return
    /// the same bytes.
    fn release_tensor_pages(&self, _name: &str) {}
}

/// Open a model from a path, auto-detecting the format.
/// - If path is a directory with config.json: opens as SafetensorsSource
/// - If path ends in .hfq: opens as HfqFile
/// - Otherwise: tries HfqFile first, then directory
pub fn open_model(path: &std::path::Path) -> Result<Box<dyn ModelSource>, String> {
    if path.is_dir() {
        let config_path = path.join("config.json");
        if config_path.exists() {
            let source = crate::safetensors_source::SafetensorsSource::open(path)
                .map_err(|e| format!("safetensors open failed: {e}"))?;
            Ok(Box::new(source))
        } else {
            Err(format!("{}: directory has no config.json", path.display()))
        }
    } else {
        let hfq = crate::hfq::HfqFile::open(path).map_err(|e| format!("{e}"))?;
        Ok(Box::new(hfq))
    }
}
#[cfg(test)]
mod range_tests {
    use super::*;
    use std::io::Write as _;

    struct TestReader {
        identity: SourceIdentity,
        file: File,
        short: bool,
    }

    impl SourceReaderImpl for TestReader {
        fn identity(&self) -> &SourceIdentity {
            &self.identity
        }

        fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
            if self.short {
                return Err(SourceError::ShortRead {
                    offset,
                    expected: dst.len(),
                    actual: dst.len().saturating_sub(1),
                });
            }
            read_file_exact_at(&self.file, &self.identity.files[0], offset, dst)
        }
    }

    fn test_descriptor(
        path: &Path,
        offset: u64,
        length: u64,
        short: bool,
    ) -> SourceRangeDescriptor {
        let file = File::open(path).unwrap();
        let file_identity = capture_file_identity(path).unwrap();
        let identity = Arc::new(SourceIdentity {
            canonical_path: file_identity.canonical_path.clone(),
            format: SourceFormat::Hfq,
            files: vec![file_identity],
            metadata_json: "{}".to_string(),
            manifest: Vec::new(),
        });
        let reader = SourceReader::from_inner(TestReader {
            identity: identity.as_ref().clone(),
            file,
            short,
        });
        SourceRangeDescriptor::from_parts(
            identity,
            offset,
            length,
            "BF16".to_string(),
            vec![length as usize / 2],
            reader,
        )
        .unwrap()
    }

    #[test]
    fn descriptor_reads_first_and_last_ranges() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payload.bin");
        let mut file = File::create(&path).unwrap();
        file.write_all(b"0123456789").unwrap();

        let first = test_descriptor(&path, 0, 2, false);
        let mut first_bytes = [0u8; 2];
        first.read_exact(&mut first_bytes).unwrap();
        assert_eq!(&first_bytes, b"01");

        let last = test_descriptor(&path, 8, 2, false);
        let mut last_bytes = [0u8; 2];
        last.read_exact(&mut last_bytes).unwrap();
        assert_eq!(&last_bytes, b"89");
    }

    #[test]
    fn descriptor_propagates_short_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payload.bin");
        std::fs::write(&path, b"0123456789").unwrap();
        let descriptor = test_descriptor(&path, 0, 4, true);
        let mut bytes = [0u8; 4];
        assert!(matches!(
            descriptor.read_exact(&mut bytes),
            Err(SourceError::ShortRead { .. })
        ));
    }

    #[test]
    fn descriptor_rejects_source_identity_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payload.bin");
        std::fs::write(&path, b"0123456789").unwrap();
        let descriptor = test_descriptor(&path, 0, 4, false);
        let replacement = dir.path().join("replacement.bin");
        std::fs::write(&replacement, b"abcdefghij").unwrap();
        std::fs::rename(replacement, &path).unwrap();
        let mut bytes = [0u8; 4];
        assert!(matches!(
            descriptor.read_exact(&mut bytes),
            Err(SourceError::IdentityChanged { .. })
        ));
    }

    #[test]
    fn descriptor_rejects_offset_overflow() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payload.bin");
        std::fs::write(&path, b"0123456789").unwrap();
        let file = File::open(&path).unwrap();
        let file_identity = capture_file_identity(&path).unwrap();
        let identity = Arc::new(SourceIdentity {
            canonical_path: file_identity.canonical_path.clone(),
            format: SourceFormat::Hfq,
            files: vec![file_identity.clone()],
            metadata_json: "{}".to_string(),
            manifest: Vec::new(),
        });
        let reader = SourceReader::from_inner(TestReader {
            identity: identity.as_ref().clone(),
            file,
            short: false,
        });
        assert!(matches!(
            SourceRangeDescriptor::from_parts(
                identity,
                u64::MAX,
                1,
                "BF16".to_string(),
                vec![1],
                reader,
            ),
            Err(SourceError::Overflow { .. })
        ));
    }

    #[test]
    fn cheap_identity_check_still_refuses_in_place_growth() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payload.bin");
        std::fs::write(&path, b"0123456789").unwrap();
        let descriptor = test_descriptor(&path, 0, 4, false);
        let sealed = capture_file_identity(&path).unwrap();
        verify_path_identity(&sealed).expect("sealed identity verifies");
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[7])
            .unwrap();
        let mut bytes = [0u8; 4];
        assert!(matches!(
            descriptor.read_exact(&mut bytes),
            Err(SourceError::IdentityChanged { .. })
        ));
    }

    /// Cost of the read-path identity check. CPU + one `stat`; no GPU.
    ///
    ///   HIPFIRE_PROBE_MODEL=<artifact> cargo test -p hipfire-runtime --lib \
    ///     identity_check_cost_probe -- --ignored --nocapture
    ///
    /// Prints ns/op for the open-time capture (`canonicalize` + `stat` +
    /// allocation) and for the read-path verifier (one `stat`, no allocation).
    #[test]
    #[ignore = "timing probe; set HIPFIRE_PROBE_MODEL to a sealed file"]
    fn identity_check_cost_probe() {
        let Ok(model) = std::env::var("HIPFIRE_PROBE_MODEL") else {
            println!("identity-check-probe: skipped, HIPFIRE_PROBE_MODEL is unset");
            return;
        };
        let path = Path::new(&model);
        let sealed = capture_file_identity(path).expect("capture probe identity");
        const ITERATIONS: u32 = 20_000;

        let started = std::time::Instant::now();
        for _ in 0..ITERATIONS {
            let identity = capture_file_identity(path).expect("capture");
            std::hint::black_box(identity.len);
        }
        let capture_ns = started.elapsed().as_secs_f64() * 1e9 / ITERATIONS as f64;

        let started = std::time::Instant::now();
        for _ in 0..ITERATIONS {
            verify_path_identity(&sealed).expect("verify");
        }
        let verify_ns = started.elapsed().as_secs_f64() * 1e9 / ITERATIONS as f64;

        println!(
            "identity-check-probe {model}: capture={capture_ns:.0}ns/op verify={verify_ns:.0}ns/op \
             saving={:.0}ns/op ({:.2}x), one fewer syscall and two fewer allocations per read",
            capture_ns - verify_ns,
            capture_ns / verify_ns.max(0.001)
        );
    }
}
