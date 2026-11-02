// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared-weight host protocol v1: canonical identity, typed manifest, source
//! certificates and the Linux `SCM_RIGHTS` handshake.
//!
//! This module is generic and architecture independent. It has no GPU, HIP,
//! rdna-compute or model-family dependency; architectures describe their final
//! weights through [`WeightManifest`] (allocations plus typed views) and the
//! opaque, sorted [`CanonicalFields`] payloads.
//!
//! # Identities (design section 3)
//!
//! * [`model_content_sha256`] — ordered full-content digests of every source
//!   that supplies tensor bytes (base file first, then overlays).
//! * [`LoaderAbi::sha256`] — domain-versioned digest of the loader/representation
//!   contract. Compute-only differences (kernel images, commits, provenance) are
//!   deliberately **not** part of it.
//! * [`compute_layout_seals`] — `SHA256("hipfire-weight-layout/v1\0" ||
//!   canonical(model, abi, recipe, manifest))` plus the recipe seal.
//!
//! Canonical encoding: fixed enum tags, little-endian fixed-width integers,
//! `u32` length prefixes, UTF-8 strings, sorted maps (decoders reject unsorted or
//! duplicate keys), exact floating-point bit patterns and explicit absent/present
//! tags. Nothing depends on pointers, inodes, paths, `usize` or hash iteration
//! order.
//!
//! # Source certificates
//!
//! A [`SourceCertificate`] binds a full SHA-256 to a stable file identity
//! (dev/ino/len/mtime/ctime/uid/mode). The host hashes once through
//! [`certify_source`] while holding the file open and requiring identical
//! before/after identities; tenants call [`SourceCertificate::verify_source_path`]
//! which only `fstat`s the source (no content scan). Writable sources and
//! writable/stale/foreign-owned certificates are rejected. This is a
//! same-UID development-fabric rule, not a security boundary.
//!
//! # Wire protocol v1
//!
//! Frame: 20-byte header `magic "HFWH" | version u32 | kind u32 | payload_len u32
//! | fd_count u32`, then `payload_len` bytes. File descriptors travel only as
//! `SCM_RIGHTS` ancillary data on the header bytes; every received descriptor is
//! wrapped in `OwnedFd` immediately (never leaked on error) and received with
//! `MSG_CMSG_CLOEXEC`. At most [`MAX_FDS_PER_MESSAGE`] descriptors per frame.
//!
//! `Hello -> Ready -> FdBatch* -> ImportAck -> (live lease) -> Release`.
//! The handshake (client: connect through last FD; host: accept through last FD)
//! runs under one absolute deadline. The lease is *never* acknowledged by
//! `connect`; the importer calls [`WeightHostLease::ack_import`] only after the
//! allocations were really imported, and [`WeightHostLease::release`] only after
//! teardown. `Drop` closes resources and never sends either message. Neither
//! `SO_PEERCRED` nor socket EOF proves GPU teardown; they only prove which
//! process was on the other end and that it closed the socket.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Wire protocol version implemented by this module.
pub const PROTOCOL_VERSION: u32 = 1;
/// Maximum descriptors in one protocol message.
pub const MAX_FDS_PER_MESSAGE: usize = 32;
/// Maximum frame payload (Ready carries the whole manifest).
pub const MAX_FRAME_PAYLOAD: usize = 64 * 1024 * 1024;
/// Maximum number of allocations (one FD each) in a manifest.
pub const MAX_ALLOCATIONS: usize = 1 << 16;
/// Maximum number of views in a manifest.
pub const MAX_VIEWS: usize = 1 << 20;
/// Maximum admission token size.
pub const MAX_ADMISSION_TOKEN_BYTES: usize = 4096;
/// Maximum UTF-8 string length in canonical encodings.
pub const MAX_STRING_BYTES: usize = 4096;
/// Maximum `Bytes` blob inside a canonical encoding.
pub const MAX_BLOB_BYTES: usize = 32 * 1024 * 1024;
/// Mandatory zeroed tail every exported weight allocation carries beyond its
/// logical bytes (frozen shared-weight allocation rule): `physical >= logical +
/// EXPORT_WEIGHT_TAIL_PAD_BYTES`. Strict for protocol v1.
pub const EXPORT_WEIGHT_TAIL_PAD_BYTES: u64 = 64 * 1024;
/// Maximum element count of any list/map in a canonical encoding.
pub const MAX_LIST_ITEMS: usize = 1 << 22;
/// Maximum nesting depth of [`FieldValue`] lists/maps.
pub const MAX_FIELD_NESTING: usize = 8;
/// Maximum tensor rank in a [`WeightView`].
pub const MAX_VIEW_RANK: usize = 16;
/// Default handshake budget.
pub const DEFAULT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

const MAGIC: [u8; 4] = *b"HFWH";
const HEADER_LEN: usize = 20;
const MAX_REJECT_REASON_BYTES: usize = 512;
const MAX_CERT_FILE_BYTES: u64 = 64 * 1024;
const CERT_MAGIC: &[u8; 8] = b"HFWSCERT";
const CERT_VERSION: u32 = 1;
const DOMAIN_LAYOUT: &str = "hipfire-weight-layout/v1\0";
const DOMAIN_RECIPE: &str = "hipfire-weight-layout-recipe/v1\0";
const DOMAIN_ABI: &str = "hipfire-loader-abi/v1\0";
const DOMAIN_MODEL: &str = "hipfire-model-content/v1\0";
const DOMAIN_CERT: &str = "hipfire-source-cert/v1\0";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Every failure of this module. All protocol failures are fail-closed.
#[derive(Debug)]
pub enum WeightHostError {
    Io {
        context: &'static str,
        source: io::Error,
    },
    /// The single handshake deadline (or a caller-supplied wait) expired.
    Timeout(&'static str),
    /// Peer closed the socket. EOF is not proof of GPU/process teardown.
    PeerClosed(&'static str),
    UnsupportedVersion {
        got: u32,
        supported: u32,
    },
    Protocol(String),
    LimitExceeded(&'static str),
    PeerCredentials(String),
    InsecureSocket(String),
    /// The host sent a Reject frame (admission/identity/...).
    Rejected(String),
    /// Names the identity field that differed.
    IdentityMismatch(&'static str),
    ManifestInvalid(String),
    FdCount {
        expected: usize,
        got: usize,
    },
    FdInvalid(String),
    AdmissionDenied(String),
    ImportedBytesMismatch {
        expected: u64,
        got: u64,
    },
    InvalidState(&'static str),
    SourceMutated(String),
    SourceNotImmutable(String),
    CertificateInvalid(String),
    InvalidArgument(String),
}

impl fmt::Display for WeightHostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use WeightHostError::*;
        match self {
            Io { context, source } => write!(f, "weight host I/O error ({context}): {source}"),
            Timeout(w) => write!(f, "weight host timeout: {w}"),
            PeerClosed(w) => write!(f, "weight host peer closed: {w}"),
            UnsupportedVersion { got, supported } => {
                write!(
                    f,
                    "unsupported weight host protocol version {got} (supported {supported})"
                )
            }
            Protocol(s) => write!(f, "weight host protocol violation: {s}"),
            LimitExceeded(w) => write!(f, "weight host limit exceeded: {w}"),
            PeerCredentials(s) => write!(f, "weight host peer credentials rejected: {s}"),
            InsecureSocket(s) => write!(f, "weight host socket rejected: {s}"),
            Rejected(s) => write!(f, "weight host rejected the connection: {s}"),
            IdentityMismatch(w) => write!(f, "weight host identity mismatch: {w}"),
            ManifestInvalid(s) => write!(f, "weight manifest invalid: {s}"),
            FdCount { expected, got } => {
                write!(
                    f,
                    "weight host fd count mismatch: expected {expected}, got {got}"
                )
            }
            FdInvalid(s) => write!(f, "weight host fd invalid: {s}"),
            AdmissionDenied(s) => write!(f, "weight host admission denied: {s}"),
            ImportedBytesMismatch { expected, got } => {
                write!(f, "imported bytes mismatch: expected {expected}, got {got}")
            }
            InvalidState(s) => write!(f, "weight host invalid state: {s}"),
            SourceMutated(s) => write!(f, "weight source mutated or stale: {s}"),
            SourceNotImmutable(s) => write!(f, "weight source not immutable: {s}"),
            CertificateInvalid(s) => write!(f, "source certificate invalid: {s}"),
            InvalidArgument(s) => write!(f, "invalid argument: {s}"),
        }
    }
}

impl std::error::Error for WeightHostError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WeightHostError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn proto<S: Into<String>>(s: S) -> WeightHostError {
    WeightHostError::Protocol(s.into())
}

fn io_ctx(context: &'static str) -> impl FnOnce(io::Error) -> WeightHostError {
    move |source| WeightHostError::Io { context, source }
}

fn map_io(e: io::Error, context: &'static str) -> WeightHostError {
    match e.kind() {
        io::ErrorKind::ConnectionReset
        | io::ErrorKind::BrokenPipe
        | io::ErrorKind::ConnectionAborted
        | io::ErrorKind::UnexpectedEof => WeightHostError::PeerClosed(context),
        io::ErrorKind::InvalidData => WeightHostError::Protocol(e.to_string()),
        _ => WeightHostError::Io { context, source: e },
    }
}

// ---------------------------------------------------------------------------
// Digest primitives
// ---------------------------------------------------------------------------

/// A SHA-256 digest.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sha256Digest(pub [u8; 32]);

impl Sha256Digest {
    pub fn of(bytes: &[u8]) -> Self {
        let mut h = Sha256::new();
        h.update(bytes);
        Self::from_hasher(h)
    }

    fn from_hasher(h: Sha256) -> Self {
        let out = h.finalize();
        let mut a = [0u8; 32];
        a.copy_from_slice(&out);
        Self(a)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(64);
        for b in self.0 {
            s.push(HEX[(b >> 4) as usize] as char);
            s.push(HEX[(b & 15) as usize] as char);
        }
        s
    }

    pub fn from_hex(s: &str) -> Result<Self, WeightHostError> {
        let b = s.as_bytes();
        if b.len() != 64 {
            return Err(WeightHostError::InvalidArgument(
                "sha256 hex must be 64 characters".into(),
            ));
        }
        fn nib(c: u8) -> Option<u8> {
            match c {
                b'0'..=b'9' => Some(c - b'0'),
                b'a'..=b'f' => Some(c - b'a' + 10),
                b'A'..=b'F' => Some(c - b'A' + 10),
                _ => None,
            }
        }
        let mut out = [0u8; 32];
        for i in 0..32 {
            let (Some(hi), Some(lo)) = (nib(b[2 * i]), nib(b[2 * i + 1])) else {
                return Err(WeightHostError::InvalidArgument(
                    "sha256 hex has a non-hex digit".into(),
                ));
            };
            out[i] = (hi << 4) | lo;
        }
        Ok(Self(out))
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sha256({})", self.to_hex())
    }
}

fn domain_hash(domain: &str, body: &[u8]) -> Sha256Digest {
    let mut h = Sha256::new();
    h.update(domain.as_bytes());
    h.update(body);
    Sha256Digest::from_hasher(h)
}

/// Stream a reader through SHA-256, returning the digest and byte count.
pub fn sha256_reader<R: Read>(mut r: R) -> io::Result<(Sha256Digest, u64)> {
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        match r.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                h.update(&buf[..n]);
                total += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok((Sha256Digest::from_hasher(h), total))
}

/// 128-bit host generation epoch.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Epoch(pub [u8; 16]);

impl Epoch {
    /// Random epoch from the kernel CSPRNG.
    pub fn random() -> Result<Self, WeightHostError> {
        let mut b = [0u8; 16];
        let mut off = 0;
        while off < b.len() {
            // SAFETY: valid writable buffer region.
            let n = unsafe {
                libc::getrandom(b[off..].as_mut_ptr() as *mut libc::c_void, b.len() - off, 0)
            };
            if n < 0 {
                let e = io::Error::last_os_error();
                if e.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(WeightHostError::Io {
                    context: "getrandom",
                    source: e,
                });
            }
            off += n as usize;
        }
        Ok(Self(b))
    }

    pub fn is_zero(&self) -> bool {
        self.0 == [0u8; 16]
    }
}

impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Epoch({self})")
    }
}

// ---------------------------------------------------------------------------
// Canonical encoder / decoder
// ---------------------------------------------------------------------------

struct Enc {
    buf: Vec<u8>,
}

impl Enc {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn digest(&mut self, d: &Sha256Digest) {
        self.buf.extend_from_slice(&d.0);
    }
    fn len(&mut self, n: usize, max: usize, what: &'static str) -> Result<(), WeightHostError> {
        if n > max {
            return Err(WeightHostError::LimitExceeded(what));
        }
        self.u32(n as u32);
        Ok(())
    }
    fn str(&mut self, s: &str) -> Result<(), WeightHostError> {
        self.len(s.len(), MAX_STRING_BYTES, "string")?;
        self.buf.extend_from_slice(s.as_bytes());
        Ok(())
    }
    fn blob(&mut self, b: &[u8], max: usize, what: &'static str) -> Result<(), WeightHostError> {
        self.len(b.len(), max, what)?;
        self.buf.extend_from_slice(b);
        Ok(())
    }
    fn opt<T>(
        &mut self,
        v: &Option<T>,
        f: impl FnOnce(&mut Enc, &T) -> Result<(), WeightHostError>,
    ) -> Result<(), WeightHostError> {
        match v {
            None => self.u8(0),
            Some(x) => {
                self.u8(1);
                f(self, x)?;
            }
        }
        Ok(())
    }
}

struct Dec<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Dec<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], WeightHostError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|e| *e <= self.buf.len())
            .ok_or_else(|| proto("truncated canonical payload"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N], WeightHostError> {
        let s = self.take(N)?;
        let mut a = [0u8; N];
        a.copy_from_slice(s);
        Ok(a)
    }
    fn u8(&mut self) -> Result<u8, WeightHostError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, WeightHostError> {
        Ok(u32::from_le_bytes(self.arr::<4>()?))
    }
    fn u64(&mut self) -> Result<u64, WeightHostError> {
        Ok(u64::from_le_bytes(self.arr::<8>()?))
    }
    fn i64(&mut self) -> Result<i64, WeightHostError> {
        Ok(i64::from_le_bytes(self.arr::<8>()?))
    }
    fn digest(&mut self) -> Result<Sha256Digest, WeightHostError> {
        Ok(Sha256Digest(self.arr::<32>()?))
    }
    fn bool(&mut self) -> Result<bool, WeightHostError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(proto("non-canonical bool")),
        }
    }
    /// Read a `u32` element count, bounded by `max` and by the bytes left.
    fn count(
        &mut self,
        max: usize,
        what: &'static str,
        min_elem: usize,
    ) -> Result<usize, WeightHostError> {
        let n = self.u32()? as usize;
        if n > max {
            return Err(WeightHostError::LimitExceeded(what));
        }
        if n.checked_mul(min_elem)
            .map_or(true, |b| b > self.remaining())
        {
            return Err(proto("count exceeds remaining payload"));
        }
        Ok(n)
    }
    fn str(&mut self) -> Result<String, WeightHostError> {
        let n = self.count(MAX_STRING_BYTES, "string", 1)?;
        let b = self.take(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| proto("invalid UTF-8"))
    }
    fn blob(&mut self, max: usize, what: &'static str) -> Result<Vec<u8>, WeightHostError> {
        let n = self.count(max, what, 1)?;
        Ok(self.take(n)?.to_vec())
    }
    fn opt<T>(
        &mut self,
        f: impl FnOnce(&mut Dec<'a>) -> Result<T, WeightHostError>,
    ) -> Result<Option<T>, WeightHostError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(f(self)?)),
            _ => Err(proto("non-canonical option tag")),
        }
    }
    fn finish(&self) -> Result<(), WeightHostError> {
        if self.remaining() != 0 {
            return Err(proto("trailing bytes after canonical payload"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Field values
// ---------------------------------------------------------------------------

/// Sorted, flexible key/value payload used by recipes, ABI descriptions, manifest
/// config and per-view metadata. `BTreeMap<String, _>` order is the byte order of
/// the UTF-8 keys, which is also the canonical wire order.
pub type CanonicalFields = BTreeMap<String, FieldValue>;

/// One typed canonical field value. Floats are carried as exact bit patterns, so
/// equality and hashing are bit-exact (`-0.0 != 0.0`, NaN payloads preserved).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldValue {
    Absent,
    Bool(bool),
    U64(u64),
    I64(i64),
    F32Bits(u32),
    F64Bits(u64),
    Str(String),
    Bytes(Vec<u8>),
    Digest(Sha256Digest),
    List(Vec<FieldValue>),
    U64List(Vec<u64>),
    /// LUT/codebook style `f32` payload as exact bits.
    F32BitsList(Vec<u32>),
    Map(CanonicalFields),
}

impl FieldValue {
    pub fn f32(v: f32) -> Self {
        Self::F32Bits(v.to_bits())
    }
    pub fn f64(v: f64) -> Self {
        Self::F64Bits(v.to_bits())
    }
    pub fn str(v: &str) -> Self {
        Self::Str(v.to_owned())
    }
    pub fn f32_list(v: &[f32]) -> Self {
        Self::F32BitsList(v.iter().map(|x| x.to_bits()).collect())
    }
    pub fn as_u64(&self) -> Option<u64> {
        if let Self::U64(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        if let Self::I64(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Self::Str(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        if let Self::F32Bits(v) = self {
            Some(f32::from_bits(*v))
        } else {
            None
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        if let Self::F64Bits(v) = self {
            Some(f64::from_bits(*v))
        } else {
            None
        }
    }
    pub fn as_bytes(&self) -> Option<&[u8]> {
        if let Self::Bytes(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_list(&self) -> Option<&[FieldValue]> {
        if let Self::List(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_u64_list(&self) -> Option<&[u64]> {
        if let Self::U64List(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_map(&self) -> Option<&CanonicalFields> {
        if let Self::Map(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_digest(&self) -> Option<&Sha256Digest> {
        if let Self::Digest(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_f32_list(&self) -> Option<Vec<f32>> {
        if let Self::F32BitsList(v) = self {
            Some(v.iter().map(|b| f32::from_bits(*b)).collect())
        } else {
            None
        }
    }
}

fn enc_value(e: &mut Enc, v: &FieldValue, depth: usize) -> Result<(), WeightHostError> {
    if depth > MAX_FIELD_NESTING {
        return Err(WeightHostError::LimitExceeded("field nesting"));
    }
    match v {
        FieldValue::Absent => e.u8(0),
        FieldValue::Bool(b) => {
            e.u8(1);
            e.u8(*b as u8);
        }
        FieldValue::U64(x) => {
            e.u8(2);
            e.u64(*x);
        }
        FieldValue::I64(x) => {
            e.u8(3);
            e.i64(*x);
        }
        FieldValue::F32Bits(x) => {
            e.u8(4);
            e.u32(*x);
        }
        FieldValue::F64Bits(x) => {
            e.u8(5);
            e.u64(*x);
        }
        FieldValue::Str(s) => {
            e.u8(6);
            e.str(s)?;
        }
        FieldValue::Bytes(b) => {
            e.u8(7);
            e.blob(b, MAX_BLOB_BYTES, "blob")?;
        }
        FieldValue::Digest(d) => {
            e.u8(8);
            e.digest(d);
        }
        FieldValue::List(l) => {
            e.u8(9);
            e.len(l.len(), MAX_LIST_ITEMS, "list")?;
            for x in l {
                enc_value(e, x, depth + 1)?;
            }
        }
        FieldValue::U64List(l) => {
            e.u8(10);
            e.len(l.len(), MAX_LIST_ITEMS, "u64 list")?;
            for x in l {
                e.u64(*x);
            }
        }
        FieldValue::F32BitsList(l) => {
            e.u8(11);
            e.len(l.len(), MAX_LIST_ITEMS, "f32 list")?;
            for x in l {
                e.u32(*x);
            }
        }
        FieldValue::Map(m) => {
            e.u8(12);
            enc_fields(e, m, depth + 1)?;
        }
    }
    Ok(())
}

fn enc_fields(e: &mut Enc, m: &CanonicalFields, depth: usize) -> Result<(), WeightHostError> {
    if depth > MAX_FIELD_NESTING {
        return Err(WeightHostError::LimitExceeded("field nesting"));
    }
    e.len(m.len(), MAX_LIST_ITEMS, "field map")?;
    for (k, v) in m {
        e.str(k)?;
        enc_value(e, v, depth)?;
    }
    Ok(())
}

fn dec_value(d: &mut Dec<'_>, depth: usize) -> Result<FieldValue, WeightHostError> {
    if depth > MAX_FIELD_NESTING {
        return Err(WeightHostError::LimitExceeded("field nesting"));
    }
    Ok(match d.u8()? {
        0 => FieldValue::Absent,
        1 => FieldValue::Bool(d.bool()?),
        2 => FieldValue::U64(d.u64()?),
        3 => FieldValue::I64(d.i64()?),
        4 => FieldValue::F32Bits(d.u32()?),
        5 => FieldValue::F64Bits(d.u64()?),
        6 => FieldValue::Str(d.str()?),
        7 => FieldValue::Bytes(d.blob(MAX_BLOB_BYTES, "blob")?),
        8 => FieldValue::Digest(d.digest()?),
        9 => {
            let n = d.count(MAX_LIST_ITEMS, "list", 1)?;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(dec_value(d, depth + 1)?);
            }
            FieldValue::List(v)
        }
        10 => {
            let n = d.count(MAX_LIST_ITEMS, "u64 list", 8)?;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(d.u64()?);
            }
            FieldValue::U64List(v)
        }
        11 => {
            let n = d.count(MAX_LIST_ITEMS, "f32 list", 4)?;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(d.u32()?);
            }
            FieldValue::F32BitsList(v)
        }
        12 => FieldValue::Map(dec_fields(d, depth + 1)?),
        t => return Err(proto(format!("unknown field value tag {t}"))),
    })
}

fn dec_fields(d: &mut Dec<'_>, depth: usize) -> Result<CanonicalFields, WeightHostError> {
    if depth > MAX_FIELD_NESTING {
        return Err(WeightHostError::LimitExceeded("field nesting"));
    }
    let n = d.count(MAX_LIST_ITEMS, "field map", 5)?;
    let mut m = CanonicalFields::new();
    let mut prev: Option<String> = None;
    for _ in 0..n {
        let k = d.str()?;
        if let Some(p) = &prev {
            if k.as_bytes() <= p.as_bytes() {
                return Err(proto("canonical map keys not strictly increasing"));
            }
        }
        let v = dec_value(d, depth)?;
        prev = Some(k.clone());
        m.insert(k, v);
    }
    Ok(m)
}

fn enc_str_map(e: &mut Enc, m: &BTreeMap<String, String>) -> Result<(), WeightHostError> {
    e.len(m.len(), MAX_LIST_ITEMS, "string map")?;
    for (k, v) in m {
        e.str(k)?;
        e.str(v)?;
    }
    Ok(())
}

fn dec_str_map(d: &mut Dec<'_>) -> Result<BTreeMap<String, String>, WeightHostError> {
    let n = d.count(MAX_LIST_ITEMS, "string map", 8)?;
    let mut m = BTreeMap::new();
    let mut prev: Option<String> = None;
    for _ in 0..n {
        let k = d.str()?;
        if let Some(p) = &prev {
            if k.as_bytes() <= p.as_bytes() {
                return Err(proto("canonical map keys not strictly increasing"));
            }
        }
        let v = d.str()?;
        prev = Some(k.clone());
        m.insert(k, v);
    }
    Ok(m)
}

fn enc_u64_vec(e: &mut Enc, v: &[u64]) -> Result<(), WeightHostError> {
    e.len(v.len(), MAX_VIEW_RANK, "view rank")?;
    for x in v {
        e.u64(*x);
    }
    Ok(())
}

fn dec_u64_vec(d: &mut Dec<'_>) -> Result<Vec<u64>, WeightHostError> {
    let n = d.count(MAX_VIEW_RANK, "view rank", 8)?;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(d.u64()?);
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// Recipe, loader ABI, model content
// ---------------------------------------------------------------------------

/// The *requested* layout: a schema name plus sorted flexible fields (resolved
/// transform flags, placement, allocator mode, requant rule order, expert
/// residency, ...). The tenant computes this independently; the host echoes the
/// seal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutRecipe {
    pub schema: String,
    pub fields: CanonicalFields,
}

impl LayoutRecipe {
    pub fn new(schema: &str) -> Self {
        Self {
            schema: schema.to_owned(),
            fields: CanonicalFields::new(),
        }
    }

    pub fn with(mut self, key: &str, value: FieldValue) -> Self {
        self.fields.insert(key.to_owned(), value);
        self
    }

    pub fn set(&mut self, key: &str, value: FieldValue) {
        self.fields.insert(key.to_owned(), value);
    }

    fn encode(&self, e: &mut Enc) -> Result<(), WeightHostError> {
        if self.schema.is_empty() {
            return Err(WeightHostError::InvalidArgument(
                "recipe schema must not be empty".into(),
            ));
        }
        e.str(&self.schema)?;
        enc_fields(e, &self.fields, 0)
    }

    fn decode(d: &mut Dec<'_>) -> Result<Self, WeightHostError> {
        let schema = d.str()?;
        if schema.is_empty() {
            return Err(proto("empty recipe schema"));
        }
        Ok(Self {
            schema,
            fields: dec_fields(d, 0)?,
        })
    }

    /// Canonical bytes (no domain prefix).
    pub fn encode_canonical(&self) -> Result<Vec<u8>, WeightHostError> {
        let mut e = Enc::new();
        self.encode(&mut e)?;
        Ok(e.buf)
    }

    /// Recipe seal.
    pub fn sha256(&self) -> Result<Sha256Digest, WeightHostError> {
        Ok(domain_hash(DOMAIN_RECIPE, &self.encode_canonical()?))
    }
}

/// Loader/representation ABI. Insert the serialization schema, dtype/quant
/// representation contracts, transform implementation versions and resolved
/// feature configuration as `fields`, and digests of the loader closure source
/// files as `components`. Kernel images, commits, PM-twin SHAs and other build
/// provenance are intentionally not representable here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoaderAbi {
    pub schema_id: String,
    pub fields: CanonicalFields,
    pub components: BTreeMap<String, Sha256Digest>,
}

impl LoaderAbi {
    pub fn new(schema_id: &str) -> Self {
        Self {
            schema_id: schema_id.to_owned(),
            fields: CanonicalFields::new(),
            components: BTreeMap::new(),
        }
    }

    pub fn with(mut self, key: &str, value: FieldValue) -> Self {
        self.fields.insert(key.to_owned(), value);
        self
    }

    /// Hash and record a loader-closure component given its full bytes (e.g.
    /// `include_bytes!` of the whole source file).
    pub fn add_component_bytes(&mut self, name: &str, bytes: &[u8]) {
        self.components
            .insert(name.to_owned(), Sha256Digest::of(bytes));
    }

    /// Hash and record a component file.
    pub fn add_component_file(&mut self, name: &str, path: &Path) -> Result<(), WeightHostError> {
        let f = File::open(path).map_err(io_ctx("open loader ABI component"))?;
        let (d, _) = sha256_reader(f).map_err(io_ctx("hash loader ABI component"))?;
        self.components.insert(name.to_owned(), d);
        Ok(())
    }

    pub fn encode_canonical(&self) -> Result<Vec<u8>, WeightHostError> {
        if self.schema_id.is_empty() {
            return Err(WeightHostError::InvalidArgument(
                "loader ABI schema_id must not be empty".into(),
            ));
        }
        let mut e = Enc::new();
        e.str(&self.schema_id)?;
        enc_fields(&mut e, &self.fields, 0)?;
        e.len(self.components.len(), MAX_LIST_ITEMS, "abi components")?;
        for (k, v) in &self.components {
            e.str(k)?;
            e.digest(v);
        }
        Ok(e.buf)
    }

    /// `LoaderAbiSha256`.
    pub fn sha256(&self) -> Result<Sha256Digest, WeightHostError> {
        Ok(domain_hash(DOMAIN_ABI, &self.encode_canonical()?))
    }
}

/// Full-content digest of one source (base HFQ or overlay/external file).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceDigest {
    pub role: String,
    pub len: u64,
    pub sha256: Sha256Digest,
}

/// `ModelContentSha256` over the ordered sources that supply tensor bytes. Order
/// is significant (base first, then resolved overlays in application order).
pub fn model_content_sha256(sources: &[SourceDigest]) -> Result<Sha256Digest, WeightHostError> {
    if sources.is_empty() {
        return Err(WeightHostError::InvalidArgument(
            "model content needs at least one source".into(),
        ));
    }
    let mut e = Enc::new();
    e.len(sources.len(), MAX_LIST_ITEMS, "sources")?;
    for s in sources {
        if s.role.is_empty() {
            return Err(WeightHostError::InvalidArgument(
                "source role must not be empty".into(),
            ));
        }
        e.str(&s.role)?;
        e.u64(s.len);
        e.digest(&s.sha256);
    }
    Ok(domain_hash(DOMAIN_MODEL, &e.buf))
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/// One exported physical allocation (one FD in v1). `owner_id` derives
/// deterministically from the first canonical owner role.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedAllocation {
    pub owner_id: String,
    /// Rounded physical size mapped by the importer (multiple of `granularity`).
    pub physical_bytes: u64,
    /// Bytes that carry weight data or initialized padding that views may touch.
    pub logical_bytes: u64,
    pub granularity: u64,
    pub device_arch: String,
    pub representation_tag: String,
}

/// Matrix interpretation of a view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixDesc {
    pub m: u64,
    pub k: u64,
    pub row_stride_bytes: u64,
}

/// Quantization description of a view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantDesc {
    pub format: String,
    pub group_size: u32,
    pub block_elems: u32,
    pub block_bytes: u32,
    pub packing: String,
}

/// Typed tensor/view record. Views of the same storage (tied weights) are
/// separate records with identical `(owner_id, byte_offset, byte_extent)`; the
/// secondary names the primary in `alias_of`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightView {
    /// Logical role name; manifests are strictly sorted by it.
    pub role: String,
    pub owner_id: String,
    pub byte_offset: u64,
    pub byte_extent: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Element strides; empty means contiguous, otherwise one per dimension.
    pub strides: Vec<u64>,
    pub matrix: Option<MatrixDesc>,
    pub quant: Option<QuantDesc>,
    /// Companion views by kind (`"scales"`, `"zeros"`, `"lut"`, `"awq_recip"`, ...)
    /// to their roles.
    pub companions: BTreeMap<String, String>,
    pub alias_of: Option<String>,
    /// Host-side dispatch metadata, LUT/codebook payloads, fold flags, ...
    pub metadata: CanonicalFields,
}

/// Complete typed manifest. `allocations` are strictly sorted by `owner_id`
/// (allocation `i` travels as received FD `i`); `views` strictly sorted by `role`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightManifest {
    pub arch: String,
    /// Architecture config, dense topology, resolved dispatch/transform facts.
    pub config: CanonicalFields,
    /// Opaque, architecture-defined dispatch metadata (non-buffer dispatch facts
    /// the importer must restore). Hashed, encoded and decoded verbatim.
    pub dispatch_metadata: Vec<u8>,
    pub allocations: Vec<SharedAllocation>,
    pub views: Vec<WeightView>,
}

fn bad_manifest<S: Into<String>>(s: S) -> WeightHostError {
    WeightHostError::ManifestInvalid(s.into())
}

impl SharedAllocation {
    fn encode(&self, e: &mut Enc) -> Result<(), WeightHostError> {
        e.str(&self.owner_id)?;
        e.u64(self.physical_bytes);
        e.u64(self.logical_bytes);
        e.u64(self.granularity);
        e.str(&self.device_arch)?;
        e.str(&self.representation_tag)
    }
    fn decode(d: &mut Dec<'_>) -> Result<Self, WeightHostError> {
        Ok(Self {
            owner_id: d.str()?,
            physical_bytes: d.u64()?,
            logical_bytes: d.u64()?,
            granularity: d.u64()?,
            device_arch: d.str()?,
            representation_tag: d.str()?,
        })
    }
}

impl WeightView {
    fn encode(&self, e: &mut Enc) -> Result<(), WeightHostError> {
        e.str(&self.role)?;
        e.str(&self.owner_id)?;
        e.u64(self.byte_offset);
        e.u64(self.byte_extent);
        e.str(&self.dtype)?;
        enc_u64_vec(e, &self.shape)?;
        enc_u64_vec(e, &self.strides)?;
        e.opt(&self.matrix, |e, m| {
            e.u64(m.m);
            e.u64(m.k);
            e.u64(m.row_stride_bytes);
            Ok(())
        })?;
        e.opt(&self.quant, |e, q| {
            e.str(&q.format)?;
            e.u32(q.group_size);
            e.u32(q.block_elems);
            e.u32(q.block_bytes);
            e.str(&q.packing)
        })?;
        enc_str_map(e, &self.companions)?;
        e.opt(&self.alias_of, |e, a| e.str(a))?;
        enc_fields(e, &self.metadata, 0)
    }

    fn decode(d: &mut Dec<'_>) -> Result<Self, WeightHostError> {
        Ok(Self {
            role: d.str()?,
            owner_id: d.str()?,
            byte_offset: d.u64()?,
            byte_extent: d.u64()?,
            dtype: d.str()?,
            shape: dec_u64_vec(d)?,
            strides: dec_u64_vec(d)?,
            matrix: d.opt(|d| {
                Ok(MatrixDesc {
                    m: d.u64()?,
                    k: d.u64()?,
                    row_stride_bytes: d.u64()?,
                })
            })?,
            quant: d.opt(|d| {
                Ok(QuantDesc {
                    format: d.str()?,
                    group_size: d.u32()?,
                    block_elems: d.u32()?,
                    block_bytes: d.u32()?,
                    packing: d.str()?,
                })
            })?,
            companions: dec_str_map(d)?,
            alias_of: d.opt(|d| d.str())?,
            metadata: dec_fields(d, 0)?,
        })
    }
}

impl WeightManifest {
    /// Sort allocations by `owner_id` and views by `role` into canonical order.
    pub fn canonicalize(&mut self) {
        self.allocations
            .sort_by(|a, b| a.owner_id.as_bytes().cmp(b.owner_id.as_bytes()));
        self.views
            .sort_by(|a, b| a.role.as_bytes().cmp(b.role.as_bytes()));
    }

    /// Sum of rounded physical bytes (checked).
    pub fn unique_bytes(&self) -> Result<u64, WeightHostError> {
        self.allocations
            .iter()
            .try_fold(0u64, |acc, a| acc.checked_add(a.physical_bytes))
            .ok_or_else(|| bad_manifest("physical byte total overflows u64"))
    }

    pub fn allocation_index(&self, owner_id: &str) -> Option<usize> {
        self.allocations
            .binary_search_by(|a| a.owner_id.as_bytes().cmp(owner_id.as_bytes()))
            .ok()
    }

    pub fn view(&self, role: &str) -> Option<&WeightView> {
        self.views
            .binary_search_by(|v| v.role.as_bytes().cmp(role.as_bytes()))
            .ok()
            .map(|i| &self.views[i])
    }

    /// Structural validation: canonical ordering/uniqueness, checked extents,
    /// view bounds, alias topology and companion references.
    pub fn validate(&self) -> Result<(), WeightHostError> {
        if self.arch.is_empty() {
            return Err(bad_manifest("empty architecture tag"));
        }
        if self.allocations.is_empty() {
            return Err(bad_manifest("manifest has no allocations"));
        }
        if self.allocations.len() > MAX_ALLOCATIONS {
            return Err(WeightHostError::LimitExceeded("allocations"));
        }
        if self.views.len() > MAX_VIEWS {
            return Err(WeightHostError::LimitExceeded("views"));
        }
        let mut prev: Option<&str> = None;
        for a in &self.allocations {
            if a.owner_id.is_empty() || a.device_arch.is_empty() || a.representation_tag.is_empty()
            {
                return Err(bad_manifest(format!(
                    "allocation {:?} has an empty tag",
                    a.owner_id
                )));
            }
            if let Some(p) = prev {
                if a.owner_id.as_bytes() <= p.as_bytes() {
                    return Err(bad_manifest("allocations not strictly sorted by owner_id"));
                }
            }
            prev = Some(&a.owner_id);
            if a.granularity == 0 || a.physical_bytes == 0 {
                return Err(bad_manifest(format!(
                    "allocation {:?} has zero granularity/physical size",
                    a.owner_id
                )));
            }
            if a.physical_bytes % a.granularity != 0 {
                return Err(bad_manifest(format!(
                    "allocation {:?} physical size is not granularity aligned",
                    a.owner_id
                )));
            }
            if a.logical_bytes > a.physical_bytes {
                return Err(bad_manifest(format!(
                    "allocation {:?} logical bytes exceed physical bytes",
                    a.owner_id
                )));
            }
            if a.logical_bytes
                .checked_add(EXPORT_WEIGHT_TAIL_PAD_BYTES)
                .map_or(true, |need| a.physical_bytes < need)
            {
                return Err(bad_manifest(format!(
                    "allocation {:?} lacks the mandatory {}-byte tail pad beyond its logical bytes",
                    a.owner_id, EXPORT_WEIGHT_TAIL_PAD_BYTES
                )));
            }
        }
        self.unique_bytes()?;

        let mut prev: Option<&str> = None;
        for v in &self.views {
            if v.role.is_empty() || v.dtype.is_empty() {
                return Err(bad_manifest("view with empty role or dtype"));
            }
            if let Some(p) = prev {
                if v.role.as_bytes() <= p.as_bytes() {
                    return Err(bad_manifest("views not strictly sorted by role"));
                }
            }
            prev = Some(&v.role);
            let Some(ai) = self.allocation_index(&v.owner_id) else {
                return Err(bad_manifest(format!(
                    "view {:?} references unknown owner {:?}",
                    v.role, v.owner_id
                )));
            };
            let alloc = &self.allocations[ai];
            let end = v
                .byte_offset
                .checked_add(v.byte_extent)
                .ok_or_else(|| bad_manifest(format!("view {:?} extent overflows", v.role)))?;
            if end > alloc.logical_bytes {
                return Err(bad_manifest(format!(
                    "view {:?} exceeds owner logical bytes",
                    v.role
                )));
            }
            if v.shape.len() > MAX_VIEW_RANK || v.strides.len() > MAX_VIEW_RANK {
                return Err(WeightHostError::LimitExceeded("view rank"));
            }
            if !v.strides.is_empty() && v.strides.len() != v.shape.len() {
                return Err(bad_manifest(format!(
                    "view {:?} stride rank differs from shape rank",
                    v.role
                )));
            }
            if v.shape
                .iter()
                .try_fold(1u64, |a, d| a.checked_mul(*d))
                .is_none()
            {
                return Err(bad_manifest(format!(
                    "view {:?} element count overflows",
                    v.role
                )));
            }
            if let Some(m) = &v.matrix {
                if m.m == 0 || m.k == 0 || m.row_stride_bytes == 0 {
                    return Err(bad_manifest(format!(
                        "view {:?} has a zero matrix dimension",
                        v.role
                    )));
                }
                let need = m.m.checked_mul(m.row_stride_bytes);
                if need.map_or(true, |n| n > v.byte_extent) {
                    return Err(bad_manifest(format!(
                        "view {:?} rows exceed its byte extent",
                        v.role
                    )));
                }
            }
            if let Some(q) = &v.quant {
                if q.format.is_empty() {
                    return Err(bad_manifest(format!(
                        "view {:?} has an empty quant format",
                        v.role
                    )));
                }
            }
            for (kind, role) in &v.companions {
                if kind.is_empty() || self.view(role).is_none() {
                    return Err(bad_manifest(format!(
                        "view {:?} has a dangling companion {:?}",
                        v.role, kind
                    )));
                }
            }
            if let Some(target) = &v.alias_of {
                let Some(t) = self.view(target) else {
                    return Err(bad_manifest(format!(
                        "alias {:?} targets unknown view {:?}",
                        v.role, target
                    )));
                };
                if t.alias_of.is_some() || t.role == v.role {
                    return Err(bad_manifest(format!(
                        "alias {:?} targets an alias or itself",
                        v.role
                    )));
                }
                if t.owner_id != v.owner_id
                    || t.byte_offset != v.byte_offset
                    || t.byte_extent != v.byte_extent
                {
                    return Err(bad_manifest(format!(
                        "alias {:?} storage differs from its target",
                        v.role
                    )));
                }
            }
        }
        Ok(())
    }

    fn encode(&self, e: &mut Enc) -> Result<(), WeightHostError> {
        e.str(&self.arch)?;
        enc_fields(e, &self.config, 0)?;
        e.blob(&self.dispatch_metadata, MAX_BLOB_BYTES, "dispatch metadata")?;
        e.len(self.allocations.len(), MAX_ALLOCATIONS, "allocations")?;
        for a in &self.allocations {
            a.encode(e)?;
        }
        e.len(self.views.len(), MAX_VIEWS, "views")?;
        for v in &self.views {
            v.encode(e)?;
        }
        Ok(())
    }

    fn decode(d: &mut Dec<'_>) -> Result<Self, WeightHostError> {
        let arch = d.str()?;
        let config = dec_fields(d, 0)?;
        let dispatch_metadata = d.blob(MAX_BLOB_BYTES, "dispatch metadata")?;
        let na = d.count(MAX_ALLOCATIONS, "allocations", 30)?;
        let mut allocations = Vec::with_capacity(na);
        for _ in 0..na {
            allocations.push(SharedAllocation::decode(d)?);
        }
        let nv = d.count(MAX_VIEWS, "views", 40)?;
        let mut views = Vec::with_capacity(nv);
        for _ in 0..nv {
            views.push(WeightView::decode(d)?);
        }
        Ok(Self {
            arch,
            config,
            dispatch_metadata,
            allocations,
            views,
        })
    }

    /// Validate and canonically encode.
    pub fn encode_canonical(&self) -> Result<Vec<u8>, WeightHostError> {
        self.validate()?;
        let mut e = Enc::new();
        self.encode(&mut e)?;
        Ok(e.buf)
    }

    /// Strictly decode (no trailing bytes) and validate.
    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, WeightHostError> {
        let mut d = Dec::new(bytes);
        let m = Self::decode(&mut d)?;
        d.finish()?;
        m.validate()?;
        Ok(m)
    }
}

/// Recipe and final layout seals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutSeals {
    pub recipe_sha256: Sha256Digest,
    pub layout_sha256: Sha256Digest,
}

/// `WeightLayoutSha256 = SHA256("hipfire-weight-layout/v1\0" ||
/// canonical(model, abi, recipe, manifest))`, validating the manifest first.
pub fn compute_layout_seals(
    model_content: &Sha256Digest,
    loader_abi: &Sha256Digest,
    recipe: &LayoutRecipe,
    manifest: &WeightManifest,
) -> Result<LayoutSeals, WeightHostError> {
    manifest.validate()?;
    let mut e = Enc::new();
    e.digest(model_content);
    e.digest(loader_abi);
    recipe.encode(&mut e)?;
    manifest.encode(&mut e)?;
    Ok(LayoutSeals {
        recipe_sha256: recipe.sha256()?,
        layout_sha256: domain_hash(DOMAIN_LAYOUT, &e.buf),
    })
}

// ---------------------------------------------------------------------------
// Source identity and certificates
// ---------------------------------------------------------------------------

/// File identity used to bind a certificate to the current bytes' container.
/// Includes ctime so an mtime reset cannot hide a modification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceFileIdentity {
    pub dev: u64,
    pub ino: u64,
    pub len: u64,
    pub mtime_sec: i64,
    pub mtime_nsec: i64,
    pub ctime_sec: i64,
    pub ctime_nsec: i64,
    pub uid: u32,
    pub mode: u32,
}

impl SourceFileIdentity {
    pub fn from_metadata(md: &std::fs::Metadata) -> Self {
        Self {
            dev: md.dev(),
            ino: md.ino(),
            len: md.size(),
            mtime_sec: md.mtime(),
            mtime_nsec: md.mtime_nsec(),
            ctime_sec: md.ctime(),
            ctime_nsec: md.ctime_nsec(),
            uid: md.uid(),
            mode: md.mode(),
        }
    }
}

fn check_immutable_source(md: &std::fs::Metadata, what: &str) -> Result<(), WeightHostError> {
    if !md.file_type().is_file() {
        return Err(WeightHostError::SourceNotImmutable(format!(
            "{what} is not a regular file"
        )));
    }
    if md.mode() & 0o222 != 0 {
        return Err(WeightHostError::SourceNotImmutable(format!(
            "{what} has write permission bits (mode {:o}); provision it read-only",
            md.mode() & 0o7777
        )));
    }
    Ok(())
}

/// A locally trusted statement "this file identity hashes to this full SHA-256".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceCertificate {
    role: String,
    identity: SourceFileIdentity,
    sha256: Sha256Digest,
}

impl SourceCertificate {
    pub fn role(&self) -> &str {
        &self.role
    }
    pub fn identity(&self) -> &SourceFileIdentity {
        &self.identity
    }
    pub fn sha256(&self) -> &Sha256Digest {
        &self.sha256
    }
    pub fn len(&self) -> u64 {
        self.identity.len
    }
    /// Input for [`model_content_sha256`].
    pub fn to_source_digest(&self) -> SourceDigest {
        SourceDigest {
            role: self.role.clone(),
            len: self.identity.len,
            sha256: self.sha256,
        }
    }

    pub fn encode(&self) -> Result<Vec<u8>, WeightHostError> {
        let mut e = Enc::new();
        e.buf.extend_from_slice(CERT_MAGIC);
        e.u32(CERT_VERSION);
        e.str(&self.role)?;
        let i = &self.identity;
        e.u64(i.dev);
        e.u64(i.ino);
        e.u64(i.len);
        e.i64(i.mtime_sec);
        e.i64(i.mtime_nsec);
        e.i64(i.ctime_sec);
        e.i64(i.ctime_nsec);
        e.u32(i.uid);
        e.u32(i.mode);
        e.digest(&self.sha256);
        let sum = domain_hash(DOMAIN_CERT, &e.buf);
        e.digest(&sum);
        Ok(e.buf)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, WeightHostError> {
        let bad = |s: &str| WeightHostError::CertificateInvalid(s.to_owned());
        if bytes.len() < CERT_MAGIC.len() + 4 + 32 || &bytes[..8] != CERT_MAGIC {
            return Err(bad("bad magic or truncated"));
        }
        let (body, sum) = bytes.split_at(bytes.len() - 32);
        if domain_hash(DOMAIN_CERT, body).0[..] != sum[..] {
            return Err(bad("checksum mismatch"));
        }
        let mut d = Dec::new(&body[8..]);
        let wrap = |e: WeightHostError| WeightHostError::CertificateInvalid(e.to_string());
        if d.u32().map_err(wrap)? != CERT_VERSION {
            return Err(bad("unsupported certificate version"));
        }
        let role = d.str().map_err(wrap)?;
        let identity = SourceFileIdentity {
            dev: d.u64().map_err(wrap)?,
            ino: d.u64().map_err(wrap)?,
            len: d.u64().map_err(wrap)?,
            mtime_sec: d.i64().map_err(wrap)?,
            mtime_nsec: d.i64().map_err(wrap)?,
            ctime_sec: d.i64().map_err(wrap)?,
            ctime_nsec: d.i64().map_err(wrap)?,
            uid: d.u32().map_err(wrap)?,
            mode: d.u32().map_err(wrap)?,
        };
        let sha256 = d.digest().map_err(wrap)?;
        d.finish().map_err(wrap)?;
        if role.is_empty() {
            return Err(bad("empty role"));
        }
        Ok(Self {
            role,
            identity,
            sha256,
        })
    }

    /// Atomically publish the certificate read-only (mode 0444) at `path`.
    pub fn write_to(&self, path: &Path) -> Result<(), WeightHostError> {
        let bytes = self.encode()?;
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(format!(".tmp.{}", std::process::id()));
        let tmp = PathBuf::from(tmp);
        let result = (|| -> Result<(), WeightHostError> {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o444)
                .open(&tmp)
                .map_err(io_ctx("create certificate"))?;
            f.write_all(&bytes).map_err(io_ctx("write certificate"))?;
            f.sync_all().map_err(io_ctx("sync certificate"))?;
            std::fs::rename(&tmp, path).map_err(io_ctx("publish certificate"))
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result
    }

    /// Load a certificate, rejecting symlinks, foreign owners, any write
    /// permission bit (stale/writable certificate) and oversize files.
    pub fn load_trusted(path: &Path) -> Result<Self, WeightHostError> {
        let f = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)
            .map_err(io_ctx("open certificate"))?;
        let md = f.metadata().map_err(io_ctx("stat certificate"))?;
        if !md.file_type().is_file() {
            return Err(WeightHostError::CertificateInvalid(
                "certificate is not a regular file".into(),
            ));
        }
        if md.uid() != euid() {
            return Err(WeightHostError::CertificateInvalid(
                "certificate not owned by the current user".into(),
            ));
        }
        if md.mode() & 0o222 != 0 {
            return Err(WeightHostError::CertificateInvalid(format!(
                "certificate is writable (mode {:o}); refusing to trust it",
                md.mode() & 0o7777
            )));
        }
        if md.len() > MAX_CERT_FILE_BYTES {
            return Err(WeightHostError::CertificateInvalid(
                "certificate too large".into(),
            ));
        }
        let mut bytes = Vec::with_capacity(md.len() as usize);
        (&f).take(MAX_CERT_FILE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(io_ctx("read certificate"))?;
        Self::decode(&bytes)
    }

    /// Tenant-side check against an already open source: identity must equal the
    /// certified one and the file must be read-only. No content scan.
    pub fn verify_against_file(&self, file: &File) -> Result<(), WeightHostError> {
        let md = file.metadata().map_err(io_ctx("stat source"))?;
        check_immutable_source(&md, "source")?;
        let now = SourceFileIdentity::from_metadata(&md);
        if now != self.identity {
            return Err(WeightHostError::SourceMutated(format!(
                "source identity differs from certificate {:?} (stale or modified)",
                self.role
            )));
        }
        Ok(())
    }

    /// Load the trusted certificate, open the source and verify identity.
    /// Returns the retained source FD together with the certificate.
    pub fn verify_source_path(
        cert_path: &Path,
        source_path: &Path,
    ) -> Result<(File, Self), WeightHostError> {
        let cert = Self::load_trusted(cert_path)?;
        let file = File::open(source_path).map_err(io_ctx("open source"))?;
        cert.verify_against_file(&file)?;
        let path_md = std::fs::metadata(source_path).map_err(io_ctx("stat source path"))?;
        let fd_md = file.metadata().map_err(io_ctx("stat source"))?;
        if path_md.dev() != fd_md.dev() || path_md.ino() != fd_md.ino() {
            return Err(WeightHostError::SourceMutated(
                "source path was replaced while opening".into(),
            ));
        }
        Ok((file, cert))
    }
}

/// Host-side one-time hash: opens `path`, requires a read-only regular file,
/// streams the whole content through SHA-256 and requires identical identity
/// before/after (and that the path still names the same inode). The retained
/// `File` must be kept open for the host's lifetime.
pub fn certify_source(
    path: &Path,
    role: &str,
) -> Result<(File, SourceCertificate), WeightHostError> {
    if role.is_empty() {
        return Err(WeightHostError::InvalidArgument(
            "source role must not be empty".into(),
        ));
    }
    let file = File::open(path).map_err(io_ctx("open source"))?;
    let before = file.metadata().map_err(io_ctx("stat source"))?;
    check_immutable_source(&before, "source")?;
    let before_id = SourceFileIdentity::from_metadata(&before);
    let (sha256, total) = sha256_reader(&file).map_err(io_ctx("hash source"))?;
    let after = file.metadata().map_err(io_ctx("stat source"))?;
    let after_id = SourceFileIdentity::from_metadata(&after);
    if after_id != before_id || total != before_id.len {
        return Err(WeightHostError::SourceMutated(
            "source changed while hashing".into(),
        ));
    }
    let path_md = std::fs::metadata(path).map_err(io_ctx("stat source path"))?;
    if path_md.dev() != after.dev() || path_md.ino() != after.ino() {
        return Err(WeightHostError::SourceMutated(
            "source path was replaced while hashing".into(),
        ));
    }
    Ok((
        file,
        SourceCertificate {
            role: role.to_owned(),
            identity: after_id,
            sha256,
        },
    ))
}

// ---------------------------------------------------------------------------
// Handshake types
// ---------------------------------------------------------------------------

/// Physical card identity (selection/handshake only; never part of the layout
/// hash).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardIdentity {
    pub uuid: String,
    pub pci_bus_id: String,
    pub device_arch: String,
}

/// Build provenance for diagnostics only. Never compared, never hashed into an
/// identity.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BuildProvenance {
    pub binary_sha256: Option<Sha256Digest>,
    pub kernel_image_sha256: Option<Sha256Digest>,
    pub label: String,
}

/// Client-local connection options (not sent on the wire).
#[derive(Clone, Copy, Debug)]
pub struct ConnectOptions {
    /// One absolute deadline covering connect through the last FD batch; also the
    /// write budget of each later control message.
    pub handshake_timeout: Duration,
    /// Required host pid, checked against `SO_PEERCRED` when set.
    pub expected_host_pid: Option<u32>,
}

impl Default for ConnectOptions {
    fn default() -> Self {
        Self {
            handshake_timeout: DEFAULT_HANDSHAKE_TIMEOUT,
            expected_host_pid: None,
        }
    }
}

/// Client hello. The recipe body is sent (the host derives and compares the
/// recipe seal) and retained so the client can recompute the final layout seal.
#[derive(Clone, Debug)]
pub struct HostHello {
    pub card: CardIdentity,
    /// `ModelContentSha256`.
    pub source_sha256: Sha256Digest,
    /// `LoaderAbiSha256`.
    pub loader_abi_sha256: Sha256Digest,
    pub recipe: LayoutRecipe,
    pub admission_token: String,
    pub provenance: BuildProvenance,
    pub options: ConnectOptions,
}

impl HostHello {
    fn validate(&self) -> Result<(), WeightHostError> {
        let c = &self.card;
        if c.uuid.is_empty() || c.pci_bus_id.is_empty() || c.device_arch.is_empty() {
            return Err(WeightHostError::InvalidArgument(
                "card identity fields must not be empty".into(),
            ));
        }
        if self.admission_token.len() > MAX_ADMISSION_TOKEN_BYTES {
            return Err(WeightHostError::LimitExceeded("admission token"));
        }
        if self.options.handshake_timeout.is_zero() {
            return Err(WeightHostError::InvalidArgument(
                "handshake timeout must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

/// Peer credentials from `SO_PEERCRED`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerCredentials {
    pub pid: u32,
    pub uid: u32,
    pub gid: u32,
}

/// Same-effective-UID requirement plus optional exact pid.
pub fn verify_peer(
    cred: &PeerCredentials,
    expected_pid: Option<u32>,
) -> Result<(), WeightHostError> {
    if cred.uid != euid() {
        return Err(WeightHostError::PeerCredentials(format!(
            "peer uid {} differs from effective uid {}",
            cred.uid,
            euid()
        )));
    }
    if cred.pid == 0 {
        return Err(WeightHostError::PeerCredentials(
            "peer pid is not visible".into(),
        ));
    }
    if let Some(p) = expected_pid {
        if cred.pid != p {
            return Err(WeightHostError::PeerCredentials(format!(
                "peer pid {} is not the expected pid {p}",
                cred.pid
            )));
        }
    }
    Ok(())
}

/// Reply to a successful import, as observed by the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImportAck {
    pub epoch: Epoch,
    pub layout_sha256: Sha256Digest,
    pub imported_bytes: u64,
}

// ---------------------------------------------------------------------------
// Wire messages
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum MsgKind {
    Hello = 1,
    Ready = 2,
    FdBatch = 3,
    ImportAck = 4,
    Release = 5,
    Reject = 6,
}

impl MsgKind {
    fn from_u32(v: u32) -> Result<Self, WeightHostError> {
        Ok(match v {
            1 => Self::Hello,
            2 => Self::Ready,
            3 => Self::FdBatch,
            4 => Self::ImportAck,
            5 => Self::Release,
            6 => Self::Reject,
            _ => return Err(proto(format!("unknown message kind {v}"))),
        })
    }
}

struct Frame {
    kind: MsgKind,
    payload: Vec<u8>,
    fds: Vec<OwnedFd>,
}

fn encode_hello(h: &HostHello) -> Result<Vec<u8>, WeightHostError> {
    let mut e = Enc::new();
    e.str(&h.card.uuid)?;
    e.str(&h.card.pci_bus_id)?;
    e.str(&h.card.device_arch)?;
    e.digest(&h.source_sha256);
    e.digest(&h.loader_abi_sha256);
    h.recipe.encode(&mut e)?;
    e.str(&h.admission_token)?;
    e.opt(&h.provenance.binary_sha256, |e, d| {
        e.digest(d);
        Ok(())
    })?;
    e.opt(&h.provenance.kernel_image_sha256, |e, d| {
        e.digest(d);
        Ok(())
    })?;
    e.str(&h.provenance.label)?;
    Ok(e.buf)
}

fn decode_hello(payload: &[u8]) -> Result<HostHello, WeightHostError> {
    let mut d = Dec::new(payload);
    let card = CardIdentity {
        uuid: d.str()?,
        pci_bus_id: d.str()?,
        device_arch: d.str()?,
    };
    let source_sha256 = d.digest()?;
    let loader_abi_sha256 = d.digest()?;
    let recipe = LayoutRecipe::decode(&mut d)?;
    let admission_token = d.str()?;
    let binary_sha256 = d.opt(|d| d.digest())?;
    let kernel_image_sha256 = d.opt(|d| d.digest())?;
    let label = d.str()?;
    d.finish()?;
    let hello = HostHello {
        card,
        source_sha256,
        loader_abi_sha256,
        recipe,
        admission_token,
        provenance: BuildProvenance {
            binary_sha256,
            kernel_image_sha256,
            label,
        },
        options: ConnectOptions::default(),
    };
    if hello.card.uuid.is_empty()
        || hello.card.pci_bus_id.is_empty()
        || hello.card.device_arch.is_empty()
    {
        return Err(proto("empty card identity"));
    }
    Ok(hello)
}

struct ReadyMsg {
    epoch: Epoch,
    source_sha256: Sha256Digest,
    loader_abi_sha256: Sha256Digest,
    recipe_sha256: Sha256Digest,
    layout_sha256: Sha256Digest,
    unique_bytes: u64,
    fd_count: u32,
    manifest: WeightManifest,
}

fn encode_ready(r: &ReadyMsg) -> Result<Vec<u8>, WeightHostError> {
    let mut e = Enc::new();
    e.buf.extend_from_slice(&r.epoch.0);
    e.digest(&r.source_sha256);
    e.digest(&r.loader_abi_sha256);
    e.digest(&r.recipe_sha256);
    e.digest(&r.layout_sha256);
    e.u64(r.unique_bytes);
    e.u32(r.fd_count);
    r.manifest.encode(&mut e)?;
    Ok(e.buf)
}

fn decode_ready(payload: &[u8]) -> Result<ReadyMsg, WeightHostError> {
    let mut d = Dec::new(payload);
    let epoch = Epoch(d.arr::<16>()?);
    let source_sha256 = d.digest()?;
    let loader_abi_sha256 = d.digest()?;
    let recipe_sha256 = d.digest()?;
    let layout_sha256 = d.digest()?;
    let unique_bytes = d.u64()?;
    let fd_count = d.u32()?;
    let manifest = WeightManifest::decode(&mut d)?;
    d.finish()?;
    manifest.validate()?;
    Ok(ReadyMsg {
        epoch,
        source_sha256,
        loader_abi_sha256,
        recipe_sha256,
        layout_sha256,
        unique_bytes,
        fd_count,
        manifest,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchMsg {
    index: u32,
    total: u32,
    first: u32,
    count: u32,
}

fn encode_batch(b: &BatchMsg) -> Vec<u8> {
    let mut e = Enc::new();
    e.u32(b.index);
    e.u32(b.total);
    e.u32(b.first);
    e.u32(b.count);
    e.buf
}

fn decode_batch(payload: &[u8]) -> Result<BatchMsg, WeightHostError> {
    let mut d = Dec::new(payload);
    let b = BatchMsg {
        index: d.u32()?,
        total: d.u32()?,
        first: d.u32()?,
        count: d.u32()?,
    };
    d.finish()?;
    Ok(b)
}

fn encode_epoch_layout(epoch: &Epoch, layout: &Sha256Digest, imported: Option<u64>) -> Vec<u8> {
    let mut e = Enc::new();
    e.buf.extend_from_slice(&epoch.0);
    e.digest(layout);
    if let Some(n) = imported {
        e.u64(n);
    }
    e.buf
}

fn decode_epoch_layout(
    payload: &[u8],
    with_bytes: bool,
) -> Result<(Epoch, Sha256Digest, u64), WeightHostError> {
    let mut d = Dec::new(payload);
    let epoch = Epoch(d.arr::<16>()?);
    let layout = d.digest()?;
    let bytes = if with_bytes { d.u64()? } else { 0 };
    d.finish()?;
    Ok((epoch, layout, bytes))
}

fn encode_reject(reason: &str) -> Vec<u8> {
    let mut end = reason.len().min(MAX_REJECT_REASON_BYTES);
    while !reason.is_char_boundary(end) {
        end -= 1;
    }
    let mut e = Enc::new();
    e.u32(end as u32);
    e.buf.extend_from_slice(&reason.as_bytes()[..end]);
    e.buf
}

fn decode_reject(payload: &[u8]) -> Result<String, WeightHostError> {
    let mut d = Dec::new(payload);
    let n = d.count(MAX_REJECT_REASON_BYTES, "reject reason", 1)?;
    let s = String::from_utf8(d.take(n)?.to_vec())
        .map_err(|_| proto("invalid UTF-8 in reject reason"))?;
    d.finish()?;
    Ok(s)
}

fn frame_bytes(version: u32, kind: u32, payload: &[u8], fd_count: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(HEADER_LEN + payload.len());
    v.extend_from_slice(&MAGIC);
    v.extend_from_slice(&version.to_le_bytes());
    v.extend_from_slice(&kind.to_le_bytes());
    v.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    v.extend_from_slice(&fd_count.to_le_bytes());
    v.extend_from_slice(payload);
    v
}

// ---------------------------------------------------------------------------
// Socket plumbing (nonblocking fd + poll against one absolute deadline)
// ---------------------------------------------------------------------------

struct Conn(OwnedFd);

impl Conn {
    fn fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
    fn from_stream(s: UnixStream) -> Result<Self, WeightHostError> {
        s.set_nonblocking(true).map_err(io_ctx("set nonblocking"))?;
        Ok(Self(OwnedFd::from(s)))
    }
}

fn euid() -> u32 {
    // SAFETY: geteuid has no preconditions.
    unsafe { libc::geteuid() }
}

#[repr(C, align(8))]
struct CmsgBuf([u8; 512]);

fn wait_fd(
    fd: RawFd,
    events: i16,
    deadline: Option<Instant>,
    what: &'static str,
) -> Result<(), WeightHostError> {
    loop {
        let timeout_ms: libc::c_int = match deadline {
            None => -1,
            Some(d) => {
                let now = Instant::now();
                if now >= d {
                    return Err(WeightHostError::Timeout(what));
                }
                let ms = ((d - now).as_nanos() + 999_999) / 1_000_000;
                ms.min(libc::c_int::MAX as u128) as libc::c_int
            }
        };
        let mut p = libc::pollfd {
            fd,
            events,
            revents: 0,
        };
        // SAFETY: p is a valid pollfd for the duration of the call.
        let rc = unsafe { libc::poll(&mut p, 1, timeout_ms) };
        if rc < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(WeightHostError::Io {
                context: "poll",
                source: e,
            });
        }
        if rc == 0 {
            return Err(WeightHostError::Timeout(what));
        }
        return Ok(());
    }
}

fn sendmsg_once(fd: RawFd, data: &[u8], fds: &[RawFd]) -> io::Result<usize> {
    debug_assert!(fds.len() <= 64);
    let mut iov = libc::iovec {
        iov_base: data.as_ptr() as *mut libc::c_void,
        iov_len: data.len(),
    };
    let mut ctl = CmsgBuf([0u8; 512]);
    // SAFETY: msghdr is plain data; all pointers below outlive the sendmsg call.
    unsafe {
        let mut msg: libc::msghdr = std::mem::zeroed();
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        if !fds.is_empty() {
            let payload = (fds.len() * std::mem::size_of::<RawFd>()) as libc::c_uint;
            msg.msg_control = ctl.0.as_mut_ptr() as *mut libc::c_void;
            msg.msg_controllen = libc::CMSG_SPACE(payload) as _;
            let cmsg = libc::CMSG_FIRSTHDR(&msg);
            (*cmsg).cmsg_level = libc::SOL_SOCKET;
            (*cmsg).cmsg_type = libc::SCM_RIGHTS;
            (*cmsg).cmsg_len = libc::CMSG_LEN(payload) as _;
            std::ptr::copy_nonoverlapping(
                fds.as_ptr() as *const u8,
                libc::CMSG_DATA(cmsg),
                payload as usize,
            );
        }
        loop {
            let n = libc::sendmsg(fd, &msg, libc::MSG_NOSIGNAL | libc::MSG_DONTWAIT);
            if n < 0 {
                let e = io::Error::last_os_error();
                if e.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(e);
            }
            return Ok(n as usize);
        }
    }
}

/// Every received descriptor is wrapped into an `OwnedFd` before any error can
/// be returned, so no path leaks a descriptor.
fn recvmsg_once(fd: RawFd, buf: &mut [u8], fds: &mut Vec<OwnedFd>) -> io::Result<usize> {
    let mut iov = libc::iovec {
        iov_base: buf.as_mut_ptr() as *mut libc::c_void,
        iov_len: buf.len(),
    };
    let mut ctl = CmsgBuf([0u8; 512]);
    // SAFETY: buffers outlive the call; returned SCM_RIGHTS fds are freshly owned.
    unsafe {
        let mut msg: libc::msghdr = std::mem::zeroed();
        let n = loop {
            msg.msg_iov = &mut iov;
            msg.msg_iovlen = 1;
            msg.msg_control = ctl.0.as_mut_ptr() as *mut libc::c_void;
            msg.msg_controllen = ctl.0.len() as _;
            msg.msg_flags = 0;
            let n = libc::recvmsg(fd, &mut msg, libc::MSG_DONTWAIT | libc::MSG_CMSG_CLOEXEC);
            if n < 0 {
                let e = io::Error::last_os_error();
                if e.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(e);
            }
            break n as usize;
        };
        let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
        while !cmsg.is_null() {
            if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                let hdr = libc::CMSG_LEN(0) as usize;
                let cnt = ((*cmsg).cmsg_len as usize).saturating_sub(hdr)
                    / std::mem::size_of::<libc::c_int>();
                let data = libc::CMSG_DATA(cmsg) as *const libc::c_int;
                for i in 0..cnt {
                    let raw = std::ptr::read_unaligned(data.add(i));
                    fds.push(OwnedFd::from_raw_fd(raw));
                }
            }
            cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
        }
        if msg.msg_flags & libc::MSG_CTRUNC != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ancillary data truncated",
            ));
        }
        Ok(n)
    }
}

fn send_bytes(
    conn: &Conn,
    data: &[u8],
    fds: &[RawFd],
    deadline: Option<Instant>,
    what: &'static str,
) -> Result<(), WeightHostError> {
    let mut off = 0usize;
    let mut pending: &[RawFd] = fds;
    while off < data.len() {
        match sendmsg_once(conn.fd(), &data[off..], pending) {
            Ok(0) => return Err(proto("send made no progress")),
            Ok(n) => {
                off += n;
                // The ancillary data rides with the first byte accepted.
                pending = &[];
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                wait_fd(conn.fd(), libc::POLLOUT, deadline, what)?
            }
            Err(e) => return Err(map_io(e, what)),
        }
    }
    Ok(())
}

fn send_frame(
    conn: &Conn,
    kind: MsgKind,
    payload: &[u8],
    fds: &[RawFd],
    deadline: Option<Instant>,
    what: &'static str,
) -> Result<(), WeightHostError> {
    if payload.len() > MAX_FRAME_PAYLOAD {
        return Err(WeightHostError::LimitExceeded("frame payload"));
    }
    if fds.len() > MAX_FDS_PER_MESSAGE {
        return Err(WeightHostError::LimitExceeded("fds per message"));
    }
    let bytes = frame_bytes(PROTOCOL_VERSION, kind as u32, payload, fds.len() as u32);
    send_bytes(conn, &bytes, fds, deadline, what)
}

fn recv_exact(
    conn: &Conn,
    buf: &mut [u8],
    fds: &mut Vec<OwnedFd>,
    deadline: Option<Instant>,
    what: &'static str,
) -> Result<(), WeightHostError> {
    let mut off = 0usize;
    while off < buf.len() {
        match recvmsg_once(conn.fd(), &mut buf[off..], fds) {
            Ok(0) => return Err(WeightHostError::PeerClosed(what)),
            Ok(n) => off += n,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                wait_fd(conn.fd(), libc::POLLIN, deadline, what)?
            }
            Err(e) => return Err(map_io(e, what)),
        }
        if fds.len() > MAX_FDS_PER_MESSAGE {
            return Err(WeightHostError::LimitExceeded("fds per message"));
        }
    }
    Ok(())
}

fn read_frame(
    conn: &Conn,
    deadline: Option<Instant>,
    what: &'static str,
) -> Result<Frame, WeightHostError> {
    let mut fds: Vec<OwnedFd> = Vec::new();
    let mut hdr = [0u8; HEADER_LEN];
    recv_exact(conn, &mut hdr, &mut fds, deadline, what)?;
    if hdr[0..4] != MAGIC {
        return Err(proto("bad frame magic"));
    }
    let le = |i: usize| u32::from_le_bytes([hdr[i], hdr[i + 1], hdr[i + 2], hdr[i + 3]]);
    let version = le(4);
    if version != PROTOCOL_VERSION {
        return Err(WeightHostError::UnsupportedVersion {
            got: version,
            supported: PROTOCOL_VERSION,
        });
    }
    let kind = MsgKind::from_u32(le(8))?;
    let len = le(12) as usize;
    let fd_count = le(16) as usize;
    if len > MAX_FRAME_PAYLOAD {
        return Err(WeightHostError::LimitExceeded("frame payload"));
    }
    if fd_count > MAX_FDS_PER_MESSAGE {
        return Err(WeightHostError::LimitExceeded("fds per message"));
    }
    let mut payload = vec![0u8; len];
    recv_exact(conn, &mut payload, &mut fds, deadline, what)?;
    if fds.len() != fd_count {
        return Err(WeightHostError::FdCount {
            expected: fd_count,
            got: fds.len(),
        });
    }
    Ok(Frame { kind, payload, fds })
}

fn fd_is_cloexec(fd: RawFd) -> io::Result<bool> {
    // SAFETY: F_GETFD takes no pointer argument.
    let rc = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if rc < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(rc & libc::FD_CLOEXEC != 0)
}

fn check_fd(fd: &impl AsRawFd, what: &str) -> Result<(), WeightHostError> {
    let raw = fd.as_raw_fd();
    match fd_is_cloexec(raw) {
        Ok(true) => {}
        Ok(false) => {
            return Err(WeightHostError::FdInvalid(format!(
                "{what}: descriptor is not close-on-exec"
            )))
        }
        Err(e) => {
            return Err(WeightHostError::FdInvalid(format!(
                "{what}: not a valid descriptor ({e})"
            )))
        }
    }
    // SAFETY: st is a plain out-parameter.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::fstat(raw, &mut st) };
    if rc != 0 {
        return Err(WeightHostError::FdInvalid(format!("{what}: fstat failed")));
    }
    if st.st_mode & libc::S_IFMT == libc::S_IFDIR {
        return Err(WeightHostError::FdInvalid(format!(
            "{what}: descriptor is a directory"
        )));
    }
    Ok(())
}

fn peer_credentials(fd: RawFd) -> Result<PeerCredentials, WeightHostError> {
    // SAFETY: ucred out-parameter with correct length.
    let mut uc: libc::ucred = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    let rc = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            &mut uc as *mut _ as *mut libc::c_void,
            &mut len,
        )
    };
    if rc != 0 {
        return Err(WeightHostError::Io {
            context: "SO_PEERCRED",
            source: io::Error::last_os_error(),
        });
    }
    Ok(PeerCredentials {
        pid: uc.pid as u32,
        uid: uc.uid,
        gid: uc.gid,
    })
}

fn check_private_dir(dir: &Path) -> Result<(), WeightHostError> {
    let dir = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    let md = std::fs::metadata(dir).map_err(io_ctx("stat socket directory"))?;
    if !md.is_dir() {
        return Err(WeightHostError::InsecureSocket(
            "socket parent is not a directory".into(),
        ));
    }
    if md.uid() != euid() {
        return Err(WeightHostError::InsecureSocket(
            "socket directory is not owned by the current user".into(),
        ));
    }
    if md.mode() & 0o077 != 0 {
        return Err(WeightHostError::InsecureSocket(format!(
            "socket directory mode {:o} grants group/other access (need 0700)",
            md.mode() & 0o7777
        )));
    }
    Ok(())
}

fn check_client_socket(path: &Path) -> Result<(), WeightHostError> {
    check_private_dir(path.parent().unwrap_or(Path::new(".")))?;
    let md = std::fs::symlink_metadata(path).map_err(io_ctx("stat socket"))?;
    if !md.file_type().is_socket() {
        return Err(WeightHostError::InsecureSocket(
            "path is not a unix socket".into(),
        ));
    }
    if md.uid() != euid() {
        return Err(WeightHostError::InsecureSocket(
            "socket is not owned by the current user".into(),
        ));
    }
    if md.mode() & 0o077 != 0 {
        return Err(WeightHostError::InsecureSocket(format!(
            "socket mode {:o} grants group/other access (need 0600)",
            md.mode() & 0o7777
        )));
    }
    Ok(())
}

/// One bounded, non-blocking connect attempt. A full backlog fails at once.
fn connect_unix(path: &Path) -> Result<OwnedFd, WeightHostError> {
    let bytes = path.as_os_str().as_bytes();
    // SAFETY: sockaddr_un is plain data.
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    if bytes.is_empty() || bytes.len() >= addr.sun_path.len() || bytes.contains(&0) {
        return Err(WeightHostError::InvalidArgument(
            "socket path empty, too long or contains NUL".into(),
        ));
    }
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (i, b) in bytes.iter().enumerate() {
        addr.sun_path[i] = *b as libc::c_char;
    }
    // SAFETY: plain syscalls; fd ownership transferred immediately.
    let fd = unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
            0,
        )
    };
    if fd < 0 {
        return Err(WeightHostError::Io {
            context: "socket",
            source: io::Error::last_os_error(),
        });
    }
    let owned = unsafe { OwnedFd::from_raw_fd(fd) };
    let rc = unsafe {
        libc::connect(
            fd,
            &addr as *const _ as *const libc::sockaddr,
            std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t,
        )
    };
    if rc != 0 {
        return Err(WeightHostError::Io {
            context: "connect",
            source: io::Error::last_os_error(),
        });
    }
    Ok(owned)
}

fn batch_total(fd_count: usize) -> usize {
    (fd_count + MAX_FDS_PER_MESSAGE - 1) / MAX_FDS_PER_MESSAGE
}

fn reject_reason_or_err(frame: &Frame) -> WeightHostError {
    match decode_reject(&frame.payload) {
        Ok(r) => WeightHostError::Rejected(r),
        Err(e) => e,
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Tenant-side entry point.
pub struct WeightHostClient;

impl WeightHostClient {
    /// One bounded connection attempt and handshake under a single absolute
    /// deadline. Verifies socket ownership/modes, `SO_PEERCRED` (same UID plus the
    /// optional expected host pid), echoed identities, manifest structure, the
    /// recomputed layout seal, exact FD totals and CLOEXEC. Never sends
    /// `ImportAck`; the returned lease is `Connected`/not acknowledged.
    pub fn connect(socket: &Path, hello: HostHello) -> Result<WeightHostLease, WeightHostError> {
        hello.validate()?;
        let recipe_sha = hello.recipe.sha256()?;
        let hello_bytes = encode_hello(&hello)?;
        let deadline = Some(Instant::now() + hello.options.handshake_timeout);

        check_client_socket(socket)?;
        let conn = Conn(connect_unix(socket)?);
        let peer = peer_credentials(conn.fd())?;
        verify_peer(&peer, hello.options.expected_host_pid)?;
        send_frame(
            &conn,
            MsgKind::Hello,
            &hello_bytes,
            &[],
            deadline,
            "send Hello",
        )?;

        let frame = read_frame(&conn, deadline, "read Ready")?;
        match frame.kind {
            MsgKind::Ready => {}
            MsgKind::Reject => return Err(reject_reason_or_err(&frame)),
            k => return Err(proto(format!("expected Ready, got {k:?}"))),
        }
        if !frame.fds.is_empty() {
            return Err(WeightHostError::FdCount {
                expected: 0,
                got: frame.fds.len(),
            });
        }
        let ready = decode_ready(&frame.payload)?;
        if ready.epoch.is_zero() {
            return Err(proto("zero host epoch"));
        }
        if ready.source_sha256 != hello.source_sha256 {
            return Err(WeightHostError::IdentityMismatch("source_sha256"));
        }
        if ready.loader_abi_sha256 != hello.loader_abi_sha256 {
            return Err(WeightHostError::IdentityMismatch("loader_abi_sha256"));
        }
        if ready.recipe_sha256 != recipe_sha {
            return Err(WeightHostError::IdentityMismatch("recipe_sha256"));
        }
        let seals = compute_layout_seals(
            &hello.source_sha256,
            &hello.loader_abi_sha256,
            &hello.recipe,
            &ready.manifest,
        )?;
        if seals.layout_sha256 != ready.layout_sha256 {
            return Err(WeightHostError::IdentityMismatch("layout_sha256"));
        }
        let unique = ready.manifest.unique_bytes()?;
        if unique != ready.unique_bytes {
            return Err(WeightHostError::ImportedBytesMismatch {
                expected: unique,
                got: ready.unique_bytes,
            });
        }
        if ready.fd_count as usize != ready.manifest.allocations.len() {
            return Err(WeightHostError::FdCount {
                expected: ready.manifest.allocations.len(),
                got: ready.fd_count as usize,
            });
        }
        if let Some(a) = ready
            .manifest
            .allocations
            .iter()
            .find(|a| a.device_arch != hello.card.device_arch)
        {
            let _ = a;
            return Err(WeightHostError::IdentityMismatch("device_arch"));
        }

        let fd_count = ready.fd_count as usize;
        let total = batch_total(fd_count);
        let mut fds: Vec<OwnedFd> = Vec::with_capacity(fd_count);
        for idx in 0..total {
            let f = read_frame(&conn, deadline, "read FdBatch")?;
            match f.kind {
                MsgKind::FdBatch => {}
                MsgKind::Reject => return Err(reject_reason_or_err(&f)),
                k => return Err(proto(format!("expected FdBatch, got {k:?}"))),
            }
            let b = decode_batch(&f.payload)?;
            if b.index as usize != idx || b.total as usize != total {
                return Err(proto("FdBatch numbering mismatch"));
            }
            if b.first as usize != fds.len() {
                return Err(proto("FdBatch first index mismatch"));
            }
            if b.count == 0 || b.count as usize > MAX_FDS_PER_MESSAGE {
                return Err(proto("FdBatch count out of range"));
            }
            if b.count as usize != f.fds.len() {
                return Err(WeightHostError::FdCount {
                    expected: b.count as usize,
                    got: f.fds.len(),
                });
            }
            if fds.len() + f.fds.len() > fd_count {
                return Err(WeightHostError::FdCount {
                    expected: fd_count,
                    got: fds.len() + f.fds.len(),
                });
            }
            for (i, fd) in f.fds.into_iter().enumerate() {
                check_fd(&fd, "allocation fd")?;
                let _ = i;
                fds.push(fd);
            }
        }
        if fds.len() != fd_count {
            return Err(WeightHostError::FdCount {
                expected: fd_count,
                got: fds.len(),
            });
        }

        Ok(WeightHostLease {
            conn,
            epoch: ready.epoch,
            seals,
            source_sha256: hello.source_sha256,
            loader_abi_sha256: hello.loader_abi_sha256,
            manifest: ready.manifest,
            unique_bytes: unique,
            host: peer,
            fds,
            control_timeout: hello.options.handshake_timeout,
            state: LeaseState::Received,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeaseState {
    Received,
    ImportAcked,
    Released,
}

/// Live tenant lease: manifest, received allocation FDs and the control socket.
///
/// Dropping the lease closes the FDs and the socket. It never sends `ImportAck`
/// or `Release`: the host observes plain EOF, which it must not treat as proof of
/// teardown.
pub struct WeightHostLease {
    conn: Conn,
    epoch: Epoch,
    seals: LayoutSeals,
    source_sha256: Sha256Digest,
    loader_abi_sha256: Sha256Digest,
    manifest: WeightManifest,
    unique_bytes: u64,
    host: PeerCredentials,
    fds: Vec<OwnedFd>,
    control_timeout: Duration,
    state: LeaseState,
}

impl fmt::Debug for WeightHostLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightHostLease")
            .field("epoch", &self.epoch)
            .field("layout_sha256", &self.seals.layout_sha256)
            .field("allocations", &self.manifest.allocations.len())
            .field("fds_held", &self.fds.len())
            .field("state", &self.state)
            .finish()
    }
}

impl AsFd for WeightHostLease {
    /// Control socket, for readiness polling by the owner.
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.conn.0.as_fd()
    }
}

impl WeightHostLease {
    pub fn epoch(&self) -> Epoch {
        self.epoch
    }
    pub fn manifest(&self) -> &WeightManifest {
        &self.manifest
    }
    pub fn seals(&self) -> &LayoutSeals {
        &self.seals
    }
    pub fn layout_sha256(&self) -> &Sha256Digest {
        &self.seals.layout_sha256
    }
    pub fn recipe_sha256(&self) -> &Sha256Digest {
        &self.seals.recipe_sha256
    }
    pub fn source_sha256(&self) -> &Sha256Digest {
        &self.source_sha256
    }
    pub fn loader_abi_sha256(&self) -> &Sha256Digest {
        &self.loader_abi_sha256
    }
    /// Sum of rounded physical bytes every importer must map.
    pub fn unique_bytes(&self) -> u64 {
        self.unique_bytes
    }
    /// Verified `SO_PEERCRED` of the host.
    pub fn host_credentials(&self) -> PeerCredentials {
        self.host
    }
    pub fn is_import_acked(&self) -> bool {
        self.state == LeaseState::ImportAcked
    }
    /// Received allocation FDs, index-aligned with `manifest().allocations`
    /// (empty once taken or closed).
    pub fn allocation_fds(&self) -> &[OwnedFd] {
        &self.fds
    }
    pub fn allocation_fd(&self, index: usize) -> Option<BorrowedFd<'_>> {
        self.fds.get(index).map(|f| f.as_fd())
    }
    /// Allocation, its index and FD by owner id.
    pub fn allocation(&self, owner_id: &str) -> Option<(usize, &SharedAllocation, BorrowedFd<'_>)> {
        let i = self.manifest.allocation_index(owner_id)?;
        Some((i, &self.manifest.allocations[i], self.fds.get(i)?.as_fd()))
    }
    /// Transfer FD ownership to the importer.
    pub fn take_allocation_fds(&mut self) -> Vec<OwnedFd> {
        std::mem::take(&mut self.fds)
    }
    /// Close every held allocation FD (after import, before `release`).
    pub fn close_allocation_fds(&mut self) {
        self.fds.clear();
    }

    /// Explicit acknowledgement that every allocation was *actually* imported.
    /// `imported_bytes` must equal [`Self::unique_bytes`].
    pub fn ack_import(&mut self, imported_bytes: u64) -> Result<(), WeightHostError> {
        if self.state != LeaseState::Received {
            return Err(WeightHostError::InvalidState(
                "import already acknowledged or lease released",
            ));
        }
        if imported_bytes != self.unique_bytes {
            return Err(WeightHostError::ImportedBytesMismatch {
                expected: self.unique_bytes,
                got: imported_bytes,
            });
        }
        let deadline = Some(Instant::now() + self.control_timeout);
        let payload =
            encode_epoch_layout(&self.epoch, &self.seals.layout_sha256, Some(imported_bytes));
        send_frame(
            &self.conn,
            MsgKind::ImportAck,
            &payload,
            &[],
            deadline,
            "send ImportAck",
        )?;
        self.state = LeaseState::ImportAcked;
        Ok(())
    }

    /// Non-destructive liveness check of the host control socket (`MSG_PEEK`, never
    /// consumes data): `Ok(true)` alive, `Ok(false)` host closed/hung up (stop
    /// admitting work, no transparent rebind). Unexpected host data is a protocol
    /// error.
    pub fn host_alive(&self) -> Result<bool, WeightHostError> {
        match peek_raw(&self.conn, "host control socket")? {
            PeekResult::Empty => Ok(true),
            PeekResult::Eof => Ok(false),
            PeekResult::Data => Err(proto("unexpected data from host")),
        }
    }

    /// Explicit teardown notice. Only valid after `ack_import`, after every
    /// allocation FD was closed/taken-and-dropped, and only once. No reply is
    /// expected; the socket is shut down afterwards.
    pub fn release(&mut self) -> Result<(), WeightHostError> {
        match self.state {
            LeaseState::Received => {
                return Err(WeightHostError::InvalidState("release before import ack"))
            }
            LeaseState::Released => {
                return Err(WeightHostError::InvalidState("lease already released"))
            }
            LeaseState::ImportAcked => {}
        }
        if !self.fds.is_empty() {
            return Err(WeightHostError::InvalidState(
                "allocation fds still open; close them before release",
            ));
        }
        let deadline = Some(Instant::now() + self.control_timeout);
        let payload = encode_epoch_layout(&self.epoch, &self.seals.layout_sha256, None);
        send_frame(
            &self.conn,
            MsgKind::Release,
            &payload,
            &[],
            deadline,
            "send Release",
        )?;
        self.state = LeaseState::Released;
        // SAFETY: valid socket fd; shutdown only flushes EOF to the host.
        unsafe {
            libc::shutdown(self.conn.fd(), libc::SHUT_RDWR);
        }
        Ok(())
    }
}

enum PeekResult {
    Empty,
    Eof,
    Data,
}

/// Non-consuming probe: EOF/hangup, pending data or nothing.
fn peek_raw(conn: &Conn, what: &'static str) -> Result<PeekResult, WeightHostError> {
    let mut b = [0u8; 1];
    loop {
        // SAFETY: valid one-byte buffer.
        let n = unsafe {
            libc::recv(
                conn.fd(),
                b.as_mut_ptr() as *mut libc::c_void,
                1,
                libc::MSG_PEEK | libc::MSG_DONTWAIT,
            )
        };
        if n == 0 {
            return Ok(PeekResult::Eof);
        }
        if n > 0 {
            return Ok(PeekResult::Data);
        }
        let e = io::Error::last_os_error();
        match e.kind() {
            io::ErrorKind::WouldBlock => return Ok(PeekResult::Empty),
            io::ErrorKind::Interrupted => continue,
            io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::BrokenPipe => return Ok(PeekResult::Eof),
            _ => return Err(map_io(e, what)),
        }
    }
}

/// Bootstrap switch. `HIPFIRE_WEIGHT_HOST` unset or `0` -> `Ok(None)` (the kill
/// switch wins; the socket variable is ignored). `1` requires an absolute
/// `HIPFIRE_WEIGHT_HOST_SOCKET` and returns it. Any other value fails closed.
pub fn weight_host_socket_from_env() -> Result<Option<PathBuf>, WeightHostError> {
    let mode = match std::env::var_os("HIPFIRE_WEIGHT_HOST") {
        None => return Ok(None),
        Some(v) => v,
    };
    match mode.as_bytes() {
        b"0" => Ok(None),
        b"1" => {
            let sock = std::env::var_os("HIPFIRE_WEIGHT_HOST_SOCKET").ok_or_else(|| {
                WeightHostError::InvalidArgument(
                    "HIPFIRE_WEIGHT_HOST=1 requires HIPFIRE_WEIGHT_HOST_SOCKET".into(),
                )
            })?;
            let path = PathBuf::from(sock);
            if !path.is_absolute() || path.as_os_str().as_bytes().contains(&0) {
                return Err(WeightHostError::InvalidArgument(
                    "HIPFIRE_WEIGHT_HOST_SOCKET must be an absolute path".into(),
                ));
            }
            Ok(Some(path))
        }
        _ => Err(WeightHostError::InvalidArgument(
            "HIPFIRE_WEIGHT_HOST must be unset, 0 or 1".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

/// Everything a host serves: identities, sealed manifest and its own handles
/// (one FD per manifest allocation, index-aligned).
pub struct WeightPublication {
    epoch: Epoch,
    card: CardIdentity,
    source_sha256: Sha256Digest,
    loader_abi_sha256: Sha256Digest,
    recipe: LayoutRecipe,
    manifest: WeightManifest,
    seals: LayoutSeals,
    unique_bytes: u64,
    ready_payload: Vec<u8>,
    fds: Vec<OwnedFd>,
}

impl fmt::Debug for WeightPublication {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightPublication")
            .field("epoch", &self.epoch)
            .field("layout_sha256", &self.seals.layout_sha256)
            .field("allocations", &self.manifest.allocations.len())
            .finish()
    }
}

impl WeightPublication {
    /// Validate and seal a publication. `fds[i]` is the exportable handle for
    /// `manifest.allocations[i]` and must be CLOEXEC.
    pub fn new(
        epoch: Epoch,
        card: CardIdentity,
        source_sha256: Sha256Digest,
        loader_abi_sha256: Sha256Digest,
        recipe: LayoutRecipe,
        manifest: WeightManifest,
        fds: Vec<OwnedFd>,
    ) -> Result<Self, WeightHostError> {
        if epoch.is_zero() {
            return Err(WeightHostError::InvalidArgument(
                "epoch must be non-zero".into(),
            ));
        }
        if card.uuid.is_empty() || card.pci_bus_id.is_empty() || card.device_arch.is_empty() {
            return Err(WeightHostError::InvalidArgument(
                "card identity fields must not be empty".into(),
            ));
        }
        manifest.validate()?;
        if fds.len() != manifest.allocations.len() {
            return Err(WeightHostError::FdCount {
                expected: manifest.allocations.len(),
                got: fds.len(),
            });
        }
        for (i, fd) in fds.iter().enumerate() {
            check_fd(fd, &format!("publication fd {i}"))?;
        }
        if manifest
            .allocations
            .iter()
            .any(|a| a.device_arch != card.device_arch)
        {
            return Err(WeightHostError::IdentityMismatch("device_arch"));
        }
        let seals = compute_layout_seals(&source_sha256, &loader_abi_sha256, &recipe, &manifest)?;
        let unique_bytes = manifest.unique_bytes()?;
        let ready_payload = encode_ready(&ReadyMsg {
            epoch,
            source_sha256,
            loader_abi_sha256,
            recipe_sha256: seals.recipe_sha256,
            layout_sha256: seals.layout_sha256,
            unique_bytes,
            fd_count: fds.len() as u32,
            manifest: manifest.clone(),
        })?;
        if ready_payload.len() > MAX_FRAME_PAYLOAD {
            return Err(WeightHostError::LimitExceeded("Ready payload"));
        }
        Ok(Self {
            epoch,
            card,
            source_sha256,
            loader_abi_sha256,
            recipe,
            manifest,
            seals,
            unique_bytes,
            ready_payload,
            fds,
        })
    }

    pub fn epoch(&self) -> Epoch {
        self.epoch
    }
    pub fn card(&self) -> &CardIdentity {
        &self.card
    }
    pub fn manifest(&self) -> &WeightManifest {
        &self.manifest
    }
    pub fn seals(&self) -> &LayoutSeals {
        &self.seals
    }
    pub fn recipe(&self) -> &LayoutRecipe {
        &self.recipe
    }
    pub fn unique_bytes(&self) -> u64 {
        self.unique_bytes
    }
    pub fn fd(&self, index: usize) -> Option<BorrowedFd<'_>> {
        self.fds.get(index).map(|f| f.as_fd())
    }
}

/// Host-side connection options.
#[derive(Clone, Copy, Debug)]
pub struct ServeOptions {
    /// One absolute deadline per connection from accept through the last FD batch.
    pub handshake_timeout: Duration,
    /// Required tenant pid via `SO_PEERCRED`, when known.
    pub expected_peer_pid: Option<u32>,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            handshake_timeout: DEFAULT_HANDSHAKE_TIMEOUT,
            expected_peer_pid: None,
        }
    }
}

/// Handshake-time view of a client for the admission callback.
pub struct AdmissionRequest<'a> {
    pub peer: &'a PeerCredentials,
    pub hello: &'a HostHello,
    pub epoch: Epoch,
    pub layout_sha256: Sha256Digest,
    pub unique_bytes: u64,
    pub fd_count: usize,
}

/// Listening half. The caller owns the filesystem (private directory, 0600
/// socket, atomic publication, generation lifecycle) and hands over a bound
/// listener through [`WeightHostServer::from_listener`]; [`WeightHostServer::bind`]
/// is only a convenience that never overwrites an existing path and removes its
/// own socket inode on drop.
pub struct WeightHostServer {
    listener: UnixListener,
    publication: Arc<WeightPublication>,
    options: ServeOptions,
    unlink: Option<(PathBuf, (u64, u64))>,
}

impl fmt::Debug for WeightHostServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightHostServer")
            .field("unlink", &self.unlink)
            .finish()
    }
}

impl WeightHostServer {
    /// Serve on an already bound/listening socket owned by the caller. The
    /// listener is switched to non-blocking mode; nothing on the filesystem is
    /// touched or removed.
    pub fn from_listener(
        listener: UnixListener,
        publication: Arc<WeightPublication>,
        options: ServeOptions,
    ) -> Result<Self, WeightHostError> {
        if options.handshake_timeout.is_zero() {
            return Err(WeightHostError::InvalidArgument(
                "handshake timeout must be non-zero".into(),
            ));
        }
        listener
            .set_nonblocking(true)
            .map_err(io_ctx("set nonblocking"))?;
        Ok(Self {
            listener,
            publication,
            options,
            unlink: None,
        })
    }

    /// Convenience bind: the parent directory must already be private (0700, same
    /// owner); the socket is chmod 0600. Fails if `path` exists.
    pub fn bind(
        path: &Path,
        publication: Arc<WeightPublication>,
        options: ServeOptions,
    ) -> Result<Self, WeightHostError> {
        check_private_dir(path.parent().unwrap_or(Path::new(".")))?;
        if std::fs::symlink_metadata(path).is_ok() {
            return Err(WeightHostError::InsecureSocket(
                "socket path already exists; refusing to overwrite".into(),
            ));
        }
        let listener = UnixListener::bind(path).map_err(io_ctx("bind"))?;
        let cleanup = |e: WeightHostError| {
            let _ = std::fs::remove_file(path);
            e
        };
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(io_ctx("chmod socket"))
            .map_err(cleanup)?;
        let md = std::fs::symlink_metadata(path)
            .map_err(io_ctx("stat socket"))
            .map_err(cleanup)?;
        let mut s = Self::from_listener(listener, publication, options).map_err(cleanup)?;
        s.unlink = Some((path.to_path_buf(), (md.dev(), md.ino())));
        Ok(s)
    }

    pub fn publication(&self) -> &Arc<WeightPublication> {
        &self.publication
    }

    /// Accept one client within `wait` (`None` blocks) and run
    /// [`HostConnection::from_stream`]. A closed/failed listener is an `Err`.
    pub fn accept(&self, wait: Option<Duration>) -> Result<HostConnection, WeightHostError> {
        let wait_deadline = wait.map(|w| Instant::now() + w);
        let stream = loop {
            match self.listener.accept() {
                Ok((s, _)) => break s,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => wait_fd(
                    self.listener.as_raw_fd(),
                    libc::POLLIN,
                    wait_deadline,
                    "accept",
                )?,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    return Err(WeightHostError::Io {
                        context: "accept",
                        source: e,
                    })
                }
            }
        };
        HostConnection::from_stream(stream, self.publication.clone(), self.options)
    }
}

impl Drop for WeightHostServer {
    fn drop(&mut self) {
        if let Some((path, ident)) = &self.unlink {
            if let Ok(md) = std::fs::symlink_metadata(path) {
                if (md.dev(), md.ino()) == *ident {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
    }
}

/// An accepted, credential-checked, identity-matched client that has not been
/// admitted and has received nothing.
pub struct HostConnection {
    conn: Conn,
    peer: PeerCredentials,
    hello: HostHello,
    publication: Arc<WeightPublication>,
    deadline: Option<Instant>,
}

impl fmt::Debug for HostConnection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostConnection")
            .field("peer", &self.peer)
            .finish()
    }
}

impl HostConnection {
    /// First half of the host handshake on a stream the caller accepted, under one
    /// absolute deadline (`options.handshake_timeout`, started here): `SO_PEERCRED`
    /// (same UID, optional expected pid), `Hello` framing, then identity
    /// comparison against `publication`. A mismatch sends `Reject`; no FD is ever
    /// sent before [`HostConnection::admit`] and `HostSession::send_allocations`.
    pub fn from_stream(
        stream: UnixStream,
        publication: Arc<WeightPublication>,
        options: ServeOptions,
    ) -> Result<Self, WeightHostError> {
        if options.handshake_timeout.is_zero() {
            return Err(WeightHostError::InvalidArgument(
                "handshake timeout must be non-zero".into(),
            ));
        }
        let deadline = Some(Instant::now() + options.handshake_timeout);
        let conn = Conn::from_stream(stream)?;
        let peer = peer_credentials(conn.fd())?;
        verify_peer(&peer, options.expected_peer_pid)?;
        let frame = read_frame(&conn, deadline, "read Hello")?;
        if frame.kind != MsgKind::Hello {
            return Err(proto(format!("expected Hello, got {:?}", frame.kind)));
        }
        if !frame.fds.is_empty() {
            return Err(WeightHostError::FdCount {
                expected: 0,
                got: frame.fds.len(),
            });
        }
        let hello = decode_hello(&frame.payload)?;
        let p = &*publication;
        let mismatch: Option<&'static str> = if hello.card.uuid != p.card.uuid {
            Some("card.uuid")
        } else if hello.card.pci_bus_id != p.card.pci_bus_id {
            Some("card.pci_bus_id")
        } else if hello.card.device_arch != p.card.device_arch {
            Some("card.device_arch")
        } else if hello.source_sha256 != p.source_sha256 {
            Some("source_sha256")
        } else if hello.loader_abi_sha256 != p.loader_abi_sha256 {
            Some("loader_abi_sha256")
        } else if hello.recipe.sha256()? != p.seals.recipe_sha256 {
            Some("recipe_sha256")
        } else {
            None
        };
        if let Some(field) = mismatch {
            let _ = send_frame(
                &conn,
                MsgKind::Reject,
                &encode_reject(&format!("identity mismatch: {field}")),
                &[],
                deadline,
                "send Reject",
            );
            return Err(WeightHostError::IdentityMismatch(field));
        }
        Ok(Self {
            conn,
            peer,
            hello,
            publication,
            deadline,
        })
    }

    pub fn peer(&self) -> &PeerCredentials {
        &self.peer
    }
    pub fn hello(&self) -> &HostHello {
        &self.hello
    }

    /// Refuse the client with a reason (best effort) and close.
    pub fn reject(self, reason: &str) {
        let _ = send_frame(
            &self.conn,
            MsgKind::Reject,
            &encode_reject(reason),
            &[],
            self.deadline,
            "send Reject",
        );
    }

    /// Run the authenticated admission callback. `Ok(())` registers a pending
    /// lease: the returned [`HostSession`] exists *before* any FD is sent and
    /// must be counted by the host as soon as it is returned. `Err(reason)` sends
    /// `Reject` and fails.
    pub fn admit<F>(self, admit: F) -> Result<HostSession, WeightHostError>
    where
        F: FnOnce(&AdmissionRequest<'_>) -> Result<(), String>,
    {
        let req = AdmissionRequest {
            peer: &self.peer,
            hello: &self.hello,
            epoch: self.publication.epoch,
            layout_sha256: self.publication.seals.layout_sha256,
            unique_bytes: self.publication.unique_bytes,
            fd_count: self.publication.fds.len(),
        };
        match admit(&req) {
            Ok(()) => Ok(HostSession {
                conn: self.conn,
                peer: self.peer,
                hello: self.hello,
                publication: self.publication,
                deadline: self.deadline,
                fds_exposed: 0,
                state: HostSessionState::Admitted,
            }),
            Err(reason) => {
                let _ = send_frame(
                    &self.conn,
                    MsgKind::Reject,
                    &encode_reject(&reason),
                    &[],
                    self.deadline,
                    "send Reject",
                );
                Err(WeightHostError::AdmissionDenied(reason))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostSessionState {
    /// Admitted; nothing sent. Counts as a pending lease.
    Admitted,
    /// Ready and every FD batch sent; awaiting `ImportAck`.
    FdsSent,
    ImportAcked,
    Released,
    /// Any error, timeout or EOF. The host must keep the generation charged until
    /// the tenant process is independently proven gone.
    Failed,
}

/// Result of a non-blocking release poll. `PeerClosed` is EOF *without* a
/// `Release` message and never implies teardown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleasePoll {
    Pending,
    Released,
    PeerClosed,
}

/// Time allowed to read the rest of a frame once its first byte is visible.
const POLL_FRAME_TIMEOUT: Duration = Duration::from_secs(1);

/// Host half of one tenant lease (`Send`: may move to a connection thread).
pub struct HostSession {
    conn: Conn,
    peer: PeerCredentials,
    hello: HostHello,
    publication: Arc<WeightPublication>,
    deadline: Option<Instant>,
    fds_exposed: usize,
    state: HostSessionState,
}

impl fmt::Debug for HostSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostSession")
            .field("peer", &self.peer)
            .field("state", &self.state)
            .field("fds_exposed", &self.fds_exposed)
            .finish()
    }
}

impl AsFd for HostSession {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.conn.0.as_fd()
    }
}

impl HostSession {
    pub fn peer(&self) -> &PeerCredentials {
        &self.peer
    }
    pub fn hello(&self) -> &HostHello {
        &self.hello
    }
    pub fn epoch(&self) -> Epoch {
        self.publication.epoch
    }
    pub fn layout_sha256(&self) -> &Sha256Digest {
        &self.publication.seals.layout_sha256
    }
    pub fn unique_bytes(&self) -> u64 {
        self.publication.unique_bytes
    }
    pub fn fd_count(&self) -> usize {
        self.publication.fds.len()
    }
    pub fn state(&self) -> HostSessionState {
        self.state
    }
    /// Number of FDs whose send was *attempted*; conservative upper bound of what
    /// the tenant may hold even when a send failed midway.
    pub fn fds_exposed(&self) -> usize {
        self.fds_exposed
    }

    fn fail<T>(&mut self, e: WeightHostError) -> Result<T, WeightHostError> {
        self.state = HostSessionState::Failed;
        Err(e)
    }

    /// Send `Ready` and every numbered FD batch (at most 32 per message) under the
    /// absolute handshake deadline.
    pub fn send_allocations(&mut self) -> Result<(), WeightHostError> {
        if self.state != HostSessionState::Admitted {
            return Err(WeightHostError::InvalidState(
                "allocations already sent or session failed",
            ));
        }
        let p = self.publication.clone();
        if let Err(e) = send_frame(
            &self.conn,
            MsgKind::Ready,
            &p.ready_payload,
            &[],
            self.deadline,
            "send Ready",
        ) {
            return self.fail(e);
        }
        let total = batch_total(p.fds.len());
        for idx in 0..total {
            let first = idx * MAX_FDS_PER_MESSAGE;
            let end = (first + MAX_FDS_PER_MESSAGE).min(p.fds.len());
            let raw: Vec<RawFd> = p.fds[first..end].iter().map(|f| f.as_raw_fd()).collect();
            let payload = encode_batch(&BatchMsg {
                index: idx as u32,
                total: total as u32,
                first: first as u32,
                count: raw.len() as u32,
            });
            self.fds_exposed += raw.len();
            if let Err(e) = send_frame(
                &self.conn,
                MsgKind::FdBatch,
                &payload,
                &raw,
                self.deadline,
                "send FdBatch",
            ) {
                return self.fail(e);
            }
        }
        self.state = HostSessionState::FdsSent;
        Ok(())
    }

    fn read_control(
        &mut self,
        timeout: Option<Duration>,
        what: &'static str,
    ) -> Result<Frame, WeightHostError> {
        let deadline = timeout.map(|t| Instant::now() + t);
        match read_frame(&self.conn, deadline, what) {
            Ok(f) => {
                if !f.fds.is_empty() {
                    let got = f.fds.len();
                    return self.fail(WeightHostError::FdCount { expected: 0, got });
                }
                Ok(f)
            }
            Err(e) => self.fail(e),
        }
    }

    /// Wait for the tenant's explicit `ImportAck` (`None` blocks). EOF or timeout
    /// fails the session; neither proves the tenant released anything.
    pub fn wait_import_ack(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<ImportAck, WeightHostError> {
        if self.state != HostSessionState::FdsSent {
            return Err(WeightHostError::InvalidState("not awaiting ImportAck"));
        }
        let f = self.read_control(timeout, "read ImportAck")?;
        if f.kind != MsgKind::ImportAck {
            return self.fail(proto(format!("expected ImportAck, got {:?}", f.kind)));
        }
        let (epoch, layout, bytes) = match decode_epoch_layout(&f.payload, true) {
            Ok(v) => v,
            Err(e) => return self.fail(e),
        };
        let p = &self.publication;
        if epoch != p.epoch {
            return self.fail(WeightHostError::IdentityMismatch("epoch"));
        }
        if layout != p.seals.layout_sha256 {
            return self.fail(WeightHostError::IdentityMismatch("layout_sha256"));
        }
        if bytes != p.unique_bytes {
            let expected = p.unique_bytes;
            return self.fail(WeightHostError::ImportedBytesMismatch {
                expected,
                got: bytes,
            });
        }
        self.state = HostSessionState::ImportAcked;
        Ok(ImportAck {
            epoch,
            layout_sha256: layout,
            imported_bytes: bytes,
        })
    }

    /// Wait for the tenant's explicit `Release` (`None` blocks). EOF before
    /// `Release` is `PeerClosed` and fails the session: the host must reconcile
    /// pid/starttime before considering the lease gone.
    pub fn wait_release(&mut self, timeout: Option<Duration>) -> Result<(), WeightHostError> {
        if self.state != HostSessionState::ImportAcked {
            return Err(WeightHostError::InvalidState("not awaiting Release"));
        }
        let f = self.read_control(timeout, "read Release")?;
        if f.kind != MsgKind::Release {
            return self.fail(proto(format!("expected Release, got {:?}", f.kind)));
        }
        let (epoch, layout, _) = match decode_epoch_layout(&f.payload, false) {
            Ok(v) => v,
            Err(e) => return self.fail(e),
        };
        if epoch != self.publication.epoch || layout != self.publication.seals.layout_sha256 {
            return self.fail(WeightHostError::IdentityMismatch("release identity"));
        }
        self.state = HostSessionState::Released;
        Ok(())
    }

    /// Non-blocking release poll for a host event loop. `Pending`: nothing yet.
    /// `Released`: explicit `Release` received. `PeerClosed`: EOF without
    /// `Release` (session becomes `Failed`; this does not prove teardown).
    pub fn poll_release(&mut self) -> Result<ReleasePoll, WeightHostError> {
        match self.state {
            HostSessionState::Released => return Ok(ReleasePoll::Released),
            HostSessionState::ImportAcked => {}
            _ => return Err(WeightHostError::InvalidState("not awaiting Release")),
        }
        match peek_raw(&self.conn, "tenant control socket") {
            Ok(PeekResult::Empty) => Ok(ReleasePoll::Pending),
            Ok(PeekResult::Eof) => {
                self.state = HostSessionState::Failed;
                Ok(ReleasePoll::PeerClosed)
            }
            Ok(PeekResult::Data) => {
                self.wait_release(Some(POLL_FRAME_TIMEOUT))?;
                Ok(ReleasePoll::Released)
            }
            Err(e) => self.fail(e),
        }
    }

    /// Begin draining: shut down only the write half so the tenant's
    /// `host_alive()` turns false while `wait_release`/`poll_release` can still
    /// read the tenant's `Release`.
    pub fn begin_drain(&self) -> Result<(), WeightHostError> {
        // SAFETY: valid socket fd.
        let rc = unsafe { libc::shutdown(self.conn.fd(), libc::SHUT_WR) };
        if rc != 0 {
            let e = io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::ENOTCONN) {
                return Ok(());
            }
            return Err(WeightHostError::Io {
                context: "shutdown(write)",
                source: e,
            });
        }
        Ok(())
    }

    /// Non-destructive check whether the tenant closed its socket (EOF). `Ok(true)`
    /// does not prove teardown; pending data reports `false`.
    pub fn peer_closed(&self) -> Result<bool, WeightHostError> {
        Ok(matches!(
            peek_raw(&self.conn, "tenant control socket")?,
            PeekResult::Eof
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests (pure CPU; real memfd/pipe descriptors)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    /// Rounded physical size of the `wide_manifest` allocations (4000 + 64 KiB pad).
    const WIDE_PHYS: u64 = 69_632;
    use super::*;
    use std::os::unix::fs::FileExt;
    use std::sync::Mutex;
    use std::thread;

    // Descriptor-count assertions need no concurrent fd churn in the process.
    static FD_LOCK: Mutex<()> = Mutex::new(());

    fn fd_lock() -> std::sync::MutexGuard<'static, ()> {
        FD_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    const FD_ISOLATED_ENV: &str = "HIPFIRE_WH_FD_ISOLATED_CHILD";

    /// Process-wide descriptor counts are only meaningful when nothing else in
    /// the process opens or closes fds. The lib test binary runs unrelated modules
    /// in parallel threads, so every test asserting an exact `/proc/self/fd` count
    /// re-executes itself alone (`--exact --test-threads=1`) in a child process.
    /// Returns `true` in the parent once the child passed (caller returns early),
    /// `false` in the child (caller runs the body and its leak assertions).
    fn fd_isolated(probe_type_name: &str) -> bool {
        if std::env::var_os(FD_ISOLATED_ENV).is_some() {
            return false;
        }
        let path = probe_type_name
            .split_once("::")
            .map(|(_, rest)| rest)
            .and_then(|rest| rest.strip_suffix("::probe"))
            .unwrap_or_else(|| panic!("unexpected probe type name {probe_type_name}"));
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new(exe)
            .args([path, "--exact", "--test-threads=1", "--nocapture"])
            .env(FD_ISOLATED_ENV, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success() && stdout.contains("test result: ok. 1 passed"),
            "isolated child for {path} failed ({}):\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            out.status
        );
        true
    }

    /// First statement of every exact-fd-count test.
    macro_rules! isolate_fd {
        () => {{
            fn probe() {}
            fn name_of<T>(_: &T) -> &'static str {
                std::any::type_name::<T>()
            }
            if fd_isolated(name_of(&probe)) {
                return;
            }
        }};
    }

    /// Temp dir pinned to 0700: the socket rules reject any group/other access and
    /// `tempfile` follows the process umask on some hosts.
    fn private_tempdir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        dir
    }

    fn open_fd_count() -> usize {
        let names: Vec<_> = std::fs::read_dir("/proc/self/fd")
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        names.len()
    }

    fn memfd(content: &[u8]) -> OwnedFd {
        let name = std::ffi::CString::new("hfwh-test").unwrap();
        // SAFETY: valid NUL-terminated name.
        let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0, "memfd_create failed");
        let owned = unsafe { OwnedFd::from_raw_fd(fd) };
        let f = File::from(owned.try_clone().unwrap());
        f.write_all_at(content, 0).unwrap();
        owned
    }

    fn pipe_read_end() -> (OwnedFd, OwnedFd) {
        let mut fds = [0 as libc::c_int; 2];
        // SAFETY: fds has room for two ints.
        let rc = unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) };
        assert_eq!(rc, 0);
        unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) }
    }

    fn card() -> CardIdentity {
        CardIdentity {
            uuid: "GPU-test-0".into(),
            pci_bus_id: "0000:03:00.0".into(),
            device_arch: "gfx1201".into(),
        }
    }

    fn alloc(id: &str, phys: u64, logical: u64) -> SharedAllocation {
        SharedAllocation {
            owner_id: id.into(),
            physical_bytes: phys,
            logical_bytes: logical,
            granularity: 4096,
            device_arch: "gfx1201".into(),
            representation_tag: "hfq-final/v1".into(),
        }
    }

    fn view(role: &str, owner: &str, off: u64, ext: u64) -> WeightView {
        WeightView {
            role: role.into(),
            owner_id: owner.into(),
            byte_offset: off,
            byte_extent: ext,
            dtype: "u8".into(),
            shape: vec![ext],
            strides: vec![],
            matrix: None,
            quant: None,
            companions: BTreeMap::new(),
            alias_of: None,
            metadata: CanonicalFields::new(),
        }
    }

    fn base_manifest() -> WeightManifest {
        let mut embed = view("embed", "a.embed", 0, 8000);
        embed.quant = Some(QuantDesc {
            format: "mq4".into(),
            group_size: 256,
            block_elems: 256,
            block_bytes: 136,
            packing: "mq4v2".into(),
        });
        embed
            .metadata
            .insert("lut".into(), FieldValue::f32_list(&[0.0, -0.5, 0.25, 1.0]));
        let mut q = view("layer0.q", "b.layer0", 0, 8000);
        q.matrix = Some(MatrixDesc {
            m: 100,
            k: 160,
            row_stride_bytes: 80,
        });
        q.companions
            .insert("scales".into(), "layer0.q_scales".into());
        let scales = view("layer0.q_scales", "b.layer0", 8192, 4096);
        let mut head = view("lm_head", "a.embed", 0, 8000);
        head.alias_of = Some("embed".into());
        let mut config = CanonicalFields::new();
        config.insert("layers".into(), FieldValue::U64(2));
        config.insert("awq_rms_fdiv".into(), FieldValue::Bool(true));
        config.insert("z_fold".into(), FieldValue::Bool(false));
        WeightManifest {
            arch: "test-dense".into(),
            config,
            dispatch_metadata: b"dispatch-v1".to_vec(),
            allocations: vec![
                alloc("a.embed", 77_824, 8000),
                alloc("b.layer0", 81_920, 16000),
            ],
            views: vec![embed, q, scales, head],
        }
    }

    fn base_recipe() -> LayoutRecipe {
        LayoutRecipe::new("test/recipe")
            .with("placement", FieldValue::str("single"))
            .with("pp", FieldValue::U64(1))
            .with("expert_layers", FieldValue::U64List(vec![]))
    }

    fn digests() -> (Sha256Digest, Sha256Digest) {
        (Sha256Digest::of(b"model"), Sha256Digest::of(b"abi"))
    }

    fn seal_of(m: &WeightManifest) -> Sha256Digest {
        let (a, b) = digests();
        compute_layout_seals(&a, &b, &base_recipe(), m)
            .unwrap()
            .layout_sha256
    }

    // ---- identity ----

    #[test]
    fn sha256_known_answer() {
        assert_eq!(
            Sha256Digest::of(b"abc").to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let d = Sha256Digest::of(b"x");
        assert_eq!(Sha256Digest::from_hex(&d.to_hex()).unwrap(), d);
        assert!(Sha256Digest::from_hex("zz").is_err());
        let (h, n) = sha256_reader(&b"abc"[..]).unwrap();
        assert_eq!((h, n), (Sha256Digest::of(b"abc"), 3));
    }

    #[test]
    fn canonical_recipe_bytes_exact_and_order_independent() {
        let r = LayoutRecipe::new("a").with("k", FieldValue::U64(1));
        assert_eq!(
            r.encode_canonical().unwrap(),
            vec![1, 0, 0, 0, b'a', 1, 0, 0, 0, 1, 0, 0, 0, b'k', 2, 1, 0, 0, 0, 0, 0, 0, 0]
        );
        let x = LayoutRecipe::new("s")
            .with("b", FieldValue::U64(2))
            .with("a", FieldValue::Bool(true));
        let y = LayoutRecipe::new("s")
            .with("a", FieldValue::Bool(true))
            .with("b", FieldValue::U64(2));
        assert_eq!(x.sha256().unwrap(), y.sha256().unwrap());
        assert_ne!(
            x.sha256().unwrap(),
            x.clone().with("c", FieldValue::Absent).sha256().unwrap()
        );
        assert!(LayoutRecipe::new("").sha256().is_err());
    }

    #[test]
    fn floats_are_bit_exact() {
        let a = LayoutRecipe::new("s").with("v", FieldValue::f32(0.0));
        let b = LayoutRecipe::new("s").with("v", FieldValue::f32(-0.0));
        assert_ne!(a.sha256().unwrap(), b.sha256().unwrap());
        let n1 = FieldValue::F32Bits(0x7fc0_0000);
        let n2 = FieldValue::F32Bits(0x7fc0_0001);
        assert_ne!(n1, n2);
        let r = LayoutRecipe::new("s")
            .with("n", n2.clone())
            .with("l", FieldValue::f32_list(&[f32::NAN, -0.0]));
        let bytes = r.encode_canonical().unwrap();
        let mut d = Dec::new(&bytes);
        assert_eq!(LayoutRecipe::decode(&mut d).unwrap(), r);
        assert_eq!(FieldValue::f32(1.5).as_f32(), Some(1.5));
    }

    #[test]
    fn nested_values_roundtrip_and_depth_limit() {
        let mut inner = CanonicalFields::new();
        inner.insert(
            "x".into(),
            FieldValue::List(vec![FieldValue::I64(-3), FieldValue::Bytes(vec![1, 2, 3])]),
        );
        inner.insert("d".into(), FieldValue::Digest(Sha256Digest::of(b"q")));
        let r = LayoutRecipe::new("s")
            .with("m", FieldValue::Map(inner))
            .with("f", FieldValue::f64(2.5));
        let bytes = r.encode_canonical().unwrap();
        let mut d = Dec::new(&bytes);
        assert_eq!(LayoutRecipe::decode(&mut d).unwrap(), r);
        d.finish().unwrap();

        let mut v = FieldValue::U64(1);
        for _ in 0..(MAX_FIELD_NESTING + 2) {
            v = FieldValue::List(vec![v]);
        }
        assert!(matches!(
            LayoutRecipe::new("s").with("deep", v).encode_canonical(),
            Err(WeightHostError::LimitExceeded(_))
        ));
    }

    #[test]
    fn decoder_rejects_noncanonical_maps() {
        // two keys out of order
        let mut e = Enc::new();
        e.str("s").unwrap();
        e.u32(2);
        e.str("b").unwrap();
        e.u8(2);
        e.u64(1);
        e.str("a").unwrap();
        e.u8(2);
        e.u64(1);
        assert!(LayoutRecipe::decode(&mut Dec::new(&e.buf)).is_err());
        // duplicate key
        let mut e = Enc::new();
        e.str("s").unwrap();
        e.u32(2);
        for _ in 0..2 {
            e.str("a").unwrap();
            e.u8(2);
            e.u64(1);
        }
        assert!(LayoutRecipe::decode(&mut Dec::new(&e.buf)).is_err());
    }

    #[test]
    fn loader_abi_ignores_provenance_and_tracks_closure() {
        let mut a = LoaderAbi::new("abi/v1").with("fwht_seed", FieldValue::U64(7));
        a.add_component_bytes("model_load.rs", b"fn load() {}");
        a.add_component_bytes("weight_backend.rs", b"fn backend() {}");
        let mut b = LoaderAbi::new("abi/v1").with("fwht_seed", FieldValue::U64(7));
        b.add_component_bytes("weight_backend.rs", b"fn backend() {}");
        b.add_component_bytes("model_load.rs", b"fn load() {}");
        assert_eq!(a.sha256().unwrap(), b.sha256().unwrap());
        let mut c = b.clone();
        c.add_component_bytes("model_load.rs", b"fn load() { changed }");
        assert_ne!(a.sha256().unwrap(), c.sha256().unwrap());
        let d = LoaderAbi::new("abi/v1").with("fwht_seed", FieldValue::U64(8));
        assert_ne!(
            LoaderAbi::new("abi/v1").sha256().unwrap(),
            d.sha256().unwrap()
        );
        // provenance has no representation in the ABI or the layout seal
        let h1 = BuildProvenance {
            binary_sha256: Some(Sha256Digest::of(b"A")),
            ..Default::default()
        };
        let h2 = BuildProvenance {
            binary_sha256: Some(Sha256Digest::of(b"B")),
            kernel_image_sha256: Some(Sha256Digest::of(b"K")),
            label: "x".into(),
        };
        assert_ne!(h1, h2);
        assert_eq!(seal_of(&base_manifest()), seal_of(&base_manifest()));
    }

    #[test]
    fn model_content_order_and_content_sensitive() {
        let s = |role: &str, data: &[u8]| SourceDigest {
            role: role.into(),
            len: data.len() as u64,
            sha256: Sha256Digest::of(data),
        };
        let base = s("base", b"AAAA");
        let ov = s("overlay", b"BB");
        let ab = model_content_sha256(&[base.clone(), ov.clone()]).unwrap();
        assert_ne!(
            ab,
            model_content_sha256(&[ov.clone(), base.clone()]).unwrap()
        );
        assert_ne!(ab, model_content_sha256(&[base.clone()]).unwrap());
        assert_ne!(
            ab,
            model_content_sha256(&[base, s("overlay", b"BC")]).unwrap()
        );
        assert!(model_content_sha256(&[]).is_err());
    }

    #[test]
    fn layout_seal_mutation_matrix_rejects_each_named_change() {
        let base = seal_of(&base_manifest());
        let check = |label: &str, f: &dyn Fn(&mut WeightManifest)| {
            let mut m = base_manifest();
            f(&mut m);
            m.validate()
                .unwrap_or_else(|e| panic!("{label}: mutation invalid: {e}"));
            assert_ne!(seal_of(&m), base, "{label} must change the layout seal");
        };
        check("quant packing", &|m| {
            m.views[0].quant.as_mut().unwrap().packing = "mq4v3".into()
        });
        check("quant group", &|m| {
            m.views[0].quant.as_mut().unwrap().group_size = 128
        });
        check("awq mode", &|m| {
            m.config
                .insert("awq_rms_fdiv".into(), FieldValue::Bool(false));
        });
        check("z fold", &|m| {
            m.config.insert("z_fold".into(), FieldValue::Bool(true));
        });
        check("lut payload", &|m| {
            m.views[0].metadata.insert(
                "lut".into(),
                FieldValue::f32_list(&[0.0, -0.5, 0.25, 1.0000001]),
            );
        });
        check("source tensor dtype", &|m| m.views[1].dtype = "f16".into());
        check("alias offset topology", &|m| {
            m.views[3].alias_of = None;
        });
        check("view offset", &|m| m.views[2].byte_offset = 8704);
        check("architecture", &|m| m.arch = "other".into());
        check("device arch", &|m| {
            for a in &mut m.allocations {
                a.device_arch = "gfx1100".into();
            }
        });
        check("representation tag", &|m| {
            m.allocations[0].representation_tag = "hfq-final/v2".into()
        });
        check("granularity", &|m| {
            m.allocations[1].granularity = 8192;
        });
        check("padding extent", &|m| m.allocations[0].logical_bytes = 8100);
        check("dispatch metadata", &|m| {
            m.dispatch_metadata = b"dispatch-v2".to_vec()
        });
        check("dispatch metadata cleared", &|m| {
            m.dispatch_metadata.clear()
        });
        check("expert placement config", &|m| {
            m.config
                .insert("expert_layers".into(), FieldValue::U64List(vec![3, 4]));
        });

        let (a, b) = digests();
        let m = base_manifest();
        let seal = |model: &Sha256Digest, abi: &Sha256Digest, r: &LayoutRecipe| {
            compute_layout_seals(model, abi, r, &m)
                .unwrap()
                .layout_sha256
        };
        let ok = seal(&a, &b, &base_recipe());
        assert_ne!(
            ok,
            seal(&Sha256Digest::of(b"model2"), &b, &base_recipe()),
            "source payload"
        );
        assert_ne!(
            ok,
            seal(&a, &Sha256Digest::of(b"abi2"), &base_recipe()),
            "loader ABI"
        );
        assert_ne!(
            ok,
            seal(&a, &b, &base_recipe().with("pp", FieldValue::U64(2))),
            "recipe/placement"
        );
        assert_ne!(
            ok,
            seal(
                &a,
                &b,
                &base_recipe().with("placement", FieldValue::str("auto"))
            ),
            "placement"
        );
    }

    #[test]
    fn manifest_roundtrip_and_validation_negatives() {
        let m = base_manifest();
        let bytes = m.encode_canonical().unwrap();
        assert_eq!(WeightManifest::decode_canonical(&bytes).unwrap(), m);
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(WeightManifest::decode_canonical(&extra).is_err());
        for cut in 0..bytes.len() {
            assert!(
                WeightManifest::decode_canonical(&bytes[..cut]).is_err(),
                "prefix {cut} accepted"
            );
        }

        let bad = |label: &str, f: &dyn Fn(&mut WeightManifest)| {
            let mut m = base_manifest();
            f(&mut m);
            assert!(m.validate().is_err(), "{label} must be rejected");
            assert!(m.encode_canonical().is_err(), "{label} must not encode");
        };
        bad("empty", &|m| m.allocations.clear());
        bad("unsorted allocations", &|m| m.allocations.reverse());
        bad("duplicate owner", &|m| {
            m.allocations[1].owner_id = "a.embed".into()
        });
        bad("zero granularity", &|m| m.allocations[0].granularity = 0);
        bad("unaligned physical", &|m| {
            m.allocations[0].physical_bytes = 8000
        });
        bad("logical > physical", &|m| {
            m.allocations[0].logical_bytes = 80_000
        });
        bad("tail pad too small", &|m| {
            m.allocations[0].physical_bytes = 65_536
        });
        bad("tail pad one aligned page short", &|m| {
            m.allocations[0].logical_bytes = 77_824 - EXPORT_WEIGHT_TAIL_PAD_BYTES + 1;
        });
        bad("logical + pad overflows", &|m| {
            m.allocations[0].logical_bytes = u64::MAX
        });
        bad("unsorted views", &|m| m.views.swap(0, 1));
        bad("unknown owner", &|m| m.views[1].owner_id = "zzz".into());
        bad("view beyond logical", &|m| m.views[2].byte_extent = 9000);
        bad("extent overflow", &|m| m.views[2].byte_offset = u64::MAX);
        bad("stride rank", &|m| m.views[0].strides = vec![1, 2]);
        bad("matrix rows exceed", &|m| {
            m.views[1].matrix.as_mut().unwrap().m = 1000
        });
        bad("matrix mul overflow", &|m| {
            m.views[1].matrix.as_mut().unwrap().row_stride_bytes = u64::MAX
        });
        bad("dangling companion", &|m| {
            m.views[1].companions.insert("zeros".into(), "nope".into());
        });
        bad("alias offset mismatch", &|m| m.views[3].byte_offset = 8);
        bad("alias unknown target", &|m| {
            m.views[3].alias_of = Some("nope".into())
        });
        bad("alias chain", &|m| {
            m.views[0].alias_of = Some("lm_head".into());
        });
        bad("empty arch", &|m| m.arch.clear());
    }

    // ---- source certificates ----

    fn ro_file(path: &Path, content: &[u8]) {
        std::fs::write(path, content).unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o444)).unwrap();
    }

    #[test]
    fn source_certificate_rules() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let src = dir.path().join("model.hfq");
        let cert_path = dir.path().join("model.cert");
        ro_file(&src, b"0123456789abcdef");

        let (file, cert) = certify_source(&src, "base").unwrap();
        assert_eq!(*cert.sha256(), Sha256Digest::of(b"0123456789abcdef"));
        assert_eq!(cert.len(), 16);
        assert_eq!(cert.to_source_digest().role, "base");
        drop(file);
        cert.write_to(&cert_path).unwrap();
        assert_eq!(std::fs::metadata(&cert_path).unwrap().mode() & 0o777, 0o444);

        let (_f, loaded) = SourceCertificate::verify_source_path(&cert_path, &src).unwrap();
        assert_eq!(loaded, cert);
        assert_eq!(
            SourceCertificate::decode(&cert.encode().unwrap()).unwrap(),
            cert
        );

        // tampered bytes (checksum)
        let mut raw = cert.encode().unwrap();
        raw[20] ^= 1;
        assert!(SourceCertificate::decode(&raw).is_err());

        // writable certificate is not trusted
        std::fs::set_permissions(&cert_path, std::fs::Permissions::from_mode(0o664)).unwrap();
        assert!(matches!(
            SourceCertificate::load_trusted(&cert_path),
            Err(WeightHostError::CertificateInvalid(_))
        ));
        std::fs::set_permissions(&cert_path, std::fs::Permissions::from_mode(0o444)).unwrap();

        // symlinked certificate is not trusted
        let link = dir.path().join("link.cert");
        std::os::unix::fs::symlink(&cert_path, &link).unwrap();
        assert!(SourceCertificate::load_trusted(&link).is_err());

        // replaced source (new inode, same length) is stale
        let other = dir.path().join("other");
        ro_file(&other, b"fedcba9876543210");
        std::fs::rename(&other, &src).unwrap();
        assert!(matches!(
            SourceCertificate::verify_source_path(&cert_path, &src),
            Err(WeightHostError::SourceMutated(_))
        ));

        // writable source rejected both when certifying and when verifying
        let w = dir.path().join("writable");
        std::fs::write(&w, b"x").unwrap();
        std::fs::set_permissions(&w, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(matches!(
            certify_source(&w, "base"),
            Err(WeightHostError::SourceNotImmutable(_))
        ));

        // changed length is stale
        let src2 = dir.path().join("m2");
        ro_file(&src2, b"aaaa");
        let (_f2, c2) = certify_source(&src2, "base").unwrap();
        let mut id = *c2.identity();
        id.len += 1;
        assert_ne!(id, *c2.identity());
        let c2_path = dir.path().join("m2.cert");
        c2.write_to(&c2_path).unwrap();
        let src3 = dir.path().join("m3");
        ro_file(&src3, b"aaaaa");
        assert!(SourceCertificate::verify_source_path(&c2_path, &src3).is_err());
        // role must be non-empty
        assert!(certify_source(&src2, "").is_err());
    }

    // ---- peer credentials ----

    #[test]
    fn peer_credential_rules() {
        let _g = fd_lock();
        let me = PeerCredentials {
            pid: 42,
            uid: euid(),
            gid: 0,
        };
        assert!(verify_peer(&me, None).is_ok());
        assert!(verify_peer(&me, Some(42)).is_ok());
        assert!(matches!(
            verify_peer(&me, Some(43)),
            Err(WeightHostError::PeerCredentials(_))
        ));
        let other = PeerCredentials {
            uid: euid().wrapping_add(1),
            ..me
        };
        assert!(matches!(
            verify_peer(&other, None),
            Err(WeightHostError::PeerCredentials(_))
        ));
        assert!(verify_peer(&PeerCredentials { pid: 0, ..me }, None).is_err());
    }

    // ---- live protocol ----

    struct Fixture {
        publication: Arc<WeightPublication>,
        model: Sha256Digest,
        abi: Sha256Digest,
        recipe: LayoutRecipe,
    }

    fn wide_manifest(n: usize) -> WeightManifest {
        let mut allocations = Vec::new();
        let mut views = Vec::new();
        for i in 0..n {
            let id = format!("a{i:04}");
            allocations.push(alloc(&id, WIDE_PHYS, 4000));
            views.push(view(&format!("w{i:04}"), &id, 0, 4000));
        }
        WeightManifest {
            arch: "test-dense".into(),
            config: CanonicalFields::new(),
            dispatch_metadata: Vec::new(),
            allocations,
            views,
        }
    }

    fn fixture(n: usize) -> Fixture {
        let (model, abi) = digests();
        let recipe = base_recipe();
        let fds: Vec<OwnedFd> = (0..n).map(|i| memfd(&[i as u8 + 1; 64])).collect();
        let publication = Arc::new(
            WeightPublication::new(
                Epoch([7u8; 16]),
                card(),
                model,
                abi,
                recipe.clone(),
                wide_manifest(n),
                fds,
            )
            .unwrap(),
        );
        Fixture {
            publication,
            model,
            abi,
            recipe,
        }
    }

    fn hello_for(f: &Fixture) -> HostHello {
        HostHello {
            card: card(),
            source_sha256: f.model,
            loader_abi_sha256: f.abi,
            recipe: f.recipe.clone(),
            admission_token: "tok".to_string(),
            provenance: BuildProvenance {
                label: "client".into(),
                ..Default::default()
            },
            options: ConnectOptions {
                handshake_timeout: Duration::from_secs(5),
                expected_host_pid: None,
            },
        }
    }

    const T: Duration = Duration::from_secs(5);

    #[test]
    fn full_lifecycle_with_real_memfds() {
        isolate_fd!();
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let before = open_fd_count();
        {
            let f = fixture(3);
            let server = Arc::new(
                WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default())
                    .unwrap(),
            );
            assert_eq!(std::fs::metadata(&sock).unwrap().mode() & 0o777, 0o600);
            let s2 = server.clone();
            let host = thread::spawn(move || -> Result<(), WeightHostError> {
                let conn = s2.accept(Some(T))?;
                assert_eq!(conn.peer().pid, std::process::id());
                assert_eq!(conn.hello().provenance.label, "client");
                let mut sess = conn.admit(|req| {
                    assert_eq!(req.fd_count, 3);
                    if req.hello.admission_token == "tok" {
                        Ok(())
                    } else {
                        Err("bad token".into())
                    }
                })?;
                assert_eq!(sess.state(), HostSessionState::Admitted);
                assert_eq!(sess.fds_exposed(), 0);
                sess.send_allocations()?;
                assert_eq!(sess.fds_exposed(), 3);
                assert_eq!(sess.state(), HostSessionState::FdsSent);
                let ack = sess.wait_import_ack(Some(T))?;
                assert_eq!(ack.imported_bytes, 3 * WIDE_PHYS);
                assert_eq!(sess.state(), HostSessionState::ImportAcked);
                sess.wait_release(Some(T))?;
                assert_eq!(sess.state(), HostSessionState::Released);
                Ok(())
            });

            let mut lease = WeightHostClient::connect(&sock, hello_for(&f)).unwrap();
            assert!(!lease.is_import_acked(), "connect must never ack");
            assert_eq!(lease.epoch(), Epoch([7u8; 16]));
            assert_eq!(lease.allocation_fds().len(), 3);
            assert_eq!(lease.unique_bytes(), 3 * WIDE_PHYS);
            assert_eq!(*lease.layout_sha256(), f.publication.seals().layout_sha256);
            assert!(lease.host_alive().unwrap());
            for i in 0..3 {
                let fd = lease.allocation_fd(i).unwrap();
                assert!(fd_is_cloexec(fd.as_raw_fd()).unwrap());
                let file = File::from(fd.try_clone_to_owned().unwrap());
                let mut buf = [0u8; 64];
                file.read_exact_at(&mut buf, 0).unwrap();
                assert_eq!(buf, [i as u8 + 1; 64]);
            }
            let (idx, a, _) = lease.allocation("a0002").unwrap();
            assert_eq!((idx, a.physical_bytes), (2, WIDE_PHYS));

            assert!(
                matches!(lease.release(), Err(WeightHostError::InvalidState(_))),
                "release before ack"
            );
            assert!(matches!(
                lease.ack_import(1),
                Err(WeightHostError::ImportedBytesMismatch { .. })
            ));
            assert!(!lease.is_import_acked());
            lease.ack_import(lease.unique_bytes()).unwrap();
            assert!(matches!(
                lease.ack_import(lease.unique_bytes()),
                Err(WeightHostError::InvalidState(_))
            ));
            assert!(
                matches!(lease.release(), Err(WeightHostError::InvalidState(_))),
                "fds still open"
            );
            let taken = lease.take_allocation_fds();
            assert_eq!(taken.len(), 3);
            drop(taken);
            lease.release().unwrap();
            assert!(matches!(
                lease.release(),
                Err(WeightHostError::InvalidState(_))
            ));
            host.join().unwrap().unwrap();
        }
        assert_eq!(open_fd_count(), before, "descriptor leak");
        assert!(!sock.exists(), "server drop must unlink its own socket");
    }

    #[test]
    fn exact_batch_totals_across_32_fd_boundaries() {
        isolate_fd!();
        let _g = fd_lock();
        for n in [1usize, 31, 32, 33, 64, 65] {
            let dir = private_tempdir();
            let sock = dir.path().join("h.sock");
            let before = open_fd_count();
            {
                let f = fixture(n);
                let server = Arc::new(
                    WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default())
                        .unwrap(),
                );
                let s2 = server.clone();
                let host = thread::spawn(move || -> Result<(), WeightHostError> {
                    let mut sess = s2.accept(Some(T))?.admit(|_| Ok(()))?;
                    sess.send_allocations()?;
                    assert_eq!(sess.fds_exposed(), n);
                    sess.wait_import_ack(Some(T))?;
                    sess.wait_release(Some(T))
                });
                let mut lease = WeightHostClient::connect(&sock, hello_for(&f)).unwrap();
                assert_eq!(lease.allocation_fds().len(), n);
                let last = lease.allocation_fd(n - 1).unwrap();
                let file = File::from(last.try_clone_to_owned().unwrap());
                let mut buf = [0u8; 64];
                file.read_exact_at(&mut buf, 0).unwrap();
                assert_eq!(buf, [(n - 1) as u8 + 1; 64]);
                lease.ack_import(lease.unique_bytes()).unwrap();
                lease.close_allocation_fds();
                lease.release().unwrap();
                host.join().unwrap().unwrap();
            }
            assert_eq!(open_fd_count(), before, "leak at n={n}");
        }
    }

    #[test]
    fn pipe_descriptors_travel_too() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let (r, w) = pipe_read_end();
        let (model, abi) = digests();
        let recipe = base_recipe();
        let publication = Arc::new(
            WeightPublication::new(
                Epoch([1u8; 16]),
                card(),
                model,
                abi,
                recipe.clone(),
                wide_manifest(1),
                vec![r],
            )
            .unwrap(),
        );
        let f = Fixture {
            publication,
            model,
            abi,
            recipe,
        };
        let server = Arc::new(
            WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default()).unwrap(),
        );
        let s2 = server.clone();
        let host = thread::spawn(move || -> Result<HostSession, WeightHostError> {
            let mut sess = s2.accept(Some(T))?.admit(|_| Ok(()))?;
            sess.send_allocations()?;
            Ok(sess)
        });
        let lease = WeightHostClient::connect(&sock, hello_for(&f)).unwrap();
        let _sess = host.join().unwrap().unwrap();
        // write on the host-side write end, read through the received duplicate
        let mut wf = File::from(w);
        wf.write_all(b"ping").unwrap();
        let mut rf = File::from(
            lease
                .allocation_fd(0)
                .unwrap()
                .try_clone_to_owned()
                .unwrap(),
        );
        let mut buf = [0u8; 4];
        rf.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"ping");
    }

    #[test]
    fn dropped_lease_is_eof_never_a_fabricated_ack() {
        isolate_fd!();
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let before = open_fd_count();
        {
            let f = fixture(2);
            let server = Arc::new(
                WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default())
                    .unwrap(),
            );
            let s2 = server.clone();
            let (tx, rx) = std::sync::mpsc::channel::<()>();
            let host = thread::spawn(move || -> Result<(), WeightHostError> {
                let mut sess = s2.accept(Some(T))?.admit(|_| Ok(()))?;
                sess.send_allocations()?;
                rx.recv().unwrap();
                // lease was dropped without ack
                let e = sess.wait_import_ack(Some(T)).unwrap_err();
                assert!(matches!(e, WeightHostError::PeerClosed(_)), "got {e:?}");
                assert_eq!(sess.state(), HostSessionState::Failed);
                assert!(sess.peer_closed().unwrap());
                Ok(())
            });
            let lease = WeightHostClient::connect(&sock, hello_for(&f)).unwrap();
            assert!(!lease.is_import_acked());
            drop(lease);
            tx.send(()).unwrap();
            host.join().unwrap().unwrap();
        }
        assert_eq!(open_fd_count(), before);
    }

    #[test]
    fn ack_wait_times_out_without_ack() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let f = fixture(1);
        let server = Arc::new(
            WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default()).unwrap(),
        );
        let s2 = server.clone();
        let host = thread::spawn(move || -> Result<HostSession, WeightHostError> {
            let mut sess = s2.accept(Some(T))?.admit(|_| Ok(()))?;
            sess.send_allocations()?;
            let e = sess
                .wait_import_ack(Some(Duration::from_millis(50)))
                .unwrap_err();
            assert!(matches!(e, WeightHostError::Timeout(_)), "got {e:?}");
            assert_eq!(sess.state(), HostSessionState::Failed);
            Ok(sess)
        });
        let lease = WeightHostClient::connect(&sock, hello_for(&f)).unwrap();
        // keep the host session (and its socket) alive past the join
        let _sess = host.join().unwrap().unwrap();
        // host_alive is still Ok: the host kept the socket open
        assert!(lease.host_alive().unwrap());
    }

    #[test]
    fn host_exit_is_visible_to_live_lease() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let f = fixture(1);
        let server = Arc::new(
            WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default()).unwrap(),
        );
        let s2 = server.clone();
        let host = thread::spawn(move || -> Result<(), WeightHostError> {
            let mut sess = s2.accept(Some(T))?.admit(|_| Ok(()))?;
            sess.send_allocations()?;
            Ok(()) // session dropped here: socket closes
        });
        let lease = WeightHostClient::connect(&sock, hello_for(&f)).unwrap();
        host.join().unwrap().unwrap();
        // join() waits for the host thread to return, but the peer hangup
        // reaches this socket asynchronously: on a loaded runner a single
        // nonblocking MSG_PEEK can still see the socket Empty. Poll for the
        // hangup, then confirm through the lease's own probe.
        if lease.host_alive().unwrap() {
            wait_fd(
                lease.conn.fd(),
                libc::POLLIN,
                Some(Instant::now() + T),
                "host hangup",
            )
            .unwrap();
        }
        assert!(!lease.host_alive().unwrap());
        // FDs and manifest stay valid for draining the current request
        assert_eq!(lease.allocation_fds().len(), 1);
    }

    #[test]
    fn identity_mismatch_rejected_before_any_fd() {
        isolate_fd!();
        let _g = fd_lock();
        let before = open_fd_count();
        type Mutator = Box<dyn Fn(&mut HostHello)>;
        let cases: Vec<(&str, Mutator)> = vec![
            (
                "source_sha256",
                Box::new(|h| h.source_sha256 = Sha256Digest::of(b"other")),
            ),
            (
                "loader_abi_sha256",
                Box::new(|h| h.loader_abi_sha256 = Sha256Digest::of(b"other")),
            ),
            (
                "recipe_sha256",
                Box::new(|h| h.recipe.set("pp", FieldValue::U64(2))),
            ),
            ("card.uuid", Box::new(|h| h.card.uuid = "GPU-other".into())),
            (
                "card.pci_bus_id",
                Box::new(|h| h.card.pci_bus_id = "0000:04:00.0".into()),
            ),
            (
                "card.device_arch",
                Box::new(|h| h.card.device_arch = "gfx1100".into()),
            ),
        ];
        for (field, mutate) in cases {
            let dir = private_tempdir();
            let sock = dir.path().join("h.sock");
            {
                let f = fixture(2);
                let server = Arc::new(
                    WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default())
                        .unwrap(),
                );
                let s2 = server.clone();
                let host = thread::spawn(move || s2.accept(Some(T)).map(|_| ()));
                let mut h = hello_for(&f);
                mutate(&mut h);
                let e = WeightHostClient::connect(&sock, h).unwrap_err();
                assert!(
                    matches!(&e, WeightHostError::Rejected(r) if r.contains(field)),
                    "{field}: {e:?}"
                );
                let he = host.join().unwrap().unwrap_err();
                assert!(
                    matches!(he, WeightHostError::IdentityMismatch(x) if x == field),
                    "{field}"
                );
            }
        }
        assert_eq!(open_fd_count(), before, "descriptor leak");
    }

    #[test]
    fn admission_callback_denial_rejects_client() {
        isolate_fd!();
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let before = open_fd_count();
        {
            let f = fixture(1);
            let server = Arc::new(
                WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default())
                    .unwrap(),
            );
            let s2 = server.clone();
            let host = thread::spawn(move || {
                s2.accept(Some(T))
                    .and_then(|c| c.admit(|_| Err("no ledger capacity".to_string())))
                    .map(|_| ())
            });
            let e = WeightHostClient::connect(&sock, hello_for(&f)).unwrap_err();
            assert!(
                matches!(&e, WeightHostError::Rejected(r) if r == "no ledger capacity"),
                "{e:?}"
            );
            assert!(matches!(
                host.join().unwrap(),
                Err(WeightHostError::AdmissionDenied(_))
            ));
        }
        assert_eq!(open_fd_count(), before);
    }

    #[test]
    fn expected_pid_checks_on_both_sides() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let f = fixture(1);
        // client expects a different host pid: fails before sending Hello
        let server = Arc::new(
            WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default()).unwrap(),
        );
        let mut h = hello_for(&f);
        h.options.expected_host_pid = Some(std::process::id().wrapping_add(1));
        assert!(matches!(
            WeightHostClient::connect(&sock, h),
            Err(WeightHostError::PeerCredentials(_))
        ));
        drop(server);

        // host expects a different tenant pid
        let opts = ServeOptions {
            expected_peer_pid: Some(std::process::id().wrapping_add(1)),
            ..Default::default()
        };
        let server = Arc::new(WeightHostServer::bind(&sock, f.publication.clone(), opts).unwrap());
        let s2 = server.clone();
        let host = thread::spawn(move || s2.accept(Some(T)).map(|_| ()));
        let _ = WeightHostClient::connect(&sock, hello_for(&f)).unwrap_err();
        assert!(matches!(
            host.join().unwrap(),
            Err(WeightHostError::PeerCredentials(_))
        ));
        drop(server);

        // correct expectations succeed
        let opts = ServeOptions {
            expected_peer_pid: Some(std::process::id()),
            ..Default::default()
        };
        let server = Arc::new(WeightHostServer::bind(&sock, f.publication.clone(), opts).unwrap());
        let s2 = server.clone();
        let host = thread::spawn(move || -> Result<(), WeightHostError> {
            let mut s = s2.accept(Some(T))?.admit(|_| Ok(()))?;
            s.send_allocations()
        });
        let mut h = hello_for(&f);
        h.options.expected_host_pid = Some(std::process::id());
        let _lease = WeightHostClient::connect(&sock, h).unwrap();
        host.join().unwrap().unwrap();
    }

    #[test]
    fn socket_filesystem_rules() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let f = fixture(1);
        let server =
            WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default()).unwrap();
        // never overwrite an existing path
        assert!(
            WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default()).is_err()
        );
        // group-accessible directory is refused by clients and by bind
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(matches!(
            WeightHostClient::connect(&sock, hello_for(&f)),
            Err(WeightHostError::InsecureSocket(_))
        ));
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        // permissive socket mode refused
        std::fs::set_permissions(&sock, std::fs::Permissions::from_mode(0o666)).unwrap();
        assert!(matches!(
            WeightHostClient::connect(&sock, hello_for(&f)),
            Err(WeightHostError::InsecureSocket(_))
        ));
        std::fs::set_permissions(&sock, std::fs::Permissions::from_mode(0o600)).unwrap();
        drop(server);
        assert!(!sock.exists());
        // missing socket: single attempt, no retry
        assert!(WeightHostClient::connect(&sock, hello_for(&f)).is_err());
        // path that is not a socket
        let plain = dir.path().join("plain");
        std::fs::write(&plain, b"x").unwrap();
        std::fs::set_permissions(&plain, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(matches!(
            WeightHostClient::connect(&plain, hello_for(&f)),
            Err(WeightHostError::InsecureSocket(_))
        ));
    }

    #[test]
    fn hello_limits_enforced_before_connecting() {
        let _g = fd_lock();
        let f = fixture(1);
        let mut h = hello_for(&f);
        h.admission_token = "x".repeat(MAX_ADMISSION_TOKEN_BYTES + 1);
        assert!(matches!(
            WeightHostClient::connect(Path::new("/nonexistent/none.sock"), h),
            Err(WeightHostError::LimitExceeded(_))
        ));
        let mut h = hello_for(&f);
        h.options.handshake_timeout = Duration::ZERO;
        assert!(matches!(
            WeightHostClient::connect(Path::new("/nonexistent/none.sock"), h),
            Err(WeightHostError::InvalidArgument(_))
        ));
    }

    #[test]
    fn publication_validation() {
        let _g = fd_lock();
        let (model, abi) = digests();
        let mk = |fds: Vec<OwnedFd>, epoch: Epoch| {
            WeightPublication::new(
                epoch,
                card(),
                model,
                abi,
                base_recipe(),
                wide_manifest(2),
                fds,
            )
        };
        assert!(matches!(
            mk(vec![memfd(b"a")], Epoch([1; 16])),
            Err(WeightHostError::FdCount { .. })
        ));
        assert!(mk(vec![memfd(b"a"), memfd(b"b")], Epoch([0; 16])).is_err());
        // non-CLOEXEC descriptor refused
        let leaky = {
            // SAFETY: dup without CLOEXEC of a live fd.
            let raw = unsafe { libc::fcntl(memfd(b"c").as_raw_fd(), libc::F_DUPFD, 3) };
            unsafe { OwnedFd::from_raw_fd(raw) }
        };
        assert!(matches!(
            mk(vec![memfd(b"a"), leaky], Epoch([1; 16])),
            Err(WeightHostError::FdInvalid(_))
        ));
        // device arch mismatch between allocations and card
        let mut other_card = card();
        other_card.device_arch = "gfx1100".into();
        assert!(WeightPublication::new(
            Epoch([1; 16]),
            other_card,
            model,
            abi,
            base_recipe(),
            wide_manifest(1),
            vec![memfd(b"a")]
        )
        .is_err());
        assert!(Epoch::random().unwrap() != Epoch::random().unwrap());
    }

    #[test]
    fn host_rejects_unsupported_client_version() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let f = fixture(1);
        let server = Arc::new(
            WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default()).unwrap(),
        );
        let s2 = server.clone();
        let host = thread::spawn(move || s2.accept(Some(T)).map(|_| ()));
        let raw = Conn(connect_unix(&sock).unwrap());
        let payload = encode_hello(&hello_for(&f)).unwrap();
        let bytes = frame_bytes(2, MsgKind::Hello as u32, &payload, 0);
        send_bytes(&raw, &bytes, &[], Some(Instant::now() + T), "t").unwrap();
        assert!(matches!(
            host.join().unwrap(),
            Err(WeightHostError::UnsupportedVersion {
                got: 2,
                supported: 1
            })
        ));
    }

    // ---- malicious hosts: every case must fail closed without leaking an fd ----

    fn evil_host<F>(sock: &Path, script: F) -> thread::JoinHandle<()>
    where
        F: FnOnce(&Conn, &HostHello) + Send + 'static,
    {
        let listener = UnixListener::bind(sock).unwrap();
        std::fs::set_permissions(sock, std::fs::Permissions::from_mode(0o600)).unwrap();
        thread::spawn(move || {
            let (s, _) = listener.accept().unwrap();
            let conn = Conn::from_stream(s).unwrap();
            let frame = read_frame(&conn, Some(Instant::now() + T), "evil read").unwrap();
            let hello = decode_hello(&frame.payload).unwrap();
            script(&conn, &hello);
            // keep the socket open briefly so the client sees our data, not a reset
            thread::sleep(Duration::from_millis(30));
        })
    }

    fn dl() -> Option<Instant> {
        Some(Instant::now() + T)
    }

    fn good_ready(h: &HostHello, n: usize) -> ReadyMsg {
        let m = wide_manifest(n);
        let seals =
            compute_layout_seals(&h.source_sha256, &h.loader_abi_sha256, &h.recipe, &m).unwrap();
        ReadyMsg {
            epoch: Epoch([9; 16]),
            source_sha256: h.source_sha256,
            loader_abi_sha256: h.loader_abi_sha256,
            recipe_sha256: seals.recipe_sha256,
            layout_sha256: seals.layout_sha256,
            unique_bytes: m.unique_bytes().unwrap(),
            fd_count: n as u32,
            manifest: m,
        }
    }

    fn send_ready(c: &Conn, r: &ReadyMsg) {
        let _ = send_frame(
            c,
            MsgKind::Ready,
            &encode_ready(r).unwrap(),
            &[],
            dl(),
            "evil",
        );
    }

    fn send_batch(c: &Conn, b: BatchMsg, header_fds: u32, attach: &[RawFd]) {
        let bytes = frame_bytes(
            PROTOCOL_VERSION,
            MsgKind::FdBatch as u32,
            &encode_batch(&b),
            header_fds,
        );
        let _ = send_bytes(c, &bytes, attach, dl(), "evil");
    }

    #[test]
    fn malicious_hosts_fail_closed_without_fd_leaks() {
        isolate_fd!();
        let _g = fd_lock();
        type Script = Box<dyn FnOnce(&Conn, &HostHello) + Send>;
        let cases: Vec<(&str, Script, fn(&WeightHostError) -> bool)> = vec![
            (
                "wrong version",
                Box::new(|c, h| {
                    let p = encode_ready(&good_ready(h, 1)).unwrap();
                    let _ = send_bytes(
                        c,
                        &frame_bytes(2, MsgKind::Ready as u32, &p, 0),
                        &[],
                        dl(),
                        "evil",
                    );
                }),
                |e| matches!(e, WeightHostError::UnsupportedVersion { got: 2, .. }),
            ),
            (
                "bad magic",
                Box::new(|c, _| {
                    let mut b = frame_bytes(PROTOCOL_VERSION, MsgKind::Ready as u32, &[], 0);
                    b[0] = b'X';
                    let _ = send_bytes(c, &b, &[], dl(), "evil");
                }),
                |e| matches!(e, WeightHostError::Protocol(_)),
            ),
            (
                "oversize payload",
                Box::new(|c, _| {
                    let mut b = frame_bytes(PROTOCOL_VERSION, MsgKind::Ready as u32, &[], 0);
                    b[12..16].copy_from_slice(&((MAX_FRAME_PAYLOAD as u32) + 1).to_le_bytes());
                    let _ = send_bytes(c, &b, &[], dl(), "evil");
                }),
                |e| matches!(e, WeightHostError::LimitExceeded(_)),
            ),
            (
                "unknown kind",
                Box::new(|c, _| {
                    let b = frame_bytes(PROTOCOL_VERSION, 99, &[], 0);
                    let _ = send_bytes(c, &b, &[], dl(), "evil");
                }),
                |e| matches!(e, WeightHostError::Protocol(_)),
            ),
            (
                "wrong message kind",
                Box::new(|c, _| {
                    let _ = send_frame(
                        c,
                        MsgKind::ImportAck,
                        &encode_epoch_layout(&Epoch([1; 16]), &Sha256Digest::of(b"x"), Some(0)),
                        &[],
                        dl(),
                        "evil",
                    );
                }),
                |e| matches!(e, WeightHostError::Protocol(_)),
            ),
            (
                "fds attached to Ready",
                Box::new(|c, h| {
                    let m = memfd(b"x");
                    let p = encode_ready(&good_ready(h, 1)).unwrap();
                    let _ = send_bytes(
                        c,
                        &frame_bytes(PROTOCOL_VERSION, MsgKind::Ready as u32, &p, 1),
                        &[m.as_raw_fd()],
                        dl(),
                        "evil",
                    );
                }),
                |e| {
                    matches!(
                        e,
                        WeightHostError::FdCount {
                            expected: 0,
                            got: 1
                        }
                    )
                },
            ),
            (
                "tampered layout seal",
                Box::new(|c, h| {
                    let mut r = good_ready(h, 1);
                    r.layout_sha256 = Sha256Digest::of(b"forged");
                    send_ready(c, &r);
                }),
                |e| matches!(e, WeightHostError::IdentityMismatch("layout_sha256")),
            ),
            (
                "tampered source echo",
                Box::new(|c, h| {
                    let mut r = good_ready(h, 1);
                    r.source_sha256 = Sha256Digest::of(b"forged");
                    send_ready(c, &r);
                }),
                |e| matches!(e, WeightHostError::IdentityMismatch("source_sha256")),
            ),
            (
                "tampered recipe seal",
                Box::new(|c, h| {
                    let mut r = good_ready(h, 1);
                    r.recipe_sha256 = Sha256Digest::of(b"forged");
                    send_ready(c, &r);
                }),
                |e| matches!(e, WeightHostError::IdentityMismatch("recipe_sha256")),
            ),
            (
                "out-of-bounds view in manifest",
                Box::new(|c, h| {
                    let mut r = good_ready(h, 1);
                    r.manifest.views[0].byte_extent = 1 << 40;
                    send_ready(c, &r);
                }),
                |e| matches!(e, WeightHostError::ManifestInvalid(_)),
            ),
            (
                "manifest changed after seal",
                Box::new(|c, h| {
                    let mut r = good_ready(h, 1);
                    r.manifest.views[0].dtype = "f16".into();
                    send_ready(c, &r);
                }),
                |e| matches!(e, WeightHostError::IdentityMismatch("layout_sha256")),
            ),
            (
                "fd_count disagrees with allocations",
                Box::new(|c, h| {
                    let mut r = good_ready(h, 2);
                    r.fd_count = 1;
                    send_ready(c, &r);
                }),
                |e| matches!(e, WeightHostError::FdCount { .. }),
            ),
            (
                "unique bytes disagree",
                Box::new(|c, h| {
                    let mut r = good_ready(h, 1);
                    r.unique_bytes += 4096;
                    send_ready(c, &r);
                }),
                |e| matches!(e, WeightHostError::ImportedBytesMismatch { .. }),
            ),
            (
                "trailing bytes in Ready",
                Box::new(|c, h| {
                    let mut p = encode_ready(&good_ready(h, 1)).unwrap();
                    p.push(0);
                    let _ = send_frame(c, MsgKind::Ready, &p, &[], dl(), "evil");
                }),
                |e| matches!(e, WeightHostError::Protocol(_)),
            ),
            (
                "EOF before any fd",
                Box::new(|c, h| send_ready(c, &good_ready(h, 1))),
                |e| matches!(e, WeightHostError::PeerClosed(_)),
            ),
            (
                "missing fd in batch",
                Box::new(|c, h| {
                    send_ready(c, &good_ready(h, 1));
                    send_batch(
                        c,
                        BatchMsg {
                            index: 0,
                            total: 1,
                            first: 0,
                            count: 1,
                        },
                        0,
                        &[],
                    );
                }),
                |e| {
                    matches!(
                        e,
                        WeightHostError::FdCount {
                            expected: 1,
                            got: 0
                        }
                    )
                },
            ),
            (
                "extra fd beyond header",
                Box::new(|c, h| {
                    send_ready(c, &good_ready(h, 2));
                    let (a, b, d) = (memfd(b"a"), memfd(b"b"), memfd(b"c"));
                    send_batch(
                        c,
                        BatchMsg {
                            index: 0,
                            total: 1,
                            first: 0,
                            count: 2,
                        },
                        2,
                        &[a.as_raw_fd(), b.as_raw_fd(), d.as_raw_fd()],
                    );
                }),
                |e| {
                    matches!(
                        e,
                        WeightHostError::FdCount {
                            expected: 2,
                            got: 3
                        }
                    )
                },
            ),
            (
                "batch count differs from attached",
                Box::new(|c, h| {
                    send_ready(c, &good_ready(h, 2));
                    let a = memfd(b"a");
                    send_batch(
                        c,
                        BatchMsg {
                            index: 0,
                            total: 1,
                            first: 0,
                            count: 2,
                        },
                        1,
                        &[a.as_raw_fd()],
                    );
                }),
                |e| {
                    matches!(
                        e,
                        WeightHostError::FdCount {
                            expected: 2,
                            got: 1
                        }
                    )
                },
            ),
            (
                "batch numbering wrong",
                Box::new(|c, h| {
                    send_ready(c, &good_ready(h, 1));
                    let a = memfd(b"a");
                    send_batch(
                        c,
                        BatchMsg {
                            index: 1,
                            total: 1,
                            first: 0,
                            count: 1,
                        },
                        1,
                        &[a.as_raw_fd()],
                    );
                }),
                |e| matches!(e, WeightHostError::Protocol(_)),
            ),
            (
                "batch total wrong",
                Box::new(|c, h| {
                    send_ready(c, &good_ready(h, 1));
                    let a = memfd(b"a");
                    send_batch(
                        c,
                        BatchMsg {
                            index: 0,
                            total: 2,
                            first: 0,
                            count: 1,
                        },
                        1,
                        &[a.as_raw_fd()],
                    );
                }),
                |e| matches!(e, WeightHostError::Protocol(_)),
            ),
            (
                "batch first index wrong",
                Box::new(|c, h| {
                    send_ready(c, &good_ready(h, 1));
                    let a = memfd(b"a");
                    send_batch(
                        c,
                        BatchMsg {
                            index: 0,
                            total: 1,
                            first: 1,
                            count: 1,
                        },
                        1,
                        &[a.as_raw_fd()],
                    );
                }),
                |e| matches!(e, WeightHostError::Protocol(_)),
            ),
            (
                "empty batch",
                Box::new(|c, h| {
                    send_ready(c, &good_ready(h, 1));
                    send_batch(
                        c,
                        BatchMsg {
                            index: 0,
                            total: 1,
                            first: 0,
                            count: 0,
                        },
                        0,
                        &[],
                    );
                }),
                |e| matches!(e, WeightHostError::Protocol(_)),
            ),
            (
                "more fds than the total",
                Box::new(|c, h| {
                    send_ready(c, &good_ready(h, 1));
                    let (a, b) = (memfd(b"a"), memfd(b"b"));
                    send_batch(
                        c,
                        BatchMsg {
                            index: 0,
                            total: 1,
                            first: 0,
                            count: 2,
                        },
                        2,
                        &[a.as_raw_fd(), b.as_raw_fd()],
                    );
                }),
                |e| matches!(e, WeightHostError::FdCount { .. }),
            ),
            (
                "33 fds in one message",
                Box::new(|c, h| {
                    send_ready(c, &good_ready(h, 33));
                    let owned: Vec<OwnedFd> = (0..33).map(|_| memfd(b"z")).collect();
                    let raw: Vec<RawFd> = owned.iter().map(|f| f.as_raw_fd()).collect();
                    send_batch(
                        c,
                        BatchMsg {
                            index: 0,
                            total: 2,
                            first: 0,
                            count: 33,
                        },
                        33,
                        &raw,
                    );
                }),
                |e| {
                    matches!(
                        e,
                        WeightHostError::LimitExceeded(_) | WeightHostError::Protocol(_)
                    )
                },
            ),
            (
                "directory descriptor",
                Box::new(|c, h| {
                    send_ready(c, &good_ready(h, 1));
                    let d = File::open("/").unwrap();
                    send_batch(
                        c,
                        BatchMsg {
                            index: 0,
                            total: 1,
                            first: 0,
                            count: 1,
                        },
                        1,
                        &[d.as_raw_fd()],
                    );
                }),
                |e| matches!(e, WeightHostError::FdInvalid(_)),
            ),
            (
                "second batch missing after first",
                Box::new(|c, h| {
                    send_ready(c, &good_ready(h, 33));
                    let owned: Vec<OwnedFd> = (0..32).map(|_| memfd(b"z")).collect();
                    let raw: Vec<RawFd> = owned.iter().map(|f| f.as_raw_fd()).collect();
                    send_batch(
                        c,
                        BatchMsg {
                            index: 0,
                            total: 2,
                            first: 0,
                            count: 32,
                        },
                        32,
                        &raw,
                    );
                }),
                |e| matches!(e, WeightHostError::PeerClosed(_)),
            ),
            (
                "reject frame instead of ready",
                Box::new(|c, _| {
                    let _ = send_frame(
                        c,
                        MsgKind::Reject,
                        &encode_reject("go away"),
                        &[],
                        dl(),
                        "evil",
                    );
                }),
                |e| matches!(e, WeightHostError::Rejected(r) if r == "go away"),
            ),
        ];

        let f = fixture(1);
        for (label, script, ok) in cases {
            let dir = private_tempdir();
            let sock = dir.path().join("h.sock");
            let before = open_fd_count();
            let mut hello = hello_for(&f);
            hello.options.handshake_timeout = Duration::from_secs(2);
            let handle = evil_host(&sock, script);
            let err = WeightHostClient::connect(&sock, hello).expect_err(label);
            let _ = handle.join();
            assert!(ok(&err), "{label}: unexpected error {err:?}");
            assert_eq!(open_fd_count(), before, "{label}: descriptor leak");
        }
    }

    #[test]
    fn handshake_deadline_is_absolute() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let f = fixture(1);
        // host accepts and reads Hello, then dribbles nothing: client must time out
        let handle = evil_host(&sock, |_c, _h| thread::sleep(Duration::from_millis(600)));
        let mut hello = hello_for(&f);
        hello.options.handshake_timeout = Duration::from_millis(200);
        let t0 = Instant::now();
        let e = WeightHostClient::connect(&sock, hello).unwrap_err();
        assert!(matches!(e, WeightHostError::Timeout(_)), "{e:?}");
        assert!(
            t0.elapsed() < Duration::from_millis(550),
            "deadline did not bound the handshake"
        );
        let _ = handle.join();
    }

    #[test]
    fn host_rejects_bad_clients_without_sending_fds() {
        isolate_fd!();
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let f = fixture(1);
        let before = open_fd_count();
        {
            let server = Arc::new(
                WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default())
                    .unwrap(),
            );
            let cases: Vec<(&str, Vec<u8>, bool)> = vec![
                (
                    "hello with fd",
                    frame_bytes(
                        PROTOCOL_VERSION,
                        MsgKind::Hello as u32,
                        &encode_hello(&hello_for(&f)).unwrap(),
                        1,
                    ),
                    true,
                ),
                (
                    "not a hello",
                    frame_bytes(PROTOCOL_VERSION, MsgKind::Release as u32, &[], 0),
                    false,
                ),
                (
                    "garbage payload",
                    frame_bytes(PROTOCOL_VERSION, MsgKind::Hello as u32, &[1, 2, 3], 0),
                    false,
                ),
            ];
            for (label, bytes, attach) in cases {
                let s2 = server.clone();
                let host = thread::spawn(move || s2.accept(Some(T)).map(|_| ()));
                let raw = Conn(connect_unix(&sock).unwrap());
                let m = memfd(b"x");
                let fds: Vec<RawFd> = if attach { vec![m.as_raw_fd()] } else { vec![] };
                send_bytes(&raw, &bytes, &fds, dl(), "t").unwrap();
                assert!(host.join().unwrap().is_err(), "{label}");
            }
        }
        assert_eq!(open_fd_count(), before);
    }

    #[test]
    fn accept_wait_is_bounded() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let f = fixture(1);
        let server =
            WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default()).unwrap();
        let e = server.accept(Some(Duration::from_millis(30))).unwrap_err();
        assert!(matches!(e, WeightHostError::Timeout("accept")));
    }

    #[test]
    fn sessions_enforce_state_order() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let f = fixture(1);
        let server = Arc::new(
            WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default()).unwrap(),
        );
        let s2 = server.clone();
        let host = thread::spawn(move || -> Result<(), WeightHostError> {
            let mut sess = s2.accept(Some(T))?.admit(|_| Ok(()))?;
            assert!(matches!(
                sess.wait_import_ack(Some(T)),
                Err(WeightHostError::InvalidState(_))
            ));
            assert!(matches!(
                sess.wait_release(Some(T)),
                Err(WeightHostError::InvalidState(_))
            ));
            sess.send_allocations()?;
            assert!(matches!(
                sess.send_allocations(),
                Err(WeightHostError::InvalidState(_))
            ));
            assert!(matches!(
                sess.wait_release(Some(T)),
                Err(WeightHostError::InvalidState(_))
            ));
            Ok(())
        });
        let _lease = WeightHostClient::connect(&sock, hello_for(&f)).unwrap();
        host.join().unwrap().unwrap();
    }

    #[test]
    fn tenant_wrong_ack_identity_fails_host_session() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let f = fixture(1);
        let server = Arc::new(
            WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default()).unwrap(),
        );
        let s2 = server.clone();
        let host = thread::spawn(move || -> Result<(), WeightHostError> {
            let mut sess = s2.accept(Some(T))?.admit(|_| Ok(()))?;
            sess.send_allocations()?;
            let e = sess.wait_import_ack(Some(T)).unwrap_err();
            assert!(
                matches!(e, WeightHostError::IdentityMismatch("layout_sha256")),
                "{e:?}"
            );
            assert_eq!(sess.state(), HostSessionState::Failed);
            Ok(())
        });
        let lease = WeightHostClient::connect(&sock, hello_for(&f)).unwrap();
        let forged = encode_epoch_layout(
            &lease.epoch(),
            &Sha256Digest::of(b"forged"),
            Some(lease.unique_bytes()),
        );
        send_frame(&lease.conn, MsgKind::ImportAck, &forged, &[], dl(), "t").unwrap();
        host.join().unwrap().unwrap();
    }

    #[test]
    fn from_listener_poll_release_and_drain() {
        isolate_fd!();
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let before = open_fd_count();
        {
            let f = fixture(2);
            // caller-owned listener: the server must not touch the filesystem
            let listener = UnixListener::bind(&sock).unwrap();
            std::fs::set_permissions(&sock, std::fs::Permissions::from_mode(0o600)).unwrap();
            let server = Arc::new(
                WeightHostServer::from_listener(
                    listener,
                    f.publication.clone(),
                    ServeOptions::default(),
                )
                .unwrap(),
            );
            let s2 = server.clone();
            let (tx, rx) = std::sync::mpsc::channel::<()>();
            let host = thread::spawn(move || -> Result<(), WeightHostError> {
                let conn = s2.accept(Some(T))?;
                let admitted = conn.admit(|_| Ok(()))?;
                // HostSession is Send and moves across threads
                let mut sess = thread::spawn(move || -> Result<HostSession, WeightHostError> {
                    let mut sess = admitted;
                    sess.send_allocations()?;
                    sess.wait_import_ack(Some(T))?;
                    Ok(sess)
                })
                .join()
                .unwrap()?;
                assert_eq!(sess.poll_release()?, ReleasePoll::Pending);
                assert!(!sess.peer_closed()?);
                sess.begin_drain()?;
                tx.send(()).unwrap();
                let t0 = Instant::now();
                loop {
                    match sess.poll_release()? {
                        ReleasePoll::Pending => {
                            assert!(t0.elapsed() < T, "release never arrived");
                            thread::sleep(Duration::from_millis(5));
                        }
                        ReleasePoll::Released => break,
                        ReleasePoll::PeerClosed => panic!("EOF before Release"),
                    }
                }
                assert_eq!(sess.state(), HostSessionState::Released);
                assert_eq!(sess.epoch(), Epoch([7u8; 16]));
                assert_eq!(sess.fd_count(), 2);
                Ok(())
            });
            let mut lease = WeightHostClient::connect(&sock, hello_for(&f)).unwrap();
            lease.ack_import(lease.unique_bytes()).unwrap();
            rx.recv().unwrap();
            // drain: write half shut, so the tenant sees host EOF but may still release
            let t0 = Instant::now();
            while lease.host_alive().unwrap() {
                assert!(t0.elapsed() < T, "drain never visible");
                thread::sleep(Duration::from_millis(5));
            }
            lease.close_allocation_fds();
            lease.release().unwrap();
            host.join().unwrap().unwrap();
            drop(server);
            assert!(
                sock.exists(),
                "from_listener must not unlink caller-owned sockets"
            );
        }
        assert_eq!(open_fd_count(), before);
    }

    #[test]
    fn eof_without_release_is_peer_closed_not_released() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let f = fixture(1);
        let server = Arc::new(
            WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default()).unwrap(),
        );
        let s2 = server.clone();
        let host = thread::spawn(move || -> Result<(), WeightHostError> {
            let mut sess = s2.accept(Some(T))?.admit(|_| Ok(()))?;
            sess.send_allocations()?;
            sess.wait_import_ack(Some(T))?;
            let t0 = Instant::now();
            loop {
                match sess.poll_release()? {
                    ReleasePoll::Pending => {
                        assert!(t0.elapsed() < T);
                        thread::sleep(Duration::from_millis(5));
                    }
                    ReleasePoll::PeerClosed => break,
                    ReleasePoll::Released => panic!("EOF is not a Release"),
                }
            }
            assert_eq!(sess.state(), HostSessionState::Failed);
            Ok(())
        });
        let mut lease = WeightHostClient::connect(&sock, hello_for(&f)).unwrap();
        lease.ack_import(lease.unique_bytes()).unwrap();
        drop(lease);
        host.join().unwrap().unwrap();
    }

    #[test]
    fn release_after_host_hangup_never_fabricates_delivery() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let sock = dir.path().join("h.sock");
        let f = fixture(1);
        let server = Arc::new(
            WeightHostServer::bind(&sock, f.publication.clone(), ServeOptions::default()).unwrap(),
        );
        let s2 = server.clone();
        let host = thread::spawn(move || -> Result<(), WeightHostError> {
            let mut sess = s2.accept(Some(T))?.admit(|_| Ok(()))?;
            sess.send_allocations()?;
            sess.wait_import_ack(Some(T))?;
            Ok(()) // session (both halves) closed
        });
        let mut lease = WeightHostClient::connect(&sock, hello_for(&f)).unwrap();
        lease.ack_import(lease.unique_bytes()).unwrap();
        host.join().unwrap().unwrap();
        lease.close_allocation_fds();
        // the host is gone: the lease reports it, and release() either errors (EPIPE)
        // or, if the kernel buffered the write, still leaves host_alive() false.
        assert!(!lease.host_alive().unwrap());
        let _ = lease.release();
    }

    #[test]
    fn env_bootstrap_rules() {
        let _g = fd_lock();
        // SAFETY: serialized by FD_LOCK; no other test reads these variables.
        unsafe {
            std::env::remove_var("HIPFIRE_WEIGHT_HOST");
            std::env::remove_var("HIPFIRE_WEIGHT_HOST_SOCKET");
        }
        assert_eq!(weight_host_socket_from_env().unwrap(), None);
        unsafe {
            std::env::set_var("HIPFIRE_WEIGHT_HOST_SOCKET", "/tmp/x.sock");
            std::env::set_var("HIPFIRE_WEIGHT_HOST", "0");
        }
        assert_eq!(
            weight_host_socket_from_env().unwrap(),
            None,
            "kill switch wins over socket"
        );
        unsafe { std::env::set_var("HIPFIRE_WEIGHT_HOST", "1") };
        assert_eq!(
            weight_host_socket_from_env().unwrap(),
            Some(PathBuf::from("/tmp/x.sock"))
        );
        unsafe { std::env::set_var("HIPFIRE_WEIGHT_HOST_SOCKET", "relative.sock") };
        assert!(weight_host_socket_from_env().is_err());
        unsafe { std::env::remove_var("HIPFIRE_WEIGHT_HOST_SOCKET") };
        assert!(weight_host_socket_from_env().is_err());
        for bad in ["", "true", "2", "yes"] {
            unsafe { std::env::set_var("HIPFIRE_WEIGHT_HOST", bad) };
            assert!(
                weight_host_socket_from_env().is_err(),
                "{bad:?} must fail closed"
            );
        }
        unsafe { std::env::remove_var("HIPFIRE_WEIGHT_HOST") };
    }

    #[test]
    fn dispatch_metadata_roundtrips_and_is_sealed() {
        let m = base_manifest();
        let back = WeightManifest::decode_canonical(&m.encode_canonical().unwrap()).unwrap();
        assert_eq!(back.dispatch_metadata, b"dispatch-v1");
        assert_eq!(back, m);
        let mut big = base_manifest();
        big.dispatch_metadata = vec![0u8; MAX_BLOB_BYTES + 1];
        assert!(matches!(
            big.encode_canonical(),
            Err(WeightHostError::LimitExceeded(_))
        ));
    }

    #[test]
    fn tail_pad_exact_boundary_is_accepted() {
        let _g = fd_lock();
        let mut m = base_manifest();
        m.allocations[0].logical_bytes = 77_824 - EXPORT_WEIGHT_TAIL_PAD_BYTES;
        m.views[0].byte_extent = 8000;
        m.validate().unwrap();
    }

    fn identity_report() -> String {
        let mut abi = LoaderAbi::new("abi/v1")
            .with("fwht_seed", FieldValue::U64(7))
            .with("lut", FieldValue::f32_list(&[0.0, -0.0, f32::NAN, 1.5]));
        abi.add_component_bytes("model_load.rs", b"fn load() {}");
        abi.add_component_bytes("weight_backend.rs", b"fn backend() {}");
        let model = model_content_sha256(&[
            SourceDigest {
                role: "base".into(),
                len: 4,
                sha256: Sha256Digest::of(b"AAAA"),
            },
            SourceDigest {
                role: "overlay".into(),
                len: 2,
                sha256: Sha256Digest::of(b"BB"),
            },
        ])
        .unwrap();
        let abi_sha = abi.sha256().unwrap();
        let m = base_manifest();
        let seals = compute_layout_seals(&model, &abi_sha, &base_recipe(), &m).unwrap();
        format!(
            "model={}\nabi={}\nrecipe={}\nlayout={}\nmanifest={}\n",
            model,
            abi_sha,
            seals.recipe_sha256,
            seals.layout_sha256,
            Sha256Digest::of(&m.encode_canonical().unwrap())
        )
    }

    /// Helper executed in a child process by `identity_is_deterministic_across_processes`;
    /// a no-op when run normally.
    #[test]
    fn identity_child_helper() {
        let Some(out) = std::env::var_os("HFWH_IDENTITY_CHILD_OUT") else {
            return;
        };
        std::fs::write(out, identity_report()).unwrap();
    }

    #[test]
    fn identity_is_deterministic_across_processes() {
        let _g = fd_lock();
        let dir = private_tempdir();
        let out = dir.path().join("child-identity.txt");
        let module = module_path!().split_once("::").unwrap().1;
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("{module}::identity_child_helper"),
                "--test-threads=1",
            ])
            .env("HFWH_IDENTITY_CHILD_OUT", &out)
            .env("RUST_BACKTRACE", "0")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "child test process failed");
        let child = std::fs::read_to_string(&out).expect("child did not write its identity report");
        assert_eq!(
            child,
            identity_report(),
            "identity digests differ across processes"
        );
    }
}
