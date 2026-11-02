// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Disk tier of the session cache: snapshots evicted from device memory are
//! written to one file each under a fixed byte budget and promoted back on
//! restore. Files are bound to the daemon executable's SHA-256, so another
//! build's snapshots are never planned and are evicted first.
//!
//! File: `b"HFSNAP01"`, 32-byte build digest, u64 header length, header,
//! zero padding to a 4096-byte boundary, then the device snapshot's bytes in
//! their packed [`super::layout`] order. Integers are little-endian u64.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::SystemTime;

use hip_bridge::DeviceBuffer;
use rdna_compute::Gpu;
use sha2::{Digest, Sha256};

use super::{Key, Segment, SnapshotLocation, StoredSnapshot};
use crate::checkpoint_pool::{CheckpointBlob, CheckpointPool};
use crate::serve_contract::CacheDomain;

const MAGIC: &[u8; 8] = b"HFSNAP01";
/// Magic, build digest, header length.
const PREAMBLE: u64 = 8 + 32 + 8;
const PAYLOAD_ALIGN: u64 = 4096;
const MAX_HEADER: u64 = 1 << 20;
/// Host bounce buffer between device and file.
const DISK_STAGING: usize = 64 << 20;

/// SHA-256 of the running executable, computed once.
pub(super) fn build_digest() -> Result<[u8; 32], String> {
    static DIGEST: LazyLock<Result<[u8; 32], String>> = LazyLock::new(|| {
        let path = std::env::current_exe().map_err(|e| e.to_string())?;
        let fail = |e: std::io::Error| format!("{}: {e}", path.display());
        let mut file = File::open(&path).map_err(fail)?;
        let mut hasher = Sha256::new();
        let mut chunk = vec![0u8; 1 << 20];
        loop {
            let n = file.read(&mut chunk).map_err(fail)?;
            if n == 0 {
                break;
            }
            hasher.update(&chunk[..n]);
        }
        Ok(hasher.finalize().into())
    });
    DIGEST.clone()
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_u64(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn take<'a>(cur: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    if cur.len() < n {
        return None;
    }
    let (head, rest) = cur.split_at(n);
    *cur = rest;
    Some(head)
}

fn take_u64(cur: &mut &[u8]) -> Option<u64> {
    Some(u64::from_le_bytes(take(cur, 8)?.try_into().ok()?))
}

fn take_usize(cur: &mut &[u8]) -> Option<usize> {
    usize::try_from(take_u64(cur)?).ok()
}

fn take_bytes<'a>(cur: &mut &'a [u8]) -> Option<&'a [u8]> {
    let n = take_usize(cur)?;
    take(cur, n)
}

fn take_key(cur: &mut &[u8]) -> Option<Key> {
    let domain = CacheDomain::from_canonical_bytes(take_bytes(cur)?).ok()?;
    Some((domain, take_u64(cur)?, take_u64(cur)?))
}

fn put_key(out: &mut Vec<u8>, key: &Key) {
    put_bytes(out, &key.0.to_canonical_bytes());
    put_u64(out, key.1);
    put_u64(out, key.2);
}

fn encode_header(key: &Key, snapshot: &StoredSnapshot) -> Vec<u8> {
    let mut out = Vec::new();
    put_key(&mut out, key);
    match &snapshot.parent {
        Some(parent) => {
            out.push(1);
            put_key(&mut out, parent);
        }
        None => out.push(0),
    }
    put_u64(&mut out, snapshot.fixed_bytes.len() as u64);
    for &bytes in &snapshot.fixed_bytes {
        put_u64(&mut out, bytes as u64);
    }
    put_u64(&mut out, snapshot.segments.len() as u64);
    for segment in &snapshot.segments {
        put_u64(&mut out, segment.row_bytes as u64);
        put_u64(&mut out, segment.from as u64);
        put_u64(&mut out, segment.to as u64);
    }
    put_bytes(&mut out, &snapshot.meta);
    put_u64(&mut out, snapshot.bytes);
    out
}

/// Key and snapshot of a header; `location` is a placeholder the caller sets.
fn decode_header(mut cur: &[u8]) -> Option<(Key, StoredSnapshot)> {
    let cur = &mut cur;
    let key = take_key(cur)?;
    let parent = match take(cur, 1)?[0] {
        0 => None,
        1 => Some(take_key(cur)?),
        _ => return None,
    };
    if parent
        .as_ref()
        .is_some_and(|p| p.0 != key.0 || p.1 >= key.1)
    {
        return None;
    }
    // Mirror super::layout, but reject overflow before its unchecked helpers
    // can see data from disk.
    let mut packed_bytes = 0usize;
    let mut add_part = |bytes: usize| -> Option<()> {
        packed_bytes = packed_bytes
            .checked_next_multiple_of(super::PART_ALIGN)?
            .checked_add(bytes)?;
        Some(())
    };
    let n_fixed = take_usize(cur)?;
    if n_fixed > cur.len() / 8 {
        return None;
    }
    let fixed_bytes = (0..n_fixed)
        .map(|_| {
            let bytes = take_usize(cur)?;
            add_part(bytes)?;
            Some(bytes)
        })
        .collect::<Option<Vec<_>>>()?;
    let n_segments = take_usize(cur)?;
    if n_segments > cur.len() / 24 {
        return None;
    }
    let segments = (0..n_segments)
        .map(|_| {
            let segment = Segment {
                row_bytes: take_usize(cur)?,
                from: take_usize(cur)?,
                to: take_usize(cur)?,
            };
            let bytes = segment
                .to
                .checked_sub(segment.from)?
                .checked_mul(segment.row_bytes)?;
            add_part(bytes)?;
            Some(segment)
        })
        .collect::<Option<Vec<_>>>()?;
    let meta = take_bytes(cur)?.to_vec();
    let bytes = take_u64(cur)?;
    if !cur.is_empty() || usize::try_from(bytes).ok()? != packed_bytes {
        return None;
    }
    let snapshot = StoredSnapshot {
        location: SnapshotLocation::Disk {
            path: PathBuf::new(),
            file_bytes: 0,
        },
        parent,
        fixed_bytes,
        segments,
        meta,
        bytes,
    };
    Some((key, snapshot))
}

fn payload_offset(header_len: u64) -> u64 {
    (PREAMBLE + header_len).next_multiple_of(PAYLOAD_ALIGN)
}

/// First 32 hex chars of `sha256(build ‖ domain ‖ p ‖ fp)`.
fn file_stem(build: &[u8; 32], key: &Key) -> String {
    let mut hasher = Sha256::new();
    hasher.update(build);
    hasher.update(key.0.to_canonical_bytes());
    hasher.update(key.1.to_le_bytes());
    hasher.update(key.2.to_le_bytes());
    hasher.finalize()[..16]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn remove_file(snapshot: &StoredSnapshot) {
    if let SnapshotLocation::Disk { path, .. } = &snapshot.location {
        let _ = fs::remove_file(path);
    }
}

enum Scanned {
    Foreign(u64),
    Current(Box<(Key, StoredSnapshot)>),
}

/// Classify one `.snap` file; `None` = corrupt.
fn scan(path: &Path, build: &[u8; 32]) -> Option<Scanned> {
    let mut file = File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    let mut preamble = [0u8; PREAMBLE as usize];
    file.read_exact(&mut preamble).ok()?;
    if &preamble[..8] != MAGIC {
        return None;
    }
    if preamble[8..40] != build[..] {
        return Some(Scanned::Foreign(len));
    }
    let header_len = u64::from_le_bytes(preamble[40..].try_into().ok()?);
    if header_len > MAX_HEADER {
        return None;
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header).ok()?;
    let (key, mut snapshot) = decode_header(&header)?;
    if len != payload_offset(header_len).checked_add(snapshot.bytes)? {
        return None;
    }
    snapshot.location = SnapshotLocation::Disk {
        path: path.to_path_buf(),
        file_bytes: len,
    };
    Some(Scanned::Current(Box::new((key, snapshot))))
}

fn mtime(path: &Path) -> SystemTime {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Snapshots on disk, one file each, under one byte budget for the whole
/// directory (every build's files count).
///
/// Invariants, with the device tier in [`super::SessionCache`]:
/// - I1: the parent of every device or pending snapshot is on the device.
///   The device tier pins any key with device or pending children, and
///   capture picks parents only among device and pending snapshots.
/// - I2: the parent of every disk snapshot is on the device or on disk. A
///   disk key is pinned in [`Self::pool`] exactly when it has disk kids (or
///   a restore holds it); by I1 a disk key never has device children.
pub(super) struct DiskTier {
    dir: PathBuf,
    /// `flock` on `dir/lock`, held for the tier's lifetime.
    _lock: File,
    build: [u8; 32],
    budget: u64,
    /// This build's files; every location is `Disk`.
    pub(super) pool: CheckpointPool<StoredSnapshot>,
    /// Parent key (either tier) -> its children on disk.
    kids: HashMap<Key, Vec<Key>>,
    /// Chain keys a restore in progress uses; pinned until it clears them.
    pub(super) held: HashSet<Key>,
    /// Other builds' files, oldest first.
    foreign: VecDeque<(PathBuf, u64)>,
    foreign_bytes: u64,
    /// Allocated on first use.
    staging: Vec<u8>,
}

impl DiskTier {
    pub(super) fn open(dir: &Path, budget: u64, build: [u8; 32]) -> Result<Self, String> {
        let fail = |e: std::io::Error| format!("{}: {e}", dir.display());
        fs::create_dir_all(dir).map_err(fail)?;
        let lock = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join("lock"))
            .map_err(fail)?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => {
                return Err(format!("{} is locked by another process", dir.display()))
            }
            Err(fs::TryLockError::Error(e)) => return Err(fail(e)),
        }

        let mut foreign = Vec::new();
        let mut current = Vec::new();
        for entry in fs::read_dir(dir).map_err(fail)? {
            let Ok(entry) = entry else { continue };
            let path = entry.path();
            match path.extension().and_then(|e| e.to_str()) {
                Some("tmp") => {
                    let _ = fs::remove_file(&path);
                }
                Some("snap") => match scan(&path, &build) {
                    Some(Scanned::Foreign(len)) => foreign.push((mtime(&path), path, len)),
                    Some(Scanned::Current(entry)) => {
                        let (key, snapshot) = *entry;
                        current.push((mtime(&path), key, snapshot))
                    }
                    None => {
                        let _ = fs::remove_file(&path);
                    }
                },
                _ => {}
            }
        }
        foreign.sort_by_key(|entry| entry.0);
        current.sort_by_key(|entry| entry.0);
        let mut foreign: VecDeque<(PathBuf, u64)> = foreign
            .into_iter()
            .map(|(_, path, len)| (path, len))
            .collect();
        let mut foreign_bytes: u64 = foreign.iter().map(|entry| entry.1).sum();
        let mut current: VecDeque<(Key, StoredSnapshot)> = current
            .into_iter()
            .map(|(_, key, snapshot)| (key, snapshot))
            .collect();
        let mut current_bytes: u64 = current.iter().map(|(_, s)| s.bytes_len()).sum();

        // Trim to the budget: other builds first, then oldest.
        while current_bytes + foreign_bytes > budget {
            if let Some((path, len)) = foreign.pop_front() {
                let _ = fs::remove_file(path);
                foreign_bytes -= len;
            } else if let Some((_, snapshot)) = current.pop_front() {
                current_bytes -= snapshot.bytes_len();
                remove_file(&snapshot);
            } else {
                break;
            }
        }

        // Drop orphans; a parent always has a smaller p than its child.
        let mut by_p: Vec<usize> = (0..current.len()).collect();
        by_p.sort_by_key(|&i| current[i].0 .1);
        let mut kept = HashSet::new();
        for i in by_p {
            let (key, snapshot) = &current[i];
            if snapshot.parent.as_ref().is_none_or(|p| kept.contains(p)) {
                kept.insert(key.clone());
            }
        }

        let mut tier = Self {
            dir: dir.to_path_buf(),
            _lock: lock,
            build,
            budget,
            pool: CheckpointPool::new(budget),
            kids: HashMap::new(),
            held: HashSet::new(),
            foreign,
            foreign_bytes,
            staging: Vec::new(),
        };
        // Ascending mtime, so the pool's LRU order matches the files'.
        for (key, snapshot) in current {
            if !kept.contains(&key) {
                remove_file(&snapshot);
                continue;
            }
            if let Some(parent) = &snapshot.parent {
                tier.kids
                    .entry(parent.clone())
                    .or_default()
                    .push(key.clone());
            }
            let (domain, p, fp) = key;
            let (_, displaced) = tier.pool.insert_unaligned(domain, p, fp, snapshot);
            debug_assert!(displaced.is_empty(), "trimmed to the budget");
        }
        for parent in tier.kids.keys() {
            tier.pool.pin(&parent.0, parent.1, parent.2);
        }
        Ok(tier)
    }

    pub(super) fn contains(&self, key: &Key) -> bool {
        self.pool.contains(&key.0, key.1, key.2)
    }

    pub(super) fn peek(&self, key: &Key) -> Option<&StoredSnapshot> {
        self.pool.peek(&key.0, key.1, key.2)
    }

    pub(super) fn foreign_files(&self) -> usize {
        self.foreign.len()
    }

    fn add_kid(&mut self, parent: &Key, child: Key) {
        self.kids.entry(parent.clone()).or_default().push(child);
        self.pool.pin(&parent.0, parent.1, parent.2);
    }

    fn remove_kid(&mut self, parent: &Key, child: &Key) {
        let Some(list) = self.kids.get_mut(parent) else {
            return;
        };
        list.retain(|kid| kid != child);
        if list.is_empty() {
            self.kids.remove(parent);
            if !self.held.contains(parent) {
                self.pool.unpin(&parent.0, parent.1, parent.2);
            }
        }
    }

    /// Re-derive `key`'s pin from `held` and `kids`.
    pub(super) fn repin(&mut self, key: &Key) {
        if self.held.contains(key) || self.kids.contains_key(key) {
            self.pool.pin(&key.0, key.1, key.2);
        } else {
            self.pool.unpin(&key.0, key.1, key.2);
        }
    }

    /// Evict until `need` more bytes fit: other builds' files, then LRU
    /// leaves. `false` when only pinned files are left.
    fn make_room(&mut self, need: u64) -> bool {
        while self.pool.total_bytes() + self.foreign_bytes + need > self.budget {
            if let Some((path, len)) = self.foreign.pop_front() {
                let _ = fs::remove_file(path);
                self.foreign_bytes -= len;
                continue;
            }
            let Some((key, snapshot)) = self.pool.pop_lru() else {
                return false;
            };
            if let Some(parent) = &snapshot.parent {
                self.remove_kid(parent, &key);
            }
            remove_file(&snapshot);
        }
        true
    }

    /// Allocate the host bounce buffer on first use. Every device/file copy
    /// ([`Self::demote`], [`Self::load`]) needs it, so each caller reserves it
    /// first: refused (nothing allocated) when host RAM cannot hold it beside
    /// the memory guard's headroom and the live state's `growth` reserve, or
    /// when the allocator fails.
    pub(super) fn reserve_staging(&mut self, gpu: &Gpu, growth: u64) -> Result<(), String> {
        if !self.staging.is_empty() {
            return Ok(());
        }
        if !super::host_staging_fits(gpu, DISK_STAGING as u64, growth) {
            return Err("host memory guard".into());
        }
        self.staging
            .try_reserve_exact(DISK_STAGING)
            .map_err(|e| format!("host staging allocation: {e}"))?;
        self.staging.resize(DISK_STAGING, 0);
        Ok(())
    }

    /// Write the device `snapshot` at `key` to disk.
    pub(super) fn demote(
        &mut self,
        gpu: &mut Gpu,
        key: &Key,
        snapshot: &StoredSnapshot,
        growth: u64,
    ) -> Result<(), String> {
        let SnapshotLocation::Device(buf) = &snapshot.location else {
            return Err("not a device snapshot".into());
        };
        let header = encode_header(key, snapshot);
        let offset = payload_offset(header.len() as u64);
        let file_bytes = offset + snapshot.bytes;
        if file_bytes > self.budget {
            return Err("exceeds disk budget".into());
        }
        self.reserve_staging(gpu, growth)?;
        if !self.make_room(file_bytes) {
            return Err("disk budget pinned".into());
        }
        let stem = file_stem(&self.build, key);
        let tmp = self.dir.join(format!("{stem}.tmp"));
        let path = self.dir.join(format!("{stem}.snap"));
        let mut prefix = Vec::with_capacity(offset as usize);
        prefix.extend_from_slice(MAGIC);
        prefix.extend_from_slice(&self.build);
        put_u64(&mut prefix, header.len() as u64);
        prefix.extend_from_slice(&header);
        prefix.resize(offset as usize, 0);
        let written = write_file(
            gpu,
            &mut self.staging,
            &tmp,
            &prefix,
            buf,
            snapshot.bytes as usize,
        )
        .and_then(|()| fs::rename(&tmp, &path).map_err(|e| e.to_string()))
        .and_then(|()| sync_dir(&self.dir).map_err(|e| e.to_string()));
        if let Err(error) = written {
            let _ = fs::remove_file(&tmp);
            return Err(error);
        }
        let disk = StoredSnapshot {
            location: SnapshotLocation::Disk { path, file_bytes },
            parent: snapshot.parent.clone(),
            fixed_bytes: snapshot.fixed_bytes.clone(),
            segments: snapshot.segments.clone(),
            meta: snapshot.meta.clone(),
            bytes: snapshot.bytes,
        };
        let (domain, p, fp) = key.clone();
        let (_, displaced) = self.pool.insert_unaligned(domain, p, fp, disk);
        debug_assert!(displaced.is_empty(), "make_room reserved the bytes");
        if let Some(parent) = &snapshot.parent {
            self.add_kid(parent, key.clone());
        }
        if self.kids.contains_key(key) {
            self.pool.pin(&key.0, key.1, key.2);
        }
        Ok(())
    }

    /// The disk snapshot at `key`, refreshing its LRU stamp and file mtime.
    pub(super) fn get(&mut self, key: &Key) -> Option<&StoredSnapshot> {
        let snapshot = self.pool.get(&key.0, key.1, key.2)?;
        if let SnapshotLocation::Disk { path, .. } = &snapshot.location {
            if let Ok(file) = File::options().write(true).open(path) {
                let _ = file.set_modified(SystemTime::now());
            }
        }
        Some(snapshot)
    }

    /// Copy the payload of `key` into `dst`.
    pub(super) fn load(
        &mut self,
        gpu: &mut Gpu,
        key: &Key,
        dst: &DeviceBuffer,
    ) -> Result<(), String> {
        let snapshot = self.pool.peek(&key.0, key.1, key.2).ok_or("not on disk")?;
        let SnapshotLocation::Disk { path, file_bytes } = &snapshot.location else {
            return Err("not on disk".into());
        };
        let fail = |e: std::io::Error| format!("{}: {e}", path.display());
        let mut file = File::open(path).map_err(fail)?;
        if file.metadata().map_err(fail)?.len() != *file_bytes {
            return Err(format!("{}: size changed", path.display()));
        }
        file.seek(SeekFrom::Start(file_bytes - snapshot.bytes))
            .map_err(fail)?;
        if self.staging.is_empty() {
            return Err(unreserved());
        }
        let staging = &mut self.staging[..];
        let bytes = snapshot.bytes as usize;
        let mut off = 0;
        while off < bytes {
            let n = (bytes - off).min(DISK_STAGING);
            file.read_exact(&mut staging[..n]).map_err(fail)?;
            gpu.hip
                .memcpy_htod_offset(dst, off, &staging[..n])
                .map_err(|e| e.to_string())?;
            off += n;
        }
        Ok(())
    }

    /// Remove `key` after it was promoted to the device. Its `kids` entry
    /// stays: those disk children now name a device parent.
    pub(super) fn take(&mut self, key: &Key) {
        if let Some(snapshot) = self.pool.evict(&key.0, key.1, key.2) {
            if let Some(parent) = &snapshot.parent {
                self.remove_kid(parent, key);
            }
            remove_file(&snapshot);
        }
    }

    /// Delete every disk descendant of `key`.
    pub(super) fn drop_descendants(&mut self, key: &Key) {
        for child in self.kids.remove(key).unwrap_or_default() {
            self.drop_descendants(&child);
            if let Some(snapshot) = self.pool.evict(&child.0, child.1, child.2) {
                remove_file(&snapshot);
            }
            self.held.remove(&child);
        }
        self.repin(key);
    }

    /// Delete `key` and its disk descendants.
    pub(super) fn drop_entry(&mut self, key: &Key) {
        self.drop_descendants(key);
        self.take(key);
    }
}

/// A host bounce buffer is only ever allocated by [`DiskTier::reserve_staging`].
fn unreserved() -> String {
    "host staging buffer not reserved".into()
}

/// `prefix`, then `bytes` of `src`, into a new file at `path`, synced.
fn write_file(
    gpu: &mut Gpu,
    staging_buf: &mut Vec<u8>,
    path: &Path,
    prefix: &[u8],
    src: &DeviceBuffer,
    bytes: usize,
) -> Result<(), String> {
    let fail = |e: std::io::Error| format!("{}: {e}", path.display());
    gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
    let mut file = File::create(path).map_err(fail)?;
    file.write_all(prefix).map_err(fail)?;
    if staging_buf.is_empty() {
        return Err(unreserved());
    }
    let staging = &mut staging_buf[..];
    let mut off = 0;
    while off < bytes {
        let n = (bytes - off).min(DISK_STAGING);
        gpu.hip
            .memcpy_dtoh_at(&mut staging[..n], src, off)
            .map_err(|e| e.to_string())?;
        file.write_all(&staging[..n]).map_err(fail)?;
        off += n;
    }
    file.sync_all().map_err(fail)
}

#[cfg(unix)]
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve_contract::{
        ArchPolicy, DeviceTopology, KvLayout, SharingNamespace, TemplateIdentity, TokenizerIdentity,
    };

    const BUILD: [u8; 32] = [17; 32];
    const OTHER_BUILD: [u8; 32] = [23; 32];

    fn domain() -> CacheDomain {
        CacheDomain {
            model_content_digest: vec![1],
            model_load_epoch: 1,
            sidecar_digests: vec![],
            tokenizer: TokenizerIdentity {
                vocab_digest: vec![2],
                config_digest: vec![3],
            },
            template: TemplateIdentity {
                template_digest: vec![4],
                normalization_tag: "jinja".into(),
            },
            arch_policy: ArchPolicy {
                arch_tag: "toy".into(),
                state_abi_tag: String::new(),
                position_attention_tag: String::new(),
            },
            kv_layout: KvLayout {
                k_stride_bytes: vec![],
                v_stride_bytes: vec![],
                layout_tag: String::new(),
            },
            device: DeviceTopology {
                device_id: "gpu-0".into(),
                topology_id: "single".into(),
                allocation_epoch: 1,
            },
            namespace: SharingNamespace {
                domain_id: "default".into(),
            },
        }
    }

    fn key(p: u64) -> Key {
        (domain(), p, p + 1)
    }

    fn snapshot(parent: Option<Key>) -> StoredSnapshot {
        let fixed_bytes = vec![3, 256, 0];
        let segments = vec![
            Segment {
                row_bytes: 16,
                from: 0,
                to: 2,
            },
            Segment {
                row_bytes: 0,
                from: 5,
                to: 5,
            },
            Segment {
                row_bytes: 7,
                from: 1,
                to: 4,
            },
        ];
        let bytes = super::super::layout(
            fixed_bytes
                .iter()
                .copied()
                .chain(segments.iter().map(Segment::bytes)),
        )
        .1 as u64;
        StoredSnapshot {
            location: SnapshotLocation::Disk {
                path: PathBuf::new(),
                file_bytes: 0,
            },
            parent,
            fixed_bytes,
            segments,
            meta: vec![0, 1, 255, 7],
            bytes,
        }
    }

    fn assert_snapshot_eq(actual: &StoredSnapshot, expected: &StoredSnapshot) {
        assert_eq!(actual.parent, expected.parent);
        assert_eq!(actual.fixed_bytes, expected.fixed_bytes);
        assert!(actual.segments == expected.segments);
        assert_eq!(actual.meta, expected.meta);
        assert_eq!(actual.bytes, expected.bytes);
        assert_eq!(actual.offsets(), expected.offsets());
    }

    // CPU-only file authoring: no DeviceBuffer, Gpu or HIP initialization.
    fn write_header(path: &Path, build: &[u8; 32], header: &[u8]) -> File {
        let mut file = File::create(path).unwrap();
        file.write_all(MAGIC).unwrap();
        file.write_all(build).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file
    }

    fn write_snapshot(
        dir: &Path,
        build: &[u8; 32],
        key: &Key,
        snapshot: &StoredSnapshot,
    ) -> PathBuf {
        let path = dir.join(format!("{}.snap", file_stem(build, key)));
        let header = encode_header(key, snapshot);
        let mut file = write_header(&path, build, &header);
        file.seek(SeekFrom::Start(payload_offset(header.len() as u64)))
            .unwrap();
        file.write_all(&vec![0x5a; snapshot.bytes as usize])
            .unwrap();
        path
    }

    #[test]
    fn header_roundtrip_root_and_child() {
        for parent in [None, Some(key(128))] {
            let key = key(256);
            let snapshot = snapshot(parent);
            let encoded = encode_header(&key, &snapshot);
            let (decoded_key, decoded) = decode_header(&encoded).unwrap();
            assert_eq!(decoded_key, key);
            assert_snapshot_eq(&decoded, &snapshot);
            assert_eq!(encode_header(&decoded_key, &decoded), encoded);
            match decoded.location {
                SnapshotLocation::Disk { path, file_bytes } => {
                    assert!(path.as_os_str().is_empty());
                    assert_eq!(file_bytes, 0);
                }
                _ => panic!("decoded header must have a disk placeholder"),
            }
        }
    }

    #[test]
    fn header_rejects_truncation_trailing_bytes_and_bad_tags() {
        let key = key(256);
        for parent in [None, Some(self::key(128))] {
            let encoded = encode_header(&key, &snapshot(parent));
            for end in 0..encoded.len() {
                assert!(
                    decode_header(&encoded[..end]).is_none(),
                    "truncated at {end}"
                );
            }
            let mut trailing = encoded;
            trailing.push(0);
            assert!(decode_header(&trailing).is_none());
        }
        let mut prefix = Vec::new();
        put_key(&mut prefix, &key);
        let parent_tag = prefix.len();
        let mut bad_tag = encode_header(&key, &snapshot(None));
        bad_tag[parent_tag] = 2;
        assert!(decode_header(&bad_tag).is_none());
        let mut bad_domain = Vec::new();
        put_bytes(&mut bad_domain, b"not a canonical cache domain");
        assert!(decode_header(&bad_domain).is_none());
        assert!(decode_header(&u64::MAX.to_le_bytes()).is_none());
    }

    #[test]
    fn header_rejects_unbounded_counts() {
        for count in [u64::MAX, MAX_HEADER] {
            let mut fixed = Vec::new();
            put_key(&mut fixed, &key(128));
            fixed.push(0);
            put_u64(&mut fixed, count);
            assert!(decode_header(&fixed).is_none());
            let mut segments = Vec::new();
            put_key(&mut segments, &key(128));
            segments.push(0);
            put_u64(&mut segments, 0);
            put_u64(&mut segments, count);
            assert!(decode_header(&segments).is_none());
        }
    }

    #[test]
    fn header_rejects_invalid_segments_and_layout_overflow() {
        let key = key(128);
        for segment in [
            Segment {
                row_bytes: 1,
                from: 2,
                to: 1,
            },
            Segment {
                row_bytes: usize::MAX,
                from: 0,
                to: 2,
            },
        ] {
            let mut snapshot = snapshot(None);
            snapshot.fixed_bytes.clear();
            snapshot.segments = vec![segment];
            snapshot.bytes = 0;
            assert!(decode_header(&encode_header(&key, &snapshot)).is_none());
        }
        // Overflow in alignment, addition, and fixed-to-segment alignment.
        for (fixed, segments) in [
            (vec![usize::MAX, 0], vec![]),
            (
                vec![usize::MAX - (super::super::PART_ALIGN - 1), 256],
                vec![],
            ),
            (
                vec![usize::MAX],
                vec![Segment {
                    row_bytes: 0,
                    from: 0,
                    to: 0,
                }],
            ),
        ] {
            let mut snapshot = snapshot(None);
            snapshot.fixed_bytes = fixed;
            snapshot.segments = segments;
            snapshot.bytes = 0;
            assert!(decode_header(&encode_header(&key, &snapshot)).is_none());
        }
    }

    #[test]
    fn header_rejects_size_mismatch_and_invalid_parent() {
        let key = key(256);
        let mut wrong_size = snapshot(None);
        wrong_size.bytes += 1;
        assert!(decode_header(&encode_header(&key, &wrong_size)).is_none());
        wrong_size.bytes = u64::MAX;
        assert!(decode_header(&encode_header(&key, &wrong_size)).is_none());
        let mut foreign_domain = domain();
        foreign_domain.namespace.domain_id = "another namespace".into();
        for parent in [key.clone(), self::key(512), (foreign_domain, 128, 129)] {
            assert!(decode_header(&encode_header(&key, &snapshot(Some(parent)))).is_none());
        }
    }

    #[test]
    fn scan_rejects_payload_size_and_offset_overflow() {
        let dir = tempfile::tempdir().unwrap();
        let key = key(128);
        let snapshot = snapshot(None);
        let path = write_snapshot(dir.path(), &BUILD, &key, &snapshot);
        let len = fs::metadata(&path).unwrap().len();
        let file = File::options().write(true).open(&path).unwrap();
        for invalid_len in [len - 1, len + 1] {
            file.set_len(invalid_len).unwrap();
            assert!(scan(&path, &BUILD).is_none());
        }
        let mut huge = self::snapshot(None);
        huge.fixed_bytes = vec![usize::MAX];
        huge.segments.clear();
        huge.bytes = usize::MAX as u64;
        let header = encode_header(&key, &huge);
        // A single maximal part is a valid layout, but cannot be a file on
        // 64-bit targets once its nonzero payload offset is included.
        assert!(decode_header(&header).is_some());
        drop(write_header(&path, &BUILD, &header));
        assert!(scan(&path, &BUILD).is_none());
    }

    #[test]
    fn startup_indexes_valid_files_and_cleans_malformed_files() {
        let dir = tempfile::tempdir().unwrap();
        let root = key(128);
        let child = key(256);
        let root_snapshot = snapshot(None);
        let child_snapshot = snapshot(Some(root.clone()));
        let root_path = write_snapshot(dir.path(), &BUILD, &root, &root_snapshot);
        let child_path = write_snapshot(dir.path(), &BUILD, &child, &child_snapshot);
        let bad_magic = dir.path().join("bad-magic.snap");
        fs::write(&bad_magic, b"not a snapshot").unwrap();
        let trailing = dir.path().join("trailing.snap");
        let mut header = encode_header(&key(512), &snapshot(None));
        header.push(0);
        drop(write_header(&trailing, &BUILD, &header));
        let oversized = dir.path().join("oversized.snap");
        let mut file = write_header(&oversized, &BUILD, &[]);
        file.seek(SeekFrom::Start(40)).unwrap();
        file.write_all(&(MAX_HEADER + 1).to_le_bytes()).unwrap();
        drop(file);
        let truncated = dir.path().join("truncated.snap");
        drop(write_header(
            &truncated,
            &BUILD,
            &encode_header(&key(512), &snapshot(None)),
        ));
        let temporary = dir.path().join("interrupted.tmp");
        fs::write(&temporary, b"unfinished").unwrap();
        let unrelated = dir.path().join("keep.txt");
        fs::write(&unrelated, b"unrelated").unwrap();

        let tier = DiskTier::open(dir.path(), u64::MAX, BUILD).unwrap();
        assert!(tier.contains(&root));
        assert!(tier.contains(&child));
        assert_snapshot_eq(tier.peek(&root).unwrap(), &root_snapshot);
        assert_snapshot_eq(tier.peek(&child).unwrap(), &child_snapshot);
        assert_eq!(tier.kids.get(&root), Some(&vec![child.clone()]));
        let total =
            fs::metadata(&root_path).unwrap().len() + fs::metadata(&child_path).unwrap().len();
        assert_eq!(tier.pool.total_bytes(), total);
        assert_eq!(tier.foreign_files(), 0);
        assert!(tier.staging.is_empty());
        for path in [bad_magic, trailing, oversized, truncated, temporary] {
            assert!(!path.exists(), "not cleaned: {}", path.display());
        }
        assert!(unrelated.exists());
    }

    #[test]
    fn startup_isolates_other_builds_and_reopens_after_drop() {
        let dir = tempfile::tempdir().unwrap();
        let root = key(128);
        let child = key(256);
        let root_path = write_snapshot(dir.path(), &BUILD, &root, &snapshot(None));
        let child_path = write_snapshot(dir.path(), &BUILD, &child, &snapshot(Some(root.clone())));
        let tier = DiskTier::open(dir.path(), u64::MAX, OTHER_BUILD).unwrap();
        assert!(!tier.contains(&root));
        assert!(!tier.contains(&child));
        assert_eq!(tier.pool.total_bytes(), 0);
        assert_eq!(tier.foreign_files(), 2);
        assert_eq!(
            tier.foreign_bytes,
            fs::metadata(&root_path).unwrap().len() + fs::metadata(&child_path).unwrap().len()
        );
        drop(tier);
        let tier = DiskTier::open(dir.path(), u64::MAX, BUILD).unwrap();
        assert!(tier.contains(&root));
        assert!(tier.contains(&child));
        assert_eq!(tier.foreign_files(), 0);
        let mut file = File::open(&child_path).unwrap();
        let offset = payload_offset(encode_header(&child, &snapshot(Some(root))).len() as u64);
        file.seek(SeekFrom::Start(offset)).unwrap();
        let mut payload = Vec::new();
        file.read_to_end(&mut payload).unwrap();
        assert_eq!(
            payload,
            vec![0x5a; tier.peek(&child).unwrap().bytes as usize]
        );
    }

    #[test]
    fn startup_removes_orphans_and_their_descendants() {
        let dir = tempfile::tempdir().unwrap();
        let missing = key(128);
        let orphan = key(256);
        let descendant = key(512);
        let root = key(1024);
        let orphan_path = write_snapshot(
            dir.path(),
            &BUILD,
            &orphan,
            &snapshot(Some(missing.clone())),
        );
        let descendant_path = write_snapshot(
            dir.path(),
            &BUILD,
            &descendant,
            &snapshot(Some(orphan.clone())),
        );
        let foreign_path = write_snapshot(dir.path(), &OTHER_BUILD, &missing, &snapshot(None));
        let root_path = write_snapshot(dir.path(), &BUILD, &root, &snapshot(None));
        let tier = DiskTier::open(dir.path(), u64::MAX, BUILD).unwrap();
        assert!(!tier.contains(&missing));
        assert!(!tier.contains(&orphan));
        assert!(!tier.contains(&descendant));
        assert!(!orphan_path.exists());
        assert!(!descendant_path.exists());
        assert!(tier.contains(&root));
        assert!(root_path.exists());
        assert!(foreign_path.exists());
        assert_eq!(tier.foreign_files(), 1);
        assert!(tier.kids.is_empty());
    }

    #[test]
    fn directory_lock_refuses_concurrent_open_and_releases_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let root = key(128);
        write_snapshot(dir.path(), &BUILD, &root, &snapshot(None));
        let tier = DiskTier::open(dir.path(), u64::MAX, BUILD).unwrap();
        let path = dir.path().to_path_buf();
        let error = std::thread::spawn(move || {
            DiskTier::open(&path, u64::MAX, BUILD)
                .err()
                .expect("second open must fail")
        })
        .join()
        .unwrap();
        assert!(error.contains("locked by another process"), "{error}");
        drop(tier);
        let reopened = DiskTier::open(dir.path(), u64::MAX, BUILD).unwrap();
        assert!(reopened.contains(&root));
    }

    #[test]
    fn startup_budget_counts_file_padding_and_evicts_foreign_first() {
        let dir = tempfile::tempdir().unwrap();
        let root = key(128);
        let current = write_snapshot(dir.path(), &BUILD, &root, &snapshot(None));
        let foreign = write_snapshot(dir.path(), &OTHER_BUILD, &key(256), &snapshot(None));
        let file_bytes = fs::metadata(&current).unwrap().len();
        let tier = DiskTier::open(dir.path(), file_bytes, BUILD).unwrap();
        assert!(tier.contains(&root));
        assert_eq!(tier.pool.total_bytes(), file_bytes);
        assert_eq!(tier.foreign_files(), 0);
        assert_eq!(tier.foreign_bytes, 0);
        assert!(!foreign.exists());
        assert!(current.exists());
        drop(tier);
        let tier = DiskTier::open(dir.path(), file_bytes - 1, BUILD).unwrap();
        assert!(!tier.contains(&root));
        assert_eq!(tier.pool.total_bytes(), 0);
        assert!(!current.exists());
    }
}
