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
    let n_fixed = take_usize(cur)?;
    let fixed_bytes = (0..n_fixed)
        .map(|_| take_usize(cur))
        .collect::<Option<Vec<_>>>()?;
    let n_segments = take_usize(cur)?;
    let segments = (0..n_segments)
        .map(|_| {
            Some(Segment {
                row_bytes: take_usize(cur)?,
                from: take_usize(cur)?,
                to: take_usize(cur)?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let meta = take_bytes(cur)?.to_vec();
    let bytes = take_u64(cur)?;
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
    if len != payload_offset(header_len) + snapshot.bytes {
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

    /// Write the device `snapshot` at `key` to disk.
    pub(super) fn demote(
        &mut self,
        gpu: &mut Gpu,
        key: &Key,
        snapshot: &StoredSnapshot,
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
        let staging = staging(&mut self.staging);
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

fn staging(buf: &mut Vec<u8>) -> &mut [u8] {
    if buf.is_empty() {
        buf.resize(DISK_STAGING, 0);
    }
    buf
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
    let staging = staging(staging_buf);
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
