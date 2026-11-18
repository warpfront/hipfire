// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The explicit file-segment source mode of the expert cache (`CacheSources`): positional reads
//! of expert bundle segments from the model file into one host staging buffer, each followed by
//! a blocking H2D copy into a caller-owned device allocation at a byte offset.
//!
//! # Contract
//!
//! * Explicit only. Nothing here runs on a first use the caller did not ask for: a caller names
//!   every segment ([`FetchReq`]) and calls [`PreadSource::fetch_into_batch`]. It serves catalog
//!   tests, initial materialization and an explicitly qualified demand mode.
//! * Not the async cache transport. The copy is a blocking `hipMemcpy` on the null stream, so
//!   it is no ordering or completion proof against work on other streams; fills that race live
//!   readers go through [`DeviceIo`](super::DeviceIo) and the slot bank's fences.
//! * Bounded. The staging buffer is sized once ([`PreadSource::open`] /
//!   [`PreadSource::reserve_staging`]); a segment longer than it is the named
//!   [`PreadError::StagingTooSmall`], never growth, so the steady state allocates nothing.
//! * Checked before any I/O. Every request's destination range is validated against its
//!   allocation before the batch reads a byte; a refused batch has copied nothing.
//! * Contextual errors. Every failure names the request's [`ExpertKey`], file offset and length.

use std::fs::File;
use std::path::Path;

use hip_bridge::{DeviceBuffer, HipError, HipRuntime};

use super::ExpertKey;

/// One segment to copy from the backing file into an existing device allocation.
/// `dst_offset` is relative to the beginning of `dst`.
#[derive(Clone, Copy)]
pub struct FetchReq<'a> {
    /// The bundle this segment belongs to (error context only).
    pub key: ExpertKey,
    pub file_offset: u64,
    pub len: usize,
    pub dst: &'a DeviceBuffer,
    pub dst_offset: usize,
}

/// A refused or failed segment fetch.
#[derive(Debug)]
pub enum PreadError {
    /// `dst_offset + len` overflows or exceeds the destination allocation.
    DestinationOutOfBounds {
        key: ExpertKey,
        dst_offset: usize,
        len: usize,
        dst_size: usize,
    },
    /// The segment is longer than the staging buffer reserved at load.
    StagingTooSmall {
        key: ExpertKey,
        len: usize,
        staging: usize,
    },
    /// The positional read failed (short file, I/O error).
    Read {
        key: ExpertKey,
        file_offset: u64,
        len: usize,
        source: std::io::Error,
    },
    /// The H2D copy failed.
    Copy {
        key: ExpertKey,
        file_offset: u64,
        len: usize,
        source: HipError,
    },
}

impl std::fmt::Display for PreadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let at = |f: &mut std::fmt::Formatter<'_>, key: &ExpertKey| {
            write!(f, "group {} expert {}", key.group_id, key.expert_id)
        };
        match self {
            Self::DestinationOutOfBounds {
                key,
                dst_offset,
                len,
                dst_size,
            } => {
                at(f, key)?;
                write!(
                    f,
                    ": destination range {dst_offset}+{len} exceeds allocation size {dst_size}"
                )
            }
            Self::StagingTooSmall { key, len, staging } => {
                at(f, key)?;
                write!(
                    f,
                    ": segment of {len} bytes exceeds the {staging}-byte staging buffer"
                )
            }
            Self::Read {
                key,
                file_offset,
                len,
                source,
            } => {
                at(f, key)?;
                write!(f, ": pread {len} bytes at offset {file_offset}: {source}")
            }
            Self::Copy {
                key,
                file_offset,
                len,
                source,
            } => {
                at(f, key)?;
                write!(
                    f,
                    ": H2D copy of {len} bytes read at offset {file_offset}: {source}"
                )
            }
        }
    }
}

impl std::error::Error for PreadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Copy { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Positional-read source over one model file with a fixed host staging buffer.
pub struct PreadSource {
    file: File,
    staging: Vec<u8>,
}

impl PreadSource {
    /// Open `path` with `staging_bytes` of staging (the largest segment the caller will fetch).
    pub fn open(path: &Path, staging_bytes: usize) -> std::io::Result<Self> {
        Ok(Self {
            file: File::open(path)?,
            staging: vec![0; staging_bytes],
        })
    }

    /// Grow the staging buffer to at least `bytes`. Call at load, never on the fetch path.
    pub fn reserve_staging(&mut self, bytes: usize) {
        if self.staging.len() < bytes {
            self.staging.resize(bytes, 0);
        }
    }

    pub fn staging_bytes(&self) -> usize {
        self.staging.len()
    }

    /// Validate every request, then read and copy them in order. The first failure stops the
    /// batch; segments copied before it stay copied, so the caller must treat the bundle as
    /// unfilled (roll the reservation back).
    pub fn fetch_into_batch(
        &mut self,
        reqs: &[FetchReq<'_>],
        hip: &HipRuntime,
    ) -> Result<(), PreadError> {
        for req in reqs {
            check_segment(
                req.key,
                req.dst_offset,
                req.len,
                req.dst.size(),
                self.staging.len(),
            )?;
        }
        for req in reqs {
            self.read_segment(req.key, req.file_offset, req.len)?;
            hip.memcpy_htod_offset(req.dst, req.dst_offset, &self.staging[..req.len])
                .map_err(|source| PreadError::Copy {
                    key: req.key,
                    file_offset: req.file_offset,
                    len: req.len,
                    source,
                })?;
        }
        Ok(())
    }

    /// Read `len` bytes at `file_offset` into `staging[..len]` (`len` already checked).
    fn read_segment(
        &mut self,
        key: ExpertKey,
        file_offset: u64,
        len: usize,
    ) -> Result<(), PreadError> {
        let buf = &mut self.staging[..len];
        #[cfg(unix)]
        let result = {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(buf, file_offset)
        };
        #[cfg(not(unix))]
        let result = {
            use std::io::{Read, Seek, SeekFrom};
            self.file
                .seek(SeekFrom::Start(file_offset))
                .and_then(|_| self.file.read_exact(buf))
        };
        result.map_err(|source| PreadError::Read {
            key,
            file_offset,
            len,
            source,
        })
    }
}

/// Refuse a segment whose destination range leaves its allocation or whose length exceeds the
/// staging buffer.
fn check_segment(
    key: ExpertKey,
    dst_offset: usize,
    len: usize,
    dst_size: usize,
    staging: usize,
) -> Result<(), PreadError> {
    if dst_offset.checked_add(len).is_none_or(|end| end > dst_size) {
        return Err(PreadError::DestinationOutOfBounds {
            key,
            dst_offset,
            len,
            dst_size,
        });
    }
    if len > staging {
        return Err(PreadError::StagingTooSmall { key, len, staging });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: ExpertKey = ExpertKey {
        owner_id: 1,
        owner_generation: 0,
        group_id: 3,
        expert_id: 7,
    };

    fn source_with(bytes: &[u8], staging: usize) -> (PreadSource, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-pread-{}-{}",
            std::process::id(),
            bytes.len() * 31 + staging
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.bin");
        std::fs::write(&path, bytes).unwrap();
        (PreadSource::open(&path, staging).unwrap(), dir)
    }

    #[test]
    fn read_segment_reads_at_offset_into_staging() {
        let bytes: Vec<u8> = (0..64u8).collect();
        let (mut src, dir) = source_with(&bytes, 16);
        src.read_segment(KEY, 10, 8).unwrap();
        assert_eq!(&src.staging[..8], &bytes[10..18]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn short_read_names_key_offset_and_length() {
        let (mut src, dir) = source_with(&[0u8; 32], 16);
        let err = src.read_segment(KEY, 24, 16).unwrap_err();
        assert!(matches!(
            err,
            PreadError::Read {
                file_offset: 24,
                len: 16,
                ..
            }
        ));
        let msg = err.to_string();
        assert!(msg.contains("group 3 expert 7"), "{msg}");
        assert!(msg.contains("offset 24"), "{msg}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn staging_is_reserved_not_grown_on_fetch() {
        let (mut src, dir) = source_with(&[0u8; 8], 4);
        assert_eq!(src.staging_bytes(), 4);
        src.reserve_staging(2);
        assert_eq!(src.staging_bytes(), 4);
        src.reserve_staging(12);
        assert_eq!(src.staging_bytes(), 12);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn segment_check_refuses_out_of_bounds_and_oversized() {
        assert!(check_segment(KEY, 0, 8, 8, 8).is_ok());
        assert!(matches!(
            check_segment(KEY, 4, 8, 8, 16),
            Err(PreadError::DestinationOutOfBounds {
                dst_offset: 4,
                len: 8,
                dst_size: 8,
                ..
            })
        ));
        assert!(matches!(
            check_segment(KEY, usize::MAX, 1, 8, 16),
            Err(PreadError::DestinationOutOfBounds { .. })
        ));
        assert!(matches!(
            check_segment(KEY, 0, 8, 16, 4),
            Err(PreadError::StagingTooSmall {
                len: 8,
                staging: 4,
                ..
            })
        ));
    }
}
