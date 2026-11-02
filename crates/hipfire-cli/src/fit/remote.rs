// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sparse admission inputs, never model weights. The original length matters to
//! admission, as do appended metadata and the PLE table headers beside a base.

use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

const MAX_HEADER: u64 = 512 * 1024 * 1024;
const MAX_PLE_JSON: u64 = 16 * 1024 * 1024;
const MAX_OVERLAY: u64 = 64 * 1024 * 1024;
const OVERLAY_FLAG: u32 = 1 << 31;

#[derive(Debug)]
pub(crate) struct Replica {
    pub path: PathBuf,
    /// Bytes transferred during this call, including discovery ranges.
    pub fetched_bytes: u64,
    pub cached: bool,
}

/// Use the same URL, authentication and PLE planning as pull/load. Cache keys
/// include the whole registry entry so a changed artifact cannot reuse old data.
pub(crate) fn header_replica(
    paths: &crate::Paths,
    tag: &str,
    entry: &hipfire_registry::ModelEntry,
) -> Result<Replica> {
    ensure!(
        Path::new(&entry.file).components().count() == 1
            && matches!(
                Path::new(&entry.file).components().next(),
                Some(Component::Normal(_))
            ),
        "registry artifact must have a plain file name"
    );
    let key = format!("{:x}", Sha256::digest(serde_json::to_vec(entry)?));
    let directory = paths.root.join("cache/fit-headers").join(key);
    let destination = directory.join(&entry.file);
    let mut source = HttpRangeSource::new(crate::artifact_url(entry, &entry.file))?;
    let mut replica = build_replica(&mut source, &destination, entry.size_bytes)?;
    if entry.has_ple_sidecars() {
        let resolved = crate::resolved_for_model(paths, &entry.file, Some(tag), Some(entry))?;
        let request = crate::ple::request_from_config(&resolved);
        let arch = crate::ple::load_target_arch(&resolved);
        let plan = crate::ple::plan_pull(entry, crate::ple::PleSelection::Auto, request, &[arch]);
        for (format, sidecar) in plan.sidecars {
            let path = crate::ple::ple_sidecar_path_for_key(&destination, format.name());
            let mut source = HttpRangeSource::new(crate::artifact_url(entry, &sidecar.file))?;
            let rows = build_replica(&mut source, &path, sidecar.size_bytes)
                .with_context(|| format!("PLE {} admission header", format.name()))?;
            replica.fetched_bytes += rows.fetched_bytes;
            replica.cached &= rows.cached;
        }
    }
    Ok(replica)
}

pub(crate) struct RangeBytes {
    pub bytes: Vec<u8>,
    pub total: u64,
}

/// Inclusive, bounded byte ranges. Both the production HTTP reader and the
/// local admission-equality experiment go through precisely the same builder.
pub(crate) trait RangeSource {
    fn range(&mut self, start: u64, end: u64) -> Result<RangeBytes>;
}

pub(crate) struct HttpRangeSource {
    url: String,
    agent: ureq::Agent,
    authorization: Option<String>,
}

impl HttpRangeSource {
    pub(crate) fn new(url: String) -> Result<Self> {
        Ok(Self {
            url,
            agent: ureq::Agent::config_builder()
                .timeout_global(Some(Duration::from_secs(120)))
                .http_status_as_error(false)
                .build()
                .into(),
            authorization: hipfire_config::http_auth::hf_authorization()?,
        })
    }
}

fn content_range(value: &str, start: u64, end: u64) -> Result<u64> {
    let (range, total) = value
        .strip_prefix("bytes ")
        .and_then(|s| s.split_once('/'))
        .ok_or_else(|| anyhow!("invalid Content-Range: {value}"))?;
    let (first, last) = range
        .split_once('-')
        .ok_or_else(|| anyhow!("invalid Content-Range: {value}"))?;
    let total: u64 = total
        .parse()
        .context("Content-Range has no concrete total length")?;
    ensure!(
        first.parse::<u64>()? == start && last.parse::<u64>()? == end && end < total,
        "unexpected Content-Range {value}; requested bytes {start}-{end}"
    );
    Ok(total)
}

impl RangeSource for HttpRangeSource {
    fn range(&mut self, start: u64, end: u64) -> Result<RangeBytes> {
        let length = checked_length(start, end)?;
        let mut request = self
            .agent
            .get(&self.url)
            .header("Range", format!("bytes={start}-{end}"))
            .header("Accept-Encoding", "identity");
        if let Some(auth) = &self.authorization {
            request = request.header("Authorization", auth);
        }
        let mut response = request
            .call()
            .with_context(|| format!("header range request failed for {}", self.url))?;
        ensure!(response.status().as_u16() == 206,
            "header range request returned HTTP {} for {}; require 206 Partial Content (will not download weights)",
            response.status(), self.url);
        let announced = response
            .headers()
            .get("content-range")
            .ok_or_else(|| anyhow!("range response missing Content-Range"))?
            .to_str()?;
        let total = content_range(announced, start, end)?;
        if let Some(encoding) = response.headers().get("content-encoding") {
            ensure!(
                encoding.to_str()? == "identity",
                "encoded range response is not byte-addressable"
            );
        }
        let mut bytes = Vec::with_capacity(length as usize);
        response
            .body_mut()
            .as_reader()
            .take(length + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 == length,
            "short/oversized range response: got {} bytes, expected {length}",
            bytes.len()
        );
        Ok(RangeBytes { bytes, total })
    }
}

/// Disk ranges are used only by the CPU probe and tests, not as an estimator.
#[allow(dead_code)]
pub(crate) struct FileRangeSource {
    file: File,
    total: u64,
}

impl FileRangeSource {
    #[allow(dead_code)]
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let total = file.metadata()?.len();
        Ok(Self { file, total })
    }
}

impl RangeSource for FileRangeSource {
    fn range(&mut self, start: u64, end: u64) -> Result<RangeBytes> {
        let length = checked_length(start, end)?;
        ensure!(
            end < self.total,
            "file range {start}-{end} exceeds {} bytes",
            self.total
        );
        self.file.seek(SeekFrom::Start(start))?;
        let mut bytes = vec![0; length as usize];
        self.file.read_exact(&mut bytes)?;
        Ok(RangeBytes {
            bytes,
            total: self.total,
        })
    }
}

fn checked_length(start: u64, end: u64) -> Result<u64> {
    let length = end
        .checked_sub(start)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| anyhow!("invalid byte range {start}-{end}"))?;
    ensure!(length <= MAX_HEADER,
        "admission header range is {length} bytes; capped at {MAX_HEADER} bytes to avoid fetching model payloads");
    Ok(length)
}

#[derive(Debug, PartialEq)]
struct HfqHeader {
    data_offset: u64,
    overlay: bool,
}

fn hfq_header(bytes: &[u8], total: u64) -> Result<HfqHeader> {
    ensure!(
        bytes.len() == 32 && &bytes[..4] == b"HFQM",
        "invalid 32-byte HFQM header"
    );
    let version = u32::from_le_bytes(bytes[4..8].try_into()?);
    ensure!(
        version & !OVERLAY_FLAG == 1,
        "unsupported HFQM version {}",
        version & !OVERLAY_FLAG
    );
    let metadata = u64::from_le_bytes(bytes[16..24].try_into()?);
    let data_offset = u64::from_le_bytes(bytes[24..32].try_into()?);
    ensure!(
        metadata == 32 && data_offset >= metadata && data_offset <= total,
        "invalid HFQM metadata/data offsets {metadata}/{data_offset} for {total}-byte artifact"
    );
    ensure!(data_offset <= MAX_HEADER,
        "HFQM header region is {data_offset} bytes; capped at {MAX_HEADER} bytes to avoid fetching model payloads");
    Ok(HfqHeader {
        data_offset,
        overlay: version & OVERLAY_FLAG != 0,
    })
}

/// Admission validates three tiny Qwen4 I64 records and, for symmetric MQ4V2,
/// samples group headers. These are metadata stored in the payload, not weights
/// to estimate. Mirror the bounded reads in artifact::validate_ple_records and
/// HfqFile::open; the probe pins their equivalence to the real reader.
fn admission_payload_ranges(prefix: &[u8], total: u64) -> Result<Vec<(u64, u64)>> {
    let mut metadata =
        serde_json::Deserializer::from_slice(&prefix[32..]).into_iter::<serde_json::Value>();
    let value = metadata
        .next()
        .transpose()?
        .ok_or_else(|| anyhow!("HFQM metadata is missing"))?;
    ensure!(value.is_object(), "HFQM metadata must be an object");
    let symmetric = value
        .get("mq4v2.symmetric")
        .is_some_and(|v| v == &serde_json::json!(1) || v == &serde_json::json!(true));
    let arch = u32::from_le_bytes(prefix[8..12].try_into()?);
    let count = u32::from_le_bytes(prefix[12..16].try_into()?);
    let mut index = &prefix[32 + metadata.byte_offset()..];
    fn take<'a>(index: &mut &'a [u8], length: usize) -> Result<&'a [u8]> {
        ensure!(index.len() >= length, "truncated HFQM tensor index");
        let (bytes, rest) = index.split_at(length);
        *index = rest;
        Ok(bytes)
    }
    let indexed_count = u32::from_le_bytes(take(&mut index, 4)?.try_into()?);
    ensure!(
        count == indexed_count,
        "HFQM tensor count does not match index"
    );
    let mut offset = prefix.len() as u64;
    let mut sampled = 0;
    let mut ranges = Vec::new();
    for _ in 0..count {
        let length = u16::from_le_bytes(take(&mut index, 2)?.try_into()?) as usize;
        let name = std::str::from_utf8(take(&mut index, length)?)?;
        let qt = take(&mut index, 1)?[0];
        let dimensions = take(&mut index, 1)?[0] as usize;
        take(&mut index, dimensions * 4)?;
        take(&mut index, 4)?; // group size
        let size = u64::from_le_bytes(take(&mut index, 8)?.try_into()?);
        let end = offset
            .checked_add(size)
            .ok_or_else(|| anyhow!("HFQM tensor extent overflows"))?;
        ensure!(end <= total, "HFQM tensor extent exceeds artifact length");
        if arch == 16
            && qt == 54
            && size > 0
            && size <= 2048
            && [
                ".ple_embedding.layer_multipliers",
                ".ple_embedding.ngram_heads_offsets",
                ".ple_embedding.ngram_heads_vocab_sizes",
            ]
            .iter()
            .any(|suffix| name.ends_with(suffix))
        {
            ranges.push((offset, end - 1));
        }
        if symmetric && qt == 44 && sampled < 4 {
            sampled += 1;
            let groups = size / 136;
            if groups > 0 {
                for group in [0, groups / 2, groups - 1] {
                    let start = offset + group * 136;
                    ranges.push((start, start + 7));
                }
            }
        }
        offset = end;
    }
    ranges.sort_unstable();
    ranges.dedup();
    Ok(ranges)
}

#[derive(Serialize, Deserialize)]
struct CacheRecord {
    version: u32,
    total: u64,
    /// Length and digest for each region actually materialized in the sparse file.
    regions: Vec<(u64, u64, String)>,
}

fn record_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".fit-header.json");
    PathBuf::from(name)
}

fn cached(path: &Path, expected_size: Option<u64>) -> Option<Replica> {
    let record: CacheRecord = serde_json::from_slice(&fs::read(record_path(path)).ok()?).ok()?;
    let mut file = File::open(path).ok()?;
    if file.metadata().ok()?.len() != record.total
        || record.version != 3
        || expected_size.is_some_and(|size| size != record.total)
        || record
            .regions
            .first()
            .is_none_or(|(offset, length, _)| *offset != 0 || *length < 32)
    {
        return None;
    }
    for (offset, length, digest) in record.regions {
        if length > MAX_HEADER || offset.checked_add(length)? > record.total {
            return None;
        }
        file.seek(SeekFrom::Start(offset)).ok()?;
        let mut reader = (&mut file).take(length);
        let mut hash = Sha256::new();
        let count = std::io::copy(&mut reader, &mut hash).ok()?;
        if count != length || format!("{:x}", hash.finalize()) != digest {
            return None;
        }
    }
    Some(Replica {
        path: path.into(),
        fetched_bytes: 0,
        cached: true,
    })
}

fn fetch(
    source: &mut impl RangeSource,
    start: u64,
    end: u64,
    total: Option<u64>,
    fetched: &mut u64,
) -> Result<RangeBytes> {
    let length = checked_length(start, end)?;
    let range = source.range(start, end)?;
    ensure!(
        range.bytes.len() as u64 == length,
        "range source returned wrong byte count"
    );
    ensure!(
        end < range.total && total.is_none_or(|n| n == range.total),
        "artifact length changed between range requests"
    );
    *fetched += length;
    Ok(range)
}

/// Materialize the container header, optional HFQM metadata tail and bounded
/// metadata payload records/samples validated by admission. A rank-ordered PLE
/// table additionally needs its embedded rank (not its rows): real admission
/// checks its digest and constructs a tier index. Sparse holes retain the
/// original logical length without downloading model weights.
pub(crate) fn build_replica(
    source: &mut impl RangeSource,
    destination: &Path,
    expected_size: Option<u64>,
) -> Result<Replica> {
    if let Some(replica) = cached(destination, expected_size) {
        return Ok(replica);
    }
    let mut fetched = 0;
    let first = fetch(source, 0, 31, None, &mut fetched)?;
    let total = first.total;
    ensure!(
        expected_size.is_none_or(|n| n == total),
        "artifact length {total} does not match registry size {expected_size:?}"
    );
    let mut regions = Vec::new();
    if &first.bytes[..4] == b"HFQM" {
        let header = hfq_header(&first.bytes, total)?;
        let prefix = fetch(source, 0, header.data_offset - 1, Some(total), &mut fetched)?;
        ensure!(
            prefix.bytes[..32] == first.bytes,
            "artifact header changed between requests"
        );
        let payload_ranges = admission_payload_ranges(&prefix.bytes, total)?;
        regions.push((0, prefix.bytes));
        for (start, end) in payload_ranges {
            let range = fetch(source, start, end, Some(total), &mut fetched)?;
            regions.push((start, range.bytes));
        }
        if header.overlay {
            ensure!(total >= 52, "HFQM overlay footer is truncated");
            let footer_at = total - 52;
            let footer = fetch(source, footer_at, total - 1, Some(total), &mut fetched)?;
            ensure!(
                &footer.bytes[44..] == b"HFQMDOV1",
                "HFQM overlay footer magic is corrupt"
            );
            let length = u64::from_le_bytes(footer.bytes[..8].try_into()?);
            ensure!(
                (1..=MAX_OVERLAY).contains(&length),
                "HFQM overlay length {length} exceeds allowed 1..={MAX_OVERLAY}"
            );
            ensure!(
                u32::from_le_bytes(footer.bytes[8..12].try_into()?) == 1,
                "unsupported HFQM overlay schema"
            );
            let start = footer_at
                .checked_sub(length)
                .ok_or_else(|| anyhow!("HFQM overlay length underflows file"))?;
            ensure!(
                start >= header.data_offset,
                "HFQM overlay overlaps the header"
            );
            let overlay = fetch(source, start, footer_at - 1, Some(total), &mut fetched)?;
            ensure!(
                Sha256::digest(&overlay.bytes).as_slice() == &footer.bytes[12..44],
                "HFQM overlay checksum mismatch"
            );
            regions.push((start, overlay.bytes));
            regions.push((footer_at, footer.bytes));
        }
    } else if &first.bytes[..8] == b"HFPLEHP1" {
        let length = u64::from_le_bytes(first.bytes[8..16].try_into()?);
        ensure!(
            (1..=MAX_PLE_JSON).contains(&length),
            "PLE JSON header length {length} exceeds allowed 1..={MAX_PLE_JSON}"
        );
        let end = 16 + length;
        ensure!(end <= total, "PLE JSON header exceeds artifact length");
        let prefix = fetch(source, 0, end - 1, Some(total), &mut fetched)?;
        ensure!(
            prefix.bytes[..32.min(end as usize)] == first.bytes[..32.min(end as usize)],
            "PLE header changed between requests"
        );
        let header: serde_json::Value = serde_json::from_slice(&prefix.bytes[16..])?;
        if header.get("order").and_then(|v| v.as_str()) == Some("rank") {
            let integer = |key: &str| -> Result<u64> {
                header
                    .get(key)
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| anyhow!("PLE rank header lacks integer {key}"))
            };
            let offset = integer("rank_offset")?;
            let length = integer("rank_bytes")?;
            let rank_end = offset
                .checked_add(length)
                .ok_or_else(|| anyhow!("PLE rank extent overflows"))?;
            ensure!(
                length > 0 && offset >= end && rank_end <= total,
                "PLE embedded rank extent is outside artifact bounds"
            );
            let rank = fetch(source, offset, rank_end - 1, Some(total), &mut fetched)?;
            let digest = header
                .get("rank_payload_sha256")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("PLE rank header lacks rank_payload_sha256"))?;
            ensure!(
                format!("{:x}", Sha256::digest(&rank.bytes)) == digest,
                "PLE embedded rank checksum mismatch"
            );
            regions.push((offset, rank.bytes));
        }
        regions.insert(0, (0, prefix.bytes));
    } else {
        bail!(
            "registry artifact is neither HFQM nor a PLE table; header-only admission unsupported"
        );
    }
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow!("replica destination has no parent"))?;
    fs::create_dir_all(parent)?;
    // create_new plus a per-process counter also makes simultaneous fits safe.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let temporary = parent.join(format!(
        ".fit-header-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let temporary_record = record_path(&temporary);
    let result = (|| -> Result<()> {
        let mut output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        output.set_len(total)?;
        let mut record = CacheRecord {
            version: 3,
            total,
            regions: Vec::new(),
        };
        for (offset, bytes) in regions {
            output.seek(SeekFrom::Start(offset))?;
            output.write_all(&bytes)?;
            record.regions.push((
                offset,
                bytes.len() as u64,
                format!("{:x}", Sha256::digest(&bytes)),
            ));
        }
        output.sync_all()?;
        fs::write(&temporary_record, serde_json::to_vec(&record)?)?;
        fs::rename(&temporary, destination)?;
        fs::rename(&temporary_record, record_path(destination))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
        let _ = fs::remove_file(&temporary_record);
    }
    result
        .with_context(|| format!("writing sparse admission replica {}", destination.display()))?;
    Ok(Replica {
        path: destination.into(),
        fetched_bytes: fetched,
        cached: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn synthetic() -> Vec<u8> {
        let mut bytes = vec![0; 1024 * 1024];
        bytes[..4].copy_from_slice(b"HFQM");
        bytes[4..8].copy_from_slice(&1_u32.to_le_bytes());
        bytes[16..24].copy_from_slice(&32_u64.to_le_bytes());
        bytes[24..32].copy_from_slice(&4096_u64.to_le_bytes());
        bytes[32..4096].fill(17);
        bytes[32..34].copy_from_slice(b"{}");
        bytes[34..38].copy_from_slice(&0_u32.to_le_bytes());
        bytes
    }

    fn directory(label: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "hipfire-fit-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    struct Memory(Vec<u8>);
    impl RangeSource for Memory {
        fn range(&mut self, start: u64, end: u64) -> Result<RangeBytes> {
            Ok(RangeBytes {
                bytes: self.0[start as usize..=end as usize].to_vec(),
                total: self.0.len() as u64,
            })
        }
    }

    #[test]
    fn parses_headers_and_content_ranges() {
        let mut bytes = synthetic();
        assert_eq!(
            hfq_header(&bytes[..32], bytes.len() as u64).unwrap(),
            HfqHeader {
                data_offset: 4096,
                overlay: false
            }
        );
        bytes[4..8].copy_from_slice(&(1 | OVERLAY_FLAG).to_le_bytes());
        assert!(
            hfq_header(&bytes[..32], bytes.len() as u64)
                .unwrap()
                .overlay
        );
        bytes[24..32].copy_from_slice(&(MAX_HEADER + 1).to_le_bytes());
        assert!(hfq_header(&bytes[..32], MAX_HEADER + 2).is_err());
        assert_eq!(content_range("bytes 0-31/1000", 0, 31).unwrap(), 1000);
        assert!(content_range("bytes 0-30/1000", 0, 31).is_err());
        assert!(content_range("bytes 0-31/*", 0, 31).is_err());
    }

    #[test]
    fn bounded_admission_payload_is_preserved() {
        let mut bytes = synthetic();
        bytes[8..12].copy_from_slice(&16_u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&4_u32.to_le_bytes());
        let mut index = b"{\"mq4v2.symmetric\":1}".to_vec();
        index.extend_from_slice(&4_u32.to_le_bytes());
        for (name, qt, size) in [
            (
                "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
                54_u8,
                24_u64,
            ),
            (
                "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
                54,
                128,
            ),
            (
                "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
                54,
                128,
            ),
            ("layer.weight", 44, 408),
        ] {
            index.extend_from_slice(&(name.len() as u16).to_le_bytes());
            index.extend_from_slice(name.as_bytes());
            index.extend_from_slice(&[qt, 1]);
            index.extend_from_slice(&1_u32.to_le_bytes());
            index.extend_from_slice(&0_u32.to_le_bytes());
            index.extend_from_slice(&size.to_le_bytes());
        }
        bytes[32..32 + index.len()].copy_from_slice(&index);
        bytes[4096..4096 + 688].fill(73);
        let dir = directory("payload");
        let destination = dir.join("replica");
        let replica = build_replica(&mut Memory(bytes.clone()), &destination, None).unwrap();
        assert_eq!(replica.fetched_bytes, 32 + 4096 + 280 + 24);
        let copied = fs::read(destination).unwrap();
        assert_eq!(&copied[4096..4376], &bytes[4096..4376]);
        for offset in [4376, 4512, 4648] {
            assert_eq!(&copied[offset..offset + 8], &bytes[offset..offset + 8]);
            assert_eq!(copied[offset + 8], 0);
        }
        assert!(admission_payload_ranges(&bytes[..64], bytes.len() as u64).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sparse_overlay_and_cache() {
        let dir = directory("overlay");
        let source_path = dir.join("real");
        let destination = dir.join("replica");
        let mut bytes = synthetic();
        bytes[4..8].copy_from_slice(&(1 | OVERLAY_FLAG).to_le_bytes());
        let overlay = b"{\"test\":true}";
        bytes.extend_from_slice(overlay);
        bytes.extend_from_slice(&(overlay.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&Sha256::digest(overlay));
        bytes.extend_from_slice(b"HFQMDOV1");
        fs::write(&source_path, &bytes).unwrap();
        let mut source = FileRangeSource::open(&source_path).unwrap();
        let replica = build_replica(&mut source, &destination, Some(bytes.len() as u64)).unwrap();
        assert!(!replica.cached);
        let copied = fs::read(&destination).unwrap();
        assert_eq!(copied.len(), bytes.len());
        assert_eq!(&copied[..4096], &bytes[..4096]);
        assert_eq!(&copied[1024 * 1024..], &bytes[1024 * 1024..]);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert!(fs::metadata(&destination).unwrap().blocks() * 512 < bytes.len() as u64);
        }
        let mut empty = Memory(Vec::new());
        assert!(
            build_replica(&mut empty, &destination, Some(bytes.len() as u64))
                .unwrap()
                .cached
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ple_header_only_and_size_refusal() {
        let dir = directory("ple");
        let destination = dir.join("base.ple-q4");
        let json = b"{\"kind\":\"ple-table\"}";
        let mut bytes = vec![0; 1024 * 1024];
        bytes[..8].copy_from_slice(b"HFPLEHP1");
        bytes[8..16].copy_from_slice(&(json.len() as u64).to_le_bytes());
        bytes[16..16 + json.len()].copy_from_slice(json);
        let mut source = Memory(bytes.clone());
        assert!(build_replica(&mut source, &destination, Some(1)).is_err());
        let replica = build_replica(&mut source, &destination, Some(bytes.len() as u64)).unwrap();
        assert_eq!(replica.fetched_bytes, 32 + 16 + json.len() as u64);
        assert_eq!(
            &fs::read(&destination).unwrap()[..16 + json.len()],
            &bytes[..16 + json.len()]
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rank_ordered_ple_requires_embedded_rank_not_rows() {
        let dir = directory("rank");
        let destination = dir.join("base.ple-q4");
        let rank = vec![23_u8; 256];
        let json = serde_json::to_vec(&serde_json::json!({
            "kind": "ple-table", "order": "rank", "rank_offset": 4096,
            "rank_bytes": rank.len(), "rank_payload_sha256": format!("{:x}", Sha256::digest(&rank))
        }))
        .unwrap();
        let mut bytes = vec![0; 1024 * 1024];
        bytes[..8].copy_from_slice(b"HFPLEHP1");
        bytes[8..16].copy_from_slice(&(json.len() as u64).to_le_bytes());
        bytes[16..16 + json.len()].copy_from_slice(&json);
        bytes[4096..4352].copy_from_slice(&rank);
        bytes[8192..].fill(19);
        let replica = build_replica(&mut Memory(bytes.clone()), &destination, None).unwrap();
        assert_eq!(replica.fetched_bytes, 32 + 16 + json.len() as u64 + 256);
        let copied = fs::read(&destination).unwrap();
        assert_eq!(&copied[4096..4352], &rank);
        assert_eq!(copied[8192], 0);
        bytes[4096] ^= 1;
        assert!(build_replica(&mut Memory(bytes), &dir.join("corrupt"), None).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn http_ranges_survive_redirects() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let bytes = synthetic();
        let server = std::thread::spawn(move || {
            // Discovery and header fetch each follow a 302 to the CDN path.
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = Vec::new();
                let mut byte = [0];
                while !request.ends_with(b"\r\n\r\n") {
                    stream.read_exact(&mut byte).unwrap();
                    request.push(byte[0]);
                }
                let request = String::from_utf8(request).unwrap();
                if request.starts_with("GET /artifact ") {
                    write!(stream, "HTTP/1.1 302 Found\r\nLocation: http://{address}/cdn\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
                } else {
                    let range = request
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("range: bytes=")
                                .map(str::to_owned)
                        })
                        .expect("Range lost on redirect");
                    let (start, end) = range.split_once('-').unwrap();
                    let start: usize = start.parse().unwrap();
                    let end: usize = end.trim().parse().unwrap();
                    write!(stream, "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{end}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", bytes.len(), end - start + 1).unwrap();
                    stream.write_all(&bytes[start..=end]).unwrap();
                }
            }
        });
        let dir = directory("http");
        let mut source = HttpRangeSource::new(format!("http://{address}/artifact")).unwrap();
        source.authorization = None;
        let replica = build_replica(&mut source, &dir.join("replica"), None).unwrap();
        assert_eq!(replica.fetched_bytes, 32 + 4096);
        server.join().unwrap();
        fs::remove_dir_all(dir).unwrap();
    }
}
