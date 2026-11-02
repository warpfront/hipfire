// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host memory snapshots and per-adapter DXGI video-memory budgets.
//!
//! Everything here is a **report of what the calling process observes**. The
//! values are never an admission decision on their own: engine memory
//! admission MUST query the engine process itself (the process that owns the
//! allocations), and CLI reports built from this module describe the CLI
//! process only. On Windows both the commit numbers
//! (`GlobalMemoryStatusEx`) and the DXGI budgets (`QueryVideoMemoryInfo`) are
//! scoped to the calling process, so a value read from another process is not
//! a bound on this one.
//!
//! The parsers and mappings are pure functions so they are tested on every
//! platform; only the thin OS calls are `cfg`-gated.

// The /proc/meminfo parser is only called on Linux; it stays compiled (and
// tested) everywhere.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use serde::Serialize;

/// Provenance of a [`HostMemory`] snapshot read from `/proc/meminfo`.
pub const SOURCE_PROC_MEMINFO: &str = "/proc/meminfo";

/// Provenance of a [`HostMemory`] snapshot read from `GlobalMemoryStatusEx`.
pub const SOURCE_GLOBAL_MEMORY_STATUS_EX: &str =
    "GlobalMemoryStatusEx (calling process commit limit)";

/// Provenance of every [`DxgiAdapterBudget`] (and its [`SegmentBudget`]s).
pub const DXGI_BUDGET_SOURCE: &str = "DXGI QueryVideoMemoryInfo (calling process, node 0)";

/// PCI vendor id of AMD adapters as reported by DXGI.
pub const AMD_VENDOR_ID: u32 = 0x1002;

/// `DXGI_ADAPTER_FLAG_SOFTWARE` (`DXGI_ADAPTER_DESC1::Flags` bit).
#[cfg_attr(not(windows), allow(dead_code))]
const DXGI_ADAPTER_FLAG_SOFTWARE_BIT: u32 = 2;

/// The only video-memory node queried: node 0 of each adapter. Multi-node
/// (linked) adapters are reported through their first node.
#[cfg_attr(not(windows), allow(dead_code))]
const DXGI_QUERIED_NODE_INDEX: u32 = 0;

/// Point-in-time host memory snapshot, all sizes in bytes.
///
/// `total_physical_bytes` and `available_physical_bytes` are required: a
/// snapshot without them is not produced at all. The swap and commit fields
/// are optional and are `None` when the platform does not report them (or
/// reports them unparseably); a missing optional field never discards the
/// snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct HostMemory {
    /// Installed physical memory visible to the OS (`MemTotal` /
    /// `ullTotalPhys`).
    pub total_physical_bytes: u64,
    /// Physical memory immediately available without paging out
    /// (`MemAvailable` / `ullAvailPhys`).
    pub available_physical_bytes: u64,
    /// Swap / pagefile capacity. Linux: `SwapTotal`. Windows: an
    /// approximation, `total commit - physical` saturating at zero.
    pub swap_total_bytes: Option<u64>,
    /// Commit limit. Linux: `CommitLimit`. Windows: `ullTotalPageFile`, the
    /// commit limit of the calling process.
    pub commit_limit_bytes: Option<u64>,
    /// Commit still available. Linux: `CommitLimit - Committed_AS`,
    /// saturating at zero (the kernel may commit beyond the limit when
    /// overcommit is enabled), `None` unless both are present. Windows:
    /// `ullAvailPageFile` as reported, for the calling process.
    pub commit_available_bytes: Option<u64>,
    /// Where the snapshot (and so every field in it) came from.
    pub source: &'static str,
}

/// Snapshot of host memory as seen by the calling process; `None` when the
/// platform has no supported source or the required physical fields cannot be
/// read.
pub fn host_memory() -> Option<HostMemory> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        parse_meminfo(&meminfo)
    }
    #[cfg(windows)]
    {
        windows_host_memory()
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

/// Outcome of looking up one `/proc/meminfo` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MeminfoField {
    Missing,
    Malformed,
    Bytes(u64),
}

/// First `key:` line of a `/proc/meminfo` body, converted kB → bytes. The
/// value is a base-10 integer, optionally followed by exactly `kB`; any other
/// shape or a byte count that overflows `u64` is [`MeminfoField::Malformed`].
fn meminfo_field(meminfo: &str, key: &str) -> MeminfoField {
    for line in meminfo.lines() {
        let Some((name, rest)) = line.split_once(':') else {
            continue;
        };
        if name != key {
            continue;
        }
        let mut tokens = rest.split_whitespace();
        let Some(value) = tokens.next().and_then(|t| t.parse::<u64>().ok()) else {
            return MeminfoField::Malformed;
        };
        match tokens.next() {
            None | Some("kB") => {}
            Some(_) => return MeminfoField::Malformed,
        }
        if tokens.next().is_some() {
            return MeminfoField::Malformed;
        }
        return match value.checked_mul(1024) {
            Some(bytes) => MeminfoField::Bytes(bytes),
            None => MeminfoField::Malformed,
        };
    }
    MeminfoField::Missing
}

fn optional_bytes(field: MeminfoField) -> Option<u64> {
    match field {
        MeminfoField::Bytes(bytes) => Some(bytes),
        MeminfoField::Missing | MeminfoField::Malformed => None,
    }
}

/// `MemTotal` of a `/proc/meminfo` body in bytes. Unlike [`parse_meminfo`] it
/// does not require `MemAvailable`: callers that only fingerprint the
/// installed memory (see [`crate::calibration`]) need no more than the total.
pub(crate) fn meminfo_total_bytes(meminfo: &str) -> Option<u64> {
    optional_bytes(meminfo_field(meminfo, "MemTotal"))
}

/// Build a [`HostMemory`] from a `/proc/meminfo` body. `None` when `MemTotal`
/// or `MemAvailable` is missing or malformed; the optional fields degrade to
/// `None` individually.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_meminfo(meminfo: &str) -> Option<HostMemory> {
    let MeminfoField::Bytes(total_physical_bytes) = meminfo_field(meminfo, "MemTotal") else {
        return None;
    };
    let MeminfoField::Bytes(available_physical_bytes) = meminfo_field(meminfo, "MemAvailable")
    else {
        return None;
    };
    let swap_total_bytes = optional_bytes(meminfo_field(meminfo, "SwapTotal"));
    let commit_limit_bytes = optional_bytes(meminfo_field(meminfo, "CommitLimit"));
    let committed_bytes = optional_bytes(meminfo_field(meminfo, "Committed_AS"));
    let commit_available_bytes = match (commit_limit_bytes, committed_bytes) {
        (Some(limit), Some(committed)) => Some(limit.saturating_sub(committed)),
        _ => None,
    };
    Some(HostMemory {
        total_physical_bytes,
        available_physical_bytes,
        swap_total_bytes,
        commit_limit_bytes,
        commit_available_bytes,
        source: SOURCE_PROC_MEMINFO,
    })
}

/// Map `MEMORYSTATUSEX` fields to a [`HostMemory`].
///
/// `total_page_file` is the commit limit of the calling process (physical
/// plus pagefile, as the OS currently allows this process to commit) and
/// `avail_page_file` the commit it can still take. The swap figure is an
/// approximation (`total commit - physical`, saturating) because Windows
/// reports no direct pagefile size here.
#[cfg_attr(not(windows), allow(dead_code))]
fn host_memory_from_windows_status(
    total_phys: u64,
    avail_phys: u64,
    total_page_file: u64,
    avail_page_file: u64,
) -> HostMemory {
    HostMemory {
        total_physical_bytes: total_phys,
        available_physical_bytes: avail_phys,
        swap_total_bytes: Some(total_page_file.saturating_sub(total_phys)),
        commit_limit_bytes: Some(total_page_file),
        commit_available_bytes: Some(avail_page_file),
        source: SOURCE_GLOBAL_MEMORY_STATUS_EX,
    }
}

/// `MEMORYSTATUSEX` (sysinfoapi.h).
#[cfg(windows)]
#[repr(C)]
struct MemoryStatusEx {
    dw_length: u32,
    dw_memory_load: u32,
    ull_total_phys: u64,
    ull_avail_phys: u64,
    ull_total_page_file: u64,
    ull_avail_page_file: u64,
    ull_total_virtual: u64,
    ull_avail_virtual: u64,
    ull_avail_extended_virtual: u64,
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    /// `BOOL GlobalMemoryStatusEx(LPMEMORYSTATUSEX)`; nonzero on success.
    fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
}

#[cfg(windows)]
fn windows_host_memory() -> Option<HostMemory> {
    let mut status = MemoryStatusEx {
        // The struct is 64 bytes; the cast cannot truncate.
        dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };
    // SAFETY: `status` is a live, correctly laid out MEMORYSTATUSEX with
    // `dwLength` set as the API requires.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 {
        return None;
    }
    Some(host_memory_from_windows_status(
        status.ull_total_phys,
        status.ull_avail_phys,
        status.ull_total_page_file,
        status.ull_avail_page_file,
    ))
}

/// One DXGI video-memory segment group as the calling process sees it
/// (`DXGI_QUERY_VIDEO_MEMORY_INFO`), all sizes in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SegmentBudget {
    /// OS-provided budget for this process in the segment group.
    pub budget_bytes: u64,
    /// Bytes this process currently uses in the segment group.
    pub current_usage_bytes: u64,
    /// Bytes this process may still reserve.
    pub available_for_reservation_bytes: u64,
    /// Bytes this process currently has reserved.
    pub current_reservation_bytes: u64,
}

/// A hardware DXGI adapter with its calling-process video-memory budgets.
///
/// `source` describes the budgets (`local`, `non_local`) and applies to the
/// whole snapshot; the descriptor fields come from `DXGI_ADAPTER_DESC1`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DxgiAdapterBudget {
    pub description: String,
    pub vendor_id: u32,
    pub device_id: u32,
    /// Adapter LUID as bytes: `LowPart` little endian, then `HighPart` little
    /// endian — the byte layout HIP reports for the same device.
    pub luid: [u8; 8],
    pub dedicated_video_memory_bytes: u64,
    pub shared_system_memory_bytes: u64,
    /// Raw `DXGI_ADAPTER_DESC1::Flags`.
    pub flags: u32,
    pub source: &'static str,
    /// Video-memory node the budgets were read from.
    pub node_index: u32,
    /// `DXGI_MEMORY_SEGMENT_GROUP_LOCAL` (device-local memory).
    pub local: SegmentBudget,
    /// `DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL` (system memory visible to the
    /// adapter).
    pub non_local: SegmentBudget,
}

impl DxgiAdapterBudget {
    /// Whether this adapter's LUID equals `luid` (a HIP device LUID).
    pub fn matches_luid(&self, luid: &[u8; 8]) -> bool {
        &self.luid == luid
    }
}

/// The AMD adapters (`vendor_id == 0x1002`) of `adapters`, in input order.
pub fn amd_adapters(adapters: &[DxgiAdapterBudget]) -> impl Iterator<Item = &DxgiAdapterBudget> {
    adapters
        .iter()
        .filter(|adapter| adapter.vendor_id == AMD_VENDOR_ID)
}

/// LUID bytes as HIP exposes them: `LowPart` then `HighPart`, each little
/// endian.
#[cfg_attr(not(windows), allow(dead_code))]
fn luid_bytes(low_part: u32, high_part: i32) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    bytes[..4].copy_from_slice(&low_part.to_le_bytes());
    bytes[4..].copy_from_slice(&high_part.to_le_bytes());
    bytes
}

/// UTF-16 adapter description up to the first NUL (or the whole slice when
/// unterminated); invalid UTF-16 is replaced with U+FFFD.
#[cfg_attr(not(windows), allow(dead_code))]
fn description_from_utf16(wide: &[u16]) -> String {
    let end = wide
        .iter()
        .position(|&unit| unit == 0)
        .unwrap_or(wide.len());
    String::from_utf16_lossy(&wide[..end])
}

/// Whether `DXGI_ADAPTER_DESC1::Flags` marks a software adapter.
#[cfg_attr(not(windows), allow(dead_code))]
fn is_software_adapter(flags: u32) -> bool {
    flags & DXGI_ADAPTER_FLAG_SOFTWARE_BIT != 0
}

/// Failure while enumerating DXGI adapters; every variant names the step (and
/// the adapter, once one is known) that failed. No partial result is returned
/// alongside an error.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DxgiError {
    #[error("DXGI: CreateDXGIFactory1(IDXGIFactory1) failed: {detail}")]
    CreateFactory { detail: String },
    #[error("DXGI: IDXGIFactory1::EnumAdapters1({index}) failed: {detail}")]
    EnumAdapters { index: u32, detail: String },
    #[error("DXGI: IDXGIAdapter1::GetDesc1 failed for adapter {index}: {detail}")]
    GetDesc { index: u32, detail: String },
    #[error(
        "DXGI: adapter {index} ({description}) does not expose IDXGIAdapter3 \
         (QueryVideoMemoryInfo needs Windows 10 or newer): {detail}"
    )]
    CastAdapter3 {
        index: u32,
        description: String,
        detail: String,
    },
    #[error(
        "DXGI: IDXGIAdapter3::QueryVideoMemoryInfo(node {node_index}, {segment}) failed for \
         adapter {index} ({description}): {detail}"
    )]
    QueryVideoMemoryInfo {
        index: u32,
        description: String,
        node_index: u32,
        segment: &'static str,
        detail: String,
    },
}

/// Enumerate hardware DXGI adapters and read their node-0 video-memory
/// budgets for the calling process. Software adapters (for example the
/// Microsoft Basic Render Driver) are skipped. Any DXGI failure other than
/// `DXGI_ERROR_NOT_FOUND` (end of enumeration) is returned as an error.
///
/// No explicit `CoInitializeEx` is needed: DXGI uses its direct factory entry
/// point, not `CoCreateInstance`. See Microsoft's "Creating a COM object":
/// <https://learn.microsoft.com/en-us/windows/win32/prog-dx-with-com#creating-a-com-object>.
#[cfg(windows)]
pub fn dxgi_adapter_budgets() -> Result<Vec<DxgiAdapterBudget>, DxgiError> {
    use windows::core::Interface;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory1, DXGI_ERROR_NOT_FOUND,
        DXGI_MEMORY_SEGMENT_GROUP, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
    };

    fn query_segment(
        adapter: &IDXGIAdapter3,
        group: DXGI_MEMORY_SEGMENT_GROUP,
    ) -> windows::core::Result<SegmentBudget> {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: `info` is a live, writable DXGI_QUERY_VIDEO_MEMORY_INFO.
        unsafe { adapter.QueryVideoMemoryInfo(DXGI_QUERIED_NODE_INDEX, group, &mut info)? };
        Ok(SegmentBudget {
            budget_bytes: info.Budget,
            current_usage_bytes: info.CurrentUsage,
            available_for_reservation_bytes: info.AvailableForReservation,
            current_reservation_bytes: info.CurrentReservation,
        })
    }

    // SAFETY: plain COM factory creation; the returned interface is owned.
    let factory: IDXGIFactory1 =
        unsafe { CreateDXGIFactory1() }.map_err(|e| DxgiError::CreateFactory {
            detail: e.to_string(),
        })?;

    let mut budgets = Vec::new();
    let mut index = 0u32;
    loop {
        // SAFETY: `factory` is a live IDXGIFactory1.
        let adapter = match unsafe { factory.EnumAdapters1(index) } {
            Ok(adapter) => adapter,
            Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(e) => {
                return Err(DxgiError::EnumAdapters {
                    index,
                    detail: e.to_string(),
                })
            }
        };
        // SAFETY: `adapter` is a live IDXGIAdapter1.
        let desc = unsafe { adapter.GetDesc1() }.map_err(|e| DxgiError::GetDesc {
            index,
            detail: e.to_string(),
        })?;
        if is_software_adapter(desc.Flags) {
            index += 1;
            continue;
        }
        let description = description_from_utf16(&desc.Description);
        let adapter3: IDXGIAdapter3 = adapter.cast().map_err(|e| DxgiError::CastAdapter3 {
            index,
            description: description.clone(),
            detail: e.to_string(),
        })?;
        let local = query_segment(&adapter3, DXGI_MEMORY_SEGMENT_GROUP_LOCAL).map_err(|e| {
            DxgiError::QueryVideoMemoryInfo {
                index,
                description: description.clone(),
                node_index: DXGI_QUERIED_NODE_INDEX,
                segment: "LOCAL",
                detail: e.to_string(),
            }
        })?;
        let non_local =
            query_segment(&adapter3, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL).map_err(|e| {
                DxgiError::QueryVideoMemoryInfo {
                    index,
                    description: description.clone(),
                    node_index: DXGI_QUERIED_NODE_INDEX,
                    segment: "NON_LOCAL",
                    detail: e.to_string(),
                }
            })?;
        budgets.push(DxgiAdapterBudget {
            description,
            vendor_id: desc.VendorId,
            device_id: desc.DeviceId,
            luid: luid_bytes(desc.AdapterLuid.LowPart, desc.AdapterLuid.HighPart),
            // usize -> u64 is lossless on every supported target.
            dedicated_video_memory_bytes: desc.DedicatedVideoMemory as u64,
            shared_system_memory_bytes: desc.SharedSystemMemory as u64,
            flags: desc.Flags,
            source: DXGI_BUDGET_SOURCE,
            node_index: DXGI_QUERIED_NODE_INDEX,
            local,
            non_local,
        });
        index += 1;
    }
    Ok(budgets)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &str = "MemTotal:       131072000 kB\n\
                        MemFree:         1000 kB\n\
                        MemAvailable:   100000000 kB\n\
                        SwapTotal:       2000000 kB\n\
                        SwapFree:        2000000 kB\n\
                        CommitLimit:    90000000 kB\n\
                        Committed_AS:   30000000 kB\n";

    #[test]
    fn parses_full_meminfo_to_bytes_with_source() {
        let m = parse_meminfo(FULL).expect("snapshot");
        assert_eq!(m.total_physical_bytes, 131_072_000 * 1024);
        assert_eq!(m.available_physical_bytes, 100_000_000 * 1024);
        assert_eq!(m.swap_total_bytes, Some(2_000_000 * 1024));
        assert_eq!(m.commit_limit_bytes, Some(90_000_000 * 1024));
        assert_eq!(m.commit_available_bytes, Some(60_000_000 * 1024));
        assert_eq!(m.source, SOURCE_PROC_MEMINFO);
    }

    #[test]
    fn swap_total_policy_is_preserved() {
        let with_swap = "MemTotal: 130000000 kB\nMemAvailable: 1 kB\nSwapTotal: 2000000 kB\nSwapFree: 2000000 kB\n";
        assert_eq!(
            parse_meminfo(with_swap).unwrap().swap_total_bytes,
            Some(2_000_000 * 1024)
        );
        let no_swap = "MemTotal: 130000000 kB\nMemAvailable: 1 kB\nSwapTotal: 0 kB\n";
        assert_eq!(parse_meminfo(no_swap).unwrap().swap_total_bytes, Some(0));
        let absent = "MemTotal: 130000000 kB\nMemAvailable: 1 kB\n";
        assert_eq!(parse_meminfo(absent).unwrap().swap_total_bytes, None);
        let malformed = "MemTotal: 130000000 kB\nMemAvailable: 1 kB\nSwapTotal: lots kB\n";
        assert_eq!(parse_meminfo(malformed).unwrap().swap_total_bytes, None);
    }

    #[test]
    fn missing_or_malformed_required_fields_yield_no_snapshot() {
        assert_eq!(parse_meminfo(""), None);
        assert_eq!(parse_meminfo("MemTotal: 100 kB\n"), None);
        assert_eq!(parse_meminfo("MemAvailable: 100 kB\n"), None);
        assert_eq!(
            parse_meminfo("MemTotal: abc kB\nMemAvailable: 1 kB\n"),
            None
        );
        assert_eq!(
            parse_meminfo("MemTotal: 100 kB\nMemAvailable: -1 kB\n"),
            None
        );
        assert_eq!(
            parse_meminfo("MemTotal: 100 MB\nMemAvailable: 1 kB\n"),
            None
        );
        assert_eq!(
            parse_meminfo("MemTotal: 100 kB extra\nMemAvailable: 1 kB\n"),
            None
        );
        assert_eq!(parse_meminfo("MemTotal:\nMemAvailable: 1 kB\n"), None);
    }

    #[test]
    fn missing_optional_fields_keep_the_snapshot() {
        let m = parse_meminfo("MemTotal: 10 kB\nMemAvailable: 5 kB\n").expect("snapshot");
        assert_eq!(m.total_physical_bytes, 10 * 1024);
        assert_eq!(m.available_physical_bytes, 5 * 1024);
        assert_eq!(m.swap_total_bytes, None);
        assert_eq!(m.commit_limit_bytes, None);
        assert_eq!(m.commit_available_bytes, None);

        // Only one of the commit pair: no derived available figure.
        let m = parse_meminfo("MemTotal: 10 kB\nMemAvailable: 5 kB\nCommitLimit: 8 kB\n").unwrap();
        assert_eq!(m.commit_limit_bytes, Some(8 * 1024));
        assert_eq!(m.commit_available_bytes, None);
    }

    #[test]
    fn malformed_optional_fields_degrade_individually() {
        let body = "MemTotal: 10 kB\nMemAvailable: 5 kB\nSwapTotal: 4 kB\nCommitLimit: x kB\nCommitted_AS: 3 kB\n";
        let m = parse_meminfo(body).expect("snapshot");
        assert_eq!(m.swap_total_bytes, Some(4 * 1024));
        assert_eq!(m.commit_limit_bytes, None);
        assert_eq!(m.commit_available_bytes, None);
    }

    #[test]
    fn kb_to_bytes_overflow_is_malformed() {
        let max_kb = u64::MAX / 1024;
        let ok = format!("MemTotal: {max_kb} kB\nMemAvailable: 1 kB\n");
        assert_eq!(
            parse_meminfo(&ok).unwrap().total_physical_bytes,
            max_kb * 1024
        );
        let over = format!("MemTotal: {} kB\nMemAvailable: 1 kB\n", max_kb + 1);
        assert_eq!(parse_meminfo(&over), None);
        let optional_over = format!(
            "MemTotal: 1 kB\nMemAvailable: 1 kB\nSwapTotal: {} kB\n",
            max_kb + 1
        );
        assert_eq!(
            parse_meminfo(&optional_over).unwrap().swap_total_bytes,
            None
        );
        let wider = "MemTotal: 99999999999999999999999 kB\nMemAvailable: 1 kB\n";
        assert_eq!(parse_meminfo(wider), None);
    }

    #[test]
    fn commit_available_saturates_at_zero_when_overcommitted() {
        let body =
            "MemTotal: 10 kB\nMemAvailable: 5 kB\nCommitLimit: 100 kB\nCommitted_AS: 250 kB\n";
        let m = parse_meminfo(body).unwrap();
        assert_eq!(m.commit_available_bytes, Some(0));
    }

    #[test]
    fn first_duplicate_key_wins_and_prefixed_keys_do_not_match() {
        let body = "MemTotal: 10 kB\nMemTotalFoo: 99 kB\nMemTotal: 77 kB\nMemAvailable: 5 kB\n";
        assert_eq!(parse_meminfo(body).unwrap().total_physical_bytes, 10 * 1024);
        assert_eq!(
            meminfo_field("MemTotalFoo: 1 kB\n", "MemTotal"),
            MeminfoField::Missing
        );
    }

    #[test]
    fn windows_status_maps_commit_and_approximates_swap() {
        let gib = 1u64 << 30;
        let m = host_memory_from_windows_status(64 * gib, 40 * gib, 96 * gib, 70 * gib);
        assert_eq!(m.total_physical_bytes, 64 * gib);
        assert_eq!(m.available_physical_bytes, 40 * gib);
        assert_eq!(m.swap_total_bytes, Some(32 * gib));
        assert_eq!(m.commit_limit_bytes, Some(96 * gib));
        assert_eq!(m.commit_available_bytes, Some(70 * gib));
        assert_eq!(m.source, SOURCE_GLOBAL_MEMORY_STATUS_EX);
    }

    #[test]
    fn windows_status_saturates_swap_and_passes_avail_page_file_through() {
        // Process commit limit below physical: swap approximation is zero.
        let m = host_memory_from_windows_status(64, 10, 32, 8);
        assert_eq!(m.swap_total_bytes, Some(0));
        assert_eq!(m.commit_available_bytes, Some(8));
        // ullAvailPageFile is reported as the OS gave it, never adjusted.
        let m = host_memory_from_windows_status(64, 10, 32, 50);
        assert_eq!(m.commit_available_bytes, Some(50));
    }

    fn budget(vendor_id: u32, luid: [u8; 8], description: &str) -> DxgiAdapterBudget {
        let segment = SegmentBudget {
            budget_bytes: 8,
            current_usage_bytes: 2,
            available_for_reservation_bytes: 4,
            current_reservation_bytes: 1,
        };
        DxgiAdapterBudget {
            description: description.to_string(),
            vendor_id,
            device_id: 0x744c,
            luid,
            dedicated_video_memory_bytes: 24 << 30,
            shared_system_memory_bytes: 32 << 30,
            flags: 0,
            source: DXGI_BUDGET_SOURCE,
            node_index: 0,
            local: segment,
            non_local: segment,
        }
    }

    #[test]
    fn amd_filter_keeps_only_vendor_1002_in_order() {
        let adapters = vec![
            budget(0x10de, [1; 8], "other"),
            budget(AMD_VENDOR_ID, [2; 8], "first amd"),
            budget(0x8086, [3; 8], "integrated"),
            budget(AMD_VENDOR_ID, [4; 8], "second amd"),
        ];
        let names: Vec<&str> = amd_adapters(&adapters)
            .map(|a| a.description.as_str())
            .collect();
        assert_eq!(names, ["first amd", "second amd"]);
        assert_eq!(amd_adapters(&[]).count(), 0);
    }

    #[test]
    fn luid_bytes_are_low_then_high_little_endian() {
        assert_eq!(
            luid_bytes(0x0403_0201, 0x0807_0605),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(
            luid_bytes(0xffff_fffe, -2),
            [0xfe, 0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff]
        );
    }

    #[test]
    fn luid_match_is_exact() {
        let a = budget(AMD_VENDOR_ID, luid_bytes(0xaabb_ccdd, 0), "gpu");
        assert!(a.matches_luid(&[0xdd, 0xcc, 0xbb, 0xaa, 0, 0, 0, 0]));
        assert!(!a.matches_luid(&[0xaa, 0xbb, 0xcc, 0xdd, 0, 0, 0, 0]));
        assert!(!a.matches_luid(&[0xdd, 0xcc, 0xbb, 0xaa, 0, 0, 0, 1]));
    }

    #[test]
    fn software_flag_and_description_decoding() {
        assert!(is_software_adapter(2));
        assert!(is_software_adapter(3));
        assert!(!is_software_adapter(0));
        assert!(!is_software_adapter(1));

        let mut wide = [0u16; 128];
        for (slot, unit) in wide.iter_mut().zip("AMD Radeon".encode_utf16()) {
            *slot = unit;
        }
        assert_eq!(description_from_utf16(&wide), "AMD Radeon");
        let unterminated: Vec<u16> = "full".encode_utf16().collect();
        assert_eq!(description_from_utf16(&unterminated), "full");
        assert_eq!(description_from_utf16(&[]), "");
    }

    #[test]
    fn serde_snapshots_carry_source_and_exact_field_names() {
        let json = serde_json::to_value(budget(AMD_VENDOR_ID, [7; 8], "gpu")).unwrap();
        assert_eq!(json["source"], DXGI_BUDGET_SOURCE);
        assert_eq!(json["description"], "gpu");
        assert_eq!(json["vendor_id"], 0x1002);
        assert_eq!(json["device_id"], 0x744c);
        assert_eq!(json["luid"], serde_json::json!([7, 7, 7, 7, 7, 7, 7, 7]));
        assert_eq!(json["dedicated_video_memory_bytes"], 24u64 << 30);
        assert_eq!(json["shared_system_memory_bytes"], 32u64 << 30);
        assert_eq!(json["flags"], 0);
        assert_eq!(json["node_index"], 0);
        assert_eq!(json["local"]["budget_bytes"], 8);
        assert_eq!(json["local"]["current_usage_bytes"], 2);
        assert_eq!(json["non_local"]["available_for_reservation_bytes"], 4);
        assert_eq!(json["non_local"]["current_reservation_bytes"], 1);

        let host = serde_json::to_value(parse_meminfo(FULL).unwrap()).unwrap();
        assert_eq!(host["source"], SOURCE_PROC_MEMINFO);
        assert_eq!(host["total_physical_bytes"], 131_072_000u64 * 1024);
        let sparse =
            serde_json::to_value(parse_meminfo("MemTotal: 1 kB\nMemAvailable: 1 kB\n").unwrap())
                .unwrap();
        assert!(sparse["swap_total_bytes"].is_null());
        assert!(sparse["commit_available_bytes"].is_null());
    }

    #[test]
    fn dxgi_errors_name_the_failing_step() {
        let e = DxgiError::QueryVideoMemoryInfo {
            index: 1,
            description: "AMD Radeon".to_string(),
            node_index: 0,
            segment: "NON_LOCAL",
            detail: "E_FAIL".to_string(),
        };
        let text = e.to_string();
        assert!(text.contains("QueryVideoMemoryInfo"));
        assert!(text.contains("NON_LOCAL"));
        assert!(text.contains("adapter 1 (AMD Radeon)"));
        assert!(text.contains("E_FAIL"));
        assert!(DxgiError::CreateFactory { detail: "x".into() }
            .to_string()
            .contains("CreateDXGIFactory1"));
    }
}
