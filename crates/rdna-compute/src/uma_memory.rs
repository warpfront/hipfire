//! Free device memory on unified-memory APUs.
//!
//! On a discrete GPU `hipMemGetInfo` reports private VRAM. On an APU it
//! reports the pool the HSA agent allocates from: the BIOS carve-out, or,
//! when GTT is larger than the carve-out (Strix Halo with a 1 GiB carve-out
//! and a 120 GiB TTM `pages_limit`), TTM's cap minus the GTT in use. That GTT
//! figure ignores everything else on the host (anonymous memory, page cache),
//! so it can report 118 GiB free while `MemAvailable` is 13.5 GiB, and sizing
//! from it reaches the global OOM killer. TTM's page pool hides the other
//! way: freed GTT pages parked there are outside `MemAvailable`, but a new
//! GTT allocation takes them first.
//!
//! [`device_free_from`] is the one rule every free-memory sizing site uses
//! (through `Gpu::device_mem_info`): a discrete GPU, or an APU whose device
//! pool is its carve-out, keeps `hipMemGetInfo`'s figure; a GTT-backed APU
//! takes `min(hipMemGetInfo free, MemAvailable - CMA + TTM pool estimate)`,
//! where CMA is the larger of `CmaFree` and the CMA pageblocks (`CmaTotal`
//! or `/proc/zoneinfo` `cma`, which includes KHO scratch).

use std::path::Path;

/// TTM's page limit, in pages, which caps every GTT allocation on the host.
pub const TTM_PAGES_LIMIT: &str = "/sys/module/ttm/parameters/pages_limit";

/// TTM's page pool cap, in pages. Freed GTT pages past it go back to the
/// kernel at once.
pub const TTM_PAGE_POOL_SIZE: &str = "/sys/module/ttm/parameters/page_pool_size";

/// TTM's page size on x86_64, the only host ROCm supports.
pub const TTM_PAGE_BYTES: u64 = 4096;

/// Host memory outside every `/proc/meminfo` counter that is not TTM's pool:
/// other drivers' pages, DMA buffers, firmware. [`ttm_pool_estimate_from`]
/// never counts this much as pool. Measured on the 5-card gfx1201 host with
/// the pool empty and 46.2 GiB of live GTT: 2.76 GiB.
pub const UNTRACKED_KERNEL_BYTES: u64 = 4 << 30;

/// `/proc/meminfo` fields, in bytes, that together account for every
/// allocated page except driver pages (TTM's pool and live GTT among them).
/// Subset fields (`Shmem`, `Mlocked`, `AnonHugePages`, ...) are left out.
const MEMINFO_TRACKED: &[&str] = &[
    "MemFree",
    "Buffers",
    "Cached",
    "SwapCached",
    "AnonPages",
    "Slab",
    "KernelStack",
    "ShadowCallStack",
    "PageTables",
    "SecPageTables",
    "VmallocUsed",
    "Percpu",
    "Hugetlb",
    "Zswap",
    "Unaccepted",
    "Balloon",
];

/// A `/proc/meminfo` field (`kB`) in bytes.
pub fn meminfo_field(meminfo: &str, key: &str) -> Option<u64> {
    meminfo.lines().find_map(|line| {
        let rest = line.strip_prefix(key)?.strip_prefix(':')?;
        rest.split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()?
            .checked_mul(1024)
    })
}

/// `/proc/zoneinfo` page size. x86_64 is the only host ROCm supports.
const ZONEINFO_PAGE_BYTES: u64 = 4096;

/// Below this, a KHO scratch region is not worth a warning.
pub const KHO_WARN_BYTES: u64 = 1 << 30;

/// Pages of CMA pageblocks summed over every zone of `/proc/zoneinfo` text:
/// the per-zone `cma` line, never `nr_free_cma` (free CMA pages). Includes
/// pageblocks outside any registered CMA area (KHO scratch).
pub fn zoneinfo_cma_pages(zoneinfo: &str) -> u64 {
    zoneinfo
        .lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            (words.next()? == "cma").then_some(())?;
            words.next()?.parse::<u64>().ok()
        })
        .fold(0, u64::saturating_add)
}

/// Bytes of CMA pageblocks: `max(CmaTotal, Σ zoneinfo cma pages)`. Only
/// movable allocations can use them. Without zoneinfo this is `CmaTotal`.
/// A missing `CmaTotal` counts as 0.
pub fn cma_region_bytes(meminfo: &str, zoneinfo: Option<&str>) -> u64 {
    let cma_total = meminfo_field(meminfo, "CmaTotal").unwrap_or(0);
    let zones = zoneinfo
        .map(|text| zoneinfo_cma_pages(text).saturating_mul(ZONEINFO_PAGE_BYTES))
        .unwrap_or(0);
    cma_total.max(zones)
}

/// Bytes of CMA pageblocks outside any registered CMA area (`Σ zoneinfo cma
/// − CmaTotal`), the Kexec HandOver scratch release on kernels with
/// `CONFIG_KEXEC_HANDOVER_ENABLE_DEFAULT=y` (Ubuntu 7.0 generic). `None` when
/// zoneinfo is unreadable or there is none.
pub fn kho_scratch_from(meminfo: &str, zoneinfo: Option<&str>) -> Option<u64> {
    let zones = zoneinfo_cma_pages(zoneinfo?).saturating_mul(ZONEINFO_PAGE_BYTES);
    let scratch = zones.saturating_sub(meminfo_field(meminfo, "CmaTotal").unwrap_or(0));
    (scratch > 0).then_some(scratch)
}

/// Host bytes available to unmovable allocations, including pinned GTT.
///
/// `MemAvailable` includes CMA pageblocks, but only movable allocations can
/// use them: free pages (`CmaFree`) and page cache sitting in them (direct
/// reclaim for unmovable requests skips CMA folios). Subtract
/// `max(CmaFree, cma_region)` ([`cma_region_bytes`]) even when `CmaTotal` is
/// zero (KHO scratch is in no registered area). Missing `CmaFree` counts as
/// zero; missing `MemAvailable` returns `None`.
pub fn meminfo_available_bytes(meminfo: &str, zoneinfo: Option<&str>) -> Option<u64> {
    let cma = meminfo_field(meminfo, "CmaFree")
        .unwrap_or(0)
        .max(cma_region_bytes(meminfo, zoneinfo));
    Some(meminfo_field(meminfo, "MemAvailable")?.saturating_sub(cma))
}

/// Live `/proc/zoneinfo` text; `None` when unreadable.
pub fn read_zoneinfo() -> Option<String> {
    std::fs::read_to_string("/proc/zoneinfo").ok()
}

/// Live [`kho_scratch_from`].
pub fn kho_scratch_bytes() -> Option<u64> {
    kho_scratch_from(
        &std::fs::read_to_string("/proc/meminfo").ok()?,
        read_zoneinfo().as_deref(),
    )
}

/// The warning for KHO scratch of `bytes`, `None` under [`KHO_WARN_BYTES`].
pub fn kho_scratch_warning(bytes: u64) -> Option<String> {
    (bytes >= KHO_WARN_BYTES).then(|| {
        format!(
            "KHO scratch reserves {:.1} GiB unusable by the GPU; boot with kho=off to reclaim it",
            bytes as f64 / (1u64 << 30) as f64
        )
    })
}

/// Live host RAM available to unmovable allocations, without TTM pool credit.
pub fn mem_available_bytes() -> Option<u64> {
    meminfo_available_bytes(
        &std::fs::read_to_string("/proc/meminfo").ok()?,
        read_zoneinfo().as_deref(),
    )
}

/// Total host RAM from `/proc/meminfo`, in bytes.
pub fn host_total_bytes() -> Option<u64> {
    meminfo_field(&std::fs::read_to_string("/proc/meminfo").ok()?, "MemTotal")
}

/// A sysfs integer.
pub fn read_sysfs_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// `mem_info_gtt_used` summed over every amdgpu device. Host allocations may
/// count against another device than the one the process runs on (on a
/// 5-card gfx1201 host, a card-2 process's host-mapped experts show in card
/// 0's `mem_info_gtt_used`).
pub fn amdgpu_gtt_used_bytes() -> Option<u64> {
    let mut used_bytes = 0u64;
    for card in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let name = card.file_name();
        let is_card = name
            .to_str()
            .and_then(|name| name.strip_prefix("card"))
            .is_some_and(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()));
        if is_card {
            used_bytes +=
                read_sysfs_u64(&card.path().join("device/mem_info_gtt_used")).unwrap_or(0);
        }
    }
    Some(used_bytes)
}

/// Estimate of the freed GTT pages TTM keeps in its page pool, from
/// `/proc/meminfo` text, the GTT amdgpu devices hold, and TTM's
/// `page_pool_size` in pages.
///
/// The pool is invisible to an unprivileged process: its pages are in no
/// `/proc/meminfo` counter, so `MemAvailable` excludes them, and
/// `mem_info_gtt_used` drops them when their buffer is freed (only root's
/// `/sys/kernel/debug/ttm/page_pool` reads it). It is `MemTotal` minus every
/// tracked counter, minus the live GTT and [`UNTRACKED_KERNEL_BYTES`], capped
/// at `page_pool_size`. `None` when a field is missing.
pub fn ttm_pool_estimate_from(meminfo: &str, gtt_used: u64, pool_size_pages: u64) -> Option<u64> {
    let field = |key: &str| meminfo_field(meminfo, key);
    let total = field("MemTotal")?;
    field("MemFree")?;
    let tracked: u64 = MEMINFO_TRACKED
        .iter()
        .filter_map(|key| field(key))
        .sum::<u64>()
        + field("KReclaimable")?.saturating_sub(field("SReclaimable")?);
    let untracked = total
        .saturating_sub(tracked)
        .saturating_sub(gtt_used)
        .saturating_sub(UNTRACKED_KERNEL_BYTES);
    Some(untracked.min(pool_size_pages.saturating_mul(TTM_PAGE_BYTES)))
}

/// Host RAM held in TTM's page pool (freed GTT pages amdgpu parks there, up
/// to `page_pool_size`) that the next GTT allocation takes first and TTM's
/// shrinker returns under pressure. 0 when sysfs or `/proc/meminfo` is
/// unreadable. Ungated: callers decide whether their allocations are GTT.
pub fn ttm_pool_bytes() -> u64 {
    let estimate = || -> Option<u64> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let pool_size_pages = read_sysfs_u64(TTM_PAGE_POOL_SIZE.as_ref())?;
        ttm_pool_estimate_from(&meminfo, amdgpu_gtt_used_bytes()?, pool_size_pages)
    };
    estimate().unwrap_or(0)
}

/// Host RAM a GTT allocation can still take: `MemAvailable - CMA` (see
/// [`meminfo_available_bytes`]) plus TTM's pool estimate. `None` when
/// `/proc/meminfo` has no `MemAvailable`.
pub fn host_available_from(
    meminfo: &str,
    zoneinfo: Option<&str>,
    gtt_used: u64,
    pool_size_pages: u64,
) -> Option<u64> {
    let available = meminfo_available_bytes(meminfo, zoneinfo)?;
    let pool = ttm_pool_estimate_from(meminfo, gtt_used, pool_size_pages).unwrap_or(0);
    Some(available.saturating_add(pool))
}

/// Live [`host_available_from`]: `None` when `MemAvailable` is unreadable;
/// an unreadable TTM pool counts as empty, unreadable zoneinfo as no extra
/// CMA region.
pub fn host_available_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let pool_size_pages = read_sysfs_u64(TTM_PAGE_POOL_SIZE.as_ref()).unwrap_or(0);
    let gtt_used = amdgpu_gtt_used_bytes().unwrap_or(0);
    host_available_from(
        &meminfo,
        read_zoneinfo().as_deref(),
        gtt_used,
        pool_size_pages,
    )
}

/// The amdgpu carve-out (`mem_info_vram_total`) of the PCI device
/// `pci_bus_id` (`0000:c0:00.0`), `None` when sysfs does not expose it.
pub fn carveout_total_bytes(pci_bus_id: &str) -> Option<u64> {
    let bdf = pci_bus_id.trim().to_ascii_lowercase();
    read_sysfs_u64(
        &Path::new("/sys/bus/pci/devices")
            .join(bdf)
            .join("mem_info_vram_total"),
    )
}

/// What one free-memory query sees: `hipMemGetInfo` and, on a unified-memory
/// device, the host side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceMemory {
    /// `hipMemGetInfo` free bytes.
    pub device_free: u64,
    /// `hipMemGetInfo` total bytes.
    pub device_total: u64,
    /// `hipDeviceAttributeIntegrated`.
    pub uma: bool,
    /// The carve-out (`mem_info_vram_total`), when sysfs exposes it.
    pub carveout_total: Option<u64>,
    /// `MemAvailable - CMA` plus TTM's pool estimate ([`host_available_from`]).
    pub host_available: Option<u64>,
}

/// Whether a unified-memory device allocates from GTT (system RAM shared
/// with the host) rather than its carve-out: the device pool is larger than
/// the carve-out. An unknown carve-out counts as GTT (the safe answer).
pub fn gtt_backed(device_total: u64, carveout_total: Option<u64>) -> bool {
    carveout_total.is_none_or(|carveout| device_total > carveout)
}

/// Free bytes a device allocation can take. Discrete GPUs and carve-out
/// APUs: `hipMemGetInfo`'s free figure. GTT-backed APUs:
/// `min(hipMemGetInfo free, MemAvailable - CMA + TTM pool)`; `None` when the host
/// side is unreadable there (callers refuse rather than size blind).
pub fn device_free_from(memory: DeviceMemory) -> Option<u64> {
    if !memory.uma || !gtt_backed(memory.device_total, memory.carveout_total) {
        return Some(memory.device_free);
    }
    Some(memory.device_free.min(memory.host_available?))
}

/// Apply the emu capacity cap (EMU-DESIGN.md §3) to a raw `(free, total)`
/// already clamped by [`device_free_from`]. Pure; `Gpu::device_mem_info` feeds
/// it a ledger snapshot only while an emu profile is installed.
///
/// `used` is this process's managed charge (committed + pending), never
/// `real_total - real_free`.
///
/// - Discrete: `total = device_bytes`,
///   `free = min(raw_free, total - used.device)`.
/// - UMA: `total = G` (the GTT limit),
///   `free = min(raw_free, G - used.gtt, S - used.shared_host)`; a device
///   backing also consumes GTT and shared bytes, so those two dimensions
///   already include device charges.
pub fn emu_capped_mem_info(
    raw_free: u64,
    snapshot: &hipfire_config::emu_budget::BudgetSnapshot,
) -> (u64, u64) {
    let limits = &snapshot.limits;
    let used = |committed: u64, pending: u64| committed.saturating_add(pending);
    let total = if limits.uma {
        limits.gtt_bytes.unwrap_or(limits.device_bytes)
    } else {
        limits.device_bytes
    };
    let mut free = raw_free.min(limits.device_bytes.saturating_sub(used(
        snapshot.committed.device_bytes,
        snapshot.pending.device_bytes,
    )));
    if limits.uma {
        if let Some(gtt) = limits.gtt_bytes {
            free = free.min(gtt.saturating_sub(used(
                snapshot.committed.gtt_bytes,
                snapshot.pending.gtt_bytes,
            )));
        }
        if let Some(shared) = limits.shared_host_bytes {
            free = free.min(shared.saturating_sub(used(
                snapshot.committed.shared_host_bytes,
                snapshot.pending.shared_host_bytes,
            )));
        }
    }
    (free, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// hipx (Strix Halo, Ubuntu 7.0 generic, KHO on): `CmaTotal` 0, the KHO
    /// scratch (760 + 8,476 + 4,300 MiB) in the per-zone `cma` lines, and
    /// `CmaFree` down to 0.45 GiB once page cache fills it.
    const HIPX_MEMINFO: &str = "MemAvailable: 70000080 kB\nCmaTotal: 0 kB\nCmaFree: 470684 kB\n";
    const HIPX_ZONEINFO: &str = "\
Node 0, zone      DMA
  pages free     3840
        boost    0
        managed  3840
        cma      0
      nr_free_pages 3840
      nr_free_cma  0
Node 0, zone    DMA32
  pages free     123824
        boost    0
        spanned  1044480
        managed  353871
        cma      194560
      nr_free_pages 123824
      nr_free_cma  61919
Node 0, zone   Normal
  pages free     5000000
        managed  32000000
        cma      3270656
      nr_free_pages 5000000
      nr_free_cma  3000000
Node 0, zone  Movable
        cma      0
      nr_free_cma  0
Node 0, zone   Device
        cma      0
      nr_free_cma  0
";
    const HIPX_KHO_BYTES: u64 = 13_536 << 20;

    #[test]
    fn zoneinfo_cma_sums_the_cma_lines_only() {
        assert_eq!(zoneinfo_cma_pages(HIPX_ZONEINFO), 3_465_216);
        assert_eq!(zoneinfo_cma_pages(HIPX_ZONEINFO) * 4096, HIPX_KHO_BYTES);
        // `nr_free_cma` and malformed lines never count.
        assert_eq!(
            zoneinfo_cma_pages("      nr_free_cma  99\n        cma      x\n"),
            0
        );
        assert_eq!(zoneinfo_cma_pages(""), 0);
    }

    #[test]
    fn hipx_available_excludes_the_whole_kho_scratch() {
        let expected = 70_000_080 * 1024 - HIPX_KHO_BYTES;
        assert_eq!(
            meminfo_available_bytes(HIPX_MEMINFO, Some(HIPX_ZONEINFO)),
            Some(expected)
        );
        assert_eq!(
            host_available_from(HIPX_MEMINFO, Some(HIPX_ZONEINFO), 0, 0),
            Some(expected)
        );
        assert_eq!(
            cma_region_bytes(HIPX_MEMINFO, Some(HIPX_ZONEINFO)),
            HIPX_KHO_BYTES
        );
        assert_eq!(
            kho_scratch_from(HIPX_MEMINFO, Some(HIPX_ZONEINFO)),
            Some(HIPX_KHO_BYTES)
        );
    }

    #[test]
    fn no_cma_host_is_unchanged() {
        let host = "MemAvailable: 4096 kB\n";
        let zoneinfo = "Node 0, zone   Normal\n        cma      0\n      nr_free_cma  0\n";
        assert_eq!(meminfo_available_bytes(host, Some(zoneinfo)), Some(4 << 20));
        assert_eq!(meminfo_available_bytes(host, None), Some(4 << 20));
        assert_eq!(cma_region_bytes(host, Some(zoneinfo)), 0);
        assert_eq!(kho_scratch_from(host, Some(zoneinfo)), None);
        assert_eq!(meminfo_available_bytes("CmaFree: 4096 kB\n", None), None);
    }

    #[test]
    fn real_cma_area_subtracts_the_region_and_is_not_kho() {
        let host = "MemAvailable: 16777216 kB\nCmaTotal: 4194304 kB\nCmaFree: 1048576 kB\n";
        let zoneinfo =
            "Node 0, zone   Normal\n        cma      1048576\n      nr_free_cma  262144\n";
        assert_eq!(
            meminfo_available_bytes(host, Some(zoneinfo)),
            Some(12 * GIB)
        );
        assert_eq!(kho_scratch_from(host, Some(zoneinfo)), None);
        // Without zoneinfo CmaTotal alone is the region.
        assert_eq!(meminfo_available_bytes(host, None), Some(12 * GIB));
    }

    #[test]
    fn cma_free_above_the_region_wins() {
        // No zoneinfo and no CmaTotal: CmaFree is the only figure.
        let host = "MemAvailable: 81450396 kB\nCmaTotal: 0 kB\nCmaFree: 13858816 kB\n";
        assert_eq!(
            meminfo_available_bytes(host, None),
            Some((81_450_396 - 13_858_816) * 1024)
        );
        // CmaFree 5 GiB over a 4 GiB region.
        let host = "MemAvailable: 16777216 kB\nCmaTotal: 4194304 kB\nCmaFree: 5242880 kB\n";
        assert_eq!(meminfo_available_bytes(host, None), Some(11 * GIB));
    }

    #[test]
    fn zoneinfo_unreadable_falls_back_to_cma_total() {
        let host = "MemAvailable: 16777216 kB\nCmaTotal: 2097152 kB\nCmaFree: 1048576 kB\n";
        assert_eq!(meminfo_available_bytes(host, None), Some(14 * GIB));
        assert_eq!(cma_region_bytes(host, None), 2 * GIB);
        assert_eq!(kho_scratch_from(HIPX_MEMINFO, None), None);
    }

    #[test]
    fn meminfo_available_saturates_when_cma_exceeds_available() {
        assert_eq!(
            meminfo_available_bytes("MemAvailable: 1024 kB\nCmaFree: 2048 kB\n", None),
            Some(0)
        );
        let zoneinfo = "Node 0, zone   Normal\n        cma      262144\n";
        assert_eq!(
            meminfo_available_bytes("MemAvailable: 1024 kB\n", Some(zoneinfo)),
            Some(0)
        );
    }

    #[test]
    fn meminfo_available_rejects_byte_count_overflow() {
        assert_eq!(
            meminfo_available_bytes("MemAvailable: 18446744073709551615 kB\n", None),
            None
        );
    }

    #[test]
    fn kho_warning_text_and_threshold() {
        assert_eq!(
            kho_scratch_warning(HIPX_KHO_BYTES).as_deref(),
            Some("KHO scratch reserves 13.2 GiB unusable by the GPU; boot with kho=off to reclaim it")
        );
        assert_eq!(kho_scratch_warning(KHO_WARN_BYTES - 1), None);
        assert!(kho_scratch_warning(KHO_WARN_BYTES).is_some());
    }

    /// `/proc/meminfo` text from `(field, MiB)` pairs.
    fn meminfo(fields: &[(&str, u64)]) -> String {
        fields
            .iter()
            .map(|(key, mib)| format!("{key}:{:>16} kB\n", mib << 10))
            .collect()
    }

    /// The measured 5-card gfx1201 host (MemTotal 128865156 kB, pool cap
    /// 16108144 pages), 5 x 16 MiB of idle GTT, and `pool_mib` of pages outside
    /// every counter on top of `baseline_mib`.
    fn gfx1201_host(pool_mib: u64, baseline_mib: u64) -> String {
        let total = 128_865_156 >> 10;
        let (anon, slab, sreclaim) = (14 << 10, 3 << 10, 2 << 10);
        let cached = total - pool_mib - baseline_mib - 80 - anon - slab - 1024;
        meminfo(&[
            ("MemTotal", total),
            ("MemFree", 1024),
            ("MemAvailable", 45_800),
            ("Cached", cached),
            ("SwapCached", 0),
            ("AnonPages", anon),
            ("Shmem", 2048),
            ("KReclaimable", sreclaim),
            ("Slab", slab),
            ("SReclaimable", sreclaim),
        ])
    }

    /// hipx (Strix Halo, MemTotal 128134124 kB, `page_pool_size` 31457280
    /// pages): `live_gtt_mib` of GTT in use, `pool_mib` parked in TTM's pool,
    /// `anon_mib` of host processes, and `available_mib` of `MemAvailable`.
    fn halo_host(available_mib: u64, anon_mib: u64, pool_mib: u64, live_gtt_mib: u64) -> String {
        let total = 128_134_124 >> 10;
        let (slab, sreclaim, free) = (2 << 10, 1 << 10, 1024);
        // UNTRACKED_KERNEL_BYTES' worth of pages sit outside every counter
        // beside the pool and live GTT.
        let cached = total - anon_mib - slab - free - pool_mib - live_gtt_mib - 4096;
        meminfo(&[
            ("MemTotal", total),
            ("MemFree", free),
            ("MemAvailable", available_mib),
            ("Cached", cached),
            ("AnonPages", anon_mib),
            ("KReclaimable", sreclaim),
            ("Slab", slab),
            ("SReclaimable", sreclaim),
        ])
    }

    const HALO_POOL_PAGES: u64 = 31_457_280;
    const HALO_DEVICE_TOTAL: u64 = 120 * GIB;
    const HALO_CARVEOUT: u64 = GIB;

    fn halo(device_free: u64, meminfo: &str, live_gtt: u64) -> DeviceMemory {
        DeviceMemory {
            device_free,
            device_total: HALO_DEVICE_TOTAL,
            uma: true,
            carveout_total: Some(HALO_CARVEOUT),
            host_available: host_available_from(meminfo, None, live_gtt, HALO_POOL_PAGES),
        }
    }

    #[test]
    fn ttm_pool_estimate_credits_only_untracked_pages_past_the_baseline() {
        const POOL_PAGES: u64 = 16_108_144;
        let gtt = 5 * (16 << 20);
        let mib = |bytes: u64| bytes >> 20;
        // The measured baseline (2.76 GiB, pool empty) is never credited.
        assert_eq!(
            ttm_pool_estimate_from(&gfx1201_host(0, 2826), gtt, POOL_PAGES),
            Some(0)
        );
        // The reported leftover: 15,292,712 pool pages (59,737 MiB).
        let pool = ttm_pool_estimate_from(&gfx1201_host(59_737, 2826), gtt, POOL_PAGES).unwrap();
        assert_eq!(mib(pool), 59_737 + 2826 - (UNTRACKED_KERNEL_BYTES >> 20));
        // Capped at page_pool_size.
        let capped = ttm_pool_estimate_from(&gfx1201_host(59_737, 2826), gtt, 1 << 20).unwrap();
        assert_eq!(capped, (1 << 20) * TTM_PAGE_BYTES);
        // Live GTT (another host-mapped Flash-Next) is not pool.
        let live = ttm_pool_estimate_from(&gfx1201_host(0, 2826), 46 << 30, POOL_PAGES);
        let held = ttm_pool_estimate_from(&gfx1201_host(46 << 10, 2826), 46 << 30, POOL_PAGES);
        assert_eq!((live, held), (Some(0), Some(0)));
    }

    #[test]
    fn discrete_gpu_keeps_the_hip_figure() {
        // A full host (MemAvailable 1 GiB, nothing pooled) changes nothing on
        // a discrete card: its VRAM is private.
        let host = halo_host(1024, 100 << 10, 0, 0);
        let memory = DeviceMemory {
            device_free: 30 * GIB,
            device_total: 32 * GIB,
            uma: false,
            carveout_total: Some(32 * GIB),
            host_available: host_available_from(&host, None, 0, HALO_POOL_PAGES),
        };
        assert_eq!(device_free_from(memory), Some(30 * GIB));
        // Unreadable host memory does not matter there either.
        let blind = DeviceMemory {
            host_available: None,
            carveout_total: None,
            ..memory
        };
        assert_eq!(device_free_from(blind), Some(30 * GIB));
    }

    #[test]
    fn carveout_apu_keeps_the_hip_figure() {
        // The old 96 GiB carve-out layout: hipMemGetInfo is the carve-out,
        // private memory MemAvailable does not include.
        let host = halo_host(20 << 10, 8 << 10, 0, 0);
        let memory = DeviceMemory {
            device_free: 90 * GIB,
            device_total: 96 * GIB,
            uma: true,
            carveout_total: Some(96 * GIB),
            host_available: host_available_from(&host, None, 0, HALO_POOL_PAGES),
        };
        assert_eq!(device_free_from(memory), Some(90 * GIB));
    }

    #[test]
    fn gtt_apu_clamps_to_host_memory_under_pressure() {
        // HaloUma's pressure probe: a 20 GiB host process resident,
        // MemAvailable 13.5 GiB, nothing pooled; hipMemGetInfo still 118 GiB.
        let host = halo_host(13_824, 20 << 10, 0, 0);
        let free = device_free_from(halo(118 * GIB, &host, 0)).unwrap();
        assert_eq!(free, 13_824 << 20);
    }

    #[test]
    fn gtt_apu_counts_freed_gtt_parked_in_the_ttm_pool() {
        // After a Flash-Next process exits: MemAvailable 28 GiB, ~79 GiB of
        // its GTT parked in TTM's pool. The pool is room a GTT allocation
        // takes first.
        let pool_mib = 79 << 10;
        let host = halo_host(28 << 10, 2 << 10, pool_mib, 0);
        let free = device_free_from(halo(118 * GIB, &host, 0)).unwrap();
        assert_eq!(free, (28 << 30) + (pool_mib << 20));
        // The clamp never exceeds what hipMemGetInfo reports.
        let tight = device_free_from(halo(100 * GIB, &host, 0)).unwrap();
        assert_eq!(tight, 100 * GIB);
    }

    #[test]
    fn gtt_apu_live_gtt_is_not_pool() {
        // A resident Flash-Next (79 GiB live GTT) is not free: only
        // MemAvailable remains, and hipMemGetInfo already counts it used.
        let live_mib = 79 << 10;
        let host = halo_host(25 << 10, 2 << 10, 0, live_mib);
        let free = device_free_from(halo(40 * GIB, &host, live_mib << 20)).unwrap();
        assert_eq!(free, 25 * GIB);
    }

    #[test]
    fn gtt_apu_without_host_figures_refuses_and_unknown_carveout_clamps() {
        // MemAvailable unreadable: no figure rather than the 118 GiB.
        let blind = DeviceMemory {
            host_available: None,
            ..halo(118 * GIB, "", 0)
        };
        assert_eq!(device_free_from(blind), None);
        // No sysfs carve-out: treated as GTT-backed.
        let host = halo_host(13_824, 20 << 10, 0, 0);
        let unknown = DeviceMemory {
            carveout_total: None,
            ..halo(118 * GIB, &host, 0)
        };
        assert_eq!(device_free_from(unknown), Some(13_824 << 20));
        // An unreadable TTM pool counts as empty.
        let no_pool = host_available_from(&host, None, 0, 0);
        assert_eq!(no_pool, Some(13_824 << 20));
    }

    mod emu_cap {
        use super::*;
        use hipfire_config::emu_budget::{BudgetLimits, BudgetSnapshot, BudgetUsage};

        fn discrete(cap: u64, committed: u64, pending: u64) -> BudgetSnapshot {
            BudgetSnapshot {
                limits: BudgetLimits {
                    device_bytes: cap,
                    gtt_bytes: None,
                    shared_host_bytes: None,
                    uma: false,
                },
                committed: BudgetUsage {
                    device_bytes: committed,
                    ..Default::default()
                },
                pending: BudgetUsage {
                    device_bytes: pending,
                    ..Default::default()
                },
            }
        }

        fn uma(g: u64, s: u64, committed: BudgetUsage, pending: BudgetUsage) -> BudgetSnapshot {
            BudgetSnapshot {
                limits: BudgetLimits {
                    device_bytes: g,
                    gtt_bytes: Some(g),
                    shared_host_bytes: Some(s),
                    uma: true,
                },
                committed,
                pending,
            }
        }

        #[test]
        fn discrete_total_is_the_profile_capacity() {
            let snap = discrete(16 * GIB, 0, 0);
            assert_eq!(emu_capped_mem_info(32 * GIB, &snap), (16 * GIB, 16 * GIB));
        }

        #[test]
        fn discrete_free_subtracts_committed_and_pending() {
            let snap = discrete(16 * GIB, 5 * GIB, 2 * GIB);
            assert_eq!(emu_capped_mem_info(30 * GIB, &snap), (9 * GIB, 16 * GIB));
        }

        #[test]
        fn discrete_free_is_clamped_by_real_free() {
            // Another process holds the card: real free is below the virtual room.
            let snap = discrete(16 * GIB, GIB, 0);
            assert_eq!(emu_capped_mem_info(3 * GIB, &snap), (3 * GIB, 16 * GIB));
        }

        #[test]
        fn discrete_over_charge_saturates_to_zero() {
            let snap = discrete(16 * GIB, 15 * GIB, 3 * GIB);
            assert_eq!(emu_capped_mem_info(30 * GIB, &snap), (0, 16 * GIB));
        }

        #[test]
        fn uma_total_is_g_and_free_is_min_of_gtt_and_shared_room() {
            let (g, s) = (16 * GIB, 27 * GIB);
            let empty = uma(g, s, BudgetUsage::default(), BudgetUsage::default());
            // Plenty of real memory: the GTT limit G binds.
            assert_eq!(emu_capped_mem_info(100 * GIB, &empty), (g, g));
            // 10 GiB of GTT-pinned host + 4 GiB device: gtt used 14, shared 14.
            let used = BudgetUsage {
                device_bytes: 4 * GIB,
                host_bytes: 10 * GIB,
                gtt_bytes: 14 * GIB,
                shared_host_bytes: 14 * GIB,
            };
            let snap = uma(g, s, used, BudgetUsage::default());
            assert_eq!(emu_capped_mem_info(100 * GIB, &snap), (2 * GIB, g));
        }

        #[test]
        fn uma_shared_host_room_can_bind_before_gtt() {
            // Userptr host-application bytes consume only shared-host room.
            let (g, s) = (16 * GIB, 27 * GIB);
            let used = BudgetUsage {
                device_bytes: 0,
                host_bytes: 25 * GIB,
                gtt_bytes: 0,
                shared_host_bytes: 25 * GIB,
            };
            let snap = uma(g, s, used, BudgetUsage::default());
            assert_eq!(emu_capped_mem_info(100 * GIB, &snap), (2 * GIB, g));
        }

        #[test]
        fn uma_pending_reservations_participate_and_real_free_clamps() {
            let (g, s) = (16 * GIB, 27 * GIB);
            let pending = BudgetUsage {
                device_bytes: 6 * GIB,
                host_bytes: 0,
                gtt_bytes: 6 * GIB,
                shared_host_bytes: 6 * GIB,
            };
            let snap = uma(g, s, BudgetUsage::default(), pending);
            assert_eq!(emu_capped_mem_info(100 * GIB, &snap), (10 * GIB, g));
            // The existing UMA-safe real free (MemAvailable-clamped) still wins.
            assert_eq!(emu_capped_mem_info(3 * GIB, &snap), (3 * GIB, g));
        }
    }
}
