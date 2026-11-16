// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.

//! When a VMM map needs a GPU TLB flush ("poke"), and when one must not run.
//!
//! The in-tree amdgpu maps a HIP VMM range through the DRM GEM VA ioctl
//! (ROCm/rocm-systems#11693). For a compute VM its path neither flushes the
//! TLB nor waits for the page-table update; AMD's amdgpu-dkms does both. Two
//! failures follow:
//!
//! * **VA reuse (page tables written by the CPU, the large-BAR default).** A VA
//!   that was mapped, unmapped and mapped again keeps its old translation for
//!   about 1-2 ms: kernels read zeros or write into the previous backing, with
//!   no fault. Reproduced on gfx1100 and gfx1201 (upstream also on gfx1200).
//!   hip-bridge never hands a VA back ([`crate::vmm`] retires every range), so
//!   no hipfire path reaches this today; [`UnmappedVa`] detects a map over a
//!   range unmapped before, so a future path that reuses a VA gets the flush.
//! * **Fresh maps when the page tables are written by SDMA** (visible VRAM
//!   smaller than VRAM, or `amdgpu.vm_update_mode` without the compute CPU
//!   bit): the ioctl returns before the update lands, and work touching the
//!   new range faults. Reported on a USB4-attached R9700.
//!
//! The poke is a throwaway `hipMalloc` + `hipFree` of [`VMM_TLB_POKE_BYTES`]:
//! KFD's unmap ioctl waits for the VM's page-table updates and flushes its TLB.
//! `hipFree` also drains the device, so a poke costs the queued work's time.
//! `hipMalloc` is refused while a global-mode capture is open on any thread
//! (or a thread-local one on this thread), and the refusal invalidates that
//! capture, so [`OpenCaptures`] refuses the map instead of poking then.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::thread::ThreadId;

/// `HIPFIRE_VMM_TLB_POKE` (developer): `1` pokes after every VMM map on every
/// device, `0` never. Unset: on the in-tree amdgpu (no
/// `/sys/module/amdgpu/version`), poke on [`VMM_TLB_POKE_ARCHS`] devices for a
/// map over a previously unmapped VA, and for every map on a device whose
/// compute page tables are written by SDMA.
pub const VMM_TLB_POKE_ENV: &str = "HIPFIRE_VMM_TLB_POKE";

/// Archs the default covers: gfx1100 and gfx1201 reproduce the stale
/// translation (upstream also gfx1200); gfx1030 and gfx1151 did not.
pub const VMM_TLB_POKE_ARCHS: &[&str] = &["gfx1100", "gfx1200", "gfx1201"];

/// Large enough that HIP allocates and frees it through KFD (a 64 KiB block
/// is served from HIP's cache and issues no ioctl).
pub(crate) const VMM_TLB_POKE_BYTES: usize = 4 << 20;

/// hipErrorStreamCaptureUnsupported.
pub(crate) const HIP_ERROR_STREAM_CAPTURE_UNSUPPORTED: u32 = 900;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// `HIPFIRE_VMM_TLB_POKE=1`: every map, every device.
    Forced,
    /// The in-tree default (see [`VMM_TLB_POKE_ENV`]).
    Auto,
}

/// `None` = never poke. Decided once per process.
static MODE: LazyLock<Option<Mode>> = LazyLock::new(|| {
    let in_tree = !std::path::Path::new("/sys/module/amdgpu/version").exists();
    let set = hipfire_config::developer_var(VMM_TLB_POKE_ENV).ok();
    if !hipfire_config::developer_bool(VMM_TLB_POKE_ENV, in_tree) {
        return None;
    }
    Some(if set.is_some() {
        eprintln!(
            "  hip-bridge: VMM TLB flush poke forced after every VMM map ({VMM_TLB_POKE_ENV}=1)"
        );
        Mode::Forced
    } else {
        Mode::Auto
    })
});

pub(crate) fn mode() -> Option<Mode> {
    *MODE
}

static POKES: AtomicU64 = AtomicU64::new(0);

/// Pokes issued by this process (one per device per poked map).
pub fn vmm_tlb_pokes() -> u64 {
    POKES.load(Ordering::Relaxed)
}

pub(crate) fn count_poke() {
    POKES.fetch_add(1, Ordering::Relaxed);
}

/// VA ranges unmapped by this process, coalesced (`start -> end`).
#[derive(Default)]
pub(crate) struct UnmappedVa(BTreeMap<usize, usize>);

impl UnmappedVa {
    pub(crate) fn note(&mut self, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        let (mut start, mut end) = (start, start.saturating_add(len));
        let touching: Vec<(usize, usize)> = self
            .0
            .range(..=end)
            .rev()
            .take_while(|&(_, &e)| e >= start)
            .map(|(&s, &e)| (s, e))
            .collect();
        for (s, e) in touching {
            self.0.remove(&s);
            start = start.min(s);
            end = end.max(e);
        }
        self.0.insert(start, end);
    }

    pub(crate) fn overlaps(&self, start: usize, len: usize) -> bool {
        let end = start.saturating_add(len);
        len != 0
            && self
                .0
                .range(..end)
                .next_back()
                .is_some_and(|(_, &e)| e > start)
    }
}

static UNMAPPED: Mutex<UnmappedVa> = Mutex::new(UnmappedVa(BTreeMap::new()));

fn unmapped() -> MutexGuard<'static, UnmappedVa> {
    UNMAPPED.lock().unwrap_or_else(|e| e.into_inner())
}

/// Record a successful `hipMemUnmap` (only while a poke mode is on).
pub(crate) fn note_unmap(start: usize, len: usize) {
    if mode().is_some() {
        unmapped().note(start, len);
    }
}

/// Whether `start..start+len` overlaps a range this process unmapped before.
pub(crate) fn reuses_va(start: usize, len: usize) -> bool {
    unmapped().overlaps(start, len)
}

/// Whether the compute page tables of a device are written by SDMA (so a map
/// returns before they land). `vm_update_mode`: amdgpu's parameter (`-1`
/// default: the CPU writes compute page tables when all VRAM is CPU-visible;
/// otherwise bit 1 selects the CPU for compute). Unknown sizes count as SDMA.
pub(crate) fn sdma_page_tables(
    vm_update_mode: Option<i64>,
    visible_vram: Option<u64>,
    vram: Option<u64>,
) -> bool {
    match vm_update_mode {
        Some(mode) if mode >= 0 => mode & 2 == 0,
        _ => match (visible_vram, vram) {
            (Some(visible), Some(total)) => visible < total,
            _ => true,
        },
    }
}

/// [`sdma_page_tables`] from sysfs for the PCI device `bdf` (`0000:03:00.0`).
pub(crate) fn sdma_page_tables_sysfs(bdf: &str) -> bool {
    let read = |path: String| {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| s.trim().parse::<i64>().ok())
    };
    let dev = format!("/sys/bus/pci/devices/{}", bdf.to_ascii_lowercase());
    sdma_page_tables(
        read("/sys/module/amdgpu/parameters/vm_update_mode".into()),
        read(format!("{dev}/mem_info_vis_vram_total")).map(|v| v as u64),
        read(format!("{dev}/mem_info_vram_total")).map(|v| v as u64),
    )
}

/// Stream captures opened through [`crate::HipRuntime::stream_begin_capture`]
/// and not yet ended: `(stream, mode, thread)`.
#[derive(Default)]
pub(crate) struct OpenCaptures(Vec<(usize, u32, ThreadId)>);

impl OpenCaptures {
    pub(crate) fn begin(&mut self, stream: usize, mode: u32, thread: ThreadId) {
        self.0.push((stream, mode, thread));
    }

    pub(crate) fn end(&mut self, stream: usize) {
        self.0.retain(|&(s, _, _)| s != stream);
    }

    /// The capture a `hipMalloc` on `thread` would hit: any global-mode (0)
    /// capture, or a thread-local (1) one on `thread`. Relaxed (2) never does.
    pub(crate) fn blocking(&self, thread: ThreadId) -> Option<(u32, ThreadId)> {
        self.0
            .iter()
            .find(|&&(_, mode, t)| mode == 0 || (mode == 1 && t == thread))
            .map(|&(_, mode, t)| (mode, t))
    }
}

static CAPTURES: Mutex<OpenCaptures> = Mutex::new(OpenCaptures(Vec::new()));

pub(crate) fn captures() -> MutexGuard<'static, OpenCaptures> {
    CAPTURES.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unmapped_ranges_coalesce_and_detect_reuse() {
        let mut va = UnmappedVa::default();
        assert!(!va.overlaps(0x1000, 0x1000));
        va.note(0x10_0000, 0x10_0000);
        va.note(0x30_0000, 0x10_0000);
        assert!(va.overlaps(0x10_0000, 1));
        assert!(va.overlaps(0x1f_ffff, 1));
        assert!(!va.overlaps(0x20_0000, 0x10_0000), "the gap is not reused");
        assert!(
            !va.overlaps(0x0f_0000, 0x1_0000),
            "ends exactly at the start"
        );
        assert!(va.overlaps(0x0f_0000, 0x1_0001));
        assert!(
            va.overlaps(0x28_0000, 0x10_0000),
            "straddles into the second"
        );
        assert!(!va.overlaps(0x10_0000, 0));
        va.note(0x20_0000, 0x10_0000);
        assert_eq!(va.0.len(), 1, "adjacent ranges coalesce");
        assert_eq!(va.0.get(&0x10_0000), Some(&0x40_0000));
        va.note(0x18_0000, 0x4_0000);
        assert_eq!(va.0.len(), 1, "a contained range adds nothing");
    }

    #[test]
    fn sdma_page_tables_follow_vm_update_mode_then_bar_size() {
        let gib = 1u64 << 30;
        // Default: full-visible VRAM means CPU updates (no fresh-map race).
        assert!(!sdma_page_tables(Some(-1), Some(32 * gib), Some(32 * gib)));
        assert!(sdma_page_tables(Some(-1), Some(gib / 4), Some(32 * gib)));
        assert!(
            sdma_page_tables(None, None, Some(32 * gib)),
            "unknown: assume SDMA"
        );
        // Explicit parameter: bit 1 is "CPU for compute".
        assert!(sdma_page_tables(Some(0), Some(32 * gib), Some(32 * gib)));
        assert!(sdma_page_tables(Some(1), Some(32 * gib), Some(32 * gib)));
        assert!(!sdma_page_tables(Some(2), Some(gib / 4), Some(32 * gib)));
        assert!(!sdma_page_tables(Some(3), None, None));
    }

    #[test]
    fn only_global_or_own_thread_local_captures_block_hip_malloc() {
        let me = std::thread::current().id();
        let other = std::thread::spawn(|| std::thread::current().id())
            .join()
            .unwrap();
        let mut caps = OpenCaptures::default();
        assert_eq!(caps.blocking(me), None);
        caps.begin(1, 2, me);
        caps.begin(2, 2, other);
        assert_eq!(caps.blocking(me), None, "relaxed never blocks");
        caps.begin(3, 1, other);
        assert_eq!(caps.blocking(me), None, "another thread's thread-local");
        assert_eq!(caps.blocking(other), Some((1, other)));
        caps.begin(4, 0, other);
        assert_eq!(
            caps.blocking(me),
            Some((0, other)),
            "global blocks every thread"
        );
        caps.end(4);
        caps.end(3);
        assert_eq!(caps.blocking(other), None);
        caps.begin(5, 1, me);
        assert_eq!(caps.blocking(me), Some((1, me)));
        caps.end(5);
        assert_eq!(caps.blocking(me), None);
    }
}
