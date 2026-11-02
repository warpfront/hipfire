// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! CPU-only loaded-code snapshot registry (diagnostic, default OFF at the caller).
//!
//! After a HIP module is loaded and its HSA executable is frozen, the caller
//! registers the module's source device ELF. This module finds the matching
//! loaded code object through the AMD loader extension, resolves a *genuine*
//! CPU alias of the loaded allocation through `hsa_amd_pointer_info`, and
//! snapshots the relocated, immutable (non-writable `PT_LOAD`) bytes into a
//! baseline buffer. A same-sized `current` buffer is preallocated. At the first
//! propagated `HipError700` the caller invokes [`capture_after_700`], which
//! copies the cached aliases into the preallocated buffers on the CPU, hashes
//! and byte-compares them with the baseline, and prints one stderr report.
//!
//! # What this is NOT (claim boundaries -- keep in sync with the docs)
//!
//! * A mismatch shows *persistent, CPU-visible* corruption of the loaded
//!   allocation at the time of the later sample. It does NOT identify the
//!   writer, and it is not necessarily the initiating fault.
//! * An equal result excludes only such visible corruption, in the covered
//!   ranges, at the sample time. It does NOT establish that the bytes were equal
//!   at the instant of the fault; it does NOT exclude GPU instruction-cache,
//!   PTE/translation defects, a transient write-and-restore, or a bad-PC / CWSR
//!   root cause. CPU/GPU visibility of unsynchronized GPU writes is a further
//!   limitation.
//! * If the process aborts before the host observes the first 700, there is no
//!   sample at all.
//! * Ranges/modules are `Unavailable` (never reported "intact" or "equal")
//!   whenever any step cannot be proven: missing HSA symbol, non-success status,
//!   no/invalid alias, text mismatch, ambiguity, unparsable image. There is no
//!   hipMemcpy/hsa_memory_copy/GPU fallback.
//!
//! # Why `hsa_amd_pointer_info` and never `hsa_ven_amd_loader_query_host_address`
//!
//! For BASE-profile code ROCr's `RegionMemory` owns a device `ptr_` and a
//! separate staging `host_ptr_`; the loader's host-address query returns the
//! staging copy (copied into the real allocation at Freeze). Hashing that shadow
//! verifies the wrong bytes. The loader table entry is declared below only to
//! keep the table layout; it is never called. The alias used here is
//! `hostBaseAddress + (va - agentBaseAddress)` from `hsa_amd_pointer_info`,
//! with a recognized live allocation, non-null base addresses, and checked full
//! containment of the requested range in `sizeInBytes` (and of the range in the
//! loader's `LOAD_BASE`/`LOAD_SIZE`).
//!
//! # ABI sources (installed ROCm headers, /opt/rocm/include/hsa)
//!
//! * `hsa.h`: `hsa_status_t` (`HSA_STATUS_SUCCESS` = 0, :126),
//!   `hsa_system_major_extension_supported` (:705-709),
//!   `hsa_system_get_major_extension_table` (:785-789),
//!   `HSA_EXTENSION_AMD_LOADER` = 0x201 (:615), `hsa_executable_t` /
//!   `hsa_loaded_code_object_t` / `hsa_agent_t` (`{ uint64_t handle }`),
//!   `hsa_executable_state_t` FROZEN = 1 (:4214), `HSA_EXECUTABLE_INFO_STATE` = 2
//!   (:4489), `hsa_executable_get_info` (:4518).
//! * `hsa_ven_amd_loader.h`: function table `hsa_ven_amd_loader_1_03_pfn_t`
//!   (:622-661, seven pointers in this order), `hsa_ven_amd_loader_iterate_executables`
//!   (:522), `..._executable_iterate_loaded_code_objects` (:273),
//!   `..._loaded_code_object_get_info` (:443); attributes `LOAD_DELTA` = 8
//!   (`int64_t`, signed), `LOAD_BASE` = 9 (`uint64_t`), `LOAD_SIZE` = 10
//!   (`uint64_t`) (:354-369), defined only for a frozen executable.
//! * `hsa_ext_amd.h`: `hsa_amd_pointer_info` (:3192), `hsa_amd_pointer_info_t`
//!   (:3093-3151; `size` field is the version control: set to our struct size
//!   before the call, the runtime lowers it if older; only fields fully inside
//!   the returned `size` are used), `hsa_amd_pointer_type_t` (HSA = 1,
//!   HSA_VMEM = 6; :3019-3048), `HSA_AMD_POINTER_INFO_ALLOC_FLAG_HOST_ACCESS`
//!   = 1 << 4 (:3076; field added in 1.18, consulted only if the returned size
//!   covers it and it is non-zero).
//!
//! # Lifecycle / locking
//!
//! One global `Mutex<Registry>`. [`register_module`] (after load + freeze) and
//! [`unregister_module`] (BEFORE `hipModuleUnload`) take the lock blocking, so an
//! unload can never overlap a capture that is reading the alias. [`capture_after_700`]
//! uses `try_lock`: if busy or poisoned it prints a "skipped" line and returns
//! without consuming the once-only slot and without reading anything. It
//! performs no HIP/HSA call, no allocation of GPU resources, no synchronization;
//! the snapshot is copy + hash + compare on preallocated buffers (no per-range
//! allocation). Report formatting happens afterwards.
//!
//! A loaded object already attributed to a registered module is not attributed
//! again (claimed by `LOAD_BASE`). If a module image is loaded several times
//! (e.g. once per device) and registered after each load, each registration
//! claims the unclaimed matching objects; if several unclaimed identical objects
//! exist at once, all are claimed by that registration (they are real loaded
//! copies of the same image). Every registration leaves a record: either
//! attributed ranges or at least one `unavailable` reason, counted in the report.

use std::ffi::c_void;
#[cfg(target_os = "linux")]
use std::ffi::{c_char, c_int, CStr};
use std::fmt::Write as _;
use std::sync::{Mutex, MutexGuard, TryLockError};

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Public (crate) API
// ---------------------------------------------------------------------------

/// Register a freshly loaded + frozen module's source device ELF (raw ELF64, or
/// an uncompressed clang offload bundle whose ELF entries are all tried).
/// Never panics; an unparsable image or any failed step is recorded as an
/// `Unavailable` entry counted by the capture report. Call after the module
/// load succeeded.
pub(crate) fn register_module(module: usize, name: &str, image: &[u8]) {
    let mut reg = lock_blocking(&REGISTRY);
    reg.register_hsa(module, name, image);
}

/// Drop the module's record (baseline/current buffers, cached aliases). Must be
/// called BEFORE `hipModuleUnload`: it takes the registry lock, so it waits for
/// an in-flight capture and no alias is read after the unload.
pub(crate) fn unregister_module(module: usize) {
    let mut reg = lock_blocking(&REGISTRY);
    reg.remove(module);
}

/// Once-only CPU snapshot compare + single stderr report; see the module docs
/// for the claim boundaries. Safe to call from the first propagated HipError700
/// path: no HIP/HSA call, no sync. A busy/poisoned registry is skipped (and does
/// not consume the once-only slot); later calls after a completed capture are
/// silent.
pub(crate) fn capture_after_700() {
    let _ = capture_into(&REGISTRY, &mut |s| {
        use std::io::Write as _;
        let _ = std::io::stderr().lock().write_all(s.as_bytes());
    });
}

/// Same CPU sample + report as [`capture_after_700`], labelled
/// `hipfire: code-check self-test: ...`, returned as a `String` (never printed
/// here). It never reads or writes the once-only HIP 700 slot, so a later (or
/// earlier, already consumed) [`capture_after_700`] is unaffected. Same
/// `try_lock` rules: busy/poisoned registry yields a "skipped" line; no HIP/HSA
/// call, no sync.
pub(crate) fn self_test_report() -> String {
    let mut out = String::new();
    let _ = capture_mode_into(&REGISTRY, CaptureMode::SelfTest, &mut |s| out.push_str(s));
    out
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

static REGISTRY: Mutex<Registry> = Mutex::new(Registry::new());

fn lock_blocking<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // Registration/unregistration must still work after an unrelated panic so
    // unload can drop its record: records are pushed whole after construction,
    // so a poisoned registry is still structurally consistent. (Capture does NOT
    // do this; it skips explicitly on poison.)
    m.lock().unwrap_or_else(|p| p.into_inner())
}

struct RangeRecord {
    /// Virtual address in the source ELF.
    elf_vaddr: u64,
    /// Loaded VA (`LOAD_DELTA + elf_vaddr`, checked).
    loaded_va: u64,
    /// Validated CPU alias of `loaded_va`. Only ever dereferenced under the
    /// registry lock while the owning module is registered.
    host: usize,
    exec: bool,
    baseline: Box<[u8]>,
    current: Box<[u8]>,
    baseline_hash: [u8; 32],
    current_hash: [u8; 32],
    equal: bool,
}

struct ObjectRecord {
    load_base: u64,
    load_delta: i64,
    ranges: Vec<RangeRecord>,
}

struct ModuleRecord {
    module: usize,
    name: String,
    objects: Vec<ObjectRecord>,
    unavailable: Vec<String>,
}

struct Registry {
    modules: Vec<ModuleRecord>,
    hsa: Option<HsaLoader>,
    captured: bool,
}

impl Registry {
    const fn new() -> Self {
        Registry {
            modules: Vec::new(),
            hsa: None,
            captured: false,
        }
    }

    fn remove(&mut self, module: usize) {
        self.modules.retain(|m| m.module != module);
    }

    fn claimed_bases(&self) -> Vec<u64> {
        self.modules
            .iter()
            .flat_map(|m| m.objects.iter().map(|o| o.load_base))
            .collect()
    }

    fn register_hsa(&mut self, module: usize, name: &str, image: &[u8]) {
        if self.hsa.is_none() {
            match HsaLoader::resolve() {
                Ok(l) => self.hsa = Some(l),
                Err(e) => {
                    self.remove(module);
                    self.modules.push(ModuleRecord {
                        module,
                        name: name.to_owned(),
                        objects: Vec::new(),
                        unavailable: vec![e],
                    });
                    return;
                }
            }
        }
        if let Some(api) = self.hsa {
            self.register_with(&api, module, name, image);
        }
    }

    fn register_with(&mut self, api: &dyn LoaderApi, module: usize, name: &str, image: &[u8]) {
        self.remove(module);
        let claimed = self.claimed_bases();
        let mut attr = attribute(api, &claimed, image);
        if attr.objects.is_empty() && attr.unavailable.is_empty() {
            // Invariant: a registration is never silently dropped.
            attr.unavailable
                .push("registration produced no ranges and no reason".into());
        }
        self.modules.push(ModuleRecord {
            module,
            name: name.to_owned(),
            objects: attr.objects,
            unavailable: attr.unavailable,
        });
    }

    /// Copy + hash + compare on preallocated buffers. No allocation, no HIP/HSA.
    fn sample(&mut self) -> SampleStats {
        let mut st = SampleStats::default();
        for (mi, m) in self.modules.iter_mut().enumerate() {
            st.unavailable += m.unavailable.len();
            for (oi, o) in m.objects.iter_mut().enumerate() {
                for (ri, r) in o.ranges.iter_mut().enumerate() {
                    st.ranges += 1;
                    // SAFETY: `host` was produced by `resolve_alias` (validated
                    // allocation, full containment of `len` bytes) and the module
                    // is still registered; unload is serialized by this lock.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            r.host as *const u8,
                            r.current.as_mut_ptr(),
                            r.current.len(),
                        );
                    }
                    r.current_hash = sha256(&r.current);
                    r.equal = r.current[..] == r.baseline[..];
                    if r.equal {
                        st.equal += 1;
                    } else {
                        st.differ += 1;
                        if st.first.is_none() {
                            st.first = Some((mi, oi, ri));
                        }
                    }
                }
            }
        }
        st
    }

    /// ONE stderr line: summary, then (if any) the first differing range, then
    /// compact unavailable reasons, then the claim-boundary note. The byte
    /// window around the first difference is at most 16 bytes per side.
    fn format_report(&self, mode: CaptureMode, st: &SampleStats) -> String {
        let mut s = String::new();
        let _ = write!(
            s,
            "{} {} ranges, {} equal, {} differ, {} unavailable",
            mode.label(),
            st.ranges,
            st.equal,
            st.differ,
            st.unavailable
        );
        if let Some((mi, oi, ri)) = st.first {
            let m = &self.modules[mi];
            let o = &m.objects[oi];
            let r = &o.ranges[ri];
            // Offset is relative to the start of this range.
            let off = r
                .baseline
                .iter()
                .zip(r.current.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            // <=16 bytes: up to 8 before the first difference, the byte, then the rest.
            let lo = off.saturating_sub(8);
            let hi = (lo + 16).min(r.baseline.len());
            let _ = write!(
                s,
                "; first differing range: module '{}' (handle {:#x}) object base {:#x} delta {:#x} {} range \
                 loaded VA {:#x} (ELF vaddr {:#x}) len {} first diff at offset {} (VA {:#x}) \
                 baseline sha256 {} current sha256 {} baseline[{lo}..{hi}] {} current[{lo}..{hi}] {}",
                m.name,
                m.module,
                o.load_base,
                o.load_delta,
                if r.exec { "exec" } else { "read-only" },
                r.loaded_va,
                r.elf_vaddr,
                r.baseline.len(),
                off,
                r.loaded_va.wrapping_add(off as u64),
                hex(&r.baseline_hash),
                hex(&r.current_hash),
                hex(&r.baseline[lo..hi]),
                hex(&r.current[lo..hi]),
            );
        }
        let mut total = 0usize;
        let mut shown = 0usize;
        let mut list = String::new();
        for m in &self.modules {
            for u in &m.unavailable {
                total += 1;
                if shown < 4 {
                    shown += 1;
                    let _ = write!(
                        list,
                        "{}module '{}' (handle {:#x}): {}",
                        if shown > 1 { " | " } else { "" },
                        m.name,
                        m.module,
                        u.replace('\n', " ")
                    );
                }
            }
        }
        if total > 0 {
            let _ = write!(s, "; unavailable (not checked): {list}");
            if total > shown {
                let _ = write!(s, " | +{} more", total - shown);
            }
        }
        let _ = write!(
            s,
            "; note: differ = persistent CPU-visible loaded-code corruption at this sample (not the writer, not necessarily \
             the initiating fault); equal does not exclude fault-time state, icache/PTE defects, transient writes or a bad PC\n"
        );
        s
    }
}

#[derive(Default)]
struct SampleStats {
    ranges: usize,
    equal: usize,
    differ: usize,
    unavailable: usize,
    first: Option<(usize, usize, usize)>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CaptureOutcome {
    Captured {
        ranges: usize,
        equal: usize,
        differ: usize,
        unavailable: usize,
    },
    AlreadyCaptured,
    /// Registry lock held (registration/unload in progress); nothing sampled.
    Busy,
    /// A panic poisoned the registry; nothing sampled.
    Poisoned,
}

/// Which trigger a sample serves. `After700` is once-only; `SelfTest` never
/// consults or sets the once-only flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaptureMode {
    After700,
    SelfTest,
}

impl CaptureMode {
    /// Line prefix, including the trailing colon.
    fn label(self) -> &'static str {
        match self {
            CaptureMode::After700 => "hipfire: code-check after HIP 700:",
            CaptureMode::SelfTest => "hipfire: code-check self-test:",
        }
    }

    fn once_only(self) -> bool {
        matches!(self, CaptureMode::After700)
    }
}

/// Testable core of [`capture_after_700`]: `reg` supplies the once-only state.
fn capture_into(reg: &Mutex<Registry>, emit: &mut dyn FnMut(&str)) -> CaptureOutcome {
    capture_mode_into(reg, CaptureMode::After700, emit)
}

fn capture_mode_into(
    reg: &Mutex<Registry>,
    mode: CaptureMode,
    emit: &mut dyn FnMut(&str),
) -> CaptureOutcome {
    let mut g = match reg.try_lock() {
        Ok(g) => g,
        Err(TryLockError::Poisoned(_)) => {
            emit(&format!(
                "{} skipped: registry poisoned (no range checked)\n",
                mode.label()
            ));
            return CaptureOutcome::Poisoned;
        }
        Err(TryLockError::WouldBlock) => {
            emit(&format!(
                "{} skipped: registry busy (registration or unload in progress; no range checked)\n",
                mode.label()
            ));
            return CaptureOutcome::Busy;
        }
    };
    if mode.once_only() {
        if g.captured {
            return CaptureOutcome::AlreadyCaptured;
        }
        g.captured = true;
    }
    let st = g.sample();
    let report = g.format_report(mode, &st);
    emit(&report);
    CaptureOutcome::Captured {
        ranges: st.ranges,
        equal: st.equal,
        differ: st.differ,
        unavailable: st.unavailable,
    }
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize().into()
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        let _ = write!(s, "{x:02x}");
    }
    s
}

// ---------------------------------------------------------------------------
// Loader abstraction (HSA behind a small trait so tests use owned memory)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct LoadedObject {
    load_base: u64,
    load_size: u64,
    load_delta: i64,
}

#[derive(Clone, Copy, Debug)]
struct PointerInfo {
    /// Recognized live allocation of a CPU-aliasable kind (HSA or HSA_VMEM).
    known: bool,
    agent_base: u64,
    host_base: u64,
    size: u64,
    /// `Some(host_access)` only when the runtime's returned struct covers
    /// `alloc_flags` and the field is non-zero.
    host_access: Option<bool>,
}

trait LoaderApi {
    /// Loaded code objects of all FROZEN executables.
    fn loaded_objects(&self) -> Result<Vec<LoadedObject>, String>;
    /// `hsa_amd_pointer_info` for `va`. Never the loader's host-address query.
    fn pointer_info(&self, va: u64) -> Result<PointerInfo, String>;
}

/// `delta + vaddr` with checked arithmetic; `None` on wrap in either direction.
fn loaded_va(delta: i64, vaddr: u64) -> Option<u64> {
    vaddr.checked_add_signed(delta)
}

/// Validated CPU alias for `[va, va+len)`.
fn resolve_alias(api: &dyn LoaderApi, va: u64, len: u64) -> Result<usize, String> {
    if len == 0 || va == 0 {
        return Err("empty range or null loaded VA".into());
    }
    va.checked_add(len).ok_or("loaded VA + len overflows")?;
    let info = api.pointer_info(va)?;
    if !info.known {
        return Err(format!(
            "pointer_info: {va:#x} is not a recognized live HSA allocation"
        ));
    }
    if info.agent_base == 0 || info.host_base == 0 {
        return Err(format!(
            "pointer_info: null base (agent {:#x}, host {:#x}): no CPU alias",
            info.agent_base, info.host_base
        ));
    }
    if info.host_access == Some(false) {
        return Err("pointer_info: allocation flags lack HOST_ACCESS".into());
    }
    let off = va
        .checked_sub(info.agent_base)
        .ok_or("loaded VA below allocation agentBaseAddress")?;
    let end = off.checked_add(len).ok_or("range offset + len overflows")?;
    if end > info.size {
        return Err(format!(
            "range [{va:#x}, +{len:#x}) not contained in allocation (agent base {:#x}, size {:#x})",
            info.agent_base, info.size
        ));
    }
    let host = info
        .host_base
        .checked_add(off)
        .ok_or("host alias overflows")?;
    host.checked_add(len).ok_or("host alias + len overflows")?;
    usize::try_from(host).map_err(|_| "host alias exceeds usize".to_string())
}

/// Copy `dst.len()` bytes from a validated alias.
///
/// # Safety
/// `host` must come from [`resolve_alias`] for at least `dst.len()` bytes and the
/// allocation must still be live (registry lock held, module registered).
unsafe fn copy_from_alias(host: usize, dst: &mut [u8]) {
    std::ptr::copy_nonoverlapping(host as *const u8, dst.as_mut_ptr(), dst.len());
}

/// Chunked compare of a validated alias with `src` (early exit, no big alloc).
///
/// # Safety
/// As [`copy_from_alias`] for `src.len()` bytes.
unsafe fn alias_eq(host: usize, src: &[u8]) -> bool {
    let mut buf = [0u8; 4096];
    let mut pos = 0usize;
    while pos < src.len() {
        let n = (src.len() - pos).min(buf.len());
        copy_from_alias(host + pos, &mut buf[..n]);
        if buf[..n] != src[pos..pos + n] {
            return false;
        }
        pos += n;
    }
    true
}

// ---------------------------------------------------------------------------
// ELF64 parsing
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RangePlan {
    vaddr: u64,
    file_off: u64,
    len: u64,
    exec: bool,
}

const PT_LOAD: u32 = 1;
const PF_X: u32 = 1;
const PF_W: u32 = 2;
const SHF_WRITE: u64 = 1;
const SHF_ALLOC: u64 = 2;
const SHT_NULL: u32 = 0;
const SHT_NOBITS: u32 = 8;

fn rd_u16(d: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        d.get(o..o.checked_add(2)?)?.try_into().ok()?,
    ))
}
fn rd_u32(d: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        d.get(o..o.checked_add(4)?)?.try_into().ok()?,
    ))
}
fn rd_u64(d: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        d.get(o..o.checked_add(8)?)?.try_into().ok()?,
    ))
}

/// Non-writable, file-backed `PT_LOAD` bytes, trimmed to non-writable `SHF_ALLOC`
/// non-NOBITS sections inside each segment (drops ELF headers / padding between
/// sections). Writable segments and BSS (`memsz > filesz`) are never selected.
/// If a segment has no qualifying section (stripped section table) the whole
/// file-backed segment (`p_filesz` bytes) is used.
fn parse_elf_ranges(elf: &[u8]) -> Result<Vec<RangePlan>, String> {
    const E: &str = "malformed ELF64";
    if elf.len() < 64 || &elf[..4] != b"\x7fELF" || elf[4] != 2 || elf[5] != 1 {
        return Err("not a little-endian ELF64".into());
    }
    let phoff = rd_u64(elf, 32).ok_or(E)?;
    let shoff = rd_u64(elf, 40).ok_or(E)?;
    let phentsize = rd_u16(elf, 54).ok_or(E)? as u64;
    let phnum = rd_u16(elf, 56).ok_or(E)? as u64;
    let shentsize = rd_u16(elf, 58).ok_or(E)? as u64;
    let shnum = rd_u16(elf, 60).ok_or(E)? as u64;
    if phnum == 0 || phentsize < 56 {
        return Err("no program headers".into());
    }
    struct Seg {
        flags: u32,
        off: u64,
        vaddr: u64,
        filesz: u64,
    }
    let mut segs: Vec<Seg> = Vec::new();
    for i in 0..phnum {
        let base = phoff
            .checked_add(i.checked_mul(phentsize).ok_or(E)?)
            .ok_or(E)?;
        let base = usize::try_from(base).map_err(|_| E)?;
        if rd_u32(elf, base).ok_or(E)? != PT_LOAD {
            continue;
        }
        let flags = rd_u32(elf, base + 4).ok_or(E)?;
        let off = rd_u64(elf, base + 8).ok_or(E)?;
        let vaddr = rd_u64(elf, base + 16).ok_or(E)?;
        let filesz = rd_u64(elf, base + 32).ok_or(E)?;
        let memsz = rd_u64(elf, base + 40).ok_or(E)?;
        if filesz > memsz {
            return Err("PT_LOAD filesz > memsz".into());
        }
        let file_end = off.checked_add(filesz).ok_or(E)?;
        if file_end > elf.len() as u64 {
            return Err("PT_LOAD file range exceeds image".into());
        }
        vaddr.checked_add(memsz).ok_or(E)?;
        if flags & PF_W != 0 || filesz == 0 {
            continue; // writable / BSS-only segments are never selected
        }
        segs.push(Seg {
            flags,
            off,
            vaddr,
            filesz,
        });
    }

    // Candidate non-writable ALLOC non-NOBITS sections: (addr, offset, size).
    let mut secs: Vec<(u64, u64, u64)> = Vec::new();
    if shoff != 0 && shnum != 0 && shentsize >= 64 {
        for i in 0..shnum {
            let Some(base) = shoff.checked_add(i * shentsize) else {
                break;
            };
            let Ok(base) = usize::try_from(base) else {
                break;
            };
            let (Some(ty), Some(fl), Some(addr), Some(off), Some(size)) = (
                rd_u32(elf, base + 4),
                rd_u64(elf, base + 8),
                rd_u64(elf, base + 16),
                rd_u64(elf, base + 24),
                rd_u64(elf, base + 32),
            ) else {
                break;
            };
            if ty != SHT_NULL
                && ty != SHT_NOBITS
                && fl & SHF_ALLOC != 0
                && fl & SHF_WRITE == 0
                && size != 0
            {
                secs.push((addr, off, size));
            }
        }
    }

    let mut out: Vec<RangePlan> = Vec::new();
    for seg in &segs {
        let seg_end = seg.vaddr + seg.filesz; // checked via vaddr+memsz above
        let mut inside: Vec<(u64, u64)> = Vec::new(); // (addr, end)
        for &(addr, off, size) in &secs {
            let Some(end) = addr.checked_add(size) else {
                continue;
            };
            if addr >= seg.vaddr && end <= seg_end && off == seg.off + (addr - seg.vaddr) {
                inside.push((addr, end));
            }
        }
        inside.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (a, e) in inside {
            match merged.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(e),
                _ => merged.push((a, e)),
            }
        }
        if merged.is_empty() {
            merged.push((seg.vaddr, seg_end));
        }
        for (a, e) in merged {
            out.push(RangePlan {
                vaddr: a,
                file_off: seg.off + (a - seg.vaddr),
                len: e - a,
                exec: seg.flags & PF_X != 0,
            });
        }
    }
    out.sort_by_key(|r| r.vaddr);
    for w in out.windows(2) {
        if w[0].vaddr + w[0].len > w[1].vaddr {
            return Err("overlapping loadable ranges".into());
        }
    }
    if out.iter().all(|r| !r.exec) {
        return Err("no executable non-writable PT_LOAD range".into());
    }
    Ok(out)
}

/// ELF images inside `image`: the image itself if it is ELF, or every ELF entry
/// of an uncompressed clang offload bundle. Compressed bundles (`CCOB`) and
/// anything else are an error (recorded as Unavailable by the caller).
fn candidate_elfs(image: &[u8]) -> Result<Vec<&[u8]>, String> {
    if image.starts_with(b"\x7fELF") {
        return Ok(vec![image]);
    }
    const MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
    if !image.starts_with(MAGIC) {
        return Err("image is neither ELF nor an uncompressed clang offload bundle".into());
    }
    let n = rd_u64(image, MAGIC.len()).ok_or("truncated bundle")?;
    let mut pos = MAGIC.len() + 8;
    let mut out = Vec::new();
    for _ in 0..n {
        let off = rd_u64(image, pos).ok_or("truncated bundle entry")?;
        let size = rd_u64(image, pos + 8).ok_or("truncated bundle entry")?;
        let idlen = rd_u64(image, pos + 16).ok_or("truncated bundle entry")?;
        let next = usize::try_from(idlen)
            .ok()
            .and_then(|l| pos.checked_add(24)?.checked_add(l))
            .ok_or("bundle id overflow")?;
        if next > image.len() {
            return Err("truncated bundle id".into());
        }
        pos = next;
        let Some(end) = off.checked_add(size) else {
            continue;
        };
        if end <= image.len() as u64 && size >= 4 {
            let e = &image[off as usize..end as usize];
            if e.starts_with(b"\x7fELF") {
                out.push(e);
            }
        }
    }
    if out.is_empty() {
        return Err("bundle has no ELF entry".into());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Attribution: match loaded objects to the source image
// ---------------------------------------------------------------------------

struct Attribution {
    objects: Vec<ObjectRecord>,
    unavailable: Vec<String>,
}

enum MatchResult {
    Match(ObjectRecord),
    /// Executable text read through a validated alias differs from the source.
    Mismatch,
    /// Alias/metadata not provable before text could be compared.
    NotProvable(String),
    /// Text matched (it is ours) but a later step failed.
    OursUnavailable(String),
}

fn attribute(api: &dyn LoaderApi, claimed: &[u64], image: &[u8]) -> Attribution {
    let mut objects = Vec::new();
    let mut unavailable = Vec::new();
    let cands = match candidate_elfs(image) {
        Ok(c) => c,
        Err(e) => {
            return Attribution {
                objects,
                unavailable: vec![e],
            }
        }
    };
    let mut plans: Vec<(&[u8], Vec<RangePlan>)> = Vec::new();
    let mut parse_notes = Vec::new();
    for c in cands {
        match parse_elf_ranges(c) {
            Ok(p) => plans.push((c, p)),
            Err(e) => parse_notes.push(e),
        }
    }
    if plans.is_empty() {
        return Attribution {
            objects,
            unavailable: vec![format!(
                "no usable ELF in image: {}",
                parse_notes.join("; ")
            )],
        };
    }
    let loaded = match api.loaded_objects() {
        Ok(l) => l,
        Err(e) => {
            return Attribution {
                objects,
                unavailable: vec![e],
            }
        }
    };
    let (mut examined, mut mismatched, mut not_provable) = (0usize, 0usize, 0usize);
    let mut first_reason: Option<String> = None;
    for obj in &loaded {
        if obj.load_base == 0 || obj.load_size == 0 || claimed.contains(&obj.load_base) {
            continue;
        }
        examined += 1;
        for (elf, plan) in &plans {
            match match_object(api, obj, elf, plan) {
                MatchResult::Match(rec) => {
                    objects.push(rec);
                    break;
                }
                MatchResult::OursUnavailable(e) => {
                    unavailable.push(format!("object base {:#x}: {e}", obj.load_base));
                    break;
                }
                MatchResult::Mismatch => mismatched += 1,
                MatchResult::NotProvable(e) => {
                    not_provable += 1;
                    first_reason.get_or_insert(e);
                }
            }
        }
    }
    if objects.is_empty() && unavailable.is_empty() {
        unavailable.push(format!(
            "no loaded code object matched the image text ({examined} unclaimed objects examined, \
             {mismatched} text mismatches, {not_provable} without a provable CPU alias{})",
            first_reason
                .map(|r| format!("; first: {r}"))
                .unwrap_or_default()
        ));
    }
    Attribution {
        objects,
        unavailable,
    }
}

fn match_object(
    api: &dyn LoaderApi,
    obj: &LoadedObject,
    elf: &[u8],
    plan: &[RangePlan],
) -> MatchResult {
    let obj_end = match obj.load_base.checked_add(obj.load_size) {
        Some(e) => e,
        None => return MatchResult::NotProvable("LOAD_BASE + LOAD_SIZE overflows".into()),
    };
    // Resolve every range's loaded VA + alias (checked) before reading anything.
    let mut resolved: Vec<(RangePlan, u64, usize)> = Vec::with_capacity(plan.len());
    for r in plan {
        let Some(va) = loaded_va(obj.load_delta, r.vaddr) else {
            return MatchResult::NotProvable(format!(
                "LOAD_DELTA {:#x} + vaddr {:#x} overflows",
                obj.load_delta, r.vaddr
            ));
        };
        let Some(va_end) = va.checked_add(r.len) else {
            return MatchResult::NotProvable(format!(
                "loaded VA {va:#x} + len {:#x} overflows",
                r.len
            ));
        };
        if va < obj.load_base || va_end > obj_end {
            return MatchResult::NotProvable(format!(
                "range [{va:#x}, {va_end:#x}) outside loaded object [{:#x}, {obj_end:#x})",
                obj.load_base
            ));
        }
        let Some(src_end) = r.file_off.checked_add(r.len) else {
            return MatchResult::NotProvable("source range overflows".into());
        };
        if src_end > elf.len() as u64 {
            return MatchResult::NotProvable("source range outside image".into());
        }
        match resolve_alias(api, va, r.len) {
            Ok(h) => resolved.push((*r, va, h)),
            Err(e) => {
                if r.exec {
                    return MatchResult::NotProvable(e);
                }
                // A read-only alias failure matters only once the text matched;
                // defer it (0 is never a valid alias) and re-resolve strictly.
                resolved.push((*r, va, 0));
            }
        }
    }
    // Text attribution: all executable ranges through the validated alias.
    for (r, _, host) in &resolved {
        if !r.exec {
            continue;
        }
        let src = &elf[r.file_off as usize..(r.file_off + r.len) as usize];
        // SAFETY: `host` validated by resolve_alias for r.len bytes; the caller
        // holds the registry lock and the executable is frozen and live.
        if !unsafe { alias_eq(*host, src) } {
            return MatchResult::Mismatch;
        }
    }
    // Ours. Re-resolve deferred read-only ranges strictly and snapshot baselines.
    let mut ranges = Vec::with_capacity(resolved.len());
    for (r, va, host) in resolved {
        let host = if host == 0 {
            match resolve_alias(api, va, r.len) {
                Ok(h) => h,
                Err(e) => {
                    return MatchResult::OursUnavailable(format!(
                        "text matched but read-only range at {va:#x} has no CPU alias: {e}"
                    ))
                }
            }
        } else {
            host
        };
        let len = r.len as usize;
        let mut baseline = vec![0u8; len].into_boxed_slice();
        // SAFETY: validated alias, see above.
        unsafe { copy_from_alias(host, &mut baseline) };
        let baseline_hash = sha256(&baseline);
        ranges.push(RangeRecord {
            elf_vaddr: r.vaddr,
            loaded_va: va,
            host,
            exec: r.exec,
            current: vec![0u8; len].into_boxed_slice(),
            baseline,
            baseline_hash,
            current_hash: [0; 32],
            equal: true,
        });
    }
    MatchResult::Match(ObjectRecord {
        load_base: obj.load_base,
        load_delta: obj.load_delta,
        ranges,
    })
}

// ---------------------------------------------------------------------------
// Real HSA adapter (libhsa-runtime64 already loaded; RTLD_NOLOAD; no init/shutdown)
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}
#[cfg(target_os = "linux")]
const RTLD_NOW: c_int = 0x2;
#[cfg(target_os = "linux")]
const RTLD_NOLOAD: c_int = 0x4;

const HSA_STATUS_SUCCESS: u32 = 0;
const HSA_EXTENSION_AMD_LOADER: u16 = 0x201;
const HSA_EXECUTABLE_INFO_STATE: u32 = 2;
const HSA_EXECUTABLE_STATE_FROZEN: u32 = 1;
const LOADER_INFO_LOAD_DELTA: u32 = 8;
const LOADER_INFO_LOAD_BASE: u32 = 9;
const LOADER_INFO_LOAD_SIZE: u32 = 10;
const POINTER_TYPE_HSA: u32 = 1;
const POINTER_TYPE_HSA_VMEM: u32 = 6;
const ALLOC_FLAG_HOST_ACCESS: u32 = 1 << 4;

#[repr(C)]
#[derive(Clone, Copy)]
struct HsaExecutable {
    handle: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct HsaLoadedCodeObject {
    handle: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)] // layout only
struct HsaAgent {
    handle: u64,
}

/// `hsa_amd_pointer_info_t` (hsa_ext_amd.h:3096-3151), fields in header order.
#[repr(C)]
#[allow(dead_code)] // layout mirrors the header; only a prefix is read
struct AmdPointerInfo {
    size: u32,
    ptr_type: u32,
    agent_base_address: *mut c_void,
    host_base_address: *mut c_void,
    size_in_bytes: usize,
    user_data: *mut c_void,
    agent_owner: HsaAgent,
    global_flags: u32,
    registered: bool,
    alloc_flags: u32,
}

type SupportedFn = unsafe extern "C" fn(u16, u16, *mut u16, *mut bool) -> u32;
type GetTableFn = unsafe extern "C" fn(u16, u16, usize, *mut c_void) -> u32;
type PointerInfoFn = unsafe extern "C" fn(
    *const c_void,
    *mut AmdPointerInfo,
    Option<unsafe extern "C" fn(usize) -> *mut c_void>,
    *mut u32,
    *mut *mut HsaAgent,
) -> u32;
type ExecGetInfoFn = unsafe extern "C" fn(HsaExecutable, u32, *mut c_void) -> u32;
type IterExecCb = unsafe extern "C" fn(HsaExecutable, *mut c_void) -> u32;
type IterExecFn = unsafe extern "C" fn(Option<IterExecCb>, *mut c_void) -> u32;
type IterLcoCb = unsafe extern "C" fn(HsaExecutable, HsaLoadedCodeObject, *mut c_void) -> u32;
type IterLcoFn = unsafe extern "C" fn(HsaExecutable, Option<IterLcoCb>, *mut c_void) -> u32;
type LcoGetInfoFn = unsafe extern "C" fn(HsaLoadedCodeObject, u32, *mut c_void) -> u32;
type OpaqueFn = Option<unsafe extern "C" fn()>;

/// `hsa_ven_amd_loader_1_03_pfn_t` (hsa_ven_amd_loader.h:622-661). Pointer order
/// is part of the ABI. The `unused_*` entries are never called (notably
/// `query_host_address`, see module docs).
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)] // layout only for the unused_* entries
struct LoaderTable103 {
    unused_query_host_address: OpaqueFn,
    unused_query_segment_descriptors: OpaqueFn,
    unused_query_executable: OpaqueFn,
    iterate_loaded_code_objects: Option<IterLcoFn>,
    loaded_code_object_get_info: Option<LcoGetInfoFn>,
    unused_reader_create_from_file_with_offset_size: OpaqueFn,
    iterate_executables: Option<IterExecFn>,
}

#[derive(Clone, Copy)]
struct HsaLoader {
    pointer_info: PointerInfoFn,
    exec_get_info: ExecGetInfoFn,
    iterate_executables: IterExecFn,
    iterate_lcos: IterLcoFn,
    lco_get_info: LcoGetInfoFn,
}

unsafe extern "C" fn collect_executable(e: HsaExecutable, data: *mut c_void) -> u32 {
    (*(data as *mut Vec<HsaExecutable>)).push(e);
    HSA_STATUS_SUCCESS
}

unsafe extern "C" fn collect_lco(
    _e: HsaExecutable,
    l: HsaLoadedCodeObject,
    data: *mut c_void,
) -> u32 {
    (*(data as *mut Vec<HsaLoadedCodeObject>)).push(l);
    HSA_STATUS_SUCCESS
}

impl HsaLoader {
    #[cfg(target_os = "linux")]
    fn resolve() -> Result<HsaLoader, String> {
        // SAFETY: dlopen/dlsym on a library that must already be loaded; symbol
        // types are the documented signatures from the installed headers.
        unsafe {
            let mut handle = std::ptr::null_mut();
            for name in [c"libhsa-runtime64.so.1", c"libhsa-runtime64.so"] {
                handle = dlopen(name.as_ptr(), RTLD_NOW | RTLD_NOLOAD);
                if !handle.is_null() {
                    break;
                }
            }
            if handle.is_null() {
                return Err("libhsa-runtime64 is not loaded in this process (RTLD_NOLOAD)".into());
            }
            let r = Self::resolve_from(handle);
            if r.is_err() {
                // Drop the reference taken by dlopen(RTLD_NOLOAD) so repeated failed
                // attempts do not leak one reference each. On success the reference
                // is kept: the cached function pointers must stay valid for the
                // process lifetime. This is not hsa_init/hsa_shut_down.
                dlclose(handle);
            }
            r
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn resolve() -> Result<HsaLoader, String> {
        Err(format!(
            "loaded-code HSA adapter is unsupported on {} (Linux only)",
            std::env::consts::OS
        ))
    }

    /// # Safety
    /// `handle` must be a live handle returned by `dlopen`.
    #[cfg(target_os = "linux")]
    unsafe fn resolve_from(handle: *mut c_void) -> Result<HsaLoader, String> {
        unsafe {
            let sym = |n: &CStr| -> Result<*mut c_void, String> {
                let p = dlsym(handle, n.as_ptr());
                if p.is_null() {
                    Err(format!("missing HSA symbol {}", n.to_string_lossy()))
                } else {
                    Ok(p)
                }
            };
            let supported: SupportedFn =
                std::mem::transmute(sym(c"hsa_system_major_extension_supported")?);
            let get_table: GetTableFn =
                std::mem::transmute(sym(c"hsa_system_get_major_extension_table")?);
            let pointer_info: PointerInfoFn = std::mem::transmute(sym(c"hsa_amd_pointer_info")?);
            let exec_get_info: ExecGetInfoFn =
                std::mem::transmute(sym(c"hsa_executable_get_info")?);

            let (mut minor, mut ok) = (0u16, false);
            let st = supported(HSA_EXTENSION_AMD_LOADER, 1, &mut minor, &mut ok);
            if st != HSA_STATUS_SUCCESS || !ok {
                return Err(format!(
                    "AMD loader extension major 1 unsupported (status {st:#x})"
                ));
            }
            // Do not gate on `minor`: ROCr reports loader minor 0 yet fills the
            // full 1.03 table (rocm-systems 18f797bd hsa.cpp:315-318 vs :470-489;
            // seen as "1.0" on ROCr 1.21). The table is zero-initialised and
            // copied for at most our size, so a runtime without the 1.03
            // entries leaves them null and is rejected below.
            let _ = minor;
            let mut t = LoaderTable103::default();
            let st = get_table(
                HSA_EXTENSION_AMD_LOADER,
                1,
                std::mem::size_of::<LoaderTable103>(),
                &mut t as *mut _ as *mut c_void,
            );
            if st != HSA_STATUS_SUCCESS {
                return Err(format!(
                    "hsa_system_get_major_extension_table failed ({st:#x})"
                ));
            }
            let (Some(iterate_executables), Some(iterate_lcos), Some(lco_get_info)) = (
                t.iterate_executables,
                t.iterate_loaded_code_objects,
                t.loaded_code_object_get_info,
            ) else {
                return Err("AMD loader table has null required entries".into());
            };
            Ok(HsaLoader {
                pointer_info,
                exec_get_info,
                iterate_executables,
                iterate_lcos,
                lco_get_info,
            })
        }
    }

    fn get_info<T: Copy>(&self, l: HsaLoadedCodeObject, attr: u32, mut v: T) -> Result<T, String> {
        // SAFETY: attribute types are fixed by the header (:354-369); `v` has the
        // exact documented size (int64_t / uint64_t).
        let st = unsafe { (self.lco_get_info)(l, attr, &mut v as *mut T as *mut c_void) };
        if st != HSA_STATUS_SUCCESS {
            return Err(format!(
                "loaded_code_object_get_info({attr}) failed ({st:#x})"
            ));
        }
        Ok(v)
    }
}

impl LoaderApi for HsaLoader {
    fn loaded_objects(&self) -> Result<Vec<LoadedObject>, String> {
        let mut execs: Vec<HsaExecutable> = Vec::new();
        // SAFETY: the callback only pushes into `execs`; it must not call
        // executable create/destroy (header :499-509), so enumeration is two-phase.
        let st = unsafe {
            (self.iterate_executables)(
                Some(collect_executable),
                &mut execs as *mut _ as *mut c_void,
            )
        };
        if st != HSA_STATUS_SUCCESS {
            return Err(format!("iterate_executables failed ({st:#x})"));
        }
        let mut out = Vec::new();
        for e in execs {
            let mut state = 0u32;
            // SAFETY: documented out-type for HSA_EXECUTABLE_INFO_STATE is a 4-byte enum.
            let st = unsafe {
                (self.exec_get_info)(
                    e,
                    HSA_EXECUTABLE_INFO_STATE,
                    &mut state as *mut _ as *mut c_void,
                )
            };
            if st != HSA_STATUS_SUCCESS {
                return Err(format!("hsa_executable_get_info(STATE) failed ({st:#x})"));
            }
            if state != HSA_EXECUTABLE_STATE_FROZEN {
                continue;
            }
            let mut lcos: Vec<HsaLoadedCodeObject> = Vec::new();
            // SAFETY: as above.
            let st = unsafe {
                (self.iterate_lcos)(e, Some(collect_lco), &mut lcos as *mut _ as *mut c_void)
            };
            if st != HSA_STATUS_SUCCESS {
                return Err(format!("iterate_loaded_code_objects failed ({st:#x})"));
            }
            for l in lcos {
                let delta: i64 = self.get_info(l, LOADER_INFO_LOAD_DELTA, 0i64)?;
                let base: u64 = self.get_info(l, LOADER_INFO_LOAD_BASE, 0u64)?;
                let size: u64 = self.get_info(l, LOADER_INFO_LOAD_SIZE, 0u64)?;
                out.push(LoadedObject {
                    load_base: base,
                    load_size: size,
                    load_delta: delta,
                });
            }
        }
        Ok(out)
    }

    fn pointer_info(&self, va: u64) -> Result<PointerInfo, String> {
        let want = std::mem::size_of::<AmdPointerInfo>() as u32;
        let mut info = AmdPointerInfo {
            size: want,
            ptr_type: 0,
            agent_base_address: std::ptr::null_mut(),
            host_base_address: std::ptr::null_mut(),
            size_in_bytes: 0,
            user_data: std::ptr::null_mut(),
            agent_owner: HsaAgent { handle: 0 },
            global_flags: 0,
            registered: false,
            alloc_flags: 0,
        };
        // SAFETY: `info.size` is set; alloc/num/accessible are null so no agent
        // list is returned (header :3125-3140).
        let st = unsafe {
            (self.pointer_info)(
                va as usize as *const c_void,
                &mut info,
                None,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if st != HSA_STATUS_SUCCESS {
            return Err(format!("hsa_amd_pointer_info failed ({st:#x})"));
        }
        // Only trust fields wholly inside the runtime-reported prefix.
        let covers = |end: usize| (info.size as usize) >= end;
        let used_end =
            std::mem::offset_of!(AmdPointerInfo, size_in_bytes) + std::mem::size_of::<usize>();
        if !covers(used_end) {
            return Err(format!(
                "pointer_info struct too old (returned size {}, need {used_end})",
                info.size
            ));
        }
        let flags_end =
            std::mem::offset_of!(AmdPointerInfo, alloc_flags) + std::mem::size_of::<u32>();
        let host_access = if covers(flags_end) && info.alloc_flags != 0 {
            Some(info.alloc_flags & ALLOC_FLAG_HOST_ACCESS != 0)
        } else {
            None
        };
        Ok(PointerInfo {
            known: info.ptr_type == POINTER_TYPE_HSA || info.ptr_type == POINTER_TYPE_HSA_VMEM,
            agent_base: info.agent_base_address as usize as u64,
            host_base: info.host_base_address as usize as u64,
            size: info.size_in_bytes as u64,
            host_access,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests (owned memory only; the fake adapter never hands out arbitrary pointers)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    // Synthetic ELF64 layout.
    const RO_VA: u64 = 0x100; // 64-byte KD-like .rodata, R segment [0, 0x140)
    const TEXT_VA: u64 = 0x1000; // 0x180 bytes .text, RX segment
    const TEXT_LEN: usize = 0x180;
    const DATA_VA: u64 = 0x2000; // RW segment, filesz 0x40 memsz 0x100
    const IMG_SPAN: usize = 0x3000;

    fn w16(v: &mut [u8], o: usize, x: u16) {
        v[o..o + 2].copy_from_slice(&x.to_le_bytes())
    }
    fn w32(v: &mut [u8], o: usize, x: u32) {
        v[o..o + 4].copy_from_slice(&x.to_le_bytes())
    }
    fn w64(v: &mut [u8], o: usize, x: u64) {
        v[o..o + 8].copy_from_slice(&x.to_le_bytes())
    }

    fn build_elf() -> Vec<u8> {
        let shoff = 0x2100usize;
        let mut e = vec![0u8; shoff + 5 * 64];
        e[..4].copy_from_slice(b"\x7fELF");
        e[4] = 2;
        e[5] = 1;
        w64(&mut e, 32, 64); // phoff
        w64(&mut e, 40, shoff as u64);
        w16(&mut e, 54, 56);
        w16(&mut e, 56, 3);
        w16(&mut e, 58, 64);
        w16(&mut e, 60, 5);
        let ph = |e: &mut Vec<u8>, i: usize, fl: u32, off: u64, va: u64, fsz: u64, msz: u64| {
            let b = 64 + i * 56;
            w32(e, b, PT_LOAD);
            w32(e, b + 4, fl);
            w64(e, b + 8, off);
            w64(e, b + 16, va);
            w64(e, b + 24, va);
            w64(e, b + 32, fsz);
            w64(e, b + 40, msz);
            w64(e, b + 48, 0x1000);
        };
        ph(&mut e, 0, 4, 0, 0, 0x140, 0x140);
        ph(
            &mut e,
            1,
            5,
            0x1000,
            TEXT_VA,
            TEXT_LEN as u64,
            TEXT_LEN as u64,
        );
        ph(&mut e, 2, 6, 0x2000, DATA_VA, 0x40, 0x100);
        // Only .rodata of the R segment is selected (headers/padding are not).
        for (k, b) in e[0x100..0x140].iter_mut().enumerate() {
            *b = (k as u8).wrapping_mul(5).wrapping_add(1);
        }
        for (k, b) in e[0x1000..0x1000 + TEXT_LEN].iter_mut().enumerate() {
            *b = (k as u8).wrapping_mul(13).wrapping_add(0x55);
        }
        for (k, b) in e[0x2000..0x2040].iter_mut().enumerate() {
            *b = 0xA0 ^ k as u8;
        }
        let sh = |e: &mut Vec<u8>, i: usize, ty: u32, fl: u64, addr: u64, off: u64, size: u64| {
            let b = shoff + i * 64;
            w32(e, b + 4, ty);
            w64(e, b + 8, fl);
            w64(e, b + 16, addr);
            w64(e, b + 24, off);
            w64(e, b + 32, size);
        };
        sh(&mut e, 1, 1, SHF_ALLOC, RO_VA, 0x100, 0x40); // .rodata
        sh(
            &mut e,
            2,
            1,
            SHF_ALLOC | 4,
            TEXT_VA,
            0x1000,
            TEXT_LEN as u64,
        ); // .text
        sh(&mut e, 3, 1, SHF_ALLOC | SHF_WRITE, DATA_VA, 0x2000, 0x40); // .data
        sh(
            &mut e,
            4,
            SHT_NOBITS,
            SHF_ALLOC | SHF_WRITE,
            DATA_VA + 0x40,
            0x2040,
            0xc0,
        ); // .bss
        e
    }

    /// "Loaded" memory built from the ELF's PT_LOAD file bytes.
    fn load_image(elf: &[u8]) -> Box<[u8]> {
        let mut m = vec![0u8; IMG_SPAN].into_boxed_slice();
        m[0..0x140].copy_from_slice(&elf[0..0x140]);
        m[0x1000..0x1000 + TEXT_LEN].copy_from_slice(&elf[0x1000..0x1000 + TEXT_LEN]);
        m[0x2000..0x2040].copy_from_slice(&elf[0x2000..0x2040]);
        m
    }

    struct Fake {
        objects: Vec<LoadedObject>,
        info: Result<PointerInfo, String>,
        calls: Cell<usize>,
    }
    impl LoaderApi for Fake {
        fn loaded_objects(&self) -> Result<Vec<LoadedObject>, String> {
            Ok(self.objects.clone())
        }
        fn pointer_info(&self, _va: u64) -> Result<PointerInfo, String> {
            self.calls.set(self.calls.get() + 1);
            self.info.clone()
        }
    }

    const FAKE_AGENT_BASE: u64 = 0x7000_0000_0000; // never dereferenced; alias is host_base

    /// Fake whose alias resolves into the owned buffer at `host` (agent base
    /// differs from host base, like a real aliasing mapping).
    fn fake_for(host: usize) -> Fake {
        Fake {
            objects: vec![LoadedObject {
                load_base: FAKE_AGENT_BASE,
                load_size: IMG_SPAN as u64,
                load_delta: FAKE_AGENT_BASE as i64,
            }],
            info: Ok(PointerInfo {
                known: true,
                agent_base: FAKE_AGENT_BASE,
                host_base: host as u64,
                size: IMG_SPAN as u64,
                host_access: None,
            }),
            calls: Cell::new(0),
        }
    }

    fn capture_string(reg: &Mutex<Registry>) -> (CaptureOutcome, String) {
        let mut s = String::new();
        let o = capture_into(reg, &mut |t| s.push_str(t));
        (o, s)
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn non_linux_hsa_registration_is_unavailable_and_unregisters() {
        let elf = build_elf();
        let reg = Mutex::new(Registry::new());
        {
            let mut registry = lock_blocking(&reg);
            registry.register_hsa(7, "unsupported", &elf);
            assert!(
                registry.hsa.is_none(),
                "no unsupported loader may be cached"
            );
            assert_eq!(
                registry.modules.len(),
                1,
                "registration must not be dropped"
            );
            let module = &registry.modules[0];
            assert_eq!(module.module, 7);
            assert!(module.objects.is_empty(), "no loaded ranges are attributed");
            assert_eq!(module.unavailable.len(), 1);
            assert!(module.unavailable[0].contains("loaded-code HSA adapter is unsupported"));
            assert!(module.unavailable[0].contains(std::env::consts::OS));
            assert!(module.unavailable[0].contains("Linux only"));
        }
        let (outcome, report) = self_test_string(&reg);
        assert_eq!(
            outcome,
            CaptureOutcome::Captured {
                ranges: 0,
                equal: 0,
                differ: 0,
                unavailable: 1,
            }
        );
        assert!(report.contains("unavailable (not checked): module 'unsupported'"));
        assert!(!report.contains("intact"));
        {
            let mut registry = lock_blocking(&reg);
            // unregister_module uses this removal path before hipModuleUnload.
            registry.remove(7);
            assert!(registry.modules.is_empty());
            assert!(registry.claimed_bases().is_empty());
        }
        let (outcome, _) = self_test_string(&reg);
        assert_eq!(
            outcome,
            CaptureOutcome::Captured {
                ranges: 0,
                equal: 0,
                differ: 0,
                unavailable: 0,
            }
        );
        lock_blocking(&reg).register_hsa(7, "reloaded", &elf);
        assert_eq!(lock_blocking(&reg).modules.len(), 1);
        assert_eq!(lock_blocking(&reg).modules[0].unavailable.len(), 1);
        lock_blocking(&reg).remove(7);
        assert!(lock_blocking(&reg).modules.is_empty());
    }

    #[test]
    fn elf_ranges_exclude_writable_bss_and_headers() {
        let plan = parse_elf_ranges(&build_elf()).unwrap();
        assert_eq!(
            plan,
            vec![
                RangePlan {
                    vaddr: RO_VA,
                    file_off: 0x100,
                    len: 0x40,
                    exec: false
                },
                RangePlan {
                    vaddr: TEXT_VA,
                    file_off: 0x1000,
                    len: TEXT_LEN as u64,
                    exec: true
                },
            ]
        );
    }

    #[test]
    fn equal_snapshot() {
        let elf = build_elf();
        let mut mem = load_image(&elf);
        let api = fake_for(mem.as_mut_ptr() as usize);
        let reg = Mutex::new(Registry::new());
        lock_blocking(&reg).register_with(&api, 1, "m", &elf);
        assert_eq!(lock_blocking(&reg).modules[0].objects[0].ranges.len(), 2);
        let (o, s) = capture_string(&reg);
        assert_eq!(
            o,
            CaptureOutcome::Captured {
                ranges: 2,
                equal: 2,
                differ: 0,
                unavailable: 0
            }
        );
        assert!(
            s.contains(
                "hipfire: code-check after HIP 700: 2 ranges, 2 equal, 0 differ, 0 unavailable"
            ),
            "{s}"
        );
    }

    fn mutate_and_check(va: u64, off_in_range: usize, exec: bool) {
        let elf = build_elf();
        let mut mem = load_image(&elf);
        let p = mem.as_mut_ptr();
        let api = fake_for(p as usize);
        let reg = Mutex::new(Registry::new());
        lock_blocking(&reg).register_with(&api, 1, "m", &elf);
        // SAFETY: in-bounds write through the pointer the alias was derived from.
        unsafe { *p.add(va as usize + off_in_range) ^= 0x01 };
        let (o, s) = capture_string(&reg);
        assert_eq!(
            o,
            CaptureOutcome::Captured {
                ranges: 2,
                equal: 1,
                differ: 1,
                unavailable: 0
            }
        );
        assert!(
            s.contains(&format!("first diff at offset {off_in_range} ")),
            "{s}"
        );
        assert!(
            s.contains(if exec {
                "exec range"
            } else {
                "read-only range"
            }),
            "{s}"
        );
        let want_va = FAKE_AGENT_BASE + va + off_in_range as u64;
        assert!(s.contains(&format!("(VA {want_va:#x})")), "{s}");
        assert!(
            s.contains("baseline sha256") && s.contains("current sha256"),
            "{s}"
        );
        assert_eq!(s.matches('\n').count(), 1, "single line: {s}");
        drop(mem);
    }

    #[test]
    fn text_one_byte_mutation() {
        // Offset is relative to the .text range start (range VA == TEXT_VA).
        mutate_and_check(TEXT_VA, 5, true);
    }

    #[test]
    fn kernel_descriptor_one_byte_mutation() {
        mutate_and_check(RO_VA, 0x10, false);
    }

    #[test]
    fn bounded_context_in_report() {
        let elf = build_elf();
        let mut mem = load_image(&elf);
        let p = mem.as_mut_ptr();
        let api = fake_for(p as usize);
        let reg = Mutex::new(Registry::new());
        lock_blocking(&reg).register_with(&api, 1, "m", &elf);
        // SAFETY: in-bounds write through the alias source pointer.
        unsafe { *p.add(TEXT_VA as usize + 100) ^= 0xff };
        let (_, s) = capture_string(&reg);
        assert_eq!(s.matches('\n').count(), 1, "single line: {s}");
        assert!(s.ends_with('\n'), "{s}");
        // Window is <=16 bytes (<=32 hex chars) per side and contains the diff byte.
        assert!(s.contains("baseline[92..108] "), "{s}");
        let w = s.split("baseline[92..108] ").nth(1).unwrap();
        let hex_len = w
            .split(|c: char| !c.is_ascii_hexdigit())
            .next()
            .unwrap()
            .len();
        assert_eq!(hex_len, 32, "{s}");
        drop(mem);
    }

    #[test]
    fn unavailable_aliases_do_not_read() {
        let elf = build_elf();
        // Poisoned host base that would fault if dereferenced.
        let poison = 0xdead_0000_0000u64;
        let obj = LoadedObject {
            load_base: FAKE_AGENT_BASE,
            load_size: IMG_SPAN as u64,
            load_delta: FAKE_AGENT_BASE as i64,
        };
        let sz = IMG_SPAN as u64;
        let infos = [
            PointerInfo {
                known: true,
                agent_base: FAKE_AGENT_BASE,
                host_base: 0,
                size: sz,
                host_access: None,
            },
            PointerInfo {
                known: true,
                agent_base: 0,
                host_base: poison,
                size: sz,
                host_access: None,
            },
            // Allocation too small: range outside containment.
            PointerInfo {
                known: true,
                agent_base: FAKE_AGENT_BASE,
                host_base: poison,
                size: 0x200,
                host_access: None,
            },
            // Allocation starts above the range VA.
            PointerInfo {
                known: true,
                agent_base: FAKE_AGENT_BASE + 0x8000,
                host_base: poison,
                size: sz,
                host_access: None,
            },
            PointerInfo {
                known: false,
                agent_base: FAKE_AGENT_BASE,
                host_base: poison,
                size: sz,
                host_access: None,
            },
            PointerInfo {
                known: true,
                agent_base: FAKE_AGENT_BASE,
                host_base: poison,
                size: sz,
                host_access: Some(false),
            },
        ];
        for info in infos {
            let api = Fake {
                objects: vec![obj],
                info: Ok(info),
                calls: Cell::new(0),
            };
            let reg = Mutex::new(Registry::new());
            lock_blocking(&reg).register_with(&api, 1, "m", &elf);
            assert!(lock_blocking(&reg).modules[0].objects.is_empty());
            let (o, s) = capture_string(&reg);
            assert_eq!(
                o,
                CaptureOutcome::Captured {
                    ranges: 0,
                    equal: 0,
                    differ: 0,
                    unavailable: 1
                },
                "{info:?}"
            );
            assert!(s.contains("1 unavailable") && !s.contains("intact"), "{s}");
        }
        // pointer_info error is also unavailable.
        let api = Fake {
            objects: vec![obj],
            info: Err("status".into()),
            calls: Cell::new(0),
        };
        let mut reg = Registry::new();
        reg.register_with(&api, 1, "m", &elf);
        assert_eq!(reg.modules[0].unavailable.len(), 1);
    }

    #[test]
    fn unparsable_image_is_counted_unavailable() {
        let api = Fake {
            objects: vec![],
            info: Err("x".into()),
            calls: Cell::new(0),
        };
        let reg = Mutex::new(Registry::new());
        lock_blocking(&reg).register_with(&api, 9, "bad", b"not an elf");
        let (o, _) = capture_string(&reg);
        assert_eq!(
            o,
            CaptureOutcome::Captured {
                ranges: 0,
                equal: 0,
                differ: 0,
                unavailable: 1
            }
        );
    }

    #[test]
    fn text_mismatch_is_not_registered() {
        let elf = build_elf();
        let mut mem = load_image(&elf);
        mem[TEXT_VA as usize + 7] ^= 0x40;
        let api = fake_for(mem.as_mut_ptr() as usize);
        let mut reg = Registry::new();
        reg.register_with(&api, 1, "m", &elf);
        assert!(reg.modules[0].objects.is_empty());
        assert!(reg.modules[0].unavailable[0].contains("1 text mismatches"));
    }

    #[test]
    fn unload_removes_record_and_claims() {
        let elf = build_elf();
        let mut mem = load_image(&elf);
        let api = fake_for(mem.as_mut_ptr() as usize);
        let reg = Mutex::new(Registry::new());
        lock_blocking(&reg).register_with(&api, 1, "m", &elf);
        // Same object cannot be attributed to a second module while claimed.
        lock_blocking(&reg).register_with(&api, 2, "m2", &elf);
        assert!(lock_blocking(&reg).modules[1].objects.is_empty());
        assert_eq!(lock_blocking(&reg).modules[1].unavailable.len(), 1);
        lock_blocking(&reg).remove(1);
        lock_blocking(&reg).remove(2);
        assert!(lock_blocking(&reg).modules.is_empty());
        let (o, _) = capture_string(&reg);
        assert_eq!(
            o,
            CaptureOutcome::Captured {
                ranges: 0,
                equal: 0,
                differ: 0,
                unavailable: 0
            }
        );
    }

    #[test]
    fn second_capture_is_once() {
        let elf = build_elf();
        let mut mem = load_image(&elf);
        let api = fake_for(mem.as_mut_ptr() as usize);
        let reg = Mutex::new(Registry::new());
        lock_blocking(&reg).register_with(&api, 1, "m", &elf);
        let mut emitted = 0;
        let a = capture_into(&reg, &mut |_| emitted += 1);
        let b = capture_into(&reg, &mut |_| emitted += 1);
        assert!(matches!(a, CaptureOutcome::Captured { .. }));
        assert_eq!(b, CaptureOutcome::AlreadyCaptured);
        assert_eq!(emitted, 1);
    }

    #[test]
    fn busy_registry_skips_without_consuming_once() {
        let elf = build_elf();
        let mut mem = load_image(&elf);
        let api = fake_for(mem.as_mut_ptr() as usize);
        let reg = Mutex::new(Registry::new());
        lock_blocking(&reg).register_with(&api, 1, "m", &elf);
        let guard = lock_blocking(&reg);
        let mut s = String::new();
        let o = capture_into(&reg, &mut |t| s.push_str(t));
        assert_eq!(o, CaptureOutcome::Busy);
        assert!(
            s.contains("skipped: registry busy") && !s.contains("equal"),
            "{s}"
        );
        assert_eq!(s.matches('\n').count(), 1, "{s}");
        drop(guard);
        let (o, _) = capture_string(&reg);
        assert!(matches!(
            o,
            CaptureOutcome::Captured {
                ranges: 2,
                equal: 2,
                ..
            }
        ));
    }

    #[test]
    fn poisoned_registry_is_explicit_skip() {
        let reg = Mutex::new(Registry::new());
        let _ = std::thread::scope(|sc| {
            sc.spawn(|| {
                let _g = reg.lock().unwrap_or_else(|p| p.into_inner());
                panic!("poison");
            })
            .join()
        });
        let (o, s) = capture_string(&reg);
        assert_eq!(o, CaptureOutcome::Poisoned);
        assert!(
            s.contains("skipped: registry poisoned") && !s.contains("equal"),
            "{s}"
        );
    }

    fn self_test_string(reg: &Mutex<Registry>) -> (CaptureOutcome, String) {
        let mut s = String::new();
        let o = capture_mode_into(reg, CaptureMode::SelfTest, &mut |t| s.push_str(t));
        (o, s)
    }

    const SELF_TEST_LABEL: &str = "hipfire: code-check self-test:";

    #[test]
    fn self_test_equal_label() {
        let elf = build_elf();
        let mut mem = load_image(&elf);
        let api = fake_for(mem.as_mut_ptr() as usize);
        let reg = Mutex::new(Registry::new());
        lock_blocking(&reg).register_with(&api, 1, "m", &elf);
        let (o, s) = self_test_string(&reg);
        assert_eq!(
            o,
            CaptureOutcome::Captured {
                ranges: 2,
                equal: 2,
                differ: 0,
                unavailable: 0
            }
        );
        assert!(
            s.starts_with(
                "hipfire: code-check self-test: 2 ranges, 2 equal, 0 differ, 0 unavailable"
            ),
            "{s}"
        );
        assert!(!s.contains("after HIP 700"), "{s}");
        assert_eq!(s.matches('\n').count(), 1, "{s}");
    }

    #[test]
    fn self_test_mismatch_label() {
        let elf = build_elf();
        let mut mem = load_image(&elf);
        let p = mem.as_mut_ptr();
        let api = fake_for(p as usize);
        let reg = Mutex::new(Registry::new());
        lock_blocking(&reg).register_with(&api, 1, "m", &elf);
        // SAFETY: in-bounds write through the pointer the alias was derived from.
        unsafe { *p.add(TEXT_VA as usize + 5) ^= 0x01 };
        let (o, s) = self_test_string(&reg);
        assert_eq!(
            o,
            CaptureOutcome::Captured {
                ranges: 2,
                equal: 1,
                differ: 1,
                unavailable: 0
            }
        );
        assert!(
            s.starts_with(
                "hipfire: code-check self-test: 2 ranges, 1 equal, 1 differ, 0 unavailable"
            ),
            "{s}"
        );
        assert!(s.contains("first diff at offset 5 "), "{s}");
        assert!(!s.contains("after HIP 700"), "{s}");
        drop(mem);
    }

    #[test]
    fn self_test_unavailable_label() {
        let api = Fake {
            objects: vec![],
            info: Err("x".into()),
            calls: Cell::new(0),
        };
        let reg = Mutex::new(Registry::new());
        lock_blocking(&reg).register_with(&api, 9, "bad", b"not an elf");
        let (o, s) = self_test_string(&reg);
        assert_eq!(
            o,
            CaptureOutcome::Captured {
                ranges: 0,
                equal: 0,
                differ: 0,
                unavailable: 1
            }
        );
        assert!(
            s.starts_with(
                "hipfire: code-check self-test: 0 ranges, 0 equal, 0 differ, 1 unavailable"
            ),
            "{s}"
        );
        assert!(s.contains("unavailable (not checked): module 'bad'"), "{s}");
    }

    #[test]
    fn self_test_skipped_labels() {
        let reg = Mutex::new(Registry::new());
        let guard = lock_blocking(&reg);
        let (o, s) = self_test_string(&reg);
        assert_eq!(o, CaptureOutcome::Busy);
        assert!(
            s.starts_with(&format!("{SELF_TEST_LABEL} skipped: registry busy"))
                && !s.contains("equal"),
            "{s}"
        );
        assert_eq!(s.matches('\n').count(), 1, "{s}");
        drop(guard);

        let reg = Mutex::new(Registry::new());
        let _ = std::thread::scope(|sc| {
            sc.spawn(|| {
                let _g = reg.lock().unwrap_or_else(|p| p.into_inner());
                panic!("poison");
            })
            .join()
        });
        let (o, s) = self_test_string(&reg);
        assert_eq!(o, CaptureOutcome::Poisoned);
        assert!(
            s.starts_with(&format!("{SELF_TEST_LABEL} skipped: registry poisoned"))
                && !s.contains("equal"),
            "{s}"
        );
        // A skipped self-test never touches the once-only slot.
        assert!(!reg.lock().unwrap_or_else(|p| p.into_inner()).captured);
    }

    #[test]
    fn self_test_then_700_captures_exactly_once() {
        let elf = build_elf();
        let mut mem = load_image(&elf);
        let api = fake_for(mem.as_mut_ptr() as usize);
        let reg = Mutex::new(Registry::new());
        lock_blocking(&reg).register_with(&api, 1, "m", &elf);
        // Repeated self-tests never consume the slot.
        for _ in 0..2 {
            let (o, _) = self_test_string(&reg);
            assert!(matches!(
                o,
                CaptureOutcome::Captured {
                    ranges: 2,
                    equal: 2,
                    ..
                }
            ));
            assert!(!lock_blocking(&reg).captured);
        }
        let (a, sa) = capture_string(&reg);
        assert!(matches!(
            a,
            CaptureOutcome::Captured {
                ranges: 2,
                equal: 2,
                ..
            }
        ));
        assert!(
            sa.starts_with("hipfire: code-check after HIP 700: 2 ranges"),
            "{sa}"
        );
        let (b, sb) = capture_string(&reg);
        assert_eq!(b, CaptureOutcome::AlreadyCaptured);
        assert!(sb.is_empty(), "{sb}");
    }

    #[test]
    fn self_test_after_700_consumed_still_reports_and_keeps_slot() {
        let elf = build_elf();
        let mut mem = load_image(&elf);
        let api = fake_for(mem.as_mut_ptr() as usize);
        let reg = Mutex::new(Registry::new());
        lock_blocking(&reg).register_with(&api, 1, "m", &elf);
        let (a, _) = capture_string(&reg);
        assert!(matches!(a, CaptureOutcome::Captured { .. }));
        let (o, s) = self_test_string(&reg);
        assert!(matches!(
            o,
            CaptureOutcome::Captured {
                ranges: 2,
                equal: 2,
                ..
            }
        ));
        assert!(s.starts_with(SELF_TEST_LABEL), "{s}");
        assert!(lock_blocking(&reg).captured);
        let (b, sb) = capture_string(&reg);
        assert_eq!(b, CaptureOutcome::AlreadyCaptured);
        assert!(sb.is_empty(), "{sb}");
    }

    #[test]
    fn checked_delta_vaddr_len_overflow() {
        assert_eq!(loaded_va(0x1000, 0x20), Some(0x1020));
        assert_eq!(loaded_va(-0x10, 0x20), Some(0x10));
        assert_eq!(loaded_va(-0x30, 0x20), None);
        assert_eq!(loaded_va(i64::MAX, u64::MAX), None);
        let obj = LoadedObject {
            load_base: 1,
            load_size: 100,
            load_delta: 0,
        };
        let cases: [(i64, u64, u64); 3] = [
            (0, u64::MAX - 8, 16),             // vaddr + len wraps
            (i64::MIN, 0x100, 0x10),           // negative delta underflows
            (i64::MAX, u64::MAX - 0x10, 0x10), // delta + vaddr wraps
        ];
        for (delta, vaddr, len) in cases {
            let api = Fake {
                objects: vec![],
                info: Err("must not be called".into()),
                calls: Cell::new(0),
            };
            let obj = LoadedObject {
                load_delta: delta,
                ..obj
            };
            let plan = [RangePlan {
                vaddr,
                file_off: 0,
                len,
                exec: true,
            }];
            let r = match_object(&api, &obj, &[], &plan);
            assert!(matches!(r, MatchResult::NotProvable(_)));
            assert_eq!(api.calls.get(), 0);
        }
        // LOAD_BASE + LOAD_SIZE overflow.
        let api = Fake {
            objects: vec![],
            info: Err("no".into()),
            calls: Cell::new(0),
        };
        let obj = LoadedObject {
            load_base: u64::MAX,
            load_size: 2,
            load_delta: 0,
        };
        let plan = [RangePlan {
            vaddr: 0x10,
            file_off: 0,
            len: 4,
            exec: true,
        }];
        assert!(matches!(
            match_object(&api, &obj, &[], &plan),
            MatchResult::NotProvable(_)
        ));
        assert_eq!(api.calls.get(), 0);
        // resolve_alias with len overflow.
        assert!(resolve_alias(&api, u64::MAX - 4, 16).is_err());
        assert_eq!(api.calls.get(), 0);
    }

    #[test]
    fn bundle_with_elf_entry_is_accepted() {
        let elf = build_elf();
        let id = b"hipv4-amdgcn-amd-amdhsa--gfx1201";
        let mut b = b"__CLANG_OFFLOAD_BUNDLE__".to_vec();
        b.extend_from_slice(&1u64.to_le_bytes());
        let off = (b.len() + 24 + id.len()) as u64;
        b.extend_from_slice(&off.to_le_bytes());
        b.extend_from_slice(&(elf.len() as u64).to_le_bytes());
        b.extend_from_slice(&(id.len() as u64).to_le_bytes());
        b.extend_from_slice(id);
        b.extend_from_slice(&elf);
        let mut mem = load_image(&elf);
        let api = fake_for(mem.as_mut_ptr() as usize);
        let mut reg = Registry::new();
        reg.register_with(&api, 1, "m", &b);
        assert_eq!(reg.modules[0].objects[0].ranges.len(), 2);
        let mut reg = Registry::new();
        reg.register_with(&api, 1, "m", b"CCOB....");
        assert!(reg.modules[0].objects.is_empty());
        assert_eq!(reg.modules[0].unavailable.len(), 1);
    }
}
