//! `hipfire-xdna`: raw `amdxdna` ioctl control plane for the gfx1151 NPU spillover.
//!
//! Opt-in, exact-gfx1151-only, Linux-only. Talks to `/dev/accel/accel0` with
//! `libc` ioctls only (no libdrm link, no XRT, no Python at runtime).
//!
//! # Measured ABI provenance
//!
//! Byte-for-byte contract taken from `raw_npu.c` (bare-dispatch probe,
//! validated bit-for-bit vs the XRT path on kernel 7.0.0-31):
//!
//! * Session: 64 MiB `DEV_HEAP` BO (plus the shim-identical 512 MiB
//!   64 MiB-aligned CPU reservation) → `CREATE_HWCTX` (all-zero QoS,
//!   `max_opc` 2048, `num_tiles` 32) → `DEV` BO for the PDI +
//!   `CONFIG_HWCTX` CU (16 B payload) → `DEV` BO for the instructions → `SHARE`
//!   BOs for data + one `CMD` BO for the packet.
//! * ERT packet (68 bytes): header `0x30010001` (state NEW, count 16,
//!   opcode `START_CU`, type `CU`), `cu_mask` 1, payload opcode `u64` 3
//!   (mlir-aie txn path), instruction device address `u64`, instruction size
//!   `u32` in **bytes** (not word count), then `bo0..bo4` device addresses as
//!   `u64`s. `bo0` sits at payload-relative offset `0x14` (packet-absolute
//!   `0x1C`), i.e. 4-mod-8 unaligned: the builder writes words, never an
//!   aligned `u64` store, preserving the byte ABI.
//! * `EXEC_CMD` carries the command BO handle plus the argument BO handles
//!   (`[insts, args...]`); the driver pins them for the job. Completion via
//!   `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` on the `CREATE_HWCTX`-returned timeline
//!   syncobj (point = `EXEC_CMD` seq); kernel 7.0 has **no** `WAIT_CMD` ioctl.
//!   The packet word is then checked for `COMPLETED`.
//! * Coherence is carried by CPU cache-line flush + `MFENCE`; `SYNC_BO
//!   FROM_DEVICE` returns `EINVAL` on this driver and its failure is ignored
//!   on readback. `SYNC_BO` must **never** be issued on an imported dma-buf
//!   handle (kernel oops); imported BOs use flush only.
//! * Vendored UAPI structs mirror `/usr/include/drm/amdxdna_accel.h`, which
//!   the probe diffed identical against the kernel 7.0.0-31 uapi (modulo the
//!   header-guard name), plus generic `<drm/drm.h>` GEM/prime/syncobj ioctls.
//!
//! # Usage contract
//!
//! * Single-threaded, strictly sequential submit→wait. The device holds one
//!   preallocated CMD BO, so at most one submission may be in flight;
//!   submitting while another is pending fails with `EBUSY` instead of
//!   silently corrupting the shared packet. [`XdnaDevice::quiesce`] reports
//!   whether the device is idle.
//! * A `wait` timeout is **not** permission to reuse BOs: the device may still
//!   write them. Recover by dropping the device.
//! * [`Bo`] memory is shared with the device. `as_slice`/`as_mut_slice`
//!   require the caller to hold no other live slice of the same BO
//!   (submissions pin BOs but never create slices through the pins).

use std::os::unix::io::RawFd;
use std::path::Path;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Failure from any `hipfire-xdna` operation: which stage failed plus the
/// preserved `errno` value (`0` only where no OS error exists).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XdnaError {
    /// Static stage tag, e.g. `"open"`, `"create-hwctx"`, `"exec-cmd"`, `"wait"`.
    pub stage: &'static str,
    /// Preserved `errno` (`libc::E*`); `0` where no OS error applies.
    pub errno: i32,
}

impl XdnaError {
    fn os(stage: &'static str) -> Self {
        Self {
            stage,
            errno: std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(libc::EIO),
        }
    }

    fn code(stage: &'static str, errno: i32) -> Self {
        Self { stage, errno }
    }
}

impl std::fmt::Display for XdnaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "xdna {} failed: errno {}", self.stage, self.errno)
    }
}

impl std::error::Error for XdnaError {}

/// Crate result type.
pub type Result<T, E = XdnaError> = std::result::Result<T, E>;

// ---------------------------------------------------------------------------
// Vendored UAPI (installed `/usr/include/drm/amdxdna_accel.h` contract)
// ---------------------------------------------------------------------------

pub(crate) const INVALID_ADDR: u64 = u64::MAX;
pub(crate) const INVALID_BO_HANDLE: u32 = 0;

pub(crate) const BO_SHMEM: u32 = 1;
pub(crate) const BO_DEV_HEAP: u32 = 2;
pub(crate) const BO_DEV: u32 = 3;
pub(crate) const BO_CMD: u32 = 4;

pub(crate) const SYNC_TO_DEVICE: u32 = 0;
pub(crate) const SYNC_FROM_DEVICE: u32 = 1;

pub(crate) const HWCTX_CONFIG_CU: u32 = 0;
pub(crate) const CMD_SUBMIT_EXEC_BUF: u32 = 0;

pub(crate) const ERT_STATE_NEW: u32 = 1;
pub(crate) const ERT_STATE_COMPLETED: u32 = 4;
/// ERT header: state NEW | count<<12 | type CU(3)<<28. Opcode field [27:23] is
/// 0 (`START_CU`); the "opcode 3" of the probe notes is the payload word.
pub(crate) const ERT_HDR_COUNT16: u32 = ERT_STATE_NEW | (16 << 12) | (3 << 28);
pub(crate) const CU_MASK_CU0: u32 = 1;
/// Payload opcode word (mlir-aie txn path), distinct from the ERT header opcode.
pub(crate) const TXN_OPCODE: u64 = 3;

pub(crate) const SYNCOBJ_WAIT_FOR_SUBMIT: u32 = 1 << 1;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct QosInfo {
    pub gops: u32,
    pub fps: u32,
    pub dma_bandwidth: u32,
    pub latency: u32,
    pub frame_exec_time: u32,
    pub priority: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CreateHwctx {
    pub ext: u64,
    pub ext_flags: u64,
    pub qos_p: u64,
    pub umq_bo: u32,
    pub log_buf_bo: u32,
    pub max_opc: u32,
    pub num_tiles: u32,
    pub mem_size: u32,
    pub umq_doorbell: u32,
    pub handle: u32,
    pub syncobj_handle: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DestroyHwctx {
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CuConfig {
    pub cu_bo: u32,
    pub cu_func: u8,
    pub pad: [u8; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct HwctxParamConfigCu {
    pub num_cus: u16,
    pub pad: [u16; 3],
    pub cu: CuConfig,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ConfigHwctx {
    pub handle: u32,
    pub param_type: u32,
    pub param_val: u64,
    pub param_val_size: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CreateBo {
    pub flags: u64,
    pub vaddr: u64,
    pub size: u64,
    pub bo_type: u32,
    pub handle: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GetBoInfo {
    pub ext: u64,
    pub ext_flags: u64,
    pub handle: u32,
    pub pad: u32,
    pub map_offset: u64,
    pub vaddr: u64,
    pub xdna_addr: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SyncBo {
    pub handle: u32,
    pub direction: u32,
    pub offset: u64,
    pub size: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ExecCmd {
    pub ext: u64,
    pub ext_flags: u64,
    pub hwctx: u32,
    pub cmd_type: u32,
    pub cmd_handles: u64,
    pub args: u64,
    pub cmd_count: u32,
    pub arg_count: u32,
    pub seq: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PrimeHandle {
    pub handle: u32,
    pub flags: u32,
    pub fd: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GemClose {
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SyncobjDestroy {
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SyncobjTimelineWait {
    pub handles: u64,
    pub points: u64,
    pub timeout_nsec: i64,
    pub count_handles: u32,
    pub flags: u32,
    pub first_signaled: u32,
    pub pad: u32,
    pub deadline_nsec: u64,
}

// `static_assert`-style size checks against the C header (see unit tests for
// the full offset table and the C-probe ioctl literals).
const _: () = assert!(size_of::<QosInfo>() == 24);
const _: () = assert!(size_of::<CreateHwctx>() == 56);
const _: () = assert!(size_of::<DestroyHwctx>() == 8);
const _: () = assert!(size_of::<HwctxParamConfigCu>() == 16);
const _: () = assert!(size_of::<ConfigHwctx>() == 24);
const _: () = assert!(size_of::<CreateBo>() == 32);
const _: () = assert!(size_of::<GetBoInfo>() == 48);
const _: () = assert!(size_of::<SyncBo>() == 24);
const _: () = assert!(size_of::<ExecCmd>() == 56);
const _: () = assert!(size_of::<PrimeHandle>() == 12);
const _: () = assert!(size_of::<GemClose>() == 8);
const _: () = assert!(size_of::<SyncobjDestroy>() == 8);
const _: () = assert!(size_of::<SyncobjTimelineWait>() == 48);
const _: () = assert!(ERT_HDR_COUNT16 == 0x3001_0001);

// ---------------------------------------------------------------------------
// ioctl request codes (DRM `_IOWR`/`_IOW` math over base 'd')
// ---------------------------------------------------------------------------

const DRM_IOCTL_BASE: u32 = b'd' as u32;
const DRM_COMMAND_BASE: u32 = 0x40;

const fn drm_iowr(nr: u32, size: usize) -> u32 {
    (2 | 1) << 30 | ((size as u32) << 16) | (DRM_IOCTL_BASE << 8) | nr
}

const fn drm_iow(nr: u32, size: usize) -> u32 {
    1 << 30 | ((size as u32) << 16) | (DRM_IOCTL_BASE << 8) | nr
}

pub(crate) const DRM_IOCTL_AMDXDNA_CREATE_HWCTX: u32 =
    drm_iowr(DRM_COMMAND_BASE + 0, size_of::<CreateHwctx>());
pub(crate) const DRM_IOCTL_AMDXDNA_DESTROY_HWCTX: u32 =
    drm_iowr(DRM_COMMAND_BASE + 1, size_of::<DestroyHwctx>());
pub(crate) const DRM_IOCTL_AMDXDNA_CONFIG_HWCTX: u32 =
    drm_iowr(DRM_COMMAND_BASE + 2, size_of::<ConfigHwctx>());
pub(crate) const DRM_IOCTL_AMDXDNA_CREATE_BO: u32 =
    drm_iowr(DRM_COMMAND_BASE + 3, size_of::<CreateBo>());
pub(crate) const DRM_IOCTL_AMDXDNA_GET_BO_INFO: u32 =
    drm_iowr(DRM_COMMAND_BASE + 4, size_of::<GetBoInfo>());
pub(crate) const DRM_IOCTL_AMDXDNA_SYNC_BO: u32 =
    drm_iowr(DRM_COMMAND_BASE + 5, size_of::<SyncBo>());
pub(crate) const DRM_IOCTL_AMDXDNA_EXEC_CMD: u32 =
    drm_iowr(DRM_COMMAND_BASE + 6, size_of::<ExecCmd>());
pub(crate) const DRM_IOCTL_GEM_CLOSE: u32 = drm_iow(0x09, size_of::<GemClose>());
pub(crate) const DRM_IOCTL_PRIME_FD_TO_HANDLE: u32 =
    drm_iowr(0x2e, size_of::<PrimeHandle>());
pub(crate) const DRM_IOCTL_SYNCOBJ_DESTROY: u32 =
    drm_iowr(0xc0, size_of::<SyncobjDestroy>());
pub(crate) const DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT: u32 =
    drm_iowr(0xca, size_of::<SyncobjTimelineWait>());

// ---------------------------------------------------------------------------
// ERT packet builder (pure; unit-tested against the C golden bytes)
// ---------------------------------------------------------------------------

/// Packet geometry from the probe: 17 words / 68 bytes, count field 16.
pub(crate) const PACKET_WORDS: usize = 17;
pub(crate) const PACKET_BYTES: usize = PACKET_WORDS * 4;
/// Packet-absolute byte offset of `bo0` (payload-relative `0x14` + 8-byte
/// header/mask prefix). 4-mod-8 unaligned by construction.
    #[allow(dead_code)] // pinned by the golden layout test
    pub(crate) const BO0_PACKET_OFFSET: usize = 0x1c;
/// Maximum buffer arguments per submit (`bo0..bo4`; unused slots are zero).
pub(crate) const MAX_BO_ARGS: usize = 5;

/// Build the 17-word ERT packet. `boot_addrs` holds at most [`MAX_BO_ARGS`]
/// device addresses; shorter lists are zero-padded exactly like `raw_npu.c`
/// (`bo3`/`bo4` unused → 0).
pub(crate) fn build_packet_words(
    instr_addr: u64,
    ninstr_bytes: u32,
    boot_addrs: &[u64],
) -> Result<[u32; PACKET_WORDS]> {
    if boot_addrs.len() > MAX_BO_ARGS {
        return Err(XdnaError::code("submit-args", libc::EINVAL));
    }
    let mut w = [0u32; PACKET_WORDS];
    w[0] = ERT_HDR_COUNT16;
    w[1] = CU_MASK_CU0;
    w[2] = TXN_OPCODE as u32;
    w[3] = (TXN_OPCODE >> 32) as u32;
    w[4] = instr_addr as u32;
    w[5] = (instr_addr >> 32) as u32;
    w[6] = ninstr_bytes;
    for (i, &a) in boot_addrs.iter().enumerate() {
        w[7 + 2 * i] = a as u32;
        w[8 + 2 * i] = (a >> 32) as u32;
    }
    Ok(w)
}

// ---------------------------------------------------------------------------
// Public API (Linux implementation in `imp`, stub elsewhere)
// ---------------------------------------------------------------------------

/// Buffer-object kind, mirroring the `AMDXDNA_BO_*` types used by the probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoKind {
    /// CPU-mappable shared buffer (A/B/C data, ordinary path).
    Share,
    /// Device-heap buffer (PDI, instructions); CPU access is heap-relative.
    Dev,
    /// Command buffer holding the ERT packet.
    Cmd,
}

#[cfg(target_os = "linux")]
pub use imp::{Bo, BoArg, LoadedKernel, Submission, XdnaDevice};

#[cfg(not(target_os = "linux"))]
pub use imp_stub::{Bo, BoArg, LoadedKernel, Submission, XdnaDevice};

// ---------------------------------------------------------------------------
// Linux implementation
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod imp {
    use super::*;
    use parking_lot::Mutex;
    use std::ffi::CString;
    use std::sync::Arc;

    const HEAP_SIZE: usize = 64 << 20;
    const HEAP_MAX: usize = 512 << 20;
    const HEAP_ALIGN: usize = 64 << 20;
    const CMD_BO_SIZE: usize = 4096;
    const MAX_OPC: u32 = 2048;
    /// 8 columns × 4 core rows on npu5 (probe: `cols * core_rows`).
    const NUM_TILES: u32 = 32;

    fn ioctl(fd: RawFd, req: u32, arg: *mut std::ffi::c_void) -> Result<()> {
        // SAFETY: raw ioctl with a caller-provided, correctly-typed argument.
        let rc = unsafe { libc::ioctl(fd, req as libc::c_ulong, arg) };
        if rc < 0 {
            return Err(XdnaError::os(match req {
                super::DRM_IOCTL_AMDXDNA_CREATE_HWCTX => "create-hwctx",
                super::DRM_IOCTL_AMDXDNA_DESTROY_HWCTX => "destroy-hwctx",
                super::DRM_IOCTL_AMDXDNA_CONFIG_HWCTX => "config-hwctx",
                super::DRM_IOCTL_AMDXDNA_CREATE_BO => "create-bo",
                super::DRM_IOCTL_AMDXDNA_GET_BO_INFO => "get-bo-info",
                super::DRM_IOCTL_AMDXDNA_SYNC_BO => "sync-bo",
                super::DRM_IOCTL_AMDXDNA_EXEC_CMD => "exec-cmd",
                super::DRM_IOCTL_GEM_CLOSE => "gem-close",
                super::DRM_IOCTL_PRIME_FD_TO_HANDLE => "prime-import",
                super::DRM_IOCTL_SYNCOBJ_DESTROY => "syncobj-destroy",
                super::DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT => "wait",
                _ => "ioctl",
            }));
        }
        Ok(())
    }

    /// Shim-identical CPU flush for the non-coherent NPU path: every cache
    /// line covering `[ptr, ptr+len)` flushed, then `MFENCE`.
    pub(crate) fn cpu_flush(ptr: *const u8, len: usize) {
        if len == 0 {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: reads no memory; flushes lines covering a valid mapping.
            unsafe {
                let mut cl: libc::c_long =
                    libc::sysconf(libc::_SC_LEVEL1_DCACHE_LINESIZE);
                if cl <= 0 {
                    cl = 64;
                }
                let cl = cl as usize;
                let base = ptr as usize;
                let mut cur = base & !(cl - 1);
                let end = base + len;
                while cur < end {
                    std::arch::x86_64::_mm_clflush(cur as *const u8);
                    cur += cl;
                }
                std::arch::x86_64::_mm_mfence();
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (ptr, len);
        }
    }

    fn round_page(n: usize) -> usize {
        // SAFETY: sysconf with a valid name.
        let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let p = if p == 0 { 4096 } else { p };
        (n + p - 1) & !(p - 1)
    }

    struct CmdSlot {
        handle: u32,
        /// 4096-byte CMD BO mapping; only `PACKET_WORDS` are used per submit.
        ptr: *mut u32,
    }

    /// Shared device state. `Bo`s hold an `Arc` to this, so the fd and heap
    /// mapping outlive every buffer object.
    struct DeviceInner {
        fd: RawFd,
        hwctx: u32,
        syncobj: u32,
        heap_handle: u32,
        heap_xdna: u64,
        heap_cpu: *mut u8,
        heap_res_base: *mut u8,
        heap_res_len: usize,
        /// Preallocated CMD BO: no allocation in the submit hot path.
        cmd: Mutex<CmdSlot>,
        /// Sequence of the in-flight submission, if any (single-flight guard).
        inflight: Mutex<Option<u64>>,
    }

    // SAFETY: `heap_cpu`/`cmd` memory is only touched under the crate's
    // serialized submit→wait protocol (`cmd` additionally under its mutex);
    // the fd is only used for ioctls. No thread spawns device access itself.
    unsafe impl Send for DeviceInner {}
    unsafe impl Sync for DeviceInner {}

    impl Drop for DeviceInner {
        fn drop(&mut self) {
            let fd = self.fd;
            // Best-effort teardown in shim order; errors are unreportable here.
            let slot = self.cmd.lock();
            if slot.handle != super::INVALID_BO_HANDLE {
                let map_len = round_page(CMD_BO_SIZE);
                // SAFETY: mapping created by this device, still live.
                unsafe { libc::munmap(slot.ptr as *mut _, map_len) };
                let mut g = super::GemClose { handle: slot.handle, pad: 0 };
                let _ = ioctl(fd, super::DRM_IOCTL_GEM_CLOSE, &mut g as *mut _ as *mut _);
            }
            drop(slot);
            let mut d = super::SyncobjDestroy { handle: self.syncobj, pad: 0 };
            let _ = ioctl(fd, super::DRM_IOCTL_SYNCOBJ_DESTROY, &mut d as *mut _ as *mut _);
            let mut h = super::DestroyHwctx { handle: self.hwctx, pad: 0 };
            let _ = ioctl(fd, super::DRM_IOCTL_AMDXDNA_DESTROY_HWCTX, &mut h as *mut _ as *mut _);
            let mut g = super::GemClose { handle: self.heap_handle, pad: 0 };
            let _ = ioctl(fd, super::DRM_IOCTL_GEM_CLOSE, &mut g as *mut _ as *mut _);
            if !self.heap_res_base.is_null() {
                // SAFETY: reservation created by `open`, still live.
                unsafe { libc::munmap(self.heap_res_base as *mut _, self.heap_res_len) };
            }
            // SAFETY: fd owned by this device.
            unsafe { libc::close(fd) };
        }
    }

    struct BoState {
        dev: Arc<DeviceInner>,
        handle: u32,
        kind: BoKind,
        imported: bool,
        /// CPU pointer: mmap base for Share/Cmd/imported, heap-relative for Dev.
        ptr: *mut u8,
        len: usize,
        /// mmap length (0 for Dev BOs, which are not munmapped per-BO).
        map_len: usize,
        xdna_addr: u64,
    }

    /// A device buffer object. `Clone` shares ownership (`Arc`); the backing
    /// store is released when the last clone drops.
    #[derive(Clone)]
    pub struct Bo {
        state: Arc<BoState>,
    }

    // SAFETY: same reasoning as `DeviceInner`; slice creation is the caller's
    // serialized contract (see crate docs).
    unsafe impl Send for Bo {}
    unsafe impl Sync for Bo {}

    impl Drop for BoState {
        fn drop(&mut self) {
            if self.map_len > 0 && !self.ptr.is_null() {
                // SAFETY: mapping created by alloc/import, still live; the
                // device fd outlives every Bo via the Arc.
                unsafe { libc::munmap(self.ptr as *mut _, self.map_len) };
            }
            if self.handle != super::INVALID_BO_HANDLE {
                let mut g = super::GemClose { handle: self.handle, pad: 0 };
                let _ = ioctl(
                    self.dev.fd,
                    super::DRM_IOCTL_GEM_CLOSE,
                    &mut g as *mut _ as *mut _,
                );
            }
        }
    }

    impl Bo {
        /// Payload device address: `xdna_addr` when valid, else the post-mmap
        /// CPU VA (`buffer::paddr` semantics from the probe; `INVALID` is
        /// normal for ordinary SHARE BOs).
        pub fn vaddr(&self) -> u64 {
            if self.state.xdna_addr != super::INVALID_ADDR {
                return self.state.xdna_addr;
            }
            self.state.ptr as u64
        }

        /// BO handle (for diagnostics / `EXEC_CMD` arg lists).
        pub fn handle(&self) -> u32 {
            self.state.handle
        }

        /// Requested byte length.
        pub fn len(&self) -> usize {
            self.state.len
        }

        /// Whether this BO was PRIME-imported (never synchronized via SYNC_BO).
        pub fn is_imported(&self) -> bool {
            self.state.imported
        }

        /// BO kind.
        pub fn kind(&self) -> BoKind {
            self.state.kind
        }

        /// Read-only view of the BO memory. Caller must hold no live mutable
        /// slice of the same BO.
        pub fn as_slice(&self) -> &[u8] {
            // SAFETY: mapping live (Arc keeps device + mapping); aliasing is
            // the caller's documented contract.
            unsafe { std::slice::from_raw_parts(self.state.ptr, self.state.len) }
        }

        /// Mutable view of the BO memory. Caller must hold no other live
        /// slice of the same BO.
        pub fn as_mut_slice(&self) -> &mut [u8] {
            // SAFETY: as above; device writes only happen between submit and
            // wait, during which the caller must not touch the memory.
            unsafe { std::slice::from_raw_parts_mut(self.state.ptr, self.state.len) }
        }

        /// Publish CPU writes to the device: cache-line flush + `SYNC_BO
        /// TO_DEVICE`, except for imported BOs (flush only — never SYNC_BO
        /// an imported handle).
        pub fn publish(&self) -> Result<()> {
            cpu_flush(self.state.ptr, self.state.len);
            if self.state.imported {
                return Ok(());
            }
            let mut s = super::SyncBo {
                handle: self.state.handle,
                direction: super::SYNC_TO_DEVICE,
                offset: 0,
                size: self.state.len as u64,
            };
            ioctl(self.state.dev.fd, super::DRM_IOCTL_AMDXDNA_SYNC_BO, &mut s as *mut _ as *mut _)
                .map_err(|_| XdnaError::os("publish-sync"))
        }

        /// Invalidate CPU caches after device write (readback path): flush,
        /// best-effort `SYNC_BO FROM_DEVICE` (returns `EINVAL` on this
        /// driver; ignored), flush. Never issues SYNC_BO for imported BOs.
        pub fn invalidate(&self) {
            cpu_flush(self.state.ptr, self.state.len);
            if !self.state.imported {
                let mut s = super::SyncBo {
                    handle: self.state.handle,
                    direction: super::SYNC_FROM_DEVICE,
                    offset: 0,
                    size: self.state.len as u64,
                };
                let _ = ioctl(
                    self.state.dev.fd,
                    super::DRM_IOCTL_AMDXDNA_SYNC_BO,
                    &mut s as *mut _ as *mut _,
                );
            }
            cpu_flush(self.state.ptr, self.state.len);
        }
    }

    /// A kernel loaded on the device: PDI configured into the context plus
    /// the instruction buffer. Holds its BOs alive.
    pub struct LoadedKernel {
        /// Retained pin: the context references the PDI after CONFIG_HWCTX.
        #[allow(dead_code)]
        pdi: Bo,
        insts: Bo,
        insts_addr: u64,
        insts_bytes: usize,
    }

    impl LoadedKernel {
        /// Instruction-stream length in bytes (packet `ninstr` field).
        pub fn insts_len(&self) -> usize {
            self.insts_bytes
        }
    }

    /// One buffer argument (`bo0..bo4`) to [`XdnaDevice::submit`].
    #[derive(Clone, Copy)]
    pub struct BoArg<'a> {
        /// Buffer whose payload address is written into the packet.
        pub bo: &'a Bo,
    }

    impl<'a> From<&'a Bo> for BoArg<'a> {
        fn from(bo: &'a Bo) -> Self {
            Self { bo }
        }
    }

    /// Opened NPU device: session, hardware context, and the preallocated
    /// CMD BO shared by all submissions.
    #[derive(Clone)]
    pub struct XdnaDevice {
        inner: Arc<DeviceInner>,
    }

    impl XdnaDevice {
        /// Open `path` (e.g. `/dev/accel/accel0`), create the 64 MiB device
        /// heap, the hardware context (`max_opc` 2048, 32 tiles), and
        /// preallocate the CMD BO.
        pub fn open(path: &Path) -> Result<Self> {
            let cpath = CString::new(path.as_os_str().as_encoded_bytes())
                .map_err(|_| XdnaError::code("open", libc::EINVAL))?;
            // SAFETY: `open` with a valid C string.
            let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
            if fd < 0 {
                return Err(XdnaError::os("open"));
            }

            // --- device heap (must precede hwctx, per probe) ---
            let (heap_handle, heap_xdna, heap_map_off) = create_bo_get(fd, super::BO_DEV_HEAP, HEAP_SIZE, "heap")?;
            if heap_xdna == super::INVALID_ADDR {
                cleanup_handle(fd, heap_handle);
                // SAFETY: fd owned here.
                unsafe { libc::close(fd) };
                return Err(XdnaError::code("heap-addr", libc::ENXIO));
            }
            // Shim-identical reservation: 512 MiB range, 64 MiB aligned.
            let total = HEAP_MAX + HEAP_ALIGN;
            // SAFETY: anonymous reservation.
            let res = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    total,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            if res == libc::MAP_FAILED {
                let e = XdnaError::os("heap-reserve");
                cleanup_handle(fd, heap_handle);
                // SAFETY: fd owned here.
                unsafe { libc::close(fd) };
                return Err(e);
            }
            let base = res as usize;
            let aligned = (base + HEAP_ALIGN - 1) & !(HEAP_ALIGN - 1);
            let head = aligned - base;
            if head > 0 {
                // SAFETY: inside our reservation.
                unsafe { libc::munmap(res, head) };
            }
            let tail = total - head - HEAP_MAX;
            if tail > 0 {
                // SAFETY: inside our reservation.
                unsafe { libc::munmap((aligned + HEAP_MAX) as *mut _, tail) };
            }
            // SAFETY: MAP_FIXED inside our aligned reservation.
            let heap_cpu = unsafe {
                libc::mmap(
                    aligned as *mut _,
                    HEAP_SIZE,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    fd,
                    heap_map_off as libc::off_t,
                )
            };
            if heap_cpu == libc::MAP_FAILED {
                let e = XdnaError::os("heap-map");
                // SAFETY: reservation still ours.
                unsafe { libc::munmap(aligned as *mut _, HEAP_MAX) };
                cleanup_handle(fd, heap_handle);
                // SAFETY: fd owned here.
                unsafe { libc::close(fd) };
                return Err(e);
            }
            // mlock attempted; failure only degrades (probe behavior).
            // SAFETY: our mapping.
            unsafe {
                libc::mlock(heap_cpu, HEAP_SIZE);
            }

            // --- hardware context ---
            let qos = super::QosInfo::default(); // reference QoS: all zero
            let mut c = super::CreateHwctx {
                ext: 0,
                ext_flags: 0,
                qos_p: &qos as *const _ as u64,
                umq_bo: 0,
                log_buf_bo: 0,
                max_opc: MAX_OPC,
                num_tiles: NUM_TILES,
                ..Default::default()
            };
            if let Err(e) = ioctl(fd, super::DRM_IOCTL_AMDXDNA_CREATE_HWCTX, &mut c as *mut _ as *mut _)
                .map_err(|_| XdnaError::os("create-hwctx"))
            {
                // SAFETY: owned mappings/fd.
                unsafe { libc::munmap(aligned as *mut _, HEAP_MAX) };
                cleanup_handle(fd, heap_handle);
                // SAFETY: fd owned here.
                unsafe { libc::close(fd) };
                return Err(e);
            }
            let (hwctx, syncobj) = (c.handle, c.syncobj_handle);

            // --- preallocated CMD BO ---
            let (cmd_handle, _, cmd_map_off) = match create_bo_get(fd, super::BO_CMD, CMD_BO_SIZE, "cmd-bo") {
                Ok(v) => v,
                Err(e) => {
                    destroy_hwctx_syncobj(fd, hwctx, syncobj);
                    // SAFETY: owned mappings/fd.
                    unsafe { libc::munmap(aligned as *mut _, HEAP_MAX) };
                    cleanup_handle(fd, heap_handle);
                    // SAFETY: fd owned here.
                    unsafe { libc::close(fd) };
                    return Err(e);
                }
            };
            if cmd_map_off == super::INVALID_ADDR {
                let e = XdnaError::code("cmd-bo-map", libc::ENXIO);
                destroy_hwctx_syncobj(fd, hwctx, syncobj);
                // SAFETY: owned mappings/fd.
                unsafe { libc::munmap(aligned as *mut _, HEAP_MAX) };
                cleanup_handle(fd, heap_handle);
                cleanup_handle(fd, cmd_handle);
                // SAFETY: fd owned here.
                unsafe { libc::close(fd) };
                return Err(e);
            }
            let map_len = round_page(CMD_BO_SIZE);
            // SAFETY: driver-provided offset for this BO.
            let cmd_ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    map_len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    cmd_map_off as libc::off_t,
                )
            };
            if cmd_ptr == libc::MAP_FAILED {
                let e = XdnaError::os("cmd-bo-map");
                destroy_hwctx_syncobj(fd, hwctx, syncobj);
                // SAFETY: owned mappings/fd.
                unsafe { libc::munmap(aligned as *mut _, HEAP_MAX) };
                cleanup_handle(fd, heap_handle);
                cleanup_handle(fd, cmd_handle);
                // SAFETY: fd owned here.
                unsafe { libc::close(fd) };
                return Err(e);
            }
            // SAFETY: our mapping.
            unsafe {
                libc::mlock(cmd_ptr, map_len);
            }

            Ok(Self {
                inner: Arc::new(DeviceInner {
                    fd,
                    hwctx,
                    syncobj,
                    heap_handle,
                    heap_xdna,
                    heap_cpu: heap_cpu as *mut u8,
                    heap_res_base: aligned as *mut u8,
                    heap_res_len: HEAP_MAX,
                    cmd: Mutex::new(CmdSlot { handle: cmd_handle, ptr: cmd_ptr as *mut u32 }),
                    inflight: Mutex::new(None),
                }),
            })
        }

        fn create_bo_locked(
            &self,
            bo_type: u32,
            size: usize,
            imported: bool,
            handle: u32,
            xdna_addr: u64,
            map_offset: u64,
        ) -> Result<Bo> {
            let inner = self.inner.clone();
            let (ptr, map_len) = if bo_type == super::BO_DEV {
                if xdna_addr == super::INVALID_ADDR {
                    cleanup_handle(inner.fd, handle);
                    return Err(XdnaError::code("alloc-addr", libc::ENXIO));
                }
                let off = xdna_addr.wrapping_sub(inner.heap_xdna);
                if off > (HEAP_SIZE as u64) || size as u64 > (HEAP_SIZE as u64) - off {
                    cleanup_handle(inner.fd, handle);
                    return Err(XdnaError::code("alloc-range", libc::ERANGE));
                }
                // SAFETY: heap mapping live via Arc; bounds checked above.
                (unsafe { inner.heap_cpu.add(off as usize) }, 0)
            } else {
                if map_offset == super::INVALID_ADDR {
                    cleanup_handle(inner.fd, handle);
                    return Err(XdnaError::code("alloc-map", libc::ENXIO));
                }
                let map_len = round_page(size);
                // SAFETY: driver-provided offset for this BO.
                let p = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        map_len,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED,
                        inner.fd,
                        map_offset as libc::off_t,
                    )
                };
                if p == libc::MAP_FAILED {
                    let e = XdnaError::os("alloc-map");
                    cleanup_handle(inner.fd, handle);
                    return Err(e);
                }
                // SAFETY: our mapping.
                unsafe {
                    libc::mlock(p, map_len);
                }
                (p as *mut u8, map_len)
            };
            let kind = match bo_type {
                super::BO_DEV => BoKind::Dev,
                super::BO_CMD => BoKind::Cmd,
                _ => BoKind::Share,
            };
            Ok(Bo {
                state: Arc::new(BoState {
                    dev: inner,
                    handle,
                    kind,
                    imported,
                    ptr,
                    len: size,
                    map_len,
                    xdna_addr,
                }),
            })
        }

        /// Allocate a buffer object of `bytes` with the given kind.
        pub fn alloc(&self, bytes: usize, kind: BoKind) -> Result<Bo> {
            if bytes == 0 {
                return Err(XdnaError::code("alloc", libc::EINVAL));
            }
            let bo_type = match kind {
                BoKind::Share => super::BO_SHMEM,
                BoKind::Dev => super::BO_DEV,
                BoKind::Cmd => super::BO_CMD,
            };
            let (handle, xdna_addr, map_offset) =
                create_bo_get(self.inner.fd, bo_type, bytes, "alloc")?;
            self.create_bo_locked(bo_type, bytes, false, handle, xdna_addr, map_offset)
        }

        /// Import a dma-buf `fd` (e.g. GPU-owned memory exported for sharing)
        /// as usable NPU address space. The imported BO is never passed to
        /// `SYNC_BO` (flush-only coherence); the caller retains `fd`
        /// ownership.
        pub fn import_dmabuf(&self, fd: RawFd, bytes: usize) -> Result<Bo> {
            if bytes == 0 {
                return Err(XdnaError::code("import", libc::EINVAL));
            }
            let mut p = super::PrimeHandle { handle: 0, flags: 0, fd };
            ioctl(self.inner.fd, super::DRM_IOCTL_PRIME_FD_TO_HANDLE, &mut p as *mut _ as *mut _)
                .map_err(|_| XdnaError::os("prime-import"))?;
            if p.handle == super::INVALID_BO_HANDLE {
                return Err(XdnaError::code("prime-import", libc::ENXIO));
            }
            let mut g = super::GetBoInfo { handle: p.handle, ..Default::default() };
            if ioctl(self.inner.fd, super::DRM_IOCTL_AMDXDNA_GET_BO_INFO, &mut g as *mut _ as *mut _)
                .map_err(|_| XdnaError::os("import-bo-info"))
                .is_err()
            {
                cleanup_handle(self.inner.fd, p.handle);
                return Err(XdnaError::os("import-bo-info"));
            }
            // `xdna_addr == INVALID` is the expected outcome for an imported
            // handle; the post-mmap VA is the usable NPU address.
            self.create_bo_locked(super::BO_SHMEM, bytes, true, p.handle, g.xdna_addr, g.map_offset)
        }

        /// Load a kernel: copy the PDI into a `DEV` BO and configure the
        /// context CU (16 B payload, `cu_func` 0), copy the instructions into
        /// a `DEV` BO. Returns the pinned kernel handle.
        pub fn load_kernel(&self, pdi: &[u8], insts: &[u8]) -> Result<LoadedKernel> {
            if pdi.is_empty() || insts.is_empty() || insts.len() % 4 != 0 {
                return Err(XdnaError::code("load-kernel", libc::EINVAL));
            }
            // PDI DEV BO + visibility.
            let (pdi_handle, pdi_xdna, pdi_map) =
                create_bo_get(self.inner.fd, super::BO_DEV, pdi.len(), "pdi")?;
            let pdi_bo =
                self.create_bo_locked(super::BO_DEV, pdi.len(), false, pdi_handle, pdi_xdna, pdi_map)?;
            pdi_bo.as_mut_slice().copy_from_slice(pdi);
            pdi_bo.publish().map_err(|_| XdnaError::os("pdi-sync"))?;

            // CONFIG_HWCTX CU: u16 num_cus + pad + {u32 cu_bo, u8 func, pad}.
            let cfg = super::HwctxParamConfigCu {
                num_cus: 1,
                pad: [0; 3],
                cu: super::CuConfig { cu_bo: pdi_handle, cu_func: 0, pad: [0; 3] },
            };
            let mut c = super::ConfigHwctx {
                handle: self.inner.hwctx,
                param_type: super::HWCTX_CONFIG_CU,
                param_val: &cfg as *const _ as u64,
                param_val_size: size_of::<super::HwctxParamConfigCu>() as u32,
                pad: 0,
            };
            ioctl(self.inner.fd, super::DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &mut c as *mut _ as *mut _)
                .map_err(|_| XdnaError::os("config-hwctx"))?;

            // Instructions DEV BO + visibility; device address is mandatory.
            let (in_handle, in_xdna, in_map) =
                create_bo_get(self.inner.fd, super::BO_DEV, insts.len(), "insts")?;
            if in_xdna == super::INVALID_ADDR {
                cleanup_handle(self.inner.fd, in_handle);
                return Err(XdnaError::code("insts-addr", libc::ENXIO));
            }
            let insts_bo = self.create_bo_locked(super::BO_DEV, insts.len(), false, in_handle, in_xdna, in_map)?;
            insts_bo.as_mut_slice().copy_from_slice(insts);
            insts_bo.publish().map_err(|_| XdnaError::os("insts-sync"))?;

            Ok(LoadedKernel {
                pdi: pdi_bo,
                insts: insts_bo,
                insts_addr: in_xdna,
                insts_bytes: insts.len(),
            })
        }

        /// Submit the kernel with up to [`MAX_BO_ARGS`](super::MAX_BO_ARGS)
        /// buffer arguments. Reuses the preallocated CMD BO: no allocation on
        /// the hot path after warmup (pins are a fixed-size array of `Arc`
        /// clones). Fails with `EBUSY` if a previous submission is in flight.
        pub fn submit(&self, kernel: &LoadedKernel, args: &[BoArg<'_>]) -> Result<Submission> {
            if args.len() > super::MAX_BO_ARGS {
                return Err(XdnaError::code("submit-args", libc::EINVAL));
            }
            let addrs: [u64; super::MAX_BO_ARGS] = {
                let mut a = [0u64; super::MAX_BO_ARGS];
                for (i, arg) in args.iter().enumerate() {
                    a[i] = arg.bo.vaddr();
                }
                a
            };
            let words = super::build_packet_words(
                kernel.insts_addr,
                kernel.insts_bytes as u32,
                &addrs[..args.len()],
            )?;

            let slot = self.inner.cmd.lock();
            {
                let mut inflight = self.inner.inflight.lock();
                if inflight.is_some() {
                    return Err(XdnaError::code("submit-busy", libc::EBUSY));
                }
                // Reset state NEW, write the packet, publish (flush + SYNC_TO).
                // SAFETY: CMD mapping live via Arc; len checked by construction.
                unsafe {
                    std::ptr::write_bytes(slot.ptr, 0, super::PACKET_WORDS);
                    std::ptr::copy_nonoverlapping(words.as_ptr(), slot.ptr, super::PACKET_WORDS);
                }
                cpu_flush(slot.ptr as *const u8, super::PACKET_BYTES);
                let mut s = super::SyncBo {
                    handle: slot.handle,
                    direction: super::SYNC_TO_DEVICE,
                    offset: 0,
                    size: super::PACKET_BYTES as u64,
                };
                ioctl(self.inner.fd, super::DRM_IOCTL_AMDXDNA_SYNC_BO, &mut s as *mut _ as *mut _)
                    .map_err(|_| XdnaError::os("submit-publish"))?;

                // EXEC_CMD arg handles: [insts, args...].
                let mut handles = [0u32; 1 + super::MAX_BO_ARGS];
                handles[0] = kernel.insts.handle();
                for (i, arg) in args.iter().enumerate() {
                    handles[1 + i] = arg.bo.handle();
                }
                let mut e = super::ExecCmd {
                    hwctx: self.inner.hwctx,
                    cmd_type: super::CMD_SUBMIT_EXEC_BUF,
                    cmd_handles: slot.handle as u64,
                    args: handles.as_ptr() as u64,
                    cmd_count: 1,
                    arg_count: (1 + args.len()) as u32,
                    ..Default::default()
                };
                ioctl(self.inner.fd, super::DRM_IOCTL_AMDXDNA_EXEC_CMD, &mut e as *mut _ as *mut _)
                    .map_err(|_| XdnaError::os("exec-cmd"))?;

                *inflight = Some(e.seq);

                // Fixed-size pins: kernel BOs + data BOs, no heap allocation.
                let mut pins: [Option<Bo>; 1 + super::MAX_BO_ARGS] = Default::default();
                pins[0] = Some(kernel.insts.clone());
                for (i, arg) in args.iter().enumerate() {
                    pins[1 + i] = Some(arg.bo.clone());
                }
                // NOTE: `kernel.pdi` stays alive via the caller's
                // `LoadedKernel`; only execution pins travel with the job.
                Ok(Submission { dev: self.inner.clone(), seq: e.seq, pins })
            }
        }

        /// Explicit idle check: `Ok` when no submission is in flight,
        /// `EBUSY` otherwise. No threads or background work exist; this is
        /// the serialization fence for the single-flight protocol.
        pub fn quiesce(&self) -> Result<()> {
            let inflight = self.inner.inflight.lock();
            if inflight.is_some() {
                return Err(XdnaError::code("quiesce", libc::EBUSY));
            }
            Ok(())
        }
    }

    /// A submitted job. Holds `Arc` pins of the kernel instruction BO and
    /// every argument BO until the submission is dropped (wait first).
    pub struct Submission {
        dev: Arc<DeviceInner>,
        seq: u64,
        #[allow(dead_code)]
        pins: [Option<Bo>; 1 + super::MAX_BO_ARGS],
    }

    impl Submission {
        /// Job sequence number (syncobj timeline point).
        pub fn seq(&self) -> u64 {
            self.seq
        }

        /// Wait for completion with a `CLOCK_MONOTONIC`-absolute deadline
        /// derived from `timeout`, then verify the packet reached
        /// `COMPLETED`. On timeout the job stays in flight: the device may
        /// still write the BOs, so the caller must not resubmit or touch
        /// them — recover by dropping the device.
        pub fn wait(&self, timeout: Duration) -> Result<()> {
            // SAFETY: clock_gettime with a valid timespec.
            let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
            if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) } != 0 {
                return Err(XdnaError::os("wait-clock"));
            }
            let now: i128 = ts.tv_sec as i128 * 1_000_000_000 + ts.tv_nsec as i128;
            let dl = now + timeout.as_nanos() as i128;
            let dl = dl.clamp(0, i64::MAX as i128) as i64;

            let syncobj = self.dev.syncobj;
            let seq = self.seq;
            let mut w = super::SyncobjTimelineWait {
                handles: &syncobj as *const _ as u64,
                points: &seq as *const _ as u64,
                timeout_nsec: dl,
                count_handles: 1,
                flags: super::SYNCOBJ_WAIT_FOR_SUBMIT,
                ..Default::default()
            };
            ioctl(self.dev.fd, super::DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &mut w as *mut _ as *mut _)
                .map_err(|_| XdnaError::os("wait"))?;

            let state = {
                let slot = self.dev.cmd.lock();
                // SAFETY: CMD mapping live via Arc; device wrote it before
                // signaling the point we just waited on.
                unsafe { std::ptr::read_volatile(slot.ptr) & 0xf }
            };
            if state != super::ERT_STATE_COMPLETED {
                return Err(XdnaError::code("packet-state", libc::EIO));
            }
            // Clear single-flight only for our own sequence.
            let mut inflight = self.dev.inflight.lock();
            if *inflight == Some(self.seq) {
                *inflight = None;
            }
            Ok(())
        }
    }

    fn cleanup_handle(fd: RawFd, handle: u32) {
        if handle != super::INVALID_BO_HANDLE {
            let mut g = super::GemClose { handle, pad: 0 };
            let _ = ioctl(fd, super::DRM_IOCTL_GEM_CLOSE, &mut g as *mut _ as *mut _);
        }
    }

    fn destroy_hwctx_syncobj(fd: RawFd, hwctx: u32, syncobj: u32) {
        let mut d = super::SyncobjDestroy { handle: syncobj, pad: 0 };
        let _ = ioctl(fd, super::DRM_IOCTL_SYNCOBJ_DESTROY, &mut d as *mut _ as *mut _);
        let mut h = super::DestroyHwctx { handle: hwctx, pad: 0 };
        let _ = ioctl(fd, super::DRM_IOCTL_AMDXDNA_DESTROY_HWCTX, &mut h as *mut _ as *mut _);
    }

    fn create_bo_get(fd: RawFd, bo_type: u32, size: usize, stage: &'static str) -> Result<(u32, u64, u64)> {
        let mut c = super::CreateBo { flags: 0, vaddr: 0, size: size as u64, bo_type, handle: 0 };
        ioctl(fd, super::DRM_IOCTL_AMDXDNA_CREATE_BO, &mut c as *mut _ as *mut _)
            .map_err(|_| XdnaError::os(stage))?;
        if c.handle == super::INVALID_BO_HANDLE {
            return Err(XdnaError::code(stage, libc::ENXIO));
        }
        let mut g = super::GetBoInfo { handle: c.handle, ..Default::default() };
        if ioctl(fd, super::DRM_IOCTL_AMDXDNA_GET_BO_INFO, &mut g as *mut _ as *mut _)
            .map_err(|_| XdnaError::os("get-bo-info"))
            .is_err()
        {
            cleanup_handle(fd, c.handle);
            return Err(XdnaError::os("get-bo-info"));
        }
        Ok((c.handle, g.xdna_addr, g.map_offset))
    }
}

// ---------------------------------------------------------------------------
// Non-Linux stub (same API surface, every op fails)
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "linux"))]
mod imp_stub {
    use super::*;

    #[derive(Clone)]
    pub struct Bo;
    #[derive(Clone, Copy)]
    pub struct BoArg<'a> {
        pub bo: &'a Bo,
    }
    pub struct LoadedKernel;
    pub struct Submission;
    #[derive(Clone)]
    pub struct XdnaDevice;

    const OFF: XdnaError = XdnaError { stage: "linux-only", errno: libc::ENOSYS };

    impl Bo {
        pub fn vaddr(&self) -> u64 {
            0
        }
        pub fn handle(&self) -> u32 {
            0
        }
        pub fn len(&self) -> usize {
            0
        }
        pub fn is_imported(&self) -> bool {
            false
        }
        pub fn kind(&self) -> BoKind {
            BoKind::Share
        }
        pub fn as_slice(&self) -> &[u8] {
            &[]
        }
        pub fn as_mut_slice(&self) -> &mut [u8] {
            // SAFETY: zero-length, aligned, never dereferenced for I/O.
            unsafe { &mut *std::ptr::slice_from_raw_parts_mut(std::ptr::NonNull::dangling().as_ptr(), 0) }
        }
        pub fn publish(&self) -> Result<()> {
            Err(OFF)
        }
        pub fn invalidate(&self) {}
    }

    impl LoadedKernel {
        pub fn insts_len(&self) -> usize {
            0
        }
    }

    impl XdnaDevice {
        pub fn open(_path: &Path) -> Result<Self> {
            Err(OFF)
        }
        pub fn alloc(&self, _bytes: usize, _kind: BoKind) -> Result<Bo> {
            Err(OFF)
        }
        pub fn import_dmabuf(&self, _fd: RawFd, _bytes: usize) -> Result<Bo> {
            Err(OFF)
        }
        pub fn load_kernel(&self, _pdi: &[u8], _insts: &[u8]) -> Result<LoadedKernel> {
            Err(OFF)
        }
        pub fn submit(&self, _kernel: &LoadedKernel, _args: &[BoArg<'_>]) -> Result<Submission> {
            Err(OFF)
        }
        pub fn quiesce(&self) -> Result<()> {
            Err(OFF)
        }
    }

    impl Submission {
        pub fn seq(&self) -> u64 {
            0
        }
        pub fn wait(&self, _timeout: Duration) -> Result<()> {
            Err(OFF)
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn uapi_struct_sizes_match_c_header() {
        // Locked to the installed `/usr/include/drm/amdxdna_accel.h`
        // == kernel 7.0.0-31 uapi contract (see `_Static_assert`s in
        // raw_npu.c:144-153).
        assert_eq!(size_of::<QosInfo>(), 24);
        assert_eq!(size_of::<CreateHwctx>(), 56);
        assert_eq!(size_of::<DestroyHwctx>(), 8);
        assert_eq!(size_of::<HwctxParamConfigCu>(), 16);
        assert_eq!(size_of::<ConfigHwctx>(), 24);
        assert_eq!(size_of::<CreateBo>(), 32);
        assert_eq!(size_of::<GetBoInfo>(), 48);
        assert_eq!(size_of::<SyncBo>(), 24);
        assert_eq!(size_of::<ExecCmd>(), 56);
        assert_eq!(size_of::<PrimeHandle>(), 12);
        assert_eq!(size_of::<GemClose>(), 8);
        assert_eq!(size_of::<SyncobjDestroy>(), 8);
        assert_eq!(size_of::<SyncobjTimelineWait>(), 48);
    }

    #[test]
    fn uapi_field_offsets_match_c_header() {
        // Spot offsets from the C `offsetof` probe (`/tmp/ioctl_probe`).
        assert_eq!(offset_of!(CreateBo, bo_type), 24);
        assert_eq!(offset_of!(CreateBo, handle), 28);
        assert_eq!(offset_of!(SyncBo, direction), 4);
        assert_eq!(offset_of!(GetBoInfo, map_offset), 24);
        assert_eq!(offset_of!(GetBoInfo, vaddr), 32);
        assert_eq!(offset_of!(GetBoInfo, xdna_addr), 40);
        assert_eq!(offset_of!(SyncBo, size), 16);
        assert_eq!(offset_of!(ExecCmd, hwctx), 16);
        assert_eq!(offset_of!(ExecCmd, cmd_type), 20);
        assert_eq!(offset_of!(ExecCmd, cmd_handles), 24);
        assert_eq!(offset_of!(ExecCmd, args), 32);
        assert_eq!(offset_of!(ExecCmd, cmd_count), 40);
        assert_eq!(offset_of!(ExecCmd, arg_count), 44);
        assert_eq!(offset_of!(ExecCmd, seq), 48);
        assert_eq!(offset_of!(PrimeHandle, flags), 4);
        assert_eq!(offset_of!(PrimeHandle, fd), 8);
        assert_eq!(offset_of!(SyncobjTimelineWait, points), 8);
        assert_eq!(offset_of!(SyncobjTimelineWait, timeout_nsec), 16);
        assert_eq!(offset_of!(SyncobjTimelineWait, count_handles), 24);
        assert_eq!(offset_of!(SyncobjTimelineWait, flags), 28);
        assert_eq!(offset_of!(HwctxParamConfigCu, cu), 8);
        assert_eq!(offset_of!(CuConfig, cu_func), 4);
    }

    #[test]
    fn ioctl_numbers_match_c_probe() {
        // Literals printed by compiling the installed headers
        // (`cc ioctl_probe.c -o ioctl_probe`); NOT recomputed here.
        assert_eq!(DRM_IOCTL_AMDXDNA_CREATE_HWCTX, 0xc038_6440);
        assert_eq!(DRM_IOCTL_AMDXDNA_DESTROY_HWCTX, 0xc008_6441);
        assert_eq!(DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, 0xc018_6442);
        assert_eq!(DRM_IOCTL_AMDXDNA_CREATE_BO, 0xc020_6443);
        assert_eq!(DRM_IOCTL_AMDXDNA_GET_BO_INFO, 0xc030_6444);
        assert_eq!(DRM_IOCTL_AMDXDNA_SYNC_BO, 0xc018_6445);
        assert_eq!(DRM_IOCTL_AMDXDNA_EXEC_CMD, 0xc038_6446);
        assert_eq!(DRM_IOCTL_GEM_CLOSE, 0x4008_6409);
        assert_eq!(DRM_IOCTL_PRIME_FD_TO_HANDLE, 0xc00c_642e);
        assert_eq!(DRM_IOCTL_SYNCOBJ_DESTROY, 0xc008_64c0);
        assert_eq!(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, 0xc030_64ca);
    }

    #[test]
    fn ert_header_encodes_new_count16_startcu_typecu() {
        // 0x30010001: state NEW(1), count 16, opcode START_CU(0), type CU(3).
        assert_eq!(ERT_HDR_COUNT16, 0x3001_0001);
        assert_eq!(ERT_HDR_COUNT16 & 0xf, ERT_STATE_NEW);
        assert_eq!((ERT_HDR_COUNT16 >> 12) & 0x7ff, 16);
        assert_eq!((ERT_HDR_COUNT16 >> 23) & 0x1f, 0);
        assert_eq!((ERT_HDR_COUNT16 >> 28) & 0xf, 3);
    }

    /// Golden ERT packet bytes from `raw_npu.c::build_packet` (68 bytes):
    /// header, cu_mask, opcode u64=3, instr u64, ninstr u32 (bytes),
    /// bo0..bo2 u64, bo3..bo4 zero. `bo0` at packet `0x1c`
    /// (payload-relative `0x14`, unaligned by 4).
    #[test]
    fn ert_packet_bytes_match_c_golden() {
        let words = build_packet_words(
            0x0102_0304_0506_0708,
            142_544, // stock gate insts bytes
            &[0x1111_1111_2222_2222, 0x3333_3333_4444_4444, 0x5555_5555_6666_6666],
        )
        .unwrap();
        assert_eq!(words.len(), PACKET_WORDS);
        let mut bytes = [0u8; PACKET_BYTES];
        for (i, w) in words.iter().enumerate() {
            bytes[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        #[rustfmt::skip]
        const GOLDEN: [u8; PACKET_BYTES] = [
            0x01, 0x00, 0x01, 0x30, // header 0x30010001
            0x01, 0x00, 0x00, 0x00, // cu_mask
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // opcode u64 = 3
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // instr addr
            0xd0, 0x2c, 0x02, 0x00, // ninstr bytes = 142544
            0x22, 0x22, 0x22, 0x22, 0x11, 0x11, 0x11, 0x11, // bo0
            0x44, 0x44, 0x44, 0x44, 0x33, 0x33, 0x33, 0x33, // bo1
            0x66, 0x66, 0x66, 0x66, 0x55, 0x55, 0x55, 0x55, // bo2
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // bo3 unused
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // bo4 unused
        ];
        assert_eq!(bytes, GOLDEN);
        // bo0 lands at packet 0x1c == payload-relative 0x14 (8-byte prefix).
        assert_eq!(BO0_PACKET_OFFSET, 0x1c);
        assert_eq!(&bytes[BO0_PACKET_OFFSET..BO0_PACKET_OFFSET + 8], &GOLDEN[28..36]);
        assert_eq!(BO0_PACKET_OFFSET - 8, 0x14);
        assert_eq!(BO0_PACKET_OFFSET % 8, 4);
    }

    #[test]
    fn packet_builder_pads_and_rejects() {
        // No args: everything past ninstr is zero, header/count unchanged.
        let words = build_packet_words(0xabcd, 4, &[]).unwrap();
        assert_eq!(words[0], 0x3001_0001);
        assert_eq!(words[4], 0xabcd);
        let words = build_packet_words(0, 0, &[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(words[7], 1);
        assert_eq!(words[15], 5);
        assert_eq!(words[16], 0); // high word of bo4
        // Six args do not fit bo0..bo4.
        let err = build_packet_words(0, 0, &[1, 2, 3, 4, 5, 6]).unwrap_err();
        assert_eq!(err, XdnaError { stage: "submit-args", errno: libc::EINVAL });
    }

    /// Integration: open `/dev/accel/accel0` and run a tiny GEMM if the
    /// artifact files are provided. Skips (passes) when the device or the
    /// env-provided artifact is absent.
    ///
    /// Env: `HIPFIRE_XDNA_PDI`, `HIPFIRE_XDNA_INSTS`, `HIPFIRE_XDNA_A`,
    /// `HIPFIRE_XDNA_B`, `HIPFIRE_XDNA_CREF`, `HIPFIRE_XDNA_MKN` (`MxKxN`,
    /// A = M*K int8 bytes, B = K*N int8 bytes, C = M*N int32 LE bytes).
    #[test]
    #[ignore]
    #[cfg(target_os = "linux")]
    fn tiny_gemm_on_device_if_present() {
        use std::time::Duration;

        fn env(k: &str) -> Option<String> {
            std::env::var(k).ok()
        }
        let (pdi_p, insts_p, a_p, b_p, cref_p, mkn) = match (
            env("HIPFIRE_XDNA_PDI"),
            env("HIPFIRE_XDNA_INSTS"),
            env("HIPFIRE_XDNA_A"),
            env("HIPFIRE_XDNA_B"),
            env("HIPFIRE_XDNA_CREF"),
            env("HIPFIRE_XDNA_MKN"),
        ) {
            (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) => (a, b, c, d, e, f),
            _ => {
                eprintln!("skip: HIPFIRE_XDNA_* artifact env not set");
                return;
            }
        };
        if !Path::new("/dev/accel/accel0").exists() {
            eprintln!("skip: no /dev/accel/accel0");
            return;
        }
        let dims: Vec<usize> = mkn.split('x').filter_map(|s| s.parse().ok()).collect();
        assert_eq!(dims.len(), 3, "HIPFIRE_XDNA_MKN must look like 512x64x256");
        let (m, k, n) = (dims[0], dims[1], dims[2]);

        let dev = XdnaDevice::open(Path::new("/dev/accel/accel0")).expect("open device");
        let kernel = dev
            .load_kernel(&std::fs::read(&pdi_p).unwrap(), &std::fs::read(&insts_p).unwrap())
            .expect("load kernel");
        let a = dev.alloc(m * k, BoKind::Share).unwrap();
        let b = dev.alloc(k * n, BoKind::Share).unwrap();
        let c = dev.alloc(m * n * 4, BoKind::Share).unwrap();
        a.as_mut_slice().copy_from_slice(&std::fs::read(&a_p).unwrap());
        b.as_mut_slice().copy_from_slice(&std::fs::read(&b_p).unwrap());
        a.publish().unwrap();
        b.publish().unwrap();
        let sub = dev.submit(&kernel, &[(&a).into(), (&b).into(), (&c).into()]).unwrap();
        sub.wait(Duration::from_secs(30)).expect("tiny GEMM completion");
        c.invalidate();
        assert_eq!(c.as_slice(), std::fs::read(&cref_p).unwrap().as_slice());
        dev.quiesce().unwrap();
    }
}
