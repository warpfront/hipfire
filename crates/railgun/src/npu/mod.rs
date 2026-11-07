// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// ioctl ABI transcribed from the Linux amdxdna UAPI header (GPL-2.0 WITH Linux-syscall-note); see NOTICE.
//! railgun::npu: direct amdxdna dispatch for the XDNA2 (AIE2P) NPU — DRM ioctls and libc only.
//!
//! ABI source of truth: the in-tree kernel UAPI `include/uapi/drm/amdxdna_accel.h` (kernel 7.0,
//! driver 0.7.0) and `drivers/accel/amdxdna/aie2_message.c` (how CONFIG_CU and EXEC_CMD payloads
//! reach the firmware). No libxrt, no vendor runtime.
//!
//! Lifecycle: `Device::open` → `heap` (64 MiB DEV_HEAP, 64 MiB-aligned map) → `HwCtx::create`
//! → `config_cu(pdi)` → BOs → `submit` (ERT_START_CU) → `wait` (syncobj timeline) → drop.
#![allow(clippy::missing_safety_doc)]

pub mod fclk;
pub mod hip_runtime;
pub mod tape;
pub mod trunk;
use core::ffi::c_void;
use core::mem::size_of;

#[cfg(unix)]
mod sys {
    use core::ffi::c_void;
    extern "C" {
        // Must match libc's variadic C signature exactly; rustc >= 1.99 rejects a
        // mismatched declaration of a symbol the standard library also uses.
        pub fn open(path: *const core::ffi::c_char, flags: i32, ...) -> i32;
        pub fn close(fd: i32) -> i32;
        pub fn ioctl(fd: i32, request: u64, arg: *mut c_void) -> i32;
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
        pub fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        pub fn lseek(fd: i32, off: i64, whence: i32) -> i64;
        pub fn __errno_location() -> *mut i32;
        pub fn clock_gettime(clock_id: i32, tp: *mut [i64; 2]) -> i32;
        /// `struct rlimit { rlim_t rlim_cur, rlim_max; }` with 64-bit `rlim_t` (x86_64 Linux).
        pub fn getrlimit(resource: i32, rlim: *mut [u64; 2]) -> i32;
        pub fn setrlimit(resource: i32, rlim: *const [u64; 2]) -> i32;
    }
    pub const O_RDWR: i32 = 2;
    pub const SEEK_SET: i32 = 0;
    pub const SEEK_END: i32 = 2;
    pub const O_CLOEXEC: i32 = 0o2000000;
    pub const PROT_RW: i32 = 3;
    pub const PROT_NONE: i32 = 0;
    pub const MAP_SHARED: i32 = 1;
    pub const MAP_PRIVATE: i32 = 2;
    pub const MAP_FIXED: i32 = 0x10;
    pub const MAP_ANON: i32 = 0x20;
    pub const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;
    pub const MADV_HUGEPAGE: i32 = 14;
    pub const CLOCK_MONOTONIC: i32 = 1;
    pub const RLIMIT_MEMLOCK: i32 = 8;
}

/// Windows has no amdxdna driver, dma-buf, `/dev/accel/*`, POSIX VM syscalls or
/// monotonic-clock UAPI: every entry fails closed with the caller's own
/// sentinel (fd -1, null/`MAP_FAILED`, rc -1), which the existing checks turn
/// into `Err`. `clock_gettime` is the exception: it is real (wall clock) so
/// the shared `monotonic_ns` assert holds and trace deltas stay ordered; the
/// NPU backend itself can never succeed here. Linux behavior is unchanged.
#[cfg(windows)]
mod sys {
    use core::ffi::{c_char, c_void};

    pub unsafe fn open(_path: *const c_char, _flags: i32) -> i32 {
        -1
    }
    pub unsafe fn close(_fd: i32) -> i32 {
        -1
    }
    pub unsafe fn ioctl(_fd: i32, _request: u64, _arg: *mut c_void) -> i32 {
        -1
    }
    pub unsafe fn mmap(
        _addr: *mut c_void,
        _len: usize,
        _prot: i32,
        _flags: i32,
        _fd: i32,
        _off: i64,
    ) -> *mut c_void {
        MAP_FAILED
    }
    pub unsafe fn munmap(_addr: *mut c_void, _len: usize) -> i32 {
        -1
    }
    pub unsafe fn madvise(_addr: *mut c_void, _len: usize, _advice: i32) -> i32 {
        // Hint only; the caller ignores the result. The mapping itself
        // already failed closed above.
        -1
    }
    pub unsafe fn lseek(_fd: i32, _off: i64, _whence: i32) -> i64 {
        -1
    }
    pub unsafe fn __errno_location() -> *mut i32 {
        use core::sync::atomic::AtomicI32;
        static ERRNO: AtomicI32 = AtomicI32::new(2); // ENOENT: no NPU driver
        ERRNO.as_ptr()
    }
    pub unsafe fn clock_gettime(_clock_id: i32, tp: *mut [i64; 2]) -> i32 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        unsafe {
            *tp = [now.as_secs() as i64, now.subsec_nanos() as i64];
        }
        0
    }
    pub unsafe fn getrlimit(_resource: i32, _rlim: *mut [u64; 2]) -> i32 {
        -1
    }
    pub unsafe fn setrlimit(_resource: i32, _rlim: *const [u64; 2]) -> i32 {
        -1
    }
    pub const O_RDWR: i32 = 2;
    pub const SEEK_SET: i32 = 0;
    pub const SEEK_END: i32 = 2;
    pub const O_CLOEXEC: i32 = 0o2000000;
    pub const PROT_RW: i32 = 3;
    pub const PROT_NONE: i32 = 0;
    pub const MAP_SHARED: i32 = 1;
    pub const MAP_PRIVATE: i32 = 2;
    pub const MAP_FIXED: i32 = 0x10;
    pub const MAP_ANON: i32 = 0x20;
    pub const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;
    pub const MADV_HUGEPAGE: i32 = 14;
    pub const CLOCK_MONOTONIC: i32 = 1;
    pub const RLIMIT_MEMLOCK: i32 = 8;
}

pub fn errno() -> i32 {
    unsafe { *sys::__errno_location() }
}

// ---- ioctl numbers: _IOWR('d', DRM_COMMAND_BASE + id, T) ----
const fn iowr(nr: u32, size: usize) -> u64 {
    (3u64 << 30) | ((size as u64) << 16) | (0x64u64 << 8) | nr as u64
}
const fn iow(nr: u32, size: usize) -> u64 {
    (1u64 << 30) | ((size as u64) << 16) | (0x64u64 << 8) | nr as u64
}
fn amdxdna<T>(id: u32) -> u64 {
    iowr(0x40 + id, size_of::<T>())
}
const DRM_IOCTL_GEM_CLOSE: u64 = iow(0x09, 8);
const DRM_IOCTL_PRIME_FD_TO_HANDLE: u64 = iowr(0x2e, 12);
const DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT: u64 = iowr(0xCA, 48);
const DRM_IOCTL_SYNCOBJ_DESTROY: u64 = iowr(0xC0, 8);
const SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT: u32 = 1 << 1;

// enum amdxdna_drm_ioctl_id
const CREATE_HWCTX: u32 = 0;
const DESTROY_HWCTX: u32 = 1;
const CONFIG_HWCTX: u32 = 2;
const CREATE_BO: u32 = 3;
const GET_BO_INFO: u32 = 4;
const EXEC_CMD: u32 = 6;
const GET_INFO: u32 = 7;
// enum amdxdna_bo_type
const BO_SHMEM: u32 = 1;
const BO_DEV_HEAP: u32 = 2;
const BO_DEV: u32 = 3;
const BO_CMD: u32 = 4;
// enum amdxdna_drm_get_param
const QUERY_AIE_STATUS: u32 = 0;
const QUERY_AIE_METADATA: u32 = 1;
const QUERY_CLOCK_METADATA: u32 = 3;
const QUERY_FIRMWARE_VERSION: u32 = 8;
// ERT (xrt ert.h): header = state | count<<12 | opcode<<23 | type<<28
const ERT_START_CU: u32 = 0;
const ERT_CU: u32 = 3;
pub const ERT_STATE_NEW: u32 = 1;
pub const ERT_STATE_COMPLETED: u32 = 4;

#[repr(C)]
#[derive(Default)]
struct QosInfo {
    gops: u32,
    fps: u32,
    dma_bandwidth: u32,
    latency: u32,
    frame_exec_time: u32,
    priority: u32,
}
#[repr(C)]
#[derive(Default)]
struct CreateHwctx {
    ext: u64,
    ext_flags: u64,
    qos_p: u64,
    umq_bo: u32,
    log_buf_bo: u32,
    max_opc: u32,
    num_tiles: u32,
    mem_size: u32,
    umq_doorbell: u32,
    handle: u32,
    syncobj_handle: u32,
}
#[repr(C)]
#[derive(Default)]
struct DestroyHwctx {
    handle: u32,
    pad: u32,
}
#[repr(C)]
#[derive(Default)]
struct ConfigHwctx {
    handle: u32,
    param_type: u32,
    param_val: u64,
    param_val_size: u32,
    pad: u32,
}
#[repr(C)]
#[derive(Default)]
struct ConfigCu {
    num_cus: u16,
    pad: [u16; 3],
    cu_bo: u32,
    cu_func: u8,
    pad2: [u8; 3],
}
#[repr(C)]
#[derive(Default)]
struct CreateBo {
    flags: u64,
    vaddr: u64,
    size: u64,
    ty: u32,
    handle: u32,
}
#[repr(C)]
#[derive(Default)]
struct GetBoInfo {
    ext: u64,
    ext_flags: u64,
    handle: u32,
    pad: u32,
    map_offset: u64,
    vaddr: u64,
    xdna_addr: u64,
}
#[repr(C)]
#[derive(Default)]
struct PrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}
#[repr(C)]
#[derive(Default)]
struct ExecCmd {
    ext: u64,
    ext_flags: u64,
    hwctx: u32,
    ty: u32,
    cmd_handles: u64,
    args: u64,
    cmd_count: u32,
    arg_count: u32,
    seq: u64,
}
#[repr(C)]
#[derive(Default)]
struct GetInfo {
    param: u32,
    buffer_size: u32,
    buffer: u64,
}
#[repr(C)]
#[derive(Default)]
struct QueryAieStatus {
    buffer: u64,
    buffer_size: u32,
    cols_filled: u32,
}
#[repr(C)]
#[derive(Default)]
struct SyncobjTimelineWait {
    handles: u64,
    points: u64,
    timeout_nsec: i64,
    count_handles: u32,
    flags: u32,
    first_signaled: u32,
    pad: u32,
    deadline_nsec: u64,
}

const _: () = {
    assert!(size_of::<CreateHwctx>() == 56);
    assert!(size_of::<ConfigHwctx>() == 24);
    assert!(size_of::<ConfigCu>() == 16);
    assert!(size_of::<CreateBo>() == 32);
    assert!(size_of::<GetBoInfo>() == 48);
    assert!(size_of::<ExecCmd>() == 56);
    assert!(size_of::<SyncobjTimelineWait>() == 48);
};

fn ioctl<T>(fd: i32, req: u64, arg: &mut T, what: &str) -> Result<(), String> {
    let r = unsafe { sys::ioctl(fd, req, arg as *mut T as *mut c_void) };
    if r < 0 {
        Err(format!("ioctl {what}: rc={r} errno={}", errno()))
    } else {
        Ok(())
    }
}

/// Write back + invalidate a CPU cache range. The installed shim's default (non-coherent) sync is
/// exactly this (`clflush_data`); SYNC_BO FROM_DEVICE is rejected by this driver for SHMEM BOs.
pub unsafe fn clflush(p: *const u8, len: usize) {
    use core::arch::x86_64::{_mm_clflush, _mm_mfence};
    _mm_mfence();
    let base = (p as usize) & !63;
    let end = p as usize + len;
    let mut a = base;
    while a < end {
        _mm_clflush(a as *const u8);
        a += 64;
    }
    _mm_mfence();
}

/// AIE array geometry from DRM_AMDXDNA_QUERY_AIE_METADATA.
#[derive(Debug, Clone, Copy, Default)]
pub struct AieMeta {
    pub col_size: u32,
    pub cols: u16,
    pub rows: u16,
    pub version: (u32, u32),
    /// (row_count, row_start, dma_channel_count, lock_count, event_reg_count) per tile kind
    pub core: (u16, u16, u16, u16, u16),
    pub mem: (u16, u16, u16, u16, u16),
    pub shim: (u16, u16, u16, u16, u16),
}

pub struct Device {
    pub fd: i32,
    pub meta: AieMeta,
    heap_host: *mut u8,
    heap_xdna: u64,
    heap_len: usize,
    heap_handle: u32,
}

/// A buffer object: DEV BOs live inside the heap window (device address `xdna`, CPU view `host`);
/// SHMEM/CMD BOs are mmap'd separately and the NPU reaches them through their host VA.
pub struct Bo {
    pub handle: u32,
    pub xdna: u64,
    pub host: *mut u8,
    pub len: usize,
    mapped: bool,
    fd: i32,
}

impl Drop for Bo {
    fn drop(&mut self) {
        unsafe {
            if self.mapped {
                sys::munmap(self.host as *mut c_void, self.len);
            }
            let mut c = [self.handle, 0u32];
            sys::ioctl(self.fd, DRM_IOCTL_GEM_CLOSE, c.as_mut_ptr() as *mut c_void);
        }
    }
}

impl Bo {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.host, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.host, self.len) }
    }
    pub fn flush(&self) {
        unsafe { clflush(self.host, self.len) }
    }
    /// The address the firmware patches into shim BDs for this buffer (`DDR_PATCH` arg value):
    /// the device address when the driver gives one, else the host VA (SHMEM BOs under SVA).
    pub fn dev_addr(&self) -> u64 {
        if self.xdna != u64::MAX {
            self.xdna
        } else {
            self.host as u64
        }
    }
}

impl Device {
    pub fn open() -> Result<Device, String> {
        let fd = unsafe { sys::open(c"/dev/accel/accel0".as_ptr(), sys::O_RDWR | sys::O_CLOEXEC) };
        if fd < 0 {
            return Err(format!("open /dev/accel/accel0: errno={}", errno()));
        }
        let mut d = Device {
            fd,
            meta: AieMeta::default(),
            heap_host: core::ptr::null_mut(),
            heap_xdna: 0,
            heap_len: 0,
            heap_handle: 0,
        };
        d.meta = d.query_meta()?;
        Ok(d)
    }

    fn get_info(&self, param: u32, buf: &mut [u8]) -> Result<(), String> {
        let mut gi = GetInfo {
            param,
            buffer_size: buf.len() as u32,
            buffer: buf.as_mut_ptr() as u64,
        };
        ioctl(self.fd, amdxdna::<GetInfo>(GET_INFO), &mut gi, "get_info")
    }

    fn query_meta(&self) -> Result<AieMeta, String> {
        let mut b = [0u8; 64];
        self.get_info(QUERY_AIE_METADATA, &mut b)?;
        let u16at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let u32at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let tile = |o: usize| {
            (
                u16at(o),
                u16at(o + 2),
                u16at(o + 4),
                u16at(o + 6),
                u16at(o + 8),
            )
        };
        Ok(AieMeta {
            col_size: u32at(0),
            cols: u16at(4),
            rows: u16at(6),
            version: (u32at(8), u32at(12)),
            core: tile(16),
            mem: tile(32),
            shim: tile(48),
        })
    }

    pub fn firmware_version(&self) -> Result<[u32; 4], String> {
        let mut b = [0u8; 16];
        self.get_info(QUERY_FIRMWARE_VERSION, &mut b)?;
        Ok(core::array::from_fn(|i| {
            u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
        }))
    }

    /// (name, MHz) for the MP-NPU clock and the H-clock.
    pub fn clocks(&self) -> Result<[(String, u32); 2], String> {
        let mut b = [0u8; 48];
        self.get_info(QUERY_CLOCK_METADATA, &mut b)?;
        let one = |o: usize| {
            let name = String::from_utf8_lossy(&b[o..o + 16])
                .trim_end_matches('\0')
                .to_string();
            (
                name,
                u32::from_le_bytes(b[o + 16..o + 20].try_into().unwrap()),
            )
        };
        Ok([one(0), one(24)])
    }

    /// Raw per-column AIE status dump (firmware-defined layout), `cols_filled` columns.
    pub fn aie_status(&self, buf: &mut [u8]) -> Result<u32, String> {
        let mut st = QueryAieStatus {
            buffer: buf.as_mut_ptr() as u64,
            buffer_size: buf.len() as u32,
            cols_filled: 0,
        };
        let mut gi = GetInfo {
            param: QUERY_AIE_STATUS,
            buffer_size: size_of::<QueryAieStatus>() as u32,
            buffer: &mut st as *mut _ as u64,
        };
        ioctl(
            self.fd,
            amdxdna::<GetInfo>(GET_INFO),
            &mut gi,
            "get_info aie_status",
        )?;
        Ok(st.cols_filled)
    }

    // BO mappings (heap, SHMEM/CMD, imported dma-buf, userptr) are never `MAP_LOCKED`: the driver owns their
    // residency and `MAP_LOCKED` would only test RLIMIT_MEMLOCK (8 MiB under ssh) without locking anything
    // (Linux 7.3 amdxdna). Pages: drm_gem_get_pages marks the shmem mapping unevictable (drm_gem.c:690); the
    // heap is pinned for the hwctx lifetime (amdxdna/aie2_ctx.c:699), exec BOs from submit until free
    // (amdxdna_ctx.c:556, amdxdna_gem.c:649), userptr pages are pin_user_pages(FOLL_LONGTERM)
    // (amdxdna_ubuf.c:146). The mmap is VM_PFNMAP/VM_MIXEDMAP (drm_gem_shmem_helper.c:784, amdxdna_gem.c:543,
    // amdxdna_cbuf.c:187, ttm_bo_vm.c:511) and prefaulted page by page (amdxdna_gem.c:483). For such vmas
    // do_mmap checks the rlimit and fails with EAGAIN (mm/mmap.c:428), then drops VM_LOCKED without charging
    // locked_vm (mm/vma.c:2747), so VmLck stays 0.
    /// Allocate the DEV_HEAP and map it at a 64 MiB-aligned host VA (the firmware requires the
    /// heap's host mapping alignment under SVA; see the installed shim's `buffer::vaddr`).
    pub fn map_heap(&mut self, len: usize) -> Result<(), String> {
        let mut cb = CreateBo {
            size: len as u64,
            ty: BO_DEV_HEAP,
            ..Default::default()
        };
        ioctl(
            self.fd,
            amdxdna::<CreateBo>(CREATE_BO),
            &mut cb,
            "create_bo dev_heap",
        )?;
        let mut gi = GetBoInfo {
            handle: cb.handle,
            ..Default::default()
        };
        ioctl(
            self.fd,
            amdxdna::<GetBoInfo>(GET_BO_INFO),
            &mut gi,
            "get_bo_info dev_heap",
        )?;
        let align = 64usize << 20;
        let reserve = unsafe {
            sys::mmap(
                core::ptr::null_mut(),
                len + align,
                sys::PROT_NONE,
                sys::MAP_PRIVATE | sys::MAP_ANON,
                -1,
                0,
            )
        };
        if reserve == sys::MAP_FAILED {
            return Err(format!("reserve heap VA: errno={}", errno()));
        }
        let aligned = ((reserve as usize + align - 1) & !(align - 1)) as *mut c_void;
        let host = unsafe {
            sys::mmap(
                aligned,
                len,
                sys::PROT_RW,
                sys::MAP_SHARED | sys::MAP_FIXED,
                self.fd,
                gi.map_offset as i64,
            )
        };
        if host == sys::MAP_FAILED {
            return Err(format!("mmap heap: errno={}", errno()));
        }
        // Trim the unused parts of the reservation.
        unsafe {
            let head = aligned as usize - reserve as usize;
            if head > 0 {
                sys::munmap(reserve, head);
            }
            let tail = align - head;
            if tail > 0 {
                sys::munmap((aligned as usize + len) as *mut c_void, tail);
            }
        }
        self.heap_host = host as *mut u8;
        self.heap_xdna = gi.xdna_addr;
        self.heap_len = len;
        self.heap_handle = cb.handle;
        Ok(())
    }

    /// DEV BO sub-allocated from the heap, filled with `src` (PDI, transaction stream).
    pub fn dev_bo(&self, src: &[u8]) -> Result<Bo, String> {
        let mut cb = CreateBo {
            size: src.len() as u64,
            ty: BO_DEV,
            ..Default::default()
        };
        ioctl(
            self.fd,
            amdxdna::<CreateBo>(CREATE_BO),
            &mut cb,
            "create_bo dev",
        )?;
        let mut gi = GetBoInfo {
            handle: cb.handle,
            ..Default::default()
        };
        ioctl(
            self.fd,
            amdxdna::<GetBoInfo>(GET_BO_INFO),
            &mut gi,
            "get_bo_info dev",
        )?;
        let off = gi
            .xdna_addr
            .checked_sub(self.heap_xdna)
            .ok_or("dev bo below heap")? as usize;
        if off + src.len() > self.heap_len {
            return Err(format!("dev bo {off:#x}+{:#x} outside heap", src.len()));
        }
        let host = unsafe { self.heap_host.add(off) };
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), host, src.len());
            clflush(host, src.len());
        }
        Ok(Bo {
            handle: cb.handle,
            xdna: gi.xdna_addr,
            host,
            len: src.len(),
            mapped: false,
            fd: self.fd,
        })
    }

    fn mapped_bo(&self, len: usize, ty: u32) -> Result<Bo, String> {
        let mut cb = CreateBo {
            size: len as u64,
            ty,
            ..Default::default()
        };
        ioctl(
            self.fd,
            amdxdna::<CreateBo>(CREATE_BO),
            &mut cb,
            "create_bo mapped",
        )?;
        let mut gi = GetBoInfo {
            handle: cb.handle,
            ..Default::default()
        };
        ioctl(
            self.fd,
            amdxdna::<GetBoInfo>(GET_BO_INFO),
            &mut gi,
            "get_bo_info mapped",
        )?;
        let host = unsafe {
            sys::mmap(
                core::ptr::null_mut(),
                len,
                sys::PROT_RW,
                sys::MAP_SHARED,
                self.fd,
                gi.map_offset as i64,
            )
        };
        if host == sys::MAP_FAILED {
            return Err(format!("mmap bo: errno={}", errno()));
        }
        Ok(Bo {
            handle: cb.handle,
            xdna: gi.xdna_addr,
            host: host as *mut u8,
            len,
            mapped: true,
            fd: self.fd,
        })
    }

    /// Host-shared buffer (kernel argument), zero-filled.
    pub fn shmem_bo(&self, len: usize) -> Result<Bo, String> {
        let b = self.mapped_bo(len, BO_SHMEM)?;
        unsafe {
            core::ptr::write_bytes(b.host, 0, len);
            clflush(b.host, len);
        }
        Ok(b)
    }

    pub fn cmd_bo(&self) -> Result<Bo, String> {
        self.mapped_bo(4096, BO_CMD)
    }

    /// Import a dma-buf `fd` as a SHMEM BO of `size` bytes (`size` <= the dma-buf's size). The
    /// caller keeps ownership of `fd` (PRIME_FD_TO_HANDLE takes its own reference); the returned
    /// `Bo` owns the GEM handle and mapping. The mmap is required: under identity SVA the NPU
    /// reaches the buffer through this host VA. Importing a dma-buf this fd's file already
    /// imported yields the same GEM handle, so keep one live `Bo` per dma-buf.
    #[cfg(unix)]
    pub fn import_dmabuf(&self, fd: std::os::fd::RawFd, size: usize) -> std::io::Result<Bo> {
        use std::io::{Error, ErrorKind};
        if size == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "import_dmabuf: size is 0",
            ));
        }
        // dma-buf llseek(SEEK_END, 0) reports the object size; restore the shared file offset.
        let actual = unsafe { sys::lseek(fd, 0, sys::SEEK_END) };
        if actual < 0 {
            return Err(Error::last_os_error());
        }
        if unsafe { sys::lseek(fd, 0, sys::SEEK_SET) } < 0 {
            return Err(Error::last_os_error());
        }
        if size as u64 > actual as u64 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("import_dmabuf: size {size:#x} exceeds dma-buf size {actual:#x}"),
            ));
        }
        let mut ph = PrimeHandle {
            handle: 0,
            flags: 0,
            fd,
        };
        if unsafe {
            sys::ioctl(
                self.fd,
                DRM_IOCTL_PRIME_FD_TO_HANDLE,
                &mut ph as *mut PrimeHandle as *mut c_void,
            )
        } < 0
        {
            return Err(Error::last_os_error());
        }
        // From here `bo` owns the handle: any early return closes it via Drop.
        let mut bo = Bo {
            handle: ph.handle,
            xdna: u64::MAX,
            host: core::ptr::null_mut(),
            len: size,
            mapped: false,
            fd: self.fd,
        };
        let mut gi = GetBoInfo {
            handle: ph.handle,
            ..Default::default()
        };
        if unsafe {
            sys::ioctl(
                self.fd,
                amdxdna::<GetBoInfo>(GET_BO_INFO),
                &mut gi as *mut GetBoInfo as *mut c_void,
            )
        } < 0
        {
            return Err(Error::last_os_error());
        }
        let host = unsafe {
            sys::mmap(
                core::ptr::null_mut(),
                size,
                sys::PROT_RW,
                sys::MAP_SHARED,
                self.fd,
                gi.map_offset as i64,
            )
        };
        if host == sys::MAP_FAILED {
            return Err(Error::last_os_error());
        }
        bo.host = host as *mut u8;
        bo.mapped = true;
        bo.xdna = gi.xdna_addr;
        Ok(bo)
    }

    /// SHMEM BO over caller-owned user pages (`ptr`, `len` page aligned): the driver pins them (`amdxdna_get_ubuf`,
    /// `pin_user_pages_fast(FOLL_WRITE | FOLL_LONGTERM)`) and charges them to `RLIMIT_MEMLOCK` through the process's
    /// pinned_vm (`amdxdna_ubuf.c:127-133`, `ENOMEM` above the limit); see [`ensure_pin_headroom`].
    /// It imports the resulting dma-buf (`amdxdna_drm_va_tbl` with one entry). `Bo::host` is the caller's own mapping
    /// (`ptr`, unowned: Drop does not munmap it); under identity SVA the NPU reaches the buffer through that VA. The pages
    /// must stay mapped until the `Bo` is dropped. The BO is never mmapped: amdxdna's ubuf dma-buf has no mmap since Linux
    /// 7.1 ("accel/amdxdna: Remove mmap and export support for ubuf"), so every attempt failed `-EINVAL` and logged
    /// `amdxdna_insert_pages: Failed to mmap dma buf -22` / `amdxdna_gem_obj_mmap: Failed insert pages, ret -22`.
    pub fn userptr_bo(&self, ptr: *mut u8, len: usize) -> Result<Bo, String> {
        if ptr.is_null() || len == 0 || (ptr as usize) % 4096 != 0 || len % 4096 != 0 {
            return Err(format!(
                "userptr_bo: {ptr:p} + {len:#x} must be non-empty and page aligned"
            ));
        }
        // struct amdxdna_drm_va_tbl { s32 dmabuf_fd; u32 num_entries; { u64 vaddr; u64 len } va_entries[1] }
        #[repr(C)]
        struct VaTbl {
            dmabuf_fd: i32,
            num_entries: u32,
            vaddr: u64,
            len: u64,
        }
        let tbl = VaTbl {
            dmabuf_fd: -1,
            num_entries: 1,
            vaddr: ptr as u64,
            len: len as u64,
        };
        let mut cb = CreateBo {
            vaddr: &tbl as *const VaTbl as u64,
            size: len as u64,
            ty: BO_SHMEM,
            ..Default::default()
        };
        ioctl(
            self.fd,
            amdxdna::<CreateBo>(CREATE_BO),
            &mut cb,
            "create_bo userptr",
        )?;
        // From here `bo` owns the handle: any early return closes it via Drop.
        let mut bo = Bo {
            handle: cb.handle,
            xdna: u64::MAX,
            host: core::ptr::null_mut(),
            len,
            mapped: false,
            fd: self.fd,
        };
        let mut gi = GetBoInfo {
            handle: cb.handle,
            ..Default::default()
        };
        ioctl(
            self.fd,
            amdxdna::<GetBoInfo>(GET_BO_INFO),
            &mut gi,
            "get_bo_info userptr",
        )?;
        bo.host = ptr;
        bo.xdna = gi.xdna_addr;
        Ok(bo)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            if !self.heap_host.is_null() {
                sys::munmap(self.heap_host as *mut c_void, self.heap_len);
                let mut c = [self.heap_handle, 0u32];
                sys::ioctl(self.fd, DRM_IOCTL_GEM_CLOSE, c.as_mut_ptr() as *mut c_void);
            }
            sys::close(self.fd);
        }
    }
}

/// `RLIMIT_MEMLOCK` headroom for `need` more bytes of driver-pinned userptr pages ([`Device::userptr_bo`]).
///
/// The driver admits a userptr BO only while pinned_vm (`VmPin`) plus the new pages stays within the soft limit,
/// unless the process has CAP_IPC_LOCK. When the soft limit is short and the hard limit allows, it is raised to
/// exactly `VmPin + need`. Otherwise this fails closed and names the bytes and the host fix.
pub fn ensure_pin_headroom(need: usize) -> Result<(), String> {
    let status = std::fs::read_to_string("/proc/self/status")
        .map_err(|e| format!("read /proc/self/status: {e}"))?;
    let field = |key: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .map(str::trim)
    };
    let cap_eff = field("CapEff:")
        .and_then(|v| u64::from_str_radix(v, 16).ok())
        .ok_or("CapEff missing from /proc/self/status")?;
    if cap_eff & (1 << CAP_IPC_LOCK) != 0 {
        return Ok(());
    }
    let pinned = field("VmPin:")
        .and_then(|v| v.strip_suffix("kB"))
        .and_then(|v| v.trim().parse::<u64>().ok())
        .ok_or("VmPin missing from /proc/self/status")?
        * 1024;
    let mut lim = [0u64; 2];
    if unsafe { sys::getrlimit(sys::RLIMIT_MEMLOCK, &mut lim) } != 0 {
        return Err(format!("getrlimit(RLIMIT_MEMLOCK): errno={}", errno()));
    }
    let required = pinned + need as u64;
    if lim[0] >= required {
        return Ok(());
    }
    if lim[1] < required {
        return Err(pin_limit_error(need as u64, pinned, lim[1]));
    }
    let raised = [required, lim[1]];
    if unsafe { sys::setrlimit(sys::RLIMIT_MEMLOCK, &raised) } != 0 {
        return Err(format!(
            "setrlimit(RLIMIT_MEMLOCK soft {} -> {required} B, hard {}): errno={}",
            lim[0],
            lim[1],
            errno()
        ));
    }
    eprintln!("npu: RLIMIT_MEMLOCK soft limit raised {} -> {required} B for {need} B of userptr pins (hard limit {} B)", lim[0], lim[1]);
    Ok(())
}

const CAP_IPC_LOCK: u32 = 14;

/// Host limit for `required` pinned bytes plus 25% margin, rounded up to the MiB: a bounded value, not `infinity`.
pub fn bounded_memlock_bytes(required: u64) -> u64 {
    (required + required.div_ceil(4)).div_ceil(1 << 20) << 20
}

fn pin_limit_error(need: u64, pinned: u64, hard: u64) -> String {
    let required = pinned + need;
    let bounded = bounded_memlock_bytes(required);
    format!("RLIMIT_MEMLOCK hard limit {hard} B is below the {required} B the NPU needs to pin ({need} B of userptr \
        arenas + {pinned} B already pinned); raise the memlock limit of the hipfire process to {bounded} B ({} MiB = \
        required x1.25): systemd `LimitMEMLOCK={bounded}`, shell `ulimit -l {}` (KiB) or limits.d `memlock {}`; see \
        docs/npu/npu-trunk.md#host-memlock-limit", bounded >> 20, bounded >> 10, bounded >> 10)
}

/// Declared hot-path patch locations, indexed in u32 words from each BO's start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchBo {
    Cmd(usize),
    Chain,
    Insts,
}
#[derive(Debug, Clone, Copy)]
pub struct PatchSite {
    pub bo: PatchBo,
    pub word: usize,
    pub what: &'static str,
}

fn encode_start_raw(words: &mut [u32; 17], inst_addr: u64, inst_bytes: usize, args: &[u64]) {
    assert!(args.len() <= 5);
    *words = [0; 17];
    words[0] = ERT_STATE_NEW | (16 << 12) | (ERT_START_CU << 23) | (ERT_CU << 28);
    words[1] = 1;
    words[2] = 3;
    words[4] = inst_addr as u32;
    words[5] = (inst_addr >> 32) as u32;
    words[6] = inst_bytes as u32;
    for (i, addr) in args.iter().enumerate() {
        words[7 + 2 * i] = *addr as u32;
        words[8 + 2 * i] = (*addr >> 32) as u32;
    }
}

/// Pure START_CU encoding; preserves the eager path's use of the instruction DEV address.
pub fn encode_start_cu(words: &mut [u32; 17], insts: &Bo, args: &[&Bo]) {
    assert!(args.len() <= 5);
    let mut addresses = [0u64; 5];
    for (i, arg) in args.iter().enumerate() {
        addresses[i] = arg.dev_addr();
    }
    encode_start_raw(words, insts.xdna, insts.len, &addresses[..args.len()]);
}

fn write_words(bo: &Bo, words: &[u32]) {
    assert!(words.len() * 4 <= bo.len);
    unsafe {
        for (i, word) in words.iter().enumerate() {
            core::ptr::write_volatile((bo.host as *mut u32).add(i), *word);
        }
        clflush(bo.host, words.len() * 4);
    }
}

fn snapshot(bo: &Bo, words: &mut [u32]) {
    unsafe {
        clflush(bo.host, words.len() * 4);
        for (i, word) in words.iter_mut().enumerate() {
            *word = core::ptr::read_volatile((bo.host as *const u32).add(i));
        }
    }
}

/// Retained command. Operand/instruction BOs must outlive submission and its completion.
pub struct Prepared {
    pub cmd: Bo,
    handles: [u32; 6],
    addresses: [u64; 5],
    arg_count: usize,
    inst_addr: u64,
}

impl Prepared {
    pub fn new(dev: &Device, insts: &Bo, args: &[&Bo]) -> Result<Self, String> {
        if args.len() > 5 {
            return Err("START_CU supports at most five arguments".into());
        }
        let cmd = dev.cmd_bo()?;
        let mut words = [0; 17];
        encode_start_cu(&mut words, insts, args);
        write_words(&cmd, &words);
        let mut handles = [0; 6];
        let mut addresses = [0; 5];
        handles[0] = insts.handle;
        for (i, arg) in args.iter().enumerate() {
            handles[i + 1] = arg.handle;
            addresses[i] = arg.dev_addr();
        }
        Ok(Self {
            cmd,
            handles,
            addresses,
            arg_count: args.len(),
            inst_addr: insts.xdna,
        })
    }

    pub fn patch_arg(&mut self, i: usize, bo: &Bo) -> bool {
        assert!(i < self.arg_count);
        let addr = bo.dev_addr();
        if self.addresses[i] == addr && self.handles[i + 1] == bo.handle {
            return false;
        }
        self.addresses[i] = addr;
        self.handles[i + 1] = bo.handle;
        unsafe {
            let p = (self.cmd.host as *mut u32).add(7 + 2 * i);
            core::ptr::write_volatile(p, addr as u32);
            core::ptr::write_volatile(p.add(1), (addr >> 32) as u32);
            // A pair can straddle the 64-byte boundary (argument 4); flush only touched lines.
            clflush(p as *const u8, 8);
        }
        true
    }

    /// No CPU header patch: aie2_ctx.c's job run sets NEW itself before dispatch.
    pub fn reset_state(&mut self) {}

    pub fn patch_sites(&self) -> Vec<PatchSite> {
        (0..self.arg_count)
            .flat_map(|i| {
                [
                    PatchSite {
                        bo: PatchBo::Cmd(0),
                        word: 7 + 2 * i,
                        what: "arg address low",
                    },
                    PatchSite {
                        bo: PatchBo::Cmd(0),
                        word: 8 + 2 * i,
                        what: "arg address high",
                    },
                ]
            })
            .collect()
    }

    pub fn words(&self) -> [u32; 17] {
        let mut words = [0; 17];
        snapshot(&self.cmd, &mut words);
        words
    }

    pub fn state(&self) -> u32 {
        self.words()[0] & 15
    }
    pub fn inst_addr(&self) -> u64 {
        self.inst_addr
    }
}

fn encode_chain(handles: &[u32]) -> Result<Vec<u32>, String> {
    let n = handles.len();
    if !(1..=36).contains(&n) {
        return Err("CHAIN requires 1..=36 START_CU commands".into());
    }
    let count = 6 + 2 * n;
    let mut words = vec![0; count + 1];
    words[0] = ERT_STATE_NEW | ((count as u32) << 12) | (19 << 23) | (ERT_CU << 28);
    words[1] = n as u32;
    for (i, handle) in handles.iter().enumerate() {
        words[7 + 2 * i] = *handle;
    }
    Ok(words)
}

fn dedup_handles(handles: impl Iterator<Item = u32>) -> Vec<u32> {
    let mut result = Vec::new();
    for handle in handles {
        if !result.contains(&handle) {
            result.push(handle);
        }
    }
    result
}

/// Immutable chain payload: only the driver changes its state/submit/error fields.
pub struct Chain {
    cmd: Bo,
    commands: Vec<u32>,
    handles: Vec<u32>,
    source_handles: Vec<u32>,
}

impl Chain {
    pub fn new(dev: &Device, commands: &[&Prepared]) -> Result<Self, String> {
        let command_handles: Vec<u32> = commands.iter().map(|p| p.cmd.handle).collect();
        let words = encode_chain(&command_handles)?;
        let source_handles: Vec<u32> = commands
            .iter()
            .flat_map(|p| p.handles[..p.arg_count + 1].iter().copied())
            .collect();
        let handles = dedup_handles(source_handles.iter().copied());
        let cmd = dev.cmd_bo()?;
        write_words(&cmd, &words);
        Ok(Self {
            cmd,
            commands: command_handles,
            handles,
            source_handles,
        })
    }

    /// Same retained commands, in the same order; rebuild pins only after handle changes.
    pub fn refresh(&mut self, commands: &[&Prepared]) {
        assert_eq!(commands.len(), self.commands.len());
        let mut changed = false;
        let mut slot = 0;
        for (i, p) in commands.iter().enumerate() {
            assert_eq!(p.cmd.handle, self.commands[i]);
            for handle in &p.handles[..p.arg_count + 1] {
                assert!(slot < self.source_handles.len());
                changed |= self.source_handles[slot] != *handle;
                self.source_handles[slot] = *handle;
                slot += 1;
            }
        }
        assert_eq!(slot, self.source_handles.len());
        if changed {
            self.handles.clear();
            for handle in &self.source_handles {
                if !self.handles.contains(handle) {
                    self.handles.push(*handle);
                }
            }
        }
    }

    /// No chain BO word is CPU-patched; driver-owned status words are not patch sites.
    pub fn patch_sites(&self) -> Vec<PatchSite> {
        Vec::new()
    }
    pub fn words(&self) -> Vec<u32> {
        let mut words = vec![0; 7 + 2 * self.commands.len()];
        snapshot(&self.cmd, &mut words);
        words
    }
    pub fn error_index(&self) -> u32 {
        unsafe {
            clflush(self.cmd.host, 64);
            core::ptr::read_volatile((self.cmd.host as *const u32).add(3))
        }
    }
}

/// A hardware context bound to one PDI (CU 0).
pub struct HwCtx<'d> {
    dev: &'d Device,
    pub handle: u32,
    pub syncobj: u32,
}

impl<'d> HwCtx<'d> {
    /// `num_tiles` = columns × core rows requested from the partition solver.
    pub fn create(dev: &'d Device, num_tiles: u32, max_opc: u32) -> Result<HwCtx<'d>, String> {
        let qos = QosInfo::default();
        let mut c = CreateHwctx {
            qos_p: &qos as *const _ as u64,
            max_opc,
            num_tiles,
            ..Default::default()
        };
        ioctl(
            dev.fd,
            amdxdna::<CreateHwctx>(CREATE_HWCTX),
            &mut c,
            "create_hwctx",
        )?;
        Ok(HwCtx {
            dev,
            handle: c.handle,
            syncobj: c.syncobj_handle,
        })
    }

    /// Load the PDI (already copied into a DEV BO) as CU 0, function 0.
    pub fn config_cu(&self, pdi: &Bo) -> Result<(), String> {
        let mut cu = ConfigCu {
            num_cus: 1,
            cu_bo: pdi.handle,
            cu_func: 0,
            ..Default::default()
        };
        let mut c = ConfigHwctx {
            handle: self.handle,
            param_type: 0,
            param_val: &mut cu as *mut _ as u64,
            param_val_size: size_of::<ConfigCu>() as u32,
            pad: 0,
        };
        ioctl(
            self.dev.fd,
            amdxdna::<ConfigHwctx>(CONFIG_HWCTX),
            &mut c,
            "config_hwctx cu",
        )
    }

    /// Submit one ERT_START_CU to CU 0 running the DPU transaction stream in `insts` with up to
    /// five buffer arguments. Payload words (copied verbatim to firmware by aie2_init_exec_cu_req):
    /// cu_mask, opcode u64 = 3 (transaction), insts device addr u64, insts size in bytes u32,
    /// bo0..bo4 addresses u64. Returns the timeline point to wait on.
    pub fn submit(&self, cmd: &mut Bo, insts: &Bo, args: &[&Bo]) -> Result<u64, String> {
        let mut encoded = [0u32; 17];
        encode_start_cu(&mut encoded, insts, args);
        write_words(cmd, &encoded);
        let mut handles = [0u32; 6];
        handles[0] = insts.handle;
        for (i, arg) in args.iter().enumerate() {
            handles[i + 1] = arg.handle;
        }
        let mut ec = ExecCmd {
            hwctx: self.handle,
            ty: 0,
            cmd_handles: cmd.handle as u64,
            args: handles.as_ptr() as u64,
            cmd_count: 1,
            arg_count: (args.len() + 1) as u32,
            ..Default::default()
        };
        ioctl(
            self.dev.fd,
            amdxdna::<ExecCmd>(EXEC_CMD),
            &mut ec,
            "exec_cmd",
        )?;
        Ok(ec.seq)
    }

    pub fn submit_prepared(&self, cmd: &Prepared) -> Result<u64, String> {
        self.submit_handles(&cmd.cmd, &cmd.handles[..cmd.arg_count + 1])
    }

    pub fn submit_chain(&self, chain: &Chain) -> Result<u64, String> {
        self.submit_handles(&chain.cmd, &chain.handles)
    }

    fn submit_handles(&self, cmd: &Bo, handles: &[u32]) -> Result<u64, String> {
        let mut ec = ExecCmd {
            hwctx: self.handle,
            cmd_handles: cmd.handle as u64,
            args: handles.as_ptr() as u64,
            cmd_count: 1,
            arg_count: handles.len() as u32,
            ..Default::default()
        };
        ioctl(
            self.dev.fd,
            amdxdna::<ExecCmd>(EXEC_CMD),
            &mut ec,
            "exec_cmd",
        )?;
        Ok(ec.seq)
    }

    pub fn wait_chain(&self, chain: &Chain, seq: u64, timeout_ms: u64) -> Result<u32, String> {
        self.wait(&chain.cmd, seq, timeout_ms)
    }

    /// Wait for timeline point `seq`, then return the ERT state word's state field.
    pub fn wait(&self, cmd: &Bo, seq: u64, timeout_ms: u64) -> Result<u32, String> {
        let mut ts = [0i64; 2];
        unsafe { sys::clock_gettime(sys::CLOCK_MONOTONIC, &mut ts) };
        let deadline = ts[0] * 1_000_000_000 + ts[1] + (timeout_ms as i64) * 1_000_000;
        let handles = [self.syncobj];
        let points = [seq];
        let mut w = SyncobjTimelineWait {
            handles: handles.as_ptr() as u64,
            points: points.as_ptr() as u64,
            timeout_nsec: deadline,
            count_handles: 1,
            flags: SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
            ..Default::default()
        };
        ioctl(
            self.dev.fd,
            DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT,
            &mut w,
            "syncobj_timeline_wait",
        )?;
        unsafe {
            clflush(cmd.host, 64);
            Ok(core::ptr::read_volatile(cmd.host as *const u32) & 0xF)
        }
    }
}

impl Drop for HwCtx<'_> {
    fn drop(&mut self) {
        let mut s = [self.syncobj, 0u32];
        unsafe {
            sys::ioctl(
                self.dev.fd,
                DRM_IOCTL_SYNCOBJ_DESTROY,
                s.as_mut_ptr() as *mut c_void,
            )
        };
        let mut d = DestroyHwctx {
            handle: self.handle,
            pad: 0,
        };
        let _ = ioctl(
            self.dev.fd,
            amdxdna::<DestroyHwctx>(DESTROY_HWCTX),
            &mut d,
            "destroy_hwctx",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memlock_bound_is_required_plus_quarter_rounded_to_mib() {
        // The shipped Flash-Next trunk: 2 banks x (A 8192x2880, C GdnIn + QsaIn, three 4 KiB placeholders).
        assert_eq!(bounded_memlock_bytes(1_053_843_456), 1_318_060_032);
        assert_eq!(bounded_memlock_bytes(4 << 20), 5 << 20);
        let e = pin_limit_error(1_053_843_456, 0, 8 << 20);
        assert!(
            e.contains("hard limit 8388608 B is below the 1053843456 B"),
            "{e}"
        );
        assert!(
            e.contains("LimitMEMLOCK=1318060032") && e.contains("ulimit -l 1287168"),
            "{e}"
        );
    }

    #[test]
    fn start_cu_matches_hand_computed_layout() {
        let mut words = [u32::MAX; 17];
        encode_start_raw(
            &mut words,
            0x1234_5678_9abc_def0,
            4096,
            &[
                0x1111_2222_3333_4444,
                0x5555_6666_7777_8888,
                0x9999_aaaa_bbbb_cccc,
            ],
        );
        assert_eq!(
            words,
            [
                0x3001_0001,
                1,
                3,
                0,
                0x9abc_def0,
                0x1234_5678,
                4096,
                0x3333_4444,
                0x1111_2222,
                0x7777_8888,
                0x5555_6666,
                0xbbbb_cccc,
                0x9999_aaaa,
                0,
                0,
                0,
                0,
            ]
        );
    }

    #[test]
    fn chain_driver_masks_and_payload_bounds() {
        for n in [1, 2, 36] {
            let handles: Vec<u32> = (1..=n).collect();
            let words = encode_chain(&handles).unwrap();
            // Driver GENMASK(22,12), GENMASK(27,23), GENMASK(11,10).
            let count = (words[0] & 0x007f_f000) >> 12;
            assert_eq!(count, 6 + 2 * n);
            assert_eq!((words[0] & 0x0f80_0000) >> 23, 19);
            assert_eq!(words[0] & 0x0000_0c00, 0);
            assert_eq!(words[0] & 15, ERT_STATE_NEW);
            assert_eq!(words[0] >> 28, ERT_CU);
            assert!(count * 4 >= 24 + 8 * n);
            assert_eq!(words.len(), count as usize + 1);
            assert_eq!(words[1], n);
            assert_eq!(&words[2..7], &[0; 5]);
            for (i, handle) in handles.iter().enumerate() {
                assert_eq!(words[7 + 2 * i], *handle);
                assert_eq!(words[8 + 2 * i], 0);
            }
        }
        assert!(encode_chain(&[0; 37]).is_err());
        assert!(encode_chain(&[]).is_err());
    }

    #[test]
    fn chain_pins_each_unique_handle_once() {
        assert_eq!(
            dedup_handles([9, 1, 2, 9, 2, 3, 9, 1].into_iter()),
            vec![9, 1, 2, 3]
        );
    }
}
