// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `va-bridge`: drive the VCN JPEG engine through libva directly.
//!
//! Product decode path for VL images (`image.decode = vcn|auto`). Pipeline
//! per frame: [`VaSession::decode_jpeg`] (parse → pooled VA surface/context
//! → five VA buffers → `vaBegin/Render/EndPicture` → `vaSyncSurface` →
//! `vaExportSurfaceHandle`, exported + imported once per key) then the
//! `vl_yuv_preprocess` kernel consumes the pooled device pointer.
//!
//! Allocation is always `DRM_FORMAT_MOD_LINEAR`, so the exported dma-buf is
//! kernel-readable (0.000 LSB vs derived readback on
//! gfx1010/1030/1100/1151/1201, 2026-09-07). There is no tiled/derived
//! fallback: the derived-pixel path was deleted with the investigation (see
//! `experiment/vcn-jpeg`); [`VaSession::decode_jpeg_planes`] remains as the
//! `vaDeriveImage` oracle for `vl_vcn_444check`.
//!
//! [`DecodeOutcome`]: explicit-session contract — `Decoded` carries the
//! pooled frame, `Unsupported` (progressive/arithmetic/12-bit/CMYK) means
//! "take the CPU path" with NO session teardown; `Err` is real failures
//! only. The shared session instead returns [`SharedDecodeOutcome`], whose
//! [`SharedVcnLease`] holds the pool mutex until the consumer's device reads
//! complete, so a same-key decode cannot overwrite the surface mid-read.

mod ffi;
mod interop;
mod jpeg;
pub use ffi::{
    VaBufferId, VaConfigId, VaContextId, VaDisplay, VaDrmPrimeDescriptor, VaDrmPrimeLayer,
    VaDrmPrimeObject, VaError, VaJpegHuffmanBuffer, VaJpegIQMatrix, VaJpegPicParam,
    VaJpegSliceParam, VaLib, VaSurfaceId, VA_EXPORT_SURFACE_READ_ONLY,
    VA_EXPORT_SURFACE_SEPARATE_LAYERS, VA_FOURCC_NV12, VA_MEM_TYPE_DRM_PRIME_2,
    VA_STATUS_SUCCESS,
};
pub use interop::HipMapping;
pub use jpeg::{parse_for_va, JpegVaParams};

use ffi::{VA_ENTRYPOINT_VLD, VA_PROFILE_JPEG_BASELINE};
use std::collections::HashMap;
use std::ffi::c_void;
// Unix-only imports: VA-API/DRM render nodes, fd ownership, and PCI/sysfs
// probing have no backend off Unix. Non-Unix builds compile the public API
// against stubs below and every caller takes the CPU path.
#[cfg(unix)]
use std::ffi::{c_char, c_int};
#[cfg(unix)]
use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// DRM render-node candidates, in order: `HIPFIRE_VCN_DRM_NODE` (via
/// `developer_var`) wins; otherwise the node whose PCI slot matches HIP
/// device 0; otherwise every `renderD*` node, sorted (the planes oracle
/// needs no HIP runtime, so a missing HIP library still gets a scan).
#[cfg(unix)]
fn candidate_nodes() -> Vec<String> {
    if let Ok(one) = hipfire_config::developer_var("HIPFIRE_VCN_DRM_NODE") {
        return vec![one];
    }
    if let Some(pci) = hip_pci_bus_id(0) {
        if let Some(node) = render_node_for_pci(&pci) {
            return vec![node];
        }
    }
    sysfs_render_nodes()
}

/// Every DRM render node, sorted by name; the hardcoded 128..132 fallback
/// only fires when sysfs is unreadable.
#[cfg(unix)]
fn sysfs_render_nodes() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/sys/class/drm") {
        for e in rd.flatten() {
            let name = e.file_name();
            let Some(s) = name.to_str() else { continue };
            if s.starts_with("renderD") {
                out.push(format!("/dev/dri/{s}"));
            }
        }
    }
    out.sort();
    if out.is_empty() {
        (128..132).map(|i| format!("/dev/dri/renderD{i}")).collect()
    } else {
        out
    }
}

/// PCI bus id (`0000:03:00.0` form) of a HIP device, dlopen'd from the same
/// `libamdhip64` instance `hip-bridge` uses. `va-bridge` must not depend on
/// `hip-bridge` (same layer), so the one symbol is resolved locally.
#[cfg(unix)]
fn hip_pci_bus_id(device: i32) -> Option<String> {
    let candidates =
        hipfire_config::rocm::library_candidates(hipfire_config::rocm::HIP_RUNTIME_LIBRARIES);
    for c in &candidates {
        // SAFETY: system HIP runtime; no Rust invariants involved.
        let lib = unsafe { libloading::Library::new(c) }.ok()?;
        // SAFETY: signature matches hip_runtime_api.h
        // (`hipError_t hipDeviceGetPCIBusId(char*, int, int)`).
        let bus = unsafe {
            let f: libloading::Symbol<unsafe extern "C" fn(*mut c_char, c_int, c_int) -> u32> =
                lib.get(b"hipDeviceGetPCIBusId").ok()?;
            let mut buf = [0 as c_char; 64];
            let code = f(buf.as_mut_ptr(), buf.len() as c_int, device as c_int);
            if code != 0 {
                continue;
            }
            let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            buf[..len]
                .iter()
                .map(|&b| b as u8 as char)
                .collect::<String>()
        };
        return Some(bus);
    }
    None
}

/// `/dev/dri/<node>` whose `device/uevent` `PCI_SLOT_NAME` matches `pci`
/// (case-insensitive; both are `domain:bus:device.function`).
#[cfg(unix)]
fn render_node_for_pci(pci: &str) -> Option<String> {
    let want = pci.trim().to_ascii_lowercase();
    let rd = std::fs::read_dir("/sys/class/drm").ok()?;
    let mut names: Vec<String> = Vec::new();
    for e in rd.flatten() {
        let name = e.file_name();
        let Some(s) = name.to_str() else { continue };
        if s.starts_with("renderD") {
            names.push(s.to_string());
        }
    }
    names.sort();
    for n in names {
        let uevent = std::fs::read_to_string(format!("/sys/class/drm/{n}/device/uevent")).ok()?;
        for line in uevent.lines() {
            if let Some(slot) = line.strip_prefix("PCI_SLOT_NAME=") {
                if slot.trim().to_ascii_lowercase() == want {
                    return Some(format!("/dev/dri/{n}"));
                }
            }
        }
    }
    None
}

/// An initialised VA display with a JPEG-baseline VLD config.
///
/// Surfaces, decode contexts, and HIP imports are pooled by
/// (`rt_format`, width, height): the first decode of a key allocates
/// (always `DRM_FORMAT_MOD_LINEAR`), exports + imports once, and every later
/// decode of that key costs only parse + submit + sync. The pooled device
/// pointer is stable per key; its *contents* reflect the most recent decode
/// of that key. Explicit-session owners serialize on `&mut`; shared-session
/// consumers hold a [`SharedVcnLease`] instead, which blocks the next shared
/// decode until their reads complete.
pub struct VaSession {
    lib: VaLib,
    dpy: VaDisplay,
    /// DRM fd backing `dpy`. Unix-only (see above); non-Unix never opens a
    /// session, so the field is absent there and `open` fails closed.
    #[cfg(unix)]
    _drm_fd: OwnedFd,
    vendor: String,
    config: VaConfigId,
    node: String,
    pool: HashMap<PoolKey, PooledEntry>,
    /// Quarantine flag for the process-wide shared session (see
    /// [`VaSession::quarantine_shared`]): set under the already-held pool
    /// mutex when a consumer's terminal sync fails, while possibly-live GPU
    /// reads are still outstanding. A poisoned session is retained — never
    /// decoded through or torn down — and every later shared decode fails
    /// closed to the CPU fallback. Explicit-session owners never set this.
    poisoned: bool,
}

/// Pool key: the VA render-target format (from the SOF0 sampling factors)
/// plus frame geometry.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PoolKey {
    rt_format: u32,
    width: u32,
    height: u32,
}

/// The dma-buf export + HIP import cached per pool key. The import binds the
/// surface's backing pages, not their contents, so it survives re-decode.
struct CachedExport {
    max_h: u8,
    max_v: u8,
    fourcc: u32,
    layers: [VaDrmPrimeLayer; 4],
    num_layers: u32,
    mapping: HipMapping,
}

/// One pooled VA surface + decode context. Destroyed with the session.
struct PooledEntry {
    surf: VaSurfaceId,
    ctx: VaContextId,
    export: Option<CachedExport>,
}

/// Cached [`VaSession::probe`] failure: the first failed probe disables later
/// ones for the process (VA init failure is environmental, not transient).
static PROBE_FAILED: OnceLock<()> = OnceLock::new();

/// Process-wide pooled VCN session for the product decode path. Module scope
/// (not function-local) so [`VaSession::shared_decode_jpeg_lease`] and
/// [`VaSession::try_shared_decode_jpeg`] observe one pool: two function-local
/// statics would be two sessions and the lease exclusion would be fiction.
static SHARED: std::sync::LazyLock<Mutex<Option<VaSession>>> =
    std::sync::LazyLock::new(|| Mutex::new(None));

// The VA display handle is a raw pointer, so `VaSession` is `!Send` by
// default. Sharing is explicitly single-threaded behind the caller's
// `&mut` (or the shared-session lease); cross-thread transfer of the
// session itself is the caller's responsibility, as with `HipMapping`.
unsafe impl Send for VaSession {}

impl VaSession {
    /// Open the first working render-node candidate, initialise VA, create
    /// the JPEG-baseline VLD config. See [`candidate_nodes`] for selection.
    pub fn open() -> Result<Self, VaError> {
        Self::open_profile(VA_PROFILE_JPEG_BASELINE)
    }

    /// Open the first working render node with a VLD config for `profile`
    /// (the image path only ever requests JPEG baseline).
    #[cfg(unix)]
    pub(crate) fn open_profile(profile: i32) -> Result<Self, VaError> {
        let mut lib = Some(VaLib::load()?);
        let mut last_err = VaError::NoRenderNode;
        for node in candidate_nodes() {
            let l = match lib.take() {
                Some(l) => l,
                None => break,
            };
            match Self::open_node(l, &node, profile) {
                Ok(s) => return Ok(s),
                Err((l, e)) => {
                    lib = Some(l);
                    last_err = e;
                }
            }
        }
        Err(last_err)
    }
    /// Non-Unix stub: VA-API/DRM has no backend here, so session open fails
    /// closed; every caller maps this to the CPU path (never a build failure).
    #[cfg(not(unix))]
    pub(crate) fn open_profile(_profile: i32) -> Result<Self, VaError> {
        Err(VaError::NoRenderNode)
    }

    #[cfg(unix)]
    fn open_node(lib: VaLib, node: &str, profile: i32) -> Result<Self, (VaLib, VaError)> {
        // O_RDWR: radeonsi winsys creation (GEM backing) fails on O_RDONLY
        // with EACCES (`amdgpu_bo_cpu_map failed (-13)`), which surfaces as
        // vaInitialize succeeding but context creation segfaulting.
        let fd = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(node)
        {
            Ok(f) => f,
            Err(_) => return Err((lib, VaError::NoRenderNode)),
        };
        let owned: OwnedFd = fd.into();
        let dpy = unsafe { (lib.va_get_display_drm)(owned.as_raw_fd()) };
        if dpy.is_null() {
            return Err((lib, VaError::Corrupt("vaGetDisplayDRM returned NULL")));
        }
        let mut major = 0i32;
        let mut minor = 0i32;
        // SAFETY: out-params are valid i32 slots.
        let st = unsafe { (lib.va_initialize)(dpy, &mut major, &mut minor) };
        if st != VA_STATUS_SUCCESS {
            let e = VaError::Status {
                op: "vaInitialize",
                code: st,
                msg: lib.error_str(st),
            };
            return Err((lib, e));
        }
        // SAFETY: static vendor string.
        let vendor = unsafe {
            let p = (lib.va_query_vendor)(dpy);
            if p.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        // Requested profile advertised?
        let max = unsafe { (lib.va_max_profiles)(dpy) }.max(0) as usize;
        let mut profiles = vec![0i32; max.min(64).max(1)];
        let mut n = profiles.len() as i32;
        let st = unsafe { (lib.va_query_profiles)(dpy, profiles.as_mut_ptr(), &mut n) };
        if st != VA_STATUS_SUCCESS {
            unsafe {
                (lib.va_terminate)(dpy);
            }
            let e = VaError::Status {
                op: "vaQueryConfigProfiles",
                code: st,
                msg: lib.error_str(st),
            };
            return Err((lib, e));
        }
        if !profiles[..n.max(0) as usize].contains(&profile) {
            unsafe {
                (lib.va_terminate)(dpy);
            }
            return Err((
                lib,
                VaError::Unsupported("driver lacks the requested VA profile"),
            ));
        }
        let mut config = 0;
        let st = unsafe {
            (lib.va_create_config)(
                dpy,
                profile,
                VA_ENTRYPOINT_VLD,
                std::ptr::null_mut(),
                0,
                &mut config,
            )
        };
        if st != VA_STATUS_SUCCESS {
            unsafe {
                (lib.va_terminate)(dpy);
            }
            let e = VaError::Status {
                op: "vaCreateConfig(VLD)",
                code: st,
                msg: lib.error_str(st),
            };
            return Err((lib, e));
        }
        Ok(Self {
            lib,
            dpy,
            _drm_fd: owned,
            vendor,
            config,
            node: node.to_string(),
            pool: HashMap::new(),
            poisoned: false,
        })
    }

    pub fn vendor(&self) -> &str {
        &self.vendor
    }

    /// The DRM render node this session decodes on (PCI-matched to HIP
    /// device 0 unless `HIPFIRE_VCN_DRM_NODE` overrode selection).
    pub fn node(&self) -> &str {
        &self.node
    }

    /// Open a session when VCN JPEG is available, else `None`. A failure is
    /// cached in a [`OnceLock`] so repeated probes never spam dlopen/VA
    /// init; a success always opens fresh (the caller holds the session and
    /// its pool — see [`VaSession::shared_decode_jpeg_lease`] for the
    /// process-wide pooled session the product path uses).
    pub fn probe() -> Option<VaSession> {
        if PROBE_FAILED.get().is_some() {
            return None;
        }
        match Self::open() {
            Ok(s) => Some(s),
            Err(_) => {
                let _ = PROBE_FAILED.set(());
                None
            }
        }
    }

    /// Process-wide pooled session for the product decode path. Opens on
    /// first use; `Err(NoRenderNode | Dlopen | MissingSymbol)` means "no
    /// VCN — take the CPU path", any other `Err` is a real failure.
    /// `Ok(Unsupported)` is a per-stream CPU fallback with no teardown.
    ///
    /// The decoded frame is leased, not copied: the returned
    /// [`SharedVcnLease`] holds the session mutex until the consumer's
    /// device reads complete (see its release-boundary docs). A `Copy` frame
    /// here previously let a same-key decode on another thread overwrite the
    /// surface mid-read; no unguarded shared-frame result escapes anymore.
    pub fn shared_decode_jpeg_lease(jpeg: &[u8]) -> Result<SharedDecodeOutcome<'static>, VaError> {
        let mut guard = SHARED.lock().unwrap_or_else(|e| e.into_inner());
        // Quarantined (see `quarantine_shared`): fail closed to the CPU path
        // without waiting — the pooled surface may still be under hung reads.
        if guard.as_ref().is_some_and(|s| s.poisoned) {
            return Err(VaError::Status {
                op: "shared VCN session",
                code: -1,
                msg: "shared VCN session quarantined after terminal sync failure".to_string(),
            });
        }
        if guard.is_none() {
            *guard = Some(VaSession::open()?);
        }
        match guard
            .as_mut()
            .expect("pooled VCN session just opened")
            .decode_jpeg(jpeg)
        {
            Ok(DecodeOutcome::Decoded(frame)) => Ok(SharedDecodeOutcome::Decoded(SharedVcnLease {
                _guard: guard,
                frame,
            })),
            Ok(DecodeOutcome::Unsupported(reason)) => Ok(SharedDecodeOutcome::Unsupported(reason)),
            Err(e) => Err(e),
        }
    }

    /// Non-blocking variant of [`Self::shared_decode_jpeg_lease`]: `Ok(None)`
    /// means the shared session is currently leased to another consumer
    /// (their surface may still be mid-read — assume nothing about contents).
    /// Lock poisoning is recovered like the blocking path. Exists for
    /// contention probes (`vl_vcn_lease_race`); the product path takes the
    /// blocking lease.
    pub fn try_shared_decode_jpeg(
        jpeg: &[u8],
    ) -> Result<Option<SharedDecodeOutcome<'static>>, VaError> {
        let mut guard = match SHARED.try_lock() {
            Ok(g) => g,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        };
        // Quarantined: an error, not contention — `Ok(None)` would spin the
        // probe's exclusion check forever. Fails closed to the CPU path.
        if guard.as_ref().is_some_and(|s| s.poisoned) {
            return Err(VaError::Status {
                op: "shared VCN session",
                code: -1,
                msg: "shared VCN session quarantined after terminal sync failure".to_string(),
            });
        }
        if guard.is_none() {
            *guard = Some(VaSession::open()?);
        }
        match guard
            .as_mut()
            .expect("pooled VCN session just opened")
            .decode_jpeg(jpeg)
        {
            Ok(DecodeOutcome::Decoded(frame)) => {
                Ok(Some(SharedDecodeOutcome::Decoded(SharedVcnLease {
                    _guard: guard,
                    frame,
                })))
            }
            Ok(DecodeOutcome::Unsupported(reason)) => {
                Ok(Some(SharedDecodeOutcome::Unsupported(reason)))
            }
            Err(e) => Err(e),
        }
    }

    /// Quarantine the shared session after a consumer's terminal sync fails.
    ///
    /// Takes the outstanding lease BY VALUE: the pool mutex is already held
    /// inside it, so locking here would self-deadlock, and dropping the
    /// guard normally (no forgetting) means no wedged requests. The flag is
    /// pool state under that same mutex; the backing session and its HIP
    /// mappings are RETAINED — never decoded through again, never torn down
    /// — so nothing is freed or reused under possibly-live GPU reads (a
    /// `sync_with_deadline` `Err` means "stopped waiting", not "GPU
    /// stopped"). Every later shared decode fails closed to the CPU path;
    /// the request at hand takes the CPU fallback on its retained bytes.
    /// Pre-enqueue early errors and unconsumed leases keep the normal
    /// release (no quarantine: nothing was ever launched).
    ///
    /// Simulated-fault probe recipe (proves the observable quarantine
    /// behavior WITHOUT injecting a real GPU hang — label it simulated, not
    /// a timeout reproduction): (1) lease-decode A and prove contender
    /// exclusion via `try_shared_decode_jpeg(B) == Ok(None)`; (2) consume A
    /// with a successful terminal sync; (3) INSTEAD of dropping the lease,
    /// pass it here to simulate a completion failure; (4) assert
    /// `try_shared_decode_jpeg(B)` and `shared_decode_jpeg_lease(B)` both
    /// return `Err(VaError::Status)` naming quarantine — refusal happens
    /// before any parse/render, so the pool is never reused. Poison is
    /// process-lifetime: run quarantine assertions last. (No CPU-only unit
    /// test covers this: a lease requires a live VA session, so the
    /// hardware probe owns the regression.)
    pub fn quarantine_shared(mut lease: SharedVcnLease<'_>) {
        lease
            ._guard
            .as_mut()
            .expect("quarantined lease holds a live shared session")
            .poisoned = true;
        drop(lease);
    }

    /// Allocate a pooled surface + decode context for `key`. Linear
    /// modifiers are the ONLY allocation mode: the exported dma-buf must be
    /// kernel-readable.
    fn alloc_pooled(
        lib: &VaLib,
        dpy: VaDisplay,
        config: VaConfigId,
        key: PoolKey,
    ) -> Result<(VaSurfaceId, VaContextId), VaError> {
        let fail = |op: &'static str, code: i32| VaError::Status {
            op,
            code,
            msg: lib.error_str(code),
        };
        let mut mods = [ffi::DRM_FORMAT_MOD_LINEAR];
        let mut mod_list = ffi::VaDrmFormatModifierList {
            num_modifiers: 1,
            modifiers: mods.as_mut_ptr(),
        };
        let mut attribs = [ffi::VaSurfaceAttrib {
            ty: ffi::VA_SURFACE_ATTRIB_DRM_FORMAT_MODIFIERS,
            flags: ffi::VA_SURFACE_ATTRIB_SETTABLE,
            value: ffi::VaGenericValue {
                ty: ffi::VA_GENERIC_VALUE_TYPE_POINTER,
                value: ffi::VaGenericValueUnion {
                    p: (&mut mod_list as *mut ffi::VaDrmFormatModifierList).cast(),
                },
            },
        }];
        let mut surf = 0;
        let mut st = unsafe {
            (lib.va_create_surfaces)(
                dpy,
                key.rt_format,
                key.width,
                key.height,
                &mut surf,
                1,
                attribs.as_mut_ptr().cast(),
                1,
            )
        };
        if st != VA_STATUS_SUCCESS {
            return Err(fail("vaCreateSurfaces", st));
        }
        let mut ctx = 0;
        st = unsafe {
            (lib.va_create_context)(
                dpy,
                config,
                key.width as i32,
                key.height as i32,
                0,
                &mut surf,
                1,
                &mut ctx,
            )
        };
        if st != VA_STATUS_SUCCESS {
            // SAFETY: surface was just created; destroy exactly once.
            unsafe {
                (lib.va_destroy_surfaces)(dpy, &mut surf, 1);
            }
            return Err(fail("vaCreateContext", st));
        }
        Ok((surf, ctx))
    }

    /// Fill the five VA buffers from parsed params, begin/render/end, sync.
    /// Buffers are destroyed before return regardless of submit outcome.
    fn render(
        lib: &VaLib,
        dpy: VaDisplay,
        surf: VaSurfaceId,
        ctx: VaContextId,
        p: &JpegVaParams<'_>,
    ) -> Result<(), VaError> {
        let fail = |op: &'static str, code: i32| VaError::Status {
            op,
            code,
            msg: lib.error_str(code),
        };
        let mut bufs = Self::create_buffers(lib, dpy, ctx, p)?;
        let mut st = unsafe { (lib.va_begin_picture)(dpy, ctx, surf) };
        if st == VA_STATUS_SUCCESS {
            st = unsafe { (lib.va_render_picture)(dpy, ctx, bufs.as_mut_ptr(), 5) };
        }
        if st == VA_STATUS_SUCCESS {
            st = unsafe { (lib.va_end_picture)(dpy, ctx) };
        }
        for b in bufs {
            // SAFETY: created above; destroy regardless of submit outcome.
            unsafe {
                (lib.va_destroy_buffer)(dpy, b);
            }
        }
        if st != VA_STATUS_SUCCESS {
            return Err(fail("submit(begin/render/end)Picture", st));
        }
        st = unsafe { (lib.va_sync_surface)(dpy, surf) };
        if st != VA_STATUS_SUCCESS {
            return Err(fail("vaSyncSurface", st));
        }
        Ok(())
    }

    /// Create the five VA buffers from parsed params (data copied by
    /// `vaCreateBuffer`). On failure, destroys what was created.
    fn create_buffers(
        lib: &VaLib,
        dpy: VaDisplay,
        ctx: VaContextId,
        p: &JpegVaParams<'_>,
    ) -> Result<[u32; 5], VaError> {
        let fail = |op: &'static str, code: i32| VaError::Status {
            op,
            code,
            msg: lib.error_str(code),
        };
        // Fill the five buffers (data copied by vaCreateBuffer).
        let mut pic = p.pic;
        let mut iq = p.iq;
        let mut huff = p.huff;
        let mut slice = p.slice;
        let mut bufs = [0u32; 5];
        let specs: [(i32, *mut c_void); 5] = [
            (
                ffi::VA_PIC_PARAM_TYPE,
                (&mut pic as *mut ffi::VaJpegPicParam).cast(),
            ),
            (
                ffi::VA_IQ_MATRIX_TYPE,
                (&mut iq as *mut ffi::VaJpegIQMatrix).cast(),
            ),
            (
                ffi::VA_HUFFMAN_TABLE_TYPE,
                (&mut huff as *mut ffi::VaJpegHuffmanBuffer).cast(),
            ),
            (
                ffi::VA_SLICE_PARAM_TYPE,
                (&mut slice as *mut ffi::VaJpegSliceParam).cast(),
            ),
            (ffi::VA_SLICE_DATA_TYPE, p.entropy.as_ptr() as *mut c_void),
        ];
        for (i, (ty, data)) in specs.iter().enumerate() {
            let size = match *ty {
                ffi::VA_SLICE_DATA_TYPE => p.entropy.len() as u32,
                ffi::VA_PIC_PARAM_TYPE => std::mem::size_of::<crate::ffi::VaJpegPicParam>() as u32,
                ffi::VA_IQ_MATRIX_TYPE => std::mem::size_of::<crate::ffi::VaJpegIQMatrix>() as u32,
                ffi::VA_HUFFMAN_TABLE_TYPE => {
                    std::mem::size_of::<crate::ffi::VaJpegHuffmanBuffer>() as u32
                }
                _ => std::mem::size_of::<crate::ffi::VaJpegSliceParam>() as u32,
            };
            let st = unsafe { (lib.va_create_buffer)(dpy, ctx, *ty, size, 1, *data, &mut bufs[i]) };
            if st != VA_STATUS_SUCCESS {
                for b in bufs[..i].iter() {
                    // SAFETY: created above.
                    unsafe {
                        (lib.va_destroy_buffer)(dpy, *b);
                    }
                }
                return Err(fail("vaCreateBuffer", st));
            }
        }
        Ok(bufs)
    }

    /// Export `surf` + import into HIP, once per pool key. The layer layout
    /// is normalized to separate-layers form (see [`normalize_export_layers`])
    /// before import; anything not safely consumable closes every exported
    /// fd and falls back to CPU — never a partial import. Imported fds are
    /// closed after import (HIP holds its own reference).
    fn export_once(
        lib: &VaLib,
        dpy: VaDisplay,
        surf: VaSurfaceId,
        p: &JpegVaParams<'_>,
    ) -> Result<CachedExport, VaError> {
        let fail = |op: &'static str, code: i32| VaError::Status {
            op,
            code,
            msg: lib.error_str(code),
        };
        let mut desc = ffi::VaDrmPrimeDescriptor {
            fourcc: 0,
            width: 0,
            height: 0,
            num_objects: 0,
            objects: [ffi::VaDrmPrimeObject::default(); 4],
            num_layers: 0,
            layers: [ffi::VaDrmPrimeLayer::default(); 4],
        };
        let st = unsafe {
            (lib.va_export_surface_handle)(
                dpy,
                surf,
                VA_MEM_TYPE_DRM_PRIME_2,
                VA_EXPORT_SURFACE_READ_ONLY,
                (&mut desc as *mut ffi::VaDrmPrimeDescriptor).cast(),
            )
        };
        if st != VA_STATUS_SUCCESS {
            return Err(fail("vaExportSurfaceHandle", st));
        }
        if hipfire_config::developer_var("HIPFIRE_VCN_DEBUG").is_ok() {
            eprintln!(
                "[va-bridge] export fourcc=0x{:08x} {}x{} objects={} layers={}",
                desc.fourcc, desc.width, desc.height, desc.num_objects, desc.num_layers
            );
            for i in 0..desc.num_objects.min(4) as usize {
                let o = &desc.objects[i];
                let end = libc_lseek_end(o.fd);
                eprintln!(
                    "[va-bridge]   obj{i}: fd={} size_field={} lseek_end={} mod=0x{:x}",
                    o.fd, o.size, end, o.drm_format_modifier
                );
            }
            for i in 0..desc.num_layers.min(4) as usize {
                let l = &desc.layers[i];
                eprintln!(
                    "[va-bridge]   layer{i}: fmt=0x{:08x} planes={} pitch={:?} off={:?} obj={:?}",
                    l.drm_format,
                    l.num_planes,
                    &l.pitch[..l.num_planes.min(4) as usize],
                    &l.offset[..l.num_planes.min(4) as usize],
                    &l.object_index[..l.num_planes.min(4) as usize]
                );
            }
        }
        // Normalize BEFORE import: every consumed plane must live in the
        // single mapped object with a checked byte range, and Y must sit at
        // the surface base the kernel reads. Anything else is a clean CPU
        // fallback, never a partial import.
        let (layers, num_layers) =
            match normalize_export_layers(&desc, p.width as u32, p.height as u32) {
                Ok(v) => v,
                Err(reason) => {
                    close_export_fds(&desc);
                    return Err(VaError::Corrupt(reason));
                }
            };
        let fd = desc.objects[0].fd;
        let size = desc.objects[0].size as usize;
        let mapping = HipMapping::import_dma_buf(fd, size);
        // fds came from vaExportSurfaceHandle; we own these copies.
        close_export_fds(&desc);
        let mapping = mapping?;
        Ok(CachedExport {
            max_h: p.max_h,
            max_v: p.max_v,
            fourcc: desc.fourcc,
            layers,
            num_layers,
            mapping,
        })
    }

    /// Decode one JPEG to the pooled device frame.
    ///
    /// `Ok(Decoded)` carries geometry, the layer layout, and the pooled
    /// device pointer (stable per key — see the struct docs). The surface is
    /// linear-only, so the pointer is directly kernel-readable.
    ///
    /// `Ok(Unsupported)` (progressive/arithmetic/12-bit/CMYK) is a per-stream
    /// CPU fallback: the pool entry and session are untouched, NO teardown.
    /// `Err` is real failures only (driver errors, corrupt streams, export).
    pub fn decode_jpeg(&mut self, jpeg: &[u8]) -> Result<DecodeOutcome, VaError> {
        let p = match parse_for_va(jpeg) {
            Ok(p) => p,
            Err(VaError::Unsupported(reason)) => return Ok(DecodeOutcome::Unsupported(reason)),
            Err(e) => return Err(e),
        };
        let key = PoolKey {
            rt_format: p.rt_format,
            width: p.width as u32,
            height: p.height as u32,
        };
        if !self.pool.contains_key(&key) {
            let (surf, ctx) = Self::alloc_pooled(&self.lib, self.dpy, self.config, key)?;
            self.pool.insert(
                key,
                PooledEntry {
                    surf,
                    ctx,
                    export: None,
                },
            );
        }
        let (surf, ctx) = {
            let e = self.pool.get(&key).expect("pool entry just inserted");
            (e.surf, e.ctx)
        };
        Self::render(&self.lib, self.dpy, surf, ctx, &p)?;
        let needs_export = self
            .pool
            .get(&key)
            .expect("pool entry just rendered")
            .export
            .is_none();
        if needs_export {
            let cached = Self::export_once(&self.lib, self.dpy, surf, &p)?;
            self.pool
                .get_mut(&key)
                .expect("pool entry just rendered")
                .export = Some(cached);
        }
        let e = self.pool.get(&key).expect("pool entry just exported");
        let c = e.export.as_ref().expect("export just cached");
        Ok(DecodeOutcome::Decoded(VcnFrame {
            width: key.width,
            height: key.height,
            max_h: c.max_h,
            max_v: c.max_v,
            fourcc: c.fourcc,
            layers: c.layers,
            num_layers: c.num_layers,
            ptr: c.mapping.ptr(),
        }))
    }

    /// Decode one baseline JPEG and read the pixels back through
    /// `vaDeriveImage`/`vaMapBuffer` (the driver resolves the linear surface
    /// into a CPU mapping). The `vl_vcn_444check` oracle for VCN decode
    /// correctness; it costs a GPU→CPU copy but no JPEG entropy work on CPU.
    /// Format-generic readback: every plane the driver reports, packed
    /// row-by-row (pitch stripped). Plane geometry follows the fourcc:
    /// NV12 = [Y w*h, CbCr cw*ch*2]; 444P = [Y, Cb, Cr] each w*h; anything
    /// else is returned as-is with its pitch-stripped rows sized from the
    /// image height and pitch, and the caller must know the layout.
    pub fn decode_jpeg_planes(&mut self, jpeg: &[u8]) -> Result<DerivedPlanes, VaError> {
        let lib = &self.lib;
        let fail = |op: &'static str, code: i32| VaError::Status {
            op,
            code,
            msg: lib.error_str(code),
        };
        let p = parse_for_va(jpeg)?;
        let key = PoolKey {
            rt_format: p.rt_format,
            width: p.width as u32,
            height: p.height as u32,
        };
        if !self.pool.contains_key(&key) {
            let (surf, ctx) = Self::alloc_pooled(&self.lib, self.dpy, self.config, key)?;
            self.pool.insert(
                key,
                PooledEntry {
                    surf,
                    ctx,
                    export: None,
                },
            );
        }
        let (width, height, surf) = {
            let e = self.pool.get(&key).expect("pool entry just inserted");
            (key.width, key.height, e.surf)
        };
        let ctx = self.pool.get(&key).expect("pool entry just inserted").ctx;
        Self::render(&self.lib, self.dpy, surf, ctx, &p)?;
        // Pooled surface/context stay alive in the session (no guards).
        let mut img = ffi::VaImage {
            image_id: 0,
            format: ffi::VaImageFormat {
                fourcc: 0,
                byte_order: 0,
                bits_per_pixel: 0,
                depth: 0,
                red_mask: 0,
                green_mask: 0,
                blue_mask: 0,
                alpha_mask: 0,
                va_reserved: [0; 4],
            },
            buf: 0,
            width: 0,
            height: 0,
            data_size: 0,
            num_planes: 0,
            pitches: [0; 3],
            offsets: [0; 3],
            num_palette_entries: 0,
            entry_bytes: 0,
            component_order: [0; 4],
            va_reserved: [0; 4],
        };
        // SAFETY: out-param is a valid VaImage slot.
        let mut st = unsafe { (lib.va_derive_image)(self.dpy, surf, &mut img) };
        if st != VA_STATUS_SUCCESS {
            return Err(fail("vaDeriveImage", st));
        }
        struct ImgGuard<'a> {
            lib: &'a VaLib,
            dpy: VaDisplay,
            id: u32,
        }
        impl Drop for ImgGuard<'_> {
            fn drop(&mut self) {
                // SAFETY: image was derived; destroy exactly once.
                unsafe {
                    (self.lib.va_destroy_image)(self.dpy, self.id);
                }
            }
        }
        let _img_guard = ImgGuard {
            lib,
            dpy: self.dpy,
            id: img.image_id,
        };
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: out-param is a valid pointer slot; unmapped below.
        st = unsafe { (lib.va_map_buffer)(self.dpy, img.buf, &mut ptr) };
        if st != VA_STATUS_SUCCESS || ptr.is_null() {
            return Err(fail("vaMapBuffer", st));
        }
        let (w, h) = (width as usize, height as usize);
        let np = img.num_planes.min(3) as usize;
        // Row width per plane by fourcc; fall back to pitch for unknown layouts.
        let row_w = |i: usize| -> usize {
            match (img.format.fourcc, i) {
                (VA_FOURCC_NV12, 0) => w,
                (VA_FOURCC_NV12, _) => ((w + 1) / 2) * 2,
                (0x5034_3434, _) => w, // 444P
                _ => img.pitches[i] as usize,
            }
        };
        let rows = |i: usize| -> usize {
            match (img.format.fourcc, i) {
                (VA_FOURCC_NV12, 0) => h,
                (VA_FOURCC_NV12, _) => (h + 1) / 2,
                _ => h,
            }
        };
        let mut planes = Vec::with_capacity(np);
        // SAFETY: mapped bytes; every row read stays inside [offset, offset + pitch*rows).
        unsafe {
            let base = ptr as *const u8;
            for i in 0..np {
                let (rw, nr, pitch, off) = (
                    row_w(i),
                    rows(i),
                    img.pitches[i] as usize,
                    img.offsets[i] as usize,
                );
                let mut out = vec![0u8; rw * nr];
                for r in 0..nr {
                    let src = std::slice::from_raw_parts(base.add(off + r * pitch), rw);
                    out[r * rw..(r + 1) * rw].copy_from_slice(src);
                }
                planes.push(out);
            }
            (lib.va_unmap_buffer)(self.dpy, img.buf);
        }
        Ok(DerivedPlanes {
            width,
            height,
            fourcc: img.format.fourcc,
            planes,
        })
    }
}

/// Planar-444 fourcc (`444P`) the driver reports for 4:4:4 surfaces.
const FOURCC_444P: u32 = 0x5034_3434;

/// Close every fd `vaExportSurfaceHandle` handed us, exactly once. Called on
/// ALL exits after a successful export: layout rejection, import failure,
/// and import success (HIP holds its own reference then).
fn close_export_fds(desc: &ffi::VaDrmPrimeDescriptor) {
    for i in 0..desc.num_objects.min(4) as usize {
        libc_close(desc.objects[i].fd);
    }
}

/// Every plane of `layer` must live in the mapped (first) object. Counts
/// are validated before indexing so a corrupt `num_planes` cannot panic.
fn check_single_object(layer: &ffi::VaDrmPrimeLayer) -> Result<(), &'static str> {
    if layer.num_planes == 0 || layer.num_planes > 4 {
        return Err("export plane count out of range");
    }
    for j in 0..layer.num_planes as usize {
        if layer.object_index[j] != 0 {
            return Err("export plane lives outside the mapped object");
        }
    }
    Ok(())
}

/// Checked `[offset, offset + pitch * (rows - 1) + row_bytes]` containment
/// in the mapped object (`size`). Requires `row_bytes <= pitch`; the
/// pitch/row tail is padding the kernel never reads. All math is checked:
/// overflow rejects, never wraps.
fn check_plane_range(
    offset: u32,
    pitch: u32,
    row_bytes: u64,
    rows: u64,
    size: u64,
) -> Result<(), &'static str> {
    if pitch == 0 {
        return Err("export plane has zero pitch");
    }
    if row_bytes > pitch as u64 {
        return Err("export plane row wider than pitch");
    }
    let tail = rows
        .checked_sub(1)
        .ok_or("export plane has zero rows")?
        .checked_mul(pitch as u64)
        .ok_or("export plane span overflows")?;
    let end = (offset as u64)
        .checked_add(tail)
        .ok_or("export plane span overflows")?
        .checked_add(row_bytes)
        .ok_or("export plane span overflows")?;
    if end > size {
        return Err("export plane range outside mapped object");
    }
    Ok(())
}

/// Normalized separate-layers view of a `vaExportSurfaceHandle` descriptor.
///
/// Export flags do not force one layout: a driver may legally return one
/// multi-plane (composed) layer or several single-plane (separate) layers,
/// and may return multiple dma-buf objects. The preprocess kernel, however,
/// reads Y at the mapping base plus pitched planes addressed by
/// [`VcnFrame::y_pitch`]/[`VcnFrame::uv_offset`]/[`VcnFrame::uv_pitch`] (or
/// `layers[1..3]` for planar 444). This helper accepts exactly the layouts
/// the kernel can consume and rewrites composed single layers into
/// separate-layers form:
///
/// * NV12: one 2-plane layer, or two 1-plane layers (Y, interleaved UV).
/// * planar 444: one 3-plane layer, or three 1-plane layers (Y, U, V).
/// * anything else: every reported plane must already sit in object 0 with
///   a checked in-range row span (the caller falls back on the fourcc
///   before any kernel reads it, but a cached export is never left OOB).
///
/// Rejection rules (every `Err` is a clean CPU fallback before import):
/// counts are validated before any indexing; every consumed plane must have
/// `object_index == 0` (the only object ever mapped); Y must sit at offset
/// 0 (the kernel reads Y at the base); every consumed plane's checked byte
/// range must lie inside the mapped object's size, with NV12 chroma sized
/// for odd dimensions (`row = ceil(w/2)*2`, `rows = ceil(h/2)`) and 444
/// chroma full-size.
fn normalize_export_layers(
    desc: &ffi::VaDrmPrimeDescriptor,
    width: u32,
    height: u32,
) -> Result<([ffi::VaDrmPrimeLayer; 4], u32), &'static str> {
    if desc.num_objects == 0 || desc.num_objects > 4 {
        return Err("export object count out of range");
    }
    if desc.num_layers == 0 || desc.num_layers > 4 {
        return Err("export layer count out of range");
    }
    if width == 0 || height == 0 {
        return Err("export surface has zero geometry");
    }
    // Y is read at the mapping base: a nonzero Y offset has no kernel arm.
    if desc.layers[0].offset[0] != 0 {
        return Err("export Y plane has nonzero offset");
    }
    let size = desc.objects[0].size as u64;
    let w = width as u64;
    let h = height as u64;
    if desc.fourcc == VA_FOURCC_NV12 {
        // NV12 chroma with odd dimensions: ceil(w/2) pairs interleaved,
        // ceil(h/2) rows.
        let cw = ((w + 1) / 2) * 2;
        let ch = (h + 1) / 2;
        if desc.num_layers == 2 {
            let (y, uv) = (&desc.layers[0], &desc.layers[1]);
            check_single_object(y)?;
            check_single_object(uv)?;
            check_plane_range(y.offset[0], y.pitch[0], w, h, size)?;
            check_plane_range(uv.offset[0], uv.pitch[0], cw, ch, size)?;
            Ok((desc.layers, 2))
        } else if desc.num_layers == 1 {
            let l = &desc.layers[0];
            if l.num_planes < 2 {
                return Err("export NV12 layer has fewer than 2 planes");
            }
            check_single_object(l)?;
            check_plane_range(l.offset[0], l.pitch[0], w, h, size)?;
            check_plane_range(l.offset[1], l.pitch[1], cw, ch, size)?;
            let mut out = desc.layers;
            out[0].num_planes = 1;
            out[1] = ffi::VaDrmPrimeLayer {
                drm_format: l.drm_format,
                num_planes: 1,
                object_index: [0, 0, 0, 0],
                offset: [l.offset[1], 0, 0, 0],
                pitch: [l.pitch[1], 0, 0, 0],
            };
            Ok((out, 2))
        } else {
            Err("export NV12 layer count not 1 (composed) or 2 (separate)")
        }
    } else if desc.fourcc == FOURCC_444P {
        if desc.num_layers == 3 {
            for l in desc.layers.iter().take(3) {
                check_single_object(l)?;
                check_plane_range(l.offset[0], l.pitch[0], w, h, size)?;
            }
            Ok((desc.layers, 3))
        } else if desc.num_layers == 1 {
            let l = &desc.layers[0];
            if l.num_planes < 3 {
                return Err("export 444 layer has fewer than 3 planes");
            }
            check_single_object(l)?;
            for j in 0..3 {
                check_plane_range(l.offset[j], l.pitch[j], w, h, size)?;
            }
            let mut out = desc.layers;
            out[0].num_planes = 1;
            for j in 1..3 {
                out[j] = ffi::VaDrmPrimeLayer {
                    drm_format: l.drm_format,
                    num_planes: 1,
                    object_index: [0, 0, 0, 0],
                    offset: [l.offset[j], 0, 0, 0],
                    pitch: [l.pitch[j], 0, 0, 0],
                };
            }
            Ok((out, 3))
        } else {
            Err("export 444 layer count not 1 (composed) or 3 (separate)")
        }
    } else {
        // Unknown fourcc: the caller falls back before any kernel read, but
        // only single-object, base-resident, span-checked planes are cached.
        for l in desc.layers.iter().take(desc.num_layers as usize) {
            check_single_object(l)?;
            for j in 0..l.num_planes as usize {
                check_plane_range(l.offset[j], l.pitch[j], l.pitch[j] as u64, h, size)?;
            }
        }
        Ok((desc.layers, desc.num_layers))
    }
}

// `libc` is not a workspace dep; close(2)/lseek(2) via direct externs.
// Unix-only: these symbols don't exist elsewhere, so the externs and the
// real wrappers are gated and stubbed (callers take CPU fallback there).
#[cfg(unix)]
unsafe extern "C" {
    fn close(fd: i32) -> i32;
    fn lseek(fd: i32, offset: i64, whence: i32) -> i64;
}
#[cfg(unix)]
pub(crate) fn libc_close(fd: i32) {
    // SAFETY: close(2) on an owned fd; return value intentionally ignored.
    unsafe {
        close(fd);
    }
}
#[cfg(not(unix))]
pub(crate) fn libc_close(_fd: i32) {}
/// SEEK_END probe for diagnostics; -1 on error (fd untouched otherwise).
#[cfg(unix)]
fn libc_lseek_end(fd: i32) -> i64 {
    // SAFETY: lseek(2) with SEEK_END does not mutate file offset usefully
    // for dma-bufs and returns the size; -1 on error.
    unsafe { lseek(fd, 0, 2) }
}
#[cfg(not(unix))]
fn libc_lseek_end(_fd: i32) -> i64 {
    -1
}

impl Drop for VaSession {
    fn drop(&mut self) {
        // SAFETY: every pooled surface/context was created; destroy exactly
        // once each, then the config and display.
        unsafe {
            for entry in self.pool.values() {
                (self.lib.va_destroy_context)(self.dpy, entry.ctx);
                let mut surf = entry.surf;
                (self.lib.va_destroy_surfaces)(self.dpy, &mut surf, 1);
            }
            // The pooled HIP imports die with their `HipMapping` drops here.
            (self.lib.va_destroy_config)(self.dpy, self.config);
            (self.lib.va_terminate)(self.dpy);
        }
    }
}

/// Explicit-session decode contract ([`VaSession::decode_jpeg`], oracles and
/// examples holding their own session). `Decoded` carries the pooled frame;
/// `Unsupported` (progressive/arithmetic/12-bit/CMYK — anything
/// [`parse_for_va`] rejects) means "take the CPU path" with NO session
/// teardown. `Err` is real failures only.
///
/// The shared (process-wide) session never returns this: concurrent callers
/// cannot hold `&mut` on one session, so it returns [`SharedDecodeOutcome`]
/// (leased) instead. See [`SharedVcnLease`].
#[derive(Clone, Copy)]
pub enum DecodeOutcome {
    Decoded(VcnFrame),
    Unsupported(&'static str),
}

/// Shared-session decode contract ([`VaSession::shared_decode_jpeg_lease`]).
/// Like [`DecodeOutcome`], but the decoded frame rides inside a
/// [`SharedVcnLease`] holding the session mutex, so no unguarded
/// shared-frame result escapes. `Unsupported` and `Err` hold no lease (the
/// mutex is released before return — nothing is outstanding).
pub enum SharedDecodeOutcome<'a> {
    Decoded(SharedVcnLease<'a>),
    Unsupported(&'static str),
}

/// Exclusive lease on a shared-session decode result.
///
/// [`VaSession::shared_decode_jpeg_lease`] locks the process-wide session
/// mutex into this guard: while the lease lives, no other thread can decode
/// through the shared session, so the pooled surface the [`VcnFrame`]
/// metadata points at cannot be overwritten. Neither `Clone` nor `Copy` —
/// there is exactly one owner.
///
/// Release boundary: drop the lease only after the consumer's device reads
/// are proven complete (checked HIP stream/device synchronization or event
/// completion), never at kernel-launch time. Launch is asynchronous; the
/// surface is still being read until the sync returns `Ok`.
///
/// Single-lease rule: the shared mutex is not reentrant. Never hold two
/// leases at once — decode image N+1 only after the previous lease is
/// consumed — or the second decode self-deadlocks.
pub struct SharedVcnLease<'a> {
    /// Held exclusively until the consumer's reads complete. Underscored:
    /// the lock itself is the value (its `Drop` releases the session);
    /// callers observe the frame through [`Self::frame`], never the session.
    _guard: MutexGuard<'a, Option<VaSession>>,
    /// Metadata snapshot (geometry, layout, device pointer) of the pooled
    /// surface at decode time. Meaningful only while the lease lives.
    frame: VcnFrame,
}

impl SharedVcnLease<'_> {
    /// The decoded frame: geometry + layer layout + pooled device pointer.
    /// Valid only while this lease is alive — never copy it out.
    pub fn frame(&self) -> &VcnFrame {
        &self.frame
    }
}

/// A VCN-decoded frame: geometry + layer layout + the pooled device pointer.
///
/// The pointer is stable per (`rt_format`, width, height) pool key, but its
/// *contents* reflect the most recent decode of that key — launch + sync
/// consumer kernels before the next decode. Valid until the session drops.
/// Shared-session callers never receive this directly (see
/// [`SharedVcnLease`]); only explicit-session owners holding `&mut` do.
///
/// `layers`/`num_layers` are NORMALIZED to separate-layers form by
/// [`normalize_export_layers`] before import: NV12 is exactly 2
/// single-plane layers (Y, interleaved UV), planar 444 exactly 3 (Y, U, V);
/// Y always sits at offset 0 and every consumed plane's checked byte range
/// lies inside the single mapped object. The accessors below therefore read
/// `layers` directly; anything else fell back to CPU before the frame was
/// built.
#[derive(Clone, Copy)]
pub struct VcnFrame {
    pub width: u32,
    pub height: u32,
    /// Max sampling factors (2,2 ⇒ 4:2:0; 1,1 ⇒ 4:4:4/gray).
    pub max_h: u8,
    pub max_v: u8,
    pub fourcc: u32,
    pub layers: [ffi::VaDrmPrimeLayer; 4],
    pub num_layers: u32,
    ptr: *mut c_void,
}

// A device address; validity is tied to the owning session (see above).
unsafe impl Send for VcnFrame {}

impl VcnFrame {
    /// Device pointer to the start of the exported (single-object) surface.
    pub fn device_ptr(&self) -> *mut c_void {
        self.ptr
    }
    /// Y-plane pitch in bytes (normalized layer 0, plane 0).
    pub fn y_pitch(&self) -> u32 {
        self.layers[0].pitch[0]
    }
    /// UV-plane byte offset from the surface base (normalized layer 1).
    pub fn uv_offset(&self) -> u32 {
        if self.num_layers > 1 {
            self.layers[1].offset[0]
        } else {
            self.height * self.y_pitch()
        }
    }
    /// UV-plane pitch in bytes (normalized layer 1).
    pub fn uv_pitch(&self) -> u32 {
        if self.num_layers > 1 {
            self.layers[1].pitch[0]
        } else {
            self.y_pitch()
        }
    }
    /// Diagnostics-only D2H copy of `len` bytes at `offset` (the debug and
    /// 444check examples). Opens the HIP runtime directly; never used on a
    /// hot path.
    pub fn copy_to_host(&self, offset: usize, len: usize) -> Result<Vec<u8>, VaError> {
        let candidates =
            hipfire_config::rocm::library_candidates(hipfire_config::rocm::HIP_RUNTIME_LIBRARIES);
        // SAFETY: system HIP runtime; signature matches hip_runtime_api.h.
        unsafe {
            for c in &candidates {
                let Ok(lib) = libloading::Library::new(c) else {
                    continue;
                };
                let Ok(f_memcpy): Result<
                    libloading::Symbol<
                        unsafe extern "C" fn(*mut c_void, *const c_void, usize, u32) -> u32,
                    >,
                    _,
                > = lib.get(b"hipMemcpy") else {
                    continue;
                };
                let mut host = vec![0u8; len];
                let src = (self.ptr as *const u8).add(offset) as *const c_void;
                // hipMemcpyDeviceToHost = 2.
                let code = f_memcpy(host.as_mut_ptr() as *mut c_void, src, len, 2);
                if code != 0 {
                    return Err(VaError::Status {
                        op: "hipMemcpy(D2H)",
                        code: code as i32,
                        msg: format!("code {code}"),
                    });
                }
                return Ok(host);
            }
        }
        Err(VaError::Dlopen {
            lib: candidates.join(","),
            msg: "no HIP runtime for VcnFrame::copy_to_host".to_string(),
        })
    }
}

/// A VCN-decoded frame read back through `vaDeriveImage`: packed planes with
/// driver-resolved (linear) layout. The `vl_vcn_444check` oracle for VCN
/// decode correctness (see [`VaSession::decode_jpeg_planes`]).
pub struct DerivedPlanes {
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    pub planes: Vec<Vec<u8>>,
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    fn mk_layer(
        planes: u32,
        objs: [u32; 4],
        offs: [u32; 4],
        pitches: [u32; 4],
    ) -> ffi::VaDrmPrimeLayer {
        ffi::VaDrmPrimeLayer {
            drm_format: 0,
            num_planes: planes,
            object_index: objs,
            offset: offs,
            pitch: pitches,
        }
    }

    fn mk_desc(
        fourcc: u32,
        sizes: &[u32],
        layers: &[ffi::VaDrmPrimeLayer],
    ) -> ffi::VaDrmPrimeDescriptor {
        let mut d = ffi::VaDrmPrimeDescriptor {
            fourcc,
            width: 0,
            height: 0,
            num_objects: sizes.len() as u32,
            objects: [ffi::VaDrmPrimeObject::default(); 4],
            num_layers: layers.len() as u32,
            layers: [ffi::VaDrmPrimeLayer::default(); 4],
        };
        for (i, s) in sizes.iter().enumerate().take(4) {
            d.objects[i] = ffi::VaDrmPrimeObject {
                fd: -1,
                size: *s,
                drm_format_modifier: 0,
            };
        }
        for (i, l) in layers.iter().enumerate().take(4) {
            d.layers[i] = *l;
        }
        d
    }

    fn frame_of(layers: [ffi::VaDrmPrimeLayer; 4], num_layers: u32) -> VcnFrame {
        VcnFrame {
            width: 64,
            height: 32,
            max_h: 2,
            max_v: 2,
            fourcc: VA_FOURCC_NV12,
            layers,
            num_layers,
            ptr: std::ptr::null_mut(),
        }
    }

    #[test]
    fn nv12_separate_even_passes_through() {
        // 64x32, pitch 64: Y [0, 2048), UV [2048, 3072).
        let d = mk_desc(
            VA_FOURCC_NV12,
            &[3072],
            &[
                mk_layer(1, [0, 0, 0, 0], [0, 0, 0, 0], [64, 0, 0, 0]),
                mk_layer(1, [0, 0, 0, 0], [2048, 0, 0, 0], [64, 0, 0, 0]),
            ],
        );
        let (layers, n) = normalize_export_layers(&d, 64, 32).unwrap();
        assert_eq!(n, 2);
        assert_eq!(layers[0].pitch[0], 64);
        assert_eq!(layers[1].offset[0], 2048);
        let f = frame_of(layers, n);
        assert_eq!((f.y_pitch(), f.uv_offset(), f.uv_pitch()), (64, 2048, 64));
    }

    #[test]
    fn nv12_composed_normalizes_to_separate() {
        // Same geometry, one 2-plane layer.
        let d = mk_desc(
            VA_FOURCC_NV12,
            &[3072],
            &[mk_layer(2, [0, 0, 0, 0], [0, 2048, 0, 0], [64, 64, 0, 0])],
        );
        let (layers, n) = normalize_export_layers(&d, 64, 32).unwrap();
        assert_eq!(n, 2);
        assert_eq!(layers[0].pitch[0], 64);
        assert_eq!((layers[1].offset[0], layers[1].pitch[0]), (2048, 64));
        let f = frame_of(layers, n);
        assert_eq!((f.y_pitch(), f.uv_offset(), f.uv_pitch()), (64, 2048, 64));
    }

    #[test]
    fn nv12_odd_dimensions_use_ceil_chroma() {
        // 5x3, pitch 8: Y [0, 21), chroma row 6 x 2 rows, UV [24, 38).
        let d = mk_desc(
            VA_FOURCC_NV12,
            &[38],
            &[mk_layer(2, [0, 0, 0, 0], [0, 24, 0, 0], [8, 8, 0, 0])],
        );
        let (layers, n) = normalize_export_layers(&d, 5, 3).unwrap();
        assert_eq!(n, 2);
        assert_eq!(layers[1].offset[0], 24);
        // One byte less backing store and the chroma range no longer fits.
        let short = mk_desc(
            VA_FOURCC_NV12,
            &[37],
            &[mk_layer(2, [0, 0, 0, 0], [0, 24, 0, 0], [8, 8, 0, 0])],
        );
        assert!(normalize_export_layers(&short, 5, 3).is_err());
    }

    #[test]
    fn planes_outside_first_object_reject() {
        // Composed layer whose chroma plane names object 1.
        let d = mk_desc(
            VA_FOURCC_NV12,
            &[3072, 3072],
            &[mk_layer(2, [0, 1, 0, 0], [0, 2048, 0, 0], [64, 64, 0, 0])],
        );
        assert!(normalize_export_layers(&d, 64, 32).is_err());
        // Separate layers with UV in object 1.
        let d = mk_desc(
            VA_FOURCC_NV12,
            &[2048, 1024],
            &[
                mk_layer(1, [0, 0, 0, 0], [0, 0, 0, 0], [64, 0, 0, 0]),
                mk_layer(1, [1, 0, 0, 0], [0, 0, 0, 0], [64, 0, 0, 0]),
            ],
        );
        assert!(normalize_export_layers(&d, 64, 32).is_err());
    }

    #[test]
    fn out_of_bounds_and_overflow_reject_without_panic() {
        let uv = mk_layer(1, [0, 0, 0, 0], [2048, 0, 0, 0], [64, 0, 0, 0]);
        let y = mk_layer(1, [0, 0, 0, 0], [0, 0, 0, 0], [64, 0, 0, 0]);
        // UV range ends at 3072; one byte short rejects.
        let d = mk_desc(VA_FOURCC_NV12, &[3071], &[y, uv]);
        assert!(normalize_export_layers(&d, 64, 32).is_err());
        // Absurd pitch blows the checked span past the object: rejects.
        let wide = mk_layer(1, [0, 0, 0, 0], [0, 0, 0, 0], [u32::MAX, 0, 0, 0]);
        let d = mk_desc(VA_FOURCC_NV12, &[u32::MAX], &[wide, uv]);
        assert!(normalize_export_layers(&d, 64, 64).is_err());
        // UV offset near u32::MAX: end computation overflows, rejects.
        let far = mk_layer(
            1,
            [0, 0, 0, 0],
            [u32::MAX, 0, 0, 0],
            [64, 0, 0, 0],
        );
        let d = mk_desc(VA_FOURCC_NV12, &[u32::MAX], &[y, far]);
        assert!(normalize_export_layers(&d, 64, 32).is_err());
    }

    #[test]
    fn nonzero_y_offset_and_bad_counts_reject() {
        let uv = mk_layer(1, [0, 0, 0, 0], [2048, 0, 0, 0], [64, 0, 0, 0]);
        // Y must sit at the mapping base the kernel reads.
        let shifted = mk_layer(1, [0, 0, 0, 0], [64, 0, 0, 0], [64, 0, 0, 0]);
        let d = mk_desc(VA_FOURCC_NV12, &[3072], &[shifted, uv]);
        assert!(normalize_export_layers(&d, 64, 32).is_err());
        let y = mk_layer(1, [0, 0, 0, 0], [0, 0, 0, 0], [64, 0, 0, 0]);
        // No objects / no layers: nothing to map.
        assert!(normalize_export_layers(&mk_desc(VA_FOURCC_NV12, &[], &[]), 64, 32).is_err());
        // Layer count past the descriptor array: rejected before indexing.
        let mut too_many = mk_desc(VA_FOURCC_NV12, &[3072], &[y, uv]);
        too_many.num_layers = 5;
        assert!(normalize_export_layers(&too_many, 64, 32).is_err());
        // Zero planes, five planes, zero pitch, row wider than pitch.
        for bad in [
            mk_layer(0, [0, 0, 0, 0], [0, 0, 0, 0], [64, 0, 0, 0]),
            mk_layer(5, [0, 0, 0, 0], [0, 0, 0, 0], [64, 0, 0, 0]),
            mk_layer(1, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]),
            mk_layer(1, [0, 0, 0, 0], [0, 0, 0, 0], [8, 0, 0, 0]),
        ] {
            let d = mk_desc(VA_FOURCC_NV12, &[3072], &[bad, uv]);
            assert!(normalize_export_layers(&d, 64, 32).is_err());
        }
        // Zero geometry rejects.
        let d = mk_desc(VA_FOURCC_NV12, &[3072], &[y, uv]);
        assert!(normalize_export_layers(&d, 0, 32).is_err());
    }

    #[test]
    fn p444_separate_and_composed() {
        // 16x8, pitch 16: Y [0, 128), U [128, 256), V [256, 384).
        let planes = [
            mk_layer(1, [0, 0, 0, 0], [0, 0, 0, 0], [16, 0, 0, 0]),
            mk_layer(1, [0, 0, 0, 0], [128, 0, 0, 0], [16, 0, 0, 0]),
            mk_layer(1, [0, 0, 0, 0], [256, 0, 0, 0], [16, 0, 0, 0]),
        ];
        let d = mk_desc(FOURCC_444P, &[384], &planes);
        let (layers, n) = normalize_export_layers(&d, 16, 8).unwrap();
        assert_eq!(n, 3);
        assert_eq!((layers[1].offset[0], layers[2].offset[0]), (128, 256));
        // Same extents as one composed 3-plane layer.
        let d = mk_desc(
            FOURCC_444P,
            &[384],
            &[mk_layer(
                3,
                [0, 0, 0, 0],
                [0, 128, 256, 0],
                [16, 16, 16, 0],
            )],
        );
        let (layers, n) = normalize_export_layers(&d, 16, 8).unwrap();
        assert_eq!(n, 3);
        assert_eq!((layers[1].offset[0], layers[2].offset[0]), (128, 256));
        // Chroma clipped by one byte rejects.
        let d = mk_desc(FOURCC_444P, &[383], &planes);
        assert!(normalize_export_layers(&d, 16, 8).is_err());
    }

    #[test]
    fn unknown_fourcc_stays_single_object_and_span_checked() {
        let one = mk_layer(1, [0, 0, 0, 0], [0, 0, 0, 0], [64, 0, 0, 0]);
        let d = mk_desc(0xDEAD_BEEF, &[2048], &[one]);
        let (layers, n) = normalize_export_layers(&d, 64, 32).unwrap();
        assert_eq!(n, 1);
        assert_eq!(layers[0].pitch[0], 64);
        let far = mk_layer(1, [1, 0, 0, 0], [0, 0, 0, 0], [64, 0, 0, 0]);
        let d = mk_desc(0xDEAD_BEEF, &[2048, 2048], &[far]);
        assert!(normalize_export_layers(&d, 64, 32).is_err());
    }
}
