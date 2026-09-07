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
//! [`DecodeOutcome`]: `Decoded` carries the pooled frame, `Unsupported`
//! (progressive/arithmetic/12-bit/CMYK) means "take the CPU path" with NO
//! session teardown; `Err` is real failures only.

mod ffi;
mod h264;
mod interop;
mod jpeg;
pub use ffi::{
    VaBufferId, VaConfigId, VaContextId, VaDisplay, VaDrmPrimeDescriptor, VaDrmPrimeLayer,
    VaDrmPrimeObject, VaError, VaJpegHuffmanBuffer, VaJpegIQMatrix, VaJpegPicParam,
    VaJpegSliceParam, VaLib, VaSurfaceId, VA_EXPORT_SURFACE_READ_ONLY,
    VA_EXPORT_SURFACE_SEPARATE_LAYERS, VA_FOURCC_NV12, VA_MEM_TYPE_DRM_PRIME_2,
    VA_PROFILE_AV1_PROFILE0, VA_PROFILE_H264_HIGH, VA_PROFILE_HEVC_MAIN, VA_STATUS_SUCCESS,
};
pub use interop::HipMapping;
pub use jpeg::{parse_for_va, JpegVaParams};

use ffi::{VA_ENTRYPOINT_VLD, VA_PROFILE_JPEG_BASELINE};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void};
use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::{Mutex, OnceLock};

/// DRM render-node candidates, in order: `HIPFIRE_VCN_DRM_NODE` (via
/// `developer_var`) wins; otherwise the node whose PCI slot matches HIP
/// device 0; otherwise every `renderD*` node, sorted (the planes oracle
/// needs no HIP runtime, so a missing HIP library still gets a scan).
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
/// of that key — consume (launch + sync kernels) before the next decode.
pub struct VaSession {
    lib: VaLib,
    dpy: VaDisplay,
    _drm_fd: OwnedFd,
    vendor: String,
    config: VaConfigId,
    node: String,
    pool: HashMap<PoolKey, PooledEntry>,
    /// HIP imports backing the most recent `h264::decode_h264_annexb_zc`
    /// frames. The video lane has no surface pool, so each zc frame's import
    /// lives here (not in the `Copy` `VcnFrame`); frames are valid until the
    /// next zc decode or the session drops — the video analogue of the pool
    /// contract above.
    video_zc: Vec<HipMapping>,
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

// The VA display handle is a raw pointer, so `VaSession` is `!Send` by
// default. Sharing is explicitly single-threaded behind the caller's
// `&mut` (or the `shared_decode_jpeg` mutex); cross-thread transfer of the
// session itself is the caller's responsibility, as with `HipMapping`.
unsafe impl Send for VaSession {}

impl VaSession {
    /// Open the first working render-node candidate, initialise VA, create
    /// the JPEG-baseline VLD config. See [`candidate_nodes`] for selection.
    pub fn open() -> Result<Self, VaError> {
        Self::open_profile(VA_PROFILE_JPEG_BASELINE)
    }

    /// Open the first working render node with a VLD config for `profile`
    /// (see `ffi::VA_PROFILE_*`; video lane uses H.264 High = 7).
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
            video_zc: Vec::new(),
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
    /// its pool — see [`VaSession::shared_decode_jpeg`] for the process-wide
    /// pooled session the product path uses).
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
    pub fn shared_decode_jpeg(jpeg: &[u8]) -> Result<DecodeOutcome, VaError> {
        static SHARED: std::sync::LazyLock<Mutex<Option<VaSession>>> =
            std::sync::LazyLock::new(|| Mutex::new(None));
        let mut guard = SHARED.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(VaSession::open()?);
        }
        guard
            .as_mut()
            .expect("pooled VCN session just opened")
            .decode_jpeg(jpeg)
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

    /// Export `surf` + import into HIP, once per pool key. The export fds
    /// are closed after import (HIP holds its own reference).
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
        if desc.num_objects == 0 {
            return Err(VaError::Corrupt("export yielded no objects"));
        }
        let fd = desc.objects[0].fd;
        let size = desc.objects[0].size as usize;
        let mapping = HipMapping::import_dma_buf(fd, size);
        // fds came from vaExportSurfaceHandle; we own these copies.
        libc_close(fd);
        for i in 1..desc.num_objects.min(4) as usize {
            libc_close(desc.objects[i].fd);
        }
        let mapping = mapping?;
        Ok(CachedExport {
            max_h: p.max_h,
            max_v: p.max_v,
            fourcc: desc.fourcc,
            layers: desc.layers,
            num_layers: desc.num_layers,
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

// `libc` is not a workspace dep; close(2)/lseek(2) via direct externs.
unsafe extern "C" {
    fn close(fd: i32) -> i32;
    fn lseek(fd: i32, offset: i64, whence: i32) -> i64;
}
pub(crate) fn libc_close(fd: i32) {
    // SAFETY: close(2) on an owned fd; return value intentionally ignored.
    unsafe {
        close(fd);
    }
}
/// SEEK_END probe for diagnostics; -1 on error (fd untouched otherwise).
fn libc_lseek_end(fd: i32) -> i64 {
    // SAFETY: lseek(2) with SEEK_END does not mutate file offset usefully
    // for dma-bufs and returns the size; -1 on error.
    unsafe { lseek(fd, 0, 2) }
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

/// Product decode contract. `Decoded` carries the pooled frame;
/// `Unsupported` (progressive/arithmetic/12-bit/CMYK — anything
/// [`parse_for_va`] rejects) means "take the CPU path" with NO session
/// teardown. `Err` is real failures only.
#[derive(Clone, Copy)]
pub enum DecodeOutcome {
    Decoded(VcnFrame),
    Unsupported(&'static str),
}

/// A VCN-decoded frame: geometry + layer layout + the pooled device pointer.
///
/// The pointer is stable per (`rt_format`, width, height) pool key, but its
/// *contents* reflect the most recent decode of that key — launch + sync
/// consumer kernels before the next decode. Valid until the session drops.
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
    /// Y-plane pitch in bytes (layer 0, plane 0).
    pub fn y_pitch(&self) -> u32 {
        self.layers[0].pitch[0]
    }
    /// UV-plane byte offset from the surface base.
    pub fn uv_offset(&self) -> u32 {
        if self.num_layers > 1 {
            self.layers[1].offset[0]
        } else {
            self.height * self.y_pitch()
        }
    }
    /// UV-plane pitch in bytes.
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

/// The H.264 lane's derived frame (see `h264::decode_h264_collect`): packed
/// NV12 luma + interleaved chroma read back through `vaDeriveImage`.
/// (`experiment/vcn-video`; restored verbatim in the vcn-consolidated merge —
/// it lived beside the old JPEG derived path the pool lane deleted.)
pub struct DerivedFrame {
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    pub y_pitch: u32,
    pub uv_pitch: u32,
    /// Packed luma, `width * height` bytes.
    pub y: Vec<u8>,
    /// Packed interleaved chroma, `(width/2) * (height/2) * 2` bytes.
    pub uv: Vec<u8>,
}
