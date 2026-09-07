// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `va-bridge`: drive the VCN JPEG engine through libva directly.
//!
//! Experiment `experiment/vcn-jpeg` only. Nothing here runs unless a caller
//! opts in (`vl_vcn_parity` lab example); absence of libva at runtime is a
//! recoverable [`VaError`], and the caller falls back to the
//! `libjpeg-turbo-rs` CPU oracle (byte-identical to PIL per
//! `docs/VALIDATION.md`).
//!
//! Pipeline per frame: [`VaSession::decode_jpeg`] (parse → VA buffers →
//! `vaBegin/Render/EndPicture` → `vaSyncSurface` → `vaExportSurfaceHandle`)
//! then [`HipMapping::import_dma_buf`] for the zero-copy HIP device pointer
//! the `vl_yuv_preprocess` kernel consumes.

mod ffi;
mod interop;
mod jpeg;

pub use ffi::{
    VaBufferId, VaConfigId, VaContextId, VaDisplay, VaDrmPrimeDescriptor, VaDrmPrimeLayer,
    VaDrmPrimeObject, VaJpegHuffmanBuffer, VaJpegIQMatrix, VaJpegPicParam, VaJpegSliceParam,
    VaLib, VaSurfaceId, VaError, VA_EXPORT_SURFACE_READ_ONLY,
    VA_EXPORT_SURFACE_SEPARATE_LAYERS, VA_FOURCC_NV12, VA_MEM_TYPE_DRM_PRIME_2,
    VA_STATUS_SUCCESS,
};
pub use interop::HipMapping;
pub use jpeg::{parse_for_va, JpegVaParams};

use ffi::{VA_ENTRYPOINT_VLD, VA_PROFILE_JPEG_BASELINE, VA_RT_FORMAT_YUV420};
use std::ffi::c_void;
use std::os::fd::{AsRawFd, OwnedFd};

/// DRM render nodes to try, in order. `HIPFIRE_VCN_DRM_NODE` (via
/// `developer_var`) overrides with a single path.
fn render_nodes() -> Vec<String> {
    if let Ok(one) = hipfire_config::developer_var("HIPFIRE_VCN_DRM_NODE") {
        return vec![one];
    }
    (128..132)
        .map(|i| format!("/dev/dri/renderD{i}"))
        .collect()
}

/// An initialised VA display with a JPEG-baseline VLD config.
///
/// Surfaces and decode contexts are created per frame (fixtures differ in
/// size) and destroyed after the dma-buf is imported into HIP; the
/// [`HipMapping`] holds its own dma-buf reference, so the device pointer
/// outlives the VA objects.
pub struct VaSession {
    lib: VaLib,
    dpy: VaDisplay,
    _drm_fd: OwnedFd,
    vendor: String,
    config: VaConfigId,
}

impl VaSession {
    /// Open the first working render node, initialise VA, create the JPEG
    pub fn open() -> Result<Self, VaError> {
        let mut lib = Some(VaLib::load()?);
        let mut last_err = VaError::NoRenderNode;
        for node in render_nodes() {
            let l = match lib.take() {
                Some(l) => l,
                None => break,
            };
            match Self::open_node(l, &node) {
                Ok(s) => return Ok(s),
                Err((l, e)) => {
                    lib = Some(l);
                    last_err = e;
                }
            }
        }
        Err(last_err)
    }

    fn open_node(lib: VaLib, node: &str) -> Result<Self, (VaLib, VaError)> {
        // O_RDWR: radeonsi winsys creation (GEM backing) fails on O_RDONLY
        // with EACCES (`amdgpu_bo_cpu_map failed (-13)`), which surfaces as
        // vaInitialize succeeding but context creation segfaulting.
        let fd = match std::fs::OpenOptions::new().read(true).write(true).open(node) {
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
        // JPEG baseline advertised?
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
        if !profiles[..n.max(0) as usize].contains(&VA_PROFILE_JPEG_BASELINE) {
            unsafe {
                (lib.va_terminate)(dpy);
            }
            return Err((lib, VaError::Unsupported("driver has no VAProfileJPEGBaseline")));
        }
        let mut config = 0;
        let st = unsafe {
            (lib.va_create_config)(
                dpy,
                VA_PROFILE_JPEG_BASELINE,
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
                op: "vaCreateConfig(JPEG,VLD)",
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
        })
    }

    pub fn vendor(&self) -> &str {
        &self.vendor
    }

    /// Decode one baseline JPEG to an exported dma-buf + HIP mapping.
    ///
    /// Returns the frame geometry, the NV12 layer layout (pitches/offsets),
    /// and the device mapping. All VA objects are destroyed before return;
    /// only the [`HipMapping`] (plus plain geometry) escapes.
    pub fn decode_jpeg(&self, jpeg: &[u8]) -> Result<VcnFrame, VaError> {
        let p = parse_for_va(jpeg)?;
        let lib = &self.lib;
        let fail = |op: &'static str, code: i32| VaError::Status {
            op,
            code,
            msg: lib.error_str(code),
        };

        // Surfaces sized to the frame.
        let mut surf = 0;
        let mut st = unsafe {
            (lib.va_create_surfaces)(
                self.dpy,
                VA_RT_FORMAT_YUV420,
                p.width as u32,
                p.height as u32,
                &mut surf,
                1,
                std::ptr::null_mut(),
                0,
            )
        };
        if st != VA_STATUS_SUCCESS {
            return Err(fail("vaCreateSurfaces", st));
        }
        struct SurfGuard<'a> {
            lib: &'a VaLib,
            dpy: VaDisplay,
            surf: VaSurfaceId,
        }
        impl Drop for SurfGuard<'_> {
            fn drop(&mut self) {
                // SAFETY: surface was created; destroy exactly once.
                unsafe {
                    (self.lib.va_destroy_surfaces)(self.dpy, &mut self.surf, 1);
                }
            }
        }
        let _surf_guard = SurfGuard {
            lib,
            dpy: self.dpy,
            surf,
        };

        let mut ctx = 0;
        st = unsafe {
            (lib.va_create_context)(
                self.dpy,
                self.config,
                p.width as i32,
                p.height as i32,
                0,
                &mut surf,
                1,
                &mut ctx,
            )
        };
        if st != VA_STATUS_SUCCESS {
            return Err(fail("vaCreateContext", st));
        }
        struct CtxGuard<'a> {
            lib: &'a VaLib,
            dpy: VaDisplay,
            ctx: VaContextId,
        }
        impl Drop for CtxGuard<'_> {
            fn drop(&mut self) {
                // SAFETY: context was created; destroy exactly once.
                unsafe {
                    (self.lib.va_destroy_context)(self.dpy, self.ctx);
                }
            }
        }
        let _ctx_guard = CtxGuard {
            lib,
            dpy: self.dpy,
            ctx,
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
            (ffi::VA_IQ_MATRIX_TYPE, (&mut iq as *mut ffi::VaJpegIQMatrix).cast()),
            (
                ffi::VA_HUFFMAN_TABLE_TYPE,
                (&mut huff as *mut ffi::VaJpegHuffmanBuffer).cast(),
            ),
            (
                ffi::VA_SLICE_PARAM_TYPE,
                (&mut slice as *mut ffi::VaJpegSliceParam).cast(),
            ),
            (
                ffi::VA_SLICE_DATA_TYPE,
                p.entropy.as_ptr() as *mut c_void,
            ),
        ];
        for (i, (ty, data)) in specs.iter().enumerate() {
            let size = match *ty {
                ffi::VA_SLICE_DATA_TYPE => p.entropy.len() as u32,
                ffi::VA_PIC_PARAM_TYPE => {
                    std::mem::size_of::<crate::ffi::VaJpegPicParam>() as u32
                }
                ffi::VA_IQ_MATRIX_TYPE => {
                    std::mem::size_of::<crate::ffi::VaJpegIQMatrix>() as u32
                }
                ffi::VA_HUFFMAN_TABLE_TYPE => {
                    std::mem::size_of::<crate::ffi::VaJpegHuffmanBuffer>() as u32
                }
                _ => std::mem::size_of::<crate::ffi::VaJpegSliceParam>() as u32,
            };
            st = unsafe {
                (lib.va_create_buffer)(self.dpy, ctx, *ty, size, 1, *data, &mut bufs[i])
            };
            if st != VA_STATUS_SUCCESS {
                for b in bufs[..i].iter() {
                    // SAFETY: created above.
                    unsafe {
                        (lib.va_destroy_buffer)(self.dpy, *b);
                    }
                }
                return Err(fail("vaCreateBuffer", st));
            }
        }

        st = unsafe { (lib.va_begin_picture)(self.dpy, ctx, surf) };
        if st == VA_STATUS_SUCCESS {
            st = unsafe { (lib.va_render_picture)(self.dpy, ctx, bufs.as_mut_ptr(), 5) };
        }
        if st == VA_STATUS_SUCCESS {
            st = unsafe { (lib.va_end_picture)(self.dpy, ctx) };
        }
        for b in bufs {
            // SAFETY: created above; destroy regardless of submit outcome.
            unsafe {
                (lib.va_destroy_buffer)(self.dpy, b);
            }
        }
        if st != VA_STATUS_SUCCESS {
            return Err(fail("submit(begin/render/end)Picture", st));
        }
        st = unsafe { (lib.va_sync_surface)(self.dpy, surf) };
        if st != VA_STATUS_SUCCESS {
            return Err(fail("vaSyncSurface", st));
        }

        // Export + import. The export fds are closed after import (HIP holds
        // its own reference); the surface/context die with the guards.
        let mut desc = ffi::VaDrmPrimeDescriptor {
            fourcc: 0,
            width: 0,
            height: 0,
            num_objects: 0,
            objects: [ffi::VaDrmPrimeObject::default(); 4],
            num_layers: 0,
            layers: [ffi::VaDrmPrimeLayer::default(); 4],
        };
        st = unsafe {
            (lib.va_export_surface_handle)(
                self.dpy,
                surf,
                VA_MEM_TYPE_DRM_PRIME_2,
                VA_EXPORT_SURFACE_READ_ONLY,
                (&mut desc as *mut ffi::VaDrmPrimeDescriptor).cast(),
            )
        };
        if st != VA_STATUS_SUCCESS {
            return Err(fail("vaExportSurfaceHandle", st));
        }
        if std::env::var_os("HIPFIRE_VCN_DEBUG").is_some() {
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
        Ok(VcnFrame {
            width: p.width as u32,
            height: p.height as u32,
            max_h: p.max_h,
            max_v: p.max_v,
            fourcc: desc.fourcc,
            layers: desc.layers,
            num_layers: desc.num_layers,
            mapping,
        })
    }
}

// `libc` is not a workspace dep; close(2)/lseek(2) via direct externs.
unsafe extern "C" {
    fn close(fd: i32) -> i32;
    fn lseek(fd: i32, offset: i64, whence: i32) -> i64;
}
fn libc_close(fd: i32) {
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
        // SAFETY: config/display were created; destroy exactly once.
        unsafe {
            (self.lib.va_destroy_config)(self.dpy, self.config);
            (self.lib.va_terminate)(self.dpy);
        }
    }
}

/// A VCN-decoded frame: geometry + NV12 dma-buf mapped into HIP.
pub struct VcnFrame {
    pub width: u32,
    pub height: u32,
    /// Max sampling factors (2,2 ⇒ 4:2:0; 1,1 ⇒ 4:4:4/gray).
    pub max_h: u8,
    pub max_v: u8,
    pub fourcc: u32,
    pub layers: [ffi::VaDrmPrimeLayer; 4],
    pub num_layers: u32,
    pub mapping: HipMapping,
}

impl VcnFrame {
    /// Device pointer to the start of the exported (single-object) surface.
    pub fn device_ptr(&self) -> *mut c_void {
        self.mapping.ptr()
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
}
