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

use ffi::{VA_ENTRYPOINT_VLD, VA_PROFILE_JPEG_BASELINE, VA_RT_FORMAT_YUV420};
use std::ffi::c_void;
use std::os::fd::{AsRawFd, OwnedFd};

/// DRM render nodes to try, in order. `HIPFIRE_VCN_DRM_NODE` (via
/// `developer_var`) overrides with a single path.
fn render_nodes() -> Vec<String> {
    if let Ok(one) = hipfire_config::developer_var("HIPFIRE_VCN_DRM_NODE") {
        return vec![one];
    }
    (128..132).map(|i| format!("/dev/dri/renderD{i}")).collect()
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
/// A decoded-then-synced VA surface with its context alive: either consumer
/// (dma-buf export or derived-image readback) runs before the guards drop.
struct Submitted<'a> {
    width: u32,
    height: u32,
    max_h: u8,
    max_v: u8,
    surf: VaSurfaceId,
    _surf_guard: SurfGuard<'a>,
    _ctx_guard: CtxGuard<'a>,
}

impl VaSession {
    /// Open the first working render node, initialise VA, create a JPEG
    /// baseline VLD config.
    pub fn open() -> Result<Self, VaError> {
        Self::open_profile(VA_PROFILE_JPEG_BASELINE)
    }

    /// Open the first working render node with a VLD config for `profile`
    /// (see `ffi::VA_PROFILE_*`; video lane uses H.264 High = 7).
    pub(crate) fn open_profile(profile: i32) -> Result<Self, VaError> {
        let mut lib = Some(VaLib::load()?);
        let mut last_err = VaError::NoRenderNode;
        for node in render_nodes() {
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
        })
    }

    pub fn vendor(&self) -> &str {
        &self.vendor
    }

    /// Parse → surface/context → five VA buffers → begin/render/end → sync.
    /// Returns the synced surface with its guards alive for either consumer
    /// (dma-buf export or derived-image readback).
    fn submit(&self, jpeg: &[u8]) -> Result<Submitted<'_>, VaError> {
        let p = parse_for_va(jpeg)?;
        let lib = &self.lib;
        let fail = |op: &'static str, code: i32| VaError::Status {
            op,
            code,
            msg: lib.error_str(code),
        };

        // Surfaces sized to the frame. Investigation knob: HIPFIRE_VCN_LINEAR=1
        // asks the driver for a DRM_FORMAT_MOD_LINEAR surface via
        // VASurfaceAttribDRMFormatModifiers so the exported dma-buf can be
        // read linearly by a compute kernel (zero-copy question).
        let want_linear = std::env::var_os("HIPFIRE_VCN_LINEAR").is_some();
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
        let (attr_ptr, attr_n): (*mut c_void, u32) = if want_linear {
            (attribs.as_mut_ptr().cast(), 1)
        } else {
            (std::ptr::null_mut(), 0)
        };
        let mut surf = 0;
        let mut st = unsafe {
            (lib.va_create_surfaces)(
                self.dpy,
                p.rt_format,
                p.width as u32,
                p.height as u32,
                &mut surf,
                1,
                attr_ptr,
                attr_n,
            )
        };
        if st != VA_STATUS_SUCCESS {
            return Err(fail("vaCreateSurfaces", st));
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
            st =
                unsafe { (lib.va_create_buffer)(self.dpy, ctx, *ty, size, 1, *data, &mut bufs[i]) };
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
        Ok(Submitted {
            width: p.width as u32,
            height: p.height as u32,
            max_h: p.max_h,
            max_v: p.max_v,
            surf,
            _surf_guard,
            _ctx_guard,
        })
    }

    /// Decode one baseline JPEG to an exported dma-buf + HIP mapping.
    ///
    /// Returns the frame geometry, the NV12 layer layout (pitches/offsets),
    /// and the device mapping. All VA objects are destroyed before return;
    /// only the [`HipMapping`] (plus plain geometry) escapes.
    ///
    /// The default surface allocation is tiled (+DCC on gfx12), which a
    /// compute kernel cannot read linearly. With `HIPFIRE_VCN_LINEAR=1` the
    /// surface is requested with `DRM_FORMAT_MOD_LINEAR` and the mapping is
    /// directly kernel-readable; verified 0.000 LSB against
    /// [`decode_jpeg_derived`] on gfx1010/1030/1100/1151/1201 (2026-09-07).
    /// A product path should make linear the only allocation mode.
    pub fn decode_jpeg(&self, jpeg: &[u8]) -> Result<VcnFrame, VaError> {
        let lib = &self.lib;
        let fail = |op: &'static str, code: i32| VaError::Status {
            op,
            code,
            msg: lib.error_str(code),
        };
        let Submitted {
            width,
            height,
            max_h,
            max_v,
            surf,
            _surf_guard,
            _ctx_guard,
        } = self.submit(jpeg)?;
        // `_surf_guard`/`_ctx_guard` stay alive until return.
        let mut st = 0;

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
            width,
            height,
            max_h,
            max_v,
            fourcc: desc.fourcc,
            layers: desc.layers,
            num_layers: desc.num_layers,
            mapping,
        })
    }

    /// Decode one baseline JPEG and read the pixels back through
    /// `vaDeriveImage`/`vaMapBuffer` (the driver resolves tiling/DCC into a
    /// linear CPU mapping). Returns packed NV12 planes. This is the
    /// validation path for VCN decode correctness; it costs a GPU→CPU copy
    /// but no JPEG entropy work on the CPU.
    /// Format-generic derived readback: every plane the driver reports,
    /// packed row-by-row (pitch stripped). Plane geometry follows the
    /// fourcc: NV12 = [Y w*h, CbCr cw*ch*2]; 444P = [Y, Cb, Cr] each w*h;
    /// anything else is returned as-is with its pitch-stripped rows sized
    /// from the image height and pitch, and the caller must know the layout.
    pub fn decode_jpeg_planes(&self, jpeg: &[u8]) -> Result<DerivedPlanes, VaError> {
        let lib = &self.lib;
        let fail = |op: &'static str, code: i32| VaError::Status {
            op,
            code,
            msg: lib.error_str(code),
        };
        let Submitted {
            width,
            height,
            surf,
            _surf_guard,
            _ctx_guard,
            ..
        } = self.submit(jpeg)?;
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

    pub fn decode_jpeg_derived(&self, jpeg: &[u8]) -> Result<DerivedFrame, VaError> {
        let lib = &self.lib;
        let fail = |op: &'static str, code: i32| VaError::Status {
            op,
            code,
            msg: lib.error_str(code),
        };
        let Submitted {
            width,
            height,
            max_h: _,
            max_v: _,
            surf,
            _surf_guard,
            _ctx_guard,
        } = self.submit(jpeg)?;
        // `_surf_guard`/`_ctx_guard` stay alive until return.
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
        if img.format.fourcc != VA_FOURCC_NV12 || img.num_planes != 2 {
            eprintln!(
                "[va-bridge] derived image fourcc=0x{:08x} ({}) planes={} pitches={:?} offsets={:?}",
                img.format.fourcc,
                String::from_utf8_lossy(&img.format.fourcc.to_le_bytes()),
                img.num_planes,
                &img.pitches[..img.num_planes.min(4) as usize],
                &img.offsets[..img.num_planes.min(4) as usize]
            );
            return Err(VaError::Corrupt("derived image is not 2-plane NV12"));
        }
        if img.width as u32 != width || img.height as u32 != height {
            return Err(VaError::Corrupt("derived image geometry mismatch"));
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: out-param is a valid pointer slot; unmapped below.
        st = unsafe { (lib.va_map_buffer)(self.dpy, img.buf, &mut ptr) };
        if st != VA_STATUS_SUCCESS || ptr.is_null() {
            return Err(fail("vaMapBuffer", st));
        }
        // SAFETY: mapped NV12 bytes; rows copied before unmap. Chroma dims
        // are ceil(w/2) x ceil(h/2) so odd-size frames (e.g. 537x529) pack
        // exactly.
        let (y, uv) = unsafe {
            let base = ptr as *const u8;
            let (w, h) = (width as usize, height as usize);
            let (yp, up) = (img.pitches[0] as usize, img.pitches[1] as usize);
            let (yo, uo) = (img.offsets[0] as usize, img.offsets[1] as usize);
            let mut y = vec![0u8; w * h];
            for r in 0..h {
                let src = std::slice::from_raw_parts(base.add(yo + r * yp), w);
                y[r * w..(r + 1) * w].copy_from_slice(src);
            }
            let (cw, chh) = ((w + 1) / 2, (h + 1) / 2);
            let mut uv = vec![0u8; cw * chh * 2];
            for r in 0..chh {
                let src = std::slice::from_raw_parts(base.add(uo + r * up), cw * 2);
                uv[r * cw * 2..(r + 1) * cw * 2].copy_from_slice(src);
            }
            (y, uv)
        };
        Ok(DerivedFrame {
            width,
            height,
            fourcc: img.format.fourcc,
            y_pitch: img.pitches[0],
            uv_pitch: img.pitches[1],
            y,
            uv,
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

/// A VCN-decoded frame read back through `vaDeriveImage`: packed NV12
/// planes with driver-resolved (linear) layout. Validation path for decode
/// correctness when the dma-buf export is tiled/DCC (see
/// [`VaSession::decode_jpeg_derived`]).
/// Format-generic derived readback (see `decode_jpeg_planes`).
pub struct DerivedPlanes {
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    pub planes: Vec<Vec<u8>>,
}

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
