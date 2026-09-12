// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Raw VA-API bindings for the VCN JPEG experiment (`experiment/vcn-jpeg`).
//!
//! `libva.so.2` + `libva-drm.so.2` are opened with `libloading` (same pattern
//! as `hip-bridge/src/rccl.rs`); there is deliberately no link-time
//! dependency. Absence of the libraries is a recoverable [`VaError`], never a
//! build failure.
//!
//! Struct layouts are transcribed from libva master (VA-API 1.23;
//! `va/va.h` @ `6b07f71`, `va/va_dec_jpeg.h`, `va/va_drmcommon.h`). libva 2.x
//! ABI is stable; every transcribed struct carries a `const` size assert
//! against the header-derived value so drift fails at compile time.

use libloading::Library;
use std::ffi::c_void;

// ── IDs ─────────────────────────────────────────────────────────────────
pub type VaDisplay = *mut c_void;
pub type VaConfigId = u32;
pub type VaContextId = u32;
pub type VaSurfaceId = u32;
pub type VaBufferId = u32;

// ── Profiles / entrypoints / formats (va/va.h) ──────────────────────────
pub const VA_PROFILE_JPEG_BASELINE: i32 = 12;
pub const VA_ENTRYPOINT_VLD: i32 = 1;
pub const VA_RT_FORMAT_YUV420: u32 = 0x0000_0001;
pub const VA_RT_FORMAT_YUV422: u32 = 0x0000_0002;
pub const VA_RT_FORMAT_YUV444: u32 = 0x0000_0004;
pub const VA_RT_FORMAT_YUV400: u32 = 0x0000_0010;
pub const VA_STATUS_SUCCESS: i32 = 0;

// ── Buffer types (va/va.h `VABufferType`) ───────────────────────────────
pub const VA_PIC_PARAM_TYPE: i32 = 0;
pub const VA_IQ_MATRIX_TYPE: i32 = 1;
pub const VA_SLICE_PARAM_TYPE: i32 = 4;
pub const VA_SLICE_DATA_TYPE: i32 = 5;
pub const VA_HUFFMAN_TABLE_TYPE: i32 = 12;
pub const VA_SLICE_DATA_FLAG_ALL: u32 = 0x00;

// ── Export (va/va.h + va/va_drmcommon.h) ────────────────────────────────
pub const VA_MEM_TYPE_DRM_PRIME_2: u32 = 0x4000_0000;
pub const VA_EXPORT_SURFACE_READ_ONLY: u32 = 0x0001;
pub const VA_EXPORT_SURFACE_SEPARATE_LAYERS: u32 = 0x0004;
pub const VA_FOURCC_NV12: u32 = 0x3231_564E;

// ── Transcribed structs ─────────────────────────────────────────────────
// NOTE: `components[255]` is the header's fixed array (up to 255 frame
// components per ISO 10918-1 B.2.2); only the first `num_components` slots
// are filled on submit.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VaJpegPicParam {
    pub picture_width: u16,
    pub picture_height: u16,
    pub components: [VaJpegComponent; 255],
    pub num_components: u8,
    pub color_space: u8, // 0 = YUV
    pub rotation: u32,
    pub crop_x: i16,
    pub crop_y: i16,
    pub crop_width: u16,
    pub crop_height: u16,
    pub va_reserved: [u32; 5], // VA_PADDING_MEDIUM(8) - 3
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VaJpegComponent {
    pub component_id: u8,
    pub h_sampling_factor: u8,
    pub v_sampling_factor: u8,
    pub quantiser_table_selector: u8,
}
// 4 + 255*4 + 2 + 2(pad) + 4 + 8 + 20 = 1060 (C ground truth, gcc LP64)
const _: () = assert!(std::mem::size_of::<VaJpegPicParam>() == 1060);

// --- Surface attributes (libva 2.23 va.h) ------------------------------------
// VASurfaceAttribType: DRMFormatModifiers = 9. VA_SURFACE_ATTRIB_SETTABLE = 2.
// VAGenericValueTypePointer = 3.
pub const VA_SURFACE_ATTRIB_DRM_FORMAT_MODIFIERS: i32 = 9;
pub const VA_SURFACE_ATTRIB_SETTABLE: u32 = 0x2;
pub const VA_GENERIC_VALUE_TYPE_POINTER: i32 = 3;
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;

#[repr(C)]
#[derive(Clone, Copy)]
pub union VaGenericValueUnion {
    pub i: i32,
    pub f: f32,
    pub p: *mut c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VaGenericValue {
    pub ty: i32,
    pub value: VaGenericValueUnion,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VaSurfaceAttrib {
    pub ty: i32,
    pub flags: u32,
    pub value: VaGenericValue,
}
#[repr(C)]
pub struct VaDrmFormatModifierList {
    pub num_modifiers: u32,
    pub modifiers: *mut u64,
}
const _: () = assert!(std::mem::size_of::<VaGenericValue>() == 16);
const _: () = assert!(std::mem::size_of::<VaSurfaceAttrib>() == 24);
const _: () = assert!(std::mem::size_of::<VaDrmFormatModifierList>() == 16);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VaJpegIQMatrix {
    pub load_quantiser_table: [u8; 4],
    pub quantiser_table: [[u8; 64]; 4], // zig-zag order, as in DQT
    pub va_reserved: [u32; 4],
}
// 4 + 256 + 16 = 276
const _: () = assert!(std::mem::size_of::<VaJpegIQMatrix>() == 276);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VaJpegHuffmanTable {
    pub num_dc_codes: [u8; 16],
    pub dc_values: [u8; 12],
    pub num_ac_codes: [u8; 16],
    pub ac_values: [u8; 162],
    pub pad: [u8; 2],
}
// 16+12+16+162+2 = 208
const _: () = assert!(std::mem::size_of::<VaJpegHuffmanTable>() == 208);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VaJpegHuffmanBuffer {
    pub load_huffman_table: [u8; 2],
    pub huffman_table: [VaJpegHuffmanTable; 2], // indexed by Th
    pub va_reserved: [u32; 4],
}
// 2 + 2*208 + 2(pad) + 16 = 436 (C ground truth, gcc LP64)
const _: () = assert!(std::mem::size_of::<VaJpegHuffmanBuffer>() == 436);

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VaJpegSliceComponent {
    pub component_selector: u8,
    pub dc_table_selector: u8,
    pub ac_table_selector: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VaJpegSliceParam {
    pub slice_data_size: u32,
    pub slice_data_offset: u32,
    pub slice_data_flag: u32,
    pub slice_horizontal_position: u32,
    pub slice_vertical_position: u32,
    pub components: [VaJpegSliceComponent; 4],
    pub num_components: u8,
    // 1 byte pad before u16 (C layout)
    pub _pad: u8,
    pub restart_interval: u16,
    pub num_mcus: u32,
    pub va_reserved: [u32; 4],
}
// 12 + 8 + 12 + 1 + 1(pad) + 2 + 4 + 16 = 56
const _: () = assert!(std::mem::size_of::<VaJpegSliceParam>() == 56);

/// `VADRMPRIMESurfaceDescriptor` (va/va_drmcommon.h). Only the fields the
/// experiment reads are named; the rest keeps the C layout. Total:
/// 4*4 + 4*(4+4+8) + 4 + 4*(4+4+16+16+16) = 16+64+4+208 = 292.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VaDrmPrimeDescriptor {
    pub fourcc: u32,
    pub width: u32,
    pub height: u32,
    pub num_objects: u32,
    pub objects: [VaDrmPrimeObject; 4],
    pub num_layers: u32,
    pub layers: [VaDrmPrimeLayer; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VaDrmPrimeObject {
    pub fd: i32,
    pub size: u32,
    pub drm_format_modifier: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VaDrmPrimeLayer {
    pub drm_format: u32,
    pub num_planes: u32,
    pub object_index: [u32; 4],
    pub offset: [u32; 4],
    pub pitch: [u32; 4],
}
// 16 + 64 + 4 + 224 = 308, +4 tail pad to align 8 = 312 (C ground truth, gcc LP64)
const _: () = assert!(std::mem::size_of::<VaDrmPrimeDescriptor>() == 312);
// ── Derived images (va/va.h `VAImage`; tiling/DCC-resolving CPU readback) ──
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VaImageFormat {
    pub fourcc: u32,
    pub byte_order: u32,
    pub bits_per_pixel: u32,
    pub depth: u32,
    pub red_mask: u32,
    pub green_mask: u32,
    pub blue_mask: u32,
    pub alpha_mask: u32,
    pub va_reserved: [u32; 4], // VA_PADDING_LOW
}
// 8*4 + 16 = 48
const _: () = assert!(std::mem::size_of::<VaImageFormat>() == 48);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VaImage {
    pub image_id: u32,
    pub format: VaImageFormat,
    pub buf: u32,
    pub width: u16,
    pub height: u16,
    pub data_size: u32,
    pub num_planes: u32,
    pub pitches: [u32; 3],
    pub offsets: [u32; 3],
    pub num_palette_entries: i32,
    pub entry_bytes: i32,
    pub component_order: [i8; 4],
    pub va_reserved: [u32; 4], // VA_PADDING_LOW
}
// 4+48+4+2+2+4+4+12+12+4+4+4+16 = 120 (C ground truth, gcc LP64)
const _: () = assert!(std::mem::size_of::<VaImage>() == 120);

// ── Loaded library ──────────────────────────────────────────────────────
macro_rules! fn_ty {
    ($name:ident($($arg:ty),* $(,)?) -> $ret:ty) => {
        unsafe extern "C" fn($($arg),*) -> $ret
    };
}

/// `libva.so.2` + `libva-drm.so.2` with the VCN-JPEG subset resolved.
/// Mirrors `hip-bridge/src/rccl.rs`: construction failure ⇒ caller falls
/// back (here: to the `libjpeg-turbo-rs` CPU oracle).
pub struct VaLib {
    _va: Library,
    _va_drm: Library,
    pub va_get_display_drm: fn_ty!(va_get_display_drm(i32) -> VaDisplay),
    pub va_initialize: fn_ty!(va_initialize(VaDisplay, *mut i32, *mut i32) -> i32),
    pub va_terminate: fn_ty!(va_terminate(VaDisplay) -> i32),
    pub va_error_str: fn_ty!(va_error_str(i32) -> *const i8),
    pub va_query_vendor: fn_ty!(va_query_vendor(VaDisplay) -> *const i8),
    pub va_max_profiles: fn_ty!(va_max_profiles(VaDisplay) -> i32),
    pub va_query_profiles: fn_ty!(va_query_profiles(VaDisplay, *mut i32, *mut i32) -> i32),
    pub va_create_config:
        fn_ty!(va_create_config(VaDisplay, i32, i32, *mut c_void, i32, *mut VaConfigId) -> i32),
    pub va_destroy_config: fn_ty!(va_destroy_config(VaDisplay, VaConfigId) -> i32),
    pub va_create_surfaces: fn_ty!(
        va_create_surfaces(
            VaDisplay,
            u32,
            u32,
            u32,
            *mut VaSurfaceId,
            u32,
            *mut c_void,
            u32,
        ) -> i32
    ),
    pub va_destroy_surfaces: fn_ty!(va_destroy_surfaces(VaDisplay, *mut VaSurfaceId, i32) -> i32),
    pub va_create_context: fn_ty!(
        va_create_context(
            VaDisplay,
            VaConfigId,
            i32,
            i32,
            i32,
            *mut VaSurfaceId,
            i32,
            *mut VaContextId,
        ) -> i32
    ),
    pub va_destroy_context: fn_ty!(va_destroy_context(VaDisplay, VaContextId) -> i32),
    pub va_create_buffer: fn_ty!(
        va_create_buffer(
            VaDisplay,
            VaContextId,
            i32,
            u32,
            u32,
            *mut c_void,
            *mut VaBufferId,
        ) -> i32
    ),
    pub va_destroy_buffer: fn_ty!(va_destroy_buffer(VaDisplay, VaBufferId) -> i32),
    pub va_begin_picture: fn_ty!(va_begin_picture(VaDisplay, VaContextId, VaSurfaceId) -> i32),
    pub va_render_picture:
        fn_ty!(va_render_picture(VaDisplay, VaContextId, *mut VaBufferId, i32) -> i32),
    pub va_end_picture: fn_ty!(va_end_picture(VaDisplay, VaContextId) -> i32),
    pub va_sync_surface: fn_ty!(va_sync_surface(VaDisplay, VaSurfaceId) -> i32),
    pub va_export_surface_handle:
        fn_ty!(va_export_surface_handle(VaDisplay, VaSurfaceId, u32, u32, *mut c_void) -> i32),
    pub va_derive_image: fn_ty!(va_derive_image(VaDisplay, VaSurfaceId, *mut VaImage) -> i32),
    pub va_map_buffer: fn_ty!(va_map_buffer(VaDisplay, VaBufferId, *mut *mut c_void) -> i32),
    pub va_unmap_buffer: fn_ty!(va_unmap_buffer(VaDisplay, VaBufferId) -> i32),
    pub va_destroy_image: fn_ty!(va_destroy_image(VaDisplay, u32) -> i32),
}

impl VaLib {
    /// dlopen `libva.so.2` + `libva-drm.so.2` from the loader path and
    /// resolve the VCN-JPEG subset. `HIPFIRE_VCN_LIBVA_PATH` (read via
    /// `developer_var`) overrides the `libva.so.2` soname for dev.
    pub fn load() -> Result<Self, VaError> {
        let va_name = hipfire_config::developer_var("HIPFIRE_VCN_LIBVA_PATH")
            .unwrap_or_else(|_| "libva.so.2".to_string());
        // SAFETY: dlopen of a system media library; no Rust invariants involved.
        let va = unsafe { Library::new(&va_name) }.map_err(|e| VaError::Dlopen {
            lib: va_name.clone(),
            msg: e.to_string(),
        })?;
        // SAFETY: same.
        let va_drm = unsafe { Library::new("libva-drm.so.2") }.map_err(|e| VaError::Dlopen {
            lib: "libva-drm.so.2".to_string(),
            msg: e.to_string(),
        })?;
        // SAFETY: each symbol is looked up once with its exact C type below.
        unsafe {
            let sym = |lib: &Library, name: &[u8]| -> Result<*mut c_void, VaError> {
                lib.get::<*mut c_void>(name)
                    .map(|s| *s)
                    .map_err(|_| VaError::MissingSymbol {
                        symbol: String::from_utf8_lossy(name).into_owned(),
                    })
            };
            macro_rules! resolve {
                ($lib:expr, $sym:literal, $ty:ty) => {
                    std::mem::transmute::<*mut c_void, $ty>(sym($lib, $sym)?)
                };
            }
            Ok(Self {
                va_get_display_drm: resolve!(&va_drm, b"vaGetDisplayDRM", _),
                va_initialize: resolve!(&va, b"vaInitialize", _),
                va_terminate: resolve!(&va, b"vaTerminate", _),
                va_error_str: resolve!(&va, b"vaErrorStr", _),
                va_query_vendor: resolve!(&va, b"vaQueryVendorString", _),
                va_max_profiles: resolve!(&va, b"vaMaxNumProfiles", _),
                va_query_profiles: resolve!(&va, b"vaQueryConfigProfiles", _),
                va_create_config: resolve!(&va, b"vaCreateConfig", _),
                va_destroy_config: resolve!(&va, b"vaDestroyConfig", _),
                va_create_surfaces: resolve!(&va, b"vaCreateSurfaces", _),
                va_destroy_surfaces: resolve!(&va, b"vaDestroySurfaces", _),
                va_create_context: resolve!(&va, b"vaCreateContext", _),
                va_destroy_context: resolve!(&va, b"vaDestroyContext", _),
                va_create_buffer: resolve!(&va, b"vaCreateBuffer", _),
                va_destroy_buffer: resolve!(&va, b"vaDestroyBuffer", _),
                va_begin_picture: resolve!(&va, b"vaBeginPicture", _),
                va_render_picture: resolve!(&va, b"vaRenderPicture", _),
                va_end_picture: resolve!(&va, b"vaEndPicture", _),
                va_sync_surface: resolve!(&va, b"vaSyncSurface", _),
                va_export_surface_handle: resolve!(&va, b"vaExportSurfaceHandle", _),
                va_derive_image: resolve!(&va, b"vaDeriveImage", _),
                va_map_buffer: resolve!(&va, b"vaMapBuffer", _),
                va_unmap_buffer: resolve!(&va, b"vaUnmapBuffer", _),
                va_destroy_image: resolve!(&va, b"vaDestroyImage", _),
                _va: va,
                _va_drm: va_drm,
            })
        }
    }

    pub fn error_str(&self, status: i32) -> String {
        // SAFETY: vaErrorStr returns a static C string for any status.
        let p = unsafe { (self.va_error_str)(status) };
        if p.is_null() {
            return format!("VA status 0x{status:x}");
        }
        // SAFETY: libva guarantees NUL-terminated static storage.
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    }
}

// `Library` is `Send`/`Sync`-neutral; the session is explicitly single-threaded.
unsafe impl Send for VaLib {}
unsafe impl Sync for VaLib {}

/// Recoverable VA-API failure. Callers fall back to the CPU oracle.
#[derive(Debug)]
pub enum VaError {
    Dlopen {
        lib: String,
        msg: String,
    },
    MissingSymbol {
        symbol: String,
    },
    NoRenderNode,
    Status {
        op: &'static str,
        code: i32,
        msg: String,
    },
    Unsupported(&'static str),
    Corrupt(&'static str),
}

impl std::fmt::Display for VaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dlopen { lib, msg } => write!(f, "va-bridge: dlopen {lib} failed: {msg}"),
            Self::MissingSymbol { symbol } => write!(f, "va-bridge: missing symbol {symbol}"),
            Self::NoRenderNode => write!(f, "va-bridge: no usable /dev/dri/renderD* node"),
            Self::Status { op, code, msg } => {
                write!(f, "va-bridge: {op} failed: 0x{code:x} ({msg})")
            }
            Self::Unsupported(s) => write!(f, "va-bridge: unsupported JPEG: {s}"),
            Self::Corrupt(s) => write!(f, "va-bridge: corrupt JPEG: {s}"),
        }
    }
}
impl std::error::Error for VaError {}
