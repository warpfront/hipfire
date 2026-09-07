// Bisect: blank (T0-replica, no submit) vs decoded export+import,
// under HipRuntime HIP vs raw-dlopen HIP.
use va_bridge::{VaLib, VA_STATUS_SUCCESS as OK};

fn blank() {
    use std::os::fd::AsRawFd;
    let lib = VaLib::load().expect("VaLib");
    let f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/dri/renderD128")
        .expect("render node");
    let dpy = unsafe { (lib.va_get_display_drm)(f.as_raw_fd()) };
    let (mut major, mut minor) = (0, 0);
    assert_eq!(
        unsafe { (lib.va_initialize)(dpy, &mut major, &mut minor) },
        OK
    );
    let mut cfg = 0;
    assert_eq!(
        unsafe { (lib.va_create_config)(dpy, 12, 1, std::ptr::null_mut(), 0, &mut cfg) },
        OK
    );
    let mut surf = 0;
    assert_eq!(
        unsafe {
            (lib.va_create_surfaces)(dpy, 1, 946, 1024, &mut surf, 1, std::ptr::null_mut(), 0)
        },
        OK
    );
    let mut ctx = 0;
    assert_eq!(
        unsafe { (lib.va_create_context)(dpy, cfg, 946, 1024, 0, &mut surf, 1, &mut ctx) },
        OK
    );
    let mut desc = va_bridge::VaDrmPrimeDescriptor {
        fourcc: 0,
        width: 0,
        height: 0,
        num_objects: 0,
        objects: [va_bridge::VaDrmPrimeObject::default(); 4],
        num_layers: 0,
        layers: [va_bridge::VaDrmPrimeLayer::default(); 4],
    };
    assert_eq!(
        unsafe {
            (lib.va_export_surface_handle)(
                dpy,
                surf,
                va_bridge::VA_MEM_TYPE_DRM_PRIME_2,
                va_bridge::VA_EXPORT_SURFACE_READ_ONLY,
                (&mut desc as *mut va_bridge::VaDrmPrimeDescriptor).cast(),
            )
        },
        OK
    );
    println!(
        "[bisect] export objects={} size={}",
        desc.num_objects, desc.objects[0].size
    );
    match va_bridge::HipMapping::import_dma_buf(desc.objects[0].fd, desc.objects[0].size as usize) {
        Ok(m) => println!("[bisect] IMPORT OK ptr={:?}", m.ptr()),
        Err(e) => println!("[bisect] IMPORT FAIL: {e}"),
    }
}

fn decoded() {
    let s = va_bridge::VaSession::open().expect("VaSession");
    println!("[bisect] VA up");
    let bytes = std::fs::read(
        "/home/kaden/ClaudeCode/warpfront/wt-vcn/benchmarks/vision/images/general_qa.jpg",
    )
    .expect("fixture");
    match s.decode_jpeg(&bytes) {
        Ok(f) => println!(
            "[bisect] decode+import OK {}x{} ptr={:?} size={}",
            f.width,
            f.height,
            f.device_ptr(),
            f.mapping.size()
        ),
        Err(e) => println!("[bisect] decode FAIL: {e}"),
    }
}

/// VA blank-surface export with NO HIP initialised (T0 order). Surfaces and
/// display are intentionally leaked; the fd holds the dma-buf regardless.
fn blank_export_only() -> (i32, usize) {
    use std::os::fd::AsRawFd;
    let lib = VaLib::load().expect("VaLib");
    let f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/dri/renderD128")
        .expect("render node");
    let dpy = unsafe { (lib.va_get_display_drm)(f.as_raw_fd()) };
    let (mut major, mut minor) = (0, 0);
    assert_eq!(
        unsafe { (lib.va_initialize)(dpy, &mut major, &mut minor) },
        OK
    );
    let mut cfg = 0;
    assert_eq!(
        unsafe { (lib.va_create_config)(dpy, 12, 1, std::ptr::null_mut(), 0, &mut cfg) },
        OK
    );
    let mut surf = 0;
    assert_eq!(
        unsafe {
            (lib.va_create_surfaces)(dpy, 1, 946, 1024, &mut surf, 1, std::ptr::null_mut(), 0)
        },
        OK
    );
    let mut ctx = 0;
    assert_eq!(
        unsafe { (lib.va_create_context)(dpy, cfg, 946, 1024, 0, &mut surf, 1, &mut ctx) },
        OK
    );
    let mut desc = va_bridge::VaDrmPrimeDescriptor {
        fourcc: 0,
        width: 0,
        height: 0,
        num_objects: 0,
        objects: [va_bridge::VaDrmPrimeObject::default(); 4],
        num_layers: 0,
        layers: [va_bridge::VaDrmPrimeLayer::default(); 4],
    };
    assert_eq!(
        unsafe {
            (lib.va_export_surface_handle)(
                dpy,
                surf,
                va_bridge::VA_MEM_TYPE_DRM_PRIME_2,
                va_bridge::VA_EXPORT_SURFACE_READ_ONLY,
                (&mut desc as *mut va_bridge::VaDrmPrimeDescriptor).cast(),
            )
        },
        OK
    );
    std::mem::forget((lib, f));
    (desc.objects[0].fd, desc.objects[0].size as usize)
}

fn main() {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "full".to_string());
    match mode.as_str() {
        // Raw dlopen HIP (T0-C replica) instead of HipRuntime.
        "rawhip" => {
            let lib = unsafe { libloading::Library::new("/opt/rocm/core/lib/libamdhip64.so.7") }
                .expect("dlopen hip");
            unsafe {
                let set_d: libloading::Symbol<unsafe extern "C" fn(i32) -> i32> =
                    lib.get(b"hipSetDevice").expect("setDevice");
                assert_eq!(set_d(0), 0);
            }
            std::mem::forget(lib);
            println!("[bisect] raw HIP up");
            blank();
        }
        "vafirst" => {
            // T0 order: VA export with NO HIP initialised, then setDevice, then import.
            let (fd, size) = blank_export_only();
            println!("[bisect] exported fd={fd} size={size}, now bringing HIP up");
            let lib = unsafe { libloading::Library::new("/opt/rocm/core/lib/libamdhip64.so.7") }
                .expect("dlopen hip");
            unsafe {
                let set_d: libloading::Symbol<unsafe extern "C" fn(i32) -> i32> =
                    lib.get(b"hipSetDevice").expect("setDevice");
                assert_eq!(set_d(0), 0);
            }
            std::mem::forget(lib);
            match va_bridge::HipMapping::import_dma_buf(fd, size) {
                Ok(m) => println!("[bisect vafirst] IMPORT OK ptr={:?}", m.ptr()),
                Err(e) => println!("[bisect vafirst] IMPORT FAIL: {e}"),
            }
        }
        "blank" => {
            let hip = hip_bridge::HipRuntime::load().expect("HipRuntime::load");
            hip.set_device(0).expect("set_device");
            println!("[bisect] HIP up via HipRuntime");
            blank();
        }
        _ => {
            let hip = hip_bridge::HipRuntime::load().expect("HipRuntime::load");
            hip.set_device(0).expect("set_device");
            println!("[bisect] HIP up via HipRuntime");
            decoded();
        }
    }
}
