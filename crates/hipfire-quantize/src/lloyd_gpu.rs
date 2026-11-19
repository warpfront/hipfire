// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Stream (2) local research: GPU free fit for MQ4V2-Lloyd. Default off.
// The CPU path stays the reference; this path refuses cleanly on any GPU
// failure instead of falling back silently.

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};
use std::cell::RefCell;
use std::sync::OnceLock;

const KERNEL_SRC: &str = include_str!("../kernels/lloyd_fit.hip");
const CHUNK: usize = 1024;
const LANES: usize = 34;
const MAX_ITERS: usize = 25;
const BLOCK: u32 = 256;

thread_local! {
    // Gpu is !Send (raw HIP handles); the quantize pipeline is single-threaded.
    static GPU: RefCell<Option<Gpu>> = const { RefCell::new(None) };
}
static KERNEL_READY: OnceLock<Result<(), String>> = OnceLock::new();

/// `HIPFIRE_QUANT_GPU=1` opts in to the device path. Off by default.
pub(crate) fn gpu_lloyd_enabled() -> bool {
    static P: OnceLock<bool> = OnceLock::new();
    *P.get_or_init(|| hipfire_config::developer_var("HIPFIRE_QUANT_GPU").as_deref() == Ok("1"))
}

fn with_gpu<R>(f: impl FnOnce(&mut Gpu) -> Result<R, String>) -> Result<R, String> {
    GPU.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(Gpu::init().map_err(|e| format!("GPU init failed: {e:?}"))?);
        }
        f(slot.as_mut().expect("just initialized"))
    })
}

unsafe fn as_bytes<T>(v: &[T]) -> &[u8] {
    std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
}

/// Run the unconstrained 25-iteration fit on device. Returns the levels and
/// the per-iteration MSE history, same shape as `lloyd_max_fit`.
pub(crate) fn gpu_lloyd_free_fit(
    vals: &[f64],
    wts: &[f64],
) -> Result<([f64; 16], Vec<f64>), String> {
    debug_assert_eq!(vals.len(), wts.len());
    let n = vals.len();
    if n == 0 {
        return Ok(([0.0; 16], vec![0.0]));
    }

    with_gpu(|gpu| {
        let ready = KERNEL_READY.get_or_init(|| {
            gpu.ensure_kernel_public("lloyd_fit", KERNEL_SRC, "lloyd_partial")
                .map(|_| ())
                .map_err(|e| format!("lloyd_fit JIT/link failed: {e:?}"))
        });
        ready.as_ref().map_err(|e| e.clone())?;

        let nchunks = n.div_ceil(CHUNK);
        let d_vals = gpu
            .alloc_tensor(&[n * 8], DType::Raw)
            .map_err(|e| format!("alloc vals: {e:?}"))?;
        let d_wts = gpu
            .alloc_tensor(&[n * 8], DType::Raw)
            .map_err(|e| format!("alloc wts: {e:?}"))?;
        let d_levels = gpu
            .alloc_tensor(&[16 * 8], DType::Raw)
            .map_err(|e| format!("alloc levels: {e:?}"))?;
        let d_partials = gpu
            .alloc_tensor(&[nchunks * LANES * 8], DType::Raw)
            .map_err(|e| format!("alloc partials: {e:?}"))?;

        gpu.hip
            .memcpy_htod(&d_vals.buf, unsafe { as_bytes(vals) })
            .map_err(|e| format!("htod vals: {e:?}"))?;
        gpu.hip
            .memcpy_htod(&d_wts.buf, unsafe { as_bytes(wts) })
            .map_err(|e| format!("htod wts: {e:?}"))?;

        let mut partials = vec![0.0f64; nchunks * LANES];
        let mut levels = [0.0f64; 16];
        for (i, l) in levels.iter_mut().enumerate() {
            *l = i as f64;
        }
        let mut mse_hist = Vec::with_capacity(MAX_ITERS + 1);
        let grid = [
            ((nchunks + BLOCK as usize - 1) / BLOCK as usize) as u32,
            1,
            1,
        ];

        for _it in 0..MAX_ITERS {
            gpu.hip
                .memcpy_htod(&d_levels.buf, unsafe { as_bytes(&levels) })
                .map_err(|e| format!("htod levels: {e:?}"))?;

            let mut b = KernargBlob::new();
            b.push_ptr(d_vals.buf.as_ptr() as *const std::ffi::c_void);
            b.push_ptr(d_wts.buf.as_ptr() as *const std::ffi::c_void);
            b.push_ptr(d_levels.buf.as_ptr() as *const std::ffi::c_void);
            b.push_ptr(d_partials.buf.as_ptr() as *const std::ffi::c_void);
            b.push_u64(n as u64);
            gpu.launch_kernel_blob("lloyd_partial", grid, [BLOCK, 1, 1], 0, b.as_mut_slice())
                .map_err(|e| format!("launch lloyd_partial: {e:?}"))?;
            gpu.hip
                .device_synchronize()
                .map_err(|e| format!("sync: {e:?}"))?;
            gpu.hip
                .memcpy_dtoh(
                    unsafe {
                        std::slice::from_raw_parts_mut(
                            partials.as_mut_ptr() as *mut u8,
                            partials.len() * 8,
                        )
                    },
                    &d_partials.buf,
                )
                .map_err(|e| format!("dtoh partials: {e:?}"))?;

            let mut sw = [0.0f64; 16];
            let mut sv = [0.0f64; 16];
            let mut sse = 0.0f64;
            let mut wsum = 0.0f64;
            for c in 0..nchunks {
                let p = &partials[c * LANES..(c + 1) * LANES];
                for k in 0..16 {
                    sw[k] += p[k];
                    sv[k] += p[16 + k];
                }
                sse += p[32];
                wsum += p[33];
            }
            mse_hist.push(if wsum > 0.0 { sse / wsum } else { 0.0 });

            let mut next = levels;
            for k in 0..16 {
                if sw[k] > 0.0 {
                    next[k] = sv[k] / sw[k];
                }
            }
            next.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut moved = 0.0f64;
            for k in 0..16 {
                moved = moved.max((next[k] - levels[k]).abs());
            }
            levels = next;
            if moved < 1e-6 {
                break;
            }
        }

        Ok((levels, mse_hist))
    })
}
