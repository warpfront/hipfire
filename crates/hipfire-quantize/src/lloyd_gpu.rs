// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Stream (2) local research: GPU free fit for MQ4V2-Lloyd. Default off.
// The CPU path stays the reference; this path refuses cleanly on any GPU
// failure instead of falling back silently.

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::cell::RefCell;
use std::sync::OnceLock;

const KERNEL_SRC: &str = include_str!("../kernels/lloyd_fit.hip");
const CHUNK: usize = 1024;
const LANES: usize = 34;
const MAX_ITERS: usize = 25;
const BLOCK: u32 = 256;
/// `coordinate_descent_polish` subsamples to at most this many pairs.
const POLISH_TARGET: usize = 256_000;

/// Device buffers reused across tensors. Capacity is in f64 elements;
/// reallocation happens only when a larger pool appears.
struct GpuBuffers {
    vals: GpuTensor,
    wts: GpuTensor,
    levels: GpuTensor,
    partials: GpuTensor,
    capacity: usize,
    polish_vals: GpuTensor,
    polish_wts: GpuTensor,
    polish_levels: GpuTensor,
    polish_partials: GpuTensor,
    polish_n: usize,
}

impl GpuBuffers {
    fn new(gpu: &mut Gpu, n: usize) -> Result<Self, String> {
        let nchunks = n.div_ceil(CHUNK);
        let vals = gpu
            .alloc_tensor(&[n * 8], DType::Raw)
            .map_err(|e| format!("alloc vals: {e:?}"))?;
        let wts = gpu
            .alloc_tensor(&[n * 8], DType::Raw)
            .map_err(|e| format!("alloc wts: {e:?}"))?;
        let levels = gpu
            .alloc_tensor(&[16 * 8], DType::Raw)
            .map_err(|e| format!("alloc levels: {e:?}"))?;
        let partials = gpu
            .alloc_tensor(&[nchunks * LANES * 8], DType::Raw)
            .map_err(|e| format!("alloc partials: {e:?}"))?;
        let polish_chunks = POLISH_TARGET.div_ceil(CHUNK);
        let polish_vals = gpu
            .alloc_tensor(&[POLISH_TARGET * 8], DType::Raw)
            .map_err(|e| format!("alloc polish vals: {e:?}"))?;
        let polish_wts = gpu
            .alloc_tensor(&[POLISH_TARGET * 8], DType::Raw)
            .map_err(|e| format!("alloc polish wts: {e:?}"))?;
        let polish_levels = gpu
            .alloc_tensor(&[4 * 16 * 8], DType::Raw)
            .map_err(|e| format!("alloc polish levels: {e:?}"))?;
        let polish_partials = gpu
            .alloc_tensor(&[polish_chunks * 8 * 8], DType::Raw)
            .map_err(|e| format!("alloc polish partials: {e:?}"))?;
        Ok(Self {
            vals,
            wts,
            levels,
            partials,
            capacity: n,
            polish_vals,
            polish_wts,
            polish_levels,
            polish_partials,
            polish_n: 0,
        })
    }
}

/// Buffers drop before the Gpu that owns the runtime handles.
struct GpuState {
    buffers: Option<GpuBuffers>,
    gpu: Gpu,
}

thread_local! {
    // Gpu is !Send (raw HIP handles); the quantize pipeline is single-threaded.
    static GPU: RefCell<Option<GpuState>> = const { RefCell::new(None) };
}
static KERNEL_READY: OnceLock<Result<(), String>> = OnceLock::new();
static POLISH_KERNEL_READY: OnceLock<Result<(), String>> = OnceLock::new();

/// `HIPFIRE_QUANT_GPU=1` opts in to the device path. Off by default.
pub(crate) fn gpu_lloyd_enabled() -> bool {
    static P: OnceLock<bool> = OnceLock::new();
    *P.get_or_init(|| hipfire_config::developer_var("HIPFIRE_QUANT_GPU").as_deref() == Ok("1"))
}

fn with_gpu<R>(f: impl FnOnce(&mut GpuState) -> Result<R, String>) -> Result<R, String> {
    GPU.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(GpuState {
                buffers: None,
                gpu: Gpu::init().map_err(|e| format!("GPU init failed: {e:?}"))?,
            });
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

    with_gpu(|state| {
        let GpuState { buffers, gpu } = state;
        let ready = KERNEL_READY.get_or_init(|| {
            gpu.ensure_kernel_public("lloyd_fit", KERNEL_SRC, "lloyd_partial")
                .map(|_| ())
                .map_err(|e| format!("lloyd_fit JIT/link failed: {e:?}"))
        });
        ready.as_ref().map_err(|e| e.clone())?;

        if buffers.as_ref().map_or(true, |b| b.capacity < n) {
            *buffers = Some(GpuBuffers::new(gpu, n)?);
        }
        let bufs = buffers.as_mut().expect("just sized");

        gpu.hip
            .memcpy_htod(&bufs.vals.buf, unsafe { as_bytes(vals) })
            .map_err(|e| format!("htod vals: {e:?}"))?;
        gpu.hip
            .memcpy_htod(&bufs.wts.buf, unsafe { as_bytes(wts) })
            .map_err(|e| format!("htod wts: {e:?}"))?;

        // Chunk count is per-tensor. The reused partials buffer is sized
        // from the largest pool seen, but only this tensor's chunks are live.
        let nchunks = n.div_ceil(CHUNK);
        let mut partials = vec![0.0f64; nchunks * LANES];
        let mut levels = [0.0f64; 16];
        for (i, l) in levels.iter_mut().enumerate() {
            *l = i as f64;
        }
        let mut mse_hist = Vec::with_capacity(MAX_ITERS + 1);
        let grid = [nchunks.div_ceil(BLOCK as usize) as u32, 1, 1];

        for _it in 0..MAX_ITERS {
            gpu.hip
                .memcpy_htod(&bufs.levels.buf, unsafe { as_bytes(&levels) })
                .map_err(|e| format!("htod levels: {e:?}"))?;

            let mut b = KernargBlob::new();
            b.push_ptr(bufs.vals.buf.as_ptr() as *const std::ffi::c_void);
            b.push_ptr(bufs.wts.buf.as_ptr() as *const std::ffi::c_void);
            b.push_ptr(bufs.levels.buf.as_ptr() as *const std::ffi::c_void);
            b.push_ptr(bufs.partials.buf.as_ptr() as *const std::ffi::c_void);
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
                    &bufs.partials.buf,
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

/// Stage this tensor's polish subsample on device. Called once per tensor
/// before the polish acceptance loop.
pub(crate) fn gpu_lloyd_polish_init(vals: &[f64], wts: &[f64]) -> Result<(), String> {
    debug_assert_eq!(vals.len(), wts.len());
    if vals.len() > POLISH_TARGET {
        return Err(format!(
            "polish pool {} exceeds {POLISH_TARGET}",
            vals.len()
        ));
    }
    with_gpu(|state| {
        let GpuState { buffers, gpu } = state;
        let ready = POLISH_KERNEL_READY.get_or_init(|| {
            gpu.ensure_kernel_public("lloyd_fit", KERNEL_SRC, "lloyd_polish_partial")
                .map(|_| ())
                .map_err(|e| format!("lloyd_polish_partial JIT/link failed: {e:?}"))
        });
        ready.as_ref().map_err(|e| e.clone())?;
        if buffers.is_none() {
            *buffers = Some(GpuBuffers::new(gpu, 1)?);
        }
        let bufs = buffers.as_mut().expect("just sized");
        gpu.hip
            .memcpy_htod(&bufs.polish_vals.buf, unsafe { as_bytes(vals) })
            .map_err(|e| format!("htod polish vals: {e:?}"))?;
        gpu.hip
            .memcpy_htod(&bufs.polish_wts.buf, unsafe { as_bytes(wts) })
            .map_err(|e| format!("htod polish wts: {e:?}"))?;
        bufs.polish_n = vals.len();
        Ok(())
    })
}

/// Score four candidate level-sets in one device pass. Returns the same
/// values `weighted_mse_4` would, up to fixed-order chunk reduction.
pub(crate) fn gpu_lloyd_polish_scores(sets: &[[f64; 16]; 4]) -> Result<[f64; 4], String> {
    with_gpu(|state| {
        let GpuState { buffers, gpu } = state;
        let bufs = buffers
            .as_mut()
            .ok_or_else(|| "polish device not initialized".to_string())?;
        let n = bufs.polish_n;
        if n == 0 {
            return Ok([0.0; 4]);
        }
        let mut flat = [0.0f64; 64];
        for (c, set) in sets.iter().enumerate() {
            flat[c * 16..(c + 1) * 16].copy_from_slice(set);
        }
        gpu.hip
            .memcpy_htod(&bufs.polish_levels.buf, unsafe { as_bytes(&flat) })
            .map_err(|e| format!("htod polish levels: {e:?}"))?;

        let nchunks = n.div_ceil(CHUNK);
        let grid = [nchunks.div_ceil(BLOCK as usize) as u32, 1, 1];
        let mut b = KernargBlob::new();
        b.push_ptr(bufs.polish_vals.buf.as_ptr() as *const std::ffi::c_void);
        b.push_ptr(bufs.polish_wts.buf.as_ptr() as *const std::ffi::c_void);
        b.push_ptr(bufs.polish_levels.buf.as_ptr() as *const std::ffi::c_void);
        b.push_ptr(bufs.polish_partials.buf.as_ptr() as *const std::ffi::c_void);
        b.push_u64(n as u64);
        gpu.launch_kernel_blob(
            "lloyd_polish_partial",
            grid,
            [BLOCK, 1, 1],
            0,
            b.as_mut_slice(),
        )
        .map_err(|e| format!("launch lloyd_polish_partial: {e:?}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;

        let mut partials = vec![0.0f64; nchunks * 8];
        gpu.hip
            .memcpy_dtoh(
                unsafe {
                    std::slice::from_raw_parts_mut(
                        partials.as_mut_ptr() as *mut u8,
                        partials.len() * 8,
                    )
                },
                &bufs.polish_partials.buf,
            )
            .map_err(|e| format!("dtoh polish partials: {e:?}"))?;

        let mut sse = [0.0f64; 4];
        let mut wsum = [0.0f64; 4];
        for c in 0..nchunks {
            let p = &partials[c * 8..(c + 1) * 8];
            for k in 0..4 {
                sse[k] += p[k];
                wsum[k] += p[4 + k];
            }
        }
        let mut out = [0.0f64; 4];
        for k in 0..4 {
            out[k] = if wsum[k] > 0.0 { sse[k] / wsum[k] } else { 0.0 };
        }
        Ok(out)
    })
}
