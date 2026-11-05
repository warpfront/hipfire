//! gfx1151 iGPU cold arm for the Qwen4 D2 split (heterogeneous expert split).
//!
//! A drop-in for [`crate::cpu_exec::moe_cpu_host_slot_outputs`]: the same
//! arguments and the same output representation (raw F32 down values at
//! `out[s * dim .. (s + 1) * dim]` for every host slot `s`), computed on a Strix
//! Halo iGPU instead of host CPU cores.
//!
//! It runs the kernels the gfx1151 single-device Qwen4 decode runs for the
//! routed experts of a layer (recorded from a Strix-alone decode, 48 layers x
//! 2 tokens):
//!
//! 1. `gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched` (qt44 gate/up,
//!    launched by [`rdna_compute::Gpu::gemv_mq4g256v2_moe_gate_up_top10_indexed_batched`])
//! 2. `mq_rotate_x_128_v2_silu_bf16`
//!    ([`rdna_compute::Gpu::silu_mul_bf16_rt_rotate_x_mq_128_v2_shared`])
//! 3. `gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_k640_r4` (qt53
//!    down, [`rdna_compute::Gpu::gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded`])
//!
//! The canonical combine stays on the primary device.
//!
//! The iGPU reads everything in place: the routed activation, the cold expert
//! bytes and the output slot rows are pinned host memory (gfx1201-pinned D2
//! staging and tier allocations), and a pinned host address is the same device
//! address on both GPUs (measured: `tools/hetero/README.md`, Proof A). Hot slots
//! get the masked id `-1`, which both kernels skip (zero rows, no weight reads).
//!
//! The `Gpu` lives on one worker thread (a `Gpu` never crosses threads). A call
//! sends one job and waits, bounded, for its reply. On timeout the arm is
//! poisoned and every later call fails closed. A job still in flight keeps
//! writing only into the caller's executor buffers, which outlive the arm's
//! callers, and the worker drains it before the arm drops.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::Duration;

use hipfire_cpu::quant::CpuQuant;

use crate::cpu_exec::CpuMoeExpert;
use crate::types::DispatchError;

fn err(message: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!("iGPU cold arm: {message}"))
}

/// Masked routing ids for the iGPU: every slot `-1` except the host slots, which
/// keep their original expert id. `out` has one entry per routed slot.
pub fn igpu_masked_ids(ids: &[i32], host_slots: &[u32], out: &mut [i32]) -> Result<(), DispatchError> {
    if out.len() != ids.len() {
        return Err(err(format!("{} masked ids for {} routed slots", out.len(), ids.len())));
    }
    out.fill(-1);
    for &slot in host_slots {
        let s = slot as usize;
        let id = *ids
            .get(s)
            .ok_or_else(|| err(format!("host slot {s} outside {} routed slots", ids.len())))?;
        out[s] = id;
    }
    Ok(())
}

/// Write the gate/up and down byte addresses of each host slot's expert into the
/// per-expert pointer tables (`experts[i]` belongs to `host_slots[i]`). Every
/// expert must be qt44 gate/up + qt53 down, the formats the iGPU kernels decode.
pub fn fill_ptr_tables(
    ids: &[i32],
    host_slots: &[u32],
    experts: &[CpuMoeExpert<'_>],
    gate_up: &mut [u64],
    down: &mut [u64],
) -> Result<(), DispatchError> {
    if experts.len() != host_slots.len() || gate_up.len() != down.len() {
        return Err(err("pointer-table inputs disagree in length"));
    }
    for (&slot, e) in host_slots.iter().zip(experts) {
        if e.gate_up_quant != CpuQuant::Mq4G256V2 || e.down_quant != CpuQuant::Mq4G128V2 {
            return Err(err(format!(
                "expert formats {:?}/{:?}: the iGPU arm decodes qt44 gate/up (Mq4G256V2) and qt53 down (Mq4G128V2) only",
                e.gate_up_quant, e.down_quant
            )));
        }
        let s = slot as usize;
        let id = *ids
            .get(s)
            .ok_or_else(|| err(format!("host slot {s} outside {} routed slots", ids.len())))?;
        let e_idx = usize::try_from(id)
            .ok()
            .filter(|&i| i < gate_up.len())
            .ok_or_else(|| err(format!("expert id {id} outside the {}-entry table", gate_up.len())))?;
        gate_up[e_idx] = e.gate_up.as_ptr() as u64;
        down[e_idx] = e.down.as_ptr() as u64;
    }
    Ok(())
}

/// Every nonzero address must be HIP-registered memory whose device address is
/// the same address (pinned host memory): `lookup(p)` returns the device pointer
/// HIP reports for `p`, or `None` when HIP does not know it. Anything else would
/// make the iGPU fault, so it is refused before any launch.
pub fn check_device_accessible(
    ptrs: &[u64],
    lookup: impl Fn(u64) -> Option<u64>,
) -> Result<(), DispatchError> {
    for &p in ptrs.iter().filter(|&&p| p != 0) {
        if lookup(p) != Some(p) {
            return Err(err(format!(
                "address {p:#x} is not device-accessible pinned memory (expert bytes, activation and output must be pinned host memory)"
            )));
        }
    }
    Ok(())
}

/// Geometry of the routed experts the arm serves (one arm per load, all layers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColdArmDims {
    /// Hidden size (the gate/up K and the down M).
    pub dim: usize,
    /// Expert intermediate size (the down K).
    pub mi: usize,
    pub top_k: usize,
    pub n_experts: usize,
    /// Largest `rows` a call may carry.
    pub max_rows: usize,
}

/// One cold-slot computation, by address. The caller blocks until it is
/// answered (or the arm is poisoned), so the borrowed spans outlive it.
struct Job {
    rows: usize,
    masked_ids: Vec<i32>,
    gate_up: Vec<u64>,
    down: Vec<u64>,
    x: (usize, usize),
    out: (usize, usize),
}

/// A gfx1151 `Gpu` on a worker thread, computing D2 cold slots.
pub struct IgpuColdArm {
    dims: ColdArmDims,
    /// `HIPFIRE_IGPU_ARM_TRACE=1`: per-call timing totals, printed at drop.
    trace: Option<Mutex<ArmTrace>>,
    tx: Mutex<Option<Sender<Job>>>,
    rx: Mutex<Receiver<Result<(u64, u64), String>>>,
    poisoned: AtomicBool,
    worker: Mutex<Option<JoinHandle<()>>>,
}

/// Accumulated timings (ns) of a traced arm.
#[derive(Default, Debug)]
struct ArmTrace {
    calls: u64,
    slots: u64,
    caller_ns: u64,
    copy_ns: u64,
    kernel_sync_ns: u64,
}

impl ArmTrace {
    fn report(&self) {
        let per = |ns: u64| ns as f64 / self.calls.max(1) as f64 / 1e3;
        eprintln!(
            "igpu-arm trace: {} calls, {:.2} cold slots/call; per call: caller wait {:.1} us = copies {:.1} us + kernels+sync {:.1} us + handoff",
            self.calls,
            self.slots as f64 / self.calls.max(1) as f64,
            per(self.caller_ns),
            per(self.copy_ns),
            per(self.kernel_sync_ns)
        );
    }
}

struct Worker {
    gpu: rdna_compute::Gpu,
    /// The two pointer tables and the masked ids are written by the CPU into
    /// one pinned staging area, then reach one device buffer by ONE host-async
    /// copy queued ahead of the kernels (kernels read cached device memory; a
    /// coherent host table is read uncached, and a synchronous copy blocks).
    /// Each job's final device synchronize retires the copy before a rewrite.
    pinned: [(*mut std::ffi::c_void, usize); 3],
    staging: (*mut u8, usize),
    tables: rdna_compute::GpuTensor,
    gate_up_ptrs: rdna_compute::GpuTensor,
    down_ptrs: rdna_compute::GpuTensor,
    ids: rdna_compute::GpuTensor,
    /// Addresses already proven device-accessible (expert spans repeat across
    /// tokens, activation and output buffers per executor): one query each.
    verified: std::collections::HashSet<u64>,
    gate: rdna_compute::GpuTensor,
    up: rdna_compute::GpuTensor,
    rot: rdna_compute::GpuTensor,
}

/// A non-owning tensor over pinned host memory, which is the same address on
/// the iGPU (Proof A). `size` is in bytes; launchers check byte spans only.
///
/// # Safety
/// `addr` must be `size` bytes of live pinned host memory for the launch.
unsafe fn borrowed(addr: usize, size: usize) -> rdna_compute::GpuTensor {
    rdna_compute::GpuTensor {
        buf: hip_bridge::DeviceBuffer::from_raw(addr as *mut std::ffi::c_void, size),
        shape: vec![size / 4],
        dtype: rdna_compute::DType::F32,
    }
}

impl Worker {
    fn open(ordinal: i32, d: ColdArmDims) -> Result<Self, String> {
        let mut gpu = rdna_compute::Gpu::init_with_device(ordinal).map_err(|e| e.to_string())?;
        if gpu.arch != "gfx1151" {
            return Err(format!("device {ordinal} is {}, not gfx1151", gpu.arch));
        }
        let f32s = |gpu: &mut rdna_compute::Gpu, n: usize| {
            gpu.alloc_tensor(&[n], rdna_compute::DType::F32).map_err(|e| e.to_string())
        };
        let slots = d.max_rows * d.top_k;
        let flags = hip_bridge::HIP_HOST_MALLOC_MAPPED | hip_bridge::HIP_HOST_MALLOC_COHERENT;
        // Layout of both the staging area and the device buffer.
        let sizes = [d.n_experts * 8, d.n_experts * 8, slots * 4];
        let total: usize = sizes.iter().sum();
        let staging = gpu.hip.host_malloc(total, flags).map_err(|e| e.to_string())?.cast::<u8>();
        let tables = match gpu.alloc_tensor(&[total / 4], rdna_compute::DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.hip.host_free(staging.cast());
                return Err(e.to_string());
            }
        };
        let base = tables.buf.as_ptr() as usize;
        let mut pinned = [(std::ptr::null_mut(), 0usize); 3];
        let mut off = 0;
        for (i, &bytes) in sizes.iter().enumerate() {
            // SAFETY: inside the staging allocation (offsets sum to `total`).
            pinned[i] = (unsafe { staging.add(off) }.cast(), bytes);
            off += bytes;
        }
        // SAFETY: views into `tables` (live for the worker), same offsets.
        let (gate_up_ptrs, down_ptrs, ids) = unsafe {
            (
                borrowed(base, sizes[0]),
                borrowed(base + sizes[0], sizes[1]),
                borrowed(base + sizes[0] + sizes[1], sizes[2]),
            )
        };
        Ok(Self {
            pinned,
            staging: (staging, total),
            tables,
            gate_up_ptrs,
            down_ptrs,
            ids,
            verified: std::collections::HashSet::new(),
            gate: f32s(&mut gpu, slots * d.mi)?,
            up: f32s(&mut gpu, slots * d.mi)?,
            rot: f32s(&mut gpu, slots * d.mi)?,
            gpu,
        })
    }

    /// Returns `(copy_ns, kernel_sync_ns)`.
    fn run(&mut self, d: ColdArmDims, job: &Job) -> Result<(u64, u64), String> {
        let t0 = std::time::Instant::now();
        let s = |e: hip_bridge::HipError| e.to_string();
        // Refuse (never fault on) any span the iGPU cannot read in place; each
        // distinct address is queried once.
        {
            let hip = &self.gpu.hip;
            let verified = &self.verified;
            let lookup = |p: u64| {
                if verified.contains(&p) {
                    return Some(p);
                }
                // SAFETY: a one-byte non-owning view, used only for the attribute query.
                let probe = unsafe { hip_bridge::DeviceBuffer::from_raw(p as *mut std::ffi::c_void, 1) };
                hip.pointer_get_attributes(&probe).ok().map(|a| a.device_pointer as u64)
            };
            let spans = [job.x.0 as u64, job.out.0 as u64];
            check_device_accessible(&spans, lookup)
                .and_then(|()| check_device_accessible(&job.gate_up, lookup))
                .and_then(|()| check_device_accessible(&job.down, lookup))
                .map_err(|e| format!("{e:?}"))?;
        }
        let fresh = [job.x.0 as u64, job.out.0 as u64]
            .into_iter()
            .chain(job.gate_up.iter().copied())
            .chain(job.down.iter().copied())
            .filter(|&p| p != 0);
        self.verified.extend(fresh);
        // SAFETY: the pinned buffers hold exactly these extents (sizes from the
        // same dims) and no kernel reads them: the previous job synchronized.
        unsafe {
            std::ptr::copy_nonoverlapping(job.gate_up.as_ptr(), self.pinned[0].0.cast::<u64>(), job.gate_up.len());
            std::ptr::copy_nonoverlapping(job.down.as_ptr(), self.pinned[1].0.cast::<u64>(), job.down.len());
            std::ptr::copy_nonoverlapping(job.masked_ids.as_ptr(), self.pinned[2].0.cast::<i32>(), job.masked_ids.len());
        }
        let g = &mut self.gpu;
        // SAFETY: the staging area was just written above and stays unchanged
        // until this job's final synchronize, after the copy has executed.
        let staged = unsafe { std::slice::from_raw_parts(self.staging.0, self.staging.1) };
        g.hip.memcpy_htod_async_default(&self.tables.buf, staged).map_err(s)?;
        let t1 = std::time::Instant::now();
        // SAFETY: the caller blocks on this job, so both spans stay live.
        let (x, out) = unsafe { (borrowed(job.x.0, job.x.1), borrowed(job.out.0, job.out.1)) };
        g.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
            &self.gate_up_ptrs, &self.ids, &x, &self.gate, &self.up, 2 * d.mi, d.dim, job.rows,
        )
        .map_err(s)?;
        g.silu_mul_bf16_rt_rotate_x_mq_128_v2_shared(
            &self.gate, &self.up, &self.rot, d.mi, job.rows * d.top_k, None,
        )
        .map_err(s)?;
        g.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            &self.down_ptrs, &self.ids, &self.rot, &out, d.dim, d.mi, job.rows, d.n_experts, d.top_k,
        )
        .map_err(s)?;
        g.hip.device_synchronize().map_err(s)?;
        let t2 = std::time::Instant::now();
        Ok(((t1 - t0).as_nanos() as u64, (t2 - t1).as_nanos() as u64))
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.gpu.hip.device_synchronize();
        if !self.staging.0.is_null() {
            let _ = self.gpu.hip.host_free(self.staging.0.cast());
        }
    }
}

impl IgpuColdArm {
    /// Open gfx1151 device `ordinal` on a worker thread and allocate the arm's
    /// tables and scratch. Returns once the worker is ready, or its error.
    pub fn new(ordinal: i32, dims: ColdArmDims) -> Result<Self, DispatchError> {
        if dims.top_k != 10 {
            return Err(err(format!("top_k {} (the qt44/qt53 top-10 kernels need 10)", dims.top_k)));
        }
        let (tx, jobs) = mpsc::channel::<Job>();
        let (reply, rx) = mpsc::channel::<Result<(u64, u64), String>>();
        let worker = std::thread::Builder::new()
            .name("igpu-cold-arm".into())
            .spawn(move || {
                let mut w = match Worker::open(ordinal, dims) {
                    Ok(w) => {
                        let _ = reply.send(Ok((0, 0)));
                        w
                    }
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        return;
                    }
                };
                for job in jobs {
                    if reply.send(w.run(dims, &job)).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| err(format!("spawn worker: {e}")))?;
        rx.recv()
            .map_err(|_| err("worker exited before it was ready"))?
            .map_err(|e| err(format!("open gfx1151 device {ordinal}: {e}")))?;
        let trace = hipfire_config::developer_var("HIPFIRE_IGPU_ARM_TRACE")
            .is_ok_and(|v| v == "1")
            .then(|| Mutex::new(ArmTrace::default()));
        Ok(Self {
            dims,
            trace,
            tx: Mutex::new(Some(tx)),
            rx: Mutex::new(rx),
            poisoned: AtomicBool::new(false),
            worker: Mutex::new(Some(worker)),
        })
    }

    pub fn dims(&self) -> ColdArmDims {
        self.dims
    }

    /// The iGPU twin of [`crate::cpu_exec::moe_cpu_host_slot_outputs`]: same
    /// inputs and output representation. Waits at most `bound` for the reply;
    /// a timeout poisons the arm.
    #[allow(clippy::too_many_arguments)]
    pub fn host_slot_outputs(
        &self,
        rows: usize,
        ids: &[i32],
        host_slots: &[u32],
        experts: &[CpuMoeExpert<'_>],
        x_rot: &[f32],
        out: &mut [f32],
        bound: Duration,
    ) -> Result<(), DispatchError> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(err("poisoned by an earlier timeout or worker failure"));
        }
        let d = self.dims;
        let slots = rows * d.top_k;
        if rows == 0 || rows > d.max_rows || ids.len() < slots || x_rot.len() < rows * d.dim || out.len() < slots * d.dim {
            return Err(err(format!("extent mismatch for {rows} rows")));
        }
        if host_slots.is_empty() {
            return Ok(());
        }
        let mut masked_ids = vec![-1; slots];
        igpu_masked_ids(&ids[..slots], host_slots, &mut masked_ids)?;
        let (mut gate_up, mut down) = (vec![0u64; d.n_experts], vec![0u64; d.n_experts]);
        fill_ptr_tables(&ids[..slots], host_slots, experts, &mut gate_up, &mut down)?;
        let job = Job {
            rows,
            masked_ids,
            gate_up,
            down,
            x: (x_rot.as_ptr() as usize, rows * d.dim * 4),
            out: (out.as_mut_ptr() as usize, slots * d.dim * 4),
        };
        let tx = self.tx.lock().map_err(|_| err("sender lock poisoned"))?;
        tx.as_ref().ok_or_else(|| err("arm is shut down"))?.send(job).map_err(|_| {
            self.poisoned.store(true, Ordering::Release);
            err("worker is gone")
        })?;
        let t_call = std::time::Instant::now();
        let rx = self.rx.lock().map_err(|_| err("reply lock poisoned"))?;
        match rx.recv_timeout(bound) {
            Ok(Ok((copy_ns, kernel_ns))) => {
                if let Some(t) = &self.trace {
                    if let Ok(mut t) = t.lock() {
                        t.calls += 1;
                        t.slots += host_slots.len() as u64;
                        t.caller_ns += t_call.elapsed().as_nanos() as u64;
                        t.copy_ns += copy_ns;
                        t.kernel_sync_ns += kernel_ns;
                        if t.calls % 2048 == 0 {
                            t.report();
                        }
                    }
                }
                Ok(())
            }
            Ok(Err(e)) => {
                self.poisoned.store(true, Ordering::Release);
                Err(err(e))
            }
            Err(RecvTimeoutError::Timeout) => {
                self.poisoned.store(true, Ordering::Release);
                Err(err(format!("no reply within {bound:?}")))
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.poisoned.store(true, Ordering::Release);
                Err(err("worker exited"))
            }
        }
    }
}

impl Drop for IgpuColdArm {
    fn drop(&mut self) {
        if let Some(Ok(t)) = self.trace.as_ref().map(|t| t.lock()) {
            t.report();
        }
        // Closing the job channel ends the worker loop after its in-flight job,
        // whose device_synchronize drains the iGPU before the Gpu drops.
        if let Ok(mut tx) = self.tx.lock() {
            tx.take();
        }
        if let Ok(mut w) = self.worker.lock() {
            if let Some(h) = w.take() {
                let _ = h.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expert<'a>(gu: &'a [u8], dn: &'a [u8]) -> CpuMoeExpert<'a> {
        CpuMoeExpert {
            gate_up_quant: CpuQuant::Mq4G256V2,
            gate_up: gu,
            down_quant: CpuQuant::Mq4G128V2,
            down: dn,
        }
    }

    #[test]
    fn spans_must_be_device_accessible_at_the_same_address() {
        // A fake attribute lookup: 0x1000 and 0x2000 are pinned (device == host),
        // 0x3000 is unknown to HIP, 0x4000 maps to a different device address.
        let lookup = |p: u64| match p {
            0x1000 | 0x2000 => Some(p),
            0x4000 => Some(0x9000),
            _ => None,
        };
        assert!(check_device_accessible(&[0x1000, 0, 0x2000], lookup).is_ok());
        let e = format!("{:?}", check_device_accessible(&[0x1000, 0x3000], lookup).unwrap_err());
        assert!(e.contains("not device-accessible") && e.contains("0x3000"), "{e}");
        assert!(check_device_accessible(&[0x4000], lookup).is_err());
    }

    #[test]
    fn masked_ids_keep_only_host_slots() {
        let mut out = [0; 4];
        igpu_masked_ids(&[5, 7, 9, 11], &[1, 3], &mut out).unwrap();
        assert_eq!(out, [-1, 7, -1, 11]);
    }

    #[test]
    fn masked_ids_duplicate_expert_slots_stay_distinct() {
        let mut out = [0; 3];
        igpu_masked_ids(&[4, 4, 4], &[0, 2], &mut out).unwrap();
        assert_eq!(out, [4, -1, 4]);
    }

    #[test]
    fn masked_ids_zero_host_slots_mask_everything() {
        let mut out = [9; 3];
        igpu_masked_ids(&[1, 2, 3], &[], &mut out).unwrap();
        assert_eq!(out, [-1, -1, -1]);
    }

    #[test]
    fn masked_ids_refuse_out_of_range_slot() {
        let mut out = [0; 2];
        assert!(igpu_masked_ids(&[1, 2], &[2], &mut out).is_err());
    }

    #[test]
    fn ptr_tables_point_at_each_host_slots_expert_bytes() {
        let (a, b, c, d) = ([1u8; 8], [2u8; 8], [3u8; 8], [4u8; 8]);
        let experts = [expert(&a, &b), expert(&c, &d)];
        let (mut gu, mut dn) = ([0u64; 6], [0u64; 6]);
        fill_ptr_tables(&[0, 2, 5], &[1, 2], &experts, &mut gu, &mut dn).unwrap();
        assert_eq!(gu[2], a.as_ptr() as u64);
        assert_eq!(dn[2], b.as_ptr() as u64);
        assert_eq!(gu[5], c.as_ptr() as u64);
        assert_eq!(dn[5], d.as_ptr() as u64);
        assert_eq!(gu[0], 0, "a hot slot's expert is never written");
    }

    #[test]
    fn ptr_tables_refuse_unsupported_formats() {
        let (a, b) = ([1u8; 8], [2u8; 8]);
        let mut e = expert(&a, &b);
        e.down_quant = CpuQuant::Mq4G256V2;
        let (mut gu, mut dn) = ([0u64; 4], [0u64; 4]);
        let msg = format!("{:?}", fill_ptr_tables(&[1], &[0], &[e], &mut gu, &mut dn).unwrap_err());
        assert!(msg.contains("qt53"), "{msg}");
    }

    #[test]
    fn ptr_tables_refuse_expert_id_outside_the_table() {
        let (a, b) = ([1u8; 8], [2u8; 8]);
        let (mut gu, mut dn) = ([0u64; 4], [0u64; 4]);
        assert!(fill_ptr_tables(&[4], &[0], &[expert(&a, &b)], &mut gu, &mut dn).is_err());
    }
}
