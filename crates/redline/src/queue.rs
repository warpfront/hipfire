// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compute queue — submit command buffers to the GPU.
//!
//! Raw-chunk submission (experiment/vcn-ring): one internal raw submit serves
//! every engine (compute, VCN JPEG) with optional syncobj dependencies.
//! `submit_async` never waits on the host; `wait_fence` is the sole terminal
//! wait. The BO list and chunk arrays live through the ioctl; the BO list is
//! destroyed immediately after submit returns.

use crate::device::{Device, GpuBuffer};
pub use crate::drm::AmdgpuBoListHandle;
use crate::drm::*;
use crate::{RedlineError, Result};

pub const AMDGPU_HW_IP_COMPUTE: u32 = 1;

/// Engine selector: HW IP block + instance + ring within the block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Engine {
    pub ip_type: u32,
    pub ip_instance: u32,
    pub ring: u32,
}

impl Engine {
    /// Default compute engine (compute IP, instance 0, ring 0).
    pub const COMPUTE: Self = Self {
        ip_type: AMDGPU_HW_IP_COMPUTE,
        ip_instance: 0,
        ring: 0,
    };
}

/// Kernel sync object (drm syncobj handle). Created unsignaled; signaled by
/// the GPU via SYNCOBJ_OUT or by the host via `signal` (bring-up/tests).
/// Decoder-owned: the decoder creates its ready-syncobj and destroys it on
/// decoder destroy, exactly once (destroy consumes self).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyncObj {
    pub handle: u32,
}

impl SyncObj {
    /// Create an unsignaled syncobj.
    pub fn new(dev: &Device) -> Result<Self> {
        let mut handle: u32 = 0;
        let ret = unsafe { (dev.drm.cs_create_syncobj)(dev.handle, &mut handle) };
        if ret != 0 {
            return Err(RedlineError {
                code: ret,
                message: format!("cs_create_syncobj failed: {ret}"),
            });
        }
        Ok(Self { handle })
    }

    /// Signal from the host (bring-up/tests; production signals via SYNCOBJ_OUT).
    pub fn signal(&self, dev: &Device) -> Result<()> {
        let ret = unsafe { (dev.drm.cs_syncobj_signal)(dev.handle, &self.handle, 1) };
        if ret != 0 {
            return Err(RedlineError {
                code: ret,
                message: format!("cs_syncobj_signal failed: {ret}"),
            });
        }
        Ok(())
    }

    /// Destroy. Consumes self so each syncobj is released exactly once.
    pub fn destroy(self, dev: &Device) {
        unsafe {
            (dev.drm.cs_destroy_syncobj)(dev.handle, self.handle);
        }
    }
}

/// Fence for one async submission. Carries no queue state — the context used
/// for fence queries comes from the Queue at wait time, so a SubmissionFence
/// is invalid after its Queue is destroyed.
#[derive(Clone, Copy, Debug)]
pub struct SubmissionFence {
    pub engine: Engine,
    pub seq_no: u64,
}

/// Per-submit syncobj dependencies and IB flags.
#[derive(Clone, Copy, Debug)]
pub struct SubmitSync<'a> {
    /// Syncobjs the GPU waits on before executing the IB (SYNCOBJ_IN).
    pub wait: &'a [SyncObj],
    /// Syncobj signaled when the IB completes (SYNCOBJ_OUT).
    pub signal: Option<SyncObj>,
    /// IB flags (e.g. AMDGPU_IB_FLAG_EMIT_MEM_SYNC on compute-after-JPEG so
    /// repeated-frame reads cannot observe stale compute caches).
    pub ib_flags: u32,
}

impl<'a> Default for SubmitSync<'a> {
    fn default() -> Self {
        Self {
            wait: &[],
            signal: None,
            ib_flags: 0,
        }
    }
}

/// A GPU context for command submission (owns the amdgpu context).
pub struct ComputeQueue {
    ctx: AmdgpuContext,
}

impl ComputeQueue {
    pub fn new(dev: &Device) -> Result<Self> {
        let mut ctx: AmdgpuContext = std::ptr::null_mut();
        let ret = unsafe { (dev.drm.cs_ctx_create2)(dev.handle, 0, &mut ctx) };
        if ret != 0 {
            return Err(RedlineError {
                code: ret,
                message: format!("cs_ctx_create2 failed: {ret}"),
            });
        }
        eprintln!("[redline] Compute context created");
        Ok(Self { ctx })
    }

    /// Submit an indirect buffer and return without any host wait.
    /// `ib_size_dwords`: number of dwords to execute from `ib_buf`.
    /// `bo_refs`: all GPU buffers the commands reference (for the BO list).
    /// `sync`: syncobj dependencies, output signal, and IB flags.
    pub fn submit_async(
        &self,
        dev: &Device,
        engine: Engine,
        ib_buf: &GpuBuffer,
        ib_size_dwords: u32,
        bo_refs: &[&GpuBuffer],
        sync: &SubmitSync<'_>,
    ) -> Result<SubmissionFence> {
        let bo_list = Self::create_bo_list(dev, bo_refs)?;
        let seq = self.submit_raw(dev, engine, ib_buf.gpu_addr, ib_size_dwords, bo_list, sync);
        // The BO list only needs to live through the ioctl.
        unsafe {
            (dev.drm.bo_list_destroy)(bo_list);
        }
        Ok(SubmissionFence {
            engine,
            seq_no: seq?,
        })
    }

    /// Sole terminal wait API. Returns true when the fence completed within
    /// `timeout_ns`, false on timeout (a zero timeout polls).
    pub fn wait_fence(
        &self,
        dev: &Device,
        fence: &SubmissionFence,
        timeout_ns: u64,
    ) -> Result<bool> {
        let mut f = CsFence {
            context: self.ctx,
            ip_type: fence.engine.ip_type,
            ip_instance: fence.engine.ip_instance,
            ring: fence.engine.ring,
            fence: fence.seq_no,
        };
        let mut expired = 0u32;
        let ret = unsafe { (dev.drm.cs_query_fence_status)(&mut f, timeout_ns, 0, &mut expired) };
        if ret != 0 {
            return Err(RedlineError {
                code: ret,
                message: format!("fence wait failed: {ret}"),
            });
        }
        Ok(expired != 0)
    }

    /// Submit an indirect buffer (PM4 commands) and wait for completion.
    /// `ib_buf`: GPU buffer containing PM4 dwords
    /// `ib_size_dwords`: number of PM4 dwords to execute
    /// `bo_refs`: all GPU buffers the commands reference (for BO list)
    pub fn submit_and_wait(
        &self,
        dev: &Device,
        ib_buf: &GpuBuffer,
        ib_size_dwords: u32,
        bo_refs: &[&GpuBuffer],
    ) -> Result<()> {
        let fence = self.submit_async(
            dev,
            Engine::COMPUTE,
            ib_buf,
            ib_size_dwords,
            bo_refs,
            &SubmitSync::default(),
        )?;
        if !self.wait_fence(dev, &fence, 10_000_000_000)? {
            return Err(RedlineError {
                code: -1,
                message: "GPU timeout (10s)".into(),
            });
        }
        Ok(())
    }

    /// Submit with a pre-created BO list (avoids bo_list_create/destroy per dispatch).
    pub fn submit_with_bo_list(
        &self,
        dev: &Device,
        ib_buf: &GpuBuffer,
        ib_size_dwords: u32,
        bo_list: AmdgpuBoListHandle,
    ) -> Result<()> {
        let seq_no = self.submit_raw(
            dev,
            Engine::COMPUTE,
            ib_buf.gpu_addr,
            ib_size_dwords,
            bo_list,
            &SubmitSync::default(),
        )?;
        let fence = SubmissionFence {
            engine: Engine::COMPUTE,
            seq_no,
        };
        if !self.wait_fence(dev, &fence, 10_000_000_000)? {
            return Err(RedlineError {
                code: -1,
                message: "GPU timeout (10s)".into(),
            });
        }
        Ok(())
    }

    pub fn destroy(self, dev: &Device) {
        unsafe {
            (dev.drm.cs_ctx_free)(self.ctx);
        }
    }

    /// One internal raw-chunk submit for every engine. Builds IB + optional
    /// SYNCOBJ_IN/OUT chunks and returns the sequence number; the caller wraps
    /// it in a SubmissionFence and decides whether to wait.
    fn submit_raw(
        &self,
        dev: &Device,
        engine: Engine,
        ib_va: u64,
        ib_size_dwords: u32,
        bo_list: AmdgpuBoListHandle,
        sync: &SubmitSync<'_>,
    ) -> Result<u64> {
        let ib_bytes = ib_size_dwords.checked_mul(4).ok_or_else(|| RedlineError {
            code: -1,
            message: format!("IB size {ib_size_dwords} dwords overflows u32 bytes"),
        })?;
        let ib = DrmCsChunkIb {
            _pad: 0,
            flags: sync.ib_flags,
            va_start: ib_va,
            ib_bytes,
            ip_type: engine.ip_type,
            ip_instance: engine.ip_instance,
            ring: engine.ring,
        };
        // Payload backing store — must outlive the ioctl below.
        let wait_handles: Vec<u32> = sync.wait.iter().map(|s| s.handle).collect();
        let signal_handles: [u32; 1] = match sync.signal {
            Some(s) => [s.handle],
            None => [0],
        };
        let mut chunks = Vec::with_capacity(3);
        chunks.push(DrmCsChunk {
            chunk_id: AMDGPU_CHUNK_ID_IB,
            length_dw: (std::mem::size_of::<DrmCsChunkIb>() / 4) as u32,
            chunk_data: &ib as *const DrmCsChunkIb as u64,
        });
        if !wait_handles.is_empty() {
            chunks.push(DrmCsChunk {
                chunk_id: AMDGPU_CHUNK_ID_SYNCOBJ_IN,
                length_dw: wait_handles.len() as u32,
                chunk_data: wait_handles.as_ptr() as u64,
            });
        }
        if sync.signal.is_some() {
            chunks.push(DrmCsChunk {
                chunk_id: AMDGPU_CHUNK_ID_SYNCOBJ_OUT,
                length_dw: 1,
                chunk_data: signal_handles.as_ptr() as u64,
            });
        }
        let mut seq_no: u64 = 0;
        let ret = unsafe {
            (dev.drm.cs_submit_raw)(
                dev.handle,
                self.ctx,
                bo_list,
                chunks.len() as i32,
                chunks.as_mut_ptr(),
                &mut seq_no,
            )
        };
        if ret != 0 {
            return Err(RedlineError {
                code: ret,
                message: format!("cs_submit_raw failed: {ret}"),
            });
        }
        Ok(seq_no)
    }

    /// Create a BO list over the referenced buffers (destroy after submit).
    fn create_bo_list(dev: &Device, bo_refs: &[&GpuBuffer]) -> Result<AmdgpuBoListHandle> {
        let bo_handles: Vec<AmdgpuBoHandle> = bo_refs.iter().map(|b| b.handle).collect();
        let prios: Vec<u8> = vec![0; bo_handles.len()];
        let mut bo_list: AmdgpuBoListHandle = std::ptr::null_mut();
        let ret = unsafe {
            (dev.drm.bo_list_create)(
                dev.handle,
                bo_handles.len() as u32,
                bo_handles.as_ptr(),
                prios.as_ptr(),
                &mut bo_list,
            )
        };
        if ret != 0 {
            return Err(RedlineError {
                code: ret,
                message: format!("bo_list_create failed: {ret}"),
            });
        }
        Ok(bo_list)
    }
}
