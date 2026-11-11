// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host staging for one forward's PLE rows, and the lifetime rule that makes
//! an asynchronous upload out of it safe.
//!
//! A deferred (mid-program) PLE upload used to be a blocking `hipMemcpy`,
//! which also waits for every layer enqueued ahead of it. The asynchronous
//! upload returns at once, but the device may read the host bytes any time
//! until the stream reaches the copy, so the single reusable staging buffer is
//! shared state with a hazard: it must not be overwritten, reallocated or
//! freed while a copy still reads it. [`PleHostStage`] is the only owner of
//! that buffer. It records a completion fence behind every asynchronous copy
//! and every write goes through [`PleHostStage::stage_buffer`], which first
//! drains the fence. A forward that fails after the enqueue, or drops the
//! stage without [`PleHostStage::free_gpu`], cannot skip the drain: the next
//! write still finds the state, and [`Drop`] leaks the bytes rather than free
//! memory a copy may still read.

use hip_bridge::{DeviceBuffer, Event, HipError, HipResult};
use rdna_compute::Gpu;

/// `HIPFIRE_QWEN4_PLE_ASYNC_UPLOAD` selects the fenced asynchronous upload for
/// a deferred PLE stage. Default on (identity-gated: warm-cache and
/// cache-quality outputs were byte-identical to the blocking upload on the
/// RC4 gate); `=0` opts out to the blocking upload, `=1` forces it on. Graph
/// capture and the device-token branch stay blocking regardless
/// ([`ple_async_upload`]).
pub const PLE_ASYNC_UPLOAD_ENV: &str = "HIPFIRE_QWEN4_PLE_ASYNC_UPLOAD";

/// Default of [`PLE_ASYNC_UPLOAD_ENV`] when it is unset or not `0`/`1`.
pub(crate) const PLE_ASYNC_UPLOAD_DEFAULT: bool = true;

/// Resolves [`PLE_ASYNC_UPLOAD_ENV`] from its process-snapshot value with the
/// strict `developer_bool` table: `"1"` on, `"0"` off, anything else (absent,
/// empty, other spelling) [`PLE_ASYNC_UPLOAD_DEFAULT`].
pub(crate) fn ple_async_upload_enabled(raw: Option<&str>) -> bool {
    match raw {
        Some("1") => true,
        Some("0") => false,
        _ => PLE_ASYNC_UPLOAD_DEFAULT,
    }
}

/// Whether an asynchronous copy may still read the staging bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PleUpload {
    /// No enqueued copy reads the bytes.
    Idle,
    /// An asynchronous copy reads the bytes and the fence was recorded right
    /// behind it on the same stream.
    Fenced,
    /// An asynchronous copy may read the bytes and no fence covers it (the
    /// enqueue or the fence record failed). Only draining the device proves
    /// it complete.
    Unfenced,
}

/// What must complete before the staging bytes may be written again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PleDrain {
    Nothing,
    Fence,
    Device,
}

impl PleUpload {
    /// The wait that retires the state.
    pub(crate) fn drain(self) -> PleDrain {
        match self {
            Self::Idle => PleDrain::Nothing,
            Self::Fenced => PleDrain::Fence,
            Self::Unfenced => PleDrain::Device,
        }
    }

    /// State just before an asynchronous copy is enqueued. The copy may
    /// already be reading when the call fails, so this is pessimistic.
    pub(crate) fn enqueue(self) -> Self {
        debug_assert_eq!(
            self,
            Self::Idle,
            "an upload was enqueued over an undrained one"
        );
        Self::Unfenced
    }

    /// State once the fence recorded right behind the copy succeeded.
    pub(crate) fn fence_recorded(self) -> Self {
        match self {
            Self::Unfenced => Self::Fenced,
            other => other,
        }
    }
}

/// Whether a deferred PLE stage uploads asynchronously: the knob is on, the
/// forward holds no device token (that branch carries its own readback
/// ordering), staging is deferred past layer 0 (otherwise nothing is queued
/// ahead of the copy to drain and a retained body may follow), and no graph
/// capture is open (the null stream is forbidden inside one).
pub(crate) fn ple_async_upload(
    enabled: bool,
    device_token: bool,
    deferred: bool,
    capturing: bool,
) -> bool {
    enabled && !device_token && deferred && !capturing
}

/// The device side of one staging upload, abstracted so the ordering rule
/// ([`PleSequencer`]) is testable without a GPU.
pub(crate) trait StageOps {
    /// Enqueue the stream-ordered copy of the staged bytes.
    fn copy(&mut self, len: usize) -> HipResult<()>;
    /// Record the completion fence on the copy's stream, right behind it.
    fn record_fence(&mut self) -> HipResult<()>;
    /// Host-wait until the fence completed.
    fn wait_fence(&mut self) -> HipResult<()>;
    /// Host-wait until the whole device drained.
    fn wait_device(&mut self) -> HipResult<()>;
}

/// The upload state machine: the one place that decides when the staging
/// bytes may be written again.
#[derive(Debug)]
pub(crate) struct PleSequencer {
    upload: PleUpload,
    fence_created: bool,
}

impl PleSequencer {
    pub(crate) fn new() -> Self {
        Self {
            upload: PleUpload::Idle,
            fence_created: false,
        }
    }

    /// Wait until no enqueued copy reads the bytes. On failure the state is
    /// unchanged, so the bytes stay unwritable and the next call retries.
    pub(crate) fn drain(&mut self, ops: &mut impl StageOps) -> HipResult<()> {
        match self.upload.drain() {
            PleDrain::Nothing => {}
            PleDrain::Fence if self.fence_created => ops.wait_fence()?,
            // A Fenced state without an event cannot arise; drain the whole
            // device rather than trust it.
            PleDrain::Fence | PleDrain::Device => ops.wait_device()?,
        }
        self.upload = PleUpload::Idle;
        Ok(())
    }

    /// Enqueue the copy and fence it. Every `?` after the enqueue leaves the
    /// state pending, so a failing forward cannot release the buffer early.
    /// `fenced == false` skips the fence (the caller proves ordering itself).
    pub(crate) fn upload(
        &mut self,
        ops: &mut impl StageOps,
        len: usize,
        fenced: bool,
    ) -> HipResult<()> {
        if !fenced {
            return ops.copy(len);
        }
        self.fence_created = true;
        self.upload = self.upload.enqueue();
        ops.copy(len)?;
        ops.record_fence()?;
        self.upload = self.upload.fence_recorded();
        Ok(())
    }

    pub(crate) fn pending(&self) -> bool {
        self.upload != PleUpload::Idle
    }
}

/// Pinned (`hipHostMalloc`) host memory owned by a [`PleHostStage`].
struct PinnedBytes {
    ptr: *mut u8,
    cap: usize,
}

/// The reusable pinned host staging buffer of one forward, its upload fence,
/// and the stream-ordered copy that reads it.
///
/// Every device read of the buffer is a `hipMemcpyAsync` on the forward's
/// compute stream (`gpu.active_stream`), never the legacy NULL stream: a
/// NULL-stream copy forces the runtime to insert dependency markers in both
/// directions against the blocking compute stream. The buffer is pinned, so
/// the copy is a plain stream-ordered DMA and the call never host-waits.
///
/// Ordering: the next host write goes through [`Self::stage_buffer`], which
/// first drains the fence recorded on the compute stream right behind the
/// previous copy ([`PleSequencer`]). The device-token forward alone skips the
/// fence ([`Self::upload_unfenced`]).
pub(crate) struct PleHostStage {
    /// `cap` bytes, allocated pinned on first use (`bind_thread` first) and
    /// kept until [`Self::free_gpu`].
    bytes: Option<PinnedBytes>,
    cap: usize,
    fence: Option<Event>,
    seq: PleSequencer,
    enabled: bool,
}

// SAFETY: the pinned allocation is exclusively owned by the stage (no
// aliasing handle exists) and all device access to it is ordered by the
// sequencer; the raw pointer is only dereferenced through `&mut self`/`&self`.
unsafe impl Send for PleHostStage {}

struct GpuStageOps<'a> {
    gpu: &'a Gpu,
    dst: Option<&'a DeviceBuffer>,
    src: &'a [u8],
    fence: Option<&'a Event>,
}

impl StageOps for GpuStageOps<'_> {
    fn copy(&mut self, len: usize) -> HipResult<()> {
        let dst = self.dst.expect("copy needs a destination");
        self.gpu
            .hip
            .memcpy_htod_async_on(dst, &self.src[..len], self.gpu.active_stream.as_ref())
    }
    fn record_fence(&mut self) -> HipResult<()> {
        let fence = self.fence.expect("fence created before the copy");
        self.gpu
            .hip
            .event_record(fence, self.gpu.active_stream.as_ref())
    }
    fn wait_fence(&mut self) -> HipResult<()> {
        self.gpu
            .hip
            .event_synchronize(self.fence.expect("fence exists when Fenced"))
    }
    fn wait_device(&mut self) -> HipResult<()> {
        self.gpu.hip.device_synchronize()
    }
}

/// The first `len` staged bytes, for a device copy to read.
fn pinned_slice(bytes: &Option<PinnedBytes>, len: usize) -> &[u8] {
    let pinned = bytes.as_ref().expect("pinned staging allocated");
    assert!(len <= pinned.cap, "PLE staging write exceeds capacity");
    // SAFETY: `ptr` is a live `cap`-byte host allocation owned by the stage;
    // the shared borrow excludes a concurrent `&mut` host write.
    unsafe { std::slice::from_raw_parts(pinned.ptr, len) }
}

impl PleHostStage {
    /// A stage whose pinned buffer holds `cap` bytes (the maximum of every
    /// upload it will stage).
    pub(crate) fn new(cap: usize, enabled: bool) -> Self {
        Self {
            bytes: None,
            cap,
            fence: None,
            seq: PleSequencer::new(),
            enabled,
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    fn ensure_pinned(&mut self, gpu: &Gpu) -> HipResult<()> {
        if self.bytes.is_none() {
            gpu.bind_thread()?;
            let raw = gpu.hip.host_malloc(self.cap.max(1), 0)?;
            self.bytes = Some(PinnedBytes {
                ptr: raw as *mut u8,
                cap: self.cap.max(1),
            });
        }
        Ok(())
    }

    fn drain(&mut self, gpu: &Gpu) -> HipResult<()> {
        let src: &[u8] = &[];
        let mut ops = GpuStageOps {
            gpu,
            dst: None,
            src,
            fence: self.fence.as_ref(),
        };
        self.seq.drain(&mut ops)
    }

    /// The first `len` bytes for the host copy to fill. This is the only
    /// mutable access to the buffer, and it drains any pending upload first.
    pub(crate) fn stage_buffer(&mut self, gpu: &Gpu, len: usize) -> HipResult<&mut [u8]> {
        self.ensure_pinned(gpu)?;
        self.drain(gpu)?;
        let pinned = self.bytes.as_mut().expect("pinned staging allocated");
        assert!(len <= pinned.cap, "PLE staging write exceeds capacity");
        // SAFETY: live `cap`-byte pinned allocation exclusively borrowed via
        // `&mut self`; the drain above retired every device read of it.
        Ok(unsafe { std::slice::from_raw_parts_mut(pinned.ptr, len) })
    }

    /// Upload of the staged bytes. Under graph capture, a stream-ordered copy
    /// on the capturing stream; otherwise a stream-ordered copy on the compute
    /// stream, fenced, with the host waiting for that fence (the blocking
    /// opt-out; no NULL-stream operation).
    pub(crate) fn upload_blocking(
        &mut self,
        gpu: &Gpu,
        dst: &DeviceBuffer,
        len: usize,
    ) -> HipResult<()> {
        if gpu.graphs.capture_mode {
            gpu.bind_thread()?;
            let stream = gpu
                .active_stream
                .as_ref()
                .expect("capture mode requires an active stream");
            return gpu
                .hip
                .memcpy_htod_async(dst, pinned_slice(&self.bytes, len), stream);
        }
        self.upload_async(gpu, dst, len)?;
        self.drain(gpu)
    }

    /// The device-token forward's asynchronous upload, not fenced (decode hot
    /// path), stream-ordered on the compute stream.
    ///
    /// Ordering proof (no fence needed): the previous forward enqueued this
    /// copy on the compute stream C *before* its argmax and the event `E`
    /// recorded behind it (`gpu_forward.rs` token readback `event_record`);
    /// this forward's `stage_ple` first runs `read_token`, a
    /// `hipStreamSynchronize` of the readback stream, which waits on `E`
    /// (`hipStreamWaitEvent`). So the previous copy completed before the host
    /// could reach the next `stage_buffer` write.
    pub(crate) fn upload_unfenced(
        &mut self,
        gpu: &Gpu,
        dst: &DeviceBuffer,
        len: usize,
    ) -> HipResult<()> {
        gpu.bind_thread()?;
        let mut ops = GpuStageOps {
            gpu,
            dst: Some(dst),
            src: pinned_slice(&self.bytes, len),
            fence: None,
        };
        self.seq.upload(&mut ops, len, false)
    }

    /// Enqueue the staged bytes on the compute stream without blocking the
    /// host and fence the copy.
    pub(crate) fn upload_async(
        &mut self,
        gpu: &Gpu,
        dst: &DeviceBuffer,
        len: usize,
    ) -> HipResult<()> {
        gpu.bind_thread()?;
        if self.fence.is_none() {
            self.fence = Some(
                gpu.hip
                    .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
            );
        }
        let mut ops = GpuStageOps {
            gpu,
            dst: Some(dst),
            src: pinned_slice(&self.bytes, len),
            fence: self.fence.as_ref(),
        };
        self.seq.upload(&mut ops, len, true)
    }

    /// Teardown: retire any pending upload, destroy the fence, free the
    /// pinned buffer (kept, not freed, when the drain failed).
    pub(crate) fn free_gpu(mut self, gpu: &Gpu) -> Option<HipError> {
        let mut first = self.drain(gpu).err();
        if let Some(fence) = self.fence.take() {
            if let Err(error) = gpu.hip.event_destroy(fence) {
                first.get_or_insert(error);
            }
        }
        if !self.seq.pending() {
            if let Some(pinned) = self.bytes.take() {
                if let Err(error) = gpu.hip.host_free(pinned.ptr as *mut std::ffi::c_void) {
                    first.get_or_insert(error);
                }
            }
        }
        first
    }
}

impl Drop for PleHostStage {
    fn drop(&mut self) {
        // The pinned buffer needs a `Gpu` to be freed (`free_gpu`); a stage
        // dropped without it leaks the allocation rather than free memory a
        // copy may still read.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_needs_no_wait() {
        assert_eq!(PleUpload::Idle.drain(), PleDrain::Nothing);
    }

    #[test]
    fn fenced_copy_waits_on_the_fence_only() {
        assert_eq!(PleUpload::Fenced.drain(), PleDrain::Fence);
    }

    #[test]
    fn unfenced_copy_waits_on_the_device() {
        assert_eq!(PleUpload::Unfenced.drain(), PleDrain::Device);
    }

    #[test]
    fn enqueue_is_pending_until_the_fence_is_recorded() {
        // Success path: enqueue then fence record.
        let enqueued = PleUpload::Idle.enqueue();
        assert_eq!(enqueued, PleUpload::Unfenced);
        assert_eq!(enqueued.fence_recorded(), PleUpload::Fenced);
        // Failure between the two (enqueue or record error): the state never
        // reaches Idle, so the next write still drains.
        assert_ne!(enqueued.drain(), PleDrain::Nothing);
    }

    #[test]
    fn fence_record_never_downgrades_or_revives() {
        assert_eq!(PleUpload::Idle.fence_recorded(), PleUpload::Idle);
        assert_eq!(PleUpload::Fenced.fence_recorded(), PleUpload::Fenced);
    }

    #[test]
    fn every_pending_state_has_a_drain() {
        for state in [PleUpload::Fenced, PleUpload::Unfenced] {
            assert_ne!(state.drain(), PleDrain::Nothing);
        }
    }

    #[test]
    fn async_upload_only_for_a_deferred_stage_without_a_device_token() {
        // (enabled, device_token, deferred, capturing)
        assert!(ple_async_upload(true, false, true, false));
        assert!(!ple_async_upload(false, false, true, false), "opt-out");
        assert!(
            !ple_async_upload(true, true, true, false),
            "device-token branch"
        );
        assert!(!ple_async_upload(true, false, false, false), "not deferred");
        assert!(!ple_async_upload(true, false, true, true), "graph capture");
    }

    #[test]
    fn async_upload_truth_table_requires_every_condition() {
        for bits in 0u8..16 {
            let [enabled, device_token, deferred, capturing] =
                [0, 1, 2, 3].map(|bit| bits >> bit & 1 == 1);
            assert_eq!(
                ple_async_upload(enabled, device_token, deferred, capturing),
                enabled && !device_token && deferred && !capturing,
            );
        }
    }

    #[test]
    fn async_upload_default_is_on_and_zero_opts_out() {
        assert!(PLE_ASYNC_UPLOAD_DEFAULT);
        assert!(ple_async_upload_enabled(None), "unset");
        assert!(!ple_async_upload_enabled(Some("0")), "=0 opts out");
        assert!(ple_async_upload_enabled(Some("1")), "=1 forces on");
        assert!(
            ple_async_upload_enabled(Some("")),
            "empty falls back to default"
        );
        assert!(
            ple_async_upload_enabled(Some("true")),
            "strict table: other spelling is default"
        );
    }

    /// Fake device: a copy reads the staged bytes only when the "device"
    /// executes it (at a fence/device wait), like a stream-ordered copy.
    struct FakeDevice {
        staging: Vec<u8>,
        pending: Vec<Vec<u8>>,
        consumed: Vec<Vec<u8>>,
        fail_fence_record: bool,
        waits: Vec<&'static str>,
    }

    impl FakeDevice {
        fn execute(&mut self) {
            // Model: the copy reads staging *now*, not at enqueue.
            let n = self.pending.len();
            for _ in 0..n {
                self.pending.pop();
                self.consumed.push(self.staging.clone());
            }
        }
    }

    impl StageOps for FakeDevice {
        fn copy(&mut self, _len: usize) -> HipResult<()> {
            self.pending.push(Vec::new());
            Ok(())
        }
        fn record_fence(&mut self) -> HipResult<()> {
            if self.fail_fence_record {
                return Err(HipError::new(0, "fence"));
            }
            Ok(())
        }
        fn wait_fence(&mut self) -> HipResult<()> {
            self.waits.push("fence");
            self.execute();
            Ok(())
        }
        fn wait_device(&mut self) -> HipResult<()> {
            self.waits.push("device");
            self.execute();
            Ok(())
        }
    }

    fn fake() -> FakeDevice {
        FakeDevice {
            staging: Vec::new(),
            pending: Vec::new(),
            consumed: Vec::new(),
            fail_fence_record: false,
            waits: Vec::new(),
        }
    }

    /// The host write gate: drain, then write.
    fn host_write(seq: &mut PleSequencer, dev: &mut FakeDevice, bytes: &[u8]) {
        seq.drain(dev).unwrap();
        dev.staging = bytes.to_vec();
    }

    #[test]
    fn next_write_waits_for_the_copy_to_consume_the_staging() {
        let mut dev = fake();
        let mut seq = PleSequencer::new();
        host_write(&mut seq, &mut dev, b"token-1");
        seq.upload(&mut dev, 7, true).unwrap();
        // The copy has not run yet; overwriting now would corrupt it.
        assert_eq!(dev.pending.len(), 1);
        host_write(&mut seq, &mut dev, b"token-2");
        assert_eq!(dev.consumed, vec![b"token-1".to_vec()], "copy saw token-1");
        seq.upload(&mut dev, 7, true).unwrap();
        host_write(&mut seq, &mut dev, b"token-3");
        assert_eq!(dev.consumed, vec![b"token-1".to_vec(), b"token-2".to_vec()]);
        assert_eq!(dev.waits, vec!["fence", "fence"]);
    }

    #[test]
    fn a_write_that_skips_the_gate_is_caught_by_the_fake() {
        // Negative control: without the drain the copy reads the new bytes.
        let mut dev = fake();
        let mut seq = PleSequencer::new();
        dev.staging = b"token-1".to_vec();
        seq.upload(&mut dev, 7, true).unwrap();
        dev.staging = b"token-2".to_vec(); // no drain
        dev.execute();
        assert_eq!(dev.consumed, vec![b"token-2".to_vec()]);
    }

    #[test]
    fn failed_fence_record_forces_a_device_drain_before_the_next_write() {
        let mut dev = fake();
        let mut seq = PleSequencer::new();
        host_write(&mut seq, &mut dev, b"a");
        dev.fail_fence_record = true;
        assert!(seq.upload(&mut dev, 1, true).is_err());
        dev.fail_fence_record = false;
        host_write(&mut seq, &mut dev, b"b");
        assert_eq!(dev.waits, vec!["device"]);
        assert_eq!(dev.consumed, vec![b"a".to_vec()]);
    }

    #[test]
    fn unfenced_upload_leaves_no_pending_state() {
        let mut dev = fake();
        let mut seq = PleSequencer::new();
        host_write(&mut seq, &mut dev, b"a");
        seq.upload(&mut dev, 1, false).unwrap();
        assert!(!seq.pending());
        assert!(dev.waits.is_empty());
    }
}
