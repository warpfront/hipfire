//! Host-owned developer S0 checkpoint; uses the public session restore contract.
use super::{ensure, R};
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_runtime::session_cache::{SessionRoute, SessionState, SnapshotParts};
use rdna_compute::Gpu;

pub(super) struct HostCheckpoint {
    meta: Vec<u8>,
    parts: Vec<Vec<u8>>,
}
impl HostCheckpoint {
    pub(super) fn capture(bundle: &mut Qwen4Bundle, gpu: &Gpu) -> R<Self> {
        let position = bundle.state.position;
        let SnapshotParts { meta, layout } =
            SessionState::snapshot_parts(bundle, SessionRoute::Ar, position)?;
        gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
        let mut parts = Vec::new();
        for (buf, offset, bytes) in layout
            .fixed
            .iter()
            .map(|p| (p.buf, p.offset, p.bytes))
            .chain(layout.rows.iter().map(|p| (p.buf, 0, p.rows * p.row_bytes)))
        {
            let mut data = vec![0; bytes];
            if bytes > 0 {
                gpu.hip
                    .memcpy_dtoh_at(&mut data, buf, offset)
                    .map_err(|e| e.to_string())?;
            }
            parts.push(data);
        }
        Ok(Self { meta, parts })
    }
    pub(super) fn bytes(&self) -> usize {
        self.parts.iter().map(Vec::len).sum()
    }
    pub(super) fn restore(&self, bundle: &mut Qwen4Bundle, gpu: &mut Gpu) -> R<()> {
        // A normal successful oracle boundary, never recovery of unknown work.
        gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
        {
            let layout = SessionState::restore_parts(bundle, gpu, SessionRoute::Ar, &self.meta)?;
            let ranges: Vec<_> = layout
                .fixed
                .iter()
                .map(|p| (p.buf, p.offset, p.bytes))
                .chain(layout.rows.iter().map(|p| (p.buf, 0, p.rows * p.row_bytes)))
                .collect();
            ensure(ranges.len() == self.parts.len(), "S0 restore part count")?;
            for ((buf, offset, bytes), source) in ranges.into_iter().zip(&self.parts) {
                ensure(
                    bytes == source.len()
                        && offset
                            .checked_add(bytes)
                            .is_some_and(|end| end <= buf.size()),
                    "S0 restore byte extent",
                )?;
                if bytes == 0 {
                    continue;
                }
                // SAFETY: checked subrange of a live state root. This borrowed
                // alias is never freed; H2D is synchronous, owner stays live.
                let dst = unsafe {
                    hip_bridge::DeviceBuffer::from_raw(
                        (buf.as_ptr() as usize + offset) as *mut std::ffi::c_void,
                        bytes,
                    )
                };
                gpu.hip
                    .memcpy_htod(&dst, source)
                    .map_err(|e| e.to_string())?;
            }
            gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
        }
        SessionState::finish_restore(bundle, gpu, SessionRoute::Ar, &self.meta)
    }
}
