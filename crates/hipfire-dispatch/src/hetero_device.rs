//! One layer's attachment to the device-side hetero split (design v3).
//!
//! Sealed MoE decode calls whose routed experts expose a [`HeteroDeviceSplit`]
//! (via `RoutedExpertWeights::hetero_split`) lower `PublishCold` before the
//! routed GPU tail and `AwaitCold` + `InstallCold` before the canonical combine,
//! in place of the host-synchronous D2 `PrepareSplit`/`JoinSplit`. Every stage
//! is a device kernel, so the call is capturable and replayable.

use rdna_compute::{Gpu, GpuTensor, HeteroSlotPtrs};

use crate::types::DispatchError;

/// Device state one trunk layer's split stages need.
pub struct HeteroDeviceSplit {
    pub layer: u32,
    pub layers: u32,
    /// This layer's mailbox slot (pinned host memory).
    pub slot: HeteroSlotPtrs,
    /// The load's device token counter (shared by every layer; a view).
    pub counter: GpuTensor,
    /// 1 = the expert is cold (host tier, computed by the expert device).
    pub cold_bitmap: GpuTensor,
    /// The primary's routed ids with cold slots masked to -1.
    pub masked_ids: GpuTensor,
    pub n_experts: i32,
    /// `hetero_await_cold` poll bound.
    pub max_polls: u32,
}

fn hip(e: hip_bridge::HipError) -> DispatchError {
    DispatchError::Hip(format!("hetero device split: {e}"))
}

impl HeteroDeviceSplit {
    /// `PublishCold`: route into the mailbox, device-masked ids, request word.
    pub fn publish(
        &self,
        gpu: &mut Gpu,
        rows: usize,
        top_k: usize,
        dim: usize,
        x_rot: &GpuTensor,
        ids: &GpuTensor,
        weights: &GpuTensor,
    ) -> Result<(), DispatchError> {
        gpu.hetero_publish_cold(
            &self.counter,
            self.layer,
            self.layers,
            x_rot,
            ids,
            weights,
            &self.cold_bitmap,
            self.n_experts,
            rows as i32,
            top_k as i32,
            dim as i32,
            &self.slot,
            &self.masked_ids,
        )
        .map_err(hip)
    }

    /// `AwaitCold`: bounded wait for the expert device's response.
    pub fn await_cold(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        gpu.hetero_await_cold(&self.counter, self.layer, self.layers, &self.slot, self.max_polls)
            .map_err(hip)
    }

    /// `InstallCold`: the cold slots' rows into the expanded down buffer.
    pub fn install(
        &self,
        gpu: &mut Gpu,
        slots: usize,
        dim: usize,
        ids: &GpuTensor,
        down_expanded: &GpuTensor,
    ) -> Result<(), DispatchError> {
        gpu.hetero_install_cold(
            &self.slot,
            ids,
            &self.cold_bitmap,
            self.n_experts,
            slots as u32,
            dim as i32,
            down_expanded,
        )
        .map_err(hip)
    }
}
