//! Qwen3.5 DeltaNet checkpoint plumbing (spec §4.5 C5).
//!
//! The generic, family-free checkpoint pool lives in
//! [`hipfire_runtime::checkpoint_pool`]. This module holds only the
//! Qwen3.5-specific pieces: `CheckpointBlob` for `DeltaNetSnapshot`, the
//! GPU capture/restore over a live `DeltaNetState`, and re-exports so
//! existing `crate::checkpoint::…` paths keep working.

use crate::qwen35::DeltaNetState;
use crate::speculative::DeltaNetSnapshot;
use hip_bridge::{HipError, HipResult};
pub use hipfire_runtime::checkpoint_pool::{
    plan_resume, prefix_fingerprint, CheckpointBlob, CheckpointId, CheckpointPool,
};
use hipfire_runtime::serve_contract::CacheDomain;
use rdna_compute::Gpu;

impl CheckpointBlob for DeltaNetSnapshot {
    fn bytes_len(&self) -> u64 {
        DeltaNetSnapshot::bytes_len(self)
    }
}

/// Capture a checkpoint from the live `DeltaNetState` at boundary `p`
/// and insert it into `pool` (spec §4.5 C5).
///
/// `p` must be page-aligned (`p % 128 == 0` or `p == 0`). Allocates a
/// fresh `DeltaNetSnapshot` via [`DeltaNetSnapshot::new_for`], copies the
/// live state into it via [`DeltaNetSnapshot::save_from`], and inserts it
/// into the pool. The pool evicts oldest unpinned entries as needed.
///
/// Returns the minted [`CheckpointId`], [`CheckpointId::NONE`] when the
/// pool ceiling cannot cover the capture even after evicting every
/// unpinned entry, or an error if snapshot allocation or the
/// device-to-device copy fails.
///
/// **P2-wire will call this** from the serve engine's commit/prefill
/// path at page-aligned completed boundaries.
pub fn capture_checkpoint(
    gpu: &mut Gpu,
    pool: &mut CheckpointPool<DeltaNetSnapshot>,
    domain: &CacheDomain,
    p: u64,
    boundary_tokens: &[u32],
    state: &DeltaNetState,
) -> HipResult<CheckpointId> {
    if !CheckpointPool::<DeltaNetSnapshot>::is_aligned(p) {
        return Err(HipError::new(
            0,
            "capture_checkpoint: boundary not page-aligned",
        ));
    }

    // Pre-check the pool ceiling BEFORE allocating the snapshot and paying
    // for the device-to-device copy: a capture the pool cannot afford is a
    // soft refusal, indistinguishable from a hard failure only after the
    // work is already spent.
    let bytes = DeltaNetSnapshot::bytes_for(state);
    if !pool.can_afford(
        &(domain.clone(), p, prefix_fingerprint(boundary_tokens)),
        bytes,
    ) {
        return Ok(CheckpointId::NONE);
    }

    let mut snap = DeltaNetSnapshot::new_for(gpu, state)?;
    snap.save_from(state, gpu)?;

    let (id, displaced) = pool.insert(domain.clone(), p, prefix_fingerprint(boundary_tokens), snap);
    // Displaced blobs (LRU evictions / same-key replacement / a ceiling
    // refusal of this very capture) own device memory with no freeing
    // `Drop` — free them here or they leak for the process lifetime.
    for blob in displaced {
        blob.free_gpu(gpu);
    }
    Ok(id)
}

/// Restore an immutable cached checkpoint into a caller-owned **private**
/// `DeltaNetSnapshot` via device-to-device copy (spec §4.5 C5).
///
/// The pool's snapshot is never mutated; `dst` receives a private copy.
/// `dst` must have been pre-allocated with matching shapes (e.g. via
/// [`DeltaNetSnapshot::new_for`] against the same model state).
///
/// Returns an error if the checkpoint is not found or the copy fails.
///
/// **P2-wire will call this** when executing a [`ResumePlan`] to obtain
/// private recurrent state for a running request.
pub fn restore_private(
    gpu: &mut Gpu,
    pool: &mut CheckpointPool<DeltaNetSnapshot>,
    domain: &CacheDomain,
    p: u64,
    fp: u64,
    dst: &mut DeltaNetSnapshot,
) -> HipResult<()> {
    let src = pool
        .peek(domain, p, fp)
        .ok_or_else(|| HipError::new(0, "restore_private: checkpoint not found"))?;
    src.copy_to(dst, gpu)
}

// ───────────────────────────────────────────────────────────────────────────
// Tests (host-only — no GPU/HIP required)
// ───────────────────────────────────────────────────────────────────────────
