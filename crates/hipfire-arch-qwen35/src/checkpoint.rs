//! Qwen3.5 DeltaNet checkpoint plumbing (spec §4.5 C5).
//!
//! The generic, family-free checkpoint pool lives in
//! [`hipfire_runtime::checkpoint_pool`]. This module holds only the
//! Qwen3.5-specific pieces: `CheckpointBlob` for `DeltaNetSnapshot`, the
//! GPU capture/restore over a live `DeltaNetState`, and re-exports so
//! existing `crate::checkpoint::…` paths keep working.

use std::collections::HashSet;

use hipfire_runtime::checkpoint_pool::checkpoint_drain_bytes;

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

/// Bytes left free for the next VMM granule map. A snapshot that would
/// consume this reserve is refused instead of being allocated.
pub(crate) const CHECKPOINT_MAP_RESERVE_BYTES: u64 = 512 << 20;

/// Bytes of unpinned checkpoints to free before allocating a new checkpoint
/// of `capture_bytes` while preserving `reserve` bytes for VMM granule maps.
///
/// Returns:
/// - `Some(0)` when `free >= capture_bytes + reserve` (no drain needed);
/// - `Some(target)` (`target > 0`) for the smallest prefix of `sizes`
///   (ordered unprotected-first, then soft-protected) whose sum covers the
///   shortfall;
/// - `None` when even draining every candidate in `sizes` cannot cover the
///   shortfall — callers must refuse the capture without destroying existing
///   reusable checkpoints for a capture that will be refused anyway.
pub(crate) fn checkpoint_capture_drain_bytes(
    free: u64,
    capture_bytes: u64,
    reserve: u64,
    sizes: impl IntoIterator<Item = u64>,
) -> Option<u64> {
    let need = capture_bytes.saturating_add(reserve);
    let mut missing = need.saturating_sub(free);
    if missing == 0 {
        return Some(0);
    }
    let mut target = 0u64;
    for size in sizes {
        target = target.saturating_add(size);
        missing = missing.saturating_sub(size);
        if missing == 0 {
            return Some(target);
        }
    }
    None
}

fn pop_unpinned_preferring_unprotected<B: CheckpointBlob>(
    pool: &mut CheckpointPool<B>,
    protected: &HashSet<CheckpointId>,
) -> Option<B> {
    pool.pop_lru_excluding(protected)
        .or_else(|| pool.pop_lru())
        .map(|(_, blob)| blob)
}

fn free_checkpoints_up_to(
    gpu: &mut Gpu,
    pool: &mut CheckpointPool<DeltaNetSnapshot>,
    protected: &HashSet<CheckpointId>,
    target: u64,
) -> u64 {
    let mut freed = 0u64;
    while freed < target {
        let Some(blob) = pop_unpinned_preferring_unprotected(pool, protected) else {
            break;
        };
        freed = freed.saturating_add(blob.bytes_len());
        blob.free_gpu(gpu);
    }
    freed
}

/// True when `bytes` can be hipMalloc'd without eating the KV map reserve.
///
/// Reads `device_mem_info` once after draining the tensor pool, evicts
/// unprotected checkpoints before soft-protected ones, and tracks freed bytes
/// arithmetically (avoiding `hipMemGetInfo` async-free lag). Returns `false`
/// without evicting anything if draining every unpinned entry still cannot
/// cover `bytes + CHECKPOINT_MAP_RESERVE_BYTES`.
fn checkpoint_device_room(
    gpu: &mut Gpu,
    pool: &mut CheckpointPool<DeltaNetSnapshot>,
    bytes: u64,
    protected: &HashSet<CheckpointId>,
) -> bool {
    gpu.drain_pool();
    let free = gpu
        .device_mem_info()
        .map(|(free, _)| free as u64)
        .unwrap_or(0);
    let (unprotected, soft) = pool.drain_candidates(protected);
    let Some(target) = checkpoint_capture_drain_bytes(
        free,
        bytes,
        CHECKPOINT_MAP_RESERVE_BYTES,
        unprotected.into_iter().chain(soft),
    ) else {
        return false;
    };
    if target == 0 {
        return true;
    }
    let freed = free_checkpoints_up_to(gpu, pool, protected, target);
    if freed > 0 {
        gpu.drain_pool();
        eprintln!(
            "[prefix-cache] freed {freed} checkpoint bytes to leave room for a {bytes} B capture (device free {free})"
        );
    }
    free.saturating_add(freed) >= bytes.saturating_add(CHECKPOINT_MAP_RESERVE_BYTES)
}

/// True when a pending VMM granule map fits within the mapper's configured
/// byte budget and therefore warrants freeing checkpoints to make device room.
///
/// A map that would exceed `budget` (when `budget > 0`) is refused by the
/// mapper before any HIP allocation happens, so draining live checkpoints for
/// it would destroy reusable state for a map that is guaranteed to fail.
pub(crate) fn should_drain_checkpoints_for_vmm_map(need: u64, mapped: u64, budget: u64) -> bool {
    need > 0
        && (budget == 0
            || mapped
                .checked_add(need)
                .is_some_and(|total| total <= budget))
}

/// Drain every unpinned checkpoint under emergency VMM memory pressure,
/// popping unprotected entries first (LRU) and then soft-protected entries
/// (LRU). Hard-pinned entries remain in `pool`.
pub(crate) fn drain_checkpoints_under_pressure<B: CheckpointBlob>(
    pool: &mut CheckpointPool<B>,
    protected: &HashSet<CheckpointId>,
) -> Vec<B> {
    let mut out = Vec::new();
    while let Some((_, blob)) = pool.pop_lru_excluding(protected) {
        out.push(blob);
    }
    while let Some((_, blob)) = pool.pop_lru() {
        out.push(blob);
    }
    out
}

/// Free unpinned checkpoints until the next VMM granule map has its reserve.
///
/// Snapshots are raw `hipMalloc`s beside the VMM pool. A granule map that
/// runs out of device memory poisons the mapper, so this runs before the
/// map, not only when a new checkpoint is captured. Soft-protected blobs go
/// last (protection is a priority, not a pin), but hard pins survive: a
/// request pins its checkpoint between `plan_resume` and restore.
///
/// `hipMemGetInfo` lags after a `hipFree`, so free memory is read once and
/// each freed blob is counted arithmetically; re-querying would over-free.
/// The caller only invokes this when a map is actually pending.
///
/// Returns freed bytes.
pub(crate) fn free_checkpoints_for_vmm_map(
    gpu: &mut Gpu,
    pool: &mut CheckpointPool<DeltaNetSnapshot>,
    need: u64,
    protected: &HashSet<CheckpointId>,
) -> u64 {
    if need == 0 || pool.is_empty() {
        return 0;
    }
    // Drain the tensor pool before reading `hipMemGetInfo` so cached freed
    // buffers are not miscounted as occupied device memory.
    gpu.drain_pool();
    let (unprotected, soft) = pool.drain_candidates(protected);
    let free = gpu
        .device_mem_info()
        .map(|(free, _)| free as u64)
        .unwrap_or(0);
    let target = checkpoint_drain_bytes(
        free,
        need,
        CHECKPOINT_MAP_RESERVE_BYTES,
        unprotected.into_iter().chain(soft),
    );
    if target == 0 {
        return 0;
    }
    let freed = free_checkpoints_up_to(gpu, pool, protected, target);
    // The tensor pool can still account the freed buffers as used until it
    // is drained (same reason as the idle reclaim).
    if freed > 0 {
        gpu.drain_pool();
        eprintln!("[prefix-cache] freed {freed} checkpoint bytes before a VMM map of {need} bytes");
    }
    freed
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
    capture_checkpoint_excluding(
        gpu,
        pool,
        domain,
        p,
        boundary_tokens,
        state,
        &HashSet::new(),
    )
}

/// [`capture_checkpoint`] honoring soft-protected live-session checkpoint ids
/// in `protected` whenever ceiling or device-room eviction runs.
pub fn capture_checkpoint_excluding(
    gpu: &mut Gpu,
    pool: &mut CheckpointPool<DeltaNetSnapshot>,
    domain: &CacheDomain,
    p: u64,
    boundary_tokens: &[u32],
    state: &DeltaNetState,
    protected: &HashSet<CheckpointId>,
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
    // The pool ceiling is a byte counter, not device memory. Snapshots are
    // hipMalloc'd beside the VMM page pool; allocating one the card cannot
    // hold fails the next KV granule map. Free unpinned snapshots until the
    // capture fits with a map reserve, or skip it.
    if !checkpoint_device_room(gpu, pool, bytes, protected) {
        return Ok(CheckpointId::NONE);
    }

    let mut snap = match DeltaNetSnapshot::new_for(gpu, state) {
        Ok(snap) => snap,
        Err(err) => {
            // Do not re-run `checkpoint_device_room` here: `hipMemGetInfo` lags
            // after `hipFree`. Evict one unpinned candidate and retry once.
            let Some(blob) = pop_unpinned_preferring_unprotected(pool, protected) else {
                return Err(err);
            };
            blob.free_gpu(gpu);
            gpu.drain_pool();
            DeltaNetSnapshot::new_for(gpu, state)?
        }
    };
    snap.save_from(state, gpu)?;

    let (id, displaced) = pool.insert_excluding(
        domain.clone(),
        p,
        prefix_fingerprint(boundary_tokens),
        snap,
        protected,
    );
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

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::serve_contract::{
        ArchPolicy, DeviceTopology, KvLayout, SharingNamespace, TemplateIdentity, TokenizerIdentity,
    };

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TaggedBlob {
        tag: &'static str,
        bytes: u64,
    }

    impl CheckpointBlob for TaggedBlob {
        fn bytes_len(&self) -> u64 {
            self.bytes
        }
    }

    fn test_domain() -> CacheDomain {
        CacheDomain {
            model_content_digest: vec![0u8; 32],
            model_load_epoch: 1,
            sidecar_digests: vec![],
            tokenizer: TokenizerIdentity {
                vocab_digest: vec![1u8; 16],
                config_digest: vec![2u8; 16],
            },
            template: TemplateIdentity {
                template_digest: vec![3u8; 16],
                normalization_tag: "default".to_string(),
            },
            arch_policy: ArchPolicy {
                arch_tag: "qwen35".to_string(),
                state_abi_tag: "q8".to_string(),
                position_attention_tag: "causal".to_string(),
            },
            kv_layout: KvLayout {
                k_stride_bytes: vec![128],
                v_stride_bytes: vec![128],
                layout_tag: "q8".to_string(),
            },
            device: DeviceTopology {
                device_id: "gpu0".to_string(),
                topology_id: "single".to_string(),
                allocation_epoch: 1,
            },
            namespace: SharingNamespace {
                domain_id: "test".to_string(),
            },
        }
    }

    #[test]
    fn should_drain_checkpoints_for_vmm_map_respects_budget_and_need() {
        // Zero need never drains.
        assert!(!should_drain_checkpoints_for_vmm_map(0, 0, 1024));
        assert!(!should_drain_checkpoints_for_vmm_map(0, 512, 0));

        // Unlimited budget (budget == 0) drains whenever need > 0.
        assert!(should_drain_checkpoints_for_vmm_map(256, 10_000, 0));

        // Under budget and exact fit both drain.
        assert!(should_drain_checkpoints_for_vmm_map(256, 512, 1024));
        assert!(should_drain_checkpoints_for_vmm_map(512, 512, 1024));

        // Exceeding budget refuses to drain (map will be refused by budget anyway).
        assert!(!should_drain_checkpoints_for_vmm_map(513, 512, 1024));
        assert!(!should_drain_checkpoints_for_vmm_map(1, u64::MAX, u64::MAX));
    }

    #[test]
    fn drain_checkpoints_under_pressure_orders_unprotected_then_soft_and_keeps_hard_pins() {
        let domain = test_domain();
        let mut pool = CheckpointPool::new(1024 * 1024);

        let (id_soft_old, _) = pool.insert(
            domain.clone(),
            128,
            1,
            TaggedBlob {
                tag: "soft_old",
                bytes: 100,
            },
        );
        let (id_unprot_old, _) = pool.insert(
            domain.clone(),
            256,
            2,
            TaggedBlob {
                tag: "unprot_old",
                bytes: 100,
            },
        );
        let (id_hard_pin, _) = pool.insert(
            domain.clone(),
            384,
            3,
            TaggedBlob {
                tag: "hard_pin",
                bytes: 100,
            },
        );
        let (id_soft_new, _) = pool.insert(
            domain.clone(),
            512,
            4,
            TaggedBlob {
                tag: "soft_new",
                bytes: 100,
            },
        );
        let (_id_unprot_new, _) = pool.insert(
            domain.clone(),
            640,
            5,
            TaggedBlob {
                tag: "unprot_new",
                bytes: 100,
            },
        );

        assert!(pool.pin(&domain, 384, 3));
        assert!(id_unprot_old.is_some());
        assert!(id_hard_pin.is_some());

        let protected: HashSet<CheckpointId> = [id_soft_old, id_soft_new].into_iter().collect();
        let drained = drain_checkpoints_under_pressure(&mut pool, &protected);
        let tags: Vec<&str> = drained.iter().map(|b| b.tag).collect();

        // Unprotected entries are drained first (LRU order), then soft-protected
        // entries (LRU order). Hard-pinned entry survives in the pool.
        assert_eq!(
            tags,
            vec!["unprot_old", "unprot_new", "soft_old", "soft_new"]
        );
        assert_eq!(pool.len(), 1);
        assert!(pool.contains(&domain, 384, 3));
        assert!(pool.is_pinned(&domain, 384, 3));
    }

    #[test]
    fn checkpoint_capture_drain_bytes_refuses_futile_drain_and_sizes_minimal_prefix() {
        // Enough free device memory for capture + reserve: no drain needed.
        assert_eq!(
            checkpoint_capture_drain_bytes(1000, 100, 500, [200, 300]),
            Some(0)
        );
        // Shortfall of 200: drains only the minimal prefix covering 200.
        assert_eq!(
            checkpoint_capture_drain_bytes(400, 100, 500, [64, 150, 512]),
            Some(214)
        );
        // Even draining every candidate cannot reach capture + reserve: refuse
        // without destroying existing reusable checkpoints.
        assert_eq!(
            checkpoint_capture_drain_bytes(100, 100, 500, [64, 150]),
            None
        );
        assert_eq!(checkpoint_capture_drain_bytes(100, 100, 500, []), None);
    }
}
