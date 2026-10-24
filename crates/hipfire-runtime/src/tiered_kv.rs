// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! First storage slice for issue #925, deliberately lab-only. No selector,
//! model-forward integration, DN checkpoint, graph or speculative commit is
//! installed. A block holds *already rotated, encoded* token-major KV rows.
//! Moving it never quantizes again or changes its absolute RoPE positions.
//!
//! The caller owns one device buffer per (FA layer, K/V side). Those buffers
//! have a fixed selected-history prefix and a separate generation reserve.
//! Only complete, committed history blocks may enter this archive. Partial
//! blocks and speculative rows stay with the caller until committed. All
//! transfers here are synchronous; async staging needs a separate lifetime
//! and failure contract before it can replace this reference implementation.

use hip_bridge::{DeviceBuffer, HipError, HipResult, HipRuntime, PinnedHostBuffer};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub mod streaming;

fn invalid(message: impl AsRef<str>) -> HipError {
    HipError::new(0, message.as_ref())
}

fn product(a: usize, b: usize) -> HipResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("tiered KV: byte/token calculation overflow"))
}

/// Explicit limits, including both K/V sides of every full-attention layer.
/// `device_limit_bytes` covers this KV owner, NOT model/scratch/driver memory.
/// A whole-process managed-capacity ledger and an external VRAM monitor must
/// additionally bound an experiment on a larger physical card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchiveConfig {
    pub block_tokens: usize,
    pub row_bytes_per_side: usize,
    pub full_attention_layers: usize,
    pub max_history_tokens: usize,
    pub window_blocks: usize,
    pub generation_tokens: usize,
    pub host_limit_bytes: usize,
    pub device_limit_bytes: usize,
}

/// Validated, overflow-checked storage geometry. There are no allocations here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchiveLayout {
    config: ArchiveConfig,
    sides: usize,
    block_side_bytes: usize,
    block_bytes: usize,
    device_side_bytes: usize,
    host_capacity_bytes: usize,
    device_capacity_bytes: usize,
}

impl ArchiveLayout {
    pub fn new(config: ArchiveConfig) -> HipResult<Self> {
        if [
            config.block_tokens,
            config.row_bytes_per_side,
            config.full_attention_layers,
            config.max_history_tokens,
            config.window_blocks,
            config.generation_tokens,
        ]
        .contains(&0)
        {
            return Err(invalid("tiered KV: zero-sized geometry"));
        }
        if !config
            .max_history_tokens
            .is_multiple_of(config.block_tokens)
        {
            return Err(invalid(
                "tiered KV: history capacity must contain whole blocks",
            ));
        }
        let sides = product(config.full_attention_layers, 2)?;
        let block_side_bytes = product(config.block_tokens, config.row_bytes_per_side)?;
        let block_bytes = product(block_side_bytes, sides)?;
        let history_rows = product(config.window_blocks, config.block_tokens)?;
        if history_rows > config.max_history_tokens {
            return Err(invalid(
                "tiered KV: selected capacity exceeds history capacity",
            ));
        }
        let device_rows = history_rows
            .checked_add(config.generation_tokens)
            .ok_or_else(|| invalid("tiered KV: device row count overflow"))?;
        let device_side_bytes = product(device_rows, config.row_bytes_per_side)?;
        let host_capacity_bytes = product(
            product(config.max_history_tokens, config.row_bytes_per_side)?,
            sides,
        )?;
        let device_capacity_bytes = product(device_side_bytes, sides)?;
        if host_capacity_bytes > config.host_limit_bytes {
            return Err(invalid(
                "tiered KV: declared history exceeds host byte budget",
            ));
        }
        if device_capacity_bytes > config.device_limit_bytes {
            return Err(invalid(
                "tiered KV: working set and generation exceed device byte budget",
            ));
        }
        Ok(Self {
            config,
            sides,
            block_side_bytes,
            block_bytes,
            device_side_bytes,
            host_capacity_bytes,
            device_capacity_bytes,
        })
    }

    pub fn host_capacity_bytes(self) -> usize {
        self.host_capacity_bytes
    }

    pub fn device_capacity_bytes(self) -> usize {
        self.device_capacity_bytes
    }

    pub fn device_side_bytes(self) -> usize {
        self.device_side_bytes
    }

    pub fn selected_capacity_tokens(self) -> usize {
        self.config.window_blocks * self.config.block_tokens
    }

    /// Absolute source rows, NOT the packed destinations used by KV writes.
    pub fn absolute_positions(self, keys: &[BlockKey]) -> HipResult<Vec<usize>> {
        validate_keys(keys, self.config.window_blocks)?;
        let mut out = Vec::with_capacity(product(keys.len(), self.config.block_tokens)?);
        for key in keys {
            let end = key
                .start_token
                .checked_add(self.config.block_tokens)
                .ok_or_else(|| invalid("tiered KV: absolute position overflow"))?;
            if !key.start_token.is_multiple_of(self.config.block_tokens)
                || end > self.config.max_history_tokens
            {
                return Err(invalid("tiered KV: block outside declared history"));
            }
            out.extend(key.start_token..end);
        }
        Ok(out)
    }
}

/// Revision and owner prevent reuse of a stale divergent suffix or another
/// request's bytes, even if the absolute token positions happen to match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockKey {
    owner: u64,
    revision: u64,
    pub start_token: usize,
}

fn validate_keys(keys: &[BlockKey], capacity: usize) -> HipResult<()> {
    if keys.len() > capacity {
        return Err(invalid(
            "tiered KV: selected blocks exceed fixed working set",
        ));
    }
    for pair in keys.windows(2) {
        if pair[0].owner != pair[1].owner || pair[0].start_token >= pair[1].start_token {
            return Err(invalid(
                "tiered KV: selection must be unique, chronological and single-owner",
            ));
        }
    }
    Ok(())
}

/// Published mapping of physical block slots. An interrupted copy invalidates
/// the WHOLE mapping: neither the old nor a partly installed new view is usable.
struct WindowMap {
    capacity: usize,
    keys: Option<Vec<BlockKey>>,
}

impl WindowMap {
    fn new(layout: ArchiveLayout) -> Self {
        Self {
            capacity: layout.config.window_blocks,
            keys: None,
        }
    }

    fn keys(&self) -> Option<&[BlockKey]> {
        self.keys.as_deref()
    }

    fn invalidate(&mut self) {
        self.keys = None;
    }

    fn install(
        &mut self,
        keys: &[BlockKey],
        mut copy: impl FnMut(usize, BlockKey) -> HipResult<()>,
    ) -> HipResult<usize> {
        // Reject invalid metadata before touching any device bytes or map.
        validate_keys(keys, self.capacity)?;
        let old = self.keys.take();
        let mut changed = 0;
        for (slot, &key) in keys.iter().enumerate() {
            if old.as_ref().and_then(|v| v.get(slot)) != Some(&key) {
                copy(slot, key)?;
                changed += 1;
            }
        }
        self.keys = Some(keys.to_vec());
        Ok(changed)
    }
}

/// One fixed device allocation owner AND its mapping. They cannot be moved to
/// unrelated allocations or have K/V side order changed independently. This
/// avoids pointer-address/generation reuse masquerading as a zero-copy hit.
pub struct FixedKvWindow {
    hip: Arc<HipRuntime>,
    device: i32,
    layout: ArchiveLayout,
    sides: Vec<DeviceBuffer>,
    map: WindowMap,
}

impl FixedKvWindow {
    pub fn new(hip: Arc<HipRuntime>, layout: ArchiveLayout) -> HipResult<Self> {
        let device = hip.current_device()?;
        let mut window = Self {
            hip,
            device,
            layout,
            sides: Vec::with_capacity(layout.sides),
            map: WindowMap::new(layout),
        };
        for _ in 0..layout.sides {
            window
                .sides
                .push(window.hip.malloc(layout.device_side_bytes)?);
        }
        Ok(window)
    }

    /// Fixed (FA layer, K/V) order. History bytes may only be changed by `stage`;
    /// the future forward owner may write its separate generation reserve.
    pub fn sides(&self) -> &[DeviceBuffer] {
        &self.sides
    }

    pub fn keys(&self) -> Option<&[BlockKey]> {
        self.map.keys()
    }
}

impl Drop for FixedKvWindow {
    fn drop(&mut self) {
        let previous = self.hip.current_device().ok();
        if self.hip.set_device(self.device).is_err() || self.hip.device_synchronize().is_err() {
            // Fail closed: do not free allocations a faulted device may still
            // access. Keep both backing descriptors and the HIP library alive.
            std::mem::forget(std::mem::take(&mut self.sides));
            std::mem::forget(Arc::clone(&self.hip));
        } else {
            for side in self.sides.drain(..) {
                let _ = self.hip.free(side);
            }
        }
        if let Some(previous) = previous {
            let _ = self.hip.set_device(previous);
        }
    }
}

struct HostBlock {
    key: BlockKey,
    // [layer0 K][layer0 V][layer1 K][layer1 V]..., token-major within a side.
    bytes: PinnedHostBuffer,
}

/// Immutable committed history in lazily allocated pinned host blocks. No VMM,
/// host-mapped GPU KV, CASK eviction or fallback to full device history.
pub struct PinnedKvArchive {
    layout: ArchiveLayout,
    owner: u64,
    revision: u64,
    blocks: BTreeMap<usize, HostBlock>,
}

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

impl PinnedKvArchive {
    pub fn new(layout: ArchiveLayout) -> HipResult<Self> {
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| invalid("tiered KV: owner identity exhausted"))?;
        Ok(Self {
            layout,
            owner,
            revision: 0,
            blocks: BTreeMap::new(),
        })
    }

    pub fn committed_tokens(&self) -> usize {
        self.blocks.len() * self.layout.config.block_tokens
    }

    pub fn host_bytes(&self) -> usize {
        self.blocks.len() * self.layout.block_bytes
    }

    /// Diagnostic readback of every selected encoded byte against the immutable
    /// archive. Deliberately synchronous; not a fast-path check.
    pub fn verify_window(&mut self, window: &FixedKvWindow) -> HipResult<usize> {
        if window.layout != self.layout || window.hip.current_device()? != window.device {
            return Err(invalid("tiered KV: verification owner/layout mismatch"));
        }
        let keys = window
            .keys()
            .ok_or_else(|| invalid("tiered KV: unpublished window"))?;
        window.hip.device_synchronize()?;
        let mut actual = vec![0u8; self.layout.block_side_bytes];
        let mut verified = 0;
        for (slot, key) in keys.iter().enumerate() {
            let block = self
                .blocks
                .get_mut(&key.start_token)
                .filter(|block| block.key == *key)
                .ok_or_else(|| invalid("tiered KV: stale verification key"))?;
            for (side, target) in window.sides.iter().enumerate() {
                window.hip.memcpy_dtoh_at(
                    &mut actual,
                    target,
                    slot * self.layout.block_side_bytes,
                )?;
                let begin = side * self.layout.block_side_bytes;
                if actual != block.bytes.as_bytes_mut()[begin..begin + self.layout.block_side_bytes]
                {
                    return Err(invalid(
                        "tiered KV: staged bytes differ from archived bytes",
                    ));
                }
                verified += actual.len();
            }
        }
        Ok(verified)
    }

    /// Publish the next full block after ALL sides have copied successfully.
    /// Source sides must be in the same (FA layer, K/V) order as the layout.
    /// This method cannot decide whether model/speculative rows are committed;
    /// the future model owner must supply that proof before calling it.
    pub fn append_from_device(
        &mut self,
        hip: &HipRuntime,
        source_sides: &[&DeviceBuffer],
        source_start_row: usize,
    ) -> HipResult<BlockKey> {
        let start_token = self.committed_tokens();
        if start_token >= self.layout.config.max_history_tokens {
            return Err(invalid("tiered KV: host history capacity exhausted"));
        }
        let source_offset = product(source_start_row, self.layout.config.row_bytes_per_side)?;
        let source_end = source_offset
            .checked_add(self.layout.block_side_bytes)
            .ok_or_else(|| invalid("tiered KV: source span overflow"))?;
        if source_sides.len() != self.layout.sides
            || source_sides.iter().any(|s| s.size() < source_end)
        {
            return Err(invalid("tiered KV: source side count or extent mismatch"));
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| invalid("tiered KV: block revision exhausted"))?;
        // Drain earlier kernels before reading their committed KV. No archive
        // owner has outstanding async operations, including on failure paths.
        hip.device_synchronize()?;
        let mut bytes = hip.host_buffer(self.layout.block_bytes)?;
        for (side, source) in source_sides.iter().enumerate() {
            let begin = side * self.layout.block_side_bytes;
            hip.memcpy_dtoh_at(
                &mut bytes.as_bytes_mut()[begin..begin + self.layout.block_side_bytes],
                source,
                source_offset,
            )?;
        }
        let key = BlockKey {
            owner: self.owner,
            revision,
            start_token,
        };
        self.blocks.insert(start_token, HostBlock { key, bytes });
        self.revision = revision;
        Ok(key)
    }

    /// First prototype supports only block-aligned divergent resume. Reject an
    /// intra-block checkpoint rather than silently discarding/reusing its rows.
    /// Revision never rewinds; the divergent suffix gets fresh identities.
    pub fn truncate(&mut self, tokens: usize, window: &mut FixedKvWindow) -> HipResult<()> {
        if tokens > self.committed_tokens()
            || !tokens.is_multiple_of(self.layout.config.block_tokens)
        {
            return Err(invalid(
                "tiered KV: truncate requires an existing block boundary",
            ));
        }
        window.map.invalidate();
        self.blocks.retain(|&start, _| start < tokens);
        Ok(())
    }

    /// Install only selected complete blocks, sorted by the caller, leaving the
    /// dedicated generation reserve untouched. Returns actual H2D payload bytes.
    /// If any transfer fails, `window.keys()` is None and attention MUST NOT run.
    pub fn stage(
        &mut self,
        window: &mut FixedKvWindow,
        selected_starts: &[usize],
    ) -> HipResult<usize> {
        if window.layout != self.layout
            || selected_starts.len() > window.map.capacity
            || window.hip.current_device()? != window.device
        {
            return Err(invalid(
                "tiered KV: device sides must match the exact fixed allocation",
            ));
        }
        let hip = &window.hip;
        let map = &mut window.map;
        let keys: Vec<_> = selected_starts
            .iter()
            .map(|start| {
                self.blocks
                    .get(start)
                    .map(|b| b.key)
                    .ok_or_else(|| invalid("tiered KV: selected block is not committed"))
            })
            .collect::<HipResult<_>>()?;
        validate_keys(&keys, map.capacity)?;
        // A failed drain cannot certify the previous mapping as safe to read.
        if let Err(error) = hip.device_synchronize() {
            map.invalidate();
            return Err(error);
        }
        let changed = map.install(&keys, |slot, key| {
            let bytes = self
                .blocks
                .get_mut(&key.start_token)
                .expect("validated block")
                .bytes
                .as_bytes_mut();
            for (side, target) in window.sides.iter().enumerate() {
                let begin = side * self.layout.block_side_bytes;
                hip.memcpy_htod_offset(
                    target,
                    slot * self.layout.block_side_bytes,
                    &bytes[begin..begin + self.layout.block_side_bytes],
                )?;
            }
            hip.device_synchronize()
        })?;
        Ok(changed * self.layout.block_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ArchiveConfig {
        ArchiveConfig {
            block_tokens: 128,
            row_bytes_per_side: 1088,
            full_attention_layers: 16,
            max_history_tokens: 262144,
            window_blocks: 256,
            generation_tokens: 8192,
            host_limit_bytes: 10 << 30,
            device_limit_bytes: 2 << 30,
        }
    }

    fn key(owner: u64, start_token: usize, revision: u64) -> BlockKey {
        BlockKey {
            owner,
            revision,
            start_token,
        }
    }

    #[test]
    fn qwen27_q8_bytes_include_both_sides_and_generation_reserve() {
        let layout = ArchiveLayout::new(config()).unwrap();
        assert_eq!(layout.host_capacity_bytes(), 9_126_805_504);
        assert_eq!(layout.device_capacity_bytes(), 1_426_063_360);
        assert_eq!(layout.selected_capacity_tokens(), 32768);
    }

    #[test]
    fn byte_budgets_refuse_before_allocation() {
        let base = ArchiveLayout::new(config()).unwrap();
        for c in [
            ArchiveConfig {
                host_limit_bytes: base.host_capacity_bytes() - 1,
                ..config()
            },
            ArchiveConfig {
                device_limit_bytes: base.device_capacity_bytes() - 1,
                ..config()
            },
            ArchiveConfig {
                block_tokens: 0,
                ..config()
            },
            ArchiveConfig {
                generation_tokens: 0,
                ..config()
            },
            ArchiveConfig {
                max_history_tokens: 262143,
                ..config()
            },
            ArchiveConfig {
                full_attention_layers: usize::MAX,
                ..config()
            },
            ArchiveConfig {
                window_blocks: usize::MAX,
                ..config()
            },
            ArchiveConfig {
                row_bytes_per_side: usize::MAX,
                ..config()
            },
            ArchiveConfig {
                generation_tokens: usize::MAX,
                ..config()
            },
            ArchiveConfig {
                max_history_tokens: 128,
                ..config()
            },
        ] {
            assert!(ArchiveLayout::new(c).is_err(), "{c:?}");
        }
    }

    #[test]
    fn physical_slots_do_not_replace_absolute_positions() {
        let layout = ArchiveLayout::new(config()).unwrap();
        let keys = [key(1, 0, 1), key(1, 8192, 65), key(1, 261888, 2047)];
        let positions = layout.absolute_positions(&keys).unwrap();
        assert_eq!(positions.len(), 384);
        assert_eq!(&positions[128..256], &(8192..8320).collect::<Vec<_>>());
        assert_eq!(positions[383], 262015);
        assert!(layout.absolute_positions(&[key(1, 1, 1)]).is_err());
        assert!(layout.absolute_positions(&[key(1, 262144, 1)]).is_err());
    }

    #[test]
    fn unchanged_blocks_have_zero_transfers_and_one_change_copies_once() {
        let mut map = WindowMap::new(ArchiveLayout::new(config()).unwrap());
        let a = [key(1, 0, 1), key(1, 256, 3)];
        assert_eq!(map.install(&a, |_, _| Ok(())).unwrap(), 2);
        assert_eq!(
            map.install(&a, |_, _| panic!("unexpected transfer"))
                .unwrap(),
            0
        );
        let b = [a[0], key(1, 512, 5)];
        let mut slots = vec![];
        assert_eq!(
            map.install(&b, |slot, _| {
                slots.push(slot);
                Ok(())
            })
            .unwrap(),
            1
        );
        assert_eq!(slots, [1]);
        assert_eq!(map.keys().unwrap(), b);
    }

    #[test]
    fn duplicate_unsorted_mixed_owner_and_overbudget_selection_is_atomic() {
        let mut map = WindowMap::new(ArchiveLayout::new(config()).unwrap());
        let a = [key(1, 0, 1), key(1, 256, 3)];
        map.install(&a, |_, _| Ok(())).unwrap();
        for bad in [
            vec![a[0], a[0]],
            vec![a[1], a[0]],
            vec![a[0], key(2, 256, 3)],
            vec![a[0]; 257],
        ] {
            assert!(map.install(&bad, |_, _| panic!("must not copy")).is_err());
            assert_eq!(map.keys().unwrap(), a);
        }
    }

    #[test]
    fn partial_transfer_failure_poison_and_retry_recopies_every_block() {
        let mut map = WindowMap::new(ArchiveLayout::new(config()).unwrap());
        let old = [key(1, 0, 1), key(1, 128, 2)];
        map.install(&old, |_, _| Ok(())).unwrap();
        let new = [key(1, 256, 3), key(1, 384, 4)];
        assert!(map
            .install(&new, |slot, _| {
                if slot == 1 {
                    Err(invalid("injected second block copy failure"))
                } else {
                    Ok(())
                }
            })
            .is_err());
        assert!(map.keys().is_none());
        assert_eq!(map.install(&old, |_, _| Ok(())).unwrap(), 2);
    }

    #[test]
    fn revised_suffix_or_recycled_request_never_reuses_stale_bytes() {
        let mut map = WindowMap::new(ArchiveLayout::new(config()).unwrap());
        map.install(&[key(1, 0, 1), key(1, 128, 2)], |_, _| Ok(()))
            .unwrap();
        assert_eq!(
            map.install(&[key(1, 0, 1), key(1, 128, 3)], |_, _| Ok(()))
                .unwrap(),
            1
        );
        assert_eq!(
            map.install(&[key(2, 0, 1), key(2, 128, 2)], |_, _| Ok(()))
                .unwrap(),
            2
        );
        map.invalidate();
        assert!(map.keys().is_none());
    }
}
