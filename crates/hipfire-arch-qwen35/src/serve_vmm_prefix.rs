// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Cross-session prefix cache for the VMM multi-slot route.
//!
//! The arena route shares 128-token physical pages through `PagePool`; VMM
//! owners are linear reservations with no page table, so this cache keeps a
//! bounded number of **host snapshots** instead. A snapshot is the exact
//! payload `VmmSlotKv::capture_slot` produces (per-FA-layer KV prefix plus the
//! DeltaNet state) at a committed prompt boundary. A later cold request whose
//! prompt starts with the entry's token span restores it into its own owner
//! and prefills only the suffix.
//!
//! Correctness properties:
//! * every entry is keyed by the exact token prefix and the full
//!   [`SnapshotStamp`] (model, KV tier, strides, layer count, DN layout, cap);
//! * a hit restores KV **and** recurrent state together, or is discarded
//!   (a KV-only restore would silently corrupt the suffix);
//! * the snapshot is taken only after a successful prefill chunk, before the
//!   next chunk can overwrite the prefix, and only when a suffix remains (a
//!   full-prompt snapshot has no logits to continue from);
//! * retained bytes are LRU-bounded by `serve.prefix_cache_max_bytes`.
//!
//! This is a host-memory cache, not a physical-page sharing scheme. It trades
//! RAM for correctness and VA stability: no owner churn and no retired VA.
//! A granule-fork cache remains a possible future optimization if measured
//! host pressure justifies it.

use hipfire_runtime::swap::snapshot::{SlotSnapshot, SnapshotStamp};

/// Shortest prefix worth retaining. One prefill chunk is normally 512
/// tokens; this keeps short prompts from filling the cache with one-shot
/// entries that cannot be shared.
pub const VMM_PREFIX_MIN_TOKENS: usize = 256;

struct Entry {
    tokens: Vec<u32>,
    snapshot: SlotSnapshot,
    last_used: u64,
}

/// Bounded LRU of host VMM prefix snapshots.
pub struct VmmPrefixCache {
    max_bytes: usize,
    used_bytes: usize,
    entries: Vec<Entry>,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl VmmPrefixCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            used_bytes: 0,
            entries: Vec::new(),
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn stats(&self) -> (usize, usize, u64, u64) {
        (self.entries.len(), self.used_bytes, self.hits, self.misses)
    }

    fn entry_bytes(tokens: &[u32], snapshot: &SlotSnapshot) -> usize {
        std::mem::size_of::<Entry>()
            + tokens.len() * std::mem::size_of::<u32>()
            + snapshot.payload.len()
            + snapshot.tokens.len() * std::mem::size_of::<u32>()
    }

    /// Longest cached prefix of `prompt` that validates against `stamp`.
    /// Returns `(prefix_len, snapshot)`; the caller owns the clone.
    pub fn lookup(
        &mut self,
        prompt: &[u32],
        stamp: SnapshotStamp,
    ) -> Option<(usize, SlotSnapshot)> {
        self.clock += 1;
        let mut best: Option<usize> = None;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.tokens.len() > prompt.len()
                || entry.snapshot.seq_len != entry.tokens.len()
                || entry.snapshot.stamp != stamp
            {
                continue;
            }
            if prompt[..entry.tokens.len()] != entry.tokens[..] {
                continue;
            }
            if entry.snapshot.validate(stamp).is_err() {
                continue;
            }
            if best.is_none_or(|b| entry.tokens.len() > self.entries[b].tokens.len()) {
                best = Some(i);
            }
        }
        match best {
            Some(i) => {
                self.entries[i].last_used = self.clock;
                self.hits += 1;
                Some((
                    self.entries[i].tokens.len(),
                    self.entries[i].snapshot.clone(),
                ))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Retain one successful prefill boundary. Rejects invalid stamps,
    /// mismatched token lengths, and entries larger than the whole cache.
    /// Returns whether the entry was retained.
    pub fn insert(
        &mut self,
        tokens: Vec<u32>,
        snapshot: SlotSnapshot,
        stamp: SnapshotStamp,
    ) -> bool {
        if self.max_bytes == 0
            || tokens.len() < VMM_PREFIX_MIN_TOKENS
            || snapshot.seq_len != tokens.len()
            || snapshot.stamp != stamp
            || snapshot.validate(stamp).is_err()
        {
            return false;
        }
        let bytes = Self::entry_bytes(&tokens, &snapshot);
        if bytes > self.max_bytes {
            return false;
        }
        // Exact-token replacement: a later boundary at the same prefix is
        // strictly newer (same tokens, later state is not possible at the
        // same length, but a re-capture after a reset can be).
        if let Some(i) = self.entries.iter().position(|e| e.tokens == tokens) {
            self.used_bytes = self.used_bytes.saturating_sub(Self::entry_bytes(
                &self.entries[i].tokens,
                &self.entries[i].snapshot,
            ));
            self.entries.remove(i);
        }
        while self.used_bytes.saturating_add(bytes) > self.max_bytes && !self.entries.is_empty() {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
                .expect("non-empty");
            let e = self.entries.remove(victim);
            self.used_bytes = self
                .used_bytes
                .saturating_sub(Self::entry_bytes(&e.tokens, &e.snapshot));
        }
        if self.used_bytes.saturating_add(bytes) > self.max_bytes {
            return false;
        }
        self.clock += 1;
        self.entries.push(Entry {
            tokens,
            snapshot,
            last_used: self.clock,
        });
        self.used_bytes += bytes;
        true
    }

    /// Drop the entry whose token span is exactly `prefix` (a failed restore
    /// proved it unusable).
    pub fn forget(&mut self, prefix: &[u32]) {
        if let Some(i) = self.entries.iter().position(|e| e.tokens == prefix) {
            let e = self.entries.remove(i);
            self.used_bytes = self
                .used_bytes
                .saturating_sub(Self::entry_bytes(&e.tokens, &e.snapshot));
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.used_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::swap::snapshot::checksum_of;

    fn stamp() -> SnapshotStamp {
        SnapshotStamp {
            model_hash: 7,
            instance_id: 9,
            kv_dtype_tag: 3,
            per_pos_bytes: 4,
            per_pos_v_bytes: 0,
            n_fa_layers: 1,
            dn_layout_version: 1,
            cap: 4096,
            dn_bytes: 0,
        }
    }

    fn snap(tokens: &[u32], stamp: SnapshotStamp) -> SlotSnapshot {
        let payload = tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<u8>>();
        SlotSnapshot {
            stamp,
            seq_len: tokens.len(),
            tokens: tokens.to_vec(),
            checksum: checksum_of(&payload),
            payload,
        }
    }

    #[test]
    fn lookup_returns_the_longest_valid_prefix() {
        let s = stamp();
        let mut c = VmmPrefixCache::new(1 << 20);
        let short: Vec<u32> = (0..VMM_PREFIX_MIN_TOKENS as u32).collect();
        let long: Vec<u32> = (0..VMM_PREFIX_MIN_TOKENS as u32 + 128).collect();
        assert!(c.insert(short.clone(), snap(&short, s), s));
        assert!(c.insert(long.clone(), snap(&long, s), s));
        let prompt: Vec<u32> = long.iter().copied().chain([99, 100]).collect();
        let (n, found) = c.lookup(&prompt, s).expect("hit");
        assert_eq!(n, long.len());
        assert_eq!(found.tokens, long);
    }

    #[test]
    fn wrong_tokens_or_stamp_miss() {
        let s = stamp();
        let mut c = VmmPrefixCache::new(1 << 20);
        let tokens: Vec<u32> = (0..VMM_PREFIX_MIN_TOKENS as u32).collect();
        assert!(c.insert(tokens.clone(), snap(&tokens, s), s));
        let mut other = tokens.clone();
        other[0] = 1;
        assert!(c.lookup(&other, s).is_none());
        let mut wrong = s;
        wrong.model_hash += 1;
        assert!(c.lookup(&tokens, wrong).is_none());
    }

    #[test]
    fn lru_evicts_to_fit_and_clears_accounting() {
        let s = stamp();
        let tokens: Vec<u32> = (0..VMM_PREFIX_MIN_TOKENS as u32).collect();
        let one = snap(&tokens, s);
        let bytes = VmmPrefixCache::entry_bytes(&tokens, &one);
        let mut c = VmmPrefixCache::new(bytes * 2);
        assert!(c.insert(tokens.clone(), one.clone(), s));
        let mut t2 = tokens.clone();
        t2[1] = 2;
        assert!(c.insert(t2.clone(), snap(&t2, s), s));
        let mut t3 = tokens.clone();
        t3[2] = 3;
        assert!(c.insert(t3.clone(), snap(&t3, s), s));
        assert_eq!(c.len(), 2, "third insert evicts the LRU entry");
        assert!(c.used_bytes() <= c.max_bytes());
        assert!(c.lookup(&tokens, s).is_none(), "oldest entry was evicted");
        c.clear();
        assert_eq!(c.used_bytes(), 0);
        assert!(c.is_empty());
    }

    #[test]
    fn short_or_invalid_entries_are_not_retained() {
        let s = stamp();
        let mut c = VmmPrefixCache::new(1 << 20);
        let short: Vec<u32> = (0..VMM_PREFIX_MIN_TOKENS as u32 - 1).collect();
        assert!(!c.insert(short.clone(), snap(&short, s), s));
        let tokens: Vec<u32> = (0..VMM_PREFIX_MIN_TOKENS as u32).collect();
        let mut bad = snap(&tokens, s);
        bad.checksum ^= 1;
        assert!(!c.insert(tokens.clone(), bad, s));
    }
}
