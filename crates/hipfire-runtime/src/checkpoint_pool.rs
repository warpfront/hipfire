//! Resumable-prefix checkpoint pool (spec §4.5 C5).
//!
//! Family-free: a page-aligned LRU of recurrent-state checkpoint bundles
//! keyed by `(CacheDomain, boundary_p, prefix_fingerprint)`. Each
//! `CheckpointBlob` is one model family's recurrent-state snapshot; the
//! family crate implements the trait and provides capture/restore. The
//! radix index (P2-index) stores only `CheckpointId` + boundary; this pool
//! owns the bytes. Ids are not stable across pool restarts.
//!
//! Moved out of `hipfire-arch-qwen35` so other hybrid families (DeltaNet,
//! conv-tail, …) share the pool rather than re-implementing it.

use crate::serve_contract::{
    CacheDomain, DrafterDecision, LastTokenHandling, MissReason, PrefixLookup, ResumeBundle,
    ResumePlan, ResumePlanError,
};
use rdna_compute::page_pool::PAGE_TOKENS;
use std::collections::{HashMap, HashSet};

pub use crate::serve_contract::CheckpointId;

// ───────────────────────────────────────────────────────────────────────────
// CheckpointBlob trait
// ───────────────────────────────────────────────────────────────────────────

/// Abstraction over the stored checkpoint bytes, used so the pool can be
/// tested on the host without GPU device buffers (spec §4.5 C5).
///
/// For GPU use, `DeltaNetSnapshot` implements this via its `bytes_len()`
/// method. For host tests, a simple byte-counting test double suffices.
pub trait CheckpointBlob {
    /// Total device/host bytes this blob occupies, for LRU accounting.
    fn bytes_len(&self) -> u64;
}



// ───────────────────────────────────────────────────────────────────────────
// Pool entry
// ───────────────────────────────────────────────────────────────────────────

/// Key for a checkpoint: `(domain, boundary_p, prefix_fingerprint)`.
///
/// The fingerprint is `prefix_fingerprint(tokens[..p])` — a hash of the
/// exact token prefix that produced the captured recurrent state. Two
/// prompts that share a `p`-token prefix produce the SAME fingerprint and
/// correctly share the checkpoint (the state at boundary `p` is identical);
/// two prompts that diverge before `p` produce DIFFERENT fingerprints and
/// get distinct entries. Without it, a second prompt capturing at the same
/// `(domain, p)` would replace the first prompt's recurrent state while the
/// first prompt's radix node still pointed at that `CheckpointId` — a
/// silent cross-prefix state clobber (the second audit's R1).
type CheckpointKey = (CacheDomain, u64, u64);

/// SHA-256 (first 8 bytes, little-endian) over the little-endian token
/// bytes of a prefix.
///
/// Deterministic across processes. Previously FNV-1a, which is not
/// collision-resistant: a crafted second prefix colliding on `(domain, p,
/// fp)` could make `plan_resume` restore another tenant's recurrent state
/// on top of this request's KV pages (the radix only gates the token PATH,
/// and `insert` reuses the colliding entry's `CheckpointId`). A
/// collision-resistant fingerprint closes that wrong-state restore — a
/// fingerprint match now also certifies the prefix contents.
pub fn prefix_fingerprint(tokens: &[u32]) -> u64 {
    use sha2::{Digest as Sha256Digest, Sha256};
    let mut h = Sha256::new();
    for &t in tokens {
        h.update(t.to_le_bytes());
    }
    let d = h.finalize();
    u64::from_le_bytes(d[..8].try_into().expect("sha256 >= 8 bytes"))
}

#[derive(Debug)]
struct CheckpointEntry<B> {
    id: CheckpointId,
    blob: B,
    pinned: bool,
    /// Monotonic LRU access stamp; smaller = older.
    lru_stamp: u64,
}

// ───────────────────────────────────────────────────────────────────────────
// QwenCheckpointPool
// ───────────────────────────────────────────────────────────────────────────

/// Byte-bounded LRU pool of immutable Qwen3.5 hybrid-state checkpoint
/// bundles (spec §4.5 C5).
///
/// Keyed by `(CacheDomain, boundary_p)`. Entries are captured only at
/// page-aligned boundaries. When the pool exceeds `max_bytes`, the oldest
/// **unpinned** checkpoint is evicted until the pool fits. Pinned
/// checkpoints survive eviction.
///
/// Generic over `B: CheckpointBlob` so host tests can use a byte-counting
/// test double without GPU device buffers. The GPU-backed capture and
/// restore paths use `B = DeltaNetSnapshot`.
pub struct QwenCheckpointPool<B: CheckpointBlob> {
    entries: HashMap<CheckpointKey, CheckpointEntry<B>>,
    /// Keys that were explicitly evicted (for distinguishing
    /// [`MissReason::Evicted`] from [`MissReason::NoCheckpoint`]).
    evicted: HashSet<CheckpointKey>,
    total_bytes: u64,
    max_bytes: u64,
    next_id: u64,
    lru_clock: u64,
}

impl<B: CheckpointBlob> QwenCheckpointPool<B> {
    /// Create a pool with a byte capacity of `max_bytes`.
    pub fn new(max_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            evicted: HashSet::new(),
            total_bytes: 0,
            max_bytes,
            next_id: 1,
            lru_clock: 0,
        }
    }

    /// Maximum byte capacity.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Current total bytes across all entries.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Number of entries currently in the pool.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether a blob of `bytes` could be inserted without exceeding the
    /// ceiling — accounting for the entry it would replace at `key` and for
    /// evicting every unpinned entry. Lets `capture_checkpoint` refuse
    /// BEFORE paying for the GPU allocation and device-to-device copy.
    pub fn can_afford(&self, key: &CheckpointKey, bytes: u64) -> bool {
        let replaced = self
            .entries
            .get(key)
            .map(|e| e.blob.bytes_len())
            .unwrap_or(0);
        let unpinned_bytes: u64 = self
            .entries
            .iter()
            .filter(|(k, e)| !e.pinned && **k != *key)
            .map(|(_, e)| e.blob.bytes_len())
            .sum();
        let floor = self.total_bytes.saturating_sub(replaced + unpinned_bytes);
        floor.saturating_add(bytes) <= self.max_bytes
    }

    /// Verify `p` is a valid capture boundary: `p == 0` or `p` is a
    /// multiple of [`PAGE_TOKENS`].
    pub fn is_aligned(p: u64) -> bool {
        p == 0 || p % PAGE_TOKENS as u64 == 0
    }

    /// Insert (or replace) a captured checkpoint blob at `(domain, p)`.
    ///
    /// `p` must be page-aligned (`p % 128 == 0` or `p == 0`); otherwise the
    /// entry is **not** inserted and [`CheckpointId::NONE`] is returned
    /// (spec §4.5: "a state at the end of a chunk cannot be relabelled as
    /// an earlier state").
    ///
    /// If the pool cannot afford the new capture, the oldest **unpinned**
    /// checkpoint is evicted repeatedly until the pool fits. If every
    /// remaining entry is pinned and the capture still does not fit, the NEW
    /// capture is dropped and [`CheckpointId::NONE`] is returned — the byte
    /// ceiling is a hard bound (spec §4.4, §9.1: "a cache-retention limit is
    /// a ceiling"), never oversubscribed; the caller simply publishes
    /// without a checkpoint and the boundary stays honestly unresumable.
    ///
    /// Returns the minted [`CheckpointId`] for the radix index to store, or
    /// [`CheckpointId::NONE`] when the capture was refused.
    ///
    /// GPU memory discipline: every blob this call displaces (the replaced
    /// entry at the same key, LRU evictions, and a refused capture itself)
    /// is RETURNED to the caller. `DeviceBuffer` has no freeing `Drop`, so
    /// a blob dropped here would leak its device memory permanently while
    /// `total_bytes` is decremented as if freed. The caller routes each
    /// returned blob through `free_gpu`.
    pub fn insert(
        &mut self,
        domain: CacheDomain,
        p: u64,
        fp: u64,
        blob: B,
    ) -> (CheckpointId, Vec<B>) {
        let mut displaced: Vec<B> = Vec::new();
        if !Self::is_aligned(p) {
            displaced.push(blob);
            return (CheckpointId::NONE, displaced);
        }

        let bytes = blob.bytes_len();
        let key = (domain, p, fp);

        // If an entry already exists at this key, replace it and KEEP its
        // id: radix nodes store the CheckpointId from the first capture, so
        // minting a fresh id on re-capture would strand those references.
        let mut reuse_id = None;
        if let Some(old) = self.entries.remove(&key) {
            self.total_bytes = self.total_bytes.saturating_sub(old.blob.bytes_len());
            reuse_id = Some(old.id);
            displaced.push(old.blob);
        }

        // Clear any prior eviction record for this key.
        self.evicted.remove(&key);

        // Evict oldest unpinned until we can afford the new blob.
        while self.total_bytes + bytes > self.max_bytes {
            match self.find_oldest_unpinned_key() {
                Some(evict_key) => {
                    if let Some(blob) = self.evict_internal(evict_key) {
                        displaced.push(blob);
                    }
                }
                None => {
                    // Everything left is pinned and the ceiling cannot be
                    // honored. Refuse the capture (handing the blob back for
                    // the caller to free) rather than exceeding the ceiling.
                    displaced.push(blob);
                    return (CheckpointId::NONE, displaced);
                }
            }
        }

        let id = reuse_id.unwrap_or_else(|| {
            let id = CheckpointId(self.next_id);
            self.next_id += 1;
            id
        });
        self.lru_clock += 1;
        self.total_bytes += bytes;
        self.entries.insert(
            key,
            CheckpointEntry {
                id,
                blob,
                pinned: false,
                lru_stamp: self.lru_clock,
            },
        );

        (id, displaced)
    }

    /// Find the key of the oldest (smallest `lru_stamp`) unpinned entry.
    fn find_oldest_unpinned_key(&self) -> Option<CheckpointKey> {
        self.entries
            .iter()
            .filter(|(_, e)| !e.pinned)
            .min_by_key(|(_, e)| e.lru_stamp)
            .map(|(k, _)| k.clone())
    }

    /// Remove an entry by key, accounting bytes and recording eviction.
    /// Returns the removed blob for the caller to free on the GPU — never
    /// dropped here (see [`Self::insert`] for the no-`Drop` rationale).
    fn evict_internal(&mut self, key: CheckpointKey) -> Option<B> {
        self.entries.remove(&key).map(|entry| {
            self.total_bytes = self.total_bytes.saturating_sub(entry.blob.bytes_len());
            self.evicted.insert(key);
            entry.blob
        })
    }

    /// Explicitly evict the checkpoint at `(domain, p, fp)`.
    ///
    /// Returns the removed blob (for the caller to `free_gpu`), or `None`
    /// when no entry existed.
    pub fn evict(&mut self, domain: &CacheDomain, p: u64, fp: u64) -> Option<B> {
        let key = (domain.clone(), p, fp);
        self.evict_internal(key)
    }

    /// Check whether a checkpoint exists at `(domain, p, fp)`.
    pub fn contains(&self, domain: &CacheDomain, p: u64, fp: u64) -> bool {
        self.entries.contains_key(&(domain.clone(), p, fp))
    }

    /// Get the [`CheckpointId`] for `(domain, p, fp)`, if present.
    pub fn id_of(&self, domain: &CacheDomain, p: u64, fp: u64) -> Option<CheckpointId> {
        self.entries.get(&(domain.clone(), p, fp)).map(|e| e.id)
    }

    /// Borrow the blob at `(domain, p, fp)`, refreshing its LRU stamp.
    pub fn get(&mut self, domain: &CacheDomain, p: u64, fp: u64) -> Option<&B> {
        let key = (domain.clone(), p, fp);
        if let Some(entry) = self.entries.get_mut(&key) {
            self.lru_clock += 1;
            entry.lru_stamp = self.lru_clock;
            Some(&entry.blob)
        } else {
            None
        }
    }

    /// Borrow the blob at `(domain, p, fp)` without refreshing LRU (read-only).
    pub fn peek(&self, domain: &CacheDomain, p: u64, fp: u64) -> Option<&B> {
        self.entries.get(&(domain.clone(), p, fp)).map(|e| &e.blob)
    }

    /// Pin the checkpoint at `(domain, p, fp)` so it survives LRU eviction.
    ///
    /// Returns `true` if the entry was found and pinned.
    pub fn pin(&mut self, domain: &CacheDomain, p: u64, fp: u64) -> bool {
        let key = (domain.clone(), p, fp);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.pinned = true;
            true
        } else {
            false
        }
    }

    /// Unpin the checkpoint at `(domain, p, fp)`, making it eligible for LRU
    /// eviction again.
    ///
    /// Returns `true` if the entry was found and unpinned.
    pub fn unpin(&mut self, domain: &CacheDomain, p: u64, fp: u64) -> bool {
        let key = (domain.clone(), p, fp);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.pinned = false;
            true
        } else {
            false
        }
    }

    /// Whether the checkpoint at `(domain, p, fp)` is pinned.
    pub fn is_pinned(&self, domain: &CacheDomain, p: u64, fp: u64) -> bool {
        self.entries
            .get(&(domain.clone(), p, fp))
            .map(|e| e.pinned)
            .unwrap_or(false)
    }

    /// Drain and return all stored blobs, clearing the pool. Used by the
    /// serve engine's `free_gpu` to explicitly free each `DeltaNetSnapshot`'s
    /// device buffers on shutdown (the pool itself has no `Drop` impl that
    /// touches the GPU).
    pub fn drain_blobs(&mut self) -> Vec<B> {
        self.total_bytes = 0;
        self.entries.drain().map(|(_, e)| e.blob).collect()
    }

    /// Check whether a checkpoint at any page-aligned boundary `≤ max_p`
    /// for `domain` was previously evicted (for [`MissReason::Evicted`]
    /// reporting). Matches on `(domain, p)` regardless of fingerprint — a
    /// boundary that had ANY checkpoint evicted is reported as Evicted.
    fn was_evicted(&self, domain: &CacheDomain, max_p: u64) -> bool {
        self.evicted
            .iter()
            .any(|(d, p, _fp)| d == domain && *p <= max_p && *p > 0)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// plan_resume
// ───────────────────────────────────────────────────────────────────────────

/// Build a [`ResumeBundle`] with all required component flags true.
///
/// When a checkpoint exists in the pool at boundary `p`, the snapshot
/// covers DN matrices, scales, conv rings/indices, and EF residuals.
/// `attention_pages` is true because `lookup.resumable_tokens >= p`
/// (the index guarantees all state components exist at that boundary).
fn complete_bundle(drafter: DrafterDecision) -> ResumeBundle {
    ResumeBundle {
        attention_pages: true,
        dn_matrices_scales: true,
        conv_rings: true,
        ef_residual: true,
        drafter,
    }
}

/// Find the largest page-aligned checkpoint boundary `p > 0` with
/// `p <= max_p` in the pool for `domain` whose stored prefix fingerprint
/// matches `tokens[..p]`. The fingerprint is what makes the lookup
/// prefix-exact: a checkpoint captured under a DIFFERENT prefix at the same
/// boundary is not returned (it is a different recurrent state).
///
/// Does NOT return `p = 0` — the initial state does not require a
/// checkpoint and is handled separately by the caller.
fn find_largest_pool_checkpoint<B: CheckpointBlob>(
    pool: &QwenCheckpointPool<B>,
    domain: &CacheDomain,
    tokens: &[u32],
    max_p: u64,
) -> Option<u64> {
    let page = PAGE_TOKENS as u64;
    let mut p = (max_p / page) * page;
    while p > 0 {
        if let Some(prefix) = tokens.get(..p as usize) {
            if pool.contains(domain, p, prefix_fingerprint(prefix)) {
                return Some(p);
            }
        }
        p -= page;
    }
    None
}

/// Find the largest page-aligned checkpoint boundary `p > 0` with
/// `p < below_p` in the pool for `domain` matching `tokens[..p]`.
fn find_largest_pool_checkpoint_below<B: CheckpointBlob>(
    pool: &QwenCheckpointPool<B>,
    domain: &CacheDomain,
    tokens: &[u32],
    below_p: u64,
) -> Option<u64> {
    let page = PAGE_TOKENS as u64;
    if below_p <= page {
        return None;
    }
    let mut p = below_p - page;
    while p > 0 {
        if let Some(prefix) = tokens.get(..p as usize) {
            if pool.contains(domain, p, prefix_fingerprint(prefix)) {
                return Some(p);
            }
        }
        p -= page;
    }
    None
}

/// Plan a resume from the checkpoint pool (spec §4.5 C5).
///
/// Chooses the largest `p ≤ lookup.resumable_tokens` that is page-aligned
/// and has a complete Qwen bundle in the pool. By default uses
/// [`LastTokenHandling::SuffixRecompute`] with `p < prompt_len`.
///
/// # Last-token semantics
///
/// If `prompt_len == p` (the prompt exactly matches the cached boundary),
/// an **earlier** valid checkpoint or the initial state (`p = 0`) is
/// selected with [`LastTokenHandling::EarlierBoundary`] — never restore
/// `S_prompt_len` and re-run the last token (spec §4.5). An empty prompt
/// (`prompt_len == 0`) cannot underflow: `p = 0` is returned directly.
///
/// # Drafter decision
///
/// `drafter` is an input from admission. Reusing the target prefix is not
/// evidence that the drafter is ready; the caller decides
/// [`DrafterDecision::Checkpoint`] only when a separately identity-qualified
/// drafter checkpoint exists (spec §4.5).
///
/// # Errors
///
/// Returns [`MissReason::NoCheckpoint`] if no checkpoint exists at any
/// page-aligned boundary `≤ resumable_tokens` (and `resumable_tokens > 0`).
/// Returns [`MissReason::Evicted`] if a checkpoint was previously resident
/// but has been evicted.
///
/// # Materialized boundary
///
/// The returned `ResumePlan.boundary` is the **materialized committed
/// prefix** — tokens `[0, p)` processed by the target. It does NOT include
/// the last sampled token, which may not yet have a KV row (spec §4.5,
/// §6.1/X1). When constructing a [`hipfire_runtime::serve_contract::CommitBoundary`],
/// `committed_tokens` and `materialized_rows` must reflect only this
/// processed prefix, not the accepted token history.
pub fn plan_resume<B: CheckpointBlob>(
    pool: &mut QwenCheckpointPool<B>,
    domain: &CacheDomain,
    prompt_tokens: &[u32],
    lookup: &PrefixLookup,
    drafter: DrafterDecision,
) -> Result<ResumePlan, MissReason> {
    let plan = plan_resume_inner(pool, domain, prompt_tokens, lookup, drafter)?;
    // Pin the chosen boundary so an unrelated capture cannot evict the
    // checkpoint between this plan and the caller's restore. p == 0 is the
    // initial state — no entry exists to pin.
    if plan.boundary > 0 {
        if let Some(prefix) = prompt_tokens.get(..plan.boundary as usize) {
            pool.pin(domain, plan.boundary, prefix_fingerprint(prefix));
        }
    }
    Ok(plan)
}

fn plan_resume_inner<B: CheckpointBlob>(
    pool: &QwenCheckpointPool<B>,
    domain: &CacheDomain,
    prompt_tokens: &[u32],
    lookup: &PrefixLookup,
    drafter: DrafterDecision,
) -> Result<ResumePlan, MissReason> {
    let prompt_len = prompt_tokens.len() as u64;
    let resumable = lookup.resumable_tokens;

    // Find the largest page-aligned checkpoint p > 0, p <= resumable whose
    // stored fingerprint matches this prompt's prefix at p.
    let best_p = find_largest_pool_checkpoint(pool, domain, prompt_tokens, resumable);

    match best_p {
        Some(p) if prompt_len == p && prompt_len > 0 => {
            // Exact match: select an earlier boundary, never restore S_prompt_len.
            // Prefer an earlier checkpoint; fall back to the initial state p=0.
            let earlier = find_largest_pool_checkpoint_below(pool, domain, prompt_tokens, p);
            let ep = earlier.unwrap_or(0);
            let byte_cost = if ep == 0 {
                0
            } else {
                pool.peek(
                    domain,
                    ep,
                    prefix_fingerprint(&prompt_tokens[..ep as usize]),
                )
                .map(|b| b.bytes_len())
                .unwrap_or(0)
            };
            let bundle = complete_bundle(drafter);
            ResumePlan::new(ep, bundle, byte_cost, LastTokenHandling::EarlierBoundary).map_err(
                |e| match e {
                    ResumePlanError::MissingComponent(_) => MissReason::NoCheckpoint,
                },
            )
        }
        Some(p) => {
            // Normal: p < prompt_len (or prompt_len == 0 with p > 0 — shouldn't
            // normally happen, but SuffixRecompute is still safe).
            let byte_cost = pool
                .peek(domain, p, prefix_fingerprint(&prompt_tokens[..p as usize]))
                .map(|b| b.bytes_len())
                .unwrap_or(0);
            let bundle = complete_bundle(drafter);
            ResumePlan::new(p, bundle, byte_cost, LastTokenHandling::SuffixRecompute).map_err(|e| {
                match e {
                    ResumePlanError::MissingComponent(_) => MissReason::NoCheckpoint,
                }
            })
        }
        None => {
            // No checkpoint at any page-aligned boundary <= resumable.
            if resumable == 0 && prompt_len == 0 {
                // Empty prompt: p=0, no underflow (spec §4.5).
                let bundle = complete_bundle(drafter);
                ResumePlan::new(0, bundle, 0, LastTokenHandling::SuffixRecompute).map_err(|e| {
                    match e {
                        ResumePlanError::MissingComponent(_) => MissReason::NoCheckpoint,
                    }
                })
            } else if pool.was_evicted(domain, resumable) {
                Err(MissReason::Evicted)
            } else {
                Err(MissReason::NoCheckpoint)
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────

// ───────────────────────────────────────────────────────────────────────────
// Tests (host-only — no GPU/HIP required)
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve_contract::{
        ArchPolicy, DeviceTopology, KvLayout, SharingNamespace, TemplateIdentity, TokenizerIdentity,
    };

    /// Host test double: just byte-length accounting, no device buffers.
    #[derive(Clone, Debug)]
    struct HostBlob {
        bytes: u64,
    }

    impl CheckpointBlob for HostBlob {
        fn bytes_len(&self) -> u64 {
            self.bytes
        }
    }

    /// Build a minimal `CacheDomain` for testing.
    fn test_domain(tag: &str) -> CacheDomain {
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
                arch_tag: tag.to_string(),
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

    /// Build a `PrefixLookup` with the given `resumable_tokens`.
    fn lookup(resumable: u64) -> PrefixLookup {
        PrefixLookup {
            matched_tokens: resumable,
            resident_kv_tokens: resumable,
            resumable_tokens: resumable,
        }
    }

    const PAGE: u64 = PAGE_TOKENS as u64; // 128

    /// Deterministic prompt of `n` tokens for fingerprint tests.
    fn toks(n: u64) -> Vec<u32> {
        (0..n as u32)
            .map(|i| i.wrapping_mul(2654435761).wrapping_add(1))
            .collect()
    }

    /// Fingerprint of `toks(n)[..p]` — the prefix a checkpoint at boundary p
    /// was captured under.
    fn fp(p: u64) -> u64 {
        prefix_fingerprint(&toks(p.max(1))[..p as usize])
    }

    // ── A7: structural — capture at p=128, plan 200-token prompt ────────

    #[test]
    fn a7_capture_at_128_plan_200() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a7");

        // Capture at p=128 (page-aligned).
        let (id, _) = pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });
        assert_ne!(id, CheckpointId::NONE, "insert should mint a nonzero id");

        // Plan for a 200-token prompt: p=128 < 200 → SuffixRecompute.
        let plan = plan_resume(
            &mut pool,
            &dom,
            &toks(200),
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect("resume should succeed");

        assert_eq!(plan.boundary, 128);
        assert_eq!(plan.last_token, LastTokenHandling::SuffixRecompute);
        assert!(plan.bundle.attention_pages);
        assert!(plan.bundle.dn_matrices_scales);
        assert!(plan.bundle.conv_rings);
        assert!(plan.bundle.ef_residual);
        assert_eq!(plan.bundle.drafter, DrafterDecision::Ar);
    }

    // ── A7: exact-match prompt selects earlier boundary ─────────────────

    #[test]
    fn a7_exact_match_selects_earlier_boundary() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a7-exact");

        // Capture at p=128 only.
        pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });

        // Prompt of exactly 128 tokens: must NOT resume at p=128.
        // Should select p=0 (initial state) with EarlierBoundary.
        let plan = plan_resume(
            &mut pool,
            &dom,
            &toks(128),
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect("resume should succeed");

        assert_ne!(
            plan.boundary, 128,
            "must not restore S_prompt_len for an exact-match prompt"
        );
        assert_eq!(plan.boundary, 0);
        assert_eq!(plan.last_token, LastTokenHandling::EarlierBoundary);
    }

    // ── A7: exact match with two checkpoints selects previous page ──────

    #[test]
    fn a7_exact_match_with_prior_checkpoint() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a7-prior");

        // Capture at p=128 and p=256.
        pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });
        pool.insert(dom.clone(), 256, fp(256), HostBlob { bytes: 4096 });

        // Prompt of exactly 256 tokens: should select p=128, not p=256.
        let plan = plan_resume(
            &mut pool,
            &dom,
            &toks(256),
            &lookup(256),
            DrafterDecision::Ar,
        )
        .expect("resume should succeed");

        assert_eq!(plan.boundary, 128);
        assert_eq!(plan.last_token, LastTokenHandling::EarlierBoundary);
    }

    // ── A7: empty prompt does not underflow ──────────────────────────────

    #[test]
    fn a7_empty_prompt_no_underflow() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a7-empty");

        // No checkpoints, empty prompt, resumable=0.
        let plan = plan_resume(&mut pool, &dom, &toks(0), &lookup(0), DrafterDecision::Ar)
            .expect("empty prompt should not underflow");

        assert_eq!(plan.boundary, 0);
        assert_eq!(plan.last_token, LastTokenHandling::SuffixRecompute);
    }

    // ── A8: missing checkpoint → NoCheckpoint error ─────────────────────

    #[test]
    fn a8_missing_checkpoint_is_no_checkpoint() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a8-missing");

        // Lookup claims 128 resumable tokens, but no checkpoint in pool.
        let err = plan_resume(
            &mut pool,
            &dom,
            &toks(200),
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect_err("should be a miss");

        assert_eq!(err, MissReason::NoCheckpoint);
    }

    // ── A8: evicted checkpoint → Evicted error ──────────────────────────

    #[test]
    fn a8_evicted_checkpoint_is_evicted() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a8-evicted");

        // Insert at p=128, then evict it.
        pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });
        assert!(pool.evict(&dom, 128, fp(128)).is_some());

        // Lookup still claims 128 resumable, but checkpoint was evicted.
        let err = plan_resume(
            &mut pool,
            &dom,
            &toks(200),
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect_err("should be a miss");

        assert_eq!(err, MissReason::Evicted);
    }

    // ── A8: not a silent Hit ─────────────────────────────────────────────

    #[test]
    fn a8_missing_is_not_silent_hit() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a8-silent");

        // resumable > 0 but no checkpoint → must error, not return a plan.
        let result = plan_resume(
            &mut pool,
            &dom,
            &toks(200),
            &lookup(64),
            DrafterDecision::Ar,
        );

        assert!(result.is_err(), "must not be a silent Hit");
    }

    // ── A9: materialized boundary excludes uncomputed last token ────────

    #[test]
    fn a9_boundary_is_materialized_prefix() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a9");

        // Capture at p=128.
        pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });

        // Plan for 200-token prompt.
        let plan = plan_resume(
            &mut pool,
            &dom,
            &toks(200),
            &lookup(128),
            DrafterDecision::Ar,
        )
        .unwrap();

        // The boundary is the materialized committed prefix (tokens [0,128)
        // processed by the target). It does NOT include the 129th token or
        // any sampled-but-not-yet-materialized token. The suffix [128,200)
        // will be processed to obtain first-token logits.
        assert_eq!(plan.boundary, 128);
        assert!(plan.boundary < 200, "boundary must be < prompt_len");
        // If a CommitBoundary were constructed from this plan,
        // committed_tokens = materialized_rows = plan.boundary = 128,
        // NOT 200 (the full prompt) or 129 (a sampled last token).
    }

    // ── A9: exact-match boundary is strictly less than prompt_len ───────

    #[test]
    fn a9_exact_match_boundary_strictly_less() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("a9-exact");

        pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });
        pool.insert(dom.clone(), 256, fp(256), HostBlob { bytes: 4096 });

        let plan = plan_resume(
            &mut pool,
            &dom,
            &toks(256), // exact match
            &lookup(256),
            DrafterDecision::Ar,
        )
        .unwrap();

        // For an exact-match prompt, the resume boundary must be strictly
        // less than prompt_len — the last sampled token is NOT included
        // in the materialized prefix.
        assert!(plan.boundary < 256);
        assert_eq!(plan.boundary, 128);
    }

    // ── LRU byte bound: inserting over capacity evicts oldest unpinned ──

    #[test]
    fn lru_evicts_oldest_unpinned() {
        // Capacity: 2 entries of 4096 bytes each.
        let mut pool = QwenCheckpointPool::<HostBlob>::new(8192);
        let dom = test_domain("lru");

        let (id0, _) = pool.insert(dom.clone(), 0, fp(0), HostBlob { bytes: 4096 });
        let (id128, _) = pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.total_bytes(), 8192);

        // Insert a third — should evict the oldest (p=0).
        let (id256, _) = pool.insert(dom.clone(), 256, fp(256), HostBlob { bytes: 4096 });
        assert_eq!(pool.len(), 2, "should still have 2 entries after eviction");
        assert!(
            !pool.contains(&dom, 0, fp(0)),
            "oldest (p=0) should be evicted"
        );
        assert!(pool.contains(&dom, 128, fp(128)));
        assert!(pool.contains(&dom, 256, fp(256)));
        assert_ne!(id0, CheckpointId::NONE);
        assert_ne!(id128, CheckpointId::NONE);
        assert_ne!(id256, CheckpointId::NONE);
    }

    /// GPU-memory discipline: every blob displaced by an insert (LRU
    /// eviction, same-key replacement, ceiling refusal) is RETURNED to the
    /// caller — nothing is dropped inside the pool, because a dropped
    /// `DeltaNetSnapshot` leaks its device buffers (no freeing `Drop`).
    #[test]
    fn displaced_blobs_are_returned_never_dropped() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(8192);
        let dom = test_domain("displaced");

        // LRU eviction returns the evicted blob.
        let _ = pool.insert(dom.clone(), 0, fp(0), HostBlob { bytes: 4096 });
        let _ = pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });
        let (_, displaced) = pool.insert(dom.clone(), 256, fp(256), HostBlob { bytes: 4096 });
        assert_eq!(displaced.len(), 1, "evicted blob must be handed back");
        assert_eq!(displaced[0].bytes, 4096);
        assert!(!pool.contains(&dom, 0, fp(0)), "p=0 was the one evicted");

        // Same-key replacement returns the replaced blob.
        let (_, displaced) = pool.insert(dom.clone(), 256, fp(256), HostBlob { bytes: 2048 });
        assert_eq!(displaced.len(), 1, "replaced blob must be handed back");
        assert_eq!(displaced[0].bytes, 4096);
        assert_eq!(pool.total_bytes(), 4096 + 2048);

        // Ceiling refusal with everything pinned returns the refused blob.
        pool.pin(&dom, 128, fp(128));
        pool.pin(&dom, 256, fp(256));
        let (id, displaced) = pool.insert(dom.clone(), 384, fp(384), HostBlob { bytes: 8192 });
        assert_eq!(id, CheckpointId::NONE, "refused capture reports NONE");
        assert_eq!(
            displaced.len(),
            1,
            "the refused capture's blob must be handed back for freeing"
        );
        assert_eq!(displaced[0].bytes, 8192);

        // Unaligned boundary: same contract.
        let (id, displaced) = pool.insert(dom.clone(), 100, fp(100), HostBlob { bytes: 4096 });
        assert_eq!(id, CheckpointId::NONE);
        assert_eq!(displaced.len(), 1);
    }

    // ── LRU byte bound: pinned checkpoints survive eviction ─────────────

    #[test]
    fn lru_pinned_survives_eviction() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(8192);
        let dom = test_domain("lru-pinned");

        pool.insert(dom.clone(), 0, fp(0), HostBlob { bytes: 4096 });
        pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });

        // Pin the oldest (p=0).
        assert!(pool.pin(&dom, 0, fp(0)));
        assert!(pool.is_pinned(&dom, 0, fp(0)));

        // Insert a third — p=128 (unpinned, newer) should be evicted,
        // NOT p=0 (pinned, older).
        pool.insert(dom.clone(), 256, fp(256), HostBlob { bytes: 4096 });

        assert!(
            pool.contains(&dom, 0, fp(0)),
            "pinned p=0 must survive eviction"
        );
        assert!(
            !pool.contains(&dom, 128, fp(128)),
            "unpinned p=128 should be evicted"
        );
        assert!(pool.contains(&dom, 256, fp(256)));
    }

    // ── LRU: access refreshes recency ───────────────────────────────────

    #[test]
    fn lru_access_refreshes_recency() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(8192);
        let dom = test_domain("lru-recency");

        pool.insert(dom.clone(), 0, fp(0), HostBlob { bytes: 4096 });
        pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });

        // Access p=0 to make it more recent than p=128.
        let _ = pool.get(&dom, 0, fp(0));

        // Insert a third — p=128 (now oldest) should be evicted.
        pool.insert(dom.clone(), 256, fp(256), HostBlob { bytes: 4096 });

        assert!(
            pool.contains(&dom, 0, fp(0)),
            "recently accessed p=0 survives"
        );
        assert!(!pool.contains(&dom, 128, fp(128)), "oldest p=128 evicted");
    }

    // ── Domain isolation: different domains don't share state ───────────

    #[test]
    fn domain_isolation() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom_a = test_domain("isolation-a");
        let dom_b = test_domain("isolation-b");

        pool.insert(dom_a.clone(), 128, fp(128), HostBlob { bytes: 4096 });

        // dom_b has no checkpoint at p=128.
        assert!(!pool.contains(&dom_b, 128, fp(128)));
        assert!(pool.contains(&dom_a, 128, fp(128)));

        // Planning with dom_b should miss.
        let err = plan_resume(
            &mut pool,
            &dom_b,
            &toks(200),
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect_err("different domain should miss");

        assert_eq!(err, MissReason::NoCheckpoint);

        // Planning with dom_a should succeed.
        let plan = plan_resume(
            &mut pool,
            &dom_a,
            &toks(200),
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect("same domain should hit");

        assert_eq!(plan.boundary, 128);
    }

    // ── Domain isolation: eviction in one domain doesn't affect another ─

    #[test]
    fn domain_isolation_eviction() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(8192);
        let dom_a = test_domain("iso-evict-a");
        let dom_b = test_domain("iso-evict-b");

        pool.insert(dom_a.clone(), 128, fp(128), HostBlob { bytes: 4096 });
        pool.insert(dom_b.clone(), 128, fp(128), HostBlob { bytes: 4096 });

        // Inserting a third entry evicts the oldest unpinned (dom_a, 128).
        pool.insert(dom_a.clone(), 256, fp(256), HostBlob { bytes: 4096 });

        assert!(!pool.contains(&dom_a, 128, fp(128)), "dom_a p=128 evicted");
        assert!(
            pool.contains(&dom_b, 128, fp(128)),
            "dom_b p=128 must survive"
        );
    }

    // ── Page alignment: non-aligned boundary is rejected ────────────────

    #[test]
    fn non_aligned_boundary_rejected() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("align");

        let (id, _) = pool.insert(dom.clone(), 100, fp(100), HostBlob { bytes: 4096 });
        assert_eq!(
            id,
            CheckpointId::NONE,
            "non-page-aligned boundary must be rejected"
        );
        assert!(!pool.contains(&dom, 100, fp(100)));
        assert_eq!(pool.total_bytes(), 0);
    }

    // ── p=0 is a valid capture boundary ──────────────────────────────────

    #[test]
    fn p0_is_valid_boundary() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("p0");

        let (id, _) = pool.insert(dom.clone(), 0, fp(0), HostBlob { bytes: 0 });
        assert_ne!(id, CheckpointId::NONE);
        assert!(pool.contains(&dom, 0, fp(0)));
    }

    // ── CheckpointId is monotonic ────────────────────────────────────────

    #[test]
    fn checkpoint_ids_are_monotonic() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("monotonic");

        let (id1, _) = pool.insert(dom.clone(), 0, fp(0), HostBlob { bytes: 100 });
        let (id2, _) = pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 100 });
        let (id3, _) = pool.insert(dom.clone(), 256, fp(256), HostBlob { bytes: 100 });

        assert!(id1 < id2);
        assert!(id2 < id3);
        assert_eq!(id1, CheckpointId(1));
    }

    // ── Byte accounting is accurate ──────────────────────────────────────

    #[test]
    fn byte_accounting() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("bytes");

        pool.insert(dom.clone(), 0, fp(0), HostBlob { bytes: 1000 });
        assert_eq!(pool.total_bytes(), 1000);

        pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 2000 });
        assert_eq!(pool.total_bytes(), 3000);

        let _ = pool.evict(&dom, 0, fp(0));
        assert_eq!(pool.total_bytes(), 2000);
    }

    // ── DrafterDecision is passed through, not inferred ──────────────────

    #[test]
    fn drafter_decision_is_input() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("drafter");

        pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });

        // Default Ar.
        let plan_ar = plan_resume(
            &mut pool,
            &dom,
            &toks(200),
            &lookup(128),
            DrafterDecision::Ar,
        )
        .unwrap();
        assert_eq!(plan_ar.bundle.drafter, DrafterDecision::Ar);

        // Admission can choose Checkpoint.
        let plan_ckpt = plan_resume(
            &mut pool,
            &dom,
            &toks(200),
            &lookup(128),
            DrafterDecision::Checkpoint,
        )
        .unwrap();
        assert_eq!(plan_ckpt.bundle.drafter, DrafterDecision::Checkpoint);

        // Or Reseed.
        let plan_reseed = plan_resume(
            &mut pool,
            &dom,
            &toks(200),
            &lookup(128),
            DrafterDecision::Reseed,
        )
        .unwrap();
        assert_eq!(plan_reseed.bundle.drafter, DrafterDecision::Reseed);
    }

    // ── ResumePlan refuses incomplete bundle ─────────────────────────────

    #[test]
    fn resume_plan_refuses_incomplete_bundle() {
        let incomplete = ResumeBundle {
            attention_pages: false,
            dn_matrices_scales: true,
            conv_rings: true,
            ef_residual: true,
            drafter: DrafterDecision::Ar,
        };
        let err = ResumePlan::new(128, incomplete, 4096, LastTokenHandling::SuffixRecompute);
        assert!(err.is_err());
    }

    // ── Exact match with only p=0 checkpoint ─────────────────────────────

    #[test]
    fn exact_match_falls_back_to_p0() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("fallback-p0");

        // Only a checkpoint at p=0.
        pool.insert(dom.clone(), 0, fp(0), HostBlob { bytes: 0 });

        // Prompt of exactly 128 tokens, lookup says 128 resumable.
        // But no checkpoint at 128 → NoCheckpoint (p=0 checkpoint doesn't
        // help for a 128-token prompt with resumable=128).
        let err = plan_resume(
            &mut pool,
            &dom,
            &toks(128),
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect_err("no checkpoint at 128");

        assert_eq!(err, MissReason::NoCheckpoint);
    }

    // ── Resumable between page boundaries rounds down ────────────────────

    #[test]
    fn resumable_rounds_down_to_page_boundary() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("round");

        // Checkpoint at p=128 only.
        pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });

        // Lookup says 200 resumable (not page-aligned). Should find p=128.
        let plan = plan_resume(
            &mut pool,
            &dom,
            &toks(300),
            &lookup(200),
            DrafterDecision::Ar,
        )
        .unwrap();

        assert_eq!(plan.boundary, 128);
    }

    // ── R1: two prefixes at the same boundary must not clobber ──────────
    //
    // The pre-fix pool keyed on (domain, boundary) only: a second prompt
    // capturing at the same page boundary replaced the first prompt's
    // recurrent state while the first prompt's radix node still pointed at
    // that CheckpointId — a silent cross-prefix restore of the WRONG state.
    // With the fingerprint in the key the two prefixes get distinct entries
    // and a lookup for prefix A can never resolve prefix B's blob.

    /// A prompt that shares `toks(n)[..p]` for its first `p` tokens but
    /// diverges after — the "other" prefix at the same boundary.
    fn toks_variant(n: u64) -> Vec<u32> {
        (0..n as u32)
            .map(|i| {
                i.wrapping_mul(2654435761)
                    .wrapping_add(1)
                    .wrapping_add(0x9e3779b9)
            })
            .collect()
    }

    fn fp_variant(p: u64) -> u64 {
        prefix_fingerprint(&toks_variant(p.max(1))[..p as usize])
    }

    #[test]
    fn r1_divergent_prefixes_same_boundary_get_distinct_entries() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("r1");

        // Prefix A captures at p=128.
        let (id_a, _) = pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });
        // Prefix B (diverges before 128) captures at the SAME boundary.
        let (id_b, _) = pool.insert(dom.clone(), 128, fp_variant(128), HostBlob { bytes: 8192 });

        // Distinct entries, distinct ids — B did NOT replace A.
        assert_ne!(
            id_a, id_b,
            "divergent prefixes must not share a checkpoint id"
        );
        assert_eq!(pool.len(), 2, "both prefixes' checkpoints must coexist");
        // Each resolves to its own blob.
        assert_eq!(
            pool.peek(&dom, 128, fp(128)).map(|b| b.bytes_len()),
            Some(4096)
        );
        assert_eq!(
            pool.peek(&dom, 128, fp_variant(128)).map(|b| b.bytes_len()),
            Some(8192)
        );
    }

    #[test]
    fn r1_resume_for_prefix_a_never_returns_prefix_b_state() {
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("r1b");

        // A captures at 128; B (divergent) captures at 128 too.
        pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });
        pool.insert(dom.clone(), 128, fp_variant(128), HostBlob { bytes: 8192 });

        // A 200-token prompt on prefix A resumes at 128 with A's state.
        let plan_a = plan_resume(
            &mut pool,
            &dom,
            &toks(200),
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect("prefix A should resume");
        assert_eq!(plan_a.boundary, 128);
        assert_eq!(
            pool.peek(&dom, plan_a.boundary, fp(128))
                .map(|b| b.bytes_len()),
            Some(4096),
            "prefix A must resolve its own checkpoint, not B's"
        );

        // A prompt on prefix B resolves B's state, not A's.
        let plan_b = plan_resume(
            &mut pool,
            &dom,
            &toks_variant(200),
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect("prefix B should resume");
        assert_eq!(
            pool.peek(&dom, plan_b.boundary, fp_variant(128))
                .map(|b| b.bytes_len()),
            Some(8192),
            "prefix B must resolve its own checkpoint, not A's"
        );
    }

    #[test]
    fn r1_shared_prefix_reuses_same_checkpoint() {
        // Two prompts that genuinely share the first 128 tokens MUST share
        // the checkpoint — the fingerprint is over tokens[..p], so a shared
        // prefix yields the same key. This is the correct-sharing case.
        let mut pool = QwenCheckpointPool::<HostBlob>::new(1 << 20);
        let dom = test_domain("r1c");
        pool.insert(dom.clone(), 128, fp(128), HostBlob { bytes: 4096 });

        // A longer prompt on the SAME prefix (toks(300)[..128] == toks(128)).
        let plan = plan_resume(
            &mut pool,
            &dom,
            &toks(300),
            &lookup(128),
            DrafterDecision::Ar,
        )
        .expect("shared prefix should resume");
        assert_eq!(plan.boundary, 128);
    }

}
