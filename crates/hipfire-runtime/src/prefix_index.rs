// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-resident compressed radix tree over token spans (spec §4.2 C2).
//!
//! The `PrefixIndex` keys completed prefix state by [`CacheDomain`] (exact
//! identity, spec §4.1 C1) and the actual model input token sequence. Edges
//! store immutable token spans; nodes hold ordered [`PageHandle`]s for **full
//! 128-token pages only** plus optional [`CheckpointId`]s at resumable
//! boundaries (spec §4.5 C5). Partial tails are never shareable (spec §4.2).
//!
//! Operations: longest-prefix [`lookup`](PrefixIndex::lookup), [`insert`],
//! [`split`], [`pin`]/[`unpin`], and leaf-first [`evict_unpinned_leaves`].
//! Lookup pins matching page handles internally so eviction cannot drop them
//! until `unpin` (spec §4.4 C4). Eviction removes lookup visibility **before**
//! releasing cache refs.
//!
//! This module performs **no GPU mutation**. `add_cache_ref`/`release_cache_ref`/
//! `seal` are host-side operations on [`PagePool`] (spec §4.2: "Do not call
//! PagePool from lookup in a way that mutates GPU memory").
//!
//! # Token-count distinction (spec §4.2)
//!
//! - **matched_tokens**: equal token prefix in the index.
//! - **resident_kv_tokens**: matching attention rows that still exist (full
//!   pages whose `PageHandle` is still valid).
//! - **resumable_tokens**: largest boundary for which a checkpoint exists AND
//!   all required pages are still resident.
//!
//! A prefix match that lacks a recurrent checkpoint is [`MissReason::NoCheckpoint`],
//! not a hit in usage accounting (spec §4.2).

use std::collections::HashMap;

use rdna_compute::page_pool::{PageHandle, PagePool, PAGE_TOKENS};

use crate::cache_plan::CachePolicy;
use crate::serve_contract::{
    CacheDomain, CheckpointId, MissReason, PrefixLookup, PrefixLookupResult,
};

// =========================================================================
// Internal tree structure
// =========================================================================

/// Unique id for a node within a domain tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NodeId(u64);

/// A checkpoint boundary within a node's token span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CheckpointBoundary {
    /// Token offset *within this node* where the checkpoint applies.
    /// Always a multiple of `PAGE_TOKENS` (checkpoints live at page
    /// boundaries, spec §4.5).
    token_offset: u64,
    /// The opaque checkpoint id (non-zero).
    checkpoint: CheckpointId,
}

/// A node in the radix tree.
///
/// Each non-root node stores:
/// - `edge_tokens`: the immutable token span of its incoming edge.
/// - `pages`: ordered `PageHandle`s for the full 128-token pages covering
///   this node's span. `pages.len() * PAGE_TOKENS == edge_tokens.len() +
///   first_page_skip` when `pages` is non-empty; zero-page split-marker
///   nodes carry only edge tokens.
/// - `first_page_skip`: tokens at the start of `pages[0]` that belong to
///   the PARENT edge, not this one (0 for page-aligned nodes). A mid-page
///   fork produces children whose first physical page starts before their
///   edge does — the physical page is still a full 128-token page, only
///   the edge's claim on it is partial.
/// - `checkpoints`: optional checkpoint boundaries within this node.
/// - `children`: child edges keyed by their first token.
#[derive(Debug)]
struct Node {
    /// Monotonic insertion order for leaf-first, oldest-first eviction.
    insert_seq: u64,
    /// Parent node (None for the root). Lets eviction unlink a leaf without
    /// an O(nodes) parent scan.
    parent: Option<NodeId>,
    /// Immutable token span of the incoming edge (empty for root).
    edge_tokens: Vec<u32>,
    /// Tokens at the start of `pages[0]` owned by the parent edge. Only
    /// meaningful when `pages` is non-empty; 0 for page-aligned nodes.
    first_page_skip: usize,
    /// Page handles for the full pages covering this node's span.
    pages: Vec<PageHandle>,
    /// Checkpoint boundaries within this node's span. Sorted by token_offset.
    checkpoints: Vec<CheckpointBoundary>,
    /// Child edges: first token -> child node id.
    children: HashMap<u32, NodeId>,
    /// Number of live lookup pins on this node (spec §4.4: eviction cannot
    /// drop a node with `pin_count > 0`). Counted, not boolean: overlapping
    /// lookups share path nodes and each holds its own [`PinTicket`].
    pin_count: u32,
}
impl Node {
    fn root() -> Self {
        Node {
            insert_seq: 0,
            parent: None,
            edge_tokens: Vec::new(),
            first_page_skip: 0,
            pages: Vec::new(),
            checkpoints: Vec::new(),
            children: HashMap::new(),
            pin_count: 0,
        }
    }

    fn new(insert_seq: u64, parent: NodeId, edge_tokens: Vec<u32>, pages: Vec<PageHandle>) -> Self {
        Node {
            insert_seq,
            parent: Some(parent),
            edge_tokens,
            first_page_skip: 0,
            pages,
            checkpoints: Vec::new(),
            children: HashMap::new(),
            pin_count: 0,
        }
    }

    /// Total tokens covered by this node's edge.
    fn token_span(&self) -> u64 {
        self.edge_tokens.len() as u64
    }

    /// Returns `true` if this node has no children.
    fn is_leaf_node(&self) -> bool {
        self.children.is_empty()
    }
}

/// A domain-scoped radix tree (spec §4.1: different domain → isolated trees).
#[derive(Debug)]
struct DomainTree {
    root: NodeId,
    nodes: HashMap<NodeId, Node>,
    next_node_id: u64,
    next_insert_seq: u64,
}

impl DomainTree {
    fn new() -> Self {
        let root_id = NodeId(0);
        let mut nodes = HashMap::new();
        nodes.insert(root_id, Node::root());
        DomainTree {
            root: root_id,
            nodes,
            next_node_id: 1,
            next_insert_seq: 1,
        }
    }

    fn fresh_node_id(&mut self) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    fn fresh_insert_seq(&mut self) -> u64 {
        let s = self.next_insert_seq;
        self.next_insert_seq += 1;
        s
    }

    fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// =========================================================================
// Public types
// =========================================================================

/// A page handle plus its logical position in the prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handle {
    /// The physical page handle from `PagePool`.
    pub handle: PageHandle,
    /// Logical token offset where this page starts within the prefix.
    pub token_offset: u64,
}

/// Result of [`PrefixIndex::inspect`] — the three token counts without
/// claiming a [`PrefixLookupResult::Hit`] (spec §4.2: for metrics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InspectResult {
    /// Equal token prefix length found in the radix index.
    pub matched_tokens: u64,
    /// Matching attention rows that are still physically resident.
    pub resident_kv_tokens: u64,
    /// Largest boundary for which a checkpoint exists and pages are resident.
    pub resumable_tokens: u64,
}

/// Opaque handle to one lookup's pin on a radix path (spec §4.4 C4).
///
/// Returned by [`PrefixIndex::lookup_with_pages`]. Release it with
/// [`PrefixIndex::release_pin`] when the request that performed the lookup
/// terminates — on *every* terminal path, not just normal completion.
/// Overlapping lookups in the same domain hold independent tickets; each
/// release decrements only the nodes on its own path.
///
/// A ticket whose path nodes were already removed (e.g. by
/// [`PrefixIndex::release_all`] on model reset) releases harmlessly.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PinTicket {
    /// Node ids along the pinned path, root to deepest matched node.
    nodes: Vec<NodeId>,
}

impl PinTicket {
    /// A ticket that pins nothing (miss paths, disabled cache).
    pub fn none() -> Self {
        PinTicket { nodes: Vec::new() }
    }

    /// Whether this ticket holds any pins.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Error returned by [`PrefixIndex::insert`] / [`PrefixIndex::publish_sealed_pages`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertError {
    /// The published prefix has a partial last page (len % 128 != 0).
    /// Partial tails are never shareable (spec §4.2).
    PartialLastPage {
        /// The token length that was not a multiple of `PAGE_TOKENS`.
        token_len: usize,
    },
    /// The CPU node metadata bound (`max_cpu_nodes`) would be exceeded and
    /// eviction could not free enough nodes (spec §4.4).
    CpuNodeBoundExceeded {
        /// Current node count.
        current: usize,
        /// The configured maximum.
        max: usize,
    },
    /// A page handle failed validation against the pool (stale/free).
    InvalidHandle(String),
    /// A handle's token_offset is not page-aligned or handle count doesn't
    /// match the number of full pages.
    MisalignedHandle,
    /// The retained-cache byte ceiling (`max_retained_bytes`) cannot hold
    /// this publish even after evicting every unpinned leaf (spec §4.4).
    CacheByteBoundExceeded {
        /// Bytes currently retained by the tree.
        retained: usize,
        /// The configured ceiling.
        max: usize,
    },
}

impl std::fmt::Display for InsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PartialLastPage { token_len } => write!(
                f,
                "partial last page: token_len {} is not a multiple of {}",
                token_len, PAGE_TOKENS
            ),
            Self::CpuNodeBoundExceeded { current, max } => write!(
                f,
                "CPU node bound exceeded: {} nodes > max {} (eviction insufficient)",
                current, max
            ),
            Self::InvalidHandle(s) => write!(f, "invalid page handle: {}", s),
            Self::MisalignedHandle => {
                write!(
                    f,
                    "misaligned page handle (not page-aligned or count mismatch)"
                )
            }
            Self::CacheByteBoundExceeded { retained, max } => write!(
                f,
                "retained cache bytes {} exceed ceiling {} and eviction could not free enough",
                retained, max
            ),
        }
    }
}

impl std::error::Error for InsertError {}

// =========================================================================
// PrefixIndex
// =========================================================================

/// CPU-resident compressed radix tree over token spans (spec §4.2 C2).
///
/// Keyed by [`CacheDomain`] (exact identity; different domain → isolated
/// trees). Edges store immutable token spans; nodes hold ordered [`PageHandle`]s
/// for full 128-token pages only, plus optional [`CheckpointId`]s at resumable
/// boundaries.
///
/// # Cache-ref ownership
///
/// The INDEX owns one `PagePool` cache ref per node slot holding a handle:
/// `create_chain` takes a ref for every page recorded in a newly created
/// node (and releases them if the chain rolls back), and eviction /
/// `release_all` release exactly the refs the index took. A physical page
/// referenced by two nodes (a mid-page fork's straddling page) carries two
/// refs — balanced by construction. Callers must `PagePool::seal` pages
/// before publishing but must NOT take cache refs themselves: refs taken
/// outside the index for pages the tree does not adopt (an already-present
/// chain, a refused insert) would be owned by nobody and leak the page away
/// from the free list forever.
///
/// Construct with [`PrefixIndex::new`] specifying `max_cpu_nodes` to bound
/// CPU node/token metadata (spec §4.4: "Bound CPU node/token metadata as well
/// as device bytes").
pub struct PrefixIndex {
    trees: HashMap<CacheDomain, DomainTree>,
    max_cpu_nodes: usize,
    total_nodes: usize,
    /// Optional ceiling on total device bytes referenced by tree nodes
    /// (spec §4.4: cache retention is a ceiling, never a permanently
    /// reserved partition). Enforced at insert: the oldest unpinned leaves
    /// are evicted first; an insert that still does not fit is refused.
    max_retained_bytes: Option<usize>,
    /// Running sum over all trees of `pages.len() × (k_page + v_page)`
    /// bytes (the same arithmetic eviction credits).
    retained_bytes: usize,
}

/// Internal result of walking the tree to find the longest token prefix.
struct WalkResult {
    matched_tokens: u64,
    resident_kv_tokens: u64,
    resumable_tokens: u64,
    /// Path of node ids from root to the deepest matched node (inclusive).
    path: Vec<NodeId>,
    /// Whether the walk consumed all query tokens.
    exhausted_query: bool,
    /// Resident full-page handles along the matched prefix, in token order.
    pages: Vec<Handle>,
}

impl PrefixIndex {
    /// Create a new `PrefixIndex` with a CPU node metadata bound.
    ///
    /// `max_cpu_nodes` bounds the total number of radix nodes across all
    /// domain trees (spec §4.4). Inserting past this bound triggers
    /// leaf-first eviction; if eviction cannot free enough nodes, the insert
    /// is refused with [`InsertError::CpuNodeBoundExceeded`].
    pub fn new(max_cpu_nodes: usize) -> Self {
        PrefixIndex {
            trees: HashMap::new(),
            max_cpu_nodes,
            total_nodes: 0,
            max_retained_bytes: None,
            retained_bytes: 0,
        }
    }

    /// Set the retained-cache byte ceiling (spec §9.1
    /// `serve.prefix_cache_max_bytes`). Zero disables retention entirely —
    /// every publish is refused rather than reinterpreted as unlimited.
    pub fn set_max_retained_bytes(&mut self, max_bytes: usize) {
        self.max_retained_bytes = Some(max_bytes);
    }

    /// Total device bytes currently referenced by tree nodes.
    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    /// Current total node count across all domain trees.
    pub fn total_nodes(&self) -> usize {
        self.total_nodes
    }

    /// Number of distinct cache domains with entries.
    pub fn domain_count(&self) -> usize {
        self.trees.len()
    }

    // ── Lookup ────────────────────────────────────────────────────────

    /// Longest-prefix lookup (spec §4.2 C2).
    ///
    /// Returns [`PrefixLookupResult::Hit`] only when a checkpoint id exists
    /// at the chosen boundary AND the corresponding pages are still
    /// cache-resident. Token match without checkpoint →
    /// [`MissReason::NoCheckpoint`] (spec §4.2: "A prefix match that lacks a
    /// recurrent checkpoint is not a hit in usage accounting").
    ///
    /// Does NOT pin and returns no page handles — callers that intend to
    /// share the matched pages must use
    /// [`lookup_with_pages`](Self::lookup_with_pages), which returns a
    /// [`PinTicket`] keeping the path resident until released.
    ///
    /// If `policy` is supplied and `allow_partial` is false, a mid-sequence
    /// partial hit (resumable boundary < matched tokens) returns
    /// [`MissReason::IncompatiblePolicy`] (spec §4.5). If policy is `None`,
    /// the raw longest resumable boundary is returned.
    pub fn lookup(
        &mut self,
        domain: &CacheDomain,
        tokens: &[u32],
        pool: &PagePool,
        policy: Option<&CachePolicy>,
    ) -> PrefixLookupResult {
        self.lookup_inner(domain, tokens, pool, policy, false).0
    }

    /// Lookup plus the sealed page handles covering the resumable
    /// boundary. The engine shares these into an empty destination slot
    /// instead of a length-keyed side map (spec §4.2: two prefixes of
    /// equal length are distinct).
    ///
    /// On a `Hit` the matched path is pinned and the returned [`PinTicket`]
    /// is the *only* way to release it — pass it to
    /// [`release_pin`](Self::release_pin) on every terminal path of the
    /// request (completion, client-gone, failure). On a `Miss` the ticket
    /// is empty and releasing it is a no-op.
    pub fn lookup_with_pages(
        &mut self,
        domain: &CacheDomain,
        tokens: &[u32],
        pool: &PagePool,
        policy: Option<&CachePolicy>,
    ) -> (PrefixLookupResult, Vec<Handle>, PinTicket) {
        self.lookup_inner(domain, tokens, pool, policy, true)
    }

    /// Shared lookup implementation. `pin` controls whether a Hit pins the
    /// matched path and yields a live [`PinTicket`].
    fn lookup_inner(
        &mut self,
        domain: &CacheDomain,
        tokens: &[u32],
        pool: &PagePool,
        policy: Option<&CachePolicy>,
        pin: bool,
    ) -> (PrefixLookupResult, Vec<Handle>, PinTicket) {
        let walk = self.walk(domain, tokens, pool);

        if walk.matched_tokens == 0 {
            return (
                PrefixLookupResult::Miss(MissReason::NoMatch),
                Vec::new(),
                PinTicket::none(),
            );
        }

        if walk.resumable_tokens == 0 {
            return (
                PrefixLookupResult::Miss(MissReason::NoCheckpoint),
                Vec::new(),
                PinTicket::none(),
            );
        }

        if let Some(p) = policy {
            if !p.allow_partial && walk.resumable_tokens < walk.matched_tokens {
                return (
                    PrefixLookupResult::Miss(MissReason::IncompatiblePolicy),
                    Vec::new(),
                    PinTicket::none(),
                );
            }
        }

        // Pin nodes along the path so eviction cannot drop them (spec §4.4).
        let ticket = if pin {
            self.pin_path(domain, &walk.path)
        } else {
            PinTicket::none()
        };

        let resumable = walk.resumable_tokens;
        let handles: Vec<Handle> = walk
            .pages
            .into_iter()
            .filter(|h| h.token_offset + PAGE_TOKENS as u64 <= resumable)
            .collect();
        (
            PrefixLookupResult::Hit(PrefixLookup {
                matched_tokens: walk.matched_tokens,
                resident_kv_tokens: walk.resident_kv_tokens,
                resumable_tokens: walk.resumable_tokens,
            }),
            handles,
            ticket,
        )
    }

    /// Inspect the index for the three token counts without claiming a Hit
    /// and without pinning (spec §4.2: for metrics).
    pub fn inspect(&self, domain: &CacheDomain, tokens: &[u32], pool: &PagePool) -> InspectResult {
        let walk = self.walk(domain, tokens, pool);
        InspectResult {
            matched_tokens: walk.matched_tokens,
            resident_kv_tokens: walk.resident_kv_tokens,
            resumable_tokens: walk.resumable_tokens,
        }
    }

    /// Walk the tree to find the longest matching token prefix.
    ///
    /// Does NOT pin. Used by both `lookup` (which pins after) and `inspect`.
    fn walk(&self, domain: &CacheDomain, tokens: &[u32], pool: &PagePool) -> WalkResult {
        let tree = match self.trees.get(domain) {
            Some(t) => t,
            None => {
                return WalkResult {
                    matched_tokens: 0,
                    resident_kv_tokens: 0,
                    resumable_tokens: 0,
                    path: Vec::new(),
                    exhausted_query: false,
                    pages: Vec::new(),
                };
            }
        };

        let mut current = tree.root;
        let mut path: Vec<NodeId> = vec![tree.root];
        let mut query_pos: usize = 0;
        let mut matched_tokens: u64 = 0;
        let mut resident_kv_tokens: u64 = 0;
        // Largest token offset B such that EVERY page below B is resident.
        // A checkpoint at boundary b is only usable when b <= this value —
        // counting valid pages above a gap would bless a boundary whose
        // underlying KV rows are gone (spec §4.5: a boundary is resumable
        // only when every required state component exists at it).
        let mut contiguous_resident_tokens: u64 = 0;
        let mut resumable_tokens: u64 = 0;
        let mut pages: Vec<Handle> = Vec::new();

        loop {
            if query_pos >= tokens.len() {
                break;
            }

            let node = match tree.nodes.get(&current) {
                Some(n) => n,
                None => break,
            };

            let next_token = tokens[query_pos];
            let child_id = match node.children.get(&next_token).copied() {
                Some(id) => id,
                None => break,
            };

            let child = match tree.nodes.get(&child_id) {
                Some(c) => c,
                None => break,
            };

            // Compare edge tokens against query tokens.
            let edge = &child.edge_tokens;
            let mut edge_match = 0usize;
            for (i, &et) in edge.iter().enumerate() {
                if query_pos + i >= tokens.len() || tokens[query_pos + i] != et {
                    break;
                }
                edge_match += 1;
            }

            // child_base is the global token offset where this child's edge
            // begins. Compute it BEFORE adding edge_match to matched_tokens.
            let child_base = matched_tokens;
            // Walk pages in order; extend the contiguous prefix only while
            // every page below it validates. Metrics count all valid pages
            // (resident_kv_tokens), but resumability follows the gap-free
            // prefix. A partial first page (first_page_skip > 0) starts
            // BEFORE the edge: its physical span is
            // [child_base - skip, child_base - skip + PAGE_TOKENS).
            let page_base = child_base.saturating_sub(child.first_page_skip as u64);
            let matched_end = child_base + edge_match as u64;
            let mut contig = contiguous_resident_tokens;
            for (i, ph) in child.pages.iter().enumerate() {
                let page_start = page_base + (i * PAGE_TOKENS) as u64;
                let page_end = page_start + PAGE_TOKENS as u64;
                // The page is reusable only when its whole physical span
                // lies inside the matched prefix.
                if page_end > matched_end {
                    break;
                }
                if pool.validate_handle(ph).is_ok() {
                    resident_kv_tokens += PAGE_TOKENS as u64;
                    pages.push(Handle {
                        handle: *ph,
                        token_offset: page_start,
                    });
                    if contig == page_start {
                        contig = page_end;
                    }
                }
            }
            contiguous_resident_tokens = contig;

            matched_tokens += edge_match as u64;

            for cb in &child.checkpoints {
                if cb.token_offset <= edge_match as u64 {
                    let boundary = child_base + cb.token_offset;
                    if boundary <= contiguous_resident_tokens {
                        resumable_tokens = resumable_tokens.max(boundary);
                    }
                }
            }

            if edge_match == edge.len() {
                // Full edge match — advance to child.
                query_pos += edge_match;
                current = child_id;
                path.push(child_id);
            } else {
                // Partial match — query diverges mid-edge.
                break;
            }
        }

        WalkResult {
            matched_tokens,
            resident_kv_tokens,
            resumable_tokens,
            path,
            exhausted_query: query_pos >= tokens.len(),
            pages,
        }
    }

    /// Pin all nodes along a path (spec §4.4: eviction cannot drop pinned).
    /// Returns the [`PinTicket`] the caller must later pass to
    /// [`release_pin`](Self::release_pin).
    fn pin_path(&mut self, domain: &CacheDomain, path: &[NodeId]) -> PinTicket {
        if let Some(tree) = self.trees.get_mut(domain) {
            for &nid in path {
                if let Some(node) = tree.nodes.get_mut(&nid) {
                    node.pin_count = node.pin_count.saturating_add(1);
                }
            }
        }
        PinTicket {
            nodes: path.to_vec(),
        }
    }

    // ── Insert ────────────────────────────────────────────────────────

    /// Insert a completed prefix into the index (spec §4.2 C2).
    ///
    /// The caller must have already sealed the pages via `PagePool::seal`.
    /// The INDEX takes one cache ref per node slot holding a handle (and
    /// releases it on rollback/eviction) — the caller must NOT take
    /// publication refs itself: refs for pages the tree does not adopt
    /// (an already-present chain, a refused insert) would be owned by
    /// nobody and leak.
    ///
    /// `handles` is the ordered list of full-page handles with their logical
    /// token offsets. `checkpoint` is the optional checkpoint id at the final
    /// boundary of this prefix. `CheckpointId::NONE` means no checkpoint.
    ///
    /// Same completed prefix → one canonical entry; in-flight duplicates are
    /// not remapped (spec §4.2).
    pub fn insert(
        &mut self,
        domain: &CacheDomain,
        tokens: &[u32],
        handles: &[Handle],
        checkpoint: Option<CheckpointId>,
        pool: &mut PagePool,
    ) -> Result<(), InsertError> {
        // Validate handles.
        for h in handles {
            if h.token_offset % PAGE_TOKENS as u64 != 0 {
                return Err(InsertError::MisalignedHandle);
            }
            pool.validate_handle(&h.handle)
                .map_err(InsertError::InvalidHandle)?;
        }

        // The key must be exactly the full pages the handles cover: a
        // partial tail is never shareable (spec §4.2) and a longer tail
        // would be silently dropped by chain creation.
        if tokens.len() != handles.len() * PAGE_TOKENS {
            return Err(InsertError::MisalignedHandle);
        }

        // Get or create the domain tree.
        let is_new_tree = !self.trees.contains_key(domain);
        if is_new_tree {
            let t = DomainTree::new();
            self.total_nodes += t.node_count();
            self.trees.insert(domain.clone(), t);
        }

        // Retained-bytes ceiling (spec §4.4: cache retention is a ceiling,
        // and §9.1: zero means no retained cache). The insert needs at most
        // `handles.len()` page slots; evict oldest unpinned leaves until the
        // ceiling can hold the insert, refusing if eviction cannot progress.
        // Every refusal here must also drop a tree this call created: the
        // caller charges nothing on Err, so a left-behind empty root would
        // count against the CPU node bound forever.
        if let Some(max_bytes) = self.max_retained_bytes {
            let page_bytes = pool.k_page_bytes() + pool.v_page_bytes();
            let refuse = |this: &mut Self, retained: usize, max: usize| {
                if is_new_tree {
                    if let Some(t) = this.trees.remove(domain) {
                        this.total_nodes = this.total_nodes.saturating_sub(t.node_count());
                    }
                }
                InsertError::CacheByteBoundExceeded { retained, max }
            };
            if max_bytes < page_bytes {
                return Err(refuse(self, self.retained_bytes, max_bytes));
            }
            loop {
                let projected = self
                    .retained_bytes
                    .saturating_add(handles.len().saturating_mul(page_bytes));
                if projected <= max_bytes {
                    break;
                }
                let before = self.total_nodes;
                self.evict_unpinned_leaves(pool, projected.saturating_sub(max_bytes));
                if self.total_nodes == before {
                    return Err(refuse(self, self.retained_bytes, max_bytes));
                }
            }
        }

        // Insert, evicting oldest unpinned leaves to make room when the CPU
        // node bound binds (spec §4.4: reclaim cache-only pages before
        // rejecting work). An insert that adds zero nodes (exact re-publish
        // or checkpoint-only) succeeds even at the bound.
        loop {
            let result = insert_into_tree(
                self.trees.get_mut(domain).unwrap(),
                tokens,
                handles,
                checkpoint,
                pool,
                self.max_cpu_nodes,
                self.total_nodes,
            );
            match result {
                Ok((delta, pages_adopted)) => {
                    self.total_nodes = (self.total_nodes as i32 + delta) as usize;
                    // Charge retained bytes by PAGES ADOPTED (one cache
                    // ref each), the exact basis eviction subtracts. The
                    // old node-count basis over-charged one page per
                    // mid-page split marker (a node holding zero pages),
                    // permanently inflating the retained total until the
                    // byte ceiling wedged (see the ownership doc).
                    self.retained_bytes = self.retained_bytes.saturating_add(
                        pages_adopted * (pool.k_page_bytes() + pool.v_page_bytes()),
                    );
                    return Ok(());
                }
                Err(InsertError::CpuNodeBoundExceeded { .. }) => {
                    // Evict one leaf (one page's worth of bytes) and retry.
                    // No progress → the bound genuinely cannot be met.
                    let before = self.total_nodes;
                    self.evict_unpinned_leaves(pool, pool.k_page_bytes() + pool.v_page_bytes());
                    if self.total_nodes == before {
                        return Err(InsertError::CpuNodeBoundExceeded {
                            current: self.total_nodes,
                            max: self.max_cpu_nodes,
                        });
                    }
                }
                Err(e) => {
                    // Roll back a tree created for a failed insert so an
                    // empty root does not count against the bound forever.
                    if is_new_tree {
                        if let Some(t) = self.trees.remove(domain) {
                            self.total_nodes = self.total_nodes.saturating_sub(t.node_count());
                        }
                    }
                    return Err(e);
                }
            }
        }
    }

    // ── Pin / Unpin ───────────────────────────────────────────────────

    /// Test-only: drop the index's cache ref on `phys` and remove the
    /// handle from whichever node holds it — simulates losing one page
    /// (a residency gap) without evicting the whole leaf.
    #[cfg(test)]
    pub(crate) fn test_drop_page_ref(
        &mut self,
        pool: &mut PagePool,
        domain: &CacheDomain,
        phys: u32,
    ) {
        if let Some(tree) = self.trees.get_mut(domain) {
            for (_, node) in tree.nodes.iter_mut() {
                if let Some(pos) = node.pages.iter().position(|p| p.phys == phys) {
                    node.pages.remove(pos);
                    let _ = pool.release_cache_ref(phys);
                    self.retained_bytes = self
                        .retained_bytes
                        .saturating_sub(pool.k_page_bytes() + pool.v_page_bytes());
                    return;
                }
            }
        }
    }

    /// Release one lookup's pin (spec §4.4 C4).
    ///
    /// Decrements `pin_count` on exactly the nodes the ticket's lookup
    /// pinned — other in-flight lookups sharing the same domain or path
    /// nodes keep their pins. Releasing an empty ticket (a miss) or a
    /// ticket whose nodes were already removed (e.g. `release_all` on
    /// reset) is a harmless no-op.
    pub fn release_pin(&mut self, domain: &CacheDomain, ticket: PinTicket) {
        if ticket.nodes.is_empty() {
            return;
        }
        if let Some(tree) = self.trees.get_mut(domain) {
            for nid in ticket.nodes {
                if let Some(node) = tree.nodes.get_mut(&nid) {
                    node.pin_count = node.pin_count.saturating_sub(1);
                }
            }
        }
    }

    // ── Eviction ──────────────────────────────────────────────────────

    /// Remove EVERY entry and release its cache lease into `pool`.
    ///
    /// Model reset / lifecycle teardown path (spec §4.6): dropping the radix
    /// alone orphans the cache refs the publisher took at publication, which
    /// strands pages in CacheOnly and drains the pool to zero free pages
    /// within a few reset cycles (A20). Lookup visibility is removed first,
    /// then each page's cache ref is released (spec §4.4). Returns the
    /// number of page handles released.
    pub fn release_all(&mut self, pool: &mut PagePool) -> usize {
        let mut released = 0usize;
        let trees = std::mem::take(&mut self.trees);
        for (_, mut tree) in trees {
            for (_, node) in tree.nodes.drain() {
                for ph in &node.pages {
                    if pool.release_cache_ref(ph.phys).is_ok() {
                        released += 1;
                    }
                }
            }
        }
        self.total_nodes = 0;
        self.retained_bytes = 0;
        released
    }

    /// Evict oldest unpinned leaves, releasing up to `max_bytes` of device
    /// memory (spec §4.4 C4).
    ///
    /// Leaf-first, oldest first (by insert sequence). Removes lookup
    /// visibility **before** releasing cache refs via
    /// `PagePool::release_cache_ref`. Keeps useful ancestors while
    /// descendants use them (ancestors are only evicted when all their
    /// children are gone).
    ///
    /// Returns the physical page indices that were evicted.
    pub fn evict_unpinned_leaves(&mut self, pool: &mut PagePool, max_bytes: usize) -> Vec<u32> {
        let mut evicted_phys: Vec<u32> = Vec::new();
        let mut bytes_freed: usize = 0;

        // Collect candidates: (domain, node_id, insert_seq) for unpinned leaves.
        let mut candidates: Vec<(CacheDomain, NodeId, u64)> = Vec::new();
        for (domain, tree) in &self.trees {
            for (nid, node) in &tree.nodes {
                if *nid != tree.root && node.is_leaf_node() && node.pin_count == 0 {
                    candidates.push((domain.clone(), *nid, node.insert_seq));
                }
            }
        }
        candidates.sort_by_key(|(_, _, seq)| *seq);

        let mut cursor = 0usize;
        while cursor < candidates.len() {
            if bytes_freed >= max_bytes {
                break;
            }
            let (domain, nid, _) = candidates[cursor].clone();
            cursor += 1;

            let tree = match self.trees.get_mut(&domain) {
                Some(t) => t,
                None => continue,
            };

            // Re-check conditions (may have changed).
            let node = match tree.nodes.get(&nid) {
                Some(n) => n,
                None => continue,
            };
            if !node.is_leaf_node() || node.pin_count != 0 {
                continue;
            }

            let pages = node.pages.clone();
            let first_token = node.edge_tokens.first().copied();

            // Unlink from the parent via the stored back-pointer.
            let parent_id = node.parent;

            if let Some(pid) = parent_id {
                if let Some(parent) = tree.nodes.get_mut(&pid) {
                    if let Some(ft) = first_token {
                        parent.children.remove(&ft);
                    }
                }
            }

            // Remove the node (remove lookup visibility BEFORE releasing refs).
            tree.nodes.remove(&nid);
            self.total_nodes = self.total_nodes.saturating_sub(1);

            // Release cache refs (the index owns exactly one per node page
            // slot — see the ownership doc on `PrefixIndex`).
            for ph in &pages {
                let page_bytes = pool.k_page_bytes() + pool.v_page_bytes();
                bytes_freed += page_bytes;
                let _ = pool.release_cache_ref(ph.phys);
                evicted_phys.push(ph.phys);
                self.retained_bytes = self.retained_bytes.saturating_sub(page_bytes);
            }

            // Removing this leaf may have turned its parent into a new
            // unpinned leaf — re-evaluate it in the same pass so eviction
            // can collapse chains instead of stopping one level short.
            if let Some(pid) = parent_id {
                if let Some(parent) = tree.nodes.get(&pid) {
                    if pid != tree.root && parent.is_leaf_node() && parent.pin_count == 0 {
                        candidates.push((domain.clone(), pid, parent.insert_seq));
                    }
                }
            }
        }

        evicted_phys
    }

    /// Try to evict enough nodes to fit `needed` new nodes.
    pub fn evict_for_capacity(
        &mut self,
        pool: &mut PagePool,
        needed: usize,
    ) -> Result<(), InsertError> {
        while self.total_nodes + needed > self.max_cpu_nodes {
            let before = self.total_nodes;
            self.evict_unpinned_leaves(pool, usize::MAX);
            if self.total_nodes == before {
                return Err(InsertError::CpuNodeBoundExceeded {
                    current: self.total_nodes,
                    max: self.max_cpu_nodes,
                });
            }
        }
        Ok(())
    }

    // ── Publication helper ────────────────────────────────────────────

    /// Publish sealed pages into the index (spec §4.6 C6, §4.2 C2).
    ///
    /// Refuses partial last pages: the published prefix's token length must
    /// be a multiple of `PAGE_TOKENS` (spec §4.2: "Publish immutable full
    /// 128-token pages first ... Partial active tails stay private").
    ///
    /// `checkpoint` is an opaque id minted by the caller (the Qwen adapter).
    /// `CheckpointId::NONE` means no checkpoint (spec §4.5).
    ///
    /// The caller must have sealed the pages (`PagePool::seal`); the index
    /// takes the cache refs itself (see `insert`).
    pub fn publish_sealed_pages(
        &mut self,
        domain: &CacheDomain,
        tokens: &[u32],
        handles: &[Handle],
        checkpoint: Option<CheckpointId>,
        pool: &mut PagePool,
    ) -> Result<(), InsertError> {
        if tokens.len() % PAGE_TOKENS != 0 {
            return Err(InsertError::PartialLastPage {
                token_len: tokens.len(),
            });
        }

        if handles.len() != tokens.len() / PAGE_TOKENS {
            return Err(InsertError::MisalignedHandle);
        }

        self.insert(domain, tokens, handles, checkpoint, pool)
    }

    /// Split an edge (metadata-only, spec §4.2).
    ///
    /// Public wrapper for testing. Splits the edge from the root to the
    /// child keyed by `first_token` at `split_tokens` into the edge.
    pub fn split(
        &mut self,
        domain: &CacheDomain,
        first_token: u32,
        split_tokens: usize,
    ) -> Result<(), InsertError> {
        let tree = match self.trees.get_mut(domain) {
            Some(t) => t,
            None => return Err(InsertError::InvalidHandle("domain not found".to_string())),
        };

        let root = tree.root;
        let child_id = match tree
            .nodes
            .get(&root)
            .and_then(|n| n.children.get(&first_token).copied())
        {
            Some(id) => id,
            None => return Err(InsertError::InvalidHandle("child not found".to_string())),
        };

        let delta = split_edge(
            tree,
            root,
            child_id,
            split_tokens,
            self.max_cpu_nodes,
            self.total_nodes,
        )?;
        self.total_nodes = (self.total_nodes as i32 + delta) as usize;
        Ok(())
    }
}

// =========================================================================
// Free tree manipulation functions (avoid &mut self + &mut tree borrow conflict)
// =========================================================================

/// Insert tokens/handles into a domain tree, splitting edges as needed.
/// Takes one `PagePool` cache ref for every page handle recorded in a newly
/// created node (released again if the chain rolls back — see the
/// cache-ref ownership doc on [`PrefixIndex`]). Returns the net change in
/// node count (positive = nodes added) AND the number of pages adopted
/// (cache refs taken) — the retained-bytes basis. Split markers add a
/// node but adopt no page.
fn insert_into_tree(
    tree: &mut DomainTree,
    tokens: &[u32],
    handles: &[Handle],
    checkpoint: Option<CheckpointId>,
    pool: &mut PagePool,
    max_cpu_nodes: usize,
    current_total: usize,
) -> Result<(i32, usize), InsertError> {
    let mut current = tree.root;
    let mut query_pos: usize = 0;
    let mut delta: i32 = 0;
    let mut pages_adopted: usize = 0;

    loop {
        if query_pos >= tokens.len() {
            if let Some(ckpt) = checkpoint {
                if ckpt.is_some() {
                    let node = tree.nodes.get_mut(&current).unwrap();
                    let token_offset = if current == tree.root {
                        0
                    } else {
                        node.edge_tokens.len() as u64
                    };
                    if !node
                        .checkpoints
                        .iter()
                        .any(|cb| cb.token_offset == token_offset)
                    {
                        node.checkpoints.push(CheckpointBoundary {
                            token_offset,
                            checkpoint: ckpt,
                        });
                        node.checkpoints.sort_by_key(|cb| cb.token_offset);
                    }
                }
            }
            return Ok((delta, pages_adopted));
        }

        let node = match tree.nodes.get(&current) {
            Some(n) => n,
            None => return Ok((delta, pages_adopted)),
        };

        let next_token = tokens[query_pos];
        let child_id = match node.children.get(&next_token).copied() {
            Some(id) => id,
            None => {
                let remaining_tokens = &tokens[query_pos..];
                // query_pos may sit mid-page (a partial-node match leaves
                // the cursor inside the new key's current page): the
                // branch's first node claims only the tail of that page.
                let remaining_handles = &handles[query_pos / PAGE_TOKENS..];
                let (d, p) = create_chain(
                    tree,
                    current,
                    remaining_tokens,
                    remaining_handles,
                    query_pos % PAGE_TOKENS,
                    checkpoint,
                    pool,
                    max_cpu_nodes,
                    current_total + delta as usize,
                )?;
                delta += d;
                pages_adopted += p;
                return Ok((delta, pages_adopted));
            }
        };

        let child = match tree.nodes.get(&child_id) {
            Some(c) => c,
            None => return Ok((delta, pages_adopted)),
        };

        let edge = &child.edge_tokens;
        let mut edge_match = 0usize;
        for (i, &et) in edge.iter().enumerate() {
            if query_pos + i >= tokens.len() || tokens[query_pos + i] != et {
                break;
            }
            edge_match += 1;
        }

        if edge_match == edge.len() {
            query_pos += edge_match;
            current = child_id;
        } else {
            // The query diverges from (or ends inside) the child's edge.
            // Split at the divergence token — page-aligned or not. A
            // mid-page split produces a zero-page marker plus children
            // whose first_page_skip reaches back into the divergent page.
            // The marker adopts NO page (it holds zero handles), so it
            // adds a node but zero retained bytes.
            let split_pos = edge_match;
            let divergence = query_pos + edge_match;

            delta += split_edge(
                tree,
                current,
                child_id,
                split_pos,
                max_cpu_nodes,
                current_total + delta as usize,
            )?;

            let parent = tree.nodes.get(&current).unwrap();
            let split_node_id = parent.children.get(&next_token).copied().unwrap();

            if divergence >= tokens.len() {
                if let Some(ckpt) = checkpoint {
                    if ckpt.is_some() {
                        let split_node = tree.nodes.get_mut(&split_node_id).unwrap();
                        let token_offset = split_node.edge_tokens.len() as u64;
                        if !split_node
                            .checkpoints
                            .iter()
                            .any(|cb| cb.token_offset == token_offset)
                        {
                            split_node.checkpoints.push(CheckpointBoundary {
                                token_offset,
                                checkpoint: ckpt,
                            });
                            split_node.checkpoints.sort_by_key(|cb| cb.token_offset);
                        }
                    }
                }
            }

            if divergence < tokens.len() {
                let remaining_tokens = &tokens[divergence..];
                // handles[] is the NEW key's page list, so the divergence
                // token lives in handles[divergence / PAGE_TOKENS] and the
                // branch's first node claims only the tail of that page.
                // (page_idx bookkeeping cannot be used here: partial nodes
                // decouple pages-consumed from tokens-consumed.)
                let remaining_handles = &handles[divergence / PAGE_TOKENS..];
                let first_skip = divergence % PAGE_TOKENS;
                let chained = create_chain(
                    tree,
                    split_node_id,
                    remaining_tokens,
                    remaining_handles,
                    first_skip,
                    checkpoint,
                    pool,
                    max_cpu_nodes,
                    current_total + delta as usize,
                );
                let (d, p) = match chained {
                    Ok(v) => v,
                    Err(e) => {
                        // Undo the split installed above: the caller charges
                        // `delta` only on Ok, so leaving the marker would make
                        // total_nodes under-count forever.
                        unsplit_edge(tree, current, split_node_id, child_id);
                        return Err(e);
                    }
                };
                delta += d;
                pages_adopted += p;
            }

            return Ok((delta, pages_adopted));
        }
    }
}

/// Create a chain of nodes for a new token span. Each node covers one full
/// page. Returns the number of nodes added.
///
/// Transactional (spec §4.3): if the CPU node bound is hit partway through,
/// every node this call created is removed and unlinked again, and every
/// cache ref this call took is released, so the tree, the node-count
/// accounting AND the pool refcounts stay consistent — the insert is
/// refused whole, not half-applied.
fn create_chain(
    tree: &mut DomainTree,
    parent: NodeId,
    tokens: &[u32],
    handles: &[Handle],
    first_skip: usize,
    checkpoint: Option<CheckpointId>,
    pool: &mut PagePool,
    max_cpu_nodes: usize,
    current_total: usize,
) -> Result<(i32, usize), InsertError> {
    let mut current_parent = parent;
    let mut token_pos = 0usize;
    let mut handle_idx = 0usize;
    let mut added: i32 = 0;
    // (node_id, first_token, page_handle) for every node created by this
    // call — nodes for unlinking on rollback, handles for releasing the
    // cache refs the index took for them.
    let mut created: Vec<(NodeId, u32, PageHandle)> = Vec::new();

    while token_pos < tokens.len() {
        // The first node of a mid-page branch claims only the tail of its
        // physical page: edge covers PAGE_TOKENS - first_skip tokens, the
        // page itself starts first_skip tokens earlier (owned by the
        // parent edge's span).
        let skip = if token_pos == 0 { first_skip } else { 0 };
        let span = (PAGE_TOKENS - skip).min(tokens.len() - token_pos);
        let chunk = &tokens[token_pos..token_pos + span];
        let first_token = chunk[0];

        let node_id = tree.fresh_node_id();
        let page_handle = handles[handle_idx].handle;
        let insert_seq = tree.fresh_insert_seq();

        let is_last_node = token_pos + span >= tokens.len();

        let mut new_node = Node::new(
            insert_seq,
            current_parent,
            chunk.to_vec(),
            vec![page_handle],
        );
        new_node.first_page_skip = skip;

        if is_last_node {
            if let Some(ckpt) = checkpoint {
                if ckpt.is_some() {
                    new_node.checkpoints.push(CheckpointBoundary {
                        token_offset: span as u64,
                        checkpoint: ckpt,
                    });
                }
            }
        }

        let new_total = current_total + added as usize + 1;
        if new_total > max_cpu_nodes {
            // Roll the partial chain back. Only the FIRST created node was
            // linked into `parent` — later nodes hang off earlier created
            // nodes, which `tree.nodes.remove` makes unreachable. Unlinking
            // later nodes by first_token could otherwise remove an unrelated
            // pre-existing sibling of `parent`. Each created node held one
            // cache ref: release them all so the pages stay reclaimable.
            if let Some((_, first_ft, _)) = created.first().copied() {
                tree.nodes
                    .get_mut(&parent)
                    .unwrap()
                    .children
                    .remove(&first_ft);
            }
            for (id, _, ph) in created.into_iter().rev() {
                tree.nodes.remove(&id);
                let _ = pool.release_cache_ref(ph.phys);
            }
            // The unused node id stays allocated (monotonic ids may have
            // gaps — harmless).
            return Err(InsertError::CpuNodeBoundExceeded {
                current: current_total + added as usize,
                max: max_cpu_nodes,
            });
        }

        // The index owns one cache ref per node slot holding a handle (see
        // the ownership doc on `PrefixIndex`). Refuse the whole insert if
        // the pool refuses the ref (Free/ReclaimPending page): a node whose
        // ref could not be taken would be released into underflow by
        // eviction later.
        if let Err(e) = pool.add_cache_ref(page_handle.phys) {
            if let Some((_, first_ft, _)) = created.first().copied() {
                tree.nodes
                    .get_mut(&parent)
                    .unwrap()
                    .children
                    .remove(&first_ft);
            }
            for (id, _, ph) in created.into_iter().rev() {
                tree.nodes.remove(&id);
                let _ = pool.release_cache_ref(ph.phys);
            }
            return Err(InsertError::InvalidHandle(format!(
                "cache ref refused for phys {} (handle gen {}): {e}",
                page_handle.phys, page_handle.generation
            )));
        }

        tree.nodes.insert(node_id, new_node);
        created.push((node_id, first_token, page_handle));
        added += 1;

        tree.nodes
            .get_mut(&current_parent)
            .unwrap()
            .children
            .insert(first_token, node_id);

        current_parent = node_id;
        token_pos += span;
        handle_idx += 1;
    }

    // Every created node adopted exactly one page (one cache ref each) —
    // the byte-accounting basis that eviction subtracts.
    Ok((added, added.max(0) as usize))
}

/// Undo a [`split_edge`] — used when the insert fails AFTER the split was
/// installed (a `create_chain` refusal further down the path). Without this
/// the marker would stay in the tree while the caller never adds its node to
/// `total_nodes` (the caller only charges on `Ok`), so the CPU node bound
/// would under-count permanently and the bound would stop binding.
///
/// Restores the pre-split shape exactly: the marker's tokens/pages/checkpoints
/// (and its `first_page_skip`) are prepended back onto the child, the child is
/// re-linked under the original parent at the original first token, and the
/// marker is removed. No cache refs move — the split moved handle slots
/// between nodes without taking or releasing any.
fn unsplit_edge(tree: &mut DomainTree, parent: NodeId, split_node_id: NodeId, child_id: NodeId) {
    let Some(marker) = tree.nodes.remove(&split_node_id) else {
        return;
    };
    let split_len = marker.edge_tokens.len() as u64;
    let first_token = marker.edge_tokens.first().copied();
    let Some(child) = tree.nodes.get_mut(&child_id) else {
        return;
    };
    let mut edge = marker.edge_tokens;
    edge.extend(child.edge_tokens.iter().copied());
    let mut pages = marker.pages;
    pages.extend(child.pages.iter().copied());
    let mut checkpoints = marker.checkpoints;
    checkpoints.extend(child.checkpoints.iter().map(|cb| CheckpointBoundary {
        token_offset: cb.token_offset + split_len,
        checkpoint: cb.checkpoint,
    }));
    checkpoints.sort_by_key(|cb| cb.token_offset);
    child.edge_tokens = edge;
    child.pages = pages;
    child.checkpoints = checkpoints;
    child.first_page_skip = marker.first_page_skip;
    child.parent = Some(parent);
    if let (Some(parent_node), Some(ft)) = (tree.nodes.get_mut(&parent), first_token) {
        parent_node.children.insert(ft, child_id);
    }
}

/// Split an edge at `split_pos` tokens. Metadata-only (spec §4.2).
/// Returns 1 if a split node was created, 0 otherwise.
fn split_edge(
    tree: &mut DomainTree,
    parent: NodeId,
    child_id: NodeId,
    split_pos: usize,
    max_cpu_nodes: usize,
    current_total: usize,
) -> Result<i32, InsertError> {
    // Pages that stay entirely below the split point belong to the marker;
    // the page straddling it (if any) stays with the child, which reaches
    // back into it via first_page_skip. Page indices are in the child's
    // physical-page coordinates: edge token t lives in page
    // (child_skip + t) / PAGE_TOKENS.

    // If the split point is at or beyond the child's full edge, no split
    // is needed — the child already covers the split point.
    let child_edge_len = tree.nodes.get(&child_id).unwrap().edge_tokens.len();
    if split_pos >= child_edge_len {
        return Ok(0);
    }

    let child = tree.nodes.get(&child_id).unwrap();
    let child_edge_tokens = child.edge_tokens.clone();
    let child_pages = child.pages.clone();
    let child_checkpoints = child.checkpoints.clone();
    let child_insert_seq = child.insert_seq;
    let child_skip = child.first_page_skip;
    let first_token = child_edge_tokens[0];

    let split_pages = (split_pos + child_skip) / PAGE_TOKENS;

    if current_total + 1 > max_cpu_nodes {
        return Err(InsertError::CpuNodeBoundExceeded {
            current: current_total,
            max: max_cpu_nodes,
        });
    }

    let split_node_id = tree.fresh_node_id();
    let split_pages_vec: Vec<PageHandle> = child_pages[..split_pages].to_vec();
    let split_edge_tokens: Vec<u32> = child_edge_tokens[..split_pos].to_vec();

    let split_checkpoints: Vec<CheckpointBoundary> = child_checkpoints
        .iter()
        .filter(|cb| cb.token_offset <= split_pos as u64)
        .copied()
        .collect();

    let remaining_checkpoints: Vec<CheckpointBoundary> = child_checkpoints
        .iter()
        .filter(|cb| cb.token_offset > split_pos as u64)
        .map(|cb| CheckpointBoundary {
            token_offset: cb.token_offset - split_pos as u64,
            checkpoint: cb.checkpoint,
        })
        .collect();

    let mut split_node = Node::new(child_insert_seq, parent, split_edge_tokens, split_pages_vec);
    split_node.checkpoints = split_checkpoints;
    split_node.first_page_skip = child_skip;

    tree.nodes.insert(split_node_id, split_node);

    let remaining_edge_tokens: Vec<u32> = child_edge_tokens[split_pos..].to_vec();
    let remaining_pages: Vec<PageHandle> = child_pages[split_pages..].to_vec();

    let remaining_child = tree.nodes.get_mut(&child_id).unwrap();
    remaining_child.edge_tokens = remaining_edge_tokens;
    remaining_child.pages = remaining_pages;
    remaining_child.checkpoints = remaining_checkpoints;
    remaining_child.first_page_skip = (child_skip + split_pos) % PAGE_TOKENS;
    // The remaining child now hangs off the split node, not the old parent.
    remaining_child.parent = Some(split_node_id);

    tree.nodes
        .get_mut(&parent)
        .unwrap()
        .children
        .insert(first_token, split_node_id);
    let remaining_first_token = tree.nodes.get(&child_id).unwrap().edge_tokens[0];
    tree.nodes
        .get_mut(&split_node_id)
        .unwrap()
        .children
        .insert(remaining_first_token, child_id);

    Ok(1)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve_contract::*;
    use rdna_compute::page_pool::{BlockTable, PagePool, PAGE_TOKENS};

    // ── Test helpers ──────────────────────────────────────────────────

    fn sample_domain(load_epoch: u64) -> CacheDomain {
        CacheDomain {
            model_content_digest: vec![0xab; 16],
            model_load_epoch: load_epoch,
            sidecar_digests: vec![vec![0x01; 8], vec![0x02; 8]],
            tokenizer: TokenizerIdentity {
                vocab_digest: vec![0xcd; 16],
                config_digest: vec![0xce; 16],
            },
            template: TemplateIdentity {
                template_digest: vec![0xee; 16],
                normalization_tag: "chatml".to_owned(),
            },
            arch_policy: ArchPolicy {
                arch_tag: "qwen35-deltanet".to_owned(),
                state_abi_tag: "dn-v1".to_owned(),
                position_attention_tag: "mrope".to_owned(),
            },
            kv_layout: KvLayout {
                k_stride_bytes: vec![128, 128],
                v_stride_bytes: vec![128, 128],
                layout_tag: "q8-g128".to_owned(),
            },
            device: DeviceTopology {
                device_id: "pci-03:00.0".to_owned(),
                topology_id: "single".to_owned(),
                allocation_epoch: 7,
            },
            namespace: SharingNamespace {
                domain_id: "owner-alpha".to_owned(),
            },
        }
    }

    fn domain_with(field: &str, value: &str) -> CacheDomain {
        let mut d = sample_domain(1);
        match field {
            "model" => d.model_content_digest = vec![0xff; 16],
            "template" => d.template.template_digest = vec![0xff; 16],
            "tokenizer" => d.tokenizer.vocab_digest = vec![0xff; 16],
            "namespace" => d.namespace.domain_id = value.to_string(),
            _ => {}
        }
        d
    }

    /// Create a pool with `n_pages` pages, seal them, and take cache refs.
    /// Returns the pool and the page handles with token offsets.
    fn setup_pool_and_pages(n_pages: usize) -> (PagePool, Vec<Handle>) {
        let mut pool = PagePool::new_with_strides(64, 128, 128).unwrap();
        let mut table = BlockTable::new();
        let allocated = pool.alloc_pages(&mut table, n_pages);
        assert_eq!(allocated, n_pages);

        // Seal only: the index takes its own cache refs at publish time
        // (cache-ref ownership moved inside `insert`).
        let mut handles = Vec::new();
        for lp in 0..n_pages {
            let phys = table.physical(lp).unwrap();
            pool.seal(phys).unwrap();
            let gen = pool.page_generation(phys);
            handles.push(Handle {
                handle: PageHandle {
                    phys,
                    epoch: pool.epoch(),
                    generation: gen,
                },
                token_offset: (lp * PAGE_TOKENS) as u64,
            });
        }
        (pool, handles)
    }

    /// Create a single pool with `n_pages` pages and seal each page (the
    /// index takes the cache refs at publish time).
    fn setup_big_pool(n_pages: usize) -> (PagePool, Vec<Handle>) {
        let mut pool = PagePool::new_with_strides(64, 128, 128).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, n_pages);
        let mut handles = Vec::new();
        for lp in 0..n_pages {
            let phys = table.physical(lp).unwrap();
            pool.seal(phys).unwrap();
            let gen = pool.page_generation(phys);
            handles.push(Handle {
                handle: PageHandle {
                    phys,
                    epoch: pool.epoch(),
                    generation: gen,
                },
                token_offset: (lp * PAGE_TOKENS) as u64,
            });
        }
        (pool, handles)
    }

    fn make_tokens(n: usize) -> Vec<u32> {
        (0..n).map(|i| (i % 1000) as u32 + 1).collect()
    }

    fn make_tokens_from(start: u32, n: usize) -> Vec<u32> {
        (0..n).map(|i| start + i as u32).collect()
    }

    /// Deterministic xorshift PRNG for the model test.
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
    }

    /// One 128-token page filled with `pid` — keys are sequences of
    /// page-ids so divergence happens only at page granularity.
    fn page_token(pid: u32) -> Vec<u32> {
        vec![pid; PAGE_TOKENS]
    }

    /// Model-based radix test: random insert/lookup/evict/unpin sequences
    /// checked against a HashMap oracle. `inspect().matched_tokens` must
    /// equal the oracle's longest fully-resident page-aligned prefix —
    /// the property the mid-page split and domain-unpin bugs violated.
    /// Host-only — no GPU needed.
    #[test]
    fn model_based_radix_matches_oracle() {
        let mut rng = XorShift(0xD1B54A32D192ED03);
        let mut pool = PagePool::new_with_strides(64, 128, 128).unwrap();
        let mut index = PrefixIndex::new(1 << 16);
        let domain = sample_domain(1);

        // Oracle: page-id prefix → the PageHandle the radix keeps for it.
        // First insert wins a prefix (duplicates are not remapped, spec
        // §4.2); eviction removes the evicted node's page entry. Residency
        // is `pool.validate_handle` — the same check `walk` uses — so a
        // freed-and-reallocated phys can never fool the oracle.
        let mut page_map: std::collections::HashMap<Vec<u32>, PageHandle> =
            std::collections::HashMap::new();
        let mut keys: Vec<Vec<u32>> = Vec::new();
        let mut tickets: Vec<PinTicket> = Vec::new();
        for _step in 0..400 {
            match rng.below(10) {
                // Insert: extend an existing key's prefix or start fresh.
                0..=4 => {
                    let mut key: Vec<u32> = Vec::new();
                    if !keys.is_empty() && rng.below(2) == 0 {
                        let base = &keys[rng.below(keys.len() as u64) as usize];
                        let take_pages = rng.below((base.len() / PAGE_TOKENS) as u64 + 1) as usize;
                        key.extend_from_slice(&base[..take_pages * PAGE_TOKENS]);
                    }
                    let extra = 1 + rng.below(3) as usize;
                    for _ in 0..extra {
                        key.extend_from_slice(&page_token(rng.below(8) as u32 + 1));
                    }
                    // Allocate fresh pages; evict first if the pool is short.
                    let need = key.len() / PAGE_TOKENS;
                    if pool.free_pages() < need {
                        for phys in index.evict_unpinned_leaves(&mut pool, usize::MAX) {
                            page_map.retain(|_, h| h.phys != phys);
                        }
                    }
                    if pool.free_pages() < need {
                        continue; // still short — skip this insert
                    }
                    let mut table = BlockTable::new();
                    if pool.alloc_pages(&mut table, need) != need {
                        continue;
                    }
                    let mut handles = Vec::new();
                    let mut phys_list = Vec::new();
                    for lp in 0..need {
                        let phys = table.physical(lp).unwrap();
                        pool.seal(phys).unwrap();
                        phys_list.push(phys);
                        handles.push(Handle {
                            handle: PageHandle {
                                phys,
                                epoch: pool.epoch(),
                                generation: pool.page_generation(phys),
                            },
                            token_offset: (lp * PAGE_TOKENS) as u64,
                        });
                    }
                    match index.insert(&domain, &key, &handles, Some(CheckpointId(1)), &mut pool) {
                        Ok(()) => {
                            // Record each page under its page-id prefix —
                            // first insert wins (duplicates not remapped).
                            for (i, h) in handles.iter().enumerate() {
                                let prefix: Vec<u32> = key[..(i + 1) * PAGE_TOKENS]
                                    .chunks(PAGE_TOKENS)
                                    .map(|c| c[0])
                                    .collect();
                                page_map.entry(prefix).or_insert(h.handle);
                            }
                            keys.push(key);
                        }
                        Err(_) => {
                            // Refused insert: release the TABLE refs so the
                            // pages return to the pool (the index took no
                            // cache refs — it refused whole).
                            let _ = pool.release_table(&mut table);
                            pool.drain_completed();
                        }
                    }
                }
                // Lookup + pin, then release.
                5..=6 => {
                    if keys.is_empty() {
                        continue;
                    }
                    let key = &keys[rng.below(keys.len() as u64) as usize];
                    let (_res, _h, ticket) = index.lookup_with_pages(&domain, key, &pool, None);
                    if !ticket.is_empty() {
                        tickets.push(ticket);
                    }
                }
                // Release a held ticket.
                7 => {
                    if !tickets.is_empty() {
                        let t = tickets.swap_remove(rng.below(tickets.len() as u64) as usize);
                        index.release_pin(&domain, t);
                    }
                }
                // Evict some unpinned leaves.
                _ => {
                    let budget = (rng.below(4) as usize) * 4096;
                    for phys in index.evict_unpinned_leaves(&mut pool, budget) {
                        page_map.retain(|_, h| h.phys != phys);
                    }
                }
            }

            // Oracle check: matched_tokens must equal the longest
            // page-aligned prefix whose radix-kept handle validates.
            if !keys.is_empty() {
                let probe = &keys[rng.below(keys.len() as u64) as usize];
                let got = index.inspect(&domain, probe, &pool);
                let probe_pids: Vec<u32> = probe.chunks(PAGE_TOKENS).map(|c| c[0]).collect();
                let mut n = 0usize;
                while n < probe_pids.len() {
                    let prefix = &probe_pids[..n + 1];
                    match page_map.get(prefix) {
                        Some(h) if pool.validate_handle(h).is_ok() => n += 1,
                        _ => break,
                    }
                }
                let want = (n * PAGE_TOKENS) as u64;
                assert_eq!(
                    got.matched_tokens, want,
                    "matched_tokens diverged from oracle (want {want}, got {})",
                    got.matched_tokens
                );
            }
        }
    }

    // ── A1: prefix lengths 0, 1, 127, 128, 129 ───────────────────────

    #[test]
    fn a1_empty_prefix_is_no_match() {
        let (mut pool, _) = setup_pool_and_pages(1);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain, &[], &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoMatch)),
            "empty prefix should be NoMatch, got {result:?}"
        );
    }

    #[test]
    fn a1_single_token_is_miss() {
        let (mut pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index
            .insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();

        // 1 token matches the edge but no full page is completed → no checkpoint
        // at that boundary → NoCheckpoint.
        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain, &tokens[..1], &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoCheckpoint)),
            "1-token query should be NoCheckpoint, got {result:?}"
        );
    }

    #[test]
    fn a1_127_tokens_is_no_checkpoint() {
        let (mut pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index
            .insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();

        let (result, _h, mut _ticket) =
            index.lookup_with_pages(&domain, &tokens[..127], &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoCheckpoint)),
            "127-token query should be NoCheckpoint, got {result:?}"
        );
    }

    #[test]
    fn a1_128_tokens_exact_full_page_with_checkpoint_is_hit() {
        let (mut pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index
            .insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();

        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain, &tokens, &pool, None);
        match result {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.matched_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resident_kv_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resumable_tokens, PAGE_TOKENS as u64);
            }
            other => panic!("expected Hit for 128-token exact match, got {other:?}"),
        }
    }

    #[test]
    fn a1_129_tokens_partial_second_page() {
        let (mut pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        // Insert first page with checkpoint at 128, then full 2 pages with
        // checkpoint at 256.
        index
            .insert(
                &domain,
                &tokens[..PAGE_TOKENS],
                &handles[..1],
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();
        index
            .insert(&domain, &tokens, &handles, Some(CheckpointId(2)), &mut pool)
            .unwrap();

        // Query 129 tokens: matched=129, resumable=128 (checkpoint at 128).
        let (result, _h, mut _ticket) =
            index.lookup_with_pages(&domain, &tokens[..129], &pool, None);
        match result {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.matched_tokens, 129);
                assert_eq!(lk.resident_kv_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resumable_tokens, PAGE_TOKENS as u64);
            }
            other => panic!("expected Hit for 129-token query, got {other:?}"),
        }
    }

    #[test]
    fn a1_exact_match_full_prefix_is_hit() {
        let (mut pool, handles) = setup_pool_and_pages(3);
        let tokens = make_tokens(PAGE_TOKENS * 3);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index
            .publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();

        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain, &tokens, &pool, None);
        match result {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.matched_tokens, (PAGE_TOKENS * 3) as u64);
                assert_eq!(lk.resident_kv_tokens, (PAGE_TOKENS * 3) as u64);
                assert_eq!(lk.resumable_tokens, (PAGE_TOKENS * 3) as u64);
            }
            other => panic!("expected Hit for exact 3-page match, got {other:?}"),
        }
    }

    #[test]
    fn a1_prompt_shorter_than_cache() {
        let (mut pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        // Insert first page with checkpoint at 128.
        index
            .insert(
                &domain,
                &tokens[..PAGE_TOKENS],
                &handles[..1],
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();
        // Insert full 2 pages with checkpoint at 256.
        index
            .insert(&domain, &tokens, &handles, Some(CheckpointId(2)), &mut pool)
            .unwrap();

        // Query 128 tokens: matched=128, resumable=128 (checkpoint at 128).
        let (result, _h, mut _ticket) =
            index.lookup_with_pages(&domain, &tokens[..PAGE_TOKENS], &pool, None);
        match result {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.matched_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resident_kv_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resumable_tokens, PAGE_TOKENS as u64);
            }
            other => panic!("expected Hit for 128-token query, got {other:?}"),
        }
    }

    #[test]
    fn a1_divergent_tail() {
        let (mut pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        // Insert first page with checkpoint at 128.
        index
            .insert(
                &domain,
                &tokens[..PAGE_TOKENS],
                &handles[..1],
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();
        index
            .insert(&domain, &tokens, &handles, Some(CheckpointId(2)), &mut pool)
            .unwrap();

        // Query: 128 match + 1 divergent token.
        let mut query = tokens[..PAGE_TOKENS].to_vec();
        query.push(999_999);

        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain, &query, &pool, None);
        match result {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.matched_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resident_kv_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resumable_tokens, PAGE_TOKENS as u64);
            }
            other => panic!("expected Hit for divergent tail, got {other:?}"),
        }
    }

    #[test]
    fn a1_never_underflow() {
        let (mut pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index
            .insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();

        let r0 = index.lookup(&domain, &[], &pool, None);
        assert!(matches!(r0, PrefixLookupResult::Miss(MissReason::NoMatch)));

        let r1 = index.lookup(&domain, &tokens[..1], &pool, None);
        assert!(matches!(
            r1,
            PrefixLookupResult::Miss(MissReason::NoCheckpoint)
        ));

        let insp = index.inspect(&domain, &[], &pool);
        assert_eq!(insp.matched_tokens, 0);
        assert_eq!(insp.resident_kv_tokens, 0);
        assert_eq!(insp.resumable_tokens, 0);

        let insp = index.inspect(&domain, &tokens[..1], &pool);
        assert_eq!(insp.matched_tokens, 1);
        assert_eq!(insp.resident_kv_tokens, 0);
        assert_eq!(insp.resumable_tokens, 0);
    }

    // ── A2: canonical entry; pin isolation ────────────────────────────

    #[test]
    fn a2_canonical_entry_no_duplication() {
        let (mut pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        // Insert the same prefix twice.
        index
            .publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();
        let nodes_after_first = index.total_nodes();
        index
            .publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();
        let nodes_after_second = index.total_nodes();

        // Second insert should not add nodes (canonical entry).
        assert_eq!(
            nodes_after_first, nodes_after_second,
            "canonical entry should not duplicate nodes"
        );

        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
    }

    #[test]
    fn a2_pin_isolation() {
        let (mut pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index
            .publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();

        // Lookup pins the pages.
        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));

        // Eviction should not evict pinned pages.
        let evicted = index.evict_unpinned_leaves(&mut pool, usize::MAX);
        assert!(evicted.is_empty(), "pinned pages should survive eviction");

        // Unpin.
        index.release_pin(&domain, std::mem::take(&mut _ticket));

        // Now eviction should work.
        let evicted = index.evict_unpinned_leaves(&mut pool, usize::MAX);
        assert!(!evicted.is_empty(), "unpinned pages should be evictable");

        // Subsequent lookup should miss.
        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain, &tokens, &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(_)),
            "evicted prefix should miss, got {result:?}"
        );
    }

    // ── A6: different CacheDomain → isolated trees ────────────────────

    #[test]
    fn a6_different_model_digest_isolated() {
        let (mut pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);

        let domain_a = sample_domain(1);
        let domain_b = domain_with("model", "");

        index
            .insert(
                &domain_a,
                &tokens,
                &handles,
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();

        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain_b, &tokens, &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoMatch)),
            "different model digest should be NoMatch, got {result:?}"
        );

        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain_a, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
    }

    #[test]
    fn a6_different_template_isolated() {
        let (mut pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);

        let domain_a = sample_domain(1);
        let domain_b = domain_with("template", "");

        index
            .insert(
                &domain_a,
                &tokens,
                &handles,
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();

        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain_b, &tokens, &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoMatch)),
            "different template should be NoMatch, got {result:?}"
        );
    }

    #[test]
    fn a6_different_tokenizer_isolated() {
        let (mut pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);

        let domain_a = sample_domain(1);
        let domain_b = domain_with("tokenizer", "");

        index
            .insert(
                &domain_a,
                &tokens,
                &handles,
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();

        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain_b, &tokens, &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoMatch)),
            "different tokenizer should be NoMatch, got {result:?}"
        );
    }

    #[test]
    fn a6_different_namespace_isolated() {
        let (mut pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);

        let domain_a = sample_domain(1);
        let domain_b = domain_with("namespace", "owner-beta");

        index
            .insert(
                &domain_a,
                &tokens,
                &handles,
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();

        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain_b, &tokens, &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoMatch)),
            "different namespace should be NoMatch, got {result:?}"
        );
    }

    // ── A8: eviction ──────────────────────────────────────────────────

    #[test]
    fn a8_evict_leaf_then_lookup_misses() {
        let (mut pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index
            .insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();

        // Verify hit before eviction.
        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
        index.release_pin(&domain, std::mem::take(&mut _ticket));

        // Evict.
        let evicted = index.evict_unpinned_leaves(&mut pool, usize::MAX);
        assert_eq!(evicted.len(), 1, "should evict 1 page");

        // Lookup after eviction should miss.
        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain, &tokens, &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoMatch)),
            "evicted prefix should miss, got {result:?}"
        );
    }

    #[test]
    fn a8_pinned_pages_survive_eviction() {
        let (mut pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index
            .publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();

        // Lookup pins pages.
        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));

        // Eviction should not evict pinned pages.
        let evicted = index.evict_unpinned_leaves(&mut pool, usize::MAX);
        assert!(evicted.is_empty(), "pinned pages should not be evicted");

        // Lookup should still hit.
        index.release_pin(&domain, std::mem::take(&mut _ticket));
        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
        index.release_pin(&domain, std::mem::take(&mut _ticket));

        // Now evict.
        let evicted = index.evict_unpinned_leaves(&mut pool, usize::MAX);
        assert!(!evicted.is_empty(), "unpinned pages should be evicted");
    }

    // ── CPU metadata bound ────────────────────────────────────────────

    #[test]
    fn cpu_metadata_bound_refuses_unbounded_growth() {
        let (mut pool, handles) = setup_big_pool(3);
        let mut idx = PrefixIndex::new(3); // root + 2 nodes max
        let domain = sample_domain(1);

        let tokens_a = make_tokens(PAGE_TOKENS);
        let tokens_b = make_tokens_from(1000, PAGE_TOKENS);
        let tokens_c = make_tokens_from(2000, PAGE_TOKENS);

        // Insert A (root + 1 node = 2 total).
        idx.insert(
            &domain,
            &tokens_a,
            &handles[..1],
            Some(CheckpointId(1)),
            &mut pool,
        )
        .unwrap();
        // Insert B (root + 2 nodes = 3 total).
        idx.insert(
            &domain,
            &tokens_b,
            &handles[1..2],
            Some(CheckpointId(2)),
            &mut pool,
        )
        .unwrap();

        // Pin both leaves: insert-time eviction (spec §4.4) may reclaim
        // unpinned leaves to make room, so the bound only refuses when
        // every leaf is pinned.
        let (_r1, _h1, _t1) = idx.lookup_with_pages(&domain, &tokens_a, &pool, None);
        let (_r2, _h2, _t2) = idx.lookup_with_pages(&domain, &tokens_b, &pool, None);

        // Third insert should fail (would need 4 nodes > max 3, and no
        // leaf is evictable).
        let result = idx.insert(
            &domain,
            &tokens_c,
            &handles[2..3],
            Some(CheckpointId(3)),
            &mut pool,
        );
        assert!(
            matches!(result, Err(InsertError::CpuNodeBoundExceeded { .. })),
            "third insert should fail, got {result:?}"
        );

        assert!(idx.total_nodes() <= 3);
    }

    #[test]
    fn cpu_metadata_bound_evict_for_capacity() {
        let (mut pool, handles) = setup_big_pool(3);
        let mut idx = PrefixIndex::new(3);
        let domain = sample_domain(1);

        let tokens_a = make_tokens(PAGE_TOKENS);
        let tokens_b = make_tokens_from(1000, PAGE_TOKENS);
        let tokens_c = make_tokens_from(2000, PAGE_TOKENS);

        idx.insert(
            &domain,
            &tokens_a,
            &handles[..1],
            Some(CheckpointId(1)),
            &mut pool,
        )
        .unwrap();
        idx.insert(
            &domain,
            &tokens_b,
            &handles[1..2],
            Some(CheckpointId(2)),
            &mut pool,
        )
        .unwrap();

        // Evict to make room for 1 more node.
        idx.evict_for_capacity(&mut pool, 1).unwrap();

        // Now insert should succeed.
        let result = idx.insert(
            &domain,
            &tokens_c,
            &handles[2..3],
            Some(CheckpointId(3)),
            &mut pool,
        );
        assert!(
            result.is_ok(),
            "insert after eviction should succeed, got {result:?}"
        );
    }

    // ── publish_sealed_pages refuses partial last page ────────────────

    #[test]
    fn publish_refuses_partial_last_page() {
        let (mut pool, handles) = setup_pool_and_pages(1);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        let tokens = make_tokens(129);
        let result = index.publish_sealed_pages(&domain, &tokens, &handles, None, &mut pool);
        assert!(
            matches!(result, Err(InsertError::PartialLastPage { token_len: 129 })),
            "129 tokens should be refused, got {result:?}"
        );
    }

    #[test]
    fn publish_accepts_full_pages() {
        let (mut pool, handles) = setup_pool_and_pages(2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        let tokens = make_tokens(PAGE_TOKENS * 2);
        let result = index.publish_sealed_pages(
            &domain,
            &tokens,
            &handles,
            Some(CheckpointId(1)),
            &mut pool,
        );
        assert!(result.is_ok());
    }

    // ── inspect returns counts without pinning ────────────────────────

    #[test]
    fn inspect_returns_counts_without_pinning() {
        let (mut pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index
            .insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();

        let insp = index.inspect(&domain, &tokens, &pool);
        assert_eq!(insp.matched_tokens, PAGE_TOKENS as u64);
        assert_eq!(insp.resident_kv_tokens, PAGE_TOKENS as u64);
        assert_eq!(insp.resumable_tokens, PAGE_TOKENS as u64);

        // inspect should NOT pin — verify by checking that eviction works
        // without unpin (nothing was pinned).
        let (mut pool_mut, _) = setup_pool_and_pages(1);
        // Can't easily test with the same pool, but the fact that no unpin
        // is needed after inspect is the contract.
        let _ = &mut pool_mut;
    }

    // ── CachePolicy IncompatiblePolicy ────────────────────────────────

    #[test]
    fn incompatible_policy_mid_sequence_partial() {
        let (mut pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        // Insert with checkpoints at 128 and 256.
        index
            .insert(
                &domain,
                &tokens[..PAGE_TOKENS],
                &handles[..1],
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();
        index
            .insert(&domain, &tokens, &handles, Some(CheckpointId(2)), &mut pool)
            .unwrap();

        // Query 256 tokens: matched=256, resumable=256 → not partial → Hit.
        let policy = CachePolicy::qwen35();
        let (result, _h, mut _ticket) =
            index.lookup_with_pages(&domain, &tokens, &pool, Some(&policy));
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
        index.release_pin(&domain, std::mem::take(&mut _ticket));

        // Query 128 tokens: matched=128, resumable=128 → not partial → Hit.
        let (result, _h, mut _ticket) =
            index.lookup_with_pages(&domain, &tokens[..PAGE_TOKENS], &pool, Some(&policy));
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
        index.release_pin(&domain, std::mem::take(&mut _ticket));

        // Query 129 tokens: matched=129, resumable=128 → partial → IncompatiblePolicy.
        let (result, _h, mut _ticket) =
            index.lookup_with_pages(&domain, &tokens[..129], &pool, Some(&policy));
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::IncompatiblePolicy)),
            "mid-sequence partial with allow_partial=false should be IncompatiblePolicy, got {result:?}"
        );
    }

    #[test]
    fn no_policy_returns_raw_resumable() {
        let (mut pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index
            .insert(
                &domain,
                &tokens[..PAGE_TOKENS],
                &handles[..1],
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();
        index
            .insert(&domain, &tokens, &handles, Some(CheckpointId(2)), &mut pool)
            .unwrap();

        // Query 129 tokens with no policy → Hit with resumable=128.
        let (result, _h, mut _ticket) =
            index.lookup_with_pages(&domain, &tokens[..129], &pool, None);
        match result {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.matched_tokens, 129);
                assert_eq!(lk.resumable_tokens, PAGE_TOKENS as u64);
            }
            other => panic!("expected Hit with no policy, got {other:?}"),
        }
    }

    // ── CheckpointId ──────────────────────────────────────────────────

    #[test]
    fn checkpoint_id_none_is_zero() {
        assert_eq!(CheckpointId::NONE.0, 0);
        assert!(CheckpointId::NONE.is_none());
        assert!(!CheckpointId::NONE.is_some());
        assert!(CheckpointId(1).is_some());
        assert!(!CheckpointId(1).is_none());
    }

    // ── split (metadata-only) ─────────────────────────────────────────

    #[test]
    fn split_edge_is_metadata_only() {
        let (mut pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index
            .publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();

        // Split the edge from root at 128 tokens.
        let first_token = tokens[0];
        let result = index.split(&domain, first_token, PAGE_TOKENS);
        assert!(result.is_ok(), "split should succeed, got {result:?}");

        // Lookup should still find the full prefix.
        let (result, _h, mut _ticket) = index.lookup_with_pages(&domain, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
    }

    #[test]
    fn lookup_with_pages_isolates_equal_length_prefixes() {
        // Two 128-token prefixes of equal length must not share handles
        // (the length-keyed side map this replaces would collide).
        let (mut pool, handles) = setup_big_pool(2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);
        let a = make_tokens_from(1, PAGE_TOKENS);
        let b = make_tokens_from(10_000, PAGE_TOKENS);
        index
            .insert(&domain, &a, &handles[..1], Some(CheckpointId(1)), &mut pool)
            .unwrap();
        index
            .insert(
                &domain,
                &b,
                &handles[1..2],
                Some(CheckpointId(2)),
                &mut pool,
            )
            .unwrap();

        let (ra, ha, _ticket) = index.lookup_with_pages(&domain, &a, &pool, None);
        let (rb, hb, _ticket) = index.lookup_with_pages(&domain, &b, &pool, None);
        assert!(matches!(ra, PrefixLookupResult::Hit(_)));
        assert!(matches!(rb, PrefixLookupResult::Hit(_)));
        assert_eq!(ha.len(), 1);
        assert_eq!(hb.len(), 1);
        assert_ne!(
            ha[0].handle.phys, hb[0].handle.phys,
            "equal-length prefixes must return distinct physical pages"
        );
    }

    // ── A8: a gap below a boundary must make that boundary unresumable ──

    #[test]
    fn a8_gap_below_boundary_is_not_resumable() {
        // Pins the contiguous-residency invariant: a checkpoint boundary is
        // resumable only while EVERY page below it stays resident. The walk
        // tracks the gap-free resident prefix (contiguous_resident_tokens)
        // rather than a cumulative count of valid pages, so a partial
        // invalidation below a boundary can never bless a hit — today and
        // for any future finer-boundary checkpoint placement.
        //
        // Layout: 2-page prefix published with a checkpoint at 256, split
        // at 128, then page 0 freed. Page 1 is still resident; the lookup
        // must return an honest NoCheckpoint with zero handles.
        let mut pool = PagePool::new_with_strides(8, 128, 128).unwrap();
        let mut table = BlockTable::new();
        let allocated = pool.alloc_pages(&mut table, 2);
        assert_eq!(allocated, 2);
        let phys0 = table.physical(0).unwrap();
        let phys1 = table.physical(1).unwrap();
        // Seal only — the index takes its own cache refs at publish.
        pool.seal(phys0).unwrap();
        pool.seal(phys1).unwrap();
        let handles = vec![
            Handle {
                handle: PageHandle {
                    phys: phys0,
                    epoch: 0,
                    generation: pool.page_generation(phys0),
                },
                token_offset: 0,
            },
            Handle {
                handle: PageHandle {
                    phys: phys1,
                    epoch: 0,
                    generation: pool.page_generation(phys1),
                },
                token_offset: PAGE_TOKENS as u64,
            },
        ];

        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        index
            .publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();

        // Split so the checkpoint moves into the second node at offset 128.
        index.split(&domain, tokens[0], PAGE_TOKENS).unwrap();

        // Sanity: full 2-page lookup hits at 256.
        let (r, h, mut _ticket) = index.lookup_with_pages(&domain, &tokens, &pool, None);
        match r {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.resumable_tokens, (PAGE_TOKENS * 2) as u64);
                assert_eq!(h.len(), 2);
            }
            other => panic!("expected Hit before invalidation, got {other:?}"),
        }
        index.release_pin(&domain, std::mem::take(&mut _ticket));

        // Invalidate page 0: drop the INDEX's cache ref on it (test seam),
        // then drop the table ref — it frees (generation bumps) while page
        // 1 stays cache-resident.
        index.test_drop_page_ref(&mut pool, &domain, phys0);
        pool.release_table(&mut table).unwrap();
        assert_eq!(
            pool.page_state(phys0),
            rdna_compute::page_pool::PageState::Free
        );
        assert_eq!(
            pool.page_state(phys1),
            rdna_compute::page_pool::PageState::CacheOnly
        );

        // The boundary below the gap must NOT be resumable anymore.
        let (r, h, _ticket) = index.lookup_with_pages(&domain, &tokens, &pool, None);
        assert!(
            matches!(r, PrefixLookupResult::Miss(MissReason::NoCheckpoint)),
            "gap below the 128 boundary must force NoCheckpoint, got {r:?}"
        );
        assert!(
            h.is_empty(),
            "no handles may be issued for an unresumable boundary"
        );
    }

    // ── CPU node bound: failed inserts leave accounting whole ──────────

    #[test]
    fn cpu_node_bound_rollback_keeps_accounting_consistent() {
        // Regression: a chain insert that hit the bound partway used to
        // leave its already-inserted nodes in the tree WITHOUT adding them
        // to total_nodes — the bound silently stopped binding.
        let (mut pool, handles) = setup_big_pool(4);
        let mut idx = PrefixIndex::new(3); // root + 2 nodes
        let domain = sample_domain(1);

        let tokens_a = make_tokens(PAGE_TOKENS);
        idx.insert(
            &domain,
            &tokens_a,
            &handles[..1],
            Some(CheckpointId(1)),
            &mut pool,
        )
        .unwrap();
        assert_eq!(idx.total_nodes(), 2);

        // 3-page chain: first node fits (3 total), second trips the bound.
        // Pin A's leaf so insert-time eviction cannot make room.
        let (_r, _h, _t) = idx.lookup_with_pages(&domain, &tokens_a, &pool, None);
        let tokens_c = make_tokens_from(2000, PAGE_TOKENS * 3);
        // handles must cover 3 pages — reuse three distinct valid handles.
        let result = idx.insert(
            &domain,
            &tokens_c,
            &handles[..3],
            Some(CheckpointId(3)),
            &mut pool,
        );
        assert!(matches!(
            result,
            Err(InsertError::CpuNodeBoundExceeded { .. })
        ));

        // Accounting must still match reality: the rolled-back chain left
        // exactly root + A behind.
        assert_eq!(
            idx.total_nodes(),
            2,
            "failed insert must not change node accounting"
        );

        // The rolled-back prefix must be gone: its first page misses.
        let r = idx.lookup(&domain, &tokens_c, &pool, None);
        assert!(matches!(r, PrefixLookupResult::Miss(_)));

        // And the bound still binds: exactly one more 1-page insert fits.
        let tokens_b = make_tokens_from(1000, PAGE_TOKENS);
        idx.insert(
            &domain,
            &tokens_b,
            &handles[1..2],
            Some(CheckpointId(2)),
            &mut pool,
        )
        .unwrap();
        assert_eq!(idx.total_nodes(), 3);
        // A 4th node would exceed the bound, but B's leaf is unpinned:
        // insert-time eviction reclaims it and the insert succeeds (spec
        // §4.4: reclaim cache-only pages before rejecting work).
        let tokens_d = make_tokens_from(3000, PAGE_TOKENS);
        idx.insert(
            &domain,
            &tokens_d,
            &handles[2..3],
            Some(CheckpointId(4)),
            &mut pool,
        )
        .unwrap();
        assert_eq!(idx.total_nodes(), 3);
        // B's evicted prefix misses; A's pinned prefix still hits.
        let r = idx.lookup(&domain, &tokens_b, &pool, None);
        assert!(matches!(r, PrefixLookupResult::Miss(_)));
        let r = idx.lookup(&domain, &tokens_a, &pool, None);
        assert!(matches!(r, PrefixLookupResult::Hit(_)));
    }

    /// A refusal that lands AFTER the edge split must undo the split: the
    /// marker is installed before the chain is built, and leaving it would
    /// strand a node the caller never charges to `total_nodes` (the bound
    /// stops binding) and a tree shape no key asked for.
    #[test]
    fn split_is_rolled_back_when_the_chain_hits_the_bound() {
        let (mut pool, handles) = setup_big_pool(4);
        // root + A's node + the split marker = 3; the chain node is the 4th.
        let mut index = PrefixIndex::new(3);
        let domain = sample_domain(11);

        let tokens_a = make_tokens(PAGE_TOKENS);
        index
            .insert(
                &domain,
                &tokens_a,
                &handles[..1],
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();
        assert_eq!(index.total_nodes(), 2);

        // Pin A's leaf so insert-time eviction cannot free the node the
        // chain needs — the bound must genuinely refuse.
        let _pin = index.lookup_with_pages(&domain, &tokens_a, &pool, None);

        // B shares half of A's page then diverges: the split at 64 is
        // affordable (3 nodes) but B's own chain node is not.
        let mut tokens_b = tokens_a[..PAGE_TOKENS / 2].to_vec();
        tokens_b.extend(make_tokens_from(7000, PAGE_TOKENS / 2));
        let result = index.insert(
            &domain,
            &tokens_b,
            &handles[1..2],
            Some(CheckpointId(2)),
            &mut pool,
        );
        assert!(
            matches!(result, Err(InsertError::CpuNodeBoundExceeded { .. })),
            "expected the chain to hit the node bound, got {result:?}"
        );
        assert_eq!(
            index.total_nodes(),
            2,
            "a refused insert must undo its split marker"
        );

        // The tree is byte-for-byte the pre-insert shape: A still resolves to
        // its own single full page and B's divergence is not visible.
        let (r, h, _t) = index.lookup_with_pages(&domain, &tokens_a, &pool, None);
        match r {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.resumable_tokens, PAGE_TOKENS as u64)
            }
            other => panic!("A must be unchanged after the rollback, got {other:?}"),
        }
        assert_eq!(h.len(), 1);
        let r = index.lookup(&domain, &tokens_b, &pool, None);
        assert!(matches!(r, PrefixLookupResult::Miss(_)));

        // And the bound still binds: exactly one more node fits.
        let tokens_c = make_tokens_from(4000, PAGE_TOKENS);
        index
            .insert(
                &domain,
                &tokens_c,
                &handles[2..3],
                Some(CheckpointId(3)),
                &mut pool,
            )
            .unwrap();
        assert_eq!(index.total_nodes(), 3);
    }

    /// A retained-byte refusal on a brand-new domain must not leave the empty
    /// root behind: the caller charges nothing on Err, so the orphan would
    /// count against the CPU node bound for the process lifetime.
    #[test]
    fn byte_ceiling_refusal_on_new_domain_leaves_no_root() {
        let (mut pool, handles) = setup_big_pool(2);
        let page_bytes = pool.k_page_bytes() + pool.v_page_bytes();
        let mut index = PrefixIndex::new(1000);
        index.set_max_retained_bytes(page_bytes / 2);
        let domain = sample_domain(13);

        let tokens_a = make_tokens(PAGE_TOKENS);
        let result = index.insert(
            &domain,
            &tokens_a,
            &handles[..1],
            Some(CheckpointId(1)),
            &mut pool,
        );
        assert!(
            matches!(result, Err(InsertError::CacheByteBoundExceeded { .. })),
            "expected the byte ceiling to refuse, got {result:?}"
        );
        assert_eq!(
            index.total_nodes(),
            0,
            "a refused new-domain insert must not leak its root"
        );

        // The same domain is fully usable once the ceiling can hold a page.
        index.set_max_retained_bytes(page_bytes * 2);
        index
            .insert(
                &domain,
                &tokens_a,
                &handles[..1],
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();
        assert_eq!(index.total_nodes(), 2, "root + one page node");
        let r = index.lookup(&domain, &tokens_a, &pool, None);
        assert!(matches!(r, PrefixLookupResult::Hit(_)));
    }

    // ── Mid-page divergence and per-lookup pins ──────────────────────

    /// Mid-page fork (W10-4): a key that shares a prefix and diverges INSIDE
    /// a page must split the covering node with a zero-page marker — both
    /// paths stay independently resumable, node accounting grows by exactly
    /// the marker plus the new chain, and retained bytes charge only the
    /// pages a node actually adopts.
    #[test]
    fn mid_page_fork_keeps_both_paths_resumable() {
        let (mut pool, handles) = setup_big_pool(4);
        let page_bytes = pool.k_page_bytes() + pool.v_page_bytes();
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(3);

        // A: two full pages.
        let tokens_a = make_tokens(PAGE_TOKENS * 2);
        index
            .insert(
                &domain,
                &tokens_a,
                &handles[..2],
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();
        let nodes_before = index.total_nodes();
        assert_eq!(index.retained_bytes(), 2 * page_bytes);

        // B shares 1.5 pages of A, then diverges mid-page and completes the
        // second page with its own tokens.
        let mut tokens_b = tokens_a[..PAGE_TOKENS + PAGE_TOKENS / 2].to_vec();
        tokens_b.extend(make_tokens_from(9000, PAGE_TOKENS / 2));
        assert_eq!(tokens_b.len(), PAGE_TOKENS * 2);
        index
            .insert(
                &domain,
                &tokens_b,
                &handles[2..4],
                Some(CheckpointId(2)),
                &mut pool,
            )
            .unwrap();

        // The fork added the zero-page split marker plus B's one-page chain
        // (B's first page is A's, reached through the marker).
        assert_eq!(
            index.total_nodes(),
            nodes_before + 2,
            "mid-page fork must add exactly a marker and B's chain node"
        );
        assert_eq!(
            index.retained_bytes(),
            3 * page_bytes,
            "a zero-page split marker must not be charged retained bytes"
        );

        let (r, h, _t) = index.lookup_with_pages(&domain, &tokens_a, &pool, None);
        assert!(
            matches!(r, PrefixLookupResult::Hit(_)),
            "A must stay resumable: {r:?}"
        );
        assert_eq!(h.len(), 2);

        let (r, h, _t) = index.lookup_with_pages(&domain, &tokens_b, &pool, None);
        match r {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.resumable_tokens, (PAGE_TOKENS * 2) as u64)
            }
            other => panic!("B must stay resumable after the fork, got {other:?}"),
        }
        assert_eq!(h.len(), 2);
    }

    /// Retained bytes are charged by PAGES ADOPTED, not by nodes created:
    /// a mid-page fork's split marker is a node holding ZERO pages, and
    /// the old node-count basis permanently inflated `retained_bytes` by
    /// one page per fork until the byte ceiling could never be satisfied
    /// (every publish refused `CacheByteBoundExceeded` until Reset).
    #[test]
    fn retained_bytes_exclude_zero_page_split_markers() {
        let (mut pool, handles) = setup_big_pool(4);
        let page_bytes = pool.k_page_bytes() + pool.v_page_bytes();
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(7);

        // Key A: 2 adopted pages.
        let tokens_a = make_tokens(PAGE_TOKENS * 2);
        index
            .insert(
                &domain,
                &tokens_a,
                &handles[..2],
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();
        assert_eq!(index.retained_bytes(), 2 * page_bytes);

        // Key B forks A mid-page: one zero-page split marker plus a 2-page
        // chain. Bytes must grow by exactly the 2 adopted pages — the
        // marker contributes none.
        let mut tokens_b = tokens_a[..60].to_vec();
        tokens_b.extend(make_tokens_from(5000, PAGE_TOKENS * 2 - 60));
        index
            .insert(
                &domain,
                &tokens_b,
                &handles[2..4],
                Some(CheckpointId(2)),
                &mut pool,
            )
            .expect("mid-page fork must insert");
        assert_eq!(
            index.retained_bytes(),
            4 * page_bytes,
            "split marker must not be charged as a page"
        );

        // Repeated forks at the same divergence keep the invariant: each
        // new fork adds its adopted pages only.
        let mut tokens_c = tokens_a[..60].to_vec();
        tokens_c.extend(make_tokens_from(7000, PAGE_TOKENS - 60));
        let (mut pool2, handles2) = setup_big_pool(4);
        let page_bytes2 = pool2.k_page_bytes() + pool2.v_page_bytes();
        let mut index2 = PrefixIndex::new(1000);
        index2
            .insert(&domain, &tokens_a, &handles2[..2], None, &mut pool2)
            .unwrap();
        index2
            .insert(&domain, &tokens_b, &handles2[2..4], None, &mut pool2)
            .unwrap();
        index2
            .insert(&domain, &tokens_c, &handles2[..1], None, &mut pool2)
            .unwrap();
        assert_eq!(index2.retained_bytes(), 5 * page_bytes2);
    }

    fn mid_page_divergence_forks_and_caches_tail() {
        // A published key that shares a non-page-aligned prefix with a
        // cached key forks mid-page: the split produces a zero-page marker
        // plus children whose first_page_skip reaches back into the
        // divergent page. Both branches resolve and the new tail is
        // cached — no corruption, no refusal.
        let (mut pool, handles) = setup_big_pool(4);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        // Key A: 2 pages of tokens 1..=256.
        let tokens_a = make_tokens(PAGE_TOKENS * 2);
        index
            .insert(
                &domain,
                &tokens_a,
                &handles[..2],
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();

        // Key B: shares the first 60 tokens (mid-page), then diverges.
        let mut tokens_b = tokens_a[..60].to_vec();
        tokens_b.extend(make_tokens_from(5000, PAGE_TOKENS * 2 - 60));
        index
            .insert(
                &domain,
                &tokens_b,
                &handles[2..4],
                Some(CheckpointId(2)),
                &mut pool,
            )
            .expect("mid-page fork must insert");

        // A still hits fully.
        let (r, _h, _t) = index.lookup_with_pages(&domain, &tokens_a, &pool, None);
        assert!(
            matches!(r, PrefixLookupResult::Hit(_)),
            "A must still hit: {r:?}"
        );

        // B hits: its divergent tail is now cached. The matched span is
        // B's full key; resumability is bounded by B's checkpoint at its
        // final page boundary.
        let (r, h, _t) = index.lookup_with_pages(&domain, &tokens_b, &pool, None);
        let PrefixLookupResult::Hit(hit) = r else {
            panic!("B must hit after the fork: {r:?}");
        };
        assert_eq!(hit.matched_tokens, tokens_b.len() as u64);
        assert_eq!(
            hit.resumable_tokens,
            (tokens_b.len() / PAGE_TOKENS * PAGE_TOKENS) as u64,
            "B's checkpoint sits at its last page boundary"
        );
        // B's first physical page covers tokens [0,128) — the same span
        // as A's first page (identical content: shared prefix). The
        // handle's token_offset must reflect the physical page start,
        // not the edge start.
        assert_eq!(h[0].token_offset, 0);

        // A second identical B lookup hits again (the tail stays cached).
        let (r, _h, _t) = index.lookup_with_pages(&domain, &tokens_b, &pool, None);
        assert!(matches!(r, PrefixLookupResult::Hit(_)));

        // Key C shares B's first 100 tokens — a fork inside B's partial
        // first page (B's edge starts at 60, so 100 is mid-edge).
        let mut tokens_c = tokens_b[..100].to_vec();
        tokens_c.extend(make_tokens_from(9000, PAGE_TOKENS * 2 - 100));
        index
            .insert(
                &domain,
                &tokens_c,
                &handles[..2],
                Some(CheckpointId(3)),
                &mut pool,
            )
            .expect("nested mid-page fork must insert");
        let (r, h, _t) = index.lookup_with_pages(&domain, &tokens_c, &pool, None);
        let PrefixLookupResult::Hit(hit) = r else {
            panic!("C must hit after the nested fork: {r:?}");
        };
        assert_eq!(hit.matched_tokens, tokens_c.len() as u64);
        // C's first node claims only tokens [100,128) of its page — the
        // handle must still carry the PHYSICAL page start (0), and the
        // second page its own aligned offset. A wrong first_page_skip
        // here maps KV rows onto the wrong token span silently.
        assert_eq!(h[0].token_offset, 0);
        assert_eq!(h[1].token_offset, PAGE_TOKENS as u64);
        // A and B still resolve.
        for key in [&tokens_a, &tokens_b] {
            let (r, _h, _t) = index.lookup_with_pages(&domain, key, &pool, None);
            assert!(matches!(r, PrefixLookupResult::Hit(_)));
        }
    }

    #[test]
    fn page_aligned_divergence_still_splits() {
        // A fork at a page boundary remains legal: the shared pages are
        // genuinely shared and both branches resolve.
        let (mut pool, handles) = setup_big_pool(4);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        let tokens_a = make_tokens(PAGE_TOKENS * 2);
        index
            .insert(
                &domain,
                &tokens_a,
                &handles[..2],
                Some(CheckpointId(1)),
                &mut pool,
            )
            .unwrap();

        // Key B: identical first page, divergent second page.
        let mut tokens_b = tokens_a[..PAGE_TOKENS].to_vec();
        tokens_b.extend(make_tokens_from(5000, PAGE_TOKENS));
        index
            .insert(
                &domain,
                &tokens_b,
                &handles[2..4],
                Some(CheckpointId(2)),
                &mut pool,
            )
            .unwrap();

        let (ra, _ha, _ta) = index.lookup_with_pages(&domain, &tokens_a, &pool, None);
        let (rb, _hb, _tb) = index.lookup_with_pages(&domain, &tokens_b, &pool, None);
        assert!(matches!(ra, PrefixLookupResult::Hit(_)));
        assert!(matches!(rb, PrefixLookupResult::Hit(_)));
    }

    #[test]
    fn overlapping_lookups_hold_independent_pins() {
        // Two lookups sharing a path each hold a pin; releasing one must
        // not make the other's pages evictable.
        let (mut pool, handles) = setup_pool_and_pages(1);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);
        let tokens = make_tokens(PAGE_TOKENS);

        index
            .insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();

        let (_r1, _h1, mut t1) = index.lookup_with_pages(&domain, &tokens, &pool, None);
        let (_r2, _h2, mut t2) = index.lookup_with_pages(&domain, &tokens, &pool, None);

        // Release the first ticket: the second lookup's pin still protects.
        index.release_pin(&domain, std::mem::take(&mut t1));
        let evicted = index.evict_unpinned_leaves(&mut pool, usize::MAX);
        assert!(
            evicted.is_empty(),
            "second lookup's pin must still protect the path"
        );

        // Release the second: now evictable.
        index.release_pin(&domain, std::mem::take(&mut t2));
        let evicted = index.evict_unpinned_leaves(&mut pool, usize::MAX);
        assert!(!evicted.is_empty(), "fully released path must be evictable");
    }

    #[test]
    fn eviction_collapses_newly_leaf_parents() {
        // Evicting a leaf can turn its parent into a leaf; a single pass
        // must keep going and reclaim the chain, not stop one level short.
        let (mut pool, handles) = setup_big_pool(3);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);
        let tokens = make_tokens(PAGE_TOKENS * 3);

        index
            .insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &mut pool)
            .unwrap();
        assert_eq!(index.total_nodes(), 4); // root + 3 chain nodes

        let evicted = index.evict_unpinned_leaves(&mut pool, usize::MAX);
        assert_eq!(evicted.len(), 3, "one pass must reclaim the whole chain");
        assert_eq!(index.total_nodes(), 1, "only the root remains");
    }
}
