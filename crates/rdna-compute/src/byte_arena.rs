// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bounded byte arenas for the S1 (slow-link) weight and F16-shadow storage.
//!
//! This module is pure host logic: a deterministic named-member layout
//! ([`NamedByteArenaPlan`]), the model-level pair of plans that the census, the
//! weight loader and [`crate::Gpu`] all share ([`ModelWeightArenaPlan`]), and the
//! ownership state machine for lazily allocated shadow roots ([`ShadowRootSet`])
//! generic over an exact-capacity device ([`ExactRootDevice`]) so every rule is
//! testable on the CPU against a mock allocator.
//!
//! Layout rules (frozen): members sort by Rust string (UTF-8 bytewise) name;
//! each member offset is aligned up to `alignment` (a power of two); placement is
//! deterministic *next-fit* (a member that does not fit the current root opens a
//! new root and earlier roots are never revisited); a root request is its used
//! extent but never below [`MIN_ROOT_BYTES`] (the allocation minimum the pool
//! already applies); no root exceeds `max_root_bytes` (clamped up to
//! [`MIN_ROOT_BYTES`]). Backing is each root request rounded up to the device
//! allocation granule, once per root.

use std::collections::HashMap;
use std::fmt::{self, Display};
use std::sync::Arc;

/// Smallest root request: the pool's existing minimum allocation.
pub const MIN_ROOT_BYTES: usize = 256;

/// One named member of a [`NamedByteArenaPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedByteArenaSpan {
    pub name: String,
    /// Index into [`NamedByteArenaPlan::roots`].
    pub root: usize,
    /// Byte offset inside that root (a multiple of the plan alignment).
    pub offset: usize,
    pub bytes: usize,
}

/// Deterministic next-fit layout of named byte members into bounded roots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedByteArenaPlan {
    /// Exact request, in bytes, of every root (each `>= MIN_ROOT_BYTES`).
    pub roots: Vec<usize>,
    /// Members sorted by name (UTF-8 bytewise), unique.
    pub members: Vec<NamedByteArenaSpan>,
    pub alignment: usize,
    /// Effective per-root cap (the requested cap clamped up to `MIN_ROOT_BYTES`).
    pub max_root_bytes: usize,
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    // `alignment` is a validated power of two.
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
}

/// `bytes` rounded up to a multiple of `granule`, checked.
pub fn round_up_to_granule(bytes: usize, granule: usize) -> Result<u64, String> {
    if granule == 0 {
        return Err("allocation granule must be non-zero".to_string());
    }
    let g = granule as u64;
    (bytes as u64)
        .div_ceil(g)
        .checked_mul(g)
        .ok_or_else(|| format!("{bytes} bytes rounded up to granule {granule} overflows u64"))
}

fn check_alignment(alignment: usize) -> Result<(), String> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(format!(
            "arena alignment {alignment} must be a non-zero power of two"
        ));
    }
    Ok(())
}

impl NamedByteArenaPlan {
    pub fn new(
        mut members: Vec<(String, usize)>,
        alignment: usize,
        max_root_bytes: usize,
    ) -> Result<Self, String> {
        check_alignment(alignment)?;
        let cap = max_root_bytes.max(MIN_ROOT_BYTES);
        members.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        for (name, _) in &members {
            if name.is_empty() {
                return Err("arena member name must not be empty".to_string());
            }
        }
        for pair in members.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(format!("duplicate arena member name {:?}", pair[0].0));
            }
        }

        let mut roots: Vec<usize> = Vec::new();
        let mut spans: Vec<NamedByteArenaSpan> = Vec::with_capacity(members.len());
        // Used extent of the root currently being filled (`None`: no root yet).
        let mut extent: Option<usize> = None;
        for (name, bytes) in members {
            if bytes > cap {
                return Err(format!(
                    "arena member {name:?} of {bytes} bytes exceeds the {cap}-byte root cap"
                ));
            }
            let fitted = match extent {
                Some(used) => {
                    let start = align_up(used, alignment).ok_or_else(|| {
                        format!("arena offset overflow placing member {name:?} after {used} bytes")
                    })?;
                    start
                        .checked_add(bytes)
                        .filter(|&end| end <= cap)
                        .map(|end| (start, end))
                }
                None => None,
            };
            let (start, end) = match fitted {
                Some(placed) => placed,
                None => {
                    if let Some(used) = extent {
                        roots.push(used.max(MIN_ROOT_BYTES));
                    }
                    (0, bytes)
                }
            };
            extent = Some(end);
            spans.push(NamedByteArenaSpan {
                name,
                root: roots.len(),
                offset: start,
                bytes,
            });
        }
        if let Some(used) = extent {
            roots.push(used.max(MIN_ROOT_BYTES));
        }
        Ok(Self {
            roots,
            members: spans,
            alignment,
            max_root_bytes: cap,
        })
    }

    /// Binary search over the name-sorted members.
    pub fn member(&self, name: &str) -> Option<&NamedByteArenaSpan> {
        self.members
            .binary_search_by(|span| span.name.as_bytes().cmp(name.as_bytes()))
            .ok()
            .map(|i| &self.members[i])
    }

    /// Checked sum of every member's payload bytes.
    pub fn logical_bytes(&self) -> Result<u64, String> {
        self.members.iter().try_fold(0u64, |sum, span| {
            sum.checked_add(span.bytes as u64)
                .ok_or_else(|| "arena logical bytes overflow u64".to_string())
        })
    }

    /// Checked sum of each root request rounded UP to `granule`, once per root.
    pub fn backing_bytes(&self, granule: usize) -> Result<u64, String> {
        sum_backing(&self.roots, granule)
    }

    /// Prove actual allocator capacities against the plan: exactly one capacity per
    /// root, in root order, each EQUAL to the root request. An oversized (reused)
    /// buffer is rejected rather than treated as if request == capacity. Returns
    /// the granule-rounded backing of the actual capacities, which is checked equal
    /// to [`Self::backing_bytes`].
    pub fn validate_capacities(&self, actual: &[usize], granule: usize) -> Result<u64, String> {
        if granule == 0 {
            return Err("allocation granule must be non-zero".to_string());
        }
        if actual.len() != self.roots.len() {
            return Err(format!(
                "{} actual root capacities for a {}-root plan",
                actual.len(),
                self.roots.len()
            ));
        }
        for (i, (&want, &got)) in self.roots.iter().zip(actual).enumerate() {
            if got != want {
                return Err(format!(
                    "root {i} capacity {got} differs from its exact request {want}"
                ));
            }
        }
        let backing = sum_backing(actual, granule)?;
        let planned = self.backing_bytes(granule)?;
        if backing != planned {
            return Err(format!(
                "actual backing {backing} differs from planned backing {planned}"
            ));
        }
        Ok(backing)
    }

    /// Input-plan validation: names strictly bytewise-sorted, every range inside its
    /// root and aligned, and the whole layout equal to the canonical next-fit result
    /// for these members, alignment and cap. Not a runtime observer.
    pub fn validate(&self) -> Result<(), String> {
        check_alignment(self.alignment)?;
        if self.max_root_bytes < MIN_ROOT_BYTES {
            return Err(format!(
                "root cap {} is below the {MIN_ROOT_BYTES}-byte minimum",
                self.max_root_bytes
            ));
        }
        for pair in self.members.windows(2) {
            if pair[0].name.as_bytes() >= pair[1].name.as_bytes() {
                return Err(format!(
                    "arena members {:?} and {:?} are not strictly bytewise sorted",
                    pair[0].name, pair[1].name
                ));
            }
        }
        for span in &self.members {
            let Some(&root_bytes) = self.roots.get(span.root) else {
                return Err(format!(
                    "member {:?} names root {} of {}",
                    span.name,
                    span.root,
                    self.roots.len()
                ));
            };
            if span.offset % self.alignment != 0 {
                return Err(format!(
                    "member {:?} offset {} is not {}-byte aligned",
                    span.name, span.offset, self.alignment
                ));
            }
            match span.offset.checked_add(span.bytes) {
                Some(end) if end <= root_bytes => {}
                _ => {
                    return Err(format!(
                        "member {:?} range [{}, +{}) exceeds root {} of {root_bytes} bytes",
                        span.name, span.offset, span.bytes, span.root
                    ))
                }
            }
        }
        for (i, &root_bytes) in self.roots.iter().enumerate() {
            if root_bytes < MIN_ROOT_BYTES || root_bytes > self.max_root_bytes {
                return Err(format!(
                    "root {i} request {root_bytes} outside [{MIN_ROOT_BYTES}, {}]",
                    self.max_root_bytes
                ));
            }
        }
        let canonical = Self::new(
            self.members
                .iter()
                .map(|s| (s.name.clone(), s.bytes))
                .collect(),
            self.alignment,
            self.max_root_bytes,
        )?;
        if canonical != *self {
            return Err("arena layout is not the canonical next-fit layout".to_string());
        }
        Ok(())
    }
}

fn sum_backing(roots: &[usize], granule: usize) -> Result<u64, String> {
    roots.iter().try_fold(0u64, |sum, &root| {
        let rounded = round_up_to_granule(root, granule)?;
        sum.checked_add(rounded)
            .ok_or_else(|| "arena backing bytes overflow u64".to_string())
    })
}

/// The pair of plans one S1 model load shares: constructed once, borrowed by the
/// census and passed unchanged to the weight loader and [`crate::Gpu`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelWeightArenaPlan {
    /// Eligible core weights (actual payload bytes).
    pub weights: NamedByteArenaPlan,
    /// F16 shadows of the BF16 shadow owners (a subset of `weights` names).
    pub f16_shadows: NamedByteArenaPlan,
    /// The device allocation granule the backing was computed with.
    pub granule_bytes: usize,
}

impl ModelWeightArenaPlan {
    pub fn validate(&self) -> Result<(), String> {
        if self.granule_bytes == 0 {
            return Err("allocation granule must be non-zero".to_string());
        }
        self.weights
            .validate()
            .map_err(|e| format!("weight arena: {e}"))?;
        self.f16_shadows
            .validate()
            .map_err(|e| format!("f16 shadow arena: {e}"))?;
        for span in &self.f16_shadows.members {
            if self.weights.member(&span.name).is_none() {
                return Err(format!(
                    "f16 shadow member {:?} has no weight arena member",
                    span.name
                ));
            }
        }
        Ok(())
    }
}

/// A device that hands out roots with EXACT capacities and can quiesce itself.
pub trait ExactRootDevice {
    type Root;
    type Error: Display;
    /// Allocate exactly `bytes` (never from a reuse pool).
    fn alloc_exact(&mut self, bytes: usize) -> Result<Self::Root, Self::Error>;
    /// Whether the root is a unique owning allocation (not a borrowed view).
    fn is_owning(root: &Self::Root) -> bool;
    /// The allocator's actual capacity of `root`.
    fn capacity(root: &Self::Root) -> usize;
    /// Wait until no queued work can still touch any root.
    fn quiesce(&mut self) -> Result<(), Self::Error>;
    /// Release a root. The root is consumed: an error means the device may have
    /// leaked it (it is never retried).
    fn release(&mut self, root: Self::Root) -> Result<(), Self::Error>;
}

/// Failure of an exact allocation.
#[derive(Debug)]
pub enum ExactAllocError<E> {
    /// The device allocator failed (error preserved, e.g. out-of-memory codes).
    Alloc(E),
    /// The device returned a capacity different from the request; the buffer was
    /// released (`release_error` set if that release failed too).
    CapacityMismatch {
        requested: usize,
        actual: usize,
        release_error: Option<String>,
    },
    /// The device returned a non-owning buffer for an owning request.
    NotOwning { requested: usize },
    /// The request does not match the plan (unknown root, span overrun, ...).
    Plan(String),
}

impl<E: Display> Display for ExactAllocError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => write!(f, "{e}"),
            Self::CapacityMismatch {
                requested,
                actual,
                release_error,
            } => {
                write!(
                    f,
                    "exact allocation of {requested} bytes returned capacity {actual}"
                )?;
                if let Some(e) = release_error {
                    write!(f, " (and releasing it failed: {e})")?;
                }
                Ok(())
            }
            Self::NotOwning { requested } => write!(
                f,
                "exact allocation of {requested} bytes returned a non-owning buffer"
            ),
            Self::Plan(msg) => write!(f, "{msg}"),
        }
    }
}

/// Allocate one root of exactly `max(bytes, MIN_ROOT_BYTES)` bytes and validate the
/// actual capacity equals the request. Never accepts an oversized buffer.
pub fn exact_alloc<D: ExactRootDevice>(
    dev: &mut D,
    bytes: usize,
) -> Result<D::Root, ExactAllocError<D::Error>> {
    let requested = bytes.max(MIN_ROOT_BYTES);
    let root = dev.alloc_exact(requested).map_err(ExactAllocError::Alloc)?;
    if !D::is_owning(&root) {
        return Err(ExactAllocError::NotOwning { requested });
    }
    let actual = D::capacity(&root);
    if actual != requested {
        let release_error = dev.release(root).err().map(|e| e.to_string());
        return Err(ExactAllocError::CapacityMismatch {
            requested,
            actual,
            release_error,
        });
    }
    Ok(root)
}

/// Ownership state of the lazily allocated F16-shadow roots.
///
/// Inactive (default) = no S1 arena: every query answers "not planned" and
/// [`Self::release`] touches the device zero times.
pub struct ShadowRootSet<R> {
    plan: Option<Arc<ModelWeightArenaPlan>>,
    source_ptrs: Vec<usize>,
    by_ptr: HashMap<usize, usize>,
    roots: Vec<Option<R>>,
    /// Roots whose release could not be proven safe (failed quiesce). Never freed
    /// until a later quiesce succeeds.
    quarantined: Vec<R>,
}

impl<R> Default for ShadowRootSet<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R> ShadowRootSet<R> {
    pub fn new() -> Self {
        Self {
            plan: None,
            source_ptrs: Vec::new(),
            by_ptr: HashMap::new(),
            roots: Vec::new(),
            quarantined: Vec::new(),
        }
    }

    /// Metadata-only bind: `source_ptrs[i]` is the source weight pointer of
    /// `plan.f16_shadows.members[i]` (sorted order). No device allocation.
    pub fn bind(
        &mut self,
        plan: Arc<ModelWeightArenaPlan>,
        source_ptrs: &[usize],
    ) -> Result<(), String> {
        if self.plan.is_some() {
            return Err("an F16 shadow arena is already installed".to_string());
        }
        if !self.quarantined.is_empty() {
            return Err(format!(
                "{} quarantined F16 shadow roots from a failed release are still held",
                self.quarantined.len()
            ));
        }
        plan.validate()?;
        let members = &plan.f16_shadows.members;
        if source_ptrs.len() != members.len() {
            return Err(format!(
                "{} source pointers for {} F16 shadow members",
                source_ptrs.len(),
                members.len()
            ));
        }
        let mut by_ptr = HashMap::with_capacity(source_ptrs.len());
        for (i, &ptr) in source_ptrs.iter().enumerate() {
            if ptr == 0 {
                return Err(format!(
                    "null source pointer bound to F16 shadow member {:?}",
                    members[i].name
                ));
            }
            if let Some(prev) = by_ptr.insert(ptr, i) {
                return Err(format!(
                    "source pointer 0x{ptr:x} bound to both {:?} and {:?}",
                    members[prev].name, members[i].name
                ));
            }
        }
        self.roots = (0..plan.f16_shadows.roots.len()).map(|_| None).collect();
        self.source_ptrs = source_ptrs.to_vec();
        self.by_ptr = by_ptr;
        self.plan = Some(plan);
        Ok(())
    }

    pub fn is_active(&self) -> bool {
        self.plan.is_some()
    }

    pub fn plan(&self) -> Option<&Arc<ModelWeightArenaPlan>> {
        self.plan.as_ref()
    }

    /// Member index (into `plan.f16_shadows.members`) of a registered source pointer.
    pub fn lookup(&self, source_ptr: usize) -> Option<usize> {
        self.by_ptr.get(&source_ptr).copied()
    }

    pub fn source_ptrs(&self) -> &[usize] {
        &self.source_ptrs
    }

    pub fn span(&self, member: usize) -> Option<&NamedByteArenaSpan> {
        self.plan.as_ref()?.f16_shadows.members.get(member)
    }

    pub fn root(&self, index: usize) -> Option<&R> {
        self.roots.get(index)?.as_ref()
    }

    /// Per planned root (plan order): `None` = lazy root not allocated yet,
    /// `Some(actual capacity)` for an allocated one.
    pub fn root_capacities(&self, capacity: impl Fn(&R) -> usize) -> Vec<Option<usize>> {
        self.roots
            .iter()
            .map(|r| r.as_ref().map(&capacity))
            .collect()
    }

    pub fn live_roots(&self) -> usize {
        self.roots.iter().flatten().count()
    }

    pub fn quarantined_roots(&self) -> usize {
        self.quarantined.len()
    }

    /// Lazily exact-allocate (once) the root that holds `member`, check that
    /// `need_bytes` fits inside the member's reserved span (a prefix is fine, an
    /// overrun never is), and return `(root, span offset)`. A failed allocation
    /// leaves the root unallocated and is the caller's error: a planned source never
    /// falls back to an individual allocation.
    pub fn ensure_member<D: ExactRootDevice<Root = R>>(
        &mut self,
        dev: &mut D,
        member: usize,
        need_bytes: usize,
    ) -> Result<(&R, usize), ExactAllocError<D::Error>> {
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| ExactAllocError::Plan("no F16 shadow arena installed".to_string()))?;
        let span = plan.f16_shadows.members.get(member).ok_or_else(|| {
            ExactAllocError::Plan(format!("F16 shadow member {member} is out of range"))
        })?;
        if need_bytes > span.bytes {
            return Err(ExactAllocError::Plan(format!(
                "F16 shadow {:?} needs {need_bytes} bytes but its arena span reserves {}",
                span.name, span.bytes
            )));
        }
        let (root_index, offset) = (span.root, span.offset);
        let request = *plan.f16_shadows.roots.get(root_index).ok_or_else(|| {
            ExactAllocError::Plan(format!("F16 shadow root {root_index} is out of range"))
        })?;
        if self.roots[root_index].is_none() {
            self.roots[root_index] = Some(exact_alloc(dev, request)?);
        }
        Ok((
            self.roots[root_index].as_ref().expect("just ensured"),
            offset,
        ))
    }

    /// Unbind everything, then quiesce and release every root (live and
    /// quarantined) exactly once. Bindings are cleared FIRST so nothing can
    /// resolve a view into a root that is about to go. If the quiesce fails every
    /// root is quarantined (never freed) and an error is returned; a later call
    /// retries after a successful quiesce. A call with nothing to release touches
    /// the device zero times (idempotent; the non-S1 path). Returns the number of
    /// roots released.
    pub fn release<D: ExactRootDevice<Root = R>>(&mut self, dev: &mut D) -> Result<usize, String> {
        self.plan = None;
        self.source_ptrs.clear();
        self.by_ptr.clear();
        let live: Vec<R> = self.roots.drain(..).flatten().collect();
        self.quarantined.extend(live);
        if self.quarantined.is_empty() {
            return Ok(0);
        }
        if let Err(e) = dev.quiesce() {
            return Err(format!(
                "quiesce before releasing F16 shadow roots failed ({e}); {} roots quarantined, not freed",
                self.quarantined.len()
            ));
        }
        let mut released = 0usize;
        let mut errors: Vec<String> = Vec::new();
        for root in self.quarantined.drain(..) {
            match dev.release(root) {
                Ok(()) => released += 1,
                Err(e) => errors.push(e.to_string()),
            }
        }
        if errors.is_empty() {
            Ok(released)
        } else {
            Err(format!(
                "releasing F16 shadow roots failed ({} of {}): {}",
                errors.len(),
                errors.len() + released,
                errors.join("; ")
            ))
        }
    }
}

/// Clear a shadow cache and its arena in the only safe order: first drop every
/// borrowed view (`is_view`) from `cache` so no view outlives its root, then
/// release the roots (see [`ShadowRootSet::release`]). Owning cache entries are
/// left for the caller's existing free path. Returns the released root count.
pub fn clear_shadow_arena<R, V, D: ExactRootDevice<Root = R>>(
    cache: &mut HashMap<usize, V>,
    is_view: impl Fn(&V) -> bool,
    set: &mut ShadowRootSet<R>,
    dev: &mut D,
) -> Result<usize, String> {
    cache.retain(|_, v| !is_view(v));
    set.release(dev)
}

/// Prepare an in-place rewrite of one arena weight's bytes/format (its pointer
/// stays). With an active arena the device is quiesced ONCE, whether or not the
/// weight has a cached F16 view or any shadow member: only after the fence succeeds
/// is anything mutated. Then the weight's shadow cache entry (view or owning) is
/// removed and returned for the caller to free if owning, and `forget_value_caches`
/// drops the other pointer-keyed value-dependent state. The planned roots and
/// bindings are never touched, so the next conversion for the same source re-fills
/// the same span with no allocation. A failed fence returns the error with the cache,
/// callback, binding and roots all untouched. Without an arena this is a no-op
/// (`Ok(None)`: no fence, no mutation, no callback).
pub fn invalidate_weight_source<R, V, D: ExactRootDevice<Root = R>>(
    cache: &mut HashMap<usize, V>,
    set: &ShadowRootSet<R>,
    dev: &mut D,
    source_ptr: usize,
    forget_value_caches: impl FnOnce(),
) -> Result<Option<Option<V>>, String> {
    if !set.is_active() {
        return Ok(None);
    }
    dev.quiesce().map_err(|e| {
        format!(
            "quiesce before rewriting the arena weight at 0x{source_ptr:x} failed ({e}); \
             cache, binding and root kept"
        )
    })?;
    let removed = cache.remove(&source_ptr);
    forget_value_caches();
    Ok(Some(removed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn plan(members: &[(&str, usize)], alignment: usize, cap: usize) -> NamedByteArenaPlan {
        NamedByteArenaPlan::new(
            members.iter().map(|(n, b)| (n.to_string(), *b)).collect(),
            alignment,
            cap,
        )
        .unwrap()
    }

    #[test]
    fn canonical_order_is_bytewise_and_input_order_independent() {
        let names = ["b", "a", "B", "é", "a_1", "a/1", "a1", "Z"];
        let forward: Vec<(String, usize)> =
            names.iter().map(|n| (n.to_string(), 300usize)).collect();
        let mut reversed = forward.clone();
        reversed.reverse();
        let a = NamedByteArenaPlan::new(forward, 256, 1 << 20).unwrap();
        let b = NamedByteArenaPlan::new(reversed, 256, 1 << 20).unwrap();
        assert_eq!(a, b);
        let order: Vec<&str> = a.members.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(order, ["B", "Z", "a", "a/1", "a1", "a_1", "b", "é"]);
    }

    #[test]
    fn members_are_256_aligned_and_disjoint() {
        let p = plan(&[("a", 1), ("b", 257), ("c", 256), ("d", 3)], 256, 1 << 20);
        let offsets: Vec<usize> = p.members.iter().map(|m| m.offset).collect();
        assert_eq!(offsets, [0, 256, 768, 1024]);
        assert_eq!(p.roots, [1027]);
        for pair in p.members.windows(2) {
            assert!(pair[0].offset + pair[0].bytes <= pair[1].offset);
        }
        p.validate().unwrap();
    }

    #[test]
    fn next_fit_never_revisits_an_earlier_root() {
        // c would fit root 0 (768..1024) under best-fit; next-fit keeps it in root 1.
        let p = plan(&[("a", 768), ("b", 512), ("c", 256)], 256, 1024);
        let roots: Vec<usize> = p.members.iter().map(|m| m.root).collect();
        assert_eq!(roots, [0, 1, 1]);
        assert_eq!(p.member("c").unwrap().offset, 512);
        assert_eq!(p.roots, [768, 768]);
        assert!(p.roots.iter().all(|&r| r <= p.max_root_bytes));
    }

    #[test]
    fn caps_empty_zero_and_minimum_root() {
        let empty = plan(&[], 256, 1024);
        assert!(empty.roots.is_empty() && empty.members.is_empty());
        assert_eq!(empty.backing_bytes(4096).unwrap(), 0);
        assert_eq!(empty.logical_bytes().unwrap(), 0);
        empty.validate().unwrap();

        // Zero-sized members get a real minimum-sized root, no fake pointer.
        let zero = plan(&[("z", 0)], 256, 1);
        assert_eq!(zero.max_root_bytes, MIN_ROOT_BYTES);
        assert_eq!(zero.roots, [MIN_ROOT_BYTES]);
        assert_eq!(zero.member("z").unwrap().offset, 0);
        zero.validate().unwrap();

        // A tiny member still requests the 256-byte minimum.
        let tiny = plan(&[("t", 3)], 256, 1 << 20);
        assert_eq!(tiny.roots, [MIN_ROOT_BYTES]);

        // Exactly the cap fits; one byte over is an error.
        assert_eq!(plan(&[("m", 1024)], 256, 1024).roots, [1024]);
        let err = NamedByteArenaPlan::new(vec![("m".into(), 1025)], 256, 1024).unwrap_err();
        assert!(err.contains("exceeds"), "{err}");

        // Zero members interleave without breaking order or caps.
        let mixed = plan(&[("a", 1024), ("b", 0), ("c", 0)], 256, 1024);
        assert!(mixed.roots.iter().all(|&r| r <= 1024));
        mixed.validate().unwrap();
    }

    #[test]
    fn overflow_is_rejected_not_wrapped() {
        // Alignment overflow placing a member after a near-usize::MAX extent.
        let err = NamedByteArenaPlan::new(
            vec![("a".into(), usize::MAX - 1), ("b".into(), 1)],
            256,
            usize::MAX,
        )
        .unwrap_err();
        assert!(err.contains("overflow"), "{err}");
        // Logical bytes overflow u64.
        let big = NamedByteArenaPlan {
            roots: vec![usize::MAX, usize::MAX],
            members: vec![
                NamedByteArenaSpan {
                    name: "a".into(),
                    root: 0,
                    offset: 0,
                    bytes: usize::MAX,
                },
                NamedByteArenaSpan {
                    name: "b".into(),
                    root: 1,
                    offset: 0,
                    bytes: usize::MAX,
                },
            ],
            alignment: 256,
            max_root_bytes: usize::MAX,
        };
        assert!(big.logical_bytes().is_err());
        assert!(big.backing_bytes(0).is_err());
        assert!(big.backing_bytes(256).is_err(), "rounding up overflows u64");
    }

    #[test]
    fn duplicate_names_empty_names_and_bad_alignment_are_rejected() {
        let dup = NamedByteArenaPlan::new(
            vec![("a".into(), 1), ("b".into(), 1), ("a".into(), 2)],
            256,
            1024,
        )
        .unwrap_err();
        assert!(dup.contains("duplicate"), "{dup}");
        assert!(NamedByteArenaPlan::new(vec![("".into(), 1)], 256, 1024).is_err());
        for bad in [0usize, 3, 100, 255] {
            assert!(
                NamedByteArenaPlan::new(vec![("a".into(), 1)], bad, 1024).is_err(),
                "alignment {bad}"
            );
        }
        assert!(NamedByteArenaPlan::new(vec![("a".into(), 1)], 1, 1024).is_ok());
    }

    #[test]
    fn member_lookup_and_tails() {
        let p = plan(&[("w.b", 10), ("w.a", 20), ("w.c", 30)], 256, 4096);
        assert_eq!(p.member("w.a").unwrap().bytes, 20);
        assert_eq!(p.member("w.c").unwrap().bytes, 30);
        assert!(p.member("w.d").is_none());
        assert!(p.member("").is_none());
        assert_eq!(p.logical_bytes().unwrap(), 60);
        // The last member's tail is not padded into the root request.
        let last = p.members.last().unwrap();
        assert_eq!(p.roots[last.root], last.offset + last.bytes);
    }

    #[test]
    fn validate_rejects_every_kind_of_tampering() {
        let base = plan(&[("a", 600), ("b", 600), ("c", 100)], 256, 1024);
        base.validate().unwrap();

        let mut misaligned = base.clone();
        misaligned.members[2].offset += 1;
        assert!(misaligned.validate().is_err());

        let mut unsorted = base.clone();
        unsorted.members.swap(0, 1);
        assert!(unsorted.validate().is_err());

        let mut past_root = base.clone();
        past_root.roots[0] -= 1;
        assert!(past_root.validate().is_err());

        let mut bad_root = base.clone();
        bad_root.members[0].root = 9;
        assert!(bad_root.validate().is_err());

        let mut over_cap = base.clone();
        over_cap.roots[0] = 2048;
        assert!(over_cap.validate().is_err());

        let mut under_min = base.clone();
        under_min.roots[1] = 8;
        assert!(under_min.validate().is_err());

        // Valid ranges but not the canonical next-fit layout.
        let mut non_canonical = base.clone();
        non_canonical.members[2].root = 0;
        non_canonical.members[2].offset = 768;
        non_canonical.roots = vec![868, 600];
        assert!(non_canonical.validate().is_err());

        let mut bad_alignment = base.clone();
        bad_alignment.alignment = 100;
        assert!(bad_alignment.validate().is_err());

        let mut bad_cap = base;
        bad_cap.max_root_bytes = 8;
        assert!(bad_cap.validate().is_err());
    }

    #[test]
    fn model_plan_validate() {
        let weights = plan(&[("a", 4096), ("b", 8192), ("c", 100)], 256, 1 << 16);
        let shadows = plan(&[("a", 8192), ("c", 200)], 256, 1 << 16);
        let model = ModelWeightArenaPlan {
            weights: weights.clone(),
            f16_shadows: shadows.clone(),
            granule_bytes: 4096,
        };
        model.validate().unwrap();
        assert_eq!(model.clone(), model);

        let mut zero_granule = model.clone();
        zero_granule.granule_bytes = 0;
        assert!(zero_granule.validate().is_err());

        let orphan = ModelWeightArenaPlan {
            weights,
            f16_shadows: plan(&[("zz", 8)], 256, 1 << 16),
            granule_bytes: 4096,
        };
        assert!(orphan.validate().unwrap_err().contains("zz"));

        let mut broken = model;
        broken.f16_shadows.members[0].offset = 1;
        assert!(broken.validate().is_err());
    }

    #[test]
    fn capacity_validation_is_exact_ordered_and_checked() {
        let p = plan(&[("a", 600), ("b", 600), ("c", 100)], 256, 1024);
        assert_eq!(p.roots, [600, 868]);
        let granule = 4096;
        let backing = p.backing_bytes(granule).unwrap();
        assert_eq!(backing, 2 * 4096);
        assert_eq!(
            p.validate_capacities(&[600, 868], granule).unwrap(),
            backing
        );
        // Oversized (e.g. a reused pool buffer) is rejected even if the granule
        // rounding would hide the difference.
        assert!(p.validate_capacities(&[601, 868], granule).is_err());
        assert!(p.validate_capacities(&[600, 4096], granule).is_err());
        // Undersized, wrong order, wrong count, zero granule.
        assert!(p.validate_capacities(&[599, 868], granule).is_err());
        assert!(p.validate_capacities(&[868, 600], granule).is_err());
        assert!(p.validate_capacities(&[600], granule).is_err());
        assert!(p.validate_capacities(&[600, 868, 256], granule).is_err());
        assert!(p.validate_capacities(&[600, 868], 0).is_err());
        // Backing counts each root once, rounded per root (not total rounded).
        assert_eq!(p.backing_bytes(1000).unwrap(), 1000 + 1000);
        assert_eq!(
            plan(&[("a", 600), ("b", 600)], 256, 700)
                .backing_bytes(256)
                .unwrap(),
            2 * 768
        );
    }

    // ── mock exact allocator ──────────────────────────────────────────────

    #[derive(Debug)]
    struct MockRoot {
        addr: usize,
        capacity: usize,
    }

    #[derive(Default)]
    struct MockDevice {
        next: usize,
        live: HashMap<usize, usize>,
        /// Pool-like device: capacity rounded up to a multiple of this.
        oversize_granule: Option<usize>,
        fail_alloc_after: Option<usize>,
        fail_quiesce: bool,
        fail_release: bool,
        mallocs: usize,
        frees: usize,
        quiesces: usize,
        log: Rc<RefCell<Vec<String>>>,
    }

    impl MockDevice {
        fn new() -> Self {
            Self {
                next: 0x10_0000,
                ..Default::default()
            }
        }
    }

    impl ExactRootDevice for MockDevice {
        type Root = MockRoot;
        type Error = String;

        fn alloc_exact(&mut self, bytes: usize) -> Result<MockRoot, String> {
            if self.fail_alloc_after == Some(self.mallocs) {
                return Err("mock hipMalloc out of memory".to_string());
            }
            self.mallocs += 1;
            let capacity = match self.oversize_granule {
                Some(g) => bytes.div_ceil(g) * g,
                None => bytes,
            };
            let addr = self.next;
            self.next += capacity.div_ceil(256) * 256 + 256;
            self.live.insert(addr, capacity);
            Ok(MockRoot { addr, capacity })
        }

        fn is_owning(_: &MockRoot) -> bool {
            true
        }

        fn capacity(root: &MockRoot) -> usize {
            root.capacity
        }

        fn quiesce(&mut self) -> Result<(), String> {
            self.quiesces += 1;
            self.log.borrow_mut().push("quiesce".into());
            if self.fail_quiesce {
                Err("mock sync failure".to_string())
            } else {
                Ok(())
            }
        }

        fn release(&mut self, root: MockRoot) -> Result<(), String> {
            self.frees += 1;
            self.log.borrow_mut().push(format!("release {}", root.addr));
            assert!(
                self.live.remove(&root.addr).is_some(),
                "release of a live mock root exactly once"
            );
            if self.fail_release {
                Err("mock hipFree failure".to_string())
            } else {
                Ok(())
            }
        }
    }

    /// Deterministic pseudo-random sizes (no external RNG).
    fn fixture_members(count: usize) -> Vec<(String, usize)> {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        (0..count)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bytes = 1 + (state >> 33) as usize % 90_000_000;
                (format!("model.layers.{i}.weight"), bytes)
            })
            .collect()
    }

    #[test]
    fn census_backing_equals_actual_allocator_capacity_for_every_root() {
        let granule = 4096;
        let cap = 120_000_000;
        let weights = NamedByteArenaPlan::new(fixture_members(400), 256, cap).unwrap();
        assert!(weights.roots.len() > 3);
        assert!(weights.roots.iter().all(|&r| (256..=cap).contains(&r)));
        let mut dev = MockDevice::new();
        let mut actual = Vec::new();
        let mut roots = Vec::new();
        for &request in &weights.roots {
            let root = exact_alloc(&mut dev, request).unwrap();
            // Per-root proof, not just the total.
            assert_eq!(
                round_up_to_granule(root.capacity, granule).unwrap(),
                round_up_to_granule(request, granule).unwrap()
            );
            actual.push(root.capacity);
            roots.push(root);
        }
        let backing = weights.backing_bytes(granule).unwrap();
        assert_eq!(
            weights.validate_capacities(&actual, granule).unwrap(),
            backing
        );
        let per_root: u64 = roots
            .iter()
            .map(|r| round_up_to_granule(r.capacity, granule).unwrap())
            .sum();
        assert_eq!(per_root, backing);
        assert!(backing >= weights.logical_bytes().unwrap());
        assert_eq!(
            dev.mallocs,
            weights.roots.len(),
            "one exact malloc per root"
        );
    }

    #[test]
    fn oversized_allocator_capacity_is_rejected_and_released() {
        let mut dev = MockDevice::new();
        dev.oversize_granule = Some(1 << 20);
        let err = exact_alloc(&mut dev, 600)
            .err()
            .expect("oversized rejected");
        match err {
            ExactAllocError::CapacityMismatch {
                requested,
                actual,
                release_error,
            } => {
                assert_eq!((requested, actual), (600, 1 << 20));
                assert!(release_error.is_none());
            }
            other => panic!("unexpected {other}"),
        }
        assert!(dev.live.is_empty(), "the oversized buffer was released");
        assert_eq!((dev.mallocs, dev.frees), (1, 1));
        // A failed release of the oversized buffer is reported, not hidden.
        let mut dev = MockDevice::new();
        dev.oversize_granule = Some(512);
        dev.fail_release = true;
        let err = exact_alloc(&mut dev, 600).err().unwrap();
        assert!(err.to_string().contains("releasing it failed"), "{err}");
        // The 256-byte minimum applies to zero and tiny requests.
        let mut dev = MockDevice::new();
        assert_eq!(exact_alloc(&mut dev, 0).unwrap().capacity, MIN_ROOT_BYTES);
        assert_eq!(exact_alloc(&mut dev, 9).unwrap().capacity, MIN_ROOT_BYTES);
    }

    // ── shadow ownership ──────────────────────────────────────────────────

    fn model_plan() -> Arc<ModelWeightArenaPlan> {
        let weights = plan(
            &[("a", 4096), ("b", 8192), ("c", 4096), ("d", 100)],
            256,
            1 << 16,
        );
        // cap 3000: a(2048)+c-part... forces two shadow roots.
        let shadows = plan(&[("a", 2048), ("c", 2048), ("d", 200)], 256, 3000);
        let model = ModelWeightArenaPlan {
            weights,
            f16_shadows: shadows,
            granule_bytes: 4096,
        };
        model.validate().unwrap();
        assert_eq!(model.f16_shadows.roots.len(), 2);
        Arc::new(model)
    }

    #[test]
    fn lazy_root_allocated_once_pointers_stable_prefix_allowed() {
        let model = model_plan();
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        set.bind(model.clone(), &[0x1000, 0x2000, 0x3000]).unwrap();
        assert_eq!(dev.mallocs, 0, "bind is metadata only");
        assert_eq!(set.root_capacities(|r| r.capacity), [None, None]);

        let m = set.lookup(0x2000).unwrap();
        assert_eq!(set.span(m).unwrap().name, "c");
        let (root, offset) = set.ensure_member(&mut dev, m, 1000).unwrap();
        let (addr, off_c) = (root.addr, offset);
        assert_eq!(dev.mallocs, 1);
        // Same root again: no new allocation, same pointer.
        let (root, offset) = set.ensure_member(&mut dev, m, 2048).unwrap();
        assert_eq!((root.addr, offset), (addr, off_c));
        assert_eq!(dev.mallocs, 1);
        // The other member sharing that root (d) does not allocate either.
        let d = set.lookup(0x3000).unwrap();
        let (root, _) = set.ensure_member(&mut dev, d, 200).unwrap();
        assert_eq!(root.addr, addr);
        assert_eq!(dev.mallocs, 1);
        // The other root stays lazy until its member is first used.
        let caps = set.root_capacities(|r| r.capacity);
        assert_eq!(caps.iter().filter(|c| c.is_some()).count(), 1);
        assert_eq!(set.live_roots(), 1);
        let a = set.lookup(0x1000).unwrap();
        set.ensure_member(&mut dev, a, 2048).unwrap();
        assert_eq!(dev.mallocs, 2);
        let (root, _) = set.ensure_member(&mut dev, m, 2048).unwrap();
        assert_eq!(
            root.addr, addr,
            "pointer stays stable after the second root"
        );
        assert_eq!(dev.mallocs, 2);
        let caps: Vec<usize> = set
            .root_capacities(|r| r.capacity)
            .into_iter()
            .map(Option::unwrap)
            .collect();
        assert_eq!(
            model.f16_shadows.validate_capacities(&caps, 4096).unwrap(),
            model.f16_shadows.backing_bytes(4096).unwrap()
        );
    }

    #[test]
    fn span_overrun_and_allocation_failure_are_errors_not_fallbacks() {
        let model = model_plan();
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        set.bind(model, &[0x1000, 0x2000, 0x3000]).unwrap();
        let d = set.lookup(0x3000).unwrap();
        let err = set.ensure_member(&mut dev, d, 201).err().unwrap();
        assert!(err.to_string().contains("reserves"), "{err}");
        assert_eq!(dev.mallocs, 0, "overrun checked before any allocation");

        dev.fail_alloc_after = Some(0);
        let err = set.ensure_member(&mut dev, d, 200).err().unwrap();
        assert!(matches!(err, ExactAllocError::Alloc(_)));
        assert_eq!(set.live_roots(), 0, "failed root stays lazy");
        dev.fail_alloc_after = None;
        set.ensure_member(&mut dev, d, 200).unwrap();
        assert_eq!(set.live_roots(), 1, "a later retry allocates it once");
        assert!(set.ensure_member(&mut dev, 99, 1).is_err());
    }

    #[test]
    fn bind_rejects_bad_bindings_and_double_install() {
        let model = model_plan();
        let mut set = ShadowRootSet::<MockRoot>::new();
        assert!(set.bind(model.clone(), &[1, 2]).is_err(), "count");
        assert!(set.bind(model.clone(), &[1, 1, 3]).is_err(), "duplicate");
        assert!(set.bind(model.clone(), &[1, 0, 3]).is_err(), "null");
        let mut broken = (*model).clone();
        broken.granule_bytes = 0;
        assert!(set.bind(Arc::new(broken), &[1, 2, 3]).is_err(), "plan");
        assert!(!set.is_active());
        set.bind(model.clone(), &[1, 2, 3]).unwrap();
        assert!(set.bind(model, &[1, 2, 3]).is_err(), "conflicting arena");
    }

    #[test]
    fn unknown_sources_and_non_s1_are_legacy_with_no_device_traffic() {
        let model = model_plan();
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        // Never installed (non-S1): nothing planned, release touches nothing.
        assert!(set.lookup(0x1000).is_none());
        assert!(set.plan().is_none());
        assert_eq!(set.release(&mut dev).unwrap(), 0);
        assert_eq!((dev.quiesces, dev.mallocs, dev.frees), (0, 0, 0));
        // Installed: only registered pointers are planned.
        set.bind(model, &[0x1000, 0x2000, 0x3000]).unwrap();
        assert!(set.lookup(0x9999).is_none(), "unknown source is legacy");
        assert!(set.lookup(0x1000).is_some());
    }

    #[test]
    fn release_unbinds_first_quiesces_once_and_double_clear_is_a_no_op() {
        let model = model_plan();
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        set.bind(model.clone(), &[0x1000, 0x2000, 0x3000]).unwrap();
        for ptr in [0x1000usize, 0x3000] {
            let m = set.lookup(ptr).unwrap();
            set.ensure_member(&mut dev, m, 16).unwrap();
        }
        assert_eq!(set.live_roots(), 2);
        assert_eq!(set.release(&mut dev).unwrap(), 2);
        assert!(!set.is_active());
        assert!(set.lookup(0x1000).is_none(), "bindings cleared");
        assert_eq!((dev.quiesces, dev.frees), (1, 2));
        assert!(dev.live.is_empty());
        // Double clear: nothing more is quiesced or freed.
        assert_eq!(set.release(&mut dev).unwrap(), 0);
        assert_eq!((dev.quiesces, dev.frees), (1, 2));
        // The set can be installed again for the next model.
        set.bind(model, &[0x1000, 0x2000, 0x3000]).unwrap();
    }

    #[test]
    fn failed_quiesce_quarantines_and_never_frees_then_retry_releases_once() {
        let model = model_plan();
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        set.bind(model.clone(), &[0x1000, 0x2000, 0x3000]).unwrap();
        let a = set.lookup(0x1000).unwrap();
        let (root, _) = set.ensure_member(&mut dev, a, 16).unwrap();
        let addr = root.addr;
        dev.fail_quiesce = true;
        let err = set.release(&mut dev).unwrap_err();
        assert!(err.contains("quarantined"), "{err}");
        assert_eq!(dev.frees, 0, "never freed behind a failed quiesce");
        assert!(dev.live.contains_key(&addr));
        assert_eq!(set.quarantined_roots(), 1);
        assert!(!set.is_active() && set.lookup(0x1000).is_none());
        // A new install is refused while quarantined roots are held.
        assert!(set
            .bind(model.clone(), &[0x1000, 0x2000, 0x3000])
            .unwrap_err()
            .contains("quarantined"));
        // Second failed attempt keeps (not duplicates) the quarantine.
        assert!(set.release(&mut dev).is_err());
        assert_eq!((set.quarantined_roots(), dev.frees), (1, 0));
        // Once the device quiesces the root is released exactly once.
        dev.fail_quiesce = false;
        assert_eq!(set.release(&mut dev).unwrap(), 1);
        assert_eq!((set.quarantined_roots(), dev.frees), (0, 1));
        assert!(dev.live.is_empty());
        assert_eq!(set.release(&mut dev).unwrap(), 0);
        assert_eq!(dev.frees, 1);
        set.bind(model, &[0x1000, 0x2000, 0x3000]).unwrap();
    }

    #[test]
    fn failed_release_is_reported_and_not_retried() {
        let model = model_plan();
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        set.bind(model, &[0x1000, 0x2000, 0x3000]).unwrap();
        let a = set.lookup(0x1000).unwrap();
        set.ensure_member(&mut dev, a, 16).unwrap();
        dev.fail_release = true;
        let err = set.release(&mut dev).unwrap_err();
        assert!(err.contains("hipFree"), "{err}");
        assert_eq!(set.live_roots() + set.quarantined_roots(), 0);
        assert_eq!(set.release(&mut dev).unwrap(), 0);
        assert_eq!(dev.frees, 1);
    }

    /// A view whose drop is recorded, standing in for a borrowed cache GpuTensor.
    #[derive(Debug)]
    struct LoggedView {
        borrowed: bool,
        name: &'static str,
        log: Rc<RefCell<Vec<String>>>,
    }

    impl Drop for LoggedView {
        fn drop(&mut self) {
            self.log.borrow_mut().push(format!("drop {}", self.name));
        }
    }

    #[test]
    fn views_are_dropped_before_any_root_is_quiesced_or_released() {
        let model = model_plan();
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        let log = dev.log.clone();
        set.bind(model, &[0x1000, 0x2000, 0x3000]).unwrap();
        let mut cache: HashMap<usize, LoggedView> = HashMap::new();
        for (ptr, name) in [(0x1000usize, "view_a"), (0x3000, "view_d")] {
            let m = set.lookup(ptr).unwrap();
            set.ensure_member(&mut dev, m, 16).unwrap();
            cache.insert(
                ptr,
                LoggedView {
                    borrowed: true,
                    name,
                    log: log.clone(),
                },
            );
        }
        // A legacy owning entry (unknown source) is untouched by the arena clear.
        cache.insert(
            0x9000,
            LoggedView {
                borrowed: false,
                name: "legacy",
                log: log.clone(),
            },
        );
        let released = clear_shadow_arena(&mut cache, |v| v.borrowed, &mut set, &mut dev).unwrap();
        assert_eq!(released, 2);
        let events = log.borrow().clone();
        let first_device = events
            .iter()
            .position(|e| e == "quiesce" || e.starts_with("release"))
            .expect("device events");
        assert_eq!(
            events[..first_device]
                .iter()
                .filter(|e| e.starts_with("drop view_"))
                .count(),
            2,
            "both views dropped before the first quiesce/release: {events:?}"
        );
        assert!(!events.iter().any(|e| e == "drop legacy"));
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&0x9000));
        // Repeated clears leave the legacy entry and do no device work.
        let before = events.len();
        assert_eq!(
            clear_shadow_arena(&mut cache, |v| v.borrowed, &mut set, &mut dev).unwrap(),
            0
        );
        assert_eq!(log.borrow().len(), before);
        drop(cache);
    }

    #[test]
    fn views_are_dropped_even_when_the_device_cannot_quiesce() {
        let model = model_plan();
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        let log = dev.log.clone();
        set.bind(model, &[0x1000, 0x2000, 0x3000]).unwrap();
        let m = set.lookup(0x1000).unwrap();
        set.ensure_member(&mut dev, m, 16).unwrap();
        let mut cache: HashMap<usize, LoggedView> = HashMap::new();
        cache.insert(
            0x1000,
            LoggedView {
                borrowed: true,
                name: "view_a",
                log: log.clone(),
            },
        );
        dev.fail_quiesce = true;
        assert!(clear_shadow_arena(&mut cache, |v| v.borrowed, &mut set, &mut dev).is_err());
        assert!(cache.is_empty(), "no dangling view into a quarantined root");
        assert_eq!(dev.frees, 0);
        assert_eq!(set.quarantined_roots(), 1);
    }

    #[test]
    fn fixture_scale_shadow_plan_validates_capacities_per_root() {
        // Model-agnostic: many shadow owners over a bounded cap; every root's
        // exact capacity is proven and no pool-style oversizing is accepted.
        let granule = 65536;
        let members: Vec<(String, usize)> = fixture_members(291)
            .into_iter()
            .map(|(n, b)| (n, b * 2))
            .collect();
        let shadows = NamedByteArenaPlan::new(members.clone(), 256, 675_430_400).unwrap();
        let weights = NamedByteArenaPlan::new(
            members.into_iter().map(|(n, b)| (n, b / 2)).collect(),
            256,
            675_430_400,
        )
        .unwrap();
        let model = Arc::new(ModelWeightArenaPlan {
            weights,
            f16_shadows: shadows,
            granule_bytes: granule,
        });
        model.validate().unwrap();
        let ptrs: Vec<usize> = (0..291).map(|i| 0x4000_0000 + i * 0x1000).collect();
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        set.bind(model.clone(), &ptrs).unwrap();
        for i in 0..ptrs.len() {
            let bytes = model.f16_shadows.members[i].bytes;
            set.ensure_member(&mut dev, i, bytes).unwrap();
        }
        assert_eq!(dev.mallocs, model.f16_shadows.roots.len());
        let caps: Vec<usize> = set
            .root_capacities(|r| r.capacity)
            .into_iter()
            .map(|c| c.expect("every root allocated"))
            .collect();
        assert_eq!(
            model
                .f16_shadows
                .validate_capacities(&caps, granule)
                .unwrap(),
            model.f16_shadows.backing_bytes(granule).unwrap()
        );
        assert_eq!(set.release(&mut dev).unwrap(), caps.len());
        assert!(dev.live.is_empty());
    }

    fn cache_view(
        cache: &mut HashMap<usize, LoggedView>,
        ptr: usize,
        name: &'static str,
        log: &Rc<RefCell<Vec<String>>>,
    ) {
        cache.insert(
            ptr,
            LoggedView {
                borrowed: true,
                name,
                log: log.clone(),
            },
        );
    }

    /// Production helper under test, with the value-cache callback recorded in the
    /// same event log as the device and the view drops.
    fn invalidate(
        cache: &mut HashMap<usize, LoggedView>,
        set: &ShadowRootSet<MockRoot>,
        dev: &mut MockDevice,
        ptr: usize,
    ) -> Result<Option<Option<LoggedView>>, String> {
        let log = dev.log.clone();
        invalidate_weight_source(cache, set, dev, ptr, || {
            log.borrow_mut().push("forget value caches".into())
        })
    }

    #[test]
    fn invalidate_fences_once_before_any_mutation_and_keeps_root_and_binding() {
        let model = model_plan();
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        let log = dev.log.clone();
        set.bind(model, &[0x1000, 0x2000, 0x3000]).unwrap();
        let c = set.lookup(0x2000).unwrap();
        let (root, offset) = set.ensure_member(&mut dev, c, 2048).unwrap();
        let (addr, off) = (root.addr, offset);
        let mut cache: HashMap<usize, LoggedView> = HashMap::new();
        cache_view(&mut cache, 0x2000, "view_c", &log);

        let removed = invalidate(&mut cache, &set, &mut dev, 0x2000).unwrap();
        let removed = removed.expect("active arena fences").expect("view removed");
        assert!(removed.borrowed);
        drop(removed);
        let events = log.borrow().clone();
        assert_eq!(events.iter().filter(|e| *e == "quiesce").count(), 1);
        let quiesce = events.iter().position(|e| e == "quiesce").unwrap();
        let forget = events
            .iter()
            .position(|e| e == "forget value caches")
            .unwrap();
        let drop_view = events.iter().position(|e| e == "drop view_c").unwrap();
        assert!(
            quiesce < forget && quiesce < drop_view,
            "fence first: {events:?}"
        );
        assert!(cache.is_empty());
        // Root, binding and pointer survive; nothing was allocated or freed.
        assert_eq!(set.lookup(0x2000), Some(c));
        assert_eq!(set.live_roots(), 1);
        assert_eq!((dev.mallocs, dev.frees), (1, 0));
        let (root, offset) = set.ensure_member(&mut dev, c, 2048).unwrap();
        assert_eq!((root.addr, offset), (addr, off));
        assert_eq!(dev.mallocs, 1);

        // The arena clear afterwards still releases that one root exactly once.
        assert_eq!(set.release(&mut dev).unwrap(), 1);
        assert_eq!(dev.frees, 1);
    }

    #[test]
    fn refill_by_a_different_conversion_arm_reuses_the_same_span_without_allocating() {
        let model = model_plan();
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        let log = dev.log.clone();
        set.bind(model, &[0x1000, 0x2000, 0x3000]).unwrap();
        let c = set.lookup(0x2000).unwrap();
        let mut cache: HashMap<usize, LoggedView> = HashMap::new();

        // First conversion arm fills the full reserved span, a later arm (changed
        // source format, smaller prefix) refills inside it: same root, same offset.
        let mut seen = Vec::new();
        for need in [2048usize, 1024, 2048, 256] {
            let (root, offset) = set.ensure_member(&mut dev, c, need).unwrap();
            seen.push((root.addr, offset));
            cache_view(&mut cache, 0x2000, "view_c", &log);
            let removed = invalidate(&mut cache, &set, &mut dev, 0x2000).unwrap();
            assert!(removed.unwrap().is_some());
            assert!(cache.is_empty());
        }
        assert!(
            seen.windows(2).all(|w| w[0] == w[1]),
            "pointer stable: {seen:?}"
        );
        assert_eq!((dev.mallocs, dev.frees), (1, 0), "refills never allocate");
        // A refill that would overrun the reserved span is still refused.
        assert!(set.ensure_member(&mut dev, c, 2049).is_err());
        assert_eq!(dev.mallocs, 1);
        assert_eq!(dev.quiesces, 4, "one fence per invalidation");
    }

    #[test]
    fn failed_fence_mutates_nothing_and_returns_err() {
        let model = model_plan();
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        let log = dev.log.clone();
        set.bind(model, &[0x1000, 0x2000, 0x3000]).unwrap();
        let c = set.lookup(0x2000).unwrap();
        set.ensure_member(&mut dev, c, 2048).unwrap();
        let mut cache: HashMap<usize, LoggedView> = HashMap::new();
        cache_view(&mut cache, 0x2000, "view_c", &log);

        dev.fail_quiesce = true;
        let err = invalidate(&mut cache, &set, &mut dev, 0x2000).unwrap_err();
        assert!(err.contains("kept"), "{err}");
        assert!(cache.contains_key(&0x2000), "view kept");
        let events = log.borrow().clone();
        assert!(
            !events
                .iter()
                .any(|e| e.starts_with("drop") || e == "forget value caches"),
            "no mutation behind a failed fence: {events:?}"
        );
        assert_eq!(set.lookup(0x2000), Some(c), "binding kept");
        assert_eq!(set.live_roots(), 1, "root kept");
        assert_eq!((dev.mallocs, dev.frees), (1, 0));

        // Recoverable: once the device quiesces the same view drops.
        dev.fail_quiesce = false;
        let removed = invalidate(&mut cache, &set, &mut dev, 0x2000).unwrap();
        assert!(removed.unwrap().is_some());
        assert!(cache.is_empty());
    }

    #[test]
    fn weight_without_a_shadow_member_or_view_still_fences_and_forgets() {
        let model = model_plan();
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        let log = dev.log.clone();
        let mut cache: HashMap<usize, LoggedView> = HashMap::new();
        set.bind(model, &[0x1000, 0x2000, 0x3000]).unwrap();

        // A weight of the arena that has no F16 shadow member and no cached entry
        // (e.g. a direct BF16 reader): one fence, value caches forgotten.
        let removed = invalidate(&mut cache, &set, &mut dev, 0x7777).unwrap();
        assert_eq!(removed.map(|r| r.is_none()), Some(true));
        assert_eq!(dev.quiesces, 1);
        assert!(log.borrow().iter().any(|e| e == "forget value caches"));
        // A registered shadow member with no view yet behaves the same.
        let removed = invalidate(&mut cache, &set, &mut dev, 0x2000).unwrap();
        assert_eq!(removed.map(|r| r.is_none()), Some(true));
        assert_eq!(dev.quiesces, 2);
        assert_eq!((dev.mallocs, dev.frees), (0, 0));
        assert!(set.lookup(0x2000).is_some(), "binding kept");

        // An owning (legacy, e.g. HFQ4 rocBLAS) shadow under the weight's pointer is
        // stale after an in-place rewrite too: it is handed back for the caller to
        // free through its normal path.
        cache.insert(
            0x7777,
            LoggedView {
                borrowed: false,
                name: "owned",
                log: log.clone(),
            },
        );
        let removed = invalidate(&mut cache, &set, &mut dev, 0x7777).unwrap();
        let owned = removed.unwrap().expect("owning entry returned");
        assert!(!owned.borrowed);
        assert!(cache.is_empty());
        assert_eq!(dev.quiesces, 3);
        assert!(!log.borrow().iter().any(|e| e == "drop owned"));
        std::mem::forget(owned);
    }

    #[test]
    fn no_arena_means_no_fence_no_mutation_no_callback() {
        let mut set = ShadowRootSet::<MockRoot>::new();
        let mut dev = MockDevice::new();
        let log = dev.log.clone();
        let mut cache: HashMap<usize, LoggedView> = HashMap::new();
        cache_view(&mut cache, 0x2000, "view_c", &log);
        let removed = invalidate(&mut cache, &set, &mut dev, 0x2000).unwrap();
        assert!(removed.is_none(), "inactive arena: legacy no-op");
        assert_eq!(cache.len(), 1);
        assert_eq!((dev.quiesces, dev.mallocs, dev.frees), (0, 0, 0));
        assert!(log.borrow().is_empty());
        // After a clear the arena is inactive again.
        set.bind(model_plan(), &[1, 2, 3]).unwrap();
        assert_eq!(set.release(&mut dev).unwrap(), 0);
        log.borrow_mut().clear();
        let removed = invalidate(&mut cache, &set, &mut dev, 0x2000).unwrap();
        assert!(removed.is_none());
        assert_eq!(dev.quiesces, 0);
    }
}
