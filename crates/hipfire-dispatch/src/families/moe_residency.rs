//! Explicit mutable placement of immutable routed experts.
//!
//! This capability is separate from RoutedExpertWeights::immutable_identity:
//! canonical expert bytes/views never change. Only the device tables' selected
//! backing may change. The architecture owns copies, LRU and slot tickets.

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

type Result<T> = std::result::Result<T, String>;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Device-visible address and exact available byte extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencySpan {
    pub address: usize,
    pub bytes: usize,
}

impl ResidencySpan {
    fn end(self) -> Result<usize> {
        if self.address == 0 || self.bytes == 0 {
            return Err("mutable residency: empty/null span".into());
        }
        self.address
            .checked_add(self.bytes)
            .ok_or_else(|| "mutable residency: span overflow".into())
    }

    fn overlaps(self, other: Self) -> Result<bool> {
        Ok(self.address < other.end()? && other.address < self.end()?)
    }
}

/// One fixed VRAM slot; the two projections always belong to the same expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencySlot {
    pub gate_up: ResidencySpan,
    pub down: ResidencySpan,
}

/// A copy-manager statement about the completed contents of a slot.
/// This is not an LRU decision or a GPU event. Publication's unsafe caller must
/// prove that this exact generation's bytes have finished becoming visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencySlotTicket {
    pub slot: usize,
    pub expert: usize,
    pub generation: u64,
}

/// The only two admissible sources for one logical expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyLocation {
    Canonical,
    Slot(ResidencySlotTicket),
}

/// Immutable owners, canonical views and table/pool topology.
pub struct MutableExpertResidencySpec {
    pub owner_generation: u64,
    pub canonical_gate_up: Vec<ResidencySpan>,
    pub canonical_down: Vec<ResidencySpan>,
    pub gate_up_table: ResidencySpan,
    pub down_table: ResidencySpan,
    pub slots: Vec<ResidencySlot>,
    /// Retains all canonical, table and pool allocations until the last token,
    /// bound cache or retained tape releases its capability.
    pub keepalive: Arc<dyn Any + Send + Sync>,
}

/// A whole-table transaction. The expected epoch prevents stale host decisions.
pub struct ResidencyPublication {
    pub expected_epoch: u64,
    pub mapping: Vec<ResidencyLocation>,
    /// Newly completed tickets, or repeated identical tickets. Generations may
    /// advance but never go backwards or change expert within one generation.
    pub completed_slots: Vec<ResidencySlotTicket>,
}

/// An immutable published mirror, cheaply borrowed by one decode descriptor.
#[derive(Clone, Debug)]
pub struct ResidencySnapshot {
    pub owner_generation: u64,
    pub epoch: u64,
    pub gate_up: Arc<[usize]>,
    pub down: Arc<[usize]>,
    pub mapping: Arc<[ResidencyLocation]>,
}

struct State {
    published: ResidencySnapshot,
    completed: Vec<Option<ResidencySlotTicket>>,
    active: bool,
    poisoned: bool,
}

struct Inner {
    id: u64,
    spec: MutableExpertResidencySpec,
    fingerprint: String,
    state: Mutex<State>,
}

/// Cloneable authority shared by the loader, copy manager and sealed binding.
/// Equality denotes the same construction, not merely equal numeric addresses.
#[derive(Clone)]
pub struct MutableExpertResidency(Arc<Inner>);

impl fmt::Debug for MutableExpertResidency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MutableExpertResidency")
            .field("id", &self.0.id)
            .field("owner_generation", &self.owner_generation())
            .finish_non_exhaustive()
    }
}
impl PartialEq for MutableExpertResidency {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for MutableExpertResidency {}

impl MutableExpertResidency {
    /// Bind fixed ownership and an initially canonical pair of device tables.
    ///
    /// # Safety
    /// All spans must be real live allocations on the executing device. The
    /// keepalive must retain every canonical/table/slot owner, and all addresses
    /// and extents must stay fixed. Canonical weight bytes must remain immutable.
    /// The two device tables must already contain the canonical addresses, with
    /// all uploads completed/visible. Slots must not alias unrelated allocations.
    /// A stable-address replacement allocation is a new owner generation and must
    /// receive a fresh capability; reusing this capability for it is forbidden.
    pub unsafe fn new(spec: MutableExpertResidencySpec) -> Result<Self> {
        if spec.owner_generation == 0
            || spec.canonical_gate_up.is_empty()
            || spec.canonical_gate_up.len() != spec.canonical_down.len()
        {
            return Err("mutable residency: invalid generation/canonical cardinality".into());
        }
        let bytes = spec
            .canonical_gate_up
            .len()
            .checked_mul(8)
            .ok_or("mutable residency: table extent overflow")?;
        for table in [spec.gate_up_table, spec.down_table] {
            table.end()?;
            if table.bytes != bytes || table.address % 8 != 0 {
                return Err("mutable residency: pointer table extent/alignment".into());
            }
        }
        if spec.gate_up_table.overlaps(spec.down_table)? {
            return Err("mutable residency: overlapping tables".into());
        }
        let canonical: Vec<_> = spec
            .canonical_gate_up
            .iter()
            .chain(&spec.canonical_down)
            .copied()
            .collect();
        for span in &canonical {
            span.end()?;
            if span.address % 4 != 0 {
                return Err("mutable residency: canonical span alignment".into());
            }
            for table in [spec.gate_up_table, spec.down_table] {
                if span.overlaps(table)? {
                    return Err("mutable residency: canonical bytes overlap a table".into());
                }
            }
        }
        let max_gu = spec
            .canonical_gate_up
            .iter()
            .map(|s| s.bytes)
            .max()
            .unwrap();
        let max_dn = spec.canonical_down.iter().map(|s| s.bytes).max().unwrap();
        let mut pool = Vec::with_capacity(spec.slots.len() * 2);
        for slot in &spec.slots {
            if slot.gate_up.bytes < max_gu || slot.down.bytes < max_dn {
                return Err("mutable residency: slot too small for canonical projection".into());
            }
            for span in [slot.gate_up, slot.down] {
                span.end()?;
                if span.address % 4 != 0 {
                    return Err("mutable residency: slot alignment".into());
                }
                for other in canonical
                    .iter()
                    .copied()
                    .chain([spec.gate_up_table, spec.down_table])
                    .chain(pool.iter().copied())
                {
                    if span.overlaps(other)? {
                        return Err(
                            "mutable residency: slot overlaps canonical/table/other slot".into(),
                        );
                    }
                }
                pool.push(span);
            }
        }
        let id = NEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| "mutable residency: identity exhausted")?;
        let published = ResidencySnapshot {
            owner_generation: spec.owner_generation,
            epoch: 0,
            gate_up: spec.canonical_gate_up.iter().map(|s| s.address).collect(),
            down: spec.canonical_down.iter().map(|s| s.address).collect(),
            mapping: vec![ResidencyLocation::Canonical; spec.canonical_gate_up.len()].into(),
        };
        // Deliberately excludes current entries, slot tickets and publication epoch.
        let fingerprint =
            format!(
            "mutable-residency/v1|id={id}|owner={}|gu={:?}|dn={:?}|gut={:?}|dnt={:?}|slots={:?}",
            spec.owner_generation, spec.canonical_gate_up, spec.canonical_down,
            spec.gate_up_table, spec.down_table, spec.slots
        );
        let completed = vec![None; spec.slots.len()];
        Ok(Self(Arc::new(Inner {
            id,
            spec,
            fingerprint,
            state: Mutex::new(State {
                published,
                completed,
                active: false,
                poisoned: false,
            }),
        })))
    }

    pub fn owner_generation(&self) -> u64 {
        self.0.spec.owner_generation
    }
    pub fn stable_fingerprint(&self) -> &str {
        &self.0.fingerprint
    }
    pub fn canonical_gate_up(&self) -> &[ResidencySpan] {
        &self.0.spec.canonical_gate_up
    }
    pub fn canonical_down(&self) -> &[ResidencySpan] {
        &self.0.spec.canonical_down
    }
    pub fn slots(&self) -> &[ResidencySlot] {
        &self.0.spec.slots
    }
    pub fn gate_up_table(&self) -> ResidencySpan {
        self.0.spec.gate_up_table
    }
    pub fn down_table(&self) -> ResidencySpan {
        self.0.spec.down_table
    }
    /// Retains both the authority and its underlying allocation keepalive.
    pub fn keepalive(&self) -> Arc<dyn Any + Send + Sync> {
        self.0.clone()
    }

    pub fn snapshot(&self) -> Result<ResidencySnapshot> {
        let state = self
            .0
            .state
            .lock()
            .map_err(|_| "mutable residency: mutex poisoned")?;
        if state.poisoned {
            return Err("mutable residency: poisoned".into());
        }
        Ok(state.published.clone())
    }

    /// Validate the immutable canonical views and actual table descriptors.
    /// This is also legal at bind time before a token has acquired the tables.
    pub fn validate_binding(
        &self,
        owner_generation: u64,
        canonical_gate_up: &[ResidencySpan],
        canonical_down: &[ResidencySpan],
        gate_up_table: ResidencySpan,
        down_table: ResidencySpan,
    ) -> Result<()> {
        if owner_generation != self.owner_generation() {
            return Err("mutable residency: stale owner generation".into());
        }
        if canonical_gate_up != self.canonical_gate_up() || canonical_down != self.canonical_down()
        {
            return Err("mutable residency: canonical views differ from bound owner".into());
        }
        if gate_up_table != self.gate_up_table() || down_table != self.down_table() {
            return Err("mutable residency: pointer table owner/extent changed".into());
        }
        self.snapshot().map(|_| ())
    }

    /// Acquire one single-token reader before any eager/recorded/retained submit.
    /// The architecture must hold all layer guards through the token's final
    /// proven GPU wait, including error paths. This is not a CPU launch guard.
    pub fn begin_token(&self, expected_epoch: u64) -> Result<ResidencyTokenGuard> {
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| "mutable residency: mutex poisoned")?;
        if state.poisoned || state.active {
            return Err("mutable residency: poisoned or token already active".into());
        }
        if state.published.epoch != expected_epoch {
            return Err("mutable residency: stale token epoch".into());
        }
        let snapshot = state.published.clone();
        state.active = true;
        Ok(ResidencyTokenGuard {
            residency: self.clone(),
            snapshot,
            retired: false,
        })
    }

    /// Check a host mirror during load/binding, including a warm seed already
    /// published while idle. This does not acquire a GPU reader or authorize a
    /// launch: every submit must use `validate_current_entries` under its guard.
    pub fn validate_published_entries(&self, gate_up: &[usize], down: &[usize]) -> Result<()> {
        let state = self
            .0
            .state
            .lock()
            .map_err(|_| "mutable residency: mutex poisoned")?;
        if state.poisoned {
            return Err("mutable residency: poisoned".into());
        }
        if gate_up != state.published.gate_up.as_ref() || down != state.published.down.as_ref() {
            return Err("mutable residency: host entries differ from published mapping".into());
        }
        Ok(())
    }

    /// The host mirrors must be the published mapping protected by an active
    /// reader guard. Canonical get()/tier validation remains the sealer's job.
    pub fn validate_current_entries(&self, gate_up: &[usize], down: &[usize]) -> Result<()> {
        let state = self
            .0
            .state
            .lock()
            .map_err(|_| "mutable residency: mutex poisoned")?;
        if state.poisoned || !state.active {
            return Err("mutable residency: launch requires a live, unpoisoned token guard".into());
        }
        if gate_up != state.published.gate_up.as_ref() || down != state.published.down.as_ref() {
            return Err("mutable residency: host entries differ from published mapping".into());
        }
        Ok(())
    }

    /// Validate, upload and atomically publish one whole mapping transaction.
    /// The callback runs under the same mutex as begin_token: no token can
    /// begin between validation and the actual table upload. The callback must
    /// use its supplied snapshot and must not call back into this capability.
    /// An upload error/panic poisons the authority; partial table writes cannot
    /// be represented by the previous published mirrors.
    ///
    /// # Safety
    /// Before callback entry all prior GPU readers must have retired. Completed
    /// tickets must describe the current bytes of their exact slot generation,
    /// copied without requantization from that logical canonical expert; copies
    /// must be complete and visible. No copy may overwrite a still-readable slot.
    /// The callback must upload BOTH exact supplied tables and prove ordering and
    /// visibility to every subsequent HIP/PM4 reader before returning Ok. It must
    /// not submit a token or free/replace any captured owner. GPU synchronization,
    /// event verification and the copy manager's ticket authority are caller-owned.
    pub unsafe fn publish_after_gpu_wait<F>(
        &self,
        publication: ResidencyPublication,
        upload: F,
    ) -> Result<ResidencySnapshot>
    where
        F: FnOnce(&ResidencySnapshot) -> Result<()>,
    {
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| "mutable residency: mutex poisoned")?;
        if state.poisoned || state.active {
            return Err(
                "mutable residency: publication refused during active/poisoned token".into(),
            );
        }
        if publication.expected_epoch != state.published.epoch {
            return Err("mutable residency: stale publication epoch".into());
        }
        if publication.mapping.len() != self.canonical_gate_up().len() {
            return Err("mutable residency: mapping cardinality".into());
        }
        let epoch = state
            .published
            .epoch
            .checked_add(1)
            .ok_or("mutable residency: publication epoch exhausted")?;
        let mut completed = state.completed.clone();
        let mut changed = vec![false; completed.len()];
        for ticket in publication.completed_slots {
            if ticket.slot >= completed.len()
                || ticket.expert >= publication.mapping.len()
                || ticket.generation == 0
                || changed[ticket.slot]
            {
                return Err("mutable residency: invalid/duplicate completed slot ticket".into());
            }
            changed[ticket.slot] = true;
            if let Some(previous) = completed[ticket.slot] {
                if ticket.generation < previous.generation
                    || (ticket.generation == previous.generation
                        && ticket.expert != previous.expert)
                {
                    return Err(
                        "mutable residency: stale/reassigned completed slot generation".into(),
                    );
                }
            }
            completed[ticket.slot] = Some(ticket);
        }
        let mut used = vec![false; self.slots().len()];
        let mut gate_up = Vec::with_capacity(publication.mapping.len());
        let mut down = Vec::with_capacity(publication.mapping.len());
        for (expert, location) in publication.mapping.iter().copied().enumerate() {
            match location {
                ResidencyLocation::Canonical => {
                    gate_up.push(self.canonical_gate_up()[expert].address);
                    down.push(self.canonical_down()[expert].address);
                }
                ResidencyLocation::Slot(ticket) => {
                    if ticket.expert != expert
                        || ticket.slot >= used.len()
                        || used[ticket.slot]
                        || completed[ticket.slot] != Some(ticket)
                    {
                        return Err(
                            "mutable residency: uncompleted/wrong-expert/aliased slot mapping"
                                .into(),
                        );
                    }
                    used[ticket.slot] = true;
                    let slot = self.slots()[ticket.slot];
                    gate_up.push(slot.gate_up.address);
                    down.push(slot.down.address);
                }
            }
        }
        let snapshot = ResidencySnapshot {
            owner_generation: self.owner_generation(),
            epoch,
            gate_up: gate_up.into(),
            down: down.into(),
            mapping: publication.mapping.into(),
        };
        if let Err(error) = upload(&snapshot) {
            state.poisoned = true;
            return Err(format!(
                "mutable residency: table upload failed; poisoned: {error}"
            ));
        }
        state.published = snapshot.clone();
        state.completed = completed;
        Ok(snapshot)
    }
}

/// An active single-token reader. Drop without a proven GPU retirement poisons
/// the authority permanently; a CPU error cannot silently reopen table writes.
pub struct ResidencyTokenGuard {
    residency: MutableExpertResidency,
    snapshot: ResidencySnapshot,
    retired: bool,
}
impl ResidencyTokenGuard {
    pub fn snapshot(&self) -> &ResidencySnapshot {
        &self.snapshot
    }

    /// # Safety
    /// Every GPU read submitted under this token has completed, or no GPU work
    /// was submitted. A timeout with unknown quiescence cannot retire the guard.
    pub unsafe fn retire_after_gpu_wait(mut self) -> Result<()> {
        let mut state = self
            .residency
            .0
            .state
            .lock()
            .map_err(|_| "mutable residency: mutex poisoned")?;
        if state.poisoned || !state.active || state.published.epoch != self.snapshot.epoch {
            return Err("mutable residency: invalid token retirement".into());
        }
        state.active = false;
        self.retired = true;
        Ok(())
    }
}
impl Drop for ResidencyTokenGuard {
    fn drop(&mut self) {
        if !self.retired {
            if let Ok(mut state) = self.residency.0.state.lock() {
                state.poisoned = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

    fn span(address: usize, bytes: usize) -> ResidencySpan {
        ResidencySpan { address, bytes }
    }
    fn spec(generation: u64) -> MutableExpertResidencySpec {
        MutableExpertResidencySpec {
            owner_generation: generation,
            canonical_gate_up: vec![span(0x1000, 32), span(0x2000, 32)],
            canonical_down: vec![span(0x3000, 16), span(0x4000, 16)],
            gate_up_table: span(0x5000, 16),
            down_table: span(0x6000, 16),
            slots: vec![ResidencySlot {
                gate_up: span(0x8000, 32),
                down: span(0x9000, 16),
            }],
            keepalive: Arc::new(()),
        }
    }
    fn fixture() -> MutableExpertResidency {
        // SAFETY: CPU metadata-only tests never dereference these fake spans
        // or submit GPU work; keepalive has no real allocation obligations.
        unsafe { MutableExpertResidency::new(spec(7)).unwrap() }
    }
    fn publication(epoch: u64, generation: u64, expert: usize) -> ResidencyPublication {
        let ticket = ResidencySlotTicket {
            slot: 0,
            expert,
            generation,
        };
        let mut mapping = vec![ResidencyLocation::Canonical; 2];
        mapping[expert] = ResidencyLocation::Slot(ticket);
        ResidencyPublication {
            expected_epoch: epoch,
            mapping,
            completed_slots: vec![ticket],
        }
    }
    unsafe fn publish(
        r: &MutableExpertResidency,
        p: ResidencyPublication,
    ) -> Result<ResidencySnapshot> {
        // SAFETY: metadata-only; there are no GPU readers/copies/tables.
        unsafe { r.publish_after_gpu_wait(p, |_| Ok(())) }
    }

    #[test]
    fn publication_preserves_binding_and_retired_token_allows_next_epoch() {
        let r = fixture();
        let identity = r.stable_fingerprint().to_owned();
        let same = r.clone();
        // SAFETY: metadata-only publication, no actual GPU readers.
        let snapshot = unsafe { publish(&r, publication(0, 1, 0)) }.unwrap();
        assert_eq!(&*snapshot.gate_up, &[0x8000, 0x2000]);
        assert_eq!(&*snapshot.down, &[0x9000, 0x4000]);
        let guard = r.begin_token(1).unwrap();
        r.validate_current_entries(&snapshot.gate_up, &snapshot.down)
            .unwrap();
        assert!(r
            .validate_current_entries(&[0xdead, 0x2000], &snapshot.down)
            .is_err());
        // SAFETY: no GPU work was submitted in this metadata-only test.
        unsafe { guard.retire_after_gpu_wait() }.unwrap();
        // SAFETY: metadata-only, no actual GPU writers/readers.
        unsafe { publish(&r, publication(1, 2, 1)) }.unwrap();
        assert_eq!(r, same);
        assert_eq!(r.stable_fingerprint(), identity);
    }

    #[test]
    fn active_token_rejects_publication_before_upload_callback() {
        let r = fixture();
        let guard = r.begin_token(0).unwrap();
        let called = AtomicBool::new(false);
        // SAFETY: rejected before the callback; no GPU work exists.
        let result = unsafe {
            r.publish_after_gpu_wait(publication(0, 1, 0), |_| {
                called.store(true, AtomicOrdering::Relaxed);
                Ok(())
            })
        };
        assert!(result.is_err());
        assert!(!called.load(AtomicOrdering::Relaxed));
        // SAFETY: no GPU work was submitted.
        unsafe { guard.retire_after_gpu_wait() }.unwrap();
    }

    #[test]
    fn idle_binding_checks_do_not_authorize_a_launch() {
        let r = fixture();
        // SAFETY: metadata-only publication, no real GPU allocations.
        let snapshot = unsafe { publish(&r, publication(0, 1, 0)) }.unwrap();
        r.validate_published_entries(&snapshot.gate_up, &snapshot.down)
            .unwrap();
        assert!(r
            .validate_published_entries(&snapshot.gate_up, &[0xdead, 0x4000])
            .is_err());
        assert!(r
            .validate_current_entries(&snapshot.gate_up, &snapshot.down)
            .is_err());
        let guard = r.begin_token(snapshot.epoch).unwrap();
        r.validate_current_entries(&snapshot.gate_up, &snapshot.down)
            .unwrap();
        // SAFETY: no GPU work was submitted.
        unsafe { guard.retire_after_gpu_wait() }.unwrap();
        assert!(r
            .validate_current_entries(&snapshot.gate_up, &snapshot.down)
            .is_err());
    }

    #[test]
    fn retained_capability_keeps_payload_and_table_owners_alive() {
        struct Owner(Arc<AtomicBool>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.store(true, AtomicOrdering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let mut spec = spec(7);
        spec.keepalive = Arc::new(Owner(dropped.clone()));
        // SAFETY: metadata-only, synthetic spans are never dereferenced.
        let r = unsafe { MutableExpertResidency::new(spec) }.unwrap();
        let retained = r.keepalive();
        drop(r);
        assert!(!dropped.load(AtomicOrdering::SeqCst));
        drop(retained);
        assert!(dropped.load(AtomicOrdering::SeqCst));
    }

    #[test]
    fn upload_panic_cannot_reopen_the_old_table_epoch() {
        let r = fixture();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: no actual uploads; simulate a panic after a partial write.
            unsafe {
                r.publish_after_gpu_wait(publication(0, 1, 0), |_| {
                    panic!("second table upload panicked")
                })
            }
        }));
        assert!(result.is_err());
        assert!(r.snapshot().is_err());
        assert!(r.begin_token(0).is_err());
    }

    #[test]
    fn unretired_guard_and_partial_upload_poison() {
        let r = fixture();
        drop(r.begin_token(0).unwrap());
        assert!(r.snapshot().is_err());
        assert!(r.begin_token(0).is_err());
        // SAFETY: metadata-only; poisoned authority refuses before callback.
        assert!(unsafe { publish(&r, publication(0, 1, 0)) }.is_err());
        let r = fixture();
        // SAFETY: metadata-only failure simulation; no GPU memory exists.
        assert!(unsafe {
            r.publish_after_gpu_wait(publication(0, 1, 0), |_| Err("second table failed".into()))
        }
        .is_err());
        assert!(r.snapshot().is_err());
        assert!(r.begin_token(0).is_err());
    }

    #[test]
    fn uncompleted_wrong_expert_and_stale_generations_refuse() {
        let r = fixture();
        let mut p = publication(0, 1, 0);
        p.completed_slots.clear();
        // SAFETY: all tests below only validate fake metadata, no GPU operations.
        assert!(unsafe { publish(&r, p) }.is_err());
        let mut p = publication(0, 1, 0);
        p.completed_slots[0].expert = 1;
        assert!(unsafe { publish(&r, p) }.is_err());
        assert!(unsafe { publish(&r, publication(0, 5, 0)) }.is_ok());
        assert!(unsafe { publish(&r, publication(0, 6, 0)) }.is_err());
        assert!(unsafe { publish(&r, publication(1, 4, 0)) }.is_err());
        assert!(unsafe { publish(&r, publication(1, 5, 1)) }.is_err());
        assert_eq!(r.snapshot().unwrap().epoch, 1);
    }

    #[test]
    fn changed_table_owner_generation_and_overlapping_slots_refuse() {
        let r = fixture();
        let mut table = r.gate_up_table();
        table.address += 8;
        assert!(r
            .validate_binding(
                7,
                r.canonical_gate_up(),
                r.canonical_down(),
                table,
                r.down_table()
            )
            .is_err());
        assert!(r
            .validate_binding(
                8,
                r.canonical_gate_up(),
                r.canonical_down(),
                r.gate_up_table(),
                r.down_table()
            )
            .is_err());
        let mut canonical = r.canonical_gate_up().to_vec();
        canonical[0].address += 4;
        assert!(r
            .validate_binding(
                7,
                &canonical,
                r.canonical_down(),
                r.gate_up_table(),
                r.down_table()
            )
            .is_err());
        let mut invalid = spec(7);
        invalid.slots[0].gate_up = invalid.canonical_gate_up[0];
        // SAFETY: fake ranges are never dereferenced; constructor rejects overlap.
        assert!(unsafe { MutableExpertResidency::new(invalid) }.is_err());
        let other = fixture();
        assert_ne!(r, other);
        assert_ne!(r.stable_fingerprint(), other.stable_fingerprint());
    }
}
