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

/// The reader lease of the single token currently executing against the
/// published image: the owner generation it reads and the image epoch it was
/// admitted at. Publication is refused while a lease is live, so a lease whose
/// epoch differs from the published epoch is never a valid launch authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyLease {
    pub owner_generation: u64,
    pub epoch: u64,
}

struct State {
    published: ResidencySnapshot,
    completed: Vec<Option<ResidencySlotTicket>>,
    lease: Option<ResidencyLease>,
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
                lease: None,
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
        if state.poisoned || state.lease.is_some() {
            return Err("mutable residency: poisoned or token already active".into());
        }
        if state.published.epoch != expected_epoch {
            return Err("mutable residency: stale token epoch".into());
        }
        let snapshot = state.published.clone();
        state.lease = Some(ResidencyLease {
            owner_generation: snapshot.owner_generation,
            epoch: snapshot.epoch,
        });
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
        if state.poisoned || state.lease.is_none() {
            return Err("mutable residency: launch requires a live, unpoisoned token guard".into());
        }
        if gate_up != state.published.gate_up.as_ref() || down != state.published.down.as_ref() {
            return Err("mutable residency: host entries differ from published mapping".into());
        }
        Ok(())
    }

    /// Launch authority for a replay that bypasses the sealed call (retained
    /// PM4/AQL tape, recorded-HIP replay): the residency must be unpoisoned,
    /// hold a live reader lease, and that lease must name `owner_generation`
    /// (the generation the program recorded) at the current image epoch. A
    /// stable identity string never substitutes for this check. Returns the
    /// lease the submission runs under.
    pub fn authorize_launch(&self, owner_generation: u64) -> Result<ResidencyLease> {
        let state = self
            .0
            .state
            .lock()
            .map_err(|_| "mutable residency: mutex poisoned")?;
        if state.poisoned {
            return Err("mutable residency launch refused: authority poisoned".into());
        }
        let Some(lease) = state.lease else {
            return Err(
                "mutable residency launch refused: no live generation lease (no token guard held)"
                    .into(),
            );
        };
        if owner_generation != state.published.owner_generation
            || lease.owner_generation != owner_generation
        {
            return Err(format!(
                "mutable residency launch refused: program recorded owner generation \
                 {owner_generation}, lease holds {} of published generation {}",
                lease.owner_generation, state.published.owner_generation
            ));
        }
        if lease.epoch != state.published.epoch {
            return Err(format!(
                "mutable residency launch refused: lease epoch {} is not the current image epoch {}",
                lease.epoch, state.published.epoch
            ));
        }
        Ok(lease)
    }

    /// [`Self::authorize_launch`] for a caller that holds its own lease (a token
    /// guard): that exact lease must be the live one at the current epoch.
    pub fn authorize_lease(&self, lease: ResidencyLease) -> Result<()> {
        let live = self.authorize_launch(lease.owner_generation)?;
        if live != lease {
            return Err(format!(
                "mutable residency launch refused: held lease epoch {} is not the current \
                 image epoch {}",
                lease.epoch, live.epoch
            ));
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
        if state.poisoned || state.lease.is_some() {
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

    /// The lease this guard holds: its generation and the image epoch it was
    /// admitted at.
    pub fn lease(&self) -> ResidencyLease {
        ResidencyLease {
            owner_generation: self.snapshot.owner_generation,
            epoch: self.snapshot.epoch,
        }
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
        if state.poisoned || state.lease.is_none() || state.published.epoch != self.snapshot.epoch {
            return Err("mutable residency: invalid token retirement".into());
        }
        state.lease = None;
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

impl rdna_compute::launch_authority::LaunchAuthority for MutableExpertResidency {
    fn authority_id(&self) -> u64 {
        self.0.id
    }

    fn authorize_launch(&self, owner_generation: u64) -> std::result::Result<(), String> {
        MutableExpertResidency::authorize_launch(self, owner_generation).map(|_| ())
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

#[cfg(test)]
mod launch_authority_tests {
    use super::*;
    use rdna_compute::replay::{ReplayBackendRequest, ReplayController};

    fn span(address: usize, bytes: usize) -> ResidencySpan {
        ResidencySpan { address, bytes }
    }
    fn fixture() -> MutableExpertResidency {
        // SAFETY: metadata-only fake spans, never dereferenced.
        unsafe {
            MutableExpertResidency::new(MutableExpertResidencySpec {
                owner_generation: 7,
                canonical_gate_up: vec![span(0x1000, 32), span(0x2000, 32)],
                canonical_down: vec![span(0x3000, 16), span(0x4000, 16)],
                gate_up_table: span(0x5000, 16),
                down_table: span(0x6000, 16),
                slots: vec![ResidencySlot {
                    gate_up: span(0x8000, 32),
                    down: span(0x9000, 16),
                }],
                keepalive: Arc::new(()),
            })
            .unwrap()
        }
    }
    fn republish(r: &MutableExpertResidency, epoch: u64, generation: u64) {
        let ticket = ResidencySlotTicket {
            slot: 0,
            expert: 0,
            generation,
        };
        let mut mapping = vec![ResidencyLocation::Canonical; 2];
        mapping[0] = ResidencyLocation::Slot(ticket);
        // SAFETY: metadata-only publication; no GPU readers or tables exist.
        unsafe {
            r.publish_after_gpu_wait(
                ResidencyPublication {
                    expected_epoch: epoch,
                    mapping,
                    completed_slots: vec![ticket],
                },
                |_| Ok(()),
            )
        }
        .unwrap();
    }

    #[test]
    fn launch_needs_a_live_lease_of_the_recorded_generation_at_the_current_epoch() {
        let r = fixture();
        let refusal = r.authorize_launch(7).unwrap_err();
        assert!(refusal.contains("no live generation lease"), "{refusal}");
        let guard = r.begin_token(0).unwrap();
        assert_eq!(
            r.authorize_launch(7).unwrap(),
            ResidencyLease {
                owner_generation: 7,
                epoch: 0
            }
        );
        r.authorize_lease(guard.lease()).unwrap();
        let wrong = r.authorize_launch(8).unwrap_err();
        assert!(wrong.contains("owner generation"), "{wrong}");
        // SAFETY: no GPU work was submitted.
        unsafe { guard.retire_after_gpu_wait() }.unwrap();
        assert!(r.authorize_launch(7).is_err());
    }

    #[test]
    fn a_lease_from_an_earlier_epoch_is_not_authority() {
        let r = fixture();
        let guard = r.begin_token(0).unwrap();
        let old = guard.lease();
        // SAFETY: no GPU work was submitted.
        unsafe { guard.retire_after_gpu_wait() }.unwrap();
        republish(&r, 0, 1);
        let guard = r.begin_token(1).unwrap();
        r.authorize_launch(7).unwrap();
        let stale = r.authorize_lease(old).unwrap_err();
        assert!(stale.contains("current image epoch"), "{stale}");
        r.authorize_lease(guard.lease()).unwrap();
        // SAFETY: no GPU work was submitted.
        unsafe { guard.retire_after_gpu_wait() }.unwrap();
    }

    #[test]
    fn a_poisoned_authority_refuses_every_launch() {
        let r = fixture();
        drop(r.begin_token(0).unwrap());
        let refusal = r.authorize_launch(7).unwrap_err();
        assert!(refusal.contains("poisoned"), "{refusal}");
    }

    /// Every retained replay entry of the controller refuses a wrong-epoch /
    /// wrong-generation / missing lease BEFORE it looks at (or submits) the
    /// prepared plan: an unprepared controller reports the authority refusal,
    /// not "no prepared replay", and reports the plan error once authorized.
    #[test]
    fn every_controller_replay_entry_refuses_before_submission() {
        let r = fixture();
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.note_launch_authority(&r, 7).unwrap();
        assert_eq!(controller.launch_authority_count(), 1);

        // No lease: AQL, PM4 (+ checked) and the PM4 profile all refuse.
        // SAFETY: nothing is prepared, so none of these can submit work.
        let aql = unsafe { controller.replay_linear_aql(0) }.unwrap_err();
        assert!(aql.contains("no live generation lease"), "{aql}");
        let pm4 = unsafe { controller.replay_pm4(0) }.unwrap_err();
        assert!(pm4.contains("no live generation lease"), "{pm4}");
        let checked = unsafe { controller.replay_pm4_checked(0) }.unwrap_err();
        assert!(checked.error.contains("no live generation lease"));
        assert_eq!(
            checked.quiescence,
            rdna_compute::replay::ReplayQuiescence::Proven
        );
        let profile = unsafe { controller.replay_pm4_dispatch_profile(0) }.unwrap_err();
        assert!(profile.contains("no live generation lease"), "{profile}");
        assert!(controller
            .authorize_retained_launch()
            .unwrap_err()
            .contains("no live generation lease"));

        // A tape recorded under another generation is never authorized by this
        // owner's lease.
        let guard = r.begin_token(0).unwrap();
        let mut other = ReplayController::new(ReplayBackendRequest::Auto);
        other.note_launch_authority(&r, 6).unwrap();
        let wrong = unsafe { other.replay_linear_aql(0) }.unwrap_err();
        assert!(wrong.contains("owner generation"), "{wrong}");
        let wrong = unsafe { other.replay_pm4(0) }.unwrap_err();
        assert!(wrong.contains("owner generation"), "{wrong}");
        let wrong = unsafe { other.replay_pm4_dispatch_profile(0) }.unwrap_err();
        assert!(wrong.contains("owner generation"), "{wrong}");

        // Authorized: the entries proceed to the plan check.
        controller.authorize_retained_launch().unwrap();
        let aql = unsafe { controller.replay_linear_aql(0) }.unwrap_err();
        assert_eq!(aql, "no prepared AQL replay");
        let pm4 = unsafe { controller.replay_pm4(0) }.unwrap_err();
        assert_eq!(pm4, "no prepared PM4 replay");
        let profile = unsafe { controller.replay_pm4_dispatch_profile(0) }.unwrap_err();
        assert_eq!(profile, "no prepared PM4 replay");
        // SAFETY: no GPU work was submitted.
        unsafe { guard.retire_after_gpu_wait() }.unwrap();

        // After the token retires the same tape is refused again.
        assert!(unsafe { controller.replay_linear_aql(0) }
            .unwrap_err()
            .contains("no live generation lease"));
    }

    #[test]
    fn an_unregistered_controller_keeps_its_behaviour() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        assert_eq!(controller.launch_authority_count(), 0);
        controller.authorize_retained_launch().unwrap();
        // SAFETY: nothing is prepared, so these cannot submit work.
        assert_eq!(
            unsafe { controller.replay_linear_aql(0) }.unwrap_err(),
            "no prepared AQL replay"
        );
        assert_eq!(
            unsafe { controller.replay_pm4(0) }.unwrap_err(),
            "no prepared PM4 replay"
        );
    }

    #[test]
    fn the_authority_set_is_latched_per_tape_and_sealed_with_it() {
        let r = fixture();
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.note_launch_authority(&r, 7).unwrap();
        controller.note_launch_authority(&r, 7).unwrap();
        assert_eq!(controller.launch_authority_count(), 1);
        assert!(controller.note_launch_authority(&r, 8).is_err());
        let other = fixture();
        controller.note_launch_authority(&other, 7).unwrap();
        assert_eq!(controller.launch_authority_count(), 2);
    }
}
