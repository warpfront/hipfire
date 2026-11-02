//! Emu-only process-local managed-capacity ledger (EMU-DESIGN.md §3).
//!
//! A GPU-free ledger shared by every adapter in the process (HIP, ROCr, ...).
//! One physical device per process: [`install`] is immutable and idempotent for
//! an equal PCI BDF + [`BudgetLimits`], and refuses anything else.
//!
//! Lifecycle of a charge:
//!
//! 1. [`reserve`] atomically adds a *pending* charge, or refuses with an
//!    out-of-memory [`BudgetError`] (see [`BudgetError::is_out_of_memory`]) if
//!    `committed + pending + charge` would exceed a limited dimension.
//! 2. Dropping the [`Reservation`] rolls the pending charge back (raw
//!    allocation failed). [`Reservation::commit`] moves it to *committed* and
//!    returns an [`AllocationToken`].
//! 3. [`release`] subtracts the committed charge. Call it only after the
//!    backing free succeeded. A token dropped without `release` keeps its
//!    charge forever: leaks stay charged.
//!
//! Off-state cost: [`active`] is one atomic load. No lock, map or allocation is
//! touched unless a budget was installed.
//!
//! This module never touches CU masks or any hardware queue state.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Opaque handle for an installed budget. `BudgetId(0)` is never valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BudgetId(pub u64);

/// Where a managed allocation's backing lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocationClass {
    /// Device-local backing (VRAM on discrete, GPU-visible carve of shared RAM on UMA).
    Device,
    /// GTT-pinned host backing (`hipHostMalloc`-style).
    HostPinnedGtt,
    /// Userptr-pinned host backing.
    HostPinnedUserptr,
    /// Explicit host-application allocation admitted by a higher-level owner.
    HostApplication,
}

/// Immutable capacity limits for one installed device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetLimits {
    /// Device capacity cap. On UMA this is checked in addition to GTT/shared limits.
    pub device_bytes: u64,
    /// Aggregate (virtual) GTT cap; `None` = unchecked.
    pub gtt_bytes: Option<u64>,
    /// Managed shared-host cap (UMA); `None` = unchecked.
    pub shared_host_bytes: Option<u64>,
    /// UMA charging rules (device backing also consumes GTT + shared host).
    pub uma: bool,
}

/// Per-dimension byte counters (also used as the charge vector of one allocation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BudgetUsage {
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub gtt_bytes: u64,
    pub shared_host_bytes: u64,
}

/// Point-in-time view of a ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetSnapshot {
    pub limits: BudgetLimits,
    pub committed: BudgetUsage,
    pub pending: BudgetUsage,
}

/// Limited dimension that refused a charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetDimension {
    Device,
    Gtt,
    SharedHost,
}

impl fmt::Display for BudgetDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BudgetDimension::Device => "device",
            BudgetDimension::Gtt => "gtt",
            BudgetDimension::SharedHost => "shared-host",
        })
    }
}

/// Ledger failure. Adapters map [`BudgetError::is_out_of_memory`] to their own
/// out-of-memory error (HIP: `hipErrorOutOfMemory` = 2) so pool drain/retry works.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BudgetError {
    /// `committed + pending + requested` exceeds the limit of `dimension`.
    OutOfMemory {
        dimension: BudgetDimension,
        requested: u64,
        used: u64,
        limit: u64,
    },
    /// Zero-byte charge.
    ZeroBytes,
    /// Arithmetic overflow while computing a charge or total.
    Overflow,
    /// The id is not the installed budget (or nothing is installed).
    UnknownBudget(BudgetId),
    /// Installed budget differs in device (BDF) or limits.
    InstallConflict {
        installed_bdf: String,
        requested_bdf: String,
    },
    /// A different install was requested after managed allocations exist.
    InstallAfterAllocations,
}

impl BudgetError {
    /// True when the refusal must be reported as a normal out-of-memory.
    pub fn is_out_of_memory(&self) -> bool {
        matches!(self, BudgetError::OutOfMemory { .. })
    }
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BudgetError::OutOfMemory {
                dimension,
                requested,
                used,
                limit,
            } => write!(
                f,
                "emu budget out of memory: {dimension} request {requested} B with {used} B \
                 committed+pending exceeds limit {limit} B"
            ),
            BudgetError::ZeroBytes => f.write_str("emu budget: zero-byte allocation refused"),
            BudgetError::Overflow => f.write_str("emu budget: byte count overflow"),
            BudgetError::UnknownBudget(id) => {
                write!(f, "emu budget not initialized: unknown budget id {}", id.0)
            }
            BudgetError::InstallConflict {
                installed_bdf,
                requested_bdf,
            } => write!(
                f,
                "emu budget already installed for device {installed_bdf} with different \
                 device or limits (requested {requested_bdf}); installation is immutable"
            ),
            BudgetError::InstallAfterAllocations => {
                f.write_str("emu budget install refused: managed allocations already exist")
            }
        }
    }
}

impl std::error::Error for BudgetError {}

/// Charge vector of one allocation under `limits` (pure).
///
/// Discrete: `Device` -> device; `HostPinnedGtt` -> host + gtt;
/// `HostPinnedUserptr`/`HostApplication` -> host only.
///
/// UMA: `Device` -> device + gtt + shared; `HostPinnedGtt` -> host + gtt + shared;
/// `HostPinnedUserptr`/`HostApplication` -> host + shared. Nothing is counted
/// twice through GPU/CPU aliases.
pub fn charge_vector(limits: &BudgetLimits, class: AllocationClass, bytes: u64) -> BudgetUsage {
    let mut u = BudgetUsage::default();
    match class {
        AllocationClass::Device => {
            u.device_bytes = bytes;
            if limits.uma {
                u.gtt_bytes = bytes;
                u.shared_host_bytes = bytes;
            }
        }
        AllocationClass::HostPinnedGtt => {
            u.host_bytes = bytes;
            u.gtt_bytes = bytes;
            if limits.uma {
                u.shared_host_bytes = bytes;
            }
        }
        AllocationClass::HostPinnedUserptr | AllocationClass::HostApplication => {
            u.host_bytes = bytes;
            if limits.uma {
                u.shared_host_bytes = bytes;
            }
        }
    }
    u
}

fn add_usage(a: &BudgetUsage, b: &BudgetUsage) -> Option<BudgetUsage> {
    Some(BudgetUsage {
        device_bytes: a.device_bytes.checked_add(b.device_bytes)?,
        host_bytes: a.host_bytes.checked_add(b.host_bytes)?,
        gtt_bytes: a.gtt_bytes.checked_add(b.gtt_bytes)?,
        shared_host_bytes: a.shared_host_bytes.checked_add(b.shared_host_bytes)?,
    })
}

fn sub_usage(a: &mut BudgetUsage, b: &BudgetUsage) {
    // Invariant: a charge is only subtracted after being added; saturate rather
    // than panic (Drop paths must not panic).
    a.device_bytes = a.device_bytes.saturating_sub(b.device_bytes);
    a.host_bytes = a.host_bytes.saturating_sub(b.host_bytes);
    a.gtt_bytes = a.gtt_bytes.saturating_sub(b.gtt_bytes);
    a.shared_host_bytes = a.shared_host_bytes.saturating_sub(b.shared_host_bytes);
}

/// Check one limited dimension. Dimensions with a zero charge are skipped.
fn check_dim(
    dimension: BudgetDimension,
    limit: Option<u64>,
    committed: u64,
    pending: u64,
    add: u64,
) -> Result<(), BudgetError> {
    let Some(limit) = limit else { return Ok(()) };
    if add == 0 {
        return Ok(());
    }
    let used = committed
        .checked_add(pending)
        .ok_or(BudgetError::Overflow)?;
    let total = used.checked_add(add).ok_or(BudgetError::Overflow)?;
    if total > limit {
        return Err(BudgetError::OutOfMemory {
            dimension,
            requested: add,
            used,
            limit,
        });
    }
    Ok(())
}

fn check_charge(
    limits: &BudgetLimits,
    committed: &BudgetUsage,
    pending: &BudgetUsage,
    charge: &BudgetUsage,
) -> Result<(), BudgetError> {
    check_dim(
        BudgetDimension::Device,
        Some(limits.device_bytes),
        committed.device_bytes,
        pending.device_bytes,
        charge.device_bytes,
    )?;
    check_dim(
        BudgetDimension::Gtt,
        limits.gtt_bytes,
        committed.gtt_bytes,
        pending.gtt_bytes,
        charge.gtt_bytes,
    )?;
    check_dim(
        BudgetDimension::SharedHost,
        limits.shared_host_bytes,
        committed.shared_host_bytes,
        pending.shared_host_bytes,
        charge.shared_host_bytes,
    )?;
    // Host dimension is intentionally unlimited; still guard counter overflow.
    committed
        .host_bytes
        .checked_add(pending.host_bytes)
        .and_then(|u| u.checked_add(charge.host_bytes))
        .ok_or(BudgetError::Overflow)?;
    Ok(())
}

#[derive(Debug, Default)]
struct State {
    committed: BudgetUsage,
    pending: BudgetUsage,
}

/// The ledger proper. The process singleton wraps one of these; tests build
/// independent instances with [`Ledger::new_isolated`].
#[derive(Debug)]
pub(crate) struct Ledger {
    id: BudgetId,
    bdf: String,
    limits: BudgetLimits,
    state: Mutex<State>,
}

impl Ledger {
    fn new(id: BudgetId, bdf: &str, limits: BudgetLimits) -> Arc<Ledger> {
        Arc::new(Ledger {
            id,
            bdf: bdf.to_ascii_lowercase(),
            limits,
            state: Mutex::new(State::default()),
        })
    }

    /// Independent ledger that does not touch the global singleton.
    #[cfg(test)]
    pub(crate) fn new_isolated(bdf: &str, limits: BudgetLimits) -> Arc<Ledger> {
        static NEXT: AtomicU64 = AtomicU64::new(1 << 32);
        Ledger::new(BudgetId(NEXT.fetch_add(1, Ordering::Relaxed)), bdf, limits)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // Counters are updated in single critical sections, so a poisoned
        // lock still holds consistent state.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn id(&self) -> BudgetId {
        self.id
    }

    pub(crate) fn reserve(
        self: &Arc<Self>,
        class: AllocationClass,
        bytes: u64,
    ) -> Result<Reservation, BudgetError> {
        if bytes == 0 {
            return Err(BudgetError::ZeroBytes);
        }
        let charge = charge_vector(&self.limits, class, bytes);
        {
            let mut st = self.lock();
            check_charge(&self.limits, &st.committed, &st.pending, &charge)?;
            // check_charge proved the sums fit in u64.
            st.pending = add_usage(&st.pending, &charge).ok_or(BudgetError::Overflow)?;
        }
        Ok(Reservation {
            ledger: Some(Arc::clone(self)),
            class,
            bytes,
            charge,
        })
    }

    pub(crate) fn check_additional(
        &self,
        class: AllocationClass,
        bytes: u64,
    ) -> Result<(), BudgetError> {
        if bytes == 0 {
            return Err(BudgetError::ZeroBytes);
        }
        let charge = charge_vector(&self.limits, class, bytes);
        let st = self.lock();
        check_charge(&self.limits, &st.committed, &st.pending, &charge)
    }

    pub(crate) fn snapshot(&self) -> BudgetSnapshot {
        let st = self.lock();
        BudgetSnapshot {
            limits: self.limits,
            committed: st.committed,
            pending: st.pending,
        }
    }

    fn rollback(&self, charge: &BudgetUsage) {
        sub_usage(&mut self.lock().pending, charge);
    }

    fn commit(
        self: &Arc<Self>,
        class: AllocationClass,
        bytes: u64,
        charge: BudgetUsage,
    ) -> AllocationToken {
        {
            let mut st = self.lock();
            sub_usage(&mut st.pending, &charge);
            // Fits: committed + pending + charge was checked at reserve time and
            // the charge moved from pending to committed.
            st.committed = add_usage(&st.committed, &charge).unwrap_or(st.committed);
        }
        AllocationToken {
            ledger: Arc::clone(self),
            class,
            bytes,
            charge,
        }
    }

    fn release(&self, charge: &BudgetUsage) {
        sub_usage(&mut self.lock().committed, charge);
    }
}

/// Pending charge. Dropping it rolls the charge back; [`Reservation::commit`]
/// converts it into a committed [`AllocationToken`].
#[derive(Debug)]
#[must_use = "dropping a Reservation rolls the pending charge back"]
pub struct Reservation {
    /// `None` once committed (Drop then does nothing).
    ledger: Option<Arc<Ledger>>,
    class: AllocationClass,
    bytes: u64,
    charge: BudgetUsage,
}

impl Reservation {
    pub fn class(&self) -> AllocationClass {
        self.class
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Move the pending charge to committed. Call only after the raw
    /// allocation succeeded. The token does not release on Drop.
    pub fn commit(mut self) -> AllocationToken {
        let ledger = self
            .ledger
            .take()
            .expect("reservation holds its ledger until committed");
        ledger.commit(self.class, self.bytes, self.charge)
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if let Some(ledger) = self.ledger.take() {
            ledger.rollback(&self.charge);
        }
    }
}

/// Committed charge identity. Deliberately not `Clone` and with no `Drop`
/// release: a live or leaked backing keeps its charge until an explicit
/// [`release`] after a successful free.
#[derive(Debug)]
#[must_use = "a token not passed to release() keeps its charge forever"]
pub struct AllocationToken {
    ledger: Arc<Ledger>,
    class: AllocationClass,
    bytes: u64,
    charge: BudgetUsage,
}

impl AllocationToken {
    pub fn budget_id(&self) -> BudgetId {
        self.ledger.id()
    }

    pub fn class(&self) -> AllocationClass {
        self.class
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

// ---- process singleton ----------------------------------------------------

/// Id of the globally installed budget; 0 = nothing installed.
static ACTIVE: AtomicU64 = AtomicU64::new(0);
static GLOBAL: OnceLock<Arc<Ledger>> = OnceLock::new();
static INSTALL_LOCK: Mutex<()> = Mutex::new(());
const GLOBAL_ID: BudgetId = BudgetId(1);

/// Install the process budget. Immutable and idempotent for an equal BDF
/// (ASCII case-insensitive) + limits; a different device or limits is refused,
/// as is a different install after managed allocations exist.
pub fn install(pci_bdf: &str, limits: BudgetLimits) -> Result<BudgetId, BudgetError> {
    let _g = INSTALL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(l) = GLOBAL.get() {
        if l.bdf.eq_ignore_ascii_case(pci_bdf) && l.limits == limits {
            return Ok(l.id);
        }
        let st = l.lock();
        if st.committed != BudgetUsage::default() || st.pending != BudgetUsage::default() {
            return Err(BudgetError::InstallAfterAllocations);
        }
        return Err(BudgetError::InstallConflict {
            installed_bdf: l.bdf.clone(),
            requested_bdf: pci_bdf.to_ascii_lowercase(),
        });
    }
    let ledger = Ledger::new(GLOBAL_ID, pci_bdf, limits);
    // Serialized by INSTALL_LOCK, so this cannot lose a race.
    let _ = GLOBAL.set(ledger);
    ACTIVE.store(GLOBAL_ID.0, Ordering::Release);
    Ok(GLOBAL_ID)
}

/// The installed budget, if any. A single atomic load; no lock.
#[inline]
pub fn active() -> Option<BudgetId> {
    match ACTIVE.load(Ordering::Acquire) {
        0 => None,
        n => Some(BudgetId(n)),
    }
}

fn global(id: BudgetId) -> Result<&'static Arc<Ledger>, BudgetError> {
    if id.0 == 0 || ACTIVE.load(Ordering::Acquire) != id.0 {
        return Err(BudgetError::UnknownBudget(id));
    }
    GLOBAL.get().ok_or(BudgetError::UnknownBudget(id))
}

/// Atomically charge `bytes` as pending, or refuse (OOM-like for cap overruns).
pub fn reserve(
    id: BudgetId,
    class: AllocationClass,
    bytes: u64,
) -> Result<Reservation, BudgetError> {
    global(id)?.reserve(class, bytes)
}

/// Subtract a committed charge. Call only after the backing free succeeded.
pub fn release(token: AllocationToken) {
    token.ledger.release(&token.charge);
}

/// Limits + committed + pending. An unknown id yields an all-zero snapshot
/// (ids come only from [`install`], so this is a caller bug, not a runtime state).
pub fn snapshot(id: BudgetId) -> BudgetSnapshot {
    match global(id) {
        Ok(l) => l.snapshot(),
        Err(_) => BudgetSnapshot {
            limits: BudgetLimits {
                device_bytes: 0,
                gtt_bytes: None,
                shared_host_bytes: None,
                uma: false,
            },
            committed: BudgetUsage::default(),
            pending: BudgetUsage::default(),
        },
    }
}

/// Non-mutating preflight: would `reserve` of the same request succeed now?
pub fn check_additional(
    id: BudgetId,
    class: AllocationClass,
    bytes: u64,
) -> Result<(), BudgetError> {
    global(id)?.check_additional(class, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    const BDF: &str = "0000:03:00.0";

    fn discrete(device: u64) -> BudgetLimits {
        BudgetLimits {
            device_bytes: device,
            gtt_bytes: None,
            shared_host_bytes: None,
            uma: false,
        }
    }

    fn uma(device: u64, gtt: Option<u64>, shared: Option<u64>) -> BudgetLimits {
        BudgetLimits {
            device_bytes: device,
            gtt_bytes: gtt,
            shared_host_bytes: shared,
            uma: true,
        }
    }

    #[test]
    fn charge_vectors() {
        let d = discrete(100);
        let u = uma(100, Some(100), Some(100));
        assert_eq!(
            charge_vector(&d, AllocationClass::Device, 7),
            BudgetUsage {
                device_bytes: 7,
                ..Default::default()
            }
        );
        assert_eq!(
            charge_vector(&d, AllocationClass::HostPinnedGtt, 7),
            BudgetUsage {
                host_bytes: 7,
                gtt_bytes: 7,
                ..Default::default()
            }
        );
        assert_eq!(
            charge_vector(&d, AllocationClass::HostPinnedUserptr, 7),
            BudgetUsage {
                host_bytes: 7,
                ..Default::default()
            }
        );
        assert_eq!(
            charge_vector(&d, AllocationClass::HostApplication, 7),
            BudgetUsage {
                host_bytes: 7,
                ..Default::default()
            }
        );
        assert_eq!(
            charge_vector(&u, AllocationClass::Device, 7),
            BudgetUsage {
                device_bytes: 7,
                host_bytes: 0,
                gtt_bytes: 7,
                shared_host_bytes: 7
            }
        );
        assert_eq!(
            charge_vector(&u, AllocationClass::HostPinnedGtt, 7),
            BudgetUsage {
                device_bytes: 0,
                host_bytes: 7,
                gtt_bytes: 7,
                shared_host_bytes: 7
            }
        );
        assert_eq!(
            charge_vector(&u, AllocationClass::HostPinnedUserptr, 7),
            BudgetUsage {
                device_bytes: 0,
                host_bytes: 7,
                gtt_bytes: 0,
                shared_host_bytes: 7
            }
        );
        assert_eq!(
            charge_vector(&u, AllocationClass::HostApplication, 7),
            BudgetUsage {
                device_bytes: 0,
                host_bytes: 7,
                gtt_bytes: 0,
                shared_host_bytes: 7
            }
        );
    }

    #[test]
    fn cap_plus_one_refused_exact_fit_ok() {
        let l = Ledger::new_isolated(BDF, discrete(100));
        let err = l.reserve(AllocationClass::Device, 101).unwrap_err();
        assert!(err.is_out_of_memory());
        assert!(matches!(
            err,
            BudgetError::OutOfMemory {
                dimension: BudgetDimension::Device,
                ..
            }
        ));
        assert_eq!(l.snapshot().pending, BudgetUsage::default());

        let r = l.reserve(AllocationClass::Device, 100).unwrap();
        assert_eq!(l.snapshot().pending.device_bytes, 100);
        // Full: one more byte refused, preflight agrees.
        assert!(l
            .reserve(AllocationClass::Device, 1)
            .unwrap_err()
            .is_out_of_memory());
        assert!(l
            .check_additional(AllocationClass::Device, 1)
            .unwrap_err()
            .is_out_of_memory());
        let _t = r.commit();
        assert!(l
            .reserve(AllocationClass::Device, 1)
            .unwrap_err()
            .is_out_of_memory());
    }

    #[test]
    fn check_additional_does_not_mutate() {
        let l = Ledger::new_isolated(BDF, discrete(100));
        l.check_additional(AllocationClass::Device, 100).unwrap();
        assert_eq!(l.snapshot().pending, BudgetUsage::default());
        assert_eq!(l.snapshot().committed, BudgetUsage::default());
        assert!(l.check_additional(AllocationClass::Device, 101).is_err());
        assert_eq!(
            l.check_additional(AllocationClass::Device, 0),
            Err(BudgetError::ZeroBytes)
        );
    }

    #[test]
    fn zero_and_overflow_refused() {
        let l = Ledger::new_isolated(BDF, discrete(u64::MAX));
        assert_eq!(
            l.reserve(AllocationClass::Device, 0).unwrap_err(),
            BudgetError::ZeroBytes
        );
        let _t = l
            .reserve(AllocationClass::Device, u64::MAX - 1)
            .unwrap()
            .commit();
        // used = MAX-1; +2 overflows u64 and is refused as Overflow, not wrapped.
        assert_eq!(
            l.reserve(AllocationClass::Device, 2).unwrap_err(),
            BudgetError::Overflow
        );
        assert_eq!(l.snapshot().pending, BudgetUsage::default());
        // Host dimension is unlimited but counter overflow still refuses.
        let _h = l
            .reserve(AllocationClass::HostApplication, u64::MAX)
            .unwrap()
            .commit();
        assert_eq!(
            l.reserve(AllocationClass::HostApplication, 1).unwrap_err(),
            BudgetError::Overflow
        );
        assert!(!BudgetError::Overflow.is_out_of_memory());
    }

    #[test]
    fn drop_rolls_back_pending() {
        let l = Ledger::new_isolated(BDF, discrete(100));
        {
            let r = l.reserve(AllocationClass::Device, 60).unwrap();
            assert_eq!(r.bytes(), 60);
            assert_eq!(r.class(), AllocationClass::Device);
            assert_eq!(l.snapshot().pending.device_bytes, 60);
        }
        let s = l.snapshot();
        assert_eq!(s.pending, BudgetUsage::default());
        assert_eq!(s.committed, BudgetUsage::default());
        drop(l.reserve(AllocationClass::Device, 100).unwrap());
    }

    #[test]
    fn commit_then_release() {
        let l = Ledger::new_isolated(BDF, discrete(100));
        let t = l.reserve(AllocationClass::Device, 40).unwrap().commit();
        assert_eq!(t.bytes(), 40);
        assert_eq!(t.class(), AllocationClass::Device);
        assert_eq!(t.budget_id(), l.id());
        let s = l.snapshot();
        assert_eq!(s.committed.device_bytes, 40);
        assert_eq!(s.pending, BudgetUsage::default());
        release(t);
        assert_eq!(l.snapshot().committed, BudgetUsage::default());
    }

    #[test]
    fn release_fn_uses_token_ledger() {
        // The public release() follows the token to its ledger, not the global.
        let l = Ledger::new_isolated(BDF, discrete(100));
        let t = l.reserve(AllocationClass::Device, 25).unwrap().commit();
        release(t);
        assert_eq!(l.snapshot().committed, BudgetUsage::default());
    }

    #[test]
    fn unreleased_token_stays_charged() {
        let l = Ledger::new_isolated(BDF, discrete(100));
        let t = l.reserve(AllocationClass::Device, 100).unwrap().commit();
        drop(t); // leak: no auto-release
        assert_eq!(l.snapshot().committed.device_bytes, 100);
        assert!(l
            .reserve(AllocationClass::Device, 1)
            .unwrap_err()
            .is_out_of_memory());
    }

    #[test]
    fn token_is_send() {
        fn is_send<T: Send>() {}
        is_send::<AllocationToken>();
        is_send::<Reservation>();
    }

    #[test]
    fn discrete_host_classes_do_not_consume_device_cap() {
        let l = Ledger::new_isolated(BDF, discrete(100));
        let _a = l
            .reserve(AllocationClass::HostPinnedGtt, 1_000)
            .unwrap()
            .commit();
        let _b = l
            .reserve(AllocationClass::HostPinnedUserptr, 1_000)
            .unwrap()
            .commit();
        let _c = l
            .reserve(AllocationClass::HostApplication, 1_000)
            .unwrap()
            .commit();
        let s = l.snapshot();
        assert_eq!(s.committed.device_bytes, 0);
        assert_eq!(s.committed.host_bytes, 3_000);
        assert_eq!(s.committed.shared_host_bytes, 0);
        drop(l.reserve(AllocationClass::Device, 100).unwrap());
    }

    #[test]
    fn discrete_gtt_limit_applies_to_gtt_host_only() {
        let l = Ledger::new_isolated(
            BDF,
            BudgetLimits {
                device_bytes: 100,
                gtt_bytes: Some(50),
                shared_host_bytes: None,
                uma: false,
            },
        );
        let t = l
            .reserve(AllocationClass::HostPinnedGtt, 50)
            .unwrap()
            .commit();
        let err = l.reserve(AllocationClass::HostPinnedGtt, 1).unwrap_err();
        assert!(matches!(
            err,
            BudgetError::OutOfMemory {
                dimension: BudgetDimension::Gtt,
                ..
            }
        ));
        // Userptr host and device are unaffected by the GTT cap.
        let _u = l
            .reserve(AllocationClass::HostPinnedUserptr, 10_000)
            .unwrap()
            .commit();
        let _d = l.reserve(AllocationClass::Device, 100).unwrap().commit();
        release(t);
        drop(l.reserve(AllocationClass::HostPinnedGtt, 50).unwrap());
    }

    #[test]
    fn uma_no_double_count_and_independent_dimensions() {
        // device cap 100, gtt cap 60, shared cap 90.
        let l = Ledger::new_isolated(BDF, uma(100, Some(60), Some(90)));

        // One device allocation charges device+gtt+shared exactly once.
        let d = l.reserve(AllocationClass::Device, 30).unwrap().commit();
        let s = l.snapshot();
        assert_eq!(
            s.committed,
            BudgetUsage {
                device_bytes: 30,
                host_bytes: 0,
                gtt_bytes: 30,
                shared_host_bytes: 30
            }
        );

        // GTT dimension hit independently: gtt 30 used of 60.
        let g = l
            .reserve(AllocationClass::HostPinnedGtt, 30)
            .unwrap()
            .commit();
        assert_eq!(l.snapshot().committed.gtt_bytes, 60);
        let e = l.reserve(AllocationClass::Device, 1).unwrap_err();
        assert!(matches!(
            e,
            BudgetError::OutOfMemory {
                dimension: BudgetDimension::Gtt,
                ..
            }
        ));
        let e = l.reserve(AllocationClass::HostPinnedGtt, 1).unwrap_err();
        assert!(matches!(
            e,
            BudgetError::OutOfMemory {
                dimension: BudgetDimension::Gtt,
                ..
            }
        ));

        // Userptr/application never touch device or GTT but do hit shared:
        // shared used = 60 of 90.
        let u = l
            .reserve(AllocationClass::HostPinnedUserptr, 20)
            .unwrap()
            .commit();
        let a = l
            .reserve(AllocationClass::HostApplication, 10)
            .unwrap()
            .commit();
        let s = l.snapshot();
        assert_eq!(s.committed.device_bytes, 30);
        assert_eq!(s.committed.gtt_bytes, 60);
        assert_eq!(s.committed.shared_host_bytes, 90);
        assert_eq!(s.committed.host_bytes, 60);
        let e = l.reserve(AllocationClass::HostApplication, 1).unwrap_err();
        assert!(matches!(
            e,
            BudgetError::OutOfMemory {
                dimension: BudgetDimension::SharedHost,
                ..
            }
        ));
        assert!(l
            .check_additional(AllocationClass::HostPinnedUserptr, 1)
            .is_err());

        // Releasing the GTT-pinned host buffer frees gtt and shared together.
        release(g);
        let s = l.snapshot();
        assert_eq!(
            (s.committed.gtt_bytes, s.committed.shared_host_bytes),
            (30, 60)
        );
        drop(l.reserve(AllocationClass::HostPinnedGtt, 30).unwrap());

        release(d);
        release(u);
        release(a);
    }

    #[test]
    fn uma_shared_limit_hit_without_gtt_limit() {
        let l = Ledger::new_isolated(BDF, uma(1_000, None, Some(40)));
        let _t = l.reserve(AllocationClass::Device, 40).unwrap().commit();
        let e = l.reserve(AllocationClass::Device, 1).unwrap_err();
        assert!(matches!(
            e,
            BudgetError::OutOfMemory {
                dimension: BudgetDimension::SharedHost,
                ..
            }
        ));
        // GTT is counted even when unlimited.
        assert_eq!(l.snapshot().committed.gtt_bytes, 40);
    }

    #[test]
    fn uma_refusal_charges_nothing() {
        // Shared has room, GTT does not: refusal must not leak a partial charge.
        let l = Ledger::new_isolated(BDF, uma(100, Some(10), Some(100)));
        assert!(l.reserve(AllocationClass::Device, 11).is_err());
        let s = l.snapshot();
        assert_eq!(s.pending, BudgetUsage::default());
        assert_eq!(s.committed, BudgetUsage::default());
    }

    #[test]
    fn pending_blocks_concurrent_overcommit() {
        // Every thread holds every reservation it wins; nothing is released, so
        // exactly limit/size reservations can succeed across all threads.
        const SIZE: u64 = 7;
        const SLOTS: u64 = 100;
        let l = Ledger::new_isolated(BDF, discrete(SIZE * SLOTS));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let l = Arc::clone(&l);
                thread::spawn(move || {
                    let mut held = Vec::new();
                    for _ in 0..200 {
                        if let Ok(r) = l.reserve(AllocationClass::Device, SIZE) {
                            held.push(r);
                        }
                        let s = l.snapshot();
                        assert!(
                            s.committed.device_bytes + s.pending.device_bytes
                                <= s.limits.device_bytes
                        );
                    }
                    held
                })
            })
            .collect();
        let held: Vec<Vec<Reservation>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let won: usize = held.iter().map(Vec::len).sum();
        assert_eq!(won as u64, SLOTS);
        assert_eq!(l.snapshot().pending.device_bytes, SIZE * SLOTS);
        drop(held);
        assert_eq!(l.snapshot().pending, BudgetUsage::default());
    }

    #[test]
    fn concurrent_churn_never_exceeds_limit_and_balances() {
        const LIMIT: u64 = 1_000;
        let l = Ledger::new_isolated(BDF, uma(LIMIT, Some(LIMIT), Some(LIMIT)));
        let handles: Vec<_> = (0..8u64)
            .map(|t| {
                let l = Arc::clone(&l);
                thread::spawn(move || {
                    let mut tokens = Vec::new();
                    for i in 0..2_000u64 {
                        let class = match (i + t) % 4 {
                            0 => AllocationClass::Device,
                            1 => AllocationClass::HostPinnedGtt,
                            2 => AllocationClass::HostPinnedUserptr,
                            _ => AllocationClass::HostApplication,
                        };
                        if let Ok(r) = l.reserve(class, 13 + (i % 5)) {
                            let s = l.snapshot();
                            assert!(s.committed.device_bytes + s.pending.device_bytes <= LIMIT);
                            assert!(s.committed.gtt_bytes + s.pending.gtt_bytes <= LIMIT);
                            assert!(
                                s.committed.shared_host_bytes + s.pending.shared_host_bytes
                                    <= LIMIT
                            );
                            match i % 3 {
                                0 => tokens.push(r.commit()),
                                1 => drop(r),
                                _ => release(r.commit()),
                            }
                        }
                        if i % 7 == 0 {
                            if let Some(t) = tokens.pop() {
                                release(t);
                            }
                        }
                    }
                    tokens
                })
            })
            .collect();
        let tokens: Vec<AllocationToken> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        let s = l.snapshot();
        assert_eq!(s.pending, BudgetUsage::default());
        let expect = tokens
            .iter()
            .map(|t| charge_vector(&l.limits, t.class(), t.bytes()))
            .fold(BudgetUsage::default(), |a, b| add_usage(&a, &b).unwrap());
        assert_eq!(s.committed, expect);
        for t in tokens {
            release(t);
        }
        assert_eq!(l.snapshot().committed, BudgetUsage::default());
    }

    /// The only test touching the process singleton; sequenced in one test so
    /// parallel test threads cannot interleave installs.
    #[test]
    fn global_install_idempotent_and_refusals() {
        assert_eq!(active(), None);
        assert_eq!(
            reserve(BudgetId(1), AllocationClass::Device, 1).unwrap_err(),
            BudgetError::UnknownBudget(BudgetId(1))
        );
        assert_eq!(
            check_additional(BudgetId(1), AllocationClass::Device, 1).unwrap_err(),
            BudgetError::UnknownBudget(BudgetId(1))
        );

        let lim = discrete(100);
        let id = install(BDF, lim).unwrap();
        assert_eq!(active(), Some(id));
        // Idempotent for equal BDF (case-insensitive) + limits.
        assert_eq!(install("0000:03:00.0", lim).unwrap(), id);
        assert_eq!(install(&BDF.to_ascii_uppercase(), lim).unwrap(), id);

        // Different device or limits refused while empty.
        assert!(matches!(
            install("0000:04:00.0", lim).unwrap_err(),
            BudgetError::InstallConflict { .. }
        ));
        assert!(matches!(
            install(BDF, discrete(101)).unwrap_err(),
            BudgetError::InstallConflict { .. }
        ));
        assert!(!install(BDF, discrete(101)).unwrap_err().is_out_of_memory());

        // Unknown ids refused.
        assert_eq!(
            reserve(BudgetId(999), AllocationClass::Device, 1).unwrap_err(),
            BudgetError::UnknownBudget(BudgetId(999))
        );
        assert_eq!(
            reserve(BudgetId(0), AllocationClass::Device, 1).unwrap_err(),
            BudgetError::UnknownBudget(BudgetId(0))
        );

        // Free functions round-trip.
        assert!(reserve(id, AllocationClass::Device, 101)
            .unwrap_err()
            .is_out_of_memory());
        check_additional(id, AllocationClass::Device, 100).unwrap();
        let t = reserve(id, AllocationClass::Device, 100).unwrap().commit();
        assert_eq!(t.budget_id(), id);
        assert_eq!(snapshot(id).committed.device_bytes, 100);
        assert!(check_additional(id, AllocationClass::Device, 1).is_err());

        // Equal install stays idempotent after allocations; a different one is
        // refused specifically for existing allocations.
        assert_eq!(install(BDF, lim).unwrap(), id);
        assert_eq!(
            install(BDF, discrete(200)).unwrap_err(),
            BudgetError::InstallAfterAllocations
        );
        assert_eq!(
            install("0000:09:00.0", lim).unwrap_err(),
            BudgetError::InstallAfterAllocations
        );

        release(t);
        assert_eq!(snapshot(id).committed, BudgetUsage::default());
        assert_eq!(snapshot(BudgetId(999)).limits.device_bytes, 0);
    }

    #[test]
    fn error_display() {
        let e = BudgetError::OutOfMemory {
            dimension: BudgetDimension::Gtt,
            requested: 5,
            used: 10,
            limit: 12,
        };
        let s = e.to_string();
        assert!(s.contains("gtt") && s.contains("12"));
        let _: &dyn std::error::Error = &e;
        assert!(BudgetError::UnknownBudget(BudgetId(3))
            .to_string()
            .contains("not initialized"));
    }
}
