// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Adapter between the HIP allocation wrappers and the emu managed-capacity
//! ledger (`hipfire_config::emu_budget`, EMU-DESIGN.md §3 "HIP adapter
//! obligations").
//!
//! The off state is one `active()` check per wrapper call: no lock, map or
//! allocation is touched unless a profile is installed.
//!
//! Every charge/commit/rollback/release *decision* lives in the pure helpers
//! below, generic over [`Ledger`], so they are testable with a fake ledger and
//! an injected raw allocator result (no GPU):
//!
//! - a budget refusal returns `hipErrorOutOfMemory` and never calls the raw
//!   allocator (so the shipped pool drain/retry still works);
//! - a raw failure drops the reservation (rollback);
//! - a success commits, and the token rides with the allocation owner;
//! - a token is released only after the raw free/release returned success; a
//!   failed free/release keeps the charge.
//!
//! Token storage: `DeviceBuffer` and `PinnedHostBuffer` carry their own
//! `Option<AllocationToken>`. Raw-pointer APIs with no owner object
//! (`host_malloc`/`host_free`, `mem_create`/`mem_release`) keep tokens in
//! process tables keyed by host pointer / physical handle, created lazily and
//! only while a budget is active.

use crate::error::{HipError, HipResult, HIP_ERROR_OUT_OF_MEMORY};
use hipfire_config::emu_budget as ledger;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Adapter-level placement class (mapped onto the ledger's classes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    /// Device-local VRAM backing (`hipMalloc`, device-located `hipMemCreate`).
    Device,
    /// Pinned host backing (`hipHostMalloc`, host-located `hipMemCreate`).
    HostPinned,
}

/// Ledger operations the adapter needs. `commit`/`release` take no `self`:
/// the real ledger's tokens are self-describing.
pub(crate) trait Ledger {
    type Reservation;
    type Token;
    fn reserve(&self, class: Class, bytes: u64) -> Result<Self::Reservation, String>;
    fn check(&self, class: Class, bytes: u64) -> Result<(), String>;
    fn commit(reservation: Self::Reservation) -> Self::Token;
    fn release(token: Self::Token);
}

/// The process ledger installed by `Gpu` initialization.
pub(crate) struct Process {
    id: u64,
}

pub(crate) type Token = ledger::AllocationToken;

/// `Some` iff an emu budget is installed. One cheap check; nothing else runs
/// on the disabled path.
#[inline]
pub(crate) fn active() -> Option<Process> {
    ledger::active().map(|id| Process { id: id.0 })
}

fn to_ledger_class(class: Class) -> ledger::AllocationClass {
    match class {
        Class::Device => ledger::AllocationClass::Device,
        Class::HostPinned => ledger::AllocationClass::HostPinnedGtt,
    }
}

impl Ledger for Process {
    type Reservation = ledger::Reservation;
    type Token = ledger::AllocationToken;

    fn reserve(&self, class: Class, bytes: u64) -> Result<Self::Reservation, String> {
        ledger::reserve(ledger::BudgetId(self.id), to_ledger_class(class), bytes)
            .map_err(|e| e.to_string())
    }

    fn check(&self, class: Class, bytes: u64) -> Result<(), String> {
        ledger::check_additional(ledger::BudgetId(self.id), to_ledger_class(class), bytes)
            .map_err(|e| e.to_string())
    }

    fn commit(reservation: Self::Reservation) -> Self::Token {
        reservation.commit()
    }

    fn release(token: Self::Token) {
        ledger::release(token)
    }
}

/// A cap refusal looks like a normal HIP out-of-memory so the existing pool
/// drain/retry path still runs.
pub(crate) fn oom_error(detail: &str) -> HipError {
    HipError::new(
        HIP_ERROR_OUT_OF_MEMORY,
        &format!("emu capacity budget refused allocation: {detail}"),
    )
}

/// Reserve `bytes`, run `raw`, commit on success. Returns the raw value and
/// the token (`None` for a zero-byte request, which holds no capacity).
///
/// - Refusal: OOM error, `raw` is never called.
/// - `raw` failure: the reservation is dropped (rolled back).
pub(crate) fn charged_alloc<L: Ledger, T>(
    ledger: &L,
    class: Class,
    bytes: usize,
    raw: impl FnOnce() -> HipResult<T>,
) -> HipResult<(T, Option<L::Token>)> {
    if bytes == 0 {
        return Ok((raw()?, None));
    }
    let reservation = ledger
        .reserve(class, bytes as u64)
        .map_err(|msg| oom_error(&msg))?;
    let value = raw()?; // Err: `reservation` drops here => rollback.
    Ok((value, Some(L::commit(reservation))))
}

/// Run the raw free. Release the token only if it succeeded; a failed free
/// keeps the charge (dropping a token never releases).
pub(crate) fn charged_free<L: Ledger>(
    token: Option<L::Token>,
    raw: impl FnOnce() -> HipResult<()>,
) -> HipResult<()> {
    let result = raw();
    if result.is_ok() {
        if let Some(token) = token {
            L::release(token);
        }
    }
    result
}

/// Capacity precheck that reserves nothing (used before irreversible setup).
pub(crate) fn precheck<L: Ledger>(ledger: &L, class: Class, bytes: usize) -> HipResult<()> {
    if bytes == 0 {
        return Ok(());
    }
    ledger
        .check(class, bytes as u64)
        .map_err(|msg| oom_error(&msg))
}

/// Raw imports carry no byte/placement metadata, so they cannot be charged.
pub(crate) fn untyped_import_gate(emu_active: bool) -> HipResult<()> {
    if emu_active {
        Err(HipError::new(
            0,
            "emu capacity budget active: untyped hipMemImportFromShareableHandle \
             (no byte/placement metadata) is refused",
        ))
    } else {
        Ok(())
    }
}

/// Tokens for raw-pointer APIs without an owner object, keyed by address or
/// handle value.
pub(crate) struct TokenTable<T> {
    map: Mutex<HashMap<usize, T>>,
}

impl<T> TokenTable<T> {
    pub(crate) fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<usize, T>> {
        self.map.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn insert(&self, key: usize, token: T) {
        // HIP never re-issues a live key; a stale entry would mean the earlier
        // backing was freed without a recorded release, and its charge is
        // conservatively kept (the old token is dropped, not released).
        let _ = self.lock().insert(key, token);
    }

    pub(crate) fn take(&self, key: usize) -> Option<T> {
        self.lock().remove(&key)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().len()
    }
}

/// [`charged_alloc`] for owner-less backings: the token is stored in `table`
/// under `key_of(&value)`.
pub(crate) fn charged_alloc_keyed<L: Ledger, T>(
    ledger: &L,
    table: &TokenTable<L::Token>,
    class: Class,
    bytes: usize,
    key_of: impl FnOnce(&T) -> usize,
    raw: impl FnOnce() -> HipResult<T>,
) -> HipResult<T> {
    let (value, token) = charged_alloc(ledger, class, bytes, raw)?;
    if let Some(token) = token {
        table.insert(key_of(&value), token);
    }
    Ok(value)
}

/// Release of a keyed backing. The entry is removed first (so a concurrent
/// re-issue of the same key after a successful free cannot be confused with
/// it) and restored if the raw release fails: a failed release keeps the
/// charge. A key without an entry (uncharged or pre-install) is a no-op.
pub(crate) fn release_keyed<L: Ledger>(
    table: &TokenTable<L::Token>,
    key: usize,
    raw: impl FnOnce() -> HipResult<()>,
) -> HipResult<()> {
    let token = table.take(key);
    let result = raw();
    match (result.is_ok(), token) {
        (true, Some(token)) => L::release(token),
        (false, Some(token)) => table.insert(key, token),
        (_, None) => {}
    }
    result
}

/// Placement class of a physical `hipMemCreate` from `prop.location.type_`.
/// Unknown placement refuses.
pub(crate) fn vmm_class(location_type: u32) -> HipResult<Class> {
    match location_type {
        crate::ffi::HIP_MEM_LOCATION_TYPE_DEVICE => Ok(Class::Device),
        crate::ffi::HIP_MEM_LOCATION_TYPE_HOST => Ok(Class::HostPinned),
        other => Err(HipError::new(
            0,
            &format!("emu capacity budget: unknown VMM allocation location type {other}"),
        )),
    }
}

/// Pinned-host tokens for `host_malloc`/`host_free` (keyed by host pointer).
/// Only dereferenced while a budget is active.
pub(crate) static HOST_TOKENS: LazyLock<TokenTable<Token>> = LazyLock::new(TokenTable::new);

/// Physical-handle tokens for `mem_create`/`mem_import_posix`/`mem_release`
/// (keyed by handle value; one token per physical handle). Only dereferenced
/// while a budget is active.
pub(crate) static VMM_TOKENS: LazyLock<TokenTable<Token>> = LazyLock::new(TokenTable::new);

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    thread_local! {
        static USED: Cell<u64> = const { Cell::new(0) };
        static PENDING: Cell<u64> = const { Cell::new(0) };
        static RELEASED: Cell<u64> = const { Cell::new(0) };
        static CLASSES: RefCell<Vec<Class>> = const { RefCell::new(Vec::new()) };
    }

    fn reset() {
        USED.set(0);
        PENDING.set(0);
        RELEASED.set(0);
        CLASSES.with_borrow_mut(Vec::clear);
    }

    struct Fake {
        cap: u64,
    }

    struct FakeRes {
        bytes: u64,
        armed: bool,
    }

    impl Drop for FakeRes {
        fn drop(&mut self) {
            if self.armed {
                PENDING.set(PENDING.get() - self.bytes);
            }
        }
    }

    /// Dropping a token must NOT release (mirrors the real ledger contract).
    struct FakeToken(u64);

    impl Ledger for Fake {
        type Reservation = FakeRes;
        type Token = FakeToken;
        fn reserve(&self, class: Class, bytes: u64) -> Result<FakeRes, String> {
            if USED.get() + PENDING.get() + bytes > self.cap {
                return Err(format!("cap {} exceeded by {bytes}", self.cap));
            }
            CLASSES.with_borrow_mut(|c| c.push(class));
            PENDING.set(PENDING.get() + bytes);
            Ok(FakeRes { bytes, armed: true })
        }
        fn check(&self, _class: Class, bytes: u64) -> Result<(), String> {
            if USED.get() + PENDING.get() + bytes > self.cap {
                Err("cap".to_string())
            } else {
                Ok(())
            }
        }
        fn commit(mut r: FakeRes) -> FakeToken {
            r.armed = false;
            PENDING.set(PENDING.get() - r.bytes);
            USED.set(USED.get() + r.bytes);
            FakeToken(r.bytes)
        }
        fn release(t: FakeToken) {
            USED.set(USED.get() - t.0);
            RELEASED.set(RELEASED.get() + t.0);
        }
    }

    fn raw_err() -> HipError {
        HipError::new(1, "raw failure")
    }

    #[test]
    fn refusal_never_calls_raw_and_looks_like_oom() {
        reset();
        let ledger = Fake { cap: 100 };
        let raw_calls = Cell::new(0);
        let err = charged_alloc(&ledger, Class::Device, 101, || {
            raw_calls.set(raw_calls.get() + 1);
            Ok(())
        })
        .err()
        .expect("must refuse");
        assert_eq!(err.code, HIP_ERROR_OUT_OF_MEMORY);
        assert_eq!(raw_calls.get(), 0);
        assert_eq!((USED.get(), PENDING.get()), (0, 0));
    }

    #[test]
    fn raw_failure_rolls_back_reservation() {
        reset();
        let ledger = Fake { cap: 100 };
        let err = charged_alloc::<_, ()>(&ledger, Class::Device, 60, || Err(raw_err()))
            .err()
            .expect("raw failure propagates");
        assert_eq!(err.code, 1);
        assert_eq!((USED.get(), PENDING.get()), (0, 0));
        // The rolled-back bytes are available again.
        charged_alloc(&ledger, Class::Device, 100, || Ok(())).unwrap();
        assert_eq!(USED.get(), 100);
    }

    #[test]
    fn success_commits_and_successful_free_releases() {
        reset();
        let ledger = Fake { cap: 100 };
        let ((), token) = charged_alloc(&ledger, Class::HostPinned, 40, || Ok(())).unwrap();
        assert_eq!((USED.get(), PENDING.get()), (40, 0));
        CLASSES.with_borrow(|c| assert_eq!(c.as_slice(), [Class::HostPinned]));
        charged_free::<Fake>(token, || Ok(())).unwrap();
        assert_eq!((USED.get(), RELEASED.get()), (0, 40));
    }

    #[test]
    fn free_failure_keeps_charge() {
        reset();
        let ledger = Fake { cap: 100 };
        let ((), token) = charged_alloc(&ledger, Class::Device, 70, || Ok(())).unwrap();
        let err = charged_free::<Fake>(token, || Err(raw_err()))
            .err()
            .expect("free failure propagates");
        assert_eq!(err.code, 1);
        // Token consumed and dropped, but the charge stays.
        assert_eq!((USED.get(), RELEASED.get()), (70, 0));
        assert!(charged_alloc(&ledger, Class::Device, 31, || Ok(())).is_err());
    }

    #[test]
    fn zero_byte_request_holds_no_capacity() {
        reset();
        let ledger = Fake { cap: 0 };
        let ((), token) = charged_alloc(&ledger, Class::Device, 0, || Ok(())).unwrap();
        assert!(token.is_none());
        charged_free::<Fake>(token, || Ok(())).unwrap();
        assert_eq!((USED.get(), RELEASED.get()), (0, 0));
    }

    #[test]
    fn precheck_reserves_nothing() {
        reset();
        let ledger = Fake { cap: 100 };
        precheck(&ledger, Class::Device, 100).unwrap();
        assert_eq!((USED.get(), PENDING.get()), (0, 0));
        assert_eq!(
            precheck(&ledger, Class::Device, 101).err().unwrap().code,
            HIP_ERROR_OUT_OF_MEMORY
        );
    }

    #[test]
    fn vmm_handle_released_only_on_successful_release() {
        reset();
        let ledger = Fake { cap: 100 };
        let table = TokenTable::<FakeToken>::new();
        let handle = 0xabcdusize;
        charged_alloc_keyed(&ledger, &table, Class::Device, 64, |_| handle, || Ok(())).unwrap();
        assert_eq!((USED.get(), table.len()), (64, 1));
        // Unmap never touches the table; only mem_release does. A failed
        // release keeps both the entry and the charge.
        release_keyed::<Fake>(&table, handle, || Err(raw_err())).unwrap_err();
        assert_eq!((USED.get(), table.len(), RELEASED.get()), (64, 1, 0));
        // The retried (pending-release) call succeeds: last release releases.
        release_keyed::<Fake>(&table, handle, || Ok(())).unwrap();
        assert_eq!((USED.get(), table.len(), RELEASED.get()), (0, 0, 64));
        // A second release of an uncharged/unknown handle is a no-op.
        release_keyed::<Fake>(&table, handle, || Ok(())).unwrap();
        assert_eq!(RELEASED.get(), 64);
    }

    #[test]
    fn vmm_create_refusal_and_raw_failure_leave_no_entry() {
        reset();
        let ledger = Fake { cap: 100 };
        let table = TokenTable::<FakeToken>::new();
        charged_alloc_keyed::<_, usize>(&ledger, &table, Class::Device, 101, |h| *h, || Ok(1))
            .unwrap_err();
        charged_alloc_keyed::<_, usize>(
            &ledger,
            &table,
            Class::Device,
            50,
            |h| *h,
            || Err(raw_err()),
        )
        .unwrap_err();
        assert_eq!((USED.get(), PENDING.get(), table.len()), (0, 0, 0));
    }

    #[test]
    fn cow_alias_is_not_double_charged_new_handle_is() {
        reset();
        let ledger = Fake { cap: 200 };
        let table = TokenTable::<FakeToken>::new();
        charged_alloc_keyed(&ledger, &table, Class::Device, 64, |_| 1usize, || Ok(())).unwrap();
        // A fork/COW alias maps the same handle: no mem_create, no charge.
        assert_eq!(USED.get(), 64);
        // A new COW physical handle is its own charge.
        charged_alloc_keyed(&ledger, &table, Class::Device, 64, |_| 2usize, || Ok(())).unwrap();
        assert_eq!((USED.get(), table.len()), (128, 2));
    }

    #[test]
    fn untyped_import_refused_only_when_active() {
        assert!(untyped_import_gate(false).is_ok());
        let err = untyped_import_gate(true).unwrap_err();
        assert!(err.message.contains("untyped"));
    }

    #[test]
    fn vmm_class_follows_location_and_refuses_unknown() {
        assert_eq!(
            vmm_class(crate::ffi::HIP_MEM_LOCATION_TYPE_DEVICE).unwrap(),
            Class::Device
        );
        assert_eq!(
            vmm_class(crate::ffi::HIP_MEM_LOCATION_TYPE_HOST).unwrap(),
            Class::HostPinned
        );
        assert!(vmm_class(0).is_err());
    }
}
