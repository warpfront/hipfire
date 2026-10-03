//! Live device-allocation registry (railgun design §2.4).
//!
//! Every device allocation this crate hands out is recorded here while it is
//! live: `hipMalloc`, `hipExtMallocWithFlags(signal)` and every mapped VMM
//! chunk. railgun's check mode takes its comparison surfaces from this
//! registry, never from a kernel effect list, so a write the effect list does
//! not know about is still compared.
//!
//! Each allocation carries the role that was in force when it was made.
//! Model loaders mark weight uploads [`AllocationRole::Weights`]; checkers
//! mark their own snapshot buffers [`AllocationRole::Internal`]; everything
//! else is [`AllocationRole::State`] (mutable model state: KV, recurrent
//! state, scratch, logits, control words, pool slots). `State` entries are
//! numbered in allocation order so two processes that run the same program
//! can pair their surfaces.
//!
//! The registry records only while railgun check mode or the G2 state digest
//! is configured (`HIPFIRE_RAILGUN_CHECK` other than `off`, or
//! `HIPFIRE_RAILGUN_DIGEST=1`); otherwise every allocation path returns
//! before taking the lock.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{LazyLock, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AllocationRole {
    State,
    Weights,
    Internal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationKind {
    Malloc,
    Signal,
    VmmChunk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveAllocation {
    pub base: usize,
    pub bytes: usize,
    pub kind: AllocationKind,
    pub role: AllocationRole,
    /// Allocation order within the role (per process).
    pub seq: u64,
}

struct Registry {
    live: BTreeMap<usize, LiveAllocation>,
    next_seq: [u64; 3],
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry { live: BTreeMap::new(), next_seq: [0; 3] });
static ROLE: AtomicU8 = AtomicU8::new(0);

fn role_from(v: u8) -> AllocationRole {
    match v {
        1 => AllocationRole::Weights,
        2 => AllocationRole::Internal,
        _ => AllocationRole::State,
    }
}

fn role_index(role: AllocationRole) -> usize {
    match role {
        AllocationRole::State => 0,
        AllocationRole::Weights => 1,
        AllocationRole::Internal => 2,
    }
}

/// Restores the previous role on drop.
pub struct RoleGuard {
    previous: u8,
}

impl Drop for RoleGuard {
    fn drop(&mut self) {
        ROLE.store(self.previous, Ordering::SeqCst);
    }
}

/// Tag every allocation made until the guard drops with `role`
/// (process-wide; loaders and checkers run on the owning thread).
pub fn role_scope(role: AllocationRole) -> RoleGuard {
    let previous = ROLE.swap(role_index(role) as u8, Ordering::SeqCst);
    RoleGuard { previous }
}

static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    let var = |name: &str| hipfire_config::developer_var(name).ok().filter(|v| !v.trim().is_empty());
    var("HIPFIRE_RAILGUN_CHECK").is_some_and(|v| v.trim() != "off")
        || var("HIPFIRE_RAILGUN_DIGEST").as_deref() == Some("1")
});

/// Whether allocations are being recorded (latched at first use).
pub fn enabled() -> bool {
    *ENABLED
}

pub(crate) fn record(base: usize, bytes: usize, kind: AllocationKind) {
    if enabled() {
        insert(base, bytes, kind);
    }
}

pub(crate) fn forget(base: usize) {
    if enabled() {
        remove(base);
    }
}

fn insert(base: usize, bytes: usize, kind: AllocationKind) {
    if base == 0 || bytes == 0 {
        return;
    }
    let role = role_from(ROLE.load(Ordering::SeqCst));
    let mut registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let seq = registry.next_seq[role_index(role)];
    registry.next_seq[role_index(role)] += 1;
    registry.live.insert(base, LiveAllocation { base, bytes, kind, role, seq });
}

fn remove(base: usize) {
    let mut registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    registry.live.remove(&base);
}

/// Every live allocation, in address order.
pub fn live_allocations() -> Vec<LiveAllocation> {
    let registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    registry.live.values().copied().collect()
}

/// Live allocations of one role, in allocation order.
pub fn live_allocations_with_role(role: AllocationRole) -> Vec<LiveAllocation> {
    let mut v: Vec<_> = live_allocations().into_iter().filter(|a| a.role == role).collect();
    v.sort_by_key(|a| a.seq);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_scope_and_pairing_order() {
        let _g = role_scope(AllocationRole::Weights);
        insert(0x7000_0000, 64, AllocationKind::Malloc);
        {
            let _i = role_scope(AllocationRole::Internal);
            insert(0x7000_1000, 64, AllocationKind::Malloc);
        }
        insert(0x7000_2000, 64, AllocationKind::Malloc);
        let weights: Vec<_> = live_allocations_with_role(AllocationRole::Weights)
            .into_iter()
            .filter(|a| a.base >= 0x7000_0000 && a.base < 0x7000_3000)
            .collect();
        assert_eq!(weights.iter().map(|a| a.base).collect::<Vec<_>>(), vec![0x7000_0000, 0x7000_2000]);
        assert!(weights[0].seq < weights[1].seq);
        assert!(live_allocations().iter().any(|a| a.base == 0x7000_1000 && a.role == AllocationRole::Internal));
        for base in [0x7000_0000, 0x7000_1000, 0x7000_2000] {
            remove(base);
        }
        assert!(!live_allocations().iter().any(|a| (0x7000_0000..0x7000_3000).contains(&a.base)));
    }
}
