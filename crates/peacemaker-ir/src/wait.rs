use crate::{cfg::InstId, effects::{MemClass, OrderType}, reg::RegSet};
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum Counter { Load, Store, Ds, Km, Sample, Bvh, Exp, Vm, Vs, Lgkm }
pub const N: usize = 10;
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CounterSet(pub u16);
impl CounterSet {
    pub fn contains(self, counter: Counter) -> bool { self.0 & (1 << counter as u8) != 0 }
    pub fn insert(&mut self, counter: Counter) { self.0 |= 1 << counter as u8; }
    pub fn union(self, other: Self) -> Self { Self(self.0 | other.0) }
    pub fn intersection(self, other: Self) -> Self { Self(self.0 & other.0) }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WaitImm { pub per_counter: [Option<u8>; N] }
impl Default for WaitImm { fn default() -> Self { Self { per_counter: [None; N] } } }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EventId(pub u64);
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingEvent {
    pub id: EventId, pub inst: InstId, pub counters: CounterSet,
    pub units: [u8; N], pub satisfied: CounterSet,
    /// Lower bound on younger outstanding units on paths where this event is pending.
    /// Saturation at 255 preserves all representable wait thresholds.
    pub younger: [u8; N],
    pub class: MemClass, pub defs: RegSet, pub src_locks: RegSet, pub in_order_type: OrderType,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WaitState { pub pending: Vec<PendingEvent> }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WaitFact { pub wait: InstId, pub satisfies: Vec<EventId> }
