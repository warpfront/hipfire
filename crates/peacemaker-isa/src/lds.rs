// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SlotState {
    Free,
    Publishing,
    Published,
    Reading,
}
/// The per-wave layout of a wave-private slot: wave `w` of the workgroup owns
/// `[base + w * stride, base + (w + 1) * stride)` and no other wave touches it.
/// Its transitions never pass a barrier (see [`Lds::private_publish`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct WavePrivate {
    /// Bytes each wave owns.
    pub stride: u32,
    /// Waves that own a slice (the slot covers `stride * waves` bytes).
    pub waves: u32,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LdsSlot {
    pub name: String,
    pub base: u32,
    pub len: u32,
    pub state: SlotState,
    /// `Some` for a wave-private slot (`len == stride * waves`); the proof
    /// record of the layout a certifier checks DS addresses against.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wave_private: Option<WavePrivate>,
}
/// One barrier-carried slot transition (shared with the typed core).
pub use peacemaker_author::SlotTransition as Transition;
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lds {
    pub slots: Vec<LdsSlot>,
    in_flight: Vec<Transition>,
    signalled: bool,
    retired: Vec<LdsSlot>,
}
impl Lds {
    pub fn add(&mut self, name: impl Into<String>, base: u32, len: u32) -> Result<usize, String> {
        if len == 0 || base.checked_add(len).is_none() {
            return Err("invalid LDS slot".into());
        }
        if self
            .slots
            .iter()
            .any(|s| base < s.base + s.len && s.base < base + len)
        {
            return Err("overlapping LDS slots".into());
        }
        self.slots.push(LdsSlot {
            name: name.into(),
            base,
            len,
            state: SlotState::Free,
            wave_private: None,
        });
        Ok(self.slots.len() - 1)
    }
    /// Declare a wave-private slot: `waves` slices of `stride` bytes at `base`
    /// (slice `w` is `[base + w * stride, base + (w + 1) * stride)`), one slot id
    /// whose state is each wave's own (all waves run one program). The slices
    /// are disjoint by construction and the whole window may not overlap any
    /// other slot, so no shared slot can alias a wave's private bytes.
    pub fn add_wave_private(
        &mut self,
        name: impl Into<String>,
        base: u32,
        stride: u32,
        waves: u32,
    ) -> Result<usize, String> {
        if stride == 0 || waves == 0 {
            return Err("a wave-private LDS slot needs a nonzero stride and wave count".into());
        }
        if stride % 16 != 0 || base % 16 != 0 {
            return Err(
                "a wave-private LDS slot's base and stride must be multiples of 16 bytes (every wave's slice starts 16-byte aligned for DS b128 accesses)"
                    .into(),
            );
        }
        let len = stride
            .checked_mul(waves)
            .ok_or("wave-private LDS window overflows")?;
        let id = self.add(name, base, len)?;
        self.slots[id].wave_private = Some(WavePrivate { stride, waves });
        Ok(id)
    }
    /// The slot is a wave-private one (the barrier and shared store/load entry
    /// points refuse it).
    fn refuse_private(&self, id: usize, what: &str) -> Result<(), String> {
        match self.slots.get(id) {
            Some(s) if s.wave_private.is_some() => Err(format!(
                "{} is wave-private: {what} would give other waves access to it; its ownership never crosses a barrier",
                s.name
            )),
            _ => Ok(()),
        }
    }
    fn require_private(&self, id: usize, what: &str) -> Result<(), String> {
        match self.slots.get(id) {
            Some(s) if s.wave_private.is_none() => Err(format!(
                "{} is a workgroup-shared slot: {what} is for wave-private slots",
                s.name
            )),
            _ => Ok(()),
        }
    }
    fn slot(&mut self, id: usize) -> Result<&mut LdsSlot, String> {
        if self.signalled
            && self
                .in_flight
                .iter()
                .any(|t| matches!(t,Transition::Retire(n)|Transition::Ready(n) if *n==id))
        {
            return Err("LDS slot in split-barrier transition".into());
        }
        self.slots.get_mut(id).ok_or("unknown LDS slot".into())
    }
    pub fn store(&mut self, id: usize) -> Result<(), String> {
        self.refuse_private(id, "a workgroup-shared store")?;
        let s = self.slot(id)?;
        if !matches!(s.state, SlotState::Free | SlotState::Publishing) {
            return Err(format!("{} cannot be stored while {:?}", s.name, s.state));
        }
        s.state = SlotState::Publishing;
        Ok(())
    }
    pub fn load(&mut self, id: usize) -> Result<(), String> {
        self.refuse_private(id, "a workgroup-shared load")?;
        let s = self.slot(id)?;
        if !matches!(s.state, SlotState::Published | SlotState::Reading) {
            return Err(format!("{} is not published", s.name));
        }
        s.state = SlotState::Reading;
        Ok(())
    }
    /// A wave-private store: `Free | Publishing -> Publishing`.
    pub fn private_store(&mut self, id: usize) -> Result<(), String> {
        self.require_private(id, "a wave-private store")?;
        let s = self.slot(id)?;
        if !matches!(s.state, SlotState::Free | SlotState::Publishing) {
            return Err(format!("{} cannot be stored while {:?}", s.name, s.state));
        }
        s.state = SlotState::Publishing;
        Ok(())
    }
    /// A wave-private load: `Published | Reading -> Reading`.
    pub fn private_load(&mut self, id: usize) -> Result<(), String> {
        self.require_private(id, "a wave-private load")?;
        let s = self.slot(id)?;
        if !matches!(s.state, SlotState::Published | SlotState::Reading) {
            return Err(format!("{} is not published to its wave", s.name));
        }
        s.state = SlotState::Reading;
        Ok(())
    }
    /// Publish a wave-private slot to its owning wave: `Publishing ->
    /// Published`, no barrier and no instruction. The only reader is the wave
    /// that stored, and the LDS keeps one wave's DS instructions in issue
    /// order (RDNA3 ISA 12.5 "Data Share Indexed and Atomic Access": "LDS
    /// instructions stay in-order with other LDS instructions from the same
    /// wave"; RDNA4 ISA 5.7.1.4 "LDS instructions are kept in order"), so a load
    /// issued after the store reads the stored bytes with no `lgkmcnt`/`dscnt`
    /// wait between them. A wait is only for the loaded VGPRs (the ledger's
    /// use waits), never for this ordering.
    pub fn private_publish(&mut self, id: usize) -> Result<(), String> {
        self.require_private(id, "a wave-private publish")?;
        let s = self.slot(id)?;
        if s.state != SlotState::Publishing {
            return Err(format!(
                "{} cannot be published to its wave while {:?}: it needs a store first",
                s.name, s.state
            ));
        }
        s.state = SlotState::Published;
        Ok(())
    }
    /// Retire a wave-private slot: `Published | Reading -> Free`, no barrier
    /// and no instruction. The wave's own loads are already issued ahead of any
    /// later store of its own (same in-order rule), and no other wave reads the
    /// slice, so there is no cross-wave write-after-read to wait out.
    pub fn private_retire(&mut self, id: usize) -> Result<(), String> {
        self.require_private(id, "a wave-private retire")?;
        let s = self.slot(id)?;
        if !matches!(s.state, SlotState::Published | SlotState::Reading) {
            return Err(format!(
                "{} cannot be retired while {:?}: nothing was published",
                s.name, s.state
            ));
        }
        s.state = SlotState::Free;
        Ok(())
    }
    pub fn barrier_signal(
        &mut self,
        transitions: &[Transition],
        stores_drained: bool,
    ) -> Result<(), String> {
        if self.signalled {
            return Err("split barrier already in flight".into());
        }
        if !stores_drained {
            return Err("LDS stores must be drained before barrier signal".into());
        }
        for t in transitions {
            let (id, expected) = match t {
                Transition::Retire(id) => (*id, SlotState::Reading),
                Transition::Ready(id) => (*id, SlotState::Publishing),
            };
            self.refuse_private(id, "a barrier transition (a cross-wave publish or retire)")?;
            if self.slots.get(id).map(|s| s.state) != Some(expected) {
                return Err(format!("invalid LDS transition {t:?}"));
            }
        }
        self.in_flight = transitions.to_vec();
        self.signalled = true;
        Ok(())
    }
    pub fn barrier_wait(&mut self) -> Result<(), String> {
        if !self.signalled {
            return Err("no barrier signal in flight".into());
        }
        for t in self.in_flight.drain(..) {
            match t {
                Transition::Retire(id) => self.slots[id].state = SlotState::Free,
                Transition::Ready(id) => self.slots[id].state = SlotState::Published,
            }
        }
        self.signalled = false;
        Ok(())
    }
    pub fn barrier(
        &mut self,
        transitions: &[Transition],
        stores_drained: bool,
    ) -> Result<(), String> {
        self.barrier_signal(transitions, stores_drained)?;
        self.barrier_wait()
    }
    /// End the current slot layout so a later program phase can declare its
    /// own (possibly overlapping) slots. Every slot must be Free (its readers
    /// retired by a barrier) and no split barrier may be in flight; LDS bytes
    /// carry no typed meaning across the boundary. Slot ids restart at zero.
    pub fn relayout(&mut self) -> Result<(), String> {
        if self.signalled {
            return Err("relayout with a split barrier in flight".into());
        }
        if let Some(s) = self.slots.iter().find(|s| s.state != SlotState::Free) {
            return Err(format!("{} is {:?} at relayout", s.name, s.state));
        }
        self.retired.append(&mut self.slots);
        Ok(())
    }
    /// Join another control path's slot states into this point. Equal states
    /// stay; a slot published on both paths and read on one is `Reading` (its
    /// reads still need a barrier to retire); a slot stored on one path and
    /// untouched on the other is `Publishing` (a typed region `Writing` on both:
    /// the next barrier publishes whatever this and the other waves stored);
    /// any other disagreement, or a different layout or barrier in flight, is
    /// refused.
    pub fn join(&mut self, other: &Lds) -> Result<(), String> {
        if self.signalled != other.signalled || self.in_flight != other.in_flight {
            return Err("paths join with different split barriers in flight".into());
        }
        if self.slots.len() != other.slots.len() || self.retired.len() != other.retired.len() {
            return Err("paths join with different LDS layouts".into());
        }
        for (mine, theirs) in self.slots.iter_mut().zip(&other.slots) {
            if (&mine.name, mine.base, mine.len, mine.wave_private)
                != (&theirs.name, theirs.base, theirs.len, theirs.wave_private)
            {
                return Err("paths join with different LDS layouts".into());
            }
            mine.state = match (mine.state, theirs.state) {
                (a, b) if a == b => a,
                (
                    SlotState::Published | SlotState::Reading,
                    SlotState::Published | SlotState::Reading,
                ) => SlotState::Reading,
                // Stored on one path only: a shared region is `Writing` on both
                // (the next barrier publishes what any wave stored). A
                // wave-private slice is each wave's own, so a wave that took the
                // other path holds nothing there: no merge.
                (
                    SlotState::Free | SlotState::Publishing,
                    SlotState::Free | SlotState::Publishing,
                ) if mine.wave_private.is_none() => SlotState::Publishing,
                (a, b) => {
                    return Err(format!(
                        "paths join with {} {a:?} on one and {b:?} on the other",
                        mine.name
                    ))
                }
            };
        }
        Ok(())
    }
    /// Every slot of every layout, retired phases first (the proof record).
    pub fn all_slots(&self) -> Vec<LdsSlot> {
        self.retired.iter().chain(&self.slots).cloned().collect()
    }
}

/// One wave-private LDS window as the builder's proof records it
/// (`BuilderProof::lds_slots` entries with `wave_private`): wave `w` of the
/// workgroup owns bytes `[base + w * stride, base + (w + 1) * stride)` and
/// no other wave touches them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivateWindow {
    pub base: u32,
    pub stride: u32,
    pub waves: u32,
}
impl PrivateWindow {
    /// One past the window's last byte (saturating).
    pub fn end(&self) -> u32 {
        self.base
            .saturating_add(self.stride.saturating_mul(self.waves))
    }
}

/// The wave-private windows of a builder proof's LDS slots.
pub fn private_windows(slots: &[LdsSlot]) -> Vec<PrivateWindow> {
    slots
        .iter()
        .filter_map(|s| {
            s.wave_private.map(|p| PrivateWindow {
                base: s.base,
                stride: p.stride,
                waves: p.waves,
            })
        })
        .collect()
}

/// Prefix of the assembly comment that records one wave-private window of a kernel.
const PRIVATE_MARKER: &str = "; pm.wave_private";

/// The assembly line the builder emits for one wave-private window of `symbol` (before the kernel's
/// `.protected` line): `; pm.wave_private symbol=<name> base=<u32> stride=<u32> waves=<u32>`. It is
/// derived from the builder's LDS slots on every emission, so a kernel that has wave-private slots
/// cannot omit it; [`private_markers`] reads it back.
pub fn private_marker(symbol: &str, w: &PrivateWindow) -> String {
    format!(
        "{PRIVATE_MARKER} symbol={symbol} base={} stride={} waves={}",
        w.base, w.stride, w.waves
    )
}

/// The wave-private windows `source` declares for `symbol` (in emission order; markers of other
/// kernels in a combined module are ignored). A malformed marker line is an error, not a skip.
pub fn private_markers(source: &str, symbol: &str) -> Result<Vec<PrivateWindow>, String> {
    let mut out = Vec::new();
    for line in source.lines() {
        let Some(rest) = line.trim().strip_prefix(PRIVATE_MARKER) else {
            continue;
        };
        let bad = || format!("malformed wave-private marker `{}`", line.trim());
        let mut fields = rest.split_whitespace();
        let mut field = |key: &str| -> Result<&str, String> {
            fields
                .next()
                .and_then(|f| f.strip_prefix(key))
                .and_then(|f| f.strip_prefix('='))
                .ok_or_else(bad)
        };
        let name = field("symbol")?;
        let base = field("base")?.parse().map_err(|_| bad())?;
        let stride = field("stride")?.parse().map_err(|_| bad())?;
        let waves = field("waves")?.parse().map_err(|_| bad())?;
        if fields.next().is_some() {
            return Err(bad());
        }
        if name == symbol {
            out.push(PrivateWindow {
                base,
                stride,
                waves,
            })
        }
    }
    Ok(out)
}
