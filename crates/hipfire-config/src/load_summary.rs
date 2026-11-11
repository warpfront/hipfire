//! Per-load decision notes behind the one-line `hipfire load summary:`.
//!
//! The engine's own decision sites (loader, arch crates) call [`note`] next to
//! the log line they already print; the daemon calls [`begin`] when a load
//! request starts and [`take`] once the model is resident, then formats the
//! summary from the notes plus the state it already holds. Notes carry the
//! *reason* an engine decision came out the way it did; nothing here derives
//! policy.
//!
//! Process-global because the decision sites are deep in loader code with no
//! shared context to thread a collector through. Loads are serialised in the
//! daemon, and [`begin`] resets the store, so a note can never leak into the
//! next load's summary.

use std::sync::Mutex;

/// Which summary facet a note belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Facet {
    /// Speculation route that is on (e.g. `mtp K=3`). Last write wins.
    SpecOn,
    /// A reason a speculation route is off. Accumulates; identical text is
    /// recorded once.
    SpecOff,
    /// The MTP n-gram copy lane verdict. Last write wins.
    CopyLane,
    /// Weight placement (resident vs host-mapped, partial offload).
    /// Accumulates; identical text is recorded once.
    Placement,
    /// Load-time prefill decisions (chunk rows, expert staging, trunk route).
    /// Accumulates; identical text is recorded once.
    Prefill,
    /// Post-load measured context-fit verdict. Last write wins.
    ContextFit,
    /// The effective KV layout the carrier resolved from the requested mode
    /// (e.g. `fp8` for a requested `auto`). Last write wins.
    Kv,
}

/// Everything noted since the last [`begin`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadNotes {
    pub spec_on: Option<String>,
    pub spec_off: Vec<String>,
    pub copy_lane: Option<String>,
    pub placement: Vec<String>,
    pub prefill: Vec<String>,
    pub context_fit: Option<String>,
    pub kv: Option<String>,
}

static NOTES: Mutex<LoadNotes> = Mutex::new(LoadNotes {
    spec_on: None,
    spec_off: Vec::new(),
    copy_lane: None,
    placement: Vec::new(),
    prefill: Vec::new(),
    context_fit: None,
    kv: None,
});

fn with_notes<R>(f: impl FnOnce(&mut LoadNotes) -> R) -> R {
    // A poisoned lock only means another thread panicked mid-note; the data
    // is plain strings, so keep going rather than lose the summary.
    let mut guard = NOTES.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}

fn push_unique(list: &mut Vec<String>, text: String) {
    if !list.iter().any(|existing| *existing == text) {
        list.push(text);
    }
}

/// Start a new load: discard every note from the previous one.
pub fn begin() {
    with_notes(|n| *n = LoadNotes::default());
}

/// Record a decision for `facet`. Blank text is ignored.
pub fn note(facet: Facet, text: impl Into<String>) {
    let text = text.into();
    if text.trim().is_empty() {
        return;
    }
    with_notes(|n| match facet {
        Facet::SpecOn => n.spec_on = Some(text),
        Facet::SpecOff => push_unique(&mut n.spec_off, text),
        Facet::CopyLane => n.copy_lane = Some(text),
        Facet::Placement => push_unique(&mut n.placement, text),
        Facet::Prefill => push_unique(&mut n.prefill, text),
        Facet::ContextFit => n.context_fit = Some(text),
        Facet::Kv => n.kv = Some(text),
    });
}

/// Take the notes recorded since [`begin`], leaving the store empty.
pub fn take() -> LoadNotes {
    with_notes(std::mem::take)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The store is process-global; serialise the tests that touch it.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn begin_note_take_roundtrip() {
        let _g = lock();
        begin();
        note(Facet::SpecOn, "mtp K=3");
        note(Facet::CopyLane, "mtp n-gram copy lane: on");
        note(Facet::ContextFit, "measured fit unchanged");
        note(Facet::Placement, "experts: fully resident");
        note(Facet::Prefill, "chunk 512 rows");
        let notes = take();
        assert_eq!(notes.spec_on.as_deref(), Some("mtp K=3"));
        assert_eq!(notes.copy_lane.as_deref(), Some("mtp n-gram copy lane: on"));
        assert_eq!(notes.context_fit.as_deref(), Some("measured fit unchanged"));
        assert_eq!(notes.placement, vec!["experts: fully resident".to_string()]);
        assert_eq!(notes.prefill, vec!["chunk 512 rows".to_string()]);
        assert!(notes.spec_off.is_empty());
    }

    #[test]
    fn spec_off_accumulates_and_dedupes() {
        let _g = lock();
        begin();
        note(Facet::SpecOff, "MTP off: a");
        note(Facet::SpecOff, "kv_adaptive suppresses speculation");
        note(Facet::SpecOff, "MTP off: a");
        assert_eq!(
            take().spec_off,
            vec![
                "MTP off: a".to_string(),
                "kv_adaptive suppresses speculation".to_string()
            ]
        );
    }

    #[test]
    fn last_write_wins_for_single_slot_facets() {
        let _g = lock();
        begin();
        note(Facet::SpecOn, "dflash (block 16)");
        note(Facet::SpecOn, "dspark (block 8)");
        note(Facet::ContextFit, "measured fit unchanged");
        note(Facet::ContextFit, "measured fit 4096 -> 2048");
        let notes = take();
        assert_eq!(notes.spec_on.as_deref(), Some("dspark (block 8)"));
        assert_eq!(
            notes.context_fit.as_deref(),
            Some("measured fit 4096 -> 2048")
        );
    }

    #[test]
    fn begin_resets_and_take_empties() {
        let _g = lock();
        begin();
        note(Facet::SpecOff, "stale");
        note(Facet::SpecOn, "stale");
        begin();
        assert_eq!(take(), LoadNotes::default());
        note(Facet::SpecOn, "x");
        let _ = take();
        assert_eq!(take(), LoadNotes::default());
    }

    #[test]
    fn blank_notes_are_ignored() {
        let _g = lock();
        begin();
        note(Facet::SpecOn, "   ");
        note(Facet::SpecOff, "");
        assert_eq!(take(), LoadNotes::default());
    }
}
