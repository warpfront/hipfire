// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Terminal control plane for the daemon — single-request and continuous-batch
//! handshakes, active attempt TLS, and the two-phase `commit_ready` protocol.
//!
//! Relocated verbatim from `crates/hipfire-daemon/src/main.rs` (wave 3)
//! to break the `daemon -> loader -> daemon` cycle. No behaviour change.

use crate::emit::TerminalEmitOutcome;
use std::cell::Cell;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Outcome of the two-phase client terminal handshake
/// (`commit_ready` → matching `commit` / `abort`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientTerminalDecision {
    /// Matching `commit` after readiness — producers may release tools/cache/done.
    Commit,
    /// Matching `abort`, timeout, or any fail-closed control outcome.
    Abort,
}

/// Control decision latched against the active generate transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalControlDecision {
    Abort,
    Commit,
}

/// Active generate terminal-control transaction keyed by exact
/// `(request id, attempt_id)`. The stdin reader posts matching
/// `abort` (any time) / `commit` (only after ready); producers wait
/// via [`await_client_terminal_commit`]. Terminal writers claim the
/// transaction before emitting a terminal so an abort/error race cannot
/// produce a second terminal.
#[derive(Debug, Clone)]
pub struct ActiveTerminalControl {
    pub id: String,
    pub attempt_id: u64,
    /// Monotonic lifecycle generation. A writer that captured an earlier
    /// generation can never claim a reused `(id, attempt_id)`.
    pub generation: u64,
    pub ready: bool,
    pub decision: Option<TerminalControlDecision>,
    pub terminal_claimed: bool,
}

/// Ownership handed from a batch lane to the sequential singleton.
///
/// The batch admission is provenance only after handoff; the singleton
/// transaction snapshot is restored verbatim by the main loop. In particular,
/// a pre-latched abort and its lifecycle generation must survive the driver's
/// return and the outer terminal-control guard. The exact key and admission
/// token are retained so adoption cannot rediscover a reused wire key.
#[derive(Debug, Clone)]
pub struct SingletonTransfer {
    key: AttemptKey,
    admission: BatchGeneration,
    batch_abort_latched: bool,
    singleton: Option<ActiveTerminalControl>,
}

impl SingletonTransfer {
    pub fn admission(&self) -> BatchGeneration {
        self.admission
    }

    pub fn batch_abort_latched(&self) -> bool {
        self.batch_abort_latched
    }
}

// State intentionally carries no retired singleton key: a cleared lifecycle
// must fail closed until an explicit activation establishes a new owner.

pub struct TerminalControlState {
    pub active: Option<ActiveTerminalControl>,
    pub next_generation: u64,
}

impl TerminalControlState {
    pub const fn new() -> Self {
        Self {
            active: None,
            next_generation: 0,
        }
    }
}

pub struct TerminalControlCell {
    pub mu: Mutex<TerminalControlState>,
    pub cv: Condvar,
}

pub fn terminal_control() -> &'static TerminalControlCell {
    static CELL: OnceLock<TerminalControlCell> = OnceLock::new();
    CELL.get_or_init(|| TerminalControlCell {
        mu: Mutex::new(TerminalControlState::new()),
        cv: Condvar::new(),
    })
}

/// Bound on how long the daemon waits for a matching commit/abort after
/// emitting `commit_ready`. Timeout fails closed as Abort (never commit).
pub const CLIENT_TERMINAL_COMMIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Activate a fresh terminal-control transaction for this generate.
///
/// The batch tombstone is checked under the same terminal→batch lock order as
/// handoff and reader control. A stale caller cannot overwrite a transfer that
/// is still waiting for adoption.
pub fn activate_terminal_control(id: &str, attempt_id: u64) {
    let terminal_cell = terminal_control();
    let mut terminal = terminal_cell.mu.lock().unwrap();
    let batch_cell = batch_terminal_control();
    let batch = batch_cell.mu.lock().unwrap();
    if batch
        .handoffs
        .contains_key(&AttemptKey::new(id, attempt_id))
    {
        return;
    }
    terminal.next_generation = terminal.next_generation.checked_add(1).unwrap_or(1);
    let generation = terminal.next_generation;
    terminal.active = Some(ActiveTerminalControl {
        id: id.to_string(),
        attempt_id,
        generation,
        ready: false,
        decision: None,
        terminal_claimed: false,
    });
    terminal_cell.cv.notify_all();
}

/// Clear the active terminal-control transaction (request end / guard drop).
///
/// Clearing is fail-closed: a writer may not infer ownership from a nonzero
/// attempt after the lifecycle ends. An adopted handoff tombstone is released
/// only with its matching singleton owner; a pending transfer stays protected
/// across the outer batch driver's guard drop.
pub fn clear_terminal_control() {
    let terminal_cell = terminal_control();
    let mut terminal = terminal_cell.mu.lock().unwrap();
    let completed_key = terminal
        .active
        .as_ref()
        .map(|active| AttemptKey::new(&active.id, active.attempt_id));
    terminal.active = None;
    let batch_cell = batch_terminal_control();
    let mut batch = batch_cell.mu.lock().unwrap();
    if let Some(key) = completed_key {
        if batch
            .handoffs
            .get(&key)
            .is_some_and(|handoff| handoff.adopted)
        {
            batch.handoffs.remove(&key);
            batch_cell.cv.notify_all();
        }
    }
    terminal_cell.cv.notify_all();
}

/// Return the active singleton generation for an exact request key.
pub fn terminal_generation(id: &str, attempt_id: u64) -> Option<u64> {
    let cell = terminal_control();
    let g = cell.mu.lock().unwrap();
    g.active.as_ref().and_then(|active| {
        (active.id == id && active.attempt_id == attempt_id).then_some(active.generation)
    })
}

fn claim_terminal_state(
    state: &mut TerminalControlState,
    id: &str,
    attempt_id: u64,
    generation: Option<u64>,
) -> bool {
    let Some(active) = state.active.as_mut() else {
        return false;
    };
    if active.id != id
        || active.attempt_id != attempt_id
        || generation.is_some_and(|expected| active.generation != expected)
    {
        return false;
    }
    if active.terminal_claimed {
        return false;
    }
    active.terminal_claimed = true;
    true
}

/// Claim a terminal only for a captured singleton lifecycle generation.
///
/// This is the strict form used by race/reuse tests and any caller that holds
/// a request-owned generation token. It is intentionally not inferred from
/// the current active state: inferring it would let an old writer claim a
/// freshly reactivated request with the same wire key.
pub fn claim_terminal_at_generation(id: &str, attempt_id: u64, generation: u64) -> bool {
    let cell = terminal_control();
    let mut g = cell.mu.lock().unwrap();
    claim_terminal_state(&mut g, id, attempt_id, Some(generation))
}

/// Claim the sole terminal slot for the current request key.
///
/// Unknown/mismatched active attempts, inactive lifecycles, and all
/// attempt-zero writers fail closed. Pre-admission failures must use
/// `emit_uncorrelated_error` instead of claiming a singleton lifecycle that
/// has not been activated.
pub fn claim_terminal(id: &str, attempt_id: u64) -> bool {
    if attempt_id == 0 {
        return false;
    }
    let cell = terminal_control();
    let mut g = cell.mu.lock().unwrap();
    claim_terminal_state(&mut g, id, attempt_id, None)
}

/// Key for multiplexed terminal control and inbox, as required by the
/// continuous-batch contract: every lifecycle event is keyed by
/// `(id, attempt_id)` and unknown/stale keys fail closed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AttemptKey {
    pub id: String,
    pub attempt_id: u64,
}

impl AttemptKey {
    pub fn new(id: &str, attempt_id: u64) -> Self {
        Self {
            id: id.to_string(),
            attempt_id,
        }
    }
}

/// Opaque admission generation owned by the keyed batch registry.
///
/// The registry is the only production code that can mint a token. Keeping
/// the counter private prevents a lane or producer from manufacturing a
/// generation that happens to match a later admission for the same wire key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BatchGeneration(u64);

/// Generation-owned lane ticket. The scheduler's `generation` remains the
/// lane-reuse generation; `admission` is the opaque registry generation.
/// Both are required to identify a live producer owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaneTicket {
    pub lane: usize,
    pub generation: u64,
    pub admission: BatchGeneration,
}

impl LaneTicket {
    pub fn admission(self) -> BatchGeneration {
        self.admission
    }
}

// ── Keyed terminal registry ────────────────────────────────────────────

/// Registry state for a single `(id, attempt_id)`. Each variant can retain
/// an Abort; only Ready accepts Commit. Deadline is 30 s after Ready.
#[derive(Debug, Clone)]
pub enum BatchRegistryState {
    Announced,
    Queued,
    Active { owner: LaneTicket },
    Ready { owner: LaneTicket },
}

#[derive(Debug, Clone)]
pub struct BatchRegistryEntry {
    pub state: BatchRegistryState,
    pub abort_latched: bool,
    pub commit_latched: bool,
    pub terminal_claimed: bool,
    /// Opaque admission generation. It is copied into every producer-owned
    /// request/lane ticket and is required for terminal operations.
    pub generation: BatchGeneration,
    pub pending_done: Option<serde_json::Value>,
    pub deadline: Option<Instant>,
}
#[derive(Debug, Clone)]
struct SingletonHandoff {
    admission: BatchGeneration,
    singleton: Option<ActiveTerminalControl>,
    abort_latched: bool,
    adopted: bool,
}

impl SingletonHandoff {
    fn new(
        admission: BatchGeneration,
        singleton: Option<ActiveTerminalControl>,
        abort_latched: bool,
    ) -> Self {
        Self {
            admission,
            singleton,
            abort_latched,
            adopted: false,
        }
    }
}

// Entries are removed only after adoption/terminal cleanup. A handoff tombstone
// reserves the exact wire key so the next admission cannot overtake its owner.
//
// Every operation that needs both cells takes terminal_control().mu first and
// batch_terminal_control().mu second. This ordering is the transfer boundary:
// reader controls, handoff, claims, admission, and cleanup cannot deadlock or
// observe a removal-before-adoption gap.
pub struct BatchTerminalState {
    pub entries: std::collections::HashMap<AttemptKey, BatchRegistryEntry>,
    handoffs: std::collections::HashMap<AttemptKey, SingletonHandoff>,
    /// Monotonic epoch for ordinary admissions. Retired batch entries are
    /// removed; only in-flight singleton handoffs use the separate tombstone
    /// map above to reserve their exact wire key.
    pub next_generation: u64,
}

impl BatchTerminalState {
    pub fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            handoffs: std::collections::HashMap::new(),
            next_generation: 0,
        }
    }
}

pub struct BatchTerminalCell {
    pub mu: Mutex<BatchTerminalState>,
    pub cv: Condvar,
}

pub fn batch_terminal_control() -> &'static BatchTerminalCell {
    static CELL: OnceLock<BatchTerminalCell> = OnceLock::new();
    CELL.get_or_init(|| BatchTerminalCell {
        mu: Mutex::new(BatchTerminalState::new()),
        cv: Condvar::new(),
    })
}

/// Claim exactly one wire-terminal owner for a continuous-batch attempt.
///
/// A request-owned [`BatchAttemptScope`] is required. This makes a cleared
/// generation fail closed even if the same `(id, attempt_id)` is announced
/// again before an old producer returns.
pub fn batch_claim_terminal(id: &str, attempt_id: u64) -> bool {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    let key = AttemptKey::new(id, attempt_id);
    let Some(entry) = g.entries.get_mut(&key) else {
        return false;
    };
    if active_batch_generation() != Some(entry.generation) || entry.terminal_claimed {
        return false;
    }
    entry.terminal_claimed = true;
    cell.cv.notify_all();
    true
}

/// Claim the wire-terminal boundary for either a continuous-batch lane or
/// the sequential active attempt. Batch keys are checked first so a stale or
/// wrong-attempt writer cannot fall through to the singleton claim.
///
/// This takes the terminal lock before the batch lock. The handoff tombstone
/// therefore keeps the adopted singleton eligible while rejecting stale batch
/// producers and same-key re-admission.
pub fn claim_wire_terminal(id: &str, attempt_id: u64) -> bool {
    let terminal_cell = terminal_control();
    let mut terminal = terminal_cell.mu.lock().unwrap();
    let batch_cell = batch_terminal_control();
    let mut batch = batch_cell.mu.lock().unwrap();
    let key = AttemptKey::new(id, attempt_id);
    if let Some(entry) = batch.entries.get_mut(&key) {
        if active_batch_generation() != Some(entry.generation) || entry.terminal_claimed {
            return false;
        }
        entry.terminal_claimed = true;
        batch_cell.cv.notify_all();
        return true;
    }
    if let Some(handoff) = batch.handoffs.get(&key) {
        if active_batch_generation().is_some() || !handoff.adopted {
            return false;
        }
        let claimed = claim_terminal_state(&mut terminal, id, attempt_id, None);
        if claimed {
            terminal_cell.cv.notify_all();
        }
        return claimed;
    }
    // A scope with no live entry is a stale batch producer. Never let it
    // fall through to the singleton after its keyed generation retired.
    if active_batch_generation().is_some()
        || batch.entries.keys().any(|candidate| candidate.id == id)
    {
        return false;
    }
    let claimed = claim_terminal_state(&mut terminal, id, attempt_id, None);
    if claimed {
        terminal_cell.cv.notify_all();
    }
    claimed
}

/// Announce a generate key before queueing and return its opaque admission
/// generation. A present key or transfer tombstone is not re-owned.
pub fn batch_announce_terminal(id: &str, attempt_id: u64) -> Option<BatchGeneration> {
    let terminal_cell = terminal_control();
    let _terminal = terminal_cell.mu.lock().unwrap();
    let batch_cell = batch_terminal_control();
    let mut batch = batch_cell.mu.lock().unwrap();
    let key = AttemptKey::new(id, attempt_id);
    if batch.entries.contains_key(&key) || batch.handoffs.contains_key(&key) {
        return None;
    }
    batch.next_generation = batch.next_generation.checked_add(1).unwrap_or(1);
    let generation = BatchGeneration(batch.next_generation);
    batch.entries.insert(
        key,
        BatchRegistryEntry {
            state: BatchRegistryState::Announced,
            abort_latched: false,
            commit_latched: false,
            terminal_claimed: false,
            generation,
            pending_done: None,
            deadline: None,
        },
    );
    batch_cell.cv.notify_all();
    Some(generation)
}

/// Explicit batch admission alias. Returns the newly owned token.
pub fn batch_activate_terminal(id: &str, attempt_id: u64) -> Option<BatchGeneration> {
    batch_announce_terminal(id, attempt_id)
}

/// Promote an exact admission from Announced to Queued. Repeating the
/// transition for the same owner is idempotent; a stale token fails closed.
pub fn batch_transition_to_queued(id: &str, attempt_id: u64, generation: BatchGeneration) -> bool {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    if let Some(e) = g.entries.get_mut(&AttemptKey::new(id, attempt_id)) {
        if e.generation != generation {
            return false;
        }
        if matches!(
            e.state,
            BatchRegistryState::Announced | BatchRegistryState::Queued
        ) {
            e.state = BatchRegistryState::Queued;
            cell.cv.notify_all();
            return true;
        }
    }
    false
}

/// Bind a lane owner only when both the admission token and the lane ticket
/// agree with the live registry entry.
pub fn batch_bind_active(
    id: &str,
    attempt_id: u64,
    generation: BatchGeneration,
    owner: LaneTicket,
) -> bool {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    if let Some(e) = g.entries.get_mut(&AttemptKey::new(id, attempt_id)) {
        if e.generation == generation
            && owner.admission == generation
            && matches!(e.state, BatchRegistryState::Queued)
        {
            e.state = BatchRegistryState::Active { owner };
            cell.cv.notify_all();
            return true;
        }
    }
    false
}

pub fn batch_mark_ready_with_pending(
    id: &str,
    attempt_id: u64,
    generation: BatchGeneration,
    owner: LaneTicket,
    pending_done: serde_json::Value,
) -> bool {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    if let Some(e) = g.entries.get_mut(&AttemptKey::new(id, attempt_id)) {
        if e.generation == generation
            && owner.admission == generation
            && matches!(e.state, BatchRegistryState::Active { owner: o } if o == owner)
        {
            e.state = BatchRegistryState::Ready { owner };
            e.pending_done = Some(pending_done);
            e.deadline = Some(Instant::now() + CLIENT_TERMINAL_COMMIT_TIMEOUT);
            cell.cv.notify_all();
            return true;
        }
    }
    false
}

/// Host-only ready marker. Producer paths should use
/// [`batch_mark_ready_with_pending`] with their captured token.
pub fn batch_mark_ready(id: &str, attempt_id: u64, generation: BatchGeneration) -> bool {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    if let Some(e) = g.entries.get_mut(&AttemptKey::new(id, attempt_id)) {
        if e.generation != generation {
            return false;
        }
        if let BatchRegistryState::Active { owner } = e.state {
            e.state = BatchRegistryState::Ready { owner };
            e.deadline = Some(Instant::now() + CLIENT_TERMINAL_COMMIT_TIMEOUT);
            if e.pending_done.is_none() {
                e.pending_done =
                    Some(serde_json::json!({"type":"done","id":id,"attempt_id":attempt_id}));
            }
            cell.cv.notify_all();
            return true;
        }
    }
    false
}

/// Legacy administrative/test teardown. Producer paths must use
/// [`batch_clear_terminal_at_generation`] so a stale owner cannot clear B.
pub fn batch_clear_terminal(id: &str, attempt_id: u64) {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    g.entries.remove(&AttemptKey::new(id, attempt_id));
    cell.cv.notify_all();
}

/// Return the admission generation for an exact batch key.
pub fn batch_terminal_generation(id: &str, attempt_id: u64) -> Option<BatchGeneration> {
    let cell = batch_terminal_control();
    let g = cell.mu.lock().unwrap();
    g.entries
        .get(&AttemptKey::new(id, attempt_id))
        .map(|entry| entry.generation)
}

/// Remove a batch entry only when its opaque admission generation still
/// matches. A retired A owner cannot remove later B.
pub fn batch_clear_terminal_at_generation(
    id: &str,
    attempt_id: u64,
    generation: BatchGeneration,
) -> bool {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    let key = AttemptKey::new(id, attempt_id);
    let matches = g
        .entries
        .get(&key)
        .is_some_and(|entry| entry.generation == generation);
    if matches {
        g.entries.remove(&key);
        cell.cv.notify_all();
    }
    matches
}

/// True when the exact admission is still live, regardless of lifecycle
/// state. This is used before copying a request token into scheduler state.
pub fn batch_is_current(id: &str, attempt_id: u64, generation: BatchGeneration) -> bool {
    let cell = batch_terminal_control();
    let g = cell.mu.lock().unwrap();
    g.entries
        .get(&AttemptKey::new(id, attempt_id))
        .is_some_and(|entry| entry.generation == generation)
}

pub fn batch_active_owner_matches(
    id: &str,
    attempt_id: u64,
    generation: BatchGeneration,
    owner: LaneTicket,
) -> bool {
    let cell = batch_terminal_control();
    let g = cell.mu.lock().unwrap();
    matches!(
        g.entries.get(&AttemptKey::new(id, attempt_id)),
        Some(e)
            if e.generation == generation
                && matches!(e.state, BatchRegistryState::Active { owner: o } if o == owner)
    )
}

pub fn batch_ready_owner_matches(
    id: &str,
    attempt_id: u64,
    generation: BatchGeneration,
    owner: LaneTicket,
) -> bool {
    let cell = batch_terminal_control();
    let g = cell.mu.lock().unwrap();
    matches!(
        g.entries.get(&AttemptKey::new(id, attempt_id)),
        Some(e)
            if e.generation == generation
                && matches!(e.state, BatchRegistryState::Ready { owner: o } if o == owner)
    )
}

pub fn batch_clear_all_terminals() {
    let terminal_cell = terminal_control();
    let _terminal = terminal_cell.mu.lock().unwrap();
    let batch_cell = batch_terminal_control();
    let mut batch = batch_cell.mu.lock().unwrap();
    batch.entries.clear();
    batch.handoffs.clear();
    batch_cell.cv.notify_all();
}

fn apply_handoff_control(
    terminal: &mut TerminalControlState,
    key: &AttemptKey,
    handoff: &mut SingletonHandoff,
    kind: &str,
) -> bool {
    let mut changed = false;
    match kind {
        "abort" => {
            if !handoff.abort_latched {
                handoff.abort_latched = true;
                changed = true;
            }
            if let Some(singleton) = handoff.singleton.as_mut() {
                if singleton.decision.is_none() {
                    singleton.decision = Some(TerminalControlDecision::Abort);
                    changed = true;
                }
            }
            if handoff.adopted {
                if let Some(active) = terminal
                    .active
                    .as_mut()
                    .filter(|active| active.id == key.id && active.attempt_id == key.attempt_id)
                {
                    if active.decision.is_none() {
                        active.decision = Some(TerminalControlDecision::Abort);
                        changed = true;
                    }
                }
            }
        }
        "commit" if !handoff.abort_latched => {
            if let Some(singleton) = handoff.singleton.as_mut() {
                if singleton.ready && singleton.decision.is_none() {
                    singleton.decision = Some(TerminalControlDecision::Commit);
                    changed = true;
                }
            }
            if handoff.adopted {
                if let Some(active) = terminal
                    .active
                    .as_mut()
                    .filter(|active| active.id == key.id && active.attempt_id == key.attempt_id)
                {
                    if active.ready && active.decision.is_none() {
                        active.decision = Some(TerminalControlDecision::Commit);
                        changed = true;
                    }
                }
            }
        }
        _ => {}
    }
    changed
}

/// Apply abort/commit control by current wire key. The wire protocol has no
/// generation, so this is intentionally the sole key-only producer input.
///
/// The terminal→batch lock order makes a handoff tombstone visible to the
/// reader without a snapshot/removal gap.
pub fn batch_apply_terminal_control(kind: &str, id: &str, attempt_id: u64) {
    let terminal_cell = terminal_control();
    let mut terminal = terminal_cell.mu.lock().unwrap();
    let batch_cell = batch_terminal_control();
    let mut batch = batch_cell.mu.lock().unwrap();
    let key = AttemptKey::new(id, attempt_id);
    if let Some(handoff) = batch.handoffs.get_mut(&key) {
        if apply_handoff_control(&mut terminal, &key, handoff, kind) {
            terminal_cell.cv.notify_all();
            batch_cell.cv.notify_all();
        }
        return;
    }
    if let Some(entry) = batch.entries.get_mut(&key) {
        match kind {
            "abort" => {
                if !entry.abort_latched {
                    entry.abort_latched = true;
                    batch_cell.cv.notify_all();
                }
            }
            "commit" => {
                if entry.abort_latched {
                    return;
                }
                if matches!(entry.state, BatchRegistryState::Ready { .. }) && !entry.commit_latched
                {
                    entry.commit_latched = true;
                    batch_cell.cv.notify_all();
                }
            }
            _ => {}
        }
    }
}

pub fn batch_check_abort(id: &str, attempt_id: u64, generation: BatchGeneration) -> bool {
    let cell = batch_terminal_control();
    let g = cell.mu.lock().unwrap();
    let key = AttemptKey::new(id, attempt_id);
    if let Some(entry) = g.entries.get(&key) {
        return entry.generation == generation && entry.abort_latched;
    }
    g.handoffs.get(&key).is_some_and(|handoff| {
        handoff.admission == generation && !handoff.adopted && handoff.abort_latched
    })
}
/// Non-mutating generation-checked poll.
pub fn batch_poll_decision(
    id: &str,
    attempt_id: u64,
    generation: BatchGeneration,
) -> Option<ClientTerminalDecision> {
    let cell = batch_terminal_control();
    let g = cell.mu.lock().unwrap();
    let key = AttemptKey::new(id, attempt_id);
    if let Some(e) = g.entries.get(&key) {
        if e.generation != generation {
            return None;
        }
        if e.abort_latched {
            return Some(ClientTerminalDecision::Abort);
        }
        if let Some(deadline) = e.deadline {
            if Instant::now() >= deadline {
                return Some(ClientTerminalDecision::Abort);
            }
        }
        if e.commit_latched && matches!(e.state, BatchRegistryState::Ready { .. }) {
            return Some(ClientTerminalDecision::Commit);
        }
        return None;
    }
    if let Some(handoff) = g.handoffs.get(&key) {
        if handoff.admission == generation && !handoff.adopted {
            return Some(ClientTerminalDecision::Abort);
        }
    }

    None
}
/// Blocking generation-checked wait used by lane commit polling.
pub fn batch_wait_decision(
    id: &str,
    attempt_id: u64,
    generation: BatchGeneration,
    timeout: Duration,
) -> ClientTerminalDecision {
    let cell = batch_terminal_control();
    let mut g = cell.mu.lock().unwrap();
    let deadline = Instant::now() + timeout;
    loop {
        let entry = g.entries.get(&AttemptKey::new(id, attempt_id)).cloned();
        match entry {
            None => return ClientTerminalDecision::Abort,
            Some(e) => {
                if e.generation != generation || e.abort_latched {
                    return ClientTerminalDecision::Abort;
                }
                if let Some(dl) = e.deadline {
                    if Instant::now() >= dl {
                        return ClientTerminalDecision::Abort;
                    }
                }
                if e.commit_latched && matches!(e.state, BatchRegistryState::Ready { .. }) {
                    return ClientTerminalDecision::Commit;
                }
            }
        }
        let now = Instant::now();
        if now >= deadline {
            return ClientTerminalDecision::Abort;
        }
        let remaining = deadline - now;
        let (guard, wait_res) = cell.cv.wait_timeout(g, remaining).unwrap();
        g = guard;
        if wait_res.timed_out() {
            continue;
        }
    }
}

/// Atomically replace an exact batch admission with a singleton handoff
/// tombstone and capture the matching singleton transaction.
///
/// The terminal lock is held before the batch lock. The tombstone remains in
/// the batch state until adoption and the matching singleton lifecycle close,
/// so controls cannot fall into a removal-before-adoption gap and a same-key
/// next generation cannot overtake the old owner.
pub fn batch_handoff_to_singleton_and_clear(
    id: &str,
    attempt_id: u64,
    generation: BatchGeneration,
) -> Option<SingletonTransfer> {
    let terminal_cell = terminal_control();
    let mut terminal = terminal_cell.mu.lock().unwrap();
    let batch_cell = batch_terminal_control();
    let mut batch = batch_cell.mu.lock().unwrap();
    let key = AttemptKey::new(id, attempt_id);
    if batch.handoffs.contains_key(&key) {
        return None;
    }
    let batch_abort_latched = batch
        .entries
        .get(&key)
        .filter(|entry| entry.generation == generation)
        .map(|entry| entry.abort_latched)?;
    let singleton = terminal
        .active
        .as_ref()
        .filter(|active| active.id == id && active.attempt_id == attempt_id);
    let singleton = singleton.cloned();
    batch.entries.remove(&key);
    batch.handoffs.insert(
        key.clone(),
        SingletonHandoff::new(generation, singleton.clone(), batch_abort_latched),
    );
    if terminal
        .active
        .as_ref()
        .is_some_and(|active| active.id == id && active.attempt_id == attempt_id)
    {
        terminal.active = None;
        terminal_cell.cv.notify_all();
    }
    batch_cell.cv.notify_all();
    Some(SingletonTransfer {
        key,
        admission: generation,
        batch_abort_latched,
        singleton,
    })
}

/// Restore a transferred singleton transaction without allocating a new
/// lifecycle generation or resetting an already-latched decision.
///
/// Adoption consumes the exact tombstone token but keeps its reservation
/// until [`clear_terminal_control`] closes the adopted singleton lifecycle.
pub fn adopt_singleton_transfer(id: &str, attempt_id: u64, transfer: SingletonTransfer) -> bool {
    let key = AttemptKey::new(id, attempt_id);
    if transfer.key != key {
        return false;
    }
    let terminal_cell = terminal_control();
    let mut terminal = terminal_cell.mu.lock().unwrap();
    let batch_cell = batch_terminal_control();
    let mut batch = batch_cell.mu.lock().unwrap();
    let Some(handoff) = batch.handoffs.get_mut(&key) else {
        return false;
    };
    if handoff.admission != transfer.admission || handoff.adopted {
        return false;
    }
    let mut singleton = handoff
        .singleton
        .clone()
        .or_else(|| transfer.singleton.clone());
    if singleton.is_none() {
        terminal.next_generation = terminal.next_generation.checked_add(1).unwrap_or(1);
        singleton = Some(ActiveTerminalControl {
            id: id.to_string(),
            attempt_id,
            generation: terminal.next_generation,
            ready: false,
            decision: None,
            terminal_claimed: false,
        });
    }
    let mut singleton = singleton.expect("singleton handoff allocation");
    if singleton.id != id || singleton.attempt_id != attempt_id {
        return false;
    }
    if (handoff.abort_latched || transfer.batch_abort_latched) && singleton.decision.is_none() {
        singleton.decision = Some(TerminalControlDecision::Abort);
    }
    handoff.singleton = Some(singleton.clone());
    handoff.adopted = true;
    terminal.active = Some(singleton);
    terminal_cell.cv.notify_all();
    batch_cell.cv.notify_all();
    true
}

/// Transfer an exact batch admission to the sequential singleton. The
/// handoff is adopted while both ownership cells remain serialized; its
/// tombstone is released with the eventual singleton cleanup.
pub fn batch_transfer_abort_to_singleton_and_clear(
    id: &str,
    attempt_id: u64,
    generation: BatchGeneration,
) -> bool {
    let Some(transfer) = batch_handoff_to_singleton_and_clear(id, attempt_id, generation) else {
        return false;
    };
    let had_abort = transfer.batch_abort_latched();
    if !adopt_singleton_transfer(id, attempt_id, transfer) {
        return false;
    }
    had_abort || check_abort(id)
}

/// Pure commit-teardown classifier: success `done` is allowed only after both
/// fallible GPU reset and host `commit_lane` succeed. Used by the driver and
/// covered by same-file tests so ordering cannot regress silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchCommitTeardownClass {
    /// `reset_lane` failed — use fail_all; never emit done.
    ResetFailed,
    /// Reset ok but `commit_lane` failed — fail closed for that key; no done.
    CommitFailed,
    /// Both transitions ok — emit the staged terminal done.
    EmitDone,
}

pub fn batch_commit_teardown_class(reset_ok: bool, commit_ok: bool) -> BatchCommitTeardownClass {
    if !reset_ok {
        BatchCommitTeardownClass::ResetFailed
    } else if !commit_ok {
        BatchCommitTeardownClass::CommitFailed
    } else {
        BatchCommitTeardownClass::EmitDone
    }
}

/// After the current token is committed, `seq_pos` is the next decode index.
/// A lane is at capacity when that index is no longer strictly below capacity.
pub fn batch_lane_at_capacity(seq_pos: usize, lane_capacity: usize) -> bool {
    seq_pos >= lane_capacity
}

/// Pure LFM capacity gate: `prompt_len + max_tokens` must fit strictly within
/// `lane_capacity`. Uses `saturating_add` so `u64::MAX` never wraps under the cap.
/// Returns `true` when the request exceeds capacity (must be rejected before
/// `gen_start`/GPU).
pub fn batch_lfm_exceeds_capacity(
    prompt_len: usize,
    max_tokens: usize,
    lane_capacity: usize,
) -> bool {
    prompt_len.saturating_add(max_tokens) > lane_capacity
}

/// Shared LFM admission decision: `true` when the request is valid and fits
/// within `lane_capacity` (including `max_tokens`). Invalid (empty or over-cap)
/// must emit validation/context error with no `gen_start`/GPU work.
pub fn batch_lfm_admission_ok(prompt_len: usize, max_tokens: usize, lane_capacity: usize) -> bool {
    if prompt_len == 0 {
        return false;
    }
    !batch_lfm_exceeds_capacity(prompt_len, max_tokens, lane_capacity)
}

/// Length-cap terminal when max_tokens or lane capacity is hit without a
/// competing stop cause (EOS / filter / loop guard).
pub fn batch_hit_length_cap(
    hit_max_tokens: bool,
    hit_lane_capacity: bool,
    is_eos: bool,
    stopped: bool,
    loop_hit: bool,
) -> bool {
    (hit_max_tokens || hit_lane_capacity) && !is_eos && !stopped && !loop_hit
}

pub fn batch_should_finish_decode(
    is_eos: bool,
    hit_max_tokens: bool,
    hit_lane_capacity: bool,
    stopped: bool,
    loop_hit: bool,
) -> bool {
    is_eos || hit_max_tokens || hit_lane_capacity || stopped || loop_hit
}

pub fn batch_pending_deadline(
    id: &str,
    attempt_id: u64,
    generation: BatchGeneration,
) -> Option<Instant> {
    let cell = batch_terminal_control();
    let g = cell.mu.lock().unwrap();
    g.entries
        .get(&AttemptKey::new(id, attempt_id))
        .filter(|e| e.generation == generation)
        .and_then(|e| e.deadline)
}

pub fn batch_is_ready(id: &str, attempt_id: u64, generation: BatchGeneration) -> bool {
    let cell = batch_terminal_control();
    let g = cell.mu.lock().unwrap();
    matches!(
        g.entries.get(&AttemptKey::new(id, attempt_id)),
        Some(e)
            if e.generation == generation
                && matches!(e.state, BatchRegistryState::Ready { .. })
    )
}

// ── TLS active attempt id ───────────────────────────────────────────────

thread_local! {
    /// Active generate attempt_id for typed errors emitted during a request.
    /// Reset path parses attempt_id from the message directly.
    static ACTIVE_ATTEMPT_ID: Cell<u64> = const { Cell::new(0) };
    /// Opaque batch admission generation paired with [`ACTIVE_ATTEMPT_ID`].
    /// `None` means this scope is not a live keyed batch producer.
    static ACTIVE_BATCH_GENERATION: Cell<Option<BatchGeneration>> = const { Cell::new(None) };
}

pub fn active_attempt_id() -> u64 {
    ACTIVE_ATTEMPT_ID.with(|c| c.get())
}

pub fn set_active_attempt_id(id: u64) {
    ACTIVE_ATTEMPT_ID.with(|c| c.set(id));
}

pub fn active_batch_generation() -> Option<BatchGeneration> {
    ACTIVE_BATCH_GENERATION.with(Cell::get)
}

fn batch_generation_for(id: Option<&str>, attempt_id: u64) -> Option<BatchGeneration> {
    let cell = batch_terminal_control();
    let g = cell.mu.lock().unwrap();
    let mut generations = g.entries.iter().filter_map(|(key, entry)| {
        (key.attempt_id == attempt_id && id.is_none_or(|id| key.id == id))
            .then_some(entry.generation)
    });
    let generation = generations.next()?;
    // An attempt id is globally unique in production. If a host-only test
    // deliberately aliases it across lanes, refuse to guess a generation.
    generations.next().is_none().then_some(generation)
}

pub struct ActiveAttemptGuard;
impl Drop for ActiveAttemptGuard {
    fn drop(&mut self) {
        set_active_attempt_id(0);
    }
}

/// Temporarily bind batch-lane emissions to their request attempt and
/// admission generation.
pub struct BatchAttemptScope {
    pub previous: u64,
    previous_batch_generation: Option<BatchGeneration>,
    admission_generation: Option<BatchGeneration>,
}

impl BatchAttemptScope {
    /// Test/admin convenience lookup. Producer paths should use
    /// [`Self::enter_for_generation`] with their captured token.
    pub fn enter(attempt_id: u64) -> Self {
        Self::enter_with_generation(attempt_id, batch_generation_for(None, attempt_id))
    }

    /// Test/admin convenience lookup. Producer paths should use
    /// [`Self::enter_for_generation`] with their captured token.
    pub fn enter_for(id: &str, attempt_id: u64) -> Self {
        Self::enter_with_generation(attempt_id, batch_generation_for(Some(id), attempt_id))
    }

    /// Bind a sequential singleton without consulting the keyed batch registry.
    ///
    /// A handoff can race with a fresh same-key batch admission; looking up
    /// the key here would bind the old singleton producer to the new owner.
    pub fn enter_singleton(attempt_id: u64) -> Self {
        Self::enter_with_generation(attempt_id, None)
    }

    pub fn enter_for_generation(id: &str, attempt_id: u64, generation: BatchGeneration) -> Self {
        let generation = batch_is_current(id, attempt_id, generation).then_some(generation);
        Self::enter_with_generation(attempt_id, generation)
    }

    /// Rebind an existing outer scope after its keyed batch entry is retired.
    /// Sequential fallback then remains eligible for the singleton claim.
    pub fn rebind_for(&mut self, attempt_id: u64) {
        set_active_attempt_id(attempt_id);
        ACTIVE_BATCH_GENERATION.with(|c| c.set(None));
        self.admission_generation = None;
    }

    pub fn admission_generation(&self) -> Option<BatchGeneration> {
        self.admission_generation
    }

    fn enter_with_generation(attempt_id: u64, generation: Option<BatchGeneration>) -> Self {
        let previous = active_attempt_id();
        let previous_batch_generation = active_batch_generation();
        set_active_attempt_id(attempt_id);
        ACTIVE_BATCH_GENERATION.with(|c| c.set(generation));
        Self {
            previous,
            previous_batch_generation,
            admission_generation: generation,
        }
    }
}

impl Drop for BatchAttemptScope {
    fn drop(&mut self) {
        set_active_attempt_id(self.previous);
        ACTIVE_BATCH_GENERATION.with(|c| c.set(self.previous_batch_generation));
    }
}

// ── singleton terminal control wrappers ──────────────────────────────────

/// Drop guard: clears the active terminal-control transaction.
pub struct TerminalControlGuard;
impl Drop for TerminalControlGuard {
    fn drop(&mut self) {
        clear_terminal_control();
    }
}

/// True if the in-flight request with `req_id` has been aborted for the
/// active attempt or a handoff tombstone that is waiting for adoption.
/// Does not clear the latch (abort remains authoritative through the rest of
/// the turn / handshake).
pub fn check_abort(req_id: &str) -> bool {
    let terminal_cell = terminal_control();
    let terminal = terminal_cell.mu.lock().unwrap();
    if terminal.active.as_ref().is_some_and(|active| {
        active.id == req_id && matches!(active.decision, Some(TerminalControlDecision::Abort))
    }) {
        return true;
    }
    let batch_cell = batch_terminal_control();
    let batch = batch_cell.mu.lock().unwrap();
    batch
        .handoffs
        .iter()
        .any(|(key, handoff)| key.id == req_id && handoff.abort_latched)
}

/// Apply a control message from the stdin reader.
/// - `abort`: accepted throughout generation when `(id, attempt_id)` matches.
/// - `commit`: accepted only after readiness for the matching pair.
/// - transfer tombstones retain either control until singleton adoption.
/// Stale / malformed controls are ignored without mutating state.
pub fn apply_terminal_control(kind: &str, id: &str, attempt_id: u64) {
    let terminal_cell = terminal_control();
    let mut terminal = terminal_cell.mu.lock().unwrap();
    let batch_cell = batch_terminal_control();
    let mut batch = batch_cell.mu.lock().unwrap();
    let key = AttemptKey::new(id, attempt_id);
    if let Some(handoff) = batch.handoffs.get_mut(&key) {
        if apply_handoff_control(&mut terminal, &key, handoff, kind) {
            terminal_cell.cv.notify_all();
            batch_cell.cv.notify_all();
        }
        return;
    }
    let Some(active) = terminal.active.as_mut() else {
        return;
    };
    if active.id != id || active.attempt_id != attempt_id || active.decision.is_some() {
        return;
    }
    match kind {
        "abort" => {
            active.decision = Some(TerminalControlDecision::Abort);
            terminal_cell.cv.notify_all();
        }
        "commit" => {
            if active.ready {
                active.decision = Some(TerminalControlDecision::Commit);
                terminal_cell.cv.notify_all();
            }
            // Early commit before ready: ignore (must not commit).
        }
        _ => {}
    }
}

/// Mark the active transaction ready to accept `commit` for `(id, attempt)`.
/// Returns false if there is no matching active transaction.
pub fn mark_terminal_control_ready(id: &str, attempt_id: u64) -> bool {
    let cell = terminal_control();
    let mut g = cell.mu.lock().unwrap();
    match g.active.as_mut() {
        Some(active) if active.id == id && active.attempt_id == attempt_id => {
            active.ready = true;
            // Abort may already be latched; wake any waiter.
            cell.cv.notify_all();
            true
        }
        _ => false,
    }
}

/// Wait for a matching commit/abort decision on the active transaction.
/// Fail-closed: timeout or missing/mismatched active state → Abort.
pub fn wait_terminal_control_decision(
    id: &str,
    attempt_id: u64,
    timeout: Duration,
) -> ClientTerminalDecision {
    let cell = terminal_control();
    let mut g = cell.mu.lock().unwrap();
    let deadline = Instant::now() + timeout;
    loop {
        let matched = match g.active.as_ref() {
            Some(active) if active.id == id && active.attempt_id == attempt_id => {
                Some(active.decision)
            }
            // No matching active transaction: fail closed.
            _ => return ClientTerminalDecision::Abort,
        };
        if let Some(Some(decision)) = matched {
            return match decision {
                TerminalControlDecision::Commit => ClientTerminalDecision::Commit,
                TerminalControlDecision::Abort => ClientTerminalDecision::Abort,
            };
        }
        let now = Instant::now();
        if now >= deadline {
            // Timeout: latch Abort so subsequent check_abort sees it.
            if let Some(active) = g.active.as_mut() {
                if active.id == id && active.attempt_id == attempt_id && active.decision.is_none() {
                    active.decision = Some(TerminalControlDecision::Abort);
                }
            }
            cell.cv.notify_all();
            return ClientTerminalDecision::Abort;
        }
        let remaining = deadline - now;
        let (guard, wait_res) = cell.cv.wait_timeout(g, remaining).unwrap();
        g = guard;
        if wait_res.timed_out() {
            // Re-check once more under the lock before classifying timeout.
            continue;
        }
    }
}

/// Two-phase correlated terminal handshake.
///
/// After successful producer terminal classification and before tool-call
/// release / assistant-cache insertion / normal `done`:
/// 1. Mark the active `(id, attempt_id)` ready.
/// 2. Emit flushed `commit_ready` = clone of `pending_done` with only
///    `type` changed (`commit_ready`). All other fields (id, attempt_id,
///    finish_reason, usage/timing, route-specific terminals) are preserved.
/// 3. Wait for matching `commit` or `abort` (or bounded timeout → Abort).
///
/// On [`ClientTerminalDecision::Commit`] the caller must emit the same
/// `pending_done` value as `done` (payload-identical after normalizing type).
/// On Abort the caller must not emit normal done / cache / tool release.
///
/// Direct CLI generation auto-acks `commit_ready` on the engine side.
pub fn await_client_terminal_commit(
    stdout: &mut impl std::io::Write,
    id: &str,
    pending_done: &serde_json::Value,
) -> ClientTerminalDecision {
    let attempt_id = active_attempt_id();
    if !mark_terminal_control_ready(id, attempt_id) {
        return ClientTerminalDecision::Abort;
    }
    // Abort may already be latched before ready — still emit commit_ready so
    // the engine observes the handshake edge, then wait returns Abort.
    let mut envelope = pending_done.clone();
    if let Some(obj) = envelope.as_object_mut() {
        obj.insert(
            "type".to_string(),
            serde_json::Value::String("commit_ready".to_string()),
        );
    } else {
        return ClientTerminalDecision::Abort;
    }
    if writeln!(stdout, "{}", envelope).is_err() {
        return ClientTerminalDecision::Abort;
    }
    if stdout.flush().is_err() {
        return ClientTerminalDecision::Abort;
    }
    wait_terminal_control_decision(id, attempt_id, CLIENT_TERMINAL_COMMIT_TIMEOUT)
}

/// Emit the correlated aborted terminal after a [`ClientTerminalDecision::Abort`]
/// from [`await_client_terminal_commit`] (matching `abort`, disconnect, or
/// bounded-timeout fail-closed).
///
/// Writes the canonical correlated pair the client abort drain unblocks on
/// (`hipfire-client` `abort_and_drain_with_rx`): `aborted` plus a `done` with
/// `finish_reason: "aborted"`, both carrying the active attempt id — the same
/// dialect the complete Qwen route emits via `ep_emit_abort`. The
/// wire-terminal claim makes this exactly-once per `(id, attempt_id)`: a
/// repeat call or a racing error path emits nothing.
///
/// Never emits a success `done`, releases no tool calls, and stores no cache.
/// Returns true when the terminal was delivered.
pub fn emit_aborted_terminal_after_abort(
    stdout: &mut impl std::io::Write,
    id: &str,
    completion_tokens: usize,
) -> bool {
    crate::emit::emit_qwen_ar_cancelled(stdout, id, completion_tokens)
}

/// Emit a previously staged `done` envelope after Commit. Payload must be the
/// same value passed to [`await_client_terminal_commit`] as `pending_done`.
/// A matching active attempt may claim only one terminal envelope.
pub fn emit_staged_terminal_done(
    stdout: &mut impl std::io::Write,
    pending_done: &serde_json::Value,
) -> bool {
    emit_staged_terminal_done_outcome(stdout, pending_done).delivered()
}

/// Emit a staged `done` while retaining whether its terminal claim was
/// consumed when the writer fails.
pub fn emit_staged_terminal_done_outcome(
    stdout: &mut impl std::io::Write,
    pending_done: &serde_json::Value,
) -> TerminalEmitOutcome {
    let mut claimed = false;
    if let Some(obj) = pending_done.as_object() {
        if let Some(id) = obj.get("id").and_then(|value| value.as_str()) {
            let attempt_id = obj
                .get("attempt_id")
                .and_then(|value| value.as_u64())
                .unwrap_or_else(active_attempt_id);
            if !claim_wire_terminal(id, attempt_id) {
                return TerminalEmitOutcome::new(false, false);
            }
            claimed = true;
        }
    }
    let delivered = writeln!(stdout, "{}", pending_done).is_ok() && stdout.flush().is_ok();
    TerminalEmitOutcome::new(claimed, delivered)
}

/// Force-answer target request ID, set by the stdin-reader thread on
/// `{type:"force_answer","id":"..."}`. Unlike `abort` (which kills the
/// turn), force-answer asks the decode loop to STOP THINKING and commit
/// to the answer — the model's `<think>` span is force-closed (the same
/// continuation the `max_think_tokens` budget splices) and generation
/// continues. The CLI sends this when a turn is taking too long so the
/// stream produces a real answer instead of the client timing out and
/// terminating mid-think.
pub fn force_answer_for_id() -> &'static Mutex<Option<String>> {
    static CELL: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(None))
}

/// True if the in-flight request `req_id` was asked to force-answer.
/// Clears on match (one-shot).
pub fn check_force_answer(req_id: &str) -> bool {
    let mut g = force_answer_for_id().lock().unwrap();
    if g.as_deref() == Some(req_id) {
        *g = None;
        true
    } else {
        false
    }
}

/// The text spliced into the stream to force-close a `<think>` span (on
/// either the `max_think_tokens` budget OR a CLI force-answer signal),
/// making the model commit to its answer. Default closes the think tag
/// per Qwen's trained post-think format; override with
/// `HIPFIRE_THINK_CONTINUATION` to inject a richer "now produce the
/// answer" nudge (keep it short — it's prepended to the visible answer).
pub fn think_continuation() -> String {
    hipfire_config::developer_var("HIPFIRE_THINK_CONTINUATION")
        .unwrap_or_else(|_| "</think>\n\n".to_string())
}
