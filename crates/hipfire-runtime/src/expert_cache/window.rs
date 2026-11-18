// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Execution-window bookkeeping and the retirement cadence.
//!
//! A *window* is one publication epoch of the table image: every consumer that reads the image
//! (lanes, grouped rows, draft heads, verify and replay submissions) registers before it is
//! enqueued on a queue and joins when that queue proves the consumer retired. The image may only
//! be retired (and so its slots withdrawn) once the window is closed to new consumers and every
//! registered consumer joined. The book is pure CPU state with capacities fixed at construction;
//! device completion proof stays with the caller (railgun fences, S1b queue adapters).

use super::{CacheError, WindowConsumerPhase};

/// Handle of one registered consumer of the open window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerId(pub(crate) u32);

/// What the retirement of a cancelled window proved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelOutcome {
    /// Every registered consumer's queue proved retirement: the window closes normally.
    Retired,
    /// A consumer may still be running: the book poisons itself, so no later window opens and the
    /// image is never retired (the owner quarantines everything it can reach).
    Unknown,
}

/// What a finished window joined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowReceipt {
    /// The window's index (counted from 1).
    pub epoch: u64,
    /// Consumers registered and joined.
    pub consumers: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Idle,
    Open,
    Closing,
    Poisoned,
}

#[derive(Clone, Copy, Debug)]
struct Consumer {
    queue: u16,
    phase: WindowConsumerPhase,
    lane: u32,
    joined: bool,
}

/// The book of one cache's current window.
#[derive(Debug)]
pub struct WindowBook {
    state: State,
    epoch: u64,
    consumers: Vec<Consumer>,
    capacity: usize,
    registered: Vec<u32>,
    joined: Vec<u32>,
}

impl WindowBook {
    /// Room for `max_consumers` consumers per window over `queues` queues.
    pub fn new(max_consumers: usize, queues: usize) -> Self {
        Self {
            state: State::Idle,
            epoch: 0,
            consumers: Vec::with_capacity(max_consumers),
            capacity: max_consumers,
            registered: vec![0; queues],
            joined: vec![0; queues],
        }
    }

    pub fn is_idle(&self) -> bool {
        self.state == State::Idle
    }

    pub fn is_open(&self) -> bool {
        self.state == State::Open
    }

    pub fn is_closing(&self) -> bool {
        self.state == State::Closing
    }

    pub fn is_poisoned(&self) -> bool {
        self.state == State::Poisoned
    }

    /// Windows begun so far (the epoch of the open or last window).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn max_consumers(&self) -> usize {
        self.capacity
    }

    pub fn queues(&self) -> usize {
        self.registered.len()
    }

    /// Consumers of the current window.
    pub fn consumers(&self) -> usize {
        self.consumers.len()
    }

    /// Registered consumers that have not joined yet.
    pub fn outstanding(&self) -> usize {
        self.consumers.iter().filter(|c| !c.joined).count()
    }

    /// `(registered, joined)` of one queue in the current (or last) window.
    pub fn queue_counts(&self, queue: usize) -> Option<(u32, u32)> {
        Some((*self.registered.get(queue)?, *self.joined.get(queue)?))
    }

    /// The `(queue, phase, lane)` a consumer registered with.
    pub fn consumer(&self, id: ConsumerId) -> Option<(u16, WindowConsumerPhase, u32)> {
        let c = self.consumers.get(id.0 as usize)?;
        Some((c.queue, c.phase, c.lane))
    }

    /// Open a window: only an idle book (no open window, nothing to join) admits one.
    pub fn begin(&mut self) -> Result<u64, CacheError> {
        match self.state {
            State::Idle => {}
            State::Poisoned => return Err(CacheError::WindowPoisoned),
            State::Open | State::Closing => return Err(CacheError::WindowBusy),
        }
        self.epoch += 1;
        self.consumers.clear();
        self.registered.iter_mut().for_each(|n| *n = 0);
        self.joined.iter_mut().for_each(|n| *n = 0);
        self.state = State::Open;
        Ok(self.epoch)
    }

    /// Register one consumer on `queue` BEFORE it is enqueued. Refused once the window closed (a
    /// closing window never admits a new entrant) and past the preallocated capacity.
    pub fn register(
        &mut self,
        queue: usize,
        phase: WindowConsumerPhase,
        lane: u32,
    ) -> Result<ConsumerId, CacheError> {
        match self.state {
            State::Open => {}
            State::Idle => return Err(CacheError::WindowNotOpen),
            State::Closing => return Err(CacheError::WindowClosed),
            State::Poisoned => return Err(CacheError::WindowPoisoned),
        }
        if queue >= self.registered.len() {
            return Err(CacheError::QueueOutOfRange {
                queue,
                queues: self.registered.len(),
            });
        }
        if self.consumers.len() >= self.capacity {
            return Err(CacheError::ConsumerOverflow {
                capacity: self.capacity,
            });
        }
        let id = ConsumerId(self.consumers.len() as u32);
        self.consumers.push(Consumer {
            queue: queue as u16,
            phase,
            lane,
            joined: false,
        });
        self.registered[queue] += 1;
        Ok(id)
    }

    /// The consumer's queue proved it retired (real completion proof, from the queue adapter).
    pub fn join(&mut self, id: ConsumerId) -> Result<(), CacheError> {
        match self.state {
            State::Open | State::Closing => {}
            State::Idle => return Err(CacheError::WindowNotOpen),
            State::Poisoned => return Err(CacheError::WindowPoisoned),
        }
        let Some(c) = self.consumers.get_mut(id.0 as usize) else {
            return Err(CacheError::UnknownConsumer(id.0));
        };
        if c.joined {
            return Err(CacheError::ConsumerAlreadyJoined(id.0));
        }
        c.joined = true;
        self.joined[c.queue as usize] += 1;
        Ok(())
    }

    /// Close the window to new consumers (publication may now wait for the join).
    pub fn close(&mut self) -> Result<(), CacheError> {
        match self.state {
            State::Open => {
                self.state = State::Closing;
                Ok(())
            }
            State::Closing => Ok(()),
            State::Idle => Err(CacheError::WindowNotOpen),
            State::Poisoned => Err(CacheError::WindowPoisoned),
        }
    }

    /// Whether every registered consumer of every queue joined.
    pub fn joined(&self) -> bool {
        self.consumers.iter().all(|c| c.joined)
            && self
                .registered
                .iter()
                .zip(&self.joined)
                .all(|(r, j)| r == j)
    }

    /// Finish the closed window: it must have joined every consumer.
    pub fn finish(&mut self) -> Result<WindowReceipt, CacheError> {
        match self.state {
            State::Closing => {}
            State::Open => return Err(CacheError::WindowNotClosed),
            State::Idle => return Err(CacheError::WindowNotOpen),
            State::Poisoned => return Err(CacheError::WindowPoisoned),
        }
        let outstanding = self.outstanding();
        if outstanding != 0 {
            return Err(CacheError::ConsumersOutstanding { outstanding });
        }
        self.state = State::Idle;
        Ok(WindowReceipt {
            epoch: self.epoch,
            consumers: self.consumers.len() as u32,
        })
    }

    /// Cancel the window (timeout, unload, error). With every queue proven retired the window
    /// closes and the book is idle again; with unknown completion it is poisoned for good.
    pub fn cancel(&mut self, outcome: CancelOutcome) -> Result<(), CacheError> {
        match (self.state, outcome) {
            (State::Poisoned, _) => Err(CacheError::WindowPoisoned),
            (State::Idle, _) => Err(CacheError::WindowNotOpen),
            (_, CancelOutcome::Retired) => {
                self.state = State::Idle;
                Ok(())
            }
            (_, CancelOutcome::Unknown) => {
                self.state = State::Poisoned;
                Ok(())
            }
        }
    }
}

/// Default of [`RetireCadence`]'s group cap (D2's `RETIRE_MAX_WINDOWS`).
pub const DEFAULT_GROUP_MAX_WINDOWS: u32 = 8;

/// The retirement cadence: which window ends close the accumulated group.
///
/// Every window reports `routed` (routed rows x top-k, counted before anything rank-local can
/// change it). The group retires at a window end when, after adding the window,
///
/// * its routed ids reach `retire_ids`, or
/// * it holds `max_windows` windows, or
/// * the next window could overflow the usage log: routed ids so far plus the LARGEST window
///   seen exceed `log_ids`.
#[derive(Clone, Debug)]
pub struct RetireCadence {
    retire_ids: u64,
    log_ids: u64,
    max_windows: u32,
    windows: u32,
    routed: u64,
    largest: u64,
}

impl RetireCadence {
    pub fn new(retire_ids: u64, log_ids: u64) -> Self {
        Self::with_max_windows(retire_ids, log_ids, DEFAULT_GROUP_MAX_WINDOWS)
    }

    pub fn with_max_windows(retire_ids: u64, log_ids: u64, max_windows: u32) -> Self {
        Self {
            retire_ids,
            log_ids,
            max_windows: max_windows.max(1),
            windows: 0,
            routed: 0,
            largest: 0,
        }
    }

    /// One window ended with `routed` routed ids. `Some(n)`: retire now, the group is `n`
    /// windows (and the group is empty again); `None`: keep accumulating.
    pub fn window_end(&mut self, routed: u64) -> Option<u32> {
        self.windows += 1;
        self.routed += routed;
        self.largest = self.largest.max(routed);
        let retire = self.routed >= self.retire_ids
            || self.windows >= self.max_windows
            || self.routed + self.largest > self.log_ids;
        retire.then(|| {
            let windows = self.windows;
            self.windows = 0;
            self.routed = 0;
            windows
        })
    }

    /// Windows accumulated since the last retirement.
    pub fn pending(&self) -> u32 {
        self.windows
    }

    /// Drop the accumulated group (the cache froze or was released).
    pub fn clear(&mut self) {
        self.windows = 0;
        self.routed = 0;
    }
}
