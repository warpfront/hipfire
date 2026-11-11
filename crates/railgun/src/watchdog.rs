// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Completion cancel of a persistent program (the "persistent watchdog").
//!
//! A persistent window's kernels may legitimately not finish on their own (a
//! resident server kernel). Its completion lease therefore cannot be the
//! queue's single global wait: a timeout there must *stop the program*, and
//! must never report quiescence while workgroups may still be resident. The
//! lease of a persistent window runs three steps, in order:
//!
//! 1. **wait** [`PersistentWatchdog::budget`] for the window's end-of-pipe
//!    completion signal ([`WindowQueue::wait_timeout`]; a timeout there tears
//!    nothing down);
//! 2. **poison**: raise the declared [`PoisonCell`] — a 4-byte cell of the
//!    program's own `EventCells` resource, written by the host and polled by
//!    every worker at its next poll ([`WindowQueue::raise_poison`], which must
//!    not wait on the stuck queue) — then wait [`PersistentWatchdog::grace`].
//!    The signal firing proves every workgroup retired:
//!    [`FailureKind::Poisoned`], [`Quiescence::Proven`], outputs invalid;
//! 3. **escalate**: [`WindowQueue::teardown`] (queue inactivation) and observe
//!    up to [`PersistentWatchdog::escalation_grace`] for retirement evidence:
//!    [`FailureKind::TornDown`] with the [`Quiescence`] the queue's teardown
//!    *proves* — `Unknown` unless it observed retirement. The queue (not this
//!    crate) quarantines its device when it cannot prove it.
//!
//! The poison value is set-only: nothing ever clears it during a window, so a
//! worker that polls late still observes it (the OR-poison contract). The
//! owner zeroes the cell before the next submit of the same program.
//!
//! Non-persistent windows never reach this module: their lease is the queue's
//! own `wait()` with its own timeout, unchanged.

use std::time::Duration;

use crate::window::{QueueFailure, Quiescence, TimedWait, WindowQueue};
use crate::{Binding, ResourceId, ResourceKind, EVENT_CELL_BYTES};

/// What a failed lease means. Typed so a caller never parses a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FailureKind {
    /// Ordinary queue failure (submit refused, launch error, queue fault, the
    /// queue's own bounded wait expiring): today's semantics.
    Queue,
    /// A persistent window outlived its budget, the poison cell was raised and
    /// every workgroup retired within the grace. The window is retired
    /// ([`Quiescence::Proven`]); its outputs are invalid.
    Poisoned,
    /// A persistent window did not retire after the poison: its queue was torn
    /// down. The queue refuses every further submit; the failure's
    /// [`Quiescence`] is what the teardown proved.
    TornDown,
    /// The device is quarantined (an earlier teardown could not prove
    /// retirement): the submit was refused before anything was written.
    DeviceQuarantined,
}

/// A 4-byte poison cell inside one declared `EventCells` resource of the
/// window's program. Non-zero means "exit": workers poll it with acquire at
/// device scope; the host only ever stores `value`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PoisonCell {
    pub resource: ResourceId,
    /// Byte offset of the cell inside `resource` (4-byte aligned).
    pub offset: u64,
    /// The non-zero word the host stores.
    pub value: u32,
}

/// The completion budget of a persistent window (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PersistentWatchdog {
    /// How long the window may run before it is poisoned.
    pub budget: Duration,
    /// How long after the poison every worker has to retire.
    pub grace: Duration,
    /// How long the escalation observes for retirement evidence after the
    /// queue teardown.
    pub escalation_grace: Duration,
    pub poison: PoisonCell,
}

impl PersistentWatchdog {
    /// Check the declaration against `program`: nonzero durations and value, a
    /// 4-byte-aligned cell inside a resource declared `EventCells` (whose
    /// accesses the event contract already limits to atomics and acquire
    /// loads, so a device store cannot clear it), inside its declared cells and
    /// its resource's extent.
    pub fn validate(&self, program: &crate::kernel::KernelProgram) -> Result<(), String> {
        validate_timings(self.budget, self.grace, self.escalation_grace)?;
        validate_value(self.poison.value)?;
        let r = self.poison.resource;
        if self.poison.offset % 4 != 0 {
            return Err(format!(
                "poison cell offset {} is not 4-byte aligned",
                self.poison.offset
            ));
        }
        let Some(ResourceKind::EventCells { cells }) = program.kinds.get(&r) else {
            return Err(format!(
                "poison resource {} is not declared EventCells",
                r.0
            ));
        };
        let end = self
            .poison
            .offset
            .checked_add(4)
            .ok_or_else(|| "poison cell offset overflows".to_owned())?;
        let cell_bytes = u64::from(*cells) * EVENT_CELL_BYTES;
        if end > cell_bytes {
            return Err(format!(
                "poison cell [{}, {end}) lies outside the {cells} declared event cells of resource {} ({cell_bytes} bytes)",
                self.poison.offset, r.0
            ));
        }
        let resource = program
            .resources
            .get(r.0 as usize)
            .ok_or_else(|| format!("poison resource {} is not a program resource", r.0))?;
        if end > resource.bytes {
            return Err(format!(
                "poison cell [{}, {end}) exceeds resource {} ({} bytes)",
                self.poison.offset, r.0, resource.bytes
            ));
        }
        Ok(())
    }

    /// The device address of the cell under the live `bindings` (one per
    /// program resource, in id order), checked inside its allocation.
    pub fn resolve(&self, bindings: &[Binding]) -> Result<ResolvedWatchdog, String> {
        let b = bindings
            .get(self.poison.resource.0 as usize)
            .ok_or_else(|| format!("no binding for poison resource {}", self.poison.resource.0))?;
        let end = self
            .poison
            .offset
            .checked_add(4)
            .ok_or_else(|| "poison cell offset overflows".to_owned())?;
        if end > b.bytes {
            return Err(format!(
                "poison cell [{}, {end}) exceeds the {}-byte allocation of resource {}",
                self.poison.offset, b.bytes, self.poison.resource.0
            ));
        }
        let address = b
            .base
            .checked_add(self.poison.offset)
            .ok_or_else(|| "poison cell address overflows".to_owned())?;
        ResolvedWatchdog::new(
            self.budget,
            self.grace,
            self.escalation_grace,
            address,
            self.poison.value,
        )
    }
}

fn validate_timings(budget: Duration, grace: Duration, escalation: Duration) -> Result<(), String> {
    for (name, d) in [
        ("budget", budget),
        ("grace", grace),
        ("escalation_grace", escalation),
    ] {
        if d.is_zero() {
            return Err(format!("persistent watchdog {name} is zero"));
        }
    }
    Ok(())
}

fn validate_value(value: u32) -> Result<(), String> {
    if value == 0 {
        return Err("poison value is zero: the cell's cleared state".to_owned());
    }
    Ok(())
}

/// A [`PersistentWatchdog`] with the poison cell resolved to a device address:
/// what the lease runs against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedWatchdog {
    pub budget: Duration,
    pub grace: Duration,
    pub escalation_grace: Duration,
    pub poison_address: u64,
    pub poison_value: u32,
}

impl ResolvedWatchdog {
    /// Nonzero durations and value, a non-null 4-byte-aligned address.
    pub fn new(
        budget: Duration,
        grace: Duration,
        escalation_grace: Duration,
        poison_address: u64,
        poison_value: u32,
    ) -> Result<Self, String> {
        validate_timings(budget, grace, escalation_grace)?;
        validate_value(poison_value)?;
        if poison_address == 0 || poison_address % 4 != 0 {
            return Err(format!(
                "poison address {poison_address:#x} is null or not 4-byte aligned"
            ));
        }
        Ok(Self {
            budget,
            grace,
            escalation_grace,
            poison_address,
            poison_value,
        })
    }
}

/// What a queue's [`WindowQueue::teardown`] proved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Teardown {
    /// [`Quiescence::Proven`] only when the teardown observed the window's
    /// workgroups retire; anything less is [`Quiescence::Unknown`].
    pub quiescence: Quiescence,
    pub message: String,
}

/// The persistent completion lease on `queue` (module docs). `Ok` only when
/// the window completed on its own within `watchdog.budget`; every other
/// outcome is a [`QueueFailure`] whose `quiescence` is what is *proven*.
pub fn run_watchdog<Q: WindowQueue + ?Sized>(
    queue: &mut Q,
    watchdog: &ResolvedWatchdog,
) -> Result<(), QueueFailure> {
    if let TimedWait::Done(outcome) = queue.wait_timeout(watchdog.budget) {
        return outcome;
    }
    let poison_failure = match queue.raise_poison(watchdog.poison_address, watchdog.poison_value) {
        Ok(()) => match queue.wait_timeout(watchdog.grace) {
            TimedWait::Done(Ok(())) => {
                return Err(QueueFailure::new(
                    format!(
                        "persistent window exceeded its {:?} budget; poison cell {:#x} raised and every workgroup retired within {:?} (outputs invalid)",
                        watchdog.budget, watchdog.poison_address, watchdog.grace
                    ),
                    Quiescence::Proven,
                )
                .with_kind(FailureKind::Poisoned));
            }
            // The queue's own failure (fault, proven or unknown teardown) is
            // what is known; it stands as reported.
            TimedWait::Done(Err(failure)) => return Err(failure),
            TimedWait::Pending => format!(
                "poison cell {:#x} raised but the window did not retire within {:?}",
                watchdog.poison_address, watchdog.grace
            ),
        },
        Err(failure) => format!(
            "poison cell {:#x} could not be raised ({})",
            watchdog.poison_address, failure.message
        ),
    };
    let teardown = queue.teardown(watchdog.escalation_grace);
    Err(QueueFailure::new(
        format!(
            "persistent window exceeded its {:?} budget; {poison_failure}; queue torn down: {}",
            watchdog.budget, teardown.message
        ),
        teardown.quiescence,
    )
    .with_kind(FailureKind::TornDown))
}
