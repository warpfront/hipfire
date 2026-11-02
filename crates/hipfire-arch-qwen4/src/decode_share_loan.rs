// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The donor-tail loan controller of decode-share (`fixed` / `on` modes).
//!
//! While a long prefill runs, the forward scratch lives on the donor tail (a
//! contiguous run of cold resident experts' VRAM). The controller is the one
//! state machine that decides when those bytes may be overwritten and when
//! they must be whole again:
//!
//! ```text
//!   Resident --acquire--> Dirty --require_resident / teardown / (SYNC=1) finish_prompt--> Restoring --wait--> Resident
//!        any state --(error)--> recovery (device drain + blocking copy) --> Resident | Poisoned
//! ```
//!
//! A prompt that ends `Ok` leaves the tail `Dirty` and enqueues nothing: the
//! first token has already come from the prefill head, and a following prompt
//! tile continues the same borrow (back-to-back prefills share one epoch, no
//! restore between them). The next decode / verify / lane / sub-512-row call
//! outside a prompt (or teardown) enqueues the whole restore and host-waits
//! for it, so the second-token latency after a 16K prompt includes the full
//! restore (~88 ms on R9700: 4.8 GB at ~57 GB/s). With
//! `HIPFIRE_QWEN4_DECODE_SHARE_SYNC=1` the restore is enqueued and waited at
//! prompt end instead.
//!
//! It is pure: every device effect goes through [`LoanOps`], so each
//! transition is unit-tested on the CPU with a recording fake. The forward's
//! `GpuLoanOps` (gpu_forward.rs) is the real implementation: the compute
//! stream, one completion event and the pinned mirror.
//!
//! Invariants:
//! * No byte of the donor tail is written before the state is `Dirty`.
//! * `Resident` is entered only after the restore copy completed (waited, or
//!   a blocking recovery copy), never on a promise.
//! * A failed recovery, or any need to restore under a stream capture,
//!   poisons the loan: every later call fails and names the reason.

use std::time::Instant;

/// Device effects of the loan protocol. Every method is one real operation;
/// errors are plain text (the forward wraps them).
pub(crate) trait LoanOps {
    /// Whether the compute stream is being captured (a capture can neither
    /// run a restore nor host-wait on one).
    fn capturing(&mut self) -> Result<bool, String>;
    /// Enqueue a zero fill of `bytes` at `offset` into the donor tail on the
    /// compute stream.
    fn memset_zero(&mut self, offset: u64, bytes: u64) -> Result<(), String>;
    /// Enqueue the restore of the first `bytes` of the tail from its mirror:
    /// an async H2D copy on the compute stream, then `done` recorded on the
    /// compute stream.
    fn enqueue_restore(&mut self, bytes: u64) -> Result<(), String>;
    /// Host-wait for the enqueued restore's `done` event and check the async
    /// error state.
    fn wait_restore(&mut self) -> Result<(), String>;
    /// Drain every queue of the device.
    fn device_sync(&mut self) -> Result<(), String>;
    /// Blocking H2D copy of the first `bytes` of the tail from its mirror.
    fn blocking_restore(&mut self, bytes: u64) -> Result<(), String>;
    /// The `HIPFIRE_QWEN4_DECODE_SHARE_VERIFY` oracle over the first `bytes`
    /// (a no-op unless the knob is set).
    fn verify(&mut self, bytes: u64) -> Result<(), String>;
    /// Rebuild every borrowed weight shadow in place (the first
    /// `shadow_bytes` of the donor tail) on the compute stream. Called once
    /// per epoch, at its first acquire, after the scratch zero fill; a
    /// controller without shadows never calls it.
    fn rebuild_shadows(&mut self) -> Result<(), String> {
        Ok(())
    }
    /// The byte oracle of the rebuilt weight shadows (a no-op unless
    /// `HIPFIRE_QWEN4_DECODE_SHARE_VERIFY=1`, and then only after the first
    /// rebuild of the load): called right after a successful rebuild. An `Err`
    /// names the first differing shadow member/offset and poisons the loan.
    fn verify_shadows(&mut self) -> Result<(), String> {
        Ok(())
    }
    /// Device time of the epoch's shadow rebuild in milliseconds, if it was
    /// measured (only under the epoch log knob); queried after the restore
    /// completed, so the events are done.
    fn rebuild_ms(&mut self) -> Option<f64> {
        None
    }
}

/// Where the donor tail stands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LoanState {
    /// The tail holds its experts.
    Resident,
    /// Scratch may have overwritten the first `dirty_bytes`; the first
    /// `zeroed_bytes` were zero-filled for this epoch.
    Dirty {
        epoch: u64,
        dirty_bytes: u64,
        zeroed_bytes: u64,
    },
    /// The restore of `bytes` is enqueued (the `done` event lives in the ops).
    Restoring {
        epoch: u64,
        bytes: u64,
        zeroed_bytes: u64,
    },
    /// The tail's contents are unknown; nothing may run.
    Poisoned(String),
}

/// Receipt counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LoanCounters {
    pub epochs: u64,
    pub tiles: u64,
    pub restores: u64,
    pub restored_bytes: u64,
    pub recoveries: u64,
    pub zeroed_bytes: u64,
    /// Epochs whose first acquire rebuilt the borrowed weight shadows.
    pub shadow_rebuilds: u64,
    /// Time actually blocked in the host wait for a restore.
    pub wait_ns: u64,
}

pub(crate) struct LoanController {
    state: LoanState,
    epoch: u64,
    epoch_tiles: u64,
    /// `HIPFIRE_QWEN4_DECODE_SHARE_SYNC`: host-wait right after each enqueue.
    sync_restore: bool,
    /// `HIPFIRE_QWEN4_DECODE_SHARE_LOG`: one line per epoch.
    log_epochs: bool,
    /// Bytes at the start of the tail that hold the borrowed weight shadows
    /// (0: resident shadows). Every epoch rebuilds them at its first acquire;
    /// the scratch zero fill starts after them.
    shadow_bytes: u64,
    counters: LoanCounters,
}

const MIB: u64 = 1 << 20;

impl LoanController {
    pub(crate) fn new(sync_restore: bool, log_epochs: bool) -> Self {
        Self {
            state: LoanState::Resident,
            epoch: 0,
            epoch_tiles: 0,
            sync_restore,
            shadow_bytes: 0,
            log_epochs,
            counters: LoanCounters::default(),
        }
    }

    /// A controller whose donor tail starts with `shadow_bytes` of borrowed
    /// weight shadows: the scratch zero fill skips them and the first acquire
    /// of every epoch rebuilds them ([`LoanOps::rebuild_shadows`]).
    pub(crate) fn with_shadow_bytes(mut self, shadow_bytes: u64) -> Self {
        self.shadow_bytes = shadow_bytes;
        self
    }

    #[cfg(test)]
    pub(crate) fn state(&self) -> &LoanState {
        &self.state
    }

    pub(crate) fn is_resident(&self) -> bool {
        self.state == LoanState::Resident
    }

    /// Bytes of the donor tail the current borrow epoch has dirtied (the
    /// largest class span of its tiles so far); `None` outside an epoch.
    pub(crate) fn dirty_extent(&self) -> Option<u64> {
        match self.state {
            LoanState::Dirty { dirty_bytes, .. } => Some(dirty_bytes),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn counters(&self) -> LoanCounters {
        self.counters
    }

    fn poison_error(reason: &str) -> String {
        format!("decode-share donor tail is poisoned ({reason}); reload the model")
    }

    /// `Err` naming the reason when poisoned.
    pub(crate) fn check(&self) -> Result<(), String> {
        match &self.state {
            LoanState::Poisoned(reason) => Err(Self::poison_error(reason)),
            _ => Ok(()),
        }
    }

    fn poison(&mut self, reason: String) -> String {
        let error = Self::poison_error(&reason);
        self.state = LoanState::Poisoned(reason);
        error
    }

    /// Bytes a recovery must copy back from the current state.
    fn pending_bytes(&self) -> u64 {
        match self.state {
            LoanState::Dirty { dirty_bytes, .. } => dirty_bytes,
            LoanState::Restoring { bytes, .. } => bytes,
            _ => 0,
        }
    }

    /// Recovery: drain every queue, copy `bytes` back with a blocking copy,
    /// drain again, and only then mark the tail `Resident`. Under a capture,
    /// or if any step fails, the loan is poisoned. Returns `original` with the
    /// recovery failure appended, if any.
    fn recover(&mut self, ops: &mut dyn LoanOps, bytes: u64, original: String) -> String {
        self.counters.recoveries += 1;
        match ops.capturing() {
            Ok(false) => {}
            Ok(true) => {
                let why = self.poison(format!(
                    "{original}; the donor tail needs a restore that a stream capture cannot run"
                ));
                return format!("{original}; {why}");
            }
            Err(error) => {
                let why = self.poison(format!("{original}; capture query failed: {error}"));
                return format!("{original}; {why}");
            }
        }
        let outcome = ops
            .device_sync()
            .and_then(|()| ops.blocking_restore(bytes))
            .and_then(|()| ops.device_sync());
        match outcome {
            Ok(()) => {
                self.counters.restored_bytes += bytes;
                self.state = LoanState::Resident;
                original
            }
            Err(error) => {
                let why = self.poison(format!("{original}; recovery failed: {error}"));
                format!("{original}; {why}")
            }
        }
    }

    /// Enqueue the restore of `bytes` and move to `Restoring`.
    fn enqueue(
        &mut self,
        ops: &mut dyn LoanOps,
        bytes: u64,
        zeroed_bytes: u64,
    ) -> Result<(), String> {
        match ops.enqueue_restore(bytes) {
            Ok(()) => {
                self.counters.restores += 1;
                self.counters.restored_bytes += bytes;
                self.state = LoanState::Restoring {
                    epoch: self.epoch,
                    bytes,
                    zeroed_bytes,
                };
                Ok(())
            }
            Err(error) => Err(self.recover(ops, bytes, format!("enqueue donor restore: {error}"))),
        }
    }

    /// Host-wait the enqueued restore: `Restoring` -> `Resident`.
    fn wait(&mut self, ops: &mut dyn LoanOps) -> Result<(), String> {
        let LoanState::Restoring {
            epoch,
            bytes,
            zeroed_bytes,
        } = self.state
        else {
            return Ok(());
        };
        let started = Instant::now();
        let waited = ops.wait_restore();
        let wait_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.counters.wait_ns = self.counters.wait_ns.saturating_add(wait_ns);
        if let Err(error) = waited {
            return Err(self.recover(ops, bytes, format!("wait donor restore: {error}")));
        }
        if let Err(error) = ops.verify(bytes) {
            let reason = format!("donor tail verification after restore: {error}");
            return Err(self.poison(reason));
        }
        self.state = LoanState::Resident;
        if self.log_epochs {
            let shadows = if self.shadow_bytes == 0 {
                String::new()
            } else {
                match ops.rebuild_ms() {
                    Some(ms) => format!(
                        ", shadows {} MiB rebuilt in {ms:.3} ms",
                        self.shadow_bytes / MIB
                    ),
                    None => format!(", shadows {} MiB rebuilt", self.shadow_bytes / MIB),
                }
            };
            eprintln!(
                "  decode-share epoch {epoch}: tiles {}, class span {} MiB, zeroed {} MiB, restore {} MiB{shadows}, wait {:.3} ms",
                self.epoch_tiles,
                bytes / MIB,
                zeroed_bytes / MIB,
                bytes / MIB,
                wait_ns as f64 / 1.0e6
            );
        }
        Ok(())
    }

    /// A borrowed prompt tile whose scratch spans `span` bytes of the tail is
    /// about to run. Writes nothing unless the state is (made) `Dirty` first;
    /// zero-fills only the growth beyond what this epoch already zeroed.
    pub(crate) fn acquire(&mut self, ops: &mut dyn LoanOps, span: u64) -> Result<(), String> {
        self.check()?;
        match ops.capturing() {
            Ok(false) => {}
            Ok(true) => {
                return Err(
                    "a decode-share prompt tile cannot run inside a stream capture (its scratch \
                     overwrites donor experts that need a restore)"
                        .to_string(),
                )
            }
            Err(error) => return Err(format!("decode-share capture query: {error}")),
        }
        if matches!(self.state, LoanState::Restoring { .. }) {
            self.wait(ops)?;
        }
        let mut fresh_epoch = false;
        if self.state == LoanState::Resident {
            self.epoch += 1;
            self.epoch_tiles = 0;
            self.counters.epochs += 1;
            fresh_epoch = true;
            // The borrowed weight shadows (if any) fill the start of the tail:
            // the scratch zero fill below begins after them.
            self.state = LoanState::Dirty {
                epoch: self.epoch,
                dirty_bytes: 0,
                zeroed_bytes: self.shadow_bytes,
            };
        }
        let LoanState::Dirty {
            epoch,
            dirty_bytes,
            zeroed_bytes,
        } = self.state
        else {
            return Err(format!("decode-share acquire in state {:?}", self.state));
        };
        // The write may land as soon as it is enqueued: record the extent first.
        let dirty_bytes = dirty_bytes.max(span);
        self.state = LoanState::Dirty {
            epoch,
            dirty_bytes,
            zeroed_bytes,
        };
        if span > zeroed_bytes {
            // Donor bytes are arbitrary bit patterns (NaNs among them): padded
            // rows of a scratch must read zeros, as in a zero-initialized allocation.
            if let Err(error) = ops.memset_zero(zeroed_bytes, span - zeroed_bytes) {
                return Err(self.recover(ops, dirty_bytes, format!("zero donor scratch: {error}")));
            }
            self.counters.zeroed_bytes += span - zeroed_bytes;
            self.state = LoanState::Dirty {
                epoch,
                dirty_bytes,
                zeroed_bytes: span,
            };
        }
        // The first acquire of an epoch rebuilds the borrowed weight shadows
        // (stream-ordered after the zero fill; the regions are disjoint).
        if fresh_epoch && self.shadow_bytes > 0 {
            if let Err(error) = ops.rebuild_shadows() {
                return Err(self.recover(
                    ops,
                    dirty_bytes,
                    format!("rebuild donor-tail weight shadows: {error}"),
                ));
            }
            if let Err(error) = ops.verify_shadows() {
                return Err(self.poison(format!("borrowed weight shadow oracle: {error}")));
            }
            self.counters.shadow_rebuilds += 1;
        }
        self.epoch_tiles += 1;
        self.counters.tiles += 1;
        Ok(())
    }

    /// A call that runs on the resident bank and the Resident runtimes
    /// (decode, verify, lanes, a tile below 512 rows) is about to start: the
    /// donors must be whole. Enqueues the restore if dirty, then host-waits.
    pub(crate) fn require_resident(&mut self, ops: &mut dyn LoanOps) -> Result<(), String> {
        self.check()?;
        if let LoanState::Dirty {
            dirty_bytes,
            zeroed_bytes,
            ..
        } = self.state
        {
            match ops.capturing() {
                Ok(false) => {}
                Ok(true) | Err(_) => {
                    return Err(self.poison(
                        "a resident call under a stream capture found the donor tail dirty"
                            .to_string(),
                    ))
                }
            }
            self.enqueue(ops, dirty_bytes, zeroed_bytes)?;
        }
        self.wait(ops)
    }

    /// After every tile with the resident bank back in place: an `Err` tile
    /// must not leave dirty weights behind even if the caller abandons the
    /// prompt. Returns `Err` only if the recovery itself failed.
    pub(crate) fn after_tile(&mut self, ops: &mut dyn LoanOps, ok: bool) -> Result<(), String> {
        if ok {
            return Ok(());
        }
        let bytes = self.pending_bytes();
        if matches!(
            self.state,
            LoanState::Dirty { .. } | LoanState::Restoring { .. }
        ) {
            let text = self.recover(ops, bytes, "decode-share tile failed".to_string());
            if let LoanState::Poisoned(_) = self.state {
                return Err(text);
            }
        }
        Ok(())
    }

    /// The outermost whole-prompt entry ended (`ok`: it returned `Ok`). A dirty
    /// tail stays dirty (nothing is enqueued): the next prompt tile continues
    /// the same borrow, and the next resident call restores it. With
    /// `sync_restore` the restore is enqueued and waited here. An abandoned
    /// prompt takes the blocking recovery.
    pub(crate) fn finish_prompt(&mut self, ops: &mut dyn LoanOps, ok: bool) -> Result<(), String> {
        let LoanState::Dirty {
            dirty_bytes,
            zeroed_bytes,
            ..
        } = self.state
        else {
            return Ok(());
        };
        if !ok {
            let text = self.recover(ops, dirty_bytes, "decode-share prompt failed".to_string());
            return match self.state {
                LoanState::Poisoned(_) => Err(text),
                _ => Ok(()),
            };
        }
        match ops.capturing() {
            Ok(false) => {}
            Ok(true) | Err(_) => {
                return Err(self.poison(
                    "a decode-share prompt ended under a stream capture with a dirty donor tail"
                        .to_string(),
                ))
            }
        }
        // A prompt that ends well leaves the tail dirty: the first token comes
        // from the prompt's head before any restore, the next prompt tile
        // continues the same borrow (no restore between back-to-back
        // prefills), and the next decode, verify, lane or sub-512-row call
        // enqueues and waits for the restore (`require_resident`). The sync
        // knob restores here instead, for A/B measurement.
        if self.sync_restore {
            self.enqueue(ops, dirty_bytes, zeroed_bytes)?;
            self.wait(ops)?;
        }
        Ok(())
    }

    /// Teardown: the donors must be whole before anything the lease owns is
    /// freed. `Err` means the restore could not be proven complete and the
    /// caller must leak the lease and the restore resources, not free them
    /// under a possibly pending DMA.
    pub(crate) fn teardown(&mut self, ops: &mut dyn LoanOps) -> Result<(), String> {
        let outcome = self.require_resident(ops);
        // A recovered tail is whole (a blocking copy completed after a drain):
        // safe to release even though the fast path failed.
        if self.is_resident() {
            Ok(())
        } else {
            outcome.and(Err(format!(
                "decode-share teardown left the donor tail in state {:?}",
                self.state
            )))
        }
    }

    /// One summary line for the free log.
    pub(crate) fn summary(&self) -> String {
        let counters = self.counters;
        let shadows = if self.shadow_bytes == 0 {
            String::new()
        } else {
            format!(
                ", {} weight-shadow rebuilds ({} MiB each)",
                counters.shadow_rebuilds,
                self.shadow_bytes / MIB
            )
        };
        format!(
            "{} epochs, {} prompt tiles, {} restores ({} MiB), {} recoveries, {} MiB zeroed{shadows}, host wait {:.3} ms",
            counters.epochs,
            counters.tiles,
            counters.restores,
            counters.restored_bytes / MIB,
            counters.recoveries,
            counters.zeroed_bytes / MIB,
            counters.wait_ns as f64 / 1.0e6
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Recording fake: every call is logged; one named failure and the
    /// capture flag are injected.
    #[derive(Default)]
    struct Fake {
        log: Vec<String>,
        fail: Option<&'static str>,
        capturing: bool,
        /// A failure that stays after the first hit.
        sticky: bool,
    }

    impl Fake {
        fn hit(&mut self, name: &'static str, text: String) -> Result<(), String> {
            self.log.push(text);
            if self.fail == Some(name) {
                if !self.sticky {
                    self.fail = None;
                }
                Err(format!("{name} failed"))
            } else {
                Ok(())
            }
        }
    }

    impl LoanOps for Fake {
        fn capturing(&mut self) -> Result<bool, String> {
            self.log.push("capturing".to_string());
            if self.fail == Some("capturing") {
                return Err("capturing failed".to_string());
            }
            Ok(self.capturing)
        }
        fn memset_zero(&mut self, offset: u64, bytes: u64) -> Result<(), String> {
            self.hit("memset", format!("memset {offset}+{bytes}"))
        }
        fn enqueue_restore(&mut self, bytes: u64) -> Result<(), String> {
            self.hit("enqueue", format!("enqueue {bytes}"))
        }
        fn wait_restore(&mut self) -> Result<(), String> {
            self.hit("wait", "wait".to_string())
        }
        fn device_sync(&mut self) -> Result<(), String> {
            self.hit("sync", "sync".to_string())
        }
        fn blocking_restore(&mut self, bytes: u64) -> Result<(), String> {
            self.hit("blocking", format!("blocking {bytes}"))
        }
        fn verify(&mut self, bytes: u64) -> Result<(), String> {
            self.hit("verify", format!("verify {bytes}"))
        }
        fn rebuild_shadows(&mut self) -> Result<(), String> {
            self.hit("rebuild", "rebuild".to_string())
        }
        fn verify_shadows(&mut self) -> Result<(), String> {
            self.hit("verify_shadows", "verify_shadows".to_string())
        }
        fn rebuild_ms(&mut self) -> Option<f64> {
            self.log.push("rebuild_ms".to_string());
            Some(1.5)
        }
    }

    fn writes(fake: &Fake) -> Vec<&String> {
        fake.log
            .iter()
            .filter(|line| line.starts_with("memset") || line.starts_with("enqueue"))
            .collect()
    }

    fn controller() -> LoanController {
        LoanController::new(false, false)
    }

    #[test]
    fn acquire_from_resident_enters_dirty_before_the_first_write_and_zeroes_the_span() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 1000).unwrap();
        assert_eq!(
            *ctl.state(),
            LoanState::Dirty {
                epoch: 1,
                dirty_bytes: 1000,
                zeroed_bytes: 1000
            }
        );
        assert_eq!(fake.log, vec!["capturing", "memset 0+1000"]);
        assert_eq!(ctl.counters().epochs, 1);
        assert_eq!(ctl.counters().zeroed_bytes, 1000);
        assert_eq!(ctl.counters().tiles, 1);
    }

    #[test]
    fn class_growth_zeroes_only_the_delta_and_a_smaller_class_nothing() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 1000).unwrap();
        ctl.acquire(&mut fake, 2500).unwrap();
        ctl.acquire(&mut fake, 400).unwrap();
        let memsets: Vec<_> = fake
            .log
            .iter()
            .filter(|line| line.starts_with("memset"))
            .collect();
        assert_eq!(memsets, ["memset 0+1000", "memset 1000+1500"]);
        assert_eq!(
            *ctl.state(),
            LoanState::Dirty {
                epoch: 1,
                dirty_bytes: 2500,
                zeroed_bytes: 2500
            }
        );
        assert_eq!(ctl.counters().tiles, 3);
        assert_eq!(ctl.counters().zeroed_bytes, 2500);
        assert_eq!(ctl.counters().epochs, 1, "one epoch for three tiles");
    }

    /// Epoch class selection: a big tile followed by a small one in the same
    /// epoch keeps reading the big class view; after the restore a fresh epoch
    /// starts from its own tile's class.
    #[test]
    fn the_borrow_view_class_follows_the_epoch_and_restarts_after_a_restore() {
        use crate::gpu_forward::borrow_view_class;
        let spans = [100u64, 200, 400, 800, 1600];
        let mut ctl = controller();
        let mut fake = Fake::default();
        let view = |ctl: &LoanController, tile: usize| {
            borrow_view_class(5, tile, &spans, ctl.dirty_extent())
        };
        assert_eq!(ctl.dirty_extent(), None);
        // A 16384-row tile (class 4), then a 600-row tile (class 0).
        ctl.acquire(&mut fake, spans[4]).unwrap();
        assert_eq!(view(&ctl, 4), 4);
        ctl.acquire(&mut fake, spans[0]).unwrap();
        assert_eq!(ctl.dirty_extent(), Some(spans[4]));
        assert_eq!(view(&ctl, 0), 4, "the small tile still reads the max view");
        // The prompt ends: the tail stays dirty (lazy restore) until a resident call.
        ctl.finish_prompt(&mut fake, true).unwrap();
        assert_eq!(ctl.dirty_extent(), Some(spans[4]));
        ctl.require_resident(&mut fake).unwrap();
        assert_eq!(ctl.dirty_extent(), None);
        // A fresh epoch starts from its first tile's class and grows with it.
        ctl.acquire(&mut fake, spans[0]).unwrap();
        assert_eq!(view(&ctl, 0), 0);
        ctl.acquire(&mut fake, spans[2]).unwrap();
        assert_eq!(view(&ctl, 2), 2);
        ctl.acquire(&mut fake, spans[1]).unwrap();
        assert_eq!(view(&ctl, 1), 2);
        assert_eq!(ctl.counters().epochs, 2);
    }

    #[test]
    fn finish_prompt_leaves_the_tail_dirty_until_the_next_resident_call() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 4096).unwrap();
        fake.log.clear();
        ctl.finish_prompt(&mut fake, true).unwrap();
        assert_eq!(
            *ctl.state(),
            LoanState::Dirty {
                epoch: 1,
                dirty_bytes: 4096,
                zeroed_bytes: 4096
            }
        );
        assert!(writes(&fake).is_empty(), "no enqueue at prompt end");
        assert!(!fake.log.contains(&"wait".to_string()), "no wait either");
        assert_eq!(ctl.counters().restores, 0);
        // The next resident call enqueues the whole restore and waits for it.
        fake.log.clear();
        ctl.require_resident(&mut fake).unwrap();
        assert!(ctl.is_resident());
        assert_eq!(
            fake.log,
            vec!["capturing", "enqueue 4096", "wait", "verify 4096"]
        );
        assert_eq!(ctl.counters().restores, 1);
        assert_eq!(ctl.counters().restored_bytes, 4096);
    }

    #[test]
    fn finish_prompt_under_capture_with_a_dirty_tail_poisons() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 100).unwrap();
        fake.capturing = true;
        fake.log.clear();
        let error = ctl.finish_prompt(&mut fake, true).unwrap_err();
        assert!(error.contains("poisoned"), "{error}");
        assert!(matches!(ctl.state(), LoanState::Poisoned(_)));
        assert!(writes(&fake).is_empty());
    }

    #[test]
    fn sync_knob_waits_right_after_the_enqueue() {
        let mut ctl = LoanController::new(true, false);
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 64).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        assert!(ctl.is_resident());
        let enqueue = fake.log.iter().position(|l| l == "enqueue 64").unwrap();
        let wait = fake.log.iter().position(|l| l == "wait").unwrap();
        assert!(enqueue < wait);
        assert_eq!(ctl.counters().restores, 1);
    }

    #[test]
    fn require_resident_from_each_state() {
        // Resident: nothing.
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.require_resident(&mut fake).unwrap();
        assert!(fake.log.is_empty());
        // Dirty: enqueue then wait.
        ctl.acquire(&mut fake, 512).unwrap();
        fake.log.clear();
        ctl.require_resident(&mut fake).unwrap();
        assert!(ctl.is_resident());
        assert_eq!(
            fake.log,
            vec!["capturing", "enqueue 512", "wait", "verify 512"]
        );
        // Restoring: wait only.
        ctl.state = LoanState::Restoring {
            epoch: 1,
            bytes: 512,
            zeroed_bytes: 512,
        };
        fake.log.clear();
        ctl.require_resident(&mut fake).unwrap();
        assert_eq!(fake.log, vec!["wait", "verify 512"]);
        assert!(ctl.is_resident());
    }

    #[test]
    fn acquire_from_restoring_waits_first_and_starts_a_new_epoch() {
        // Restoring is reached through a resident call or the sync knob; a
        // following tile waits, then opens epoch 2.
        let mut ctl = LoanController::new(true, false);
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 100).unwrap();
        ctl.state = LoanState::Restoring {
            epoch: 1,
            bytes: 100,
            zeroed_bytes: 100,
        };
        fake.log.clear();
        ctl.acquire(&mut fake, 100).unwrap();
        assert_eq!(
            fake.log,
            vec!["capturing", "wait", "verify 100", "memset 0+100"]
        );
        assert_eq!(
            *ctl.state(),
            LoanState::Dirty {
                epoch: 2,
                dirty_bytes: 100,
                zeroed_bytes: 100
            }
        );
        assert_eq!(ctl.counters().epochs, 2);
    }

    #[test]
    fn back_to_back_prompts_share_one_epoch_and_never_enqueue() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 1000).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        ctl.acquire(&mut fake, 1000).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        assert!(writes(&fake).iter().all(|l| l.starts_with("memset")));
        assert!(!fake.log.contains(&"wait".to_string()));
        assert_eq!(ctl.counters().epochs, 1);
        assert_eq!(ctl.counters().restores, 0);
        assert!(matches!(ctl.state(), LoanState::Dirty { epoch: 1, .. }));
    }

    #[test]
    fn class_growth_in_the_second_prompt_zeroes_only_the_delta() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 1000).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        ctl.acquire(&mut fake, 2500).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        let memsets: Vec<_> = fake
            .log
            .iter()
            .filter(|line| line.starts_with("memset"))
            .collect();
        assert_eq!(memsets, ["memset 0+1000", "memset 1000+1500"]);
        assert_eq!(ctl.counters().zeroed_bytes, 2500);
        assert_eq!(ctl.dirty_extent(), Some(2500));
    }

    #[test]
    fn capture_refuses_a_borrowed_tile_before_any_write() {
        let mut ctl = controller();
        let mut fake = Fake {
            capturing: true,
            ..Fake::default()
        };
        let error = ctl.acquire(&mut fake, 100).unwrap_err();
        assert!(error.contains("stream capture"), "{error}");
        assert!(writes(&fake).is_empty());
        assert!(ctl.is_resident(), "no state change either");
        assert_eq!(ctl.counters().epochs, 0);
    }

    #[test]
    fn capture_with_a_dirty_tail_poisons_instead_of_restoring() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 100).unwrap();
        fake.capturing = true;
        fake.log.clear();
        let error = ctl.require_resident(&mut fake).unwrap_err();
        assert!(error.contains("poisoned"), "{error}");
        assert!(matches!(ctl.state(), LoanState::Poisoned(_)));
        assert!(writes(&fake).is_empty());
    }

    #[test]
    fn tile_error_recovers_with_a_blocking_copy_and_returns_resident() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 2048).unwrap();
        fake.log.clear();
        ctl.after_tile(&mut fake, false).unwrap();
        assert!(ctl.is_resident());
        assert_eq!(
            fake.log,
            vec!["capturing", "sync", "blocking 2048", "sync"],
            "drain, blocking copy, drain, then Resident"
        );
        assert_eq!(ctl.counters().recoveries, 1);
        // A successful tile never recovers.
        ctl.acquire(&mut fake, 2048).unwrap();
        fake.log.clear();
        ctl.after_tile(&mut fake, true).unwrap();
        assert!(fake.log.is_empty());
    }

    #[test]
    fn failed_recovery_poisons_and_everything_then_refuses() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 2048).unwrap();
        fake.fail = Some("blocking");
        let error = ctl.after_tile(&mut fake, false).unwrap_err();
        assert!(error.contains("recovery failed"), "{error}");
        assert!(matches!(ctl.state(), LoanState::Poisoned(_)));
        fake.log.clear();
        for result in [
            ctl.acquire(&mut fake, 10),
            ctl.require_resident(&mut fake),
            ctl.check(),
        ] {
            let error = result.unwrap_err();
            assert!(
                error.contains("poisoned") && error.contains("recovery failed"),
                "{error}"
            );
        }
        assert!(fake.log.is_empty(), "a poisoned loan touches nothing");
        // finish_prompt and teardown of a poisoned loan do not restore either.
        ctl.finish_prompt(&mut fake, true).unwrap();
        assert!(ctl.teardown(&mut fake).is_err());
        assert!(fake.log.is_empty());
    }

    #[test]
    fn recovery_under_a_capture_poisons() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 64).unwrap();
        fake.capturing = true;
        let error = ctl.after_tile(&mut fake, false).unwrap_err();
        assert!(error.contains("capture"), "{error}");
        assert!(matches!(ctl.state(), LoanState::Poisoned(_)));
    }

    #[test]
    fn zero_fill_failure_recovers_the_extent_it_may_have_dirtied() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        fake.fail = Some("memset");
        let error = ctl.acquire(&mut fake, 300).unwrap_err();
        assert!(error.contains("zero donor scratch"), "{error}");
        assert!(ctl.is_resident());
        assert!(fake.log.contains(&"blocking 300".to_string()));
    }

    #[test]
    fn enqueue_failure_recovers_and_keeps_the_original_error() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 300).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        fake.fail = Some("enqueue");
        let error = ctl.require_resident(&mut fake).unwrap_err();
        assert!(error.contains("enqueue donor restore"), "{error}");
        assert!(ctl.is_resident());
        assert!(fake.log.contains(&"blocking 300".to_string()));
    }

    #[test]
    fn wait_failure_recovers_the_restoring_extent() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 300).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        fake.fail = Some("wait");
        let error = ctl.require_resident(&mut fake).unwrap_err();
        assert!(error.contains("wait donor restore"), "{error}");
        assert!(ctl.is_resident());
        assert!(fake.log.contains(&"blocking 300".to_string()));
    }

    #[test]
    fn verify_failure_after_a_restore_poisons() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 128).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        fake.fail = Some("verify");
        let error = ctl.require_resident(&mut fake).unwrap_err();
        assert!(error.contains("verification"), "{error}");
        assert!(matches!(ctl.state(), LoanState::Poisoned(_)));
    }

    #[test]
    fn an_abandoned_prompt_recovers_instead_of_enqueuing() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 640).unwrap();
        fake.log.clear();
        ctl.finish_prompt(&mut fake, false).unwrap();
        assert!(ctl.is_resident());
        assert!(fake.log.contains(&"blocking 640".to_string()));
        assert!(!fake.log.iter().any(|l| l.starts_with("enqueue")));
        // Nothing dirty: finish is a no-op either way.
        fake.log.clear();
        ctl.finish_prompt(&mut fake, true).unwrap();
        ctl.finish_prompt(&mut fake, false).unwrap();
        assert!(fake.log.is_empty());
    }

    #[test]
    fn teardown_restores_before_release_and_reports_failure() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 256).unwrap();
        fake.log.clear();
        ctl.teardown(&mut fake).unwrap();
        assert!(ctl.is_resident());
        let enqueue = fake.log.iter().position(|l| l == "enqueue 256").unwrap();
        let wait = fake.log.iter().position(|l| l == "wait").unwrap();
        assert!(enqueue < wait);
        // An enqueue that fails at teardown falls back to the blocking recovery
        // copy: the tail is whole, so teardown succeeds.
        let mut ctl = controller();
        ctl.acquire(&mut fake, 256).unwrap();
        fake.fail = Some("enqueue");
        fake.sticky = true;
        fake.capturing = false;
        ctl.teardown(&mut fake).unwrap();
        assert!(ctl.is_resident());
        // A restore that is needed under a stream capture cannot run: the loan
        // poisons and teardown is an error (the caller leaks).
        let mut ctl = controller();
        fake.fail = None;
        fake.sticky = false;
        ctl.acquire(&mut fake, 256).unwrap();
        fake.capturing = true;
        assert!(ctl.teardown(&mut fake).is_err());
        assert!(matches!(ctl.state(), LoanState::Poisoned(_)));
    }

    #[test]
    fn epoch_numbers_and_counters_accumulate_across_epochs() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        for epoch in 1..=3u64 {
            ctl.acquire(&mut fake, 1 << 20).unwrap();
            ctl.acquire(&mut fake, 2 << 20).unwrap();
            assert_eq!(
                *ctl.state(),
                LoanState::Dirty {
                    epoch,
                    dirty_bytes: 2 << 20,
                    zeroed_bytes: 2 << 20
                }
            );
            ctl.finish_prompt(&mut fake, true).unwrap();
            ctl.require_resident(&mut fake).unwrap();
        }
        let counters = ctl.counters();
        assert_eq!(counters.epochs, 3);
        assert_eq!(counters.tiles, 6);
        assert_eq!(counters.restores, 3);
        assert_eq!(counters.restored_bytes, 3 * (2 << 20));
        assert_eq!(counters.zeroed_bytes, 3 * (2 << 20));
        assert_eq!(counters.recoveries, 0);
        assert!(ctl.summary().contains("3 epochs"));
        assert!(ctl.summary().contains("6 prompt tiles"));
    }

    fn shadow_controller(shadow_bytes: u64) -> LoanController {
        LoanController::new(false, false).with_shadow_bytes(shadow_bytes)
    }

    /// The first acquire of an epoch zero-fills only the scratch part (after
    /// the shadow region) and then rebuilds the shadows; continuation acquires
    /// (a bigger class too) rebuild nothing.
    #[test]
    fn shadows_rebuild_once_per_epoch_after_the_scratch_zero_fill() {
        let mut ctl = shadow_controller(300);
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 1000).unwrap();
        assert_eq!(
            *ctl.state(),
            LoanState::Dirty {
                epoch: 1,
                dirty_bytes: 1000,
                zeroed_bytes: 1000
            }
        );
        assert_eq!(
            fake.log,
            vec!["capturing", "memset 300+700", "rebuild", "verify_shadows"]
        );
        ctl.acquire(&mut fake, 2500).unwrap();
        ctl.acquire(&mut fake, 400).unwrap();
        let rebuilds = fake.log.iter().filter(|line| *line == "rebuild").count();
        assert_eq!(rebuilds, 1, "continuation acquires do not rebuild");
        let memsets: Vec<_> = fake
            .log
            .iter()
            .filter(|line| line.starts_with("memset"))
            .collect();
        assert_eq!(memsets, ["memset 300+700", "memset 1000+1500"]);
        assert_eq!(
            ctl.counters().zeroed_bytes,
            2200,
            "the shadow region is never zero-filled"
        );
        assert_eq!(ctl.counters().shadow_rebuilds, 1);
        // The restore covers the whole dirty prefix, shadow region included.
        ctl.require_resident(&mut fake).unwrap();
        assert!(fake.log.contains(&"enqueue 2500".to_string()));
        assert!(ctl.is_resident());
        // A fresh epoch rebuilds again.
        ctl.acquire(&mut fake, 1000).unwrap();
        assert_eq!(ctl.counters().shadow_rebuilds, 2);
        assert_eq!(fake.log.iter().filter(|line| *line == "rebuild").count(), 2);
    }

    /// A class span equal to the shadow region (no scratch) still rebuilds
    /// and never zero-fills.
    #[test]
    fn a_span_that_is_only_the_shadow_region_zeroes_nothing() {
        let mut ctl = shadow_controller(512);
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 512).unwrap();
        assert_eq!(fake.log, vec!["capturing", "rebuild", "verify_shadows"]);
    }

    /// A controller without shadows never rebuilds (the plain path is unchanged).
    #[test]
    fn no_shadow_bytes_means_no_rebuild() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 1000).unwrap();
        assert!(!fake.log.iter().any(|line| line == "rebuild"));
        assert_eq!(ctl.counters().shadow_rebuilds, 0);
    }

    /// A failed rebuild recovers (blocking copy of the whole dirty prefix) and
    /// returns the error, leaving the tail resident.
    #[test]
    fn a_failed_shadow_rebuild_recovers_the_tail() {
        let mut ctl = shadow_controller(300);
        let mut fake = Fake {
            fail: Some("rebuild"),
            ..Fake::default()
        };
        let error = ctl.acquire(&mut fake, 1000).unwrap_err();
        assert!(
            error.contains("rebuild donor-tail weight shadows"),
            "{error}"
        );
        assert!(fake.log.contains(&"blocking 1000".to_string()));
        assert!(ctl.is_resident());
        assert_eq!(ctl.counters().shadow_rebuilds, 0);
    }

    /// After a good rebuild a failing tile whose recovery copy also fails
    /// poisons the loan.
    #[test]
    fn a_failed_tile_after_a_shadow_rebuild_with_a_failed_recovery_poisons() {
        let mut ctl = shadow_controller(300);
        let mut fake = Fake {
            fail: Some("blocking"),
            sticky: true,
            ..Fake::default()
        };
        ctl.acquire(&mut fake, 1000).unwrap();
        let error = ctl.after_tile(&mut fake, false).unwrap_err();
        assert!(error.contains("poisoned"), "{error}");
        assert!(ctl.check().is_err());
    }

    /// A byte-oracle mismatch after the rebuild poisons the loan (the donor
    /// tail is left as is: nothing may run) and is named in the error.
    #[test]
    fn a_shadow_oracle_mismatch_poisons() {
        let mut ctl = shadow_controller(300);
        let mut fake = Fake {
            fail: Some("verify_shadows"),
            ..Fake::default()
        };
        let error = ctl.acquire(&mut fake, 1000).unwrap_err();
        assert!(error.contains("borrowed weight shadow oracle"), "{error}");
        assert!(ctl.check().is_err());
        assert!(!fake.log.iter().any(|line| line.starts_with("blocking")));
        assert_eq!(ctl.counters().shadow_rebuilds, 0);
    }

    /// The epoch log path reads the rebuild time only after the restore
    /// completed (the timing events are done by then).
    #[test]
    fn the_epoch_log_reads_the_rebuild_time_after_the_restore() {
        let mut ctl = LoanController::new(false, true).with_shadow_bytes(300);
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 1000).unwrap();
        ctl.require_resident(&mut fake).unwrap();
        let wait = fake.log.iter().position(|line| line == "wait").unwrap();
        let ms = fake
            .log
            .iter()
            .position(|line| line == "rebuild_ms")
            .unwrap();
        assert!(ms > wait);
        assert!(ctl.summary().contains("1 weight-shadow rebuilds"));
    }
}
