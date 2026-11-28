// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Secondary retained-PM4 bodies keyed by `(rows, identity)` (HIPF-297).
//!
//! A model's ordinary-AR tape lives in `Gpu::replay`. A speculative verify body
//! has a different launch sequence per row count, and bakes the one moving
//! pointer it was recorded at (the Qwen4 GDN capture-ring half, `identity`).
//! Each `(rows, identity)` therefore owns its own [`ReplayController`], which
//! the model's retained-body boundary swaps into `gpu.replay` for exactly one
//! admitted forward ([`RetainedKeys::admit`] / [`RetainedKeys::release`]), so
//! the AR controller never sees a verify body.
//!
//! Lifecycle per key: the first admitted forward records and is kept as a
//! calibration sample; the next admitted forward at least `cal_gap` positions
//! later records again, is differenced against it
//! ([`ReplayController::synthesize_position_bindings`]) and is prepared with
//! [`ReplayController::prepare_pm4_prefix_calibrated`]; later forwards replay.
//! A scratch growth (any [`crate::scratch::scratch_growth_epoch`] change)
//! re-resolves every prepared tape's resources, as the AR controller's
//! post-growth refresh does; a moved or freed resource drops that key back to
//! recording, and a calibration pair that straddles a growth restarts. A
//! growth inside an admitted forward is also applied to the parked AR
//! controller. A prepared key that later falls back (replay failure, a route
//! identity change) poisons every key. A prepare failure poisons only its key.
//!
//! The owner of the speculative lifecycle reads [`RetainedKeys::window`] before
//! a window to keep record windows out of its cost model, and
//! [`RetainedKeys::pending`] to drive a pre-warm.

use crate::replay::{RecordedKernargSnapshot, ReplayController, ReplayState};
use crate::Gpu;

/// What an admitted forward of a key does next.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KeyWindow {
    /// No live key admits this shape: the forward runs eager.
    None,
    /// The forward records (calibration or the preparing recording); it runs
    /// on the recording route and pays the record (and maybe prepare) cost.
    Record,
    /// The forward replays a prepared tape.
    Replay,
}

/// Counters of one key, for cost reports and route proofs.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeyStats {
    pub records: u32,
    pub rearms: u32,
    /// Host ms of the last close that prepared the key (sync + comparator +
    /// prepare).
    pub prepare_ms: f64,
    pub synthesized_bindings: usize,
    /// Positions the prepared tape was calibrated at.
    pub calibrated_at: Option<(usize, usize)>,
}

struct RetainedKey {
    rows: usize,
    identity: usize,
    controller: ReplayController,
    /// The first recording, its position and the scratch-growth epoch it was
    /// taken at, until the second recording lands.
    calibration: Option<(RecordedKernargSnapshot, usize, u64)>,
    prepared: bool,
    poisoned: bool,
    stats: KeyStats,
}

impl RetainedKey {
    fn rearm(&mut self) {
        self.controller.rearm_after_layout_growth();
        self.calibration = None;
        self.prepared = false;
        self.stats.rearms += 1;
    }

    fn window(&self, position: usize, cal_gap: usize) -> KeyWindow {
        if self.poisoned {
            return KeyWindow::None;
        }
        if self.prepared {
            return KeyWindow::Replay;
        }
        match &self.calibration {
            Some((_, first, _)) if position < first + cal_gap => KeyWindow::None,
            _ => KeyWindow::Record,
        }
    }
}

pub struct RetainedKeys {
    rows: Vec<usize>,
    routes_recordable: bool,
    cal_gap: usize,
    /// At most two keys per row count (the compact ring's two halves). The
    /// double ring maps a row count to more pointers; those keys are refused.
    max_keys: usize,
    /// Last scratch-growth epoch the parked keys were brought up to date with.
    epoch: u64,
    /// Epoch when the active key was admitted.
    admit_epoch: u64,
    keys: Vec<RetainedKey>,
    active: Option<usize>,
}

// SAFETY: the controllers hold HSA queue and kernarg pointers, like the
// `Gpu::replay` they swap with; they are used only by the thread that drives
// that `Gpu` and move with the model like the `Gpu` does.
unsafe impl Send for RetainedKeys {}

impl RetainedKeys {
    /// Keys for the verify row counts `rows`. `routes_recordable` admits the
    /// bitwise-identical eager verify routes into the tapes
    /// ([`ReplayController::set_verify_routes_recordable`]).
    pub fn new(rows: Vec<usize>, routes_recordable: bool, cal_gap: usize) -> Self {
        Self {
            max_keys: 2 * rows.len(),
            rows,
            routes_recordable,
            cal_gap: cal_gap.max(1),
            epoch: crate::scratch::scratch_growth_epoch(),
            admit_epoch: 0,
            keys: Vec::new(),
            active: None,
        }
    }

    fn find(&self, rows: usize, identity: usize) -> Option<usize> {
        self.keys
            .iter()
            .position(|key| key.rows == rows && key.identity == identity)
    }

    fn admits_shape(&self, gpu: &Gpu, rows: usize) -> bool {
        // PM4 retained verify is validated on exact gfx1151 only (HIPF-297).
        gpu.arch_caps.is_gfx1151() && self.rows.contains(&rows)
    }

    /// The next forward of this shape, without side effects. A key not yet
    /// created reports `Record` while there is room for it.
    pub fn window(&self, gpu: &Gpu, rows: usize, identity: usize, position: usize) -> KeyWindow {
        if !self.admits_shape(gpu, rows) {
            return KeyWindow::None;
        }
        match self.find(rows, identity) {
            Some(index) => self.keys[index].window(position, self.cal_gap),
            None if self.keys.len() < self.max_keys => KeyWindow::Record,
            None => KeyWindow::None,
        }
    }

    /// Swap the key of this shape into `gpu.replay` when its next forward
    /// records or replays; `false` (and nothing swapped) otherwise.
    pub fn admit(&mut self, gpu: &mut Gpu, rows: usize, identity: usize, position: usize) -> bool {
        debug_assert!(self.active.is_none(), "a retained key is already active");
        if !self.admits_shape(gpu, rows) {
            return false;
        }
        // A scratch growth while the keys were parked: re-resolve every
        // prepared tape's resources (the AR controller's post-growth refresh);
        // a moved or freed resource drops that key back to recording.
        let epoch = crate::scratch::scratch_growth_epoch();
        if epoch != self.epoch {
            self.epoch = epoch;
            for key in self.keys.iter_mut().filter(|k| k.prepared && !k.poisoned) {
                key.controller.arm_binding_refresh_for_scratch_growth();
                if let Err(reason) = key.controller.refresh_bindings_after_growth(&gpu.hip) {
                    eprintln!(
                        "[retained-keys] rows={} identity={:#x} rearmed: {reason}",
                        key.rows, key.identity
                    );
                    key.rearm();
                }
            }
        }
        self.admit_epoch = epoch;
        let index = match self.find(rows, identity) {
            Some(index) => index,
            None if self.keys.len() < self.max_keys => {
                let mut controller = ReplayController::new_auto_pm4();
                controller.set_verify_routes_recordable(self.routes_recordable);
                self.keys.push(RetainedKey {
                    rows,
                    identity,
                    controller,
                    calibration: None,
                    prepared: false,
                    poisoned: false,
                    stats: KeyStats::default(),
                });
                eprintln!(
                    "[retained-keys] key rows={rows} identity={identity:#x} keys={}",
                    self.keys.len()
                );
                self.keys.len() - 1
            }
            None => return false,
        };
        let key = &mut self.keys[index];
        if key.window(position, self.cal_gap) == KeyWindow::None {
            return false;
        }
        std::mem::swap(&mut gpu.replay, &mut key.controller);
        self.active = Some(index);
        true
    }

    /// Whether a key's controller is `gpu.replay` now.
    pub fn active(&self) -> bool {
        self.active.is_some()
    }

    /// Swap the AR controller back after an admitted forward.
    pub fn release(&mut self, gpu: &mut Gpu) {
        let Some(index) = self.active.take() else {
            return;
        };
        let key = &mut self.keys[index];
        std::mem::swap(&mut gpu.replay, &mut key.controller);
        // A scratch growth inside the admitted forward reached only the key's
        // controller: apply it to the parked AR controller too.
        if crate::scratch::scratch_growth_epoch() != self.admit_epoch {
            if gpu.replay.prepared_pm4_route_active() {
                gpu.replay.arm_binding_refresh_for_scratch_growth();
            } else if matches!(
                gpu.replay.state(),
                ReplayState::Captured | ReplayState::ShadowValidated
            ) {
                gpu.replay.rearm_after_layout_growth();
            }
            if key.controller.binding_refresh_pending() {
                if let Err(reason) = key.controller.refresh_bindings_after_growth(&gpu.hip) {
                    eprintln!(
                        "[retained-keys] rows={} identity={:#x} rearmed: {reason}",
                        key.rows, key.identity
                    );
                    key.rearm();
                }
            }
        }
        if key.controller.state() == ReplayState::Fallback && !key.poisoned {
            let reason = key
                .controller
                .fallback_reason()
                .unwrap_or("retained key fell back")
                .to_owned();
            if key.prepared {
                self.poison_all(&reason);
            } else {
                key.poisoned = true;
            }
        }
    }

    /// Stop every key; later forwards run eager.
    pub fn poison_all(&mut self, reason: &str) {
        eprintln!("[retained-keys] poison all keys: {reason}");
        for key in &mut self.keys {
            key.controller.poison(reason.to_owned());
            key.poisoned = true;
        }
    }

    /// Keys that still need a recording before they replay, for a pre-warm
    /// driver: `(rows, identity)`.
    pub fn pending(&self) -> Vec<(usize, usize)> {
        self.keys
            .iter()
            .filter(|key| !key.prepared && !key.poisoned)
            .map(|key| (key.rows, key.identity))
            .collect()
    }

    /// `(rows, identity, stats)` of every key.
    pub fn stats(&self) -> Vec<(usize, usize, KeyStats)> {
        self.keys
            .iter()
            .map(|key| (key.rows, key.identity, key.stats))
            .collect()
    }

    /// Close the recording window of the active key's forward at `position`.
    /// The first recording is kept for the comparator and the key rearms; the
    /// next one is differenced against it and prepared.
    pub fn finish_capture(&mut self, gpu: &mut Gpu, position: usize) {
        let Some(index) = self.active else {
            return;
        };
        let started = std::time::Instant::now();
        let epoch = self.admit_epoch;
        let key = &mut self.keys[index];
        let (rows, identity) = (key.rows, key.identity);
        let effects = gpu.replay.capture_window_effects();
        let fail = |gpu: &mut Gpu, reason: String| {
            eprintln!("[retained-keys] rows={rows} identity={identity:#x} fallback: {reason}");
            gpu.replay.poison(reason);
        };
        if let Err(error) = gpu.hip.device_synchronize() {
            return fail(gpu, format!("capture sync failed: {error:?}"));
        }
        let summary = match gpu.replay.finish_capture() {
            Ok(summary) => summary,
            // A scratch growth inside the recording rearmed the controller:
            // the next admit starts this key over.
            Err(_) if crate::scratch::scratch_growth_epoch() != epoch => return,
            Err(reason) => return fail(gpu, reason.to_owned()),
        };
        key.stats.records += 1;
        eprintln!(
            "[retained-keys] recorded rows={rows} identity={identity:#x} position={position} \
             computed={} recorded={} body={} unique_kernels={} sequence_hash={:016x} htod={} \
             dtod={} dtoh={} memset={}",
            effects.launches,
            summary.launch_count,
            gpu.replay.retained_body_launches(),
            summary.unique_kernel_count,
            summary.sequence_hash,
            effects.htod,
            effects.dtod,
            effects.dtoh,
            effects.memset
        );
        let now = crate::scratch::scratch_growth_epoch();
        let Some((earlier, earlier_position, earlier_epoch)) = key.calibration.take() else {
            key.calibration = Some((gpu.replay.snapshot_recorded_kernargs(), position, now));
            gpu.replay.rearm_after_layout_growth();
            return;
        };
        match gpu
            .replay
            .synthesize_position_bindings(&earlier, earlier_position, position)
        {
            Ok(count) => {
                key.stats.synthesized_bindings = count;
                eprintln!(
                    "[retained-keys] rows={rows} identity={identity:#x} position comparator \
                     {earlier_position}->{position}: synthesized={count}"
                );
            }
            // A scratch growth between the two recordings may have moved a
            // buffer: this recording becomes the first sample.
            Err(reason) if earlier_epoch != now => {
                eprintln!(
                    "[retained-keys] rows={rows} identity={identity:#x} recalibrate at \
                     {position} (scratch grew since {earlier_position}): {reason}"
                );
                key.calibration = Some((gpu.replay.snapshot_recorded_kernargs(), position, now));
                gpu.replay.rearm_after_layout_growth();
                return;
            }
            Err(reason) => {
                return fail(
                    gpu,
                    format!("position comparator {earlier_position}->{position}: {reason}"),
                )
            }
        }
        let launches = gpu.replay.retained_body_launches();
        match gpu
            .replay
            .prepare_pm4_prefix_calibrated(gpu.device_id as usize, launches)
        {
            Ok(_) => {
                key.prepared = true;
                key.stats.prepare_ms = started.elapsed().as_secs_f64() * 1e3;
                key.stats.calibrated_at = Some((earlier_position, position));
                eprintln!(
                    "[retained-keys] prepared rows={rows} identity={identity:#x} \
                     dispatches={launches} close_ms={:.1}",
                    key.stats.prepare_ms
                );
            }
            Err(reason) => fail(gpu, format!("prepare failed: {reason}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(prepared: bool, calibration_at: Option<usize>) -> RetainedKey {
        RetainedKey {
            rows: 3,
            identity: 1,
            controller: ReplayController::new_auto_pm4(),
            calibration: calibration_at.map(|at| (RecordedKernargSnapshot::default(), at, 0)),
            prepared,
            poisoned: false,
            stats: KeyStats::default(),
        }
    }

    /// The S9 query: a key records until prepared, waits out the calibration
    /// gap between its two recordings, and replays once prepared.
    #[test]
    fn key_window_follows_the_calibration_lifecycle() {
        assert_eq!(key(false, None).window(10, 100), KeyWindow::Record);
        assert_eq!(key(false, Some(10)).window(50, 100), KeyWindow::None);
        assert_eq!(key(false, Some(10)).window(110, 100), KeyWindow::Record);
        assert_eq!(key(true, None).window(0, 100), KeyWindow::Replay);
        let mut poisoned = key(true, None);
        poisoned.poisoned = true;
        assert_eq!(poisoned.window(0, 100), KeyWindow::None);
    }
}
