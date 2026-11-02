// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Plan-mode tape (railgun design A, "Authoring, not capture").
//!
//! While a plan tape is active on the owning thread, [`crate::HipRuntime`]
//! does not touch the device: blob launches, device-to-device copies and
//! memsets append a typed [`PlanEvent`] to the tape and return `Ok(())`;
//! every other HIP API (host transfers, synchronization, allocation, events,
//! graphs, raw `kernelParams` launches, module loads) records a refusal and
//! returns an error, so the plan is discarded rather than partially executed.
//!
//! The tape is independent of hipGraph capture and of Redline recording:
//! it neither sets nor reads `capture_mode` / `is_recording`, so every route
//! decision upstream is untouched.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::thread::ThreadId;

use crate::error::{HipError, HipResult};

/// One planned device operation, in launch order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanEvent {
    /// A `hipModuleLaunchKernel` `extra`-blob launch. `func` is the raw
    /// `hipFunction_t` handle (identity only).
    Launch {
        func: usize,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg: Vec<u8>,
        label: &'static str,
    },
    /// Device-to-device copy of `bytes` bytes between absolute addresses.
    Copy { dst: u64, src: u64, bytes: u64 },
    /// 32-bit-word fill of `words` words at absolute address `dst`.
    Fill { dst: u64, value: u32, words: u64 },
}

/// What a finished plan produced.
#[derive(Debug, Default)]
pub struct PlanTape {
    pub events: Vec<PlanEvent>,
    /// First refusal (the plan is discarded when set).
    pub refusal: Option<String>,
}

#[derive(Default)]
struct PlanInner {
    owner: Option<ThreadId>,
    tape: PlanTape,
    label: &'static str,
}

/// The per-runtime plan gate: one relaxed atomic load on every HIP call
/// when idle.
#[derive(Default)]
pub struct PlanGate {
    active: AtomicBool,
    inner: Mutex<PlanInner>,
    witness_active: AtomicBool,
    witness: Mutex<(Option<ThreadId>, Vec<PlanEvent>)>,
}

impl PlanGate {
    /// Start recording every *real* blob launch of this thread (they still
    /// execute / get captured) so an oracle can compare a real window's
    /// launch list against a plan's.
    pub fn witness_begin(&self) -> HipResult<()> {
        let mut w = self.witness.lock().unwrap_or_else(|e| e.into_inner());
        if w.0.is_some() {
            return Err(HipError::new(0, "launch witness is already active"));
        }
        w.0 = Some(std::thread::current().id());
        w.1.clear();
        self.witness_active.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub fn witness_end(&self) -> Vec<PlanEvent> {
        let mut w = self.witness.lock().unwrap_or_else(|e| e.into_inner());
        self.witness_active.store(false, Ordering::Relaxed);
        w.0 = None;
        std::mem::take(&mut w.1)
    }

    /// True only on the thread that began the witness.
    #[inline]
    pub fn witnessing(&self) -> bool {
        if !self.witness_active.load(Ordering::Relaxed) {
            return false;
        }
        let w = self.witness.lock().unwrap_or_else(|e| e.into_inner());
        w.0 == Some(std::thread::current().id())
    }

    pub fn witness_launch(
        &self,
        func: usize,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg: &[u8],
    ) {
        self.witness
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .1
            .push(PlanEvent::Launch {
                func,
                grid,
                block,
                shared_mem,
                kernarg: kernarg.to_vec(),
                label: "",
            });
    }

    /// True only on the thread that began the plan.
    #[inline]
    pub fn active(&self) -> bool {
        if !self.active.load(Ordering::Relaxed) {
            return false;
        }
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.owner == Some(std::thread::current().id())
    }

    pub fn begin(&self) -> HipResult<()> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.owner.is_some() {
            return Err(HipError::new(0, "plan mode is already active"));
        }
        inner.owner = Some(std::thread::current().id());
        inner.tape = PlanTape::default();
        inner.label = "";
        self.active.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub fn end(&self) -> PlanTape {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        self.active.store(false, Ordering::Relaxed);
        inner.owner = None;
        std::mem::take(&mut inner.tape)
    }

    pub fn set_label(&self, label: &'static str) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).label = label;
    }

    pub fn event_count(&self) -> usize {
        if !self.active() {
            return 0;
        }
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .tape
            .events
            .len()
    }

    /// Index of the most recent `Launch` event.
    pub fn last_launch_index(&self) -> Option<usize> {
        if !self.active() {
            return None;
        }
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .tape
            .events
            .iter()
            .rposition(|e| matches!(e, PlanEvent::Launch { .. }))
    }

    /// Record the first refusal and return the error every refused API
    /// reports.
    pub fn refuse(&self, why: &str) -> HipError {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.tape.refusal.is_none() {
            inner.tape.refusal = Some(why.to_string());
        }
        HipError::new(0, &format!("plan mode refused: {why}"))
    }

    pub fn launch(
        &self,
        func: usize,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg: &[u8],
    ) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let label = inner.label;
        inner.tape.events.push(PlanEvent::Launch {
            func,
            grid,
            block,
            shared_mem,
            kernarg: kernarg.to_vec(),
            label,
        });
    }

    /// Record `bytes` bytes copied from `src_base + src_off` to
    /// `dst_base + dst_off`, bounds-checked against the buffer extents.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_range(
        &self,
        api: &str,
        dst_base: u64,
        dst_size: usize,
        dst_off: usize,
        src_base: u64,
        src_size: usize,
        src_off: usize,
        bytes: usize,
    ) -> HipResult<()> {
        let fits = |off: usize, size: usize| off.checked_add(bytes).is_some_and(|end| end <= size);
        if !fits(dst_off, dst_size) || !fits(src_off, src_size) {
            return Err(self.refuse(&format!("{api}: copy out of bounds")));
        }
        if bytes == 0 {
            return Ok(());
        }
        self.copy_raw(
            dst_base + dst_off as u64,
            src_base + src_off as u64,
            bytes as u64,
        );
        Ok(())
    }

    pub fn copy_raw(&self, dst: u64, src: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .tape
            .events
            .push(PlanEvent::Copy { dst, src, bytes });
    }

    /// Record a byte memset (`hipMemset*` semantics): only word-aligned
    /// extents are representable as a word fill.
    pub fn memset_bytes(
        &self,
        api: &str,
        base: u64,
        buf_size: usize,
        value: i32,
        size: usize,
    ) -> HipResult<()> {
        if size > buf_size {
            return Err(self.refuse(&format!("{api}: memset out of bounds")));
        }
        if size == 0 {
            return Ok(());
        }
        if size % 4 != 0 || base % 4 != 0 {
            return Err(self.refuse(&format!(
                "{api}: memset of {size} bytes at {base:#x} is not word aligned"
            )));
        }
        let byte = (value & 0xff) as u32;
        self.fill(base, byte * 0x0101_0101, (size / 4) as u64);
        Ok(())
    }

    pub fn memset_words(
        &self,
        api: &str,
        base: u64,
        buf_size: usize,
        value: i32,
        words: usize,
    ) -> HipResult<()> {
        if words.checked_mul(4).is_none_or(|b| b > buf_size) {
            return Err(self.refuse(&format!("{api}: memset out of bounds")));
        }
        if words == 0 {
            return Ok(());
        }
        if base % 4 != 0 {
            return Err(self.refuse(&format!("{api}: memset at {base:#x} is not word aligned")));
        }
        self.fill(base, value as u32, words as u64);
        Ok(())
    }

    fn fill(&self, dst: u64, value: u32, words: u64) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .tape
            .events
            .push(PlanEvent::Fill { dst, value, words });
    }
}
