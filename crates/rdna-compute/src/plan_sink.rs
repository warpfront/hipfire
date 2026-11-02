// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Plan sink (railgun design A, "Authoring, not capture").
//!
//! While [`Gpu::with_plan_sink`] runs its closure the HIP runtime is in
//! *plan mode* (`hip_bridge::HipRuntime::plan_begin`): every launch that
//! reaches `HipRuntime::launch_kernel_blob` — which is every blob launch the
//! existing funnels (`launch_maybe_blob*`, `launch_blob_recorded`,
//! `Gpu::launch_kernel_blob`) and every hand-rolled capture-blob site make —
//! is recorded with the exact kernarg bytes, grid, block and LDS the ordinary
//! launch would have used, D2D copies and word-aligned memsets are recorded
//! as typed `Copy` / `Fill` ops, and **any other HIP API** (host transfers,
//! synchronization, allocation, module load, events, graphs, raw
//! `kernelParams` launches) refuses the whole plan.
//!
//! Plan mode is a separate flag. It neither sets nor reads
//! `graphs.capture_mode` or `replay.is_recording()`, so no route decision
//! flips: an owner that wants the *HIP-graph body* route (e.g. the DFlash
//! verify forward, whose route decisions read `capture_mode`) sets
//! `capture_mode` itself around the closure; the launch funnels check the plan
//! flag first, so nothing is captured or launched. Lazy scratch/JIT must
//! already be materialized (a `hipModuleLoad` / `hipMalloc` in plan mode is a
//! refusal): plans are built after one ordinary window of the same shape.

use crate::dispatch::Gpu;
use hip_bridge::{HipError, HipResult, PlanEvent};
use railgun::kernel::{DeclaredWord, PatchKind};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;

/// One planned kernel launch: the exact image the ordinary launch builds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedKernel {
    /// Loaded module that owns `symbol`.
    pub module: String,
    pub symbol: String,
    /// SHA-256 of the admitted code object `symbol` was loaded from.
    pub code_object_digest: [u8; 32],
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub dynamic_lds: u32,
    /// The launch's kernarg bytes (HIP `extra` ABI, tail-padded to 16).
    pub kernarg: Vec<u8>,
    /// Declared per-window patch sites. `PatchKind` node indices are the
    /// ordinal of the kernel among the plan's `PlannedOp::Kernel` ops; the
    /// integration owner rebases them when plans are concatenated.
    pub patches: Vec<PatchKind>,
    pub label: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlannedOp {
    Kernel(PlannedKernel),
    /// D2D copy between absolute device addresses.
    Copy {
        dst: u64,
        src: u64,
        bytes: u64,
    },
    /// 32-bit-word fill at an absolute device address.
    Fill {
        dst: u64,
        value: u32,
        words: u32,
    },
}

#[derive(Clone, Debug, Default)]
pub struct PlanSink {
    pub ops: Vec<PlannedOp>,
    /// Set when plan mode refused: the plan is discarded.
    pub refusal: Option<String>,
}

thread_local! {
    /// `(event index, declared words)` collected by the launch funnel.
    static DECLARED: RefCell<Vec<(usize, Vec<DeclaredWord>)>> = const { RefCell::new(Vec::new()) };
    /// `(event index, patches)` attached by `plan_annotate_last`.
    static PATCHES: RefCell<Vec<(usize, Vec<PatchKind>)>> = const { RefCell::new(Vec::new()) };
    /// `(op index in the sink, declared words)` of the last finished plan.
    static TAKEN_DECLARED: RefCell<Vec<(usize, Vec<DeclaredWord>)>> = const { RefCell::new(Vec::new()) };
}

/// Called by the launch funnel in plan mode with the launch's railgun-declared
/// dynamic words (`GdnFrame` etc.); `event_index` is the index the launch is
/// about to be recorded at.
pub(crate) fn note_declared(event_index: usize, words: &[DeclaredWord]) {
    if words.is_empty() {
        return;
    }
    DECLARED.with(|d| d.borrow_mut().push((event_index, words.to_vec())));
}

/// The railgun-declared words (`WordRole::GdnFrame`, ...) of the kernels of
/// the most recently finished plan, as `(op index in the sink, words)`.
/// Drains the record.
pub fn take_declared_words() -> Vec<(usize, Vec<DeclaredWord>)> {
    TAKEN_DECLARED.with(|t| std::mem::take(&mut *t.borrow_mut()))
}

impl Gpu {
    /// True while a plan sink is collecting on this thread.
    pub fn plan_mode_active(&self) -> bool {
        self.hip.plan_active()
    }

    /// Number of ops recorded so far in the active plan (0 outside plan mode).
    pub fn plan_op_count(&self) -> usize {
        self.hip.plan_event_count()
    }

    /// Label stamped on every kernel recorded from now on (plan mode only).
    pub fn plan_set_label(&mut self, label: &'static str) {
        self.hip.plan_set_label(label);
    }

    /// Refuse the active plan: it is discarded when `with_plan_sink` returns.
    pub fn plan_refuse(&mut self, why: &str) {
        self.hip.plan_refuse(why);
    }

    /// Attach declared patch sites to the most recent recorded kernel.
    pub fn plan_annotate_last(&mut self, patches: Vec<PatchKind>) {
        match self.hip.plan_last_launch_index() {
            Some(index) => PATCHES.with(|p| p.borrow_mut().push((index, patches))),
            None => self
                .hip
                .plan_refuse("plan_annotate_last with no recorded kernel"),
        }
    }

    /// Run `f` in plan mode and append what it would have issued to `sink`.
    ///
    /// On refusal (`sink.refusal` set, ops appended by this call dropped) or
    /// closure failure the call returns `Err`; nothing was launched either way.
    pub fn with_plan_sink<T>(
        &mut self,
        sink: &mut PlanSink,
        f: impl FnOnce(&mut Gpu) -> HipResult<T>,
    ) -> HipResult<T> {
        self.hip.plan_begin()?;
        DECLARED.with(|d| d.borrow_mut().clear());
        PATCHES.with(|p| p.borrow_mut().clear());
        // The capture-route launch sites push a copy of every blob into
        // `capture_blobs`; planning must not leave them behind.
        let blobs = self.graphs.capture_blobs.len();
        let result = f(self);
        let tape = self.hip.plan_end();
        self.graphs.capture_blobs.truncate(blobs);
        let declared = DECLARED.with(|d| std::mem::take(&mut *d.borrow_mut()));
        let mut patches = PATCHES.with(|p| std::mem::take(&mut *p.borrow_mut()));

        if let Some(mut why) = tape.refusal {
            if let Some(pos) = why.find("function handle 0x") {
                let hex: String = why[pos + 18..]
                    .chars()
                    .take_while(|c| c.is_ascii_hexdigit())
                    .collect();
                if let Ok(handle) = usize::from_str_radix(&hex, 16) {
                    if let Some(Some(name)) = self.loaded_function_names().get(&handle) {
                        why = format!("{why} ({name})");
                    }
                }
            }
            sink.refusal = Some(why.clone());
            return Err(HipError::new(0, &format!("plan refused: {why}")));
        }
        let value = result?;

        let names = self.loaded_function_names();
        let base = sink.ops.len();
        let mut ops: Vec<PlannedOp> = Vec::with_capacity(tape.events.len());
        let mut op_of_event: Vec<usize> = Vec::with_capacity(tape.events.len());
        for event in tape.events {
            let op = match event {
                PlanEvent::Launch {
                    func,
                    grid,
                    block,
                    shared_mem,
                    kernarg,
                    label,
                } => match self
                    .planned_kernel(&names, func, grid, block, shared_mem, kernarg, label)
                {
                    Ok(kernel) => PlannedOp::Kernel(kernel),
                    Err(why) => {
                        sink.refusal = Some(why.clone());
                        return Err(HipError::new(0, &format!("plan refused: {why}")));
                    }
                },
                PlanEvent::Copy { dst, src, bytes } => PlannedOp::Copy { dst, src, bytes },
                PlanEvent::Fill { dst, value, words } => {
                    let Ok(words) = u32::try_from(words) else {
                        let why = format!("fill of {words} words exceeds u32");
                        sink.refusal = Some(why.clone());
                        return Err(HipError::new(0, &format!("plan refused: {why}")));
                    };
                    PlannedOp::Fill { dst, value, words }
                }
            };
            op_of_event.push(ops.len());
            ops.push(op);
        }
        for (event, declared_patches) in patches.drain(..) {
            if let Some(PlannedOp::Kernel(k)) = op_of_event.get(event).and_then(|&i| ops.get_mut(i))
            {
                k.patches.extend(declared_patches);
            }
        }
        let taken: Vec<(usize, Vec<DeclaredWord>)> = declared
            .into_iter()
            .filter_map(|(event, words)| op_of_event.get(event).map(|&i| (base + i, words)))
            .collect();
        TAKEN_DECLARED.with(|t| *t.borrow_mut() = taken);
        sink.ops.extend(ops);
        Ok(value)
    }

    /// Loaded function handle -> symbol (`None` when two loaded symbols share
    /// one handle: ambiguous, refused).
    fn loaded_function_names(&self) -> HashMap<usize, Option<String>> {
        let mut names: HashMap<usize, Option<String>> = HashMap::new();
        for (name, function) in &self.functions {
            names
                .entry(function.raw_ptr() as usize)
                .and_modify(|slot| *slot = None)
                .or_insert_with(|| Some(name.clone()));
        }
        names
    }

    #[allow(clippy::too_many_arguments)]
    fn planned_kernel(
        &self,
        names: &HashMap<usize, Option<String>>,
        func: usize,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg: Vec<u8>,
        label: &'static str,
    ) -> Result<PlannedKernel, String> {
        let Some(Some(symbol)) = names.get(&func) else {
            return Err(format!(
                "launch of function handle {func:#x} with no unique loaded symbol"
            ));
        };
        let (module, artifact) = self
            .loaded_owner(symbol)
            .map_err(|e| format!("{symbol}: {e}"))?;
        Ok(PlannedKernel {
            module,
            symbol: symbol.clone(),
            code_object_digest: artifact.id().0,
            grid,
            block,
            dynamic_lds: shared_mem,
            kernarg,
            patches: Vec::new(),
            label,
        })
    }

    /// Start recording every *real* blob launch on this thread (they still
    /// execute / get graph-captured). The oracle's ground truth for what an
    /// ordinary window launches.
    pub fn launch_witness_begin(&mut self) -> HipResult<()> {
        self.hip.launch_witness_begin()
    }

    /// Stop the launch witness; the launches in order, with exact kernarg
    /// bytes, as [`PlannedKernel`]s.
    pub fn launch_witness_end(&mut self) -> Result<Vec<PlannedKernel>, String> {
        let events = self.hip.launch_witness_end();
        let names = self.loaded_function_names();
        let mut kernels = Vec::with_capacity(events.len());
        for event in events {
            if let PlanEvent::Launch {
                func,
                grid,
                block,
                shared_mem,
                kernarg,
                label,
            } = event
            {
                kernels.push(
                    self.planned_kernel(&names, func, grid, block, shared_mem, kernarg, label)?,
                );
            }
        }
        Ok(kernels)
    }

    /// The emitted-HIP twin: issue `ops` through ordinary HIP calls on the
    /// active stream (the null stream when none), in order. Kernels launch
    /// with the planned kernarg bytes; `Copy` / `Fill` use stream-ordered raw
    /// D2D copy / word fill. The caller synchronizes.
    pub fn run_planned_ops(&mut self, ops: &[PlannedOp]) -> HipResult<()> {
        self.bind_thread()?;
        if self.hip.plan_active() {
            return Err(HipError::new(
                0,
                "run_planned_ops: refused inside plan mode",
            ));
        }
        for op in ops {
            match op {
                PlannedOp::Kernel(k) => {
                    let func = self.functions.get(&k.symbol).ok_or_else(|| {
                        HipError::new(
                            0,
                            &format!("run_planned_ops: function '{}' not loaded", k.symbol),
                        )
                    })?;
                    let mut kernarg = k.kernarg.clone();
                    // SAFETY: the bytes are the exact kernarg image recorded
                    // from this loaded function; every pointee is owned by the
                    // caller's models/scratch and outlives the launch.
                    let r = unsafe {
                        self.hip.launch_kernel_blob(
                            func,
                            k.grid,
                            k.block,
                            k.dynamic_lds,
                            self.active_stream.as_ref(),
                            kernarg.as_mut_slice(),
                        )
                    };
                    r.map_err(|e| e.with_kernel(&k.symbol))?;
                }
                PlannedOp::Copy { dst, src, bytes } => {
                    let bytes = usize::try_from(*bytes).map_err(|_| {
                        HipError::new(0, "run_planned_ops: copy size exceeds usize")
                    })?;
                    // SAFETY: planned from live allocations of the caller.
                    let r = unsafe {
                        self.hip.memcpy_dtod_async_raw(
                            *dst as *mut c_void,
                            *src as *const c_void,
                            bytes,
                            self.active_stream.as_ref(),
                        )
                    };
                    r?;
                }
                PlannedOp::Fill { dst, value, words } => {
                    // SAFETY: planned from a live allocation of the caller.
                    let r = unsafe {
                        self.hip.memset_d32_async_raw(
                            *dst as *mut c_void,
                            *value as i32,
                            *words as usize,
                            self.active_stream.as_ref(),
                        )
                    };
                    r?;
                }
            }
        }
        Ok(())
    }
}

/// Symbols of the `Kernel` ops of `ops`, in order (`Copy` / `Fill` appear as
/// `<copy>` / `<fill>` so two plans compare op-for-op).
pub fn plan_symbols(ops: &[PlannedOp]) -> Vec<String> {
    ops.iter()
        .map(|op| match op {
            PlannedOp::Kernel(k) => k.symbol.clone(),
            PlannedOp::Copy { .. } => "<copy>".to_string(),
            PlannedOp::Fill { .. } => "<fill>".to_string(),
        })
        .collect()
}

/// One differing kernarg word between two plans of the same topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernargDiff {
    /// Index in the plan's op list.
    pub op: usize,
    /// Ordinal among the plan's kernels.
    pub kernel: usize,
    pub symbol: String,
    /// Byte offset of the differing 4-byte word in the kernarg image.
    pub offset: usize,
    pub a: u32,
    pub b: u32,
}

/// Compare two plans built at two different window shapes. `Err` when the
/// topologies differ (op kinds, symbols, grids, blocks, LDS, kernarg sizes,
/// copy/fill sizes); otherwise every differing kernarg dword and every
/// differing copy/fill address is reported, so a caller can check that
/// each one is a declared patch site.
pub fn diff_plans(a: &[PlannedOp], b: &[PlannedOp]) -> Result<Vec<KernargDiff>, String> {
    if a.len() != b.len() {
        return Err(format!("{} ops vs {} ops", a.len(), b.len()));
    }
    let mut diffs = Vec::new();
    let mut kernel = 0usize;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        match (x, y) {
            (PlannedOp::Kernel(p), PlannedOp::Kernel(q)) => {
                if p.symbol != q.symbol
                    || p.grid != q.grid
                    || p.block != q.block
                    || p.dynamic_lds != q.dynamic_lds
                    || p.kernarg.len() != q.kernarg.len()
                {
                    return Err(format!(
                        "op {i} kernel {} differs in topology ({} {:?}/{:?} lds {} len {} vs {} {:?}/{:?} lds {} len {})",
                        kernel,
                        p.symbol, p.grid, p.block, p.dynamic_lds, p.kernarg.len(),
                        q.symbol, q.grid, q.block, q.dynamic_lds, q.kernarg.len(),
                    ));
                }
                for (w, (cp, cq)) in p.kernarg.chunks(4).zip(q.kernarg.chunks(4)).enumerate() {
                    if cp != cq {
                        let word = |c: &[u8]| {
                            let mut buf = [0u8; 4];
                            buf[..c.len()].copy_from_slice(c);
                            u32::from_le_bytes(buf)
                        };
                        diffs.push(KernargDiff {
                            op: i,
                            kernel,
                            symbol: p.symbol.clone(),
                            offset: w * 4,
                            a: word(cp),
                            b: word(cq),
                        });
                    }
                }
                kernel += 1;
            }
            (PlannedOp::Copy { .. }, PlannedOp::Copy { .. })
            | (PlannedOp::Fill { .. }, PlannedOp::Fill { .. })
                if x == y => {}
            _ => {
                return Err(format!(
                    "op {i} differs in kind, extent or address (an undeclared Interior64 site)"
                ))
            }
        }
    }
    Ok(diffs)
}

impl Gpu {
    /// A `kernelParams`-form launch that plan mode can also record: outside
    /// plan mode it is the original raw launch (same stream, same params);
    /// in plan mode it goes through the blob funnel with `blob_builder`'s
    /// exact kernarg bytes (the same values, natural alignment).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_params_or_plan(
        &mut self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        params: &mut [*mut c_void],
        blob_builder: impl FnOnce() -> hip_bridge::KernargBlob,
    ) -> HipResult<()> {
        if self.hip.plan_active() {
            return self.launch_maybe_blob(
                func_name,
                grid,
                block,
                shared_mem,
                params,
                blob_builder,
            );
        }
        let func = &self.functions[func_name];
        // SAFETY: forwarded from the typed launch wrapper that assembled
        // `params` for this kernel signature.
        unsafe {
            self.hip
                .launch_kernel(func, grid, block, shared_mem, self.stream_ref(), params)
        }
    }
}
