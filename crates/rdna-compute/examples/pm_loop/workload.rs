// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! The contract between `pm_loop` (driver: certify, load, ABBA, JSON) and a
//! production-generator kernel module (`iu4.rs`, `moe.rs`): knob parsing, shape
//! grammar, emit/link of the candidate, the shipped baseline, deterministic
//! synthetic inputs, kernarg packing, and the CPU reference.
//!
//! Included with `#[path = "pm_loop/workload.rs"] mod workload;`.
#![allow(dead_code)]

use rdna_compute::Gpu;
use serde_json::Value;
use std::collections::BTreeMap;

pub type Result<T, E = String> = std::result::Result<T, E>;

/// Which kernel a launch/output belongs to. `Baseline` is timing arm A,
/// `Candidate` arm B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Baseline,
    Candidate,
}

/// One kernel dispatch exactly as the production launcher issues it.
#[derive(Clone, Debug)]
pub struct Launch {
    /// Complete kernarg blob (explicit args; the driver passes it to
    /// `hip.launch_kernel_blob`, which fills the hidden block).
    pub kernarg: Vec<u8>,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Dynamic LDS bytes.
    pub lds: u32,
}

/// A loadable code object (ELF `.co` or clang offload bundle `.hxaco`) and
/// the symbol the workload launches.
#[derive(Clone, Debug)]
pub struct Object {
    pub bytes: Vec<u8>,
    /// Kernel symbol launched by timing/correctness.
    pub launch_symbol: String,
    /// Every export of `bytes` (all certified for the candidate; resources
    /// are reported for each).
    pub exports: Vec<String>,
    /// JSON description: `{knobs:{..}}` for an emitted object, or
    /// `{artifact:"<path>"}` for a committed object.
    pub desc: Value,
    /// JSON provenance of the selection: `{source:"file:line", ..}`.
    pub source: Value,
}

/// Candidate-vs-CPU-reference result.
#[derive(Clone, Debug)]
pub struct RefReport {
    /// Human name of the reference (e.g. `cpu_iu4_fold_rn`).
    pub reference: String,
    pub max_abs_err: f64,
    pub tolerance: f64,
    /// `max_abs_err <= tolerance` and no NaN/Inf mismatch.
    pub pass: bool,
}

/// The `--kernel`s that take `--pool` / `--flush-l2` / `--flush` (cold timing with a rotating weight pool).
pub const COLD_KERNELS: [&str; 2] = ["iu4_gemm", "qkvzaset"];

/// Raw `--key value` flags the driver did not consume. A module `take`s the
/// flags it understands and the driver errors on whatever is left.
#[derive(Clone, Debug, Default)]
pub struct Knobs(pub BTreeMap<String, String>);

impl Knobs {
    pub fn take(&mut self, key: &str) -> Option<String> {
        self.0.remove(key)
    }
    /// `take`, or `default`.
    pub fn take_or(&mut self, key: &str, default: &str) -> String {
        self.take(key).unwrap_or_else(|| default.to_owned())
    }
    /// Error naming every flag nobody consumed.
    pub fn finish(self) -> Result<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "unknown option(s) for this kernel: {}",
                self.0
                    .keys()
                    .map(|k| format!("--{k}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        }
    }
}

/// One production-generator kernel under test. Constructed GPU-free (knob
/// parsing + shape grammar + candidate/baseline selection); `setup` then
/// allocates and fills the device inputs.
pub trait Workload {
    /// `--kernel` name.
    fn kernel(&self) -> &'static str;
    /// Target arch (`--arch`; `gfx1201` unless the workload supports more).
    fn arch(&self) -> &'static str;
    /// Parsed shape as JSON (every dimension the launch derives from).
    fn shape_json(&self) -> Value;
    /// Candidate knobs as JSON (the resolved Spec axes).
    fn candidate_knobs(&self) -> Value;
    /// Extra top-level record fields (e.g. `sparse`), merged by the driver.
    fn record_extra(&self) -> Vec<(String, Value)> {
        Vec::new()
    }

    /// Emit + link the candidate with the native writer (the path of
    /// `peacemaker-isa emit --co`). Errors (e.g. a closed fold axis) are
    /// reported verbatim by the driver.
    fn candidate(&self) -> Result<Object>;
    /// The shipped baseline (what production selects for `arch()`), or the
    /// `--base-*` knob variant. `source` names file:line / artifact + sha256.
    fn baseline(&self) -> Result<Object>;

    /// Allocate device buffers and upload deterministic synthetic inputs.
    fn setup(&mut self, gpu: &Gpu) -> Result<()>;
    /// Launch parameters for `arm` (kernarg points at that arm's buffers).
    fn launch(&self, arm: Arm) -> Launch;
    /// Restore `arm`'s accumulating state (e.g. `Y` for epi add; zero `Y`
    /// otherwise) to its pre-launch bytes. Called outside event intervals.
    fn reset(&self, gpu: &Gpu, arm: Arm) -> Result<()>;
    /// Download `arm`'s output (the payload the correctness check covers).
    fn output(&self, gpu: &Gpu, arm: Arm) -> Result<Vec<u8>>;
    /// Compare a downloaded output to the CPU reference.
    fn check(&self, out: &[u8]) -> Result<RefReport>;
    /// [`Workload::check`] of `arm`'s output, for a workload whose arms follow
    /// different references (a numeric candidate against an exact baseline).
    fn check_arm(&self, _arm: Arm, out: &[u8]) -> Result<RefReport> {
        self.check(out)
    }
    /// Declared numeric contract `{baseline, candidate, relation}` of the record
    /// (`relation` `"exact"` or `"numeric"`, as [`ChainWorkload::numeric_contract`]);
    /// `Null`: the workload declares none and the record carries no
    /// `numeric_contract` field.
    fn numeric_contract(&self) -> Value {
        Value::Null
    }
    /// Max absolute element difference between two outputs (interpreting the
    /// bytes in the kernel's output dtype); `inf` if any element differs in
    /// NaN-ness. 0.0 iff element-wise equal.
    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64>;

    /// `true` when correctness requires the candidate's output to be bitwise
    /// equal to the shipped baseline in addition to passing the CPU reference.
    fn requires_baseline_identity(&self) -> bool {
        false
    }

    /// `--emu`: run the candidate on the peacemaker emulator at a small
    /// shape if every instruction is qualified; otherwise
    /// `Value::String("refused (<reason>)")`. Never fabricate a result.
    fn emu(&self) -> Value;

    /// `--emu`, pre-card gate: run `cand` and the incumbent `base` object on the peacemaker emulator
    /// (complete grid, every buffer its own exact-size region with unmapped guard gaps, definedness-tracked
    /// LDS) and return the `emu_gate` record `{status: "pass"|"fail", reason, shape, instructions,
    /// workgroups, cand_out_sha256, base_out_sha256, ..}`. `None`: this workload has no emulator gate.
    fn emu_gate(&self, _cand: &Object, _base: &Object) -> Option<Value> {
        None
    }

    /// `--pool` / `--flush-l2` / `--flush` (cold timing, `support/cold.rs`): called once after
    /// construction, before `setup`, with the mode (never `Hot`). A workload that supports it
    /// allocates a rotating weight pool in `setup` ([`Workload::pool_sets`] sets whose total exceeds
    /// the last-level cache) and records the mode with `cold::annotate_shape` in its `shape_json`;
    /// the driver evicts the caches between timed dispatches. The default refuses.
    fn set_cold(&mut self, mode: crate::cold::ColdMode) -> Result<()> {
        Err(format!(
            "{} is not supported by --kernel {} (cold timing: {})",
            mode.flag(),
            self.kernel(),
            COLD_KERNELS.join(", ")
        ))
    }
    /// Distinct weight sets a timing pool cycles (1: no pool, every dispatch reads the same weights).
    fn pool_sets(&self) -> usize {
        1
    }
    /// [`Workload::launch`] with weight set `set` (`0..pool_sets()`; set 0 is the correctness set the
    /// CPU reference reads). The driver issues timed dispatch `i` on set `i % pool_sets()`.
    fn launch_set(&self, arm: Arm, _set: usize) -> Launch {
        self.launch(arm)
    }
    /// A NULL kernel with the baseline's resource descriptor (VGPRs, LDS, kernarg size) whose body is
    /// a single `s_endpgm`, launched with [`Workload::launch_set`]'s grid, block, LDS and kernarg: its
    /// hipEvent time is the fixed dispatch cost the driver reports as `timing.null_dispatch`.
    /// `Ok(None)`: this workload has none.
    fn null_object(&self) -> Result<Option<Object>> {
        Ok(None)
    }
}

/// Construction context the driver hands a [`ChainWorkload`].
#[derive(Clone, Copy, Debug)]
pub struct ChainCtx {
    /// `--arch` (`gfx1201` default): the target the chain is selected for; the
    /// driver refuses a device of another arch.
    pub arch: &'static str,
    /// `--null-control aa`: the candidate arm runs the BASELINE chain (same
    /// launchers, same selector, same loaded modules) into the candidate arm's
    /// own buffers, so the record is an A/A null of the whole chain.
    pub null_control: bool,
}

/// R3 complete-chain workload. Each arm is an ordered multi-kernel chain
/// enqueued on the default stream through `rdna_compute::Gpu`'s public
/// launchers (the baseline arm: the chain the shipped selector issues at this
/// shape/arch/residency), timed as ONE unit per launch slot (hipEvents around
/// the whole chain). Constructed GPU-free (shape grammar, knob parsing, arm
/// resolution); `setup` allocates and fills deterministic inputs.
///
/// Every chain workload is eager today (`mode() == "eager"`); graph replay is
/// a later additive mode.
pub trait ChainWorkload {
    /// `--kernel` name.
    fn kernel(&self) -> &'static str;
    /// Target arch (from [`ChainCtx::arch`]).
    fn arch(&self) -> &'static str;
    /// Execution mode of both arms (`"eager"`).
    fn mode(&self) -> &'static str {
        "eager"
    }
    /// Parsed shape as JSON (every dimension the chain derives from,
    /// including residency).
    fn shape_json(&self) -> Value;
    /// Candidate knobs as JSON.
    fn candidate_knobs(&self) -> Value;
    /// `{baseline: {...}, candidate: {...}}`: per arm the ordered stages
    /// (`[{stage, launcher, symbols, origin: "hip"|"pm", source}]`), the
    /// selector provenance (file:line of every predicate that picked a
    /// stage), residency, scratch bytes. Called after `setup` (selection can
    /// depend on the device).
    fn chain_json(&self) -> Value;
    /// Declared numeric contract: `{baseline, candidate, relation}` where
    /// `relation` is `"exact"` (the candidate must reproduce the baseline's
    /// bytes) or `"numeric"` (a different recipe: `identical_to_baseline` is
    /// reported honestly and each arm passes on its own reference).
    fn numeric_contract(&self) -> Value;
    /// PM code objects `arm` launches (every export is M7-certified by the
    /// driver for the candidate arm); empty when the arm launches no PM object.
    fn pm_objects(&self, arm: Arm) -> Result<Vec<Object>>;

    /// Allocate device buffers (two complete buffer sets, outputs and
    /// scratch, one per arm under the current binding) and upload
    /// deterministic synthetic inputs.
    fn setup(&mut self, gpu: &mut Gpu) -> Result<()>;
    /// Crossover binding of arms to buffer sets: `false` binds the baseline to
    /// set 0 and the candidate to set 1, `true` swaps them. `encode_chain`,
    /// `reset` and `outputs` follow the current binding. The driver flips it
    /// every ABBA round so a buffer-placement effect alternates sign instead of
    /// biasing one arm (an A/A with a fixed binding measured up to 0.9%).
    fn set_buffer_swap(&mut self, swapped: bool);
    /// Enqueue `arm`'s complete chain on the default stream. No host
    /// synchronization, allocation or upload inside (the driver brackets it
    /// with events).
    fn encode_chain(&mut self, gpu: &mut Gpu, arm: Arm) -> Result<()>;
    /// Restore `arm`'s accumulating state (e.g. the residual the combine adds
    /// into) to its pre-chain bytes. Called outside event intervals.
    fn reset(&mut self, gpu: &mut Gpu, arm: Arm) -> Result<()>;
    /// Download `arm`'s outputs (every byte the correctness check covers,
    /// including poisoned guard/tail regions).
    fn outputs(&self, gpu: &mut Gpu, arm: Arm) -> Result<Vec<u8>>;
    /// Compare `arm`'s downloaded outputs to that arm's CPU reference.
    fn reference(&self, arm: Arm, out: &[u8]) -> Result<RefReport>;
    /// Max absolute element difference between two output payloads; `inf`
    /// on a NaN-ness mismatch; 0.0 iff element-wise equal.
    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64>;
    /// `true` iff `numeric_contract().relation == "exact"`.
    fn requires_baseline_identity(&self) -> bool;
}
