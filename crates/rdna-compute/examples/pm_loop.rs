// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! PM loop: one command that emits a PeaceMaker production-generator variant,
//! certifies it, checks it on the card, and ABBA-times it against the shipped
//! kernel (gfx1201 by default; `--arch gfx1151|gfx1100` for the tri-arch kernels
//! `qwen4_moe_sym` and `qwen4_moe_verify`). Prints ONE JSON object (schema 1) on stdout.
//!
//! ```text
//! pm_loop --kernel iu4_gemm|qkvzaset|mq6_i8_gemm|fp8_gemm|fa_prefill|qwen4_moe_sym|qwen4_moe_verify --shape <S> [kernel knobs]
//!         [--arch gfx1201|gfx1151|gfx1100] [--base-<knob> V ...] [--rip FILE.rip] [--time N] [--rounds R] [--flush]
//!         [--cand-object FILE: qwen4_moe_sym --arch gfx1100 only, a prebuilt candidate object]
//!         [--max-rounds R] [--time-budget-s S] [--min-effect F] [--emu] [--offline] [--out DIR]
//!         [--null-control aa] [--noise-floor FILE]
//!         (--offline: stop after step 2 below, no device opened, record `offline: true`; the
//!          legality screen of `pm-census tune run`; not qwen4_moe_verify)
//!         (--rip: iu4_gemm, mq6_i8_gemm, fp8_gemm, fa_prefill, qwen4_moe_sym and the PM producers
//!          iu4_c8_producer_pm / moe_rotate_i4_pm / gdn_chunk_prep_pm; not qwen4_moe_verify)
//! ```
//!
//! `--arch` (default `gfx1201`) is the target the candidate is emitted, M7-certified and
//! timed for; the record's top-level `arch` is the workload's arch, and a device of another
//! arch is refused (`device is X, --arch Y`, stage `device`). Only `qwen4_moe_sym` and
//! `qwen4_moe_verify` are tri-arch; every other kernel with `--arch` other than gfx1201 is a
//! usage error.
//!
//! Pipeline (every step lands in the JSON record):
//! 1. Emit + link the candidate with the native writer (the path of
//!    `peacemaker-isa emit --co`, no ROCm tools) and resolve the baseline: the
//!    SHIPPED production object (committed `.hxaco` that `rdna-compute` embeds;
//!    `baseline.source` names the selecting file:line and the artifact sha256),
//!    or, when any `--base-<knob>` is given, a second emitted variant whose
//!    unspecified knobs inherit the candidate's (time two arbitrary variants).
//! 2. Certify: M7 (`pm_check`, behind `peacemaker-isa emit --m7`) on EVERY
//!    export of the candidate object through the cert cache (`m7.cache` is
//!    `hit`, `miss` or `disabled`; `HIPFIRE_CERT_CACHE=0` disables it); the
//!    lifted kernel descriptor/metadata give VGPRs, SGPRs, LDS, scratch and code
//!    bytes per export, plus `cost_lint` loop statistics.
//! 3. Correctness on the card: deterministic synthetic inputs at `--shape`,
//!    candidate and baseline each launched once with the production kernarg
//!    packing; `correctness` reports candidate vs the CPU reference
//!    (`max_abs_err`, `tolerance`, `pass`), the baseline vs the same
//!    reference, and candidate vs baseline (`identical_to_baseline`,
//!    `max_abs_diff_vs_baseline`). An epilogue with no CPU reference (iu4_gemm
//!    silu) reports `reference: "none"` and `pass` is then
//!    `identical_to_baseline` (`pass_basis` says which). `--emu` runs the candidate on the peacemaker emulator
//!    when every instruction is qualified, else `refused (<reason>)`.
//!    **Pre-card emulator gate** (`--emu`, with or without `--offline`; `--kernel qwen4_moe_sym` today): before any
//!    device is opened the candidate AND the shipped incumbent object run their whole grid on the emulator at
//!    tiny shapes (aligned and ragged `T`), each buffer an exact-size region with >= 128 KiB unmapped guard gaps,
//!    LDS definedness tracked per workgroup, under the emulator's gate model (`Emulator::hw_model`, no opt-out):
//!    loads retire only at their `s_waitcnt` (`[in-flight]`), cross-wave LDS accesses need a retired access and a
//!    barrier epoch between them (`[race]`), every workgroup runs under round-robin, run-ahead-per-wave and
//!    seeded-random wave orders with bit-identical stores (`[schedule]`). The record's `emu_gate` is `pass` only if
//!    there is no access outside a mapped buffer, no LDS access out of bounds or read of an LDS byte the workgroup
//!    never wrote, no barrier deadlock or instruction-limit stop, every opcode is executable (a refusal names it),
//!    none of those hazards, the candidate output is
//!    bit-identical to the emulated incumbent's and both pass the CPU reference. A failed gate is stage
//!    `emu_gate` (exit 3) and the card is never touched. gfx1100 runs use the gfx1151 WMMA/transcendental model
//!    (`Emulator::gfx1100_as_gfx1151`; recorded in `emu_gate.numeric_model`) with narrowed weight scales.
//! 4. Timing: ABBA, A = baseline, B = candidate, like `pm_decode_twins time`:
//!    one checked warm launch per arm, then `--rounds R` (default 6) ABBA
//!    rounds of `--time N` (default 50) launches per block, per-launch
//!    hipEvents on the default stream, accumulating output state restored
//!    outside the event interval. Each block of N launches has a block
//!    median, and each ABBA round pairs the candidate's two block medians with
//!    the baseline's two from that same round (ratio = candidate mean / baseline
//!    mean; this cancels between-round clock and power drift, which is several
//!    times larger than a few-percent effect). `separable` is true iff the exact
//!    two-sided sign test over those paired rounds has `sign_p <= 1e-3` (ties
//!    count against it, so at least 11 rounds are needed); `direction` is
//!    `faster`/`slower` from the majority sign, with `rounds_faster/slower/tied`
//!    and `round_ratio_min/max`. `block_ratio` = median(candidate block
//!    medians) / median(baseline block medians). The stricter
//!    `separable_block_disjoint` (every candidate block median on one side of
//!    every baseline block median) and the twins' full-range rule
//!    `separable_full_range` (unchanged for `pm_decode_twins`; one hiccup launch
//!    vetoes it) stay in the record. Not separable: ABBA rounds keep being added
//!    (doubling) until `separable`, `--max-rounds` (default 256) or
//!    `--time-budget-s` (default 240, under the fabric's timing cap) is reached;
//!    `rounds_run`, `extended` and `stop` report it. The sign test also
//!    detects real but immaterial placement and order effects (two arms with
//!    byte-identical code objects came out separable at 0.2-0.3%), so
//!    `material` = `separable` AND `|block_ratio - 1| >= --min-effect`
//!    (default 0.005); the human verdict says `faster/slower (material)` or
//!    `separable but below min-effect`. Sampling is refused when the
//!    candidate fails correctness.
//!
//!    Noise fields (additive, schema stays 1): `timing.noise` carries
//!    `round_ratios` (the per-ABBA-round candidate/baseline ratios the sign test
//!    runs over), `round_ratio` / `baseline_block_median` /
//!    `candidate_block_median` (dispersion objects `{n, median, mad, iqr, p05,
//!    p95, min, max, cv, log_sd}`; block medians in us, `mad` unscaled, quantiles
//!    linearly interpolated, `log_sd` = sample sd of ln) and
//!    `baseline_drift_per_block` / `candidate_drift_per_block` (OLS slope of the
//!    block median over the block index, divided by the mean block median).
//!
//!    `--null-control aa`: A/A run. No candidate is emitted (no `--rip`, no
//!    `--base-*`: usage error); the candidate object is a clone of the resolved
//!    baseline object (same bytes, launch symbol and exports) loaded as a
//!    SEPARATE module, so the two arms are distinct loaded objects of identical
//!    code. The launch layout (grid/kernarg) still comes from the workload's
//!    candidate knobs, so pass knobs that describe the baseline launch; the
//!    correctness stage (`identical_to_baseline`) enforces that. The record gains
//!    top-level `null_control: {kind: "aa", identical_objects, baseline_sha256,
//!    ..}`; the candidate's M7 certification is the baseline's (same bytes) and
//!    `candidate.m7.reused_from` says so. Everything else is unchanged.
//!
//!    `--noise-floor FILE`: a `pm-census noise` output (`hipfire.pm_noise.v1`).
//!    The A/A group with the same kernel, shape and baseline sha256 supplies
//!    `timing.vs_noise_floor = {floor, effect: |block_ratio - 1|, exceeds,
//!    group}`, or `{status: "no matching A/A group"}` (also `{status, group}` when
//!    the matching group has too few runs for a floor). It never changes
//!    `material`.
//!
//! Complete chains (`--kernel qwen4_moe_verify`, `pm_loop/moe_verify.rs`): each arm is an
//! ordered multi-kernel chain enqueued through `rdna_compute::Gpu`'s public launchers (the
//! baseline arm is the chain the shipped selector issues; the candidate arm swaps in PM
//! objects). The record keeps schema 1 and adds `mode` (`eager`), `numeric_contract`,
//! `chain` (per-arm stages, selector provenance, residency, scratch), and per arm
//! `sha256` (sha256 of the canonical JSON of `chain.<arm>`; the pm-census noise group key)
//! and `pm_objects` (one entry per PM object the arm launches: M7 per export, resources,
//! cost lint). Every candidate export is M7-certified (a fail is stage `m7`, exit 3); the
//! baseline's certification is recorded, not gating. Correctness runs each arm's chain
//! twice: warm on each arm's own buffer set, repeat on the swapped sets (`correctness` adds
//! `baseline_repeat_identical`, `candidate_repeat_identical`, `repeat_on_swapped_buffers`
//! and `numeric_relation`); `pass` needs both arms to pass their own CPU reference, both
//! repeats bit-identical, and (relation `exact`) candidate == baseline. `emu` is `not
//! applicable (chain)`. Timing is the same ABBA machinery with hipEvents around the WHOLE
//! chain per launch slot (`timing.unit` says so), accumulating state restored outside the
//! interval, as a crossover: the arms swap buffer sets every ABBA round
//! (`ChainWorkload::set_buffer_swap`), because a fixed binding let buffer placement alone
//! separate an A/A by up to 0.9% (gfx1201, host-mapped T=2/3). `timing.crossover` splits the
//! round ratios by binding (`round_ratio_own`, `round_ratio_swapped`) and reports
//! `crossover_ratio` (geometric mean of their medians, placement cancelled) and
//! `placement_ratio`. `--null-control aa` runs the BASELINE chain as the candidate arm
//! through the same launchers and loaded modules (no `--rip`/`--base-*`); the record gains
//! `null_control: {kind: "aa", identical_chains, separate_modules: false, baseline_sha256,
//! candidate_sha256, note}` and the candidate's M7 is the baseline's (`reused_from:
//! "baseline"`). `--rip` is refused for chains.
//!
//! Route judge (`--kernel qwen4_moe_route --arch gfx1100`, `pm_loop/moe_route.rs`; `--offline` plans
//! the launches and certifies the candidate's shipped PM bundles without a device): baseline = the
//! shipped F16 grouped WMMA routed-MoE route of an S1 window, candidate = the symmetric-IU4 PM chain
//! (`HIPFIRE_QWEN4_MOE_SYM_IU4=1`), the production panel schedule over one synthetic route. No
//! identity gate (A4 vs F16); the record carries per-stage times, the ABBA chain ratio and the
//! numeric distance of the two routes. `--rip`, `--emu` are refused; `--null-control aa` runs the
//! baseline chain as the candidate and requires bit identity.
//!
//! Cold timing (`--pool`, `--flush-l2`, `--flush`; `--kernel iu4_gemm|qkvzaset`; `support/cold.rs`, shared with
//! `hip_loop`): the default loop is HOT: the same weights back to back, so a verify-shape matrix of at most ~47 MB
//! stays resident in the 64 MB Infinity Cache and a DRAM-bound change reads far smaller than in the model (the shipped
//! `_v64` down: 62.6 us here, 97.6 us in-model). Modes, each a superset of the previous:
//! * `--pool`: only the weights are cold. A pool of distinct seeded weight sets whose total exceeds 192 MiB (dispatch
//!   `i`, either arm, uses set `i % sets`; the set count is odd so the ABBA arms do not alias set parity) exceeds
//!   L2 + MALL; activations, instructions and the TLB entries of the small buffers stay warm.
//! * `--flush-l2`: the pool plus, after every timed dispatch, a device-to-device copy of 2 x the device L2 per buffer
//!   (the `hipDeviceAttributeL2CacheSize` attribute, else the documented 8 MiB of gfx1201; recorded as
//!   `timing.l2_bytes` / `l2_source`): evicts L2, not the MALL.
//! * `--flush`: the pool plus the full 2 x 96 MiB copy: also MALL, TLB, instruction and scalar caches. (`--flush` and
//!   `--flush-l2` are exclusive; `--pool` combines with either.)
//! The copy is enqueued OUTSIDE the hipEvent interval. Set 0 of the pool is the correctness set (the CPU reference
//! reads it); the arms are additionally compared bit for bit on every other pool member (`correctness.pool =
//! {sets, identical, per_set_identical, max_abs_diff_vs_baseline}`; a candidate identical on set 0 must stay
//! identical on all of them), and the post-sampling check re-runs set 0. Recording: `shape.timing_pool` for every
//! pooled mode, `shape.flush` (true only for `--flush`) and `shape.flush_l2` (true only for `--flush-l2`) (only when
//! pooled: a hot run's `shape` keeps its bytes, so `pm-census noise` groups still match; the four modes never share a
//! shape), `timing.cold_mode` / `timing.flush` always, `timing.flush_method` and `flush_bytes_per_buffer` when
//! flushing.
//!
//! `timing.null_dispatch` (`iu4_gemm`, `qkvzaset`, every mode): a NULL kernel with the baseline's emitted descriptor
//! (VGPRs, LDS, kernarg) and a body of one `s_endpgm`, launched with the workload's own grid, block, LDS and kernarg
//! in the same process and cold mode, hipEvent-bracketed like a timed arm (400 launches after 40 warm-up). Every
//! `*_us` in the record contains this fixed host-launch + dispatch cost; it is REPORTED (`subtracted: false`), never
//! removed. `--pool|--flush-l2|--flush` with a 2:4 sparse arm is refused (the packed image has no pool); other kernels
//! refuse the flags.
//!
//! Exit status: 0 ok; 1 correctness failure (record still printed); 3 the
//! candidate could not be emitted/certified/launched (record printed with the
//! error); 2 usage error.
//!
//! Kernel knobs and shape grammar: see the doc comment of `pm_loop/qkvzaset.rs` (`--kernel qkvzaset`, the fused
//! verify GDN projection, 120-byte ABI, identity vs the shipped `_b1s`/`_v64` export) and of `pm_loop/iu4.rs`
//! (`--kernel iu4_gemm`), `pm_loop/mq6.rs` (`--kernel mq6_i8_gemm`), `pm_loop/fp8.rs`
//! (`--kernel fp8_gemm`), `pm_loop/fa.rs` (`--kernel fa_prefill`) and `pm_loop/moe.rs`
//! (`--kernel qwen4_moe_sym`).
//!
//! Use (build once, then take a gfx1201 card through the fabric; check
//! `fabric-run.sh --status` first and pick an idle hiptrx card A/B/C/E):
//! ```text
//! cargo build --release -p rdna-compute --features lab --example pm_loop
//! /home/kaden/qcal/locks/fabric/fabric-run.sh --class timing --card B --mem-gb 4 \
//!   --max-secs 900 --owner PmLoop -- target/release/examples/pm_loop \
//!   --kernel iu4_gemm --fold k128 --tile 256x128x16 --epi set \
//!   --shape M=17408,N=512,K=5120 --time 50 --rounds 6
//! ```
//! 2:4-sparse candidate vs the shipped dense kernel (pruned weights), and a `.rip` candidate:
//! ```text
//! pm_loop --kernel iu4_gemm --mma s24x64 --epi set --shape M=17408,N=512,K=5120 --emu
//! pm_loop --kernel iu4_gemm --mma s24x32 --epi add --shape M=5120,N=512,K=17408
//! pm_loop --kernel iu4_gemm --rip my_twin.rip --mma s24x64 --shape M=17408,N=512,K=5120
//! ```
//! The fabric exports `ROCR_VISIBLE_DEVICES`/`FABRIC_CARDS`, recorded under
//! `device`. GPU-free tests: `cargo test -p rdna-compute --features lab --example pm_loop`.

#[path = "support/abba.rs"]
mod abba;
#[path = "pm_loop/c8.rs"]
mod c8;
#[path = "support/cold.rs"]
mod cold;
#[path = "pm_loop/common.rs"]
mod common;
#[path = "pm_loop/fa.rs"]
mod fa;
#[path = "pm_loop/fp8.rs"]
mod fp8;
#[path = "pm_loop/gdn.rs"]
mod gdn;
#[path = "pm_loop/iu4.rs"]
mod iu4;
#[path = "pm_loop/moe.rs"]
mod moe;
#[path = "pm_loop/moe_route.rs"]
mod moe_route;
#[path = "pm_loop/moe_verify.rs"]
mod moe_verify;
#[path = "pm_loop/moe_verify_iu4.rs"]
mod moe_verify_iu4;
#[path = "pm_loop/mq6.rs"]
mod mq6;
#[path = "pm_loop/prod.rs"]
mod prod;
#[path = "pm_loop/qkvzaset.rs"]
mod qkvzaset;
#[path = "pm_loop/rotate.rs"]
mod rotate;
#[path = "pm_loop/workload.rs"]
mod workload;

use abba::{timing_relation, timing_schedule, timing_stats, TimingArm, TimingStats};
use rdna_compute::Gpu;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use workload::{Arm, ChainCtx, ChainWorkload, Knobs, Object, Result, Workload};

const SCHEMA: u32 = 1;
/// The kernel that takes the complete-chain path (`run_chain`).
const CHAIN_KERNEL: &str = "qwen4_moe_verify";
/// `--arch` values.
const ARCHES: [&str; 3] = ["gfx1201", "gfx1151", "gfx1100"];
/// Kernels that accept an `--arch` other than gfx1201 (`qwen4_moe_route` only gfx1100, enforced by
/// `moe_route::check_cli`).
const TRI_ARCH_KERNELS: [&str; 3] = ["qwen4_moe_sym", "qwen4_moe_verify", "qwen4_moe_route"];
const DEFAULT_TIME: usize = 50;
const DEFAULT_ROUNDS: usize = 6;
const DEFAULT_MAX_ROUNDS: usize = 256;
/// Below the fabric timing cap (1800 s) and the usual `--max-secs 900`.
const DEFAULT_TIME_BUDGET_S: f64 = 240.0;
/// Smallest |block_ratio - 1| that counts as a material difference (0.5%): the paired
/// sign test also detects real but immaterial placement/order effects (two arms with
/// byte-identical code objects came out "separable" at 0.2-0.3%).
const DEFAULT_MIN_EFFECT: f64 = 0.005;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---------------------------------------------------------------------------
// CLI

#[derive(Debug)]
struct Opts {
    kernel: String,
    /// `--arch`: target arch (default gfx1201).
    arch: &'static str,
    shape: Option<String>,
    time: usize,
    rounds: usize,
    /// Auto-extension ceiling for the total ABBA rounds.
    max_rounds: usize,
    /// Wall-clock budget of the sampling phase, seconds.
    time_budget_s: f64,
    /// `--min-effect`: `material` needs `|block_ratio - 1| >=` this.
    min_effect: f64,
    emu: bool,
    out: Option<PathBuf>,
    /// `--rip FILE`: candidate compiled from a `.rip` source (iu4_gemm only).
    rip: Option<PathBuf>,
    /// `--cand-object FILE`: a prebuilt code object used as the candidate instead of the emit
    /// (`--kernel qwen4_moe_sym --arch gfx1100` only; built from a candidate copy of the gfx11 NT1
    /// module by the standalone emitter project, `pm-krsi-1100/emit`).
    cand_object: Option<PathBuf>,
    /// `--null-control aa`: time the baseline object against a clone of itself.
    null_control: bool,
    /// `--noise-floor FILE`: a `pm-census noise` document.
    noise_floor: Option<PathBuf>,
    /// Parsed `noise_floor` (loaded by `main` before the run).
    noise_floor_doc: Option<Value>,
    /// `--offline`: stop after emit + M7 + baseline resolution (no device).
    offline: bool,
    /// `--pool` / `--flush-l2` / `--flush`: cold-cache timing (`support/cold.rs`): a rotating weight
    /// pool whose total exceeds the last-level cache, plus (L2: 2 x the device L2, full: 2 x 96 MiB)
    /// a device copy after every timed dispatch, outside the event interval; `iu4_gemm` and
    /// `qkvzaset` only.
    cold: cold::ColdMode,
    knobs: Knobs,
}

const DRIVER_FLAGS: &str =
    "--kernel K --shape S [--arch gfx1201|gfx1151|gfx1100] [--time N] [--rounds R] [--max-rounds R] [--time-budget-s S] [--min-effect F] [--emu] [--offline] [--pool | --flush-l2 | --flush] [--out DIR] [--rip FILE.rip] [--cand-object FILE] [--null-control aa] [--noise-floor FILE] [--inputs capture:ACT_CAPTURE.json[@site]] [--<knob> V]... [--base-<knob> V]...";

fn parse_args(mut a: impl Iterator<Item = String>) -> Result<Opts> {
    let mut o = Opts {
        kernel: String::new(),
        arch: ARCHES[0],
        shape: None,
        time: DEFAULT_TIME,
        rounds: DEFAULT_ROUNDS,
        max_rounds: DEFAULT_MAX_ROUNDS,
        time_budget_s: DEFAULT_TIME_BUDGET_S,
        min_effect: DEFAULT_MIN_EFFECT,
        emu: false,
        out: None,
        rip: None,
        cand_object: None,
        null_control: false,
        noise_floor: None,
        noise_floor_doc: None,
        offline: false,
        cold: cold::ColdMode::Hot,
        knobs: Knobs::default(),
    };
    while let Some(arg) = a.next() {
        let key = arg
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument {arg} (usage: {DRIVER_FLAGS})"))?
            .to_owned();
        if key == "emu" {
            o.emu = true;
            continue;
        }
        if key == "offline" {
            o.offline = true;
            continue;
        }
        let mode = match key.as_str() {
            "pool" => Some(cold::ColdMode::Pool),
            "flush-l2" => Some(cold::ColdMode::FlushL2),
            "flush" => Some(cold::ColdMode::Flush),
            _ => None,
        };
        if let Some(mode) = mode {
            o.cold = o.cold.combine(mode)?;
            continue;
        }
        let value = a
            .next()
            .ok_or_else(|| format!("--{key} needs a value (usage: {DRIVER_FLAGS})"))?;
        match key.as_str() {
            "kernel" => o.kernel = value,
            "arch" => {
                o.arch = ARCHES
                    .iter()
                    .copied()
                    .find(|a| *a == value)
                    .ok_or_else(|| {
                        format!("--arch {value}: expected one of {}", ARCHES.join("|"))
                    })?
            }
            "shape" => o.shape = Some(value),
            "time" => o.time = abba::positive_count(&value).map_err(|e| format!("--time: {e}"))?,
            "rounds" => {
                o.rounds = abba::positive_count(&value).map_err(|e| format!("--rounds: {e}"))?
            }
            "max-rounds" => {
                o.max_rounds =
                    abba::positive_count(&value).map_err(|e| format!("--max-rounds: {e}"))?
            }
            "time-budget-s" => {
                o.time_budget_s = value
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .ok_or_else(|| format!("--time-budget-s {value}: expected seconds > 0"))?
            }
            "min-effect" => {
                o.min_effect = value
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .ok_or_else(|| format!("--min-effect {value}: expected a fraction >= 0"))?
            }
            "out" => o.out = Some(PathBuf::from(value)),
            "rip" => o.rip = Some(PathBuf::from(value)),
            "cand-object" => o.cand_object = Some(PathBuf::from(value)),
            "null-control" => {
                if value != "aa" {
                    return Err(format!(
                        "--null-control {value}: the only null control is `aa`"
                    ));
                }
                o.null_control = true;
            }
            "noise-floor" => o.noise_floor = Some(PathBuf::from(value)),
            _ => {
                if o.knobs.0.insert(key.clone(), value).is_some() {
                    return Err(format!("--{key} given twice"));
                }
            }
        }
    }
    if o.kernel.is_empty() {
        return Err(format!("missing --kernel (usage: {DRIVER_FLAGS})"));
    }
    if o.kernel == moe_route::KERNEL {
        moe_route::check_cli(o.arch, o.emu)?;
    }
    if o.arch != ARCHES[0] && !TRI_ARCH_KERNELS.contains(&o.kernel.as_str()) {
        return Err(format!(
            "--kernel {} targets gfx1201 only (got --arch {})",
            o.kernel, o.arch
        ));
    }
    if o.kernel == CHAIN_KERNEL && o.rip.is_some() {
        return Err(format!(
            "--rip is not supported with --kernel {CHAIN_KERNEL}: there are no .rip candidates for chains yet"
        ));
    }
    if o.kernel == CHAIN_KERNEL && o.offline {
        return Err(format!(
            "--offline is not supported with --kernel {CHAIN_KERNEL}: chain arms are enqueued through the device launchers"
        ));
    }
    if o.cold.pool() && !workload::COLD_KERNELS.contains(&o.kernel.as_str()) {
        return Err(format!(
            "{} is only supported with --kernel {} (got --kernel {}): cold-cache timing needs a weight pool",
            o.cold.flag(),
            workload::COLD_KERNELS.join("|"),
            o.kernel
        ));
    }
    if o.rip.is_some()
        && !matches!(
            o.kernel.as_str(),
            "iu4_gemm"
                | "mq6_i8_gemm"
                | "fp8_gemm"
                | "fa_prefill"
                | "qwen4_moe_sym"
                | "iu4_c8_producer_pm"
                | "moe_rotate_i4_pm"
                | "gdn_chunk_prep_pm"
        )
    {
        return Err(format!(
            "--rip is only supported with --kernel iu4_gemm, mq6_i8_gemm, fp8_gemm, fa_prefill, qwen4_moe_sym, iu4_c8_producer_pm, moe_rotate_i4_pm or gdn_chunk_prep_pm: only those generators are exposed to .rip so far (got --kernel {})",
            o.kernel
        ));
    }
    if o.cand_object.is_some() {
        if o.kernel != "qwen4_moe_sym" || !matches!(o.arch, "gfx1100" | "gfx1201") {
            return Err(format!(
                "--cand-object is only supported with --kernel qwen4_moe_sym --arch gfx1100|gfx1201 (got --kernel {} --arch {})",
                o.kernel, o.arch
            ));
        }
        if o.rip.is_some() {
            return Err(
                "--cand-object cannot be combined with --rip (two candidate sources)".into(),
            );
        }
        if o.null_control {
            return Err(
                "--null-control aa cannot be combined with --cand-object (no candidate is emitted)"
                    .into(),
            );
        }
    }
    if o.null_control {
        if o.rip.is_some() {
            return Err(
                "--null-control aa cannot be combined with --rip (no candidate is emitted)".into(),
            );
        }
        // `--base-shipped` only picks WHICH shipped object is the baseline (the
        // incumbent); the A/A then times that baseline against itself.
        if let Some(k) = o
            .knobs
            .0
            .keys()
            .find(|k| k.starts_with("base-") && k.as_str() != "base-shipped")
        {
            return Err(format!(
                "--null-control aa cannot be combined with --{k} (the candidate is the resolved baseline)"
            ));
        }
    }
    if o.max_rounds < o.rounds {
        return Err(format!(
            "--max-rounds {} is below --rounds {}",
            o.max_rounds, o.rounds
        ));
    }
    abba::timing_counts(o.time, o.max_rounds)?;
    Ok(o)
}

fn arch_enum(arch: &str) -> Result<peacemaker_isa::Arch> {
    match arch {
        "gfx1201" => Ok(peacemaker_isa::Arch::Gfx1201),
        "gfx1151" => Ok(peacemaker_isa::Arch::Gfx1151),
        "gfx1100" => Ok(peacemaker_isa::Arch::Gfx1100),
        other => Err(format!("unsupported --arch {other}")),
    }
}

fn build_workload(o: &mut Opts) -> Result<Box<dyn Workload>> {
    let shape = o.shape.clone().unwrap_or_default();
    let mut knobs = std::mem::take(&mut o.knobs);
    let w = match o.kernel.as_str() {
        "iu4_gemm" => iu4::new_with_rip(&shape, &mut knobs, o.rip.as_deref())?,
        "mq6_i8_gemm" => mq6::new(&shape, &mut knobs, o.rip.as_deref())?,
        "fp8_gemm" => fp8::new(&shape, &mut knobs, o.rip.as_deref())?,
        "fa_prefill" => fa::new(&shape, &mut knobs, o.rip.as_deref())?,
        "qwen4_moe_sym" => moe::new_with_cand_object(
            &shape,
            &mut knobs,
            o.rip.as_deref(),
            arch_enum(o.arch)?,
            o.cand_object.as_deref(),
        )?,
        "iu4_c8_producer_pm" => c8::new_with_rip(&shape, &mut knobs, o.rip.as_deref())?,
        "moe_rotate_i4_pm" => rotate::new_with_rip(&shape, &mut knobs, o.rip.as_deref())?,
        "gdn_chunk_prep_pm" => gdn::new_with_rip(&shape, &mut knobs, o.rip.as_deref())?,
        "qkvzaset" => qkvzaset::new(&shape, &mut knobs)?,
        other => {
            return Err(format!(
                "unknown --kernel {other} (iu4_gemm|mq6_i8_gemm|fp8_gemm|fa_prefill|qwen4_moe_sym|qwen4_moe_verify|iu4_c8_producer_pm|moe_rotate_i4_pm|gdn_chunk_prep_pm|qkvzaset)"
            ))
        }
    };
    knobs.finish()?;
    Ok(w)
}

fn build_chain(o: &mut Opts) -> Result<Box<dyn ChainWorkload>> {
    let shape = o.shape.clone().unwrap_or_default();
    let mut knobs = std::mem::take(&mut o.knobs);
    let ctx = ChainCtx {
        arch: o.arch,
        null_control: o.null_control,
    };
    let w = match o.kernel.as_str() {
        CHAIN_KERNEL => moe_verify::new(&shape, &mut knobs, ctx)?,
        other => return Err(format!("--kernel {other} is not a chain kernel")),
    };
    knobs.finish()?;
    Ok(w)
}

// ---------------------------------------------------------------------------
// Object facts: M7, resources, cost lint

fn write_object(dir: &Path, tag: &str, obj: &Object) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(format!("{tag}.{}.co", &sha256_hex(&obj.bytes)[..16]));
    std::fs::write(&path, &obj.bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// Descriptor/metadata resources of every kernel of a lifted object, keyed by
/// symbol. `code_bytes` is the re-encoded instruction stream length.
fn resources(program: &peacemaker_ir::Program, symbols: &[String]) -> Result<Vec<Value>> {
    use peacemaker_ir::inst::Abi;
    let mut rows = Vec::new();
    for symbol in symbols {
        let kernel = program
            .kernels
            .iter()
            .find(|k| &k.symbol.0 == symbol)
            .ok_or_else(|| format!("object has no kernel {symbol}"))?;
        let Abi::Hsa {
            descriptor,
            metadata,
        } = &kernel.abi
        else {
            return Err(format!("{symbol}: not an HSA kernel"));
        };
        let m = &metadata.parsed;
        let code = peacemaker_lift::emit::bytes(kernel, program.target.arch).map_err(err)?;
        rows.push(json!({
            "symbol": symbol,
            "vgprs": m.vgpr_count,
            "sgprs": m.sgpr_count,
            "lds_bytes": descriptor.group_segment_fixed_size,
            "scratch_bytes": descriptor.private_segment_fixed_size,
            "code_bytes": code.len(),
            "code_sha256": sha256_hex(&code),
            "kernarg_bytes": descriptor.kernarg_size,
            "vgpr_spills": m.vgpr_spill_count,
            "sgpr_spills": m.sgpr_spill_count,
            "max_flat_workgroup_size": m.max_flat_workgroup_size,
        }));
    }
    Ok(rows)
}

fn lift(bytes: &[u8]) -> Result<peacemaker_ir::Program> {
    peacemaker_lift::lift_object(
        bytes,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .map(|l| l.program)
    .map_err(err)
}

/// `cost_lint` loop statistics per kernel, `{symbol: [loop..]}`; `{"error": ..}`
/// if the static analysis refuses the object.
fn cost_lint(program: &peacemaker_ir::Program) -> Value {
    match peacemaker_isa::cost_lint::analyze(program, &Default::default()) {
        Ok(report) => {
            let mut out = serde_json::Map::new();
            for k in report.kernels {
                out.insert(
                    k.symbol,
                    serde_json::to_value(k.loops).unwrap_or(Value::Null),
                );
            }
            Value::Object(out)
        }
        Err(e) => json!({"error": e}),
    }
}

/// M7 on every export through the cert cache. Err only when M7 itself rejects;
/// the cache status of each export is kept alongside its receipt.
fn certify(path: &Path, symbols: &[String], arch: &str) -> Value {
    let mut cache = Vec::new();
    let mut receipts = serde_json::Map::new();
    let mut status = "pass";
    let mut error = Value::Null;
    for symbol in symbols {
        match peacemaker_isa::pm_check::m7_status(path, arch, symbol) {
            Ok((receipt, c)) => {
                cache.push(json!({"symbol": symbol, "cache": c.as_str()}));
                receipts.insert(symbol.clone(), receipt);
            }
            Err(e) => {
                status = "fail";
                error = json!({"symbol": symbol, "error": e});
                break;
            }
        }
    }
    let summary = if !cache.is_empty() && cache.iter().all(|c| c["cache"] == "hit") {
        "hit"
    } else if cache.iter().any(|c| c["cache"] == "miss") {
        "miss"
    } else if cache.iter().any(|c| c["cache"] == "disabled") {
        "disabled"
    } else {
        "hit"
    };
    json!({"status": status, "cache": summary, "per_export": cache,
        "receipts": receipts, "error": error})
}

// ---------------------------------------------------------------------------
// Device

struct Loaded {
    _module: hip_bridge::Module,
    func: hip_bridge::Function,
}

fn load(gpu: &Gpu, obj: &Object) -> Result<Loaded> {
    let module = gpu.hip.module_load_data(&obj.bytes).map_err(err)?;
    let func = gpu
        .hip
        .module_get_function(&module, &obj.launch_symbol)
        .map_err(err)?;
    Ok(Loaded {
        _module: module,
        func,
    })
}

fn launch(gpu: &Gpu, w: &dyn Workload, arm: Arm, f: &Loaded) -> Result<()> {
    launch_set(gpu, w, arm, 0, f)
}

/// [`launch`] on weight set `set` of the workload's timing pool (`Workload::launch_set`).
fn launch_set(gpu: &Gpu, w: &dyn Workload, arm: Arm, set: usize, f: &Loaded) -> Result<()> {
    let mut l = w.launch_set(arm, set);
    // SAFETY: the workload packs the kernarg per the kernel's declared ABI
    // and every pointer addresses a live allocation it owns.
    unsafe {
        gpu.hip
            .launch_kernel_blob(&f.func, l.grid, l.block, l.lds, None, &mut l.kernarg)
    }
    .map_err(err)
}

/// Identity of the two arms on every member of the `--flush` weight pool (sets `1..`; set 0 is the
/// correctness set the caller already compared): each arm runs once on each set from its restored
/// state, and the outputs are compared bit for bit.
fn pool_identity(gpu: &Gpu, w: &dyn Workload, bf: &Loaded, cf: &Loaded) -> Result<Value> {
    let sets = w.pool_sets();
    let (mut per_set, mut worst) = (Vec::new(), 0f64);
    for set in 1..sets {
        for arm in [Arm::Baseline, Arm::Candidate] {
            w.reset(gpu, arm)?;
        }
        launch_set(gpu, w, Arm::Baseline, set, bf)?;
        launch_set(gpu, w, Arm::Candidate, set, cf)?;
        gpu.hip.device_synchronize().map_err(err)?;
        let (b, c) = (
            w.output(gpu, Arm::Baseline)?,
            w.output(gpu, Arm::Candidate)?,
        );
        worst = worst.max(w.max_abs_diff(&b, &c)?);
        per_set.push(b == c);
    }
    Ok(json!({
        "sets": sets, "members_checked": per_set.len(),
        "identical": per_set.iter().all(|x| *x), "per_set_identical": per_set,
        "max_abs_diff_vs_baseline": worst,
        "note": "candidate vs baseline on every pool member (set 0 above); no CPU reference for sets >= 1",
    }))
}

/// Null dispatches timed per run (after [`NULL_WARM`] unrecorded ones).
const NULL_LAUNCHES: usize = 400;
const NULL_WARM: usize = 40;

/// `timing.null_dispatch`: the hipEvent time of a NULL kernel (the baseline's descriptor, body
/// `s_endpgm`) launched with the workload's own grid, block, LDS and kernarg, in the same process,
/// the same cold mode (pool rotation, eviction after every launch) and the same event bracketing as
/// the timed arms: the fixed dispatch cost every reading in this record contains. It is REPORTED,
/// never subtracted. `None`: the workload has no null kernel.
fn measure_null(gpu: &Gpu, w: &dyn Workload, flush: Option<&cold::Flush>) -> Option<Value> {
    let obj = match w.null_object() {
        Ok(Some(o)) => o,
        Ok(None) => return None,
        Err(e) => return Some(json!({"error": e})),
    };
    let run = || -> Result<Value> {
        let nf = load(gpu, &obj)?;
        let sets = w.pool_sets();
        let start = gpu.hip.event_create().map_err(err)?;
        let stop = match gpu.hip.event_create() {
            Ok(e) => e,
            Err(e) => {
                let _ = gpu.hip.event_destroy(start);
                return Err(err(e));
            }
        };
        let timed = (|| -> Result<Vec<f64>> {
            cold::evict(&gpu.hip, flush)?;
            let mut ns = Vec::with_capacity(NULL_LAUNCHES);
            for i in 0..NULL_WARM + NULL_LAUNCHES {
                gpu.hip.event_record(&start, None).map_err(err)?;
                launch_set(gpu, w, Arm::Baseline, i % sets, &nf)?;
                gpu.hip.event_record(&stop, None).map_err(err)?;
                gpu.hip.event_synchronize(&stop).map_err(err)?;
                if i >= NULL_WARM {
                    ns.push(f64::from(gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?) * 1e6);
                }
                cold::evict(&gpu.hip, flush)?;
            }
            Ok(ns)
        })();
        let d1 = gpu.hip.event_destroy(start).map_err(err);
        let d2 = gpu.hip.event_destroy(stop).map_err(err);
        let ns = timed?;
        d1?;
        d2?;
        Ok(json!({
            "us": arm_json(&timing_stats(&ns)?),
            "launches": ns.len(),
            "kernel": obj.launch_symbol,
            "object_sha256": sha256_hex(&obj.bytes),
            "body": "s_endpgm",
            "method": "the baseline's emitted descriptor (VGPRs, LDS, kernarg) with the body replaced by one s_endpgm, \
                launched with the workload's grid/block/LDS/kernarg, hipEvent-bracketed like a timed arm, in this process and cold mode",
            "subtracted": false,
            "note": "every timing.*_us in this record contains this fixed cost; it is reported, not removed",
        }))
    };
    Some(run().unwrap_or_else(|e| json!({"error": e})))
}

fn device_json(gpu: &Gpu) -> Value {
    let env = |k: &str| std::env::var(k).ok();
    let cards: Vec<String> = env("FABRIC_CARDS")
        .map(|s| s.split(',').map(str::to_owned).collect())
        .unwrap_or_default();
    json!({
        "arch": gpu.arch, "device_id": gpu.device_id,
        "uuid": env("ROCR_VISIBLE_DEVICES").or_else(|| env("HIPFIRE_DEVICES")),
        "fabric_cards": cards,
        "fabric_job": env("FABRIC_JOB"),
    })
}

// ---------------------------------------------------------------------------
// Timing record

fn us(v: f64) -> f64 {
    v / 1e3
}

fn arm_json(s: &TimingStats) -> Value {
    json!({"median": us(s.median), "min": us(s.min), "max": us(s.max),
        "p90": us(s.p90), "mean": us(s.mean)})
}

const RULE: &str = "separable = exact two-sided sign test over the per-round paired ratios \
    (each ABBA round: mean of the candidate's two block medians over the mean of the baseline's \
    two, same round) has p <= 1e-3, ties counting against; direction = majority sign; \
    material = separable AND |block_ratio - 1| >= min_effect (the sign test also detects real \
    but immaterial placement/order effects). \
    separable_block_disjoint = every candidate block median is below every baseline block \
    median, or every one above. separable_full_range = the pm_decode_twins rule: |median delta| \
    > larger full spread AND ranges disjoint";

/// Pure timing record from the two arms' samples (nanoseconds, in schedule
/// order, whole ABBA rounds): the paired-round sign test is `separable`; the
/// block-disjoint rule and the twins' full-range rule are kept as
/// `separable_block_disjoint` and `separable_full_range`. `ratio` is the
/// candidate/baseline ratio of the full-sample medians.
fn timing_record(
    s: &Sampled,
    n: usize,
    rounds: usize,
    stop: abba::ExtendStop,
    sampling_s: f64,
    min_effect: f64,
) -> Result<Value> {
    let a = timing_stats(&s.base_ns)?;
    let b = timing_stats(&s.cand_ns)?;
    let (ratio, spread, delta, full_range) = timing_relation(&a, &b);
    let r = abba::block_rule(&s.base_ns, &s.cand_ns, n)?;
    let rounds_run = s.base_ns.len() / (2 * n);
    let ext = |v: &[f64]| {
        (
            v.iter().copied().fold(f64::INFINITY, f64::min),
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let ratios = abba::round_ratios(&s.base_ns, &s.cand_ns, n)?;
    let block_us = |v: &[f64]| v.iter().map(|x| us(*x)).collect::<Vec<_>>();
    let noise = json!({
        "round_ratios": ratios,
        "round_ratio": abba::dispersion(&ratios)?.json(),
        "baseline_block_median": abba::dispersion(&block_us(&r.baseline_block_medians))?.json(),
        "candidate_block_median": abba::dispersion(&block_us(&r.candidate_block_medians))?.json(),
        "baseline_drift_per_block": abba::drift_per_round(&r.baseline_block_medians),
        "candidate_drift_per_block": abba::drift_per_round(&r.candidate_block_medians),
    });
    let (bmin, bmax) = ext(&r.baseline_block_medians);
    let (cmin, cmax) = ext(&r.candidate_block_medians);
    Ok(json!({
        "n": n, "rounds": rounds, "rounds_run": rounds_run, "extended": rounds_run > rounds,
        "stop": stop.as_str(), "sampling_s": sampling_s, "calls_per_arm": a.count,
        "baseline_us": arm_json(&a), "candidate_us": arm_json(&b),
        "ratio": ratio, "separable": r.separable, "direction": r.direction,
        "material": r.separable && (r.block_ratio - 1.0).abs() >= min_effect,
        "min_effect": min_effect,
        "sign_p": r.sign_p, "p_max": abba::PAIRED_P_MAX,
        "rounds_faster": r.rounds_faster, "rounds_slower": r.rounds_slower,
        "rounds_tied": r.rounds_tied,
        "block_ratio": r.block_ratio,
        "round_ratio_min": r.round_ratio_min, "round_ratio_max": r.round_ratio_max,
        "baseline_block_median_us": {"min": us(bmin), "max": us(bmax)},
        "candidate_block_median_us": {"min": us(cmin), "max": us(cmax)},
        "separable_block_disjoint": r.block_disjoint,
        "separable_full_range": full_range,
        "observed_spread_us": us(spread), "median_delta_us": us(delta),
        "unit": "us per launch (hipEvent, default stream)",
        "rule": RULE,
        "noise": noise,
    }))
}

/// `timing.vs_noise_floor` from a `pm-census noise` document: the A/A group
/// with the same kernel, shape and baseline sha256 gives the floor (the
/// largest, if several match); the effect is `|block_ratio - 1|`.
fn vs_noise_floor(
    doc: &Value,
    kernel: &str,
    shape: &Value,
    baseline_sha: &str,
    block_ratio: f64,
) -> Value {
    let empty = Vec::new();
    let matching: Vec<&Value> = doc["groups"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .filter(|g| {
            g["aa"] == true
                && g["kernel"] == kernel
                && g["shape"] == *shape
                && g["baseline_sha256"] == baseline_sha
        })
        .collect();
    if matching.is_empty() {
        return json!({"status": "no matching A/A group"});
    }
    let best = matching
        .iter()
        .filter_map(|g| g["floor"].as_f64().map(|f| (f, g["id"].clone())))
        .max_by(|a, b| a.0.total_cmp(&b.0));
    match best {
        Some((floor, id)) => {
            let effect = (block_ratio - 1.0).abs();
            json!({"floor": floor, "effect": effect, "exceeds": effect > floor, "group": id})
        }
        None => json!({"status": "matching A/A group has no floor (insufficient runs)",
            "group": matching[0]["id"].clone()}),
    }
}

/// An emulator refusal carries the faulting instruction's full register dump
/// after ` | `; keep the reason, drop the dump.
fn tidy_emu(v: Value) -> Value {
    match v {
        Value::String(s) if s.starts_with("refused (") && s.contains(" | ") => {
            let reason = s.split(" | ").next().unwrap_or(&s);
            Value::String(format!("{reason})"))
        }
        other => other,
    }
}

#[derive(Default)]
struct Sampled {
    base_ns: Vec<f64>,
    cand_ns: Vec<f64>,
}

/// What the ABBA sampler drives: one timing arm = `reset` (outside the event
/// interval) then `launch` (inside it). A single-object workload launches one
/// kernel per slot; a chain workload enqueues its whole chain.
trait ArmExec {
    fn hip(&self) -> &hip_bridge::HipRuntime;
    /// Called before the first launch of ABBA round `round` (0-based, counted
    /// across extension batches). Single-object workloads ignore it.
    fn begin_round(&mut self, _round: usize) {}
    fn reset(&mut self, arm: Arm) -> Result<()>;
    fn launch(&mut self, arm: Arm) -> Result<()>;
    /// Evict the caches (`--flush`) the way the other kernels of a decode step do. Enqueued before
    /// the first launch of a batch and after every timed launch, so it is never inside an event
    /// interval. Default: nothing.
    fn evict(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Single-object workload: the workload plus the two loaded modules.
struct SingleExec<'a> {
    gpu: &'a Gpu,
    w: &'a dyn Workload,
    base: &'a Loaded,
    cand: &'a Loaded,
    /// `--flush`: the eviction buffers.
    flush: Option<&'a cold::Flush>,
    /// Timed launches issued so far: dispatch `i` runs on weight set `i % w.pool_sets()`.
    slot: usize,
}

impl ArmExec for SingleExec<'_> {
    fn hip(&self) -> &hip_bridge::HipRuntime {
        &self.gpu.hip
    }
    fn reset(&mut self, arm: Arm) -> Result<()> {
        self.w.reset(self.gpu, arm)
    }
    fn launch(&mut self, arm: Arm) -> Result<()> {
        let f = match arm {
            Arm::Baseline => self.base,
            Arm::Candidate => self.cand,
        };
        let set = self.slot % self.w.pool_sets();
        self.slot += 1;
        launch_set(self.gpu, self.w, arm, set, f)
    }
    fn evict(&mut self) -> Result<()> {
        cold::evict(&self.gpu.hip, self.flush)
    }
}

/// Complete-chain workload: `encode_chain` is the launch.
struct ChainExec<'a> {
    gpu: &'a mut Gpu,
    w: &'a mut dyn ChainWorkload,
}

impl ArmExec for ChainExec<'_> {
    fn hip(&self) -> &hip_bridge::HipRuntime {
        &self.gpu.hip
    }
    /// Crossover: even rounds bind the arms to their own buffer sets, odd
    /// rounds swap them (`ChainWorkload::set_buffer_swap`).
    fn begin_round(&mut self, round: usize) {
        self.w.set_buffer_swap(round % 2 == 1);
    }
    fn reset(&mut self, arm: Arm) -> Result<()> {
        self.w.reset(&mut *self.gpu, arm)
    }
    fn launch(&mut self, arm: Arm) -> Result<()> {
        self.w.encode_chain(&mut *self.gpu, arm)
    }
}

/// Append `rounds` whole ABBA rounds of `n` launches per block to `s`.
fn sample_rounds(ex: &mut dyn ArmExec, n: usize, rounds: usize, s: &mut Sampled) -> Result<()> {
    let done = s.base_ns.len() / (2 * n);
    let (calls, _) = abba::timing_counts(n, rounds)?;
    s.base_ns.reserve(calls);
    s.cand_ns.reserve(calls);
    let start = ex.hip().event_create().map_err(err)?;
    let stop = match ex.hip().event_create() {
        Ok(e) => e,
        Err(e) => {
            let _ = ex.hip().event_destroy(start);
            return Err(err(e));
        }
    };
    let measured = (|| -> Result<()> {
        ex.evict()?;
        for (i, which) in timing_schedule(n, rounds)?.enumerate() {
            if i % (4 * n) == 0 {
                ex.begin_round(done + i / (4 * n));
            }
            let arm = match which {
                TimingArm::Incumbent => Arm::Baseline,
                TimingArm::Candidate => Arm::Candidate,
            };
            ex.reset(arm)?;
            ex.hip().event_record(&start, None).map_err(err)?;
            ex.launch(arm)?;
            ex.hip().event_record(&stop, None).map_err(err)?;
            ex.hip().event_synchronize(&stop).map_err(err)?;
            let ns = f64::from(ex.hip().event_elapsed_ms(&start, &stop).map_err(err)?) * 1e6;
            match which {
                TimingArm::Incumbent => s.base_ns.push(ns),
                TimingArm::Candidate => s.cand_ns.push(ns),
            }
            ex.evict()?;
        }
        Ok(())
    })();
    let d1 = ex.hip().event_destroy(start).map_err(err);
    let d2 = ex.hip().event_destroy(stop).map_err(err);
    measured?;
    d1?;
    d2
}

/// Sampling plan: `rounds` initial ABBA rounds of `n` launches per block,
/// then doubling until the block-median rule separates, `max_rounds` or
/// `budget_s` (seconds of `clock`) is reached. `run(extra, s)` appends `extra`
/// whole rounds. Returns the samples, why it stopped and the elapsed time.
fn drive_sampling(
    n: usize,
    rounds: usize,
    max_rounds: usize,
    budget_s: f64,
    clock: &mut dyn FnMut() -> f64,
    run: &mut dyn FnMut(usize, &mut Sampled) -> Result<()>,
) -> Result<(Sampled, abba::ExtendStop, f64)> {
    let t0 = clock();
    let mut s = Sampled::default();
    let mut extra = rounds;
    loop {
        run(extra, &mut s)?;
        let done = s.base_ns.len() / (2 * n);
        let separable = abba::block_rule(&s.base_ns, &s.cand_ns, n)?.separable;
        let elapsed = clock() - t0;
        match abba::extend_rounds(done, max_rounds, elapsed, budget_s, separable) {
            Ok(more) => extra = more,
            Err(stop) => return Ok((s, stop, elapsed)),
        }
    }
}

fn sample(ex: &mut dyn ArmExec, o: &Opts) -> Result<(Sampled, abba::ExtendStop, f64)> {
    let start = std::time::Instant::now();
    drive_sampling(
        o.time,
        o.rounds,
        o.max_rounds,
        o.time_budget_s,
        &mut || start.elapsed().as_secs_f64(),
        &mut |extra, s| sample_rounds(&mut *ex, o.time, extra, s),
    )
}

// ---------------------------------------------------------------------------
// Driver

/// Exit classification of a run.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Ok,
    CorrectnessFailed,
    CandidateUnusable,
}

fn object_json(obj: &Object, resources: Value, extra: serde_json::Map<String, Value>) -> Value {
    let mut v = json!({
        "co_sha256": sha256_hex(&obj.bytes), "co_bytes": obj.bytes.len(),
        "launch_symbol": obj.launch_symbol, "symbols": obj.exports,
        "resources": resources,
    });
    let o = v.as_object_mut().unwrap();
    if let Some(d) = obj.desc.as_object() {
        o.extend(d.clone());
    }
    o.extend(extra);
    v
}

fn fail(mut rec: Value, stage: &str, e: String) -> (Value, Outcome) {
    rec["error"] = json!({"stage": stage, "message": e});
    (rec, Outcome::CandidateUnusable)
}

fn out_dir_of(o: &Opts) -> PathBuf {
    o.out
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join(format!("pm_loop-{}", std::process::id())))
}

/// The sampled record of a run that reached timing: `timing_record` plus the
/// noise-floor comparison keyed by `baseline_sha`.
fn timed_record(
    o: &Opts,
    shape: &Value,
    baseline_sha: &str,
    s: &Sampled,
    stop: abba::ExtendStop,
    secs: f64,
) -> Result<Value> {
    let mut t = timing_record(s, o.time, o.rounds, stop, secs, o.min_effect)?;
    if let Some(doc) = &o.noise_floor_doc {
        t["vs_noise_floor"] = vs_noise_floor(
            doc,
            &o.kernel,
            shape,
            baseline_sha,
            t["block_ratio"].as_f64().unwrap_or(f64::NAN),
        );
    }
    t["post_outputs_unchanged"] = json!(true);
    t["flush"] = json!(o.cold == cold::ColdMode::Flush);
    t["cold_mode"] = json!(o.cold.name());
    if o.cold == cold::ColdMode::Flush {
        t["flush_method"] = json!(cold::flush_method());
    }
    Ok(t)
}

/// `timing.crossover` of a chain record: the per-round ratios split by the
/// buffer binding of their round (even rounds: each arm on its own set; odd:
/// swapped). With a pure buffer-placement effect `p` the two medians sit at
/// about `1 - p` and `1 + p`; `crossover_ratio` (their geometric mean) cancels
/// it and `placement_ratio` (sqrt of their quotient) measures it.
fn crossover_json(t: &Value) -> Result<Value> {
    let ratios: Vec<f64> = t["noise"]["round_ratios"]
        .as_array()
        .ok_or("timing.noise.round_ratios missing")?
        .iter()
        .map(|v| v.as_f64().ok_or("non-numeric round ratio"))
        .collect::<std::result::Result<_, _>>()?;
    let (own, swapped): (Vec<(usize, f64)>, Vec<(usize, f64)>) = ratios
        .iter()
        .copied()
        .enumerate()
        .partition(|(i, _)| i % 2 == 0);
    let strip = |v: Vec<(usize, f64)>| v.into_iter().map(|(_, r)| r).collect::<Vec<f64>>();
    let (own, swapped) = (strip(own), strip(swapped));
    let mut rec = json!({
        "kind": "buffer sets alternate every ABBA round (even rounds: own sets, odd rounds: swapped)",
        "rounds_own": own.len(),
        "rounds_swapped": swapped.len(),
    });
    if own.is_empty() || swapped.is_empty() {
        return Ok(rec);
    }
    let (a, b) = (abba::dispersion(&own)?, abba::dispersion(&swapped)?);
    rec["round_ratio_own"] = a.json();
    rec["round_ratio_swapped"] = b.json();
    rec["crossover_ratio"] = json!((a.median * b.median).sqrt());
    rec["placement_ratio"] = json!((a.median / b.median).sqrt());
    Ok(rec)
}

fn run(o: &mut Opts) -> (Value, Outcome) {
    let mut rec = json!({"schema": SCHEMA, "kernel": o.kernel, "arch": o.arch});
    if o.kernel == moe_route::KERNEL {
        return moe_route::run(o);
    }
    if o.kernel == CHAIN_KERNEL {
        return match build_chain(o) {
            Ok(w) => run_chain(o, w),
            Err(e) => fail(rec, "args", e),
        };
    }
    let mut w = match build_workload(o) {
        Ok(w) => w,
        Err(e) => return fail(rec, "args", e),
    };
    if o.cold.pool() {
        if let Err(e) = w.set_cold(o.cold) {
            return fail(rec, "args", e);
        }
    }
    let arch = w.arch();
    rec["arch"] = json!(arch);
    rec["shape"] = w.shape_json();
    for (key, value) in w.record_extra() {
        rec[key] = value;
    }
    rec["candidate"] = json!({"knobs": w.candidate_knobs()});
    let contract = w.numeric_contract();
    if !contract.is_null() {
        rec["numeric_contract"] = contract;
    }
    let out_dir = out_dir_of(o);

    // 1. emit + link. `--null-control aa`: no candidate is emitted; it is a
    // clone of the baseline object (loaded as a separate module below).
    let aa = o.null_control;
    let emitted = if aa {
        None
    } else {
        match w.candidate() {
            Ok(c) => Some(c),
            Err(e) => {
                let stage = if o.rip.is_some() {
                    "rip"
                } else if o.cand_object.is_some() {
                    "cand_object"
                } else {
                    "emit"
                };
                return fail(rec, stage, e);
            }
        }
    };
    let base = match w.baseline() {
        Ok(b) => b,
        Err(e) => return fail(rec, "baseline", e),
    };
    let cand = emitted.unwrap_or_else(|| base.clone());

    // Pre-card emulator gate (`--emu`, offline or not): no candidate reaches a device unless the candidate and the
    // shipped incumbent both run their whole grid on the emulator (guarded buffers, definedness-tracked LDS) and
    // agree bit-for-bit. The record keeps `emu_gate` even when M7 or the gate itself ends the run.
    let mut emu_gate_failed: Option<String> = None;
    if o.emu {
        match w.emu_gate(&cand, &base) {
            Some(g) => {
                if g["status"] != "pass" {
                    emu_gate_failed = Some(
                        g["reason"]
                            .as_str()
                            .unwrap_or("emulator gate failed")
                            .to_owned(),
                    );
                }
                rec["emu_gate"] = g;
            }
            None => {
                rec["emu_gate"] = json!({
                    "status": "unavailable",
                    "reason": format!("no emulator gate for --kernel {}", o.kernel),
                });
            }
        }
    }

    // 2. certify + resources
    if !aa {
        let cand_path = match write_object(&out_dir, "candidate", &cand) {
            Ok(p) => p,
            Err(e) => return fail(rec, "write", e),
        };
        let m7 = certify(&cand_path, &cand.exports, arch);
        let (cand_res, cand_lint) = match lift(&cand.bytes).and_then(|p| {
            let r = resources(&p, &cand.exports)?;
            Ok((r, cost_lint(&p)))
        }) {
            Ok((r, l)) => (json!(r), l),
            Err(e) => (json!({"error": e}), Value::Null),
        };
        let mut cextra = serde_json::Map::new();
        cextra.insert("m7".into(), {
            let mut v = m7.clone();
            v.as_object_mut().map(|o| o.remove("receipts"));
            v
        });
        cextra.insert("m7_receipts".into(), m7["receipts"].clone());
        cextra.insert("cost_lint".into(), cand_lint);
        cextra.insert("object_path".into(), json!(cand_path.display().to_string()));
        cextra.insert("knobs".into(), w.candidate_knobs());
        rec["candidate"] = object_json(&cand, cand_res, cextra);
        if m7["status"] != "pass" {
            let msg = m7["error"].to_string();
            return fail(rec, "m7", msg);
        }
    }
    let (base_res, base_lint) = match lift(&base.bytes).and_then(|p| {
        let r = resources(&p, std::slice::from_ref(&base.launch_symbol))?;
        Ok((r, cost_lint(&p)))
    }) {
        Ok((r, l)) => (json!(r), l),
        Err(e) => (json!({"error": e}), Value::Null),
    };
    let base_path = write_object(&out_dir, "baseline", &base);
    let base_m7 = match &base_path {
        Ok(p) => certify(p, std::slice::from_ref(&base.launch_symbol), arch),
        Err(e) => json!({"status": "skipped", "error": e}),
    };
    let base_receipts = base_m7["receipts"].clone();
    let base_res_for_cand = base_res.clone();
    let mut bextra = serde_json::Map::new();
    bextra.insert("source".into(), base.source.clone());
    bextra.insert("m7".into(), {
        let mut v = base_m7;
        v.as_object_mut().map(|o| o.remove("receipts"));
        v
    });
    // The shipped bundle carries every epilogue symbol: keep the launched one.
    bextra.insert(
        "cost_lint".into(),
        match base_lint.get(&base.launch_symbol) {
            Some(loops) => json!({ base.launch_symbol.clone(): loops }),
            None => base_lint,
        },
    );
    rec["baseline"] = object_json(&base, base_res, bextra);
    if aa {
        // The candidate is the baseline object: its certification IS the
        // baseline's (same bytes), reported as such.
        let mut m7 = rec["baseline"]["m7"].clone();
        if let Some(m) = m7.as_object_mut() {
            m.insert("reused_from".into(), json!("baseline"));
        }
        let mut cextra = serde_json::Map::new();
        cextra.insert("m7".into(), m7.clone());
        cextra.insert("m7_receipts".into(), base_receipts);
        cextra.insert("cost_lint".into(), rec["baseline"]["cost_lint"].clone());
        cextra.insert(
            "object_path".into(),
            json!(base_path.as_ref().ok().map(|p| p.display().to_string())),
        );
        cextra.insert("knobs".into(), w.candidate_knobs());
        rec["candidate"] = object_json(&cand, base_res_for_cand, cextra);
        rec["null_control"] = json!({
            "kind": "aa",
            "identical_objects": cand.bytes == base.bytes
                && cand.launch_symbol == base.launch_symbol
                && cand.exports == base.exports,
            "separate_modules": true,
            "baseline_sha256": sha256_hex(&base.bytes),
            "candidate_sha256": sha256_hex(&cand.bytes),
            "certification": "candidate certification reuses the baseline's (same bytes)",
        });
        if m7["status"] == "fail" {
            let msg = m7["error"].to_string();
            return fail(rec, "m7", msg);
        }
    }
    if let Some(reason) = emu_gate_failed {
        return fail(rec, "emu_gate", reason);
    }
    if o.offline {
        // Offline legality screen (R4 pruning): emit + link + M7 + baseline
        // resolution only; no device is opened and nothing is timed.
        rec["offline"] = json!(true);
        return (rec, Outcome::Ok);
    }

    // 3. correctness on the card
    let gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => return fail(rec, "gpu_init", err(e)),
    };
    rec["device"] = device_json(&gpu);
    if gpu.arch != arch {
        let msg = format!("device is {}, --arch {arch}", gpu.arch);
        return fail(rec, "device", msg);
    }
    let staged = (|| -> Result<(Loaded, Loaded)> {
        w.setup(&gpu)?;
        Ok((load(&gpu, &base)?, load(&gpu, &cand)?))
    })();
    let (bf, cf) = match staged {
        Ok(v) => v,
        Err(e) => return fail(rec, "setup", e),
    };
    let warm = (|| -> Result<(Vec<u8>, Vec<u8>)> {
        for arm in [Arm::Baseline, Arm::Candidate] {
            w.reset(&gpu, arm)?;
        }
        launch(&gpu, &*w, Arm::Baseline, &bf)?;
        launch(&gpu, &*w, Arm::Candidate, &cf)?;
        gpu.hip.device_synchronize().map_err(err)?;
        Ok((
            w.output(&gpu, Arm::Baseline)?,
            w.output(&gpu, Arm::Candidate)?,
        ))
    })();
    let (base_out, cand_out) = match warm {
        Ok(v) => v,
        Err(e) => return fail(rec, "launch", e),
    };
    let reference = w.check_arm(Arm::Candidate, &cand_out);
    let base_report = w.check_arm(Arm::Baseline, &base_out);
    let diff = match w.max_abs_diff(&base_out, &cand_out) {
        Ok(d) => d,
        Err(e) => return fail(rec, "compare", e),
    };
    let identical = base_out == cand_out;
    // Without a CPU reference (e.g. an epilogue whose device intrinsics are not
    // reproducible) bit identity with the shipped baseline is the only
    // admissible evidence: `pass` is then `identical_to_baseline`.
    // A declared NUMERIC candidate with no CPU reference cannot be identical;
    // it passes when finite and NaN-pattern-equal to the baseline (`diff` is
    // `inf` otherwise), and the KLD/state gate decides promotion.
    let numeric = w.numeric_contract()["relation"] == "numeric";
    let pass = match &reference {
        Ok(r) => r.pass && (identical || !w.requires_baseline_identity()),
        Err(_) if numeric => diff.is_finite(),
        Err(_) => identical,
    };
    let emu_field = if !o.emu {
        json!("not requested (--emu)")
    } else if rec["emu_gate"]["status"] == "pass" {
        json!("see emu_gate")
    } else {
        tidy_emu(w.emu())
    };
    rec["correctness"] = json!({
        "reference": reference.as_ref().map(|r| r.reference.as_str()).unwrap_or("none"),
        "reference_error": reference.as_ref().err(),
        "max_abs_err": reference.as_ref().ok().map(|r| r.max_abs_err),
        "tolerance": reference.as_ref().ok().map(|r| r.tolerance),
        "pass": pass,
        "pass_basis": if reference.is_ok() && w.requires_baseline_identity() { "cpu_reference and identical_to_baseline" } else if reference.is_ok() { "cpu_reference" } else if numeric { "numeric_contract (no CPU reference; candidate finite and NaN-pattern-equal to baseline)" } else { "identical_to_baseline (no CPU reference)" },
        "baseline_max_abs_err": base_report.as_ref().ok().map(|r| r.max_abs_err),
        "baseline_pass": base_report.as_ref().ok().map(|r| r.pass),
        "baseline_reference_error": base_report.as_ref().err(),
        "identical_to_baseline": identical,
        "max_abs_diff_vs_baseline": diff,
        "output_bytes": cand_out.len(),
        "emu": emu_field,
    });
    // `--flush`: the arms must also agree on every member of the weight pool.
    let mut pass = pass;
    if w.pool_sets() > 1 {
        match pool_identity(&gpu, &*w, &bf, &cf) {
            Ok(p) => {
                // Identical on the correctness set: identical on every pool member too.
                if identical && p["identical"] != json!(true) {
                    pass = false;
                    rec["correctness"]["pass"] = json!(false);
                }
                rec["correctness"]["pool"] = p;
            }
            Err(e) => return fail(rec, "launch", e),
        }
    }
    if !pass {
        rec["timing"] = json!({"status": "refused: candidate failed correctness"});
        return (rec, Outcome::CorrectnessFailed);
    }

    // 4. ABBA timing
    let l2 = cold::l2_bytes(&gpu.hip, gpu.device_id);
    let flush = match cold::flush_bytes(o.cold, l2.0) {
        Some(bytes) => match cold::Flush::alloc_bytes(&gpu.hip, bytes) {
            Ok(f) => Some(f),
            Err(e) => return fail(rec, "setup", e),
        },
        None => None,
    };
    let sampled = sample(
        &mut SingleExec {
            gpu: &gpu,
            w: &*w,
            base: &bf,
            cand: &cf,
            flush: flush.as_ref(),
            slot: 0,
        },
        o,
    );
    let post = (|| -> Result<bool> {
        gpu.hip.device_synchronize().map_err(err)?;
        if w.pool_sets() > 1 {
            // The last timed launch of each arm ran on whichever pool set it fell on: re-run both
            // arms on set 0 and compare with the warm outputs.
            for (arm, f) in [(Arm::Baseline, &bf), (Arm::Candidate, &cf)] {
                w.reset(&gpu, arm)?;
                launch(&gpu, &*w, arm, f)?;
            }
            gpu.hip.device_synchronize().map_err(err)?;
        }
        // The last timed launch of each arm must still equal its warm output.
        Ok(w.output(&gpu, Arm::Baseline)? == base_out
            && w.output(&gpu, Arm::Candidate)? == cand_out)
    })();
    // The fixed dispatch cost of this very launch shape, measured in the same process and mode.
    let null_dispatch = measure_null(&gpu, &*w, flush.as_ref());
    if let Some(f) = flush {
        f.free(&gpu.hip);
    }
    match (sampled, post) {
        (Ok((s, stop, secs)), Ok(true)) => {
            let baseline_sha = rec["baseline"]["co_sha256"]
                .as_str()
                .unwrap_or("")
                .to_owned();
            match timed_record(o, &rec["shape"], &baseline_sha, &s, stop, secs) {
                Ok(t) => rec["timing"] = t,
                Err(e) => return fail(rec, "timing", e),
            }
            if let Some(bytes) = cold::flush_bytes(o.cold, l2.0) {
                if o.cold == cold::ColdMode::FlushL2 {
                    rec["timing"]["flush_method"] = json!(cold::flush_method_of(bytes));
                }
                rec["timing"]["flush_bytes_per_buffer"] = json!(bytes);
                rec["timing"]["l2_bytes"] = json!(l2.0);
                rec["timing"]["l2_source"] = json!(l2.1);
            }
            if let Some(n) = null_dispatch {
                rec["timing"]["null_dispatch"] = n;
            }
        }
        (Ok(_), Ok(false)) => {
            rec["timing"] =
                json!({"status": "post-sampling outputs changed; no performance claim"});
            return (rec, Outcome::CorrectnessFailed);
        }
        (Err(e), _) | (_, Err(e)) => return fail(rec, "timing", e),
    }
    rec["device_after"] = device_json(&gpu);
    (rec, Outcome::Ok)
}

// ---------------------------------------------------------------------------
// Complete chains (`--kernel qwen4_moe_verify`)

/// Write and M7-certify every export of every PM object of one chain arm.
/// Returns one `object_json` row per object and the first failure as
/// `(stage, message)` (`write` or `m7`); the caller decides whether it gates.
fn certify_chain_objects(
    dir: &Path,
    tag: &str,
    objs: &[Object],
    arch: &str,
) -> (Vec<Value>, Option<(&'static str, String)>) {
    let mut rows = Vec::new();
    let mut failure: Option<(&'static str, String)> = None;
    for obj in objs {
        let (path, m7) = match write_object(dir, tag, obj) {
            Ok(p) => {
                let m7 = certify(&p, &obj.exports, arch);
                (Some(p), m7)
            }
            Err(e) => {
                failure.get_or_insert(("write", e.clone()));
                (None, json!({"status": "skipped", "error": e}))
            }
        };
        if m7["status"] == "fail" {
            failure.get_or_insert(("m7", m7["error"].to_string()));
        }
        let (res, lint) = match lift(&obj.bytes).and_then(|p| {
            let r = resources(&p, &obj.exports)?;
            Ok((r, cost_lint(&p)))
        }) {
            Ok((r, l)) => (json!(r), l),
            Err(e) => (json!({"error": e}), Value::Null),
        };
        let mut extra = serde_json::Map::new();
        extra.insert("m7".into(), {
            let mut v = m7.clone();
            v.as_object_mut().map(|o| o.remove("receipts"));
            v
        });
        extra.insert("m7_receipts".into(), m7["receipts"].clone());
        extra.insert("cost_lint".into(), lint);
        extra.insert(
            "object_path".into(),
            json!(path.as_ref().map(|p| p.display().to_string())),
        );
        rows.push(object_json(obj, res, extra));
    }
    (rows, failure)
}

/// sha256 of the canonical JSON of one arm of `chain_json()`: the identity
/// `pm-census noise` groups chains by.
fn chain_sha(chain: &Value, arm: &str) -> String {
    sha256_hex(&serde_json::to_vec(&chain[arm]).unwrap_or_default())
}

/// One checked pass: per arm reset, enqueue the whole chain, synchronize and
/// download the outputs. Returns `(baseline, candidate)` outputs.
fn chain_pass(gpu: &mut Gpu, w: &mut dyn ChainWorkload) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut one = |arm: Arm| -> Result<Vec<u8>> {
        w.reset(gpu, arm)?;
        w.encode_chain(gpu, arm)?;
        gpu.hip.device_synchronize().map_err(err)?;
        w.outputs(gpu, arm)
    };
    let base = one(Arm::Baseline)?;
    let cand = one(Arm::Candidate)?;
    Ok((base, cand))
}

fn run_chain(o: &Opts, mut w: Box<dyn ChainWorkload>) -> (Value, Outcome) {
    let arch = w.arch();
    let aa = o.null_control;
    let mut rec = json!({
        "schema": SCHEMA, "kernel": w.kernel(), "arch": arch, "mode": w.mode(),
        "shape": w.shape_json(), "numeric_contract": w.numeric_contract(),
        "candidate": {"knobs": w.candidate_knobs()},
    });
    let out_dir = out_dir_of(o);

    // 1. certify the PM objects of each arm (no GPU needed). Baseline first:
    // `--null-control aa` reuses it for the candidate.
    let base_objs = match w.pm_objects(Arm::Baseline) {
        Ok(v) => v,
        Err(e) => return fail(rec, "baseline", e),
    };
    let (base_rows, base_fail) = certify_chain_objects(&out_dir, "baseline", &base_objs, arch);
    rec["baseline"] = json!({"pm_objects": base_rows});
    let (cand_rows, cand_fail) = if aa {
        let rows: Vec<Value> = base_rows
            .iter()
            .cloned()
            .map(|mut r| {
                if let Some(m) = r.get_mut("m7").and_then(Value::as_object_mut) {
                    m.insert("reused_from".into(), json!("baseline"));
                }
                r
            })
            .collect();
        (rows, base_fail.filter(|(stage, _)| *stage == "m7"))
    } else {
        let objs = match w.pm_objects(Arm::Candidate) {
            Ok(v) => v,
            Err(e) => return fail(rec, "emit", e),
        };
        certify_chain_objects(&out_dir, "candidate", &objs, arch)
    };
    rec["candidate"]["pm_objects"] = json!(cand_rows);
    if let Some((stage, msg)) = cand_fail {
        return fail(rec, stage, msg);
    }

    // 2. the card
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => return fail(rec, "gpu_init", err(e)),
    };
    rec["device"] = device_json(&gpu);
    if gpu.arch != arch {
        let msg = format!("device is {}, --arch {arch}", gpu.arch);
        return fail(rec, "device", msg);
    }
    if let Err(e) = w.setup(&mut gpu) {
        return fail(rec, "setup", e);
    }
    let chain = w.chain_json();
    let (base_sha, cand_sha) = (
        chain_sha(&chain, "baseline"),
        chain_sha(&chain, "candidate"),
    );
    rec["chain"] = chain;
    rec["baseline"]["sha256"] = json!(base_sha);
    rec["candidate"]["sha256"] = json!(cand_sha);
    if aa {
        rec["null_control"] = json!({
            "kind": "aa",
            "identical_chains": base_sha == cand_sha,
            "separate_modules": false,
            "baseline_sha256": base_sha,
            "candidate_sha256": cand_sha,
            "note": "candidate arm runs the baseline chain through the same rdna-compute launchers and loaded modules into its own buffers",
        });
    }

    // 3. correctness: warm pass on the arms' own buffer sets, then a repeat
    // pass on the swapped sets (the crossover binding timing alternates) that
    // must reproduce it bit for bit.
    w.set_buffer_swap(false);
    let (base_out, cand_out) = match chain_pass(&mut gpu, &mut *w) {
        Ok(v) => v,
        Err(e) => return fail(rec, "launch", e),
    };
    w.set_buffer_swap(true);
    let (base_rep, cand_rep) = match chain_pass(&mut gpu, &mut *w) {
        Ok(v) => v,
        Err(e) => return fail(rec, "launch", e),
    };
    let reference = w.reference(Arm::Candidate, &cand_out);
    let base_report = w.reference(Arm::Baseline, &base_out);
    let diff = match w.max_abs_diff(&base_out, &cand_out) {
        Ok(d) => d,
        Err(e) => return fail(rec, "compare", e),
    };
    let identical = base_out == cand_out;
    let (base_repeat, cand_repeat) = (base_out == base_rep, cand_out == cand_rep);
    let pass = reference.as_ref().is_ok_and(|r| r.pass)
        && base_report.as_ref().is_ok_and(|r| r.pass)
        && base_repeat
        && cand_repeat
        && (identical || !w.requires_baseline_identity());
    rec["correctness"] = json!({
        "reference": reference.as_ref().map(|r| r.reference.as_str()).unwrap_or("none"),
        "reference_error": reference.as_ref().err(),
        "max_abs_err": reference.as_ref().ok().map(|r| r.max_abs_err),
        "tolerance": reference.as_ref().ok().map(|r| r.tolerance),
        "pass": pass,
        "pass_basis": "candidate and baseline each pass their own cpu_reference, both arms reproduce their outputs bit-for-bit on a repeat pass, and (numeric_relation exact) candidate is identical_to_baseline",
        "baseline_max_abs_err": base_report.as_ref().ok().map(|r| r.max_abs_err),
        "baseline_pass": base_report.as_ref().ok().map(|r| r.pass),
        "baseline_reference_error": base_report.as_ref().err(),
        "identical_to_baseline": identical,
        "max_abs_diff_vs_baseline": diff,
        "output_bytes": cand_out.len(),
        "baseline_repeat_identical": base_repeat,
        "candidate_repeat_identical": cand_repeat,
        "numeric_relation": rec["numeric_contract"]["relation"].clone(),
        "repeat_on_swapped_buffers": true,
        "emu": "not applicable (chain)",
    });
    if !pass {
        rec["timing"] = json!({"status": "refused: candidate failed correctness"});
        return (rec, Outcome::CorrectnessFailed);
    }

    // 4. ABBA timing, events around one whole chain per launch slot
    let sampled = sample(
        &mut ChainExec {
            gpu: &mut gpu,
            w: &mut *w,
        },
        o,
    );
    let post = (|| -> Result<bool> {
        // The last timed chain of each arm must still equal its warm output.
        gpu.hip.device_synchronize().map_err(err)?;
        Ok(w.outputs(&mut gpu, Arm::Baseline)? == base_out
            && w.outputs(&mut gpu, Arm::Candidate)? == cand_out)
    })();
    match (sampled, post) {
        (Ok((s, stop, secs)), Ok(true)) => {
            match timed_record(o, &rec["shape"], &base_sha, &s, stop, secs) {
                Ok(mut t) => {
                    t["unit"] =
                        json!("us per chain (hipEvent around the whole chain, default stream)");
                    t["crossover"] = match crossover_json(&t) {
                        Ok(c) => c,
                        Err(e) => return fail(rec, "timing", e),
                    };
                    rec["timing"] = t;
                }
                Err(e) => return fail(rec, "timing", e),
            }
        }
        (Ok(_), Ok(false)) => {
            rec["timing"] =
                json!({"status": "post-sampling outputs changed; no performance claim"});
            return (rec, Outcome::CorrectnessFailed);
        }
        (Err(e), _) | (_, Err(e)) => return fail(rec, "timing", e),
    }
    rec["device_after"] = device_json(&gpu);
    (rec, Outcome::Ok)
}

/// The human verdict line of a timing record.
fn verdict(ts: &Value) -> String {
    let f = |k: &str| ts[k].as_f64().unwrap_or(f64::NAN);
    if ts["separable"] == true {
        let (dir, ratio) = (ts["direction"].as_str().unwrap_or("?"), f("block_ratio"));
        if ts["material"] == true {
            format!(
                "{dir} (material): block ratio {ratio:.6} after {} rounds, sign p {:.3e}",
                ts["rounds_run"],
                f("sign_p")
            )
        } else {
            format!(
                "separable but below min-effect ({:.3}%): {dir} by {:.3}% (block ratio {ratio:.6}), sign p {:.3e}, {} rounds",
                f("min_effect") * 100.0,
                (ratio - 1.0).abs() * 100.0,
                f("sign_p"),
                ts["rounds_run"]
            )
        }
    } else {
        format!(
            "not separable after {} rounds ({}); {}/{}/{} rounds faster/slower/tied, sign p {:.3e}, block ratio {:.6}",
            ts["rounds_run"],
            ts["stop"],
            ts["rounds_faster"],
            ts["rounds_slower"],
            ts["rounds_tied"],
            f("sign_p"),
            f("block_ratio")
        )
    }
}

fn main() {
    let mut opts = match parse_args(std::env::args().skip(1)) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("pm_loop: {e}");
            std::process::exit(2);
        }
    };
    if let Some(path) = &opts.noise_floor {
        match std::fs::read_to_string(path)
            .map_err(err)
            .and_then(|t| serde_json::from_str::<Value>(&t).map_err(err))
        {
            Ok(v) if v["schema"] == "hipfire.pm_noise.v1" => opts.noise_floor_doc = Some(v),
            Ok(_) => {
                eprintln!(
                    "pm_loop: --noise-floor {}: not a hipfire.pm_noise.v1 document",
                    path.display()
                );
                std::process::exit(2);
            }
            Err(e) => {
                eprintln!("pm_loop: --noise-floor {}: {e}", path.display());
                std::process::exit(2);
            }
        }
    }
    let (rec, outcome) = run(&mut opts);
    println!(
        "{}",
        serde_json::to_string(&rec).expect("record serializes")
    );
    if let Some(ts) = rec.get("timing").filter(|t| t["ratio"].is_number()) {
        let verdict = verdict(ts);
        eprintln!("pm_loop: {verdict}");
    }
    std::process::exit(match outcome {
        Outcome::Ok => 0,
        Outcome::CorrectnessFailed => 1,
        Outcome::CandidateUnusable => 3,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> impl Iterator<Item = String> + '_ {
        s.split_whitespace().map(str::to_owned)
    }

    #[test]
    fn crossover_splits_rounds_by_binding() {
        // Pure placement effect: own-binding rounds 1% faster, swapped 1% slower.
        let t = json!({"noise": {"round_ratios": [0.99, 1.01, 0.99, 1.01, 0.99]}});
        let c = crossover_json(&t).unwrap();
        assert_eq!(
            (c["rounds_own"].as_u64(), c["rounds_swapped"].as_u64()),
            (Some(3), Some(2))
        );
        let cross = c["crossover_ratio"].as_f64().unwrap();
        let place = c["placement_ratio"].as_f64().unwrap();
        assert!((cross - (0.99f64 * 1.01).sqrt()).abs() < 1e-12, "{cross}");
        assert!((place - (0.99f64 / 1.01).sqrt()).abs() < 1e-12, "{place}");
        // A single round has no swapped half: counts only.
        let c = crossover_json(&json!({"noise": {"round_ratios": [1.0]}})).unwrap();
        assert!(c.get("crossover_ratio").is_none());
        assert!(crossover_json(&json!({})).is_err());
    }

    #[test]
    fn cand_object_is_gfx1100_or_gfx1201_qwen4_moe_sym_only() {
        let ok = "--kernel qwen4_moe_sym --arch gfx1100 --offline --cand-object /tmp/c.co";
        let o = parse_args(args(ok)).unwrap();
        assert_eq!(
            o.cand_object.as_deref(),
            Some(std::path::Path::new("/tmp/c.co"))
        );
        assert!(parse_args(args("--kernel qwen4_moe_sym"))
            .unwrap()
            .cand_object
            .is_none());
        for bad in [
            "--kernel qwen4_moe_sym --arch gfx1151 --cand-object /tmp/c.co",
            "--kernel qwen4_moe_sym --arch gfx1201 --cand-object /tmp/c.co --rip /tmp/a.rip",
            "--kernel iu4_gemm --arch gfx1100 --cand-object /tmp/c.co",
            "--kernel qwen4_moe_sym --arch gfx1100 --cand-object /tmp/c.co --rip /tmp/a.rip",
            "--kernel qwen4_moe_sym --arch gfx1100 --cand-object /tmp/c.co --null-control aa",
            "--kernel qwen4_moe_sym --arch gfx1100 --cand-object",
        ] {
            assert!(parse_args(args(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn defaults_and_knob_passthrough() {
        let o = parse_args(args(
            "--kernel iu4_gemm --shape M=128,N=128,K=256 --fold k128 --base-tile 128x128x8",
        ))
        .unwrap();
        assert_eq!(o.kernel, "iu4_gemm");
        assert_eq!(o.shape.as_deref(), Some("M=128,N=128,K=256"));
        assert_eq!(
            (o.time, o.rounds, o.emu),
            (DEFAULT_TIME, DEFAULT_ROUNDS, false)
        );
        assert_eq!(o.knobs.0.get("fold").map(String::as_str), Some("k128"));
        assert_eq!(
            o.knobs.0.get("base-tile").map(String::as_str),
            Some("128x128x8")
        );
        assert_eq!(o.knobs.0.len(), 2, "driver flags are not knobs");
    }

    #[test]
    fn rip_flag_parses_and_is_iu4_only() {
        let o = parse_args(args("--kernel iu4_gemm --rip /tmp/a.rip --fold k128")).unwrap();
        assert_eq!(o.rip, Some(PathBuf::from("/tmp/a.rip")));
        assert_eq!(o.knobs.0.len(), 1, "--rip is not a knob");
        assert!(parse_args(args("--kernel iu4_gemm")).unwrap().rip.is_none());
        let e = parse_args(args("--kernel nope --rip /tmp/a.rip")).unwrap_err();
        assert!(e.contains("--rip is only supported with --kernel"), "{e}");
        // qwen4_moe_sym takes --rip (every MQ4 entry; the entry check is the workload's).
        let o = parse_args(args(
            "--kernel qwen4_moe_sym --rip /tmp/a.rip --epi down_gp_nt4",
        ))
        .unwrap();
        assert_eq!(o.rip, Some(PathBuf::from("/tmp/a.rip")));
        assert!(parse_args(args("--kernel iu4_gemm --rip")).is_err());
        // The PM producers take --rip (the twin of the selected export / variant).
        for k in [
            "iu4_c8_producer_pm",
            "moe_rotate_i4_pm",
            "gdn_chunk_prep_pm",
        ] {
            let o = parse_args(args(&format!("--kernel {k} --rip /tmp/a.rip"))).unwrap();
            assert_eq!(o.rip, Some(PathBuf::from("/tmp/a.rip")), "{k}");
        }
    }

    #[test]
    fn driver_flags_and_emu() {
        let o = parse_args(args(
            "--kernel qwen4_moe_sym --time 7 --rounds 2 --emu --out /tmp/x --epi down_nt4",
        ))
        .unwrap();
        assert_eq!((o.time, o.rounds, o.emu), (7, 2, true));
        assert_eq!(o.out, Some(PathBuf::from("/tmp/x")));
        assert!(o.shape.is_none());
        assert_eq!(o.knobs.0.len(), 1);
    }

    #[test]
    fn bad_args_are_refused() {
        for bad in [
            "",
            "--shape M=1",
            "--kernel",
            "--kernel k --time 0",
            "--kernel k --rounds 0",
            "--kernel k --time x",
            "--kernel k positional",
            "--kernel k --fold a --fold b",
            "--kernel k --time 18446744073709551615",
        ] {
            assert!(parse_args(args(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn unknown_kernel_and_leftover_knobs_are_refused() {
        let mut o = parse_args(args("--kernel nope")).unwrap();
        assert!(build_workload(&mut o)
            .err()
            .is_some_and(|e| e.contains("unknown --kernel")));
        let mut k = Knobs::default();
        k.0.insert("bogus".into(), "1".into());
        assert!(k.finish().unwrap_err().contains("--bogus"));
    }

    #[test]
    fn abba_schedule_is_balanced_and_blocked() {
        let seq: Vec<_> = timing_schedule(3, 5).unwrap().collect();
        assert_eq!(seq.len(), 3 * 5 * 4);
        assert_eq!(
            seq.iter().filter(|a| **a == TimingArm::Incumbent).count(),
            30
        );
        // A A A | B B B | B B B | A A A, repeated per round.
        assert_eq!(
            &seq[..12],
            &[
                TimingArm::Incumbent,
                TimingArm::Incumbent,
                TimingArm::Incumbent,
                TimingArm::Candidate,
                TimingArm::Candidate,
                TimingArm::Candidate,
                TimingArm::Candidate,
                TimingArm::Candidate,
                TimingArm::Candidate,
                TimingArm::Incumbent,
                TimingArm::Incumbent,
                TimingArm::Incumbent,
            ]
        );
    }

    fn samples(base: &[f64], cand: &[f64]) -> Sampled {
        Sampled {
            base_ns: base.to_vec(),
            cand_ns: cand.to_vec(),
        }
    }

    /// `rounds` ABBA rounds of `n`-launch blocks, every baseline launch `base`
    /// and every candidate launch `cand` (ns).
    fn flat(n: usize, rounds: usize, base: f64, cand: f64) -> Sampled {
        samples(&vec![base; 2 * n * rounds], &vec![cand; 2 * n * rounds])
    }

    #[test]
    fn timing_record_units_ratio_and_separability() {
        use abba::ExtendStop::{MaxRounds, Separable};
        let t = timing_record(
            &flat(2, 12, 100_000.0, 200_000.0),
            2,
            6,
            Separable,
            1.5,
            0.005,
        )
        .unwrap();
        assert_eq!(t["separable"], true);
        assert_eq!(t["direction"], "slower");
        assert_eq!(t["separable_block_disjoint"], true);
        assert_eq!(t["separable_full_range"], true);
        assert_eq!(t["baseline_us"]["median"], 100.0);
        assert_eq!(t["candidate_us"]["min"], 200.0);
        assert_eq!(t["ratio"], 2.0);
        assert_eq!(t["block_ratio"], 2.0);
        assert_eq!(
            (t["rounds_run"].as_u64(), t["extended"].as_bool()),
            (Some(12), Some(true))
        );
        assert_eq!(
            (t["rounds_slower"].as_u64(), t["rounds_faster"].as_u64()),
            (Some(12), Some(0))
        );
        assert_eq!(t["stop"], "separable");
        assert_eq!(t["p_max"], 1e-3);
        assert!(t["sign_p"].as_f64().unwrap() <= 1e-3);
        assert_eq!(t["round_ratio_min"], 2.0);
        assert_eq!(t["candidate_block_median_us"]["max"], 200.0);
        // Equal arms: every round ties, never separable.
        let same = timing_record(&flat(2, 12, 7.0, 7.0), 2, 12, MaxRounds, 1.0, 0.005).unwrap();
        assert_eq!(same["separable"], false);
        assert_eq!(
            (same["direction"].as_str(), same["rounds_tied"].as_u64()),
            (Some("tie"), Some(12))
        );
        assert_eq!(
            (same["ratio"].clone(), same["sign_p"].clone()),
            (json!(1.0), json!(1.0))
        );
        assert!(timing_record(&samples(&[], &[1.0; 8]), 1, 1, Separable, 0.0, 0.005).is_err());
        // The rule text names all three rules.
        let rule = t["rule"].as_str().unwrap();
        assert!(
            rule.contains("sign test")
                && rule.contains("block_disjoint")
                && rule.contains("full_range")
        );
    }

    /// The 7 ms hiccup of the noprio record: the paired rule stays separable,
    /// the twins' full-range rule (kept as `separable_full_range`) does not.
    #[test]
    fn one_hiccup_vetoes_only_the_full_range_rule() {
        let mut s = flat(2, 12, 330_000.0, 320_000.0);
        s.base_ns[3] = 7_003_000.0;
        let t = timing_record(&s, 2, 6, abba::ExtendStop::Separable, 1.0, 0.005).unwrap();
        assert_eq!(t["separable"], true);
        assert_eq!(t["direction"], "faster");
        assert_eq!(t["separable_full_range"], false);
        assert!(t["block_ratio"].as_f64().unwrap() < 1.0);
    }

    /// Fake sampler: `gap` ns between the arms, 1 s per round, `noise` makes
    /// the arms overlap until it is zero.
    fn fake_run(
        gap: f64,
        per_round_s: f64,
        now: &std::cell::Cell<f64>,
    ) -> impl FnMut(usize, &mut Sampled) -> Result<()> + '_ {
        move |extra, s| {
            for r in 0..extra {
                for _ in 0..4 {
                    let j = f64::from((s.base_ns.len() + r) as u32 % 7);
                    s.base_ns.push(300_000.0 + j * 1_000.0);
                    s.cand_ns.push(300_000.0 + j * 1_000.0 - gap);
                }
            }
            now.set(now.get() + per_round_s * extra as f64);
            Ok(())
        }
    }

    #[test]
    fn sampling_extends_by_doubling_until_a_stop() {
        use abba::ExtendStop::{MaxRounds, Separable, TimeBudget};
        let now = std::cell::Cell::new(0.0);
        let clock = &mut || now.get();
        // Identical arms never separate: 2 -> 4 -> 8 -> 16 -> 20 (max_rounds).
        let mut run = fake_run(0.0, 1.0, &now);
        let (s, stop, secs) = drive_sampling(2, 2, 20, 1e9, clock, &mut run).unwrap();
        assert_eq!((s.base_ns.len() / 4, stop, secs), (20, MaxRounds, 20.0));
        // A clear gap needs 11 paired rounds: 12 initial rounds stop at once ...
        now.set(0.0);
        let clock = &mut || now.get();
        let mut run = fake_run(50_000.0, 1.0, &now);
        let (s, stop, _) = drive_sampling(2, 12, 64, 1e9, clock, &mut run).unwrap();
        assert_eq!((s.base_ns.len() / 4, stop), (12, Separable));
        // ... and 2 initial rounds double 2 -> 4 -> 8 -> 16 (p <= 1e-3 first at 16).
        now.set(0.0);
        let clock = &mut || now.get();
        let mut run = fake_run(50_000.0, 1.0, &now);
        let (s, stop, _) = drive_sampling(2, 2, 64, 1e9, clock, &mut run).unwrap();
        assert_eq!((s.base_ns.len() / 4, stop), (16, Separable));
        // Time budget: 1 s/round, 10 s budget, never separable: 2, 4, 8 (6+2...) then stop.
        now.set(0.0);
        let clock = &mut || now.get();
        let mut run = fake_run(0.0, 1.0, &now);
        let (s, stop, secs) = drive_sampling(2, 2, 64, 10.0, clock, &mut run).unwrap();
        assert_eq!(stop, TimeBudget);
        assert!(
            secs <= 10.0 && s.base_ns.len() / 4 == 10,
            "{secs} {}",
            s.base_ns.len()
        );
    }

    /// Two byte-identical arms that a placement effect makes 0.25% slower every
    /// round: statistically separable, but not material at the default gate.
    #[test]
    fn immaterial_separable_difference_is_not_material() {
        use abba::ExtendStop::Separable;
        let s = flat(2, 40, 300_000.0, 300_750.0);
        let t = timing_record(&s, 2, 6, Separable, 1.0, DEFAULT_MIN_EFFECT).unwrap();
        assert_eq!(t["separable"], true);
        assert_eq!(t["direction"], "slower");
        assert_eq!(t["material"], false);
        assert_eq!(t["min_effect"], 0.005);
        let v = verdict(&t);
        assert!(v.starts_with("separable but below min-effect"), "{v}");
        // The same data is material once the gate is below the effect, and a real 2% win is material.
        assert_eq!(
            timing_record(&s, 2, 6, Separable, 1.0, 0.001).unwrap()["material"],
            true
        );
        let win = timing_record(
            &flat(2, 40, 300_000.0, 294_000.0),
            2,
            6,
            Separable,
            1.0,
            DEFAULT_MIN_EFFECT,
        )
        .unwrap();
        assert_eq!(
            (win["material"].clone(), win["direction"].clone()),
            (json!(true), json!("faster"))
        );
        assert!(
            verdict(&win).starts_with("faster (material)"),
            "{}",
            verdict(&win)
        );
        // Not separable is never material.
        let tie = timing_record(
            &flat(2, 12, 7.0, 7.0),
            2,
            12,
            abba::ExtendStop::MaxRounds,
            1.0,
            0.0,
        )
        .unwrap();
        assert_eq!(tie["material"], false);
        assert!(verdict(&tie).starts_with("not separable"));
    }

    #[test]
    fn min_effect_flag_parses() {
        assert_eq!(
            parse_args(args("--kernel iu4_gemm")).unwrap().min_effect,
            0.005
        );
        let o = parse_args(args("--kernel iu4_gemm --min-effect 0.02")).unwrap();
        assert_eq!(o.min_effect, 0.02);
        assert!(o.knobs.0.is_empty());
        for bad in ["--min-effect -1", "--min-effect x", "--min-effect nan"] {
            assert!(
                parse_args(args(&format!("--kernel k {bad}"))).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn extension_flags_parse() {
        let o = parse_args(args("--kernel iu4_gemm")).unwrap();
        assert_eq!((o.time, o.rounds, o.max_rounds), (50, 6, 256));
        assert_eq!(o.time_budget_s, 240.0);
        let o = parse_args(args(
            "--kernel iu4_gemm --rounds 3 --max-rounds 9 --time-budget-s 12.5",
        ))
        .unwrap();
        assert_eq!((o.rounds, o.max_rounds, o.time_budget_s), (3, 9, 12.5));
        assert!(o.knobs.0.is_empty());
        for bad in [
            "--kernel k --max-rounds 0",
            "--kernel k --rounds 8 --max-rounds 4",
            "--kernel k --time-budget-s 0",
            "--kernel k --time-budget-s -3",
            "--kernel k --time-budget-s nan",
            "--kernel k --time-budget-s x",
        ] {
            assert!(parse_args(args(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn emu_refusals_drop_register_dumps() {
        let long = json!(
            "refused (wg (0,0): k inst=3 v_div_scale_f32: not measured | arch=Gfx1201 w0=[0x1])"
        );
        assert_eq!(
            tidy_emu(long),
            json!("refused (wg (0,0): k inst=3 v_div_scale_f32: not measured)")
        );
        let short = json!("refused (no semantics for s_setprio)");
        assert_eq!(tidy_emu(short.clone()), short);
        let ran = json!({"status": "pass"});
        assert_eq!(tidy_emu(ran.clone()), ran);
    }

    #[test]
    fn m7_cache_summary_vocabulary() {
        use peacemaker_isa::pm_check::M7Cache;
        let all = [M7Cache::Hit, M7Cache::Miss, M7Cache::Disabled].map(M7Cache::as_str);
        assert_eq!(all, ["hit", "miss", "disabled"]);
    }

    #[test]
    fn noise_fields_on_synthetic_samples() {
        use abba::ExtendStop::Separable;
        let t = timing_record(
            &flat(2, 12, 100_000.0, 200_000.0),
            2,
            6,
            Separable,
            1.0,
            0.005,
        )
        .unwrap();
        let n = &t["noise"];
        assert_eq!(n["round_ratios"].as_array().unwrap().len(), 12);
        assert!(n["round_ratios"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| *r == 2.0));
        assert_eq!(n["round_ratio"]["n"], 12);
        assert_eq!(n["round_ratio"]["median"], 2.0);
        assert_eq!(n["round_ratio"]["mad"], 0.0);
        assert_eq!(n["baseline_block_median"]["n"], 24);
        assert_eq!(n["baseline_block_median"]["median"], 100.0);
        assert_eq!(n["candidate_block_median"]["median"], 200.0);
        assert_eq!(n["baseline_drift_per_block"], 0.0);
        assert_eq!(n["candidate_drift_per_block"], 0.0);
        // Existing fields keep their meaning next to the new block.
        assert_eq!(t["block_ratio"], 2.0);
        // A ramping baseline drifts upward, a flat candidate does not (n=1: one launch per block).
        let base: Vec<f64> = (0..8)
            .map(|i| 100_000.0 + 10_000.0 * f64::from(i))
            .collect();
        let ramp = timing_record(
            &samples(&base, &[100_000.0; 8]),
            1,
            2,
            Separable,
            1.0,
            0.005,
        )
        .unwrap();
        assert!(ramp["noise"]["baseline_drift_per_block"].as_f64().unwrap() > 0.0);
        assert_eq!(ramp["noise"]["candidate_drift_per_block"], 0.0);
        // 8 launches per arm at n = 1 are 4 ABBA rounds (2 blocks per arm per round).
        assert_eq!(ramp["noise"]["round_ratios"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn null_control_and_noise_floor_flags_parse() {
        let o = parse_args(args(
            "--kernel iu4_gemm --null-control aa --noise-floor /tmp/n.json",
        ))
        .unwrap();
        assert!(o.null_control);
        assert_eq!(o.noise_floor, Some(PathBuf::from("/tmp/n.json")));
        assert!(o.knobs.0.is_empty(), "driver flags are not knobs");
        let d = parse_args(args("--kernel iu4_gemm")).unwrap();
        assert!(!d.null_control && d.noise_floor.is_none());
        for bad in [
            "--kernel k --null-control bb",
            "--kernel k --null-control",
            "--kernel k --null-control aa --rip /tmp/a.rip",
            "--kernel k --null-control aa --base-tile 128x128x8",
            "--kernel k --noise-floor",
        ] {
            assert!(parse_args(args(bad)).is_err(), "{bad}");
        }
        // --base-* without the null control is still a knob.
        assert!(parse_args(args("--kernel k --base-tile 1")).is_ok());
    }

    #[test]
    fn arch_flag_defaults_parses_and_is_tri_arch_only_for_moe_kernels() {
        assert_eq!(
            parse_args(args("--kernel iu4_gemm")).unwrap().arch,
            "gfx1201"
        );
        for a in ["gfx1201", "gfx1151", "gfx1100"] {
            let o = parse_args(args(&format!("--kernel qwen4_moe_sym --arch {a}"))).unwrap();
            assert_eq!(o.arch, a);
            assert!(o.knobs.0.is_empty(), "--arch is not a knob");
        }
        let e = parse_args(args("--kernel qwen4_moe_sym --arch gfx942")).unwrap_err();
        assert!(e.contains("--arch gfx942"), "{e}");
        assert!(parse_args(args("--kernel qwen4_moe_sym --arch")).is_err());
        // gfx1201 stays accepted everywhere; every other arch is tri-arch kernels only.
        assert!(parse_args(args("--kernel iu4_gemm --arch gfx1201")).is_ok());
        let e = parse_args(args("--kernel iu4_gemm --arch gfx1151")).unwrap_err();
        assert_eq!(
            e,
            "--kernel iu4_gemm targets gfx1201 only (got --arch gfx1151)"
        );
        assert!(parse_args(args("--kernel fa_prefill --arch gfx1100")).is_err());
        assert!(parse_args(args("--kernel qwen4_moe_sym --arch gfx1151")).is_ok());
        assert!(parse_args(args("--kernel qwen4_moe_verify --arch gfx1151")).is_ok());
        assert!(parse_args(args("--kernel qwen4_moe_verify --arch gfx1100")).is_ok());
    }

    #[test]
    fn chain_kernel_refuses_rip_and_takes_the_null_control() {
        let e = parse_args(args("--kernel qwen4_moe_verify --rip /tmp/a.rip")).unwrap_err();
        assert!(
            e.contains("--rip is not supported with --kernel qwen4_moe_verify"),
            "{e}"
        );
        let o = parse_args(args(
            "--kernel qwen4_moe_verify --null-control aa --arch gfx1151 --shape T=4",
        ))
        .unwrap();
        assert!(o.null_control && o.rip.is_none());
        assert_eq!(o.arch, "gfx1151");
        assert!(parse_args(args(
            "--kernel qwen4_moe_verify --null-control aa --base-tile 1"
        ))
        .is_err());
        assert_eq!(arch_enum("gfx1151"), Ok(peacemaker_isa::Arch::Gfx1151));
        assert!(arch_enum("gfx942").is_err());
    }

    #[test]
    fn vs_noise_floor_matches_aa_group_and_never_guesses() {
        let shape = json!({"M": 128, "N": 128});
        let doc = json!({"schema": "hipfire.pm_noise.v1", "groups": [
            {"id": "g1", "aa": true, "kernel": "iu4_gemm", "shape": {"M": 128, "N": 128},
             "baseline_sha256": "aa", "floor": 0.002},
            {"id": "g2", "aa": true, "kernel": "iu4_gemm", "shape": {"M": 128, "N": 128},
             "baseline_sha256": "aa", "floor": 0.004},
            {"id": "g3", "aa": false, "kernel": "iu4_gemm", "shape": {"M": 128, "N": 128},
             "baseline_sha256": "aa", "floor": 0.5},
            {"id": "g4", "aa": true, "kernel": "iu4_gemm", "shape": {"M": 1, "N": 1},
             "baseline_sha256": "aa", "floor": 0.5},
            {"id": "g5", "aa": true, "kernel": "mq6_i8_gemm", "shape": {"M": 128, "N": 128},
             "baseline_sha256": "bb", "floor": Value::Null},
        ]});
        // The largest matching floor wins; A/B groups and other shapes are ignored.
        let v = vs_noise_floor(&doc, "iu4_gemm", &shape, "aa", 0.99);
        assert_eq!(v["group"], "g2");
        assert_eq!(v["floor"], 0.004);
        assert!((v["effect"].as_f64().unwrap() - 0.01).abs() < 1e-12);
        assert_eq!(v["exceeds"], true);
        let v = vs_noise_floor(&doc, "iu4_gemm", &shape, "aa", 1.003);
        assert_eq!(v["exceeds"], false);
        // No match: wrong sha, wrong kernel.
        assert_eq!(
            vs_noise_floor(&doc, "iu4_gemm", &shape, "cc", 0.9)["status"],
            "no matching A/A group"
        );
        assert_eq!(
            vs_noise_floor(&doc, "fp8_gemm", &shape, "aa", 0.9)["status"],
            "no matching A/A group"
        );
        // A matching group without a floor does not invent one.
        let v = vs_noise_floor(&doc, "mq6_i8_gemm", &shape, "bb", 0.9);
        assert!(v["floor"].is_null() && v["status"].is_string());
        assert_eq!(v["group"], "g5");
    }

    #[test]
    fn flush_flag_parses_and_is_cold_kernels_only() {
        for k in ["iu4_gemm", "qkvzaset"] {
            let o = parse_args(args(&format!("--kernel {k} --flush --shape N=64"))).unwrap();
            assert_eq!(o.cold, cold::ColdMode::Flush, "{k}");
            assert!(
                o.knobs.0.is_empty(),
                "--flush is not a knob: {:?}",
                o.knobs.0
            );
            assert_eq!(
                parse_args(args(&format!("--kernel {k}"))).unwrap().cold,
                cold::ColdMode::Hot
            );
        }
        // `--flush` takes no value: the next flag is parsed as itself.
        let o = parse_args(args("--kernel iu4_gemm --flush --time 7 --epi add")).unwrap();
        assert_eq!((o.cold, o.time), (cold::ColdMode::Flush, 7));
        assert_eq!(o.knobs.0.get("epi").map(String::as_str), Some("add"));
        for k in [
            "qwen4_moe_sym",
            "qwen4_moe_verify",
            "mq6_i8_gemm",
            "fa_prefill",
            "nope",
        ] {
            let e = parse_args(args(&format!("--kernel {k} --flush"))).unwrap_err();
            assert!(
                e.contains("--flush is only supported with --kernel iu4_gemm|qkvzaset"),
                "{k}: {e}"
            );
        }
        assert!(DRIVER_FLAGS.contains("[--pool | --flush-l2 | --flush]"));
        assert_eq!(workload::COLD_KERNELS, ["iu4_gemm", "qkvzaset"]);
    }

    #[test]
    fn cold_aware_workloads_build_and_the_rest_refuse_flush() {
        // The kernels that take --flush build GPU-free and report a pool once cold.
        for (kernel, shape) in [("iu4_gemm", "M=5120,N=64,K=17408"), ("qkvzaset", "N=64")] {
            let mut o = parse_args(args(&format!(
                "--kernel {kernel} --flush --shape {shape} --epi add"
            )))
            .unwrap();
            if kernel == "qkvzaset" {
                o.knobs.0.remove("epi");
            }
            let mut w = build_workload(&mut o).unwrap();
            assert_eq!(w.pool_sets(), 1);
            w.set_cold(cold::ColdMode::Flush).unwrap();
            assert!(w.pool_sets() >= 3 && w.pool_sets() % 2 == 1, "{kernel}");
            assert_eq!(w.shape_json()["flush"], true);
            assert!(
                w.shape_json()["timing_pool"]["total_bytes"]
                    .as_u64()
                    .unwrap()
                    > cold::POOL_BYTES as u64
            );
        }
        // A workload without a pool refuses (the default `Workload::set_cold`).
        let mut o = parse_args(args("--kernel fa_prefill --shape fn:small")).unwrap();
        if let Ok(mut w) = build_workload(&mut o) {
            let e = w.set_cold(cold::ColdMode::Pool).unwrap_err();
            assert!(
                e.contains("--pool is not supported by --kernel fa_prefill"),
                "{e}"
            );
        }
    }

    #[test]
    fn timing_record_names_the_flush() {
        let mut hot = parse_args(args("--kernel iu4_gemm")).unwrap();
        let mut cold = parse_args(args("--kernel iu4_gemm --flush")).unwrap();
        for o in [&mut hot, &mut cold] {
            (o.time, o.rounds) = (2, 6);
        }
        let s = flat(2, 12, 100_000.0, 100_000.0);
        let shape = json!({});
        let t = timed_record(&hot, &shape, "sha", &s, abba::ExtendStop::MaxRounds, 1.0).unwrap();
        assert_eq!(t["flush"], false);
        assert!(t.get("flush_method").is_none());
        let t = timed_record(&cold, &shape, "sha", &s, abba::ExtendStop::MaxRounds, 1.0).unwrap();
        assert_eq!(t["flush"], true);
        let m = t["flush_method"].as_str().unwrap();
        assert!(
            m.starts_with("2 x 96 MiB device-to-device copy") && m.contains("outside"),
            "{m}"
        );
    }

    /// Dispatch `i` of the timing schedule runs on set `i % sets` (odd): both arms cycle the whole pool.
    #[test]
    fn rotating_pool_gives_every_arm_every_set() {
        for sets in [3usize, 5, 13] {
            let (mut base, mut cand) = (vec![0usize; sets], vec![0usize; sets]);
            for (slot, which) in timing_schedule(7, 6).unwrap().enumerate() {
                let set = slot % sets;
                match which {
                    TimingArm::Incumbent => base[set] += 1,
                    TimingArm::Candidate => cand[set] += 1,
                }
            }
            for (name, v) in [("base", &base), ("cand", &cand)] {
                assert!(
                    v.iter().all(|c| *c > 0),
                    "{name} misses a set at {sets} sets: {v:?}"
                );
                let (lo, hi) = (v.iter().min().unwrap(), v.iter().max().unwrap());
                assert!(hi - lo <= 2 * 7, "{name} is lopsided at {sets} sets: {v:?}");
            }
        }
    }

    #[test]
    fn pool_and_flush_l2_flags_parse_compose_and_exclude() {
        use cold::ColdMode::*;
        let mode =
            |flags: &str| parse_args(args(&format!("--kernel iu4_gemm {flags}"))).map(|o| o.cold);
        assert_eq!(mode("--pool").unwrap(), Pool);
        assert_eq!(mode("--flush-l2").unwrap(), FlushL2);
        assert_eq!(mode("--pool --flush-l2").unwrap(), FlushL2);
        assert_eq!(mode("--flush-l2 --pool").unwrap(), FlushL2);
        assert_eq!(mode("--pool --flush").unwrap(), Flush);
        assert_eq!(mode("").unwrap(), Hot);
        assert!(mode("--flush --flush-l2")
            .unwrap_err()
            .contains("exclusive"));
        assert!(mode("--flush-l2 --flush").is_err());
        // None of them is a knob, and a later flag with a value still parses.
        let o = parse_args(args("--kernel qkvzaset --pool --time 9 --variant b1s")).unwrap();
        assert_eq!((o.cold, o.time), (Pool, 9));
        assert_eq!(o.knobs.0.keys().collect::<Vec<_>>(), ["variant"]);
        for k in ["qwen4_moe_sym", "mq6_i8_gemm", "nope"] {
            for f in ["--pool", "--flush-l2", "--flush"] {
                let e = parse_args(args(&format!("--kernel {k} {f}"))).unwrap_err();
                assert!(
                    e.starts_with(&format!(
                        "{f} is only supported with --kernel iu4_gemm|qkvzaset"
                    )),
                    "{k} {f}: {e}"
                );
            }
        }
        assert!(DRIVER_FLAGS.contains("--pool | --flush-l2 | --flush"));
    }

    #[test]
    fn every_cold_mode_reaches_the_workload_shape() {
        use cold::ColdMode::*;
        for (kernel, shape) in [("iu4_gemm", "M=5120,N=64,K=17408"), ("qkvzaset", "N=64")] {
            let mut shapes = Vec::new();
            for m in [Hot, Pool, FlushL2, Flush] {
                let mut o = parse_args(args(&format!(
                    "--kernel {kernel} --shape {shape} --epi add"
                )))
                .unwrap();
                if kernel == "qkvzaset" {
                    o.knobs.0.remove("epi");
                }
                let mut w = build_workload(&mut o).unwrap();
                if m.pool() {
                    w.set_cold(m).unwrap();
                }
                assert_eq!(w.pool_sets() > 1, m.pool(), "{kernel} {m:?}");
                let s = w.shape_json();
                assert_eq!(s.get("timing_pool").is_some(), m.pool());
                assert_eq!(
                    s.get("flush").map(|v| v == &json!(true)),
                    m.pool().then_some(m == Flush)
                );
                assert_eq!(s.get("flush_l2").is_some(), m == FlushL2);
                shapes.push(s);
            }
            // The four modes never share a shape (a noise group keys on it).
            for i in 0..4 {
                for j in 0..i {
                    assert_ne!(shapes[i], shapes[j], "{kernel} {i} {j}");
                }
            }
        }
    }

    #[test]
    fn timing_record_names_the_cold_mode() {
        let s = flat(2, 12, 100_000.0, 100_000.0);
        for (flags, mode, flush) in [
            ("", "hot", false),
            ("--pool", "pool", false),
            ("--flush-l2", "flush-l2", false),
            ("--flush", "flush", true),
        ] {
            let mut o = parse_args(args(&format!("--kernel iu4_gemm {flags}"))).unwrap();
            (o.time, o.rounds) = (2, 6);
            let t =
                timed_record(&o, &json!({}), "sha", &s, abba::ExtendStop::MaxRounds, 1.0).unwrap();
            assert_eq!(
                (t["cold_mode"].as_str(), t["flush"].as_bool()),
                (Some(mode), Some(flush))
            );
            assert_eq!(t.get("flush_method").is_some(), flush, "{mode}: only the full flush is named here; flush-l2 names its bytes after the L2 query");
        }
    }
}
