// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! npu-railgun: host-overhead bench + certification of eager / prepared / chained START_CU submission on a resident
//! array design chosen with `--variant V9|V10|V10-prefix` (default V9): `gemm_array::design_v9`, `design_v10` or
//! `design_v10_prefix`, each `(m, n, k, V8_DEFAULT_EPILOGUE, Control::Fast)` (int8 shift 12, fast control, as npu-gemm).
//! An unknown variant exits 2 before any device is opened; so does a shape the chosen design rejects. The
//! `railgun-config:` line reports the selected design; every path below receives that one design.
//!
//! usage: npu-railgun M N K --gemms G --mode eager|prepared|chain [--variant V9|V10|V10-prefix] [--secs S | --steps S]
//!                     [--sets 2] [--neg stale-patch|stale-key|collision] [--distinct | --noop] [--lean [--tape]]
//!                     [--verify-every] [--timeout-ms T]
//!
//! Example prefix run: `npu-railgun 160 2560 640 --variant V10-prefix --gemms 8 --mode prepared --lean
//! --verify-every --steps 200`. `NPU_RAILGUN_PHASE=prefix tools/npu/npu-railgun.sh` compares both down shapes across all
//! three variants, with every full baseline before the lean/verify runs.
//!
//! One hardware context, one PDI and one instruction stream (DEV heap BOs, shared by every command). `G` GEMMs per
//! step, by default all with the SAME operand contents (one random A/B, one CPU reference) but distinct BOs;
//! `--distinct` (small shapes only, M*N*K <= 512^3) gives GEMM g its own seeded operands and reference.
//! `--noop` replaces the instruction stream by the minimal valid TXN (`Txn::aie2p_8col()` with ONE `write32` of 0 to
//! memtile col 0 data word 0x7FFC0, `Location::new(0, 1).address(0x7FFC0)`): same PDI, hwctx and A/B/C args, but no GEMM
//! runs, so there is no CPU reference, no poisoning/eager baseline/output verification (`exact=n/a`,
//! `busy_tops=wall_tops=n/a`, `bytes_equal_eager=n.a.`); the patch inventory/fresh-encode certification still applies;
//! it excludes `--neg`/`--distinct`. `--sets N` (default 2)
//! gives N buffer sets (A, B, C BOs per set and GEMM, identical inputs in every set); step `s` uses set `s % N`, so
//! the retained modes really have to patch argument words every step. `--steps S` (default 200 when neither given)
//! runs exactly S steps; `--secs S` runs steps until the wall clock passed S, then ONE more step (the final one,
//! decided at the start of that step so its C can be poisoned first) and stops: at least 2 steps, about one step more
//! than S seconds. `--steps`/`--secs` are exclusive.
//!
//! `--lean` runs every TIMED command with the design's lean instruction stream (`ArrayDesign::lean_insts`, its own
//! instruction BO; the full stream is never submitted in the timed region). A lean TXN needs one previous COMPLETED full
//! TXN on the same design/hwctx, so per hwctx ONE untimed full-TXN eager step (submit + wait, set 0, C poisoned) runs
//! first and must be exact against the CPU reference; it is also the byte baseline of the first/last step (for the eager
//! mode too). If `lean_insts()` fails the run is rejected with exit 2 before the device is opened. `--lean` excludes
//! `--noop` (no GEMM TXN to slim). The result line reports the timed stream size as `txn_bytes=`.
//!
//! * eager: per GEMM `HwCtx::submit` (encodes the command every time) + `HwCtx::wait`.
//! * prepared: G retained `Prepared`; per step patch the set's args, `submit_prepared` x G, wait the LAST seq, then
//!   every command header must read COMPLETED.
//! * chain: G `Prepared` + one `Chain` (G <= 36); per step patch, `Chain::refresh`, one `submit_chain`, one wait.
//!
//! Timing: t_step = the hot CPU work (eager: command encode inside `submit`; prepared/chain: patch + chain refresh)
//! through the last wait return, ONE `Instant` pair around it. Pre-step snapshots are taken BEFORE that region; the
//! snapshot diff, fresh-encode comparison and header certification run AFTER it (wall only), as do poisoning/verify.
//! `busy_tops` = G*useful_ops*steps / sum(t_step); `wall_tops` = the same / the whole loop wall (certification and
//! verification included). Percentiles are nearest-rank over all steps (step 0 included).
//!
//! Certification (always on):
//! (a) the C of EVERY GEMM of the first step and of the last step is poisoned (0xA5) beforehand and must equal the CPU
//!     reference exactly (`exact=<exact GEMM checks>/<checks>`; intermediate steps are neither poisoned nor checked,
//!     unless `--verify-every`: then, in every mode, EVERY step's C buffers are poisoned (plus flush) before the step
//!     OUTSIDE the timed region and every GEMM of every step is compared after it; `exact=` totals all steps;
//!     excludes `--noop`);
//! (b) retained modes (and eager with `--lean`) first run one full-TXN eager step on set 0 (same inputs), and the
//!     first/last step C buffers must be byte-identical to it (`bytes_equal_eager`); plain eager prints `n.a.`;
//! (c) retained modes: between the pre-step snapshot (before the timed region) and the post-wait snapshot every word of
//!     every retained BO (command BOs, chain BO) that differs (header state nibble ignored: the driver owns it) must be
//!     a declared `patch_sites()` word, the instruction BO must be unchanged, and each retained BO after the step must
//!     equal a fresh from-scratch encoding (`encode_start_cu`; the chain from its documented layout) -> `inventory:`
//!     line, any `undeclared`/`mismatched` fails the run;
//! (d) `--neg stale-patch` (retained modes, `--sets 2`, at least 2 steps): on step 1 the C argument of GEMM 0 is NOT
//!     patched. The fresh-encode diff MUST flag it (`NEG stale-patch: detected by fresh-encode diff`), the step is
//!     still submitted (it harmlessly rewrites set 0's C), and the C of set 1 / GEMM 0 (the buffer a correct patch would
//!     have filled) MUST fail the reference compare (`NEG stale-patch: detected by output check`). Exit 0 only if both
//!     fired and everything else passed.
//!     With `--verify-every` the stale step is still checked in full (G checks): its stale GEMM 0 must be inexact (that
//!     one intended miss is tolerated, without byte compare, only together with the output-check detection above).
//! (e) `--tape` (needs `--lean` and a retained mode, excludes `--noop`/`--neg stale-patch`): the retained submission is
//!     driven by a railgun::npu `Tape`. Setup records ONE step of G entries once (program key = the LEAN program's
//!     `ProgramKey`, config key = the PDI's, bindings A/B/C as `ArgAddress{0..2}` with the current full u64 device
//!     addresses; the program's dynamic list is the six declared cmd words 7..=12 of `Prepared::patch_sites()`); the
//!     full and the lean instruction streams (actual DEV BOs) are inserted into a `ProgramCache<&Bo>` (2 misses). Per
//!     step: the binding values are updated from the step's set, `Tape::check_replay(recorded sequence_hash)` runs and
//!     the retained commands are patched FROM the tape's bindings (each `ArgAddress{arg}` binding is paired with the
//!     set's BO and its address must equal the binding, else the step fails; only the declared words are patched);
//!     all timed. `Tape::transitions()` plus the previous step's last ConfigKey (entry 0 of a step) pick the program of
//!     every entry, resolved by a cache lookup that MUST be a hit (the builder is a sentinel error: a miss builds
//!     nothing) and that returns the actual instruction BO. That selection drives the submission: the first timed
//!     entry (step 0, GEMM 0) runs the FULL stream (`txn_bytes=` reports the lean size), every other entry the lean
//!     one. A `Prepared` cannot change its instruction BO, so two immutable pools (and Chains) are retained: step 0's
//!     (GEMM 0 on the full BO) and the steady one; the pool whose instruction BOs equal the selected ones is made
//!     active before the snapshots (swap, outside the timed region), no match fails the step. Untimed per entry:
//!     `verify_patch` (changed words within the declared sites, state nibble normalized) and `verify_fresh` against a
//!     from-scratch encoding with the selected program's BO. Cache lookups (they hash the whole instruction stream)
//!     are untimed certification. Result line:
//!     `tape: entries=G sequence_hash=0x.. programs=n hits=h misses=m full=f lean=l replay_checks=n fresh_equal=n`.
//! (f) `--neg stale-key` (needs `--tape`): right after recording (before the hardware context exists, so no GEMM is
//!     submitted) one byte of a copy of the lean stream is flipped and looked up: the new `ProgramKey` MUST miss the
//!     cache (the sentinel builder fails, nothing is built or submitted: `NEG stale-key: detected by cache miss (new
//!     key)`) and a replay of the tape with the new program key MUST fail `check_replay`. `--neg collision` (needs
//!     `--tape`) only prints that forced key collisions are covered by the railgun::npu unit test
//!     `tape::tests::forced_collision_errors_without_calling_builder` (production keys offer no collision hook) and
//!     exits 0 without building a mutated program.
//!
//! Output ends with `RESULT: PASS|FAIL`; exit status 0 pass, 1 fail, 2 bad usage/shape.
use npu_tools::util::{lcg, parse_num, percentile};
use peacemaker_npu::dma::Location;
use peacemaker_npu::kernels::gemm_array::{
    design_v10, design_v10_prefix, design_v9, ArrayDesign, V8_DEFAULT_EPILOGUE,
};
use peacemaker_npu::kernels::gemm_core::Control;
use peacemaker_npu::kernels::gemm_i8::ArgKind;
use peacemaker_npu::txn::Txn;
use railgun::npu::tape::{
    verify_fresh, verify_patch, BindingKind, CacheOutcome, ConfigKey, ProgramCache, ProgramSpec,
    Tape, TapeEntry, Transition,
};
use railgun::npu::{
    encode_start_cu, Bo, Chain, Device, HwCtx, PatchBo, PatchSite, Prepared, ERT_STATE_COMPLETED,
};
use std::time::Instant;

const USAGE: &str = "usage: npu-railgun M N K --gemms G --mode eager|prepared|chain [--variant V9|V10|V10-prefix] [--secs S | --steps S] [--sets 2] [--neg stale-patch|stale-key|collision] [--distinct | --noop] [--lean [--tape]] [--verify-every] [--timeout-ms T]";
/// Chain limit from the firmware command buffer (36 START_CU slots).
const MAX_CHAIN: usize = 36;
const NARGS: usize = 3;
/// Header word 0 low nibble is the driver-managed command state.
const STATE_CLEAR: u32 = !0xF;
const DEFAULT_STEPS: usize = 200;
/// `--distinct` computes one CPU reference per GEMM: only for shapes up to 512x512x512.
const DISTINCT_MAX_MNK: usize = 512 * 512 * 512;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Eager,
    Prepared,
    Chain,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Eager => "eager",
            Mode::Prepared => "prepared",
            Mode::Chain => "chain",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Neg {
    None,
    StalePatch,
    StaleKey,
    Collision,
}

impl Neg {
    fn name(self) -> &'static str {
        match self {
            Neg::None => "none",
            Neg::StalePatch => "stale-patch",
            Neg::StaleKey => "stale-key",
            Neg::Collision => "collision",
        }
    }
}

/// Array design under test; every path (eager/prepared/chain/lean/tape/verify-every) receives the one built from it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DesignVariant {
    V9,
    V10,
    V10Prefix,
}

impl DesignVariant {
    fn name(self) -> &'static str {
        match self {
            DesignVariant::V9 => "V9",
            DesignVariant::V10 => "V10",
            DesignVariant::V10Prefix => "V10-prefix",
        }
    }

    /// May panic on an unsupported shape (the `gemm_array` builders assert); `main` guards it with `catch_unwind`.
    fn design(self, m: usize, n: usize, k: usize) -> ArrayDesign {
        let build = match self {
            DesignVariant::V9 => design_v9,
            DesignVariant::V10 => design_v10,
            DesignVariant::V10Prefix => design_v10_prefix,
        };
        build(m, n, k, V8_DEFAULT_EPILOGUE, Control::Fast)
    }
}

struct Cfg {
    variant: DesignVariant,
    m: usize,
    n: usize,
    k: usize,
    gemms: usize,
    mode: Mode,
    secs: Option<f64>,
    steps: Option<usize>,
    sets: usize,
    neg: Neg,
    distinct: bool,
    noop: bool,
    lean: bool,
    tape: bool,
    verify_every: bool,
    timeout_ms: u64,
}

/// `Ok(None)` = help requested.
fn parse(args: &[String]) -> Result<Option<Cfg>, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return Ok(None);
    }
    if args.len() < 3 {
        return Err("M N K are required".into());
    }
    let dim = |i: usize, name: &str| -> Result<usize, String> {
        let v: usize = parse_num(name, &args[i])?;
        if v == 0 {
            Err(format!("{name} must be > 0"))
        } else {
            Ok(v)
        }
    };
    let (m, n, k) = (dim(0, "M")?, dim(1, "N")?, dim(2, "K")?);
    let (
        mut gemms,
        mut mode,
        mut secs,
        mut steps,
        mut sets,
        mut neg,
        mut timeout_ms,
        mut distinct,
        mut noop,
        mut lean,
    ) = (
        1usize,
        None,
        None,
        None,
        2usize,
        Neg::None,
        3000u64,
        false,
        false,
        false,
    );
    let (mut tape, mut verify_every, mut variant) = (false, false, DesignVariant::V9);
    let mut i = 3;
    while i < args.len() {
        let flag = args[i].as_str();
        if matches!(
            flag,
            "--distinct" | "--noop" | "--lean" | "--tape" | "--verify-every"
        ) {
            match flag {
                "--noop" => noop = true,
                "--lean" => lean = true,
                "--tape" => tape = true,
                "--verify-every" => verify_every = true,
                _ => distinct = true,
            }
            i += 1;
            continue;
        }
        let val = args
            .get(i + 1)
            .map(String::as_str)
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag {
            "--gemms" => gemms = parse_num(flag, val)?,
            "--mode" => {
                mode = Some(match val {
                    "eager" => Mode::Eager,
                    "prepared" => Mode::Prepared,
                    "chain" => Mode::Chain,
                    _ => {
                        return Err(format!(
                            "--mode must be eager, prepared or chain, got {val:?}"
                        ))
                    }
                })
            }
            "--secs" => secs = Some(parse_num::<f64>(flag, val)?),
            "--steps" => steps = Some(parse_num::<usize>(flag, val)?),
            "--sets" => sets = parse_num(flag, val)?,
            "--neg" => {
                neg = match val {
                    "stale-patch" => Neg::StalePatch,
                    "stale-key" => Neg::StaleKey,
                    "collision" => Neg::Collision,
                    _ => {
                        return Err(format!(
                            "--neg must be stale-patch, stale-key or collision, got {val:?}"
                        ))
                    }
                };
            }
            "--timeout-ms" => timeout_ms = parse_num(flag, val)?,
            "--variant" => {
                variant = match val {
                    "V9" => DesignVariant::V9,
                    "V10" => DesignVariant::V10,
                    "V10-prefix" => DesignVariant::V10Prefix,
                    _ => {
                        return Err(format!(
                            "--variant must be V9, V10 or V10-prefix, got {val:?}"
                        ))
                    }
                };
            }
            _ => return Err(format!("unknown argument {flag:?}")),
        }
        i += 2;
    }
    let mode = mode.ok_or("--mode is required")?;
    if gemms == 0 {
        return Err("--gemms must be >= 1".into());
    }
    if mode == Mode::Chain && gemms > MAX_CHAIN {
        return Err(format!(
            "--mode chain supports at most {MAX_CHAIN} gemms, got {gemms}"
        ));
    }
    if !(1..=16).contains(&sets) {
        return Err("--sets must be in 1..=16".into());
    }
    if timeout_ms == 0 {
        return Err("--timeout-ms must be >= 1".into());
    }
    if secs.is_some() && steps.is_some() {
        return Err("--secs and --steps are exclusive".into());
    }
    if let Some(s) = secs {
        if !(s.is_finite() && s > 0.0) {
            return Err("--secs must be finite and > 0".into());
        }
    }
    if steps == Some(0) {
        return Err("--steps must be >= 1".into());
    }
    if neg == Neg::StalePatch {
        if mode == Mode::Eager {
            return Err("--neg stale-patch needs a retained mode (prepared or chain)".into());
        }
        if sets != 2 {
            return Err("--neg stale-patch needs --sets 2".into());
        }
        if steps.is_some_and(|s| s < 2) {
            return Err("--neg stale-patch needs at least 2 steps".into());
        }
        if tape {
            return Err(
                "--neg stale-patch excludes --tape (it certifies the plain retained patch path)"
                    .into(),
            );
        }
    }
    if matches!(neg, Neg::StaleKey | Neg::Collision) && !tape {
        return Err(format!("--neg {} needs --tape", neg.name()));
    }
    if distinct && m.saturating_mul(n).saturating_mul(k) > DISTINCT_MAX_MNK {
        return Err(format!("--distinct is for small shapes only (M*N*K <= {DISTINCT_MAX_MNK}: one CPU reference per GEMM)"));
    }
    if noop && (neg != Neg::None || distinct) {
        return Err("--noop has no output to verify: it excludes --neg and --distinct".into());
    }
    if noop && lean {
        return Err("--noop has no GEMM TXN to slim: it excludes --lean".into());
    }
    if noop && tape {
        return Err("--noop has no program to replay: it excludes --tape".into());
    }
    if noop && verify_every {
        return Err("--noop has no output to verify: it excludes --verify-every".into());
    }
    if tape && !lean {
        return Err(
            "--tape needs --lean (the tape selects between the full and the lean stream)".into(),
        );
    }
    if tape && mode == Mode::Eager {
        return Err("--tape needs a retained mode (prepared or chain)".into());
    }
    if secs.is_none() && steps.is_none() {
        steps = Some(DEFAULT_STEPS);
    }
    Ok(Some(Cfg {
        m,
        n,
        k,
        variant,
        gemms,
        mode,
        secs,
        steps,
        sets,
        neg,
        distinct,
        noop,
        lean,
        tape,
        verify_every,
        timeout_ms,
    }))
}

/// One buffer set: A, B, C BO of every GEMM.
struct Set {
    a: Vec<Bo>,
    b: Vec<Bo>,
    c: Vec<Bo>,
}

impl Set {
    fn args(&self, g: usize) -> [&Bo; NARGS] {
        [&self.a[g], &self.b[g], &self.c[g]]
    }
    fn poison(&mut self) {
        for bo in &mut self.c {
            bo.as_mut_slice().fill(0xA5);
            bo.flush();
        }
    }
}

struct Verified {
    exact: usize,
    checked: usize,
    /// None when no eager baseline applies.
    bytes_equal: Option<bool>,
    bad: Vec<usize>,
    /// `expect_bad_g0`: GEMM 0 was inexact as the negative test intends (counted in `checked`, not in `exact`/`bad`).
    expected_bad: usize,
}

/// Exact compare of every GEMM's C of `set` with the CPU reference, and the byte compare against the eager baseline
/// when there is one. `skip_g0`: leave the negative-test GEMM 0 out entirely. `expect_bad_g0`: GEMM 0 is the stale-patch
/// GEMM, still checked (counted in `checked`) but an inexact result is the intended outcome (`expected_bad`; no byte
/// compare of it, not a failure).
fn verify(
    d: &ArrayDesign,
    set: &Set,
    want: &[Vec<i8>],
    baseline: Option<&[Vec<u8>]>,
    skip_g0: bool,
    expect_bad_g0: bool,
) -> Verified {
    let mut v = Verified {
        exact: 0,
        checked: 0,
        bytes_equal: baseline.map(|_| true),
        bad: Vec::new(),
        expected_bad: 0,
    };
    for g in usize::from(skip_g0)..set.c.len() {
        set.c[g].flush();
        let out = set.c[g].as_slice();
        v.checked += 1;
        let stale = expect_bad_g0 && g == 0;
        // One shared reference unless `--distinct` (then one per GEMM).
        if matches_reference(d, out, &want[g.min(want.len() - 1)]) {
            v.exact += 1;
        } else if stale {
            v.expected_bad += 1;
            continue;
        } else {
            v.bad.push(g);
        }
        if let (Some(base), Some(eq)) = (baseline, v.bytes_equal.as_mut()) {
            *eq &= out == base[g].as_slice();
        }
    }
    v
}

fn matches_reference(d: &ArrayDesign, out: &[u8], want: &[i8]) -> bool {
    let got = d.unpack_out(out);
    got.len() == want.len() && got.iter().zip(want).all(|(g, w)| *g == i32::from(*w))
}

/// Indices (word 0 compared without its state nibble) where the two snapshots differ.
fn diff_words(a: &[u32], b: &[u32]) -> Vec<usize> {
    assert_eq!(a.len(), b.len());
    (0..a.len())
        .filter(|&i| {
            if i == 0 {
                a[0] & STATE_CLEAR != b[0] & STATE_CLEAR
            } else {
                a[i] != b[i]
            }
        })
        .collect()
}

/// Builder of every tape lookup: a lookup that is not a cache hit fails here, so nothing is ever built or inserted.
const CACHE_MISS: &str = "program cache MISS: no retained program for this key";

/// True when every retained command of `pool` uses exactly the instruction BO selected for its entry.
fn pool_fits(pool: &[Prepared], chosen: &[&Bo]) -> bool {
    pool.len() == chosen.len()
        && pool
            .iter()
            .zip(chosen)
            .all(|(p, bo)| p.inst_addr() == bo.xdna)
}

/// The recorded one-step tape, its program cache (actual instruction DEV BOs) and its counters.
struct TapeRun<'a> {
    pdi: &'a [u8],
    full_txn: &'a [u8],
    lean_txn: &'a [u8],
    dyn_list: Vec<(usize, BindingKind)>,
    cache: ProgramCache<&'a Bo>,
    tape: Tape,
    hash: u64,
    trans: Vec<Transition>,
    /// ConfigKey of the last entry of the previous step (decides entry 0's transition).
    prev: Option<ConfigKey>,
    /// Selected instruction BO of every entry of the current step.
    chosen: Vec<&'a Bo>,
    hits: usize,
    misses: usize,
    full: usize,
    lean: usize,
    replay_checks: usize,
    fresh_equal: usize,
    bad: usize,
}

impl<'a> TapeRun<'a> {
    /// Insert the full and the lean program (both must miss), then record ONE step of `gemms` entries from `set0`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        pdi: &'a [u8],
        full_txn: &'a [u8],
        lean_txn: &'a [u8],
        full_bo: &'a Bo,
        lean_bo: &'a Bo,
        sites: &[PatchSite],
        set0: &Set,
        gemms: usize,
    ) -> Result<Self, String> {
        let mut dyn_list = Vec::with_capacity(sites.len());
        for s in sites {
            if s.bo != PatchBo::Cmd(0) || s.word < 7 || (s.word - 7) / 2 >= NARGS {
                return Err(format!("tape: unexpected patch site {s:?}"));
            }
            dyn_list.push((
                s.word,
                BindingKind::ArgAddress {
                    arg: ((s.word - 7) / 2) as u8,
                },
            ));
        }
        let mut cache: ProgramCache<&'a Bo> = ProgramCache::new();
        let mut misses = 0;
        for (txn, bo) in [(full_txn, full_bo), (lean_txn, lean_bo)] {
            let spec = ProgramSpec {
                pdi,
                txn,
                dynamic: &dyn_list,
            };
            let (_, outcome) = cache
                .get_or_insert_with(spec, || Ok(bo))
                .map_err(|e| format!("tape cache insert: {e}"))?;
            if outcome != CacheOutcome::Miss {
                return Err(
                    "tape: the full and the lean instruction streams share one program key".into(),
                );
            }
            misses += 1;
        }
        let lean_key = ProgramSpec {
            pdi,
            txn: lean_txn,
            dynamic: &dyn_list,
        }
        .program_key();
        let config = ConfigKey::from_pdi(pdi);
        let entries: Vec<TapeEntry> = (0..gemms)
            .map(|g| TapeEntry {
                program: lean_key,
                config,
                bindings: set0
                    .args(g)
                    .iter()
                    .enumerate()
                    .map(|(i, bo)| (BindingKind::ArgAddress { arg: i as u8 }, bo.dev_addr()))
                    .collect(),
            })
            .collect();
        let tape = Tape::record(entries);
        let trans: Vec<Transition> = tape.transitions().collect();
        let hash = tape.sequence_hash;
        Ok(TapeRun {
            pdi,
            full_txn,
            lean_txn,
            dyn_list,
            cache,
            tape,
            hash,
            trans,
            prev: None,
            chosen: Vec::with_capacity(gemms),
            hits: 0,
            misses,
            full: 0,
            lean: 0,
            replay_checks: 0,
            fresh_equal: 0,
            bad: 0,
        })
    }

    /// Resolve the program of every entry of the next step: `Full` only for the very first entry of the run, else
    /// `Lean` (same ConfigKey as the previous entry, across step boundaries too). Every lookup must hit.
    fn select(&mut self) -> Result<(), String> {
        self.chosen.clear();
        for i in 0..self.tape.entries.len() {
            let config = self.tape.entries[i].config;
            let transition = if i == 0 {
                if self.prev == Some(config) {
                    Transition::Lean
                } else {
                    Transition::Full
                }
            } else {
                self.trans[i]
            };
            let full = transition == Transition::Full;
            let spec = ProgramSpec {
                pdi: self.pdi,
                txn: if full { self.full_txn } else { self.lean_txn },
                dynamic: &self.dyn_list,
            };
            match self
                .cache
                .get_or_insert_with(spec, || Err(CACHE_MISS.to_string()))
            {
                Ok((bo, CacheOutcome::Hit)) => self.chosen.push(*bo),
                Ok((_, CacheOutcome::Miss)) => {
                    return Err(format!("tape entry {i}: unexpected cache insert"))
                }
                Err(e) => {
                    return Err(format!(
                        "tape entry {i} ({}): {e}",
                        if full { "full" } else { "lean" }
                    ))
                }
            }
            self.hits += 1;
            if full {
                self.full += 1;
            } else {
                self.lean += 1;
            }
        }
        self.prev = self.tape.entries.last().map(|e| e.config);
        Ok(())
    }

    /// Update the binding values of entry `g` from the step's A/B/C BOs.
    fn bind(&mut self, g: usize, args: &[&Bo; NARGS]) {
        for (kind, value) in &mut self.tape.entries[g].bindings {
            if let BindingKind::ArgAddress { arg } = *kind {
                *value = args[arg as usize].dev_addr();
            }
        }
    }

    /// Untimed per-entry certification (header state nibble normalized: the driver owns it).
    fn certify(
        &mut self,
        s: usize,
        g: usize,
        pre: &[u32; 17],
        post: &[u32; 17],
        fresh: &[u32; 17],
        declared: &[usize],
    ) {
        let norm = |w: &[u32; 17]| {
            let mut n = *w;
            n[0] &= STATE_CLEAR;
            n
        };
        let (pre, post, fresh) = (norm(pre), norm(post), norm(fresh));
        if let Err(e) = verify_patch(&pre, &post, declared) {
            self.bad += 1;
            println!("step {s} gemm {g}: tape verify_patch: {e}");
        }
        match verify_fresh(&post, &fresh) {
            Ok(()) => self.fresh_equal += 1,
            Err(e) => {
                self.bad += 1;
                println!("step {s} gemm {g}: tape verify_fresh: {e}");
            }
        }
    }

    /// `--neg stale-key`: flip one byte of a copy of the lean stream; the lookup must be a cache MISS (sentinel builder,
    /// nothing built or inserted) and a tape replayed with the new program key must fail `check_replay`.
    fn neg_stale_key(&mut self) -> bool {
        let mut stale = self.lean_txn.to_vec();
        let at = stale.len() / 2;
        stale[at] ^= 0x01;
        let spec = ProgramSpec {
            pdi: self.pdi,
            txn: &stale,
            dynamic: &self.dyn_list,
        };
        let new_key = spec.program_key();
        let old_key = self.tape.entries[0].program;
        let before = self.cache.len();
        let miss = match self
            .cache
            .get_or_insert_with(spec, || Err(CACHE_MISS.to_string()))
        {
            Err(e) if e == CACHE_MISS => true,
            Ok((_, outcome)) => {
                println!("NEG stale-key: NOT detected by cache miss (lookup returned {outcome:?})");
                false
            }
            Err(e) => {
                println!("NEG stale-key: NOT detected by cache miss (unexpected error: {e})");
                false
            }
        };
        let untouched = self.cache.len() == before;
        if miss && new_key != old_key && untouched {
            println!("NEG stale-key: detected by cache miss (new key)");
            println!("NEG stale-key: lean byte {at} flipped, key {old_key:?} -> {new_key:?}; sentinel builder failed, no program built or inserted, no GEMM submitted");
        } else {
            println!("NEG stale-key: NOT detected by cache miss (miss={miss} new_key_differs={} cache_untouched={untouched})", new_key != old_key);
        }
        let mut replayed = self.tape.clone();
        for e in &mut replayed.entries {
            e.program = new_key;
        }
        let replay = replayed.check_replay(self.hash);
        match &replay {
            Err(e) => {
                println!("NEG stale-key: tape replay with the new key fails check_replay: {e}")
            }
            Ok(()) => println!("NEG stale-key: tape replay with the new key NOT detected"),
        }
        miss && new_key != old_key && untouched && replay.is_err()
    }
}

#[derive(Default)]
struct Inventory {
    steps: usize,
    changed_words: usize,
    undeclared: usize,
    mismatched: usize,
    neg_fresh_detected: bool,
    neg_output_detected: bool,
}

/// `lean_bytes`: the design's lean instruction stream when `--lean` (already validated by `main`, before any hardware).
fn run(cfg: &Cfg, d: &ArrayDesign, lean_bytes: Option<&[u8]>) -> Result<bool, String> {
    let (g_n, mode, timeout_ms) = (cfg.gemms, cfg.mode, cfg.timeout_ms);
    assert!(
        d.args.len() == NARGS
            && d.args[0].kind == ArgKind::In
            && d.args[1].kind == ArgKind::In
            && d.args[2].kind == ArgKind::Out,
        "design args are A, B, C"
    );
    let useful = d.useful_ops();
    // Instruction stream of the TIMED commands: the design's full one, its lean one with --lean, or with --noop the
    // minimal valid TXN: ONE write32 of 0 to memtile col 0 data word 0x7FFC0 (same PDI, hwctx and A/B/C args; the NPU
    // does no GEMM).
    let noop_bytes: Vec<u8>;
    let insts_src: &[u8] = if cfg.noop {
        let mut t = Txn::aie2p_8col();
        t.write32(Location::new(0, 1).address(0x7FFC0), 0);
        noop_bytes = t.to_bytes();
        &noop_bytes
    } else if let Some(lean) = lean_bytes {
        lean
    } else {
        &d.insts
    };
    let lean = lean_bytes.is_some();
    let neg_tag = match cfg.neg {
        Neg::None => String::new(),
        n => format!(" neg={}", n.name()),
    };
    println!(
        "railgun-config: design={} shape={}x{}x{} gemms={g_n} mode={} sets={} waves={} pdi={} B insts={} B args={:?} useful_ops/gemm={useful} timeout_ms={timeout_ms}{neg_tag}{}{}{}{}{}",
        cfg.variant.name(), cfg.m, cfg.n, cfg.k, mode.name(), cfg.sets, d.waves(), d.pdi.len(), insts_src.len(), d.args,
        if cfg.distinct { " distinct" } else { "" },
        if cfg.noop { " noop(insts=1 write32 memtile col0 0x7FFC0)" } else { "" },
        if lean { format!(" lean(timed insts={} B, untimed full warmup insts={} B)", insts_src.len(), d.insts.len()) } else { String::new() },
        if cfg.tape { " tape(step 0 GEMM 0 on the full stream, every other timed command lean)" } else { "" },
        if cfg.verify_every { " verify_every" } else { "" }
    );
    if cfg.tape {
        println!("NEG collision: unit-tested (tape::tests::forced_collision_errors_without_calling_builder)");
    }
    if cfg.neg == Neg::Collision {
        return Ok(true);
    }
    let neg_patch = cfg.neg == Neg::StalePatch;

    // ---- setup: heap, PDI, insts, buffers, commands; the context is created LAST so it drops first.
    let mut dev = Device::open().map_err(|e| format!("device open failed: {e}"))?;
    dev.map_heap(64 << 20)
        .map_err(|e| format!("device heap map failed: {e}"))?;
    let ntiles = dev.meta.cols as u32 * dev.meta.core.0 as u32;
    let pdi = dev
        .dev_bo(&d.pdi)
        .map_err(|e| format!("pdi bo failed: {e}"))?;
    let insts = dev
        .dev_bo(insts_src)
        .map_err(|e| format!("insts bo failed: {e}"))?;
    // --lean: the full stream's own BO, used ONLY by the untimed warmup/baseline step (timed commands use `insts`).
    let full_insts = if lean {
        Some(
            dev.dev_bo(&d.insts)
                .map_err(|e| format!("full insts bo failed: {e}"))?,
        )
    } else {
        None
    };
    let warm_insts: &Bo = full_insts.as_ref().unwrap_or(&insts);

    let mut sets: Vec<Set> = (0..cfg.sets)
        .map(|_| Set {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
        })
        .collect();
    // Operands: one random A/B and ONE reference shared by every GEMM (still distinct BOs per set and GEMM);
    // `--distinct` gives GEMM g its own seeded operands and reference. `--noop`: no operands and no reference at all
    // (A/B stay zero-filled, C is never poisoned or verified).
    let mut want: Vec<Vec<i8>> = Vec::with_capacity(if cfg.distinct { g_n } else { 1 });
    let mut packed: Option<[Vec<u8>; 2]> = None;
    for g in 0..g_n {
        if !cfg.noop && (cfg.distinct || packed.is_none()) {
            let mut seed = if cfg.distinct {
                0x5eed_u64 ^ (g as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            } else {
                0x5eed_u64
            };
            let a: Vec<i8> = (0..cfg.m * cfg.k).map(|_| lcg(&mut seed)).collect();
            let b: Vec<i8> = (0..cfg.k * cfg.n).map(|_| lcg(&mut seed)).collect();
            packed = Some(d.pack_in(&a, &b));
            // `reference` is the multi-threaded (row blocks) CPU GEMM + epilogue; results are int8, kept as i8.
            want.push(
                d.reference(&a, &b)
                    .into_iter()
                    .map(|x| i8::try_from(x).expect("reference is int8"))
                    .collect(),
            );
        }
        for (s, set) in sets.iter_mut().enumerate() {
            for (i, dst) in [(0, &mut set.a), (1, &mut set.b)] {
                let mut bo = dev
                    .shmem_bo(d.args[i].bytes)
                    .map_err(|e| format!("set {s} gemm {g} arg {i} bo failed: {e}"))?;
                if let Some(p) = &packed {
                    bo.as_mut_slice().copy_from_slice(&p[i]);
                    bo.flush();
                }
                dst.push(bo);
            }
            let mut c = dev
                .shmem_bo(d.args[2].bytes)
                .map_err(|e| format!("set {s} gemm {g} C bo failed: {e}"))?;
            c.as_mut_slice().fill(0xA5);
            c.flush();
            set.c.push(c);
        }
    }
    // Eager commands: the eager mode's per-GEMM command BOs and the retained modes' baseline step.
    let mut ecmds: Vec<Bo> = Vec::with_capacity(g_n);
    for g in 0..g_n {
        ecmds.push(
            dev.cmd_bo()
                .map_err(|e| format!("eager cmd bo {g} failed: {e}"))?,
        );
    }
    // Retained pools. `Prepared` cannot swap its instruction BO, so with --tape two immutable pools exist: the first
    // step's (GEMM 0 on the full stream's BO) and the steady one (all lean); the tape's selection picks the active one.
    let build_pool = |first_full: bool| -> Result<(Vec<Prepared>, Option<Chain>), String> {
        let mut pool: Vec<Prepared> = Vec::with_capacity(g_n);
        for g in 0..g_n {
            let ib: &Bo = if first_full && g == 0 {
                warm_insts
            } else {
                &insts
            };
            pool.push(
                Prepared::new(&dev, ib, &sets[0].args(g))
                    .map_err(|e| format!("prepared {g} failed: {e}"))?,
            );
        }
        let ch = if mode == Mode::Chain {
            let refs: Vec<&Prepared> = pool.iter().collect();
            Some(Chain::new(&dev, &refs).map_err(|e| format!("chain failed: {e}"))?)
        } else {
            None
        };
        Ok((pool, ch))
    };
    let (mut prepared, mut chain) = if mode == Mode::Eager {
        (Vec::new(), None)
    } else {
        build_pool(cfg.tape)?
    };
    let mut spare = if cfg.tape {
        Some(build_pool(false)?)
    } else {
        None
    };

    // Tape: recorded right after the retained inventory exists and before any hardware context/baseline, so the
    // negatives submit no GEMM.
    let mut tr: Option<TapeRun> = None;
    if cfg.tape {
        let sites = prepared
            .first()
            .ok_or("tape needs a retained mode")?
            .patch_sites();
        let t = TapeRun::new(
            &d.pdi, &d.insts, insts_src, warm_insts, &insts, &sites, &sets[0], g_n,
        )?;
        println!("tape: recorded entries={g_n} sequence_hash={:#018x} programs={} (full + lean, cache misses={})", t.tape.sequence_hash, t.cache.len(), t.misses);
        let t = tr.insert(t);
        if cfg.neg == Neg::StaleKey {
            return Ok(t.neg_stale_key());
        }
    }
    let ctx = HwCtx::create(&dev, ntiles, 2048).map_err(|e| format!("create_hwctx failed: {e}"))?;
    ctx.config_cu(&pdi)
        .map_err(|e| format!("config_cu failed: {e}"))?;

    let eager_step = |set: &Set, ecmds: &mut [Bo], ib: &Bo| -> Result<(), String> {
        for (g, cmd) in ecmds.iter_mut().enumerate() {
            let seq = ctx
                .submit(cmd, ib, &set.args(g))
                .map_err(|e| format!("gemm {g}: submit error: {e}"))?;
            let st = ctx
                .wait(cmd, seq, timeout_ms)
                .map_err(|e| format!("gemm {g}: wait error: {e}"))?;
            if st != ERT_STATE_COMPLETED {
                return Err(format!("gemm {g}: state {st} != COMPLETED"));
            }
        }
        Ok(())
    };

    // ---- full-TXN eager baseline (retained modes, and eager with --lean): set 0, poisoned C, byte reference for the
    // first/last compare. With --lean it is also the one untimed full-TXN warmup per hwctx that lean TXNs require.
    let mut baseline: Option<Vec<Vec<u8>>> = None;
    if (mode != Mode::Eager || lean) && !cfg.noop {
        sets[0].poison();
        eager_step(&sets[0], &mut ecmds, warm_insts).map_err(|e| format!("eager baseline: {e}"))?;
        let v = verify(d, &sets[0], &want, None, false, false);
        println!(
            "eager baseline{} (set 0): exact {}/{}",
            if lean { " (full-TXN warmup)" } else { "" },
            v.exact,
            v.checked
        );
        if v.exact != v.checked {
            println!("eager baseline not exact, inexact gemms {:?}", v.bad);
            return Ok(false);
        }
        baseline = Some(sets[0].c.iter().map(|c| c.as_slice().to_vec()).collect());
        if mode != Mode::Eager {
            ecmds.clear();
        }
    }

    // ---- inventory: declared patch sites and the pre-step snapshots.
    let mut allowed_cmd = [false; 17];
    if let Some(p) = prepared.first() {
        for site in p.patch_sites() {
            assert!(
                site.bo == PatchBo::Cmd(0) && site.word < 17,
                "unexpected Prepared patch site {:?}",
                site
            );
            allowed_cmd[site.word] = true;
        }
    }
    let chain_sites: Vec<usize> = chain.as_ref().map_or(Vec::new(), |c| {
        c.patch_sites()
            .into_iter()
            .inspect(|s| assert!(s.bo == PatchBo::Chain))
            .map(|s| s.word)
            .collect()
    });
    let mut pre_cmd: Vec<[u32; 17]> = prepared.iter().map(Prepared::words).collect();
    let mut pre_chain: Vec<u32> = chain.as_ref().map_or(Vec::new(), Chain::words);
    let mut inv = Inventory::default();
    let declared_cmd: Vec<usize> = (0..allowed_cmd.len()).filter(|&w| allowed_cmd[w]).collect();

    // ---- the loop
    let mut t_us: Vec<f64> = Vec::new();
    let (mut exact_ok, mut exact_checked, mut stale_expected) = (0usize, 0usize, 0usize);
    let mut bytes_equal: Option<bool> = baseline.as_ref().map(|_| true);
    let mut fail = false;
    let loop_start = Instant::now();
    let mut s = 0usize;
    loop {
        let set_idx = s % cfg.sets;
        let is_final = match (cfg.steps, cfg.secs) {
            (Some(n), _) => s + 1 == n,
            (None, Some(secs)) => s >= 1 && loop_start.elapsed().as_secs_f64() >= secs,
            (None, None) => unreachable!(),
        };
        let neg_step = neg_patch && s == 1;
        // First/last step, or every step with --verify-every: poison C beforehand, compare after (both untimed).
        let check_step = (s == 0 || is_final || cfg.verify_every) && !cfg.noop;
        if check_step {
            sets[set_idx].poison();
        }

        let step_secs: f64;
        let res: Result<(), String> = 'step: {
            match mode {
                Mode::Eager => {
                    let t0 = Instant::now();
                    let r = eager_step(&sets[set_idx], &mut ecmds, &insts);
                    step_secs = t0.elapsed().as_secs_f64();
                    r
                }
                Mode::Prepared | Mode::Chain => {
                    // Tape: the cache lookups pick the program (instruction BO) of every entry; the pool using exactly
                    // those BOs becomes active (untimed swap).
                    if let Some(t) = tr.as_mut() {
                        if let Err(e) = t.select() {
                            step_secs = 0.0;
                            break 'step Err(e);
                        }
                        if !pool_fits(&prepared, &t.chosen) {
                            if let Some(sp) = spare.as_mut() {
                                std::mem::swap(&mut prepared, &mut sp.0);
                                std::mem::swap(&mut chain, &mut sp.1);
                            }
                            if !pool_fits(&prepared, &t.chosen) {
                                step_secs = 0.0;
                                break 'step Err(
                                    "tape: the selected programs match neither retained pool"
                                        .into(),
                                );
                            }
                        }
                    }
                    // Pre-step snapshots, BEFORE the timed region.
                    for (g, p) in prepared.iter().enumerate() {
                        pre_cmd[g] = p.words();
                    }
                    if let Some(c) = chain.as_ref() {
                        pre_chain = c.words();
                    }
                    let set = &sets[set_idx];
                    // t_step: the hot CPU work (patch + refresh) through the last wait return.
                    let t0 = Instant::now();
                    if let Some(t) = tr.as_mut() {
                        // Tape replay: current addresses into the bindings, replay check, then patch FROM the bindings.
                        for g in 0..g_n {
                            t.bind(g, &set.args(g));
                        }
                        if let Err(e) = t.tape.check_replay(t.hash) {
                            step_secs = t0.elapsed().as_secs_f64();
                            break 'step Err(e);
                        }
                        t.replay_checks += 1;
                        for (g, p) in prepared.iter_mut().enumerate() {
                            let args = set.args(g);
                            for &(kind, value) in &t.tape.entries[g].bindings {
                                let BindingKind::ArgAddress { arg } = kind else {
                                    step_secs = t0.elapsed().as_secs_f64();
                                    break 'step Err(format!(
                                        "tape entry {g}: unexpected binding {kind:?}"
                                    ));
                                };
                                let bo = args[arg as usize];
                                if bo.dev_addr() != value {
                                    step_secs = t0.elapsed().as_secs_f64();
                                    break 'step Err(format!("tape entry {g} arg {arg}: binding {value:#x} != BO address {:#x}", bo.dev_addr()));
                                }
                                p.patch_arg(arg as usize, bo);
                            }
                        }
                    } else {
                        for (g, p) in prepared.iter_mut().enumerate() {
                            for (i, bo) in set.args(g).into_iter().enumerate() {
                                if neg_step && g == 0 && i == 2 {
                                    continue;
                                }
                                p.patch_arg(i, bo);
                            }
                        }
                    }
                    if let Some(c) = chain.as_mut() {
                        let refs: Vec<&Prepared> = prepared.iter().collect();
                        c.refresh(&refs);
                    }
                    let waited: Result<u32, String> = match mode {
                        Mode::Prepared => (|| {
                            let last_cmd = prepared.last().ok_or("no commands")?;
                            let mut last = 0u64;
                            for (g, p) in prepared.iter().enumerate() {
                                last = ctx
                                    .submit_prepared(p)
                                    .map_err(|e| format!("gemm {g}: submit error: {e}"))?;
                            }
                            ctx.wait(&last_cmd.cmd, last, timeout_ms)
                                .map_err(|e| format!("wait error: {e}"))
                        })(),
                        _ => (|| {
                            let c = chain.as_ref().expect("chain mode has a chain");
                            let seq = ctx
                                .submit_chain(c)
                                .map_err(|e| format!("submit_chain error: {e}"))?;
                            ctx.wait_chain(c, seq, timeout_ms).map_err(|e| {
                                format!("wait_chain error: {e} (error_index {})", c.error_index())
                            })
                        })(),
                    };
                    step_secs = t0.elapsed().as_secs_f64();

                    // Everything below is outside the timed region.
                    let st = match waited {
                        Ok(st) => st,
                        Err(e) => break 'step Err(e),
                    };
                    if st != ERT_STATE_COMPLETED {
                        break 'step Err(match chain.as_ref() {
                            Some(c) => format!(
                                "chain state {st} != COMPLETED, error_index {}",
                                c.error_index()
                            ),
                            None => format!("last command state {st} != COMPLETED"),
                        });
                    }
                    inv.steps += 1;
                    let mut fresh = [0u32; 17];
                    let mut hdr_bad: Vec<usize> = Vec::new();
                    for (g, p) in prepared.iter().enumerate() {
                        let post = p.words();
                        if mode == Mode::Prepared && post[0] & 15 != ERT_STATE_COMPLETED {
                            hdr_bad.push(g);
                        }
                        for w in diff_words(&pre_cmd[g], &post) {
                            inv.changed_words += 1;
                            if !allowed_cmd[w] {
                                inv.undeclared += 1;
                                println!("step {s} gemm {g}: UNDECLARED changed word {w} ({:#010x} -> {:#010x})", pre_cmd[g][w], post[w]);
                            }
                        }
                        let ibo: &Bo = tr.as_ref().map_or(&insts, |t| t.chosen[g]);
                        if tr.is_some() && p.inst_addr() != ibo.xdna {
                            inv.mismatched += 1;
                            println!("step {s} gemm {g}: selected program BO ({:#x}) is not the retained command's instruction BO ({:#x})", ibo.xdna, p.inst_addr());
                        }
                        encode_start_cu(&mut fresh, ibo, &set.args(g));
                        let bad = diff_words(&fresh, &post);
                        if neg_step && g == 0 {
                            if bad.is_empty() {
                                println!("NEG stale-patch: NOT detected by fresh-encode diff");
                            } else {
                                inv.neg_fresh_detected = true;
                                println!("NEG stale-patch: detected by fresh-encode diff (step {s} gemm 0 words {bad:?})");
                            }
                        } else if !bad.is_empty() {
                            inv.mismatched += bad.len();
                            println!("step {s} gemm {g}: patched words differ from fresh encoding at {bad:?}");
                        }
                        if let Some(t) = tr.as_mut() {
                            t.certify(s, g, &pre_cmd[g], &post, &fresh, &declared_cmd);
                        }
                    }
                    if let Some(c) = chain.as_ref() {
                        let post = c.words();
                        for w in diff_words(&pre_chain, &post) {
                            inv.changed_words += 1;
                            if !chain_sites.contains(&w) {
                                inv.undeclared += 1;
                                println!("step {s} chain: UNDECLARED changed word {w} ({:#010x} -> {:#010x})", pre_chain[w], post[w]);
                            }
                        }
                        let mut expect = vec![0u32; post.len()];
                        expect[0] = 1 | ((6 + 2 * g_n as u32) << 12) | (19 << 23) | (3 << 28);
                        expect[1] = g_n as u32;
                        for (g, p) in prepared.iter().enumerate() {
                            expect[7 + 2 * g] = p.cmd.handle;
                        }
                        let bad = diff_words(&expect, &post);
                        if !bad.is_empty() {
                            inv.mismatched += bad.len();
                            println!(
                                "step {s} chain: words differ from fresh chain layout at {bad:?}"
                            );
                        }
                    }
                    if insts.as_slice() != insts_src {
                        inv.undeclared += 1;
                        println!("step {s}: instruction BO changed");
                    }
                    if let (Some(_), Some(f)) = (tr.as_ref(), full_insts.as_ref()) {
                        if f.as_slice() != d.insts.as_slice() {
                            inv.undeclared += 1;
                            println!("step {s}: full instruction BO changed");
                        }
                    }
                    if hdr_bad.is_empty() {
                        Ok(())
                    } else {
                        Err(format!(
                            "command header state != COMPLETED for gemms {hdr_bad:?}"
                        ))
                    }
                }
            }
        };
        if let Err(e) = res {
            println!("step {s}: FAIL {e}");
            fail = true;
            break;
        }
        t_us.push(step_secs * 1e6);

        // Negative test output check: the buffer a correct patch would have filled must NOT be exact.
        if neg_step {
            let c0 = &sets[set_idx].c[0];
            c0.flush();
            if matches_reference(d, c0.as_slice(), &want[0]) {
                println!("NEG stale-patch: NOT detected by output check (set {set_idx} gemm 0 C is exact)");
            } else {
                inv.neg_output_detected = true;
                println!("NEG stale-patch: detected by output check");
            }
        }
        // Exactness (untimed): first/last step, every step with --verify-every.
        if check_step {
            // With --verify-every the stale-patch step is checked in full: its stale GEMM 0 must be inexact.
            let v = verify(
                d,
                &sets[set_idx],
                &want,
                baseline.as_deref(),
                neg_step && !cfg.verify_every,
                neg_step && cfg.verify_every,
            );
            exact_ok += v.exact;
            stale_expected += v.expected_bad;
            exact_checked += v.checked;
            if let (Some(all), Some(this)) = (bytes_equal.as_mut(), v.bytes_equal) {
                *all &= this;
            }
            if s == 0 || is_final || !v.bad.is_empty() || v.bytes_equal == Some(false) {
                println!(
                    "step {s}{} set {set_idx}: exact {}/{}{}",
                    if is_final { " (last)" } else { "" },
                    v.exact,
                    v.checked,
                    if v.bytes_equal == Some(false) {
                        " BYTES DIFFER from eager"
                    } else {
                        ""
                    }
                );
            }
            if !v.bad.is_empty() {
                println!("step {s}: inexact gemms {:?}", v.bad);
                fail = true;
            }
        }
        if fail || is_final {
            break;
        }
        s += 1;
    }
    let wall = loop_start.elapsed().as_secs_f64();

    // ---- report
    if !t_us.is_empty() {
        let steps = t_us.len();
        let sum_us: f64 = t_us.iter().sum();
        let mut sorted = t_us.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let mean = sum_us / steps as f64;
        let ops = g_n as f64 * useful as f64 * steps as f64;
        // --noop executes no GEMM: TOPS and exactness are not defined.
        let (busy_tops, wall_tops, exact) = if cfg.noop {
            ("n/a".to_string(), "n/a".to_string(), "n/a".to_string())
        } else {
            (
                format!("{:.4}", ops / (sum_us * 1e-6) / 1e12),
                format!("{:.4}", ops / wall / 1e12),
                format!("{exact_ok}/{exact_checked}"),
            )
        };
        println!(
            "railgun: mode={} shape={}x{}x{} gemms={g_n} steps={steps} txn_bytes={} step_us mean={:.1} p50={:.1} p99={:.1} min={:.1} per_gemm_us={:.1} busy_tops={busy_tops} wall_tops={wall_tops} exact={exact} bytes_equal_eager={}",
            mode.name(), cfg.m, cfg.n, cfg.k, insts_src.len(), mean, percentile(&sorted, 0.50), percentile(&sorted, 0.99), sorted[0],
            mean / g_n as f64,
            match bytes_equal { Some(true) => "yes", Some(false) => "no", None => "n.a." }
        );
    }
    let mut pass = !fail && exact_ok + stale_expected == exact_checked && !t_us.is_empty();
    if let Some(false) = bytes_equal {
        pass = false;
    }
    if mode == Mode::Eager {
        println!("inventory: n.a. (eager encodes every command from scratch)");
    } else {
        println!(
            "inventory: steps={} changed_words={} undeclared={} mismatched_vs_fresh={}",
            inv.steps, inv.changed_words, inv.undeclared, inv.mismatched
        );
        if inv.undeclared != 0 || inv.mismatched != 0 {
            pass = false;
        }
    }
    if let Some(t) = tr.as_ref() {
        println!(
            "tape: entries={g_n} sequence_hash={:#018x} programs={} hits={} misses={} full={} lean={} replay_checks={} fresh_equal={}",
            t.tape.sequence_hash, t.cache.len(), t.hits, t.misses, t.full, t.lean, t.replay_checks, t.fresh_equal
        );
        if t.bad != 0
            || t.fresh_equal != inv.steps * g_n
            || t.misses != 2
            || t.cache.len() != 2
            || t.full != 1
            || t.hits != t.replay_checks * g_n
            || t.full + t.lean != t.hits
        {
            println!("tape: FAILED (verify errors={} expected fresh_equal={} full=1 misses=2 programs=2)", t.bad, inv.steps * g_n);
            pass = false;
        }
    }
    if neg_patch {
        println!(
            "NEG stale-patch: fresh-encode diff {} output check {}",
            if inv.neg_fresh_detected {
                "detected"
            } else {
                "MISSED"
            },
            if inv.neg_output_detected {
                "detected"
            } else {
                "MISSED"
            }
        );
        pass &= inv.neg_fresh_detected && inv.neg_output_detected;
    }
    Ok(pass)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = match parse(&args) {
        Ok(Some(c)) => c,
        Ok(None) => {
            println!("{USAGE}");
            return;
        }
        Err(e) => {
            eprintln!("npu-railgun: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    let (m, n, k) = (cfg.m, cfg.n, cfg.k);
    let (variant, label) = (cfg.variant, cfg.variant.name());
    let d = match std::panic::catch_unwind(|| variant.design(m, n, k)) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("npu-railgun: design {label} rejected shape {m}x{n}x{k}");
            std::process::exit(2);
        }
    };
    // --lean: validated BEFORE any hardware is opened.
    let lean_bytes = if cfg.lean {
        match d.lean_insts() {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("npu-railgun: --lean rejected for shape {m}x{n}x{k}: {e}");
                std::process::exit(2);
            }
        }
    } else {
        None
    };
    if cfg.tape && lean_bytes.as_deref() == Some(d.insts.as_slice()) {
        eprintln!("npu-railgun: --tape needs distinct full and lean instruction streams for shape {m}x{n}x{k} (they are byte-identical)");
        std::process::exit(2);
    }
    let pass = match run(&cfg, &d, lean_bytes.as_deref()) {
        Ok(p) => p,
        Err(e) => {
            println!("{e}");
            false
        }
    };
    println!("RESULT: {}", if pass { "PASS" } else { "FAIL" });
    std::process::exit(if pass { 0 } else { 1 });
}
