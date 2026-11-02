// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! npu-experts: silicon harness for the grouped experts submission (`peacemaker_npu::kernels::experts::grouped_v9` for `--design V9`,
//! `peacemaker_npu::kernels::experts::grouped` over a one- or two-wave G80 design for `--design G80`).
//!
//! usage: npu-experts [--design V9|G80] --proj gate_up|down --m M --experts E [--ring] [--secs S | --rounds R]
//!                    [--timeout-ms T] [--verify-every] [--insts-only]
//!
//! `--design` defaults to V9 (M-wave 512, `M` in 1..=512). `--design G80` is `gate_up` only (N=1280, K=2560; `down` is
//! rejected at usage time, before any hardware access): the design is `design_g80(round_up(M, 256), ..)` (one 256-row
//! wave for `M` in 1..=256, two waves for 257..=512; Int8 shift 12, `Control::Fast`) and every expert's M is padded to
//! those rows by `experts::grouped` (`wave_m` = `waves() * wave_m()` of the design). Everything else (verification,
//! timing, ring, reuse) is shared with V9; the G80 `setup:` / `insts:` / `experts:` lines carry an extra ` design=G80`
//! field, the V9 lines are unchanged.
//!
//! One group = `E` experts of the same projection and the same `M` (1..=512), each an independent
//! `M x N x K` int8 GEMM with its OWN deterministic pseudo-random A and B (never a shared operand):
//! `gate_up` is N=1280, K=2560; `down` is N=2560, K=640. The whole group is ONE instruction stream (first expert's body
//! FULL, the others lean) in ONE retained instruction BO, driven by ONE retained `Prepared` START_CU command that is
//! resubmitted every round on ONE hardware context and ONE PDI. The operands are shmem arenas: expert `e` has its
//! packed A / B / C at 64-byte aligned offsets `e * stride` of args 0 / 1 / 2.
//!
//! * `--ring` wraps the entire group once in the persistent-ring protocol (`ring::RingLayout { nslots: 1 }`, slot 0):
//!   arg 3 is the shmem ring BO and the instruction stream polls ring slot 0 for `seq` before the first expert and
//!   writes the DONE line `seq` after the last one. The stream has exactly two patch sites (`PollSeq(0)` and
//!   `DoneSeq(0)`); round `r` uses `seq = 1 + r` (round 0 is built in, later rounds change only those two words with
//!   `ring::patch_seq`; the dirty cache lines are flushed). The CPU plays the producer and PRE-publishes the slot line
//!   (slot fields, flush, `seq` written LAST, flush, sfence) before the submit. After the wait the DONE line must carry
//!   `seq`, slot 0 and the 'DONE' magic; a missing or malformed DONE line fails the run.
//! * `--rounds R` runs exactly R timed rounds (default 20 when neither option is given); `--secs S` runs rounds until
//!   the loop wall clock passed S and then ONE more, final round (decided at its start so its C can be verified); at
//!   least 2 rounds. The two options are exclusive.
//! * `--timeout-ms T` bounds every wait and DONE spin (default 3000).
//! * `--insts-only` builds the designs and prints `insts:` sizes, then exits 0 WITHOUT opening the device.
//!
//! Sequence (all in one process, one device):
//! 1. Operands and packing: expert `e` seeds its own LCG; A/B are generated, packed with `GroupedDesign::pack_in(e, ..)`
//!    and copied straight into the arenas (threaded, disjoint ranges).
//! 2. Eager baseline (before any timed round): every expert runs as its OWN full-TXN eager submit (the same design with
//!    that one expert, same PDI/context; the same operand bytes copied into per-expert BOs). C is poisoned (0xA5) first and
//!    captured as the oracle. `eager_us_per_expert` = mean `submit + wait` over those E submits (poison/copy excluded).
//! 3. CPU reference: the exact `GroupedDesign::reference` (already row-parallel over all host threads, so experts are
//!    computed one after the other) of a deterministic sample that always contains expert 0 and the last expert (at
//!    least 4 distinct experts when E >= 4, all when E < 4: indices 0, E/3, 2E/3, E-1) runs first. Its measured time
//!    decides whether EVERY remaining expert also fits roughly 2 s (sample time + per-expert time x remaining <= 2 s;
//!    then all E experts are CPU-exact), otherwise only the sample is. The actual CPU time is printed. Each CPU
//!    reference must equal the unpacked eager oracle of the same expert, else the run fails before any timing.
//! 4. Timed rounds: before each round the whole C arena is poisoned and flushed; the timed region is only the hot path
//!    (ring: patch + publish; then `submit_prepared` + wait). Poisoning, DONE validation, the retained-BO audit and
//!    verification are outside it. The first round, the last round and (with `--verify-every`) every round compare EVERY
//!    expert's C byte for byte with its eager oracle, and unpack + compare the CPU-checked experts with their exact CPU
//!    reference. Any mismatch, bad command state or bad DONE line fails the run.
//!
//! Output: `setup:` line, one `round` line per round, and the report
//! `experts: proj=.. m=.. E=.. rounds=.. us_per_group=.. us_per_expert=.. useful_tops=.. exact=a/b
//! eager_us_per_expert=.. insts_bytes=.. ring=on|off`, where `exact=a/b` counts EVERY expert's group C that is byte-equal
//! to its eager oracle (b = verified rounds x E), then `check:` with the direct group-vs-CPU counts
//! (`cpu_direct_exact=a/b`, b = verified rounds x CPU-checked experts), the CPU coverage (`cpu_experts=k/E`) and the
//! CPU-vs-eager-oracle count, and `RESULT: PASS|FAIL`. `us_per_group` is the mean timed-region time per round,
//! `useful_tops` = `E * 2*M*N*K / us_per_group`. Exit status: 0 pass, 1 fail, 2 bad usage or a shape the design rejects.
use npu_tools::util::{lcg, parse_num, percentile, round_up, us};
use peacemaker_npu::kernels::experts::{grouped, grouped_v9, ExpertPlan, GroupedDesign};
use peacemaker_npu::kernels::gemm_array::V8_DEFAULT_EPILOGUE;
use peacemaker_npu::kernels::gemm_core::Control;
use peacemaker_npu::kernels::gemm_g80::design_g80;
use peacemaker_npu::kernels::ring::{patch_seq, PatchKind, RingLayout, DONE_MAGIC};
use railgun::npu::{clflush, Bo, Device, HwCtx, Prepared, ERT_STATE_COMPLETED};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const USAGE: &str = "usage: npu-experts [--design V9|G80] --proj gate_up|down --m M --experts E [--ring] [--secs S | --rounds R] [--timeout-ms T] [--verify-every] [--insts-only]";
const POISON: u8 = 0xA5;
/// `max_opc` for the hardware context (as the other array harnesses).
const MAX_OPC: u32 = 2048;
const DEFAULT_ROUNDS: usize = 20;
const DEFAULT_TIMEOUT_MS: u64 = 3000;
/// Threaded CPU budget under which every expert (not just the sample) is checked exactly.
const CPU_ALL_BUDGET_S: f64 = 2.0;
const ARENA_ALIGN: usize = 64;
/// Rows of one G80 M-wave (`gemm_g80::WAVE_M`). A G80 group is built on the design of at most `G80_WAVES_MAX` waves that
/// covers the largest expert M (`design_g80(round_up(max_m, 256), ..)`, 256 or 512 rows).
const G80_WAVE_M: usize = 256;
const G80_WAVES_MAX: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Proj {
    GateUp,
    Down,
}

impl Proj {
    fn name(self) -> &'static str {
        match self {
            Proj::GateUp => "gate_up",
            Proj::Down => "down",
        }
    }
    /// `(N, K)`.
    fn nk(self) -> (usize, usize) {
        match self {
            Proj::GateUp => (1280, 2560),
            Proj::Down => (2560, 640),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Design {
    V9,
    G80,
}

impl Design {
    /// Largest `--m` (the design's M-wave).
    fn max_m(self) -> usize {
        match self {
            Design::V9 => 512,
            Design::G80 => G80_WAVE_M * G80_WAVES_MAX,
        }
    }
    /// Output field appended to the G80 report lines; empty for V9 so its lines stay byte-identical.
    fn tag(self) -> &'static str {
        match self {
            Design::V9 => "",
            Design::G80 => " design=G80",
        }
    }
    fn name(self) -> &'static str {
        match self {
            Design::V9 => "V9",
            Design::G80 => "G80",
        }
    }
    fn builder(self) -> &'static str {
        match self {
            Design::V9 => "grouped_v9",
            Design::G80 => "grouped G80",
        }
    }
    /// `experts` as ONE group of this design; panics on a shape the design rejects.
    fn group(
        self,
        n: usize,
        k: usize,
        experts: &[ExpertPlan],
        ring: Option<(RingLayout, usize, u32)>,
    ) -> GroupedDesign {
        match self {
            Design::V9 => grouped_v9(n, k, experts, ring),
            Design::G80 => {
                let rows = round_up(experts.iter().map(|x| x.m).max().unwrap_or(1), G80_WAVE_M);
                let d = design_g80(rows, n, k, V8_DEFAULT_EPILOGUE, Control::Fast);
                let wave_m = d.waves() * d.wave_m();
                grouped(d, wave_m, n, k, experts, ring)
            }
        }
    }
}

struct Cfg {
    design: Design,
    proj: Proj,
    m: usize,
    experts: usize,
    ring: bool,
    secs: Option<f64>,
    rounds: Option<usize>,
    timeout_ms: u64,
    verify_every: bool,
    insts_only: bool,
}

/// `Ok(None)` = help requested.
fn parse(args: &[String]) -> Result<Option<Cfg>, String> {
    let (mut design, mut proj, mut m, mut experts) = (Design::V9, None, None, None);
    let (mut ring, mut verify_every, mut insts_only) = (false, false, false);
    let (mut secs, mut rounds, mut timeout_ms) = (None, None, DEFAULT_TIMEOUT_MS);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match a.as_str() {
            "-h" | "--help" => return Ok(None),
            "--design" => {
                design = match val("--design")?.as_str() {
                    "V9" => Design::V9,
                    "G80" => Design::G80,
                    o => return Err(format!("--design expects V9|G80, got {o:?}")),
                }
            }
            "--proj" => {
                proj = Some(match val("--proj")?.as_str() {
                    "gate_up" => Proj::GateUp,
                    "down" => Proj::Down,
                    o => return Err(format!("--proj expects gate_up|down, got {o:?}")),
                })
            }
            "--m" => m = Some(parse_num::<usize>("--m", &val("--m")?)?),
            "--experts" => experts = Some(parse_num::<usize>("--experts", &val("--experts")?)?),
            "--ring" => ring = true,
            "--verify-every" => verify_every = true,
            "--insts-only" => insts_only = true,
            "--secs" => {
                let s: f64 = parse_num("--secs", &val("--secs")?)?;
                if !s.is_finite() || s <= 0.0 {
                    return Err("--secs must be a positive number".into());
                }
                secs = Some(s);
            }
            "--rounds" => {
                let r: usize = parse_num("--rounds", &val("--rounds")?)?;
                if r == 0 {
                    return Err("--rounds must be >= 1".into());
                }
                rounds = Some(r);
            }
            "--timeout-ms" => {
                timeout_ms = parse_num("--timeout-ms", &val("--timeout-ms")?)?;
                if timeout_ms == 0 {
                    return Err("--timeout-ms must be >= 1".into());
                }
            }
            o => return Err(format!("unknown argument {o:?}")),
        }
    }
    if secs.is_some() && rounds.is_some() {
        return Err("--secs and --rounds are exclusive".into());
    }
    if insts_only && (secs.is_some() || rounds.is_some() || verify_every) {
        return Err("--insts-only takes no --secs/--rounds/--verify-every".into());
    }
    let proj = proj.ok_or("--proj is required")?;
    let m = m.ok_or("--m is required")?;
    let experts = experts.ok_or("--experts is required")?;
    if design == Design::G80 && proj != Proj::GateUp {
        return Err("--design G80 supports only --proj gate_up".into());
    }
    if !(1..=design.max_m()).contains(&m) {
        return Err(format!(
            "--m must be 1..={} for --design {}, got {m}",
            design.max_m(),
            design.name()
        ));
    }
    if experts == 0 {
        return Err("--experts must be >= 1".into());
    }
    Ok(Some(Cfg {
        design,
        proj,
        m,
        experts,
        ring,
        secs,
        rounds,
        timeout_ms,
        verify_every,
        insts_only,
    }))
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Nearest-rank percentile of an ascending sample.

fn fmt(x: f64, prec: usize) -> String {
    if x.is_nan() {
        "n/a".to_string()
    } else {
        format!("{x:.prec$}")
    }
}

/// Deterministic int8 generator (the LCG of the other harnesses).

/// Expert `e`'s own operands: distinct deterministic random A (m*k) and B (k*n); never shared between experts.
fn gen_ab(e: usize, m: usize, n: usize, k: usize) -> (Vec<i8>, Vec<i8>) {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ (e as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
    for _ in 0..4 {
        lcg(&mut seed);
    }
    let a: Vec<i8> = (0..m * k).map(|_| lcg(&mut seed)).collect();
    let b: Vec<i8> = (0..k * n).map(|_| lcg(&mut seed)).collect();
    (a, b)
}

/// `f(0..n)` on up to `threads` scoped threads (dynamic index queue); results in index order.
fn par_map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, threads: usize, f: F) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    if n == 0 {
        return Vec::new();
    }
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads.clamp(1, n))
            .map(|_| {
                s.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        local.push((i, f(i)));
                    }
                    local
                })
            })
            .collect();
        for h in handles {
            for (i, t) in h.join().expect("worker thread panicked") {
                out[i] = Some(t);
            }
        }
    });
    out.into_iter()
        .map(|o| o.expect("every index computed"))
        .collect()
}

/// Deterministic CPU sample: all experts when E < 4, else the distinct indices {0, E/3, 2E/3, E-1} (E >= 4 gives
/// exactly four distinct values).
fn sample_indices(e: usize) -> Vec<usize> {
    let mut v: Vec<usize> = if e < 4 {
        (0..e).collect()
    } else {
        vec![0, e / 3, 2 * e / 3, e - 1]
    };
    v.sort_unstable();
    v.dedup();
    v
}

// ---- ring access (non-coherent CPU view: flush before reading device-written lines, after writing) ----

fn rd32(bo: &Bo, off: usize) -> u32 {
    assert!(off + 4 <= bo.len);
    unsafe { core::ptr::read_volatile(bo.host.add(off) as *const u32) }
}
fn wr32(bo: &Bo, off: usize, v: u32) {
    assert!(off + 4 <= bo.len);
    unsafe { core::ptr::write_volatile(bo.host.add(off) as *mut u32, v) }
}
fn wr64(bo: &Bo, off: usize, v: u64) {
    assert!(off + 8 <= bo.len);
    unsafe { core::ptr::write_volatile(bo.host.add(off) as *mut u64, v) }
}

struct Ring {
    layout: RingLayout,
    slot: usize,
    bo: Bo,
    /// Last seq published into the slot (0 = never).
    prev: u32,
}

impl Ring {
    fn done(&self) -> (u32, u32, u32) {
        let off = self.layout.done_line(self.slot);
        unsafe { clflush(self.bo.host.add(off), 64) };
        (
            rd32(&self.bo, off),
            rd32(&self.bo, off + 4),
            rd32(&self.bo, off + 8),
        )
    }

    /// The slot may take a new seq only when the previous one has its DONE line (or it was never used).
    fn may_publish(&self) -> bool {
        self.done().0 == self.prev
    }

    /// Slot fields, flush, `seq` LAST, flush, sfence. The fields besides `seq` are informational (the NPU reads `seq`).
    fn publish(&mut self, seq: u32, m: u32, offs: (u64, u64, u64)) {
        let base = self.layout.slot_line(self.slot);
        wr32(&self.bo, base + 4, 0);
        wr32(&self.bo, base + 8, m);
        wr32(&self.bo, base + 12, 0);
        wr64(&self.bo, base + 16, offs.0);
        wr64(&self.bo, base + 24, offs.1);
        wr64(&self.bo, base + 32, offs.2);
        unsafe { clflush(self.bo.host.add(base), 64) };
        core::sync::atomic::compiler_fence(Ordering::SeqCst);
        wr32(&self.bo, base, seq);
        unsafe {
            clflush(self.bo.host.add(base), 64);
            core::arch::x86_64::_mm_sfence();
        }
        self.prev = seq;
    }

    /// The DONE line of the slot must carry `seq`, the slot index and the DONE magic.
    fn check_done(&self, seq: u32, timeout_ms: u64) -> Result<(), String> {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            let (s, slot, magic) = self.done();
            if s == seq {
                if slot != self.slot as u32 || magic != DONE_MAGIC {
                    return Err(format!(
                        "malformed DONE line: seq {s:#x} slot field {slot} magic {magic:#x}"
                    ));
                }
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "DONE line seq {s:#x}, want {seq:#x} (command completed without acknowledging)"
                ));
            }
            core::hint::spin_loop();
        }
    }
}

/// Hot path: rewrite the declared seq words of the retained instruction BO (host image + device words) and flush
/// exactly their cache lines. `sites` must be the two groups of `PollSeq(0)` / `DoneSeq(0)`.
fn patch_hot(bo: &Bo, host: &mut [u8], sites: &[(usize, PatchKind)], seq: u32) {
    patch_seq(host, sites, seq);
    for &(w, _) in sites {
        let v = u32::from_le_bytes(host[4 * w..4 * w + 4].try_into().unwrap());
        unsafe { core::ptr::write_volatile((bo.host as *mut u32).add(w), v) };
    }
    for &(w, _) in sites {
        unsafe { clflush(bo.host.add(4 * w), 4) };
    }
}

/// Audit (untimed): the retained BO equals the host image, and the host image differs from the freshly built stream
/// only at the declared patch words.
fn audit_insts(
    bo: &Bo,
    host: &[u8],
    built: &[u8],
    sites: &[(usize, PatchKind)],
) -> Result<(), String> {
    unsafe { clflush(bo.host, host.len()) };
    if bo.as_slice()[..host.len()] != *host {
        return Err("retained insts BO differs from the patched host image".into());
    }
    for (w, (a, b)) in host.chunks_exact(4).zip(built.chunks_exact(4)).enumerate() {
        if a != b && !sites.iter().any(|s| s.0 == w) {
            return Err(format!(
                "insts word {w} changed but is not a declared patch site"
            ));
        }
    }
    Ok(())
}

#[derive(Default)]
struct Counts {
    exact_ok: usize,
    exact_total: usize,
    bytes_ok: usize,
    bytes_total: usize,
}

/// Compare every expert's C of the group arena with its eager oracle (bytes), and the CPU-checked experts also with
/// their exact CPU reference (unpacked). Returns the first few failure descriptions.
fn verify_round(
    g: &GroupedDesign,
    plans: &[ExpertPlan],
    c: &Bo,
    oracle: &[Vec<u8>],
    refs: &[Option<Vec<i32>>],
    cnt: &mut Counts,
) -> Vec<String> {
    let mut bad = Vec::new();
    c.flush();
    let all = c.as_slice();
    for (e, p) in plans.iter().enumerate() {
        let len = g.c_bytes(e);
        let got = &all[p.c_off as usize..p.c_off as usize + len];
        cnt.bytes_total += 1;
        if got == oracle[e].as_slice() {
            cnt.bytes_ok += 1;
        } else if bad.len() < 5 {
            bad.push(format!("expert {e}: C differs from its eager oracle"));
        }
        if let Some(want) = &refs[e] {
            cnt.exact_total += 1;
            if g.unpack_out(e, got) == *want {
                cnt.exact_ok += 1;
            } else if bad.len() < 5 {
                bad.push(format!(
                    "expert {e}: C differs from the exact CPU reference"
                ));
            }
        }
    }
    bad
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
            eprintln!("npu-experts: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    let (n, k) = cfg.proj.nk();
    fn reject(cfg: &Cfg, what: &str) -> ! {
        eprintln!(
            "npu-experts: {} rejected {} m={} E={}{}: {what}",
            cfg.design.builder(),
            cfg.proj.name(),
            cfg.m,
            cfg.experts,
            if cfg.ring { " --ring" } else { "" }
        );
        std::process::exit(2)
    }
    // One-expert design: per-expert byte sizes (-> arena strides) and the eager full-TXN baseline (same PDI).
    let single = match catch_unwind(AssertUnwindSafe(|| {
        cfg.design.group(
            n,
            k,
            &[ExpertPlan {
                m: cfg.m,
                a_off: 0,
                b_off: 0,
                c_off: 0,
            }],
            None,
        )
    })) {
        Ok(d) => d,
        Err(_) => reject(&cfg, "single-expert baseline design"),
    };
    let strides = [
        round_up(single.a_bytes(0), ARENA_ALIGN),
        round_up(single.b_bytes(0), ARENA_ALIGN),
        round_up(single.c_bytes(0), ARENA_ALIGN),
    ];
    let plans: Vec<ExpertPlan> = (0..cfg.experts)
        .map(|e| ExpertPlan {
            m: cfg.m,
            a_off: (e * strides[0]) as u64,
            b_off: (e * strides[1]) as u64,
            c_off: (e * strides[2]) as u64,
        })
        .collect();
    let ring = cfg.ring.then_some((RingLayout { nslots: 1 }, 0usize, 1u32));
    let g = match catch_unwind(AssertUnwindSafe(|| cfg.design.group(n, k, &plans, ring))) {
        Ok(d) => d,
        Err(_) => reject(&cfg, "group design"),
    };
    if cfg.insts_only {
        println!(
            "insts: proj={} m={} E={} ring={} insts_bytes={} pdi_bytes={} patch_sites={} a_bytes={} b_bytes={} c_bytes={}{}",
            cfg.proj.name(),
            cfg.m,
            cfg.experts,
            if cfg.ring { "on" } else { "off" },
            g.insts.len(),
            g.pdi.len(),
            g.patch_sites.len(),
            g.a_bytes(0),
            g.b_bytes(0),
            g.c_bytes(0),
            cfg.design.tag()
        );
        return;
    }
    match run(&cfg, &g, &single, &plans, strides) {
        Ok(pass) => {
            println!("RESULT: {}", if pass { "PASS" } else { "FAIL" });
            std::process::exit(if pass { 0 } else { 1 });
        }
        Err(e) => {
            println!("FAIL: {e}");
            println!("RESULT: FAIL");
            std::process::exit(1);
        }
    }
}

fn run(
    cfg: &Cfg,
    g: &GroupedDesign,
    single: &GroupedDesign,
    plans: &[ExpertPlan],
    strides: [usize; 3],
) -> Result<bool, String> {
    let (n, k) = cfg.proj.nk();
    let (m, ne) = (cfg.m, cfg.experts);
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    if g.pdi != single.pdi {
        return Err("group PDI differs from the single-expert PDI: eager baseline and group cannot share one context".into());
    }
    let expect_args = if cfg.ring { 4 } else { 3 };
    if g.args.len() != expect_args {
        return Err(format!(
            "group design has {} args, expected {expect_args}",
            g.args.len()
        ));
    }
    if single.args.len() != 3 {
        return Err(format!(
            "single-expert design has {} args, expected 3",
            single.args.len()
        ));
    }
    for e in 0..ne {
        if g.a_bytes(e) != single.a_bytes(0)
            || g.b_bytes(e) != single.b_bytes(0)
            || g.c_bytes(e) != single.c_bytes(0)
        {
            return Err(format!(
                "expert {e} packed sizes differ from the single-expert design"
            ));
        }
    }
    if cfg.ring {
        let ok = g.patch_sites.len() == 2
            && g.patch_sites.iter().any(|s| s.1 == PatchKind::PollSeq(0))
            && g.patch_sites.iter().any(|s| s.1 == PatchKind::DoneSeq(0));
        if !ok {
            return Err(format!(
                "ring group has patch sites {:?}, expected exactly PollSeq(0) and DoneSeq(0)",
                g.patch_sites
            ));
        }
    } else if !g.patch_sites.is_empty() {
        return Err(format!(
            "non-ring group unexpectedly declares {} patch sites",
            g.patch_sites.len()
        ));
    }
    // Arena sizes: every expert's range, at least the design's declared argument bytes, page rounded.
    let need = [
        (ne - 1) * strides[0] + g.a_bytes(ne - 1),
        (ne - 1) * strides[1] + g.b_bytes(ne - 1),
        (ne - 1) * strides[2] + g.c_bytes(ne - 1),
    ];
    let sizes: Vec<usize> = (0..3)
        .map(|i| round_up(g.args[i].bytes.max(need[i]).max(4096), 4096))
        .collect();
    println!(
        "setup: proj={} shape={m}x{n}x{k} E={ne} ring={} insts_bytes={} pdi_bytes={} patch_sites={} per_expert packed A {} B {} C {} arenas A {} B {} C {} cpu_threads={threads}{}",
        cfg.proj.name(),
        if cfg.ring { "on" } else { "off" },
        g.insts.len(),
        g.pdi.len(),
        g.patch_sites.len(),
        g.a_bytes(0),
        g.b_bytes(0),
        g.c_bytes(0),
        sizes[0],
        sizes[1],
        sizes[2],
        cfg.design.tag()
    );

    let mut dev = Device::open().map_err(|e| format!("device open failed: {e}"))?;
    dev.map_heap(64 << 20)
        .map_err(|e| format!("device heap map failed: {e}"))?;
    let ntiles = dev.meta.cols as u32 * dev.meta.core.0 as u32;
    let ctx = HwCtx::create(&dev, ntiles, MAX_OPC)?;
    let pdi = dev.dev_bo(&g.pdi)?;
    ctx.config_cu(&pdi)?;

    let a = dev.shmem_bo(sizes[0])?;
    let b = dev.shmem_bo(sizes[1])?;
    let mut c = dev.shmem_bo(sizes[2])?;
    // 1. Operands: expert e's own random A/B, packed and copied straight into its arena range (disjoint ranges).
    {
        let (a_base, b_base) = (a.host as usize, b.host as usize);
        par_map(ne, threads, |e| {
            let (av, bv) = gen_ab(e, m, n, k);
            let [pa, pb] = g.pack_in(e, &av, &bv);
            assert_eq!(
                (pa.len(), pb.len()),
                (g.a_bytes(e), g.b_bytes(e)),
                "packed operand sizes of expert {e}"
            );
            unsafe {
                core::ptr::copy_nonoverlapping(
                    pa.as_ptr(),
                    (a_base + plans[e].a_off as usize) as *mut u8,
                    pa.len(),
                );
                core::ptr::copy_nonoverlapping(
                    pb.as_ptr(),
                    (b_base + plans[e].b_off as usize) as *mut u8,
                    pb.len(),
                );
            }
        });
    }
    a.flush();
    b.flush();
    c.as_mut_slice().fill(POISON);
    c.flush();

    // 2. Eager baseline: one full-TXN submit per expert on the SAME context; the oracle C of every expert.
    let insts_eager = dev.dev_bo(&single.insts)?;
    let mut cmd_eager = dev.cmd_bo()?;
    let mut ea = dev.shmem_bo(round_up(single.args[0].bytes.max(4096), 4096))?;
    let mut eb = dev.shmem_bo(round_up(single.args[1].bytes.max(4096), 4096))?;
    let mut ec = dev.shmem_bo(round_up(single.args[2].bytes.max(4096), 4096))?;
    let mut oracle: Vec<Vec<u8>> = Vec::with_capacity(ne);
    let mut eager_us = Vec::with_capacity(ne);
    for (e, p) in plans.iter().enumerate() {
        let (alen, blen, clen) = (g.a_bytes(e), g.b_bytes(e), g.c_bytes(e));
        ea.as_mut_slice()[..alen]
            .copy_from_slice(&a.as_slice()[p.a_off as usize..p.a_off as usize + alen]);
        eb.as_mut_slice()[..blen]
            .copy_from_slice(&b.as_slice()[p.b_off as usize..p.b_off as usize + blen]);
        ea.flush();
        eb.flush();
        ec.as_mut_slice().fill(POISON);
        ec.flush();
        let t0 = Instant::now();
        let seq = ctx.submit(&mut cmd_eager, &insts_eager, &[&ea, &eb, &ec])?;
        let state = ctx.wait(&cmd_eager, seq, cfg.timeout_ms)?;
        let dt = Instant::now() - t0;
        if state != ERT_STATE_COMPLETED {
            return Err(format!(
                "eager submit of expert {e}: state {state} (want {ERT_STATE_COMPLETED})"
            ));
        }
        eager_us.push(us(dt));
        ec.flush();
        oracle.push(ec.as_slice()[..clen].to_vec());
    }
    let eager_per_expert = mean(&eager_us);
    println!(
        "eager: {ne} full-TXN submits, eager_us_per_expert={}",
        fmt(eager_per_expert, 2)
    );

    // 3. CPU reference. `GroupedDesign::reference` is already row-parallel over all host threads, so the experts are
    //    computed one after the other (no nested threading). The sample runs first and its measured time decides
    //    whether EVERY remaining expert also fits the ~2 s budget: sample time + per-expert time x remaining <= 2 s.
    let cpu_ref = |e: usize| {
        let (av, bv) = gen_ab(e, m, n, k);
        g.reference(e, &av, &bv)
    };
    let sample = sample_indices(ne);
    let mut refs: Vec<Option<Vec<i32>>> = (0..ne).map(|_| None).collect();
    let t_cpu = Instant::now();
    for &e in &sample {
        refs[e] = Some(cpu_ref(e));
    }
    let t_sample = t_cpu.elapsed().as_secs_f64();
    let per_expert_cpu = t_sample / sample.len() as f64;
    let rest: Vec<usize> = (0..ne).filter(|e| refs[*e].is_none()).collect();
    let est_total = t_sample + per_expert_cpu * rest.len() as f64;
    let all_cpu = !rest.is_empty() && est_total <= CPU_ALL_BUDGET_S;
    if all_cpu {
        for &e in &rest {
            refs[e] = Some(cpu_ref(e));
        }
    }
    let cpu_elapsed = t_cpu.elapsed().as_secs_f64();
    let cpu_experts = refs.iter().filter(|r| r.is_some()).count();
    println!(
        "cpu: sample {:?} took {t_sample:.2} s ({per_expert_cpu:.3} s/expert, each reference threaded over {threads} host threads); estimated all-expert total {est_total:.2} s -> {}; cpu_experts={cpu_experts}/{ne}; actual cpu_elapsed_s={cpu_elapsed:.2}",
        sample,
        if rest.is_empty() { "no remaining experts" } else if all_cpu { "all experts CPU-exact" } else { "sample only (estimate over the ~2 s budget)" }
    );
    // The eager oracle must itself be exact vs the CPU, else the group comparison is meaningless.
    let (mut oracle_ok, mut oracle_total) = (0usize, 0usize);
    for (e, r) in refs.iter().enumerate() {
        if let Some(want) = r {
            oracle_total += 1;
            oracle_ok += (g.unpack_out(e, &oracle[e]) == *want) as usize;
        }
    }
    println!("oracle: eager C vs exact CPU reference {oracle_ok}/{oracle_total}");
    if oracle_ok != oracle_total {
        println!(
            "check: cpu_experts={cpu_experts}/{ne} oracle_cpu_exact={oracle_ok}/{oracle_total}"
        );
        return Err("eager oracle is not CPU-exact; the group comparison is meaningless".into());
    }

    // 4. Timed rounds on the one retained instruction BO and the one retained Prepared command.
    let mut ring = if cfg.ring {
        let layout = RingLayout { nslots: 1 };
        let mut bo = dev.shmem_bo(round_up(layout.bytes().max(4096), 4096))?;
        layout.initialize(&mut bo.as_mut_slice()[..layout.bytes()]);
        bo.flush();
        Some(Ring {
            layout,
            slot: 0,
            bo,
            prev: 0,
        })
    } else {
        None
    };
    let insts = dev.dev_bo(&g.insts)?;
    let mut insts_host = g.insts.clone();
    let prepared = {
        let mut refs_bo: Vec<&Bo> = vec![&a, &b, &c];
        if let Some(r) = &ring {
            refs_bo.push(&r.bo);
        }
        Prepared::new(&dev, &insts, &refs_bo)?
    };
    let useful_ops = plans.len() as f64 * 2.0 * m as f64 * n as f64 * k as f64;
    let mut times: Vec<f64> = Vec::new();
    let mut cnt = Counts::default();
    let mut ok = true;
    let mut verified_rounds = 0usize;
    let start = Instant::now();
    let mut r = 0usize;
    loop {
        let final_round = match (cfg.rounds, cfg.secs) {
            (Some(total), _) => r + 1 == total,
            (None, Some(s)) => r >= 1 && start.elapsed().as_secs_f64() >= s,
            (None, None) => r + 1 == DEFAULT_ROUNDS,
        };
        let verify = r == 0 || final_round || cfg.verify_every;
        let seq = 1 + r as u32;
        // Untimed: poison the whole C arena; ring: the producer guard.
        c.as_mut_slice().fill(POISON);
        c.flush();
        if let Some(rg) = &ring {
            if !rg.may_publish() {
                return Err(format!("round {r}: host guard refused publish of seq {seq:#x}: previous seq {:#x} has no DONE line", rg.prev));
            }
        }
        // Timed: hot path only.
        let t0 = Instant::now();
        if let Some(rg) = ring.as_mut() {
            if r > 0 {
                patch_hot(&insts, &mut insts_host, &g.patch_sites, seq);
            }
            rg.publish(
                seq,
                m as u32,
                (plans[0].a_off, plans[0].b_off, plans[0].c_off),
            );
        }
        let sq = ctx.submit_prepared(&prepared)?;
        let waited = ctx.wait(&prepared.cmd, sq, cfg.timeout_ms);
        let dt = us(t0.elapsed());
        let state = waited.map_err(|e| format!("round {r}: wait failed: {e}"))?;
        if state != ERT_STATE_COMPLETED {
            return Err(format!(
                "round {r}: state {state} (want {ERT_STATE_COMPLETED})"
            ));
        }
        times.push(dt);
        // Untimed: DONE validation, retained-BO audit, verification.
        if let Some(rg) = &ring {
            rg.check_done(seq, cfg.timeout_ms)
                .map_err(|e| format!("round {r}: {e}"))?;
            audit_insts(&insts, &insts_host, &g.insts, &g.patch_sites)
                .map_err(|e| format!("round {r}: {e}"))?;
        }
        let mut status = "unverified".to_string();
        if verify {
            verified_rounds += 1;
            let bad = verify_round(g, plans, &c, &oracle, &refs, &mut cnt);
            if bad.is_empty() {
                status = "exact".into();
            } else {
                ok = false;
                status = format!("MISMATCH ({})", bad.join("; "));
            }
        }
        println!(
            "round {r}: seq={}{} us={dt:.1} {status}",
            if cfg.ring {
                format!("{seq} ")
            } else {
                "n/a ".to_string()
            },
            if cfg.ring { "done=ok" } else { "" }
        );
        if !ok || final_round {
            break;
        }
        r += 1;
    }

    let rounds = times.len();
    let mean_group = mean(&times);
    let mut sorted = times.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let tops = useful_ops / (mean_group * 1e-6) / 1e12;
    println!(
        "rounds: n={rounds} p50_us={} min_us={} p99_us={} total_wall_s={:.2}",
        fmt(percentile(&sorted, 0.5), 1),
        fmt(sorted[0], 1),
        fmt(percentile(&sorted, 0.99), 1),
        start.elapsed().as_secs_f64()
    );
    println!(
        "experts: proj={} m={m} E={ne} rounds={rounds} us_per_group={} us_per_expert={} useful_tops={} exact={}/{} eager_us_per_expert={} insts_bytes={} ring={}{}",
        cfg.proj.name(),
        fmt(mean_group, 2),
        fmt(mean_group / ne as f64, 3),
        fmt(tops, 4),
        cnt.bytes_ok,
        cnt.bytes_total,
        fmt(eager_per_expert, 2),
        g.insts.len(),
        if cfg.ring { "on" } else { "off" },
        cfg.design.tag()
    );
    println!(
        "check: exact= above counts EVERY expert's group C byte-equal to its eager oracle (verified rounds x E, verified_rounds={verified_rounds}); cpu_direct_exact={}/{} (group-vs-CPU, verified rounds x CPU experts) cpu_experts={cpu_experts}/{ne} oracle_cpu_exact={oracle_ok}/{oracle_total} (eager oracle vs CPU)",
        cnt.exact_ok, cnt.exact_total
    );
    if cnt.exact_ok != cnt.exact_total || cnt.bytes_ok != cnt.bytes_total {
        ok = false;
    }
    Ok(ok)
}
