//! Wide expert-run tile byte oracle: PeaceMaker symmetric IU4 MoE GEMMs at
//! tile width NT4 vs NT8, A3B geometry (256 experts, top-8, hidden 2048,
//! intermediate 512). Lab only; no new kernel, no dispatch change.
//!
//! This module is a child of the `qwen4_moe_sym` example (`mod wide_tiles`) so
//! it reuses that example's private oracle machinery (`make_case`, `Case`,
//! `entry_gemm`, `cpu_group`, `symmetric_weights`, the entry tables) without a
//! copy. The `moe_iu4_wide_tiles_oracle` example is a small wrapper that
//! includes `qwen4_moe_sym.rs` by `#[path]` and calls [`run`].
//!
//!   moe_iu4_wide_tiles_oracle --plan          CPU: shapes, entries, routing, launch and allocation plan
//!   moe_iu4_wide_tiles_oracle --self-check    CPU: invariants of every geometry/routing/grouping/leader rule
//!   moe_iu4_wide_tiles_oracle                 GPU: one process, byte oracle + timing, <= 120 s
//!
//! GPU run, per residency x rows (512, 4096, 8192 tokens) x routing x input
//! (zero, random):
//!   production producers `moe_scatter_stable_top10`, `qwen4_moe_rotate256_i4`
//!   (gate A4) and `qwen4_moe_rotate128_i4` (down A4) feed the builder entries
//!   `..._pm_<arch>_nt4` and `..._nt8` through the example's `entry_gemm`.
//!   Chain A (NT4 gate/up -> rotate128 -> NT4 down) and chain B (NT8 gate/up ->
//!   rotate128 -> NT8 down) write poisoned output buffers; both whole buffers,
//!   padding, unused tiles, the guard tail and the poison included, are
//!   downloaded and compared byte for byte. NT8 down is also run on chain A's
//!   own down sidecar ("down_isolated"), so a down mismatch is attributable to
//!   the down entry rather than to gate/up. The first mismatching byte is
//!   reported with buffer, region, row, tile, expert, flat slot and column.
//!   Timing: >= 20 interleaved (ABBA-alternating) samples per variant and
//!   stage (gate/up, down), median microseconds from HIP events.
//!
//! Production selection (`--plan` and `--self-check` state and check it, no GPU): a
//! VRAM-resident run widens GATE/UP ONLY, to NT8, when `HIPFIRE_MOE_IU4_WIDE_TILES` resolves
//! on (unset follows its exact-gfx1201 qualification, explicit `1` opts in, `0` kills),
//! the arch is capable (NT8 admitted below) and real total slots >= 128 per table
//! expert; resident DOWN is always NT4. An explicit `0` disables resident widening;
//! host-mapped experts keep NT8 for both stages. Every plan cell and GPU cell carries
//! the stage-specific selection record. The GPU run is a diagnostic: it still compares
//! and times NT4 vs NT8 for BOTH stages, whatever production selects (owner measured
//! 12/12 cells byte-identical; NT8 wins resident gate/up and loses resident down).
//!
//! Admission is by raw bundle capability, not by model or card name: the
//! typed ISA target (`gpu.arch` parsed to `peacemaker_isa::Arch`) must plan the
//! A3B geometry with both widths, the example's entry table must name the
//! planned NT4/NT8 symbols, and the embedded builder bundle of that target
//! must contain all four symbols (NUL-terminated names in the code object)
//! and load them. Anything else is refused with the reasons, nonzero exit.
//!
//! Inputs, weights and sharing: symmetric QT44/QT53 synthetic experts with
//! the qwen4_moe_sym example's seeds, synthesized once (rayon, per-expert
//! seed) and uploaded once per residency; every shape's `Case` is built with
//! a one-expert placeholder and then lent the shared full tensors (swap, no
//! copy). `--distinct-experts N < 256` aliases expert `e` to synthetic expert
//! `e % N` through the pointer table; the setup record states it. X is one
//! random vector of the largest row count whose prefixes serve every shape.
//!
//! Deadline: wall clock from process start (`--deadline-s`, at most 120),
//! checked between every step and every timing sample. Exceeding it, or any
//! mismatch, exits 1. Output: one JSON object per line on stdout.

use super::*;
use peacemaker_isa::kernels::qwen4_moe_sym::{ExpertGeometry, Kind, WeightCodec, MAX_SLOTS};
use peacemaker_isa::Arch;
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

const GEOM: ExpertGeometry = ExpertGeometry {
    experts: 256,
    top_k: 8,
    hidden: 2048,
    intermediate: 512,
};
const ROWS: [usize; 3] = [512, 4096, 8192];
/// Largest row count the lab sizes its buffers for.
const MAX_ROWS: usize = 8192;
const MIN_TIMING_ITERS: usize = 20;
const WARMUP: usize = 2;
const MAX_DEADLINE_S: u64 = 120;
/// Routing `skew_empty`: experts `0..HOT` take every token, one slot in 64
/// tokens goes to a sparse expert (`SPARSE_BASE + (token / 64) % SPARSE_N`),
/// the rest are distinct uniform picks from `WARM`; the other experts are empty.
const HOT: usize = 3;
const WARM: std::ops::Range<usize> = 3..67;
const SPARSE_BASE: usize = 128;
const SPARSE_N: usize = 64;
/// Thresholds a `skew_empty` shape must meet to exercise the wide tiles.
const EMPTY_MIN: usize = 100;
const SINGLES_MIN: usize = 8;

#[derive(Debug)]
struct DeadlineExceeded(String);
impl std::fmt::Display for DeadlineExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for DeadlineExceeded {}

struct Deadline {
    t0: Instant,
    limit: Duration,
}
impl Deadline {
    fn new(limit_s: u64) -> Self {
        Deadline {
            t0: Instant::now(),
            limit: Duration::from_secs(limit_s),
        }
    }
    fn elapsed_s(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }
    fn check(&self, step: &str) -> Res<()> {
        if self.t0.elapsed() > self.limit {
            return Err(Box::new(DeadlineExceeded(format!(
                "wall-clock deadline {} s exceeded before step `{step}` (elapsed {:.1} s)",
                self.limit.as_secs(),
                self.elapsed_s()
            ))));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Input {
    Zero,
    Random,
}
impl Input {
    fn name(self) -> &'static str {
        match self {
            Input::Zero => "zero",
            Input::Random => "random",
        }
    }
    /// Distinct poison byte per input, memset into every output before the run.
    fn poison(self) -> i32 {
        match self {
            Input::Zero => 0x7B,
            Input::Random => 0xA5,
        }
    }
}

struct Opts {
    plan: bool,
    check: bool,
    rows: Vec<usize>,
    routings: Vec<String>,
    inputs: Vec<Input>,
    iters: usize,
    /// Residencies to run: `false` VRAM, `true` host-mapped expert weights.
    hosts: Vec<bool>,
    seed: u64,
    deadline_s: u64,
    distinct: usize,
}

const USAGE: &str = "usage: moe_iu4_wide_tiles_oracle [--plan | --self-check] [--rows 512,4096,8192] \
[--routing skew_empty[,all|skew]] [--inputs zero,random] [--iters N>=20] [--residency vram|host|both] \
[--seed N] [--deadline-s N<=120] [--distinct-experts N<=256]";

fn parse_opts(args: &[String]) -> Res<Opts> {
    let mut o = Opts {
        plan: false,
        check: false,
        rows: ROWS.to_vec(),
        routings: vec!["skew_empty".into()],
        inputs: vec![Input::Zero, Input::Random],
        iters: MIN_TIMING_ITERS,
        hosts: vec![false, true],
        seed: 1,
        deadline_s: MAX_DEADLINE_S,
        distinct: GEOM.experts as usize,
    };
    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        let mut value = || -> Res<String> {
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("{a} needs a value; {USAGE}").into())
        };
        match a {
            "--plan" => o.plan = true,
            "--self-check" => o.check = true,
            "--rows" => {
                o.rows = value()?
                    .split(',')
                    .map(|s| s.parse::<usize>())
                    .collect::<Result<_, _>>()?
            }
            "--routing" => o.routings = value()?.split(',').map(str::to_owned).collect(),
            "--inputs" => {
                o.inputs = value()?
                    .split(',')
                    .map(|s| match s {
                        "zero" => Ok(Input::Zero),
                        "random" => Ok(Input::Random),
                        other => Err(format!("--inputs {other}: expected zero or random")),
                    })
                    .collect::<Result<_, _>>()?
            }
            "--iters" => o.iters = value()?.parse()?,
            "--residency" => {
                o.hosts = match value()?.as_str() {
                    "vram" => vec![false],
                    "host" => vec![true],
                    "both" => vec![false, true],
                    other => {
                        return Err(
                            format!("--residency {other}: expected vram, host or both").into()
                        )
                    }
                }
            }
            "--seed" => o.seed = value()?.parse()?,
            "--deadline-s" => o.deadline_s = value()?.parse()?,
            "--distinct-experts" => o.distinct = value()?.parse()?,
            "--help" | "-h" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}; {USAGE}").into()),
        }
        i += 1;
    }
    if o.plan && o.check {
        return Err("--plan and --self-check are exclusive".into());
    }
    if o.iters < MIN_TIMING_ITERS {
        return Err(format!("--iters {} < {MIN_TIMING_ITERS} timing samples", o.iters).into());
    }
    if o.deadline_s == 0 || o.deadline_s > MAX_DEADLINE_S {
        return Err(format!("--deadline-s must be 1..={MAX_DEADLINE_S}").into());
    }
    if o.distinct == 0 || o.distinct > GEOM.experts as usize {
        return Err(format!("--distinct-experts must be 1..={}", GEOM.experts).into());
    }
    if o.rows.is_empty() || o.routings.is_empty() || o.inputs.is_empty() {
        return Err("--rows, --routing and --inputs must not be empty".into());
    }
    for &r in &o.rows {
        if r == 0 || r > MAX_ROWS || r * GEOM.top_k as usize > MAX_SLOTS as usize {
            return Err(format!("--rows {r}: 1..={MAX_ROWS} tokens").into());
        }
    }
    for r in &o.routings {
        if !matches!(r.as_str(), "skew_empty" | "all" | "skew") {
            return Err(format!("--routing {r}: expected skew_empty, all or skew").into());
        }
    }
    Ok(o)
}

// ------------------------------------------------------------ host helpers

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha_hex(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

fn pmax_of(slots: usize) -> usize {
    (slots + geo().e * 15).div_ceil(16) * 16
}

fn route(kind: &str, tokens: usize, seed: u64) -> Res<Vec<i32>> {
    match kind {
        "skew_empty" => {
            let mut r = Rng(seed ^ 0x5eed_e0e0);
            let mut v = Vec::with_capacity(tokens * geo().topk);
            for tok in 0..tokens {
                let mut row: Vec<i32> = (0..HOT as i32).collect();
                if tok % 64 == 7 {
                    row.push((SPARSE_BASE + (tok / 64) % SPARSE_N) as i32);
                }
                while row.len() < geo().topk {
                    let e = (WARM.start + r.below(WARM.len() as u64) as usize) as i32;
                    if !row.contains(&e) {
                        row.push(e);
                    }
                }
                v.extend(row);
            }
            Ok(v)
        }
        "all" | "skew" => routing(kind, tokens, seed),
        other => Err(format!("routing {other}").into()),
    }
}

/// Documented leader rule of the expert-run entries (emitter docs, `nt`
/// module): a CTA `(x, tile)` leads when the tile's position inside its
/// expert's run, counted by a 32-lane walk back over `expert_tile_ids`, is a
/// multiple of `nt`; the leader folds `cnt` = up to `nt` consecutive tiles of
/// the same expert; tiles with an id outside `[0, experts)` return.
fn run_position_walk(tiles: &[i32], t: usize) -> usize {
    let id = tiles[t];
    let (mut pos, mut hi) = (0usize, t);
    while hi > 0 {
        let lo = hi.saturating_sub(32);
        let mut run = 0;
        for k in (lo..hi).rev() {
            if tiles[k] == id {
                run += 1;
            } else {
                break;
            }
        }
        pos += run;
        if run < hi - lo {
            break;
        }
        hi = lo;
    }
    pos
}

/// Leader `(tile, cnt)` list of width `nt`.
fn leaders(tiles: &[i32], nt: usize, experts: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for t in 0..tiles.len() {
        let id = tiles[t];
        if id < 0 || id as usize >= experts {
            continue;
        }
        if run_position_walk(tiles, t) % nt != 0 {
            continue;
        }
        let mut cnt = 1;
        while cnt < nt && t + cnt < tiles.len() && tiles[t + cnt] == id {
            cnt += 1;
        }
        out.push((t, cnt));
    }
    out
}

struct Model {
    rows: usize,
    routing: String,
    l: usize,
    pmax: usize,
    topk: Vec<i32>,
    g: Grouping,
    raw: Vec<usize>,
    /// (expert, tiles in its run) of every non-empty expert.
    runs: Vec<(usize, usize)>,
    leaders4: Vec<(usize, usize)>,
    leaders8: Vec<(usize, usize)>,
}

fn model(rows: usize, routing_kind: &str, seed: u64) -> Res<Model> {
    let topk = route(routing_kind, rows, seed)?;
    let l = topk.len();
    let pmax = pmax_of(l);
    let g = cpu_group(&topk, pmax);
    let mut raw = vec![0usize; geo().e];
    for &e in &topk {
        raw[e as usize] += 1;
    }
    let runs = (0..geo().e)
        .filter(|&e| g.counts[e] > 0)
        .map(|e| (e, g.counts[e] as usize / 16))
        .collect();
    let leaders4 = leaders(&g.tiles, 4, geo().e);
    let leaders8 = leaders(&g.tiles, 8, geo().e);
    Ok(Model {
        rows,
        routing: routing_kind.to_owned(),
        l,
        pmax,
        topk,
        g,
        raw,
        runs,
        leaders4,
        leaders8,
    })
}

impl Model {
    fn empty(&self) -> usize {
        self.raw.iter().filter(|&&n| n == 0).count()
    }
    fn singles(&self) -> usize {
        self.raw.iter().filter(|&&n| (1..=15).contains(&n)).count()
    }
    fn summary(&self) -> Value {
        let p = self.g.offsets[geo().e] as usize;
        json!({
            "rows":self.rows,"routing":self.routing,"slots_L":self.l,"padded_P":p,"pmax":self.pmax,
            "tiles_live":p / 16,"tiles_grid_y":self.pmax / 16,
            "active_experts":self.runs.len(),"empty_experts":self.empty(),
            "experts_with_1_to_15_slots":self.singles(),
            "max_expert_slots":self.raw.iter().max(),
            "longest_run_tiles":self.runs.iter().map(|r| r.1).max(),
            "runs_ge_8_tiles":self.runs.iter().filter(|r| r.1 >= 8).count(),
            "leaders_nt4":self.leaders4.len(),"leaders_nt8":self.leaders8.len(),
            "sha256_routing":sha_hex(&i32_bytes(&self.topk)),
        })
    }
}

/// Synthetic QT44/QT53 experts, one rayon task per expert, the seeds of the
/// `qwen4_moe_sym` example (`seed * 1000 + e`, down `+ 600`).
fn synth_weights(distinct: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let g = geo();
    let mut gu = vec![0u8; distinct * g.gu_stride];
    let mut dn = vec![0u8; distinct * g.dn_stride];
    gu.par_chunks_mut(g.gu_stride)
        .enumerate()
        .for_each(|(e, c)| {
            c.copy_from_slice(&symmetric_weights(
                g.gu_m,
                g.hid,
                GU_GB,
                seed * 1000 + e as u64,
            ))
        });
    dn.par_chunks_mut(g.dn_stride)
        .enumerate()
        .for_each(|(e, c)| {
            c.copy_from_slice(&symmetric_weights(
                g.dn_m,
                g.mi,
                DN_GB,
                seed * 1000 + 600 + e as u64,
            ))
        });
    (gu, dn)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

fn stat_json(v: &[f64]) -> Value {
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    json!({"samples":v.len(),"median_us":median(v),"min_us":min,"max_us":max})
}

/// Region and coordinates of byte `off` of a `[pmax rows][row_bytes]` buffer
/// followed by the guard tail.
fn locate_byte(
    off: usize,
    row_bytes: usize,
    live_rows: usize,
    pmax_rows: usize,
    g: &Grouping,
) -> Value {
    if off >= pmax_rows * row_bytes {
        return json!({"region":"guard","offset":off,"guard_byte":off - pmax_rows * row_bytes});
    }
    let row = off / row_bytes;
    let tile = row / 16;
    let region = if row >= live_rows {
        "unused_tile"
    } else if g.sorted[row] < 0 {
        "padding_slot"
    } else {
        "live_slot"
    };
    json!({"region":region,"offset":off,"row":row,"tile":tile,"tile_expert":g.tiles[tile],
        "lane":row % 16,"flat_slot":g.sorted[row],"byte_in_row":off % row_bytes,
        "column":(off % row_bytes) / 2})
}

/// Byte comparison of two whole output buffers: `(identical, record)`.
fn compare(
    name: &str,
    a: &[u8],
    b: &[u8],
    row_bytes: usize,
    live_rows: usize,
    pmax_rows: usize,
    g: &Grouping,
) -> (bool, Value) {
    let identical = a == b;
    let mut v = json!({"buffer":name,"bytes":a.len(),"byte_identical":identical});
    if !identical {
        let n = a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len());
        let first = a
            .iter()
            .zip(b)
            .position(|(x, y)| x != y)
            .unwrap_or(a.len().min(b.len()));
        let mut loc = locate_byte(first, row_bytes, live_rows, pmax_rows, g);
        loc["nt4_byte"] = json!(a.get(first));
        loc["nt8_byte"] = json!(b.get(first));
        v["mismatched_bytes"] = json!(n);
        v["bytes_nt8"] = json!(b.len());
        v["first_mismatch"] = loc;
    }
    (identical, v)
}

// -------------------------------------------------------------- admission

struct Admission {
    arch: String,
    nt4: Entry,
    nt8: Entry,
    bundle_sha256: String,
    bundle_bytes: usize,
    symbols: Vec<String>,
}

fn arch_table(arch: Arch) -> &'static [Entry] {
    match arch {
        Arch::Gfx1201 => &ENTRIES_GFX1201,
        Arch::Gfx1151 => &ENTRIES_GFX1151,
        _ => &[],
    }
}

fn arch_bundle(arch: Arch) -> Option<&'static [u8]> {
    match arch {
        Arch::Gfx1201 => Some(rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201),
        Arch::Gfx1151 => Some(rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1151),
        _ => None,
    }
}

fn has_symbol(image: &[u8], sym: &str) -> bool {
    let mut needle = sym.as_bytes().to_vec();
    needle.push(0);
    image.windows(needle.len()).any(|w| w == needle.as_slice())
}

/// Pure raw-capability admission of NT4 + NT8 for `arch_name` (no GPU).
fn admit(arch_name: &str) -> Result<Admission, Vec<String>> {
    let arch: Arch = arch_name.parse().map_err(|e: String| vec![e])?;
    let plan = GEOM
        .plan(arch, WeightCodec::Mq4Sym)
        .map_err(|e| vec![format!("plan: {e}")])?;
    let mut reasons: Vec<String> = Vec::new();
    let mut symbols: Vec<String> = Vec::new();
    let mut sym = |nt: u8, kind: Kind| match plan.spec(kind, nt) {
        Ok(s) => {
            symbols.push(s.symbol());
            Some(s.symbol())
        }
        Err(e) => {
            reasons.push(format!("NT{nt} {}: {e}", kind.tag()));
            None
        }
    };
    let n4 = (sym(4, Kind::GateUp), sym(4, Kind::Down));
    let n8 = (sym(8, Kind::GateUp), sym(8, Kind::Down));
    let table = arch_table(arch);
    let mut pick = |name: &str, pair: &(Option<String>, Option<String>)| {
        let found = table.iter().find(|e| e.name == name).copied();
        match (found, pair) {
            (Some(e), (Some(g), Some(d)))
                if e.pm && e.gate_up == g.as_str() && e.down == d.as_str() =>
            {
                Some(e)
            }
            (Some(_), (Some(_), Some(_))) => {
                reasons.push(format!(
                    "entry table {name} does not name the planned symbols"
                ));
                None
            }
            (None, _) => {
                reasons.push(format!("entry table has no {name}"));
                None
            }
            _ => None,
        }
    };
    let e4 = pick("pm_nt4", &n4);
    let e8 = pick("pm_nt8", &n8);
    let mut bundle_sha256 = String::new();
    let mut bundle_bytes = 0;
    match arch_bundle(arch) {
        None => reasons.push(format!("no embedded builder bundle for {}", arch.name())),
        Some(image) => {
            bundle_bytes = image.len();
            bundle_sha256 = sha_hex(image);
            if !(image.starts_with(b"__CLANG_OFFLOAD_BUNDLE__") || image.starts_with(b"\x7fELF")) {
                reasons.push("embedded bundle is neither an offload bundle nor an ELF".into());
            }
            for s in &symbols {
                if !has_symbol(image, s) {
                    reasons.push(format!("bundle does not export {s}"));
                }
            }
        }
    }
    match (e4, e8) {
        (Some(nt4), Some(nt8)) if reasons.is_empty() => Ok(Admission {
            arch: arch.name().to_string(),
            nt4,
            nt8,
            bundle_sha256,
            bundle_bytes,
            symbols,
        }),
        _ => Err(reasons),
    }
}

// ---------------------------------------------------- production selection

/// The flag the production selector reads (resolved: unset follows qualification, on at
/// exact gfx1201; an explicit `0` disables resident widening).
const WIDE_FLAG_ENV: &str = "HIPFIRE_MOE_IU4_WIDE_TILES";
/// Real routed slots per expert from which a resident run widens (gate/up only).
const WIDE_MIN_SLOTS_PER_EXPERT: u128 = 128;

/// Tile width per stage the production selector is expected to choose.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct StageNt {
    gate_up: u8,
    down: u8,
}

/// Expected production selection, written from the contract, not from the selector:
/// an incapable arch is NT4/NT4; host-mapped experts take NT8 for both stages; a
/// VRAM-resident run takes NT8 for gate/up only (`wide_flag` on and real total slots
/// >= 128 per table expert, exact 128-bit arithmetic) and NT4 for down always.
/// `total_slots` is the same for both stages: gate/up `x_src_rows * x_row_div` and
/// down flat slots are both `rows * top_k`.
fn expected_stage_nt(
    capable: bool,
    host_mapped: bool,
    wide_flag: bool,
    total_slots: usize,
    n_experts: usize,
) -> StageNt {
    if !capable {
        return StageNt {
            gate_up: 4,
            down: 4,
        };
    }
    if host_mapped {
        return StageNt {
            gate_up: 8,
            down: 8,
        };
    }
    let wide = wide_flag
        && n_experts > 0
        && (total_slots as u128) >= WIDE_MIN_SLOTS_PER_EXPERT * (n_experts as u128);
    StageNt {
        gate_up: if wide { 8 } else { 4 },
        down: 4,
    }
}

fn stage_symbol(arch: &str, kind: Kind, nt: u8) -> Option<String> {
    let arch: Arch = arch.parse().ok()?;
    let plan = GEOM.plan(arch, WeightCodec::Mq4Sym).ok()?;
    plan.spec(kind, nt).ok().map(|s| s.symbol())
}

fn stage_json(arch: &str, s: StageNt) -> Value {
    json!({
        "gate_up":{"nt":s.gate_up,"entry":format!("pm_nt{}", s.gate_up),
            "symbol":stage_symbol(arch, Kind::GateUp, s.gate_up)},
        "down":{"nt":s.down,"entry":format!("pm_nt{}", s.down),
            "symbol":stage_symbol(arch, Kind::Down, s.down)},
        "mixed":s.gate_up != s.down,
    })
}

/// Machine-readable production selection of one shape on `arch`: host-mapped, and
/// VRAM-resident with the flag resolved off (explicit `0`) and resolved on (unset on a
/// qualified arch, or explicit `1`).
fn selection_record(arch: &str, capable: bool, total_slots: usize, n_experts: usize) -> Value {
    let pick = |host, flag| {
        stage_json(
            arch,
            expected_stage_nt(capable, host, flag, total_slots, n_experts),
        )
    };
    json!({
        "arch":arch,"capable":capable,"flag":WIDE_FLAG_ENV,
        "total_slots":total_slots,"n_experts":n_experts,
        "slots_per_expert":total_slots as f64 / n_experts as f64,
        "wide_shape":n_experts > 0 && (total_slots as u128) >= WIDE_MIN_SLOTS_PER_EXPERT * (n_experts as u128),
        "host_mapped":pick(true, false),
        "resident_flag_off":pick(false, false),
        "resident_flag_on":pick(false, true),
    })
}

// ------------------------------------------------------------------- plan

fn plan_cell(m: &Model) -> Value {
    let (gu, dn) = (geo().gu_m, geo().dn_m);
    let grid_y = m.pmax / 16;
    let gate_bytes = m.pmax * geo().mi * 2 + GUARD;
    let down_bytes = m.pmax * geo().dn_m * 2 + GUARD;
    let mut v = m.summary();
    v["kind"] = json!("plan_cell");
    v["gate_up"] = json!({"m":gu,"k":geo().hid,"x_row_div":geo().topk,"grid":[gu / 64, grid_y],"block":128,
        "output_bytes_with_guard":gate_bytes,"guard_bytes":GUARD,
        "leader_ctas":{"nt4":m.leaders4.len() * (gu / 64),"nt8":m.leaders8.len() * (gu / 64)}});
    v["down"] = json!({"m":dn,"k":geo().mi,"x_row_div":1,"grid":[dn / 64, grid_y],"block":128,
        "output_bytes_with_guard":down_bytes,"guard_bytes":GUARD,
        "leader_ctas":{"nt4":m.leaders4.len() * (dn / 64),"nt8":m.leaders8.len() * (dn / 64)}});
    v["compared_bytes_per_variant"] = json!(gate_bytes + down_bytes);
    v["production_selection"] = selection_record("gfx1201", admit("gfx1201").is_ok(), m.l, geo().e);
    v["timing"] = json!({"samples_per_variant_stage":MIN_TIMING_ITERS,"warmup":WARMUP,"order":"ABBA alternating","unit":"median microseconds, HIP events"});
    v
}

fn print_plan(o: &Opts) -> Res<()> {
    let g = geo();
    println!(
        "{}",
        json!({"kind":"plan_header","lab":"moe_iu4_wide_tiles_oracle","gpu_launched":false,
            "geometry":{"experts":g.e,"top_k":g.topk,"hidden":g.hid,"intermediate":g.mi,"gate_up_m":g.gu_m,"down_m":g.dn_m},
            "rows":o.rows,"routings":o.routings,"inputs":o.inputs.iter().map(|i| i.name()).collect::<Vec<_>>(),
            "residencies":o.hosts.iter().map(|&h| if h {"host"} else {"vram"}).collect::<Vec<_>>(),
            "iters":o.iters,"warmup":WARMUP,"seed":o.seed,"deadline_s":o.deadline_s,
            "producers":["moe_scatter_stable_top10","qwen4_moe_rotate256_i4","qwen4_moe_rotate128_i4"],
            "variants":["pm_nt4","pm_nt8"],"nt16":"not built: Spec::validate caps gfx12 at NT8, gfx11 at NT4",
            "stages":["gate_up","down","down_isolated (NT8 down on NT4's down sidecar)"],
            "production_selection":{"flag":WIDE_FLAG_ENV,"flag_0":"resident NT4 for both stages",
                "flag_unset":"follows qualification: exact gfx1201 resolves on, other arches off",
                "resident_flag_on":"flag resolved on (unset on gfx1201 or explicit 1): gate_up NT8 only when capable and total_slots >= 128 * n_experts; down always NT4",
                "host_mapped":"NT8 for both stages on a capable arch, unchanged by the flag",
                "min_slots_per_expert":128,"incapable_arch":"NT4 for both stages",
                "owner_measured":"R9700 VRAM random median us: GU wins at 4096/8192 rows, down loses (+4.5%/+5.2%); 12/12 cells byte-identical"},
            "diagnostic":"every cell still byte-compares and times NT4 vs NT8 for BOTH stages whatever production selects",
            "weights":{"distinct_experts":o.distinct,"aliased":o.distinct < g.e,
                "gate_up_bytes_per_expert":g.gu_stride,"down_bytes_per_expert":g.dn_stride,
                "total_bytes":o.distinct * (g.gu_stride + g.dn_stride),
                "shared_across_shapes":true,"synthesis":"rayon, per-expert seed seed*1000+e (down +600)"}})
    );
    for arch in ["gfx1201", "gfx1151", "gfx1100", "gfx942"] {
        println!(
            "{}",
            match admit(arch) {
                Ok(a) => json!({"kind":"plan_admission","arch":arch,"admitted":true,
                    "symbols":a.symbols,"bundle_bytes":a.bundle_bytes,"bundle_sha256":a.bundle_sha256,
                    "resident_wide_selection":stage_json(arch, expected_stage_nt(true, false, true, MAX_ROWS * g.topk, g.e))}),
                Err(r) => json!({"kind":"plan_admission","arch":arch,"admitted":false,"reasons":r}),
            }
        );
    }
    for &rows in &o.rows {
        for r in &o.routings {
            println!("{}", plan_cell(&model(rows, r, o.seed)?));
        }
    }
    println!("{}", json!({"plan":true,"gpu_launched":false}));
    Ok(())
}

// ------------------------------------------------------------- self check

fn self_check() -> Res<usize> {
    let mut n = 0usize;
    macro_rules! ck {
        ($c:expr, $($m:tt)+) => {{
            if !($c) {
                return Err(format!($($m)+).into());
            }
            n += 1;
        }};
    }
    // Geometry: the example's A3B table, the plan and the arithmetic agree.
    let g = geo();
    ck!(g.name == "a3b", "geometry is {}, not a3b", g.name);
    ck!(
        (g.e, g.topk, g.hid, g.mi)
            == (
                GEOM.experts as usize,
                GEOM.top_k as usize,
                GEOM.hidden as usize,
                GEOM.intermediate as usize
            ),
        "example geometry differs from the lab geometry"
    );
    ck!(
        g.gu_m == 1024 && g.dn_m == 2048,
        "projection M {} {}",
        g.gu_m,
        g.dn_m
    );
    ck!(
        g.gu_stride == 1024 * 8 * GU_GB && g.dn_stride == 2048 * 4 * DN_GB,
        "expert strides"
    );
    for arch in [Arch::Gfx1201, Arch::Gfx1151] {
        let p = GEOM
            .plan(arch, WeightCodec::Mq4Sym)
            .map_err(|e| format!("plan {}: {e}", arch.name()))?;
        ck!(
            (p.gate_up.m, p.gate_up.k, p.gate_up.epochs) == (1024, 2048, 16),
            "gate/up plan"
        );
        ck!(
            (p.down.m, p.down.k, p.down.epochs) == (2048, 512, 4),
            "down plan"
        );
        ck!(
            p.gate_up.row_bytes as usize == (g.hid / 256) * GU_GB,
            "gate/up row bytes"
        );
        ck!(
            p.down.row_bytes as usize == (g.mi / 128) * DN_GB,
            "down row bytes"
        );
        ck!((p.gate_up.grid_x, p.down.grid_x) == (16, 32), "grid x");
        ck!(
            (p.gate_up.x_row_div, p.down.x_row_div) == (8, 1),
            "x_row_div"
        );
        ck!(
            p.descriptor_table_bytes as usize == g.e * 8,
            "descriptor table bytes"
        );
        ck!(
            p.spec(Kind::GateUp, 16).is_err() && p.spec(Kind::Down, 16).is_err(),
            "NT16 must stay refused on {}",
            arch.name()
        );
        ck!(
            p.spec(Kind::GateUp, 4).is_ok() && p.spec(Kind::Down, 4).is_ok(),
            "NT4 on {}",
            arch.name()
        );
        ck!(
            p.spec(Kind::GateUp, 8).is_ok() == (arch == Arch::Gfx1201)
                && p.spec(Kind::Down, 8).is_ok() == (arch == Arch::Gfx1201),
            "NT8 admission follows the planned tile widths on {}",
            arch.name()
        );
    }
    // Raw bundle admission, pure CPU.
    let a = admit("gfx1201").map_err(|r| format!("gfx1201 admission: {r:?}"))?;
    ck!(
        a.nt4.name == "pm_nt4" && a.nt8.name == "pm_nt8" && a.symbols.len() == 4,
        "gfx1201 admission contents"
    );
    ck!(
        a.nt4.gu_block == 128 && a.nt8.gu_block == 128 && a.nt4.down_rr == 1 && a.nt8.down_rr == 1,
        "NT4/NT8 launch shape"
    );
    ck!(
        a.symbols.iter().all(|s| s.contains("_pm_gfx1201_nt")),
        "symbol names {:?}",
        a.symbols
    );
    ck!(
        has_symbol(
            rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1151,
            &a.nt4.gate_up.replace("gfx1201", "gfx1151")
        ),
        "gfx1151 bundle must carry NT4"
    );
    for arch in ["gfx1151", "gfx1100", "gfx942", "gfx1030", ""] {
        ck!(admit(arch).is_err(), "{arch:?} must not admit NT4+NT8");
    }
    // Production selection: resident widening is gate/up only; host-mapped is NT8/NT8.
    let arches = ["gfx1201", "gfx1151", "gfx1100", "gfx942", "gfx1030", ""];
    let caps: Vec<bool> = arches.iter().map(|a| admit(a).is_ok()).collect();
    ck!(
        caps == [true, false, false, false, false, false],
        "capability by arch {caps:?}"
    );
    let (n4, n8, mixed) = (
        StageNt {
            gate_up: 4,
            down: 4,
        },
        StageNt {
            gate_up: 8,
            down: 8,
        },
        StageNt {
            gate_up: 8,
            down: 4,
        },
    );
    // Pinned A3B table (256 experts, top-8): rows -> total slots rows * 8.
    for (rows, wide) in [(512usize, false), (4096, true), (8192, true)] {
        let slots = rows * g.topk;
        ck!(
            expected_stage_nt(true, false, true, slots, g.e) == if wide { mixed } else { n4 },
            "resident flag on, {rows} rows"
        );
        ck!(
            expected_stage_nt(true, false, false, slots, g.e) == n4,
            "resident flag off, {rows} rows"
        );
        ck!(
            expected_stage_nt(true, true, false, slots, g.e) == n8,
            "host flag off, {rows} rows"
        );
        ck!(
            expected_stage_nt(true, true, true, slots, g.e) == n8,
            "host flag on, {rows} rows"
        );
        ck!(
            expected_stage_nt(false, true, true, slots, g.e) == n4,
            "incapable host, {rows} rows"
        );
        ck!(
            expected_stage_nt(false, false, true, slots, g.e) == n4,
            "incapable resident, {rows} rows"
        );
    }
    // Threshold, zero experts and overflow shapes over every arch, residency and flag state.
    let mut shapes: Vec<(usize, usize)> =
        vec![(0, 0), (1, 0), (usize::MAX, 0), (usize::MAX, usize::MAX)];
    for experts in [1usize, 3, 256, 512] {
        for slots in [
            0,
            127 * experts,
            128 * experts - 1,
            128 * experts,
            128 * experts + 1,
        ] {
            shapes.push((slots, experts));
        }
    }
    shapes.extend([
        (usize::MAX, usize::MAX / 128 + 1),
        (usize::MAX, usize::MAX / 64),
        (usize::MAX - 127, usize::MAX / 128),
    ]);
    for (arch, &cap) in arches.iter().zip(&caps) {
        for host in [false, true] {
            for flag in [false, true] {
                for &(slots, ne) in &shapes {
                    let got = expected_stage_nt(cap, host, flag, slots, ne);
                    let wide = flag && ne > 0 && (slots as u128) >= 128 * (ne as u128);
                    let want = match (*arch == "gfx1201", host, wide) {
                        (false, _, _) => n4,
                        (true, true, _) => n8,
                        (true, false, true) => mixed,
                        (true, false, false) => n4,
                    };
                    ck!(got == want, "{arch:?} host={host} flag={flag} slots={slots} experts={ne}: {got:?} != {want:?}");
                    ck!(
                        (host && cap) || got.down == 4,
                        "{arch:?} resident down must be NT4: {got:?}"
                    );
                    ck!(
                        !wide || host || !cap || got.gate_up == 8,
                        "{arch:?} wide shape must widen gate/up"
                    );
                }
            }
        }
    }
    // Stage-specific symbols: every selected pair resolves in the plan, is named by the entry
    // table (gate/up from one entry, down from the other when mixed) and is admitted.
    for s in [n4, mixed, n8] {
        let gs = stage_symbol("gfx1201", Kind::GateUp, s.gate_up);
        let ds = stage_symbol("gfx1201", Kind::Down, s.down);
        let ge = if s.gate_up == 8 { a.nt8 } else { a.nt4 };
        let de = if s.down == 8 { a.nt8 } else { a.nt4 };
        ck!(
            gs.as_deref() == Some(ge.gate_up) && ds.as_deref() == Some(de.down),
            "stage symbols {s:?}: {gs:?} {ds:?}"
        );
        ck!(
            gs.iter().chain(&ds).all(|x| a.symbols.contains(x)),
            "selected symbols of {s:?} must be admitted symbols"
        );
    }
    ck!(
        stage_symbol("gfx1201", Kind::GateUp, 8) != stage_symbol("gfx1201", Kind::GateUp, 4)
            && stage_symbol("gfx1201", Kind::Down, 8) != stage_symbol("gfx1201", Kind::Down, 4),
        "NT4 and NT8 symbols must differ per stage"
    );
    // Machine-readable records.
    for (rows, wide) in [(512usize, false), (4096, true), (8192, true)] {
        let rec = selection_record("gfx1201", true, rows * g.topk, g.e);
        let on = &rec["resident_flag_on"];
        ck!(
            on["gate_up"]["nt"] == if wide { 8 } else { 4 }
                && on["down"]["nt"] == 4
                && on["mixed"] == wide
                && on["down"]["entry"] == "pm_nt4"
                && rec["wide_shape"] == wide,
            "resident flag-on record {rows}: {rec}"
        );
        ck!(
            rec["resident_flag_off"]["mixed"] == false
                && rec["resident_flag_off"]["gate_up"]["nt"] == 4
                && rec["host_mapped"]["gate_up"]["nt"] == 8
                && rec["host_mapped"]["down"]["nt"] == 8
                && on["gate_up"]["symbol"].is_string()
                && on["down"]["symbol"].is_string(),
            "flag-off / host record {rows}: {rec}"
        );
        let cell = plan_cell(&model(rows, "skew_empty", 1)?);
        ck!(
            cell["production_selection"] == rec,
            "plan_cell {rows} must carry the selection record"
        );
    }
    ck!(
        selection_record("gfx1151", false, 65536, g.e)["resident_flag_on"]["mixed"] == false,
        "gfx1151 never mixes"
    );
    // Production threshold: every shape reaches the route.
    ck!(
        ROWS.iter()
            .all(|&r| r >= rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS),
        "rows below QWEN4_MOE_SYM_IU4_MIN_ROWS {}",
        rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS
    );
    // Weights: parallel synthesis equals the sequential per-expert bytes.
    let (gu, dn) = synth_weights(2, 7);
    ck!(
        gu.len() == 2 * g.gu_stride && dn.len() == 2 * g.dn_stride,
        "synth sizes"
    );
    ck!(
        gu[g.gu_stride..] == symmetric_weights(g.gu_m, g.hid, GU_GB, 7001)[..],
        "gate/up expert 1 bytes"
    );
    ck!(
        dn[g.dn_stride..] == symmetric_weights(g.dn_m, g.mi, DN_GB, 7601)[..],
        "down expert 1 bytes"
    );
    ck!(
        median(&[3.0, 1.0, 2.0]) == 2.0 && median(&[4.0, 1.0, 3.0, 2.0]) == 2.5,
        "median"
    );
    // Per-shape, per-routing models.
    for &rows in &ROWS {
        for kind in ["skew_empty", "all", "skew"] {
            let m = model(rows, kind, 1)?;
            let again = route(kind, rows, 1)?;
            ck!(
                m.topk == again,
                "{kind} {rows}: routing is not deterministic"
            );
            ck!(
                m.l == rows * g.topk && m.l <= MAX_SLOTS as usize,
                "{kind} {rows}: slots {}",
                m.l
            );
            ck!(
                m.topk.iter().all(|&e| e >= 0 && (e as usize) < g.e),
                "{kind} {rows}: expert id range"
            );
            ck!(
                m.topk.chunks_exact(g.topk).all(|row| {
                    let mut s = row.to_vec();
                    s.sort_unstable();
                    s.windows(2).all(|w| w[0] != w[1])
                }),
                "{kind} {rows}: duplicate expert within a token"
            );
            // Grouping: stable (expert, flat) order, pad16, sentinel tiles.
            let gr = &m.g;
            ck!(
                m.pmax % 16 == 0 && m.pmax >= gr.offsets[g.e] as usize,
                "{kind} {rows}: pmax {}",
                m.pmax
            );
            ck!(
                gr.counts.iter().all(|c| c % 16 == 0),
                "{kind} {rows}: counts not pad16"
            );
            ck!(
                gr.counts
                    .iter()
                    .zip(&m.raw)
                    .all(|(&c, &r)| c as usize == r.div_ceil(16) * 16),
                "{kind} {rows}: padded counts"
            );
            ck!(
                (0..g.e).all(|e| gr.offsets[e + 1] - gr.offsets[e] == gr.counts[e])
                    && gr.offsets[0] == 0,
                "{kind} {rows}: offsets"
            );
            let mut order: Vec<usize> = (0..m.l).collect();
            order.sort_by_key(|&f| (m.topk[f], f));
            let live: Vec<i32> = gr.sorted.iter().copied().filter(|&s| s >= 0).collect();
            ck!(
                live == order.iter().map(|&f| f as i32).collect::<Vec<_>>(),
                "{kind} {rows}: sorted is not the stable (expert, flat) order"
            );
            let mut ok = true;
            for e in 0..g.e {
                let (lo, hi) = (gr.offsets[e] as usize, gr.offsets[e + 1] as usize);
                let cnt = m.raw[e];
                ok &= gr.sorted[lo..lo + cnt]
                    .iter()
                    .all(|&f| f >= 0 && m.topk[f as usize] as usize == e);
                ok &= gr.sorted[lo + cnt..hi].iter().all(|&f| f < 0);
                ok &= gr.tiles[lo / 16..hi / 16].iter().all(|&t| t as usize == e);
            }
            ck!(ok, "{kind} {rows}: per-expert grouping");
            let p = gr.offsets[g.e] as usize;
            ck!(
                gr.sorted[p..].iter().all(|&s| s < 0) && gr.tiles[p / 16..].iter().all(|&t| t < 0),
                "{kind} {rows}: tail sentinels"
            );
            ck!(
                (0..m.l).all(|f| gr.sorted[gr.inverse[f] as usize] == f as i32),
                "{kind} {rows}: inverse"
            );
            // Leader metadata arithmetic order, NT4 and NT8.
            let live_tiles = p / 16;
            for (nt, ld) in [(4usize, &m.leaders4), (8usize, &m.leaders8)] {
                let mut covered = vec![0u8; gr.tiles.len()];
                for &(t, cnt) in ld {
                    ck!((1..=nt).contains(&cnt), "{kind} {rows} NT{nt}: cnt {cnt}");
                    ck!(
                        (0..cnt).all(|j| gr.tiles[t + j] == gr.tiles[t]),
                        "{kind} {rows} NT{nt}: span leaves its run"
                    );
                    for j in 0..cnt {
                        covered[t + j] += 1;
                    }
                }
                ck!(
                    covered
                        .iter()
                        .enumerate()
                        .all(|(t, &c)| c == (gr.tiles[t] >= 0) as u8),
                    "{kind} {rows} NT{nt}: live tiles are not covered exactly once"
                );
                ck!(
                    ld.iter().map(|l| l.1).sum::<usize>() == live_tiles,
                    "{kind} {rows} NT{nt}: covered tile count"
                );
                ck!(
                    ld.len() == m.runs.iter().map(|r| r.1.div_ceil(nt)).sum::<usize>(),
                    "{kind} {rows} NT{nt}: leaders = sum ceil(run/nt)"
                );
                // Straight position (count back) equals the 32-lane walk.
                let mut start = 0;
                let mut agree = true;
                for t in 0..gr.tiles.len() {
                    if t == 0 || gr.tiles[t] != gr.tiles[t - 1] {
                        start = t;
                    }
                    if gr.tiles[t] >= 0 {
                        agree &= run_position_walk(&gr.tiles, t) == t - start;
                    }
                }
                ck!(
                    agree,
                    "{kind} {rows} NT{nt}: 32-lane walk differs from run position"
                );
                // Per-leader cnt = min(nt, run remaining).
                ck!(
                    ld.iter().all(|&(t, cnt)| {
                        let pos = run_position_walk(&gr.tiles, t);
                        let run = m.runs.iter().find(|r| r.0 as i32 == gr.tiles[t]).unwrap().1;
                        cnt == nt.min(run - pos)
                    }),
                    "{kind} {rows} NT{nt}: cnt != min(nt, remaining)"
                );
            }
            let nt4_tiles: std::collections::BTreeSet<usize> =
                m.leaders4.iter().map(|l| l.0).collect();
            ck!(
                m.leaders8.iter().all(|l| nt4_tiles.contains(&l.0)),
                "{kind} {rows}: NT8 leaders are not NT4 leaders"
            );
            ck!(
                m.leaders8.len() <= m.leaders4.len(),
                "{kind} {rows}: NT8 leaders exceed NT4"
            );
            // Wide-tile exercise thresholds.
            if kind == "skew_empty" {
                ck!(
                    m.empty() >= EMPTY_MIN,
                    "{kind} {rows}: only {} empty experts (< {EMPTY_MIN})",
                    m.empty()
                );
                ck!(
                    m.raw[..HOT].iter().all(|&c| c == rows),
                    "{kind} {rows}: hot experts must take every token"
                );
                ck!(
                    m.singles() >= SINGLES_MIN,
                    "{kind} {rows}: only {} experts with 1..=15 slots",
                    m.singles()
                );
                ck!(
                    m.runs.iter().any(|r| r.1 >= 8),
                    "{kind} {rows}: no full NT8 run"
                );
                ck!(
                    m.runs.iter().any(|r| r.1 < 4),
                    "{kind} {rows}: no run shorter than NT4"
                );
                ck!(
                    m.runs.iter().any(|r| r.1 % 8 != 0) && m.runs.iter().any(|r| r.1 % 4 != 0),
                    "{kind} {rows}: no partial leader"
                );
                ck!(
                    m.leaders8.len() < m.leaders4.len(),
                    "{kind} {rows}: NT8 must be coarser than NT4"
                );
            }
            if kind == "all" {
                ck!(
                    m.l % g.e == 0 && m.raw.iter().all(|&c| c == m.l / g.e),
                    "all {rows}: round robin is not uniform"
                );
            }
        }
    }
    // locate/compare: byte classification on a synthetic grouping.
    let m = model(512, "skew_empty", 1)?;
    let rb = g.mi * 2;
    let p = m.g.offsets[g.e] as usize;
    let a = vec![7u8; m.pmax * rb + GUARD];
    for (off, region) in [
        (0usize, "live_slot"),
        (p * rb, "unused_tile"),
        (m.pmax * rb, "guard"),
        (m.pmax * rb + GUARD - 1, "guard"),
    ] {
        let mut b = a.clone();
        b[off] ^= 0xFF;
        let (same, rec) = compare("t", &a, &b, rb, p, m.pmax, &m.g);
        ck!(
            !same
                && rec["first_mismatch"]["region"] == region
                && rec["first_mismatch"]["offset"] == off
                && rec["mismatched_bytes"] == 1,
            "compare at {off}: {rec}"
        );
    }
    let pad_row = (0..p)
        .find(|&r| m.g.sorted[r] < 0)
        .ok_or("no padding slot in live tiles")?;
    ck!(
        locate_byte(pad_row * rb + 6, rb, p, m.pmax, &m.g)["region"] == "padding_slot",
        "padding slot classification"
    );
    ck!(
        locate_byte(rb + 6, rb, p, m.pmax, &m.g)["column"] == 3,
        "column of byte 6"
    );
    ck!(
        compare("t", &a, &a, rb, p, m.pmax, &m.g).0,
        "identical buffers"
    );
    // Option invariants.
    let bad =
        |v: &[&str]| parse_opts(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>()).is_err();
    ck!(
        bad(&["x", "--iters", "19"])
            && bad(&["x", "--deadline-s", "121"])
            && bad(&["x", "--rows", "8193"])
            && bad(&["x", "--bogus"]),
        "option validation"
    );
    ck!(parse_opts(&["x".to_string()]).is_ok(), "default options");
    Ok(n)
}

// ------------------------------------------------------------------ device

/// The full-size weight tensors of one residency, lent to each shape's `Case`.
struct Shared {
    gu_t: GpuTensor,
    dn_t: GpuTensor,
    gu_ptrs: GpuTensor,
    dn_ptrs: GpuTensor,
}

fn upload_shared(gpu: &mut Gpu, gu: &[u8], dn: &[u8], distinct: usize, host: bool) -> Res<Shared> {
    let (gu_t, dn_t) = if host {
        (
            gpu.upload_raw_host_mapped(gu, &[gu.len()])?,
            gpu.upload_raw_host_mapped(dn, &[dn.len()])?,
        )
    } else {
        (
            gpu.upload_raw(gu, &[gu.len()])?,
            gpu.upload_raw(dn, &[dn.len()])?,
        )
    };
    let gu_ptrs = gpu.upload_raw(&ptr_table(&gu_t, geo().gu_stride, distinct), &[geo().e * 8])?;
    let dn_ptrs = gpu.upload_raw(&ptr_table(&dn_t, geo().dn_stride, distinct), &[geo().e * 8])?;
    Ok(Shared {
        gu_t,
        dn_t,
        gu_ptrs,
        dn_ptrs,
    })
}

/// Swap the weight tensors of `c` and `s`; calling it twice restores both.
fn lend(c: &mut Case, s: &mut Shared) {
    std::mem::swap(&mut c.gu_t, &mut s.gu_t);
    std::mem::swap(&mut c.dn_t, &mut s.dn_t);
    std::mem::swap(&mut c.gu_ptrs, &mut s.gu_ptrs);
    std::mem::swap(&mut c.dn_ptrs, &mut s.dn_ptrs);
}

fn gate_bytes(c: &Case) -> usize {
    c.pmax * geo().mi * 2 + GUARD
}
fn down_bytes(c: &Case) -> usize {
    c.pmax * geo().dn_m * 2 + GUARD
}

/// One variant's chain from the production gate sidecar: poisoned outputs,
/// gate/up, production down sidecar, down. `iso` also runs another entry's
/// down on that same sidecar.
fn chain(
    gpu: &mut Gpu,
    c: &Case,
    e: Entry,
    ygu: &GpuTensor,
    ydn: &GpuTensor,
    poison: i32,
    iso: Option<(Entry, &GpuTensor)>,
) -> Res<()> {
    gpu.hip.memset(&ygu.buf, poison, gate_bytes(c))?;
    gpu.hip.memset(&ydn.buf, poison, down_bytes(c))?;
    entry_gemm(gpu, c, e, false, ygu)?;
    let _down_sidecar = gpu.qwen4_moe_rotate128_i4(ygu, &c.sorted, geo().mi, c.pmax, c.l)?;
    entry_gemm(gpu, c, e, true, ydn)?;
    if let Some((other, y)) = iso {
        gpu.hip.memset(&y.buf, poison, down_bytes(c))?;
        entry_gemm(gpu, c, other, true, y)?;
    }
    Ok(())
}

/// >= `iters` interleaved samples per variant and stage; needs the gate and
/// down sidecars of a correct chain to be live.
fn time_cell(
    gpu: &mut Gpu,
    c: &Case,
    adm: &Admission,
    iters: usize,
    dl: &Deadline,
    host: bool,
    total_slots: usize,
) -> Res<Value> {
    let ev: Vec<_> = (0..3)
        .map(|_| gpu.hip.event_create())
        .collect::<Result<_, _>>()?;
    let outs = [(&c.ygu, &c.ydn), (&c.h_ygu, &c.h_ydn)];
    let ents = [adm.nt4, adm.nt8];
    let mut gu: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut dn: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for it in 0..WARMUP + iters {
        dl.check("timing sample")?;
        let order = if it % 2 == 0 { [0, 1] } else { [1, 0] };
        for v in order {
            gpu.hip.event_record(&ev[0], None)?;
            entry_gemm(gpu, c, ents[v], false, outs[v].0)?;
            gpu.hip.event_record(&ev[1], None)?;
            entry_gemm(gpu, c, ents[v], true, outs[v].1)?;
            gpu.hip.event_record(&ev[2], None)?;
            gpu.hip.event_synchronize(&ev[2])?;
            if it >= WARMUP {
                gu[v].push(gpu.hip.event_elapsed_ms(&ev[0], &ev[1])? as f64 * 1000.0);
                dn[v].push(gpu.hip.event_elapsed_ms(&ev[1], &ev[2])? as f64 * 1000.0);
            }
        }
    }
    let tot = |v: usize| median(&gu[v]) + median(&dn[v]);
    // The production pair's stage medians (flag on, resident / this residency): the
    // gate/up and down medians of whichever width each stage selects.
    let resident = expected_stage_nt(true, false, true, total_slots, geo().e);
    let current = expected_stage_nt(true, host, true, total_slots, geo().e);
    let prod_timing = |s: StageNt| {
        let (g, d) = (usize::from(s.gate_up == 8), usize::from(s.down == 8));
        let sum = median(&gu[g]) + median(&dn[d]);
        json!({"gate_up_nt":s.gate_up,"down_nt":s.down,"gate_up_median_us":median(&gu[g]),
            "down_median_us":median(&dn[d]),"sum_of_medians_us":sum,"over_nt4_nt4_sum":sum / tot(0)})
    };
    Ok(json!({
        "samples":iters,"warmup":WARMUP,"order":"ABBA alternating",
        "nt4":{"gate_up":stat_json(&gu[0]),"down":stat_json(&dn[0]),"sum_of_medians_us":tot(0)},
        "nt8":{"gate_up":stat_json(&gu[1]),"down":stat_json(&dn[1]),"sum_of_medians_us":tot(1)},
        "nt8_over_nt4_median":{"gate_up":median(&gu[1]) / median(&gu[0]),"down":median(&dn[1]) / median(&dn[0]),"sum":tot(1) / tot(0)},
        "production_resident_flag_on":prod_timing(resident),
        "production_current_residency":prod_timing(current),
    }))
}

fn set_input(gpu: &Gpu, c: &Case, current: &mut Input, want: Input, xbig: &[f32]) -> Res<()> {
    if *current == want {
        return Ok(());
    }
    let n = c.t * geo().hid;
    match want {
        Input::Zero => gpu.hip.memset(&c.x_in.buf, 0, n * 4)?,
        Input::Random => {
            let bytes: Vec<u8> = xbig[..n].iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.hip.memcpy_htod(&c.x_in.buf, &bytes)?;
        }
    }
    *current = want;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_cell(
    gpu: &mut Gpu,
    c: &Case,
    m: &Model,
    adm: &Admission,
    input: Input,
    host: bool,
    o: &Opts,
    dl: &Deadline,
) -> Res<(bool, Value)> {
    let t0 = dl.elapsed_s();
    dl.check("producers")?;
    gpu.moe_scatter_stable_top10(
        &c.topk,
        &c.counts,
        &c.offsets,
        &c.sorted,
        &c.tiles,
        &c.inverse,
        c.l,
        geo().e,
        c.pmax,
    )?;
    let _gate_sidecar = gpu.qwen4_moe_rotate256_i4(&c.x_in, geo().hid, c.t)?;
    dl.check("chain NT4/NT8")?;
    let poison = input.poison();
    chain(
        gpu,
        c,
        adm.nt4,
        &c.ygu,
        &c.ydn,
        poison,
        Some((adm.nt8, &c.i_ydn)),
    )?;
    chain(gpu, c, adm.nt8, &c.h_ygu, &c.h_ydn, poison, None)?;
    gpu.hip.device_synchronize()?;
    dl.check("download")?;
    let g = &m.g;
    let group_ok = to_i32(&download(gpu, &c.sorted.buf, c.pmax * 4)?) == g.sorted
        && to_i32(&download(gpu, &c.tiles.buf, c.pmax / 16 * 4)?) == g.tiles
        && to_i32(&download(gpu, &c.inverse.buf, c.l * 4)?) == g.inverse
        && to_i32(&download(gpu, &c.counts.buf, geo().e * 4)?) == g.counts
        && to_i32(&download(gpu, &c.offsets.buf, (geo().e + 1) * 4)?) == g.offsets;
    let live_rows = g.offsets[geo().e] as usize;
    let (gb, db) = (gate_bytes(c), down_bytes(c));
    let gate_a = download(gpu, &c.ygu.buf, gb)?;
    let gate_b = download(gpu, &c.h_ygu.buf, gb)?;
    let (gate_same, gate) = compare(
        "gate_up",
        &gate_a,
        &gate_b,
        geo().mi * 2,
        live_rows,
        c.pmax,
        g,
    );
    drop((gate_a, gate_b));
    let down_a = download(gpu, &c.ydn.buf, db)?;
    let down_b = download(gpu, &c.h_ydn.buf, db)?;
    let (down_same, down) = compare(
        "down",
        &down_a,
        &down_b,
        geo().dn_m * 2,
        live_rows,
        c.pmax,
        g,
    );
    drop(down_b);
    let down_i = download(gpu, &c.i_ydn.buf, db)?;
    let (iso_same, iso) = compare(
        "down_isolated",
        &down_a,
        &down_i,
        geo().dn_m * 2,
        live_rows,
        c.pmax,
        g,
    );
    drop((down_a, down_i));
    let identical = gate_same && down_same && iso_same;
    let timing = if identical && group_ok {
        time_cell(gpu, c, adm, o.iters, dl, host, m.l)?
    } else {
        Value::Null
    };
    let ok = identical && group_ok;
    Ok((
        ok,
        json!({
            "kind":"cell","residency":if host {"host"} else {"vram"},"rows":m.rows,"routing":m.routing,
            "input":input.name(),"poison_byte":poison,"slots_L":m.l,"padded_P":live_rows,"pmax":c.pmax,
            "active_experts":m.runs.len(),"empty_experts":m.empty(),
            "leaders_nt4":m.leaders4.len(),"leaders_nt8":m.leaders8.len(),
            "group_matches_cpu_stable":group_ok,
            "gate_up":gate,"down":down,"down_isolated":iso,
            "all_byte_identical":identical,"timing_us":timing,
            "production_selection":selection_record(&adm.arch, true, m.l, geo().e),
            "diagnostic":"NT4 and NT8 are compared and timed for BOTH stages whatever production selects",
            "cell_s":dl.elapsed_s() - t0,
        }),
    ))
}

fn run_gpu(o: &Opts, dl: &Deadline) -> Res<bool> {
    dl.check("init")?;
    let mut gpu = Gpu::init()?;
    let adm = match admit(&gpu.arch) {
        Ok(a) => a,
        Err(reasons) => {
            println!(
                "{}",
                json!({"kind":"admission","arch":gpu.arch,"admitted":false,"reasons":reasons})
            );
            return Ok(false);
        }
    };
    // Capability is proven by the bundle loading the symbols, not by a name.
    for s in &adm.symbols {
        gpu.ensure_qwen4_moe_sym_pm_entry(s)?;
    }
    let g = *geo();
    println!(
        "{}",
        json!({"kind":"setup","arch":gpu.arch,"admitted":true,
            "admission":"raw bundle capability: typed ISA plan, entry table and embedded bundle symbols, loaded",
            "bundle_bytes":adm.bundle_bytes,"bundle_sha256":adm.bundle_sha256,"symbols":adm.symbols,
            "entries":[adm.nt4.name,adm.nt8.name],"geometry":[g.e,g.topk,g.hid,g.mi],
            "rows":o.rows,"routings":o.routings,"inputs":o.inputs.iter().map(|i| i.name()).collect::<Vec<_>>(),
            "residencies":o.hosts.iter().map(|&h| if h {"host"} else {"vram"}).collect::<Vec<_>>(),
            "iters":o.iters,"warmup":WARMUP,"seed":o.seed,"deadline_s":o.deadline_s,
            "producers":["moe_scatter_stable_top10","qwen4_moe_rotate256_i4","qwen4_moe_rotate128_i4"],
            "weights":{"distinct_experts":o.distinct,"aliased_by_pointer_table":o.distinct < g.e,
                "alias_rule":"expert e uses synthetic expert e % distinct_experts",
                "bytes_per_residency":o.distinct * (g.gu_stride + g.dn_stride),
                "shared_across_shapes_and_inputs":true,"synthesized_once":true},
            "x":"random uniform [-2, 2), one vector of the largest row count; each shape uses its prefix; zero input is a memset"})
    );
    dl.check("weight synthesis")?;
    let (gu_w, dn_w) = synth_weights(o.distinct, o.seed);
    dl.check("x synthesis")?;
    let max_rows = *o.rows.iter().max().unwrap();
    let mut r = Rng(o.seed ^ 0x77);
    let xbig: Vec<f32> = (0..max_rows * g.hid)
        .map(|_| (r.unit() - 0.5) * 4.0)
        .collect();
    println!(
        "{}",
        json!({"kind":"inputs","sha256_gate_up_weights":sha_hex(&gu_w),"sha256_down_weights":sha_hex(&dn_w)})
    );
    let (mut cells, mut failures) = (0usize, 0usize);
    for &host in &o.hosts {
        dl.check("weight upload")?;
        let mut shared = upload_shared(&mut gpu, &gu_w, &dn_w, o.distinct, host)?;
        let mut headers_checked = false;
        for &rows in &o.rows {
            for routing_kind in &o.routings {
                dl.check("case build")?;
                let m = model(rows, routing_kind, o.seed)?;
                if m.pmax != pmax_of(m.l) {
                    return Err("model pmax differs from make_case".into());
                }
                let mut c = make_case(
                    &mut gpu,
                    m.topk.clone(),
                    &xbig[..rows * g.hid],
                    gu_w[..g.gu_stride].to_vec(),
                    dn_w[..g.dn_stride].to_vec(),
                    host,
                )?;
                if c.pmax != m.pmax {
                    return Err(format!("case pmax {} != model pmax {}", c.pmax, m.pmax).into());
                }
                lend(&mut c, &mut shared);
                if !headers_checked {
                    let (gok, dok) = header_checks(&mut gpu, &c)?;
                    println!(
                        "{}",
                        json!({"kind":"headers","residency":if host {"host"} else {"vram"},
                            "gate_up_symmetric":gok,"down_symmetric":dok})
                    );
                    if !(gok && dok) {
                        return Ok(false);
                    }
                    headers_checked = true;
                }
                let mut current = Input::Random;
                for &input in &o.inputs {
                    set_input(&gpu, &c, &mut current, input, &xbig)?;
                    let (ok, rec) = run_cell(&mut gpu, &c, &m, &adm, input, host, o, dl)?;
                    println!("{rec}");
                    cells += 1;
                    failures += !ok as usize;
                }
                lend(&mut c, &mut shared);
            }
        }
    }
    let elapsed = dl.elapsed_s();
    let over = elapsed > o.deadline_s as f64;
    println!(
        "{}",
        json!({"summary":true,"mode":"gpu","arch":gpu.arch,"cells":cells,"failures":failures,
            "all_byte_identical":failures == 0,"elapsed_s":elapsed,"deadline_s":o.deadline_s,
            "deadline_exceeded":over})
    );
    Ok(failures == 0 && !over)
}

fn run_inner(args: &[String]) -> Res<bool> {
    let o = parse_opts(args)?;
    let dl = Deadline::new(o.deadline_s);
    let _ = GEO.set(GEO_A3B);
    if geo().name != "a3b" {
        return Err("geometry already selected and is not a3b".into());
    }
    if o.plan {
        print_plan(&o)?;
        return Ok(true);
    }
    let checks = self_check()?;
    if o.check {
        println!(
            "{}",
            json!({"self_check":"passed","checks":checks,"gpu_launched":false,"elapsed_s":dl.elapsed_s()})
        );
        return Ok(true);
    }
    run_gpu(&o, &dl)
}

/// Process exit code: 0 all identical (and within the deadline), 1 otherwise.
pub fn run() -> i32 {
    let args: Vec<String> = std::env::args().collect();
    match run_inner(&args) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(e) => {
            println!(
                "{}",
                json!({"error":e.to_string(),"deadline_exceeded":e.downcast_ref::<DeadlineExceeded>().is_some()})
            );
            eprintln!("moe_iu4_wide_tiles_oracle: {e}");
            1
        }
    }
}
