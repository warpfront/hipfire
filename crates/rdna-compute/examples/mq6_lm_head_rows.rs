// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte gate and timer of the MQ6G256V2 LM-head projection at verify row counts.
//!
//! ```text
//! mq6_lm_head_rows [--m M] [--k K] [--rows R,R,...] [--warmup N] [--iters N]
//!                  [--seed S] [--out PATH]
//! ```
//!
//! | flag       | default                      | meaning                                         |
//! |------------|------------------------------|-------------------------------------------------|
//! | `--m`      | `248320`                     | weight rows (vocabulary), positive              |
//! | `--k`      | `2560`                       | weight columns, positive, `K % 256 == 0`        |
//! | `--rows`   | `1,2,4,8,9,12,16,17,24,32`   | activation row counts `R`, each in `1..=32`     |
//! | `--warmup` | `3`                          | untimed calls per arm and `R`                   |
//! | `--iters`  | `10`                         | timed calls per arm and `R`, at least 1         |
//! | `--seed`   | `0x6d71366c6d686431`         | nonzero fixture seed (decimal or `0x` hex)      |
//! | `--out`    | `mq6_lm_head_rows.json`      | JSON report path (cwd-relative)                 |
//!
//! `--flag value` and `--flag=value` are both accepted; `-h`/`--help` prints the
//! usage and exits 0 before any GPU is initialized.
//!
//! # Fixture
//!
//! One MQ6G256V2 weight `[M][K/256]` of 200-byte groups (finite f16 headers
//! `s0 z0 s1 z1`, random 192-byte six-bit payload, as `mq6_wide_rows_oracle`) and
//! 32 F32 activation rows with mixed magnitudes, alternating-sign, cancelling,
//! subnormal and `+-0` rows (rows 6, 14, 22 and 30 are all `+-0`). Activations
//! live inside a NaN-filled allocation and every launch gets an exact-size view.
//! The output allocation is `64 guard floats | 32 * M floats | 64 guard floats`,
//! memset to the `0xFF` NaN sentinel before every gated launch; an active launch
//! of `R` rows may write only the first `R` rows.
//!
//! # Arms (per `R`)
//!
//! * `ar`: `R` separate `Gpu::gemv_mq6g256v2` launches (the AR decode launch);
//!   also the reference. One timed call is the whole `R`-launch sequence.
//! * `route`: the production public launcher: `R = 1` is `Gpu::gemv_mq6g256v2`,
//!   `R >= 2` is `Gpu::gemm_mq6g256v2_f32_rows`. The example opens the verify gate
//!   (`set_wide_verify_rows(32)`, so `few_row_limit() == 32`) and sets
//!   `qwen4_scope = true` and `lane_rows_scope = true` as the lane step does, then leaves both alone. What
//!   the launcher selects depends on the arch, the gate, `lane_rows_scope`,
//!   `HIPFIRE_FN_LANE_PM_ROWS` and `HIPFIRE_QWEN4_WIDE_TWIN_MQ6` (below).
//! * `pm_wide`: exact gfx1151 only: the `qwen4_lane_rows_pm` export
//!   `gemv_mq6g256v2_xbatch_wide_r16_pm` (`R <= 16`) or `..._r32_pm` (the exports
//!   take any `R` in `1..=cap`), loaded with `Gpu::ensure_lane_rows_pm_entry` and
//!   launched by name (kernargs `A@0 x@8 y@16 M@24 K@28 R@32`, grid `ceil(M/8)`, block 256), independent of
//!   `HIPFIRE_FN_LANE_PM_ROWS`, `lane_rows_scope` and the wide-twin switch. Any
//!   other arch, or a shape the PM kernels' 32-bit byte offsets refuse, is
//!   reported as an explicit `skipped` reason, never silently dropped.
//!
//! # Switches
//!
//! `HIPFIRE_QWEN4_WIDE_TWIN_MQ6` (default off) decides whether `route` at
//! `R in 9..=32` takes the PM wide export (gfx1151; no HIP twin remains) or the
//! exact 8-row group loop. The raw environment value and the
//! effective `WideTwinFamily::Mq6.enabled()` are printed and put in the JSON; this
//! tool never calls `set_var`. Run it twice, with and without the switch, to time
//! both routes.
//!
//! # Gate
//!
//! For every arm and `R` the whole output allocation (head guard, 32 rows, tail
//! guard) is downloaded and compared byte for byte (every `to_bits` word) with the
//! `ar` reference for `R` rows, inactive rows `R..32` and both guards expected to
//! still hold the sentinel. The check runs after the first (prelaunch) call and
//! again after the timed calls; the report merges both. `mismatch_bytes` counts
//! every differing byte, `mismatch_guard_bytes` the subset inside the guards,
//! `mismatch_rows` the distinct output rows (active and inactive) holding a
//! differing byte and `mismatch_inactive_rows` those with index `>= R`. The `ar`
//! arm runs in its own output allocation against a separate reference run, and
//! sentinel violations of the reference run itself are counted in the `ar` arm.
//!
//! # Timing
//!
//! Every arm and `R` first gets one untimed prelaunch/gate call (loads modules, so
//! nothing compiles in the timed window), then `--warmup` untimed calls, then
//! `--iters` timed calls. Calls of the arms of one `R` are interleaved round robin
//! (each arm has its own output allocation) so slow clock or thermal drift hits all
//! arms alike. A timed call is `device_synchronize`, start the host clock, one call
//! (for `ar` the whole `R`-launch sequence), `device_synchronize`, stop; it
//! therefore includes host launch cost. Reported: `median_us` and `min_us` per
//! call, `us_per_row = median_us / R` and `gbs = weight_bytes / median_us / 1000`
//! (weight traffic only; activations and outputs are not counted).
//!
//! # Symbols
//!
//! `ar` and `route` report what `Gpu::last_launched_kernel` showed after the call.
//! When `route` at `R > 8` did not end in a wide twin it ran the exact 8-row group
//! loop of the launcher; each group is then replayed once, untimed, through the
//! same public entry points to record the per-group symbols in order
//! (`symbols`, `launches_per_call`, `symbol_probe_consistent`). `direct_*` and
//! `pm_wide` launch by name, which `last_launched_kernel` does not track, so their
//! `symbols` are the launched names (`symbols_from: "named_launch"`).
//!
//! # Output and exit status
//!
//! One result line per arm and `R` on stdout, plus the JSON report (metadata:
//! arch, M, K, weight/allocation bytes, switch state, `few_row_limit`, gate and
//! scope state, per-`R` wide-twin cap; one entry per arm and `R`). Exit 0 when every
//! gate holds, 1 on any mismatch, 2 on argument, setup, launch or JSON errors.

use hip_bridge::KernargBlob;
use rdna_compute::gemv::WideTwinFamily;
use rdna_compute::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::ffi::c_void;
use std::time::Instant;

type Result<T, E = String> = std::result::Result<T, E>;

const MAXR: usize = 32;
/// Guard floats on each side of the output allocation.
const GUARD: usize = 64;
/// Output prefill: `0xFFFFFFFF` is a NaN no kernel produces.
const SENTINEL: u8 = 0xff;
/// NaN activation rows before and after the `[MAXR, K]` X view.
const PRE_ROWS: usize = 3;
const GROUP_BYTES: usize = 200;
const ENV_SWITCH: &str = "HIPFIRE_QWEN4_WIDE_TWIN_MQ6";
const DEFAULT_OUT: &str = "mq6_lm_head_rows.json";
const DEFAULT_SEED: u64 = 0x6d71_366c_6d68_6431;
const DEFAULT_ROWS: [usize; 10] = [1, 2, 4, 8, 9, 12, 16, 17, 24, 32];
/// Per-row magnitudes cycle; `0.0` is the all-zero (+-0) row.
const AMPS: [f32; 8] = [1.0, 1.0 / 1024.0, 1.0 / 16.0, 4.0, 64.0, 0.5, 0.0, 1024.0];
const PM_WIDE: [&str; 2] = [
    "gemv_mq6g256v2_xbatch_wide_r16_pm",
    "gemv_mq6g256v2_xbatch_wide_r32_pm",
];

const USAGE: &str = "usage: mq6_lm_head_rows [--m M] [--k K] [--rows R,R,...] [--warmup N] [--iters N] [--seed S] [--out PATH]
  --m       weight rows, positive                      (default 248320)
  --k       weight columns, positive, K % 256 == 0     (default 2560)
  --rows    activation row counts, each 1..=32         (default 1,2,4,8,9,12,16,17,24,32)
  --warmup  untimed calls per arm and R                (default 3)
  --iters   timed calls per arm and R, at least 1      (default 10)
  --seed    nonzero fixture seed, decimal or 0x hex    (default 0x6d71366c6d686431)
  --out     JSON report path                           (default mq6_lm_head_rows.json)
exit: 0 all gates hold, 1 byte mismatch, 2 argument/setup/launch/JSON error";

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

struct Rng(u64);

impl Rng {
    /// Nonzero stream state derived from `seed` and a per-purpose `salt`.
    fn derive(seed: u64, salt: u64) -> Self {
        let mut z = seed.wrapping_add(salt.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        Self(z | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn fill(&mut self, out: &mut [u8]) {
        let mut chunks = out.chunks_exact_mut(8);
        for c in &mut chunks {
            c.copy_from_slice(&self.next().to_le_bytes());
        }
        let rest = chunks.into_remainder();
        let tail = self.next().to_le_bytes();
        rest.copy_from_slice(&tail[..rest.len()]);
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    /// A finite f16: mostly `2^-lo..2^hi`, sometimes `+-0` or subnormal.
    fn f16(&mut self, lo_exp: u16, hi_exp: u16) -> u16 {
        let sign = (self.below(2) as u16) << 15;
        match self.below(32) {
            0 => sign,
            1 => sign | (1 + self.below(0x3ff) as u16),
            _ => {
                sign | ((15 - lo_exp + self.below(u64::from(lo_exp + hi_exp + 1)) as u16) << 10)
                    | self.below(0x400) as u16
            }
        }
    }
}

/// MQ6G256V2 weights: `[m][k/256]` groups of 8 header bytes (`s0 z0 s1 z1`, f16,
/// independent per half) and 192 random packed payload bytes.
fn gen_mq6(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; m * (k / 256) * GROUP_BYTES];
    for g in bytes.chunks_exact_mut(GROUP_BYTES) {
        rng.fill(&mut g[8..]);
        for half in 0..2 {
            let (s, z) = (rng.f16(9, 2), rng.f16(7, 5));
            g[4 * half..4 * half + 2].copy_from_slice(&s.to_le_bytes());
            g[4 * half + 2..4 * half + 4].copy_from_slice(&z.to_le_bytes());
        }
    }
    bytes
}

/// `[MAXR][k]` activations: row magnitudes cycle through [`AMPS`]; row kinds
/// add alternating signs (`r % 11 == 3`), +-2^20 cancellation with a small
/// residue (`5`) and pure subnormals/zeros (`7`); rows with a zero magnitude are
/// all `+-0`.
fn gen_x(k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(MAXR * k);
    for r in 0..MAXR {
        let amp = AMPS[r % AMPS.len()];
        for i in 0..k {
            let v = match r % 11 {
                _ if amp == 0.0 => {
                    if rng.below(2) == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                3 => {
                    if i % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                5 => match i % 4 {
                    0 => 1_048_576.0,
                    1 => -1_048_576.0,
                    _ => rng.unit() * 1e-3,
                },
                7 => f32::from_bits(((rng.below(2) as u32) << 31) | rng.below(1 << 23) as u32),
                _ => match rng.below(64) {
                    0 => 0.0,
                    1 => -0.0,
                    2..=5 => {
                        f32::from_bits(((rng.below(2) as u32) << 31) | rng.below(1 << 23) as u32)
                    }
                    _ => rng.unit() * amp,
                },
            };
            x.push(v);
        }
    }
    x
}

struct Opts {
    m: usize,
    k: usize,
    rows: Vec<usize>,
    warmup: usize,
    iters: usize,
    seed: u64,
    out: String,
}

fn parse_usize(flag: &str, v: &str) -> Result<usize> {
    v.parse::<usize>().map_err(|e| format!("{flag} {v:?}: {e}"))
}

fn parse_seed(v: &str) -> Result<u64> {
    let parsed = match v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => v.parse::<u64>(),
    };
    let seed = parsed.map_err(|e| format!("--seed {v:?}: {e}"))?;
    if seed == 0 {
        return Err("--seed must be nonzero".to_string());
    }
    Ok(seed)
}

fn parse_rows(v: &str) -> Result<Vec<usize>> {
    let mut rows = Vec::new();
    for part in v.split(',') {
        let r = parse_usize("--rows", part.trim())?;
        if !(1..=MAXR).contains(&r) {
            return Err(format!("--rows entry {r} outside 1..={MAXR}"));
        }
        if rows.contains(&r) {
            return Err(format!("--rows entry {r} is repeated"));
        }
        rows.push(r);
    }
    Ok(rows)
}

fn take_value(
    flag: &str,
    inline: Option<String>,
    args: &[String],
    i: &mut usize,
) -> Result<String> {
    if let Some(v) = inline {
        return Ok(v);
    }
    let v = args
        .get(*i)
        .ok_or_else(|| format!("{flag} needs a value"))?;
    *i += 1;
    Ok(v.clone())
}

/// `None` is `--help`.
fn parse_args(args: &[String]) -> Result<Option<Opts>> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        return Ok(None);
    }
    let mut o = Opts {
        m: 248_320,
        k: 2560,
        rows: DEFAULT_ROWS.to_vec(),
        warmup: 3,
        iters: 10,
        seed: DEFAULT_SEED,
        out: DEFAULT_OUT.to_string(),
    };
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].clone();
        i += 1;
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (arg, None),
        };
        let v = match flag.as_str() {
            "--m" | "--k" | "--rows" | "--warmup" | "--iters" | "--seed" | "--out" => {
                take_value(&flag, inline, args, &mut i)?
            }
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        };
        match flag.as_str() {
            "--m" => o.m = parse_usize("--m", &v)?,
            "--k" => o.k = parse_usize("--k", &v)?,
            "--rows" => o.rows = parse_rows(&v)?,
            "--warmup" => o.warmup = parse_usize("--warmup", &v)?,
            "--iters" => o.iters = parse_usize("--iters", &v)?,
            "--seed" => o.seed = parse_seed(&v)?,
            _ => o.out = v,
        }
    }
    Ok(Some(o))
}

struct Geometry {
    weight_bytes: usize,
    out_bytes: usize,
    x_bytes: usize,
}

/// Positive `M`, `K % 256 == 0`, `iters > 0` and allocation/index arithmetic that
/// fits `i32` element indices and `usize` byte sizes.
fn validate(o: &Opts) -> Result<Geometry> {
    if o.m == 0 || o.k == 0 {
        return Err("--m and --k must be positive".to_string());
    }
    if o.k % 256 != 0 {
        return Err(format!("--k {} is not a multiple of 256", o.k));
    }
    if o.iters == 0 {
        return Err("--iters must be at least 1".to_string());
    }
    let i32_max = i32::MAX as usize;
    if o.m > i32_max || o.k > i32_max {
        return Err("--m and --k must fit i32".to_string());
    }
    let weight_bytes =
        o.m.checked_mul(o.k / 256)
            .and_then(|n| n.checked_mul(GROUP_BYTES))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or("weight byte size overflows")?;
    let out_elems = MAXR
        .checked_mul(o.m)
        .and_then(|n| n.checked_add(2 * GUARD))
        .filter(|&n| n <= i32_max)
        .ok_or("32 output rows of M floats plus guards overflow i32 element indices")?;
    let x_elems = (2 * PRE_ROWS + MAXR)
        .checked_mul(o.k)
        .filter(|&n| n <= i32_max)
        .ok_or("the activation allocation overflows i32 element indices")?;
    Ok(Geometry {
        weight_bytes,
        out_bytes: out_elems * 4,
        x_bytes: x_elems * 4,
    })
}

/// One output allocation: `GUARD | MAXR * m f32 | GUARD`.
struct Out {
    full: GpuTensor,
    m: usize,
    bytes: usize,
}

impl Out {
    fn new(gpu: &mut Gpu, m: usize) -> Result<Self> {
        let elems = 2 * GUARD + MAXR * m;
        let full = gpu.zeros(&[elems], DType::F32).map_err(err)?;
        Ok(Self {
            full,
            m,
            bytes: elems * 4,
        })
    }
    fn view(&self) -> GpuTensor {
        self.full.sub_offset(GUARD, MAXR * self.m)
    }
    fn reset(&self, gpu: &Gpu) -> Result<()> {
        gpu.hip
            .memset(&self.full.buf, i32::from(SENTINEL), self.bytes)
            .map_err(err)?;
        gpu.hip.device_synchronize().map_err(err)
    }
    fn download(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        gpu.hip.device_synchronize().map_err(err)?;
        let mut v = vec![0u8; self.bytes];
        gpu.hip.memcpy_dtoh(&mut v, &self.full.buf).map_err(err)?;
        Ok(v)
    }
    /// Expected bytes: sentinel everywhere, `reference`'s first `rows` rows in place.
    fn expected(&self, reference: &[u8], rows: usize) -> Vec<u8> {
        let mut e = vec![SENTINEL; self.bytes];
        let (lo, hi) = (GUARD * 4, GUARD * 4 + rows * self.m * 4);
        e[lo..hi].copy_from_slice(&reference[lo..hi]);
        e
    }
    fn locate(&self, idx: usize) -> String {
        let (lo, hi) = (GUARD * 4, GUARD * 4 + MAXR * self.m * 4);
        if idx < lo {
            format!("head guard byte {idx}")
        } else if idx >= hi {
            format!("tail guard byte {}", idx - hi)
        } else {
            let off = idx - lo;
            let (row, rem) = (off / (self.m * 4), off % (self.m * 4));
            format!("row {row} column {} byte {}", rem / 4, rem % 4)
        }
    }
}

/// Byte-level difference of one output allocation against its expectation.
#[derive(Default, Clone)]
struct Check {
    bytes: u64,
    guard_bytes: u64,
    /// Bit `r` is set when output row `r` (active or not) holds a differing byte.
    row_mask: u64,
    first: Option<String>,
}

impl Check {
    fn merge(&mut self, other: &Check) {
        self.bytes += other.bytes;
        self.guard_bytes += other.guard_bytes;
        self.row_mask |= other.row_mask;
        if self.first.is_none() {
            self.first.clone_from(&other.first);
        }
    }
    fn rows(&self) -> u32 {
        self.row_mask.count_ones()
    }
    fn inactive_rows(&self, active: usize) -> u32 {
        (self.row_mask >> active).count_ones()
    }
}

fn compare(out: &Out, got: &[u8], want: &[u8]) -> Check {
    let mut c = Check::default();
    if got == want {
        return c;
    }
    let (lo, hi) = (GUARD * 4, GUARD * 4 + MAXR * out.m * 4);
    let row_bytes = out.m * 4;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g == w {
            continue;
        }
        c.bytes += 1;
        if i < lo || i >= hi {
            c.guard_bytes += 1;
        } else {
            c.row_mask |= 1u64 << ((i - lo) / row_bytes);
        }
        if c.first.is_none() {
            c.first = Some(format!("{} (got {g:#04x}, want {w:#04x})", out.locate(i)));
        }
    }
    let extra = got.len().abs_diff(want.len()) as u64;
    c.bytes += extra;
    c.guard_bytes += extra;
    c
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Ar = 0,
    Route = 1,
    PmWide = 2,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Ar => "ar",
            Arm::Route => "route",
            Arm::PmWide => "pm_wide",
        }
    }
    /// The kernel a by-name arm launches at `rows`.
    fn named_symbol(self, rows: usize) -> Option<&'static str> {
        match self {
            Arm::PmWide => Some(PM_WIDE[usize::from(rows > 16)]),
            Arm::Ar | Arm::Route => None,
        }
    }
}

struct Fixture {
    m: usize,
    k: usize,
    w: GpuTensor,
    /// `[PRE_ROWS NaN rows | MAXR rows | PRE_ROWS NaN rows] x k`.
    x_full: GpuTensor,
}

impl Fixture {
    /// Exact-size view of the first `rows` activation rows.
    fn x_view(&self, rows: usize) -> GpuTensor {
        self.x_full.sub_offset(PRE_ROWS * self.k, rows * self.k)
    }
}

/// Views one call of one arm needs, built outside the timed window.
struct Prep {
    x: GpuTensor,
    y: GpuTensor,
    /// `ar`: one `(x row, y row)` pair per activation row.
    ar: Vec<(GpuTensor, GpuTensor)>,
}

impl Prep {
    fn new(fx: &Fixture, out: &Out, arm: Arm, rows: usize) -> Self {
        let x = fx.x_view(rows);
        let view = out.view();
        let mut ar = Vec::new();
        let y = if arm == Arm::Ar {
            for r in 0..rows {
                ar.push((
                    x.sub_offset(r * fx.k, fx.k),
                    view.sub_offset(r * fx.m, fx.m),
                ));
            }
            view
        } else if arm == Arm::Route && rows == 1 {
            view.sub_offset(0, fx.m)
        } else {
            view
        };
        Self { x, y, ar }
    }
}

/// One launch of `name` with the ABI of the wide twins (`A@0 x@8 y@16 M@24 K@28
/// R@32`, grid `ceil(M/8)`, block 256).
fn direct(gpu: &Gpu, name: &str, fx: &Fixture, prep: &Prep, rows: usize) -> Result<()> {
    let mut kb = KernargBlob::new();
    kb.push_ptr(fx.w.buf.as_ptr() as *const c_void);
    kb.push_ptr(prep.x.buf.as_ptr() as *const c_void);
    kb.push_ptr(prep.y.buf.as_ptr() as *const c_void);
    kb.push_i32(fx.m as i32);
    kb.push_i32(fx.k as i32);
    kb.push_i32(rows as i32);
    let blocks = fx.m.div_ceil(8) as u32;
    gpu.launch_kernel_blob(name, [blocks, 1, 1], [256, 1, 1], 0, kb.as_mut_slice())
        .map_err(err)
}

/// One call of `arm` at `rows` (for `ar`, the whole `rows`-launch sequence).
fn call(gpu: &mut Gpu, fx: &Fixture, arm: Arm, prep: &Prep, rows: usize) -> Result<()> {
    match arm {
        Arm::Ar => {
            for (x, y) in &prep.ar {
                gpu.gemv_mq6g256v2(&fx.w, x, y, fx.m, fx.k).map_err(err)?;
            }
            Ok(())
        }
        Arm::Route if rows == 1 => gpu
            .gemv_mq6g256v2(&fx.w, &prep.x, &prep.y, fx.m, fx.k)
            .map_err(err),
        Arm::Route => gpu
            .gemm_mq6g256v2_f32_rows(&fx.w, &prep.x, &prep.y, fx.m, fx.k, rows)
            .map_err(err),
        Arm::PmWide => {
            let name = arm
                .named_symbol(rows)
                .ok_or("by-name arm without a symbol")?;
            direct(gpu, name, fx, prep, rows)
        }
    }
}

fn last_symbol(gpu: &Gpu) -> String {
    gpu.last_launched_kernel().unwrap_or("<none>").to_string()
}

struct Symbols {
    list: Vec<String>,
    launches: usize,
    source: &'static str,
    probe_consistent: bool,
    /// `last_launched_kernel` right after the gate call (`None` for by-name arms,
    /// which that tracker does not see).
    last: Option<String>,
}

/// Symbols of the call just made (before any further launch).
fn capture(gpu: &mut Gpu, fx: &Fixture, arm: Arm, prep: &Prep, rows: usize) -> Result<Symbols> {
    if let Some(name) = arm.named_symbol(rows) {
        return Ok(Symbols {
            list: vec![name.to_string()],
            launches: 1,
            source: "named_launch",
            probe_consistent: true,
            last: None,
        });
    }
    let last = last_symbol(gpu);
    let source = "last_launched_kernel";
    if arm == Arm::Ar {
        return Ok(Symbols {
            list: vec![last.clone()],
            launches: rows,
            source,
            probe_consistent: true,
            last: Some(last),
        });
    }
    if rows <= 8 || last.contains("_wide_r") {
        return Ok(Symbols {
            list: vec![last.clone()],
            launches: 1,
            source,
            probe_consistent: true,
            last: Some(last),
        });
    }
    // The launcher ran its 8-row group loop: replay each group through the same
    // public entry points (`n == 1` is the AR decode launch) and record its symbol.
    let mut seq: Vec<String> = Vec::new();
    for first in (0..rows).step_by(8) {
        let n = 8.min(rows - first);
        let xg = prep.x.sub_offset(first * fx.k, n * fx.k);
        let yg = prep.y.sub_offset(first * fx.m, n * fx.m);
        if n == 1 {
            gpu.gemv_mq6g256v2(&fx.w, &xg, &yg, fx.m, fx.k)
                .map_err(err)?;
        } else {
            gpu.gemm_mq6g256v2_f32_rows(&fx.w, &xg, &yg, fx.m, fx.k, n)
                .map_err(err)?;
        }
        seq.push(last_symbol(gpu));
    }
    gpu.hip.device_synchronize().map_err(err)?;
    let probe_consistent = seq.last() == Some(&last);
    let launches = seq.len();
    let mut list: Vec<String> = Vec::new();
    for s in seq {
        if !list.contains(&s) {
            list.push(s);
        }
    }
    Ok(Symbols {
        list,
        launches,
        source,
        probe_consistent,
        last: Some(last),
    })
}

/// Exact gfx1151 PM wide eligibility; `Some(reason)` is the skip reason.
fn pm_wide_skip(gpu: &Gpu, geo: &Geometry, fx: &Fixture, rows: usize) -> Option<String> {
    if !gpu.arch_caps.is_gfx1151() {
        return Some(format!("exact gfx1151 only (arch {})", gpu.arch));
    }
    let u32_max = u32::MAX as usize;
    let fits = |bytes: Option<usize>| bytes.is_some_and(|b| b <= u32_max);
    if !fits(Some(geo.weight_bytes))
        || !fits(rows.checked_mul(fx.k).and_then(|n| n.checked_mul(4)))
        || !fits(rows.checked_mul(fx.m).and_then(|n| n.checked_mul(4)))
    {
        return Some("PM kernels address weights, activations and outputs with 32-bit byte offsets and this shape exceeds them".to_string());
    }
    None
}

struct ArmRun {
    arm: Arm,
    prep: Prep,
    symbols: Symbols,
    check: Check,
    samples: Vec<f64>,
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

fn run() -> Result<bool> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(opts) = parse_args(&args)? else {
        println!("{USAGE}");
        return Ok(true);
    };
    let geo = validate(&opts)?;
    let (m, k) = (opts.m, opts.k);

    let mut gpu = Gpu::init().map_err(err)?;
    // Production verify: the lane step opens the row-local PM scope and the verify
    // forward opens the wide gate; both stay as set.
    gpu.qwen4_scope = true;
    gpu.lane_rows_scope = true;
    gpu.set_wide_verify_rows(WIDE_VERIFY_MAX_ROWS);
    let raw_switch = match std::env::var(ENV_SWITCH) {
        Ok(v) => Some(v),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(s)) => Some(s.to_string_lossy().into_owned()),
    };
    let effective = WideTwinFamily::Mq6.enabled();
    let few_row_limit = gpu.few_row_limit();
    println!(
        "arch {} M {m} K {k} weight_buffer_bytes {} out_alloc_bytes {} x_alloc_bytes {}",
        gpu.arch, geo.weight_bytes, geo.out_bytes, geo.x_bytes
    );
    println!(
        "{ENV_SWITCH} raw={} effective={} few_row_limit {few_row_limit} wide_verify_rows {WIDE_VERIFY_MAX_ROWS} lane_rows_scope {} lane_rows_pm_enabled {}",
        raw_switch.as_deref().map_or("<unset>".to_string(), |v| format!("{v:?}")),
        if effective { "on" } else { "off" },
        gpu.lane_rows_scope,
        gpu.lane_rows_pm_enabled()
    );
    println!(
        "seed {:#x} warmup {} iters {} rows {:?}",
        opts.seed, opts.warmup, opts.iters, opts.rows
    );

    let weights = gen_mq6(m, k, &mut Rng::derive(opts.seed, 1));
    let w = gpu.upload_raw(&weights, &[weights.len()]).map_err(err)?;
    drop(weights);
    let x_host = gen_x(k, &mut Rng::derive(opts.seed, 2));
    let mut x_all = vec![f32::NAN; (2 * PRE_ROWS + MAXR) * k];
    x_all[PRE_ROWS * k..(PRE_ROWS + MAXR) * k].copy_from_slice(&x_host);
    let x_full = gpu.upload_f32(&x_all, &[x_all.len()]).map_err(err)?;
    let fx = Fixture { m, k, w, x_full };
    let ref_out = Out::new(&mut gpu, m)?;
    let mut outs = Vec::new();
    for _ in 0..3 {
        outs.push(Out::new(&mut gpu, m)?);
    }

    let mut results: Vec<Value> = Vec::new();
    let mut per_rows: Vec<Value> = Vec::new();
    let mut bad = false;
    for &rows in &opts.rows {
        per_rows.push(json!({
            "rows": rows,
            "row_tile_twin_enabled": gpu.row_tile_twin_enabled(WideTwinFamily::Mq6, rows),
        }));

        // Reference: `rows` separate AR launches into their own allocation.
        let ref_prep = Prep::new(&fx, &ref_out, Arm::Ar, rows);
        ref_out.reset(&gpu)?;
        call(&mut gpu, &fx, Arm::Ar, &ref_prep, rows)?;
        let ref_bytes = ref_out.download(&gpu)?;
        let expected = ref_out.expected(&ref_bytes, rows);
        let ref_check = compare(&ref_out, &ref_bytes, &expected);

        let mut arms = vec![Arm::Ar, Arm::Route];
        match pm_wide_skip(&gpu, &geo, &fx, rows) {
            None => arms.push(Arm::PmWide),
            Some(reason) => {
                println!("pm_wide R={rows}: SKIPPED ({reason})");
                results.push(json!({"arm": "pm_wide", "rows": rows, "skipped": reason}));
            }
        }

        // Prelaunch + first gated call per arm.
        let mut runs: Vec<ArmRun> = Vec::new();
        for &arm in &arms {
            let out = &outs[arm as usize];
            let prep = Prep::new(&fx, out, arm, rows);
            if let Some(name) = arm.named_symbol(rows) {
                gpu.ensure_lane_rows_pm_entry(name).map_err(err)?;
            }
            out.reset(&gpu)?;
            call(&mut gpu, &fx, arm, &prep, rows)?;
            let got = out.download(&gpu)?;
            let mut check = compare(out, &got, &expected);
            if arm == Arm::Ar {
                check.merge(&ref_check);
            }
            let symbols = capture(&mut gpu, &fx, arm, &prep, rows)?;
            runs.push(ArmRun {
                arm,
                prep,
                symbols,
                check,
                samples: Vec::with_capacity(opts.iters),
            });
        }

        for _ in 0..opts.warmup {
            for run in &runs {
                call(&mut gpu, &fx, run.arm, &run.prep, rows)?;
                gpu.hip.device_synchronize().map_err(err)?;
            }
        }
        for _ in 0..opts.iters {
            for run in &mut runs {
                gpu.hip.device_synchronize().map_err(err)?;
                let t0 = Instant::now();
                call(&mut gpu, &fx, run.arm, &run.prep, rows)?;
                gpu.hip.device_synchronize().map_err(err)?;
                run.samples.push(t0.elapsed().as_secs_f64() * 1e6);
            }
        }

        for run in &mut runs {
            let out = &outs[run.arm as usize];
            let got = out.download(&gpu)?;
            let post = compare(out, &got, &expected);
            let gate = run.check.clone();
            let mut all = gate.clone();
            all.merge(&post);
            let mut sorted = run.samples.clone();
            sorted.sort_by(f64::total_cmp);
            let (med, min) = (median(&sorted), sorted[0]);
            let upr = med / rows as f64;
            let gbs = geo.weight_bytes as f64 / med / 1000.0;
            bad |= all.bytes > 0;
            println!(
                "{:<10} R={rows:<2} median {med:>10.2} us  min {min:>10.2} us  {upr:>9.3} us/row  {gbs:>8.2} GB/s  mismatch_bytes {} rows {} (inactive {}) guard_bytes {}  launches/call {}  symbols {}{}",
                run.arm.name(),
                all.bytes,
                all.rows(),
                all.inactive_rows(rows),
                all.guard_bytes,
                run.symbols.launches,
                run.symbols.list.join(","),
                all.first.as_deref().map_or(String::new(), |f| format!("  FIRST {f}")),
            );
            if !run.symbols.probe_consistent {
                println!("{:<10} R={rows:<2} WARNING: group replay did not end in the symbol the route call left", run.arm.name());
            }
            results.push(json!({
                "arm": run.arm.name(),
                "rows": rows,
                "symbols": run.symbols.list,
                "symbols_from": run.symbols.source,
                "launches_per_call": run.symbols.launches,
                "symbol_probe_consistent": run.symbols.probe_consistent,
                "last_launched_kernel": run.symbols.last,
                "warmup": opts.warmup,
                "iters": opts.iters,
                "median_us": med,
                "min_us": min,
                "us_per_row": upr,
                "gbs": gbs,
                "samples_us": run.samples,
                "mismatch_bytes": all.bytes,
                "mismatch_rows": all.rows(),
                "mismatch_inactive_rows": all.inactive_rows(rows),
                "mismatch_guard_bytes": all.guard_bytes,
                "gate_mismatch_bytes": gate.bytes,
                "post_timing_mismatch_bytes": post.bytes,
                "first_mismatch": all.first,
            }));
        }
    }

    let report = json!({
        "schema": "mq6-lm-head-rows-v1",
        "arch": gpu.arch,
        "m": m,
        "k": k,
        "rows": opts.rows,
        "warmup": opts.warmup,
        "iters": opts.iters,
        "seed": opts.seed,
        "weight_buffer_bytes": geo.weight_bytes,
        "out_alloc_bytes": geo.out_bytes,
        "x_alloc_bytes": geo.x_bytes,
        "switch": {"var": ENV_SWITCH, "raw": raw_switch, "effective": effective},
        "few_row_limit": few_row_limit,
        "wide_verify_rows": WIDE_VERIFY_MAX_ROWS,
        "lane_rows_scope": gpu.lane_rows_scope,
        "lane_rows_pm_enabled": gpu.lane_rows_pm_enabled(),
        "per_rows": per_rows,
        "mismatch": bad,
        "results": results,
    });
    let text = serde_json::to_vec_pretty(&report).map_err(err)?;
    std::fs::write(&opts.out, text).map_err(|e| format!("writing {}: {e}", opts.out))?;
    println!("wrote {}", opts.out);
    Ok(!bad)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("mq6_lm_head_rows: byte mismatch");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("mq6_lm_head_rows: {e}");
            std::process::exit(2);
        }
    }
}
