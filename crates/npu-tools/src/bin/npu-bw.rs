// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! npu-bw: shim DMA bandwidth probe (`peacemaker_npu::kernels::bw_probe`) on the NPU through railgun::npu.
//!
//! usage: npu-bw --cols C --read R --write W --bytes N[K|M|G] [--burst E] [--axcache V] [--axqos V] [--iters I] [--timeout-ms T]
//!        npu-bw --sweep [--iters I] [--timeout-ms T]
//!        npu-bw --sweep-axi [--iters I] [--timeout-ms T]
//!        npu-bw --active-channels N --queued-tasks Q --bd-bytes N[K|M] --direction read|write|mixed [--memory MODE] [--burst E] [--iters I] [--timeout-ms T]
//!        npu-bw --sweep-concurrency [--active-channels N] [--queued-tasks Q] [--bd-bytes N[K|M]] [--direction D] [--memory MODE] [--iters I] [--timeout-ms T]
//!
//! `C` columns (0..C), `R` shim MM2S (DDR read) and `W` shim S2MM (DDR write) channels per column, `N` bytes per
//! channel (multiple of 4 KiB, <= 64 MiB), at least one of R/W > 0. One hardware context (full-array partition,
//! like `npu-gemm --array`), one PDI/insts/argument set per configuration; submit `I` times (default 8, >= 2).
//! Submit 0 (fresh context) is reported separately on the `first` line; `best_us`/`median_us` are over the warm
//! submits 1..I (time = submit -> syncobj completion). Rates use `best_us`: bandwidth in GB/s (1e9 B/s),
//! `per_*_ch_GBps` divide by the number of active channels of that direction, and
//! `bytes_per_aie_cycle_per_ch(@1.8GHz)` = bytes per channel / (best_us * 1.8 GHz) (one active direction: that
//! direction's per-channel rate; read + write both active: both directions move `bytes` concurrently so it is the
//! same per-channel figure). The write buffer is poisoned (0xA5) before submit 0 and before the last submit and
//! compared with the memtile pattern after both; `write_ok` is `true` only if both are exact (`n/a` if W = 0);
//! `write_first_ok` / `write_final_ok` (appended at the end of the result line) report the two checks separately.
//! `--burst E` / `--axcache V` / `--axqos V` set the shim BD AXI attributes (`ShimAxi`: burst encoding 0..=3 =
//! 64/128/256/512 B, AxCACHE 2, AxQoS 0 by default). Non-vendor word-5 attributes require `--allow-hazardous-axi`.
//! `--memory native|vmm-host|vmm-host-uncached` selects the backing of both argument BOs. The HIP modes use
//! exported host VMM allocations, not device mallocs that migrate on import. Read buffers contain a deterministic
//! pattern, but the cyclic read sink has no return path: `read_ok=timing-only` explicitly makes no exactness claim.
//!
//! `--sweep` runs a fixed list, one context per configuration: bytes = 16 MiB per channel for cols {1,2,4,8} x
//! read {1,2} and cols {1,2,4,8} x write {1,2}, cols 8 read 2 write 2, then cols 8 read 2 at 1 MiB and 64 MiB.
//! `--sweep-axi` runs cols 8 x 2 channels, 16 MiB per channel, read-only then write-only, for burst encodings
//! 0..=3 with vendor word-5 attributes (AxCACHE 2, AxQoS 0). `--allow-hazardous-axi` is single-probe only:
//! register fit does not establish safety. AxQoS 15 wedged Halo and subsequent SMU resume failed with -22.
//!
//! Concurrency probes (`bw_probe::concurrency_design`) measure the ceiling cause: `N` active channels per active
//! direction (N in {1,2,4,8,16}; direction `read` = N read channels, `write` = N write channels, `mixed` = N of
//! each), `Q` queued DDR tasks per channel (1, 2, 4), `bd_bytes` per BD/task (power of two, 4 KiB..=1 MiB).
//! The single-config flags `--active-channels --queued-tasks --bd-bytes --direction` are all required unless
//! `--sweep-concurrency` is given. `--sweep-concurrency` runs the full 5 x 3 x 9 x 3 = 405 grid; each of the
//! four flags is then an optional exact-value filter (never ignored), and the selected count and filters are
//! printed (`npu-bw: concurrency sweep selected K of 405 ...`) before any hardware is touched. Concurrency
//! probes use the same runner as the classic probes and print `mode=concurrency direction=..
//! active_channels_per_direction=.. read_channels=.. write_channels=.. queued=.. bd_bytes=.. read_bytes=.. write_bytes=..`
//! (`mixed` has N read AND N write channels, 2N total; `--active-channels` is always per direction),
//! `read_ok=timing-only` for reads
//! and `write_first_ok`/`write_final_ok`/`write_ok` (`n/a` without writes). They never accept
//! `--allow-hazardous-axi` (single or sweep), reject any `--axcache` != 2 / `--axqos` != 0, and reject the
//! legacy geometry/sweep flags (`--cols --read --write --bytes --sweep --sweep-axi`) and unknown flags.
//! A timeout / submit error prints the state and continues with the next configuration. Exit status is non-zero if
//! any configuration failed (bad arguments exit 2).
use peacemaker_npu::dma::ShimAxi;
use peacemaker_npu::kernels::bw_probe::{
    concurrency_design, design, BwConfig, BwDesign, ConcurrencyConfig, ConcurrencyDesign,
    WriteCheck, AIE_HZ,
};
use railgun::npu::hip_runtime::{HipDmabufA, HipMemoryKind};
use railgun::npu::{Bo, Device, HwCtx, ERT_STATE_COMPLETED};
use std::time::Instant;

fn opt_str<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).map(|i| {
        args.get(i + 1)
            .unwrap_or_else(|| usage(&format!("{name} needs a value")))
            .as_str()
    })
}

fn usage(msg: &str) -> ! {
    eprintln!("npu-bw: {msg}\nusage: npu-bw --cols C --read R --write W --bytes N[K|M|G] [--memory native|vmm-host|vmm-host-uncached] [--burst E] [--axcache V] [--axqos V] [--allow-hazardous-axi] [--iters I] [--timeout-ms T]\n       npu-bw --sweep|--sweep-axi [--memory MODE] [--iters I] [--timeout-ms T]\n       npu-bw --active-channels N --queued-tasks Q --bd-bytes N[K|M] --direction read|write|mixed [--memory MODE] [--burst E] [--iters I] [--timeout-ms T]\n       npu-bw --sweep-concurrency [--active-channels N] [--queued-tasks Q] [--bd-bytes N[K|M]] [--direction read|write|mixed] [--memory MODE] [--burst E] [--iters I] [--timeout-ms T]\n(concurrency probes never accept --allow-hazardous-axi, --axcache != 2 or --axqos != 0)");
    std::process::exit(2)
}

fn opt_num(args: &[String], name: &str) -> Option<usize> {
    opt_str(args, name).map(|s| {
        s.parse()
            .unwrap_or_else(|_| usage(&format!("{name} expects an integer, got {s:?}")))
    })
}

/// `N[K|M|G]` bytes with checked overflow.
fn parse_bytes(name: &str, s: &str) -> usize {
    let (num, shift) = match s.chars().last() {
        Some('k' | 'K') => (&s[..s.len() - 1], 10),
        Some('m' | 'M') => (&s[..s.len() - 1], 20),
        Some('g' | 'G') => (&s[..s.len() - 1], 30),
        _ => (s, 0),
    };
    num.parse::<usize>()
        .ok()
        .and_then(|n| n.checked_mul(1usize << shift))
        .unwrap_or_else(|| {
            usage(&format!(
                "{name} expects N[K|M|G] without overflow, got {s:?}"
            ))
        })
}

fn fmt_gbps(v: f64) -> String {
    format!("{v:.3}")
}

/// Concurrency probe direction: `N` channels of the selected direction(s).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dir {
    Read,
    Write,
    Mixed,
}

impl Dir {
    const ALL: [Dir; 3] = [Dir::Read, Dir::Write, Dir::Mixed];
    fn name(self) -> &'static str {
        match self {
            Dir::Read => "read",
            Dir::Write => "write",
            Dir::Mixed => "mixed",
        }
    }
    fn parse(s: &str) -> Option<Dir> {
        Dir::ALL.into_iter().find(|d| d.name() == s)
    }
}

/// One concurrency grid point; converted to [`ConcurrencyConfig`] only when validated/built.
#[derive(Clone, Copy, Debug)]
struct ConcSpec {
    dir: Dir,
    n: usize,
    q: usize,
    bd: usize,
    axi: ShimAxi,
}

impl ConcSpec {
    fn config(&self) -> ConcurrencyConfig {
        ConcurrencyConfig {
            read_channels: if self.dir == Dir::Write { 0 } else { self.n },
            write_channels: if self.dir == Dir::Read { 0 } else { self.n },
            queued_tasks: self.q,
            bd_bytes: self.bd,
            axi: self.axi,
        }
    }
}

const CONC_CHANNELS: [usize; 5] = [1, 2, 4, 8, 16];
const CONC_QUEUED: [usize; 3] = [1, 2, 4];
const CONC_BD_MIN_LOG2: u32 = 12;
const CONC_BD_MAX_LOG2: u32 = 20;
/// 5 channel counts x 3 queue depths x 9 BD sizes x 3 directions.
const CONC_FULL: usize = CONC_CHANNELS.len()
    * CONC_QUEUED.len()
    * (CONC_BD_MAX_LOG2 - CONC_BD_MIN_LOG2 + 1) as usize
    * Dir::ALL.len();

/// Either probe family; the single runner handles both through the helpers below.
#[derive(Clone, Copy, Debug)]
enum ProbeConfig {
    Classic(BwConfig),
    Concurrency(ConcSpec),
}

enum ProbeDesign {
    Classic(BwDesign),
    Concurrency(ConcurrencyDesign),
}

impl ProbeConfig {
    fn axi(&self) -> ShimAxi {
        match self {
            ProbeConfig::Classic(c) => c.axi,
            ProbeConfig::Concurrency(c) => c.axi,
        }
    }
    /// Pre-hardware validation. Concurrency probes never allow non-vendor word-5 attributes.
    fn validate(&self, hazardous: bool) -> Result<(), String> {
        match self {
            ProbeConfig::Classic(c) => {
                if hazardous {
                    c.validate_for_hazardous_probe()
                } else {
                    c.validate()
                }
            }
            ProbeConfig::Concurrency(c) => {
                if hazardous {
                    return Err(
                        "--allow-hazardous-axi is not accepted for concurrency probes".into(),
                    );
                }
                c.axi.validate_vendor_word5()?;
                c.config().validate()
            }
        }
    }
    fn build(&self, hazardous: bool) -> ProbeDesign {
        match self {
            ProbeConfig::Classic(c) => ProbeDesign::Classic(if hazardous {
                peacemaker_npu::kernels::bw_probe::design_for_hazardous_probe(*c)
            } else {
                design(*c)
            }),
            ProbeConfig::Concurrency(c) => ProbeDesign::Concurrency(concurrency_design(c.config())),
        }
    }
}

impl ProbeDesign {
    fn pdi(&self) -> &[u8] {
        match self {
            ProbeDesign::Classic(d) => &d.pdi,
            ProbeDesign::Concurrency(d) => &d.pdi,
        }
    }
    fn insts(&self) -> &[u8] {
        match self {
            ProbeDesign::Classic(d) => &d.insts,
            ProbeDesign::Concurrency(d) => &d.insts,
        }
    }
    fn arg_bytes(&self) -> [usize; 2] {
        match self {
            ProbeDesign::Classic(d) => [d.args[0].bytes, d.args[1].bytes],
            ProbeDesign::Concurrency(d) => [d.args[0].bytes, d.args[1].bytes],
        }
    }
    fn read_total(&self) -> usize {
        match self {
            ProbeDesign::Classic(d) => d.cfg.read_total(),
            ProbeDesign::Concurrency(d) => d.cfg.read_total(),
        }
    }
    fn write_total(&self) -> usize {
        match self {
            ProbeDesign::Classic(d) => d.cfg.write_total(),
            ProbeDesign::Concurrency(d) => d.cfg.write_total(),
        }
    }
    /// Total active read / write channels (the divisors of the per-channel rates).
    fn read_channels(&self) -> usize {
        match self {
            ProbeDesign::Classic(d) => d.cfg.cols * d.cfg.read_ch,
            ProbeDesign::Concurrency(d) => d.cfg.read_channels,
        }
    }
    fn write_channels(&self) -> usize {
        match self {
            ProbeDesign::Classic(d) => d.cfg.cols * d.cfg.write_ch,
            ProbeDesign::Concurrency(d) => d.cfg.write_channels,
        }
    }
    /// Bytes moved by each active channel per submit.
    fn bytes_per_channel(&self) -> usize {
        match self {
            ProbeDesign::Classic(d) => d.cfg.bytes,
            ProbeDesign::Concurrency(d) => d.cfg.queued_tasks * d.cfg.bd_bytes,
        }
    }
    fn check_write(&self, buf: &[u8]) -> WriteCheck {
        match self {
            ProbeDesign::Classic(d) => d.check_write(buf),
            ProbeDesign::Concurrency(d) => d.check_write(buf),
        }
    }
    /// Identification fields shared by the design/first/result lines. Classic output is unchanged.
    fn ident(&self, spec: &ProbeConfig) -> String {
        match (self, spec) {
            (ProbeDesign::Classic(d), _) => format!("cols={} read={} write={} bytes={}", d.cfg.cols, d.cfg.read_ch, d.cfg.write_ch, d.cfg.bytes),
            (ProbeDesign::Concurrency(d), ProbeConfig::Concurrency(s)) => format!(
                "mode=concurrency direction={} active_channels_per_direction={} read_channels={} write_channels={} queued={} bd_bytes={} read_bytes={} write_bytes={}",
                s.dir.name(), s.n, d.cfg.read_channels, d.cfg.write_channels, d.cfg.queued_tasks, d.cfg.bd_bytes, d.cfg.read_total(), d.cfg.write_total()),
            _ => unreachable!("design built from a different probe family"),
        }
    }
}

/// Runs one configuration in one hardware context; returns whether it fully succeeded.
fn run_config(
    dev: &Device,
    probe: ProbeConfig,
    iters: usize,
    timeout_ms: u64,
    memory: Option<HipMemoryKind>,
    hazardous: bool,
) -> bool {
    let d = probe.build(hazardous);
    let ident = d.ident(&probe);
    let arg_bytes = d.arg_bytes();
    println!(
        "design {ident} axi={:?} pdi_B={} insts_B={} arg0_B={} arg1_B={}",
        probe.axi(),
        d.pdi().len(),
        d.insts().len(),
        arg_bytes[0],
        arg_bytes[1]
    );
    let axi = probe.axi();
    let memory_name = memory.map_or("native", HipMemoryKind::describe);
    let head = format!(
        "bw {ident} burst={} axcache={} axqos={} memory={memory_name} iters={iters}",
        axi.burst, axi.cache, axi.qos
    );
    // Keep HIP allocations alive until imported BOs and the hardware context have gone away.
    let mut hip_backings = Vec::new();
    let ntiles = dev.meta.cols as u32 * dev.meta.core.0 as u32;
    let ctx = match HwCtx::create(dev, ntiles, 2048) {
        Ok(c) => c,
        Err(e) => {
            println!("{head} FAIL create_hwctx: {e}");
            return false;
        }
    };
    let setup = (|| -> Result<(Bo, Bo, Vec<Bo>, Bo), String> {
        let pdi = dev.dev_bo(d.pdi())?;
        ctx.config_cu(&pdi)?;
        let insts = dev.dev_bo(d.insts())?;
        // A direction without channels still needs a BO for its (unused) argument slot.
        let mut bos = Vec::with_capacity(arg_bytes.len());
        for (slot, &arg_len) in arg_bytes.iter().enumerate() {
            let len = arg_len.max(4096);
            let bo = if let Some(kind) = memory {
                let mut initial = vec![0u8; len];
                if slot == 0 {
                    for (i, byte) in initial.iter_mut().enumerate() {
                        *byte = (i as u8).wrapping_mul(37).wrapping_add(11);
                    }
                }
                let backing = HipDmabufA::create_kind(&initial, kind)?;
                let bo = dev
                    .import_dmabuf(backing.fd(), backing.bytes())
                    .map_err(|e| format!("import {}: {e}", kind.describe()))?;
                println!("memory arg={slot} {}", backing.describe());
                hip_backings.push(backing);
                bo
            } else {
                let bo = dev.shmem_bo(len)?;
                if slot == 0 {
                    let bytes = unsafe { std::slice::from_raw_parts_mut(bo.host, bo.len) };
                    for (i, byte) in bytes.iter_mut().enumerate() {
                        *byte = (i as u8).wrapping_mul(37).wrapping_add(11);
                    }
                    bo.flush();
                }
                bo
            };
            bos.push(bo);
        }
        Ok((pdi, insts, bos, dev.cmd_bo()?))
    })();
    let (_pdi, insts, bos, mut cmd) = match setup {
        Ok(v) => v,
        Err(e) => {
            println!("{head} FAIL setup: {e}");
            return false;
        }
    };
    let refs: Vec<&Bo> = bos.iter().collect();
    let poison = || {
        let out = &bos[1];
        unsafe { std::ptr::write_bytes(out.host, 0xA5, out.len) };
        out.flush();
    };
    let write_total = d.write_total();
    let check = || -> String {
        bos[1].flush();
        let got = d.check_write(&bos[1].as_slice()[..write_total]);
        match got.first_bad {
            None => "ok".into(),
            Some((at, g, w)) => format!(
                "{} bad words, first at byte {at}: got {g:#010x} want {w:#010x}",
                got.bad_words
            ),
        }
    };
    let has_write = write_total > 0;
    let has_read = d.read_total() > 0;
    let mut first_us = 0.0;
    let mut warm: Vec<f64> = Vec::new();
    let mut write_state: Vec<String> = Vec::new();
    for it in 0..iters {
        let checked = has_write && (it == 0 || it == iters - 1);
        if checked {
            poison();
        }
        let t0 = Instant::now();
        let seq = match ctx.submit(&mut cmd, &insts, &refs) {
            Ok(s) => s,
            Err(e) => {
                println!("{head} FAIL submit {it}: {e}");
                return false;
            }
        };
        let state = ctx.wait(&cmd, seq, timeout_ms);
        let us = t0.elapsed().as_secs_f64() * 1e6;
        if !matches!(state, Ok(ERT_STATE_COMPLETED)) {
            println!("{head} FAIL submit {it}: state={state:?} after {us:.0} us (timeout {timeout_ms} ms)");
            return false;
        }
        if it == 0 {
            first_us = us
        } else {
            warm.push(us)
        }
        if checked {
            write_state.push(check());
        }
    }
    warm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = warm[0];
    let median = if warm.len() % 2 == 1 {
        warm[warm.len() / 2]
    } else {
        (warm[warm.len() / 2 - 1] + warm[warm.len() / 2]) / 2.0
    };
    let gbps = |bytes: usize| bytes as f64 / (best * 1e-6) / 1e9;
    let (read_g, write_g) = (gbps(d.read_total()), gbps(d.write_total()));
    let per = |g: f64, n: usize| {
        if n == 0 {
            "n/a".to_string()
        } else {
            fmt_gbps(g / n as f64)
        }
    };
    let exact = |s: Option<&String>| {
        if !has_write {
            "n/a".to_string()
        } else {
            (s.map(String::as_str) == Some("ok")).to_string()
        }
    };
    let write_ok = if !has_write {
        "n/a".to_string()
    } else {
        (write_state.iter().all(|s| s == "ok")).to_string()
    };
    let read_ok = if !has_read { "n/a" } else { "timing-only" };
    println!("first {ident} burst={} axcache={} axqos={} memory={memory_name} first_us={first_us:.1} (fresh context, submit 0)", axi.burst, axi.cache, axi.qos);
    println!("{head} best_us={best:.1} median_us={median:.1} read_GBps={} write_GBps={} per_read_ch_GBps={} per_write_ch_GBps={} bytes_per_aie_cycle_per_ch(@1.8GHz)={:.3} read_ok={read_ok} write_ok={write_ok} write_first_ok={} write_final_ok={}",
        fmt_gbps(read_g), fmt_gbps(write_g), per(read_g, d.read_channels()), per(write_g, d.write_channels()), d.bytes_per_channel() as f64 / (best * 1e-6) / AIE_HZ,
        exact(write_state.first()), exact(write_state.last()));
    if has_write && write_ok != "true" {
        println!("write_check submits(first,last)={write_state:?}");
        return false;
    }
    true
}

fn sweep_list() -> Vec<ProbeConfig> {
    const M: usize = 1 << 20;
    let c = |cols, read_ch, write_ch, bytes| {
        ProbeConfig::Classic(BwConfig {
            cols,
            read_ch,
            write_ch,
            bytes,
            axi: ShimAxi::default(),
        })
    };
    let mut v = Vec::new();
    for cols in [1, 2, 4, 8] {
        for read_ch in [1, 2] {
            v.push(c(cols, read_ch, 0, 16 * M));
        }
    }
    for cols in [1, 2, 4, 8] {
        for write_ch in [1, 2] {
            v.push(c(cols, 0, write_ch, 16 * M));
        }
    }
    v.push(c(8, 2, 2, 16 * M));
    v.push(c(8, 2, 0, M));
    v.push(c(8, 2, 0, 64 * M));
    v
}

fn sweep_axi_list() -> Vec<ProbeConfig> {
    let mut axis = vec![ShimAxi::default()];
    axis.extend((0..3).map(|burst| ShimAxi {
        burst,
        ..ShimAxi::default()
    }));
    let mut v = Vec::new();
    for (read_ch, write_ch) in [(2, 0), (0, 2)] {
        v.extend(axis.iter().map(|&axi| {
            ProbeConfig::Classic(BwConfig {
                cols: 8,
                read_ch,
                write_ch,
                bytes: 16 << 20,
                axi,
            })
        }));
    }
    v
}

/// Optional exact-value filters of the concurrency grid (`None` = every value).
#[derive(Clone, Copy, Default)]
struct ConcFilter {
    n: Option<usize>,
    q: Option<usize>,
    bd: Option<usize>,
    dir: Option<Dir>,
}

impl ConcFilter {
    fn describe(&self) -> String {
        let f = |o: Option<String>| o.unwrap_or_else(|| "all".into());
        format!(
            "active_channels_per_direction={} queued_tasks={} bd_bytes={} direction={}",
            f(self.n.map(|v| v.to_string())),
            f(self.q.map(|v| v.to_string())),
            f(self.bd.map(|v| v.to_string())),
            f(self.dir.map(|d| d.name().into()))
        )
    }
}

/// Full grid (N x Q x BD x direction) restricted by `filter`.
fn concurrency_list(filter: ConcFilter, axi: ShimAxi) -> Vec<ConcSpec> {
    let mut v = Vec::new();
    for n in CONC_CHANNELS
        .into_iter()
        .filter(|n| filter.n.map_or(true, |f| f == *n))
    {
        for q in CONC_QUEUED
            .into_iter()
            .filter(|q| filter.q.map_or(true, |f| f == *q))
        {
            for log2 in CONC_BD_MIN_LOG2..=CONC_BD_MAX_LOG2 {
                let bd = 1usize << log2;
                if filter.bd.map_or(false, |f| f != bd) {
                    continue;
                }
                for dir in Dir::ALL
                    .into_iter()
                    .filter(|d| filter.dir.map_or(true, |f| f == *d))
                {
                    v.push(ConcSpec { dir, n, q, bd, axi });
                }
            }
        }
    }
    v
}

const CONC_SELECTORS: [&str; 5] = [
    "--sweep-concurrency",
    "--active-channels",
    "--queued-tasks",
    "--bd-bytes",
    "--direction",
];
const LEGACY_SELECTORS: [&str; 6] = [
    "--sweep",
    "--sweep-axi",
    "--cols",
    "--read",
    "--write",
    "--bytes",
];

/// Strict argument check for concurrency mode: no legacy selectors/geometry, no hazardous opt-in, no unknown,
/// duplicate, or valueless flags and no stray positional arguments (nothing is silently ignored).
fn check_concurrency_args(args: &[String]) {
    const VALUE_FLAGS: [&str; 10] = [
        "--active-channels",
        "--queued-tasks",
        "--bd-bytes",
        "--direction",
        "--memory",
        "--iters",
        "--timeout-ms",
        "--burst",
        "--axcache",
        "--axqos",
    ];
    let mut seen: Vec<&str> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--allow-hazardous-axi" {
            usage("--allow-hazardous-axi is never accepted for concurrency probes (single or sweep); AxCACHE must be 2 and AxQoS 0");
        }
        if LEGACY_SELECTORS.contains(&a) {
            usage(&format!(
                "{a} is a legacy probe flag and cannot be combined with concurrency flags ({})",
                CONC_SELECTORS.join(" ")
            ));
        }
        if seen.contains(&a) {
            usage(&format!("{a} given more than once"));
        }
        if a == "--sweep-concurrency" {
            seen.push(a);
        } else if VALUE_FLAGS.contains(&a) {
            match args.get(i + 1) {
                Some(v) if !v.starts_with("--") => i += 1,
                _ => usage(&format!("{a} needs a value")),
            }
            seen.push(a);
        } else {
            usage(&format!(
                "unknown or unexpected argument {a:?} for a concurrency probe"
            ));
        }
        i += 1;
    }
}

fn parse_concurrency(args: &[String]) -> Vec<ProbeConfig> {
    check_concurrency_args(args);
    let axi_field = |name: &str, default: u32, max: u32| -> u32 {
        let v = opt_num(args, name).map_or(default, |v| u32::try_from(v).unwrap_or(u32::MAX));
        if v > max {
            usage(&format!("{name} must be 0..={max}"));
        }
        v
    };
    let d = ShimAxi::default();
    let axi = ShimAxi {
        burst: axi_field("--burst", d.burst, 3),
        cache: axi_field("--axcache", d.cache, 15),
        qos: axi_field("--axqos", d.qos, 15),
    };
    if let Err(e) = axi.validate_vendor_word5() {
        usage(&e);
    }

    let n = opt_num(args, "--active-channels");
    let q = opt_num(args, "--queued-tasks");
    let bd = opt_str(args, "--bd-bytes").map(|s| parse_bytes("--bd-bytes", s));
    let dir = opt_str(args, "--direction").map(|s| {
        Dir::parse(s)
            .unwrap_or_else(|| usage(&format!("--direction expects read|write|mixed, got {s:?}")))
    });
    if let Some(n) = n {
        if !CONC_CHANNELS.contains(&n) {
            usage(&format!(
                "--active-channels must be one of {CONC_CHANNELS:?}, got {n}"
            ));
        }
    }
    if let Some(q) = q {
        if !CONC_QUEUED.contains(&q) {
            usage(&format!(
                "--queued-tasks must be one of {CONC_QUEUED:?}, got {q}"
            ));
        }
    }
    if let Some(bd) = bd {
        if !bd.is_power_of_two()
            || !(1usize << CONC_BD_MIN_LOG2..=1usize << CONC_BD_MAX_LOG2).contains(&bd)
        {
            usage(&format!(
                "--bd-bytes must be a power of two in {}..={}, got {bd}",
                1usize << CONC_BD_MIN_LOG2,
                1usize << CONC_BD_MAX_LOG2
            ));
        }
    }

    let sweep = args.iter().any(|a| a == "--sweep-concurrency");
    let configs = if sweep {
        let filter = ConcFilter { n, q, bd, dir };
        let configs: Vec<ProbeConfig> = concurrency_list(filter, axi)
            .into_iter()
            .map(ProbeConfig::Concurrency)
            .collect();
        println!("npu-bw: concurrency sweep selected {} of {CONC_FULL} configs; filters: {} burst={} axcache={} axqos={}{}",
            configs.len(), filter.describe(), axi.burst, axi.cache, axi.qos,
            if configs.len() == CONC_FULL { " (full grid)" } else { " (REDUCED sweep)" });
        configs
    } else {
        let need = |name: &str, present: bool| {
            if !present {
                usage(&format!("missing {name} (--active-channels, --queued-tasks, --bd-bytes and --direction are all required without --sweep-concurrency)"));
            }
        };
        need("--active-channels", n.is_some());
        need("--queued-tasks", q.is_some());
        need("--bd-bytes", bd.is_some());
        need("--direction", dir.is_some());
        let spec = ConcSpec {
            dir: dir.unwrap(),
            n: n.unwrap(),
            q: q.unwrap(),
            bd: bd.unwrap(),
            axi,
        };
        println!("npu-bw: concurrency single config selected (1 of {CONC_FULL} grid points; no sweep): direction={} active_channels_per_direction={} queued_tasks={} bd_bytes={} burst={} axcache={} axqos={}",
            spec.dir.name(), spec.n, spec.q, spec.bd, axi.burst, axi.cache, axi.qos);
        vec![ProbeConfig::Concurrency(spec)]
    };
    if configs.is_empty() {
        usage("concurrency selection is empty");
    }
    configs
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help") {
        println!("usage: npu-bw --cols C --read R --write W --bytes N[K|M|G]\n       npu-bw --sweep|--sweep-axi\n       npu-bw --active-channels N --queued-tasks Q --bd-bytes N[K|M] --direction read|write|mixed\n       npu-bw --sweep-concurrency [--active-channels N] [--queued-tasks Q] [--bd-bytes N[K|M]] [--direction read|write|mixed]\noptions: --memory native|vmm-host|vmm-host-uncached --burst 0..3 --axcache 0..15 --axqos 0..15 --allow-hazardous-axi --iters I --timeout-ms T\nMemory selects both read and write BOs. Read traffic is timing-only (no returned data); writes are pattern-checked first and last (write_first_ok / write_final_ok).");
        println!("Concurrency probes: --active-channels N is channels PER ACTIVE DIRECTION, N in 1,2,4,8,16 (read = N read channels; write = N write channels; mixed = N read AND N write = 2N total; output: active_channels_per_direction, read_channels, write_channels), Q in 1,2,4 queued tasks per channel, bd-bytes power of two 4K..1M. --sweep-concurrency runs all 405 combinations; each of the four single-config flags then filters it (the selected count and filters are printed). Legacy flags (--cols --read --write --bytes --sweep --sweep-axi) and unknown flags are rejected with concurrency flags.");
        println!("Default emitters and automatic sweeps require AxCACHE 2 / AxQoS 0. --allow-hazardous-axi is explicit single classic-probe opt-in only and is ALWAYS rejected for concurrency probes (single or sweep): other word-5 attributes can wedge the platform (AxQoS 15 caused SMU resume -22 on Halo). Declare hazardous experiments before running them.");
        return;
    }
    let concurrency = args.iter().any(|a| CONC_SELECTORS.contains(&a.as_str()));
    // Concurrency mode is validated strictly first so no flag is silently dropped.
    let configs_conc = if concurrency {
        Some(parse_concurrency(&args))
    } else {
        None
    };
    let memory = match opt_str(&args, "--memory").unwrap_or("native") {
        "native" => None,
        value => Some(HipMemoryKind::parse(value).unwrap_or_else(|e| usage(&e))),
    };
    let iters = opt_num(&args, "--iters").unwrap_or(8);
    if iters < 2 {
        usage("--iters must be >= 2 (submit 0 is the fresh-context submit, the rest are warm)");
    }
    let timeout_ms = opt_num(&args, "--timeout-ms").unwrap_or(5000) as u64;
    if timeout_ms == 0 {
        usage("--timeout-ms must be >= 1");
    }
    let hazardous = args.iter().any(|a| a == "--allow-hazardous-axi");
    if hazardous && args.iter().any(|a| a == "--sweep" || a == "--sweep-axi") {
        usage(
            "--allow-hazardous-axi is allowed only for one explicit probe, not an automatic sweep",
        );
    }
    let configs = if let Some(c) = configs_conc {
        c
    } else if args.iter().any(|a| a == "--sweep") {
        sweep_list()
    } else if args.iter().any(|a| a == "--sweep-axi") {
        sweep_axi_list()
    } else {
        let need =
            |name: &str| opt_str(&args, name).unwrap_or_else(|| usage(&format!("missing {name}")));
        let axi_field = |name: &str, default: u32, max: u32| -> u32 {
            let v = opt_num(&args, name).map_or(default, |v| v as u32);
            if v > max {
                usage(&format!("{name} must be 0..={max}"));
            }
            v
        };
        let d = ShimAxi::default();
        let cfg = BwConfig {
            cols: need("--cols")
                .parse()
                .unwrap_or_else(|_| usage("--cols expects an integer")),
            read_ch: opt_num(&args, "--read").unwrap_or(0),
            write_ch: opt_num(&args, "--write").unwrap_or(0),
            bytes: parse_bytes("--bytes", need("--bytes")),
            axi: ShimAxi {
                burst: axi_field("--burst", d.burst, 3),
                cache: axi_field("--axcache", d.cache, 15),
                qos: axi_field("--axqos", d.qos, 15),
            },
        };
        vec![ProbeConfig::Classic(cfg)]
    };
    // Reject every unsafe or invalid configuration before the device is opened.
    for cfg in &configs {
        if let Err(e) = cfg.validate(hazardous) {
            usage(&e);
        }
    }

    let mut dev = Device::open().expect("open");
    // Only the PDI and instruction stream live in the DEV heap (dev_bo); arguments are SHMEM BOs outside it.
    dev.map_heap(64 << 20).expect("heap");
    let start = Instant::now();
    let mut failed = 0;
    for cfg in &configs {
        if !run_config(&dev, *cfg, iters, timeout_ms, memory, hazardous) {
            failed += 1;
        }
    }
    println!(
        "npu-bw: {} configs, {failed} failed, {:.1} s",
        configs.len(),
        start.elapsed().as_secs_f64()
    );
    if failed > 0 {
        std::process::exit(1);
    }
}
