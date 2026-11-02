//! `hipfire calibrate`: measure this host's link, CPU and GPU rates once and
//! store them as the fingerprinted `[calibration]` table in `config.toml`.
//!
//! The engine's measured-rate selectors read that table at model load and fall
//! back to their own load-time probe when it is missing or stale. Nothing here
//! leaves the host: the command opens the local GPUs, writes one TOML table, and
//! prints what it did.
//!
//! The record format, fingerprint and staleness rules live in
//! [`hipfire_config::calibration`]; this module is the measurement driver plus the
//! `--show` / `--check` / `--clear` views over a stored record. The formatting and
//! verdict logic is pure (record + live host facts in, text/JSON/verdict out) so it
//! is tested without a GPU.

use crate::{live_daemon_pids, Paths};
use anyhow::{anyhow, bail, Context, Result};
use clap::Args;
use hipfire_arch_qwen4::d2_default;
use hipfire_arch_qwen4::host_link::{
    link_from_calibration, link_route_choices, probe_pinned_h2d_gbps, probe_zero_copy_gbps,
    HostLink, RouteChoice,
};
use hipfire_config::calibration::{
    self, Calibration, CpuReadRate, DetectedGpu, DeviceCalibration, DeviceIdentity, EngineRate,
    HostFacts, Lookup, PcieLink, UnmeasuredDevice, PROBE_VERSION,
};
use rdna_compute::Gpu;
use std::fmt::Write as _;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// A calibration run that would start an optional GPU probe past this much wall
/// time skips it instead, so `hipfire calibrate` stays a "takes up to a minute"
/// command on hosts with many or slow devices.
const BUDGET: Duration = Duration::from_secs(60);

/// CPU read-bandwidth buffer: a quarter of `MemAvailable`, capped, floored.
const CPU_READ_MAX_BYTES: u64 = 4 << 30;
const CPU_READ_MIN_BYTES: u64 = 256 << 20;
const CPU_READ_GRANULE: u64 = 1 << 20;
const CPU_READ_REPS: usize = 3;

#[derive(Args, Debug, Default)]
pub(crate) struct CalibrateArgs {
    /// Print the stored calibration and whether it still matches this host
    /// (measures nothing, opens no GPU).
    #[arg(long, group = "mode")]
    show: bool,
    /// Exit 0 when a fresh calibration covers every detected GPU, else exit 1.
    #[arg(long, group = "mode")]
    check: bool,
    /// Remove the stored calibration; the engine goes back to probing at load.
    #[arg(long, group = "mode")]
    clear: bool,
    /// Machine-readable JSON (measure or `--show`).
    #[arg(short = 'j', long, conflicts_with_all = ["check", "clear"])]
    json: bool,
}

pub(crate) fn calibrate_command(paths: &Paths, args: CalibrateArgs) -> Result<()> {
    if args.show {
        show_command(paths, args.json)
    } else if args.check {
        check_command(paths)
    } else if args.clear {
        clear_command(paths)
    } else {
        run_command(paths, args.json)
    }
}

// ---------------------------------------------------------------------------
// Freshness and per-device rows (pure)
// ---------------------------------------------------------------------------

/// How one device's stored entry relates to the live host.
#[derive(Clone, Debug, PartialEq)]
enum Freshness {
    Fresh,
    /// The record no longer describes this host; carries the one-line reason.
    Stale(String),
    /// The device is present but has no usable entry; carries the reason.
    Missing(String),
    /// Present at calibration time but not measured by that run (the record's
    /// `unmeasured` list); carries the recorded reason.
    Unmeasured(String),
    /// The record names a device this host does not currently list.
    Absent,
}

impl Freshness {
    fn status(&self) -> &'static str {
        match self {
            Freshness::Fresh => "fresh",
            Freshness::Stale(_) => "stale",
            Freshness::Missing(_) => "missing",
            Freshness::Unmeasured(_) => "unmeasured",
            Freshness::Absent => "absent",
        }
    }

    fn reason(&self) -> Option<&str> {
        match self {
            Freshness::Stale(r) | Freshness::Missing(r) | Freshness::Unmeasured(r) => Some(r),
            Freshness::Fresh | Freshness::Absent => None,
        }
    }
}

impl From<&Lookup> for Freshness {
    fn from(lookup: &Lookup) -> Self {
        match lookup {
            Lookup::Fresh { .. } => Freshness::Fresh,
            Lookup::Stale(reason) => Freshness::Stale(reason.clone()),
            Lookup::Missing(reason) => Freshness::Missing(reason.clone()),
        }
    }
}

/// One device as `--show` and the post-run summary present it.
struct DeviceRow {
    pci_bdf: String,
    freshness: Freshness,
    /// The stored entry, when the record has one for this device.
    measured: Option<DeviceCalibration>,
    /// The device as this host reports it now, when the host lists it.
    live: Option<DeviceIdentity>,
    /// What the engine's selectors would pick from this entry; empty unless the
    /// entry is fresh, discrete, and carries a usable pinned H2D rate.
    routes: Vec<RouteChoice>,
    /// The D2 default's standing for this device; `None` unless the entry is
    /// fresh.
    d2: Option<D2Show>,
}

/// What `--show` may say about the D2 default (decode of host-placed routed
/// experts on the CPU, see [`d2_default`]) on this device. The decision needs
/// the loaded model's real CPU expert arm and live link samples, so a stored
/// record cannot make it: this is [`d2_default::calibration_verdict`], the
/// selector's own function with no model arm.
struct D2Show {
    /// `load` (decided at model load from the real arm) or `off` (unified
    /// memory: no link to beat). Never `on`.
    status: &'static str,
    text: String,
}

fn d2_show(lookup: &Lookup) -> Option<D2Show> {
    let Lookup::Fresh { device, .. } = lookup else {
        return None;
    };
    let (status, text) = d2_default::calibration_verdict(device.unified);
    Some(D2Show { status, text })
}

fn identity_of(gpu: &DetectedGpu) -> DeviceIdentity {
    DeviceIdentity {
        pci_bdf: gpu.pci_bdf.clone(),
        arch: gpu.arch.clone(),
        vram_bytes: gpu.vram_bytes.unwrap_or(0),
        link: gpu.link.clone(),
    }
}

/// Live identities of the GPUs the kernel lists (KFD topology + sysfs).
fn detected_identities() -> Vec<DeviceIdentity> {
    calibration::detect_gpus().iter().map(identity_of).collect()
}

fn host_freshness(record: Option<&Calibration>, host: &HostFacts) -> Freshness {
    match record {
        None => Freshness::Missing("no calibration stored".to_string()),
        Some(record) => match calibration::host_staleness(record, host) {
            Some(stale) => Freshness::Stale(stale.to_string()),
            None => Freshness::Fresh,
        },
    }
}

/// Route choices for one device whose lookup is `lookup`: the same functions the
/// engine calls, fed the same calibrated link. Only discrete devices have a link
/// policy, so a unified device (or one with no usable pinned rate) has none.
fn routes_for(lookup: &Lookup, arch: &str) -> Vec<RouteChoice> {
    match link_from_calibration(lookup) {
        Some(link @ HostLink::Discrete { .. }) => link_route_choices(arch, link),
        _ => Vec::new(),
    }
}

/// Every device that matters: the record's devices first (in record order), then
/// live GPUs the record has no entry for. `live` is what this host reports now.
fn device_rows(
    record: Option<&Calibration>,
    host: &HostFacts,
    live: &[DeviceIdentity],
) -> Vec<DeviceRow> {
    let mut rows = Vec::new();
    let stored = record.map(|r| r.devices.as_slice()).unwrap_or(&[]);
    for entry in stored {
        match live.iter().find(|id| id.pci_bdf == entry.pci_bdf) {
            None => rows.push(DeviceRow {
                pci_bdf: entry.pci_bdf.clone(),
                freshness: Freshness::Absent,
                measured: Some(entry.clone()),
                live: None,
                routes: Vec::new(),
                d2: None,
            }),
            Some(id) => {
                let lookup = calibration::lookup_in(record, host, id);
                rows.push(DeviceRow {
                    pci_bdf: entry.pci_bdf.clone(),
                    freshness: Freshness::from(&lookup),
                    measured: Some(entry.clone()),
                    live: Some(id.clone()),
                    routes: routes_for(&lookup, &id.arch),
                    d2: d2_show(&lookup),
                });
            }
        }
    }
    for id in live {
        if stored.iter().any(|entry| entry.pci_bdf == id.pci_bdf) {
            continue;
        }
        let freshness = match record.and_then(|r| unmeasured_entry(r, host, id)) {
            Some(entry) => Freshness::Unmeasured(entry.reason.clone()),
            None => Freshness::from(&calibration::lookup_in(record, host, id)),
        };
        rows.push(DeviceRow {
            pci_bdf: id.pci_bdf.clone(),
            freshness,
            measured: None,
            live: Some(id.clone()),
            routes: Vec::new(),
            d2: None,
        });
    }
    rows
}

/// The record's `unmeasured` entry for `id`, when the host still matches the
/// record and the entry names the same PCI address and architecture.
fn unmeasured_entry<'a>(
    record: &'a Calibration,
    host: &HostFacts,
    id: &DeviceIdentity,
) -> Option<&'a UnmeasuredDevice> {
    if calibration::host_staleness(record, host).is_some() || !record.lists_unmeasured(id) {
        return None;
    }
    record
        .unmeasured
        .iter()
        .find(|entry| entry.pci_bdf == id.pci_bdf)
}

/// `--check` verdict: `Ok(line)` when a stored record is fresh for this host and
/// every detected GPU has a fresh entry or is the same GPU the run listed as
/// unmeasured (re-running would not measure it either); no detected GPU: the
/// record's existence and host freshness suffice. `Err(line)` with the one
/// reason otherwise.
fn check_verdict(
    record: Option<&Calibration>,
    host: &HostFacts,
    live: &[DeviceIdentity],
) -> std::result::Result<String, String> {
    let Some(record) = record else {
        return Err("no calibration stored; run `hipfire calibrate`".to_string());
    };
    if let Some(stale) = calibration::host_staleness(record, host) {
        return Err(format!("calibration is stale: {stale}"));
    }
    let mut measured = 0;
    for id in live {
        match calibration::lookup_in(Some(record), host, id) {
            Lookup::Fresh { .. } => measured += 1,
            Lookup::Missing(_) if record.lists_unmeasured(id) => {}
            Lookup::Stale(reason) => {
                return Err(format!("{}: calibration is stale: {reason}", id.pci_bdf))
            }
            Lookup::Missing(reason) => {
                return Err(format!("{}: not calibrated: {reason}", id.pci_bdf))
            }
        }
    }
    let unmeasured = live.len() - measured;
    Ok(format!(
        "calibration is fresh ({measured} GPU{} measured{})",
        if measured == 1 { "" } else { "s" },
        if unmeasured == 0 {
            String::new()
        } else {
            format!(", {unmeasured} not measurable by this run")
        }
    ))
}

// ---------------------------------------------------------------------------
// Rendering (pure)
// ---------------------------------------------------------------------------

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn link_text(link: &PcieLink) -> String {
    let mut text = format!(
        "{:.1} GT/s x{} (bottleneck {})",
        link.gts, link.width, link.bottleneck_bdf
    );
    if let Some(current) = link.current_gts {
        if (current - link.gts).abs() >= 0.01 {
            let _ = write!(text, ", running at {current:.1} GT/s");
        }
    }
    text
}

fn host_lines(host: &HostFacts) -> String {
    let mut out = String::new();
    let cores = match host.physical_cores {
        Some(physical) => format!("{} logical / {physical} physical cores", host.logical_cpus),
        None => format!("{} logical cores", host.logical_cpus),
    };
    let _ = writeln!(
        out,
        "cpu      {} — {cores}, {:.1} GiB RAM",
        host.cpu_model,
        gib(host.ram_total_bytes)
    );
    if !host.isa.is_empty() {
        let _ = writeln!(out, "isa      {}", host.isa.join(" "));
    }
    out
}

/// The summary `--show` prints and a measuring run ends with.
fn render_summary(record: &Calibration, host: &Freshness, rows: &[DeviceRow]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "calibration measured {} by hipfire {} (probe v{}, {:.1} s)",
        record.measured_at, record.hipfire_version, record.probe_version, record.duration_s
    );
    if let Freshness::Stale(reason) = host {
        let _ = writeln!(out, "host     stale: {reason}");
    }
    out.push_str(&host_lines(&record.host));
    match &record.cpu_read {
        Some(cpu) => {
            let _ = writeln!(
                out,
                "cpu read {:.1} GB/s ({} threads, {:.1} GiB, median of {})",
                cpu.gbps,
                cpu.threads,
                gib(cpu.bytes),
                cpu.reps
            );
        }
        None => out.push_str("cpu read not measured\n"),
    }
    if rows.is_empty() {
        out.push_str("gpus     none\n");
    }
    for row in rows {
        out.push_str(&render_row(row));
    }
    out
}

fn render_row(row: &DeviceRow) -> String {
    let mut out = String::new();
    let tag = match &row.freshness {
        Freshness::Fresh => "fresh".to_string(),
        Freshness::Stale(reason) => format!("stale: {reason}"),
        Freshness::Missing(reason) => format!("not calibrated: {reason}"),
        Freshness::Unmeasured(reason) => format!("not measured: {reason}"),
        Freshness::Absent => "not detected on this host".to_string(),
    };
    match &row.measured {
        Some(dev) => {
            let _ = writeln!(
                out,
                "gpu {}  {}  {:.1} GiB  {}  [{tag}]",
                row.pci_bdf,
                dev.arch,
                gib(dev.vram_bytes),
                if dev.unified {
                    "unified memory"
                } else {
                    "discrete"
                }
            );
            if let Some(link) = &dev.link {
                let _ = writeln!(out, "    link          {}", link_text(link));
            }
            let rate = |value: Option<f64>| match value {
                Some(gbps) => format!("{gbps:.2} GB/s"),
                None => "not measured".to_string(),
            };
            if !dev.unified {
                let _ = writeln!(out, "    pinned H2D    {}", rate(dev.pinned_h2d_gbps));
                let _ = writeln!(out, "    zero-copy     {}", rate(dev.zero_copy_gbps));
            }
            for e in &dev.engine {
                let _ = writeln!(
                    out,
                    "    engine        {} k={} m={} ({} pool, {} reps): cpu {:.1} GB/s, gpu {:.1} GB/s",
                    e.format, e.k, e.m, e.pool, e.reps, e.cpu_gbps, e.gpu_gbps
                );
            }
        }
        None => match &row.live {
            Some(id) => {
                let link = match &id.link {
                    Some(link) => link_text(link),
                    None => "link unknown".to_string(),
                };
                let _ = writeln!(
                    out,
                    "gpu {}  {}  {:.1} GiB  {link}  [{tag}]",
                    row.pci_bdf,
                    id.arch,
                    gib(id.vram_bytes)
                );
            }
            None => {
                let _ = writeln!(out, "gpu {}  [{tag}]", row.pci_bdf);
            }
        },
    }
    if !row.routes.is_empty() {
        out.push_str("    engine routes from this calibration:\n");
        for route in &row.routes {
            let _ = writeln!(
                out,
                "      {} = {} — {}",
                route.name, route.value, route.why
            );
        }
    }
    if let Some(d2) = &row.d2 {
        let _ = writeln!(out, "    D2 default    {}", d2.text);
    }
    match row.freshness {
        Freshness::Stale(_) | Freshness::Missing(_) => {
            out.push_str(
                "    the engine probes this device at model load until `hipfire calibrate` measures it\n",
            );
        }
        Freshness::Unmeasured(_) => {
            out.push_str("    the engine probes this device at model load\n");
        }
        Freshness::Fresh | Freshness::Absent => {}
    }
    out
}

fn show_json(
    record: Option<&Calibration>,
    host: &Freshness,
    rows: &[DeviceRow],
) -> serde_json::Result<serde_json::Value> {
    let devices: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| {
            serde_json::json!({
                "pci_bdf": row.pci_bdf,
                "status": row.freshness.status(),
                "reason": row.freshness.reason(),
                "routes": row.routes.iter().map(|r| serde_json::json!({
                    "name": r.name,
                    "value": r.value,
                    "why": r.why,
                })).collect::<Vec<_>>(),
                "d2_default": row.d2.as_ref().map(|d2| serde_json::json!({
                    "status": d2.status,
                    "text": d2.text,
                })),
            })
        })
        .collect();
    Ok(serde_json::json!({
        "calibration": match record {
            Some(record) => serde_json::to_value(record)?,
            None => serde_json::Value::Null,
        },
        "host": { "status": host.status(), "reason": host.reason() },
        "devices": devices,
    }))
}

// ---------------------------------------------------------------------------
// --show / --check / --clear
// ---------------------------------------------------------------------------

fn show_command(paths: &Paths, json: bool) -> Result<()> {
    let record = calibration::load(&paths.config).context("reading the stored calibration")?;
    let host = calibration::detect_host();
    let live = detected_identities();
    let rows = device_rows(record.as_ref(), &host, &live);
    let host_state = host_freshness(record.as_ref(), &host);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&show_json(record.as_ref(), &host_state, &rows)?)?
        );
        return Ok(());
    }
    match &record {
        Some(record) => print!("{}", render_summary(record, &host_state, &rows)),
        None => {
            println!(
                "no calibration stored in {}; run `hipfire calibrate` to measure this host \
                 (until then the engine probes at model load)",
                paths.config.config_toml.display()
            );
            print!("{}", host_lines(&host));
            for row in &rows {
                print!("{}", render_row(row));
            }
        }
    }
    Ok(())
}

fn check_command(paths: &Paths) -> Result<()> {
    let verdict = match calibration::load(&paths.config) {
        Ok(record) => check_verdict(
            record.as_ref(),
            &calibration::detect_host(),
            &detected_identities(),
        ),
        Err(e) => Err(format!("unreadable calibration: {e}")),
    };
    match verdict {
        Ok(line) => {
            println!("{line}");
            Ok(())
        }
        Err(line) => {
            println!("{line}");
            std::process::exit(1);
        }
    }
}

fn clear_command(paths: &Paths) -> Result<()> {
    let removed = calibration::clear(&paths.config).context("clearing the stored calibration")?;
    let path = paths.config.config_toml.display();
    if removed {
        println!("removed the calibration from {path}; the engine probes at model load again");
    } else {
        println!("no calibration stored in {path}; nothing to clear");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Measuring run
// ---------------------------------------------------------------------------

/// CPU read-bandwidth buffer size: a quarter of `MemAvailable` (so the buffer
/// never pressures the host), capped at 4 GiB, floored to a whole MiB and never
/// below 256 MiB (a smaller buffer fits cache and measures the wrong thing).
fn cpu_read_bytes(available: Option<u64>) -> u64 {
    let quarter = available.unwrap_or(0) / 4;
    let capped = quarter.min(CPU_READ_MAX_BYTES);
    (capped - capped % CPU_READ_GRANULE).max(CPU_READ_MIN_BYTES)
}

fn budget_spent(elapsed: Duration) -> bool {
    elapsed >= BUDGET
}

/// Whether an optional probe may still start; says so on stderr when it may not.
fn within_budget(started: Instant, probe: &str, bdf: &str) -> bool {
    if budget_spent(started.elapsed()) {
        eprintln!("skipped {probe} on {bdf}: 60 s budget spent");
        return false;
    }
    true
}

fn usable(gbps: f64) -> bool {
    gbps.is_finite() && gbps > 0.0
}

/// Print what the detection phase found (stderr: progress, not the result).
fn announce_detection(host: &HostFacts, gpus: &[DetectedGpu]) {
    eprint!("{}", host_lines(host));
    if gpus.is_empty() {
        eprintln!("gpus     none detected by the kernel (CPU rates are still measured)");
    }
    for gpu in gpus {
        let link = match &gpu.link {
            Some(link) => link_text(link),
            None => "link unknown".to_string(),
        };
        let vram = match gpu.vram_bytes {
            Some(bytes) => format!("{:.1} GiB", gib(bytes)),
            None => "VRAM unknown".to_string(),
        };
        eprintln!("gpu      {}  {}  {vram}  {link}", gpu.pci_bdf, gpu.arch);
    }
}

/// Engine split probes: the offload-bench matrix (every CPU-decodable dense
/// format, rayon global pool as the CPU arm), one format at a time so the 60 s
/// budget is checked before each and a format that errors is reported and skipped
/// while the others still run. The D2 default's cost comparison is not
/// recorded here: it needs the loaded model's real CPU arm and is measured at
/// model load ([`d2_default`]).
fn engine_probes(gpu: &mut Gpu, started: Instant, bdf: &str) -> Vec<EngineRate> {
    use hipfire_runtime::offload_calibrate::{
        calibrate, format_names, CalibrateOptions, DEFAULT_BUFFER_MB, DEFAULT_REPS,
    };
    let mut rates = Vec::new();
    for name in format_names() {
        if !within_budget(started, &format!("engine split probe ({name})"), bdf) {
            continue;
        }
        let t = Instant::now();
        let opts = CalibrateOptions {
            format: Some(name.to_string()),
            k: None,
            buffer_mb: DEFAULT_BUFFER_MB,
            reps: DEFAULT_REPS,
        };
        match calibrate(gpu, &opts) {
            Ok(reports) => {
                for report in reports {
                    eprintln!(
                        "engine split {} on {bdf}: cpu {:.1} GB/s, gpu {:.1} GB/s ({:.1} s)",
                        report.format,
                        report.cpu_bytes_per_s / 1e9,
                        report.gpu_bytes_per_s / 1e9,
                        t.elapsed().as_secs_f64()
                    );
                    rates.push(EngineRate {
                        format: report.format,
                        k: report.k as u64,
                        m: report.m as u64,
                        pool: "global".to_string(),
                        reps: DEFAULT_REPS as u32,
                        cpu_gbps: report.cpu_bytes_per_s / 1e9,
                        gpu_gbps: report.gpu_bytes_per_s / 1e9,
                    });
                }
            }
            Err(e) => eprintln!("engine split probe ({name}) on {bdf} failed: {}", e.message),
        }
    }
    rates
}

/// Run one rate probe: deadline-checked, timed, reported. A failure leaves the
/// field `None` and never aborts the run.
fn rate_probe(
    started: Instant,
    probe: &str,
    bdf: &str,
    run: impl FnOnce() -> std::result::Result<f64, String>,
) -> Option<f64> {
    if !within_budget(started, probe, bdf) {
        return None;
    }
    let t = Instant::now();
    match run() {
        Ok(gbps) if usable(gbps) => {
            eprintln!(
                "{probe} on {bdf}: {gbps:.2} GB/s ({:.1} s)",
                t.elapsed().as_secs_f64()
            );
            Some(gbps)
        }
        Ok(gbps) => {
            eprintln!("{probe} on {bdf}: unusable figure {gbps}; not recorded");
            None
        }
        Err(e) => {
            eprintln!("{probe} on {bdf} failed: {e}");
            None
        }
    }
}

/// Measure one open device. `None` when it cannot even be identified.
fn calibrate_device(mut gpu: Gpu, ordinal: i32, started: Instant) -> Option<DeviceCalibration> {
    let raw_bdf = match gpu.hip.device_pci_bus_id(gpu.device_id) {
        Ok(raw) => raw,
        Err(e) => {
            eprintln!(
                "device {ordinal}: cannot read its PCI bus id ({} [code {}]); skipped",
                e.message, e.code
            );
            return None;
        }
    };
    let Some(bdf) = calibration::normalize_bdf(&raw_bdf) else {
        eprintln!("device {ordinal}: unrecognized PCI bus id '{raw_bdf}'; skipped");
        return None;
    };
    let vram = gpu
        .bind_thread()
        .ok()
        .and_then(|()| gpu.hip.get_vram_info().ok())
        .map(|(_, total)| total as u64);
    let id = DeviceIdentity::live(&bdf, &gpu.arch, vram);
    let unified = gpu.is_uma();
    let mut device = DeviceCalibration {
        pci_bdf: id.pci_bdf,
        arch: id.arch,
        vram_bytes: id.vram_bytes,
        unified,
        link: id.link,
        pinned_h2d_gbps: None,
        zero_copy_gbps: None,
        engine: Vec::new(),
    };
    if unified {
        eprintln!("{bdf}: unified memory, no host link to probe");
        return Some(device);
    }
    device.pinned_h2d_gbps =
        rate_probe(started, "pinned H2D", &bdf, || probe_pinned_h2d_gbps(&gpu));
    device.zero_copy_gbps = rate_probe(started, "zero-copy read", &bdf, || {
        probe_zero_copy_gbps(&mut gpu)
    });
    device.engine = engine_probes(&mut gpu, started, &bdf);
    Some(device)
}

/// Measure every HIP-visible device, one at a time (each `Gpu` is dropped before
/// the next opens). No GPU, or an ordinal that will not open, is reported and the
/// run goes on: a CPU-only calibration is a valid record.
fn calibrate_gpus(started: Instant) -> Vec<DeviceCalibration> {
    let mut devices = Vec::new();
    let first = match Gpu::init_with_device(0) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!(
                "no GPU opened ({} [code {}]); recording zero devices",
                e.message, e.code
            );
            return devices;
        }
    };
    let count = first
        .hip
        .device_count()
        .unwrap_or_else(|e| {
            eprintln!(
                "cannot count devices ({} [code {}]); measuring device 0 only",
                e.message, e.code
            );
            1
        })
        .max(1);
    let mut open = Some(first);
    for ordinal in 0..count {
        let gpu = match open.take() {
            Some(gpu) => gpu,
            None => match Gpu::init_with_device(ordinal) {
                Ok(gpu) => gpu,
                Err(e) => {
                    eprintln!(
                        "device {ordinal}: cannot open ({} [code {}]); skipped",
                        e.message, e.code
                    );
                    continue;
                }
            },
        };
        if let Some(device) = calibrate_device(gpu, ordinal, started) {
            devices.push(device);
        }
    }
    devices
}

fn run_command(paths: &Paths, json: bool) -> Result<()> {
    let live = live_daemon_pids(paths);
    if !live.is_empty() {
        bail!(
            "a hipfire daemon is live (pid {}); stop it first (`hipfire stop`) — calibrate \
             measures the devices directly, and a concurrent daemon makes every rate \
             meaningless",
            live.iter()
                .map(|pid| pid.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let started = Instant::now();

    let host = calibration::detect_host();
    let gpus = calibration::detect_gpus();
    announce_detection(&host, &gpus);
    eprintln!("detected in {:.1} s", started.elapsed().as_secs_f64());

    let t = Instant::now();
    let available = hipfire_config::host_memory().map(|m| m.available_physical_bytes);
    let bytes = usize::try_from(cpu_read_bytes(available))
        .map_err(|_| anyhow!("CPU read buffer does not fit this platform's address space"))?;
    let bw = hipfire_dispatch::cpu_exec::splice_read_bandwidth(bytes, CPU_READ_REPS);
    eprintln!(
        "cpu read: {:.1} GB/s over {:.1} GiB, {} threads ({:.1} s)",
        bw.gbps,
        gib(bw.bytes),
        bw.threads,
        t.elapsed().as_secs_f64()
    );

    let devices = calibrate_gpus(started);
    let unmeasured = gpus
        .iter()
        .filter(|gpu| !devices.iter().any(|device| device.pci_bdf == gpu.pci_bdf))
        .map(|gpu| {
            eprintln!(
                "gpu {} ({}): not measured (not visible to HIP in this run, or it failed to open)",
                gpu.pci_bdf, gpu.arch
            );
            UnmeasuredDevice {
                pci_bdf: gpu.pci_bdf.clone(),
                arch: gpu.arch.clone(),
                reason: "not visible to HIP in this run, or it failed to open".to_string(),
            }
        })
        .collect();

    let duration_s = started.elapsed().as_secs_f64();
    let record = Calibration {
        probe_version: PROBE_VERSION,
        measured_at: calibration::utc_rfc3339(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        ),
        hipfire_version: env!("HIPFIRE_BUILD_VERSION").to_string(),
        duration_s,
        host: host.clone(),
        cpu_read: Some(CpuReadRate {
            gbps: bw.gbps,
            threads: bw.threads,
            bytes: bw.bytes,
            reps: bw.reps,
        }),
        devices,
        unmeasured,
    };
    calibration::store(&paths.config, &record).context("writing the calibration")?;
    eprintln!("wrote {}", paths.config.config_toml.display());

    // The devices were identified live a moment ago, so their identities are the
    // record's own: the summary reads back the same way `--show` will.
    let mut identities: Vec<DeviceIdentity> = record
        .devices
        .iter()
        .map(|d| DeviceIdentity {
            pci_bdf: d.pci_bdf.clone(),
            arch: d.arch.clone(),
            vram_bytes: d.vram_bytes,
            link: d.link.clone(),
        })
        .collect();
    identities.extend(
        gpus.iter()
            .filter(|gpu| record.lists_unmeasured(&identity_of(gpu)))
            .map(identity_of),
    );
    let rows = device_rows(Some(&record), &host, &identities);
    let host_state = host_freshness(Some(&record), &host);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&show_json(Some(&record), &host_state, &rows)?)?
        );
    } else {
        print!("{}", render_summary(&record, &host_state, &rows));
        println!("calibrated in {duration_s:.1} s");
    }
    if budget_spent(Duration::from_secs_f64(duration_s)) {
        eprintln!("warning: calibration took longer than the 60 s budget");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cli;
    use clap::Parser;

    fn host() -> HostFacts {
        HostFacts {
            cpu_model: "Test CPU".to_string(),
            logical_cpus: 32,
            physical_cores: Some(16),
            isa: vec!["avx2".to_string(), "fma".to_string()],
            ram_total_bytes: 128 << 30,
        }
    }

    fn link() -> PcieLink {
        PcieLink {
            gts: 16.0,
            width: 4,
            current_gts: Some(16.0),
            bottleneck_bdf: "0000:65:00.0".to_string(),
        }
    }

    fn device(bdf: &str) -> DeviceCalibration {
        DeviceCalibration {
            pci_bdf: bdf.to_string(),
            arch: "gfx1100".to_string(),
            vram_bytes: 24 << 30,
            unified: false,
            link: Some(link()),
            pinned_h2d_gbps: Some(7.1),
            zero_copy_gbps: Some(7.0),
            engine: vec![EngineRate {
                format: "mq4g256v2".to_string(),
                k: 2560,
                m: 123,
                pool: "global".to_string(),
                reps: 5,
                cpu_gbps: 40.1,
                gpu_gbps: 6.1,
            }],
        }
    }

    fn record(devices: Vec<DeviceCalibration>) -> Calibration {
        Calibration {
            probe_version: PROBE_VERSION,
            measured_at: "2026-10-18T14:03:22Z".to_string(),
            hipfire_version: "0.4.3".to_string(),
            duration_s: 21.4,
            host: host(),
            cpu_read: Some(CpuReadRate {
                gbps: 113.5,
                threads: 16,
                bytes: 4 << 30,
                reps: 3,
            }),
            devices,
            unmeasured: Vec::new(),
        }
    }

    fn live(bdf: &str) -> DeviceIdentity {
        let d = device(bdf);
        DeviceIdentity {
            pci_bdf: d.pci_bdf,
            arch: d.arch,
            vram_bytes: d.vram_bytes,
            link: d.link,
        }
    }

    #[test]
    fn check_fails_without_a_record() {
        let verdict = check_verdict(None, &host(), &[live("0000:67:00.0")]);
        assert!(verdict.unwrap_err().contains("hipfire calibrate"));
    }

    #[test]
    fn check_passes_for_a_fresh_record_covering_every_gpu() {
        let rec = record(vec![device("0000:67:00.0")]);
        let verdict = check_verdict(Some(&rec), &host(), &[live("0000:67:00.0")]);
        assert_eq!(
            verdict,
            Ok("calibration is fresh (1 GPU measured)".to_string())
        );
    }

    #[test]
    fn check_passes_with_no_detected_gpu_when_record_and_host_are_fresh() {
        let rec = record(Vec::new());
        assert!(check_verdict(Some(&rec), &host(), &[]).is_ok());
    }

    #[test]
    fn check_fails_for_a_stale_device() {
        let rec = record(vec![device("0000:67:00.0")]);
        let mut changed = live("0000:67:00.0");
        changed.arch = "gfx1201".to_string();
        let line = check_verdict(Some(&rec), &host(), &[changed]).unwrap_err();
        assert!(
            line.contains("0000:67:00.0") && line.contains("stale"),
            "{line}"
        );
    }

    #[test]
    fn check_fails_when_a_detected_gpu_is_missing_from_the_record() {
        let rec = record(vec![device("0000:67:00.0")]);
        let verdict = check_verdict(
            Some(&rec),
            &host(),
            &[live("0000:67:00.0"), live("0000:03:00.0")],
        );
        assert!(verdict.unwrap_err().contains("0000:03:00.0"));
    }

    #[test]
    fn check_fails_when_the_host_changed() {
        let rec = record(vec![device("0000:67:00.0")]);
        let mut other = host();
        other.logical_cpus = 8;
        let line = check_verdict(Some(&rec), &other, &[live("0000:67:00.0")]).unwrap_err();
        assert!(line.contains("stale"), "{line}");
    }

    #[test]
    fn check_passes_for_a_gpu_the_run_could_not_measure() {
        let mut rec = record(vec![device("0000:67:00.0")]);
        rec.unmeasured.push(UnmeasuredDevice {
            pci_bdf: "0000:c0:00.0".to_string(),
            arch: "gfx1151".to_string(),
            reason: "not visible to HIP in this run, or it failed to open".to_string(),
        });
        let mut igpu = live("0000:c0:00.0");
        igpu.arch = "gfx1151".to_string();
        let verdict = check_verdict(Some(&rec), &host(), &[live("0000:67:00.0"), igpu.clone()]);
        assert_eq!(
            verdict,
            Ok("calibration is fresh (1 GPU measured, 1 not measurable by this run)".to_string())
        );
        let rows = device_rows(Some(&rec), &host(), &[live("0000:67:00.0"), igpu.clone()]);
        assert_eq!(rows[1].freshness.status(), "unmeasured");
        assert!(rows[1].routes.is_empty());
        // A different GPU in that slot is not the one the run skipped.
        igpu.arch = "gfx1150".to_string();
        assert!(check_verdict(Some(&rec), &host(), &[igpu]).is_err());
    }

    #[test]
    fn clap_rejects_conflicting_modes() {
        assert!(Cli::try_parse_from(["hipfire", "calibrate", "--show", "--clear"]).is_err());
        assert!(Cli::try_parse_from(["hipfire", "calibrate", "--check", "--show"]).is_err());
        assert!(Cli::try_parse_from(["hipfire", "calibrate", "--check", "--json"]).is_err());
        assert!(Cli::try_parse_from(["hipfire", "calibrate", "--clear", "--json"]).is_err());
        assert!(Cli::try_parse_from(["hipfire", "calibrate", "--show", "--json"]).is_ok());
        assert!(Cli::try_parse_from(["hipfire", "calibrate"]).is_ok());
    }

    #[test]
    fn show_json_lists_routes_for_a_fresh_narrow_device() {
        let rec = record(vec![device("0000:67:00.0")]);
        let rows = device_rows(Some(&rec), &host(), &[live("0000:67:00.0")]);
        assert_eq!(rows[0].freshness, Freshness::Fresh);
        let value = show_json(Some(&rec), &host_freshness(Some(&rec), &host()), &rows).unwrap();
        assert_eq!(value["calibration"]["probe_version"], PROBE_VERSION);
        let dev = &value["devices"][0];
        assert_eq!(dev["pci_bdf"], "0000:67:00.0");
        assert_eq!(dev["status"], "fresh");
        let routes = dev["routes"].as_array().unwrap();
        assert!(
            !routes.is_empty(),
            "a fresh discrete device lists its routes"
        );
        for route in routes {
            assert!(route["name"].is_string());
            assert!(route["value"].is_string());
            assert!(route["why"].is_string());
        }
    }

    #[test]
    fn show_lists_stale_missing_and_absent_devices_without_routes() {
        let rec = record(vec![device("0000:67:00.0"), device("0000:05:00.0")]);
        let mut changed = live("0000:67:00.0");
        changed.vram_bytes = 16 << 30;
        let rows = device_rows(Some(&rec), &host(), &[changed, live("0000:03:00.0")]);
        let status: Vec<(&str, &str)> = rows
            .iter()
            .map(|r| (r.pci_bdf.as_str(), r.freshness.status()))
            .collect();
        assert_eq!(
            status,
            [
                ("0000:67:00.0", "stale"),
                ("0000:05:00.0", "absent"),
                ("0000:03:00.0", "missing"),
            ]
        );
        assert!(rows.iter().all(|r| r.routes.is_empty()));
        let text = render_summary(&rec, &Freshness::Fresh, &rows);
        assert!(text.contains("probes this device at model load"), "{text}");
    }

    #[test]
    fn summary_names_the_link_and_rates() {
        let rec = record(vec![device("0000:67:00.0")]);
        let rows = device_rows(Some(&rec), &host(), &[live("0000:67:00.0")]);
        let text = render_summary(&rec, &Freshness::Fresh, &rows);
        assert!(
            text.contains("16.0 GT/s x4 (bottleneck 0000:65:00.0)"),
            "{text}"
        );
        assert!(text.contains("7.10 GB/s"), "{text}");
        assert!(text.contains("113.5 GB/s"), "{text}");
    }

    #[test]
    fn cpu_read_buffer_is_a_quarter_of_available_within_bounds() {
        assert_eq!(cpu_read_bytes(None), 256 << 20);
        assert_eq!(cpu_read_bytes(Some(1 << 30)), 256 << 20);
        assert_eq!(cpu_read_bytes(Some(8 << 30)), 2 << 30);
        assert_eq!(cpu_read_bytes(Some(512 << 30)), 4 << 30);
        // Floors to a whole MiB.
        assert_eq!(cpu_read_bytes(Some((8 << 30) + 12345)), 2 << 30);
    }

    #[test]
    fn budget_is_spent_at_sixty_seconds() {
        assert!(!budget_spent(Duration::from_millis(59_999)));
        assert!(budget_spent(Duration::from_secs(60)));
    }

    /// `device` plus a legacy splice-pool engine entry, as a record written
    /// before the D2 selector measured the real arm at load carries it.
    fn device_with_legacy_splice(bdf: &str, cpu_gbps: f64, link_gbps: f64) -> DeviceCalibration {
        let mut dev = device(bdf);
        dev.engine.push(EngineRate {
            format: "mq4g256v2".to_string(),
            k: 5120,
            m: 100_000,
            pool: "splice".to_string(),
            reps: 5,
            cpu_gbps,
            gpu_gbps: link_gbps,
        });
        dev
    }

    /// Stored rates, legacy splice entries included, never state a D2 verdict:
    /// the decision needs the model's real CPU arm and live link samples, so
    /// `--show` says it is decided at load, whatever the record's CPU and
    /// link figures are.
    #[test]
    fn show_never_claims_a_d2_decision_from_stored_rates() {
        // j1760's XTX legacy entry (CPU 105.9 / link 4.25 GB/s) and its
        // inverse, and a record with no D2 entry at all.
        for dev in [
            device_with_legacy_splice("0000:67:00.0", 105.9, 4.25),
            device_with_legacy_splice("0000:67:00.0", 2.0, 10.0),
            device("0000:67:00.0"),
        ] {
            let rec = record(vec![dev]);
            let rows = device_rows(Some(&rec), &host(), &[live("0000:67:00.0")]);
            let d2 = rows[0].d2.as_ref().expect("a fresh device has a standing");
            assert_eq!(d2.status, "load");
            assert!(
                d2.text.contains("calibration rates alone cannot enable it"),
                "{}",
                d2.text
            );
            let text = render_summary(&rec, &Freshness::Fresh, &rows);
            assert!(text.contains("D2 default"), "{text}");
            assert!(!text.contains("D2 default    Flash-Next"), "{text}");
            assert!(!text.contains("ON ("), "{text}");
            let value = show_json(Some(&rec), &host_freshness(Some(&rec), &host()), &rows).unwrap();
            let json = &value["devices"][0]["d2_default"];
            assert_eq!(json["status"], "load");
            assert!(json.get("cpu_gbps").is_none() && json.get("stored_k").is_none());
        }
    }

    #[test]
    fn show_reports_unified_off_and_no_standing_for_stale_devices() {
        // Unified memory has no link to beat.
        let mut unified = device_with_legacy_splice("0000:67:00.0", 105.9, 4.25);
        unified.unified = true;
        let rec = record(vec![unified]);
        let rows = device_rows(Some(&rec), &host(), &[live("0000:67:00.0")]);
        assert_eq!(rows[0].d2.as_ref().unwrap().status, "off");
        // A stale, missing or absent device states no D2 standing.
        let rec = record(vec![device("0000:67:00.0")]);
        let mut changed = live("0000:67:00.0");
        changed.vram_bytes = 16 << 30;
        let rows = device_rows(Some(&rec), &host(), &[changed, live("0000:03:00.0")]);
        assert!(rows.iter().all(|row| row.d2.is_none()));
        let value = show_json(Some(&rec), &host_freshness(Some(&rec), &host()), &rows).unwrap();
        assert!(value["devices"][0]["d2_default"].is_null());
    }
}
