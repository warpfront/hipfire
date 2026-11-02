//! Loud warning when the GPU's VRAM is full and allocations spill into
//! system RAM (GTT; "shared GPU memory" on Windows).
//!
//! A spilled model keeps running, 2-3x slower, with no error anywhere. The
//! monitor samples the amdgpu sysfs counters of the card the daemon drives
//! (resolved by `bench_power::resolve_device`, the same resolution the bench
//! `power_device` report uses) once the model load finishes, then again after
//! every completed request, and warns when GTT grew while VRAM is full.
//!
//! Linux (amdgpu sysfs) only. Anywhere the counters are unreadable (no
//! amdgpu, no permission, non-Linux) the monitor is simply absent.
//! A Windows signal would need the DXGI budget query matched to
//! the daemon's adapter, which the CLI cannot do without a device resolver
//! of its own, so Windows is a no-op.

use crate::bench_power::PowerDevice;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const MIB: u64 = 1024 * 1024;
/// GTT growth over the post-load baseline that counts as a spill, and the
/// further growth required before the warning repeats.
pub(crate) const GTT_GROWTH_BYTES: u64 = 256 * MIB;
/// VRAM counts as full within this much of its total.
pub(crate) const VRAM_HEADROOM_BYTES: u64 = 512 * MIB;

/// One reading of the three amdgpu counters, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpillSample {
    pub gtt_used: u64,
    pub vram_used: u64,
    pub vram_total: u64,
}

/// A warning to print; `growth_bytes` is what the next re-warn is measured
/// from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SpillWarning {
    pub growth_bytes: u64,
    pub message: String,
}

fn gtt_growth(baseline: &SpillSample, sample: &SpillSample) -> u64 {
    sample.gtt_used.saturating_sub(baseline.gtt_used)
}

fn vram_full(sample: &SpillSample) -> bool {
    sample.vram_total > 0
        && sample.vram_used >= sample.vram_total.saturating_sub(VRAM_HEADROOM_BYTES)
}

/// GTT grew by at least the threshold over `baseline` while VRAM is full.
pub(crate) fn is_spilled(baseline: &SpillSample, sample: &SpillSample) -> bool {
    gtt_growth(baseline, sample) >= GTT_GROWTH_BYTES && vram_full(sample)
}

/// The warning, if `sample` is a spill that has not been reported yet.
/// `last_warned` is the GTT growth (bytes over `baseline`) at the previous
/// warning, `0` when none was printed: the warning repeats only once GTT grew
/// another [`GTT_GROWTH_BYTES`] past it.
pub(crate) fn decide(
    baseline: &SpillSample,
    sample: &SpillSample,
    last_warned: u64,
) -> Option<SpillWarning> {
    if !is_spilled(baseline, sample) {
        return None;
    }
    let growth = gtt_growth(baseline, sample);
    if last_warned > 0 && growth < last_warned.saturating_add(GTT_GROWTH_BYTES) {
        return None;
    }
    Some(SpillWarning {
        growth_bytes: growth,
        message: format!(
            "WARNING: GPU memory is spilling to system RAM: GTT +{} MiB since load while VRAM is full ({}/{} MiB). Decode/prefill can be 2-3x slower. Lower memory.max_seq, use a smaller KV mode (kv_cache), a smaller tier, or turn speculation off.",
            growth / MIB,
            sample.vram_used / MIB,
            sample.vram_total / MIB,
        ),
    })
}

#[cfg(target_os = "linux")]
fn read_bytes(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Read the counters under `dir` (the card's sysfs device directory).
#[cfg(target_os = "linux")]
fn read_sample(dir: &Path) -> Option<SpillSample> {
    Some(SpillSample {
        gtt_used: read_bytes(&dir.join("mem_info_gtt_used"))?,
        vram_used: read_bytes(&dir.join("mem_info_vram_used"))?,
        vram_total: read_bytes(&dir.join("mem_info_vram_total"))?,
    })
}

#[cfg(not(target_os = "linux"))]
fn read_sample(_dir: &Path) -> Option<SpillSample> {
    None
}

/// Spill detector for one resident model on one card.
#[derive(Debug)]
pub(crate) struct SpillMonitor {
    dir: PathBuf,
    baseline: SpillSample,
    last_warned: u64,
}

impl SpillMonitor {
    /// Attach to `device` and take the baseline. `None` when the counters
    /// are not readable: the monitor is silently off.
    pub(crate) fn attach(device: &PowerDevice) -> Option<Self> {
        let baseline = read_sample(&device.dir)?;
        Some(Self {
            dir: device.dir.clone(),
            baseline,
            last_warned: 0,
        })
    }

    /// Sample now; the warning text when a (new) spill is detected.
    pub(crate) fn check(&mut self) -> Option<String> {
        let sample = read_sample(&self.dir)?;
        let warning = decide(&self.baseline, &sample, self.last_warned)?;
        self.last_warned = warning.growth_bytes;
        Some(warning.message)
    }

    /// The bench report's per-run `spill` object, from a fresh sample.
    pub(crate) fn report_json(&self) -> Value {
        match read_sample(&self.dir) {
            Some(sample) => json!({
                "baseline_gtt": self.baseline.gtt_used,
                "gtt": sample.gtt_used,
                "vram_used": sample.vram_used,
                "vram_total": sample.vram_total,
                "spilled": is_spilled(&self.baseline, &sample),
            }),
            None => Value::Null,
        }
    }
}

/// Resolve the card behind the daemon's logical device 0 and attach, with the
/// bench's resolution (`hardware.devices`, else `ROCR_VISIBLE_DEVICES`, else
/// the first ROCr agent of the daemon's arch from `diag`). Call right after
/// the daemon's `loaded` ack.
pub(crate) fn attach_for_engine(
    engine: &hipfire_client::Engine,
    devices_spec: Option<&str>,
) -> Option<SpillMonitor> {
    let rocr = std::env::var("ROCR_VISIBLE_DEVICES").ok();
    let filtered = devices_spec.is_some() || rocr.as_deref().is_some_and(|v| !v.trim().is_empty());
    let arch = if filtered {
        None
    } else {
        engine
            .request(&json!({ "type": "diag" }))
            .ok()
            .and_then(|diag| diag.get("arch").and_then(Value::as_str).map(str::to_owned))
    };
    let device = crate::bench_power::resolve_device(
        Path::new("/"),
        devices_spec,
        rocr.as_deref(),
        arch.as_deref(),
    )
    .ok()?;
    SpillMonitor::attach(&device)
}

/// Print the warning to stderr (once per [`SpillMonitor::check`] hit).
pub(crate) fn emit(message: &str) {
    eprintln!("{message}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(gtt_mib: u64, vram_mib: u64, total_mib: u64) -> SpillSample {
        SpillSample {
            gtt_used: gtt_mib * MIB,
            vram_used: vram_mib * MIB,
            vram_total: total_mib * MIB,
        }
    }

    #[test]
    fn no_growth_no_warning() {
        let base = s(300, 16_000, 16_368);
        assert_eq!(decide(&base, &base, 0), None);
        // Below the threshold, even with VRAM full.
        assert_eq!(decide(&base, &s(300 + 255, 16_300, 16_368), 0), None);
    }

    #[test]
    fn growth_without_full_vram_does_not_warn() {
        let base = s(300, 8_000, 16_368);
        assert_eq!(decide(&base, &s(300 + 4096, 12_000, 16_368), 0), None);
        // 513 MiB of headroom is not full.
        assert_eq!(decide(&base, &s(300 + 4096, 16_368 - 513, 16_368), 0), None);
        // Unknown total never counts as full.
        assert_eq!(decide(&base, &s(300 + 4096, 0, 0), 0), None);
    }

    #[test]
    fn growth_with_full_vram_warns() {
        let base = s(300, 15_000, 16_368);
        let warning = decide(&base, &s(300 + 1024, 16_368 - 512, 16_368), 0).unwrap();
        assert_eq!(warning.growth_bytes, 1024 * MIB);
        assert_eq!(
            warning.message,
            "WARNING: GPU memory is spilling to system RAM: GTT +1024 MiB since load while VRAM is full (15856/16368 MiB). Decode/prefill can be 2-3x slower. Lower memory.max_seq, use a smaller KV mode (kv_cache), a smaller tier, or turn speculation off."
        );
        // Exactly the threshold counts.
        assert!(decide(&base, &s(300 + 256, 16_368, 16_368), 0).is_some());
    }

    #[test]
    fn rewarns_only_after_another_256_mib() {
        let base = s(300, 15_000, 16_368);
        let first = decide(&base, &s(300 + 512, 16_368, 16_368), 0).unwrap();
        let last = first.growth_bytes;
        assert_eq!(decide(&base, &s(300 + 512, 16_368, 16_368), last), None);
        assert_eq!(
            decide(&base, &s(300 + 512 + 255, 16_368, 16_368), last),
            None
        );
        let again = decide(&base, &s(300 + 512 + 256, 16_368, 16_368), last).unwrap();
        assert_eq!(again.growth_bytes, 768 * MIB);
        assert!(again.message.contains("GTT +768 MiB"));
    }

    #[test]
    fn gtt_shrinking_below_baseline_is_not_growth() {
        let base = s(2000, 16_368, 16_368);
        assert_eq!(decide(&base, &s(100, 16_368, 16_368), 0), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn monitor_reads_sysfs_and_reports_json() {
        let dir = std::env::temp_dir().join(format!("hipfire-spill-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let put = |gtt: u64, used: u64| {
            std::fs::write(dir.join("mem_info_gtt_used"), format!("{}\n", gtt * MIB)).unwrap();
            std::fs::write(dir.join("mem_info_vram_used"), format!("{}\n", used * MIB)).unwrap();
            std::fs::write(
                dir.join("mem_info_vram_total"),
                format!("{}\n", 16_368 * MIB),
            )
            .unwrap();
        };
        put(100, 15_000);
        let mut monitor = SpillMonitor {
            dir: dir.clone(),
            baseline: read_sample(&dir).unwrap(),
            last_warned: 0,
        };
        assert_eq!(monitor.check(), None);
        put(100 + 600, 16_300);
        let message = monitor.check().unwrap();
        assert!(message.starts_with("WARNING: GPU memory is spilling to system RAM: GTT +600 MiB"));
        assert_eq!(monitor.check(), None);
        let report = monitor.report_json();
        assert_eq!(report["spilled"], true);
        assert_eq!(report["baseline_gtt"], 100 * MIB);
        assert_eq!(report["gtt"], 700 * MIB);
        std::fs::remove_file(dir.join("mem_info_gtt_used")).unwrap();
        assert_eq!(monitor.check(), None);
        assert_eq!(monitor.report_json(), Value::Null);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
