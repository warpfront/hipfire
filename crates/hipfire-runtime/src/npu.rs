//! Host-side probes for XDNA/Ryzen AI accelerator nodes.
//!
//! This module is deliberately runtime-light: it does not link XRT and it does
//! not submit work. It reports whether the host has an accessible `/dev/accel`
//! node and whether `xrt-smi examine` can enumerate it under the current
//! process limits. The actual compute smoke lives in `scripts/strix-npu-smoke.sh`.

use serde::Serialize;
use std::fs;
use std::io;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const DEFAULT_ACCEL_PATH: &str = "/dev/accel/accel0";
pub const DEFAULT_MIN_MEMLOCK_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, Serialize)]
pub struct DeviceNodeProbe {
    pub path: PathBuf,
    pub exists: bool,
    pub is_char_device: bool,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
}

impl DeviceNodeProbe {
    pub fn probe(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        match fs::metadata(&path) {
            Ok(meta) => Ok(Self {
                path,
                exists: true,
                is_char_device: meta.file_type().is_char_device(),
                mode: meta.permissions().mode() & 0o7777,
                uid: meta.uid(),
                gid: meta.gid(),
            }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Self {
                path,
                exists: false,
                is_char_device: false,
                mode: 0,
                uid: 0,
                gid: 0,
            }),
            Err(e) => Err(e),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct MemlockLimit {
    /// `None` means unlimited.
    pub soft_bytes: Option<u64>,
    /// `None` means unlimited.
    pub hard_bytes: Option<u64>,
}

impl MemlockLimit {
    pub fn current() -> io::Result<Self> {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        let rc = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut lim) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            soft_bytes: rlim_to_option(lim.rlim_cur),
            hard_bytes: rlim_to_option(lim.rlim_max),
        })
    }

    pub fn soft_at_least(&self, bytes: u64) -> bool {
        match self.soft_bytes {
            Some(v) => v >= bytes,
            None => true,
        }
    }
}

fn rlim_to_option(v: libc::rlim_t) -> Option<u64> {
    if v == libc::RLIM_INFINITY {
        None
    } else {
        Some(v as u64)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandProbe {
    pub program: String,
    pub found: bool,
    pub status: Option<i32>,
    pub output: String,
}

impl CommandProbe {
    pub fn ok(&self) -> bool {
        self.status == Some(0)
    }
}

pub fn run_xrt_smi_examine() -> CommandProbe {
    let program = "xrt-smi".to_string();
    match Command::new(&program)
        .arg("examine")
        .stdin(Stdio::null())
        .output()
    {
        Ok(out) => {
            let mut text = String::new();
            text.push_str(&String::from_utf8_lossy(&out.stdout));
            text.push_str(&String::from_utf8_lossy(&out.stderr));
            CommandProbe {
                program,
                found: true,
                status: out.status.code(),
                output: text,
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => CommandProbe {
            program,
            found: false,
            status: None,
            output: "xrt-smi not found".to_string(),
        },
        Err(e) => CommandProbe {
            program,
            found: true,
            status: None,
            output: e.to_string(),
        },
    }
}

#[derive(Debug, Clone, Serialize)]
pub enum NpuReadiness {
    Ready,
    MissingDeviceNode,
    DeviceNodeNotChar,
    XrtSmiMissing,
    MemlockTooLow,
    XrtSmiFailed,
}

#[derive(Debug, Clone, Serialize)]
pub struct NpuProbeReport {
    pub accel: DeviceNodeProbe,
    pub memlock: MemlockLimit,
    pub min_memlock_bytes: u64,
    pub xrt_smi: CommandProbe,
    pub readiness: NpuReadiness,
    pub advice: Vec<String>,
}

impl NpuProbeReport {
    pub fn ready(&self) -> bool {
        matches!(self.readiness, NpuReadiness::Ready)
    }
}

pub fn probe_strix_npu(
    accel_path: impl AsRef<Path>,
    min_memlock_bytes: u64,
) -> io::Result<NpuProbeReport> {
    let accel = DeviceNodeProbe::probe(accel_path)?;
    let memlock = MemlockLimit::current()?;
    let xrt_smi = run_xrt_smi_examine();
    let (readiness, advice) = classify_npu_readiness(&accel, memlock, &xrt_smi, min_memlock_bytes);

    Ok(NpuProbeReport {
        accel,
        memlock,
        min_memlock_bytes,
        xrt_smi,
        readiness,
        advice,
    })
}

fn classify_npu_readiness(
    accel: &DeviceNodeProbe,
    memlock: MemlockLimit,
    xrt_smi: &CommandProbe,
    min_memlock_bytes: u64,
) -> (NpuReadiness, Vec<String>) {
    let mut advice = Vec::new();

    let readiness = if !accel.exists {
        advice.push(format!(
            "expected XDNA device node at {}",
            accel.path.display()
        ));
        NpuReadiness::MissingDeviceNode
    } else if !accel.is_char_device {
        advice.push("device node exists but is not a character device".to_string());
        NpuReadiness::DeviceNodeNotChar
    } else if !xrt_smi.found {
        advice.push("install XRT utilities so xrt-smi is available".to_string());
        NpuReadiness::XrtSmiMissing
    } else if !xrt_smi.ok()
        && xrt_smi.output.contains("Resource temporarily unavailable")
        && !memlock.soft_at_least(min_memlock_bytes)
    {
        advice.push(format!(
            "raise RLIMIT_MEMLOCK to at least {min_memlock_bytes} bytes for the hipfire daemon/user"
        ));
        advice.push(
            "temporary smoke wrapper: sudo prlimit --memlock=134217728:134217728 -- setpriv --reuid=$(id -u) --regid=$(id -g) --groups=$(id -G | tr ' ' ',') <command>"
                .to_string(),
        );
        NpuReadiness::MemlockTooLow
    } else if !xrt_smi.ok() {
        advice.push("xrt-smi examine failed; inspect output for driver/runtime state".to_string());
        NpuReadiness::XrtSmiFailed
    } else {
        NpuReadiness::Ready
    };

    (readiness, advice)
}

#[derive(Debug, Clone)]
pub struct StrixNpuSmokeConfig {
    pub script_path: PathBuf,
    pub mlir_aie_dir: Option<PathBuf>,
    pub device_name: String,
    pub min_memlock_bytes: u64,
    pub run_mlir_aie_smoke: bool,
}

impl StrixNpuSmokeConfig {
    pub fn default_probe_only() -> Self {
        Self {
            script_path: default_strix_npu_smoke_script(),
            mlir_aie_dir: None,
            device_name: "npu2".to_string(),
            min_memlock_bytes: DEFAULT_MIN_MEMLOCK_BYTES,
            run_mlir_aie_smoke: false,
        }
    }

    pub fn default_compute() -> Self {
        let mut cfg = Self::default_probe_only();
        cfg.run_mlir_aie_smoke = true;
        cfg
    }
}

pub fn default_strix_npu_smoke_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(|repo| repo.join("scripts/strix-npu-smoke.sh"))
        .unwrap_or_else(|| PathBuf::from("scripts/strix-npu-smoke.sh"))
}

#[derive(Debug, Clone)]
pub struct CommandPlan {
    pub program: PathBuf,
    pub env: Vec<(String, String)>,
}

impl CommandPlan {
    pub fn env_value(&self, key: &str) -> Option<&str> {
        self.env
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

pub fn plan_strix_npu_smoke(cfg: &StrixNpuSmokeConfig) -> CommandPlan {
    let mut env = vec![
        (
            "HIPFIRE_NPU_RUN_MLIR_AIE_SMOKE".to_string(),
            if cfg.run_mlir_aie_smoke { "1" } else { "0" }.to_string(),
        ),
        (
            "HIPFIRE_NPU_MIN_MEMLOCK_BYTES".to_string(),
            cfg.min_memlock_bytes.to_string(),
        ),
        (
            "HIPFIRE_NPU_DEVICE_NAME".to_string(),
            cfg.device_name.clone(),
        ),
    ];
    if let Some(path) = &cfg.mlir_aie_dir {
        env.push((
            "MLIR_AIE_DIR".to_string(),
            path.to_string_lossy().to_string(),
        ));
    }
    CommandPlan {
        program: cfg.script_path.clone(),
        env,
    }
}

pub fn run_strix_npu_smoke(cfg: &StrixNpuSmokeConfig) -> CommandProbe {
    let plan = plan_strix_npu_smoke(cfg);
    let mut cmd = Command::new(&plan.program);
    cmd.stdin(Stdio::null());
    for (key, value) in &plan.env {
        cmd.env(key, value);
    }

    match cmd.output() {
        Ok(out) => {
            let mut text = String::new();
            text.push_str(&String::from_utf8_lossy(&out.stdout));
            text.push_str(&String::from_utf8_lossy(&out.stderr));
            CommandProbe {
                program: plan.program.display().to_string(),
                found: true,
                status: out.status.code(),
                output: text,
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => CommandProbe {
            program: plan.program.display().to_string(),
            found: false,
            status: None,
            output: "strix NPU smoke script not found".to_string(),
        },
        Err(e) => CommandProbe {
            program: plan.program.display().to_string(),
            found: true,
            status: None,
            output: e.to_string(),
        },
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StrixNpuSmokeParsed {
    pub xrt_detected: bool,
    pub device_name: Option<String>,
    pub avg_npu_time_us: Option<u64>,
    pub compute_passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct StrixNpuSmokeReport {
    pub command: CommandProbe,
    pub parsed: StrixNpuSmokeParsed,
}

pub fn run_strix_npu_smoke_report(cfg: &StrixNpuSmokeConfig) -> StrixNpuSmokeReport {
    let command = run_strix_npu_smoke(cfg);
    let parsed = parse_strix_npu_smoke_output(&command.output);
    StrixNpuSmokeReport { command, parsed }
}

pub fn parse_strix_npu_smoke_output(output: &str) -> StrixNpuSmokeParsed {
    let device_name = extract_xrt_device_name(output);
    StrixNpuSmokeParsed {
        xrt_detected: device_name.is_some() || output.contains("Device(s) Present"),
        device_name,
        avg_npu_time_us: extract_avg_npu_time_us(output),
        compute_passed: output.lines().any(|line| line.trim() == "PASS!"),
    }
}

fn extract_xrt_device_name(output: &str) -> Option<String> {
    for line in output.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("|[") {
            continue;
        }
        let mut cols = trimmed.split('|').map(str::trim);
        let _empty = cols.next();
        let _bdf = cols.next();
        if let Some(name) = cols.next() {
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

fn extract_avg_npu_time_us(output: &str) -> Option<u64> {
    for line in output.lines() {
        let trimmed = line.trim();
        let Some(value) = trimmed.strip_prefix("Avg NPU time:") else {
            continue;
        };
        let value = value.trim();
        let digits: String = value.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            return None;
        }
        return digits.parse().ok();
    }
    None
}

#[derive(Debug, Clone, Serialize)]
pub struct AcceleratorSummary {
    pub kind: &'static str,
    pub ready: bool,
    pub compute_validated: bool,
    pub daemon_usable: bool,
    pub device_name: Option<String>,
    pub avg_npu_time_us: Option<u64>,
    pub advice: Vec<String>,
}

pub fn summarize_strix_npu_accelerator(
    probe: &NpuProbeReport,
    smoke: &StrixNpuSmokeReport,
) -> AcceleratorSummary {
    let ready = probe.ready();
    let command_ok = smoke.command.status == Some(0);
    let compute_validated = command_ok && smoke.parsed.compute_passed;
    AcceleratorSummary {
        kind: "strix-xdna-npu",
        ready,
        compute_validated,
        daemon_usable: ready && compute_validated,
        device_name: smoke.parsed.device_name.clone(),
        avg_npu_time_us: smoke.parsed.avg_npu_time_us,
        advice: probe.advice.clone(),
    }
}

pub trait AcceleratorNode {
    fn kind(&self) -> &'static str;
    fn probe(&self) -> io::Result<NpuProbeReport>;
    fn validate_compute(&self) -> io::Result<AcceleratorSummary>;
}

#[derive(Debug, Clone)]
pub struct StrixNpuAcceleratorNode {
    pub accel_path: PathBuf,
    pub min_memlock_bytes: u64,
    pub smoke_config: StrixNpuSmokeConfig,
}

impl Default for StrixNpuAcceleratorNode {
    fn default() -> Self {
        Self {
            accel_path: PathBuf::from(DEFAULT_ACCEL_PATH),
            min_memlock_bytes: DEFAULT_MIN_MEMLOCK_BYTES,
            smoke_config: StrixNpuSmokeConfig::default_compute(),
        }
    }
}

impl AcceleratorNode for StrixNpuAcceleratorNode {
    fn kind(&self) -> &'static str {
        "strix-xdna-npu"
    }

    fn probe(&self) -> io::Result<NpuProbeReport> {
        probe_strix_npu(&self.accel_path, self.min_memlock_bytes)
    }

    fn validate_compute(&self) -> io::Result<AcceleratorSummary> {
        let probe = self.probe()?;
        let smoke = run_strix_npu_smoke_report(&self.smoke_config);
        Ok(summarize_strix_npu_accelerator(&probe, &smoke))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accel(exists: bool, is_char_device: bool) -> DeviceNodeProbe {
        DeviceNodeProbe {
            path: PathBuf::from("/dev/accel/accel0"),
            exists,
            is_char_device,
            mode: 0o660,
            uid: 0,
            gid: 992,
        }
    }

    fn xrt(found: bool, status: Option<i32>, output: &str) -> CommandProbe {
        CommandProbe {
            program: "xrt-smi".to_string(),
            found,
            status,
            output: output.to_string(),
        }
    }

    #[test]
    fn classifies_ready_when_device_and_xrt_are_usable() {
        let memlock = MemlockLimit {
            soft_bytes: Some(DEFAULT_MIN_MEMLOCK_BYTES),
            hard_bytes: Some(DEFAULT_MIN_MEMLOCK_BYTES),
        };

        let (readiness, advice) = classify_npu_readiness(
            &accel(true, true),
            memlock,
            &xrt(true, Some(0), ""),
            DEFAULT_MIN_MEMLOCK_BYTES,
        );

        assert!(matches!(readiness, NpuReadiness::Ready));
        assert!(advice.is_empty());
    }

    #[test]
    fn classifies_xrt_mmap_eagain_as_memlock_too_low_when_limit_is_small() {
        let memlock = MemlockLimit {
            soft_bytes: Some(8 * 1024 * 1024),
            hard_bytes: Some(8 * 1024 * 1024),
        };

        let (readiness, advice) = classify_npu_readiness(
            &accel(true, true),
            memlock,
            &xrt(
                true,
                Some(1),
                "mmap failed: Resource temporarily unavailable",
            ),
            DEFAULT_MIN_MEMLOCK_BYTES,
        );

        assert!(matches!(readiness, NpuReadiness::MemlockTooLow));
        assert!(advice.iter().any(|line| line.contains("RLIMIT_MEMLOCK")));
    }

    #[test]
    fn missing_device_node_takes_precedence_over_xrt_state() {
        let memlock = MemlockLimit {
            soft_bytes: None,
            hard_bytes: None,
        };

        let (readiness, advice) = classify_npu_readiness(
            &accel(false, false),
            memlock,
            &xrt(true, Some(0), ""),
            DEFAULT_MIN_MEMLOCK_BYTES,
        );

        assert!(matches!(readiness, NpuReadiness::MissingDeviceNode));
        assert!(advice.iter().any(|line| line.contains("/dev/accel/accel0")));
    }

    #[test]
    fn classifies_plain_xrt_failure_separately_from_memlock() {
        let memlock = MemlockLimit {
            soft_bytes: None,
            hard_bytes: None,
        };

        let (readiness, advice) = classify_npu_readiness(
            &accel(true, true),
            memlock,
            &xrt(true, Some(1), "firmware probe failed"),
            DEFAULT_MIN_MEMLOCK_BYTES,
        );

        assert!(matches!(readiness, NpuReadiness::XrtSmiFailed));
        assert!(advice
            .iter()
            .any(|line| line.contains("xrt-smi examine failed")));
    }

    #[test]
    fn reports_missing_xrt_before_memlock_advice() {
        let memlock = MemlockLimit {
            soft_bytes: Some(8 * 1024 * 1024),
            hard_bytes: Some(8 * 1024 * 1024),
        };

        let (readiness, advice) = classify_npu_readiness(
            &accel(true, true),
            memlock,
            &xrt(false, None, "xrt-smi not found"),
            DEFAULT_MIN_MEMLOCK_BYTES,
        );

        assert!(matches!(readiness, NpuReadiness::XrtSmiMissing));
        assert!(advice.iter().any(|line| line.contains("XRT utilities")));
        assert!(!advice.iter().any(|line| line.contains("RLIMIT_MEMLOCK")));
    }

    #[test]
    fn default_compute_smoke_plan_runs_mlir_aie_on_npu2() {
        let cfg = StrixNpuSmokeConfig::default_compute();

        let plan = plan_strix_npu_smoke(&cfg);

        assert!(plan.program.ends_with("scripts/strix-npu-smoke.sh"));
        assert_eq!(plan.env_value("HIPFIRE_NPU_RUN_MLIR_AIE_SMOKE"), Some("1"));
        assert_eq!(plan.env_value("HIPFIRE_NPU_DEVICE_NAME"), Some("npu2"));
        assert_eq!(
            plan.env_value("HIPFIRE_NPU_MIN_MEMLOCK_BYTES"),
            Some("134217728")
        );
    }

    #[test]
    fn smoke_plan_carries_explicit_mlir_aie_dir() {
        let mut cfg = StrixNpuSmokeConfig::default_probe_only();
        cfg.mlir_aie_dir = Some(PathBuf::from("/opt/mlir-aie"));

        let plan = plan_strix_npu_smoke(&cfg);

        assert_eq!(plan.env_value("HIPFIRE_NPU_RUN_MLIR_AIE_SMOKE"), Some("0"));
        assert_eq!(plan.env_value("MLIR_AIE_DIR"), Some("/opt/mlir-aie"));
    }

    #[test]
    fn parses_compute_smoke_pass_and_timing() {
        let output = "\
Device(s) Present
|[0000:c0:00.1]  |RyzenAI-npu5  |

Avg NPU time: 256us.

PASS!
";

        let parsed = parse_strix_npu_smoke_output(output);

        assert!(parsed.xrt_detected);
        assert_eq!(parsed.device_name.as_deref(), Some("RyzenAI-npu5"));
        assert_eq!(parsed.avg_npu_time_us, Some(256));
        assert!(parsed.compute_passed);
    }

    #[test]
    fn parses_probe_only_detection_without_compute_pass() {
        let output = "\
Device(s) Present
|[0000:c0:00.1]  |RyzenAI-npu5  |
";

        let parsed = parse_strix_npu_smoke_output(output);

        assert!(parsed.xrt_detected);
        assert_eq!(parsed.device_name.as_deref(), Some("RyzenAI-npu5"));
        assert_eq!(parsed.avg_npu_time_us, None);
        assert!(!parsed.compute_passed);
    }

    #[test]
    fn summarizes_ready_probe_and_passing_compute_as_validated_node() {
        let probe = synthetic_probe(NpuReadiness::Ready, true);
        let smoke = synthetic_smoke(Some("RyzenAI-npu5"), Some(287), true, Some(0));

        let summary = summarize_strix_npu_accelerator(&probe, &smoke);

        assert_eq!(summary.kind, "strix-xdna-npu");
        assert!(summary.ready);
        assert!(summary.compute_validated);
        assert!(summary.daemon_usable);
        assert_eq!(summary.device_name.as_deref(), Some("RyzenAI-npu5"));
        assert_eq!(summary.avg_npu_time_us, Some(287));
    }

    #[test]
    fn compute_can_pass_even_when_probe_is_not_daemon_ready() {
        let probe = synthetic_probe(NpuReadiness::MemlockTooLow, false);
        let smoke = synthetic_smoke(Some("RyzenAI-npu5"), Some(287), true, Some(0));

        let summary = summarize_strix_npu_accelerator(&probe, &smoke);

        assert!(!summary.ready);
        assert!(summary.compute_validated);
        assert!(!summary.daemon_usable);
    }

    fn synthetic_probe(readiness: NpuReadiness, xrt_ok: bool) -> NpuProbeReport {
        NpuProbeReport {
            accel: accel(true, true),
            memlock: MemlockLimit {
                soft_bytes: Some(DEFAULT_MIN_MEMLOCK_BYTES),
                hard_bytes: Some(DEFAULT_MIN_MEMLOCK_BYTES),
            },
            min_memlock_bytes: DEFAULT_MIN_MEMLOCK_BYTES,
            xrt_smi: xrt(true, Some(if xrt_ok { 0 } else { 1 }), ""),
            readiness,
            advice: Vec::new(),
        }
    }

    fn synthetic_smoke(
        device_name: Option<&str>,
        avg_npu_time_us: Option<u64>,
        compute_passed: bool,
        status: Option<i32>,
    ) -> StrixNpuSmokeReport {
        StrixNpuSmokeReport {
            command: CommandProbe {
                program: "strix-npu-smoke.sh".to_string(),
                found: true,
                status,
                output: String::new(),
            },
            parsed: StrixNpuSmokeParsed {
                xrt_detected: device_name.is_some(),
                device_name: device_name.map(str::to_string),
                avg_npu_time_us,
                compute_passed,
            },
        }
    }
}
