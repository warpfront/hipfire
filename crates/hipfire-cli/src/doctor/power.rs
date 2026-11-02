// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `power` check: is this Strix Halo running below full power?
//!
//! hipfire's published speeds are measured with the machine on its performance
//! power mode and the GPU at its default power limit. A low-power or balanced
//! system profile, a GPU power-saving profile, a power limit well under the
//! default, or a lowered clock ceiling all cost speed without any error, so the
//! doctor says so. Reads only sysfs under the probe root; never writes and never
//! blocks.

use super::report::{Check, CheckStatus};
use std::path::{Path, PathBuf};

/// The power limit counts as "well below" the default under this share of it.
const CAP_WARN_PERCENT: u64 = 80;
/// The clock ceiling counts as "lowered" under this share of the hardware maximum.
const CLOCK_WARN_PERCENT: f64 = 90.0;

/// What the sysfs files said. Every field is `None` when unreadable.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Facts {
    /// `power1_cap` in microwatts.
    pub cap_uw: Option<u64>,
    /// `power1_cap_default` in microwatts.
    pub cap_default_uw: Option<u64>,
    /// `power1_cap_max` in microwatts.
    pub cap_max_uw: Option<u64>,
    /// ACPI platform profile: `low-power`, `balanced`, `performance`, ...
    pub platform_profile: Option<String>,
    /// Active entry of `pp_power_profile_mode`, e.g. `POWER_SAVING`.
    pub gpu_profile: Option<String>,
    /// `power_dpm_force_performance_level`.
    pub perf_level: Option<String>,
    /// Current gfx clock in MHz.
    pub sclk_mhz: Option<u32>,
    /// Highest gfx clock the driver currently offers, MHz.
    pub sclk_ceiling_mhz: Option<u32>,
    /// Highest gfx clock the hardware allows (`OD_RANGE`), MHz.
    pub sclk_hw_max_mhz: Option<u32>,
}

fn read_trimmed(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

fn read_u64(path: &Path) -> Option<u64> {
    read_trimmed(path)?.parse().ok()
}

fn sorted_children(dir: &Path) -> Vec<PathBuf> {
    let mut children: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| entries.flatten().map(|entry| entry.path()).collect())
        .unwrap_or_default();
    children.sort();
    children
}

/// The device's amdgpu hwmon directory: the one named `amdgpu`, else the first.
fn hwmon_dir(device: &Path) -> Option<PathBuf> {
    let dirs = sorted_children(&device.join("hwmon"));
    dirs.iter()
        .find(|dir| read_trimmed(&dir.join("name")).as_deref() == Some("amdgpu"))
        .or_else(|| dirs.first())
        .cloned()
}

/// `/sys/firmware/acpi/platform_profile`, else the first readable
/// `/sys/class/platform-profile/*/profile`.
fn platform_profile(root: &Path) -> Option<String> {
    read_trimmed(&root.join("sys/firmware/acpi/platform_profile")).or_else(|| {
        sorted_children(&root.join("sys/class/platform-profile"))
            .iter()
            .find_map(|dir| read_trimmed(&dir.join("profile")))
    })
}

/// The `*`-marked entry of `pp_power_profile_mode`. Rows look like
/// `  0 BOOTUP_DEFAULT*: ...` or `  1 3D_FULL_SCREEN*`; the header and the
/// per-clock continuation lines never carry the mark on the name.
pub(super) fn parse_gpu_profile(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let (index, rest) = line.trim().split_once(char::is_whitespace)?;
        index.parse::<u32>().ok()?;
        let name_field = rest.split(':').next().unwrap_or(rest);
        if !name_field.contains('*') {
            return None;
        }
        let name = name_field.replace('*', "");
        name.split_whitespace().next().map(str::to_owned)
    })
}

fn mhz(token: &str) -> Option<u32> {
    token
        .trim()
        .trim_end_matches(|c: char| c.is_ascii_alphabetic())
        .parse::<f64>()
        .ok()
        .map(|value| value.round() as u32)
}

/// `pp_dpm_sclk`: `(current level MHz, highest level MHz)`. Rows are
/// `1: 2900Mhz *`; the `*` marks the current level.
pub(super) fn parse_dpm_sclk(text: &str) -> (Option<u32>, Option<u32>) {
    let mut current = None;
    let mut top = None;
    for line in text.lines() {
        let Some((level, rest)) = line.split_once(':') else {
            continue;
        };
        if level.trim().parse::<u32>().is_err() {
            continue;
        }
        let Some(value) = rest.split_whitespace().next().and_then(mhz) else {
            continue;
        };
        top = top.max(Some(value));
        if rest.contains('*') {
            current = Some(value);
        }
    }
    (current, top)
}

pub(super) fn read(root: &Path, device: &Path) -> Facts {
    let mut facts = Facts {
        platform_profile: platform_profile(root),
        gpu_profile: std::fs::read_to_string(device.join("pp_power_profile_mode"))
            .ok()
            .and_then(|text| parse_gpu_profile(&text)),
        perf_level: read_trimmed(&device.join("power_dpm_force_performance_level")),
        ..Facts::default()
    };
    if let Some(hwmon) = hwmon_dir(device) {
        facts.cap_uw = read_u64(&hwmon.join("power1_cap"));
        facts.cap_default_uw = read_u64(&hwmon.join("power1_cap_default"));
        facts.cap_max_uw = read_u64(&hwmon.join("power1_cap_max"));
        let label = read_trimmed(&hwmon.join("freq1_label"));
        if label.as_deref().is_none_or(|label| label == "sclk") {
            facts.sclk_mhz = read_u64(&hwmon.join("freq1_input"))
                .map(|hz| (hz as f64 / 1_000_000.0).round() as u32);
        }
    }
    if let Ok(text) = std::fs::read_to_string(device.join("pp_dpm_sclk")) {
        let (current, top) = parse_dpm_sclk(&text);
        facts.sclk_mhz = facts.sclk_mhz.or(current);
        facts.sclk_ceiling_mhz = top;
    }
    if let Some(od) = std::fs::read_to_string(device.join("pp_od_clk_voltage"))
        .ok()
        .and_then(|text| crate::bench_power::parse_od_sclk(&text))
    {
        facts.sclk_hw_max_mhz = od.range_max_mhz.map(|value| value.round() as u32);
        facts.sclk_ceiling_mhz = facts
            .sclk_ceiling_mhz
            .or(od.max_mhz.map(|value| value.round() as u32));
    }
    facts
}

fn watts(microwatts: u64) -> String {
    format!("{:.0} W", microwatts as f64 / 1_000_000.0)
}

/// Which fix lines apply.
#[derive(Default)]
struct Causes {
    system_profile: bool,
    gpu_profile: bool,
    cap: bool,
    clocks: bool,
}

/// What is below full power, in plain words, and which fixes go with it.
fn reasons(facts: &Facts) -> (Vec<String>, Causes) {
    let mut found = Vec::new();
    let mut causes = Causes::default();
    if let Some(profile) = facts.platform_profile.as_deref() {
        if matches!(profile, "low-power" | "quiet" | "cool" | "balanced") {
            found.push(format!(
                "the system power mode is {profile}, not performance"
            ));
            causes.system_profile = true;
        }
    }
    if let Some(profile) = facts.gpu_profile.as_deref() {
        if profile.eq_ignore_ascii_case("POWER_SAVING") {
            found.push("the GPU power profile is POWER_SAVING".to_owned());
            causes.gpu_profile = true;
        }
    }
    if let (Some(cap), Some(default)) = (facts.cap_uw, facts.cap_default_uw) {
        if default > 0 && cap.saturating_mul(100) < default.saturating_mul(CAP_WARN_PERCENT) {
            found.push(format!(
                "the GPU power limit is {}, well below its {} default",
                watts(cap),
                watts(default)
            ));
            causes.cap = true;
        }
    }
    if let (Some(ceiling), Some(hardware)) = (facts.sclk_ceiling_mhz, facts.sclk_hw_max_mhz) {
        if hardware > 0 && f64::from(ceiling) * 100.0 < f64::from(hardware) * CLOCK_WARN_PERCENT {
            found.push(format!(
                "the GPU clock ceiling is {ceiling} MHz of {hardware} MHz"
            ));
            causes.clocks = true;
        }
    }
    if matches!(facts.perf_level.as_deref(), Some("low")) {
        found.push("the GPU clocks are forced to their lowest level".to_owned());
        causes.clocks = true;
    }
    (found, causes)
}

/// The settings that were readable, for the detail line.
fn readings(facts: &Facts) -> Vec<String> {
    let mut parts = Vec::new();
    if let Some(profile) = &facts.platform_profile {
        parts.push(format!("system mode {profile}"));
    }
    if let Some(profile) = &facts.gpu_profile {
        parts.push(format!("GPU profile {profile}"));
    }
    if let Some(cap) = facts.cap_uw {
        parts.push(match facts.cap_default_uw.or(facts.cap_max_uw) {
            Some(reference) => {
                let label = if facts.cap_default_uw.is_some() {
                    "default"
                } else {
                    "max"
                };
                format!("power limit {} ({label} {})", watts(cap), watts(reference))
            }
            None => format!("power limit {}", watts(cap)),
        });
    }
    if let Some(current) = facts.sclk_mhz {
        let max = facts.sclk_hw_max_mhz.or(facts.sclk_ceiling_mhz);
        parts.push(match max {
            Some(max) => format!("GPU clock {current} of {max} MHz now"),
            None => format!("GPU clock {current} MHz now"),
        });
    }
    parts
}

const FIX_SYSTEM: &str = "Switch the system to its performance power mode, for example `powerprofilesctl set performance`.";
const FIX_GPU: &str = "Set the GPU power profile back to its default.";
const FIX_CAP: &str =
    "Raise the GPU power limit to its default or higher in the BIOS or your vendor's power tool.";
const FIX_CLOCKS: &str = "Lift the GPU clock limit: set the GPU performance level back to auto and remove any maximum-clock setting.";
const FIX_TAIL: &str = "Run this check again afterwards to confirm.";

/// Judge the facts. Never `Fail`: low power slows hipfire, it does not block it.
pub(super) fn assess(facts: &Facts) -> Check {
    let (found, causes) = reasons(facts);
    let parts = readings(facts);
    if found.is_empty() {
        let (status, detail) = if parts.is_empty() {
            (
                CheckStatus::Info,
                "no power settings could be read on this system".to_owned(),
            )
        } else {
            (
                CheckStatus::Pass,
                format!("full power: {}", parts.join("; ")),
            )
        };
        return Check {
            id: "power",
            status,
            detail,
            fix: None,
        };
    }
    let mut detail = format!(
        "{}. hipfire's published speeds assume full power, so expect lower numbers until this is changed",
        capitalize(&found.join("; "))
    );
    if !parts.is_empty() {
        detail.push_str(&format!(" ({})", parts.join("; ")));
    }
    let mut fix = Vec::new();
    for (applies, line) in [
        (causes.system_profile, FIX_SYSTEM),
        (causes.gpu_profile, FIX_GPU),
        (causes.cap, FIX_CAP),
        (causes.clocks, FIX_CLOCKS),
    ] {
        if applies {
            fix.push(line);
        }
    }
    fix.push(FIX_TAIL);
    Check {
        id: "power",
        status: CheckStatus::Warn,
        detail,
        fix: Some(fix.join("\n")),
    }
}

fn capitalize(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static NEXT: AtomicU32 = AtomicU32::new(0);
    const DEVICE: &str = "sys/bus/pci/devices/0000:c0:00.0";

    /// A temp tree standing in for `/`, removed on drop.
    struct Tree(PathBuf);

    impl Tree {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "hipfire-doctor-power-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&root).unwrap();
            Self(root)
        }

        fn put(&self, relative: &str, text: &str) {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }

        fn device(&self, file: &str, text: &str) {
            self.put(&format!("{DEVICE}/{file}"), text);
        }

        fn hwmon(&self, file: &str, text: &str) {
            self.put(&format!("{DEVICE}/hwmon/hwmon5/{file}"), text);
        }

        fn facts(&self) -> Facts {
            read(&self.0, &self.0.join(DEVICE))
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const MODES: &str = "PROFILE_INDEX(NAME)\n  0 BOOTUP_DEFAULT*\n  1 3D_FULL_SCREEN\n  2 POWER_SAVING\n  3 VIDEO\n  4 VR\n  5 COMPUTE\n  6 CUSTOM\n";

    /// A Halo at full power: performance profile, default limit, full clocks.
    fn full_power() -> Tree {
        let tree = Tree::new();
        tree.put("sys/firmware/acpi/platform_profile", "performance\n");
        tree.device("pp_power_profile_mode", MODES);
        tree.device("power_dpm_force_performance_level", "auto\n");
        tree.device("pp_dpm_sclk", "0: 600Mhz\n1: 2900Mhz *\n");
        tree.device(
            "pp_od_clk_voltage",
            "OD_SCLK:\n0:        600Mhz\n1:       2900Mhz\nOD_RANGE:\nSCLK:     600Mhz       2900Mhz\n",
        );
        tree.hwmon("name", "amdgpu\n");
        tree.hwmon("power1_cap", "120000000\n");
        tree.hwmon("power1_cap_default", "120000000\n");
        tree.hwmon("power1_cap_max", "140000000\n");
        tree.hwmon("freq1_label", "sclk\n");
        tree.hwmon("freq1_input", "2900000000\n");
        tree
    }

    #[test]
    fn full_power_host_passes_and_reports_what_it_read() {
        let tree = full_power();
        let facts = tree.facts();
        assert_eq!(
            facts,
            Facts {
                cap_uw: Some(120_000_000),
                cap_default_uw: Some(120_000_000),
                cap_max_uw: Some(140_000_000),
                platform_profile: Some("performance".to_owned()),
                gpu_profile: Some("BOOTUP_DEFAULT".to_owned()),
                perf_level: Some("auto".to_owned()),
                sclk_mhz: Some(2900),
                sclk_ceiling_mhz: Some(2900),
                sclk_hw_max_mhz: Some(2900),
            }
        );
        let check = assess(&facts);
        assert_eq!(check.status, CheckStatus::Pass, "{}", check.detail);
        assert_eq!(check.id, "power");
        assert!(check.fix.is_none());
        for part in [
            "system mode performance",
            "GPU profile BOOTUP_DEFAULT",
            "power limit 120 W (default 120 W)",
            "GPU clock 2900 of 2900 MHz now",
        ] {
            assert!(check.detail.contains(part), "{}", check.detail);
        }
    }

    #[test]
    fn idle_clock_is_not_a_warning() {
        let tree = full_power();
        tree.hwmon("freq1_input", "600000000\n");
        let check = assess(&tree.facts());
        assert_eq!(check.status, CheckStatus::Pass, "{}", check.detail);
        assert!(check.detail.contains("GPU clock 600 of 2900 MHz now"));
    }

    #[test]
    fn nothing_readable_is_informational() {
        let tree = Tree::new();
        let facts = tree.facts();
        assert_eq!(facts, Facts::default());
        let check = assess(&facts);
        assert_eq!(check.status, CheckStatus::Info);
        assert!(check.fix.is_none());
    }

    #[test]
    fn low_power_and_balanced_system_profiles_warn() {
        for profile in ["low-power", "quiet", "cool", "balanced"] {
            let tree = full_power();
            tree.put(
                "sys/firmware/acpi/platform_profile",
                &format!("{profile}\n"),
            );
            let check = assess(&tree.facts());
            assert_eq!(check.status, CheckStatus::Warn, "{profile}");
            assert!(
                check
                    .detail
                    .contains(&format!("system power mode is {profile}")),
                "{}",
                check.detail
            );
            assert!(check.detail.contains("assume full power"));
            let fix = check.fix.unwrap();
            assert!(fix.contains("powerprofilesctl set performance"), "{fix}");
        }
    }

    #[test]
    fn balanced_performance_and_performance_do_not_warn() {
        for profile in ["balanced-performance", "performance"] {
            let tree = full_power();
            tree.put(
                "sys/firmware/acpi/platform_profile",
                &format!("{profile}\n"),
            );
            assert_eq!(assess(&tree.facts()).status, CheckStatus::Pass, "{profile}");
        }
    }

    #[test]
    fn platform_profile_falls_back_to_the_class_directory() {
        let tree = full_power();
        tree.remove("sys/firmware/acpi/platform_profile");
        tree.put(
            "sys/class/platform-profile/platform-profile-0/profile",
            "low-power\n",
        );
        let facts = tree.facts();
        assert_eq!(facts.platform_profile.as_deref(), Some("low-power"));
        assert_eq!(assess(&facts).status, CheckStatus::Warn);
    }

    impl Tree {
        fn remove(&self, relative: &str) {
            std::fs::remove_file(self.0.join(relative)).unwrap();
        }
    }

    #[test]
    fn gpu_power_saving_profile_warns() {
        let tree = full_power();
        tree.device(
            "pp_power_profile_mode",
            &MODES
                .replace("BOOTUP_DEFAULT*", "BOOTUP_DEFAULT")
                .replace("POWER_SAVING", "POWER_SAVING*"),
        );
        let check = assess(&tree.facts());
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.detail.contains("GPU power profile is POWER_SAVING"));
        assert!(check
            .fix
            .unwrap()
            .contains("power profile back to its default"));
    }

    #[test]
    fn power_limit_well_below_default_warns_but_a_small_dip_does_not() {
        let tree = full_power();
        tree.hwmon("power1_cap", "55000000\n");
        let check = assess(&tree.facts());
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(
            check
                .detail
                .contains("power limit is 55 W, well below its 120 W default"),
            "{}",
            check.detail
        );
        assert!(check.fix.unwrap().contains("Raise the GPU power limit"));

        tree.hwmon("power1_cap", "110000000\n");
        assert_eq!(assess(&tree.facts()).status, CheckStatus::Pass);
        // Exactly the threshold is not "below" it.
        tree.hwmon("power1_cap", "96000000\n");
        assert_eq!(assess(&tree.facts()).status, CheckStatus::Pass);
        tree.hwmon("power1_cap", "95000000\n");
        assert_eq!(assess(&tree.facts()).status, CheckStatus::Warn);
    }

    #[test]
    fn power_limit_without_a_default_is_reported_not_judged() {
        let tree = full_power();
        tree.remove(&format!("{DEVICE}/hwmon/hwmon5/power1_cap_default"));
        tree.hwmon("power1_cap", "30000000\n");
        let check = assess(&tree.facts());
        assert_eq!(check.status, CheckStatus::Pass, "{}", check.detail);
        assert!(check.detail.contains("power limit 30 W (max 140 W)"));
    }

    #[test]
    fn lowered_clock_ceiling_warns() {
        let tree = full_power();
        tree.device("pp_dpm_sclk", "0: 600Mhz *\n1: 1800Mhz\n");
        let check = assess(&tree.facts());
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(
            check
                .detail
                .contains("clock ceiling is 1800 MHz of 2900 MHz"),
            "{}",
            check.detail
        );
        assert!(check.fix.unwrap().contains("Lift the GPU clock limit"));
    }

    #[test]
    fn forced_low_performance_level_warns() {
        let tree = full_power();
        tree.device("power_dpm_force_performance_level", "low\n");
        let check = assess(&tree.facts());
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.detail.contains("forced to their lowest level"));
    }

    #[test]
    fn several_problems_are_all_named_with_one_fix_each() {
        let tree = full_power();
        tree.put("sys/firmware/acpi/platform_profile", "balanced\n");
        tree.hwmon("power1_cap", "55000000\n");
        let check = assess(&tree.facts());
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.detail.contains("system power mode is balanced"));
        assert!(check.detail.contains("power limit is 55 W"));
        let fix = check.fix.unwrap();
        assert_eq!(fix.lines().count(), 3, "{fix}");
        assert!(fix.ends_with("Run this check again afterwards to confirm."));
    }

    #[test]
    fn warning_text_has_no_paths_or_pci_addresses() {
        let tree = full_power();
        tree.put("sys/firmware/acpi/platform_profile", "low-power\n");
        tree.hwmon("power1_cap", "55000000\n");
        let check = assess(&tree.facts());
        let text = format!("{} {}", check.detail, check.fix.unwrap());
        assert!(!text.contains("/sys") && !text.contains("0000:"), "{text}");
    }

    #[test]
    fn hwmon_without_a_name_is_still_used_and_another_hwmon_is_skipped() {
        let tree = Tree::new();
        tree.put(&format!("{DEVICE}/hwmon/hwmon1/name"), "other\n");
        tree.put(&format!("{DEVICE}/hwmon/hwmon1/power1_cap"), "1000000\n");
        tree.put(&format!("{DEVICE}/hwmon/hwmon7/name"), "amdgpu\n");
        tree.put(&format!("{DEVICE}/hwmon/hwmon7/power1_cap"), "120000000\n");
        assert_eq!(tree.facts().cap_uw, Some(120_000_000));

        let tree = Tree::new();
        tree.put(&format!("{DEVICE}/hwmon/hwmon3/power1_cap"), "55000000\n");
        assert_eq!(tree.facts().cap_uw, Some(55_000_000));
    }

    #[test]
    fn current_clock_falls_back_to_pp_dpm_sclk_and_a_non_sclk_label_is_ignored() {
        let tree = Tree::new();
        tree.device("pp_dpm_sclk", "0: 600Mhz *\n1: 2900Mhz\n");
        tree.hwmon("freq1_label", "mclk\n");
        tree.hwmon("freq1_input", "1000000000\n");
        let facts = tree.facts();
        assert_eq!(facts.sclk_mhz, Some(600));
        assert_eq!(facts.sclk_ceiling_mhz, Some(2900));
        assert_eq!(facts.sclk_hw_max_mhz, None);
    }

    #[test]
    fn gpu_profile_parser_handles_the_driver_layouts() {
        assert_eq!(
            parse_gpu_profile("PROFILE_INDEX(NAME)\n  0 BOOTUP_DEFAULT*\n  1 3D_FULL_SCREEN\n"),
            Some("BOOTUP_DEFAULT".to_owned())
        );
        assert_eq!(
            parse_gpu_profile(
                "NUM        MODE_NAME BUSY_SET_POINT FPS USE_RLC_BUSY MIN_ACTIVE_LEVEL\n  0 BOOTUP_DEFAULT : 70 60 0 0\n  5 COMPUTE*: 30 60 1 3\n"
            ),
            Some("COMPUTE".to_owned())
        );
        assert_eq!(
            parse_gpu_profile("  1 3D_FULL_SCREEN *: 1 2 3\n"),
            Some("3D_FULL_SCREEN".to_owned())
        );
        assert_eq!(
            parse_gpu_profile("PROFILE_INDEX(NAME)\n  0 BOOTUP_DEFAULT\n"),
            None
        );
        assert_eq!(parse_gpu_profile(""), None);
    }

    #[test]
    fn dpm_sclk_parser_finds_current_and_top() {
        assert_eq!(
            parse_dpm_sclk("0: 600Mhz\n1: 2900Mhz *\n"),
            (Some(2900), Some(2900))
        );
        assert_eq!(
            parse_dpm_sclk("0: 600Mhz *\n1: 1500Mhz\n2: 2900Mhz\n"),
            (Some(600), Some(2900))
        );
        assert_eq!(parse_dpm_sclk("garbage\n"), (None, None));
    }
}
