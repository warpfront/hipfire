// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Per-repetition GPU clock/power/temperature sampling from sysfs, shared by `hip_loop` and the
//! `raon_oracle` bench. Reading sysfs is a handful of small file reads and MUST happen outside any
//! timed hipEvent interval (callers sample between launches / repetitions). Every field is `null`
//! when its file is missing or unparsable. Included with `#[path = ".../support/clk.rs"] mod clk;`.
#![allow(dead_code)]

use serde_json::{json, Value};

/// The MHz of the active (`*`) line of a `pp_dpm_sclk` table, e.g. `"1: 2100Mhz *"` -> 2100.
pub fn parse_dpm_active_mhz(text: &str) -> Option<f64> {
    let line = text.lines().find(|l| l.contains('*'))?;
    let tok = line
        .split_whitespace()
        .find(|t| t.to_ascii_lowercase().ends_with("mhz"))?;
    tok[..tok.len() - 3].parse::<f64>().ok()
}

/// `freq1_input` (Hz) -> MHz.
pub fn parse_hz_as_mhz(text: &str) -> Option<f64> {
    text.trim().parse::<f64>().ok().map(|v| v / 1e6)
}

/// `power1_average` (microwatts) -> W.
pub fn parse_uw_as_w(text: &str) -> Option<f64> {
    text.trim().parse::<f64>().ok().map(|v| v / 1e6)
}

/// `temp1_input` (millidegrees C) -> C.
pub fn parse_mc_as_c(text: &str) -> Option<f64> {
    text.trim().parse::<f64>().ok().map(|v| v / 1e3)
}

fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn num(v: Option<f64>) -> Value {
    v.map(|x| json!(x)).unwrap_or(Value::Null)
}

/// One sample of the device under `/sys/bus/pci/devices/<pci>`:
/// `{sclk_mhz, sclk_source, sclk_dpm_mhz, sclk_hwmon_mhz, power_w, temp_c, t_unix_ms}`.
/// `sclk_mhz` is the active `pp_dpm_sclk` line when present, else hwmon `freq1_input`.
pub fn sample_at(base: &str) -> Value {
    let hwmon = std::fs::read_dir(format!("{base}/hwmon"))
        .ok()
        .and_then(|mut d| d.next())
        .and_then(|e| e.ok())
        .map(|e| e.path().display().to_string());
    let hw = |name: &str| hwmon.as_ref().and_then(|h| read(&format!("{h}/{name}")));
    let dpm = read(&format!("{base}/pp_dpm_sclk")).and_then(|s| parse_dpm_active_mhz(&s));
    let hwf = hw("freq1_input").and_then(|s| parse_hz_as_mhz(&s));
    let (sclk, source) = match (dpm, hwf) {
        (Some(v), _) => (Some(v), "pp_dpm_sclk"),
        (None, Some(v)) => (Some(v), "hwmon_freq1_input"),
        (None, None) => (None, "none"),
    };
    let t_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    json!({
        "sclk_mhz": num(sclk),
        "sclk_source": source,
        "sclk_dpm_mhz": num(dpm),
        "sclk_hwmon_mhz": num(hwf),
        "power_w": num(hw("power1_average").and_then(|s| parse_uw_as_w(&s))),
        "temp_c": num(hw("temp1_input").and_then(|s| parse_mc_as_c(&s))),
        "t_unix_ms": t_ms,
    })
}

/// [`sample_at`] for the PCI bus id `pci` (any case).
pub fn sample(pci: &str) -> Value {
    sample_at(&format!(
        "/sys/bus/pci/devices/{}",
        pci.to_ascii_lowercase()
    ))
}

/// `{n, min, max, median}` of the numeric field `key` over `samples` (nulls skipped); `n` = 0 and
/// null stats when no sample carries it.
pub fn field_summary(samples: &[Value], key: &str) -> Value {
    let mut v: Vec<f64> = samples
        .iter()
        .filter_map(|s| s.get(key)?.as_f64())
        .collect();
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return json!({"n": 0, "min": null, "max": null, "median": null});
    }
    let n = v.len();
    let median = if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    };
    json!({"n": n, "min": v[0], "max": v[n - 1], "median": median})
}

/// min/max/median of sclk, power and temperature over `samples`.
pub fn summary(samples: &[Value]) -> Value {
    json!({
        "samples": samples.len(),
        "sclk_mhz": field_summary(samples, "sclk_mhz"),
        "power_w": field_summary(samples, "power_w"),
        "temp_c": field_summary(samples, "temp_c"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DPM: &str = "0: 500Mhz\n1: 1500Mhz\n2: 2400Mhz *\n";

    #[test]
    fn dpm_active_line() {
        assert_eq!(parse_dpm_active_mhz(DPM), Some(2400.0));
        assert_eq!(
            parse_dpm_active_mhz("0: 500Mhz *\n1: 900Mhz\n"),
            Some(500.0)
        );
        assert_eq!(parse_dpm_active_mhz("0: 500MHz *"), Some(500.0));
    }

    #[test]
    fn dpm_missing_or_malformed() {
        assert_eq!(parse_dpm_active_mhz(""), None);
        assert_eq!(parse_dpm_active_mhz("0: 500Mhz\n1: 900Mhz\n"), None);
        assert_eq!(parse_dpm_active_mhz("1: bogus *\n"), None);
        assert_eq!(parse_dpm_active_mhz("1: Mhz *\n"), None);
    }

    #[test]
    fn hwmon_units() {
        assert_eq!(parse_hz_as_mhz("2400000000\n"), Some(2400.0));
        assert_eq!(parse_uw_as_w(" 250000000 \n"), Some(250.0));
        assert_eq!(parse_mc_as_c("65000\n"), Some(65.0));
        assert_eq!(parse_hz_as_mhz("n/a"), None);
        assert_eq!(parse_mc_as_c(""), None);
    }

    #[test]
    fn missing_files_give_nulls() {
        let s = sample_at("/nonexistent/clk-fixture");
        assert!(s["sclk_mhz"].is_null() && s["power_w"].is_null() && s["temp_c"].is_null());
        assert_eq!(s["sclk_source"], "none");
        assert!(sample("0000:ff:ff.7")["sclk_mhz"].is_null());
    }

    #[test]
    fn summary_min_max_median_skips_nulls() {
        let s = [
            json!({"sclk_mhz": 2400.0, "power_w": null, "temp_c": 60.0}),
            json!({"sclk_mhz": null, "power_w": null, "temp_c": 62.0}),
            json!({"sclk_mhz": 1800.0, "temp_c": 64.0}),
            json!({"sclk_mhz": 2200.0, "temp_c": 66.0}),
        ];
        let m = summary(&s);
        assert_eq!(m["samples"], 4);
        assert_eq!(m["sclk_mhz"]["n"], 3);
        assert_eq!(m["sclk_mhz"]["min"], 1800.0);
        assert_eq!(m["sclk_mhz"]["max"], 2400.0);
        assert_eq!(m["sclk_mhz"]["median"], 2200.0);
        assert_eq!(m["temp_c"]["median"], 63.0);
        assert_eq!(m["power_w"]["n"], 0);
        assert!(m["power_w"]["median"].is_null());
        assert_eq!(summary(&[])["sclk_mhz"]["n"], 0);
    }
}
