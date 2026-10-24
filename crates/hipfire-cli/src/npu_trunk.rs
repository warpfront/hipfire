// SPDX-License-Identifier: Apache-2.0
//! Offline calibration assembly. This command never runs a benchmark or opens a device.
use anyhow::{bail, Context, Result};
use clap::Args;
use hipfire_arch_qwen4::npu_trunk::FLASH_NEXT;
use hipfire_runtime::coproc::{
    CalKey, Calibration, ContentionCalibration, FamilyCalibration, GlueCalibration, GpuCalibration,
    NpuCalibration, QsaOrder, CALIBRATION_SCHEMA_VERSION,
};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::PathBuf, process::Command};

#[derive(Args, Debug)]
pub(crate) struct CalibrateArgs {
    #[arg(long, value_name = "MEASUREMENTS.JSON")]
    from: PathBuf,
    #[arg(long, value_name = "CAL.JSON")]
    out: PathBuf,
}
/// All measured costs are mandatory; only build defaults to the release contract.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Measurements {
    schema_version: u32,
    key: CalKey,
    gpu: GpuCalibration,
    families: BTreeMap<String, FamilyCalibration>,
    glue: GlueCalibration,
    npu: NpuCalibration,
    contention: ContentionCalibration,
    margin: f64,
    #[serde(default)]
    qsa_order: Option<QsaOrder>,
}
impl Measurements {
    fn assemble(self, uname_release: &str) -> Result<Calibration> {
        if self.schema_version != CALIBRATION_SCHEMA_VERSION {
            bail!("unsupported measurements schema_version");
        }
        // Device/fw and index identity come from the measurement collector.
        // CalKey defaults an omitted build to this release's stable NPU contract.
        if self.key.kernel.uname_release != uname_release {
            bail!(
                "measurement key mismatch: kernel.uname_release (measured {}, current {})",
                self.key.kernel.uname_release,
                uname_release
            );
        }
        let cal = Calibration {
            schema_version: CALIBRATION_SCHEMA_VERSION,
            key: self.key,
            gpu: self.gpu,
            families: self.families,
            glue: self.glue,
            npu: self.npu,
            contention: self.contention,
            margin: self.margin,
            qsa_order: self.qsa_order,
        };
        cal.validate(&FLASH_NEXT).map_err(anyhow::Error::msg)?;
        Ok(cal)
    }
}
pub(crate) fn calibrate_command(args: CalibrateArgs) -> Result<()> {
    let bytes = fs::read(&args.from).with_context(|| format!("read {}", args.from.display()))?;
    let input: Measurements =
        serde_json::from_slice(&bytes).context("invalid or incomplete measurement record")?;
    let output = Command::new("uname")
        .arg("-r")
        .output()
        .context("read current kernel release")?;
    if !output.status.success() {
        bail!("uname -r failed");
    }
    let release = std::str::from_utf8(&output.stdout)
        .context("uname output is not UTF-8")?
        .trim();
    let cal = input.assemble(release)?;
    cal.store(&args.out, &FLASH_NEXT)
        .map_err(anyhow::Error::msg)?;
    println!(
        "npu-trunk: calibration={} schema_version={}",
        args.out.display(),
        cal.schema_version
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn npu_trunk_cli_requires_both_paths() {
        use clap::Parser;
        assert!(crate::Cli::try_parse_from([
            "hipfire",
            "calibrate-npu-trunk",
            "--from",
            "in.json",
            "--out",
            "out.json"
        ])
        .is_ok());
        assert!(crate::Cli::try_parse_from([
            "hipfire",
            "calibrate-npu-trunk",
            "--from",
            "in.json"
        ])
        .is_err());
    }
    #[test]
    fn npu_trunk_measurements_reject_missing_fields() {
        assert!(serde_json::from_str::<Measurements>("{}").is_err());
        assert!(
            serde_json::from_str::<Measurements>(r#"{"schema_version":4,"margin":0}"#).is_err()
        );
    }
    #[test]
    fn npu_trunk_measurements_accept_schema_v4_stock_and_overlay() {
        for source in [
            include_str!("../../hipfire-arch-qwen4/tests/fixtures/npu_trunk/g80bf16-stock-v2.json"),
            include_str!(
                "../../hipfire-arch-qwen4/tests/fixtures/npu_trunk/g80bf16-overlay-v2.json"
            ),
        ] {
            let input: Measurements = serde_json::from_str(source).unwrap();
            let release = input.key.kernel.uname_release.clone();
            let cal = input.assemble(&release).unwrap();
            assert_eq!(cal.schema_version, 4);
            assert_eq!(cal.families["qkv"].out_dtype, "bf16");
            assert_eq!(cal.families["zab"].out_dtype, "f32");
            assert_eq!(cal.glue.consume_mode, "retile");
            assert_eq!(cal.glue.consume_ms_per_col_8192["bf16"], 2.85 / 16480.0);
            let mut direct: serde_json::Value = serde_json::from_str(source).unwrap();
            direct["glue"]["consume_mode"] = serde_json::json!("direct");
            let input: Measurements = serde_json::from_value(direct).unwrap();
            assert!(input.assemble(&release).is_ok());
            let input: Measurements = serde_json::from_str(source).unwrap();
            assert!(input.assemble("wrong-kernel").is_err());
            let mut defaults: serde_json::Value = serde_json::from_str(source).unwrap();
            defaults["key"].as_object_mut().unwrap().remove("build");
            let input: Measurements = serde_json::from_value(defaults).unwrap();
            assert_eq!(
                input.assemble(&release).unwrap().key.build,
                hipfire_runtime::coproc::calibration_build_identity()
            );
            for (parent, field) in [
                ("key", "trunk_route"),
                ("glue", "consume_mode"),
                ("glue", "consume_ms_per_col_8192"),
            ] {
                let mut incomplete: serde_json::Value = serde_json::from_str(source).unwrap();
                incomplete[parent].as_object_mut().unwrap().remove(field);
                assert!(serde_json::from_value::<Measurements>(incomplete).is_err());
            }
        }
    }
}
