// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! NPU-trunk mode: the single user-facing switch of the Qwen4 integration.
//! The mode borrows the central process-config snapshot without allocating.

use hipfire_runtime::coproc::Switch;

/// Developer environment key of the mode.
pub const NPU_TRUNK_ENV: &str = "HIPFIRE_QWEN4_NPU_TRUNK";
/// Daemon configuration key of the mode.
pub const NPU_TRUNK_DAEMON_KEY: &str = "npu_trunk";

/// The NPU-trunk switch as the coproc layer sees it.
pub const SWITCH: Switch = Switch {
    env: NPU_TRUNK_ENV,
    config_key: "fusions.npu_trunk",
    tag: "npu-trunk",
    name: NPU_TRUNK_DAEMON_KEY,
};

/// Whether `auto` may arm the NPU trunk on this GPU architecture.
pub fn arch_admits_auto(arch: &str) -> bool {
    arch == "gfx1151"
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::coproc::{admission::off_line, Mode};

    #[test]
    fn from_value_absent_is_auto_present_is_strict() {
        assert_eq!(Mode::from_value(SWITCH, None, Mode::Auto), Ok(Mode::Auto));
        assert_eq!(
            Mode::from_value(SWITCH, Some("on"), Mode::Auto),
            Ok(Mode::On)
        );
        let err = Mode::from_value(SWITCH, Some(""), Mode::Auto).unwrap_err();
        assert!(err.starts_with(NPU_TRUNK_ENV), "{err}");
        let err = Mode::from_value(SWITCH, Some("maybe"), Mode::Auto).unwrap_err();
        assert!(err.contains("\"maybe\""), "{err}");
    }

    #[test]
    fn env_key_is_the_documented_developer_key() {
        assert_eq!(NPU_TRUNK_ENV, "HIPFIRE_QWEN4_NPU_TRUNK");
        assert_eq!(NPU_TRUNK_DAEMON_KEY, "npu_trunk");
        assert_eq!(SWITCH.env, NPU_TRUNK_ENV);
        assert_eq!(SWITCH.config_key, "fusions.npu_trunk");
        assert_eq!(SWITCH.name, NPU_TRUNK_DAEMON_KEY);
        let field =
            hipfire_config::field(NPU_TRUNK_DAEMON_KEY).expect("registered npu_trunk schema");
        assert_eq!(field.env_compat, Some(NPU_TRUNK_ENV));
        assert!(matches!(
            field.default,
            hipfire_config::DefaultValue::String("auto")
        ));
        for mode in Mode::ALL {
            assert!(field
                .validate(&hipfire_config::ConfigValue::String(mode.to_string()))
                .is_ok());
        }
        assert!(field
            .validate(&hipfire_config::ConfigValue::String("invalid".into()))
            .is_err());
    }

    #[test]
    fn arch_admits_auto_only_on_gfx1151() {
        assert!(arch_admits_auto("gfx1151"));
        assert!(!arch_admits_auto("gfx1201"));
        assert!(!arch_admits_auto("gfx1100"));
    }

    #[test]
    fn off_line_keeps_the_switch_wording() {
        assert_eq!(off_line(SWITCH, false, "x"), "npu-trunk: off (x)");
        assert_eq!(
            off_line(SWITCH, true, "x"),
            "WARN npu-trunk: off (x); npu_trunk=auto requested the NPU but the trunk did not arm, prefill runs GPU-only (npu_trunk=on makes this a load error)"
        );
    }
}
