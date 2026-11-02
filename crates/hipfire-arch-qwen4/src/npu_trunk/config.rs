// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! NPU-trunk mode: the single user-facing switch of the Qwen4 integration.
//!
//! The mode borrows the central process-config snapshot without allocating.

use std::fmt;
use std::str::FromStr;

/// Developer environment key of the mode.
pub const NPU_TRUNK_ENV: &str = "HIPFIRE_QWEN4_NPU_TRUNK";
/// Daemon configuration key of the mode.
pub const NPU_TRUNK_DAEMON_KEY: &str = "npu_trunk";

/// How the Flash-Next trunk projections reach the NPU.
///
/// `Auto` is the default; it resolves silently to `Off` outside gfx1151.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum NpuTrunkMode {
    /// Trunk projections stay on the GPU; nothing NPU-related is touched.
    Off,
    /// Use the NPU when it is present and the plan predicts a gain, else fall back to the GPU.
    #[default]
    Auto,
    /// Require the NPU; failure to open it is an error.
    On,
    /// Run the bit-identical GPU emulation of the NPU numerics (no device).
    Emulate,
}

impl NpuTrunkMode {
    /// Every mode, in documentation order.
    pub const ALL: [NpuTrunkMode; 4] = [Self::Off, Self::Auto, Self::On, Self::Emulate];

    /// The canonical lower-case spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::On => "on",
            Self::Emulate => "emulate",
        }
    }

    /// `true` for [`NpuTrunkMode::Off`].
    pub const fn is_off(self) -> bool {
        matches!(self, Self::Off)
    }

    /// An absent key defaults to `Auto`; a present value must be valid.
    pub fn from_value(value: Option<&str>) -> Result<Self, String> {
        match value {
            None => Ok(Self::Auto),
            Some(raw) => raw
                .parse::<Self>()
                .map_err(|e| format!("{NPU_TRUNK_ENV}: {e}")),
        }
    }

    /// Borrow the validated process snapshot; no per-load allocation or I/O.
    pub fn from_env() -> Result<Self, String> {
        let config = hipfire_config::active_or_local_process_config();
        let raw = match config.values.values.get("fusions.npu_trunk") {
            Some(hipfire_config::ConfigValue::String(raw)) => Some(raw.as_str()),
            _ => None,
        };
        Self::from_value(raw)
    }

    /// Resolve auto before any arch-specific work, logging or allocation.
    pub fn resolve(self, arch: &str) -> Self {
        if self == Self::Auto && arch != "gfx1151" {
            Self::Off
        } else {
            self
        }
    }
}

impl FromStr for NpuTrunkMode {
    type Err = String;

    /// Case-insensitive, surrounding whitespace ignored: `off`, `auto`, `on`, `emulate`.
    fn from_str(s: &str) -> Result<Self, String> {
        let t = s.trim();
        for mode in Self::ALL {
            if t.eq_ignore_ascii_case(mode.as_str()) {
                return Ok(mode);
            }
        }
        Err(format!(
            "invalid npu trunk mode {s:?} (expected off, auto, on or emulate)"
        ))
    }
}

impl fmt::Display for NpuTrunkMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_auto_resolves_per_arch() {
        assert_eq!(NpuTrunkMode::default(), NpuTrunkMode::Auto);
        assert_eq!(NpuTrunkMode::Auto.resolve("gfx1151"), NpuTrunkMode::Auto);
        for arch in ["gfx1201", "gfx1100"] {
            assert_eq!(NpuTrunkMode::Auto.resolve(arch), NpuTrunkMode::Off);
            for mode in [NpuTrunkMode::Off, NpuTrunkMode::On, NpuTrunkMode::Emulate] {
                assert_eq!(mode.resolve(arch), mode);
            }
        }
        assert!(!NpuTrunkMode::On.is_off());
    }

    #[test]
    fn parse_all_modes_roundtrip() {
        for mode in NpuTrunkMode::ALL {
            assert_eq!(mode.to_string().parse::<NpuTrunkMode>(), Ok(mode));
            assert_eq!(mode.to_string(), mode.as_str());
        }
        assert_eq!("off".parse(), Ok(NpuTrunkMode::Off));
        assert_eq!("auto".parse(), Ok(NpuTrunkMode::Auto));
        assert_eq!("on".parse(), Ok(NpuTrunkMode::On));
        assert_eq!("emulate".parse(), Ok(NpuTrunkMode::Emulate));
    }

    #[test]
    fn parse_is_case_and_space_insensitive() {
        assert_eq!(" EMULATE\n".parse(), Ok(NpuTrunkMode::Emulate));
        assert_eq!("Auto".parse(), Ok(NpuTrunkMode::Auto));
        assert_eq!("\tOn ".parse(), Ok(NpuTrunkMode::On));
    }

    #[test]
    fn parse_rejects_unknown_and_empty() {
        for bad in [
            "", " ", "1", "0", "true", "yes", "enable", "emulated", "of f",
        ] {
            let err = bad.parse::<NpuTrunkMode>().unwrap_err();
            assert!(err.contains("invalid npu trunk mode"), "{bad:?}: {err}");
        }
    }

    #[test]
    fn from_value_absent_is_auto_present_is_strict() {
        assert_eq!(NpuTrunkMode::from_value(None), Ok(NpuTrunkMode::Auto));
        assert_eq!(NpuTrunkMode::from_value(Some("on")), Ok(NpuTrunkMode::On));
        let err = NpuTrunkMode::from_value(Some("")).unwrap_err();
        assert!(err.starts_with(NPU_TRUNK_ENV), "{err}");
        let err = NpuTrunkMode::from_value(Some("maybe")).unwrap_err();
        assert!(err.contains("\"maybe\""), "{err}");
    }

    #[test]
    fn env_key_is_the_documented_developer_key() {
        assert_eq!(NPU_TRUNK_ENV, "HIPFIRE_QWEN4_NPU_TRUNK");
        assert_eq!(NPU_TRUNK_DAEMON_KEY, "npu_trunk");
        let field =
            hipfire_config::field(NPU_TRUNK_DAEMON_KEY).expect("registered npu_trunk schema");
        assert_eq!(field.env_compat, Some(NPU_TRUNK_ENV));
        assert!(matches!(
            field.default,
            hipfire_config::DefaultValue::String("auto")
        ));
        for mode in NpuTrunkMode::ALL {
            assert!(field
                .validate(&hipfire_config::ConfigValue::String(mode.to_string()))
                .is_ok());
        }
        assert!(field
            .validate(&hipfire_config::ConfigValue::String("invalid".into()))
            .is_err());
    }
}
