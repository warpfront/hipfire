// SPDX-License-Identifier: Apache-2.0
//! The user-facing switch of a co-processor stage: its names and the mode it selects.
//!
//! The mode borrows the central process-config snapshot without allocating.

use std::fmt;
use std::str::FromStr;

/// The names one stage answers to: the developer environment key, the config key, the log tag and the daemon key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Switch {
    /// Developer environment key of the mode.
    pub env: &'static str,
    /// Config registry key of the mode (`fusions.*`).
    pub config_key: &'static str,
    /// Prefix of the stage's log lines.
    pub tag: &'static str,
    /// Daemon configuration key of the mode; also the `=on`/`=auto` name in the log hints.
    pub name: &'static str,
}

/// How a co-processor stage reaches its device.
///
/// `Auto` is the default; the caller resolves it to `Off` where the stage is not admitted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Mode {
    /// The stage stays on the GPU; nothing co-processor related is touched.
    Off,
    /// Use the device when it is present and the plan predicts a gain, else fall back to the GPU.
    #[default]
    Auto,
    /// Require the device; failure to open it is an error.
    On,
    /// Run the bit-identical GPU emulation of the device numerics (no device).
    Emulate,
}

impl Mode {
    /// Every mode, in documentation order.
    pub const ALL: [Mode; 4] = [Self::Off, Self::Auto, Self::On, Self::Emulate];

    /// The canonical lower-case spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::On => "on",
            Self::Emulate => "emulate",
        }
    }

    /// `true` for [`Mode::Off`].
    pub const fn is_off(self) -> bool {
        matches!(self, Self::Off)
    }

    /// Resolves `Auto` before any stage-specific work: `Off` unless the caller admits `Auto` on this host.
    pub fn resolve(self, auto_admitted: bool) -> Self {
        if self == Self::Auto && !auto_admitted {
            Self::Off
        } else {
            self
        }
    }

    /// An absent key takes `default`; a present value must be valid.
    pub fn from_value(switch: Switch, value: Option<&str>, default: Self) -> Result<Self, String> {
        match value {
            None => Ok(default),
            Some(raw) => raw
                .parse::<Self>()
                .map_err(|e| format!("{}: {e}", switch.env)),
        }
    }

    /// Borrows the validated process snapshot; no per-load allocation or I/O.
    pub fn from_config(switch: Switch, default: Self) -> Result<Self, String> {
        let config = hipfire_config::active_or_local_process_config();
        let raw = match config.values.values.get(switch.config_key) {
            Some(hipfire_config::ConfigValue::String(raw)) => Some(raw.as_str()),
            _ => None,
        };
        Self::from_value(switch, raw, default)
    }
}

impl FromStr for Mode {
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
            "invalid mode {s:?} (expected off, auto, on or emulate)"
        ))
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST: Switch = Switch {
        env: "HIPFIRE_TEST_STAGE",
        config_key: "fusions.test_stage",
        tag: "test-stage",
        name: "test_stage",
    };

    #[test]
    fn default_auto_resolves_only_where_admitted() {
        assert_eq!(Mode::default(), Mode::Auto);
        assert_eq!(Mode::Auto.resolve(true), Mode::Auto);
        assert_eq!(Mode::Auto.resolve(false), Mode::Off);
        for mode in [Mode::Off, Mode::On, Mode::Emulate] {
            assert_eq!(mode.resolve(false), mode);
            assert_eq!(mode.resolve(true), mode);
        }
        assert!(Mode::Off.is_off() && !Mode::On.is_off());
    }

    #[test]
    fn parse_all_modes_roundtrip() {
        for mode in Mode::ALL {
            assert_eq!(mode.to_string().parse::<Mode>(), Ok(mode));
            assert_eq!(mode.to_string(), mode.as_str());
        }
        assert_eq!("off".parse(), Ok(Mode::Off));
        assert_eq!("auto".parse(), Ok(Mode::Auto));
        assert_eq!("on".parse(), Ok(Mode::On));
        assert_eq!("emulate".parse(), Ok(Mode::Emulate));
    }

    #[test]
    fn parse_is_case_and_space_insensitive() {
        assert_eq!(" EMULATE\n".parse(), Ok(Mode::Emulate));
        assert_eq!("Auto".parse(), Ok(Mode::Auto));
        assert_eq!("\tOn ".parse(), Ok(Mode::On));
    }

    #[test]
    fn parse_rejects_unknown_and_empty() {
        for bad in [
            "", " ", "1", "0", "true", "yes", "enable", "emulated", "of f",
        ] {
            let err = bad.parse::<Mode>().unwrap_err();
            assert!(err.contains("invalid mode"), "{bad:?}: {err}");
        }
    }

    #[test]
    fn from_value_absent_takes_default_present_is_strict() {
        assert_eq!(Mode::from_value(TEST, None, Mode::Auto), Ok(Mode::Auto));
        assert_eq!(Mode::from_value(TEST, None, Mode::Off), Ok(Mode::Off));
        assert_eq!(Mode::from_value(TEST, Some("on"), Mode::Auto), Ok(Mode::On));
        let err = Mode::from_value(TEST, Some(""), Mode::Auto).unwrap_err();
        assert!(err.starts_with(TEST.env), "{err}");
        let err = Mode::from_value(TEST, Some("maybe"), Mode::Auto).unwrap_err();
        assert!(err.contains("\"maybe\""), "{err}");
    }
}
