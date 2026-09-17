// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

use std::fmt;
use std::str::FromStr;

/// Public-queue fan-out policy for independent retained work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum QueuePolicy {
    #[default]
    Auto = 0,
    One = 1,
    Two = 2,
    Four = 4,
}

impl QueuePolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::One => "1",
            Self::Two => "2",
            Self::Four => "4",
        }
    }

    pub const fn explicit_lanes(self) -> Option<usize> {
        match self {
            Self::Auto => None,
            Self::One => Some(1),
            Self::Two => Some(2),
            Self::Four => Some(4),
        }
    }

    pub fn resolve(self, device_name: &str, independent_width: usize) -> usize {
        let available = independent_width.max(1);
        let requested = self
            .explicit_lanes()
            .unwrap_or_else(|| automatic_lane_limit(device_name));
        requested.min(available)
    }
}

/// Default lane count for `QueuePolicy::Auto`: the measured per-device table
/// in `crate::lanes` where the device is known, otherwise the pre-existing
/// family fallback (gfx12 family 2, gfx11 family 4, everything else 1).
///
/// Explicit `One`/`Two`/`Four` policies (including the `HIPFIRE_REPLAY_PM4_QUEUES`
/// env override that selects them) bypass this function entirely, so operator
/// overrides keep precedence over the measured default.
fn automatic_lane_limit(device_name: &str) -> usize {
    let normalized = device_name.to_ascii_lowercase();
    if let Some(measured) = crate::lanes::measured_lanes(&normalized) {
        return measured;
    }
    if normalized.starts_with("gfx12") {
        2
    } else if normalized.starts_with("gfx11") {
        4
    } else {
        1
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueuePolicyParseError(String);

impl fmt::Display for QueuePolicyParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown queue policy {:?}; expected auto, 1, 2, or 4",
            self.0
        )
    }
}

impl std::error::Error for QueuePolicyParseError {}

impl FromStr for QueuePolicy {
    type Err = QueuePolicyParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "1" | "one" => Ok(Self::One),
            "2" | "two" => Ok(Self::Two),
            "4" | "four" => Ok(Self::Four),
            _ => Err(QueuePolicyParseError(value.to_owned())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_policy_uses_certified_architecture_caps() {
        assert_eq!(QueuePolicy::Auto.resolve("gfx1100", 16), 4);
        assert_eq!(QueuePolicy::Auto.resolve("gfx1151", 16), 4);
        assert_eq!(QueuePolicy::Auto.resolve("gfx1201", 16), 2);
        // gfx1030 comes from the measured no-op sweep in crate::lanes (4
        // lanes at 0.0952 us/dispatch vs 1 lane at 0.2009); the family
        // fallback below only covers devices with no measured entry.
        assert_eq!(QueuePolicy::Auto.resolve("gfx1030", 16), 4);
        assert_eq!(QueuePolicy::Auto.resolve("gfx9999", 16), 1);
    }

    #[test]
    fn policy_never_exceeds_independent_width() {
        assert_eq!(QueuePolicy::Four.resolve("gfx1100", 2), 2);
        assert_eq!(QueuePolicy::Two.resolve("gfx1201", 1), 1);
        assert_eq!(QueuePolicy::Auto.resolve("gfx1151", 0), 1);
    }
}
