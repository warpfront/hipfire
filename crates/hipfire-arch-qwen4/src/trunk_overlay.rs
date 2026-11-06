// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Load-time policy for the `<model>.trunk-overlay` sidecar of a Qwen4
//! Flash-Next model (see `hipfire_runtime::trunk_overlay` for the container
//! and its binding).
//!
//! [`resolve_trunk_overlay`] decides whether the sidecar next to a model
//! applies to this load.  An absent sidecar is silent (a load without one logs
//! exactly as before); every other outcome prints exactly one
//! `  qwen4 trunk overlay: ` line, except an accepted overlay, whose
//! application logs ([`crate::weights::Qwen4Weights::apply_trunk_overlay`]).
//! Only a bad `HIPFIRE_QWEN4_TRUNK_OVERLAY` value is an error: a sidecar that
//! cannot be used never fails a load, the base weights run.

use std::path::Path;

use hipfire_config::memory::trunk_overlay_sidecar_path;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::trunk_overlay::{bind_trunk_overlay, BoundTrunkOverlay, TrunkOverlayBase};
use rdna_compute::TrunkIu4Default;

use crate::config::Qwen4Config;
use crate::weights::TrunkOverlayApplied;

/// The developer variable selecting the load policy.
pub const TRUNK_OVERLAY_ENV: &str = "HIPFIRE_QWEN4_TRUNK_OVERLAY";

/// Log prefix of every trunk overlay line.
pub(crate) const LOG_PREFIX: &str = "  qwen4 trunk overlay: ";

/// `HIPFIRE_QWEN4_TRUNK_OVERLAY`: whether a present sidecar is applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrunkOverlayPolicy {
    /// Unset or `auto`: apply only on the exact architecture of the overlay's
    /// trunk default.
    Auto,
    /// `0`: never apply (kill switch).
    Off,
    /// `1`: apply on any architecture.
    On,
}

impl TrunkOverlayPolicy {
    /// Parse the raw variable value (`None` = unset).
    pub fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw {
            None | Some("auto") => Ok(Self::Auto),
            Some("0") => Ok(Self::Off),
            Some("1") => Ok(Self::On),
            Some(other) => Err(format!(
                "{TRUNK_OVERLAY_ENV}={other:?} is not one of auto, 0, 1"
            )),
        }
    }

    /// The policy from the process configuration.
    pub fn from_env() -> Result<Self, String> {
        Self::parse(
            hipfire_config::developer_var(TRUNK_OVERLAY_ENV)
                .ok()
                .as_deref(),
        )
    }
}

/// A sidecar bound to the base and accepted for this load.
#[derive(Debug)]
pub struct Qwen4TrunkOverlay {
    pub(crate) bound: BoundTrunkOverlay,
    pub(crate) default: TrunkIu4Default,
}

impl Qwen4TrunkOverlay {
    /// The trunk default the overlay arms.
    pub fn default_marker(&self) -> TrunkIu4Default {
        self.default
    }

    /// Device bytes the overlay adds to the base weights once applied (see
    /// [`BoundTrunkOverlay::net_extra_bytes`]); a placement that budgets before
    /// `attach_forward` charges `max(0, ..)` of it.
    pub fn net_extra_bytes(&self) -> i64 {
        self.bound.net_extra_bytes()
    }

    pub fn bound(&self) -> &BoundTrunkOverlay {
        &self.bound
    }

    pub fn path(&self) -> &Path {
        self.bound.path()
    }
}

/// What to do with a bound sidecar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum OverlayDecision {
    /// Apply it, arming this default.
    Apply(TrunkIu4Default),
    /// Present but not applied: the text after `<path> present, `.
    Skip(String),
    /// Refused: the reason after `REFUSED <path>: `.
    Refuse(String),
}

/// The pure decision for a sidecar already bound to its base.
pub(crate) fn decide_overlay(
    policy: TrunkOverlayPolicy,
    gpu_arch: &str,
    marker: &str,
    flash_next: bool,
    base_default: Option<TrunkIu4Default>,
) -> OverlayDecision {
    if policy == TrunkOverlayPolicy::Off {
        return OverlayDecision::Skip(format!("not applied ({TRUNK_OVERLAY_ENV}=0)"));
    }
    let Some(default) = TrunkIu4Default::from_marker(marker) else {
        return OverlayDecision::Refuse(format!("unknown trunk default marker {marker:?}"));
    };
    if !flash_next {
        return OverlayDecision::Refuse("profile is not Flash-Next".to_string());
    }
    if let Some(existing) = base_default {
        return OverlayDecision::Refuse(format!(
            "base already carries the trunk default marker {:?}",
            existing.marker()
        ));
    }
    if policy == TrunkOverlayPolicy::Auto && gpu_arch != default.arch() {
        return OverlayDecision::Skip(format!(
            "not applied on {gpu_arch} (overlay default {} is {} only; \
             {TRUNK_OVERLAY_ENV}=1 applies it)",
            default.marker(),
            default.arch()
        ));
    }
    OverlayDecision::Apply(default)
}

/// The log line (without newline) of a non-applying outcome.
pub(crate) fn overlay_line(path: &Path, decision: &OverlayDecision) -> Option<String> {
    match decision {
        OverlayDecision::Apply(_) => None,
        OverlayDecision::Skip(text) => Some(format!(
            "{LOG_PREFIX}{} present, {text}; base weights",
            path.display()
        )),
        OverlayDecision::Refuse(reason) => Some(overlay_refusal_line(path, reason)),
    }
}

/// `REFUSED <path>: <reason>; base weights`.
pub(crate) fn overlay_refusal_line(path: &Path, reason: &str) -> String {
    format!(
        "{LOG_PREFIX}REFUSED {}: {reason}; base weights",
        path.display()
    )
}

/// The line logged when an overlay was applied.
pub(crate) fn applied_line(overlay: &Qwen4TrunkOverlay, applied: &TrunkOverlayApplied) -> String {
    applied_line_for(
        overlay.path(),
        overlay.default,
        &overlay.bound.manifest().base,
        applied,
    )
}

fn applied_line_for(
    path: &Path,
    default: TrunkIu4Default,
    base: &TrunkOverlayBase,
    applied: &TrunkOverlayApplied,
) -> String {
    let tensors = match (applied.tensors - applied.qt3_tensors, applied.qt3_tensors) {
        (qt44, 0) => format!("{qt44} QT44"),
        (0, qt3) => format!("{qt3} QT3 (Q8F16)"),
        (qt44, qt3) => format!("{qt44} QT44 + {qt3} QT3 (Q8F16)"),
    };
    format!(
        "{LOG_PREFIX}applied {tensors} tensors from {} (default {}; bound to base {} B, \
         index sha256 {}, file sha256 {}; {} -> {} bytes)",
        path.display(),
        default.marker(),
        base.file_bytes,
        short(&base.index_sha256),
        short(&base.sha256),
        applied.before_bytes,
        applied.after_bytes
    )
}

/// The first 12 characters of a digest.
fn short(hex: &str) -> &str {
    hex.get(..12).unwrap_or(hex)
}

/// Resolve the trunk overlay sidecar next to `model_path` for this load.
///
/// `hfq` is the open base: the binding reads its length and index digest
/// (positional reads, so a dropped mapping is fine).
/// `Ok(None)`: absent, kill-switched, not applicable here or refused (each
/// but the first logged).  `Err` only for a bad `HIPFIRE_QWEN4_TRUNK_OVERLAY`.
pub fn resolve_trunk_overlay(
    hfq: &HfqFile,
    model_path: &Path,
    gpu_arch: &str,
    config: &Qwen4Config,
) -> Result<Option<Qwen4TrunkOverlay>, String> {
    resolve_overlay(hfq, model_path, gpu_arch, config, true)
}

/// Resident device bytes the sidecar next to `model_path` adds to the base
/// weights once applied: `max(0, net_extra_bytes)` of the overlay
/// [`resolve_trunk_overlay`] would accept for this load, `0` when none applies.
/// Silent (it logs nothing: the load's own resolve logs the outcome once), so
/// plan-time placement and per-rank budgets can charge the same number the
/// load will add after placement ran.
pub fn trunk_overlay_extra_bytes(
    hfq: &HfqFile,
    model_path: &Path,
    gpu_arch: &str,
    config: &Qwen4Config,
) -> Result<u64, String> {
    Ok(
        resolve_overlay(hfq, model_path, gpu_arch, config, false)?.map_or(0, |overlay| {
            u64::try_from(overlay.net_extra_bytes()).unwrap_or(0)
        }),
    )
}

fn resolve_overlay(
    hfq: &HfqFile,
    model_path: &Path,
    gpu_arch: &str,
    config: &Qwen4Config,
    log: bool,
) -> Result<Option<Qwen4TrunkOverlay>, String> {
    let say = |line: String| {
        if log {
            eprintln!("{line}");
        }
    };
    let policy = TrunkOverlayPolicy::from_env()?;
    let path = trunk_overlay_sidecar_path(model_path);
    match std::fs::metadata(&path) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            say(overlay_refusal_line(
                &path,
                &format!("cannot stat the sidecar: {error}"),
            ));
            return Ok(None);
        }
    }
    if policy == TrunkOverlayPolicy::Off {
        let decision = decide_overlay(policy, gpu_arch, "", false, None);
        if let Some(line) = overlay_line(&path, &decision) {
            say(line);
        }
        return Ok(None);
    }
    let bound = match bind_trunk_overlay(hfq, &path) {
        Ok(bound) => bound,
        Err(reason) => {
            say(overlay_refusal_line(&path, &reason));
            return Ok(None);
        }
    };
    let decision = decide_overlay(
        policy,
        gpu_arch,
        bound.marker(),
        config.profile.is_flash_next(),
        config.trunk_iu4_default,
    );
    match decision {
        OverlayDecision::Apply(default) => Ok(Some(Qwen4TrunkOverlay { bound, default })),
        other => {
            if let Some(line) = overlay_line(&path, &other) {
                say(line);
            }
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MARKER: &str = "gfx1151-zab-v1";

    #[test]
    fn policy_parses_the_documented_values() {
        assert_eq!(
            TrunkOverlayPolicy::parse(None),
            Ok(TrunkOverlayPolicy::Auto)
        );
        assert_eq!(
            TrunkOverlayPolicy::parse(Some("auto")),
            Ok(TrunkOverlayPolicy::Auto)
        );
        assert_eq!(
            TrunkOverlayPolicy::parse(Some("0")),
            Ok(TrunkOverlayPolicy::Off)
        );
        assert_eq!(
            TrunkOverlayPolicy::parse(Some("1")),
            Ok(TrunkOverlayPolicy::On)
        );
        for bad in ["", "2", "on", "AUTO", "true", " 1"] {
            let error = TrunkOverlayPolicy::parse(Some(bad)).unwrap_err();
            assert!(error.contains("HIPFIRE_QWEN4_TRUNK_OVERLAY"), "{error}");
        }
    }

    #[test]
    fn auto_applies_only_on_the_default_arch() {
        let auto = TrunkOverlayPolicy::Auto;
        assert_eq!(
            decide_overlay(auto, "gfx1151", MARKER, true, None),
            OverlayDecision::Apply(TrunkIu4Default::Gfx1151ZabV1)
        );
        assert_eq!(
            decide_overlay(auto, "gfx1201", MARKER, true, None),
            OverlayDecision::Skip(
                "not applied on gfx1201 (overlay default gfx1151-zab-v1 is gfx1151 only; \
                 HIPFIRE_QWEN4_TRUNK_OVERLAY=1 applies it)"
                    .to_string()
            )
        );
    }

    #[test]
    fn the_q8f16_marker_applies_under_auto_on_gfx1201_only() {
        let auto = TrunkOverlayPolicy::Auto;
        assert_eq!(
            decide_overlay(auto, "gfx1201", "gfx1201-q8f16-v1", true, None),
            OverlayDecision::Apply(TrunkIu4Default::Gfx1201Q8f16V1)
        );
        assert_eq!(
            decide_overlay(auto, "gfx1151", "gfx1201-q8f16-v1", true, None),
            OverlayDecision::Skip(
                "not applied on gfx1151 (overlay default gfx1201-q8f16-v1 is gfx1201 only; \
                 HIPFIRE_QWEN4_TRUNK_OVERLAY=1 applies it)"
                    .to_string()
            )
        );
    }

    #[test]
    fn on_applies_on_any_arch() {
        assert_eq!(
            decide_overlay(TrunkOverlayPolicy::On, "gfx1201", MARKER, true, None),
            OverlayDecision::Apply(TrunkIu4Default::Gfx1151ZabV1)
        );
    }

    #[test]
    fn off_never_applies() {
        for arch in ["gfx1151", "gfx1201"] {
            assert_eq!(
                decide_overlay(TrunkOverlayPolicy::Off, arch, MARKER, true, None),
                OverlayDecision::Skip("not applied (HIPFIRE_QWEN4_TRUNK_OVERLAY=0)".to_string())
            );
        }
    }

    #[test]
    fn refuses_unknown_marker_profile_and_marked_base() {
        for policy in [TrunkOverlayPolicy::Auto, TrunkOverlayPolicy::On] {
            assert_eq!(
                decide_overlay(policy, "gfx1151", "gfx1151-zab-v2", true, None),
                OverlayDecision::Refuse(
                    "unknown trunk default marker \"gfx1151-zab-v2\"".to_string()
                )
            );
            assert_eq!(
                decide_overlay(policy, "gfx1151", MARKER, false, None),
                OverlayDecision::Refuse("profile is not Flash-Next".to_string())
            );
            assert_eq!(
                decide_overlay(
                    policy,
                    "gfx1151",
                    MARKER,
                    true,
                    Some(TrunkIu4Default::Gfx1151ZabV1)
                ),
                OverlayDecision::Refuse(
                    "base already carries the trunk default marker \"gfx1151-zab-v1\"".to_string()
                )
            );
        }
    }

    #[test]
    fn applied_line_names_the_binding_and_the_byte_delta() {
        let base = TrunkOverlayBase {
            file_bytes: 125_144_775_192,
            index_sha256: "0123456789abcdef".repeat(4),
            sha256: "fedcba9876543210".repeat(4),
        };
        let applied = TrunkOverlayApplied {
            tensors: 108,
            qt3_tensors: 0,
            before_bytes: 400,
            after_bytes: 300,
        };
        assert_eq!(
            applied_line_for(
                Path::new("/m/q.trunk-overlay"),
                TrunkIu4Default::Gfx1151ZabV1,
                &base,
                &applied
            ),
            "  qwen4 trunk overlay: applied 108 QT44 tensors from /m/q.trunk-overlay \
             (default gfx1151-zab-v1; bound to base 125144775192 B, index sha256 0123456789ab, \
             file sha256 fedcba987654; 400 -> 300 bytes)"
        );
    }

    #[test]
    fn applied_line_names_the_q8f16_tensors() {
        let base = TrunkOverlayBase {
            file_bytes: 7,
            index_sha256: "0123456789abcdef".repeat(4),
            sha256: "fedcba9876543210".repeat(4),
        };
        let line = |tensors, qt3_tensors| {
            applied_line_for(
                Path::new("/m/q.trunk-overlay"),
                TrunkIu4Default::Gfx1151ZabV1,
                &base,
                &TrunkOverlayApplied {
                    tensors,
                    qt3_tensors,
                    before_bytes: 1,
                    after_bytes: 2,
                },
            )
        };
        assert!(line(60, 60).contains("applied 60 QT3 (Q8F16) tensors from "));
        assert!(line(60, 12).contains("applied 48 QT44 + 12 QT3 (Q8F16) tensors from "));
        assert!(line(60, 0).contains("applied 60 QT44 tensors from "));
    }

    #[test]
    fn log_lines_match_the_contract() {
        let path = Path::new("/m/q.mq4-xts.trunk-overlay");
        assert_eq!(
            overlay_line(
                path,
                &decide_overlay(TrunkOverlayPolicy::Off, "gfx1151", MARKER, true, None)
            )
            .unwrap(),
            "  qwen4 trunk overlay: /m/q.mq4-xts.trunk-overlay present, not applied \
             (HIPFIRE_QWEN4_TRUNK_OVERLAY=0); base weights"
        );
        assert_eq!(
            overlay_line(
                path,
                &decide_overlay(TrunkOverlayPolicy::Auto, "gfx1201", MARKER, true, None)
            )
            .unwrap(),
            "  qwen4 trunk overlay: /m/q.mq4-xts.trunk-overlay present, not applied on gfx1201 \
             (overlay default gfx1151-zab-v1 is gfx1151 only; HIPFIRE_QWEN4_TRUNK_OVERLAY=1 \
             applies it); base weights"
        );
        assert_eq!(
            overlay_refusal_line(path, "base is 1 bytes, overlay binds 2"),
            "  qwen4 trunk overlay: REFUSED /m/q.mq4-xts.trunk-overlay: base is 1 bytes, \
             overlay binds 2; base weights"
        );
        assert_eq!(
            overlay_line(
                path,
                &decide_overlay(TrunkOverlayPolicy::Auto, "gfx1151", MARKER, true, None)
            ),
            None
        );
    }
}
