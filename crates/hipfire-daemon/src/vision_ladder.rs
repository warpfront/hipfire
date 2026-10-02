// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Vision-tower sidecar ladder shared by the ordinary and multi-slot load arms.

use hipfire_config::developer_var;

/// Daemon-side `vision_mode` gate for the tower sidecar path.
///
/// `off` (the default) is a hard override that drops even an explicit
/// sidecar, mirroring the `dflash_mode=off` draft guard at the load site.
/// Any other mode passes the `HIPFIRE_VISION_SIDECAR` / `params.vision`
/// ladder result through untouched. Pure string plumbing — no arch or
/// tensor knowledge; admission still validates the surviving path.
pub(crate) fn apply_vision_mode_gate(
    vision_mode: &str,
    raw_vision: Option<String>,
) -> Option<String> {
    if vision_mode == "off" {
        None
    } else {
        raw_vision
    }
}

/// Resolve the vision-tower ladder for a load request.
///
/// Returns `(vision_mode, gated_sidecar, suppressed_sidecar)`:
/// `vision_mode` is `params.vision_mode` (default `off`); `gated_sidecar` is
/// the `HIPFIRE_VISION_SIDECAR` (non-empty wins, empty opts out) →
/// `params.vision` ladder result with `apply_vision_mode_gate` applied; and
/// `suppressed_sidecar` is the explicit sidecar that `off` dropped, so the
/// caller can name it in a later "no vision encoder" error.
///
/// Both load arms (ordinary and multi-slot) MUST use this — the multi-slot
/// arm used to discover a `.vl` sibling of its own accord and never read
/// either knob, so `vision_mode=off` still paid the tower's ~1 GB and
/// `serve --vision` was silently ignored on that route.
pub(crate) fn resolve_vision_ladder(
    msg: &serde_json::Value,
) -> (String, Option<String>, Option<String>) {
    let vision_mode = msg
        .get("params")
        .and_then(|p| p.get("vision_mode"))
        .and_then(|v| v.as_str())
        .unwrap_or("off")
        .to_string();
    let env_vision = developer_var("HIPFIRE_VISION_SIDECAR").ok();
    let raw_vision: Option<String> = match env_vision.as_deref() {
        Some("") => None,
        Some(p) => Some(p.to_string()),
        None => msg
            .get("params")
            .and_then(|p| p.get("vision"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string()),
    };
    let suppressed = if vision_mode == "off" {
        if let Some(v) = raw_vision.as_deref() {
            eprintln!("[hipfire-daemon] vision_mode=off — skipping tower sidecar load ({v})");
            Some(v.to_string())
        } else {
            None
        }
    } else {
        None
    };
    let gated = apply_vision_mode_gate(&vision_mode, raw_vision);
    (vision_mode, gated, suppressed)
}
