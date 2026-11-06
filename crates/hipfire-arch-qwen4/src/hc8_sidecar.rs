// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Load-time policy for the `<model>.hc8` sidecar of a Qwen4 Flash-Next model
//! (see `hipfire_runtime::hc8_sidecar` for the container and its binding): the
//! BF16 hyper-connection mixer weights replaced by OCP E4M3 rows with
//! per-row power-of-two scales.
//!
//! [`resolve_hc8_sidecar`] decides whether the sidecar next to a model applies
//! to this load.  An absent sidecar, a disabled kill switch
//! (`HIPFIRE_QWEN4_HC8=0`, passed as `enabled == false`) and a non-Flash-Next
//! profile are silent (the load logs exactly as before); a present sidecar
//! that fails to bind logs one `REFUSED` line and the BF16 weights run.  An
//! accepted sidecar logs when applied
//! ([`crate::bundle::Qwen4Bundle::attach_forward`]).

use std::path::Path;

use hipfire_config::memory::hc8_sidecar_path;
use hipfire_runtime::hc8_sidecar::{bind_hc8_sidecar, BoundHc8Sidecar};
use hipfire_runtime::hfq::HfqFile;

use crate::config::Qwen4Config;
use crate::weights::Hc8Applied;

/// Log prefix of every hc8 sidecar line.
pub(crate) const LOG_PREFIX: &str = "  qwen4 hc8: ";

/// A sidecar bound to the base and accepted for this load.
#[derive(Debug)]
pub struct Qwen4Hc8Sidecar {
    pub(crate) bound: BoundHc8Sidecar,
}

impl Qwen4Hc8Sidecar {
    pub fn bound(&self) -> &BoundHc8Sidecar {
        &self.bound
    }

    pub fn path(&self) -> &Path {
        self.bound.path()
    }

    pub fn tensor_count(&self) -> usize {
        self.bound.tensor_count()
    }
}

/// `  qwen4 hc8: REFUSED <path>: <reason>; BF16 base weights`.
pub(crate) fn hc8_refusal_line(path: &Path, reason: &str) -> String {
    format!(
        "{LOG_PREFIX}REFUSED {}: {reason}; BF16 base weights",
        path.display()
    )
}

/// The line logged when the sidecar was applied.
pub(crate) fn hc8_applied_line(applied: &Hc8Applied) -> String {
    let mut line = format!(
        "{LOG_PREFIX}applied {} tensors (before {} B, after {} B)",
        applied.tensors, applied.before_bytes, applied.after_bytes
    );
    if applied.skipped > 0 {
        line.push_str(&format!(", {} skipped", applied.skipped));
    }
    line
}

/// Resolve the hc8 sidecar next to `model_path` for this load.
///
/// `enabled` is `gpu.flags.qwen4_hc8_enabled()` (kill switch on and exact
/// gfx1201).  `Ok(None)`: absent, disabled, not Flash-Next, or refused (the
/// last logged).  `Err` is reserved for the signature's symmetry with the
/// trunk overlay resolver; no current path returns it.
pub fn resolve_hc8_sidecar(
    hfq: &HfqFile,
    model_path: &Path,
    enabled: bool,
    config: &Qwen4Config,
) -> Result<Option<Qwen4Hc8Sidecar>, String> {
    resolve(hfq, model_path, enabled, config, true)
}

/// Device bytes the sidecar next to `model_path` SAVES once applied:
/// `replaced - payload` of the sidecar [`resolve_hc8_sidecar`] would accept
/// for this load, `0` when none applies.  Silent (the load's own resolve logs
/// the outcome once).
///
/// Census contract: an applied HC8 sidecar is a net saving (each BF16 `[m, k]`
/// becomes `m*k + 4*m` bytes), never a charge.  Plan-time placement keeps
/// charging the manifest's BF16 sizes (conservative: the saving only appears
/// after placement ran, so a net saving never raises placement), and the lazy
/// F16 weight shadows are unchanged (`2*m*k` per owner, built from the E4M3
/// weight).  The saving counts every sidecar tensor, including ones a
/// pipeline band or a disabled MTP leaves unloaded, so it is an upper bound
/// and must never be subtracted from a budget.
pub fn hc8_sidecar_saved_bytes(
    hfq: &HfqFile,
    model_path: &Path,
    enabled: bool,
    config: &Qwen4Config,
) -> Result<u64, String> {
    Ok(resolve(hfq, model_path, enabled, config, false)?
        .map_or(0, |sidecar| {
            sidecar
                .bound
                .replaced_bytes()
                .saturating_sub(sidecar.bound.payload_bytes())
        }))
}

fn resolve(
    hfq: &HfqFile,
    model_path: &Path,
    enabled: bool,
    config: &Qwen4Config,
    log: bool,
) -> Result<Option<Qwen4Hc8Sidecar>, String> {
    if !enabled || !config.profile.is_flash_next() {
        return Ok(None);
    }
    let say = |line: String| {
        if log {
            eprintln!("{line}");
        }
    };
    let path = hc8_sidecar_path(model_path);
    match std::fs::metadata(&path) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            say(hc8_refusal_line(
                &path,
                &format!("cannot stat the sidecar: {error}"),
            ));
            return Ok(None);
        }
    }
    match bind_hc8_sidecar(hfq, &path) {
        Ok(bound) => Ok(Some(Qwen4Hc8Sidecar { bound })),
        Err(reason) => {
            say(hc8_refusal_line(&path, &reason));
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusal_line_names_the_path_reason_and_fallback() {
        assert_eq!(
            hc8_refusal_line(Path::new("/m/x.hfq.hc8"), "bad digest"),
            "  qwen4 hc8: REFUSED /m/x.hfq.hc8: bad digest; BF16 base weights"
        );
    }

    #[test]
    fn applied_line_counts_skips_only_when_present() {
        let mut applied = Hc8Applied {
            tensors: 98,
            skipped: 0,
            before_bytes: 2000,
            after_bytes: 1000,
        };
        assert_eq!(
            hc8_applied_line(&applied),
            "  qwen4 hc8: applied 98 tensors (before 2000 B, after 1000 B)"
        );
        applied.skipped = 4;
        assert_eq!(
            hc8_applied_line(&applied),
            "  qwen4 hc8: applied 98 tensors (before 2000 B, after 1000 B), 4 skipped"
        );
    }
}
