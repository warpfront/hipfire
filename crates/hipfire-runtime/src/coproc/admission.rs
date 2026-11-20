// SPDX-License-Identifier: Apache-2.0
//! The admission barrier keeps all weight and device work behind cheap checks.

use super::{Mode, Switch};
use std::fmt::Display;

/// Why a stage did not arm.
#[derive(Debug)]
pub enum Refusal<E> {
    /// The model, placement or scratch plan does not qualify; the device was never involved.
    Ineligible(E),
    /// The model qualifies but the device path (firmware, calibration, preparation, runtime open) failed.
    Failed(E),
}

/// Runs the four steps in order and stops at the first failure: `eligible` refuses as `Ineligible`, every later step as `Failed`.
pub fn prepare_after_checks<D, C, T, E>(
    eligible: impl FnOnce() -> Result<(), E>,
    device: impl FnOnce() -> Result<D, E>,
    calibration: impl FnOnce(D) -> Result<C, E>,
    prepare: impl FnOnce(C) -> Result<T, E>,
) -> Result<T, Refusal<E>> {
    eligible().map_err(Refusal::Ineligible)?;
    let device = device().map_err(Refusal::Failed)?;
    let calibration = calibration(device).map_err(Refusal::Failed)?;
    prepare(calibration).map_err(Refusal::Failed)
}

/// Settles a stage's admission. Under `auto` a refusal is GPU-only, including preparation and open errors: an
/// ineligible model logs one plain `{tag}: off (<reason>)` line, and a failure to arm a qualifying model logs it at
/// WARN level. Any other mode returns the error, which fails the load.
pub fn finish<T, E: Display>(
    switch: Switch,
    mode: Mode,
    result: Result<Option<T>, Refusal<E>>,
) -> Result<Option<T>, E> {
    let (failed, error) = match result {
        Ok(armed) => return Ok(armed),
        Err(Refusal::Ineligible(error)) => (false, error),
        Err(Refusal::Failed(error)) => (true, error),
    };
    if mode != Mode::Auto {
        return Err(error);
    }
    let rendered = error.to_string();
    let prefix = format!("{}: ", switch.tag);
    let reason = rendered
        .split_once(prefix.as_str())
        .map_or(rendered.as_str(), |(_, reason)| reason);
    eprintln!("{}", off_line(switch, failed, reason));
    Ok(None)
}

/// The one line a stage that stays off logs. `failed` marks a qualifying model whose stage did not arm under `auto`:
/// that line is a WARN and names the key that turns the failure into a load error.
pub fn off_line(switch: Switch, failed: bool, reason: &str) -> String {
    let Switch { tag, name, .. } = switch;
    if failed {
        format!(
            "WARN {tag}: off ({reason}); {name}=auto requested the NPU but the trunk did not arm, prefill runs GPU-only ({name}=on makes this a load error)"
        )
    } else {
        format!("{tag}: off ({reason})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    const TEST: Switch = Switch {
        env: "HIPFIRE_TEST_STAGE",
        config_key: "fusions.test_stage",
        tag: "test-stage",
        name: "test_stage",
    };

    #[test]
    fn checks_precede_download_pack_alloc_and_open() {
        // Step `failed` refuses; no later step may run.
        for failed in 0..=4 {
            let calls = RefCell::new(Vec::new());
            let stage = |n: usize| {
                calls.borrow_mut().push(n);
                if n == failed {
                    Err("refused")
                } else {
                    Ok(())
                }
            };
            let result = prepare_after_checks(
                || stage(0),
                || stage(1),
                |_| stage(2),
                |_| {
                    stage(3)?;
                    stage(4)
                },
            );
            assert_eq!(matches!(&result, Err(Refusal::Ineligible(_))), failed == 0);
            assert_eq!(matches!(&result, Err(Refusal::Failed(_))), failed > 0);
            assert_eq!(*calls.borrow(), (0..=failed).collect::<Vec<_>>());
            assert!(finish(TEST, Mode::Auto, result.map(Some))
                .unwrap()
                .is_none());
        }
        assert!(finish::<(), _>(TEST, Mode::On, Err(Refusal::Failed("open failed"))).is_err());
        assert!(
            finish::<(), _>(TEST, Mode::On, Err(Refusal::Ineligible("not flash-next"))).is_err()
        );
    }

    #[test]
    fn auto_failure_to_arm_is_a_warning_and_ineligible_stays_plain() {
        assert_eq!(
            off_line(TEST, false, "requires family B"),
            "test-stage: off (requires family B)"
        );
        let warn = off_line(TEST, true, "mmap heap: errno=11");
        assert!(
            warn.starts_with("WARN test-stage: off (mmap heap: errno=11); test_stage=auto"),
            "{warn}"
        );
    }
}
