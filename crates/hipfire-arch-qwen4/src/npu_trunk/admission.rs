// SPDX-License-Identifier: Apache-2.0
//! The admission barrier keeps all weight/device work behind cheap checks.

/// Why the trunk did not arm.
#[derive(Debug)]
pub(crate) enum Refusal<E> {
    /// The model, placement or scratch plan does not qualify; the NPU was never involved.
    Ineligible(E),
    /// The model qualifies but the NPU path (firmware, calibration, preparation, runtime open) failed.
    Failed(E),
}

pub(crate) fn prepare_after_checks<D, C, T, E>(
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

/// Auto refusals are GPU-only, including preparation/open errors: an ineligible model logs one plain
/// `npu-trunk: off (<reason>)` line, a failure to arm a qualifying model logs it at WARN level. Forced
/// modes fail, which fails the load.
pub(crate) fn finish<T, E: std::fmt::Display>(
    mode: super::NpuTrunkMode,
    result: Result<Option<T>, Refusal<E>>,
) -> Result<Option<T>, E> {
    let (failed, error) = match result {
        Ok(armed) => return Ok(armed),
        Err(Refusal::Ineligible(error)) => (false, error),
        Err(Refusal::Failed(error)) => (true, error),
    };
    if mode != super::NpuTrunkMode::Auto {
        return Err(error);
    }
    let rendered = error.to_string();
    let reason = rendered
        .split_once("npu-trunk: ")
        .map_or(rendered.as_str(), |(_, reason)| reason);
    eprintln!("{}", off_line(failed, reason));
    Ok(None)
}

fn off_line(failed: bool, reason: &str) -> String {
    if failed {
        format!("WARN npu-trunk: off ({reason}); npu_trunk=auto requested the NPU but the trunk did not arm, prefill runs GPU-only (npu_trunk=on makes this a load error)")
    } else {
        format!("npu-trunk: off ({reason})")
    }
}

#[cfg(test)]
mod tests {
    use super::super::NpuTrunkMode;
    use super::*;
    use std::cell::RefCell;
    #[test]
    fn checks_precede_download_pack_alloc_and_open() {
        for failed in 0..=4 {
            let calls = RefCell::new(Vec::new());
            let stage = |n| {
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
                    stage(3)?; // download, pack and allocate only beyond the barrier
                    stage(4) // runtime open
                },
            );
            match &result {
                Err(Refusal::Ineligible(_)) => assert_eq!(failed, 0),
                Err(Refusal::Failed(_)) => assert!(failed > 0),
                Ok(()) => unreachable!("every iteration refuses"),
            }
            assert_eq!(*calls.borrow(), (0..=failed).collect::<Vec<_>>());
            assert!(finish(NpuTrunkMode::Auto, result.map(Some))
                .unwrap()
                .is_none());
        }
        assert!(finish::<(), _>(NpuTrunkMode::On, Err(Refusal::Failed("open failed"))).is_err());
        assert!(
            finish::<(), _>(NpuTrunkMode::On, Err(Refusal::Ineligible("not flash-next"))).is_err()
        );
    }

    #[test]
    fn auto_failure_to_arm_is_a_warning_and_ineligible_stays_plain() {
        assert_eq!(
            off_line(false, "requires Flash-Next family B"),
            "npu-trunk: off (requires Flash-Next family B)"
        );
        let warn = off_line(true, "mmap heap: errno=11");
        assert!(
            warn.starts_with("WARN npu-trunk: off (mmap heap: errno=11); npu_trunk=auto"),
            "{warn}"
        );
    }
}
