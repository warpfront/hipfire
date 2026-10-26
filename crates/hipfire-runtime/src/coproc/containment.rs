// SPDX-License-Identifier: Apache-2.0
//! Co-processor fault containment: the process-wide route latch and the GPU-only rerun of a faulted request.

/// What failed inside a pair: the NPU runtime (chain watchdog, submit, ERT state, relay) or the GPU (a HIP error, whose
/// context is sticky, so a GPU-only rerun could only fail too).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultClass {
    NpuRuntime,
    GpuHip,
}

impl FaultClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NpuRuntime => "NPU runtime",
            Self::GpuHip => "GPU HipError",
        }
    }
}

/// The route's fault latch. The first pair fault of the process stores its reason and turns the route off for good. An
/// NPU runtime fault also marks the request in flight so its entry point reruns it GPU-only; a GPU HipError never does
/// (the request fails with its original error).
#[derive(Default)]
pub struct FaultLatch {
    fault: Option<String>,
    request: bool,
}

impl FaultLatch {
    /// Records a pair fault. Returns true only for the first trip of the process (the caller logs once).
    pub fn trip(&mut self, class: FaultClass, reason: &str) -> bool {
        if class == FaultClass::NpuRuntime {
            self.request = true;
        }
        if self.fault.is_some() {
            return false;
        }
        self.fault = Some(reason.to_string());
        true
    }

    pub fn tripped(&self) -> bool {
        self.fault.is_some()
    }

    pub fn reason(&self) -> Option<&str> {
        self.fault.as_deref()
    }

    /// Whether the request in flight faulted; clears the per-request flag.
    pub fn take_request_fault(&mut self) -> bool {
        std::mem::take(&mut self.request)
    }

    /// A new request starts: forget the previous request's fault mark (the process latch stays).
    pub fn begin_request(&mut self) {
        self.request = false;
    }
}

/// Runs `attempt(ctx, start)`; when it fails and the failure tripped the fault latch (`take_fault`), runs it once more
/// from `replan(ctx)`. `tag` prefixes the restart log line.
///
/// The start is re-planned because the faulted attempt consumed and mutated the live state: the session driver's
/// `begin` took the live continuation, the chunks already forwarded advanced the state, and any uncommitted snapshots
/// of the attempt are dropped by the next turn's start. The only valid reuse points after the fault are the committed
/// snapshots (or a cold start), which is what `replan` returns.
///
/// The rerun is byte-identical to a GPU-only run of the same request from that start: the latched route plans no pair
/// (the scheduling is exactly the `off` one) and the session machinery restores the chosen snapshot or resets the state
/// before the first row, so nothing the NPU wrote can reach the rerun. A failure that did not trip the latch (abort,
/// validation, GPU error, `on` mode) is returned unchanged and never rerun.
pub fn rerun_after_fault<C: ?Sized, T, E>(
    tag: &str,
    ctx: &mut C,
    rows: usize,
    start: usize,
    mut attempt: impl FnMut(&mut C, usize) -> Result<T, E>,
    take_fault: impl Fn(&mut C) -> bool,
    replan: impl Fn(&C) -> usize,
) -> Result<T, E> {
    match attempt(ctx, start) {
        Err(error) => {
            if !take_fault(ctx) {
                return Err(error);
            }
            let restart = replan(ctx);
            eprintln!("{tag}: the faulted request restarts GPU-only from row {restart} of {rows}");
            attempt(ctx, restart)
        }
        ok => ok,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_fault_error_is_not_rerun() {
        let mut calls = 0;
        let out: Result<(), &str> = rerun_after_fault(
            "test",
            &mut calls,
            4,
            0,
            |calls, _| {
                *calls += 1;
                Err("abort")
            },
            |_| false,
            |_| 0,
        );
        assert_eq!(out, Err("abort"));
        assert_eq!(calls, 1);
    }

    #[test]
    fn faulted_request_reruns_once_from_the_replanned_start() {
        let mut starts = Vec::new();
        let mut faulted = true;
        let out: Result<usize, ()> = rerun_after_fault(
            "test",
            &mut starts,
            8,
            0,
            |starts, start| {
                starts.push(start);
                if std::mem::take(&mut faulted) {
                    Err(())
                } else {
                    Ok(start)
                }
            },
            |_| true,
            |_| 3,
        );
        assert_eq!(out, Ok(3));
        assert_eq!(starts, vec![0, 3]);
    }

    #[test]
    fn latch_trips_once_and_request_flag_clears() {
        let mut latch = FaultLatch::default();
        assert!(!latch.tripped());
        assert!(latch.trip(FaultClass::NpuRuntime, "chain"));
        assert!(!latch.trip(FaultClass::NpuRuntime, "again"));
        assert_eq!(latch.reason(), Some("chain"));
        assert!(latch.take_request_fault());
        assert!(!latch.take_request_fault());
        latch.begin_request();
        assert!(latch.tripped());

        // A GPU fault latches the route without marking the request.
        let mut gpu = FaultLatch::default();
        assert!(gpu.trip(FaultClass::GpuHip, "hip"));
        assert!(!gpu.take_request_fault());
    }
}
