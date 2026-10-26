// SPDX-License-Identifier: Apache-2.0
//! NPU trunk fault classification: an NPU runtime fault reruns its request GPU-only, a GPU error fails it.

use hipfire_runtime::coproc::containment::FaultClass;

/// The class of a pair executor error, by variant (never by message text). Every `HipError` is a GPU error, including a
/// `rdna_compute::router_fault::RouterFault` (non-finite MoE router row), which `Gpu::take_device_wait_error` returns as
/// `HipError` code 0 and which therefore fails the request with its original error and is never rerun.
pub(crate) fn fault_class_of(error: &crate::gpu_forward::Qwen4GpuForwardError) -> FaultClass {
    if matches!(error, crate::gpu_forward::Qwen4GpuForwardError::Hip(_)) {
        FaultClass::GpuHip
    } else {
        FaultClass::NpuRuntime
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_forward::{qwen4_prefill_segment_rows, SessionBoundaries};
    use hipfire_runtime::coproc::containment::{rerun_after_fault, FaultLatch};
    use std::collections::BTreeMap;

    const SCHEDULE_CHUNK: usize = 8192;
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    const NPU_MIX: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const PAIR: usize = 16;
    const CHUNK: usize = 8;
    const FAULT: &str = "NPU chain watchdog timed out";

    /// A toy request owner: a running state, a committed session cache of snapshots at every 8th row, and the route.
    struct Toy {
        state: u64,
        snapshots: BTreeMap<usize, u64>,
        staged: BTreeMap<usize, u64>,
        /// The live continuation of the previous turn: (position, state). `prefill` consumes it.
        live: Option<(usize, u64)>,
        latch: FaultLatch,
        npu_armed: bool,
        /// The injected diagnostic GPU-only route (production: the process-wide getter, read once per request).
        gpu_route: bool,
        /// The route captured at this request's begin (production `request_gpu`).
        request_gpu: bool,
        fault_at_pair: Option<usize>,
        fault_class: FaultClass,
        npu_pairs_attempted: usize,
        attempts: usize,
    }

    impl Toy {
        fn new(npu_armed: bool, fault_at_pair: Option<usize>) -> Self {
            Self {
                state: 0,
                snapshots: BTreeMap::new(),
                staged: BTreeMap::new(),
                live: None,
                latch: FaultLatch::default(),
                npu_armed,
                gpu_route: false,
                request_gpu: false,
                fault_at_pair,
                fault_class: FaultClass::NpuRuntime,
                npu_pairs_attempted: 0,
                attempts: 0,
            }
        }
        /// A new request starts: forget the previous request's fault mark and capture the route once.
        fn begin_request(&mut self) {
            self.latch.begin_request();
            self.request_gpu = self.gpu_route;
        }
        /// GPU-only scheduling: the process latch is set or this request is diagnostic GPU-only (no latch mutation).
        fn route_off(&self) -> bool {
            self.latch.tripped() || self.request_gpu
        }
        /// Whether this request plans NPU pairs.
        fn planned(&self) -> bool {
            self.npu_armed && !self.route_off()
        }
        fn gpu_row(state: u64, tok: usize) -> u64 {
            state.rotate_left(5) ^ (tok as u64).wrapping_mul(GOLDEN)
        }
        fn npu_pair(state: u64, toks: &[usize]) -> u64 {
            let sum = toks.iter().fold(0u64, |a, &t| a.wrapping_add(t as u64));
            state.rotate_left(11) ^ sum.wrapping_mul(NPU_MIX)
        }
        /// The prefill entry: take the live continuation at `start`, else restore `snapshots[start]` (or 0), run NPU pairs while armed and unlatched, then 8-row GPU
        /// chunks; snapshots are staged at each boundary and committed only on success.
        fn prefill(&mut self, prompt: &[usize], start: usize) -> Result<u64, String> {
            self.begin_request();
            self.attempts += 1;
            self.staged.clear();
            let live = self.live.take().filter(|&(position, _)| position == start);
            self.state = match live {
                Some((_, state)) => state,
                None if start == 0 => 0,
                None => *self
                    .snapshots
                    .get(&start)
                    .ok_or("no snapshot at the start")?,
            };
            let mut pos = start;
            while pos < prompt.len() {
                let remaining = prompt.len() - pos;
                if self.planned() && remaining >= PAIR {
                    let pair = self.npu_pairs_attempted;
                    self.npu_pairs_attempted += 1;
                    if self.fault_at_pair == Some(pair) {
                        self.state = 0xDEAD_BEEF_DEAD_BEEF;
                        self.latch.trip(self.fault_class, FAULT);
                        return Err(FAULT.to_string());
                    }
                    self.state = Self::npu_pair(self.state, &prompt[pos..pos + PAIR]);
                    pos += PAIR;
                } else {
                    let end = (pos + CHUNK).min(prompt.len());
                    for &tok in &prompt[pos..end] {
                        self.state = Self::gpu_row(self.state, tok);
                    }
                    pos = end;
                }
                if pos.is_multiple_of(CHUNK) && pos < prompt.len() {
                    self.staged.insert(pos, self.state);
                }
            }
            self.snapshots.append(&mut self.staged);
            Ok(self.state)
        }
        /// The production shape: the entry wrapped in the rerun helper.
        fn request(&mut self, prompt: &[usize], start: usize) -> Result<u64, String> {
            let rows = prompt.len();
            rerun_after_fault(
                crate::npu_trunk::SWITCH.tag,
                self,
                rows,
                start,
                |toy, start| toy.prefill(prompt, start),
                |toy| toy.latch.take_request_fault(),
                |toy| {
                    toy.snapshots
                        .range(..=start)
                        .next_back()
                        .map_or(0, |(&pos, _)| pos)
                },
            )
        }
    }

    fn prompt(rows: usize) -> Vec<usize> {
        (0..rows).map(|i| i * 7 + 3).collect()
    }

    #[test]
    fn faulted_request_matches_gpu_only() {
        let request = prompt(40);
        let mut reference = Toy::new(false, None);
        let mut faulted = Toy::new(true, Some(0));
        let expected = reference.request(&request, 0).expect("gpu-only request");
        let actual = faulted
            .request(&request, 0)
            .expect("faulted request reruns");
        assert_eq!(actual, expected);
        assert!(faulted.latch.tripped());
        assert_eq!(faulted.latch.reason(), Some(FAULT));
        assert_eq!(faulted.npu_pairs_attempted, 1);
        assert_eq!(
            faulted.snapshots, reference.snapshots,
            "the rerun commits the GPU-only snapshots"
        );
        // The same request through an unfaulted NPU model differs: the toy numerics tell the routes apart.
        let mut healthy = Toy::new(true, None);
        assert_ne!(healthy.request(&request, 0).unwrap(), expected);
    }

    #[test]
    fn diagnostic_gpu_route_matches_off() {
        let request = prompt(40);
        let mut off = Toy::new(false, None);
        let mut routed = Toy::new(true, None);
        routed.gpu_route = true;
        let expected = off.request(&request, 0).expect("off request");
        let actual = routed.request(&request, 0).expect("routed request");
        assert_eq!(actual, expected);
        assert_eq!(routed.snapshots, off.snapshots);
        assert!(routed.request_gpu);
        assert_eq!(
            routed.npu_pairs_attempted, 0,
            "a GPU-routed request plans no pair"
        );
        assert_eq!(routed.attempts, 1);
        assert!(!routed.latch.tripped() && routed.latch.reason().is_none());
        assert!(!routed.latch.take_request_fault());
        // A continuation from a committed snapshot is identical too.
        let longer = prompt(64);
        assert_eq!(
            routed.request(&longer, 32).unwrap(),
            off.request(&longer, 32).unwrap()
        );
        assert_eq!(routed.snapshots, off.snapshots);
        assert_eq!(routed.npu_pairs_attempted, 0);
        // Control: the same request through an auto route differs, so the equality above is not vacuous.
        let mut auto = Toy::new(true, None);
        assert_ne!(auto.request(&request, 0).unwrap(), expected);
        assert!(!auto.request_gpu);
    }

    #[test]
    fn route_switches_gpu_auto_gpu_per_request() {
        let mut off = Toy::new(false, None);
        let mut toy = Toy::new(true, None);
        // (diagnostic route, rows, NPU pairs the request plans)
        for (gpu, rows, pairs) in [(true, 40, 0), (false, 56, 3), (true, 40, 0), (false, 48, 3)] {
            let request = prompt(rows);
            toy.gpu_route = gpu;
            let before = toy.npu_pairs_attempted;
            let actual = toy.request(&request, 0).unwrap();
            let expected = off.request(&request, 0).unwrap();
            assert_eq!(
                toy.request_gpu, gpu,
                "the route is captured at the request's begin"
            );
            assert_eq!(toy.npu_pairs_attempted - before, pairs);
            if gpu {
                assert_eq!(actual, expected);
                assert!(
                    toy.snapshots.range(..rows).eq(off.snapshots.range(..rows)),
                    "a GPU request commits the off snapshots"
                );
            } else {
                assert_ne!(actual, expected, "auto pairs again after a GPU request");
            }
            assert!(!toy.latch.tripped() && toy.latch.reason().is_none());
            assert!(!toy.latch.take_request_fault());
        }
    }

    #[test]
    fn diagnostic_route_does_not_consume_a_pending_fault() {
        let request = prompt(40);
        let mut off = Toy::new(false, None);
        let expected = off.request(&request, 0).unwrap();
        let mut toy = Toy::new(true, Some(0));
        toy.gpu_route = true;
        assert_eq!(toy.request(&request, 0).unwrap(), expected);
        assert_eq!(
            toy.npu_pairs_attempted, 0,
            "the GPU route never reaches the first pair"
        );
        assert!(!toy.latch.tripped() && !toy.latch.take_request_fault());
        // Back on auto the first pair faults exactly as if the GPU request had not happened.
        toy.gpu_route = false;
        assert_eq!(
            toy.request(&request, 0).unwrap(),
            expected,
            "the faulted rerun is GPU-only"
        );
        assert_eq!(toy.npu_pairs_attempted, 1);
        assert_eq!(toy.latch.reason(), Some(FAULT));
        // A later GPU request leaves the tripped latch and its reason as they were.
        toy.gpu_route = true;
        assert_eq!(toy.request(&request, 0).unwrap(), expected);
        assert_eq!(toy.npu_pairs_attempted, 1);
        assert_eq!(toy.latch.reason(), Some(FAULT));
    }

    #[test]
    fn diagnostic_route_leaves_a_tripped_latch_untouched() {
        let request = prompt(40);
        let mut off = Toy::new(false, None);
        let expected = off.request(&request, 0).unwrap();
        for class in [FaultClass::NpuRuntime, FaultClass::GpuHip] {
            let mut toy = Toy::new(true, Some(0));
            toy.fault_class = class;
            let first = toy.request(&request, 0);
            assert_eq!(first.is_ok(), class == FaultClass::NpuRuntime);
            assert!(toy.latch.tripped());
            assert_eq!(toy.npu_pairs_attempted, 1);
            toy.gpu_route = true;
            for _ in 0..2 {
                let attempts = toy.attempts;
                assert_eq!(toy.request(&request, 0).unwrap(), expected);
                assert_eq!(toy.attempts, attempts + 1, "no fault, no rerun");
                assert!(toy.latch.tripped());
                assert_eq!(toy.latch.reason(), Some(FAULT));
                assert!(
                    !toy.latch.take_request_fault(),
                    "the request fault mark is not set"
                );
                assert_eq!(toy.npu_pairs_attempted, 1);
            }
            // Switching back to auto does not clear the latch: still no pair.
            toy.gpu_route = false;
            assert_eq!(toy.request(&request, 0).unwrap(), expected);
            assert!(toy.latch.tripped());
            assert_eq!(toy.latch.reason(), Some(FAULT));
            assert_eq!(toy.npu_pairs_attempted, 1);
        }
    }

    /// The segment lengths of the real schedule helper from `start` to `end`, with `cuts` as the session's pending
    /// snapshot boundaries.
    fn segments(start: usize, end: usize, cuts: &[usize], planned: bool) -> Vec<usize> {
        let mut rows = Vec::new();
        let mut pos = start;
        while pos < end {
            let mut pending = cuts.iter().copied().filter(|&cut| cut > pos);
            let boundaries = SessionBoundaries {
                next: pending.next(),
                after: pending.next(),
            };
            let segment = qwen4_prefill_segment_rows(pos, end, SCHEDULE_CHUNK, boundaries, planned);
            assert!(segment > 0 && pos + segment <= end);
            rows.push(segment);
            pos += segment;
        }
        rows
    }

    /// The `off` schedule written without the helper: sequential chunks, each cut at the next session boundary.
    fn off_segments(start: usize, end: usize, cuts: &[usize]) -> Vec<usize> {
        let mut rows = Vec::new();
        let mut pos = start;
        while pos < end {
            let mut segment = SCHEDULE_CHUNK.min(end - pos);
            if let Some(cut) = cuts
                .iter()
                .copied()
                .find(|&cut| cut > pos && cut < pos + segment)
            {
                segment = cut - pos;
            }
            rows.push(segment);
            pos += segment;
        }
        rows
    }

    #[test]
    fn diagnostic_route_schedule_is_the_off_schedule() {
        let mut off = Toy::new(false, None);
        let mut diagnostic = Toy::new(true, None);
        diagnostic.gpu_route = true;
        let mut auto = Toy::new(true, None);
        let mut tripped = Toy::new(true, None);
        tripped.latch.trip(FaultClass::NpuRuntime, FAULT);
        for toy in [&mut off, &mut diagnostic, &mut auto, &mut tripped] {
            toy.begin_request();
        }
        assert!(!off.planned() && !diagnostic.planned() && !tripped.planned());
        assert!(auto.planned());
        let cases: [(usize, &[usize]); 3] = [
            (0, &[]),
            (0, &[5000, 9000, 12000, 20000, 20001, 33000, 40000]),
            (3000, &[8192, 16384, 24576]),
        ];
        for (start, cuts) in cases {
            let expected = off_segments(start, 40000, cuts);
            assert!(expected.iter().all(|&rows| rows <= SCHEDULE_CHUNK));
            assert_eq!(segments(start, 40000, cuts, off.planned()), expected);
            assert_eq!(segments(start, 40000, cuts, diagnostic.planned()), expected);
            assert_eq!(segments(start, 40000, cuts, tripped.planned()), expected);
            let paired = segments(start, 40000, cuts, auto.planned());
            assert_ne!(
                paired, expected,
                "auto groups rows into pairs, so the schedules differ"
            );
            assert!(paired.iter().any(|&rows| rows > SCHEDULE_CHUNK));
            assert_eq!(paired.iter().sum::<usize>(), expected.iter().sum::<usize>());
        }
    }

    #[test]
    fn latch_persists_across_requests() {
        let mut reference = Toy::new(false, None);
        let mut faulted = Toy::new(true, Some(0));
        let first = prompt(40);
        assert_eq!(
            faulted.request(&first, 0).unwrap(),
            reference.request(&first, 0).unwrap()
        );
        let attempted = faulted.npu_pairs_attempted;
        for rows in [48, 64] {
            let request = prompt(rows);
            assert_eq!(
                faulted.request(&request, 0).unwrap(),
                reference.request(&request, 0).unwrap()
            );
            assert_eq!(
                faulted.npu_pairs_attempted, attempted,
                "a latched route plans no NPU pair"
            );
            assert!(faulted.latch.tripped());
            assert_eq!(faulted.latch.reason(), Some(FAULT));
        }
    }

    #[test]
    fn faulted_live_continuation_replans_to_a_snapshot() {
        let request = prompt(64);
        let mut reference = Toy::new(false, None);
        let live_state = reference.request(&request[..24], 0).unwrap();
        assert_eq!(
            reference.snapshots.keys().copied().collect::<Vec<_>>(),
            [8, 16]
        );
        // The faulted model holds the same committed snapshots and continues live at row 24, which is not a snapshot.
        let mut faulted = Toy::new(true, Some(0));
        faulted.snapshots = reference.snapshots.clone();
        faulted.live = Some((24, live_state));
        let actual = faulted
            .request(&request, 24)
            .expect("faulted continuation reruns");
        assert!(faulted.latch.tripped());
        assert_eq!(faulted.npu_pairs_attempted, 1);
        // The live state was consumed: the rerun started at the largest committed snapshot <= 24 (row 16).
        assert!(faulted.live.is_none());
        let expected = reference.request(&request, 16).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(faulted.snapshots, reference.snapshots);
    }

    #[test]
    fn gpu_hip_error_fails_the_request_without_a_rerun() {
        let mut faulted = Toy::new(true, Some(0));
        faulted.fault_class = FaultClass::GpuHip;
        let error = faulted.request(&prompt(40), 0).unwrap_err();
        assert_eq!(error, FAULT, "the original error is returned unchanged");
        assert_eq!(faulted.attempts, 1, "a sticky GPU context is never rerun");
        assert!(faulted.latch.tripped(), "the route still latches off");
        assert!(
            faulted.snapshots.is_empty(),
            "nothing of the failed attempt is committed"
        );
        // Later requests run GPU-only (no NPU pair) and match a GPU-only model.
        let mut reference = Toy::new(false, None);
        let request = prompt(48);
        assert_eq!(
            faulted.request(&request, 0).unwrap(),
            reference.request(&request, 0).unwrap()
        );
        assert_eq!(faulted.npu_pairs_attempted, 1);
    }

    #[test]
    fn router_fault_is_a_gpu_request_failure_never_rerun() {
        use crate::gpu_forward::Qwen4GpuForwardError;
        use rdna_compute::router_fault::RouterFault;
        // The production path: the fault word's RouterFault -> `to_hip_error` (code 0, as `take_device_wait_error` returns it)
        // -> `fault_class_of` (what `NpuForward::contain_fault` classifies with).
        let fault = RouterFault::decode(3, RouterFault::encode(17, 1234)).expect("decodes");
        let error = Qwen4GpuForwardError::from(fault.to_hip_error());
        assert!(matches!(&error, Qwen4GpuForwardError::Hip(_)));
        assert_eq!(fault_class_of(&error), FaultClass::GpuHip);
        // The layer/row message survives the conversion.
        let message = error.to_string();
        assert!(
            message.contains("GPU 3")
                && message.contains("layer 17")
                && message.contains("row 1234"),
            "{message}"
        );
        // A non-Hip error (the NPU runtime's relay/validation failures) stays an NPU runtime fault.
        assert_eq!(
            fault_class_of(&Qwen4GpuForwardError::Dispatch("npu relay".to_string())),
            FaultClass::NpuRuntime
        );
        // Latch semantics for that class: the route latches, but the request is never marked for a GPU-only rerun, so
        // `rerun_after_fault` runs it once and returns the router error unchanged.
        let mut latch = FaultLatch::default();
        assert!(latch.trip(fault_class_of(&error), &error.to_string()));
        assert!(latch.tripped() && !latch.take_request_fault());
        let mut attempts = 0;
        let result: Result<(), Qwen4GpuForwardError> = rerun_after_fault(
            crate::npu_trunk::SWITCH.tag,
            &mut latch,
            10,
            0,
            |_, _| {
                attempts += 1;
                Err(Qwen4GpuForwardError::from(fault.to_hip_error()))
            },
            |latch| latch.take_request_fault(),
            |_| panic!("a router fault is never replanned"),
        );
        assert_eq!(attempts, 1);
        let message = result.unwrap_err().to_string();
        assert!(
            message.contains("layer 17") && message.contains("row 1234"),
            "{message}"
        );
    }
}
