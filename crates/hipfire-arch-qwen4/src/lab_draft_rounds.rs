// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab oracle of the batched MTP head round (`HIPFIRE_QWEN4_MTP_DRAFT_ROUNDS=1`):
//! per-round bytes of one batched round against `n` serial one-token steps,
//! both started from the same head state of every drafting lane.
//!
//! Armed by `HIPFIRE_LAB_MTP_ROUNDS_PROBE_DIR=<dir>` (and the rounds flag). Each
//! mesh draft round of two or more lanes calls [`probe_round`] before its own
//! work: the lanes' head states are copied into scratch twins
//! ([`MtpGpuState::lab_twin`]), the serial arm runs `n` `mtp_forward_token_deferred`
//! steps (one twin each), the batched arm runs the round's batched head calls
//! (`mtp_lanes::forward_draft_rows`: `n` twins in calls of at most the load's
//! `Qwen4Bundle::mtp_draft_call_rows` lanes, 8, or 32 on exact single-device gfx942), and
//! every lane's draft token, margin bits, 72 draft-top1
//! bytes, wide hidden, head KV state digest and draft request state are compared
//! arm against arm. The lane's own state, request and every window structure are
//! left untouched. One JSON line per lane per round goes to
//! `<dir>/rank<r>.jsonl`, plus a kernel-symbol census line per round.

use crate::bundle::Qwen4Bundle;
use crate::config::Qwen4Config;
use crate::lane::{LaneSpec, Qwen4LaneState};
use crate::mtp_gpu::{lab_head_items, DraftRow, MtpGpuState};
use crate::mtp_lanes::lane_call;
use hipfire_dispatch::pipeline::DraftHeadRequestState;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

/// Directory the per-rank probe files are written to.
pub(crate) const PROBE_DIR_ENV: &str = "HIPFIRE_LAB_MTP_ROUNDS_PROBE_DIR";

/// Whether the probe is armed: its directory is set and the bundle's batched rounds are on
/// (the rows scratch is allocated).
pub(crate) fn probe_enabled(bundle: &crate::bundle::Qwen4Bundle) -> bool {
    bundle.mtp_rows_ready()
        && hipfire_config::developer_var(PROBE_DIR_ENV).is_ok_and(|dir| !dir.is_empty())
}

/// What one arm produced for one lane.
struct ArmResult {
    token: u32,
    margin_bits: u32,
    top1: Vec<u8>,
    request_full_steps: u32,
    /// `(family, fnv1a64 digest, byte length)` of every head state family.
    items: Vec<(String, u64, usize)>,
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

type SymbolTally = Arc<Mutex<BTreeMap<String, u64>>>;
type LaunchObserver = dyn for<'a> Fn(&Gpu, &rdna_compute::LabLaunchWitness<'a>) -> hip_bridge::HipResult<()>
    + Send
    + Sync;

/// Count every kernel launch `f` makes into `tally`.
fn tallied<R>(gpu: &mut Gpu, tally: &SymbolTally, f: impl FnOnce(&mut Gpu) -> R) -> R {
    let sink = Arc::clone(tally);
    let observer: Arc<LaunchObserver> = Arc::new(move |_gpu, witness| {
        *sink
            .lock()
            .expect("launch tally")
            .entry(witness.symbol.to_string())
            .or_default() += 1;
        Ok(())
    });
    let previous = gpu.set_lab_launch_observer(Some(observer));
    let result = f(gpu);
    gpu.set_lab_launch_observer(previous);
    result
}

fn download_top1(gpu: &mut Gpu, top1: &GpuTensor, bytes: usize) -> Result<Vec<u8>, String> {
    let mut host = vec![0u8; bytes];
    gpu.hip
        .memcpy_dtoh(&mut host, &top1.buf)
        .map_err(|error| format!("probe draft download: {error}"))?;
    gpu.take_device_wait_error()
        .map_err(|error| format!("probe draft download: {error}"))?;
    Ok(host)
}

fn digest_items(gpu: &Gpu, state: &MtpGpuState) -> Result<Vec<(String, u64, usize)>, String> {
    Ok(lab_head_items(gpu, state)?
        .into_iter()
        .map(|(name, bytes)| (name, fnv1a(&bytes), bytes.len()))
        .collect())
}

fn symbols_json(map: &BTreeMap<String, u64>) -> Value {
    Value::Object(map.iter().map(|(k, v)| (k.clone(), json!(v))).collect())
}

/// One lane's inputs of this round, read off the lane before any arm runs.
struct LaneInput {
    token: u32,
    position: usize,
    j: usize,
}

/// One arm's results over the round's lanes and its wall time (µs) from the
/// first enqueue to the last parsed draft: the work a draft round does.
struct ArmRun {
    lanes: Vec<ArmResult>,
    micros: f64,
}

/// Scratch twins of every lane's head state, for one arm.
struct Twins {
    states: Vec<MtpGpuState>,
    requests: Vec<DraftHeadRequestState>,
}

impl Twins {
    fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        lanes: &[&mut Qwen4LaneState],
    ) -> Result<Self, String> {
        let mut twins = Self {
            states: Vec::with_capacity(lanes.len()),
            requests: Vec::with_capacity(lanes.len()),
        };
        for lane in lanes {
            let spec = lane
                .spec
                .as_ref()
                .ok_or_else(|| "probe: drafting lane lost its MTP state".to_string())?;
            match spec.head.lab_twin(gpu, config) {
                Ok(state) => {
                    twins.states.push(state);
                    twins.requests.push(spec.draft);
                }
                Err(error) => {
                    twins.free(gpu);
                    return Err(error.to_string());
                }
            }
        }
        Ok(twins)
    }

    fn free(self, gpu: &mut Gpu) {
        for state in self.states {
            let _ = state.free_gpu(gpu);
        }
    }
}

/// The serial arm: `n` `mtp_forward_token_deferred` steps (one twin each, the
/// lane's own `with_lane` swap), one download of the `n` slots, each lane's
/// own parse: exactly `draft_round_device`'s serial work.
fn serial_arm(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lanes: &mut [&mut Qwen4LaneState],
    inputs: &[LaneInput],
    config: &Qwen4Config,
    top1: &GpuTensor,
    tally: Option<&SymbolTally>,
    digest: bool,
) -> Result<ArmRun, String> {
    let slot = Gpu::TOPK8_OUT_BYTES;
    let n = inputs.len();
    let mut twins = Twins::new(gpu, config, lanes)?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("probe sync: {e}"))?;
    let started = std::time::Instant::now();
    let result = (|| -> Result<Vec<(u32, u32, Vec<u8>, u32)>, String> {
        for (index, input) in inputs.iter().enumerate() {
            let out = top1.sub_offset(index * slot, slot);
            let Qwen4LaneState { state, spec, .. } = &mut *lanes[index];
            let LaneSpec { pending_hidden, .. } = spec
                .as_mut()
                .ok_or_else(|| "probe: drafting lane lost its MTP state".to_string())?;
            let hidden = (input.j == 0).then_some(&*pending_hidden);
            let (twin, request) = (&mut twins.states[index], &mut twins.requests[index]);
            let mut step = |gpu: &mut Gpu| {
                lane_call(bundle.with_lane(state, Some((twin, request)), |b| {
                    b.mtp_forward_token_deferred(
                        gpu,
                        input.token,
                        hidden,
                        input.position,
                        input.j == 0,
                        &out,
                    )
                }))
            };
            match tally {
                Some(tally) => tallied(gpu, tally, step),
                None => step(gpu),
            }?;
        }
        let tops = download_top1(gpu, top1, n * slot)?;
        let mut parsed = Vec::with_capacity(n);
        for index in 0..n {
            let top = &tops[index * slot..(index + 1) * slot];
            let Qwen4LaneState { state, .. } = &mut *lanes[index];
            let (twin, request) = (&mut twins.states[index], &mut twins.requests[index]);
            let (token, margin) = lane_call(bundle.with_lane(state, Some((twin, request)), |b| {
                let token = b.mtp_draft_finish(top)?;
                Ok::<_, crate::bundle::BundleError>((token, b.mtp_draft_margin()))
            }))?;
            parsed.push((
                token,
                margin.to_bits(),
                top.to_vec(),
                twins.requests[index].full_steps,
            ));
        }
        Ok(parsed)
    })();
    let micros = started.elapsed().as_secs_f64() * 1e6;
    let lanes_out = result.and_then(|parsed| {
        parsed
            .into_iter()
            .zip(&twins.states)
            .map(|((token, margin_bits, top1, request_full_steps), twin)| {
                Ok(ArmResult {
                    token,
                    margin_bits,
                    top1,
                    request_full_steps,
                    items: if digest {
                        digest_items(gpu, twin)?
                    } else {
                        Vec::new()
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()
    });
    twins.free(gpu);
    Ok(ArmRun {
        lanes: lanes_out.map_err(|e| format!("probe serial arm: {e}"))?,
        micros,
    })
}

/// The batched arm: the round's batched head calls over `n` twins
/// ([`crate::mtp_lanes::forward_draft_rows`]), one download, each lane's parse.
fn batched_arm(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lanes: &mut [&mut Qwen4LaneState],
    inputs: &[LaneInput],
    config: &Qwen4Config,
    top1: &GpuTensor,
    tally: Option<&SymbolTally>,
    digest: bool,
) -> Result<ArmRun, String> {
    let slot = Gpu::TOPK8_OUT_BYTES;
    let n = inputs.len();
    let mut twins = Twins::new(gpu, config, lanes)?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("probe sync: {e}"))?;
    let started = std::time::Instant::now();
    let result = (|| -> Result<Vec<(u32, u32, Vec<u8>, u32)>, String> {
        {
            let mut rows: Vec<DraftRow<'_>> = twins
                .states
                .iter_mut()
                .zip(twins.requests.iter_mut())
                .zip(lanes.iter())
                .zip(inputs)
                .map(|(((state, request), lane), input)| DraftRow {
                    state,
                    request,
                    token: input.token,
                    backbone_hidden: (input.j == 0)
                        .then(|| &lane.spec.as_ref().expect("checked above").pending_hidden),
                    position: input.position,
                    fresh_qsa_selection: input.j == 0,
                })
                .collect();
            let mut run = |gpu: &mut Gpu| {
                crate::mtp_lanes::forward_draft_rows(bundle, gpu, &mut rows, top1)
                    .map_err(|e| e.to_string())
            };
            match tally {
                Some(tally) => tallied(gpu, tally, run),
                None => run(gpu),
            }?;
        }
        let tops = download_top1(gpu, top1, n * slot)?;
        let mut parsed = Vec::with_capacity(n);
        for index in 0..n {
            let top = &tops[index * slot..(index + 1) * slot];
            let request = &mut twins.requests[index];
            let token = bundle
                .mtp_draft_finish_request(request, top)
                .map_err(|e| e.to_string())?;
            parsed.push((
                token,
                request.margin.to_bits(),
                top.to_vec(),
                request.full_steps,
            ));
        }
        Ok(parsed)
    })();
    let micros = started.elapsed().as_secs_f64() * 1e6;
    let lanes_out = result.and_then(|parsed| {
        parsed
            .into_iter()
            .zip(&twins.states)
            .map(|((token, margin_bits, top1, request_full_steps), twin)| {
                Ok(ArmResult {
                    token,
                    margin_bits,
                    top1,
                    request_full_steps,
                    items: if digest {
                        digest_items(gpu, twin)?
                    } else {
                        Vec::new()
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()
    });
    twins.free(gpu);
    Ok(ArmRun {
        lanes: lanes_out.map_err(|e| format!("probe batched arm: {e}"))?,
        micros,
    })
}

/// Timing repetitions per round (`HIPFIRE_LAB_MTP_ROUNDS_PROBE_REPS`, default 0:
/// byte comparison only). Each repetition runs both arms on fresh twins,
/// alternating which goes first.
const PROBE_REPS_ENV: &str = "HIPFIRE_LAB_MTP_ROUNDS_PROBE_REPS";

/// Rounds probed per process (`HIPFIRE_LAB_MTP_ROUNDS_PROBE_MAX`, default all);
/// the counter is shared by the in-process EP ranks, which probe each round
/// together, so [`PROBE_RANKS`] calls make one round.
const PROBE_MAX_ENV: &str = "HIPFIRE_LAB_MTP_ROUNDS_PROBE_MAX";
const PROBE_RANKS: usize = 2;

/// Run the probe for one mesh draft round: `ops[i]` is `(input token, position)`
/// of `lanes[i]`. Writes the comparison records; returns `Err` only when an arm
/// itself failed to run (a comparison mismatch is a record, not an error).
pub(crate) fn probe_round(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lanes: &mut [&mut Qwen4LaneState],
    ops: &[(u32, usize)],
    rank: usize,
    round: u32,
) -> Result<(), String> {
    let n = ops.len();
    if n < 2 || n != lanes.len() {
        return Ok(());
    }
    // The probe's own row gathers (serial and batched arms on twins) are not the
    // round's: the mesh ledger and the protocol's per-round gather count must not
    // see them.
    struct TallyGuard(crate::ep::GatherTally);
    impl Drop for TallyGuard {
        fn drop(&mut self) {
            crate::ep::lab_restore_gather_tally(self.0);
        }
    }
    let _tally_guard = TallyGuard(crate::ep::gather_tally());
    // Every rank counts the same rounds, so a cap stops them together.
    static PROBED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let cap: usize = hipfire_config::developer_var(PROBE_MAX_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    if PROBED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) / PROBE_RANKS >= cap {
        return Ok(());
    }
    let dir = hipfire_config::developer_var(PROBE_DIR_ENV).map_err(|e| e.to_string())?;
    let reps: usize = hipfire_config::developer_var(PROBE_REPS_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let config = bundle.config.clone();
    let mut inputs = Vec::with_capacity(n);
    for (lane, &(token, position)) in lanes.iter().zip(ops) {
        let spec = lane
            .spec
            .as_ref()
            .ok_or_else(|| "probe: drafting lane lost its MTP state".to_string())?;
        let window = spec
            .window
            .as_ref()
            .ok_or_else(|| "probe: draft op without a pending window".to_string())?;
        inputs.push(LaneInput {
            token,
            position,
            j: position - window.position,
        });
    }
    let slot = Gpu::TOPK8_OUT_BYTES;
    let top1 = gpu
        .zeros(&[n * slot], DType::Raw)
        .map_err(|e| format!("probe top1: {e}"))?;
    let serial_tally: SymbolTally = Arc::default();
    let batched_tally: SymbolTally = Arc::default();
    let outcome = (|| -> Result<(ArmRun, ArmRun, Vec<f64>, Vec<f64>), String> {
        let serial = serial_arm(
            bundle,
            gpu,
            lanes,
            &inputs,
            &config,
            &top1,
            Some(&serial_tally),
            true,
        )?;
        let batched = batched_arm(
            bundle,
            gpu,
            lanes,
            &inputs,
            &config,
            &top1,
            Some(&batched_tally),
            true,
        )?;
        let (mut serial_us, mut batched_us) = (Vec::new(), Vec::new());
        for rep in 0..reps {
            let order = if rep % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for batched_first in order {
                if batched_first {
                    batched_us.push(
                        batched_arm(bundle, gpu, lanes, &inputs, &config, &top1, None, false)?
                            .micros,
                    );
                } else {
                    serial_us.push(
                        serial_arm(bundle, gpu, lanes, &inputs, &config, &top1, None, false)?
                            .micros,
                    );
                }
            }
        }
        Ok((serial, batched, serial_us, batched_us))
    })();
    let _ = gpu.free_tensor(top1);
    let (serial, batched, serial_us, batched_us) = outcome?;

    // Records.
    let mut lines: Vec<String> = Vec::with_capacity(n + 1);
    for (index, (s, b)) in serial.lanes.iter().zip(&batched.lanes).enumerate() {
        let mut bad_families: Vec<String> = Vec::new();
        for ((name, digest, len), (_, other, other_len)) in s.items.iter().zip(&b.items) {
            if digest != other || len != other_len {
                bad_families.push(name.clone());
            }
        }
        let kv_bad = bad_families.iter().any(|f| f != "head.wide_hidden");
        let wide_bad = bad_families.iter().any(|f| f == "head.wide_hidden");
        let record = json!({
            "kind": "lane", "rank": rank, "round": round, "n": n, "slot": index,
            "j": inputs[index].j, "position": inputs[index].position,
            "serial": arm_json(s), "batched": arm_json(b),
            "mismatch": {
                "token": s.token != b.token,
                "margin": s.margin_bits != b.margin_bits,
                "top1": s.top1 != b.top1,
                "wide_hidden": wide_bad,
                "kv": kv_bad,
                "request": s.request_full_steps != b.request_full_steps,
            },
            "bad_families": bad_families,
        });
        lines.push(record.to_string());
    }
    lines.push(
        json!({
            "kind": "census", "rank": rank, "round": round, "n": n,
            "j": inputs.iter().map(|i| i.j).collect::<Vec<_>>(),
            "serial": symbols_json(&serial_tally.lock().expect("tally")),
            "batched": symbols_json(&batched_tally.lock().expect("tally")),
            "serial_us_observed": serial.micros, "batched_us_observed": batched.micros,
            "serial_us": serial_us, "batched_us": batched_us,
        })
        .to_string(),
    );
    let path = std::path::Path::new(&dir).join(format!("rank{rank}.jsonl"));
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| format!("probe output {}: {e}", path.display()))?;
    for line in lines {
        writeln!(file, "{line}").map_err(|e| format!("probe output: {e}"))?;
    }
    Ok(())
}

fn arm_json(arm: &ArmResult) -> Value {
    json!({
        "token": arm.token,
        "margin_bits": arm.margin_bits,
        "top1": hex(&arm.top1),
        "request_full_steps": arm.request_full_steps,
        "items": arm.items.iter().map(|(name, digest, len)| json!({
            "family": name, "fnv1a64": format!("{digest:016x}"), "bytes": len,
        })).collect::<Vec<_>>(),
    })
}

/// `HIPFIRE_LAB_MTP_STALE_FILL=nan|big`: before each draft round overwrite every
/// drafting lane's head arena rows beyond the valid lengths with an F32 NaN
/// (`nan`) or a huge finite sentinel (`big`), the PARITY.md 4.3 settling test of
/// the draft-history mask. The run's ids must equal the unfilled run's.
const STALE_FILL_ENV: &str = "HIPFIRE_LAB_MTP_STALE_FILL";

/// The fill byte of [`STALE_FILL_ENV`], when armed.
pub(crate) fn stale_fill_byte() -> Option<u8> {
    match hipfire_config::developer_var(STALE_FILL_ENV).ok()?.as_str() {
        "nan" => Some(0xFF),
        "big" => Some(0x7F),
        _ => None,
    }
}

/// Apply [`stale_fill_byte`] to every lane of the round (a no-op unarmed).
pub(crate) fn fill_stale(gpu: &Gpu, lanes: &[&mut Qwen4LaneState]) -> Result<(), String> {
    let Some(byte) = stale_fill_byte() else {
        return Ok(());
    };
    for lane in lanes {
        let spec = lane
            .spec
            .as_ref()
            .ok_or_else(|| "stale fill: drafting lane lost its MTP state".to_string())?;
        spec.head
            .lab_fill_stale(gpu, byte)
            .map_err(|e| format!("stale fill: {e}"))?;
    }
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("stale fill sync: {e}"))
}
