//! No-model asynchronous expert-slot publication oracle on exact gfx1201.
//!
//! Uses the production mutable-residency authority and unchanged indexed
//! kernels with the original S0 fixtures. This checks copy/publication order,
//! not the full-model transport manager, routing, quality or performance.
//! Default: PLAN_ONLY. --execute requires the caller's shared GPU lock.

#[path = "../../rdna-compute/examples/support/fn_expert_fixture.rs"]
#[allow(dead_code)] // Shared S0 helpers also contain its optional hipGraph arm.
mod fixture;
use fixture::*;
use hipfire_dispatch::families::moe::{
    MutableExpertResidency, MutableExpertResidencySpec, ResidencyLocation, ResidencyPublication,
    ResidencySlot, ResidencySlotTicket, ResidencySnapshot, ResidencySpan,
};
use rdna_compute::replay::{ReplayBackendRequest, ReplayController, ReplayQuiescence};
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Value};
use std::cell::Cell;
use std::sync::{Arc, Mutex};

struct PinnedSource {
    pointer: usize,
    bytes: usize,
}
impl PinnedSource {
    fn slice(&self, start: usize, bytes: usize) -> R<&[u8]> {
        check(
            start
                .checked_add(bytes)
                .is_some_and(|end| end <= self.bytes),
            "pinned source extent",
        )?;
        // SAFETY: a live hipHostMalloc owner initialized these exact bytes.
        // Sources remain immutable and live until all copy events retire.
        Ok(unsafe { std::slice::from_raw_parts((self.pointer + start) as *const u8, bytes) })
    }
}

#[derive(Default)]
struct Dma {
    stream: Option<hip_bridge::Stream>,
    events: Vec<hip_bridge::Event>,
    sources: Vec<PinnedSource>,
}
impl Dma {
    fn event(&mut self, gpu: &Gpu) -> R<usize> {
        let event = hip(gpu
            .hip
            .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING))?;
        self.events.push(event);
        Ok(self.events.len() - 1)
    }
    fn source(&mut self, gpu: &Gpu, bytes: &[u8]) -> R<usize> {
        let pointer = hip(gpu.hip.host_malloc(bytes.len(), 0))? as usize;
        check(pointer != 0, "null pinned source")?;
        self.sources.push(PinnedSource {
            pointer,
            bytes: bytes.len(),
        });
        // SAFETY: hipHostMalloc returned this initialized writable extent;
        // no GPU access is enqueued until the one-time initialization ends.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer as *mut u8, bytes.len());
        }
        Ok(self.sources.len() - 1)
    }
    fn stream(&self) -> R<&hip_bridge::Stream> {
        self.stream
            .as_ref()
            .ok_or_else(|| "copy stream absent".into())
    }
    fn release(&mut self, gpu: &Gpu) -> R<()> {
        let mut errors = Vec::new();
        while let Some(event) = self.events.pop() {
            if let Err(error) = gpu.hip.event_destroy(event) {
                errors.push(error.to_string());
            }
        }
        if let Some(stream) = self.stream.take() {
            if let Err(error) = gpu.hip.stream_destroy(stream) {
                errors.push(error.to_string());
            }
        }
        while let Some(source) = self.sources.pop() {
            if let Err(error) = gpu.hip.host_free(source.pointer as *mut std::ffi::c_void) {
                errors.push(error.to_string());
            }
        }
        check(errors.is_empty(), format!("copy cleanup: {errors:?}"))
    }
}

fn words(entries: &[usize]) -> Vec<u8> {
    entries
        .iter()
        .flat_map(|p| (*p as u64).to_le_bytes())
        .collect()
}

fn tables_match(gpu: &Gpu, f: &Fixture, snapshot: &ResidencySnapshot) -> R<()> {
    check(
        hip(gpu.download_raw_bytes(&f.gu_table))? == words(&snapshot.gate_up),
        "GU table readback",
    )?;
    check(
        hip(gpu.download_raw_bytes(&f.dn_table))? == words(&snapshot.down),
        "DN table readback",
    )
}

fn publish(
    gpu: &Gpu,
    f: &Fixture,
    residency: &MutableExpertResidency,
    mapping: Vec<ResidencyLocation>,
    completed_slots: Vec<ResidencySlotTicket>,
) -> R<ResidencySnapshot> {
    let old = residency.snapshot()?;
    // SAFETY: every prior reader retired and every supplied slot ticket has a
    // completed pair of copies. Synchronous H2D writes finish before callback
    // return, including before an independent retained PM4 reader can submit.
    let next = unsafe {
        residency.publish_after_gpu_wait(
            ResidencyPublication {
                expected_epoch: old.epoch,
                mapping,
                completed_slots,
            },
            |next| {
                hip(gpu.hip.memcpy_htod(&f.gu_table.buf, &words(&next.gate_up)))?;
                hip(gpu.hip.memcpy_htod(&f.dn_table.buf, &words(&next.down)))?;
                Ok(())
            },
        )
    }?;
    tables_match(gpu, f, &next)?;
    Ok(next)
}

fn capability(
    owner: &Arc<Mutex<Arena>>,
    arena: &Arena,
    f: &Fixture,
    gu_table: &GpuTensor,
    dn_table: &GpuTensor,
) -> R<MutableExpertResidency> {
    let canonical = |root, stride| {
        let base = arena.get(root).buf.as_ptr() as usize;
        (0..E)
            .map(|e| ResidencySpan {
                address: base + (e % DISTINCT) * stride,
                bytes: stride,
            })
            .collect()
    };
    let span = |tensor: &GpuTensor| ResidencySpan {
        address: tensor.buf.as_ptr() as usize,
        bytes: tensor.buf.size(),
    };
    // SAFETY: the retained Arena owns every fixed canonical/table/slot root.
    // It is explicitly freed only after all capabilities and tapes are gone
    // and both HIP and PM4 quiescence have been proved. Canonical payloads are
    // never overwritten. Both tables already contain canonical addresses.
    unsafe {
        MutableExpertResidency::new(MutableExpertResidencySpec {
            owner_generation: 1,
            canonical_gate_up: canonical(f.gu_host, GU_STRIDE),
            canonical_down: canonical(f.dn_host, DN_STRIDE),
            gate_up_table: span(gu_table),
            down_table: span(dn_table),
            slots: vec![ResidencySlot {
                gate_up: ResidencySpan {
                    address: arena.get(f.gu_device).buf.as_ptr() as usize,
                    bytes: GU_STRIDE,
                },
                down: ResidencySpan {
                    address: arena.get(f.dn_device).buf.as_ptr() as usize,
                    bytes: DN_STRIDE,
                },
            }],
            keepalive: owner.clone(),
        })
    }
}

fn refused_without_upload(
    residency: &MutableExpertResidency,
    publication: ResidencyPublication,
    label: &str,
) -> R<()> {
    let called = Cell::new(false);
    // SAFETY: an intentionally invalid metadata transaction must be rejected
    // before the callback. No GPU pointers are accessed by this callback.
    let result = unsafe {
        residency.publish_after_gpu_wait(publication, |_| {
            called.set(true);
            Ok(())
        })
    };
    check(
        result.is_err() && !called.get(),
        format!("{label}: invalid publication reached upload"),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &mut Gpu,
    arena: &Arena,
    f: &Fixture,
    tape: &Tape,
    residency: &MutableExpertResidency,
    baseline: &Snapshot,
    name: &str,
    idle: &mut bool,
    report: &mut Value,
) -> R<()> {
    let snapshot = residency.snapshot()?;
    tables_match(gpu, f, &snapshot)?;
    for which in [Arm::Eager, Arm::RecordedHip, Arm::Pm4] {
        let guard = residency.begin_token(snapshot.epoch)?;
        residency.validate_current_entries(&snapshot.gate_up, &snapshot.down)?;
        let outcome = arm(gpu, arena, f, tape, None, which, idle);
        if *idle {
            // SAFETY: arm returned with GPU/HSA retirement proved, even if an
            // output assertion failed afterwards. Unknown quiescence poisons
            // the authority by dropping an unretired guard.
            unsafe { guard.retire_after_gpu_wait() }?;
        } else {
            drop(guard);
        }
        let (output, mut proof) = outcome?;
        check(
            output.differences(baseline) == [0; 3],
            format!("{name}/{} changed output bytes", which.name()),
        )?;
        proof["case"] = json!(name);
        proof["epoch"] = json!(snapshot.epoch);
        proof["production_token_guard"] = json!(true);
        proof["full_outputs_and_guards_match_canonical_host"] = json!(true);
        proof["status"] = json!("PASS");
        report["checks"].as_array_mut().unwrap().push(proof);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn fill(
    gpu: &Gpu,
    arena: &Arena,
    f: &Fixture,
    residency: &MutableExpertResidency,
    dma: &mut Dma,
    sources: [usize; 2],
    ticket: ResidencySlotTicket,
    wrong_payload_control: bool,
    idle: &mut bool,
    report: &mut Value,
) -> R<()> {
    let snapshot = residency.snapshot()?;
    check(
        snapshot
            .mapping
            .iter()
            .all(|m| !matches!(m, ResidencyLocation::Slot(t) if t.slot == ticket.slot)),
        "victim still published before overwrite",
    )?;
    // This event orders the completed pointer retirement before any write on
    // the independent nonblocking stream; host enqueue order is insufficient.
    let retired = dma.event(gpu)?;
    hip(gpu
        .hip
        .event_record(&dma.events[retired], gpu.active_stream.as_ref()))?;
    hip(gpu
        .hip
        .stream_wait_event(dma.stream()?, &dma.events[retired]))?;
    let gu_done = dma.event(gpu)?;
    let dn_done = dma.event(gpu)?;
    let source = ticket.expert % DISTINCT;
    let gu_dest = arena.get(f.gu_device).sub_offset(0, GU_STRIDE);
    let dn_dest = arena.get(f.dn_device).sub_offset(0, DN_STRIDE);
    let copy_gu = |payload| {
        hip(gpu.hip.memcpy_htod_async(
            &gu_dest.buf,
            dma.sources[sources[0]].slice(payload * GU_STRIDE, GU_STRIDE)?,
            dma.stream()?,
        ))
    };
    *idle = false;
    if wrong_payload_control {
        copy_gu((source + 1) % DISTINCT)?;
        hip(gpu
            .hip
            .event_record(&dma.events[gu_done], Some(dma.stream()?)))?;
        hip(gpu.hip.event_synchronize(&dma.events[gu_done]))?;
        *idle = true;
        let wrong = hip(gpu.download_raw_bytes(&gu_dest))?;
        let correct = hip(gpu.download_raw_bytes(
            &arena
                .get(f.gu_host)
                .sub_offset(source * GU_STRIDE, GU_STRIDE),
        ))?;
        check(wrong != correct, "wrong-payload control was vacuous")?;
        report["negative_controls"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "case":"wrong_expert_payload_detected_before_publication", "status":"PASS",
                "wrong_gu_sha256":hash(&wrong),"expected_gu_sha256":hash(&correct),
                "mapping_stayed_canonical":true
            }));
        *idle = false;
    }
    copy_gu(source)?;
    hip(gpu
        .hip
        .event_record(&dma.events[gu_done], Some(dma.stream()?)))?;
    hip(gpu.hip.event_synchronize(&dma.events[gu_done]))?;
    *idle = true;
    check(
        hip(gpu.hip.event_query(&dma.events[gu_done]))?,
        "GU completion not ready after wait",
    )?;
    // DN has not been enqueued: this half-filled state cannot disappear just
    // because the small copies finish before the CPU manages to query them.
    let mut mapping = snapshot.mapping.to_vec();
    mapping[ticket.expert] = ResidencyLocation::Slot(ticket);
    refused_without_upload(
        residency,
        ResidencyPublication {
            expected_epoch: snapshot.epoch,
            mapping: mapping.clone(),
            completed_slots: vec![],
        },
        "GU-only completion",
    )?;
    tables_match(gpu, f, &snapshot)?;
    report["negative_controls"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "case":"gu_complete_dn_not_enqueued", "status":"PASS",
            "gu_completion_event_ready":true, "dn_enqueued":false,
            "publication_rejected_before_upload":true
        }));
    *idle = false;
    hip(gpu.hip.memcpy_htod_async(
        &dn_dest.buf,
        dma.sources[sources[1]].slice(source * DN_STRIDE, DN_STRIDE)?,
        dma.stream()?,
    ))?;
    hip(gpu
        .hip
        .event_record(&dma.events[dn_done], Some(dma.stream()?)))?;
    hip(gpu.hip.event_synchronize(&dma.events[dn_done]))?;
    *idle = true;
    check(
        hip(gpu.hip.event_query(&dma.events[gu_done]))?
            && hip(gpu.hip.event_query(&dma.events[dn_done]))?,
        "paired completion events not ready",
    )?;
    for (device, host, stride) in [
        (f.gu_device, f.gu_host, GU_STRIDE),
        (f.dn_device, f.dn_host, DN_STRIDE),
    ] {
        check(
            hip(gpu.download_raw_bytes(&arena.get(device).sub_offset(0, stride)))?
                == hip(
                    gpu.download_raw_bytes(&arena.get(host).sub_offset(source * stride, stride))
                )?,
            "completed slot bytes differ from canonical source",
        )?;
    }
    publish(gpu, f, residency, mapping, vec![ticket])?;
    report["copies"].as_array_mut().unwrap().push(json!({
        "expert":ticket.expert,"generation":ticket.generation,"slot":ticket.slot,
        "retired_to_canonical_before_overwrite":true,"explicit_compute_to_copy_event":true,
        "copy_api":"memcpy_htod_async","pinned_source":true,"nonblocking_copy_stream":true,
        "gu_bytes":GU_STRIDE,"dn_bytes":DN_STRIDE,
        "both_completion_events_ready_before_publish":true,"full_payload_copy_exact":true
    }));
    Ok(())
}

fn id_copy_oracle(gpu: &mut Gpu, arena: &mut Arena, idle: &mut bool, report: &mut Value) -> R<()> {
    if let Err(failure) = gpu.replay.shutdown() {
        *idle = failure.quiescence == ReplayQuiescence::Proven;
        return Err(format!(
            "ID-copy preceding replay shutdown: {}",
            failure.error
        ));
    }
    let ids = [0_i32, 1, 255, 511, 3, 128, 256, 510, 42, 17];
    let bytes: Vec<u8> = ids.into_iter().flat_map(i32::to_le_bytes).collect();
    let root = arena.raw(gpu, &bytes)?;
    // SAFETY: the Raw root owns exactly these 40 GPU-visible bytes. Its F32
    // borrowed descriptor has ten elements, not a ten-byte Raw sub_offset.
    let source = GpuTensor {
        buf: unsafe {
            hip_bridge::DeviceBuffer::from_raw(arena.get(root).buf.as_ptr(), bytes.len())
        },
        shape: vec![TOP],
        dtype: rdna_compute::DType::F32,
    };
    check(
        source.buf.size() == TOP * 4,
        "ID-copy F32 view must cover forty bytes",
    )?;
    let output = Guarded::new(gpu, arena, TOP)?;
    gpu.replay = ReplayController::new(ReplayBackendRequest::Hip);
    *idle = false;
    hip(gpu.copy_f32_buffer(&output.view, &source, TOP))?;
    hip(gpu.hip.device_synchronize())?;
    *idle = true;
    gpu.replay = ReplayController::new_manual_pm4();
    gpu.replay.set_forward_eligible(true);
    hip(gpu.replay.begin_capture())?;
    *idle = false;
    hip(gpu.copy_f32_buffer(&output.view, &source, TOP))?;
    hip(gpu.replay.finish_capture())?;
    hip(gpu.hip.device_synchronize())?;
    *idle = true;
    check(
        gpu.replay.recorded_launches().len() == 1,
        "ID-copy single-kernel census",
    )?;
    let launches = gpu.replay.recorded_launches().to_vec();
    let (count, _, _) = gpu.replay.prepare_pm4_prefix(gpu.device_id as usize, 1)?;
    check(count == 1, "ID-copy PM4 dispatch census")?;
    let identity = gpu
        .replay
        .prepared_route_identity()
        .ok_or("ID-copy prepared identity absent")?;
    let mut expected = POISON.to_le_bytes().repeat(TOP + 2 * GUARD);
    expected[GUARD * 4..(GUARD + TOP) * 4].copy_from_slice(&bytes);
    for which in [Arm::Eager, Arm::RecordedHip, Arm::Pm4] {
        output.reset(gpu, arena)?;
        hip(gpu.hip.device_synchronize())?;
        let before = launch_count();
        gpu.replay.begin_replay_observation_window();
        *idle = false;
        match which {
            Arm::Eager => hip(gpu.copy_f32_buffer(&output.view, &source, TOP))?,
            Arm::RecordedHip => hip(gpu.replay_recorded_hip_prefix_at(1, 0))?,
            Arm::Pm4 => {
                if let Err(failure) = unsafe { gpu.replay.replay_pm4_checked(0) } {
                    *idle = failure.quiescence == ReplayQuiescence::Proven;
                    return Err(format!("ID-copy PM4: {}", failure.error));
                }
            }
            Arm::HipGraph => unreachable!(),
        }
        hip(gpu.hip.device_synchronize())?;
        *idle = true;
        let actual = hip(gpu.download_raw_bytes(arena.get(output.root)))?;
        check(
            actual == expected,
            format!("ID-copy {} changed raw i32 bits or guards", which.name()),
        )?;
        let observed = gpu.replay.replay_observation();
        let hip_launches = launch_count().saturating_sub(before);
        check(
            hip_launches == u64::from(!matches!(which, Arm::Pm4)),
            "ID-copy actual launch census",
        )?;
        check(
            observed.count == u64::from(matches!(which, Arm::Pm4)) && !observed.failed,
            "ID-copy replay observation",
        )?;
        check(
            gpu.replay.recorded_launches() == launches.as_slice()
                && gpu.replay.prepared_route_identity() == Some(identity),
            "ID-copy tape changed",
        )?;
        report["typed_id_copy"].as_array_mut().unwrap().push(json!({
            "arm":which.name(),"status":"PASS","ids":ids,"raw_view_bytes":source.buf.size(),
            "same_separate_id_tape":true,"full_bytes_and_guards_exact":true,
            "output_sha256":hash(&actual),"hip_kernel_launches":hip_launches,"pm4_replays":observed.count
        }));
    }
    Ok(())
}

fn exercise(
    gpu: &mut Gpu,
    owner: &Arc<Mutex<Arena>>,
    dma: &mut Dma,
    idle: &mut bool,
    report: &mut Value,
) -> R<()> {
    check(
        gpu.arch == "gfx1201",
        format!("requires exact gfx1201, got {}", gpu.arch),
    )?;
    hip(gpu.ensure_capture_stream())?;
    dma.stream = Some(hip(gpu.hip.stream_create_non_blocking())?);
    let mut arena = owner.lock().map_err(|_| "arena lock poisoned")?;
    let f = Fixture::build(gpu, &mut arena, report)?;
    f.publish(
        gpu,
        &arena,
        Case {
            name: "canonical",
            gu_host: true,
            dn_host: true,
            wrong_gu: false,
            wrong_dn: false,
        },
    )?;
    // Separate, immutable pinned H2D sources contain the exact canonical bytes.
    // Their lifetimes cover all event waits, just as the transport's tier owner.
    let source_gu = dma.source(gpu, &hip(gpu.download_raw_bytes(arena.get(f.gu_host)))?)?;
    let source_dn = dma.source(gpu, &hip(gpu.download_raw_bytes(arena.get(f.dn_host)))?)?;
    let sources = [source_gu, source_dn];
    // A separate table pair permits a physical partial-upload failure without
    // corrupting either fixed table captured by the primary PM4 oracle.
    let bytes_gu = hip(gpu.download_raw_bytes(&f.gu_table))?;
    let bytes_dn = hip(gpu.download_raw_bytes(&f.dn_table))?;
    let bad_gu_root = arena.raw(gpu, &bytes_gu)?;
    let bad_dn_root = arena.raw(gpu, &bytes_dn)?;
    let bad_gu = arena.get(bad_gu_root).sub_offset(0, E * 8);
    let bad_dn = arena.get(bad_dn_root).sub_offset(0, E * 8);
    let residency = capability(owner, &arena, &f, &f.gu_table, &f.dn_table)?;
    let bad_residency = capability(owner, &arena, &f, &bad_gu, &bad_dn)?;
    let fixed_identity = residency.stable_fingerprint().to_owned();
    let initial = residency.snapshot()?;
    let guard = residency.begin_token(initial.epoch)?;
    refused_without_upload(
        &residency,
        ResidencyPublication {
            expected_epoch: initial.epoch,
            mapping: initial.mapping.to_vec(),
            completed_slots: vec![],
        },
        "busy token",
    )?;
    // SAFETY: this guard has not submitted any GPU work.
    unsafe { guard.retire_after_gpu_wait() }?;
    report["negative_controls"]
        .as_array_mut()
        .unwrap()
        .push(json!({"case":"busy_token_blocks_publication","status":"PASS"}));

    gpu.replay = ReplayController::new(ReplayBackendRequest::Hip);
    let guard = residency.begin_token(initial.epoch)?;
    residency.validate_current_entries(&initial.gate_up, &initial.down)?;
    f.reset(gpu, &arena)?;
    *idle = false;
    f.launch(gpu)?;
    hip(gpu.hip.device_synchronize())?;
    *idle = true;
    // SAFETY: the warmup's GPU work has completed.
    unsafe { guard.retire_after_gpu_wait() }?;
    let baseline = f.read(gpu, &arena)?;
    f.reset(gpu, &arena)?;
    gpu.replay = ReplayController::new_manual_pm4();
    gpu.replay.set_forward_eligible(true);
    let guard = residency.begin_token(initial.epoch)?;
    residency.validate_current_entries(&initial.gate_up, &initial.down)?;
    hip(gpu.replay.begin_capture())?;
    *idle = false;
    f.launch(gpu)?;
    hip(gpu.replay.finish_capture())?;
    hip(gpu.hip.device_synchronize())?;
    *idle = true;
    // SAFETY: captured HIP body completed, not just CPU recording.
    unsafe { guard.retire_after_gpu_wait() }?;
    check(
        f.read(gpu, &arena)?.differences(&baseline) == [0; 3],
        "capture differs from independent eager baseline",
    )?;
    let (dispatches, _, _) = gpu.replay.prepare_pm4_prefix(gpu.device_id as usize, 2)?;
    check(
        dispatches == 2 && gpu.replay.recorded_launches().len() == 2,
        "two-kernel capture census",
    )?;
    let tape = Tape {
        launches: gpu.replay.recorded_launches().to_vec(),
        prepared: gpu
            .replay
            .prepared_route_identity()
            .ok_or("prepared identity absent")?,
        gu_table: f.gu_table.buf.as_ptr() as usize,
        dn_table: f.dn_table.buf.as_ptr() as usize,
    };
    report["capture"] = json!({"captures":1,"dispatches":2,"prepared_identity":format!("{:?}",tape.prepared),
        "fixed_table_allocations":true,"all_transfers_outside_capture":true});
    run_case(
        gpu,
        &arena,
        &f,
        &tape,
        &residency,
        &baseline,
        "canonical_host",
        idle,
        report,
    )?;
    fill(
        gpu,
        &arena,
        &f,
        &residency,
        dma,
        sources,
        ResidencySlotTicket {
            slot: 0,
            expert: 0,
            generation: 1,
        },
        true,
        idle,
        report,
    )?;
    run_case(
        gpu,
        &arena,
        &f,
        &tape,
        &residency,
        &baseline,
        "slot_expert0",
        idle,
        report,
    )?;
    publish(
        gpu,
        &f,
        &residency,
        vec![ResidencyLocation::Canonical; E],
        vec![],
    )?;
    run_case(
        gpu,
        &arena,
        &f,
        &tape,
        &residency,
        &baseline,
        "victim_host_before_overwrite",
        idle,
        report,
    )?;
    fill(
        gpu,
        &arena,
        &f,
        &residency,
        dma,
        sources,
        ResidencySlotTicket {
            slot: 0,
            expert: 1,
            generation: 2,
        },
        false,
        idle,
        report,
    )?;
    run_case(
        gpu,
        &arena,
        &f,
        &tape,
        &residency,
        &baseline,
        "slot_reused_expert1",
        idle,
        report,
    )?;
    let snapshot = residency.snapshot()?;
    let stale = ResidencySlotTicket {
        slot: 0,
        expert: 0,
        generation: 1,
    };
    let mut stale_mapping = vec![ResidencyLocation::Canonical; E];
    stale_mapping[0] = ResidencyLocation::Slot(stale);
    refused_without_upload(
        &residency,
        ResidencyPublication {
            expected_epoch: snapshot.epoch,
            mapping: stale_mapping,
            completed_slots: vec![stale],
        },
        "stale slot generation",
    )?;
    tables_match(gpu, &f, &snapshot)?;
    report["negative_controls"]
        .as_array_mut()
        .unwrap()
        .push(json!({"case":"stale_slot_generation","status":"PASS"}));
    publish(
        gpu,
        &f,
        &residency,
        vec![ResidencyLocation::Canonical; E],
        vec![],
    )?;
    run_case(
        gpu,
        &arena,
        &f,
        &tape,
        &residency,
        &baseline,
        "canonical_after_reuse",
        idle,
        report,
    )?;
    check(
        residency.stable_fingerprint() == fixed_identity,
        "publication changed stable owner identity",
    )?;

    let snapshot = bad_residency.snapshot()?;
    // SAFETY: this independent authority has no readers; the synchronous first
    // table write completes. Failure before the second upload must poison the
    // old epoch. These separate tables are never used for a kernel launch.
    let failed = unsafe {
        bad_residency.publish_after_gpu_wait(
            ResidencyPublication {
                expected_epoch: snapshot.epoch,
                mapping: snapshot.mapping.to_vec(),
                completed_slots: vec![],
            },
            |_| {
                let mut wrong = words(&snapshot.gate_up);
                wrong[..8]
                    .copy_from_slice(&(arena.get(f.gu_device).buf.as_ptr() as u64).to_le_bytes());
                hip(gpu.hip.memcpy_htod(&bad_gu.buf, &wrong))?;
                Err("injected failure after first actual table upload".into())
            },
        )
    };
    check(
        failed.is_err()
            && bad_residency.snapshot().is_err()
            && bad_residency.begin_token(0).is_err(),
        "partial upload failed to poison authority",
    )?;
    check(
        hip(gpu.download_raw_bytes(&bad_gu))? != words(&snapshot.gate_up),
        "partial injection did not change bytes",
    )?;
    report["negative_controls"].as_array_mut().unwrap().push(json!({
        "case":"actual_partial_table_upload_poisoned","status":"PASS","actual_first_table_bytes_changed":true,
        "subsequent_token_refused":true
    }));
    tape.verify(gpu, &f)?;
    // All residency cases used the same GU/DN tape. The collector-copy oracle
    // now records its own one-kernel tape, never recapturing that main body.
    id_copy_oracle(gpu, &mut arena, idle, report)?;
    report["checks_expected"] = json!(15);
    report["typed_id_copy_checks_expected"] = json!(3);
    report["all_checks_complete"] = json!(true);
    Ok(())
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("fn_expert_cache_publication [--execute]\nNo model; exact gfx1201; caller holds shared GPU lock. Default PLAN_ONLY.");
        return;
    }
    let mut report = json!({"schema":1,"scope":"async_publication_mechanism","no_model":true,
        "performance_claim":false,"production_mutable_residency_contract_exercised":false,
        "production_transport_manager_exercised":false,"sealed_moe_binding_exercised":false,
        "checks":[],"copies":[],"negative_controls":[],"typed_id_copy":[],"all_checks_complete":false,
        "limitations":["No full model, logits, routing, quality or speed measurement.",
        "Independent GU/DN inputs isolate publication; this is not an MoE activation chain.",
        "Uses production residency authority, not the full model transport manager or sealed binding.",
        "Deterministic half-filled state is GU complete with DN not enqueued; no pending-time claim.",
        "H2D source allocations are separate pinned copies of canonical host bytes, verified by full readback."]});
    if args.iter().any(|a| a != "--execute") {
        report["status"] = json!("FAIL");
        report["error"] = json!("unknown argument");
        println!("{report}");
        std::process::exit(1);
    }
    if !args.iter().any(|a| a == "--execute") {
        report["status"] = json!("PLAN_ONLY");
        report["checks_planned"] = json!(15);
        println!("{report}");
        return;
    }
    let mut gpu = match Gpu::init() {
        Ok(gpu) => gpu,
        Err(error) => {
            report["status"] = json!("FAIL");
            report["error"] = json!(error.to_string());
            println!("{report}");
            std::process::exit(1);
        }
    };
    let owner = Arc::new(Mutex::new(Arena::default()));
    let mut dma = Dma::default();
    let mut idle = true;
    let result = exercise(&mut gpu, &owner, &mut dma, &mut idle, &mut report);
    let mut cleanup = Vec::new();
    if let Err(error) = &result {
        report["error"] = json!(error);
    }
    if idle {
        if let Err(failure) = gpu.replay.shutdown() {
            idle = failure.quiescence == ReplayQuiescence::Proven;
            cleanup.push(format!(
                "retained shutdown: {}; {:?}",
                failure.error, failure.quiescence
            ));
        }
    }
    if idle {
        if let Err(error) = dma.release(&gpu) {
            cleanup.push(error);
        }
        match owner.lock() {
            Ok(mut arena) => {
                if let Err(error) = arena.release(&mut gpu) {
                    cleanup.push(error);
                }
            }
            Err(_) => {
                cleanup.push("arena mutex poisoned; allocations retained until process exit".into())
            }
        }
    } else {
        cleanup.push(
            "GPU quiescence unknown: captured owners/controller are retained until process exit"
                .into(),
        );
    }
    let pass = result.is_ok() && cleanup.is_empty();
    report["cleanup_errors"] = json!(cleanup);
    report["gpu_quiescence_proven_at_cleanup"] = json!(idle);
    report["production_mutable_residency_contract_exercised"] = json!(pass);
    report["status"] = json!(if pass { "PASS" } else { "FAIL" });
    println!("{report}");
    if !idle {
        std::mem::forget(owner);
        std::mem::forget(gpu);
    }
    if !pass {
        std::process::exit(1);
    }
}
