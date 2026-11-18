//! S3b plan gate R on exact gfx1201: a retained single-position MoE step
//! replays across a forced expert remap, and every replay bypass refuses a
//! launch that has no valid generation lease.
//!
//! One recorded step (the S0 gate/up + down indexed kernels over fixed pointer
//! tables) is captured once under a live reader lease of the production
//! `MutableExpertResidency`, with the residency registered on the tape through
//! `ReplayController::note_launch_authority` exactly where the sealed MoE call
//! (`SealedMoeCall::note_retained_route_identity`) registers it. Then, per
//! backend (recorded HIP, PM4, AQL):
//!   * replay A under a valid lease is byte-equal to eager;
//!   * the routed experts move to different slots (fill generation g -> g+1,
//!     published through the real residency path; table addresses, extents and
//!     the stable identity stay fixed, the pointer contents change);
//!   * replay B of the SAME recorded program under a valid lease is byte-equal
//!     to eager recomputed on the remapped placement (the retired slots are
//!     scrubbed first, so a replay that read the old placement could not pass);
//!   * a retired (stale) lease, no live lease and a poisoned residency are each
//!     refused by `authorize_retained_launch` before submission: the named
//!     error, no HIP launch, no replay observation, sentinel outputs untouched.
//!
//! Scope: no model; the sealed MoE call itself is NOT run (the fixture launches
//! the same indexed kernels the call dispatches and registers the authority by
//! hand); single position only. Default: PLAN_ONLY. `--execute` needs the
//! caller's shared GPU lock.
//!
//! ```text
//! cargo run --release -p hipfire-arch-qwen4 --features lab \
//!     --example fn_expert_cache_replay_remap -- --execute --json out.json
//! ```

#[path = "../../rdna-compute/examples/support/fn_expert_fixture.rs"]
#[allow(dead_code)] // Shared S0 helpers also contain its optional hipGraph arm.
mod fixture;
use fixture::*;
use hipfire_dispatch::families::moe::{
    MutableExpertResidency, MutableExpertResidencySpec, ResidencyLease, ResidencyLocation,
    ResidencyPublication, ResidencySlot, ResidencySlotTicket, ResidencySnapshot, ResidencySpan,
};
use rdna_compute::replay::{
    RecordedHipLaunch, ReplayBackendRequest, ReplayController, ReplayQuiescence,
};
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

/// Slots 0..3 hold the routed experts before the remap, 3..6 after it.
const SLOTS: usize = 6;
const MOVED: usize = 3;
const SCRUB: u8 = 0xa5;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Backend {
    RecordedHip,
    Pm4,
    Aql,
}
impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::RecordedHip => "recorded_hip",
            Self::Pm4 => "pm4",
            Self::Aql => "aql",
        }
    }
}

struct Phase {
    name: &'static str,
    aql: bool,
    backends: &'static [Backend],
}
const PHASES: [Phase; 2] = [
    Phase {
        name: "recorded_hip+pm4",
        aql: false,
        backends: &[Backend::RecordedHip, Backend::Pm4],
    },
    Phase {
        name: "aql",
        aql: true,
        backends: &[Backend::Aql],
    },
];

#[derive(Default)]
struct Report {
    checks: Vec<Value>,
    pass: usize,
    fail: usize,
    skip: usize,
}
impl Report {
    fn record(&mut self, scope: &str, id: &str, outcome: R<String>) -> bool {
        let (status, detail) = match outcome {
            Ok(detail) => ("PASS", detail),
            Err(detail) => ("FAIL", detail),
        };
        if status == "PASS" {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
        println!("{status} {scope} {id}: {detail}");
        self.checks
            .push(json!({"scope": scope, "check": id, "status": status, "detail": detail}));
        status == "PASS"
    }
    fn skip(&mut self, scope: &str, id: &str, reason: &str) {
        self.skip += 1;
        println!("SKIP {scope} {id}: {reason}");
        self.checks
            .push(json!({"scope": scope, "check": id, "status": "SKIP", "detail": reason}));
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
    // SAFETY: every prior reader retired (all tokens were retired after a device
    // synchronize), every supplied ticket names a slot whose synchronous H2D
    // copies and readback completed, and no filled slot is currently mapped.
    // The synchronous table writes plus device sync complete before any
    // subsequent HIP/PM4/AQL submission.
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
                hip(gpu.hip.device_synchronize())
            },
        )
    }?;
    tables_match(gpu, f, &next)?;
    Ok(next)
}

fn mapping(tickets: &[ResidencySlotTicket]) -> Vec<ResidencyLocation> {
    let mut mapping = vec![ResidencyLocation::Canonical; E];
    for ticket in tickets {
        mapping[ticket.expert] = ResidencyLocation::Slot(*ticket);
    }
    mapping
}

fn capability(
    owner: &Arc<Mutex<Arena>>,
    arena: &Arena,
    f: &Fixture,
    slots: &[(usize, usize)],
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
    // SAFETY: the retained Arena owns every fixed canonical/table/slot root and
    // outlives every tape and capability (it is freed only after replay
    // shutdown proved quiescence). Canonical payloads are never overwritten.
    // Both tables already contain the canonical addresses.
    unsafe {
        MutableExpertResidency::new(MutableExpertResidencySpec {
            owner_generation: 1,
            canonical_gate_up: canonical(f.gu_host, GU_STRIDE),
            canonical_down: canonical(f.dn_host, DN_STRIDE),
            gate_up_table: span(&f.gu_table),
            down_table: span(&f.dn_table),
            slots: slots
                .iter()
                .map(|&(gu, dn)| ResidencySlot {
                    gate_up: ResidencySpan {
                        address: arena.get(gu).buf.as_ptr() as usize,
                        bytes: GU_STRIDE,
                    },
                    down: ResidencySpan {
                        address: arena.get(dn).buf.as_ptr() as usize,
                        bytes: DN_STRIDE,
                    },
                })
                .collect(),
            keepalive: owner.clone(),
        })
    }
}

/// Synchronously copy logical expert `expert`'s canonical payload into a slot.
fn fill_slot(
    gpu: &Gpu,
    arena: &Arena,
    slot: (usize, usize),
    expert: usize,
    gu: &[u8],
    dn: &[u8],
) -> R<()> {
    let payload = expert % DISTINCT;
    let g = &gu[payload * GU_STRIDE..(payload + 1) * GU_STRIDE];
    let d = &dn[payload * DN_STRIDE..(payload + 1) * DN_STRIDE];
    hip(gpu.hip.memcpy_htod(&arena.get(slot.0).buf, g))?;
    hip(gpu.hip.memcpy_htod(&arena.get(slot.1).buf, d))?;
    hip(gpu.hip.device_synchronize())?;
    check(
        hip(gpu.download_raw_bytes(arena.get(slot.0)))? == g
            && hip(gpu.download_raw_bytes(arena.get(slot.1)))? == d,
        "slot fill readback",
    )
}

/// Overwrite a retired (unmapped) slot so a reader of the old placement would
/// compute different bytes.
fn scrub_slot(gpu: &Gpu, arena: &Arena, slot: (usize, usize)) -> R<()> {
    hip(gpu
        .hip
        .memcpy_htod(&arena.get(slot.0).buf, &vec![SCRUB; GU_STRIDE]))?;
    hip(gpu
        .hip
        .memcpy_htod(&arena.get(slot.1).buf, &vec![SCRUB; DN_STRIDE]))?;
    hip(gpu.hip.device_synchronize())
}

/// Run `body` under one production reader lease at the published epoch. The
/// guard retires only when `idle` proves GPU quiescence; otherwise dropping it
/// poisons the authority (unknown readers can never be retired).
fn under_lease<T>(
    residency: &MutableExpertResidency,
    idle: &mut bool,
    body: impl FnOnce(&mut bool, ResidencyLease) -> R<T>,
) -> R<T> {
    let snapshot = residency.snapshot()?;
    let guard = residency.begin_token(snapshot.epoch)?;
    let admitted = residency
        .validate_current_entries(&snapshot.gate_up, &snapshot.down)
        .and_then(|()| residency.authorize_lease(guard.lease()));
    if let Err(error) = admitted {
        // SAFETY: no GPU work was submitted under this guard.
        unsafe { guard.retire_after_gpu_wait() }?;
        return Err(error);
    }
    let lease = guard.lease();
    let outcome = body(idle, lease);
    if *idle {
        // SAFETY: the body returned with GPU/HSA retirement proved.
        unsafe { guard.retire_after_gpu_wait() }?;
    } else {
        drop(guard);
    }
    outcome
}

fn eager(
    gpu: &mut Gpu,
    arena: &Arena,
    f: &Fixture,
    residency: &MutableExpertResidency,
    idle: &mut bool,
) -> R<Snapshot> {
    under_lease(residency, idle, |idle, _lease| {
        f.reset(gpu, arena)?;
        *idle = false;
        f.launch(gpu)?;
        hip(gpu.hip.device_synchronize())?;
        *idle = true;
        f.read(gpu, arena)
    })
}

struct Recorded {
    launches: Vec<RecordedHipLaunch>,
    gu_table: usize,
    dn_table: usize,
    aql: bool,
}
impl Recorded {
    fn verify(&self, gpu: &Gpu, f: &Fixture) -> R<()> {
        check(
            gpu.replay.recorded_launches() == self.launches.as_slice(),
            "recorded tape changed or was recaptured",
        )?;
        check(
            if self.aql {
                gpu.replay.prepared_aql_plan_ready()
            } else {
                gpu.replay.prepared_pm4_plan_ready()
            },
            "prepared retained plan absent",
        )?;
        check(
            f.gu_table.buf.as_ptr() as usize == self.gu_table
                && f.dn_table.buf.as_ptr() as usize == self.dn_table,
            "pointer table allocation changed",
        )
    }
}

type Refusal = (String, Option<ReplayQuiescence>);

fn submit(gpu: &mut Gpu, backend: Backend) -> Result<(), Refusal> {
    match backend {
        Backend::RecordedHip => hip(gpu.replay_recorded_hip_prefix_at(2, 0)).map_err(|e| (e, None)),
        Backend::Pm4 => {
            // SAFETY: every captured allocation/pointee is retained by the
            // Arena and only table contents ever change.
            unsafe { gpu.replay.replay_pm4_checked(0) }
                .map(|_| ())
                .map_err(|failure| (failure.error, Some(failure.quiescence)))
        }
        Backend::Aql => {
            // SAFETY: as for PM4.
            unsafe { gpu.replay.replay_linear_aql(0) }
                .map(|_| ())
                .map_err(|e| (e, None))
        }
    }
}

/// A replay under a valid lease: byte-checked against the caller's reference.
#[allow(clippy::too_many_arguments)]
fn replay_ok(
    gpu: &mut Gpu,
    arena: &Arena,
    f: &Fixture,
    rec: &Recorded,
    backend: Backend,
    residency: &MutableExpertResidency,
    idle: &mut bool,
) -> R<(Snapshot, ResidencyLease)> {
    under_lease(residency, idle, |idle, lease| {
        rec.verify(gpu, f)?;
        f.reset(gpu, arena)?;
        let (hip0, graph0) = (launch_count(), graph_count());
        gpu.replay.begin_replay_observation_window();
        *idle = false;
        if let Err((error, quiescence)) = submit(gpu, backend) {
            if quiescence == Some(ReplayQuiescence::Proven) {
                *idle = true;
            }
            return Err(format!("{}: {error}", backend.name()));
        }
        hip(gpu.hip.device_synchronize())?;
        *idle = true;
        let launches = launch_count().saturating_sub(hip0);
        let graphs = graph_count().saturating_sub(graph0);
        let observation = gpu.replay.replay_observation();
        match backend {
            Backend::RecordedHip => {
                check(
                    launches == 2 && graphs == 0,
                    format!("recorded HIP launch count {launches}/{graphs}"),
                )?;
                check(
                    observation.count == 0 && !observation.failed,
                    "recorded HIP replay was observed as retained",
                )?;
            }
            Backend::Pm4 | Backend::Aql => {
                check(
                    launches == 0 && graphs == 0,
                    format!("{} silently ran a HIP body", backend.name()),
                )?;
                check(
                    observation.count == 1
                        && observation.first_position == Some(0)
                        && observation.last_position == Some(0)
                        && !observation.failed,
                    format!("replay observation: {observation:?}"),
                )?;
            }
        }
        rec.verify(gpu, f)?;
        Ok((f.read(gpu, arena)?, lease))
    })
}

fn outputs_untouched(gpu: &Gpu, arena: &Arena, f: &Fixture) -> R<bool> {
    for guarded in [&f.gate, &f.up, &f.down] {
        let sentinel = POISON.to_le_bytes().repeat(guarded.elems + 2 * GUARD);
        if hip(gpu.download_raw_bytes(arena.get(guarded.root)))? != sentinel {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A replay attempt that must be refused before anything is submitted.
fn replay_refused(
    gpu: &mut Gpu,
    arena: &Arena,
    f: &Fixture,
    rec: &Recorded,
    backend: Backend,
    expect: &str,
) -> R<String> {
    rec.verify(gpu, f)?;
    f.reset(gpu, arena)?;
    let (hip0, graph0) = (launch_count(), graph_count());
    gpu.replay.begin_replay_observation_window();
    let outcome = submit(gpu, backend);
    let alternate = if backend == Backend::RecordedHip {
        Some(hip(gpu.replay_recorded_hip_prefix(2)))
    } else {
        None
    };
    hip(gpu.hip.device_synchronize())?;
    let (error, quiescence) = match outcome {
        Ok(()) => {
            return Err(format!(
                "{}: replay was submitted without launch authority",
                backend.name()
            ))
        }
        Err(refusal) => refusal,
    };
    check(
        error.contains(expect),
        format!(
            "{}: refusal {error:?} does not name {expect:?}",
            backend.name()
        ),
    )?;
    if backend == Backend::Pm4 {
        check(
            quiescence == Some(ReplayQuiescence::Proven),
            format!("PM4 refusal quiescence {quiescence:?}"),
        )?;
    }
    if let Some(alternate) = alternate {
        match alternate {
            Ok(()) => {
                return Err("replay_recorded_hip_prefix submitted without launch authority".into())
            }
            Err(error) => check(
                error.contains(expect),
                format!("replay_recorded_hip_prefix refusal {error:?} does not name {expect:?}"),
            )?,
        }
    }
    check(
        launch_count() == hip0 && graph_count() == graph0,
        "a refused replay still submitted a HIP launch",
    )?;
    check(
        gpu.replay.replay_observation().count == 0,
        "a refused replay was observed as executed",
    )?;
    check(
        outputs_untouched(gpu, arena, f)?,
        "a refused replay wrote the sentinel outputs",
    )?;
    Ok(format!("refused before submission: {error}"))
}

#[derive(Debug, PartialEq)]
struct Identity {
    gu_table: (usize, usize),
    dn_table: (usize, usize),
    residency_tables: (ResidencySpan, ResidencySpan),
    recorded_tables: (usize, usize),
    fingerprint: String,
    owner_generation: u64,
    authorities: usize,
}
fn identity(
    gpu: &Gpu,
    f: &Fixture,
    residency: &MutableExpertResidency,
    rec: &Recorded,
) -> Identity {
    Identity {
        gu_table: (f.gu_table.buf.as_ptr() as usize, f.gu_table.buf.size()),
        dn_table: (f.dn_table.buf.as_ptr() as usize, f.dn_table.buf.size()),
        residency_tables: (residency.gate_up_table(), residency.down_table()),
        recorded_tables: (rec.gu_table, rec.dn_table),
        fingerprint: residency.stable_fingerprint().to_owned(),
        owner_generation: residency.owner_generation(),
        authorities: gpu.replay.launch_authority_count(),
    }
}

fn slot_generation(snapshot: &ResidencySnapshot) -> Option<u64> {
    match snapshot.mapping[0] {
        ResidencyLocation::Slot(ticket) => Some(ticket.generation),
        ResidencyLocation::Canonical => None,
    }
}

fn exercise(
    gpu: &mut Gpu,
    owner: &Arc<Mutex<Arena>>,
    phase: &Phase,
    idle: &mut bool,
    rep: &mut Report,
) -> R<()> {
    let scope = phase.name;
    check(
        gpu.arch == "gfx1201",
        format!("requires exact gfx1201, got {}", gpu.arch),
    )?;
    hip(gpu.ensure_capture_stream())?;
    let mut arena = owner.lock().map_err(|_| "arena lock poisoned")?;
    let f = Fixture::build(gpu, &mut arena, &mut json!({}))?;
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
    let canonical_gu = hip(gpu.download_raw_bytes(arena.get(f.gu_host)))?;
    let canonical_dn = hip(gpu.download_raw_bytes(arena.get(f.dn_host)))?;
    let mut slots = Vec::new();
    for _ in 0..SLOTS {
        let gu = arena.raw(gpu, &vec![0u8; GU_STRIDE])?;
        let dn = arena.raw(gpu, &vec![0u8; DN_STRIDE])?;
        slots.push((gu, dn));
    }
    let residency = capability(owner, &arena, &f, &slots)?;
    tables_match(gpu, &f, &residency.snapshot()?)?;

    // Placement A: the routed experts 0..3 live in slots 0..3, fill generation 1.
    for expert in 0..MOVED {
        fill_slot(
            gpu,
            &arena,
            slots[expert],
            expert,
            &canonical_gu,
            &canonical_dn,
        )?;
    }
    let tickets_a: Vec<_> = (0..MOVED)
        .map(|i| ResidencySlotTicket {
            slot: i,
            expert: i,
            generation: 1,
        })
        .collect();
    let a = publish(gpu, &f, &residency, mapping(&tickets_a), tickets_a.clone())?;
    let a_tables = (
        hip(gpu.download_raw_bytes(&f.gu_table))?,
        hip(gpu.download_raw_bytes(&f.dn_table))?,
    );

    gpu.replay = ReplayController::new(ReplayBackendRequest::Hip);
    let baseline = eager(gpu, &arena, &f, &residency, idle)?;

    // Record ONE sealed-step equivalent under a live lease at placement A, with
    // the residency registered on the tape the way the sealed call registers it.
    gpu.replay = if phase.aql {
        ReplayController::new_manual_aql()
    } else {
        ReplayController::new_manual_pm4()
    };
    gpu.replay.set_forward_eligible(true);
    let snapshot = residency.snapshot()?;
    let guard = residency.begin_token(snapshot.epoch)?;
    residency.validate_current_entries(&snapshot.gate_up, &snapshot.down)?;
    let recorded_lease = guard.lease();
    f.reset(gpu, &arena)?;
    hip(gpu.replay.begin_capture())?;
    gpu.replay
        .note_launch_authority(&residency, residency.owner_generation())?;
    *idle = false;
    f.launch(gpu)?;
    hip(gpu.replay.finish_capture())?;
    hip(gpu.hip.device_synchronize())?;
    *idle = true;
    // SAFETY: the captured HIP body completed, not just CPU recording.
    unsafe { guard.retire_after_gpu_wait() }?;
    check(
        f.read(gpu, &arena)?.differences(&baseline) == [0; 3],
        "capture differs from the eager baseline",
    )?;
    let prepared = if phase.aql {
        gpu.replay
            .prepare_linear_aql_prefix(gpu.device_id as usize, 2)
            .map(|(dispatches, _, _)| dispatches)
    } else {
        gpu.replay
            .prepare_pm4_prefix(gpu.device_id as usize, 2)
            .map(|(dispatches, _, _)| dispatches)
    };
    match prepared {
        Ok(dispatches) => check(
            dispatches == 2 && gpu.replay.recorded_launches().len() == 2,
            "two-kernel capture census",
        )?,
        Err(reason) if phase.aql => {
            rep.skip(
                scope,
                "backend_prepare",
                &format!("AQL prepare refused: {reason}"),
            );
            return Ok(());
        }
        Err(reason) => return Err(format!("PM4 prepare refused: {reason}")),
    }
    let rec = Recorded {
        launches: gpu.replay.recorded_launches().to_vec(),
        gu_table: f.gu_table.buf.as_ptr() as usize,
        dn_table: f.dn_table.buf.as_ptr() as usize,
        aql: phase.aql,
    };
    rep.record(
        scope,
        "authority_registered_on_tape",
        check(
            gpu.replay.launch_authority_count() == 1,
            format!(
                "launch authority count {}",
                gpu.replay.launch_authority_count()
            ),
        )
        .map(|()| {
            format!(
                "1 authority recorded at owner generation {}",
                recorded_lease.owner_generation
            )
        }),
    );
    let controller_gate = {
        let without = gpu.replay.authorize_retained_launch();
        let with = under_lease(&residency, idle, |_, _| {
            Ok(gpu.replay.authorize_retained_launch())
        })?;
        match (without, with) {
            (Err(e), Ok(())) if e.contains("no live generation lease") => {
                Ok("controller refuses without a lease, accepts under one".to_owned())
            }
            other => Err(format!("controller authorization {other:?}")),
        }
    };
    rep.record(scope, "controller_authorize_follows_lease", controller_gate);

    // Replay A under a valid lease at generation 1.
    let eager_a = eager(gpu, &arena, &f, &residency, idle)?;
    rep.record(
        scope,
        "eager_stable_across_capture",
        check(
            eager_a.differences(&baseline) == [0; 3],
            "eager output changed after capture",
        )
        .map(|()| "eager output unchanged".to_owned()),
    );
    for &backend in phase.backends {
        let outcome = replay_ok(gpu, &arena, &f, &rec, backend, &residency, idle);
        let (output, lease) = match outcome {
            Ok(done) => done,
            Err(error) if *idle => {
                rep.record(
                    scope,
                    &format!("{}/replay_a_byte_equal_eager", backend.name()),
                    Err(error),
                );
                continue;
            }
            Err(error) => return Err(error),
        };
        rep.record(
            scope,
            &format!("{}/replay_a_byte_equal_eager", backend.name()),
            check(
                output.differences(&eager_a) == [0; 3],
                format!("{} replay A differs from eager", backend.name()),
            )
            .map(|()| {
                format!(
                    "lease generation {} epoch {}, down sha256 {}",
                    lease.owner_generation,
                    lease.epoch,
                    &hash(&output.down)[..16]
                )
            }),
        );
    }

    // Forced remap: the routed experts move to different slots at fill
    // generation 2, published through the real residency path.
    let before = identity(gpu, &f, &residency, &rec);
    for i in 0..MOVED {
        fill_slot(
            gpu,
            &arena,
            slots[MOVED + i],
            i,
            &canonical_gu,
            &canonical_dn,
        )?;
    }
    let tickets_b: Vec<_> = (0..MOVED)
        .map(|i| ResidencySlotTicket {
            slot: MOVED + i,
            expert: i,
            generation: 2,
        })
        .collect();
    let b = publish(gpu, &f, &residency, mapping(&tickets_b), tickets_b.clone())?;
    for &slot in &slots[..MOVED] {
        scrub_slot(gpu, &arena, slot)?;
    }
    let after = identity(gpu, &f, &residency, &rec);
    rep.record(
        scope,
        "remap_published_new_generation",
        check(
            b.epoch == a.epoch + 1
                && slot_generation(&a) == Some(1)
                && slot_generation(&b) == Some(2)
                && (0..MOVED).all(|e| {
                    matches!(
                        (a.mapping[e], b.mapping[e]),
                        (ResidencyLocation::Slot(x), ResidencyLocation::Slot(y))
                            if x.slot != y.slot && y.generation == x.generation + 1
                    )
                }),
            format!(
                "epoch {} -> {}, fill generation {:?} -> {:?}",
                a.epoch,
                b.epoch,
                slot_generation(&a),
                slot_generation(&b)
            ),
        )
        .map(|()| {
            format!(
                "epoch {} -> {}, experts 0..{MOVED} slots 0..{MOVED} -> {}..{}, fill generation 1 -> 2",
                a.epoch,
                b.epoch,
                MOVED,
                2 * MOVED
            )
        }),
    );
    rep.record(
        scope,
        "remap_table_address_extent_identity_unchanged",
        check(
            before == after,
            format!("identity moved: {before:?} -> {after:?}"),
        )
        .map(|()| {
            format!(
                "GU table {:#x}+{}, DN table {:#x}+{}, owner generation {}, stable fingerprint unchanged",
                after.gu_table.0,
                after.gu_table.1,
                after.dn_table.0,
                after.dn_table.1,
                after.owner_generation
            )
        }),
    );
    let b_tables = (
        hip(gpu.download_raw_bytes(&f.gu_table))?,
        hip(gpu.download_raw_bytes(&f.dn_table))?,
    );
    let changed: Vec<usize> = (0..E).filter(|&e| a.gate_up[e] != b.gate_up[e]).collect();
    let changed_dn: Vec<usize> = (0..E).filter(|&e| a.down[e] != b.down[e]).collect();
    let moved: Vec<usize> = (0..MOVED).collect();
    rep.record(
        scope,
        "remap_pointer_contents_changed",
        check(
            changed == moved
                && changed_dn == moved
                && a_tables.0 != b_tables.0
                && a_tables.1 != b_tables.1,
            format!(
                "changed gate_up {changed:?}, down {changed_dn:?}, device tables changed {}/{}",
                a_tables.0 != b_tables.0,
                a_tables.1 != b_tables.1
            ),
        )
        .map(|()| {
            format!(
                "entries {moved:?} changed in both tables; GU sha256 {} -> {}",
                &hash(&a_tables.0)[..12],
                &hash(&b_tables.0)[..12]
            )
        }),
    );
    rep.record(
        scope,
        "recorded_tape_unchanged_by_remap",
        rec.verify(gpu, &f).map(|()| {
            "same recorded launches, same prepared plan, same table allocations".to_owned()
        }),
    );
    let eager_b = eager(gpu, &arena, &f, &residency, idle)?;
    rep.record(
        scope,
        "remapped_eager_equals_original_eager",
        check(
            eager_b.differences(&eager_a) == [0; 3],
            "remapped eager output differs from the original placement",
        )
        .map(|()| "byte-copied experts compute identical bytes".to_owned()),
    );

    // Negative (a): after the remap no token holds a lease; the caller's lease
    // from the old epoch is retired and cannot launch the tape.
    let snapshot = residency.snapshot()?;
    let guard = residency.begin_token(snapshot.epoch)?;
    let stale = residency.authorize_lease(recorded_lease);
    let current = residency.authorize_lease(guard.lease());
    // SAFETY: no GPU work was submitted under this guard.
    unsafe { guard.retire_after_gpu_wait() }?;
    rep.record(
        scope,
        "stale_lease_not_authority_for_new_epoch",
        match (stale, current) {
            (Err(e), Ok(())) if e.contains("current image epoch") => Ok(format!(
                "lease epoch {} refused at epoch {}: {e}",
                recorded_lease.epoch, snapshot.epoch
            )),
            other => Err(format!("stale/current lease authorization {other:?}")),
        },
    );
    for &backend in phase.backends {
        let outcome = replay_refused(gpu, &arena, &f, &rec, backend, "no live generation lease");
        rep.record(
            scope,
            &format!("{}/refused_stale_lease_after_remap", backend.name()),
            outcome,
        );
    }

    // Replay B under a valid lease at the new epoch.
    for &backend in phase.backends {
        let outcome = replay_ok(gpu, &arena, &f, &rec, backend, &residency, idle);
        let (output, lease) = match outcome {
            Ok(done) => done,
            Err(error) if *idle => {
                rep.record(
                    scope,
                    &format!("{}/replay_b_byte_equal_eager", backend.name()),
                    Err(error),
                );
                continue;
            }
            Err(error) => return Err(error),
        };
        rep.record(
            scope,
            &format!("{}/replay_b_byte_equal_eager", backend.name()),
            check(
                output.differences(&eager_b) == [0; 3],
                format!(
                    "{} replay B differs from eager on the remapped placement",
                    backend.name()
                ),
            )
            .map(|()| {
                format!(
                    "lease generation {} epoch {}, retired slots scrubbed, down sha256 {}",
                    lease.owner_generation,
                    lease.epoch,
                    &hash(&output.down)[..16]
                )
            }),
        );
    }

    // Negative (b): the token is retired again; no live lease at the current epoch.
    for &backend in phase.backends {
        let outcome = replay_refused(gpu, &arena, &f, &rec, backend, "no live generation lease");
        rep.record(
            scope,
            &format!("{}/refused_no_live_lease", backend.name()),
            outcome,
        );
    }

    // Negative (c): a token dropped without proven retirement poisons the authority.
    let snapshot = residency.snapshot()?;
    drop(residency.begin_token(snapshot.epoch)?);
    rep.record(
        scope,
        "poisoned_authority_refuses_tokens",
        check(
            residency.snapshot().is_err() && residency.begin_token(snapshot.epoch).is_err(),
            "poisoned residency still serves snapshots or tokens",
        )
        .map(|()| "snapshot and begin_token refused".to_owned()),
    );
    for &backend in phase.backends {
        let outcome = replay_refused(gpu, &arena, &f, &rec, backend, "poisoned");
        rep.record(
            scope,
            &format!("{}/refused_poisoned_residency", backend.name()),
            outcome,
        );
    }
    Ok(())
}

fn finish(rep: &Report, json_path: Option<&str>, plan_only: bool) -> bool {
    let pass = !plan_only && rep.fail == 0 && rep.pass > 0;
    let status = if plan_only {
        "PLAN_ONLY"
    } else if pass {
        "PASS"
    } else {
        "FAIL"
    };
    println!(
        "SUMMARY {status}: {} PASS, {} FAIL, {} SKIP",
        rep.pass, rep.fail, rep.skip
    );
    if let Some(path) = json_path {
        let report = json!({
            "schema": 1,
            "scope": "s3b_replay_launch_authority_remap",
            "arch": "gfx1201",
            "no_model": true,
            "performance_claim": false,
            "sealed_moe_call_exercised": false,
            "authority_registration": "gpu.replay.note_launch_authority(&residency, residency.owner_generation()) after begin_capture, as SealedMoeCall::note_retained_route_identity does",
            "positions": 1,
            "backends": ["recorded_hip", "pm4", "aql"],
            "status": status,
            "pass": rep.pass, "fail": rep.fail, "skip": rep.skip,
            "checks": rep.checks,
        });
        match serde_json::to_string_pretty(&report)
            .map_err(|e| e.to_string())
            .and_then(|text| std::fs::write(path, text).map_err(|e| e.to_string()))
        {
            Ok(()) => println!("wrote {path}"),
            Err(error) => {
                println!("FAIL json {path}: {error}");
                return false;
            }
        }
    }
    plan_only || pass
}

fn main() {
    let mut execute = false;
    let mut json_path: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!(
                    "fn_expert_cache_replay_remap [--execute] [--json <path>]\n\
                     S3b plan gate R, exact gfx1201, no model; caller holds the shared GPU lock. \
                     Default PLAN_ONLY."
                );
                return;
            }
            "--execute" => execute = true,
            "--json" => match args.next() {
                Some(path) => json_path = Some(path),
                None => {
                    println!("FAIL args: --json needs a path");
                    std::process::exit(2);
                }
            },
            other => {
                println!("FAIL args: unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let mut rep = Report::default();
    if !execute {
        for phase in &PHASES {
            for backend in phase.backends {
                println!(
                    "PLAN {} {}: replay A, remap, replay B, refused stale/no-lease/poisoned",
                    phase.name,
                    backend.name()
                );
            }
        }
        let ok = finish(&rep, json_path.as_deref(), true);
        std::process::exit(if ok { 0 } else { 1 });
    }
    let mut gpu = match Gpu::init() {
        Ok(gpu) => gpu,
        Err(error) => {
            rep.record("init", "gpu_init", Err(error.to_string()));
            finish(&rep, json_path.as_deref(), false);
            std::process::exit(1);
        }
    };
    for phase in &PHASES {
        let owner = Arc::new(Mutex::new(Arena::default()));
        let mut idle = true;
        if let Err(error) = exercise(&mut gpu, &owner, phase, &mut idle, &mut rep) {
            rep.record(phase.name, "phase_aborted", Err(error));
        }
        let mut cleanup = Vec::new();
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
            match owner.lock() {
                Ok(mut arena) => {
                    if let Err(error) = arena.release(&mut gpu) {
                        cleanup.push(error);
                    }
                }
                Err(_) => cleanup.push("arena mutex poisoned; allocations retained".into()),
            }
        } else {
            cleanup.push("GPU quiescence unknown: owners retained until process exit".into());
        }
        let clean = cleanup.is_empty();
        rep.record(
            phase.name,
            "cleanup",
            if clean {
                Ok("shutdown proved quiescence; allocations released".to_owned())
            } else {
                Err(format!("{cleanup:?}"))
            },
        );
        if !idle {
            let ok = finish(&rep, json_path.as_deref(), false);
            std::mem::forget(owner);
            std::mem::forget(gpu);
            std::process::exit(if ok { 0 } else { 1 });
        }
    }
    let ok = finish(&rep, json_path.as_deref(), false);
    std::process::exit(if ok { 0 } else { 1 });
}
