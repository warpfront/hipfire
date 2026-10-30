//! No-model gate: the exact production route collector plus batch residency authority.
//! Pair publication uses tiny payloads; async transport/indexed-kernel proof remains
//! the separate existing `fn_expert_cache_publication --execute` gate.
use super::receipts::{check_routes, ensure, route_pattern, EXPERTS, LAYERS, MAX_ROWS, R, TOPK};
use hipfire_arch_qwen4::gpu_forward::{ExpertCacheRouteOracle, Qwen4GpuForwardError};
use hipfire_dispatch::families::moe::{
    MutableExpertResidency, MutableExpertResidencySpec, ResidencyLocation, ResidencyPublication,
    ResidencySlot, ResidencySlotTicket, ResidencySpan,
};
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::sync::{Arc, Mutex};

#[path = "collector_policy.rs"]
mod collector_policy;
use collector_policy::{merge_cleanup, require_missing_layer, CollectionFailure};

#[path = "../../../rdna-compute/examples/support/fn_expert_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::Arena;
fn hip<T, E: std::fmt::Display>(v: Result<T, E>) -> R<T> {
    v.map_err(|e| e.to_string())
}
fn words(v: &[usize]) -> Vec<u8> {
    v.iter().flat_map(|p| (*p as u64).to_le_bytes()).collect()
}
fn span(t: &GpuTensor) -> ResidencySpan {
    ResidencySpan {
        address: t.buf.as_ptr() as usize,
        bytes: t.buf.size(),
    }
}
fn refused(cap: &MutableExpertResidency, p: ResidencyPublication) -> R<()> {
    let called = Cell::new(false);
    // SAFETY: invalid metadata must reject before this callback. It never accesses GPU data.
    let r = unsafe {
        cap.publish_after_gpu_wait(p, |_| {
            called.set(true);
            Ok(())
        })
    };
    ensure(
        r.is_err() && !called.get(),
        "invalid/busy publication reached upload",
    )
}
fn publish(
    gpu: &Gpu,
    cap: &MutableExpertResidency,
    gu: &GpuTensor,
    dn: &GpuTensor,
    mapping: Vec<ResidencyLocation>,
    completed_slots: Vec<ResidencySlotTicket>,
) -> R<()> {
    let old = cap.snapshot()?;
    // SAFETY: caller retired every reader and waited for both slot payloads.
    // Both table uploads plus visibility wait finish before publication returns.
    unsafe {
        cap.publish_after_gpu_wait(
            ResidencyPublication {
                expected_epoch: old.epoch,
                mapping,
                completed_slots,
            },
            |next| {
                hip(gpu.hip.memcpy_htod(&gu.buf, &words(&next.gate_up)))?;
                hip(gpu.hip.memcpy_htod(&dn.buf, &words(&next.down)))?;
                hip(gpu.hip.device_synchronize())
            },
        )
    }?;
    let next = cap.snapshot()?;
    ensure(
        hip(gpu.download_raw_bytes(gu))? == words(&next.gate_up),
        "actual GU table bytes",
    )?;
    ensure(
        hip(gpu.download_raw_bytes(dn))? == words(&next.down),
        "actual DN table bytes",
    )
}
fn exercise(
    gpu: &mut Gpu,
    owner: &Arc<Mutex<Arena>>,
    out: &mut Value,
    retain_owners: &mut bool,
) -> R<()> {
    ensure(
        gpu.arch == "gfx1201",
        "no-model MTP/LRU oracle requires exact gfx1201",
    )?;
    let mut arena = owner.lock().map_err(|_| "fixture owner poisoned")?;
    let gu_bytes: Vec<u8> = (0..32).map(|x| (x * 3 + 11) as u8).collect();
    let dn_bytes: Vec<u8> = (0..32).map(|x| (x * 7 + 29) as u8).collect();
    let gu_host = arena.own(hip(gpu.upload_raw_host_mapped(&gu_bytes, &[32]))?);
    let dn_host = arena.own(hip(gpu.upload_raw_host_mapped(&dn_bytes, &[32]))?);
    let gu_slot = arena.raw(gpu, &[0; 16])?;
    let dn_slot = arena.raw(gpu, &[0; 16])?;
    let canonical = |root: usize| {
        (0..EXPERTS)
            .map(|e| ResidencySpan {
                address: arena.get(root).buf.as_ptr() as usize + (e % 2) * 16,
                bytes: 16,
            })
            .collect::<Vec<_>>()
    };
    let cg = canonical(gu_host);
    let cd = canonical(dn_host);
    let gt = arena.raw(
        gpu,
        &words(&cg.iter().map(|s| s.address).collect::<Vec<_>>()),
    )?;
    let dt = arena.raw(
        gpu,
        &words(&cd.iter().map(|s| s.address).collect::<Vec<_>>()),
    )?;
    hip(gpu.hip.device_synchronize())?;
    let gu = arena.get(gt).sub_offset(0, EXPERTS * 8);
    let dn = arena.get(dt).sub_offset(0, EXPERTS * 8);
    // SAFETY: canonical mapped roots, fixed tables and distinct slots are live in
    // retained Arena. Initial canonical GU/DN tables were uploaded and waited.
    let cap = unsafe {
        MutableExpertResidency::new(MutableExpertResidencySpec {
            owner_generation: 1,
            canonical_gate_up: cg,
            canonical_down: cd,
            gate_up_table: span(&gu),
            down_table: span(&dn),
            slots: vec![ResidencySlot {
                gate_up: span(&arena.get(gu_slot).sub_offset(0, 16)),
                down: span(&arena.get(dn_slot).sub_offset(0, 16)),
            }],
            keepalive: owner.clone(),
        })
    }?;
    let mut collector =
        ExpertCacheRouteOracle::new(gpu, LAYERS, TOPK, MAX_ROWS).map_err(|e| e.to_string())?;
    let result = (|| -> R<()> {
        let mut cells = Vec::new();
        for rows in 1..=MAX_ROWS {
            let before = cap.snapshot()?;
            let guard = if rows == 1 {
                cap.begin_token(before.epoch)?
            } else {
                cap.begin_verify(before.epoch, rows)?
            };
            ensure(guard.rows() == rows, "guard did not retain batch width")?;
            cap.validate_active_rows(rows)?;
            ensure(
                cap.validate_active_rows(if rows == 1 { 2 } else { 1 })
                    .is_err(),
                "wrong batch-width authority accepted",
            )?;
            collector.reset(gpu, rows).map_err(|e| e.to_string())?;
            for layer in 0..LAYERS {
                let pattern = route_pattern(layer, rows);
                let bytes: Vec<_> = pattern.iter().flat_map(|id| id.to_le_bytes()).collect();
                let root = arena.raw(gpu, &bytes)?;
                let source = arena.get(root).sub_offset(0, bytes.len());
                collector
                    .capture_layer(gpu, layer, &source)
                    .map_err(|e| e.to_string())?;
            }
            refused(
                &cap,
                ResidencyPublication {
                    expected_epoch: before.epoch,
                    mapping: before.mapping.to_vec(),
                    completed_slots: vec![],
                },
            )?;
            let actual = collector.collect(gpu).map_err(|e| e.to_string())?;
            check_routes(rows, &actual)?;
            let held = cap.snapshot()?;
            ensure(
                held.epoch == before.epoch
                    && held.gate_up == before.gate_up
                    && held.down == before.down,
                "tables changed while verify guard held",
            )?;
            cap.validate_active_rows(rows)?;
            hip(gpu.hip.device_synchronize())?;
            // SAFETY: only HIP collector work was submitted and its fence completed.
            unsafe { guard.retire_after_gpu_wait() }?;
            let raw: Vec<u8> = actual.iter().flat_map(|x| x.to_le_bytes()).collect();
            cells.push(json!({"rows":rows,"layers":LAYERS,"topk":TOPK,"actual_ids":actual,"actual_ids_sha256":format!("{:x}",Sha256::digest(&raw)),"guard_rows":rows,"epoch_before":before.epoch,"epoch_after_readback":held.epoch,"publication_refused_while_guarded":true,"wrong_width_refused":true,"guard_active_through_readback":true}));
        }
        for rows in [0, 1, MAX_ROWS + 1, usize::MAX] {
            let s = cap.snapshot()?;
            ensure(
                cap.begin_verify(s.epoch, rows).is_err(),
                format!("invalid verify rows {rows} admitted"),
            )?;
        }
        // Deliberately omit a layer. A success with old/zero/unwritten routes
        // is forbidden, even if all the supplied layers were copied correctly.
        collector.reset(gpu, 4).map_err(|e| e.to_string())?;
        for layer in (0..LAYERS).filter(|l| *l != 17) {
            let bytes: Vec<_> = route_pattern(layer, 4)
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect();
            let root = arena.raw(gpu, &bytes)?;
            collector
                .capture_layer(gpu, layer, &arena.get(root).sub_offset(0, bytes.len()))
                .map_err(|e| e.to_string())?;
        }
        // Accept only the collector's exact typed invalid-ID diagnostic.
        // A failed transfer/wait is a tool failure, never successful coverage.
        let diagnostic = match collector.collect(gpu) {
            Err(Qwen4GpuForwardError::Invalid(message)) => {
                require_missing_layer(CollectionFailure::InvalidInput(&message))
            }
            Err(Qwen4GpuForwardError::Hip(error)) => {
                require_missing_layer(CollectionFailure::Hip(&error.to_string()))
            }
            Err(error) => require_missing_layer(CollectionFailure::Other(&error.to_string())),
            Ok(_) => require_missing_layer(CollectionFailure::UnexpectedSuccess),
        }?;
        out["missing_layer_negative"] = json!({"omitted_layer":17,"rows":4,"detected":true,"error_kind":"Invalid","diagnostic":diagnostic});
        let mut copies = Vec::new();
        for (expert, generation) in [(0, 1), (1, 2)] {
            publish(
                gpu,
                &cap,
                &gu,
                &dn,
                vec![ResidencyLocation::Canonical; EXPERTS],
                vec![],
            )?;
            let before = cap.snapshot()?;
            ensure(
                before
                    .mapping
                    .iter()
                    .all(|m| *m == ResidencyLocation::Canonical),
                "victim was not canonical before overwrite",
            )?;
            let gs = arena.get(gu_slot).sub_offset(0, 16);
            let ds = arena.get(dn_slot).sub_offset(0, 16);
            hip(gpu
                .hip
                .memcpy_htod(&gs.buf, &gu_bytes[expert * 16..expert * 16 + 16]))?;
            hip(gpu.hip.device_synchronize())?;
            let ticket = ResidencySlotTicket {
                slot: 0,
                expert,
                generation,
            };
            let mut mapping = vec![ResidencyLocation::Canonical; EXPERTS];
            mapping[expert] = ResidencyLocation::Slot(ticket);
            refused(
                &cap,
                ResidencyPublication {
                    expected_epoch: before.epoch,
                    mapping: mapping.clone(),
                    completed_slots: vec![],
                },
            )?;
            hip(gpu
                .hip
                .memcpy_htod(&ds.buf, &dn_bytes[expert * 16..expert * 16 + 16]))?;
            hip(gpu.hip.device_synchronize())?;
            ensure(
                hip(gpu.download_raw_bytes(&gs))? == gu_bytes[expert * 16..expert * 16 + 16],
                "GU copy payload",
            )?;
            ensure(
                hip(gpu.download_raw_bytes(&ds))? == dn_bytes[expert * 16..expert * 16 + 16],
                "DN copy payload",
            )?;
            publish(gpu, &cap, &gu, &dn, mapping, vec![ticket])?;
            let next = cap.snapshot()?;
            let guard = cap.begin_verify(next.epoch, MAX_ROWS)?;
            cap.validate_current_entries(&next.gate_up, &next.down)?;
            cap.validate_active_rows(MAX_ROWS)?;
            refused(
                &cap,
                ResidencyPublication {
                    expected_epoch: next.epoch,
                    mapping: vec![ResidencyLocation::Canonical; EXPERTS],
                    completed_slots: vec![],
                },
            )?;
            // SAFETY: no work is submitted under this inspection-only guard.
            unsafe { guard.retire_after_gpu_wait() }?;
            copies.push(json!({"expert":expert,"slot":0,"generation":generation,"epoch":next.epoch,"gate_up":next.gate_up[expert],"down":next.down[expert],"gu_complete_dn_not_enqueued_refused":true,"full_payload_readback_exact":true,"table_pair_readback_exact":true,"batch8_publication_refused":true}));
        }
        out["collector_cells"] = json!(cells);
        out["publications"] = json!(copies);
        out["invalid_verify_rows_refused"] = json!([0, 1, MAX_ROWS + 1]);
        Ok(())
    })();
    // This tool submits HIP work only. This fence establishes HIP quiescence;
    // it is not a proof for an independent PM4 queue (none is used here).
    if let Err(error) = gpu.hip.device_synchronize() {
        *retain_owners = true;
        std::mem::forget(collector);
        return merge_cleanup(
            result,
            vec![format!(
                "HIP cleanup fence failed; collector and fixture owners retained: {error}"
            )],
        );
    }
    if result.is_err() {
        // A readback failure can leave the landing owner's transfer state
        // unresolved even after a device fence. Do not drop or free it here.
        *retain_owners = true;
        std::mem::forget(collector);
        return result;
    }
    let cleanup = match collector.free_gpu(gpu) {
        Ok(()) => Vec::new(),
        Err(error) => {
            // The consuming API owns any bounded handle leak; never reconstruct
            // or retry a handle whose destruction may already have happened.
            *retain_owners = true;
            vec![format!(
                "collector free failed; fixture owners retained: {error}"
            )]
        }
    };
    merge_cleanup(result, cleanup)
}

pub(super) fn run(gpu: &mut Gpu) -> R<Value> {
    let owner = Arc::new(Mutex::new(Arena::default()));
    let mut out = json!({"no_model":true,"scope":"production-route-collector-and-verify-batch-authority","production_transport_manager_exercised":false,"async_payload_transport_exercised":false,"model_or_native_head_exercised":false,"collector_cells":[],"publications":[]});
    let mut retain_owners = false;
    let primary = exercise(gpu, &owner, &mut out, &mut retain_owners);
    if retain_owners {
        // Sticky failure disposition: a later wait must not turn an unknown
        // transfer lifetime into permission to free its fixture owners.
        std::mem::forget(owner);
        return primary.and(Err(
            "collector owners retained after incomplete cleanup".into()
        ));
    }
    if let Err(error) = gpu.hip.device_synchronize() {
        std::mem::forget(owner);
        return merge_cleanup(
            primary,
            vec![format!(
                "outer HIP cleanup fence failed; fixture owners retained: {error}"
            )],
        )
        .map(|()| out);
    }
    let cleanup = {
        match owner.lock() {
            Ok(mut arena) => arena
                .release(gpu)
                .err()
                .map(|error| format!("fixture release: {error}")),
            Err(error) => Some(format!("fixture cleanup mutex poisoned: {error}")),
        }
    };
    if let Some(error) = cleanup {
        std::mem::forget(owner);
        return merge_cleanup(primary, vec![error]).map(|()| out);
    }
    primary?;
    Ok(out)
}
