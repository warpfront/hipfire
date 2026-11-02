// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! SKU-emulation reporting and model-free validation.

use hipfire_generate::batch::emit_uncorrelated_error;
use std::io::Write;

/// JSON object for a ledger snapshot: limits, committed and pending class
/// bytes. All values are the process-local managed application charge
/// (`accounting_kind = "managed_application_charge"`), never whole-driver
/// physical residency.
fn emu_ledger_json(snapshot: &hipfire_config::emu_budget::BudgetSnapshot) -> serde_json::Value {
    let usage = |u: &hipfire_config::emu_budget::BudgetUsage| {
        serde_json::json!({
            "device_bytes": u.device_bytes,
            "host_bytes": u.host_bytes,
            "gtt_bytes": u.gtt_bytes,
            "shared_host_bytes": u.shared_host_bytes,
        })
    };
    serde_json::json!({
        "accounting_kind": "managed_application_charge",
        "limits": {
            "device_bytes": snapshot.limits.device_bytes,
            "gtt_bytes": snapshot.limits.gtt_bytes,
            "shared_host_bytes": snapshot.limits.shared_host_bytes,
            "uma": snapshot.limits.uma,
        },
        "committed": usage(&snapshot.committed),
        "pending": usage(&snapshot.pending),
    })
}

/// Emulated (budgeted) memory fields for diag/error reports, present only
/// while an emulation profile is installed (`None` leaves every no-profile
/// report byte-identical). The existing `vram_free_mb`/`vram_total_mb` fields
/// next to it stay the raw hardware query; `physical_fields` names them so
/// the two are never confused.
fn emu_memory_json(gpu: &rdna_compute::Gpu) -> Option<serde_json::Value> {
    let profile = gpu.emu_profile()?;
    let mb = |bytes: usize| bytes / (1024 * 1024);
    let (budgeted_free_mb, budgeted_total_mb) = match gpu.device_mem_info() {
        Ok((free, total)) => (Some(mb(free)), Some(mb(total))),
        Err(_) => (None, None),
    };
    Some(serde_json::json!({
        "profile_id": profile.id,
        "physical_fields": ["vram_free_mb", "vram_total_mb"],
        "budgeted_free_mb": budgeted_free_mb,
        "budgeted_total_mb": budgeted_total_mb,
        "pool_parked_mb": mb(gpu.pool_parked_bytes()),
        "ledger": gpu.emu_budget_snapshot().map(|s| emu_ledger_json(&s)),
    }))
}

/// `"<arch> (<free> MB free / <total> MB total)"` for load-failure messages.
/// With no emulation profile this is exactly the historical text; under a
/// profile the raw figures are labelled `physical` and the budgeted figures
/// follow.
pub(crate) fn gpu_memory_text(gpu: &rdna_compute::Gpu, free_mb: usize, total_mb: usize) -> String {
    let Some(emu) = emu_memory_json(gpu) else {
        return format!("{} ({free_mb} MB free / {total_mb} MB total)", gpu.arch);
    };
    let field = |key: &str| {
        emu.get(key)
            .and_then(serde_json::Value::as_u64)
            .map_or_else(|| "unknown".to_owned(), |v| v.to_string())
    };
    let committed_mb = emu
        .pointer("/ledger/committed/device_bytes")
        .and_then(serde_json::Value::as_u64)
        .map_or_else(|| "unknown".to_owned(), |b| (b / (1024 * 1024)).to_string());
    format!(
        "{} (physical: {free_mb} MB free / {total_mb} MB total; emu profile `{}` budgeted: {} MB free / {} MB total, managed device charge committed {committed_mb} MB)",
        gpu.arch,
        emu.get("profile_id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?"),
        field("budgeted_free_mb"),
        field("budgeted_total_mb"),
    )
}

pub(crate) fn append_diag_emu(gpu: &rdna_compute::Gpu, diag_line: &mut String) {
    // Under an emulation profile the line gains an `emu` object with the
    // budgeted free/total and the ledger; `vram_free_mb`/`vram_total_mb`
    // stay the raw physical query. No profile: line unchanged.
    if let Some(emu) = emu_memory_json(gpu) {
        diag_line.pop();
        diag_line.push_str(",\"emu\":");
        diag_line.push_str(&emu.to_string());
        diag_line.push('}');
    }
}

/// The `emu_validate` verb: model-free SKU-emulation validation (EMU-DESIGN.md §6).
/// `models` is (loaded model, slot backend); either being `Some` refuses, as do a
/// missing installed profile and a request naming a different profile than this
/// process installed at startup.
pub(crate) fn validate<M, S>(
    gpu: &mut rdna_compute::Gpu,
    msg: &serde_json::Value,
    mut stdout: &mut std::io::Stdout,
    models: (&Option<M>, &Option<S>),
) {
    let request_id = msg.get("id").and_then(|v| v.as_str());
    let outcome: Result<serde_json::Value, (String, &'static str)> = if models.0.is_some()
        || models.1.is_some()
    {
        Err((
            "emu_validate refused: a model is loaded; it is a model-free validation \
             (run it on a fresh daemon)"
                .to_owned(),
            "validation",
        ))
    } else {
        match gpu.emu_profile() {
            None => Err((
                "emu_validate refused: no emulation profile is installed in this \
                 daemon (start it with HIPFIRE_EMU / developer.emu)"
                    .to_owned(),
                "validation",
            )),
            Some(installed) => {
                let requested = msg.get("profile_id").and_then(|v| v.as_str());
                if requested != Some(installed.id) {
                    Err((
                        format!(
                            "emu_validate refused: request profile_id {} differs from \
                             the profile `{}` installed in this daemon",
                            requested
                                .map_or_else(|| "(missing)".to_owned(), |id| format!("`{id}`")),
                            installed.id
                        ),
                        "validation",
                    ))
                } else {
                    match serde_json::from_value::<rdna_compute::emu_validate::EmuValidationRequest>(
                        msg.clone(),
                    ) {
                        Err(e) => Err((format!("invalid emu_validate request: {e}"), "validation")),
                        Ok(request) => match rdna_compute::emu_validate::validate(gpu, &request) {
                            Ok(receipt) => serde_json::to_value(&receipt).map_err(|e| {
                                (
                                    format!("emu_validate receipt serialization failed: {e}"),
                                    "internal",
                                )
                            }),
                            Err(e) => Err((format!("emu_validate failed: {e}"), "gpu")),
                        },
                    }
                }
            }
        }
    };
    match outcome {
        Ok(receipt) => {
            let _ = writeln!(
                stdout,
                "{}",
                serde_json::json!({ "type": "emu_validate", "receipt": receipt })
            );
        }
        Err((message, kind)) => {
            emit_uncorrelated_error(&mut stdout, request_id, &message, kind, false, false);
        }
    }
    let _ = stdout.flush();
}
