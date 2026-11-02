//! Weight-host tenant wiring (`HIPFIRE_WEIGHT_HOST=1`): claim kind, v1 load
//! envelope and lease reconciliation. Kept out of `main.rs` so the daemon
//! stays arch-neutral.

use crate::gpu_lock::{self, GpuLocks};
use hipfire_loader::LoadedModel;

/// Opens the GPU claim: the default exclusive claim when the weight-host opt-in
/// is unset or `0`, else the fabric-admitted shared claim. Returns the socket.
pub(crate) fn open_locks() -> Result<(Option<std::path::PathBuf>, GpuLocks), String> {
    let socket = socket_from_env()?;
    let locks = match &socket {
        None => GpuLocks::open()?,
        Some(socket) => open_claim(socket)?,
    };
    Ok((socket, locks))
}

#[cfg(target_os = "linux")]
fn socket_from_env() -> Result<Option<std::path::PathBuf>, String> {
    hipfire_runtime::weight_host::weight_host_socket_from_env().map_err(|e| e.to_string())
}

#[cfg(not(target_os = "linux"))]
fn socket_from_env() -> Result<Option<std::path::PathBuf>, String> {
    match std::env::var("HIPFIRE_WEIGHT_HOST").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => Ok(None),
        Ok("1") => Err("HIPFIRE_WEIGHT_HOST=1 is supported on Linux only".into()),
        _ => Err("HIPFIRE_WEIGHT_HOST must be unset, 0 or 1".into()),
    }
}

/// Shared-correctness claim for a weight-host tenant: admitted by the dev
/// fabric (`HIPFIRE_FABRIC_ADMISSION`, bootstrap env), never a private-load
/// fallback.
#[cfg(target_os = "linux")]
fn open_claim(socket: &std::path::Path) -> Result<GpuLocks, String> {
    let token = std::env::var(gpu_lock::fabric::ADMISSION_ENV).map_err(|_| {
        format!(
            "HIPFIRE_WEIGHT_HOST=1 (socket {}) needs a dev-fabric correct-class admission: {} is unset",
            socket.display(),
            gpu_lock::fabric::ADMISSION_ENV
        )
    })?;
    eprintln!("[weight-host] tenant mode: socket={}", socket.display());
    GpuLocks::open_shared(&token)
}

#[cfg(not(target_os = "linux"))]
fn open_claim(_socket: &std::path::Path) -> Result<GpuLocks, String> {
    Err("HIPFIRE_WEIGHT_HOST=1 is supported on Linux only".into())
}

/// Weight-host v1 envelope (dense Qwen3.5 arch 5, single card, no drafter,
/// MTP, vision, PFlash, head overlay or slot engine). Requests outside it are
/// refused before any teardown: nothing may load privately once opted in.
pub(crate) fn load_refusal(opted_in: bool, msg: &serde_json::Value) -> Option<String> {
    if !opted_in {
        return None;
    }
    let params = msg.get("params");
    let number = |key: &str| {
        params
            .and_then(|p| p.get(key))
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
    };
    let text = |key: &str| {
        params
            .and_then(|p| p.get(key))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
    };
    let refuse = |what: &str| {
        Some(format!(
            "load refused: HIPFIRE_WEIGHT_HOST=1 does not support {what}"
        ))
    };
    if number("tp") != 1 || number("pp") != 1 {
        return refuse("tp/pp > 1");
    }
    if params
        .and_then(|p| p.get("experimental_multi_slot"))
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        return refuse("experimental_multi_slot");
    }
    if crate::vision_ladder::resolve_vision_ladder(msg).1.is_some() {
        return refuse("a vision tower");
    }
    for key in ["draft", "drafter", "prefill_drafter", "head", "vision"] {
        if text(key).is_some() {
            return refuse(&format!("params.{key}"));
        }
    }
    // Resolved defaults may be `auto`; speculation must be explicitly off.
    for key in ["mtp_mode", "dflash_mode", "dspark_mode"] {
        if text(key) != Some("off") {
            return Some(format!(
                "load refused: HIPFIRE_WEIGHT_HOST=1 requires params.{key}=\"off\""
            ));
        }
    }
    None
}

/// A weight-host tenant whose host closed the lease unloads the imported
/// model before the next command runs (the current request already finished:
/// commands are sequential; the caller only asks while no batch lane is live,
/// and keeps this command's admitted terminal so it can still answer).
/// Returns true when it unloaded; the caller then drops its model-bound state.
pub(crate) fn reconcile_lease(
    model: &mut Option<LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
) -> bool {
    let Some(status) = model
        .as_mut()
        .and_then(|m| m.qwen35_mut())
        .and_then(|b| b.weight_lease.as_ref())
        .map(|lease| lease.host_alive())
    else {
        return false;
    };
    let reason = match status {
        Ok(true) => return false,
        Ok(false) => "weight host closed the lease".to_string(),
        Err(error) => format!("weight host lease check failed: {error}"),
    };
    let unload = model.take().map(|m| hipfire_loader::unload_model(m, gpu));
    let outcome = match unload {
        Some(Err(error)) => format!("imported model unload incomplete: {error}"),
        _ => "imported model unloaded".to_string(),
    };
    eprintln!("[weight-host] {reason}; {outcome}; reload against a live host");
    true
}
