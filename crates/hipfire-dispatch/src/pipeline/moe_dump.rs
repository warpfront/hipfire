// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Opt-in developer dump of the decode MoE stage boundaries, used by the
//! single-vs-EP bitwise bisection (`tools/ep_bisect.py`).
//!
//! Off by default: with `HIPFIRE_MOE_DUMP_DIR` unset every hook is one
//! `OnceLock` load that returns `None` — no allocation, no device
//! synchronization, no change to the launch sequence. When set, each hook
//! synchronizes the device and copies the named tensor to the host, so a dump
//! run is a diagnostic run only (use `HIPFIRE_GRAPH=0`).
//!
//! Layout: `$DIR/<arm>/rank-<gpu.device_id>/position-<P>/layer-<L>/`, where
//! `<arm>`/`<P>` come from [`set_context`] and `<L>` is `MoeParams::layer_idx`.
//! All floating-point files are raw little-endian `f32`; `route-indices.i32`
//! holds the router's `i32` ids (the scratch buffer stores them as f32 bits).
//!
//! Files (a stage that did not run leaves its files absent):
//! * `shared-input.f32` / `shared-output.f32` — the residual the shared-expert
//!   down stage accumulates into, before / after it (NOT the shared vector).
//! * `routed-residual-input.f32` — the residual `x` immediately before the
//!   routed fold: `x` after the shared add on both arms.
//! * `fold-input.f32` / `fold-output.f32` — the fold *target* before / after.
//!   Single: the target is `x` itself, so fold-input == routed-residual-input
//!   and fold-output is the post-MoE residual. Root-routed EP: the target is
//!   the zeroed partial (`fold_base: zeroed_partial`), so fold-input is all
//!   zero and fold-output is the routed sum alone; `x` is only updated later
//!   by `residual_finish` (see `final-residual.f32`).
//! * `expanded-rows.f32` — the `slots x hidden` rows the combine kernel read.
//!   Absent (with `expanded-rows.ABSENT` carrying the reason) when the down
//!   kernel fused the fold (e.g. `ninepath_d4`), because no rows exist.
//! * `route-indices.i32`, `route-weights.f32` — `k` entries each.
//! * `final-residual.f32` — `x` after the whole MoE (after `residual_finish`
//!   under EP). `final-residual.DEFERRED` if the combine is deferred to the
//!   next layer.
//! * `meta-fold.json`, `meta-shared.json`, `meta-final.json` — which path ran.

use crate::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

static ROOT: LazyLock<Option<PathBuf>> = LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_MOE_DUMP_DIR")
        .ok()
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
});
static CONTEXT: Mutex<Option<(String, usize)>> = Mutex::new(None);

fn root() -> Option<&'static Path> {
    ROOT.as_deref()
}

/// True when `HIPFIRE_MOE_DUMP_DIR` is set.
pub fn enabled() -> bool {
    root().is_some()
}

/// Tag every following dump with `arm` (e.g. `single`, `ep2`) and the decode
/// position. Call before each decode step; has no effect unless
/// `HIPFIRE_MOE_DUMP_DIR` is set.
pub fn set_context(arm: &str, position: usize) {
    if enabled() {
        *CONTEXT.lock().expect("moe dump context poisoned") = Some((arm.to_owned(), position));
    }
}

fn err(message: String) -> DispatchError {
    DispatchError::Hip(format!("moe dump: {message}"))
}

struct Site {
    dir: PathBuf,
}

impl Site {
    fn open(gpu: &Gpu, layer: usize) -> Result<Option<Self>, DispatchError> {
        let Some(root) = root() else {
            return Ok(None);
        };
        let (arm, position) = CONTEXT
            .lock()
            .expect("moe dump context poisoned")
            .clone()
            .ok_or_else(|| {
                err("HIPFIRE_MOE_DUMP_DIR is set but set_context was never called".into())
            })?;
        let dir = root
            .join(arm)
            .join(format!("rank-{}", gpu.device_id))
            .join(format!("position-{position}"))
            .join(format!("layer-{layer}"));
        std::fs::create_dir_all(&dir).map_err(|e| err(format!("create {}: {e}", dir.display())))?;
        Ok(Some(Self { dir }))
    }

    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), DispatchError> {
        let path = self.dir.join(name);
        std::fs::write(&path, bytes).map_err(|e| err(format!("write {}: {e}", path.display())))
    }
}

/// Host copy of the first `elems` 4-byte values of `tensor` (device-synchronized).
fn read(gpu: &Gpu, tensor: &GpuTensor, elems: usize) -> Result<Vec<u8>, DispatchError> {
    let mut bytes = vec![0u8; elems * 4];
    gpu.bind_thread().map_err(|e| err(e.to_string()))?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| err(e.to_string()))?;
    gpu.hip
        .memcpy_dtoh(&mut bytes, &tensor.buf)
        .map_err(|e| err(e.to_string()))?;
    Ok(bytes)
}

fn same_buffer(a: &GpuTensor, b: &GpuTensor) -> bool {
    a.buf.as_ptr() as usize == b.buf.as_ptr() as usize
}

/// Pre-stage snapshot of the shared-expert accumulation target.
pub struct SharedPending {
    site: Site,
    before: Vec<u8>,
}

pub fn shared_begin(
    gpu: &Gpu,
    layer: usize,
    target: &GpuTensor,
    hidden: usize,
) -> Result<Option<SharedPending>, DispatchError> {
    let Some(site) = Site::open(gpu, layer)? else {
        return Ok(None);
    };
    let before = read(gpu, target, hidden)?;
    Ok(Some(SharedPending { site, before }))
}

pub fn shared_end(
    gpu: &Gpu,
    pending: Option<SharedPending>,
    target: &GpuTensor,
    hidden: usize,
    target_is_partial: bool,
) -> Result<(), DispatchError> {
    let Some(pending) = pending else {
        return Ok(());
    };
    let after = read(gpu, target, hidden)?;
    pending.site.write("shared-input.f32", &pending.before)?;
    pending.site.write("shared-output.f32", &after)?;
    pending.site.write(
        "meta-shared.json",
        format!(
            "{{\"hidden\":{hidden},\"target\":\"{}\"}}\n",
            if target_is_partial {
                "routed_out_partial"
            } else {
                "x_residual"
            }
        )
        .as_bytes(),
    )
}

/// Pre-fold snapshots.
pub struct FoldPending {
    site: Site,
    residual: Vec<u8>,
    target: Vec<u8>,
}

pub fn fold_begin(
    gpu: &Gpu,
    layer: usize,
    hidden: usize,
    residual: &GpuTensor,
    target: &GpuTensor,
) -> Result<Option<FoldPending>, DispatchError> {
    let Some(site) = Site::open(gpu, layer)? else {
        return Ok(None);
    };
    let residual_bytes = read(gpu, residual, hidden)?;
    let target_bytes = if same_buffer(residual, target) {
        residual_bytes.clone()
    } else {
        read(gpu, target, hidden)?
    };
    Ok(Some(FoldPending {
        site,
        residual: residual_bytes,
        target: target_bytes,
    }))
}

/// What the fold boundary was, for the meta file.
pub struct FoldInfo<'a> {
    /// Kernel family that folded, e.g. `expanded_k8_combine`, `ninepath_d4`.
    pub path: &'a str,
    pub ep_mode: &'a str,
    pub hidden: usize,
    pub slots: usize,
    /// `Some((tensor, row_stride_elems))` when the fold read an expanded buffer.
    pub expanded: Option<(&'a GpuTensor, usize)>,
    pub indices: &'a GpuTensor,
    pub weights: &'a GpuTensor,
    /// Reason the rows are absent when `expanded` is `None`.
    pub expanded_absent_reason: &'a str,
}

pub fn fold_end(
    gpu: &Gpu,
    pending: Option<FoldPending>,
    target: &GpuTensor,
    target_is_partial: bool,
    info: &FoldInfo<'_>,
) -> Result<(), DispatchError> {
    let Some(pending) = pending else {
        return Ok(());
    };
    let site = &pending.site;
    let after = read(gpu, target, info.hidden)?;
    site.write("routed-residual-input.f32", &pending.residual)?;
    site.write("fold-input.f32", &pending.target)?;
    site.write("fold-output.f32", &after)?;
    site.write("route-indices.i32", &read(gpu, info.indices, info.slots)?)?;
    site.write("route-weights.f32", &read(gpu, info.weights, info.slots)?)?;
    match info.expanded {
        Some((rows, stride)) => {
            site.write("expanded-rows.f32", &read(gpu, rows, info.slots * stride)?)?;
        }
        None => site.write(
            "expanded-rows.ABSENT",
            info.expanded_absent_reason.as_bytes(),
        )?,
    }
    site.write(
        "meta-fold.json",
        format!(
            "{{\"path\":\"{}\",\"ep_mode\":\"{}\",\"fold_base\":\"{}\",\"hidden\":{},\"slots\":{},\"expanded\":{}}}\n",
            info.path,
            info.ep_mode,
            if target_is_partial {
                "zeroed_partial"
            } else {
                "x_residual_after_shared"
            },
            info.hidden,
            info.slots,
            info.expanded.is_some(),
        )
        .as_bytes(),
    )
}

/// `x` after the whole MoE for this layer. `source` names the hook point.
pub fn final_residual(
    gpu: &Gpu,
    layer: usize,
    x: &GpuTensor,
    hidden: usize,
    source: &str,
) -> Result<(), DispatchError> {
    let Some(site) = Site::open(gpu, layer)? else {
        return Ok(());
    };
    site.write("final-residual.f32", &read(gpu, x, hidden)?)?;
    site.write(
        "meta-final.json",
        format!("{{\"source\":\"{source}\",\"hidden\":{hidden}}}\n").as_bytes(),
    )
}

/// The routed fold was deferred to the next layer: `x` is not final here.
pub fn final_residual_deferred(gpu: &Gpu, layer: usize) -> Result<(), DispatchError> {
    let Some(site) = Site::open(gpu, layer)? else {
        return Ok(());
    };
    site.write(
        "final-residual.DEFERRED",
        b"routed combine deferred to the next layer (defer_routed_combine)",
    )
}
