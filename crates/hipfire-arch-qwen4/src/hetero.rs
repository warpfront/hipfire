//! `topology.hetero_experts`: the Qwen4 heterogeneous expert split.
//!
//! Experimental and opt-in, and not yet admitted by default (prefill still
//! moves the cold expert weights across the link; see the design in
//! `docs/superpowers/specs/2026-11-04-qwen4-hetero-expert-server-design.md`).
//!
//! With `topology.hetero_experts = "gfx1151"` and `hardware.devices =
//! "gfx1201,gfx1151"`, the gfx1201 primary runs the trunk and the hot routed
//! experts. Every D2 split executor then computes its cold slots on one shared
//! gfx1151 iGPU arm ([`hipfire_dispatch::igpu_split::IgpuColdArm`], HIP logical
//! device 1) instead of host CPU cores. The arm is process-wide: a daemon holds
//! one model.

use std::sync::{Arc, Mutex};

use hipfire_dispatch::igpu_split::{ColdArmDims, IgpuColdArm};

/// HIP logical device of the expert iGPU (`hardware.devices = "gfx1201,gfx1151"`).
const EXPERT_DEVICE: i32 = 1;

/// The primary (HIP logical device 0) must be the gfx1201.
pub(crate) fn check_primary(primary_arch: &str) -> Result<(), String> {
    if primary_arch == "gfx1201" {
        Ok(())
    } else {
        Err(format!(
            "topology.hetero_experts: the primary device is {primary_arch}; set hardware.devices = \"gfx1201,gfx1151\" (the gfx1201 first)"
        ))
    }
}

/// `topology.hetero_experts` for this process: `Some(arch)` when the split is
/// requested and its combination rules hold.
pub(crate) fn requested() -> Result<Option<String>, String> {
    use hipfire_config::topology;
    let value = hipfire_config::process_value(topology::HETERO_EXPERTS_ENV);
    topology::resolve_hetero_experts(
        value.as_deref(),
        hipfire_runtime::config::get().allow_mixed_arch,
        topology::qwen4_ep()?,
        topology::pp()?,
        1,
    )
}

static ARM: Mutex<Option<Arc<IgpuColdArm>>> = Mutex::new(None);

/// The shared iGPU arm when the split is requested, opened on first use; `None`
/// when it is not. `dims.max_rows` is raised to the D2 storage cap so every
/// executor of the load (wider verify reservations included) fits one arm.
pub(crate) fn arm_for(primary_arch: &str, dims: ColdArmDims) -> Result<Option<Arc<IgpuColdArm>>, String> {
    if requested()?.is_none() {
        return Ok(None);
    }
    check_primary(primary_arch)?;
    let mut slot = ARM.lock().map_err(|_| "topology.hetero_experts: arm lock poisoned".to_owned())?;
    if let Some(arm) = slot.as_ref() {
        return Ok(Some(Arc::clone(arm)));
    }
    let dims = ColdArmDims {
        max_rows: hipfire_dispatch::cpu_split::CPU_SPLIT_STORAGE_MAX_ROWS,
        ..dims
    };
    let arm = Arc::new(
        IgpuColdArm::new(EXPERT_DEVICE, dims)
            .map_err(|e| format!("topology.hetero_experts: {e:?}"))?,
    );
    eprintln!(
        "  qwen4 hetero experts: igpu arm on gfx1151 (HIP device {EXPERT_DEVICE}), cold D2 slots run there"
    );
    *slot = Some(Arc::clone(&arm));
    Ok(Some(arm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hetero_requires_primary_gfx1201_then_gfx1151() {
        assert!(check_primary("gfx1201").is_ok());
        let e = check_primary("gfx1151").unwrap_err();
        assert!(e.contains("topology.hetero_experts") && e.contains("gfx1201,gfx1151"), "{e}");
    }
}
