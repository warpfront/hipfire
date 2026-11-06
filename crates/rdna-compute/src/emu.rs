//! SKU emulation hooks on `Gpu`: capacity cap, Phase-2 resident budget, receipts (EMU-DESIGN.md §3, E6).
//!
//! Capacity-only emulation. `HIPFIRE_EMU=<id>` installs a process-local
//! managed-capacity ledger (`hipfire_config::emu_budget`) at `Gpu`
//! initialization, before any pool, model, scratch or rocBLAS allocation. The
//! HIP/ROCr allocation adapters then charge against it, and
//! [`Gpu::device_mem_info`] reports the capped `(free, total)` so every sizing
//! site agrees with the allocator.
//!
//! No CU masks, queue masks, pacing hooks or kernels live here; the profile
//! never rewrites `Gpu::arch`, `arch_caps` or tuning (those stay physical).
//! With `HIPFIRE_EMU` unset every method below is a no-op / pass-through and
//! nothing here allocates, locks or touches HIP beyond the original call.

use crate::dispatch::{Gpu, HIP_DEVICE_ATTRIBUTE_INTEGRATED};
use hip_bridge::{HipError, HipResult, HipRuntime, HIP_ERROR_OUT_OF_MEMORY};
use hipfire_config::emu::{self, EmuOptions, EmuProfile, HostFacts, MemoryKind};
use hipfire_config::emu_budget::{
    self, AllocationClass, BudgetError, BudgetId, BudgetLimits, BudgetSnapshot,
};

/// Env spelling of the logical multi-GPU emulation knob (unrelated to SKU
/// emulation; refused in combination because it needs a per-rank budget).
const ENV_EMULATE_GPUS: &str = "HIPFIRE_EMULATE_GPUS";

/// Per-`Gpu` emulation state; `None` on the `Gpu` when emulation is off.
pub(crate) struct EmuGpuState {
    pub(crate) profile: &'static EmuProfile,
    pub(crate) facts: HostFacts,
    pub(crate) budget: BudgetId,
}

// ── Pure decisions (CPU-testable) ───────────────────────────────────────────

/// Active logical `HIPFIRE_EMULATE_GPUS` (>= 2, same parse as the runtime
/// config) cannot be combined with a single-device capacity budget.
pub(crate) fn emulate_gpus_refusal(raw: Option<&str>) -> Result<(), String> {
    match raw
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&count| count >= 2)
    {
        None => Ok(()),
        Some(count) => Err(format!(
            "HIPFIRE_EMU capacity emulation is single-device only; {ENV_EMULATE_GPUS}={count} \
             (logical multi-GPU) needs a per-rank budget and is refused"
        )),
    }
}

/// Only one physical device may own the process budget. `Err` when the
/// BDFs differ (ASCII case-insensitive).
pub(crate) fn second_device_refusal(
    installed_bdf: &str,
    requested_bdf: &str,
) -> Result<(), String> {
    if installed_bdf.eq_ignore_ascii_case(requested_bdf) {
        Ok(())
    } else {
        Err(format!(
            "HIPFIRE_EMU capacity budget is installed for device {installed_bdf}; a second \
             device {requested_bdf} cannot borrow it (one physical device per process)"
        ))
    }
}

/// Refusal text for [`Gpu::device_mem_info_for`] on a device other than the
/// budget owner.
pub(crate) fn other_device_refusal(
    installed_ordinal: i32,
    installed_bdf: &str,
    requested_ordinal: i32,
    requested_bdf: Option<&str>,
) -> String {
    match requested_bdf.map(|bdf| second_device_refusal(installed_bdf, bdf)) {
        Some(Err(message)) => message,
        _ => format!(
            "HIPFIRE_EMU capacity emulation is single-device: memory query for device \
             {requested_ordinal} refused (budget owner is device {installed_ordinal}, \
             {installed_bdf})"
        ),
    }
}

/// Ledger limits for the requested profile. Discrete: `device = capacity`.
/// UMA: `device = G`, `gtt = G`, `shared = S = H - C - R`, `uma = true`.
pub(crate) fn budget_limits(
    profile: &EmuProfile,
    options: &EmuOptions,
) -> Result<BudgetLimits, String> {
    match (profile.memory, options.uma.as_ref()) {
        (MemoryKind::Discrete { capacity_bytes }, None) => Ok(BudgetLimits {
            device_bytes: capacity_bytes,
            gtt_bytes: None,
            shared_host_bytes: None,
            uma: false,
        }),
        (MemoryKind::Uma, Some(scenario)) => {
            scenario.validate().map_err(|error| error.to_string())?;
            let shared = scenario
                .shared_host_bytes()
                .map_err(|error| error.to_string())?;
            Ok(BudgetLimits {
                device_bytes: scenario.gtt_limit_bytes,
                gtt_bytes: Some(scenario.gtt_limit_bytes),
                shared_host_bytes: Some(shared),
                uma: true,
            })
        }
        (MemoryKind::Discrete { .. }, Some(_)) => Err(format!(
            "profile `{}` is discrete but a UMA scenario was supplied",
            profile.id
        )),
        (MemoryKind::Uma, None) => Err(format!(
            "profile `{}` is UMA but no UMA scenario was supplied",
            profile.id
        )),
    }
}

/// `min(requested, free.saturating_sub(reserved))` — the Phase-2 handoff math.
pub(crate) fn resident_budget_from(
    requested_bytes: u64,
    free_bytes: u64,
    reserved_bytes: u64,
) -> u64 {
    requested_bytes.min(free_bytes.saturating_sub(reserved_bytes))
}

/// Allocation class of an admitted managed-host allocation.
pub(crate) fn host_allocation_class(gtt_backed: bool) -> AllocationClass {
    if gtt_backed {
        AllocationClass::HostPinnedGtt
    } else {
        AllocationClass::HostPinnedUserptr
    }
}

/// Map a ledger refusal to a HIP error: a cap overrun is a normal
/// out-of-memory (code 2); anything else is a budget fault (code 0).
pub(crate) fn host_refusal_error(error: &BudgetError) -> HipError {
    let code = if error.is_out_of_memory() {
        HIP_ERROR_OUT_OF_MEMORY
    } else {
        0
    };
    HipError::new(code, &format!("emu host allocation refused: {error}"))
}

/// Map an install failure. A different-device install conflict is reported
/// as the second-device refusal.
fn install_error(error: &BudgetError) -> HipError {
    if let BudgetError::InstallConflict {
        installed_bdf,
        requested_bdf,
    } = error
    {
        if let Err(message) = second_device_refusal(installed_bdf, requested_bdf) {
            return HipError::new(0, &message);
        }
    }
    HipError::new(0, &format!("HIPFIRE_EMU budget install failed: {error}"))
}

// ── Gpu initialization hook ─────────────────────────────────────────────────

/// Install the emu budget for `device_id` when `HIPFIRE_EMU` selects a
/// profile. Call after `hip.set_device(id)` and the real-arch read, before any
/// pool/model/rocBLAS/scratch allocation. `Ok(None)` when emulation is off.
///
/// Every failure is a hard init error: there is no uncapped fallback.
pub(crate) fn install_for_device(
    hip: &HipRuntime,
    device_id: i32,
    detected_arch: &str,
) -> HipResult<Option<EmuGpuState>> {
    let config = hipfire_config::active_or_local_process_config();
    let options = match emu::from_process_config(config) {
        Ok(None) => return Ok(None),
        Ok(Some(options)) => options,
        Err(error) => return Err(HipError::new(0, &format!("HIPFIRE_EMU: {error}"))),
    };
    let profile = emu::profile(&options.profile_id)
        .map_err(|error| HipError::new(0, &format!("HIPFIRE_EMU: {error}")))?;

    emulate_gpus_refusal(config.legacy_value(ENV_EMULATE_GPUS).as_deref())
        .map_err(|message| HipError::new(0, &message))?;

    let pci_bdf = hip.device_pci_bus_id(device_id).map_err(|error| {
        HipError::new(
            0,
            &format!("HIPFIRE_EMU: device {device_id} PCI BDF unreadable: {error}"),
        )
    })?;
    let integrated = hip
        .get_device_attribute(HIP_DEVICE_ATTRIBUTE_INTEGRATED, device_id)
        .map(|value| value != 0)
        .map_err(|error| {
            HipError::new(
                0,
                &format!(
                    "HIPFIRE_EMU: device {device_id} integrated attribute unreadable: {error}"
                ),
            )
        })?;
    // HIP reports WGPs on RDNA; an unreadable/non-positive count stays 0 and
    // fails closed in `check_host`.
    let multiprocessors = hip
        .get_device_attribute(
            crate::profiler::HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            device_id,
        )
        .ok()
        .and_then(|count| u32::try_from(count).ok())
        .unwrap_or(0);
    let raw_device_total_bytes =
        hip.get_vram_info()
            .map(|(_, total)| total as u64)
            .map_err(|error| {
                HipError::new(
                    0,
                    &format!("HIPFIRE_EMU: raw device memory query failed: {error}"),
                )
            })?;
    let host_total_bytes = crate::uma_memory::host_total_bytes();
    let facts = HostFacts {
        pci_bdf: pci_bdf.clone(),
        detected_gfx: detected_arch.to_owned(),
        integrated,
        physical_cus: emu::wgp_to_physical_cus(multiprocessors, detected_arch),
        raw_device_total_bytes,
        host_total_bytes,
    };
    emu::check_host(&options, &facts)
        .map_err(|error| HipError::new(0, &format!("HIPFIRE_EMU host check failed: {error}")))?;

    let limits = budget_limits(profile, &options)
        .map_err(|message| HipError::new(0, &format!("HIPFIRE_EMU: {message}")))?;
    let budget = emu_budget::install(&pci_bdf, limits).map_err(|error| install_error(&error))?;
    eprintln!(
        "[emu] profile {} installed on device {device_id} ({pci_bdf}): capacity cap {} B \
         (capacity-only; execution stays on the physical {detected_arch} host)",
        profile.id, limits.device_bytes,
    );
    Ok(Some(EmuGpuState {
        profile,
        facts,
        budget,
    }))
}

// ── Gpu methods ─────────────────────────────────────────────────────────────

impl Gpu {
    /// The active emulation profile, `None` when `HIPFIRE_EMU` is off.
    pub fn emu_profile(&self) -> Option<&'static EmuProfile> {
        self.emu.as_ref().map(|state| state.profile)
    }

    /// Ledger snapshot (limits, committed, pending) for receipts; `None` when
    /// emulation is off.
    pub fn emu_budget_snapshot(&self) -> Option<BudgetSnapshot> {
        self.emu
            .as_ref()
            .map(|state| emu_budget::snapshot(state.budget))
    }

    /// Host facts captured at install (BDF, detected gfx, physical CUs, raw
    /// totals); `None` when emulation is off.
    pub fn emu_host_facts(&self) -> Option<&HostFacts> {
        self.emu.as_ref().map(|state| &state.facts)
    }

    /// Phase-2 handoff (EMU-DESIGN.md §3). Off: `requested_bytes` unchanged,
    /// no device query. On: `min(requested, device_mem_info().free - reserved)`.
    /// Explicit budgets never raise the profile cap.
    pub fn emu_resident_budget(&self, requested_bytes: u64, reserved_bytes: u64) -> HipResult<u64> {
        if self.emu.is_none() {
            return Ok(requested_bytes);
        }
        let (free, _) = self.device_mem_info()?;
        Ok(resident_budget_from(
            requested_bytes,
            free as u64,
            reserved_bytes,
        ))
    }

    /// Preflight a managed host allocation against the ledger (shared-host /
    /// GTT room). Call ALONGSIDE the real host checks (`MemAvailable - CMA`,
    /// TTM, GTT cap), never instead of them. Off: `Ok`. A refusal is a normal
    /// out-of-memory (`hipErrorOutOfMemory`). Zero bytes hold no capacity.
    pub fn emu_check_host_allocation(&self, bytes: u64, gtt_backed: bool) -> HipResult<()> {
        let Some(state) = self.emu.as_ref() else {
            return Ok(());
        };
        if bytes == 0 {
            return Ok(());
        }
        emu_budget::check_additional(state.budget, host_allocation_class(gtt_backed), bytes)
            .map_err(|error| host_refusal_error(&error))
    }

    /// `(free, total)` of `device_id`.
    ///
    /// The budget owner's own device goes through [`Gpu::device_mem_info`]
    /// (UMA-safe clamp, then the emu cap). Another device under emu is
    /// refused. Off-state, another device keeps the loader's historic raw
    /// behavior: switch with raw HIP setters, query (free clamped like
    /// `device_mem_info` from that device's own integrated attribute, BDF and
    /// carve-out; no emu cap), and restore the previous
    /// ACTUAL current device on every return path (`bind_thread`'s cache can
    /// disagree with it, so the cache is neither consulted nor updated).
    pub fn device_mem_info_for(&self, device_id: i32) -> HipResult<(usize, usize)> {
        if device_id == self.device_id {
            return self.device_mem_info();
        }
        if let Some(state) = self.emu.as_ref() {
            let requested_bdf = self.hip.device_pci_bus_id(device_id).ok();
            return Err(HipError::new(
                0,
                &other_device_refusal(
                    self.device_id,
                    &state.facts.pci_bdf,
                    device_id,
                    requested_bdf.as_deref(),
                ),
            ));
        }
        let previous = self.hip.current_device()?;
        let queried = self.hip.set_device(device_id).and_then(|_| {
            let (free, total) = self.hip.get_vram_info()?;
            // Same UMA-safe clamp as `Gpu::device_mem_info`, fed the queried
            // device's own attribute/BDF/carve-out. An unreadable attribute
            // counts as UMA (`Gpu::is_uma`'s conservative answer).
            let uma = self
                .hip
                .get_device_attribute(HIP_DEVICE_ATTRIBUTE_INTEGRATED, device_id)
                .map_or(true, |value| value != 0);
            if !uma {
                return Ok((free, total));
            }
            let carveout_total = self
                .hip
                .device_pci_bus_id(device_id)
                .ok()
                .and_then(|bdf| crate::uma_memory::carveout_total_bytes(&bdf));
            let memory = crate::uma_memory::DeviceMemory {
                device_free: free as u64,
                device_total: total as u64,
                uma: true,
                carveout_total,
                host_available: crate::uma_memory::gtt_backed(total as u64, carveout_total)
                    .then(crate::uma_memory::host_available_bytes)
                    .flatten(),
            };
            let free = crate::uma_memory::device_free_from(memory).ok_or_else(|| {
                HipError::new(
                    0,
                    "unified-memory free bytes: /proc/meminfo has no readable MemAvailable",
                )
            })?;
            Ok((free.min(usize::MAX as u64) as usize, total))
        });
        let restored = self.hip.set_device(previous);
        match (queried, restored) {
            (Ok(info), Ok(())) => Ok(info),
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
        }
    }

    /// [`Gpu::planned_capacity`] of `device_id`: the budget owner's own device
    /// uses its ledger-aware figure; another device holds none of this
    /// runtime's allocations before its rank loads (the own-bytes ledger is one
    /// counter for the runtime, not per device), so it plans against total
    /// minus the fixed reserve. UMA parts keep the free figure.
    pub fn planned_capacity_for(&self, device_id: i32) -> HipResult<crate::PlannedCapacity> {
        if device_id == self.device_id {
            return self.planned_capacity();
        }
        let (free, total) = self.device_mem_info_for(device_id)?;
        let stable = self
            .hip
            .get_device_attribute(HIP_DEVICE_ATTRIBUTE_INTEGRATED, device_id)
            .map_or(false, |value| value == 0);
        Ok(crate::PlannedCapacity::new(free, total, 0, stable))
    }

    /// Refuse a multi-device path (EP/TP/PP/heterogeneous) under emulation;
    /// `Ok` when emulation is off. `what` names the refused path.
    pub fn emu_refuse_multi_device(&self, what: &str) -> HipResult<()> {
        if self.emu.is_some() {
            return Err(HipError::new(
                0,
                &format!(
                    "HIPFIRE_EMU capacity emulation is single-device only: {what} is not \
                     supported under an emu profile (needs a per-rank budget); unset HIPFIRE_EMU"
                ),
            ));
        }
        Ok(())
    }

    /// Apply the emu cap to the UMA-safe `(free, total)`; identity when off.
    pub(crate) fn emu_cap_mem_info(&self, free: usize, total: usize) -> (usize, usize) {
        let Some(state) = self.emu.as_ref() else {
            return (free, total);
        };
        let snapshot = emu_budget::snapshot(state.budget);
        let (free, total) = crate::uma_memory::emu_capped_mem_info(free as u64, &snapshot);
        (
            free.min(usize::MAX as u64) as usize,
            total.min(usize::MAX as u64) as usize,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_config::emu::UmaScenario;

    const GIB: u64 = 1 << 30;

    #[test]
    fn emulate_gpus_two_or_more_is_refused() {
        assert!(emulate_gpus_refusal(None).is_ok());
        assert!(emulate_gpus_refusal(Some("")).is_ok());
        assert!(emulate_gpus_refusal(Some("malformed")).is_ok());
        assert!(emulate_gpus_refusal(Some("0")).is_ok());
        assert!(emulate_gpus_refusal(Some("1")).is_ok());
        let two = emulate_gpus_refusal(Some("2")).unwrap_err();
        assert!(two.contains("HIPFIRE_EMULATE_GPUS=2"), "{two}");
        assert!(emulate_gpus_refusal(Some("4")).is_err());
    }

    #[test]
    fn second_device_is_refused_unless_same_bdf() {
        assert!(second_device_refusal("0000:03:00.0", "0000:03:00.0").is_ok());
        assert!(second_device_refusal("0000:0A:00.0", "0000:0a:00.0").is_ok());
        let error = second_device_refusal("0000:03:00.0", "0000:44:00.0").unwrap_err();
        assert!(
            error.contains("0000:03:00.0") && error.contains("0000:44:00.0"),
            "{error}"
        );
    }

    #[test]
    fn other_device_query_message_names_both_devices() {
        let different = other_device_refusal(0, "0000:03:00.0", 1, Some("0000:44:00.0"));
        assert!(different.contains("second"), "{different}");
        let unknown_bdf = other_device_refusal(0, "0000:03:00.0", 3, None);
        assert!(
            unknown_bdf.contains("device 3") && unknown_bdf.contains("device 0"),
            "{unknown_bdf}"
        );
        let aliased = other_device_refusal(0, "0000:03:00.0", 1, Some("0000:03:00.0"));
        assert!(aliased.contains("single-device"), "{aliased}");
    }

    #[test]
    fn install_conflict_maps_to_second_device_refusal() {
        let conflict = BudgetError::InstallConflict {
            installed_bdf: "0000:03:00.0".into(),
            requested_bdf: "0000:44:00.0".into(),
        };
        let error = install_error(&conflict);
        assert!(error.message.contains("second"), "{}", error.message);
        let after = install_error(&BudgetError::InstallAfterAllocations);
        assert!(
            after.message.contains("install failed"),
            "{}",
            after.message
        );
    }

    #[test]
    fn discrete_limits_use_profile_capacity() {
        let profile = emu::profile("rx9070xt").unwrap();
        let options = EmuOptions {
            profile_id: profile.id.to_owned(),
            uma: None,
        };
        let limits = budget_limits(profile, &options).unwrap();
        let MemoryKind::Discrete { capacity_bytes } = profile.memory else {
            panic!("rx9070xt is discrete");
        };
        assert_eq!(
            limits,
            BudgetLimits {
                device_bytes: capacity_bytes,
                gtt_bytes: None,
                shared_host_bytes: None,
                uma: false,
            }
        );
    }

    #[test]
    fn uma_limits_are_g_and_s_with_uma_charging() {
        let profile = emu::profile("strix-point").unwrap();
        let options = EmuOptions {
            profile_id: profile.id.to_owned(),
            uma: Some(UmaScenario::default()),
        };
        let limits = budget_limits(profile, &options).unwrap();
        assert_eq!(
            limits,
            BudgetLimits {
                device_bytes: 16 * GIB,
                gtt_bytes: Some(16 * GIB),
                shared_host_bytes: Some(27 * GIB),
                uma: true,
            }
        );
    }

    #[test]
    fn limits_reject_scenario_kind_mismatch() {
        let discrete = emu::profile("rx9070xt").unwrap();
        let uma_profile = emu::profile("strix-point").unwrap();
        let with_scenario = EmuOptions {
            profile_id: discrete.id.to_owned(),
            uma: Some(UmaScenario::default()),
        };
        assert!(budget_limits(discrete, &with_scenario).is_err());
        let without = EmuOptions {
            profile_id: uma_profile.id.to_owned(),
            uma: None,
        };
        assert!(budget_limits(uma_profile, &without).is_err());
        let invalid = EmuOptions {
            profile_id: uma_profile.id.to_owned(),
            uma: Some(UmaScenario {
                gtt_limit_bytes: 28 * GIB,
                ..UmaScenario::default()
            }),
        };
        assert!(budget_limits(uma_profile, &invalid).is_err());
    }

    #[test]
    fn resident_budget_is_min_of_request_and_free_minus_reserved() {
        // Request below the room: the request wins.
        assert_eq!(resident_budget_from(4 * GIB, 10 * GIB, 2 * GIB), 4 * GIB);
        // Request above the room: free - reserved wins (explicit budgets never
        // raise the profile cap).
        assert_eq!(resident_budget_from(12 * GIB, 10 * GIB, 2 * GIB), 8 * GIB);
        // Reserved beyond free saturates to zero.
        assert_eq!(resident_budget_from(GIB, 3 * GIB, 5 * GIB), 0);
        assert_eq!(resident_budget_from(0, 3 * GIB, 0), 0);
        assert_eq!(resident_budget_from(u64::MAX, u64::MAX, 0), u64::MAX);
    }

    #[test]
    fn host_class_and_refusal_codes() {
        assert_eq!(host_allocation_class(true), AllocationClass::HostPinnedGtt);
        assert_eq!(
            host_allocation_class(false),
            AllocationClass::HostPinnedUserptr
        );
        let oom = host_refusal_error(&BudgetError::OutOfMemory {
            dimension: emu_budget::BudgetDimension::SharedHost,
            requested: 2,
            used: 3,
            limit: 4,
        });
        assert_eq!(oom.code, HIP_ERROR_OUT_OF_MEMORY);
        assert_eq!(host_refusal_error(&BudgetError::Overflow).code, 0);
    }
}
